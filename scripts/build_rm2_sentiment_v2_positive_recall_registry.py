from __future__ import annotations

import hashlib
import html
import json
import re
import unicodedata
from collections import defaultdict
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


ROOT = Path(__file__).resolve().parents[1]
HUMAN_V1_DIR = ROOT / "output/rm2_sentiment/validation/human_v1"
HUMAN_V2_DIR = ROOT / "output/rm2_sentiment/validation/human_v2"
HUMAN_V3_DIR = ROOT / "output/rm2_sentiment/validation/human_v3"
OUT_DIR = ROOT / "output/rm2_sentiment/validation/human_v2_positive_recall"

DATASET = ROOT / "dataset.csv"
OBS_V2_INFERENCE = ROOT / "output/rm2_sentiment/final/comment_sentiment_v2_observational.csv"
V2_V3_SAME_TEST = ROOT / "output/rm2_sentiment/experiments/indobert_v3/final_test_evaluation/v2_v3_same_test_predictions.csv"

V1_VALIDATED = HUMAN_V1_DIR / "sentiment_human_annotation_validated.csv"
V2_VALIDATED = HUMAN_V2_DIR / "sentiment_human_annotation_v2_validated.csv"
V2_ADJUDICATION_FINAL = HUMAN_V2_DIR / "sentiment_v2_adjudication_template_final.csv"
V2_REPLACEMENT_ADJUDICATION = HUMAN_V2_DIR / "sentiment_v2_replacement_adjudication_final.csv"
V2_LOCKED_FINAL = HUMAN_V2_DIR / "locked_test_v2_observational_final.csv"
V3_REGISTRY = HUMAN_V3_DIR / "human_label_registry_v3.csv"

OPTIONAL_POSITIVE_ADJUDICATION = OUT_DIR / "positive_active_learning_adjudication.csv"

OUT_REGISTRY = OUT_DIR / "human_label_registry.csv"
OUT_DUPLICATE_AUDIT = OUT_DIR / "duplicate_cluster_audit.csv"
OUT_CONFLICT_AUDIT = OUT_DIR / "label_conflict_audit.csv"
OUT_SPLIT_MANIFEST = OUT_DIR / "split_manifest.csv"
OUT_SOURCE_SUMMARY = OUT_DIR / "annotation_source_summary.csv"
OUT_FOLD_CANDIDATES = OUT_DIR / "fold_balance_candidates.csv"
OUT_SELECTED_FOLD = OUT_DIR / "selected_fold_manifest.csv"
OUT_FOLD_DIST = OUT_DIR / "fold_class_distribution.csv"
OUT_FOLD_LEAKAGE = OUT_DIR / "fold_leakage_audit.csv"
OUT_AL_BLIND = OUT_DIR / "positive_active_learning_blind.csv"
OUT_AL_A1 = OUT_DIR / "positive_active_learning_annotator_1.csv"
OUT_AL_A2 = OUT_DIR / "positive_active_learning_annotator_2.csv"
OUT_AL_ADJ = OUT_DIR / "positive_active_learning_adjudication.csv"
OUT_AL_MANIFEST = OUT_DIR / "positive_active_learning_sampling_manifest.csv"
OUT_CODEBOOK = OUT_DIR / "positive_annotation_codebook.md"
OUT_LOCKED_SAMPLE_MANIFEST = OUT_DIR / "new_locked_test_sampling_manifest.csv"
OUT_LOCKED_A1 = OUT_DIR / "new_locked_test_annotator_1_blind.csv"
OUT_LOCKED_A2 = OUT_DIR / "new_locked_test_annotator_2_blind.csv"
OUT_LOCKED_ADJ = OUT_DIR / "new_locked_test_adjudication.csv"
OUT_LOCKED_FINAL = OUT_DIR / "new_locked_test_final.csv"
OUT_LOCKED_FREEZE_MANIFEST = OUT_DIR / "new_locked_test_freeze_manifest.json"
OUT_LOCKED_SHA = OUT_DIR / "new_locked_test_sha256.txt"
OUT_MANIFEST_JSON = OUT_DIR / "human_label_registry_manifest.json"

LABELS = ["Negative", "Neutral", "Positive"]
NON_EVALUABLE = {"", "No Text", "Uncertain", "INJ"}
FOLD_SEEDS = [42, 52, 62, 72, 82]
N_SPLITS = 5
ACTIVE_LEARNING_N = 500
NEW_LOCKED_N = 600


class UnionFind:
    def __init__(self, values: list[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left == root_right:
            return
        self.parent[max(root_left, root_right)] = min(root_left, root_right)


def read_csv(path: Path, required: bool = True) -> pd.DataFrame:
    if not path.exists():
        if required:
            raise FileNotFoundError(path)
        return pd.DataFrame()
    return pd.read_csv(path, dtype=str, keep_default_na=False, low_memory=False)


def norm_blank(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none", "null", "<na>"}:
        return ""
    return text


def normalize_username(value: object) -> str:
    return re.sub(r"\s+", "", norm_blank(value).lower().lstrip("@"))


def canonical_label(value: object) -> str:
    text = norm_blank(value)
    mapping = {label.casefold(): label for label in LABELS + sorted(NON_EVALUABLE)}
    return mapping.get(text.casefold(), text)


def is_injected_id(value: object) -> bool:
    text = norm_blank(value)
    return text.upper().startswith("INJ") or "synthetic" in text.casefold()


def clean_bool(value: object) -> bool:
    return norm_blank(value).casefold() in {"true", "1", "yes", "y"}


def sha256_file(path: Path) -> str:
    if not path.exists():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_for_model(text: object) -> str:
    s = norm_blank(text)
    s = unicodedata.normalize("NFKC", s)
    s = html.unescape(s)
    s = "".join(ch for ch in s if unicodedata.category(ch)[0] != "C" or ch in "\t\n\r")
    s = re.sub(r"https?://\S+|www\.\S+", "HTTPURL", s, flags=re.IGNORECASE)
    s = re.sub(r"(?<!\w)@\w+", "@USER", s)
    return re.sub(r"\s+", " ", s).strip()


def normalize_for_group(text: object) -> str:
    s = normalize_for_model(text).casefold()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def standard_frame(data: pd.DataFrame, **constants: object) -> pd.DataFrame:
    out = data.copy()
    for key, value in constants.items():
        out[key] = value
    columns = [
        "comment_id",
        "comment_text_original",
        "video_id",
        "brand_or_video_context",
        "final_sentiment_label",
        "sentiment_target",
        "complaint_scope",
        "source_version",
        "source_file",
        "source_sample_role",
        "preferred_registry_role",
        "source_priority",
        "annotator_1_label",
        "annotator_2_label",
        "adjudication_notes",
        "adjudication_status",
    ]
    for col in columns:
        if col not in out.columns:
            out[col] = ""
    return out[columns].copy()


def rows_from_v1() -> pd.DataFrame:
    v1 = read_csv(V1_VALIDATED)
    out = pd.DataFrame(
        {
            "comment_id": v1["comment_id"].map(norm_blank),
            "comment_text_original": v1["comment_text_original"].map(norm_blank),
            "video_id": v1["video_id"].map(norm_blank),
            "brand_or_video_context": v1["brand_or_video_context"].map(norm_blank),
            "final_sentiment_label": v1["adjudicated_human_label"].map(canonical_label),
            "sentiment_target": "",
            "complaint_scope": "",
            "source_sample_role": v1["sample_set"].map(norm_blank),
            "annotator_1_label": v1.get("annotator_1_label", "").map(norm_blank),
            "annotator_2_label": v1.get("annotator_2_label", "").map(norm_blank),
            "adjudication_notes": v1.get("adjudication_notes", "").map(norm_blank),
        }
    )
    out["preferred_registry_role"] = np.where(
        out["source_sample_role"].eq("locked_test"),
        "legacy_diagnostic_test_already_opened",
        "development_human_v1",
    )
    out["source_priority"] = np.where(out["preferred_registry_role"].str.startswith("development"), 50, 10)
    return standard_frame(
        out,
        source_version="human_v1",
        source_file=V1_VALIDATED.relative_to(ROOT).as_posix(),
        adjudication_status="final_or_adjudicated",
    )


def rows_from_v2_adjudication() -> pd.DataFrame:
    adj = read_csv(V2_ADJUDICATION_FINAL)
    out = pd.DataFrame(
        {
            "comment_id": adj["comment_id"].map(norm_blank),
            "comment_text_original": adj["comment_text_original"].map(norm_blank),
            "video_id": adj["video_id"].map(norm_blank),
            "brand_or_video_context": adj["brand_or_video_context"].map(norm_blank),
            "final_sentiment_label": adj["adjudicated_sentiment_label"].map(canonical_label),
            "sentiment_target": adj.get("adjudicated_sentiment_target", "").map(norm_blank),
            "complaint_scope": adj.get("adjudicated_complaint_scope", "").map(norm_blank),
            "source_sample_role": adj["sample_role"].map(norm_blank),
            "annotator_1_label": adj.get("annotator_1_sentiment_label", "").map(norm_blank),
            "annotator_2_label": adj.get("annotator_2_sentiment_label", "").map(norm_blank),
            "adjudication_notes": adj.get("adjudication_notes", "").map(norm_blank),
        }
    )
    out["preferred_registry_role"] = np.where(
        out["source_sample_role"].eq("development_v2"),
        "development_human_v2_adjudicated",
        "legacy_diagnostic_test_already_opened",
    )
    out["source_priority"] = np.where(out["preferred_registry_role"].str.startswith("development"), 95, 15)
    return standard_frame(
        out,
        source_version="human_v2_adjudication_final",
        source_file=V2_ADJUDICATION_FINAL.relative_to(ROOT).as_posix(),
        adjudication_status="final_or_adjudicated",
    )


def rows_from_v2_validated() -> pd.DataFrame:
    v2 = read_csv(V2_VALIDATED)
    out = pd.DataFrame(
        {
            "comment_id": v2["comment_id"].map(norm_blank),
            "comment_text_original": v2["comment_text_original"].map(norm_blank),
            "video_id": v2["video_id"].map(norm_blank),
            "brand_or_video_context": v2["brand_or_video_context"].map(norm_blank),
            "final_sentiment_label": v2["final_sentiment_label"].map(canonical_label),
            "sentiment_target": v2.get("final_sentiment_target", "").map(norm_blank),
            "complaint_scope": v2.get("final_complaint_scope", "").map(norm_blank),
            "source_sample_role": v2["sample_role"].map(norm_blank),
            "annotator_1_label": v2.get("annotator_1_label", "").map(norm_blank),
            "annotator_2_label": v2.get("annotator_2_label", "").map(norm_blank),
            "adjudication_notes": v2.get("adjudication_notes", "").map(norm_blank),
        }
    )
    out["preferred_registry_role"] = np.where(
        out["source_sample_role"].eq("development_v2"),
        "development_human_v2_validated",
        "legacy_diagnostic_test_already_opened",
    )
    out["source_priority"] = np.where(out["preferred_registry_role"].str.startswith("development"), 85, 12)
    return standard_frame(
        out,
        source_version="human_v2_validated",
        source_file=V2_VALIDATED.relative_to(ROOT).as_posix(),
        adjudication_status="final_or_adjudicated",
    )


def rows_from_v2_replacements() -> pd.DataFrame:
    repl = read_csv(V2_REPLACEMENT_ADJUDICATION)
    out = pd.DataFrame(
        {
            "comment_id": repl["comment_id"].map(norm_blank),
            "comment_text_original": repl["comment_text_original"].map(norm_blank),
            "video_id": repl["video_id"].map(norm_blank),
            "brand_or_video_context": repl["brand_or_video_context"].map(norm_blank),
            "final_sentiment_label": repl["adjudicated_sentiment_label"].map(canonical_label),
            "sentiment_target": repl.get("adjudicated_sentiment_target", "").map(norm_blank),
            "complaint_scope": repl.get("adjudicated_complaint_scope", "").map(norm_blank),
            "source_sample_role": repl["sample_role"].map(norm_blank),
            "annotator_1_label": repl.get("annotator_1_sentiment_label", "").map(norm_blank),
            "annotator_2_label": repl.get("annotator_2_sentiment_label", "").map(norm_blank),
            "adjudication_notes": repl.get("adjudication_reason", "").map(norm_blank),
        }
    )
    out["preferred_registry_role"] = "legacy_diagnostic_test_already_opened"
    out["source_priority"] = 18
    return standard_frame(
        out,
        source_version="human_v2_locked_replacement_adjudication",
        source_file=V2_REPLACEMENT_ADJUDICATION.relative_to(ROOT).as_posix(),
        adjudication_status="final_or_adjudicated",
    )


def rows_from_v2_locked_final() -> pd.DataFrame:
    locked = read_csv(V2_LOCKED_FINAL)
    out = pd.DataFrame(
        {
            "comment_id": locked["comment_id"].map(norm_blank),
            "comment_text_original": locked["comment_text_original"].map(norm_blank),
            "video_id": locked["video_id"].map(norm_blank),
            "brand_or_video_context": locked["brand_or_video_context"].map(norm_blank),
            "final_sentiment_label": locked["final_sentiment_label"].map(canonical_label),
            "sentiment_target": locked.get("final_sentiment_target", "").map(norm_blank),
            "complaint_scope": locked.get("final_complaint_scope", "").map(norm_blank),
            "source_sample_role": locked["sample_role"].map(norm_blank),
            "annotator_1_label": locked.get("annotator_1_label", "").map(norm_blank),
            "annotator_2_label": locked.get("annotator_2_label", "").map(norm_blank),
            "adjudication_notes": locked.get("adjudication_notes", "").map(norm_blank),
        }
    )
    out["preferred_registry_role"] = "legacy_diagnostic_test_already_opened"
    out["source_priority"] = 16
    return standard_frame(
        out,
        source_version="human_v2_locked_observational_final",
        source_file=V2_LOCKED_FINAL.relative_to(ROOT).as_posix(),
        adjudication_status="final_or_adjudicated",
    )


def rows_from_v3_registry() -> pd.DataFrame:
    v3 = read_csv(V3_REGISTRY)
    out = pd.DataFrame(
        {
            "comment_id": v3["comment_id"].map(norm_blank),
            "comment_text_original": v3["comment_text_original"].map(norm_blank),
            "video_id": v3["video_id"].map(norm_blank),
            "brand_or_video_context": v3["brand_or_video_context"].map(norm_blank),
            "final_sentiment_label": v3["final_sentiment_label"].map(canonical_label),
            "sentiment_target": v3.get("sentiment_target", "").map(norm_blank),
            "complaint_scope": v3.get("complaint_scope", "").map(norm_blank),
            "source_sample_role": v3["registry_role"].map(norm_blank),
            "annotator_1_label": v3.get("annotator_1_label", "").map(norm_blank),
            "annotator_2_label": v3.get("annotator_2_label", "").map(norm_blank),
            "adjudication_notes": v3.get("annotation_notes", "").map(norm_blank),
        }
    )
    split_family = v3["split_family"].map(norm_blank)
    source_role = v3["registry_role"].map(norm_blank)
    historical_role = source_role.str.contains("historical|final_test", case=False, regex=True, na=False)
    out["preferred_registry_role"] = np.select(
        [
            split_family.eq("development") & ~historical_role,
            split_family.eq("final_test") | historical_role,
        ],
        [
            "development_human_v3_registry",
            "legacy_diagnostic_test_already_opened",
        ],
        default="excluded_non_evaluable_or_integrity",
    )
    out["source_priority"] = np.select(
        [
            out["preferred_registry_role"].str.startswith("development"),
            out["preferred_registry_role"].eq("legacy_diagnostic_test_already_opened"),
        ],
        [45, 8],
        default=1,
    )
    return standard_frame(
        out,
        source_version="human_v3_registry",
        source_file=V3_REGISTRY.relative_to(ROOT).as_posix(),
        adjudication_status="final_or_adjudicated_registry",
    )


def rows_from_optional_positive_adjudication() -> pd.DataFrame:
    opt = read_csv(OPTIONAL_POSITIVE_ADJUDICATION, required=False)
    if opt.empty:
        return pd.DataFrame()
    label_col = next(
        (
            col
            for col in [
                "adjudicated_sentiment_label",
                "final_sentiment_label",
                "adjudicated_label",
                "human_label",
            ]
            if col in opt.columns
        ),
        "",
    )
    if not label_col:
        return pd.DataFrame()
    text_col = "comment_text_original" if "comment_text_original" in opt.columns else "text"
    out = pd.DataFrame(
        {
            "comment_id": opt["comment_id"].map(norm_blank),
            "comment_text_original": opt.get(text_col, "").map(norm_blank),
            "video_id": opt.get("video_id", "").map(norm_blank),
            "brand_or_video_context": opt.get("brand_or_video_context", opt.get("product_category", "")).map(norm_blank),
            "final_sentiment_label": opt[label_col].map(canonical_label),
            "sentiment_target": opt.get("adjudicated_sentiment_target", opt.get("final_sentiment_target", "")).map(norm_blank),
            "complaint_scope": opt.get("adjudicated_complaint_scope", opt.get("final_complaint_scope", "")).map(norm_blank),
            "source_sample_role": opt.get("sample_role", pd.Series("positive_active_learning", index=opt.index)).map(norm_blank),
            "annotator_1_label": opt.get("annotator_1_sentiment_label", opt.get("annotator_1_label", "")).map(norm_blank),
            "annotator_2_label": opt.get("annotator_2_sentiment_label", opt.get("annotator_2_label", "")).map(norm_blank),
            "adjudication_notes": opt.get("adjudication_notes", "").map(norm_blank),
        }
    )
    out = out.loc[out["final_sentiment_label"].map(norm_blank).ne("")].copy()
    if out.empty:
        return pd.DataFrame()
    out["preferred_registry_role"] = "development_positive_active_learning"
    out["source_priority"] = 120
    return standard_frame(
        out,
        source_version="human_positive_active_learning_adjudication",
        source_file=OPTIONAL_POSITIVE_ADJUDICATION.relative_to(ROOT).as_posix(),
        adjudication_status="final_or_adjudicated",
    )


def source_rows() -> pd.DataFrame:
    frames = [
        rows_from_v1(),
        rows_from_v2_adjudication(),
        rows_from_v2_validated(),
        rows_from_v2_replacements(),
        rows_from_v2_locked_final(),
        rows_from_v3_registry(),
        rows_from_optional_positive_adjudication(),
    ]
    frames = [frame for frame in frames if not frame.empty]
    raw = pd.concat(frames, ignore_index=True)
    raw["comment_id"] = raw["comment_id"].map(norm_blank)
    raw["final_sentiment_label"] = raw["final_sentiment_label"].map(canonical_label)
    raw["source_priority"] = pd.to_numeric(raw["source_priority"], errors="coerce").fillna(0).astype(int)
    return raw.loc[raw["comment_id"].ne("")].reset_index(drop=True)


def deduplicate_sources(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    audit_rows: list[dict[str, object]] = []
    for comment_id, group in raw.groupby("comment_id", dropna=False):
        labels = sorted(set(group["final_sentiment_label"].map(canonical_label)))
        evaluable_labels = sorted(label for label in labels if label in LABELS)
        source_files = sorted(set(group["source_file"]))
        roles = sorted(set(group["preferred_registry_role"]))
        status = "PASS"
        notes = "Labels agree across retained human-final/adjudicated sources."
        if len(labels) > 1:
            status = "CONFLICT_REVIEWED_RETAIN_HIGHEST_PRIORITY"
            notes = "Multiple final labels exist across human sources; highest-priority adjudicated source is retained, lower-priority rows remain provenance only."
        if not evaluable_labels and all(label in NON_EVALUABLE for label in labels):
            status = "NON_EVALUABLE_EXCLUDED"
            notes = "No evaluable three-class label."
        if any(norm_blank(v) == "" for v in labels):
            status = "MISSING_LABEL_EXCLUDED"
            notes = "At least one source row has blank final/adjudicated label."
        audit_rows.append(
            {
                "audit_type": "comment_id_label_conflict",
                "comment_id": comment_id,
                "labels_observed": ";".join(labels),
                "roles_observed": ";".join(roles),
                "source_files": ";".join(source_files),
                "n_source_rows": int(len(group)),
                "status": status,
                "notes": notes,
            }
        )

    ordered = raw.sort_values(["comment_id", "source_priority", "source_version"], ascending=[True, False, True])
    retained = ordered.drop_duplicates("comment_id", keep="first").reset_index(drop=True)
    return retained, pd.DataFrame(audit_rows)


def add_dataset_context(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    data = read_csv(DATASET)
    keep_cols = [
        "comment_id",
        "product_category",
        "comment_type",
        "timestamp",
        "username",
        "user_id",
        "parent_comment_id",
        "parent_user",
        "text",
        "video_id",
    ]
    context = data[[c for c in keep_cols if c in data.columns]].drop_duplicates("comment_id")
    out = out.merge(context, on="comment_id", how="left", suffixes=("", "_dataset"))
    for col in ["product_category", "comment_type", "timestamp", "username", "user_id", "parent_comment_id", "parent_user", "text"]:
        if col not in out.columns:
            out[col] = ""
        out[col] = out[col].map(norm_blank)
    dataset_text = out["text"].map(norm_blank)
    out["comment_text_original"] = np.where(out["comment_text_original"].map(norm_blank).ne(""), out["comment_text_original"], dataset_text)
    out["video_id"] = np.where(out["video_id"].map(norm_blank).ne(""), out["video_id"], out.get("video_id_dataset", "").map(norm_blank))
    out["brand_or_video_context"] = np.where(
        out["brand_or_video_context"].map(norm_blank).ne(""),
        out["brand_or_video_context"],
        out["product_category"],
    )
    out["username_norm"] = out["username"].map(normalize_username)
    out["dataset_text_matches"] = out["comment_text_original"].map(normalize_for_group).eq(dataset_text.map(normalize_for_group))
    return out.drop(columns=[c for c in ["video_id_dataset"] if c in out.columns])


def add_text_clusters(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["model_text"] = out["comment_text_original"].map(normalize_for_model)
    out["normalized_text_for_group"] = out["comment_text_original"].map(normalize_for_group)
    out["exact_text_hash"] = out["normalized_text_for_group"].map(text_hash)

    ids = out["comment_id"].tolist()
    uf = UnionFind(ids)
    for _, group in out.groupby("exact_text_hash"):
        members = group["comment_id"].tolist()
        for member in members[1:]:
            uf.union(members[0], member)

    texts = out[["comment_id", "normalized_text_for_group"]].drop_duplicates("comment_id").to_dict("records")
    for i, left in enumerate(texts):
        left_text = left["normalized_text_for_group"]
        if len(left_text) < 8:
            continue
        for right in texts[i + 1 :]:
            right_text = right["normalized_text_for_group"]
            if len(right_text) < 8:
                continue
            max_len = max(len(left_text), len(right_text))
            if abs(len(left_text) - len(right_text)) > max(10, 0.20 * max_len):
                continue
            score = SequenceMatcher(a=left_text, b=right_text, autojunk=False).ratio()
            if score >= 0.96:
                uf.union(left["comment_id"], right["comment_id"])

    roots = {comment_id: uf.find(comment_id) for comment_id in ids}
    root_to_cluster = {root: f"text_cluster_{idx:05d}" for idx, root in enumerate(sorted(set(roots.values())), 1)}
    out["text_cluster_id"] = out["comment_id"].map(lambda value: root_to_cluster[roots[value]])
    out["text_cluster_size"] = out.groupby("text_cluster_id")["comment_id"].transform("nunique").astype(int)
    out["near_duplicate_cluster_method"] = np.where(out["text_cluster_size"].gt(1), "exact_or_sequence_ratio_ge_0.96", "singleton")
    return out


def assign_split_family(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["is_injected"] = out["comment_id"].map(is_injected_id)
    out["is_observational"] = ~out["is_injected"]
    out["is_evaluable_three_class"] = out["final_sentiment_label"].isin(LABELS)
    role = out["preferred_registry_role"].map(norm_blank)
    out["split_family"] = np.select(
        [
            out["is_injected"],
            ~out["is_evaluable_three_class"],
            role.str.startswith("development", na=False),
            role.eq("legacy_diagnostic_test_already_opened"),
        ],
        [
            "excluded_injected",
            "excluded_non_evaluable",
            "development",
            "legacy_diagnostic_test_already_opened",
        ],
        default="excluded_other",
    )
    out["registry_role"] = np.select(
        [
            out["split_family"].eq("development"),
            out["split_family"].eq("legacy_diagnostic_test_already_opened"),
            out["split_family"].eq("excluded_injected"),
            out["split_family"].eq("excluded_non_evaluable"),
        ],
        [
            role,
            "LEGACY_DIAGNOSTIC_TEST_ALREADY_OPENED",
            "excluded_injected",
            "excluded_non_evaluable",
        ],
        default=role,
    )
    duplicate_group = out["text_cluster_size"].astype(int).gt(1)
    out["video_group_id"] = np.where(
        out["video_id"].map(norm_blank).ne(""),
        "video:" + out["video_id"].map(norm_blank),
        "video:missing",
    )
    out["cv_group_id"] = np.where(
        duplicate_group,
        "dup:" + out["text_cluster_id"],
        "comment:" + out["comment_id"].map(norm_blank),
    )
    out["selected_fold"] = ""
    return out


def fold_candidate_rows(dev: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    tmp = dev.copy()
    tmp["candidate_fold"] = ""
    splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    y = tmp["final_sentiment_label"].to_numpy()
    groups = tmp["cv_group_id"].to_numpy()
    for fold, (_, val_idx) in enumerate(splitter.split(tmp, y, groups), start=1):
        tmp.iloc[val_idx, tmp.columns.get_loc("candidate_fold")] = str(fold)

    overall_props = tmp["final_sentiment_label"].value_counts(normalize=True).reindex(LABELS, fill_value=0.0)
    ideal_size = len(tmp) / N_SPLITS
    rows: list[dict[str, object]] = []
    missing_penalty = 0.0
    max_size_deviation = 0.0
    max_class_prop_deviation = 0.0
    for fold, group in tmp.groupby("candidate_fold"):
        counts = group["final_sentiment_label"].value_counts().reindex(LABELS, fill_value=0)
        props = group["final_sentiment_label"].value_counts(normalize=True).reindex(LABELS, fill_value=0.0)
        missing = [label for label in LABELS if counts[label] == 0]
        missing_penalty += 10 * len(missing)
        size_deviation = abs(len(group) - ideal_size) / max(ideal_size, 1)
        class_deviation = float((props - overall_props).abs().max())
        max_size_deviation = max(max_size_deviation, float(size_deviation))
        max_class_prop_deviation = max(max_class_prop_deviation, class_deviation)
        rows.append(
            {
                "candidate_seed": seed,
                "fold": fold,
                "n_rows": int(len(group)),
                "n_negative": int(counts["Negative"]),
                "n_neutral": int(counts["Neutral"]),
                "n_positive": int(counts["Positive"]),
                "prop_negative": float(props["Negative"]),
                "prop_neutral": float(props["Neutral"]),
                "prop_positive": float(props["Positive"]),
                "missing_classes": ";".join(missing),
                "size_deviation_from_ideal": float(size_deviation),
                "max_class_prop_deviation_from_overall": class_deviation,
            }
        )
    leakage = (
        tmp.groupby("text_cluster_id")["candidate_fold"].nunique().gt(1).sum()
        + tmp.groupby("cv_group_id")["candidate_fold"].nunique().gt(1).sum()
    )
    aggregate_score = missing_penalty + max_size_deviation + max_class_prop_deviation + 100 * int(leakage)
    frame = pd.DataFrame(rows)
    frame["candidate_score"] = float(aggregate_score)
    frame["candidate_has_leakage"] = bool(leakage > 0)
    frame["candidate_missing_class_penalty"] = float(missing_penalty)
    return frame, tmp[["comment_id", "candidate_fold"]]


def assign_selected_folds(registry: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    out = registry.copy()
    dev = out.loc[out["split_family"].eq("development") & out["is_evaluable_three_class"]].copy()
    if dev.empty:
        raise RuntimeError("No development rows available after human-label audit.")

    candidate_frames = []
    manifests: dict[int, pd.DataFrame] = {}
    for seed in FOLD_SEEDS:
        frame, manifest = fold_candidate_rows(dev, seed)
        candidate_frames.append(frame)
        manifests[seed] = manifest
    candidates = pd.concat(candidate_frames, ignore_index=True)
    summary = candidates.groupby("candidate_seed", as_index=False).agg(
        candidate_score=("candidate_score", "first"),
        candidate_has_leakage=("candidate_has_leakage", "first"),
        candidate_missing_class_penalty=("candidate_missing_class_penalty", "first"),
        max_size_deviation=("size_deviation_from_ideal", "max"),
        max_class_prop_deviation=("max_class_prop_deviation_from_overall", "max"),
    )
    eligible = summary.loc[~summary["candidate_has_leakage"] & summary["candidate_missing_class_penalty"].eq(0)].copy()
    selected_seed = int(
        (eligible if not eligible.empty else summary)
        .sort_values(["candidate_score", "max_class_prop_deviation", "max_size_deviation", "candidate_seed"])
        .iloc[0]["candidate_seed"]
    )
    candidates["selected_candidate"] = candidates["candidate_seed"].eq(selected_seed)
    fold_map = dict(zip(manifests[selected_seed]["comment_id"], manifests[selected_seed]["candidate_fold"]))
    out["selected_fold"] = out["comment_id"].map(fold_map).fillna("")

    selected_dev = out.loc[out["split_family"].eq("development")].copy()
    distribution_rows = []
    for fold, group in selected_dev.groupby("selected_fold"):
        counts = group["final_sentiment_label"].value_counts().reindex(LABELS, fill_value=0)
        for label in LABELS:
            distribution_rows.append(
                {
                    "fold": fold,
                    "label": label,
                    "count": int(counts[label]),
                    "fold_rows": int(len(group)),
                    "proportion": float(counts[label] / len(group)) if len(group) else 0.0,
                    "selected_seed": selected_seed,
                }
            )
    distribution = pd.DataFrame(distribution_rows)
    leakage = fold_leakage_audit(out)
    return out, candidates, distribution, leakage


def fold_leakage_audit(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    dev = frame.loc[frame["split_family"].eq("development")].copy()
    for group_col in ["text_cluster_id", "cv_group_id"]:
        for group_id, group in dev.groupby(group_col):
            folds = sorted(set(group["selected_fold"].map(norm_blank)) - {""})
            if len(folds) > 1:
                rows.append(
                    {
                        "audit_type": f"{group_col}_cross_fold",
                        "group_id": group_id,
                        "folds": ";".join(folds),
                        "n_rows": int(len(group)),
                        "comment_ids": "|".join(group["comment_id"].astype(str).head(50)),
                        "status": "FAIL",
                        "notes": "Duplicate/near-duplicate or video group appears in multiple development folds.",
                    }
                )
    for group_id, group in dev.groupby("video_group_id"):
        folds = sorted(set(group["selected_fold"].map(norm_blank)) - {""})
        if len(folds) > 1:
            rows.append(
                {
                    "audit_type": "video_id_cross_fold_soft_audit",
                    "group_id": group_id,
                    "folds": ";".join(folds),
                    "n_rows": int(len(group)),
                    "comment_ids": "|".join(group["comment_id"].astype(str).head(50)),
                    "status": "WARN",
                    "notes": "Video_id was audited but not used as a hard fold group because full video grouping caused severe fold imbalance.",
                }
            )
    for fold, group in dev.groupby("selected_fold"):
        missing = [label for label in LABELS if not group["final_sentiment_label"].eq(label).any()]
        rows.append(
            {
                "audit_type": "fold_class_presence",
                "group_id": fold,
                "folds": fold,
                "n_rows": int(len(group)),
                "comment_ids": "",
                "status": "PASS" if not missing else "FAIL",
                "notes": "All three labels present." if not missing else f"Missing labels: {';'.join(missing)}",
            }
        )
    if not rows:
        rows.append(
            {
                "audit_type": "development_fold_integrity",
                "group_id": "",
                "folds": "",
                "n_rows": 0,
                "comment_ids": "",
                "status": "PASS",
                "notes": "No development fold leakage found.",
            }
        )
    return pd.DataFrame(rows)


def duplicate_cluster_audit(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cluster_id, group in frame.groupby("text_cluster_id"):
        if len(group) <= 1:
            continue
        split_families = sorted(set(group["split_family"]))
        labels = sorted(set(group["final_sentiment_label"]))
        rows.append(
            {
                "text_cluster_id": cluster_id,
                "text_cluster_size": int(group["comment_id"].nunique()),
                "exact_text_hashes": ";".join(sorted(set(group["exact_text_hash"]))),
                "labels_observed": ";".join(labels),
                "split_families": ";".join(split_families),
                "registry_roles": ";".join(sorted(set(group["registry_role"]))),
                "source_files": ";".join(sorted(set(group["source_file"]))),
                "video_ids": ";".join(sorted(set(group["video_id"].map(norm_blank)) - {""})),
                "comment_ids": "|".join(group["comment_id"].astype(str)),
                "status": "REVIEW" if len(labels) > 1 or len(split_families) > 1 else "PASS",
                "notes": "Exact or near-identical text cluster; kept within a single development fold when used for development.",
            }
        )
    if not rows:
        rows.append(
            {
                "text_cluster_id": "",
                "text_cluster_size": 0,
                "exact_text_hashes": "",
                "labels_observed": "",
                "split_families": "",
                "registry_roles": "",
                "source_files": "",
                "video_ids": "",
                "comment_ids": "",
                "status": "PASS",
                "notes": "No duplicate or near-identical text clusters found.",
            }
        )
    return pd.DataFrame(rows)


def source_summary(raw: pd.DataFrame, registry: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (source_version, source_file, source_role), group in raw.groupby(["source_version", "source_file", "source_sample_role"], dropna=False):
        row = {
            "source_version": source_version,
            "source_file": source_file,
            "source_sha256": sha256_file(ROOT / source_file),
            "source_sample_role": source_role,
            "raw_rows": int(len(group)),
            "raw_unique_comment_id": int(group["comment_id"].nunique()),
            "raw_evaluable_rows": int(group["final_sentiment_label"].isin(LABELS).sum()),
        }
        for label in LABELS + ["No Text", "Uncertain", ""]:
            row[f"raw_{label.lower().replace(' ', '_') or 'blank'}"] = int(group["final_sentiment_label"].eq(label).sum())
        rows.append(row)
    retained_counts = (
        registry.groupby(["source_version", "source_file", "source_sample_role"])["comment_id"]
        .nunique()
        .rename("retained_unique_comment_id")
        .reset_index()
    )
    result = pd.DataFrame(rows).merge(
        retained_counts,
        on=["source_version", "source_file", "source_sample_role"],
        how="left",
    )
    result["retained_unique_comment_id"] = result["retained_unique_comment_id"].fillna(0).astype(int)
    return result.sort_values(["source_version", "source_sample_role", "source_file"]).reset_index(drop=True)


def positive_signal_flags(text: object) -> dict[str, bool]:
    s = norm_blank(text)
    lower = s.casefold()
    explicit_positive = bool(re.search(r"\b(bagus|mantap|keren|ampuh|recommended|recommend|suka|cocok|worth|love)\b", lower))
    return {
        "testimonial_result": bool(re.search(r"\b(hasil|perubahan|berubah|hilang|mudar|memudar|sembuh|glow|glowing|cerah|mencerahkan|jerawat.*(hilang|mendingan)|bekas.*(pudar|hilang))\b", lower)),
        "implicit_support": bool(re.search(r"\b(aku pake|aku pakai|udah pake|sudah pakai|tetap pake|repurchase|beli lagi|langganan|setia|habis.*beli)\b", lower)),
        "short_recommendation": bool(re.search(r"\b(coba deh|wajib coba|rekomen|recommended|recommend|worth it|mantul|gas)\b", lower)),
        "positive_emoji": bool(re.search(r"[❤♥💕😍🥰😘😊😁✨👍🤩💖]", s)),
        "positive_without_explicit_bagus": not explicit_positive,
        "positive_neutral_ambiguous_text": bool(re.search(r"\b(cocok ga|aman ga|baru coba|semoga|mudah-mudahan|spill|pakai apa|efeknya)\b", lower)),
        "short_context_needed": len(lower.split()) <= 5,
    }


def build_active_learning_package(registry: pd.DataFrame) -> pd.DataFrame:
    obs = read_csv(OBS_V2_INFERENCE)
    used_ids = set(registry["comment_id"].map(norm_blank))
    obs = obs.loc[~obs["comment_id"].map(norm_blank).isin(used_ids)].copy()
    obs = obs.loc[~obs["comment_id"].map(is_injected_id)].copy()
    for col in ["probability_negative", "probability_neutral", "probability_positive", "max_probability"]:
        obs[col] = pd.to_numeric(obs.get(col, 0), errors="coerce").fillna(0.0)

    probs = obs[["probability_negative", "probability_neutral", "probability_positive"]].to_numpy()
    pos_rank = (probs > probs[:, [2]]).sum(axis=1) + 1
    obs["positive_probability_rank"] = pos_rank
    obs["neutral_positive_margin"] = obs["probability_neutral"] - obs["probability_positive"]
    obs["v2_neutral_positive_second"] = obs.get("predicted_sentiment", "").eq("Neutral") & obs["positive_probability_rank"].eq(2)
    obs["small_neutral_positive_margin"] = obs["neutral_positive_margin"].abs().le(0.12)
    obs["positive_probability_for_review"] = obs["probability_positive"].ge(0.25)
    flags = obs["text"].apply(positive_signal_flags).apply(pd.Series)
    obs = pd.concat([obs, flags], axis=1)

    disagreement_ids: set[str] = set()
    if V2_V3_SAME_TEST.exists():
        same = read_csv(V2_V3_SAME_TEST)
        if {"comment_id", "v2_predicted_label", "v3_predicted_label"}.issubset(same.columns):
            disagreement_ids = set(same.loc[~same["v2_predicted_label"].eq(same["v3_predicted_label"]), "comment_id"])
    obs["v2_indobert_v3_disagreement_if_available"] = obs["comment_id"].isin(disagreement_ids)

    criteria_cols = [
        "v2_neutral_positive_second",
        "small_neutral_positive_margin",
        "v2_indobert_v3_disagreement_if_available",
        "testimonial_result",
        "implicit_support",
        "short_recommendation",
        "positive_emoji",
        "positive_without_explicit_bagus",
        "positive_neutral_ambiguous_text",
        "short_context_needed",
        "positive_probability_for_review",
    ]
    weights = {
        "v2_neutral_positive_second": 4,
        "small_neutral_positive_margin": 3,
        "v2_indobert_v3_disagreement_if_available": 2,
        "testimonial_result": 2,
        "implicit_support": 2,
        "short_recommendation": 2,
        "positive_emoji": 1,
        "positive_without_explicit_bagus": 1,
        "positive_neutral_ambiguous_text": 2,
        "short_context_needed": 1,
        "positive_probability_for_review": 2,
    }
    obs["active_learning_priority"] = sum(obs[col].astype(int) * weight for col, weight in weights.items())
    candidates = obs.loc[obs["active_learning_priority"].gt(0)].copy()
    candidates = candidates.sort_values(
        ["active_learning_priority", "probability_positive", "small_neutral_positive_margin", "neutral_positive_margin", "comment_id"],
        ascending=[False, False, False, True, True],
    ).head(ACTIVE_LEARNING_N)
    candidates = candidates.reset_index(drop=True)
    candidates["annotation_item_id"] = [f"PRAL{i:04d}" for i in range(1, len(candidates) + 1)]
    candidates["sample_role"] = "positive_active_learning_candidate_blind"
    candidates["sampling_reason"] = candidates[criteria_cols].apply(
        lambda row: ";".join([col for col in criteria_cols if bool(row[col])]),
        axis=1,
    )

    manifest_cols = [
        "annotation_item_id",
        "sample_role",
        "comment_id",
        "text",
        "video_id",
        "product_category",
        "comment_type",
        "timestamp",
        "username",
        "predicted_sentiment",
        "probability_negative",
        "probability_neutral",
        "probability_positive",
        "neutral_positive_margin",
        "active_learning_priority",
        "sampling_reason",
    ] + criteria_cols
    for col in manifest_cols:
        if col not in candidates.columns:
            candidates[col] = ""
    candidates[manifest_cols].to_csv(OUT_AL_MANIFEST, index=False, encoding="utf-8-sig")

    blind_cols = [
        "annotation_item_id",
        "sample_role",
        "comment_id",
        "comment_text_original",
        "video_id",
        "brand_or_video_context",
        "product_category",
        "comment_type",
        "timestamp",
        "username",
    ]
    blind = candidates.rename(columns={"text": "comment_text_original"}).copy()
    for col in blind_cols:
        if col not in blind.columns:
            blind[col] = ""
    blind = blind[blind_cols]
    blind.to_csv(OUT_AL_BLIND, index=False, encoding="utf-8-sig")

    annotator_cols = blind_cols + [
        "annotator_sentiment_label",
        "annotator_sentiment_target",
        "annotator_complaint_scope",
        "annotator_notes",
    ]
    for out_path in [OUT_AL_A1, OUT_AL_A2]:
        template = blind.copy()
        for col in annotator_cols:
            if col not in template.columns:
                template[col] = ""
        template[annotator_cols].to_csv(out_path, index=False, encoding="utf-8-sig")

    adj_cols = blind_cols + [
        "annotator_1_sentiment_label",
        "annotator_1_notes",
        "annotator_2_sentiment_label",
        "annotator_2_notes",
        "adjudicated_sentiment_label",
        "adjudicated_sentiment_target",
        "adjudicated_complaint_scope",
        "adjudication_notes",
    ]
    template = blind.copy()
    for col in adj_cols:
        if col not in template.columns:
            template[col] = ""
    template[adj_cols].to_csv(OUT_AL_ADJ, index=False, encoding="utf-8-sig")
    return candidates


def build_new_locked_test_templates(registry: pd.DataFrame, active_learning: pd.DataFrame) -> None:
    obs = read_csv(OBS_V2_INFERENCE)
    used = set(registry["comment_id"].map(norm_blank)) | set(active_learning["comment_id"].map(norm_blank))
    obs = obs.loc[~obs["comment_id"].map(norm_blank).isin(used)].copy()
    obs = obs.loc[~obs["comment_id"].map(is_injected_id)].copy()
    for col in ["probability_negative", "probability_neutral", "probability_positive", "max_probability"]:
        obs[col] = pd.to_numeric(obs.get(col, 0), errors="coerce").fillna(0.0)
    obs["sampling_bucket"] = np.select(
        [
            obs["probability_positive"].ge(0.35),
            obs["probability_negative"].ge(0.35),
            obs["probability_neutral"].ge(0.45),
        ],
        ["model_positive_enrichment", "model_negative_enrichment", "model_neutral_enrichment"],
        default="uncertain_or_low_confidence",
    )
    targets = {
        "model_positive_enrichment": 180,
        "model_negative_enrichment": 160,
        "model_neutral_enrichment": 200,
        "uncertain_or_low_confidence": 60,
    }
    sampled_parts = []
    rng_seed = 20260724
    for bucket, n in targets.items():
        pool = obs.loc[obs["sampling_bucket"].eq(bucket)].copy()
        if len(pool) == 0:
            continue
        sampled_parts.append(pool.sample(n=min(n, len(pool)), random_state=rng_seed))
    sample = pd.concat(sampled_parts, ignore_index=True) if sampled_parts else pd.DataFrame()
    if len(sample) < NEW_LOCKED_N:
        extra = obs.loc[~obs["comment_id"].isin(set(sample.get("comment_id", [])))].copy()
        if not extra.empty:
            extra = extra.sample(n=min(NEW_LOCKED_N - len(sample), len(extra)), random_state=rng_seed + 1)
            sample = pd.concat([sample, extra], ignore_index=True)
    sample = sample.head(NEW_LOCKED_N).reset_index(drop=True)
    sample["locked_test_item_id"] = [f"PRLT{i:04d}" for i in range(1, len(sample) + 1)]
    sample["sample_role"] = "new_locked_test_candidate_blind"
    sample["sampling_note"] = "Blind human annotation required before this can become a locked test."

    manifest_cols = [
        "locked_test_item_id",
        "sample_role",
        "comment_id",
        "text",
        "video_id",
        "product_category",
        "comment_type",
        "timestamp",
        "username",
        "sampling_bucket",
        "predicted_sentiment",
        "probability_negative",
        "probability_neutral",
        "probability_positive",
        "max_probability",
        "sampling_note",
    ]
    for col in manifest_cols:
        if col not in sample.columns:
            sample[col] = ""
    sample[manifest_cols].to_csv(OUT_LOCKED_SAMPLE_MANIFEST, index=False, encoding="utf-8-sig")

    blind_cols = [
        "locked_test_item_id",
        "sample_role",
        "comment_id",
        "comment_text_original",
        "video_id",
        "brand_or_video_context",
        "product_category",
        "comment_type",
        "timestamp",
        "username",
    ]
    blind = sample.rename(columns={"text": "comment_text_original"}).copy()
    if "brand_or_video_context" not in blind.columns:
        blind["brand_or_video_context"] = blind.get("product_category", "")
    blind = blind[blind_cols]
    for out_path in [OUT_LOCKED_A1, OUT_LOCKED_A2]:
        template = blind.copy()
        for col in blind_cols + ["annotator_sentiment_label", "annotator_sentiment_target", "annotator_complaint_scope", "annotator_notes"]:
            if col not in template.columns:
                template[col] = ""
        template.to_csv(out_path, index=False, encoding="utf-8-sig")

    adjudication_cols = blind_cols + [
        "annotator_1_sentiment_label",
        "annotator_1_notes",
        "annotator_2_sentiment_label",
        "annotator_2_notes",
        "adjudicated_sentiment_label",
        "adjudicated_sentiment_target",
        "adjudicated_complaint_scope",
        "adjudication_notes",
    ]
    adj = blind.copy()
    for col in adjudication_cols:
        if col not in adj.columns:
            adj[col] = ""
    adj[adjudication_cols].to_csv(OUT_LOCKED_ADJ, index=False, encoding="utf-8-sig")

    final_cols = blind_cols + [
        "final_sentiment_label",
        "final_sentiment_target",
        "final_complaint_scope",
        "human_adjudication_status",
        "freeze_status",
    ]
    final_template = blind.copy()
    for col in final_cols:
        if col not in final_template.columns:
            final_template[col] = ""
    final_template["human_adjudication_status"] = "PENDING_HUMAN_ADJUDICATION"
    final_template["freeze_status"] = "NOT_OPENED_FOR_MODEL_EVALUATION"
    final_template[final_cols].to_csv(OUT_LOCKED_FINAL, index=False, encoding="utf-8-sig")

    final_hash = sha256_file(OUT_LOCKED_FINAL)
    OUT_LOCKED_SHA.write_text(f"{final_hash}  {OUT_LOCKED_FINAL.relative_to(ROOT).as_posix()}  PENDING_HUMAN_ADJUDICATION\n", encoding="utf-8")
    freeze_manifest = {
        "status": "NEW_LOCKED_TEST_TEMPLATE_PREPARED_PENDING_HUMAN_ADJUDICATION",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "n_candidate_rows": int(len(sample)),
        "required_before_evaluation": [
            "two independent blind annotator files completed",
            "disagreements adjudicated",
            "INJ/No Text/Uncertain removed from evaluable locked test",
            "at least 100 Positive and 100 Negative adjudicated evaluable rows",
            "no exact or near-duplicate leakage with development or active-learning data",
            "candidate model and thresholds frozen before final labels are used for evaluation",
        ],
        "template_sha256": final_hash,
        "locked_test_used_for_training_or_selection": False,
        "low_confidence_to_positive_rule": False,
    }
    OUT_LOCKED_FREEZE_MANIFEST.write_text(json.dumps(freeze_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_codebook() -> None:
    lines = [
        "# Positive-Recall Sentiment Annotation Codebook",
        "",
        "Labels allowed for evaluable sentiment:",
        "",
        "- Negative: complaint, disappointment, harm, distrust, rejection, or clearly unfavorable sentiment.",
        "- Neutral: question, factual request, transaction/logistics, unclear stance, or informational comment without sentiment.",
        "- Positive: support, recommendation, favorable experience, trust, satisfaction, or clearly favorable sentiment.",
        "",
        "Non-evaluable labels:",
        "",
        "- No Text: empty, deleted, or text cannot be evaluated.",
        "- Uncertain: insufficient evidence, needs context that is not available, or annotators cannot adjudicate reliably.",
        "- INJ: injected/synthetic diagnostic comment; exclude from training and locked tests.",
        "",
        "Positive recall focus:",
        "",
        "- Testimony may be Positive even without the word 'bagus' when it reports beneficial results.",
        "- Implicit support can be Positive when the favorable stance is clear.",
        "- Short recommendations can be Positive only when the recommendation is explicit enough.",
        "- Emoji can support interpretation, but emoji alone should not override ambiguous text.",
        "",
        "Forbidden shortcuts:",
        "",
        "- Do not label all unknown comments as Positive.",
        "- Do not convert all Neutral or Uncertain comments to Positive.",
        "- Do not use HCC status, promotion suspicion, model prediction, lexicon, or goal orientation as ground truth.",
        "- Do not infer sentiment from sampling reason; sampling reason is not a label.",
    ]
    OUT_CODEBOOK.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_outputs(registry: pd.DataFrame, raw: pd.DataFrame, conflict: pd.DataFrame, fold_candidates: pd.DataFrame, fold_distribution: pd.DataFrame, leakage: pd.DataFrame) -> None:
    ordered = [
        "comment_id",
        "registry_role",
        "split_family",
        "selected_fold",
        "final_sentiment_label",
        "is_evaluable_three_class",
        "is_observational",
        "is_injected",
        "comment_text_original",
        "model_text",
        "video_id",
        "brand_or_video_context",
        "product_category",
        "comment_type",
        "timestamp",
        "username",
        "username_norm",
        "source_version",
        "source_file",
        "source_sample_role",
        "sentiment_target",
        "complaint_scope",
        "annotator_1_label",
        "annotator_2_label",
        "adjudication_notes",
        "adjudication_status",
        "text_cluster_id",
        "text_cluster_size",
        "near_duplicate_cluster_method",
        "exact_text_hash",
        "cv_group_id",
        "video_group_id",
        "dataset_text_matches",
    ]
    for col in ordered:
        if col not in registry.columns:
            registry[col] = ""
    registry[ordered].sort_values(["split_family", "registry_role", "selected_fold", "comment_id"]).to_csv(
        OUT_REGISTRY, index=False, encoding="utf-8-sig"
    )
    duplicate_cluster_audit(registry).to_csv(OUT_DUPLICATE_AUDIT, index=False, encoding="utf-8-sig")
    conflict.to_csv(OUT_CONFLICT_AUDIT, index=False, encoding="utf-8-sig")
    registry[[
        "comment_id",
        "split_family",
        "registry_role",
        "selected_fold",
        "final_sentiment_label",
        "video_id",
        "text_cluster_id",
        "text_cluster_size",
        "cv_group_id",
        "video_group_id",
        "source_version",
        "source_file",
        "source_sample_role",
    ]].to_csv(OUT_SPLIT_MANIFEST, index=False, encoding="utf-8-sig")
    source_summary(raw, registry).to_csv(OUT_SOURCE_SUMMARY, index=False, encoding="utf-8-sig")
    fold_candidates.to_csv(OUT_FOLD_CANDIDATES, index=False, encoding="utf-8-sig")
    registry.loc[registry["split_family"].eq("development"), [
        "comment_id",
        "selected_fold",
        "final_sentiment_label",
        "video_id",
        "text_cluster_id",
        "cv_group_id",
    ]].to_csv(OUT_SELECTED_FOLD, index=False, encoding="utf-8-sig")
    fold_distribution.to_csv(OUT_FOLD_DIST, index=False, encoding="utf-8-sig")
    leakage.to_csv(OUT_FOLD_LEAKAGE, index=False, encoding="utf-8-sig")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    raw = source_rows()
    deduped, conflict = deduplicate_sources(raw)
    registry = add_dataset_context(deduped)
    registry = add_text_clusters(registry)
    registry = assign_split_family(registry)
    registry, fold_candidates, fold_distribution, leakage = assign_selected_folds(registry)
    active = build_active_learning_package(registry)
    build_new_locked_test_templates(registry, active)
    write_codebook()
    write_outputs(registry, raw, conflict, fold_candidates, fold_distribution, leakage)

    development = registry.loc[registry["split_family"].eq("development") & registry["is_evaluable_three_class"]].copy()
    class_counts = {label: int(development["final_sentiment_label"].eq(label).sum()) for label in LABELS}
    target_gate = {
        "positive_250_to_350": 250 <= class_counts["Positive"] <= 350,
        "negative_min_250": class_counts["Negative"] >= 250,
        "neutral_min_400": class_counts["Neutral"] >= 400,
    }
    integrity_pass = not leakage["status"].eq("FAIL").any()
    manifest = {
        "status": "HUMAN_SUPERVISED_V2_POSITIVE_RECALL_REGISTRY_BUILT",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "registry_rows": int(len(registry)),
        "development_rows": int(len(development)),
        "development_class_counts": class_counts,
        "target_development_data_gate": target_gate,
        "target_development_data_gate_passed": bool(all(target_gate.values())),
        "split_leakage_audit_passed": bool(integrity_pass),
        "selected_fold_seed": int(fold_candidates.loc[fold_candidates["selected_candidate"].eq(True), "candidate_seed"].iloc[0]),
        "annotation_sources": sorted(set(raw["source_file"])),
        "methodological_confirmations": {
            "ground_truth_sources": "human final/adjudicated labels only",
            "v2_predictions_used_as_ground_truth": False,
            "indobert_predictions_used_as_ground_truth": False,
            "lexicon_used_as_ground_truth": False,
            "pseudo_labels_used_as_ground_truth": False,
            "llm_labels_used_as_ground_truth": False,
            "hcc_status_used_as_ground_truth": False,
            "legacy_locked_tests_used_for_training_or_tuning": False,
            "unknown_comments_forced_positive": False,
        },
        "new_annotation_status": {
            "positive_active_learning_blind_rows": int(len(active)),
            "positive_active_learning_adjudicated_rows_loaded": int(
                raw["source_version"].eq("human_positive_active_learning_adjudication").sum()
            ),
            "new_locked_test_template_rows": int(len(read_csv(OUT_LOCKED_FINAL))),
            "new_locked_test_final_human_labels_available": False,
        },
        "input_sha256": {
            path.relative_to(ROOT).as_posix(): sha256_file(path)
            for path in [
                V1_VALIDATED,
                V2_VALIDATED,
                V2_ADJUDICATION_FINAL,
                V2_REPLACEMENT_ADJUDICATION,
                V2_LOCKED_FINAL,
                V3_REGISTRY,
                OBS_V2_INFERENCE,
            ]
        },
        "outputs": {
            "registry": OUT_REGISTRY.relative_to(ROOT).as_posix(),
            "duplicate_cluster_audit": OUT_DUPLICATE_AUDIT.relative_to(ROOT).as_posix(),
            "label_conflict_audit": OUT_CONFLICT_AUDIT.relative_to(ROOT).as_posix(),
            "split_manifest": OUT_SPLIT_MANIFEST.relative_to(ROOT).as_posix(),
            "annotation_source_summary": OUT_SOURCE_SUMMARY.relative_to(ROOT).as_posix(),
            "positive_active_learning_blind": OUT_AL_BLIND.relative_to(ROOT).as_posix(),
            "new_locked_test_final_template": OUT_LOCKED_FINAL.relative_to(ROOT).as_posix(),
        },
    }
    OUT_MANIFEST_JSON.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if not integrity_pass:
        raise AssertionError("Fold leakage audit failed.")
    print(json.dumps({
        "status": manifest["status"],
        "development_class_counts": class_counts,
        "target_development_data_gate_passed": manifest["target_development_data_gate_passed"],
        "positive_active_learning_blind_rows": len(active),
        "new_locked_test_template_rows": manifest["new_annotation_status"]["new_locked_test_template_rows"],
    }, indent=2))


if __name__ == "__main__":
    main()
