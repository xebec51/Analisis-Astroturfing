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

V1_VALIDATED = HUMAN_V1_DIR / "sentiment_human_annotation_validated.csv"
V2_VALIDATED = HUMAN_V2_DIR / "sentiment_human_annotation_v2_validated.csv"
V2_LOCKED_FINAL = HUMAN_V2_DIR / "locked_test_v2_observational_final.csv"
DATASET = ROOT / "dataset.csv"
METADATA = ROOT / "video_metadata_clean.csv"

OUT_REGISTRY = HUMAN_V3_DIR / "human_label_registry_v3.csv"
OUT_CONFLICT_AUDIT = HUMAN_V3_DIR / "human_label_conflict_audit_v3.csv"
OUT_SPLIT_MANIFEST = HUMAN_V3_DIR / "data_split_manifest_v3.csv"
OUT_MANIFEST = HUMAN_V3_DIR / "human_label_registry_v3_manifest.json"

LABELS = ["Negative", "Neutral", "Positive"]
NON_EVALUABLE = {"No Text", "Uncertain", ""}
RANDOM_SEED = 42
N_SPLITS = 5


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


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, dtype=str, keep_default_na=False, low_memory=False)


def norm_blank(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none", "null", "<na>"}:
        return ""
    return text


def clean_bool(value: object) -> bool:
    return norm_blank(value).lower() in {"true", "1", "yes", "y"}


def sha256_file(path: Path) -> str:
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


def is_injected_id(comment_id: object) -> bool:
    return norm_blank(comment_id).upper().startswith("INJ")


def canonical_label(value: object) -> str:
    label = norm_blank(value)
    mapping = {label.lower(): label for label in LABELS + sorted(NON_EVALUABLE)}
    return mapping.get(label.lower(), label)


def dataset_context() -> pd.DataFrame:
    if not DATASET.exists():
        return pd.DataFrame(columns=["comment_id", "product_category", "comment_type", "timestamp", "username", "text"])
    data = read_csv(DATASET)
    keep = [c for c in ["comment_id", "product_category", "comment_type", "timestamp", "username", "text"] if c in data.columns]
    data = data[keep].drop_duplicates("comment_id")
    return data


def metadata_context() -> pd.DataFrame:
    if not METADATA.exists():
        return pd.DataFrame(columns=["video_id", "caption"])
    meta = read_csv(METADATA)
    keep = [c for c in ["video_id", "caption", "product_category", "brand", "brand_or_video_context"] if c in meta.columns]
    return meta[keep].drop_duplicates("video_id")


def rows_from_v1() -> pd.DataFrame:
    v1 = read_csv(V1_VALIDATED)
    out = pd.DataFrame(
        {
            "comment_id": v1["comment_id"].map(norm_blank),
            "comment_text_original": v1["comment_text_original"].map(norm_blank),
            "video_id": v1["video_id"].map(norm_blank),
            "brand_or_video_context": v1["brand_or_video_context"].map(norm_blank),
            "final_sentiment_label": v1["adjudicated_human_label"].map(canonical_label),
            "source_version": "human_v1",
            "source_file": V1_VALIDATED.relative_to(ROOT).as_posix(),
            "source_sample_role": v1["sample_set"].map(norm_blank),
            "sentiment_target": "",
            "complaint_scope": "",
            "annotation_notes": v1.get("adjudication_notes", "").map(norm_blank),
            "annotator_1_label": v1.get("annotator_1_label", "").map(norm_blank),
            "annotator_2_label": v1.get("annotator_2_label", "").map(norm_blank),
        }
    )
    out["source_priority"] = 10
    out["preferred_registry_role"] = np.where(
        out["source_sample_role"].eq("locked_test"),
        "development_historical_v1",
        "development_v1",
    )
    return out


def rows_from_v2_development() -> pd.DataFrame:
    v2 = read_csv(V2_VALIDATED)
    v2 = v2.loc[v2["sample_role"].eq("development_v2")].copy()
    out = pd.DataFrame(
        {
            "comment_id": v2["comment_id"].map(norm_blank),
            "comment_text_original": v2["comment_text_original"].map(norm_blank),
            "video_id": v2["video_id"].map(norm_blank),
            "brand_or_video_context": v2["brand_or_video_context"].map(norm_blank),
            "final_sentiment_label": v2["final_sentiment_label"].map(canonical_label),
            "source_version": "human_v2",
            "source_file": V2_VALIDATED.relative_to(ROOT).as_posix(),
            "source_sample_role": v2["sample_role"].map(norm_blank),
            "sentiment_target": v2.get("final_sentiment_target", "").map(norm_blank),
            "complaint_scope": v2.get("final_complaint_scope", "").map(norm_blank),
            "annotation_notes": v2.get("adjudication_notes", "").map(norm_blank),
            "annotator_1_label": v2.get("annotator_1_label", "").map(norm_blank),
            "annotator_2_label": v2.get("annotator_2_label", "").map(norm_blank),
        }
    )
    out["source_priority"] = 20
    out["preferred_registry_role"] = "development_v2"
    return out


def rows_from_v2_locked_final() -> pd.DataFrame:
    locked = read_csv(V2_LOCKED_FINAL)
    out = pd.DataFrame(
        {
            "comment_id": locked["comment_id"].map(norm_blank),
            "comment_text_original": locked["comment_text_original"].map(norm_blank),
            "video_id": locked["video_id"].map(norm_blank),
            "brand_or_video_context": locked["brand_or_video_context"].map(norm_blank),
            "final_sentiment_label": locked["final_sentiment_label"].map(canonical_label),
            "source_version": "human_v2_locked_observational_final",
            "source_file": V2_LOCKED_FINAL.relative_to(ROOT).as_posix(),
            "source_sample_role": locked["sample_role"].map(norm_blank),
            "sentiment_target": locked.get("final_sentiment_target", "").map(norm_blank),
            "complaint_scope": locked.get("final_complaint_scope", "").map(norm_blank),
            "annotation_notes": locked.get("adjudication_notes", "").map(norm_blank),
            "annotator_1_label": locked.get("annotator_1_label", "").map(norm_blank),
            "annotator_2_label": locked.get("annotator_2_label", "").map(norm_blank),
        }
    )
    out["source_priority"] = 100
    out["preferred_registry_role"] = "final_test_v3"
    return out


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
            if abs(len(left_text) - len(right_text)) > max(10, 0.20 * max(len(left_text), len(right_text))):
                continue
            score = SequenceMatcher(a=left_text, b=right_text, autojunk=False).ratio()
            if score >= 0.96:
                uf.union(left["comment_id"], right["comment_id"])

    roots = {comment_id: uf.find(comment_id) for comment_id in ids}
    root_to_cluster = {root: f"text_cluster_{idx:04d}" for idx, root in enumerate(sorted(set(roots.values())), 1)}
    out["text_cluster_id"] = out["comment_id"].map(lambda x: root_to_cluster[roots[x]])
    out["text_cluster_size"] = out.groupby("text_cluster_id")["comment_id"].transform("nunique")
    return out


def assign_roles(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["is_injected"] = out["comment_id"].map(is_injected_id)
    out["is_observational"] = ~out["is_injected"]
    out["is_evaluable_three_class"] = out["final_sentiment_label"].isin(LABELS)

    final_test_clusters = set(
        out.loc[
            out["preferred_registry_role"].eq("final_test_v3")
            & out["is_observational"]
            & out["is_evaluable_three_class"],
            "text_cluster_id",
        ]
    )

    out["registry_role"] = out["preferred_registry_role"]
    out.loc[out["is_injected"], "registry_role"] = "excluded_injected"
    out.loc[~out["is_evaluable_three_class"] & ~out["is_injected"], "registry_role"] = "excluded_non_evaluable"
    dev_mask = out["preferred_registry_role"].str.startswith("development", na=False)
    text_overlap = dev_mask & out["text_cluster_id"].isin(final_test_clusters) & out["is_evaluable_three_class"]
    out.loc[text_overlap, "registry_role"] = "excluded_final_test_text_cluster_overlap"

    out["split_family"] = np.select(
        [
            out["registry_role"].eq("final_test_v3") & out["is_evaluable_three_class"],
            out["registry_role"].str.startswith("development", na=False) & out["is_evaluable_three_class"],
        ],
        ["final_test", "development"],
        default="excluded",
    )

    duplicate_group = out["text_cluster_size"].astype(int).gt(1)
    out["cv_group_id"] = np.where(
        duplicate_group,
        "dup:" + out["text_cluster_id"],
        "video:" + out["video_id"].replace("", "missing_video"),
    )
    out["fold_v3"] = ""
    dev = out.loc[out["split_family"].eq("development")].copy()
    if not dev.empty:
        y = dev["final_sentiment_label"].to_numpy()
        groups = dev["cv_group_id"].to_numpy()
        splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_SEED)
        for fold, (_, val_idx) in enumerate(splitter.split(dev, y, groups), start=1):
            out.loc[dev.iloc[val_idx].index, "fold_v3"] = str(fold)
    return out


def deduplicate_sources(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for comment_id, group in raw.groupby("comment_id", dropna=False):
        labels = sorted(set(group["final_sentiment_label"].map(canonical_label)))
        roles = sorted(set(group["preferred_registry_role"]))
        files = sorted(set(group["source_file"]))
        status = "PASS"
        notes = ""
        if comment_id == "":
            status = "FAIL"
            notes = "Blank comment_id."
        elif len(labels) > 1:
            status = "CONFLICT_REVIEWED"
            notes = "Multiple labels found across sources; highest-priority adjudicated source retained."
        rows.append(
            {
                "audit_type": "comment_id_label_conflict",
                "comment_id": comment_id,
                "labels_observed": ";".join(labels),
                "roles_observed": ";".join(roles),
                "source_files": ";".join(files),
                "n_source_rows": len(group),
                "status": status,
                "notes": notes,
            }
        )

    ordered = raw.sort_values(["comment_id", "source_priority"], ascending=[True, False]).copy()
    deduped = ordered.drop_duplicates("comment_id", keep="first").reset_index(drop=True)
    audit = pd.DataFrame(rows)
    return deduped, audit


def split_audit(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for cluster_id, group in frame.groupby("text_cluster_id"):
        split_families = sorted(set(group["split_family"]))
        folds = sorted(set(group.loc[group["fold_v3"].ne(""), "fold_v3"]))
        if len([x for x in split_families if x in {"development", "final_test"}]) > 1:
            rows.append(
                {
                    "audit_type": "text_cluster_split_overlap",
                    "comment_id": "",
                    "labels_observed": ";".join(sorted(set(group["final_sentiment_label"]))),
                    "roles_observed": ";".join(sorted(set(group["registry_role"]))),
                    "source_files": ";".join(sorted(set(group["source_file"]))),
                    "n_source_rows": len(group),
                    "status": "FAIL",
                    "notes": f"{cluster_id} appears in both development and final_test.",
                }
            )
        if len(folds) > 1:
            rows.append(
                {
                    "audit_type": "text_cluster_fold_overlap",
                    "comment_id": "",
                    "labels_observed": ";".join(sorted(set(group["final_sentiment_label"]))),
                    "roles_observed": ";".join(sorted(set(group["registry_role"]))),
                    "source_files": ";".join(sorted(set(group["source_file"]))),
                    "n_source_rows": len(group),
                    "status": "FAIL",
                    "notes": f"{cluster_id} appears in multiple development folds: {';'.join(folds)}.",
                }
            )
    if not rows:
        rows.append(
            {
                "audit_type": "split_group_integrity",
                "comment_id": "",
                "labels_observed": "",
                "roles_observed": "",
                "source_files": "",
                "n_source_rows": 0,
                "status": "PASS",
                "notes": "No exact or near-identical text cluster crosses development/final-test or multiple development folds.",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    HUMAN_V3_DIR.mkdir(parents=True, exist_ok=True)

    raw = pd.concat([rows_from_v1(), rows_from_v2_development(), rows_from_v2_locked_final()], ignore_index=True)
    raw = raw.loc[raw["comment_id"].ne("")].copy()
    deduped, conflict = deduplicate_sources(raw)

    context = dataset_context()
    if not context.empty:
        deduped = deduped.merge(context, on="comment_id", how="left", suffixes=("", "_dataset"))
        deduped["product_category"] = deduped.get("product_category", "").map(norm_blank)
        deduped["comment_type"] = deduped.get("comment_type", "").map(norm_blank)
        deduped["timestamp"] = deduped.get("timestamp", "").map(norm_blank)
        deduped["username"] = deduped.get("username", "").map(norm_blank)
        dataset_text = deduped.get("text", pd.Series("", index=deduped.index)).map(norm_blank)
        deduped["dataset_text_matches"] = (
            deduped["comment_text_original"].map(normalize_for_group).eq(dataset_text.map(normalize_for_group))
        )
    else:
        deduped["product_category"] = ""
        deduped["comment_type"] = ""
        deduped["timestamp"] = ""
        deduped["username"] = ""
        deduped["dataset_text_matches"] = False

    deduped = add_text_clusters(deduped)
    deduped = assign_roles(deduped)

    ordered_columns = [
        "comment_id",
        "registry_role",
        "split_family",
        "fold_v3",
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
        "source_version",
        "source_file",
        "source_sample_role",
        "sentiment_target",
        "complaint_scope",
        "annotation_notes",
        "annotator_1_label",
        "annotator_2_label",
        "text_cluster_id",
        "text_cluster_size",
        "exact_text_hash",
        "cv_group_id",
        "dataset_text_matches",
    ]
    for col in ordered_columns:
        if col not in deduped.columns:
            deduped[col] = ""
    registry = deduped[ordered_columns].sort_values(["split_family", "registry_role", "fold_v3", "comment_id"])
    registry.to_csv(OUT_REGISTRY, index=False, encoding="utf-8-sig")

    audit = pd.concat([conflict, split_audit(registry)], ignore_index=True)
    audit.to_csv(OUT_CONFLICT_AUDIT, index=False, encoding="utf-8-sig")

    split_cols = [
        "comment_id",
        "split_family",
        "registry_role",
        "fold_v3",
        "final_sentiment_label",
        "video_id",
        "text_cluster_id",
        "text_cluster_size",
        "cv_group_id",
        "source_version",
        "source_sample_role",
    ]
    registry[split_cols].to_csv(OUT_SPLIT_MANIFEST, index=False, encoding="utf-8-sig")

    summary_rows = []
    for (split_family, role, label), group in registry.groupby(["split_family", "registry_role", "final_sentiment_label"], dropna=False):
        summary_rows.append(
            {
                "split_family": split_family,
                "registry_role": role,
                "label": label,
                "count": int(len(group)),
                "evaluable_three_class": bool(label in LABELS),
            }
        )
    manifest = {
        "status": "HUMAN_LABEL_REGISTRY_V3_BUILT",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "rules": {
            "ground_truth": "human annotation/adjudication only",
            "classification_labels": LABELS,
            "non_evaluable_labels": sorted(NON_EVALUABLE),
            "excluded_id_prefix": "INJ",
            "final_test_source": V2_LOCKED_FINAL.relative_to(ROOT).as_posix(),
            "test_usage": "final evaluation only; not used for tuning/model selection/threshold/preprocessing",
            "duplicate_policy": "exact and near-identical text clusters are not allowed to cross development/final-test or multiple development folds",
            "positive_shift_policy": "no post-hoc Positive shifting is applied",
        },
        "input_checksums_sha256": {
            V1_VALIDATED.relative_to(ROOT).as_posix(): sha256_file(V1_VALIDATED),
            V2_VALIDATED.relative_to(ROOT).as_posix(): sha256_file(V2_VALIDATED),
            V2_LOCKED_FINAL.relative_to(ROOT).as_posix(): sha256_file(V2_LOCKED_FINAL),
        },
        "counts": {
            "registry_rows": int(len(registry)),
            "development_evaluable_rows": int(registry["split_family"].eq("development").sum()),
            "final_test_evaluable_rows": int(registry["split_family"].eq("final_test").sum()),
            "excluded_rows": int(registry["split_family"].eq("excluded").sum()),
            "text_clusters": int(registry["text_cluster_id"].nunique()),
            "development_cv_groups": int(registry.loc[registry["split_family"].eq("development"), "cv_group_id"].nunique()),
        },
        "label_distribution": summary_rows,
        "outputs": {
            "registry": OUT_REGISTRY.relative_to(ROOT).as_posix(),
            "conflict_audit": OUT_CONFLICT_AUDIT.relative_to(ROOT).as_posix(),
            "split_manifest": OUT_SPLIT_MANIFEST.relative_to(ROOT).as_posix(),
        },
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(json.dumps(manifest["counts"], indent=2))
    failing = audit["status"].eq("FAIL")
    if failing.any():
        raise AssertionError("V3 registry split integrity audit failed.")


if __name__ == "__main__":
    main()
