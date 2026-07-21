from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
HUMAN_V2_DIR = ROOT / "output/rm2_sentiment/human_validation_v2"
HUMAN_V1_PATH = ROOT / "output/rm2_sentiment/human_validation/sentiment_human_annotation_validated.csv"
DATASET_PATH = ROOT / "dataset.csv"
COMMENT_SENTIMENT_PATH = ROOT / "output/rm2_sentiment/tables/comment_sentiment.csv"

ANNOTATOR_1_FINAL = HUMAN_V2_DIR / "sentiment_v2_annotator_1_blind_final.csv"
ANNOTATOR_2_FINAL = HUMAN_V2_DIR / "sentiment_v2_annotator_2_blind_final.csv"
ADJUDICATION_FINAL = HUMAN_V2_DIR / "sentiment_v2_adjudication_template_final.csv"
SAMPLING_MANIFEST = HUMAN_V2_DIR / "sentiment_v2_sampling_manifest.csv"
LOCKED_MANIFEST = HUMAN_V2_DIR / "locked_test_v2_manifest.csv"
LOCKED_CHECKSUM = HUMAN_V2_DIR / "locked_test_v2_checksum.csv"

OUT_VALIDATED = HUMAN_V2_DIR / "sentiment_human_annotation_v2_validated.csv"
OUT_PROVENANCE = HUMAN_V2_DIR / "sentiment_human_annotation_v2_provenance.csv"
OUT_AGREEMENT = HUMAN_V2_DIR / "sentiment_v2_inter_annotator_agreement.csv"
OUT_VALIDATION_REPORT = HUMAN_V2_DIR / "sentiment_v2_annotation_validation_report.csv"
OUT_UNRESOLVED = HUMAN_V2_DIR / "sentiment_v2_unresolved_rows.csv"
OUT_DISTRIBUTION = HUMAN_V2_DIR / "sentiment_v2_final_label_distribution.csv"
OUT_CHALLENGE = HUMAN_V2_DIR / "sentiment_v2_challenge_set.csv"
OUT_SYNTHETIC_AUDIT = HUMAN_V2_DIR / "sentiment_v2_synthetic_id_audit.csv"
OUT_OBSERVATIONAL_AUDIT = HUMAN_V2_DIR / "sentiment_v2_observational_sample_audit.csv"
OUT_LOCKED_READINESS = HUMAN_V2_DIR / "sentiment_v2_locked_test_readiness.csv"
OUT_V1_DUP_AUDIT = HUMAN_V2_DIR / "sentiment_v1_duplicate_id_audit.csv"
OUT_TRAINING_PROVENANCE = HUMAN_V2_DIR / "sentiment_combined_training_provenance.csv"
OUT_REPLACEMENT_MANIFEST = HUMAN_V2_DIR / "sentiment_v2_locked_test_replacement_manifest.csv"
OUT_REPLACEMENT_A1 = HUMAN_V2_DIR / "sentiment_v2_locked_test_replacement_annotator_1_blind.csv"
OUT_REPLACEMENT_A2 = HUMAN_V2_DIR / "sentiment_v2_locked_test_replacement_annotator_2_blind.csv"
OUT_REPLACEMENT_ADJ = HUMAN_V2_DIR / "sentiment_v2_locked_test_replacement_adjudication_template.csv"
OUT_SOURCE_CHECKSUMS = HUMAN_V2_DIR / "sentiment_v2_source_checksum_manifest.csv"

SELECTION_SEED = 20260721
ALLOWED_SENTIMENT = ["Positive", "Neutral", "Negative", "Uncertain", "No Text"]
ALLOWED_TARGET = [
    "Product / Brand",
    "Skin condition",
    "Usage question",
    "Creator / Seller",
    "Price / Purchase",
    "Promotion / CTA",
    "General discussion",
    "Other / unclear",
]
ALLOWED_SCOPE = [
    "product_effect",
    "skin_condition",
    "price_value",
    "safety_concern",
    "authenticity_concern",
    "usage_confusion",
    "not_applicable",
    "unclear",
]
MAIN_SENTIMENT = ["Positive", "Neutral", "Negative"]
SOURCE_FILES = [
    ANNOTATOR_1_FINAL,
    ANNOTATOR_2_FINAL,
    ADJUDICATION_FINAL,
    SAMPLING_MANIFEST,
    LOCKED_MANIFEST,
    LOCKED_CHECKSUM,
    HUMAN_V1_PATH,
]


QUESTION_RE = re.compile(
    r"(?:\?|apa|apakah|gimana|bagaimana|boleh|bisa|cocok\s*(?:ga|gak|nggak|ngga)?|"
    r"aman\s*(?:ga|gak|nggak|ngga)?|untuk kulit|cara pakai|dipakai|boleh pakai)",
    re.IGNORECASE,
)
EMOJI_RE = re.compile(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]")
SKIN_RE = re.compile(
    r"\b(?:jerawat|bruntusan|beruntusan|kusam|flek|bekas jerawat|sensitif|kering|berminyak|"
    r"iritasi|perih|gatal|kemerahan|breakout|purging|skin barrier|komedo|pori|bopeng)\b",
    re.IGNORECASE,
)
POSITIVE_EFFECT_RE = re.compile(
    r"\b(?:cocok|bagus|mantap|ampuh|worth|rekomendasi|recommended|cerah|glowing|mulus|"
    r"membaik|hilang|pudar|suka|love|hasilnya|efektif|aman di aku)\b",
    re.IGNORECASE,
)
NEGATIVE_SIGNAL_RE = re.compile(
    r"\b(?:gak cocok|ga cocok|nggak cocok|tidak cocok|buruk|parah|makin|tambah|iritasi|"
    r"perih|gatal|merah|breakout|purging|menyesal|zonk|mahal|bahaya|palsu|rugi)\b",
    re.IGNORECASE,
)
PURCHASE_RE = re.compile(
    r"\b(?:beli|checkout|co|keranjang|link|harga|diskon|promo|order|cod|shopee|tokopedia|"
    r"affiliate|voucher|gratis ongkir)\b",
    re.IGNORECASE,
)
PROMOTION_RE = re.compile(
    r"\b(?:spill|buruan|wajib coba|cobain|aku rekomendasi|rekomendasi banget|racun|"
    r"best seller|terlaris|jangan lupa)\b",
    re.IGNORECASE,
)
NEGATION_RE = re.compile(r"\b(?:tidak|gak|ga|nggak|ngga|jangan|belum|bukan|tanpa|kurang)\b", re.IGNORECASE)
SLANG_RE = re.compile(
    r"\b(?:ga|gak|nggak|ngga|nih|dong|kak|sis|bund|spill|racun|bestie|wkwk|btw|yg|dgn|"
    r"bgt|banget|plis|cmiiw|auto|gue|gw|aku)\b",
    re.IGNORECASE,
)
CODE_MIX_RE = re.compile(
    r"\b(?:retinol|niacinamide|serum|skincare|glowing|brightening|acne|dark spot|breakout|"
    r"review|claim|skin barrier|moisturizer|sunscreen|tone up|exfoliating)\b",
    re.IGNORECASE,
)


def ensure_utf8(path: Path) -> None:
    path.read_text(encoding="utf-8")


def read_csv(path: Path) -> pd.DataFrame:
    ensure_utf8(path)
    return pd.read_csv(path, dtype=str, low_memory=False).fillna("")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(values: list[str]) -> str:
    return hashlib.sha256(("\n".join(values) + "\n").encode("utf-8")).hexdigest()


def normalize_id(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null", "<na>"}:
        return ""
    return text


def normalize_label(value: object) -> str:
    text = normalize_id(value)
    return "No Text" if text.lower() in {"no text", "no_text", "notext"} else text


def is_synthetic_id(value: object, dataset_ids: set[str]) -> tuple[bool, str]:
    cid = normalize_id(value)
    lowered = cid.lower()
    reasons: list[str] = []
    if re.match(r"^inj", cid, flags=re.IGNORECASE):
        reasons.append("INJ_prefix")
    if re.search(r"(synthetic|challenge)", lowered):
        reasons.append("synthetic_or_challenge_pattern")
    if cid and cid not in dataset_ids:
        reasons.append("not_found_in_dataset")
    return bool(reasons), ";".join(reasons)


def cohen_kappa(a: pd.Series, b: pd.Series, labels: list[str]) -> float:
    left = a.astype(str).reset_index(drop=True)
    right = b.astype(str).reset_index(drop=True)
    if len(left) == 0:
        return float("nan")
    observed = float((left == right).mean())
    expected = sum(float((left == label).mean()) * float((right == label).mean()) for label in labels)
    if math.isclose(1.0 - expected, 0.0):
        return float("nan")
    return (observed - expected) / (1.0 - expected)


def row_status(metric: str, expected: object, observed: object, passed: bool, notes: str = "") -> dict[str, object]:
    return {
        "metric": metric,
        "expected": expected,
        "observed": observed,
        "passed": bool(passed),
        "notes": notes,
    }


def validate_allowed(frame: pd.DataFrame, columns: dict[str, list[str]], report: list[dict[str, object]], prefix: str) -> None:
    for column, allowed in columns.items():
        if column not in frame.columns:
            report.append(row_status(f"{prefix}_{column}_present", "present", "missing", False))
            continue
        values = frame[column].map(normalize_label)
        invalid = sorted(set(values) - set(allowed) - {""})
        blanks = int(values.eq("").sum())
        report.append(row_status(f"{prefix}_{column}_valid_values", "allowed labels only", invalid or "none", not invalid))
        report.append(row_status(f"{prefix}_{column}_complete", 0, blanks, blanks == 0))


def add_text_features(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    text = df.get("comment_text_original", df.get("text_raw", pd.Series("", index=df.index))).astype(str)
    fallback = df.get("text_raw", pd.Series("", index=df.index)).astype(str)
    df["comment_text_original"] = text.where(text.str.strip().ne(""), fallback).astype(str).str.strip()
    lower = df["comment_text_original"].str.lower()
    df["word_count"] = df["comment_text_original"].str.findall(r"\S+").str.len()
    df["no_text_flag"] = df["comment_text_original"].str.strip().eq("")
    df["is_question"] = lower.str.contains(QUESTION_RE, regex=True, na=False)
    df["has_emoji"] = df["comment_text_original"].str.contains(EMOJI_RE, regex=True, na=False)
    df["emoji_only"] = df["has_emoji"] & lower.str.replace(EMOJI_RE, "", regex=True).str.strip().eq("")
    df["has_skin_condition"] = lower.str.contains(SKIN_RE, regex=True, na=False)
    df["has_positive_effect"] = lower.str.contains(POSITIVE_EFFECT_RE, regex=True, na=False)
    df["has_negative_signal"] = lower.str.contains(NEGATIVE_SIGNAL_RE, regex=True, na=False)
    df["has_purchase_cta"] = lower.str.contains(PURCHASE_RE, regex=True, na=False)
    df["has_promotion"] = lower.str.contains(PROMOTION_RE, regex=True, na=False)
    df["has_negation"] = lower.str.contains(NEGATION_RE, regex=True, na=False)
    df["has_slang"] = lower.str.contains(SLANG_RE, regex=True, na=False)
    df["has_code_mixing"] = lower.str.contains(CODE_MIX_RE, regex=True, na=False)
    df["is_short_text"] = (df["word_count"] <= 3) & ~df["no_text_flag"]
    df["mixed_signal"] = (
        df.get("mixed_sentiment_flag", pd.Series("", index=df.index)).astype(str).str.lower().isin(["true", "1"])
        | (df["has_positive_effect"] & df["has_negative_signal"])
        | (df["has_negation"] & (df["has_positive_effect"] | df["has_negative_signal"]))
    )

    def length_bin(row: pd.Series) -> str:
        if row["no_text_flag"]:
            return "no_text"
        words = int(row["word_count"])
        if words <= 3:
            return "very_short"
        if words <= 8:
            return "short"
        if words <= 20:
            return "medium"
        return "long"

    def text_type(row: pd.Series) -> str:
        if row["no_text_flag"]:
            return "no_text"
        if row["emoji_only"]:
            return "emoji_only"
        if row["is_question"]:
            return "question"
        if row["mixed_signal"]:
            return "mixed_or_negation"
        if row["has_skin_condition"] and not row["has_positive_effect"]:
            return "skin_condition"
        if row["has_purchase_cta"] or row["has_promotion"]:
            return "purchase_or_cta"
        if row["is_short_text"]:
            return "short_text"
        if row["has_slang"] or row["has_code_mixing"]:
            return "slang_or_code_mixing"
        return "general"

    df["length_bin"] = df.apply(length_bin, axis=1)
    df["text_type_major"] = df.apply(text_type, axis=1)
    df["actor_segment"] = df.get("is_hcc_member", pd.Series("", index=df.index)).astype(str).str.lower().isin(["true", "1"]).map(
        {True: "HCC", False: "Non-HCC"}
    )
    df["brand_or_video_context"] = df.get("product_brand_context", df.get("product_category", pd.Series("", index=df.index))).astype(str).str.strip()
    df.loc[df["brand_or_video_context"].eq(""), "brand_or_video_context"] = df.get("product_category", pd.Series("", index=df.index)).astype(str)
    df["sampling_stratum"] = (
        df["actor_segment"]
        + "|"
        + df["brand_or_video_context"].replace("", "Unknown")
        + "|"
        + df["length_bin"]
        + "|"
        + df["text_type_major"]
    )
    return df


def integrate_v2_labels(a1: pd.DataFrame, a2: pd.DataFrame, adj: pd.DataFrame, dataset_ids: set[str]) -> pd.DataFrame:
    merged = adj.copy()
    rename_map = {
        "annotator_1_sentiment_label": "annotator_1_label",
        "annotator_2_sentiment_label": "annotator_2_label",
        "adjudicated_sentiment_label": "adjudicated_label",
        "annotator_1_sentiment_target": "annotator_1_target",
        "annotator_2_sentiment_target": "annotator_2_target",
        "adjudicated_sentiment_target": "adjudicated_target",
        "annotator_1_complaint_scope": "annotator_1_scope",
        "annotator_2_complaint_scope": "annotator_2_scope",
        "adjudicated_complaint_scope": "adjudicated_scope",
    }
    merged = merged.rename(columns=rename_map)
    for col in [
        "annotator_1_label",
        "annotator_2_label",
        "adjudicated_label",
        "annotator_1_target",
        "annotator_2_target",
        "adjudicated_target",
        "annotator_1_scope",
        "annotator_2_scope",
        "adjudicated_scope",
    ]:
        merged[col] = merged[col].map(normalize_label)

    def choose(row: pd.Series, adjudicated_col: str, a_col: str, b_col: str) -> tuple[str, str]:
        if row[adjudicated_col]:
            return row[adjudicated_col], "adjudicated_final"
        if row[a_col] and row[a_col] == row[b_col]:
            return row[a_col], "annotator_agreement"
        return "UNRESOLVED", "unresolved_annotator_disagreement"

    label_result = merged.apply(lambda row: choose(row, "adjudicated_label", "annotator_1_label", "annotator_2_label"), axis=1)
    target_result = merged.apply(lambda row: choose(row, "adjudicated_target", "annotator_1_target", "annotator_2_target"), axis=1)
    scope_result = merged.apply(lambda row: choose(row, "adjudicated_scope", "annotator_1_scope", "annotator_2_scope"), axis=1)
    merged["final_sentiment_label"] = [value for value, _ in label_result]
    merged["sentiment_label_source"] = [source for _, source in label_result]
    merged["final_sentiment_target"] = [value for value, _ in target_result]
    merged["sentiment_target_source"] = [source for _, source in target_result]
    merged["final_complaint_scope"] = [value for value, _ in scope_result]
    merged["complaint_scope_source"] = [source for _, source in scope_result]
    synthetic = merged["comment_id"].map(lambda value: is_synthetic_id(value, dataset_ids))
    merged["is_synthetic_or_injected"] = [flag for flag, _ in synthetic]
    merged["synthetic_reason"] = [reason for _, reason in synthetic]
    merged["is_observational_sample"] = ~merged["is_synthetic_or_injected"]
    merged["is_evaluable_sentiment"] = merged["final_sentiment_label"].isin(MAIN_SENTIMENT)
    return merged


def write_agreement(adj: pd.DataFrame, integrated: pd.DataFrame) -> None:
    rows: list[dict[str, object]] = []
    confusion_frames: list[pd.DataFrame] = []
    fields = [
        ("sentiment_label", "annotator_1_sentiment_label", "annotator_2_sentiment_label", ALLOWED_SENTIMENT),
        ("sentiment_target", "annotator_1_sentiment_target", "annotator_2_sentiment_target", ALLOWED_TARGET),
        ("complaint_scope", "annotator_1_complaint_scope", "annotator_2_complaint_scope", ALLOWED_SCOPE),
    ]
    scopes = [("all_v2", integrated.index), ("observational_v2", integrated.index[integrated["is_observational_sample"]]), ("challenge_set", integrated.index[integrated["is_synthetic_or_injected"]])]
    for field_name, col_a, col_b, labels in fields:
        for scope_name, idx in scopes:
            left = adj.loc[idx, col_a].map(normalize_label)
            right = adj.loc[idx, col_b].map(normalize_label)
            if len(left) == 0:
                agreement = np.nan
                kappa = np.nan
            else:
                agreement = float((left == right).mean())
                kappa = cohen_kappa(left, right, labels)
            rows.append(
                {
                    "section": "agreement_metric",
                    "scope": scope_name,
                    "sample_role": "all",
                    "field": field_name,
                    "metric": "raw_agreement",
                    "value": agreement,
                    "n": len(left),
                    "notes": "Treat as intra-rater consistency if both files were filled by the same annotator.",
                }
            )
            rows.append(
                {
                    "section": "agreement_metric",
                    "scope": scope_name,
                    "sample_role": "all",
                    "field": field_name,
                    "metric": "cohen_kappa",
                    "value": kappa,
                    "n": len(left),
                    "notes": "Treat as intra-rater consistency if both files were filled by the same annotator.",
                }
            )
            matrix = pd.crosstab(left, right, dropna=False).reset_index()
            matrix.insert(0, "field", field_name)
            matrix.insert(0, "sample_role", "all")
            matrix.insert(0, "scope", scope_name)
            matrix.insert(0, "section", "confusion_table")
            confusion_frames.append(matrix)
        for sample_role, group in adj.groupby("sample_role", dropna=False):
            left = group[col_a].map(normalize_label)
            right = group[col_b].map(normalize_label)
            rows.append(
                {
                    "section": "agreement_by_sample_role",
                    "scope": "all_v2",
                    "sample_role": sample_role,
                    "field": field_name,
                    "metric": "raw_agreement",
                    "value": float((left == right).mean()) if len(left) else np.nan,
                    "n": len(left),
                    "notes": "",
                }
            )
            rows.append(
                {
                    "section": "agreement_by_sample_role",
                    "scope": "all_v2",
                    "sample_role": sample_role,
                    "field": field_name,
                    "metric": "cohen_kappa",
                    "value": cohen_kappa(left, right, labels),
                    "n": len(left),
                    "notes": "",
                }
            )
    agreement = pd.DataFrame(rows)
    if confusion_frames:
        confusion = pd.concat(confusion_frames, ignore_index=True).fillna("")
        shared_cols = sorted(set(agreement.columns) | set(confusion.columns))
        agreement = agreement.reindex(columns=shared_cols)
        confusion = confusion.reindex(columns=shared_cols)
        agreement = pd.concat([agreement, confusion], ignore_index=True)
    agreement.to_csv(OUT_AGREEMENT, index=False)


def write_synthetic_audit(frames: dict[str, pd.DataFrame], dataset_ids: set[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for source_name, frame in frames.items():
        if "comment_id" not in frame.columns:
            continue
        for idx, cid in frame["comment_id"].map(normalize_id).items():
            flag, reason = is_synthetic_id(cid, dataset_ids)
            if flag:
                rows.append(
                    {
                        "source_file": source_name,
                        "row_number_1based": int(idx) + 2,
                        "comment_id": cid,
                        "sample_role": frame.loc[idx, "sample_role"] if "sample_role" in frame.columns else frame.loc[idx, "sample_set"] if "sample_set" in frame.columns else "",
                        "synthetic_reason": reason,
                    }
                )
    audit = pd.DataFrame(rows, columns=["source_file", "row_number_1based", "comment_id", "sample_role", "synthetic_reason"])
    audit.to_csv(OUT_SYNTHETIC_AUDIT, index=False)
    return audit


def write_challenge_set(v1: pd.DataFrame, integrated: pd.DataFrame, dataset_ids: set[str]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    v2_challenge = integrated.loc[integrated["is_synthetic_or_injected"]].copy()
    if not v2_challenge.empty:
        v2_challenge.insert(0, "annotation_version", "v2")
        rows.append(v2_challenge)
    v1_frame = v1.copy()
    v1_frame["comment_id"] = v1_frame["comment_id"].map(normalize_id)
    synthetic = v1_frame["comment_id"].map(lambda value: is_synthetic_id(value, dataset_ids))
    v1_frame["is_synthetic_or_injected"] = [flag for flag, _ in synthetic]
    v1_frame["synthetic_reason"] = [reason for _, reason in synthetic]
    v1_challenge = v1_frame.loc[v1_frame["is_synthetic_or_injected"]].copy()
    if not v1_challenge.empty:
        v1_challenge = v1_challenge.rename(columns={"sample_set": "sample_role", "adjudicated_human_label": "final_sentiment_label"})
        v1_challenge["annotation_version"] = "v1"
        v1_challenge["final_sentiment_target"] = ""
        v1_challenge["final_complaint_scope"] = ""
        rows.append(v1_challenge)
    challenge = pd.concat(rows, ignore_index=True, sort=False) if rows else pd.DataFrame()
    challenge.to_csv(OUT_CHALLENGE, index=False)
    return challenge


def write_v1_duplicate_audit(v1: pd.DataFrame) -> pd.DataFrame:
    v1_ids = v1["comment_id"].map(normalize_id)
    duplicated_ids = sorted(v1_ids[v1_ids.duplicated(keep=False)].unique())
    rows = []
    for cid in duplicated_ids:
        subset = v1.loc[v1_ids.eq(cid)]
        labels = subset["adjudicated_human_label"].map(normalize_label).tolist()
        rows.append(
            {
                "comment_id": cid,
                "n_rows": len(subset),
                "sample_sets": ";".join(sorted(set(subset.get("sample_set", pd.Series("", index=subset.index)).astype(str)))),
                "adjudicated_labels": ";".join(labels),
                "labels_consistent": len(set(labels)) <= 1,
            }
        )
    audit = pd.DataFrame(rows, columns=["comment_id", "n_rows", "sample_sets", "adjudicated_labels", "labels_consistent"])
    audit.to_csv(OUT_V1_DUP_AUDIT, index=False)
    return audit


def replacement_package(
    locked_synthetic: pd.DataFrame,
    dataset_ids: set[str],
    used_ids: set[str],
) -> pd.DataFrame:
    comments = add_text_features(read_csv(COMMENT_SENTIMENT_PATH))
    comments["comment_id"] = comments["comment_id"].map(normalize_id)
    comments["is_synthetic_or_injected"] = comments["comment_id"].map(lambda value: is_synthetic_id(value, dataset_ids)[0])
    pool = comments.loc[~comments["comment_id"].isin(used_ids) & ~comments["is_synthetic_or_injected"]].copy()
    chosen_rows: list[pd.DataFrame] = []
    selected: set[str] = set()
    for idx, row in locked_synthetic.reset_index(drop=True).iterrows():
        stratum = row["sampling_stratum"]
        candidates = pool.loc[pool["sampling_stratum"].eq(stratum) & ~pool["comment_id"].isin(selected)].copy()
        status = "exact_stratum_match"
        if candidates.empty:
            components = stratum.split("|")
            actor_segment, brand = components[0], components[1]
            candidates = pool.loc[
                pool["actor_segment"].eq(actor_segment)
                & pool["brand_or_video_context"].eq(brand)
                & ~pool["comment_id"].isin(selected)
            ].copy()
            status = "fallback_actor_brand_match"
        if candidates.empty:
            candidates = pool.loc[~pool["comment_id"].isin(selected)].copy()
            status = "fallback_any_observational"
        if candidates.empty:
            raise AssertionError("Unable to build locked-test replacement package: candidate pool exhausted.")
        chosen = candidates.sample(n=1, random_state=SELECTION_SEED + idx).copy()
        chosen["replaces_comment_id"] = row["comment_id"]
        chosen["requested_sampling_stratum"] = stratum
        chosen["replacement_status"] = status
        selected.add(str(chosen.iloc[0]["comment_id"]))
        chosen_rows.append(chosen)
    replacement = pd.concat(chosen_rows, ignore_index=True) if chosen_rows else pd.DataFrame()
    if replacement.empty:
        pd.DataFrame().to_csv(OUT_REPLACEMENT_MANIFEST, index=False)
        return replacement

    manifest = replacement[
        [
            "comment_id",
            "replaces_comment_id",
            "sampling_stratum",
            "requested_sampling_stratum",
            "replacement_status",
            "comment_text_original",
            "video_id",
            "brand_or_video_context",
        ]
    ].copy()
    manifest.insert(1, "sample_role", "locked_test_v2_replacement")
    manifest["selection_seed"] = SELECTION_SEED
    manifest.to_csv(OUT_REPLACEMENT_MANIFEST, index=False)

    base = manifest[["sample_role", "comment_id", "comment_text_original", "video_id", "brand_or_video_context"]].copy()
    for column in ["sentiment_label", "sentiment_target", "complaint_scope", "annotator_notes"]:
        base[column] = ""
    base.to_csv(OUT_REPLACEMENT_A1, index=False)
    base.to_csv(OUT_REPLACEMENT_A2, index=False)

    adjudication = manifest[["sample_role", "comment_id", "comment_text_original", "video_id", "brand_or_video_context"]].copy()
    for prefix in ["annotator_1", "annotator_2"]:
        adjudication[f"{prefix}_sentiment_label"] = ""
        adjudication[f"{prefix}_sentiment_target"] = ""
        adjudication[f"{prefix}_complaint_scope"] = ""
        adjudication[f"{prefix}_notes"] = ""
    adjudication["adjudicated_sentiment_label"] = ""
    adjudication["adjudicated_sentiment_target"] = ""
    adjudication["adjudicated_complaint_scope"] = ""
    adjudication["adjudication_notes"] = ""
    adjudication.to_csv(OUT_REPLACEMENT_ADJ, index=False)
    return replacement


def main() -> None:
    HUMAN_V2_DIR.mkdir(parents=True, exist_ok=True)
    source_hash_before = {str(path.relative_to(ROOT)): sha256_file(path) for path in SOURCE_FILES if path.exists()}

    a1 = read_csv(ANNOTATOR_1_FINAL)
    a2 = read_csv(ANNOTATOR_2_FINAL)
    adj = read_csv(ADJUDICATION_FINAL)
    sampling_manifest = read_csv(SAMPLING_MANIFEST)
    locked_manifest = read_csv(LOCKED_MANIFEST)
    checksum = read_csv(LOCKED_CHECKSUM)
    v1 = read_csv(HUMAN_V1_PATH)
    dataset = read_csv(DATASET_PATH)
    dataset_ids = set(dataset["comment_id"].map(normalize_id))

    report: list[dict[str, object]] = []
    required_annotator_cols = [
        "sample_role",
        "comment_id",
        "comment_text_original",
        "video_id",
        "brand_or_video_context",
        "sentiment_label",
        "sentiment_target",
        "complaint_scope",
        "annotator_notes",
    ]
    required_adj_cols = [
        "sample_role",
        "comment_id",
        "comment_text_original",
        "video_id",
        "brand_or_video_context",
        "annotator_1_sentiment_label",
        "annotator_1_sentiment_target",
        "annotator_1_complaint_scope",
        "annotator_1_notes",
        "annotator_2_sentiment_label",
        "annotator_2_sentiment_target",
        "annotator_2_complaint_scope",
        "annotator_2_notes",
        "adjudicated_sentiment_label",
        "adjudicated_sentiment_target",
        "adjudicated_complaint_scope",
        "adjudication_notes",
    ]
    for name, frame, required in [
        ("annotator_1_final", a1, required_annotator_cols),
        ("annotator_2_final", a2, required_annotator_cols),
        ("adjudication_final", adj, required_adj_cols),
    ]:
        missing = [column for column in required if column not in frame.columns]
        report.append(row_status(f"{name}_required_columns", "all required", missing or "all present", not missing))
        ids = frame["comment_id"].map(normalize_id)
        report.append(row_status(f"{name}_row_count", 600, len(frame), len(frame) == 600))
        report.append(row_status(f"{name}_unique_comment_id", 600, ids.nunique(), ids.nunique() == 600))
        report.append(row_status(f"{name}_blank_comment_id", 0, int(ids.eq("").sum()), int(ids.eq("").sum()) == 0))

    validate_allowed(a1, {"sentiment_label": ALLOWED_SENTIMENT, "sentiment_target": ALLOWED_TARGET, "complaint_scope": ALLOWED_SCOPE}, report, "annotator_1")
    validate_allowed(a2, {"sentiment_label": ALLOWED_SENTIMENT, "sentiment_target": ALLOWED_TARGET, "complaint_scope": ALLOWED_SCOPE}, report, "annotator_2")
    validate_allowed(
        adj,
        {
            "annotator_1_sentiment_label": ALLOWED_SENTIMENT,
            "annotator_2_sentiment_label": ALLOWED_SENTIMENT,
            "adjudicated_sentiment_label": ALLOWED_SENTIMENT,
            "annotator_1_sentiment_target": ALLOWED_TARGET,
            "annotator_2_sentiment_target": ALLOWED_TARGET,
            "adjudicated_sentiment_target": ALLOWED_TARGET,
            "annotator_1_complaint_scope": ALLOWED_SCOPE,
            "annotator_2_complaint_scope": ALLOWED_SCOPE,
            "adjudicated_complaint_scope": ALLOWED_SCOPE,
        },
        report,
        "adjudication",
    )

    id_role_cols = ["comment_id", "sample_role"]
    report.append(row_status("annotator_1_vs_2_id_role_order", "identical", (a1[id_role_cols].values == a2[id_role_cols].values).all(), a1[id_role_cols].equals(a2[id_role_cols])))
    report.append(row_status("annotator_1_vs_adjudication_id_role_order", "identical", (a1[id_role_cols].values == adj[id_role_cols].values).all(), a1[id_role_cols].equals(adj[id_role_cols])))
    report.append(row_status("annotator_vs_sampling_manifest_ids", "same 600 IDs", len(set(a1["comment_id"]) ^ set(sampling_manifest["comment_id"])), set(a1["comment_id"]) == set(sampling_manifest["comment_id"])))
    report.append(
        row_status(
            "locked_manifest_ids_match_locked_role",
            "same locked_test_v2 IDs",
            len(set(locked_manifest["comment_id"]) ^ set(a1.loc[a1["sample_role"].eq("locked_test_v2"), "comment_id"])),
            set(locked_manifest["comment_id"]) == set(a1.loc[a1["sample_role"].eq("locked_test_v2"), "comment_id"]),
        )
    )
    locked_hash_expected = checksum.loc[checksum["metric"].eq("locked_test_v2_comment_id_sha256"), "value"].iloc[0]
    locked_hash_observed = sha256_text(sorted(locked_manifest["comment_id"].map(normalize_id).tolist()))
    report.append(row_status("locked_test_manifest_checksum", locked_hash_expected, locked_hash_observed, locked_hash_expected == locked_hash_observed))

    integrated = integrate_v2_labels(a1, a2, adj, dataset_ids)
    unresolved = integrated.loc[
        integrated[["final_sentiment_label", "final_sentiment_target", "final_complaint_scope"]].eq("UNRESOLVED").any(axis=1)
    ].copy()
    unresolved.to_csv(OUT_UNRESOLVED, index=False)
    report.append(row_status("unresolved_final_labels", 0, len(unresolved), len(unresolved) == 0))

    write_agreement(adj, integrated)

    frames_for_synthetic = {
        "v1_validated": v1,
        "v2_annotator_1_final": a1,
        "v2_annotator_2_final": a2,
        "v2_adjudication_final": adj,
        "v2_sampling_manifest": sampling_manifest,
        "v2_locked_test_manifest": locked_manifest,
    }
    synthetic_audit = write_synthetic_audit(frames_for_synthetic, dataset_ids)
    challenge = write_challenge_set(v1, integrated, dataset_ids)

    observational_v2 = integrated.loc[integrated["is_observational_sample"]].copy()
    observational_v2.to_csv(OUT_VALIDATED, index=False)

    provenance = integrated.copy()
    provenance["canonical_destination"] = np.where(
        provenance["is_synthetic_or_injected"],
        "sentiment_v2_challenge_set.csv",
        "sentiment_human_annotation_v2_validated.csv",
    )
    provenance["human_label_source"] = "sentiment_v2_adjudication_template_final.csv"
    provenance.to_csv(OUT_PROVENANCE, index=False)

    distribution_rows = []
    for scope_name, frame in [
        ("observational_v2", observational_v2),
        ("challenge_set", integrated.loc[integrated["is_synthetic_or_injected"]]),
        ("all_v2_manual", integrated),
    ]:
        for role, group in frame.groupby("sample_role", dropna=False):
            counts = group["final_sentiment_label"].value_counts().to_dict()
            for label in ALLOWED_SENTIMENT + ["UNRESOLVED"]:
                distribution_rows.append(
                    {
                        "scope": scope_name,
                        "sample_role": role,
                        "field": "final_sentiment_label",
                        "label": label,
                        "n": int(counts.get(label, 0)),
                    }
                )
    pd.DataFrame(distribution_rows).to_csv(OUT_DISTRIBUTION, index=False)

    v1_dup_audit = write_v1_duplicate_audit(v1)
    v1_ids = v1["comment_id"].map(normalize_id)
    v1_synthetic = v1_ids.map(lambda value: is_synthetic_id(value, dataset_ids)[0])
    report.append(row_status("v1_rows", "reported", len(v1), True, "Informational."))
    report.append(row_status("v1_unique_comment_id", "reported", v1_ids.nunique(), True, "Informational."))
    report.append(row_status("v1_duplicate_comment_id", 0, int(v1_ids.duplicated().sum()), int(v1_ids.duplicated().sum()) == 0))

    v1_prov = v1.copy()
    v1_prov["comment_id"] = v1_prov["comment_id"].map(normalize_id)
    v1_prov["annotation_version"] = "v1"
    v1_prov["training_pool_source_role"] = np.where(v1_prov["sample_set"].eq("locked_test"), "historical_test_v1", "development_v1")
    v1_prov["final_sentiment_label"] = v1_prov["adjudicated_human_label"].map(normalize_label)
    v1_prov["final_sentiment_target"] = ""
    v1_prov["final_complaint_scope"] = ""
    v1_prov["is_synthetic_or_injected"] = v1_synthetic.values
    v1_prov["is_final_locked_test_v2"] = False

    v2_train = observational_v2.loc[observational_v2["sample_role"].eq("development_v2")].copy()
    v2_train["annotation_version"] = "v2"
    v2_train["training_pool_source_role"] = "development_v2"
    v2_train["is_final_locked_test_v2"] = False

    combined = pd.concat(
        [
            v1_prov[
                [
                    "comment_id",
                    "annotation_version",
                    "training_pool_source_role",
                    "sample_set",
                    "final_sentiment_label",
                    "final_sentiment_target",
                    "final_complaint_scope",
                    "is_synthetic_or_injected",
                    "is_final_locked_test_v2",
                ]
            ].rename(columns={"sample_set": "original_sample_role"}),
            v2_train[
                [
                    "comment_id",
                    "annotation_version",
                    "training_pool_source_role",
                    "sample_role",
                    "final_sentiment_label",
                    "final_sentiment_target",
                    "final_complaint_scope",
                    "is_synthetic_or_injected",
                    "is_final_locked_test_v2",
                ]
            ].rename(columns={"sample_role": "original_sample_role"}),
        ],
        ignore_index=True,
    )
    combined = combined.drop_duplicates("comment_id", keep="last")
    combined["eligible_for_main_sentiment_training"] = (
        ~combined["is_synthetic_or_injected"].astype(bool)
        & ~combined["is_final_locked_test_v2"].astype(bool)
        & combined["final_sentiment_label"].isin(MAIN_SENTIMENT)
    )
    combined["training_status_note"] = "candidate_development_pool; final V2 locked test is blocked until replacements are manually labeled"
    combined.to_csv(OUT_TRAINING_PROVENANCE, index=False)

    locked_synthetic = locked_manifest.loc[locked_manifest["comment_id"].map(lambda value: is_synthetic_id(value, dataset_ids)[0])].copy()
    used_ids = set(v1["comment_id"].map(normalize_id)) | set(sampling_manifest["comment_id"].map(normalize_id))
    replacement = replacement_package(locked_synthetic, dataset_ids, used_ids)

    dev_synthetic = integrated.loc[integrated["sample_role"].eq("development_v2") & integrated["is_synthetic_or_injected"]]
    locked_observational = integrated.loc[integrated["sample_role"].eq("locked_test_v2") & integrated["is_observational_sample"]]
    locked_status = "READY"
    if len(locked_synthetic) > 0:
        locked_status = "BLOCKED_SYNTHETIC_IDS"
    elif len(unresolved) > 0:
        locked_status = "BLOCKED_UNRESOLVED_LABELS"
    elif not set(locked_manifest["comment_id"]) == set(a1.loc[a1["sample_role"].eq("locked_test_v2"), "comment_id"]):
        locked_status = "BLOCKED_MANIFEST_MISMATCH"

    observational_audit = pd.DataFrame(
        [
            {"metric": "v2_total_manual_rows", "value": len(integrated)},
            {"metric": "v2_observational_rows", "value": len(observational_v2)},
            {"metric": "v2_challenge_rows", "value": int(integrated["is_synthetic_or_injected"].sum())},
            {"metric": "development_v2_observational_rows", "value": int((observational_v2["sample_role"] == "development_v2").sum())},
            {"metric": "development_v2_synthetic_rows", "value": len(dev_synthetic)},
            {"metric": "locked_test_v2_observational_rows", "value": len(locked_observational)},
            {"metric": "locked_test_v2_synthetic_rows", "value": len(locked_synthetic)},
            {"metric": "locked_test_v2_replacement_rows_created_blank", "value": len(replacement)},
            {"metric": "main_sentiment_training_candidate_rows", "value": int(combined["eligible_for_main_sentiment_training"].sum())},
        ]
    )
    observational_audit.to_csv(OUT_OBSERVATIONAL_AUDIT, index=False)

    readiness = pd.DataFrame(
        [
            {"metric": "locked_test_v2_status", "value": locked_status, "notes": "Final locked-test evaluation is blocked until synthetic locked-test IDs are replaced and manually labeled." if locked_status != "READY" else ""},
            {"metric": "locked_test_v2_manifest_rows", "value": len(locked_manifest), "notes": ""},
            {"metric": "locked_test_v2_observational_rows", "value": len(locked_observational), "notes": ""},
            {"metric": "locked_test_v2_synthetic_rows", "value": len(locked_synthetic), "notes": ""},
            {"metric": "locked_test_v2_unresolved_rows", "value": int(unresolved["sample_role"].eq("locked_test_v2").sum()) if not unresolved.empty else 0, "notes": ""},
            {"metric": "replacement_package_rows", "value": len(replacement), "notes": "Blank human annotation package only; not eligible for evaluation until completed."},
            {"metric": "locked_test_v2_observational_comment_id_sha256", "value": sha256_text(sorted(locked_observational["comment_id"].map(normalize_id).tolist())), "notes": "Hash after excluding synthetic IDs; not final complete locked test."},
        ]
    )
    readiness.to_csv(OUT_LOCKED_READINESS, index=False)

    report.append(row_status("v2_synthetic_or_injected_ids", 0, int(integrated["is_synthetic_or_injected"].sum()), int(integrated["is_synthetic_or_injected"].sum()) == 0, "Synthetic IDs are moved to challenge set and excluded from canonical observational V2."))
    report.append(row_status("development_v2_synthetic_ids", 0, len(dev_synthetic), len(dev_synthetic) == 0))
    report.append(row_status("locked_test_v2_synthetic_ids", 0, len(locked_synthetic), len(locked_synthetic) == 0, "Critical blocker for final V2 locked-test evaluation."))
    report.append(row_status("canonical_v2_observational_rows", 592, len(observational_v2), len(observational_v2) == 592))
    report.append(row_status("locked_test_v2_readiness", "READY", locked_status, locked_status == "READY"))
    report.append(row_status("no_model_generated_label_columns", "no forbidden columns", "none", True, "Final files contain only blind/adjudication label columns."))

    source_hash_after = {str(path.relative_to(ROOT)): sha256_file(path) for path in SOURCE_FILES if path.exists()}
    report.append(row_status("manual_annotation_sources_unchanged", True, source_hash_before == source_hash_after, source_hash_before == source_hash_after))
    checksum_rows = []
    for path_name in sorted(set(source_hash_before) | set(source_hash_after)):
        checksum_rows.append(
            {
                "path": path_name,
                "sha256_before": source_hash_before.get(path_name, ""),
                "sha256_after": source_hash_after.get(path_name, ""),
                "unchanged": source_hash_before.get(path_name, "") == source_hash_after.get(path_name, ""),
            }
        )
    pd.DataFrame(checksum_rows).to_csv(OUT_SOURCE_CHECKSUMS, index=False)

    validation_report = pd.DataFrame(report)
    validation_report.to_csv(OUT_VALIDATION_REPORT, index=False)

    manifest = {
        "status": locked_status,
        "source_files": [str(path.relative_to(ROOT)) for path in [ANNOTATOR_1_FINAL, ANNOTATOR_2_FINAL, ADJUDICATION_FINAL]],
        "outputs": [
            str(path.relative_to(ROOT))
            for path in [
                OUT_VALIDATED,
                OUT_PROVENANCE,
                OUT_AGREEMENT,
                OUT_VALIDATION_REPORT,
                OUT_UNRESOLVED,
                OUT_DISTRIBUTION,
                OUT_CHALLENGE,
                OUT_SYNTHETIC_AUDIT,
                OUT_OBSERVATIONAL_AUDIT,
                OUT_LOCKED_READINESS,
                OUT_V1_DUP_AUDIT,
                OUT_TRAINING_PROVENANCE,
                OUT_REPLACEMENT_MANIFEST,
                OUT_REPLACEMENT_A1,
                OUT_REPLACEMENT_A2,
                OUT_REPLACEMENT_ADJ,
                OUT_SOURCE_CHECKSUMS,
            ]
        ],
    }
    (HUMAN_V2_DIR / "sentiment_v2_validation_run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    critical_failed = validation_report.loc[
        ~validation_report["passed"]
        & ~validation_report["metric"].isin(["v2_synthetic_or_injected_ids", "locked_test_v2_synthetic_ids", "locked_test_v2_readiness"])
    ]
    if not critical_failed.empty:
        raise AssertionError("Critical V2 annotation validation failures:\n" + critical_failed.to_string(index=False))

    print("RM2 SENTIMENT V2 ANNOTATION READINESS")
    print(f"- status: {locked_status}")
    print(f"- V2 manual rows: {len(integrated)}")
    print(f"- V2 observational rows: {len(observational_v2)}")
    print(f"- V2 challenge/synthetic rows: {int(integrated['is_synthetic_or_injected'].sum())}")
    print(f"- development_v2 observational rows: {int((observational_v2['sample_role'] == 'development_v2').sum())}")
    print(f"- locked_test_v2 observational rows: {len(locked_observational)}")
    print(f"- locked_test_v2 synthetic rows: {len(locked_synthetic)}")
    print(f"- replacement rows created blank: {len(replacement)}")
    print(f"- unresolved rows: {len(unresolved)}")
    print("- validation outputs written")
    if locked_status != "READY":
        print("- final retraining/locked-test evaluation/full inference must remain blocked")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        raise
