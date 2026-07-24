from __future__ import annotations

import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
HUMAN_DIR = ROOT / "output/rm2_sentiment/validation/human_v2"
MODEL_DIR = ROOT / "output/rm2_sentiment/model/frozen"

DATASET = ROOT / "dataset.csv"
COMMENT_SENTIMENT = ROOT / "output/rm2_sentiment/legacy/v1/tables/comment_sentiment.csv"
TRAINING_PROVENANCE = MODEL_DIR / "development_training_pool_provenance.csv"
MODEL_ARTIFACT = MODEL_DIR / "selected_model_development_frozen.joblib"
MODEL_CONFIG = MODEL_DIR / "selected_model_development_frozen_config.json"

LOCKED_MANIFEST = HUMAN_DIR / "locked_test_v2_manifest.csv"
V2_VALIDATED = HUMAN_DIR / "sentiment_human_annotation_v2_validated.csv"
REPLACEMENT_MANIFEST = HUMAN_DIR / "sentiment_v2_locked_test_replacement_manifest.csv"

REPLACEMENT_A1_CANONICAL = HUMAN_DIR / "sentiment_v2_locked_test_replacement_annotator_1_blind.csv"
REPLACEMENT_A2_CANONICAL = HUMAN_DIR / "sentiment_v2_locked_test_replacement_annotator_2_blind.csv"
REPLACEMENT_A1_LABELED = HUMAN_DIR / "sentiment_v2_locked_test_replacement_annotator_1_blind_labeled.csv"
REPLACEMENT_A2_LABELED = HUMAN_DIR / "sentiment_v2_locked_test_replacement_annotator_2_blind_labeled.csv"
REPLACEMENT_ADJ_TEMPLATE = HUMAN_DIR / "sentiment_v2_locked_test_replacement_adjudication_template.csv"
REPLACEMENT_ADJ_FINAL = HUMAN_DIR / "sentiment_v2_replacement_adjudication_final.csv"

LOCKED_FINAL = HUMAN_DIR / "locked_test_v2_observational_final.csv"
LOCKED_FINAL_MANIFEST = HUMAN_DIR / "locked_test_v2_observational_final_manifest.json"
LOCKED_FINAL_INTEGRITY = HUMAN_DIR / "locked_test_v2_observational_final_integrity.csv"
LOCKED_FINAL_CHECKSUM = HUMAN_DIR / "locked_test_v2_observational_final_checksum.csv"

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
EXPECTED_MODEL_SHA256 = "477bfe11ebc3463eeff5a5fb8359f86022d719c8bea49dfdad0460fa0c5e2ccc"


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False, low_memory=False)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def normalize_label(value: object) -> str:
    text = "" if pd.isna(value) else str(value).strip()
    return "No Text" if text.lower() in {"no_text", "notext", "no text"} else text


def gate(metric: str, expected: object, observed: object, passed: bool, notes: str = "") -> dict[str, object]:
    return {
        "metric": metric,
        "expected": expected,
        "observed": observed,
        "passed": bool(passed),
        "notes": notes,
    }


def question_flag(text: str) -> bool:
    return bool(
        re.search(
            r"(?:\?|apa|apakah|gimana|bagaimana|boleh|bisa|cocok\s*(?:ga|gak|nggak|ngga)?|"
            r"aman\s*(?:ga|gak|nggak|ngga)?|untuk kulit|cara pakai|dipakai|boleh pakai)",
            str(text),
            flags=re.IGNORECASE,
        )
    )


def validate_annotation_frame(df: pd.DataFrame, name: str, expected_ids: set[str]) -> list[dict[str, object]]:
    report: list[dict[str, object]] = []
    required = [
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
    for col in required:
        report.append(gate(f"{name}_{col}_present", "present", col in df.columns, col in df.columns))
    if any(not row["passed"] for row in report):
        return report
    ids = set(df["comment_id"].astype(str))
    report.extend(
        [
            gate(f"{name}_rows", 8, len(df), len(df) == 8),
            gate(f"{name}_unique_comment_id", 8, df["comment_id"].nunique(), df["comment_id"].nunique() == 8),
            gate(f"{name}_matches_replacement_ids", sorted(expected_ids), sorted(ids), ids == expected_ids),
            gate(f"{name}_no_inj_prefix", 0, int(df["comment_id"].str.match(r"(?i)^INJ").sum()), int(df["comment_id"].str.match(r"(?i)^INJ").sum()) == 0),
        ]
    )
    allowed = {
        "sentiment_label": ALLOWED_SENTIMENT,
        "sentiment_target": ALLOWED_TARGET,
        "complaint_scope": ALLOWED_SCOPE,
    }
    for col, values in allowed.items():
        normalized = df[col].map(normalize_label)
        invalid = sorted(set(normalized) - set(values))
        blanks = int(normalized.eq("").sum())
        report.append(gate(f"{name}_{col}_valid", "allowed values", invalid or "none", not invalid))
        report.append(gate(f"{name}_{col}_complete", 0, blanks, blanks == 0))
    notes_blank = int(df["annotator_notes"].astype(str).str.strip().eq("").sum())
    report.append(gate(f"{name}_annotator_notes_complete", 0, notes_blank, notes_blank == 0))
    return report


def make_replacement_adjudication(a1: pd.DataFrame, a2: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    left = a1.rename(
        columns={
            "sentiment_label": "annotator_1_sentiment_label",
            "sentiment_target": "annotator_1_sentiment_target",
            "complaint_scope": "annotator_1_complaint_scope",
            "annotator_notes": "annotator_1_notes",
        }
    )
    right = a2.rename(
        columns={
            "sentiment_label": "annotator_2_sentiment_label",
            "sentiment_target": "annotator_2_sentiment_target",
            "complaint_scope": "annotator_2_complaint_scope",
            "annotator_notes": "annotator_2_notes",
        }
    )
    keep_right = [
        "comment_id",
        "annotator_2_sentiment_label",
        "annotator_2_sentiment_target",
        "annotator_2_complaint_scope",
        "annotator_2_notes",
    ]
    merged = left.merge(right[keep_right], on="comment_id", how="inner")
    merged = manifest[
        [
            "comment_id",
            "replaces_comment_id",
            "sampling_stratum",
            "requested_sampling_stratum",
            "replacement_status",
            "selection_seed",
        ]
    ].merge(merged, on="comment_id", how="left")
    for field in ["sentiment_label", "sentiment_target", "complaint_scope"]:
        merged[f"{field}_agreement"] = (
            merged[f"annotator_1_{field}"].map(normalize_label)
            == merged[f"annotator_2_{field}"].map(normalize_label)
        )
    merged["all_fields_agree"] = merged[
        ["sentiment_label_agreement", "sentiment_target_agreement", "complaint_scope_agreement"]
    ].all(axis=1)
    merged["adjudicated_sentiment_label"] = merged["annotator_1_sentiment_label"].map(normalize_label)
    merged["adjudicated_sentiment_target"] = merged["annotator_1_sentiment_target"].map(normalize_label)
    merged["adjudicated_complaint_scope"] = merged["annotator_1_complaint_scope"].map(normalize_label)
    merged["adjudication_reason"] = merged.apply(
        lambda row: "Automatic agreement: both annotators provided identical sentiment label, target, and complaint scope."
        if row["all_fields_agree"]
        else "PENDING_HUMAN_ADJUDICATION",
        axis=1,
    )
    merged["adjudicator_status"] = merged["all_fields_agree"].map(
        {True: "AUTO_AGREEMENT_RECORDED", False: "PENDING_HUMAN_ADJUDICATION"}
    )
    merged["provenance"] = "locked_test_v2_replacement"
    merged["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    return merged


def build_final_locked_test(replacement_adj: pd.DataFrame) -> pd.DataFrame:
    old = read_csv(V2_VALIDATED)
    old_locked = old[old["sample_role"].eq("locked_test_v2")].copy()
    old_locked["source_segment"] = "original_locked_test_v2_observational"
    old_locked["replaces_comment_id"] = ""
    old_locked["sampling_stratum"] = ""
    old_locked["requested_sampling_stratum"] = ""
    old_locked["replacement_status"] = ""
    old_locked["selection_seed"] = ""

    repl_cols = [
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
        "adjudication_reason",
        "replaces_comment_id",
        "sampling_stratum",
        "requested_sampling_stratum",
        "replacement_status",
        "selection_seed",
    ]
    repl = replacement_adj.copy()
    repl["sample_role"] = "locked_test_v2"
    repl_final = pd.DataFrame(
        {
            "sample_role": repl["sample_role"],
            "comment_id": repl["comment_id"],
            "comment_text_original": repl["comment_text_original"],
            "video_id": repl["video_id"],
            "brand_or_video_context": repl["brand_or_video_context"],
            "annotator_1_label": repl["annotator_1_sentiment_label"],
            "annotator_1_target": repl["annotator_1_sentiment_target"],
            "annotator_1_scope": repl["annotator_1_complaint_scope"],
            "annotator_1_notes": repl["annotator_1_notes"],
            "annotator_2_label": repl["annotator_2_sentiment_label"],
            "annotator_2_target": repl["annotator_2_sentiment_target"],
            "annotator_2_scope": repl["annotator_2_complaint_scope"],
            "annotator_2_notes": repl["annotator_2_notes"],
            "adjudicated_label": repl["adjudicated_sentiment_label"],
            "adjudicated_target": repl["adjudicated_sentiment_target"],
            "adjudicated_scope": repl["adjudicated_complaint_scope"],
            "adjudication_notes": repl["adjudication_reason"],
            "final_sentiment_label": repl["adjudicated_sentiment_label"],
            "sentiment_label_source": "replacement_agreement",
            "final_sentiment_target": repl["adjudicated_sentiment_target"],
            "sentiment_target_source": "replacement_agreement",
            "final_complaint_scope": repl["adjudicated_complaint_scope"],
            "complaint_scope_source": "replacement_agreement",
            "is_synthetic_or_injected": False,
            "synthetic_reason": "",
            "is_observational_sample": True,
            "is_evaluable_sentiment": repl["adjudicated_sentiment_label"].isin(MAIN_SENTIMENT),
            "source_segment": "locked_test_v2_replacement_observational",
            "replaces_comment_id": repl["replaces_comment_id"],
            "sampling_stratum": repl["sampling_stratum"],
            "requested_sampling_stratum": repl["requested_sampling_stratum"],
            "replacement_status": repl["replacement_status"],
            "selection_seed": repl["selection_seed"],
        }
    )

    final = pd.concat([old_locked, repl_final], ignore_index=True, sort=False).fillna("")

    # Preserve the original locked-test manifest order, replacing INJ IDs with their observational replacements.
    manifest = read_csv(LOCKED_MANIFEST)
    replacement_map = dict(zip(repl["replaces_comment_id"], repl["comment_id"]))
    order = [replacement_map.get(cid, cid) for cid in manifest["comment_id"].astype(str)]
    order_index = {cid: i for i, cid in enumerate(order)}
    final["locked_test_order"] = final["comment_id"].map(order_index)
    final = final.sort_values("locked_test_order").reset_index(drop=True)
    final["locked_test_order"] = final["locked_test_order"].astype(int)

    dataset = read_csv(DATASET)
    dataset_meta = dataset[
        ["comment_id", "comment_type", "product_category", "timestamp", "username", "user_id", "parent_comment_id", "parent_user", "text"]
    ].drop_duplicates("comment_id")
    final = final.merge(dataset_meta, on="comment_id", how="left", suffixes=("", "_dataset"))
    final["dataset_text_matches"] = final["text"].astype(str).str.strip().eq(final["comment_text_original"].astype(str).str.strip())
    final["is_question"] = final["comment_text_original"].map(question_flag)

    if COMMENT_SENTIMENT.exists():
        cs = read_csv(COMMENT_SENTIMENT)
        meta_cols = [
            "comment_id",
            "is_hcc_member",
            "is_hcc",
            "hcc_id",
            "community",
            "brand_label_auto",
            "primary_brand",
        ]
        available = [c for c in meta_cols if c in cs.columns]
        final = final.merge(cs[available].drop_duplicates("comment_id"), on="comment_id", how="left")
    return final


def main() -> None:
    report: list[dict[str, object]] = []
    start_head = git_head()
    config = json.loads(MODEL_CONFIG.read_text(encoding="utf-8"))
    model_hash = sha256_file(MODEL_ARTIFACT)
    config_hash = sha256_file(MODEL_CONFIG)
    report.extend(
        [
            gate("git_head_present", "non-empty", start_head, bool(start_head)),
            gate("model_hash", EXPECTED_MODEL_SHA256, model_hash, model_hash == EXPECTED_MODEL_SHA256),
            gate("threshold", 0.42, config.get("threshold"), float(config.get("threshold")) == 0.42),
            gate("model_status", "DEVELOPMENT_MODEL_FROZEN_PENDING_LOCKED_TEST", config.get("status"), config.get("status") == "DEVELOPMENT_MODEL_FROZEN_PENDING_LOCKED_TEST"),
        ]
    )

    dataset = read_csv(DATASET)
    dataset_ids = set(dataset["comment_id"].astype(str))
    training = read_csv(TRAINING_PROVENANCE)
    training_ids = set(
        training.loc[
            training["included_in_training"].astype(str).str.lower().eq("true"),
            "comment_id",
        ].astype(str)
    )
    replacement_manifest = read_csv(REPLACEMENT_MANIFEST)
    replacement_ids = set(replacement_manifest["comment_id"].astype(str))
    a1 = read_csv(REPLACEMENT_A1_LABELED)
    a2 = read_csv(REPLACEMENT_A2_LABELED)

    report.extend(validate_annotation_frame(a1, "annotator_1_replacement", replacement_ids))
    report.extend(validate_annotation_frame(a2, "annotator_2_replacement", replacement_ids))
    report.extend(
        [
            gate("replacement_rows", 8, len(replacement_manifest), len(replacement_manifest) == 8),
            gate("replacement_unique_comment_id", 8, replacement_manifest["comment_id"].nunique(), replacement_manifest["comment_id"].nunique() == 8),
            gate("replacement_no_inj_prefix", 0, int(replacement_manifest["comment_id"].str.match(r"(?i)^INJ").sum()), int(replacement_manifest["comment_id"].str.match(r"(?i)^INJ").sum()) == 0),
            gate("replacement_ids_in_dataset", 8, len(replacement_ids & dataset_ids), len(replacement_ids & dataset_ids) == 8),
            gate("replacement_overlap_training", 0, len(replacement_ids & training_ids), len(replacement_ids & training_ids) == 0),
        ]
    )
    if any(not row["passed"] for row in report):
        pd.DataFrame(report).to_csv(LOCKED_FINAL_INTEGRITY, index=False)
        raise AssertionError(pd.DataFrame(report).loc[lambda df: ~df["passed"]].to_string(index=False))

    # Save human-filled labels into canonical replacement files for downstream reproducibility.
    a1.to_csv(REPLACEMENT_A1_CANONICAL, index=False)
    a2.to_csv(REPLACEMENT_A2_CANONICAL, index=False)

    replacement_adj = make_replacement_adjudication(a1, a2, replacement_manifest)
    REPLACEMENT_ADJ_TEMPLATE.write_text(
        replacement_adj[
            [
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
                "adjudication_reason",
            ]
        ].to_csv(index=False),
        encoding="utf-8",
    )
    replacement_adj.to_csv(REPLACEMENT_ADJ_FINAL, index=False)
    disagreement_count = int((~replacement_adj["all_fields_agree"]).sum())
    report.append(gate("replacement_disagreement_count", 0, disagreement_count, disagreement_count == 0))
    if disagreement_count:
        pd.DataFrame(report).to_csv(LOCKED_FINAL_INTEGRITY, index=False)
        raise AssertionError("Replacement annotations contain disagreement; human adjudication is required.")

    locked_final = build_final_locked_test(replacement_adj)
    locked_ids = set(locked_final["comment_id"].astype(str))
    old_locked_count = int(locked_final["source_segment"].eq("original_locked_test_v2_observational").sum())
    replacement_count = int(locked_final["source_segment"].eq("locked_test_v2_replacement_observational").sum())
    report.extend(
        [
            gate("locked_final_rows", 300, len(locked_final), len(locked_final) == 300),
            gate("locked_final_unique_comment_id", 300, locked_final["comment_id"].nunique(), locked_final["comment_id"].nunique() == 300),
            gate("locked_final_synthetic_or_injected", 0, int(locked_final["comment_id"].str.match(r"(?i)^INJ").sum()), int(locked_final["comment_id"].str.match(r"(?i)^INJ").sum()) == 0),
            gate("locked_final_old_observational_rows", 292, old_locked_count, old_locked_count == 292),
            gate("locked_final_replacement_rows", 8, replacement_count, replacement_count == 8),
            gate("locked_final_overlap_training", 0, len(locked_ids & training_ids), len(locked_ids & training_ids) == 0),
            gate("locked_final_all_ids_in_dataset", 300, len(locked_ids & dataset_ids), len(locked_ids & dataset_ids) == 300),
            gate("locked_final_missing_label", 0, int(locked_final["final_sentiment_label"].astype(str).str.strip().eq("").sum()), int(locked_final["final_sentiment_label"].astype(str).str.strip().eq("").sum()) == 0),
            gate("locked_final_unresolved", 0, int(locked_final["final_sentiment_label"].eq("UNRESOLVED").sum()), int(locked_final["final_sentiment_label"].eq("UNRESOLVED").sum()) == 0),
            gate("locked_final_duplicate_comment_id", 0, int(locked_final["comment_id"].duplicated().sum()), int(locked_final["comment_id"].duplicated().sum()) == 0),
            gate("locked_test_predictions_still_absent", 0, int((MODEL_DIR / "final_locked_test_predictions.csv").exists()), not (MODEL_DIR / "final_locked_test_predictions.csv").exists()),
        ]
    )
    if any(not row["passed"] for row in report):
        pd.DataFrame(report).to_csv(LOCKED_FINAL_INTEGRITY, index=False)
        raise AssertionError(pd.DataFrame(report).loc[lambda df: ~df["passed"]].to_string(index=False))

    locked_final.to_csv(LOCKED_FINAL, index=False)
    final_hash = sha256_file(LOCKED_FINAL)
    pd.DataFrame([{"file": str(LOCKED_FINAL.relative_to(ROOT)), "sha256": final_hash}]).to_csv(
        LOCKED_FINAL_CHECKSUM, index=False
    )

    def value_counts(column: str) -> dict[str, int]:
        if column not in locked_final.columns:
            return {}
        return {str(k): int(v) for k, v in locked_final[column].value_counts(dropna=False).sort_index().items()}

    manifest = {
        "status": "READY_FOR_ONE_TIME_LOCKED_TEST_EVALUATION",
        "timestamp_freeze_utc": datetime.now(timezone.utc).isoformat(),
        "git_head_before_freeze": start_head,
        "rows": int(len(locked_final)),
        "unique_comment_id": int(locked_final["comment_id"].nunique()),
        "class_distribution": value_counts("final_sentiment_label"),
        "brand_distribution": value_counts("brand_or_video_context"),
        "hcc_nonhcc_distribution": value_counts("is_hcc_member"),
        "video_distribution": value_counts("video_id"),
        "comment_type_distribution": value_counts("comment_type"),
        "question_distribution": value_counts("is_question"),
        "sentiment_target_distribution": value_counts("final_sentiment_target"),
        "complaint_scope_distribution": value_counts("final_complaint_scope"),
        "original_292_ids": locked_final.loc[
            locked_final["source_segment"].eq("original_locked_test_v2_observational"), "comment_id"
        ].astype(str).tolist(),
        "replacement_8_ids": locked_final.loc[
            locked_final["source_segment"].eq("locked_test_v2_replacement_observational"), "comment_id"
        ].astype(str).tolist(),
        "training_overlap_count": int(len(locked_ids & training_ids)),
        "synthetic_count": int(locked_final["comment_id"].str.match(r"(?i)^INJ").sum()),
        "locked_test_final_sha256": final_hash,
        "model_hash": model_hash,
        "config_hash": config_hash,
        "threshold": float(config["threshold"]),
        "replacement_agreement_count": int(replacement_adj["all_fields_agree"].sum()),
        "replacement_disagreement_count": disagreement_count,
        "replacement_replaces_ids": dict(
            zip(replacement_manifest["replaces_comment_id"], replacement_manifest["comment_id"])
        ),
    }
    LOCKED_FINAL_MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    report.extend(
        [
            gate("locked_final_sha256_present", "sha256", final_hash, len(final_hash) == 64),
            gate("locked_final_status", "READY_FOR_ONE_TIME_LOCKED_TEST_EVALUATION", manifest["status"], manifest["status"] == "READY_FOR_ONE_TIME_LOCKED_TEST_EVALUATION"),
        ]
    )
    integrity = pd.DataFrame(report)
    integrity.to_csv(LOCKED_FINAL_INTEGRITY, index=False)
    if not integrity["passed"].all():
        raise AssertionError(integrity.loc[lambda df: ~df["passed"]].to_string(index=False))

    print("LOCKED_TEST_V2_OBSERVATIONAL_FINAL_READY")
    print(f"rows={len(locked_final)} unique={locked_final['comment_id'].nunique()} sha256={final_hash}")
    print(f"replacement_agreement={manifest['replacement_agreement_count']} disagreement={disagreement_count}")
    print(f"integrity_passed={integrity['passed'].all()} gates={len(integrity)}")


if __name__ == "__main__":
    main()
