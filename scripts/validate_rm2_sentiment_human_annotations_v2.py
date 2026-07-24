from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DIR = ROOT / "output/rm2_sentiment/validation/human_v2"
V1_VALIDATED_PATH = ROOT / "output/rm2_sentiment/validation/human_v1/sentiment_human_annotation_validated.csv"

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


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, dtype=str, low_memory=False).fillna("")


def clean_id(value: object) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def is_blank(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().eq("")


def cohen_kappa(labels_a: pd.Series, labels_b: pd.Series, allowed_labels: list[str]) -> float:
    a = labels_a.astype(str).reset_index(drop=True)
    b = labels_b.astype(str).reset_index(drop=True)
    if len(a) == 0:
        return float("nan")
    observed = float((a == b).mean())
    expected = 0.0
    for label in allowed_labels:
        expected += float((a == label).mean()) * float((b == label).mean())
    if np.isclose(1.0 - expected, 0.0):
        return float("nan")
    return (observed - expected) / (1.0 - expected)


def status_row(metric: str, expected: object, observed: object, status: str, notes: str = "") -> dict[str, object]:
    return {
        "metric": metric,
        "expected": expected,
        "observed": observed,
        "status": status,
        "notes": notes,
    }


def validate_allowed(frame: pd.DataFrame, column: str, allowed: list[str]) -> tuple[str, int, str]:
    values = frame[column].astype(str).str.strip()
    blank_count = int(values.eq("").sum())
    nonblank = values.loc[values.ne("")]
    invalid = sorted(set(nonblank) - set(allowed))
    if len(nonblank) == 0:
        return "NOT_AVAILABLE", blank_count, "Template labels are still blank."
    if invalid:
        return "FAIL", blank_count, "Invalid labels: " + ", ".join(invalid[:20])
    if blank_count > 0:
        return "FAIL", blank_count, "Some labels are still blank."
    return "PASS", 0, "All labels are valid and complete."


def agreement_rows(adjudication: pd.DataFrame, prefix_a: str, prefix_b: str, field: str, allowed: list[str]) -> list[dict[str, object]]:
    col_a = f"{prefix_a}_{field}"
    col_b = f"{prefix_b}_{field}"
    if col_a not in adjudication.columns or col_b not in adjudication.columns:
        return [status_row(f"{field}_agreement", "columns present", "missing", "FAIL", f"Missing {col_a} or {col_b}")]
    labels_a = adjudication[col_a].astype(str).str.strip()
    labels_b = adjudication[col_b].astype(str).str.strip()
    complete = labels_a.ne("") & labels_b.ne("")
    if not complete.any():
        return [
            status_row(f"{field}_agreement", "completed annotator labels", "not available", "NOT_AVAILABLE", "No completed labels yet."),
            status_row(f"{field}_cohen_kappa", "completed annotator labels", "not available", "NOT_AVAILABLE", "No completed labels yet."),
        ]
    if (~complete).any():
        return [
            status_row(f"{field}_agreement", "all paired labels complete", int(complete.sum()), "FAIL", "Some paired labels are blank."),
            status_row(f"{field}_cohen_kappa", "all paired labels complete", "not computed", "FAIL", "Some paired labels are blank."),
        ]
    invalid_a = sorted(set(labels_a) - set(allowed))
    invalid_b = sorted(set(labels_b) - set(allowed))
    if invalid_a or invalid_b:
        return [
            status_row(
                f"{field}_agreement",
                "allowed labels only",
                "invalid",
                "FAIL",
                "Invalid labels: " + ", ".join((invalid_a + invalid_b)[:20]),
            )
        ]
    agreement = float((labels_a == labels_b).mean())
    kappa = cohen_kappa(labels_a, labels_b, allowed)
    return [
        status_row(f"{field}_agreement", "computed", round(agreement, 6), "PASS", ""),
        status_row(f"{field}_cohen_kappa", "computed", round(kappa, 6) if np.isfinite(kappa) else "nan", "PASS", ""),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate RM2 sentiment V2 human annotation package.")
    parser.add_argument("--human-dir", default=str(DEFAULT_DIR), help="Path to validation/human_v2 directory.")
    parser.add_argument(
        "--output",
        default=str(DEFAULT_DIR / "v2_annotation_validation_report.csv"),
        help="Validation report CSV path.",
    )
    args = parser.parse_args()

    human_dir = Path(args.human_dir)
    ann1 = read_csv(human_dir / "sentiment_v2_annotator_1_blind.csv")
    ann2 = read_csv(human_dir / "sentiment_v2_annotator_2_blind.csv")
    adjudication = read_csv(human_dir / "sentiment_v2_adjudication_template.csv")
    provenance = read_csv(human_dir / "annotation_provenance_v2.csv")
    locked_manifest = read_csv(human_dir / "locked_test_v2_manifest.csv")
    v1 = read_csv(V1_VALIDATED_PATH)

    rows: list[dict[str, object]] = []
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
    for name, frame in [("annotator_1", ann1), ("annotator_2", ann2)]:
        missing_cols = [col for col in required_annotator_cols if col not in frame.columns]
        rows.append(
            status_row(
                f"{name}_required_columns",
                "all required columns",
                ",".join(missing_cols) if missing_cols else "all present",
                "FAIL" if missing_cols else "PASS",
                "",
            )
        )
        role_counts = frame["sample_role"].value_counts().to_dict() if "sample_role" in frame.columns else {}
        rows.append(status_row(f"{name}_development_v2_count", 300, role_counts.get("development_v2", 0), "PASS" if role_counts.get("development_v2", 0) == 300 else "FAIL", ""))
        rows.append(status_row(f"{name}_locked_test_v2_count", 300, role_counts.get("locked_test_v2", 0), "PASS" if role_counts.get("locked_test_v2", 0) == 300 else "FAIL", ""))
        unique_ids = frame["comment_id"].map(clean_id).nunique() if "comment_id" in frame.columns else 0
        rows.append(status_row(f"{name}_unique_comment_ids", 600, unique_ids, "PASS" if unique_ids == 600 else "FAIL", ""))
        for column, allowed in [
            ("sentiment_label", ALLOWED_SENTIMENT),
            ("sentiment_target", ALLOWED_TARGET),
            ("complaint_scope", ALLOWED_SCOPE),
        ]:
            status, blank_count, notes = validate_allowed(frame, column, allowed)
            rows.append(status_row(f"{name}_{column}_validity", "allowed labels when completed", f"blank={blank_count}", status, notes))

    ann1_ids = set(ann1["comment_id"].map(clean_id)) - {""}
    ann2_ids = set(ann2["comment_id"].map(clean_id)) - {""}
    v2_ids = ann1_ids | ann2_ids
    dev_ids = set(ann1.loc[ann1["sample_role"].eq("development_v2"), "comment_id"].map(clean_id))
    locked_ids = set(ann1.loc[ann1["sample_role"].eq("locked_test_v2"), "comment_id"].map(clean_id))
    v1_ids = set(v1["comment_id"].map(clean_id)) - {""}

    rows.append(status_row("annotator_files_same_comment_ids", "same 600 IDs", len(ann1_ids ^ ann2_ids), "PASS" if ann1_ids == ann2_ids else "FAIL", ""))
    rows.append(status_row("development_locked_overlap", 0, len(dev_ids & locked_ids), "PASS" if not (dev_ids & locked_ids) else "FAIL", ""))
    rows.append(status_row("v1_v2_overlap", 0, len(v1_ids & v2_ids), "PASS" if not (v1_ids & v2_ids) else "FAIL", ""))

    locked_manifest_ids = set(locked_manifest["comment_id"].map(clean_id)) - {""}
    rows.append(status_row("locked_manifest_count", 300, len(locked_manifest_ids), "PASS" if len(locked_manifest_ids) == 300 else "FAIL", ""))
    rows.append(status_row("locked_manifest_matches_annotator_file", "same locked_test_v2 IDs", len(locked_manifest_ids ^ locked_ids), "PASS" if locked_manifest_ids == locked_ids else "FAIL", ""))

    prov_v2 = provenance.loc[provenance["annotation_version"].eq("v2_pending")].copy()
    rows.append(status_row("provenance_v2_count", 600, prov_v2["comment_id"].map(clean_id).nunique(), "PASS" if prov_v2["comment_id"].map(clean_id).nunique() == 600 else "FAIL", ""))
    rows.append(status_row("provenance_v1_retained", "current V1 IDs included", len(v1_ids - set(provenance["comment_id"].map(clean_id))), "PASS" if v1_ids.issubset(set(provenance["comment_id"].map(clean_id))) else "FAIL", ""))

    rows.extend(agreement_rows(adjudication, "annotator_1", "annotator_2", "sentiment_label", ALLOWED_SENTIMENT))
    rows.extend(agreement_rows(adjudication, "annotator_1", "annotator_2", "sentiment_target", ALLOWED_TARGET))
    rows.extend(agreement_rows(adjudication, "annotator_1", "annotator_2", "complaint_scope", ALLOWED_SCOPE))

    for field, allowed in [
        ("adjudicated_sentiment_label", ALLOWED_SENTIMENT),
        ("adjudicated_sentiment_target", ALLOWED_TARGET),
        ("adjudicated_complaint_scope", ALLOWED_SCOPE),
    ]:
        if field in adjudication.columns:
            status, blank_count, notes = validate_allowed(adjudication, field, allowed)
            rows.append(status_row(f"{field}_coverage", "complete after adjudication", f"blank={blank_count}", status, notes))
        else:
            rows.append(status_row(f"{field}_coverage", "column present", "missing", "FAIL", ""))

    report = pd.DataFrame(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output, index=False)
    print(report.to_string(index=False))
    if report["status"].eq("FAIL").any():
        sys.exit(1)


if __name__ == "__main__":
    main()
