from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FROZEN_DIR = ROOT / "output/rm2_sentiment/model/frozen"
LEGACY_V3_EVAL_DIR = ROOT / "output/rm2_sentiment/experiments/indobert_v3/final_test_evaluation"
OUT_DIR = ROOT / "output/rm2_sentiment/experiments/v2_positive_recall"

METRICS = FROZEN_DIR / "final_locked_test_metrics.csv"
PER_CLASS = FROZEN_DIR / "final_locked_test_per_class_metrics.csv"
CONFUSION = FROZEN_DIR / "final_locked_test_confusion_matrix.csv"
REPORT = FROZEN_DIR / "final_locked_test_report.json"
V2_V3_SAME_TEST = LEGACY_V3_EVAL_DIR / "v2_v3_same_test_predictions.csv"

OUT_METRICS = OUT_DIR / "baseline_v2_metrics.json"
OUT_CONFUSION = OUT_DIR / "baseline_v2_confusion_matrix.csv"
OUT_POSITIVE_ERRORS = OUT_DIR / "baseline_v2_positive_error_cases.csv"

LABELS = ["Negative", "Neutral", "Positive"]
LEGACY_SCOPE = "LEGACY_DIAGNOSTIC_TEST_ALREADY_OPENED"


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, dtype=str, keep_default_na=False, low_memory=False)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def metric_value(metrics: pd.DataFrame, name: str) -> float:
    value = metrics.loc[metrics["metric"].eq(name), "value"].iloc[0]
    return float(value)


def per_class_value(per_class: pd.DataFrame, label: str, metric: str) -> float:
    value = per_class.loc[per_class["class_label"].eq(label), metric].iloc[0]
    return float(value)


def write_positive_error_cases() -> dict[str, int]:
    columns = [
        "legacy_scope",
        "comment_id",
        "true_label",
        "v2_predicted_label",
        "v2_covered",
        "v2_final_eval_label",
        "v2_confidence",
        "v2_prob_negative",
        "v2_prob_neutral",
        "v2_prob_positive",
        "brand_or_video_context",
        "comment_text_original",
    ]
    if not V2_V3_SAME_TEST.exists():
        pd.DataFrame(columns=columns).to_csv(OUT_POSITIVE_ERRORS, index=False, encoding="utf-8-sig")
        return {"positive_true": 0, "positive_predicted_positive": 0, "positive_to_neutral": 0, "positive_to_negative": 0, "positive_abstain": 0}

    pred = read_csv(V2_V3_SAME_TEST)
    positive = pred.loc[pred["final_sentiment_label"].eq("Positive")].copy()
    covered = positive["v2_covered"].astype(str).str.lower().eq("true")
    positive["v2_final_eval_label"] = positive["v2_predicted_label"].where(covered, "Uncertain")
    errors = positive.loc[~positive["v2_final_eval_label"].eq("Positive")].copy()
    errors.insert(0, "legacy_scope", LEGACY_SCOPE)
    errors = errors.rename(columns={"final_sentiment_label": "true_label"})
    for col in columns:
        if col not in errors.columns:
            errors[col] = ""
    errors[columns].to_csv(OUT_POSITIVE_ERRORS, index=False, encoding="utf-8-sig")

    return {
        "positive_true": int(len(positive)),
        "positive_predicted_positive": int(positive["v2_final_eval_label"].eq("Positive").sum()),
        "positive_to_neutral": int(positive["v2_final_eval_label"].eq("Neutral").sum()),
        "positive_to_negative": int(positive["v2_final_eval_label"].eq("Negative").sum()),
        "positive_abstain": int(positive["v2_final_eval_label"].eq("Uncertain").sum()),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics = read_csv(METRICS)
    per_class = read_csv(PER_CLASS)
    confusion = read_csv(CONFUSION)

    missing_labels = set(LABELS) - set(confusion.columns[1:])
    if missing_labels:
        raise AssertionError(f"Confusion matrix missing columns: {sorted(missing_labels)}")
    if confusion["true_label"].tolist() != LABELS:
        raise AssertionError("Unexpected confusion matrix label order.")

    positive_error_counts = write_positive_error_cases()
    report = json.loads(REPORT.read_text(encoding="utf-8")) if REPORT.exists() else {}
    payload = {
        "status": "BASELINE_V2_FROZEN_SNAPSHOT_FOR_POSITIVE_RECALL_EXPERIMENT",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": LEGACY_SCOPE,
        "methodological_warning": (
            "This old locked test has already been opened. It is stored only as baseline "
            "provenance and diagnostics, not for training, threshold tuning, feature "
            "selection, class-weight selection, preprocessing selection, or model selection."
        ),
        "metrics": {
            "accuracy_covered": metric_value(metrics, "accuracy_covered"),
            "macro_f1_covered": metric_value(metrics, "macro_f1_covered"),
            "balanced_accuracy": metric_value(metrics, "balanced_accuracy_covered"),
            "mcc": metric_value(metrics, "mcc_covered"),
            "coverage": metric_value(metrics, "coverage"),
            "positive_precision": per_class_value(per_class, "Positive", "precision"),
            "positive_recall": per_class_value(per_class, "Positive", "recall"),
            "positive_f1": per_class_value(per_class, "Positive", "f1"),
        },
        "positive_error_summary_on_legacy_test": positive_error_counts,
        "source_files": {
            METRICS.relative_to(ROOT).as_posix(): sha256_file(METRICS),
            PER_CLASS.relative_to(ROOT).as_posix(): sha256_file(PER_CLASS),
            CONFUSION.relative_to(ROOT).as_posix(): sha256_file(CONFUSION),
            REPORT.relative_to(ROOT).as_posix(): sha256_file(REPORT) if REPORT.exists() else "",
            V2_V3_SAME_TEST.relative_to(ROOT).as_posix(): sha256_file(V2_V3_SAME_TEST) if V2_V3_SAME_TEST.exists() else "",
        },
        "reported_legacy_test_hash": report.get("locked_test_sha256", ""),
        "policy_confirmations": {
            "v2_frozen_left_unchanged": True,
            "old_locked_test_used_for_model_selection": False,
            "low_confidence_to_positive_rule": False,
            "neutral_to_positive_mass_shift": False,
            "uncertain_to_positive_mass_shift": False,
        },
    }
    OUT_METRICS.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    confusion.to_csv(OUT_CONFUSION, index=False, encoding="utf-8-sig")

    print("BASELINE_V2_FROZEN_SNAPSHOT_FOR_POSITIVE_RECALL_EXPERIMENT")
    print(json.dumps(payload["metrics"], indent=2))
    print(json.dumps(payload["positive_error_summary_on_legacy_test"], indent=2))


if __name__ == "__main__":
    main()
