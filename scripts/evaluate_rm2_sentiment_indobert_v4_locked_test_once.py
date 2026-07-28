from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report
from transformers import AutoModelForSequenceClassification, AutoTokenizer

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_rm2_sentiment_indobert_v4_final import (
    FORBIDDEN_LABEL_SOURCES,
    ID_TO_LABEL,
    LABELS,
    LABEL_TO_ID,
    LOCKED_REGISTRY,
    MODEL_DIR,
    OUT_DIR,
    ROOT,
    calibration_metrics,
    confusion_frame,
    device_info,
    metric_bundle,
    model_input,
    package_versions,
    per_class_frame,
    predict_proba,
    read_development_data,
    read_locked_test_data,
    save_json,
    sha256_dataframe,
    sha256_file,
)


LOCKED_OUT_DIR = OUT_DIR / "locked_test_evaluation"
TRAINING_MANIFEST = OUT_DIR / "INDOBERT_V4_TRAINING_MANIFEST.json"
EVAL_MANIFEST = LOCKED_OUT_DIR / "LOCKED_TEST_EVALUATION_MANIFEST.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def assert_not_evaluated(force: bool) -> None:
    if force:
        return
    if EVAL_MANIFEST.exists():
        manifest = load_json(EVAL_MANIFEST)
        if manifest.get("evaluated_once") is True:
            raise RuntimeError(
                "Locked test V4 has already been evaluated once. "
                "Do not rerun or tune based on locked-test results."
            )


def selected_text_mode(model_dir: Path) -> str:
    config_path = model_dir / "selected_trial_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing selected model config: {config_path}")
    config = load_json(config_path)
    return str(config.get("text_mode") or config.get("final_training_config", {}).get("text_mode") or "context_sep_comment")


def selected_max_length(model_dir: Path) -> int:
    config = load_json(model_dir / "selected_trial_config.json")
    value = config.get("max_length") or config.get("final_training_config", {}).get("max_length")
    if value is None:
        raise ValueError("selected_trial_config.json does not include max_length")
    return int(value)


def prediction_batch_size(device: torch.device, max_length: int) -> int:
    if device.type != "cuda":
        return 8
    return 48 if max_length <= 128 else 32 if max_length <= 192 else 24


def verify_model_hashes(model_dir: Path, frozen_manifest: dict[str, Any]) -> dict[str, str]:
    sha_path = model_dir / "SHA256SUMS.txt"
    if not sha_path.exists():
        raise FileNotFoundError(f"Missing SHA256SUMS.txt in frozen model dir: {model_dir}")
    verified: dict[str, str] = {}
    for line in sha_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, name = line.split(maxsplit=1)
        path = model_dir / name.strip()
        if not path.exists():
            raise FileNotFoundError(f"Checksum references missing model file: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"Frozen model checksum mismatch for {name.strip()}: expected {expected}, got {actual}")
        verified[name.strip()] = actual
    checkpoint_hash = verified.get("model.safetensors") or verified.get("pytorch_model.bin")
    manifest_hash = frozen_manifest.get("checkpoint_sha256")
    if manifest_hash and checkpoint_hash and str(manifest_hash) != str(checkpoint_hash):
        raise RuntimeError("Frozen manifest checkpoint_sha256 does not match model checkpoint hash.")
    return verified


def acceptance_decision(metrics: dict[str, Any], per_class: pd.DataFrame, training_manifest: dict[str, Any]) -> dict[str, Any]:
    positive = per_class.loc[per_class["label"].eq("Positive")].iloc[0]
    negative = per_class.loc[per_class["label"].eq("Negative")].iloc[0]
    neutral = per_class.loc[per_class["label"].eq("Neutral")].iloc[0]
    macro_f1 = float(metrics["macro_f1"])
    accuracy = float(metrics["accuracy"])
    min_recall = float(metrics["min_class_recall"])
    mcc = float(metrics.get("mcc", float("nan")))
    predicted_support = {
        label: int(metrics.get(f"{label.lower()}_predicted_support", 0))
        for label in LABELS
    }
    criteria = {
        "no_data_leakage": True,
        "locked_test_excluded_from_training_or_selection": True,
        "macro_f1_gte_v2_baseline_0p7309": bool(macro_f1 >= 0.7309),
        "positive_recall_gte_0p70": bool(float(positive["recall"]) >= 0.70),
        "accuracy_gte_0p8159": bool(accuracy >= 0.8159),
        "minimum_class_recall_gte_0p60": bool(min_recall >= 0.60),
        "mcc_gte_0p60": bool(np.isfinite(mcc) and mcc >= 0.60),
        "no_class_collapse": bool(all(value > 0 for value in predicted_support.values())),
        "label_mapping_verified": True,
    }
    recommended = bool(all(criteria.values()))
    return {
        "status": (
            "INDOBERT_V4_ACCEPTED_AS_FINAL_RM2_SENTIMENT_MODEL"
            if recommended
            else "INDOBERT_V4_NOT_ACCEPTED_KEEP_V2"
        ),
        "created_at_utc": utc_now(),
        "recommended_for_rm2_sentiment_final": recommended,
        "accepted_as_final_rm2_sentiment_model": recommended,
        "recommendation_basis": (
            "Strict preregistered gate for this application: no data leakage, locked-test exclusion, "
            "macro-F1 >= 0.7309, Positive recall >= 0.70, accuracy >= 0.8159, minimum class recall >= 0.60, "
            "MCC >= 0.60, no class collapse, and verified label mapping. These gates are not used for tuning."
        ),
        "acceptance_criteria": criteria,
        "failed_criteria": [key for key, passed in criteria.items() if not passed],
        "baseline_legacy_v2": {
            "accuracy": 0.8359,
            "macro_f1": 0.7309,
            "balanced_accuracy": 0.7188,
            "mcc": 0.6369,
            "positive_recall": 0.4773,
            "positive_f1": 0.5753,
        },
        "metrics": {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "weighted_f1": float(metrics["weighted_f1"]),
            "balanced_accuracy": float(metrics["balanced_accuracy"]),
            "mcc": mcc,
            "min_class_recall": min_recall,
            "negative_recall": float(negative["recall"]),
            "negative_precision": float(negative["precision"]),
            "negative_f1": float(negative["f1"]),
            "neutral_recall": float(neutral["recall"]),
            "neutral_precision": float(neutral["precision"]),
            "neutral_f1": float(neutral["f1"]),
            "positive_recall": float(positive["recall"]),
            "positive_precision": float(positive["precision"]),
            "positive_f1": float(positive["f1"]),
        },
        "methodology": {
            "model_selection_source": "development OOF only",
            "locked_test_used_for_training_or_selection": False,
            "locked_test_used_for_threshold_selection": False,
            "prediction_rule": "argmax, no locked-test threshold tuning",
            "label_source_column": "final_human_label",
            "prediction_label_source_used": False,
            "forbidden_supervision_sources": FORBIDDEN_LABEL_SOURCES,
        },
        "training_manifest_selected_trial_id": training_manifest.get("selected_trial_id", ""),
    }


def write_report_and_error_analysis(predictions: pd.DataFrame, y_true: np.ndarray, pred: np.ndarray) -> None:
    report = classification_report(
        y_true,
        pred,
        labels=list(range(len(LABELS))),
        target_names=LABELS,
        output_dict=True,
        zero_division=0,
    )
    save_json(LOCKED_OUT_DIR / "locked_test_classification_report.json", report)
    errors = predictions.loc[predictions["final_human_label"].ne(predictions["predicted_label"])].copy()
    errors["error_type"] = np.select(
        [
            errors["predicted_label"].eq("Positive") & ~errors["final_human_label"].eq("Positive"),
            errors["final_human_label"].eq("Positive") & ~errors["predicted_label"].eq("Positive"),
        ],
        ["positive_false_positive", "positive_false_negative"],
        default="other_error",
    )
    errors.loc[errors["error_type"].eq("positive_false_positive")].to_csv(
        LOCKED_OUT_DIR / "error_analysis_false_positive.csv",
        index=False,
        encoding="utf-8-sig",
    )
    errors.loc[errors["error_type"].eq("positive_false_negative")].to_csv(
        LOCKED_OUT_DIR / "error_analysis_false_negative.csv",
        index=False,
        encoding="utf-8-sig",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the frozen RM2 V4 IndoBERT candidate on locked test once.")
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Administrative escape hatch only; default enforces one-time locked-test evaluation.",
    )
    parser.add_argument("--model-dir", default=str(MODEL_DIR), help="Frozen model directory to evaluate.")
    parser.add_argument("--frozen-manifest", default=str(TRAINING_MANIFEST), help="Training/freeze manifest certifying locked-test exclusion.")
    args = parser.parse_args()
    assert_not_evaluated(args.force_rerun)
    model_dir = Path(args.model_dir)
    frozen_manifest_path = Path(args.frozen_manifest)
    if not model_dir.is_absolute():
        model_dir = ROOT / model_dir
    if not frozen_manifest_path.is_absolute():
        frozen_manifest_path = ROOT / frozen_manifest_path
    if not frozen_manifest_path.exists():
        raise FileNotFoundError(f"Train/freeze the V4 candidate first: {frozen_manifest_path}")

    training_manifest = load_json(frozen_manifest_path)
    if training_manifest.get("locked_test_used_for_training_or_selection") is not False:
        raise AssertionError("Training manifest does not certify locked-test exclusion.")
    if training_manifest.get("locked_test_evaluated") is not False:
        raise AssertionError("Frozen manifest must be created before locked-test evaluation.")

    LOCKED_OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    info = device_info()
    precision = str(info["mixed_precision"])
    text_mode = selected_text_mode(model_dir)
    max_length = selected_max_length(model_dir)
    locked = read_locked_test_data()
    dev = read_development_data()
    comment_overlap = set(dev["comment_id"]) & set(locked["comment_id"])
    text_overlap = set(dev["text_cluster_id"]) & set(locked["text_cluster_id"])
    exact_overlap = set(dev["exact_duplicate_group_id"]) & set(locked["exact_duplicate_group_id"])
    near_overlap = set(dev["near_duplicate_cluster_id"]) & set(locked["near_duplicate_cluster_id"])
    if comment_overlap or text_overlap or exact_overlap:
        raise AssertionError("Hard development/locked-test leakage detected.")

    model_hashes = verify_model_hashes(model_dir, training_manifest)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=False)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, trust_remote_code=False)
    model.to(device)
    texts = locked.apply(lambda row: model_input(row, text_mode), axis=1).tolist()
    probs = predict_proba(
        model,
        tokenizer,
        texts,
        max_length,
        prediction_batch_size(device, max_length),
        device,
        precision,
    )
    y_true = locked["label_id"].to_numpy(dtype=int)
    pred = probs.argmax(axis=1)
    metrics = metric_bundle(y_true, probs)
    metrics.update(calibration_metrics(y_true, probs))
    for label_id, label in ID_TO_LABEL.items():
        metrics[f"{label.lower()}_predicted_support"] = int((pred == label_id).sum())
    metrics_frame = pd.DataFrame(
        [
            {
                "metric": key,
                "value": value,
            }
            for key, value in metrics.items()
            if isinstance(value, int | float | np.integer | np.floating)
        ]
    )
    metrics_frame.to_csv(LOCKED_OUT_DIR / "locked_test_metrics.csv", index=False, encoding="utf-8-sig")
    per_class = per_class_frame(y_true, probs)
    per_class.to_csv(LOCKED_OUT_DIR / "locked_test_per_class_metrics.csv", index=False, encoding="utf-8-sig")
    confusion_frame(y_true, probs).to_csv(LOCKED_OUT_DIR / "locked_test_confusion_matrix.csv", index=False, encoding="utf-8-sig")
    predictions = locked[
        [
            "comment_id",
            "video_id",
            "comment_text",
            "model_text",
            "product_category",
            "brand_or_video_context",
            "final_human_label",
            "evaluable_three_class",
            "text_cluster_id",
            "exact_duplicate_group_id",
            "near_duplicate_cluster_id",
        ]
    ].copy()
    predictions["predicted_label"] = [ID_TO_LABEL[int(idx)] for idx in pred]
    predictions["confidence"] = probs.max(axis=1)
    predictions["prob_negative"] = probs[:, LABEL_TO_ID["Negative"]]
    predictions["prob_neutral"] = probs[:, LABEL_TO_ID["Neutral"]]
    predictions["prob_positive"] = probs[:, LABEL_TO_ID["Positive"]]
    predictions.to_csv(LOCKED_OUT_DIR / "locked_test_predictions.csv", index=False, encoding="utf-8-sig")
    write_report_and_error_analysis(predictions, y_true, pred)

    manifest = {
        "status": "LOCKED_TEST_EVALUATION_COMPLETE",
        "created_at_utc": utc_now(),
        "evaluated_once": True,
        "locked_test_execution_count": 1,
        "force_rerun_used": bool(args.force_rerun),
        "locked_test_registry": LOCKED_REGISTRY.relative_to(ROOT).as_posix(),
        "locked_test_registry_sha256": sha256_file(LOCKED_REGISTRY),
        "locked_test_evaluable_rows": int(len(locked)),
        "locked_test_class_counts": locked["final_human_label"].value_counts().reindex(LABELS).fillna(0).astype(int).to_dict(),
        "locked_test_neutral_target_shortfall": 6,
        "development_registry_hash_for_audit": sha256_dataframe(dev, ["comment_id", "final_human_label", "cv_group_id"]),
        "locked_registry_hash": sha256_dataframe(locked, ["comment_id", "final_human_label", "cv_group_id"]),
        "hard_split_overlap": {
            "comment_id": int(len(comment_overlap)),
            "text_cluster_id": int(len(text_overlap)),
            "exact_duplicate_group_id": int(len(exact_overlap)),
            "near_duplicate_cluster_id_report_only": int(len(near_overlap)),
        },
        "near_duplicate_overlap_note": (
            "Near-duplicate clusters are reported because the locked test is frozen; "
            "comment_id, text_cluster_id, and exact_duplicate_group_id hard leakage are zero."
        ),
        "model_dir": model_dir.relative_to(ROOT).as_posix(),
        "frozen_manifest": frozen_manifest_path.relative_to(ROOT).as_posix(),
        "model_hashes": model_hashes,
        "text_mode": text_mode,
        "max_length": int(max_length),
        "prediction_rule": "argmax_no_threshold_tuning",
        "label_source_column": "final_human_label",
        "label_vocabulary": LABELS,
        "prediction_label_source_used": False,
        "locked_test_used_for_training_or_selection": False,
        "locked_test_used_for_threshold_selection": False,
        "locked_test_used_for_early_stopping": False,
        "full_corpus_inference_run": False,
        "metrics": metrics,
        "package_versions": package_versions(),
        "device_info": info,
        "training_manifest_selected_trial_id": training_manifest.get("selected_trial_id", training_manifest.get("exact_trial_id", "")),
    }
    save_json(EVAL_MANIFEST, manifest)
    decision = acceptance_decision(metrics, per_class, training_manifest)
    save_json(LOCKED_OUT_DIR / "FINAL_ACCEPTANCE_DECISION.json", decision)
    print(
        json.dumps(
            {
                "locked_test_accuracy": metrics["accuracy"],
                "locked_test_macro_f1": metrics["macro_f1"],
                "recommended_for_rm2_sentiment_final": decision["recommended_for_rm2_sentiment_final"],
                "output_dir": LOCKED_OUT_DIR.relative_to(ROOT).as_posix(),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
