from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

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


def selected_text_mode() -> str:
    config_path = MODEL_DIR / "selected_trial_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Missing selected model config: {config_path}")
    config = load_json(config_path)
    return str(config.get("text_mode") or config.get("final_training_config", {}).get("text_mode") or "context_sep_comment")


def selected_max_length() -> int:
    config = load_json(MODEL_DIR / "selected_trial_config.json")
    value = config.get("max_length") or config.get("final_training_config", {}).get("max_length")
    if value is None:
        raise ValueError("selected_trial_config.json does not include max_length")
    return int(value)


def prediction_batch_size(device: torch.device, max_length: int) -> int:
    if device.type != "cuda":
        return 8
    return 48 if max_length <= 128 else 32 if max_length <= 192 else 24


def acceptance_decision(metrics: dict[str, Any], per_class: pd.DataFrame, training_manifest: dict[str, Any]) -> dict[str, Any]:
    positive = per_class.loc[per_class["label"].eq("Positive")].iloc[0]
    negative = per_class.loc[per_class["label"].eq("Negative")].iloc[0]
    neutral = per_class.loc[per_class["label"].eq("Neutral")].iloc[0]
    macro_f1 = float(metrics["macro_f1"])
    accuracy = float(metrics["accuracy"])
    min_recall = float(metrics["min_class_recall"])
    recommended = bool(macro_f1 >= 0.70 and accuracy >= 0.75 and min_recall >= 0.60)
    return {
        "status": "LOCKED_TEST_EVALUATED_ONCE",
        "created_at_utc": utc_now(),
        "recommended_for_rm2_sentiment_final": recommended,
        "recommendation_basis": (
            "Recommend promotion only if locked-test accuracy >= 0.75, macro-F1 >= 0.70, "
            "and every class recall >= 0.60. These gates are report-only and were not used for tuning."
        ),
        "metrics": {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "weighted_f1": float(metrics["weighted_f1"]),
            "balanced_accuracy": float(metrics["balanced_accuracy"]),
            "min_class_recall": min_recall,
            "negative_recall": float(negative["recall"]),
            "negative_f1": float(negative["f1"]),
            "neutral_recall": float(neutral["recall"]),
            "neutral_f1": float(neutral["f1"]),
            "positive_recall": float(positive["recall"]),
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the frozen RM2 V4 IndoBERT candidate on locked test once.")
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Administrative escape hatch only; default enforces one-time locked-test evaluation.",
    )
    args = parser.parse_args()
    assert_not_evaluated(args.force_rerun)
    if not TRAINING_MANIFEST.exists():
        raise FileNotFoundError(f"Train/freeze the V4 candidate first: {TRAINING_MANIFEST}")

    training_manifest = load_json(TRAINING_MANIFEST)
    if training_manifest.get("locked_test_used_for_training_or_selection") is not False:
        raise AssertionError("Training manifest does not certify locked-test exclusion.")

    LOCKED_OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    info = device_info()
    precision = str(info["mixed_precision"])
    text_mode = selected_text_mode()
    max_length = selected_max_length()
    locked = read_locked_test_data()
    dev = read_development_data()
    comment_overlap = set(dev["comment_id"]) & set(locked["comment_id"])
    text_overlap = set(dev["text_cluster_id"]) & set(locked["text_cluster_id"])
    exact_overlap = set(dev["exact_duplicate_group_id"]) & set(locked["exact_duplicate_group_id"])
    near_overlap = set(dev["near_duplicate_cluster_id"]) & set(locked["near_duplicate_cluster_id"])
    if comment_overlap or text_overlap or exact_overlap:
        raise AssertionError("Hard development/locked-test leakage detected.")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=False)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, trust_remote_code=False)
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

    model_hashes = {
        path.name: sha256_file(path)
        for path in MODEL_DIR.glob("*")
        if path.is_file() and path.suffix in {".safetensors", ".bin", ".json"}
    }
    manifest = {
        "status": "LOCKED_TEST_EVALUATION_COMPLETE",
        "created_at_utc": utc_now(),
        "evaluated_once": True,
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
        "model_dir": MODEL_DIR.relative_to(ROOT).as_posix(),
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
        "training_manifest_selected_trial_id": training_manifest.get("selected_trial_id", ""),
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
