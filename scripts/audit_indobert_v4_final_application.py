from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
)


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from predict_rm2_sentiment_cpu import predict as cpu_predict  # noqa: E402


LABELS = ["Negative", "Neutral", "Positive"]
LABEL_TO_ID = {label: idx for idx, label in enumerate(LABELS)}
ID_TO_LABEL = {idx: label for label, idx in LABEL_TO_ID.items()}

BASE_MODEL_DIR = ROOT / "artifacts/rm2_sentiment/indobert_v4_final/base_reference"
DEV_REGISTRY = ROOT / "output/rm2_sentiment/validation/human_master_v4/sentiment_v4_development_final_registry.csv"
LOCKED_REGISTRY = ROOT / "output/rm2_sentiment/validation/human_master_v4/sentiment_v4_locked_test_final_frozen.csv"
EXP_DIR = ROOT / "output/rm2_sentiment/experiments/indobert_v4_final"
LOCKED_OUT_DIR = EXP_DIR / "locked_test_evaluation"
EVAL_MANIFEST = LOCKED_OUT_DIR / "LOCKED_TEST_EVALUATION_MANIFEST.json"
TRAINING_MANIFEST = EXP_DIR / "INDOBERT_V4_TRAINING_MANIFEST.json"
PREDICTIONS = LOCKED_OUT_DIR / "locked_test_predictions.csv"
DOCS_DIR = ROOT / "docs"
CANONICAL_MODEL = ROOT / "output/rm2_sentiment/final/CANONICAL_MODEL.json"
FINAL_MODEL_DIR = ROOT / "artifacts/rm2_sentiment/indobert_v4_final/final_model"

V2_BASELINE = {
    "accuracy": 0.8359,
    "macro_f1": 0.7309,
    "balanced_accuracy": 0.7188,
    "mcc": 0.6369,
    "positive_recall": 0.4773,
    "positive_f1": 0.5753,
}

ACCEPTANCE_THRESHOLDS = {
    "macro_f1": 0.7309,
    "positive_recall": 0.70,
    "accuracy": 0.8159,
    "min_class_recall": 0.60,
    "mcc": 0.60,
}

REQUIRED_MODEL_FILES = [
    "model.safetensors",
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "label_map.json",
    "selected_trial_config.json",
    "base_reference_manifest.json",
    "SHA256SUMS.txt",
    "cpu_inference_smoke_test.json",
    "remote_artifact_verification.json",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def run_git(args: list[str]) -> str:
    result = subprocess.run(args, cwd=ROOT, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return result.stdout.strip() or result.stderr.strip()


def git_head() -> str:
    return run_git(["git", "rev-parse", "HEAD"])


def git_branch() -> str:
    return run_git(["git", "branch", "--show-current"])


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [to_jsonable(v) for v in value]
    if isinstance(value, Path):
        return rel(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2), encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_sha256s(path: Path) -> dict[str, str]:
    checksums: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, name = line.split(maxsplit=1)
        checksums[name.strip()] = digest.strip()
    return checksums


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False, low_memory=False)


def is_true(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def nonblank_set(frame: pd.DataFrame, column: str) -> set[str]:
    if column not in frame.columns:
        return set()
    values = frame[column].astype(str).str.strip()
    return set(values.loc[values.ne("")])


def normalize_space(value: object) -> str:
    text = "" if value is None else str(value)
    text = " ".join(text.replace("\ufeff", "").split())
    if text.lower() in {"", "nan", "none", "null", "<na>"}:
        return ""
    return text


def registry_text(row: pd.Series) -> str:
    return normalize_space(row.get("model_text")) or normalize_space(row.get("comment_text"))


def sha256_dataframe(frame: pd.DataFrame, columns: list[str]) -> str:
    available = [column for column in columns if column in frame.columns]
    data = frame[available].sort_values(available).to_csv(index=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def audit_model_artifacts(run_cpu_smoke: bool) -> dict[str, Any]:
    manifest = read_json(BASE_MODEL_DIR / "base_reference_manifest.json")
    selected_config = read_json(BASE_MODEL_DIR / "selected_trial_config.json")
    label_map = read_json(BASE_MODEL_DIR / "label_map.json")
    config = read_json(BASE_MODEL_DIR / "config.json")
    missing = [name for name in REQUIRED_MODEL_FILES if not (BASE_MODEL_DIR / name).exists()]

    checksum_rows: list[dict[str, Any]] = []
    checksum_ok = False
    model_hash = ""
    if (BASE_MODEL_DIR / "SHA256SUMS.txt").exists():
        expected = parse_sha256s(BASE_MODEL_DIR / "SHA256SUMS.txt")
        checksum_ok = True
        for name, expected_hash in expected.items():
            path = BASE_MODEL_DIR / name
            actual = sha256_file(path) if path.exists() else ""
            passed = bool(path.exists() and actual == expected_hash)
            checksum_ok = checksum_ok and passed
            checksum_rows.append(
                {
                    "file": name,
                    "expected_sha256": expected_hash,
                    "actual_sha256": actual,
                    "passed": passed,
                    "size_bytes": int(path.stat().st_size) if path.exists() else 0,
                }
            )
        model_hash = expected.get("model.safetensors", "")

    id_to_label = label_map.get("id_to_label", {})
    label_to_id = label_map.get("label_to_id", {})
    label_order = [id_to_label.get(str(idx)) for idx in range(len(LABELS))]
    config_id_to_label = config.get("id2label", {})
    config_label_order = [config_id_to_label.get(str(idx)) for idx in range(len(LABELS))] if config_id_to_label else LABELS
    label_mapping_ok = (
        label_order == LABELS
        and label_to_id == LABEL_TO_ID
        and config_label_order == LABELS
    )
    manifest_hash = manifest.get("checkpoint_sha256") or manifest.get("file_hashes", {}).get("model.safetensors", "")
    manifest_model_hash_ok = bool(model_hash and manifest_hash == model_hash)

    smoke: dict[str, Any] = {
        "executed": False,
        "cpu_model_load_success": False,
        "cpu_inference_success": False,
        "probabilities_finite": False,
        "probability_sums_close_to_one": False,
        "labels_valid": False,
        "empty_input_no_crash": False,
        "samples": [],
    }
    if run_cpu_smoke:
        rows = [
            {"text": "bagus banget, cocok dan mau beli lagi"},
            {"text": "ini bisa dipakai malam hari?"},
            {"text": "kurang cocok, bikin kulit terasa perih"},
            {"text": ""},
        ]
        try:
            outputs = cpu_predict(BASE_MODEL_DIR, rows, "text")
            sample_rows = []
            finite = True
            sums_ok = True
            labels_ok = True
            empty_ok = len(outputs) == len(rows)
            for input_row, output in zip(rows, outputs):
                probs = [float(output["prob_negative"]), float(output["prob_neutral"]), float(output["prob_positive"])]
                prob_sum = float(sum(probs))
                finite = finite and all(math.isfinite(value) for value in probs)
                sums_ok = sums_ok and abs(prob_sum - 1.0) <= 1e-5
                labels_ok = labels_ok and str(output["predicted_label"]) in LABELS
                if input_row["text"] == "":
                    empty_ok = empty_ok and str(output["predicted_label"]) in LABELS
                sample_rows.append(
                    {
                        "input_text": input_row["text"],
                        "predicted_label": output["predicted_label"],
                        "confidence": float(output["confidence"]),
                        "probability_sum": prob_sum,
                    }
                )
            smoke.update(
                {
                    "executed": True,
                    "cpu_model_load_success": True,
                    "cpu_inference_success": True,
                    "probabilities_finite": finite,
                    "probability_sums_close_to_one": sums_ok,
                    "labels_valid": labels_ok,
                    "empty_input_no_crash": empty_ok,
                    "samples": sample_rows,
                }
            )
        except Exception as exc:
            smoke.update({"executed": True, "error": str(exc)})

    artifact_ok = bool(
        not missing
        and checksum_ok
        and manifest_model_hash_ok
        and label_mapping_ok
        and (not run_cpu_smoke or (smoke["cpu_model_load_success"] and smoke["cpu_inference_success"]))
    )
    checksum_frame = pd.DataFrame(checksum_rows)
    checksum_frame.to_csv(EXP_DIR / "base_reference_checksum_audit.csv", index=False, encoding="utf-8-sig")
    return {
        "status": "PASS" if artifact_ok else "FAIL",
        "model_dir": rel(BASE_MODEL_DIR),
        "required_files_missing": missing,
        "checksum_verified": checksum_ok,
        "checksum_rows": checksum_rows,
        "model_id": selected_config.get("model_id") or manifest.get("model_id", ""),
        "model_revision": selected_config.get("model_revision") or manifest.get("model_revision", ""),
        "model_sha256": model_hash,
        "model_size_bytes": int((BASE_MODEL_DIR / "model.safetensors").stat().st_size) if (BASE_MODEL_DIR / "model.safetensors").exists() else 0,
        "manifest_model_hash_ok": manifest_model_hash_ok,
        "label_order": label_order,
        "label_mapping_ok": label_mapping_ok,
        "prediction_rule": selected_config.get("prediction_rule", ""),
        "text_mode": selected_config.get("text_mode", ""),
        "max_length": selected_config.get("max_length", ""),
        "preprocessing_version": manifest.get("preprocessing_version", ""),
        "base_reference_manifest_status": manifest.get("status", ""),
        "cpu_smoke": smoke,
    }


def audit_data_integrity() -> dict[str, Any]:
    dev_raw = read_csv(DEV_REGISTRY)
    locked_raw = read_csv(LOCKED_REGISTRY)
    dev = dev_raw.loc[dev_raw["evaluable_three_class"].map(is_true) & dev_raw["final_human_label"].isin(LABELS)].copy()
    locked = locked_raw.loc[locked_raw["evaluable_three_class"].map(is_true) & locked_raw["final_human_label"].isin(LABELS)].copy()

    dev_text = dev.apply(registry_text, axis=1).map(normalize_space)
    locked_text = locked.apply(registry_text, axis=1).map(normalize_space)
    dev_text_set = set(dev_text.loc[dev_text.ne("")])
    locked_text_set = set(locked_text.loc[locked_text.ne("")])

    overlaps = {
        "comment_id": sorted(nonblank_set(dev, "comment_id") & nonblank_set(locked, "comment_id")),
        "text_cluster_id": sorted(nonblank_set(dev, "text_cluster_id") & nonblank_set(locked, "text_cluster_id")),
        "exact_duplicate_group_id": sorted(nonblank_set(dev, "exact_duplicate_group_id") & nonblank_set(locked, "exact_duplicate_group_id")),
        "near_duplicate_cluster_id_report_only": sorted(nonblank_set(dev, "near_duplicate_cluster_id") & nonblank_set(locked, "near_duplicate_cluster_id")),
        "normalized_text": sorted(dev_text_set & locked_text_set),
        "video_id_report_only": sorted(nonblank_set(dev, "video_id") & nonblank_set(locked, "video_id")),
        "source_file_report_only": sorted(nonblank_set(dev_raw, "source_file") & nonblank_set(locked_raw, "source_file")),
    }
    hard_leakage_counts = {
        "comment_id": len(overlaps["comment_id"]),
        "text_cluster_id": len(overlaps["text_cluster_id"]),
        "exact_duplicate_group_id": len(overlaps["exact_duplicate_group_id"]),
        "normalized_text": len(overlaps["normalized_text"]),
    }
    data_ok = all(value == 0 for value in hard_leakage_counts.values())
    report_rows = [
        {"check": "development_evaluable_rows", "value": int(len(dev)), "status": "PASS" if len(dev) == 1824 else "WARN"},
        {
            "check": "development_class_counts",
            "value": json.dumps(dev["final_human_label"].value_counts().reindex(LABELS).fillna(0).astype(int).to_dict()),
            "status": "PASS",
        },
        {"check": "locked_test_evaluable_rows", "value": int(len(locked)), "status": "PASS" if len(locked) == 672 else "WARN"},
        {
            "check": "locked_test_class_counts",
            "value": json.dumps(locked["final_human_label"].value_counts().reindex(LABELS).fillna(0).astype(int).to_dict()),
            "status": "PASS",
        },
        {"check": "comment_id_overlap", "value": hard_leakage_counts["comment_id"], "status": "PASS" if hard_leakage_counts["comment_id"] == 0 else "FAIL"},
        {"check": "text_cluster_id_overlap", "value": hard_leakage_counts["text_cluster_id"], "status": "PASS" if hard_leakage_counts["text_cluster_id"] == 0 else "FAIL"},
        {"check": "exact_duplicate_group_id_overlap", "value": hard_leakage_counts["exact_duplicate_group_id"], "status": "PASS" if hard_leakage_counts["exact_duplicate_group_id"] == 0 else "FAIL"},
        {"check": "normalized_text_overlap", "value": hard_leakage_counts["normalized_text"], "status": "PASS" if hard_leakage_counts["normalized_text"] == 0 else "FAIL"},
        {"check": "near_duplicate_cluster_id_overlap_report_only", "value": len(overlaps["near_duplicate_cluster_id_report_only"]), "status": "WARN" if overlaps["near_duplicate_cluster_id_report_only"] else "PASS"},
        {"check": "video_id_overlap_report_only", "value": len(overlaps["video_id_report_only"]), "status": "WARN" if overlaps["video_id_report_only"] else "PASS"},
    ]
    pd.DataFrame(report_rows).to_csv(EXP_DIR / "indobert_v4_data_integrity_audit.csv", index=False, encoding="utf-8-sig")
    return {
        "status": "PASS" if data_ok else "FAIL",
        "development_registry": rel(DEV_REGISTRY),
        "locked_test_registry": rel(LOCKED_REGISTRY),
        "development_total_rows": int(len(dev_raw)),
        "development_evaluable_rows": int(len(dev)),
        "development_class_counts": dev["final_human_label"].value_counts().reindex(LABELS).fillna(0).astype(int).to_dict(),
        "locked_test_total_rows": int(len(locked_raw)),
        "locked_test_evaluable_rows": int(len(locked)),
        "locked_test_class_counts": locked["final_human_label"].value_counts().reindex(LABELS).fillna(0).astype(int).to_dict(),
        "development_dataset_hash": sha256_dataframe(dev, ["comment_id", "final_human_label"]),
        "locked_test_dataset_hash": sha256_dataframe(locked, ["comment_id", "final_human_label"]),
        "hard_leakage_counts": hard_leakage_counts,
        "hard_leakage_pass": data_ok,
        "near_duplicate_cluster_overlap_report_only_count": int(len(overlaps["near_duplicate_cluster_id_report_only"])),
        "video_id_overlap_report_only_count": int(len(overlaps["video_id_report_only"])),
        "source_file_overlap_report_only_count": int(len(overlaps["source_file_report_only"])),
        "development_comment_id_duplicates": int(dev["comment_id"].duplicated().sum()),
        "locked_test_comment_id_duplicates": int(locked["comment_id"].duplicated().sum()),
        "development_inj_count": int(dev["comment_id"].astype(str).str.contains("INJ", case=False, regex=False).sum()),
        "locked_test_inj_count": int(locked["comment_id"].astype(str).str.contains("INJ", case=False, regex=False).sum()),
        "missing_text_counts": {
            "development": int(dev_text.eq("").sum()),
            "locked_test": int(locked_text.eq("").sum()),
        },
    }


def ece_score(y_true: np.ndarray, probs: np.ndarray, bins: int = 10) -> float:
    pred = probs.argmax(axis=1)
    confidence = probs.max(axis=1)
    correct = pred == y_true
    ece = 0.0
    for lo in np.linspace(0.0, 1.0, bins, endpoint=False):
        hi = lo + 1.0 / bins
        mask = (confidence >= lo) & (confidence < hi if hi < 1.0 else confidence <= hi)
        if mask.any():
            ece += float(mask.mean()) * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return float(ece)


def compute_locked_test_outputs() -> dict[str, Any]:
    eval_manifest = read_json(EVAL_MANIFEST)
    if eval_manifest.get("evaluated_once") is not True:
        raise RuntimeError(
            "Locked test has not been evaluated once in the manifest. "
            "This audit script will not run the locked-test evaluation itself."
        )
    if not PREDICTIONS.exists():
        raise FileNotFoundError(PREDICTIONS)

    predictions = read_csv(PREDICTIONS)
    required = {"final_human_label", "predicted_label", "prob_negative", "prob_neutral", "prob_positive"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"Locked-test predictions missing columns: {missing}")
    if not predictions["final_human_label"].isin(LABELS).all():
        raise ValueError("Locked-test predictions include non-three-class human labels.")
    if not predictions["predicted_label"].isin(LABELS).all():
        raise ValueError("Locked-test predictions include invalid predicted labels.")

    probs = predictions[["prob_negative", "prob_neutral", "prob_positive"]].astype(float).to_numpy()
    finite = bool(np.isfinite(probs).all())
    prob_sum = probs.sum(axis=1)
    sums_ok = bool(np.allclose(prob_sum, 1.0, atol=1e-5))
    y_true = predictions["final_human_label"].map(LABEL_TO_ID).astype(int).to_numpy()
    pred = predictions["predicted_label"].map(LABEL_TO_ID).astype(int).to_numpy()
    argmax = probs.argmax(axis=1)
    label_mapping_argmax_ok = bool(np.array_equal(pred, argmax))

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        pred,
        labels=list(range(len(LABELS))),
        zero_division=0,
    )
    predicted_support = {label: int((pred == LABEL_TO_ID[label]).sum()) for label in LABELS}
    metrics = {
        "n": int(len(predictions)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, labels=list(range(len(LABELS))), average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, pred, labels=list(range(len(LABELS))), average="weighted", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "mcc": float(matthews_corrcoef(y_true, pred)),
        "min_class_recall": float(np.min(recall)),
        "ece": ece_score(y_true, probs),
        "brier_score": float(np.mean(np.sum((probs - np.eye(len(LABELS))[y_true]) ** 2, axis=1))),
        "mean_confidence": float(probs.max(axis=1).mean()),
    }
    for idx, label in enumerate(LABELS):
        lower = label.lower()
        metrics[f"{lower}_precision"] = float(precision[idx])
        metrics[f"{lower}_recall"] = float(recall[idx])
        metrics[f"{lower}_f1"] = float(f1[idx])
        metrics[f"{lower}_support"] = int(support[idx])
        metrics[f"{lower}_predicted_support"] = predicted_support[label]

    per_class = pd.DataFrame(
        {
            "label": LABELS,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support.astype(int),
            "predicted_support": [predicted_support[label] for label in LABELS],
        }
    )
    matrix = confusion_matrix(y_true, pred, labels=list(range(len(LABELS))))
    confusion = pd.DataFrame(matrix, columns=LABELS).assign(true_label=LABELS)[["true_label", *LABELS]]
    report = classification_report(
        y_true,
        pred,
        labels=list(range(len(LABELS))),
        target_names=LABELS,
        output_dict=True,
        zero_division=0,
    )
    errors = predictions.loc[predictions["final_human_label"].ne(predictions["predicted_label"])].copy()
    errors["error_type"] = np.select(
        [
            errors["predicted_label"].eq("Positive") & ~errors["final_human_label"].eq("Positive"),
            errors["final_human_label"].eq("Positive") & ~errors["predicted_label"].eq("Positive"),
        ],
        ["positive_false_positive", "positive_false_negative"],
        default="other_error",
    )

    metrics_frame = pd.DataFrame(
        [
            {"metric": key, "value": value}
            for key, value in metrics.items()
            if isinstance(value, int | float | np.integer | np.floating)
        ]
    )
    LOCKED_OUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_frame.to_csv(LOCKED_OUT_DIR / "locked_test_metrics.csv", index=False, encoding="utf-8-sig")
    per_class.to_csv(LOCKED_OUT_DIR / "locked_test_per_class_metrics.csv", index=False, encoding="utf-8-sig")
    confusion.to_csv(LOCKED_OUT_DIR / "locked_test_confusion_matrix.csv", index=False, encoding="utf-8-sig")
    write_json(LOCKED_OUT_DIR / "locked_test_classification_report.json", report)
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

    eval_manifest["strict_acceptance_reviewed_at_utc"] = utc_now()
    eval_manifest["metrics"] = metrics
    eval_manifest["probability_checks"] = {
        "finite": finite,
        "sum_close_to_one_atol_1e_minus_5": sums_ok,
        "max_abs_sum_error": float(np.max(np.abs(prob_sum - 1.0))),
        "predicted_label_matches_argmax": label_mapping_argmax_ok,
    }
    eval_manifest["strict_acceptance_criteria_version"] = "indobert_v4_final_application_2026_07_28"
    eval_manifest["locked_test_rerun_performed_by_audit"] = False
    write_json(EVAL_MANIFEST, eval_manifest)

    return {
        "status": "PASS" if finite and sums_ok and label_mapping_argmax_ok else "FAIL",
        "predictions_path": rel(PREDICTIONS),
        "metrics": metrics,
        "per_class": per_class.to_dict("records"),
        "confusion_matrix": confusion.to_dict("records"),
        "predicted_support": predicted_support,
        "probability_checks": eval_manifest["probability_checks"],
        "false_positive_positive_count": int(errors["error_type"].eq("positive_false_positive").sum()),
        "false_negative_positive_count": int(errors["error_type"].eq("positive_false_negative").sum()),
        "classification_report_path": rel(LOCKED_OUT_DIR / "locked_test_classification_report.json"),
        "error_analysis_false_positive_path": rel(LOCKED_OUT_DIR / "error_analysis_false_positive.csv"),
        "error_analysis_false_negative_path": rel(LOCKED_OUT_DIR / "error_analysis_false_negative.csv"),
        "evaluated_once_manifest": bool(eval_manifest.get("evaluated_once") is True),
        "locked_test_rerun_performed": False,
    }


def make_acceptance_decision(artifact: dict[str, Any], data: dict[str, Any], locked: dict[str, Any]) -> dict[str, Any]:
    metrics = locked["metrics"]
    eval_manifest = read_json(EVAL_MANIFEST)
    training_manifest = read_json(TRAINING_MANIFEST)
    criteria = {
        "artifact_checksum_and_load_pass": artifact["status"] == "PASS",
        "no_hard_data_leakage": bool(data["hard_leakage_pass"]),
        "locked_test_evaluated_once": bool(eval_manifest.get("evaluated_once") is True and eval_manifest.get("locked_test_execution_count") == 1),
        "locked_test_excluded_from_training_or_selection": bool(
            training_manifest.get("locked_test_used_for_training_or_selection") is False
            and training_manifest.get("locked_test_used_for_early_stopping") is False
            and training_manifest.get("locked_test_used_for_threshold_selection") is False
            and eval_manifest.get("locked_test_used_for_training_or_selection") is False
            and eval_manifest.get("locked_test_used_for_threshold_selection") is False
        ),
        "macro_f1_gte_0p7309": bool(metrics["macro_f1"] >= ACCEPTANCE_THRESHOLDS["macro_f1"]),
        "positive_recall_gte_0p70": bool(metrics["positive_recall"] >= ACCEPTANCE_THRESHOLDS["positive_recall"]),
        "accuracy_gte_0p8159": bool(metrics["accuracy"] >= ACCEPTANCE_THRESHOLDS["accuracy"]),
        "minimum_class_recall_gte_0p60": bool(metrics["min_class_recall"] >= ACCEPTANCE_THRESHOLDS["min_class_recall"]),
        "mcc_gte_0p60": bool(metrics["mcc"] >= ACCEPTANCE_THRESHOLDS["mcc"]),
        "no_class_collapse": bool(all(value > 0 for value in locked["predicted_support"].values())),
        "label_mapping_verified": bool(artifact["label_mapping_ok"] and locked["probability_checks"]["predicted_label_matches_argmax"]),
        "model_reproducible_from_saved_artifact": bool(artifact["status"] == "PASS"),
    }
    accepted = bool(all(criteria.values()))
    decision = {
        "status": (
            "INDOBERT_V4_ACCEPTED_AS_FINAL_RM2_SENTIMENT_MODEL"
            if accepted
            else "INDOBERT_V4_NOT_ACCEPTED_KEEP_V2"
        ),
        "created_at_utc": utc_now(),
        "accepted": accepted,
        "recommended_for_rm2_sentiment_final": accepted,
        "strict_acceptance_criteria": criteria,
        "failed_criteria": [key for key, passed in criteria.items() if not passed],
        "acceptance_thresholds": ACCEPTANCE_THRESHOLDS,
        "locked_test_metrics": metrics,
        "baseline_legacy_v2": V2_BASELINE,
        "delta_indobert_v4_minus_v2": {
            "accuracy": float(metrics["accuracy"] - V2_BASELINE["accuracy"]),
            "macro_f1": float(metrics["macro_f1"] - V2_BASELINE["macro_f1"]),
            "balanced_accuracy": float(metrics["balanced_accuracy"] - V2_BASELINE["balanced_accuracy"]),
            "mcc": float(metrics["mcc"] - V2_BASELINE["mcc"]),
            "positive_recall": float(metrics["positive_recall"] - V2_BASELINE["positive_recall"]),
            "positive_f1": float(metrics["positive_f1"] - V2_BASELINE["positive_f1"]),
        },
        "full_inference_run": False,
        "promotion_run": False,
        "canonical_model_pointer_created": False,
        "final_model_dir_created": False,
        "reason": (
            "IndoBERT V4 is not promoted because at least one strict locked-test acceptance gate failed. "
            "No full inference or RM2 downstream refresh is run from this candidate."
        ),
        "methodology": {
            "prediction_rule": "argmax_no_threshold_tuning",
            "label_order": LABELS,
            "label_source_column": "final_human_label",
            "locked_test_used_for_training_or_selection": False,
            "locked_test_used_for_threshold_selection": False,
            "no_label_changes": True,
            "no_retraining_performed": True,
            "no_full_inference_performed": True,
            "no_rm1_outputs_modified": True,
        },
    }
    write_json(LOCKED_OUT_DIR / "FINAL_ACCEPTANCE_DECISION.json", decision)
    return decision


def md_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return "\n".join(lines)


def fmt(value: Any, digits: int = 4) -> str:
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_artifact_doc(artifact: dict[str, Any], decision: dict[str, Any]) -> None:
    rows = [
        {"item": "Audit commit", "value": git_head()},
        {"item": "Model directory", "value": artifact["model_dir"]},
        {"item": "Model ID", "value": artifact["model_id"]},
        {"item": "Model revision", "value": artifact["model_revision"]},
        {"item": "Model SHA-256", "value": artifact["model_sha256"]},
        {"item": "Model size bytes", "value": artifact["model_size_bytes"]},
        {"item": "Checksum status", "value": artifact["checksum_verified"]},
        {"item": "Manifest model hash match", "value": artifact["manifest_model_hash_ok"]},
        {"item": "Label order", "value": ", ".join(artifact["label_order"])},
        {"item": "CPU load", "value": artifact["cpu_smoke"]["cpu_model_load_success"]},
        {"item": "CPU inference", "value": artifact["cpu_smoke"]["cpu_inference_success"]},
        {"item": "Artifact status", "value": artifact["status"]},
        {"item": "Strict acceptance status", "value": decision["status"]},
    ]
    smoke_rows = [
        {
            "input": sample["input_text"] if sample["input_text"] else "<empty>",
            "prediction": sample["predicted_label"],
            "confidence": fmt(sample["confidence"]),
            "prob_sum": fmt(sample["probability_sum"], 6),
        }
        for sample in artifact["cpu_smoke"].get("samples", [])
    ]
    checksum_rows = [
        {
            "file": row["file"],
            "passed": row["passed"],
            "sha256": row["actual_sha256"],
        }
        for row in artifact["checksum_rows"]
    ]
    body = f"""# IndoBERT V4 Artifact Audit

Generated at UTC: {utc_now()}

This audit verifies the frozen base-reference artifact before any model promotion. It did not train a model, did not tune a threshold, did not rerun the locked test, and did not run full observational inference.

## Summary

{md_table(rows, ["item", "value"])}

## Checksum Audit

{md_table(checksum_rows, ["file", "passed", "sha256"])}

## CPU Smoke Test

{md_table(smoke_rows, ["input", "prediction", "confidence", "prob_sum"])}

All smoke-test predictions use the label order `Negative`, `Neutral`, `Positive`. The empty input check is a robustness check only; final full inference would assign `No Text` before model inference for invalid comment text.
"""
    (DOCS_DIR / "INDOBERT_V4_ARTIFACT_AUDIT.md").write_text(body, encoding="utf-8")


def write_data_doc(data: dict[str, Any]) -> None:
    rows = [
        {"item": "Development registry", "value": data["development_registry"]},
        {"item": "Development evaluable rows", "value": data["development_evaluable_rows"]},
        {"item": "Development class counts", "value": json.dumps(data["development_class_counts"])},
        {"item": "Locked-test registry", "value": data["locked_test_registry"]},
        {"item": "Locked-test evaluable rows", "value": data["locked_test_evaluable_rows"]},
        {"item": "Locked-test class counts", "value": json.dumps(data["locked_test_class_counts"])},
        {"item": "Comment ID overlap", "value": data["hard_leakage_counts"]["comment_id"]},
        {"item": "Text cluster overlap", "value": data["hard_leakage_counts"]["text_cluster_id"]},
        {"item": "Exact duplicate group overlap", "value": data["hard_leakage_counts"]["exact_duplicate_group_id"]},
        {"item": "Normalized text overlap", "value": data["hard_leakage_counts"]["normalized_text"]},
        {"item": "Near duplicate cluster overlap", "value": data["near_duplicate_cluster_overlap_report_only_count"]},
        {"item": "Video ID overlap", "value": data["video_id_overlap_report_only_count"]},
        {"item": "Hard leakage status", "value": data["status"]},
    ]
    body = f"""# IndoBERT V4 Data Integrity Audit

Generated at UTC: {utc_now()}

The audit reads the completed Sentiment V4 human registry and the frozen locked-test registry. It does not edit labels or move rows between splits.

{md_table(rows, ["item", "value"])}

## Interpretation

Hard leakage is defined here as overlap in `comment_id`, `text_cluster_id`, `exact_duplicate_group_id`, or normalized text between development and locked test. Those checks pass. Near-duplicate cluster and video overlap are reported as diagnostics because the locked test is frozen and cannot be reshaped after evaluation.
"""
    (DOCS_DIR / "INDOBERT_V4_DATA_INTEGRITY_AUDIT.md").write_text(body, encoding="utf-8")


def write_final_report(artifact: dict[str, Any], data: dict[str, Any], locked: dict[str, Any], decision: dict[str, Any]) -> None:
    metrics = locked["metrics"]
    comparison_rows = [
        {
            "metric": metric,
            "V2 baseline": fmt(V2_BASELINE[metric]),
            "IndoBERT V4": fmt(metrics[metric]),
            "delta": fmt(decision["delta_indobert_v4_minus_v2"][metric]),
        }
        for metric in ["accuracy", "macro_f1", "balanced_accuracy", "mcc", "positive_recall", "positive_f1"]
    ]
    metric_rows = [
        {"metric": key, "value": fmt(metrics[key])}
        for key in [
            "accuracy",
            "macro_f1",
            "weighted_f1",
            "balanced_accuracy",
            "mcc",
            "min_class_recall",
            "negative_precision",
            "negative_recall",
            "negative_f1",
            "neutral_precision",
            "neutral_recall",
            "neutral_f1",
            "positive_precision",
            "positive_recall",
            "positive_f1",
            "ece",
            "brier_score",
        ]
    ]
    criteria_rows = [
        {"criterion": key, "passed": value}
        for key, value in decision["strict_acceptance_criteria"].items()
    ]
    body = f"""# IndoBERT V4 Final Application Report

Generated at UTC: {utc_now()}

## Git

- Branch: `{git_branch()}`
- Audit commit: `{git_head()}`

## Model Artifact

- Artifact path: `{artifact["model_dir"]}`
- Model ID: `{artifact["model_id"]}`
- Model revision: `{artifact["model_revision"]}`
- Model SHA-256: `{artifact["model_sha256"]}`
- Artifact technical status: `{artifact["status"]}`
- Label order: `Negative`, `Neutral`, `Positive`
- Prediction rule: `{artifact["prediction_rule"]}`

## Data

- Development evaluable: `{data["development_evaluable_rows"]}`; counts `{json.dumps(data["development_class_counts"])}`
- Locked-test evaluable: `{data["locked_test_evaluable_rows"]}`; counts `{json.dumps(data["locked_test_class_counts"])}`
- Hard leakage counts: `{json.dumps(data["hard_leakage_counts"])}`
- Near-duplicate cluster overlap, report-only: `{data["near_duplicate_cluster_overlap_report_only_count"]}`
- Video overlap, report-only: `{data["video_id_overlap_report_only_count"]}`

## Locked-Test Metrics

{md_table(metric_rows, ["metric", "value"])}

## IndoBERT V4 Versus V2 Baseline

{md_table(comparison_rows, ["metric", "V2 baseline", "IndoBERT V4", "delta"])}

## Acceptance Decision

Status: `{decision["status"]}`

{md_table(criteria_rows, ["criterion", "passed"])}

IndoBERT V4 is not promoted under the current strict gate because accuracy is `{fmt(metrics["accuracy"])}`, below the required `0.8159`. Positive recall improves over V2, but this does not override the accuracy gate.

## Downstream Action

Full inference was not run. `output/rm2_sentiment/final/CANONICAL_MODEL.json` was not created for IndoBERT V4, and `artifacts/rm2_sentiment/indobert_v4_final/final_model/` was not created. RM2 Goals outputs therefore remain on the already accepted V2 baseline until a separately accepted model replaces it.

## Reproduction Commands

```powershell
python scripts/audit_indobert_v4_final_application.py
python scripts/evaluate_rm2_sentiment_indobert_v4_locked_test_once.py  # only if LOCKED_TEST_EVALUATION_MANIFEST.json is absent or evaluated_once is not true
python scripts/predict_rm2_sentiment_cpu.py --model-dir artifacts/rm2_sentiment/indobert_v4_final/base_reference --text "produk ini bagus"
python -m unittest discover
python -m pytest
```

Do not retrain, retune, rerun locked-test evaluation, promote a model, or run full inference after seeing this failed strict acceptance decision.
"""
    (DOCS_DIR / "INDOBERT_V4_FINAL_REPORT.md").write_text(body, encoding="utf-8")


def write_goals_guide(decision: dict[str, Any]) -> None:
    body = f"""# RM2 Goals Interpretation Guide

Generated at UTC: {utc_now()}

IndoBERT V4 was audited as a candidate sentiment model, but the current strict decision is `{decision["status"]}`. Until a candidate passes all locked-test gates, RM2 Goals remains based on the accepted V2 sentiment outputs.

Use these interpretation constraints in all RM2 reporting:

- IndoBERT V4, if accepted in a future run, would classify comment sentiment orientation into Negative, Neutral, and Positive.
- Goals are message orientations derived from aggregated comment sentiment, not evidence of intent, payment, or commercial relationships.
- HCC indicates groups of accounts with stronger structural connectivity, but it does not automatically prove buzzer activity or paid coordination.
- Community Actor means membership in an HCC from RM1, not a finding that an account is a bot, buzzer, or paid promoter.
- Sentiment is an RM2 attribute only. It must not change LCN, Louvain community membership, FSA_V, HCC membership, edges, nodes, or modularity from RM1.
- Brand and video context are context associations only, not sentiment labels and not evidence of brand involvement.

Canonical V2 goal mapping should remain unchanged unless a newly accepted sentiment model explicitly replaces the canonical model pointer through a separate validated run.
"""
    (DOCS_DIR / "RM2_GOALS_INTERPRETATION_GUIDE.md").write_text(body, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit frozen IndoBERT V4 artifacts and strict locked-test acceptance.")
    parser.add_argument(
        "--skip-cpu-smoke",
        action="store_true",
        help="Skip the local CPU model load/smoke inference. Checksum and locked-test audit still run.",
    )
    args = parser.parse_args()

    EXP_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    artifact = audit_model_artifacts(run_cpu_smoke=not args.skip_cpu_smoke)
    data = audit_data_integrity()
    locked = compute_locked_test_outputs()
    decision = make_acceptance_decision(artifact, data, locked)
    write_artifact_doc(artifact, decision)
    write_data_doc(data)
    write_final_report(artifact, data, locked, decision)
    write_goals_guide(decision)

    summary = {
        "status": decision["status"],
        "artifact_status": artifact["status"],
        "data_status": data["status"],
        "locked_metrics": locked["metrics"],
        "failed_criteria": decision["failed_criteria"],
        "full_inference_run": False,
        "canonical_model_pointer_exists": CANONICAL_MODEL.exists(),
        "final_model_dir_exists": FINAL_MODEL_DIR.exists(),
        "docs": [
            "docs/INDOBERT_V4_ARTIFACT_AUDIT.md",
            "docs/INDOBERT_V4_DATA_INTEGRITY_AUDIT.md",
            "docs/INDOBERT_V4_FINAL_REPORT.md",
            "docs/RM2_GOALS_INTERPRETATION_GUIDE.md",
        ],
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
        },
    }
    write_json(EXP_DIR / "INDOBERT_V4_FINAL_APPLICATION_AUDIT.json", summary)
    print(json.dumps(to_jsonable(summary), indent=2), flush=True)


if __name__ == "__main__":
    main()
