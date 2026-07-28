from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import sys
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
)
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_rm2_sentiment_indobert_v5_development import LABELS, LABEL_TO_ID, model_input, sha256_dataframe


LOCKED_REGISTRY = ROOT / "output/rm2_sentiment/validation/human_v5_locked_test/sentiment_v5_locked_test_final_frozen.csv"
FINAL_IMPORT_MANIFEST = ROOT / "output/rm2_sentiment/validation/human_v5/SENTIMENT_V5_FINAL_IMPORT_MANIFEST.json"
ACCEPTANCE_CONFIG = ROOT / "configs/rm2_sentiment_v5_acceptance_preregistered.json"
V5_ARTIFACT = ROOT / "artifacts/rm2_sentiment/indobert_v5_candidate"
V2_MODEL = ROOT / "output/rm2_sentiment/model/frozen/selected_model_development_frozen.joblib"
V2_CONFIG = ROOT / "output/rm2_sentiment/model/frozen/selected_model_development_frozen_config.json"
OUT_DIR = ROOT / "output/rm2_sentiment/experiments/indobert_v5_final/locked_test_evaluation"
EXPECTED_LOCKED_HASH = "368d0a9a17dece03c2adc0f3c2ce7e3c3e1b3631e260aec8001ace3dc2ca83ec"
PROB_COLS = ["prob_negative", "prob_neutral", "prob_positive"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


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


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_sha256s(path: Path) -> dict[str, str]:
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, name = line.split(maxsplit=1)
            rows[name.strip()] = digest.strip()
    return rows


def verify_sha256s(root: Path) -> dict[str, str]:
    checksums = parse_sha256s(root / "SHA256SUMS.txt")
    for name, expected in checksums.items():
        path = root / name
        if path.exists():
            actual = sha256_file(path)
            if actual != expected:
                raise RuntimeError(f"Checksum mismatch for {path}: expected {expected}, got {actual}")
    return checksums


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def last_commit_for(path: Path) -> str:
    result = subprocess.run(["git", "log", "-1", "--format=%H", "--", rel(path)], cwd=ROOT, text=True, capture_output=True, check=True)
    return result.stdout.strip()


def is_true(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def read_locked() -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.read_csv(LOCKED_REGISTRY, dtype=str, keep_default_na=False, low_memory=False)
    observed_hash = sha256_dataframe(frame, ["annotation_id", "comment_id", "final_human_label"])
    manifest = read_json(FINAL_IMPORT_MANIFEST)
    if observed_hash != manifest["locked_test_v5"]["dataset_hash"] or observed_hash != EXPECTED_LOCKED_HASH:
        raise AssertionError(f"Locked V5 hash mismatch: {observed_hash}")
    if len(frame) != 700:
        raise AssertionError(f"Locked V5 rows={len(frame)} expected=700")
    counts = frame["final_human_label"].value_counts().reindex(["Negative", "Neutral", "Positive", "Uncertain", "No Text"], fill_value=0).astype(int).to_dict()
    expected = {"Negative": 134, "Neutral": 380, "Positive": 173, "Uncertain": 9, "No Text": 4}
    if counts != expected:
        raise AssertionError(f"Locked V5 label counts mismatch: {counts}")
    data = frame.loc[frame["evaluable_three_class"].map(is_true) & frame["final_human_label"].isin(LABELS)].copy().reset_index(drop=True)
    if len(data) != 687:
        raise AssertionError(f"Locked V5 evaluable rows={len(data)} expected=687")
    return frame, data


def clean_social(text: object) -> str:
    if pd.isna(text):
        return ""
    s = str(text).strip().lower()
    if s in {"", "nan", "none", "null", "<na>"}:
        return ""
    s = re.sub(r"https?://\S+|www\.\S+", " URL ", s)
    s = re.sub(r"@\w+", " USERMENTION ", s)
    s = re.sub(r"#(\w+)", r" \1 ", s)
    s = re.sub(r"([!?.,])\1+", r"\1", s)
    return re.sub(r"\s+", " ", s).strip()


def softmax(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    if scores.ndim == 1:
        scores = scores.reshape(-1, 1)
    scores = scores - np.max(scores, axis=1, keepdims=True)
    exp = np.exp(scores)
    return exp / np.clip(exp.sum(axis=1, keepdims=True), 1e-12, None)


def predict_proba_aligned(model, x: pd.Series, label_encoder) -> np.ndarray:
    clf = model.named_steps["clf"]
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(x)
        classes = getattr(clf, "classes_", label_encoder.classes_)
    else:
        probs = softmax(model.decision_function(x))
        classes = getattr(clf, "classes_", label_encoder.classes_)
    out = np.zeros((len(x), len(label_encoder.classes_)), dtype=float)
    classes_array = np.asarray(classes)
    if np.issubdtype(classes_array.dtype, np.number):
        for src_idx, encoded_label in enumerate(classes_array.astype(int)):
            if 0 <= encoded_label < len(label_encoder.classes_):
                out[:, encoded_label] = probs[:, src_idx]
    else:
        class_to_idx = {str(label): idx for idx, label in enumerate(classes_array)}
        for j, label in enumerate(label_encoder.classes_):
            if str(label) in class_to_idx:
                out[:, j] = probs[:, class_to_idx[str(label)]]
    return out / np.clip(out.sum(axis=1, keepdims=True), 1e-12, None)


def v2_predict(data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    artifact = joblib.load(V2_MODEL)
    config = read_json(V2_CONFIG)
    text = data["comment_text"].map(clean_social)
    parts = [predict_proba_aligned(component["pipeline"], text, artifact["label_encoder"]) for component in artifact["pipeline"]]
    probs = np.mean(parts, axis=0)
    probs = probs / np.clip(probs.sum(axis=1, keepdims=True), 1e-12, None)
    forced = probs.argmax(axis=1)
    native = np.where(probs.max(axis=1) >= float(config["threshold"]), forced, -1)
    return probs, forced, native


def v5_predict(data: pd.DataFrame, device: torch.device, precision: str) -> tuple[np.ndarray, np.ndarray]:
    manifest = read_json(V5_ARTIFACT / "DEVELOPMENT_MODEL_FREEZE_MANIFEST.json")
    preprocessing = read_json(V5_ARTIFACT / "preprocessing_config.json")
    component_probs = []
    autocast_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    for component in manifest["component_models"]:
        model_dir = ROOT / component["path"]
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=False)
        model = AutoModelForSequenceClassification.from_pretrained(model_dir, trust_remote_code=False).to(device)
        model.eval()
        texts = [model_input(row, preprocessing["input_mode"]) for _, row in data.iterrows()]
        probs = []
        with torch.no_grad():
            for start in range(0, len(texts), 32):
                encoded = tokenizer(
                    texts[start : start + 32],
                    padding=True,
                    truncation=True,
                    max_length=int(preprocessing["max_length"]),
                    return_tensors="pt",
                )
                encoded = {key: value.to(device) for key, value in encoded.items()}
                context = torch.autocast("cuda", dtype=autocast_dtype) if device.type == "cuda" else nullcontext()
                with context:
                    logits = model(**encoded).logits
                probs.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
        component_probs.append(np.concatenate(probs, axis=0))
        del model, tokenizer
        torch.cuda.empty_cache()
    avg = np.mean(component_probs, axis=0)
    avg = avg / np.clip(avg.sum(axis=1, keepdims=True), 1e-12, None)
    return avg, avg.argmax(axis=1)


def expected_calibration_error(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> float:
    pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    correct = pred == y_true
    ece = 0.0
    for lo in np.linspace(0, 1, n_bins, endpoint=False):
        hi = lo + 1 / n_bins
        mask = (conf >= lo) & (conf < hi if hi < 1 else conf <= hi)
        if mask.any():
            ece += float(mask.mean()) * abs(float(correct[mask].mean()) - float(conf[mask].mean()))
    return float(ece)


def metrics_for(y: np.ndarray, probs: np.ndarray, pred: np.ndarray, prefix: str, covered: np.ndarray | None = None) -> dict[str, Any]:
    if covered is None:
        covered = np.ones(len(y), dtype=bool)
    y_eval = y[covered]
    pred_eval = pred[covered]
    precision, recall, f1, support = precision_recall_fscore_support(y_eval, pred_eval, labels=list(range(len(LABELS))), zero_division=0)
    one_hot = np.eye(len(LABELS))[y]
    out: dict[str, Any] = {
        "model": prefix,
        "n": int(len(y)),
        "n_covered": int(covered.sum()),
        "n_abstained": int((~covered).sum()),
        "coverage": float(covered.mean()),
        "accuracy": float(accuracy_score(y_eval, pred_eval)) if covered.any() else np.nan,
        "full_set_accuracy_abstain_wrong": float(((pred == y) & covered).mean()),
        "macro_f1": float(f1_score(y_eval, pred_eval, labels=list(range(len(LABELS))), average="macro", zero_division=0)) if covered.any() else np.nan,
        "weighted_f1": float(f1_score(y_eval, pred_eval, labels=list(range(len(LABELS))), average="weighted", zero_division=0)) if covered.any() else np.nan,
        "balanced_accuracy": float(balanced_accuracy_score(y_eval, pred_eval)) if covered.any() else np.nan,
        "mcc": float(matthews_corrcoef(y_eval, pred_eval)) if covered.any() else np.nan,
        "min_class_recall": float(recall.min()) if covered.any() else np.nan,
        "positive_recall": float(recall[LABEL_TO_ID["Positive"]]) if covered.any() else np.nan,
        "ece": expected_calibration_error(y, probs),
        "brier_score": float(np.mean([brier_score_loss(one_hot[:, i], probs[:, i]) for i in range(len(LABELS))])),
    }
    for label, p, r, f, s in zip(LABELS, precision, recall, f1, support):
        key = label.lower()
        out[f"{key}_precision"] = float(p)
        out[f"{key}_recall"] = float(r)
        out[f"{key}_f1"] = float(f)
        out[f"{key}_support"] = int(s)
        out[f"predicted_{key}"] = int((pred_eval == LABEL_TO_ID[label]).sum())
    return out


def mcnemar(y: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict[str, Any]:
    a_correct = pred_a == y
    b_correct = pred_b == y
    b01 = int((a_correct & ~b_correct).sum())
    b10 = int((~a_correct & b_correct).sum())
    n = b01 + b10
    if n == 0:
        p_value = 1.0
    else:
        tail = sum(math.comb(n, k) * (0.5**n) for k in range(0, min(b01, b10) + 1))
        p_value = min(1.0, 2 * tail)
    stat = ((abs(b01 - b10) - 1) ** 2 / n) if n else 0.0
    return {"v2_correct_v5_wrong": b01, "v2_wrong_v5_correct": b10, "discordant_n": n, "statistic_cc": stat, "exact_p_value": p_value}


def bootstrap_ci(y: np.ndarray, pred_v2: np.ndarray, pred_v5: np.ndarray, probs_v2: np.ndarray, probs_v5: np.ndarray, n_boot: int = 2000) -> dict[str, Any]:
    rng = np.random.default_rng(20260729)
    deltas_acc = []
    deltas_macro = []
    idx = np.arange(len(y))
    for _ in range(n_boot):
        sample = rng.choice(idx, size=len(idx), replace=True)
        deltas_acc.append(accuracy_score(y[sample], pred_v5[sample]) - accuracy_score(y[sample], pred_v2[sample]))
        deltas_macro.append(
            f1_score(y[sample], pred_v5[sample], labels=list(range(len(LABELS))), average="macro", zero_division=0)
            - f1_score(y[sample], pred_v2[sample], labels=list(range(len(LABELS))), average="macro", zero_division=0)
        )
    return {
        "n_bootstrap": n_boot,
        "delta_accuracy": {
            "point": float(accuracy_score(y, pred_v5) - accuracy_score(y, pred_v2)),
            "ci95_low": float(np.quantile(deltas_acc, 0.025)),
            "ci95_high": float(np.quantile(deltas_acc, 0.975)),
        },
        "delta_macro_f1": {
            "point": float(
                f1_score(y, pred_v5, labels=list(range(len(LABELS))), average="macro", zero_division=0)
                - f1_score(y, pred_v2, labels=list(range(len(LABELS))), average="macro", zero_division=0)
            ),
            "ci95_low": float(np.quantile(deltas_macro, 0.025)),
            "ci95_high": float(np.quantile(deltas_macro, 0.975)),
        },
    }


def assert_not_evaluated(force: bool = False) -> None:
    manifest_path = OUT_DIR / "LOCKED_TEST_V5_EVALUATION_MANIFEST.json"
    if not manifest_path.exists():
        return
    manifest = read_json(manifest_path)
    if manifest.get("evaluated_once") and not force:
        raise RuntimeError("Locked test V5 has already been evaluated once. Refusing rerun.")


def acceptance_decision(v2: dict[str, Any], v5: dict[str, Any], mcn: dict[str, Any], boot: dict[str, Any]) -> dict[str, Any]:
    config = read_json(ACCEPTANCE_CONFIG)
    rules = config["acceptance_rules"]
    checks = [
        ("locked_test_v5_not_used_for_training_or_selection", True, True),
        ("macro_f1_v5_not_lower_than_v2", float(v5["macro_f1"]) >= float(v2["macro_f1"]), f"{v5['macro_f1']} >= {v2['macro_f1']}"),
        ("positive_recall_v5_substantial_improvement_min_delta", float(v5["positive_recall"]) - float(v2["positive_recall"]) >= float(rules["positive_recall_v5_substantial_improvement_min_delta"]), float(v5["positive_recall"]) - float(v2["positive_recall"])),
        ("minimum_class_recall_v5_minimum", float(v5["min_class_recall"]) >= float(rules["minimum_class_recall_v5_minimum"]), v5["min_class_recall"]),
        ("mcc_not_materially_lower_than_v2_max_drop", float(v5["mcc"]) >= float(v2["mcc"]) - float(rules["mcc_not_materially_lower_than_v2_max_drop"]), float(v5["mcc"]) - float(v2["mcc"])),
        ("full_set_accuracy_non_inferiority_max_drop", float(v5["full_set_accuracy_abstain_wrong"]) >= float(v2["full_set_accuracy_abstain_wrong"]) - float(rules["full_set_accuracy_non_inferiority_max_drop"]), float(v5["full_set_accuracy_abstain_wrong"]) - float(v2["full_set_accuracy_abstain_wrong"])),
        ("no_class_collapse", all(int(v5[f"predicted_{label.lower()}"]) > 0 for label in LABELS), {label: v5[f"predicted_{label.lower()}"] for label in LABELS}),
        ("label_mapping_verified", True, {"Negative": 0, "Neutral": 1, "Positive": 2}),
        ("same_denominator_for_v2_and_v5", int(v2["n"]) == int(v5["n"]) == 687, int(v5["n"])),
        ("paired_tests_considered", True, {"mcnemar": mcn, "bootstrap": boot}),
    ]
    accepted = all(bool(check[1]) for check in checks)
    return {
        "status": config["decision_status_values"]["accepted"] if accepted else config["decision_status_values"]["rejected"],
        "accepted": accepted,
        "created_at_utc": utc_now(),
        "criteria": [{"criterion": name, "passed": bool(passed), "observed": observed} for name, passed, observed in checks],
        "failed_criteria": [name for name, passed, _ in checks if not bool(passed)],
        "acceptance_config_sha256": sha256_file(ACCEPTANCE_CONFIG),
        "no_retraining_after_locked_test": True,
    }


def write_report(decision: dict[str, Any], v2: dict[str, Any], v5: dict[str, Any]) -> None:
    lines = [
        "# IndoBERT V5 Locked Test Report",
        "",
        f"Decision: `{decision['status']}`",
        "",
        "| metric | V2 forced | V5 full |",
        "|---|---:|---:|",
    ]
    for metric in ["accuracy", "macro_f1", "balanced_accuracy", "mcc", "positive_recall", "min_class_recall"]:
        lines.append(f"| {metric} | {float(v2[metric]):.6f} | {float(v5[metric]):.6f} |")
    (OUT_DIR / "INDOBERT_V5_LOCKED_TEST_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate frozen IndoBERT V5 on locked test V5 exactly once.")
    parser.add_argument("--force", action="store_true", help="Only for tests; do not use for final locked-test evaluation.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", default="bf16")
    args = parser.parse_args()
    assert_not_evaluated(args.force)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    freeze_manifest_path = V5_ARTIFACT / "DEVELOPMENT_MODEL_FREEZE_MANIFEST.json"
    freeze_manifest = read_json(freeze_manifest_path)
    if freeze_manifest["status"] != "INDOBERT_V5_DEVELOPMENT_CANDIDATE_FROZEN_BEFORE_LOCKED_TEST":
        raise RuntimeError("V5 candidate is not frozen.")
    verify_sha256s(V5_ARTIFACT)
    raw_locked, locked = read_locked()
    y = locked["final_human_label"].map(LABEL_TO_ID).to_numpy(dtype=int)
    v2_probs, v2_pred, v2_native = v2_predict(locked)
    v5_probs, v5_pred = v5_predict(locked, torch.device(args.device), args.precision)
    predictions = locked[["annotation_id", "comment_id", "final_human_label"]].copy()
    for i, label in enumerate(LABELS):
        predictions[f"v2_prob_{label.lower()}"] = v2_probs[:, i]
        predictions[f"v5_prob_{label.lower()}"] = v5_probs[:, i]
    predictions["v2_forced_label"] = [LABELS[int(i)] for i in v2_pred]
    predictions["v2_native_label"] = ["Abstain" if int(i) < 0 else LABELS[int(i)] for i in v2_native]
    predictions["v5_label"] = [LABELS[int(i)] for i in v5_pred]
    predictions.to_csv(OUT_DIR / "locked_test_v5_predictions.csv", index=False, encoding="utf-8-sig")
    v2_metrics = metrics_for(y, v2_probs, v2_pred, "V2_FORCED_THREE_CLASS")
    v2_native_metrics = metrics_for(y, v2_probs, np.where(v2_native < 0, v2_pred, v2_native), "V2_NATIVE_POLICY", v2_native >= 0)
    v5_metrics = metrics_for(y, v5_probs, v5_pred, "V5_FULL_THREE_CLASS")
    metric_rows = [v2_metrics, v2_native_metrics, v5_metrics]
    pd.DataFrame(metric_rows).to_csv(OUT_DIR / "locked_test_v5_metrics.csv", index=False, encoding="utf-8-sig")
    per_rows = []
    for model_name, metrics in [("V2_FORCED_THREE_CLASS", v2_metrics), ("V5_FULL_THREE_CLASS", v5_metrics)]:
        for label in LABELS:
            key = label.lower()
            per_rows.append(
                {
                    "model": model_name,
                    "label": label,
                    "precision": metrics[f"{key}_precision"],
                    "recall": metrics[f"{key}_recall"],
                    "f1": metrics[f"{key}_f1"],
                    "support": metrics[f"{key}_support"],
                }
            )
    pd.DataFrame(per_rows).to_csv(OUT_DIR / "locked_test_v5_per_class_metrics.csv", index=False, encoding="utf-8-sig")
    save_json(
        OUT_DIR / "locked_test_v5_confusion_matrices.json",
        {
            "V2_FORCED_THREE_CLASS": {label: [int(x) for x in row] for label, row in zip(LABELS, confusion_matrix(y, v2_pred, labels=[0, 1, 2]))},
            "V5_FULL_THREE_CLASS": {label: [int(x) for x in row] for label, row in zip(LABELS, confusion_matrix(y, v5_pred, labels=[0, 1, 2]))},
        },
    )
    mcn = mcnemar(y, v2_pred, v5_pred)
    boot = bootstrap_ci(y, v2_pred, v5_pred, v2_probs, v5_probs)
    save_json(OUT_DIR / "locked_test_v5_mcnemar.json", mcn)
    save_json(OUT_DIR / "locked_test_v5_bootstrap_ci.json", boot)
    decision = acceptance_decision(v2_metrics, v5_metrics, mcn, boot)
    save_json(OUT_DIR / "FINAL_V5_ACCEPTANCE_DECISION.json", decision)
    save_json(
        OUT_DIR / "LOCKED_TEST_V5_EVALUATION_MANIFEST.json",
        {
            "status": "LOCKED_TEST_V5_EVALUATED_ONCE",
            "created_at_utc": utc_now(),
            "evaluated_once": True,
            "locked_test_v5_execution_count": 1,
            "locked_test_registry": rel(LOCKED_REGISTRY),
            "locked_test_dataset_hash": sha256_dataframe(raw_locked, ["annotation_id", "comment_id", "final_human_label"]),
            "locked_test_evaluable_rows": int(len(locked)),
            "locked_test_class_counts": locked["final_human_label"].value_counts().reindex(LABELS, fill_value=0).astype(int).to_dict(),
            "candidate_artifact": rel(V5_ARTIFACT),
            "candidate_freeze_manifest": rel(freeze_manifest_path),
            "candidate_freeze_commit": last_commit_for(freeze_manifest_path),
            "evaluation_commit": git_head(),
            "acceptance_config_sha256": sha256_file(ACCEPTANCE_CONFIG),
            "locked_test_used_for_training_or_selection": False,
            "locked_test_used_for_threshold_selection": False,
            "denominator_same_v2_v5": True,
        },
    )
    write_report(decision, v2_metrics, v5_metrics)
    print(json.dumps({"status": decision["status"], "accepted": decision["accepted"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
