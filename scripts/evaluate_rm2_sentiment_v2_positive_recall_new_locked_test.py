from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
)

from rm2_sentiment_v2_positive_recall import (
    LABELS,
    LABEL_TO_ID,
    PROB_COLUMNS,
    apply_threshold_policy,
    artifact_predict_proba,
    no_text_flag,
    normalize_for_model,
    predict_labels_from_artifact,
    predict_proba_aligned,
)


ROOT = Path(__file__).resolve().parents[1]
HUMAN_DIR = ROOT / "output/rm2_sentiment/validation/human_v2_positive_recall"
EXP_DIR = ROOT / "output/rm2_sentiment/experiments/v2_positive_recall"
OUT_DIR = EXP_DIR / "new_locked_test_evaluation"
CANDIDATE_DIR = ROOT / "output/rm2_sentiment/model/v2_positive_recall_candidate"
V2_DIR = ROOT / "output/rm2_sentiment/model/frozen"

LOCKED_FINAL = HUMAN_DIR / "new_locked_test_final.csv"
LOCKED_FREEZE_MANIFEST = HUMAN_DIR / "new_locked_test_freeze_manifest.json"
REGISTRY = HUMAN_DIR / "human_label_registry.csv"
CANDIDATE_MODEL = CANDIDATE_DIR / "selected_model.joblib"
CANDIDATE_CONFIG = CANDIDATE_DIR / "selected_model_config.json"
V2_MODEL = V2_DIR / "selected_model_development_frozen.joblib"
V2_CONFIG = V2_DIR / "selected_model_development_frozen_config.json"

OUT_READINESS = OUT_DIR / "new_locked_test_evaluation_readiness.csv"
OUT_PREDICTIONS = OUT_DIR / "v2_candidate_same_new_locked_test_predictions.csv"
OUT_METRICS = OUT_DIR / "v2_candidate_same_new_locked_test_metrics.csv"
OUT_PER_CLASS = OUT_DIR / "v2_candidate_same_new_locked_test_per_class_metrics.csv"
OUT_CONFUSION = OUT_DIR / "v2_candidate_same_new_locked_test_confusion_matrices.csv"
OUT_BOOTSTRAP = OUT_DIR / "v2_candidate_same_new_locked_test_bootstrap_ci.csv"
OUT_PAIRED = OUT_DIR / "v2_candidate_paired_bootstrap_delta.csv"
OUT_MCNEMAR = OUT_DIR / "v2_candidate_mcnemar.csv"
OUT_ERRORS = OUT_DIR / "v2_candidate_error_flows.csv"
OUT_ACCEPTANCE = OUT_DIR / "FINAL_ACCEPTANCE_DECISION.json"

RANDOM_SEED = 42
BOOTSTRAPS = 1000
FINAL_ACCEPTED = "FINAL_MODEL_VALIDATED_V2_POSITIVE_RECALL"
FINAL_REJECTED = "V2_POSITIVE_RECALL_CANDIDATE_NOT_ACCEPTED_KEEP_ORIGINAL_V2"


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


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def readiness_checks() -> tuple[pd.DataFrame, bool]:
    locked = read_csv(LOCKED_FINAL)
    registry = read_csv(REGISTRY)
    manifest = json.loads(LOCKED_FREEZE_MANIFEST.read_text(encoding="utf-8"))
    label_col = "final_sentiment_label"
    labels = locked[label_col].map(str) if label_col in locked.columns else pd.Series("", index=locked.index)
    evaluable = labels.isin(LABELS)
    no_text = labels.eq("No Text")
    uncertain = labels.eq("Uncertain")
    pending = (
        labels.eq("")
        | locked.get("human_adjudication_status", pd.Series("", index=locked.index)).astype(str).str.contains("PENDING", case=False, na=False)
    )
    registry_dev_ids = set(registry.loc[registry["split_family"].eq("development"), "comment_id"].map(str))
    overlap_ids = set(locked["comment_id"].map(str)) & registry_dev_ids
    counts = labels.value_counts().reindex(LABELS, fill_value=0)
    duplicate_count = int(locked["comment_id"].duplicated().sum()) if "comment_id" in locked.columns else len(locked)
    checks = [
        {
            "check": "candidate_model_exists",
            "expected": True,
            "observed": CANDIDATE_MODEL.exists(),
            "passed": CANDIDATE_MODEL.exists(),
            "notes": "",
        },
        {
            "check": "candidate_config_exists",
            "expected": True,
            "observed": CANDIDATE_CONFIG.exists(),
            "passed": CANDIDATE_CONFIG.exists(),
            "notes": "",
        },
        {
            "check": "locked_test_final_rows",
            "expected": ">=400; ideal 500-600",
            "observed": len(locked),
            "passed": len(locked) >= 400,
            "notes": "",
        },
        {
            "check": "no_pending_human_adjudication",
            "expected": 0,
            "observed": int(pending.sum()),
            "passed": int(pending.sum()) == 0,
            "notes": "Two blind annotator files and adjudication must be completed before evaluation.",
        },
        {
            "check": "no_non_evaluable_in_locked_test",
            "expected": 0,
            "observed": int((no_text | uncertain | labels.eq("INJ") | labels.eq("")).sum()),
            "passed": int((no_text | uncertain | labels.eq("INJ") | labels.eq("")).sum()) == 0,
            "notes": "",
        },
        {
            "check": "minimum_positive_support",
            "expected": ">=100",
            "observed": int(counts["Positive"]),
            "passed": int(counts["Positive"]) >= 100,
            "notes": "",
        },
        {
            "check": "minimum_negative_support",
            "expected": ">=100",
            "observed": int(counts["Negative"]),
            "passed": int(counts["Negative"]) >= 100,
            "notes": "",
        },
        {
            "check": "unique_comment_id",
            "expected": "no duplicates",
            "observed": duplicate_count,
            "passed": duplicate_count == 0,
            "notes": "",
        },
        {
            "check": "no_comment_id_overlap_with_development",
            "expected": 0,
            "observed": len(overlap_ids),
            "passed": len(overlap_ids) == 0,
            "notes": "Development registry overlap check.",
        },
        {
            "check": "candidate_frozen_before_test_opened",
            "expected": True,
            "observed": bool(manifest.get("candidate_model_frozen_before_final_locked_test_labels", False)),
            "passed": bool(manifest.get("candidate_model_frozen_before_final_locked_test_labels", False)),
            "notes": "",
        },
    ]
    frame = pd.DataFrame(checks)
    return frame, bool(frame["passed"].all())


def predict_v2(locked: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame, dict[str, object]]:
    artifact = joblib.load(V2_MODEL)
    config = json.loads(V2_CONFIG.read_text(encoding="utf-8"))
    text = locked["comment_text_original"].map(normalize_for_model)
    labels = [str(x) for x in artifact["label_encoder"].classes_]
    pipelines = artifact["pipeline"] if isinstance(artifact["pipeline"], list) else [{"pipeline": artifact["pipeline"]}]
    probs = []
    for component in pipelines:
        probs.append(predict_proba_aligned(component["pipeline"], text, labels=labels))
    probs_arr = np.mean(probs, axis=0)
    threshold = float(config.get("selected_threshold", config.get("threshold", artifact.get("selected_threshold", 0.0))))
    pred = apply_threshold_policy(
        probs_arr,
        positive_threshold=1.01,
        margin_positive_neutral=0.0,
        margin_positive_negative=0.0,
        abstention_threshold=threshold,
        no_text_mask=locked["comment_text_original"].map(no_text_flag),
        labels=labels,
    )
    argmax = np.array(labels)[probs_arr.argmax(axis=1)]
    pred["predicted_label"] = np.where(
        pred["predicted_label"].eq("Uncertain") | pred["predicted_label"].eq("No Text"),
        pred["predicted_label"],
        argmax,
    )
    pred["model"] = "V2_frozen"
    return probs_arr, pred, config


def predict_candidate(locked: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame, dict[str, object]]:
    artifact = joblib.load(CANDIDATE_MODEL)
    config = json.loads(CANDIDATE_CONFIG.read_text(encoding="utf-8"))
    probs = artifact_predict_proba(artifact, locked["comment_text_original"])
    pred = predict_labels_from_artifact(artifact, locked["comment_text_original"])
    pred["model"] = "V2_positive_recall_candidate"
    return probs, pred, config


def calibration_metrics(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> dict[str, float]:
    pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    correct = pred == y_true
    ece = 0.0
    for low in np.linspace(0, 1, n_bins, endpoint=False):
        high = low + 1 / n_bins
        mask = (conf >= low) & (conf < high if high < 1 else conf <= high)
        if mask.any():
            ece += float(mask.mean()) * abs(float(correct[mask].mean()) - float(conf[mask].mean()))
    one_hot = np.eye(len(LABELS))[y_true]
    return {"ece": float(ece), "brier_score": float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))}


def metric_bundle(y_true_label: pd.Series, pred_label: pd.Series, probs: np.ndarray, model: str) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    covered = pred_label.isin(LABELS)
    y_true = y_true_label.loc[covered].astype(str)
    y_pred = pred_label.loc[covered].astype(str)
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=LABELS, zero_division=0)
    y_idx = y_true.map(LABEL_TO_ID).to_numpy(dtype=int)
    p_cov = probs[covered.to_numpy()]
    metrics = {
        "model": model,
        "n_evaluable": int(len(y_true_label)),
        "n_covered": int(covered.sum()),
        "n_abstained": int((~covered).sum()),
        "coverage": float(covered.mean()),
        "abstention_rate": float((~covered).mean()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="weighted", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "minimum_recall": float(np.min(recall)),
        "class_collapse": bool(len(set(y_pred.tolist())) < len(LABELS)),
        "positive_precision": float(precision[LABEL_TO_ID["Positive"]]),
        "positive_recall": float(recall[LABEL_TO_ID["Positive"]]),
        "positive_f1": float(f1[LABEL_TO_ID["Positive"]]),
        "neutral_recall": float(recall[LABEL_TO_ID["Neutral"]]),
        "negative_recall": float(recall[LABEL_TO_ID["Negative"]]),
        "positive_to_neutral": int(((y_true_label == "Positive") & (pred_label == "Neutral")).sum()),
        "positive_to_negative": int(((y_true_label == "Positive") & (pred_label == "Negative")).sum()),
        "neutral_to_positive": int(((y_true_label == "Neutral") & (pred_label == "Positive")).sum()),
        "negative_to_positive": int(((y_true_label == "Negative") & (pred_label == "Positive")).sum()),
        **calibration_metrics(y_idx, p_cov),
    }
    per_class = pd.DataFrame(
        [
            {
                "model": model,
                "label": label,
                "precision": float(precision[idx]),
                "recall": float(recall[idx]),
                "f1": float(f1[idx]),
                "support": int(support[idx]),
            }
            for idx, label in enumerate(LABELS)
        ]
    )
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    cm_rows = []
    for i, true_label in enumerate(LABELS):
        for j, predicted_label in enumerate(LABELS):
            cm_rows.append({"model": model, "true_label": true_label, "predicted_label": predicted_label, "count": int(cm[i, j])})
    return metrics, per_class, pd.DataFrame(cm_rows)


def bootstrap_ci(y: pd.Series, pred: pd.Series, probs: np.ndarray, model: str) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    idx = np.arange(len(y))
    metrics = defaultdict(list)  # type: ignore[name-defined]
    for _ in range(BOOTSTRAPS):
        sample = rng.choice(idx, size=len(idx), replace=True)
        m, _, _ = metric_bundle(y.iloc[sample].reset_index(drop=True), pred.iloc[sample].reset_index(drop=True), probs[sample], model)
        for name in ["accuracy", "macro_f1", "weighted_f1", "balanced_accuracy", "mcc", "positive_precision", "positive_recall", "positive_f1"]:
            metrics[name].append(float(m[name]))
    return pd.DataFrame(
        [
            {
                "model": model,
                "metric": name,
                "ci_lower_95": float(np.quantile(values, 0.025)),
                "ci_upper_95": float(np.quantile(values, 0.975)),
                "n_bootstrap": BOOTSTRAPS,
            }
            for name, values in metrics.items()
        ]
    )


def paired_bootstrap(y: pd.Series, pred_v2: pd.Series, probs_v2: np.ndarray, pred_c: pd.Series, probs_c: np.ndarray) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    idx = np.arange(len(y))
    deltas = defaultdict(list)  # type: ignore[name-defined]
    for _ in range(BOOTSTRAPS):
        sample = rng.choice(idx, size=len(idx), replace=True)
        m2, _, _ = metric_bundle(y.iloc[sample].reset_index(drop=True), pred_v2.iloc[sample].reset_index(drop=True), probs_v2[sample], "V2_frozen")
        mc, _, _ = metric_bundle(y.iloc[sample].reset_index(drop=True), pred_c.iloc[sample].reset_index(drop=True), probs_c[sample], "V2_positive_recall_candidate")
        for name in ["accuracy", "macro_f1", "weighted_f1", "balanced_accuracy", "mcc", "positive_precision", "positive_recall", "positive_f1"]:
            deltas[name].append(float(mc[name]) - float(m2[name]))
    return pd.DataFrame(
        [
            {
                "metric": name,
                "delta_candidate_minus_v2_mean": float(np.mean(values)),
                "ci_lower_95": float(np.quantile(values, 0.025)),
                "ci_upper_95": float(np.quantile(values, 0.975)),
                "paired_bootstrap_shows_decrease": bool(np.quantile(values, 0.975) < 0),
                "n_bootstrap": BOOTSTRAPS,
            }
            for name, values in deltas.items()
        ]
    )


def mcnemar(y: pd.Series, pred_v2: pd.Series, pred_c: pd.Series) -> pd.DataFrame:
    correct_v2 = pred_v2.eq(y)
    correct_c = pred_c.eq(y)
    b = int((correct_v2 & ~correct_c).sum())
    c = int((~correct_v2 & correct_c).sum())
    n = b + c
    p = float(binomtest(min(b, c), n=n, p=0.5).pvalue) if n else 1.0
    return pd.DataFrame([{"b_v2_correct_candidate_wrong": b, "c_v2_wrong_candidate_correct": c, "discordant_pairs": n, "mcnemar_exact_p": p}])


def final_gate(metrics: pd.DataFrame, paired: pd.DataFrame, readiness: pd.DataFrame) -> tuple[str, list[dict[str, object]]]:
    m = metrics.set_index("model")
    v2 = m.loc["V2_frozen"]
    cand = m.loc["V2_positive_recall_candidate"]
    paired_decrease = bool(paired["paired_bootstrap_shows_decrease"].any())
    checks = [
        ("readiness_all_passed", True, bool(readiness["passed"].all()), bool(readiness["passed"].all())),
        ("macro_f1_candidate_gt_v2", float(v2["macro_f1"]), float(cand["macro_f1"]), float(cand["macro_f1"]) > float(v2["macro_f1"])),
        ("balanced_accuracy_candidate_gt_v2", float(v2["balanced_accuracy"]), float(cand["balanced_accuracy"]), float(cand["balanced_accuracy"]) > float(v2["balanced_accuracy"])),
        ("mcc_candidate_gt_v2", float(v2["mcc"]), float(cand["mcc"]), float(cand["mcc"]) > float(v2["mcc"])),
        ("positive_recall_candidate_ge_0_60", 0.60, float(cand["positive_recall"]), float(cand["positive_recall"]) >= 0.60),
        ("positive_recall_candidate_ge_v2_plus_0_10", float(v2["positive_recall"]) + 0.10, float(cand["positive_recall"]), float(cand["positive_recall"]) >= float(v2["positive_recall"]) + 0.10),
        ("positive_precision_candidate_ge_0_65", 0.65, float(cand["positive_precision"]), float(cand["positive_precision"]) >= 0.65),
        ("positive_f1_candidate_gt_v2", float(v2["positive_f1"]), float(cand["positive_f1"]), float(cand["positive_f1"]) > float(v2["positive_f1"])),
        ("neutral_recall_ge_0_75", 0.75, float(cand["neutral_recall"]), float(cand["neutral_recall"]) >= 0.75),
        ("negative_recall_ge_0_60", 0.60, float(cand["negative_recall"]), float(cand["negative_recall"]) >= 0.60),
        ("coverage_ge_0_90", 0.90, float(cand["coverage"]), float(cand["coverage"]) >= 0.90),
        ("no_class_collapse", True, not bool(cand["class_collapse"]), not bool(cand["class_collapse"])),
        ("paired_bootstrap_no_decrease", False, paired_decrease, not paired_decrease),
    ]
    status = FINAL_ACCEPTED if all(passed for _, _, _, passed in checks) else FINAL_REJECTED
    return status, [
        {"gate": name, "reference_or_threshold": reference, "observed": observed, "passed": bool(passed)}
        for name, reference, observed, passed in checks
    ]


def write_blocked(readiness: pd.DataFrame) -> None:
    readiness.to_csv(OUT_READINESS, index=False, encoding="utf-8-sig")
    decision = {
        "status": FINAL_REJECTED,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "reason": "New locked test human adjudication is not ready; candidate is not accepted and full inference is blocked.",
        "readiness": readiness.to_dict(orient="records"),
        "locked_test_used_for_training_or_selection": False,
        "full_inference_allowed": False,
        "promotion_allowed": False,
        "no_failed_recognition_to_positive": True,
    }
    write_json(OUT_ACCEPTANCE, decision)
    print(json.dumps({"status": FINAL_REJECTED, "ready": False}, indent=2))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    readiness, ready = readiness_checks()
    if not ready:
        write_blocked(readiness)
        return

    locked = read_csv(LOCKED_FINAL)
    locked = locked.loc[locked["final_sentiment_label"].isin(LABELS)].copy().reset_index(drop=True)
    y = locked["final_sentiment_label"]
    probs_v2, pred_v2, config_v2 = predict_v2(locked)
    probs_c, pred_c, config_c = predict_candidate(locked)
    metrics_rows = []
    per_class_frames = []
    confusion_frames = []
    for model, probs, pred in [
        ("V2_frozen", probs_v2, pred_v2["predicted_label"]),
        ("V2_positive_recall_candidate", probs_c, pred_c["predicted_label"]),
    ]:
        metrics, per_class, cm = metric_bundle(y, pred, probs, model)
        metrics_rows.append(metrics)
        per_class_frames.append(per_class)
        confusion_frames.append(cm)
    metrics = pd.DataFrame(metrics_rows)
    per_class = pd.concat(per_class_frames, ignore_index=True)
    confusion = pd.concat(confusion_frames, ignore_index=True)
    bootstrap = pd.concat(
        [
            bootstrap_ci(y, pred_v2["predicted_label"], probs_v2, "V2_frozen"),
            bootstrap_ci(y, pred_c["predicted_label"], probs_c, "V2_positive_recall_candidate"),
        ],
        ignore_index=True,
    )
    paired = paired_bootstrap(y, pred_v2["predicted_label"], probs_v2, pred_c["predicted_label"], probs_c)
    mc = mcnemar(y, pred_v2["predicted_label"], pred_c["predicted_label"])
    status, gates = final_gate(metrics, paired, readiness)

    pred_out = locked[["comment_id", "final_sentiment_label", "comment_text_original", "video_id", "brand_or_video_context"]].copy()
    for prefix, probs, pred in [("v2", probs_v2, pred_v2), ("candidate", probs_c, pred_c)]:
        pred_out[f"{prefix}_predicted_label"] = pred["predicted_label"]
        pred_out[f"{prefix}_confidence"] = pred["prediction_confidence"]
        for idx, label in enumerate(LABELS):
            pred_out[f"{prefix}_probability_{label.lower()}"] = probs[:, idx]
    error_rows = []
    for model, pred in [("V2_frozen", pred_v2["predicted_label"]), ("V2_positive_recall_candidate", pred_c["predicted_label"])]:
        for true_label, predicted_label, name in [
            ("Positive", "Neutral", "positive_to_neutral"),
            ("Positive", "Negative", "positive_to_negative"),
            ("Neutral", "Positive", "neutral_to_positive"),
            ("Negative", "Positive", "negative_to_positive"),
        ]:
            error_rows.append(
                {
                    "model": model,
                    "error_flow": name,
                    "count": int(((y == true_label) & (pred == predicted_label)).sum()),
                }
            )

    readiness.to_csv(OUT_READINESS, index=False, encoding="utf-8-sig")
    pred_out.to_csv(OUT_PREDICTIONS, index=False, encoding="utf-8-sig")
    metrics.to_csv(OUT_METRICS, index=False, encoding="utf-8-sig")
    per_class.to_csv(OUT_PER_CLASS, index=False, encoding="utf-8-sig")
    confusion.to_csv(OUT_CONFUSION, index=False, encoding="utf-8-sig")
    bootstrap.to_csv(OUT_BOOTSTRAP, index=False, encoding="utf-8-sig")
    paired.to_csv(OUT_PAIRED, index=False, encoding="utf-8-sig")
    mc.to_csv(OUT_MCNEMAR, index=False, encoding="utf-8-sig")
    pd.DataFrame(error_rows).to_csv(OUT_ERRORS, index=False, encoding="utf-8-sig")
    write_json(
        OUT_ACCEPTANCE,
        {
            "status": status,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "locked_test_sha256": sha256_file(LOCKED_FINAL),
            "candidate_model_sha256": sha256_file(CANDIDATE_MODEL),
            "v2_model_sha256": sha256_file(V2_MODEL),
            "acceptance_gates": gates,
            "metrics": metrics.to_dict(orient="records"),
            "policy_confirmations": {
                "locked_test_used_for_training_or_selection": False,
                "threshold_selected_from_development_only": True,
                "no_failed_recognition_to_positive": True,
                "full_inference_allowed": status == FINAL_ACCEPTED,
            },
            "model_configs": {
                "v2": config_v2,
                "candidate": config_c,
            },
        },
    )
    print(json.dumps({"status": status, "ready": True}, indent=2))


if __name__ == "__main__":
    main()
