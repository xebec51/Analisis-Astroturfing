from __future__ import annotations

import hashlib
import json
import platform
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
)


ROOT = Path(__file__).resolve().parents[1]
HUMAN_DIR = ROOT / "output/rm2_sentiment/human_validation_v2"
MODEL_DIR = ROOT / "output/rm2_sentiment/model_v2"

LOCKED_FINAL = HUMAN_DIR / "locked_test_v2_observational_final.csv"
LOCKED_FINAL_MANIFEST = HUMAN_DIR / "locked_test_v2_observational_final_manifest.json"
LOCKED_FINAL_INTEGRITY = HUMAN_DIR / "locked_test_v2_observational_final_integrity.csv"
MODEL_ARTIFACT = MODEL_DIR / "selected_model_development_frozen.joblib"
MODEL_CONFIG = MODEL_DIR / "selected_model_development_frozen_config.json"
TRAINING_PROVENANCE = MODEL_DIR / "development_training_pool_provenance.csv"

OUT_PREDICTIONS = MODEL_DIR / "final_locked_test_predictions.csv"
OUT_METRICS = MODEL_DIR / "final_locked_test_metrics.csv"
OUT_PER_CLASS = MODEL_DIR / "final_locked_test_per_class_metrics.csv"
OUT_CONFUSION = MODEL_DIR / "final_locked_test_confusion_matrix.csv"
OUT_BOOTSTRAP = MODEL_DIR / "final_locked_test_bootstrap_ci.csv"
OUT_CALIBRATION = MODEL_DIR / "final_locked_test_calibration.csv"
OUT_SUBGROUP = MODEL_DIR / "final_locked_test_subgroup_metrics.csv"
OUT_REPORT = MODEL_DIR / "final_locked_test_report.json"
OUT_ACCEPTANCE = MODEL_DIR / "final_locked_test_acceptance_decision.csv"
OUT_LOCK = MODEL_DIR / "final_locked_test_evaluation_lock.json"

EXPECTED_MODEL_SHA256 = "477bfe11ebc3463eeff5a5fb8359f86022d719c8bea49dfdad0460fa0c5e2ccc"
LABELS = ["Negative", "Neutral", "Positive"]
NON_EVALUABLE = {"Uncertain", "No Text", ""}
RANDOM_SEED = 20260721
MIN_SUBGROUP_SUPPORT = 20


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False, low_memory=False)


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def clean_social(text: object) -> str:
    if pd.isna(text):
        return ""
    s = str(text).strip().lower()
    if s.lower() in {"nan", "none", "null", "<na>"}:
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
    denom = out.sum(axis=1, keepdims=True)
    zero = denom.squeeze() == 0
    out[~zero] = out[~zero] / denom[~zero]
    if zero.any():
        out[zero] = 1.0 / len(label_encoder.classes_)
    return out


def ensemble_predict_proba(artifact: dict, text: pd.Series) -> np.ndarray:
    parts = []
    for component in artifact["pipeline"]:
        model = component["pipeline"]
        parts.append(predict_proba_aligned(model, text, artifact["label_encoder"]))
    probs = np.mean(parts, axis=0)
    return probs / np.clip(probs.sum(axis=1, keepdims=True), 1e-12, None)


def multiclass_ece(y_true_idx: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> float:
    confidence = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true_idx).astype(float)
    ece = 0.0
    for low in np.linspace(0, 1, n_bins, endpoint=False):
        high = low + 1.0 / n_bins
        mask = (confidence >= low) & (confidence < high if high < 1 else confidence <= high)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(ece)


def multiclass_brier(y_true_idx: np.ndarray, probs: np.ndarray, n_classes: int) -> float:
    total = 0.0
    for class_idx in range(n_classes):
        total += brier_score_loss((y_true_idx == class_idx).astype(int), probs[:, class_idx])
    return float(total / n_classes)


def covered_metrics(frame: pd.DataFrame) -> dict[str, object]:
    covered = frame[frame["is_evaluable_reference"] & frame["is_covered"]].copy()
    y_true = covered["true_label"].astype(str)
    y_pred = covered["predicted_label"].astype(str)
    probs = covered[[f"probability_{label.lower()}" for label in LABELS]].astype(float).to_numpy()
    y_true_idx = np.array([LABELS.index(v) for v in y_true], dtype=int)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, zero_division=0
    )
    return {
        "covered": covered,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
        "macro_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="weighted", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "ece": multiclass_ece(y_true_idx, probs),
        "brier_score": multiclass_brier(y_true_idx, probs, len(LABELS)),
        "confusion": confusion_matrix(y_true, y_pred, labels=LABELS),
    }


def subgroup_metrics(pred: pd.DataFrame, column: str) -> pd.DataFrame:
    rows = []
    values = pred[column].fillna("").astype(str) if column in pred.columns else pd.Series("", index=pred.index)
    for value, group in pred.assign(_group_value=values).groupby("_group_value", dropna=False):
        evaluable = group[group["is_evaluable_reference"]].copy()
        covered = evaluable[evaluable["is_covered"]].copy()
        row = {
            "subgroup_type": column,
            "subgroup_value": value if value != "" else "Not available",
            "n_rows": int(len(group)),
            "n_evaluable": int(len(evaluable)),
            "n_covered": int(len(covered)),
            "coverage": float(len(covered) / len(evaluable)) if len(evaluable) else np.nan,
            "status": "AVAILABLE" if len(evaluable) >= MIN_SUBGROUP_SUPPORT else "INSUFFICIENT_SUPPORT",
            "minimum_support_rule": f"n_evaluable >= {MIN_SUBGROUP_SUPPORT}",
        }
        if len(evaluable) >= MIN_SUBGROUP_SUPPORT and len(covered) > 0:
            y_true = covered["true_label"].astype(str)
            y_pred = covered["predicted_label"].astype(str)
            row.update(
                {
                    "macro_f1_covered": float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)),
                    "accuracy_covered": float(accuracy_score(y_true, y_pred)),
                    "balanced_accuracy_covered": float(balanced_accuracy_score(y_true, y_pred)),
                    "mcc_covered": float(matthews_corrcoef(y_true, y_pred)),
                }
            )
        else:
            row.update(
                {
                    "macro_f1_covered": np.nan,
                    "accuracy_covered": np.nan,
                    "balanced_accuracy_covered": np.nan,
                    "mcc_covered": np.nan,
                }
            )
        rows.append(row)
    return pd.DataFrame(rows)


def gate(metric: str, threshold: object, observed: object, passed: bool, notes: str = "") -> dict[str, object]:
    return {
        "metric": metric,
        "threshold": threshold,
        "observed": observed,
        "passed": bool(passed),
        "notes": notes,
    }


def write_atomic(frames: dict[Path, pd.DataFrame], jsons: dict[Path, dict]) -> None:
    tmp_paths: list[tuple[Path, Path]] = []
    try:
        for path, frame in frames.items():
            tmp = path.with_suffix(path.suffix + ".tmp")
            frame.to_csv(tmp, index=False)
            tmp_paths.append((tmp, path))
        for path, payload in jsons.items():
            tmp = path.with_suffix(path.suffix + ".tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            tmp_paths.append((tmp, path))
        for tmp, path in tmp_paths:
            tmp.replace(path)
    finally:
        for tmp, _ in tmp_paths:
            if tmp.exists():
                tmp.unlink()


def main() -> None:
    if OUT_LOCK.exists():
        raise FileExistsError(f"Evaluation lock already exists: {OUT_LOCK}")
    final_outputs = [
        OUT_PREDICTIONS,
        OUT_METRICS,
        OUT_PER_CLASS,
        OUT_CONFUSION,
        OUT_BOOTSTRAP,
        OUT_CALIBRATION,
        OUT_SUBGROUP,
        OUT_REPORT,
        OUT_ACCEPTANCE,
    ]
    existing = [str(p) for p in final_outputs if p.exists()]
    if existing:
        raise FileExistsError(f"Refusing to run without lock because final evaluation outputs already exist: {existing}")

    freeze_commit = git_head()
    run_id = hashlib.sha256(f"{freeze_commit}|{datetime.now(timezone.utc).isoformat()}".encode("utf-8")).hexdigest()[:16]
    locked_manifest = json.loads(LOCKED_FINAL_MANIFEST.read_text(encoding="utf-8"))
    config = json.loads(MODEL_CONFIG.read_text(encoding="utf-8"))
    model_hash = sha256_file(MODEL_ARTIFACT)
    config_hash = sha256_file(MODEL_CONFIG)
    locked_hash = sha256_file(LOCKED_FINAL)

    if model_hash != EXPECTED_MODEL_SHA256:
        raise AssertionError(f"Unexpected model hash: {model_hash}")
    if config_hash != locked_manifest["config_hash"]:
        raise AssertionError("Config hash does not match locked-test freeze manifest.")
    if locked_hash != locked_manifest["locked_test_final_sha256"]:
        raise AssertionError("Locked-test hash does not match freeze manifest.")
    threshold = float(config["threshold"])
    if threshold != 0.42:
        raise AssertionError(f"Unexpected threshold: {threshold}")

    integrity = read_csv(LOCKED_FINAL_INTEGRITY)
    if not integrity["passed"].astype(str).str.lower().eq("true").all():
        raise AssertionError("Locked-test final integrity is not all PASS.")

    artifact = joblib.load(MODEL_ARTIFACT)
    if artifact["ensemble_components"] != config["ensemble_components"]:
        raise AssertionError("Ensemble components changed.")
    if artifact["model_name"] != config["model_name"]:
        raise AssertionError("Model family changed.")

    locked = read_csv(LOCKED_FINAL)
    if len(locked) != 300 or locked["comment_id"].nunique() != 300:
        raise AssertionError("Locked-test final must contain 300 unique rows.")
    training = read_csv(TRAINING_PROVENANCE)
    training_ids = set(training.loc[training["included_in_training"].astype(str).str.lower().eq("true"), "comment_id"])
    if set(locked["comment_id"]) & training_ids:
        raise AssertionError("Locked-test overlaps training pool.")

    locked["text_social_normalized"] = locked["comment_text_original"].map(clean_social)
    probs = ensemble_predict_proba(artifact, locked["text_social_normalized"])
    label_order = [str(x) for x in artifact["label_encoder"].classes_]
    prob_df = pd.DataFrame(probs, columns=[f"probability_{label.lower()}" for label in label_order])
    pred_idx = probs.argmax(axis=1)
    pred_label = [label_order[i] for i in pred_idx]
    confidence = probs.max(axis=1)

    pred = locked.copy()
    for col in prob_df.columns:
        pred[col] = prob_df[col].astype(float)
    pred["predicted_label"] = pred_label
    pred["prediction_confidence"] = confidence
    pred["threshold"] = threshold
    pred["abstained"] = pred["prediction_confidence"].astype(float) < threshold
    pred["is_covered"] = ~pred["abstained"]
    pred["true_label"] = pred["final_sentiment_label"].astype(str)
    pred["is_evaluable_reference"] = pred["true_label"].isin(LABELS)
    pred["evaluation_exclusion_reason"] = np.where(pred["is_evaluable_reference"], "", "non_evaluable_reference_label")
    pred["final_output_label_for_evaluation"] = np.where(
        pred["is_evaluable_reference"] & pred["is_covered"],
        pred["predicted_label"],
        np.where(pred["is_evaluable_reference"], "Uncertain", pred["true_label"]),
    )
    pred["correct_if_covered"] = np.where(
        pred["is_evaluable_reference"] & pred["is_covered"],
        pred["predicted_label"].eq(pred["true_label"]),
        "",
    )
    pred["model_name"] = config["model_name"]
    pred["selected_candidate"] = config["selected_candidate_id"]
    pred["model_hash"] = model_hash
    pred["config_hash"] = config_hash
    pred["locked_test_sha256"] = locked_hash
    pred["freeze_commit"] = freeze_commit
    pred["evaluation_run_id"] = run_id
    pred["evaluation_timestamp_utc"] = datetime.now(timezone.utc).isoformat()

    n_total = len(pred)
    evaluable = pred[pred["is_evaluable_reference"]].copy()
    metrics_pack = covered_metrics(pred)
    covered = metrics_pack["covered"]
    n_evaluable = len(evaluable)
    n_covered = len(covered)
    n_abstained = int(n_evaluable - n_covered)
    coverage = n_covered / n_evaluable
    abstention_rate = n_abstained / n_evaluable
    correct_covered = int(covered["predicted_label"].eq(covered["true_label"]).sum())
    selective_error = 1.0 - (correct_covered / n_covered)
    whole_test_correct_covered_share = correct_covered / n_total
    all_evaluable_probs = evaluable[[f"probability_{label.lower()}" for label in LABELS]].astype(float).to_numpy()
    all_evaluable_idx = np.array([LABELS.index(v) for v in evaluable["true_label"]], dtype=int)

    per_class = pd.DataFrame(
        {
            "class_label": LABELS,
            "precision": metrics_pack["precision"],
            "recall": metrics_pack["recall"],
            "f1": metrics_pack["f1"],
            "support": metrics_pack["support"],
            "covered_predicted_count": [int((covered["predicted_label"] == label).sum()) for label in LABELS],
            "abstained_true_count": [
                int((evaluable["true_label"].eq(label) & evaluable["abstained"]).sum()) for label in LABELS
            ],
        }
    )
    metrics = pd.DataFrame(
        [
            {"metric": "n_locked_test_rows", "value": n_total},
            {"metric": "n_reference_evaluable", "value": n_evaluable},
            {"metric": "n_non_evaluable_reference", "value": n_total - n_evaluable},
            {"metric": "n_covered", "value": n_covered},
            {"metric": "n_abstained", "value": n_abstained},
            {"metric": "coverage", "value": coverage},
            {"metric": "abstention_rate", "value": abstention_rate},
            {"metric": "macro_f1_covered", "value": metrics_pack["macro_f1"]},
            {"metric": "weighted_f1_covered", "value": metrics_pack["weighted_f1"]},
            {"metric": "accuracy_covered", "value": metrics_pack["accuracy"]},
            {"metric": "balanced_accuracy_covered", "value": metrics_pack["balanced_accuracy"]},
            {"metric": "mcc_covered", "value": metrics_pack["mcc"]},
            {"metric": "minimum_per_class_recall", "value": float(np.min(metrics_pack["recall"]))},
            {"metric": "minimum_per_class_precision", "value": float(np.min(metrics_pack["precision"]))},
            {"metric": "correct_covered", "value": correct_covered},
            {"metric": "correct_covered_share_of_locked_test", "value": whole_test_correct_covered_share},
            {"metric": "selective_error", "value": selective_error},
            {"metric": "ece_covered", "value": metrics_pack["ece"]},
            {"metric": "brier_score_covered", "value": metrics_pack["brier_score"]},
            {"metric": "ece_all_evaluable", "value": multiclass_ece(all_evaluable_idx, all_evaluable_probs)},
            {"metric": "brier_score_all_evaluable", "value": multiclass_brier(all_evaluable_idx, all_evaluable_probs, len(LABELS))},
        ]
    )

    cm = pd.DataFrame(metrics_pack["confusion"], columns=LABELS)
    cm.insert(0, "true_label", LABELS)

    rng = np.random.default_rng(RANDOM_SEED)
    idx = np.arange(len(covered))
    boots = []
    y_true = covered["true_label"].astype(str).reset_index(drop=True)
    y_pred = covered["predicted_label"].astype(str).reset_index(drop=True)
    for _ in range(1000):
        sample_idx = rng.choice(idx, size=len(idx), replace=True)
        boots.append(f1_score(y_true.iloc[sample_idx], y_pred.iloc[sample_idx], labels=LABELS, average="macro", zero_division=0))
    bootstrap = pd.DataFrame(
        [
            {
                "metric": "macro_f1_covered",
                "n_bootstrap": 1000,
                "mean": float(np.mean(boots)),
                "ci_lower_95": float(np.quantile(boots, 0.025)),
                "ci_upper_95": float(np.quantile(boots, 0.975)),
                "random_seed": RANDOM_SEED,
            }
        ]
    )

    bins = pd.cut(pred["prediction_confidence"].astype(float), bins=np.linspace(0, 1, 11), include_lowest=True)
    confidence_dist = (
        pred.assign(confidence_bin=bins.astype(str))
        .groupby("confidence_bin", dropna=False)
        .agg(n_rows=("comment_id", "count"), n_evaluable=("is_evaluable_reference", "sum"), n_covered=("is_covered", "sum"))
        .reset_index()
    )
    abstention_by_class = (
        pred[pred["is_evaluable_reference"]]
        .groupby("true_label")
        .agg(n_reference=("comment_id", "count"), n_abstained=("abstained", "sum"), n_covered=("is_covered", "sum"))
        .reset_index()
    )
    calibration = pd.concat(
        [
            pd.DataFrame(
                [
                    {"section": "calibration", "metric": "ece_covered", "value": metrics_pack["ece"]},
                    {"section": "calibration", "metric": "brier_score_covered", "value": metrics_pack["brier_score"]},
                    {"section": "calibration", "metric": "ece_all_evaluable", "value": multiclass_ece(all_evaluable_idx, all_evaluable_probs)},
                    {"section": "calibration", "metric": "brier_score_all_evaluable", "value": multiclass_brier(all_evaluable_idx, all_evaluable_probs, len(LABELS))},
                ]
            ),
            confidence_dist.rename(columns={"confidence_bin": "metric", "n_rows": "value"}).assign(section="confidence_distribution")[
                ["section", "metric", "value", "n_evaluable", "n_covered"]
            ],
            abstention_by_class.rename(columns={"true_label": "metric", "n_abstained": "value"}).assign(section="abstention_by_true_class")[
                ["section", "metric", "value", "n_reference", "n_covered"]
            ],
        ],
        ignore_index=True,
        sort=False,
    )

    subgroup_cols = [
        "is_hcc_member",
        "brand_or_video_context",
        "video_id",
        "comment_type",
        "is_question",
        "final_sentiment_target",
        "final_complaint_scope",
    ]
    subgroup = pd.concat([subgroup_metrics(pred, col) for col in subgroup_cols], ignore_index=True, sort=False)

    no_class_collapse = set(covered["predicted_label"]) == set(LABELS) and set(covered["true_label"]) == set(LABELS)
    negative_precision = float(per_class.loc[per_class["class_label"].eq("Negative"), "precision"].iloc[0])
    positive_recall = float(per_class.loc[per_class["class_label"].eq("Positive"), "recall"].iloc[0])
    acceptance = pd.DataFrame(
        [
            gate("no_class_collapse", True, no_class_collapse, no_class_collapse),
            gate("coverage", ">=0.85", coverage, coverage >= 0.85),
            gate("macro_f1_covered", ">=0.60", metrics_pack["macro_f1"], metrics_pack["macro_f1"] >= 0.60),
            gate("balanced_accuracy_covered", ">=0.60", metrics_pack["balanced_accuracy"], metrics_pack["balanced_accuracy"] >= 0.60),
            gate("mcc_covered", ">=0.35", metrics_pack["mcc"], metrics_pack["mcc"] >= 0.35),
            gate("minimum_per_class_recall", ">=0.40", float(np.min(metrics_pack["recall"])), float(np.min(metrics_pack["recall"])) >= 0.40),
            gate("minimum_per_class_precision", ">=0.40", float(np.min(metrics_pack["precision"])), float(np.min(metrics_pack["precision"])) >= 0.40),
            gate("positive_recall", ">0.341", positive_recall, positive_recall > 0.341),
            gate("negative_precision", ">0.415", negative_precision, negative_precision > 0.415),
            gate("integrity_no_corruption_leakage_overlap", True, True, True, "Frozen locked-test integrity and hash checks passed before scoring."),
        ]
    )
    all_gates_pass = bool(acceptance["passed"].all())
    if all_gates_pass and coverage >= 0.90:
        final_status = "FINAL_MODEL_VALIDATED"
    elif all_gates_pass and coverage >= 0.85:
        final_status = "FINAL_MODEL_VALIDATED_WITH_COVERAGE_CAUTION"
    else:
        final_status = "FINAL_MODEL_EVALUATED_NOT_ACCEPTED"
    acceptance["final_acceptance_status"] = final_status

    report = {
        "status": "FINAL_LOCKED_TEST_EVALUATED_ONCE",
        "final_acceptance_status": final_status,
        "evaluation_run_id": run_id,
        "evaluation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "freeze_commit": freeze_commit,
        "locked_test_sha256": locked_hash,
        "model_sha256": model_hash,
        "config_sha256": config_hash,
        "threshold": threshold,
        "n_locked_test_rows": n_total,
        "n_reference_evaluable": n_evaluable,
        "n_non_evaluable_reference": n_total - n_evaluable,
        "metrics": {row["metric"]: row["value"] for row in metrics.to_dict("records")},
        "acceptance_gates": acceptance.to_dict("records"),
        "per_class_metrics": per_class.to_dict("records"),
        "package_versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }
    lock = {
        "locked_test_sha256": locked_hash,
        "model_sha256": model_hash,
        "config_sha256": config_hash,
        "threshold": threshold,
        "freeze_commit": freeze_commit,
        "evaluation_timestamp_utc": report["evaluation_timestamp_utc"],
        "evaluation_run_id": run_id,
        "status": "FINAL_LOCKED_TEST_EVALUATED_ONCE",
        "final_acceptance_status": final_status,
    }

    frames = {
        OUT_PREDICTIONS: pred,
        OUT_METRICS: metrics,
        OUT_PER_CLASS: per_class,
        OUT_CONFUSION: cm,
        OUT_BOOTSTRAP: bootstrap,
        OUT_CALIBRATION: calibration,
        OUT_SUBGROUP: subgroup,
        OUT_ACCEPTANCE: acceptance,
    }
    jsons = {
        OUT_REPORT: report,
        OUT_LOCK: lock,
    }
    write_atomic(frames, jsons)
    print("FINAL_LOCKED_TEST_EVALUATED_ONCE")
    print(f"status={final_status}")
    print(f"coverage={coverage:.6f} macro_f1={metrics_pack['macro_f1']:.6f} balanced_accuracy={metrics_pack['balanced_accuracy']:.6f} mcc={metrics_pack['mcc']:.6f}")
    print(f"lock={OUT_LOCK}")


if __name__ == "__main__":
    main()
