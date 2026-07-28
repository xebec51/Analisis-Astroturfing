from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
)


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "output/rm2_sentiment/experiments/fair_same_test_comparison"
V2_MODEL_DIR = ROOT / "output/rm2_sentiment/model/frozen"
V2_MODEL_PATH = V2_MODEL_DIR / "selected_model_development_frozen.joblib"
V2_CONFIG_PATH = V2_MODEL_DIR / "selected_model_development_frozen_config.json"
V4_MODEL_DIR = ROOT / "artifacts/rm2_sentiment/indobert_v4_final/base_reference"
V4_DECISION_PATH = ROOT / "output/rm2_sentiment/experiments/indobert_v4_final/locked_test_evaluation/FINAL_ACCEPTANCE_DECISION.json"
LOCKED_REGISTRY = ROOT / "output/rm2_sentiment/validation/human_master_v4/sentiment_v4_locked_test_final_frozen.csv"
V4_PREDICTIONS = ROOT / "output/rm2_sentiment/experiments/indobert_v4_final/locked_test_evaluation/locked_test_predictions.csv"
V4_EVAL_MANIFEST = ROOT / "output/rm2_sentiment/experiments/indobert_v4_final/locked_test_evaluation/LOCKED_TEST_EVALUATION_MANIFEST.json"
DOC_PATH = ROOT / "docs/INDOBERT_V4_FAIR_COMPARISON_REPORT.md"

LABELS = ["Negative", "Neutral", "Positive"]
LABEL_TO_ID = {label: idx for idx, label in enumerate(LABELS)}
ID_TO_LABEL = {idx: label for label, idx in LABEL_TO_ID.items()}
ABSTAIN_LABEL = "Abstain"
RANDOM_SEED = 20260728


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2), encoding="utf-8")


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False, low_memory=False)


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_dataframe(frame: pd.DataFrame, columns: list[str]) -> str:
    data = frame[columns].sort_values(columns).to_csv(index=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def is_true(value: object) -> bool:
    return str(value).strip().lower() == "true"


def clean_social(text: object) -> str:
    if pd.isna(text):
        return ""
    value = str(text).strip().lower()
    if value in {"", "nan", "none", "null", "<na>"}:
        return ""
    value = re.sub(r"https?://\S+|www\.\S+", " URL ", value)
    value = re.sub(r"@\w+", " USERMENTION ", value)
    value = re.sub(r"#(\w+)", r" \1 ", value)
    value = re.sub(r"([!?.,])\1+", r"\1", value)
    return re.sub(r"\s+", " ", value).strip()


def no_text(text: object) -> bool:
    value = "" if pd.isna(text) else str(text).strip()
    return value.lower() in {"", "nan", "none", "null", "<na>", "[deleted]", "deleted"}


def softmax(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    if scores.ndim == 1:
        scores = scores.reshape(-1, 1)
    scores = scores - scores.max(axis=1, keepdims=True)
    exp = np.exp(scores)
    return exp / np.clip(exp.sum(axis=1, keepdims=True), 1e-12, None)


def predict_proba_aligned(model: Any, texts: pd.Series, label_encoder: Any) -> np.ndarray:
    clf = model.named_steps["clf"]
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(texts)
        classes = getattr(clf, "classes_", label_encoder.classes_)
    else:
        probs = softmax(model.decision_function(texts))
        classes = getattr(clf, "classes_", label_encoder.classes_)
    out = np.zeros((len(texts), len(label_encoder.classes_)), dtype=float)
    classes_array = np.asarray(classes)
    if np.issubdtype(classes_array.dtype, np.number):
        for src_idx, encoded_label in enumerate(classes_array.astype(int)):
            if 0 <= encoded_label < len(label_encoder.classes_):
                out[:, encoded_label] = probs[:, src_idx]
    else:
        class_to_idx = {str(label): idx for idx, label in enumerate(classes_array)}
        for dst_idx, label in enumerate(label_encoder.classes_):
            if str(label) in class_to_idx:
                out[:, dst_idx] = probs[:, class_to_idx[str(label)]]
    denom = out.sum(axis=1, keepdims=True)
    zero = denom.squeeze() == 0
    out[~zero] = out[~zero] / denom[~zero]
    if zero.any():
        out[zero] = 1.0 / len(label_encoder.classes_)
    return out


def ensemble_predict_proba(artifact: dict[str, Any], texts: pd.Series) -> np.ndarray:
    parts = [
        predict_proba_aligned(component["pipeline"], texts, artifact["label_encoder"])
        for component in artifact["pipeline"]
    ]
    probs = np.mean(parts, axis=0)
    return probs / np.clip(probs.sum(axis=1, keepdims=True), 1e-12, None)


def brier(y_true: np.ndarray, probs: np.ndarray) -> float:
    return float(np.mean(np.sum((probs - np.eye(len(LABELS))[y_true]) ** 2, axis=1)))


def ece(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> float:
    pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    correct = pred == y_true
    score = 0.0
    for lo in np.linspace(0, 1, n_bins, endpoint=False):
        hi = lo + 1 / n_bins
        mask = (conf >= lo) & (conf < hi if hi < 1 else conf <= hi)
        if mask.any():
            score += float(mask.mean()) * abs(float(correct[mask].mean()) - float(conf[mask].mean()))
    return float(score)


def metrics_for_mode(
    y_true: np.ndarray,
    pred: np.ndarray,
    probs: np.ndarray,
    *,
    model_name: str,
    mode: str,
    abstained: np.ndarray,
) -> dict[str, Any]:
    covered = ~abstained
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        pred,
        labels=list(range(len(LABELS))),
        zero_division=0,
    )
    covered_accuracy = float(accuracy_score(y_true[covered], pred[covered])) if covered.any() else float("nan")
    return {
        "model": model_name,
        "mode": mode,
        "n": int(len(y_true)),
        "coverage": float(covered.mean()),
        "number_abstained": int(abstained.sum()),
        "full_set_accuracy_abstain_wrong": float((y_true == pred).mean()),
        "accuracy": float((y_true == pred).mean()),
        "covered_accuracy": covered_accuracy,
        "selective_accuracy": covered_accuracy,
        "macro_f1": float(f1_score(y_true, pred, labels=list(range(len(LABELS))), average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, pred, labels=list(range(len(LABELS))), average="weighted", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "mcc": float(matthews_corrcoef(y_true, pred)),
        "min_class_recall": float(np.min(recall)),
        "ece": ece(y_true, probs),
        "brier_score": brier(y_true, probs),
        "mean_confidence": float(probs.max(axis=1).mean()),
        **{
            f"{label.lower()}_{metric}": float(values[idx]) if metric != "support" else int(values[idx])
            for idx, label in enumerate(LABELS)
            for metric, values in {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
            }.items()
        },
        **{f"{label.lower()}_predicted_support": int((pred == idx).sum()) for idx, label in enumerate(LABELS)},
    }


def per_class_for_mode(model_name: str, mode: str, y_true: np.ndarray, pred: np.ndarray) -> pd.DataFrame:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        pred,
        labels=list(range(len(LABELS))),
        zero_division=0,
    )
    return pd.DataFrame(
        {
            "model": model_name,
            "mode": mode,
            "label": LABELS,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support.astype(int),
            "predicted_support": [(pred == idx).sum() for idx in range(len(LABELS))],
        }
    )


def confusion_for_mode(y_true: np.ndarray, pred: np.ndarray, include_abstain: bool) -> dict[str, Any]:
    if include_abstain:
        labels = list(range(len(LABELS))) + [-1]
        pred_names = LABELS + [ABSTAIN_LABEL]
        matrix = confusion_matrix(y_true, pred, labels=labels)
        return {
            LABELS[row_idx]: {pred_names[col_idx]: int(value) for col_idx, value in enumerate(row)}
            for row_idx, row in enumerate(matrix[: len(LABELS)])
        }
    matrix = confusion_matrix(y_true, pred, labels=list(range(len(LABELS))))
    return {
        LABELS[row_idx]: {LABELS[col_idx]: int(value) for col_idx, value in enumerate(row)}
        for row_idx, row in enumerate(matrix)
    }


def mcnemar(correct_a: np.ndarray, correct_b: np.ndarray) -> dict[str, Any]:
    b = int((correct_a & ~correct_b).sum())
    c = int((~correct_a & correct_b).sum())
    n = b + c
    chi2 = 0.0 if n == 0 else (max(abs(b - c) - 1, 0) ** 2) / n
    p_chi2 = float(math.erfc(math.sqrt(chi2 / 2))) if n else 1.0
    if n:
        tail = sum(math.comb(n, k) * 0.5**n for k in range(0, min(b, c) + 1))
        p_exact = min(1.0, 2 * tail)
    else:
        p_exact = 1.0
    return {
        "a_correct_b_wrong": b,
        "a_wrong_b_correct": c,
        "discordant_pairs": n,
        "chi_square_continuity_corrected": float(chi2),
        "p_value_chi_square_df1": p_chi2,
        "p_value_exact_binomial_two_sided": p_exact,
    }


def bootstrap_delta(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray, n_boot: int = 5000) -> dict[str, Any]:
    rng = np.random.default_rng(RANDOM_SEED)
    n = len(y_true)
    accuracy_delta = []
    macro_delta = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        pa = pred_a[idx]
        pb = pred_b[idx]
        accuracy_delta.append(float((pb == yt).mean() - (pa == yt).mean()))
        macro_delta.append(
            float(
                f1_score(yt, pb, labels=list(range(len(LABELS))), average="macro", zero_division=0)
                - f1_score(yt, pa, labels=list(range(len(LABELS))), average="macro", zero_division=0)
            )
        )
    return {
        "n_bootstrap": n_boot,
        "seed": RANDOM_SEED,
        "delta_definition": "model_b_minus_model_a",
        "accuracy_delta_mean": float(np.mean(accuracy_delta)),
        "accuracy_delta_ci_95": [float(np.quantile(accuracy_delta, 0.025)), float(np.quantile(accuracy_delta, 0.975))],
        "macro_f1_delta_mean": float(np.mean(macro_delta)),
        "macro_f1_delta_ci_95": [float(np.quantile(macro_delta, 0.025)), float(np.quantile(macro_delta, 0.975))],
    }


def md_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    out = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(row.get(column, "")) for column in columns) + " |")
    return "\n".join(out)


def fmt(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def load_same_test() -> pd.DataFrame:
    locked = read_csv(LOCKED_REGISTRY)
    locked = locked.loc[locked["evaluable_three_class"].map(is_true) & locked["final_human_label"].isin(LABELS)].copy()
    if len(locked) != 672:
        raise AssertionError(f"Expected 672 V4 locked-test rows, got {len(locked)}")
    return locked.drop_duplicates("comment_id").reset_index(drop=True)


def compute_v2_predictions(locked: pd.DataFrame) -> pd.DataFrame:
    artifact = joblib.load(V2_MODEL_PATH)
    config = read_json(V2_CONFIG_PATH)
    labels = [str(label) for label in artifact["label_encoder"].classes_]
    if labels != LABELS:
        raise AssertionError(f"Unexpected V2 labels: {labels}")
    text = locked["model_text"].where(locked["model_text"].astype(str).str.strip().ne(""), locked["comment_text"]).map(clean_social)
    probs = ensemble_predict_proba(artifact, text)
    pred_id = probs.argmax(axis=1)
    pred_label = np.asarray([ID_TO_LABEL[int(idx)] for idx in pred_id])
    abstained = probs.max(axis=1) < float(config["threshold"])
    native_label = pred_label.astype(object)
    native_label[abstained] = ABSTAIN_LABEL
    return pd.DataFrame(
        {
            "comment_id": locked["comment_id"].to_numpy(),
            "v2_probability_negative": probs[:, LABEL_TO_ID["Negative"]],
            "v2_probability_neutral": probs[:, LABEL_TO_ID["Neutral"]],
            "v2_probability_positive": probs[:, LABEL_TO_ID["Positive"]],
            "v2_confidence": probs.max(axis=1),
            "v2_forced_label": pred_label,
            "v2_native_label": native_label,
            "v2_native_abstained": abstained,
            "v2_threshold": float(config["threshold"]),
        }
    )


def load_v4_predictions(locked: pd.DataFrame) -> pd.DataFrame:
    v4 = read_csv(V4_PREDICTIONS)
    v4 = v4.drop_duplicates("comment_id")
    merged = locked[["comment_id"]].merge(v4, on="comment_id", how="left", validate="one_to_one")
    if merged["predicted_label"].eq("").any() or merged["predicted_label"].isna().any():
        raise AssertionError("Missing V4 predictions for same-test rows.")
    probs = merged[["prob_negative", "prob_neutral", "prob_positive"]].astype(float).to_numpy()
    if not np.allclose(probs.sum(axis=1), 1.0, atol=1e-5):
        raise AssertionError("V4 probabilities do not sum to one.")
    argmax_labels = [ID_TO_LABEL[int(idx)] for idx in probs.argmax(axis=1)]
    if argmax_labels != merged["predicted_label"].tolist():
        raise AssertionError("V4 predicted labels do not match argmax probabilities.")
    return pd.DataFrame(
        {
            "comment_id": merged["comment_id"],
            "v4_probability_negative": probs[:, LABEL_TO_ID["Negative"]],
            "v4_probability_neutral": probs[:, LABEL_TO_ID["Neutral"]],
            "v4_probability_positive": probs[:, LABEL_TO_ID["Positive"]],
            "v4_confidence": probs.max(axis=1),
            "v4_argmax_label": merged["predicted_label"],
            "v4_abstained": False,
        }
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)
    v4_decision = read_json(V4_DECISION_PATH)
    if v4_decision.get("status") != "INDOBERT_V4_NOT_ACCEPTED_KEEP_V2":
        raise AssertionError("Existing strict V4 decision must remain INDOBERT_V4_NOT_ACCEPTED_KEEP_V2.")
    v4_eval = read_json(V4_EVAL_MANIFEST)
    if v4_eval.get("evaluated_once") is not True:
        raise AssertionError("V4 locked test must already be evaluated once; this script does not rerun it.")

    locked = load_same_test()
    v2 = compute_v2_predictions(locked)
    v4 = load_v4_predictions(locked)
    same = locked[
        [
            "comment_id",
            "video_id",
            "comment_text",
            "model_text",
            "product_category",
            "brand_or_video_context",
            "final_human_label",
            "text_cluster_id",
            "exact_duplicate_group_id",
            "near_duplicate_cluster_id",
        ]
    ].merge(v2, on="comment_id", how="left", validate="one_to_one").merge(v4, on="comment_id", how="left", validate="one_to_one")
    y_true = same["final_human_label"].map(LABEL_TO_ID).astype(int).to_numpy()
    v2_forced_pred = same["v2_forced_label"].map(LABEL_TO_ID).astype(int).to_numpy()
    v2_native_pred = np.where(
        same["v2_native_abstained"].astype(bool).to_numpy(),
        -1,
        v2_forced_pred,
    )
    v4_pred = same["v4_argmax_label"].map(LABEL_TO_ID).astype(int).to_numpy()
    v2_probs = same[["v2_probability_negative", "v2_probability_neutral", "v2_probability_positive"]].astype(float).to_numpy()
    v4_probs = same[["v4_probability_negative", "v4_probability_neutral", "v4_probability_positive"]].astype(float).to_numpy()

    same["v2_native_correct_fullset"] = (~same["v2_native_abstained"].astype(bool)) & same["v2_forced_label"].eq(same["final_human_label"])
    same["v2_forced_correct"] = same["v2_forced_label"].eq(same["final_human_label"])
    same["v4_correct"] = same["v4_argmax_label"].eq(same["final_human_label"])
    same["v2_v4_forced_disagreement"] = same["v2_forced_label"].ne(same["v4_argmax_label"])
    same.to_csv(OUT_DIR / "same_test_predictions.csv", index=False, encoding="utf-8-sig")

    modes = [
        ("Legacy V2", "V2_NATIVE_POLICY", v2_native_pred, v2_probs, same["v2_native_abstained"].astype(bool).to_numpy()),
        ("Legacy V2", "V2_FORCED_THREE_CLASS", v2_forced_pred, v2_probs, np.zeros(len(same), dtype=bool)),
        ("IndoBERT V4", "V4_ARGMAX_THREE_CLASS", v4_pred, v4_probs, np.zeros(len(same), dtype=bool)),
    ]
    metrics = [metrics_for_mode(y_true, pred, probs, model_name=model, mode=mode, abstained=abstained) for model, mode, pred, probs, abstained in modes]
    pd.DataFrame(metrics).to_csv(OUT_DIR / "same_test_metrics.csv", index=False, encoding="utf-8-sig")
    per_class = pd.concat([per_class_for_mode(model, mode, y_true, pred) for model, mode, pred, _, _ in modes], ignore_index=True)
    per_class.to_csv(OUT_DIR / "same_test_per_class_metrics.csv", index=False, encoding="utf-8-sig")
    confusion_payload = {
        "V2_NATIVE_POLICY": confusion_for_mode(y_true, v2_native_pred, include_abstain=True),
        "V2_FORCED_THREE_CLASS": confusion_for_mode(y_true, v2_forced_pred, include_abstain=False),
        "V4_ARGMAX_THREE_CLASS": confusion_for_mode(y_true, v4_pred, include_abstain=False),
    }
    write_json(OUT_DIR / "same_test_confusion_matrices.json", confusion_payload)

    disagreement = same.loc[same["v2_v4_forced_disagreement"]].copy()
    disagreement["case_type"] = np.select(
        [
            disagreement["v2_forced_correct"] & ~disagreement["v4_correct"],
            ~disagreement["v2_forced_correct"] & disagreement["v4_correct"],
            disagreement["v2_forced_correct"] & disagreement["v4_correct"],
        ],
        ["v2_correct_v4_wrong", "v4_correct_v2_wrong", "both_correct_different_impossible"],
        default="both_wrong_different",
    )
    disagreement.to_csv(OUT_DIR / "same_test_disagreement.csv", index=False, encoding="utf-8-sig")

    mcnemar_payload = {
        "V2_FORCED_THREE_CLASS_vs_V4_ARGMAX_THREE_CLASS": mcnemar(same["v2_forced_correct"].to_numpy(dtype=bool), same["v4_correct"].to_numpy(dtype=bool)),
        "V2_NATIVE_POLICY_FULLSET_vs_V4_ARGMAX_THREE_CLASS": mcnemar(same["v2_native_correct_fullset"].to_numpy(dtype=bool), same["v4_correct"].to_numpy(dtype=bool)),
    }
    write_json(OUT_DIR / "same_test_mcnemar.json", mcnemar_payload)
    bootstrap_payload = {
        "V2_FORCED_THREE_CLASS_vs_V4_ARGMAX_THREE_CLASS": bootstrap_delta(y_true, v2_forced_pred, v4_pred),
        "V2_NATIVE_POLICY_FULLSET_vs_V4_ARGMAX_THREE_CLASS": bootstrap_delta(y_true, v2_native_pred, v4_pred),
    }
    write_json(OUT_DIR / "same_test_bootstrap_ci.json", bootstrap_payload)

    examples = {
        "v2_correct_v4_wrong": disagreement.loc[disagreement["case_type"].eq("v2_correct_v4_wrong")].head(25).to_dict("records"),
        "v4_correct_v2_wrong": disagreement.loc[disagreement["case_type"].eq("v4_correct_v2_wrong")].head(25).to_dict("records"),
    }
    write_json(OUT_DIR / "same_test_examples.json", examples)

    metric_rows = [
        {
            "mode": row["mode"],
            "coverage": fmt(row["coverage"]),
            "full_accuracy": fmt(row["full_set_accuracy_abstain_wrong"]),
            "covered_accuracy": fmt(row["covered_accuracy"]),
            "macro_f1": fmt(row["macro_f1"]),
            "balanced_accuracy": fmt(row["balanced_accuracy"]),
            "mcc": fmt(row["mcc"]),
            "positive_recall": fmt(row["positive_recall"]),
            "abstained": row["number_abstained"],
        }
        for row in metrics
    ]
    report = f"""# IndoBERT V4 Fair Same-Test Comparison

Generated at UTC: {utc_now()}

Status: `INDOBERT_V4_SAME_TEST_FAIR_COMPARISON`

This report compares the frozen legacy V2 sentiment model and frozen IndoBERT V4 base-reference model on the identical V4 locked-test denominator of 672 human-labeled comments. It does not change the existing strict decision `INDOBERT_V4_NOT_ACCEPTED_KEEP_V2`, does not tune V4 or V2, and does not update the canonical RM2 model.

## Same-Test Denominator

- Rows: 672
- Negative: 160
- Neutral: 294
- Positive: 218
- Dataset hash: `{sha256_dataframe(locked, ["comment_id", "final_human_label"])}`

## Metrics

{md_table(metric_rows, ["mode", "coverage", "full_accuracy", "covered_accuracy", "macro_f1", "balanced_accuracy", "mcc", "positive_recall", "abstained"])}

## Paired Tests

- V2 forced three-class vs V4 argmax McNemar: `{json.dumps(mcnemar_payload["V2_FORCED_THREE_CLASS_vs_V4_ARGMAX_THREE_CLASS"])}`
- V2 native full-set vs V4 argmax McNemar: `{json.dumps(mcnemar_payload["V2_NATIVE_POLICY_FULLSET_vs_V4_ARGMAX_THREE_CLASS"])}`
- Bootstrap CI JSON: `{rel(OUT_DIR / "same_test_bootstrap_ci.json")}`

## Interpretation

This comparison is descriptive and same-test only. V4 improves Positive recall relative to the historical V2 test summary, but this same-test report must not be used to retune V4, V5, thresholds, preprocessing, losses, class weights, or sampling rules. Locked-test errors are archived only for descriptive comparison.
"""
    (OUT_DIR / "SAME_TEST_COMPARISON_REPORT.md").write_text(report, encoding="utf-8")
    DOC_PATH.write_text(report, encoding="utf-8")
    manifest = {
        "status": "INDOBERT_V4_SAME_TEST_FAIR_COMPARISON",
        "created_at_utc": utc_now(),
        "git_sha": git_head(),
        "same_test_rows": int(len(same)),
        "same_test_class_counts": same["final_human_label"].value_counts().reindex(LABELS).fillna(0).astype(int).to_dict(),
        "same_test_dataset_hash": sha256_dataframe(locked, ["comment_id", "final_human_label"]),
        "v2_model_path": rel(V2_MODEL_PATH),
        "v2_model_sha256": sha256_file(V2_MODEL_PATH),
        "v2_config_path": rel(V2_CONFIG_PATH),
        "v2_native_threshold": float(read_json(V2_CONFIG_PATH)["threshold"]),
        "v2_native_threshold_source": "frozen V2 development OOF threshold, not selected on this locked test",
        "v4_model_path": rel(V4_MODEL_DIR),
        "v4_model_sha256": read_json(V4_MODEL_DIR / "base_reference_manifest.json").get("checkpoint_sha256"),
        "v4_decision_preserved": v4_decision.get("status"),
        "canonical_model_changed": False,
        "locked_test_used_for_tuning": False,
        "outputs": [rel(path) for path in sorted(OUT_DIR.glob("*"))],
    }
    write_json(OUT_DIR / "SAME_TEST_COMPARISON_MANIFEST.json", manifest)
    print(json.dumps({"status": manifest["status"], "metrics": metric_rows}, indent=2), flush=True)


if __name__ == "__main__":
    main()
