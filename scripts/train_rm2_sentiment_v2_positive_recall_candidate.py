from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
)
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
import warnings

from rm2_sentiment_v2_positive_recall import (
    LABELS,
    LABEL_TO_ID,
    PROB_COLUMNS,
    apply_threshold_policy,
    normalize_for_model,
    predict_proba_aligned,
)


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "output/rm2_sentiment/validation/human_v2_positive_recall/human_label_registry.csv"
REGISTRY_MANIFEST = ROOT / "output/rm2_sentiment/validation/human_v2_positive_recall/human_label_registry_manifest.json"
FOLD_LEAKAGE = ROOT / "output/rm2_sentiment/validation/human_v2_positive_recall/fold_leakage_audit.csv"
BASELINE = ROOT / "output/rm2_sentiment/experiments/v2_positive_recall/baseline_v2_metrics.json"
OUT_DIR = ROOT / "output/rm2_sentiment/experiments/v2_positive_recall"
MODEL_DIR = ROOT / "output/rm2_sentiment/model/v2_positive_recall_candidate"
NEW_LOCKED_FREEZE_MANIFEST = ROOT / "output/rm2_sentiment/validation/human_v2_positive_recall/new_locked_test_freeze_manifest.json"

OUT_GRID = OUT_DIR / "candidate_grid_manifest.csv"
OUT_OOF = OUT_DIR / "development_oof_predictions.csv"
OUT_SEED_METRICS = OUT_DIR / "development_seed_metrics.csv"
OUT_SUMMARY = OUT_DIR / "development_trial_summary.csv"
OUT_THRESHOLD_GRID = OUT_DIR / "threshold_grid_results.csv"
OUT_SELECTED_THRESHOLD = OUT_DIR / "selected_threshold_policy.json"
OUT_HARD_ERRORS = OUT_DIR / "development_hard_error_audit.csv"
OUT_HARD_ERROR_SUMMARY = OUT_DIR / "development_hard_error_taxonomy_summary.csv"
OUT_STABILITY = OUT_DIR / "development_stability_gate.csv"
OUT_MANIFEST = OUT_DIR / "V2_POSITIVE_RECALL_DEVELOPMENT_MANIFEST.json"

OUT_MODEL = MODEL_DIR / "selected_model.joblib"
OUT_MODEL_CONFIG = MODEL_DIR / "selected_model_config.json"
OUT_MODEL_HASH = MODEL_DIR / "selected_model_sha256.txt"
OUT_ACCEPTANCE = MODEL_DIR / "acceptance_decision.json"

SEEDS = [42, 52, 62, 72, 82]
THRESHOLD_GRID = {
    "positive_threshold": [0.30, 0.35, 0.40, 0.45, 0.50],
    "margin_positive_neutral": [-0.05, 0.00, 0.05, 0.10],
    "margin_positive_negative": [-0.05, 0.00, 0.05],
    "abstention_threshold": [0.35, 0.40, 0.42, 0.45, 0.50],
}
BASELINE_POSITIVE_RECALL = 0.4773


@dataclass(frozen=True)
class CandidateConfig:
    model_id: str
    family: str
    feature_kind: str
    classifier_kind: str
    class_weight_policy: str
    calibration_method: str
    char_analyzer: str = "char_wb"
    char_ngram_min: int = 3
    char_ngram_max: int = 5
    char_min_df: int = 2
    char_max_features: int | None = 100000
    word_ngram_min: int = 1
    word_ngram_max: int = 2
    word_min_df: int = 2
    word_max_features: int | None = 50000
    c_value: float = 1.0
    temperature: float = 1.0


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, dtype=str, keep_default_na=False, low_memory=False)


def sha256_file(path: Path) -> str:
    if not path.exists():
        return ""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dataframe_hash(frame: pd.DataFrame, columns: list[str]) -> str:
    return hashlib.sha256(frame[columns].sort_values(columns).to_csv(index=False).encode("utf-8")).hexdigest()


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def load_development() -> pd.DataFrame:
    registry = read_csv(REGISTRY)
    dev = registry.loc[
        registry["split_family"].eq("development")
        & registry["is_evaluable_three_class"].astype(str).str.lower().eq("true")
        & registry["final_sentiment_label"].isin(LABELS)
    ].copy()
    dev["model_text"] = dev["comment_text_original"].map(normalize_for_model)
    if dev.empty:
        raise RuntimeError("No human-supervised development rows found.")
    if dev["selected_fold"].eq("").any():
        raise AssertionError("Development rows missing selected folds.")
    if dev["comment_id"].duplicated().any():
        raise AssertionError("Duplicate comment_id in development data.")
    if dev["comment_id"].str.upper().str.startswith("INJ").any():
        raise AssertionError("INJ leaked into development data.")
    leakage = read_csv(FOLD_LEAKAGE)
    if leakage["status"].eq("FAIL").any():
        raise AssertionError("Fold leakage audit contains FAIL rows.")
    return dev.reset_index(drop=True)


def class_weight(policy: str, y: pd.Series) -> str | dict[str, float] | None:
    counts = y.value_counts().reindex(LABELS, fill_value=0).astype(float)
    if policy == "none":
        return None
    if policy == "balanced":
        return "balanced"
    inv = len(y) / np.clip(len(LABELS) * counts, 1e-12, None)
    if policy == "inverse_frequency":
        weights = inv
    elif policy == "sqrt_inverse_frequency":
        weights = np.sqrt(inv)
        weights = weights / weights.mean()
    elif policy == "manual_moderate_1":
        weights = pd.Series({"Negative": 1.10, "Neutral": 0.85, "Positive": 1.35})
    elif policy == "manual_moderate_2":
        weights = pd.Series({"Negative": 1.00, "Neutral": 0.90, "Positive": 1.40})
    else:
        raise ValueError(policy)
    return {label: float(weights[label]) for label in LABELS}


def make_features(config: CandidateConfig):
    word = TfidfVectorizer(
        lowercase=True,
        analyzer="word",
        ngram_range=(config.word_ngram_min, config.word_ngram_max),
        min_df=config.word_min_df,
        max_features=config.word_max_features,
        sublinear_tf=True,
        token_pattern=r"(?u)\b\w+\b|[!?]+",
    )
    char = TfidfVectorizer(
        lowercase=True,
        analyzer=config.char_analyzer,
        ngram_range=(config.char_ngram_min, config.char_ngram_max),
        min_df=config.char_min_df,
        max_features=config.char_max_features,
        sublinear_tf=True,
    )
    if config.feature_kind == "word":
        return word
    if config.feature_kind == "char":
        return char
    if config.feature_kind == "word_char":
        return FeatureUnion([("word", word), ("char", char)])
    raise ValueError(config.feature_kind)


def calibrated_classifier(base: LinearSVC, method: str) -> CalibratedClassifierCV:
    try:
        return CalibratedClassifierCV(estimator=base, method=method, cv=3)
    except TypeError:  # pragma: no cover - for older sklearn
        return CalibratedClassifierCV(base_estimator=base, method=method, cv=3)


def make_pipeline(config: CandidateConfig, y_train: pd.Series, seed: int) -> Pipeline:
    weights = class_weight(config.class_weight_policy, y_train)
    if config.classifier_kind == "linearsvc":
        clf = LinearSVC(C=config.c_value, class_weight=weights, random_state=seed, dual="auto")
    elif config.classifier_kind == "calibrated_linearsvc":
        base = LinearSVC(C=config.c_value, class_weight=weights, random_state=seed, dual="auto")
        clf = calibrated_classifier(base, config.calibration_method)
    elif config.classifier_kind == "logistic_regression":
        clf = LogisticRegression(
            C=config.c_value,
            class_weight=weights,
            solver="liblinear",
            max_iter=2000,
            random_state=seed,
        )
    else:
        raise ValueError(config.classifier_kind)
    return Pipeline([("features", make_features(config)), ("clf", clf)])


def candidate_configs() -> list[CandidateConfig]:
    configs: list[CandidateConfig] = []
    policies = ["balanced", "inverse_frequency", "sqrt_inverse_frequency", "manual_moderate_1", "manual_moderate_2"]
    for policy in policies:
        configs.append(
            CandidateConfig(
                model_id=f"char_wb_3_6_linearsvc_{policy}",
                family="TF-IDF character n-gram + LinearSVC",
                feature_kind="char",
                classifier_kind="linearsvc",
                class_weight_policy=policy,
                calibration_method="decision_softmax",
                char_analyzer="char_wb",
                char_ngram_min=3,
                char_ngram_max=6,
                char_min_df=2,
                char_max_features=100000,
            )
        )
        configs.append(
            CandidateConfig(
                model_id=f"word_1_2_linearsvc_{policy}",
                family="TF-IDF word n-gram + LinearSVC",
                feature_kind="word",
                classifier_kind="linearsvc",
                class_weight_policy=policy,
                calibration_method="decision_softmax",
                word_ngram_min=1,
                word_ngram_max=2,
                word_min_df=2,
                word_max_features=50000,
            )
        )
        configs.append(
            CandidateConfig(
                model_id=f"word_char_linearsvc_{policy}",
                family="TF-IDF word-character FeatureUnion + LinearSVC",
                feature_kind="word_char",
                classifier_kind="linearsvc",
                class_weight_policy=policy,
                calibration_method="decision_softmax",
                char_analyzer="char_wb",
                char_ngram_min=3,
                char_ngram_max=5,
                char_min_df=2,
                char_max_features=100000,
                word_ngram_min=1,
                word_ngram_max=2,
                word_min_df=2,
                word_max_features=50000,
            )
        )
    configs.extend(
        [
            CandidateConfig(
                model_id="char_3_5_linearsvc_balanced",
                family="TF-IDF character n-gram + LinearSVC",
                feature_kind="char",
                classifier_kind="linearsvc",
                class_weight_policy="balanced",
                calibration_method="decision_softmax",
                char_analyzer="char",
                char_ngram_min=3,
                char_ngram_max=5,
                char_min_df=2,
                char_max_features=100000,
            ),
            CandidateConfig(
                model_id="word_1_3_linearsvc_manual_moderate_1",
                family="TF-IDF word n-gram + LinearSVC",
                feature_kind="word",
                classifier_kind="linearsvc",
                class_weight_policy="manual_moderate_1",
                calibration_method="decision_softmax",
                word_ngram_min=1,
                word_ngram_max=3,
                word_min_df=2,
                word_max_features=100000,
            ),
            CandidateConfig(
                model_id="word_char_calibrated_sigmoid_balanced",
                family="calibrated LinearSVC",
                feature_kind="word_char",
                classifier_kind="calibrated_linearsvc",
                class_weight_policy="balanced",
                calibration_method="sigmoid",
            ),
            CandidateConfig(
                model_id="word_char_calibrated_sigmoid_manual_moderate_1",
                family="calibrated LinearSVC",
                feature_kind="word_char",
                classifier_kind="calibrated_linearsvc",
                class_weight_policy="manual_moderate_1",
                calibration_method="sigmoid",
            ),
            CandidateConfig(
                model_id="word_char_calibrated_isotonic_balanced",
                family="calibrated LinearSVC",
                feature_kind="word_char",
                classifier_kind="calibrated_linearsvc",
                class_weight_policy="balanced",
                calibration_method="isotonic",
            ),
            CandidateConfig(
                model_id="word_char_logreg_balanced",
                family="Logistic Regression word-character",
                feature_kind="word_char",
                classifier_kind="logistic_regression",
                class_weight_policy="balanced",
                calibration_method="native_probability",
            ),
            CandidateConfig(
                model_id="word_char_logreg_manual_moderate_1",
                family="Logistic Regression word-character",
                feature_kind="word_char",
                classifier_kind="logistic_regression",
                class_weight_policy="manual_moderate_1",
                calibration_method="native_probability",
            ),
            CandidateConfig(
                model_id="word_char_logreg_manual_moderate_2",
                family="Logistic Regression word-character",
                feature_kind="word_char",
                classifier_kind="logistic_regression",
                class_weight_policy="manual_moderate_2",
                calibration_method="native_probability",
            ),
        ]
    )
    return configs


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
    brier = float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))
    return {"ece": float(ece), "brier_score": brier, "mean_confidence": float(conf.mean())}


def metric_bundle(true_labels: pd.Series, predicted_labels: pd.Series, probs: np.ndarray) -> dict[str, object]:
    evaluable_pred = predicted_labels.isin(LABELS)
    coverage = float(evaluable_pred.mean()) if len(predicted_labels) else 0.0
    y_true = true_labels.loc[evaluable_pred].astype(str)
    y_pred = predicted_labels.loc[evaluable_pred].astype(str)
    if len(y_true) == 0:
        precision = recall = f1 = np.zeros(len(LABELS), dtype=float)
        support = np.zeros(len(LABELS), dtype=int)
        metrics = {
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "weighted_f1": 0.0,
            "balanced_accuracy": 0.0,
            "mcc": 0.0,
        }
    else:
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, labels=LABELS, zero_division=0
        )
        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(y_true, y_pred, labels=LABELS, average="weighted", zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
            "mcc": float(matthews_corrcoef(y_true, y_pred)),
        }
    y_true_idx_all = true_labels.map(LABEL_TO_ID).to_numpy(dtype=int)
    metrics.update(
        {
            "coverage": coverage,
            "abstention_rate": 1.0 - coverage,
            "class_collapse": bool(len(set(y_pred.tolist())) < len(LABELS)),
            "minimum_recall": float(np.min(recall)) if len(recall) else 0.0,
            "positive_precision": float(precision[LABEL_TO_ID["Positive"]]),
            "positive_recall": float(recall[LABEL_TO_ID["Positive"]]),
            "positive_f1": float(f1[LABEL_TO_ID["Positive"]]),
            "neutral_recall": float(recall[LABEL_TO_ID["Neutral"]]),
            "negative_recall": float(recall[LABEL_TO_ID["Negative"]]),
            "positive_to_neutral": int(((true_labels == "Positive") & (predicted_labels == "Neutral")).sum()),
            "positive_to_negative": int(((true_labels == "Positive") & (predicted_labels == "Negative")).sum()),
            "positive_to_uncertain": int(((true_labels == "Positive") & (predicted_labels == "Uncertain")).sum()),
            "neutral_to_positive": int(((true_labels == "Neutral") & (predicted_labels == "Positive")).sum()),
            "negative_to_positive": int(((true_labels == "Negative") & (predicted_labels == "Positive")).sum()),
            "all_evaluable_accuracy_abstain_wrong": float(((predicted_labels == true_labels) & evaluable_pred).mean()),
        }
    )
    for idx, label in enumerate(LABELS):
        lower = label.lower()
        metrics[f"{lower}_precision"] = float(precision[idx])
        metrics[f"{lower}_recall"] = float(recall[idx])
        metrics[f"{lower}_f1"] = float(f1[idx])
        metrics[f"{lower}_support"] = int(support[idx])
    metrics.update({f"all_{k}": v for k, v in calibration_metrics(y_true_idx_all, probs).items()})
    if evaluable_pred.any():
        metrics.update(
            {
                f"covered_{k}": v
                for k, v in calibration_metrics(y_true_idx_all[evaluable_pred.to_numpy()], probs[evaluable_pred.to_numpy()]).items()
            }
        )
    else:
        metrics.update({"covered_ece": 0.0, "covered_brier_score": 0.0, "covered_mean_confidence": 0.0})
    return metrics


def threshold_grid_for_oof(oof: pd.DataFrame, model_id: str, seed: int | str) -> tuple[pd.DataFrame, dict[str, object]]:
    probs = oof[PROB_COLUMNS].to_numpy(dtype=float)
    rows: list[dict[str, object]] = []
    for pos_t in THRESHOLD_GRID["positive_threshold"]:
        for margin_pn in THRESHOLD_GRID["margin_positive_neutral"]:
            for margin_png in THRESHOLD_GRID["margin_positive_negative"]:
                for abst_t in THRESHOLD_GRID["abstention_threshold"]:
                    pred = apply_threshold_policy(
                        probs,
                        positive_threshold=pos_t,
                        margin_positive_neutral=margin_pn,
                        margin_positive_negative=margin_png,
                        abstention_threshold=abst_t,
                    )
                    metrics = metric_bundle(oof["true_label"], pred["predicted_label"], probs)
                    safety = (
                        metrics["positive_precision"] >= 0.65
                        and metrics["neutral_recall"] >= 0.70
                        and metrics["coverage"] >= 0.90
                        and metrics["minimum_recall"] >= 0.50
                        and not metrics["class_collapse"]
                    )
                    full_constraints = (
                        safety
                        and metrics["balanced_accuracy"] >= 0.7188
                        and metrics["mcc"] >= 0.6369
                        and metrics["positive_recall"] > BASELINE_POSITIVE_RECALL
                    )
                    rows.append(
                        {
                            "model_id": model_id,
                            "seed": seed,
                            "positive_threshold": pos_t,
                            "margin_positive_neutral": margin_pn,
                            "margin_positive_negative": margin_png,
                            "abstention_threshold": abst_t,
                            "safety_constraints_passed": bool(safety),
                            "full_development_constraints_passed": bool(full_constraints),
                            **metrics,
                        }
                    )
    grid = pd.DataFrame(rows)
    if grid["full_development_constraints_passed"].any():
        subset = grid.loc[grid["full_development_constraints_passed"]].copy()
        status = "FULL_DEVELOPMENT_CONSTRAINTS_MET"
    elif grid["safety_constraints_passed"].any():
        subset = grid.loc[grid["safety_constraints_passed"]].copy()
        status = "SAFETY_CONSTRAINTS_ONLY"
    else:
        subset = grid.copy()
        status = "NO_THRESHOLD_POLICY_MET_SAFETY_CONSTRAINTS"
    selected = subset.sort_values(
        [
            "macro_f1",
            "balanced_accuracy",
            "mcc",
            "positive_f1",
            "positive_recall",
            "positive_precision",
            "coverage",
        ],
        ascending=[False, False, False, False, False, False, False],
    ).iloc[0].to_dict()
    selected["threshold_selection_status"] = status
    return grid, selected


def run_oof_for_config(config: CandidateConfig, data: pd.DataFrame, seed: int) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    folds = sorted(data["selected_fold"].unique(), key=lambda value: int(value))
    for fold in folds:
        train = data.loc[~data["selected_fold"].eq(fold)].copy()
        val = data.loc[data["selected_fold"].eq(fold)].copy()
        model = make_pipeline(config, train["final_sentiment_label"], seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(train["model_text"], train["final_sentiment_label"])
        probs = predict_proba_aligned(model, val["model_text"], labels=LABELS, temperature=config.temperature)
        pred_idx = probs.argmax(axis=1)
        part = val[
            [
                "comment_id",
                "selected_fold",
                "final_sentiment_label",
                "comment_text_original",
                "video_id",
                "brand_or_video_context",
                "text_cluster_id",
                "cv_group_id",
            ]
        ].copy()
        part = part.rename(columns={"final_sentiment_label": "true_label"})
        part["model_id"] = config.model_id
        part["seed"] = seed
        part["argmax_label"] = [LABELS[i] for i in pred_idx]
        part["argmax_confidence"] = probs.max(axis=1)
        for idx, col in enumerate(PROB_COLUMNS):
            part[col] = probs[:, idx]
        rows.append(part)
    return pd.concat(rows, ignore_index=True)


def summarize_seed_metrics(seed_metrics: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "balanced_accuracy",
        "mcc",
        "coverage",
        "positive_precision",
        "positive_recall",
        "positive_f1",
        "neutral_recall",
        "negative_recall",
        "minimum_recall",
        "all_ece",
        "all_brier_score",
    ]
    rows = []
    for model_id, group in seed_metrics.groupby("model_id"):
        first = group.iloc[0]
        row: dict[str, object] = {
            "model_id": model_id,
            "family": first["family"],
            "feature_kind": first["feature_kind"],
            "classifier_kind": first["classifier_kind"],
            "class_weight_policy": first["class_weight_policy"],
            "calibration_method": first["calibration_method"],
            "n_seeds": int(group["seed"].nunique()),
        }
        for col in metric_cols:
            values = pd.to_numeric(group[col], errors="coerce")
            row[f"mean_{col}"] = float(values.mean())
            row[f"median_{col}"] = float(values.median())
            row[f"min_{col}"] = float(values.min())
            row[f"std_{col}"] = float(values.std(ddof=0))
        row["n_class_collapse_seeds"] = int(group["class_collapse"].astype(bool).sum())
        row["stability_gate_passed"] = bool(
            row["std_macro_f1"] <= 0.04
            and row["n_class_collapse_seeds"] == 0
            and row["min_positive_precision"] >= 0.60
            and row["min_positive_recall"] >= 0.50
        )
        row["selection_safety_passed"] = bool(
            row["min_positive_precision"] >= 0.65
            and row["min_neutral_recall"] >= 0.70
            and row["mean_coverage"] >= 0.90
            and row["min_minimum_recall"] >= 0.50
            and row["n_class_collapse_seeds"] == 0
        )
        rows.append(row)
    summary = pd.DataFrame(rows)
    return summary.sort_values(
        [
            "selection_safety_passed",
            "stability_gate_passed",
            "median_macro_f1",
            "median_balanced_accuracy",
            "median_mcc",
            "median_positive_f1",
            "std_macro_f1",
        ],
        ascending=[False, False, False, False, False, False, True],
    ).reset_index(drop=True)


def error_flags(text: object) -> dict[str, bool]:
    lower = str(text).casefold()
    return {
        "komentar_terlalu_pendek": len(lower.split()) <= 5,
        "dukungan_implisit": any(token in lower for token in ["aku pake", "aku pakai", "beli lagi", "langganan", "setia"]),
        "pertanyaan_positif": "?" in lower and any(token in lower for token in ["bagus", "cocok", "aman", "worth"]),
        "testimoni": any(token in lower for token in ["hasil", "perubahan", "glow", "cerah", "jerawat", "mudar"]),
        "promosi": any(token in lower for token in ["checkout", "diskon", "promo", "keranjang", "cod"]),
        "negasi": any(token in lower for token in ["tidak", "nggak", "gak", "ga ", "belum", "bukan", "jangan"]),
        "sarkasme": any(token in lower for token in ["wkwk", "haha", "yaelah", "katanya"]),
        "emoji": any(ord(ch) > 127 for ch in str(text)),
        "mixed_language": any(token in lower for token in ["review", "glow", "skin", "cream", "serum", "worth"]),
        "konteks_video_diperlukan": len(lower.split()) <= 8 or any(token in lower for token in ["ini", "itu", "yang mana"]),
        "konflik_anotasi": False,
        "teks_tidak_cukup": len(lower.strip()) < 4,
    }


def hard_error_audit(selected_oof: pd.DataFrame, policy: dict[str, object]) -> tuple[pd.DataFrame, pd.DataFrame]:
    pred = apply_threshold_policy(
        selected_oof[PROB_COLUMNS].to_numpy(dtype=float),
        positive_threshold=float(policy["positive_threshold"]),
        margin_positive_neutral=float(policy["margin_positive_neutral"]),
        margin_positive_negative=float(policy["margin_positive_negative"]),
        abstention_threshold=float(policy["abstention_threshold"]),
    )
    audit = selected_oof.copy()
    audit["predicted_label"] = pred["predicted_label"]
    audit["prediction_confidence"] = pred["prediction_confidence"]
    wanted = (
        ((audit["true_label"] == "Positive") & audit["predicted_label"].isin(["Neutral", "Negative"]))
        | ((audit["true_label"] == "Neutral") & (audit["predicted_label"] == "Positive"))
        | ((audit["true_label"] == "Negative") & (audit["predicted_label"] == "Positive"))
    )
    audit = audit.loc[wanted].copy()
    audit["error_type"] = audit["true_label"] + "_to_" + audit["predicted_label"]
    flags = audit["comment_text_original"].apply(error_flags).apply(pd.Series)
    audit = pd.concat([audit, flags], axis=1)
    flag_cols = list(flags.columns)
    audit["taxonomy_flags"] = audit[flag_cols].apply(
        lambda row: ";".join([col for col in flag_cols if bool(row[col])]) or "general_misclassification",
        axis=1,
    )
    summary = (
        audit.assign(flag=audit["taxonomy_flags"].str.split(";"))
        .explode("flag")
        .groupby(["error_type", "flag"], dropna=False)
        .size()
        .rename("count")
        .reset_index()
    )
    return audit, summary


def fit_components(config: CandidateConfig, data: pd.DataFrame, seeds: list[int]) -> list[dict[str, object]]:
    components = []
    for seed in seeds:
        model = make_pipeline(config, data["final_sentiment_label"], seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            model.fit(data["model_text"], data["final_sentiment_label"])
        components.append({"seed": seed, "model_id": config.model_id, "pipeline": model, "temperature": config.temperature})
    return components


def update_locked_freeze_manifest(candidate_config: dict[str, object]) -> None:
    if not NEW_LOCKED_FREEZE_MANIFEST.exists():
        return
    payload = json.loads(NEW_LOCKED_FREEZE_MANIFEST.read_text(encoding="utf-8"))
    payload.update(
        {
            "candidate_model_frozen_before_final_locked_test_labels": True,
            "candidate_model_config": candidate_config,
            "candidate_model_hash": sha256_file(OUT_MODEL),
            "candidate_threshold_policy": candidate_config["threshold_policy"],
            "status": "CANDIDATE_FROZEN_BUT_NEW_LOCKED_TEST_PENDING_HUMAN_ADJUDICATION",
        }
    )
    NEW_LOCKED_FREEZE_MANIFEST.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    data = load_development()
    registry_manifest = json.loads(REGISTRY_MANIFEST.read_text(encoding="utf-8"))
    baseline = json.loads(BASELINE.read_text(encoding="utf-8"))
    baseline_metrics = baseline["metrics"]
    configs = candidate_configs()

    grid_reference = {
        "character": {
            "analyzer": ["char_wb", "char"],
            "ngram_range": [(3, 5), (3, 6), (4, 6), (2, 6)],
            "min_df": [2, 3],
            "max_features": [100000, 200000, None],
            "sublinear_tf": True,
        },
        "word": {
            "ngram_range": [(1, 2), (1, 3)],
            "min_df": [2, 3],
            "max_features": [50000, 100000],
            "sublinear_tf": True,
        },
        "class_weight_policies": [
            "balanced",
            "inverse_frequency",
            "sqrt_inverse_frequency",
            "manual_moderate_1",
            "manual_moderate_2",
        ],
        "curated_run_note": "The mandatory V2-family candidates are run across all seeds; full Cartesian feature expansion is recorded here as search space provenance.",
    }

    grid_rows = [asdict(config) | {"random_seeds": ";".join(map(str, SEEDS)), "run_status": "development_oof_run"} for config in configs]
    grid_rows.append(
        {
            "model_id": "v2_frozen_baseline_legacy_diagnostic",
            "family": "V2 frozen baseline",
            "feature_kind": "legacy_frozen",
            "classifier_kind": "legacy_frozen",
            "class_weight_policy": "legacy_frozen",
            "calibration_method": "legacy_threshold",
            "random_seeds": "",
            "run_status": "legacy_diagnostic_only_not_for_selection",
        }
    )
    pd.DataFrame(grid_rows).to_csv(
        OUT_GRID, index=False, encoding="utf-8-sig"
    )

    all_oof: list[pd.DataFrame] = []
    threshold_rows: list[pd.DataFrame] = []
    seed_metric_rows: list[dict[str, object]] = []
    config_by_id = {config.model_id: config for config in configs}

    for config in configs:
        print(f"Running {config.model_id}", flush=True)
        for seed in SEEDS:
            oof = run_oof_for_config(config, data, seed)
            threshold_grid, selected_policy = threshold_grid_for_oof(oof, config.model_id, seed)
            pred = apply_threshold_policy(
                oof[PROB_COLUMNS].to_numpy(dtype=float),
                positive_threshold=float(selected_policy["positive_threshold"]),
                margin_positive_neutral=float(selected_policy["margin_positive_neutral"]),
                margin_positive_negative=float(selected_policy["margin_positive_negative"]),
                abstention_threshold=float(selected_policy["abstention_threshold"]),
            )
            metrics = metric_bundle(oof["true_label"], pred["predicted_label"], oof[PROB_COLUMNS].to_numpy(dtype=float))
            seed_metric_rows.append(
                {
                    "model_id": config.model_id,
                    "seed": seed,
                    "family": config.family,
                    "feature_kind": config.feature_kind,
                    "classifier_kind": config.classifier_kind,
                    "class_weight_policy": config.class_weight_policy,
                    "calibration_method": config.calibration_method,
                    "threshold_selection_status": selected_policy["threshold_selection_status"],
                    "positive_threshold": selected_policy["positive_threshold"],
                    "margin_positive_neutral": selected_policy["margin_positive_neutral"],
                    "margin_positive_negative": selected_policy["margin_positive_negative"],
                    "abstention_threshold": selected_policy["abstention_threshold"],
                    **metrics,
                }
            )
            oof["threshold_selected_positive_threshold"] = selected_policy["positive_threshold"]
            oof["threshold_selected_abstention_threshold"] = selected_policy["abstention_threshold"]
            all_oof.append(oof)
            threshold_rows.append(threshold_grid)

    oof_all = pd.concat(all_oof, ignore_index=True)
    threshold_all = pd.concat(threshold_rows, ignore_index=True)
    seed_metrics = pd.DataFrame(seed_metric_rows)
    summary = summarize_seed_metrics(seed_metrics)

    if summary["selection_safety_passed"].any():
        selected_model_id = summary.loc[summary["selection_safety_passed"]].iloc[0]["model_id"]
    else:
        selected_model_id = summary.iloc[0]["model_id"]
    selected_config = config_by_id[str(selected_model_id)]
    selected_seed_rows = seed_metrics.loc[seed_metrics["model_id"].eq(selected_model_id)].sort_values(
        ["macro_f1", "balanced_accuracy", "mcc", "positive_f1"], ascending=False
    )
    top_seeds = [int(seed) for seed in selected_seed_rows["seed"].head(3)]

    selected_seed_oofs = []
    for seed in top_seeds:
        selected_seed_oofs.append(
            oof_all.loc[oof_all["model_id"].eq(selected_model_id) & oof_all["seed"].eq(seed)].set_index("comment_id")
        )
    common_ids = sorted(set.intersection(*(set(frame.index) for frame in selected_seed_oofs)))
    ensemble = selected_seed_oofs[0].loc[common_ids].copy()
    for col in PROB_COLUMNS:
        ensemble[col] = np.mean([frame.loc[common_ids, col].astype(float).to_numpy() for frame in selected_seed_oofs], axis=0)
    ensemble = ensemble.reset_index()
    ensemble["model_id"] = "ensemble_top3_seed_probability_average"
    ensemble["seed"] = "top3:" + ";".join(map(str, top_seeds))
    threshold_ensemble, selected_ensemble_policy = threshold_grid_for_oof(
        ensemble,
        "ensemble_top3_seed_probability_average",
        "top3:" + ";".join(map(str, top_seeds)),
    )
    threshold_all = pd.concat([threshold_all, threshold_ensemble], ignore_index=True)
    pred_ensemble = apply_threshold_policy(
        ensemble[PROB_COLUMNS].to_numpy(dtype=float),
        positive_threshold=float(selected_ensemble_policy["positive_threshold"]),
        margin_positive_neutral=float(selected_ensemble_policy["margin_positive_neutral"]),
        margin_positive_negative=float(selected_ensemble_policy["margin_positive_negative"]),
        abstention_threshold=float(selected_ensemble_policy["abstention_threshold"]),
    )
    ensemble_metrics = metric_bundle(ensemble["true_label"], pred_ensemble["predicted_label"], ensemble[PROB_COLUMNS].to_numpy(dtype=float))
    seed_metrics = pd.concat(
        [
            seed_metrics,
            pd.DataFrame(
                [
                    {
                        "model_id": "ensemble_top3_seed_probability_average",
                        "seed": "top3:" + ";".join(map(str, top_seeds)),
                        "family": "ensemble kandidat terbaik",
                        "feature_kind": selected_config.feature_kind,
                        "classifier_kind": selected_config.classifier_kind,
                        "class_weight_policy": selected_config.class_weight_policy,
                        "calibration_method": "probability_average",
                        "threshold_selection_status": selected_ensemble_policy["threshold_selection_status"],
                        "positive_threshold": selected_ensemble_policy["positive_threshold"],
                        "margin_positive_neutral": selected_ensemble_policy["margin_positive_neutral"],
                        "margin_positive_negative": selected_ensemble_policy["margin_positive_negative"],
                        "abstention_threshold": selected_ensemble_policy["abstention_threshold"],
                        **ensemble_metrics,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    ensemble_summary = pd.DataFrame(
        [
            {
                "model_id": "ensemble_top3_seed_probability_average",
                "family": "ensemble kandidat terbaik",
                "feature_kind": selected_config.feature_kind,
                "classifier_kind": selected_config.classifier_kind,
                "class_weight_policy": selected_config.class_weight_policy,
                "calibration_method": "probability_average",
                "n_seeds": len(top_seeds),
                **{f"mean_{k}": v for k, v in ensemble_metrics.items() if isinstance(v, (int, float, bool, np.integer, np.floating, np.bool_))},
                **{f"median_{k}": v for k, v in ensemble_metrics.items() if isinstance(v, (int, float, bool, np.integer, np.floating, np.bool_))},
                **{f"min_{k}": v for k, v in ensemble_metrics.items() if isinstance(v, (int, float, bool, np.integer, np.floating, np.bool_))},
                "std_macro_f1": float(selected_seed_rows["macro_f1"].std(ddof=0)),
                "n_class_collapse_seeds": int(selected_seed_rows["class_collapse"].astype(bool).sum()),
                "stability_gate_passed": bool(
                    selected_seed_rows["macro_f1"].std(ddof=0) <= 0.04
                    and not selected_seed_rows["class_collapse"].astype(bool).any()
                    and selected_seed_rows["positive_precision"].min() >= 0.60
                    and selected_seed_rows["positive_recall"].min() >= 0.50
                ),
                "selection_safety_passed": bool(
                    ensemble_metrics["positive_precision"] >= 0.65
                    and ensemble_metrics["neutral_recall"] >= 0.70
                    and ensemble_metrics["coverage"] >= 0.90
                    and ensemble_metrics["minimum_recall"] >= 0.50
                    and not ensemble_metrics["class_collapse"]
                ),
            }
        ]
    )
    summary["selected_for_ensemble_basis"] = summary["model_id"].eq(selected_model_id)
    summary["selected_for_freeze"] = False
    summary = pd.concat([summary, ensemble_summary], ignore_index=True, sort=False)
    summary["selected_for_freeze"] = summary["model_id"].eq("ensemble_top3_seed_probability_average")
    baseline_row = {
        "model_id": "v2_frozen_baseline_legacy_diagnostic",
        "family": "V2 frozen baseline",
        "feature_kind": "legacy_frozen",
        "classifier_kind": "legacy_frozen",
        "class_weight_policy": "legacy_frozen",
        "calibration_method": "legacy_threshold",
        "n_seeds": 0,
        "mean_accuracy": baseline_metrics["accuracy_covered"],
        "median_accuracy": baseline_metrics["accuracy_covered"],
        "mean_macro_f1": baseline_metrics["macro_f1_covered"],
        "median_macro_f1": baseline_metrics["macro_f1_covered"],
        "mean_balanced_accuracy": baseline_metrics["balanced_accuracy"],
        "median_balanced_accuracy": baseline_metrics["balanced_accuracy"],
        "mean_mcc": baseline_metrics["mcc"],
        "median_mcc": baseline_metrics["mcc"],
        "mean_coverage": baseline_metrics["coverage"],
        "median_coverage": baseline_metrics["coverage"],
        "mean_positive_precision": baseline_metrics["positive_precision"],
        "median_positive_precision": baseline_metrics["positive_precision"],
        "mean_positive_recall": baseline_metrics["positive_recall"],
        "median_positive_recall": baseline_metrics["positive_recall"],
        "mean_positive_f1": baseline_metrics["positive_f1"],
        "median_positive_f1": baseline_metrics["positive_f1"],
        "selection_safety_passed": False,
        "stability_gate_passed": False,
        "selected_for_ensemble_basis": False,
        "selected_for_freeze": False,
        "selection_notes": "LEGACY_DIAGNOSTIC_TEST_ALREADY_OPENED; not used for training, tuning, or selection.",
    }
    summary = pd.concat([summary, pd.DataFrame([baseline_row])], ignore_index=True, sort=False)

    hard_errors, hard_error_summary = hard_error_audit(ensemble, selected_ensemble_policy)
    components = fit_components(selected_config, data, top_seeds)
    artifact = {
        "status": "V2_POSITIVE_RECALL_CANDIDATE_FROZEN_PENDING_NEW_LOCKED_TEST",
        "model_name": "v2_positive_recall_candidate",
        "base_model_id": selected_model_id,
        "ensemble_model_id": "ensemble_top3_seed_probability_average",
        "components": components,
        "labels": LABELS,
        "threshold_policy": {
            "positive_threshold": float(selected_ensemble_policy["positive_threshold"]),
            "margin_positive_neutral": float(selected_ensemble_policy["margin_positive_neutral"]),
            "margin_positive_negative": float(selected_ensemble_policy["margin_positive_negative"]),
            "abstention_threshold": float(selected_ensemble_policy["abstention_threshold"]),
        },
        "preprocessing": "Unicode NFKC, HTML unescape, URL/mention sentinel, whitespace normalization; negation, emoji, emoticon, elongated words, brand/product names, numbers, question marks, exclamation marks, and mixed Indonesian-English text are retained.",
        "development_data_hash": dataframe_hash(data, ["comment_id", "final_sentiment_label", "selected_fold", "cv_group_id", "model_text"]),
        "locked_test_used_for_training_or_selection": False,
        "low_confidence_to_positive_rule": False,
    }
    joblib.dump(artifact, OUT_MODEL)
    model_hash = sha256_file(OUT_MODEL)
    OUT_MODEL_HASH.write_text(model_hash + "\n", encoding="utf-8")

    data_gate_passed = bool(registry_manifest["target_development_data_gate_passed"])
    stability_gate_passed = bool(ensemble_summary["stability_gate_passed"].iloc[0])
    safety_gate_passed = bool(ensemble_summary["selection_safety_passed"].iloc[0])
    new_locked_ready = False
    acceptance_status = "V2_POSITIVE_RECALL_CANDIDATE_NOT_ACCEPTED_KEEP_ORIGINAL_V2"
    candidate_config = {
        "status": artifact["status"],
        "acceptance_status": acceptance_status,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_head_at_training": git_head(),
        "base_candidate_config": asdict(selected_config),
        "top_ensemble_seeds": top_seeds,
        "threshold_policy": artifact["threshold_policy"],
        "threshold_selection_data_scope": "OOF_DEVELOPMENT_ONLY",
        "locked_test_used_for_training_or_selection": False,
        "legacy_locked_test_used_for_training_or_selection": False,
        "low_confidence_to_positive_rule": False,
        "unknown_or_failed_recognition_to_positive_rule": False,
        "class_weight_policy": selected_config.class_weight_policy,
        "development_rows": int(len(data)),
        "development_class_counts": {label: int(data["final_sentiment_label"].eq(label).sum()) for label in LABELS},
        "development_data_target_gate_passed": data_gate_passed,
        "development_stability_gate_passed": stability_gate_passed,
        "development_selection_safety_gate_passed": safety_gate_passed,
        "new_locked_test_ready": new_locked_ready,
        "candidate_model_sha256": model_hash,
        "baseline_v2_legacy_metrics": baseline_metrics,
        "selected_development_ensemble_metrics": ensemble_metrics,
        "selection_limitations": [
            "Development target data counts are not yet sufficient for the requested Positive/Negative minima."
            if not data_gate_passed
            else "",
            "New locked test final human adjudication is not available, so the candidate cannot be accepted or promoted.",
        ],
        "feature_grid_reference": grid_reference,
        "package_versions": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "sklearn": __import__("sklearn").__version__,
            "joblib": joblib.__version__,
        },
    }
    OUT_MODEL_CONFIG.write_text(json.dumps(candidate_config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    OUT_ACCEPTANCE.write_text(
        json.dumps(
            {
                "status": acceptance_status,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "gates": {
                    "development_data_target_gate_passed": data_gate_passed,
                    "development_stability_gate_passed": stability_gate_passed,
                    "development_selection_safety_gate_passed": safety_gate_passed,
                    "new_locked_test_ready": new_locked_ready,
                    "full_inference_allowed": False,
                    "promote_to_final_allowed": False,
                },
                "reason": "Candidate remains unaccepted until development data target and new locked-test adjudication/evaluation pass.",
                "no_failed_recognition_to_positive": True,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    update_locked_freeze_manifest(candidate_config)

    oof_all.to_csv(OUT_OOF, index=False, encoding="utf-8-sig")
    seed_metrics.to_csv(OUT_SEED_METRICS, index=False, encoding="utf-8-sig")
    summary.to_csv(OUT_SUMMARY, index=False, encoding="utf-8-sig")
    threshold_all.to_csv(OUT_THRESHOLD_GRID, index=False, encoding="utf-8-sig")
    OUT_SELECTED_THRESHOLD.write_text(json.dumps(selected_ensemble_policy, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    hard_errors.to_csv(OUT_HARD_ERRORS, index=False, encoding="utf-8-sig")
    hard_error_summary.to_csv(OUT_HARD_ERROR_SUMMARY, index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [
            {
                "gate": "std_macro_f1_across_selected_top3_seeds",
                "threshold": "<=0.04",
                "observed": float(selected_seed_rows["macro_f1"].head(3).std(ddof=0)),
                "passed": float(selected_seed_rows["macro_f1"].head(3).std(ddof=0)) <= 0.04,
            },
            {
                "gate": "no_seed_class_collapse",
                "threshold": True,
                "observed": bool(not selected_seed_rows["class_collapse"].astype(bool).any()),
                "passed": bool(not selected_seed_rows["class_collapse"].astype(bool).any()),
            },
            {
                "gate": "minimum_positive_precision_across_seeds",
                "threshold": ">=0.60",
                "observed": float(selected_seed_rows["positive_precision"].min()),
                "passed": float(selected_seed_rows["positive_precision"].min()) >= 0.60,
            },
            {
                "gate": "minimum_positive_recall_across_seeds",
                "threshold": ">=0.50",
                "observed": float(selected_seed_rows["positive_recall"].min()),
                "passed": float(selected_seed_rows["positive_recall"].min()) >= 0.50,
            },
        ]
    ).to_csv(OUT_STABILITY, index=False, encoding="utf-8-sig")

    manifest = {
        "status": "V2_POSITIVE_RECALL_DEVELOPMENT_COMPLETE_CANDIDATE_NOT_ACCEPTED",
        "acceptance_status": acceptance_status,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "registry": REGISTRY.relative_to(ROOT).as_posix(),
        "model_dir": MODEL_DIR.relative_to(ROOT).as_posix(),
        "development_selection_basis": "OOF development predictions only",
        "selected_base_candidate": selected_model_id,
        "selected_ensemble_seeds": top_seeds,
        "selected_threshold_policy": artifact["threshold_policy"],
        "candidate_model_sha256": model_hash,
        "locked_test_used_for_training_or_selection": False,
        "full_inference_generated": False,
        "positive_shift_policy": "No failed-recognition/low-confidence/unknown/Neutral/Uncertain/HCC/promotion shortcut to Positive.",
        "development_data_target_gate_passed": data_gate_passed,
        "new_locked_test_ready": new_locked_ready,
        "outputs": {
            "candidate_grid_manifest": OUT_GRID.relative_to(ROOT).as_posix(),
            "development_oof_predictions": OUT_OOF.relative_to(ROOT).as_posix(),
            "development_seed_metrics": OUT_SEED_METRICS.relative_to(ROOT).as_posix(),
            "development_trial_summary": OUT_SUMMARY.relative_to(ROOT).as_posix(),
            "threshold_grid_results": OUT_THRESHOLD_GRID.relative_to(ROOT).as_posix(),
            "model_artifact": OUT_MODEL.relative_to(ROOT).as_posix(),
            "model_config": OUT_MODEL_CONFIG.relative_to(ROOT).as_posix(),
        },
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps({
        "status": acceptance_status,
        "selected_base_candidate": selected_model_id,
        "top_ensemble_seeds": top_seeds,
        "threshold_policy": artifact["threshold_policy"],
        "development_ensemble_macro_f1": ensemble_metrics["macro_f1"],
        "development_ensemble_positive_precision": ensemble_metrics["positive_precision"],
        "development_ensemble_positive_recall": ensemble_metrics["positive_recall"],
        "development_data_target_gate_passed": data_gate_passed,
        "new_locked_test_ready": new_locked_ready,
    }, indent=2))


if __name__ == "__main__":
    main()
