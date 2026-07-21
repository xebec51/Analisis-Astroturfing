from __future__ import annotations

import hashlib
import json
import platform
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
)
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedGroupKFold
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
import warnings


ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "dataset.csv"
COMMENT_SENTIMENT_PATH = ROOT / "output/rm2_sentiment/tables/comment_sentiment.csv"
HCC_NODES_PATH = ROOT / "output/gephi/gephi_hcc_nodes.csv"
V1_VALIDATED_PATH = ROOT / "output/rm2_sentiment/human_validation/sentiment_human_annotation_validated.csv"
V2_VALIDATED_PATH = ROOT / "output/rm2_sentiment/human_validation_v2/sentiment_human_annotation_v2_validated.csv"
V2_ADJUDICATION_FINAL_PATH = ROOT / "output/rm2_sentiment/human_validation_v2/sentiment_v2_adjudication_template_final.csv"
V2_LOCKED_MANIFEST_PATH = ROOT / "output/rm2_sentiment/human_validation_v2/locked_test_v2_manifest.csv"
V2_SYNTHETIC_AUDIT_PATH = ROOT / "output/rm2_sentiment/human_validation_v2/sentiment_v2_synthetic_id_audit.csv"

OUT_DIR = ROOT / "output/rm2_sentiment/model_v2"
OUT_MODEL = OUT_DIR / "selected_model_development_frozen.joblib"
OUT_MODEL_CONFIG = OUT_DIR / "selected_model_development_frozen_config.json"
OUT_MODEL_HASH = OUT_DIR / "selected_model_development_frozen_hash.txt"
OUT_MANIFEST = OUT_DIR / "development_model_manifest.json"
OUT_SELECTION = OUT_DIR / "development_model_selection_summary.csv"
OUT_CV = OUT_DIR / "development_cv_metrics.csv"
OUT_CONFUSION = OUT_DIR / "development_confusion_matrix.csv"
OUT_OOF = OUT_DIR / "development_oof_predictions.csv"
OUT_ERROR = OUT_DIR / "development_error_analysis.csv"
OUT_PROVENANCE = OUT_DIR / "development_training_pool_provenance.csv"
OUT_READINESS = OUT_DIR / "final_locked_test_evaluation_readiness.csv"

OUT_PER_CLASS = OUT_DIR / "development_per_class_metrics.csv"
OUT_THRESHOLD = OUT_DIR / "development_threshold_selection.csv"
OUT_GROUP_DIAGNOSTICS = OUT_DIR / "development_group_diagnostic_metrics.csv"
OUT_GROUPED_CV = OUT_DIR / "development_grouped_cv_metrics.csv"
OUT_FOLD_DISTRIBUTION = OUT_DIR / "development_fold_class_distribution.csv"
OUT_CALIBRATION = OUT_DIR / "development_calibration_metrics.csv"
OUT_BOOTSTRAP = OUT_DIR / "development_bootstrap_ci.csv"
OUT_POOL_SUMMARY = OUT_DIR / "development_pool_summary.csv"
OUT_EXCLUDED_LOCKED_IDS = OUT_DIR / "excluded_locked_test_v2_comment_ids.csv"
OUT_PACKAGE_VERSIONS = OUT_DIR / "development_package_versions.csv"

STATUS = "DEVELOPMENT_MODEL_FROZEN_PENDING_LOCKED_TEST"
READINESS_STATUS = "BLOCKED_WAITING_FOR_8_HUMAN_ANNOTATED_REPLACEMENTS"
RANDOM_SEED = 20260721
LABELS = ["Negative", "Neutral", "Positive"]
NON_EVALUABLE_LABELS = {"Uncertain", "No Text", ""}
SYNTHETIC_PATTERN = re.compile(r"(^INJ|synthetic|challenge)", flags=re.IGNORECASE)


class TextColumnSelector(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):  # noqa: N803
        return self

    def transform(self, X):  # noqa: N803
        if isinstance(X, pd.DataFrame):
            return X.iloc[:, 0].fillna("").astype(str).to_numpy()
        return pd.Series(X).fillna("").astype(str).to_numpy()


@dataclass(frozen=True)
class CandidateConfig:
    model_id: str
    preprocessing: str
    feature_kind: str
    classifier_kind: str
    class_weight: str | None
    c_value: float
    max_features: int | None = 40000


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, dtype=str, low_memory=False).fillna("")


def normalize_blank(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null", "<na>"}:
        return ""
    return text


def normalize_username(value: object) -> str:
    return re.sub(r"\s+", "", normalize_blank(value).lower().lstrip("@"))


def is_synthetic_id(value: object) -> bool:
    return bool(SYNTHETIC_PATTERN.search(normalize_blank(value)))


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def dataframe_hash(df: pd.DataFrame, columns: list[str] | None = None) -> str:
    if columns:
        data = df[columns].copy()
    else:
        data = df.copy()
    csv_bytes = data.sort_values(list(data.columns)).to_csv(index=False).encode("utf-8")
    return hashlib.sha256(csv_bytes).hexdigest()


def clean_social(text: object) -> str:
    s = normalize_blank(text).lower()
    s = re.sub(r"https?://\S+|www\.\S+", " URL ", s)
    s = re.sub(r"@\w+", " USERMENTION ", s)
    s = re.sub(r"#(\w+)", r" \1 ", s)
    s = re.sub(r"([!?.,])\1+", r"\1", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def clean_minimal(text: object) -> str:
    return re.sub(r"\s+", " ", normalize_blank(text)).strip()


def text_flags(text: object) -> dict[str, bool]:
    s = normalize_blank(text)
    lower = s.lower()
    question_terms = ["?", "apa", "gimana", "gmna", "kenapa", "kapan", "boleh", "aman", "cocok", "bagaimana", "kah"]
    negation_terms = ["tidak", "nggak", "ga ", "gak", "belum", "bukan", "jangan", "tak"]
    slang_terms = ["yg", "gk", "ga", "gak", "bgt", "banget", "dong", "kak", "kk", "gw", "gue", "aku", "nih", "sih"]
    ascii_chars = sum(1 for ch in s if ord(ch) < 128)
    non_ascii = len(s) - ascii_chars
    return {
        "question": any(term in lower for term in question_terms),
        "emoji": bool(non_ascii > 0),
        "negation": any(term in lower for term in negation_terms),
        "slang": any(re.search(rf"\b{re.escape(term)}\b", lower) for term in slang_terms if term.strip()),
        "code_mixing": bool(re.search(r"\b(the|for|sure|try|review|serum|cream|glow|skin|best)\b", lower) and re.search(r"\b(aku|kak|yg|banget|gak|cocok)\b", lower)),
        "very_short": len(lower.split()) <= 3,
        "mixed_sentiment": bool(re.search(r"\b(tapi|but|namun|cuma|walau)\b", lower)),
    }


def load_hcc_users() -> set[str]:
    hcc = read_csv(HCC_NODES_PATH)
    return set(hcc["id"].map(normalize_username))


def load_dataset_context() -> pd.DataFrame:
    data = read_csv(DATASET_PATH)
    data["comment_id"] = data["comment_id"].map(normalize_blank)
    data["username_norm"] = data["username"].map(normalize_username)
    hcc_users = load_hcc_users()
    context = data.drop_duplicates("comment_id").copy()
    context["is_hcc"] = context["username_norm"].isin(hcc_users)
    context["text_dataset"] = context["text"]
    return context[
        [
            "comment_id",
            "username",
            "username_norm",
            "video_id",
            "product_category",
            "comment_type",
            "timestamp",
            "text_dataset",
            "is_hcc",
        ]
    ]


def normalize_v1(v1: pd.DataFrame) -> pd.DataFrame:
    out = v1.copy()
    out["comment_id"] = out["comment_id"].map(normalize_blank)
    out["human_label"] = out["adjudicated_human_label"].map(normalize_blank)
    out["source_version"] = "V1"
    out["sample_role_original"] = out["sample_set"].map(normalize_blank)
    out["sample_role_final"] = np.where(out["sample_set"].eq("locked_test"), "historical_test_v1", "development_v1")
    out["human_label_source"] = "sentiment_human_annotation_validated.csv"
    out["sentiment_target"] = ""
    out["complaint_scope"] = ""
    out["annotator_agreement_status"] = np.where(
        out["annotator_1_label"].map(normalize_blank).eq(out["annotator_2_label"].map(normalize_blank)),
        "agreement",
        "adjudicated_disagreement",
    )
    out["adjudication_status"] = np.where(out["human_label"].ne(""), "adjudicated", "missing")
    return out[
        [
            "comment_id",
            "comment_text_original",
            "video_id",
            "brand_or_video_context",
            "human_label",
            "sentiment_target",
            "complaint_scope",
            "source_version",
            "sample_role_original",
            "sample_role_final",
            "human_label_source",
            "annotator_agreement_status",
            "adjudication_status",
        ]
    ]


def normalize_v2(v2: pd.DataFrame) -> pd.DataFrame:
    out = v2.copy()
    out["comment_id"] = out["comment_id"].map(normalize_blank)
    out["human_label"] = out["final_sentiment_label"].map(normalize_blank)
    out["source_version"] = "V2"
    out["sample_role_original"] = out["sample_role"].map(normalize_blank)
    out["sample_role_final"] = out["sample_role"].map(normalize_blank)
    out["human_label_source"] = "sentiment_human_annotation_v2_validated.csv"
    out["sentiment_target"] = out["final_sentiment_target"].map(normalize_blank)
    out["complaint_scope"] = out["final_complaint_scope"].map(normalize_blank)
    out["annotator_agreement_status"] = np.where(
        out["annotator_1_label"].map(normalize_blank).eq(out["annotator_2_label"].map(normalize_blank)),
        "agreement",
        "adjudicated_disagreement",
    )
    out["adjudication_status"] = np.where(out["human_label"].ne(""), "adjudicated", "missing")
    return out[
        [
            "comment_id",
            "comment_text_original",
            "video_id",
            "brand_or_video_context",
            "human_label",
            "sentiment_target",
            "complaint_scope",
            "source_version",
            "sample_role_original",
            "sample_role_final",
            "human_label_source",
            "annotator_agreement_status",
            "adjudication_status",
        ]
    ]


def load_locked_and_synthetic_ids() -> tuple[set[str], set[str], set[str]]:
    locked_manifest = read_csv(V2_LOCKED_MANIFEST_PATH)
    locked_manifest_ids = set(locked_manifest["comment_id"].map(normalize_blank))
    v2_validated = read_csv(V2_VALIDATED_PATH)
    locked_observational = set(v2_validated.loc[v2_validated["sample_role"].eq("locked_test_v2"), "comment_id"].map(normalize_blank))
    synthetic_ids: set[str] = set()
    if V2_SYNTHETIC_AUDIT_PATH.exists():
        audit = read_csv(V2_SYNTHETIC_AUDIT_PATH)
        synthetic_ids |= set(audit["comment_id"].map(normalize_blank))
    if V2_ADJUDICATION_FINAL_PATH.exists():
        adj = read_csv(V2_ADJUDICATION_FINAL_PATH)
        synthetic_ids |= {cid for cid in adj["comment_id"].map(normalize_blank) if is_synthetic_id(cid)}
    return locked_manifest_ids, locked_observational, synthetic_ids


def build_provenance_and_training_pool() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    context = load_dataset_context()
    context_by_id = context.drop_duplicates("comment_id").set_index("comment_id")
    v1_norm = normalize_v1(read_csv(V1_VALIDATED_PATH))
    v2_norm = normalize_v2(read_csv(V2_VALIDATED_PATH))
    locked_manifest_ids, locked_observational_ids, synthetic_ids = load_locked_and_synthetic_ids()

    combined = pd.concat([v1_norm, v2_norm], ignore_index=True)
    combined["comment_id"] = combined["comment_id"].map(normalize_blank)
    combined = combined.loc[combined["comment_id"].ne("")].copy()
    combined["is_synthetic"] = combined["comment_id"].isin(synthetic_ids) | combined["comment_id"].map(is_synthetic_id)
    combined["is_locked_test_v2"] = combined["comment_id"].isin(locked_manifest_ids)
    combined["is_locked_test_v2_observational"] = combined["comment_id"].isin(locked_observational_ids)
    combined["has_valid_human_label"] = combined["human_label"].isin(LABELS)

    priority = {
        "development_v2": 0,
        "development_v1": 1,
        "historical_test_v1": 2,
        "locked_test_v2": 99,
    }
    combined["priority"] = combined["sample_role_final"].map(priority).fillna(50).astype(int)
    combined["candidate_in_training_pool"] = (
        combined["has_valid_human_label"]
        & ~combined["is_synthetic"]
        & ~combined["is_locked_test_v2"]
        & combined["sample_role_final"].isin(["development_v2", "development_v1", "historical_test_v1"])
    )

    selected_ids = (
        combined.loc[combined["candidate_in_training_pool"]]
        .sort_values(["priority", "source_version", "comment_id"])
        .drop_duplicates("comment_id", keep="first")["comment_id"]
    )
    selected_set = set(selected_ids)

    duplicate_counts = combined.groupby("comment_id")["comment_id"].transform("count")
    combined["included_in_training"] = combined["comment_id"].isin(selected_set) & combined["candidate_in_training_pool"]
    # Keep only the highest-priority duplicate in training.
    selected_rows = (
        combined.loc[combined["included_in_training"]]
        .sort_values(["priority", "source_version", "comment_id"])
        .drop_duplicates("comment_id", keep="first")
    )
    keep_index = set(selected_rows.index)
    combined["included_in_training"] = combined.index.isin(keep_index)

    def reason(row: pd.Series) -> str:
        if row["included_in_training"]:
            return ""
        if row["is_synthetic"]:
            return "synthetic_or_injected_id"
        if row["is_locked_test_v2"]:
            return "excluded_locked_test_v2"
        if not row["has_valid_human_label"]:
            return "non_evaluable_sentiment_label"
        if duplicate_counts.loc[row.name] > 1 and row["comment_id"] in selected_set:
            return "duplicate_superseded_by_higher_priority_human_label"
        if row["sample_role_final"] not in {"development_v2", "development_v1", "historical_test_v1"}:
            return "not_development_training_role"
        return "not_selected"

    combined["exclusion_reason"] = combined.apply(reason, axis=1)
    combined = combined.drop(columns=["priority", "candidate_in_training_pool", "has_valid_human_label"])

    merged = combined.merge(context, on="comment_id", how="left", suffixes=("", "_dataset"))
    merged["comment_text_original"] = np.where(
        merged["comment_text_original"].map(normalize_blank).ne(""),
        merged["comment_text_original"],
        merged["text_dataset"],
    )
    merged["video_id"] = np.where(merged["video_id"].map(normalize_blank).ne(""), merged["video_id"], merged["video_id_dataset"])
    merged["brand_or_video_context"] = np.where(
        merged["brand_or_video_context"].map(normalize_blank).ne(""),
        merged["brand_or_video_context"],
        merged["product_category"],
    )
    merged["is_hcc"] = merged["is_hcc"].fillna(False).astype(bool)

    flags = merged["comment_text_original"].apply(text_flags).apply(pd.Series)
    merged = pd.concat([merged, flags], axis=1)
    merged["text_minimal_raw"] = merged["comment_text_original"].map(clean_minimal)
    merged["text_social_normalized"] = merged["comment_text_original"].map(clean_social)

    train = merged.loc[merged["included_in_training"]].copy()
    train = train.loc[train["human_label"].isin(LABELS)].drop_duplicates("comment_id").reset_index(drop=True)
    train["label"] = train["human_label"]

    meta = {
        "locked_manifest_ids": locked_manifest_ids,
        "locked_observational_ids": locked_observational_ids,
        "synthetic_ids": synthetic_ids,
        "context_rows": len(context),
    }
    return merged.reset_index(drop=True), train, meta


def make_pipeline(config: CandidateConfig) -> Pipeline:
    word = TfidfVectorizer(
        preprocessor=None,
        tokenizer=None,
        lowercase=False,
        analyzer="word",
        ngram_range=(1, 2),
        min_df=1,
        max_features=config.max_features,
        sublinear_tf=True,
    )
    char = TfidfVectorizer(
        preprocessor=None,
        tokenizer=None,
        lowercase=False,
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=1,
        max_features=config.max_features,
        sublinear_tf=True,
    )
    if config.feature_kind == "word":
        features = word
    elif config.feature_kind == "char":
        features = char
    elif config.feature_kind == "word_char":
        features = FeatureUnion([("word", word), ("char", char)])
    else:
        raise ValueError(config.feature_kind)

    if config.classifier_kind == "logreg":
        clf = LogisticRegression(
            C=config.c_value,
            max_iter=2000,
            solver="liblinear",
            class_weight=config.class_weight,
            random_state=RANDOM_SEED,
        )
    elif config.classifier_kind == "linearsvc":
        clf = LinearSVC(C=config.c_value, class_weight=config.class_weight, random_state=RANDOM_SEED, dual="auto")
    elif config.classifier_kind == "calibrated_linearsvc":
        base = LinearSVC(C=config.c_value, class_weight=config.class_weight, random_state=RANDOM_SEED, dual="auto")
        clf = CalibratedClassifierCV(base, method="sigmoid", cv=3)
    else:
        raise ValueError(config.classifier_kind)
    return Pipeline([("features", features), ("clf", clf)])


def softmax(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    if scores.ndim == 1:
        scores = scores.reshape(-1, 1)
    scores = scores - np.max(scores, axis=1, keepdims=True)
    exp = np.exp(scores)
    return exp / np.clip(exp.sum(axis=1, keepdims=True), 1e-12, None)


def predict_proba_aligned(model: Pipeline, X: pd.Series, label_encoder: LabelEncoder) -> np.ndarray:
    clf = model.named_steps["clf"]
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        classes = getattr(clf, "classes_", label_encoder.classes_)
    else:
        scores = model.decision_function(X)
        probs = softmax(scores)
        classes = getattr(clf, "classes_", label_encoder.classes_)
    out = np.zeros((len(X), len(label_encoder.classes_)), dtype=float)
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


def multiclass_ece(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> float:
    confidence = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y_true).astype(float)
    ece = 0.0
    for low in np.linspace(0, 1, n_bins, endpoint=False):
        high = low + 1.0 / n_bins
        mask = (confidence >= low) & (confidence < high if high < 1 else confidence <= high)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidence[mask].mean())
    return float(ece)


def multiclass_brier(y_true: np.ndarray, probs: np.ndarray, n_classes: int) -> float:
    total = 0.0
    for class_idx in range(n_classes):
        total += brier_score_loss((y_true == class_idx).astype(int), probs[:, class_idx])
    return float(total / n_classes)


def metric_dict(y_true: np.ndarray, y_pred: np.ndarray, probs: np.ndarray | None = None, prefix: str = "") -> dict[str, float]:
    out = {
        f"{prefix}macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        f"{prefix}weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        f"{prefix}accuracy": float(accuracy_score(y_true, y_pred)),
        f"{prefix}balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        f"{prefix}mcc": float(matthews_corrcoef(y_true, y_pred)),
    }
    if probs is not None:
        out[f"{prefix}ece"] = multiclass_ece(y_true, probs)
        out[f"{prefix}brier_score"] = multiclass_brier(y_true, probs, probs.shape[1])
    return out


def evaluate_candidate(config: CandidateConfig, train: pd.DataFrame, label_encoder: LabelEncoder) -> tuple[pd.DataFrame, pd.DataFrame]:
    text_col = "text_social_normalized" if config.preprocessing == "social_normalized" else "text_minimal_raw"
    X = train[text_col].reset_index(drop=True)
    y = label_encoder.transform(train["label"])
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=RANDOM_SEED)
    fold_rows = []
    oof_rows = []
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y), start=1):
        model = make_pipeline(config)
        model.fit(X.iloc[train_idx], y[train_idx])
        pred = model.predict(X.iloc[val_idx])
        probs = predict_proba_aligned(model, X.iloc[val_idx], label_encoder)
        fold_metric = metric_dict(y[val_idx], pred, probs)
        precision, recall, f1, support = precision_recall_fscore_support(
            y[val_idx], pred, labels=np.arange(len(label_encoder.classes_)), zero_division=0
        )
        fold_metric.update(
            {
                "model_id": config.model_id,
                "fold": fold_idx,
                "preprocessing": config.preprocessing,
                "feature_kind": config.feature_kind,
                "classifier_kind": config.classifier_kind,
                "class_weight": config.class_weight or "None",
                "C": config.c_value,
                "coverage": 1.0,
                "abstention_rate": 0.0,
                "class_collapse": bool(len(set(pred)) < len(label_encoder.classes_)),
            }
        )
        for class_idx, label in enumerate(label_encoder.classes_):
            fold_metric[f"precision_{label}"] = float(precision[class_idx])
            fold_metric[f"recall_{label}"] = float(recall[class_idx])
            fold_metric[f"f1_{label}"] = float(f1[class_idx])
            fold_metric[f"support_{label}"] = int(support[class_idx])
        fold_rows.append(fold_metric)
        for local_i, original_idx in enumerate(val_idx):
            oof_rows.append(
                {
                    "model_id": config.model_id,
                    "comment_id": train.loc[original_idx, "comment_id"],
                    "repeat_fold": fold_idx,
                    "true_label": train.loc[original_idx, "label"],
                    "predicted_label": label_encoder.inverse_transform([int(pred[local_i])])[0],
                    "probability_negative": float(probs[local_i, label_encoder.transform(["Negative"])[0]]),
                    "probability_neutral": float(probs[local_i, label_encoder.transform(["Neutral"])[0]]),
                    "probability_positive": float(probs[local_i, label_encoder.transform(["Positive"])[0]]),
                    "prediction_confidence": float(probs[local_i].max()),
                }
            )
    return pd.DataFrame(fold_rows), pd.DataFrame(oof_rows)


def aggregate_oof(oof: pd.DataFrame, label_encoder: LabelEncoder) -> pd.DataFrame:
    prob_cols = ["probability_negative", "probability_neutral", "probability_positive"]
    agg = (
        oof.groupby(["model_id", "comment_id", "true_label"], as_index=False)[prob_cols]
        .mean()
        .reset_index(drop=True)
    )
    probs = agg[prob_cols].to_numpy()
    labels = np.array(["Negative", "Neutral", "Positive"])
    agg["predicted_label"] = labels[probs.argmax(axis=1)]
    agg["prediction_confidence"] = probs.max(axis=1)
    return agg


def summarize_candidate_metrics(fold_metrics: pd.DataFrame, oof_agg: pd.DataFrame, label_encoder: LabelEncoder) -> pd.DataFrame:
    rows = []
    for model_id, group in fold_metrics.groupby("model_id"):
        row = {"model_id": model_id}
        for metric in ["macro_f1", "weighted_f1", "accuracy", "balanced_accuracy", "mcc", "ece", "brier_score"]:
            row[f"mean_{metric}"] = float(group[metric].mean())
            row[f"std_{metric}"] = float(group[metric].std(ddof=0))
        row["mean_coverage"] = 1.0
        row["mean_abstention_rate"] = 0.0
        row["class_collapse_folds"] = int(group["class_collapse"].sum())
        for label in label_encoder.classes_:
            for metric in ["precision", "recall", "f1"]:
                row[f"mean_{metric}_{label}"] = float(group[f"{metric}_{label}"].mean())
        rows.append(row)

    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    summary = summary.sort_values(["mean_macro_f1", "mean_balanced_accuracy", "mean_mcc"], ascending=False).reset_index(drop=True)
    summary["rank"] = np.arange(1, len(summary) + 1)
    return summary


def baseline_v1_metrics(train: pd.DataFrame, label_encoder: LabelEncoder) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not COMMENT_SENTIMENT_PATH.exists():
        return pd.DataFrame(), pd.DataFrame()
    pred = read_csv(COMMENT_SENTIMENT_PATH)[["comment_id", "sentiment_label_final", "sentiment_label", "probability_negative", "probability_neutral", "probability_positive"]].copy()
    pred["baseline_label"] = pred["sentiment_label_final"].where(pred["sentiment_label_final"].isin(LABELS), pred["sentiment_label"])
    pred["baseline_label"] = pred["baseline_label"].where(pred["baseline_label"].isin(LABELS), "Neutral")
    merged = train[["comment_id", "label"]].merge(pred, on="comment_id", how="left")
    merged["baseline_label"] = merged["baseline_label"].fillna("Neutral")
    for col in ["probability_negative", "probability_neutral", "probability_positive"]:
        merged[col] = pd.to_numeric(merged[col], errors="coerce")
    missing_probs = merged[["probability_negative", "probability_neutral", "probability_positive"]].isna().any(axis=1)
    merged.loc[missing_probs, ["probability_negative", "probability_neutral", "probability_positive"]] = [1 / 3, 1 / 3, 1 / 3]
    probs = merged[["probability_negative", "probability_neutral", "probability_positive"]].astype(float).to_numpy()
    probs = probs / np.clip(probs.sum(axis=1, keepdims=True), 1e-12, None)
    y_true = label_encoder.transform(merged["label"])
    y_pred = label_encoder.transform(merged["baseline_label"])
    metrics = metric_dict(y_true, y_pred, probs)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=np.arange(len(label_encoder.classes_)), zero_division=0
    )
    row = {"model_id": "baseline_v1_existing_pipeline", **{f"mean_{k}": v for k, v in metrics.items()}}
    for idx, label in enumerate(label_encoder.classes_):
        row[f"mean_precision_{label}"] = float(precision[idx])
        row[f"mean_recall_{label}"] = float(recall[idx])
        row[f"mean_f1_{label}"] = float(f1[idx])
        row[f"support_{label}"] = int(support[idx])
    row["mean_coverage"] = 1.0
    row["mean_abstention_rate"] = 0.0
    row["class_collapse_folds"] = int(len(set(y_pred)) < len(label_encoder.classes_))
    oof = merged[["comment_id", "label", "baseline_label"]].copy()
    oof["model_id"] = "baseline_v1_existing_pipeline"
    oof["true_label"] = oof["label"]
    oof["predicted_label"] = oof["baseline_label"]
    oof[["probability_negative", "probability_neutral", "probability_positive"]] = probs
    oof["prediction_confidence"] = probs.max(axis=1)
    return pd.DataFrame([row]), oof.drop(columns=["label", "baseline_label"])


def threshold_selection(oof_selected: pd.DataFrame, label_encoder: LabelEncoder) -> tuple[pd.DataFrame, float]:
    y_true = label_encoder.transform(oof_selected["true_label"])
    probs = oof_selected[["probability_negative", "probability_neutral", "probability_positive"]].to_numpy()
    pred_all = probs.argmax(axis=1)
    rows = []
    for threshold in np.round(np.arange(0.0, 0.951, 0.01), 2):
        covered = probs.max(axis=1) >= threshold
        if covered.sum() == 0:
            continue
        y_c = y_true[covered]
        p_c = pred_all[covered]
        recalls = precision_recall_fscore_support(y_c, p_c, labels=np.arange(len(label_encoder.classes_)), zero_division=0)[1]
        row = {
            "threshold": float(threshold),
            "coverage": float(covered.mean()),
            "abstention_rate": float(1 - covered.mean()),
            "macro_f1": float(f1_score(y_c, p_c, average="macro", zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y_c, p_c)),
            "mcc": float(matthews_corrcoef(y_c, p_c)),
            "minimum_per_class_recall": float(recalls.min()),
            "objective": float(
                f1_score(y_c, p_c, average="macro", zero_division=0)
                + 0.10 * recalls.min()
                + 0.05 * balanced_accuracy_score(y_c, p_c)
                - 0.03 * (1 - covered.mean())
            ),
        }
        rows.append(row)
    table = pd.DataFrame(rows)
    eligible = table.loc[table["coverage"].ge(0.80)].copy()
    if eligible.empty:
        eligible = table.copy()
    selected = eligible.sort_values(["objective", "macro_f1", "minimum_per_class_recall", "coverage"], ascending=False).iloc[0]
    table["selected"] = table["threshold"].eq(selected["threshold"])
    return table, float(selected["threshold"])


def selected_metrics_with_threshold(oof_selected: pd.DataFrame, label_encoder: LabelEncoder, threshold: float) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    probs = oof_selected[["probability_negative", "probability_neutral", "probability_positive"]].to_numpy()
    confidence = probs.max(axis=1)
    covered = confidence >= threshold
    labels = np.array(["Negative", "Neutral", "Positive"])
    pred_label = labels[probs.argmax(axis=1)]
    out = oof_selected.copy()
    out["selected_threshold"] = threshold
    out["is_covered"] = covered
    out["abstained_label"] = np.where(covered, pred_label, "Abstain")
    out["correct_if_covered"] = np.where(covered, out["true_label"].eq(pred_label), "")

    y_true = label_encoder.transform(out.loc[covered, "true_label"])
    y_pred = label_encoder.transform(out.loc[covered, "predicted_label"])
    per = precision_recall_fscore_support(y_true, y_pred, labels=np.arange(len(label_encoder.classes_)), zero_division=0)
    per_class = pd.DataFrame(
        {
            "label": label_encoder.classes_,
            "precision": per[0],
            "recall": per[1],
            "f1": per[2],
            "support": per[3],
        }
    )
    conf = pd.DataFrame(
        confusion_matrix(y_true, y_pred, labels=np.arange(len(label_encoder.classes_))),
        index=label_encoder.classes_,
        columns=label_encoder.classes_,
    ).reset_index().rename(columns={"index": "true_label"})
    return out, per_class, conf


def bootstrap_macro_f1(oof: pd.DataFrame, threshold: float, n_bootstrap: int = 1000) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    labels = np.array(["Negative", "Neutral", "Positive"])
    probs = oof[["probability_negative", "probability_neutral", "probability_positive"]].to_numpy()
    confidence = probs.max(axis=1)
    pred = labels[probs.argmax(axis=1)]
    true = oof["true_label"].to_numpy()
    covered_idx = np.where(confidence >= threshold)[0]
    if len(covered_idx) == 0:
        values = np.array([np.nan])
    else:
        values = []
        for _ in range(n_bootstrap):
            idx = rng.choice(covered_idx, size=len(covered_idx), replace=True)
            values.append(f1_score(true[idx], pred[idx], labels=LABELS, average="macro", zero_division=0))
        values = np.asarray(values)
    return pd.DataFrame(
        [
            {
                "metric": "macro_f1",
                "threshold": threshold,
                "n_bootstrap": n_bootstrap,
                "mean": float(np.nanmean(values)),
                "ci_lower_95": float(np.nanpercentile(values, 2.5)),
                "ci_upper_95": float(np.nanpercentile(values, 97.5)),
            }
        ]
    )


def group_diagnostics(oof: pd.DataFrame, train: pd.DataFrame, threshold: float) -> pd.DataFrame:
    merged = oof.merge(
        train[
            [
                "comment_id",
                "is_hcc",
                "brand_or_video_context",
                "question",
                "sentiment_target",
                "complaint_scope",
            ]
        ],
        on="comment_id",
        how="left",
    )
    labels = np.array(["Negative", "Neutral", "Positive"])
    probs = merged[["probability_negative", "probability_neutral", "probability_positive"]].to_numpy()
    merged["predicted_label_thresholded"] = np.where(probs.max(axis=1) >= threshold, labels[probs.argmax(axis=1)], "Abstain")
    rows = []
    group_defs = {
        "hcc_segment": merged["is_hcc"].map(lambda x: "HCC" if bool(x) else "Non-HCC"),
        "brand_or_video_context": merged["brand_or_video_context"].replace("", "Unknown"),
        "question_segment": merged["question"].map(lambda x: "Question" if bool(x) else "Non-question"),
        "sentiment_target": merged["sentiment_target"].replace("", "Not available"),
        "complaint_scope": merged["complaint_scope"].replace("", "Not available"),
    }
    for group_name, series in group_defs.items():
        for value, group in merged.groupby(series, dropna=False):
            covered = group["predicted_label_thresholded"].ne("Abstain")
            if covered.sum() == 0:
                macro = np.nan
                acc = np.nan
            else:
                macro = f1_score(group.loc[covered, "true_label"], group.loc[covered, "predicted_label_thresholded"], labels=LABELS, average="macro", zero_division=0)
                acc = accuracy_score(group.loc[covered, "true_label"], group.loc[covered, "predicted_label_thresholded"])
            rows.append(
                {
                    "group_type": group_name,
                    "group_value": value,
                    "n_rows": int(len(group)),
                    "coverage": float(covered.mean()),
                    "macro_f1_covered": float(macro) if not pd.isna(macro) else "",
                    "accuracy_covered": float(acc) if not pd.isna(acc) else "",
                    "abstention_rate": float(1 - covered.mean()),
                }
            )
    return pd.DataFrame(rows)


def grouped_cv_selected(
    config: CandidateConfig,
    train: pd.DataFrame,
    label_encoder: LabelEncoder,
    ensemble_components: list[CandidateConfig] | None = None,
) -> pd.DataFrame:
    if train["video_id"].nunique() < 5:
        return pd.DataFrame([{"status": "NOT_AVAILABLE", "notes": "Not enough video_id groups."}])
    text_col = "text_social_normalized" if config.preprocessing == "social_normalized" else "text_minimal_raw"
    X = train[text_col].reset_index(drop=True)
    y = label_encoder.transform(train["label"])
    groups = train["video_id"].fillna("unknown").astype(str).to_numpy()
    n_splits = min(5, int(pd.Series(y).value_counts().min()), train["video_id"].nunique())
    if n_splits < 2:
        return pd.DataFrame([{"status": "NOT_AVAILABLE", "notes": "Insufficient class/group distribution."}])
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_SEED)
    rows = []
    for fold, (tr, va) in enumerate(cv.split(X, y, groups=groups), start=1):
        if ensemble_components:
            probs_parts = []
            for component in ensemble_components:
                component_col = "text_social_normalized" if component.preprocessing == "social_normalized" else "text_minimal_raw"
                component_model = make_pipeline(component)
                component_model.fit(train.iloc[tr][component_col], y[tr])
                probs_parts.append(predict_proba_aligned(component_model, train.iloc[va][component_col], label_encoder))
            probs = np.mean(probs_parts, axis=0)
            pred = probs.argmax(axis=1)
        else:
            model = make_pipeline(config)
            model.fit(X.iloc[tr], y[tr])
            pred = model.predict(X.iloc[va])
            probs = predict_proba_aligned(model, X.iloc[va], label_encoder)
        row = metric_dict(y[va], pred, probs)
        row.update(
            {
                "status": "AVAILABLE",
                "fold": fold,
                "n_train": int(len(tr)),
                "n_validation": int(len(va)),
                "n_train_groups": int(len(set(groups[tr]))),
                "n_validation_groups": int(len(set(groups[va]))),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def class_distribution_per_fold(train: pd.DataFrame) -> pd.DataFrame:
    label_encoder = LabelEncoder().fit(LABELS)
    y = label_encoder.transform(train["label"])
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=RANDOM_SEED)
    rows = []
    for fold, (tr, va) in enumerate(cv.split(train, y), start=1):
        for split_name, idx in [("train", tr), ("validation", va)]:
            counts = pd.Series(train.iloc[idx]["label"]).value_counts().to_dict()
            row = {"fold": fold, "split": split_name, "n_rows": int(len(idx))}
            for label in LABELS:
                row[f"n_{label}"] = int(counts.get(label, 0))
            rows.append(row)
    return pd.DataFrame(rows)


def fit_frozen_model(
    config: CandidateConfig,
    train: pd.DataFrame,
    label_encoder: LabelEncoder,
    selected_model_id: str,
    ensemble_components: list[CandidateConfig] | None = None,
) -> dict[str, object]:
    text_col = "text_social_normalized" if config.preprocessing == "social_normalized" else "text_minimal_raw"
    if ensemble_components:
        fitted_components = []
        y = label_encoder.transform(train["label"])
        for component in ensemble_components:
            component_col = "text_social_normalized" if component.preprocessing == "social_normalized" else "text_minimal_raw"
            model = make_pipeline(component)
            model.fit(train[component_col], y)
            fitted_components.append(
                {
                    "candidate_id": component.model_id,
                    "pipeline": model,
                    "preprocessing": component.preprocessing,
                    "text_column": component_col,
                    "feature_kind": component.feature_kind,
                    "classifier_kind": component.classifier_kind,
                    "class_weight": component.class_weight,
                    "C": component.c_value,
                }
            )
        model_object: object = fitted_components
    else:
        model = make_pipeline(config)
        model.fit(train[text_col], label_encoder.transform(train["label"]))
        model_object = model
    return {
        "model_name": "model_C_human_supervised",
        "base_candidate_id": selected_model_id,
        "pipeline": model_object,
        "is_ensemble": bool(ensemble_components),
        "ensemble_components": [component.model_id for component in ensemble_components or []],
        "label_encoder": label_encoder,
        "preprocessing": config.preprocessing,
        "text_column": text_col,
        "labels": list(label_encoder.classes_),
        "random_seed": RANDOM_SEED,
        "status": STATUS,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    source_hashes_before = {
        "dataset": sha256_file(DATASET_PATH),
        "comment_sentiment": sha256_file(COMMENT_SENTIMENT_PATH),
        "v1_validated": sha256_file(V1_VALIDATED_PATH),
        "v2_validated": sha256_file(V2_VALIDATED_PATH),
        "v2_adjudication_final": sha256_file(V2_ADJUDICATION_FINAL_PATH),
        "v2_locked_manifest": sha256_file(V2_LOCKED_MANIFEST_PATH),
    }

    provenance, train, meta = build_provenance_and_training_pool()
    provenance.to_csv(OUT_PROVENANCE, index=False)
    pd.DataFrame({"comment_id": sorted(meta["locked_observational_ids"])}).to_csv(OUT_EXCLUDED_LOCKED_IDS, index=False)

    if len(meta["locked_observational_ids"]) != 292:
        raise AssertionError(f"Expected 292 locked-test observational IDs, found {len(meta['locked_observational_ids'])}.")
    if len(train) < 100:
        raise AssertionError("Development training pool is unexpectedly small.")
    if set(train["comment_id"]) & set(meta["locked_observational_ids"]):
        raise AssertionError("Locked-test V2 observational IDs leaked into training pool.")
    if set(train["comment_id"]) & set(meta["synthetic_ids"]):
        raise AssertionError("Synthetic/injected IDs leaked into training pool.")
    if train["comment_id"].duplicated().any():
        raise AssertionError("Duplicate comment_id detected in training pool.")
    if not set(train["label"]).issubset(set(LABELS)):
        raise AssertionError("Training labels include non-primary sentiment labels.")

    label_encoder = LabelEncoder().fit(LABELS)
    candidates = [
        CandidateConfig("tfidf_word_logreg_social_C1_balanced", "social_normalized", "word", "logreg", "balanced", 1.0),
        CandidateConfig("tfidf_word_logreg_raw_C1_balanced", "minimal_raw", "word", "logreg", "balanced", 1.0),
        CandidateConfig("tfidf_word_logreg_social_C2_balanced", "social_normalized", "word", "logreg", "balanced", 2.0),
        CandidateConfig("tfidf_word_linearsvc_social_C1_balanced", "social_normalized", "word", "linearsvc", "balanced", 1.0),
        CandidateConfig("tfidf_char_logreg_social_C1_balanced", "social_normalized", "char", "logreg", "balanced", 1.0),
        CandidateConfig("tfidf_char_linearsvc_social_C1_balanced", "social_normalized", "char", "linearsvc", "balanced", 1.0),
        CandidateConfig("tfidf_word_char_logreg_social_C1_balanced", "social_normalized", "word_char", "logreg", "balanced", 1.0),
        CandidateConfig("calibrated_linearsvc_word_char_social_C1_balanced", "social_normalized", "word_char", "calibrated_linearsvc", "balanced", 1.0),
    ]

    all_fold_metrics = []
    all_oof = []
    for config in candidates:
        print(f"Evaluating {config.model_id} ...")
        fold_metrics, oof = evaluate_candidate(config, train, label_encoder)
        all_fold_metrics.append(fold_metrics)
        all_oof.append(oof)
    cv_metrics = pd.concat(all_fold_metrics, ignore_index=True)
    oof_raw = pd.concat(all_oof, ignore_index=True)
    oof_agg = aggregate_oof(oof_raw, label_encoder)

    summary = summarize_candidate_metrics(cv_metrics, oof_agg, label_encoder)
    summary["selection_eligible"] = True
    summary["selection_notes"] = "Human-supervised V2 development candidate."
    baseline_summary, baseline_oof = baseline_v1_metrics(train, label_encoder)
    if not baseline_summary.empty:
        baseline_summary["rank"] = ""
        baseline_summary["selection_eligible"] = False
        baseline_summary["selection_notes"] = "Diagnostic legacy V1 baseline; not frozen as the V2 development model."
        summary = pd.concat([summary, baseline_summary], ignore_index=True, sort=False)

    # Optional ensemble: average the top two human-supervised OOF probabilities,
    # include it only if it improves development macro-F1.
    human_summary = summary.loc[~summary["model_id"].eq("baseline_v1_existing_pipeline")].copy()
    top_model_ids = human_summary.sort_values("mean_macro_f1", ascending=False)["model_id"].head(2).tolist()
    if len(top_model_ids) == 2:
        top_oof = [
            oof_agg.loc[oof_agg["model_id"].eq(mid)].set_index("comment_id")
            for mid in top_model_ids
        ]
        common_ids = sorted(set(top_oof[0].index) & set(top_oof[1].index))
        ens = top_oof[0].loc[common_ids].copy()
        for col in ["probability_negative", "probability_neutral", "probability_positive"]:
            ens[col] = (top_oof[0].loc[common_ids, col].astype(float) + top_oof[1].loc[common_ids, col].astype(float)) / 2
        labels_arr = np.array(["Negative", "Neutral", "Positive"])
        ens["predicted_label"] = labels_arr[ens[["probability_negative", "probability_neutral", "probability_positive"]].to_numpy().argmax(axis=1)]
        ens["prediction_confidence"] = ens[["probability_negative", "probability_neutral", "probability_positive"]].max(axis=1)
        y_true_ens = label_encoder.transform(ens["true_label"])
        y_pred_ens = label_encoder.transform(ens["predicted_label"])
        ens_metrics = metric_dict(y_true_ens, y_pred_ens, ens[["probability_negative", "probability_neutral", "probability_positive"]].to_numpy())
        best_macro = float(human_summary["mean_macro_f1"].max())
        if ens_metrics["macro_f1"] > best_macro + 0.002:
            ens["model_id"] = "ensemble_top2_human_supervised_development_only"
            oof_agg = pd.concat([oof_agg, ens.reset_index()], ignore_index=True, sort=False)
            ens_row = {"model_id": "ensemble_top2_human_supervised_development_only"}
            ens_row.update({f"mean_{k}": v for k, v in ens_metrics.items()})
            ens_row["ensemble_components"] = ";".join(top_model_ids)
            ens_row["selection_eligible"] = True
            ens_row["selection_notes"] = "Eligible because averaged top-2 human-supervised OOF probabilities improved development macro-F1."
            ens_row["mean_coverage"] = 1.0
            ens_row["mean_abstention_rate"] = 0.0
            summary = pd.concat([summary, pd.DataFrame([ens_row])], ignore_index=True, sort=False)

    human_candidates = summary.loc[summary["selection_eligible"].fillna(False).astype(bool)].copy()
    selected_model_id = human_candidates.sort_values(["mean_macro_f1", "mean_balanced_accuracy", "mean_mcc"], ascending=False).iloc[0]["model_id"]
    selected_config = next((c for c in candidates if c.model_id == selected_model_id), None)
    selected_ensemble_components: list[CandidateConfig] | None = None
    if selected_config is None:
        if selected_model_id != "ensemble_top2_human_supervised_development_only":
            raise AssertionError(f"Selected model is not supported for freezing: {selected_model_id}")
        selected_ensemble_components = [next(c for c in candidates if c.model_id == mid) for mid in top_model_ids]
        selected_config = selected_ensemble_components[0]
    summary["selected_for_freeze"] = summary["model_id"].eq(selected_model_id)
    summary["model_family_name"] = np.where(summary["selected_for_freeze"], "model_C_human_supervised", "")
    summary["selection_status"] = STATUS
    summary.to_csv(OUT_SELECTION, index=False)
    cv_metrics.to_csv(OUT_CV, index=False)
    class_distribution_per_fold(train).to_csv(OUT_FOLD_DISTRIBUTION, index=False)

    selected_oof = oof_agg.loc[oof_agg["model_id"].eq(selected_model_id)].copy().reset_index(drop=True)
    threshold_table, selected_threshold = threshold_selection(selected_oof, label_encoder)
    threshold_table.to_csv(OUT_THRESHOLD, index=False)
    oof_thresholded, per_class, confusion = selected_metrics_with_threshold(selected_oof, label_encoder, selected_threshold)
    oof_thresholded.to_csv(OUT_OOF, index=False)
    per_class.to_csv(OUT_PER_CLASS, index=False)
    confusion.to_csv(OUT_CONFUSION, index=False)
    bootstrap_macro_f1(selected_oof, selected_threshold).to_csv(OUT_BOOTSTRAP, index=False)
    group_diagnostics(selected_oof, train, selected_threshold).to_csv(OUT_GROUP_DIAGNOSTICS, index=False)
    grouped_cv_selected(selected_config, train, label_encoder, selected_ensemble_components).to_csv(OUT_GROUPED_CV, index=False)

    covered = oof_thresholded["is_covered"].astype(bool)
    errors = oof_thresholded.loc[covered & ~oof_thresholded["true_label"].eq(oof_thresholded["predicted_label"])].merge(
        train[
            [
                "comment_id",
                "comment_text_original",
                "video_id",
                "brand_or_video_context",
                "is_hcc",
                "question",
                "emoji",
                "negation",
                "slang",
                "code_mixing",
                "very_short",
                "mixed_sentiment",
                "sentiment_target",
                "complaint_scope",
            ]
        ],
        on="comment_id",
        how="left",
    )
    errors.to_csv(OUT_ERROR, index=False)

    probs = selected_oof[["probability_negative", "probability_neutral", "probability_positive"]].to_numpy()
    y_true = label_encoder.transform(selected_oof["true_label"])
    y_pred = label_encoder.transform(selected_oof["predicted_label"])
    pd.DataFrame(
        [
            {
                "model_id": selected_model_id,
                "threshold": selected_threshold,
                "ece": multiclass_ece(y_true, probs),
                "brier_score": multiclass_brier(y_true, probs, len(LABELS)),
                "coverage_at_threshold": float(covered.mean()),
                "abstention_rate_at_threshold": float(1 - covered.mean()),
            }
        ]
    ).to_csv(OUT_CALIBRATION, index=False)

    artifact = fit_frozen_model(selected_config, train, label_encoder, selected_model_id, selected_ensemble_components)
    artifact["selected_threshold"] = selected_threshold
    artifact["development_data_hash"] = dataframe_hash(train, ["comment_id", "label", "text_social_normalized", "text_minimal_raw"])
    artifact["excluded_locked_test_v2_hash"] = hashlib.sha256("\n".join(sorted(meta["locked_observational_ids"])).encode("utf-8")).hexdigest()
    joblib.dump(artifact, OUT_MODEL)
    model_hash = sha256_file(OUT_MODEL)
    OUT_MODEL_HASH.write_text(model_hash + "\n", encoding="utf-8")

    model_config = {
        "status": STATUS,
        "model_name": "model_C_human_supervised",
        "selected_candidate_id": selected_model_id,
        "is_ensemble": bool(selected_ensemble_components),
        "ensemble_components": [component.model_id for component in selected_ensemble_components or []],
        "threshold": selected_threshold,
        "preprocessing": selected_config.preprocessing,
        "feature_kind": selected_config.feature_kind,
        "classifier_kind": selected_config.classifier_kind,
        "class_weight": selected_config.class_weight,
        "C": selected_config.c_value,
        "labels": LABELS,
        "random_seed": RANDOM_SEED,
        "development_data_hash": artifact["development_data_hash"],
        "excluded_locked_test_v2_hash": artifact["excluded_locked_test_v2_hash"],
        "model_artifact_sha256": model_hash,
    }
    OUT_MODEL_CONFIG.write_text(json.dumps(model_config, indent=2), encoding="utf-8")

    source_hashes_after = {
        "dataset": sha256_file(DATASET_PATH),
        "comment_sentiment": sha256_file(COMMENT_SENTIMENT_PATH),
        "v1_validated": sha256_file(V1_VALIDATED_PATH),
        "v2_validated": sha256_file(V2_VALIDATED_PATH),
        "v2_adjudication_final": sha256_file(V2_ADJUDICATION_FINAL_PATH),
        "v2_locked_manifest": sha256_file(V2_LOCKED_MANIFEST_PATH),
    }

    pool_summary = pd.DataFrame(
        [
            {"metric": "training_pool_rows", "value": len(train), "notes": "Primary sentiment labels only: Positive, Neutral, Negative."},
            {"metric": "training_pool_unique_comment_id", "value": train["comment_id"].nunique(), "notes": ""},
            {"metric": "v1_rows_available", "value": int(provenance["source_version"].eq("V1").sum()), "notes": ""},
            {"metric": "v2_rows_available", "value": int(provenance["source_version"].eq("V2").sum()), "notes": ""},
            {"metric": "synthetic_ids_excluded", "value": int(provenance["exclusion_reason"].eq("synthetic_or_injected_id").sum()), "notes": ""},
            {"metric": "locked_test_v2_observational_ids_excluded", "value": len(meta["locked_observational_ids"]), "notes": ""},
            {"metric": "locked_test_v2_manifest_ids_excluded", "value": len(meta["locked_manifest_ids"]), "notes": "Includes 8 synthetic/injected locked-test IDs."},
            {"metric": "final_locked_test_status", "value": READINESS_STATUS, "notes": ""},
            {"metric": "selected_model", "value": selected_model_id, "notes": "Frozen as model_C_human_supervised."},
            {"metric": "selected_model_is_ensemble", "value": bool(selected_ensemble_components), "notes": ""},
            {"metric": "selected_ensemble_components", "value": ";".join(component.model_id for component in selected_ensemble_components or []), "notes": ""},
            {"metric": "selected_threshold", "value": selected_threshold, "notes": "Selected on development OOF predictions only."},
        ]
    )
    for label, count in train["label"].value_counts().sort_index().items():
        pool_summary = pd.concat(
            [pool_summary, pd.DataFrame([{"metric": f"class_count_{label}", "value": int(count), "notes": ""}])],
            ignore_index=True,
        )
    pool_summary.to_csv(OUT_POOL_SUMMARY, index=False)

    readiness = pd.DataFrame(
        [
            {"metric": "final_locked_test_evaluation_status", "value": READINESS_STATUS, "passed": False, "notes": "Do not evaluate until 8 human-annotated replacements are complete."},
            {"metric": "locked_test_v2_observational_rows_available", "value": len(meta["locked_observational_ids"]), "passed": len(meta["locked_observational_ids"]) == 300, "notes": "Current observational locked test remains 292 rows."},
            {"metric": "locked_test_v2_manifest_rows", "value": len(meta["locked_manifest_ids"]), "passed": len(meta["locked_manifest_ids"]) == 300, "notes": "Manifest includes 8 synthetic/injected IDs."},
            {"metric": "locked_test_predictions_generated", "value": 0, "passed": True, "notes": "Script never scores locked-test V2 rows."},
            {"metric": "full_inference_generated", "value": 0, "passed": True, "notes": "comment_sentiment.csv is not modified."},
            {"metric": "source_hashes_unchanged", "value": source_hashes_before == source_hashes_after, "passed": source_hashes_before == source_hashes_after, "notes": "Input sentiment/RM1 files unchanged."},
        ]
    )
    readiness.to_csv(OUT_READINESS, index=False)

    pd.DataFrame(
        [
            {"package": "python", "version": sys.version.split()[0]},
            {"package": "platform", "version": platform.platform()},
            {"package": "pandas", "version": pd.__version__},
            {"package": "numpy", "version": np.__version__},
            {"package": "sklearn", "version": __import__("sklearn").__version__},
            {"package": "scipy", "version": __import__("scipy").__version__},
            {"package": "joblib", "version": joblib.__version__},
        ]
    ).to_csv(OUT_PACKAGE_VERSIONS, index=False)

    manifest = {
        "status": STATUS,
        "final_locked_test_evaluation_status": READINESS_STATUS,
        "model_name": "model_C_human_supervised",
        "selected_candidate_id": selected_model_id,
        "is_ensemble": bool(selected_ensemble_components),
        "ensemble_components": [component.model_id for component in selected_ensemble_components or []],
        "selected_threshold": selected_threshold,
        "training_pool_rows": int(len(train)),
        "training_pool_unique_comment_id": int(train["comment_id"].nunique()),
        "class_distribution": {k: int(v) for k, v in train["label"].value_counts().sort_index().items()},
        "source_distribution_training": {
            k: int(v) for k, v in train["source_version"].value_counts().sort_index().items()
        },
        "sample_role_distribution_training": {
            k: int(v) for k, v in train["sample_role_final"].value_counts().sort_index().items()
        },
        "synthetic_ids_excluded": int(provenance["exclusion_reason"].eq("synthetic_or_injected_id").sum()),
        "locked_test_v2_observational_ids_excluded": int(len(meta["locked_observational_ids"])),
        "locked_test_v2_manifest_ids_excluded": int(len(meta["locked_manifest_ids"])),
        "development_data_hash": artifact["development_data_hash"],
        "model_artifact_sha256": model_hash,
        "source_hashes_before": source_hashes_before,
        "source_hashes_after": source_hashes_after,
        "outputs": [
            str(path.relative_to(ROOT))
            for path in [
                OUT_MODEL,
                OUT_MODEL_CONFIG,
                OUT_MODEL_HASH,
                OUT_MANIFEST,
                OUT_SELECTION,
                OUT_CV,
                OUT_CONFUSION,
                OUT_OOF,
                OUT_ERROR,
                OUT_PROVENANCE,
                OUT_READINESS,
                OUT_PER_CLASS,
                OUT_THRESHOLD,
                OUT_GROUP_DIAGNOSTICS,
                OUT_GROUPED_CV,
                OUT_FOLD_DISTRIBUTION,
                OUT_CALIBRATION,
                OUT_BOOTSTRAP,
                OUT_POOL_SUMMARY,
                OUT_EXCLUDED_LOCKED_IDS,
                OUT_PACKAGE_VERSIONS,
            ]
        ],
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if not readiness["passed"].iloc[[2, 3, 4, 5]].all():
        raise AssertionError("A non-locked-test development gate failed.")
    if set(train["comment_id"]) & set(meta["locked_observational_ids"]):
        raise AssertionError("Locked-test leakage detected after model freeze.")
    if source_hashes_before["comment_sentiment"] != source_hashes_after["comment_sentiment"]:
        raise AssertionError("comment_sentiment.csv changed unexpectedly.")

    print("RM2 SENTIMENT V2 DEVELOPMENT MODEL FROZEN")
    print(f"- status: {STATUS}")
    print(f"- training rows: {len(train):,}")
    print(f"- unique comment_id: {train['comment_id'].nunique():,}")
    print(f"- class distribution: {train['label'].value_counts().sort_index().to_dict()}")
    print(f"- selected model: {selected_model_id}")
    print(f"- selected threshold: {selected_threshold:.2f}")
    print(f"- model hash: {model_hash}")
    print(f"- locked-test evaluation: {READINESS_STATUS}")
    print("- locked-test predictions generated: 0")
    print("- full inference generated: 0")


if __name__ == "__main__":
    main()
