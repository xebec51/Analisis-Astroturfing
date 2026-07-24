from __future__ import annotations

import html
import re
import unicodedata
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd


LABELS = ["Negative", "Neutral", "Positive"]
LABEL_TO_ID = {label: idx for idx, label in enumerate(LABELS)}
PROB_COLUMNS = [f"probability_{label.lower()}" for label in LABELS]


def norm_blank(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"nan", "none", "null", "<na>"}:
        return ""
    return text


def normalize_for_model(text: object) -> str:
    s = norm_blank(text)
    s = unicodedata.normalize("NFKC", s)
    s = html.unescape(s)
    s = "".join(ch for ch in s if unicodedata.category(ch)[0] != "C" or ch in "\t\n\r")
    s = re.sub(r"https?://\S+|www\.\S+", "HTTPURL", s, flags=re.IGNORECASE)
    s = re.sub(r"(?<!\w)@\w+", "@USER", s)
    return re.sub(r"\s+", " ", s).strip()


def no_text_flag(text: object) -> bool:
    s = norm_blank(text)
    return s.casefold() in {"", "nan", "none", "null", "<na>", "[deleted]", "deleted"}


def softmax(scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    arr = np.asarray(scores, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    arr = arr / max(float(temperature), 1e-9)
    arr = arr - np.max(arr, axis=1, keepdims=True)
    exp = np.exp(arr)
    return exp / np.clip(exp.sum(axis=1, keepdims=True), 1e-12, None)


def normalize_probabilities(probs: np.ndarray) -> np.ndarray:
    arr = np.asarray(probs, dtype=float)
    arr = np.clip(arr, 0.0, 1.0)
    denom = arr.sum(axis=1, keepdims=True)
    zero = denom.squeeze() <= 0
    if (~zero).any():
        arr[~zero] = arr[~zero] / denom[~zero]
    if zero.any():
        arr[zero] = 1.0 / arr.shape[1]
    return arr


def align_probabilities(probs: np.ndarray, classes: Iterable[object], labels: list[str] | None = None) -> np.ndarray:
    label_order = labels or LABELS
    out = np.zeros((len(probs), len(label_order)), dtype=float)
    classes_array = np.asarray(list(classes))
    if np.issubdtype(classes_array.dtype, np.number):
        for src_idx, encoded_label in enumerate(classes_array.astype(int)):
            if 0 <= encoded_label < len(label_order):
                out[:, encoded_label] = probs[:, src_idx]
    else:
        class_to_idx = {str(label): idx for idx, label in enumerate(classes_array)}
        for dst_idx, label in enumerate(label_order):
            if label in class_to_idx:
                out[:, dst_idx] = probs[:, class_to_idx[label]]
    return normalize_probabilities(out)


def predict_proba_aligned(model, texts: pd.Series, labels: list[str] | None = None, temperature: float = 1.0) -> np.ndarray:
    label_order = labels or LABELS
    clf = model.named_steps["clf"] if hasattr(model, "named_steps") and "clf" in model.named_steps else model
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(texts)
        classes = getattr(clf, "classes_", label_order)
    else:
        scores = model.decision_function(texts)
        probs = softmax(scores, temperature=temperature)
        classes = getattr(clf, "classes_", label_order)
    return align_probabilities(probs, classes, label_order)


def load_candidate_artifact(path: str | Path) -> dict[str, object]:
    return joblib.load(path)


def artifact_predict_proba(artifact: dict[str, object], texts: pd.Series) -> np.ndarray:
    normalized = texts.map(normalize_for_model)
    labels = [str(label) for label in artifact.get("labels", LABELS)]
    components = artifact.get("components", [])
    if not components and "pipeline" in artifact:
        components = [{"pipeline": artifact["pipeline"], "temperature": artifact.get("temperature", 1.0)}]
    if not components:
        raise ValueError("Candidate artifact contains no prediction components.")
    probs = []
    for component in components:
        probs.append(
            predict_proba_aligned(
                component["pipeline"],
                normalized,
                labels=labels,
                temperature=float(component.get("temperature", 1.0)),
            )
        )
    return normalize_probabilities(np.mean(probs, axis=0))


def apply_threshold_policy(
    probs: np.ndarray,
    *,
    positive_threshold: float,
    margin_positive_neutral: float,
    margin_positive_negative: float,
    abstention_threshold: float,
    no_text_mask: Iterable[bool] | None = None,
    labels: list[str] | None = None,
) -> pd.DataFrame:
    label_order = labels or LABELS
    arr = normalize_probabilities(probs)
    if arr.shape[1] != len(label_order):
        raise ValueError("Probability array width does not match label order.")
    no_text = np.array(list(no_text_mask), dtype=bool) if no_text_mask is not None else np.zeros(len(arr), dtype=bool)
    rows: list[dict[str, object]] = []
    neg_idx = label_order.index("Negative")
    neu_idx = label_order.index("Neutral")
    pos_idx = label_order.index("Positive")
    for i, row in enumerate(arr):
        p_neg = float(row[neg_idx])
        p_neu = float(row[neu_idx])
        p_pos = float(row[pos_idx])
        confidence = float(row.max())
        argmax_label = label_order[int(row.argmax())]
        if no_text[i]:
            final_label = "No Text"
            reason = "no_text"
        elif (
            p_pos >= positive_threshold
            and (p_pos - p_neu) >= margin_positive_neutral
            and (p_pos - p_neg) >= margin_positive_negative
        ):
            final_label = "Positive"
            reason = "positive_development_threshold_met"
        elif confidence < abstention_threshold:
            final_label = "Uncertain"
            reason = "max_confidence_below_abstention_threshold"
        else:
            non_positive_label = "Negative" if p_neg >= p_neu else "Neutral"
            non_positive_prob = max(p_neg, p_neu)
            if non_positive_prob >= abstention_threshold:
                final_label = non_positive_label
                reason = "non_positive_confidence_met"
            elif argmax_label != "Positive":
                final_label = argmax_label
                reason = "argmax_non_positive_used"
            else:
                final_label = "Uncertain"
                reason = "positive_threshold_not_met"
        rows.append(
            {
                "predicted_label": final_label,
                "argmax_label": argmax_label,
                "prediction_confidence": confidence,
                "abstention_reason": "" if final_label in label_order else reason,
                "policy_reason": reason,
                "probability_negative": p_neg,
                "probability_neutral": p_neu,
                "probability_positive": p_pos,
                "positive_threshold": float(positive_threshold),
                "margin_positive_neutral": float(margin_positive_neutral),
                "margin_positive_negative": float(margin_positive_negative),
                "abstention_threshold": float(abstention_threshold),
            }
        )
    return pd.DataFrame(rows)


def predict_labels_from_artifact(artifact: dict[str, object], texts: pd.Series) -> pd.DataFrame:
    probs = artifact_predict_proba(artifact, texts)
    policy = artifact["threshold_policy"]
    no_text_mask = texts.map(no_text_flag)
    return apply_threshold_policy(
        probs,
        positive_threshold=float(policy["positive_threshold"]),
        margin_positive_neutral=float(policy["margin_positive_neutral"]),
        margin_positive_negative=float(policy["margin_positive_negative"]),
        abstention_threshold=float(policy["abstention_threshold"]),
        no_text_mask=no_text_mask,
        labels=[str(label) for label in artifact.get("labels", LABELS)],
    )
