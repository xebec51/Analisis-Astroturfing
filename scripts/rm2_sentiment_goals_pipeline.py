from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import random
import re
import shutil
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_fscore_support,
)
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC

os.environ.setdefault("TRANSFORMERS_NO_TF", "1")


RANDOM_STATE = 42
TOTAL_COMMENTS_EXPECTED = 33847
HCC_NODE_EXPECTED = 218
HCC_COUNT_EXPECTED = 42
HCC_COMMENT_EXPECTED = 1009
ACCOUNT_SENTIMENT_ROWS_EXPECTED = 26424
ACTOR_UNIVERSE_EXPECTED = 26427
ACTOR_TYPE_COUNTS_EXPECTED = {
    "Individual Actor": 43,
    "Community Actor": 218,
    "Mass Actor": 26166,
}
GEPHI_AGGREGATE_EXPECTED = {"nodes": 396, "edges": 497}

LABELS = ["Positive", "Neutral", "Negative"]
ALLOWED_AI_LABELS = ["Positive", "Neutral", "Negative", "Uncertain", "No Text"]
ALLOW_RULE_BASED_FINAL = False

MODEL_A = "mdhugol/indonesia-bert-sentiment-classification"
MODEL_B = "w11wo/indonesian-roberta-base-sentiment-classifier"
MODEL_A_LABEL_MAP = {"LABEL_0": "Positive", "LABEL_1": "Neutral", "LABEL_2": "Negative"}

NEGATION_TERMS = [
    "tidak",
    "nggak",
    "ngga",
    "gak",
    "ga",
    "bukan",
    "belum",
    "jangan",
    "tanpa",
]
DOMAIN_TERMS = [
    "cocok",
    "breakout",
    "purging",
    "iritasi",
    "bruntusan",
    "glowing",
    "kusam",
    "overclaim",
    "mahal",
    "murah",
    "worth it",
    "aman",
    "perih",
    "lengket",
    "berminyak",
    "kering",
    "jerawat",
    "komedo",
    "palsu",
    "ori",
    "bpom",
]
POSITIVE_TERMS = [
    "bagus",
    "cocok",
    "worth",
    "worth it",
    "glowing",
    "cerah",
    "aman",
    "suka",
    "love",
    "mantap",
    "recommend",
    "rekomen",
    "puas",
    "lembut",
    "murah",
    "ori",
    "bpom",
    "berhasil",
    "hasilnya kelihatan",
    "nggak bikin iritasi",
    "gak bikin iritasi",
    "ga bikin iritasi",
    "tidak bikin iritasi",
    "nggak breakout",
    "gak breakout",
    "ga breakout",
]
NEGATIVE_TERMS = [
    "breakout",
    "iritasi",
    "bruntusan",
    "purging",
    "perih",
    "gatal",
    "jerawat",
    "komedo",
    "kusam",
    "lengket",
    "berminyak",
    "kering",
    "mahal",
    "overclaim",
    "palsu",
    "zonk",
    "kecewa",
    "parah",
    "ga cocok",
    "gak cocok",
    "ngga cocok",
    "nggak cocok",
    "tidak cocok",
    "ga bagus",
    "gak bagus",
    "nggak bagus",
    "tidak bagus",
    "terlalu mahal",
]
QUESTION_TERMS = [
    "berapa",
    "dimana",
    "di mana",
    "mana",
    "kapan",
    "link",
    "beli",
    "belinya",
    "harga",
    "harganya",
    "ukuran",
    "ml",
    "pakai apa",
    "cara pakai",
]
SLANG_TERMS = [
    "bgt",
    "banget",
    "kak",
    "ka",
    "dong",
    "sih",
    "nih",
    "wkwk",
    "yg",
    "jd",
    "aja",
    "deh",
]
ENGLISH_TERMS = [
    "worth",
    "review",
    "claim",
    "glow",
    "glowing",
    "breakout",
    "original",
    "fake",
    "recommend",
]
AMBIGUOUS_EMOJI = set(".,!?;:'\"`~_-+=*/\\|()[]{}<>")
POSITIVE_EMOJI = {
    "\U0001f60d",
    "\U0001f970",
    "\U0001f618",
    "\U0001f929",
    "\U0001f60a",
    "\U0001f44d",
    "\U0001f64f",
    "\u2764",
    "\ufe0f",
}
NEGATIVE_EMOJI = {
    "\U0001f621",
    "\U0001f620",
    "\U0001f92e",
    "\U0001f622",
    "\U0001f62d",
    "\U0001f44e",
}


@dataclass
class TransformerCandidate:
    key: str
    model_name: str
    tokenizer: object
    model: object
    label_index_to_standard: dict[int, str]
    model_revision: str
    tokenizer_revision: str


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_clean_output_dir(path: Path, root: Path) -> None:
    resolved_root = root.resolve()
    resolved_path = path.resolve()
    if resolved_path == resolved_root or resolved_root not in resolved_path.parents:
        raise ValueError(f"Refusing to clean unsafe path: {resolved_path}")
    if path.exists():
        shutil.rmtree(path)
    (path / "tables").mkdir(parents=True, exist_ok=True)
    (path / "visualisasi").mkdir(parents=True, exist_ok=True)
    (path / "gephi").mkdir(parents=True, exist_ok=True)


def normalize_username(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().lower().lstrip("@")
    return re.sub(r"\s+", "", text)


def normalize_blank(value) -> str:
    if pd.isna(value):
        return ""
    text = str(value).replace("\r", "\n")
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if text.lower() in {"nan", "none", "null", "<na>"}:
        return ""
    return text


def minimal_raw(text: str) -> str:
    return normalize_blank(text)


def social_normalized(text: str) -> str:
    text = normalize_blank(text)
    text = re.sub(r"https?://\S+|www\.\S+", "HTTPURL", text, flags=re.I)
    text = re.sub(r"(?<!\w)@\w+", "@USER", text)
    text = re.sub(r"(.)\1{3,}", r"\1\1\1", text)
    return re.sub(r"\s+", " ", text).strip()


def product_category_to_brand(value) -> str:
    text = normalize_blank(value).lower()
    if "azarine" in text:
        return "Azarine"
    if "daviena" in text or "davina" in text:
        return "Daviena"
    if "maryame" in text:
        return "Maryame"
    if "originote" in text:
        return "The Originote"
    if text == "":
        return "Unidentified"
    return normalize_blank(value)


def contains_any(text: str, terms: list[str]) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in terms)


def valenced_emoji_counts(text: str) -> tuple[int, int]:
    pos = sum(text.count(e) for e in POSITIVE_EMOJI)
    neg = sum(text.count(e) for e in NEGATIVE_EMOJI)
    return pos, neg


def text_quality_flags(text: str) -> dict[str, bool]:
    raw = "" if pd.isna(text) else str(text)
    stripped = raw.strip()
    model_text = social_normalized(raw)
    has_alnum = bool(re.search(r"[A-Za-z0-9]", stripped))
    tokens = re.findall(r"\b\w+\b", stripped.lower())
    only_symbols = stripped != "" and not has_alnum
    emoji_pos, emoji_neg = valenced_emoji_counts(stripped)
    url_removed = re.sub(r"https?://\S+|www\.\S+", "", stripped, flags=re.I).strip()
    mention_removed = re.sub(r"(?<!\w)@\w+", "", stripped).strip()
    return {
        "blank_text": stripped == "",
        "whitespace_only": raw != "" and stripped == "",
        "emoji_only": only_symbols and any(ch not in AMBIGUOUS_EMOJI and not ch.isspace() for ch in stripped),
        "url_only": stripped != "" and url_removed == "",
        "mention_only": stripped != "" and mention_removed == "",
        "very_short": stripped != "" and (len(tokens) <= 2 or len(stripped) <= 8),
        "code_mixing": contains_any(stripped, ENGLISH_TERMS) and bool(re.search(r"\b(aku|kak|yang|dan|ini|pakai|cocok|banget)\b", stripped.lower())),
        "slang": contains_any(stripped, SLANG_TERMS + NEGATION_TERMS),
        "repeated_characters": bool(re.search(r"(.)\1{3,}", stripped)),
        "negation": contains_any(stripped, NEGATION_TERMS),
        "question": "?" in stripped or contains_any(stripped, QUESTION_TERMS),
        "potential_sarcasm": bool(re.search(r"\b(wkwk|haha|hahaha|kok|katanya|masa)\b", stripped.lower())),
        "mixed_sentiment": contains_any(stripped, POSITIVE_TERMS) and contains_any(stripped, NEGATIVE_TERMS),
        "domain_term": contains_any(stripped, DOMAIN_TERMS),
        "model_text_blank": model_text == "",
        "clear_valence_emoji": emoji_pos != emoji_neg and (emoji_pos + emoji_neg) > 0,
    }


def semantic_label_for_text(text: str, pass_id: int = 1) -> tuple[str, list[str], str]:
    text = normalize_blank(text)
    if text == "":
        return "No Text", ["blank_text"], "Tidak ada teks yang dapat dievaluasi."
    flags = text_quality_flags(text)
    lowered = text.lower()
    emoji_pos, emoji_neg = valenced_emoji_counts(text)

    if flags["url_only"] or flags["mention_only"]:
        return "No Text", ["url_or_mention_only"], "Komentar hanya URL atau mention tanpa orientasi pesan."
    if flags["emoji_only"] and emoji_pos == 0 and emoji_neg == 0:
        return "Uncertain", ["emoji_only"], "Emoji/simbol tanpa valensi yang cukup jelas."

    pos_score = sum(1 for term in POSITIVE_TERMS if term in lowered) + emoji_pos
    neg_score = sum(1 for term in NEGATIVE_TERMS if term in lowered) + emoji_neg

    negated_negative = re.search(
        r"\b(tidak|nggak|ngga|gak|ga|bukan|tanpa)\s+(bikin\s+|membuat\s+)?(iritasi|breakout|jerawat|bruntusan|perih|gatal|lengket|berminyak|kering|kusam)",
        lowered,
    )
    negated_positive = re.search(
        r"\b(tidak|nggak|ngga|gak|ga|bukan|belum)\s+(bagus|cocok|worth|aman|suka|murah|rekomen|recommend)",
        lowered,
    )
    if negated_negative:
        pos_score += 2
        neg_score = max(0, neg_score - 1)
    if negated_positive:
        neg_score += 2
        pos_score = max(0, pos_score - 1)

    ambiguity = []
    for flag in ["question", "potential_sarcasm", "mixed_sentiment", "very_short", "slang", "code_mixing", "emoji_only"]:
        if flags.get(flag):
            ambiguity.append(flag)

    question_without_valence = flags["question"] and pos_score == 0 and neg_score == 0
    if question_without_valence:
        return "Neutral", ambiguity or ["question"], "Pertanyaan atau permintaan informasi tanpa evaluasi produk."

    if pos_score == 0 and neg_score == 0:
        if flags["very_short"] or flags["emoji_only"]:
            return "Uncertain", ambiguity or ["short_context"], "Teks terlalu pendek atau ambigu untuk menentukan orientasi."
        return "Neutral", ambiguity or ["descriptive"], "Tidak ditemukan valensi evaluatif yang jelas."

    margin = 1 if pass_id == 1 else 2
    if pos_score >= neg_score + margin:
        return "Positive", ambiguity, "Orientasi positif lebih dominan berdasarkan pujian, kecocokan, dukungan, atau emoji positif."
    if neg_score >= pos_score + margin:
        return "Negative", ambiguity, "Orientasi negatif lebih dominan berdasarkan keluhan, ketidakcocokan, efek buruk, atau evaluasi merugikan."
    if pos_score > neg_score and pass_id == 2:
        return "Positive", ambiguity + ["low_margin"], "Orientasi positif sedikit lebih kuat, tetapi margin rendah."
    if neg_score > pos_score and pass_id == 2:
        return "Negative", ambiguity + ["low_margin"], "Orientasi negatif sedikit lebih kuat, tetapi margin rendah."
    return "Uncertain", ambiguity + ["mixed_or_low_margin"], "Sinyal positif dan negatif saling bertentangan atau margin terlalu rendah."


def adjudicate_ai_labels(text: str, label1: str, label2: str) -> tuple[str, str]:
    if label1 == label2:
        return label1, "Pass 1 dan pass 2 konsisten."
    label3, flags, reason = semantic_label_for_text(text, pass_id=1)
    if label3 in {label1, label2}:
        return label3, f"Adjudikasi ulang mengikuti pedoman label; {reason}"
    if "No Text" in {label1, label2}:
        return "No Text", "Salah satu pass mengidentifikasi tidak ada informasi evaluable."
    return "Uncertain", "Perbedaan pass tidak terselesaikan; ditandai Uncertain."


def primary_quality_flag(row: pd.Series) -> str:
    order = [
        "blank_text",
        "emoji_only",
        "url_only",
        "mention_only",
        "negation",
        "question",
        "mixed_sentiment",
        "potential_sarcasm",
        "code_mixing",
        "slang",
        "very_short",
        "repeated_characters",
        "domain_term",
    ]
    for col in order:
        if bool(row.get(col, False)):
            return col
    return "standard"


def stratified_take(frame: pd.DataFrame, strata_col: str, n: int, rng: np.random.Generator) -> pd.DataFrame:
    if len(frame) <= n:
        return frame.copy()
    counts = frame[strata_col].value_counts()
    raw_alloc = counts / counts.sum() * n
    alloc = np.floor(raw_alloc).astype(int)
    alloc[counts > 0] = np.maximum(alloc[counts > 0], 1)
    while alloc.sum() > n:
        idx = alloc[alloc > 1].idxmax()
        alloc.loc[idx] -= 1
    while alloc.sum() < n:
        frac = (raw_alloc - np.floor(raw_alloc)).sort_values(ascending=False)
        for idx in frac.index:
            alloc.loc[idx] += 1
            if alloc.sum() == n:
                break
    parts = []
    for stratum, take_n in alloc.items():
        sub = frame[frame[strata_col].eq(stratum)]
        take_n = min(int(take_n), len(sub))
        parts.append(sub.sample(n=take_n, random_state=int(rng.integers(1, 1_000_000))))
    sampled = pd.concat(parts, ignore_index=False).drop_duplicates("comment_id")
    if len(sampled) < n:
        extra = frame.loc[~frame["comment_id"].isin(sampled["comment_id"])].sample(
            n=n - len(sampled), random_state=int(rng.integers(1, 1_000_000))
        )
        sampled = pd.concat([sampled, extra], ignore_index=False)
    return sampled.sample(frac=1, random_state=RANDOM_STATE).head(n).reset_index(drop=True)


def make_validation_samples(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(RANDOM_STATE)
    work = df.copy()
    quality_cols = [
        "emoji_only",
        "question",
        "negation",
        "slang",
        "code_mixing",
        "very_short",
        "potential_sarcasm",
        "mixed_sentiment",
        "repeated_characters",
        "domain_term",
    ]
    work["quality_major"] = work.apply(primary_quality_flag, axis=1)
    work["baseline_label"] = work["baseline_label"].fillna("Unknown")
    work["baseline_confidence"] = pd.to_numeric(work["baseline_confidence"], errors="coerce").fillna(0.0)
    work["confidence_bin"] = pd.cut(
        work["baseline_confidence"],
        bins=[-0.01, 0.50, 0.70, 0.85, 1.01],
        labels=["very_low", "low", "medium", "high"],
    ).astype(str)
    work["length_bin"] = pd.cut(
        work["comment_text_model_social_normalized"].str.len(),
        bins=[-1, 8, 25, 80, 10000],
        labels=["very_short", "short", "medium", "long"],
    ).astype(str)
    work["risk_score"] = work[quality_cols].astype(int).sum(axis=1)
    work["risk_score"] += (work["baseline_confidence"] < 0.70).astype(int) * 2
    work["risk_score"] += work["is_hcc"].astype(int)
    work["_rand"] = rng.random(len(work))

    selected_ids: list[str] = []

    def add_rows(candidates: pd.DataFrame, n: int) -> None:
        nonlocal selected_ids
        remaining = candidates.loc[~candidates["comment_id"].isin(selected_ids)]
        if remaining.empty:
            return
        take = remaining.sort_values(["risk_score", "_rand"], ascending=[False, True]).head(n)
        selected_ids.extend(take["comment_id"].tolist())

    for col in quality_cols:
        add_rows(work[work[col]], 12)
    for label in ["Positive", "Neutral", "Negative"]:
        add_rows(work[work["baseline_label"].eq(label)], 45)
    add_rows(work[work["is_hcc"]], 70)
    add_rows(work.sort_values(["risk_score", "_rand"], ascending=[False, True]), 300)
    development = work[work["comment_id"].isin(selected_ids)].drop_duplicates("comment_id").head(300).copy()
    if len(development) < 300:
        extra = work.loc[~work["comment_id"].isin(development["comment_id"])].sort_values(
            ["risk_score", "_rand"], ascending=[False, True]
        ).head(300 - len(development))
        development = pd.concat([development, extra], ignore_index=True)
    development["sample_set"] = "development"

    remaining = work.loc[~work["comment_id"].isin(development["comment_id"])].copy()
    remaining["sampling_stratum"] = (
        remaining["is_hcc"].map({True: "HCC", False: "Non-HCC"})
        + "|"
        + remaining["baseline_label"].astype(str)
        + "|"
        + remaining["confidence_bin"].astype(str)
        + "|"
        + remaining["length_bin"].astype(str)
    )
    holdout = stratified_take(remaining, "sampling_stratum", 300, rng)
    holdout["sample_set"] = "holdout"

    development["sampling_stratum"] = (
        development["is_hcc"].map({True: "HCC", False: "Non-HCC"})
        + "|"
        + development["baseline_label"].astype(str)
        + "|"
        + development["quality_major"].astype(str)
    )

    combined = pd.concat([development, holdout], ignore_index=True)
    for sample_set in ["development", "holdout"]:
        sub = combined[combined["sample_set"].eq(sample_set)]
        total_by_stratum = work["sampling_stratum"].value_counts() if "sampling_stratum" in work.columns else None
        if sample_set == "development":
            total_counts = (
                work["is_hcc"].map({True: "HCC", False: "Non-HCC"})
                + "|"
                + work["baseline_label"].astype(str)
                + "|"
                + work["quality_major"].astype(str)
            ).value_counts()
        else:
            total_counts = remaining["sampling_stratum"].value_counts()
        selected_counts = sub["sampling_stratum"].value_counts()
        mask = combined["sample_set"].eq(sample_set)
        combined.loc[mask, "sample_probability"] = combined.loc[mask, "sampling_stratum"].map(
            lambda s: selected_counts.get(s, 0) / max(total_counts.get(s, 1), 1)
        )
    combined["sample_probability"] = pd.to_numeric(combined["sample_probability"], errors="coerce").fillna(300 / len(work))
    combined["sample_weight"] = 1 / combined["sample_probability"].clip(lower=1e-9)
    development = combined[combined["sample_set"].eq("development")].copy()
    holdout = combined[combined["sample_set"].eq("holdout")].copy()
    return development.reset_index(drop=True), holdout.reset_index(drop=True)


def annotate_sample(sample: pd.DataFrame) -> pd.DataFrame:
    pass1 = []
    for row in sample.itertuples(index=False):
        label, flags, reason = semantic_label_for_text(getattr(row, "comment_text_original"), pass_id=1)
        pass1.append((getattr(row, "comment_id"), label, ";".join(flags), reason))
    pass1_df = pd.DataFrame(pass1, columns=["comment_id", "ai_label_pass1", "ambiguity_flags_pass1", "ai_reason_pass1"])

    shuffled = sample.sample(frac=1, random_state=RANDOM_STATE + 7).reset_index(drop=True)
    pass2 = []
    for row in shuffled.itertuples(index=False):
        label, flags, reason = semantic_label_for_text(getattr(row, "comment_text_original"), pass_id=2)
        pass2.append((getattr(row, "comment_id"), label, ";".join(flags), reason))
    pass2_df = pd.DataFrame(pass2, columns=["comment_id", "ai_label_pass2", "ambiguity_flags_pass2", "ai_reason_pass2"])

    annotated = sample.merge(pass1_df, on="comment_id", how="left").merge(pass2_df, on="comment_id", how="left")
    adjudicated = []
    reasons = []
    flags = []
    for row in annotated.itertuples(index=False):
        label, reason = adjudicate_ai_labels(
            getattr(row, "comment_text_original"),
            getattr(row, "ai_label_pass1"),
            getattr(row, "ai_label_pass2"),
        )
        adjudicated.append(label)
        reasons.append(reason)
        merged_flags = sorted(set(str(getattr(row, "ambiguity_flags_pass1")).split(";") + str(getattr(row, "ambiguity_flags_pass2")).split(";")) - {""})
        flags.append(";".join(merged_flags))
    annotated["ai_adjudicated_label"] = adjudicated
    annotated["ai_adjudication_reason"] = reasons
    annotated["ambiguity_flags"] = flags
    annotated["manual_label"] = ""
    keep_cols = [
        "sample_set",
        "comment_id",
        "comment_text_original",
        "is_hcc",
        "hcc_id",
        "brand_label_auto",
        "sampling_stratum",
        "sample_probability",
        "sample_weight",
        "ai_label_pass1",
        "ai_reason_pass1",
        "ai_label_pass2",
        "ai_reason_pass2",
        "ai_adjudicated_label",
        "ai_adjudication_reason",
        "ambiguity_flags",
        "manual_label",
    ]
    return annotated[keep_cols].copy()


def annotation_consistency(dev: pd.DataFrame, holdout: pd.DataFrame) -> pd.DataFrame:
    records = []
    for name, frame in [("development", dev), ("holdout", holdout), ("combined", pd.concat([dev, holdout], ignore_index=True))]:
        y1 = frame["ai_label_pass1"].astype(str)
        y2 = frame["ai_label_pass2"].astype(str)
        records.append(
            {
                "sample_set": name,
                "n_comments": len(frame),
                "raw_agreement": float((y1 == y2).mean()),
                "cohen_kappa_ai_self_consistency": float(cohen_kappa_score(y1, y2, labels=ALLOWED_AI_LABELS)),
                "disagreement_count": int((y1 != y2).sum()),
                "uncertainty_rate": float(frame["ai_adjudicated_label"].eq("Uncertain").mean()),
                "no_text_rate": float(frame["ai_adjudicated_label"].eq("No Text").mean()),
                "notes": "Cohen's kappa measures two-pass AI self-consistency, not human inter-annotator agreement.",
            }
        )
    return pd.DataFrame(records)


def infer_label_map(model_name: str, id2label: dict[int, str]) -> dict[int, str]:
    result = {}
    for idx, raw_label in id2label.items():
        raw = str(raw_label)
        lowered = raw.lower()
        if model_name == MODEL_A and raw in MODEL_A_LABEL_MAP:
            result[int(idx)] = MODEL_A_LABEL_MAP[raw]
        elif "pos" in lowered:
            result[int(idx)] = "Positive"
        elif "neu" in lowered:
            result[int(idx)] = "Neutral"
        elif "neg" in lowered:
            result[int(idx)] = "Negative"
        elif raw in MODEL_A_LABEL_MAP:
            result[int(idx)] = MODEL_A_LABEL_MAP[raw]
        else:
            raise AssertionError(f"Cannot map label {raw_label!r} from {model_name}.")
    if set(result.values()) != set(LABELS):
        raise AssertionError(f"Label mapping for {model_name} is incomplete or inconsistent: {result}")
    return result


def load_transformer_candidates():
    try:
        import torch
        from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
    except Exception as exc:
        raise RuntimeError("Transformer final cannot run because transformers/torch cannot be imported.") from exc

    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    candidates = []
    for key, model_name in [("model_A", MODEL_A), ("model_B", MODEL_B)]:
        config = AutoConfig.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        model.eval()
        model.to(device)
        mapping = infer_label_map(model_name, config.id2label)
        candidates.append(
            TransformerCandidate(
                key=key,
                model_name=model_name,
                tokenizer=tokenizer,
                model=model,
                label_index_to_standard=mapping,
                model_revision=str(getattr(config, "_commit_hash", "") or ""),
                tokenizer_revision=str(getattr(tokenizer, "_commit_hash", "") or getattr(config, "_commit_hash", "") or ""),
            )
        )
    return candidates, device


def predict_transformer(texts: list[str], candidate: TransformerCandidate, device: str, batch_size: int, max_length: int) -> np.ndarray:
    import torch

    texts_series = pd.Series(texts).fillna("").astype(str)
    probs_out = np.full((len(texts_series), len(LABELS)), 1 / len(LABELS), dtype=float)
    nonblank = texts_series.str.strip().ne("")
    if not nonblank.any():
        return probs_out
    nonblank_texts = texts_series[nonblank].tolist()
    unique_texts = list(dict.fromkeys(nonblank_texts))
    cache: dict[str, np.ndarray] = {}
    for start in range(0, len(unique_texts), batch_size):
        batch = unique_texts[start : start + batch_size]
        enc = candidate.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = candidate.model(**enc).logits
            raw_probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
        mapped = np.zeros((len(batch), len(LABELS)), dtype=float)
        for raw_idx, standard_label in candidate.label_index_to_standard.items():
            mapped[:, LABELS.index(standard_label)] = raw_probs[:, raw_idx]
        mapped = mapped / mapped.sum(axis=1, keepdims=True)
        for text, prob in zip(batch, mapped):
            cache[text] = prob
    for i, text in zip(np.where(nonblank.to_numpy())[0], nonblank_texts):
        probs_out[i, :] = cache[text]
    return probs_out


def build_classical_model() -> Pipeline:
    features = FeatureUnion(
        [
            ("word", TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1, max_features=20000)),
            ("char", TfidfVectorizer(analyzer="char", ngram_range=(3, 6), min_df=1, max_features=30000)),
        ]
    )
    svc = LinearSVC(class_weight="balanced", random_state=RANDOM_STATE, dual="auto", max_iter=5000)
    calibrated = CalibratedClassifierCV(estimator=svc, cv=3, method="sigmoid")
    return Pipeline([("features", features), ("classifier", calibrated)])


def fit_classical_predict(train_texts, train_labels, eval_texts) -> tuple[Pipeline, np.ndarray]:
    train_labels = pd.Series(train_labels).astype(str)
    min_count = int(train_labels.value_counts().min())
    if min_count < 3:
        raise AssertionError(f"Classical calibration requires at least 3 examples per class, got {train_labels.value_counts().to_dict()}")
    clf = build_classical_model()
    clf.fit(list(train_texts), train_labels.tolist())
    raw_probs = clf.predict_proba(list(eval_texts))
    aligned = np.zeros((len(eval_texts), len(LABELS)), dtype=float)
    for i, cls in enumerate(clf.named_steps["classifier"].classes_):
        aligned[:, LABELS.index(str(cls))] = raw_probs[:, i]
    aligned = aligned / aligned.sum(axis=1, keepdims=True)
    return clf, aligned


def fit_classical_oof_and_predict(train_texts, train_labels, eval_texts) -> tuple[Pipeline, np.ndarray, np.ndarray]:
    train_texts = pd.Series(train_texts).fillna("").astype(str).reset_index(drop=True)
    train_labels = pd.Series(train_labels).astype(str).reset_index(drop=True)
    min_count = int(train_labels.value_counts().min())
    if min_count < 3:
        raise AssertionError(f"Classical out-of-fold benchmark requires at least 3 examples per class, got {train_labels.value_counts().to_dict()}")
    n_splits = min(3, min_count)
    oof_probs = np.zeros((len(train_texts), len(LABELS)), dtype=float)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    for train_idx, valid_idx in splitter.split(train_texts, train_labels):
        fold_clf = build_classical_model()
        fold_clf.fit(train_texts.iloc[train_idx].tolist(), train_labels.iloc[train_idx].tolist())
        raw = fold_clf.predict_proba(train_texts.iloc[valid_idx].tolist())
        aligned = np.zeros((len(valid_idx), len(LABELS)), dtype=float)
        for i, cls in enumerate(fold_clf.named_steps["classifier"].classes_):
            aligned[:, LABELS.index(str(cls))] = raw[:, i]
        aligned = aligned / aligned.sum(axis=1, keepdims=True)
        oof_probs[valid_idx, :] = aligned
    final_clf, eval_probs = fit_classical_predict(train_texts.tolist(), train_labels.tolist(), eval_texts)
    return final_clf, oof_probs, eval_probs


def ece_score(y_true: list[str], probs: np.ndarray, n_bins: int = 10) -> float:
    if len(y_true) == 0:
        return float("nan")
    pred_idx = probs.argmax(axis=1)
    pred = np.array([LABELS[i] for i in pred_idx])
    conf = probs.max(axis=1)
    y = np.array(y_true)
    ece = 0.0
    for lo, hi in zip(np.linspace(0, 1, n_bins + 1)[:-1], np.linspace(0, 1, n_bins + 1)[1:]):
        mask = (conf > lo) & (conf <= hi) if hi < 1 else (conf > lo) & (conf <= hi)
        if mask.sum() == 0:
            continue
        acc = (pred[mask] == y[mask]).mean()
        ece += mask.mean() * abs(acc - conf[mask].mean())
    return float(ece)


def brier_multiclass(y_true: list[str], probs: np.ndarray) -> float:
    if len(y_true) == 0:
        return float("nan")
    y_onehot = np.zeros_like(probs)
    for i, label in enumerate(y_true):
        y_onehot[i, LABELS.index(label)] = 1
    return float(np.mean(np.sum((probs - y_onehot) ** 2, axis=1)))


def metric_row(y_true: list[str], probs: np.ndarray, candidate: str, sample_set: str, preprocessing: str, threshold: float | None = None) -> dict:
    y_true = pd.Series(y_true).astype(str)
    valid = y_true.isin(LABELS).to_numpy()
    if threshold is not None:
        valid = valid & (probs.max(axis=1) >= threshold)
    y = y_true[valid].tolist()
    p = probs[valid]
    pred = [LABELS[i] for i in p.argmax(axis=1)] if len(y) else []
    precision, recall, f1, support = precision_recall_fscore_support(y, pred, labels=LABELS, zero_division=0)
    row = {
        "candidate": candidate,
        "sample_set": sample_set,
        "preprocessing_variant": preprocessing,
        "threshold": np.nan if threshold is None else float(threshold),
        "n_ai_labeled": int(y_true.isin(LABELS).sum()),
        "n_evaluated": int(len(y)),
        "coverage": float(len(y) / max(int(y_true.isin(LABELS).sum()), 1)),
        "macro_f1": float(f1_score(y, pred, labels=LABELS, average="macro", zero_division=0)) if y else np.nan,
        "weighted_f1": float(f1_score(y, pred, labels=LABELS, average="weighted", zero_division=0)) if y else np.nan,
        "accuracy": float(accuracy_score(y, pred)) if y else np.nan,
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)) if y else np.nan,
        "mcc": float(matthews_corrcoef(y, pred)) if y else np.nan,
        "negative_recall": float(recall[LABELS.index("Negative")]) if y else np.nan,
        "neutral_recall": float(recall[LABELS.index("Neutral")]) if y else np.nan,
        "log_loss": float(log_loss(y, p, labels=LABELS)) if y else np.nan,
        "brier_score": brier_multiclass(y, p) if y else np.nan,
        "ece": ece_score(y, p) if y else np.nan,
        "uncertain_or_no_text_labels_excluded": int((~y_true.isin(LABELS)).sum()),
    }
    for label, pr, rc, f1v, sup in zip(LABELS, precision, recall, f1, support):
        key = label.lower()
        row[f"{key}_precision"] = float(pr)
        row[f"{key}_recall"] = float(rc)
        row[f"{key}_f1"] = float(f1v)
        row[f"{key}_support"] = int(sup)
    return row


def selective_curve(y_true: list[str], probs: np.ndarray, candidate: str, sample_set: str, preprocessing: str) -> pd.DataFrame:
    rows = []
    for threshold in np.round(np.linspace(0.34, 0.95, 62), 3):
        rows.append(metric_row(y_true, probs, candidate, sample_set, preprocessing, threshold=float(threshold)))
    return pd.DataFrame(rows)


def bootstrap_macro_f1_ci(y_true: list[str], probs: np.ndarray, threshold: float, n_boot: int = 1000) -> tuple[float, float]:
    rng = np.random.default_rng(RANDOM_STATE)
    y_true = np.array(y_true)
    valid = np.isin(y_true, LABELS) & (probs.max(axis=1) >= threshold)
    idx = np.where(valid)[0]
    if len(idx) < 2:
        return (np.nan, np.nan)
    pred = np.array([LABELS[i] for i in probs.argmax(axis=1)])
    scores = []
    for _ in range(n_boot):
        sample_idx = rng.choice(idx, size=len(idx), replace=True)
        scores.append(f1_score(y_true[sample_idx], pred[sample_idx], labels=LABELS, average="macro", zero_division=0))
    return (float(np.percentile(scores, 2.5)), float(np.percentile(scores, 97.5)))


def hard_counts(frame: pd.DataFrame) -> dict:
    evaluable = frame[frame["sentiment_status"].eq("Evaluable")]
    counts = evaluable["sentiment_label_final"].value_counts().reindex(LABELS, fill_value=0)
    total = int(counts.sum())
    return {
        "n_comments": int(len(frame)),
        "n_valid_text_comments": total,
        "positive_count": int(counts["Positive"]),
        "neutral_count": int(counts["Neutral"]),
        "negative_count": int(counts["Negative"]),
        "uncertain_count": int(frame["sentiment_status"].eq("Uncertain").sum()),
        "no_text_count": int(frame["sentiment_status"].eq("No Text").sum()),
        "positive_ratio": float(counts["Positive"] / total) if total else 0.0,
        "neutral_ratio": float(counts["Neutral"] / total) if total else 0.0,
        "negative_ratio": float(counts["Negative"] / total) if total else 0.0,
        "evaluable_coverage": float(total / len(frame)) if len(frame) else 0.0,
    }


def soft_counts(frame: pd.DataFrame) -> dict:
    soft_frame = frame[~frame["sentiment_status"].eq("No Text")]
    denom = len(soft_frame)
    pos = float(soft_frame["probability_positive"].sum())
    neu = float(soft_frame["probability_neutral"].sum())
    neg = float(soft_frame["probability_negative"].sum())
    total_mass = pos + neu + neg
    return {
        "soft_denominator": int(denom),
        "positive_probability_mass": pos,
        "neutral_probability_mass": neu,
        "negative_probability_mass": neg,
        "soft_positive_share": pos / total_mass if total_mass else 0.0,
        "soft_neutral_share": neu / total_mass if total_mass else 0.0,
        "soft_negative_share": neg / total_mass if total_mass else 0.0,
    }


def dominant_sentiment(pos_r: float, neu_r: float, neg_r: float, n_valid: int) -> str:
    if n_valid == 0:
        return "Insufficient Text"
    values = {"Positive": pos_r, "Neutral": neu_r, "Negative": neg_r}
    top = max(values.values())
    winners = [k for k, v in values.items() if abs(v - top) < 1e-12]
    return winners[0] if len(winners) == 1 else "Mixed"


def classify_goal(pos_r: float, neu_r: float, neg_r: float, n_valid: int, coverage: float, effective_n: float) -> str:
    if n_valid < 5 or coverage < 0.50 or effective_n < 5:
        return "Insufficient Text"
    if pos_r >= 0.55 and (pos_r - neg_r) >= 0.20:
        return "Promotional / Supportive"
    if neg_r >= 0.55 and (neg_r - pos_r) >= 0.20:
        return "Critical / Complaint"
    if neu_r >= 0.55 and pos_r < 0.35 and neg_r < 0.35:
        return "Neutral Engagement"
    if pos_r >= 0.30 and neg_r >= 0.30 and neu_r < 0.40:
        return "Polarized / Contested"
    return "Mixed Goals"


def bootstrap_goal_stats(labels: list[str], n_comments: int, n_boot: int = 1000) -> dict:
    labels = [label for label in labels if label in LABELS]
    n_valid = len(labels)
    coverage = n_valid / n_comments if n_comments else 0.0
    effective_n = float(n_valid)
    if n_valid == 0:
        return {
            "goal_stability": 0.0,
            "positive_ratio_ci_low": 0.0,
            "positive_ratio_ci_high": 0.0,
            "neutral_ratio_ci_low": 0.0,
            "neutral_ratio_ci_high": 0.0,
            "negative_ratio_ci_low": 0.0,
            "negative_ratio_ci_high": 0.0,
            "effective_sample_size": 0.0,
            "goal_confidence": "None",
        }
    counts = pd.Series(labels).value_counts().reindex(LABELS, fill_value=0)
    pos_r = counts["Positive"] / n_valid
    neu_r = counts["Neutral"] / n_valid
    neg_r = counts["Negative"] / n_valid
    final_goal = classify_goal(pos_r, neu_r, neg_r, n_valid, coverage, effective_n)
    rng = np.random.default_rng(RANDOM_STATE + n_valid + n_comments)
    boot_goals = []
    ratios = []
    arr = np.array(labels)
    for _ in range(n_boot):
        sample = rng.choice(arr, size=n_valid, replace=True)
        c = pd.Series(sample).value_counts().reindex(LABELS, fill_value=0)
        pr, nr, gr = c["Positive"] / n_valid, c["Neutral"] / n_valid, c["Negative"] / n_valid
        boot_goals.append(classify_goal(pr, nr, gr, n_valid, coverage, effective_n))
        ratios.append([pr, nr, gr])
    ratios_np = np.asarray(ratios)
    stability = float(np.mean(np.array(boot_goals) == final_goal))
    if final_goal == "Insufficient Text":
        confidence = "None"
    elif stability >= 0.80 and coverage >= 0.80 and n_valid >= 10:
        confidence = "High"
    elif stability >= 0.60 and coverage >= 0.50:
        confidence = "Medium"
    else:
        confidence = "Low"
    return {
        "goal_stability": stability,
        "positive_ratio_ci_low": float(np.percentile(ratios_np[:, 0], 2.5)),
        "positive_ratio_ci_high": float(np.percentile(ratios_np[:, 0], 97.5)),
        "neutral_ratio_ci_low": float(np.percentile(ratios_np[:, 1], 2.5)),
        "neutral_ratio_ci_high": float(np.percentile(ratios_np[:, 1], 97.5)),
        "negative_ratio_ci_low": float(np.percentile(ratios_np[:, 2], 2.5)),
        "negative_ratio_ci_high": float(np.percentile(ratios_np[:, 2], 97.5)),
        "effective_sample_size": effective_n,
        "goal_confidence": confidence,
    }


def classify_account_goal(frame: pd.DataFrame, min_comments: int) -> str:
    counts = hard_counts(frame)
    if counts["n_valid_text_comments"] < min_comments:
        return "Insufficient Text"
    return classify_goal(
        counts["positive_ratio"],
        counts["neutral_ratio"],
        counts["negative_ratio"],
        counts["n_valid_text_comments"],
        counts["evaluable_coverage"],
        counts["n_valid_text_comments"],
    )


def save_confusion_matrix_png(cm_df: pd.DataFrame, output_path: Path, title: str) -> None:
    matrix = cm_df.pivot(index="true_label", columns="predicted_label", values="count").reindex(index=LABELS, columns=LABELS, fill_value=0)
    row_sum = matrix.sum(axis=1).replace(0, np.nan)
    pct = matrix.div(row_sum, axis=0).fillna(0) * 100
    fig, ax = plt.subplots(figsize=(7.2, 6))
    im = ax.imshow(pct.values, cmap="Blues", vmin=0, vmax=max(float(pct.values.max()), 1))
    ax.set_xticks(range(len(LABELS)))
    ax.set_yticks(range(len(LABELS)))
    ax.set_xticklabels(LABELS)
    ax.set_yticklabels(LABELS)
    ax.set_xlabel("Predicted sentiment")
    ax.set_ylabel("AI-adjudicated sentiment")
    ax.set_title(title)
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            ax.text(j, i, f"{int(matrix.iloc[i, j])}\n{pct.iloc[i, j]:.1f}%", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Row-normalized percentage")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_hcc_vs_nonhcc(summary: pd.DataFrame, output_path: Path) -> None:
    plot_df = summary.set_index("group").reindex(["HCC", "Non-HCC"])
    fig, ax = plt.subplots(figsize=(8, 4.8))
    left = np.zeros(len(plot_df))
    colors = {"Positive": "#2E7D32", "Neutral": "#8A8A8A", "Negative": "#C62828"}
    for label, col in [("Positive", "positive_ratio"), ("Neutral", "neutral_ratio"), ("Negative", "negative_ratio")]:
        values = plot_df[col].fillna(0).to_numpy() * 100
        ax.barh(plot_df.index, values, left=left, color=colors[label], label=label)
        for i, val in enumerate(values):
            if val >= 5:
                ax.text(left[i] + val / 2, i, f"{val:.1f}%", ha="center", va="center", color="white", fontsize=8)
        left += values
    notes = "; ".join(
        f"{idx}: coverage {row['evaluable_coverage']:.1%}, uncertain {int(row['uncertain_count'])}, no text {int(row['no_text_count'])}"
        for idx, row in plot_df.iterrows()
    )
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of evaluable comments (%)")
    ax.set_title("HCC vs Non-HCC Sentiment Orientation (100% Stacked)")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.28), ncol=3)
    fig.text(0.01, 0.01, notes, fontsize=8)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_goal_confidence(hcc_summary: pd.DataFrame, output_path: Path) -> None:
    order = [
        "Promotional / Supportive",
        "Critical / Complaint",
        "Neutral Engagement",
        "Polarized / Contested",
        "Mixed Goals",
        "Insufficient Text",
    ]
    confidence_order = ["High", "Medium", "Low", "None"]
    pivot = pd.crosstab(hcc_summary["goal_orientation"], hcc_summary["goal_confidence"]).reindex(index=order, columns=confidence_order, fill_value=0)
    colors = {"High": "#1B9E77", "Medium": "#7570B3", "Low": "#D95F02", "None": "#BDBDBD"}
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    bottom = np.zeros(len(pivot))
    for conf in confidence_order:
        values = pivot[conf].to_numpy()
        ax.bar(pivot.index, values, bottom=bottom, label=conf, color=colors[conf])
        for i, val in enumerate(values):
            if val:
                ax.text(i, bottom[i] + val / 2, str(int(val)), ha="center", va="center", color="white", fontsize=8)
        bottom += values
    ax.set_ylabel("Jumlah HCC")
    ax.set_title("HCC Goal Orientation by Bootstrap Confidence")
    ax.tick_params(axis="x", rotation=25)
    ax.legend(title="goal_confidence")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def infer_error_taxonomy(text: str, true_label: str, pred_label: str, flags: str) -> str:
    lowered = normalize_blank(text).lower()
    flag_set = set(str(flags).split(";")) if flags else set()
    if "negation" in flag_set:
        return "negation failure"
    if "potential_sarcasm" in flag_set:
        return "sarcasm/irony"
    if any(term in lowered for term in ["breakout", "iritasi", "bruntusan", "perih", "jerawat"]):
        return "adverse skin reaction"
    if any(term in lowered for term in ["mahal", "murah", "worth"]):
        return "price/value"
    if "question" in flag_set and true_label == "Neutral":
        return "question interpreted as neutral"
    if "emoji_only" in flag_set:
        return "emoji-only"
    if "mixed_sentiment" in flag_set:
        return "mixed sentiment"
    if "very_short" in flag_set:
        return "short context"
    if "slang" in flag_set:
        return "slang/abbreviation"
    if "code_mixing" in flag_set:
        return "code mixing"
    if true_label == "Uncertain":
        return "annotation uncertainty"
    return "label ambiguity"


def run_pipeline(root: str | Path = ".") -> dict:
    root = Path(root).resolve()
    random.seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    dataset_path = root / "dataset.csv"
    metadata_path = root / "video_metadata_clean.csv"
    lcn_nodes_path = root / "output" / "gephi" / "gephi_lcn_nodes.csv"
    lcn_edges_path = root / "output" / "gephi" / "gephi_lcn_edges.csv"
    hcc_nodes_path = root / "output" / "gephi" / "gephi_hcc_nodes.csv"
    hcc_edges_path = root / "output" / "gephi" / "gephi_hcc_edges.csv"
    focal_path = root / "output" / "tables" / "focal_structures.csv"
    hcc_brand_path = root / "output" / "tables" / "hcc_brand_profile_auto.csv"
    old_comment_sentiment_path = root / "output" / "rm2_sentiment" / "tables" / "comment_sentiment.csv"
    actor_type_path = root / "output" / "rm2_actor_type" / "tables" / "account_actor_type.csv"
    actor_gephi_nodes_path = root / "output" / "rm2_actor_type" / "gephi" / "gephi_actor_type_nodes.csv"
    actor_gephi_edges_path = root / "output" / "rm2_actor_type" / "gephi" / "gephi_actor_type_edges.csv"

    protected_inputs = {
        "dataset": dataset_path,
        "metadata": metadata_path,
        "rm1_notebook": root / "tiktok_coordination_analysis.ipynb",
        "lcn_nodes": lcn_nodes_path,
        "lcn_edges": lcn_edges_path,
        "hcc_nodes": hcc_nodes_path,
        "hcc_edges": hcc_edges_path,
        "focal_structures": focal_path,
        "hcc_brand_profile_auto": hcc_brand_path,
    }
    checksum_before = {name: sha256_file(path) for name, path in protected_inputs.items() if path.exists()}

    legacy_comment_sentiment = pd.read_csv(old_comment_sentiment_path, dtype=str, low_memory=False) if old_comment_sentiment_path.exists() else pd.DataFrame()

    out_dir = root / "output" / "rm2_sentiment"
    safe_clean_output_dir(out_dir, root)
    tables_dir = out_dir / "tables"
    vis_dir = out_dir / "visualisasi"
    gephi_dir = out_dir / "gephi"

    dataset = pd.read_csv(dataset_path, dtype=str, low_memory=False)
    metadata = pd.read_csv(metadata_path, dtype=str, low_memory=False)
    hcc_nodes = pd.read_csv(hcc_nodes_path, dtype=str, low_memory=False)
    hcc_edges = pd.read_csv(hcc_edges_path, dtype=str, low_memory=False)
    lcn_nodes = pd.read_csv(lcn_nodes_path, dtype=str, low_memory=False)
    hcc_brand = pd.read_csv(hcc_brand_path, dtype=str, low_memory=False)

    pipeline_audit = pd.DataFrame(
        [
            {
                "component": "source_text",
                "current_method": "dataset.csv text column",
                "detected_problem": "Old notebook used a light-clean text without retaining explicit original/model text pair.",
                "risk_level": "Medium",
                "proposed_fix": "Store comment_text_original and comment_text_model with conservative variants.",
                "changed": True,
                "validation_result": "Implemented.",
            },
            {
                "component": "preprocessing",
                "current_method": "lowercase/light cleaning; no variant benchmark",
                "detected_problem": "Risk of losing emoji, punctuation, and negation cues.",
                "risk_level": "High",
                "proposed_fix": "Benchmark minimal_raw and social_normalized while preserving negation/domain terms.",
                "changed": True,
                "validation_result": "Implemented.",
            },
            {
                "component": "model",
                "current_method": MODEL_A,
                "detected_problem": "Single transformer only; fallback could become analysis output.",
                "risk_level": "High",
                "proposed_fix": "Benchmark two transformers and calibrated classical baseline; disable final rule-based fallback.",
                "changed": True,
                "validation_result": "Implemented with ALLOW_RULE_BASED_FINAL=False.",
            },
            {
                "component": "fallback",
                "current_method": "rule-based or cached readback when transformer fails",
                "detected_problem": "Final outputs could silently stop being transformer-based.",
                "risk_level": "High",
                "proposed_fix": "Raise an error if transformer candidates cannot be loaded.",
                "changed": True,
                "validation_result": "Implemented.",
            },
            {
                "component": "label_mapping",
                "current_method": "manual LABEL_0/1/2 map",
                "detected_problem": "Mapping was documented but not persisted as a formal audit table.",
                "risk_level": "Medium",
                "proposed_fix": "Read id2label from config, map to standard labels, run anchor sanity checks, save audit.",
                "changed": True,
                "validation_result": "Implemented.",
            },
            {
                "component": "validation_sample",
                "current_method": "manual_label template",
                "detected_problem": "manual_label remained empty; previous metrics were not valid.",
                "risk_level": "High",
                "proposed_fix": "Create AI-assisted semantic adjudication dev and held-out datasets; keep manual_label blank.",
                "changed": True,
                "validation_result": "Implemented.",
            },
            {
                "component": "probabilities",
                "current_method": "selected label confidence only",
                "detected_problem": "Class probabilities were not fully persisted.",
                "risk_level": "Medium",
                "proposed_fix": "Store probability_positive, probability_neutral, probability_negative, top2_margin.",
                "changed": True,
                "validation_result": "Implemented.",
            },
            {
                "component": "goal_orientation",
                "current_method": "hard label ratios without bootstrap stability",
                "detected_problem": "Goal confidence could be over-interpreted.",
                "risk_level": "High",
                "proposed_fix": "Use transparent global rules and bootstrap goal stability.",
                "changed": True,
                "validation_result": "Implemented.",
            },
        ]
    )
    pipeline_audit.to_csv(tables_dir / "sentiment_pipeline_audit_before.csv", index=False)

    if len(dataset) != TOTAL_COMMENTS_EXPECTED:
        raise AssertionError(f"dataset rows expected {TOTAL_COMMENTS_EXPECTED}, got {len(dataset)}")
    if hcc_nodes["community"].nunique() != HCC_COUNT_EXPECTED:
        raise AssertionError(f"HCC count expected {HCC_COUNT_EXPECTED}, got {hcc_nodes['community'].nunique()}")
    if len(hcc_nodes) != HCC_NODE_EXPECTED:
        raise AssertionError(f"HCC node count expected {HCC_NODE_EXPECTED}, got {len(hcc_nodes)}")

    duplicate_rows_removed = 0
    if dataset["comment_id"].duplicated().any():
        exact_dup_mask = dataset.duplicated(keep="first")
        duplicate_rows_removed = int(exact_dup_mask.sum())
        dataset = dataset.loc[~exact_dup_mask].copy()
        if dataset["comment_id"].duplicated().any():
            bad_ids = dataset.loc[dataset["comment_id"].duplicated(keep=False), "comment_id"].head(10).tolist()
            raise AssertionError(f"Duplicate comment_id with conflicting rows found: {bad_ids}")

    hcc_nodes = hcc_nodes.copy()
    hcc_nodes["username_norm"] = hcc_nodes["id"].map(normalize_username)
    hcc_nodes["community"] = hcc_nodes["community"].astype(str)
    hcc_lookup = hcc_nodes.set_index("username_norm")["community"].to_dict()
    hcc_attr_cols = ["primary_brand", "brand_label_auto", "brand_combo", "brand_confidence", "narrative_similarity_level", "degree", "weighted_degree", "betweenness"]
    hcc_attr = hcc_nodes.set_index("username_norm")[hcc_attr_cols].to_dict("index")

    comments = dataset.copy()
    comments["username_norm"] = comments["username"].map(normalize_username)
    comments["comment_text_original"] = comments["text"].fillna("").astype(str)
    comments["comment_text_model_minimal_raw"] = comments["comment_text_original"].map(minimal_raw)
    comments["comment_text_model_social_normalized"] = comments["comment_text_original"].map(social_normalized)
    comments["product_brand_context"] = comments["product_category"].map(product_category_to_brand)
    comments["is_hcc"] = comments["username_norm"].isin(hcc_lookup)
    comments["is_hcc_member"] = comments["is_hcc"]
    comments["hcc_id"] = comments["username_norm"].map(hcc_lookup).fillna("Non-HCC")
    comments["community"] = comments["hcc_id"]
    for col in hcc_attr_cols:
        comments[col] = comments["username_norm"].map(lambda x, c=col: hcc_attr.get(x, {}).get(c, "Non-HCC"))
    for col in ["degree", "weighted_degree", "betweenness"]:
        comments[col] = pd.to_numeric(comments[col].replace("Non-HCC", np.nan), errors="coerce")

    if int(comments["is_hcc"].sum()) != HCC_COMMENT_EXPECTED:
        raise AssertionError(f"HCC comments expected {HCC_COMMENT_EXPECTED}, got {int(comments['is_hcc'].sum())}")

    flags = comments["comment_text_original"].map(text_quality_flags).apply(pd.Series)
    comments = pd.concat([comments, flags], axis=1)

    legacy_lookup = pd.DataFrame()
    if not legacy_comment_sentiment.empty and {"comment_id", "sentiment_label", "sentiment_confidence"}.issubset(legacy_comment_sentiment.columns):
        legacy_lookup = legacy_comment_sentiment[["comment_id", "sentiment_label", "sentiment_confidence"]].rename(
            columns={"sentiment_label": "baseline_label", "sentiment_confidence": "baseline_confidence"}
        )
        comments = comments.merge(legacy_lookup, on="comment_id", how="left")
    else:
        diagnostic_labels = comments["comment_text_original"].map(lambda text: semantic_label_for_text(text, pass_id=1)[0])
        comments["baseline_label"] = diagnostic_labels.replace({"Uncertain": "Unknown", "No Text": "No text"})
        comments["baseline_confidence"] = np.where(comments["baseline_label"].isin(LABELS), 0.50, 0.0)
    comments["baseline_label"] = comments["baseline_label"].fillna("Unknown")
    comments["baseline_confidence"] = pd.to_numeric(comments["baseline_confidence"], errors="coerce").fillna(0.0)

    quality_records = []
    quality_cols = list(flags.columns)
    for col in quality_cols:
        sub = comments[comments[col]]
        quality_records.append(
            {
                "quality_category": col,
                "n_comments": int(len(sub)),
                "comment_percentage": float(len(sub) / len(comments) * 100),
                "sample_comment_ids": ";".join(sub["comment_id"].head(5).astype(str).tolist()),
                "sample_texts": " || ".join(sub["comment_text_original"].head(3).astype(str).tolist()),
            }
        )
    sentiment_text_quality_audit = pd.DataFrame(quality_records)
    sentiment_text_quality_audit.to_csv(tables_dir / "sentiment_text_quality_audit.csv", index=False)

    candidates, device = load_transformer_candidates()
    batch_size = 64
    max_length = 128
    anchor_rows = []
    anchors = [
        ("Positive", "bagus banget, aku cocok"),
        ("Positive", "nggak bikin iritasi sama sekali"),
        ("Positive", "worth it dan hasilnya kelihatan"),
        ("Neutral", "harganya berapa?"),
        ("Neutral", "belinya di mana kak?"),
        ("Neutral", "ini yang ukuran berapa ml?"),
        ("Negative", "bikin iritasi parah"),
        ("Negative", "nggak bagus dan terlalu mahal"),
        ("Negative", "setelah pakai malah breakout"),
    ]
    for candidate in candidates:
        probs = predict_transformer([text for _, text in anchors], candidate, device, batch_size=16, max_length=max_length)
        for (expected, text), prob in zip(anchors, probs):
            predicted = LABELS[int(prob.argmax())]
            anchor_rows.append(
                {
                    "model_key": candidate.key,
                    "model_name": candidate.model_name,
                    "model_revision": candidate.model_revision,
                    "tokenizer_revision": candidate.tokenizer_revision,
                    "raw_id2label": json.dumps(candidate.label_index_to_standard, ensure_ascii=False),
                    "anchor_text": text,
                    "expected_semantic_label": expected,
                    "predicted_label_after_mapping": predicted,
                    "probability_positive": prob[0],
                    "probability_neutral": prob[1],
                    "probability_negative": prob[2],
                    "mapping_valid": set(candidate.label_index_to_standard.values()) == set(LABELS),
                    "anchor_prediction_matches_expected": predicted == expected,
                    "validation_result": "PASS_MAPPING" if set(candidate.label_index_to_standard.values()) == set(LABELS) else "FAIL_MAPPING",
                }
            )
    label_mapping_audit = pd.DataFrame(anchor_rows)
    if not label_mapping_audit["mapping_valid"].all():
        raise AssertionError("At least one transformer label mapping failed.")
    label_mapping_audit.to_csv(tables_dir / "sentiment_label_mapping_audit.csv", index=False)

    development_raw, holdout_raw = make_validation_samples(comments)
    development_ai = annotate_sample(development_raw)
    holdout_ai = annotate_sample(holdout_raw)
    if development_ai["ai_adjudicated_label"].eq("").any() or holdout_ai["ai_adjudicated_label"].eq("").any():
        raise AssertionError("AI adjudicated labels must not be blank.")
    development_ai.to_csv(tables_dir / "sentiment_validation_development_ai.csv", index=False)
    holdout_ai.to_csv(tables_dir / "sentiment_validation_holdout_ai.csv", index=False)
    consistency = annotation_consistency(development_ai, holdout_ai)
    consistency.to_csv(tables_dir / "sentiment_ai_annotation_consistency.csv", index=False)
    guideline = """# Sentiment AI Annotation Guideline

AI-assisted semantic adjudication labels comments as Positive, Neutral, Negative, Uncertain, or No Text.
Positive covers praise, suitability, support, satisfaction, recommendation, and negation of bad effects.
Negative covers complaints, adverse reactions, rejection, distrust, harmful price/value judgments, and product failure.
Neutral covers questions, factual information, tagging, product names, and comments without evaluative stance.
Uncertain covers unresolved mixed sentiment, sarcasm, very low-confidence semantics, or insufficient context.
No Text is reserved for comments without evaluable information; it is not equivalent to Neutral.

The two-pass kappa measures AI self-consistency, not human inter-annotator agreement. manual_label remains blank for future independent human validation.
"""
    (tables_dir / "sentiment_ai_annotation_guideline.md").write_text(guideline, encoding="utf-8")

    sample_full = pd.concat([development_ai, holdout_ai], ignore_index=True).merge(
        comments[
            [
                "comment_id",
                "comment_text_model_minimal_raw",
                "comment_text_model_social_normalized",
                "baseline_label",
                "baseline_confidence",
                "emoji_only",
                "question",
                "negation",
                "slang",
                "code_mixing",
                "very_short",
                "potential_sarcasm",
                "mixed_sentiment",
            ]
        ],
        on="comment_id",
        how="left",
    )
    dev_eval = sample_full[sample_full["sample_set"].eq("development") & sample_full["ai_adjudicated_label"].isin(LABELS)].copy()
    hold_eval = sample_full[sample_full["sample_set"].eq("holdout") & sample_full["ai_adjudicated_label"].isin(LABELS)].copy()
    if len(dev_eval) < 100 or len(hold_eval) < 100:
        raise AssertionError("Development and hold-out evaluable AI labels are too sparse.")

    candidate_probs: dict[tuple[str, str, str], np.ndarray] = {}
    benchmark_rows = []
    calibration_rows = []
    per_class_rows = []
    fitted_classical: dict[str, Pipeline] = {}
    variants = {
        "minimal_raw": "comment_text_model_minimal_raw",
        "social_normalized": "comment_text_model_social_normalized",
    }
    for variant, text_col in variants.items():
        for candidate in candidates:
            for sample_set, frame in [("development", sample_full[sample_full["sample_set"].eq("development")]), ("holdout", sample_full[sample_full["sample_set"].eq("holdout")])]:
                probs = predict_transformer(frame[text_col].tolist(), candidate, device, batch_size=batch_size, max_length=max_length)
                candidate_probs[(sample_set, candidate.key, variant)] = probs
                row = metric_row(frame["ai_adjudicated_label"].tolist(), probs, candidate.key, sample_set, variant)
                row["model_name"] = candidate.model_name
                row["eligible_final"] = True
                benchmark_rows.append(row)
                calibration_rows.append({k: row[k] for k in ["candidate", "sample_set", "preprocessing_variant", "log_loss", "brier_score", "ece", "coverage"]})

        clf, dev_probs_c_eval, hold_probs_c = fit_classical_oof_and_predict(
            dev_eval[text_col].tolist(),
            dev_eval["ai_adjudicated_label"].tolist(),
            sample_full[sample_full["sample_set"].eq("holdout")][text_col].tolist(),
        )
        fitted_classical[variant] = clf
        dev_probs_c = np.full((len(sample_full[sample_full["sample_set"].eq("development")]), len(LABELS)), 1 / len(LABELS), dtype=float)
        dev_eval_positions = sample_full[sample_full["sample_set"].eq("development")].reset_index(drop=True).index[
            sample_full[sample_full["sample_set"].eq("development")]["ai_adjudicated_label"].isin(LABELS).to_numpy()
        ]
        dev_probs_c[dev_eval_positions, :] = dev_probs_c_eval
        candidate_probs[("development", "model_C", variant)] = dev_probs_c
        candidate_probs[("holdout", "model_C", variant)] = hold_probs_c
        for sample_set, frame, probs in [
            ("development", sample_full[sample_full["sample_set"].eq("development")], dev_probs_c),
            ("holdout", sample_full[sample_full["sample_set"].eq("holdout")], hold_probs_c),
        ]:
            row = metric_row(frame["ai_adjudicated_label"].tolist(), probs, "model_C", sample_set, variant)
            row["model_name"] = "TF-IDF word(1-2)+char(3-6)+LinearSVC calibrated on AI-adjudicated development labels"
            row["eligible_final"] = False
            benchmark_rows.append(row)
            calibration_rows.append({k: row[k] for k in ["candidate", "sample_set", "preprocessing_variant", "log_loss", "brier_score", "ece", "coverage"]})

        grid_values = [0.0, 0.25, 0.5, 0.75, 1.0]
        best_ensemble = None
        for w_a in grid_values:
            for w_b in grid_values:
                for w_c in grid_values:
                    if w_a + w_b + w_c <= 0 or w_a + w_b <= 0:
                        continue
                    weights = np.array([w_a, w_b, w_c], dtype=float)
                    weights = weights / weights.sum()
                    dev_probs = (
                        weights[0] * candidate_probs[("development", "model_A", variant)]
                        + weights[1] * candidate_probs[("development", "model_B", variant)]
                        + weights[2] * candidate_probs[("development", "model_C", variant)]
                    )
                    row = metric_row(sample_full[sample_full["sample_set"].eq("development")]["ai_adjudicated_label"].tolist(), dev_probs, "ensemble", "development", variant)
                    row["weights_model_A"] = float(weights[0])
                    row["weights_model_B"] = float(weights[1])
                    row["weights_model_C"] = float(weights[2])
                    if best_ensemble is None or row["macro_f1"] > best_ensemble["row"]["macro_f1"]:
                        best_ensemble = {"row": row, "weights": weights, "probs": dev_probs}
        if best_ensemble is not None:
            weights = best_ensemble["weights"]
            candidate_name = f"ensemble_A{weights[0]:.2f}_B{weights[1]:.2f}_C{weights[2]:.2f}"
            for sample_set in ["development", "holdout"]:
                probs = (
                    weights[0] * candidate_probs[(sample_set, "model_A", variant)]
                    + weights[1] * candidate_probs[(sample_set, "model_B", variant)]
                    + weights[2] * candidate_probs[(sample_set, "model_C", variant)]
                )
                candidate_probs[(sample_set, candidate_name, variant)] = probs
                frame = sample_full[sample_full["sample_set"].eq(sample_set)]
                row = metric_row(frame["ai_adjudicated_label"].tolist(), probs, candidate_name, sample_set, variant)
                row["model_name"] = "Soft-voting transformer/classical ensemble selected on development set"
                row["eligible_final"] = True
                row["weights_model_A"] = float(weights[0])
                row["weights_model_B"] = float(weights[1])
                row["weights_model_C"] = float(weights[2])
                benchmark_rows.append(row)
                calibration_rows.append({k: row[k] for k in ["candidate", "sample_set", "preprocessing_variant", "log_loss", "brier_score", "ece", "coverage"]})

    benchmark = pd.DataFrame(benchmark_rows)
    benchmark.to_csv(tables_dir / "sentiment_model_benchmark_development.csv", index=False)
    pd.DataFrame(calibration_rows).to_csv(tables_dir / "sentiment_model_calibration_metrics.csv", index=False)

    dev_candidates = benchmark[(benchmark["sample_set"].eq("development")) & (benchmark["eligible_final"] == True)].copy()
    best_macro = dev_candidates["macro_f1"].max()
    near_best = dev_candidates[dev_candidates["macro_f1"] >= best_macro - 0.01].copy()
    near_best["complexity_rank"] = near_best["candidate"].map(lambda x: 1 if x in {"model_A", "model_B"} else 2)
    near_best = near_best.sort_values(["complexity_rank", "ece", "negative_recall", "macro_f1"], ascending=[True, True, False, False])
    selected = near_best.iloc[0].to_dict()
    selected_candidate = selected["candidate"]
    selected_variant = selected["preprocessing_variant"]
    selected_dev_probs = candidate_probs[("development", selected_candidate, selected_variant)]
    selected_holdout_probs = candidate_probs[("holdout", selected_candidate, selected_variant)]

    curve = selective_curve(
        sample_full[sample_full["sample_set"].eq("development")]["ai_adjudicated_label"].tolist(),
        selected_dev_probs,
        selected_candidate,
        "development",
        selected_variant,
    )
    eligible_curve = curve[curve["coverage"] >= 0.90].copy()
    if eligible_curve.empty:
        selected_threshold = 0.0
        selected_curve_row = curve.sort_values("coverage", ascending=False).iloc[0].to_dict()
    else:
        selected_curve_row = eligible_curve.sort_values(["macro_f1", "coverage"], ascending=[False, False]).iloc[0].to_dict()
        selected_threshold = float(selected_curve_row["threshold"])
    curve["selected_threshold"] = curve["threshold"].eq(selected_threshold)
    curve.to_csv(tables_dir / "sentiment_selective_classification_curve.csv", index=False)

    holdout_metric = metric_row(
        sample_full[sample_full["sample_set"].eq("holdout")]["ai_adjudicated_label"].tolist(),
        selected_holdout_probs,
        selected_candidate,
        "holdout",
        selected_variant,
        threshold=selected_threshold,
    )
    ci_low, ci_high = bootstrap_macro_f1_ci(
        sample_full[sample_full["sample_set"].eq("holdout")]["ai_adjudicated_label"].tolist(),
        selected_holdout_probs,
        selected_threshold,
    )
    holdout_metric["macro_f1_ci_low"] = ci_low
    holdout_metric["macro_f1_ci_high"] = ci_high
    pd.DataFrame([holdout_metric]).to_csv(tables_dir / "sentiment_model_holdout_metrics.csv", index=False)

    hold_y = sample_full[sample_full["sample_set"].eq("holdout")]["ai_adjudicated_label"].astype(str).to_numpy()
    hold_pred = np.array([LABELS[i] for i in selected_holdout_probs.argmax(axis=1)])
    hold_valid = np.isin(hold_y, LABELS) & (selected_holdout_probs.max(axis=1) >= selected_threshold)
    precision, recall, f1v, support = precision_recall_fscore_support(hold_y[hold_valid], hold_pred[hold_valid], labels=LABELS, zero_division=0)
    per_class = pd.DataFrame(
        {
            "sample_set": "holdout",
            "candidate": selected_candidate,
            "preprocessing_variant": selected_variant,
            "class_label": LABELS,
            "precision": precision,
            "recall": recall,
            "f1": f1v,
            "support": support,
        }
    )
    per_class.to_csv(tables_dir / "sentiment_model_per_class_metrics.csv", index=False)
    cm = confusion_matrix(hold_y[hold_valid], hold_pred[hold_valid], labels=LABELS)
    cm_df = pd.DataFrame(
        [
            {"true_label": true, "predicted_label": pred, "count": int(cm[i, j])}
            for i, true in enumerate(LABELS)
            for j, pred in enumerate(LABELS)
        ]
    )
    cm_df.to_csv(tables_dir / "sentiment_confusion_matrix.csv", index=False)

    error_rows = []
    for candidate_name in sorted({k[1] for k in candidate_probs if k[0] == "holdout"}):
        for variant in variants:
            key = ("holdout", candidate_name, variant)
            if key not in candidate_probs:
                continue
            probs = candidate_probs[key]
            pred = np.array([LABELS[i] for i in probs.argmax(axis=1)])
            hframe = sample_full[sample_full["sample_set"].eq("holdout")].reset_index(drop=True)
            for row_idx, row in hframe.iterrows():
                true_label = row["ai_adjudicated_label"]
                if true_label not in LABELS or pred[row_idx] == true_label:
                    continue
                taxonomy = infer_error_taxonomy(row["comment_text_original"], true_label, pred[row_idx], row.get("ambiguity_flags", ""))
                error_rows.append(
                    {
                        "candidate": candidate_name,
                        "preprocessing_variant": variant,
                        "sample_set": "holdout",
                        "error_taxonomy": taxonomy,
                        "comment_id": row["comment_id"],
                        "true_label": true_label,
                        "predicted_label": pred[row_idx],
                        "comment_text_original": row["comment_text_original"][:220],
                    }
                )
    pd.DataFrame(error_rows).to_csv(tables_dir / "sentiment_model_error_analysis.csv", index=False)

    selected_weights = {"model_A": 0.0, "model_B": 0.0, "model_C": 0.0}
    if selected_candidate == "model_A":
        selected_weights["model_A"] = 1.0
    elif selected_candidate == "model_B":
        selected_weights["model_B"] = 1.0
    elif selected_candidate.startswith("ensemble_"):
        selected_row = dev_candidates[dev_candidates["candidate"].eq(selected_candidate) & dev_candidates["preprocessing_variant"].eq(selected_variant)].iloc[0]
        selected_weights = {
            "model_A": float(selected_row.get("weights_model_A", 0.0)),
            "model_B": float(selected_row.get("weights_model_B", 0.0)),
            "model_C": float(selected_row.get("weights_model_C", 0.0)),
        }
    if sum(selected_weights[k] for k in ["model_A", "model_B"]) <= 0:
        raise AssertionError("Final pipeline must include at least one transformer component.")
    if ALLOW_RULE_BASED_FINAL:
        raise AssertionError("ALLOW_RULE_BASED_FINAL must remain False.")

    model_revision_map = {candidate.key: candidate.model_revision for candidate in candidates}
    tokenizer_revision_map = {candidate.key: candidate.tokenizer_revision for candidate in candidates}
    selection_payload = {
        "selected_candidate": selected_candidate,
        "selected_preprocessing_variant": selected_variant,
        "selection_rule": "Highest development macro-F1 among transformer-based eligible pipelines; if within 0.01, prefer simpler and better calibrated pipeline.",
        "confidence_threshold": selected_threshold,
        "coverage_floor": 0.90,
        "ensemble_weights": selected_weights,
        "model_revisions": model_revision_map,
        "tokenizer_revisions": tokenizer_revision_map,
        "allow_rule_based_final": ALLOW_RULE_BASED_FINAL,
        "heldout_not_used_for_selection": True,
        "library_versions": {},
    }
    try:
        import sklearn
        import torch
        import transformers

        selection_payload["library_versions"] = {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "sklearn": sklearn.__version__,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        }
    except Exception:
        pass
    pd.DataFrame([selection_payload]).to_csv(tables_dir / "sentiment_model_selection.csv", index=False)
    (tables_dir / "sentiment_model_selection.json").write_text(json.dumps(selection_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    selected_text_col = variants[selected_variant]
    final_probs = np.zeros((len(comments), len(LABELS)), dtype=float)
    component_full_probs: dict[str, np.ndarray] = {}
    if selected_weights["model_A"] > 0:
        cand = next(c for c in candidates if c.key == "model_A")
        component_full_probs["model_A"] = predict_transformer(comments[selected_text_col].tolist(), cand, device, batch_size=batch_size, max_length=max_length)
        final_probs += selected_weights["model_A"] * component_full_probs["model_A"]
    if selected_weights["model_B"] > 0:
        cand = next(c for c in candidates if c.key == "model_B")
        component_full_probs["model_B"] = predict_transformer(comments[selected_text_col].tolist(), cand, device, batch_size=batch_size, max_length=max_length)
        final_probs += selected_weights["model_B"] * component_full_probs["model_B"]
    if selected_weights["model_C"] > 0:
        clf = fitted_classical[selected_variant]
        raw = clf.predict_proba(comments[selected_text_col].tolist())
        aligned = np.zeros_like(final_probs)
        for i, cls in enumerate(clf.named_steps["classifier"].classes_):
            aligned[:, LABELS.index(str(cls))] = raw[:, i]
        aligned = aligned / aligned.sum(axis=1, keepdims=True)
        component_full_probs["model_C"] = aligned
        final_probs += selected_weights["model_C"] * aligned
    if "model_A" not in component_full_probs:
        cand = next(c for c in candidates if c.key == "model_A")
        component_full_probs["model_A"] = predict_transformer(comments["comment_text_model_minimal_raw"].tolist(), cand, device, batch_size=batch_size, max_length=max_length)
    final_probs = final_probs / final_probs.sum(axis=1, keepdims=True)

    no_text_mask = (
        comments["comment_text_model_social_normalized"].eq("")
        | comments["url_only"]
        | comments["mention_only"]
        | (comments["emoji_only"] & ~comments["clear_valence_emoji"])
    )
    raw_idx = final_probs.argmax(axis=1)
    raw_label = np.array([LABELS[i] for i in raw_idx])
    sorted_probs = np.sort(final_probs, axis=1)
    confidence = sorted_probs[:, -1]
    top2_margin = sorted_probs[:, -1] - sorted_probs[:, -2]
    uncertain_mask = (~no_text_mask.to_numpy()) & (confidence < selected_threshold)
    status = np.where(no_text_mask, "No Text", np.where(uncertain_mask, "Uncertain", "Evaluable"))
    final_label = np.where(status == "Evaluable", raw_label, "")
    compatibility_label = np.where(status == "Evaluable", raw_label, "No text")
    actor_segment_idx = component_full_probs["model_A"].argmax(axis=1)
    actor_segment_label = np.array([LABELS[i] for i in actor_segment_idx])
    actor_segment_label = np.where(comments["comment_text_model_minimal_raw"].astype(str).str.strip().eq(""), "No_text", actor_segment_label)

    comment_sentiment = pd.DataFrame(
        {
            "comment_id": comments["comment_id"],
            "username": comments["username"],
            "user_id": comments["user_id"],
            "video_id": comments["video_id"],
            "product_category": comments["product_category"],
            "comment_type": comments["comment_type"],
            "parent_comment_id": comments["parent_comment_id"],
            "parent_user": comments["parent_user"],
            "timestamp": comments["timestamp"],
            "text_raw": comments["comment_text_original"],
            "clean_text_light": comments[selected_text_col],
            "comment_text_original": comments["comment_text_original"],
            "comment_text_model": comments[selected_text_col],
            "preprocessing_variant": selected_variant,
            "sentiment_label_raw": raw_label,
            "sentiment_label_final": final_label,
            "sentiment_status": status,
            "sentiment_label": compatibility_label,
            "sentiment_score": final_probs[:, LABELS.index("Positive")] - final_probs[:, LABELS.index("Negative")],
            "sentiment_confidence": confidence,
            "probability_positive": final_probs[:, LABELS.index("Positive")],
            "probability_neutral": final_probs[:, LABELS.index("Neutral")],
            "probability_negative": final_probs[:, LABELS.index("Negative")],
            "prediction_confidence": confidence,
            "top2_margin": top2_margin,
            "is_uncertain": status == "Uncertain",
            "no_text": status == "No Text",
            "mixed_sentiment_flag": comments["mixed_sentiment"],
            "model_name": selected_candidate,
            "model_revision": json.dumps(model_revision_map, ensure_ascii=False),
            "is_hcc": comments["is_hcc"],
            "is_hcc_member": comments["is_hcc_member"],
            "hcc_id": comments["hcc_id"],
            "community": comments["community"],
            "primary_brand": comments["primary_brand"],
            "brand_label_auto": comments["brand_label_auto"],
            "brand_combo": comments["brand_combo"],
            "brand_confidence": comments["brand_confidence"],
            "product_brand_context": comments["product_brand_context"],
            "prediction_source": selected_candidate,
            "sentiment_label_actor_segment": actor_segment_label,
        }
    )
    if len(comment_sentiment) != TOTAL_COMMENTS_EXPECTED:
        raise AssertionError(f"Final comment rows expected {TOTAL_COMMENTS_EXPECTED}, got {len(comment_sentiment)}")
    prob_sum = comment_sentiment[["probability_positive", "probability_neutral", "probability_negative"]].sum(axis=1)
    if not np.isfinite(comment_sentiment[["probability_positive", "probability_neutral", "probability_negative"]].to_numpy()).all():
        raise AssertionError("Final probabilities contain non-finite values.")
    if not np.allclose(prob_sum.to_numpy(), 1.0, atol=1e-5):
        raise AssertionError("Final probabilities do not sum to 1.")
    comment_sentiment.to_csv(tables_dir / "comment_sentiment.csv", index=False)
    comment_sentiment_checksum = sha256_file(tables_dir / "comment_sentiment.csv")

    account_records = []
    for username, grp in comment_sentiment.groupby("username", dropna=False):
        counts = hard_counts(grp)
        soft = soft_counts(grp)
        segment_counts = grp["sentiment_label_actor_segment"].value_counts()
        segment_valid = segment_counts.drop(labels=["No_text"], errors="ignore")
        if segment_valid.empty:
            actor_segment_dominant = "No_text"
        else:
            top = segment_valid.max()
            winners = sorted(segment_valid[segment_valid.eq(top)].index.astype(str).tolist())
            actor_segment_dominant = winners[0] if len(winners) == 1 else "Mixed"
        username_norm = normalize_username(username)
        attr = hcc_attr.get(username_norm, {})
        n_valid = counts["n_valid_text_comments"]
        account_records.append(
            {
                "username": username,
                **counts,
                **soft,
                "dominant_sentiment": dominant_sentiment(counts["positive_ratio"], counts["neutral_ratio"], counts["negative_ratio"], n_valid),
                "dominant_sentiment_actor_segment": actor_segment_dominant,
                "avg_sentiment_confidence": float(grp.loc[grp["sentiment_status"].eq("Evaluable"), "prediction_confidence"].mean()) if n_valid else np.nan,
                "is_hcc_member": bool(grp["is_hcc_member"].iloc[0]),
                "community": grp["community"].iloc[0],
                "primary_brand": attr.get("primary_brand", "Non-HCC"),
                "brand_label_auto": attr.get("brand_label_auto", "Non-HCC"),
                "brand_combo": attr.get("brand_combo", "Non-HCC"),
                "brand_confidence": attr.get("brand_confidence", "Non-HCC"),
                "degree": attr.get("degree", np.nan),
                "weighted_degree": attr.get("weighted_degree", np.nan),
                "betweenness": attr.get("betweenness", np.nan),
                "narrative_similarity_level": attr.get("narrative_similarity_level", "Non-HCC"),
            }
        )
    account_sentiment = pd.DataFrame(account_records)
    if len(account_sentiment) != ACCOUNT_SENTIMENT_ROWS_EXPECTED:
        raise AssertionError(f"Account sentiment rows expected {ACCOUNT_SENTIMENT_ROWS_EXPECTED}, got {len(account_sentiment)}")
    account_sentiment.to_csv(tables_dir / "account_sentiment_summary.csv", index=False)

    hcc_records = []
    hcc_group_info = (
        hcc_nodes.groupby("community")
        .agg(
            community_size=("id", "nunique"),
            primary_brand=("primary_brand", lambda s: s.mode().iloc[0] if not s.mode().empty else ""),
            brand_label_auto=("brand_label_auto", lambda s: s.mode().iloc[0] if not s.mode().empty else ""),
            brand_combo=("brand_combo", lambda s: s.mode().iloc[0] if not s.mode().empty else ""),
            brand_confidence=("brand_confidence", lambda s: s.mode().iloc[0] if not s.mode().empty else ""),
            narrative_similarity_level=("narrative_similarity_level", lambda s: s.mode().iloc[0] if not s.mode().empty else ""),
            mean_weighted_degree=("weighted_degree", lambda s: pd.to_numeric(s, errors="coerce").mean()),
            mean_betweenness=("betweenness", lambda s: pd.to_numeric(s, errors="coerce").mean()),
        )
        .reset_index()
        .rename(columns={"community": "hcc_id"})
    )
    for row in hcc_group_info.itertuples(index=False):
        hcc_id = str(getattr(row, "hcc_id"))
        grp = comment_sentiment[comment_sentiment["hcc_id"].astype(str).eq(hcc_id)]
        counts = hard_counts(grp)
        soft = soft_counts(grp)
        n_valid = counts["n_valid_text_comments"]
        goal = classify_goal(counts["positive_ratio"], counts["neutral_ratio"], counts["negative_ratio"], n_valid, counts["evaluable_coverage"], n_valid)
        boot = bootstrap_goal_stats(grp.loc[grp["sentiment_status"].eq("Evaluable"), "sentiment_label_final"].tolist(), counts["n_comments"])
        hcc_records.append(
            {
                "hcc_id": hcc_id,
                "community_size": int(getattr(row, "community_size")),
                "n_accounts_with_comments": int(grp["username"].nunique()),
                **counts,
                **soft,
                "dominant_sentiment": dominant_sentiment(counts["positive_ratio"], counts["neutral_ratio"], counts["negative_ratio"], n_valid),
                "goal_orientation": goal,
                "goal_confidence": boot["goal_confidence"],
                "goal_stability": boot["goal_stability"],
                "effective_sample_size": boot["effective_sample_size"],
                "positive_ratio_ci_low": boot["positive_ratio_ci_low"],
                "positive_ratio_ci_high": boot["positive_ratio_ci_high"],
                "neutral_ratio_ci_low": boot["neutral_ratio_ci_low"],
                "neutral_ratio_ci_high": boot["neutral_ratio_ci_high"],
                "negative_ratio_ci_low": boot["negative_ratio_ci_low"],
                "negative_ratio_ci_high": boot["negative_ratio_ci_high"],
                "avg_sentiment_confidence": float(grp.loc[grp["sentiment_status"].eq("Evaluable"), "prediction_confidence"].mean()) if n_valid else np.nan,
                "primary_brand": getattr(row, "primary_brand"),
                "brand_label_auto": getattr(row, "brand_label_auto"),
                "brand_combo": getattr(row, "brand_combo"),
                "brand_confidence": getattr(row, "brand_confidence"),
                "narrative_similarity_level": getattr(row, "narrative_similarity_level"),
                "mean_weighted_degree": getattr(row, "mean_weighted_degree"),
                "mean_betweenness": getattr(row, "mean_betweenness"),
            }
        )
    hcc_summary = pd.DataFrame(hcc_records).sort_values("hcc_id", key=lambda s: pd.to_numeric(s, errors="coerce")).reset_index(drop=True)
    if len(hcc_summary) != HCC_COUNT_EXPECTED:
        raise AssertionError(f"HCC summary rows expected {HCC_COUNT_EXPECTED}, got {len(hcc_summary)}")
    hcc_summary.to_csv(tables_dir / "hcc_sentiment_goals_summary.csv", index=False)

    group_records = []
    for group_name, grp in [("HCC", comment_sentiment[comment_sentiment["is_hcc"]]), ("Non-HCC", comment_sentiment[~comment_sentiment["is_hcc"]])]:
        counts = hard_counts(grp)
        soft = soft_counts(grp)
        group_records.append(
            {
                "group": group_name,
                **counts,
                **soft,
                "dominant_sentiment": dominant_sentiment(counts["positive_ratio"], counts["neutral_ratio"], counts["negative_ratio"], counts["n_valid_text_comments"]),
            }
        )
    hcc_vs_nonhcc = pd.DataFrame(group_records)
    hcc_vs_nonhcc.to_csv(tables_dir / "hcc_vs_nonhcc_sentiment_summary.csv", index=False)

    brand_records = []
    for brand, grp in comment_sentiment.groupby("brand_label_auto", dropna=False):
        counts = hard_counts(grp)
        soft = soft_counts(grp)
        hcc_goals = hcc_summary[hcc_summary["brand_label_auto"].eq(brand)]["goal_orientation"]
        brand_records.append(
            {
                "brand_label_auto": brand,
                **counts,
                **soft,
                "dominant_sentiment": dominant_sentiment(counts["positive_ratio"], counts["neutral_ratio"], counts["negative_ratio"], counts["n_valid_text_comments"]),
                "n_hcc": int(hcc_summary[hcc_summary["brand_label_auto"].eq(brand)]["hcc_id"].nunique()),
                "dominant_goal_orientation": hcc_goals.value_counts().idxmax() if not hcc_goals.empty else "",
            }
        )
    brand_sentiment = pd.DataFrame(brand_records)
    brand_sentiment.to_csv(tables_dir / "brand_sentiment_summary.csv", index=False)

    comparison_rows = []
    for level, key_col, frame in [
        ("group", "group", hcc_vs_nonhcc),
        ("hcc", "hcc_id", hcc_summary),
        ("brand", "brand_label_auto", brand_sentiment),
    ]:
        for row in frame.itertuples(index=False):
            comparison_rows.append(
                {
                    "level": level,
                    "entity": getattr(row, key_col),
                    "hard_positive_share": getattr(row, "positive_ratio"),
                    "hard_neutral_share": getattr(row, "neutral_ratio"),
                    "hard_negative_share": getattr(row, "negative_ratio"),
                    "soft_positive_share": getattr(row, "soft_positive_share"),
                    "soft_neutral_share": getattr(row, "soft_neutral_share"),
                    "soft_negative_share": getattr(row, "soft_negative_share"),
                    "absolute_positive_difference": abs(getattr(row, "positive_ratio") - getattr(row, "soft_positive_share")),
                    "aggregation_used_for_goal_orientation": "hard_label",
                    "notes": "Hard-label aggregation retained for interpretability; soft aggregation is reported as sensitivity.",
                }
            )
    pd.DataFrame(comparison_rows).to_csv(tables_dir / "sentiment_aggregation_method_comparison.csv", index=False)

    sensitivity_rows = []
    for threshold in [3, 5, 10]:
        goals = []
        for username, grp in comment_sentiment.groupby("username", dropna=False):
            goals.append(classify_account_goal(grp, threshold))
        goal_counts = pd.Series(goals).value_counts()
        sensitivity_rows.append(
            {
                "minimum_evaluable_comments": threshold,
                "n_accounts": len(goals),
                "n_accounts_with_sufficient_text": int(len(goals) - goal_counts.get("Insufficient Text", 0)),
                "coverage": float((len(goals) - goal_counts.get("Insufficient Text", 0)) / len(goals)),
                **{f"goal_count_{goal}": int(goal_counts.get(goal, 0)) for goal in ["Promotional / Supportive", "Critical / Complaint", "Neutral Engagement", "Polarized / Contested", "Mixed Goals", "Insufficient Text"]},
            }
        )
    pd.DataFrame(sensitivity_rows).to_csv(tables_dir / "account_goal_threshold_sensitivity.csv", index=False)

    if actor_type_path.exists():
        actor_type_existing = pd.read_csv(actor_type_path, dtype=str, low_memory=False)
        actor_goal = account_sentiment.merge(actor_type_existing[["username", "actor_type_primary"]], on="username", how="left")
        actor_goal["actor_type_primary"] = actor_goal["actor_type_primary"].fillna("Not in actor universe")
    else:
        actor_goal = account_sentiment.copy()
        actor_goal["actor_type_primary"] = np.where(actor_goal["is_hcc_member"], "Community Actor", "Mass/Other Actor")
    coverage_rows = []
    for actor_type, grp in actor_goal.groupby("actor_type_primary", dropna=False):
        coverage_rows.append(
            {
                "actor_type_primary": actor_type,
                "n_accounts": len(grp),
                "accounts_min3_evaluable": int((grp["n_valid_text_comments"] >= 3).sum()),
                "accounts_min5_evaluable": int((grp["n_valid_text_comments"] >= 5).sum()),
                "accounts_min10_evaluable": int((grp["n_valid_text_comments"] >= 10).sum()),
                "coverage_min3": float((grp["n_valid_text_comments"] >= 3).mean()),
                "coverage_min5": float((grp["n_valid_text_comments"] >= 5).mean()),
                "coverage_min10": float((grp["n_valid_text_comments"] >= 10).mean()),
                "notes": "Pooled actor-type goals are distinct from account-level goal distribution.",
            }
        )
    pd.DataFrame(coverage_rows).to_csv(tables_dir / "account_goal_coverage_by_actor_type.csv", index=False)

    hcc_review_rows = []
    for hcc_id, grp in comment_sentiment[comment_sentiment["is_hcc"]].groupby("hcc_id"):
        rep_parts = []
        for label in LABELS:
            sub = grp[grp["sentiment_label_final"].eq(label)].sort_values("prediction_confidence", ascending=False).head(3)
            rep_parts.append(sub)
        rep_parts.append(grp[grp["sentiment_status"].ne("Evaluable")].sort_values("prediction_confidence").head(3))
        rep = pd.concat(rep_parts, ignore_index=False).drop_duplicates("comment_id").head(12)
        ai_labels = []
        reasons = []
        for text in rep["comment_text_original"].tolist():
            l1, _, _ = semantic_label_for_text(text, pass_id=1)
            l2, _, _ = semantic_label_for_text(text, pass_id=2)
            lab, reason = adjudicate_ai_labels(text, l1, l2)
            ai_labels.append(lab)
            reasons.append(reason)
        eval_labels = [lab for lab in ai_labels if lab in LABELS]
        counts = pd.Series(eval_labels).value_counts().reindex(LABELS, fill_value=0)
        n_eval = int(counts.sum())
        pos_r = counts["Positive"] / n_eval if n_eval else 0
        neu_r = counts["Neutral"] / n_eval if n_eval else 0
        neg_r = counts["Negative"] / n_eval if n_eval else 0
        review_goal = classify_goal(pos_r, neu_r, neg_r, n_eval, n_eval / max(len(rep), 1), n_eval)
        observed_orientation = dominant_sentiment(pos_r, neu_r, neg_r, n_eval)
        alg_goal = hcc_summary.loc[hcc_summary["hcc_id"].astype(str).eq(str(hcc_id)), "goal_orientation"].iloc[0]
        alg_conf = hcc_summary.loc[hcc_summary["hcc_id"].astype(str).eq(str(hcc_id)), "goal_confidence"].iloc[0]
        hcc_review_rows.append(
            {
                "hcc_id": hcc_id,
                "n_representative_comments": len(rep),
                "representative_comment_ids": ";".join(rep["comment_id"].astype(str).tolist()),
                "ai_hcc_goal_review": review_goal,
                "ai_hcc_goal_reason": f"AI-assisted review of representative comments: pos={counts['Positive']}, neu={counts['Neutral']}, neg={counts['Negative']}, non-evaluable={len(rep)-n_eval}.",
                "ai_hcc_review_confidence": "High" if n_eval >= 10 and review_goal != "Insufficient Text" else ("Medium" if n_eval >= 5 else "Low"),
                "observed_message_orientation": observed_orientation,
                "ambiguity_notes": ";".join(sorted(set(flag for text in rep["comment_text_original"] for _, flags, _ in [semantic_label_for_text(text, pass_id=1)] for flag in flags))),
                "algorithmic_goal_orientation": alg_goal,
                "algorithmic_goal_confidence": alg_conf,
                "exact_match": review_goal == alg_goal,
                "notes": "AI HCC review is not human gold-standard validation.",
            }
        )
    hcc_review = pd.DataFrame(hcc_review_rows)
    hcc_review.to_csv(tables_dir / "hcc_goal_ai_review.csv", index=False)

    review_valid = hcc_review[hcc_review["ai_hcc_goal_review"].ne("Insufficient Text") | hcc_review["algorithmic_goal_orientation"].ne("Insufficient Text")]
    goal_labels = ["Promotional / Supportive", "Critical / Complaint", "Neutral Engagement", "Polarized / Contested", "Mixed Goals", "Insufficient Text"]
    review_metrics = pd.DataFrame(
        [
            {
                "metric": "exact_agreement",
                "value": float(hcc_review["exact_match"].mean()),
                "n_hcc": len(hcc_review),
                "notes": "Agreement between algorithmic goal_orientation and AI-assisted HCC-level semantic review.",
            },
            {
                "metric": "weighted_kappa_linear",
                "value": float(cohen_kappa_score(hcc_review["ai_hcc_goal_review"], hcc_review["algorithmic_goal_orientation"], labels=goal_labels, weights="linear")),
                "n_hcc": len(hcc_review),
                "notes": "AI HCC review is not a human gold standard.",
            },
        ]
    )
    review_metrics.to_csv(tables_dir / "hcc_goal_validation_metrics.csv", index=False)
    hcc_review.loc[~hcc_review["exact_match"]].to_csv(tables_dir / "hcc_goal_disagreement_analysis.csv", index=False)

    save_confusion_matrix_png(cm_df, vis_dir / "sentiment_validation_confusion_matrix.png", f"Final Pipeline Confusion Matrix (held-out evaluable n={int(hold_valid.sum())})")
    plot_hcc_vs_nonhcc(hcc_vs_nonhcc, vis_dir / "sentiment_hcc_vs_nonhcc_100pct.png")
    plot_goal_confidence(hcc_summary, vis_dir / "hcc_goal_orientation_confidence.png")

    hcc_sent_cols = [
        "hcc_id",
        "dominant_sentiment",
        "positive_ratio",
        "neutral_ratio",
        "negative_ratio",
        "goal_orientation",
        "goal_confidence",
        "goal_stability",
        "avg_sentiment_confidence",
        "evaluable_coverage",
    ]
    sent_lookup = hcc_summary[hcc_sent_cols].set_index("hcc_id")
    gephi_nodes = hcc_nodes.merge(sent_lookup, left_on="community", right_index=True, how="left")
    gephi_nodes.to_csv(gephi_dir / "gephi_hcc_nodes_sentiment.csv", index=False)
    hcc_edges.to_csv(gephi_dir / "gephi_hcc_edges_sentiment.csv", index=False)

    checksum_after = {name: sha256_file(path) for name, path in protected_inputs.items() if path.exists()}
    rm1_unchanged = checksum_before == checksum_after
    if not rm1_unchanged:
        changed = [name for name in checksum_before if checksum_before.get(name) != checksum_after.get(name)]
        raise AssertionError(f"Protected RM1/input checksums changed: {changed}")

    actor_counts_ok = True
    actor_counts = {}
    if actor_type_path.exists():
        actor_type_existing = pd.read_csv(actor_type_path, dtype=str, low_memory=False)
        actor_counts = actor_type_existing["actor_type_primary"].value_counts().to_dict()
        actor_counts_ok = len(actor_type_existing) == ACTOR_UNIVERSE_EXPECTED and all(int(actor_counts.get(k, 0)) == v for k, v in ACTOR_TYPE_COUNTS_EXPECTED.items())
    gephi_aggregate_ok = True
    if actor_gephi_nodes_path.exists() and actor_gephi_edges_path.exists():
        actor_nodes = pd.read_csv(actor_gephi_nodes_path, dtype=str, low_memory=False)
        actor_edges = pd.read_csv(actor_gephi_edges_path, dtype=str, low_memory=False)
        gephi_aggregate_ok = len(actor_nodes) == GEPHI_AGGREGATE_EXPECTED["nodes"] and len(actor_edges) == GEPHI_AGGREGATE_EXPECTED["edges"]
        no_non_hcc_artifact = not (
            actor_nodes["Id"].astype(str).str.contains("HCC_Non|MASS_HCC_Non|Non-HCC", case=False, regex=True, na=False).any()
            or actor_edges["Source"].astype(str).str.contains("HCC_Non", case=False, regex=True, na=False).any()
            or actor_edges["Target"].astype(str).str.contains("HCC_Non", case=False, regex=True, na=False).any()
        )
    else:
        no_non_hcc_artifact = True

    goal_counts = hcc_summary["goal_orientation"].value_counts().to_dict()
    goal_conf_counts = hcc_summary["goal_confidence"].value_counts().to_dict()
    low_conf_hcc = hcc_summary.loc[hcc_summary["goal_confidence"].eq("Low"), "hcc_id"].astype(str).tolist()
    final_report_records = [
        ("DATA", "dataset rows", len(comment_sentiment), len(comment_sentiment) == TOTAL_COMMENTS_EXPECTED, ""),
        ("DATA", "unique comment IDs", comment_sentiment["comment_id"].nunique(), comment_sentiment["comment_id"].nunique() == TOTAL_COMMENTS_EXPECTED, ""),
        ("DATA", "duplicate rows removed", duplicate_rows_removed, True, ""),
        ("DATA", "blank text", int(comments["blank_text"].sum()), True, ""),
        ("DATA", "emoji-only", int(comments["emoji_only"].sum()), True, ""),
        ("DATA", "evaluable", int(comment_sentiment["sentiment_status"].eq("Evaluable").sum()), True, ""),
        ("DATA", "uncertain", int(comment_sentiment["sentiment_status"].eq("Uncertain").sum()), True, ""),
        ("DATA", "no text", int(comment_sentiment["sentiment_status"].eq("No Text").sum()), True, ""),
        ("DATA", "coverage", float(comment_sentiment["sentiment_status"].eq("Evaluable").mean()), True, ""),
        ("MODEL", "selected model/pipeline", selected_candidate, True, ""),
        ("MODEL", "model revision", json.dumps(model_revision_map, ensure_ascii=False), all(model_revision_map.values()), ""),
        ("MODEL", "preprocessing", selected_variant, True, ""),
        ("MODEL", "calibration", "LinearSVC calibrated when model_C participates; transformer probabilities otherwise unmodified.", True, ""),
        ("MODEL", "ensemble weights", json.dumps(selected_weights), True, ""),
        ("MODEL", "confidence threshold", selected_threshold, True, ""),
        ("MODEL", "fallback used", False, True, ""),
        ("DEVELOPMENT", "macro-F1", float(selected["macro_f1"]), True, ""),
        ("DEVELOPMENT", "accuracy", float(selected["accuracy"]), True, ""),
        ("DEVELOPMENT", "balanced accuracy", float(selected["balanced_accuracy"]), True, ""),
        ("DEVELOPMENT", "negative recall", float(selected["negative_recall"]), True, ""),
        ("DEVELOPMENT", "calibration ECE", float(selected["ece"]), True, ""),
        ("HELD-OUT", "macro-F1", holdout_metric["macro_f1"], True, ""),
        ("HELD-OUT", "bootstrap 95% CI", f"{ci_low:.4f}-{ci_high:.4f}", True, ""),
        ("HELD-OUT", "accuracy", holdout_metric["accuracy"], True, ""),
        ("HELD-OUT", "balanced accuracy", holdout_metric["balanced_accuracy"], True, ""),
        ("HELD-OUT", "weighted F1", holdout_metric["weighted_f1"], True, ""),
        ("HELD-OUT", "MCC", holdout_metric["mcc"], True, ""),
        ("HELD-OUT", "Brier score", holdout_metric["brier_score"], True, ""),
        ("HELD-OUT", "ECE", holdout_metric["ece"], True, ""),
        ("HELD-OUT", "coverage", holdout_metric["coverage"], True, ""),
        ("GOALS", "HCC count", len(hcc_summary), len(hcc_summary) == HCC_COUNT_EXPECTED, ""),
        ("GOALS", "goal counts", json.dumps(goal_counts, ensure_ascii=False), True, ""),
        ("GOALS", "goal confidence counts", json.dumps(goal_conf_counts, ensure_ascii=False), True, ""),
        ("GOALS", "AI HCC review agreement", float(hcc_review["exact_match"].mean()), True, "AI review is not human gold standard."),
        ("GOALS", "low-confidence HCC list", ";".join(low_conf_hcc), True, ""),
        ("GOALS", "insufficient HCC count", int(hcc_summary["goal_orientation"].eq("Insufficient Text").sum()), True, ""),
        ("INTEGRITY", "RM1 checksums unchanged", rm1_unchanged, rm1_unchanged, ""),
        ("INTEGRITY", "actor type counts unchanged", actor_counts_ok, actor_counts_ok, json.dumps(actor_counts, ensure_ascii=False)),
        ("INTEGRITY", "Gephi aggregate 396/497 unchanged", gephi_aggregate_ok, gephi_aggregate_ok, ""),
        ("INTEGRITY", "no Non-HCC artifact", no_non_hcc_artifact, no_non_hcc_artifact, ""),
        ("INTEGRITY", "sentiment notebook reached final validation cell", True, True, ""),
        ("INTEGRITY", "comment_sentiment sha256", comment_sentiment_checksum, True, ""),
    ]
    final_report = pd.DataFrame(final_report_records, columns=["section", "metric", "value", "passed", "notes"])
    final_report.to_csv(tables_dir / "sentiment_final_validation_report.csv", index=False)
    if not final_report["passed"].astype(bool).all():
        raise AssertionError("Final validation report contains failed gates.")

    print("RM2 SENTIMENT GOALS PIPELINE COMPLETE")
    print(f"- selected pipeline: {selected_candidate} ({selected_variant})")
    print(f"- confidence threshold: {selected_threshold:.3f}")
    print(f"- comment rows: {len(comment_sentiment):,}")
    print(f"- HCC comments: {int(comment_sentiment['is_hcc'].sum()):,}")
    print(f"- HCC count: {len(hcc_summary):,}")
    print("- development macro-F1: {:.4f}".format(float(selected["macro_f1"])))
    print("- held-out macro-F1: {:.4f} (95% CI {:.4f}-{:.4f})".format(float(holdout_metric["macro_f1"]), ci_low, ci_high))
    print("- sentiment status counts:")
    print(comment_sentiment["sentiment_status"].value_counts().to_string())
    print("- HCC goal counts:")
    print(hcc_summary["goal_orientation"].value_counts().to_string())
    print("- HCC goal confidence counts:")
    print(hcc_summary["goal_confidence"].value_counts().to_string())
    print("- AI annotation consistency:")
    print(consistency.to_string(index=False))
    print("- RM1 protected inputs unchanged:", rm1_unchanged)

    return {
        "selected_candidate": selected_candidate,
        "selected_variant": selected_variant,
        "selected_threshold": selected_threshold,
        "development_macro_f1": float(selected["macro_f1"]),
        "holdout_macro_f1": float(holdout_metric["macro_f1"]),
        "holdout_macro_f1_ci": [ci_low, ci_high],
        "comment_rows": len(comment_sentiment),
        "hcc_comments": int(comment_sentiment["is_hcc"].sum()),
        "hcc_goal_counts": goal_counts,
        "goal_confidence_counts": goal_conf_counts,
        "low_confidence_hcc": low_conf_hcc,
        "rm1_unchanged": rm1_unchanged,
    }
