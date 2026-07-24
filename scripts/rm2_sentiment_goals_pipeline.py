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
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
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
ALLOWED_REFERENCE_LABELS = ["Positive", "Neutral", "Negative", "Uncertain", "No Text"]
ALLOW_RULE_BASED_FINAL = False
REFERENCE_LABEL_SOURCE = "heuristic_pseudo_label"
VALIDATION_MODE = "PROVISIONAL"
FINAL_VALIDATION_STATUS = "PROVISIONAL"
HUMAN_VALIDATION_COMPLETED = False
COMPLETED_HUMAN_ANNOTATION_FILENAME = "sentiment_human_annotation_validated.csv"

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


def safe_clean_output_dir(path: Path, root: Path, preserve_names: set[str] | None = None) -> None:
    resolved_root = root.resolve()
    resolved_path = path.resolve()
    if resolved_path == resolved_root or resolved_root not in resolved_path.parents:
        raise ValueError(f"Refusing to clean unsafe path: {resolved_path}")
    protected_preserve_names = {
        "final",
        "legacy",
        "model",
        "validation",
        "direct_interaction",
        "audit",
    }
    preserve_names = protected_preserve_names | (preserve_names or set())
    if path.exists():
        for child in path.iterdir():
            if child.name in preserve_names:
                continue
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
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


def heuristic_label_for_text(text: str, pass_id: int = 1) -> tuple[str, list[str], str]:
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


def adjudicate_heuristic_labels(text: str, label1: str, label2: str) -> tuple[str, str]:
    if label1 == label2:
        return label1, "Pass 1 dan pass 2 konsisten."
    label3, flags, reason = heuristic_label_for_text(text, pass_id=1)
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


def make_validation_samples(df: pd.DataFrame, locked_test_exclusion_ids: set[str] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    locked_test_exclusion_ids = {str(x) for x in (locked_test_exclusion_ids or set())}
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

    remaining = work.loc[
        ~work["comment_id"].astype(str).isin(set(development["comment_id"].astype(str)) | locked_test_exclusion_ids)
    ].copy()
    if len(remaining) < 300:
        raise AssertionError(
            f"Not enough unseen comments for locked test after excluding current development and previous validation samples: {len(remaining)}"
        )
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
    holdout["sample_set"] = "locked_test"

    development["sampling_stratum"] = (
        development["is_hcc"].map({True: "HCC", False: "Non-HCC"})
        + "|"
        + development["baseline_label"].astype(str)
        + "|"
        + development["quality_major"].astype(str)
    )

    combined = pd.concat([development, holdout], ignore_index=True)
    for sample_set in ["development", "locked_test"]:
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
    holdout = combined[combined["sample_set"].eq("locked_test")].copy()
    if set(development["comment_id"].astype(str)) & set(holdout["comment_id"].astype(str)):
        raise AssertionError("Development and locked test samples overlap.")
    if locked_test_exclusion_ids & set(holdout["comment_id"].astype(str)):
        raise AssertionError("Locked test contains previously used validation comment IDs.")
    return development.reset_index(drop=True), holdout.reset_index(drop=True)


def annotate_heuristic_reference_sample(sample: pd.DataFrame) -> pd.DataFrame:
    pass1 = []
    for row in sample.itertuples(index=False):
        label, flags, reason = heuristic_label_for_text(getattr(row, "comment_text_original"), pass_id=1)
        pass1.append((getattr(row, "comment_id"), label, ";".join(flags), reason))
    pass1_df = pd.DataFrame(pass1, columns=["comment_id", "heuristic_label_pass1", "ambiguity_flags_pass1", "heuristic_reason_pass1"])

    shuffled = sample.sample(frac=1, random_state=RANDOM_STATE + 7).reset_index(drop=True)
    pass2 = []
    for row in shuffled.itertuples(index=False):
        label, flags, reason = heuristic_label_for_text(getattr(row, "comment_text_original"), pass_id=2)
        pass2.append((getattr(row, "comment_id"), label, ";".join(flags), reason))
    pass2_df = pd.DataFrame(pass2, columns=["comment_id", "heuristic_label_pass2", "ambiguity_flags_pass2", "heuristic_reason_pass2"])

    annotated = sample.merge(pass1_df, on="comment_id", how="left").merge(pass2_df, on="comment_id", how="left")
    adjudicated = []
    reasons = []
    flags = []
    for row in annotated.itertuples(index=False):
        label, reason = adjudicate_heuristic_labels(
            getattr(row, "comment_text_original"),
            getattr(row, "heuristic_label_pass1"),
            getattr(row, "heuristic_label_pass2"),
        )
        adjudicated.append(label)
        reasons.append(reason)
        merged_flags = sorted(set(str(getattr(row, "ambiguity_flags_pass1")).split(";") + str(getattr(row, "ambiguity_flags_pass2")).split(";")) - {""})
        flags.append(";".join(merged_flags))
    annotated["heuristic_reference_label"] = adjudicated
    annotated["heuristic_reference_reason"] = reasons
    annotated["ambiguity_flags"] = flags
    annotated["manual_label"] = ""
    keep_cols = [
        "sample_set",
        "comment_id",
        "comment_text_original",
        "video_id",
        "product_category",
        "is_hcc",
        "hcc_id",
        "brand_label_auto",
        "sampling_stratum",
        "sample_probability",
        "sample_weight",
        "heuristic_label_pass1",
        "heuristic_reason_pass1",
        "heuristic_label_pass2",
        "heuristic_reason_pass2",
        "heuristic_reference_label",
        "heuristic_reference_reason",
        "ambiguity_flags",
        "manual_label",
    ]
    return annotated[keep_cols].copy()


def deterministic_rule_reproducibility(dev: pd.DataFrame, holdout: pd.DataFrame) -> pd.DataFrame:
    records = []
    for name, frame in [("development", dev), ("locked_test", holdout), ("combined", pd.concat([dev, holdout], ignore_index=True))]:
        y1 = frame["heuristic_label_pass1"].astype(str)
        y2 = frame["heuristic_label_pass2"].astype(str)
        records.append(
            {
                "sample_set": name,
                "n_comments": len(frame),
                "deterministic_rule_reproducibility": float((y1 == y2).mean()),
                "cohen_kappa_deterministic_reproducibility": float(cohen_kappa_score(y1, y2, labels=ALLOWED_REFERENCE_LABELS)),
                "disagreement_count": int((y1 != y2).sum()),
                "uncertainty_rate": float(frame["heuristic_reference_label"].eq("Uncertain").mean()),
                "no_text_rate": float(frame["heuristic_reference_label"].eq("No Text").mean()),
                "passes_independent": False,
                "use_as_validation_evidence": False,
                "notes": "This metric measures reproducibility of the same deterministic rule system, not annotation reliability or human inter-annotator agreement.",
            }
        )
    return pd.DataFrame(records)


def write_human_validation_package(sample_full: pd.DataFrame, human_dir: Path) -> pd.DataFrame:
    human_dir.mkdir(parents=True, exist_ok=True)
    allowed = [
        {
            "label": "Positive",
            "definition": "Pujian, pengalaman cocok, dukungan, rekomendasi, kepuasan, atau evaluasi yang jelas menguntungkan.",
        },
        {
            "label": "Neutral",
            "definition": "Pertanyaan, informasi faktual, tagging, nama produk/brand saja, atau komentar tanpa posisi evaluatif.",
        },
        {
            "label": "Negative",
            "definition": "Keluhan, ketidakcocokan, efek buruk, kekecewaan, penolakan, peringatan, atau evaluasi yang jelas merugikan.",
        },
        {
            "label": "Uncertain",
            "definition": "Sarkasme, mixed sentiment, target ambigu, atau konteks tidak cukup untuk menetapkan Positive/Neutral/Negative.",
        },
        {
            "label": "No Text",
            "definition": "Tidak ada informasi yang dapat dievaluasi; bukan sinonim Neutral.",
        },
    ]
    pd.DataFrame(allowed).to_csv(human_dir / "sentiment_human_annotation_codebook.csv", index=False)

    blind_cols = ["sample_set", "comment_id", "comment_text_original", "video_id", "product_category"]
    blind = sample_full[blind_cols].copy()
    blind = blind.rename(columns={"product_category": "brand_or_video_context"})
    for col in [
        "annotator_1_label",
        "annotator_1_notes",
        "annotator_2_label",
        "annotator_2_notes",
        "adjudicated_human_label",
        "adjudication_notes",
    ]:
        blind[col] = ""
    forbidden = {
        "sentiment_label_raw",
        "sentiment_label_final",
        "heuristic_reference_label",
        "prediction_confidence",
        "baseline_confidence",
        "goal_orientation",
        "hcc_id",
    }
    if forbidden & set(blind.columns):
        raise AssertionError(f"Blind human annotation file contains forbidden columns: {sorted(forbidden & set(blind.columns))}")
    if len(blind) < 600:
        raise AssertionError(f"Human validation package requires at least 600 comments, got {len(blind)}")
    if blind["comment_id"].duplicated().any():
        raise AssertionError("Human validation package contains duplicate comment_id values.")
    blind.to_csv(human_dir / "sentiment_human_annotation_blind.csv", index=False)
    blind.to_csv(human_dir / "sentiment_human_validation_template.csv", index=False)

    guideline = """# Sentiment Human Annotation Guideline

Tujuan anotasi ini adalah memberi reference labels manusia untuk evaluasi sentimen komentar TikTok skincare.
Label manusia tidak boleh diisi oleh pipeline, Codex, heuristic rules, atau model.

Allowed labels: Positive, Neutral, Negative, Uncertain, No Text.

Sentimen dipakai sebagai indikator orientasi pesan, bukan bukti niat, afiliasi, pembayaran, kontrol, pengaruh kausal, buzzer, bot, atau astroturfing.
Gunakan `Uncertain` untuk sarkasme, mixed sentiment yang tidak terselesaikan, target ambigu, atau konteks terlalu pendek.
Gunakan `No Text` hanya jika komentar tidak memiliki informasi yang dapat dievaluasi.
"""
    (human_dir / "sentiment_human_annotation_guideline.md").write_text(guideline, encoding="utf-8")

    readme = """# RM2 Sentiment Human Validation Package

File utama untuk anotator adalah `sentiment_human_annotation_blind.csv`.
File tersebut tidak menyertakan prediksi model, heuristic reference label, confidence model, HCC goal result, atau hcc_id.

Isi `annotator_1_label` dan `annotator_2_label` menggunakan label pada `sentiment_human_annotation_codebook.csv`.
Setelah adjudication, isi `adjudicated_human_label`.

Validator menolak label di luar daftar allowed labels, menghitung agreement, Cohen's kappa, per-class disagreement, confusion matrix, dan adjudication coverage. Selama label manusia kosong, pipeline RM2 sentiment berjalan sebagai exploratory/provisional dan tidak boleh diklaim validated.
"""
    (human_dir / "README_HUMAN_VALIDATION.md").write_text(readme, encoding="utf-8")

    status = validate_human_annotation_frame(blind)
    status.to_csv(human_dir / "human_validation_status.csv", index=False)
    return status


def validate_human_annotation_frame(frame: pd.DataFrame) -> pd.DataFrame:
    required = [
        "comment_id",
        "comment_text_original",
        "annotator_1_label",
        "annotator_2_label",
        "adjudicated_human_label",
    ]
    missing = [col for col in required if col not in frame.columns]
    if missing:
        return pd.DataFrame(
            [
                {
                    "metric": "required_columns",
                    "value": ";".join(missing),
                    "status": "FAIL",
                    "notes": "Human validation file is missing required columns.",
                }
            ]
        )
    allowed = set(ALLOWED_REFERENCE_LABELS)
    a1 = frame["annotator_1_label"].fillna("").astype(str).str.strip()
    a2 = frame["annotator_2_label"].fillna("").astype(str).str.strip()
    adjudicated = frame["adjudicated_human_label"].fillna("").astype(str).str.strip()
    complete_pair = a1.ne("") & a2.ne("")
    complete_adjudication = adjudicated.ne("")
    invalid = sorted((set(a1[a1.ne("")]) | set(a2[a2.ne("")]) | set(adjudicated[adjudicated.ne("")])) - allowed)
    rows = [
        {
            "metric": "human_validation_completed",
            "value": bool(complete_pair.all() and complete_adjudication.all() and not invalid),
            "status": "PASS" if complete_pair.all() and complete_adjudication.all() and not invalid else "NOT_AVAILABLE",
            "notes": "Human labels are read from a completed annotation file; the pipeline does not create human labels.",
        },
        {
            "metric": "blank_annotator_1_labels",
            "value": int(a1.eq("").sum()),
            "status": "PASS" if int(a1.eq("").sum()) == 0 else "NOT_AVAILABLE",
            "notes": "",
        },
        {
            "metric": "blank_annotator_2_labels",
            "value": int(a2.eq("").sum()),
            "status": "PASS" if int(a2.eq("").sum()) == 0 else "NOT_AVAILABLE",
            "notes": "",
        },
        {
            "metric": "blank_adjudicated_human_labels",
            "value": int(adjudicated.eq("").sum()),
            "status": "PASS" if int(adjudicated.eq("").sum()) == 0 else "NOT_AVAILABLE",
            "notes": "",
        },
        {
            "metric": "invalid_human_labels",
            "value": ";".join(invalid),
            "status": "FAIL" if invalid else "PASS",
            "notes": "Allowed labels are Positive, Neutral, Negative, Uncertain, No Text.",
        },
        {
            "metric": "adjudication_coverage",
            "value": float(complete_adjudication.mean()) if len(frame) else 0.0,
            "status": "PASS" if complete_adjudication.all() and len(frame) else "NOT_AVAILABLE",
            "notes": "",
        },
    ]
    if complete_pair.any() and not invalid:
        rows.append(
            {
                "metric": "annotator_raw_agreement",
                "value": float((a1[complete_pair] == a2[complete_pair]).mean()),
                "status": "PASS",
                "notes": "Agreement between human annotators only.",
            }
        )
        rows.append(
            {
                "metric": "cohens_kappa_human_annotators",
                "value": float(cohen_kappa_score(a1[complete_pair], a2[complete_pair], labels=ALLOWED_REFERENCE_LABELS)),
                "status": "PASS",
                "notes": "Human inter-annotator agreement.",
            }
        )
    else:
        rows.extend(
            [
                {
                    "metric": "annotator_raw_agreement",
                    "value": "",
                    "status": "NOT_AVAILABLE",
                    "notes": "Human labels are blank.",
                },
                {
                    "metric": "cohens_kappa_human_annotators",
                    "value": "",
                    "status": "NOT_AVAILABLE",
                    "notes": "Human labels are blank.",
                },
            ]
        )
    return pd.DataFrame(rows)


def human_validation_is_completed(status: pd.DataFrame) -> bool:
    if status.empty or "metric" not in status.columns or "value" not in status.columns:
        return False
    value = status.loc[status["metric"].eq("human_validation_completed"), "value"]
    return bool(value.astype(str).str.lower().eq("true").any())


def load_completed_human_annotations(human_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    path = human_dir / COMPLETED_HUMAN_ANNOTATION_FILENAME
    if not path.exists():
        return pd.DataFrame(), pd.DataFrame(), False
    frame = pd.read_csv(path, dtype=str, low_memory=False).fillna("")
    status = validate_human_annotation_frame(frame)
    if status["status"].eq("FAIL").any():
        failures = status.loc[status["status"].eq("FAIL"), ["metric", "value", "notes"]].to_dict("records")
        raise AssertionError(f"Human annotation file has invalid labels or structure: {failures}")
    completed = human_validation_is_completed(status)
    if not completed:
        return frame, status, False
    if frame["comment_id"].astype(str).duplicated().any():
        duplicated = frame.loc[frame["comment_id"].astype(str).duplicated(keep=False), "comment_id"].head(10).tolist()
        raise AssertionError(f"Human annotation file contains duplicate comment_id values: {duplicated}")
    sample_counts = frame["sample_set"].astype(str).value_counts().to_dict()
    expected_counts = {"development": 300, "locked_test": 300}
    if sample_counts != expected_counts:
        raise AssertionError(f"Human annotation sample_set counts must be {expected_counts}, got {sample_counts}")
    if set(frame["sample_set"].astype(str)) != set(expected_counts):
        raise AssertionError(f"Human annotation sample_set values must be {sorted(expected_counts)}, got {sorted(set(frame['sample_set'].astype(str)))}")
    return frame, status, True


def build_human_reference_samples(comments: pd.DataFrame, human_annotations: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    required_cols = ["sample_set", "comment_id"]
    missing = [col for col in required_cols if col not in human_annotations.columns]
    if missing:
        raise AssertionError(f"Human annotation file is missing required sample columns: {missing}")
    human_order = human_annotations[required_cols].copy()
    human_order["comment_id"] = human_order["comment_id"].astype(str)
    human_order["_human_order"] = np.arange(len(human_order))
    comment_lookup = comments.copy()
    comment_lookup["comment_id"] = comment_lookup["comment_id"].astype(str)
    merged = human_order.merge(comment_lookup, on="comment_id", how="left", validate="one_to_one")
    missing_ids = merged.loc[merged["username"].isna(), "comment_id"].astype(str).head(10).tolist()
    if missing_ids:
        raise AssertionError(f"Human annotations contain comment_id values not found in dataset: {missing_ids}")
    merged = merged.sort_values("_human_order").drop(columns=["_human_order"]).reset_index(drop=True)
    merged["sampling_stratum"] = "human_validated|" + merged["sample_set"].astype(str)
    merged["sample_probability"] = 1.0
    merged["sample_weight"] = 1.0
    development = merged[merged["sample_set"].eq("development")].copy()
    locked_test = merged[merged["sample_set"].eq("locked_test")].copy()
    if set(development["comment_id"].astype(str)) & set(locked_test["comment_id"].astype(str)):
        raise AssertionError("Human development and locked test samples overlap.")
    return development.reset_index(drop=True), locked_test.reset_index(drop=True)


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
        "n_reference_labeled": int(y_true.isin(LABELS).sum()),
        "n_evaluated": int(len(y)),
        "coverage": float(len(y) / max(int(y_true.isin(LABELS).sum()), 1)),
        "macro_f1": float(f1_score(y, pred, labels=LABELS, average="macro", zero_division=0)) if y else np.nan,
        "weighted_f1": float(f1_score(y, pred, labels=LABELS, average="weighted", zero_division=0)) if y else np.nan,
        "accuracy": float(accuracy_score(y, pred)) if y else np.nan,
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)) if y else np.nan,
        "mcc": float(matthews_corrcoef(y, pred)) if y else np.nan,
        "negative_recall": float(recall[LABELS.index("Negative")]) if y else np.nan,
        "neutral_recall": float(recall[LABELS.index("Neutral")]) if y else np.nan,
        "min_per_class_recall": float(np.min(recall)) if y else np.nan,
        "min_per_class_f1": float(np.min(f1)) if y else np.nan,
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


def repeated_cv_diagnostics(
    development_frame: pd.DataFrame,
    variants: dict[str, str],
    candidate_probs: dict[tuple[str, str, str], np.ndarray],
    reference_col: str,
    reference_label_source: str,
    validation_mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    dev = development_frame.reset_index(drop=True).copy()
    valid_mask = dev[reference_col].isin(LABELS).to_numpy()
    y = dev.loc[valid_mask, reference_col].astype(str).reset_index(drop=True)
    if len(y) < 30 or y.value_counts().min() < 3:
        return pd.DataFrame(), pd.DataFrame()
    splits = RepeatedStratifiedKFold(n_splits=3, n_repeats=5, random_state=RANDOM_STATE)
    valid_positions = np.where(valid_mask)[0]
    for variant, text_col in variants.items():
        valid_text = dev.loc[valid_mask, text_col].fillna("").astype(str).reset_index(drop=True)
        transformer_names = ["model_A", "model_B"]
        transformer_probs = {
            name: candidate_probs[("development", name, variant)][valid_positions]
            for name in transformer_names
            if ("development", name, variant) in candidate_probs
        }
        if len(transformer_probs) == 2:
            transformer_probs["ensemble_transformer_A0.50_B0.50"] = (
                0.5 * transformer_probs["model_A"] + 0.5 * transformer_probs["model_B"]
            )
        for split_no, (train_idx, test_idx) in enumerate(splits.split(valid_text, y), start=1):
            repeat = (split_no - 1) // 3 + 1
            fold = (split_no - 1) % 3 + 1
            for candidate_name, probs in transformer_probs.items():
                row = metric_row(y.iloc[test_idx].tolist(), probs[test_idx], candidate_name, "development_cv", variant)
                row.update(
                    {
                        "repeat": repeat,
                        "fold": fold,
                        "reference_label_source": reference_label_source,
                        "modeling_mode": validation_mode,
                        "notes": f"Repeated CV over {reference_label_source}.",
                    }
                )
                rows.append(row)
            clf, fold_probs = fit_classical_predict(
                valid_text.iloc[train_idx].tolist(),
                y.iloc[train_idx].tolist(),
                valid_text.iloc[test_idx].tolist(),
            )
            row = metric_row(y.iloc[test_idx].tolist(), fold_probs, "model_C_pseudo_label_experiment", "development_cv", variant)
            row.update(
                {
                    "repeat": repeat,
                    "fold": fold,
                    "reference_label_source": reference_label_source,
                    "modeling_mode": validation_mode,
                    "notes": f"Model C is trained inside each fold on {reference_label_source}.",
                }
            )
            rows.append(row)
    cv = pd.DataFrame(rows)
    if cv.empty:
        return cv, pd.DataFrame()
    metric_cols = [
        "macro_f1",
        "balanced_accuracy",
        "mcc",
        "accuracy",
        "weighted_f1",
        "positive_precision",
        "positive_recall",
        "positive_f1",
        "neutral_precision",
        "neutral_recall",
        "neutral_f1",
        "negative_precision",
        "negative_recall",
        "negative_f1",
        "min_per_class_recall",
        "min_per_class_f1",
        "brier_score",
        "ece",
        "coverage",
    ]
    summary = (
        cv.groupby(["candidate", "preprocessing_variant"], dropna=False)[metric_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    summary.columns = ["_".join([str(part) for part in col if str(part) != ""]).rstrip("_") for col in summary.columns]
    summary["reference_label_source"] = reference_label_source
    summary["modeling_mode"] = validation_mode
    summary["notes"] = f"Mean/std over repeated stratified CV folds using {reference_label_source}."
    return cv, summary


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
    ax.set_ylabel("Reference sentiment")
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


def plot_goal_confidence(hcc_summary: pd.DataFrame, output_path: Path, validation_status: str = "PROVISIONAL") -> None:
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
    title_status = "Human-Validated Sentiment Model" if validation_status == "VALIDATED" else validation_status
    ax.set_title(f"HCC Goal Orientation by Bootstrap Stability ({title_status})")
    ax.tick_params(axis="x", rotation=25)
    ax.legend(title="goal_confidence")
    fig.text(0.01, 0.01, "Confidence reflects bootstrap stability, not correctness or direct HCC-goal human validation.", fontsize=8)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
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


def infer_neutral_error_taxonomy(text: str, flags: str) -> str:
    lowered = normalize_blank(text).lower()
    flag_set = set(str(flags).split(";")) if flags else set()
    if "question" in flag_set or "?" in lowered:
        if any(term in lowered for term in ["harga", "harganya", "link", "beli", "belinya", "checkout"]):
            return "harga/link/beli"
        return "pertanyaan"
    if re.fullmatch(r"(@\w+\s*)+", lowered.strip()):
        return "mention/tagging"
    if any(term in lowered for term in ["azarine", "daviena", "maryame", "originote", "the originote"]) and not (
        contains_any(lowered, POSITIVE_TERMS) or contains_any(lowered, NEGATIVE_TERMS)
    ):
        return "brand mention without evaluation"
    if "emoji_only" in flag_set:
        return "emoji"
    if "very_short" in flag_set:
        return "short acknowledgement"
    if "slang" in flag_set:
        return "slang"
    if "code_mixing" in flag_set:
        return "code mixing"
    if "negation" in flag_set:
        return "negation"
    if "mixed_sentiment" in flag_set:
        return "mixed evaluation"
    if "potential_sarcasm" in flag_set:
        return "sarcasm"
    tokens = re.findall(r"\b\w+\b", lowered)
    if 1 <= len(tokens) <= 3 and any(term in lowered for term in DOMAIN_TERMS):
        return "product name only"
    return "brand mention without evaluation" if any(term in lowered for term in DOMAIN_TERMS) else "label ambiguity"


def run_pipeline(root: str | Path = ".") -> dict:
    global REFERENCE_LABEL_SOURCE, VALIDATION_MODE, FINAL_VALIDATION_STATUS, HUMAN_VALIDATION_COMPLETED

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
    previous_validation_id_path = root / "output" / "rm2_sentiment" / "tables" / "sentiment_previous_validation_ids_excluded.csv"
    legacy_validation_names = [
        "sentiment_validation_development_" + "ai.csv",
        "sentiment_validation_holdout_" + "ai.csv",
    ]
    actor_type_path = root / "output" / "rm2_actor_type" / "tables" / "account_actor_type.csv"
    actor_gephi_nodes_path = root / "output" / "rm2_actor_type" / "gephi" / "gephi_actor_type_nodes.csv"
    actor_gephi_edges_path = root / "output" / "rm2_actor_type" / "gephi" / "gephi_actor_type_edges.csv"

    protected_inputs = {
        "dataset": dataset_path,
        "metadata": metadata_path,
        "rm1_notebook": root / "notebooks" / "rm1" / "tiktok_coordination_analysis.ipynb",
        "lcn_nodes": lcn_nodes_path,
        "lcn_edges": lcn_edges_path,
        "hcc_nodes": hcc_nodes_path,
        "hcc_edges": hcc_edges_path,
        "focal_structures": focal_path,
        "hcc_brand_profile_auto": hcc_brand_path,
    }
    checksum_before = {name: sha256_file(path) for name, path in protected_inputs.items() if path.exists()}

    legacy_comment_sentiment = pd.read_csv(old_comment_sentiment_path, dtype=str, low_memory=False) if old_comment_sentiment_path.exists() else pd.DataFrame()
    previously_seen_validation_ids: set[str] = set()
    if previous_validation_id_path.exists():
        previous_validation_ids = pd.read_csv(previous_validation_id_path, dtype=str, low_memory=False)
        if "comment_id" in previous_validation_ids.columns:
            previously_seen_validation_ids.update(previous_validation_ids["comment_id"].dropna().astype(str).tolist())
    for legacy_name in legacy_validation_names:
        old_validation_path = root / "output" / "rm2_sentiment" / "legacy" / "v1" / "tables" / legacy_name
        if old_validation_path.exists():
            old_validation = pd.read_csv(old_validation_path, dtype=str, low_memory=False)
            if "comment_id" in old_validation.columns:
                previously_seen_validation_ids.update(old_validation["comment_id"].dropna().astype(str).tolist())

    out_dir = root / "output" / "rm2_sentiment" / "legacy" / "v1"
    safe_clean_output_dir(out_dir, root)
    tables_dir = out_dir / "tables"
    vis_dir = out_dir / "visualisasi"
    gephi_dir = out_dir / "gephi"
    human_dir = root / "output" / "rm2_sentiment" / "validation" / "human_v1"
    human_annotations, human_validation_status, human_validation_completed_input = load_completed_human_annotations(human_dir)
    if human_validation_completed_input:
        REFERENCE_LABEL_SOURCE = "human_adjudicated_label"
        VALIDATION_MODE = "HUMAN_VALIDATED"
        HUMAN_VALIDATION_COMPLETED = True
        FINAL_VALIDATION_STATUS = "PENDING"
        human_validation_status.to_csv(human_dir / "human_validation_metrics.csv", index=False)
        human_validation_status.to_csv(human_dir / "human_validation_status.csv", index=False)
    else:
        REFERENCE_LABEL_SOURCE = "heuristic_pseudo_label"
        VALIDATION_MODE = "PROVISIONAL"
        HUMAN_VALIDATION_COMPLETED = False
        FINAL_VALIDATION_STATUS = "PROVISIONAL"
    if previously_seen_validation_ids:
        pd.DataFrame(
            {
                "comment_id": sorted(previously_seen_validation_ids),
                "exclusion_reason": "Previously used validation sample excluded from new locked test.",
            }
        ).to_csv(tables_dir / "sentiment_previous_validation_ids_excluded.csv", index=False)

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
                "proposed_fix": "Create deterministic heuristic pseudo-label diagnostics plus a blind human-validation package; keep manual_label/human labels blank.",
                "changed": True,
                "validation_result": "Completed human validation loaded." if HUMAN_VALIDATION_COMPLETED else "Implemented as PROVISIONAL; not human validation.",
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
        diagnostic_labels = comments["comment_text_original"].map(lambda text: heuristic_label_for_text(text, pass_id=1)[0])
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

    if human_validation_completed_input:
        development_raw, holdout_raw = build_human_reference_samples(comments, human_annotations)
    else:
        development_raw, holdout_raw = make_validation_samples(comments, locked_test_exclusion_ids=previously_seen_validation_ids)
    development_reference = annotate_heuristic_reference_sample(development_raw)
    locked_test_reference = annotate_heuristic_reference_sample(holdout_raw)
    if development_reference["heuristic_reference_label"].eq("").any() or locked_test_reference["heuristic_reference_label"].eq("").any():
        raise AssertionError("heuristic reference labels must not be blank.")
    development_reference.to_csv(tables_dir / "sentiment_development_heuristic_reference.csv", index=False)
    locked_test_reference.to_csv(tables_dir / "sentiment_holdout_heuristic_reference.csv", index=False)
    consistency = deterministic_rule_reproducibility(development_reference, locked_test_reference)
    consistency.to_csv(tables_dir / "sentiment_deterministic_rule_reproducibility.csv", index=False)
    guideline = """# Sentiment Heuristic Reference Guideline

heuristic pseudo-labeling labels comments as Positive, Neutral, Negative, Uncertain, or No Text.
Positive covers praise, suitability, support, satisfaction, recommendation, and negation of bad effects.
Negative covers complaints, adverse reactions, rejection, distrust, harmful price/value judgments, and product failure.
Neutral covers questions, factual information, tagging, product names, and comments without evaluative stance.
Uncertain covers unresolved mixed sentiment, sarcasm, very low-confidence semantics, or insufficient context.
No Text is reserved for comments without evaluable information; it is not equivalent to Neutral.

The reproducibility metric measures the same deterministic lexical rule system, not annotation reliability or human inter-annotator agreement. manual_label remains blank for future independent human validation.
"""
    (tables_dir / "sentiment_heuristic_reference_guideline.md").write_text(guideline, encoding="utf-8")

    sample_full = pd.concat([development_reference, locked_test_reference], ignore_index=True).merge(
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
                "domain_term",
            ]
        ],
        on="comment_id",
        how="left",
    )
    if set(development_reference["comment_id"].astype(str)) & set(locked_test_reference["comment_id"].astype(str)):
        raise AssertionError("Development and locked test reference samples overlap.")
    if human_validation_completed_input:
        human_cols = [
            "comment_id",
            "sample_set",
            "annotator_1_label",
            "annotator_1_notes",
            "annotator_2_label",
            "annotator_2_notes",
            "adjudicated_human_label",
            "adjudication_notes",
        ]
        human_merge = human_annotations[human_cols].copy()
        human_merge["comment_id"] = human_merge["comment_id"].astype(str)
        sample_full["comment_id"] = sample_full["comment_id"].astype(str)
        sample_full = sample_full.merge(human_merge, on="comment_id", how="left", suffixes=("", "_human"), validate="one_to_one")
        if not sample_full["sample_set"].astype(str).eq(sample_full["sample_set_human"].astype(str)).all():
            raise AssertionError("Human annotation sample_set does not match reconstructed validation sample.")
        sample_full = sample_full.drop(columns=["sample_set_human"])
        sample_full["reference_label"] = sample_full["adjudicated_human_label"].fillna("").astype(str).str.strip()
        sample_full["reference_label_source"] = REFERENCE_LABEL_SOURCE
        sample_full[sample_full["sample_set"].eq("development")].to_csv(tables_dir / "sentiment_development_human_reference.csv", index=False)
        sample_full[sample_full["sample_set"].eq("locked_test")].to_csv(tables_dir / "sentiment_locked_test_human_reference.csv", index=False)
    else:
        human_validation_status = write_human_validation_package(sample_full, human_dir)
        sample_full["reference_label"] = sample_full["heuristic_reference_label"]
        sample_full["reference_label_source"] = REFERENCE_LABEL_SOURCE
    dev_eval = sample_full[sample_full["sample_set"].eq("development") & sample_full["reference_label"].isin(LABELS)].copy()
    hold_eval = sample_full[sample_full["sample_set"].eq("locked_test") & sample_full["reference_label"].isin(LABELS)].copy()
    if len(dev_eval) < 100 or len(hold_eval) < 100:
        raise AssertionError(f"Development and locked-test evaluable {REFERENCE_LABEL_SOURCE} labels are too sparse.")

    candidate_probs: dict[tuple[str, str, str], np.ndarray] = {}
    benchmark_rows = []
    calibration_rows = []
    per_class_rows = []
    fitted_classical: dict[str, Pipeline] = {}
    reference_col = "reference_label"
    variants = {
        "minimal_raw": "comment_text_model_minimal_raw",
        "social_normalized": "comment_text_model_social_normalized",
    }
    for variant, text_col in variants.items():
        for candidate in candidates:
            for sample_set, frame in [("development", sample_full[sample_full["sample_set"].eq("development")]), ("locked_test", sample_full[sample_full["sample_set"].eq("locked_test")])]:
                probs = predict_transformer(frame[text_col].tolist(), candidate, device, batch_size=batch_size, max_length=max_length)
                candidate_probs[(sample_set, candidate.key, variant)] = probs
                row = metric_row(frame[reference_col].tolist(), probs, candidate.key, sample_set, variant)
                row["model_name"] = candidate.model_name
                row["eligible_final"] = True
                row["reference_label_source"] = REFERENCE_LABEL_SOURCE
                row["modeling_mode"] = VALIDATION_MODE
                row["model_c_role"] = "not_applicable"
                benchmark_rows.append(row)
                calibration_rows.append({k: row[k] for k in ["candidate", "sample_set", "preprocessing_variant", "log_loss", "brier_score", "ece", "coverage"]})

        clf, dev_probs_c_eval, hold_probs_c = fit_classical_oof_and_predict(
            dev_eval[text_col].tolist(),
            dev_eval[reference_col].tolist(),
            sample_full[sample_full["sample_set"].eq("locked_test")][text_col].tolist(),
        )
        fitted_classical[variant] = clf
        dev_probs_c = np.full((len(sample_full[sample_full["sample_set"].eq("development")]), len(LABELS)), 1 / len(LABELS), dtype=float)
        dev_eval_positions = sample_full[sample_full["sample_set"].eq("development")].reset_index(drop=True).index[
            sample_full[sample_full["sample_set"].eq("development")][reference_col].isin(LABELS).to_numpy()
        ]
        dev_probs_c[dev_eval_positions, :] = dev_probs_c_eval
        candidate_probs[("development", "model_C", variant)] = dev_probs_c
        candidate_probs[("locked_test", "model_C", variant)] = hold_probs_c
        for sample_set, frame, probs in [
            ("development", sample_full[sample_full["sample_set"].eq("development")], dev_probs_c),
            ("locked_test", sample_full[sample_full["sample_set"].eq("locked_test")], hold_probs_c),
        ]:
            row = metric_row(frame[reference_col].tolist(), probs, "model_C", sample_set, variant)
            row["model_name"] = f"TF-IDF word(1-2)+char(3-6)+LinearSVC calibrated on {REFERENCE_LABEL_SOURCE} development labels"
            row["eligible_final"] = bool(HUMAN_VALIDATION_COMPLETED)
            row["reference_label_source"] = REFERENCE_LABEL_SOURCE
            row["modeling_mode"] = VALIDATION_MODE
            row["model_c_role"] = "eligible human-supervised baseline" if HUMAN_VALIDATION_COMPLETED else "pseudo-label adaptation experiment; not eligible as final without human labels"
            benchmark_rows.append(row)
            calibration_rows.append({k: row[k] for k in ["candidate", "sample_set", "preprocessing_variant", "log_loss", "brier_score", "ece", "coverage"]})

        grid_values = [0.0, 0.25, 0.5, 0.75, 1.0]
        best_ensemble = None
        for w_a in grid_values:
            for w_b in grid_values:
                for w_c in (grid_values if HUMAN_VALIDATION_COMPLETED else [0.0]):
                    if w_a + w_b + w_c <= 0:
                        continue
                    if not HUMAN_VALIDATION_COMPLETED and w_c > 0:
                        continue
                    weights = np.array([w_a, w_b, w_c], dtype=float)
                    weights = weights / weights.sum()
                    dev_probs = (
                        weights[0] * candidate_probs[("development", "model_A", variant)]
                        + weights[1] * candidate_probs[("development", "model_B", variant)]
                        + weights[2] * candidate_probs[("development", "model_C", variant)]
                    )
                    row = metric_row(sample_full[sample_full["sample_set"].eq("development")][reference_col].tolist(), dev_probs, "ensemble", "development", variant)
                    row["weights_model_A"] = float(weights[0])
                    row["weights_model_B"] = float(weights[1])
                    row["weights_model_C"] = float(weights[2])
                    if best_ensemble is None or (
                        row["macro_f1"],
                        row["min_per_class_recall"],
                        row["balanced_accuracy"],
                        -row["ece"],
                    ) > (
                        best_ensemble["row"]["macro_f1"],
                        best_ensemble["row"]["min_per_class_recall"],
                        best_ensemble["row"]["balanced_accuracy"],
                        -best_ensemble["row"]["ece"],
                    ):
                        best_ensemble = {"row": row, "weights": weights, "probs": dev_probs}
        if best_ensemble is not None:
            weights = best_ensemble["weights"]
            if HUMAN_VALIDATION_COMPLETED:
                candidate_name = f"ensemble_human_A{weights[0]:.2f}_B{weights[1]:.2f}_C{weights[2]:.2f}"
            else:
                candidate_name = f"ensemble_transformer_A{weights[0]:.2f}_B{weights[1]:.2f}"
            for sample_set in ["development", "locked_test"]:
                probs = (
                    weights[0] * candidate_probs[(sample_set, "model_A", variant)]
                    + weights[1] * candidate_probs[(sample_set, "model_B", variant)]
                    + weights[2] * candidate_probs[(sample_set, "model_C", variant)]
                )
                candidate_probs[(sample_set, candidate_name, variant)] = probs
                frame = sample_full[sample_full["sample_set"].eq(sample_set)]
                row = metric_row(frame[reference_col].tolist(), probs, candidate_name, sample_set, variant)
                row["model_name"] = f"Soft-voting ensemble selected on development {REFERENCE_LABEL_SOURCE}"
                row["eligible_final"] = True
                row["weights_model_A"] = float(weights[0])
                row["weights_model_B"] = float(weights[1])
                row["weights_model_C"] = float(weights[2])
                row["reference_label_source"] = REFERENCE_LABEL_SOURCE
                row["modeling_mode"] = VALIDATION_MODE
                row["model_c_role"] = "eligible if human validation is completed; otherwise weight remains 0"
                benchmark_rows.append(row)
                calibration_rows.append({k: row[k] for k in ["candidate", "sample_set", "preprocessing_variant", "log_loss", "brier_score", "ece", "coverage"]})

    benchmark = pd.DataFrame(benchmark_rows)
    benchmark.to_csv(tables_dir / "sentiment_model_benchmark_development.csv", index=False)
    pd.DataFrame(calibration_rows).to_csv(tables_dir / "sentiment_model_calibration_metrics.csv", index=False)
    cv_rows, cv_summary = repeated_cv_diagnostics(
        sample_full[sample_full["sample_set"].eq("development")],
        variants,
        candidate_probs,
        reference_col,
        REFERENCE_LABEL_SOURCE,
        VALIDATION_MODE,
    )
    cv_rows.to_csv(tables_dir / "sentiment_repeated_cv_metrics.csv", index=False)
    cv_summary.to_csv(tables_dir / "sentiment_repeated_cv_summary.csv", index=False)

    dev_candidates = benchmark[(benchmark["sample_set"].eq("development")) & (benchmark["eligible_final"] == True)].copy()
    if VALIDATION_MODE == "PROVISIONAL" and dev_candidates["candidate"].astype(str).str.contains("model_C|_C", regex=True).any():
        raise AssertionError("Model C must not be eligible for PROVISIONAL final selection.")
    best_macro = dev_candidates["macro_f1"].max()
    near_best = dev_candidates[dev_candidates["macro_f1"] >= best_macro - 0.01].copy()
    near_best["complexity_rank"] = near_best["candidate"].map(lambda x: 1 if x in {"model_A", "model_B"} else 2)
    near_best = near_best.sort_values(
        ["complexity_rank", "min_per_class_recall", "balanced_accuracy", "macro_f1", "ece", "coverage"],
        ascending=[True, False, False, False, True, False],
    )
    selected = near_best.iloc[0].to_dict()
    selected_candidate = selected["candidate"]
    selected_variant = selected["preprocessing_variant"]
    selected_dev_probs = candidate_probs[("development", selected_candidate, selected_variant)]
    selected_holdout_probs = candidate_probs[("locked_test", selected_candidate, selected_variant)]

    curve = selective_curve(
        sample_full[sample_full["sample_set"].eq("development")][reference_col].tolist(),
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
        selected_curve_row = eligible_curve.sort_values(
            ["macro_f1", "min_per_class_recall", "balanced_accuracy", "neutral_recall", "ece", "coverage"],
            ascending=[False, False, False, False, True, False],
        ).iloc[0].to_dict()
        selected_threshold = float(selected_curve_row["threshold"])
    curve["selected_threshold"] = curve["threshold"].eq(selected_threshold)
    curve.to_csv(tables_dir / "sentiment_selective_classification_curve.csv", index=False)

    holdout_metric = metric_row(
        sample_full[sample_full["sample_set"].eq("locked_test")][reference_col].tolist(),
        selected_holdout_probs,
        selected_candidate,
        "locked_test",
        selected_variant,
        threshold=selected_threshold,
    )
    ci_low, ci_high = bootstrap_macro_f1_ci(
        sample_full[sample_full["sample_set"].eq("locked_test")][reference_col].tolist(),
        selected_holdout_probs,
        selected_threshold,
    )
    holdout_metric["macro_f1_ci_low"] = ci_low
    holdout_metric["macro_f1_ci_high"] = ci_high
    holdout_metric["reference_label_source"] = REFERENCE_LABEL_SOURCE
    holdout_metric["modeling_mode"] = VALIDATION_MODE
    holdout_metric["human_validation_completed"] = HUMAN_VALIDATION_COMPLETED
    sentiment_model_gate_passed = bool(
        HUMAN_VALIDATION_COMPLETED
        and float(holdout_metric["macro_f1"]) >= 0.55
        and float(holdout_metric["neutral_recall"]) >= 0.50
        and float(holdout_metric["min_per_class_f1"]) >= 0.40
    )
    if HUMAN_VALIDATION_COMPLETED:
        FINAL_VALIDATION_STATUS = "VALIDATED" if sentiment_model_gate_passed else "FAILED"
    pd.DataFrame([holdout_metric]).to_csv(tables_dir / "sentiment_model_locked_test_metrics.csv", index=False)
    pd.DataFrame([holdout_metric]).to_csv(tables_dir / "sentiment_model_holdout_metrics.csv", index=False)

    hold_y = sample_full[sample_full["sample_set"].eq("locked_test")][reference_col].astype(str).to_numpy()
    hold_pred = np.array([LABELS[i] for i in selected_holdout_probs.argmax(axis=1)])
    hold_valid = np.isin(hold_y, LABELS) & (selected_holdout_probs.max(axis=1) >= selected_threshold)
    precision, recall, f1v, support = precision_recall_fscore_support(hold_y[hold_valid], hold_pred[hold_valid], labels=LABELS, zero_division=0)
    per_class = pd.DataFrame(
        {
            "sample_set": "locked_test",
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

    locked_frame = sample_full[sample_full["sample_set"].eq("locked_test")].reset_index(drop=True).copy()
    locked_frame["predicted_label"] = hold_pred
    locked_frame["selected_model_confidence"] = selected_holdout_probs.max(axis=1)
    locked_frame["included_at_threshold"] = hold_valid
    neutral_error_mask = hold_valid & (hold_y == "Neutral") & np.isin(hold_pred, ["Positive", "Negative"])
    neutral_errors = locked_frame.loc[neutral_error_mask].copy()
    if not neutral_errors.empty:
        neutral_errors["neutral_error_taxonomy"] = neutral_errors.apply(
            lambda row: infer_neutral_error_taxonomy(row["comment_text_original"], row.get("ambiguity_flags", "")),
            axis=1,
        )
    else:
        neutral_errors["neutral_error_taxonomy"] = pd.Series(dtype=str)
    neutral_common_cols = [
        "comment_id",
        "comment_text_original",
        "video_id",
        "product_category",
        reference_col,
        "predicted_label",
        "selected_model_confidence",
        "ambiguity_flags",
        "neutral_error_taxonomy",
    ]
    for predicted_label, filename in [
        ("Positive", "neutral_false_positive_to_positive.csv"),
        ("Negative", "neutral_false_positive_to_negative.csv"),
    ]:
        sub = neutral_errors.loc[neutral_errors["predicted_label"].eq(predicted_label), neutral_common_cols].copy()
        sub.to_csv(tables_dir / filename, index=False)
    if neutral_errors.empty:
        neutral_taxonomy = pd.DataFrame(columns=["neutral_error_taxonomy", "n_errors", "error_percentage", "example_comment_ids"])
    else:
        neutral_taxonomy = (
            neutral_errors.groupby("neutral_error_taxonomy", dropna=False)
            .agg(
                n_errors=("comment_id", "count"),
                example_comment_ids=("comment_id", lambda s: ";".join(s.astype(str).head(8))),
            )
            .reset_index()
        )
        neutral_taxonomy["error_percentage"] = neutral_taxonomy["n_errors"] / max(len(neutral_errors), 1) * 100
        neutral_taxonomy = neutral_taxonomy[["neutral_error_taxonomy", "n_errors", "error_percentage", "example_comment_ids"]]
    neutral_taxonomy.to_csv(tables_dir / "neutral_error_taxonomy.csv", index=False)

    neutral_metric_rows = []
    text_type_cols = [
        "question",
        "emoji_only",
        "very_short",
        "slang",
        "code_mixing",
        "negation",
        "mixed_sentiment",
        "potential_sarcasm",
        "domain_term",
    ]
    neutral_eval_frame = locked_frame.loc[hold_valid & (hold_y == "Neutral")].copy()
    for text_type in text_type_cols + ["standard"]:
        if text_type == "standard":
            mask = ~neutral_eval_frame[text_type_cols].fillna(False).astype(bool).any(axis=1)
        else:
            mask = neutral_eval_frame[text_type].fillna(False).astype(bool)
        sub = neutral_eval_frame.loc[mask]
        correct = int(sub["predicted_label"].eq("Neutral").sum()) if not sub.empty else 0
        neutral_metric_rows.append(
            {
                "text_type": text_type,
                "neutral_support": int(len(sub)),
                "neutral_correct": correct,
                "neutral_recall": float(correct / len(sub)) if len(sub) else np.nan,
                "predicted_positive": int(sub["predicted_label"].eq("Positive").sum()) if not sub.empty else 0,
                "predicted_negative": int(sub["predicted_label"].eq("Negative").sum()) if not sub.empty else 0,
                "reference_label_source": REFERENCE_LABEL_SOURCE,
                "notes": f"Neutral metrics use locked-test {REFERENCE_LABEL_SOURCE} labels.",
            }
        )
    pd.DataFrame(neutral_metric_rows).to_csv(tables_dir / "neutral_class_metrics_by_text_type.csv", index=False)

    error_rows = []
    for candidate_name in sorted({k[1] for k in candidate_probs if k[0] == "locked_test"}):
        for variant in variants:
            key = ("locked_test", candidate_name, variant)
            if key not in candidate_probs:
                continue
            probs = candidate_probs[key]
            pred = np.array([LABELS[i] for i in probs.argmax(axis=1)])
            hframe = sample_full[sample_full["sample_set"].eq("locked_test")].reset_index(drop=True)
            for row_idx, row in hframe.iterrows():
                true_label = row[reference_col]
                if true_label not in LABELS or pred[row_idx] == true_label:
                    continue
                taxonomy = infer_error_taxonomy(row["comment_text_original"], true_label, pred[row_idx], row.get("ambiguity_flags", ""))
                error_rows.append(
                    {
                        "candidate": candidate_name,
                        "preprocessing_variant": variant,
                        "sample_set": "locked_test",
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
    elif selected_candidate == "model_C":
        selected_weights["model_C"] = 1.0
    elif selected_candidate.startswith("ensemble_"):
        selected_row = dev_candidates[dev_candidates["candidate"].eq(selected_candidate) & dev_candidates["preprocessing_variant"].eq(selected_variant)].iloc[0]
        selected_weights = {
            "model_A": float(selected_row.get("weights_model_A", 0.0)),
            "model_B": float(selected_row.get("weights_model_B", 0.0)),
            "model_C": float(selected_row.get("weights_model_C", 0.0)),
        }
    if sum(selected_weights[k] for k in ["model_A", "model_B", "model_C"]) <= 0:
        raise AssertionError("Final pipeline must include at least one eligible model component.")
    if VALIDATION_MODE == "PROVISIONAL" and selected_weights.get("model_C", 0.0) > 0:
        raise AssertionError("Model C is a pseudo-label adaptation experiment and cannot be part of the PROVISIONAL final model.")
    if ALLOW_RULE_BASED_FINAL:
        raise AssertionError("ALLOW_RULE_BASED_FINAL must remain False.")

    model_revision_map = {candidate.key: candidate.model_revision for candidate in candidates}
    tokenizer_revision_map = {candidate.key: candidate.tokenizer_revision for candidate in candidates}
    selection_payload = {
        "selected_candidate": selected_candidate,
        "selected_preprocessing_variant": selected_variant,
        "validation_mode": VALIDATION_MODE,
        "final_validation_status": FINAL_VALIDATION_STATUS,
        "human_validation_completed": HUMAN_VALIDATION_COMPLETED,
        "reference_label_source": REFERENCE_LABEL_SOURCE,
        "selection_rule": (
            "HUMAN_VALIDATED mode: choose among Model A, Model B, Model C, and soft-voting ensembles using development human-adjudicated labels; "
            "locked test is evaluated once after selection."
            if HUMAN_VALIDATION_COMPLETED
            else "PROVISIONAL mode: choose among Model A, Model B, and transformer-only ensembles using development heuristic pseudo-labels; prioritize macro-F1, minimum per-class recall, balanced accuracy, calibration, and coverage. Model C is diagnostic only until human labels exist."
        ),
        "confidence_threshold": selected_threshold,
        "coverage_floor": 0.90,
        "ensemble_weights": selected_weights,
        "model_revisions": model_revision_map,
        "tokenizer_revisions": tokenizer_revision_map,
        "allow_rule_based_final": ALLOW_RULE_BASED_FINAL,
        "locked_test_not_used_for_selection": True,
        "previous_validation_ids_excluded_from_locked_test": len(previously_seen_validation_ids),
        "model_c_status": "eligible human-supervised baseline" if HUMAN_VALIDATION_COMPLETED else "pseudo-label adaptation experiment; excluded from PROVISIONAL final selection",
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
                "goal_orientation_status": "Insufficient Text" if goal == "Insufficient Text" else "Assigned",
                "goal_confidence": boot["goal_confidence"],
                "goal_stability": boot["goal_stability"],
                "goal_validation_status": (
                    "Human-validated sentiment model applied"
                    if FINAL_VALIDATION_STATUS == "VALIDATED"
                    else ("Human validation completed; sentiment model failed validation gate" if HUMAN_VALIDATION_COMPLETED else "Provisional")
                ),
                "goal_method": "hard_label_ratios_with_bootstrap_stability",
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
    goal_method_rows = []
    for hcc_id, grp in comment_sentiment[comment_sentiment["is_hcc"]].groupby("hcc_id"):
        base_counts = hard_counts(grp)
        n_valid = base_counts["n_valid_text_comments"]
        methods = [
            (
                "hard_label_ratios",
                base_counts["positive_ratio"],
                base_counts["neutral_ratio"],
                base_counts["negative_ratio"],
                n_valid,
                base_counts["evaluable_coverage"],
            )
        ]
        soft = soft_counts(grp)
        methods.append(
            (
                "soft_probability_mass",
                soft["soft_positive_share"],
                soft["soft_neutral_share"],
                soft["soft_negative_share"],
                soft["soft_denominator"],
                base_counts["evaluable_coverage"],
            )
        )
        prob_frame = grp.loc[~grp["sentiment_status"].eq("No Text")].copy()
        if prob_frame.empty:
            cw_pos = cw_neu = cw_neg = 0.0
            cw_n = 0
        else:
            weights = pd.to_numeric(prob_frame["prediction_confidence"], errors="coerce").fillna(0.0).to_numpy()
            if weights.sum() <= 0:
                weights = np.ones(len(prob_frame))
            cw_pos = float(np.average(prob_frame["probability_positive"], weights=weights))
            cw_neu = float(np.average(prob_frame["probability_neutral"], weights=weights))
            cw_neg = float(np.average(prob_frame["probability_negative"], weights=weights))
            total_cw = cw_pos + cw_neu + cw_neg
            cw_pos, cw_neu, cw_neg = cw_pos / total_cw, cw_neu / total_cw, cw_neg / total_cw
            cw_n = int(len(prob_frame))
        methods.append(("confidence_weighted_probability_mass", cw_pos, cw_neu, cw_neg, cw_n, base_counts["evaluable_coverage"]))
        hard_label_counts = grp.loc[grp["sentiment_status"].eq("Evaluable"), "sentiment_label_final"].value_counts().reindex(LABELS, fill_value=0)
        smoothed = (hard_label_counts + 1) / (hard_label_counts.sum() + 3) if hard_label_counts.sum() else pd.Series([1 / 3, 1 / 3, 1 / 3], index=LABELS)
        methods.append(
            (
                "dirichlet_smoothed_hard_labels_alpha1",
                float(smoothed["Positive"]),
                float(smoothed["Neutral"]),
                float(smoothed["Negative"]),
                int(hard_label_counts.sum()),
                base_counts["evaluable_coverage"],
            )
        )
        final_goal = hcc_summary.loc[hcc_summary["hcc_id"].astype(str).eq(str(hcc_id)), "goal_orientation"].iloc[0]
        for method, pos_r, neu_r, neg_r, effective_n, coverage in methods:
            method_goal = classify_goal(pos_r, neu_r, neg_r, int(effective_n), coverage, float(effective_n))
            goal_method_rows.append(
                {
                    "hcc_id": hcc_id,
                    "goal_method": method,
                    "positive_share": pos_r,
                    "neutral_share": neu_r,
                    "negative_share": neg_r,
                    "effective_sample_size": effective_n,
                    "evaluable_coverage": coverage,
                    "method_goal_orientation": method_goal,
                    "matches_selected_goal": method_goal == final_goal,
                    "selected_goal_method": "hard_label_ratios_with_bootstrap_stability",
                    "selection_basis": (
                        "Reported as sensitivity; selected goal method uses model labels from the human-validated sentiment pipeline."
                        if FINAL_VALIDATION_STATUS == "VALIDATED"
                        else "Reported as sensitivity only; downstream goals should not be treated as validated unless sentiment validation gates pass."
                    ),
                }
            )
    pd.DataFrame(goal_method_rows).to_csv(tables_dir / "hcc_goal_method_sensitivity.csv", index=False)
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
        heuristic_labels = []
        reasons = []
        for text in rep["comment_text_original"].tolist():
            l1, _, _ = heuristic_label_for_text(text, pass_id=1)
            l2, _, _ = heuristic_label_for_text(text, pass_id=2)
            lab, reason = adjudicate_heuristic_labels(text, l1, l2)
            heuristic_labels.append(lab)
            reasons.append(reason)
        eval_labels = [lab for lab in heuristic_labels if lab in LABELS]
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
                "heuristic_hcc_goal_review": review_goal,
                "heuristic_hcc_goal_reason": f"Deterministic lexical reference over representative comments: pos={counts['Positive']}, neu={counts['Neutral']}, neg={counts['Negative']}, non-evaluable={len(rep)-n_eval}.",
                "heuristic_hcc_review_confidence": "High" if n_eval >= 10 and review_goal != "Insufficient Text" else ("Medium" if n_eval >= 5 else "Low"),
                "observed_message_orientation": observed_orientation,
                "ambiguity_notes": ";".join(sorted(set(flag for text in rep["comment_text_original"] for _, flags, _ in [heuristic_label_for_text(text, pass_id=1)] for flag in flags))),
                "algorithmic_goal_orientation": alg_goal,
                "algorithmic_goal_confidence": alg_conf,
                "exact_match": review_goal == alg_goal,
                "notes": "Heuristic HCC review is an internal diagnostic, not human gold-standard validation.",
            }
        )
    hcc_review = pd.DataFrame(hcc_review_rows)
    hcc_review.to_csv(tables_dir / "hcc_goal_heuristic_review.csv", index=False)

    goal_labels = ["Promotional / Supportive", "Critical / Complaint", "Neutral Engagement", "Polarized / Contested", "Mixed Goals", "Insufficient Text"]
    goal_exact_agreement = float(hcc_review["exact_match"].mean())
    goal_weighted_kappa = float(
        cohen_kappa_score(
            hcc_review["heuristic_hcc_goal_review"],
            hcc_review["algorithmic_goal_orientation"],
            labels=goal_labels,
            weights="linear",
        )
    )

    def goal_metric_status(metric: str, value: float) -> str:
        if metric == "exact_agreement":
            if value < 0.60:
                return "FAIL"
            if value < 0.75:
                return "WARNING"
            return "PASS"
        if value < 0.40:
            return "FAIL"
        if value < 0.60:
            return "WARNING"
        return "PASS"

    review_metrics = pd.DataFrame(
        [
            {
                "metric": "exact_agreement",
                "value": goal_exact_agreement,
                "n_hcc": len(hcc_review),
                "status": goal_metric_status("exact_agreement", goal_exact_agreement),
                "notes": "Agreement between algorithmic goal_orientation and heuristic HCC-level diagnostic review. This is not human validation.",
            },
            {
                "metric": "weighted_kappa_linear",
                "value": goal_weighted_kappa,
                "n_hcc": len(hcc_review),
                "status": goal_metric_status("weighted_kappa_linear", goal_weighted_kappa),
                "notes": "Heuristic HCC review is not a human gold standard.",
            },
        ]
    )
    review_metrics.to_csv(tables_dir / "hcc_goal_validation_metrics.csv", index=False)
    disagreement = (
        hcc_review.loc[~hcc_review["exact_match"]]
        .merge(
            hcc_summary[
                [
                    "hcc_id",
                    "positive_ratio",
                    "neutral_ratio",
                    "negative_ratio",
                    "soft_positive_share",
                    "soft_neutral_share",
                    "soft_negative_share",
                    "evaluable_coverage",
                    "n_valid_text_comments",
                    "goal_stability",
                    "goal_confidence",
                ]
            ],
            on="hcc_id",
            how="left",
        )
        .rename(
            columns={
                "algorithmic_goal_orientation": "algorithmic_goal",
                "heuristic_hcc_goal_review": "semantic_review_goal",
                "n_valid_text_comments": "n_valid",
                "heuristic_hcc_goal_reason": "disagreement_reason",
            }
        )
    )
    if not disagreement.empty:
        disagreement["suggested_action"] = (
            "Requires targeted HCC-level human review before goal counts are treated as directly validated."
            if HUMAN_VALIDATION_COMPLETED
            else "Requires human validation before goal counts can be treated as validated."
        )
    disagreement[
        [
            "hcc_id",
            "algorithmic_goal",
            "semantic_review_goal",
            "positive_ratio",
            "neutral_ratio",
            "negative_ratio",
            "soft_positive_share",
            "soft_neutral_share",
            "soft_negative_share",
            "evaluable_coverage",
            "n_valid",
            "goal_stability",
            "goal_confidence",
            "disagreement_reason",
            "suggested_action",
        ]
    ].to_csv(tables_dir / "hcc_goal_disagreement_analysis.csv", index=False)

    cm_title = (
        f"Human-reference evaluation (locked test evaluable n={int(hold_valid.sum())})"
        if HUMAN_VALIDATION_COMPLETED
        else f"Heuristic-reference evaluation (locked test evaluable n={int(hold_valid.sum())})"
    )
    save_confusion_matrix_png(cm_df, vis_dir / "sentiment_validation_confusion_matrix.png", cm_title)
    plot_hcc_vs_nonhcc(hcc_vs_nonhcc, vis_dir / "sentiment_hcc_vs_nonhcc_100pct.png")
    plot_goal_confidence(hcc_summary, vis_dir / "hcc_goal_orientation_confidence.png", FINAL_VALIDATION_STATUS)

    hcc_sent_cols = [
        "hcc_id",
        "dominant_sentiment",
        "positive_ratio",
        "neutral_ratio",
        "negative_ratio",
        "goal_orientation",
        "goal_orientation_status",
        "goal_confidence",
        "goal_stability",
        "goal_validation_status",
        "goal_method",
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
    human_validation_completed = bool(
        human_validation_status.loc[
            human_validation_status["metric"].eq("human_validation_completed"),
            "value",
        ].astype(str).str.lower().eq("true").any()
    )

    def status_from_bool(ok: bool) -> str:
        return "PASS" if bool(ok) else "FAIL"

    def status_from_threshold(value: float, pass_at: float, warning_at: float | None = None) -> str:
        if pd.isna(value):
            return "NOT_AVAILABLE"
        if value >= pass_at:
            return "PASS"
        if warning_at is not None and value >= warning_at:
            return "WARNING"
        return "FAIL"

    macro_status = status_from_threshold(float(holdout_metric["macro_f1"]), 0.55, 0.45)
    neutral_recall_status = status_from_threshold(float(holdout_metric["neutral_recall"]), 0.50, 0.35)
    min_class_f1_status = status_from_threshold(float(holdout_metric["min_per_class_f1"]), 0.40, 0.30)
    calibration_status = status_from_threshold(0.25 - float(holdout_metric["ece"]), 0.0, -0.10)
    goal_exact_status = review_metrics.loc[review_metrics["metric"].eq("exact_agreement"), "status"].iloc[0]
    goal_kappa_status = review_metrics.loc[review_metrics["metric"].eq("weighted_kappa_linear"), "status"].iloc[0]
    hcc_goal_confidence_status = "WARNING" if goal_conf_counts.get("Low", 0) > 0 or goal_conf_counts.get("None", 0) > 0 else "PASS"

    final_report_records = []

    def add_report(section: str, metric: str, value, status: str, notes: str = "") -> None:
        final_report_records.append(
            {
                "overall_pipeline_status": FINAL_VALIDATION_STATUS,
                "section": section,
                "metric": metric,
                "value": value,
                "status": status,
                "passed": status == "PASS",
                "notes": notes,
            }
        )

    overall_report_status = {"VALIDATED": "PASS", "PROVISIONAL": "WARNING", "FAILED": "FAIL"}.get(FINAL_VALIDATION_STATUS, "WARNING")
    reference_notes = (
        "Reference labels are adjudicated human labels."
        if HUMAN_VALIDATION_COMPLETED
        else "Reference labels are deterministic heuristic pseudo-labels."
    )
    reference_report_status = "PASS" if HUMAN_VALIDATION_COMPLETED else "WARNING"
    add_report("SUMMARY", "overall_pipeline_status", FINAL_VALIDATION_STATUS, overall_report_status, reference_notes)
    add_report("SUMMARY", "validation_mode", VALIDATION_MODE, reference_report_status, reference_notes)
    add_report("DATA", "dataset rows", len(comment_sentiment), status_from_bool(len(comment_sentiment) == TOTAL_COMMENTS_EXPECTED), "")
    add_report("DATA", "unique comment IDs", comment_sentiment["comment_id"].nunique(), status_from_bool(comment_sentiment["comment_id"].nunique() == TOTAL_COMMENTS_EXPECTED), "")
    add_report("DATA", "duplicate rows removed", duplicate_rows_removed, "PASS", "")
    add_report("DATA", "blank text", int(comments["blank_text"].sum()), "PASS", "")
    add_report("DATA", "emoji-only", int(comments["emoji_only"].sum()), "PASS", "")
    add_report("DATA", "evaluable", int(comment_sentiment["sentiment_status"].eq("Evaluable").sum()), "PASS", "")
    add_report("DATA", "uncertain", int(comment_sentiment["sentiment_status"].eq("Uncertain").sum()), "PASS", "")
    add_report("DATA", "no text", int(comment_sentiment["sentiment_status"].eq("No Text").sum()), "PASS", "")
    add_report("DATA", "coverage", float(comment_sentiment["sentiment_status"].eq("Evaluable").mean()), "PASS", "")
    add_report("MODEL", "selected model/pipeline", selected_candidate, macro_status, f"Selected on development {REFERENCE_LABEL_SOURCE}; locked test evaluated once after selection.")
    add_report("MODEL", "model revision", json.dumps(model_revision_map, ensure_ascii=False), status_from_bool(all(model_revision_map.values())), "")
    add_report("MODEL", "preprocessing", selected_variant, "PASS", "")
    add_report("MODEL", "calibration", "Transformer probabilities are unmodified; Model C uses calibrated LinearSVC probabilities when selected.", "WARNING", "ECE is reported separately and remains a model-quality gate.")
    add_report("MODEL", "ensemble weights", json.dumps(selected_weights), "PASS" if (HUMAN_VALIDATION_COMPLETED or selected_weights.get("model_C", 0.0) == 0.0) else "FAIL", "Model C is eligible only when human validation is completed.")
    add_report("MODEL", "confidence threshold", selected_threshold, reference_report_status, f"Threshold selected on development {REFERENCE_LABEL_SOURCE}.")
    add_report("MODEL", "fallback used", False, "PASS", "")
    add_report("MODEL", "human_validation_completed", human_validation_completed, "PASS" if human_validation_completed else "NOT_AVAILABLE", "Human labels are loaded from the completed validation file." if human_validation_completed else "Human labels remain blank in the generated package.")
    add_report("MODEL", "model generalization", holdout_metric["macro_f1"], macro_status, "Validated status requires locked-test macro-F1 >= 0.55 on human labels.")
    add_report("MODEL", "per-class performance minimum F1", holdout_metric["min_per_class_f1"], min_class_f1_status, "Validated status requires every class F1 >= 0.40.")
    add_report("MODEL", "Neutral recall", holdout_metric["neutral_recall"], neutral_recall_status, "Validated status requires Neutral recall >= 0.50.")
    add_report("MODEL", "calibration ECE", holdout_metric["ece"], calibration_status, "Lower ECE is better.")
    add_report("DEVELOPMENT", "macro-F1", float(selected["macro_f1"]), reference_report_status, f"Measured against {REFERENCE_LABEL_SOURCE}.")
    add_report("DEVELOPMENT", "accuracy", float(selected["accuracy"]), reference_report_status, f"Measured against {REFERENCE_LABEL_SOURCE}.")
    add_report("DEVELOPMENT", "balanced accuracy", float(selected["balanced_accuracy"]), reference_report_status, f"Measured against {REFERENCE_LABEL_SOURCE}.")
    add_report("DEVELOPMENT", "negative recall", float(selected["negative_recall"]), reference_report_status, f"Measured against {REFERENCE_LABEL_SOURCE}.")
    add_report("DEVELOPMENT", "calibration ECE", float(selected["ece"]), "WARNING", f"Measured against {REFERENCE_LABEL_SOURCE}.")
    add_report("LOCKED_TEST", "macro-F1", holdout_metric["macro_f1"], macro_status, f"{REFERENCE_LABEL_SOURCE} locked test.")
    add_report("LOCKED_TEST", "bootstrap 95% CI", f"{ci_low:.4f}-{ci_high:.4f}", macro_status, f"Bootstrap over {REFERENCE_LABEL_SOURCE} locked test.")
    add_report("LOCKED_TEST", "accuracy", holdout_metric["accuracy"], reference_report_status, f"{REFERENCE_LABEL_SOURCE} locked test.")
    add_report("LOCKED_TEST", "balanced accuracy", holdout_metric["balanced_accuracy"], reference_report_status, f"{REFERENCE_LABEL_SOURCE} locked test.")
    add_report("LOCKED_TEST", "weighted F1", holdout_metric["weighted_f1"], reference_report_status, f"{REFERENCE_LABEL_SOURCE} locked test.")
    add_report("LOCKED_TEST", "MCC", holdout_metric["mcc"], reference_report_status, f"{REFERENCE_LABEL_SOURCE} locked test.")
    add_report("LOCKED_TEST", "Brier score", holdout_metric["brier_score"], reference_report_status, f"{REFERENCE_LABEL_SOURCE} locked test.")
    add_report("LOCKED_TEST", "ECE", holdout_metric["ece"], calibration_status, f"{REFERENCE_LABEL_SOURCE} locked test.")
    add_report("LOCKED_TEST", "coverage", holdout_metric["coverage"], "PASS" if holdout_metric["coverage"] >= 0.90 else "WARNING", "")
    add_report("GOALS", "HCC count", len(hcc_summary), status_from_bool(len(hcc_summary) == HCC_COUNT_EXPECTED), "")
    add_report("GOALS", "goal counts", json.dumps(goal_counts, ensure_ascii=False), "PASS" if FINAL_VALIDATION_STATUS == "VALIDATED" else "WARNING", "Goal counts are derived from model sentiment labels; confidence/stability is not accuracy.")
    add_report("GOALS", "goal confidence counts", json.dumps(goal_conf_counts, ensure_ascii=False), hcc_goal_confidence_status, "Confidence is bootstrap stability, not correctness.")
    add_report("GOALS", "heuristic HCC review exact agreement", goal_exact_agreement, goal_exact_status, "Exact agreement <0.60 fails the diagnostic gate.")
    add_report("GOALS", "heuristic HCC review weighted kappa", goal_weighted_kappa, goal_kappa_status, "Weighted kappa <0.40 fails the diagnostic gate.")
    add_report("GOALS", "low-confidence HCC list", ";".join(low_conf_hcc), hcc_goal_confidence_status, "")
    add_report("GOALS", "insufficient HCC count", int(hcc_summary["goal_orientation"].eq("Insufficient Text").sum()), "PASS", "")
    add_report("INTEGRITY", "RM1 checksums unchanged", rm1_unchanged, status_from_bool(rm1_unchanged), "")
    add_report("INTEGRITY", "actor type counts unchanged", actor_counts_ok, status_from_bool(actor_counts_ok), json.dumps(actor_counts, ensure_ascii=False))
    add_report("INTEGRITY", "Gephi aggregate 396/497 unchanged", gephi_aggregate_ok, status_from_bool(gephi_aggregate_ok), "")
    add_report("INTEGRITY", "no Non-HCC artifact", no_non_hcc_artifact, status_from_bool(no_non_hcc_artifact), "")
    add_report("INTEGRITY", "sentiment notebook reached final validation cell", True, "PASS", "")
    add_report("INTEGRITY", "comment_sentiment sha256", comment_sentiment_checksum, "PASS", "")
    final_report = pd.DataFrame(final_report_records)
    final_report.to_csv(tables_dir / "sentiment_final_validation_report.csv", index=False)
    critical_integrity = final_report[
        final_report["section"].eq("INTEGRITY")
        | (final_report["section"].eq("DATA") & final_report["metric"].isin(["dataset rows", "unique comment IDs"]))
        | (final_report["section"].eq("MODEL") & final_report["metric"].eq("fallback used"))
    ]
    if critical_integrity["status"].eq("FAIL").any():
        raise AssertionError("Critical integrity validation failed.")

    print("RM2 SENTIMENT GOALS PIPELINE COMPLETE")
    print(f"- selected pipeline: {selected_candidate} ({selected_variant})")
    print(f"- confidence threshold: {selected_threshold:.3f}")
    print(f"- comment rows: {len(comment_sentiment):,}")
    print(f"- HCC comments: {int(comment_sentiment['is_hcc'].sum()):,}")
    print(f"- HCC count: {len(hcc_summary):,}")
    print(f"- overall pipeline status: {FINAL_VALIDATION_STATUS}")
    print(f"- human validation completed: {human_validation_completed}")
    print("- development macro-F1: {:.4f}".format(float(selected["macro_f1"])))
    print("- locked-test {} macro-F1: {:.4f} (95% CI {:.4f}-{:.4f})".format(REFERENCE_LABEL_SOURCE, float(holdout_metric["macro_f1"]), ci_low, ci_high))
    print("- sentiment status counts:")
    print(comment_sentiment["sentiment_status"].value_counts().to_string())
    print("- HCC goal counts:")
    print(hcc_summary["goal_orientation"].value_counts().to_string())
    print("- HCC goal confidence counts:")
    print(hcc_summary["goal_confidence"].value_counts().to_string())
    print("- deterministic rule reproducibility (not annotation reliability):")
    print(consistency.to_string(index=False))
    print("- RM1 protected inputs unchanged:", rm1_unchanged)

    return {
        "selected_candidate": selected_candidate,
        "selected_variant": selected_variant,
        "selected_threshold": selected_threshold,
        "overall_pipeline_status": FINAL_VALIDATION_STATUS,
        "human_validation_completed": human_validation_completed,
        "development_macro_f1": float(selected["macro_f1"]),
        "locked_test_macro_f1": float(holdout_metric["macro_f1"]),
        "locked_test_macro_f1_ci": [ci_low, ci_high],
        "locked_test_neutral_recall": float(holdout_metric["neutral_recall"]),
        "comment_rows": len(comment_sentiment),
        "hcc_comments": int(comment_sentiment["is_hcc"].sum()),
        "hcc_goal_counts": goal_counts,
        "goal_confidence_counts": goal_conf_counts,
        "low_confidence_hcc": low_conf_hcc,
        "rm1_unchanged": rm1_unchanged,
    }
