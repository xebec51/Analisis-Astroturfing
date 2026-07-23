from __future__ import annotations

import hashlib
import math
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SELECTION_SEED = 20260720
BASELINE_COMMIT = "cdd87d240fb70ec5167f3465dbdf8445a494e2a5"

DATASET_PATH = ROOT / "dataset.csv"
COMMENT_SENTIMENT_PATH = ROOT / "output/rm2_sentiment/tables/comment_sentiment.csv"
V1_VALIDATED_PATH = ROOT / "output/rm2_sentiment/human_validation/sentiment_human_annotation_validated.csv"
HCC_NODES_PATH = ROOT / "output/gephi/gephi_hcc_nodes.csv"
ACTOR_TYPE_PATH = ROOT / "output/rm2_actor_type/tables/account_actor_type.csv"
GEPHI_ACTOR_NODES_PATH = ROOT / "output/rm2_actor_type/gephi/gephi_actor_type_nodes.csv"
GEPHI_ACTOR_EDGES_PATH = ROOT / "output/rm2_actor_type/gephi/gephi_actor_type_edges.csv"
PIPELINE_SCRIPT_PATH = ROOT / "scripts/rm2_sentiment_goals_pipeline.py"

OUT_DIR = ROOT / "output/rm2_sentiment/human_validation_v2"

ALLOWED_SENTIMENT = ["Positive", "Neutral", "Negative", "Uncertain", "No Text"]
ALLOWED_TARGET = [
    "Product / Brand",
    "Skin condition",
    "Usage question",
    "Creator / Seller",
    "Price / Purchase",
    "Promotion / CTA",
    "General discussion",
    "Other / unclear",
]
ALLOWED_SCOPE = [
    "product_effect",
    "skin_condition",
    "price_value",
    "safety_concern",
    "authenticity_concern",
    "usage_confusion",
    "not_applicable",
    "unclear",
]

MAIN_BRANDS = ["Azarine", "Daviena", "Maryame", "The Originote"]


QUESTION_RE = re.compile(
    r"(?:\?|apa|apakah|gimana|bagaimana|boleh|bisa|cocok\\s*(?:ga|gak|nggak|ngga)?|"
    r"aman\\s*(?:ga|gak|nggak|ngga)?|untuk kulit|cara pakai|dipakai|boleh pakai)",
    re.IGNORECASE,
)
EMOJI_RE = re.compile(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]")
SKIN_RE = re.compile(
    r"\b(?:jerawat|bruntusan|beruntusan|kusam|flek|bekas jerawat|sensitif|kering|berminyak|"
    r"iritasi|perih|gatal|kemerahan|breakout|purging|skin barrier|komedo|pori|bopeng)\b",
    re.IGNORECASE,
)
POSITIVE_EFFECT_RE = re.compile(
    r"\b(?:cocok|bagus|mantap|ampuh|worth|rekomendasi|recommended|cerah|glowing|mulus|"
    r"membaik|hilang|pudar|suka|love|hasilnya|efektif|aman di aku)\b",
    re.IGNORECASE,
)
NEGATIVE_SIGNAL_RE = re.compile(
    r"\b(?:gak cocok|ga cocok|nggak cocok|tidak cocok|buruk|parah|makin|tambah|iritasi|"
    r"perih|gatal|merah|breakout|purging|menyesal|zonk|mahal|bahaya|palsu|rugi)\b",
    re.IGNORECASE,
)
PURCHASE_RE = re.compile(
    r"\b(?:beli|checkout|co|keranjang|link|harga|diskon|promo|order|cod|shopee|tokopedia|"
    r"affiliate|voucher|gratis ongkir)\b",
    re.IGNORECASE,
)
PROMOTION_RE = re.compile(
    r"\b(?:spill|buruan|wajib coba|cobain|aku rekomendasi|rekomendasi banget|racun|"
    r"best seller|terlaris|jangan lupa)\b",
    re.IGNORECASE,
)
NEGATION_RE = re.compile(r"\b(?:tidak|gak|ga|nggak|ngga|jangan|belum|bukan|tanpa|kurang)\b", re.IGNORECASE)
SLANG_RE = re.compile(
    r"\b(?:ga|gak|nggak|ngga|nih|dong|kak|sis|bund|spill|racun|bestie|wkwk|btw|yg|dgn|"
    r"bgt|banget|plis|cmiiw|auto|gue|gw|aku)\b",
    re.IGNORECASE,
)
CODE_MIX_RE = re.compile(
    r"\b(?:retinol|niacinamide|serum|skincare|glowing|brightening|acne|dark spot|breakout|"
    r"review|claim|skin barrier|moisturizer|sunscreen|tone up|exfoliating)\b",
    re.IGNORECASE,
)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, dtype=str, low_memory=False).fillna("")


def normalize_id(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null", "<na>"}:
        return ""
    return text


def normalize_bool(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def contains_non_hcc_artifact(value: object) -> bool:
    text = normalize_id(value).lower()
    compact = re.sub(r"[\s_\-]+", "", text)
    return "hccnonhcc" in compact or "masshccnonhcc" in compact


def text_or_fallback(frame: pd.DataFrame) -> pd.Series:
    text = frame.get("comment_text_original", pd.Series("", index=frame.index)).astype(str)
    fallback = frame.get("text_raw", pd.Series("", index=frame.index)).astype(str)
    return text.where(text.str.strip().ne(""), fallback)


def add_text_features(frame: pd.DataFrame) -> pd.DataFrame:
    df = frame.copy()
    df["comment_id"] = df["comment_id"].map(normalize_id)
    df["comment_text_original"] = text_or_fallback(df).map(lambda x: str(x).strip())
    lower = df["comment_text_original"].str.lower()
    df["char_count"] = df["comment_text_original"].str.len()
    df["word_count"] = df["comment_text_original"].str.findall(r"\S+").str.len()
    df["no_text_flag"] = df["comment_text_original"].str.strip().eq("")
    df["is_question"] = lower.str.contains(QUESTION_RE, regex=True, na=False)
    df["has_emoji"] = df["comment_text_original"].str.contains(EMOJI_RE, regex=True, na=False)
    df["emoji_only"] = df["has_emoji"] & lower.str.replace(EMOJI_RE, "", regex=True).str.strip().eq("")
    df["has_skin_condition"] = lower.str.contains(SKIN_RE, regex=True, na=False)
    df["has_positive_effect"] = lower.str.contains(POSITIVE_EFFECT_RE, regex=True, na=False)
    df["has_negative_signal"] = lower.str.contains(NEGATIVE_SIGNAL_RE, regex=True, na=False)
    df["has_purchase_cta"] = lower.str.contains(PURCHASE_RE, regex=True, na=False)
    df["has_promotion"] = lower.str.contains(PROMOTION_RE, regex=True, na=False)
    df["has_negation"] = lower.str.contains(NEGATION_RE, regex=True, na=False)
    df["has_slang"] = lower.str.contains(SLANG_RE, regex=True, na=False)
    df["has_code_mixing"] = lower.str.contains(CODE_MIX_RE, regex=True, na=False)
    df["is_short_text"] = (df["word_count"] <= 3) & ~df["no_text_flag"]
    df["mixed_signal"] = (
        df.get("mixed_sentiment_flag", pd.Series("", index=df.index)).map(normalize_bool)
        | (df["has_positive_effect"] & df["has_negative_signal"])
        | (df["has_negation"] & (df["has_positive_effect"] | df["has_negative_signal"]))
    )

    def length_bin(row: pd.Series) -> str:
        if row["no_text_flag"]:
            return "no_text"
        words = int(row["word_count"])
        if words <= 3:
            return "very_short"
        if words <= 8:
            return "short"
        if words <= 20:
            return "medium"
        return "long"

    def text_type(row: pd.Series) -> str:
        if row["no_text_flag"]:
            return "no_text"
        if row["emoji_only"]:
            return "emoji_only"
        if row["is_question"]:
            return "question"
        if row["mixed_signal"]:
            return "mixed_or_negation"
        if row["has_skin_condition"] and not row["has_positive_effect"]:
            return "skin_condition"
        if row["has_purchase_cta"] or row["has_promotion"]:
            return "purchase_or_cta"
        if row["is_short_text"]:
            return "short_text"
        if row["has_slang"] or row["has_code_mixing"]:
            return "slang_or_code_mixing"
        return "general"

    df["length_bin"] = df.apply(length_bin, axis=1)
    df["text_type_major"] = df.apply(text_type, axis=1)
    df["actor_segment"] = df.get("is_hcc_member", pd.Series("", index=df.index)).map(normalize_bool).map(
        {True: "HCC", False: "Non-HCC"}
    )
    df["brand_or_video_context"] = df.get("product_brand_context", pd.Series("", index=df.index)).astype(str).str.strip()
    df.loc[df["brand_or_video_context"].eq(""), "brand_or_video_context"] = df.get(
        "product_category", pd.Series("", index=df.index)
    ).astype(str)
    df["baseline_prediction"] = df.get("sentiment_label_final", pd.Series("", index=df.index)).astype(str)
    df.loc[df["baseline_prediction"].eq(""), "baseline_prediction"] = "No Text"
    df["prediction_confidence_numeric"] = pd.to_numeric(
        df.get("prediction_confidence", pd.Series(0.0, index=df.index)), errors="coerce"
    ).fillna(0.0)
    df["top2_margin_numeric"] = pd.to_numeric(
        df.get("top2_margin", pd.Series(0.0, index=df.index)), errors="coerce"
    ).fillna(0.0)
    df["random_key"] = np.random.default_rng(SELECTION_SEED).random(len(df))
    df["error_focus_score"] = (
        (1.0 - df["prediction_confidence_numeric"]).clip(lower=0, upper=1) * 2.0
        + (1.0 - df["top2_margin_numeric"]).clip(lower=0, upper=1) * 1.5
        + df["is_question"].astype(float) * 1.1
        + df["has_skin_condition"].astype(float) * 1.0
        + df["mixed_signal"].astype(float) * 1.0
        + df["has_negation"].astype(float) * 0.8
        + df["has_positive_effect"].astype(float) * 0.7
        + df["has_negative_signal"].astype(float) * 0.7
        + df["has_purchase_cta"].astype(float) * 0.5
        + df["has_emoji"].astype(float) * 0.3
        + df["is_short_text"].astype(float) * 0.3
        + df["actor_segment"].eq("HCC").astype(float) * 0.6
    )
    return df


def select_development_v2(eligible: pd.DataFrame, n: int = 300) -> pd.DataFrame:
    selected_ids: set[str] = set()
    pieces: list[pd.DataFrame] = []

    def choose(reason: str, mask: pd.Series, quota: int) -> None:
        nonlocal pieces, selected_ids
        candidates = eligible.loc[mask & ~eligible["comment_id"].isin(selected_ids)].copy()
        if candidates.empty:
            return
        candidates = candidates.sort_values(
            ["error_focus_score", "prediction_confidence_numeric", "top2_margin_numeric", "random_key", "comment_id"],
            ascending=[False, True, True, True, True],
        ).head(quota)
        candidates["sample_role"] = "development_v2"
        candidates["sampling_strategy"] = "active_learning_error_focused"
        candidates["sampling_reason"] = reason
        pieces.append(candidates)
        selected_ids.update(candidates["comment_id"])

    pred = eligible["baseline_prediction"]
    choose(
        "predicted_positive_challenge",
        pred.eq("Positive") & (eligible["is_question"] | eligible["has_negation"] | eligible["has_negative_signal"] | eligible["mixed_signal"] | (eligible["prediction_confidence_numeric"] < 0.60)),
        45,
    )
    choose(
        "negative_question_or_skin_condition",
        pred.eq("Negative") & (eligible["is_question"] | eligible["has_skin_condition"] | eligible["has_purchase_cta"]),
        45,
    )
    choose(
        "skin_condition_without_product_evaluation",
        eligible["has_skin_condition"] & ~eligible["has_positive_effect"] & ~eligible["has_negative_signal"],
        35,
    )
    choose("improvement_or_clear_positive_result", eligible["has_positive_effect"], 35)
    choose("promotion_recommendation_or_cta", eligible["has_purchase_cta"] | eligible["has_promotion"], 35)
    choose("mixed_sentiment_negation_or_sarcasm_proxy", eligible["mixed_signal"] | eligible["has_negation"], 35)
    choose(
        "emoji_slang_code_mixing_or_short_text",
        eligible["has_emoji"] | eligible["has_slang"] | eligible["has_code_mixing"] | eligible["is_short_text"],
        35,
    )
    choose(
        "hcc_brand_balance",
        eligible["actor_segment"].eq("HCC"),
        25,
    )
    choose(
        "low_confidence_brand_balance",
        eligible["prediction_confidence_numeric"] < 0.48,
        10,
    )

    if pieces:
        selected = pd.concat(pieces, ignore_index=True)
    else:
        selected = pd.DataFrame(columns=list(eligible.columns) + ["sample_role", "sampling_strategy", "sampling_reason"])

    if len(selected) < n:
        fill = eligible.loc[~eligible["comment_id"].isin(set(selected["comment_id"]))].copy()
        fill = fill.sort_values(
            ["error_focus_score", "prediction_confidence_numeric", "top2_margin_numeric", "random_key", "comment_id"],
            ascending=[False, True, True, True, True],
        ).head(n - len(selected))
        fill["sample_role"] = "development_v2"
        fill["sampling_strategy"] = "active_learning_error_focused"
        fill["sampling_reason"] = "error_focus_fill"
        selected = pd.concat([selected, fill], ignore_index=True)

    if len(selected) != n:
        raise AssertionError(f"development_v2 count is {len(selected)}, expected {n}")
    if selected["comment_id"].duplicated().any():
        raise AssertionError("development_v2 contains duplicate comment_id")
    return selected


def allocate_stratified(counts: pd.Series, n: int) -> pd.Series:
    counts = counts.sort_index()
    raw = counts / counts.sum() * n
    allocation = np.floor(raw).astype(int)
    allocation = allocation.clip(upper=counts)
    remaining = n - int(allocation.sum())

    remainder_order = (raw - allocation).sort_values(ascending=False).index.tolist()
    while remaining > 0:
        changed = False
        for key in remainder_order:
            if allocation.loc[key] < counts.loc[key]:
                allocation.loc[key] += 1
                remaining -= 1
                changed = True
                if remaining == 0:
                    break
        if not changed:
            break

    while remaining < 0:
        for key in (raw - allocation).sort_values(ascending=True).index.tolist():
            if allocation.loc[key] > 0:
                allocation.loc[key] -= 1
                remaining += 1
                if remaining == 0:
                    break

    if int(allocation.sum()) != n:
        raise AssertionError(f"Unable to allocate locked_test_v2 sample exactly: {allocation.sum()} of {n}")
    return allocation


def stratified_sample_from_pool(pool: pd.DataFrame, n: int, seed_offset: int = 0) -> pd.DataFrame:
    if n <= 0:
        return pool.head(0).copy()
    counts = pool["sampling_stratum"].value_counts()
    allocation = allocate_stratified(counts, n)
    pieces: list[pd.DataFrame] = []
    for idx, (stratum, take_n) in enumerate(allocation.items()):
        if take_n <= 0:
            continue
        group = pool.loc[pool["sampling_stratum"].eq(stratum)]
        sampled = group.sample(n=int(take_n), random_state=SELECTION_SEED + seed_offset + idx)
        pieces.append(sampled)
    return pd.concat(pieces, ignore_index=True) if pieces else pool.head(0).copy()


def select_locked_test_v2(eligible: pd.DataFrame, development_ids: set[str], n: int = 300) -> pd.DataFrame:
    pool = eligible.loc[~eligible["comment_id"].isin(development_ids)].copy()
    pool["sampling_stratum"] = (
        pool["actor_segment"]
        + "|"
        + pool["brand_or_video_context"].replace("", "Unknown")
        + "|"
        + pool["length_bin"]
        + "|"
        + pool["text_type_major"]
    )
    hcc_pool = pool.loc[pool["actor_segment"].eq("HCC")]
    non_hcc_pool = pool.loc[pool["actor_segment"].eq("Non-HCC")]
    hcc_n = min(30, len(hcc_pool))
    non_hcc_n = n - hcc_n
    selected = pd.concat(
        [
            stratified_sample_from_pool(hcc_pool, hcc_n, seed_offset=101),
            stratified_sample_from_pool(non_hcc_pool, non_hcc_n, seed_offset=501),
        ],
        ignore_index=True,
    )
    selected = selected.sample(frac=1, random_state=SELECTION_SEED + 991).reset_index(drop=True)
    selected["sample_role"] = "locked_test_v2"
    selected["sampling_strategy"] = "stratified_random_locked_test"
    selected["sampling_reason"] = "stratified_random_not_error_selected"
    selected["selection_seed"] = SELECTION_SEED
    if len(selected) != n:
        raise AssertionError(f"locked_test_v2 count is {len(selected)}, expected {n}")
    if selected["comment_id"].duplicated().any():
        raise AssertionError("locked_test_v2 contains duplicate comment_id")
    return selected


def write_codebook() -> None:
    rows: list[dict[str, str]] = []
    sentiment_defs = {
        "Positive": "Clear favorable evaluation, improvement, recommendation, support, or promotion.",
        "Neutral": "Question, factual mention, condition description, tagging, purchase logistics, or unclear context without dominant sentiment.",
        "Negative": "Clear unfavorable evaluation, adverse product effect, safety concern, authenticity concern, complaint, or rejection.",
        "Uncertain": "Balanced or ambiguous positive and negative signals with no dominant polarity.",
        "No Text": "Empty, deleted, unreadable, or non-linguistic content that cannot be interpreted.",
    }
    target_defs = {
        "Product / Brand": "Sentiment is mainly toward the product or brand.",
        "Skin condition": "Text mainly describes a skin condition without necessarily evaluating a product.",
        "Usage question": "Text asks how, when, or whether to use a product or ingredient.",
        "Creator / Seller": "Sentiment is mainly toward the creator, seller, service, or account.",
        "Price / Purchase": "Text concerns price, purchase process, stock, link, shipping, or availability.",
        "Promotion / CTA": "Text is mainly a recommendation, sales prompt, affiliate-like call to action, or promotional cue.",
        "General discussion": "General skincare or conversational content not tied to a specific product evaluation.",
        "Other / unclear": "Target cannot be determined confidently.",
    }
    scope_defs = {
        "product_effect": "Complaint or praise about the product effect on skin or usage result.",
        "skin_condition": "Complaint about skin condition itself, not necessarily caused by the product.",
        "price_value": "Concern about price, value, affordability, shipping, or purchase terms.",
        "safety_concern": "Concern about irritation, danger, ingredient safety, or health risk.",
        "authenticity_concern": "Concern about fake product, official seller, originality, or trustworthiness.",
        "usage_confusion": "Confusion about sequence, frequency, compatibility, or how to use.",
        "not_applicable": "No complaint scope applies.",
        "unclear": "Scope cannot be determined from the comment.",
    }
    for value, definition in sentiment_defs.items():
        rows.append({"field": "sentiment_label", "allowed_value": value, "definition": definition})
    for value, definition in target_defs.items():
        rows.append({"field": "sentiment_target", "allowed_value": value, "definition": definition})
    for value, definition in scope_defs.items():
        rows.append({"field": "complaint_scope", "allowed_value": value, "definition": definition})
    pd.DataFrame(rows).to_csv(OUT_DIR / "sentiment_human_annotation_v2_codebook.csv", index=False)


def write_guideline() -> None:
    guideline = """# RM2 Sentiment Human Annotation V2 Guideline

## Objective

This package collects additional human labels for RM2 comment-level sentiment analysis. The goal is to improve future diagnosis of Positive recall, Negative precision, questions about skin condition, and model coverage. Do not use model predictions, heuristic labels, HCC goal outputs, or prior automated results when labeling.

## Main Sentiment Labels

- Positive: use when the comment clearly praises, recommends, supports, promotes, or reports improvement. Example: "Aku cocok, bekas jerawat makin pudar."
- Neutral: use for questions, factual statements, tagging, product names only, price or purchase logistics without evaluation, or skin condition descriptions without product evaluation. Example: "Ini dipakai pagi atau malam?"
- Negative: use when the comment clearly complains about product or brand effect, safety, authenticity, value, or usage result. Example: "Setelah pakai ini wajahku perih dan merah."
- Uncertain: use when positive and negative signals are balanced or context is insufficient. Example: "Bagus sih, tapi di aku bikin kering."
- No Text: use when text is empty, deleted, unreadable, or only contains content that cannot be interpreted.

## Sentiment Target

- Product / Brand: the evaluation is directed at a skincare product or brand.
- Skin condition: the comment mainly describes acne, dullness, irritation, oiliness, or another skin condition.
- Usage question: the comment asks whether, when, or how to use a product or ingredient.
- Creator / Seller: the evaluation is directed at the creator, seller, service, or account.
- Price / Purchase: the comment concerns price, link, cart, checkout, shipping, stock, or availability.
- Promotion / CTA: the comment is mainly a recommendation, sales prompt, affiliate-style cue, or call to purchase.
- General discussion: general skincare conversation without a specific target.
- Other / unclear: the target cannot be determined.

## Complaint Scope

- product_effect: product effect or result is the object of complaint or praise.
- skin_condition: skin condition is mentioned as a problem without clear product causality.
- price_value: price, value, shipping, or purchase terms are central.
- safety_concern: irritation, ingredient safety, danger, or health risk is central.
- authenticity_concern: fake product, official store, originality, or trust is central.
- usage_confusion: sequence, frequency, compatibility, or how-to confusion is central.
- not_applicable: no complaint scope applies.
- unclear: scope cannot be determined.

## Decision Rules

1. Questions about skin condition without product evaluation tend to be Neutral with target Skin condition or Usage question.
2. Skin complaints are not automatically Negative toward Product / Brand. Mark Negative toward Product / Brand only when the comment links the bad effect to product usage.
3. Bad effects after product use can be Negative, target Product / Brand, scope product_effect or safety_concern.
4. Improvement, recommendation, support, and clear promotion can be Positive.
5. If positive and negative signals are equally strong, use Uncertain.
6. Emoji should not override the main text meaning.
7. Words such as jerawat, bruntusan, kusam, mahal, murah, aman, and cocok must be judged by sentence context.

## Examples

| Comment example | Sentiment | Target | Complaint scope | Note |
|---|---|---|---|---|
| "Kak ini aman buat kulit sensitif?" | Neutral | Usage question | usage_confusion | A question, not a product complaint. |
| "Jerawatku lagi parah banget" | Neutral | Skin condition | skin_condition | Skin condition without product causality. |
| "Pakai ini malah breakout" | Negative | Product / Brand | product_effect | Product effect is blamed. |
| "Aku cocok banget, jadi lebih cerah" | Positive | Product / Brand | product_effect | Clear improvement. |
| "Mahal tapi worth it" | Positive | Product / Brand | price_value | Positive dominates despite price mention. |
| "Bagus tapi bikin kering" | Uncertain | Product / Brand | product_effect | Mixed sentiment with no dominant polarity. |
| "Checkout sekarang, lagi promo" | Positive | Promotion / CTA | not_applicable | Clear promotional CTA. |
| "😂😂😂" | No Text | Other / unclear | unclear | Emoji-only with no interpretable sentiment. |
"""
    (OUT_DIR / "sentiment_human_annotation_v2_guideline.md").write_text(guideline, encoding="utf-8")


def write_guideline() -> None:
    guideline = """# Panduan Anotasi Manusia Sentimen RM2 V2

## Tujuan

Paket ini mengumpulkan label manusia tambahan untuk analisis sentimen RM2 pada level komentar. Tujuannya adalah memperbaiki diagnosis lanjutan terhadap recall kelas Positive, precision kelas Negative, pembedaan pertanyaan atau kondisi kulit dari evaluasi produk, serta coverage model.

Saat memberi label, jangan menggunakan prediksi model, label heuristik, output goal HCC, probability, confidence, atau hasil otomatis sebelumnya. Nilai komentar hanya berdasarkan teks komentar dan konteks video/brand yang tersedia di file anotasi.

## Label Sentimen Utama

- Positive: gunakan jika komentar jelas memuji, merekomendasikan, mendukung, mempromosikan, atau melaporkan hasil membaik. Contoh: "Aku cocok, bekas jerawat makin pudar."
- Neutral: gunakan untuk pertanyaan, pernyataan faktual, tagging, nama produk saja, logistik harga/pembelian tanpa evaluasi, atau deskripsi kondisi kulit tanpa evaluasi produk. Contoh: "Ini dipakai pagi atau malam?"
- Negative: gunakan jika komentar jelas mengeluhkan efek produk/brand, keamanan, keaslian, nilai harga, atau hasil pemakaian. Contoh: "Setelah pakai ini wajahku perih dan merah."
- Uncertain: gunakan jika sinyal positif dan negatif sama kuat atau konteks tidak cukup untuk menentukan polaritas dominan. Contoh: "Bagus sih, tapi di aku bikin kering."
- No Text: gunakan jika teks kosong, terhapus, tidak terbaca, atau hanya berisi konten yang tidak dapat ditafsirkan.

## Target Sentimen

- Product / Brand: evaluasi terutama diarahkan pada produk skincare atau brand.
- Skin condition: komentar terutama menjelaskan jerawat, kusam, iritasi, berminyak, kering, atau kondisi kulit lain.
- Usage question: komentar menanyakan apakah, kapan, atau bagaimana memakai produk atau bahan tertentu.
- Creator / Seller: evaluasi diarahkan pada creator, penjual, layanan, atau akun.
- Price / Purchase: komentar membahas harga, link, keranjang, checkout, pengiriman, stok, atau ketersediaan.
- Promotion / CTA: komentar terutama berupa rekomendasi, ajakan membeli, promosi, atau call to action.
- General discussion: percakapan skincare umum tanpa target evaluasi spesifik.
- Other / unclear: target tidak dapat ditentukan dengan cukup jelas.

## Complaint Scope

- product_effect: keluhan atau pujian berhubungan dengan efek produk atau hasil pemakaian.
- skin_condition: kondisi kulit disebut sebagai masalah tanpa hubungan sebab-akibat produk yang jelas.
- price_value: harga, nilai, ongkir, atau syarat pembelian menjadi isu utama.
- safety_concern: iritasi, keamanan bahan, bahaya, atau risiko kesehatan menjadi isu utama.
- authenticity_concern: keaslian produk, official store, originalitas, atau kepercayaan menjadi isu utama.
- usage_confusion: urutan pemakaian, frekuensi, kompatibilitas, atau cara pakai menjadi sumber kebingungan.
- not_applicable: tidak ada cakupan keluhan yang relevan.
- unclear: cakupan tidak dapat ditentukan dari komentar.

## Aturan Keputusan

1. Pertanyaan tentang kondisi kulit tanpa evaluasi produk cenderung Neutral dengan target Skin condition atau Usage question.
2. Keluhan kondisi kulit tidak otomatis berarti Negative terhadap Product / Brand. Beri Negative terhadap Product / Brand hanya jika komentar mengaitkan efek buruk dengan pemakaian produk.
3. Efek buruk setelah menggunakan produk dapat diberi Negative, target Product / Brand, scope product_effect atau safety_concern.
4. Hasil membaik, rekomendasi, dukungan, dan promosi yang jelas dapat diberi Positive.
5. Jika sinyal positif dan negatif sama kuat, gunakan Uncertain.
6. Emoji tidak boleh mengalahkan makna utama teks.
7. Kata seperti jerawat, bruntusan, kusam, mahal, murah, aman, dan cocok harus dinilai berdasarkan konteks kalimat.

## Contoh

| Contoh komentar | Sentimen | Target | Complaint scope | Catatan |
|---|---|---|---|---|
| "Kak ini aman buat kulit sensitif?" | Neutral | Usage question | usage_confusion | Pertanyaan, bukan keluhan produk. |
| "Jerawatku lagi parah banget" | Neutral | Skin condition | skin_condition | Kondisi kulit tanpa sebab-akibat produk. |
| "Pakai ini malah breakout" | Negative | Product / Brand | product_effect | Efek produk disalahkan. |
| "Aku cocok banget, jadi lebih cerah" | Positive | Product / Brand | product_effect | Ada hasil membaik yang jelas. |
| "Mahal tapi worth it" | Positive | Product / Brand | price_value | Sinyal positif dominan meskipun harga disebut. |
| "Bagus tapi bikin kering" | Uncertain | Product / Brand | product_effect | Sentimen campuran tanpa polaritas dominan. |
| "Checkout sekarang, lagi promo" | Positive | Promotion / CTA | not_applicable | Ajakan promosi yang jelas. |
| "[emoji tertawa]" | No Text | Other / unclear | unclear | Emoji saja tanpa sentimen yang dapat ditafsirkan. |
"""
    (OUT_DIR / "sentiment_human_annotation_v2_guideline.md").write_text(guideline, encoding="utf-8")


def build_annotation_files(sample: pd.DataFrame) -> None:
    annotator_cols = [
        "sample_role",
        "comment_id",
        "comment_text_original",
        "video_id",
        "brand_or_video_context",
        "sentiment_label",
        "sentiment_target",
        "complaint_scope",
        "annotator_notes",
    ]
    blind = sample[["sample_role", "comment_id", "comment_text_original", "video_id", "brand_or_video_context"]].copy()
    for col in ["sentiment_label", "sentiment_target", "complaint_scope", "annotator_notes"]:
        blind[col] = ""
    blind = blind[annotator_cols]
    blind.to_csv(OUT_DIR / "sentiment_v2_annotator_1_blind.csv", index=False)
    blind.to_csv(OUT_DIR / "sentiment_v2_annotator_2_blind.csv", index=False)

    adjudication = sample[["sample_role", "comment_id", "comment_text_original", "video_id", "brand_or_video_context"]].copy()
    for prefix in ["annotator_1", "annotator_2"]:
        adjudication[f"{prefix}_sentiment_label"] = ""
        adjudication[f"{prefix}_sentiment_target"] = ""
        adjudication[f"{prefix}_complaint_scope"] = ""
        adjudication[f"{prefix}_notes"] = ""
    adjudication["adjudicated_sentiment_label"] = ""
    adjudication["adjudicated_sentiment_target"] = ""
    adjudication["adjudicated_complaint_scope"] = ""
    adjudication["adjudication_notes"] = ""
    adjudication.to_csv(OUT_DIR / "sentiment_v2_adjudication_template.csv", index=False)


def write_provenance(v1: pd.DataFrame, development: pd.DataFrame, locked: pd.DataFrame) -> pd.DataFrame:
    v1_frame = v1.copy()
    v1_frame["comment_id"] = v1_frame["comment_id"].map(normalize_id)
    role_source = "sample_set" if "sample_set" in v1_frame.columns else "original_sample_set"
    v1_frame["original_sample_role"] = v1_frame.get(role_source, "").astype(str)
    v1_frame["v2_sample_role"] = np.where(
        v1_frame["original_sample_role"].str.lower().eq("locked_test"),
        "historical_test_v1",
        "human_development_v1",
    )
    v1_prov = pd.DataFrame(
        {
            "comment_id": v1_frame["comment_id"],
            "annotation_version": "v1",
            "original_sample_role": v1_frame["original_sample_role"],
            "v2_sample_role": v1_frame["v2_sample_role"],
            "human_label_source": "sentiment_human_annotation_validated.csv",
        }
    )
    v2_prov = pd.DataFrame(
        {
            "comment_id": pd.concat([development["comment_id"], locked["comment_id"]], ignore_index=True),
            "annotation_version": "v2_pending",
            "original_sample_role": "not_previously_annotated",
            "v2_sample_role": pd.concat([development["sample_role"], locked["sample_role"]], ignore_index=True),
            "human_label_source": "pending_human_annotation_v2",
        }
    )
    provenance = pd.concat([v1_prov, v2_prov], ignore_index=True)
    provenance.to_csv(OUT_DIR / "annotation_provenance_v2.csv", index=False)
    return provenance


def sha256_text(values: list[str]) -> str:
    payload = "\n".join(values) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def weak_component_count(nodes: set[str], edges: pd.DataFrame) -> int:
    parent = {node: node for node in nodes}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for row in edges[["Source", "Target"]].itertuples(index=False):
        source, target = normalize_id(row.Source), normalize_id(row.Target)
        if source in parent and target in parent:
            union(source, target)
    return len({find(node) for node in nodes})


def git_diff_quiet(path: str) -> bool:
    result = subprocess.run(
        ["git", "diff", "--quiet", BASELINE_COMMIT, "--", path],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.returncode == 0


def build_integrity_report(dataset: pd.DataFrame, comments: pd.DataFrame, hcc_nodes: pd.DataFrame, actor: pd.DataFrame) -> pd.DataFrame:
    nodes = read_csv(GEPHI_ACTOR_NODES_PATH)
    edges = read_csv(GEPHI_ACTOR_EDGES_PATH)
    node_ids = set(nodes["Id"].map(normalize_id))
    missing_endpoints = int(
        (~edges["Source"].map(normalize_id).isin(node_ids) | ~edges["Target"].map(normalize_id).isin(node_ids)).sum()
    )
    artifact_count = int(
        nodes["Id"].map(contains_non_hcc_artifact).sum()
        + edges["Source"].map(contains_non_hcc_artifact).sum()
        + edges["Target"].map(contains_non_hcc_artifact).sum()
    )
    hcc_count = int(hcc_nodes["community"].map(normalize_id).nunique())
    hcc_member_count = len(hcc_nodes)
    hcc_comment_count = int(comments.get("is_hcc_member", pd.Series("", index=comments.index)).map(normalize_bool).sum())
    actor_counts = actor["actor_type_primary"].value_counts().to_dict()
    wcc = weak_component_count(node_ids, edges)
    protected_rm1_paths = [
        "dataset.csv",
        "video_metadata_clean.csv",
        "notebooks/rm1/tiktok_coordination_analysis.ipynb",
        "output/gephi/gephi_lcn_nodes.csv",
        "output/gephi/gephi_lcn_edges.csv",
        "output/gephi/gephi_hcc_nodes.csv",
        "output/gephi/gephi_hcc_edges.csv",
    ]
    rm1_diff_paths = [path for path in protected_rm1_paths if (ROOT / path).exists() and not git_diff_quiet(path)]
    pipeline_text = PIPELINE_SCRIPT_PATH.read_text(encoding="utf-8")

    checks = [
        ("dataset rows", 33847, len(dataset), len(dataset) == 33847, ""),
        ("unique comment_id", 33847, dataset["comment_id"].map(normalize_id).nunique(), dataset["comment_id"].map(normalize_id).nunique() == 33847, ""),
        ("HCC count", 42, hcc_count, hcc_count == 42, ""),
        ("HCC members", 218, hcc_member_count, hcc_member_count == 218, ""),
        ("HCC comments", 1009, hcc_comment_count, hcc_comment_count == 1009, ""),
        ("actor universe", 26427, len(actor), len(actor) == 26427, ""),
        ("Individual Actor", 43, actor_counts.get("Individual Actor", 0), actor_counts.get("Individual Actor", 0) == 43, ""),
        ("Community Actor", 218, actor_counts.get("Community Actor", 0), actor_counts.get("Community Actor", 0) == 218, ""),
        ("Mass Actor", 26166, actor_counts.get("Mass Actor", 0), actor_counts.get("Mass Actor", 0) == 26166, ""),
        ("Gephi actor type nodes", 396, len(nodes), len(nodes) == 396, ""),
        ("Gephi actor type edges", 497, len(edges), len(edges) == 497, ""),
        ("missing endpoint", 0, missing_endpoints, missing_endpoints == 0, ""),
        ("weakly connected components", 1, wcc, wcc == 1, "weakly connected for directed graph"),
        ("HCC_Non-HCC artifact", 0, artifact_count, artifact_count == 0, ""),
        ("RM1 checksums unchanged", "no diff vs baseline", ",".join(rm1_diff_paths) or "no diff", len(rm1_diff_paths) == 0, ""),
        (
            "safe_clean preserves human_validation_v2",
            "human_validation_v2 protected",
            "human_validation_v2" if "human_validation_v2" in pipeline_text else "missing",
            "human_validation_v2" in pipeline_text,
            "",
        ),
    ]
    report = pd.DataFrame(
        [
            {
                "check": name,
                "expected": expected,
                "observed": observed,
                "status": "PASS" if passed else "FAIL",
                "notes": notes,
            }
            for name, expected, observed, passed, notes in checks
        ]
    )
    report.to_csv(OUT_DIR / "sentiment_v2_integrity_report.csv", index=False)
    if report["status"].eq("FAIL").any():
        raise AssertionError("Integrity checks failed:\n" + report.loc[report["status"].eq("FAIL")].to_string(index=False))
    return report


def write_readme(v1_count: int) -> None:
    text = f"""# RM2 Sentiment Human Validation V2

This directory contains the second human annotation package for RM2 comment-level sentiment analysis.

Phase status: package creation only. No model retraining, threshold selection, ensemble selection, inference rerun, HCC goal update, actor-type update, or Gephi topology update was performed.

Current V1 exclusion set: {v1_count} unique comment_id values from `output/rm2_sentiment/human_validation/sentiment_human_annotation_validated.csv`. The V1 file is preserved and not overwritten by this package.

Files for annotators:

- `sentiment_v2_annotator_1_blind.csv`
- `sentiment_v2_annotator_2_blind.csv`
- `sentiment_v2_adjudication_template.csv`

Do not add model predictions, heuristic labels, HCC IDs, goal outputs, probabilities, or error flags to annotator blind files.

Use `sentiment_human_annotation_v2_guideline.md` and `sentiment_human_annotation_v2_codebook.csv` when labeling.

After annotation is complete, run:

```powershell
python scripts/validate_rm2_sentiment_human_annotations_v2.py
```

The locked-test V2 IDs are fixed in `locked_test_v2_manifest.csv`; they must not be used for model selection, threshold selection, ensemble weighting, preprocessing selection, or error-driven tuning.
"""
    (OUT_DIR / "README_HUMAN_VALIDATION_V2.md").write_text(text, encoding="utf-8")


def write_readme(v1_count: int) -> None:
    text = f"""# Validasi Manusia Sentimen RM2 V2

Direktori ini berisi paket anotasi manusia tahap kedua untuk analisis sentimen RM2 pada level komentar.

Status fase: hanya pembuatan paket anotasi. Tidak ada retraining model, pemilihan threshold, pemilihan ensemble, inference ulang, pembaruan goal HCC, pembaruan actor type, atau perubahan topology Gephi.

Daftar eksklusi V1 saat ini berisi {v1_count} nilai `comment_id` unik dari `output/rm2_sentiment/human_validation/sentiment_human_annotation_validated.csv`. File V1 dipertahankan dan tidak ditimpa oleh paket ini.

File yang perlu diisi anotator:

- `sentiment_v2_annotator_1_blind.csv`
- `sentiment_v2_annotator_2_blind.csv`
- `sentiment_v2_adjudication_template.csv`

Jangan menambahkan prediksi model, label heuristik, ID HCC, output goal, probability, confidence, atau penanda error ke file blind anotator.

Gunakan `sentiment_human_annotation_v2_guideline.md` dan `sentiment_human_annotation_v2_codebook.csv` saat memberi label.

Setelah anotasi selesai, jalankan:

```powershell
python scripts/validate_rm2_sentiment_human_annotations_v2.py
```

ID locked-test V2 sudah dikunci di `locked_test_v2_manifest.csv`. ID tersebut tidak boleh digunakan untuk model selection, threshold selection, ensemble weighting, preprocessing selection, atau tuning berbasis error.
"""
    (OUT_DIR / "README_HUMAN_VALIDATION_V2.md").write_text(text, encoding="utf-8")


def write_sampling_outputs(development: pd.DataFrame, locked: pd.DataFrame) -> pd.DataFrame:
    sample = pd.concat([development, locked], ignore_index=True)
    sample = sample.sample(frac=1, random_state=SELECTION_SEED + 17).reset_index(drop=True)
    sample["selection_seed"] = SELECTION_SEED

    manifest_cols = [
        "sample_role",
        "comment_id",
        "sampling_strategy",
        "sampling_reason",
        "actor_segment",
        "brand_or_video_context",
        "length_bin",
        "text_type_major",
        "baseline_prediction",
        "selection_seed",
    ]
    sample[manifest_cols].to_csv(OUT_DIR / "sentiment_v2_sampling_manifest.csv", index=False)

    locked_manifest = locked[["comment_id", "sample_role", "sampling_stratum"]].copy()
    locked_manifest["selection_seed"] = SELECTION_SEED
    locked_manifest.to_csv(OUT_DIR / "locked_test_v2_manifest.csv", index=False)
    locked_ids_sorted = sorted(locked_manifest["comment_id"].map(normalize_id).tolist())
    checksum = sha256_text(locked_ids_sorted)
    pd.DataFrame(
        [
            {"metric": "selection_seed", "value": str(SELECTION_SEED)},
            {"metric": "n_locked_test_v2_ids", "value": str(len(locked_ids_sorted))},
            {"metric": "locked_test_v2_comment_id_sha256", "value": checksum},
        ]
    ).to_csv(OUT_DIR / "locked_test_v2_checksum.csv", index=False)

    distribution_frames = []
    for group_col in ["sample_role", "actor_segment", "brand_or_video_context", "text_type_major", "length_bin", "baseline_prediction"]:
        if group_col == "sample_role":
            dist = sample.groupby(group_col, dropna=False)["comment_id"].nunique().reset_index(name="n_comments")
            dist = dist.rename(columns={group_col: "category"})
            dist.insert(0, "sample_role", "all")
        else:
            dist = sample.groupby(["sample_role", group_col], dropna=False)["comment_id"].nunique().reset_index(name="n_comments")
            dist = dist.rename(columns={group_col: "category"})
        dist.insert(1, "distribution_dimension", group_col)
        distribution_frames.append(dist)
    distribution = pd.concat(distribution_frames, ignore_index=True)
    distribution.to_csv(OUT_DIR / "sentiment_v2_sampling_distribution.csv", index=False)
    return sample


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    dataset = read_csv(DATASET_PATH)
    comments = add_text_features(read_csv(COMMENT_SENTIMENT_PATH))
    v1 = read_csv(V1_VALIDATED_PATH)
    actor = read_csv(ACTOR_TYPE_PATH)
    hcc_nodes = read_csv(HCC_NODES_PATH)

    v1_ids = set(v1["comment_id"].map(normalize_id)) - {""}
    eligible = comments.loc[~comments["comment_id"].isin(v1_ids)].copy()
    if len(eligible) < 600:
        raise AssertionError(f"Only {len(eligible)} comments are eligible for V2 after excluding V1.")

    development = select_development_v2(eligible, 300)
    locked = select_locked_test_v2(eligible, set(development["comment_id"]), 300)

    overlap_v1_dev = set(development["comment_id"]) & v1_ids
    overlap_v1_locked = set(locked["comment_id"]) & v1_ids
    overlap_dev_locked = set(development["comment_id"]) & set(locked["comment_id"])
    if overlap_v1_dev or overlap_v1_locked or overlap_dev_locked:
        raise AssertionError(
            "V2 overlap detected: "
            f"v1-development={len(overlap_v1_dev)}, "
            f"v1-locked={len(overlap_v1_locked)}, "
            f"development-locked={len(overlap_dev_locked)}"
        )

    sample = write_sampling_outputs(development, locked)
    write_provenance(v1, development, locked)
    write_codebook()
    write_guideline()
    build_annotation_files(sample)
    write_readme(len(v1_ids))
    integrity = build_integrity_report(dataset, comments, hcc_nodes, actor)

    summary = pd.DataFrame(
        [
            {"metric": "v1_unique_comment_ids_preserved", "value": len(v1_ids), "status": "INFO", "notes": "Current V1 file is used as the exclusion set and is not overwritten."},
            {"metric": "development_v2_comments", "value": len(development), "status": "PASS", "notes": "Active-learning and error-focused sample."},
            {"metric": "locked_test_v2_comments", "value": len(locked), "status": "PASS", "notes": "Stratified random sample, not selected from error list."},
            {"metric": "v1_v2_overlap", "value": 0, "status": "PASS", "notes": "No V2 comment_id appears in current V1."},
            {"metric": "development_locked_overlap", "value": 0, "status": "PASS", "notes": "Development V2 and locked test V2 are disjoint."},
            {"metric": "model_retraining", "value": "not run", "status": "PASS", "notes": "Package creation only."},
            {"metric": "comment_sentiment_update", "value": "not modified by this script", "status": "PASS", "notes": "Existing predictions are read only for sampling diagnostics."},
        ]
    )
    summary.to_csv(OUT_DIR / "sentiment_v2_package_summary.csv", index=False)

    print("RM2 SENTIMENT V2 ANNOTATION PACKAGE")
    print(f"- V1 unique comment_id excluded: {len(v1_ids)}")
    print(f"- development_v2: {len(development)}")
    print(f"- locked_test_v2: {len(locked)}")
    print(f"- locked_test_v2 checksum: {pd.read_csv(OUT_DIR / 'locked_test_v2_checksum.csv').iloc[2]['value']}")
    print("- integrity checks: PASS")
    print(integrity.to_string(index=False))


if __name__ == "__main__":
    main()
