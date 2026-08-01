from __future__ import annotations

import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.io_utils import save_figure, write_csv_atomic, write_json_atomic
from scripts.project_paths import RM2_SENTIMENT_FINAL_DIR, RM2_SENTIMENT_FINAL_TABLES_DIR, RM2_SENTIMENT_FINAL_VISUALIZATION_DIR

SOURCE = RM2_SENTIMENT_FINAL_DIR / "indobert_v5_comment_sentiment.csv"
TOP_TERMS = RM2_SENTIMENT_FINAL_TABLES_DIR / "wordcloud_top_terms_indobert_v5_final.csv"
MANIFEST = RM2_SENTIMENT_FINAL_DIR / "INDOBERT_V5_FINAL_WORDCLOUD_MANIFEST.json"

HCC_VS_NONHCC_FIG = RM2_SENTIMENT_FINAL_VISUALIZATION_DIR / "indobert_v5_wordcloud_hcc_vs_nonhcc_final.png"
BY_SENTIMENT_FIG = RM2_SENTIMENT_FINAL_VISUALIZATION_DIR / "indobert_v5_wordcloud_by_sentiment_final.png"
HCC_BY_SENTIMENT_FIG = RM2_SENTIMENT_FINAL_VISUALIZATION_DIR / "indobert_v5_wordcloud_hcc_by_sentiment_final.png"

LABEL_ORDER = ["Negative", "Neutral", "Positive"]
STATUS_ORDER = ["HCC", "Non-HCC"]

TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_+\-]{1,}")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
HASHTAG_RE = re.compile(r"#(\w+)")
MENTION_RE = re.compile(r"@\w+")

STOPWORDS = {
    "a",
    "ada",
    "adalah",
    "aja",
    "aku",
    "akun",
    "ama",
    "anak",
    "apa",
    "apakah",
    "atau",
    "banget",
    "banyak",
    "baru",
    "bawah",
    "begini",
    "begitu",
    "beli",
    "belinya",
    "bener",
    "beneran",
    "berarti",
    "bgt",
    "biar",
    "bisa",
    "boleh",
    "buat",
    "coba",
    "cuma",
    "dah",
    "dan",
    "dari",
    "deh",
    "dengan",
    "di",
    "dia",
    "dok",
    "dong",
    "dr",
    "dulu",
    "for",
    "ga",
    "gak",
    "gimana",
    "gue",
    "gw",
    "hah",
    "harus",
    "huhu",
    "ini",
    "itu",
    "iya",
    "iyaa",
    "iyah",
    "jadi",
    "jangan",
    "jg",
    "juga",
    "justru",
    "ka",
    "kak",
    "kakak",
    "kalo",
    "kalau",
    "kan",
    "karena",
    "kayak",
    "kaya",
    "ke",
    "cek",
    "kok",
    "klo",
    "ku",
    "lagi",
    "lah",
    "langsung",
    "lebih",
    "lo",
    "loh",
    "lu",
    "maaf",
    "mana",
    "mau",
    "memang",
    "min",
    "minta",
    "mohon",
    "mulu",
    "nah",
    "nih",
    "nya",
    "oh",
    "on",
    "pas",
    "pake",
    "pakek",
    "pakai",
    "pakainya",
    "pengen",
    "pkai",
    "pke",
    "pun",
    "sama",
    "sangat",
    "sapa",
    "saya",
    "se",
    "sih",
    "si",
    "sm",
    "sma",
    "sumpah",
    "supaya",
    "tak",
    "tapi",
    "td",
    "the",
    "ti",
    "to",
    "trs",
    "tu",
    "tuh",
    "udah",
    "udh",
    "untuk",
    "utk",
    "wkwk",
    "ya",
    "yaa",
    "yah",
    "yang",
    "yg",
}

# Brand names are removed so the final sentiment wordcloud emphasizes language
# used in comments rather than restating the video or HCC brand context.
BRAND_STOPWORDS = {
    "azarine",
    "avoskin",
    "daviena",
    "maryame",
    "originote",
    "theoriginote",
}


def normalize_bool(value: object) -> bool:
    text = str(value).strip().lower()
    return text in {"1", "true", "t", "yes", "y"}


def tokenize(text: object) -> list[str]:
    value = "" if pd.isna(text) else str(text).lower()
    value = URL_RE.sub(" ", value)
    value = HASHTAG_RE.sub(r" \1 ", value)
    value = MENTION_RE.sub(" ", value)
    value = value.replace("'", " ")
    tokens = []
    for token in TOKEN_RE.findall(value):
        normalized = token.strip("_+-").lower()
        if len(normalized) < 3:
            continue
        if normalized in STOPWORDS or normalized in BRAND_STOPWORDS:
            continue
        tokens.append(normalized)
    return tokens


def frequencies(texts: pd.Series) -> Counter[str]:
    counter: Counter[str] = Counter()
    for text in texts.dropna():
        counter.update(tokenize(text))
    return counter


def draw_wordcloud(ax: plt.Axes, freq: Counter[str], title: str, color: str) -> None:
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.axis("off")
    if not freq:
        ax.text(0.5, 0.5, "No eligible terms", ha="center", va="center", fontsize=12)
        return
    wc = WordCloud(
        width=1100,
        height=720,
        background_color="white",
        colormap=color,
        max_words=120,
        random_state=42,
        collocations=False,
        prefer_horizontal=0.9,
        relative_scaling=0.45,
        min_font_size=8,
        normalize_plurals=False,
    ).generate_from_frequencies(freq)
    ax.imshow(wc, interpolation="bilinear")


def save_panel(groups: list[tuple[str, Counter[str], str]], path: Path, ncols: int) -> None:
    nrows = (len(groups) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6.3 * ncols, 4.4 * nrows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (title, freq, color) in zip(axes.ravel(), groups):
        draw_wordcloud(ax, freq, title, color)
    fig.tight_layout(pad=1.6)
    save_figure(fig, path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def append_top_terms(rows: list[dict[str, object]], view: str, group: str, sentiment: str, freq: Counter[str], limit: int = 50) -> None:
    total = sum(freq.values())
    for rank, (term, count) in enumerate(freq.most_common(limit), start=1):
        rows.append(
            {
                "view": view,
                "group": group,
                "sentiment": sentiment,
                "rank": rank,
                "term": term,
                "count": count,
                "share": count / total if total else 0.0,
            }
        )


def main() -> None:
    if not SOURCE.exists():
        raise FileNotFoundError(SOURCE)

    df = pd.read_csv(SOURCE)
    required = {"text", "final_sentiment_label", "is_hcc"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in {SOURCE}: {sorted(missing)}")

    frame = df.copy()
    frame["final_sentiment_label"] = frame["final_sentiment_label"].astype(str).str.strip()
    frame["hcc_status"] = frame["is_hcc"].map(normalize_bool).map({True: "HCC", False: "Non-HCC"})

    top_term_rows: list[dict[str, object]] = []

    status_groups = []
    status_colors = {"HCC": "viridis", "Non-HCC": "cividis"}
    for status in STATUS_ORDER:
        subset = frame[frame["hcc_status"] == status]
        freq = frequencies(subset["text"])
        append_top_terms(top_term_rows, "hcc_vs_nonhcc", status, "All", freq)
        status_groups.append((f"{status} comments", freq, status_colors[status]))
    save_panel(status_groups, HCC_VS_NONHCC_FIG, ncols=2)

    sentiment_colors = {"Negative": "Reds", "Neutral": "Greys", "Positive": "Greens"}
    sentiment_groups = []
    for label in LABEL_ORDER:
        subset = frame[frame["final_sentiment_label"] == label]
        freq = frequencies(subset["text"])
        append_top_terms(top_term_rows, "all_comments_by_sentiment", "All comments", label, freq)
        sentiment_groups.append((f"All comments: {label}", freq, sentiment_colors[label]))
    save_panel(sentiment_groups, BY_SENTIMENT_FIG, ncols=3)

    hcc_groups = []
    hcc_frame = frame[frame["hcc_status"] == "HCC"]
    for label in LABEL_ORDER:
        subset = hcc_frame[hcc_frame["final_sentiment_label"] == label]
        freq = frequencies(subset["text"])
        append_top_terms(top_term_rows, "hcc_by_sentiment", "HCC", label, freq)
        hcc_groups.append((f"HCC comments: {label}", freq, sentiment_colors[label]))
    save_panel(hcc_groups, HCC_BY_SENTIMENT_FIG, ncols=3)

    top_terms = pd.DataFrame(top_term_rows)
    write_csv_atomic(top_terms, TOP_TERMS)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": SOURCE.relative_to(ROOT).as_posix(),
        "model": "indobert_v5_final",
        "text_column": "text",
        "sentiment_column": "final_sentiment_label",
        "hcc_column": "is_hcc",
        "comment_rows": int(len(frame)),
        "figures": [
            HCC_VS_NONHCC_FIG.relative_to(ROOT).as_posix(),
            BY_SENTIMENT_FIG.relative_to(ROOT).as_posix(),
            HCC_BY_SENTIMENT_FIG.relative_to(ROOT).as_posix(),
        ],
        "tables": [TOP_TERMS.relative_to(ROOT).as_posix()],
        "notes": [
            "Wordcloud terms are lowercased comment tokens from final IndoBERT V5 inference output.",
            "Common Indonesian function words, TikTok conversational fillers, and skincare brand names are excluded.",
            "This artifact is descriptive/presentation support, not model validation evidence.",
        ],
    }
    write_json_atomic(manifest, MANIFEST)

    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
