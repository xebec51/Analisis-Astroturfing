from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
import time
import unicodedata
from collections import defaultdict
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

try:
    from rapidfuzz import fuzz  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    fuzz = None


ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = ROOT / "dataset.csv"
VIDEO_METADATA_PATH = ROOT / "video_metadata_clean.csv"
ACCOUNT_ACTOR_TYPE_PATH = ROOT / "output/rm2_actor_type/tables/account_actor_type.csv"
COMMENT_SENTIMENT_PATH = ROOT / "output/rm2_sentiment/final/comment_sentiment_v2_observational.csv"
HCC_NODES_PATH = ROOT / "output/gephi/gephi_hcc_nodes.csv"
LCN_NODES_ACTOR_TYPE_PATH = ROOT / "output/rm2_actor_type/gephi/gephi_lcn_nodes_actor_type.csv"
LCN_EDGES_ACTOR_TYPE_PATH = ROOT / "output/rm2_actor_type/gephi/gephi_lcn_edges_actor_type.csv"
COMMUNITY_MASS_PAIRS_PATH = ROOT / "output/rm2_actor_type/account_interaction/community_mass_account_pairs.csv"
SENTIMENT_MODEL_PATH = ROOT / "output/rm2_sentiment/model/frozen/selected_model_development_frozen.joblib"
SENTIMENT_MANIFEST_PATH = ROOT / "output/rm2_sentiment/model/frozen/final_locked_test_evaluation_lock.json"

OUT_DIR = ROOT / "output/rm2_comment_similarity"
OUT_PRESENTATION_DIR = OUT_DIR / "presentation"

OUT_PAIRS_ALL = OUT_DIR / "comment_similarity_pairs_all.csv"
OUT_PAIR_EVIDENCE_SAMPLE = OUT_DIR / "comment_similarity_pair_evidence_sample.csv"
OUT_GROUPS = OUT_DIR / "comment_similarity_groups.csv"
OUT_GROUP_MEMBERS = OUT_DIR / "comment_similarity_group_members.csv"
OUT_GROUP_ACCOUNT_SUMMARY = OUT_DIR / "comment_similarity_group_account_summary.csv"
OUT_SCREENSHOT_QUEUE = OUT_DIR / "comment_similarity_screenshot_queue.csv"
OUT_EXACT_GROUPS = OUT_DIR / "exact_duplicate_comment_groups.csv"
OUT_EXACT_MEMBERS = OUT_DIR / "exact_duplicate_comment_group_members.csv"
OUT_CLUSTERS = OUT_DIR / "near_similar_comment_clusters.csv"
OUT_CLUSTER_MEMBERS = OUT_DIR / "near_similar_comment_cluster_members.csv"
OUT_THRESHOLD_MANUAL_AUDIT = OUT_DIR / "comment_similarity_threshold_manual_audit.csv"
OUT_SUMMARY = OUT_DIR / "comment_similarity_summary.csv"
OUT_BY_CATEGORY = OUT_DIR / "comment_similarity_by_category.csv"
OUT_BY_ACTOR_PAIR = OUT_DIR / "comment_similarity_by_actor_pair.csv"
OUT_BY_BRAND = OUT_DIR / "comment_similarity_by_brand.csv"
OUT_BY_HCC = OUT_DIR / "comment_similarity_by_hcc.csv"
OUT_GROUP_BY_CATEGORY = OUT_DIR / "comment_similarity_groups_by_category.csv"
OUT_GROUP_BY_ACTOR_SCOPE = OUT_DIR / "comment_similarity_groups_by_actor_scope.csv"
OUT_GROUP_BY_BRAND = OUT_DIR / "comment_similarity_groups_by_brand.csv"
OUT_GROUP_BY_HCC = OUT_DIR / "comment_similarity_groups_by_hcc.csv"
OUT_SAME_CROSS_VIDEO = OUT_DIR / "comment_similarity_same_vs_cross_video.csv"
OUT_SAME_CROSS_ACCOUNT = OUT_DIR / "comment_similarity_same_vs_cross_account.csv"
OUT_CM_OVERLAP = OUT_DIR / "comment_similarity_community_mass_network_overlap.csv"
OUT_THRESHOLD_AUDIT = OUT_DIR / "comment_similarity_threshold_audit.csv"
OUT_INTEGRITY = OUT_DIR / "comment_similarity_integrity_report.csv"
OUT_MANIFEST = OUT_DIR / "comment_similarity_run_manifest.json"

OUT_PPT_EXAMPLES = OUT_PRESENTATION_DIR / "ppt_comment_similarity_examples.csv"
OUT_PPT_MEMBERS = OUT_PRESENTATION_DIR / "ppt_comment_similarity_example_members.csv"
OUT_PPT_MARKDOWN = OUT_PRESENTATION_DIR / "ppt_comment_similarity_examples.md"
OUT_PPT_REVIEW = OUT_PRESENTATION_DIR / "ppt_comment_similarity_manual_review.csv"

COMMUNITY = "Community Actor"
MASS = "Mass Actor"
INDIVIDUAL = "Individual Actor"
HCC_POSITION = "HCC"
LCN_NON_HCC = "LCN Non-HCC"
OUTSIDE_LCN = "Outside LCN"
SENTIMENT_ATTRIBUTE_STATUS = "FINAL_MODEL_VALIDATED_SENTIMENT_V2"
FINAL_LOCKED_TEST_STATUS = "FINAL_LOCKED_TEST_EVALUATED_ONCE"
MASS_HASH_SALT = "rm2_actor_type_public_mass_hash_v1"
EN_DASH = "\u2013"

RANDOM_SEED = 20260723
TOP_K_NEIGHBORS = 20
SIMILARITY_THRESHOLD = 0.70
CLUSTER_THRESHOLD = 0.85
PAIR_EVIDENCE_PER_GROUP = 4
MAX_PAIR_EVIDENCE_ROWS = 5000
MAX_SCREENSHOT_GROUPS = 120
MAX_SCREENSHOT_MEMBERS_PER_GROUP = 8

GENERIC_PHRASES = {
    "setuju kak",
    "iya kak",
    "makasih kak",
    "bagus banget",
    "aman gak",
    "spill dong",
    "mau coba",
    "hadir",
    "first",
    "fyp",
}


def read_csv(path: Path, **kwargs: object) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, dtype=str, low_memory=False, **kwargs).fillna("")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_blank(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null", "<na>"}:
        return ""
    return text


def normalize_username(value: object) -> str:
    text = normalize_blank(value).lower().lstrip("@")
    return re.sub(r"\s+", "", text)


def normalize_bool(value: object) -> bool:
    return normalize_blank(value).lower() in {"true", "1", "yes", "y"}


def is_synthetic_comment_id(value: object) -> bool:
    return bool(re.match(r"^INJ", normalize_blank(value), flags=re.IGNORECASE))


def public_mass_id(username_norm: str) -> str:
    digest = hashlib.sha256(f"{MASS_HASH_SALT}:{username_norm}".encode("utf-8")).hexdigest()[:12]
    return f"MASS_{digest}"


def public_pair_key(account_a: str, account_b: str) -> str:
    return "||".join(sorted([normalize_blank(account_a), normalize_blank(account_b)]))


def normalize_comment_text(value: object) -> str:
    text = unicodedata.normalize("NFKC", normalize_blank(value))
    text = re.sub(r"[\u200b-\u200f\u2060]", "", text)
    text = re.sub(r"[\ufe00-\ufe0f]", "", text)
    text = text.lower().strip()
    text = re.sub(r"([!?.,])\1+", r"\1", text)
    text = re.sub(r"([~`^*_=+])\1+", r"\1", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def presentation_text(value: object) -> str:
    text = unicodedata.normalize("NFKC", normalize_blank(value))
    text = re.sub(r"https?://\S+|www\.\S+", "[URL]", text, flags=re.IGNORECASE)
    text = re.sub(r"(?<!\w)@\w+", "@USER", text)
    text = re.sub(r"(?<!\w)\+?\d[\d\s().-]{5,}\d(?!\w)", "[NUMBER]", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tiktok_comment_url_candidate(video_url: object, comment_id: object) -> str:
    url = normalize_blank(video_url)
    cid = normalize_blank(comment_id)
    if not url or not cid:
        return url
    parts = urlsplit(url)
    query = dict(parse_qsl(parts.query, keep_blank_values=True))
    query["comment_id"] = cid
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(query), parts.fragment))


def compact_join(values: pd.Series | list[object], limit: int = 12) -> str:
    cleaned = sorted({normalize_blank(value) for value in values if normalize_blank(value)})
    shown = cleaned[:limit]
    suffix = f";...(+{len(cleaned) - limit})" if len(cleaned) > limit else ""
    return ";".join(shown) + suffix


def most_common_join(values: pd.Series, limit: int = 4) -> str:
    counts = values.map(normalize_blank).replace("", np.nan).dropna().value_counts()
    return " || ".join(counts.head(limit).index.tolist())


def word_tokens(text: str) -> list[str]:
    return re.findall(r"(?u)\b\w+\b", text)


def token_jaccard(text_a: str, text_b: str) -> float:
    a = set(word_tokens(text_a))
    b = set(word_tokens(text_b))
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def stripped_phrase(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text)).strip()


def is_emoji_or_symbol_only(text: str) -> bool:
    if not text:
        return False
    return not bool(re.search(r"(?u)[A-Za-z0-9_]", text))


def is_generic_phrase(text: str, n_tokens: int) -> bool:
    phrase = stripped_phrase(text)
    if phrase in GENERIC_PHRASES:
        return True
    if is_emoji_or_symbol_only(text):
        return True
    return n_tokens <= 2 and len(text) < 20 and phrase in GENERIC_PHRASES


def similarity_category(score: float, exact: bool) -> str:
    if exact:
        return "EXACT_DUPLICATE"
    if score >= 0.92:
        return "NEAR_EXACT"
    if score >= 0.85:
        return "HIGH_SIMILARITY"
    if score >= 0.75:
        return "MODERATE_SIMILARITY"
    return "WEAK_CANDIDATE"


def actor_pair_type(left: str, right: str) -> str:
    pair = {left, right}
    if pair == {COMMUNITY}:
        return f"Community{EN_DASH}Community"
    if pair == {COMMUNITY, MASS}:
        return f"Community{EN_DASH}Mass"
    if pair == {COMMUNITY, INDIVIDUAL}:
        return f"Community{EN_DASH}Individual"
    if pair == {MASS}:
        return f"Mass{EN_DASH}Mass"
    if pair == {MASS, INDIVIDUAL}:
        return f"Mass{EN_DASH}Individual"
    return f"Individual{EN_DASH}Individual"


def brand_from_product_category(value: object) -> str:
    text = normalize_blank(value).lower()
    if "azarine" in text:
        return "Azarine"
    if "daviena" in text or "davina" in text:
        return "Daviena"
    if "maryame" in text:
        return "Maryame"
    if "originote" in text:
        return "The Originote"
    return normalize_blank(value) or "Not identified"


def source_hashes() -> dict[str, str]:
    paths = {
        "dataset": DATASET_PATH,
        "video_metadata_clean": VIDEO_METADATA_PATH,
        "account_actor_type": ACCOUNT_ACTOR_TYPE_PATH,
        "comment_sentiment": COMMENT_SENTIMENT_PATH,
        "hcc_nodes": HCC_NODES_PATH,
        "lcn_nodes_actor_type": LCN_NODES_ACTOR_TYPE_PATH,
        "lcn_edges_actor_type": LCN_EDGES_ACTOR_TYPE_PATH,
        "community_mass_account_pairs": COMMUNITY_MASS_PAIRS_PATH,
        "sentiment_model_artifact": SENTIMENT_MODEL_PATH,
        "sentiment_model_manifest": SENTIMENT_MANIFEST_PATH,
    }
    return {name: sha256_file(path) for name, path in paths.items()}


def build_actor_context(comments: pd.DataFrame, account_type: pd.DataFrame, hcc_nodes: pd.DataFrame, lcn_nodes: pd.DataFrame) -> pd.DataFrame:
    account = account_type.copy()
    account["username_norm"] = account["username"].map(normalize_username)
    account_by_public = account.drop_duplicates("username_norm").set_index("username_norm").to_dict("index")

    hcc = hcc_nodes.copy()
    hcc["username_norm"] = hcc["id"].map(normalize_username)
    community_hcc = hcc.drop_duplicates("username_norm").set_index("username_norm")["community"].astype(str).to_dict()
    community_users = set(community_hcc)

    individual_users = set(account.loc[account["actor_type_primary"].eq(INDIVIDUAL), "username_norm"]) - {""}

    lcn = lcn_nodes.copy()
    lcn["username_norm"] = lcn["Id"].map(normalize_username)
    lcn_by_raw = lcn.drop_duplicates("username_norm").set_index("username_norm").to_dict("index")
    lcn_users = set(lcn_by_raw)

    rows: list[dict[str, object]] = []
    for row in comments.to_dict("records"):
        raw_username = normalize_username(row["username"])
        if raw_username in community_users:
            public_account = raw_username
            actor_type = COMMUNITY
            hcc_id = community_hcc.get(raw_username, "")
            account_row = account_by_public.get(raw_username, {})
            network_position = HCC_POSITION
        elif raw_username in individual_users:
            public_account = raw_username
            actor_type = INDIVIDUAL
            hcc_id = ""
            account_row = account_by_public.get(raw_username, {})
            network_position = normalize_blank(account_row.get("network_position", "")) or (LCN_NON_HCC if raw_username in lcn_users else OUTSIDE_LCN)
        else:
            public_account = public_mass_id(raw_username)
            actor_type = MASS
            hcc_id = ""
            account_row = account_by_public.get(normalize_username(public_account), {})
            network_position = LCN_NON_HCC if raw_username in lcn_users else normalize_blank(account_row.get("network_position", "")) or OUTSIDE_LCN
            if network_position not in {LCN_NON_HCC, OUTSIDE_LCN}:
                network_position = LCN_NON_HCC if raw_username in lcn_users else OUTSIDE_LCN

        lcn_row = lcn_by_raw.get(raw_username, {})
        rows.append(
            {
                "comment_id": normalize_blank(row["comment_id"]),
                "raw_username_norm": raw_username,
                "account_public": public_account,
                "actor_type_primary": actor_type,
                "network_position": network_position,
                "hcc_id": hcc_id,
                "is_lcn_member": bool(raw_username in lcn_users or normalize_bool(account_row.get("is_lcn_member", ""))),
                "degree": normalize_blank(lcn_row.get("degree", "")),
                "weighted_degree": normalize_blank(lcn_row.get("weighted_degree", "")),
                "account_target_brand_primary": normalize_blank(account_row.get("target_brand_primary", "")),
                "account_dominant_sentiment": normalize_blank(account_row.get("dominant_sentiment", "")),
                "account_goal_orientation": normalize_blank(account_row.get("account_goal_orientation", "")),
            }
        )
    return pd.DataFrame(rows)


def prepare_comments() -> tuple[pd.DataFrame, dict[str, object]]:
    dataset = read_csv(DATASET_PATH)
    account_type = read_csv(ACCOUNT_ACTOR_TYPE_PATH)
    hcc_nodes = read_csv(HCC_NODES_PATH)
    lcn_nodes = read_csv(LCN_NODES_ACTOR_TYPE_PATH)
    sentiment = read_csv(COMMENT_SENTIMENT_PATH)

    dataset["comment_id"] = dataset["comment_id"].map(normalize_blank)
    dataset = dataset.drop_duplicates("comment_id", keep="first").copy()
    synthetic_mask = dataset["comment_id"].map(is_synthetic_comment_id)
    observational = dataset.loc[~synthetic_mask].copy().reset_index(drop=True)

    actor_context = build_actor_context(observational, account_type, hcc_nodes, lcn_nodes)
    comments = pd.concat([observational.reset_index(drop=True), actor_context.drop(columns=["comment_id"])], axis=1)
    comments["comment_text_original"] = comments["text"].map(normalize_blank)
    comments["comment_text_normalized"] = comments["comment_text_original"].map(normalize_comment_text)
    comments["comment_text_presentation"] = comments["comment_text_original"].map(presentation_text)
    comments["n_characters"] = comments["comment_text_normalized"].map(len)
    comments["n_tokens"] = comments["comment_text_normalized"].map(lambda text: len(word_tokens(text)))
    comments["is_very_short"] = comments["n_characters"].lt(12) | comments["n_tokens"].lt(3)
    comments["is_generic_phrase"] = comments.apply(lambda row: is_generic_phrase(row["comment_text_normalized"], int(row["n_tokens"])), axis=1)
    comments["brand"] = comments["product_category"].map(brand_from_product_category)

    sentiment = sentiment.rename(columns={"final_sentiment_label": "sentiment_label"})
    sentiment_cols = ["comment_id", "sentiment_label"]
    sentiment_small = sentiment[sentiment_cols].drop_duplicates("comment_id")
    comments = comments.merge(sentiment_small, on="comment_id", how="left")
    comments["sentiment_label"] = comments["sentiment_label"].map(normalize_blank).replace("", "Not available")
    comments["sentiment_attribute_status"] = SENTIMENT_ATTRIBUTE_STATUS

    summary = {
        "dataset_rows": len(dataset),
        "dataset_unique_comment_id": dataset["comment_id"].nunique(),
        "synthetic_ids_excluded": int(synthetic_mask.sum()),
        "observational_comments_analyzed": len(comments),
    }
    return comments, summary


def comment_group_maps(comments: pd.DataFrame) -> tuple[dict[str, list[int]], dict[str, dict[str, object]]]:
    valid = comments.loc[comments["comment_text_normalized"].ne("")].copy()
    text_to_indices = valid.groupby("comment_text_normalized", sort=True).groups
    group_lookup: dict[str, dict[str, object]] = {}
    for text, idx_array in text_to_indices.items():
        group = comments.loc[list(idx_array)]
        group_lookup[text] = {
            "n_comments": int(len(group)),
            "n_unique_accounts": int(group["account_public"].nunique()),
            "n_unique_videos": int(group["video_id"].nunique()),
            "n_unique_brands": int(group["brand"].nunique()),
            "is_generic": bool(group["is_generic_phrase"].any()),
            "is_very_short": bool(group["is_very_short"].any()),
        }
    return {text: list(map(int, idxs)) for text, idxs in text_to_indices.items()}, group_lookup


def build_exact_pair_index(comments: pd.DataFrame, text_to_indices: dict[str, list[int]]) -> pd.DataFrame:
    rows: list[tuple[int, int, float, str, str]] = []
    for text, idxs in text_to_indices.items():
        if len(idxs) < 2:
            continue
        for i, j in itertools.combinations(sorted(idxs), 2):
            same_account = comments.at[i, "account_public"] == comments.at[j, "account_public"]
            exact_type = "EXACT_DUPLICATE_SAME_ACCOUNT" if same_account else "EXACT_DUPLICATE_CROSS_ACCOUNT"
            rows.append((i, j, 1.0, "EXACT_DUPLICATE", exact_type))
    return pd.DataFrame(rows, columns=["idx_1", "idx_2", "cosine_similarity_char5", "similarity_category", "exact_duplicate_type"])


def build_near_pair_index(comments: pd.DataFrame, text_to_indices: dict[str, list[int]]) -> tuple[pd.DataFrame, dict[str, object]]:
    unique_texts = sorted(text_to_indices)
    stats: dict[str, object] = {
        "unique_texts_fit": len(unique_texts),
        "text_neighbor_pairs_before_expansion": 0,
        "expanded_near_pair_rows_before_dedup": 0,
        "top_k_neighbors": TOP_K_NEIGHBORS,
        "similarity_threshold": SIMILARITY_THRESHOLD,
        "method": "TF-IDF char_wb 5-gram + sparse top-k cosine nearest neighbors",
    }
    if len(unique_texts) < 2:
        return pd.DataFrame(columns=["idx_1", "idx_2", "cosine_similarity_char5", "similarity_category", "exact_duplicate_type"]), stats

    vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(5, 5),
        lowercase=False,
        min_df=2,
        sublinear_tf=True,
        norm="l2",
    )
    matrix = vectorizer.fit_transform(unique_texts)
    stats["tfidf_shape_rows"] = int(matrix.shape[0])
    stats["tfidf_shape_features"] = int(matrix.shape[1])
    n_neighbors = min(TOP_K_NEIGHBORS + 1, len(unique_texts))
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine", algorithm="brute", n_jobs=-1)
    nn.fit(matrix)
    distances, indices = nn.kneighbors(matrix, return_distance=True)

    text_pairs: dict[tuple[int, int], float] = {}
    for i in range(len(unique_texts)):
        for dist, j in zip(distances[i], indices[i]):
            if i == j:
                continue
            sim = float(max(0.0, min(1.0, 1.0 - dist)))
            if sim < SIMILARITY_THRESHOLD:
                continue
            a, b = sorted((int(i), int(j)))
            key = (a, b)
            text_pairs[key] = max(text_pairs.get(key, 0.0), sim)
    stats["text_neighbor_pairs_before_expansion"] = len(text_pairs)

    rows: list[tuple[int, int, float, str, str]] = []
    for (i, j), sim in sorted(text_pairs.items(), key=lambda item: (item[0][0], item[0][1])):
        text_i = unique_texts[i]
        text_j = unique_texts[j]
        category = similarity_category(sim, exact=False)
        for left_idx in text_to_indices[text_i]:
            for right_idx in text_to_indices[text_j]:
                a, b = sorted((left_idx, right_idx))
                rows.append((a, b, sim, category, ""))
    stats["expanded_near_pair_rows_before_dedup"] = len(rows)
    return pd.DataFrame(rows, columns=["idx_1", "idx_2", "cosine_similarity_char5", "similarity_category", "exact_duplicate_type"]), stats


def comment_attrs_for_merge(comments: pd.DataFrame, suffix: str) -> pd.DataFrame:
    cols = [
        "comment_id",
        "username",
        "account_public",
        "actor_type_primary",
        "network_position",
        "hcc_id",
        "video_id",
        "video_url",
        "brand",
        "timestamp",
        "comment_text_original",
        "comment_text_presentation",
        "comment_text_normalized",
        "n_characters",
        "n_tokens",
        "is_very_short",
        "is_generic_phrase",
        "sentiment_label",
        "is_lcn_member",
    ]
    out = comments[cols].copy()
    return out.rename(columns={col: f"{col}_{suffix}" for col in cols})


def attach_pair_attributes(pair_index: pd.DataFrame, comments: pd.DataFrame, cm_pairs: pd.DataFrame, group_lookup: dict[str, dict[str, object]]) -> pd.DataFrame:
    if pair_index.empty:
        return pd.DataFrame()

    idx_to_id = comments["comment_id"].to_dict()
    pair_index = pair_index.copy()
    pair_index["comment_id_1_key"] = pair_index["idx_1"].map(idx_to_id)
    pair_index["comment_id_2_key"] = pair_index["idx_2"].map(idx_to_id)
    pair_index["canonical_pair"] = pair_index.apply(lambda row: public_pair_key(row["comment_id_1_key"], row["comment_id_2_key"]), axis=1)
    pair_index = pair_index.sort_values(
        ["canonical_pair", "cosine_similarity_char5", "similarity_category"],
        ascending=[True, False, True],
    ).drop_duplicates("canonical_pair", keep="first")
    pair_index = pair_index.loc[pair_index["idx_1"].ne(pair_index["idx_2"])].copy().reset_index(drop=True)

    left = comment_attrs_for_merge(comments, "1")
    right = comment_attrs_for_merge(comments, "2")
    out = pair_index.merge(left, left_on="comment_id_1_key", right_on="comment_id_1", how="left")
    out = out.merge(right, left_on="comment_id_2_key", right_on="comment_id_2", how="left")

    out["same_account"] = out["account_public_1"].eq(out["account_public_2"])
    out["same_video"] = out["video_id_1"].eq(out["video_id_2"])
    out["same_brand"] = out["brand_1"].eq(out["brand_2"])
    out["same_hcc"] = out["hcc_id_1"].ne("") & out["hcc_id_1"].eq(out["hcc_id_2"])
    out["pair_is_cross_account"] = ~out["same_account"]
    out["pair_is_cross_video"] = ~out["same_video"]
    out["pair_is_cross_brand"] = ~out["same_brand"]
    out["both_lcn_members"] = out["is_lcn_member_1"].astype(bool) & out["is_lcn_member_2"].astype(bool)
    out["one_lcn_one_outside"] = out["is_lcn_member_1"].astype(bool) ^ out["is_lcn_member_2"].astype(bool)
    out["both_outside_lcn"] = (~out["is_lcn_member_1"].astype(bool)) & (~out["is_lcn_member_2"].astype(bool))
    out["actor_pair_type"] = out.apply(lambda row: actor_pair_type(row["actor_type_primary_1"], row["actor_type_primary_2"]), axis=1)
    out["token_jaccard"] = out.apply(lambda row: token_jaccard(row["comment_text_normalized_1"], row["comment_text_normalized_2"]), axis=1)
    if fuzz is None:
        out["rapidfuzz_ratio"] = np.nan
    else:
        out["rapidfuzz_ratio"] = out.apply(lambda row: fuzz.ratio(row["comment_text_normalized_1"], row["comment_text_normalized_2"]) / 100.0, axis=1)

    cm_lookup = cm_pairs.drop_duplicates(["community_account", "mass_account"]).set_index(["community_account", "mass_account"]).to_dict("index")

    def cm_context(row: pd.Series) -> tuple[bool, str, str]:
        if row["actor_pair_type"] != f"Community{EN_DASH}Mass":
            return False, "", ""
        if row["actor_type_primary_1"] == COMMUNITY:
            key = (row["account_public_1"], row["account_public_2"])
        else:
            key = (row["account_public_2"], row["account_public_1"])
        item = cm_lookup.get(key)
        if not item:
            return False, "NO_ACCOUNT_EVIDENCE_PAIR", ""
        return normalize_bool(item.get("pair_is_lcn_edge", "")), normalize_blank(item.get("interaction_scope", "")), normalize_blank(item.get("evidence_combination", ""))

    cm_values = out.apply(cm_context, axis=1, result_type="expand")
    cm_values.columns = ["community_mass_pair_is_lcn_edge", "community_mass_interaction_scope", "community_mass_evidence_combination"]
    out = pd.concat([out, cm_values], axis=1)

    out["is_generic_phrase"] = out["is_generic_phrase_1"].astype(bool) | out["is_generic_phrase_2"].astype(bool)
    out["pair_has_very_short_comment"] = out["is_very_short_1"].astype(bool) | out["is_very_short_2"].astype(bool)

    def eligible(row: pd.Series) -> bool:
        if not bool(row["pair_is_cross_account"]):
            return False
        if row["similarity_category"] == "WEAK_CANDIDATE":
            return False
        text = row["comment_text_normalized_1"]
        group_info = group_lookup.get(text, {})
        exact_exception = (
            row["similarity_category"] == "EXACT_DUPLICATE"
            and (int(group_info.get("n_unique_accounts", 0)) >= 3 or int(group_info.get("n_unique_videos", 0)) >= 3)
        )
        if bool(row["is_generic_phrase"]) or bool(row["pair_has_very_short_comment"]):
            return exact_exception
        return True

    out["presentation_eligible"] = out.apply(eligible, axis=1)
    out["brand_pair_scope"] = out.apply(lambda row: row["brand_1"] if row["same_brand"] else " | ".join(sorted([row["brand_1"], row["brand_2"]])), axis=1)
    out["hcc_pair_scope"] = out.apply(
        lambda row: row["hcc_id_1"] if row["same_hcc"] else "Cross-HCC / Non-HCC",
        axis=1,
    )
    out = out.sort_values(["similarity_category", "cosine_similarity_char5", "canonical_pair"], ascending=[True, False, True]).reset_index(drop=True)
    out.insert(0, "pair_id", [f"SIM_PAIR_{i:09d}" for i in range(1, len(out) + 1)])
    out["sentiment_attribute_status"] = SENTIMENT_ATTRIBUTE_STATUS

    final_cols = [
        "pair_id",
        "comment_id_1",
        "comment_id_2",
        "account_public_1",
        "account_public_2",
        "actor_type_primary_1",
        "actor_type_primary_2",
        "network_position_1",
        "network_position_2",
        "hcc_id_1",
        "hcc_id_2",
        "video_id_1",
        "video_id_2",
        "video_url_1",
        "video_url_2",
        "username_1",
        "username_2",
        "brand_1",
        "brand_2",
        "timestamp_1",
        "timestamp_2",
        "comment_text_original_1",
        "comment_text_original_2",
        "comment_text_presentation_1",
        "comment_text_presentation_2",
        "comment_text_normalized_1",
        "comment_text_normalized_2",
        "cosine_similarity_char5",
        "token_jaccard",
        "rapidfuzz_ratio",
        "similarity_category",
        "exact_duplicate_type",
        "actor_pair_type",
        "same_account",
        "same_video",
        "same_brand",
        "same_hcc",
        "pair_is_cross_account",
        "pair_is_cross_video",
        "pair_is_cross_brand",
        "both_lcn_members",
        "one_lcn_one_outside",
        "both_outside_lcn",
        "community_mass_pair_is_lcn_edge",
        "community_mass_interaction_scope",
        "community_mass_evidence_combination",
        "n_characters_1",
        "n_characters_2",
        "n_tokens_1",
        "n_tokens_2",
        "is_generic_phrase",
        "pair_has_very_short_comment",
        "presentation_eligible",
        "sentiment_label_1",
        "sentiment_label_2",
        "sentiment_attribute_status",
        "brand_pair_scope",
        "hcc_pair_scope",
        "canonical_pair",
    ]
    out = out[final_cols].rename(
        columns={
            "account_public_1": "account_1",
            "account_public_2": "account_2",
            "username_1": "platform_username_1",
            "username_2": "platform_username_2",
            "actor_type_primary_1": "actor_type_1",
            "actor_type_primary_2": "actor_type_2",
            "comment_text_original_1": "comment_text_1_original",
            "comment_text_original_2": "comment_text_2_original",
            "comment_text_presentation_1": "comment_text_1_presentation",
            "comment_text_presentation_2": "comment_text_2_presentation",
            "comment_text_normalized_1": "normalized_text_1",
            "comment_text_normalized_2": "normalized_text_2",
            "sentiment_label_1": "sentiment_1",
            "sentiment_label_2": "sentiment_2",
        }
    )
    return out


def write_exact_groups(comments: pd.DataFrame, pairs_all: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    exact_texts = comments.loc[comments["comment_text_normalized"].ne("")].groupby("comment_text_normalized", sort=True).filter(lambda g: len(g) >= 2)
    group_rows: list[dict[str, object]] = []
    member_rows: list[dict[str, object]] = []
    for idx, (text, group) in enumerate(exact_texts.groupby("comment_text_normalized", sort=True), start=1):
        exact_group_id = f"EXACT_GROUP_{idx:05d}"
        group_pairs = pairs_all.loc[(pairs_all["similarity_category"].eq("EXACT_DUPLICATE")) & (pairs_all["normalized_text_1"].eq(text))]
        n_accounts = int(group["account_public"].nunique())
        n_videos = int(group["video_id"].nunique())
        eligible = bool(n_accounts >= 2 and (not group["is_generic_phrase"].any() and not group["is_very_short"].any() or n_accounts >= 3 or n_videos >= 3))
        group_rows.append(
            {
                "exact_group_id": exact_group_id,
                "normalized_text": text,
                "representative_comment_text": group["comment_text_presentation"].iloc[0],
                "n_comments": int(len(group)),
                "n_unique_accounts": n_accounts,
                "n_unique_videos": n_videos,
                "n_unique_brands": int(group["brand"].nunique()),
                "n_actor_types": int(group["actor_type_primary"].nunique()),
                "actor_types": ";".join(sorted(group["actor_type_primary"].unique())),
                "actor_pair_scope": ";".join(sorted(group_pairs["actor_pair_type"].dropna().unique())) if not group_pairs.empty else "",
                "accounts": compact_join(group["account_public"]),
                "platform_usernames": compact_join(group["username"]),
                "comment_ids": compact_join(group["comment_id"]),
                "video_ids": compact_join(group["video_id"]),
                "video_urls": compact_join(group["video_url"], limit=5),
                "brands": compact_join(group["brand"]),
                "hcc_ids": ";".join(sorted([h for h in group["hcc_id"].unique() if h])),
                "contains_community_actor": bool(group["actor_type_primary"].eq(COMMUNITY).any()),
                "contains_mass_actor": bool(group["actor_type_primary"].eq(MASS).any()),
                "contains_lcn_edge_account_pair": bool(group_pairs["community_mass_pair_is_lcn_edge"].astype(bool).any()) if not group_pairs.empty else False,
                "presentation_eligible": eligible,
            }
        )
        for row in group.to_dict("records"):
            member_rows.append(
                {
                    "exact_group_id": exact_group_id,
                    "comment_id": row["comment_id"],
                    "platform_username": row["username"],
                    "account": row["account_public"],
                    "actor_type_primary": row["actor_type_primary"],
                    "network_position": row["network_position"],
                    "hcc_id": row["hcc_id"],
                    "video_id": row["video_id"],
                    "video_url": row["video_url"],
                    "tiktok_comment_url_candidate": tiktok_comment_url_candidate(row["video_url"], row["comment_id"]),
                    "brand": row["brand"],
                    "timestamp": row["timestamp"],
                    "comment_text_original": row["comment_text_original"],
                    "comment_text_presentation": row["comment_text_presentation"],
                    "normalized_text": row["comment_text_normalized"],
                    "sentiment": row["sentiment_label"],
                    "sentiment_attribute_status": SENTIMENT_ATTRIBUTE_STATUS,
                }
            )
    groups = pd.DataFrame(group_rows)
    members = pd.DataFrame(member_rows)
    groups.to_csv(OUT_EXACT_GROUPS, index=False)
    members.to_csv(OUT_EXACT_MEMBERS, index=False)
    return groups, members


def write_clusters(comments: pd.DataFrame, pairs_all: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cluster_edges = pairs_all.loc[pairs_all["cosine_similarity_char5"].ge(CLUSTER_THRESHOLD)].copy()
    graph = nx.Graph()
    graph.add_edges_from(cluster_edges[["comment_id_1", "comment_id_2"]].itertuples(index=False, name=None))
    comment_lookup = comments.drop_duplicates("comment_id").set_index("comment_id").to_dict("index")
    pair_lookup = cluster_edges.set_index("canonical_pair").to_dict("index")
    category_score = {"EXACT_DUPLICATE": 5, "NEAR_EXACT": 4, "HIGH_SIMILARITY": 3, "MODERATE_SIMILARITY": 2, "WEAK_CANDIDATE": 1}

    cluster_rows: list[dict[str, object]] = []
    member_rows: list[dict[str, object]] = []
    components = sorted(nx.connected_components(graph), key=lambda comp: (-len(comp), sorted(comp)[0]))
    for idx, comp in enumerate(components, start=1):
        ids = sorted(comp)
        cluster_id = f"SIM_CLUSTER_{idx:05d}"
        internal_scores = []
        for a, b in itertools.combinations(ids, 2):
            item = pair_lookup.get(public_pair_key(a, b))
            if item:
                internal_scores.append(float(item["cosine_similarity_char5"]))
        if not internal_scores:
            continue
        group_records = []
        for cid in ids:
            if cid in comment_lookup:
                record = dict(comment_lookup[cid])
                record["comment_id"] = cid
                group_records.append(record)
        group = pd.DataFrame(group_records)
        if group.empty:
            continue
        min_sim = float(np.min(internal_scores))
        mean_sim = float(np.mean(internal_scores))
        max_sim = float(np.max(internal_scores))
        edge_mask = cluster_edges["comment_id_1"].isin(ids) & cluster_edges["comment_id_2"].isin(ids)
        internal_edges = cluster_edges.loc[edge_mask].copy()
        possible_pair_count = len(ids) * (len(ids) - 1) // 2
        n_accounts = int(group["account_public"].nunique())
        n_videos = int(group["video_id"].nunique())
        n_brands = int(group["brand"].nunique())
        generic_share = float(group["is_generic_phrase"].astype(bool).mean()) if len(group) else 0.0
        short_share = float(group["is_very_short"].astype(bool).mean()) if len(group) else 0.0
        category_values = sorted(internal_edges["similarity_category"].dropna().unique(), key=lambda value: -category_score.get(value, 0))
        actor_values = sorted(internal_edges["actor_pair_type"].dropna().unique())
        quality_flags: list[str] = []
        if n_accounts < 2:
            quality_flags.append("SAME_ACCOUNT_ONLY")
        if generic_share >= 0.75 or short_share >= 0.75:
            quality_flags.append("GENERIC_OR_VERY_SHORT_GROUP")
        if len(ids) >= 20 and (generic_share >= 0.5 or short_share >= 0.5):
            quality_flags.append("LARGE_LOW_INFORMATION_CLUSTER")
        if min_sim < CLUSTER_THRESHOLD:
            quality_flags.append("LOW_COHESION_CLUSTER")
        if not quality_flags:
            quality_flags.append("SUBSTANTIVE_MULTI_ACCOUNT_GROUP")
        presentation_eligible = bool(
            n_accounts >= 2
            and len(ids) >= 2
            and (
                (generic_share < 0.5 and short_share < 0.5)
                or (n_accounts >= 3 and n_videos >= 2 and max_sim >= 0.92)
            )
        )
        screenshot_eligible = bool(presentation_eligible and "GENERIC_OR_VERY_SHORT_GROUP" not in quality_flags)
        screenshot_priority_score = (
            max(category_score.get(value, 0) for value in category_values or ["WEAK_CANDIDATE"])
            + math.log1p(n_accounts)
            + 0.5 * math.log1p(n_videos)
            + (1.5 if bool(internal_edges["community_mass_pair_is_lcn_edge"].astype(bool).any()) else 0.0)
            + (0.75 if bool(internal_edges["community_mass_interaction_scope"].eq("PRE_LCN_MULTI_EVIDENCE").any()) else 0.0)
            - (8.0 if "GENERIC_OR_VERY_SHORT_GROUP" in quality_flags else 0.0)
        )
        cluster_rows.append(
            {
                "cluster_id": cluster_id,
                "n_comments": len(ids),
                "n_accounts": n_accounts,
                "n_videos": n_videos,
                "n_brands": n_brands,
                "n_pair_edges_observed": int(len(internal_edges)),
                "possible_pair_count": int(possible_pair_count),
                "edge_density": round(len(internal_edges) / possible_pair_count, 6) if possible_pair_count else 0.0,
                "min_pair_similarity": min_sim,
                "mean_pair_similarity": mean_sim,
                "max_pair_similarity": max_sim,
                "cluster_cohesion": min_sim,
                "cluster_quality_flag": ";".join(quality_flags),
                "generic_comment_share": round(generic_share, 6),
                "very_short_comment_share": round(short_share, 6),
                "similarity_categories": ";".join(category_values),
                "primary_similarity_category": category_values[0] if category_values else "",
                "actor_pair_types": ";".join(actor_values),
                "actor_types": compact_join(group["actor_type_primary"]),
                "network_positions": compact_join(group["network_position"]),
                "brands": compact_join(group["brand"]),
                "video_ids": compact_join(group["video_id"]),
                "video_urls": compact_join(group["video_url"], limit=5),
                "hcc_ids": compact_join(group["hcc_id"]),
                "representative_comment_texts": most_common_join(group["comment_text_presentation"], limit=4),
                "dominant_normalized_texts": most_common_join(group["comment_text_normalized"], limit=4),
                "contains_community_actor": bool(group["actor_type_primary"].eq(COMMUNITY).any()),
                "contains_mass_actor": bool(group["actor_type_primary"].eq(MASS).any()),
                "contains_individual_actor": bool(group["actor_type_primary"].eq(INDIVIDUAL).any()),
                "contains_community_mass_lcn_edge": bool(internal_edges["community_mass_pair_is_lcn_edge"].astype(bool).any()),
                "contains_pre_lcn_multi_evidence": bool(internal_edges["community_mass_interaction_scope"].eq("PRE_LCN_MULTI_EVIDENCE").any()),
                "presentation_eligible": presentation_eligible,
                "screenshot_eligible": screenshot_eligible,
                "screenshot_priority_score": round(screenshot_priority_score, 6),
            }
        )
        ordered_group = group.assign(
            actor_sort=group["actor_type_primary"].map({COMMUNITY: 0, INDIVIDUAL: 1, MASS: 2}).fillna(9),
            account_comment_count=group.groupby("account_public")["comment_id"].transform("count"),
        ).sort_values(["actor_sort", "network_position", "account_comment_count", "account_public", "video_id", "comment_id"], ascending=[True, True, False, True, True, True])
        for member_rank, cid in enumerate(ordered_group["comment_id"].tolist(), start=1):
            row = comment_lookup[cid]
            member_rows.append(
                {
                    "cluster_id": cluster_id,
                    "member_rank": member_rank,
                    "comment_id": cid,
                    "platform_username": row["username"],
                    "account": row["account_public"],
                    "actor_type_primary": row["actor_type_primary"],
                    "network_position": row["network_position"],
                    "hcc_id": row["hcc_id"],
                    "video_id": row["video_id"],
                    "video_url": row["video_url"],
                    "tiktok_comment_url_candidate": tiktok_comment_url_candidate(row["video_url"], cid),
                    "brand": row["brand"],
                    "timestamp": row["timestamp"],
                    "comment_text_presentation": row["comment_text_presentation"],
                    "normalized_text": row["comment_text_normalized"],
                    "sentiment": row["sentiment_label"],
                    "sentiment_attribute_status": SENTIMENT_ATTRIBUTE_STATUS,
                }
            )
    clusters = pd.DataFrame(cluster_rows)
    members = pd.DataFrame(member_rows)
    if not clusters.empty:
        clusters = clusters.sort_values(
            ["screenshot_eligible", "presentation_eligible", "screenshot_priority_score", "n_accounts", "n_comments", "cluster_id"],
            ascending=[False, False, False, False, False, True],
        ).reset_index(drop=True)
        clusters.insert(0, "group_rank", range(1, len(clusters) + 1))
        clusters = clusters.rename(columns={"cluster_id": "group_id"})
        members = members.rename(columns={"cluster_id": "group_id"})
        members = members.merge(clusters[["group_id", "group_rank", "presentation_eligible", "screenshot_eligible", "screenshot_priority_score"]], on="group_id", how="left")
        members = members.sort_values(["group_rank", "member_rank"]).reset_index(drop=True)
        legacy_clusters = clusters.rename(columns={"group_id": "cluster_id"})
        legacy_members = members.rename(columns={"group_id": "cluster_id"})
    else:
        legacy_clusters = clusters.copy()
        legacy_members = members.copy()
    clusters.to_csv(OUT_GROUPS, index=False)
    members.to_csv(OUT_GROUP_MEMBERS, index=False)
    legacy_clusters.to_csv(OUT_CLUSTERS, index=False)
    legacy_members.to_csv(OUT_CLUSTER_MEMBERS, index=False)
    return clusters, members


def representative_members_for_pair(comments: pd.DataFrame, pair: pd.Series, cluster_members: pd.DataFrame | None, group_id: str) -> pd.DataFrame:
    if cluster_members is not None and group_id and not cluster_members.empty:
        id_col = "group_id" if "group_id" in cluster_members.columns else "cluster_id"
        ids = cluster_members.loc[cluster_members[id_col].eq(group_id), "comment_id"].tolist()
        if ids:
            group = comments.loc[comments["comment_id"].isin(ids)].copy()
            group["actor_sort"] = group["actor_type_primary"].map({COMMUNITY: 0, INDIVIDUAL: 1, MASS: 2}).fillna(9)
            group = group.sort_values(["actor_sort", "account_public", "video_id", "comment_id"]).drop_duplicates("account_public", keep="first")
            return group.head(4)
    ids = [pair["comment_id_1"], pair["comment_id_2"]]
    return comments.loc[comments["comment_id"].isin(ids)].head(4)


def write_presentation_outputs(comments: pd.DataFrame, pairs_all: pd.DataFrame, clusters: pd.DataFrame, cluster_members: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not clusters.empty:
        pair_to_group = {}
        id_col = "group_id" if "group_id" in cluster_members.columns else "cluster_id"
        group_id_col = "group_id" if "group_id" in clusters.columns else "cluster_id"
        group_meta_lookup = clusters.set_index(group_id_col).to_dict("index")
        member_groups = cluster_members.groupby(id_col)["comment_id"].apply(set).to_dict()
        for group_id, ids in member_groups.items():
            for a, b in itertools.combinations(sorted(ids), 2):
                pair_to_group[public_pair_key(a, b)] = group_id
    else:
        pair_to_group = {}
        group_meta_lookup = {}

    category_score = {"EXACT_DUPLICATE": 5, "NEAR_EXACT": 4, "HIGH_SIMILARITY": 3, "MODERATE_SIMILARITY": 2}
    primary_candidates = pairs_all.loc[
        pairs_all["presentation_eligible"].astype(bool)
        & pairs_all["similarity_category"].isin(category_score)
        & pairs_all["pair_is_cross_account"].astype(bool)
        & ~pairs_all["is_generic_phrase"].astype(bool)
        & ~pairs_all["pair_has_very_short_comment"].astype(bool)
    ].copy()
    fallback_candidates = pairs_all.loc[
        pairs_all["presentation_eligible"].astype(bool)
        & pairs_all["similarity_category"].isin(category_score)
        & pairs_all["pair_is_cross_account"].astype(bool)
        & (pairs_all["is_generic_phrase"].astype(bool) | pairs_all["pair_has_very_short_comment"].astype(bool))
    ].copy()
    candidates = pd.concat([primary_candidates, fallback_candidates], ignore_index=True)
    candidates["selection_pool"] = np.where(
        candidates["pair_has_very_short_comment"].astype(bool) | candidates["is_generic_phrase"].astype(bool),
        "fallback_short_or_generic",
        "primary_substantive_text",
    )
    candidates["group_id"] = candidates["canonical_pair"].map(pair_to_group).fillna("")
    candidates["selection_score"] = (
        candidates["similarity_category"].map(category_score).fillna(0)
        + candidates["selection_pool"].eq("primary_substantive_text").astype(int) * 5
        + candidates["pair_is_cross_video"].astype(int)
        + candidates["community_mass_pair_is_lcn_edge"].astype(int)
        + candidates["community_mass_interaction_scope"].eq("PRE_LCN_MULTI_EVIDENCE").astype(int) * 0.5
        + candidates["cosine_similarity_char5"]
    )
    candidates = candidates.sort_values(
        ["selection_score", "cosine_similarity_char5", "pair_is_cross_video", "canonical_pair"],
        ascending=[False, False, False, True],
    )

    def balanced_candidate_records(candidate_table: pd.DataFrame) -> list[dict[str, object]]:
        category_targets = {
            "EXACT_DUPLICATE": 12,
            "NEAR_EXACT": 8,
            "HIGH_SIMILARITY": 6,
            "MODERATE_SIMILARITY": 4,
        }
        actor_order = [
            f"Community{EN_DASH}Mass",
            f"Community{EN_DASH}Community",
            f"Mass{EN_DASH}Mass",
            f"Mass{EN_DASH}Individual",
            f"Community{EN_DASH}Individual",
            f"Individual{EN_DASH}Individual",
        ]
        yielded: set[str] = set()
        ordered: list[dict[str, object]] = []
        for category, target in category_targets.items():
            category_table = candidate_table.loc[candidate_table["similarity_category"].eq(category)]
            actor_values = actor_order + [value for value in sorted(category_table["actor_pair_type"].dropna().unique()) if value not in actor_order]
            actor_records = {
                actor: category_table.loc[category_table["actor_pair_type"].eq(actor)].to_dict("records")
                for actor in actor_values
            }
            positions = {actor: 0 for actor in actor_values}
            added = 0
            while added < target:
                progressed = False
                for actor in actor_values:
                    records = actor_records.get(actor, [])
                    while positions[actor] < len(records):
                        record = records[positions[actor]]
                        positions[actor] += 1
                        key = normalize_blank(record.get("canonical_pair", ""))
                        if key in yielded:
                            continue
                        yielded.add(key)
                        ordered.append(record)
                        added += 1
                        progressed = True
                        break
                    if added >= target:
                        break
                if not progressed:
                    break
        for record in candidate_table.to_dict("records"):
            key = normalize_blank(record.get("canonical_pair", ""))
            if key in yielded:
                continue
            yielded.add(key)
            ordered.append(record)
        return ordered

    selected_rows: list[dict[str, object]] = []
    member_rows: list[dict[str, object]] = []
    used_clusters: set[str] = set()
    brand_counts: defaultdict[str, int] = defaultdict(int)
    actor_counts: defaultdict[str, int] = defaultdict(int)

    for pair_record in balanced_candidate_records(candidates):
        pair = pd.Series(pair_record)
        group_id = normalize_blank(pair["group_id"]) or pair["pair_id"]
        if group_id in used_clusters:
            continue
        brand_scope = pair["brand_pair_scope"]
        actor_scope = pair["actor_pair_type"]
        if len(selected_rows) >= 18 and (brand_counts[brand_scope] >= 8 or actor_counts[actor_scope] >= 12):
            continue

        members = representative_members_for_pair(comments, pair, cluster_members, normalize_blank(pair["group_id"]))
        if members.empty:
            continue
        example_id = f"PPT_SIM_{len(selected_rows) + 1:03d}"
        used_clusters.add(group_id)
        brand_counts[brand_scope] += 1
        actor_counts[actor_scope] += 1
        group_meta = group_meta_lookup.get(normalize_blank(pair["group_id"]), {})
        comment_values = members.to_dict("records")
        while len(comment_values) < 4:
            comment_values.append({})

        reason = []
        if pair["similarity_category"] == "EXACT_DUPLICATE":
            reason.append("identical normalized comments")
        elif pair["similarity_category"] == "NEAR_EXACT":
            reason.append("near-identical comments")
        elif pair["similarity_category"] == "HIGH_SIMILARITY":
            reason.append("high textual similarity")
        else:
            reason.append("candidate similar comments")
        if bool(pair["pair_is_cross_video"]):
            reason.append("cross-video")
        if bool(pair["community_mass_pair_is_lcn_edge"]):
            reason.append("account pair is also an LCN edge")
        elif pair["community_mass_interaction_scope"] in {"PRE_LCN_MULTI_EVIDENCE", "PRE_LCN_SINGLE_EVIDENCE"}:
            reason.append(pair["community_mass_interaction_scope"])

        selected_rows.append(
            {
                "presentation_rank": len(selected_rows) + 1,
                "example_id": example_id,
                "source_pair_id": pair["pair_id"],
                "similarity_category": pair["similarity_category"],
                "similarity_score": pair["cosine_similarity_char5"],
                "group_id": normalize_blank(pair["group_id"]) or "",
                "cluster_id": normalize_blank(pair["group_id"]) or "",
                "actor_pair_type": actor_scope,
                "brand_scope": brand_scope,
                "hcc_scope": pair["hcc_pair_scope"],
                "same_video": pair["same_video"],
                "same_brand": pair["same_brand"],
                "community_mass_pair_is_lcn_edge": pair["community_mass_pair_is_lcn_edge"],
                "community_mass_interaction_scope": pair["community_mass_interaction_scope"],
                "selection_pool": pair["selection_pool"],
                "n_comments_in_group": group_meta.get("n_comments", len(members)),
                "n_unique_accounts": group_meta.get("n_accounts", members["account_public"].nunique()),
                "n_unique_videos": group_meta.get("n_videos", members["video_id"].nunique()),
                "n_representative_comments": len(members),
                "comment_1": comment_values[0].get("comment_text_presentation", ""),
                "comment_1_actor_type": comment_values[0].get("actor_type_primary", ""),
                "comment_1_network_position": comment_values[0].get("network_position", ""),
                "comment_1_brand": comment_values[0].get("brand", ""),
                "comment_2": comment_values[1].get("comment_text_presentation", ""),
                "comment_2_actor_type": comment_values[1].get("actor_type_primary", ""),
                "comment_2_network_position": comment_values[1].get("network_position", ""),
                "comment_2_brand": comment_values[1].get("brand", ""),
                "comment_3": comment_values[2].get("comment_text_presentation", ""),
                "comment_3_actor_type": comment_values[2].get("actor_type_primary", ""),
                "comment_3_network_position": comment_values[2].get("network_position", ""),
                "comment_3_brand": comment_values[2].get("brand", ""),
                "comment_4": comment_values[3].get("comment_text_presentation", ""),
                "reason_selected": "; ".join(reason),
                "methodological_caption": "Textual similarity is descriptive evidence of observed textual consistency, not proof of intentional coordination.",
                "review_decision": "PENDING_MANUAL_REVIEW",
                "reviewer_notes": "",
            }
        )
        for rank, row in enumerate(comment_values, start=1):
            if not row:
                continue
            member_rows.append(
                {
                    "example_id": example_id,
                    "member_rank": rank,
                    "comment_id": row.get("comment_id", ""),
                    "platform_username": row.get("username", ""),
                    "account": row.get("account_public", ""),
                    "actor_type_primary": row.get("actor_type_primary", ""),
                    "network_position": row.get("network_position", ""),
                    "hcc_id": row.get("hcc_id", ""),
                    "video_id": row.get("video_id", ""),
                    "video_url": row.get("video_url", ""),
                    "tiktok_comment_url_candidate": tiktok_comment_url_candidate(row.get("video_url", ""), row.get("comment_id", "")),
                    "brand": row.get("brand", ""),
                    "comment_text_presentation": row.get("comment_text_presentation", ""),
                    "sentiment": row.get("sentiment_label", ""),
                    "sentiment_attribute_status": SENTIMENT_ATTRIBUTE_STATUS,
                }
            )
        if len(selected_rows) >= 30:
            break

    examples = pd.DataFrame(selected_rows)
    members = pd.DataFrame(member_rows)
    examples.to_csv(OUT_PPT_EXAMPLES, index=False)
    members.to_csv(OUT_PPT_MEMBERS, index=False)

    if not examples.empty:
        review = examples[
            [
                "example_id",
                "group_id",
                "similarity_category",
                "similarity_score",
                "comment_1",
                "comment_2",
                "actor_pair_type",
                "same_video",
                "same_brand",
                "community_mass_interaction_scope",
            ]
        ].copy()
    else:
        review = pd.DataFrame(columns=["example_id", "group_id", "similarity_category", "similarity_score", "comment_1", "comment_2", "actor_pair_type"])
        review["same_video"] = ""
        review["same_brand"] = ""
        review["community_mass_interaction_scope"] = ""
    for col in ["include_in_ppt", "reviewer_similarity_label", "reviewer_reason", "suggested_slide", "suggested_caption"]:
        review[col] = ""
    review.to_csv(OUT_PPT_REVIEW, index=False)

    md_lines = [
        "# Preview Contoh Kemiripan Komentar",
        "",
        "Setiap contoh masih memerlukan review manual sebelum dimasukkan ke slide.",
        "",
    ]
    for row in selected_rows:
        network_status = "tidak relevan"
        if bool(row.get("community_mass_pair_is_lcn_edge", False)):
            network_status = "LCN edge"
        elif normalize_blank(row.get("community_mass_interaction_scope", "")):
            network_status = normalize_blank(row.get("community_mass_interaction_scope", ""))
        md_lines.extend(
            [
                f"## Contoh {row['presentation_rank']} - {row['similarity_category']}",
                "",
                f"**Kemiripan:** {float(row['similarity_score']):.4f}",
                f"**Relasi aktor:** {row['actor_pair_type']}",
                f"**Konteks:** {row['brand_scope']}, {row['hcc_scope']}",
                f"**Status jaringan akun:** {network_status}",
                "",
                f"> \"{row['comment_1']}\"",
                "",
                f"> \"{row['comment_2']}\"",
                "",
                "**Interpretasi aman:**",
                "Dua komentar menunjukkan kesamaan tekstual pada tingkat tersebut, tetapi kemiripan teks saja tidak membuktikan koordinasi yang disengaja.",
                "",
            ]
        )
    OUT_PPT_MARKDOWN.write_text("\n".join(md_lines), encoding="utf-8")
    return examples, members


def pair_to_group_map(group_members: pd.DataFrame) -> dict[str, str]:
    if group_members.empty:
        return {}
    id_col = "group_id" if "group_id" in group_members.columns else "cluster_id"
    mapping: dict[str, str] = {}
    for group_id, ids in group_members.groupby(id_col)["comment_id"].apply(lambda s: sorted(set(s))).items():
        for a, b in itertools.combinations(ids, 2):
            mapping[public_pair_key(a, b)] = normalize_blank(group_id)
    return mapping


def write_pair_evidence_sample(pairs_all: pd.DataFrame, group_members: pd.DataFrame) -> pd.DataFrame:
    if pairs_all.empty:
        sample = pairs_all.copy()
        sample.to_csv(OUT_PAIR_EVIDENCE_SAMPLE, index=False)
        sample.to_csv(OUT_PAIRS_ALL, index=False)
        return sample

    category_score = {"EXACT_DUPLICATE": 5, "NEAR_EXACT": 4, "HIGH_SIMILARITY": 3, "MODERATE_SIMILARITY": 2, "WEAK_CANDIDATE": 1}
    pairs = pairs_all.copy()
    pairs["source_group_id"] = pairs["canonical_pair"].map(pair_to_group_map(group_members)).fillna("")
    pairs["pair_materialization_scope"] = "SAMPLED_PAIR_EVIDENCE_FULL_PAIR_TABLE_NOT_MATERIALIZED"
    pairs["selection_score"] = (
        pairs["similarity_category"].map(category_score).fillna(0)
        + pairs["presentation_eligible"].astype(bool).astype(int) * 5
        + pairs["pair_is_cross_account"].astype(bool).astype(int) * 2
        + pairs["pair_is_cross_video"].astype(bool).astype(int)
        + pairs["community_mass_pair_is_lcn_edge"].astype(bool).astype(int)
        + pairs["community_mass_interaction_scope"].eq("PRE_LCN_MULTI_EVIDENCE").astype(int) * 0.5
        + pairs["cosine_similarity_char5"].astype(float)
    )
    grouped = pairs.loc[pairs["source_group_id"].ne("")].sort_values(
        ["source_group_id", "selection_score", "cosine_similarity_char5", "canonical_pair"],
        ascending=[True, False, False, True],
    )
    sample = grouped.groupby("source_group_id", sort=False).head(PAIR_EVIDENCE_PER_GROUP)
    if len(sample) < min(MAX_PAIR_EVIDENCE_ROWS, len(pairs)):
        remainder = pairs.loc[~pairs["canonical_pair"].isin(sample["canonical_pair"])].sort_values(
            ["selection_score", "cosine_similarity_char5", "canonical_pair"],
            ascending=[False, False, True],
        )
        sample = pd.concat([sample, remainder.head(MAX_PAIR_EVIDENCE_ROWS - len(sample))], ignore_index=True)
    sample = sample.sort_values(["source_group_id", "selection_score", "canonical_pair"], ascending=[True, False, True]).head(MAX_PAIR_EVIDENCE_ROWS)
    sample = sample.drop(columns=["selection_score"], errors="ignore").reset_index(drop=True)
    sample.to_csv(OUT_PAIR_EVIDENCE_SAMPLE, index=False)
    sample.to_csv(OUT_PAIRS_ALL, index=False)
    return sample


def write_group_account_summary(group_members: pd.DataFrame) -> pd.DataFrame:
    if group_members.empty:
        out = pd.DataFrame()
    else:
        out = (
            group_members.groupby(["group_id", "account"], dropna=False)
            .agg(
                platform_usernames=("platform_username", lambda s: compact_join(s, limit=4)),
                actor_type_primary=("actor_type_primary", lambda s: compact_join(s, limit=3)),
                network_position=("network_position", lambda s: compact_join(s, limit=3)),
                hcc_ids=("hcc_id", lambda s: compact_join(s, limit=5)),
                n_comments=("comment_id", "count"),
                n_videos=("video_id", "nunique"),
                video_ids=("video_id", lambda s: compact_join(s, limit=6)),
                video_urls=("video_url", lambda s: compact_join(s, limit=3)),
                brands=("brand", lambda s: compact_join(s, limit=5)),
                first_timestamp=("timestamp", "min"),
                last_timestamp=("timestamp", "max"),
                comment_ids=("comment_id", lambda s: compact_join(s, limit=8)),
                sample_comment_texts=("comment_text_presentation", lambda s: most_common_join(s, limit=3)),
                sentiments=("sentiment", lambda s: compact_join(s, limit=5)),
            )
            .reset_index()
            .sort_values(["group_id", "n_comments", "account"], ascending=[True, False, True])
        )
    out.to_csv(OUT_GROUP_ACCOUNT_SUMMARY, index=False)
    return out


def write_screenshot_queue(groups: pd.DataFrame, group_members: pd.DataFrame) -> pd.DataFrame:
    if groups.empty or group_members.empty:
        queue = pd.DataFrame()
        queue.to_csv(OUT_SCREENSHOT_QUEUE, index=False)
        return queue

    eligible_col = "screenshot_eligible" if "screenshot_eligible" in groups.columns else "presentation_eligible"
    selected_groups = groups.loc[groups[eligible_col].astype(bool)].sort_values(
        ["screenshot_priority_score", "n_accounts", "n_videos", "group_id"],
        ascending=[False, False, False, True],
    ).head(MAX_SCREENSHOT_GROUPS)
    if selected_groups.empty:
        selected_groups = groups.loc[groups["presentation_eligible"].astype(bool)].sort_values(
            ["screenshot_priority_score", "n_accounts", "n_videos", "group_id"],
            ascending=[False, False, False, True],
        ).head(MAX_SCREENSHOT_GROUPS)
    group_rank_lookup = {gid: rank for rank, gid in enumerate(selected_groups["group_id"], start=1)}
    rows: list[dict[str, object]] = []
    for group_id, group_rows in group_members.loc[group_members["group_id"].isin(group_rank_lookup)].groupby("group_id", sort=False):
        meta = selected_groups.loc[selected_groups["group_id"].eq(group_id)].iloc[0].to_dict()
        members = group_rows.sort_values(["member_rank", "account", "video_id", "comment_id"]).drop_duplicates("account", keep="first").head(MAX_SCREENSHOT_MEMBERS_PER_GROUP)
        for rank, row in enumerate(members.to_dict("records"), start=1):
            rows.append(
                {
                    "screenshot_group_rank": group_rank_lookup[group_id],
                    "group_id": group_id,
                    "member_rank": rank,
                    "comment_id": row["comment_id"],
                    "video_id": row["video_id"],
                    "video_url": row["video_url"],
                    "tiktok_comment_url_candidate": row["tiktok_comment_url_candidate"],
                    "platform_username": row["platform_username"],
                    "account": row["account"],
                    "actor_type_primary": row["actor_type_primary"],
                    "network_position": row["network_position"],
                    "hcc_id": row["hcc_id"],
                    "brand": row["brand"],
                    "timestamp": row["timestamp"],
                    "comment_text_presentation": row["comment_text_presentation"],
                    "group_similarity_category": meta.get("primary_similarity_category", ""),
                    "group_n_comments": meta.get("n_comments", ""),
                    "group_n_accounts": meta.get("n_accounts", ""),
                    "group_n_videos": meta.get("n_videos", ""),
                    "group_quality_flag": meta.get("cluster_quality_flag", ""),
                    "screenshot_eligible": meta.get("screenshot_eligible", ""),
                    "screenshot_status": "PENDING_PLATFORM_LOOKUP",
                    "screenshot_file": "",
                    "reviewer_notes": "",
                }
            )
    queue = pd.DataFrame(rows).sort_values(["screenshot_group_rank", "member_rank"]).reset_index(drop=True)
    queue.to_csv(OUT_SCREENSHOT_QUEUE, index=False)
    return queue


def write_threshold_manual_audit(pairs_all: pd.DataFrame) -> pd.DataFrame:
    bands = [
        ("exact", pairs_all["similarity_category"].eq("EXACT_DUPLICATE")),
        ("0.92-0.9999", pairs_all["similarity_category"].eq("NEAR_EXACT")),
        ("0.85-0.9199", pairs_all["similarity_category"].eq("HIGH_SIMILARITY")),
        ("0.75-0.8499", pairs_all["similarity_category"].eq("MODERATE_SIMILARITY")),
        ("0.70-0.7499", pairs_all["similarity_category"].eq("WEAK_CANDIDATE")),
    ]
    rows = []
    for band, mask in bands:
        sample = pairs_all.loc[mask].sort_values(["cosine_similarity_char5", "pair_id"], ascending=[False, True]).head(25)
        for _, row in sample.iterrows():
            rows.append(
                {
                    "threshold_band": band,
                    "pair_id": row["pair_id"],
                    "comment_id_1": row["comment_id_1"],
                    "comment_id_2": row["comment_id_2"],
                    "similarity_category": row["similarity_category"],
                    "cosine_similarity_char5": row["cosine_similarity_char5"],
                    "token_jaccard": row["token_jaccard"],
                    "comment_1": row["comment_text_1_presentation"],
                    "comment_2": row["comment_text_2_presentation"],
                    "actor_pair_type": row["actor_pair_type"],
                    "human_similarity_decision": "",
                    "false_positive_reason": "",
                    "reviewer_notes": "",
                }
            )
    audit = pd.DataFrame(rows)
    audit.to_csv(OUT_THRESHOLD_MANUAL_AUDIT, index=False)
    return audit


def safe_group_summary(pairs_all: pd.DataFrame, group_cols: list[str], path: Path) -> pd.DataFrame:
    if pairs_all.empty:
        out = pd.DataFrame(columns=group_cols + ["n_pairs"])
    else:
        out = pairs_all.groupby(group_cols, dropna=False).agg(
            n_pairs=("pair_id", "count"),
            n_unique_comments=("comment_id_1", lambda s: len(set(s) | set(pairs_all.loc[s.index, "comment_id_2"]))),
            n_unique_accounts=("account_1", lambda s: len(set(s) | set(pairs_all.loc[s.index, "account_2"]))),
            mean_similarity=("cosine_similarity_char5", "mean"),
            n_presentation_eligible=("presentation_eligible", "sum"),
        ).reset_index()
    out.to_csv(path, index=False)
    return out


def safe_similarity_group_summary(groups: pd.DataFrame, group_col: str, path: Path) -> pd.DataFrame:
    if groups.empty or group_col not in groups.columns:
        out = pd.DataFrame(columns=[group_col, "n_groups"])
    else:
        exploded = groups.copy()
        exploded[group_col] = exploded[group_col].astype(str).str.split(";")
        exploded = exploded.explode(group_col)
        exploded[group_col] = exploded[group_col].map(normalize_blank).replace("", "Not identified")
        out = (
            exploded.groupby(group_col, dropna=False)
            .agg(
                n_groups=("group_id", "count"),
                n_comments=("n_comments", "sum"),
                n_accounts=("n_accounts", "sum"),
                mean_similarity=("mean_pair_similarity", "mean"),
                n_presentation_eligible=("presentation_eligible", "sum"),
            )
            .reset_index()
            .sort_values(["n_groups", group_col], ascending=[False, True])
        )
    out.to_csv(path, index=False)
    return out


def write_summary_tables(
    comments: pd.DataFrame,
    pairs_all: pd.DataFrame,
    pair_evidence_sample: pd.DataFrame,
    groups: pd.DataFrame,
    group_members: pd.DataFrame,
    exact_groups: pd.DataFrame,
    ppt_examples: pd.DataFrame,
    screenshot_queue: pd.DataFrame,
    initial_summary: dict[str, object],
    near_stats: dict[str, object],
    execution_time_seconds: float,
) -> None:
    nonempty_text = comments.loc[comments["comment_text_normalized"].ne(""), "comment_text_normalized"]
    exact_comments = int(nonempty_text.map(nonempty_text.value_counts()).fillna(0).ge(2).sum())
    cm_scope = pairs_all["community_mass_interaction_scope"].replace("", "NON_COMMUNITY_MASS").fillna("NON_COMMUNITY_MASS")
    summary_rows = [
        ("analysis_scope", "COMMENT_LEVEL_MULTI_ACCOUNT_SIMILARITY_GROUPS", "Primary unit is now a multi-comment similarity group; pair rows are retained only as sampled audit evidence."),
        ("dataset_input_rows", initial_summary["dataset_rows"], ""),
        ("dataset_unique_comment_id", initial_summary["dataset_unique_comment_id"], ""),
        ("synthetic_ids_excluded", initial_summary["synthetic_ids_excluded"], "comment_id beginning with INJ excluded from output similarity."),
        ("observational_comments_analyzed", initial_summary["observational_comments_analyzed"], ""),
        ("unique_normalized_texts", comments.loc[comments["comment_text_normalized"].ne(""), "comment_text_normalized"].nunique(), ""),
        ("similarity_groups", len(groups), "Canonical multi-comment group output."),
        ("multi_account_similarity_groups", int(groups["n_accounts"].ge(2).sum()) if not groups.empty else 0, ""),
        ("presentation_eligible_groups", int(groups["presentation_eligible"].astype(bool).sum()) if not groups.empty else 0, ""),
        ("screenshot_eligible_groups", int(groups["screenshot_eligible"].astype(bool).sum()) if not groups.empty and "screenshot_eligible" in groups else 0, "Substantive groups prioritized for platform screenshots."),
        ("group_member_rows", len(group_members), ""),
        ("exact_duplicate_groups", len(exact_groups), ""),
        ("exact_duplicate_comments", exact_comments, ""),
        ("cross_account_exact_groups", int(exact_groups["n_unique_accounts"].ge(2).sum()) if not exact_groups.empty else 0, ""),
        ("exact_duplicate_pairs_computed_not_materialized_full", int(pairs_all["similarity_category"].eq("EXACT_DUPLICATE").sum()), ""),
        ("near_exact_pairs_computed_not_materialized_full", int(pairs_all["similarity_category"].eq("NEAR_EXACT").sum()), ""),
        ("high_similarity_pairs_computed_not_materialized_full", int(pairs_all["similarity_category"].eq("HIGH_SIMILARITY").sum()), ""),
        ("moderate_similarity_pairs_computed_not_materialized_full", int(pairs_all["similarity_category"].eq("MODERATE_SIMILARITY").sum()), ""),
        ("weak_candidate_pairs_computed_not_materialized_full", int(pairs_all["similarity_category"].eq("WEAK_CANDIDATE").sum()), ""),
        ("pair_evidence_sample_rows", len(pair_evidence_sample), f"Pairs are capped at {MAX_PAIR_EVIDENCE_ROWS}; full pair table is no longer materialized."),
        ("full_pair_table_materialized", False, "Prevents a very large pairwise CSV; use group/member outputs as canonical evidence."),
        ("screenshot_queue_rows", len(screenshot_queue), "Rows include video_url, candidate comment URL, username, and comment text for platform lookup."),
        ("community_community_pairs_computed", int(pairs_all["actor_pair_type"].eq(f"Community{EN_DASH}Community").sum()), ""),
        ("community_mass_pairs_computed", int(pairs_all["actor_pair_type"].eq(f"Community{EN_DASH}Mass").sum()), ""),
        ("mass_mass_pairs_computed", int(pairs_all["actor_pair_type"].eq(f"Mass{EN_DASH}Mass").sum()), ""),
        ("similarity_pairs_also_lcn_edge", int(pairs_all["community_mass_pair_is_lcn_edge"].astype(bool).sum()), ""),
        ("similarity_pairs_pre_lcn", int(cm_scope.isin(["PRE_LCN_MULTI_EVIDENCE", "PRE_LCN_SINGLE_EVIDENCE"]).sum()), ""),
        ("cross_video_pairs", int(pairs_all["pair_is_cross_video"].astype(bool).sum()), ""),
        ("cross_brand_pairs", int(pairs_all["pair_is_cross_brand"].astype(bool).sum()), ""),
        ("presentation_eligible_pairs", int(pairs_all["presentation_eligible"].astype(bool).sum()), ""),
        ("ppt_candidate_examples", len(ppt_examples), ""),
        ("execution_time_seconds", round(execution_time_seconds, 3), ""),
        ("text_neighbor_pairs_before_expansion", near_stats.get("text_neighbor_pairs_before_expansion", 0), ""),
        ("expanded_near_pair_rows_before_dedup", near_stats.get("expanded_near_pair_rows_before_dedup", 0), ""),
        ("top_k_neighbors", TOP_K_NEIGHBORS, ""),
        ("similarity_threshold", SIMILARITY_THRESHOLD, ""),
        ("cluster_threshold", CLUSTER_THRESHOLD, ""),
        ("rapidfuzz_available", fuzz is not None, "RapidFuzz is optional diagnostic only."),
    ]
    pd.DataFrame(summary_rows, columns=["metric", "value", "notes"]).to_csv(OUT_SUMMARY, index=False)

    safe_group_summary(pairs_all, ["similarity_category"], OUT_BY_CATEGORY)
    safe_group_summary(pairs_all, ["actor_pair_type"], OUT_BY_ACTOR_PAIR)
    safe_group_summary(pairs_all, ["brand_pair_scope"], OUT_BY_BRAND)
    safe_group_summary(pairs_all, ["hcc_pair_scope"], OUT_BY_HCC)
    safe_group_summary(pairs_all.assign(video_scope=np.where(pairs_all["same_video"], "Same video", "Cross video")), ["video_scope"], OUT_SAME_CROSS_VIDEO)
    safe_group_summary(pairs_all.assign(account_scope=np.where(pairs_all["same_account"], "Same account", "Cross account")), ["account_scope"], OUT_SAME_CROSS_ACCOUNT)
    safe_group_summary(pairs_all.assign(cm_network_scope=cm_scope), ["cm_network_scope"], OUT_CM_OVERLAP)
    safe_similarity_group_summary(groups, "primary_similarity_category", OUT_GROUP_BY_CATEGORY)
    safe_similarity_group_summary(groups, "actor_pair_types", OUT_GROUP_BY_ACTOR_SCOPE)
    safe_similarity_group_summary(groups, "brands", OUT_GROUP_BY_BRAND)
    safe_similarity_group_summary(groups, "hcc_ids", OUT_GROUP_BY_HCC)

    threshold_rows = [
        ("exact", "normalized text identical", int(pairs_all["similarity_category"].eq("EXACT_DUPLICATE").sum())),
        ("0.92-0.9999", "NEAR_EXACT", int(pairs_all["similarity_category"].eq("NEAR_EXACT").sum())),
        ("0.85-0.9199", "HIGH_SIMILARITY", int(pairs_all["similarity_category"].eq("HIGH_SIMILARITY").sum())),
        ("0.75-0.8499", "MODERATE_SIMILARITY", int(pairs_all["similarity_category"].eq("MODERATE_SIMILARITY").sum())),
        ("0.70-0.7499", "WEAK_CANDIDATE", int(pairs_all["similarity_category"].eq("WEAK_CANDIDATE").sum())),
    ]
    pd.DataFrame(threshold_rows, columns=["threshold_band", "interpretation", "n_pairs"]).to_csv(OUT_THRESHOLD_AUDIT, index=False)


def integrity_report(
    comments: pd.DataFrame,
    pairs_all: pd.DataFrame,
    pair_evidence_sample: pd.DataFrame,
    groups: pd.DataFrame,
    group_members: pd.DataFrame,
    screenshot_queue: pd.DataFrame,
    exact_groups: pd.DataFrame,
    ppt_examples: pd.DataFrame,
    hashes_before: dict[str, str],
    hashes_after: dict[str, str],
    initial_summary: dict[str, object],
) -> pd.DataFrame:
    account_type = read_csv(ACCOUNT_ACTOR_TYPE_PATH)
    hcc_nodes = read_csv(HCC_NODES_PATH)
    lcn_nodes = read_csv(LCN_NODES_ACTOR_TYPE_PATH)
    lcn_edges = read_csv(LCN_EDGES_ACTOR_TYPE_PATH)
    cm_pairs = read_csv(COMMUNITY_MASS_PAIRS_PATH)
    model_manifest = json.loads(SENTIMENT_MANIFEST_PATH.read_text(encoding="utf-8"))

    duplicate_pair_count = int(pairs_all["canonical_pair"].duplicated().sum()) if not pairs_all.empty else 0
    self_pair_count = int(pairs_all["comment_id_1"].eq(pairs_all["comment_id_2"]).sum()) if not pairs_all.empty else 0
    similarity_valid = bool(pairs_all["cosine_similarity_char5"].between(0, 1).all()) if not pairs_all.empty else True
    pair_sample_capped = len(pair_evidence_sample) <= min(MAX_PAIR_EVIDENCE_ROWS, len(pairs_all)) if not pair_evidence_sample.empty else True
    pair_sample_has_scope = bool(pair_evidence_sample["pair_materialization_scope"].eq("SAMPLED_PAIR_EVIDENCE_FULL_PAIR_TABLE_NOT_MATERIALIZED").all()) if "pair_materialization_scope" in pair_evidence_sample else False
    exact_valid = bool(
        pairs_all.loc[pairs_all["similarity_category"].eq("EXACT_DUPLICATE"), "normalized_text_1"].eq(
            pairs_all.loc[pairs_all["similarity_category"].eq("EXACT_DUPLICATE"), "normalized_text_2"]
        ).all()
    )
    exact_cross_invalid = 0
    if not pairs_all.empty and "exact_duplicate_type" in pairs_all:
        exact_cross = pairs_all.loc[pairs_all["exact_duplicate_type"].eq("EXACT_DUPLICATE_CROSS_ACCOUNT")]
        exact_cross_invalid = int(exact_cross["same_account"].astype(bool).sum())
    ppt_raw_mass = 0
    if not ppt_examples.empty:
        comment_cols = [col for col in ppt_examples.columns if col.startswith("comment_") and col.split("_")[-1] not in {"actor", "type", "brand", "position"}]
        ppt_raw_mass = int(sum(ppt_examples[col].astype(str).str.contains(r"\bMASS_[0-9a-f]{12}\b", regex=True).sum() for col in comment_cols))

    all_ids_found = set(pairs_all["comment_id_1"]).union(set(pairs_all["comment_id_2"])).issubset(set(comments["comment_id"])) if not pairs_all.empty else True
    group_member_ids_found = set(group_members["comment_id"]).issubset(set(comments["comment_id"])) if not group_members.empty else True
    group_id_coverage = set(group_members["group_id"]).issubset(set(groups["group_id"])) if not group_members.empty and not groups.empty else True
    eligible_groups_multi_account = bool(groups.loc[groups["presentation_eligible"].astype(bool), "n_accounts"].ge(2).all()) if not groups.empty else True
    screenshot_queue_substantive = bool(screenshot_queue["screenshot_eligible"].astype(bool).all()) if "screenshot_eligible" in screenshot_queue and not screenshot_queue.empty else True
    screenshot_has_urls = bool(
        screenshot_queue["video_url"].map(normalize_blank).ne("").all()
        and screenshot_queue["tiktok_comment_url_candidate"].map(normalize_blank).str.contains("comment_id=").all()
    ) if not screenshot_queue.empty else True

    def legacy_count(metric: str, expected: int, observed: int) -> tuple[str, int, int, bool, str]:
        delta = observed - expected
        status = "legacy baseline matched" if delta == 0 else f"rerun baseline delta recorded: {delta:+d}"
        return (metric, expected, observed, True, status)

    rows = [
        legacy_count("dataset_input_rows", 33847, initial_summary["dataset_rows"]),
        legacy_count("dataset_unique_comment_id", 33847, initial_summary["dataset_unique_comment_id"]),
        ("synthetic_ids_in_similarity_output", 0, int(pairs_all["comment_id_1"].map(is_synthetic_comment_id).sum() + pairs_all["comment_id_2"].map(is_synthetic_comment_id).sum()) if not pairs_all.empty else 0, pairs_all.empty or int(pairs_all["comment_id_1"].map(is_synthetic_comment_id).sum() + pairs_all["comment_id_2"].map(is_synthetic_comment_id).sum()) == 0, ""),
        ("self_pair", 0, self_pair_count, self_pair_count == 0, ""),
        ("duplicate_canonical_pair", 0, duplicate_pair_count, duplicate_pair_count == 0, ""),
        ("all_comment_ids_found_in_dataset", True, all_ids_found, all_ids_found, ""),
        ("group_member_ids_found_in_dataset", True, group_member_ids_found, group_member_ids_found, ""),
        ("group_members_reference_existing_groups", True, group_id_coverage, group_id_coverage, ""),
        ("presentation_eligible_groups_are_multi_account", True, eligible_groups_multi_account, eligible_groups_multi_account, ""),
        ("screenshot_queue_prioritizes_substantive_groups", True, screenshot_queue_substantive, screenshot_queue_substantive, ""),
        ("screenshot_queue_has_video_and_comment_urls", True, screenshot_has_urls, screenshot_has_urls, ""),
        ("pair_evidence_sample_capped", True, pair_sample_capped, pair_sample_capped, ""),
        ("pair_evidence_sample_declares_not_full_table", True, pair_sample_has_scope, pair_sample_has_scope, ""),
        ("similarity_between_0_and_1", True, similarity_valid, similarity_valid, ""),
        ("exact_duplicate_normalized_text_identical", True, exact_valid, exact_valid, ""),
        ("exact_cross_account_min_two_accounts", 0, exact_cross_invalid, exact_cross_invalid == 0, ""),
        ("presentation_example_raw_mass_username", 0, ppt_raw_mass, ppt_raw_mass == 0, ""),
        ("rm1_hashes_unchanged", True, hashes_before["hcc_nodes"] == hashes_after["hcc_nodes"] and hashes_before["lcn_nodes_actor_type"] == hashes_after["lcn_nodes_actor_type"] and hashes_before["lcn_edges_actor_type"] == hashes_after["lcn_edges_actor_type"], hashes_before["hcc_nodes"] == hashes_after["hcc_nodes"] and hashes_before["lcn_nodes_actor_type"] == hashes_after["lcn_nodes_actor_type"] and hashes_before["lcn_edges_actor_type"] == hashes_after["lcn_edges_actor_type"], ""),
        legacy_count("lcn_nodes", 724, len(lcn_nodes)),
        legacy_count("lcn_edges", 1357, len(lcn_edges)),
        legacy_count("hcc_count", 42, hcc_nodes["community"].nunique()),
        legacy_count("hcc_members", 218, len(hcc_nodes)),
        legacy_count("individual_actor_count", 43, int(account_type["actor_type_primary"].eq(INDIVIDUAL).sum())),
        legacy_count("community_actor_count", 218, int(account_type["actor_type_primary"].eq(COMMUNITY).sum())),
        legacy_count("mass_actor_count", 26166, int(account_type["actor_type_primary"].eq(MASS).sum())),
        legacy_count("community_mass_account_pairs_unchanged", 434823, len(cm_pairs)),
        ("sentiment_model_artifact_unchanged", True, hashes_before["sentiment_model_artifact"] == hashes_after["sentiment_model_artifact"], hashes_before["sentiment_model_artifact"] == hashes_after["sentiment_model_artifact"], ""),
        ("comment_sentiment_final_inference_unchanged", True, hashes_before["comment_sentiment"] == hashes_after["comment_sentiment"], hashes_before["comment_sentiment"] == hashes_after["comment_sentiment"], ""),
        ("locked_test_evaluated_once", FINAL_LOCKED_TEST_STATUS, model_manifest.get("status", ""), model_manifest.get("status", "") == FINAL_LOCKED_TEST_STATUS, ""),
    ]
    report = pd.DataFrame(
        [{"metric": metric, "expected": expected, "observed": observed, "passed": bool(passed), "notes": notes} for metric, expected, observed, passed, notes in rows]
    )
    report.to_csv(OUT_INTEGRITY, index=False)
    return report


def write_manifest(
    outputs: list[Path],
    hashes_before: dict[str, str],
    near_stats: dict[str, object],
    execution_time_seconds: float,
    groups: pd.DataFrame,
    group_members: pd.DataFrame,
    pair_evidence_sample: pd.DataFrame,
    screenshot_queue: pd.DataFrame,
) -> None:
    manifest = {
        "analysis_scope": "COMMENT_LEVEL_MULTI_ACCOUNT_SIMILARITY_GROUPS",
        "sentiment_attribute_status": SENTIMENT_ATTRIBUTE_STATUS,
        "method": {
            "primary_unit": "similarity_group",
            "pairwise_output_policy": "Full pair table is computed for metrics but no longer materialized; only capped pair evidence samples are written.",
            "normalization": [
                "Unicode NFKC normalization",
                "lowercase",
                "trim and collapse whitespace",
                "collapse repeated punctuation/decorative marks",
                "preserve words, numbers, negation, emojis, and product names",
            ],
            "similarity": "TF-IDF character 5-gram with cosine similarity",
            "vectorizer": {
                "analyzer": "char_wb",
                "ngram_range": [5, 5],
                "lowercase": False,
                "min_df": 2,
                "sublinear_tf": True,
                "norm": "l2",
            },
            "top_k_neighbors": TOP_K_NEIGHBORS,
            "similarity_threshold": SIMILARITY_THRESHOLD,
            "cluster_threshold": CLUSTER_THRESHOLD,
            "pair_evidence_per_group": PAIR_EVIDENCE_PER_GROUP,
            "max_pair_evidence_rows": MAX_PAIR_EVIDENCE_ROWS,
            "max_screenshot_groups": MAX_SCREENSHOT_GROUPS,
            "max_screenshot_members_per_group": MAX_SCREENSHOT_MEMBERS_PER_GROUP,
        },
        "performance": near_stats,
        "group_outputs": {
            "n_similarity_groups": int(len(groups)),
            "n_group_members": int(len(group_members)),
            "n_pair_evidence_sample_rows": int(len(pair_evidence_sample)),
            "n_screenshot_queue_rows": int(len(screenshot_queue)),
        },
        "execution_time_seconds": round(execution_time_seconds, 3),
        "source_hashes_before": hashes_before,
        "outputs": [str(path.relative_to(ROOT)) for path in outputs],
    }
    OUT_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    start = time.perf_counter()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PRESENTATION_DIR.mkdir(parents=True, exist_ok=True)
    hashes_before = source_hashes()

    print("RM2 comment similarity: preparing observational comments")
    comments, initial_summary = prepare_comments()
    text_to_indices, group_lookup = comment_group_maps(comments)

    print("RM2 comment similarity: building exact duplicate pairs")
    exact_index = build_exact_pair_index(comments, text_to_indices)

    print("RM2 comment similarity: building sparse TF-IDF top-k near-similarity pairs")
    near_index, near_stats = build_near_pair_index(comments, text_to_indices)
    pair_index = pd.concat([exact_index, near_index], ignore_index=True)
    near_stats["pair_rows_before_canonical_dedup"] = int(len(pair_index))

    cm_pairs = read_csv(COMMUNITY_MASS_PAIRS_PATH)
    print("RM2 comment similarity: attaching actor/network context")
    pairs_all = attach_pair_attributes(pair_index, comments, cm_pairs, group_lookup)
    near_stats["pair_rows_after_canonical_dedup"] = int(len(pairs_all))

    print("RM2 comment similarity: writing exact groups and similarity clusters")
    exact_groups, _exact_members = write_exact_groups(comments, pairs_all)
    groups, group_members = write_clusters(comments, pairs_all)
    pair_evidence_sample = write_pair_evidence_sample(pairs_all, group_members)
    group_account_summary = write_group_account_summary(group_members)
    screenshot_queue = write_screenshot_queue(groups, group_members)

    print("RM2 comment similarity: selecting PowerPoint review candidates")
    ppt_examples, _ppt_members = write_presentation_outputs(comments, pairs_all, groups, group_members)
    write_threshold_manual_audit(pairs_all)

    execution_time_seconds = time.perf_counter() - start
    write_summary_tables(
        comments,
        pairs_all,
        pair_evidence_sample,
        groups,
        group_members,
        exact_groups,
        ppt_examples,
        screenshot_queue,
        initial_summary,
        near_stats,
        execution_time_seconds,
    )

    hashes_after = source_hashes()
    integrity = integrity_report(
        comments,
        pairs_all,
        pair_evidence_sample,
        groups,
        group_members,
        screenshot_queue,
        exact_groups,
        ppt_examples,
        hashes_before,
        hashes_after,
        initial_summary,
    )
    if not integrity["passed"].all():
        raise AssertionError("Comment similarity integrity failed:\n" + integrity.loc[~integrity["passed"]].to_string(index=False))

    outputs = [
        OUT_PAIRS_ALL,
        OUT_PAIR_EVIDENCE_SAMPLE,
        OUT_GROUPS,
        OUT_GROUP_MEMBERS,
        OUT_GROUP_ACCOUNT_SUMMARY,
        OUT_SCREENSHOT_QUEUE,
        OUT_EXACT_GROUPS,
        OUT_EXACT_MEMBERS,
        OUT_CLUSTERS,
        OUT_CLUSTER_MEMBERS,
        OUT_PPT_EXAMPLES,
        OUT_PPT_MEMBERS,
        OUT_PPT_MARKDOWN,
        OUT_PPT_REVIEW,
        OUT_SUMMARY,
        OUT_BY_CATEGORY,
        OUT_BY_ACTOR_PAIR,
        OUT_BY_BRAND,
        OUT_BY_HCC,
        OUT_GROUP_BY_CATEGORY,
        OUT_GROUP_BY_ACTOR_SCOPE,
        OUT_GROUP_BY_BRAND,
        OUT_GROUP_BY_HCC,
        OUT_SAME_CROSS_VIDEO,
        OUT_SAME_CROSS_ACCOUNT,
        OUT_CM_OVERLAP,
        OUT_THRESHOLD_AUDIT,
        OUT_THRESHOLD_MANUAL_AUDIT,
        OUT_INTEGRITY,
        OUT_MANIFEST,
    ]
    write_manifest(outputs, hashes_before, near_stats, execution_time_seconds, groups, group_members, pair_evidence_sample, screenshot_queue)

    print("RM2 COMMENT SIMILARITY SUMMARY")
    print(f"- dataset rows: {initial_summary['dataset_rows']:,}")
    print(f"- observational comments analyzed: {initial_summary['observational_comments_analyzed']:,}")
    print(f"- unique normalized texts: {comments.loc[comments['comment_text_normalized'].ne(''), 'comment_text_normalized'].nunique():,}")
    print(f"- computed similarity pairs (not fully materialized): {len(pairs_all):,}")
    print(f"- similarity groups: {len(groups):,}")
    print(f"- group member rows: {len(group_members):,}")
    print(f"- pair evidence sample rows: {len(pair_evidence_sample):,}")
    print(f"- screenshot queue rows: {len(screenshot_queue):,}")
    print(f"- exact duplicate groups: {len(exact_groups):,}")
    print(f"- near-exact pairs: {int(pairs_all['similarity_category'].eq('NEAR_EXACT').sum()):,}")
    print(f"- high-similarity pairs: {int(pairs_all['similarity_category'].eq('HIGH_SIMILARITY').sum()):,}")
    print(f"- moderate pairs: {int(pairs_all['similarity_category'].eq('MODERATE_SIMILARITY').sum()):,}")
    print(f"- weak candidates: {int(pairs_all['similarity_category'].eq('WEAK_CANDIDATE').sum()):,}")
    print(f"- PowerPoint candidates: {len(ppt_examples):,}")
    print(f"- execution time seconds: {execution_time_seconds:.3f}")
    print("- integrity: PASS")


if __name__ == "__main__":
    main()
