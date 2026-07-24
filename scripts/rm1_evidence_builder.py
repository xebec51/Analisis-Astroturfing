from __future__ import annotations

import itertools
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


MIN_CO_CONV = 3
MIN_CO_REPLY = 2
MIN_CO_TEMPORAL = 3
LCN_WEIGHT_PERCENTILE = 50

HASHTAG_STOPLIST = {
    "fyp",
    "foryou",
    "foryoupage",
    "viral",
    "trending",
    "tiktok",
    "skincare",
    "skincarerutine",
    "skincareroutine",
    "skintok",
    "fypviralindonesia",
    "fyppppppppppppppppppppppp",
    "fypage",
    "viraltiktok",
    "masukberanda",
    "xyzbca",
    "fypã‚·",
    "fypã‚·ã‚šviral",
    "foryoupageofficiall",
    "beranda",
    "racuntiktok",
    "racunskincare",
}


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


def canonical_pair(a: object, b: object) -> tuple[str, str]:
    x = normalize_username(a)
    y = normalize_username(b)
    return (x, y) if x <= y else (y, x)


def canonical_pair_key(a: object, b: object) -> str:
    x, y = canonical_pair(a, b)
    return f"{x}||{y}"


def max_norm(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").fillna(0.0)
    x_max = numeric.max()
    if x_max == 0:
        return pd.Series(0.0, index=series.index)
    return numeric / x_max


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, dtype=str, low_memory=False).fillna("")


def prepare_rm1_comments(comments_raw: pd.DataFrame) -> pd.DataFrame:
    df = comments_raw.copy()
    df = df.drop_duplicates(subset="comment_id", keep="first")
    df["username"] = df["username"].map(normalize_username)
    df["parent_user"] = df["parent_user"].map(normalize_username)
    df["video_id"] = df["video_id"].astype(str).str.strip()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df["is_reply"] = df["parent_user"].ne("")
    df["text"] = df["text"].fillna("") if "text" in df.columns else ""
    return df


def infer_comment_mass_users(
    comments: pd.DataFrame,
    community_users: set[str],
    individual_users: set[str],
) -> set[str]:
    comment_users = set(comments["username"].map(normalize_username))
    return comment_users - community_users - individual_users - {""}


def build_co_conv_edges_for_actor_sets(
    comments: pd.DataFrame,
    left_users: set[str],
    right_users: set[str],
    left_label: str = "left",
    right_label: str = "right",
) -> pd.DataFrame:
    counter: defaultdict[tuple[str, str], int] = defaultdict(int)
    role_rows: dict[tuple[str, str], tuple[str, str]] = {}

    video_users = comments.groupby("video_id")["username"].apply(lambda x: sorted(set(x))).to_dict()
    for _video_id, users in video_users.items():
        left = sorted(set(users) & left_users)
        right = sorted(set(users) & right_users)
        for l_user in left:
            for r_user in right:
                if not l_user or not r_user or l_user == r_user:
                    continue
                key = canonical_pair(l_user, r_user)
                counter[key] += 1
                role_rows[key] = (l_user, r_user)

    rows = []
    for (source, target), weight in sorted(counter.items()):
        l_user, r_user = role_rows[(source, target)]
        rows.append(
            {
                "source": source,
                "target": target,
                f"{left_label}_account": l_user,
                f"{right_label}_account": r_user,
                "co_conv_weight": int(weight),
                "passes_co_conv_threshold": bool(weight >= MIN_CO_CONV),
            }
        )
    return pd.DataFrame(rows)


def build_co_reply_edges_for_actor_sets(
    comments: pd.DataFrame,
    left_users: set[str],
    right_users: set[str],
    left_label: str = "left",
    right_label: str = "right",
) -> pd.DataFrame:
    replies = comments.loc[comments["is_reply"]].copy()
    if replies.empty:
        return pd.DataFrame(
            columns=[
                "source",
                "target",
                f"{left_label}_account",
                f"{right_label}_account",
                "reply_count",
                "co_reply_weight",
                "passes_co_reply_threshold",
            ]
        )

    reply_pairs = (
        replies.groupby(["username", "parent_user"])
        .agg(reply_count=("comment_id", "count"), shared_videos=("video_id", "nunique"))
        .reset_index()
        .rename(columns={"username": "source_raw", "parent_user": "target_raw"})
    )
    reply_pairs = reply_pairs.loc[
        reply_pairs["source_raw"].ne(reply_pairs["target_raw"])
        & reply_pairs["source_raw"].ne("")
        & reply_pairs["target_raw"].ne("")
    ].copy()

    rows = []
    for row in reply_pairs.to_dict("records"):
        src = normalize_username(row["source_raw"])
        tgt = normalize_username(row["target_raw"])
        src_is_left = src in left_users
        tgt_is_left = tgt in left_users
        src_is_right = src in right_users
        tgt_is_right = tgt in right_users
        if src_is_left and tgt_is_right:
            l_user, r_user = src, tgt
        elif tgt_is_left and src_is_right:
            l_user, r_user = tgt, src
        else:
            continue
        source, target = canonical_pair(src, tgt)
        rows.append(
            {
                "source": source,
                "target": target,
                f"{left_label}_account": l_user,
                f"{right_label}_account": r_user,
                "reply_count": int(row["reply_count"]),
                "co_reply_weight": int(row["shared_videos"]),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "source",
                "target",
                f"{left_label}_account",
                f"{right_label}_account",
                "reply_count",
                "co_reply_weight",
                "passes_co_reply_threshold",
            ]
        )

    df = pd.DataFrame(rows)
    # RM1 canonicalizes directed reply pairs and keeps the maximum repeated-video
    # count per undirected pair. reply_count is descriptive, so it is summed.
    out = (
        df.groupby(["source", "target", f"{left_label}_account", f"{right_label}_account"], as_index=False)
        .agg(reply_count=("reply_count", "sum"), co_reply_weight=("co_reply_weight", "max"))
        .sort_values(["source", "target"])
        .reset_index(drop=True)
    )
    out["passes_co_reply_threshold"] = out["co_reply_weight"].ge(MIN_CO_REPLY)
    return out


def calibrate_temporal_window_minutes(comments: pd.DataFrame) -> int:
    df_temp = comments.dropna(subset=["timestamp"]).sort_values(["video_id", "timestamp"])
    intervals: list[float] = []
    for _video_id, group in df_temp.groupby("video_id"):
        times = group["timestamp"].sort_values().values
        for idx in range(len(times) - 1):
            delta = (times[idx + 1] - times[idx]) / pd.Timedelta(minutes=1)
            if delta >= 0:
                intervals.append(float(delta))
    if not intervals:
        return 1
    return max(1, round(float(np.percentile(intervals, 50))))


def build_co_temporal_edges_for_actor_sets(
    comments: pd.DataFrame,
    left_users: set[str],
    right_users: set[str],
    window_minutes: int,
    left_label: str = "left",
    right_label: str = "right",
) -> pd.DataFrame:
    delta_limit = pd.Timedelta(minutes=window_minutes)
    df_temp = comments.dropna(subset=["timestamp"]).sort_values(["video_id", "timestamp"])
    pair_videos: defaultdict[tuple[str, str], set[str]] = defaultdict(set)
    role_rows: dict[tuple[str, str], tuple[str, str]] = {}

    for video_id, group in df_temp.groupby("video_id"):
        users = group["username"].values
        times = group["timestamp"].values
        n_rows = len(group)
        pairs_in_video: set[tuple[str, str]] = set()
        for i in range(n_rows):
            for j in range(i + 1, n_rows):
                if (times[j] - times[i]) > delta_limit:
                    break
                u = users[i]
                v = users[j]
                if u == v:
                    continue
                u_left = u in left_users
                v_left = v in left_users
                u_right = u in right_users
                v_right = v in right_users
                if u_left and v_right:
                    l_user, r_user = u, v
                elif v_left and u_right:
                    l_user, r_user = v, u
                else:
                    continue
                key = canonical_pair(u, v)
                pairs_in_video.add(key)
                role_rows[key] = (l_user, r_user)
        for key in pairs_in_video:
            pair_videos[key].add(str(video_id))

    rows = []
    for (source, target), videos in sorted(pair_videos.items()):
        l_user, r_user = role_rows[(source, target)]
        weight = len(videos)
        rows.append(
            {
                "source": source,
                "target": target,
                f"{left_label}_account": l_user,
                f"{right_label}_account": r_user,
                "co_temporal_weight": int(weight),
                "passes_co_temporal_threshold": bool(weight >= MIN_CO_TEMPORAL),
            }
        )
    return pd.DataFrame(rows)


def build_combined_evidence_edges(
    co_conv: pd.DataFrame,
    co_reply: pd.DataFrame,
    co_temporal: pd.DataFrame,
    normalization_maxima: dict[str, float] | None = None,
) -> pd.DataFrame:
    base_cols = ["source", "target"]
    cc = co_conv[base_cols + ["co_conv_weight", "passes_co_conv_threshold"]].copy()
    cr = co_reply[base_cols + ["co_reply_weight", "passes_co_reply_threshold"]].copy()
    ct = co_temporal[base_cols + ["co_temporal_weight", "passes_co_temporal_threshold"]].copy()

    combined = cc.merge(cr, on=base_cols, how="outer").merge(ct, on=base_cols, how="outer")
    weight_cols = ["co_conv_weight", "co_reply_weight", "co_temporal_weight"]
    flag_cols = ["passes_co_conv_threshold", "passes_co_reply_threshold", "passes_co_temporal_threshold"]
    for col in weight_cols:
        combined[col] = pd.to_numeric(combined[col], errors="coerce").fillna(0.0)
    for col in flag_cols:
        combined[col] = combined[col].where(combined[col].notna(), False).astype(bool)

    combined["passes_any_evidence_threshold"] = combined[flag_cols].any(axis=1)
    thresholded = combined.loc[combined["passes_any_evidence_threshold"]].copy()
    max_values = normalization_maxima or {
        col: pd.to_numeric(thresholded[col], errors="coerce").fillna(0.0).max() if not thresholded.empty else 0.0
        for col in weight_cols
    }
    combined["norm_co_conv"] = combined["co_conv_weight"] / max_values["co_conv_weight"] if max_values["co_conv_weight"] else 0.0
    combined["norm_co_reply"] = combined["co_reply_weight"] / max_values["co_reply_weight"] if max_values["co_reply_weight"] else 0.0
    combined["norm_co_temporal"] = combined["co_temporal_weight"] / max_values["co_temporal_weight"] if max_values["co_temporal_weight"] else 0.0
    combined["final_weight"] = combined["norm_co_conv"] + combined["norm_co_reply"] + combined["norm_co_temporal"]
    combined["n_evidence"] = (
        combined["co_conv_weight"].gt(0).astype(int)
        + combined["co_reply_weight"].gt(0).astype(int)
        + combined["co_temporal_weight"].gt(0).astype(int)
    )
    combined = combined.loc[combined["n_evidence"].gt(0)].copy()
    return combined.sort_values(["source", "target"]).reset_index(drop=True)


def evidence_combination(row: pd.Series) -> str:
    evidence = []
    if float(row.get("co_conv_weight", 0) or 0) > 0:
        evidence.append("Co-conv")
    if float(row.get("co_reply_weight", 0) or 0) > 0:
        evidence.append("Co-reply")
    if float(row.get("co_temporal_weight", 0) or 0) > 0:
        evidence.append("Co-temporal")
    if len(evidence) == 1:
        return f"{evidence[0]} only"
    if len(evidence) == 3:
        return "All three evidence"
    return " + ".join(evidence)


def excel_general_sci(id_str: object) -> str | None:
    try:
        value = float(id_str)
    except (TypeError, ValueError):
        return None
    mantissa, exp = f"{value:.5E}".split("E")
    if "." in mantissa:
        mantissa = mantissa.rstrip("0").rstrip(".")
    return mantissa + "E" + exp


def parse_tag_list(raw: object, strip_char: str, exclude: set[str] | None = None, min_len: int = 1) -> list[str]:
    if pd.isna(raw) or str(raw).strip() == "":
        return []
    tags = [tag.strip().lower().lstrip(strip_char) for tag in str(raw).split(";")]
    tags = [tag for tag in tags if tag and len(tag) >= min_len]
    if exclude:
        tags = [tag for tag in tags if tag not in exclude]
    return tags


def build_video_hashtag_lookup(comments: pd.DataFrame, metadata: pd.DataFrame) -> dict[str, list[str]]:
    df = comments.copy()
    meta = metadata.copy()
    df["video_id"] = df["video_id"].astype(str).str.strip()
    meta["video_id"] = meta["video_id"].astype(str).str.strip()

    meta_ids = set(meta["video_id"])
    match_stage = pd.Series("unmatched", index=df.index)
    match_stage[df["video_id"].isin(meta_ids)] = "video_id_exact"

    lossy_to_meta_id: dict[str | None, str] = {}
    for meta_id in meta["video_id"]:
        lossy_to_meta_id.setdefault(excel_general_sci(meta_id), meta_id)

    unmatched = match_stage.eq("unmatched")
    if unmatched.any():
        lossy = df.loc[unmatched, "video_id"].apply(excel_general_sci)
        hit_idx = lossy[lossy.isin(lossy_to_meta_id.keys())].index
        match_stage.loc[hit_idx] = "video_id_lossy"

    unmatched = match_stage.eq("unmatched")
    if unmatched.any() and "video_url" in meta.columns and meta["video_url"].notna().any():
        meta_urls = set(meta["video_url"].dropna())
        hit = unmatched & df["video_url"].isin(meta_urls)
        match_stage.loc[hit] = "video_url"

    unmatched = match_stage.eq("unmatched")
    if unmatched.any():
        key_df = df["product_category"].astype(str) + "||" + df["video_url"].astype(str)
        key_meta = set(meta["product_category"].astype(str) + "||" + meta["video_url"].astype(str))
        hit = unmatched & key_df.isin(key_meta)
        match_stage.loc[hit] = "product_category_video_url"

    vid_to_meta_id: dict[str, str] = {}
    for video_id in set(df.loc[match_stage.eq("video_id_exact"), "video_id"]) & meta_ids:
        vid_to_meta_id[str(video_id)] = str(video_id)
    for video_id in df.loc[match_stage.eq("video_id_lossy"), "video_id"].unique():
        lossy = excel_general_sci(video_id)
        if lossy in lossy_to_meta_id:
            vid_to_meta_id[str(video_id)] = str(lossy_to_meta_id[lossy])
    if match_stage.eq("video_url").any():
        url_to_meta_id = dict(zip(meta["video_url"], meta["video_id"]))
        for video_id in df.loc[match_stage.eq("video_url"), "video_id"].unique():
            url = df.loc[df["video_id"].eq(video_id), "video_url"].iloc[0]
            if url in url_to_meta_id:
                vid_to_meta_id[str(video_id)] = str(url_to_meta_id[url])
    if match_stage.eq("product_category_video_url").any():
        key_to_meta_id = dict(zip(meta["product_category"].astype(str) + "||" + meta["video_url"].astype(str), meta["video_id"]))
        for video_id in df.loc[match_stage.eq("product_category_video_url"), "video_id"].unique():
            row = df.loc[df["video_id"].eq(video_id)].iloc[0]
            key = str(row["product_category"]) + "||" + str(row["video_url"])
            if key in key_to_meta_id:
                vid_to_meta_id[str(video_id)] = str(key_to_meta_id[key])

    meta_hashtags = dict(
        zip(
            meta["video_id"],
            meta["hashtags"].apply(lambda value: parse_tag_list(value, "#", exclude=HASHTAG_STOPLIST, min_len=3)),
        )
    )
    return {video_id: meta_hashtags.get(meta_id, []) for video_id, meta_id in vid_to_meta_id.items()}


def account_hashtag_sets(comments: pd.DataFrame, video_hashtags: dict[str, list[str]]) -> dict[str, set[str]]:
    lookup: defaultdict[str, set[str]] = defaultdict(set)
    for row in comments[["username", "video_id"]].drop_duplicates().to_dict("records"):
        lookup[normalize_username(row["username"])].update(video_hashtags.get(str(row["video_id"]), []))
    return dict(lookup)
