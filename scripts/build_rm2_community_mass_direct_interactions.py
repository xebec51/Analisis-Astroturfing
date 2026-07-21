from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "dataset.csv"
ACCOUNT_ACTOR_TYPE_PATH = ROOT / "output/rm2_actor_type/tables/account_actor_type.csv"
HCC_NODES_PATH = ROOT / "output/gephi/gephi_hcc_nodes.csv"
LCN_NODES_ACTOR_TYPE_PATH = ROOT / "output/rm2_actor_type/gephi/gephi_lcn_nodes_actor_type.csv"
LCN_EDGES_ACTOR_TYPE_PATH = ROOT / "output/rm2_actor_type/gephi/gephi_lcn_edges_actor_type.csv"
AGGREGATE_NODES_PATH = ROOT / "output/rm2_actor_type/gephi/gephi_actor_type_nodes.csv"
AGGREGATE_EDGES_PATH = ROOT / "output/rm2_actor_type/gephi/gephi_actor_type_edges.csv"
SENTIMENT_READINESS_PATH = ROOT / "output/rm2_sentiment/human_validation_v2/sentiment_v2_locked_test_readiness.csv"

OUT_DIR = ROOT / "output/rm2_actor_type/direct_interaction"
OUT_GEPHI_DIR = ROOT / "output/rm2_actor_type/gephi"
OUT_PRIVATE_DIR = OUT_DIR / "private"

OUT_EVENTS = OUT_DIR / "community_mass_direct_reply_events.csv"
OUT_PROBABLE = OUT_DIR / "community_mass_probable_reply_references.csv"
OUT_PAIRS = OUT_DIR / "community_mass_direct_account_pairs.csv"
OUT_SUMMARY = OUT_DIR / "direct_interaction_summary.csv"
OUT_BY_DIRECTION = OUT_DIR / "direct_interaction_by_direction.csv"
OUT_BY_MASS_POSITION = OUT_DIR / "direct_interaction_by_mass_network_position.csv"
OUT_BY_HCC = OUT_DIR / "direct_interaction_by_hcc.csv"
OUT_BY_BRAND = OUT_DIR / "direct_interaction_by_brand.csv"
OUT_BY_SENTIMENT = OUT_DIR / "direct_interaction_by_sentiment.csv"
OUT_LCN_VS_NON = OUT_DIR / "direct_interaction_lcn_vs_non_lcn.csv"
OUT_MULTI_EVIDENCE = OUT_DIR / "direct_interaction_multi_evidence_summary.csv"
OUT_VALIDATION_AUDIT = OUT_DIR / "direct_reply_validation_audit.csv"
OUT_INTEGRITY = OUT_DIR / "direct_interaction_integrity_report.csv"
OUT_RUN_MANIFEST = OUT_DIR / "direct_interaction_run_manifest.json"
OUT_PRIVATE_MAPPING = OUT_PRIVATE_DIR / "community_mass_direct_private_mass_mapping.csv"

OUT_GEPHI_NODES = OUT_GEPHI_DIR / "gephi_community_mass_direct_nodes.csv"
OUT_GEPHI_EDGES = OUT_GEPHI_DIR / "gephi_community_mass_direct_edges.csv"
OUT_GEPHI_EDGES_UNDIRECTED = OUT_GEPHI_DIR / "gephi_community_mass_direct_edges_undirected.csv"

COMMUNITY = "Community Actor"
MASS = "Mass Actor"
MASS_HASH_SALT = "rm2_actor_type_public_mass_hash_v1"
ANALYSIS_SCOPE = "OPTIONAL_DIRECT_REPLY_DIAGNOSTIC"
SENTIMENT_ATTRIBUTE_STATUS = "DEVELOPMENT_MODEL_FROZEN_PENDING_LOCKED_TEST"
SENTIMENT_PENDING = SENTIMENT_ATTRIBUTE_STATUS
SENTIMENT_PENDING_NOTE = (
    "Sentiment V2 development model is frozen, but final inference is blocked until locked-test "
    "replacement labels are complete; the direct-reply layer stores structure only."
)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, dtype=str, low_memory=False).fillna("")


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
    cid = normalize_blank(value)
    return bool(re.match(r"^INJ", cid, flags=re.IGNORECASE) or re.search(r"(synthetic|challenge)", cid, flags=re.IGNORECASE))


def public_mass_id(username_norm: str) -> str:
    digest = hashlib.sha256(f"{MASS_HASH_SALT}:{username_norm}".encode("utf-8")).hexdigest()[:12]
    return f"MASS_{digest}"


def canonical_pair(a: str, b: str) -> str:
    return "||".join(sorted([a, b]))


def actor_lookup(account_type: pd.DataFrame, hcc_nodes: pd.DataFrame) -> tuple[dict[str, dict[str, object]], dict[str, str], dict[str, str]]:
    account = account_type.copy()
    account["username_norm"] = account["username"].map(normalize_username)
    account_by_public = account.set_index("username_norm").to_dict("index")
    community_hcc = hcc_nodes.assign(username_norm=hcc_nodes["id"].map(normalize_username)).set_index("username_norm")["community"].astype(str).to_dict()
    community_ids = set(community_hcc)

    raw_to_public_mass: dict[str, str] = {}

    def resolve(raw_username: object) -> dict[str, object]:
        raw_norm = normalize_username(raw_username)
        if raw_norm in community_ids:
            row = account_by_public.get(raw_norm, {})
            row = dict(row)
            row.setdefault("username", raw_norm)
            row["actor_type_primary"] = COMMUNITY
            row["network_position"] = row.get("network_position", "HCC") or "HCC"
            row["is_hcc_member"] = "True"
            row["is_lcn_member"] = row.get("is_lcn_member", "")
            row["community"] = str(community_hcc.get(raw_norm, row.get("community", "")))
            row["public_account"] = raw_norm
            row["raw_username_norm"] = raw_norm
            return row
        if raw_norm in account_by_public:
            row = dict(account_by_public[raw_norm])
            row["public_account"] = row.get("username", raw_norm)
            row["raw_username_norm"] = raw_norm
            return row
        mass_public = public_mass_id(raw_norm)
        raw_to_public_mass[raw_norm] = mass_public
        if normalize_username(mass_public) in account_by_public:
            row = dict(account_by_public[normalize_username(mass_public)])
        else:
            row = {
                "username": mass_public,
                "actor_type_primary": MASS,
                "network_position": "Outside LCN",
                "is_hcc_member": "False",
                "is_lcn_member": "False",
                "community": "Non-HCC",
                "n_comments": "",
                "dominant_sentiment": "",
                "account_goal_orientation": "",
                "target_brand_primary": "",
            }
        row["actor_type_primary"] = row.get("actor_type_primary", MASS) or MASS
        row["public_account"] = mass_public if row["actor_type_primary"] == MASS else row.get("username", raw_norm)
        row["raw_username_norm"] = raw_norm
        return row

    return {"resolve": resolve}, raw_to_public_mass, community_hcc


def prepare_lcn_edge_lookup(lcn_edges: pd.DataFrame) -> dict[str, dict[str, object]]:
    edge_lookup: dict[str, dict[str, object]] = {}
    for row in lcn_edges.to_dict("records"):
        source = normalize_username(row["Source"])
        target = normalize_username(row["Target"])
        edge_lookup[canonical_pair(source, target)] = row
    return edge_lookup


def parse_time(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce", utc=True)


def safe_text(value: object) -> str:
    return normalize_blank(value).replace("\r", " ").replace("\n", " ")


def validation_status_for_reply(
    child: pd.Series,
    parent: pd.Series | None,
    source_actor: dict[str, object],
    target_actor: dict[str, object] | None,
) -> tuple[str, str]:
    if parent is None:
        return "UNRESOLVED_PARENT", "parent_comment_id not found in observational dataset"
    if normalize_blank(child["video_id"]) != normalize_blank(parent["video_id"]):
        return "INVALID_PARENT_REFERENCE", "parent_comment_id maps to a different video_id"
    parent_user_norm = normalize_username(child.get("parent_user", ""))
    actual_parent_norm = normalize_username(parent.get("username", ""))
    if parent_user_norm and parent_user_norm != actual_parent_norm:
        return "PROBABLE_REPLY_REFERENCE", "parent_comment_id maps but parent_user does not match parent username"
    if target_actor is None:
        return "INVALID_PARENT_REFERENCE", "parent actor cannot be resolved"
    pair = {source_actor.get("actor_type_primary", ""), target_actor.get("actor_type_primary", "")}
    if pair == {COMMUNITY, MASS}:
        return "VERIFIED_DIRECT_REPLY", "parent_comment_id maps, video matches, parent_user matches, and actor pair is Community-Mass"
    return "VALID_PARENT_OTHER_ACTOR_PAIR", "valid parent reply but actor pair is not Community-Mass"


def response_minutes(child_ts: object, parent_ts: object) -> float | str:
    child = pd.to_datetime(child_ts, errors="coerce", utc=True)
    parent = pd.to_datetime(parent_ts, errors="coerce", utc=True)
    if pd.isna(child) or pd.isna(parent):
        return ""
    return float((child - parent).total_seconds() / 60.0)


def direction_label(source_type: str, target_type: str) -> str:
    if source_type == MASS and target_type == COMMUNITY:
        return "Mass to Community"
    if source_type == COMMUNITY and target_type == MASS:
        return "Community to Mass"
    return "Other"


def event_record(
    child: pd.Series,
    parent: pd.Series | None,
    source_actor: dict[str, object],
    target_actor: dict[str, object] | None,
    status: str,
    basis: str,
) -> dict[str, object]:
    source_type = str(source_actor.get("actor_type_primary", ""))
    target_type = str(target_actor.get("actor_type_primary", "")) if target_actor else ""
    source_public = str(source_actor.get("public_account", ""))
    target_public = str(target_actor.get("public_account", "")) if target_actor else ""
    source_is_community = source_type == COMMUNITY
    target_is_community = target_type == COMMUNITY
    community_actor = source_actor if source_is_community else target_actor if target_is_community else None
    mass_actor = source_actor if source_type == MASS else target_actor if target_type == MASS else None

    return {
        "child_comment_id": normalize_blank(child["comment_id"]),
        "parent_comment_id": normalize_blank(child["parent_comment_id"]),
        "source_account": source_public,
        "target_account": target_public,
        "source_actor_type": source_type,
        "target_actor_type": target_type,
        "source_network_position": source_actor.get("network_position", ""),
        "target_network_position": target_actor.get("network_position", "") if target_actor else "",
        "community_account": community_actor.get("public_account", "") if community_actor else "",
        "community_hcc_id": community_actor.get("community", "") if community_actor else "",
        "mass_account": mass_actor.get("public_account", "") if mass_actor else "",
        "mass_network_position": mass_actor.get("network_position", "") if mass_actor else "",
        "video_id": normalize_blank(child["video_id"]),
        "child_timestamp": normalize_blank(child["timestamp"]),
        "parent_timestamp": normalize_blank(parent["timestamp"]) if parent is not None else "",
        "response_minutes": response_minutes(child["timestamp"], parent["timestamp"]) if parent is not None else "",
        "direction": direction_label(source_type, target_type),
        "validation_status": status,
        "validation_basis": basis,
        "child_sentiment": "",
        "parent_sentiment": "",
        "same_sentiment_flag": "",
        "sentiment_attribute_status": SENTIMENT_PENDING,
        "target_brand": normalize_blank(child.get("product_category", "")),
        "source_comment_text": safe_text(child.get("text", "")),
        "parent_comment_text": safe_text(parent.get("text", "")) if parent is not None else "",
        "source_raw_username_norm": source_actor.get("raw_username_norm", ""),
        "target_raw_username_norm": target_actor.get("raw_username_norm", "") if target_actor else "",
    }


def lcn_attributes(pair_key_raw: str, lcn_lookup: dict[str, dict[str, object]]) -> dict[str, object]:
    edge = lcn_lookup.get(pair_key_raw, {})
    columns = [
        "co_conv_weight",
        "co_reply_weight",
        "co_temporal_weight",
        "norm_co_conv",
        "norm_co_reply",
        "norm_co_temporal",
        "final_weight",
        "n_evidence",
        "co_hashtag",
    ]
    result = {column: edge.get(column, "0") if edge else "0" for column in columns}
    result["pair_is_lcn_edge"] = bool(edge)
    return result


def num(value: object) -> float:
    try:
        if normalize_blank(value) == "":
            return 0.0
        return float(value)
    except ValueError:
        return 0.0


def combined_evidence(attrs: dict[str, object]) -> tuple[str, int, str]:
    evidence = ["Direct Reply"]
    if num(attrs.get("co_conv_weight")) > 0:
        evidence.append("Co-conv")
    if num(attrs.get("co_reply_weight")) > 0:
        evidence.append("Co-reply")
    if num(attrs.get("co_temporal_weight")) > 0:
        evidence.append("Co-temporal")
    lcn_count = len(evidence) - 1
    if lcn_count == 0:
        label = "Direct only"
    elif lcn_count == 1:
        label = f"Direct + {evidence[1]}"
    else:
        label = "Direct + Multiple LCN Evidence"
    return ";".join(evidence), len(evidence), label


def build_pair_table(verified: pd.DataFrame, lcn_lookup: dict[str, dict[str, object]]) -> pd.DataFrame:
    columns = [
        "community_account",
        "community_hcc_id",
        "mass_account",
        "mass_network_position",
        "community_is_lcn_member",
        "mass_is_lcn_member",
        "pair_is_lcn_edge",
        "direct_reply_count_total",
        "mass_to_community_reply_count",
        "community_to_mass_reply_count",
        "reciprocal_interaction_flag",
        "n_unique_videos",
        "n_unique_parent_comments",
        "n_unique_child_comments",
        "first_interaction_timestamp",
        "last_interaction_timestamp",
        "median_response_minutes",
        "min_response_minutes",
        "max_response_minutes",
        "positive_reply_count",
        "neutral_reply_count",
        "negative_reply_count",
        "sentiment_alignment_count",
        "sentiment_non_alignment_count",
        "dominant_interaction_sentiment",
        "sentiment_attribute_status",
        "co_conv_weight",
        "co_reply_weight",
        "co_temporal_weight",
        "norm_co_conv",
        "norm_co_reply",
        "norm_co_temporal",
        "final_weight",
        "n_evidence",
        "co_hashtag",
        "direct_evidence_types",
        "n_direct_evidence_types",
        "combined_evidence_label",
    ]
    if verified.empty:
        return pd.DataFrame(columns=columns)
    rows: list[dict[str, object]] = []
    for (community, mass), group in verified.groupby(["community_account", "mass_account"], dropna=False):
        mass_raw = group["source_raw_username_norm"].where(group["source_actor_type"].eq(MASS), group["target_raw_username_norm"]).iloc[0]
        community_raw = group["source_raw_username_norm"].where(group["source_actor_type"].eq(COMMUNITY), group["target_raw_username_norm"]).iloc[0]
        attrs = lcn_attributes(canonical_pair(community_raw, mass_raw), lcn_lookup)
        direct_evidence_types, n_direct_evidence_types, combined_label = combined_evidence(attrs)
        response = pd.to_numeric(group["response_minutes"], errors="coerce")
        mass_to_comm = int(group["direction"].eq("Mass to Community").sum())
        comm_to_mass = int(group["direction"].eq("Community to Mass").sum())
        row = {
            "community_account": community,
            "community_hcc_id": group["community_hcc_id"].replace("", np.nan).dropna().iloc[0] if group["community_hcc_id"].replace("", np.nan).notna().any() else "",
            "mass_account": mass,
            "mass_network_position": group["mass_network_position"].replace("", np.nan).dropna().iloc[0] if group["mass_network_position"].replace("", np.nan).notna().any() else "",
            "community_is_lcn_member": group["source_network_position"].where(group["source_actor_type"].eq(COMMUNITY), group["target_network_position"]).eq("HCC").any(),
            "mass_is_lcn_member": group["mass_network_position"].eq("LCN Non-HCC").any(),
            "pair_is_lcn_edge": attrs["pair_is_lcn_edge"],
            "direct_reply_count_total": len(group),
            "mass_to_community_reply_count": mass_to_comm,
            "community_to_mass_reply_count": comm_to_mass,
            "reciprocal_interaction_flag": mass_to_comm > 0 and comm_to_mass > 0,
            "n_unique_videos": group["video_id"].nunique(),
            "n_unique_parent_comments": group["parent_comment_id"].nunique(),
            "n_unique_child_comments": group["child_comment_id"].nunique(),
            "first_interaction_timestamp": group["child_timestamp"].min(),
            "last_interaction_timestamp": group["child_timestamp"].max(),
            "median_response_minutes": response.median() if response.notna().any() else "",
            "min_response_minutes": response.min() if response.notna().any() else "",
            "max_response_minutes": response.max() if response.notna().any() else "",
            "positive_reply_count": 0,
            "neutral_reply_count": 0,
            "negative_reply_count": 0,
            "sentiment_alignment_count": 0,
            "sentiment_non_alignment_count": 0,
            "dominant_interaction_sentiment": SENTIMENT_PENDING,
            "sentiment_attribute_status": SENTIMENT_PENDING,
            **attrs,
            "direct_evidence_types": direct_evidence_types,
            "n_direct_evidence_types": n_direct_evidence_types,
            "combined_evidence_label": combined_label,
        }
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def build_directed_gephi_edges(verified: pd.DataFrame, pair_table: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "Source",
        "Target",
        "Type",
        "Weight",
        "direction",
        "verified_direct_reply_count",
        "n_unique_videos",
        "median_response_minutes",
        "community_hcc_id",
        "mass_network_position",
        "pair_is_lcn_edge",
        "co_conv_weight",
        "co_reply_weight",
        "co_temporal_weight",
        "final_weight",
        "n_evidence",
        "dominant_interaction_sentiment",
        "sentiment_attribute_status",
        "combined_evidence_label",
    ]
    if verified.empty:
        return pd.DataFrame(columns=columns)
    pair_attrs = pair_table.set_index(["community_account", "mass_account"]).to_dict("index")
    rows: list[dict[str, object]] = []
    for (source, target), group in verified.groupby(["source_account", "target_account"], dropna=False):
        community = group["community_account"].iloc[0]
        mass = group["mass_account"].iloc[0]
        attrs = pair_attrs[(community, mass)]
        rows.append(
            {
                "Source": source,
                "Target": target,
                "Type": "Directed",
                "Weight": len(group),
                "direction": group["direction"].iloc[0],
                "verified_direct_reply_count": len(group),
                "n_unique_videos": group["video_id"].nunique(),
                "median_response_minutes": pd.to_numeric(group["response_minutes"], errors="coerce").median(),
                "community_hcc_id": attrs["community_hcc_id"],
                "mass_network_position": attrs["mass_network_position"],
                "pair_is_lcn_edge": attrs["pair_is_lcn_edge"],
                "co_conv_weight": attrs["co_conv_weight"],
                "co_reply_weight": attrs["co_reply_weight"],
                "co_temporal_weight": attrs["co_temporal_weight"],
                "final_weight": attrs["final_weight"],
                "n_evidence": attrs["n_evidence"],
                "dominant_interaction_sentiment": SENTIMENT_PENDING,
                "sentiment_attribute_status": SENTIMENT_PENDING,
                "combined_evidence_label": attrs["combined_evidence_label"],
            }
        )
    return pd.DataFrame(rows, columns=columns).sort_values(["Source", "Target"]).reset_index(drop=True)


def build_nodes(verified: pd.DataFrame, account_type: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "Id",
        "Label",
        "actor_type_primary",
        "network_position",
        "hcc_id",
        "is_lcn_member",
        "n_comments",
        "dominant_sentiment",
        "account_goal_orientation",
        "target_brand_primary",
        "sentiment_attribute_status",
        "direct_degree",
        "direct_weighted_degree",
    ]
    if verified.empty:
        return pd.DataFrame(columns=columns)
    public_accounts = set(verified["source_account"]) | set(verified["target_account"])
    account = account_type.copy()
    account["public_key"] = account["username"].map(normalize_blank)
    rows: list[dict[str, object]] = []
    for public in sorted(public_accounts):
        if not public:
            continue
        actor_type = COMMUNITY if public in set(verified["community_account"]) else MASS
        row_source = account.loc[account["public_key"].eq(public)]
        row = row_source.iloc[0].to_dict() if not row_source.empty else {}
        incident = verified.loc[verified["source_account"].eq(public) | verified["target_account"].eq(public)]
        hcc_id = ""
        if actor_type == COMMUNITY:
            hcc_vals = incident["community_hcc_id"].replace("", np.nan).dropna()
            hcc_id = str(hcc_vals.iloc[0]) if not hcc_vals.empty else str(row.get("community", ""))
        rows.append(
            {
                "Id": public,
                "Label": public,
                "actor_type_primary": actor_type,
                "network_position": row.get("network_position", "HCC" if actor_type == COMMUNITY else "Outside LCN"),
                "hcc_id": hcc_id,
                "is_lcn_member": row.get("is_lcn_member", "True" if actor_type == COMMUNITY else "False"),
                "n_comments": row.get("n_comments", ""),
                "dominant_sentiment": row.get("dominant_sentiment", ""),
                "account_goal_orientation": row.get("account_goal_orientation", ""),
                "target_brand_primary": row.get("target_brand_primary", ""),
                "sentiment_attribute_status": SENTIMENT_PENDING,
                "direct_degree": len((set(incident["source_account"]) | set(incident["target_account"])) - {public}),
                "direct_weighted_degree": len(incident),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def write_summary_tables(events: pd.DataFrame, verified: pd.DataFrame, pairs: pd.DataFrame) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    internal_cols = ["source_raw_username_norm", "target_raw_username_norm"]
    public_events = events.drop(columns=[col for col in internal_cols if col in events.columns])
    probable = public_events.loc[public_events["validation_status"].eq("PROBABLE_REPLY_REFERENCE")].copy()
    probable.to_csv(OUT_PROBABLE, index=False)
    public_events.to_csv(OUT_EVENTS, index=False)
    pairs.to_csv(OUT_PAIRS, index=False)

    if verified.empty:
        summary = pd.DataFrame(
            [
                {
                    "metric": "analysis_scope",
                    "value": ANALYSIS_SCOPE,
                    "sentiment_v2_status": SENTIMENT_PENDING,
                    "notes": "Reply-only Community-Mass output is retained only as an optional diagnostic.",
                },
                {
                    "metric": "verified_reply_events",
                    "value": 0,
                    "sentiment_v2_status": SENTIMENT_PENDING,
                    "notes": SENTIMENT_PENDING_NOTE,
                }
            ]
        )
    else:
        summary = pd.DataFrame(
            [
                {
                    "metric": "analysis_scope",
                    "value": ANALYSIS_SCOPE,
                    "sentiment_v2_status": SENTIMENT_PENDING,
                    "notes": "Reply-only Community-Mass output is retained only as an optional diagnostic.",
                },
                {"metric": "verified_reply_events", "value": len(verified), "sentiment_v2_status": SENTIMENT_PENDING, "notes": SENTIMENT_PENDING_NOTE},
                {"metric": "unique_community_mass_account_pairs", "value": len(pairs), "sentiment_v2_status": SENTIMENT_PENDING, "notes": ""},
                {"metric": "community_accounts_involved", "value": verified["community_account"].nunique(), "sentiment_v2_status": SENTIMENT_PENDING, "notes": ""},
                {"metric": "mass_accounts_involved", "value": verified["mass_account"].nunique(), "sentiment_v2_status": SENTIMENT_PENDING, "notes": ""},
                {"metric": "mass_lcn_non_hcc_involved", "value": int(pairs["mass_network_position"].eq("LCN Non-HCC").sum()), "sentiment_v2_status": SENTIMENT_PENDING, "notes": ""},
                {"metric": "mass_outside_lcn_involved", "value": int(pairs["mass_network_position"].eq("Outside LCN").sum()), "sentiment_v2_status": SENTIMENT_PENDING, "notes": ""},
                {"metric": "pairs_also_lcn_edge", "value": int(pairs["pair_is_lcn_edge"].astype(bool).sum()), "sentiment_v2_status": SENTIMENT_PENDING, "notes": ""},
                {"metric": "direct_reply_pairs_outside_lcn", "value": int((~pairs["pair_is_lcn_edge"].astype(bool)).sum()), "sentiment_v2_status": SENTIMENT_PENDING, "notes": ""},
                {"metric": "reciprocal_pairs", "value": int(pairs["reciprocal_interaction_flag"].astype(bool).sum()), "sentiment_v2_status": SENTIMENT_PENDING, "notes": ""},
            ]
        )
    summary.to_csv(OUT_SUMMARY, index=False)

    if verified.empty:
        pd.DataFrame(columns=["direction", "verified_reply_events", "unique_pairs"]).to_csv(OUT_BY_DIRECTION, index=False)
        pd.DataFrame(columns=["mass_network_position", "unique_pairs", "verified_reply_events", "mass_accounts"]).to_csv(OUT_BY_MASS_POSITION, index=False)
        pd.DataFrame(columns=["community_hcc_id", "unique_pairs", "verified_reply_events", "community_accounts", "mass_accounts"]).to_csv(OUT_BY_HCC, index=False)
        pd.DataFrame(columns=["target_brand", "verified_reply_events", "unique_pairs"]).to_csv(OUT_BY_BRAND, index=False)
    else:
        verified.groupby("direction", dropna=False).agg(
            verified_reply_events=("child_comment_id", "count"),
            unique_pairs=("mass_account", lambda s: len(set(zip(verified.loc[s.index, "community_account"], s)))),
        ).reset_index().to_csv(OUT_BY_DIRECTION, index=False)
        pairs.groupby("mass_network_position", dropna=False).agg(
            unique_pairs=("mass_account", "count"),
            verified_reply_events=("direct_reply_count_total", "sum"),
            mass_accounts=("mass_account", "nunique"),
        ).reset_index().to_csv(OUT_BY_MASS_POSITION, index=False)
        pairs.groupby("community_hcc_id", dropna=False).agg(
            unique_pairs=("mass_account", "count"),
            verified_reply_events=("direct_reply_count_total", "sum"),
            community_accounts=("community_account", "nunique"),
            mass_accounts=("mass_account", "nunique"),
        ).reset_index().to_csv(OUT_BY_HCC, index=False)
        verified.groupby("target_brand", dropna=False).agg(
            verified_reply_events=("child_comment_id", "count"),
            unique_pairs=("mass_account", lambda s: len(set(zip(verified.loc[s.index, "community_account"], s)))),
        ).reset_index().to_csv(OUT_BY_BRAND, index=False)
    pd.DataFrame(
        [
            {
                "sentiment_attribute_status": SENTIMENT_PENDING,
                "verified_reply_events": len(verified),
                "sentiment_coverage": 0.0,
                "notes": SENTIMENT_PENDING_NOTE,
            }
        ]
    ).to_csv(OUT_BY_SENTIMENT, index=False)
    if pairs.empty:
        pd.DataFrame(columns=["pair_is_lcn_edge", "unique_pairs", "verified_reply_events"]).to_csv(OUT_LCN_VS_NON, index=False)
        pd.DataFrame(columns=["combined_evidence_label", "unique_pairs", "verified_reply_events"]).to_csv(OUT_MULTI_EVIDENCE, index=False)
    else:
        pairs.groupby("pair_is_lcn_edge", dropna=False).agg(
            unique_pairs=("mass_account", "count"),
            verified_reply_events=("direct_reply_count_total", "sum"),
        ).reset_index().to_csv(OUT_LCN_VS_NON, index=False)
        pairs.groupby("combined_evidence_label", dropna=False).agg(
            unique_pairs=("mass_account", "count"),
            verified_reply_events=("direct_reply_count_total", "sum"),
        ).reset_index().to_csv(OUT_MULTI_EVIDENCE, index=False)
    events.groupby("validation_status", dropna=False).agg(
        reply_events=("child_comment_id", "count"),
        unique_child_comments=("child_comment_id", "nunique"),
    ).reset_index().to_csv(OUT_VALIDATION_AUDIT, index=False)


def main() -> None:
    source_hashes_before = {
        "dataset": sha256_file(DATASET_PATH),
        "account_actor_type": sha256_file(ACCOUNT_ACTOR_TYPE_PATH),
        "lcn_nodes_actor_type": sha256_file(LCN_NODES_ACTOR_TYPE_PATH),
        "lcn_edges_actor_type": sha256_file(LCN_EDGES_ACTOR_TYPE_PATH),
        "aggregate_nodes": sha256_file(AGGREGATE_NODES_PATH),
        "aggregate_edges": sha256_file(AGGREGATE_EDGES_PATH),
    }
    comments_all = read_csv(DATASET_PATH)
    account_type = read_csv(ACCOUNT_ACTOR_TYPE_PATH)
    hcc_nodes = read_csv(HCC_NODES_PATH)
    lcn_nodes = read_csv(LCN_NODES_ACTOR_TYPE_PATH)
    lcn_edges = read_csv(LCN_EDGES_ACTOR_TYPE_PATH)
    readiness = read_csv(SENTIMENT_READINESS_PATH) if SENTIMENT_READINESS_PATH.exists() else pd.DataFrame()
    sentiment_status = (
        readiness.loc[readiness["metric"].eq("locked_test_v2_status"), "value"].iloc[0]
        if not readiness.empty and readiness["metric"].eq("locked_test_v2_status").any()
        else SENTIMENT_PENDING
    )

    comments = comments_all.loc[~comments_all["comment_id"].map(is_synthetic_comment_id)].copy()
    comments["comment_id_norm"] = comments["comment_id"].map(normalize_blank)
    comments["parent_comment_id_norm"] = comments["parent_comment_id"].map(normalize_blank)
    comments["username_norm"] = comments["username"].map(normalize_username)
    comment_by_id = comments.drop_duplicates("comment_id_norm").set_index("comment_id_norm")
    actor_resolver, raw_to_public_mass, _community_hcc = actor_lookup(account_type, hcc_nodes)
    resolve_actor = actor_resolver["resolve"]
    lcn_lookup = prepare_lcn_edge_lookup(lcn_edges)

    event_rows: list[dict[str, object]] = []
    replies = comments.loc[comments["parent_comment_id_norm"].ne("")].copy()
    for _, child in replies.iterrows():
        source_actor = resolve_actor(child["username"])
        parent = None
        target_actor = None
        parent_id = child["parent_comment_id_norm"]
        if parent_id in comment_by_id.index:
            parent = comment_by_id.loc[parent_id]
            if isinstance(parent, pd.DataFrame):
                parent = parent.iloc[0]
            target_actor = resolve_actor(parent["username"])
        status, basis = validation_status_for_reply(child, parent, source_actor, target_actor)
        event_rows.append(event_record(child, parent, source_actor, target_actor, status, basis))

    events = pd.DataFrame(event_rows)
    verified = events.loc[events["validation_status"].eq("VERIFIED_DIRECT_REPLY")].copy()
    pairs = build_pair_table(verified, lcn_lookup)
    directed_edges = build_directed_gephi_edges(verified, pairs)
    nodes = build_nodes(verified, account_type)

    undirected = pairs.copy()
    if not undirected.empty:
        undirected.insert(0, "Source", undirected["community_account"])
        undirected.insert(1, "Target", undirected["mass_account"])
        undirected.insert(2, "Type", "Undirected")
        undirected.insert(3, "Weight", undirected["direct_reply_count_total"])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_GEPHI_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PRIVATE_DIR.mkdir(parents=True, exist_ok=True)
    write_summary_tables(events, verified, pairs)
    nodes.to_csv(OUT_GEPHI_NODES, index=False)
    directed_edges.to_csv(OUT_GEPHI_EDGES, index=False)
    undirected.to_csv(OUT_GEPHI_EDGES_UNDIRECTED, index=False)

    private_mapping = pd.DataFrame(
        [{"raw_mass_username_norm": raw, "public_mass_account": public} for raw, public in sorted(raw_to_public_mass.items())]
    )
    private_mapping.to_csv(OUT_PRIVATE_MAPPING, index=False)

    source_hashes_after = {
        "dataset": sha256_file(DATASET_PATH),
        "account_actor_type": sha256_file(ACCOUNT_ACTOR_TYPE_PATH),
        "lcn_nodes_actor_type": sha256_file(LCN_NODES_ACTOR_TYPE_PATH),
        "lcn_edges_actor_type": sha256_file(LCN_EDGES_ACTOR_TYPE_PATH),
        "aggregate_nodes": sha256_file(AGGREGATE_NODES_PATH),
        "aggregate_edges": sha256_file(AGGREGATE_EDGES_PATH),
    }

    node_ids = set(nodes["Id"]) if not nodes.empty else set()
    missing_endpoints = 0
    if not directed_edges.empty:
        missing_endpoints = int((~directed_edges["Source"].isin(node_ids)).sum() + (~directed_edges["Target"].isin(node_ids)).sum())
    aggregate_nodes_count = len(read_csv(AGGREGATE_NODES_PATH))
    aggregate_edges_count = len(read_csv(AGGREGATE_EDGES_PATH))
    lcn_pair_sets = lcn_edges[["source_actor_type", "target_actor_type"]].astype(str).apply(lambda row: frozenset(row), axis=1)
    lcn_community_community_edges = int((lcn_pair_sets == frozenset({COMMUNITY})).sum())
    lcn_community_mass_edges = int((lcn_pair_sets == frozenset({COMMUNITY, MASS})).sum())
    lcn_mass_mass_edges = int((lcn_pair_sets == frozenset({MASS})).sum())
    en_dash = "\u2013"
    lcn_edge_counts = {
        f"Community{en_dash}Community": lcn_community_community_edges,
        f"Community{en_dash}Mass": lcn_community_mass_edges,
        f"Mass{en_dash}Mass": lcn_mass_mass_edges,
    }

    integrity_rows = [
        ("analysis_scope", ANALYSIS_SCOPE, ANALYSIS_SCOPE, True, "Reply-only layer is optional diagnostic, not the main Community-Mass account network."),
        ("dataset_rows", 33847, len(comments_all), len(comments_all) == 33847, ""),
        ("synthetic_comments_excluded_from_direct_layer", 0, int(events["child_comment_id"].map(is_synthetic_comment_id).sum()) if not events.empty else 0, (events.empty or int(events["child_comment_id"].map(is_synthetic_comment_id).sum()) == 0), ""),
        ("actor_universe", 26427, len(account_type), len(account_type) == 26427, ""),
        ("community_actor_accounts", 218, int(account_type["actor_type_primary"].eq(COMMUNITY).sum()), int(account_type["actor_type_primary"].eq(COMMUNITY).sum()) == 218, ""),
        ("mass_actor_accounts", 26166, int(account_type["actor_type_primary"].eq(MASS).sum()), int(account_type["actor_type_primary"].eq(MASS).sum()) == 26166, ""),
        ("lcn_nodes", 724, len(lcn_nodes), len(lcn_nodes) == 724, ""),
        ("lcn_edges", 1357, len(lcn_edges), len(lcn_edges) == 1357, ""),
        ("lcn_community_community_edges", 464, lcn_edge_counts.get("Community–Community", 0), lcn_edge_counts.get("Community–Community", 0) == 464, ""),
        ("lcn_community_mass_edges", 305, lcn_edge_counts.get("Community–Mass", 0), lcn_edge_counts.get("Community–Mass", 0) == 305, ""),
        ("lcn_mass_mass_edges", 588, lcn_edge_counts.get("Mass–Mass", 0), lcn_edge_counts.get("Mass–Mass", 0) == 588, ""),
        ("aggregate_actor_type_nodes", 396, aggregate_nodes_count, aggregate_nodes_count == 396, ""),
        ("aggregate_actor_type_edges", 497, aggregate_edges_count, aggregate_edges_count == 497, ""),
        ("gephi_direct_nodes_no_aggregate_hcc", 0, int(nodes["Id"].astype(str).str.startswith("HCC_").sum()) if not nodes.empty else 0, nodes.empty or int(nodes["Id"].astype(str).str.startswith("HCC_").sum()) == 0, ""),
        ("gephi_direct_nodes_no_mass_segments", 0, int(nodes["Id"].astype(str).str.startswith("MASS_HCC_").sum()) if not nodes.empty else 0, nodes.empty or int(nodes["Id"].astype(str).str.startswith("MASS_HCC_").sum()) == 0, ""),
        ("gephi_direct_edge_missing_endpoints", 0, missing_endpoints, missing_endpoints == 0, ""),
        ("gephi_direct_edge_type_directed", "Directed", "Directed" if directed_edges.empty or directed_edges["Type"].eq("Directed").all() else "mixed", directed_edges.empty or directed_edges["Type"].eq("Directed").all(), ""),
        ("source_hashes_unchanged", True, source_hashes_before == source_hashes_after, source_hashes_before == source_hashes_after, ""),
        ("sentiment_v2_status_for_direct_layer", SENTIMENT_ATTRIBUTE_STATUS, SENTIMENT_PENDING if sentiment_status != "READY" else "READY", sentiment_status != "READY", SENTIMENT_PENDING_NOTE),
    ]
    integrity = pd.DataFrame(
        [
            {"metric": metric, "expected": expected, "observed": observed, "passed": bool(passed), "notes": notes}
            for metric, expected, observed, passed, notes in integrity_rows
        ]
    )
    integrity.to_csv(OUT_INTEGRITY, index=False)
    if not integrity["passed"].all():
        raise AssertionError("Direct interaction integrity failed:\n" + integrity.loc[~integrity["passed"]].to_string(index=False))

    manifest = {
        "analysis_scope": ANALYSIS_SCOPE,
        "sentiment_attribute_status": SENTIMENT_PENDING,
        "sentiment_status_source": str(SENTIMENT_READINESS_PATH.relative_to(ROOT)),
        "verified_direct_reply_events": int(len(verified)),
        "unique_community_mass_pairs": int(len(pairs)),
        "gephi_nodes": int(len(nodes)),
        "gephi_directed_edges": int(len(directed_edges)),
        "private_mapping_path_not_for_commit": str(OUT_PRIVATE_MAPPING.relative_to(ROOT)),
        "outputs": [
            str(path.relative_to(ROOT))
            for path in [
                OUT_EVENTS,
                OUT_PROBABLE,
                OUT_PAIRS,
                OUT_SUMMARY,
                OUT_BY_DIRECTION,
                OUT_BY_MASS_POSITION,
                OUT_BY_HCC,
                OUT_BY_BRAND,
                OUT_BY_SENTIMENT,
                OUT_LCN_VS_NON,
                OUT_MULTI_EVIDENCE,
                OUT_VALIDATION_AUDIT,
                OUT_GEPHI_NODES,
                OUT_GEPHI_EDGES,
                OUT_GEPHI_EDGES_UNDIRECTED,
                OUT_INTEGRITY,
            ]
        ],
    }
    OUT_RUN_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("RM2 COMMUNITY-MASS DIRECT INTERACTION")
    print(f"- analysis scope: {ANALYSIS_SCOPE}")
    print(f"- verified direct reply events: {len(verified)}")
    print(f"- unique Community-Mass account pairs: {len(pairs)}")
    print(f"- Community accounts: {verified['community_account'].nunique() if not verified.empty else 0}")
    print(f"- Mass accounts: {verified['mass_account'].nunique() if not verified.empty else 0}")
    print(f"- Mass Outside LCN accounts: {pairs['mass_network_position'].eq('Outside LCN').sum() if not pairs.empty else 0}")
    print(f"- directed Gephi nodes: {len(nodes)}")
    print(f"- directed Gephi edges: {len(directed_edges)}")
    print(f"- sentiment attribute status: {SENTIMENT_PENDING}")
    print("- validation: PASS")


if __name__ == "__main__":
    main()
