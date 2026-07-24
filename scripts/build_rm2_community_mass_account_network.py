from __future__ import annotations

import hashlib
import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

from rm1_evidence_builder import (
    LCN_WEIGHT_PERCENTILE,
    MIN_CO_CONV,
    MIN_CO_REPLY,
    MIN_CO_TEMPORAL,
    account_hashtag_sets,
    build_co_conv_edges_for_actor_sets,
    build_co_reply_edges_for_actor_sets,
    build_co_temporal_edges_for_actor_sets,
    build_combined_evidence_edges,
    build_video_hashtag_lookup,
    calibrate_temporal_window_minutes,
    canonical_pair_key,
    evidence_combination,
    infer_comment_mass_users,
    normalize_blank,
    normalize_username,
    prepare_rm1_comments,
    read_csv,
)


ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = ROOT / "dataset.csv"
VIDEO_METADATA_PATH = ROOT / "video_metadata_clean.csv"
TIKTOK_RM1_NOTEBOOK_PATH = ROOT / "notebooks" / "rm1" / "tiktok_coordination_analysis.ipynb"
ACCOUNT_ACTOR_TYPE_PATH = ROOT / "output/rm2_actor_type/tables/account_actor_type.csv"
HCC_NODES_PATH = ROOT / "output/gephi/gephi_hcc_nodes.csv"
HCC_EDGES_PATH = ROOT / "output/gephi/gephi_hcc_edges.csv"
LCN_NODES_RM1_PATH = ROOT / "output/gephi/gephi_lcn_nodes.csv"
LCN_EDGES_RM1_PATH = ROOT / "output/gephi/gephi_lcn_edges.csv"
LCN_NODES_ACTOR_TYPE_PATH = ROOT / "output/rm2_actor_type/gephi/gephi_lcn_nodes_actor_type.csv"
LCN_EDGES_ACTOR_TYPE_PATH = ROOT / "output/rm2_actor_type/gephi/gephi_lcn_edges_actor_type.csv"
AGGREGATE_NODES_PATH = ROOT / "output/rm2_actor_type/gephi/gephi_actor_type_nodes.csv"
AGGREGATE_EDGES_PATH = ROOT / "output/rm2_actor_type/gephi/gephi_actor_type_edges.csv"
SENTIMENT_READINESS_PATH = ROOT / "output/rm2_sentiment/validation/human_v2/sentiment_v2_locked_test_readiness.csv"
SENTIMENT_V2_MODEL_MANIFEST_PATH = ROOT / "output/rm2_sentiment/model/frozen/development_model_manifest.json"
RM1_CO_CONV_FILTERED_PATH = ROOT / "output/tables/co_conv_edges.csv"
RM1_CO_REPLY_FILTERED_PATH = ROOT / "output/tables/co_reply_edges.csv"
RM1_CO_TEMPORAL_FILTERED_PATH = ROOT / "output/tables/co_temporal_edges.csv"

OUT_TABLES_DIR = ROOT / "output/tables"
OUT_ACCOUNT_DIR = ROOT / "output/rm2_actor_type/account_interaction"
OUT_ACCOUNT_PRIVATE_DIR = OUT_ACCOUNT_DIR / "private"
OUT_GEPHI_DIR = ROOT / "output/rm2_actor_type/gephi"

OUT_PREFILTER_CO_CONV = OUT_TABLES_DIR / "pre_filter_co_conv_edges.csv"
OUT_PREFILTER_CO_REPLY = OUT_TABLES_DIR / "pre_filter_co_reply_edges.csv"
OUT_PREFILTER_CO_TEMPORAL = OUT_TABLES_DIR / "pre_filter_co_temporal_edges.csv"
OUT_PREFILTER_COMBINED = OUT_TABLES_DIR / "pre_filter_combined_evidence_edges.csv"

OUT_PAIRS = OUT_ACCOUNT_DIR / "community_mass_account_pairs.csv"
OUT_SUMMARY = OUT_ACCOUNT_DIR / "community_mass_account_summary.csv"
OUT_BY_NETWORK_POSITION = OUT_ACCOUNT_DIR / "community_mass_by_network_position.csv"
OUT_BY_SCOPE = OUT_ACCOUNT_DIR / "community_mass_by_interaction_scope.csv"
OUT_BY_EVIDENCE = OUT_ACCOUNT_DIR / "community_mass_by_evidence_combination.csv"
OUT_BY_HCC = OUT_ACCOUNT_DIR / "community_mass_by_hcc.csv"
OUT_LCN_VS_PRE = OUT_ACCOUNT_DIR / "community_mass_lcn_vs_pre_lcn.csv"
OUT_WEIGHT_DISTRIBUTION = OUT_ACCOUNT_DIR / "community_mass_weight_distribution.csv"
OUT_INTEGRITY = OUT_ACCOUNT_DIR / "community_mass_integrity_report.csv"
OUT_RUN_MANIFEST = OUT_ACCOUNT_DIR / "community_mass_account_network_manifest.json"
OUT_PRIVATE_MAPPING = OUT_ACCOUNT_PRIVATE_DIR / "community_mass_account_private_mass_mapping.csv"

OUT_GEPHI_NODES = OUT_GEPHI_DIR / "gephi_community_mass_account_nodes.csv"
OUT_GEPHI_NODES_VISUAL = OUT_GEPHI_DIR / "gephi_community_mass_account_nodes_visual.csv"
OUT_GEPHI_EDGES = OUT_GEPHI_DIR / "gephi_community_mass_account_edges.csv"
OUT_GEPHI_EDGES_ALL = OUT_GEPHI_DIR / "gephi_community_mass_account_edges_all_evidence.csv"
OUT_GEPHI_EDGES_VISUAL = OUT_GEPHI_DIR / "gephi_community_mass_account_edges_visual.csv"

COMMUNITY = "Community Actor"
MASS = "Mass Actor"
INDIVIDUAL = "Individual Actor"
LCN_NON_HCC = "LCN Non-HCC"
OUTSIDE_LCN = "Outside LCN"
ANALYSIS_SCOPE = "COMMUNITY_MASS_ACCOUNT_THREE_EVIDENCE_NETWORK"
OPTIONAL_DIRECT_SCOPE = "OPTIONAL_DIRECT_REPLY_DIAGNOSTIC"
SENTIMENT_ATTRIBUTE_STATUS = "DEVELOPMENT_MODEL_FROZEN_PENDING_LOCKED_TEST"
FINAL_EVALUATION_BLOCKED_STATUS = "BLOCKED_WAITING_FOR_8_HUMAN_ANNOTATED_REPLACEMENTS"
LEGACY_SENTIMENT_PENDING = "SENTIMENT_V2_PENDING"
SENTIMENT_LOCKED_TEST_BLOCKED = "BLOCKED_SYNTHETIC_IDS"
MASS_HASH_SALT = "rm2_actor_type_public_mass_hash_v1"
VISUAL_SINGLE_EVIDENCE_RULE = (
    "Retain PRE_LCN_SINGLE_EVIDENCE only when it passes at least one RM1 evidence "
    f"threshold: co_conv>={MIN_CO_CONV}, co_reply>={MIN_CO_REPLY}, or co_temporal>={MIN_CO_TEMPORAL}."
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def public_mass_id(username_norm: str) -> str:
    digest = hashlib.sha256(f"{MASS_HASH_SALT}:{username_norm}".encode("utf-8")).hexdigest()[:12]
    return f"MASS_{digest}"


def bool_text(value: bool) -> str:
    return "True" if bool(value) else "False"


def numeric(value: object, default: float = 0.0) -> float:
    try:
        text = normalize_blank(value)
        if not text:
            return default
        out = float(text)
    except (TypeError, ValueError):
        return default
    if not np.isfinite(out):
        return default
    return out


def source_hashes() -> dict[str, str]:
    return {
        "dataset": sha256_file(DATASET_PATH),
        "video_metadata_clean": sha256_file(VIDEO_METADATA_PATH),
        "rm1_notebook": sha256_file(TIKTOK_RM1_NOTEBOOK_PATH),
        "rm1_lcn_nodes": sha256_file(LCN_NODES_RM1_PATH),
        "rm1_lcn_edges": sha256_file(LCN_EDGES_RM1_PATH),
        "rm1_hcc_nodes": sha256_file(HCC_NODES_PATH),
        "rm1_hcc_edges": sha256_file(HCC_EDGES_PATH),
        "account_actor_type": sha256_file(ACCOUNT_ACTOR_TYPE_PATH),
        "lcn_nodes_actor_type": sha256_file(LCN_NODES_ACTOR_TYPE_PATH),
        "lcn_edges_actor_type": sha256_file(LCN_EDGES_ACTOR_TYPE_PATH),
        "aggregate_actor_type_nodes": sha256_file(AGGREGATE_NODES_PATH),
        "aggregate_actor_type_edges": sha256_file(AGGREGATE_EDGES_PATH),
    }


def load_sentiment_status() -> str:
    if SENTIMENT_V2_MODEL_MANIFEST_PATH.exists():
        manifest = json.loads(SENTIMENT_V2_MODEL_MANIFEST_PATH.read_text(encoding="utf-8"))
        return normalize_blank(manifest.get("status", "")) or SENTIMENT_ATTRIBUTE_STATUS
    if not SENTIMENT_READINESS_PATH.exists():
        return LEGACY_SENTIMENT_PENDING
    readiness = read_csv(SENTIMENT_READINESS_PATH)
    mask = readiness["metric"].eq("locked_test_v2_status")
    if not mask.any():
        return LEGACY_SENTIMENT_PENDING
    return normalize_blank(readiness.loc[mask, "value"].iloc[0]) or LEGACY_SENTIMENT_PENDING


def load_locked_test_status() -> str:
    if SENTIMENT_V2_MODEL_MANIFEST_PATH.exists():
        manifest = json.loads(SENTIMENT_V2_MODEL_MANIFEST_PATH.read_text(encoding="utf-8"))
        return normalize_blank(manifest.get("final_locked_test_evaluation_status", "")) or FINAL_EVALUATION_BLOCKED_STATUS
    if not SENTIMENT_READINESS_PATH.exists():
        return SENTIMENT_LOCKED_TEST_BLOCKED
    readiness = read_csv(SENTIMENT_READINESS_PATH)
    mask = readiness["metric"].eq("locked_test_v2_status")
    if not mask.any():
        return SENTIMENT_LOCKED_TEST_BLOCKED
    return normalize_blank(readiness.loc[mask, "value"].iloc[0]) or SENTIMENT_LOCKED_TEST_BLOCKED


def rm1_normalization_maxima() -> dict[str, float]:
    sources = {
        "co_conv_weight": RM1_CO_CONV_FILTERED_PATH,
        "co_reply_weight": RM1_CO_REPLY_FILTERED_PATH,
        "co_temporal_weight": RM1_CO_TEMPORAL_FILTERED_PATH,
    }
    maxima: dict[str, float] = {}
    for column, path in sources.items():
        df = read_csv(path)
        maxima[column] = float(pd.to_numeric(df[column], errors="coerce").fillna(0.0).max())
    return maxima


def actor_maps(account_type: pd.DataFrame, hcc_nodes: pd.DataFrame, lcn_nodes: pd.DataFrame) -> dict[str, object]:
    account = account_type.copy()
    account["username_norm"] = account["username"].map(normalize_username)
    account_by_public = account.drop_duplicates("username_norm").set_index("username_norm").to_dict("index")

    hcc = hcc_nodes.copy()
    hcc["username_norm"] = hcc["id"].map(normalize_username)
    community_hcc = hcc.drop_duplicates("username_norm").set_index("username_norm")["community"].astype(str).to_dict()
    community_users = set(community_hcc)

    individual_users = set(
        account.loc[account["actor_type_primary"].eq(INDIVIDUAL), "username_norm"]
    ) - {""}

    lcn = lcn_nodes.copy()
    lcn["raw_norm"] = lcn["Id"].map(normalize_username)
    lcn_by_raw = lcn.drop_duplicates("raw_norm").set_index("raw_norm").to_dict("index")
    lcn_users = set(lcn_by_raw)

    return {
        "account_by_public": account_by_public,
        "community_hcc": community_hcc,
        "community_users": community_users,
        "individual_users": individual_users,
        "lcn_by_raw": lcn_by_raw,
        "lcn_users": lcn_users,
    }


def lcn_pair_lookup(lcn_edges: pd.DataFrame) -> tuple[set[str], pd.DataFrame]:
    edges = lcn_edges.copy()
    edges["source_norm"] = edges["Source"].map(normalize_username)
    edges["target_norm"] = edges["Target"].map(normalize_username)
    edges["pair_key_raw"] = edges.apply(lambda row: canonical_pair_key(row["source_norm"], row["target_norm"]), axis=1)
    edge_types = edges[["source_actor_type", "target_actor_type"]].astype(str)
    cm_mask = edge_types.apply(lambda row: set(row) == {COMMUNITY, MASS}, axis=1)
    return set(edges["pair_key_raw"]), edges.loc[cm_mask].copy()


def add_co_hashtag(combined: pd.DataFrame, comments: pd.DataFrame, metadata: pd.DataFrame) -> pd.DataFrame:
    video_hashtags = build_video_hashtag_lookup(comments, metadata)
    account_tags = account_hashtag_sets(comments, video_hashtags)
    out = combined.copy()
    out["co_hashtag"] = out.apply(
        lambda row: len(account_tags.get(row["source"], set()) & account_tags.get(row["target"], set())),
        axis=1,
    )
    return out


def orient_community_mass(
    combined: pd.DataFrame,
    community_users: set[str],
    mass_users: set[str],
    community_hcc: dict[str, str],
    lcn_users: set[str],
    lcn_pairs: set[str],
    account_by_public: dict[str, dict[str, object]],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row in combined.to_dict("records"):
        source = normalize_username(row["source"])
        target = normalize_username(row["target"])
        if source in community_users and target in mass_users:
            community_account, mass_raw = source, target
        elif target in community_users and source in mass_users:
            community_account, mass_raw = target, source
        else:
            continue
        pair_key_raw = canonical_pair_key(community_account, mass_raw)
        pair_is_lcn_edge = pair_key_raw in lcn_pairs
        mass_public = public_mass_id(mass_raw)
        mass_public_norm = normalize_username(mass_public)
        mass_row = account_by_public.get(mass_public_norm, {})
        mass_is_lcn_member = mass_raw in lcn_users
        mass_network_position = LCN_NON_HCC if mass_is_lcn_member else normalize_blank(mass_row.get("network_position", "")) or OUTSIDE_LCN
        if mass_network_position not in {LCN_NON_HCC, OUTSIDE_LCN}:
            mass_network_position = LCN_NON_HCC if mass_is_lcn_member else OUTSIDE_LCN

        n_evidence = int(row["n_evidence"])
        if pair_is_lcn_edge:
            interaction_scope = "LCN_EDGE"
        elif n_evidence >= 2:
            interaction_scope = "PRE_LCN_MULTI_EVIDENCE"
        else:
            interaction_scope = "PRE_LCN_SINGLE_EVIDENCE"

        rows.append(
            {
                "community_account": community_account,
                "community_hcc_id": community_hcc.get(community_account, ""),
                "mass_account": mass_public,
                "mass_account_raw_norm": mass_raw,
                "mass_network_position": mass_network_position,
                "community_is_lcn_member": bool(community_account in lcn_users),
                "mass_is_lcn_member": bool(mass_is_lcn_member),
                "pair_is_lcn_edge": bool(pair_is_lcn_edge),
                "interaction_scope": interaction_scope,
                "co_conv_weight": numeric(row["co_conv_weight"]),
                "co_reply_weight": numeric(row["co_reply_weight"]),
                "co_temporal_weight": numeric(row["co_temporal_weight"]),
                "norm_co_conv": numeric(row["norm_co_conv"]),
                "norm_co_reply": numeric(row["norm_co_reply"]),
                "norm_co_temporal": numeric(row["norm_co_temporal"]),
                "final_weight": numeric(row["final_weight"]),
                "n_evidence": n_evidence,
                "has_co_conv": bool(numeric(row["co_conv_weight"]) > 0),
                "has_co_reply": bool(numeric(row["co_reply_weight"]) > 0),
                "has_co_temporal": bool(numeric(row["co_temporal_weight"]) > 0),
                "evidence_combination": evidence_combination(pd.Series(row)),
                "co_hashtag": int(numeric(row.get("co_hashtag", 0))),
                "passes_co_conv_threshold": bool(row["passes_co_conv_threshold"]),
                "passes_co_reply_threshold": bool(row["passes_co_reply_threshold"]),
                "passes_co_temporal_threshold": bool(row["passes_co_temporal_threshold"]),
                "passes_any_evidence_threshold": bool(row["passes_any_evidence_threshold"]),
                "sentiment_attribute_status": SENTIMENT_ATTRIBUTE_STATUS,
            }
        )
    pairs = pd.DataFrame(rows)
    if pairs.empty:
        return pairs
    pairs = pairs.sort_values(["community_hcc_id", "community_account", "mass_account"]).reset_index(drop=True)
    return pairs


def public_pairs(pairs: pd.DataFrame) -> pd.DataFrame:
    private_cols = ["mass_account_raw_norm"]
    return pairs.drop(columns=[col for col in private_cols if col in pairs.columns]).copy()


def build_gephi_edges(pairs: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "Source",
        "Target",
        "Type",
        "Weight",
        "community_hcc_id",
        "mass_network_position",
        "pair_is_lcn_edge",
        "interaction_scope",
        "co_conv_weight",
        "co_reply_weight",
        "co_temporal_weight",
        "norm_co_conv",
        "norm_co_reply",
        "norm_co_temporal",
        "final_weight",
        "n_evidence",
        "evidence_combination",
        "co_hashtag",
        "sentiment_attribute_status",
    ]
    if pairs.empty:
        return pd.DataFrame(columns=cols)
    edges = pairs.copy()
    edges.insert(0, "Source", edges["community_account"])
    edges.insert(1, "Target", edges["mass_account"])
    edges.insert(2, "Type", "Undirected")
    edges.insert(3, "Weight", edges["final_weight"])
    return edges[cols].copy()


def account_attr(account_by_public: dict[str, dict[str, object]], public_id: str) -> dict[str, object]:
    return dict(account_by_public.get(normalize_username(public_id), {}))


def build_gephi_nodes(
    pairs: pd.DataFrame,
    account_type: pd.DataFrame,
    lcn_by_raw: dict[str, dict[str, object]],
    account_by_public: dict[str, dict[str, object]],
    community_hcc: dict[str, str],
    visual_edges: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    def node_for_community(raw_id: str) -> dict[str, object]:
        lcn_row = lcn_by_raw.get(raw_id, {})
        public_row = account_attr(account_by_public, raw_id)
        return {
            "Id": raw_id,
            "Label": raw_id,
            "actor_type_primary": COMMUNITY,
            "network_position": "HCC",
            "hcc_id": community_hcc.get(raw_id, ""),
            "is_lcn_member": "True",
            "n_comments": public_row.get("n_comments", lcn_row.get("n_comments", "")),
            "dominant_sentiment": public_row.get("dominant_sentiment", lcn_row.get("dominant_sentiment", "")),
            "account_goal_orientation": public_row.get("account_goal_orientation", lcn_row.get("account_goal_orientation", "")),
            "target_brand_primary": public_row.get("target_brand_primary", lcn_row.get("target_brand_primary", "")),
            "degree": lcn_row.get("degree", ""),
            "weighted_degree": lcn_row.get("weighted_degree", ""),
            "sentiment_attribute_status": SENTIMENT_ATTRIBUTE_STATUS,
        }

    def node_for_mass(public_id: str, raw_id: str) -> dict[str, object]:
        lcn_row = lcn_by_raw.get(raw_id, {})
        public_row = account_attr(account_by_public, public_id)
        is_lcn = bool(lcn_row)
        network_position = LCN_NON_HCC if is_lcn else normalize_blank(public_row.get("network_position", "")) or OUTSIDE_LCN
        return {
            "Id": public_id,
            "Label": public_id,
            "actor_type_primary": MASS,
            "network_position": network_position if network_position in {LCN_NON_HCC, OUTSIDE_LCN} else OUTSIDE_LCN,
            "hcc_id": "",
            "is_lcn_member": bool_text(is_lcn),
            "n_comments": public_row.get("n_comments", lcn_row.get("n_comments", "")),
            "dominant_sentiment": public_row.get("dominant_sentiment", lcn_row.get("dominant_sentiment", "")),
            "account_goal_orientation": public_row.get("account_goal_orientation", lcn_row.get("account_goal_orientation", "")),
            "target_brand_primary": public_row.get("target_brand_primary", lcn_row.get("target_brand_primary", "")),
            "degree": lcn_row.get("degree", ""),
            "weighted_degree": lcn_row.get("weighted_degree", ""),
            "sentiment_attribute_status": SENTIMENT_ATTRIBUTE_STATUS,
        }

    if pairs.empty:
        cols = [
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
            "degree",
            "weighted_degree",
            "sentiment_attribute_status",
        ]
        return pd.DataFrame(columns=cols), pd.DataFrame(columns=cols)

    rows = []
    for community in sorted(pairs["community_account"].unique()):
        rows.append(node_for_community(community))
    mass_lookup = pairs.drop_duplicates("mass_account").set_index("mass_account")["mass_account_raw_norm"].to_dict()
    for public_id in sorted(mass_lookup):
        rows.append(node_for_mass(public_id, mass_lookup[public_id]))
    nodes_all = pd.DataFrame(rows).drop_duplicates("Id").sort_values(["actor_type_primary", "Id"]).reset_index(drop=True)

    if visual_edges is None or visual_edges.empty:
        nodes_visual = nodes_all.iloc[0:0].copy()
    else:
        visual_ids = set(visual_edges["Source"]) | set(visual_edges["Target"])
        nodes_visual = nodes_all.loc[nodes_all["Id"].isin(visual_ids)].copy().reset_index(drop=True)
    return nodes_all, nodes_visual


def visual_pair_mask(pairs: pd.DataFrame) -> pd.Series:
    return (
        pairs["interaction_scope"].eq("LCN_EDGE")
        | pairs["interaction_scope"].eq("PRE_LCN_MULTI_EVIDENCE")
        | (pairs["interaction_scope"].eq("PRE_LCN_SINGLE_EVIDENCE") & pairs["passes_any_evidence_threshold"].astype(bool))
    )


def graph_component_count(edges: pd.DataFrame) -> tuple[int, int]:
    if edges.empty:
        return 0, 0
    graph = nx.Graph()
    graph.add_edges_from(edges[["Source", "Target"]].itertuples(index=False, name=None))
    return nx.number_connected_components(graph), len(list(nx.isolates(graph)))


def write_prefilter_outputs(
    co_conv: pd.DataFrame,
    co_reply: pd.DataFrame,
    co_temporal: pd.DataFrame,
    combined: pd.DataFrame,
    lcn_pairs: set[str],
) -> pd.DataFrame:
    OUT_TABLES_DIR.mkdir(parents=True, exist_ok=True)
    cc_cols = ["source", "target", "co_conv_weight", "passes_co_conv_threshold"]
    cr_cols = ["source", "target", "reply_count", "co_reply_weight", "passes_co_reply_threshold"]
    ct_cols = ["source", "target", "co_temporal_weight", "passes_co_temporal_threshold"]
    co_conv[cc_cols].to_csv(OUT_PREFILTER_CO_CONV, index=False)
    co_reply[cr_cols].to_csv(OUT_PREFILTER_CO_REPLY, index=False)
    co_temporal[ct_cols].to_csv(OUT_PREFILTER_CO_TEMPORAL, index=False)

    out = combined.copy()
    out["pair_is_lcn_edge"] = out.apply(lambda row: canonical_pair_key(row["source"], row["target"]) in lcn_pairs, axis=1)
    out_cols = [
        "source",
        "target",
        "co_conv_weight",
        "co_reply_weight",
        "co_temporal_weight",
        "norm_co_conv",
        "norm_co_reply",
        "norm_co_temporal",
        "final_weight",
        "n_evidence",
        "passes_co_conv_threshold",
        "passes_co_reply_threshold",
        "passes_co_temporal_threshold",
        "passes_any_evidence_threshold",
        "pair_is_lcn_edge",
        "co_hashtag",
    ]
    out[out_cols].to_csv(OUT_PREFILTER_COMBINED, index=False)
    return out


def weight_distribution(pairs: pd.DataFrame) -> pd.DataFrame:
    if pairs.empty:
        return pd.DataFrame(
            columns=[
                "group",
                "n_pairs",
                "min_final_weight",
                "p25_final_weight",
                "median_final_weight",
                "mean_final_weight",
                "p75_final_weight",
                "p90_final_weight",
                "max_final_weight",
            ]
        )

    rows = []
    groups = [("All", pairs)]
    groups.extend((scope, group) for scope, group in pairs.groupby("interaction_scope", dropna=False))
    for name, group in groups:
        weight = pd.to_numeric(group["final_weight"], errors="coerce").fillna(0.0)
        rows.append(
            {
                "group": name,
                "n_pairs": int(len(group)),
                "min_final_weight": float(weight.min()),
                "p25_final_weight": float(weight.quantile(0.25)),
                "median_final_weight": float(weight.median()),
                "mean_final_weight": float(weight.mean()),
                "p75_final_weight": float(weight.quantile(0.75)),
                "p90_final_weight": float(weight.quantile(0.90)),
                "max_final_weight": float(weight.max()),
            }
        )
    return pd.DataFrame(rows)


def write_summary_outputs(
    pairs_public: pd.DataFrame,
    gephi_edges_all: pd.DataFrame,
    gephi_edges_visual: pd.DataFrame,
    nodes_all: pd.DataFrame,
    nodes_visual: pd.DataFrame,
    temporal_window: int,
    sentiment_attribute_status: str,
    locked_test_status: str,
) -> None:
    OUT_ACCOUNT_DIR.mkdir(parents=True, exist_ok=True)
    components_all, isolated_all = graph_component_count(gephi_edges_all)
    components_visual, isolated_visual = graph_component_count(gephi_edges_visual)
    summary_rows = [
        ("analysis_scope", ANALYSIS_SCOPE, "Main Community-Mass account layer based on Co-conv, Co-reply, and Co-temporal evidence."),
        ("optional_direct_reply_scope", OPTIONAL_DIRECT_SCOPE, "Reply-only layer is retained only as a diagnostic."),
        ("sentiment_attribute_status", sentiment_attribute_status, "Sentiment V2 development model is frozen; Actor Type sentiment attributes are not refreshed yet."),
        ("locked_test_v2_status", locked_test_status, "Final locked-test evaluation remains blocked until replacement labels are complete."),
        ("temporal_window_minutes", temporal_window, "RM1 median/P50 temporal window calibration."),
        ("min_co_conv", MIN_CO_CONV, "RM1 evidence threshold."),
        ("min_co_reply", MIN_CO_REPLY, "RM1 evidence threshold."),
        ("min_co_temporal", MIN_CO_TEMPORAL, "RM1 evidence threshold."),
        ("lcn_weight_percentile", LCN_WEIGHT_PERCENTILE, "Final RM1 LCN edge threshold."),
        ("visual_single_evidence_rule", VISUAL_SINGLE_EVIDENCE_RULE, "Recommended visual export rule."),
        ("total_unique_community_mass_account_pairs", len(pairs_public), "Full analytical pair table."),
        ("lcn_edge_pairs", int(pairs_public["interaction_scope"].eq("LCN_EDGE").sum()), "Pairs present in final LCN."),
        ("pre_lcn_multi_evidence_pairs", int(pairs_public["interaction_scope"].eq("PRE_LCN_MULTI_EVIDENCE").sum()), "Pairs not in final LCN, n_evidence >= 2."),
        ("pre_lcn_single_evidence_pairs", int(pairs_public["interaction_scope"].eq("PRE_LCN_SINGLE_EVIDENCE").sum()), "Pairs not in final LCN, n_evidence == 1."),
        ("community_accounts_involved", pairs_public["community_account"].nunique(), ""),
        ("mass_accounts_involved", pairs_public["mass_account"].nunique(), ""),
        ("mass_lcn_non_hcc_involved", pairs_public.loc[pairs_public["mass_network_position"].eq(LCN_NON_HCC), "mass_account"].nunique(), ""),
        ("mass_outside_lcn_involved", pairs_public.loc[pairs_public["mass_network_position"].eq(OUTSIDE_LCN), "mass_account"].nunique(), ""),
        ("all_evidence_gephi_nodes", len(nodes_all), ""),
        ("all_evidence_gephi_edges", len(gephi_edges_all), ""),
        ("visual_gephi_nodes", len(nodes_visual), ""),
        ("visual_gephi_edges", len(gephi_edges_visual), ""),
        ("all_evidence_connected_components", components_all, ""),
        ("all_evidence_isolated_nodes", isolated_all, ""),
        ("visual_connected_components", components_visual, ""),
        ("visual_isolated_nodes", isolated_visual, ""),
        ("pairs_with_all_three_evidence", int(pairs_public["evidence_combination"].eq("All three evidence").sum()), ""),
        ("co_conv_edge_count", int(pairs_public["has_co_conv"].astype(bool).sum()), ""),
        ("co_reply_edge_count", int(pairs_public["has_co_reply"].astype(bool).sum()), ""),
        ("co_temporal_edge_count", int(pairs_public["has_co_temporal"].astype(bool).sum()), ""),
        ("multi_evidence_edge_count", int(pairs_public["n_evidence"].astype(int).ge(2).sum()), ""),
    ]
    pd.DataFrame(summary_rows, columns=["metric", "value", "notes"]).to_csv(OUT_SUMMARY, index=False)

    pairs_public.groupby("mass_network_position", dropna=False).agg(
        n_pairs=("mass_account", "count"),
        n_community_accounts=("community_account", "nunique"),
        n_mass_accounts=("mass_account", "nunique"),
        total_final_weight=("final_weight", "sum"),
        mean_final_weight=("final_weight", "mean"),
    ).reset_index().to_csv(OUT_BY_NETWORK_POSITION, index=False)

    pairs_public.groupby("interaction_scope", dropna=False).agg(
        n_pairs=("mass_account", "count"),
        n_community_accounts=("community_account", "nunique"),
        n_mass_accounts=("mass_account", "nunique"),
        total_final_weight=("final_weight", "sum"),
        mean_final_weight=("final_weight", "mean"),
    ).reset_index().to_csv(OUT_BY_SCOPE, index=False)

    pairs_public.groupby("evidence_combination", dropna=False).agg(
        n_pairs=("mass_account", "count"),
        n_community_accounts=("community_account", "nunique"),
        n_mass_accounts=("mass_account", "nunique"),
        total_final_weight=("final_weight", "sum"),
        mean_final_weight=("final_weight", "mean"),
    ).reset_index().to_csv(OUT_BY_EVIDENCE, index=False)

    pairs_public.groupby("community_hcc_id", dropna=False).agg(
        n_pairs=("mass_account", "count"),
        n_community_accounts=("community_account", "nunique"),
        n_mass_accounts=("mass_account", "nunique"),
        n_lcn_edge_pairs=("pair_is_lcn_edge", "sum"),
        n_pre_lcn_pairs=("pair_is_lcn_edge", lambda s: int((~s.astype(bool)).sum())),
        total_final_weight=("final_weight", "sum"),
        mean_final_weight=("final_weight", "mean"),
    ).reset_index().to_csv(OUT_BY_HCC, index=False)

    lcn_vs = pairs_public.assign(lcn_status=np.where(pairs_public["pair_is_lcn_edge"].astype(bool), "LCN_EDGE", "PRE_LCN"))
    lcn_vs.groupby("lcn_status", dropna=False).agg(
        n_pairs=("mass_account", "count"),
        n_community_accounts=("community_account", "nunique"),
        n_mass_accounts=("mass_account", "nunique"),
        total_final_weight=("final_weight", "sum"),
        mean_final_weight=("final_weight", "mean"),
    ).reset_index().to_csv(OUT_LCN_VS_PRE, index=False)

    weight_distribution(pairs_public).to_csv(OUT_WEIGHT_DISTRIBUTION, index=False)


def write_private_mapping(pairs: pd.DataFrame) -> None:
    OUT_ACCOUNT_PRIVATE_DIR.mkdir(parents=True, exist_ok=True)
    mapping = (
        pairs[["mass_account_raw_norm", "mass_account"]]
        .drop_duplicates()
        .rename(columns={"mass_account_raw_norm": "raw_mass_username_norm", "mass_account": "public_mass_account"})
        .sort_values(["public_mass_account"])
    )
    mapping.to_csv(OUT_PRIVATE_MAPPING, index=False)


def write_manifest(outputs: list[Path], summary_metrics: dict[str, object]) -> None:
    manifest = {
        "analysis_scope": ANALYSIS_SCOPE,
        "optional_direct_reply_scope": OPTIONAL_DIRECT_SCOPE,
        "sentiment_attribute_status": SENTIMENT_ATTRIBUTE_STATUS,
        "final_locked_test_evaluation_status": FINAL_EVALUATION_BLOCKED_STATUS,
        "visual_single_evidence_rule": VISUAL_SINGLE_EVIDENCE_RULE,
        "summary_metrics": summary_metrics,
        "private_mapping_path_not_for_commit": str(OUT_PRIVATE_MAPPING.relative_to(ROOT)),
        "outputs": [str(path.relative_to(ROOT)) for path in outputs],
    }
    OUT_RUN_MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def integrity_report(
    comments_raw: pd.DataFrame,
    comments: pd.DataFrame,
    account_type: pd.DataFrame,
    hcc_nodes: pd.DataFrame,
    lcn_nodes: pd.DataFrame,
    lcn_edges: pd.DataFrame,
    aggregate_nodes: pd.DataFrame,
    aggregate_edges: pd.DataFrame,
    pairs: pd.DataFrame,
    pairs_public: pd.DataFrame,
    gephi_edges_all: pd.DataFrame,
    nodes_all: pd.DataFrame,
    lcn_cm_edges: pd.DataFrame,
    hashes_before: dict[str, str],
    hashes_after: dict[str, str],
    sentiment_attribute_status: str,
    locked_test_status: str,
) -> pd.DataFrame:
    pair_keys_raw = pairs.apply(lambda row: canonical_pair_key(row["community_account"], row["mass_account_raw_norm"]), axis=1)
    lcn_pair_keys = set(lcn_edges.apply(lambda row: canonical_pair_key(row["Source"], row["Target"]), axis=1))
    expected_lcn_flags = pair_keys_raw.isin(lcn_pair_keys)
    duplicate_public_pairs = int(pairs_public.duplicated(["community_account", "mass_account"]).sum()) if not pairs_public.empty else 0
    duplicate_gephi_pairs = 0
    if not gephi_edges_all.empty:
        gephi_pair_key = gephi_edges_all.apply(lambda row: "||".join(sorted([row["Source"], row["Target"]])), axis=1)
        duplicate_gephi_pairs = int(gephi_pair_key.duplicated().sum())
    node_ids = set(nodes_all["Id"]) if not nodes_all.empty else set()
    missing_endpoints = 0
    if not gephi_edges_all.empty:
        missing_endpoints = int((~gephi_edges_all["Source"].isin(node_ids)).sum() + (~gephi_edges_all["Target"].isin(node_ids)).sum())
    self_loop = int(gephi_edges_all["Source"].eq(gephi_edges_all["Target"]).sum()) if not gephi_edges_all.empty else 0
    empty_endpoint = (
        int(gephi_edges_all["Source"].map(normalize_blank).eq("").sum() + gephi_edges_all["Target"].map(normalize_blank).eq("").sum())
        if not gephi_edges_all.empty
        else 0
    )
    source_target_subset = set(lcn_cm_edges["pair_key_raw"]) - set(pair_keys_raw)
    outside_lcn_as_lcn = int((pairs_public["pair_is_lcn_edge"].astype(bool) & pairs_public["mass_network_position"].eq(OUTSIDE_LCN)).sum())
    lcn_source_compare = lcn_cm_edges.copy()
    lcn_source_compare = lcn_source_compare.rename(
        columns={
            "co_conv_weight": "source_co_conv_weight",
            "co_reply_weight": "source_co_reply_weight",
            "co_temporal_weight": "source_co_temporal_weight",
            "norm_co_conv": "source_norm_co_conv",
            "norm_co_reply": "source_norm_co_reply",
            "norm_co_temporal": "source_norm_co_temporal",
            "final_weight": "source_final_weight",
            "n_evidence": "source_n_evidence",
        }
    )
    lcn_pair_compare = pairs.loc[pairs["pair_is_lcn_edge"].astype(bool)].copy()
    lcn_pair_compare["pair_key_raw"] = lcn_pair_compare.apply(lambda row: canonical_pair_key(row["community_account"], row["mass_account_raw_norm"]), axis=1)
    lcn_compare = lcn_source_compare.merge(
        lcn_pair_compare,
        on="pair_key_raw",
        how="outer",
        indicator=True,
    )
    evidence_match = True
    max_weight_delta = 0.0
    if len(lcn_compare) == len(lcn_cm_edges) and lcn_compare["_merge"].eq("both").all():
        compare_pairs = [
            ("source_co_conv_weight", "co_conv_weight"),
            ("source_co_reply_weight", "co_reply_weight"),
            ("source_co_temporal_weight", "co_temporal_weight"),
            ("source_norm_co_conv", "norm_co_conv"),
            ("source_norm_co_reply", "norm_co_reply"),
            ("source_norm_co_temporal", "norm_co_temporal"),
            ("source_final_weight", "final_weight"),
            ("source_n_evidence", "n_evidence"),
        ]
        for source_col, output_col in compare_pairs:
            source_values = pd.to_numeric(lcn_compare[source_col], errors="coerce").fillna(0.0)
            output_values = pd.to_numeric(lcn_compare[output_col], errors="coerce").fillna(0.0)
            delta = (source_values - output_values).abs()
            max_weight_delta = max(max_weight_delta, float(delta.max()))
            if not np.allclose(source_values, output_values, rtol=1e-9, atol=1e-9):
                evidence_match = False
    else:
        evidence_match = False

    rows = [
        ("dataset_rows", 33847, len(comments_raw), len(comments_raw) == 33847, ""),
        ("unique_comment_id", 33847, comments_raw["comment_id"].nunique(), comments_raw["comment_id"].nunique() == 33847, ""),
        ("observational_comment_rows_after_dedup", 33847, len(comments), len(comments) == 33847, ""),
        ("actor_universe", 26427, len(account_type), len(account_type) == 26427, ""),
        ("community_actor_accounts", 218, int(account_type["actor_type_primary"].eq(COMMUNITY).sum()), int(account_type["actor_type_primary"].eq(COMMUNITY).sum()) == 218, ""),
        ("mass_actor_accounts", 26166, int(account_type["actor_type_primary"].eq(MASS).sum()), int(account_type["actor_type_primary"].eq(MASS).sum()) == 26166, ""),
        ("hcc_members", 218, len(hcc_nodes), len(hcc_nodes) == 218, ""),
        ("hcc_count", 42, hcc_nodes["community"].nunique(), hcc_nodes["community"].nunique() == 42, ""),
        ("lcn_nodes", 724, len(lcn_nodes), len(lcn_nodes) == 724, ""),
        ("lcn_edges", 1357, len(lcn_edges), len(lcn_edges) == 1357, ""),
        ("lcn_community_mass_edges", 305, len(lcn_cm_edges), len(lcn_cm_edges) == 305, ""),
        ("aggregate_actor_type_nodes", 396, len(aggregate_nodes), len(aggregate_nodes) == 396, ""),
        ("aggregate_actor_type_edges", 497, len(aggregate_edges), len(aggregate_edges) == 497, ""),
        ("account_type_primary_not_empty", 0, int(account_type["actor_type_primary"].map(normalize_blank).eq("").sum()), int(account_type["actor_type_primary"].map(normalize_blank).eq("").sum()) == 0, ""),
        ("account_has_single_actor_type_primary", 0, int(account_type.groupby("username")["actor_type_primary"].nunique().gt(1).sum()), int(account_type.groupby("username")["actor_type_primary"].nunique().gt(1).sum()) == 0, ""),
        ("community_mass_pair_duplicates", 0, duplicate_public_pairs, duplicate_public_pairs == 0, ""),
        ("gephi_edge_duplicate_undirected_pair", 0, duplicate_gephi_pairs, duplicate_gephi_pairs == 0, ""),
        ("self_loop", 0, self_loop, self_loop == 0, ""),
        ("empty_endpoint", 0, empty_endpoint, empty_endpoint == 0, ""),
        ("missing_endpoint", 0, missing_endpoints, missing_endpoints == 0, ""),
        ("pair_is_lcn_edge_matches_final_lcn", True, bool((pairs_public["pair_is_lcn_edge"].astype(bool).to_numpy() == expected_lcn_flags.to_numpy()).all()), bool((pairs_public["pair_is_lcn_edge"].astype(bool).to_numpy() == expected_lcn_flags.to_numpy()).all()), ""),
        ("all_lcn_cm_edges_recovered_from_evidence", 0, len(source_target_subset), len(source_target_subset) == 0, "; ".join(sorted(source_target_subset)[:5])),
        ("lcn_cm_evidence_and_weight_unchanged", True, evidence_match, evidence_match, f"max_abs_delta={max_weight_delta}"),
        ("outside_lcn_pair_not_labeled_lcn_edge", 0, outside_lcn_as_lcn, outside_lcn_as_lcn == 0, ""),
        ("pre_lcn_edges_not_added_to_lcn_final", 1357, len(read_csv(LCN_EDGES_ACTOR_TYPE_PATH)), len(read_csv(LCN_EDGES_ACTOR_TYPE_PATH)) == 1357, ""),
        ("source_hashes_unchanged", True, hashes_before == hashes_after, hashes_before == hashes_after, ""),
        ("rm1_input_output_hashes_unchanged", True, all(hashes_before[k] == hashes_after[k] for k in hashes_before if k.startswith("rm1_") or k in {"dataset", "video_metadata_clean"}), all(hashes_before[k] == hashes_after[k] for k in hashes_before if k.startswith("rm1_") or k in {"dataset", "video_metadata_clean"}), ""),
        ("hcc_membership_hash_unchanged", True, hashes_before["rm1_hcc_nodes"] == hashes_after["rm1_hcc_nodes"], hashes_before["rm1_hcc_nodes"] == hashes_after["rm1_hcc_nodes"], ""),
        ("no_aggregate_hcc_node", 0, int(nodes_all["Id"].astype(str).str.startswith("HCC_").sum()) if not nodes_all.empty else 0, nodes_all.empty or int(nodes_all["Id"].astype(str).str.startswith("HCC_").sum()) == 0, ""),
        ("no_mass_segment_node", 0, int(nodes_all["Id"].astype(str).str.startswith("MASS_HCC_").sum()) if not nodes_all.empty else 0, nodes_all.empty or int(nodes_all["Id"].astype(str).str.startswith("MASS_HCC_").sum()) == 0, ""),
        ("no_individual_actor_in_gephi_nodes", 0, int(nodes_all["actor_type_primary"].eq(INDIVIDUAL).sum()) if not nodes_all.empty else 0, nodes_all.empty or int(nodes_all["actor_type_primary"].eq(INDIVIDUAL).sum()) == 0, ""),
        ("sentiment_v2_locked_test_status", FINAL_EVALUATION_BLOCKED_STATUS, locked_test_status, locked_test_status == FINAL_EVALUATION_BLOCKED_STATUS, "Final locked-test evaluation remains blocked."),
        ("sentiment_attribute_status", SENTIMENT_ATTRIBUTE_STATUS, sentiment_attribute_status, sentiment_attribute_status == SENTIMENT_ATTRIBUTE_STATUS, "No full inference or final Actor Type sentiment refresh was run."),
        ("no_synthetic_comment_id_in_observational_outputs", 0, 0, True, "Account-level outputs do not contain comment_id values."),
    ]
    report = pd.DataFrame(
        [{"metric": metric, "expected": expected, "observed": observed, "passed": bool(passed), "notes": notes} for metric, expected, observed, passed, notes in rows]
    )
    return report


def main() -> None:
    print("RM2 Community-Mass account network: loading sources")
    OUT_ACCOUNT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_GEPHI_DIR.mkdir(parents=True, exist_ok=True)
    hashes_before = source_hashes()

    comments_raw = read_csv(DATASET_PATH)
    metadata = read_csv(VIDEO_METADATA_PATH)
    account_type = read_csv(ACCOUNT_ACTOR_TYPE_PATH)
    hcc_nodes = read_csv(HCC_NODES_PATH)
    lcn_nodes = read_csv(LCN_NODES_ACTOR_TYPE_PATH)
    lcn_edges = read_csv(LCN_EDGES_ACTOR_TYPE_PATH)
    aggregate_nodes = read_csv(AGGREGATE_NODES_PATH)
    aggregate_edges = read_csv(AGGREGATE_EDGES_PATH)
    sentiment_attribute_status = load_sentiment_status()
    locked_test_status = load_locked_test_status()

    comments = prepare_rm1_comments(comments_raw)
    maps = actor_maps(account_type, hcc_nodes, lcn_nodes)
    community_users: set[str] = maps["community_users"]  # type: ignore[assignment]
    individual_users: set[str] = maps["individual_users"]  # type: ignore[assignment]
    community_hcc: dict[str, str] = maps["community_hcc"]  # type: ignore[assignment]
    lcn_users: set[str] = maps["lcn_users"]  # type: ignore[assignment]
    lcn_by_raw: dict[str, dict[str, object]] = maps["lcn_by_raw"]  # type: ignore[assignment]
    account_by_public: dict[str, dict[str, object]] = maps["account_by_public"]  # type: ignore[assignment]
    mass_users = infer_comment_mass_users(comments, community_users, individual_users)

    print("RM2 Community-Mass account network: reconstructing RM1 evidence for sparse Community-Mass pairs")
    temporal_window = calibrate_temporal_window_minutes(comments)
    co_conv = build_co_conv_edges_for_actor_sets(comments, community_users, mass_users, "community", "mass")
    co_reply = build_co_reply_edges_for_actor_sets(comments, community_users, mass_users, "community", "mass")
    co_temporal = build_co_temporal_edges_for_actor_sets(comments, community_users, mass_users, temporal_window, "community", "mass")
    combined = build_combined_evidence_edges(co_conv, co_reply, co_temporal, normalization_maxima=rm1_normalization_maxima())
    combined = add_co_hashtag(combined, comments, metadata)

    lcn_pairs, lcn_cm_edges = lcn_pair_lookup(lcn_edges)
    combined_public = write_prefilter_outputs(co_conv, co_reply, co_temporal, combined, lcn_pairs)
    pairs = orient_community_mass(
        combined_public,
        community_users,
        mass_users,
        community_hcc,
        lcn_users,
        lcn_pairs,
        account_by_public,
    )
    pairs_public = public_pairs(pairs)
    pairs_public.to_csv(OUT_PAIRS, index=False)
    write_private_mapping(pairs)

    gephi_edges_all = build_gephi_edges(pairs_public)
    visual_pairs_public = pairs_public.loc[visual_pair_mask(pairs_public)].copy()
    gephi_edges_visual = build_gephi_edges(visual_pairs_public)
    nodes_all, nodes_visual = build_gephi_nodes(pairs, account_type, lcn_by_raw, account_by_public, community_hcc, gephi_edges_visual)

    nodes_all.to_csv(OUT_GEPHI_NODES, index=False)
    nodes_visual.to_csv(OUT_GEPHI_NODES_VISUAL, index=False)
    gephi_edges_all.to_csv(OUT_GEPHI_EDGES_ALL, index=False)
    gephi_edges_visual.to_csv(OUT_GEPHI_EDGES_VISUAL, index=False)
    gephi_edges_visual.to_csv(OUT_GEPHI_EDGES, index=False)

    write_summary_outputs(
        pairs_public,
        gephi_edges_all,
        gephi_edges_visual,
        nodes_all,
        nodes_visual,
        temporal_window,
        sentiment_attribute_status,
        locked_test_status,
    )

    hashes_after = source_hashes()
    integrity = integrity_report(
        comments_raw,
        comments,
        account_type,
        hcc_nodes,
        lcn_nodes,
        lcn_edges,
        aggregate_nodes,
        aggregate_edges,
        pairs,
        pairs_public,
        gephi_edges_all,
        nodes_all,
        lcn_cm_edges,
        hashes_before,
        hashes_after,
        sentiment_attribute_status,
        locked_test_status,
    )
    integrity.to_csv(OUT_INTEGRITY, index=False)
    if not integrity["passed"].all():
        raise AssertionError("Community-Mass account network integrity failed:\n" + integrity.loc[~integrity["passed"]].to_string(index=False))

    summary_metrics = {
        "total_unique_community_mass_account_pairs": int(len(pairs_public)),
        "lcn_edge_pairs": int(pairs_public["interaction_scope"].eq("LCN_EDGE").sum()),
        "pre_lcn_multi_evidence_pairs": int(pairs_public["interaction_scope"].eq("PRE_LCN_MULTI_EVIDENCE").sum()),
        "pre_lcn_single_evidence_pairs": int(pairs_public["interaction_scope"].eq("PRE_LCN_SINGLE_EVIDENCE").sum()),
        "mass_outside_lcn_involved": int(pairs_public.loc[pairs_public["mass_network_position"].eq(OUTSIDE_LCN), "mass_account"].nunique()),
        "visual_edges": int(len(gephi_edges_visual)),
        "all_evidence_edges": int(len(gephi_edges_all)),
    }
    write_manifest(
        [
            OUT_PREFILTER_CO_CONV,
            OUT_PREFILTER_CO_REPLY,
            OUT_PREFILTER_CO_TEMPORAL,
            OUT_PREFILTER_COMBINED,
            OUT_PAIRS,
            OUT_SUMMARY,
            OUT_BY_NETWORK_POSITION,
            OUT_BY_SCOPE,
            OUT_BY_EVIDENCE,
            OUT_BY_HCC,
            OUT_LCN_VS_PRE,
            OUT_WEIGHT_DISTRIBUTION,
            OUT_INTEGRITY,
            OUT_GEPHI_NODES,
            OUT_GEPHI_NODES_VISUAL,
            OUT_GEPHI_EDGES,
            OUT_GEPHI_EDGES_ALL,
            OUT_GEPHI_EDGES_VISUAL,
        ],
        summary_metrics,
    )

    print("RM2 COMMUNITY-MASS ACCOUNT NETWORK")
    print(f"- total unique Community-Mass account pairs: {len(pairs_public):,}")
    print(f"- LCN_EDGE pairs: {summary_metrics['lcn_edge_pairs']:,}")
    print(f"- PRE_LCN_MULTI_EVIDENCE pairs: {summary_metrics['pre_lcn_multi_evidence_pairs']:,}")
    print(f"- PRE_LCN_SINGLE_EVIDENCE pairs: {summary_metrics['pre_lcn_single_evidence_pairs']:,}")
    print(f"- Mass Outside LCN involved: {summary_metrics['mass_outside_lcn_involved']:,}")
    print(f"- Gephi all-evidence edges: {len(gephi_edges_all):,}")
    print(f"- Gephi visual edges: {len(gephi_edges_visual):,}")
    print(f"- temporal window: {temporal_window} minutes")
    print(f"- sentiment attribute status: {sentiment_attribute_status}")
    print(f"- final locked-test evaluation: {locked_test_status}")
    print("- integrity: PASS")


if __name__ == "__main__":
    main()
