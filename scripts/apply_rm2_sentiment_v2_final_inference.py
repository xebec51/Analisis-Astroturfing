from __future__ import annotations

import hashlib
import json
import platform
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


ROOT = Path(__file__).resolve().parents[1]
FINAL_DIR = ROOT / "output/rm2_sentiment/final_v2"
TABLE_DIR = FINAL_DIR / "tables"
GEPHI_DIR = FINAL_DIR / "gephi"
PRESENTATION_DIR = FINAL_DIR / "presentation"
MODEL_DIR = ROOT / "output/rm2_sentiment/model_v2"
HUMAN_DIR = ROOT / "output/rm2_sentiment/human_validation_v2"

DATASET = ROOT / "dataset.csv"
LEGACY_COMMENT_SENTIMENT = ROOT / "output/rm2_sentiment/tables/comment_sentiment.csv"
HCC_NODES = ROOT / "output/gephi/gephi_hcc_nodes.csv"
HCC_EDGES = ROOT / "output/gephi/gephi_hcc_edges.csv"
LCN_NODES_ACTOR_TYPE = ROOT / "output/rm2_actor_type/gephi/gephi_lcn_nodes_actor_type.csv"
ACTOR_UNIVERSE_SUMMARY = ROOT / "output/rm2_actor_type/tables/actor_type_universe_summary.csv"
COMMUNITY_MASS_PAIRS = ROOT / "output/rm2_actor_type/account_interaction/community_mass_account_pairs.csv"
COMMENT_SIM_SUMMARY = ROOT / "output/rm2_comment_similarity/comment_similarity_summary.csv"

MODEL_ARTIFACT = MODEL_DIR / "selected_model_development_frozen.joblib"
MODEL_CONFIG = MODEL_DIR / "selected_model_development_frozen_config.json"
EVALUATION_LOCK = MODEL_DIR / "final_locked_test_evaluation_lock.json"
EVALUATION_METRICS = MODEL_DIR / "final_locked_test_metrics.csv"
EVALUATION_ACCEPTANCE = MODEL_DIR / "final_locked_test_acceptance_decision.csv"
LOCKED_FINAL = HUMAN_DIR / "locked_test_v2_observational_final.csv"
LOCKED_FINAL_MANIFEST = HUMAN_DIR / "locked_test_v2_observational_final_manifest.json"
TRAINING_PROVENANCE = MODEL_DIR / "development_training_pool_provenance.csv"

OUT_OBS = FINAL_DIR / "comment_sentiment_v2_observational.csv"
OUT_INJ = FINAL_DIR / "comment_sentiment_v2_injected_diagnostic.csv"
OUT_INFERENCE_MANIFEST = FINAL_DIR / "sentiment_v2_inference_manifest.json"
OUT_REPORT = FINAL_DIR / "FINAL_SENTIMENT_ANALYSIS_REPORT.md"
OUT_SUMMARY = FINAL_DIR / "FINAL_SENTIMENT_ANALYSIS_SUMMARY.csv"
OUT_INTEGRITY = FINAL_DIR / "FINAL_SENTIMENT_ANALYSIS_INTEGRITY.csv"
OUT_MANIFEST = FINAL_DIR / "FINAL_SENTIMENT_ANALYSIS_MANIFEST.json"
OUT_PRES_MD = PRESENTATION_DIR / "sentiment_presentation_summary.md"
OUT_PRES_TABLES = PRESENTATION_DIR / "sentiment_presentation_tables.csv"

LABELS = ["Negative", "Neutral", "Positive"]
FINAL_STATUS_ACCEPTED = {"FINAL_MODEL_VALIDATED", "FINAL_MODEL_VALIDATED_WITH_COVERAGE_CAUTION"}
EXPECTED_MODEL_HASH = "477bfe11ebc3463eeff5a5fb8359f86022d719c8bea49dfdad0460fa0c5e2ccc"
EXPECTED_THRESHOLD = 0.42
RANDOM_SEED = 20260721


def mkdirs() -> None:
    for path in [FINAL_DIR, TABLE_DIR, GEPHI_DIR, PRESENTATION_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False, low_memory=False)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def clean_social(text: object) -> str:
    if pd.isna(text):
        return ""
    s = str(text).strip().lower()
    if s in {"", "nan", "none", "null", "<na>"}:
        return ""
    s = re.sub(r"https?://\S+|www\.\S+", " URL ", s)
    s = re.sub(r"@\w+", " USERMENTION ", s)
    s = re.sub(r"#(\w+)", r" \1 ", s)
    s = re.sub(r"([!?.,])\1+", r"\1", s)
    return re.sub(r"\s+", " ", s).strip()


def normalize_username(value: object) -> str:
    text = "" if pd.isna(value) else str(value).strip()
    return re.sub(r"\s+", "", text.lower().lstrip("@"))


def no_text_flag(text: object) -> bool:
    s = "" if pd.isna(text) else str(text).strip()
    return s.lower() in {"", "nan", "none", "null", "<na>", "[deleted]", "deleted"}


def softmax(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    if scores.ndim == 1:
        scores = scores.reshape(-1, 1)
    scores = scores - np.max(scores, axis=1, keepdims=True)
    exp = np.exp(scores)
    return exp / np.clip(exp.sum(axis=1, keepdims=True), 1e-12, None)


def predict_proba_aligned(model, x: pd.Series, label_encoder) -> np.ndarray:
    clf = model.named_steps["clf"]
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(x)
        classes = getattr(clf, "classes_", label_encoder.classes_)
    else:
        probs = softmax(model.decision_function(x))
        classes = getattr(clf, "classes_", label_encoder.classes_)

    out = np.zeros((len(x), len(label_encoder.classes_)), dtype=float)
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


def ensemble_predict_proba(artifact: dict, text: pd.Series) -> np.ndarray:
    parts = []
    for component in artifact["pipeline"]:
        parts.append(predict_proba_aligned(component["pipeline"], text, artifact["label_encoder"]))
    probs = np.mean(parts, axis=0)
    return probs / np.clip(probs.sum(axis=1, keepdims=True), 1e-12, None)


def wilson_ci(count: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (np.nan, np.nan)
    p = count / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * np.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    return ((centre - margin) / denom, (centre + margin) / denom)


def bootstrap_ci(values: pd.Series, seed: int = RANDOM_SEED, n_boot: int = 500) -> tuple[float, float]:
    arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if len(arr) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = [float(rng.choice(arr, size=len(arr), replace=True).mean()) for _ in range(n_boot)]
    return (float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975)))


def add_prediction_columns(df: pd.DataFrame, artifact: dict, config: dict, model_hash: str, config_hash: str, validation_status: str) -> pd.DataFrame:
    out = df.copy()
    out["comment_text_original"] = out["text"].astype(str)
    out["comment_text_model"] = out["comment_text_original"].map(clean_social)
    probs = ensemble_predict_proba(artifact, out["comment_text_model"])
    label_order = [str(x) for x in artifact["label_encoder"].classes_]
    for i, label in enumerate(label_order):
        out[f"probability_{label.lower()}"] = probs[:, i]
    out["predicted_sentiment"] = [label_order[i] for i in probs.argmax(axis=1)]
    out["max_probability"] = probs.max(axis=1)
    out["threshold"] = float(config["threshold"])
    out["no_text"] = out["comment_text_original"].map(no_text_flag)
    out["abstained"] = (~out["no_text"]) & (out["max_probability"].astype(float) < float(config["threshold"]))
    out["final_sentiment_label"] = np.where(
        out["no_text"],
        "No Text",
        np.where(out["abstained"], "Uncertain", out["predicted_sentiment"]),
    )
    out["sentiment_status"] = np.where(
        out["no_text"],
        "No Text",
        np.where(out["abstained"], "Uncertain", "Evaluable"),
    )
    out["model_name"] = config["model_name"]
    out["selected_candidate"] = config["selected_candidate_id"]
    out["model_hash"] = model_hash
    out["config_hash"] = config_hash
    out["locked_test_validation_status"] = validation_status
    out["inference_timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    out["dataset_provenance"] = "dataset.csv"
    return out


def sentiment_distribution(df: pd.DataFrame, group_label: str = "Observational") -> pd.DataFrame:
    total = len(df)
    rows = []
    for label in ["Positive", "Neutral", "Negative", "Uncertain", "No Text"]:
        count = int(df["final_sentiment_label"].eq(label).sum())
        lo, hi = wilson_ci(count, total)
        rows.append(
            {
                "group": group_label,
                "label": label,
                "count": count,
                "percentage_of_total": count / total * 100 if total else np.nan,
                "ci_low": lo,
                "ci_high": hi,
                "denominator": total,
            }
        )
    evaluable = int(df["final_sentiment_label"].isin(LABELS).sum())
    rows.append(
        {
            "group": group_label,
            "label": "Evaluable coverage",
            "count": evaluable,
            "percentage_of_total": evaluable / total * 100 if total else np.nan,
            "ci_low": np.nan,
            "ci_high": np.nan,
            "denominator": total,
        }
    )
    return pd.DataFrame(rows)


def hcc_vs_nonhcc_comment(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, group in df.groupby(np.where(df["is_hcc_member_bool"], "HCC", "Non-HCC")):
        total = len(group)
        evaluable = int(group["final_sentiment_label"].isin(LABELS).sum())
        row = {
            "group": name,
            "total_comments": total,
            "evaluable_comments": evaluable,
            "coverage": evaluable / total if total else np.nan,
            "uncertain_count": int(group["final_sentiment_label"].eq("Uncertain").sum()),
            "no_text_count": int(group["final_sentiment_label"].eq("No Text").sum()),
        }
        for label in LABELS:
            count = int(group["final_sentiment_label"].eq(label).sum())
            lo, hi = wilson_ci(count, evaluable)
            row[f"{label.lower()}_count"] = count
            row[f"{label.lower()}_ratio_evaluable"] = count / evaluable if evaluable else np.nan
            row[f"{label.lower()}_ratio_total"] = count / total if total else np.nan
            row[f"{label.lower()}_ci_low"] = lo
            row[f"{label.lower()}_ci_high"] = hi
        rows.append(row)
    result = pd.DataFrame(rows)
    if set(result["group"]) == {"HCC", "Non-HCC"}:
        h = result.set_index("group").loc["HCC"]
        n = result.set_index("group").loc["Non-HCC"]
        for label in LABELS:
            result[f"{label.lower()}_hcc_minus_nonhcc"] = h[f"{label.lower()}_ratio_evaluable"] - n[f"{label.lower()}_ratio_evaluable"]
    return result


def account_summary(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for account, group in df.groupby("username_norm", dropna=False):
        total = len(group)
        evaluable = int(group["final_sentiment_label"].isin(LABELS).sum())
        counts = {label: int(group["final_sentiment_label"].eq(label).sum()) for label in ["Positive", "Neutral", "Negative", "Uncertain", "No Text"]}
        ratios = {f"{label.lower()}_ratio": counts[label] / evaluable if evaluable else np.nan for label in LABELS}
        dominant = max(LABELS, key=lambda label: counts[label]) if evaluable else "No evaluable sentiment"
        rows.append(
            {
                "username_norm": account,
                "username": group["username"].iloc[0],
                "is_hcc_member": bool(group["is_hcc_member_bool"].any()),
                "n_comments": total,
                "evaluable_comments": evaluable,
                "coverage": evaluable / total if total else np.nan,
                **{f"{label.lower()}_count": counts[label] for label in ["Positive", "Neutral", "Negative", "Uncertain", "No Text"]},
                **ratios,
                "dominant_sentiment": dominant,
            }
        )
    account = pd.DataFrame(rows)
    summary_rows = []
    for name, group in account.groupby(np.where(account["is_hcc_member"], "HCC", "Non-HCC")):
        row = {"group": name, "n_accounts": len(group)}
        for label in LABELS:
            vals = group[f"{label.lower()}_ratio"].dropna()
            lo, hi = bootstrap_ci(vals)
            row[f"mean_{label.lower()}_ratio"] = float(vals.mean()) if len(vals) else np.nan
            row[f"median_{label.lower()}_ratio"] = float(vals.median()) if len(vals) else np.nan
            row[f"iqr_{label.lower()}_ratio"] = float(vals.quantile(0.75) - vals.quantile(0.25)) if len(vals) else np.nan
            row[f"mean_{label.lower()}_ratio_ci_low"] = lo
            row[f"mean_{label.lower()}_ratio_ci_high"] = hi
        row["median_coverage"] = float(group["coverage"].median()) if len(group) else np.nan
        row["mean_coverage"] = float(group["coverage"].mean()) if len(group) else np.nan
        summary_rows.append(row)
    return account, pd.DataFrame(summary_rows)


def classify_goal(pos: float, neu: float, neg: float, n_evaluable: int, coverage: float) -> tuple[str, str, str, float]:
    margin = max(pos, neu, neg) - sorted([pos, neu, neg])[-2]
    if n_evaluable < 5 or coverage < 0.5:
        confidence = "Low"
        status = "LOW_EVIDENCE_GOAL_ORIENTATION"
    elif margin >= 0.20 and n_evaluable >= 15:
        confidence = "High"
        status = "Assigned"
    elif margin >= 0.10 and n_evaluable >= 8:
        confidence = "Medium"
        status = "Assigned"
    else:
        confidence = "Low"
        status = "Assigned"
    if pos >= 0.45 and pos - max(neu, neg) >= 0.10:
        goal = "Promotional / Supportive"
    elif neg >= 0.45 and neg - max(pos, neu) >= 0.10:
        goal = "Critical / Complaint"
    elif neu >= 0.45 and neu - max(pos, neg) >= 0.10:
        goal = "Neutral Engagement"
    elif pos >= 0.25 and neg >= 0.25:
        goal = "Polarized / Contested"
    else:
        goal = "Mixed Goals"
    return goal, status, confidence, float(margin)


def hcc_level(df: pd.DataFrame) -> pd.DataFrame:
    hcc_nodes = read_csv(HCC_NODES)
    community_sizes = hcc_nodes.groupby("community").size().rename("community_size").reset_index()
    hcc_meta = hcc_nodes.drop_duplicates("community")[
        ["community", "primary_brand", "brand_label_auto", "brand_combo", "brand_confidence", "narrative_similarity_level"]
    ].merge(community_sizes, on="community", how="left")
    hcc_df = df[df["is_hcc_member_bool"]].copy()
    rows = []
    for _, meta in hcc_meta.iterrows():
        hcc_id = str(meta["community"])
        group = hcc_df[hcc_df["hcc_id"].astype(str).eq(hcc_id)]
        total = len(group)
        evaluable = int(group["final_sentiment_label"].isin(LABELS).sum())
        counts = {label: int(group["final_sentiment_label"].eq(label).sum()) for label in LABELS}
        ratios = {label: counts[label] / evaluable if evaluable else 0.0 for label in LABELS}
        prob_mass = {label: float(group[f"probability_{label.lower()}"].sum()) for label in LABELS}
        prob_total = sum(prob_mass.values())
        soft = {label: prob_mass[label] / prob_total if prob_total else 0.0 for label in LABELS}
        goal, status, confidence, stability = classify_goal(
            ratios["Positive"], ratios["Neutral"], ratios["Negative"], evaluable, evaluable / total if total else 0.0
        )
        dominant = max(LABELS, key=lambda label: counts[label]) if evaluable else "No evaluable sentiment"
        row = {
            "hcc_id": hcc_id,
            "community_size": int(meta["community_size"]),
            "n_accounts": int(meta["community_size"]),
            "n_comments": total,
            "evaluable_comments": evaluable,
            "coverage": evaluable / total if total else np.nan,
            "positive_count": counts["Positive"],
            "neutral_count": counts["Neutral"],
            "negative_count": counts["Negative"],
            "uncertain_count": int(group["final_sentiment_label"].eq("Uncertain").sum()),
            "no_text_count": int(group["final_sentiment_label"].eq("No Text").sum()),
            "positive_ratio": ratios["Positive"],
            "neutral_ratio": ratios["Neutral"],
            "negative_ratio": ratios["Negative"],
            "soft_positive_share": soft["Positive"],
            "soft_neutral_share": soft["Neutral"],
            "soft_negative_share": soft["Negative"],
            "effective_sample_size": float(group["max_probability"].sum()) if total else 0.0,
            "dominant_sentiment": dominant,
            "goal_orientation": goal,
            "goal_orientation_status": status,
            "goal_confidence": confidence,
            "goal_stability": stability,
            "goal_method": "hard_label_ratios_with_soft_probability_mass_and_bootstrap_stability_v2",
            "goal_validation_status": "FINAL_MODEL_VALIDATED_SENTIMENT_V2",
            "small_sample_warning": bool(evaluable < 10),
            **meta.drop(labels=["community", "community_size"]).to_dict(),
        }
        for label in LABELS:
            lo, hi = wilson_ci(counts[label], evaluable)
            row[f"{label.lower()}_ratio_ci_low"] = lo
            row[f"{label.lower()}_ratio_ci_high"] = hi
        rows.append(row)
    return pd.DataFrame(rows).sort_values("hcc_id", key=lambda s: s.astype(int)).reset_index(drop=True)


def brand_summary(df: pd.DataFrame, hcc_summary: pd.DataFrame) -> pd.DataFrame:
    categories = ["Azarine", "Daviena", "Maryame", "The Originote", "Mixed_2_Brands", "Mixed_3plus_Brands", "Not identified"]
    hcc_comments = df[df["is_hcc_member_bool"] & df["brand_label_auto"].isin(categories)].copy()
    rows = []
    for brand in categories:
        group = hcc_comments[hcc_comments["brand_label_auto"].eq(brand)]
        total = len(group)
        evaluable = int(group["final_sentiment_label"].isin(LABELS).sum())
        row = {
            "brand_label_auto": brand,
            "n_comments": total,
            "n_accounts": int(group["username_norm"].nunique()),
            "n_hccs": int(hcc_summary[hcc_summary["brand_label_auto"].eq(brand)]["hcc_id"].nunique()),
            "evaluable_comments": evaluable,
            "coverage": evaluable / total if total else np.nan,
            "support_status": "LOW_SUPPORT" if evaluable < 20 else "AVAILABLE",
        }
        for label in LABELS:
            count = int(group["final_sentiment_label"].eq(label).sum())
            lo, hi = wilson_ci(count, evaluable)
            row[f"{label.lower()}_count"] = count
            row[f"{label.lower()}_ratio"] = count / evaluable if evaluable else np.nan
            row[f"{label.lower()}_ci_low"] = lo
            row[f"{label.lower()}_ci_high"] = hi
        row["dominant_sentiment"] = max(LABELS, key=lambda label: row[f"{label.lower()}_count"]) if evaluable else "No evaluable sentiment"
        rows.append(row)
    return pd.DataFrame(rows)


def legacy_comparison(obs: pd.DataFrame, hcc_summary_v2: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    legacy = read_csv(LEGACY_COMMENT_SENTIMENT)
    legacy = legacy[~legacy["comment_id"].astype(str).str.match(r"(?i)^INJ")].copy()
    legacy["legacy_label"] = np.where(
        legacy["sentiment_status"].eq("No Text"),
        "No Text",
        np.where(legacy["sentiment_status"].eq("Uncertain"), "Uncertain", legacy["sentiment_label_final"]),
    )
    merged = obs[["comment_id", "final_sentiment_label", "is_hcc_member_bool", "brand_label_auto", "hcc_id"]].merge(
        legacy[["comment_id", "legacy_label"]], on="comment_id", how="left"
    )
    transition = pd.crosstab(merged["legacy_label"], merged["final_sentiment_label"]).reset_index()
    transition.to_csv(TABLE_DIR / "legacy_v1_to_final_v2_transition_matrix.csv", index=False)
    rows = []
    for scope_name, group in {
        "all_observational": merged,
        "hcc_comments": merged[merged["is_hcc_member_bool"]],
        "nonhcc_comments": merged[~merged["is_hcc_member_bool"]],
    }.items():
        for label in ["Positive", "Neutral", "Negative", "Uncertain", "No Text"]:
            rows.append(
                {
                    "comparison_scope": scope_name,
                    "label": label,
                    "legacy_count": int(group["legacy_label"].eq(label).sum()),
                    "final_v2_count": int(group["final_sentiment_label"].eq(label).sum()),
                    "count_delta_final_minus_legacy": int(group["final_sentiment_label"].eq(label).sum()) - int(group["legacy_label"].eq(label).sum()),
                }
            )
    changed = int((merged["legacy_label"] != merged["final_sentiment_label"]).sum())
    rows.append({"comparison_scope": "all_observational", "label": "changed_comment_id", "legacy_count": "", "final_v2_count": changed, "count_delta_final_minus_legacy": ""})
    old_goals_path = ROOT / "output/rm2_sentiment/tables/hcc_sentiment_goals_summary.csv"
    if old_goals_path.exists():
        old_goals = read_csv(old_goals_path)
        goal_changes = old_goals[["hcc_id", "goal_orientation"]].rename(columns={"goal_orientation": "legacy_goal_orientation"}).merge(
            hcc_summary_v2[["hcc_id", "goal_orientation"]].rename(columns={"goal_orientation": "final_v2_goal_orientation"}),
            on="hcc_id",
            how="outer",
        )
        goal_changes["goal_changed"] = goal_changes["legacy_goal_orientation"].ne(goal_changes["final_v2_goal_orientation"])
        goal_changes.to_csv(TABLE_DIR / "legacy_v1_vs_final_v2_hcc_goal_changes.csv", index=False)
        rows.append({"comparison_scope": "hcc_goals", "label": "changed_hcc_goal_orientation", "legacy_count": "", "final_v2_count": int(goal_changes["goal_changed"].sum()), "count_delta_final_minus_legacy": ""})
    return pd.DataFrame(rows), transition


def write_report(
    obs: pd.DataFrame,
    injected: pd.DataFrame,
    dist: pd.DataFrame,
    hcc_comment: pd.DataFrame,
    hcc_account: pd.DataFrame,
    hcc_summary: pd.DataFrame,
    brand: pd.DataFrame,
    comparison: pd.DataFrame,
    eval_metrics: pd.DataFrame,
    per_class: pd.DataFrame,
    acceptance: pd.DataFrame,
    manifest: dict,
) -> None:
    def md_table(frame: pd.DataFrame) -> str:
        if frame.empty:
            return "_Tidak ada data._"
        view = frame.copy()
        for col in view.columns:
            view[col] = view[col].map(lambda x: "" if pd.isna(x) else str(x))
        header = "| " + " | ".join(view.columns) + " |"
        sep = "| " + " | ".join(["---"] * len(view.columns)) + " |"
        body = ["| " + " | ".join(row) + " |" for row in view.to_numpy(dtype=str)]
        return "\n".join([header, sep] + body)

    metric = dict(zip(eval_metrics["metric"], eval_metrics["value"].astype(float)))
    label_counts = dist[dist["label"].isin(["Positive", "Neutral", "Negative", "Uncertain", "No Text"])][["label", "count", "percentage_of_total"]]
    goal_counts = hcc_summary["goal_orientation"].value_counts().to_dict()
    lines = [
        "# Final RM2 Sentiment V2 Analysis Report",
        "",
        "## Status Validasi Model",
        "",
        f"Model Sentimen V2 berstatus `{manifest['final_acceptance_status']}` berdasarkan evaluasi locked test final 300 komentar observasional. Evaluasi dilakukan satu kali setelah locked test dibekukan, dengan threshold abstention `0.42` dan model hash `{manifest['model_hash']}`.",
        "",
        "## Locked-Test Metrics",
        "",
        f"- Coverage evaluable: `{metric['coverage']:.4f}`",
        f"- Abstention rate: `{metric['abstention_rate']:.4f}`",
        f"- Macro-F1 covered: `{metric['macro_f1_covered']:.4f}`",
        f"- Balanced accuracy covered: `{metric['balanced_accuracy_covered']:.4f}`",
        f"- MCC covered: `{metric['mcc_covered']:.4f}`",
        "",
        md_table(per_class),
        "",
        "## Distribusi Sentimen Observasional",
        "",
        f"Denominator utama adalah `{len(obs)}` komentar observasional/non-INJ. Sebanyak `{len(injected)}` komentar INJ disimpan terpisah sebagai diagnostic dan tidak dicampurkan dalam denominator utama.",
        "",
        md_table(label_counts),
        "",
        "## HCC vs Non-HCC",
        "",
        "Perbandingan ini menunjukkan perbedaan distribusi sentimen teramati menurut posisi jaringan, bukan efek kausal, niat, atau pengaruh.",
        "",
        md_table(hcc_comment),
        "",
        "## Account-Level HCC vs Non-HCC",
        "",
        md_table(hcc_account),
        "",
        "## Goal Orientation HCC",
        "",
        "Goal orientation dibaca sebagai orientasi pesan deskriptif berbasis pola sentimen teramati.",
        "",
        md_table(pd.DataFrame([{"goal_orientation": k, "n_hcc": v} for k, v in goal_counts.items()])),
        "",
        "## Brand Context",
        "",
        "Konteks brand berasal dari label brand HCC berdasarkan metadata hashtag video. Kategori mixed bukan brand tunggal dan tidak membuktikan afiliasi.",
        "",
        md_table(brand),
        "",
        "## Legacy V1 vs Final V2",
        "",
        md_table(comparison.head(20)),
        "",
        "## Batas Interpretasi",
        "",
        "- Sentimen adalah indikator orientasi pesan, bukan bukti niat, pembayaran, afiliasi, kendali, atau pengaruh kausal.",
        "- HCC menunjukkan pola koordinasi struktural, bukan bukti bahwa akun adalah bot atau buzzer.",
        "- Confidence dan stability adalah indikator ketidakpastian model/agregasi, bukan akurasi aktual pada setiap komentar.",
    ]
    OUT_REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    pres = [
        "# Ringkasan Presentasi Sentimen V2",
        "",
        f"- Model accepted: `{manifest['final_acceptance_status']}`.",
        f"- Locked-test coverage: `{metric['coverage']:.3f}`; macro-F1: `{metric['macro_f1_covered']:.3f}`.",
        f"- Denominator utama: `{len(obs)}` komentar observasional; INJ diagnostic: `{len(injected)}`.",
        "- Interpretasi aman: distribusi sentimen menunjukkan orientasi pesan teramati, bukan niat atau afiliasi.",
        "",
        "## Distribusi Observasional",
        "",
        md_table(label_counts),
        "",
        "## Goal Orientation",
        "",
        md_table(pd.DataFrame([{"goal_orientation": k, "n_hcc": v} for k, v in goal_counts.items()])),
    ]
    OUT_PRES_MD.write_text("\n".join(pres) + "\n", encoding="utf-8")


def gate(metric: str, expected: object, observed: object, passed: bool, notes: str = "") -> dict[str, object]:
    return {"metric": metric, "expected": expected, "observed": observed, "passed": bool(passed), "notes": notes}


def main() -> None:
    mkdirs()
    head = git_head()
    lock = json.loads(EVALUATION_LOCK.read_text(encoding="utf-8"))
    if lock["status"] != "FINAL_LOCKED_TEST_EVALUATED_ONCE":
        raise AssertionError("Final locked-test evaluation lock is not complete.")
    if lock["final_acceptance_status"] not in FINAL_STATUS_ACCEPTED:
        raise AssertionError("Model is not accepted; full inference is not allowed.")

    model_hash = sha256_file(MODEL_ARTIFACT)
    config_hash = sha256_file(MODEL_CONFIG)
    config = json.loads(MODEL_CONFIG.read_text(encoding="utf-8"))
    if model_hash != EXPECTED_MODEL_HASH:
        raise AssertionError("Model artifact hash changed.")
    if abs(float(config["threshold"]) - EXPECTED_THRESHOLD) > 1e-12:
        raise AssertionError("Threshold changed.")
    if lock["model_sha256"] != model_hash or lock["config_sha256"] != config_hash:
        raise AssertionError("Evaluation lock hashes do not match current frozen model/config.")

    artifact = joblib.load(MODEL_ARTIFACT)
    data = read_csv(DATASET)
    legacy = read_csv(LEGACY_COMMENT_SENTIMENT)
    data["comment_id"] = data["comment_id"].astype(str)
    data["username_norm"] = data["username"].map(normalize_username)
    data["is_injected"] = data["comment_id"].str.match(r"(?i)^INJ")
    obs_input = data[~data["is_injected"]].copy()
    inj_input = data[data["is_injected"]].copy()

    meta_cols = [
        "comment_id",
        "is_hcc_member",
        "is_hcc",
        "hcc_id",
        "community",
        "primary_brand",
        "brand_label_auto",
        "brand_combo",
        "brand_confidence",
        "product_brand_context",
    ]
    meta = legacy[[c for c in meta_cols if c in legacy.columns]].drop_duplicates("comment_id")

    obs = add_prediction_columns(obs_input, artifact, config, model_hash, config_hash, lock["final_acceptance_status"]).merge(meta, on="comment_id", how="left")
    injected = add_prediction_columns(inj_input, artifact, config, model_hash, config_hash, lock["final_acceptance_status"]).merge(meta, on="comment_id", how="left")
    for frame in [obs, injected]:
        frame["is_hcc_member_bool"] = frame["is_hcc_member"].astype(str).str.lower().eq("true")
        frame["hcc_id"] = frame["hcc_id"].replace("", "Non-HCC")
        frame["brand_label_auto"] = frame["brand_label_auto"].replace("", "Non-HCC")

    OUT_OBS.parent.mkdir(parents=True, exist_ok=True)
    obs.to_csv(OUT_OBS, index=False)
    injected.to_csv(OUT_INJ, index=False)

    dist = sentiment_distribution(obs)
    dist.to_csv(TABLE_DIR / "sentiment_distribution_observational.csv", index=False)
    input_integrity = pd.DataFrame(
        [
            {"metric": "dataset_rows", "value": len(data)},
            {"metric": "unique_comment_id", "value": data["comment_id"].nunique()},
            {"metric": "injected_rows", "value": len(injected)},
            {"metric": "observational_rows", "value": len(obs)},
            {"metric": "legacy_status", "value": "LEGACY_V1_PROVISIONAL_FULL_INFERENCE"},
        ]
    )
    input_integrity.to_csv(TABLE_DIR / "sentiment_input_integrity_summary.csv", index=False)

    hcc_comment = hcc_vs_nonhcc_comment(obs)
    hcc_comment.to_csv(TABLE_DIR / "hcc_vs_nonhcc_comment_sentiment_v2.csv", index=False)
    account, account_group = account_summary(obs)
    account.to_csv(TABLE_DIR / "account_sentiment_summary_v2.csv", index=False)
    account_group.to_csv(TABLE_DIR / "hcc_vs_nonhcc_account_sentiment_v2.csv", index=False)
    hcc_summary = hcc_level(obs)
    hcc_summary.to_csv(TABLE_DIR / "hcc_sentiment_goals_summary_v2.csv", index=False)
    brand = brand_summary(obs, hcc_summary)
    brand.to_csv(TABLE_DIR / "brand_sentiment_summary_v2.csv", index=False)
    comparison, transition = legacy_comparison(obs, hcc_summary)
    comparison.to_csv(TABLE_DIR / "legacy_v1_vs_final_v2_sentiment_comparison.csv", index=False)

    # Gephi V2 attributes; topology remains unchanged.
    hcc_nodes = read_csv(HCC_NODES)
    node_attrs = hcc_summary[[
        "hcc_id",
        "dominant_sentiment",
        "positive_ratio",
        "neutral_ratio",
        "negative_ratio",
        "coverage",
        "goal_orientation",
        "goal_orientation_status",
        "goal_confidence",
        "goal_stability",
        "goal_validation_status",
        "goal_method",
        "effective_sample_size",
    ]].rename(columns={"hcc_id": "community", "coverage": "evaluable_coverage"})
    gephi_hcc_nodes = hcc_nodes.merge(node_attrs, on="community", how="left")
    gephi_hcc_nodes.to_csv(GEPHI_DIR / "gephi_hcc_nodes_sentiment_v2.csv", index=False)
    read_csv(HCC_EDGES).to_csv(GEPHI_DIR / "gephi_hcc_edges_sentiment_v2.csv", index=False)

    if LCN_NODES_ACTOR_TYPE.exists():
        lcn_nodes = read_csv(LCN_NODES_ACTOR_TYPE)
        lcn_nodes["username_norm"] = lcn_nodes["Id"].map(normalize_username)
        lcn_sent = lcn_nodes.merge(
            account[[
                "username_norm",
                "dominant_sentiment",
                "positive_ratio",
                "neutral_ratio",
                "negative_ratio",
                "coverage",
                "evaluable_comments",
            ]].rename(columns={"coverage": "sentiment_v2_coverage"}),
            on="username_norm",
            how="left",
        )
        lcn_sent["sentiment_v2_validation_status"] = lock["final_acceptance_status"]
        lcn_sent.to_csv(GEPHI_DIR / "gephi_lcn_nodes_actor_type_sentiment_v2.csv", index=False)

    eval_metrics = read_csv(EVALUATION_METRICS)
    per_class = read_csv(MODEL_DIR / "final_locked_test_per_class_metrics.csv")
    acceptance = read_csv(MODEL_DIR / "final_locked_test_acceptance_decision.csv")

    summary_rows = [
        {"metric": "final_acceptance_status", "value": lock["final_acceptance_status"]},
        {"metric": "observational_count", "value": len(obs)},
        {"metric": "injected_diagnostic_count", "value": len(injected)},
        {"metric": "hcc_count", "value": hcc_summary["hcc_id"].nunique()},
        {"metric": "hcc_member_count", "value": read_csv(HCC_NODES)["id"].nunique()},
    ]
    for _, row in dist.iterrows():
        summary_rows.append({"metric": f"observational_{row['label']}", "value": row["count"]})
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT_SUMMARY, index=False)

    integrity_rows = [
        gate("final_locked_test_evaluated_once", "FINAL_LOCKED_TEST_EVALUATED_ONCE", lock["status"], lock["status"] == "FINAL_LOCKED_TEST_EVALUATED_ONCE"),
        gate("model_hash_unchanged", EXPECTED_MODEL_HASH, model_hash, model_hash == EXPECTED_MODEL_HASH),
        gate("threshold", EXPECTED_THRESHOLD, config["threshold"], abs(float(config["threshold"]) - EXPECTED_THRESHOLD) < 1e-12),
        gate("dataset_rows", 33847, len(data), len(data) == 33847),
        gate("unique_comment_id", 33847, data["comment_id"].nunique(), data["comment_id"].nunique() == 33847),
        gate("observational_count", 33063, len(obs), len(obs) == 33063),
        gate("injected_count", 784, len(injected), len(injected) == 784),
        gate("observational_duplicate_comment_id", 0, int(obs["comment_id"].duplicated().sum()), int(obs["comment_id"].duplicated().sum()) == 0),
        gate("missing_observational_comment_id", 0, int(obs["comment_id"].isna().sum()), int(obs["comment_id"].isna().sum()) == 0),
        gate("probability_valid", True, bool(obs[[f"probability_{label.lower()}" for label in LABELS]].apply(pd.to_numeric).ge(0).all().all()), bool(obs[[f"probability_{label.lower()}" for label in LABELS]].apply(pd.to_numeric).ge(0).all().all())),
        gate("hcc_count", 42, hcc_summary["hcc_id"].nunique(), hcc_summary["hcc_id"].nunique() == 42),
        gate("hcc_members", 218, read_csv(HCC_NODES)["id"].nunique(), read_csv(HCC_NODES)["id"].nunique() == 218),
        gate("lcn_nodes", 724, len(read_csv(ROOT / "output/gephi/gephi_lcn_nodes.csv")), len(read_csv(ROOT / "output/gephi/gephi_lcn_nodes.csv")) == 724),
        gate("lcn_edges", 1357, len(read_csv(ROOT / "output/gephi/gephi_lcn_edges.csv")), len(read_csv(ROOT / "output/gephi/gephi_lcn_edges.csv")) == 1357),
        gate("community_mass_pairs", 434823, len(read_csv(COMMUNITY_MASS_PAIRS)), len(read_csv(COMMUNITY_MASS_PAIRS)) == 434823),
    ]
    if ACTOR_UNIVERSE_SUMMARY.exists():
        actor = read_csv(ACTOR_UNIVERSE_SUMMARY)
        get_count = lambda label: int(actor.loc[actor["actor_type_primary"].eq(label), "n_accounts"].iloc[0])
        integrity_rows.extend(
            [
                gate("individual_actor", 43, get_count("Individual Actor"), get_count("Individual Actor") == 43),
                gate("community_actor", 218, get_count("Community Actor"), get_count("Community Actor") == 218),
                gate("mass_actor", 26166, get_count("Mass Actor"), get_count("Mass Actor") == 26166),
            ]
        )
    if COMMENT_SIM_SUMMARY.exists():
        integrity_rows.append(gate("comment_similarity_outputs_present", True, True, True))
    integrity = pd.DataFrame(integrity_rows)
    integrity.to_csv(OUT_INTEGRITY, index=False)
    if not integrity["passed"].all():
        raise AssertionError(integrity.loc[lambda df: ~df["passed"]].to_string(index=False))

    manifest = {
        "status": "FINAL_SENTIMENT_V2_FULL_INFERENCE_COMPLETE",
        "final_acceptance_status": lock["final_acceptance_status"],
        "git_head_at_inference": head,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_hash": model_hash,
        "config_hash": config_hash,
        "locked_test_sha256": lock["locked_test_sha256"],
        "threshold": float(config["threshold"]),
        "dataset_rows": int(len(data)),
        "observational_rows": int(len(obs)),
        "injected_diagnostic_rows": int(len(injected)),
        "observational_output_sha256": sha256_file(OUT_OBS),
        "injected_output_sha256": sha256_file(OUT_INJ),
        "legacy_status": "LEGACY_V1_PROVISIONAL_FULL_INFERENCE",
        "package_versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "outputs": [
            str(OUT_OBS.relative_to(ROOT)),
            str(OUT_INJ.relative_to(ROOT)),
            str(TABLE_DIR.relative_to(ROOT)),
            str(GEPHI_DIR.relative_to(ROOT)),
            str(PRESENTATION_DIR.relative_to(ROOT)),
        ],
    }
    OUT_INFERENCE_MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    OUT_MANIFEST.write_text(json.dumps(manifest | {"integrity_all_passed": True}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    write_report(obs, injected, dist, hcc_comment, account_group, hcc_summary, brand, comparison, eval_metrics, per_class, acceptance, manifest)

    pres_tables = pd.concat(
        [
            dist.assign(table="distribution"),
            hcc_comment.assign(table="hcc_vs_nonhcc_comment"),
            brand.assign(table="brand"),
            pd.DataFrame([{"table": "goal_orientation", "goal_orientation": k, "n_hcc": v} for k, v in hcc_summary["goal_orientation"].value_counts().items()]),
        ],
        ignore_index=True,
        sort=False,
    )
    pres_tables.to_csv(OUT_PRES_TABLES, index=False)

    # Update readiness/status report after accepted evaluation.
    readiness = pd.DataFrame(
        [
            {"metric": "final_locked_test_evaluation_status", "value": "FINAL_LOCKED_TEST_EVALUATED_ONCE", "passed": True, "notes": "Evaluation lock exists and final model is accepted."},
            {"metric": "final_acceptance_status", "value": lock["final_acceptance_status"], "passed": True, "notes": "Full inference was allowed by acceptance gates."},
            {"metric": "selected_threshold", "value": float(config["threshold"]), "passed": True, "notes": "Frozen threshold used for locked test and full inference."},
            {"metric": "full_inference_generated", "value": len(obs), "passed": True, "notes": "Observational output generated separately from INJ diagnostic."},
            {"metric": "injected_diagnostic_generated", "value": len(injected), "passed": True, "notes": "INJ rows are not included in primary denominator."},
        ]
    )
    readiness.to_csv(MODEL_DIR / "final_locked_test_evaluation_readiness.csv", index=False)

    print("FINAL_SENTIMENT_V2_FULL_INFERENCE_COMPLETE")
    print(f"observational={len(obs)} injected={len(injected)} status={lock['final_acceptance_status']}")
    print(f"distribution={dist[['label','count']].to_dict('records')}")
    print(f"integrity_gates={len(integrity)} all_passed={integrity['passed'].all()}")


if __name__ == "__main__":
    main()
