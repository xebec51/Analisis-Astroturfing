from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
SENS_DIR = ROOT / "output/rm2_sentiment/sensitivity/indobert_v4"
TABLE_DIR = SENS_DIR / "tables"
V2_OBS = ROOT / "output/rm2_sentiment/final/comment_sentiment_v2_observational.csv"
V2_HCC_GOALS = ROOT / "output/rm2_sentiment/final/tables/hcc_sentiment_goals_summary_v2.csv"
V2_DISTRIBUTION = ROOT / "output/rm2_sentiment/final/tables/sentiment_distribution_observational.csv"
V2_HCC_NONHCC = ROOT / "output/rm2_sentiment/final/tables/hcc_vs_nonhcc_comment_sentiment_v2.csv"
ACTOR_TYPE = ROOT / "output/rm2_actor_type/gephi/gephi_lcn_nodes_actor_type.csv"
V4_MODEL_DIR = ROOT / "artifacts/rm2_sentiment/indobert_v4_final/base_reference"
V4_DECISION = ROOT / "output/rm2_sentiment/experiments/indobert_v4_final/locked_test_evaluation/FINAL_ACCEPTANCE_DECISION.json"
OUT_PRED = SENS_DIR / "indobert_v4_comment_sentiment_sensitivity.csv"
LABELS = ["Negative", "Neutral", "Positive"]
LABEL_TO_ID = {label: idx for idx, label in enumerate(LABELS)}
ID_TO_LABEL = {idx: label for label, idx in LABEL_TO_ID.items()}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2), encoding="utf-8")


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [to_jsonable(v) for v in value]
    if isinstance(value, Path):
        return rel(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False, low_memory=False)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_dataframe(frame: pd.DataFrame, columns: list[str]) -> str:
    data = frame[columns].sort_values(columns).to_csv(index=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def no_text(value: object) -> bool:
    text = "" if value is None or pd.isna(value) else str(value).strip()
    return text.lower() in {"", "nan", "none", "null", "<na>", "[deleted]", "deleted"}


def model_text(row: pd.Series) -> str:
    comment = str(row.get("comment_text_original") or row.get("text") or "").strip()
    context = str(row.get("product_category") or row.get("product_brand_context") or "").strip()
    return f"{context} [SEP] {comment}" if context else comment


def parse_sha256s(path: Path) -> dict[str, str]:
    rows: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, name = line.split(maxsplit=1)
            rows[name.strip()] = digest.strip()
    return rows


def verify_model() -> str:
    checksums = parse_sha256s(V4_MODEL_DIR / "SHA256SUMS.txt")
    for name, expected in checksums.items():
        actual = sha256_file(V4_MODEL_DIR / name)
        if actual != expected:
            raise RuntimeError(f"Checksum mismatch for {name}")
    label_map = read_json(V4_MODEL_DIR / "label_map.json")
    if [label_map["id_to_label"][str(i)] for i in range(3)] != LABELS:
        raise RuntimeError("Unexpected V4 label order.")
    return checksums["model.safetensors"]


def run_inference(batch_size: int, limit: int | None = None) -> pd.DataFrame:
    obs = read_csv(V2_OBS)
    if limit:
        obs = obs.head(limit).copy()
    model_hash = verify_model()
    config = read_json(V4_MODEL_DIR / "selected_trial_config.json")
    max_length = int(config.get("max_length", 256))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(V4_MODEL_DIR, trust_remote_code=False)
    model = AutoModelForSequenceClassification.from_pretrained(V4_MODEL_DIR, trust_remote_code=False)
    model.to(device)
    model.eval()

    rows: list[dict[str, Any]] = []
    valid = obs.loc[~obs["comment_text_original"].map(no_text)].copy()
    invalid = obs.loc[obs["comment_text_original"].map(no_text)].copy()
    texts = valid.apply(model_text, axis=1).tolist()
    probs_all: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch_texts = texts[start : start + batch_size]
            encoded = tokenizer(batch_texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            encoded = {key: value.to(device) for key, value in encoded.items()}
            logits = model(**encoded).logits.detach().float().cpu()
            probs_all.append(torch.softmax(logits, dim=1).numpy())
    probs = np.vstack(probs_all) if probs_all else np.empty((0, 3))
    pred = probs.argmax(axis=1) if len(probs) else np.array([], dtype=int)
    for i, (_, row) in enumerate(valid.iterrows()):
        rows.append(
            {
                **row.to_dict(),
                "v2_final_sentiment_label": row["final_sentiment_label"],
                "sentiment_label": ID_TO_LABEL[int(pred[i])],
                "probability_negative_v4": float(probs[i, LABEL_TO_ID["Negative"]]),
                "probability_neutral_v4": float(probs[i, LABEL_TO_ID["Neutral"]]),
                "probability_positive_v4": float(probs[i, LABEL_TO_ID["Positive"]]),
                "prediction_confidence_v4": float(probs[i].max()),
                "model_version": "IndoBERT_V4_NON_CANONICAL_SENSITIVITY",
                "model_id": config.get("model_id", "indobenchmark/indobert-base-p2"),
                "model_sha256": model_hash,
                "sensitivity_status": "EXPLORATORY_SENSITIVITY_ANALYSIS_NOT_CANONICAL",
            }
        )
    for _, row in invalid.iterrows():
        rows.append(
            {
                **row.to_dict(),
                "v2_final_sentiment_label": row["final_sentiment_label"],
                "sentiment_label": "No Text",
                "probability_negative_v4": 0.0,
                "probability_neutral_v4": 0.0,
                "probability_positive_v4": 0.0,
                "prediction_confidence_v4": 0.0,
                "model_version": "IndoBERT_V4_NON_CANONICAL_SENSITIVITY",
                "model_id": config.get("model_id", "indobenchmark/indobert-base-p2"),
                "model_sha256": model_hash,
                "sensitivity_status": "EXPLORATORY_SENSITIVITY_ANALYSIS_NOT_CANONICAL",
            }
        )
    out = pd.DataFrame(rows).sort_values("comment_id").reset_index(drop=True)
    out.to_csv(OUT_PRED, index=False, encoding="utf-8-sig")
    return out


def distribution(df: pd.DataFrame, label_col: str, group: str) -> pd.DataFrame:
    labels = ["Positive", "Neutral", "Negative", "No Text"]
    total = len(df)
    return pd.DataFrame(
        [
            {
                "group": group,
                "label": label,
                "count": int(df[label_col].eq(label).sum()),
                "percentage_of_total": int(df[label_col].eq(label).sum()) / total * 100 if total else np.nan,
                "denominator": total,
            }
            for label in labels
        ]
    )


def hcc_vs_nonhcc(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    rows = []
    for group_name, group in df.groupby(np.where(df["is_hcc_member_bool"].astype(str).str.lower().eq("true"), "HCC", "Non-HCC")):
        total = len(group)
        evaluable = int(group[label_col].isin(LABELS).sum())
        row = {"group": group_name, "total_comments": total, "evaluable_comments": evaluable, "coverage": evaluable / total if total else np.nan}
        for label in LABELS:
            count = int(group[label_col].eq(label).sum())
            row[f"{label.lower()}_count"] = count
            row[f"{label.lower()}_ratio_evaluable"] = count / evaluable if evaluable else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def goal_from_ratios(pos: float, neu: float, neg: float, evaluable: int, total: int) -> tuple[str, str]:
    coverage = evaluable / total if total else 0.0
    if evaluable == 0:
        return "Not Observed", "NO_EVALUABLE_COMMENTS"
    margin = max(pos, neu, neg) - sorted([pos, neu, neg])[-2]
    if evaluable < 5 or coverage < 0.5:
        status = "LOW_EVIDENCE_GOAL_ORIENTATION"
    else:
        status = "Assigned"
    if pos >= 0.45 and pos - max(neu, neg) >= 0.10:
        return "Promotional / Supportive", status
    if neg >= 0.45 and neg - max(pos, neu) >= 0.10:
        return "Critical / Complaint", status
    if neu >= 0.45 and neu - max(pos, neg) >= 0.10:
        return "Neutral Engagement", status
    if pos >= 0.25 and neg >= 0.25:
        return "Polarized / Contested", status
    return "Mixed Goals", status


def hcc_goals(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    hcc = df.loc[df["is_hcc_member_bool"].astype(str).str.lower().eq("true")].copy()
    rows = []
    for hcc_id, group in hcc.groupby("hcc_id"):
        total = len(group)
        evaluable = int(group[label_col].isin(LABELS).sum())
        counts = {label: int(group[label_col].eq(label).sum()) for label in LABELS}
        ratios = {label: counts[label] / evaluable if evaluable else 0.0 for label in LABELS}
        goal, status = goal_from_ratios(ratios["Positive"], ratios["Neutral"], ratios["Negative"], evaluable, total)
        rows.append(
            {
                "hcc_id": hcc_id,
                "n_comments": total,
                "evaluable_comments": evaluable,
                "coverage": evaluable / total if total else np.nan,
                "positive_count": counts["Positive"],
                "neutral_count": counts["Neutral"],
                "negative_count": counts["Negative"],
                "positive_ratio": ratios["Positive"],
                "neutral_ratio": ratios["Neutral"],
                "negative_ratio": ratios["Negative"],
                "dominant_sentiment": max(LABELS, key=lambda label: counts[label]) if evaluable else "No evaluable sentiment",
                "goal_orientation": goal,
                "goal_orientation_status": status,
                "brand_label_auto": group["brand_label_auto"].mode().iloc[0] if "brand_label_auto" in group and not group["brand_label_auto"].empty else "",
                "sensitivity_status": "EXPLORATORY_SENSITIVITY_ANALYSIS_NOT_CANONICAL",
            }
        )
    return pd.DataFrame(rows).sort_values("hcc_id")


def actor_type_summary(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    actor = read_csv(ACTOR_TYPE)[["Id", "actor_type_primary"]].rename(columns={"Id": "username_norm"})
    merged = df.merge(actor, on="username_norm", how="left")
    merged["actor_type_primary"] = merged["actor_type_primary"].replace("", "Not in LCN").fillna("Not in LCN")
    rows = []
    for actor_type, group in merged.groupby("actor_type_primary"):
        total = len(group)
        evaluable = int(group[label_col].isin(LABELS).sum())
        row = {"actor_type": actor_type, "n_comments": total, "evaluable_comments": evaluable, "coverage": evaluable / total if total else np.nan}
        for label in LABELS:
            count = int(group[label_col].eq(label).sum())
            row[f"{label.lower()}_count"] = count
            row[f"{label.lower()}_ratio"] = count / evaluable if evaluable else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def brand_summary(df: pd.DataFrame, label_col: str) -> pd.DataFrame:
    hcc = df.loc[df["is_hcc_member_bool"].astype(str).str.lower().eq("true")].copy()
    rows = []
    for brand, group in hcc.groupby("brand_label_auto"):
        total = len(group)
        evaluable = int(group[label_col].isin(LABELS).sum())
        row = {"brand_label_auto": brand, "n_comments": total, "evaluable_comments": evaluable, "coverage": evaluable / total if total else np.nan}
        for label in LABELS:
            count = int(group[label_col].eq(label).sum())
            row[f"{label.lower()}_count"] = count
            row[f"{label.lower()}_ratio"] = count / evaluable if evaluable else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def md_table(frame: pd.DataFrame, max_rows: int = 20) -> str:
    if frame.empty:
        return "_No rows._"
    show = frame.head(max_rows)
    cols = list(show.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in show.iterrows():
        lines.append("| " + " | ".join(str(row[col]) for col in cols) + " |")
    return "\n".join(lines)


def build_tables(df: pd.DataFrame) -> dict[str, Any]:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    dist_v4 = distribution(df, "sentiment_label", "IndoBERT V4 sensitivity")
    dist_v2 = distribution(df, "v2_final_sentiment_label", "Canonical V2")
    dist_cmp = pd.concat([dist_v2, dist_v4], ignore_index=True)
    dist_cmp.to_csv(TABLE_DIR / "v2_vs_v4_sentiment_distribution.csv", index=False, encoding="utf-8-sig")
    hcc_v4 = hcc_vs_nonhcc(df, "sentiment_label")
    hcc_v4.to_csv(TABLE_DIR / "hcc_vs_nonhcc_comment_sentiment_v4_sensitivity.csv", index=False, encoding="utf-8-sig")
    actor = actor_type_summary(df, "sentiment_label")
    actor.to_csv(TABLE_DIR / "actor_type_sentiment_v4_sensitivity.csv", index=False, encoding="utf-8-sig")
    goals_v4 = hcc_goals(df, "sentiment_label")
    goals_v4.to_csv(TABLE_DIR / "hcc_goals_v4_sensitivity.csv", index=False, encoding="utf-8-sig")
    brand = brand_summary(df, "sentiment_label")
    brand.to_csv(TABLE_DIR / "brand_sentiment_v4_sensitivity.csv", index=False, encoding="utf-8-sig")
    transition = pd.crosstab(df["v2_final_sentiment_label"], df["sentiment_label"]).reset_index()
    transition.to_csv(TABLE_DIR / "v2_vs_v4_label_transition.csv", index=False, encoding="utf-8-sig")
    v2_goals = read_csv(V2_HCC_GOALS)[["hcc_id", "goal_orientation"]].rename(columns={"goal_orientation": "v2_goal_orientation"})
    goal_cmp = v2_goals.merge(goals_v4[["hcc_id", "goal_orientation"]].rename(columns={"goal_orientation": "v4_goal_orientation"}), on="hcc_id", how="outer")
    goal_cmp["goal_changed"] = goal_cmp["v2_goal_orientation"].ne(goal_cmp["v4_goal_orientation"])
    goal_cmp.to_csv(TABLE_DIR / "v2_vs_v4_hcc_goal_changes.csv", index=False, encoding="utf-8-sig")
    conclusion = "robust" if int(goal_cmp["goal_changed"].sum()) == 0 else "model_dependent"
    report = f"""# RM2 Sentiment Sensitivity Analysis

Status: `EXPLORATORY_SENSITIVITY_ANALYSIS_NOT_CANONICAL`

Generated at UTC: {utc_now()}

This analysis compares canonical V2 RM2 sentiment outputs with non-canonical frozen IndoBERT V4 predictions. It does not modify `output/rm2_sentiment/final/` and does not change the canonical status of V2 or the previous V4 decision.

## Overall Distribution

{md_table(dist_cmp)}

## HCC Versus Non-HCC, V4 Sensitivity

{md_table(hcc_v4)}

## HCC Goal Changes

- Changed HCC goal orientations: `{int(goal_cmp["goal_changed"].sum())}` of `{len(goal_cmp)}`
- Sensitivity conclusion: `{conclusion}`

{md_table(goal_cmp)}
"""
    (SENS_DIR / "RM2_SENTIMENT_SENSITIVITY_ANALYSIS.md").write_text(report, encoding="utf-8")
    (ROOT / "docs/RM2_SENTIMENT_SENSITIVITY_ANALYSIS.md").write_text(report, encoding="utf-8")
    return {
        "distribution": rel(TABLE_DIR / "v2_vs_v4_sentiment_distribution.csv"),
        "hcc_vs_nonhcc": rel(TABLE_DIR / "hcc_vs_nonhcc_comment_sentiment_v4_sensitivity.csv"),
        "actor_type": rel(TABLE_DIR / "actor_type_sentiment_v4_sensitivity.csv"),
        "hcc_goals": rel(TABLE_DIR / "hcc_goals_v4_sensitivity.csv"),
        "brand": rel(TABLE_DIR / "brand_sentiment_v4_sensitivity.csv"),
        "transition": rel(TABLE_DIR / "v2_vs_v4_label_transition.csv"),
        "goal_changes": int(goal_cmp["goal_changed"].sum()),
        "conclusion": conclusion,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build non-canonical RM2 V4 sensitivity outputs.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--skip-inference", action="store_true")
    args = parser.parse_args()
    SENS_DIR.mkdir(parents=True, exist_ok=True)
    decision = read_json(V4_DECISION)
    if decision.get("status") != "INDOBERT_V4_NOT_ACCEPTED_KEEP_V2":
        raise RuntimeError("V4 strict decision must remain NOT_ACCEPTED_KEEP_V2.")
    if args.skip_inference and OUT_PRED.exists():
        predictions = read_csv(OUT_PRED)
    else:
        predictions = run_inference(args.batch_size, args.limit)
    tables = build_tables(predictions)
    manifest = {
        "status": "EXPLORATORY_SENSITIVITY_ANALYSIS_NOT_CANONICAL",
        "created_at_utc": utc_now(),
        "git_sha": git_head(),
        "v4_decision_preserved": decision.get("status"),
        "canonical_model_changed": False,
        "final_output_dir_modified": False,
        "input_rows": int(len(predictions)),
        "input_unique_comment_id": int(predictions["comment_id"].nunique()),
        "v4_label_distribution": predictions["sentiment_label"].value_counts().to_dict(),
        "v2_label_distribution": predictions["v2_final_sentiment_label"].value_counts().to_dict(),
        "dataset_hash": sha256_dataframe(predictions, ["comment_id", "sentiment_label"]),
        "outputs": {"predictions": rel(OUT_PRED), **tables},
    }
    write_json(SENS_DIR / "INDOBERT_V4_SENSITIVITY_MANIFEST.json", manifest)
    print(json.dumps({"status": manifest["status"], "rows": manifest["input_rows"], "goal_changes": tables["goal_changes"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
