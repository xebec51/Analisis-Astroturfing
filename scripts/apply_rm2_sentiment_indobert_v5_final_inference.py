from __future__ import annotations

import hashlib
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.apply_rm2_sentiment_v2_final_inference import (
    GEPHI_DIR,
    HCC_EDGES,
    HCC_NODES,
    LCN_NODES_ACTOR_TYPE,
    PRESENTATION_DIR,
    TABLE_DIR,
    account_summary,
    brand_summary,
    hcc_level,
    hcc_vs_nonhcc_comment,
    normalize_username,
    read_csv,
    sentiment_distribution,
    wilson_ci,
)
from scripts.prepare_rm2_sentiment_v5_annotation_package import brand_from_category
from scripts.train_rm2_sentiment_indobert_v5_development import LABELS, LABEL_TO_ID, model_input, sha256_dataframe


DATASET = ROOT / "dataset.csv"
V2_OBS = ROOT / "output/rm2_sentiment/final/comment_sentiment_v2_observational.csv"
V2_HCC_GOALS = ROOT / "output/rm2_sentiment/final/tables/hcc_sentiment_goals_summary_v2.csv"
V2_DISTRIBUTION = ROOT / "output/rm2_sentiment/final/tables/sentiment_distribution_observational.csv"
V2_ACTOR = ROOT / "output/rm2_actor_type/tables/account_actor_type.csv"
V2_ACTOR_GOALS = ROOT / "output/rm2_actor_type/tables/actor_type_goals_pooled.csv"

CANDIDATE_DIR = ROOT / "artifacts/rm2_sentiment/indobert_v5_candidate"
FINAL_MODEL_DIR = ROOT / "artifacts/rm2_sentiment/indobert_v5_final"
LOCKED_EVAL_DIR = ROOT / "output/rm2_sentiment/experiments/indobert_v5_final/locked_test_evaluation"
ACCEPTANCE_DECISION = LOCKED_EVAL_DIR / "FINAL_V5_ACCEPTANCE_DECISION.json"
LOCKED_MANIFEST = LOCKED_EVAL_DIR / "LOCKED_TEST_V5_EVALUATION_MANIFEST.json"
FINAL_DIR = ROOT / "output/rm2_sentiment/final"
VIS_DIR = FINAL_DIR / "visualisasi"
OUT_OBS = FINAL_DIR / "indobert_v5_comment_sentiment.csv"
OUT_MANIFEST = FINAL_DIR / "INDOBERT_V5_FULL_INFERENCE_MANIFEST.json"
CANONICAL_MODEL = FINAL_DIR / "CANONICAL_MODEL.json"
CHECKPOINT_DIR = FINAL_DIR / "indobert_v5_inference_checkpoints"
GUIDE = FINAL_DIR / "RM2_GOALS_INTERPRETATION_GUIDE_V5.md"
REPORT = FINAL_DIR / "INDOBERT_V5_FINAL_SENTIMENT_REPORT.md"
SUMMARY = FINAL_DIR / "INDOBERT_V5_FINAL_SENTIMENT_SUMMARY.csv"

LABEL_ORDER = ["Negative", "Neutral", "Positive"]


class TextDataset(Dataset):
    def __init__(self, texts: list[str]) -> None:
        self.texts = texts

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> str:
        return self.texts[idx]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def package_versions() -> dict[str, Any]:
    import transformers

    return {
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }


def verify_sha256s(root: Path) -> dict[str, str]:
    checksums = {}
    for line in (root / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        path = root / name
        if not path.exists():
            raise AssertionError(f"Missing checksum target: {name}")
        observed = sha256_file(path)
        if observed != digest:
            raise AssertionError(f"Checksum mismatch for {name}")
        checksums[name] = digest
    return checksums


def safe_copy_candidate_to_final() -> None:
    if not CANDIDATE_DIR.exists():
        raise FileNotFoundError(CANDIDATE_DIR)
    if FINAL_MODEL_DIR.exists():
        expected = (ROOT / "artifacts/rm2_sentiment/indobert_v5_final").resolve()
        resolved = FINAL_MODEL_DIR.resolve()
        if resolved != expected:
            raise RuntimeError(f"Refusing to remove unexpected final model directory: {resolved}")
        shutil.rmtree(resolved)
    shutil.copytree(CANDIDATE_DIR, FINAL_MODEL_DIR)
    verify_sha256s(FINAL_MODEL_DIR)


def parent_lookup(frame: pd.DataFrame) -> dict[str, str]:
    return dict(zip(frame["comment_id"].astype(str), frame["text"].astype(str)))


def build_model_rows(frame: pd.DataFrame) -> pd.DataFrame:
    lookup = parent_lookup(frame)
    out = frame.copy()
    out["comment_text"] = out["text"].astype(str)
    out["brand_or_video_context"] = out["product_category"].map(brand_from_category)
    out["parent_comment_text"] = out["parent_comment_id"].astype(str).map(lookup).fillna("")
    return out


def component_dirs(model_dir: Path) -> list[Path]:
    manifest = read_json(model_dir / "DEVELOPMENT_MODEL_FREEZE_MANIFEST.json")
    dirs = [ROOT / item["path"] for item in manifest["component_models"]]
    if model_dir == FINAL_MODEL_DIR:
        dirs = [model_dir / Path(item["path"]).name for item in manifest["component_models"]]
    if not dirs:
        raise AssertionError("No V5 component models found")
    return dirs


def predict_component(model_dir: Path, texts: list[str], max_length: int, device: torch.device, precision: str, batch_size: int = 64) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True)
    model.to(device)
    model.eval()
    probs = []
    autocast_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    loader = DataLoader(TextDataset(texts), batch_size=batch_size, shuffle=False, num_workers=0)
    with torch.no_grad():
        for batch_texts in loader:
            encoded = tokenizer(list(batch_texts), truncation=True, max_length=max_length, padding=True, return_tensors="pt")
            encoded = {key: value.to(device) for key, value in encoded.items()}
            context = torch.autocast("cuda", dtype=autocast_dtype) if device.type == "cuda" else torch.no_grad()
            with context:
                logits = model(**encoded).logits
            probs.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
    del model, tokenizer
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return np.concatenate(probs, axis=0)


def predict_v5(frame: pd.DataFrame, model_dir: Path, device: torch.device, precision: str, resume: bool = True) -> np.ndarray:
    selected = read_json(model_dir / "development_selected_candidate.json")
    max_length = int(read_json(model_dir / "preprocessing_config.json")["max_length"])
    texts = [model_input(row, selected["trial"]["input_mode"]) for _, row in build_model_rows(frame).iterrows()]
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    all_probs = []
    for idx, component in enumerate(component_dirs(model_dir), start=1):
        ckpt = CHECKPOINT_DIR / f"{component.name}_probs.npy"
        if resume and ckpt.exists():
            probs = np.load(ckpt)
            if probs.shape != (len(frame), len(LABEL_ORDER)):
                probs = predict_component(component, texts, max_length, device, precision)
                np.save(ckpt, probs)
        else:
            probs = predict_component(component, texts, max_length, device, precision)
            np.save(ckpt, probs)
        if probs.shape != (len(frame), len(LABEL_ORDER)):
            raise AssertionError(f"Bad probability shape for {component}: {probs.shape}")
        if not np.isfinite(probs).all():
            raise AssertionError(f"Non-finite probabilities for {component}")
        all_probs.append(probs)
        print(f"component {idx}/{len(component_dirs(model_dir))} complete: {component.name}", flush=True)
    avg = np.mean(all_probs, axis=0)
    avg = avg / np.clip(avg.sum(axis=1, keepdims=True), 1e-12, None)
    if not np.allclose(avg.sum(axis=1), 1.0, atol=1e-5):
        raise AssertionError("Averaged probabilities do not sum to 1")
    return avg


def add_v5_columns(data: pd.DataFrame, probs: np.ndarray, model_hashes: dict[str, str]) -> pd.DataFrame:
    out = data.copy()
    out["username_norm"] = out["username"].map(normalize_username)
    out["is_injected"] = out["comment_id"].astype(str).str.match(r"(?i)^INJ")
    out["comment_text_original"] = out["text"].astype(str)
    out["brand_or_video_context"] = out["product_category"].map(brand_from_category)
    for i, label in enumerate(LABEL_ORDER):
        out[f"probability_{label.lower()}"] = probs[:, i]
    out["predicted_sentiment"] = [LABEL_ORDER[int(i)] for i in probs.argmax(axis=1)]
    out["max_probability"] = probs.max(axis=1)
    out["threshold"] = 0.0
    out["no_text"] = False
    out["abstained"] = False
    out["final_sentiment_label"] = out["predicted_sentiment"]
    out["sentiment_status"] = "Evaluable"
    out["model_name"] = "indobenchmark/indobert-base-p2"
    out["selected_candidate"] = read_json(FINAL_MODEL_DIR / "development_selected_candidate.json")["candidate_id"]
    out["model_hash_manifest_sha256"] = sha256_file(FINAL_MODEL_DIR / "SHA256SUMS.txt")
    out["locked_test_validation_status"] = read_json(ACCEPTANCE_DECISION)["status"]
    out["inference_timestamp_utc"] = utc_now()
    out["dataset_provenance"] = "dataset.csv observational rows only; INJ rows excluded"
    return out


def join_v2_metadata(obs: pd.DataFrame) -> pd.DataFrame:
    v2 = read_csv(V2_OBS)
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
        "is_hcc_member_bool",
    ]
    meta = v2[[c for c in meta_cols if c in v2.columns]].drop_duplicates("comment_id")
    out = obs.merge(meta, on="comment_id", how="left")
    out["is_hcc_member_bool"] = out["is_hcc_member_bool"].astype(str).str.lower().eq("true")
    out["hcc_id"] = out["hcc_id"].replace("", "Non-HCC")
    out["brand_label_auto"] = out["brand_label_auto"].replace("", "Non-HCC")
    return out


def actor_type_tables(obs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    actor = read_csv(V2_ACTOR)
    actor["username_norm"] = actor["username"].map(normalize_username)
    account, _ = account_summary(obs)
    merged = actor.merge(account, on="username_norm", how="left", suffixes=("", "_v5"))
    rows = []
    for actor_type, group in merged.groupby("actor_type_primary", dropna=False):
        valid = group[pd.to_numeric(group["evaluable_comments"], errors="coerce").fillna(0) > 0]
        n_valid = int(pd.to_numeric(valid["evaluable_comments"], errors="coerce").fillna(0).sum())
        counts = {label: int(pd.to_numeric(valid[f"{label.lower()}_count"], errors="coerce").fillna(0).sum()) for label in LABEL_ORDER}
        total = sum(counts.values())
        ratios = {label: counts[label] / total if total else 0.0 for label in LABEL_ORDER}
        dominant = max(LABEL_ORDER, key=lambda label: counts[label]) if total else "No evaluable sentiment"
        rows.append(
            {
                "actor_type_primary": actor_type,
                "n_accounts": int(len(group)),
                "n_accounts_with_comments": int(valid["username_norm"].nunique()),
                "n_valid_comments": n_valid,
                "pooled_positive_count": counts["Positive"],
                "pooled_neutral_count": counts["Neutral"],
                "pooled_negative_count": counts["Negative"],
                "pooled_positive_ratio": ratios["Positive"],
                "pooled_neutral_ratio": ratios["Neutral"],
                "pooled_negative_ratio": ratios["Negative"],
                "pooled_dominant_sentiment": dominant,
                "pooled_goal_orientation": hcc_level_goal(ratios["Positive"], ratios["Neutral"], ratios["Negative"], n_valid),
                "goal_validation_status": "INDOBERT_V5_ACCEPTED_AS_FINAL_RM2_SENTIMENT_MODEL",
            }
        )
    pooled = pd.DataFrame(rows)
    by_goal = pooled.groupby(["actor_type_primary", "pooled_goal_orientation"], dropna=False).size().reset_index(name="n_actor_type_rows")
    return pooled, by_goal


def hcc_level_goal(pos: float, neu: float, neg: float, n_valid: int) -> str:
    if n_valid < 5:
        return "Insufficient Text"
    if pos >= 0.45 and pos - max(neu, neg) >= 0.10:
        return "Promotional / Supportive"
    if neg >= 0.45 and neg - max(pos, neu) >= 0.10:
        return "Critical / Complaint"
    if neu >= 0.45 and neu - max(pos, neg) >= 0.10:
        return "Neutral Engagement"
    if pos >= 0.25 and neg >= 0.25:
        return "Polarized / Contested"
    return "Mixed Goals"


def target_brand_summary(obs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for brand, group in obs.groupby("brand_label_auto", dropna=False):
        evaluable = len(group)
        row = {"brand_label_auto": brand, "n_comments": int(len(group)), "evaluable_comments": int(evaluable)}
        for label in LABEL_ORDER:
            count = int(group["final_sentiment_label"].eq(label).sum())
            lo, hi = wilson_ci(count, evaluable)
            row[f"{label.lower()}_count"] = count
            row[f"{label.lower()}_ratio"] = count / evaluable if evaluable else np.nan
            row[f"{label.lower()}_ci_low"] = lo
            row[f"{label.lower()}_ci_high"] = hi
        row["dominant_sentiment"] = max(LABEL_ORDER, key=lambda label: row[f"{label.lower()}_count"]) if evaluable else ""
        rows.append(row)
    return pd.DataFrame(rows).sort_values("n_comments", ascending=False)


def save_visuals(dist: pd.DataFrame, hcc_comment: pd.DataFrame, hcc_summary: pd.DataFrame) -> None:
    VIS_DIR.mkdir(parents=True, exist_ok=True)
    labels = dist.loc[dist["label"].isin(LABEL_ORDER), "label"].tolist()
    counts = dist.loc[dist["label"].isin(LABEL_ORDER), "count"].astype(int).tolist()
    plt.figure(figsize=(7, 4))
    plt.bar(labels, counts, color=["#b64b4b", "#6d7f8f", "#4c8f66"])
    plt.title("IndoBERT V5 Sentiment Distribution")
    plt.tight_layout()
    plt.savefig(VIS_DIR / "indobert_v5_sentiment_distribution_observational.png", dpi=160)
    plt.close()

    plot = hcc_comment.set_index("group")[[f"{label.lower()}_ratio_evaluable" for label in LABEL_ORDER]]
    plot.plot(kind="bar", stacked=True, figsize=(7, 4), color=["#b64b4b", "#6d7f8f", "#4c8f66"])
    plt.title("IndoBERT V5 HCC vs Non-HCC Sentiment")
    plt.tight_layout()
    plt.savefig(VIS_DIR / "indobert_v5_sentiment_hcc_vs_nonhcc_100pct.png", dpi=160)
    plt.close()

    hcc_summary["goal_orientation"].value_counts().plot(kind="bar", figsize=(8, 4), color="#5f7f6e")
    plt.title("IndoBERT V5 HCC Goal Orientation")
    plt.tight_layout()
    plt.savefig(VIS_DIR / "indobert_v5_hcc_goal_orientation.png", dpi=160)
    plt.close()


def write_docs(obs: pd.DataFrame, dist: pd.DataFrame, hcc_summary: pd.DataFrame, actor_pooled: pd.DataFrame, manifest: dict[str, Any]) -> None:
    def md_table(frame: pd.DataFrame) -> str:
        if frame.empty:
            return "_No data._"
        view = frame.copy().astype(str)
        return "| " + " | ".join(view.columns) + " |\n| " + " | ".join(["---"] * len(view.columns)) + " |\n" + "\n".join(
            "| " + " | ".join(row) + " |" for row in view.to_numpy(dtype=str)
        )

    label_counts = dist.loc[dist["label"].isin(LABEL_ORDER), ["label", "count", "percentage_of_total"]]
    goal_counts = hcc_summary["goal_orientation"].value_counts().reset_index()
    goal_counts.columns = ["goal_orientation", "n_hcc"]
    REPORT.write_text(
        "\n".join(
            [
                "# Final RM2 Sentiment IndoBERT V5 Report",
                "",
                f"Status: `{manifest['status']}`.",
                f"Observational denominator: `{len(obs)}` comments. INJ rows were excluded before inference.",
                f"Frozen model: `{manifest['model_artifact']}`.",
                "",
                "## Sentiment Distribution",
                "",
                md_table(label_counts),
                "",
                "## HCC Goal Orientation",
                "",
                md_table(goal_counts),
                "",
                "## Actor-Type Pooled Goals",
                "",
                md_table(actor_pooled),
                "",
                "Goal orientation is a descriptive message-orientation aggregate from sentiment, not evidence of intent, payment, coordination, or causal influence.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    GUIDE.write_text(
        "\n".join(
            [
                "# RM2 Goals Interpretation Guide V5",
                "",
                "IndoBERT V5 is the accepted canonical RM2 sentiment model. Goals remain descriptive orientations of observed messages.",
                "",
                "- `goal_orientation` summarizes sentiment distribution at HCC/account/actor-type level.",
                "- It must not be interpreted as proof of intent, payment, buzzer status, affiliation, or causal effect.",
                "- RM1 topology, LCN, HCC membership, node, edge, and community membership are not recalculated here.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    pres = PRESENTATION_DIR / "sentiment_presentation_summary_v5.md"
    pres.write_text(
        "\n".join(
            [
                "# Ringkasan Presentasi Sentimen V5",
                "",
                f"- Model canonical: `{manifest['status']}`.",
                f"- Denominator observasional: `{len(obs)}` komentar.",
                "- INJ/evaluation rows dikeluarkan dari full inference.",
                "",
                md_table(label_counts),
                "",
                md_table(goal_counts),
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for V5 full inference.")
    decision = read_json(ACCEPTANCE_DECISION)
    if not decision.get("accepted"):
        raise RuntimeError("V5 was not accepted; full inference is not allowed.")
    locked_manifest = read_json(LOCKED_MANIFEST)
    if not locked_manifest.get("evaluated_once"):
        raise RuntimeError("Locked test V5 one-time manifest is not complete.")
    for path in [FINAL_DIR, TABLE_DIR, GEPHI_DIR, PRESENTATION_DIR, VIS_DIR]:
        path.mkdir(parents=True, exist_ok=True)

    safe_copy_candidate_to_final()
    model_hashes = verify_sha256s(FINAL_MODEL_DIR)
    data = read_csv(DATASET)
    data["comment_id"] = data["comment_id"].astype(str)
    data["is_injected"] = data["comment_id"].str.match(r"(?i)^INJ")
    v2_obs_ids = set(read_csv(V2_OBS)["comment_id"].astype(str))
    obs_input = data.loc[data["comment_id"].isin(v2_obs_ids)].copy()
    if len(obs_input) != len(v2_obs_ids):
        raise AssertionError("Dataset does not cover the full V2 observational universe")
    if len(obs_input) != obs_input["comment_id"].nunique():
        raise AssertionError("Duplicate observational comment_id detected")

    precision = "bf16" if torch.cuda.is_bf16_supported() else "fp16"
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    probs = predict_v5(obs_input, FINAL_MODEL_DIR, torch.device("cuda"), precision, resume=True)
    obs = add_v5_columns(obs_input, probs, model_hashes)
    obs = join_v2_metadata(obs)
    obs.to_csv(OUT_OBS, index=False)

    dist = sentiment_distribution(obs, "Observational_V5")
    dist.to_csv(TABLE_DIR / "sentiment_distribution_observational_v5.csv", index=False)
    hcc_comment = hcc_vs_nonhcc_comment(obs)
    hcc_comment.to_csv(TABLE_DIR / "hcc_vs_nonhcc_comment_sentiment_v5.csv", index=False)
    account, account_group = account_summary(obs)
    account.to_csv(TABLE_DIR / "account_sentiment_summary_v5.csv", index=False)
    account_group.to_csv(TABLE_DIR / "hcc_vs_nonhcc_account_sentiment_v5.csv", index=False)
    hcc_summary = hcc_level(obs)
    hcc_summary["goal_method"] = "hard_label_ratios_with_soft_probability_mass_and_bootstrap_stability_v5"
    hcc_summary["goal_validation_status"] = decision["status"]
    hcc_summary.to_csv(TABLE_DIR / "hcc_sentiment_goals_summary_v5.csv", index=False)
    brand = brand_summary(obs, hcc_summary)
    brand.to_csv(TABLE_DIR / "brand_sentiment_summary_v5.csv", index=False)
    target = target_brand_summary(obs)
    target.to_csv(TABLE_DIR / "target_brand_summary_v5.csv", index=False)
    actor_pooled, actor_by_goal = actor_type_tables(obs)
    actor_pooled.to_csv(TABLE_DIR / "actor_type_goals_pooled_v5.csv", index=False)
    actor_by_goal.to_csv(TABLE_DIR / "actor_type_by_goals_v5.csv", index=False)
    actor_pooled.to_csv(ROOT / "output/rm2_actor_type/tables/actor_type_goals_pooled_v5.csv", index=False)
    actor_by_goal.to_csv(ROOT / "output/rm2_actor_type/tables/actor_type_by_goals_v5.csv", index=False)

    if V2_DISTRIBUTION.exists():
        v2_dist = read_csv(V2_DISTRIBUTION)
        comp = v2_dist[["label", "count"]].rename(columns={"count": "v2_count"}).merge(
            dist[["label", "count"]].rename(columns={"count": "v5_count"}), on="label", how="outer"
        )
        comp.to_csv(TABLE_DIR / "v2_vs_v5_sentiment_distribution.csv", index=False)
    if V2_HCC_GOALS.exists():
        v2_goal = read_csv(V2_HCC_GOALS)
        changes = v2_goal[["hcc_id", "goal_orientation"]].rename(columns={"goal_orientation": "v2_goal_orientation"}).merge(
            hcc_summary[["hcc_id", "goal_orientation"]].rename(columns={"goal_orientation": "v5_goal_orientation"}), on="hcc_id", how="outer"
        )
        changes["goal_changed"] = changes["v2_goal_orientation"].ne(changes["v5_goal_orientation"])
        changes.to_csv(TABLE_DIR / "v2_vs_v5_hcc_goal_changes.csv", index=False)

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
    hcc_nodes.merge(node_attrs, on="community", how="left").to_csv(GEPHI_DIR / "gephi_hcc_nodes_sentiment_v5.csv", index=False)
    read_csv(HCC_EDGES).to_csv(GEPHI_DIR / "gephi_hcc_edges_sentiment_v5.csv", index=False)
    if LCN_NODES_ACTOR_TYPE.exists():
        lcn_nodes = read_csv(LCN_NODES_ACTOR_TYPE)
        lcn_nodes["username_norm"] = lcn_nodes["Id"].map(normalize_username)
        lcn_nodes.merge(
            account[["username_norm", "dominant_sentiment", "positive_ratio", "neutral_ratio", "negative_ratio", "coverage", "evaluable_comments"]].rename(
                columns={"coverage": "sentiment_v5_coverage"}
            ),
            on="username_norm",
            how="left",
        ).to_csv(GEPHI_DIR / "gephi_lcn_nodes_actor_type_sentiment_v5.csv", index=False)

    save_visuals(dist, hcc_comment, hcc_summary)
    manifest = {
        "status": decision["status"],
        "created_at_utc": utc_now(),
        "git_head_at_inference": git_head(),
        "model_artifact": rel(FINAL_MODEL_DIR),
        "candidate_artifact": rel(CANDIDATE_DIR),
        "locked_test_manifest": rel(LOCKED_MANIFEST),
        "locked_test_evaluated_once": True,
        "no_retraining_after_locked_test": True,
        "dataset": rel(DATASET),
        "dataset_rows": int(len(data)),
        "observational_rows": int(len(obs_input)),
        "injected_rows_excluded": int(data["is_injected"].sum()),
        "rows_excluded_not_in_v2_observational_universe": int(len(data) - len(obs_input)),
        "unique_observational_comment_id": int(obs_input["comment_id"].nunique()),
        "prediction_rule": "probability_average_argmax",
        "component_model_count": len(component_dirs(FINAL_MODEL_DIR)),
        "precision": precision,
        "output": rel(OUT_OBS),
        "output_sha256": sha256_file(OUT_OBS),
        "label_map": {"Negative": 0, "Neutral": 1, "Positive": 2},
        "package_versions": package_versions(),
        "rm1_lcn_hcc_topology_recomputed": False,
        "full_inference_excludes_injected_evaluation_rows": True,
        "outputs": [
            rel(OUT_OBS),
            rel(TABLE_DIR / "hcc_sentiment_goals_summary_v5.csv"),
            rel(TABLE_DIR / "actor_type_goals_pooled_v5.csv"),
            rel(TABLE_DIR / "target_brand_summary_v5.csv"),
            rel(GUIDE),
            rel(REPORT),
        ],
    }
    save_json(OUT_MANIFEST, manifest)
    save_json(
        CANONICAL_MODEL,
        {
            "status": "CANONICAL_RM2_SENTIMENT_MODEL",
            "canonical_model": "indobert_v5_final",
            "canonical_status": decision["status"],
            "model_artifact": rel(FINAL_MODEL_DIR),
            "full_inference_output": rel(OUT_OBS),
            "baseline_legacy_v2": "output/rm2_sentiment/final/comment_sentiment_v2_observational.csv",
            "locked_test_evaluation": rel(LOCKED_EVAL_DIR),
            "created_at_utc": utc_now(),
        },
    )
    write_docs(obs, dist, hcc_summary, actor_pooled, manifest)
    summary_rows = [
        {"metric": "status", "value": decision["status"]},
        {"metric": "observational_rows", "value": len(obs_input)},
        {"metric": "injected_rows_excluded", "value": int(data["is_injected"].sum())},
        {"metric": "rows_excluded_not_in_v2_observational_universe", "value": int(len(data) - len(obs_input))},
        {"metric": "hcc_count", "value": int(hcc_summary["hcc_id"].nunique())},
    ]
    for _, row in dist.iterrows():
        summary_rows.append({"metric": f"observational_{row['label']}", "value": row["count"]})
    pd.DataFrame(summary_rows).to_csv(SUMMARY, index=False)
    print(json.dumps({"status": decision["status"], "observational_rows": len(obs_input), "output": rel(OUT_OBS)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
