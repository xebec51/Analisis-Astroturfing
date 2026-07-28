from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
)

from scripts.train_rm2_sentiment_indobert_v5_development import LABELS, LABEL_TO_ID, Trial


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "output/rm2_sentiment/experiments/indobert_v5_development"
OOF_PATH = OUT_DIR / "development_oof_predictions.csv"
METRICS_PATH = OUT_DIR / "development_fold_seed_metrics.csv"
ACCEPTANCE_CONFIG = ROOT / "configs/rm2_sentiment_v5_acceptance_preregistered.json"
COVERAGE_TARGETS = [0.9343, 0.95, 1.0]
PROB_COLS = ["prob_negative", "prob_neutral", "prob_positive"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


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


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2), encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_calibration_error(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> float:
    pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    correct = pred == y_true
    ece = 0.0
    for lo in np.linspace(0, 1, n_bins, endpoint=False):
        hi = lo + 1 / n_bins
        mask = (conf >= lo) & (conf < hi if hi < 1 else conf <= hi)
        if mask.any():
            ece += float(mask.mean()) * abs(float(correct[mask].mean()) - float(conf[mask].mean()))
    return float(ece)


def metric_bundle(y_true: np.ndarray, probs: np.ndarray, stage: str, instability_penalty: float = 0.0) -> dict[str, Any]:
    if len(y_true) == 0:
        raise ValueError("Empty metric input")
    probability_finite = bool(np.isfinite(probs).all())
    probability_sum_valid = bool(np.allclose(probs.sum(axis=1), 1.0, atol=1e-4))
    if not probability_finite or not probability_sum_valid:
        base_bad = 1.0
        probs = np.nan_to_num(probs, nan=1 / len(LABELS), posinf=1 / len(LABELS), neginf=1 / len(LABELS))
        probs = probs / np.clip(probs.sum(axis=1, keepdims=True), 1e-12, None)
    else:
        base_bad = 0.0
    pred = probs.argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, pred, labels=list(range(len(LABELS))), zero_division=0
    )
    true_dist = np.bincount(y_true, minlength=len(LABELS)) / len(y_true)
    pred_dist = np.bincount(pred, minlength=len(LABELS)) / len(pred)
    macro = float(f1_score(y_true, pred, labels=list(range(len(LABELS))), average="macro", zero_division=0))
    bal = float(balanced_accuracy_score(y_true, pred))
    mcc = float(matthews_corrcoef(y_true, pred))
    acc = float(accuracy_score(y_true, pred))
    min_recall = float(recall.min())
    pos_recall = float(recall[LABEL_TO_ID["Positive"]])
    negative_drift = abs(float(pred_dist[LABEL_TO_ID["Negative"]] - true_dist[LABEL_TO_ID["Negative"]]))
    class_collapse = bool((pred_dist > 0).sum() < len(LABELS))
    raw = 0.25 * macro + 0.20 * bal + 0.20 * mcc + 0.15 * min_recall + 0.15 * pos_recall + 0.05 * acc
    penalty = base_bad
    if class_collapse:
        penalty += 1.0
    if negative_drift > 0.08:
        penalty += (negative_drift - 0.08) * 0.50
    penalty += instability_penalty
    one_hot = np.eye(len(LABELS))[y_true]
    metrics: dict[str, Any] = {
        "stage": stage,
        "n": int(len(y_true)),
        "accuracy": acc,
        "macro_f1": macro,
        "weighted_f1": float(f1_score(y_true, pred, labels=list(range(len(LABELS))), average="weighted", zero_division=0)),
        "balanced_accuracy": bal,
        "mcc": mcc,
        "min_class_recall": min_recall,
        "positive_recall": pos_recall,
        "selection_score_raw": float(raw),
        "selection_score": float(raw - penalty),
        "selection_penalty": float(penalty),
        "seed_instability_penalty": float(instability_penalty),
        "class_collapse": class_collapse,
        "negative_prediction_share": float(pred_dist[LABEL_TO_ID["Negative"]]),
        "negative_prediction_drift": negative_drift,
        "probability_finite": probability_finite,
        "probability_sum_valid": probability_sum_valid,
        "oof_complete": int(len(y_true)) == 977,
        "auto_disqualified": bool(
            class_collapse
            or min_recall < 0.55
            or pos_recall < 0.60
            or int(len(y_true)) != 977
            or not probability_finite
            or not probability_sum_valid
        ),
        "ece": expected_calibration_error(y_true, probs),
        "brier_score": float(np.mean([brier_score_loss(one_hot[:, i], probs[:, i]) for i in range(len(LABELS))])),
    }
    for label, p, r, f, s in zip(LABELS, precision, recall, f1, support):
        key = label.lower()
        metrics[f"{key}_precision"] = float(p)
        metrics[f"{key}_recall"] = float(r)
        metrics[f"{key}_f1"] = float(f)
        metrics[f"{key}_support"] = int(s)
        metrics[f"predicted_{key}"] = int((pred == LABEL_TO_ID[label]).sum())
    return metrics


def confusion_payload(y_true: np.ndarray, probs: np.ndarray) -> dict[str, list[int]]:
    cm = confusion_matrix(y_true, probs.argmax(axis=1), labels=list(range(len(LABELS))))
    return {label: [int(x) for x in row] for label, row in zip(LABELS, cm)}


def threshold_for_coverage(confidence: np.ndarray, target: float) -> float:
    if target >= 1.0:
        return 0.0
    return float(np.quantile(confidence, 1 - target, method="lower"))


def risk_coverage(candidate_id: str, y_true: np.ndarray, probs: np.ndarray) -> pd.DataFrame:
    pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    rows = []
    for target in COVERAGE_TARGETS:
        threshold = threshold_for_coverage(conf, target)
        covered = conf >= threshold
        rows.append(
            {
                "candidate_id": candidate_id,
                "coverage_target": target,
                "confidence_threshold": threshold,
                "actual_coverage": float(covered.mean()),
                "covered_accuracy": float(accuracy_score(y_true[covered], pred[covered])) if covered.any() else np.nan,
                "covered_macro_f1": float(f1_score(y_true[covered], pred[covered], labels=list(range(len(LABELS))), average="macro", zero_division=0)) if covered.any() else np.nan,
                "full_set_accuracy_abstain_wrong": float(((pred == y_true) & covered).mean()),
                "n_abstained": int((~covered).sum()),
                "locked_test_used": False,
            }
        )
    return pd.DataFrame(rows)


def parse_trial_from_oof(group: pd.DataFrame) -> Trial:
    row = group.iloc[0]
    return Trial(
        input_mode=str(row["input_mode"]) if "input_mode" in row else str(row["trial_id"]).split("__")[1],
        max_length=int(row["max_length"]) if "max_length" in row else int(str(row["trial_id"]).split("__len")[1].split("__")[0]),
        learning_rate=float(row["learning_rate"]) if "learning_rate" in row else float(str(row["trial_id"]).split("__lr")[1].split("__")[0].replace("m", "-")),
        classifier_dropout=float(row["classifier_dropout"]) if "classifier_dropout" in row else 0.1,
        loss=str(row["loss"]) if "loss" in row else "cross_entropy",
        label_smoothing=float(row["label_smoothing"]) if "label_smoothing" in row else 0.0,
    )


def single_candidates(oof: pd.DataFrame, stage: str) -> tuple[list[dict[str, Any]], dict[str, Any], list[pd.DataFrame]]:
    rows: list[dict[str, Any]] = []
    confusions: dict[str, Any] = {}
    risk_rows: list[pd.DataFrame] = []
    for (trial_id, seed), group in oof.loc[oof["stage"].eq(stage)].groupby(["trial_id", "seed"]):
        group = group.sort_values("annotation_id")
        y = group["true_label"].map(LABEL_TO_ID).to_numpy(dtype=int)
        probs = group[PROB_COLS].astype(float).to_numpy()
        candidate_id = f"single__{stage}__{trial_id}__seed{seed}"
        metrics = metric_bundle(y, probs, stage)
        rows.append(
            {
                "candidate_id": candidate_id,
                "candidate_type": "single_seed",
                "trial_id": trial_id,
                "seed": int(seed),
                "component_seeds": str(seed),
                **metrics,
            }
        )
        confusions[candidate_id] = confusion_payload(y, probs)
        risk_rows.append(risk_coverage(candidate_id, y, probs))
    return rows, confusions, risk_rows


def seed_ensembles(oof: pd.DataFrame, stage: str) -> tuple[list[dict[str, Any]], dict[str, Any], list[pd.DataFrame]]:
    rows: list[dict[str, Any]] = []
    confusions: dict[str, Any] = {}
    risk_rows: list[pd.DataFrame] = []
    stage_oof = oof.loc[oof["stage"].eq(stage)].copy()
    for trial_id, group in stage_oof.groupby("trial_id"):
        seeds = sorted(group["seed"].astype(int).unique().tolist())
        if seeds != [42, 52, 62]:
            continue
        pivoted = []
        for seed in seeds:
            part = group.loc[group["seed"].astype(int).eq(seed)].sort_values("annotation_id")
            if len(part) != 977:
                pivoted = []
                break
            pivoted.append(part)
        if not pivoted or not all(pivoted[0]["annotation_id"].tolist() == p["annotation_id"].tolist() for p in pivoted):
            continue
        y = pivoted[0]["true_label"].map(LABEL_TO_ID).to_numpy(dtype=int)
        probs_stack = np.stack([p[PROB_COLS].astype(float).to_numpy() for p in pivoted], axis=0)
        seed_metrics = []
        for p in pivoted:
            seed_metrics.append(metric_bundle(y, p[PROB_COLS].astype(float).to_numpy(), stage))
        instability = float(
            0.20 * np.std([m["macro_f1"] for m in seed_metrics], ddof=0)
            + 0.20 * np.std([m["mcc"] for m in seed_metrics], ddof=0)
            + 0.20 * np.std([m["positive_recall"] for m in seed_metrics], ddof=0)
        )
        avg_probs = probs_stack.mean(axis=0)
        candidate_id = f"seed_prob_average__{stage}__{trial_id}"
        rows.append(
            {
                "candidate_id": candidate_id,
                "candidate_type": "seed_probability_average",
                "trial_id": trial_id,
                "seed": "42,52,62",
                "component_seeds": "42,52,62",
                **metric_bundle(y, avg_probs, stage, instability),
            }
        )
        confusions[candidate_id] = confusion_payload(y, avg_probs)
        risk_rows.append(risk_coverage(candidate_id, y, avg_probs))
        votes = np.stack([p[PROB_COLS].astype(float).to_numpy().argmax(axis=1) for p in pivoted], axis=0)
        vote_probs = np.zeros_like(avg_probs)
        for i in range(votes.shape[1]):
            counts = np.bincount(votes[:, i], minlength=len(LABELS)).astype(float)
            vote_probs[i] = counts / counts.sum()
        candidate_id = f"seed_majority_vote__{stage}__{trial_id}"
        rows.append(
            {
                "candidate_id": candidate_id,
                "candidate_type": "seed_majority_vote",
                "trial_id": trial_id,
                "seed": "42,52,62",
                "component_seeds": "42,52,62",
                **metric_bundle(y, vote_probs, stage, instability),
            }
        )
        confusions[candidate_id] = confusion_payload(y, vote_probs)
        risk_rows.append(risk_coverage(candidate_id, y, vote_probs))
    return rows, confusions, risk_rows


def top_two_blend(oof: pd.DataFrame, stage: str, leaderboard: pd.DataFrame) -> tuple[list[dict[str, Any]], dict[str, Any], list[pd.DataFrame]]:
    rows: list[dict[str, Any]] = []
    confusions: dict[str, Any] = {}
    risk_rows: list[pd.DataFrame] = []
    seed_avg = leaderboard.loc[(leaderboard["stage"].eq(stage)) & (leaderboard["candidate_type"].eq("seed_probability_average"))].copy()
    if len(seed_avg) < 2:
        return rows, confusions, risk_rows
    seed_avg = seed_avg.loc[~seed_avg["auto_disqualified"].astype(bool)].sort_values("selection_score", ascending=False)
    if len(seed_avg) < 2:
        return rows, confusions, risk_rows
    trials = seed_avg["trial_id"].head(2).tolist()
    pieces = []
    for trial_id in trials:
        group = oof.loc[oof["stage"].eq(stage) & oof["trial_id"].eq(trial_id)].copy()
        if sorted(group["seed"].astype(int).unique().tolist()) != [42, 52, 62]:
            return rows, confusions, risk_rows
        seed_parts = [group.loc[group["seed"].astype(int).eq(seed)].sort_values("annotation_id") for seed in [42, 52, 62]]
        pieces.append(seed_parts[0][["annotation_id", "true_label"]].assign(**{
            "prob_negative": np.stack([p["prob_negative"].astype(float).to_numpy() for p in seed_parts], axis=0).mean(axis=0),
            "prob_neutral": np.stack([p["prob_neutral"].astype(float).to_numpy() for p in seed_parts], axis=0).mean(axis=0),
            "prob_positive": np.stack([p["prob_positive"].astype(float).to_numpy() for p in seed_parts], axis=0).mean(axis=0),
        }))
    if not pieces[0]["annotation_id"].tolist() == pieces[1]["annotation_id"].tolist():
        return rows, confusions, risk_rows
    y = pieces[0]["true_label"].map(LABEL_TO_ID).to_numpy(dtype=int)
    probs = (pieces[0][PROB_COLS].to_numpy(dtype=float) + pieces[1][PROB_COLS].to_numpy(dtype=float)) / 2
    candidate_id = f"top2_probability_blend__{stage}__{trials[0]}__PLUS__{trials[1]}"
    rows.append(
        {
            "candidate_id": candidate_id,
            "candidate_type": "top2_probability_blend",
            "trial_id": " + ".join(trials),
            "seed": "42,52,62",
            "component_seeds": "42,52,62",
            **metric_bundle(y, probs, stage),
        }
    )
    confusions[candidate_id] = confusion_payload(y, probs)
    risk_rows.append(risk_coverage(candidate_id, y, probs))
    return rows, confusions, risk_rows


def select_candidate(leaderboard: pd.DataFrame, metrics: pd.DataFrame, oof: pd.DataFrame, stage: str) -> dict[str, Any]:
    eligible = leaderboard.loc[leaderboard["stage"].eq(stage) & ~leaderboard["auto_disqualified"].astype(bool)].copy()
    if eligible.empty:
        raise RuntimeError(f"No eligible candidates in {stage}")
    eligible = eligible.sort_values("selection_score", ascending=False)
    best_single = eligible.loc[eligible["candidate_type"].eq("single_seed")].head(1)
    best_any = eligible.head(1).iloc[0].to_dict()
    if not best_single.empty:
        single = best_single.iloc[0].to_dict()
        if best_any["candidate_type"] != "single_seed":
            improvement = max(float(best_any["macro_f1"]) - float(single["macro_f1"]), float(best_any["selection_score"]) - float(single["selection_score"]))
            if improvement < 0.005:
                best_any = single
    candidate_type = str(best_any["candidate_type"])
    component_seeds = [int(x) for x in str(best_any["component_seeds"]).split(",") if str(x).strip().isdigit()]
    trial_id = str(best_any["trial_id"]).split(" + ")[0]
    trial_oof = oof.loc[oof["stage"].eq(stage) & oof["trial_id"].eq(trial_id)].copy()
    trial = parse_trial_from_oof(trial_oof)
    metric_rows = metrics.loc[
        metrics["stage"].eq(stage)
        & metrics["trial_id"].eq(trial_id)
        & metrics["fold"].astype(str).ne("ALL_OOF")
    ].copy()
    best_epochs = pd.to_numeric(metric_rows["best_epoch"], errors="coerce").dropna()
    final_epochs = int(max(1, round(float(best_epochs.median())))) if not best_epochs.empty else 4
    risk = pd.read_csv(OUT_DIR / "development_risk_coverage_curve.csv", dtype=str, keep_default_na=False)
    abstention = risk.loc[risk["candidate_id"].eq(best_any["candidate_id"])].to_dict("records")
    return {
        "status": "INDOBERT_V5_DEVELOPMENT_CANDIDATE_SELECTED",
        "created_at_utc": utc_now(),
        "selection_basis": "development OOF only",
        "locked_test_used": False,
        "candidate_id": best_any["candidate_id"],
        "candidate_type": candidate_type,
        "prediction_rule": "probability_average_argmax" if candidate_type in {"seed_probability_average", "top2_probability_blend"} else "three_class_argmax",
        "trial_id": trial_id,
        "trial": asdict(trial),
        "component_seeds": component_seeds,
        "final_epoch_count": final_epochs,
        "metrics": best_any,
        "abstention_policy": abstention,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Select IndoBERT V5 candidate using development OOF only.")
    parser.add_argument("--stage", default="stage2", choices=["stage1", "stage2"])
    parser.add_argument("--output-dir", default=str(OUT_DIR))
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    globals()["OUT_DIR"] = output_dir
    oof_path = output_dir / "development_oof_predictions.csv"
    metrics_path = output_dir / "development_fold_seed_metrics.csv"
    oof = pd.read_csv(oof_path, dtype=str, keep_default_na=False, low_memory=False)
    metrics = pd.read_csv(metrics_path, dtype=str, keep_default_na=False, low_memory=False)
    for col in PROB_COLS:
        oof[col] = pd.to_numeric(oof[col], errors="coerce")
    rows, confusions, risk_rows = single_candidates(oof, args.stage)
    e_rows, e_conf, e_risk = seed_ensembles(oof, args.stage)
    rows.extend(e_rows)
    confusions.update(e_conf)
    risk_rows.extend(e_risk)
    leaderboard = pd.DataFrame(rows)
    if not leaderboard.empty:
        leaderboard["selection_score"] = pd.to_numeric(leaderboard["selection_score"], errors="coerce")
    b_rows, b_conf, b_risk = top_two_blend(oof, args.stage, leaderboard)
    rows.extend(b_rows)
    confusions.update(b_conf)
    risk_rows.extend(b_risk)
    leaderboard = pd.DataFrame(rows).sort_values("selection_score", ascending=False)
    leaderboard.to_csv(output_dir / "development_candidate_leaderboard.csv", index=False, encoding="utf-8-sig")
    leaderboard.to_csv(output_dir / "development_ensemble_selection_summary.csv", index=False, encoding="utf-8-sig")
    if risk_rows:
        pd.concat(risk_rows, ignore_index=True).to_csv(output_dir / "development_risk_coverage_curve.csv", index=False, encoding="utf-8-sig")
    save_json(output_dir / "development_confusion_matrices.json", confusions)
    selected = select_candidate(leaderboard, metrics, oof, args.stage)
    save_json(output_dir / "development_selected_candidate.json", selected)
    save_json(
        output_dir / "INDOBERT_V5_DEVELOPMENT_SELECTION_MANIFEST.json",
        {
            "status": "INDOBERT_V5_DEVELOPMENT_SELECTION_COMPLETE",
            "created_at_utc": utc_now(),
            "stage": args.stage,
            "oof_predictions": rel(oof_path),
            "locked_test_used": False,
            "selection_score_formula": "0.25*macro_f1 + 0.20*balanced_accuracy + 0.20*mcc + 0.15*minimum_class_recall + 0.15*positive_recall + 0.05*accuracy - preregistered penalties",
            "acceptance_config": rel(ACCEPTANCE_CONFIG),
            "acceptance_config_sha256": sha256_file(ACCEPTANCE_CONFIG),
            "selected_candidate": selected,
        },
    )
    print(json.dumps({"status": selected["status"], "candidate_id": selected["candidate_id"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
