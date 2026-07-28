from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, matthews_corrcoef, precision_recall_fscore_support


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "output/rm2_sentiment/experiments/indobert_v5_development"
LABELS = ["Negative", "Neutral", "Positive"]
LABEL_TO_ID = {label: idx for idx, label in enumerate(LABELS)}
COVERAGE_TARGETS = [0.9343, 0.95, 1.0]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def selection_metrics(y_true: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    pred = probs.argmax(axis=1)
    _, recall, _, _ = precision_recall_fscore_support(y_true, pred, labels=list(range(len(LABELS))), zero_division=0)
    negative_share = float((pred == LABEL_TO_ID["Negative"]).mean())
    metrics = {
        "accuracy": float(accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, labels=list(range(len(LABELS))), average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "mcc": float(matthews_corrcoef(y_true, pred)),
        "min_class_recall": float(np.min(recall)),
        "positive_recall": float(recall[LABEL_TO_ID["Positive"]]),
        "negative_prediction_share": negative_share,
        "class_collapse": bool(len(set(pred.tolist())) < len(LABELS)),
    }
    penalty = 0.0
    if metrics["class_collapse"]:
        penalty += 1.0
    if negative_share > 0.45:
        penalty += (negative_share - 0.45) * 0.5
    metrics["selection_score"] = float(
        0.25 * metrics["macro_f1"]
        + 0.20 * metrics["balanced_accuracy"]
        + 0.20 * metrics["mcc"]
        + 0.15 * metrics["min_class_recall"]
        + 0.15 * metrics["positive_recall"]
        + 0.05 * metrics["accuracy"]
        - penalty
    )
    return metrics


def threshold_for_coverage(confidence: np.ndarray, target: float) -> float:
    if target >= 1.0:
        return 0.0
    return float(np.quantile(confidence, 1 - target, method="lower"))


def risk_coverage(y_true: np.ndarray, probs: np.ndarray) -> pd.DataFrame:
    pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    rows = []
    for target in COVERAGE_TARGETS:
        threshold = threshold_for_coverage(conf, target)
        covered = conf >= threshold
        rows.append(
            {
                "coverage_target": target,
                "confidence_threshold": threshold,
                "actual_coverage": float(covered.mean()),
                "covered_accuracy": float(accuracy_score(y_true[covered], pred[covered])) if covered.any() else np.nan,
                "full_set_accuracy_abstain_wrong": float(((pred == y_true) & covered).mean()),
                "n_abstained": int((~covered).sum()),
                "locked_test_used": False,
            }
        )
    return pd.DataFrame(rows)


def plan_only() -> dict[str, Any]:
    payload = {
        "status": "INDOBERT_V5_ENSEMBLE_ABSTENTION_PLAN_READY_PENDING_OOF",
        "created_at_utc": utc_now(),
        "selection_source": "development_oof_only",
        "locked_test_used": False,
        "ensemble_candidates": [
            "single_best_seed",
            "probability_average_ensemble",
            "majority_vote_ensemble"
        ],
        "coverage_targets": COVERAGE_TARGETS,
        "selection_score_formula": "0.25*macro_f1 + 0.20*balanced_accuracy + 0.20*mcc + 0.15*min_class_recall + 0.15*positive_recall + 0.05*accuracy - penalties",
        "penalties": [
            "class_collapse",
            "excessive_negative_prediction_share_gt_0.45",
            "seed_instability_to_be_applied_after_multi_seed_oof"
        ]
    }
    write_json(OUT_DIR / "INDOBERT_V5_ENSEMBLE_ABSTENTION_PLAN.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Select V5 model/ensemble and abstention using development OOF only.")
    parser.add_argument("--oof-predictions", help="OOF CSV with true_label, trial_id, seed, and probability columns.")
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not args.oof_predictions:
        payload = plan_only()
        print(json.dumps({"status": payload["status"]}, indent=2), flush=True)
        return

    oof_path = Path(args.oof_predictions)
    if not oof_path.is_absolute():
        oof_path = ROOT / oof_path
    oof = pd.read_csv(oof_path, dtype=str, keep_default_na=False, low_memory=False)
    required = {"true_label", "trial_id", "seed", "prob_negative", "prob_neutral", "prob_positive"}
    missing = sorted(required - set(oof.columns))
    if missing:
        raise ValueError(f"OOF predictions missing columns: {missing}")
    rows = []
    risk_rows = []
    for (trial_id, seed), group in oof.groupby(["trial_id", "seed"]):
        y = group["true_label"].map(LABEL_TO_ID).astype(int).to_numpy()
        probs = group[["prob_negative", "prob_neutral", "prob_positive"]].astype(float).to_numpy()
        row = {"candidate_type": "single_seed", "trial_id": trial_id, "seed": seed, **selection_metrics(y, probs)}
        rows.append(row)
        risk = risk_coverage(y, probs)
        risk["candidate_type"] = "single_seed"
        risk["trial_id"] = trial_id
        risk["seed"] = seed
        risk_rows.append(risk)
    summary = pd.DataFrame(rows).sort_values("selection_score", ascending=False)
    summary.to_csv(OUT_DIR / "development_ensemble_selection_summary.csv", index=False, encoding="utf-8-sig")
    pd.concat(risk_rows, ignore_index=True).to_csv(OUT_DIR / "development_risk_coverage_curve.csv", index=False, encoding="utf-8-sig")
    write_json(
        OUT_DIR / "INDOBERT_V5_ENSEMBLE_ABSTENTION_SELECTION_MANIFEST.json",
        {
            "status": "INDOBERT_V5_DEVELOPMENT_ONLY_SELECTION_COMPLETE",
            "created_at_utc": utc_now(),
            "oof_predictions": rel(oof_path),
            "locked_test_used": False,
            "top_candidate": summary.iloc[0].to_dict() if not summary.empty else {},
        },
    )
    print(json.dumps({"status": "INDOBERT_V5_DEVELOPMENT_ONLY_SELECTION_COMPLETE"}, indent=2), flush=True)


if __name__ == "__main__":
    main()
