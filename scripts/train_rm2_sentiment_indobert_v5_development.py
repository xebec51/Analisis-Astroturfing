from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import re
import subprocess
import unicodedata
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


ROOT = Path(__file__).resolve().parents[1]
HUMAN_V5_DIR = ROOT / "output/rm2_sentiment/validation/human_v5"
LOCKED_V5_DIR = ROOT / "output/rm2_sentiment/validation/human_v5_locked_test"
DEV_CANDIDATES = HUMAN_V5_DIR / "sentiment_v5_development_candidates.csv"
LOCKED_CANDIDATES = LOCKED_V5_DIR / "sentiment_v5_locked_test_candidates.csv"
OUT_DIR = ROOT / "output/rm2_sentiment/experiments/indobert_v5_development"
ACCEPTANCE_CONFIG = ROOT / "configs/rm2_sentiment_v5_acceptance_preregistered.json"

LABELS = ["Negative", "Neutral", "Positive"]
LABEL_TO_ID = {label: idx for idx, label in enumerate(LABELS)}
SEEDS = [42, 52, 62]


@dataclass(frozen=True)
class Trial:
    model_id: str
    input_mode: str
    max_length: int
    learning_rate: float
    classifier_dropout: float
    loss: str
    label_smoothing: float

    @property
    def trial_id(self) -> str:
        lr = f"{self.learning_rate:.0e}".replace("-", "m").replace("+", "")
        smooth = str(self.label_smoothing).replace(".", "p")
        drop = str(self.classifier_dropout).replace(".", "p")
        return (
            f"indobert_base_p2__{self.input_mode}__len{self.max_length}"
            f"__lr{lr}__drop{drop}__{self.loss}__ls{smooth}"
        )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def rel(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False, low_memory=False)


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_text(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "null", "<na>"}:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"https?://\S+|www\.\S+", " HTTPURL ", text)
    text = re.sub(r"@\w+", " @USER ", text)
    return re.sub(r"\s+", " ", text).strip()


def model_input(row: pd.Series, input_mode: str) -> str:
    comment = normalize_text(row.get("comment_text"))
    context = normalize_text(row.get("brand_or_video_context")) or normalize_text(row.get("product_category"))
    parent = normalize_text(row.get("parent_comment_text"))
    if input_mode == "comment_only":
        return comment
    if input_mode == "context_sep_comment":
        return f"{context} [SEP] {comment}" if context else comment
    if input_mode == "target_context_parent_comment":
        parts = [part for part in [context, parent, comment] if part]
        return " [SEP] ".join(parts)
    raise ValueError(f"Unknown input_mode: {input_mode}")


def build_trials() -> list[Trial]:
    losses = [
        "cross_entropy",
        "weighted_cross_entropy",
        "focal_loss_gamma_1.0",
        "focal_loss_gamma_1.5",
        "focal_loss_gamma_2.0",
    ]
    trials = []
    for input_mode, max_length, lr, dropout, loss, smoothing in itertools.product(
        ["comment_only", "context_sep_comment", "target_context_parent_comment"],
        [128, 256],
        [1e-5, 2e-5, 3e-5],
        [0.1, 0.2, 0.3],
        losses,
        [0.0, 0.025, 0.05],
    ):
        trials.append(
            Trial(
                model_id="indobenchmark/indobert-base-p2",
                input_mode=input_mode,
                max_length=max_length,
                learning_rate=lr,
                classifier_dropout=dropout,
                loss=loss,
                label_smoothing=smoothing,
            )
        )
    return trials


def completed_development_frame() -> pd.DataFrame:
    if not DEV_CANDIDATES.exists():
        raise FileNotFoundError(DEV_CANDIDATES)
    frame = read_csv(DEV_CANDIDATES)
    label_col = "sentiment_toward_target"
    mask = frame[label_col].isin(LABELS)
    data = frame.loc[mask].copy()
    data["label_id"] = data[label_col].map(LABEL_TO_ID).astype(int)
    data["normalized_text_group"] = data["comment_text"].map(normalize_text).str.lower()
    data["hard_group"] = data["normalized_text_group"].where(data["normalized_text_group"].ne(""), data["comment_id"])
    return data


def validate_candidate_lists() -> dict[str, Any]:
    dev = read_csv(DEV_CANDIDATES)
    locked = read_csv(LOCKED_CANDIDATES)
    dev_ids = set(dev["comment_id"])
    locked_ids = set(locked["comment_id"])
    dev_text = set(dev["comment_text"].map(normalize_text).str.lower())
    locked_text = set(locked["comment_text"].map(normalize_text).str.lower())
    return {
        "development_rows": int(len(dev)),
        "locked_test_candidate_rows": int(len(locked)),
        "comment_id_overlap": int(len(dev_ids & locked_ids)),
        "normalized_text_overlap": int(len((dev_text & locked_text) - {""})),
        "development_labels_available": int(dev["sentiment_toward_target"].isin(LABELS).sum()),
        "locked_labels_available": int(locked["sentiment_toward_target"].isin(LABELS).sum()) if "sentiment_toward_target" in locked.columns else 0,
        "locked_test_v5_used_for_training": False,
    }


def build_fold_assignments(data: pd.DataFrame, n_splits: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for seed in SEEDS:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for fold, (_, val_idx) in enumerate(
            splitter.split(data, data["label_id"].to_numpy(), groups=data["hard_group"].to_numpy()),
            start=1,
        ):
            for row_idx in val_idx:
                row = data.iloc[int(row_idx)]
                rows.append(
                    {
                        "seed": seed,
                        "fold": fold,
                        "row_index": int(row_idx),
                        "annotation_id": row["annotation_id"],
                        "comment_id": row["comment_id"],
                        "label": row["sentiment_toward_target"],
                        "hard_group": row["hard_group"],
                        "video_id": row["video_id"],
                    }
                )
    return pd.DataFrame(rows)


def write_plan_outputs() -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    trials = build_trials()
    grid = pd.DataFrame([{**asdict(trial), "trial_id": trial.trial_id} for trial in trials])
    grid.to_csv(OUT_DIR / "candidate_grid_manifest.csv", index=False, encoding="utf-8-sig")
    risk = pd.DataFrame(
        [
            {"coverage_target": 0.9343, "policy": "development_oof_confidence_threshold", "locked_test_used": False},
            {"coverage_target": 0.95, "policy": "development_oof_confidence_threshold", "locked_test_used": False},
            {"coverage_target": 1.0, "policy": "argmax_full_coverage", "locked_test_used": False},
        ]
    )
    risk.to_csv(OUT_DIR / "development_risk_coverage_policy.csv", index=False, encoding="utf-8-sig")
    validation = validate_candidate_lists()
    manifest = {
        "status": "INDOBERT_V5_DEVELOPMENT_PIPELINE_READY_PENDING_HUMAN_LABELS",
        "created_at_utc": utc_now(),
        "git_sha": git_head(),
        "primary_label": "sentiment_toward_target",
        "label_order": LABELS,
        "seeds": SEEDS,
        "candidate_trial_count": len(trials),
        "selection_source": "development_oof_only",
        "locked_test_v4_errors_used": False,
        "locked_test_v5_labels_used_for_training_or_selection": False,
        "acceptance_config": rel(ACCEPTANCE_CONFIG),
        "acceptance_config_sha256": sha256_file(ACCEPTANCE_CONFIG) if ACCEPTANCE_CONFIG.exists() else "",
        "selection_score": {
            "base": "macro_f1 + balanced_accuracy + mcc + min_class_recall + positive_recall",
            "stability_penalty": "seed metric standard deviation",
            "class_collapse_penalty": True,
            "excessive_negative_prediction_penalty": True,
        },
        "validation": validation,
        "outputs": {
            "candidate_grid": rel(OUT_DIR / "candidate_grid_manifest.csv"),
            "risk_coverage_policy": rel(OUT_DIR / "development_risk_coverage_policy.csv"),
        },
    }
    write_json(OUT_DIR / "INDOBERT_V5_DEVELOPMENT_PIPELINE_MANIFEST.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare leak-free IndoBERT V5 development pipeline artifacts.")
    parser.add_argument("--make-folds", action="store_true", help="Create grouped CV folds after human labels are available.")
    parser.add_argument("--n-splits", type=int, default=5)
    args = parser.parse_args()
    manifest = write_plan_outputs()
    if args.make_folds:
        data = completed_development_frame()
        if len(data) < 100:
            raise RuntimeError("Human V5 development labels are not ready; refusing to create folds.")
        folds = build_fold_assignments(data, args.n_splits)
        folds.to_csv(OUT_DIR / "development_grouped_fold_assignments.csv", index=False, encoding="utf-8-sig")
        manifest["status"] = "INDOBERT_V5_DEVELOPMENT_FOLDS_READY"
        manifest["fold_assignment_rows"] = int(len(folds))
        write_json(OUT_DIR / "INDOBERT_V5_DEVELOPMENT_PIPELINE_MANIFEST.json", manifest)
    print(json.dumps({"status": manifest["status"], "candidate_trial_count": manifest["candidate_trial_count"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
