from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from safetensors.torch import save_file as save_safetensors
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "output/rm2_sentiment/validation/human_v3/human_label_registry_v3.csv"
OUT_DIR = ROOT / "output/rm2_sentiment/experiments/indobert_v3"
MODEL_DIR = ROOT / "output/rm2_sentiment/model/indobert_v3_candidate"

LABELS = ["Negative", "Neutral", "Positive"]
LABEL_TO_ID = {label: idx for idx, label in enumerate(LABELS)}
SEEDS = [42, 52, 62]
FOLDS = ["1", "2", "3", "4", "5"]
MAX_EPOCHS = 8
PATIENCE = 2
BATCH_SIZE_HEAD = 32
GRAD_CLIP = 1.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_REVISIONS = {
    "apriandito/indobert-sentiment-classifier": "ba24f8cea1c00090cc4fce2b63c61ed307943a78",
    "indobenchmark/indobert-large-p2": "4b280c3bfcc1ed2d6b4589be5c876076b7d73568",
    "indolem/indobertweet-base-uncased": "32e28c05b47e33b6675d2670a1745c50a65e987a",
}

CPU_RESOURCE_BLOCKED = {
    "indobenchmark/indobert-large-p2": "RESOURCE_BLOCKED_CPU_MODEL_LOAD_TIMEOUT",
}


@dataclass(frozen=True)
class Trial:
    loss_name: str
    learning_rate: float
    max_length: int
    warmup_ratio: float
    weight_decay: float
    focal_gamma: float | None

    @property
    def suffix(self) -> str:
        gamma = "na" if self.focal_gamma is None else str(self.focal_gamma).replace(".", "p")
        lr = f"{self.learning_rate:.0e}".replace("-", "m")
        return f"{self.loss_name}_lr{lr}_len{self.max_length}_warm{self.warmup_ratio}_wd{self.weight_decay}_g{gamma}"


TRIALS = [
    Trial("weighted_cross_entropy", 1e-5, 128, 0.06, 0.01, None),
    Trial("weighted_cross_entropy", 2e-5, 192, 0.10, 0.05, None),
    Trial("weighted_cross_entropy", 3e-5, 256, 0.06, 0.05, None),
    Trial("focal_loss", 1e-5, 192, 0.10, 0.01, 1.0),
    Trial("focal_loss", 2e-5, 256, 0.06, 0.01, 1.5),
    Trial("focal_loss", 3e-5, 128, 0.10, 0.05, 2.0),
    Trial("class_balanced_focal_loss", 1e-5, 256, 0.10, 0.05, 1.0),
    Trial("class_balanced_focal_loss", 2e-5, 128, 0.10, 0.01, 1.5),
    Trial("class_balanced_focal_loss", 3e-5, 192, 0.06, 0.05, 2.0),
]


def safe_model_name(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", model_id)


def read_registry() -> pd.DataFrame:
    if not REGISTRY.exists():
        raise FileNotFoundError(f"Run build_rm2_sentiment_v3_human_registry.py first: {REGISTRY}")
    frame = pd.read_csv(REGISTRY, dtype=str, keep_default_na=False, low_memory=False)
    dev = frame.loc[frame["split_family"].eq("development") & frame["final_sentiment_label"].isin(LABELS)].copy()
    if dev.empty:
        raise RuntimeError("No V3 development rows found.")
    return dev.reset_index(drop=True)


def context_text(row: pd.Series) -> str:
    parts = []
    for col in ["product_category", "brand_or_video_context"]:
        value = str(row.get(col, "")).strip()
        if value:
            parts.append(value)
    context = " | ".join(dict.fromkeys(parts))
    comment = str(row.get("model_text") or row.get("comment_text_original") or "").strip()
    return f"{context} [SEP] {comment}" if context else comment


def sha256_dataframe(frame: pd.DataFrame, columns: list[str]) -> str:
    data = frame[columns].sort_values(columns).to_csv(index=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def package_versions() -> dict[str, str]:
    versions = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": str(torch.cuda.is_available()),
        "device": str(DEVICE),
    }
    try:
        import transformers

        versions["transformers"] = transformers.__version__
    except Exception as exc:  # pragma: no cover
        versions["transformers"] = f"unavailable:{exc}"
    try:
        import sklearn

        versions["sklearn"] = sklearn.__version__
    except Exception as exc:  # pragma: no cover
        versions["sklearn"] = f"unavailable:{exc}"
    try:
        import safetensors

        versions["safetensors"] = safetensors.__version__
    except Exception as exc:  # pragma: no cover
        versions["safetensors"] = f"unavailable:{exc}"
    return versions


def extract_embeddings(model_id: str, revision: str, texts: list[str], max_length: int, batch_size: int) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=False)
    model = AutoModel.from_pretrained(model_id, revision=revision, trust_remote_code=False)
    model.to(DEVICE)
    model.eval()

    vectors: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            batch = texts[start : start + batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(DEVICE) for key, value in encoded.items()}
            output = model(**encoded)
            pooled = output.pooler_output if getattr(output, "pooler_output", None) is not None else output.last_hidden_state[:, 0, :]
            vectors.append(pooled.detach().cpu().numpy().astype("float32"))
    return np.vstack(vectors)


class LinearHead(nn.Module):
    def __init__(self, n_features: int, n_classes: int) -> None:
        super().__init__()
        self.classifier = nn.Linear(n_features, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


def class_weights(y: np.ndarray) -> torch.Tensor:
    counts = np.bincount(y, minlength=len(LABELS)).astype("float64")
    weights = len(y) / np.clip(len(LABELS) * counts, 1e-9, None)
    return torch.tensor(weights, dtype=torch.float32, device=DEVICE)


def class_balanced_weights(y: np.ndarray, beta: float = 0.999) -> torch.Tensor:
    counts = np.bincount(y, minlength=len(LABELS)).astype("float64")
    effective = 1.0 - np.power(beta, np.clip(counts, 1, None))
    weights = (1.0 - beta) / np.clip(effective, 1e-12, None)
    weights = weights / weights.sum() * len(LABELS)
    return torch.tensor(weights, dtype=torch.float32, device=DEVICE)


def compute_loss(logits: torch.Tensor, targets: torch.Tensor, loss_name: str, weights: torch.Tensor, gamma: float | None) -> torch.Tensor:
    ce = nn.functional.cross_entropy(logits, targets, weight=weights, reduction="none")
    if loss_name == "weighted_cross_entropy":
        return ce.mean()
    probs = torch.softmax(logits, dim=1)
    pt = probs.gather(1, targets.view(-1, 1)).squeeze(1).clamp(1e-6, 1.0)
    focal = ((1.0 - pt) ** float(gamma or 1.0)) * ce
    return focal.mean()


def softmax_np(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / np.clip(exp.sum(axis=1, keepdims=True), 1e-12, None)


def make_loader(x: np.ndarray, y: np.ndarray, batch_size: int, seed: int, shuffle: bool) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    ds = TensorDataset(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, generator=generator)


def train_one_fold(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    trial: Trial,
    seed: int,
) -> tuple[np.ndarray, int, float]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train).astype("float32")
    x_val_scaled = scaler.transform(x_val).astype("float32")

    model = LinearHead(x_train_scaled.shape[1], len(LABELS)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=trial.learning_rate, weight_decay=trial.weight_decay)
    total_steps = max(1, math.ceil(len(x_train_scaled) / BATCH_SIZE_HEAD) * MAX_EPOCHS)
    warmup_steps = int(total_steps * trial.warmup_ratio)

    def lr_lambda(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return max(1e-8, step / max(1, warmup_steps))
        remaining = max(1, total_steps - warmup_steps)
        return max(0.0, (total_steps - step) / remaining)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    loss_weights = class_balanced_weights(y_train) if trial.loss_name == "class_balanced_focal_loss" else class_weights(y_train)
    train_loader = make_loader(x_train_scaled, y_train, BATCH_SIZE_HEAD, seed, shuffle=True)
    val_tensor = torch.tensor(x_val_scaled, dtype=torch.float32, device=DEVICE)

    best_state: dict[str, torch.Tensor] | None = None
    best_macro = -1.0
    best_epoch = 0
    stagnant = 0
    step = 0
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = compute_loss(logits, yb, trial.loss_name, loss_weights, trial.focal_gamma)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            scheduler.step()
            step += 1

        model.eval()
        with torch.no_grad():
            logits_val = model(val_tensor).detach().cpu().numpy()
        preds = logits_val.argmax(axis=1)
        macro = f1_score(y_val, preds, labels=list(range(len(LABELS))), average="macro", zero_division=0)
        if macro > best_macro + 1e-9:
            best_macro = float(macro)
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stagnant = 0
        else:
            stagnant += 1
            if stagnant >= PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(val_tensor).detach().cpu().numpy()
    probs = softmax_np(logits)
    return probs, best_epoch, best_macro


def metric_row(y_true: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    pred = probs.argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        pred,
        labels=list(range(len(LABELS))),
        zero_division=0,
    )
    out: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, labels=list(range(len(LABELS))), average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, pred, labels=list(range(len(LABELS))), average="weighted", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "mcc": float(matthews_corrcoef(y_true, pred)),
        "coverage": 1.0,
        "positive_recall": float(recall[LABEL_TO_ID["Positive"]]),
        "positive_precision": float(precision[LABEL_TO_ID["Positive"]]),
        "positive_f1": float(f1[LABEL_TO_ID["Positive"]]),
    }
    for idx, label in enumerate(LABELS):
        lower = label.lower()
        out[f"{lower}_precision"] = float(precision[idx])
        out[f"{lower}_recall"] = float(recall[idx])
        out[f"{lower}_f1"] = float(f1[idx])
        out[f"{lower}_support"] = int(support[idx])
    out.update(calibration_metrics(y_true, probs))
    return out


def calibration_metrics(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> dict[str, float]:
    pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    correct = pred == y_true
    ece = 0.0
    for lo in np.linspace(0, 1, n_bins, endpoint=False):
        hi = lo + 1 / n_bins
        mask = (conf >= lo) & (conf < hi if hi < 1 else conf <= hi)
        if mask.any():
            ece += float(mask.mean()) * abs(float(correct[mask].mean()) - float(conf[mask].mean()))
    one_hot = np.eye(len(LABELS))[y_true]
    brier = float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))
    return {"ece": float(ece), "brier_score": brier, "mean_confidence": float(conf.mean())}


def run_trial(model_id: str, revision: str, trial: Trial, data: pd.DataFrame, texts: list[str], embeddings_cache: dict[int, np.ndarray]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if trial.max_length not in embeddings_cache:
        print(f"Extracting embeddings: {model_id} max_length={trial.max_length}", flush=True)
        embeddings_cache[trial.max_length] = extract_embeddings(model_id, revision, texts, trial.max_length, batch_size=16)

    x_all = embeddings_cache[trial.max_length]
    y_all = data["final_sentiment_label"].map(LABEL_TO_ID).to_numpy(dtype=int)
    oof_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    trial_id = f"{safe_model_name(model_id)}__{trial.suffix}"

    for seed in SEEDS:
        seed_probs = np.zeros((len(data), len(LABELS)), dtype=float)
        seed_mask = np.zeros(len(data), dtype=bool)
        best_epochs: list[int] = []
        for fold in FOLDS:
            val_idx = data.index[data["fold_v3"].eq(fold)].to_numpy()
            train_idx = data.index[~data["fold_v3"].eq(fold)].to_numpy()
            probs, best_epoch, best_macro = train_one_fold(
                x_all[train_idx],
                y_all[train_idx],
                x_all[val_idx],
                y_all[val_idx],
                trial,
                seed,
            )
            seed_probs[val_idx] = probs
            seed_mask[val_idx] = True
            best_epochs.append(best_epoch)
            fold_metrics = metric_row(y_all[val_idx], probs)
            metric_rows.append(
                {
                    "trial_id": trial_id,
                    "model_id": model_id,
                    "revision": revision,
                    "training_mode": "frozen_encoder_linear_head_cpu" if DEVICE.type == "cpu" else "frozen_encoder_linear_head_gpu",
                    "seed": seed,
                    "fold": fold,
                    "loss": trial.loss_name,
                    "learning_rate": trial.learning_rate,
                    "max_length": trial.max_length,
                    "warmup_ratio": trial.warmup_ratio,
                    "weight_decay": trial.weight_decay,
                    "focal_gamma": trial.focal_gamma if trial.focal_gamma is not None else "",
                    "best_epoch": best_epoch,
                    "best_validation_macro_f1": best_macro,
                    **fold_metrics,
                }
            )
            for row_idx, source_idx in enumerate(val_idx):
                row = data.iloc[source_idx]
                pred_idx = int(probs[row_idx].argmax())
                oof_rows.append(
                    {
                        "trial_id": trial_id,
                        "model_id": model_id,
                        "revision": revision,
                        "seed": seed,
                        "fold": fold,
                        "comment_id": row["comment_id"],
                        "true_label": row["final_sentiment_label"],
                        "predicted_label": LABELS[pred_idx],
                        "confidence": float(probs[row_idx].max()),
                        "prob_negative": float(probs[row_idx, LABEL_TO_ID["Negative"]]),
                        "prob_neutral": float(probs[row_idx, LABEL_TO_ID["Neutral"]]),
                        "prob_positive": float(probs[row_idx, LABEL_TO_ID["Positive"]]),
                        "registry_role": row["registry_role"],
                        "text_cluster_id": row["text_cluster_id"],
                        "cv_group_id": row["cv_group_id"],
                    }
                )

        if not seed_mask.all():
            raise AssertionError(f"OOF mask incomplete for {trial_id} seed={seed}")
        seed_metrics = metric_row(y_all, seed_probs)
        metric_rows.append(
            {
                "trial_id": trial_id,
                "model_id": model_id,
                "revision": revision,
                "training_mode": "frozen_encoder_linear_head_cpu" if DEVICE.type == "cpu" else "frozen_encoder_linear_head_gpu",
                "seed": seed,
                "fold": "ALL_OOF",
                "loss": trial.loss_name,
                "learning_rate": trial.learning_rate,
                "max_length": trial.max_length,
                "warmup_ratio": trial.warmup_ratio,
                "weight_decay": trial.weight_decay,
                "focal_gamma": trial.focal_gamma if trial.focal_gamma is not None else "",
                "best_epoch": int(np.median(best_epochs)),
                "best_validation_macro_f1": seed_metrics["macro_f1"],
                **seed_metrics,
            }
        )
    return oof_rows, metric_rows


def summarize_trials(metrics: pd.DataFrame) -> pd.DataFrame:
    all_oof = metrics.loc[metrics["fold"].eq("ALL_OOF")].copy()
    metric_cols = [
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "balanced_accuracy",
        "mcc",
        "positive_recall",
        "positive_precision",
        "positive_f1",
        "ece",
        "brier_score",
        "mean_confidence",
    ]
    rows = []
    for trial_id, group in all_oof.groupby("trial_id"):
        first = group.iloc[0]
        row = {
            "trial_id": trial_id,
            "model_id": first["model_id"],
            "revision": first["revision"],
            "training_mode": first["training_mode"],
            "loss": first["loss"],
            "learning_rate": first["learning_rate"],
            "max_length": first["max_length"],
            "warmup_ratio": first["warmup_ratio"],
            "weight_decay": first["weight_decay"],
            "focal_gamma": first["focal_gamma"],
            "n_seeds": int(group["seed"].nunique()),
        }
        for col in metric_cols:
            row[f"mean_{col}"] = float(pd.to_numeric(group[col]).mean())
            row[f"std_{col}"] = float(pd.to_numeric(group[col]).std(ddof=0))
        rows.append(row)
    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    summary = summary.sort_values(
        [
            "mean_macro_f1",
            "mean_balanced_accuracy",
            "mean_mcc",
            "mean_positive_recall",
            "mean_positive_f1",
            "std_macro_f1",
        ],
        ascending=[False, False, False, False, False, True],
    ).reset_index(drop=True)
    summary["selected_for_freeze"] = False
    summary.loc[0, "selected_for_freeze"] = True
    return summary


def train_full_candidate(data: pd.DataFrame, texts: list[str], selected: pd.Series) -> dict[str, object]:
    model_id = str(selected["model_id"])
    revision = str(selected["revision"])
    trial = Trial(
        loss_name=str(selected["loss"]),
        learning_rate=float(selected["learning_rate"]),
        max_length=int(selected["max_length"]),
        warmup_ratio=float(selected["warmup_ratio"]),
        weight_decay=float(selected["weight_decay"]),
        focal_gamma=None if str(selected["focal_gamma"]).strip() == "" else float(selected["focal_gamma"]),
    )
    print(f"Freezing V3 candidate: {model_id} {trial.suffix}", flush=True)
    x = extract_embeddings(model_id, revision, texts, trial.max_length, batch_size=16)
    y = data["final_sentiment_label"].map(LABEL_TO_ID).to_numpy(dtype=int)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x).astype("float32")

    best_epochs = pd.to_numeric(selected.get("median_best_epoch", 0), errors="coerce")
    epochs = int(best_epochs) if pd.notna(best_epochs) and int(best_epochs) > 0 else MAX_EPOCHS

    torch.manual_seed(42)
    model = LinearHead(x_scaled.shape[1], len(LABELS)).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=trial.learning_rate, weight_decay=trial.weight_decay)
    total_steps = max(1, math.ceil(len(x_scaled) / BATCH_SIZE_HEAD) * epochs)
    warmup_steps = int(total_steps * trial.warmup_ratio)

    def lr_lambda(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return max(1e-8, step / max(1, warmup_steps))
        remaining = max(1, total_steps - warmup_steps)
        return max(0.0, (total_steps - step) / remaining)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    weights = class_balanced_weights(y) if trial.loss_name == "class_balanced_focal_loss" else class_weights(y)
    loader = make_loader(x_scaled, y, BATCH_SIZE_HEAD, 42, shuffle=True)
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = compute_loss(logits, yb, trial.loss_name, weights, trial.focal_gamma)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            scheduler.step()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    state = {f"linear_head.{key}": value.detach().cpu() for key, value in model.state_dict().items()}
    save_safetensors(state, MODEL_DIR / "linear_head.safetensors")
    joblib.dump(scaler, MODEL_DIR / "embedding_scaler.joblib")
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=False)
    tokenizer.save_pretrained(MODEL_DIR / "tokenizer")

    config = {
        "status": "INDOBERT_V3_CANDIDATE_FROZEN",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": model_id,
        "revision": revision,
        "trust_remote_code": False,
        "training_mode": str(selected["training_mode"]),
        "encoder_update_status": "frozen_encoder_no_base_model_committed",
        "classifier_head": "single linear layer trained on human-supervised development folds",
        "preprocessing": {
            "normalization": [
                "Unicode NFKC",
                "HTML entity unescape",
                "control character removal",
                "URL to HTTPURL",
                "mention to @USER",
                "whitespace normalization",
            ],
            "preserved": [
                "negation",
                "emoji",
                "emoticon",
                "elongated words",
                "abbreviations",
                "brand names",
                "product names",
                "numbers",
                "question and exclamation marks",
                "Indonesian-English code mixing",
            ],
        },
        "context_format": "context [SEP] comment_text",
        "allowed_context_fields": ["product_category", "brand_or_video_context"],
        "forbidden_context_fields": [
            "label",
            "prediction",
            "HCC goal",
            "goal_orientation",
            "confusion matrix",
            "evaluation result",
        ],
        "label_mapping": LABEL_TO_ID,
        "hyperparameters": {
            "loss": trial.loss_name,
            "learning_rate": trial.learning_rate,
            "max_length": trial.max_length,
            "warmup_ratio": trial.warmup_ratio,
            "weight_decay": trial.weight_decay,
            "focal_gamma": trial.focal_gamma,
            "max_epochs": epochs,
            "early_stopping_patience": PATIENCE,
            "gradient_clipping": GRAD_CLIP,
            "gradient_accumulation": 1,
            "mixed_precision": bool(DEVICE.type == "cuda"),
            "class_weights_scope": "training fold only during OOF; full development pool for frozen candidate",
            "threshold": {"prediction_rule": "argmax", "abstention_threshold": 0.0},
            "calibration": {"method": "none", "selection_metric": "development OOF ECE/Brier reported only"},
            "seed": 42,
        },
        "development_data_hash": sha256_dataframe(data, ["comment_id", "final_sentiment_label", "fold_v3", "cv_group_id"]),
        "package_versions": package_versions(),
        "artifacts": {
            "linear_head_safetensors": "linear_head.safetensors",
            "embedding_scaler": "embedding_scaler.joblib",
            "tokenizer_dir": "tokenizer",
        },
    }
    (MODEL_DIR / "v3_candidate_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Develop and freeze an IndoBERT V3 sentiment candidate.")
    parser.add_argument(
        "--models",
        nargs="*",
        default=[
            "apriandito/indobert-sentiment-classifier",
            "indolem/indobertweet-base-uncased",
            "indobenchmark/indobert-large-p2",
        ],
        help="HF model ids to attempt.",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = read_registry()
    data["model_input"] = data.apply(context_text, axis=1)
    texts = data["model_input"].tolist()

    grid_rows = []
    all_oof: list[dict[str, object]] = []
    all_metrics: list[dict[str, object]] = []

    for model_id in args.models:
        revision = MODEL_REVISIONS[model_id]
        if DEVICE.type == "cpu" and model_id in CPU_RESOURCE_BLOCKED:
            for trial in TRIALS:
                grid_rows.append(
                    {
                        "model_id": model_id,
                        "revision": revision,
                        "trial": trial.suffix,
                        "status": CPU_RESOURCE_BLOCKED[model_id],
                        "notes": "Model load exceeded local CPU runtime limit; not used for V3 selection.",
                    }
                )
            continue
        embeddings_cache: dict[int, np.ndarray] = {}
        for trial in TRIALS:
            grid_rows.append(
                {
                    "model_id": model_id,
                    "revision": revision,
                    "trial": trial.suffix,
                    "status": "RUN",
                    "notes": "OOF development run on human labels only.",
                }
            )
            print(f"Running trial: {model_id} {trial.suffix}", flush=True)
            oof_rows, metric_rows = run_trial(model_id, revision, trial, data, texts, embeddings_cache)
            all_oof.extend(oof_rows)
            all_metrics.extend(metric_rows)

    grid = pd.DataFrame(grid_rows)
    grid.to_csv(OUT_DIR / "candidate_grid_manifest.csv", index=False, encoding="utf-8-sig")
    if not all_metrics:
        raise RuntimeError("No IndoBERT V3 candidate completed development training.")

    oof = pd.DataFrame(all_oof)
    metrics = pd.DataFrame(all_metrics)
    oof.to_csv(OUT_DIR / "development_oof_predictions.csv", index=False, encoding="utf-8-sig")
    metrics.to_csv(OUT_DIR / "development_fold_seed_metrics.csv", index=False, encoding="utf-8-sig")

    summary = summarize_trials(metrics)
    best_epochs = (
        metrics.loc[~metrics["fold"].eq("ALL_OOF")]
        .groupby("trial_id")["best_epoch"]
        .median()
        .rename("median_best_epoch")
        .reset_index()
    )
    summary = summary.merge(best_epochs, on="trial_id", how="left")
    summary.to_csv(OUT_DIR / "development_trial_summary.csv", index=False, encoding="utf-8-sig")

    selected = summary.loc[summary["selected_for_freeze"].eq(True)].iloc[0]
    config = train_full_candidate(data, texts, selected)

    manifest = {
        "status": "INDOBERT_V3_DEVELOPMENT_COMPLETE",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "registry": REGISTRY.relative_to(ROOT).as_posix(),
        "output_dir": OUT_DIR.relative_to(ROOT).as_posix(),
        "candidate_dir": MODEL_DIR.relative_to(ROOT).as_posix(),
        "selection_basis": "out-of-fold development predictions only",
        "locked_test_used_for_training_or_selection": False,
        "positive_shift_policy": "No post-hoc Positive shifting; final candidate uses argmax with no low-confidence-to-Positive rule.",
        "resource_note": "CPU runtime used frozen IndoBERT encoders plus supervised linear heads; full encoder fine-tuning is not claimed for resource-blocked models.",
        "selected_trial": selected.to_dict(),
        "candidate_config": config,
        "package_versions": package_versions(),
    }
    (OUT_DIR / "INDOBERT_V3_DEVELOPMENT_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"selected_trial": str(selected["trial_id"]), "mean_macro_f1": float(selected["mean_macro_f1"])}, indent=2))


if __name__ == "__main__":
    main()
