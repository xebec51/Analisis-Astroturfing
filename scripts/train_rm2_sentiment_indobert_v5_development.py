from __future__ import annotations

import argparse
import gc
import hashlib
import html
import json
import math
import os
import platform
import random
import re
import shutil
import subprocess
import sys
import time
import unicodedata
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedGroupKFold
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup


ROOT = Path(__file__).resolve().parents[1]
DEV_REGISTRY = ROOT / "output/rm2_sentiment/validation/human_v5/sentiment_v5_development_final_registry.csv"
FINAL_IMPORT_MANIFEST = ROOT / "output/rm2_sentiment/validation/human_v5/SENTIMENT_V5_FINAL_IMPORT_MANIFEST.json"
OUT_DIR_DEFAULT = ROOT / "output/rm2_sentiment/experiments/indobert_v5_development"
ARTIFACT_DIR = ROOT / "artifacts/rm2_sentiment/indobert_v5_candidate"
ACCEPTANCE_CONFIG = ROOT / "configs/rm2_sentiment_v5_acceptance_preregistered.json"

MODEL_ID = "indobenchmark/indobert-base-p2"
LABELS = ["Negative", "Neutral", "Positive"]
LABEL_TO_ID = {label: idx for idx, label in enumerate(LABELS)}
ID_TO_LABEL = {idx: label for label, idx in LABEL_TO_ID.items()}
EXPECTED_DEV_HASH = "31a5537a0483405632dd8c33bf2190ca405a2e8c77f84f571328be48d1b6004c"
EXPECTED_DEV_COUNTS = {"Negative": 178, "Neutral": 569, "Positive": 230, "Uncertain": 13, "No Text": 10}
OOF_KEYS = ["stage", "trial_id", "seed", "fold", "annotation_id"]
METRIC_KEYS = ["stage", "trial_id", "seed", "fold"]
COVERAGE_TARGETS = [0.9343, 0.95, 1.0]
MODEL_HASH_SUFFIXES = {".safetensors", ".bin", ".json", ".txt", ".model", ".vocab"}
FORBIDDEN_SUPERVISION_SOURCES = [
    "locked_test_v4_labels",
    "locked_test_v4_errors",
    "locked_test_v5_labels",
    "v2_predictions",
    "v4_predictions",
    "hcc",
    "actor_type",
    "network_position",
    "goals_previous",
    "account_name",
    "coordination_status",
    "buzzer_status",
]


@dataclass(frozen=True)
class Trial:
    input_mode: str
    max_length: int
    learning_rate: float
    classifier_dropout: float
    loss: str
    label_smoothing: float
    warmup_ratio: float = 0.06
    weight_decay: float = 0.01
    model_id: str = MODEL_ID

    @property
    def trial_id(self) -> str:
        lr = f"{self.learning_rate:.0e}".replace("-", "m").replace("+", "")
        dr = str(self.classifier_dropout).replace(".", "p")
        ls = str(self.label_smoothing).replace(".", "p")
        return (
            f"indobert_base_p2__{self.input_mode}__len{self.max_length}"
            f"__lr{lr}__drop{dr}__{self.loss}__ls{ls}"
        )


class TextDataset(Dataset):
    def __init__(self, texts: list[str], labels: np.ndarray | None = None) -> None:
        self.texts = texts
        self.labels = labels

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        item: dict[str, Any] = {"text": self.texts[idx]}
        if self.labels is not None:
            item["label"] = int(self.labels[idx])
        return item


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


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_dataframe(frame: pd.DataFrame, columns: list[str]) -> str:
    serialised = frame[columns].sort_values(columns).to_csv(index=False).encode("utf-8")
    return hashlib.sha256(serialised).hexdigest()


def write_sha256s(path: Path) -> dict[str, str]:
    hashes = {
        item.relative_to(path).as_posix(): sha256_file(item)
        for item in sorted(path.rglob("*"))
        if item.is_file() and item.name != "SHA256SUMS.txt" and not item.name.endswith("_MANIFEST.json")
    }
    (path / "SHA256SUMS.txt").write_text(
        "\n".join(f"{digest}  {name}" for name, digest in sorted(hashes.items())) + "\n",
        encoding="utf-8",
    )
    return hashes


def git_head() -> str:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, check=True, text=True, capture_output=True)
        return result.stdout.strip()
    except Exception:
        return "UNKNOWN_GIT_HEAD"


def package_versions() -> dict[str, str]:
    versions = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": str(torch.version.cuda),
        "cuda_available": str(torch.cuda.is_available()),
    }
    for package in ["transformers", "sklearn", "pandas", "numpy", "safetensors"]:
        try:
            module = __import__(package)
            versions[package] = str(getattr(module, "__version__", "unknown"))
        except Exception as exc:
            versions[package] = f"unavailable:{exc}"
    return versions


def device_info(precision: str) -> dict[str, Any]:
    info: dict[str, Any] = {"device": "cuda" if torch.cuda.is_available() else "cpu", "precision": precision}
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info.update(
            {
                "gpu_name": torch.cuda.get_device_name(0),
                "cuda_total_memory_bytes": int(props.total_memory),
                "bf16_supported": bool(torch.cuda.is_bf16_supported()),
                "torch_cuda": str(torch.version.cuda),
            }
        )
    return info


def is_true(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def nonempty(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"", "nan", "none", "null", "<na>"} else text


def normalize_text(value: object) -> str:
    text = html.unescape(nonempty(value))
    text = unicodedata.normalize("NFKC", text)
    text = "".join(ch for ch in text if unicodedata.category(ch)[0] != "C" or ch in "\t\n\r")
    text = re.sub(r"https?://\S+|www\.\S+", " HTTPURL ", text)
    text = re.sub(r"@\w+", " @USER ", text)
    return re.sub(r"\s+", " ", text).strip()


def model_input(row: pd.Series | dict[str, Any], input_mode: str) -> str:
    getter = row.get if isinstance(row, dict) else row.get
    comment = normalize_text(getter("comment_text"))
    context_parts = [normalize_text(getter("brand_or_video_context")), normalize_text(getter("product_category"))]
    context = " | ".join(dict.fromkeys(part for part in context_parts if part))
    parent = normalize_text(getter("parent_comment_text"))
    if input_mode == "comment_only":
        return comment
    if input_mode == "context_sep_comment":
        return f"{context} [SEP] {comment}" if context else comment
    if input_mode == "target_context_parent_comment":
        parts = [part for part in [context, parent, comment] if part]
        return " [SEP] ".join(parts)
    raise ValueError(f"Unknown input_mode: {input_mode}")


def read_development_registry() -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.read_csv(DEV_REGISTRY, dtype=str, keep_default_na=False, low_memory=False)
    required = {
        "annotation_id",
        "comment_id",
        "video_id",
        "product_category",
        "brand_or_video_context",
        "comment_text",
        "parent_comment_text",
        "final_human_label",
        "evaluable_three_class",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Development registry missing columns: {missing}")
    manifest = read_json(FINAL_IMPORT_MANIFEST)
    observed_hash = sha256_dataframe(frame, ["annotation_id", "comment_id", "final_human_label"])
    if observed_hash != manifest["development"]["dataset_hash"] or observed_hash != EXPECTED_DEV_HASH:
        raise AssertionError(f"Development hash mismatch: {observed_hash}")
    if len(frame) != 1000:
        raise AssertionError(f"Development registry rows={len(frame)} expected=1000")
    counts = frame["final_human_label"].value_counts().reindex(list(EXPECTED_DEV_COUNTS), fill_value=0).astype(int).to_dict()
    if counts != EXPECTED_DEV_COUNTS:
        raise AssertionError(f"Development label counts mismatch: {counts}")
    mask = frame["evaluable_three_class"].map(is_true) & frame["final_human_label"].isin(LABELS)
    data = frame.loc[mask].copy().reset_index(drop=True)
    if len(data) != 977:
        raise AssertionError(f"Development evaluable rows={len(data)} expected=977")
    data["label_id"] = data["final_human_label"].map(LABEL_TO_ID).astype(int)
    data["cv_group_id"] = data["comment_id"].astype(str)
    return frame, data


def class_distribution(data: pd.DataFrame) -> dict[str, int]:
    return data["final_human_label"].value_counts().reindex(LABELS, fill_value=0).astype(int).to_dict()


def log_event(output_dir: Path, payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "development_training_log.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(to_jsonable({"created_at_utc": utc_now(), **payload}), ensure_ascii=False) + "\n")
    print(json.dumps(to_jsonable(payload)), flush=True)


def upsert_csv(path: Path, rows: list[dict[str, Any]], keys: list[str]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    incoming = pd.DataFrame(rows)
    if path.exists() and path.stat().st_size > 0:
        existing = pd.read_csv(path, dtype=str, keep_default_na=False, low_memory=False)
        combined = pd.concat([existing, incoming], ignore_index=True, sort=False)
    else:
        combined = incoming
    available = [key for key in keys if key in combined.columns]
    combined.drop_duplicates(available, keep="last").to_csv(path, index=False, encoding="utf-8-sig")


def completed_metric_keys(path: Path) -> set[tuple[str, str, str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, low_memory=False)
    if not set(METRIC_KEYS).issubset(frame.columns):
        return set()
    fold_rows = frame.loc[~frame["fold"].eq("ALL_OOF")]
    return set(map(tuple, fold_rows[METRIC_KEYS].astype(str).to_numpy()))


def load_folds(data: pd.DataFrame, seed: int, n_splits: int, stage: str, output_dir: Path) -> pd.DataFrame:
    if stage == "stage2":
        path = output_dir / "development_grouped_fold_assignments.csv"
        if path.exists():
            folds = pd.read_csv(path, dtype=str, keep_default_na=False, low_memory=False)
        else:
            legacy = output_dir / "development_grouped_fold_assignments.csv"
            folds = pd.read_csv(legacy, dtype=str, keep_default_na=False, low_memory=False)
        if "row_index" not in folds.columns:
            index_by_id = {cid: idx for idx, cid in enumerate(data["comment_id"].astype(str))}
            folds["row_index"] = folds["comment_id"].astype(str).map(index_by_id)
        if "cv_group_id" not in folds.columns and "hard_group" in folds.columns:
            folds["cv_group_id"] = folds["hard_group"]
        folds = folds.loc[folds["seed"].astype(int).eq(int(seed))].copy()
        if folds["fold"].nunique() != n_splits:
            raise AssertionError(f"Frozen stage2 folds for seed {seed} have {folds['fold'].nunique()} folds, expected {n_splits}")
        return folds
    path = output_dir / "development_grouped_fold_assignments_stage1_3fold.csv"
    if path.exists():
        folds = pd.read_csv(path, dtype=str, keep_default_na=False, low_memory=False)
        return folds.loc[folds["seed"].astype(int).eq(int(seed))].copy()
    rows: list[dict[str, Any]] = []
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (_, val_idx) in enumerate(
        splitter.split(data, data["label_id"].to_numpy(), groups=data["cv_group_id"].to_numpy()),
        start=1,
    ):
        for row_index in val_idx:
            row = data.iloc[int(row_index)]
            rows.append(
                {
                    "seed": int(seed),
                    "fold": int(fold),
                    "row_index": int(row_index),
                    "annotation_id": row["annotation_id"],
                    "comment_id": row["comment_id"],
                    "final_human_label": row["final_human_label"],
                    "cv_group_id": row["cv_group_id"],
                }
            )
    folds = pd.DataFrame(rows)
    folds.to_csv(path, index=False, encoding="utf-8-sig")
    return folds


def validate_fold_leakage(folds: pd.DataFrame) -> None:
    if "cv_group_id" not in folds.columns and "hard_group" in folds.columns:
        folds = folds.copy()
        folds["cv_group_id"] = folds["hard_group"]
    leakage = folds.groupby(["seed", "cv_group_id"])["fold"].nunique()
    if int((leakage > 1).sum()) != 0:
        raise AssertionError("Grouped-fold leakage detected")


def collate_batch(tokenizer, max_length: int):
    def _collate(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        encoded = tokenizer(
            [item["text"] for item in batch],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        if "label" in batch[0]:
            encoded["labels"] = torch.tensor([int(item["label"]) for item in batch], dtype=torch.long)
        return encoded

    return _collate


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def class_weights_from_training(labels: np.ndarray) -> torch.Tensor:
    counts = np.bincount(labels.astype(int), minlength=len(LABELS)).astype(float)
    weights = counts.sum() / np.clip(len(LABELS) * counts, 1.0, None)
    return torch.tensor(weights, dtype=torch.float32)


def focal_gamma(loss_name: str) -> float | None:
    match = re.fullmatch(r"focal_loss_gamma_([0-9.]+)", loss_name)
    return float(match.group(1)) if match else None


def loss_for_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    trial: Trial,
    class_weights: torch.Tensor | None,
) -> torch.Tensor:
    weights = class_weights.to(logits.device) if trial.loss == "weighted_cross_entropy" and class_weights is not None else None
    gamma = focal_gamma(trial.loss)
    if gamma is None:
        return F.cross_entropy(logits, labels, weight=weights, label_smoothing=trial.label_smoothing)
    ce = F.cross_entropy(logits, labels, reduction="none", label_smoothing=trial.label_smoothing)
    pt = torch.exp(-ce)
    return (((1.0 - pt) ** gamma) * ce).mean()


def optimizer_for(model: torch.nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(token in name for token in no_decay):
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    return torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=lr,
        foreach=False,
        fused=False,
    )


def metric_bundle(y_true: np.ndarray, probs: np.ndarray) -> dict[str, Any]:
    if not np.isfinite(probs).all():
        raise AssertionError("Non-finite probabilities")
    if not np.allclose(probs.sum(axis=1), 1.0, atol=1e-4):
        raise AssertionError("Probability sums are not close to 1")
    pred = probs.argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, pred, labels=list(range(len(LABELS))), zero_division=0
    )
    true_dist = np.bincount(y_true, minlength=len(LABELS)) / len(y_true)
    pred_dist = np.bincount(pred, minlength=len(LABELS)) / len(pred)
    mcc = float(matthews_corrcoef(y_true, pred))
    macro = float(f1_score(y_true, pred, average="macro", labels=list(range(len(LABELS))), zero_division=0))
    bal = float(balanced_accuracy_score(y_true, pred))
    acc = float(accuracy_score(y_true, pred))
    min_recall = float(recall.min())
    pos_recall = float(recall[LABEL_TO_ID["Positive"]])
    negative_drift = abs(float(pred_dist[LABEL_TO_ID["Negative"]] - true_dist[LABEL_TO_ID["Negative"]]))
    class_collapse = bool((pred_dist > 0).sum() < len(LABELS))
    penalty = 0.0
    if class_collapse:
        penalty += 1.0
    if negative_drift > 0.08:
        penalty += (negative_drift - 0.08) * 0.50
    selection = 0.25 * macro + 0.20 * bal + 0.20 * mcc + 0.15 * min_recall + 0.15 * pos_recall + 0.05 * acc - penalty
    one_hot = np.eye(len(LABELS))[y_true]
    metrics: dict[str, Any] = {
        "n": int(len(y_true)),
        "accuracy": acc,
        "macro_f1": macro,
        "weighted_f1": float(f1_score(y_true, pred, average="weighted", labels=list(range(len(LABELS))), zero_division=0)),
        "balanced_accuracy": bal,
        "mcc": mcc,
        "min_class_recall": min_recall,
        "positive_recall": pos_recall,
        "selection_score_raw": float(0.25 * macro + 0.20 * bal + 0.20 * mcc + 0.15 * min_recall + 0.15 * pos_recall + 0.05 * acc),
        "selection_score": float(selection),
        "class_collapse": class_collapse,
        "auto_disqualified": bool(class_collapse or min_recall < 0.55 or pos_recall < 0.60),
        "negative_prediction_share": float(pred_dist[LABEL_TO_ID["Negative"]]),
        "negative_prediction_drift": negative_drift,
        "predicted_negative": int((pred == LABEL_TO_ID["Negative"]).sum()),
        "predicted_neutral": int((pred == LABEL_TO_ID["Neutral"]).sum()),
        "predicted_positive": int((pred == LABEL_TO_ID["Positive"]).sum()),
        "ece": expected_calibration_error(y_true, probs),
        "brier_score": float(np.mean([brier_score_loss(one_hot[:, i], probs[:, i]) for i in range(len(LABELS))])),
        "mean_confidence": float(probs.max(axis=1).mean()),
    }
    for label, p, r, f, s in zip(LABELS, precision, recall, f1, support):
        key = label.lower()
        metrics[f"{key}_precision"] = float(p)
        metrics[f"{key}_recall"] = float(r)
        metrics[f"{key}_f1"] = float(f)
        metrics[f"{key}_support"] = int(s)
    return metrics


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


def confusion_payload(y_true: np.ndarray, probs: np.ndarray) -> dict[str, list[int]]:
    cm = confusion_matrix(y_true, probs.argmax(axis=1), labels=list(range(len(LABELS))))
    return {label: [int(x) for x in row] for label, row in zip(LABELS, cm)}


def predict_probabilities(
    model: torch.nn.Module,
    tokenizer,
    texts: list[str],
    max_length: int,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    loader = DataLoader(
        TextDataset(texts),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_batch(tokenizer, max_length),
    )
    parts = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            logits = model(**batch).logits
            parts.append(torch.softmax(logits.float(), dim=-1).cpu().numpy())
    return np.concatenate(parts, axis=0)


def train_one_fold(
    trial: Trial,
    data: pd.DataFrame,
    fold_assignments: pd.DataFrame,
    seed: int,
    fold: int,
    stage: str,
    max_epochs: int,
    patience: int,
    precision: str,
    device: torch.device,
    output_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    set_seed(seed)
    val_idx = fold_assignments.loc[fold_assignments["fold"].astype(int).eq(int(fold)), "row_index"].astype(int).to_numpy()
    train_idx = np.array(sorted(set(range(len(data))) - set(val_idx.tolist())), dtype=int)
    train = data.iloc[train_idx].copy()
    val = data.iloc[val_idx].copy()
    train_labels = train["label_id"].to_numpy(dtype=int)
    val_labels = val["label_id"].to_numpy(dtype=int)
    class_weights = class_weights_from_training(train_labels)
    tokenizer = AutoTokenizer.from_pretrained(trial.model_id, trust_remote_code=False)
    config = AutoConfig.from_pretrained(
        trial.model_id,
        num_labels=len(LABELS),
        id2label={str(i): label for i, label in ID_TO_LABEL.items()},
        label2id=LABEL_TO_ID,
        classifier_dropout=trial.classifier_dropout,
        hidden_dropout_prob=trial.classifier_dropout,
        attention_probs_dropout_prob=trial.classifier_dropout,
        trust_remote_code=False,
    )
    model_revision = str(getattr(config, "_commit_hash", "") or "")
    model = AutoModelForSequenceClassification.from_pretrained(trial.model_id, config=config, trust_remote_code=False)
    model.to(device)
    batch_size = 24 if trial.max_length <= 128 else 16
    train_loader = DataLoader(
        TextDataset([model_input(row, trial.input_mode) for _, row in train.iterrows()], train_labels),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_batch(tokenizer, trial.max_length),
    )
    optimizer = optimizer_for(model, trial.learning_rate, trial.weight_decay)
    total_steps = max(1, len(train_loader) * max_epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * trial.warmup_ratio),
        num_training_steps=total_steps,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=(precision == "fp16"))
    autocast_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    best_state: dict[str, torch.Tensor] | None = None
    best_score = -1e9
    best_epoch = 0
    bad_epochs = 0
    start = time.time()
    for epoch in range(1, max_epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            labels = batch.pop("labels").to(device)
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            context = torch.autocast("cuda", dtype=autocast_dtype) if device.type == "cuda" and precision in {"bf16", "fp16"} else nullcontext()
            with context:
                logits = model(**batch).logits
                loss = loss_for_logits(logits, labels, trial, class_weights)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            losses.append(float(loss.detach().cpu()))
        val_probs = predict_probabilities(
            model,
            tokenizer,
            [model_input(row, trial.input_mode) for _, row in val.iterrows()],
            trial.max_length,
            batch_size,
            device,
        )
        val_metrics = metric_bundle(val_labels, val_probs)
        score = float(val_metrics["selection_score"])
        log_event(
            output_dir,
            {
                "event": "epoch_complete",
                "stage": stage,
                "trial_id": trial.trial_id,
                "seed": seed,
                "fold": fold,
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "validation_selection_score": score,
                "validation_macro_f1": val_metrics["macro_f1"],
            },
        )
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    val_probs = predict_probabilities(
        model,
        tokenizer,
        [model_input(row, trial.input_mode) for _, row in val.iterrows()],
        trial.max_length,
        batch_size,
        device,
    )
    metrics = {
        **asdict(trial),
        "trial_id": trial.trial_id,
        "stage": stage,
        "seed": int(seed),
        "fold": str(fold),
        "best_epoch": int(best_epoch),
        "max_epochs": int(max_epochs),
        "early_stopping_patience": int(patience),
        "class_weights_source": "training_fold_only",
        "class_weights_negative": float(class_weights[0]),
        "class_weights_neutral": float(class_weights[1]),
        "class_weights_positive": float(class_weights[2]),
        "model_revision": model_revision,
        "training_seconds": float(time.time() - start),
        **metric_bundle(val_labels, val_probs),
    }
    pred_ids = val_probs.argmax(axis=1)
    rows: list[dict[str, Any]] = []
    for i, (_, row) in enumerate(val.iterrows()):
        rows.append(
            {
                "annotation_id": row["annotation_id"],
                "comment_id": row["comment_id"],
                "true_label": row["final_human_label"],
                "trial_id": trial.trial_id,
                "seed": int(seed),
                "fold": str(fold),
                "stage": stage,
                "prob_negative": float(val_probs[i, 0]),
                "prob_neutral": float(val_probs[i, 1]),
                "prob_positive": float(val_probs[i, 2]),
                "predicted_label": ID_TO_LABEL[int(pred_ids[i])],
                "confidence": float(val_probs[i].max()),
                "best_epoch": int(best_epoch),
                "model_revision": model_revision,
                "preprocessing_version": f"rm2_sentiment_v5_{trial.input_mode}_v1",
            }
        )
    del model, tokenizer, optimizer, scheduler
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return rows, metrics


def all_oof_metric(
    oof: pd.DataFrame,
    trial: Trial,
    seed: int,
    stage: str,
    n_expected: int,
    base_row: dict[str, Any],
) -> dict[str, Any]:
    subset = oof.loc[oof["stage"].eq(stage) & oof["trial_id"].eq(trial.trial_id) & oof["seed"].astype(int).eq(seed)].copy()
    if len(subset) != n_expected or subset["annotation_id"].nunique() != n_expected:
        raise AssertionError(f"Incomplete OOF for {trial.trial_id} seed {seed} stage {stage}: {len(subset)}")
    y = subset["true_label"].map(LABEL_TO_ID).to_numpy(dtype=int)
    probs = subset[["prob_negative", "prob_neutral", "prob_positive"]].astype(float).to_numpy()
    return {
        **base_row,
        "stage": stage,
        "trial_id": trial.trial_id,
        "seed": int(seed),
        "fold": "ALL_OOF",
        "best_epoch": int(pd.to_numeric(subset["best_epoch"]).median()),
        **metric_bundle(y, probs),
    }


def run_trials(
    trials: list[Trial],
    stage: str,
    seeds: list[int],
    n_splits: int,
    max_epochs: int,
    patience: int,
    precision: str,
    device: torch.device,
    output_dir: Path,
    resume: bool,
) -> None:
    _, data = read_development_registry()
    metric_path = output_dir / "development_fold_seed_metrics.csv"
    oof_path = output_dir / "development_oof_predictions.csv"
    done = completed_metric_keys(metric_path) if resume else set()
    plan = pd.DataFrame([{**asdict(trial), "trial_id": trial.trial_id, "stage": stage} for trial in trials])
    plan_path = output_dir / ("screening_trial_plan.csv" if stage == "stage1" else "stage2_trial_plan.csv")
    plan.to_csv(plan_path, index=False, encoding="utf-8-sig")
    for seed in seeds:
        folds = load_folds(data, seed, n_splits, stage, output_dir)
        validate_fold_leakage(folds)
        for trial in trials:
            for fold in range(1, n_splits + 1):
                key = (stage, trial.trial_id, str(seed), str(fold))
                if key in done:
                    log_event(output_dir, {"event": "fold_skipped_resume", "stage": stage, "trial_id": trial.trial_id, "seed": seed, "fold": fold})
                    continue
                rows, metrics = train_one_fold(
                    trial, data, folds, seed, fold, stage, max_epochs, patience, precision, device, output_dir
                )
                upsert_csv(oof_path, rows, OOF_KEYS)
                upsert_csv(metric_path, [metrics], METRIC_KEYS)
                log_event(output_dir, {"event": "fold_saved", "stage": stage, "trial_id": trial.trial_id, "seed": seed, "fold": fold})
            oof = pd.read_csv(oof_path, dtype=str, keep_default_na=False, low_memory=False)
            seed_fold_metrics = pd.read_csv(metric_path, dtype=str, keep_default_na=False, low_memory=False)
            base = seed_fold_metrics.loc[
                seed_fold_metrics["stage"].eq(stage)
                & seed_fold_metrics["trial_id"].eq(trial.trial_id)
                & seed_fold_metrics["seed"].astype(int).eq(seed)
                & ~seed_fold_metrics["fold"].eq("ALL_OOF")
            ].iloc[-1].to_dict()
            upsert_csv(metric_path, [all_oof_metric(oof, trial, seed, stage, len(data), base)], METRIC_KEYS)
    summarize_completed(output_dir)


def summarize_completed(output_dir: Path) -> None:
    path = output_dir / "development_fold_seed_metrics.csv"
    if not path.exists():
        return
    metrics = pd.read_csv(path, dtype=str, keep_default_na=False, low_memory=False)
    all_oof = metrics.loc[metrics["fold"].eq("ALL_OOF")].copy()
    if all_oof.empty:
        return
    numeric_cols = [
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "balanced_accuracy",
        "mcc",
        "min_class_recall",
        "positive_recall",
        "selection_score",
        "negative_prediction_share",
    ]
    for col in numeric_cols:
        all_oof[col] = pd.to_numeric(all_oof[col], errors="coerce")
    seed_summary = all_oof.sort_values(["stage", "selection_score"], ascending=[True, False])
    seed_summary.to_csv(output_dir / "development_trial_seed_summary.csv", index=False, encoding="utf-8-sig")
    rows = []
    for (stage, trial_id), group in all_oof.groupby(["stage", "trial_id"]):
        row = {"stage": stage, "trial_id": trial_id, "n_seeds": int(group["seed"].nunique())}
        first = group.iloc[0].to_dict()
        for col in ["input_mode", "max_length", "learning_rate", "classifier_dropout", "loss", "label_smoothing", "model_id"]:
            row[col] = first.get(col, "")
        for col in numeric_cols:
            row[f"mean_{col}"] = float(group[col].mean())
            row[f"std_{col}"] = float(group[col].std(ddof=0))
            row[f"min_{col}"] = float(group[col].min())
            row[f"max_{col}"] = float(group[col].max())
        row["auto_disqualified_any_seed"] = bool(group["auto_disqualified"].astype(str).str.lower().eq("true").any())
        row["selected_for_stage2"] = False
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values(["stage", "mean_selection_score"], ascending=[True, False])
    if not summary.empty:
        stage1 = summary.loc[summary["stage"].eq("stage1")].copy()
        keep = stage1.loc[~stage1["auto_disqualified_any_seed"].astype(bool)].head(5)["trial_id"].tolist()
        summary["selected_for_stage2"] = summary["trial_id"].isin(keep)
    summary.to_csv(output_dir / "development_trial_summary.csv", index=False, encoding="utf-8-sig")


def audit_inputs(output_dir: Path) -> dict[str, Any]:
    raw, data = read_development_registry()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=False)
    coverage = []
    for column in ["comment_text", "brand_or_video_context", "product_category", "parent_comment_text"]:
        non_empty = raw[column].map(nonempty).ne("")
        coverage.append({"column": column, "non_empty_rows": int(non_empty.sum()), "non_empty_share": float(non_empty.mean())})
    coverage_frame = pd.DataFrame(coverage)
    coverage_frame.to_csv(output_dir / "input_mode_coverage_audit.csv", index=False, encoding="utf-8-sig")
    parent_share = float(coverage_frame.loc[coverage_frame["column"].eq("parent_comment_text"), "non_empty_share"].iloc[0])
    modes = ["comment_only", "context_sep_comment"]
    if parent_share >= 0.10:
        modes.append("target_context_parent_comment")
    token_audit: dict[str, Any] = {
        "created_at_utc": utc_now(),
        "model_id": MODEL_ID,
        "development_registry": rel(DEV_REGISTRY),
        "development_dataset_hash": sha256_dataframe(raw, ["annotation_id", "comment_id", "final_human_label"]),
        "parent_comment_text_non_empty_share": parent_share,
        "input_modes_allowed": modes,
        "input_modes_skipped": [] if parent_share >= 0.10 else ["target_context_parent_comment"],
        "lengths_by_mode": {},
    }
    evaluate_256 = False
    for mode in modes:
        texts = [model_input(row, mode) for _, row in data.iterrows()]
        lengths = [len(tokenizer(text, add_special_tokens=True)["input_ids"]) for text in texts]
        stats = {
            "median": float(np.median(lengths)),
            "p90": float(np.quantile(lengths, 0.90)),
            "p95": float(np.quantile(lengths, 0.95)),
            "p99": float(np.quantile(lengths, 0.99)),
            "truncated_share_128": float(np.mean(np.asarray(lengths) > 128)),
            "truncated_share_256": float(np.mean(np.asarray(lengths) > 256)),
        }
        if stats["truncated_share_128"] > 0.05:
            evaluate_256 = True
        token_audit["lengths_by_mode"][mode] = stats
    token_audit["max_lengths_allowed"] = [128, 256] if evaluate_256 else [128]
    save_json(output_dir / "input_token_length_audit.json", token_audit)
    return token_audit


def curated_screening_trials(token_audit: dict[str, Any], budget: int) -> list[Trial]:
    modes = token_audit["input_modes_allowed"]
    lengths = token_audit["max_lengths_allowed"]
    base_cases = [
        (1e-5, 0.1, "cross_entropy", 0.0),
        (2e-5, 0.2, "cross_entropy", 0.025),
        (3e-5, 0.3, "cross_entropy", 0.0),
        (1e-5, 0.2, "weighted_cross_entropy", 0.0),
        (2e-5, 0.3, "weighted_cross_entropy", 0.025),
        (3e-5, 0.1, "weighted_cross_entropy", 0.0),
        (1e-5, 0.3, "focal_loss_gamma_1.0", 0.025),
        (2e-5, 0.1, "focal_loss_gamma_1.0", 0.0),
        (3e-5, 0.2, "focal_loss_gamma_1.5", 0.025),
        (2e-5, 0.2, "focal_loss_gamma_1.5", 0.0),
        (3e-5, 0.3, "focal_loss_gamma_2.0", 0.025),
        (2e-5, 0.1, "cross_entropy", 0.05),
    ]
    trials: list[Trial] = []
    for length in lengths:
        for mode in modes:
            for lr, dropout, loss, smoothing in base_cases:
                trial = Trial(mode, int(length), lr, dropout, loss, smoothing)
                if trial not in trials:
                    trials.append(trial)
                if len(trials) >= budget:
                    return trials
    return trials


def trials_by_id(ids: list[str], token_audit: dict[str, Any], budget: int = 24) -> list[Trial]:
    candidates = {trial.trial_id: trial for trial in curated_screening_trials(token_audit, budget)}
    missing = sorted(set(ids) - set(candidates))
    if missing:
        raise ValueError(f"Unknown trial ids: {missing}")
    return [candidates[trial_id] for trial_id in ids]


def stage2_trials_from_summary(output_dir: Path, token_audit: dict[str, Any], top_k: int) -> list[Trial]:
    leaderboard = output_dir / "development_candidate_leaderboard.csv"
    summary = output_dir / "development_trial_summary.csv"
    if leaderboard.exists():
        frame = pd.read_csv(leaderboard, dtype=str, keep_default_na=False, low_memory=False)
        frame = frame.loc[frame["stage"].eq("stage1") & frame["candidate_type"].eq("single_seed")]
        frame["selection_score"] = pd.to_numeric(frame["selection_score"], errors="coerce")
        ids = frame.sort_values("selection_score", ascending=False)["trial_id"].drop_duplicates().head(top_k).tolist()
    elif summary.exists():
        frame = pd.read_csv(summary, dtype=str, keep_default_na=False, low_memory=False)
        frame = frame.loc[frame["stage"].eq("stage1")].copy()
        frame["mean_selection_score"] = pd.to_numeric(frame["mean_selection_score"], errors="coerce")
        ids = frame.sort_values("mean_selection_score", ascending=False)["trial_id"].head(top_k).tolist()
    else:
        raise FileNotFoundError("Run stage1 and selection before stage2, or pass --trial-ids")
    return trials_by_id(ids, token_audit, budget=max(24, top_k))


def train_full_model(
    trial: Trial,
    data: pd.DataFrame,
    seed: int,
    epochs: int,
    precision: str,
    device: torch.device,
    model_dir: Path,
) -> dict[str, Any]:
    set_seed(seed)
    model_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(trial.model_id, trust_remote_code=False)
    config = AutoConfig.from_pretrained(
        trial.model_id,
        num_labels=len(LABELS),
        id2label={str(i): label for i, label in ID_TO_LABEL.items()},
        label2id=LABEL_TO_ID,
        classifier_dropout=trial.classifier_dropout,
        hidden_dropout_prob=trial.classifier_dropout,
        attention_probs_dropout_prob=trial.classifier_dropout,
        trust_remote_code=False,
    )
    model_revision = str(getattr(config, "_commit_hash", "") or "")
    model = AutoModelForSequenceClassification.from_pretrained(trial.model_id, config=config, trust_remote_code=False)
    model.to(device)
    labels = data["label_id"].to_numpy(dtype=int)
    class_weights = class_weights_from_training(labels)
    batch_size = 24 if trial.max_length <= 128 else 16
    loader = DataLoader(
        TextDataset([model_input(row, trial.input_mode) for _, row in data.iterrows()], labels),
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_batch(tokenizer, trial.max_length),
    )
    optimizer = optimizer_for(model, trial.learning_rate, trial.weight_decay)
    total_steps = max(1, len(loader) * epochs)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(total_steps * trial.warmup_ratio), total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=(precision == "fp16"))
    autocast_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    for _ in range(epochs):
        model.train()
        for batch in loader:
            y = batch.pop("labels").to(device)
            batch = {key: value.to(device) for key, value in batch.items()}
            optimizer.zero_grad(set_to_none=True)
            context = torch.autocast("cuda", dtype=autocast_dtype) if device.type == "cuda" and precision in {"bf16", "fp16"} else nullcontext()
            with context:
                loss = loss_for_logits(model(**batch).logits, y, trial, class_weights)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
    model.save_pretrained(model_dir, safe_serialization=True)
    tokenizer.save_pretrained(model_dir)
    save_json(model_dir / "label_map.json", {"label_to_id": LABEL_TO_ID, "id_to_label": {str(k): v for k, v in ID_TO_LABEL.items()}})
    save_json(
        model_dir / "selected_trial_config.json",
        {
            **asdict(trial),
            "trial_id": trial.trial_id,
            "seed": int(seed),
            "epochs": int(epochs),
            "model_revision": model_revision,
            "prediction_rule": "three_class_argmax",
            "label_source_column": "sentiment_toward_target/final_human_label",
            "class_weights_source": "full_development_only_after_selection",
            "forbidden_supervision_sources": FORBIDDEN_SUPERVISION_SOURCES,
        },
    )
    probs = predict_probabilities(
        model,
        tokenizer,
        [model_input(row, trial.input_mode) for _, row in data.head(3).iterrows()],
        trial.max_length,
        batch_size=3,
        device=device,
    )
    smoke = {"device": str(device), "labels": [ID_TO_LABEL[int(i)] for i in probs.argmax(axis=1)], "probability_sums": probs.sum(axis=1).tolist()}
    save_json(model_dir / "gpu_smoke_test.json", smoke)
    model.to(torch.device("cpu"))
    cpu_probs = predict_probabilities(
        model,
        tokenizer,
        [model_input(row, trial.input_mode) for _, row in data.head(3).iterrows()],
        trial.max_length,
        batch_size=3,
        device=torch.device("cpu"),
    )
    cpu_smoke = {
        "device": "cpu",
        "labels": [ID_TO_LABEL[int(i)] for i in cpu_probs.argmax(axis=1)],
        "probability_sums": cpu_probs.sum(axis=1).tolist(),
    }
    save_json(model_dir / "cpu_smoke_test.json", cpu_smoke)
    hashes = write_sha256s(model_dir)
    del model, tokenizer, optimizer, scheduler
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"model_revision": model_revision, "hashes": hashes}


def freeze_candidate(output_dir: Path, precision: str, device: torch.device) -> None:
    _, data = read_development_registry()
    selected_path = output_dir / "development_selected_candidate.json"
    if not selected_path.exists():
        raise FileNotFoundError("Run selection before freeze")
    selected = read_json(selected_path)
    component_trials = selected.get("component_trials") or [selected["trial"]]
    trials = [Trial(**{key: trial_payload[key] for key in Trial.__dataclass_fields__ if key in trial_payload}) for trial_payload in component_trials]
    trial = trials[0]
    component_seeds = selected["component_seeds"]
    final_epochs = int(selected["final_epoch_count"])
    if ARTIFACT_DIR.exists():
        expected = (ROOT / "artifacts/rm2_sentiment/indobert_v5_candidate").resolve()
        resolved = ARTIFACT_DIR.resolve()
        if resolved != expected:
            raise RuntimeError(f"Refusing to remove unexpected artifact directory: {resolved}")
        shutil.rmtree(resolved)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    component_dirs = []
    for trial_index, component_trial in enumerate(trials, start=1):
        for seed in component_seeds:
            if len(trials) == 1 and len(component_seeds) == 1:
                subdir = ARTIFACT_DIR / "selected_single_model"
            elif len(trials) == 1:
                subdir = ARTIFACT_DIR / f"component_seed_{int(seed)}"
            else:
                subdir = ARTIFACT_DIR / f"component_trial_{trial_index}_seed_{int(seed)}"
            info = train_full_model(component_trial, data, int(seed), final_epochs, precision, device, subdir)
            component_dirs.append({"trial_id": component_trial.trial_id, "seed": int(seed), "path": rel(subdir), **info})
    label_map = {"label_to_id": LABEL_TO_ID, "id_to_label": {str(k): v for k, v in ID_TO_LABEL.items()}}
    save_json(ARTIFACT_DIR / "label_map.json", label_map)
    save_json(
        ARTIFACT_DIR / "preprocessing_config.json",
        {"preprocessing_version": f"rm2_sentiment_v5_{trial.input_mode}_v1", "input_mode": trial.input_mode, "max_length": trial.max_length},
    )
    shutil.copy2(selected_path, ARTIFACT_DIR / "development_selected_candidate.json")
    acceptance_hash = sha256_file(ACCEPTANCE_CONFIG)
    hashes = write_sha256s(ARTIFACT_DIR)
    manifest = {
        "status": "INDOBERT_V5_DEVELOPMENT_CANDIDATE_FROZEN_BEFORE_LOCKED_TEST",
        "created_at_utc": utc_now(),
        "git_sha_before_locked_test": git_head(),
        "development_registry": rel(DEV_REGISTRY),
        "development_dataset_hash": sha256_dataframe(pd.read_csv(DEV_REGISTRY, dtype=str, keep_default_na=False), ["annotation_id", "comment_id", "final_human_label"]),
        "development_evaluable_rows": int(len(data)),
        "development_class_distribution": class_distribution(data),
        "selected_candidate": selected,
        "exact_trial_id": trial.trial_id,
        "component_trial_ids": [component_trial.trial_id for component_trial in trials],
        "component_models": component_dirs,
        "prediction_rule": selected["prediction_rule"],
        "abstention_policy_development_only": selected.get("abstention_policy", []),
        "label_map": label_map,
        "acceptance_config": rel(ACCEPTANCE_CONFIG),
        "acceptance_config_sha256": acceptance_hash,
        "locked_test_used_for_training_or_selection": False,
        "locked_test_evaluated": False,
        "package_versions": package_versions(),
        "device_info": device_info(precision),
        "artifact_hashes": hashes,
    }
    save_json(ARTIFACT_DIR / "DEVELOPMENT_MODEL_FREEZE_MANIFEST.json", manifest)
    (ARTIFACT_DIR / "MODEL_CARD.md").write_text(
        "\n".join(
            [
                "# RM2 Sentiment IndoBERT V5 Candidate",
                "",
                "Status: frozen from V5 development data before locked-test evaluation.",
                f"Model: `{MODEL_ID}`",
                f"Trial: `{trial.trial_id}`",
                f"Prediction rule: `{selected['prediction_rule']}`",
                "Primary output is full three-class argmax. Abstention is secondary and selected from development OOF only.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    write_sha256s(ARTIFACT_DIR)
    print(json.dumps({"status": "frozen", "artifact_dir": rel(ARTIFACT_DIR), "component_count": len(component_dirs)}, indent=2), flush=True)


def parse_precision(value: str) -> str:
    if value == "auto":
        return "bf16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "fp16"
    if value not in {"bf16", "fp16"}:
        raise ValueError("--precision must be auto, bf16, or fp16")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Train efficient GPU IndoBERT V5 candidates on development data only.")
    parser.add_argument("--stage", choices=["audit", "stage1", "stage2", "freeze"], required=True)
    parser.add_argument("--trial-ids", nargs="*", default=[])
    parser.add_argument("--seeds", nargs="*", type=int, default=[])
    parser.add_argument("--n-splits", type=int, default=0)
    parser.add_argument("--max-epochs", type=int, default=0)
    parser.add_argument("--patience", type=int, default=0)
    parser.add_argument("--precision", default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--screening-budget", type=int, default=18)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output-dir", default=str(OUT_DIR_DEFAULT))
    args = parser.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for IndoBERT V5 training; refusing CPU training.")
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    precision = parse_precision(args.precision)
    device = torch.device("cuda")
    if args.stage == "audit":
        payload = audit_inputs(output_dir)
        save_json(
            output_dir / "INDOBERT_V5_DEVELOPMENT_PIPELINE_MANIFEST.json",
            {
                "status": "INDOBERT_V5_DEVELOPMENT_FOLDS_READY",
                "gpu_training_status": "INDOBERT_V5_DEVELOPMENT_GPU_TRAINING_READY",
                "created_at_utc": utc_now(),
                "git_sha": git_head(),
                "development_registry": rel(DEV_REGISTRY),
                "development_dataset_hash": payload["development_dataset_hash"],
                "development_evaluable_rows": 977,
                "primary_label": "sentiment_toward_target",
                "label_order": LABELS,
                "locked_test_v4_errors_used": False,
                "locked_test_v5_labels_used_for_training_or_selection": False,
                "validation": {
                    "development_rows": 1000,
                    "development_final_registry": rel(DEV_REGISTRY),
                    "development_labels_available": 977,
                    "development_final_distribution": {"Negative": 178, "Neutral": 569, "Positive": 230},
                    "locked_labels_available_for_training": 0,
                    "locked_test_v5_used_for_training": False,
                },
                "outputs": {
                    "candidate_grid": rel(output_dir / "candidate_grid_manifest.csv"),
                    "risk_coverage_policy": rel(output_dir / "development_risk_coverage_policy.csv"),
                    "development_fold_assignments": rel(output_dir / "development_grouped_fold_assignments.csv"),
                },
                "training_script_reads_locked_test_registry": False,
                "candidate_trial_count": 810,
                "curated_screening_budget": int(args.screening_budget),
                "device_info": device_info(precision),
                "package_versions": package_versions(),
                "input_token_length_audit": rel(output_dir / "input_token_length_audit.json"),
            },
        )
        print(json.dumps({"status": "audit_complete", "allowed_lengths": payload["max_lengths_allowed"], "allowed_modes": payload["input_modes_allowed"]}, indent=2))
        return
    token_audit_path = output_dir / "input_token_length_audit.json"
    token_audit = read_json(token_audit_path) if token_audit_path.exists() else audit_inputs(output_dir)
    if args.stage == "stage1":
        trials = trials_by_id(args.trial_ids, token_audit, args.screening_budget) if args.trial_ids else curated_screening_trials(token_audit, args.screening_budget)
        run_trials(
            trials,
            "stage1",
            args.seeds or [42],
            args.n_splits or 3,
            args.max_epochs or 8,
            args.patience or 2,
            precision,
            device,
            output_dir,
            args.resume,
        )
        return
    if args.stage == "stage2":
        trials = trials_by_id(args.trial_ids, token_audit, max(args.screening_budget, len(args.trial_ids))) if args.trial_ids else stage2_trials_from_summary(output_dir, token_audit, args.top_k)
        run_trials(
            trials,
            "stage2",
            args.seeds or [42, 52, 62],
            args.n_splits or 5,
            args.max_epochs or 12,
            args.patience or 3,
            precision,
            device,
            output_dir,
            args.resume,
        )
        return
    freeze_candidate(output_dir, precision, device)


if __name__ == "__main__":
    main()
