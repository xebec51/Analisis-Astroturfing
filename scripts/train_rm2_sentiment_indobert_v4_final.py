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
import sys
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
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedGroupKFold
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
DEV_REGISTRY = ROOT / "output/rm2_sentiment/validation/human_master_v4/sentiment_v4_development_final_registry.csv"
LOCKED_REGISTRY = ROOT / "output/rm2_sentiment/validation/human_master_v4/sentiment_v4_locked_test_final_frozen.csv"
OUT_DIR = ROOT / "output/rm2_sentiment/experiments/indobert_v4_final"
MODEL_DIR = ROOT / "output/rm2_sentiment/model/indobert_v4_final_candidate"

LABELS = ["Negative", "Neutral", "Positive"]
LABEL_TO_ID = {label: idx for idx, label in enumerate(LABELS)}
ID_TO_LABEL = {idx: label for label, idx in LABEL_TO_ID.items()}
OOF_KEYS = ["trial_id", "seed", "fold", "comment_id"]
METRIC_KEYS = ["trial_id", "seed", "fold"]
FORBIDDEN_LABEL_SOURCES = [
    "sentiment_v2_prediction",
    "model_prediction",
    "lexicon",
    "hcc",
    "actor_type",
    "goal_orientation",
    "llm",
    "locked_test",
]


@dataclass(frozen=True)
class Trial:
    model_id: str
    model_family: str
    text_mode: str
    max_length: int
    learning_rate: float
    warmup_ratio: float
    weight_decay: float
    classifier_dropout: float
    loss: str
    label_smoothing: float
    focal_gamma: float = 2.0

    @property
    def trial_id(self) -> str:
        lr = f"{self.learning_rate:.1e}".replace("-", "m").replace(".", "p")
        wr = str(self.warmup_ratio).replace(".", "p")
        wd = str(self.weight_decay).replace(".", "p")
        dr = str(self.classifier_dropout).replace(".", "p")
        ls = str(self.label_smoothing).replace(".", "p")
        safe_model = safe_model_name(self.model_id)
        return (
            f"{safe_model}__{self.text_mode}__len{self.max_length}__lr{lr}"
            f"__warm{wr}__wd{wd}__drop{dr}__{self.loss}__ls{ls}"
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


class DuplicateGroupBuilder:
    def __init__(self) -> None:
        self.parent: dict[str, str] = {}

    def find(self, value: str) -> str:
        if value not in self.parent:
            self.parent[value] = value
            return value
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: str, right: str) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left != root_right:
            self.parent[root_right] = root_left


def safe_model_name(model_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", model_id)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2), encoding="utf-8")


def log_event(payload: dict[str, Any]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    row = json.dumps(to_jsonable({"created_at_utc": utc_now(), **payload}), ensure_ascii=False)
    with (OUT_DIR / "training_events.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(row + "\n")
    print(json.dumps(to_jsonable(payload)), flush=True)


def upsert_rows_csv(path: Path, rows: list[dict[str, Any]], subset: list[str]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    incoming = pd.DataFrame(rows)
    if path.exists() and path.stat().st_size > 0:
        existing = pd.read_csv(path, dtype=str, keep_default_na=False, low_memory=False)
        combined = pd.concat([existing, incoming], ignore_index=True, sort=False)
    else:
        combined = incoming
    available_subset = [col for col in subset if col in combined.columns]
    if available_subset:
        combined = combined.drop_duplicates(subset=available_subset, keep="last")
    combined.to_csv(path, index=False, encoding="utf-8-sig")


def dedupe_frame(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    if frame.empty:
        return frame
    missing = sorted(set(keys) - set(frame.columns))
    if missing:
        raise ValueError(f"Cannot de-duplicate frame missing keys: {missing}")
    return frame.drop_duplicates(keys, keep="last").reset_index(drop=True)


def dedupe_records(records: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    if not records:
        return records
    return dedupe_frame(pd.DataFrame(records), keys).to_dict("records")


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [to_jsonable(v) for v in value]
    if isinstance(value, Path):
        return value.relative_to(ROOT).as_posix() if value.is_absolute() and ROOT in value.parents else str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if pd.isna(value) if isinstance(value, float) else False:
        return None
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_dataframe(frame: pd.DataFrame, columns: list[str]) -> str:
    data = frame[columns].sort_values(columns).to_csv(index=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def package_versions() -> dict[str, str]:
    versions = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": str(torch.version.cuda),
        "cuda_available": str(torch.cuda.is_available()),
    }
    for package in ["transformers", "sklearn", "pandas", "numpy", "safetensors", "accelerate"]:
        try:
            module = __import__(package)
            versions[package] = str(getattr(module, "__version__", "unknown"))
        except Exception as exc:  # pragma: no cover
            versions[package] = f"unavailable:{exc}"
    return versions


def device_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "cuda_available": bool(torch.cuda.is_available()),
        "mixed_precision": "none",
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        bf16 = bool(torch.cuda.is_bf16_supported())
        info.update(
            {
                "cuda_device_name": torch.cuda.get_device_name(0),
                "cuda_total_memory_bytes": int(props.total_memory),
                "bf16_supported": bf16,
                "mixed_precision": "bf16" if bf16 else "fp16",
            }
        )
    return info


def is_true(value: object) -> bool:
    return str(value).strip().lower() == "true"


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


def read_registry(path: Path, expected_evaluable: int | None = None) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, dtype=str, keep_default_na=False, low_memory=False)
    required = {
        "comment_id",
        "comment_text",
        "model_text",
        "product_category",
        "brand_or_video_context",
        "final_human_label",
        "evaluable_three_class",
        "text_cluster_id",
        "exact_duplicate_group_id",
        "near_duplicate_cluster_id",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    mask = frame["evaluable_three_class"].map(is_true) & frame["final_human_label"].isin(LABELS)
    data = frame.loc[mask].copy().reset_index(drop=True)
    if expected_evaluable is not None and len(data) != expected_evaluable:
        raise AssertionError(f"{path.name} evaluable rows={len(data)} expected={expected_evaluable}")
    data["label_id"] = data["final_human_label"].map(LABEL_TO_ID).astype(int)
    data["cv_group_id"] = build_cv_group_ids(data)
    return data


def read_development_data() -> pd.DataFrame:
    return read_registry(DEV_REGISTRY, expected_evaluable=1824)


def read_locked_test_data() -> pd.DataFrame:
    return read_registry(LOCKED_REGISTRY, expected_evaluable=672)


def build_cv_group_ids(frame: pd.DataFrame) -> list[str]:
    builder = DuplicateGroupBuilder()
    row_tokens: list[list[str]] = []
    group_cols = ["near_duplicate_cluster_id", "exact_duplicate_group_id", "text_cluster_id"]
    for _, row in frame.iterrows():
        tokens = [f"{col}:{nonempty(row.get(col))}" for col in group_cols if nonempty(row.get(col))]
        if not tokens:
            tokens = [f"comment_id:{nonempty(row.get('comment_id'))}"]
        first = tokens[0]
        for token in tokens:
            builder.union(first, token)
        row_tokens.append(tokens)
    return [builder.find(tokens[0]) for tokens in row_tokens]


def comment_text(row: pd.Series) -> str:
    return normalize_text(row.get("model_text")) or normalize_text(row.get("comment_text"))


def model_input(row: pd.Series, text_mode: str) -> str:
    comment = comment_text(row)
    if text_mode == "comment_only":
        return comment
    context_parts = [
        normalize_text(row.get("brand_or_video_context")),
        normalize_text(row.get("product_category")),
    ]
    context = " | ".join(dict.fromkeys(part for part in context_parts if part))
    return f"{context} [SEP] {comment}" if context else comment


def make_fold_assignments(data: pd.DataFrame, seeds: list[int], n_splits: int) -> pd.DataFrame:
    y = data["label_id"].to_numpy(dtype=int)
    groups = data["cv_group_id"].to_numpy()
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for fold_idx, (_, val_idx) in enumerate(splitter.split(data, y, groups), start=1):
            for source_idx in val_idx:
                row = data.iloc[source_idx]
                rows.append(
                    {
                        "seed": seed,
                        "fold": str(fold_idx),
                        "row_index": int(source_idx),
                        "comment_id": row["comment_id"],
                        "final_human_label": row["final_human_label"],
                        "text_cluster_id": row["text_cluster_id"],
                        "exact_duplicate_group_id": row["exact_duplicate_group_id"],
                        "near_duplicate_cluster_id": row["near_duplicate_cluster_id"],
                        "cv_group_id": row["cv_group_id"],
                    }
                )
    assignments = pd.DataFrame(rows)
    expected = len(data) * len(seeds)
    if len(assignments) != expected:
        raise AssertionError(f"Fold assignment rows={len(assignments)} expected={expected}")
    leakage = assignments.groupby(["seed", "cv_group_id"])["fold"].nunique()
    if int((leakage > 1).sum()) != 0:
        raise AssertionError("cv_group_id leakage across development folds")
    return assignments


def validate_fold_assignments(assignments: pd.DataFrame, data: pd.DataFrame, seeds: list[int]) -> pd.DataFrame:
    required = {
        "seed",
        "fold",
        "row_index",
        "comment_id",
        "final_human_label",
        "text_cluster_id",
        "exact_duplicate_group_id",
        "near_duplicate_cluster_id",
        "cv_group_id",
    }
    missing = sorted(required - set(assignments.columns))
    if missing:
        raise ValueError(f"Fold assignments missing required columns: {missing}")
    assignments = assignments.copy()
    assignments["seed"] = assignments["seed"].astype(int)
    assignments = assignments.loc[assignments["seed"].isin([int(seed) for seed in seeds])].copy()
    assignments["fold"] = assignments["fold"].astype(str)
    assignments["row_index"] = assignments["row_index"].astype(int)
    expected = len(data) * len(seeds)
    if len(assignments) != expected:
        raise AssertionError(f"Fold assignment rows={len(assignments)} expected={expected}")
    data_ids = data["comment_id"].astype(str).tolist()
    expected_ids = set(data_ids)
    for seed in seeds:
        seed_ids = assignments.loc[assignments["seed"].eq(int(seed)), "comment_id"].astype(str).tolist()
        if len(seed_ids) != len(data) or set(seed_ids) != expected_ids:
            raise AssertionError(f"Fold assignments for seed {seed} do not match development comment_ids")
    row_id_by_index = dict(enumerate(data_ids))
    bad_index = assignments.loc[
        assignments.apply(lambda row: str(row_id_by_index.get(int(row["row_index"]), "")) != str(row["comment_id"]), axis=1)
    ]
    if not bad_index.empty:
        row_index_by_id = {comment_id: idx for idx, comment_id in enumerate(data_ids)}
        assignments["row_index"] = assignments["comment_id"].astype(str).map(row_index_by_id).astype(int)
    leakage = assignments.groupby(["seed", "cv_group_id"])["fold"].nunique()
    if int((leakage > 1).sum()) != 0:
        raise AssertionError("cv_group_id leakage across development folds")
    return assignments


def load_or_make_fold_assignments(data: pd.DataFrame, seeds: list[int], n_splits: int, resume: bool) -> pd.DataFrame:
    path = OUT_DIR / "development_fold_assignments.csv"
    if resume and path.exists() and path.stat().st_size > 0:
        existing = pd.read_csv(path, dtype=str, keep_default_na=False, low_memory=False)
        return validate_fold_assignments(existing, data, seeds)
    assignments = make_fold_assignments(data, seeds, n_splits)
    assignments.to_csv(path, index=False, encoding="utf-8-sig")
    return assignments


def build_trials(models: list[str], search_profile: str) -> list[Trial]:
    large = "indobenchmark/indobert-large-p2"
    base = "indobenchmark/indobert-base-p2"
    tweet = "indolem/indobertweet-base-uncased"
    warm = "apriandito/indobert-sentiment-classifier"
    planned = [
        Trial(large, "large", "context_sep_comment", 128, 8e-6, 0.06, 0.01, 0.1, "cross_entropy", 0.0),
        Trial(large, "large", "context_sep_comment", 192, 1e-5, 0.10, 0.05, 0.2, "weighted_cross_entropy", 0.03),
        Trial(large, "large", "context_sep_comment", 256, 1.5e-5, 0.06, 0.05, 0.3, "focal_loss", 0.05),
        Trial(large, "large", "comment_only", 192, 2e-5, 0.10, 0.01, 0.1, "weighted_cross_entropy", 0.0),
        Trial(large, "large", "comment_only", 128, 1.5e-5, 0.06, 0.01, 0.2, "focal_loss", 0.03),
        Trial(large, "large", "context_sep_comment", 256, 8e-6, 0.10, 0.01, 0.2, "weighted_cross_entropy", 0.0),
        Trial(large, "large", "context_sep_comment", 192, 2e-5, 0.06, 0.05, 0.3, "cross_entropy", 0.05),
        Trial(large, "large", "comment_only", 256, 1e-5, 0.10, 0.05, 0.1, "focal_loss", 0.05),
        Trial(base, "base", "context_sep_comment", 128, 1e-5, 0.06, 0.01, 0.1, "cross_entropy", 0.0),
        Trial(base, "base", "context_sep_comment", 192, 2e-5, 0.10, 0.05, 0.2, "weighted_cross_entropy", 0.03),
        Trial(base, "base", "context_sep_comment", 256, 3e-5, 0.06, 0.05, 0.3, "focal_loss", 0.05),
        Trial(base, "base", "comment_only", 192, 2e-5, 0.10, 0.01, 0.1, "weighted_cross_entropy", 0.0),
        Trial(base, "base", "comment_only", 128, 3e-5, 0.06, 0.01, 0.2, "focal_loss", 0.03),
        Trial(base, "base", "context_sep_comment", 256, 1e-5, 0.10, 0.05, 0.1, "cross_entropy", 0.05),
        Trial(tweet, "tweet_base", "context_sep_comment", 128, 1e-5, 0.06, 0.01, 0.1, "cross_entropy", 0.0),
        Trial(tweet, "tweet_base", "context_sep_comment", 192, 2e-5, 0.10, 0.05, 0.2, "weighted_cross_entropy", 0.03),
        Trial(tweet, "tweet_base", "comment_only", 256, 3e-5, 0.06, 0.05, 0.3, "focal_loss", 0.05),
        Trial(tweet, "tweet_base", "comment_only", 128, 2e-5, 0.10, 0.01, 0.2, "weighted_cross_entropy", 0.0),
        Trial(warm, "warm_start", "context_sep_comment", 128, 1e-5, 0.06, 0.01, 0.1, "cross_entropy", 0.0),
        Trial(warm, "warm_start", "context_sep_comment", 192, 2e-5, 0.10, 0.05, 0.2, "weighted_cross_entropy", 0.03),
        Trial(warm, "warm_start", "comment_only", 256, 3e-5, 0.06, 0.05, 0.3, "focal_loss", 0.05),
    ]
    allowed = set(models)
    planned = [trial for trial in planned if trial.model_id in allowed]
    if search_profile == "quick":
        keep = {large, base, tweet}
        planned = [trial for trial in planned if trial.model_id in keep][:6]
    return planned


def batch_plan(trial: Trial, device: torch.device) -> tuple[int, int]:
    if device.type != "cuda":
        return 2, 16
    if trial.model_family == "large":
        per_device = 2 if trial.max_length <= 192 else 1
        effective = 48
    else:
        per_device = 12 if trial.max_length <= 128 else 8 if trial.max_length <= 192 else 6
        effective = 64
    grad_accum = max(1, math.ceil(effective / per_device))
    return per_device, grad_accum


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def collate_batch(tokenizer: Any, max_length: int):
    def _collate(items: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        encoded = tokenizer(
            [item["text"] for item in items],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        if "label" in items[0]:
            encoded["labels"] = torch.tensor([item["label"] for item in items], dtype=torch.long)
        return encoded

    return _collate


def class_weights(y: np.ndarray, device: torch.device) -> torch.Tensor:
    counts = np.bincount(y, minlength=len(LABELS)).astype("float64")
    weights = len(y) / np.clip(len(LABELS) * counts, 1e-9, None)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def compute_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    trial: Trial,
    weights: torch.Tensor | None,
) -> torch.Tensor:
    loss_weights = weights if trial.loss in {"weighted_cross_entropy", "focal_loss"} else None
    ce = F.cross_entropy(
        logits,
        targets,
        weight=loss_weights,
        reduction="none",
        label_smoothing=float(trial.label_smoothing),
    )
    if trial.loss != "focal_loss":
        return ce.mean()
    probs = torch.softmax(logits, dim=1)
    pt = probs.gather(1, targets.view(-1, 1)).squeeze(1).clamp(1e-6, 1.0)
    return (((1.0 - pt) ** float(trial.focal_gamma)) * ce).mean()


def autocast_context(device: torch.device, precision: str):
    if device.type != "cuda" or precision == "none":
        return nullcontext()
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def make_scaler(precision: str):
    enabled = precision == "fp16" and torch.cuda.is_available()
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except TypeError:  # pragma: no cover
        return torch.cuda.amp.GradScaler(enabled=enabled)


def load_model_and_tokenizer(trial: Trial, device: torch.device) -> tuple[Any, Any, str]:
    tokenizer = AutoTokenizer.from_pretrained(trial.model_id, trust_remote_code=False)
    config = AutoConfig.from_pretrained(
        trial.model_id,
        num_labels=len(LABELS),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
        trust_remote_code=False,
    )
    config.problem_type = "single_label_classification"
    config.classifier_dropout = float(trial.classifier_dropout)
    if hasattr(config, "hidden_dropout_prob"):
        config.hidden_dropout_prob = float(max(getattr(config, "hidden_dropout_prob", 0.0), trial.classifier_dropout))
    if hasattr(config, "use_cache"):
        config.use_cache = False
    model = AutoModelForSequenceClassification.from_pretrained(
        trial.model_id,
        config=config,
        ignore_mismatched_sizes=True,
        trust_remote_code=False,
    )
    if trial.model_family == "large" and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    model.to(device)
    commit = str(getattr(model.config, "_commit_hash", "") or getattr(config, "_commit_hash", "") or "")
    return model, tokenizer, commit


def optimizer_for(model: torch.nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    decay_params: list[torch.nn.Parameter] = []
    no_decay_params: list[torch.nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith(".bias") or "LayerNorm.weight" in name or "layer_norm.weight" in name:
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


def linear_schedule(optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max(1e-8, float(step + 1) / float(max(1, warmup_steps)))
        remaining = max(1, total_steps - warmup_steps)
        return max(0.0, float(total_steps - step) / float(remaining))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def softmax_np(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / np.clip(exp.sum(axis=1, keepdims=True), 1e-12, None)


def predict_proba(
    model: torch.nn.Module,
    tokenizer: Any,
    texts: list[str],
    max_length: int,
    batch_size: int,
    device: torch.device,
    precision: str,
) -> np.ndarray:
    model.eval()
    loader = DataLoader(
        TextDataset(texts),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_batch(tokenizer, max_length),
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    logits_all: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            with autocast_context(device, precision):
                logits = model(**batch).logits
            logits_all.append(logits.detach().float().cpu().numpy())
    return softmax_np(np.vstack(logits_all))


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


def metric_bundle(y_true: np.ndarray, probs: np.ndarray) -> dict[str, Any]:
    pred = probs.argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        pred,
        labels=list(range(len(LABELS))),
        zero_division=0,
    )
    metrics: dict[str, Any] = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "macro_f1": float(f1_score(y_true, pred, labels=list(range(len(LABELS))), average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, pred, labels=list(range(len(LABELS))), average="weighted", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "min_class_recall": float(np.min(recall)),
    }
    metrics["selection_score"] = float(
        0.40 * metrics["macro_f1"]
        + 0.30 * metrics["accuracy"]
        + 0.20 * metrics["balanced_accuracy"]
        + 0.10 * metrics["min_class_recall"]
    )
    for idx, label in enumerate(LABELS):
        lower = label.lower()
        metrics[f"{lower}_precision"] = float(precision[idx])
        metrics[f"{lower}_recall"] = float(recall[idx])
        metrics[f"{lower}_f1"] = float(f1[idx])
        metrics[f"{lower}_support"] = int(support[idx])
    metrics.update(calibration_metrics(y_true, probs))
    return metrics


def per_class_frame(y_true: np.ndarray, probs: np.ndarray) -> pd.DataFrame:
    pred = probs.argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        pred,
        labels=list(range(len(LABELS))),
        zero_division=0,
    )
    return pd.DataFrame(
        {
            "label": LABELS,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    )


def confusion_frame(y_true: np.ndarray, probs: np.ndarray) -> pd.DataFrame:
    pred = probs.argmax(axis=1)
    cm = confusion_matrix(y_true, pred, labels=list(range(len(LABELS))))
    return pd.DataFrame(cm, columns=LABELS).assign(true_label=LABELS)[["true_label", *LABELS]]


def train_one_fold(
    trial: Trial,
    train_texts: list[str],
    train_y: np.ndarray,
    val_texts: list[str],
    val_y: np.ndarray,
    seed: int,
    max_epochs: int,
    patience: int,
    device: torch.device,
    precision: str,
) -> tuple[np.ndarray, dict[str, Any], str]:
    set_seed(seed)
    per_device_batch, grad_accum = batch_plan(trial, device)
    model, tokenizer, commit = load_model_and_tokenizer(trial, device)
    optimizer = optimizer_for(model, trial.learning_rate, trial.weight_decay)
    train_loader = DataLoader(
        TextDataset(train_texts, train_y),
        batch_size=per_device_batch,
        shuffle=True,
        collate_fn=collate_batch(tokenizer, trial.max_length),
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    steps_per_epoch = max(1, math.ceil(len(train_loader) / grad_accum))
    total_steps = max(1, steps_per_epoch * max_epochs)
    warmup_steps = int(total_steps * trial.warmup_ratio)
    scheduler = linear_schedule(optimizer, warmup_steps, total_steps)
    weights = class_weights(train_y, device)
    scaler = make_scaler(precision)

    best_state: dict[str, torch.Tensor] | None = None
    best_metrics: dict[str, Any] | None = None
    best_epoch = 0
    stagnant = 0
    store_best_state = trial.model_family != "large"
    optimizer.zero_grad(set_to_none=True)
    global_step = 0

    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_loss = 0.0
        for step, batch in enumerate(train_loader, start=1):
            labels = batch.pop("labels").to(device)
            batch = {key: value.to(device) for key, value in batch.items()}
            with autocast_context(device, precision):
                logits = model(**batch).logits
                loss = compute_loss(logits, labels, trial, weights) / grad_accum
            if precision == "fp16":
                scaler.scale(loss).backward()
            else:
                loss.backward()
            epoch_loss += float(loss.detach().float().cpu()) * grad_accum

            if step % grad_accum == 0 or step == len(train_loader):
                if precision == "fp16":
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if precision == "fp16":
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

        probs = predict_proba(
            model,
            tokenizer,
            val_texts,
            trial.max_length,
            max(8, per_device_batch * 2),
            device,
            precision,
        )
        metrics = metric_bundle(val_y, probs)
        metrics["epoch_train_loss"] = epoch_loss / max(1, len(train_loader))
        log_event(
            {
                "event": "epoch",
                "trial_id": trial.trial_id,
                "seed": seed,
                "epoch": epoch,
                "val_macro_f1": metrics["macro_f1"],
                "val_accuracy": metrics["accuracy"],
                "selection_score": metrics["selection_score"],
            }
        )
        if best_metrics is None or metrics["selection_score"] > best_metrics["selection_score"] + 1e-9:
            best_metrics = metrics
            best_epoch = epoch
            if store_best_state:
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stagnant = 0
        else:
            stagnant += 1
            if stagnant >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    probs = predict_proba(
        model,
        tokenizer,
        val_texts,
        trial.max_length,
        max(8, per_device_batch * 2),
        device,
        precision,
    )
    fold_metrics = metric_bundle(val_y, probs)
    fold_metrics.update(
        {
            "best_epoch": int(best_epoch),
            "max_epochs": int(max_epochs),
            "early_stopping_patience": int(patience),
            "per_device_batch_size": int(per_device_batch),
            "gradient_accumulation_steps": int(grad_accum),
            "effective_batch_size": int(per_device_batch * grad_accum),
            "optimizer_steps": int(global_step),
            "best_state_restored": bool(best_state is not None),
            "best_state_restore_note": (
                "disabled_for_large_model_resource_stability"
                if not store_best_state
                else "best_validation_state_restored"
            ),
        }
    )
    del model, tokenizer, optimizer, scheduler, scaler, best_state
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return probs, fold_metrics, commit


def run_trial(
    trial: Trial,
    data: pd.DataFrame,
    assignments: pd.DataFrame,
    seeds: list[int],
    max_epochs: int,
    patience: int,
    device: torch.device,
    precision: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], str]:
    data = data.copy()
    data["model_input"] = data.apply(lambda row: model_input(row, trial.text_mode), axis=1)
    y_all = data["label_id"].to_numpy(dtype=int)
    oof_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    commit_hash = ""

    for seed in seeds:
        seed_probs = np.zeros((len(data), len(LABELS)), dtype=float)
        seed_seen = np.zeros(len(data), dtype=bool)
        seed_best_epochs: list[int] = []
        seed_assignments = assignments.loc[assignments["seed"].astype(int).eq(seed)]
        for fold in sorted(seed_assignments["fold"].unique(), key=lambda value: int(value)):
            val_idx = seed_assignments.loc[seed_assignments["fold"].eq(fold), "row_index"].astype(int).to_numpy()
            train_mask = np.ones(len(data), dtype=bool)
            train_mask[val_idx] = False
            train_idx = np.where(train_mask)[0]
            existing_oof_path = OUT_DIR / "development_oof_predictions.csv"
            if existing_oof_path.exists() and existing_oof_path.stat().st_size > 0:
                existing_oof = pd.read_csv(existing_oof_path, dtype=str, keep_default_na=False, low_memory=False)
                existing_fold = existing_oof.loc[
                    existing_oof["trial_id"].eq(trial.trial_id)
                    & existing_oof["seed"].astype(str).eq(str(seed))
                    & existing_oof["fold"].astype(str).eq(str(fold))
                ].copy()
                if len(existing_fold) == len(val_idx):
                    expected_ids = data.iloc[val_idx]["comment_id"].astype(str).tolist()
                    existing_by_id = existing_fold.drop_duplicates("comment_id", keep="last").set_index("comment_id")
                    if set(expected_ids).issubset(set(existing_by_id.index)):
                        probs_existing = existing_by_id.loc[expected_ids, ["prob_negative", "prob_neutral", "prob_positive"]].to_numpy(dtype=float)
                        seed_probs[val_idx] = probs_existing
                        seed_seen[val_idx] = True
                        existing_metrics_path = OUT_DIR / "development_fold_seed_metrics.csv"
                        if existing_metrics_path.exists() and existing_metrics_path.stat().st_size > 0:
                            existing_metrics = pd.read_csv(existing_metrics_path, dtype=str, keep_default_na=False, low_memory=False)
                            existing_metric = existing_metrics.loc[
                                existing_metrics["trial_id"].eq(trial.trial_id)
                                & existing_metrics["seed"].astype(str).eq(str(seed))
                                & existing_metrics["fold"].astype(str).eq(str(fold))
                            ]
                            if not existing_metric.empty:
                                seed_best_epochs.append(int(float(existing_metric.iloc[-1]["best_epoch"])))
                        log_event(
                            {
                                "event": "fold_skip_existing",
                                "trial_id": trial.trial_id,
                                "seed": seed,
                                "fold": fold,
                                "n_val": int(len(val_idx)),
                            }
                        )
                        continue
            log_event(
                {
                    "event": "fold_start",
                    "trial_id": trial.trial_id,
                    "model_id": trial.model_id,
                    "seed": seed,
                    "fold": fold,
                    "n_train": int(len(train_idx)),
                    "n_val": int(len(val_idx)),
                }
            )
            probs, fold_metrics, commit = train_one_fold(
                trial,
                data.iloc[train_idx]["model_input"].tolist(),
                y_all[train_idx],
                data.iloc[val_idx]["model_input"].tolist(),
                y_all[val_idx],
                seed,
                max_epochs,
                patience,
                device,
                precision,
            )
            commit_hash = commit_hash or commit
            seed_probs[val_idx] = probs
            seed_seen[val_idx] = True
            fold_metric_row = {
                **asdict(trial),
                "trial_id": trial.trial_id,
                "model_revision": commit,
                "seed": int(seed),
                "fold": str(fold),
                "training_mode": "full_fine_tuning",
                "device": str(device),
                "mixed_precision": precision,
                **fold_metrics,
            }
            metric_rows.append(fold_metric_row)
            seed_best_epochs.append(int(fold_metric_row["best_epoch"]))
            fold_oof_rows: list[dict[str, Any]] = []
            for row_pos, source_idx in enumerate(val_idx):
                row = data.iloc[source_idx]
                pred_id = int(probs[row_pos].argmax())
                fold_oof_rows.append(
                    {
                        **asdict(trial),
                        "trial_id": trial.trial_id,
                        "seed": int(seed),
                        "fold": str(fold),
                        "comment_id": row["comment_id"],
                        "true_label": row["final_human_label"],
                        "predicted_label": ID_TO_LABEL[pred_id],
                        "confidence": float(probs[row_pos].max()),
                        "prob_negative": float(probs[row_pos, LABEL_TO_ID["Negative"]]),
                        "prob_neutral": float(probs[row_pos, LABEL_TO_ID["Neutral"]]),
                        "prob_positive": float(probs[row_pos, LABEL_TO_ID["Positive"]]),
                        "text_cluster_id": row["text_cluster_id"],
                        "exact_duplicate_group_id": row["exact_duplicate_group_id"],
                        "near_duplicate_cluster_id": row["near_duplicate_cluster_id"],
                        "cv_group_id": row["cv_group_id"],
                    }
                )
            oof_rows.extend(fold_oof_rows)
            upsert_rows_csv(
                OUT_DIR / "development_fold_seed_metrics.csv",
                [fold_metric_row],
                ["trial_id", "seed", "fold"],
            )
            upsert_rows_csv(
                OUT_DIR / "development_oof_predictions.csv",
                fold_oof_rows,
                ["trial_id", "seed", "fold", "comment_id"],
            )
            log_event(
                {
                    "event": "fold_complete_checkpointed",
                    "trial_id": trial.trial_id,
                    "seed": seed,
                    "fold": fold,
                    "macro_f1": fold_metrics["macro_f1"],
                    "accuracy": fold_metrics["accuracy"],
                }
            )
        if not seed_seen.all():
            raise AssertionError(f"Incomplete OOF coverage for {trial.trial_id} seed={seed}")
        seed_metrics = metric_bundle(y_all, seed_probs)
        seed_metric_row = {
            **asdict(trial),
            "trial_id": trial.trial_id,
            "model_revision": commit_hash,
            "seed": int(seed),
            "fold": "ALL_OOF",
            "training_mode": "full_fine_tuning",
            "device": str(device),
            "mixed_precision": precision,
            "best_epoch": int(np.median(seed_best_epochs)) if seed_best_epochs else 1,
            **seed_metrics,
        }
        metric_rows.append(seed_metric_row)
        upsert_rows_csv(
            OUT_DIR / "development_fold_seed_metrics.csv",
            [seed_metric_row],
            ["trial_id", "seed", "fold"],
        )
    return oof_rows, metric_rows, commit_hash


def summarize_trials(metrics: pd.DataFrame) -> pd.DataFrame:
    all_oof = metrics.loc[metrics["fold"].eq("ALL_OOF")].copy()
    if all_oof.empty:
        return pd.DataFrame()
    metric_cols = [
        "accuracy",
        "macro_f1",
        "weighted_f1",
        "balanced_accuracy",
        "min_class_recall",
        "selection_score",
        "negative_recall",
        "negative_f1",
        "neutral_recall",
        "neutral_f1",
        "positive_recall",
        "positive_f1",
        "ece",
        "brier_score",
        "mean_confidence",
    ]
    group_cols = [
        "trial_id",
        "model_id",
        "model_family",
        "text_mode",
        "max_length",
        "learning_rate",
        "warmup_ratio",
        "weight_decay",
        "classifier_dropout",
        "loss",
        "label_smoothing",
        "focal_gamma",
        "training_mode",
        "device",
        "mixed_precision",
        "model_revision",
    ]
    rows: list[dict[str, Any]] = []
    for trial_id, group in all_oof.groupby("trial_id", sort=False):
        first = group.iloc[0]
        row = {col: first[col] for col in group_cols}
        row["n_seeds"] = int(group["seed"].nunique())
        for col in metric_cols:
            values = pd.to_numeric(group[col], errors="coerce")
            row[f"mean_{col}"] = float(values.mean())
            row[f"std_{col}"] = float(values.std(ddof=0))
            row[f"min_{col}"] = float(values.min())
            row[f"max_{col}"] = float(values.max())
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary = summary.sort_values(
        ["mean_selection_score", "mean_macro_f1", "mean_balanced_accuracy", "mean_accuracy", "std_macro_f1"],
        ascending=[False, False, False, False, True],
    ).reset_index(drop=True)
    summary["selected_for_final_training"] = False
    summary.loc[0, "selected_for_final_training"] = True
    return summary


def selected_seed(metrics: pd.DataFrame, trial_id: str) -> int:
    rows = metrics.loc[metrics["trial_id"].eq(trial_id) & metrics["fold"].eq("ALL_OOF")].copy()
    rows = rows.sort_values(["selection_score", "macro_f1", "balanced_accuracy", "accuracy"], ascending=False)
    return int(rows.iloc[0]["seed"])


def final_epoch_count(metrics: pd.DataFrame, trial_id: str, max_epochs: int) -> int:
    rows = metrics.loc[metrics["trial_id"].eq(trial_id) & ~metrics["fold"].eq("ALL_OOF")].copy()
    epochs = pd.to_numeric(rows["best_epoch"], errors="coerce").dropna()
    if epochs.empty:
        return max_epochs
    return int(max(1, min(max_epochs, math.ceil(float(epochs.median())))))


def train_final_model(
    trial: Trial,
    data: pd.DataFrame,
    seed: int,
    epochs: int,
    device: torch.device,
    precision: str,
) -> dict[str, Any]:
    set_seed(seed)
    data = data.copy()
    data["model_input"] = data.apply(lambda row: model_input(row, trial.text_mode), axis=1)
    texts = data["model_input"].tolist()
    y = data["label_id"].to_numpy(dtype=int)
    per_device_batch, grad_accum = batch_plan(trial, device)
    model, tokenizer, commit = load_model_and_tokenizer(trial, device)
    optimizer = optimizer_for(model, trial.learning_rate, trial.weight_decay)
    loader = DataLoader(
        TextDataset(texts, y),
        batch_size=per_device_batch,
        shuffle=True,
        collate_fn=collate_batch(tokenizer, trial.max_length),
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    steps_per_epoch = max(1, math.ceil(len(loader) / grad_accum))
    total_steps = max(1, steps_per_epoch * epochs)
    scheduler = linear_schedule(optimizer, int(total_steps * trial.warmup_ratio), total_steps)
    weights = class_weights(y, device)
    scaler = make_scaler(precision)
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for step, batch in enumerate(loader, start=1):
            labels = batch.pop("labels").to(device)
            batch = {key: value.to(device) for key, value in batch.items()}
            with autocast_context(device, precision):
                logits = model(**batch).logits
                loss = compute_loss(logits, labels, trial, weights) / grad_accum
            if precision == "fp16":
                scaler.scale(loss).backward()
            else:
                loss.backward()
            total_loss += float(loss.detach().float().cpu()) * grad_accum
            if step % grad_accum == 0 or step == len(loader):
                if precision == "fp16":
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if precision == "fp16":
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        print(
            json.dumps(
                {
                    "event": "final_epoch",
                    "epoch": epoch,
                    "train_loss": total_loss / max(1, len(loader)),
                    "trial_id": trial.trial_id,
                }
            ),
            flush=True,
        )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(MODEL_DIR, safe_serialization=True)
    tokenizer.save_pretrained(MODEL_DIR)
    save_json(MODEL_DIR / "label_map.json", {"label_to_id": LABEL_TO_ID, "id_to_label": ID_TO_LABEL})
    training_config = {
        **asdict(trial),
        "trial_id": trial.trial_id,
        "model_revision": commit,
        "final_seed": int(seed),
        "final_epochs": int(epochs),
        "max_epochs_during_oof": int(epochs),
        "per_device_batch_size": int(per_device_batch),
        "gradient_accumulation_steps": int(grad_accum),
        "effective_batch_size": int(per_device_batch * grad_accum),
        "training_mode": "full_fine_tuning",
        "encoder_update_status": "all_transformer_layers_trainable",
        "prediction_rule": "argmax_no_threshold_tuning",
        "label_source_column": "final_human_label",
        "forbidden_supervision_sources": FORBIDDEN_LABEL_SOURCES,
        "locked_test_used_for_training_or_selection": False,
    }
    save_json(MODEL_DIR / "selected_trial_config.json", training_config)
    del model, tokenizer, optimizer, scheduler, scaler
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return training_config


def save_selected_development_outputs(
    data: pd.DataFrame,
    oof: pd.DataFrame,
    metrics: pd.DataFrame,
    selected_trial_id: str,
    seed: int,
) -> dict[str, Any]:
    selected_oof = oof.loc[oof["trial_id"].eq(selected_trial_id) & oof["seed"].astype(int).eq(seed)].copy()
    if len(selected_oof) != len(data):
        raise AssertionError(f"Selected OOF rows={len(selected_oof)} expected={len(data)}")
    true = selected_oof["true_label"].map(LABEL_TO_ID).to_numpy(dtype=int)
    probs = selected_oof[["prob_negative", "prob_neutral", "prob_positive"]].to_numpy(dtype=float)
    development_metrics = metric_bundle(true, probs)
    confusion_frame(true, probs).to_csv(OUT_DIR / "development_confusion_matrix.csv", index=False, encoding="utf-8-sig")
    per_class_frame(true, probs).to_csv(OUT_DIR / "development_per_class_metrics.csv", index=False, encoding="utf-8-sig")
    save_json(OUT_DIR / "selected_development_oof_metrics.json", development_metrics)
    return development_metrics


def write_search_space_coverage(trials: list[Trial]) -> dict[str, Any]:
    return {
        "max_length": sorted({trial.max_length for trial in trials}),
        "large_learning_rate": sorted({trial.learning_rate for trial in trials if trial.model_family == "large"}),
        "base_learning_rate": sorted({trial.learning_rate for trial in trials if trial.model_family in {"base", "tweet_base", "warm_start"}}),
        "warmup_ratio": sorted({trial.warmup_ratio for trial in trials}),
        "weight_decay": sorted({trial.weight_decay for trial in trials}),
        "classifier_dropout": sorted({trial.classifier_dropout for trial in trials}),
        "loss": sorted({trial.loss for trial in trials}),
        "label_smoothing": sorted({trial.label_smoothing for trial in trials}),
        "text_mode": sorted({trial.text_mode for trial in trials}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the final RM2 V4 3-class IndoBERT sentiment candidate.")
    parser.add_argument("--models", nargs="*", default=[
        "indobenchmark/indobert-large-p2",
        "indobenchmark/indobert-base-p2",
        "indolem/indobertweet-base-uncased",
        "apriandito/indobert-sentiment-classifier",
    ])
    parser.add_argument("--search-profile", choices=["full", "quick"], default="full")
    parser.add_argument("--seeds", nargs="*", type=int, default=[42, 52, 62])
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--max-epochs", type=int, default=15)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--max-trials", type=int, default=0, help="0 means run all planned trials.")
    parser.add_argument(
        "--no-finalize",
        action="store_true",
        help="Run/resume development trials and write partial OOF outputs without selecting/freezing the final model.",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    info = device_info()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    precision = str(info["mixed_precision"])
    data = read_development_data()
    locked = read_locked_test_data()
    dev_locked_overlap = {
        col: int(
            len(
                set(data.loc[data[col].map(nonempty).ne(""), col])
                & set(locked.loc[locked[col].map(nonempty).ne(""), col])
            )
        )
        for col in ["comment_id", "text_cluster_id", "exact_duplicate_group_id", "near_duplicate_cluster_id"]
    }
    trials = build_trials(args.models, args.search_profile)
    if args.max_trials and args.max_trials > 0:
        trials = trials[: args.max_trials]
    if not trials:
        raise RuntimeError("No trials planned.")
    assignments = load_or_make_fold_assignments(data, args.seeds, args.n_splits, args.resume)
    grid = pd.DataFrame([{**asdict(trial), "trial_id": trial.trial_id, "status": "PENDING"} for trial in trials])
    grid.to_csv(OUT_DIR / "candidate_grid_manifest.csv", index=False, encoding="utf-8-sig")

    all_oof: list[dict[str, Any]] = []
    all_metrics: list[dict[str, Any]] = []
    completed_ids: set[str] = set()
    if args.resume and (OUT_DIR / "development_fold_seed_metrics.csv").exists() and (OUT_DIR / "development_oof_predictions.csv").exists():
        existing_metrics = pd.read_csv(OUT_DIR / "development_fold_seed_metrics.csv", dtype=str, keep_default_na=False)
        existing_oof = pd.read_csv(OUT_DIR / "development_oof_predictions.csv", dtype=str, keep_default_na=False)
        existing_metrics = dedupe_frame(existing_metrics, METRIC_KEYS)
        existing_oof = dedupe_frame(existing_oof, OOF_KEYS)
        completed_by_seed = (
            existing_metrics.loc[existing_metrics["fold"].eq("ALL_OOF")]
            .groupby("trial_id")["seed"]
            .nunique()
        )
        completed_ids = set(completed_by_seed.loc[completed_by_seed.ge(len(args.seeds))].index)
        all_metrics.extend(existing_metrics.to_dict("records"))
        all_oof.extend(existing_oof.to_dict("records"))

    grid_rows: list[dict[str, Any]] = []
    for trial in trials:
        status = "COMPLETED_FROM_RESUME" if trial.trial_id in completed_ids else "RUN"
        row = {**asdict(trial), "trial_id": trial.trial_id, "status": status, "notes": ""}
        if trial.trial_id not in completed_ids:
            try:
                oof_rows, metric_rows, commit = run_trial(
                    trial,
                    data,
                    assignments,
                    args.seeds,
                    args.max_epochs,
                    args.patience,
                    device,
                    precision,
                )
                all_oof.extend(oof_rows)
                all_metrics.extend(metric_rows)
                row.update({"status": "COMPLETED", "model_revision": commit})
            except Exception as exc:
                row.update({"status": "MODEL_LOAD_OR_TRAIN_FAILED", "notes": repr(exc)})
                print(json.dumps({"event": "trial_failed", "trial_id": trial.trial_id, "error": repr(exc)}), flush=True)
                gc.collect()
                if device.type == "cuda":
                    torch.cuda.empty_cache()
        grid_rows.append(row)
        all_oof = dedupe_records(all_oof, OOF_KEYS)
        all_metrics = dedupe_records(all_metrics, METRIC_KEYS)
        pd.DataFrame(grid_rows).to_csv(OUT_DIR / "candidate_grid_manifest.csv", index=False, encoding="utf-8-sig")
        if all_oof:
            pd.DataFrame(all_oof).to_csv(OUT_DIR / "development_oof_predictions.csv", index=False, encoding="utf-8-sig")
        if all_metrics:
            pd.DataFrame(all_metrics).to_csv(OUT_DIR / "development_fold_seed_metrics.csv", index=False, encoding="utf-8-sig")

    if not all_metrics:
        raise RuntimeError("No V4 IndoBERT trials completed.")
    oof = dedupe_frame(pd.DataFrame(all_oof), OOF_KEYS)
    metrics = dedupe_frame(pd.DataFrame(all_metrics), METRIC_KEYS)
    summary = summarize_trials(metrics)
    if summary.empty:
        raise RuntimeError("No completed ALL_OOF metrics available for model selection.")
    summary.to_csv(OUT_DIR / "development_trial_summary.csv", index=False, encoding="utf-8-sig")
    if args.no_finalize:
        partial_manifest = {
            "status": "INDOBERT_V4_DEVELOPMENT_SEARCH_PARTIAL",
            "created_at_utc": utc_now(),
            "development_registry": DEV_REGISTRY.relative_to(ROOT).as_posix(),
            "locked_test_registry": LOCKED_REGISTRY.relative_to(ROOT).as_posix(),
            "development_evaluable_rows": int(len(data)),
            "label_source_column": "final_human_label",
            "label_vocabulary": LABELS,
            "prediction_label_source_used": False,
            "locked_test_used_for_training_or_selection": False,
            "locked_test_used_for_early_stopping": False,
            "locked_test_used_for_threshold_selection": False,
            "full_corpus_inference_run": False,
            "selection_basis": "partial development OOF summary only; no final model frozen in this run",
            "planned_trials_in_this_invocation": int(len(trials)),
            "completed_trials_total": int(summary["trial_id"].nunique()),
            "current_best_trial_id": str(summary.iloc[0]["trial_id"]),
            "current_best_mean_selection_score": float(summary.iloc[0]["mean_selection_score"]),
            "device_info": info,
            "package_versions": package_versions(),
        }
        save_json(OUT_DIR / "INDOBERT_V4_TRAINING_MANIFEST.partial.json", partial_manifest)
        print(
            json.dumps(
                {
                    "status": "partial_search_saved",
                    "completed_trials_total": int(summary["trial_id"].nunique()),
                    "current_best_trial_id": str(summary.iloc[0]["trial_id"]),
                    "note": "Run again without --no-finalize after the intended search is complete.",
                },
                indent=2,
            ),
            flush=True,
        )
        return
    selected = summary.loc[summary["selected_for_final_training"].eq(True)].iloc[0]
    trial_lookup = {trial.trial_id: trial for trial in trials}
    selected_trial = trial_lookup[str(selected["trial_id"])]
    final_seed = selected_seed(metrics, selected_trial.trial_id)
    final_epochs = final_epoch_count(metrics, selected_trial.trial_id, args.max_epochs)
    development_metrics = save_selected_development_outputs(data, oof, metrics, selected_trial.trial_id, final_seed)
    final_config = train_final_model(selected_trial, data, final_seed, final_epochs, device, precision)
    selected_payload = {
        "selected_trial": selected.to_dict(),
        "selected_seed": final_seed,
        "final_epochs": final_epochs,
        "selection_basis": "development OOF only",
        "selection_score_formula": "0.40*macro_f1 + 0.30*accuracy + 0.20*balanced_accuracy + 0.10*min_class_recall",
        "development_oof_metrics_for_final_seed": development_metrics,
        "final_training_config": final_config,
    }
    save_json(OUT_DIR / "selected_trial_config.json", selected_payload)

    model_hashes = {
        path.name: sha256_file(path)
        for path in MODEL_DIR.glob("*")
        if path.is_file() and path.suffix in {".safetensors", ".bin", ".json"}
    }
    manifest = {
        "status": "INDOBERT_V4_FINAL_CANDIDATE_TRAINED",
        "created_at_utc": utc_now(),
        "branch_expected": "research/sentiment-master-annotation-v4",
        "development_registry": DEV_REGISTRY.relative_to(ROOT).as_posix(),
        "locked_test_registry": LOCKED_REGISTRY.relative_to(ROOT).as_posix(),
        "development_evaluable_rows": int(len(data)),
        "development_class_counts": data["final_human_label"].value_counts().reindex(LABELS).fillna(0).astype(int).to_dict(),
        "locked_test_evaluable_rows_read_for_split_audit_only": int(len(locked)),
        "locked_test_class_counts_read_for_split_audit_only": locked["final_human_label"].value_counts().reindex(LABELS).fillna(0).astype(int).to_dict(),
        "locked_test_neutral_target_shortfall": 6,
        "label_source_column": "final_human_label",
        "label_vocabulary": LABELS,
        "forbidden_supervision_sources": FORBIDDEN_LABEL_SOURCES,
        "prediction_label_source_used": False,
        "locked_test_used_for_training_or_selection": False,
        "locked_test_used_for_early_stopping": False,
        "locked_test_used_for_threshold_selection": False,
        "full_corpus_inference_run": False,
        "legacy_outputs_modified": False,
        "cv_splitter": "StratifiedGroupKFold",
        "n_splits": int(args.n_splits),
        "seeds": [int(seed) for seed in args.seeds],
        "grouping_policy": "union-find over near_duplicate_cluster_id, exact_duplicate_group_id, text_cluster_id; comment_id fallback",
        "development_locked_overlap_audit": dev_locked_overlap,
        "search_space_coverage": write_search_space_coverage(trials),
        "planned_trials": int(len(trials)),
        "completed_trials": int(summary["trial_id"].nunique()),
        "selected_trial_id": selected_trial.trial_id,
        "selected_seed": int(final_seed),
        "final_epochs": int(final_epochs),
        "model_dir": MODEL_DIR.relative_to(ROOT).as_posix(),
        "experiment_dir": OUT_DIR.relative_to(ROOT).as_posix(),
        "model_hashes": model_hashes,
        "package_versions": package_versions(),
        "device_info": info,
    }
    save_json(OUT_DIR / "INDOBERT_V4_TRAINING_MANIFEST.json", manifest)
    save_json(MODEL_DIR / "INDOBERT_V4_TRAINING_MANIFEST.json", manifest)
    print(
        json.dumps(
            {
                "selected_trial_id": selected_trial.trial_id,
                "selected_seed": final_seed,
                "development_macro_f1": development_metrics["macro_f1"],
                "development_accuracy": development_metrics["accuracy"],
                "model_dir": MODEL_DIR.relative_to(ROOT).as_posix(),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
