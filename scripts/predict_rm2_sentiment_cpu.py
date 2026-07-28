from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import re
import sys
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


LABELS = ["Negative", "Neutral", "Positive"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def nonempty(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
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


def model_input(row: dict[str, object], text_column: str, text_mode: str) -> str:
    comment = normalize_text(row.get(text_column, ""))
    if text_mode == "comment_only":
        return comment
    context_parts = [
        normalize_text(row.get("brand_or_video_context", "")),
        normalize_text(row.get("product_category", "")),
    ]
    context = " | ".join(dict.fromkeys(part for part in context_parts if part))
    return f"{context} [SEP] {comment}" if context else comment


def parse_sha256s(path: Path) -> dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing checksum file: {path}")
    checksums: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, name = line.split(maxsplit=1)
        checksums[name.strip()] = digest.strip()
    return checksums


def assert_not_lfs_pointer(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing checkpoint file: {path}")
    head = path.read_bytes()[:128]
    if head.startswith(b"version https://git-lfs.github.com/spec/") or path.stat().st_size < 1024 * 1024:
        raise RuntimeError(
            f"{path} looks like a Git LFS pointer or incomplete checkpoint. "
            "Run `git lfs pull` or download the release asset before inference."
        )


def verify_model_dir(model_dir: Path) -> dict[str, str]:
    checkpoint = model_dir / "model.safetensors"
    if not checkpoint.exists():
        checkpoint = model_dir / "pytorch_model.bin"
    assert_not_lfs_pointer(checkpoint)
    checksums = parse_sha256s(model_dir / "SHA256SUMS.txt")
    for name, expected in checksums.items():
        path = model_dir / name
        if not path.exists():
            raise FileNotFoundError(f"Checksum references missing file: {path}")
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"Checksum mismatch for {name}: expected {expected}, got {actual}")
    return checksums


def load_label_order(model_dir: Path) -> list[str]:
    label_map = read_json(model_dir / "label_map.json")
    labels = [label_map["id_to_label"][str(i)] for i in range(len(LABELS))]
    if labels != LABELS:
        raise RuntimeError(f"Unexpected label order: {labels}")
    return labels


def predict(model_dir: Path, rows: list[dict[str, object]], text_column: str) -> list[dict[str, object]]:
    verify_model_dir(model_dir)
    labels = load_label_order(model_dir)
    config = read_json(model_dir / "selected_trial_config.json")
    text_mode = str(config.get("text_mode") or "context_sep_comment")
    max_length = int(config.get("max_length") or 256)
    texts = [model_input(row, text_column, text_mode) for row in rows]

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=False)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, trust_remote_code=False)
    model.to(torch.device("cpu"))
    model.eval()

    encoded = tokenizer(texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    with torch.no_grad():
        probs = torch.softmax(model(**encoded).logits.float(), dim=1).cpu().numpy()
    outputs: list[dict[str, object]] = []
    for row, prob in zip(rows, probs):
        pred_idx = int(prob.argmax())
        values = {
            "predicted_label": labels[pred_idx],
            "confidence": float(prob[pred_idx]),
            "prob_negative": float(prob[0]),
            "prob_neutral": float(prob[1]),
            "prob_positive": float(prob[2]),
        }
        if any(not math.isfinite(float(v)) for v in values.values() if isinstance(v, float)):
            raise RuntimeError("Model produced NaN or Inf probability.")
        outputs.append({**row, **values})
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RM2 IndoBERT V4 sentiment inference on CPU.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--text")
    parser.add_argument("--input-csv")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--output-csv")
    args = parser.parse_args()

    if bool(args.text) == bool(args.input_csv):
        raise SystemExit("Provide exactly one of --text or --input-csv.")
    model_dir = Path(args.model_dir)
    if args.text:
        rows = [{"text": args.text}]
        result = predict(model_dir, rows, "text")[0]
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    if not args.output_csv:
        raise SystemExit("--output-csv is required with --input-csv.")
    frame = pd.read_csv(args.input_csv, dtype=str, keep_default_na=False)
    if args.text_column not in frame.columns:
        raise SystemExit(f"Missing --text-column in input CSV: {args.text_column}")
    outputs = predict(model_dir, frame.to_dict("records"), args.text_column)
    pd.DataFrame(outputs).to_csv(args.output_csv, index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
