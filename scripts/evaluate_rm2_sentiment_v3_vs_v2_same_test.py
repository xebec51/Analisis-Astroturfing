from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from safetensors.torch import load_file as load_safetensors
from scipy.stats import binomtest
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
)
from torch import nn
from transformers import AutoModel, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "output/rm2_sentiment/validation/human_v3/human_label_registry_v3.csv"
V2_MODEL = ROOT / "output/rm2_sentiment/model/frozen/selected_model_development_frozen.joblib"
V2_CONFIG = ROOT / "output/rm2_sentiment/model/frozen/selected_model_development_frozen_config.json"
V3_DIR = ROOT / "output/rm2_sentiment/model/indobert_v3_candidate"
V3_CONFIG = V3_DIR / "v3_candidate_config.json"
OUT_DIR = ROOT / "output/rm2_sentiment/experiments/indobert_v3/final_test_evaluation"

LABELS = ["Negative", "Neutral", "Positive"]
LABEL_TO_ID = {label: idx for idx, label in enumerate(LABELS)}
RANDOM_SEED = 42
BOOTSTRAPS = 1000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LinearHead(nn.Module):
    def __init__(self, n_features: int, n_classes: int) -> None:
        super().__init__()
        self.classifier = nn.Linear(n_features, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path, dtype=str, keep_default_na=False, low_memory=False)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def softmax(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    if scores.ndim == 1:
        scores = scores.reshape(-1, 1)
    scores = scores - np.max(scores, axis=1, keepdims=True)
    exp = np.exp(scores)
    return exp / np.clip(exp.sum(axis=1, keepdims=True), 1e-12, None)


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


def context_text(row: pd.Series) -> str:
    parts = []
    for col in ["product_category", "brand_or_video_context"]:
        value = str(row.get(col, "")).strip()
        if value:
            parts.append(value)
    context = " | ".join(dict.fromkeys(parts))
    comment = str(row.get("model_text") or row.get("comment_text_original") or "").strip()
    return f"{context} [SEP] {comment}" if context else comment


def load_test() -> pd.DataFrame:
    registry = read_csv(REGISTRY)
    test = registry.loc[
        registry["split_family"].eq("final_test") & registry["final_sentiment_label"].isin(LABELS)
    ].copy()
    if test.empty:
        raise RuntimeError("No final_test rows found in V3 registry.")
    test["true_id"] = test["final_sentiment_label"].map(LABEL_TO_ID).astype(int)
    return test.reset_index(drop=True)


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


def predict_v2(test: pd.DataFrame) -> tuple[np.ndarray, float, dict[str, object]]:
    artifact = joblib.load(V2_MODEL)
    config = json.loads(V2_CONFIG.read_text(encoding="utf-8"))
    text = test["comment_text_original"].map(clean_social)
    parts = []
    pipelines = artifact["pipeline"] if isinstance(artifact["pipeline"], list) else [{"pipeline": artifact["pipeline"]}]
    for component in pipelines:
        parts.append(predict_proba_aligned(component["pipeline"], text, artifact["label_encoder"]))
    probs = np.mean(parts, axis=0)
    probs = probs / np.clip(probs.sum(axis=1, keepdims=True), 1e-12, None)
    threshold = float(config.get("selected_threshold", config.get("threshold", artifact.get("selected_threshold", 0.0))))
    return probs, threshold, config


def extract_v3_embeddings(config: dict[str, object], texts: list[str]) -> np.ndarray:
    model_id = str(config["model_id"])
    revision = str(config["revision"])
    max_length = int(config["hyperparameters"]["max_length"])
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=False)
    model = AutoModel.from_pretrained(model_id, revision=revision, trust_remote_code=False)
    model.to(DEVICE)
    model.eval()
    vectors = []
    with torch.no_grad():
        for start in range(0, len(texts), 16):
            batch = texts[start : start + 16]
            encoded = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            encoded = {key: value.to(DEVICE) for key, value in encoded.items()}
            output = model(**encoded)
            pooled = output.pooler_output if getattr(output, "pooler_output", None) is not None else output.last_hidden_state[:, 0, :]
            vectors.append(pooled.detach().cpu().numpy().astype("float32"))
    return np.vstack(vectors)


def predict_v3(test: pd.DataFrame) -> tuple[np.ndarray, float, dict[str, object]]:
    config = json.loads(V3_CONFIG.read_text(encoding="utf-8"))
    texts = test.apply(context_text, axis=1).tolist()
    embeddings = extract_v3_embeddings(config, texts)
    scaler = joblib.load(V3_DIR / "embedding_scaler.joblib")
    x = scaler.transform(embeddings).astype("float32")
    state = load_safetensors(V3_DIR / "linear_head.safetensors")
    weight = state["linear_head.classifier.weight"]
    bias = state["linear_head.classifier.bias"]
    model = LinearHead(weight.shape[1], weight.shape[0])
    model.load_state_dict({"classifier.weight": weight, "classifier.bias": bias})
    model.to(DEVICE)
    model.eval()
    with torch.no_grad():
        logits = model(torch.tensor(x, dtype=torch.float32, device=DEVICE)).detach().cpu().numpy()
    probs = softmax(logits)
    threshold = float(config["hyperparameters"]["threshold"].get("abstention_threshold", 0.0))
    return probs, threshold, config


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


def metric_bundle(y_true: np.ndarray, probs: np.ndarray, threshold: float, model_name: str) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    covered = conf >= threshold
    y_cov = y_true[covered]
    pred_cov = pred[covered]
    if len(y_cov) == 0:
        raise RuntimeError(f"{model_name} covers zero rows.")
    precision, recall, f1, support = precision_recall_fscore_support(
        y_cov, pred_cov, labels=list(range(len(LABELS))), zero_division=0
    )
    metrics: dict[str, object] = {
        "model": model_name,
        "n_test_evaluable": int(len(y_true)),
        "n_covered": int(covered.sum()),
        "n_abstained": int((~covered).sum()),
        "coverage": float(covered.mean()),
        "abstention_rate": float((~covered).mean()),
        "accuracy": float(accuracy_score(y_cov, pred_cov)),
        "macro_f1": float(f1_score(y_cov, pred_cov, labels=list(range(len(LABELS))), average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_cov, pred_cov, labels=list(range(len(LABELS))), average="weighted", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_cov, pred_cov)),
        "mcc": float(matthews_corrcoef(y_cov, pred_cov)),
        "accuracy_all_evaluable_abstain_wrong": float(((pred == y_true) & covered).mean()),
        "class_collapse": bool(len(set(pred_cov.tolist())) < len(LABELS)),
        "minimum_recall": float(np.min(recall)),
        "positive_recall": float(recall[LABEL_TO_ID["Positive"]]),
        "positive_precision": float(precision[LABEL_TO_ID["Positive"]]),
        "positive_f1": float(f1[LABEL_TO_ID["Positive"]]),
        **{f"covered_{key}": value for key, value in calibration_metrics(y_cov, probs[covered]).items()},
        **{f"all_evaluable_{key}": value for key, value in calibration_metrics(y_true, probs).items()},
    }
    per_class = pd.DataFrame(
        [
            {
                "model": model_name,
                "label": LABELS[idx],
                "precision": float(precision[idx]),
                "recall": float(recall[idx]),
                "f1": float(f1[idx]),
                "support": int(support[idx]),
            }
            for idx in range(len(LABELS))
        ]
    )
    cm = confusion_matrix(y_cov, pred_cov, labels=list(range(len(LABELS))))
    cm_rows = []
    for i, true_label in enumerate(LABELS):
        for j, pred_label in enumerate(LABELS):
            cm_rows.append({"model": model_name, "true_label": true_label, "predicted_label": pred_label, "count": int(cm[i, j])})
    return metrics, per_class, pd.DataFrame(cm_rows)


def bootstrap_ci(y_true: np.ndarray, probs: np.ndarray, threshold: float, model_name: str) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    rows = []
    metric_names = ["accuracy", "macro_f1", "weighted_f1", "balanced_accuracy", "mcc", "positive_recall", "positive_precision", "positive_f1"]
    values = {name: [] for name in metric_names}
    idx_all = np.arange(len(y_true))
    for _ in range(BOOTSTRAPS):
        idx = rng.choice(idx_all, size=len(idx_all), replace=True)
        metrics, _, _ = metric_bundle(y_true[idx], probs[idx], threshold, model_name)
        for name in metric_names:
            values[name].append(float(metrics[name]))
    for name, vals in values.items():
        rows.append(
            {
                "model": model_name,
                "metric": name,
                "ci_low": float(np.quantile(vals, 0.025)),
                "ci_high": float(np.quantile(vals, 0.975)),
                "n_bootstrap": BOOTSTRAPS,
            }
        )
    return pd.DataFrame(rows)


def paired_bootstrap_delta(y_true: np.ndarray, probs_v2: np.ndarray, thr_v2: float, probs_v3: np.ndarray, thr_v3: float) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    metric_names = ["accuracy", "macro_f1", "weighted_f1", "balanced_accuracy", "mcc", "positive_recall", "positive_precision", "positive_f1"]
    deltas = {name: [] for name in metric_names}
    idx_all = np.arange(len(y_true))
    for _ in range(BOOTSTRAPS):
        idx = rng.choice(idx_all, size=len(idx_all), replace=True)
        m2, _, _ = metric_bundle(y_true[idx], probs_v2[idx], thr_v2, "V2")
        m3, _, _ = metric_bundle(y_true[idx], probs_v3[idx], thr_v3, "V3")
        for name in metric_names:
            deltas[name].append(float(m3[name]) - float(m2[name]))
    rows = []
    for name, vals in deltas.items():
        rows.append(
            {
                "metric": name,
                "delta_v3_minus_v2_mean": float(np.mean(vals)),
                "ci_low": float(np.quantile(vals, 0.025)),
                "ci_high": float(np.quantile(vals, 0.975)),
                "paired_bootstrap_shows_meaningful_decrease": bool(np.quantile(vals, 0.975) < 0),
                "n_bootstrap": BOOTSTRAPS,
            }
        )
    return pd.DataFrame(rows)


def mcnemar(y_true: np.ndarray, probs_v2: np.ndarray, thr_v2: float, probs_v3: np.ndarray, thr_v3: float) -> pd.DataFrame:
    pred2 = probs_v2.argmax(axis=1)
    pred3 = probs_v3.argmax(axis=1)
    cov2 = probs_v2.max(axis=1) >= thr_v2
    cov3 = probs_v3.max(axis=1) >= thr_v3
    correct2 = (pred2 == y_true) & cov2
    correct3 = (pred3 == y_true) & cov3
    b = int((correct2 & ~correct3).sum())
    c = int((~correct2 & correct3).sum())
    n = b + c
    p = float(binomtest(min(b, c), n=n, p=0.5, alternative="two-sided").pvalue) if n else 1.0
    return pd.DataFrame(
        [
            {
                "b_v2_correct_v3_wrong": b,
                "c_v2_wrong_v3_correct": c,
                "discordant_pairs": n,
                "mcnemar_exact_p": p,
            }
        ]
    )


def error_taxonomy(test: pd.DataFrame, probs: np.ndarray, threshold: float, model_name: str) -> pd.DataFrame:
    pred = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    covered = conf >= threshold
    rows = []
    for idx, row in test.iterrows():
        true_id = int(row["true_id"])
        if not covered[idx] or pred[idx] == true_id:
            continue
        text = str(row["comment_text_original"])
        lower = text.lower()
        flags = {
            "question": "?" in text or any(token in lower for token in ["apa", "gimana", "gmna", "aman", "cocok"]),
            "negation": any(token in lower for token in ["tidak", "nggak", "gak", "ga ", "belum", "bukan", "jangan"]),
            "emoji_or_non_ascii": any(ord(ch) > 127 for ch in text),
            "very_short": len(lower.split()) <= 3,
            "mixed_sentiment_connector": any(token in lower for token in ["tapi", "namun", "cuma", "but"]),
        }
        active = [name for name, value in flags.items() if value]
        rows.append(
            {
                "model": model_name,
                "comment_id": row["comment_id"],
                "true_label": LABELS[true_id],
                "predicted_label": LABELS[int(pred[idx])],
                "confidence": float(conf[idx]),
                "taxonomy_flags": ";".join(active) if active else "general_misclassification",
                "brand_or_video_context": row.get("brand_or_video_context", ""),
                "comment_text_original": text,
            }
        )
    return pd.DataFrame(rows)


def acceptance_decision(metrics: pd.DataFrame, paired: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    m = metrics.set_index("model")
    v2 = m.loc["V2_frozen"]
    v3 = m.loc["IndoBERT_V3_candidate"]
    paired_decrease = bool(paired["paired_bootstrap_shows_meaningful_decrease"].any())
    checks = [
        ("accuracy V3 > accuracy V2", float(v3["accuracy"]) > float(v2["accuracy"]), float(v2["accuracy"]), float(v3["accuracy"])),
        ("macro-F1 V3 >= macro-F1 V2 + 0.02", float(v3["macro_f1"]) >= float(v2["macro_f1"]) + 0.02, float(v2["macro_f1"]), float(v3["macro_f1"])),
        ("balanced accuracy V3 > V2", float(v3["balanced_accuracy"]) > float(v2["balanced_accuracy"]), float(v2["balanced_accuracy"]), float(v3["balanced_accuracy"])),
        ("MCC V3 > V2", float(v3["mcc"]) > float(v2["mcc"]), float(v2["mcc"]), float(v3["mcc"])),
        ("Positive recall V3 >= 0.60", float(v3["positive_recall"]) >= 0.60, 0.60, float(v3["positive_recall"])),
        ("Positive recall V3 >= V2 + 0.10", float(v3["positive_recall"]) >= float(v2["positive_recall"]) + 0.10, float(v2["positive_recall"]), float(v3["positive_recall"])),
        ("Positive precision V3 >= 0.65", float(v3["positive_precision"]) >= 0.65, 0.65, float(v3["positive_precision"])),
        ("Positive F1 V3 > V2", float(v3["positive_f1"]) > float(v2["positive_f1"]), float(v2["positive_f1"]), float(v3["positive_f1"])),
        ("coverage >= 0.90", float(v3["coverage"]) >= 0.90, 0.90, float(v3["coverage"])),
        ("no class collapse", not bool(v3["class_collapse"]), True, not bool(v3["class_collapse"])),
        ("no class recall below 0.50", float(v3["minimum_recall"]) >= 0.50, 0.50, float(v3["minimum_recall"])),
        ("model stable lintas seed", False, "required", "development std macro-F1 documented; CPU linear-probe variance too high for stability claim"),
        ("paired bootstrap no meaningful decrease", not paired_decrease, False, paired_decrease),
    ]
    status = "FINAL_MODEL_VALIDATED_INDOBERT_V3" if all(passed for _, passed, _, _ in checks) else "INDOBERT_V3_NOT_ACCEPTED_KEEP_V2"
    rows = [
        {
            "gate": gate,
            "passed": bool(passed),
            "reference_or_threshold": reference,
            "observed_v3": observed,
            "final_status": status,
        }
        for gate, passed, reference, observed in checks
    ]
    return pd.DataFrame(rows), status


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    test = load_test()
    y_true = test["true_id"].to_numpy(dtype=int)
    probs_v2, threshold_v2, config_v2 = predict_v2(test)
    probs_v3, threshold_v3, config_v3 = predict_v3(test)

    metrics_rows = []
    per_class_frames = []
    cm_frames = []
    for model_name, probs, threshold in [
        ("V2_frozen", probs_v2, threshold_v2),
        ("IndoBERT_V3_candidate", probs_v3, threshold_v3),
    ]:
        metrics, per_class, cm = metric_bundle(y_true, probs, threshold, model_name)
        metrics_rows.append(metrics)
        per_class_frames.append(per_class)
        cm_frames.append(cm)

    metrics_df = pd.DataFrame(metrics_rows)
    per_class_df = pd.concat(per_class_frames, ignore_index=True)
    cm_df = pd.concat(cm_frames, ignore_index=True)
    ci_df = pd.concat(
        [
            bootstrap_ci(y_true, probs_v2, threshold_v2, "V2_frozen"),
            bootstrap_ci(y_true, probs_v3, threshold_v3, "IndoBERT_V3_candidate"),
        ],
        ignore_index=True,
    )
    paired_df = paired_bootstrap_delta(y_true, probs_v2, threshold_v2, probs_v3, threshold_v3)
    mcnemar_df = mcnemar(y_true, probs_v2, threshold_v2, probs_v3, threshold_v3)
    errors_df = pd.concat(
        [
            error_taxonomy(test, probs_v2, threshold_v2, "V2_frozen"),
            error_taxonomy(test, probs_v3, threshold_v3, "IndoBERT_V3_candidate"),
        ],
        ignore_index=True,
    )
    gates_df, final_status = acceptance_decision(metrics_df, paired_df)

    pred_rows = test[["comment_id", "final_sentiment_label", "comment_text_original", "brand_or_video_context"]].copy()
    for prefix, probs, threshold in [("v2", probs_v2, threshold_v2), ("v3", probs_v3, threshold_v3)]:
        pred_idx = probs.argmax(axis=1)
        pred_rows[f"{prefix}_predicted_label"] = [LABELS[i] for i in pred_idx]
        pred_rows[f"{prefix}_confidence"] = probs.max(axis=1)
        pred_rows[f"{prefix}_covered"] = probs.max(axis=1) >= threshold
        for label in LABELS:
            pred_rows[f"{prefix}_prob_{label.lower()}"] = probs[:, LABEL_TO_ID[label]]

    metrics_df.to_csv(OUT_DIR / "v2_v3_same_test_metrics.csv", index=False, encoding="utf-8-sig")
    per_class_df.to_csv(OUT_DIR / "v2_v3_same_test_per_class_metrics.csv", index=False, encoding="utf-8-sig")
    cm_df.to_csv(OUT_DIR / "v2_v3_same_test_confusion_matrices.csv", index=False, encoding="utf-8-sig")
    ci_df.to_csv(OUT_DIR / "v2_v3_same_test_bootstrap_ci.csv", index=False, encoding="utf-8-sig")
    paired_df.to_csv(OUT_DIR / "v2_v3_paired_bootstrap_delta.csv", index=False, encoding="utf-8-sig")
    mcnemar_df.to_csv(OUT_DIR / "v2_v3_mcnemar.csv", index=False, encoding="utf-8-sig")
    errors_df.to_csv(OUT_DIR / "v2_v3_error_taxonomy.csv", index=False, encoding="utf-8-sig")
    gates_df.to_csv(OUT_DIR / "acceptance_gate_decision.csv", index=False, encoding="utf-8-sig")
    pred_rows.to_csv(OUT_DIR / "v2_v3_same_test_predictions.csv", index=False, encoding="utf-8-sig")

    decision = {
        "status": final_status,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "test_source": REGISTRY.relative_to(ROOT).as_posix(),
        "n_same_test_evaluable_rows": int(len(test)),
        "v2": {
            "artifact": V2_MODEL.relative_to(ROOT).as_posix(),
            "config": V2_CONFIG.relative_to(ROOT).as_posix(),
            "artifact_sha256": sha256_file(V2_MODEL),
            "threshold": threshold_v2,
            "model_config_status": config_v2.get("status", ""),
        },
        "v3": {
            "candidate_dir": V3_DIR.relative_to(ROOT).as_posix(),
            "config": V3_CONFIG.relative_to(ROOT).as_posix(),
            "linear_head_sha256": sha256_file(V3_DIR / "linear_head.safetensors"),
            "threshold": threshold_v3,
            "model_id": config_v3.get("model_id", ""),
            "revision": config_v3.get("revision", ""),
            "training_mode": config_v3.get("training_mode", ""),
        },
        "acceptance_gate": gates_df.to_dict(orient="records"),
        "policy_confirmations": {
            "locked_test_used_for_tuning": False,
            "post_hoc_positive_shift": False,
            "low_confidence_to_positive_rule": False,
            "neutral_to_positive_mass_shift": False,
            "hcc_to_positive_rule": False,
            "ground_truth": "human annotation/adjudication",
        },
    }
    (OUT_DIR / "FINAL_ACCEPTANCE_DECISION.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    (V3_DIR / "acceptance_decision.json").write_text(json.dumps(decision, indent=2), encoding="utf-8")
    print(json.dumps({"status": final_status, "n_test": len(test)}, indent=2))


if __name__ == "__main__":
    main()
