from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.evaluate_rm2_sentiment_indobert_v4_locked_test_once import assert_not_evaluated
from scripts.predict_rm2_sentiment_cpu import model_input as cpu_model_input
from scripts.train_rm2_sentiment_indobert_v4_final import LABELS, model_input as train_model_input


ROOT = Path(__file__).resolve().parents[1]
BASE_MODEL_DIR = ROOT / "artifacts/rm2_sentiment/indobert_v4_final/base_reference"
DEV_REGISTRY = ROOT / "output/rm2_sentiment/validation/human_master_v4/sentiment_v4_development_final_registry.csv"
LOCKED_REGISTRY = ROOT / "output/rm2_sentiment/validation/human_master_v4/sentiment_v4_locked_test_final_frozen.csv"
LOCKED_EVAL_DIR = ROOT / "output/rm2_sentiment/experiments/indobert_v4_final/locked_test_evaluation"
CANONICAL_MODEL = ROOT / "output/rm2_sentiment/final/CANONICAL_MODEL.json"
FINAL_MODEL_DIR = ROOT / "artifacts/rm2_sentiment/indobert_v4_final/final_model"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False, low_memory=False)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_sha256s(path: Path) -> dict[str, str]:
    rows: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            digest, name = line.split(maxsplit=1)
            rows[name.strip()] = digest.strip()
    return rows


def is_true(value: object) -> bool:
    return str(value).strip().lower() == "true"


def nonblank_set(frame: pd.DataFrame, column: str) -> set[str]:
    values = frame[column].astype(str).str.strip()
    return set(values.loc[values.ne("")])


class IndoBertV4FinalApplicationAuditTests(unittest.TestCase):
    def test_base_reference_artifact_checksums_pass(self):
        checksums = parse_sha256s(BASE_MODEL_DIR / "SHA256SUMS.txt")
        self.assertIn("model.safetensors", checksums)
        for name, expected in checksums.items():
            self.assertEqual(sha256_file(BASE_MODEL_DIR / name), expected, name)
        manifest = read_json(BASE_MODEL_DIR / "base_reference_manifest.json")
        self.assertEqual(manifest["checkpoint_sha256"], checksums["model.safetensors"])

    def test_label_mapping_and_prediction_rule_are_fixed(self):
        label_map = read_json(BASE_MODEL_DIR / "label_map.json")
        self.assertEqual(label_map["label_to_id"], {"Negative": 0, "Neutral": 1, "Positive": 2})
        self.assertEqual([label_map["id_to_label"][str(i)] for i in range(3)], LABELS)
        config = read_json(BASE_MODEL_DIR / "config.json")
        self.assertEqual([config["id2label"][str(i)] for i in range(3)], LABELS)
        selected = read_json(BASE_MODEL_DIR / "selected_trial_config.json")
        self.assertEqual(selected["model_id"], "indobenchmark/indobert-base-p2")
        self.assertEqual(selected["prediction_rule"], "argmax_no_threshold_tuning")

    def test_training_and_cpu_inference_preprocessing_match(self):
        row = {
            "brand_or_video_context": "azarine_retinol",
            "product_category": "retinol",
            "model_text": "bagus banget dan cocok",
            "comment_text": "",
        }
        self.assertEqual(
            train_model_input(pd.Series(row), "context_sep_comment"),
            cpu_model_input(row, "model_text", "context_sep_comment"),
        )

    def test_locked_test_one_time_guard_blocks_rerun(self):
        with self.assertRaisesRegex(RuntimeError, "already been evaluated once"):
            assert_not_evaluated(False)

    def test_locked_test_predictions_have_valid_probabilities_and_argmax_labels(self):
        predictions = read_csv(LOCKED_EVAL_DIR / "locked_test_predictions.csv")
        probs = predictions[["prob_negative", "prob_neutral", "prob_positive"]].astype(float).to_numpy()
        self.assertTrue(np.isfinite(probs).all())
        self.assertTrue(np.allclose(probs.sum(axis=1), 1.0, atol=1e-5))
        argmax_labels = [LABELS[int(idx)] for idx in probs.argmax(axis=1)]
        self.assertEqual(argmax_labels, predictions["predicted_label"].tolist())
        self.assertTrue(predictions["predicted_label"].isin(LABELS).all())

    def test_strict_decision_keeps_v2_and_does_not_promote_v4(self):
        decision = read_json(LOCKED_EVAL_DIR / "FINAL_ACCEPTANCE_DECISION.json")
        self.assertEqual(decision["status"], "INDOBERT_V4_NOT_ACCEPTED_KEEP_V2")
        self.assertFalse(decision["accepted"])
        self.assertIn("accuracy_gte_0p8159", decision["failed_criteria"])
        if CANONICAL_MODEL.exists():
            canonical = read_json(CANONICAL_MODEL)
            self.assertNotEqual(canonical["canonical_model"], "indobert_v4_final")
        self.assertFalse(FINAL_MODEL_DIR.exists())

    def test_required_locked_test_report_outputs_exist(self):
        for name in [
            "locked_test_predictions.csv",
            "locked_test_metrics.csv",
            "locked_test_per_class_metrics.csv",
            "locked_test_confusion_matrix.csv",
            "locked_test_classification_report.json",
            "FINAL_ACCEPTANCE_DECISION.json",
            "error_analysis_false_positive.csv",
            "error_analysis_false_negative.csv",
        ]:
            self.assertTrue((LOCKED_EVAL_DIR / name).exists(), name)
        metrics = read_csv(LOCKED_EVAL_DIR / "locked_test_metrics.csv")
        self.assertIn("mcc", set(metrics["metric"]))

    def test_development_locked_split_integrity(self):
        dev_raw = read_csv(DEV_REGISTRY)
        locked_raw = read_csv(LOCKED_REGISTRY)
        dev = dev_raw.loc[dev_raw["evaluable_three_class"].map(is_true) & dev_raw["final_human_label"].isin(LABELS)].copy()
        locked = locked_raw.loc[locked_raw["evaluable_three_class"].map(is_true) & locked_raw["final_human_label"].isin(LABELS)].copy()
        self.assertEqual(len(dev), 1824)
        self.assertEqual(len(locked), 672)
        self.assertFalse(nonblank_set(dev, "comment_id") & nonblank_set(locked, "comment_id"))
        self.assertFalse(nonblank_set(dev, "text_cluster_id") & nonblank_set(locked, "text_cluster_id"))
        self.assertFalse(nonblank_set(dev, "exact_duplicate_group_id") & nonblank_set(locked, "exact_duplicate_group_id"))
        self.assertEqual(int(dev["comment_id"].duplicated().sum()), 0)
        self.assertEqual(int(locked["comment_id"].duplicated().sum()), 0)
        self.assertFalse(dev["comment_id"].astype(str).str.contains("INJ", case=False, regex=False).any())
        self.assertFalse(locked["comment_id"].astype(str).str.contains("INJ", case=False, regex=False).any())

    def test_application_audit_does_not_write_rm1_network_outputs(self):
        source = (ROOT / "scripts/audit_indobert_v4_final_application.py").read_text(encoding="utf-8").lower()
        self.assertNotIn("output/gephi", source)
        self.assertNotIn("notebooks/rm1", source)
        decision = read_json(LOCKED_EVAL_DIR / "FINAL_ACCEPTANCE_DECISION.json")
        self.assertTrue(decision["methodology"]["no_rm1_outputs_modified"])

    def test_rm2_notebook_is_read_only_for_v4_by_default(self):
        notebook_text = (ROOT / "notebooks/rm2/02_rm2_sentiment_analysis.ipynb").read_text(encoding="utf-8")
        self.assertIn("RUN_V4_AUDIT_SCRIPT = False", notebook_text)
        self.assertNotIn("RUN_V4_TRAINING = True", notebook_text)
        self.assertNotIn("RUN_V4_LOCKED_TEST_ONCE = True", notebook_text)
        self.assertIn("no training, no locked-test rerun, no full inference", notebook_text)


if __name__ == "__main__":
    unittest.main()
