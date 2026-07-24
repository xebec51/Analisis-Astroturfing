from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from scripts.rm2_sentiment_v2_positive_recall import (
    LABELS,
    LABEL_TO_ID,
    apply_threshold_policy,
    artifact_predict_proba,
    load_candidate_artifact,
    normalize_probabilities,
    predict_labels_from_artifact,
)


ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_MODEL = ROOT / "output/rm2_sentiment/model/v2_positive_recall_candidate/selected_model.joblib"
CANDIDATE_CONFIG = ROOT / "output/rm2_sentiment/model/v2_positive_recall_candidate/selected_model_config.json"
REGISTRY = ROOT / "output/rm2_sentiment/validation/human_v2_positive_recall/human_label_registry.csv"
FOLD_AUDIT = ROOT / "output/rm2_sentiment/validation/human_v2_positive_recall/fold_leakage_audit.csv"
LOCKED_TEMPLATE = ROOT / "output/rm2_sentiment/validation/human_v2_positive_recall/new_locked_test_final.csv"
BASELINE_CM = ROOT / "output/rm2_sentiment/experiments/v2_positive_recall/baseline_v2_confusion_matrix.csv"
DEV_MANIFEST = ROOT / "output/rm2_sentiment/experiments/v2_positive_recall/V2_POSITIVE_RECALL_DEVELOPMENT_MANIFEST.json"
ACCEPTANCE = ROOT / "output/rm2_sentiment/model/v2_positive_recall_candidate/acceptance_decision.json"


class SentimentV2PositiveRecallTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.artifact = load_candidate_artifact(CANDIDATE_MODEL)
        cls.config = json.loads(CANDIDATE_CONFIG.read_text(encoding="utf-8"))

    def test_label_mapping_is_correct(self):
        self.assertEqual(LABELS, ["Negative", "Neutral", "Positive"])
        self.assertEqual(LABEL_TO_ID, {"Negative": 0, "Neutral": 1, "Positive": 2})

    def test_probabilities_sum_to_one(self):
        probs = normalize_probabilities(np.array([[2.0, 1.0, 1.0], [0.0, 0.0, 0.0]]))
        np.testing.assert_allclose(probs.sum(axis=1), np.ones(2))

    def test_anchor_positive_predicted_positive(self):
        pred = predict_labels_from_artifact(self.artifact, pd.Series(["bagus banget cocok di aku"]))
        self.assertEqual(pred["predicted_label"].iloc[0], "Positive")

    def test_anchor_neutral_predicted_neutral(self):
        pred = predict_labels_from_artifact(self.artifact, pd.Series(["harganya berapa kak?"]))
        self.assertEqual(pred["predicted_label"].iloc[0], "Neutral")

    def test_anchor_negative_predicted_negative(self):
        pred = predict_labels_from_artifact(self.artifact, pd.Series(["bikin muka gatal dan jerawatan"]))
        self.assertEqual(pred["predicted_label"].iloc[0], "Negative")

    def test_no_text_remains_no_text(self):
        pred = predict_labels_from_artifact(self.artifact, pd.Series([""]))
        self.assertEqual(pred["predicted_label"].iloc[0], "No Text")

    def test_low_confidence_becomes_uncertain(self):
        pred = apply_threshold_policy(
            np.array([[0.34, 0.33, 0.33]]),
            positive_threshold=0.50,
            margin_positive_neutral=0.0,
            margin_positive_negative=0.0,
            abstention_threshold=0.40,
        )
        self.assertEqual(pred["predicted_label"].iloc[0], "Uncertain")

    def test_low_confidence_is_not_automatic_positive(self):
        pred = apply_threshold_policy(
            np.array([[0.33, 0.34, 0.33]]),
            positive_threshold=0.30,
            margin_positive_neutral=0.0,
            margin_positive_negative=0.0,
            abstention_threshold=0.50,
        )
        self.assertNotEqual(pred["predicted_label"].iloc[0], "Positive")

    def test_no_inj_in_training_or_new_locked_template(self):
        registry = pd.read_csv(REGISTRY, dtype=str, keep_default_na=False, low_memory=False)
        dev = registry.loc[registry["split_family"].eq("development")]
        locked = pd.read_csv(LOCKED_TEMPLATE, dtype=str, keep_default_na=False, low_memory=False)
        self.assertFalse(dev["comment_id"].str.upper().str.startswith("INJ").any())
        self.assertFalse(locked["comment_id"].str.upper().str.startswith("INJ").any())

    def test_duplicate_does_not_cross_hard_split(self):
        audit = pd.read_csv(FOLD_AUDIT, dtype=str, keep_default_na=False, low_memory=False)
        hard = audit.loc[audit["audit_type"].isin(["text_cluster_id_cross_fold", "cv_group_id_cross_fold"])]
        self.assertFalse(hard["status"].eq("FAIL").any())

    def test_threshold_only_from_development(self):
        self.assertEqual(self.config["threshold_selection_data_scope"], "OOF_DEVELOPMENT_ONLY")
        self.assertFalse(self.config["locked_test_used_for_training_or_selection"])

    def test_locked_test_not_accessed_during_training(self):
        manifest = json.loads(DEV_MANIFEST.read_text(encoding="utf-8"))
        acceptance = json.loads(ACCEPTANCE.read_text(encoding="utf-8"))
        self.assertFalse(manifest["locked_test_used_for_training_or_selection"])
        self.assertFalse(acceptance["gates"]["full_inference_allowed"])

    def test_model_can_serialize_and_reload(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "candidate.joblib"
            joblib.dump(self.artifact, path)
            reloaded = joblib.load(path)
        self.assertEqual(reloaded["threshold_policy"], self.artifact["threshold_policy"])

    def test_prediction_consistent_after_reload(self):
        texts = pd.Series(["bagus banget cocok di aku", "harganya berapa kak?"])
        before = artifact_predict_proba(self.artifact, texts)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "candidate.joblib"
            joblib.dump(self.artifact, path)
            after = artifact_predict_proba(joblib.load(path), texts)
        np.testing.assert_allclose(before, after)

    def test_confusion_matrix_label_order(self):
        cm = pd.read_csv(BASELINE_CM, dtype=str, keep_default_na=False)
        self.assertEqual(cm["true_label"].tolist(), LABELS)
        self.assertEqual(cm.columns.tolist(), ["true_label", *LABELS])


if __name__ == "__main__":
    unittest.main()
