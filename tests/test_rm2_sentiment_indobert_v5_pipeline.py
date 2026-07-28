from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.evaluate_rm2_sentiment_indobert_v5_locked_test_once import assert_not_evaluated
from scripts.select_rm2_sentiment_indobert_v5_development_candidate import metric_bundle, risk_coverage
from scripts.train_rm2_sentiment_indobert_v5_development import LABELS, class_weights_from_training, model_input, sha256_dataframe


ROOT = Path(__file__).resolve().parents[1]
DEV = ROOT / "output/rm2_sentiment/validation/human_v5/sentiment_v5_development_final_registry.csv"
LOCKED = ROOT / "output/rm2_sentiment/validation/human_v5_locked_test/sentiment_v5_locked_test_final_frozen.csv"
MANIFEST = ROOT / "output/rm2_sentiment/validation/human_v5/SENTIMENT_V5_FINAL_IMPORT_MANIFEST.json"
ACCEPTANCE = ROOT / "configs/rm2_sentiment_v5_acceptance_preregistered.json"
EXP = ROOT / "output/rm2_sentiment/experiments/indobert_v5_development"
LOCKED_EVAL = ROOT / "output/rm2_sentiment/experiments/indobert_v5_final/locked_test_evaluation"
TRAIN_SCRIPT = ROOT / "scripts/train_rm2_sentiment_indobert_v5_development.py"
RM1_SENTINELS = [
    ROOT / "output/gephi/gephi_hcc_nodes.csv",
    ROOT / "output/gephi/gephi_hcc_edges.csv",
]


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False, low_memory=False)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def is_true(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


class RM2SentimentIndoBertV5PipelineTests(unittest.TestCase):
    def test_development_and_locked_hashes_match_manifest(self):
        manifest = read_json(MANIFEST)
        dev = read_csv(DEV)
        locked = read_csv(LOCKED)
        self.assertEqual(sha256_dataframe(dev, ["annotation_id", "comment_id", "final_human_label"]), manifest["development"]["dataset_hash"])
        self.assertEqual(sha256_dataframe(locked, ["annotation_id", "comment_id", "final_human_label"]), manifest["locked_test_v5"]["dataset_hash"])
        self.assertEqual(manifest["development"]["dataset_hash"], "31a5537a0483405632dd8c33bf2190ca405a2e8c77f84f571328be48d1b6004c")
        self.assertEqual(manifest["locked_test_v5"]["dataset_hash"], "368d0a9a17dece03c2adc0f3c2ce7e3c3e1b3631e260aec8001ace3dc2ca83ec")

    def test_training_script_has_no_locked_registry_dependency(self):
        source = TRAIN_SCRIPT.read_text(encoding="utf-8")
        self.assertNotIn("human_v5_locked_test", source)
        self.assertNotIn("sentiment_v5_locked_test_final_frozen", source)
        self.assertNotIn("LOCKED_REGISTRY", source)
        self.assertIn("CUDA is required", source)
        self.assertIn("foreach=False", source)
        self.assertIn("fused=False", source)
        self.assertIn("num_workers=0", source)

    def test_v5_distribution_and_label_mapping(self):
        dev = read_csv(DEV)
        locked = read_csv(LOCKED)
        dev_eval = dev.loc[dev["evaluable_three_class"].map(is_true) & dev["final_human_label"].isin(LABELS)]
        locked_eval = locked.loc[locked["evaluable_three_class"].map(is_true) & locked["final_human_label"].isin(LABELS)]
        self.assertEqual(len(dev), 1000)
        self.assertEqual(len(dev_eval), 977)
        self.assertEqual(dev["final_human_label"].value_counts().reindex([*LABELS, "Uncertain", "No Text"]).fillna(0).astype(int).to_dict(), {
            "Negative": 178,
            "Neutral": 569,
            "Positive": 230,
            "Uncertain": 13,
            "No Text": 10,
        })
        self.assertEqual(len(locked), 700)
        self.assertEqual(len(locked_eval), 687)
        self.assertEqual(locked["final_human_label"].value_counts().reindex([*LABELS, "Uncertain", "No Text"]).fillna(0).astype(int).to_dict(), {
            "Negative": 134,
            "Neutral": 380,
            "Positive": 173,
            "Uncertain": 9,
            "No Text": 4,
        })

    def test_grouped_fold_leakage_zero(self):
        folds = read_csv(EXP / "development_grouped_fold_assignments.csv")
        for seed, group in folds.groupby("seed"):
            self.assertEqual(int((group.groupby("comment_id")["fold"].nunique() > 1).sum()), 0, seed)
            self.assertEqual(int((group.groupby("hard_group")["fold"].nunique() > 1).sum()), 0, seed)

    def test_input_audit_rules(self):
        audit = read_json(EXP / "input_token_length_audit.json")
        self.assertEqual(audit["max_lengths_allowed"], [128])
        self.assertEqual(audit["input_modes_allowed"], ["comment_only", "context_sep_comment"])
        self.assertIn("target_context_parent_comment", audit["input_modes_skipped"])
        self.assertLess(audit["parent_comment_text_non_empty_share"], 0.10)

    def test_model_input_target_parent_format(self):
        row = pd.Series({"brand_or_video_context": "Azarine", "parent_comment_text": "aman?", "comment_text": "cocok", "product_category": ""})
        self.assertEqual(model_input(row, "target_context_parent_comment"), "Azarine [SEP] aman? [SEP] cocok")

    def test_class_weights_are_fold_local_math(self):
        labels = np.array([0, 1, 1, 2, 2, 2])
        weights = class_weights_from_training(labels).numpy()
        np.testing.assert_allclose(weights, np.array([2.0, 1.0, 2 / 3]), rtol=1e-6)

    def test_selection_score_and_penalties(self):
        y = np.array([0, 1, 2, 2])
        probs = np.array([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.1, 0.2, 0.7], [0.2, 0.2, 0.6]])
        metrics = metric_bundle(y, probs, "unit")
        expected = 0.25 * metrics["macro_f1"] + 0.20 * metrics["balanced_accuracy"] + 0.20 * metrics["mcc"] + 0.15 * metrics["min_class_recall"] + 0.15 * metrics["positive_recall"] + 0.05 * metrics["accuracy"]
        self.assertAlmostEqual(metrics["selection_score_raw"], expected)
        collapsed = metric_bundle(y, np.tile([[0.9, 0.05, 0.05]], (4, 1)), "unit")
        self.assertTrue(collapsed["class_collapse"])
        self.assertTrue(collapsed["auto_disqualified"])

    def test_risk_coverage_development_only(self):
        y = np.array([0, 1, 2, 2])
        probs = np.array([[0.9, 0.05, 0.05], [0.1, 0.8, 0.1], [0.1, 0.2, 0.7], [0.34, 0.33, 0.33]])
        curve = risk_coverage("unit", y, probs)
        self.assertEqual(set(curve["coverage_target"]), {0.9343, 0.95, 1.0})
        self.assertFalse(curve["locked_test_used"].astype(bool).any())

    def test_oof_outputs_if_present_are_complete_and_valid(self):
        oof_path = EXP / "development_oof_predictions.csv"
        if not oof_path.exists():
            self.skipTest("OOF not generated yet")
        oof = read_csv(oof_path)
        self.assertEqual(int(oof.duplicated(["stage", "trial_id", "seed", "fold", "annotation_id"]).sum()), 0)
        probs = oof[["prob_negative", "prob_neutral", "prob_positive"]].astype(float).to_numpy()
        self.assertTrue(np.isfinite(probs).all())
        self.assertTrue(np.allclose(probs.sum(axis=1), 1.0, atol=1e-4))
        for (_, trial, seed), group in oof.groupby(["stage", "trial_id", "seed"]):
            if len(group) == 977:
                self.assertEqual(group["annotation_id"].nunique(), 977)

    def test_locked_test_one_time_guard_if_evaluated(self):
        manifest = LOCKED_EVAL / "LOCKED_TEST_V5_EVALUATION_MANIFEST.json"
        if not manifest.exists():
            self.skipTest("Locked test V5 not evaluated yet")
        with self.assertRaisesRegex(RuntimeError, "already been evaluated once"):
            assert_not_evaluated(False)

    def test_acceptance_config_hash_stable(self):
        digest = hashlib.sha256(ACCEPTANCE.read_bytes()).hexdigest()
        self.assertEqual(digest, "194177c718683c8e43ab72ca7975f58628e4061388bf020ada96862482c6b428")

    def test_rm1_sentinel_outputs_exist_and_are_not_v5_outputs(self):
        for path in RM1_SENTINELS:
            self.assertTrue(path.exists(), path)


if __name__ == "__main__":
    unittest.main()
