from __future__ import annotations

import json
import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEV = ROOT / "output/rm2_sentiment/validation/human_master_v4/sentiment_v4_development_final_registry.csv"
LOCKED = ROOT / "output/rm2_sentiment/validation/human_master_v4/sentiment_v4_locked_test_final_frozen.csv"
EXP_DIR = ROOT / "output/rm2_sentiment/experiments/indobert_v4_final"
MODEL_DIR = ROOT / "output/rm2_sentiment/model/indobert_v4_final_candidate"
LOCKED_EVAL_DIR = EXP_DIR / "locked_test_evaluation"

LABELS = ["Negative", "Neutral", "Positive"]
EVALUABLE = set(LABELS)


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False, low_memory=False)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def is_true(value: object) -> bool:
    return str(value).strip().lower() == "true"


def nonempty_set(frame: pd.DataFrame, column: str) -> set[str]:
    values = frame[column].astype(str).str.strip()
    return set(values.loc[values.ne("")])


class RM2SentimentIndoBertV4FinalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dev_raw = read_csv(DEV)
        cls.locked_raw = read_csv(LOCKED)
        cls.dev = cls.dev_raw.loc[
            cls.dev_raw["evaluable_three_class"].map(is_true) & cls.dev_raw["final_human_label"].isin(EVALUABLE)
        ].copy()
        cls.locked = cls.locked_raw.loc[
            cls.locked_raw["evaluable_three_class"].map(is_true) & cls.locked_raw["final_human_label"].isin(EVALUABLE)
        ].copy()

    def test_development_data_only_evaluable_three_class(self):
        self.assertEqual(len(self.dev), 1824)
        self.assertEqual(self.dev["final_human_label"].value_counts().reindex(LABELS).fillna(0).astype(int).to_dict(), {
            "Negative": 470,
            "Neutral": 788,
            "Positive": 566,
        })
        self.assertTrue(self.dev["evaluable_three_class"].map(is_true).all())
        self.assertTrue(self.dev["final_human_label"].isin(EVALUABLE).all())

    def test_uncertain_no_text_not_in_training_or_evaluation_denominator(self):
        self.assertEqual(int(self.dev_raw["final_human_label"].eq("Uncertain").sum()), 22)
        self.assertEqual(int(self.dev_raw["final_human_label"].eq("No Text").sum()), 0)
        self.assertEqual(len(self.dev_raw) - len(self.dev), 22)
        self.assertEqual(len(self.locked), 672)
        self.assertEqual(int(self.locked_raw["final_human_label"].eq("Uncertain").sum()), 25)
        self.assertEqual(int(self.locked_raw["final_human_label"].eq("No Text").sum()), 3)
        self.assertEqual(len(self.locked_raw) - len(self.locked), 28)
        predictions = read_csv(LOCKED_EVAL_DIR / "locked_test_predictions.csv")
        self.assertEqual(len(predictions), 672)
        self.assertTrue(predictions["final_human_label"].isin(EVALUABLE).all())

    def test_locked_test_not_in_training_or_fold_selection(self):
        manifest = read_json(EXP_DIR / "INDOBERT_V4_TRAINING_MANIFEST.json")
        self.assertFalse(manifest["locked_test_used_for_training_or_selection"])
        self.assertFalse(manifest["locked_test_used_for_early_stopping"])
        self.assertFalse(manifest["locked_test_used_for_threshold_selection"])
        assignments = read_csv(EXP_DIR / "development_fold_assignments.csv")
        oof = read_csv(EXP_DIR / "development_oof_predictions.csv")
        locked_ids = set(self.locked["comment_id"])
        self.assertFalse(set(assignments["comment_id"]) & locked_ids)
        self.assertFalse(set(oof["comment_id"]) & locked_ids)

    def test_comment_id_and_text_cluster_leakage_zero_across_split_and_folds(self):
        self.assertFalse(nonempty_set(self.dev, "comment_id") & nonempty_set(self.locked, "comment_id"))
        self.assertFalse(nonempty_set(self.dev, "text_cluster_id") & nonempty_set(self.locked, "text_cluster_id"))
        self.assertFalse(nonempty_set(self.dev, "exact_duplicate_group_id") & nonempty_set(self.locked, "exact_duplicate_group_id"))
        assignments = read_csv(EXP_DIR / "development_fold_assignments.csv")
        for seed, group in assignments.groupby("seed"):
            self.assertEqual(int((group.groupby("comment_id")["fold"].nunique() > 1).sum()), 0, seed)
            text_group = group.loc[group["text_cluster_id"].astype(str).str.strip().ne("")]
            self.assertEqual(int((text_group.groupby("text_cluster_id")["fold"].nunique() > 1).sum()), 0, seed)
            cv_group = group.loc[group["cv_group_id"].astype(str).str.strip().ne("")]
            self.assertEqual(int((cv_group.groupby("cv_group_id")["fold"].nunique() > 1).sum()), 0, seed)

    def test_label_vocabulary_only_three_class(self):
        label_map = read_json(MODEL_DIR / "label_map.json")
        self.assertEqual(label_map["label_to_id"], {"Negative": 0, "Neutral": 1, "Positive": 2})
        self.assertEqual([label_map["id_to_label"][str(i)] for i in range(3)], LABELS)
        selected = read_json(MODEL_DIR / "selected_trial_config.json")
        self.assertEqual(selected["label_source_column"], "final_human_label")
        self.assertEqual(set(selected["forbidden_supervision_sources"]), {
            "sentiment_v2_prediction",
            "model_prediction",
            "lexicon",
            "hcc",
            "actor_type",
            "goal_orientation",
            "llm",
            "locked_test",
        })

    def test_model_artifact_and_tokenizer_saved(self):
        self.assertTrue((MODEL_DIR / "model.safetensors").exists() or (MODEL_DIR / "pytorch_model.bin").exists())
        self.assertTrue((MODEL_DIR / "config.json").exists())
        self.assertTrue((MODEL_DIR / "tokenizer_config.json").exists())
        self.assertTrue((MODEL_DIR / "special_tokens_map.json").exists())
        self.assertTrue((MODEL_DIR / "selected_trial_config.json").exists())
        self.assertTrue((MODEL_DIR / "INDOBERT_V4_TRAINING_MANIFEST.json").exists())

    def test_training_manifest_locks_methodology(self):
        manifest = read_json(EXP_DIR / "INDOBERT_V4_TRAINING_MANIFEST.json")
        self.assertEqual(manifest["label_source_column"], "final_human_label")
        self.assertEqual(manifest["label_vocabulary"], LABELS)
        self.assertFalse(manifest["prediction_label_source_used"])
        self.assertFalse(manifest["locked_test_used_for_training_or_selection"])
        self.assertFalse(manifest["full_corpus_inference_run"])
        self.assertEqual(manifest["development_evaluable_rows"], 1824)

    def test_locked_test_evaluation_manifest_evaluated_once(self):
        manifest = read_json(LOCKED_EVAL_DIR / "LOCKED_TEST_EVALUATION_MANIFEST.json")
        self.assertTrue(manifest["evaluated_once"])
        self.assertFalse(manifest["locked_test_used_for_training_or_selection"])
        self.assertFalse(manifest["locked_test_used_for_threshold_selection"])
        self.assertEqual(manifest["locked_test_evaluable_rows"], 672)
        self.assertEqual(manifest["locked_test_class_counts"], {
            "Negative": 160,
            "Neutral": 294,
            "Positive": 218,
        })

    def test_no_model_prediction_used_as_label(self):
        training_manifest = read_json(EXP_DIR / "INDOBERT_V4_TRAINING_MANIFEST.json")
        eval_manifest = read_json(LOCKED_EVAL_DIR / "LOCKED_TEST_EVALUATION_MANIFEST.json")
        self.assertEqual(training_manifest["label_source_column"], "final_human_label")
        self.assertFalse(training_manifest["prediction_label_source_used"])
        self.assertFalse(eval_manifest["prediction_label_source_used"])
        source = (ROOT / "scripts/train_rm2_sentiment_indobert_v4_final.py").read_text(encoding="utf-8").lower()
        self.assertNotIn("pseudo_label", source)
        for token in ["sentiment_v2_prediction", "model_prediction", "lexicon"]:
            self.assertNotIn(f"label_source_column\": \"{token}", source)


if __name__ == "__main__":
    unittest.main()
