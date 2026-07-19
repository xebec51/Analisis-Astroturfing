import unittest

import pandas as pd

from scripts.rm2_sentiment_goals_pipeline import (
    annotate_heuristic_reference_sample,
    deterministic_rule_reproducibility,
    heuristic_label_for_text,
)


class HeuristicPseudoLabelingTests(unittest.TestCase):
    def test_same_pass_is_deterministic(self):
        text = "harganya berapa kak?"
        first = heuristic_label_for_text(text, pass_id=1)
        second = heuristic_label_for_text(text, pass_id=1)
        self.assertEqual(first, second)

    def test_two_pass_report_is_not_independence_evidence(self):
        sample = pd.DataFrame(
            [
                {
                    "sample_set": "development",
                    "comment_id": "c1",
                    "comment_text_original": "bagus banget aku cocok",
                    "video_id": "v1",
                    "product_category": "Azarine",
                    "is_hcc": False,
                    "hcc_id": "Non-HCC",
                    "brand_label_auto": "Not identified",
                    "sampling_stratum": "Non-HCC|Positive|standard",
                    "sample_probability": 1.0,
                    "sample_weight": 1.0,
                },
                {
                    "sample_set": "development",
                    "comment_id": "c2",
                    "comment_text_original": "harganya berapa?",
                    "video_id": "v1",
                    "product_category": "Azarine",
                    "is_hcc": False,
                    "hcc_id": "Non-HCC",
                    "brand_label_auto": "Not identified",
                    "sampling_stratum": "Non-HCC|Neutral|question",
                    "sample_probability": 1.0,
                    "sample_weight": 1.0,
                },
            ]
        )
        annotated = annotate_heuristic_reference_sample(sample)
        reproducibility = deterministic_rule_reproducibility(annotated, annotated)
        self.assertFalse(reproducibility["passes_independent"].any())
        self.assertFalse(reproducibility["use_as_validation_evidence"].any())


if __name__ == "__main__":
    unittest.main()
