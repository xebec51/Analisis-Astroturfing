from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.train_rm2_sentiment_indobert_v4_final import (
    build_trials,
    select_only_trial_id,
    sha256_file,
    write_sha256s,
)


class RM2SentimentIndoBertV4TrainingPipelineTests(unittest.TestCase):
    def test_only_trial_id_selects_exactly_one_trial(self):
        trials = build_trials(["indobenchmark/indobert-base-p2"], "full")
        trial_id = "indobenchmark__indobert-base-p2__context_sep_comment__len256__lr3p0em05__warm0p06__wd0p05__drop0p3__focal_loss__ls0p05"
        selected = select_only_trial_id(trials, trial_id)
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].trial_id, trial_id)

    def test_only_trial_id_errors_when_missing(self):
        trials = build_trials(["indobenchmark/indobert-base-p2"], "full")
        with self.assertRaises(ValueError):
            select_only_trial_id(trials, "missing_trial")

    def test_write_sha256s_excludes_checksum_file_itself(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            payload = path / "payload.txt"
            payload.write_text("sentiment\n", encoding="utf-8")
            hashes = write_sha256s(path)
            self.assertEqual(hashes, {"payload.txt": sha256_file(payload)})
            sums = (path / "SHA256SUMS.txt").read_text(encoding="utf-8")
            self.assertIn("payload.txt", sums)
            self.assertNotIn("SHA256SUMS.txt", sums)


if __name__ == "__main__":
    unittest.main()
