from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.predict_rm2_sentiment_cpu import assert_not_lfs_pointer, sha256_file, verify_model_dir


class PredictRM2SentimentCPUTests(unittest.TestCase):
    def test_lfs_pointer_checkpoint_gets_clear_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "model.safetensors"
            checkpoint.write_text(
                "version https://git-lfs.github.com/spec/v1\n"
                "oid sha256:abc\n"
                "size 123\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "Git LFS pointer"):
                assert_not_lfs_pointer(checkpoint)

    def test_verify_model_dir_detects_checksum_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint = root / "model.safetensors"
            checkpoint.write_bytes(b"x" * (1024 * 1024 + 1))
            (root / "SHA256SUMS.txt").write_text(f"{sha256_file(checkpoint)}  model.safetensors\n", encoding="utf-8")
            checkpoint.write_bytes(b"y" * (1024 * 1024 + 1))
            with self.assertRaisesRegex(RuntimeError, "Checksum mismatch"):
                verify_model_dir(root)


if __name__ == "__main__":
    unittest.main()
