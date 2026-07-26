from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from compare_best_checkpoints import diameter_bin, discover_checkpoints, summarize_records


def _row(*, negative: bool, detected: bool, l2: float | None = None) -> dict:
    return {
        "is_negative": int(negative),
        "detected": int(detected),
        "l2_px": l2,
        "success_at_5px": int(not negative and detected and l2 is not None and l2 <= 5),
        "success_at_10px": int(not negative and detected and l2 is not None and l2 <= 10),
    }


class CompareBestCheckpointTests(unittest.TestCase):
    def test_diameter_bin_endpoints_and_gaps(self) -> None:
        self.assertIsNone(diameter_bin(69.999))
        self.assertEqual(diameter_bin(70.0), "70_90")
        self.assertEqual(diameter_bin(89.999), "70_90")
        self.assertEqual(diameter_bin(90.0), "90_110")
        self.assertEqual(diameter_bin(109.999), "90_110")
        self.assertEqual(diameter_bin(110.0), "110_130")
        self.assertEqual(diameter_bin(129.999), "110_130")
        self.assertEqual(diameter_bin(130.0), "130_150")
        self.assertEqual(diameter_bin(150.0), "130_150")
        self.assertIsNone(diameter_bin(150.001))

    def test_missed_positive_counts_as_success_failure(self) -> None:
        metrics = summarize_records([
            _row(negative=False, detected=True, l2=3.0),
            _row(negative=False, detected=True, l2=8.0),
            _row(negative=False, detected=False),
        ])
        self.assertEqual(metrics["detection_recall"], 2 / 3)
        self.assertEqual(metrics["success_at_5px"], 1 / 3)
        self.assertEqual(metrics["success_at_10px"], 2 / 3)
        self.assertEqual(metrics["l2_detected_count"], 2)

    def test_negative_fpr_and_detection_f1(self) -> None:
        metrics = summarize_records([
            _row(negative=False, detected=True, l2=2.0),
            _row(negative=False, detected=False),
            _row(negative=True, detected=True),
            _row(negative=True, detected=False),
        ])
        self.assertEqual(metrics["negative_fpr"], 0.5)
        self.assertEqual(metrics["detection_precision"], 0.5)
        self.assertEqual(metrics["detection_recall"], 0.5)
        self.assertEqual(metrics["detection_f1"], 0.5)

    def test_recursive_checkpoint_names_are_relative_and_unique(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            (tmp_path / "run_a").mkdir()
            (tmp_path / "group" / "finetune").mkdir(parents=True)
            (tmp_path / "run_a" / "best.pt").write_bytes(b"a")
            (tmp_path / "group" / "finetune" / "best.pt").write_bytes(b"b")
            (tmp_path / "group" / "finetune" / "last.pt").write_bytes(b"ignored")
            found = discover_checkpoints(tmp_path)
            self.assertEqual([name for name, _ in found], ["group/finetune", "run_a"])
            self.assertEqual(len({name for name, _ in found}), 2)


if __name__ == "__main__":
    unittest.main()
