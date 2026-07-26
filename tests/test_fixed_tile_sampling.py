from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import build_fixed_tile_dataset as builder
import generate_cctag_dataset as generator
from train_cctag_heatmap_ddp import (
    CCTagHeatmapDataset,
    SCALE_BIN_EDGES,
    ScaleBalancedBatchSampler,
)


def make_dataset() -> CCTagHeatmapDataset:
    dataset = CCTagHeatmapDataset.__new__(CCTagHeatmapDataset)
    dataset.samples = []
    for low, high in zip(SCALE_BIN_EDGES[:-1], SCALE_BIN_EDGES[1:]):
        diameter = (low + high) / 2.0
        dataset.samples.extend(
            {"ellipse_a": diameter / 2.0, "ellipse_b": diameter / 3.0, "is_negative": 0}
            for _ in range(8)
        )
    dataset.samples.extend(
        {"ellipse_a": -1.0, "ellipse_b": -1.0, "is_negative": 1}
        for _ in range(16)
    )
    return dataset


class ScaleBalancedSamplerTest(unittest.TestCase):
    def test_batch_has_fixed_class_and_scale_quotas(self) -> None:
        dataset = make_dataset()
        sampler = ScaleBalancedBatchSampler(
            dataset,
            batch_size=10,
            positive_fraction=0.6,
            num_replicas=1,
            rank=0,
            seed=7,
        )
        batch = next(iter(sampler))
        rows = [dataset.samples[index] for index in batch]
        positives = [row for row in rows if row["is_negative"] == 0]
        self.assertEqual(len(positives), 6)
        self.assertEqual(len(rows) - len(positives), 4)

        counts = [0, 0, 0, 0]
        for row in positives:
            diameter = 2.0 * max(row["ellipse_a"], row["ellipse_b"])
            for i, (low, high) in enumerate(zip(SCALE_BIN_EDGES[:-1], SCALE_BIN_EDGES[1:])):
                if low <= diameter < high:
                    counts[i] += 1
                    break
        self.assertEqual(counts, [2, 2, 1, 1])

    def test_small_batch_integer_quotas_balance_across_epoch(self) -> None:
        dataset = make_dataset()
        sampler = ScaleBalancedBatchSampler(
            dataset,
            batch_size=8,
            positive_fraction=0.6,
            num_replicas=1,
            rank=0,
            seed=7,
        )
        counts = [0, 0, 0, 0]
        for batch in sampler:
            for index in batch:
                row = dataset.samples[index]
                if row["is_negative"]:
                    continue
                diameter = 2.0 * max(row["ellipse_a"], row["ellipse_b"])
                for i, (low, high) in enumerate(
                    zip(SCALE_BIN_EDGES[:-1], SCALE_BIN_EDGES[1:])
                ):
                    if low <= diameter < high:
                        counts[i] += 1
                        break
        self.assertEqual(counts, [9, 9, 6, 6])


class ActualDiameterAcceptanceTest(unittest.TestCase):
    def test_retries_until_transformed_label_diameter_matches(self) -> None:
        image = np.zeros((16, 16), dtype=np.uint8)
        heatmap = np.zeros((4, 4), dtype=np.float32)

        def sample(diameter: float):
            return image, heatmap, {
                "ellipse_a": diameter / 2.0,
                "ellipse_b": diameter / 3.0,
                "target_clamped": 0,
            }

        with patch.object(
            generator,
            "generate_marker_sample",
            side_effect=[sample(44.0), sample(72.0)],
        ):
            _, _, meta = generator.generate_single_sample(
                output_size=(16, 16),
                force_normal_positive=True,
                actual_diameter_range=(48.0, 80.0),
                actual_diameter_max_attempts=2,
            )
        self.assertEqual(2.0 * max(meta["ellipse_a"], meta["ellipse_b"]), 72.0)

    def test_rejects_invisible_low_visibility_wrong_clamp_and_wrong_diameter(self) -> None:
        image = np.zeros((16, 16), dtype=np.uint8)
        heatmap = np.zeros((4, 4), dtype=np.float32)
        def sample(diameter: float, visible: float, clamped: int):
            return image, heatmap, {"ellipse_a": diameter / 2, "ellipse_b": diameter / 3,
                                    "visible_marker_ratio": visible, "target_clamped": clamped}
        accepted = sample(64, 0.50, 1)
        with patch.object(generator, "generate_marker_sample", side_effect=[
            sample(64, 0.0, 1), sample(64, 0.24, 1), sample(64, 0.50, 0),
            sample(40, 0.50, 1), accepted,
        ]):
            _, _, meta = generator.generate_single_sample(
                output_size=(16, 16), actual_diameter_range=(48, 80),
                visible_marker_ratio_range=(0.25, 0.998), required_target_clamped=1,
                actual_diameter_max_attempts=5,
            )
        self.assertIs(meta, accepted[2])

    def test_boundary_acceptance_requires_visible_clamped_partial(self) -> None:
        image = np.zeros((16, 16), dtype=np.uint8); heatmap = np.zeros((4, 4), dtype=np.float32)
        def sample(visible, clamped):
            return image, heatmap, {"ellipse_a": 32, "ellipse_b": 24,
                                    "visible_marker_ratio": visible, "target_clamped": clamped}
        with patch.object(generator, "generate_marker_sample", side_effect=[sample(1.0, 1), sample(.20, 1), sample(.50, 1)]):
            _, _, meta = generator.generate_single_sample(
                output_size=(16, 16), force_boundary_target=True,
                visible_marker_ratio_range=(.25, .998), actual_diameter_max_attempts=3,
            )
        self.assertEqual(meta["target_clamped"], 1)
        self.assertGreaterEqual(meta["visible_marker_ratio"], .25)
        self.assertLess(meta["visible_marker_ratio"], .999)


class MaterializedDatasetSelectionTest(unittest.TestCase):
    def test_selects_exact_scale_and_negative_targets_without_replacement(self) -> None:
        records: list[builder.SourceRow] = []
        source = Path("source")
        for low, high in zip(builder.BIN_EDGES[:-1], builder.BIN_EDGES[1:]):
            diameter = (low + high) / 2.0
            records.extend(
                builder.SourceRow(
                    source,
                    {
                        "filename": f"p_{diameter}_{index}",
                        "ellipse_a": str(diameter / 2.0),
                        "ellipse_b": str(diameter / 3.0),
                        "is_negative": "0",
                        "visible_marker_ratio": "0.5" if index == 0 else "1.0",
                        "target_clamped": "1" if index == 0 else "0",
                        "occlusion_ratio": "0",
                    },
                )
                for index in range(10)
            )
        records.extend(
            builder.SourceRow(
                source,
                {
                    "filename": f"n_{index}",
                    "ellipse_a": "-1",
                    "ellipse_b": "-1",
                    "is_negative": "1",
                },
            )
            for index in range(12)
        )

        selected, targets = builder.select_rows(records, 20, 10, seed=5)
        self.assertEqual(targets, [6, 6, 4, 4])
        self.assertEqual(len(selected), 30)
        self.assertEqual(len({record.row["filename"] for record in selected}), 30)

    def test_rejects_roi_occ_and_invisible_clamped_rows(self) -> None:
        invisible = builder.SourceRow(Path("good"), {"filename": "000055", "ellipse_a": "30", "ellipse_b": "20", "is_negative": "0", "visible_marker_ratio": "0.1", "target_clamped": "1", "occlusion_ratio": "0"})
        self.assertEqual(builder.row_mode(invisible.row), "rejected_low_visibility")
        with self.assertRaises(ValueError):
            builder.load_sources([Path("outputs/datasets/6f_labeled_1024x640_roi_occ")])


if __name__ == "__main__":
    unittest.main()
