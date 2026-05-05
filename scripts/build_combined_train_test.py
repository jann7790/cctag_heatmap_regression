"""
Split synthetic and real-world datasets, then merge them into final train/test sets.

Usage:
    uv run python scripts/build_combined_train_test.py \
        --synthetic_src outputs/training_sets/stride4_v2/mixed_train_dataset \
        --real_src outputs/real_world_stride4 \
        --output_root outputs/combined_stride4_v1 \
        --test_ratio 0.15 \
        --seed 42
"""

from __future__ import annotations

import argparse
from pathlib import Path

from merge_datasets import merge_datasets, summarize_rows
from split_train_test import split_dataset


def print_summary(label: str, rows: list[dict[str, str]]) -> None:
    positives, negatives = summarize_rows(rows)
    total = len(rows)
    pos_ratio = positives / total * 100 if total else 0.0
    print(
        f"{label}: {total}  [pos={positives}, neg={negatives}, pos_ratio={pos_ratio:.1f}%]"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic_src", required=True, type=Path)
    parser.add_argument("--real_src", required=True, type=Path)
    parser.add_argument("--output_root", required=True, type=Path)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    synthetic_train_dir = args.output_root / "synthetic_train"
    synthetic_test_dir = args.output_root / "synthetic_test"
    real_train_dir = args.output_root / "real_train"
    real_test_dir = args.output_root / "real_test"
    final_train_dir = args.output_root / "train"
    final_test_dir = args.output_root / "test"

    synthetic_train_rows, synthetic_test_rows = split_dataset(
        src_dir=args.synthetic_src,
        train_dir=synthetic_train_dir,
        test_dir=synthetic_test_dir,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    real_train_rows, real_test_rows = split_dataset(
        src_dir=args.real_src,
        train_dir=real_train_dir,
        test_dir=real_test_dir,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    final_train_rows = merge_datasets(
        src_dirs=[synthetic_train_dir, real_train_dir],
        output_dir=final_train_dir,
    )
    final_test_rows = merge_datasets(
        src_dirs=[synthetic_test_dir, real_test_dir],
        output_dir=final_test_dir,
    )

    print_summary("Synthetic train", synthetic_train_rows)
    print_summary("Synthetic test ", synthetic_test_rows)
    print_summary("Real train     ", real_train_rows)
    print_summary("Real test      ", real_test_rows)
    print_summary("Final train    ", final_train_rows)
    print_summary("Final test     ", final_test_rows)
    print(f"Done -> {args.output_root}")


if __name__ == "__main__":
    main()
