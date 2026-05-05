"""
Split a dataset into train and test sets by stratified sampling on `is_negative`.

Usage:
    uv run python scripts/split_train_test.py \
        --src outputs/real_world_stride4 \
        --train_dir outputs/real_world_stride4_train \
        --test_dir outputs/real_world_stride4_test \
        --test_ratio 0.15 \
        --seed 42
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from pathlib import Path
from typing import Iterable


OPTIONAL_ARTIFACT_DIRS = {
    "images": ".png",
    "heatmaps": ".npy",
    "labels_yolo": ".txt",
}


def read_labels(src_dir: Path) -> tuple[list[dict[str, str]], list[str]]:
    labels_path = src_dir / "labels.csv"
    with labels_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        header = list(reader.fieldnames or [])
    if not rows or not header:
        raise ValueError(f"No rows found in {labels_path}")
    return rows, header


def load_dataset_config(src_dir: Path) -> dict:
    config_path = src_dir / "config.json"
    if config_path.exists():
        return json.loads(config_path.read_text())

    config_parts_dir = src_dir / "config_parts"
    if config_parts_dir.is_dir():
        config_candidates = sorted(config_parts_dir.glob("*.json"))
        if config_candidates:
            return json.loads(config_candidates[0].read_text())

    raise FileNotFoundError(
        f"Could not find config.json or any config_parts/*.json under {src_dir}"
    )


def ensure_output_dirs(*dirs: Path, overwrite: bool = False) -> None:
    for path in dirs:
        if path.exists() and any(path.iterdir()):
            if not overwrite:
                raise FileExistsError(f"Refusing to write into non-empty directory: {path}")
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)


def stratified_split(
    rows: list[dict[str, str]],
    test_ratio: float,
    seed: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    if not 0.0 < test_ratio < 1.0:
        raise ValueError(f"test_ratio must be between 0 and 1, got {test_ratio}")

    rng = random.Random(seed)

    positives = [row for row in rows if row.get("is_negative", "0").strip() != "1"]
    negatives = [row for row in rows if row.get("is_negative", "0").strip() == "1"]

    def split_group(group_rows: list[dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
        if not group_rows:
            return [], []
        shuffled = list(group_rows)
        rng.shuffle(shuffled)
        n_test = round(len(shuffled) * test_ratio)
        if n_test == 0 and len(shuffled) > 1:
            n_test = 1
        if n_test == len(shuffled) and len(shuffled) > 1:
            n_test -= 1
        return shuffled[n_test:], shuffled[:n_test]

    train_pos, test_pos = split_group(positives)
    train_neg, test_neg = split_group(negatives)
    train_rows = train_pos + train_neg
    test_rows = test_pos + test_neg
    rng.shuffle(train_rows)
    rng.shuffle(test_rows)
    return train_rows, test_rows


def copy_artifacts(rows: Iterable[dict[str, str]], src_dir: Path, out_dir: Path) -> None:
    active_artifact_dirs = {
        dirname: suffix
        for dirname, suffix in OPTIONAL_ARTIFACT_DIRS.items()
        if (src_dir / dirname).is_dir()
    }

    for dirname in active_artifact_dirs:
        (out_dir / dirname).mkdir(parents=True, exist_ok=True)

    for row in rows:
        filename = row["filename"]
        for dirname, suffix in active_artifact_dirs.items():
            src_path = src_dir / dirname / f"{filename}{suffix}"
            if not src_path.exists():
                raise FileNotFoundError(f"Missing expected artifact: {src_path}")
            shutil.copy2(src_path, out_dir / dirname / src_path.name)


def write_split_dataset(
    rows: list[dict[str, str]],
    header: list[str],
    config: dict,
    src_dir: Path,
    out_dir: Path,
) -> None:
    copy_artifacts(rows, src_dir, out_dir)

    with (out_dir / "labels.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)

    (out_dir / "config.json").write_text(json.dumps(config, indent=2))


def summarize_rows(rows: list[dict[str, str]]) -> tuple[int, int]:
    negatives = sum(1 for row in rows if row.get("is_negative", "0").strip() == "1")
    positives = len(rows) - negatives
    return positives, negatives


def split_dataset(
    src_dir: Path,
    train_dir: Path,
    test_dir: Path,
    test_ratio: float = 0.15,
    seed: int = 42,
    overwrite: bool = False,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    rows, header = read_labels(src_dir)
    config = load_dataset_config(src_dir)

    ensure_output_dirs(train_dir, test_dir, overwrite=overwrite)
    train_rows, test_rows = stratified_split(rows, test_ratio=test_ratio, seed=seed)
    write_split_dataset(train_rows, header, config, src_dir, train_dir)
    write_split_dataset(test_rows, header, config, src_dir, test_dir)
    return train_rows, test_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, type=Path)
    parser.add_argument("--train_dir", required=True, type=Path)
    parser.add_argument("--test_dir", required=True, type=Path)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite non-empty output directories if they already exist.",
    )
    args = parser.parse_args()

    train_rows, test_rows = split_dataset(
        src_dir=args.src,
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        test_ratio=args.test_ratio,
        seed=args.seed,
        overwrite=args.overwrite,
    )

    total = len(train_rows) + len(test_rows)
    train_pos, train_neg = summarize_rows(train_rows)
    test_pos, test_neg = summarize_rows(test_rows)

    print(f"Total: {total}")
    print(
        f"Train: {len(train_rows)}  ({len(train_rows) / total * 100:.1f}%)  "
        f"[pos={train_pos}, neg={train_neg}]"
    )
    print(
        f"Test:  {len(test_rows)}  ({len(test_rows) / total * 100:.1f}%)  "
        f"[pos={test_pos}, neg={test_neg}]"
    )
    print(f"Train -> {args.train_dir}")
    print(f"Test  -> {args.test_dir}")


if __name__ == "__main__":
    main()
