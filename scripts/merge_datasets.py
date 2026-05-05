"""
Merge compatible datasets without regenerating heatmaps.

Usage:
    uv run python scripts/merge_datasets.py \
        --src outputs/synthetic_train \
        --src outputs/real_world_stride4_train \
        --output outputs/final_train
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

from split_train_test import OPTIONAL_ARTIFACT_DIRS, load_dataset_config


def read_labels(src_dir: Path) -> tuple[list[dict[str, str]], list[str]]:
    with (src_dir / "labels.csv").open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        header = list(reader.fieldnames or [])
    if not header:
        raise ValueError(f"labels.csv has no header in {src_dir}")
    return rows, header


def ensure_output_dir(output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to write into non-empty directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)


def validate_compatibility(configs: list[dict], sources: list[Path]) -> None:
    base = configs[0]
    required_keys = ("output_size", "heatmap_stride", "heatmap_size")
    for src_dir, cfg in zip(sources[1:], configs[1:]):
        for key in required_keys:
            if cfg.get(key) != base.get(key):
                raise ValueError(
                    f"Incompatible dataset config for {src_dir}: "
                    f"{key}={cfg.get(key)!r} does not match {base.get(key)!r}"
                )


def detect_duplicate_filenames(rows_by_source: list[tuple[Path, list[dict[str, str]]]]) -> None:
    seen: dict[str, Path] = {}
    for src_dir, rows in rows_by_source:
        for row in rows:
            filename = row["filename"]
            if filename in seen:
                raise ValueError(
                    f"Duplicate filename detected across datasets: {filename} "
                    f"exists in both {seen[filename]} and {src_dir}"
                )
            seen[filename] = src_dir


def copy_artifacts(src_dir: Path, out_dir: Path, rows: list[dict[str, str]]) -> None:
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


def summarize_rows(rows: list[dict[str, str]]) -> tuple[int, int]:
    negatives = sum(1 for row in rows if row.get("is_negative", "0").strip() == "1")
    positives = len(rows) - negatives
    return positives, negatives


def merge_datasets(src_dirs: list[Path], output_dir: Path) -> list[dict[str, str]]:
    if len(src_dirs) < 2:
        raise ValueError("Provide at least two --src directories to merge")

    ensure_output_dir(output_dir)

    configs = [load_dataset_config(src_dir) for src_dir in src_dirs]
    validate_compatibility(configs, src_dirs)

    rows_by_source: list[tuple[Path, list[dict[str, str]]]] = []
    merged_rows: list[dict[str, str]] = []
    header: list[str] = []
    for src_dir in src_dirs:
        rows, current_header = read_labels(src_dir)
        if not header:
            header = current_header
        elif current_header != header:
            raise ValueError(f"CSV header mismatch in {src_dir}")
        rows_by_source.append((src_dir, rows))
        merged_rows.extend(rows)

    detect_duplicate_filenames(rows_by_source)

    for src_dir, rows in rows_by_source:
        copy_artifacts(src_dir, output_dir, rows)

    with (output_dir / "labels.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        writer.writerows(merged_rows)

    out_config = dict(configs[0])
    out_config["merged_from"] = [str(src_dir) for src_dir in src_dirs]
    (output_dir / "config.json").write_text(json.dumps(out_config, indent=2))
    return merged_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", dest="src_dirs", required=True, nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    merged_rows = merge_datasets(src_dirs=args.src_dirs, output_dir=args.output)
    positives, negatives = summarize_rows(merged_rows)

    print(f"Merged {len(merged_rows)} samples  [pos={positives}, neg={negatives}]")
    print(f"Done -> {args.output}")


if __name__ == "__main__":
    main()
