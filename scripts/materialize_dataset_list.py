#!/usr/bin/env python3
"""Materialize an exact dataset_dir/filename TSV split list for DataLoader use."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from build_fixed_tile_dataset import SourceRow, materialize, row_mode


def read_split_list(list_path: Path) -> list[tuple[Path, str]]:
    entries: list[tuple[Path, str]] = []
    with list_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            dataset_dir, filename = line.split("\t", maxsplit=1)
            entries.append((Path(dataset_dir), filename))
    if not entries:
        raise ValueError(f"No entries found in {list_path}")
    return entries


def resolve_rows(entries: list[tuple[Path, str]]) -> tuple[list[SourceRow], list[str]]:
    requested: dict[Path, set[str]] = {}
    for source, filename in entries:
        requested.setdefault(source, set()).add(filename)

    rows_by_key: dict[tuple[Path, str], dict[str, str]] = {}
    header: list[str] | None = None
    for source, filenames in requested.items():
        labels_path = source / "labels.csv"
        with labels_path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            current_header = list(reader.fieldnames or [])
            if header is None:
                header = current_header
            elif current_header != header:
                raise ValueError(f"CSV header mismatch: {labels_path}")
            for row in reader:
                if row["filename"] in filenames:
                    rows_by_key[(source, row["filename"])] = row

    missing = [entry for entry in entries if entry not in rows_by_key]
    if missing:
        preview = "\n".join(f"  {source}\t{filename}" for source, filename in missing[:10])
        raise FileNotFoundError(f"Missing {len(missing)} listed samples:\n{preview}")
    assert header is not None
    return [SourceRow(source, rows_by_key[(source, filename)]) for source, filename in entries], header


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--list", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--copy-files", action="store_true")
    args = parser.parse_args()

    entries = read_split_list(args.list)
    selected, header = resolve_rows(entries)
    rejected = [record for record in selected if row_mode(record.row).startswith("rejected_")]
    selected = [record for record in selected if not row_mode(record.row).startswith("rejected_")]
    if not selected:
        raise ValueError("No eligible samples remain after visibility/occlusion validation")
    positives = sum(int(record.row.get("is_negative") or 0) == 0 for record in selected)
    config = {
        "output_size": "1024x640",
        "output_width": 1024,
        "output_height": 640,
        "heatmap_stride": 4,
        "heatmap_width": 256,
        "heatmap_height": 160,
        "heatmap_size": [256, 160],
        "split_list": str(args.list),
        "num_samples": len(selected),
        "num_positive": positives,
        "num_negative": len(selected) - positives,
        "eligibility_rules": {
            "minimum_visible_marker_ratio": 0.25,
            "boundary_requires_target_clamped": 1,
            "physical_occlusion": "rejected",
        },
        "rejected_count": len(rejected),
        "artifact_mode": "copy" if args.copy_files else "hardlink_with_copy_fallback",
    }
    materialize(
        selected,
        header,
        args.output,
        config,
        use_hardlinks=not args.copy_files,
    )
    print(
        f"Built {len(selected)} samples at {args.output} "
        f"[positive={positives}, negative={len(selected) - positives}]"
    )


if __name__ == "__main__":
    main()
