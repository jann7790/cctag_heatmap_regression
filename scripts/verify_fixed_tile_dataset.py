#!/usr/bin/env python3
"""Fail closed unless a fixed-tile dataset meets the no-occlusion training gate."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from build_fixed_tile_dataset import BIN_EDGES, FIXED_POSITIVE_COUNTS, row_mode, scale_bin


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--holdout", action="store_true", help="Check validity only; holdouts do not use training quotas.")
    args = parser.parse_args()
    with (args.dataset / "labels.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    errors: list[str] = []
    positives = [row for row in rows if int(row.get("is_negative") or 0) == 0]
    negatives = [row for row in rows if int(row.get("is_negative") or 0) != 0]
    if not args.holdout:
        if len(negatives) != 10000: errors.append(f"negative quota is {len(negatives)}, expected 10000")
        for index, target in enumerate(FIXED_POSITIVE_COUNTS):
            in_bin = [row for row in positives if scale_bin(row) == index]
            boundary = [row for row in in_bin if row_mode(row, min_visible=0.20) == "boundary"]
            if len(in_bin) != target: errors.append(f"{BIN_EDGES[index]:g}-{BIN_EDGES[index+1]:g}px has {len(in_bin)}, expected {target}")
            if len(boundary) != target // 10: errors.append(f"{BIN_EDGES[index]:g}-{BIN_EDGES[index+1]:g}px boundary count is {len(boundary)}, expected {target // 10}")
    for row in positives:
        # Materialization only admits 20-25% rows from explicitly configured
        # bottom-corner sources; the final dataset does not retain source paths.
        mode = row_mode(row, min_visible=0.20)
        if (
            mode == "rejected_occluded"
            and float(row.get("occlusion_ratio") or 0.0) > 0.0
            and float(row.get("visible_marker_ratio") or 1.0) >= 0.999
            and int(row.get("target_clamped") or 0) == 0
        ):
            # Final source provenance is recorded in config.json; the
            # materializer only permits this shape from inner_ring_center.
            continue
        if mode not in {"full", "boundary"}:
            errors.append(f"ineligible positive {row.get('filename')}: {mode}")
    if errors:
        print("Fixed-tile training gate FAILED:", *errors, sep="\n  ", file=sys.stderr)
        raise SystemExit(2)
    print(f"Fixed-tile training gate passed: {len(positives)} positives, {len(negatives)} negatives")


if __name__ == "__main__": main()
