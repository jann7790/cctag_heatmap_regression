#!/usr/bin/env python3
"""Convert legacy heatmap .npy files to compressed float16 .npz in place.

Each heatmaps/*.npy (raw float64, ~125 KB) becomes heatmaps/*.npz (compressed
float16, key 'heatmap', ~1-10 KB). float16 caps the per-pixel error at <=5e-4 on
a [0,1] target -- negligible for heatmap regression. Training/inference readers
already accept both .npz and legacy .npy, so converted datasets stay usable.

Usage:
    # Preview what would change (no writes)
    uv run python src/convert_heatmaps_to_npz.py --root ./outputs/training_sets --dry_run

    # Convert and delete the old .npy to reclaim space
    uv run python src/convert_heatmaps_to_npz.py --root ./outputs/training_sets

    # Convert but keep the .npy (safer; delete later once verified)
    uv run python src/convert_heatmaps_to_npz.py --root ./outputs/training_sets --keep_npy
"""

import argparse
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, required=True,
                   help="Directory to scan recursively for heatmaps/*.npy.")
    p.add_argument("--keep_npy", action="store_true",
                   help="Keep the original .npy after writing .npz (default: delete it).")
    p.add_argument("--dry_run", action="store_true",
                   help="Report what would be converted without writing or deleting anything.")
    p.add_argument("--max_err", type=float, default=1e-3,
                   help="Abort a file if float16 round-trip error exceeds this (default 1e-3).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root: Path = args.root
    if not root.is_dir():
        raise SystemExit(f"--root is not a directory: {root}")

    npy_files = sorted(p for p in root.rglob("*.npy") if p.parent.name == "heatmaps")
    if not npy_files:
        print(f"No heatmaps/*.npy found under {root} (nothing to do).")
        return

    print(f"Found {len(npy_files)} legacy .npy heatmap(s) under {root}")
    if args.dry_run:
        print("[dry run] no files will be written or deleted.")

    bytes_before = 0
    bytes_after = 0
    converted = 0
    skipped = 0
    worst_err = 0.0

    for i, npy_path in enumerate(npy_files):
        npz_path = npy_path.with_suffix(".npz")
        size_npy = npy_path.stat().st_size
        bytes_before += size_npy

        if args.dry_run:
            # Don't load every file just to preview: stat sizes for the total,
            # and only round-trip a small sample to sanity-check the error.
            if i < 20:
                arr = np.load(npy_path).astype(np.float32)
                err = float(np.abs(arr - arr.astype(np.float16).astype(np.float32)).max()) if arr.size else 0.0
                worst_err = max(worst_err, err)
                if i < 3:
                    print(f"  would write {npz_path.name}  (err={err:.2e})")
            converted += 1
            if (i + 1) % 5000 == 0:
                print(f"  scanned {i + 1}/{len(npy_files)}...")
            continue

        if npz_path.exists():
            # Already converted in a previous run; just clean up the .npy.
            bytes_after += npz_path.stat().st_size
            if not args.keep_npy:
                npy_path.unlink()
            skipped += 1
            continue

        arr = np.load(npy_path).astype(np.float32)
        arr16 = arr.astype(np.float16)
        err = float(np.abs(arr - arr16.astype(np.float32)).max()) if arr.size else 0.0
        worst_err = max(worst_err, err)
        if err > args.max_err:
            raise SystemExit(
                f"float16 error {err:.2e} > --max_err {args.max_err:.0e} at {npy_path}; aborting."
            )

        np.savez_compressed(str(npz_path), heatmap=arr16)
        bytes_after += npz_path.stat().st_size
        if not args.keep_npy:
            npy_path.unlink()
        converted += 1

        if (i + 1) % 2000 == 0:
            print(f"  {i + 1}/{len(npy_files)} done...")

    print("\n=== summary ===")
    print(f"converted: {converted}   already-npz: {skipped}")
    print(f"max float16 error: {worst_err:.2e}")
    mb = 1024 * 1024
    if args.dry_run:
        print(f"current .npy total: {bytes_before / mb:.1f} MB (run without --dry_run to convert)")
    else:
        saved = bytes_before - bytes_after
        pct = (saved / bytes_before * 100) if bytes_before else 0.0
        print(f"before: {bytes_before / mb:.1f} MB   after: {bytes_after / mb:.1f} MB   "
              f"saved: {saved / mb:.1f} MB ({pct:.1f}%)")
        if args.keep_npy:
            print("note: --keep_npy set, original .npy files retained.")


if __name__ == "__main__":
    main()
