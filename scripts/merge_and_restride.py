"""
Merge two real-world datasets and regenerate heatmaps at a new stride.

Usage:
    uv run python scripts/merge_and_restride.py \
        --src1 real_wrold_testing_dataset \
        --src2 outputs/real_world \
        --output outputs/real_world_stride4 \
        --new_stride 4
"""

import argparse
import json
import shutil
from pathlib import Path

import csv

import numpy as np


def generate_gaussian_heatmap(
    size: tuple[int, int],
    center: tuple[float, float],
    sigma: float,
) -> np.ndarray:
    h, w = size
    cx, cy = center
    x = np.arange(w, dtype=np.float32)
    y = np.arange(h, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    heatmap = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma**2))
    return heatmap.astype(np.float32)


def regenerate_heatmaps_from_rows(
    rows: list[dict],
    output_heatmap_dir: Path,
    output_w: int,
    output_h: int,
    new_stride: int,
    new_sigma: float,
) -> None:
    hm_w = output_w // new_stride
    hm_h = output_h // new_stride

    for row in rows:
        filename = row["filename"]
        is_negative = row.get("is_negative", "0").strip() == "1"
        has_visible = row.get("has_visible_marker", "1").strip() != "0"

        if is_negative or not has_visible:
            heatmap = np.zeros((hm_h, hm_w), dtype=np.float32)
        else:
            cx = float(row["center_x"]) / new_stride
            cy = float(row["center_y"]) / new_stride
            heatmap = generate_gaussian_heatmap((hm_h, hm_w), (cx, cy), sigma=new_sigma)

        np.save(output_heatmap_dir / f"{filename}.npy", heatmap)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge datasets and regenerate heatmaps at new stride.")
    parser.add_argument("--src1", required=True, type=Path)
    parser.add_argument("--src2", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--new_stride", type=int, default=4)
    args = parser.parse_args()

    src1: Path = args.src1
    src2: Path = args.src2
    out: Path = args.output
    new_stride: int = args.new_stride

    # Read config from src1 (both are identical)
    cfg = json.loads((src1 / "config.json").read_text())
    output_w, output_h = cfg["output_size"]
    old_sigma: float = cfg.get("heatmap_sigma", 2.0)
    old_stride: int = cfg.get("heatmap_stride", 8)

    # Scale sigma proportionally so Gaussian covers same physical area
    new_sigma = old_sigma * (old_stride / new_stride)
    new_hm_w = output_w // new_stride
    new_hm_h = output_h // new_stride

    print(f"Stride: {old_stride} -> {new_stride}")
    print(f"Sigma:  {old_sigma} -> {new_sigma}")
    print(f"Heatmap size: ({old_sigma}, {cfg['heatmap_size']}) -> ({new_hm_h}, {new_hm_w})")

    # Create output dirs
    out_images = out / "images"
    out_heatmaps = out / "heatmaps"
    out_images.mkdir(parents=True, exist_ok=True)
    out_heatmaps.mkdir(parents=True, exist_ok=True)

    # Merge labels.csv
    rows: list[dict] = []
    header: list[str] = []
    for src_dir in [src1, src2]:
        with open(src_dir / "labels.csv", newline="") as f:
            reader = csv.DictReader(f)
            if not header:
                header = reader.fieldnames or []
            rows.extend(reader)

    with open(out / "labels.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Labels merged: {len(rows)} rows")

    # Copy images
    count = 0
    for src_dir in [src1, src2]:
        for img_path in (src_dir / "images").iterdir():
            shutil.copy2(img_path, out_images / img_path.name)
            count += 1
    print(f"Copied {count} images")

    # Regenerate heatmaps
    regenerate_heatmaps_from_rows(rows, out_heatmaps, output_w, output_h, new_stride, new_sigma)
    print(f"Generated {len(rows)} heatmaps at stride {new_stride}")

    # Write new config
    new_cfg = {
        "output_size": [output_w, output_h],
        "heatmap_stride": new_stride,
        "heatmap_size": [new_hm_w, new_hm_h],
        "heatmap_sigma": new_sigma,
    }
    (out / "config.json").write_text(json.dumps(new_cfg, indent=2))
    print(f"Config written: {new_cfg}")
    print(f"Done -> {out}")


if __name__ == "__main__":
    main()
