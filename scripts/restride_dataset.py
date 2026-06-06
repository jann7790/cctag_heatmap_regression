"""Re-generate a dataset's heatmaps at a new stride/sigma without touching images.

The heatmap is fully derived from labels.csv (center_x, center_y, is_negative),
so re-striding only recomputes the heatmaps/ arrays. Images, labels.csv and
labels_yolo/ are shared via symlink to avoid duplicating large files.

Faithful to generate_cctag_dataset.py:
  - positive row (is_negative == 0): Gaussian at (center_x/stride, center_y/stride)
  - negative row (is_negative == 1): all-zero heatmap

Usage:
    uv run python scripts/restride_dataset.py \
        --src outputs/training_sets/generated_training_sets/mixed_train_dataset \
        --output outputs/training_sets/generated_training_sets/mixed_train_dataset_stride2 \
        --stride 2 --sigma 3.0
"""

import argparse
import csv
import shutil
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np


@lru_cache(maxsize=None)
def coordinate_grid(h: int, w: int) -> tuple[np.ndarray, np.ndarray]:
    x = np.arange(w, dtype=np.float32)
    y = np.arange(h, dtype=np.float32)
    return np.meshgrid(x, y)


def gaussian_heatmap(
    size: tuple[int, int], center: tuple[float, float], sigma: float
) -> np.ndarray:
    h, w = size
    cx, cy = center
    xx, yy = coordinate_grid(h, w)
    return np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma**2)).astype(
        np.float32
    )


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        return
    if src.is_dir():
        dst.symlink_to(src.resolve(), target_is_directory=True)
    else:
        shutil.copy2(src, dst)


def restride(src: Path, output: Path, stride: int, sigma: float) -> None:
    images_dir = src / "images"
    labels_csv = src / "labels.csv"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Missing images dir: {images_dir}")
    if not labels_csv.is_file():
        raise FileNotFoundError(f"Missing labels.csv: {labels_csv}")

    rows = list(csv.DictReader(labels_csv.open("r", encoding="utf-8", newline="")))
    if not rows:
        raise ValueError(f"Empty labels.csv: {labels_csv}")

    # Output image dimensions, inferred from the first image.
    first = next(iter(sorted(images_dir.glob("*.png"))))
    img_h, img_w = cv2.imread(str(first)).shape[:2]
    if img_w % stride or img_h % stride:
        raise ValueError(
            f"stride {stride} must divide image size {img_w}x{img_h}"
        )
    hm_w, hm_h = img_w // stride, img_h // stride

    output.mkdir(parents=True, exist_ok=True)
    link_or_copy(images_dir, output / "images")
    shutil.copy2(labels_csv, output / "labels.csv")
    for extra in ("labels_yolo", "config_parts", "config.json", "README.md", "README.txt"):
        p = src / extra
        if p.exists():
            link_or_copy(p, output / extra)

    heatmaps_out = output / "heatmaps"
    heatmaps_out.mkdir(exist_ok=True)

    n_pos = n_neg = 0
    for row in rows:
        filename = row.get("filename", "").strip()
        if not filename:
            continue
        is_negative = row.get("is_negative", "0").strip() == "1"
        cx = float(row.get("center_x") or row.get("x") or -1.0)
        cy = float(row.get("center_y") or row.get("y") or -1.0)
        if is_negative or cx < 0 or cy < 0:
            heatmap = np.zeros((hm_h, hm_w), dtype=np.float32)
            n_neg += 1
        else:
            heatmap = gaussian_heatmap((hm_h, hm_w), (cx / stride, cy / stride), sigma)
            n_pos += 1
        np.savez_compressed(
            heatmaps_out / f"{filename}.npz", heatmap=heatmap.astype(np.float16)
        )

    print(
        f"{src.name}: wrote {n_pos + n_neg} heatmaps "
        f"({hm_h}x{hm_w}, stride={stride}, sigma={sigma}) "
        f"positives={n_pos} negatives={n_neg} -> {output}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--sigma", type=float, default=3.0)
    args = parser.parse_args()
    restride(args.src, args.output, args.stride, args.sigma)


if __name__ == "__main__":
    main()
