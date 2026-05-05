"""
Augment positive samples in a heatmap dataset without changing geometry.

This script copies the full dataset and adds extra augmented variants for
positive rows (`is_negative != 1`). It supports photometric augmentation and
optional geometric augmentation (rotation + uniform scaling) while keeping
`images/`, `heatmaps/`, and `labels.csv` in sync.

Usage:
    python scripts/augment_positive_samples.py \
        --input_dir outputs/real_world_stride4_combined \
        --output_dir outputs/real_world_stride4_combined_pos_aug \
        --copies_per_positive 3 \
        --seed 42
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import math
from pathlib import Path

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy a dataset and add photometric augmentations for positive samples."
    )
    parser.add_argument("--input_dir", type=Path, required=True, help="Dataset root with images/, heatmaps/, labels.csv.")
    parser.add_argument("--output_dir", type=Path, required=True, help="Output dataset root.")
    parser.add_argument(
        "--copies_per_positive",
        type=int,
        default=3,
        help="Number of augmented variants to create for each positive sample.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--positive_requires_visible",
        action="store_true",
        help="Only augment positive rows where has_visible_marker == 1.",
    )
    parser.add_argument(
        "--rotate_deg",
        type=float,
        default=12.0,
        help="Maximum absolute rotation angle used for geometric augmentation.",
    )
    parser.add_argument(
        "--scale_min",
        type=float,
        default=0.9,
        help="Minimum uniform scale factor used for geometric augmentation.",
    )
    parser.add_argument(
        "--scale_max",
        type=float,
        default=1.1,
        help="Maximum uniform scale factor used for geometric augmentation.",
    )
    parser.add_argument(
        "--geom_prob",
        type=float,
        default=0.7,
        help="Probability of applying rotation+scale to an augmented positive sample.",
    )
    parser.add_argument(
        "--occlusion_prob",
        type=float,
        default=0.6,
        help="Probability of adding synthetic occlusion to an augmented positive sample.",
    )
    parser.add_argument(
        "--occlusion_mode",
        type=str,
        default="real_world",
        choices=["real_world", "bars", "mixed"],
        help="Occlusion family used during augmentation.",
    )
    parser.add_argument(
        "--occlusion_strength_min",
        type=float,
        default=0.18,
        help="Minimum target occlusion amount over the marker area.",
    )
    parser.add_argument(
        "--occlusion_strength_max",
        type=float,
        default=0.55,
        help="Maximum target occlusion amount over the marker area.",
    )
    return parser.parse_args()


def is_positive_row(row: dict[str, str], positive_requires_visible: bool) -> bool:
    if (row.get("is_negative") or "0").strip() == "1":
        return False
    if positive_requires_visible and (row.get("has_visible_marker") or "1").strip() == "0":
        return False
    return True


def ensure_dataset_layout(dataset_dir: Path) -> tuple[Path, Path, Path]:
    images_dir = dataset_dir / "images"
    heatmaps_dir = dataset_dir / "heatmaps"
    labels_csv = dataset_dir / "labels.csv"
    if not images_dir.is_dir():
        raise FileNotFoundError(f"Missing images directory: {images_dir}")
    if not heatmaps_dir.is_dir():
        raise FileNotFoundError(f"Missing heatmaps directory: {heatmaps_dir}")
    if not labels_csv.is_file():
        raise FileNotFoundError(f"Missing labels.csv: {labels_csv}")
    return images_dir, heatmaps_dir, labels_csv


def load_config(dataset_dir: Path) -> dict[str, object]:
    config_path = dataset_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing config.json: {config_path}")
    return json.loads(config_path.read_text(encoding="utf-8"))


def apply_random_photometric_aug(image_bgr: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = image_bgr.astype(np.float32) / 255.0

    brightness = float(rng.uniform(-0.12, 0.12))
    contrast = float(rng.uniform(0.85, 1.2))
    out = np.clip((out - 0.5) * contrast + 0.5 + brightness, 0.0, 1.0)

    if rng.random() < 0.8:
        gamma = float(rng.uniform(0.8, 1.25))
        out = np.clip(np.power(out, gamma), 0.0, 1.0)

    if rng.random() < 0.6:
        hsv = cv2.cvtColor((out * 255.0).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[..., 1] *= float(rng.uniform(0.85, 1.2))
        hsv[..., 2] *= float(rng.uniform(0.9, 1.1))
        hsv[..., 1:] = np.clip(hsv[..., 1:], 0.0, 255.0)
        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32) / 255.0

    if rng.random() < 0.5:
        ksize = int(rng.choice([3, 5]))
        out = cv2.GaussianBlur(out, (ksize, ksize), sigmaX=float(rng.uniform(0.2, 1.2)))

    if rng.random() < 0.7:
        noise_std = float(rng.uniform(0.005, 0.03))
        noise = rng.normal(0.0, noise_std, size=out.shape).astype(np.float32)
        out = np.clip(out + noise, 0.0, 1.0)

    if rng.random() < 0.35:
        h, w = out.shape[:2]
        yy, xx = np.ogrid[:h, :w]
        cx = w / 2.0 + float(rng.uniform(-0.1, 0.1)) * w
        cy = h / 2.0 + float(rng.uniform(-0.1, 0.1)) * h
        rx = max(w * float(rng.uniform(0.75, 1.2)), 1.0)
        ry = max(h * float(rng.uniform(0.75, 1.2)), 1.0)
        mask = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2
        vignette = 1.0 - float(rng.uniform(0.08, 0.22)) * np.clip(mask, 0.0, 1.0)
        out *= vignette[..., None].astype(np.float32)

    return np.clip(out * 255.0, 0.0, 255.0).astype(np.uint8)


def generate_gaussian_heatmap(size: tuple[int, int], center: tuple[float, float], sigma: float) -> np.ndarray:
    h, w = size
    cx, cy = center
    x = np.arange(w, dtype=np.float32)
    y = np.arange(h, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    heatmap = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma**2))
    return heatmap.astype(np.float32)


def marker_mask_from_row(row: dict[str, str], width: int, height: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    ellipse_a = float(row.get("ellipse_a") or -1.0)
    ellipse_b = float(row.get("ellipse_b") or -1.0)
    ellipse_cx = float(row.get("ellipse_cx") or row.get("center_x") or row.get("x") or 0.0)
    ellipse_cy = float(row.get("ellipse_cy") or row.get("center_y") or row.get("y") or 0.0)
    angle_rad = float(row.get("ellipse_angle_rad") or 0.0)
    if ellipse_a > 1.0 and ellipse_b > 1.0:
        cv2.ellipse(
            mask,
            (int(round(ellipse_cx)), int(round(ellipse_cy))),
            (max(int(round(ellipse_a)), 1), max(int(round(ellipse_b)), 1)),
            math.degrees(angle_rad),
            0.0,
            360.0,
            255,
            -1,
            cv2.LINE_AA,
        )
        return mask

    xmin = max(0, min(int(round(float(row.get("bbox_xmin") or 0.0))), width - 1))
    ymin = max(0, min(int(round(float(row.get("bbox_ymin") or 0.0))), height - 1))
    xmax = max(0, min(int(round(float(row.get("bbox_xmax") or 0.0))), width - 1))
    ymax = max(0, min(int(round(float(row.get("bbox_ymax") or 0.0))), height - 1))
    if xmax > xmin and ymax > ymin:
        cv2.rectangle(mask, (xmin, ymin), (xmax, ymax), 255, -1)
    return mask


def choose_occluder_color(image_bgr: np.ndarray, occ_mask: np.ndarray, rng: np.random.Generator) -> tuple[int, int, int]:
    pixels = image_bgr[occ_mask > 0]
    mean_value = float(pixels.mean()) if pixels.size > 0 else 127.0
    if mean_value >= 150.0:
        base = int(rng.integers(0, 40))
    elif mean_value >= 90.0:
        base = int(rng.integers(0, 70))
    else:
        base = int(rng.integers(160, 245)) if rng.random() < 0.35 else int(rng.integers(0, 60))
    return (base, base, base)


def apply_occlusion_mask(image_bgr: np.ndarray, occ_mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = image_bgr.copy()
    color = choose_occluder_color(image_bgr, occ_mask, rng)
    out[occ_mask > 0] = color
    if rng.random() < 0.35:
        noise = rng.normal(0.0, 10.0, size=out.shape).astype(np.float32)
        region = out.astype(np.float32)
        region[occ_mask > 0] = np.clip(region[occ_mask > 0] + noise[occ_mask > 0], 0.0, 255.0)
        out = region.astype(np.uint8)
    return out


def build_occlusion_mask(
    row: dict[str, str],
    width: int,
    height: int,
    mode: str,
    strength_min: float,
    strength_max: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, float]:
    marker_mask = marker_mask_from_row(row, width, height)
    marker_pixels = int(np.count_nonzero(marker_mask))
    if marker_pixels == 0:
        return np.zeros((height, width), dtype=np.uint8), 0.0

    center_x = float(row.get("center_x") or row.get("x") or 0.0)
    center_y = float(row.get("center_y") or row.get("y") or 0.0)
    bbox_w = max(float(row.get("bbox_xmax") or 0.0) - float(row.get("bbox_xmin") or 0.0), 8.0)
    bbox_h = max(float(row.get("bbox_ymax") or 0.0) - float(row.get("bbox_ymin") or 0.0), 8.0)
    occ_mask = np.zeros((height, width), dtype=np.uint8)
    target = float(rng.uniform(strength_min, strength_max))

    def draw_rect(cx: float, cy: float, rw: float, rh: float) -> None:
        x1 = max(0, min(int(round(cx - rw / 2.0)), width - 1))
        y1 = max(0, min(int(round(cy - rh / 2.0)), height - 1))
        x2 = max(0, min(int(round(cx + rw / 2.0)), width - 1))
        y2 = max(0, min(int(round(cy + rh / 2.0)), height - 1))
        if x2 > x1 and y2 > y1:
            cv2.rectangle(occ_mask, (x1, y1), (x2, y2), 255, -1)

    def add_side_pole() -> None:
        side = -1.0 if rng.random() < 0.5 else 1.0
        pole_w = bbox_w * float(rng.uniform(0.18, 0.38))
        pole_h = bbox_h * float(rng.uniform(0.9, 1.35))
        pole_cx = center_x + side * bbox_w * float(rng.uniform(0.15, 0.38))
        pole_cy = center_y + bbox_h * float(rng.uniform(-0.08, 0.08))
        draw_rect(pole_cx, pole_cy, pole_w, pole_h)

    def add_bottom_post() -> None:
        post_w = bbox_w * float(rng.uniform(0.12, 0.24))
        post_h = bbox_h * float(rng.uniform(0.65, 1.1))
        post_cx = center_x + bbox_w * float(rng.uniform(-0.12, 0.12))
        post_cy = center_y + bbox_h * float(rng.uniform(0.15, 0.45))
        draw_rect(post_cx, post_cy, post_w, post_h)
        if rng.random() < 0.6:
            draw_rect(post_cx, post_cy + post_h * 0.1, bbox_w * float(rng.uniform(0.4, 0.8)), bbox_h * float(rng.uniform(0.08, 0.16)))

    def add_top_bar() -> None:
        bar_w = bbox_w * float(rng.uniform(0.75, 1.25))
        bar_h = bbox_h * float(rng.uniform(0.12, 0.24))
        bar_cx = center_x + bbox_w * float(rng.uniform(-0.08, 0.08))
        bar_cy = center_y - bbox_h * float(rng.uniform(0.18, 0.4))
        draw_rect(bar_cx, bar_cy, bar_w, bar_h)

    def add_corner_block() -> None:
        sx = -1.0 if rng.random() < 0.5 else 1.0
        sy = -1.0 if rng.random() < 0.5 else 1.0
        block_w = bbox_w * float(rng.uniform(0.3, 0.6))
        block_h = bbox_h * float(rng.uniform(0.3, 0.6))
        block_cx = center_x + sx * bbox_w * float(rng.uniform(0.22, 0.38))
        block_cy = center_y + sy * bbox_h * float(rng.uniform(0.22, 0.38))
        draw_rect(block_cx, block_cy, block_w, block_h)

    def add_center_bar() -> None:
        if rng.random() < 0.5:
            draw_rect(center_x, center_y + bbox_h * float(rng.uniform(-0.12, 0.12)), bbox_w * float(rng.uniform(0.75, 1.2)), bbox_h * float(rng.uniform(0.12, 0.22)))
        else:
            draw_rect(center_x + bbox_w * float(rng.uniform(-0.12, 0.12)), center_y, bbox_w * float(rng.uniform(0.12, 0.22)), bbox_h * float(rng.uniform(0.75, 1.2)))

    families: list[tuple[str, callable]]
    families = [
        ("real_world", add_side_pole),
        ("real_world", add_bottom_post),
        ("real_world", add_top_bar),
        ("real_world", add_corner_block),
        ("bars", add_center_bar),
    ]
    allowed = [fn for family, fn in families if mode == "mixed" or family == mode]
    if not allowed:
        allowed = [add_side_pole, add_bottom_post, add_top_bar, add_corner_block]

    for _ in range(8):
        rng.choice(allowed)()
        covered = int(np.count_nonzero((marker_mask > 0) & (occ_mask > 0)))
        if covered / marker_pixels >= target:
            break

    actual = min(int(np.count_nonzero((marker_mask > 0) & (occ_mask > 0))) / marker_pixels, 1.0)
    return occ_mask, float(actual)


def make_affine_matrix(width: int, height: int, angle_deg: float, scale: float) -> np.ndarray:
    center = (width / 2.0, height / 2.0)
    return cv2.getRotationMatrix2D(center, angle_deg, scale).astype(np.float32)


def transform_points(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    ones = np.ones((points.shape[0], 1), dtype=np.float32)
    hom = np.concatenate([points.astype(np.float32), ones], axis=1)
    return hom @ matrix.T


def bbox_corners_from_row(row: dict[str, str]) -> np.ndarray:
    xmin = float(row.get("bbox_xmin") or 0.0)
    ymin = float(row.get("bbox_ymin") or 0.0)
    xmax = float(row.get("bbox_xmax") or 0.0)
    ymax = float(row.get("bbox_ymax") or 0.0)
    return np.array(
        [
            [xmin, ymin],
            [xmax, ymin],
            [xmax, ymax],
            [xmin, ymax],
        ],
        dtype=np.float32,
    )


def transformed_bbox(corners: np.ndarray) -> tuple[float, float, float, float]:
    xs = corners[:, 0]
    ys = corners[:, 1]
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


def sample_valid_affine(
    row: dict[str, str],
    width: int,
    height: int,
    rotate_deg: float,
    scale_min: float,
    scale_max: float,
    rng: np.random.Generator,
) -> np.ndarray | None:
    center_point = np.array([[float(row.get("center_x") or row.get("x") or 0.0), float(row.get("center_y") or row.get("y") or 0.0)]])
    orig_bbox = bbox_corners_from_row(row)
    for _ in range(24):
        angle = float(rng.uniform(-rotate_deg, rotate_deg))
        scale = float(rng.uniform(scale_min, scale_max))
        matrix = make_affine_matrix(width, height, angle, scale)
        new_center = transform_points(matrix, center_point)[0]
        new_bbox_corners = transform_points(matrix, orig_bbox)
        xmin, ymin, xmax, ymax = transformed_bbox(new_bbox_corners)
        if not (0.0 <= new_center[0] < width and 0.0 <= new_center[1] < height):
            continue
        if xmin < 0.0 or ymin < 0.0 or xmax >= width or ymax >= height:
            continue
        return matrix
    return None


def maybe_identity_affine(width: int, height: int) -> np.ndarray:
    return make_affine_matrix(width, height, angle_deg=0.0, scale=1.0)


def augment_row_geometry(
    row: dict[str, str],
    matrix: np.ndarray,
    width: int,
    height: int,
) -> dict[str, str]:
    out = dict(row)
    angle_delta_rad = math.atan2(float(matrix[0, 1]), float(matrix[0, 0]))
    scale = math.sqrt(float(matrix[0, 0]) ** 2 + float(matrix[0, 1]) ** 2)

    center = transform_points(
        matrix,
        np.array([[float(row.get("center_x") or row.get("x") or 0.0), float(row.get("center_y") or row.get("y") or 0.0)]], dtype=np.float32),
    )[0]
    ellipse_center = transform_points(
        matrix,
        np.array([[float(row.get("ellipse_cx") or row.get("center_x") or 0.0), float(row.get("ellipse_cy") or row.get("center_y") or 0.0)]], dtype=np.float32),
    )[0]
    bbox = transform_points(matrix, bbox_corners_from_row(row))
    xmin, ymin, xmax, ymax = transformed_bbox(bbox)

    out["x"] = f"{center[0]:.4f}"
    out["y"] = f"{center[1]:.4f}"
    out["center_x"] = f"{center[0]:.4f}"
    out["center_y"] = f"{center[1]:.4f}"
    out["ellipse_cx"] = f"{ellipse_center[0]:.4f}"
    out["ellipse_cy"] = f"{ellipse_center[1]:.4f}"
    out["ellipse_a"] = f"{float(row.get('ellipse_a') or 0.0) * scale:.4f}"
    out["ellipse_b"] = f"{float(row.get('ellipse_b') or 0.0) * scale:.4f}"
    out["ellipse_angle_rad"] = f"{float(row.get('ellipse_angle_rad') or 0.0) + angle_delta_rad:.6f}"
    out["bbox_xmin"] = f"{xmin:.4f}"
    out["bbox_ymin"] = f"{ymin:.4f}"
    out["bbox_xmax"] = f"{xmax:.4f}"
    out["bbox_ymax"] = f"{ymax:.4f}"
    out["yolo_cx"] = f"{((xmin + xmax) * 0.5) / width:.6f}"
    out["yolo_cy"] = f"{((ymin + ymax) * 0.5) / height:.6f}"
    out["yolo_w"] = f"{(xmax - xmin) / width:.6f}"
    out["yolo_h"] = f"{(ymax - ymin) / height:.6f}"
    return out


def regenerate_heatmap_for_row(
    row: dict[str, str],
    image_width: int,
    image_height: int,
    heatmap_width: int,
    heatmap_height: int,
    heatmap_stride: int,
    heatmap_sigma: float,
) -> np.ndarray:
    is_negative = (row.get("is_negative") or "0").strip() == "1"
    has_visible_marker = (row.get("has_visible_marker") or "1").strip() != "0"
    if is_negative or not has_visible_marker:
        return np.zeros((heatmap_height, heatmap_width), dtype=np.float32)

    center_x = float(row.get("center_x") or row.get("x") or 0.0) / float(heatmap_stride)
    center_y = float(row.get("center_y") or row.get("y") or 0.0) / float(heatmap_stride)
    return generate_gaussian_heatmap((heatmap_height, heatmap_width), (center_x, center_y), heatmap_sigma)


def main() -> None:
    args = parse_args()
    if args.copies_per_positive < 0:
        raise ValueError("--copies_per_positive must be >= 0")
    if args.scale_min <= 0.0 or args.scale_max <= 0.0 or args.scale_min > args.scale_max:
        raise ValueError("Expected 0 < --scale_min <= --scale_max")
    if not (0.0 <= args.geom_prob <= 1.0):
        raise ValueError("--geom_prob must be in [0, 1]")
    if not (0.0 <= args.occlusion_prob <= 1.0):
        raise ValueError("--occlusion_prob must be in [0, 1]")
    if not (0.0 <= args.occlusion_strength_min <= args.occlusion_strength_max <= 1.0):
        raise ValueError("Expected 0 <= --occlusion_strength_min <= --occlusion_strength_max <= 1")
    if args.output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {args.output_dir}")

    rng = np.random.default_rng(args.seed)
    input_images_dir, input_heatmaps_dir, labels_csv = ensure_dataset_layout(args.input_dir)
    config = load_config(args.input_dir)
    output_width, output_height = [int(v) for v in config["output_size"]]
    heatmap_stride = int(config["heatmap_stride"])
    heatmap_width, heatmap_height = [int(v) for v in config["heatmap_size"]]
    heatmap_sigma = float(config.get("heatmap_sigma", 2.0))

    output_images_dir = args.output_dir / "images"
    output_heatmaps_dir = args.output_dir / "heatmaps"
    output_images_dir.mkdir(parents=True, exist_ok=False)
    output_heatmaps_dir.mkdir(parents=True, exist_ok=False)

    shutil.copy2(args.input_dir / "config.json", args.output_dir / "config.json")

    with labels_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fieldnames = reader.fieldnames
    if not fieldnames:
        raise ValueError(f"Failed to read CSV header from {labels_csv}")

    output_rows: list[dict[str, str]] = []
    positive_count = 0
    augmented_count = 0

    for row in rows:
        filename = (row.get("filename") or "").strip()
        if not filename:
            continue

        image_path = input_images_dir / f"{filename}.png"
        heatmap_path = input_heatmaps_dir / f"{filename}.npy"
        if not image_path.is_file():
            raise FileNotFoundError(f"Missing image file: {image_path}")
        if not heatmap_path.is_file():
            raise FileNotFoundError(f"Missing heatmap file: {heatmap_path}")

        shutil.copy2(image_path, output_images_dir / image_path.name)
        shutil.copy2(heatmap_path, output_heatmaps_dir / heatmap_path.name)
        output_rows.append(dict(row))

        if not is_positive_row(row, args.positive_requires_visible):
            continue

        positive_count += 1
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Failed to read image: {image_path}")

        for aug_idx in range(args.copies_per_positive):
            aug_filename = f"{filename}__aug{aug_idx + 1:02d}"
            if rng.random() < args.geom_prob:
                matrix = sample_valid_affine(
                    row=row,
                    width=output_width,
                    height=output_height,
                    rotate_deg=args.rotate_deg,
                    scale_min=args.scale_min,
                    scale_max=args.scale_max,
                    rng=rng,
                )
            else:
                matrix = None

            if matrix is None:
                matrix = maybe_identity_affine(output_width, output_height)
                aug_row = dict(row)
            else:
                aug_row = augment_row_geometry(row, matrix, output_width, output_height)

            warped = cv2.warpAffine(
                image,
                matrix,
                (output_width, output_height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE,
            )
            if rng.random() < args.occlusion_prob:
                occ_mask, occ_ratio_added = build_occlusion_mask(
                    aug_row,
                    width=output_width,
                    height=output_height,
                    mode=args.occlusion_mode,
                    strength_min=args.occlusion_strength_min,
                    strength_max=args.occlusion_strength_max,
                    rng=rng,
                )
                if occ_ratio_added > 0.0:
                    warped = apply_occlusion_mask(warped, occ_mask, rng)
                    base_occ = float(aug_row.get("occlusion_ratio") or 0.0)
                    combined_occ = 1.0 - (1.0 - base_occ) * (1.0 - occ_ratio_added)
                    aug_row["occlusion_ratio"] = f"{min(max(combined_occ, 0.0), 1.0):.4f}"

            aug_image = apply_random_photometric_aug(warped, rng)
            ok = cv2.imwrite(str(output_images_dir / f"{aug_filename}.png"), aug_image)
            if not ok:
                raise ValueError(f"Failed to write augmented image: {output_images_dir / f'{aug_filename}.png'}")
            aug_heatmap = regenerate_heatmap_for_row(
                row=aug_row,
                image_width=output_width,
                image_height=output_height,
                heatmap_width=heatmap_width,
                heatmap_height=heatmap_height,
                heatmap_stride=heatmap_stride,
                heatmap_sigma=heatmap_sigma,
            )
            np.save(output_heatmaps_dir / f"{aug_filename}.npy", aug_heatmap)

            aug_row["filename"] = aug_filename
            output_rows.append(aug_row)
            augmented_count += 1

    with (args.output_dir / "labels.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    summary = {
        "source_dataset": str(args.input_dir),
        "copies_per_positive": int(args.copies_per_positive),
        "seed": int(args.seed),
        "positive_requires_visible": bool(args.positive_requires_visible),
        "occlusion_prob": float(args.occlusion_prob),
        "occlusion_mode": str(args.occlusion_mode),
        "occlusion_strength_range": [float(args.occlusion_strength_min), float(args.occlusion_strength_max)],
        "num_input_rows": len(rows),
        "num_positive_augmented": positive_count,
        "num_augmented_rows_added": augmented_count,
        "num_output_rows": len(output_rows),
    }
    (args.output_dir / "augmentation_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
