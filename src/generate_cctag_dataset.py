#!/usr/bin/env python3
"""
Synthetic CCTag Heatmap Regression Dataset Generator
=====================================================
Generates training data for a lightweight CNN that predicts CCTag center
coordinates under 30-50% occlusion via Gaussian heatmap regression.

Output structure:
  output_dir/
    images/       - input images (PNG)
    heatmaps/     - ground truth heatmap (NPY, float32, single channel)
    labels.csv    - center point, fitted ellipse, bbox, occlusion metadata per image
    config.json   - generation parameters for reproducibility

Usage:
  python src/generate_cctag_dataset.py --num_images 50000 --output_dir ./outputs/datasets/cctag_dataset
  python src/generate_cctag_dataset.py --num_images 100 --output_dir ./outputs/datasets/cctag_demo --visualize
"""

import argparse
import csv
import json
import random
import time
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np


# ============================================================================
# CCTag Renderer
# ============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent
CSV_HEADER = [
    "filename",
    "x",
    "y",
    "center_x",
    "center_y",
    "ellipse_cx",
    "ellipse_cy",
    "ellipse_a",
    "ellipse_b",
    "ellipse_angle_rad",
    "occlusion_ratio",
    "bbox_xmin",
    "bbox_ymin",
    "bbox_xmax",
    "bbox_ymax",
    "yolo_cx",
    "yolo_cy",
    "yolo_w",
    "yolo_h",
    "is_negative",
    "negative_mode",
    "has_visible_marker",
    "visible_marker_ratio",
    "target_clamped",
]


def parse_output_size(value: str) -> tuple[int, int]:
    raw = str(value).strip().lower()
    if "x" in raw:
        width_text, height_text = raw.split("x", 1)
        width = int(width_text)
        height = int(height_text)
    else:
        width = int(raw)
        height = width

    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("output_size must be a positive integer or WIDTHxHEIGHT")

    return width, height


def normalize_canvas_size(canvas_size: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(canvas_size, int):
        return canvas_size, canvas_size
    width, height = canvas_size
    return int(width), int(height)


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    if args.marker_min <= 0 or args.marker_max <= 0:
        parser.error("--marker_min and --marker_max must both be positive integers.")
    if args.marker_min > args.marker_max:
        parser.error(
            f"--marker_min ({args.marker_min}) cannot be larger than "
            f"--marker_max ({args.marker_max}). "
            "Use --marker_max with a value >= --marker_min, "
            "for example --marker_min 200 --marker_max 400."
        )
    if args.occ_min > args.occ_max:
        parser.error(
            f"--occ_min ({args.occ_min}) cannot be larger than --occ_max ({args.occ_max})."
        )
    if not 0.0 <= args.partial_out_prob <= 1.0:
        parser.error("--partial_out_prob must be between 0.0 and 1.0.")
    if not 0.0 <= args.empty_negative_prob <= 1.0:
        parser.error("--empty_negative_prob must be between 0.0 and 1.0.")
    if args.negative_ratio is not None and not 0.0 <= args.negative_ratio <= 1.0:
        parser.error("--negative_ratio must be between 0.0 and 1.0.")
    if args.empty_negative_ratio is not None and not 0.0 <= args.empty_negative_ratio <= 1.0:
        parser.error("--empty_negative_ratio must be between 0.0 and 1.0.")
    if args.boundary_target_ratio is not None and not 0.0 <= args.boundary_target_ratio <= 1.0:
        parser.error("--boundary_target_ratio must be between 0.0 and 1.0.")
    if args.empty_negative_ratio is not None or args.boundary_target_ratio is not None:
        if args.empty_negative_ratio is None or args.boundary_target_ratio is None:
            parser.error("--empty_negative_ratio and --boundary_target_ratio must be set together.")
        if args.empty_negative_ratio + args.boundary_target_ratio > 1.0:
            parser.error("--empty_negative_ratio + --boundary_target_ratio must be <= 1.0.")
    if args.partial_out_max_ratio < 0.0:
        parser.error("--partial_out_max_ratio must be >= 0.0.")
    if args.heatmap_stride <= 0:
        parser.error("--heatmap_stride must be a positive integer.")
    output_w, output_h = args.output_size
    if output_w % args.heatmap_stride != 0 or output_h % args.heatmap_stride != 0:
        parser.error(
            "--heatmap_stride must evenly divide both output dimensions. "
            f"Got output_size={output_w}x{output_h}, heatmap_stride={args.heatmap_stride}."
        )
    if not 0.0 <= args.soft_focus_strength <= 1.0:
        parser.error("--soft_focus_strength must be between 0.0 and 1.0.")
    if not 0.0 <= args.overexposure_prob <= 1.0:
        parser.error("--overexposure_prob must be between 0.0 and 1.0.")


@lru_cache(maxsize=None)
def load_cctag_markers(rings_count: int) -> list[list[int]]:
    marker_file = (
        SCRIPT_DIR.parent
        / "assets"
        / "markers"
        / "CCTag"
        / "markersToPrint"
        / "generators"
        / f"cctag{rings_count}.txt"
    )
    markers = []
    with open(marker_file, "r", encoding="utf-8") as f:
        for line in f:
            values = [int(x) for x in line.split()]
            if values:
                markers.append(values)
    if not markers:
        raise ValueError(f"No markers found in {marker_file}")
    return markers


def render_cctag_from_radii(
    canvas_size: int | tuple[int, int] = 256,
    ring_radii: list[int] | tuple[int, ...] | None = None,
    outer_radius_ratio: float = 0.4,
    center: tuple = None,
) -> tuple[np.ndarray, tuple[float, float]]:
    """
    Render a CCTag using the same radius table and draw order as display_marker.py.
    Radii are defined on a 0-100 scale relative to the outer black disc radius.
    """
    if not ring_radii:
        raise ValueError("ring_radii must not be empty")

    canvas_w, canvas_h = normalize_canvas_size(canvas_size)
    img = np.full((canvas_h, canvas_w), 128, dtype=np.uint8)

    if center is None:
        cx = canvas_w / 2.0
        cy = canvas_h / 2.0
    else:
        cx, cy = center

    outer_r = min(canvas_w, canvas_h) * outer_radius_ratio
    cv2.circle(img, (int(round(cx)), int(round(cy))), int(round(outer_r)), 0, -1, cv2.LINE_AA)

    fill_value = 255
    for radius_value in ring_radii:
        radius = max(int(round(outer_r * (radius_value / 100.0))), 1)
        cv2.circle(img, (int(round(cx)), int(round(cy))), radius, fill_value, -1, cv2.LINE_AA)
        fill_value = 0 if fill_value == 255 else 255

    return img, (cx, cy)

def render_cctag(
    canvas_size: int | tuple[int, int] = 256,
    num_rings: int = 5,
    outer_radius_ratio: float = 0.4,
    center: tuple = None,
) -> tuple[np.ndarray, tuple[float, float]]:
    """
    Render a clean CCTag (concentric black/white rings) on a gray canvas.

    Returns:
        img: (height, width) uint8 grayscale image
        center: (cx, cy) float coordinates of the CCTag center
    """
    canvas_w, canvas_h = normalize_canvas_size(canvas_size)
    img = np.full((canvas_h, canvas_w), 128, dtype=np.uint8)

    if center is None:
        cx = canvas_w / 2.0
        cy = canvas_h / 2.0
    else:
        cx, cy = center

    outer_r = min(canvas_w, canvas_h) * outer_radius_ratio

    # Draw rings from outer to inner
    for i in range(num_rings):
        r = outer_r * (num_rings - i) / num_rings
        # Alternate black(0) / white(255), outermost = black
        color = 0 if i % 2 == 0 else 255
        cv2.circle(img, (int(round(cx)), int(round(cy))), int(round(r)), color, -1, cv2.LINE_AA)

    # Inner filled circle (black)
    inner_r = outer_r / num_rings * 0.5
    cv2.circle(img, (int(round(cx)), int(round(cy))), max(int(round(inner_r)), 1), 0, -1, cv2.LINE_AA)

    return img, (cx, cy)


def render_reference_cctag(
    canvas_size: int | tuple[int, int] = 256,
    outer_radius_ratio: float = 0.4,
    center: tuple = None,
) -> tuple[np.ndarray, tuple[float, float]]:
    """
    Render the reference marker style from the provided image:
    three black rings, two white gaps, and a white center.
    """
    canvas_w, canvas_h = normalize_canvas_size(canvas_size)
    img = np.full((canvas_h, canvas_w), 128, dtype=np.uint8)

    if center is None:
        cx = canvas_w / 2.0
        cy = canvas_h / 2.0
    else:
        cx, cy = center

    outer_r = min(canvas_w, canvas_h) * outer_radius_ratio

    # Normalized radii estimated from the reference image.
    ring_profile = [
        (1.00, 0),
        (0.82, 255),
        (0.69, 0),
        (0.56, 255),
        (0.43, 0),
        (0.00, 255),
    ]

    for radius_ratio, color in ring_profile:
        radius = max(int(round(outer_r * radius_ratio)), 1)
        cv2.circle(
            img,
            (int(round(cx)), int(round(cy))),
            radius,
            color,
            -1,
            cv2.LINE_AA,
        )

    return img, (cx, cy)


def render_marker(
    canvas_size: int | tuple[int, int] = 256,
    marker_style: str = "cctag_source",
    num_rings: int = 5,
    marker_id: int = 0,
    outer_radius_ratio: float = 0.4,
    center: tuple = None,
) -> tuple[np.ndarray, tuple[float, float]]:
    if marker_style == "cctag_source":
        markers = load_cctag_markers(num_rings)
        safe_marker_id = max(0, min(marker_id, len(markers) - 1))
        return render_cctag_from_radii(
            canvas_size=canvas_size,
            ring_radii=markers[safe_marker_id],
            outer_radius_ratio=outer_radius_ratio,
            center=center,
        )
    if marker_style == "reference":
        return render_reference_cctag(
            canvas_size=canvas_size,
            outer_radius_ratio=outer_radius_ratio,
            center=center,
        )
    return render_cctag(
        canvas_size=canvas_size,
        num_rings=num_rings,
        outer_radius_ratio=outer_radius_ratio,
        center=center,
    )


# ============================================================================
# Perspective / Affine Transform
# ============================================================================

def random_perspective_transform(
    img: np.ndarray,
    center: tuple[float, float],
    max_pitch_deg: float = 30.0,
    max_yaw_deg: float = 30.0,
) -> tuple[np.ndarray, tuple[float, float], np.ndarray]:
    """
    Apply a random perspective warp simulating oblique viewing angles.
    Returns the warped image and the transformed center coordinate.
    """
    h, w = img.shape[:2]

    # Random corner perturbation to simulate perspective
    margin = 0.15
    pts_src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])

    def _perturb(x, y):
        dx = random.uniform(-margin, margin) * w
        dy = random.uniform(-margin, margin) * h
        return [x + dx, y + dy]

    pts_dst = np.float32([
        _perturb(0, 0),
        _perturb(w, 0),
        _perturb(w, h),
        _perturb(0, h),
    ])

    M = cv2.getPerspectiveTransform(pts_src, pts_dst)
    warped = cv2.warpPerspective(img, M, (w, h), borderValue=128)

    # Transform center point
    pt = np.array([center[0], center[1], 1.0])
    pt_new = M @ pt
    cx_new = pt_new[0] / pt_new[2]
    cy_new = pt_new[1] / pt_new[2]

    return warped, (cx_new, cy_new), M


# ============================================================================
# Occlusion Synthesis
# ============================================================================

def apply_random_occlusion(
    img: np.ndarray,
    center: tuple[float, float],
    marker_radius: float,
    occlusion_range: tuple[float, float] = (0.0, 0.6),
    adversarial_prob: float = 0.15,
    occlusion_style: str = "standard",
) -> tuple[np.ndarray, float]:
    """
    Apply random occlusion over the marker area.

    Occlusion types:
      - Solid rectangle
      - Irregular polygon
      - Gradient / noise patch
      - Adversarial black-white stripes

    Returns:
        img: occluded image
        occlusion_ratio: approximate ratio of marker area occluded
    """
    h, w = img.shape[:2]
    target_ratio = random.uniform(*occlusion_range)

    if target_ratio < 0.02:
        return img, 0.0

    img = img.copy()
    cx, cy = center
    r = max(marker_radius, 10)
    marker_mask = np.zeros((h, w), dtype=np.uint8)
    cv2.circle(marker_mask, (int(round(cx)), int(round(cy))), int(round(r)), 255, -1, cv2.LINE_AA)
    occlusion_mask = np.zeros((h, w), dtype=np.uint8)

    def contrasting_occ_color(mask: np.ndarray) -> int:
        masked_pixels = img[mask > 0]
        mean_value = float(masked_pixels.mean()) if masked_pixels.size > 0 else 128.0

        # Prefer dark blockers so occlusion stays visually obvious.
        if mean_value >= 170.0:
            return random.randint(0, 24)
        if mean_value >= 110.0:
            return random.randint(0, 40) if random.random() < 0.9 else random.randint(40, 90)
        return random.randint(0, 36) if random.random() < 0.8 else random.randint(36, 96)

    def apply_mask(mask: np.ndarray, color: int | None = None) -> None:
        paint_value = contrasting_occ_color(mask) if color is None else color
        img[mask > 0] = paint_value
        occlusion_mask[mask > 0] = 255

    def clamp_rect(x1: float, y1: float, x2: float, y2: float) -> tuple[int, int, int, int]:
        rx1 = max(0, min(int(round(x1)), w - 1))
        ry1 = max(0, min(int(round(y1)), h - 1))
        rx2 = max(0, min(int(round(x2)), w - 1))
        ry2 = max(0, min(int(round(y2)), h - 1))
        if rx2 < rx1:
            rx1, rx2 = rx2, rx1
        if ry2 < ry1:
            ry1, ry2 = ry2, ry1
        return rx1, ry1, rx2, ry2

    def draw_structured_center_occluder() -> None:
        shape_mask = np.zeros((h, w), dtype=np.uint8)
        ratio_scale = min(max((target_ratio - 0.45) / 0.55, 0.0), 1.0)

        # Model real rigid occluders such as grippers or frame arms:
        # a thick horizontal beam entering from the side plus one or two vertical legs.
        top_bar_width = random.uniform(1.35, 1.9 + 0.9 * ratio_scale) * r
        top_bar_height = random.uniform(0.42, 0.62 + 0.18 * ratio_scale) * r
        left_shift = random.uniform(0.45, 1.05 + 0.35 * ratio_scale) * r
        top_y = cy - random.uniform(0.1, 0.45) * r

        x1, y1, x2, y2 = clamp_rect(
            cx - top_bar_width / 2.0 - left_shift,
            top_y - top_bar_height / 2.0,
            cx + top_bar_width / 2.0,
            top_y + top_bar_height / 2.0,
        )
        cv2.rectangle(shape_mask, (x1, y1), (x2, y2), 255, -1)

        leg_count = 2 if random.random() < (0.75 + 0.2 * ratio_scale) else 1
        leg_offsets = [-0.42, 0.2] if leg_count == 2 else [random.uniform(-0.15, 0.18)]
        for leg_idx, offset_multiplier in enumerate(leg_offsets):
            leg_width = random.uniform(0.3, 0.55 + 0.25 * ratio_scale) * r
            leg_height = random.uniform(1.1, 1.8 + 0.55 * ratio_scale) * r
            leg_offset_x = offset_multiplier * r
            x1, y1, x2, y2 = clamp_rect(
                cx + leg_offset_x - leg_width / 2.0,
                top_y,
                cx + leg_offset_x + leg_width / 2.0,
                top_y + leg_height,
            )
            cv2.rectangle(shape_mask, (x1, y1), (x2, y2), 255, -1)

            if leg_idx == 0 and random.random() < 0.65:
                foot_width = random.uniform(0.16, 0.34) * r
                foot_height = random.uniform(0.55, 1.0) * r
                fx1, fy1, fx2, fy2 = clamp_rect(
                    cx + leg_offset_x - foot_width / 2.0,
                    top_y + leg_height * 0.42,
                    cx + leg_offset_x + foot_width / 2.0,
                    top_y + leg_height * 0.42 + foot_height,
                )
                cv2.rectangle(shape_mask, (fx1, fy1), (fx2, fy2), 255, -1)

        if random.random() < (0.35 + 0.45 * ratio_scale):
            connector_width = random.uniform(0.18, 0.42) * r
            connector_height = random.uniform(0.18, 0.34) * r
            connector_x = cx - random.uniform(0.05, 0.35) * r
            connector_y = cy + random.uniform(0.0, 0.38) * r
            x1, y1, x2, y2 = clamp_rect(
                connector_x - connector_width / 2.0,
                connector_y - connector_height / 2.0,
                connector_x + connector_width / 2.0,
                connector_y + connector_height / 2.0,
            )
            cv2.rectangle(shape_mask, (x1, y1), (x2, y2), 255, -1)

        if random.random() < (0.18 + 0.32 * ratio_scale):
            stem_width = random.uniform(0.18, 0.38) * r
            stem_height = random.uniform(0.65, 1.1) * r
            x1, y1, x2, y2 = clamp_rect(
                cx - stem_width / 2.0,
                cy - 0.05 * r,
                cx + stem_width / 2.0,
                cy - 0.05 * r + stem_height,
            )
            cv2.rectangle(shape_mask, (x1, y1), (x2, y2), 255, -1)

        apply_mask(shape_mask)

    aggressive_style = occlusion_style in {"aggressive", "center_heavy"}
    if aggressive_style or target_ratio >= 0.75:
        draw_structured_center_occluder()
        if target_ratio >= 0.85 or occlusion_style == "center_heavy":
            extra_mask = np.zeros((h, w), dtype=np.uint8)
            extra_w = random.uniform(0.4, 1.0) * r
            extra_h = random.uniform(0.8, 1.6) * r
            extra_x = cx + random.uniform(-0.5, 0.5) * r
            extra_y = cy + random.uniform(-0.1, 0.8) * r
            x1, y1, x2, y2 = clamp_rect(
                extra_x - extra_w / 2.0,
                extra_y - extra_h / 2.0,
                extra_x + extra_w / 2.0,
                extra_y + extra_h / 2.0,
            )
            cv2.rectangle(extra_mask, (x1, y1), (x2, y2), 255, -1)
            apply_mask(extra_mask)
    else:
        # Decide how many occlusion patches (1-3)
        num_patches = random.randint(1, 3)

        for _ in range(num_patches):
            patch_ratio = target_ratio / num_patches
            patch_area = np.pi * r * r * patch_ratio
            patch_side = int(np.sqrt(patch_area))
            if patch_side < 3:
                continue

            # Random position biased toward the marker
            px = int(cx + random.uniform(-r, r) - patch_side / 2)
            py = int(cy + random.uniform(-r, r) - patch_side / 2)
            px = max(0, min(px, w - patch_side))
            py = max(0, min(py, h - patch_side))

            occ_type = random.random()

            if occ_type < adversarial_prob:
                patch = np.zeros((patch_side, patch_side), dtype=np.uint8)
                stripe_w = max(2, patch_side // random.randint(3, 8))
                for s in range(0, patch_side, stripe_w * 2):
                    patch[:, s:s + stripe_w] = 255
                if random.random() < 0.5:
                    patch = patch.T[:patch_side, :patch_side]
                img[py:py + patch_side, px:px + patch_side] = patch
                occlusion_mask[py:py + patch_side, px:px + patch_side] = 255

            elif occ_type < 0.4:
                color = random.randint(0, 255)
                img[py:py + patch_side, px:px + patch_side] = color
                occlusion_mask[py:py + patch_side, px:px + patch_side] = 255

            elif occ_type < 0.7:
                num_pts = random.randint(3, 7)
                pts = []
                for _ in range(num_pts):
                    pts.append([
                        px + random.randint(0, patch_side),
                        py + random.randint(0, patch_side),
                    ])
                pts = np.array(pts, dtype=np.int32)
                color = random.randint(0, 255)
                cv2.fillPoly(img, [pts], color)
                cv2.fillPoly(occlusion_mask, [pts], 255)

            else:
                noise = np.random.randint(0, 256, (patch_side, patch_side), dtype=np.uint8)
                if random.random() < 0.5:
                    noise = cv2.GaussianBlur(noise, (7, 7), 3)
                img[py:py + patch_side, px:px + patch_side] = noise
                occlusion_mask[py:py + patch_side, px:px + patch_side] = 255

    marker_pixels = max(int(np.count_nonzero(marker_mask)), 1)
    occluded_marker_pixels = int(np.count_nonzero((marker_mask > 0) & (occlusion_mask > 0)))
    actual_ratio = occluded_marker_pixels / marker_pixels
    return img, min(actual_ratio, 1.0)


# ============================================================================
# Background Compositing
# ============================================================================

def _draw_complex_background_elements(bg: np.ndarray) -> np.ndarray:
    """Draw random curves, arcs, and circles onto a background to create
    confusing patterns that the model must learn to ignore."""
    bh, bw = bg.shape[:2]
    n_elements = random.randint(3, 12)
    for _ in range(n_elements):
        elem_type = random.random()
        color = random.randint(0, 255)
        thickness = random.randint(1, 4)
        if elem_type < 0.3:
            # Random circle / arc
            cx = random.randint(0, bw - 1)
            cy = random.randint(0, bh - 1)
            radius = random.randint(10, max(11, min(bh, bw) // 3))
            if random.random() < 0.5:
                cv2.circle(bg, (cx, cy), radius, int(color), thickness)
            else:
                angle_start = random.randint(0, 360)
                angle_end = angle_start + random.randint(30, 300)
                cv2.ellipse(bg, (cx, cy), (radius, random.randint(radius // 2, radius)),
                            random.randint(0, 180), angle_start, angle_end,
                            int(color), thickness)
        elif elem_type < 0.6:
            # Random Bezier-like polyline (smooth curve)
            n_pts = random.randint(3, 6)
            pts = np.array([(random.randint(0, bw - 1), random.randint(0, bh - 1))
                            for _ in range(n_pts)], dtype=np.int32)
            cv2.polylines(bg, [pts], isClosed=False, color=int(color), thickness=thickness)
        elif elem_type < 0.8:
            # Concentric rings (CCTag-like confuser)
            cx = random.randint(bw // 4, 3 * bw // 4)
            cy = random.randint(bh // 4, 3 * bh // 4)
            n_rings = random.randint(2, 5)
            base_r = random.randint(8, max(9, min(bh, bw) // 6))
            for j in range(n_rings):
                r = base_r + j * random.randint(3, 8)
                ring_color = random.choice([random.randint(0, 80), random.randint(180, 255)])
                cv2.circle(bg, (cx, cy), r, int(ring_color), thickness)
        else:
            # Random line
            x1, y1 = random.randint(0, bw - 1), random.randint(0, bh - 1)
            x2, y2 = random.randint(0, bw - 1), random.randint(0, bh - 1)
            cv2.line(bg, (x1, y1), (x2, y2), int(color), thickness)
    return bg


def composite_on_background(
    marker_img: np.ndarray,
    bg_size: tuple[int, int] = (256, 256),
    background_complexity: str = "standard",
) -> tuple[np.ndarray, tuple[float, float], float]:
    """
    Place the marker image onto a random synthetic background.
    The marker canvas is assumed to have gray (128) as transparent.

    Args:
        background_complexity: "standard" for normal backgrounds,
            "complex" to add random curves/arcs/circles that act as hard negatives.

    Returns the composited image, offset applied, and scale used.
    """
    bh, bw = bg_size

    # Random background type
    bg_type = random.random()
    if bg_type < 0.3:
        # Uniform random color
        bg = np.full((bh, bw), random.randint(60, 200), dtype=np.uint8)
    elif bg_type < 0.6:
        # Gradient
        start_val = random.randint(40, 150)
        end_val = random.randint(100, 220)
        if random.random() < 0.5:
            gradient = np.linspace(start_val, end_val, bh, dtype=np.float32)[:, None]
            bg = np.broadcast_to(gradient, (bh, bw)).astype(np.uint8).copy()
        else:
            gradient = np.linspace(start_val, end_val, bw, dtype=np.float32)[None, :]
            bg = np.broadcast_to(gradient, (bh, bw)).astype(np.uint8).copy()
    else:
        # Noise texture
        bg = np.random.randint(60, 200, (bh, bw), dtype=np.uint8)
        bg = cv2.GaussianBlur(bg, (15, 15), 5)

    if background_complexity == "complex":
        bg = _draw_complex_background_elements(bg)

    # Composite: overwrite where marker is not gray background
    mh, mw = marker_img.shape[:2]
    # Center the marker on the background
    ox = (bw - mw) // 2
    oy = (bh - mh) // 2

    mask = np.abs(marker_img.astype(np.int16) - 128) > 10
    roi = bg[oy:oy + mh, ox:ox + mw]
    roi[mask] = marker_img[mask]
    bg[oy:oy + mh, ox:ox + mw] = roi

    return bg, (ox, oy), 1.0


# ============================================================================
# Image Degradation (simulate 100m observation)
# ============================================================================

def apply_degradation(
    img: np.ndarray,
    blur_range: tuple[int, int] = (0, 5),
    noise_std_range: tuple[float, float] = (0, 25),
    brightness_range: tuple[float, float] = (-40, 40),
    contrast_range: tuple[float, float] = (0.6, 1.4),
    motion_blur_prob: float = 0.2,
    scintillation_prob: float = 0.15,
    degradation_preset: str = "standard",
    soft_focus_strength: float = 0.0,
    overexposure_prob: float = 0.0,
) -> np.ndarray:
    """
    Apply realistic image degradation simulating long-distance FSO observation.

    Args:
        overexposure_prob: Probability of applying overexposure simulation
            (gamma compression + brightness boost) to mimic blown-out scenes.
    """
    img = img.astype(np.float32, copy=False)

    if degradation_preset == "soft_focus":
        soft_focus_strength = max(soft_focus_strength, 0.65)
        blur_range = (4, 10)
        noise_std_range = (0.0, 6.0)
        brightness_range = (16.0, 42.0)
        contrast_range = (0.78, 0.92)
        motion_blur_prob = 0.05
        scintillation_prob = 0.05

    # Overexposure simulation: gamma compression + strong brightness lift
    if overexposure_prob > 0.0 and random.random() < overexposure_prob:
        gamma = random.uniform(0.3, 0.6)  # low gamma = brighter
        img = 255.0 * np.power(np.clip(img / 255.0, 0, 1), gamma)
        lift = random.uniform(40, 120)
        img = img + lift

    # Brightness / contrast
    alpha = random.uniform(*contrast_range)
    beta = random.uniform(*brightness_range)
    img = img * alpha + beta

    # Gaussian blur (defocus / atmospheric)
    blur_k = random.randint(blur_range[0], blur_range[1])
    if blur_k > 0:
        k = blur_k * 2 + 1
        img = cv2.GaussianBlur(img, (k, k), 0)

    # Motion blur (platform vibration)
    if random.random() < motion_blur_prob:
        k_size = random.choice([3, 5, 7])
        angle = random.uniform(0, 180)
        M_rot = cv2.getRotationMatrix2D((k_size // 2, k_size // 2), angle, 1)
        kernel = np.zeros((k_size, k_size), dtype=np.float32)
        kernel[k_size // 2, :] = 1.0
        kernel = cv2.warpAffine(kernel, M_rot, (k_size, k_size))
        kernel /= kernel.sum() + 1e-8
        img = cv2.filter2D(img, -1, kernel)

    # Atmospheric scintillation (local random warp)
    if random.random() < scintillation_prob:
        h, w = img.shape[:2]
        # Small random displacement field
        strength = random.uniform(0.5, 2.0)
        dx = (np.random.randn(h, w).astype(np.float32)) * strength
        dy = (np.random.randn(h, w).astype(np.float32)) * strength
        dx = cv2.GaussianBlur(dx, (31, 31), 8)
        dy = cv2.GaussianBlur(dy, (31, 31), 8)
        map_x = np.arange(w, dtype=np.float32)[None, :] + dx
        map_y = np.arange(h, dtype=np.float32)[:, None] + dy
        img = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderValue=128)

    # Gaussian noise (sensor noise, especially at long range)
    noise_std = random.uniform(*noise_std_range)
    if noise_std > 0:
        noise = np.random.randn(*img.shape).astype(np.float32) * noise_std
        img += noise

    if soft_focus_strength > 0.0:
        # Simulate low-contrast, out-of-focus capture by compressing local contrast
        # toward a heavily blurred base layer and adding mild veiling glare.
        base_sigma = 10.0 + 24.0 * soft_focus_strength
        base_layer = cv2.GaussianBlur(img, (0, 0), base_sigma)
        detail_gain = max(0.03, 0.30 - 0.22 * soft_focus_strength)
        img = base_layer + (img - base_layer) * detail_gain

        glow_sigma = 16.0 + 28.0 * soft_focus_strength
        glow = cv2.GaussianBlur(img, (0, 0), glow_sigma)
        glow_mix = 0.16 + 0.22 * soft_focus_strength
        img = cv2.addWeighted(img, 1.0 - glow_mix, glow, glow_mix, 0.0)

        # Pull the whole image toward a mid-gray target so black rings stop
        # looking like clean black print and become hazy low-contrast bands.
        target_gray = 172.0 + 12.0 * soft_focus_strength
        gray_mix = 0.16 + 0.26 * soft_focus_strength
        img = img * (1.0 - gray_mix) + target_gray * gray_mix

        lift = 12.0 + 20.0 * soft_focus_strength
        img = img + lift

    return np.clip(img, 0, 255).astype(np.uint8)


# ============================================================================
# Gaussian Heatmap Generation
# ============================================================================

@lru_cache(maxsize=None)
def get_heatmap_coordinate_grid(size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    h, w = size
    x = np.arange(w, dtype=np.float32)
    y = np.arange(h, dtype=np.float32)
    return np.meshgrid(x, y)


def generate_gaussian_heatmap(
    size: tuple[int, int],
    center: tuple[float, float],
    sigma: float = 2.5,
) -> np.ndarray:
    """
    Generate a 2D Gaussian heatmap centered at (cx, cy).

    Args:
        size: (height, width) of the heatmap
        center: (cx, cy) in pixel coordinates
        sigma: standard deviation of the Gaussian

    Returns:
        heatmap: (H, W) float32 array, values in [0, 1]
    """
    h, w = size
    cx, cy = center

    xx, yy = get_heatmap_coordinate_grid((h, w))

    heatmap = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))

    return heatmap


def generate_negative_heatmap(size: tuple[int, int]) -> np.ndarray:
    return np.zeros(size, dtype=np.float32)


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    ones = np.ones((points.shape[0], 1), dtype=np.float32)
    homogenous = np.concatenate([points.astype(np.float32), ones], axis=1)
    transformed = homogenous @ matrix.T
    transformed /= transformed[:, 2:3]
    return transformed[:, :2]


@lru_cache(maxsize=None)
def get_unit_circle_points(num_samples: int) -> np.ndarray:
    angles = np.linspace(0, 2 * np.pi, num_samples, endpoint=False, dtype=np.float32)
    return np.stack([np.cos(angles), np.sin(angles)], axis=1)


def sample_circle_points(
    center: tuple[float, float],
    radius: float,
    perspective_matrix: np.ndarray | None = None,
    num_samples: int = 64,
) -> np.ndarray:
    unit_circle = get_unit_circle_points(num_samples)
    points = unit_circle * np.float32(radius)
    points[:, 0] += np.float32(center[0])
    points[:, 1] += np.float32(center[1])

    if perspective_matrix is not None:
        points = transform_points(points, perspective_matrix)

    return points


def circle_bbox_from_transform(
    center: tuple[float, float],
    radius: float,
    perspective_matrix: np.ndarray | None = None,
    num_samples: int = 64,
) -> tuple[float, float, float, float]:
    points = sample_circle_points(
        center=center,
        radius=radius,
        perspective_matrix=perspective_matrix,
        num_samples=num_samples,
    )

    x_min = float(points[:, 0].min())
    y_min = float(points[:, 1].min())
    x_max = float(points[:, 0].max())
    y_max = float(points[:, 1].max())
    return x_min, y_min, x_max, y_max


def fit_ellipse_from_points(points: np.ndarray) -> dict[str, float]:
    if points.shape[0] < 5:
        raise ValueError("Need at least 5 points to fit an ellipse")

    contour = points.astype(np.float32).reshape(-1, 1, 2)
    (cx, cy), (axis_1, axis_2), angle_deg = cv2.fitEllipse(contour)

    major_diameter = float(max(axis_1, axis_2))
    minor_diameter = float(min(axis_1, axis_2))
    major_angle_deg = float(angle_deg)
    if axis_1 < axis_2:
        major_angle_deg += 90.0

    # Normalize to [0, pi) in image coordinates, where positive is clockwise.
    major_angle_rad = np.deg2rad(major_angle_deg % 180.0)

    return {
        "ellipse_cx": float(cx),
        "ellipse_cy": float(cy),
        "ellipse_a": major_diameter / 2.0,
        "ellipse_b": minor_diameter / 2.0,
        "ellipse_angle_rad": float(major_angle_rad),
    }


def clip_bbox(
    bbox: tuple[float, float, float, float],
    image_size: tuple[int, int],
) -> tuple[float, float, float, float]:
    image_width, image_height = image_size
    x_min, y_min, x_max, y_max = bbox
    x_min = float(np.clip(x_min, 0, image_width - 1))
    y_min = float(np.clip(y_min, 0, image_height - 1))
    x_max = float(np.clip(x_max, 0, image_width - 1))
    y_max = float(np.clip(y_max, 0, image_height - 1))
    if x_max < x_min:
        x_min, x_max = x_max, x_min
    if y_max < y_min:
        y_min, y_max = y_max, y_min
    return x_min, y_min, x_max, y_max


def bbox_to_yolo(
    bbox: tuple[float, float, float, float],
    image_size: tuple[int, int],
) -> tuple[float, float, float, float]:
    image_width, image_height = image_size
    x_min, y_min, x_max, y_max = bbox
    width = max(x_max - x_min, 1.0)
    height = max(y_max - y_min, 1.0)
    center_x = x_min + width / 2.0
    center_y = y_min + height / 2.0
    return (
        center_x / image_width,
        center_y / image_height,
        width / image_width,
        height / image_height,
    )


def bbox_area(bbox: tuple[float, float, float, float]) -> float:
    x_min, y_min, x_max, y_max = bbox
    return max(x_max - x_min, 0.0) * max(y_max - y_min, 0.0)


def is_point_inside_frame(point: tuple[float, float], image_size: tuple[int, int]) -> bool:
    x, y = point
    image_width, image_height = image_size
    return 0.0 <= x < image_width and 0.0 <= y < image_height


def clamp_point_to_frame(point: tuple[float, float], image_size: tuple[int, int]) -> tuple[float, float]:
    x, y = point
    image_width, image_height = image_size
    return (
        float(np.clip(x, 0.0, image_width - 1)),
        float(np.clip(y, 0.0, image_height - 1)),
    )


def sample_marker_center(
    canvas_size: tuple[int, int],
    marker_radius: float,
    partial_out_prob: float = 0.0,
    partial_out_max_ratio: float = 0.35,
    placement_mode: str = "auto",
) -> tuple[float, float]:
    render_w, render_h = canvas_size
    full_margin = marker_radius * 1.2

    def sample_fully_inside() -> tuple[float, float]:
        if render_w > full_margin * 2:
            cx = random.uniform(full_margin, render_w - full_margin)
        else:
            cx = render_w / 2.0
        if render_h > full_margin * 2:
            cy = random.uniform(full_margin, render_h - full_margin)
        else:
            cy = render_h / 2.0
        return cx, cy

    def sample_partially_outside() -> tuple[float, float] | None:
        overflow = max(marker_radius * partial_out_max_ratio, 1.0)
        side = random.choice(["left", "right", "top", "bottom"])

        if side == "left":
            cx = random.uniform(-overflow, marker_radius * 0.95)
            cy = random.uniform(-overflow, render_h + overflow)
        elif side == "right":
            cx = random.uniform(render_w - marker_radius * 0.95, render_w + overflow)
            cy = random.uniform(-overflow, render_h + overflow)
        elif side == "top":
            cx = random.uniform(-overflow, render_w + overflow)
            cy = random.uniform(-overflow, marker_radius * 0.95)
        else:
            cx = random.uniform(-overflow, render_w + overflow)
            cy = random.uniform(render_h - marker_radius * 0.95, render_h + overflow)

        bbox = (
            cx - marker_radius,
            cy - marker_radius,
            cx + marker_radius,
            cy + marker_radius,
        )
        clipped_bbox = clip_bbox(bbox, (render_w, render_h))
        visible_area = bbox_area(clipped_bbox)
        full_area = bbox_area(bbox)
        if full_area <= 0.0:
            return None

        visible_ratio = visible_area / full_area
        is_partially_out = (
            bbox[0] < 0.0 or bbox[1] < 0.0 or bbox[2] > render_w or bbox[3] > render_h
        )
        if is_partially_out and 0.08 <= visible_ratio <= 0.95:
            return cx, cy
        return None

    if placement_mode == "inside":
        return sample_fully_inside()

    if placement_mode == "boundary":
        for _ in range(32):
            sampled = sample_partially_outside()
            if sampled is not None:
                return sampled
        return sample_fully_inside()

    if partial_out_prob <= 0.0 or random.random() >= partial_out_prob:
        return sample_fully_inside()

    for _ in range(50):
        sampled = sample_partially_outside()
        if sampled is not None:
            return sampled

    return sample_fully_inside()


def build_negative_meta(
    negative_mode: str,
    occlusion_ratio: float = 0.0,
    visible_marker_ratio: float = 0.0,
) -> dict:
    return {
        "x": -1.0,
        "y": -1.0,
        "center_x": -1.0,
        "center_y": -1.0,
        "ellipse_cx": -1.0,
        "ellipse_cy": -1.0,
        "ellipse_a": -1.0,
        "ellipse_b": -1.0,
        "ellipse_angle_rad": -1.0,
        "occlusion_ratio": float(occlusion_ratio),
        "bbox_xmin": -1.0,
        "bbox_ymin": -1.0,
        "bbox_xmax": -1.0,
        "bbox_ymax": -1.0,
        "yolo_cx": -1.0,
        "yolo_cy": -1.0,
        "yolo_w": -1.0,
        "yolo_h": -1.0,
        "is_negative": 1,
        "negative_mode": negative_mode,
        "has_visible_marker": int(visible_marker_ratio > 0.0),
        "visible_marker_ratio": float(visible_marker_ratio),
        "target_clamped": 0,
    }


def generate_empty_negative_sample(
    output_size: tuple[int, int],
    heatmap_stride: int,
    degradation_preset: str,
    soft_focus_strength: float,
    background_complexity: str = "standard",
    overexposure_prob: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    output_w, output_h = output_size
    heatmap_w = output_w // heatmap_stride
    heatmap_h = output_h // heatmap_stride
    render_w = output_w * 2
    render_h = output_h * 2

    blank_marker = np.full((render_h, render_w), 128, dtype=np.uint8)
    bg, _, _ = composite_on_background(blank_marker, bg_size=(render_h, render_w),
                                       background_complexity=background_complexity)
    bg = apply_degradation(
        bg,
        degradation_preset=degradation_preset,
        soft_focus_strength=soft_focus_strength,
        overexposure_prob=overexposure_prob,
    )
    final_img = cv2.resize(bg, (output_w, output_h), interpolation=cv2.INTER_AREA)
    heatmap = generate_negative_heatmap((heatmap_h, heatmap_w))
    meta = build_negative_meta("empty_no_cctag")
    return final_img, heatmap, meta


def generate_marker_sample(
    output_size: tuple[int, int] = (128, 128),
    num_rings: int = 5,
    marker_style: str = "cctag_source",
    marker_id: int = 0,
    marker_diameter_range: tuple[int, int] = (24, 80),
    occlusion_range: tuple[float, float] = (0.0, 0.6),
    heatmap_sigma: float = 2.5,
    heatmap_stride: int = 8,
    occlusion_style: str = "standard",
    partial_out_prob: float = 0.0,
    partial_out_max_ratio: float = 0.35,
    placement_mode: str = "auto",
    degradation_preset: str = "standard",
    soft_focus_strength: float = 0.0,
    background_complexity: str = "standard",
    overexposure_prob: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    output_w, output_h = output_size
    heatmap_w = output_w // heatmap_stride
    heatmap_h = output_h // heatmap_stride

    render_w = output_w * 2
    render_h = output_h * 2

    marker_diameter = random.randint(*marker_diameter_range)
    marker_radius = marker_diameter
    render_marker_radius = marker_radius * 2.0
    render_radius_ratio = render_marker_radius / min(render_w, render_h)

    render_cx, render_cy = sample_marker_center(
        canvas_size=(render_w, render_h),
        marker_radius=render_marker_radius,
        partial_out_prob=partial_out_prob,
        partial_out_max_ratio=partial_out_max_ratio,
        placement_mode=placement_mode,
    )

    img, (cx, cy) = render_marker(
        canvas_size=(render_w, render_h),
        marker_style=marker_style,
        num_rings=num_rings,
        marker_id=marker_id,
        outer_radius_ratio=render_radius_ratio,
        center=(render_cx, render_cy),
    )

    perspective_matrix = None
    if random.random() < 0.7:
        img, (cx, cy), perspective_matrix = random_perspective_transform(img, (cx, cy))

    img, occ_ratio = apply_random_occlusion(
        img,
        (cx, cy),
        marker_radius,
        occlusion_range=occlusion_range,
        occlusion_style=occlusion_style,
    )

    bg, _, _ = composite_on_background(img, bg_size=(render_h, render_w),
                                       background_complexity=background_complexity)
    bg = apply_degradation(
        bg,
        degradation_preset=degradation_preset,
        soft_focus_strength=soft_focus_strength,
        overexposure_prob=overexposure_prob,
    )
    final_img = cv2.resize(bg, (output_w, output_h), interpolation=cv2.INTER_AREA)

    scale_x = output_w / render_w
    scale_y = output_h / render_h
    final_cx = cx * scale_x
    final_cy = cy * scale_y
    transformed_circle_points = sample_circle_points(
        center=(render_cx, render_cy),
        radius=marker_radius,
        perspective_matrix=perspective_matrix,
    )
    bbox = (
        float(transformed_circle_points[:, 0].min()),
        float(transformed_circle_points[:, 1].min()),
        float(transformed_circle_points[:, 0].max()),
        float(transformed_circle_points[:, 1].max()),
    )
    ellipse = fit_ellipse_from_points(transformed_circle_points)
    ellipse = {
        "ellipse_cx": float(ellipse["ellipse_cx"] * scale_x),
        "ellipse_cy": float(ellipse["ellipse_cy"] * scale_y),
        "ellipse_a": float(ellipse["ellipse_a"] * scale_x),
        "ellipse_b": float(ellipse["ellipse_b"] * scale_y),
        "ellipse_angle_rad": float(ellipse["ellipse_angle_rad"]),
    }
    final_bbox = (
        bbox[0] * scale_x,
        bbox[1] * scale_y,
        bbox[2] * scale_x,
        bbox[3] * scale_y,
    )
    final_bbox = clip_bbox(final_bbox, (output_w, output_h))
    raw_final_bbox = (
        bbox[0] * scale_x,
        bbox[1] * scale_y,
        bbox[2] * scale_x,
        bbox[3] * scale_y,
    )
    visible_bbox = clip_bbox(raw_final_bbox, (output_w, output_h))
    visible_area = bbox_area(visible_bbox)
    full_area = bbox_area(raw_final_bbox)
    visible_marker_ratio = visible_area / full_area if full_area > 0.0 else 0.0
    center_in_frame = is_point_inside_frame((final_cx, final_cy), (output_w, output_h))
    target_cx, target_cy = final_cx, final_cy
    target_clamped = 0
    negative_mode = ""
    if not center_in_frame:
        target_cx, target_cy = clamp_point_to_frame((final_cx, final_cy), (output_w, output_h))
        target_clamped = 1
        negative_mode = "center_clamped_to_frame"

    yolo_bbox = bbox_to_yolo(final_bbox, (output_w, output_h))
    heatmap = generate_gaussian_heatmap(
        (heatmap_h, heatmap_w),
        (target_cx / heatmap_stride, target_cy / heatmap_stride),
        sigma=heatmap_sigma,
    )

    meta = {
        "x": float(target_cx),
        "y": float(target_cy),
        "center_x": float(target_cx),
        "center_y": float(target_cy),
        "ellipse_cx": ellipse["ellipse_cx"],
        "ellipse_cy": ellipse["ellipse_cy"],
        "ellipse_a": ellipse["ellipse_a"],
        "ellipse_b": ellipse["ellipse_b"],
        "ellipse_angle_rad": ellipse["ellipse_angle_rad"],
        "occlusion_ratio": float(occ_ratio),
        "bbox_xmin": float(final_bbox[0]),
        "bbox_ymin": float(final_bbox[1]),
        "bbox_xmax": float(final_bbox[2]),
        "bbox_ymax": float(final_bbox[3]),
        "yolo_cx": float(yolo_bbox[0]),
        "yolo_cy": float(yolo_bbox[1]),
        "yolo_w": float(yolo_bbox[2]),
        "yolo_h": float(yolo_bbox[3]),
        "is_negative": 0,
        "negative_mode": negative_mode,
        "has_visible_marker": 1,
        "visible_marker_ratio": float(visible_marker_ratio),
        "target_clamped": target_clamped,
    }
    return final_img, heatmap, meta


# ============================================================================
# Single Sample Generator
# ============================================================================

def generate_single_sample(
    output_size: tuple[int, int] = (128, 128),
    num_rings: int = 5,
    marker_style: str = "cctag_source",
    marker_id: int = 0,
    marker_diameter_range: tuple[int, int] = (24, 80),
    occlusion_range: tuple[float, float] = (0.0, 0.6),
    heatmap_sigma: float = 2.5,
    heatmap_stride: int = 8,
    occlusion_style: str = "standard",
    partial_out_prob: float = 0.0,
    partial_out_max_ratio: float = 0.35,
    empty_negative_prob: float = 0.0,
    force_empty_negative: bool = False,
    force_boundary_target: bool = False,
    force_normal_positive: bool = False,
    degradation_preset: str = "standard",
    soft_focus_strength: float = 0.0,
    background_complexity: str = "standard",
    overexposure_prob: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Generate one training sample: image + heatmap + metadata.

    The marker_diameter_range simulates different working distances:
      - 24px ~ 100m
      - 80px ~ 50m (or closer)

    Returns:
        image: (output_height, output_width) uint8
        heatmap: (output_height / stride, output_width / stride) float32
        meta: dict with center point, fitted ellipse, bbox, occlusion_ratio
    """
    if force_empty_negative or random.random() < empty_negative_prob:
        return generate_empty_negative_sample(
            output_size=output_size,
            heatmap_stride=heatmap_stride,
            degradation_preset=degradation_preset,
            soft_focus_strength=soft_focus_strength,
            background_complexity=background_complexity,
            overexposure_prob=overexposure_prob,
        )

    desired_target_clamped = None
    effective_partial_out_prob = partial_out_prob
    placement_mode = "auto"
    if force_boundary_target:
        desired_target_clamped = 1
        effective_partial_out_prob = 1.0
        placement_mode = "boundary"
    elif force_normal_positive:
        desired_target_clamped = 0
        effective_partial_out_prob = 0.0
        placement_mode = "inside"

    max_attempts = 16 if desired_target_clamped == 1 else 1
    last_sample = None
    for _ in range(max_attempts):
        last_sample = generate_marker_sample(
            output_size=output_size,
            num_rings=num_rings,
            marker_style=marker_style,
            marker_id=marker_id,
            marker_diameter_range=marker_diameter_range,
            occlusion_range=occlusion_range,
            heatmap_sigma=heatmap_sigma,
            heatmap_stride=heatmap_stride,
            occlusion_style=occlusion_style,
            partial_out_prob=effective_partial_out_prob,
            partial_out_max_ratio=partial_out_max_ratio,
            placement_mode=placement_mode,
            degradation_preset=degradation_preset,
            soft_focus_strength=soft_focus_strength,
            background_complexity=background_complexity,
            overexposure_prob=overexposure_prob,
        )
        _, _, meta = last_sample
        if desired_target_clamped is None or meta["target_clamped"] == desired_target_clamped:
            return last_sample

    return last_sample


# ============================================================================
# Visualization (for debugging)
# ============================================================================

def visualize_samples(images, heatmaps, metas, save_path: str):
    """Save a grid of sample images with heatmap overlay for visual inspection."""
    n = min(len(images), 16)
    cols = 4
    rows = (n + cols - 1) // cols

    cell_size = 128
    margin = 4
    grid_w = cols * (cell_size * 2 + margin) + margin
    grid_h = rows * (cell_size + margin) + margin
    grid = np.full((grid_h, grid_w, 3), 240, dtype=np.uint8)

    for i in range(n):
        r, c = divmod(i, cols)
        x0 = margin + c * (cell_size * 2 + margin)
        y0 = margin + r * (cell_size + margin)

        # Original image
        img_rgb = cv2.cvtColor(images[i], cv2.COLOR_GRAY2BGR)
        if not metas[i]["is_negative"]:
            cx, cy = metas[i]["center_x"], metas[i]["center_y"]
            cv2.drawMarker(img_rgb, (int(cx), int(cy)), (0, 0, 255),
                           cv2.MARKER_CROSS, 10, 1)
            x_min = int(round(metas[i]["bbox_xmin"]))
            y_min = int(round(metas[i]["bbox_ymin"]))
            x_max = int(round(metas[i]["bbox_xmax"]))
            y_max = int(round(metas[i]["bbox_ymax"]))
            cv2.rectangle(img_rgb, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
        img_resized = cv2.resize(img_rgb, (cell_size, cell_size))
        grid[y0:y0 + cell_size, x0:x0 + cell_size] = img_resized

        # Heatmap overlay
        hm = (heatmaps[i] * 255).astype(np.uint8)
        hm_color = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
        hm_resized = cv2.resize(hm_color, (cell_size, cell_size))
        overlay = cv2.addWeighted(img_resized, 0.5, hm_resized, 0.5, 0)
        grid[y0:y0 + cell_size, x0 + cell_size:x0 + cell_size * 2] = overlay

        # Occlusion ratio text
        occ = metas[i]["occlusion_ratio"]
        label = metas[i]["negative_mode"] if metas[i]["is_negative"] else f"{occ:.0%}"
        cv2.putText(grid, label, (x0 + 2, y0 + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)

    cv2.imwrite(save_path, grid)
    print(f"Visualization saved to {save_path}")


# ============================================================================
# Main Dataset Generation
# ============================================================================

def generate_dataset(args):
    output_dir = Path(args.output_dir)
    img_dir = output_dir / "images"
    hm_dir = output_dir / "heatmaps"
    yolo_dir = output_dir / "labels_yolo"
    img_dir.mkdir(parents=True, exist_ok=True)
    hm_dir.mkdir(parents=True, exist_ok=True)
    yolo_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "num_images": args.num_images,
        "output_size": f"{args.output_size[0]}x{args.output_size[1]}",
        "output_width": args.output_size[0],
        "output_height": args.output_size[1],
        "marker_style": args.marker_style,
        "marker_id": args.marker_id,
        "num_rings": args.num_rings,
        "marker_diameter_range": [args.marker_min, args.marker_max],
        "occlusion_range": [args.occ_min, args.occ_max],
        "occlusion_style": args.occlusion_style,
        "partial_out_prob": args.partial_out_prob,
        "partial_out_max_ratio": args.partial_out_max_ratio,
        "empty_negative_prob": args.empty_negative_prob,
        "negative_ratio": args.negative_ratio,
        "empty_negative_ratio": args.empty_negative_ratio,
        "boundary_target_ratio": args.boundary_target_ratio,
        "heatmap_stride": args.heatmap_stride,
        "heatmap_width": args.output_size[0] // args.heatmap_stride,
        "heatmap_height": args.output_size[1] // args.heatmap_stride,
        "heatmap_sigma": args.heatmap_sigma,
        "yolo_class_id": args.yolo_class_id,
        "degradation_preset": args.degradation_preset,
        "soft_focus_strength": args.soft_focus_strength,
        "background_complexity": args.background_complexity,
        "overexposure_prob": args.overexposure_prob,
        "seed": args.seed,
    }

    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    forced_sample_modes = None
    if args.negative_ratio is not None:
        negative_count = int(round(args.num_images * args.negative_ratio))
        negative_count = max(0, min(negative_count, args.num_images))
        forced_sample_modes = ["empty_negative"] * negative_count + ["normal_positive"] * (args.num_images - negative_count)
        random.shuffle(forced_sample_modes)
    elif args.empty_negative_ratio is not None and args.boundary_target_ratio is not None:
        empty_negative_count = int(round(args.num_images * args.empty_negative_ratio))
        boundary_target_count = int(round(args.num_images * args.boundary_target_ratio))
        empty_negative_count = max(0, min(empty_negative_count, args.num_images))
        boundary_target_count = max(0, min(boundary_target_count, args.num_images - empty_negative_count))
        normal_positive_count = args.num_images - empty_negative_count - boundary_target_count
        forced_sample_modes = (
            ["empty_negative"] * empty_negative_count
            + ["boundary_target"] * boundary_target_count
            + ["normal_positive"] * normal_positive_count
        )
        random.shuffle(forced_sample_modes)

    csv_path = output_dir / "labels.csv"
    csv_rows = []

    viz_images, viz_heatmaps, viz_metas = [], [], []
    collect_viz = args.visualize

    t0 = time.time()
    for i in range(args.num_images):
        sample_mode = forced_sample_modes[i] if forced_sample_modes is not None else ""
        img, hm, meta = generate_single_sample(
            output_size=args.output_size,
            num_rings=args.num_rings,
            marker_style=args.marker_style,
            marker_id=args.marker_id,
            marker_diameter_range=(args.marker_min, args.marker_max),
            occlusion_range=(args.occ_min, args.occ_max),
            heatmap_sigma=args.heatmap_sigma,
            heatmap_stride=args.heatmap_stride,
            occlusion_style=args.occlusion_style,
            partial_out_prob=args.partial_out_prob,
            partial_out_max_ratio=args.partial_out_max_ratio,
            empty_negative_prob=0.0 if forced_sample_modes is not None else args.empty_negative_prob,
            force_empty_negative=(sample_mode == "empty_negative"),
            force_boundary_target=(sample_mode == "boundary_target"),
            force_normal_positive=(sample_mode == "normal_positive"),
            degradation_preset=args.degradation_preset,
            soft_focus_strength=args.soft_focus_strength,
            background_complexity=args.background_complexity,
            overexposure_prob=args.overexposure_prob,
        )

        fname = f"{i:06d}"
        cv2.imwrite(str(img_dir / f"{fname}.png"), img)
        np.save(str(hm_dir / f"{fname}.npy"), hm)
        csv_rows.append([
            fname,
            f"{meta['x']:.4f}",
            f"{meta['y']:.4f}",
            f"{meta['center_x']:.4f}",
            f"{meta['center_y']:.4f}",
            f"{meta['ellipse_cx']:.4f}",
            f"{meta['ellipse_cy']:.4f}",
            f"{meta['ellipse_a']:.4f}",
            f"{meta['ellipse_b']:.4f}",
            f"{meta['ellipse_angle_rad']:.6f}",
            f"{meta['occlusion_ratio']:.4f}",
            f"{meta['bbox_xmin']:.4f}",
            f"{meta['bbox_ymin']:.4f}",
            f"{meta['bbox_xmax']:.4f}",
            f"{meta['bbox_ymax']:.4f}",
            f"{meta['yolo_cx']:.6f}",
            f"{meta['yolo_cy']:.6f}",
            f"{meta['yolo_w']:.6f}",
            f"{meta['yolo_h']:.6f}",
            str(meta["is_negative"]),
            meta["negative_mode"],
            str(meta["has_visible_marker"]),
            f"{meta['visible_marker_ratio']:.6f}",
            str(meta["target_clamped"]),
        ])
        yolo_text = ""
        if not meta["is_negative"]:
            yolo_text = (
                f"{args.yolo_class_id} "
                f"{meta['yolo_cx']:.6f} "
                f"{meta['yolo_cy']:.6f} "
                f"{meta['yolo_w']:.6f} "
                f"{meta['yolo_h']:.6f}\n"
            )
        with open(yolo_dir / f"{fname}.txt", "w", encoding="utf-8") as f:
            f.write(yolo_text)

        if collect_viz and i < 16:
            viz_images.append(img)
            viz_heatmaps.append(hm)
            viz_metas.append(meta)

        if (i + 1) % 1000 == 0 or i == args.num_images - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            print(f"[{i + 1}/{args.num_images}]  {rate:.0f} img/s  "
                  f"elapsed: {elapsed:.1f}s")

    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(CSV_HEADER)
        writer.writerows(csv_rows)

    if collect_viz and len(viz_images) > 0:
        viz_path = str(output_dir / "samples_preview.png")
        visualize_samples(viz_images, viz_heatmaps, viz_metas, viz_path)

    elapsed = time.time() - t0
    print(f"\nDone! Generated {args.num_images} samples in {elapsed:.1f}s")
    print(f"Output: {output_dir}")
    print(f"  images/     - {args.num_images} PNG files")
    print(f"  heatmaps/   - {args.num_images} NPY files")
    print(
        f"                shape={args.output_size[1] // args.heatmap_stride}x"
        f"{args.output_size[0] // args.heatmap_stride} stride={args.heatmap_stride}"
    )
    print(f"  labels_yolo/ - {args.num_images} TXT files")
    print(f"  labels.csv  - center point + fitted ellipse + bbox metadata")
    print(f"  config.json - generation parameters")


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic CCTag dataset for heatmap regression"
    )
    parser.add_argument("--num_images", type=int, default=50000,
                        help="Number of images to generate (default: 50000)")
    parser.add_argument("--output_dir", type=str, default="./outputs/datasets/cctag_dataset",
                        help="Output directory")
    parser.add_argument("--output_size", type=parse_output_size, default=(128, 128),
                        help="Output image size. Use a single integer for square output or WIDTHxHEIGHT, e.g. 1920x1200.")
    parser.add_argument("--marker_style", type=str, default="cctag_source",
                        choices=["cctag_source", "classic", "reference"],
                        help="Marker style. 'cctag_source' matches assets/markers/CCTag/display_marker.py.")
    parser.add_argument("--marker_id", type=int, default=0,
                        help="Marker ID from the CCTag source table (default: 0)")
    parser.add_argument("--num_rings", type=int, default=3, choices=[3, 4, 5],
                        help="Ring count. Use 3 or 4 for cctag_source, 5 for classic.")
    parser.add_argument("--marker_min", type=int, default=24,
                        help="Min marker diameter in px (simulates far distance)")
    parser.add_argument("--marker_max", type=int, default=80,
                        help="Max marker diameter in px (simulates close distance)")
    parser.add_argument("--occ_min", type=float, default=0.0,
                        help="Min occlusion ratio (default: 0.0)")
    parser.add_argument("--occ_max", type=float, default=0.6,
                        help="Max occlusion ratio (default: 0.6)")
    parser.add_argument("--occlusion_style", type=str, default="standard",
                        choices=["standard", "aggressive", "center_heavy"],
                        help="Occlusion pattern style. 'aggressive' and 'center_heavy' create thick center-crossing blockers.")
    parser.add_argument("--partial_out_prob", type=float, default=0.25,
                        help="Probability of placing a marker partially outside the frame (default: 0.25)")
    parser.add_argument("--partial_out_max_ratio", type=float, default=0.35,
                        help="How far a marker may extend beyond the frame, as a fraction of marker size (default: 0.35)")
    parser.add_argument("--empty_negative_prob", type=float, default=0.0,
                        help="Probability of generating a pure negative sample with no CCTag at all (default: 0.0)")
    parser.add_argument("--negative_ratio", type=float, default=None,
                        help="Exact fraction of pure negative samples in the dataset. Overrides --empty_negative_prob when set.")
    parser.add_argument("--empty_negative_ratio", type=float, default=None,
                        help="Exact fraction of pure negative samples in the dataset. Must be used with --boundary_target_ratio.")
    parser.add_argument("--boundary_target_ratio", type=float, default=None,
                        help="Exact fraction of boundary-clamped target samples in the dataset. Must be used with --empty_negative_ratio.")
    parser.add_argument("--heatmap_stride", type=int, default=4,
                        help="Downsampling factor for heatmap output, e.g. 4 gives WIDTH/4 x HEIGHT/4 heatmaps")
    parser.add_argument("--heatmap_sigma", type=float, default=2.0,
                        help="Gaussian heatmap sigma in heatmap coordinates (default: 2.0)")
    parser.add_argument("--yolo_class_id", type=int, default=0,
                        help="YOLO class id to write into labels_yolo/*.txt")
    parser.add_argument("--degradation_preset", type=str, default="standard",
                        choices=["standard", "soft_focus"],
                        help="Image degradation style. 'soft_focus' produces lower-contrast blurred markers similar to defocused captures.")
    parser.add_argument("--soft_focus_strength", type=float, default=0.0,
                        help="Extra low-contrast defocus strength in [0, 1]. Useful with --degradation_preset soft_focus.")
    parser.add_argument("--background_complexity", type=str, default="standard",
                        choices=["standard", "complex"],
                        help="Background complexity. 'complex' adds random curves, arcs, and concentric rings "
                             "that act as hard-negative confusers.")
    parser.add_argument("--overexposure_prob", type=float, default=0.0,
                        help="Probability of applying overexposure simulation (gamma compression + brightness boost) "
                             "per sample. Useful for training robustness against blown-out scenes. (default: 0.0)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--visualize", action="store_true",
                        help="Save a preview grid of sample images")

    args = parser.parse_args()
    validate_args(args, parser)
    generate_dataset(args)


if __name__ == "__main__":
    main()
