#!/usr/bin/env python3
"""Augment a ROI heatmap dataset with synthetic occlusion over real markers.

Takes a labeled ROI dataset (e.g. ``outputs/datasets/6f_labeled_1024x640_roi``
produced by ``src/sample_roi_dataset.py``) and writes a NEW dataset that adds
occluded copies of every positive sample, reusing the same occluder library as
the synthetic generator (``apply_random_occlusion`` in
``src/generate_cctag_dataset.py``).

Occlusion is drawn over the real marker; it does NOT move the center, so the
center label and the existing heatmap stay valid (the model keeps learning to
regress the center under occlusion). Only ``occlusion_ratio`` is updated to the
measured coverage. Negatives are never occluded.

By default (``--realistic``) the flat geometric occluder from
``apply_random_occlusion`` is dressed up to match the real rig (see
``capture_20260608_*.png``): curved cables crossing the marker, bright metallic
glints on the occluder, and feathered / motion-blurred edges. Pass
``--no-realistic`` for the legacy flat-dark-block behaviour.

Tiers mirror ``scripts/generate_training_sets.sh`` (occlusion_style=aggressive
for all): variants alternate between a low/partial range and a hard range, so
half the occluded copies are partial and half are heavy.

The source dataset is never modified; the output goes to a fresh directory and
is drop-in compatible with ``src/train_cctag_heatmap_ddp.py`` (images/,
heatmaps/ NPZ, labels_yolo/, labels.csv with 23 columns, config.json).

Defaults: occluded positives only (``--no-keep_clean``), and markers whose
ellipse extends beyond the ROI frame are skipped (``--skip_out_of_frame``).
Pass ``--keep_clean`` to also copy the clean originals + negatives, or
``--no-skip_out_of_frame`` to occlude cut-off markers too.

Example (canonical: occluded positives only, in-frame markers only):
    uv run python src/augment_roi_occlusion.py \
        --input_dir ./outputs/datasets/6f_labeled_1024x640_roi \
        --output_dir ./outputs/datasets/6f_labeled_1024x640_roi_occ \
        --variants_per_positive 2 --no-keep_clean --seed 42
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

# Reuse the synthetic occluder library. generate_cctag_dataset.py only defines
# constants at module scope (main is guarded by __name__), so the import is
# side-effect free. Adding the script's own directory keeps the import working
# regardless of the caller's CWD.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_cctag_dataset import apply_random_occlusion  # noqa: E402

# 23-column schema, identical order to src/sample_roi_dataset.py:38-46.
CSV_HEADER = [
    "filename", "x", "y", "center_x", "center_y",
    "ellipse_cx", "ellipse_cy", "ellipse_a", "ellipse_b", "ellipse_angle_rad",
    "occlusion_ratio",
    "bbox_xmin", "bbox_ymin", "bbox_xmax", "bbox_ymax",
    "yolo_cx", "yolo_cy", "yolo_w", "yolo_h",
    "is_negative", "negative_mode", "has_visible_marker", "visible_marker_ratio",
    "target_clamped",
]


def ensure_absent(target: Path) -> None:
    if target.exists():
        raise SystemExit(f"Refusing to overwrite existing path: {target}")


def marker_radius_from_row(row: dict, scale: float) -> float:
    """Occlusion drawing radius from the fitted ellipse.

    ``ellipse_a/b`` are the outer semi-axes (the bbox is ~2*ellipse_a wide).
    The synthetic generator passes the inner radius (~half the outer radius) to
    apply_random_occlusion, so we mirror that with (a+b)/4 to keep occluder
    sizing and the measured occlusion_ratio consistent with the trained-on
    synthetic data."""
    a = float(row.get("ellipse_a") or 0.0)
    b = float(row.get("ellipse_b") or 0.0)
    if a > 0.0 and b > 0.0:
        return max((a + b) / 4.0 * scale, 8.0)
    # Fallback: derive from the bbox if the ellipse fit is missing.
    bw = float(row.get("bbox_xmax") or 0.0) - float(row.get("bbox_xmin") or 0.0)
    bh = float(row.get("bbox_ymax") or 0.0) - float(row.get("bbox_ymin") or 0.0)
    return max((bw + bh) / 8.0 * scale, 8.0)


def marker_exceeds_frame(row: dict, frame_w: int, frame_h: int) -> bool:
    """True if any part of the marker ellipse extends beyond the ROI frame.

    Uses the axis-aligned half-extents of the rotated ellipse (semi-axes
    ellipse_a/b, angle ellipse_angle_rad). Such markers are already cut off by
    the image boundary, so we don't add occlusion on top of them."""
    cx = float(row.get("center_x") or row.get("x") or 0.0)
    cy = float(row.get("center_y") or row.get("y") or 0.0)
    a = float(row.get("ellipse_a") or 0.0)
    b = float(row.get("ellipse_b") or 0.0)
    th = float(row.get("ellipse_angle_rad") or 0.0)
    ext_x = math.hypot(a * math.cos(th), b * math.sin(th))
    ext_y = math.hypot(a * math.sin(th), b * math.cos(th))
    return (cx - ext_x < 0.0 or cx + ext_x > frame_w
            or cy - ext_y < 0.0 or cy + ext_y > frame_h)


def is_positive(row: dict) -> bool:
    if int(float(row.get("is_negative", "1") or 1)) != 0:
        return False
    cx = float(row.get("center_x") or row.get("x") or -1.0)
    cy = float(row.get("center_y") or row.get("y") or -1.0)
    return np.isfinite(cx) and np.isfinite(cy) and cx >= 0.0 and cy >= 0.0


def copy_clean(stem: str, in_dir: Path, out_dir: Path) -> None:
    """Copy an existing sample's image / heatmap / yolo files verbatim."""
    shutil.copyfile(in_dir / "images" / f"{stem}.png", out_dir / "images" / f"{stem}.png")
    shutil.copyfile(in_dir / "heatmaps" / f"{stem}.npz", out_dir / "heatmaps" / f"{stem}.npz")
    src_txt = in_dir / "labels_yolo" / f"{stem}.txt"
    dst_txt = out_dir / "labels_yolo" / f"{stem}.txt"
    if src_txt.is_file():
        shutil.copyfile(src_txt, dst_txt)
    else:
        dst_txt.write_text("")


# --------------------------------------------------------------------------- #
# Realism post-processing
#
# apply_random_occlusion() paints flat dark geometric blocks. Real captures
# (see capture_20260608_*.png) show the marker occluded by a metal/printed
# mount with specular highlights, thin curved cables, soft/motion-blurred
# edges. These helpers add those cues on top of the geometric occluder so the
# augmented occlusion matches the real rig more closely.
# --------------------------------------------------------------------------- #

def occluder_mask_from_diff(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    """Binary mask (uint8 0/255) of pixels apply_random_occlusion changed."""
    diff = np.abs(after.astype(np.int16) - before.astype(np.int16)).sum(axis=2)
    return ((diff > 8).astype(np.uint8)) * 255


def _bezier_points(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, n: int = 48) -> np.ndarray:
    """Quadratic Bezier polyline as an int32 (n, 2) point array for cv2."""
    t = np.linspace(0.0, 1.0, n)[:, None]
    pts = (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2
    return np.round(pts).astype(np.int32)


def _cubic_bezier_points(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, n: int = 64) -> np.ndarray:
    """Cubic Bezier polyline as an int32 (n, 2) point array for cv2."""
    t = np.linspace(0.0, 1.0, n)[:, None]
    pts = (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3
    return np.round(pts).astype(np.int32)


def draw_cables(img: np.ndarray, mask: np.ndarray, cx: float, cy: float, r: float,
                rng: random.Random, n_cables: int) -> None:
    """Draw curved 3D-shaded cables crossing the marker, updating the occluder mask.

    Uses cubic Bezier curves to allow S-curves. Renders cables using multi-layer
    drawing (base, shadow, light sheen, specular glint) to create a realistic
    3D cylindrical appearance.
    """
    center = np.array([cx, cy], dtype=np.float64)

    for _ in range(n_cables):
        # 1. Generate cubic Bezier path crossing the marker area
        span = r * rng.uniform(2.2, 3.8)
        ang0 = rng.uniform(0.0, 2 * math.pi)
        ang3 = ang0 + rng.uniform(math.pi * 0.6, math.pi * 1.4)

        p0 = center + span * np.array([math.cos(ang0), math.sin(ang0)])
        p3 = center + span * np.array([math.cos(ang3), math.sin(ang3)])

        # Midpoint close to center to ensure occlusion
        ang_mid = rng.uniform(0.0, 2 * math.pi)
        p_mid = center + r * rng.uniform(0.0, 0.8) * np.array([math.cos(ang_mid), math.sin(ang_mid)])

        # Control points
        d0 = np.linalg.norm(p_mid - p0)
        d3 = np.linalg.norm(p_mid - p3)

        p1 = p0 + (p_mid - p0) * rng.uniform(0.4, 0.7)
        p2 = p3 + (p_mid - p3) * rng.uniform(0.4, 0.7)

        # Add some perpendicular perturbation to control points for S-curves
        perp0 = np.array([-(p_mid[1] - p0[1]), p_mid[0] - p0[0]]) / (d0 + 1e-6)
        perp3 = np.array([-(p_mid[1] - p3[1]), p_mid[0] - p3[0]]) / (d3 + 1e-6)

        p1 += perp0 * rng.uniform(-0.5, 0.5) * r
        p2 += perp3 * rng.uniform(-0.5, 0.5) * r

        pts = _cubic_bezier_points(p0, p1, p2, p3)

        # 2. Cable properties
        thick = int(max(4, rng.uniform(0.06, 0.20) * r))

        # Muted colors: mostly black/dark-grey, occasionally industrial yellow, blue, or green
        color_type = rng.choice(["grey", "grey", "grey", "yellow", "blue", "green"])
        if color_type == "grey":
            g = rng.randint(20, 55)
            base_bgr = np.array([g, g, g], dtype=np.float32)
        elif color_type == "yellow":
            # Industrial yellow/orange: high R & G, low B
            base_bgr = np.array([rng.randint(20, 45), rng.randint(90, 140), rng.randint(110, 160)], dtype=np.float32)
        elif color_type == "blue":
            # Muted blue: high B, lower R & G
            base_bgr = np.array([rng.randint(100, 150), rng.randint(60, 95), rng.randint(40, 75)], dtype=np.float32)
        else: # green
            # Industrial green/teal
            base_bgr = np.array([rng.randint(60, 95), rng.randint(90, 130), rng.randint(40, 70)], dtype=np.float32)

        # Draw base cable
        cv2.polylines(img, [pts], False, base_bgr.astype(np.uint8).tolist(), thick, cv2.LINE_AA)
        cv2.polylines(mask, [pts], False, 255, thick, cv2.LINE_AA)

        # 3. 3D shading layers
        # Light direction for cable highlight offset
        light_ang = rng.uniform(0.0, 2 * math.pi)
        ldx = math.cos(light_ang)
        ldy = math.sin(light_ang)

        # Shadow side (offset in direction of light_ang + pi)
        sh_thick = max(1, int(thick * 0.75))
        sh_offset = np.array([-ldx, -ldy]) * rng.uniform(0.5, 1.5)
        pts_sh = (pts + sh_offset).astype(np.int32)
        sh_color = (base_bgr * 0.55).astype(np.uint8).tolist()
        cv2.polylines(img, [pts_sh], False, sh_color, sh_thick, cv2.LINE_AA)

        # Highlight side (offset in direction of light_ang)
        hl_thick = max(1, int(thick * 0.35))
        hl_offset = np.array([ldx, ldy]) * rng.uniform(0.5, 1.5)
        pts_hl = (pts + hl_offset).astype(np.int32)
        hl_color = np.clip(base_bgr * 1.5 + 40, 0, 255).astype(np.uint8).tolist()
        cv2.polylines(img, [pts_hl], False, hl_color, hl_thick, cv2.LINE_AA)

        # 4. Specular glint along a portion of the cable
        if rng.random() < 0.7:
            glint_len = rng.randint(15, 35)
            max_start = len(pts) - glint_len - 5
            if max_start > 5:
                start_i = rng.randint(5, max_start)
                pts_glint = pts_hl[start_i : start_i + glint_len]
                if len(pts_glint) > 1:
                    glint_thick = max(1, int(thick * 0.15))
                    glint_color = [240, 240, 240]
                    cv2.polylines(img, [pts_glint], False, glint_color, glint_thick, cv2.LINE_AA)


def round_mask(mask: np.ndarray, rng: random.Random) -> np.ndarray:
    """Round the sharp rectangle corners and irregularise the occluder outline.

    We first round convex corners using Gaussian blur + threshold,
    then apply a smooth, low-frequency coordinate displacement (warping)
    to break up the straight lines and add organic, physical irregularities.
    """
    # 1. Rounding corners
    k = rng.choice((15, 21, 27, 35))
    m = cv2.GaussianBlur(mask, (k, k), 0)
    _, m = cv2.threshold(m, 127, 255, cv2.THRESH_BINARY)

    # 2. Add organic border irregularities using low-frequency coordinate displacement
    h, w = mask.shape[:2]
    # Downsample grid to ensure low frequency noise
    nh, nw = max(4, h // 16), max(4, w // 16)
    dx = np.random.normal(0.0, 1.0, (nh, nw)).astype(np.float32)
    dy = np.random.normal(0.0, 1.0, (nh, nw)).astype(np.float32)
    # Smooth the noise by upsampling to full resolution
    dx = cv2.resize(dx, (w, h), interpolation=cv2.INTER_CUBIC)
    dy = cv2.resize(dy, (w, h), interpolation=cv2.INTER_CUBIC)

    # Random displacement scale
    scale = rng.uniform(3.0, 10.0)
    dx *= scale
    dy *= scale

    # Map coordinates and warp
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    map_x = xs + dx
    map_y = ys + dy

    m = cv2.remap(m, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return m.astype(np.uint8)


def shade_occluder_surface(img: np.ndarray, mask: np.ndarray, rng: random.Random) -> None:
    """Replace the flat-black fill with a shaded dark-grey surface.

    Creates a highly realistic 3D lighting appearance by combining:
    1. Ambient light (base intensity)
    2. Directional light (linear gradient with random direction)
    3. Point light source / spotlight (radial highlight with random center and size)
    4. Matte plastic/metal surface texture (fine Gaussian noise)
    """
    sel = mask > 0
    if not sel.any():
        return
    h, w = mask.shape[:2]
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    sx, sy = np.where(sel)

    # 1. Linear directional light
    ang = rng.uniform(0.0, 2 * math.pi)
    grad_lin = math.cos(ang) * xs + math.sin(ang) * ys
    g_lin = grad_lin[sx, sy]
    g_lin = (g_lin - g_lin.min()) / (np.ptp(g_lin) + 1e-6)

    # 2. Radial highlight (spotlight/point light)
    cx_mask, cy_mask = sy.mean(), sx.mean()
    light_x = cx_mask + rng.uniform(-0.5 * w, 0.5 * w)
    light_y = cy_mask + rng.uniform(-0.5 * h, 0.5 * h)

    dist_sq = (xs - light_x) ** 2 + (ys - light_y) ** 2
    # Radius of highlight is related to the scale of the mask/image
    sigma = rng.uniform(0.3, 0.8) * max(h, w)
    grad_rad = np.exp(-dist_sq / (2.0 * sigma ** 2))
    g_rad = grad_rad[sx, sy]

    # Combine components
    ambient = rng.uniform(25.0, 50.0)
    linear_amp = rng.uniform(15.0, 45.0)
    radial_amp = rng.uniform(15.0, 40.0)

    # Noise/texture
    noise = np.random.normal(0.0, rng.uniform(1.5, 4.0), size=g_lin.shape)

    # Calculate final pixel values
    vals = ambient + g_lin * linear_amp + g_rad * radial_amp + noise
    vals = np.clip(vals, 0.0, 160.0).astype(np.uint8)

    # Apply to BGR channels
    img[sx, sy] = vals[:, None]


def add_rim_highlight(img: np.ndarray, mask: np.ndarray, rng: random.Random) -> None:
    """Soft grey metallic sheen along one edge of the occluder (not white spots)."""
    # 1. Distance transform to define a band parallel to the edge
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)

    # Peak distance from edge and width of the band
    d_peak = rng.uniform(1.0, 4.0)
    d_width = rng.uniform(2.0, 6.0)

    # Calculate band intensity
    band = np.exp(-((dist - d_peak) / d_width) ** 2)

    # 2. Gate to restrict to one side
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return
    cx_mask, cy_mask = xs.mean(), ys.mean()

    ang = rng.uniform(0.0, 2 * math.pi)
    h, w = mask.shape[:2]
    ys_grid, xs_grid = np.mgrid[0:h, 0:w].astype(np.float32)
    proj = (xs_grid - cx_mask) * math.cos(ang) + (ys_grid - cy_mask) * math.sin(ang)

    proj_mask = proj[ys, xs]
    p_min, p_max = proj_mask.min(), proj_mask.max()
    p_range = p_max - p_min if p_max > p_min else 1.0

    # Soft sigmoid gate selecting one side
    gate = 1.0 / (1.0 + np.exp(-(proj - (p_min + 0.65 * p_range)) / (0.08 * p_range + 1e-5)))

    # Final highlight mask restricted to the occluder surface
    highlight = band * gate * (mask > 0)
    highlight = highlight.astype(np.float32)

    # Soft blend with a metallic grey value
    val = rng.uniform(110.0, 175.0)
    a = highlight[..., None]
    img[:] = np.clip(img.astype(np.float32) * (1.0 - a) + val * a, 0.0, 255.0).astype(np.uint8)


def _motion_blur(img: np.ndarray, length: int, angle_deg: float) -> np.ndarray:
    """Directional (motion) blur with a length-px line kernel at angle_deg."""
    if length < 3:
        return img
    kernel = np.zeros((length, length), dtype=np.float32)
    kernel[length // 2, :] = 1.0
    rot = cv2.getRotationMatrix2D((length / 2.0 - 0.5, length / 2.0 - 0.5), angle_deg, 1.0)
    kernel = cv2.warpAffine(kernel, rot, (length, length))
    s = kernel.sum()
    if s > 0:
        kernel /= s
    return cv2.filter2D(img, -1, kernel)


def soften_occluder(original: np.ndarray, occluded: np.ndarray, mask: np.ndarray,
                    rng: random.Random, edge_blur: int, motion_blur_max: int
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Feather the occluder edge (+ optional motion blur) and alpha-composite.

    Returns (composited_uint8, alpha_float32) where alpha in [0,1] is the soft
    occluder coverage used to recompute occlusion_ratio.
    """
    alpha = mask.astype(np.float32) / 255.0
    k = edge_blur
    if k and k % 2 == 0:
        k += 1
    if k >= 3:
        alpha = cv2.GaussianBlur(alpha, (k, k), 0)

    layer = occluded.astype(np.float32)
    if motion_blur_max >= 3:
        mlen = rng.randint(0, motion_blur_max)
        if mlen >= 3:
            layer = _motion_blur(layer, mlen, rng.uniform(0.0, 180.0))
            ak = mlen if mlen % 2 == 1 else mlen + 1
            alpha = cv2.GaussianBlur(alpha, (ak, ak), 0)  # motion softens the edge too

    a = alpha[..., None]
    out = original.astype(np.float32) * (1.0 - a) + layer * a
    return np.clip(out, 0.0, 255.0).astype(np.uint8), alpha


def ratio_in_marker(alpha: np.ndarray, cx: float, cy: float, radius: float) -> float:
    """Soft occlusion coverage of the marker disc (matches the circle used by
    apply_random_occlusion: radius = max(marker_radius, 10))."""
    circle = np.zeros(alpha.shape, dtype=np.float32)
    cv2.circle(circle, (int(round(cx)), int(round(cy))), int(round(max(radius, 10.0))), 1.0, -1)
    denom = float(circle.sum()) + 1e-6
    return min(float((alpha * circle).sum() / denom), 1.0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input_dir", type=Path, default=Path("outputs/datasets/6f_labeled_1024x640_roi"))
    ap.add_argument("--output_dir", type=Path, default=Path("outputs/datasets/6f_labeled_1024x640_roi_occ_v2"))
    ap.add_argument("--variants_per_positive", type=int, default=2,
                    help="Occluded copies generated per positive sample.")
    ap.add_argument("--keep_clean", action=argparse.BooleanOptionalAction, default=False,
                    help="Also copy the clean originals (positives + negatives) into the output "
                         "so the dataset is self-contained. Default off (occluded positives only).")
    ap.add_argument("--occ_low_min", type=float, default=0.05)
    ap.add_argument("--occ_low_max", type=float, default=0.50)
    ap.add_argument("--occ_hard_min", type=float, default=0.50)
    ap.add_argument("--occ_hard_max", type=float, default=0.85)
    ap.add_argument("--occlusion_style", default="aggressive",
                    choices=["standard", "aggressive", "center_heavy"])
    ap.add_argument("--occluder_templates", default="auto")
    ap.add_argument("--occ_radius_scale", type=float, default=1.0,
                    help="Multiplier on the (a+b)/4 occluder radius.")
    ap.add_argument("--realistic", action=argparse.BooleanOptionalAction, default=True,
                    help="Add real-rig occlusion cues on top of the geometric occluder: "
                         "curved cables, metallic highlights, soft/motion-blurred edges. "
                         "Use --no-realistic for the legacy flat-dark-block behaviour.")
    ap.add_argument("--cable_prob", type=float, default=0.6,
                    help="Probability of drawing 1-2 curved cables across the marker.")
    ap.add_argument("--metallic_prob", type=float, default=0.5,
                    help="Probability of adding bright metallic-glint streaks on the occluder.")
    ap.add_argument("--edge_blur", type=int, default=7,
                    help="Gaussian kernel (px) used to feather the occluder edge (0 disables).")
    ap.add_argument("--motion_blur_max", type=int, default=9,
                    help="Max directional motion-blur kernel (px); per-variant random in [0,max]. "
                         "0 disables.")
    ap.add_argument("--skip_out_of_frame", action=argparse.BooleanOptionalAction, default=True,
                    help="Skip occluding markers whose ellipse extends beyond the ROI frame "
                         "(they are already cut off by the boundary). Use --no-skip_out_of_frame "
                         "to occlude them anyway.")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.variants_per_positive < 1:
        raise SystemExit("--variants_per_positive must be >= 1")

    random.seed(args.seed)
    np.random.seed(args.seed)

    in_dir = args.input_dir
    src_csv = in_dir / "labels.csv"
    if not src_csv.is_file():
        raise FileNotFoundError(src_csv)

    out_dir = args.output_dir
    ensure_absent(out_dir)
    img_dir, hm_dir, yolo_dir = out_dir / "images", out_dir / "heatmaps", out_dir / "labels_yolo"
    for d in (img_dir, hm_dir, yolo_dir):
        d.mkdir(parents=True, exist_ok=True)

    with open(src_csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        src_rows = list(reader)
        header = reader.fieldnames or CSV_HEADER

    # Frame size for the out-of-frame test: prefer the source config, else the
    # first image's dimensions.
    frame_w = frame_h = None
    cfg_path = in_dir / "config.json"
    if cfg_path.is_file():
        cfg = json.loads(cfg_path.read_text())
        frame_w, frame_h = cfg.get("image_width"), cfg.get("image_height")
    if not frame_w or not frame_h:
        probe = cv2.imread(str(in_dir / "images" / f"{src_rows[0]['filename']}.png"))
        frame_h, frame_w = probe.shape[:2]
    frame_w, frame_h = int(frame_w), int(frame_h)

    tiers = [
        (args.occ_low_min, args.occ_low_max),
        (args.occ_hard_min, args.occ_hard_max),
    ]

    out_rows: list[dict] = []
    n_clean = n_occ = n_pos = n_oof = 0

    for row in src_rows:
        stem = row["filename"]

        if args.keep_clean:
            copy_clean(stem, in_dir, out_dir)
            out_rows.append(dict(row))
            n_clean += 1

        if not is_positive(row):
            continue
        n_pos += 1

        if args.skip_out_of_frame and marker_exceeds_frame(row, frame_w, frame_h):
            n_oof += 1
            continue

        img_path = in_dir / "images" / f"{stem}.png"
        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"[warn] cannot read {img_path}, skipping occlusion for it")
            continue

        cx = float(row.get("center_x") or row.get("x"))
        cy = float(row.get("center_y") or row.get("y"))
        radius = marker_radius_from_row(row, args.occ_radius_scale)

        for v in range(args.variants_per_positive):
            lo, hi = tiers[v % len(tiers)]
            base = img.copy()
            occluded, actual_ratio = apply_random_occlusion(
                base.copy(), (cx, cy), radius,
                occlusion_range=(lo, hi),
                occlusion_style=args.occlusion_style,
                occ_distribution="uniform",
                occluder_templates=args.occluder_templates,
            )

            if args.realistic:
                # Restyle the flat geometric block into a real-rig occluder:
                # round its edges, shade it as a lit dark-grey part, add an edge
                # sheen, then cables; finally feather + motion-blur and recompute
                # the coverage from the resulting soft alpha.
                mask = occluder_mask_from_diff(base, occluded)
                if mask.any():
                    mask = round_mask(mask, random)
                    shade_occluder_surface(occluded, mask, random)
                    if random.random() < args.metallic_prob:
                        add_rim_highlight(occluded, mask, random)
                if random.random() < args.cable_prob:
                    draw_cables(occluded, mask, cx, cy, radius, random,
                                random.randint(1, 2))
                occluded, alpha = soften_occluder(
                    base, occluded, mask, random, args.edge_blur, args.motion_blur_max)
                actual_ratio = ratio_in_marker(alpha, cx, cy, radius)

            out_stem = f"{stem}_occ{v}"
            cv2.imwrite(str(img_dir / f"{out_stem}.png"), occluded)
            # Center is unchanged -> reuse the source heatmap and yolo bbox.
            shutil.copyfile(in_dir / "heatmaps" / f"{stem}.npz", hm_dir / f"{out_stem}.npz")
            src_txt = in_dir / "labels_yolo" / f"{stem}.txt"
            (yolo_dir / f"{out_stem}.txt").write_text(
                src_txt.read_text() if src_txt.is_file() else ""
            )

            new_row = dict(row)
            new_row["filename"] = out_stem
            new_row["occlusion_ratio"] = f"{actual_ratio:.4f}"
            out_rows.append(new_row)
            n_occ += 1

    with open(out_dir / "labels.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(out_rows)

    config = {
        "generation_type": "cctag_roi_occlusion_augmented",
        "source_dataset": str(in_dir),
        "keep_clean": bool(args.keep_clean),
        "variants_per_positive": args.variants_per_positive,
        "occlusion_style": args.occlusion_style,
        "occluder_templates": args.occluder_templates,
        "occ_radius_scale": args.occ_radius_scale,
        "realistic": bool(args.realistic),
        "cable_prob": args.cable_prob,
        "metallic_prob": args.metallic_prob,
        "edge_blur": args.edge_blur,
        "motion_blur_max": args.motion_blur_max,
        "skip_out_of_frame": bool(args.skip_out_of_frame),
        "frame_size": [frame_w, frame_h],
        "tiers": {"low": [args.occ_low_min, args.occ_low_max],
                  "hard": [args.occ_hard_min, args.occ_hard_max]},
        "seed": args.seed,
        "num_source_rows": len(src_rows),
        "num_source_positives": n_pos,
        "num_skipped_out_of_frame": n_oof,
        "num_occluded_positives": n_pos - n_oof,
        "num_clean_copied": n_clean,
        "num_occluded": n_occ,
        "num_samples": len(out_rows),
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))

    print(f"Done. {n_pos} positives ({n_oof} out-of-frame skipped) -> {n_occ} occluded copies"
          + (f" + {n_clean} clean originals" if args.keep_clean else "")
          + f" = {len(out_rows)} samples")
    print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
