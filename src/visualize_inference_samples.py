#!/usr/bin/env python3
"""
Pick a few samples from each testing suite, run inference, and produce
before/after heatmap overlay comparison montages.

Example:
  python src/visualize_inference_samples.py \
    --checkpoint ./outputs/runs/experiment/best.pt \
    --suites_dir ./outputs/testing --output ./outputs/inference/viz_montage \
    --num_samples 6
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

# ── import helpers from sibling scripts ───────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from infer_cctag_heatmap import (
    IMAGE_EXTS,
    compute_peak_sharpness,
    decode_center,
    decode_center_offset,
    decode_center_subpixel,
    decode_center_weighted,
    load_heatmap,
    load_model,
    preprocess,
    resize_heatmap_to_shape,
    resolve_heatmap_path,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Inference visualization montages per test suite")
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="Path to model checkpoint (best.pt)")
    p.add_argument("--suites_dir", type=Path, default=Path("outputs/testing"),
                   help="Root directory containing test suite subdirectories")
    p.add_argument("--suites", type=str, nargs="*", default=None,
                   help="Specific suite names to visualize (default: auto-discover under suites_dir)")
    p.add_argument("--output", type=Path, default=Path("outputs/inference/viz_montage"),
                   help="Output directory for montage images")
    p.add_argument("--num_samples", type=int, default=6,
                   help="Number of samples to pick per suite (default: 6)")
    p.add_argument("--device", type=str, default=None,
                   help="cpu / cuda / cuda:0 (default: auto)")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Heatmap peak threshold for detection")
    p.add_argument("--decode_method", type=str, default="weighted",
                   choices=["weighted", "subpixel", "argmax"],
                   help="Peak localization method (default: weighted)")
    p.add_argument("--min_peak_sharpness", type=float, default=0.0,
                   help="Minimum peak sharpness to accept a detection")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for sample selection")
    return p.parse_args()


def resolve_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _draw_dot(img: np.ndarray, cx: int, cy: int, radius: int = 6,
              color: tuple[int, int, int] = (0, 255, 0), thickness: int = -1) -> None:
    cv2.circle(img, (cx, cy), radius, color, thickness)


def _overlay_heatmap(orig_bgr: np.ndarray, heatmap: np.ndarray,
                     alpha: float = 0.45) -> np.ndarray:
    """Alpha-blend JET-coloured heatmap onto the original image."""
    h, w = orig_bgr.shape[:2]
    hm_resized = cv2.resize(heatmap, (w, h))
    hm_u8 = (np.clip(hm_resized, 0, 1) * 255).astype(np.uint8)
    hm_color = cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)
    return cv2.addWeighted(orig_bgr, 1.0 - alpha, hm_color, alpha, 0)


def _decode_center_by_method(heatmap: np.ndarray, threshold: float, method: str,
                               offset: np.ndarray | None = None) -> tuple[float, float] | None:
    if offset is not None:
        return decode_center_offset(heatmap, offset, threshold=threshold)
    if method == "weighted":
        return decode_center_weighted(heatmap, threshold=threshold)
    elif method == "subpixel":
        return decode_center_subpixel(heatmap, threshold=threshold)
    else:
        return decode_center(heatmap, threshold=threshold)


def _to_image_coords(hm_coord: tuple[float, float], hm_shape: tuple[int, int],
                      img_shape: tuple[int, int]) -> tuple[int, int]:
    """Convert (cx_hm, cy_hm) to integer pixel coordinates on the original image."""
    hm_h, hm_w = hm_shape
    img_h, img_w = img_shape[:2]
    cx = int(hm_coord[0] * img_w / hm_w)
    cy = int(hm_coord[1] * img_h / hm_h)
    return cx, cy


def _is_suite_dir(path: Path) -> bool:
    """Check whether a directory looks like a test suite."""
    return ((path / "images").is_dir()
            and (path / "labels.csv").is_file()
            and (path / "heatmaps").is_dir())


def _resolve_suites(suites_dir: Path, suite_specs: list[str] | None) -> list[Path]:
    """Resolve suite specs to a flat list of suite directories.

    Each spec can be:
    - A direct suite directory (has images/, labels.csv, heatmaps/)
    - A directory containing suites (auto-discovered recursively)
    - A name relative to suites_dir
    """
    if not suite_specs:
        return discover_suites(suites_dir)

    result: list[Path] = []
    for spec in suite_specs:
        p = Path(spec)
        candidate = p if p.exists() else suites_dir / spec
        if _is_suite_dir(candidate):
            result.append(candidate)
        elif candidate.is_dir():
            result.extend(discover_suites(candidate))
        else:
            raise SystemExit(f"Suite spec '{spec}' does not resolve to a directory: {candidate}")
    return result


def discover_suites(suites_dir: Path) -> list[Path]:
    """Find all suite directories that contain images/ and labels.csv."""
    found = []
    for candidate in sorted(suites_dir.rglob("images")):
        parent = candidate.parent
        if _is_suite_dir(parent):
            found.append(parent)
    return found


def load_gt_map(suite_dir: Path) -> dict[str, dict[str, Any]]:
    labels_csv = suite_dir / "labels.csv"
    heatmaps_dir = suite_dir / "heatmaps"
    gt_map: dict[str, dict[str, Any]] = {}
    with labels_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row.get("filename", "").strip()
            if not filename:
                continue
            gt_map[filename] = {
                "heatmap_path": resolve_heatmap_path(heatmaps_dir, filename),
                "center_x": float(row.get("center_x") or row.get("x") or -1.0),
                "center_y": float(row.get("center_y") or row.get("y") or -1.0),
                "is_negative": int(row.get("is_negative") or 0) == 1,
                "has_visible_marker": int(row.get("has_visible_marker") or 0) == 1,
            }
    return gt_map


def select_samples(image_paths: list[Path], gt_map: dict[str, dict[str, Any]],
                   num_samples: int, rng: np.random.Generator) -> list[Path]:
    """Pick samples with a mix of positive and negative examples."""
    positive = [p for p in image_paths if p.stem in gt_map and not gt_map[p.stem]["is_negative"]]
    negative = [p for p in image_paths if p.stem in gt_map and gt_map[p.stem]["is_negative"]]
    # also include any images not in gt_map
    other = [p for p in image_paths if p.stem not in gt_map]

    num_pos = min(len(positive), max(1, num_samples // 2))
    num_neg = min(len(negative), max(1, num_samples - num_pos))
    num_other = min(len(other), num_samples - num_pos - num_neg)

    selected: list[Path] = []
    if positive:
        idxs = rng.choice(len(positive), size=num_pos, replace=False)
        selected.extend(positive[i] for i in idxs)
    if negative:
        idxs = rng.choice(len(negative), size=num_neg, replace=False)
        selected.extend(negative[i] for i in idxs)
    if other:
        idxs = rng.choice(len(other), size=num_other, replace=False)
        selected.extend(other[i] for i in idxs)
    # fill remaining from any pool
    remaining = num_samples - len(selected)
    if remaining > 0:
        pool = [p for p in image_paths if p not in set(selected)]
        if pool:
            idxs = rng.choice(len(pool), size=min(remaining, len(pool)), replace=False)
            selected.extend(pool[i] for i in idxs)
    rng.shuffle(selected)
    return selected


def _split_pred(out):
    if isinstance(out, tuple):
        return out[0], out[1]
    return out, None


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    rng = np.random.default_rng(args.seed)

    # ── load model ──────────────────────────────────────────────────────────
    print(f"Loading checkpoint: {args.checkpoint}")
    model, config = load_model(args.checkpoint, device)
    model.eval()

    input_w = config.get("input_width", 640)
    input_h = config.get("input_height", 400)
    backbone = config.get("backbone", "efficientnet_b0")
    use_offset_head = config.get("use_offset_head", False)
    print(f"  backbone={backbone}  input={input_w}x{input_h}  device={device}")

    # ── discover suites ─────────────────────────────────────────────────────
    suite_dirs = _resolve_suites(args.suites_dir, args.suites)

    if not suite_dirs:
        raise SystemExit(f"No evaluable suites found under {args.suites_dir}")

    # ── process each suite ──────────────────────────────────────────────────
    for suite_dir in suite_dirs:
        suite_name = suite_dir.name
        images_dir = suite_dir / "images"
        image_paths = sorted(p for p in images_dir.iterdir()
                             if p.suffix.lower() in IMAGE_EXTS)
        if not image_paths:
            print(f"  [{suite_name}] no images, skipping")
            continue

        gt_map = load_gt_map(suite_dir)
        selected = select_samples(image_paths, gt_map, args.num_samples, rng)

        suite_out = args.output / suite_name
        suite_out.mkdir(parents=True, exist_ok=True)
        print(f"\n  [{suite_name}] processing {len(selected)} samples -> {suite_out}")

        with torch.inference_mode():
            for img_path in selected:
                orig_bgr = cv2.imread(str(img_path))
                if orig_bgr is None:
                    continue

                # ── inference ───────────────────────────────────────────
                tensor, (orig_w, orig_h) = preprocess(img_path, input_w, input_h, device)
                out = model(tensor)
                batch_hm, batch_off = _split_pred(out)
                heatmap = batch_hm[0, 0].float().cpu().numpy()
                hm_h, hm_w = heatmap.shape
                offset_np = batch_off[0].float().cpu().numpy() if batch_off is not None else None

                result = _decode_center_by_method(heatmap, args.threshold,
                                                   args.decode_method, offset_np)

                # Sharpness filter
                if result is not None and args.min_peak_sharpness > 0:
                    sharpness = compute_peak_sharpness(heatmap)
                    if sharpness < args.min_peak_sharpness:
                        result = None

                stem = img_path.stem

                # ── Panel 1: original ───────────────────────────────────
                p1 = orig_bgr.copy()

                # ── Panel 2: prediction heatmap + dot ────────────────────
                p2 = _overlay_heatmap(orig_bgr.copy(), heatmap)
                if result is not None:
                    cx, cy = _to_image_coords(result, (hm_h, hm_w), orig_bgr.shape)
                    _draw_dot(p2, cx, cy, color=(0, 255, 0))

                # ── Panel 3: GT heatmap + dot ────────────────────────────
                gt_row = gt_map.get(stem)
                p3 = orig_bgr.copy()
                if gt_row is not None and gt_row["heatmap_path"].is_file():
                    gt_hm_raw = load_heatmap(gt_row["heatmap_path"])
                    gt_hm = resize_heatmap_to_shape(gt_hm_raw, heatmap.shape)
                    p3 = _overlay_heatmap(orig_bgr.copy(), gt_hm)
                    if not gt_row["is_negative"]:
                        gt_cx = int(gt_row["center_x"])
                        gt_cy = int(gt_row["center_y"])
                        _draw_dot(p3, gt_cx, gt_cy, color=(255, 0, 0))

                # ── save ────────────────────────────────────────────────
                out_path = suite_out / f"{stem}.png"
                cv2.imwrite(str(out_path), np.hstack([p1, p2, p3]))

        print(f"    saved {len(selected)} samples")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        raise SystemExit(130)
