#!/usr/bin/env python3
"""Visualize false-positive images: original + predicted heatmap overlay + GT info."""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from infer_cctag_heatmap import decode_center_subpixel, load_model, preprocess


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize FP images from benchmark CSV")
    p.add_argument("--csv", type=Path, required=True,
                   help="evaluation_per_image.csv from benchmark")
    p.add_argument("--suite_dir", type=Path, required=True,
                   help="Test suite directory (contains images/)")
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="Model checkpoint for heatmap inference")
    p.add_argument("--output", type=Path, default=Path("outputs/tmp/fp_analysis.jpg"))
    p.add_argument("--tile_size", type=int, default=400)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--mode", choices=["fp", "fn"], default="fp",
                   help="Which errors to visualize")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config = load_model(args.checkpoint, device)
    model.eval()
    input_w = config.get("input_width", 640)
    input_h = config.get("input_height", 400)

    # Read CSV and filter FP/FN
    rows = list(csv.DictReader(args.csv.open(encoding="utf-8")))
    if args.mode == "fp":
        error_rows = [r for r in rows if int(r["fp_det"]) == 1]
    else:
        error_rows = [r for r in rows if int(r["fn_det"]) == 1]

    if not error_rows:
        print(f"No {args.mode.upper()} found.")
        return

    print(f"Found {len(error_rows)} {args.mode.upper()} images")

    images_dir = args.suite_dir / "images"
    tiles = []

    with torch.inference_mode():
        for r in error_rows:
            stem = r["filename"]
            # Find image file
            img_path = None
            for ext in (".png", ".jpg", ".jpeg"):
                candidate = images_dir / f"{stem}{ext}"
                if candidate.is_file():
                    img_path = candidate
                    break
            if img_path is None:
                continue

            # Original image
            orig = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if orig is None:
                continue
            oh, ow = orig.shape[:2]

            # Run model
            tensor, _ = preprocess(img_path, input_w, input_h, device)
            heatmap = model(tensor)[0, 0].cpu().numpy()
            peak_val = float(heatmap.max())
            result = decode_center_subpixel(heatmap, threshold=args.threshold)

            # Create heatmap color overlay
            hm_resized = cv2.resize(heatmap, (ow, oh), interpolation=cv2.INTER_LINEAR)
            hm_color = cv2.applyColorMap((hm_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
            overlay = cv2.addWeighted(orig, 0.5, hm_color, 0.5, 0)

            # Draw predicted center
            if result is not None:
                cx_px = int(result[0] * ow / heatmap.shape[1])
                cy_px = int(result[1] * oh / heatmap.shape[0])
                cv2.drawMarker(overlay, (cx_px, cy_px), (0, 255, 0),
                               cv2.MARKER_CROSS, 20, 2, cv2.LINE_AA)
                cv2.circle(overlay, (cx_px, cy_px), 15, (0, 255, 0), 2, cv2.LINE_AA)

            # Info banner
            is_neg = int(r["is_negative_gt"])
            sharpness = float(r["sharpness"])
            label = "NEGATIVE (no CCTag)" if is_neg else "POSITIVE (has CCTag)"
            tag = "FP" if args.mode == "fp" else "FN"

            cv2.rectangle(overlay, (0, 0), (ow, 60), (0, 0, 0), -1)
            cv2.putText(overlay, f"{stem} [{tag}] peak={peak_val:.3f} sharp={sharpness:.1f}",
                        (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.putText(overlay, f"GT: {label}",
                        (8, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

            tiles.append(overlay)

    # Make grid
    tile_h, tile_w = args.tile_size, int(args.tile_size * 1.6)
    resized = [cv2.resize(t, (tile_w, tile_h), interpolation=cv2.INTER_AREA) for t in tiles]
    cols = min(4, len(resized))
    grid_rows = math.ceil(len(resized) / cols)
    grid = np.full((grid_rows * tile_h, cols * tile_w, 3), 24, dtype=np.uint8)
    for idx, img in enumerate(resized):
        r_idx = idx // cols
        c_idx = idx % cols
        y0 = r_idx * tile_h
        x0 = c_idx * tile_w
        grid[y0:y0 + tile_h, x0:x0 + tile_w] = img

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), grid)
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()
