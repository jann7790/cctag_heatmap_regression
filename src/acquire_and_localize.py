#!/usr/bin/env python3
"""
Two-stage CCTag pipeline: YOLO acquisition -> heatmap sub-pixel localization.

  Stage 1 (acquire_yolo): YOLO-nano finds CCTags -> ROI per detection (box + pad).
  Stage 2 (this file):     crop each ROI, run the existing heatmap+offset model
                           (e.g. experiment_sizehead) for a sub-pixel centre + ellipse.

The localization model doubles as a PRECISION VERIFIER: if it cannot find a sharp
peak inside the ROI (peak < --loc_threshold), the detection is rejected. Combined
with the high YOLO confidence gate this is the precision-leaning design from the plan.

The localization model and its decode are reused unchanged from infer_cctag_heatmap.

Example:
  uv run python src/acquire_and_localize.py \
    --yolo_weights ./outputs/runs_yolo/cctag_det_n/weights/best.pt \
    --loc_checkpoint ./outputs/runs/experiment_sizehead/best.pt \
    --input 40m_example.png --conf 0.5 --loc_threshold 0.5 --vis \
    --output ./outputs/inference/acquire_and_localize
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from acquire_yolo import detect, load_detector
from infer_cctag_heatmap import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    decode_center_offset,
    decode_center_weighted,
    decode_size_at_peak,
    load_model,
    resolve_device,
)

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO acquisition + heatmap localization (two-stage).")
    parser.add_argument("--yolo_weights", type=Path, required=True, help="Stage-1 YOLO .pt weights.")
    parser.add_argument("--loc_checkpoint", type=Path, required=True, help="Stage-2 heatmap localization checkpoint.")
    parser.add_argument("--input", type=Path, required=True, help="Image file or directory.")
    parser.add_argument("--output", type=Path, default=None, help="Directory for visualization output.")
    parser.add_argument("--device", type=str, default=None, help="cpu / cuda / cuda:0 (default: auto).")
    # Stage 1 knobs
    parser.add_argument("--conf", type=float, default=0.5, help="YOLO confidence threshold (gate 1).")
    parser.add_argument("--imgsz", type=int, default=1024,
                        help="YOLO inference image size (match training; 2x2 tiles of 4096 -> ~2048 -> 1024).")
    parser.add_argument("--pad", type=float, default=0.5, help="ROI padding fraction of max(box_w, box_h).")
    parser.add_argument("--tile", type=str, default="0", help="YOLO tiling grid 'COLSxROWS' or '0'.")
    parser.add_argument("--tile_overlap", type=float, default=0.2, help="Tile overlap fraction.")
    parser.add_argument("--topk", type=int, default=0, help="Keep top-K YOLO detections (0 = all).")
    # Stage 2 knobs
    parser.add_argument("--loc_threshold", type=float, default=0.5,
                        help="Localization peak threshold; below this the detection is rejected (verifier gate).")
    parser.add_argument("--vis", action="store_true", help="Save overlay images.")
    return parser.parse_args()


def localize_roi(model: Any, config: dict, roi_bgr: np.ndarray, device: torch.device,
                 threshold: float) -> dict[str, Any] | None:
    """Run the heatmap model on an ROI crop. Returns centre+ellipse in ROI pixel
    coords, or None if the peak is below threshold (verifier rejection)."""
    in_w = int(config.get("input_width", 640))
    in_h = int(config.get("input_height", 400))
    region_h, region_w = roi_bgr.shape[:2]
    if region_w == 0 or region_h == 0:
        return None

    rgb = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2RGB)
    inp = cv2.resize(rgb, (in_w, in_h), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(inp.transpose(2, 0, 1)).float() / 255.0
    tensor = ((tensor - IMAGENET_MEAN) / IMAGENET_STD).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(tensor)

    if isinstance(out, tuple):
        hm = out[0][0, 0].cpu().numpy()
        off = out[1][0].cpu().numpy() if out[1] is not None else None
        sz = out[2][0].cpu().numpy() if len(out) > 2 and out[2] is not None else None
    else:
        hm, off, sz = out[0, 0].cpu().numpy(), None, None

    peak = float(hm.max())
    if peak < threshold:
        return None

    hm_h, hm_w = hm.shape
    center = decode_center_offset(hm, off, threshold=threshold) if off is not None \
        else decode_center_weighted(hm, threshold=threshold)
    if center is None:
        return None

    cx_roi = center[0] * (region_w / hm_w)
    cy_roi = center[1] * (region_h / hm_h)

    a = b = None
    if sz is not None:
        ab = decode_size_at_peak(hm, sz, threshold=threshold)
        if ab:
            a = ab[0] * (region_w / in_w)
            b = ab[1] * (region_h / in_h)

    return {"peak": peak, "cx_roi": cx_roi, "cy_roi": cy_roi, "a": a, "b": b}


def run_image(detector: Any, loc_model: Any, loc_config: dict, image_bgr: np.ndarray,
              device: torch.device, args: argparse.Namespace) -> list[dict[str, Any]]:
    """Full pipeline on one image. Returns confirmed targets in source coords."""
    dets = detect(detector, image_bgr, conf=args.conf, imgsz=args.imgsz, pad=args.pad,
                  tile=args.tile, tile_overlap=args.tile_overlap, topk=args.topk)

    confirmed: list[dict[str, Any]] = []
    for det in dets:
        rx0, ry0, rx1, ry1 = det["roi"]
        loc = localize_roi(loc_model, loc_config, image_bgr[ry0:ry1, rx0:rx1], device, args.loc_threshold)
        if loc is None:
            continue  # verifier rejected -> precision gate 2
        confirmed.append({
            "conf": det["conf"],
            "roi": det["roi"],
            "peak": loc["peak"],
            "cx": loc["cx_roi"] + rx0,   # source coords
            "cy": loc["cy_roi"] + ry0,
            "a": loc["a"],
            "b": loc["b"],
        })
    return confirmed


def draw_overlay(image_bgr: np.ndarray, targets: list[dict[str, Any]]) -> np.ndarray:
    out = image_bgr.copy()
    for t in targets:
        rx0, ry0, rx1, ry1 = t["roi"]
        cv2.rectangle(out, (rx0, ry0), (rx1, ry1), (0, 200, 255), 2)
        cx, cy = int(round(t["cx"])), int(round(t["cy"]))
        if t["a"] and t["b"]:
            cv2.ellipse(out, (cx, cy), (max(1, int(t["a"])), max(1, int(t["b"]))),
                        0, 0, 360, (0, 255, 255), 2)
        cv2.circle(out, (cx, cy), 4, (0, 255, 0), -1)
        cv2.circle(out, (cx, cy), 6, (0, 0, 0), 1)
        cv2.putText(out, f"d={t['conf']:.2f} p={t['peak']:.2f}", (rx0, max(0, ry0 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    return out


def _iter_images(input_path: Path) -> list[Path]:
    if input_path.is_dir():
        return sorted(p for p in input_path.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    return [input_path]


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    detector = load_detector(args.yolo_weights, args.device)
    loc_model, loc_config = load_model(args.loc_checkpoint, device)
    if args.output is not None:
        args.output.mkdir(parents=True, exist_ok=True)

    for image_path in _iter_images(args.input):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"skip (unreadable): {image_path}")
            continue
        targets = run_image(detector, loc_model, loc_config, image, device, args)
        print(f"{image_path.name}: {len(targets)} confirmed target(s)")
        for i, t in enumerate(targets):
            ell = f"a={t['a']:.1f} b={t['b']:.1f}" if t["a"] and t["b"] else "ellipse=-"
            print(f"  [{i}] det_conf={t['conf']:.3f} loc_peak={t['peak']:.3f} "
                  f"centre=({t['cx']:.1f},{t['cy']:.1f}) {ell}")
        if args.vis and args.output is not None:
            cv2.imwrite(str(args.output / f"{image_path.stem}_pipeline.png"), draw_overlay(image, targets))


if __name__ == "__main__":
    main()
