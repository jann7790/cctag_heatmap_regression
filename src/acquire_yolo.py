#!/usr/bin/env python3
"""
Stage 1 (acquisition): YOLO-nano detector that finds CCTags and returns ROIs.

The detector answers "is there a CCTag, roughly where + how big". Each YOLO box
becomes an ROI (box + padding) handed to the localization model. The box IS the
ROI -- no heatmap blob / size head needed.

Detection is precision-leaning (a false positive wastes an ROI crop), so a high
confidence threshold is the default first gate. YOLO's FPN already covers a range
of sizes in one pass; for very far/small markers, optional tiling keeps near-native
resolution per tile.

Usable as a library (`load_detector`, `detect`) or a CLI.

Example:
  uv run python src/acquire_yolo.py \
    --weights ./outputs/runs_yolo/cctag_det_n/weights/best.pt \
    --input 40m_example.png --conf 0.5 --tile 2x2 --vis \
    --output ./outputs/inference/acquire_yolo
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import cv2
import numpy as np

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO CCTag acquisition (stage 1).")
    parser.add_argument("--weights", type=Path, required=True, help="Trained YOLO .pt weights.")
    parser.add_argument("--input", type=Path, required=True, help="Image file or directory.")
    parser.add_argument("--output", type=Path, default=None, help="Directory for visualization output.")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold (precision-leaning: keep high).")
    parser.add_argument("--imgsz", type=int, default=1024,
                        help="Inference image size. Keep equal to the training imgsz. With --tile 2x2 "
                             "a 4096 frame -> ~2048 tiles -> resized to 1024.")
    parser.add_argument("--device", type=str, default=None, help="CUDA index/'cpu' (default: auto).")
    parser.add_argument("--pad", type=float, default=0.5,
                        help="ROI padding as a fraction of max(box_w, box_h). 0.5 = box +/- 50%%.")
    parser.add_argument("--tile", type=str, default="0",
                        help="Tiling grid 'COLSxROWS' (e.g. 2x2) for small far markers, or '0' to disable.")
    parser.add_argument("--tile_overlap", type=float, default=0.2, help="Tile overlap fraction (default: 0.2).")
    parser.add_argument("--topk", type=int, default=0, help="Keep only the top-K detections by confidence (0 = all).")
    parser.add_argument("--vis", action="store_true", help="Save overlay images with boxes + ROIs.")
    return parser.parse_args()


def load_detector(weights: Path, device: str | None = None) -> Any:
    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover - dependency hint
        raise SystemExit(
            "ultralytics is required for YOLO acquisition.\n"
            "Install it with:  uv sync --extra cu126 --extra detect"
        ) from exc
    model = YOLO(str(weights))
    if device is not None:
        model.to(device)
    return model


def _roi_from_box(xyxy: tuple[float, float, float, float], pad: float,
                  img_w: int, img_h: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = xyxy
    bw, bh = x1 - x0, y1 - y0
    p = pad * max(bw, bh)
    rx0 = int(max(0, x0 - p))
    ry0 = int(max(0, y0 - p))
    rx1 = int(min(img_w, x1 + p))
    ry1 = int(min(img_h, y1 + p))
    return rx0, ry0, rx1, ry1


def _parse_tiles(spec: str, img_w: int, img_h: int, overlap: float) -> list[tuple[int, int, int, int]]:
    """Return a list of (x0, y0, x1, y1) tile windows. '0' disables tiling (single full frame)."""
    if spec in ("0", "", None) or "x" not in spec:
        return [(0, 0, img_w, img_h)]
    cols, rows = (int(v) for v in spec.lower().split("x"))
    cols, rows = max(1, cols), max(1, rows)
    tiles: list[tuple[int, int, int, int]] = []
    step_x = img_w / cols
    step_y = img_h / rows
    ov_x = step_x * overlap
    ov_y = step_y * overlap
    for r in range(rows):
        for c in range(cols):
            x0 = int(max(0, c * step_x - ov_x))
            y0 = int(max(0, r * step_y - ov_y))
            x1 = int(min(img_w, (c + 1) * step_x + ov_x))
            y1 = int(min(img_h, (r + 1) * step_y + ov_y))
            tiles.append((x0, y0, x1, y1))
    return tiles


def _iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def _greedy_nms(dets: list[dict[str, Any]], iou_thr: float = 0.45) -> list[dict[str, Any]]:
    """Dedupe boxes that overlap across tiles. Keeps highest confidence first."""
    kept: list[dict[str, Any]] = []
    for det in sorted(dets, key=lambda d: d["conf"], reverse=True):
        if all(_iou(det["xyxy"], k["xyxy"]) < iou_thr for k in kept):
            kept.append(det)
    return kept


def detect(model: Any, image_bgr: np.ndarray, conf: float = 0.5, imgsz: int = 640,
           pad: float = 0.5, tile: str = "0", tile_overlap: float = 0.2,
           topk: int = 0) -> list[dict[str, Any]]:
    """Run acquisition on one BGR image. Returns detections sorted by confidence.

    Each detection: {conf, xyxy, cx, cy, w, h, roi} in source-image pixel coords.
    """
    img_h, img_w = image_bgr.shape[:2]
    raw: list[dict[str, Any]] = []
    for (tx0, ty0, tx1, ty1) in _parse_tiles(tile, img_w, img_h, tile_overlap):
        crop = image_bgr[ty0:ty1, tx0:tx1]
        results = model.predict(crop, conf=conf, imgsz=imgsz, verbose=False)
        for res in results:
            if res.boxes is None:
                continue
            for box in res.boxes:
                x0, y0, x1, y1 = (float(v) for v in box.xyxy[0].tolist())
                # tile-local -> source coords
                x0 += tx0; x1 += tx0; y0 += ty0; y1 += ty0
                raw.append({
                    "conf": float(box.conf[0]),
                    "xyxy": (x0, y0, x1, y1),
                })

    dets = _greedy_nms(raw)
    for det in dets:
        x0, y0, x1, y1 = det["xyxy"]
        det["cx"] = (x0 + x1) / 2.0
        det["cy"] = (y0 + y1) / 2.0
        det["w"] = x1 - x0
        det["h"] = y1 - y0
        det["roi"] = _roi_from_box(det["xyxy"], pad, img_w, img_h)

    dets.sort(key=lambda d: d["conf"], reverse=True)
    if topk > 0:
        dets = dets[:topk]
    return dets


def draw_overlay(image_bgr: np.ndarray, dets: list[dict[str, Any]]) -> np.ndarray:
    out = image_bgr.copy()
    for det in dets:
        x0, y0, x1, y1 = (int(v) for v in det["xyxy"])
        rx0, ry0, rx1, ry1 = det["roi"]
        cv2.rectangle(out, (rx0, ry0), (rx1, ry1), (0, 200, 255), 2)   # ROI (padded)
        cv2.rectangle(out, (x0, y0), (x1, y1), (0, 255, 0), 2)         # detection box
        cv2.putText(out, f"{det['conf']:.2f}", (x0, max(0, y0 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
    return out


def _iter_images(input_path: Path) -> list[Path]:
    if input_path.is_dir():
        return sorted(p for p in input_path.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    return [input_path]


def main() -> None:
    args = parse_args()
    model = load_detector(args.weights, args.device)
    if args.output is not None:
        args.output.mkdir(parents=True, exist_ok=True)

    for image_path in _iter_images(args.input):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"skip (unreadable): {image_path}")
            continue
        dets = detect(model, image, conf=args.conf, imgsz=args.imgsz, pad=args.pad,
                      tile=args.tile, tile_overlap=args.tile_overlap, topk=args.topk)
        print(f"{image_path.name}: {len(dets)} detection(s)")
        for i, det in enumerate(dets):
            print(f"  [{i}] conf={det['conf']:.3f} box={tuple(round(v,1) for v in det['xyxy'])} roi={det['roi']}")
        if args.vis and args.output is not None:
            cv2.imwrite(str(args.output / f"{image_path.stem}_acquire.png"), draw_overlay(image, dets))


if __name__ == "__main__":
    main()
