"""Split an image into 2x2 tiles, run YOLO detection on each, stitch boxes back.

Usage:
  uv run python src/tile_detect.py \
    --model outputs/runs_yolo/cctag_det_n/weights/best.pt \
    --source 40m_example.png \
    --output outputs/runs_yolo/predict_40m_tiled \
    --imgsz 1024 --conf 0.25
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


def starts(total: int, tile: int, overlap: float):
    """Sliding-window start offsets covering `total` with given tile size/overlap."""
    if tile >= total:
        return [0]
    step = max(1, int(tile * (1 - overlap)))
    pos = list(range(0, total - tile + 1, step))
    if pos[-1] != total - tile:
        pos.append(total - tile)
    return pos


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--output", default="outputs/runs_yolo/predict_tiled")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--tile", type=int, default=1024,
                    help="tile size in px (square window over the full image)")
    ap.add_argument("--overlap", type=float, default=0.3,
                    help="fractional overlap between adjacent tiles (0-1)")
    ap.add_argument("--iou", type=float, default=0.5,
                    help="IoU threshold for merging detections across tiles (NMS)")
    ap.add_argument("--save_tiles", action="store_true",
                    help="Also save each annotated tile")
    args = ap.parse_args()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.model)
    img = cv2.imread(args.source)
    if img is None:
        raise FileNotFoundError(args.source)
    H, W = img.shape[:2]

    full = img.copy()
    boxes_g, confs, clss = [], [], []
    xs = starts(W, args.tile, args.overlap)
    ys = starts(H, args.tile, args.overlap)
    for yi, y0 in enumerate(ys):
        for xi, x0 in enumerate(xs):
            tw = min(args.tile, W)
            th = min(args.tile, H)
            tile = img[y0:y0 + th, x0:x0 + tw]

            res = model.predict(tile, imgsz=args.imgsz, conf=args.conf,
                                verbose=False)[0]
            if args.save_tiles:
                cv2.imwrite(str(out_dir / f"tile_y{yi}x{xi}.jpg"), res.plot())

            for box in res.boxes:
                bx0, by0, bx1, by1 = box.xyxy[0].tolist()
                boxes_g.append([bx0 + x0, by0 + y0, bx1 + x0, by1 + y0])
                confs.append(float(box.conf[0]))
                clss.append(int(box.cls[0]))

    # Global NMS to merge duplicates from overlapping tiles
    keep = []
    if boxes_g:
        idxs = cv2.dnn.NMSBoxes(
            [[int(b[0]), int(b[1]), int(b[2] - b[0]), int(b[3] - b[1])]
             for b in boxes_g],
            confs, args.conf, args.iou)
        keep = np.array(idxs).flatten().tolist()

    for i in keep:
        gx0, gy0, gx1, gy1 = (int(v) for v in boxes_g[i])
        cv2.rectangle(full, (gx0, gy0), (gx1, gy1), (0, 0, 255), 2)
        label = f"{model.names[clss[i]]} {confs[i]:.2f}"
        cv2.putText(full, label, (gx0, max(0, gy0 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    out_path = out_dir / (Path(args.source).stem + "_tiled.jpg")
    cv2.imwrite(str(out_path), full)
    print(f"Tiles: {len(xs)}x{len(ys)}  raw boxes: {len(boxes_g)}  "
          f"after NMS: {len(keep)}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
