"""Overlay CSV center / GT-heatmap peak / predicted peak on a few samples to
diagnose the systematic center bias. Saves a zoomed crop grid per sample.

Usage:
  CUDA_VISIBLE_DEVICES=1 uv run python src/overlay_center_bias.py \
    --run_dir outputs/runs/newest --checkpoint best.pt \
    --dataset outputs/training_sets/generated_training_sets_1024/mixed_train_dataset \
    --num 6 --output outputs/tmp/center_bias_overlay.png
"""

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_cctag_heatmap_ddp as T  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", type=Path, required=True)
    p.add_argument("--checkpoint", type=str, default="best.pt")
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--num", type=int, default=6)
    p.add_argument("--crop", type=int, default=80, help="half-size of zoom crop (input px)")
    p.add_argument("--output", type=Path, default=Path("outputs/tmp/center_bias_overlay.png"))
    args = p.parse_args()

    cfg = json.loads((args.run_dir / "run_config.json").read_text())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input_size = (cfg["input_width"], cfg["input_height"])

    ds = T.CCTagHeatmapDataset(args.dataset, input_size=input_size, augment=None)
    # pull ellipse center too (not loaded by the dataset class)
    import csv as _csv
    ell = {}
    with (args.dataset / "labels.csv").open() as fh:
        for row in _csv.DictReader(fh):
            fn = row.get("filename", "").strip()
            if fn:
                ell[fn] = (float(row.get("ellipse_cx") or 0), float(row.get("ellipse_cy") or 0))
    heatmap_size = (ds.heatmap_height, ds.heatmap_width)
    sx = input_size[0] / heatmap_size[1]
    sy = input_size[1] / heatmap_size[0]

    model = T.build_model(
        heatmap_size=heatmap_size, device=device, rank=0, world_size=1,
        backbone=cfg["backbone"],
        use_offset_head=cfg.get("use_offset_head", False),
        use_size_head=cfg.get("use_size_head", False),
    )
    ckpt = torch.load(args.run_dir / args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt.get("model", ckpt)))
    model.eval()

    # pick positive samples
    picks = []
    for i in range(len(ds)):
        s = ds.samples[i]
        if float(s["center_x"]) > 0 and float(s["center_y"]) > 0:
            picks.append(i)
        if len(picks) >= args.num:
            break

    tiles = []
    C = args.crop
    with torch.no_grad():
        for i in picks:
            item = ds[i]
            img = item["image"].unsqueeze(0).to(device)
            gt_cx, gt_cy = item["center"].tolist()  # CSV center in input px
            raw = model(img)
            pred, off, _ = T._split_pred(raw)
            pred = pred.float()
            off = off.float() if off is not None else None
            dec = T.decode_heatmap_centers(pred, offsets=off)[0]
            px, py = float(dec[0]) * sx, float(dec[1]) * sy
            gt_hm = T.decode_heatmap_centers(item["heatmap"].unsqueeze(0).to(device))[0]
            gpx, gpy = float(gt_hm[0]) * sx, float(gt_hm[1]) * sy

            # rebuild displayable input image (denormalize)
            im = item["image"].clone()
            im = im * T.IMAGENET_STD + T.IMAGENET_MEAN
            im = (im.clamp(0, 1).numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            im = cv2.cvtColor(im, cv2.COLOR_RGB2BGR)

            ecx, ecy = ell.get(ds.samples[i]["filename"], (gt_cx, gt_cy))
            src_h, src_w = cv2.imread(str(ds.samples[i]["image_path"])).shape[:2]
            ecx *= input_size[0] / src_w
            ecy *= input_size[1] / src_h

            cx, cy = int(round(gt_cx)), int(round(gt_cy))
            x0, y0 = max(0, cx - C), max(0, cy - C)
            x1, y1 = min(im.shape[1], cx + C), min(im.shape[0], cy + C)
            scale = 4
            crop = cv2.resize(im[y0:y1, x0:x1].copy(), None, fx=scale, fy=scale,
                              interpolation=cv2.INTER_NEAREST)

            def draw(cxp, cyp, color, label):
                u = int(round((cxp - x0) * scale))
                v = int(round((cyp - y0) * scale))
                cv2.drawMarker(crop, (u, v), color, cv2.MARKER_CROSS, 18, 1)
                cv2.putText(crop, label, (u + 6, v - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

            draw(ecx, ecy, (0, 255, 255), "ell")         # yellow = ellipse center
            draw(gt_cx, gt_cy, (0, 255, 0), "CSV")       # green = label / heatmap
            draw(px, py, (0, 0, 255), "pred")            # red = model
            cv2.putText(crop, f"pred-CSV dy={py-gt_cy:+.1f} dx={px-gt_cx:+.1f}",
                        (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            tiles.append(crop)

    # grid
    h = max(t.shape[0] for t in tiles)
    w = max(t.shape[1] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, w - t.shape[1],
                                cv2.BORDER_CONSTANT, value=(40, 40, 40)) for t in tiles]
    cols = min(3, len(tiles))
    rows = (len(tiles) + cols - 1) // cols
    while len(tiles) < rows * cols:
        tiles.append(np.full((h, w, 3), 40, np.uint8))
    grid = np.vstack([np.hstack(tiles[r * cols:(r + 1) * cols]) for r in range(rows)])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), grid)
    print(f"saved {args.output}  (green=CSV center, blue=GT heatmap peak, red=prediction)")


if __name__ == "__main__":
    main()
