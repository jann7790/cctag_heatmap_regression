"""On a single dataset, bucket the model's signed center error (pred-CSV) by
whether the marker is y-clipped, to test if the y-bias comes from clipping."""

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_cctag_heatmap_ddp as T  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=Path, required=True)
    ap.add_argument("--checkpoint", type=str, default="best.pt")
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=1500)
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()

    cfg = json.loads((args.run_dir / "run_config.json").read_text())
    T.DECODE_ALIGN_CORNERS = bool(cfg.get("align_corners", False))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input_size = (cfg["input_width"], cfg["input_height"])

    # y-clip flag per filename
    yclip = {}
    cy_of = {}
    with (args.dataset / "labels.csv").open() as fh:
        for r in csv.DictReader(fh):
            fn = r.get("filename", "").strip()
            if not fn or (r.get("is_negative") or "0") == "1":
                continue
            ymin, ymax = float(r["bbox_ymin"]), float(r["bbox_ymax"])
            yclip[fn] = (ymin <= 0.5 and ymax >= input_size[1] - 1.5)
            cy_of[fn] = float(r["center_y"])

    ds = T.CCTagHeatmapDataset(args.dataset, input_size=input_size, augment=None)
    heatmap_size = (ds.heatmap_height, ds.heatmap_width)
    sx = input_size[0] / heatmap_size[1]
    sy = input_size[1] / heatmap_size[0]

    model = T.build_model(
        heatmap_size=heatmap_size, device=device, rank=0, world_size=1,
        backbone=cfg["backbone"],
        use_offset_head=cfg.get("use_offset_head", False),
        use_size_head=cfg.get("use_size_head", False),
        offset_head_hidden=cfg.get("offset_head_hidden", 0),
    )
    ckpt = torch.load(args.run_dir / args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt.get("model", ckpt)))
    model.eval()

    idxs = [i for i in range(len(ds)) if ds.samples[i]["filename"] in yclip][: args.limit]
    rows = []  # (dy, dx, is_yclip, center_y)
    with torch.no_grad():
        for s in range(0, len(idxs), args.batch_size):
            chunk = idxs[s : s + args.batch_size]
            imgs = torch.stack([ds[i]["image"] for i in chunk]).to(device)
            cen = torch.stack([ds[i]["center"] for i in chunk]).to(device)
            pred, off, _ = T._split_pred(model(imgs))
            pred = pred.float()
            off = off.float() if off is not None else None
            dec = T.decode_heatmap_centers(pred, offsets=off)
            dec[:, 0] *= sx
            dec[:, 1] *= sy
            d = (dec - cen).cpu().numpy()
            for k, i in enumerate(chunk):
                fn = ds.samples[i]["filename"]
                rows.append((d[k, 1], d[k, 0], yclip[fn], cy_of[fn]))

    a = np.array(rows, dtype=float)
    dy, dx, clip, cy = a[:, 0], a[:, 1], a[:, 2].astype(bool), a[:, 3]
    print(f"dataset={args.dataset.name}  n={len(a)}")
    print(f"ALL        : dy_mean={dy.mean():+.2f} dy_med={np.median(dy):+.2f}  "
          f"dx_mean={dx.mean():+.2f}")
    for tag, m in [("y-CLIPPED  ", clip), ("not clipped", ~clip)]:
        if m.any():
            print(f"{tag}: n={int(m.sum()):4d}  dy_mean={dy[m].mean():+.2f} "
                  f"dy_med={np.median(dy[m]):+.2f}  dx_mean={dx[m].mean():+.2f}")
    # does dy depend on vertical position of the marker in the frame?
    print("\n-- dy_mean by center_y band (frame is 0..%d) --" % input_size[1])
    for lo, hi in [(0, 160), (160, 320), (320, 480), (480, 640)]:
        m = (cy >= lo) & (cy < hi)
        if m.any():
            print(f"  center_y {lo:3d}-{hi:3d}: n={int(m.sum()):4d}  dy_mean={dy[m].mean():+.2f}")


if __name__ == "__main__":
    main()
