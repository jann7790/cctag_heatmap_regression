"""Dump the per-sample center-L2 distribution of a trained checkpoint on the
reproduced validation split. fp32 decode, offset head used if present.

Usage:
  uv run python src/eval_l2_distribution.py \
    --run_dir outputs/runs/newest --checkpoint best.pt
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

import train_cctag_heatmap_ddp as T  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", type=Path, required=True)
    p.add_argument("--checkpoint", type=str, default="best.pt")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--fp16", action="store_true",
                   help="Run the forward pass under fp16 autocast (to measure the "
                        "peak-plateau localization bias vs fp32). Decode math stays fp32.")
    args = p.parse_args()

    cfg = json.loads((args.run_dir / "run_config.json").read_text())
    T.DECODE_ALIGN_CORNERS = bool(cfg.get("align_corners", False))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input_size = (cfg["input_width"], cfg["input_height"])
    train_dirs = [Path(d) for d in cfg["train_dataset_dirs"]]

    # Reproduce the carved-out val split exactly: merged (no-augment) dataset,
    # seeded index split with the same train_ratio + seed.
    merged, base_size, heatmap_size = T.build_dataset_collection(
        train_dirs, input_size, split_name="val", augment=None
    )
    _, val_idx = T.split_indices(base_size, cfg["train_ratio"], cfg["seed"])
    val_set = Subset(merged, val_idx)
    print(f"base_size={base_size} val_size={len(val_set)} heatmap_size={heatmap_size}")

    # Map each merged-dataset index to its source dataset name (for per-source bias).
    import bisect

    if hasattr(merged, "cumulative_sizes"):
        cum = merged.cumulative_sizes
        src_names = [Path(d.dataset_dir).name for d in merged.datasets]

        def src_of(g: int) -> str:
            return src_names[bisect.bisect_right(cum, g)]
    else:
        only = Path(merged.dataset_dir).name

        def src_of(g: int) -> str:
            return only

    val_src = [src_of(g) for g in val_idx]  # in val_set (loader) order

    model = T.build_model(
        heatmap_size=heatmap_size,
        device=device,
        rank=0,
        world_size=1,
        backbone=cfg["backbone"],
        use_offset_head=cfg.get("use_offset_head", False),
        use_size_head=cfg.get("use_size_head", False),
    )
    ckpt = torch.load(
        args.run_dir / args.checkpoint, map_location=device, weights_only=False
    )
    state = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
    model.load_state_dict(state)
    model.eval()

    loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    errs: list[float] = []
    occ: list[float] = []
    dxs: list[float] = []
    dys: list[float] = []
    gt_errs: list[float] = []
    ndx: list[float] = []
    ndy: list[float] = []
    srcs: list[str] = []
    idx_ptr = 0
    with torch.no_grad():
        for batch in loader:
            bsz = batch["image"].size(0)
            batch_src = val_src[idx_ptr : idx_ptr + bsz]
            idx_ptr += bsz
            images = batch["image"].to(device, non_blocking=True)
            centers = batch["center"].to(device, non_blocking=True)
            targets = batch["heatmap"].to(device, non_blocking=True)
            if args.fp16 and device.type == "cuda":
                with torch.autocast("cuda", dtype=torch.float16):
                    raw = model(images)
            else:
                raw = model(images)
            pred, off, _ = T._split_pred(raw)
            pred = pred.float()
            off = off.float() if off is not None else None
            pos = targets.flatten(1).max(dim=1).values > 0.1
            if not pos.any():
                continue
            dec = T.decode_heatmap_centers(
                pred[pos], offsets=(off[pos] if off is not None else None)
            )
            sx = input_size[0] / heatmap_size[1]
            sy = input_size[1] / heatmap_size[0]
            dec_px = dec.clone()
            dec_px[:, 0] *= sx
            dec_px[:, 1] *= sy
            d = torch.linalg.norm(dec_px - centers[pos], dim=1)
            errs.extend(d.cpu().tolist())
            occ.extend(batch["occlusion_ratio"][pos.cpu()].cpu().tolist())
            diff = (dec_px - centers[pos]).cpu().numpy()
            dxs.extend(diff[:, 0].tolist())
            dys.extend(diff[:, 1].tolist())
            # isolate: decode predicted heatmap WITHOUT the offset head
            no_off = T.decode_heatmap_centers(pred[pos], offsets=None)
            no_off[:, 0] *= sx
            no_off[:, 1] *= sy
            nd = (no_off - centers[pos]).cpu().numpy()
            ndx.extend(nd[:, 0].tolist())
            ndy.extend(nd[:, 1].tolist())
            pos_np = pos.cpu().numpy()
            srcs.extend([s for s, keep in zip(batch_src, pos_np) if keep])
            # control: decode the GT heatmap (no offset) vs the CSV center, same scale.
            gt_dec = T.decode_heatmap_centers(targets[pos], offsets=None)
            gt_dec[:, 0] *= sx
            gt_dec[:, 1] *= sy
            gd = torch.linalg.norm(gt_dec - centers[pos], dim=1)
            gt_errs.extend(gd.cpu().tolist())

    e = np.array(errs)
    o = np.array(occ)
    dx = np.array(dxs)
    dy = np.array(dys)
    ge = np.array(gt_errs)
    print("\n-- CONTROL: GT heatmap argmax decoded vs CSV center (input px) --")
    print(
        f"  mean={ge.mean():.3f}  median={np.median(ge):.3f}  "
        f"p90={np.percentile(ge,90):.3f}  max={ge.max():.3f}"
    )
    print("  (this is the best the model could possibly score against this metric)")
    print("\n-- signed error (pred - gt), input px --")
    print(f"dx: mean={dx.mean():+.3f}  std={dx.std():.3f}  median={np.median(dx):+.3f}")
    print(f"dy: mean={dy.mean():+.3f}  std={dy.std():.3f}  median={np.median(dy):+.3f}")
    nndx = np.array(ndx)
    nndy = np.array(ndy)
    print("\n-- predicted-heatmap argmax ONLY (no offset head), signed input px --")
    print(f"dx: mean={nndx.mean():+.3f}  median={np.median(nndx):+.3f}")
    print(f"dy: mean={nndy.mean():+.3f}  median={np.median(nndy):+.3f}")
    inl = e < 20
    print(
        f"inliers(<20px) dx: mean={dx[inl].mean():+.3f} std={dx[inl].std():.3f} | "
        f"dy: mean={dy[inl].mean():+.3f} std={dy[inl].std():.3f}"
    )
    print(f"\nN positives = {e.size}")
    print(f"mean   = {e.mean():.3f} px   (this is what metrics.csv reports)")
    for q in (50, 75, 90, 95, 99):
        print(f"p{q:<3d}  = {np.percentile(e, q):.3f} px")
    print(f"max    = {e.max():.3f} px")

    print("\n-- error histogram (input px) --")
    edges = [0, 2, 4, 6, 8, 12, 20, 40, 80, 160, 1e9]
    for lo, hi in zip(edges[:-1], edges[1:]):
        n = int(((e >= lo) & (e < hi)).sum())
        bar = "#" * int(60 * n / e.size)
        label = f"{lo:>4g}-{hi:<4g}" if hi < 1e9 else f">{lo:g}"
        print(f"{label:>10s} | {n:5d} ({100*n/e.size:4.1f}%) {bar}")

    # how much of the mean comes from the tail?
    tail = e[e >= 20]
    print(
        f"\ntail (>=20px): {tail.size} samples ({100*tail.size/e.size:.1f}%), "
        f"contribute {tail.sum()/e.sum()*100:.1f}% of the summed error"
    )
    mean_no_tail = e[e < 20].mean() if (e < 20).any() else 0.0
    print(f"mean excluding tail (<20px) = {mean_no_tail:.3f} px")

    sarr = np.array(srcs)
    print("\n-- per-source signed bias (pred - gt), input px --")
    print(
        f"{'source':<42s} {'n':>5s} {'dx_med':>8s} {'dy_med':>8s} "
        f"{'dx_mean':>8s} {'dy_mean':>8s} {'L2_med':>8s}"
    )
    for name in dict.fromkeys(srcs):  # preserve order, unique
        m = sarr == name
        print(
            f"{name:<42s} {int(m.sum()):5d} "
            f"{np.median(dx[m]):+8.2f} {np.median(dy[m]):+8.2f} "
            f"{dx[m].mean():+8.2f} {dy[m].mean():+8.2f} {np.median(e[m]):8.2f}"
        )

    if o.size and o.max() > 0:
        print("\n-- mean L2 by occlusion bin --")
        for lo, hi in [(0, 1e-6), (1e-6, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 1.01)]:
            m = (o >= lo) & (o < hi)
            if m.any():
                tag = "clean" if hi <= 1e-6 else f"{lo:.1f}-{hi:.1f}"
                print(
                    f"occ {tag:>9s}: n={int(m.sum()):5d}  "
                    f"mean={e[m].mean():6.2f}  median={np.median(e[m]):6.2f}"
                )


if __name__ == "__main__":
    main()
