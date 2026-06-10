"""Quantify the localization ceiling: collect predicted vs GT centers on the
reproduced val split, fit a per-axis affine calibration on half, evaluate L2 on
the other half. Shows how much of the ~11px median error is a correctable
systematic bias vs irreducible scatter.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_cctag_heatmap_ddp as T  # noqa: E402


def l2(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    return np.linalg.norm(pred - gt, axis=1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=Path, required=True)
    ap.add_argument("--checkpoint", type=str, default="best.pt")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--num_workers", type=int, default=8)
    args = ap.parse_args()

    cfg = json.loads((args.run_dir / "run_config.json").read_text())
    T.DECODE_ALIGN_CORNERS = bool(cfg.get("align_corners", False))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input_size = (cfg["input_width"], cfg["input_height"])
    train_dirs = [Path(d) for d in cfg["train_dataset_dirs"]]

    merged, base_size, hm = T.build_dataset_collection(
        train_dirs, input_size, split_name="val", augment=None
    )
    _, val_idx = T.split_indices(base_size, cfg["train_ratio"], cfg["seed"])
    val_set = Subset(merged, val_idx)
    sx = input_size[0] / hm[1]
    sy = input_size[1] / hm[0]

    model = T.build_model(
        heatmap_size=hm, device=device, rank=0, world_size=1,
        backbone=cfg["backbone"],
        use_offset_head=cfg.get("use_offset_head", False),
        use_size_head=cfg.get("use_size_head", False),
    )
    ckpt = torch.load(args.run_dir / args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt.get("model", ckpt)))
    model.eval()

    loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    preds, gts = [], []
    with torch.no_grad():
        for batch in loader:
            imgs = batch["image"].to(device, non_blocking=True)
            cen = batch["center"].to(device, non_blocking=True)
            tgt = batch["heatmap"].to(device, non_blocking=True)
            p, off, _ = T._split_pred(model(imgs))
            p = p.float()
            off = off.float() if off is not None else None
            pos = tgt.flatten(1).max(dim=1).values > 0.1
            if not pos.any():
                continue
            dec = T.decode_heatmap_centers(p[pos], offsets=(off[pos] if off is not None else None))
            dec[:, 0] *= sx
            dec[:, 1] *= sy
            preds.append(dec.cpu().numpy())
            gts.append(cen[pos].cpu().numpy())

    pred = np.concatenate(preds)
    gt = np.concatenate(gts)
    n = len(pred)

    # honest split: fit calibration on half, evaluate on the other half
    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    fit_i, ev_i = perm[: n // 2], perm[n // 2:]

    base = l2(pred, gt)
    base_ev = l2(pred[ev_i], gt[ev_i])

    def report(name, corr_ev):
        e = l2(corr_ev, gt[ev_i])
        print(f"{name:<34s} mean={e.mean():6.2f}  median={np.median(e):6.2f}  "
              f"p90={np.percentile(e,90):6.2f}")

    print(f"N positives = {n}  (fit={len(fit_i)}, eval={len(ev_i)})\n")
    print(f"{'BASELINE (eval half)':<34s} mean={base_ev.mean():6.2f}  "
          f"median={np.median(base_ev):6.2f}  p90={np.percentile(base_ev,90):6.2f}")

    # 1) constant offset (subtract median bias)
    b = np.median(pred[fit_i] - gt[fit_i], axis=0)
    report("constant offset", pred[ev_i] - b)

    # 2) per-axis affine: coord_corr = a*coord + b  (independent x, y)
    def fit_axis(k):
        A = np.vstack([pred[fit_i, k], np.ones(len(fit_i))]).T
        sol, *_ = np.linalg.lstsq(A, gt[fit_i, k], rcond=None)
        return sol  # a, b
    ax, axb = fit_axis(0)
    ay, ayb = fit_axis(1)
    corr = np.stack([ax * pred[ev_i, 0] + axb, ay * pred[ev_i, 1] + ayb], axis=1)
    report("per-axis affine (a*p+b)", corr)
    print(f"   fitted: x: {ax:.4f}*px {axb:+.2f}   y: {ay:.4f}*py {ayb:+.2f}")

    # 3) full 2D affine (allows cross terms / shear)
    A = np.hstack([pred[fit_i], np.ones((len(fit_i), 1))])  # (m,3)
    M, *_ = np.linalg.lstsq(A, gt[fit_i], rcond=None)  # (3,2)
    Ae = np.hstack([pred[ev_i], np.ones((len(ev_i), 1))])
    report("full 2D affine", Ae @ M)

    print("\nNote: linear calibration cannot fix wrong-peak outliers, so the")
    print("median (robust) is the meaningful comparison for the systematic bias.")


if __name__ == "__main__":
    main()
