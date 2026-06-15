"""Dump the per-sample predicted heatmap peak on the reproduced validation split,
split by GT positive/negative, and sweep the detection threshold to see how the
negative false-positive rate trades off against the positive detection rate.

Reuses the exact val-split reconstruction and source-attribution logic from
eval_l2_distribution.py, and T.compute_detection_counts semantics (peak >= thr).

Usage:
  uv run python src/eval_negative_peaks.py \
    --run_dir outputs/runs/fable_occ --checkpoint best.pt
"""

import argparse
import bisect
import csv
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
    p.add_argument(
        "--out_csv",
        type=Path,
        default=None,
        help="CSV path for per-sample peaks. Default: outputs/eval/<run>_negative_peaks.csv",
    )
    p.add_argument(
        "--gt_pos_thr",
        type=float,
        default=0.1,
        help="GT heatmap peak above this counts the sample as a positive "
        "(matches compute_detection_counts gt_has_object).",
    )
    args = p.parse_args()

    cfg = json.loads((args.run_dir / "run_config.json").read_text())
    T.DECODE_ALIGN_CORNERS = bool(cfg.get("align_corners", False))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input_size = (cfg["input_width"], cfg["input_height"])
    train_dirs = [Path(d) for d in cfg["train_dataset_dirs"]]

    # Reproduce the carved-out val split exactly (same as eval_l2_distribution.py).
    merged, base_size, heatmap_size = T.build_dataset_collection(
        train_dirs, input_size, split_name="val", augment=None
    )
    _, val_idx = T.split_indices(base_size, cfg["train_ratio"], cfg["seed"])
    val_set = Subset(merged, val_idx)
    print(f"base_size={base_size} val_size={len(val_set)} heatmap_size={heatmap_size}")

    if hasattr(merged, "cumulative_sizes"):
        cum = merged.cumulative_sizes
        src_names = [Path(d.dataset_dir).name for d in merged.datasets]

        def src_of(g: int) -> str:
            return src_names[bisect.bisect_right(cum, g)]
    else:
        only = Path(merged.dataset_dir).name

        def src_of(g: int) -> str:
            return only

    val_src = [src_of(g) for g in val_idx]  # in loader order

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

    rows: list[dict] = []
    idx_ptr = 0
    with torch.no_grad():
        for batch in loader:
            bsz = batch["image"].size(0)
            batch_src = val_src[idx_ptr : idx_ptr + bsz]
            idx_ptr += bsz
            images = batch["image"].to(device, non_blocking=True)
            targets = batch["heatmap"].to(device, non_blocking=True)
            raw = model(images)
            pred, _, _ = T._split_pred(raw)
            pred = pred.float()
            pred_peak = pred.flatten(1).max(dim=1).values.cpu().numpy()
            gt_peak = targets.flatten(1).max(dim=1).values.cpu().numpy()
            for i in range(bsz):
                rows.append(
                    {
                        "filename": batch["filename"][i],
                        "source": batch_src[i],
                        "gt_peak": float(gt_peak[i]),
                        "pred_peak": float(pred_peak[i]),
                        "is_positive": int(gt_peak[i] > args.gt_pos_thr),
                    }
                )

    out_csv = args.out_csv or (
        Path("outputs/eval") / f"{args.run_dir.name}_negative_peaks.csv"
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["filename", "source", "gt_peak", "pred_peak", "is_positive"]
        )
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {out_csv}")

    pos = np.array([r["pred_peak"] for r in rows if r["is_positive"]])
    neg = np.array([r["pred_peak"] for r in rows if not r["is_positive"]])
    n_pos, n_neg = pos.size, neg.size
    print(f"\npositives={n_pos}  negatives={n_neg}")
    if n_neg:
        print(
            f"negative pred_peak: mean={neg.mean():.4f} median={np.median(neg):.4f} "
            f"p90={np.percentile(neg,90):.4f} p99={np.percentile(neg,99):.4f} max={neg.max():.4f}"
        )
    if n_pos:
        print(
            f"positive pred_peak: mean={pos.mean():.4f} median={np.median(pos):.4f} "
            f"p10={np.percentile(pos,10):.4f} p1={np.percentile(pos,1):.4f} min={pos.min():.4f}"
        )

    # Threshold sweep: FPR on negatives vs detection rate on positives.
    print("\n-- threshold sweep --")
    print(f"{'thr':>5s} {'FP':>5s} {'FPR':>9s} {'det_pos':>8s} {'det_rate':>9s}")
    for thr in [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99]:
        fp = int((neg >= thr).sum()) if n_neg else 0
        fpr = fp / n_neg if n_neg else 0.0
        det = int((pos >= thr).sum()) if n_pos else 0
        det_rate = det / n_pos if n_pos else 0.0
        print(f"{thr:5.2f} {fp:5d} {fpr:9.6f} {det:8d} {det_rate:9.6f}")

    # The lowest threshold that drives FP to zero, and what detection it costs.
    if n_neg and n_pos:
        thr0 = float(neg.max()) + 1e-6
        det_at0 = int((pos >= thr0).sum())
        print(
            f"\nFP=0 needs thr > {neg.max():.4f} (max negative peak); "
            f"at that thr positive detection = {det_at0}/{n_pos} = {det_at0/n_pos:.4f} "
            f"(loses {n_pos - det_at0} positives)"
        )

    # Who are the worst false positives at thr=0.5?
    fp_rows = sorted(
        [r for r in rows if not r["is_positive"] and r["pred_peak"] >= 0.5],
        key=lambda r: -r["pred_peak"],
    )
    print(f"\n-- false positives at thr=0.5: {len(fp_rows)} --")
    by_src: dict[str, int] = {}
    for r in fp_rows:
        by_src[r["source"]] = by_src.get(r["source"], 0) + 1
    for s, c in sorted(by_src.items(), key=lambda kv: -kv[1]):
        print(f"  {s:<48s} {c:4d}")
    print("  (top 20 by peak)")
    for r in fp_rows[:20]:
        print(f"  {r['pred_peak']:.4f}  {r['source']:<40s} {r['filename']}")


if __name__ == "__main__":
    main()
