#!/usr/bin/env python3
"""Task 1: quantify train/val leakage of the existing random split.

Replicates exactly the split used by train_cctag_heatmap_ddp.py for the
`fable_occ_1024_offset` run config (split_indices: random.Random(seed).shuffle
over the ConcatDataset of all train sources, then carve out 1 - train_ratio as
val). We then restrict to the two REAL ROI datasets the user cares about and
measure, for every real-frame crop that lands in val, the frame-index distance
to its nearest real-frame crop in train.

Read-only. Writes only into data_diagnosis/.
"""
from __future__ import annotations

import csv
import random
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent

# Concat order + lengths must mirror the launch script exactly.
SOURCES = [
    ("mixed", ROOT / "outputs/training_sets/generated_training_sets_1024/mixed_train_dataset"),
    ("roi", ROOT / "outputs/datasets/6f_labeled_1024x640_roi"),
    ("roi_occ", ROOT / "outputs/datasets/6f_labeled_1024x640_roi_occ"),
    ("real_world_merged", ROOT / "outputs/training_sets/real_world_merged_1024x640"),
    ("hardneg1", ROOT / "outputs/datasets/hard_negative_random_20260608_230541_1024x640"),
    ("hardneg2", ROOT / "outputs/datasets/hard_negative_random_20260608_231324_1024x640"),
    ("hardneg3", ROOT / "outputs/datasets/hard_negative_random_20260608_231516_1024x640"),
]
TRAIN_RATIO = 0.9
SEED = 42
REAL_TAGS = {"roi", "roi_occ"}

# frame_YYYYMMDD_HHMMSS(_suffix) -> the YYYYMMDD_HHMMSS source-frame timestamp.
TS_RE = re.compile(r"frame_(\d{8}_\d{6})")


def load_filenames(dataset_dir: Path) -> list[str]:
    rows = []
    with (dataset_dir / "labels.csv").open(newline="") as fh:
        for row in csv.DictReader(fh):
            fn = row.get("filename", "").strip()
            if fn:
                rows.append(fn)
    return rows


def split_indices(n: int, train_ratio: float, seed: int):
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)
    train_size = max(1, int(n * train_ratio))
    if n - train_size == 0:
        train_size = n - 1
    return indices[:train_size], indices[train_size:]


def main() -> None:
    # 1. Build the global sample table in concat order.
    global_tag: list[str] = []
    global_ts: list[str | None] = []
    per_source_counts = {}
    for tag, d in SOURCES:
        names = load_filenames(d)
        per_source_counts[tag] = len(names)
        for fn in names:
            global_tag.append(tag)
            m = TS_RE.search(fn)
            global_ts.append(m.group(1) if m else None)
    total = len(global_tag)

    # 2. Replicate the exact split.
    train_idx, val_idx = split_indices(total, TRAIN_RATIO, SEED)
    train_set = set(train_idx)

    # 3. Frame-index ranking over the real 6f timeline (sorted unique timestamps).
    real_ts_all = sorted({global_ts[i] for i in range(total)
                          if global_tag[i] in REAL_TAGS and global_ts[i]})
    ts_rank = {ts: r for r, ts in enumerate(real_ts_all)}

    # 4. Train/val membership of real crops, indexed by frame rank.
    train_frame_ranks: set[int] = set()
    for gi in train_idx:
        if global_tag[gi] in REAL_TAGS and global_ts[gi]:
            train_frame_ranks.add(ts_rank[global_ts[gi]])
    sorted_train_ranks = sorted(train_frame_ranks)

    val_real = [gi for gi in val_idx if global_tag[gi] in REAL_TAGS and global_ts[gi]]

    # 5. For each real val crop, nearest train frame-rank distance.
    import bisect

    distances: list[int] = []
    for gi in val_real:
        r = ts_rank[global_ts[gi]]
        if r in train_frame_ranks:
            distances.append(0)
            continue
        pos = bisect.bisect_left(sorted_train_ranks, r)
        cand = []
        if pos < len(sorted_train_ranks):
            cand.append(sorted_train_ranks[pos] - r)
        if pos > 0:
            cand.append(r - sorted_train_ranks[pos - 1])
        distances.append(min(cand) if cand else 10**9)

    # 6. Report.
    n_val_real = len(distances)
    lines = []
    lines.append("# Task 1 — existing random-split leakage (fable_occ_1024_offset config)\n")
    lines.append(f"Total merged samples: {total}")
    lines.append(f"Per-source counts: {per_source_counts}")
    lines.append(f"train_ratio={TRAIN_RATIO} seed={SEED}  ->  train={len(train_idx)} val={len(val_idx)}")
    lines.append(f"Unique real source frames on timeline: {len(real_ts_all)}")
    lines.append(f"Real-frame crops landing in val: {n_val_real}\n")
    lines.append("Nearest-train-neighbour frame-index distance for real val crops:")
    for N in (0, 1, 2, 5, 10):
        c = sum(1 for d in distances if d <= N)
        lines.append(f"  <= {N:2d} frames : {c:5d} / {n_val_real}  ({100*c/n_val_real:.1f}%)")
    same_frame = sum(1 for d in distances if d == 0)
    lines.append(
        f"\nSame source frame already in train (distance 0): {same_frame} "
        f"({100*same_frame/n_val_real:.1f}%)  <- near-duplicate crops of the SAME frame."
    )
    report = "\n".join(lines)
    print(report)
    (OUT / "task1_leakage_summary.txt").write_text(report + "\n")

    # 7. Histogram.
    clip = [min(d, 30) for d in distances]
    plt.figure(figsize=(9, 5))
    plt.hist(clip, bins=range(0, 32), color="#c0392b", edgecolor="white")
    plt.axvline(2.5, color="black", ls="--", lw=1)
    plt.xlabel("frame-index distance to nearest TRAIN crop (clipped at 30)")
    plt.ylabel("# real val crops")
    plt.title(f"Existing random split — leakage of real frames into val\n"
              f"{same_frame}/{n_val_real} val crops share a source frame with train")
    plt.tight_layout()
    plt.savefig(OUT / "task1_leakage_hist.png", dpi=130)
    print(f"\nSaved: {OUT/'task1_leakage_hist.png'}")


if __name__ == "__main__":
    main()
