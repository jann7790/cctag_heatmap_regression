#!/usr/bin/env python3
"""Task 3: distance-based re-split (leakage-free), using marker radius as the
distance axis (user choice). Produces split LIST FILES only -- does NOT touch the
dataloader or any existing data.

Design
------
* Distance proxy = full-frame marker radius r = 0.5*(ellipse_a+ellipse_b) from
  the SOURCE dataset 6f_labeled (reliable across resolution; ROI-crop ellipses
  are rescaled and unreliable).
* REAL holdout = the FAR excursions (r < R_LO). The marker goes near<->far ~4x,
  so "far" is a set of contiguous troughs; we hold out their complete frames.
* Frame-grouped: every crop (pos / neg / occ) of a source frame goes to the SAME
  bucket -> kills the same-frame near-duplicate leak found in Task 1.
* TEMPORAL buffer: any non-holdout frame within K ranks (time) of a holdout frame
  is DROPPED -> kills the boundary near-duplicate leak (adjacent frames at ~same
  distance straddling the threshold).
* SYNTHETIC dev-val = random 10% of the synthetic set (early-stopping / checkpoint
  selection only -- explicitly NOT a generalization score).

Tunables below. Read-only on datasets; writes only into data_diagnosis/.
"""
from __future__ import annotations

import csv
import json
import random
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent
LISTDIR = OUT / "split_lists"
LISTDIR.mkdir(exist_ok=True)

SRC_LABELED = ROOT / "outputs/datasets/6f_labeled"
REAL_ROI = [
    ROOT / "outputs/datasets/6f_labeled_1024x640_roi",
    ROOT / "outputs/datasets/6f_labeled_1024x640_roi_occ",
]
SYNTHETIC = ROOT / "outputs/training_sets/generated_training_sets_1024/mixed_train_dataset"
OTHER_TRAIN = [  # real-negative / hard-negative sets -> train (distance-agnostic, FPR)
    ROOT / "outputs/training_sets/real_world_merged_1024x640",
    ROOT / "outputs/datasets/hard_negative_random_20260608_230541_1024x640",
    ROOT / "outputs/datasets/hard_negative_random_20260608_231324_1024x640",
    ROOT / "outputs/datasets/hard_negative_random_20260608_231516_1024x640",
]

R_LO = 92.0          # radius threshold: r < R_LO == "far" holdout (~20th pct)
BUFFER_K = 3         # temporal buffer: drop train frames within K ranks of holdout
DEV_VAL_FRAC = 0.10  # synthetic random dev-val fraction
SEED = 42

TS_RE = re.compile(r"frame_(\d{8}_\d{6})")


def isneg(r: dict) -> bool:
    return r["is_negative"].strip() in ("1", "True", "true")


def load_filenames(d: Path) -> list[str]:
    with (d / "labels.csv").open(newline="") as fh:
        return [row["filename"].strip() for row in csv.DictReader(fh)
                if row.get("filename", "").strip()]


def main() -> None:
    # 1. Per source-frame radius + chronological rank.
    rows = list(csv.DictReader(open(SRC_LABELED / "labels.csv")))
    for r in rows:
        r["_ts"] = r["filename"].replace("frame_", "")
    rows.sort(key=lambda r: r["_ts"])
    frame_radius: dict[str, float | None] = {}
    frame_rank: dict[str, int] = {}
    for i, r in enumerate(rows):
        ts = r["_ts"]
        frame_rank[ts] = i
        frame_radius[ts] = None if isneg(r) else 0.5 * (float(r["ellipse_a"]) + float(r["ellipse_b"]))

    # 2. Holdout frames = far detected frames (r < R_LO).
    holdout_ranks = {frame_rank[ts] for ts, rad in frame_radius.items()
                     if rad is not None and rad < R_LO}
    # 3. Temporal buffer ranks = within K of a holdout rank, but not holdout themselves.
    buffer_ranks = set()
    for hr in holdout_ranks:
        for d in range(-BUFFER_K, BUFFER_K + 1):
            rr = hr + d
            if rr not in holdout_ranks:
                buffer_ranks.add(rr)

    def frame_bucket(ts: str) -> str:
        rk = frame_rank.get(ts)
        if rk is None:
            return "train"  # frame not in source labels (shouldn't happen) -> safe default
        if rk in holdout_ranks:
            return "holdout"
        if rk in buffer_ranks:
            return "buffer"
        return "train"

    # 4. Assign every REAL ROI crop by its source-frame bucket.
    buckets: dict[str, list[tuple[str, str]]] = defaultdict(list)  # bucket -> [(dataset_rel, filename)]
    real_frame_counts: dict[str, set] = defaultdict(set)
    for d in REAL_ROI:
        rel = str(d.relative_to(ROOT))
        for fn in load_filenames(d):
            m = TS_RE.search(fn)
            ts = m.group(1) if m else None
            b = frame_bucket(ts) if ts else "train"
            buckets[b].append((rel, fn))
            if ts:
                real_frame_counts[b].add(ts)

    # 5. Synthetic random dev-val split (frame-id = filename; synthetic frames are independent).
    syn_files = load_filenames(SYNTHETIC)
    rel_syn = str(SYNTHETIC.relative_to(ROOT))
    idx = list(range(len(syn_files)))
    random.Random(SEED).shuffle(idx)
    n_dev = int(len(idx) * DEV_VAL_FRAC)
    dev_set = set(idx[:n_dev])
    for i, fn in enumerate(syn_files):
        if i in dev_set:
            buckets["dev_val"].append((rel_syn, fn))
        else:
            buckets["train"].append((rel_syn, fn))

    # 6. Other real-negative sets -> train.
    for d in OTHER_TRAIN:
        rel = str(d.relative_to(ROOT))
        for fn in load_filenames(d):
            buckets["train"].append((rel, fn))

    # 7. Write the three required list files (+ buffer for transparency).
    def write_list(name: str, rows_):
        p = LISTDIR / name
        with p.open("w") as fh:
            fh.write("# dataset_dir\tfilename\n")
            for rel, fn in rows_:
                fh.write(f"{rel}\t{fn}\n")
        return len(rows_)

    n_train = write_list("train.txt", buckets["train"])
    n_dev = write_list("synthetic_dev_val.txt", buckets["dev_val"])
    n_hold = write_list("real_holdout.txt", buckets["holdout"])
    n_buf = write_list("real_buffer_dropped.txt", buckets["buffer"])

    # 8. Distance distribution: holdout vs real-train radius.
    def real_radii(bucket):
        out = []
        for rel, fn in buckets[bucket]:
            if "6f_labeled_1024x640_roi" not in rel:
                continue
            m = TS_RE.search(fn)
            if m and frame_radius.get(m.group(1)) is not None:
                out.append(frame_radius[m.group(1)])
        return np.array(out)

    tr_r, ho_r = real_radii("train"), real_radii("holdout")
    plt.figure(figsize=(10, 5))
    bins = np.linspace(50, 1100, 50)
    plt.hist(tr_r, bins=bins, alpha=0.6, label=f"real TRAIN positives (n={len(tr_r)})", color="#2980b9")
    plt.hist(ho_r, bins=bins, alpha=0.7, label=f"real HOLDOUT positives (n={len(ho_r)})", color="#c0392b")
    plt.axvline(R_LO, ls="--", color="black", label=f"R_LO={R_LO:.0f}px")
    plt.xlabel("marker radius (px)  — left = FAR (held out), right = NEAR (train)")
    plt.ylabel("# positive crops")
    plt.title("Distance distribution after re-split (no radius overlap = no boundary leak)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "task3_distance_split.png", dpi=130)
    plt.close()

    manifest = {
        "params": {"R_LO": R_LO, "BUFFER_K": BUFFER_K, "DEV_VAL_FRAC": DEV_VAL_FRAC, "SEED": SEED},
        "real_source_frames": {
            "holdout_frames": len(real_frame_counts["holdout"]),
            "train_frames": len(real_frame_counts["train"]),
            "buffer_frames_dropped": len(real_frame_counts["buffer"]),
        },
        "crop_counts": {
            "train_total": n_train,
            "synthetic_dev_val": n_dev,
            "real_holdout": n_hold,
            "real_buffer_dropped": n_buf,
        },
        "real_holdout_radius_px": {
            "n_positives": int(len(ho_r)),
            "min": float(ho_r.min()) if len(ho_r) else None,
            "max": float(ho_r.max()) if len(ho_r) else None,
        },
        "real_train_radius_px": {
            "n_positives": int(len(tr_r)),
            "min": float(tr_r.min()) if len(tr_r) else None,
            "max": float(tr_r.max()) if len(tr_r) else None,
        },
    }
    (OUT / "task3_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))
    print(f"\nLists written to {LISTDIR}/  (train.txt, synthetic_dev_val.txt, real_holdout.txt, real_buffer_dropped.txt)")


if __name__ == "__main__":
    main()
