#!/usr/bin/env python3
"""Task 2: CCTag label quality vs distance.

Distance proxy = marker radius in px on the source 4096x2160 frame,
r = 0.5*(ellipse_a + ellipse_b). Larger r = nearer, smaller r = farther.
(User confirmed time is NOT a monotonic distance axis, so we use marker size.)

Missed frames (CCTag no_detection) have no ellipse, so we estimate their radius
by linear interpolation from the temporally-nearest DETECTED frames. This is
locally valid because adjacent-second frames sit at ~the same distance, and it
lets us draw an honest detection-rate-vs-distance curve.

Read-only on the dataset. Writes only into data_diagnosis/.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent
SRC = ROOT / "outputs/datasets/6f_labeled"
SAMPLE_DIR = OUT / "task2_samples"
SAMPLE_DIR.mkdir(exist_ok=True)


def isneg(r: dict) -> bool:
    return r["is_negative"].strip() in ("1", "True", "true")


def ts_of(fn: str) -> str:
    return fn.replace("frame_", "")


def main() -> None:
    rows = list(csv.DictReader(open(SRC / "labels.csv")))
    for r in rows:
        r["_ts"] = ts_of(r["filename"])
    rows.sort(key=lambda r: r["_ts"])  # chronological
    n = len(rows)

    detected = []  # (rank, radius)
    for i, r in enumerate(rows):
        r["_rank"] = i
        if not isneg(r):
            rad = 0.5 * (float(r["ellipse_a"]) + float(r["ellipse_b"]))
            r["_radius"] = rad
            detected.append((i, rad))
        else:
            r["_radius"] = None

    det_ranks = np.array([d[0] for d in detected])
    det_rads = np.array([d[1] for d in detected])

    # Classify missed frames: "isolated" (within 3 ranks of a detection -> its
    # distance is trustworthy by local time-continuity) vs "block" (inside a long
    # all-missed run -> distance unknown, interpolation would be bogus).
    import bisect

    GAP = 3
    for r in rows:
        if r["_radius"] is not None:
            r["_radius_est"] = r["_radius"]
            r["_miss_kind"] = "detected"
            continue
        p = bisect.bisect_left(det_ranks.tolist(), r["_rank"])
        gap = min([abs(int(det_ranks[j]) - r["_rank"])
                   for j in (p - 1, p) if 0 <= j < len(det_ranks)])
        r["_radius_est"] = float(np.interp(r["_rank"], det_ranks, det_rads))
        r["_miss_kind"] = "isolated" if gap <= GAP else "block"

    # Contiguous miss-run structure (the real story: block dropouts).
    runs = []
    cur, start = 0, None
    for i, r in enumerate(rows):
        if r["_radius"] is None:
            if cur == 0:
                start = i
            cur += 1
        elif cur > 0:
            runs.append((start, cur))
            cur = 0
    if cur > 0:
        runs.append((start, cur))
    runs.sort(key=lambda x: -x[1])
    n_block = sum(1 for r in rows if r["_miss_kind"] == "block")
    n_iso = sum(1 for r in rows if r["_miss_kind"] == "isolated")

    n_det = len(detected)
    n_miss = n - n_det
    print(f"frames={n}  detected={n_det} ({100*n_det/n:.1f}%)  missed={n_miss} ({100*n_miss/n:.1f}%)")

    # ---- Plot 1: radius vs time (trajectory) + missed rug ----
    plt.figure(figsize=(11, 5))
    det_x = [r["_rank"] for r in rows if r["_radius"] is not None]
    det_y = [r["_radius"] for r in rows if r["_radius"] is not None]
    miss_x = [r["_rank"] for r in rows if r["_radius"] is None]
    plt.scatter(det_x, det_y, s=10, c="#2980b9", label="detected (labelled)")
    for mx in miss_x:
        plt.axvline(mx, color="#c0392b", alpha=0.18, lw=0.8)
    plt.scatter(miss_x, [np.interp(m, det_ranks, det_rads) for m in miss_x],
                s=12, c="#c0392b", marker="x", label="missed (no detection)")
    plt.xlabel("frame rank (chronological)  — NOT linear in distance")
    plt.ylabel("marker radius (px)  — larger = nearer")
    plt.title("Marker-size trajectory over the recording (back-and-forth) +\n"
              "where CCTag failed to detect")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "task2_radius_vs_time.png", dpi=130)
    plt.close()

    # ---- Plot 2: detection rate vs radius bin (distance proxy) ----
    # Honest curve: detected + ISOLATED misses only. Block-dropout misses are
    # excluded (their distance is unknown) and reported separately.
    curve_rows = [r for r in rows if r["_miss_kind"] in ("detected", "isolated")]
    all_rad = np.array([r["_radius_est"] for r in curve_rows])
    is_det = np.array([r["_radius"] is not None for r in curve_rows])
    edges = np.array([0, 80, 120, 160, 220, 300, 450, 700, 1100], dtype=float)
    centers, rates, counts = [], [], []
    table_lines = ["radius_bin_px        frames  detected  rate   (detected + isolated-miss only)"]
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (all_rad >= lo) & (all_rad < hi)
        tot = int(m.sum())
        if tot == 0:
            continue
        det = int(is_det[m].sum())
        centers.append((lo + hi) / 2)
        rates.append(det / tot)
        counts.append(tot)
        table_lines.append(f"[{lo:4.0f},{hi:4.0f})        {tot:5d}    {det:5d}   {det/tot:5.2f}")
    plt.figure(figsize=(10, 5))
    ax1 = plt.gca()
    ax1.bar(range(len(centers)), rates, color="#27ae60", alpha=0.85)
    ax1.set_xticks(range(len(centers)))
    ax1.set_xticklabels([f"{int(edges[i])}-{int(edges[i+1])}" for i in range(len(centers))],
                        rotation=30, ha="right")
    ax1.set_ylabel("CCTag detection rate", color="#27ae60")
    ax1.set_ylim(0, 1.05)
    ax1.set_xlabel("marker radius bin (px)  — left = FAR, right = NEAR")
    ax2 = ax1.twinx()
    ax2.plot(range(len(centers)), counts, "o--", color="#34495e")
    ax2.set_ylabel("# frames in bin", color="#34495e")
    plt.title("CCTag detection rate vs distance (marker size)\n"
              f"detected + {n_iso} isolated misses; {n_block} block-dropout misses excluded")
    plt.tight_layout()
    plt.savefig(OUT / "task2_detection_rate_vs_distance.png", dpi=130)
    plt.close()

    # ---- Plot 3: label density histogram (detected only) ----
    plt.figure(figsize=(10, 5))
    plt.hist(det_rads, bins=np.linspace(0, 1100, 45), color="#8e44ad")
    plt.xlabel("marker radius (px)  — left = FAR, right = NEAR")
    plt.ylabel("# detected (labelled) frames")
    plt.title(f"Label density vs distance — only {n_det} labelled frames, "
              f"sparse at small radius (far)")
    plt.tight_layout()
    plt.savefig(OUT / "task2_label_density.png", dpi=130)
    plt.close()

    # ---- Task 2.4: visual sampling near/mid/far + missed ----
    det_rows = sorted([r for r in rows if r["_radius"] is not None], key=lambda r: r["_radius"])
    far5 = det_rows[:5]                      # smallest radius
    mid_i = len(det_rows) // 2
    mid5 = det_rows[mid_i - 2: mid_i + 3]
    near5 = det_rows[-5:]                     # largest radius
    miss_rows = [r for r in rows if r["_radius"] is None]
    miss5 = miss_rows[:: max(1, len(miss_rows) // 5)][:5]

    def overlay_detected(r, tag):
        img = Image.open(SRC / "images" / f"{r['filename']}.png").convert("RGB")
        cx, cy = float(r["center_x"]), float(r["center_y"])
        rad = r["_radius"]
        half = int(max(rad * 1.6, 200))
        box = (int(cx - half), int(cy - half), int(cx + half), int(cy + half))
        crop = img.crop(box)
        d = ImageDraw.Draw(crop)
        lcx, lcy = cx - box[0], cy - box[1]
        d.line([(lcx - 25, lcy), (lcx + 25, lcy)], fill=(255, 0, 0), width=3)
        d.line([(lcx, lcy - 25), (lcx, lcy + 25)], fill=(255, 0, 0), width=3)
        d.ellipse([lcx - rad, lcy - rad, lcx + rad, lcy + rad], outline=(0, 255, 0), width=3)
        out = SAMPLE_DIR / f"{tag}_r{int(rad)}_{r['filename']}.png"
        crop.save(out)
        return out.name

    def save_missed(r):
        img = Image.open(SRC / "images" / f"{r['filename']}.png").convert("RGB")
        img.thumbnail((1280, 1280))
        out = SAMPLE_DIR / f"missed_{r['filename']}.png"
        img.save(out)
        return out.name

    sample_log = ["# Task 2.4 sample overlays (in data_diagnosis/task2_samples/)\n"]
    sample_log.append("## NEAR (largest marker)")
    for r in near5:
        sample_log.append(f"  r={r['_radius']:.0f}px  {overlay_detected(r, 'near')}")
    sample_log.append("## MID")
    for r in mid5:
        sample_log.append(f"  r={r['_radius']:.0f}px  {overlay_detected(r, 'mid')}")
    sample_log.append("## FAR (smallest marker)")
    for r in far5:
        sample_log.append(f"  r={r['_radius']:.0f}px  {overlay_detected(r, 'far')}")
    sample_log.append("## MISSED (CCTag no_detection, full frame downscaled)")
    for r in miss5:
        sample_log.append(f"  est_r~{r['_radius_est']:.0f}px  {save_missed(r)}")

    # ---- summary ----
    lines = ["# Task 2 — CCTag label quality vs distance (proxy = marker radius px)\n"]
    lines.append(f"Source: {SRC}")
    lines.append(f"Frames: {n}   detected(labelled): {n_det} ({100*n_det/n:.1f}%)   "
                 f"missed(no_detection): {n_miss} ({100*n_miss/n:.1f}%)")
    lines.append(f"Detected marker radius: min={det_rads.min():.0f} "
                 f"median={np.median(det_rads):.0f} max={det_rads.max():.0f} px\n")
    lines.append(f"Missed frames breakdown: isolated(gap<={GAP}, distance knowable)={n_iso}   "
                 f"block-dropout(distance unknown)={n_block}")
    lines.append("Top contiguous miss-runs (the real failure mode = whole-segment dropouts):")
    for s, l in runs[:8]:
        lines.append(f"    rank {s:4d}  len {l:3d}   {rows[s]['_ts']} -> {rows[s+l-1]['_ts']}")
    lines.append("")
    lines.append("Detection rate vs distance (detected + isolated misses only):")
    lines += ["  " + l for l in table_lines]
    lines.append("")
    lines.append("INTERPRETATION: misses are dominated by contiguous block dropouts "
                 "(dark/low-light end tail, setup start, dark mid-segments), NOT by far "
                 "distance. Within the labelled distance range, far/small markers detect "
                 "at high rate and their labels look accurate (see far_* samples).")
    lines.append("")
    lines += sample_log
    report = "\n".join(lines)
    print(report)
    (OUT / "task2_summary.txt").write_text(report + "\n")
    print(f"\nSaved plots + {SAMPLE_DIR}")


if __name__ == "__main__":
    main()
