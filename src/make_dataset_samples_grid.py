#!/usr/bin/env python3
"""Render a 2x3 "dataset samples" montage for the report (sec. 4.3.3 / 4.3.4).

A mix of synthetic and real samples spanning the difficulty groups (clean, heavy
occlusion, negatives) plus a real hardware-occlusion capture.

    Row 1:  (a) synthetic clean   (b) synthetic heavy occ   (c) real hardware occlusion
    Row 2:  (d) real clean        (e) real + synthetic occ  (f) real negative (bg)

Read-only: never modifies datasets or config. Raw images only -- no overlays drawn.
Grayscale display with fixed 0..255 range so low-light samples stay dark. Prints the
6 chosen file paths.

    uv run python src/make_dataset_samples_grid.py
"""
from __future__ import annotations

import argparse
import csv
import os
import random
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Dataset locations (from fable_occ train_dataset_dirs) ─────────────────────
ROOT = Path(__file__).resolve().parent.parent
D_SYN  = ROOT / "outputs/training_sets/generated_training_sets_1024/mixed_train_dataset"
D_REAL = ROOT / "outputs/datasets/6f_labeled_1024x640_roi"
D_OCC  = ROOT / "outputs/datasets/6f_labeled_1024x640_roi_occ"
D_NEG  = ROOT / "outputs/training_sets/real_world_merged_1024x640"

SHORT_SIDE = 640  # all sets are 1024x640


def fnum(x: str):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def load_rows(d: Path) -> list[dict]:
    with open(d / "labels.csv", newline="") as f:
        return list(csv.DictReader(f))


def img_path(d: Path, filename: str) -> Path:
    p = d / "images" / filename
    return p if p.exists() else d / "images" / f"{filename}.png"


def is_negative(r: dict) -> bool:
    return str(r.get("is_negative", "")).strip() in ("1", "true", "True")


def bbox_diag(r: dict) -> float:
    w = (fnum(r["bbox_xmax"]) or 0) - (fnum(r["bbox_xmin"]) or 0)
    h = (fnum(r["bbox_ymax"]) or 0) - (fnum(r["bbox_ymin"]) or 0)
    return float((w * w + h * h) ** 0.5)


def roundness(r: dict) -> float:
    """ellipse short/long axis ratio in (0,1]; 1.0 = perfect circle (frontal marker).

    Tilted/perspective markers project to elongated ellipses; preferring high roundness
    keeps the concentric-ring structure legible in the figure. Returns 0.0 if unknown.
    """
    a, b = fnum(r.get("ellipse_a")), fnum(r.get("ellipse_b"))
    if not a or not b or a <= 0 or b <= 0:
        return 0.0
    return min(a, b) / max(a, b)


def center_to_edge(r: dict, w: int = 1024, h: int = 640) -> float:
    cx, cy = fnum(r["center_x"]), fnum(r["center_y"])
    if cx is None or cy is None:
        return -1.0
    return float(min(cx, w - cx, cy, h - cy))


def gray_stats(path: Path) -> tuple[float, float] | None:
    im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if im is None:
        return None
    return float(cv2.Laplacian(im, cv2.CV_64F).var()), float(im.mean())


def center_crop_aspect(img: np.ndarray, aspect: float) -> np.ndarray:
    """Center-crop a grayscale image to the target width/height aspect ratio.

    A no-op when the image already matches, so panels already at the montage aspect
    (1024x640 = 1.6) pass through untouched while a wider/taller capture is trimmed to
    sit consistently in the grid.
    """
    h, w = img.shape[:2]
    cur = w / h
    if cur > aspect:                       # too wide -> trim width
        nw = int(round(h * aspect))
        x0 = (w - nw) // 2
        return img[:, x0:x0 + nw]
    if cur < aspect:                       # too tall -> trim height
        nh = int(round(w / aspect))
        y0 = (h - nh) // 2
        return img[y0:y0 + nh, :]
    return img


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="2x4 dataset sample montage")
    p.add_argument("--output", type=Path, default=ROOT / "outputs/figures/dataset_samples_grid.png")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--occ_lo", type=float, default=0.60, help="heavy-occlusion lower bound")
    p.add_argument("--occ_hi", type=float, default=0.80, help="heavy-occlusion upper bound")
    p.add_argument("--dpi", type=int, default=200)
    return p.parse_args()


def pick_first(rows, key_filter, rng, dirpath, gate=None, rank_key=None, skip=0, max_probe=400):
    """Pick a candidate whose image passes the optional stat gate.

    Candidates are seeded-shuffled first (reproducible), then, if `rank_key` is given,
    stable-sorted by it descending -- so we take the highest-ranked (e.g. roundest)
    sample, with ties broken by the seeded shuffle.

    `skip` returns the (skip+1)-th qualifying candidate instead of the first, so a panel
    can be re-rolled to a different sample. The shuffle (the only RNG draw) is unchanged
    by `skip`, so re-rolling one panel never perturbs the others' picks.
    """
    cand = [r for r in rows if key_filter(r)]
    rng.shuffle(cand)
    if rank_key is not None:
        cand.sort(key=rank_key, reverse=True)
    fallback = None
    passed = 0
    for r in cand[:max_probe]:
        p = img_path(dirpath, r["filename"])
        if not p.exists():
            continue
        if gate is not None:
            st = gray_stats(p)
            if st is None:
                continue
            if fallback is None:
                fallback = (r, p, len(cand))
            if not gate(st):
                continue
        if passed == skip:
            return r, p, len(cand)
        passed += 1
    return (fallback if fallback else (None, None, len(cand)))


def pick_extreme(rows, key_filter, rng, dirpath, score, gate=None, skip=0, max_probe=300):
    """Pick the candidate minimizing `score(sharpness, brightness)` (for 'degraded').

    `gate(sharpness, brightness)` (optional) rejects candidates outright -- used to
    require a minimum brightness so we never pick a near-black blank frame. `skip`
    returns the (skip+1)-th best by `score` (e.g. 2nd-blurriest) for re-rolling; the
    shuffle (only RNG draw) is unchanged, so re-rolling never perturbs other panels.
    """
    cand = [r for r in rows if key_filter(r)]
    rng.shuffle(cand)
    scored = []
    for r in cand[:max_probe]:
        p = img_path(dirpath, r["filename"])
        st = gray_stats(p)
        if st is None:
            continue
        if gate is not None and not gate(*st):
            continue
        scored.append((score(*st), r, p))
    if not scored:
        return None, None, len(cand)
    scored.sort(key=lambda t: t[0])  # stable -> ties keep seeded-shuffle order
    _, r, p = scored[min(skip, len(scored) - 1)]
    return r, p, len(cand)


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    syn  = load_rows(D_SYN)
    real = load_rows(D_REAL)
    occ  = load_rows(D_OCC)
    neg  = load_rows(D_NEG)

    panels = []  # (title, dirpath, row_dict, img_path, is_positive)
    warnings = []

    def add(title, dirpath, r, p, positive, note=None):
        if r is None and p is None:
            warnings.append(f"!! {title}: NO suitable sample found in {dirpath.name}")
        panels.append((title, dirpath, r, p, positive))
        chosen = (p.name if p else "NONE")
        extra = f"  [{note}]" if note else ""
        print(f"{title:34s} <- {dirpath.name}/{chosen}{extra}")

    # (a) synthetic / clean  -- unoccluded, decent-size, in-frame, sharp & lit, roundest
    r, p, n = pick_first(
        syn,
        lambda r: not is_negative(r) and (fnum(r["occlusion_ratio"]) or 0) < 0.05
                  and 120 < bbox_diag(r) < 420 and center_to_edge(r) > 0.5 * bbox_diag(r),
        rng, D_SYN, gate=lambda st: st[0] > 200 and 45 < st[1] < 210, rank_key=roundness)
    add("(a) synthetic / clean", D_SYN, r, p, True,
        note=f"b/a={roundness(r):.2f}" if r else None)

    # (b) synthetic / heavy occlusion  -- occ in [occ_lo, occ_hi], marker still partly
    # visible; among the band take the roundest underlying marker, gated to a brightly-lit
    # frame (mean > 150) so the occlusion reads clearly instead of sitting in the dark.
    r, p, n = pick_first(
        syn,
        lambda r: not is_negative(r) and args.occ_lo <= (fnum(r["occlusion_ratio"]) or 0) <= args.occ_hi
                  and str(r.get("has_visible_marker", "")).strip() in ("1", "true", "True")
                  and 120 < bbox_diag(r) < 480 and roundness(r) >= 0.85,
        rng, D_SYN, gate=lambda st: st[1] > 150, rank_key=roundness, skip=0)
    if r is None:  # fall back: widen to >0.5 occlusion
        r, p, n = pick_first(
            syn,
            lambda r: not is_negative(r) and (fnum(r["occlusion_ratio"]) or 0) > 0.5 and 120 < bbox_diag(r) < 480,
            rng, D_SYN, rank_key=roundness)
        warnings.append("(b) heavy occlusion: relaxed to occ>0.5 (no sample in band)")
    add("(b) synthetic / heavy occlusion", D_SYN, r, p, True,
        note=f"occ={fnum(r['occlusion_ratio']):.2f} b/a={roundness(r):.2f}" if r else None)

    # (c) real / hardware occlusion  -- a real CCTag occluded by actual optical hardware
    # (mounts, lens tube, clamps): the real-world occlusion the synthetic occluder mimics.
    # Fixed capture; falls back to a synthetic blurry marker if it's missing. The
    # pick_extreme call also keeps the shared RNG stream stable for panels (d)-(f).
    r, p, n = pick_extreme(
        syn,
        lambda r: not is_negative(r) and (fnum(r["occlusion_ratio"]) or 0) < 0.20
                  and 150 < bbox_diag(r) < 360 and roundness(r) >= 0.85
                  and center_to_edge(r) > 0.5 * bbox_diag(r),
        rng, D_SYN,
        score=lambda sharp, bright: sharp,      # minimize sharpness => blurriest
        gate=lambda sharp, bright: bright > 40,  # ...but keep it visible
        skip=1)
    c_capture = ROOT / "outputs/figures/capture_20260608_145930_926.png"
    if c_capture.exists():
        add("(c) real / hardware occlusion", c_capture.parent, None, c_capture, False,
            note="real hardware capture")
    else:
        warnings.append("(c) hardware capture missing; fell back to synthetic degraded")
        note = None
        if p is not None:
            st = gray_stats(p); note = f"lapvar={st[0]:.1f}, mean={st[1]:.0f}, b/a={roundness(r):.2f}"
        add("(c) synthetic / degraded", D_SYN, r, p, True, note=note)

    # (d) real / clean  -- roundest unoccluded in-frame real marker
    r, p, n = pick_first(
        real,
        lambda r: not is_negative(r) and (fnum(r["occlusion_ratio"]) or 0) < 0.05
                  and 120 < bbox_diag(r) < 420 and center_to_edge(r) > 0.6 * bbox_diag(r),
        rng, D_REAL, gate=lambda st: st[0] > 60, rank_key=roundness)
    add("(d) real / clean", D_REAL, r, p, True,
        note=f"b/a={roundness(r):.2f}" if r else None)
    d_filename = r["filename"] if r else None

    # (e) real + synthetic occlusion  -- pinned to the occluded counterpart of (d)'s real
    # marker, so the two real panels show the same marker clean (d) vs. occluded (e). The
    # pick_first call below is the fallback (roundest heavy-occlusion sample if (d) has no
    # occ variant) and also keeps the shared RNG stream stable for panel (f).
    r, p, n = pick_first(
        occ,
        lambda r: not is_negative(r) and 0.50 <= (fnum(r["occlusion_ratio"]) or 0) <= 0.72
                  and bbox_diag(r) > 120 and roundness(r) >= 0.85,
        rng, D_OCC, rank_key=roundness, skip=2)
    if d_filename is not None:
        d_stem = Path(d_filename).stem
        variants = [row for row in occ if not is_negative(row)
                    and Path(row["filename"]).stem.startswith(d_stem + "_occ")]
        if variants:
            variants.sort(key=lambda row: -(fnum(row["occlusion_ratio"]) or 0))  # most occluded
            r = variants[0]
            p = img_path(D_OCC, r["filename"])
    add("(e) real + synthetic occlusion", D_OCC, r, p, True,
        note=f"occ={fnum(r['occlusion_ratio']):.2f} b/a={roundness(r):.2f}" if r else None)

    # (f) real negative background
    r, p, n = pick_first(neg, lambda r: is_negative(r), rng, D_NEG, skip=1)
    add("(f) real / negative (background)", D_NEG, r, p, False)

    # ── Render 2x3 grid ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), dpi=args.dpi)
    axes = axes.ravel()
    for ax, (title, dirpath, r, p, positive) in zip(axes, panels):
        ax.set_xticks([]); ax.set_yticks([]); ax.axis("off")
        ax.set_title(title, fontsize=11)
        if p is None or not p.exists():
            ax.text(0.5, 0.5, "missing", ha="center", va="center", transform=ax.transAxes)
            continue
        gray = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        gray = center_crop_aspect(gray, 1024 / 640)  # trim wide captures to the grid aspect
        # fixed display range -> dark samples stay dark (no per-image normalize)
        ax.imshow(gray, cmap="gray", vmin=0, vmax=255)

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    plt.close(fig)

    print(f"\nsaved: {args.output}")
    if warnings:
        print("\nWARNINGS:")
        for w in warnings:
            print("  " + w)


if __name__ == "__main__":
    main()
