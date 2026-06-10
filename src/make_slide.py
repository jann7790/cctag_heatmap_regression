#!/usr/bin/env python3
"""Generate single-page benchmark slide: CNN vs YOLO, 4–10m."""
import csv
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from pathlib import Path
import cv2

OUT = Path("outputs/inference/eval_4to10m/slide.png")

# ── data ─────────────────────────────────────────────────────────────────────

# Accuracy (from eval_4to10m.py results)
cnn = dict(
    tp=708, fp=245, fn=148,
    precision=0.743, recall=0.827, f1=0.783,
    l2_mean=9.51, l2_median=3.90, l2_p90=16.94,
)
yolo = dict(
    tp=107, fp=1061, fn=749,
    precision=0.092, recall=0.125, f1=0.106,
    l2_mean=282.1, l2_median=82.9, l2_p90=748.0,
)
pos_gt = 856
neg_gt = 3219

# Latency (ms, GPU bs=1 forward only)
cnn_lat  = dict(mean=4.72,  p99=4.88,  pipeline=21.0,  fps_fwd=1000/4.72,  fps_pipe=47.5, cpu=185.1)
yolo_lat = dict(mean=13.16, p99=14.64, pipeline=13.16, fps_fwd=76.0,       fps_pipe=76.0, cpu=None)

# Per-distance CNN
dist_bins = ["3.8–5m", "5–8m", "8–10m"]
cnn_tp_pct   = [61.2, 80.0, 99.2]
cnn_l2_med   = [13.18, 4.25, 2.90]
yolo_tp_pct  = [0.0,   7.8,  28.1]
yolo_l2_mean = [386.6, 362.0, 173.0]

# ── layout ────────────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(20, 11.25), facecolor="#0f1117")
fig.text(0.5, 0.965, "CCTag Detection Benchmark  ·  4–10m  (real captures, 50mm, Ø20cm CCTag, 1024×640 tile)",
         ha="center", va="top", fontsize=14, color="white", fontweight="bold")
fig.text(0.5, 0.938, "CNN: heatmap_1024 (ResNet-18)   vs   YOLO: cctag_det_n_30ep (YOLOv11-n)      "
         f"Dataset: {pos_gt} pos + {neg_gt} neg  |  GPU: single tile, bs=1",
         ha="center", va="top", fontsize=9.5, color="#aaaaaa")

gs = gridspec.GridSpec(2, 4, figure=fig,
                       left=0.04, right=0.97, top=0.90, bottom=0.05,
                       wspace=0.38, hspace=0.52)

DARK  = "#1a1d27"
BLUE  = "#4c9be8"
ORG   = "#e87c4c"
GREEN = "#4ce87c"
RED   = "#e84c4c"
GRAY  = "#555566"
WHITE = "white"
LBLC  = "#ccccdd"

def dark_ax(ax):
    ax.set_facecolor(DARK)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333344")
    ax.tick_params(colors=LBLC, labelsize=8)
    ax.xaxis.label.set_color(LBLC)
    ax.yaxis.label.set_color(LBLC)

def title_ax(ax, txt):
    ax.set_title(txt, color=WHITE, fontsize=9.5, fontweight="bold", pad=6)

# ── panel A: summary KPI table ────────────────────────────────────────────────
ax_kpi = fig.add_subplot(gs[0, 0])
ax_kpi.set_facecolor(DARK)
ax_kpi.axis("off")
title_ax(ax_kpi, "Overall  (4–10m subset)")

rows = [
    ("", "CNN", "YOLO"),
    ("TP",        f"{cnn['tp']}",          f"{yolo['tp']}"),
    ("FP",        f"{cnn['fp']}",          f"{yolo['fp']}"),
    ("FN",        f"{cnn['fn']}",          f"{yolo['fn']}"),
    ("Precision", f"{cnn['precision']:.3f}", f"{yolo['precision']:.3f}"),
    ("Recall",    f"{cnn['recall']:.3f}",    f"{yolo['recall']:.3f}"),
    ("F1",        f"{cnn['f1']:.3f}",        f"{yolo['f1']:.3f}"),
    ("L2 median", f"{cnn['l2_median']:.1f} px", f"{yolo['l2_median']:.0f} px"),
    ("L2 p90",    f"{cnn['l2_p90']:.1f} px",    f"{yolo['l2_p90']:.0f} px"),
]

for i, (label, cv, yv) in enumerate(rows):
    y = 1.0 - i * 0.105
    color_c = GREEN if label in ("F1","Recall","Precision","TP") else (RED if label in ("FP","FN") else LBLC)
    color_y = RED if label in ("F1","Recall","Precision","TP") else (GREEN if label == "TN" else RED if label in ("FP","FN") else LBLC)
    if i == 0:
        ax_kpi.text(0.0, y, label, color="#888899", fontsize=8.5, transform=ax_kpi.transAxes)
        ax_kpi.text(0.42, y, cv, color=BLUE, fontsize=8.5, fontweight="bold", transform=ax_kpi.transAxes)
        ax_kpi.text(0.72, y, yv, color=ORG, fontsize=8.5, fontweight="bold", transform=ax_kpi.transAxes)
    else:
        ax_kpi.text(0.0, y, label, color=LBLC, fontsize=8, transform=ax_kpi.transAxes)
        ax_kpi.text(0.42, y, cv, color=BLUE, fontsize=8, transform=ax_kpi.transAxes)
        ax_kpi.text(0.72, y, yv, color=ORG, fontsize=8, transform=ax_kpi.transAxes)

# ── panel B: TP% by distance ─────────────────────────────────────────────────
ax_tp = fig.add_subplot(gs[0, 1])
dark_ax(ax_tp)
title_ax(ax_tp, "TP% by Distance")

x = np.arange(len(dist_bins))
w = 0.35
ax_tp.bar(x - w/2, cnn_tp_pct,  w, color=BLUE, alpha=0.85, label="CNN")
ax_tp.bar(x + w/2, yolo_tp_pct, w, color=ORG,  alpha=0.85, label="YOLO")
ax_tp.set_xticks(x); ax_tp.set_xticklabels(dist_bins, fontsize=8)
ax_tp.set_ylim(0, 115); ax_tp.set_ylabel("TP %", fontsize=8)
ax_tp.axhline(100, color=GRAY, lw=0.5, ls="--")
for i, v in enumerate(cnn_tp_pct):
    ax_tp.text(i - w/2, v + 2, f"{v:.0f}%", ha="center", fontsize=7.5, color=BLUE)
for i, v in enumerate(yolo_tp_pct):
    ax_tp.text(i + w/2, v + 2, f"{v:.0f}%", ha="center", fontsize=7.5, color=ORG)
ax_tp.legend(fontsize=7.5, facecolor=DARK, labelcolor=WHITE, framealpha=0.6)

# ── panel C: L2 median by distance ───────────────────────────────────────────
ax_l2 = fig.add_subplot(gs[0, 2])
dark_ax(ax_l2)
title_ax(ax_l2, "Center L2 Median (px)")

ax_l2.plot(dist_bins, cnn_l2_med,   "o-", color=BLUE, lw=2, ms=6, label="CNN")
ax_l2.set_ylabel("L2 median (px)", fontsize=8)
ax_l2.set_ylim(0, 18)
ax_l2_r = ax_l2.twinx()
ax_l2_r.set_facecolor(DARK)
ax_l2_r.plot(dist_bins, yolo_l2_mean, "s--", color=ORG, lw=2, ms=6, label="YOLO")
ax_l2_r.set_ylabel("YOLO L2 mean (px)", color=ORG, fontsize=7.5)
ax_l2_r.tick_params(axis="y", colors=ORG, labelsize=7.5)
ax_l2_r.set_ylim(0, 500)
for i, v in enumerate(cnn_l2_med):
    ax_l2.text(i, v + 0.5, f"{v:.1f}", ha="center", fontsize=7.5, color=BLUE)
for i, v in enumerate(yolo_l2_mean):
    ax_l2_r.text(i, v + 8, f"{v:.0f}", ha="center", fontsize=7.5, color=ORG)
lines1, labels1 = ax_l2.get_legend_handles_labels()
lines2, labels2 = ax_l2_r.get_legend_handles_labels()
ax_l2.legend(lines1+lines2, labels1+labels2, fontsize=7.5, facecolor=DARK, labelcolor=WHITE, framealpha=0.6)

# ── panel D: latency ─────────────────────────────────────────────────────────
ax_lat = fig.add_subplot(gs[0, 3])
ax_lat.set_facecolor(DARK)
ax_lat.axis("off")
title_ax(ax_lat, "Inference Latency  (GPU, bs=1)")

lat_rows = [
    ("",             "CNN",          "YOLO"),
    ("Fwd mean",     f"{cnn_lat['mean']:.2f} ms",    f"{yolo_lat['mean']:.2f} ms"),
    ("Fwd p99",      f"{cnn_lat['p99']:.2f} ms",     f"{yolo_lat['p99']:.2f} ms"),
    ("Fwd FPS",      f"{cnn_lat['fps_fwd']:.0f}",    f"{yolo_lat['fps_fwd']:.0f}"),
    ("Pipeline",     f"{cnn_lat['pipeline']:.1f} ms", "—"),
    ("Pipeline FPS", f"{cnn_lat['fps_pipe']:.0f}",   "—"),
    ("CPU mean",     f"{cnn_lat['cpu']:.0f} ms",      "N/A"),
    ("GPU bs=16",    "183 FPS",       "—"),
]

for i, (label, cv, yv) in enumerate(lat_rows):
    y = 1.0 - i * 0.112
    if i == 0:
        ax_lat.text(0.0, y, label, color="#888899", fontsize=8.5, transform=ax_lat.transAxes)
        ax_lat.text(0.42, y, cv, color=BLUE, fontsize=8.5, fontweight="bold", transform=ax_lat.transAxes)
        ax_lat.text(0.72, y, yv, color=ORG,  fontsize=8.5, fontweight="bold", transform=ax_lat.transAxes)
    else:
        ax_lat.text(0.0, y, label, color=LBLC, fontsize=8, transform=ax_lat.transAxes)
        ax_lat.text(0.42, y, cv, color=BLUE, fontsize=8, transform=ax_lat.transAxes)
        ax_lat.text(0.72, y, yv, color=ORG,  fontsize=8, transform=ax_lat.transAxes)

# ── bottom row: example images ────────────────────────────────────────────────
example_img = Path("outputs/tmp/distance_examples.jpg")
if example_img.exists():
    ax_ex = fig.add_subplot(gs[1, :3])
    ax_ex.set_facecolor(DARK)
    ax_ex.axis("off")
    ax_ex.set_title("Dataset examples  (1024×640 ROI crops, green ellipse = GT label)",
                    color=WHITE, fontsize=9.5, fontweight="bold", pad=6)
    img = cv2.imread(str(example_img))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax_ex.imshow(img, aspect="auto")

# ── bottom-right: key findings ────────────────────────────────────────────────
ax_find = fig.add_subplot(gs[1, 3])
ax_find.set_facecolor(DARK)
ax_find.axis("off")
ax_find.set_title("Key Findings", color=WHITE, fontsize=9.5, fontweight="bold", pad=6)

findings = [
    (GREEN, "CNN 8–10m: TP 99%,  L2 2.9px ✓"),
    (ORG,   "CNN 5–8m:  TP 80%,  L2 4.3px △"),
    (RED,   "CNN 3.8–5m: TP 61%, L2 13px ✗"),
    ("",    ""),
    (RED,   "YOLO all ranges: F1=0.11  fail"),
    (ORG,   "YOLO FP=1061 (False Positives on bg)"),
    (GRAY,  "YOLO No close-range training data"),
    ("",    ""),
    (BLUE,  "CNN fwd: 4.7ms (212 FPS)"),
    (ORG,   "YOLO fwd: 13ms  (76 FPS)"),
    (BLUE,  "CNN pipeline: 21ms (47 FPS)"),
]

for i, (color, text) in enumerate(findings):
    if not text:
        continue
    ax_find.text(0.04, 0.96 - i * 0.086, text,
                 color=color if color else LBLC,
                 fontsize=8, transform=ax_find.transAxes, va="top")

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
plt.close()
print(f"Saved: {OUT}")
