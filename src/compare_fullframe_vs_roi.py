#!/usr/bin/env python3
"""Compare full-frame 2048x1280 inference vs ROI-crop inference.

Metrics reported (all in full source image pixel coords):
  - predicted centre (cx, cy)
  - peak value
  - heatmap cell size (theoretical localisation limit before offset refinement)
  - offset-refined sub-cell shift
  - predicted ellipse (a, b) from size head
  - side-by-side overlay on a 1:1 crop around the marker
"""
from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np
import torch

from infer_cctag_heatmap import (
    IMAGENET_MEAN, IMAGENET_STD,
    decode_center_offset, decode_center_weighted,
    decode_size_at_peak,
    load_model, resolve_device,
)

CKPT   = Path("outputs/runs/experiment_sizehead/best.pt")
IMG    = Path("40m_example.png")
OUTDIR = Path("outputs/inference/fullframe_vs_roi")
THR    = 0.5

# ROI: 2×4 tile cell (row0, col0) = [0:1080, 0:1024]
ROI_X0, ROI_X1 = 0, 1024
ROI_Y0, ROI_Y1 = 0, 1080
# Full-frame resize target
FF_W, FF_H = 2048, 1280


def infer(model, rgb_tile, in_w, in_h, device):
    """Run model on rgb_tile resized to in_w×in_h.
    Returns (hm, offset, size, tensor_hw).
    """
    inp = cv2.resize(rgb_tile, (in_w, in_h), interpolation=cv2.INTER_AREA)
    t   = torch.from_numpy(inp.transpose(2, 0, 1)).float() / 255.0
    t   = ((t - IMAGENET_MEAN) / IMAGENET_STD).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(t)
    if isinstance(out, tuple):
        hm     = out[0][0, 0].cpu().numpy()
        offset = out[1][0].cpu().numpy() if out[1] is not None else None
        size   = out[2][0].cpu().numpy() if len(out) > 2 and out[2] is not None else None
    else:
        hm, offset, size = out[0, 0].cpu().numpy(), None, None
    return hm, offset, size, (in_h, in_w)


def decode(hm, offset, size, in_hw, src_region_wh):
    """Decode centre and size in source-image coords.

    src_region_wh: (W, H) of the source region fed into the model
                   (full image for full-frame, ROI crop for ROI method).
    """
    hm_h, hm_w = hm.shape
    src_w, src_h = src_region_wh
    peak = float(hm.max())

    # heatmap cell size in source-image px
    cell_x = src_w / hm_w
    cell_y = src_h / hm_h

    if peak < THR:
        return None

    # --- raw argmax centre (heatmap grid) ---
    flat_idx = int(np.argmax(hm))
    py_hm, px_hm = divmod(flat_idx, hm_w)

    # --- offset-refined centre (still in heatmap coords) ---
    if offset is not None:
        c_hm = decode_center_offset(hm, offset, threshold=THR)
    else:
        c_hm = decode_center_weighted(hm, threshold=THR)

    if c_hm is None:
        return None

    cx_hm, cy_hm = c_hm                     # fractional heatmap coords
    dx_off = cx_hm - px_hm                  # sub-cell offset from offset head
    dy_off = cy_hm - py_hm

    # scale to source-image coords
    cx_src = cx_hm * cell_x
    cy_src = cy_hm * cell_y

    # ellipse in source px (size head predicts in model-input px → scale)
    ellipse_src = None
    if size is not None:
        ab_model = decode_size_at_peak(hm, size, threshold=THR)
        if ab_model:
            # size head was trained on 640×400 inputs; scale to source
            # model-input-px → source-px = src / model_input
            in_h, in_w = in_hw
            scale_x = src_w / in_w
            scale_y = src_h / in_h
            ellipse_src = (ab_model[0] * scale_x, ab_model[1] * scale_y)

    return {
        "peak":      peak,
        "cx_src":    cx_src,
        "cy_src":    cy_src,
        "cell_x":    cell_x,
        "cell_y":    cell_y,
        "offset_dx": dx_off * cell_x,       # offset in src px
        "offset_dy": dy_off * cell_y,
        "ellipse":   ellipse_src,
    }


def make_overlay(bgr_crop, hm, in_hw, src_region_wh, result,
                 display_w=640, display_h=400):
    """Blend heatmap on crop, draw centre+ellipse. All coords in crop space."""
    disp = cv2.resize(bgr_crop, (display_w, display_h))
    hm_u8    = (np.clip(hm, 0, 1) * 255).astype(np.uint8)
    hm_color = cv2.applyColorMap(
        cv2.resize(hm_u8, (display_w, display_h)), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(disp, 0.55, hm_color, 0.45, 0)

    if result is not None:
        src_w, src_h = src_region_wh
        scale_x = display_w / src_w
        scale_y = display_h / src_h
        dx = int(result["cx_src"] * scale_x)
        dy = int(result["cy_src"] * scale_y)
        if result["ellipse"]:
            a = max(1, int(result["ellipse"][0] * scale_x))
            b = max(1, int(result["ellipse"][1] * scale_y))
            cv2.ellipse(overlay, (dx, dy), (a, b), 0, 0, 360, (0, 255, 255), 2)
        cv2.circle(overlay, (dx, dy), 5, (0, 255, 0), -1)
        cv2.circle(overlay, (dx, dy), 7, (0, 0, 0), 2)
    return overlay


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    device = resolve_device(None)
    model, config = load_model(CKPT, device)

    bgr_src = cv2.imread(str(IMG), cv2.IMREAD_COLOR)
    rgb_src = cv2.cvtColor(bgr_src, cv2.COLOR_BGR2RGB)
    src_h, src_w = bgr_src.shape[:2]

    # ── Method A: full-frame 2048×1280 ──────────────────────────────────────
    hm_ff, off_ff, sz_ff, hw_ff = infer(model, rgb_src, FF_W, FF_H, device)
    res_ff = decode(hm_ff, off_ff, sz_ff, hw_ff, (src_w, src_h))

    # ── Method B: ROI crop → 640×400 ────────────────────────────────────────
    roi_rgb = rgb_src[ROI_Y0:ROI_Y1, ROI_X0:ROI_X1]
    mdl_w   = config.get("input_width",  640)
    mdl_h   = config.get("input_height", 400)
    hm_roi, off_roi, sz_roi, hw_roi = infer(model, roi_rgb, mdl_w, mdl_h, device)
    # ROI starts at (ROI_X0, ROI_Y0) in source → add offset
    res_roi_local = decode(hm_roi, off_roi, sz_roi, hw_roi,
                           (ROI_X1 - ROI_X0, ROI_Y1 - ROI_Y0))
    if res_roi_local is not None:
        res_roi = dict(res_roi_local)
        res_roi["cx_src"] += ROI_X0   # convert to full-image coords
        res_roi["cy_src"] += ROI_Y0
    else:
        res_roi = None

    # ── Print comparison table ───────────────────────────────────────────────
    print("=" * 60)
    print(f"  Full-frame 2048×1280  vs  ROI crop {ROI_X0}:{ROI_X1},{ROI_Y0}:{ROI_Y1}")
    print("=" * 60)
    header = f"{'Metric':<28} {'Full-frame':>14} {'ROI crop':>14}"
    print(header)
    print("-" * 60)

    def row(name, ff_val, roi_val, fmt=".1f"):
        fv = f"{ff_val:{fmt}}"  if ff_val is not None else "—"
        rv = f"{roi_val:{fmt}}" if roi_val is not None else "—"
        print(f"{name:<28} {fv:>14} {rv:>14}")

    ff  = res_ff
    roi = res_roi
    row("peak",                ff["peak"]      if ff  else None,
                               roi["peak"]     if roi else None, ".3f")
    row("centre x (src px)",   ff["cx_src"]    if ff  else None,
                               roi["cx_src"]   if roi else None, ".1f")
    row("centre y (src px)",   ff["cy_src"]    if ff  else None,
                               roi["cy_src"]   if roi else None, ".1f")
    row("heatmap cell x (px)", ff["cell_x"]    if ff  else None,
                               res_roi_local["cell_x"] if res_roi_local else None, ".2f")
    row("heatmap cell y (px)", ff["cell_y"]    if ff  else None,
                               res_roi_local["cell_y"] if res_roi_local else None, ".2f")
    row("offset shift x (px)", ff["offset_dx"] if ff  else None,
                               res_roi_local["offset_dx"] if res_roi_local else None, ".2f")
    row("offset shift y (px)", ff["offset_dy"] if ff  else None,
                               res_roi_local["offset_dy"] if res_roi_local else None, ".2f")

    if ff and ff["ellipse"]:
        row("ellipse a (src px)",  ff["ellipse"][0], res_roi_local["ellipse"][0] if (res_roi_local and res_roi_local["ellipse"]) else None, ".1f")
        row("ellipse b (src px)",  ff["ellipse"][1], res_roi_local["ellipse"][1] if (res_roi_local and res_roi_local["ellipse"]) else None, ".1f")

    if ff and roi:
        dist = ((ff["cx_src"] - roi["cx_src"])**2 +
                (ff["cy_src"] - roi["cy_src"])**2) ** 0.5
        print("-" * 60)
        print(f"{'centre distance (src px)':<28} {dist:>14.2f}")
        print()

        # theoretical precision without offset (1 cell error budget)
        print(f"{'Theoretical precision':<28} {'±'+str(round(ff['cell_x']/2,1))+'×'+str(round(ff['cell_y']/2,1))+' px':>14}"
              f" {'±'+str(round(res_roi_local['cell_x']/2,1))+'×'+str(round(res_roi_local['cell_y']/2,1))+' px':>14}")

    print("=" * 60)

    # ── Overlays ─────────────────────────────────────────────────────────────
    bgr_roi = cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2BGR)

    ov_ff = make_overlay(bgr_src, hm_ff, hw_ff, (src_w, src_h), res_ff)
    ov_roi = make_overlay(bgr_roi, hm_roi, hw_roi,
                          (ROI_X1-ROI_X0, ROI_Y1-ROI_Y0), res_roi_local)

    # ── zoomed marker region (same crop, both methods) ───────────────────────
    if ff and roi:
        mx = int((ff["cx_src"] + roi["cx_src"]) / 2)
        my = int((ff["cy_src"] + roi["cy_src"]) / 2)
    elif ff:
        mx, my = int(ff["cx_src"]), int(ff["cy_src"])
    else:
        mx, my = 882, 390

    ZR = 200  # zoom radius in source px
    zx0, zx1 = max(0, mx-ZR), min(src_w, mx+ZR)
    zy0, zy1 = max(0, my-ZR), min(src_h, my+ZR)

    def zoom_overlay(bgr_full, result_src, label, peak):
        crop = bgr_full[zy0:zy1, zx0:zx1].copy()
        crop = cv2.resize(crop, (400, 400))
        scale = 400 / (2*ZR)
        if result_src:
            dx = int((result_src["cx_src"] - zx0) * scale)
            dy = int((result_src["cy_src"] - zy0) * scale)
            if result_src["ellipse"]:
                a = max(1, int(result_src["ellipse"][0] * scale))
                b = max(1, int(result_src["ellipse"][1] * scale))
                cv2.ellipse(crop, (dx, dy), (a, b), 0, 0, 360, (0, 255, 255), 2)
            cv2.circle(crop, (dx, dy), 6, (0, 255, 0), -1)
            cv2.circle(crop, (dx, dy), 8, (0, 0, 0), 2)
            # draw cell size rectangle
            if "cell_x" in result_src:
                cx_ff = ff["cell_x"] if result_src is res_ff else res_roi_local["cell_x"]
                cy_ff = ff["cell_y"] if result_src is res_ff else res_roi_local["cell_y"]
                half_cx = int(cx_ff * scale / 2)
                half_cy = int(cy_ff * scale / 2)
                cv2.rectangle(crop,
                              (dx - half_cx, dy - half_cy),
                              (dx + half_cx, dy + half_cy),
                              (255, 200, 0), 1)
        det = "DETECT" if peak >= THR else "MISS"
        col = (50, 180, 50) if peak >= THR else (50, 50, 210)
        bar = np.full((44, 400, 3), 255, np.uint8)
        cv2.putText(bar, f"{label}  peak={peak:.3f}  {det}",
                    (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.58, col, 2)
        return np.vstack([bar, crop])

    res_ff_for_zoom = res_ff
    if res_ff_for_zoom: res_ff_for_zoom["cell_x"] = ff["cell_x"]; res_ff_for_zoom["cell_y"] = ff["cell_y"]
    pk_ff  = ff["peak"]  if ff  else 0.0
    pk_roi = res_roi_local["peak"] if res_roi_local else 0.0

    z_ff  = zoom_overlay(bgr_src, res_ff,       "Full-frame 2048x1280",   pk_ff)
    z_roi = zoom_overlay(bgr_roi,  res_roi_local,"ROI crop 1024x1080→640x400", pk_roi)

    # save individual overlays
    cv2.imwrite(str(OUTDIR / "overlay_fullframe.png"), ov_ff)
    cv2.imwrite(str(OUTDIR / "overlay_roi.png"), ov_roi)

    # combined: wide overlays side-by-side on top, zooms side-by-side below
    def label_bar(text, w):
        bar = np.full((36, w, 3), 255, np.uint8)
        cv2.putText(bar, text, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (30,30,30), 2)
        return bar

    panel_ff  = np.vstack([label_bar("Full-frame 2048x1280", ov_ff.shape[1]),  ov_ff])
    panel_roi = np.vstack([label_bar("ROI crop (2x4 tile, cell 0,0)", ov_roi.shape[1]), ov_roi])
    gap_h = np.full((panel_ff.shape[0], 12, 3), 255, np.uint8)
    row1  = np.hstack([panel_ff, gap_h, panel_roi])
    gap_v = np.full((16, row1.shape[1], 3), 255, np.uint8)
    gap_z = np.full((z_ff.shape[0], 12, 3), 255, np.uint8)
    row2 = np.hstack([z_ff, gap_z, z_roi])
    # pad row2 to match row1 width
    dw = row1.shape[1] - row2.shape[1]
    if dw > 0:
        row2 = np.hstack([row2, np.full((row2.shape[0], dw, 3), 255, np.uint8)])

    title = np.full((44, row1.shape[1], 3), 255, np.uint8)
    cv2.putText(title, "Full-frame 2048x1280  vs  ROI crop  —  centre comparison (zoom box = 1 heatmap cell)",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (30,30,30), 2)
    fig = np.vstack([title, row1, gap_v, row2])
    out = Path("outputs/inference/fullframe_vs_roi_comparison.png")
    cv2.imwrite(str(out), fig)
    print(f"figure saved: {out}")


if __name__ == "__main__":
    main()
