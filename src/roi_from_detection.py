#!/usr/bin/env python3
"""
Two-stage inference:
  Stage 1 – full-frame at 2048×1280 → centre (cx,cy) + ellipse (a,b)
  Stage 2 – crop ROI = centre ± 1.5 × max(a,b) → run infer again

Output: 3-panel overlay on the ROI crop (original / heatmap / blend),
        displayed at high resolution so the marker is easy to inspect visually.
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
OUTDIR = Path("outputs/inference/roi_from_detection")
THR    = 0.5
DISPLAY_LONG_EDGE = 960   # how big to display the ROI panels


def infer_tensor(model, rgb, in_w, in_h, device):
    inp = cv2.resize(rgb, (in_w, in_h), interpolation=cv2.INTER_AREA)
    t   = torch.from_numpy(inp.transpose(2, 0, 1)).float() / 255.0
    t   = ((t - IMAGENET_MEAN) / IMAGENET_STD).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(t)
    if isinstance(out, tuple):
        hm  = out[0][0, 0].cpu().numpy()
        off = out[1][0].cpu().numpy() if out[1] is not None else None
        sz  = out[2][0].cpu().numpy() if len(out) > 2 and out[2] is not None else None
    else:
        hm, off, sz = out[0, 0].cpu().numpy(), None, None
    return hm, off, sz


def decode_result(hm, off, sz, in_w, in_h, region_w, region_h):
    """Returns dict with centre and ellipse in region (crop) pixel coords."""
    hm_h, hm_w = hm.shape
    peak = float(hm.max())
    if peak < THR:
        return {"peak": peak, "cx": None, "cy": None, "a": None, "b": None}

    if off is not None:
        c = decode_center_offset(hm, off, threshold=THR)
    else:
        c = decode_center_weighted(hm, threshold=THR)
    if c is None:
        return {"peak": peak, "cx": None, "cy": None, "a": None, "b": None}

    cx_region = c[0] * (region_w / hm_w)
    cy_region = c[1] * (region_h / hm_h)

    a = b = None
    if sz is not None:
        ab = decode_size_at_peak(hm, sz, threshold=THR)
        if ab:
            a = ab[0] * (region_w / in_w)
            b = ab[1] * (region_h / in_h)

    return {"peak": peak, "cx": cx_region, "cy": cy_region, "a": a, "b": b}


def brighten(bgr, percentile=99):
    """Stretch contrast: top-percentile pixel → 255."""
    hi = np.percentile(bgr, percentile)
    if hi < 1:
        hi = 1
    return np.clip(bgr.astype(np.float32) * (255.0 / hi), 0, 255).astype(np.uint8)


def make_overlay_panels(bgr_crop, hm, res, region_w, region_h,
                        display_w, display_h):
    """Return [original, heatmap, blend] each at display_w×display_h."""
    hm_h, hm_w = hm.shape

    # ── panel 1: brightened original ──
    bright = brighten(bgr_crop)
    p1 = cv2.resize(bright, (display_w, display_h), interpolation=cv2.INTER_LINEAR)

    # ── panel 2: heatmap (hot region only, rest semi-transparent) ──
    hm_disp = cv2.resize(hm, (display_w, display_h), interpolation=cv2.INTER_LINEAR)
    hm_u8   = (np.clip(hm_disp, 0, 1) * 255).astype(np.uint8)
    hm_col  = cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET)
    # darken low-activation areas so hot spots stand out
    alpha_map = np.clip(hm_disp * 4, 0, 1)[..., np.newaxis]   # boosts faint signal
    grey      = cv2.resize(bright, (display_w, display_h))
    grey3     = cv2.cvtColor(cv2.cvtColor(grey, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)
    p2 = (grey3 * (1 - alpha_map) * 0.4 + hm_col * alpha_map).astype(np.uint8)

    # ── panel 3: blend original + heatmap + annotations ──
    p3 = cv2.addWeighted(p1, 0.55, hm_col, 0.45, 0)

    # draw ellipse + centre on p3 (and p1 for reference)
    if res["cx"] is not None:
        sx = display_w / region_w
        sy = display_h / region_h
        dx, dy = int(res["cx"] * sx), int(res["cy"] * sy)
        if res["a"] and res["b"]:
            da = max(1, int(res["a"] * sx))
            db = max(1, int(res["b"] * sy))
            cv2.ellipse(p3, (dx, dy), (da, db), 0, 0, 360, (0, 255, 255), 3)
            cv2.ellipse(p1, (dx, dy), (da, db), 0, 0, 360, (0, 255, 255), 2)
        cv2.circle(p3, (dx, dy), 7, (0, 255, 0), -1)
        cv2.circle(p3, (dx, dy), 9, (0,   0,   0), 2)
        cv2.circle(p1, (dx, dy), 7, (0, 255, 0), -1)
        cv2.circle(p1, (dx, dy), 9, (0,   0,   0), 2)

    return p1, p2, p3


def add_label(panel, text, peak=None, detected=None):
    h, w = panel.shape[:2]
    bar = np.full((44, w, 3), 255, np.uint8)
    col = (50, 180, 50) if detected else (50, 50, 200) if detected is not None else (40, 40, 40)
    cv2.putText(bar, text, (8, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.70, col, 2, cv2.LINE_AA)
    if peak is not None:
        pstr = f"peak={peak:.3f}  {'DETECT' if detected else 'MISS'}"
        cv2.putText(bar, pstr, (w - 270, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, 2, cv2.LINE_AA)
    return np.vstack([bar, panel])


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    device = resolve_device(None)
    model, config = load_model(CKPT, device)
    mdl_w = config.get("input_width",  640)
    mdl_h = config.get("input_height", 400)

    bgr_src = cv2.imread(str(IMG), cv2.IMREAD_COLOR)
    rgb_src = cv2.cvtColor(bgr_src, cv2.COLOR_BGR2RGB)
    src_h, src_w = bgr_src.shape[:2]

    # ── Stage 1: full-frame 2048×1280 ───────────────────────────────────────
    FF_W, FF_H = 2048, 1280
    hm1, off1, sz1 = infer_tensor(model, rgb_src, FF_W, FF_H, device)
    r1 = decode_result(hm1, off1, sz1, FF_W, FF_H, src_w, src_h)
    print(f"Stage 1 (full-frame {FF_W}×{FF_H})")
    print(f"  peak={r1['peak']:.3f}  centre=({r1['cx']:.1f},{r1['cy']:.1f})"
          f"  ellipse a={r1['a']:.1f}  b={r1['b']:.1f}")

    if r1["cx"] is None:
        print("Stage 1 no detection – cannot define ROI"); return

    # ── Define ROI: centre ± 1.5 × max(a,b) ────────────────────────────────
    pad = 1.5 * max(r1["a"], r1["b"])
    cx, cy = r1["cx"], r1["cy"]
    x0 = max(0, int(cx - pad));  x1 = min(src_w, int(cx + pad))
    y0 = max(0, int(cy - pad));  y1 = min(src_h, int(cy + pad))
    roi_w, roi_h = x1 - x0, y1 - y0
    print(f"\nROI: ({x0},{y0})→({x1},{y1})  size={roi_w}×{roi_h}  "
          f"pad={pad:.0f}px (1.5×max({r1['a']:.0f},{r1['b']:.0f}))")

    bgr_roi = bgr_src[y0:y1, x0:x1]
    rgb_roi = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2RGB)

    # ── Stage 2: infer on ROI ───────────────────────────────────────────────
    hm2, off2, sz2 = infer_tensor(model, rgb_roi, mdl_w, mdl_h, device)
    r2 = decode_result(hm2, off2, sz2, mdl_w, mdl_h, roi_w, roi_h)
    print(f"\nStage 2 (ROI {roi_w}×{roi_h} → model {mdl_w}×{mdl_h})")
    if r2["cx"] is not None:
        # convert to source coords for reporting
        print(f"  peak={r2['peak']:.3f}  centre in ROI=({r2['cx']:.1f},{r2['cy']:.1f})"
              f"  →source=({r2['cx']+x0:.1f},{r2['cy']+y0:.1f})"
              f"  ellipse a={r2['a']:.1f}  b={r2['b']:.1f}")
    else:
        print(f"  peak={r2['peak']:.3f}  MISS (marker too large for model at this crop size)")

    # ── Compute display size (keep aspect, long edge = DISPLAY_LONG_EDGE) ───
    scale    = DISPLAY_LONG_EDGE / max(roi_w, roi_h)
    disp_w   = int(roi_w * scale)
    disp_h   = int(roi_h * scale)

    # ── Stage 1 heatmap crop for the same ROI ─────────────────────────────────
    hm1_h, hm1_w = hm1.shape
    hm1_x0 = int(x0 * hm1_w / src_w)
    hm1_x1 = int(x1 * hm1_w / src_w)
    hm1_y0 = int(y0 * hm1_h / src_h)
    hm1_y1 = int(y1 * hm1_h / src_h)
    hm1_roi = hm1[hm1_y0:hm1_y1, hm1_x0:hm1_x1]
    r1_roi = {"peak": r1["peak"],
              "cx": r1["cx"] - x0, "cy": r1["cy"] - y0,
              "a": r1["a"], "b": r1["b"]}

    p1, p2, p3_s2 = make_overlay_panels(bgr_roi, hm2, r2, roi_w, roi_h, disp_w, disp_h)
    _, _, p3_s1 = make_overlay_panels(bgr_roi, hm1_roi, r1_roi, roi_w, roi_h, disp_w, disp_h)

    # save individual panels
    cv2.imwrite(str(OUTDIR / "roi_original_bright.png"), p1)
    cv2.imwrite(str(OUTDIR / "roi_heatmap.png"),         p2)
    cv2.imwrite(str(OUTDIR / "roi_blend_stage2.png"),    p3_s2)
    cv2.imwrite(str(OUTDIR / "roi_blend_stage1.png"),    p3_s1)

    # ── Combined figure: 4 panels ─────────────────────────────────────────────
    gap  = np.full((disp_h, 10, 3), 200, np.uint8)
    p1l  = add_label(p1, "Original (brightened)", None, None)
    p2l  = add_label(p2, "Heatmap (Stage 2)", None, None)
    p3s1l = add_label(p3_s1, "Stage 1 Blend + ellipse", r1["peak"], r1["cx"] is not None)
    p3s2l = add_label(p3_s2, "Stage 2 Blend + ellipse", r2["peak"], r2["cx"] is not None)
    gap2 = np.full((p1l.shape[0], 10, 3), 200, np.uint8)
    row  = np.hstack([p1l, gap2, p2l, gap2, p3s1l, gap2, p3s2l])

    # title bar
    info = (f"S1 peak={r1['peak']:.3f} {'DETECT' if r1['cx'] else 'MISS'}  |  "
            f"S2 peak={r2['peak']:.3f} {'DETECT' if r2['cx'] else 'MISS'}  |  "
            f"ROI={roi_w}×{roi_h}px  pad=1.5×{max(r1['a'],r1['b']):.0f}px")
    title = np.full((50, row.shape[1], 3), 255, np.uint8)
    cv2.putText(title, info,
                (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (30,30,30), 2, cv2.LINE_AA)

    fig = np.vstack([title, row])
    out = Path("outputs/inference/roi_detection_overlay.png")
    cv2.imwrite(str(out), fig)
    print(f"\nfigure saved: {out}  ({fig.shape[1]}×{fig.shape[0]}px)")


if __name__ == "__main__":
    main()
