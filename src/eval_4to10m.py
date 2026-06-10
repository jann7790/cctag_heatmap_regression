#!/usr/bin/env python3
"""Per-image evaluation of CNN heatmap + YOLO on the 4-10m ROI subset.

Outputs:
  outputs/inference/eval_4to10m/results.csv   — per-image row
  outputs/inference/eval_4to10m/summary.txt   — aggregate table
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

K = 1720  # px·m calibration constant (50mm lens, 20cm CCTag)

# ─────────────────────────────────────────────────────── CNN heatmap ──────────

def load_cnn(checkpoint: Path, device: torch.device):
    from infer_cctag_heatmap import load_model
    return load_model(checkpoint, device)


def infer_cnn(model, config, img_bgr: np.ndarray, device: torch.device):
    from infer_cctag_heatmap import (
        IMAGENET_MEAN, IMAGENET_STD,
        decode_center_offset, decode_center_weighted,
        decode_size_at_peak,
    )
    in_w = config.get("input_width", 640)
    in_h = config.get("input_height", 400)
    inp = cv2.resize(img_bgr, (in_w, in_h), interpolation=cv2.INTER_AREA)
    inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(inp.transpose(2, 0, 1)).float() / 255.0
    t = ((t - IMAGENET_MEAN) / IMAGENET_STD).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(t)
    if isinstance(out, tuple):
        hm  = out[0][0, 0].cpu().numpy()
        off = out[1][0].cpu().numpy() if out[1] is not None else None
        sz  = out[2][0].cpu().numpy() if len(out) > 2 and out[2] is not None else None
    else:
        hm, off, sz = out[0, 0].cpu().numpy(), None, None

    peak = float(hm.max())
    if peak < 0.5:
        return None  # no detection

    # decode center back to image coords
    hm_h, hm_w = hm.shape
    stride_x = img_bgr.shape[1] / hm_w
    stride_y = img_bgr.shape[0] / hm_h
    # scale heatmap coords to in_w/in_h space first
    scale_x = img_bgr.shape[1] / in_w
    scale_y = img_bgr.shape[0] / in_h

    if off is not None:
        cx_hm, cy_hm = decode_center_offset(hm, off)
    else:
        cx_hm, cy_hm = decode_center_weighted(hm)

    cx = cx_hm * stride_x * scale_x
    cy = cy_hm * stride_y * scale_y

    pred_a = pred_b = None
    if sz is not None:
        pa, pb = decode_size_at_peak(hm, sz)
        # scale back to image coords
        pred_a = pa * stride_x * scale_x
        pred_b = pb * stride_y * scale_y

    return {"cx": cx, "cy": cy, "peak": peak, "pred_a": pred_a, "pred_b": pred_b}


# ─────────────────────────────────────────────────────── YOLO ──────────────────

def load_yolo(checkpoint: Path):
    from ultralytics import YOLO
    return YOLO(str(checkpoint))


def infer_yolo(model, img_bgr: np.ndarray, conf: float = 0.25, iou: float = 0.5):
    results = model.predict(img_bgr, imgsz=1024, conf=conf, iou=iou,
                            verbose=False, device=model.device)
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return None
    # Take highest confidence detection
    confs = boxes.conf.cpu().numpy()
    best = int(np.argmax(confs))
    xyxy = boxes.xyxy[best].cpu().numpy()
    cx = (xyxy[0] + xyxy[2]) / 2
    cy = (xyxy[1] + xyxy[3]) / 2
    w  = xyxy[2] - xyxy[0]
    h  = xyxy[3] - xyxy[1]
    return {"cx": cx, "cy": cy, "conf": float(confs[best]),
            "pred_a": w / 2, "pred_b": h / 2}


# ─────────────────────────────────────────────────────── main ──────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path,
                    default=Path("outputs/datasets/6f_labeled_roi_4to10m"))
    ap.add_argument("--cnn",  type=Path,
                    default=Path("outputs/runs/heatmap_1024/best.pt"))
    ap.add_argument("--yolo", type=Path,
                    default=Path("outputs/runs_yolo/cctag_det_n_30ep/weights/best.pt"))
    ap.add_argument("--output", type=Path,
                    default=Path("outputs/inference/eval_4to10m"))
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--conf",   type=float, default=0.25)
    ap.add_argument("--yolo_iou", type=float, default=0.5)
    ap.add_argument("--l2_match_thresh", type=float, default=50.0,
                    help="Max L2 distance (px) to count as TP")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.output.mkdir(parents=True, exist_ok=True)

    # Load labels
    rows = []
    with open(args.dataset / "labels.csv") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    pos_rows = [r for r in rows if r["is_negative"] == "0"]
    neg_rows = [r for r in rows if r["is_negative"] == "1"]
    print(f"Dataset: {len(pos_rows)} pos + {len(neg_rows)} neg")

    # Load source ellipse_a for distance calculation
    src_a: dict[str, float] = {}
    src_labels = Path("outputs/datasets/6f_labeled/labels.csv")
    if src_labels.exists():
        with open(src_labels) as f:
            for r in csv.DictReader(f):
                if r["is_negative"] == "0":
                    src_a[r["filename"]] = float(r["ellipse_a"])

    print(f"Loading CNN: {args.cnn}")
    cnn_model, cnn_config = load_cnn(args.cnn, device)
    cnn_model.eval()

    print(f"Loading YOLO: {args.yolo}")
    yolo_model = load_yolo(args.yolo)
    yolo_model.to(device)

    out_rows = []
    img_dir = args.dataset / "images"

    all_rows = pos_rows + neg_rows
    n = len(all_rows)

    for i, r in enumerate(all_rows):
        if i % 500 == 0:
            print(f"  {i}/{n} ...")

        fn = r["filename"]
        img_path = img_dir / f"{fn}.png"
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        is_pos = r["is_negative"] == "0"
        gt_cx = float(r["ellipse_cx"]) if is_pos else None
        gt_cy = float(r["ellipse_cy"]) if is_pos else None
        gt_a  = float(r["ellipse_a"])  if is_pos else None

        # Distance estimate from source ellipse_a
        import re
        m = re.match(r"(frame_\d{8}_\d{6})", fn)
        frame = m.group(1) if m else fn
        src_ea = src_a.get(frame, gt_a)
        dist_m = K / src_ea if src_ea else None

        # CNN
        cnn_pred = infer_cnn(cnn_model, cnn_config, img, device)
        # YOLO
        yolo_pred = infer_yolo(yolo_model, img, conf=args.conf, iou=args.yolo_iou)

        def classify(pred, gt_cx, gt_cy, is_pos):
            detected = pred is not None
            if is_pos:
                if detected:
                    l2 = np.sqrt((pred["cx"] - gt_cx)**2 + (pred["cy"] - gt_cy)**2)
                    status = "TP" if l2 <= args.l2_match_thresh else "FP+FN"
                    return status, l2
                else:
                    return "FN", None
            else:
                return ("FP" if detected else "TN"), None

        cnn_status, cnn_l2 = classify(cnn_pred, gt_cx, gt_cy, is_pos)
        yolo_status, yolo_l2 = classify(yolo_pred, gt_cx, gt_cy, is_pos)

        out_rows.append({
            "filename": fn,
            "is_positive": int(is_pos),
            "dist_m": f"{dist_m:.1f}" if dist_m else "",
            "gt_a_px": f"{gt_a:.1f}" if gt_a else "",
            "gt_cx": f"{gt_cx:.1f}" if gt_cx else "",
            "gt_cy": f"{gt_cy:.1f}" if gt_cy else "",
            # CNN
            "cnn_status": cnn_status,
            "cnn_cx": f"{cnn_pred['cx']:.1f}" if cnn_pred else "",
            "cnn_cy": f"{cnn_pred['cy']:.1f}" if cnn_pred else "",
            "cnn_l2_px": f"{cnn_l2:.2f}" if cnn_l2 is not None else "",
            "cnn_peak": f"{cnn_pred['peak']:.3f}" if cnn_pred else "",
            "cnn_pred_a": f"{cnn_pred['pred_a']:.1f}" if cnn_pred and cnn_pred['pred_a'] else "",
            # YOLO
            "yolo_status": yolo_status,
            "yolo_cx": f"{yolo_pred['cx']:.1f}" if yolo_pred else "",
            "yolo_cy": f"{yolo_pred['cy']:.1f}" if yolo_pred else "",
            "yolo_l2_px": f"{yolo_l2:.2f}" if yolo_l2 is not None else "",
            "yolo_conf": f"{yolo_pred['conf']:.3f}" if yolo_pred else "",
            "yolo_pred_a": f"{yolo_pred['pred_a']:.1f}" if yolo_pred else "",
        })

    # Write CSV
    csv_path = args.output / "results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"\nSaved per-image CSV: {csv_path}")

    # Summary
    def summarize(rows, model_key):
        pos = [r for r in rows if r["is_positive"] == 1]
        neg = [r for r in rows if r["is_positive"] == 0]
        tp = sum(1 for r in pos if r[f"{model_key}_status"] == "TP")
        fp_fn = sum(1 for r in pos if r[f"{model_key}_status"] == "FP+FN")
        fn = sum(1 for r in pos if r[f"{model_key}_status"] == "FN")
        fp = sum(1 for r in neg if r[f"{model_key}_status"] == "FP")
        tn = sum(1 for r in neg if r[f"{model_key}_status"] == "TN")
        l2s = [float(r[f"{model_key}_l2_px"]) for r in pos
               if r[f"{model_key}_l2_px"]]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec  = tp / (tp + fn + fp_fn) if (tp + fn + fp_fn) > 0 else 0
        f1   = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0
        return {
            "TP": tp, "FP": fp, "FN": fn + fp_fn, "TN": tn,
            "Precision": prec, "Recall": rec, "F1": f1,
            "L2_mean": np.mean(l2s) if l2s else float("nan"),
            "L2_median": np.median(l2s) if l2s else float("nan"),
            "L2_p90": np.percentile(l2s, 90) if l2s else float("nan"),
        }

    print("\n" + "="*60)
    print(f"SUMMARY — 4–10m subset  (pos={len([r for r in out_rows if r['is_positive']==1])}, "
          f"neg={len([r for r in out_rows if r['is_positive']==0])})")
    print("="*60)
    print(f"{'指標':<18} {'CNN heatmap_1024':>18} {'YOLO cctag_det_n':>18}")
    print("-"*60)
    cs = summarize(out_rows, "cnn")
    ys = summarize(out_rows, "yolo")
    for key in ["TP", "FP", "FN", "TN", "Precision", "Recall", "F1",
                "L2_mean", "L2_median", "L2_p90"]:
        cv = cs[key]
        yv = ys[key]
        if isinstance(cv, float):
            print(f"{key:<18} {cv:>18.4f} {yv:>18.4f}")
        else:
            print(f"{key:<18} {cv:>18} {yv:>18}")

    # Per-distance-bin breakdown
    print("\n--- CNN L2 error by distance ---")
    print(f"{'距離':<10} {'n':>5} {'mean':>8} {'median':>8} {'p90':>8} {'TP%':>7}")
    bins = [(3.8,5,"3.8–5m"),(5,8,"5–8m"),(8,10,"8–10m")]
    for d_lo, d_hi, lbl in bins:
        sub = [r for r in out_rows if r["is_positive"]==1 and r["dist_m"]
               and d_lo <= float(r["dist_m"]) < d_hi]
        if not sub: continue
        l2s = [float(r["cnn_l2_px"]) for r in sub if r["cnn_l2_px"]]
        tp_n = sum(1 for r in sub if r["cnn_status"]=="TP")
        print(f"{lbl:<10} {len(sub):>5} {np.mean(l2s) if l2s else 0:>8.2f} "
              f"{np.median(l2s) if l2s else 0:>8.2f} "
              f"{np.percentile(l2s,90) if l2s else 0:>8.2f} "
              f"{tp_n/len(sub)*100:>6.1f}%")

    print("\n--- YOLO by distance ---")
    print(f"{'距離':<10} {'n':>5} {'L2 mean':>8} {'TP%':>7}")
    for d_lo, d_hi, lbl in bins:
        sub = [r for r in out_rows if r["is_positive"]==1 and r["dist_m"]
               and d_lo <= float(r["dist_m"]) < d_hi]
        if not sub: continue
        l2s = [float(r["yolo_l2_px"]) for r in sub if r["yolo_l2_px"]]
        tp_n = sum(1 for r in sub if r["yolo_status"]=="TP")
        print(f"{lbl:<10} {len(sub):>5} {np.mean(l2s) if l2s else 0:>8.2f} "
              f"{tp_n/len(sub)*100:>6.1f}%")

    txt_path = args.output / "summary.txt"
    txt_path.write_text("See stdout. CSV at results.csv")
    print(f"\nDone. CSV: {csv_path}")


if __name__ == "__main__":
    main()
