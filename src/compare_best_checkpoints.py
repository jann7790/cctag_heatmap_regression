#!/usr/bin/env python3
"""Fair, accuracy-only comparison of every recursively discovered ``best.pt``.

The benchmark fixes the peak threshold at 0.5 and never applies a sharpness
filter.  Checkpoints retain their own architecture, input/heatmap size, decoder
alignment, and optional offset head through the production inference loader.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import multiprocessing as mp
import queue
import sys
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from infer_cctag_heatmap import (  # noqa: E402
    IMAGENET_MEAN,
    IMAGENET_STD,
    decode_center,
    decode_center_offset,
    decode_center_softargmax,
    decode_center_subpixel,
    decode_center_weighted,
    load_model,
)


ROOT = Path(__file__).resolve().parent.parent
BIN_SPECS = (
    ("70_90", 70.0, 90.0),
    ("90_110", 90.0, 110.0),
    ("110_130", 110.0, 130.0),
    ("130_150", 130.0, 150.0),
)
POSITIVE_SCENARIOS = ("clean", "degraded", "mixed_occlusion", "tile_boundary")
PER_IMAGE_FIELDS = (
    "model", "dataset", "scenario", "size_bin", "sample_id", "image_path",
    "is_negative", "diameter_px", "gt_center_x", "gt_center_y", "detected",
    "peak", "pred_center_x", "pred_center_y", "l2_px", "success_at_5px",
    "success_at_10px",
)


def diameter_bin(diameter: float) -> str | None:
    for index, (name, low, high) in enumerate(BIN_SPECS):
        if low <= diameter < high or (index == len(BIN_SPECS) - 1 and diameter == high):
            return name
    return None


def discover_checkpoints(runs_dir: Path) -> list[tuple[str, Path]]:
    """Recursively find best.pt files and name them by parent-relative path."""
    runs_dir = runs_dir.resolve()
    found = []
    for checkpoint in sorted(runs_dir.rglob("best.pt")):
        if checkpoint.is_file():
            found.append((checkpoint.parent.relative_to(runs_dir).as_posix(), checkpoint.resolve()))
    return found


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_negative(row: dict[str, Any]) -> bool:
    return str(row.get("is_negative", "0")).strip().lower() in {"1", "true", "yes"}


def _resolve_image(dataset_dir: Path, stem: str) -> Path:
    for suffix in (".png", ".jpg", ".jpeg", ".bmp", ".tiff"):
        candidate = dataset_dir / "images" / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"Missing holdout image for {dataset_dir}/{stem}")


def load_synthetic_manifest(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    required = {"sample_id", "scenario", "size_bin", "image_path", "center_x", "center_y", "is_negative"}
    missing = required - set(rows[0] if rows else ())
    if missing:
        raise ValueError(f"Synthetic manifest missing fields: {sorted(missing)}")
    records = []
    for row in rows:
        image = Path(row["image_path"])
        if not image.is_file():
            raise FileNotFoundError(image)
        records.append({
            "dataset": "synthetic", "scenario": row["scenario"], "size_bin": row["size_bin"],
            "sample_id": row["sample_id"], "image_path": str(image.resolve()),
            "is_negative": _is_negative(row), "center_x": float(row.get("center_x") or -1),
            "center_y": float(row.get("center_y") or -1),
            "diameter_px": float(row.get("diameter_px") or 0),
        })
    return records


def load_real_holdout(list_path: Path, diameter_min: float = 70.0, diameter_max: float = 150.0) -> list[dict[str, Any]]:
    entries: list[tuple[Path, str]] = []
    for line in list_path.read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        dataset_text, filename = line.split("\t", 1)
        dataset = Path(dataset_text)
        if not dataset.is_absolute():
            dataset = ROOT / dataset
        entries.append((dataset.resolve(), filename.strip()))
    label_maps: dict[Path, dict[str, dict[str, str]]] = {}
    for dataset in {item[0] for item in entries}:
        with (dataset / "labels.csv").open(newline="", encoding="utf-8") as handle:
            label_maps[dataset] = {row["filename"].strip(): row for row in csv.DictReader(handle)}
    records = []
    for dataset, filename in entries:
        row = label_maps[dataset][filename]
        negative = _is_negative(row)
        diameter = 2.0 * max(float(row.get("ellipse_a") or 0), float(row.get("ellipse_b") or 0))
        if not negative and not diameter_min <= diameter <= diameter_max:
            continue
        records.append({
            "dataset": "real_holdout", "scenario": "real_holdout",
            "size_bin": "negative" if negative else (diameter_bin(diameter) or "out_of_range"),
            "sample_id": f"real/{dataset.name}/{filename}",
            "image_path": str(_resolve_image(dataset, filename)), "is_negative": negative,
            "center_x": float(row.get("center_x") or row.get("x") or -1),
            "center_y": float(row.get("center_y") or row.get("y") or -1),
            "diameter_px": diameter,
        })
    return records


def limit_records(records: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if limit <= 0:
        return records
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[(record["dataset"], record["scenario"], record["size_bin"])].append(record)
    return [item for key in sorted(groups) for item in groups[key][:limit]]


def _split_output(output: Any) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    if not isinstance(output, tuple):
        return output, None, None
    if len(output) == 2:
        return output[0], output[1], None
    return output[0], output[1], output[2]


def _decode(heatmap: np.ndarray, offset: np.ndarray | None, threshold: float, method: str) -> tuple[float, float] | None:
    if offset is not None:
        return decode_center_offset(heatmap, offset, threshold=threshold)
    if method == "argmax":
        return decode_center(heatmap, threshold=threshold)
    if method == "subpixel":
        return decode_center_subpixel(heatmap, threshold=threshold)
    if method == "softargmax":
        return decode_center_softargmax(heatmap, threshold=threshold)
    return decode_center_weighted(heatmap, threshold=threshold)


def _prepare_batch(records: list[dict[str, Any]], width: int, height: int) -> tuple[torch.Tensor, list[tuple[int, int]]]:
    tensors: list[torch.Tensor] = []
    original_sizes = []
    for record in records:
        image = cv2.imread(record["image_path"], cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Cannot read image {record['image_path']}")
        orig_h, orig_w = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(resized.transpose(2, 0, 1)).float().div_(255.0)
        tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
        tensors.append(tensor)
        original_sizes.append((orig_w, orig_h))
    return torch.stack(tensors), original_sizes


def evaluate_records(
    model: torch.nn.Module,
    config: dict[str, Any],
    records: list[dict[str, Any]],
    device: torch.device,
    threshold: float,
    batch_size: int,
    model_name: str,
) -> list[dict[str, Any]]:
    width = int(config.get("input_width", 640))
    height = int(config.get("input_height", 400))
    decode_method = str(config.get("decode_method", "weighted"))
    output_rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for start in range(0, len(records), batch_size):
            batch_records = records[start:start + batch_size]
            tensor, original_sizes = _prepare_batch(batch_records, width, height)
            heatmaps_t, offsets_t, _ = _split_output(model(tensor.to(device, non_blocking=True)))
            heatmaps = heatmaps_t[:, 0].detach().float().cpu().numpy()
            offsets = offsets_t.detach().float().cpu().numpy() if offsets_t is not None else None
            for index, (record, (orig_w, orig_h)) in enumerate(zip(batch_records, original_sizes)):
                heatmap = heatmaps[index]
                peak = float(heatmap.max())
                offset = offsets[index] if offsets is not None else None
                center = _decode(heatmap, offset, threshold, decode_method)
                pred_x = pred_y = l2 = None
                detected = center is not None
                if detected:
                    hm_h, hm_w = heatmap.shape
                    pred_x = float(center[0] * orig_w / hm_w)
                    pred_y = float(center[1] * orig_h / hm_h)
                    if not record["is_negative"]:
                        l2 = float(math.hypot(pred_x - record["center_x"], pred_y - record["center_y"]))
                success5 = bool(not record["is_negative"] and detected and l2 is not None and l2 <= 5.0)
                success10 = bool(not record["is_negative"] and detected and l2 is not None and l2 <= 10.0)
                output_rows.append({
                    "model": model_name, "dataset": record["dataset"], "scenario": record["scenario"],
                    "size_bin": record["size_bin"], "sample_id": record["sample_id"],
                    "image_path": record["image_path"], "is_negative": int(record["is_negative"]),
                    "diameter_px": record["diameter_px"], "gt_center_x": None if record["is_negative"] else record["center_x"],
                    "gt_center_y": None if record["is_negative"] else record["center_y"],
                    "detected": int(detected), "peak": peak, "pred_center_x": pred_x,
                    "pred_center_y": pred_y, "l2_px": l2, "success_at_5px": int(success5),
                    "success_at_10px": int(success10),
                })
    return output_rows


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def summarize_records(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    rows = list(rows)
    positives = [row for row in rows if not bool(int(row["is_negative"]))]
    negatives = [row for row in rows if bool(int(row["is_negative"]))]
    tp = sum(int(row["detected"]) for row in positives)
    fn = len(positives) - tp
    fp = sum(int(row["detected"]) for row in negatives)
    tn = len(negatives) - fp
    precision = safe_divide(tp, tp + fp)
    recall = safe_divide(tp, tp + fn)
    errors = np.asarray([float(row["l2_px"]) for row in positives if row.get("l2_px") is not None], dtype=np.float64)
    return {
        "num_images": len(rows), "num_positives": len(positives), "num_negatives": len(negatives),
        "tp": tp, "fn": fn, "fp": fp, "tn": tn, "detection_precision": precision,
        "detection_recall": recall, "detection_f1": safe_divide(2 * precision * recall, precision + recall),
        "negative_fpr": safe_divide(fp, len(negatives)),
        "success_at_5px": safe_divide(sum(int(row["success_at_5px"]) for row in positives), len(positives)),
        "success_at_10px": safe_divide(sum(int(row["success_at_10px"]) for row in positives), len(positives)),
        "l2_detected_count": int(errors.size), "l2_mean": float(errors.mean()) if errors.size else None,
        "l2_median": float(np.median(errors)) if errors.size else None,
        "l2_p90": float(np.percentile(errors, 90)) if errors.size else None,
    }


def build_model_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    synthetic = [row for row in rows if row["dataset"] == "synthetic"]
    real = [row for row in rows if row["dataset"] == "real_holdout"]
    cells: dict[str, Any] = {}
    scenario_summary: dict[str, Any] = {}
    bin_summary: dict[str, Any] = {}
    for scenario in POSITIVE_SCENARIOS:
        scenario_rows = [r for r in synthetic if r["scenario"] == scenario and not int(r["is_negative"])]
        scenario_summary[scenario] = summarize_records(scenario_rows)
        for bin_name, _, _ in BIN_SPECS:
            cell_rows = [r for r in scenario_rows if r["size_bin"] == bin_name]
            cells[f"{scenario}/{bin_name}"] = summarize_records(cell_rows)
    for bin_name, _, _ in BIN_SPECS:
        bin_summary[bin_name] = summarize_records(
            r for r in synthetic if r["size_bin"] == bin_name and not int(r["is_negative"])
        )
    negative_summary = {
        scenario: summarize_records(r for r in synthetic if r["scenario"] == scenario)
        for scenario in ("complex_low_light", "overexposure")
    }
    synthetic_overall = summarize_records(synthetic)
    macro_cells = [cell["success_at_5px"] for cell in cells.values() if cell["num_positives"]]
    synthetic_overall["macro_success_at_5px"] = float(np.mean(macro_cells)) if macro_cells else 0.0
    return {
        "synthetic": synthetic_overall, "synthetic_cells": cells,
        "synthetic_by_scenario": scenario_summary, "synthetic_by_size_bin": bin_summary,
        "synthetic_negatives": negative_summary, "real_holdout": summarize_records(real),
    }


def _write_per_image(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PER_IMAGE_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_checkpoint(task: dict[str, Any], options: dict[str, Any], records: list[dict[str, Any]], device_name: str) -> dict[str, Any]:
    model_name = task["model_name"]
    result: dict[str, Any] = {**task, "status": "load_failed", "device": device_name}
    try:
        device = torch.device(device_name)
        if device.type == "cuda":
            torch.cuda.set_device(device)
        model, config = load_model(Path(task["checkpoint"]), device)
        result["config"] = {
            "backbone": config.get("backbone", "efficientnet_b0"),
            "input_width": int(config.get("input_width", 640)),
            "input_height": int(config.get("input_height", 400)),
            "heatmap_width": int(config.get("heatmap_width", 160)),
            "heatmap_height": int(config.get("heatmap_height", 100)),
            "use_offset_head": bool(config.get("use_offset_head", False)),
            "decode_method": "offset" if config.get("use_offset_head", False) else config.get("decode_method", "weighted"),
            "align_corners": bool(config.get("align_corners", False)),
        }
        rows = evaluate_records(model, config, records, device, options["threshold"], options["batch_size"], model_name)
        model_dir = Path(options["output"]) / "models" / model_name
        _write_per_image(model_dir / "per_image.csv", rows)
        summary = build_model_summary(rows)
        result.update({"status": "ok", "metrics": summary, "num_evaluated": len(rows)})
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()
        model_dir = Path(options["output"]) / "models" / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def _worker(device: str, tasks: list[dict[str, Any]], options: dict[str, Any], output_queue: Any) -> None:
    try:
        synthetic = load_synthetic_manifest(Path(options["synthetic_manifest"]))
        real = load_real_holdout(Path(options["real_holdout_list"]))
        records = limit_records(synthetic + real, options["limit_per_group"])
        for task in tasks:
            output_queue.put(("result", evaluate_checkpoint(task, options, records, device)))
    except Exception:
        output_queue.put(("worker_error", {"device": device, "traceback": traceback.format_exc()}))


def _flat_summary(result: dict[str, Any], rank: int | None) -> dict[str, Any]:
    base = {
        "rank": rank, "model": result["model_name"], "status": result["status"],
        "checkpoint": result["checkpoint"], "sha256": result["sha256"],
        "duplicate_of": result.get("duplicate_of"), "error": result.get("error"),
    }
    if result["status"] != "ok":
        return base
    synth = result["metrics"]["synthetic"]
    real = result["metrics"]["real_holdout"]
    base.update({
        "macro_success_at_5px": synth["macro_success_at_5px"],
        "synthetic_recall": synth["detection_recall"], "synthetic_f1": synth["detection_f1"],
        "synthetic_success_at_10px": synth["success_at_10px"], "synthetic_l2_mean": synth["l2_mean"],
        "synthetic_l2_median": synth["l2_median"], "synthetic_l2_p90": synth["l2_p90"],
        "negative_fpr": synth["negative_fpr"], "real_recall": real["detection_recall"],
        "real_f1": real["detection_f1"], "real_success_at_5px": real["success_at_5px"],
        "real_success_at_10px": real["success_at_10px"], "real_l2_mean": real["l2_mean"],
        "real_l2_median": real["l2_median"], "real_l2_p90": real["l2_p90"],
        "real_negative_fpr": real["negative_fpr"],
    })
    return base


def _rank_key(result: dict[str, Any]) -> tuple[float, float, float, float, str]:
    metrics = result["metrics"]["synthetic"]
    p90 = metrics["l2_p90"] if metrics["l2_p90"] is not None else math.inf
    return (-metrics["macro_success_at_5px"], -metrics["detection_recall"], p90, metrics["negative_fpr"], result["model_name"])


def write_reports(output: Path, results: list[dict[str, Any]], metadata: dict[str, Any]) -> None:
    successful = sorted((r for r in results if r["status"] == "ok"), key=_rank_key)
    ranks = {result["model_name"]: index + 1 for index, result in enumerate(successful)}
    ordered = successful + sorted((r for r in results if r["status"] != "ok"), key=lambda r: r["model_name"])
    flat = [_flat_summary(result, ranks.get(result["model_name"])) for result in ordered]
    fields = list(dict.fromkeys(key for row in flat for key in row))
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(flat)

    breakdown_fields = ("model", "section", "group", "num_images", "num_positives", "num_negatives",
                        "detection_recall", "detection_f1", "negative_fpr", "success_at_5px",
                        "success_at_10px", "l2_mean", "l2_median", "l2_p90")
    with (output / "breakdown.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=breakdown_fields); writer.writeheader()
        for result in successful:
            metrics = result["metrics"]
            sections = (("synthetic_cell", metrics["synthetic_cells"]),
                        ("synthetic_scenario", metrics["synthetic_by_scenario"]),
                        ("synthetic_size_bin", metrics["synthetic_by_size_bin"]),
                        ("synthetic_negative", metrics["synthetic_negatives"]),
                        ("real_holdout", {"all": metrics["real_holdout"]}))
            for section, groups in sections:
                for group, values in groups.items():
                    writer.writerow({"model": result["model_name"], "section": section, "group": group,
                                     **{key: values.get(key) for key in breakdown_fields[3:]}})

    report = [
        "# CCTag 70–150 px `best.pt` comparison", "",
        f"- Checkpoints discovered: {metadata['discovered_checkpoints']} (expected {metadata['expected_checkpoints']})",
        f"- Successfully evaluated: {len(successful)}; load/evaluation failures: {len(results) - len(successful)}",
        f"- Threshold: {metadata['threshold']}; sharpness filter: disabled",
        f"- Synthetic: {metadata['synthetic_positives']} positives + {metadata['synthetic_negatives']} negatives",
        f"- Real holdout: {metadata['real_positives']} positives + {metadata['real_negatives']} negatives",
        "- Ranking: macro Success@5px over the 16 scenario×size cells; tie-break recall, L2 p90, negative FPR.", "",
        "## Primary synthetic leaderboard", "",
        "| Rank | Model | Macro S@5 | Recall | S@10 | L2 mean/median/p90 | Neg FPR |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in flat:
        if row["status"] != "ok":
            continue
        report.append(
            f"| {row['rank']} | `{row['model']}` | {row['macro_success_at_5px']:.4f} | "
            f"{row['synthetic_recall']:.4f} | {row['synthetic_success_at_10px']:.4f} | "
            f"{row['synthetic_l2_mean']:.2f}/{row['synthetic_l2_median']:.2f}/{row['synthetic_l2_p90']:.2f} | {row['negative_fpr']:.4f} |"
        )
    report += ["", "## Real holdout (separate; not part of ranking)", "",
               "| Model | Recall | F1 | S@5 | S@10 | L2 mean/median/p90 | Neg FPR |",
               "|---|---:|---:|---:|---:|---:|---:|"]
    for row in flat:
        if row["status"] != "ok":
            continue
        report.append(
            f"| `{row['model']}` | {row['real_recall']:.4f} | {row['real_f1']:.4f} | "
            f"{row['real_success_at_5px']:.4f} | {row['real_success_at_10px']:.4f} | "
            f"{row['real_l2_mean']:.2f}/{row['real_l2_median']:.2f}/{row['real_l2_p90']:.2f} | {row['real_negative_fpr']:.4f} |"
        )
    report += ["", "Real positives are concentrated in the upper end of the requested range (observed diameter range is recorded in `run_metadata.json`).", "",
               "## Failures and duplicate weights", ""]
    failures = [r for r in results if r["status"] != "ok"]
    report.extend([f"- `{r['model_name']}`: {r.get('error', 'worker did not return a result')}" for r in failures] or ["- No load/evaluation failures."])
    duplicates = [r for r in results if r.get("duplicate_of")]
    report.extend([f"- Duplicate SHA-256: `{r['model_name']}` = `{r['duplicate_of']}`" for r in duplicates] or ["- No duplicate checkpoint hashes."])
    report += ["", "Full scenario×size, scenario, size-bin, and negative breakdowns are in `breakdown.csv`; per-image rows and checkpoint configs are under `models/`."]
    (output / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    (output / "run_metadata.json").write_text(json.dumps({**metadata, "results": results}, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "outputs/runs")
    parser.add_argument("--synthetic-manifest", type=Path, required=True)
    parser.add_argument("--real-holdout-list", type=Path, default=ROOT / "data_diagnosis/split_lists/real_holdout.txt")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/inference/cctag_70_150_comparison")
    parser.add_argument("--models", nargs="*", help="Exact relative model names (default: every discovered checkpoint)")
    parser.add_argument("--devices", nargs="+", default=None, help="One worker per device, e.g. cuda:0 cuda:1")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--limit-per-group", type=int, default=0, help="Smoke-test limit per dataset/scenario/bin group")
    parser.add_argument("--expected-checkpoints", type=int, default=15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.threshold != 0.5:
        raise SystemExit("This fair comparison requires --threshold 0.5")
    checkpoints = discover_checkpoints(args.runs_dir)
    if args.models is not None:
        requested = set(args.models)
        checkpoints = [item for item in checkpoints if item[0] in requested]
        missing = requested - {item[0] for item in checkpoints}
        if missing:
            raise SystemExit(f"Requested models not found: {sorted(missing)}")
    if not checkpoints:
        raise SystemExit("No best.pt checkpoints found")
    args.output.mkdir(parents=True, exist_ok=True)
    print(f"Hashing {len(checkpoints)} checkpoints...", flush=True)
    first_by_hash: dict[str, str] = {}
    tasks = []
    for model_name, checkpoint in checkpoints:
        digest = sha256_file(checkpoint)
        duplicate_of = first_by_hash.get(digest)
        first_by_hash.setdefault(digest, model_name)
        tasks.append({"model_name": model_name, "checkpoint": str(checkpoint), "sha256": digest,
                      "duplicate_of": duplicate_of})

    synthetic = limit_records(load_synthetic_manifest(args.synthetic_manifest), args.limit_per_group)
    real = limit_records(load_real_holdout(args.real_holdout_list), args.limit_per_group)
    synthetic_pos = [r for r in synthetic if not r["is_negative"]]
    synthetic_neg = [r for r in synthetic if r["is_negative"]]
    real_pos = [r for r in real if not r["is_negative"]]
    real_neg = [r for r in real if r["is_negative"]]
    devices = args.devices or (["cuda:0"] if torch.cuda.is_available() else ["cpu"])
    options = {
        "synthetic_manifest": str(args.synthetic_manifest.resolve()),
        "real_holdout_list": str(args.real_holdout_list.resolve()), "output": str(args.output.resolve()),
        "threshold": args.threshold, "batch_size": args.batch_size, "limit_per_group": args.limit_per_group,
    }
    # Each process owns exactly one device and a stable slice of checkpoints.
    chunks = [tasks[index::len(devices)] for index in range(len(devices))]
    ctx = mp.get_context("spawn")
    output_queue = ctx.Queue()
    processes = [ctx.Process(target=_worker, args=(device, chunk, options, output_queue))
                 for device, chunk in zip(devices, chunks) if chunk]
    for process in processes:
        process.start()
    results = []
    worker_errors = []
    while len(results) < len(tasks) and any(process.is_alive() for process in processes):
        try:
            kind, payload = output_queue.get(timeout=2)
        except queue.Empty:
            continue
        if kind == "result":
            results.append(payload)
            print(f"[{len(results)}/{len(tasks)}] {payload['model_name']}: {payload['status']}", flush=True)
        else:
            worker_errors.append(payload)
    for process in processes:
        process.join()
    while True:
        try:
            kind, payload = output_queue.get_nowait()
        except queue.Empty:
            break
        if kind == "result":
            results.append(payload)
        else:
            worker_errors.append(payload)
    returned = {result["model_name"] for result in results}
    for task in tasks:
        if task["model_name"] not in returned:
            results.append({**task, "status": "load_failed", "error": "worker exited without a result"})

    real_diameters = [record["diameter_px"] for record in real_pos]
    metadata = {
        "runs_dir": str(args.runs_dir.resolve()), "synthetic_manifest": str(args.synthetic_manifest.resolve()),
        "real_holdout_list": str(args.real_holdout_list.resolve()), "threshold": args.threshold,
        "sharpness_filter": False, "discovered_checkpoints": len(checkpoints),
        "expected_checkpoints": args.expected_checkpoints,
        "checkpoint_count_matches_expectation": len(checkpoints) == args.expected_checkpoints,
        "all_discovered_successful": all(r["status"] == "ok" for r in results),
        "synthetic_positives": len(synthetic_pos), "synthetic_negatives": len(synthetic_neg),
        "real_positives": len(real_pos), "real_negatives": len(real_neg),
        "real_positive_diameter_min": min(real_diameters) if real_diameters else None,
        "real_positive_diameter_max": max(real_diameters) if real_diameters else None,
        "devices": devices, "batch_size": args.batch_size, "limit_per_group": args.limit_per_group,
        "worker_errors": worker_errors,
    }
    write_reports(args.output, results, metadata)
    print(f"Reports written to {args.output}")


if __name__ == "__main__":
    main()
