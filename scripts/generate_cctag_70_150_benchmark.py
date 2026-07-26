#!/usr/bin/env python3
"""Build and validate the deterministic 70--150 px CCTag benchmark corpus.

The generated image data intentionally lives outside the repository by default.
Each leaf dataset is produced by ``generate_cctag_dataset.py`` and a single
manifest joins the leaves without copying the (large) PNG files.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
GENERATOR = ROOT / "src" / "generate_cctag_dataset.py"
BIN_SPECS = (
    ("70_90", 70.0, 90.0),
    ("90_110", 90.0, 110.0),
    ("110_130", 110.0, 130.0),
    ("130_150", 130.0, 150.0),
)
POSITIVE_SCENARIOS = ("clean", "degraded", "mixed_occlusion", "tile_boundary")
NEGATIVE_SCENARIOS = ("complex_low_light", "overexposure")
MANIFEST_FIELDS = (
    "sample_id", "dataset_kind", "scenario", "size_bin", "image_path",
    "heatmap_path", "center_x", "center_y", "ellipse_a", "ellipse_b",
    "diameter_px", "is_negative", "source_dataset", "source_filename",
)


def diameter_bin(diameter: float) -> str | None:
    """Return the canonical half-open bin (the final bin includes 150)."""
    for index, (name, low, high) in enumerate(BIN_SPECS):
        if low <= diameter < high or (index == len(BIN_SPECS) - 1 and diameter == high):
            return name
    return None


def _common_positive_args(count: int, output: Path, seed: int, low: float, high: float) -> list[str]:
    # A broad nominal marker range plus rejection sampling controls the actual,
    # post-transform ellipse diameter used by the benchmark.
    exclusive_high = high if high == 150.0 else high - 1e-4
    return [
        sys.executable, str(GENERATOR), "--num_images", str(count),
        "--output_dir", str(output), "--output_size", "1024x640",
        "--marker_style", "cctag_source", "--num_rings", "3",
        # Despite the historical CLI name, the generator treats marker_min/max
        # as an outer radius; the final unwarped diameter is approximately 2x.
        "--marker_min", str(max(1, int(math.ceil(low / 2.0)))),
        "--marker_max", str(max(1, int(math.floor(exclusive_high / 2.0)))),
        "--actual_diameter_min", str(low), "--actual_diameter_max", str(exclusive_high),
        "--actual_diameter_max_attempts", "600", "--heatmap_stride", "4",
        "--heatmap_sigma", "2.0", "--negative_ratio", "0",
        "--seed", str(seed),
    ]


def generation_jobs(root: Path, per_bin: int, negatives: int) -> list[tuple[Path, list[str]]]:
    jobs: list[tuple[Path, list[str]]] = []
    for bin_index, (bin_name, low, high) in enumerate(BIN_SPECS):
        seed_base = 701_500 + bin_index * 100
        for scenario_index, scenario in enumerate(POSITIVE_SCENARIOS):
            scenario_root = root / "datasets" / scenario / bin_name
            seed = seed_base + scenario_index * 10
            if scenario == "mixed_occlusion":
                hardware_count = per_bin // 2
                parts = (("hardware", hardware_count, "hardware"),
                         ("inner", per_bin - hardware_count, "inner_ring_center"))
                for part_index, (part, count, style) in enumerate(parts):
                    out = scenario_root / part
                    cmd = _common_positive_args(count, out, seed + part_index, low, high)
                    cmd += ["--occ_min", "0.35", "--occ_max", "0.75",
                            "--occlusion_style", style, "--partial_out_prob", "0"]
                    jobs.append((out, cmd))
                continue

            out = scenario_root / "samples"
            cmd = _common_positive_args(per_bin, out, seed, low, high)
            if scenario == "clean":
                cmd += ["--occ_min", "0", "--occ_max", "0.12",
                        "--occlusion_style", "standard", "--partial_out_prob", "0",
                        "--blur_min", "0", "--blur_max", "1", "--noise_std_min", "0",
                        "--noise_std_max", "3", "--brightness_min", "-8",
                        "--brightness_max", "8", "--contrast_min", "0.9",
                        "--contrast_max", "1.1", "--motion_blur_prob", "0",
                        "--scintillation_prob", "0"]
            elif scenario == "degraded":
                cmd += ["--occ_min", "0.05", "--occ_max", "0.35",
                        "--occlusion_style", "standard", "--partial_out_prob", "0",
                        "--degradation_preset", "soft_focus", "--soft_focus_strength", "0.72"]
            else:
                # The exact quota mode forces a clamped target at a tile edge.
                cmd += ["--occ_min", "0.05", "--occ_max", "0.35",
                        "--occlusion_style", "standard", "--empty_negative_ratio", "0",
                        "--boundary_target_ratio", "1", "--partial_out_max_ratio", "0.12"]
                # Remove the mutually exclusive --negative-ratio pair.
                pos = cmd.index("--negative_ratio")
                del cmd[pos:pos + 2]
            jobs.append((out, cmd))

    negative_common = [
        sys.executable, str(GENERATOR), "--num_images", str(negatives),
        "--output_size", "1024x640", "--marker_style", "cctag_source",
        "--num_rings", "3", "--heatmap_stride", "4", "--heatmap_sigma", "2.0",
        "--negative_ratio", "1", "--background_complexity", "complex",
    ]
    low_light = root / "datasets" / "negatives" / "complex_low_light"
    jobs.append((low_light, negative_common + [
        "--output_dir", str(low_light), "--seed", "702001", "--low_light_prob", "1",
        "--vignette_prob", "0.7", "--noise_std_min", "5", "--noise_std_max", "22",
    ]))
    overexposure = root / "datasets" / "negatives" / "overexposure"
    jobs.append((overexposure, negative_common + [
        "--output_dir", str(overexposure), "--seed", "702002", "--overexposure_prob", "1",
        "--brightness_min", "20", "--brightness_max", "55", "--contrast_min", "0.6",
        "--contrast_max", "1.15", "--scintillation_prob", "0.25",
    ]))
    return jobs


def _run_job(job: tuple[Path, list[str]]) -> Path:
    output, command = job
    subprocess.run(command, cwd=ROOT, check=True)
    return output


def _job_count(command: list[str]) -> int:
    return int(command[command.index("--num_images") + 1])


def _leaf_complete(output: Path, expected: int) -> bool:
    labels = output / "labels.csv"
    if not labels.is_file():
        return False
    with labels.open(newline="", encoding="utf-8") as handle:
        row_count = sum(1 for _ in csv.DictReader(handle))
    image_count = sum(1 for path in (output / "images").glob("*") if path.is_file())
    heatmap_count = sum(1 for path in (output / "heatmaps").glob("*") if path.is_file())
    return row_count == expected and image_count == expected and heatmap_count == expected


def _resolve_sample_file(directory: Path, stem: str, extensions: tuple[str, ...]) -> Path:
    for suffix in extensions:
        candidate = directory / f"{stem}{suffix}"
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError(f"No sample file for {stem} under {directory}")


def build_manifest(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for labels_path in sorted((root / "datasets").rglob("labels.csv")):
        leaf = labels_path.parent
        relative = leaf.relative_to(root / "datasets").parts
        if relative[0] == "negatives":
            kind, scenario, expected_bin = "negative", relative[1], ""
        else:
            kind, scenario, expected_bin = "positive", relative[0], relative[1]
        with labels_path.open(newline="", encoding="utf-8") as handle:
            for source in csv.DictReader(handle):
                stem = source["filename"].strip()
                image = _resolve_sample_file(leaf / "images", stem, (".png", ".jpg", ".jpeg"))
                heatmap = _resolve_sample_file(leaf / "heatmaps", stem, (".npz", ".npy"))
                is_negative = int(source.get("is_negative") or 0)
                diameter = 2.0 * max(float(source.get("ellipse_a") or 0), float(source.get("ellipse_b") or 0))
                actual_bin = "" if is_negative else diameter_bin(diameter)
                if kind == "positive" and actual_bin != expected_bin:
                    raise ValueError(f"{leaf}/{stem}: diameter {diameter:.6f} is in {actual_bin}, expected {expected_bin}")
                if kind == "positive" and is_negative:
                    raise ValueError(f"{leaf}/{stem}: unexpected negative in positive suite")
                if kind == "negative" and not is_negative:
                    raise ValueError(f"{leaf}/{stem}: unexpected positive in negative suite")
                sample_id = f"synthetic/{scenario}/{expected_bin or 'all'}/{leaf.name}/{stem}"
                rows.append({
                    "sample_id": sample_id, "dataset_kind": kind, "scenario": scenario,
                    "size_bin": actual_bin or "", "image_path": str(image),
                    "heatmap_path": str(heatmap), "center_x": source.get("center_x") or source.get("x") or "-1",
                    "center_y": source.get("center_y") or source.get("y") or "-1",
                    "ellipse_a": source.get("ellipse_a") or "0", "ellipse_b": source.get("ellipse_b") or "0",
                    "diameter_px": f"{diameter:.6f}", "is_negative": str(is_negative),
                    "source_dataset": str(leaf.resolve()), "source_filename": stem,
                })
    return rows


def validate_manifest(rows: list[dict[str, Any]], per_bin: int, negatives: int) -> dict[str, Any]:
    errors: list[str] = []
    count_table: dict[str, int] = {}
    for scenario in POSITIVE_SCENARIOS:
        for bin_name, _, _ in BIN_SPECS:
            count = sum(r["dataset_kind"] == "positive" and r["scenario"] == scenario and r["size_bin"] == bin_name for r in rows)
            count_table[f"{scenario}/{bin_name}"] = count
            if count != per_bin:
                errors.append(f"{scenario}/{bin_name}: {count}, expected {per_bin}")
    for scenario in NEGATIVE_SCENARIOS:
        count = sum(r["dataset_kind"] == "negative" and r["scenario"] == scenario for r in rows)
        count_table[f"negative/{scenario}"] = count
        if count != negatives:
            errors.append(f"negative/{scenario}: {count}, expected {negatives}")
    missing_images = [r["image_path"] for r in rows if not Path(r["image_path"]).is_file()]
    missing_heatmaps = [r["heatmap_path"] for r in rows if not Path(r["heatmap_path"]).is_file()]
    if missing_images:
        errors.append(f"missing images: {len(missing_images)}")
    if missing_heatmaps:
        errors.append(f"missing heatmaps: {len(missing_heatmaps)}")
    if errors:
        raise ValueError("Benchmark corpus validation failed:\n  " + "\n  ".join(errors))
    return {
        "num_rows": len(rows), "positive_rows": sum(r["dataset_kind"] == "positive" for r in rows),
        "negative_rows": sum(r["dataset_kind"] == "negative" for r in rows),
        "counts": count_table, "missing_images": 0, "missing_heatmaps": 0,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("/mnt/tmp1/cctag_70_150_benchmark_seed_701500"))
    parser.add_argument("--samples-per-bin", type=int, default=250)
    parser.add_argument("--negatives-per-scenario", type=int, default=500)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Keep complete leaves and regenerate only missing/incomplete leaves")
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.output_root.resolve()
    if args.force and args.resume:
        raise SystemExit("--force and --resume are mutually exclusive")
    if not args.verify_only:
        if root.exists():
            if args.force:
                shutil.rmtree(root)
            elif not args.resume:
                raise SystemExit(f"Refusing to overwrite {root}; use --force, --resume, or --verify-only")
        root.mkdir(parents=True, exist_ok=True)
        jobs = generation_jobs(root, args.samples_per_bin, args.negatives_per_scenario)
        if args.resume:
            pending = []
            for job in jobs:
                output, command = job
                if _leaf_complete(output, _job_count(command)):
                    print(f"resume: keeping complete leaf {output}", flush=True)
                else:
                    if output.exists():
                        shutil.rmtree(output)
                    pending.append(job)
            jobs = pending
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            futures = {executor.submit(_run_job, job): job[0] for job in jobs}
            for future in as_completed(futures):
                print(f"generated {future.result()}", flush=True)
    rows = build_manifest(root)
    integrity = validate_manifest(rows, args.samples_per_bin, args.negatives_per_scenario)
    manifest_path = root / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "seed_scheme": "701500 + bin*100 + scenario*10; negatives 702001/702002",
        "diameter_definition": "2 * max(ellipse_a, ellipse_b)",
        "bin_semantics": "[70,90), [90,110), [110,130), [130,150]",
        "output_size": "1024x640", "samples_per_positive_cell": args.samples_per_bin,
        "negatives_per_scenario": args.negatives_per_scenario, "integrity": integrity,
    }
    (root / "benchmark_config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), **integrity}, indent=2))


if __name__ == "__main__":
    main()
