#!/usr/bin/env python3
"""
Batch-generate IK expert trajectories for randomized Cartesian raw paths.

This script wraps generate_ik_seed_path.py and plot_diffusion_diagnostic.py.
Accepted paths are written under experts/{split}/path_XXXX. Rejected paths are
kept under rejected/{split}/path_XXXX with the same files plus logs/metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


LOOSER_WARNING_TYPES = {"square", "zigzag"}


def read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def copy_raw_files(raw_path_dir: Path, output_path_dir: Path) -> None:
    output_path_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(raw_path_dir / "desired_path.csv", output_path_dir / "desired_path.csv")
    if (raw_path_dir / "path_meta.json").exists():
        shutil.copy2(raw_path_dir / "path_meta.json", output_path_dir / "path_meta.json")


def path_type_for(raw_path_dir: Path) -> str:
    meta_path = raw_path_dir / "path_meta.json"
    if not meta_path.exists():
        return "unknown"
    return str(read_json(meta_path).get("path_type", "unknown"))


def run_command(cmd: List[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)


def run_ik(
    script_dir: Path,
    output_path_dir: Path,
    smooth_weight: float,
    first_point_maxiter: int,
    maxiter: int,
    ftol: float,
    num_restarts: int,
    retry_error_threshold: float,
    random_seed: int,
) -> subprocess.CompletedProcess[str]:
    cmd = [
        sys.executable,
        "generate_ik_seed_path.py",
        "--path_csv",
        str(output_path_dir / "desired_path.csv"),
        "--output_q_csv",
        str(output_path_dir / "expert_q.csv"),
        "--output_ee_csv",
        str(output_path_dir / "expert_ee.csv"),
        "--metrics_json",
        str(output_path_dir / "metrics.json"),
        "--smooth_weight",
        str(smooth_weight),
        "--first_point_maxiter",
        str(first_point_maxiter),
        "--maxiter",
        str(maxiter),
        "--ftol",
        str(ftol),
        "--num_restarts",
        str(num_restarts),
        "--retry_error_threshold",
        str(retry_error_threshold),
        "--random_seed",
        str(random_seed),
    ]
    return run_command(cmd, cwd=script_dir)


def run_plot(script_dir: Path, output_path_dir: Path) -> subprocess.CompletedProcess[str]:
    cmd = [
        sys.executable,
        "plot_diffusion_diagnostic.py",
        "--desired_path",
        str(output_path_dir / "desired_path.csv"),
        "--ee_csv",
        str(output_path_dir / "expert_ee.csv"),
        "--q_csv",
        str(output_path_dir / "expert_q.csv"),
        "--output_png",
        str(output_path_dir / "plot.png"),
    ]
    return run_command(cmd, cwd=script_dir)


def selected_splits(split: str) -> List[str]:
    if split == "all":
        return ["train", "test"]
    return [split]


def iter_raw_path_dirs(raw_dir: Path, split: str) -> Iterable[Path]:
    split_dir = raw_dir / split
    if not split_dir.exists():
        return []
    return sorted(
        p for p in split_dir.iterdir()
        if p.is_dir() and (p / "desired_path.csv").exists()
    )


def accepted_by_metrics(
    metrics: Dict[str, Any],
    max_mean_error: float,
    max_max_error: float,
) -> tuple[bool, str]:
    mean_error = float(metrics.get("mean_error", float("inf")))
    max_error = float(metrics.get("max_error", float("inf")))
    reasons = []
    if mean_error > max_mean_error:
        reasons.append(f"mean_error {mean_error:.6f} > {max_mean_error:.6f}")
    if max_error > max_max_error:
        reasons.append(f"max_error {max_error:.6f} > {max_max_error:.6f}")
    return len(reasons) == 0, "; ".join(reasons)


def move_replace(src: Path, dst: Path) -> None:
    if dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))


def write_summary(summary_csv: Path, rows: List[Dict[str, Any]]) -> None:
    summary_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "split",
        "path_id",
        "path_type",
        "accepted",
        "mean_error",
        "max_error",
        "path_error",
        "output_folder",
        "reason_if_rejected",
    ]
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Batch-generate IK experts from raw Cartesian paths.")
    parser.add_argument("--raw_dir", type=Path, default=Path("data/cartesian_expert_dataset_v2/raw_paths"))
    parser.add_argument("--output_dir", type=Path, default=Path("data/cartesian_expert_dataset_v2/experts"))
    parser.add_argument("--split", choices=["train", "test", "all"], default="all")
    parser.add_argument("--smooth_weight", type=float, default=0.01)
    parser.add_argument("--num_restarts", type=int, default=50)
    parser.add_argument("--retry_error_threshold", type=float, default=0.02)
    parser.add_argument("--max_mean_error", type=float, default=0.010)
    parser.add_argument("--max_max_error", type=float, default=0.030)
    parser.add_argument("--device", default="cpu", help="Accepted for CLI symmetry; IK generation is CPU/scipy based.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--first_point_maxiter", type=int, default=1000)
    parser.add_argument("--maxiter", type=int, default=300)
    parser.add_argument("--ftol", type=float, default=1e-10)
    parser.add_argument("--random_seed", type=int, default=0)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    script_dir = Path(__file__).resolve().parent
    rejected_root = args.output_dir.parent / "rejected"
    staging_root = args.output_dir.parent / "_ik_staging"
    rows: List[Dict[str, Any]] = []

    if not args.raw_dir.exists():
        raise FileNotFoundError(f"Raw path directory not found: {args.raw_dir}")

    for split in selected_splits(args.split):
        for raw_path_dir in iter_raw_path_dirs(args.raw_dir, split):
            path_id = raw_path_dir.name
            path_type = path_type_for(raw_path_dir)
            accepted_dir = args.output_dir / split / path_id
            rejected_dir = rejected_root / split / path_id
            staging_dir = staging_root / split / path_id

            if accepted_dir.exists() and not args.overwrite:
                print(f"Skipping existing accepted path: {split}/{path_id}")
                metrics_path = accepted_dir / "metrics.json"
                metrics = read_json(metrics_path) if metrics_path.exists() else {}
                rows.append(
                    {
                        "split": split,
                        "path_id": path_id,
                        "path_type": path_type,
                        "accepted": True,
                        "mean_error": metrics.get("mean_error", ""),
                        "max_error": metrics.get("max_error", ""),
                        "path_error": metrics.get("path_error", ""),
                        "output_folder": str(accepted_dir),
                        "reason_if_rejected": "",
                    }
                )
                continue

            if staging_dir.exists():
                shutil.rmtree(staging_dir)
            copy_raw_files(raw_path_dir, staging_dir)

            print(f"\nGenerating IK expert for {split}/{path_id} ({path_type})")
            ik_proc = run_ik(
                script_dir=script_dir,
                output_path_dir=staging_dir,
                smooth_weight=args.smooth_weight,
                first_point_maxiter=args.first_point_maxiter,
                maxiter=args.maxiter,
                ftol=args.ftol,
                num_restarts=args.num_restarts,
                retry_error_threshold=args.retry_error_threshold,
                random_seed=args.random_seed,
            )
            write_text(staging_dir / "ik_stdout.log", ik_proc.stdout)
            write_text(staging_dir / "ik_stderr.log", ik_proc.stderr)

            metrics: Dict[str, Any] = {}
            reason = ""
            accepted = False
            if ik_proc.returncode != 0:
                reason = f"generate_ik_seed_path.py failed with returncode {ik_proc.returncode}"
            elif not (staging_dir / "metrics.json").exists():
                reason = "metrics.json missing after IK generation"
            else:
                metrics = read_json(staging_dir / "metrics.json")
                accepted, reason = accepted_by_metrics(metrics, args.max_mean_error, args.max_max_error)

            if (staging_dir / "expert_q.csv").exists() and (staging_dir / "expert_ee.csv").exists():
                plot_proc = run_plot(script_dir, staging_dir)
                write_text(staging_dir / "plot_stdout.log", plot_proc.stdout)
                write_text(staging_dir / "plot_stderr.log", plot_proc.stderr)
                if plot_proc.returncode != 0 and not reason:
                    reason = f"plot_diffusion_diagnostic.py failed with returncode {plot_proc.returncode}"

            if path_type in LOOSER_WARNING_TYPES and reason:
                print(f"  WARNING ({path_type} sharp-corner path): {reason}")

            final_dir = accepted_dir if accepted else rejected_dir
            if final_dir.exists() and args.overwrite:
                shutil.rmtree(final_dir)
            move_replace(staging_dir, final_dir)

            print(
                f"  {'ACCEPTED' if accepted else 'REJECTED'} | "
                f"mean={metrics.get('mean_error', '')} | "
                f"max={metrics.get('max_error', '')} | "
                f"path_error={metrics.get('path_error', '')} | "
                f"out={final_dir}"
            )

            rows.append(
                {
                    "split": split,
                    "path_id": path_id,
                    "path_type": path_type,
                    "accepted": accepted,
                    "mean_error": metrics.get("mean_error", ""),
                    "max_error": metrics.get("max_error", ""),
                    "path_error": metrics.get("path_error", ""),
                    "output_folder": str(final_dir),
                    "reason_if_rejected": "" if accepted else reason,
                }
            )

    summary_csv = args.output_dir / "ik_generation_summary.csv"
    write_summary(summary_csv, rows)
    if staging_root.exists():
        shutil.rmtree(staging_root)

    accepted_count = sum(1 for row in rows if row["accepted"])
    print()
    print(f"Saved IK generation summary: {summary_csv}")
    print(f"Accepted: {accepted_count}/{len(rows)}")
    print(f"Rejected root: {rejected_root}")


if __name__ == "__main__":
    main()
