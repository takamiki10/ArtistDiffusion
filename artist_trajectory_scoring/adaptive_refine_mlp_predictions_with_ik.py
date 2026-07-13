#!/usr/bin/env python3
"""
Adaptive MLP + IK refinement wrapper.

This script runs the existing `refine_mlp_predictions_with_ik.py` in two stages:

Stage 1 (fast/default):
    smooth_weight = 0.01, max_iters = 200

Stage 2 (slower/stronger):
    rerun only paths whose Stage-1 `after_max_error` is above a threshold
    smooth_weight = 0.001, max_iters = 500

It then merges the two summaries into one final CSV.

Assumption:
    `refine_mlp_predictions_with_ik.py` already exists and accepts:
        --dataset_dir
        --output_csv
        --smooth_weight
        --max_iters
        --overwrite

This wrapper intentionally avoids editing the IK internals. It only controls which
path folders are sent to the existing refinement script.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run adaptive two-stage MLP-initialized IK refinement."
    )
    parser.add_argument(
        "--dataset_dir",
        type=Path,
        required=True,
        help="Dataset directory containing path folders, e.g. data/.../experts/test",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        required=True,
        help="Final merged adaptive summary CSV.",
    )
    parser.add_argument(
        "--refine_script",
        type=Path,
        default=Path("refine_mlp_predictions_with_ik.py"),
        help="Path to the existing refinement script.",
    )
    parser.add_argument("--stage1_smooth_weight", type=float, default=0.01)
    parser.add_argument("--stage1_max_iters", type=int, default=200)
    parser.add_argument("--stage2_smooth_weight", type=float, default=0.001)
    parser.add_argument("--stage2_max_iters", type=int, default=500)
    parser.add_argument(
        "--max_error_threshold",
        type=float,
        default=0.03,
        help="Rerun paths with Stage-1 after_max_error above this value in meters.",
    )
    parser.add_argument(
        "--work_dir",
        type=Path,
        default=None,
        help="Optional directory for intermediate CSVs and temporary subset folders.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Pass --overwrite to the underlying refinement script.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable used to call the underlying refinement script.",
    )
    return parser.parse_args()


def run_command(cmd: list[str]) -> None:
    print("\n$ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def run_refinement(
    *,
    python_exe: str,
    refine_script: Path,
    dataset_dir: Path,
    output_csv: Path,
    smooth_weight: float,
    max_iters: int,
    overwrite: bool,
) -> None:
    cmd = [
        python_exe,
        str(refine_script),
        "--dataset_dir",
        str(dataset_dir),
        "--output_csv",
        str(output_csv),
        "--smooth_weight",
        str(smooth_weight),
        "--max_iters",
        str(max_iters),
    ]
    if overwrite:
        cmd.append("--overwrite")
    run_command(cmd)


def find_path_column(df: pd.DataFrame) -> str:
    candidates = ["path", "path_name", "name", "folder", "path_id"]
    for col in candidates:
        if col in df.columns:
            return col

    # Fallback: choose the first object/string column that looks path-like.
    for col in df.columns:
        if df[col].dtype == object:
            sample = df[col].dropna().astype(str).head(10).tolist()
            if any(s.startswith("path_") for s in sample):
                return col

    raise ValueError(
        "Could not identify the path/folder column in the refinement summary. "
        f"Available columns: {list(df.columns)}"
    )


def make_subset_dir(dataset_dir: Path, path_names: Iterable[str], subset_dir: Path) -> None:
    if subset_dir.exists():
        shutil.rmtree(subset_dir)
    subset_dir.mkdir(parents=True, exist_ok=True)

    for name in path_names:
        src = dataset_dir / str(name)
        dst = subset_dir / str(name)
        if not src.exists():
            raise FileNotFoundError(f"Selected path folder does not exist: {src}")

        try:
            dst.symlink_to(src.resolve(), target_is_directory=True)
        except OSError:
            # Fallback for filesystems that disallow symlinks.
            # This is slower, but still allows Stage 2 metrics to be computed.
            shutil.copytree(src, dst)


def merge_stage_summaries(
    *,
    stage1: pd.DataFrame,
    stage2: pd.DataFrame | None,
    path_col: str,
    selected_names: list[str],
) -> pd.DataFrame:
    final = stage1.copy()
    final["adaptive_stage"] = 1

    if stage2 is None or len(stage2) == 0:
        return final

    stage2_final = stage2.copy()
    stage2_final["adaptive_stage"] = 2

    final = final[~final[path_col].astype(str).isin(set(map(str, selected_names)))]
    final = pd.concat([final, stage2_final], ignore_index=True)
    final = final.sort_values(path_col).reset_index(drop=True)
    return final


def print_summary(df: pd.DataFrame) -> None:
    print("\nAdaptive refinement summary")
    print("---------------------------")
    print(f"paths: {len(df)}")

    for col in ["after_path_error", "after_mean_error", "after_max_error", "solve_time_sec"]:
        if col in df.columns:
            print(
                f"{col}: mean={df[col].mean():.9g}, "
                f"median={df[col].median():.9g}, max={df[col].max():.9g}"
            )

    if "adaptive_stage" in df.columns:
        print("stage counts:")
        print(df["adaptive_stage"].value_counts().sort_index().to_string())


def main() -> None:
    args = parse_args()
    dataset_dir = args.dataset_dir.resolve()
    refine_script = args.refine_script.resolve()
    output_csv = args.output_csv.resolve()

    if not dataset_dir.exists():
        raise FileNotFoundError(f"dataset_dir does not exist: {dataset_dir}")
    if not refine_script.exists():
        raise FileNotFoundError(f"refine_script does not exist: {refine_script}")

    if args.work_dir is None:
        work_root_obj = tempfile.TemporaryDirectory(prefix="adaptive_refine_")
        work_root = Path(work_root_obj.name)
    else:
        work_root_obj = None
        work_root = args.work_dir.resolve()
        work_root.mkdir(parents=True, exist_ok=True)

    try:
        stage1_csv = work_root / "stage1_refine_summary.csv"
        stage2_csv = work_root / "stage2_refine_summary.csv"
        subset_dir = work_root / "stage2_subset"

        print("Running Stage 1: fast MLP-initialized IK refinement")
        run_refinement(
            python_exe=args.python,
            refine_script=refine_script,
            dataset_dir=dataset_dir,
            output_csv=stage1_csv,
            smooth_weight=args.stage1_smooth_weight,
            max_iters=args.stage1_max_iters,
            overwrite=args.overwrite,
        )

        stage1 = pd.read_csv(stage1_csv)
        if "after_max_error" not in stage1.columns:
            raise ValueError(
                "Stage-1 CSV does not contain `after_max_error`, which is needed "
                "to select spike cases."
            )

        path_col = find_path_column(stage1)
        needs_stage2 = stage1[stage1["after_max_error"] > args.max_error_threshold]
        selected_names = needs_stage2[path_col].astype(str).tolist()

        print(
            f"\nStage 1 complete. {len(selected_names)}/{len(stage1)} paths exceed "
            f"after_max_error > {args.max_error_threshold:.4f} m."
        )

        stage2 = None
        if selected_names:
            make_subset_dir(dataset_dir, selected_names, subset_dir)
            print("Running Stage 2: stronger refinement on spike cases only")
            run_refinement(
                python_exe=args.python,
                refine_script=refine_script,
                dataset_dir=subset_dir,
                output_csv=stage2_csv,
                smooth_weight=args.stage2_smooth_weight,
                max_iters=args.stage2_max_iters,
                overwrite=True,
            )
            stage2 = pd.read_csv(stage2_csv)

        final = merge_stage_summaries(
            stage1=stage1,
            stage2=stage2,
            path_col=path_col,
            selected_names=selected_names,
        )

        output_csv.parent.mkdir(parents=True, exist_ok=True)
        final.to_csv(output_csv, index=False)
        print(f"\nSaved final adaptive summary: {output_csv}")
        print_summary(final)

    finally:
        if work_root_obj is not None:
            work_root_obj.cleanup()


if __name__ == "__main__":
    main()
