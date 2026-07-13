#!/usr/bin/env python3
"""Refine path-conditioned MLP joint predictions with sequential IK.

For each accepted path folder, this script uses path_conditioned_pred_q.csv as
the per-timestep IK initial guess, tracks desired_path.csv in Cartesian xyz, and
adds a smoothness penalty toward the previous refined joint solution.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd

from generate_ik_seed_path import (
    DEFAULT_EE_LINK,
    DEFAULT_JOINT_NAMES,
    DEFAULT_URDF_PATH,
    get_joint_bounds,
    load_robot,
    read_desired_path,
    save_ee_csv,
    save_q_csv,
    solve_ik_from_initial_guess,
)


PRED_Q_NAME = "path_conditioned_pred_q.csv"
PRED_EE_NAME = "path_conditioned_pred_ee.csv"
REFINED_Q_NAME = "refined_mlp_ik_q.csv"
REFINED_EE_NAME = "refined_mlp_ik_ee.csv"
REFINED_METRICS_NAME = "refined_mlp_ik_metrics.json"


def resolve_urdf_path(path: Path) -> Path:
    if path.exists():
        return path
    script_relative = Path(__file__).resolve().parent / path
    if script_relative.exists():
        return script_relative
    repo_relative = Path(__file__).resolve().parent.parent / path
    if repo_relative.exists():
        return repo_relative
    return path


def path_dirs(dataset_dir: Path) -> Iterable[Path]:
    return sorted(
        p
        for p in dataset_dir.iterdir()
        if p.is_dir()
        and (p / "desired_path.csv").exists()
        and (p / PRED_Q_NAME).exists()
        and (p / PRED_EE_NAME).exists()
    )


def read_q_csv(q_csv: Path, joint_names: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(q_csv)
    if len(df) == 0:
        raise ValueError(f"{q_csv} is empty")
    if "t" not in df.columns:
        raise ValueError(f"{q_csv} missing required column: t")

    if set(joint_names).issubset(df.columns):
        q_cols = list(joint_names)
    else:
        q_cols = [f"q{i + 1}" for i in range(len(joint_names))]
        missing = [col for col in q_cols if col not in df.columns]
        if missing:
            raise ValueError(
                f"{q_csv} must contain either joint columns {list(joint_names)} "
                f"or q-columns {q_cols}; missing {missing}"
            )

    return (
        df["t"].to_numpy(dtype=np.float64),
        df[q_cols].to_numpy(dtype=np.float64),
    )


def read_ee_csv(ee_csv: Path) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(ee_csv)
    required = {"t", "x", "y", "z"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{ee_csv} missing required columns: {sorted(missing)}")
    if len(df) == 0:
        raise ValueError(f"{ee_csv} is empty")
    return (
        df["t"].to_numpy(dtype=np.float64),
        df[["x", "y", "z"]].to_numpy(dtype=np.float64),
    )


def validate_matching_lengths(path_dir: Path, arrays: Dict[str, np.ndarray]) -> None:
    lengths = {name: len(value) for name, value in arrays.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(f"{path_dir} has length mismatch: {lengths}")


def trajectory_metrics(ee: np.ndarray, desired: np.ndarray, t: np.ndarray) -> Dict[str, float]:
    if ee.shape != desired.shape:
        raise ValueError(f"Shape mismatch: ee {ee.shape}, desired {desired.shape}")
    error = np.linalg.norm(ee - desired, axis=1)
    max_idx = int(np.argmax(error))
    return {
        "path_error": float(np.mean(error * error)),
        "mean_error": float(np.mean(error)),
        "max_error": float(error[max_idx]),
        "max_error_time": float(t[max_idx]),
        "max_error_index": max_idx,
    }


def write_json(path: Path, data: Dict[str, Any], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def write_summary_csv(path: Path, rows: List[Dict[str, Any]], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} exists; pass --overwrite to replace it")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "path_id",
        "before_path_error",
        "after_path_error",
        "before_mean_error",
        "after_mean_error",
        "before_max_error",
        "after_max_error",
        "solve_time_sec",
        "accepted",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def output_paths(path_dir: Path) -> Tuple[Path, Path, Path]:
    return (
        path_dir / REFINED_Q_NAME,
        path_dir / REFINED_EE_NAME,
        path_dir / REFINED_METRICS_NAME,
    )


def ensure_outputs_may_be_written(path_dir: Path, overwrite: bool) -> None:
    existing = [path for path in output_paths(path_dir) if path.exists()]
    if existing and not overwrite:
        names = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Refined output already exists for {path_dir}: {names}")


def refine_path(
    path_dir: Path,
    robot: Any,
    joint_names: Sequence[str],
    ee_link: str,
    bounds: Sequence[Tuple[float, float]],
    smooth_weight: float,
    max_iters: int,
    overwrite: bool,
) -> Dict[str, Any]:
    ensure_outputs_may_be_written(path_dir, overwrite)

    t, desired = read_desired_path(path_dir / "desired_path.csv")
    pred_t, pred_q = read_q_csv(path_dir / PRED_Q_NAME, joint_names)
    pred_ee_t, pred_ee = read_ee_csv(path_dir / PRED_EE_NAME)
    validate_matching_lengths(
        path_dir,
        {
            "desired_path": desired,
            PRED_Q_NAME: pred_q,
            PRED_EE_NAME: pred_ee,
        },
    )
    if not np.allclose(t, pred_t) or not np.allclose(t, pred_ee_t):
        raise ValueError(f"{path_dir} has mismatched t columns")

    before = trajectory_metrics(pred_ee, desired, t)
    q_refined = np.empty_like(pred_q, dtype=np.float64)
    ee_refined = np.empty((len(t), 3), dtype=np.float64)
    attempts: List[Dict[str, Any]] = []

    start = time.perf_counter()
    q_prev: np.ndarray | None = None
    for i, p_des in enumerate(desired):
        attempt = solve_ik_from_initial_guess(
            robot=robot,
            p_des=p_des,
            q_init=pred_q[i],
            q_ref=q_prev,
            joint_names=joint_names,
            ee_link=ee_link,
            bounds=bounds,
            smooth_weight=smooth_weight,
            maxiter=max_iters,
            ftol=1e-10,
        )
        q_refined[i] = attempt.q
        ee_refined[i] = attempt.ee
        q_prev = attempt.q
        attempts.append(
            {
                "index": i,
                "t": float(t[i]),
                "error": float(attempt.error),
                "success": bool(attempt.success),
                "nit": int(attempt.nit),
                "message": attempt.message,
            }
        )
    solve_time_sec = float(time.perf_counter() - start)

    after = trajectory_metrics(ee_refined, desired, t)
    accepted = bool(
        np.isfinite(after["path_error"])
        and np.isfinite(before["path_error"])
        and after["path_error"] <= before["path_error"]
    )

    q_csv, ee_csv, metrics_json = output_paths(path_dir)
    save_q_csv(q_csv, t, q_refined)
    save_ee_csv(ee_csv, t, ee_refined)
    write_json(
        metrics_json,
        {
            "path_id": path_dir.name,
            "smooth_weight": smooth_weight,
            "max_iters": max_iters,
            "solve_time_sec": solve_time_sec,
            "accepted": accepted,
            "before": before,
            "after": after,
            "attempts": attempts,
        },
        overwrite=overwrite,
    )

    return {
        "path_id": path_dir.name,
        "before_path_error": before["path_error"],
        "after_path_error": after["path_error"],
        "before_mean_error": before["mean_error"],
        "after_mean_error": after["mean_error"],
        "before_max_error": before["max_error"],
        "after_max_error": after["max_error"],
        "solve_time_sec": solve_time_sec,
        "accepted": accepted,
    }


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Refine path-conditioned MLP q predictions with sequential IK."
    )
    parser.add_argument("--dataset_dir", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path, required=True)
    parser.add_argument("--smooth_weight", type=float, default=0.01)
    parser.add_argument("--max_iters", type=int, default=200)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", default="cpu", help="Accepted for CLI symmetry; scipy IK runs on CPU.")
    parser.add_argument("--urdf_path", type=Path, default=Path(DEFAULT_URDF_PATH))
    parser.add_argument("--ee_link", default=DEFAULT_EE_LINK)
    parser.add_argument("--joint_names", default=",".join(DEFAULT_JOINT_NAMES))
    parser.add_argument("--fallback_q_min", type=float, default=-np.pi)
    parser.add_argument("--fallback_q_max", type=float, default=np.pi)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    if args.smooth_weight < 0.0:
        raise ValueError("--smooth_weight must be non-negative")
    if args.max_iters < 1:
        raise ValueError("--max_iters must be at least 1")
    if not args.dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {args.dataset_dir}")
    if args.output_csv.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output_csv} exists; pass --overwrite to replace it")

    joint_names = [name.strip() for name in args.joint_names.split(",") if name.strip()]
    if not joint_names:
        raise ValueError("--joint_names must contain at least one joint")

    urdf_path = resolve_urdf_path(args.urdf_path)
    robot = load_robot(urdf_path)
    bounds = get_joint_bounds(robot, joint_names, args.fallback_q_min, args.fallback_q_max)

    paths = list(path_dirs(args.dataset_dir))
    if not paths:
        raise FileNotFoundError(
            f"No path folders with desired_path.csv, {PRED_Q_NAME}, and {PRED_EE_NAME} "
            f"found in {args.dataset_dir}"
        )

    print("MLP IK refinement")
    print(f"  dataset_dir:   {args.dataset_dir}")
    print(f"  output_csv:    {args.output_csv}")
    print(f"  urdf_path:     {urdf_path}")
    print(f"  ee_link:       {args.ee_link}")
    print(f"  joints:        {','.join(joint_names)}")
    print(f"  smooth_weight: {args.smooth_weight}")
    print(f"  max_iters:     {args.max_iters}")
    print(f"  device:        {args.device} (unused; scipy IK is CPU)")
    print(f"  paths:         {len(paths)}")

    rows: List[Dict[str, Any]] = []
    for i, path_dir in enumerate(paths, start=1):
        print(f"[{i}/{len(paths)}] refining {path_dir.name}")
        row = refine_path(
            path_dir=path_dir,
            robot=robot,
            joint_names=joint_names,
            ee_link=args.ee_link,
            bounds=bounds,
            smooth_weight=args.smooth_weight,
            max_iters=args.max_iters,
            overwrite=args.overwrite,
        )
        rows.append(row)
        print(
            "  "
            f"path_error {row['before_path_error']:.8e} -> {row['after_path_error']:.8e} | "
            f"mean {row['before_mean_error']:.6f} -> {row['after_mean_error']:.6f} m | "
            f"accepted={row['accepted']}"
        )

    write_summary_csv(args.output_csv, rows, overwrite=args.overwrite)
    accepted_count = sum(1 for row in rows if row["accepted"])
    print()
    print(f"Saved summary CSV: {args.output_csv}")
    print(f"Accepted refinements: {accepted_count}/{len(rows)}")


if __name__ == "__main__":
    main()
