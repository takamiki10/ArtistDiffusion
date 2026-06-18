#!/usr/bin/env python3
"""
Visualize diagnostic diffusion output.

Plots:
1. Desired Cartesian path vs FK end-effector path, x-y view
2. Desired vs FK path, x-z view
3. Tracking error over time
4. Generated joint trajectory q1..q6 over time

Expected files:
    desired_path.csv:
        t,x,y,z

    diffusion_diagnostic_pred_ee.csv:
        t,x,y,z

    diffusion_diagnostic_pred_q.csv:
        t,q1,q2,q3,q4,q5,q6

Example:
    python plot_diffusion_diagnostic.py \
      --desired_path data/synthetic_paths_test/path_001/desired_path.csv \
      --ee_csv data/synthetic_paths_test/path_001/diffusion_diagnostic_pred_ee.csv \
      --q_csv data/synthetic_paths_test/path_001/diffusion_diagnostic_pred_q.csv \
      --output_png data/synthetic_paths_test/path_001/diffusion_diagnostic_plot.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def require_columns(df: pd.DataFrame, path: Path, cols: list[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{path} is missing columns {missing}. Found: {list(df.columns)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--desired_path", type=Path, required=True)
    parser.add_argument("--ee_csv", type=Path, required=True)
    parser.add_argument("--q_csv", type=Path, required=True)
    parser.add_argument("--output_png", type=Path, required=True)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    desired = pd.read_csv(args.desired_path)
    ee = pd.read_csv(args.ee_csv)
    q = pd.read_csv(args.q_csv)

    require_columns(desired, args.desired_path, ["t", "x", "y", "z"])
    require_columns(ee, args.ee_csv, ["t", "x", "y", "z"])
    require_columns(q, args.q_csv, ["t", "q1", "q2", "q3", "q4", "q5", "q6"])

    if len(desired) != len(ee):
        raise ValueError(
            f"desired_path and ee_csv length mismatch: "
            f"{len(desired)} vs {len(ee)}"
        )

    desired_xyz = desired[["x", "y", "z"]].to_numpy(dtype=float)
    ee_xyz = ee[["x", "y", "z"]].to_numpy(dtype=float)

    error = np.linalg.norm(ee_xyz - desired_xyz, axis=1)

    mean_error = float(np.mean(error))
    max_error = float(np.max(error))
    path_error = float(np.mean(error ** 2))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ------------------------------------------------------------
    # 1. X-Y path comparison
    # ------------------------------------------------------------
    ax = axes[0, 0]
    ax.plot(desired["x"], desired["y"], label="Desired path")
    ax.plot(ee["x"], ee["y"], label="Diffusion FK path")
    ax.scatter(desired["x"].iloc[0], desired["y"].iloc[0], marker="o", label="Start")
    ax.scatter(desired["x"].iloc[-1], desired["y"].iloc[-1], marker="x", label="End")
    ax.set_title("Cartesian path comparison: x-y view")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.axis("equal")
    ax.grid(True)
    ax.legend()

    # ------------------------------------------------------------
    # 2. X-Z path comparison
    # ------------------------------------------------------------
    ax = axes[0, 1]
    ax.plot(desired["x"], desired["z"], label="Desired path")
    ax.plot(ee["x"], ee["z"], label="Diffusion FK path")
    ax.scatter(desired["x"].iloc[0], desired["z"].iloc[0], marker="o", label="Start")
    ax.scatter(desired["x"].iloc[-1], desired["z"].iloc[-1], marker="x", label="End")
    ax.set_title("Cartesian path comparison: x-z view")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    ax.axis("equal")
    ax.grid(True)
    ax.legend()

    # ------------------------------------------------------------
    # 3. Tracking error over time
    # ------------------------------------------------------------
    ax = axes[1, 0]
    ax.plot(desired["t"], error)
    ax.set_title(
        "Tracking error over time\n"
        f"mean={mean_error:.4f} m, max={max_error:.4f} m, path_error={path_error:.6f}"
    )
    ax.set_xlabel("time [s]")
    ax.set_ylabel("||p_fk - p_des|| [m]")
    ax.grid(True)

    # ------------------------------------------------------------
    # 4. Joint trajectory
    # ------------------------------------------------------------
    ax = axes[1, 1]
    for joint in ["q1", "q2", "q3", "q4", "q5", "q6"]:
        ax.plot(q["t"], q[joint], label=joint)

    ax.set_title("Generated joint trajectory")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("joint angle [rad]")
    ax.grid(True)
    ax.legend()

    fig.suptitle("Diagnostic diffusion trajectory visualization", fontsize=16)
    fig.tight_layout()

    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_png, dpi=200)

    print(f"Saved plot: {args.output_png}")
    print(f"mean_error: {mean_error:.8f} m")
    print(f"max_error: {max_error:.8f} m")
    print(f"path_error: {path_error:.8f}")

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()