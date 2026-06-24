#!/usr/bin/env python3
"""Plot the outputs of train_qlearn_cartesian_debug.py."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def create_debug_plot(
    path_csv: str | Path,
    learned_path_csv: str | Path,
    training_log_csv: str | Path,
    output_png: str | Path,
) -> None:
    desired = pd.read_csv(path_csv)
    learned = pd.read_csv(learned_path_csv)
    log = pd.read_csv(training_log_csv)
    required_desired = {"x", "y"}
    required_learned = {"x", "y", "tracking_error", "closest_path_index"}
    if missing := required_desired.difference(desired.columns):
        raise ValueError(f"Desired path is missing columns: {sorted(missing)}")
    if missing := required_learned.difference(learned.columns):
        raise ValueError(f"Learned path is missing columns: {sorted(missing)}")
    if not {"episode", "reward"}.issubset(log.columns):
        raise ValueError("Training log must contain episode and reward columns")

    output_png = Path(output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0, 0]
    ax.plot(desired["x"], desired["y"], "k--", linewidth=2.0, label="desired")
    ax.plot(learned["x"], learned["y"], color="tab:blue", linewidth=1.6, label="learned")
    ax.scatter(desired["x"].iloc[0], desired["y"].iloc[0], color="green", s=45, label="start")
    ax.scatter(desired["x"].iloc[-1], desired["y"].iloc[-1], color="red", s=45, label="finish")
    ax.set_title("Desired vs learned path")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.axis("equal")
    ax.grid(alpha=0.25)
    ax.legend()

    step = learned["step"] if "step" in learned else np.arange(len(learned))
    ax = axes[0, 1]
    ax.plot(step, learned["tracking_error"], color="tab:orange")
    ax.set_title("Tracking error to nearest segment")
    ax.set_xlabel("rollout step")
    ax.set_ylabel("distance")
    ax.grid(alpha=0.25)

    ax = axes[1, 0]
    ax.plot(step, learned["closest_path_index"], color="tab:green")
    ax.axhline(len(desired) - 1, color="black", linestyle="--", linewidth=1.0)
    ax.set_title("Closest desired-path index")
    ax.set_xlabel("rollout step")
    ax.set_ylabel("path index")
    ax.grid(alpha=0.25)

    ax = axes[1, 1]
    rewards = log["reward"].to_numpy(dtype=np.float64)
    ax.plot(log["episode"], rewards, alpha=0.30, linewidth=0.8, label="episode")
    window = min(100, len(log))
    if window > 1:
        smooth = pd.Series(rewards).rolling(window, min_periods=1).mean()
        ax.plot(log["episode"], smooth, linewidth=2.0, label=f"{window}-episode mean")
    ax.set_title("Episode reward")
    ax.set_xlabel("episode")
    ax.set_ylabel("reward")
    ax.grid(alpha=0.25)
    ax.legend()

    fig.suptitle("Q-learning Cartesian Cost Debugger", fontsize=15)
    fig.tight_layout()
    fig.savefig(output_png, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--path_csv", type=Path,
        default=Path("data/cartesian_test_paths/arc_001/desired_path.csv"),
    )
    parser.add_argument("--output_dir", type=Path, default=Path("qlearn_cartesian_debug_output"))
    parser.add_argument("--learned_path_csv", type=Path, default=None)
    parser.add_argument("--training_log_csv", type=Path, default=None)
    parser.add_argument("--output_png", type=Path, default=None)
    args = parser.parse_args()
    learned = args.learned_path_csv or args.output_dir / "learned_path.csv"
    log = args.training_log_csv or args.output_dir / "training_log.csv"
    output = args.output_png or args.output_dir / "qlearn_debug_plot.png"
    create_debug_plot(args.path_csv, learned, log, output)
    print(f"Saved {output}")


if __name__ == "__main__":
    main()

