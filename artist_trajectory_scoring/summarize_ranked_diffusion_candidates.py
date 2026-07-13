#!/usr/bin/env python3
"""Summarize best-of-K ranked diffusion candidate folders."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--roots",
        nargs="+",
        default=[
            "data/cartesian_expert_dataset_v3/diffusion_v1_ranked_samples/k4",
            "data/cartesian_expert_dataset_v3/diffusion_v1_ranked_samples/k8",
            "data/cartesian_expert_dataset_v3/diffusion_v1_ranked_samples/k16",
        ],
    )
    parser.add_argument(
        "--output_csv",
        default="data/cartesian_expert_dataset_v3/diffusion_v1_ranked_samples/best_of_k_summary.csv",
    )
    return parser.parse_args()


def summarize_root(root: Path) -> dict | None:
    csv_path = root / "diffusion_v1_best_per_path.csv"
    if not csv_path.exists():
        print(f"WARNING: missing {csv_path}")
        return None

    df = pd.read_csv(csv_path)
    if df.empty:
        print(f"WARNING: empty {csv_path}")
        return {
            "root": str(root),
            "evaluated_paths": 0,
            "accepted_paths": 0,
            "mean_path_error": float("nan"),
            "mean_mean_error": float("nan"),
            "mean_max_error": float("nan"),
            "worst_max_error": float("nan"),
            "mean_total_cost": float("nan"),
            "mean_joint_acceleration_cost": float("nan"),
        }

    return {
        "root": str(root),
        "evaluated_paths": int(len(df)),
        "accepted_paths": int(df["accepted"].sum()),
        "mean_path_error": float(df["path_error"].mean()),
        "mean_mean_error": float(df["mean_error"].mean()),
        "mean_max_error": float(df["max_error"].mean()),
        "worst_max_error": float(df["max_error"].max()),
        "mean_total_cost": float(df["total_cost"].mean()),
        "mean_joint_acceleration_cost": float(df["joint_acceleration_cost"].mean()),
    }


def print_summary(row: dict) -> None:
    print(f"root: {row['root']}")
    print(f"evaluated paths: {row['evaluated_paths']}")
    print(f"accepted paths: {row['accepted_paths']}")
    print(f"mean path_error: {row['mean_path_error']:.6f}")
    print(f"mean mean_error: {row['mean_mean_error']:.6f}")
    print(f"mean max_error: {row['mean_max_error']:.6f}")
    print(f"worst max_error: {row['worst_max_error']:.6f}")
    print(f"mean total_cost: {row['mean_total_cost']:.6f}")
    print(f"mean joint_acceleration_cost: {row['mean_joint_acceleration_cost']:.6f}")
    print("")


def main() -> None:
    args = parse_args()
    rows = []
    for root_text in args.roots:
        row = summarize_root(Path(root_text))
        if row is None:
            continue
        rows.append(row)
        print_summary(row)

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        rows,
        columns=[
            "root",
            "evaluated_paths",
            "accepted_paths",
            "mean_path_error",
            "mean_mean_error",
            "mean_max_error",
            "worst_max_error",
            "mean_total_cost",
            "mean_joint_acceleration_cost",
        ],
    ).to_csv(output_csv, index=False)
    print(f"Wrote combined summary: {output_csv}")


if __name__ == "__main__":
    main()
