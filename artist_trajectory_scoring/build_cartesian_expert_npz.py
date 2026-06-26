#!/usr/bin/env python3
"""
Build a trainable NPZ dataset from Cartesian expert trajectory folders.

Expected folder layout:
    data/cartesian_expert_dataset/
      line_001/
        desired_path.csv   # t,x,y,z
        expert_q.csv       # t,q1,q2,q3,q4,q5,q6
      arc_001/
        desired_path.csv
        expert_q.csv

Output NPZ keys are compatible with train_time_conditioned_mlp.py:
    desired_paths: (N, T, 3)
    actions:       (N, T, 6)
    times:         (N, T)
    path_ids:      (N,)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


PATH_COLS = ["x", "y", "z"]
Q_COLS = ["q1", "q2", "q3", "q4", "q5", "q6"]


def require_columns(df: pd.DataFrame, path: Path, cols: List[str]) -> None:
    missing = [col for col in cols if col not in df.columns]
    if missing:
        raise ValueError(f"{path} missing required columns {missing}. Found: {list(df.columns)}")


def load_episode(path_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    desired_csv = path_dir / "desired_path.csv"
    expert_q_csv = path_dir / "expert_q.csv"

    desired_df = pd.read_csv(desired_csv)
    q_df = pd.read_csv(expert_q_csv)

    require_columns(desired_df, desired_csv, ["t"] + PATH_COLS)
    require_columns(q_df, expert_q_csv, ["t"] + Q_COLS)

    if len(desired_df) != len(q_df):
        raise ValueError(
            f"Timestep mismatch in {path_dir.name}: "
            f"desired_path.csv has {len(desired_df)} rows, "
            f"expert_q.csv has {len(q_df)} rows"
        )

    desired_path = desired_df[PATH_COLS].to_numpy(dtype=np.float32)
    action = q_df[Q_COLS].to_numpy(dtype=np.float32)
    time = desired_df["t"].to_numpy(dtype=np.float32)
    return desired_path, action, time


def build_dataset(dataset_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    desired_paths = []
    actions = []
    times = []
    path_ids = []
    expected_timesteps = None

    for path_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
        desired_csv = path_dir / "desired_path.csv"
        expert_q_csv = path_dir / "expert_q.csv"

        if not desired_csv.exists() or not expert_q_csv.exists():
            print(
                f"Skipping {path_dir.name}: "
                f"missing {'desired_path.csv' if not desired_csv.exists() else 'expert_q.csv'}"
            )
            continue

        desired_path, action, time = load_episode(path_dir)

        if expected_timesteps is None:
            expected_timesteps = desired_path.shape[0]
        elif desired_path.shape[0] != expected_timesteps:
            raise ValueError(
                f"All episodes must have the same T for train_time_conditioned_mlp.py. "
                f"{path_dir.name} has T={desired_path.shape[0]}, expected T={expected_timesteps}."
            )

        desired_paths.append(desired_path)
        actions.append(action)
        times.append(time)
        path_ids.append(path_dir.name)
        print(f"Loaded {path_dir.name}: T={desired_path.shape[0]}")

    if not desired_paths:
        raise FileNotFoundError(
            f"No valid expert episodes found in {dataset_dir}. "
            "Each episode folder needs desired_path.csv and expert_q.csv."
        )

    return (
        np.stack(desired_paths, axis=0),
        np.stack(actions, axis=0),
        np.stack(times, axis=0),
        np.asarray(path_ids),
    )


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pack Cartesian expert subfolders into train_time_conditioned_mlp-compatible NPZ."
    )
    parser.add_argument(
        "--dataset_dir",
        type=Path,
        default=Path("data/cartesian_expert_dataset"),
        help="Directory containing path subfolders.",
    )
    parser.add_argument(
        "--output_npz",
        type=Path,
        default=None,
        help="Output NPZ path. Default: dataset_dir/cartesian_expert_episodes.npz",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    dataset_dir = args.dataset_dir
    output_npz = args.output_npz or dataset_dir / "cartesian_expert_episodes.npz"

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    desired_paths, actions, times, path_ids = build_dataset(dataset_dir)

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_npz,
        desired_paths=desired_paths,
        actions=actions,
        times=times,
        path_ids=path_ids,
    )

    print()
    print(f"Saved Cartesian expert NPZ dataset to: {output_npz}")
    print("Dataset summary:")
    print(f"  number of paths:      {desired_paths.shape[0]}")
    print(f"  timesteps per path:   {desired_paths.shape[1]}")
    print(f"  desired_paths shape:  {desired_paths.shape}")
    print(f"  actions shape:        {actions.shape}")
    print(f"  times shape:          {times.shape}")
    print(f"  path_ids:             {list(path_ids)}")


if __name__ == "__main__":
    main()
