#!/usr/bin/env python3
"""
Build normalized train/test NPZ files for conditional trajectory diffusion.

Expected input layout:
    data/cartesian_expert_dataset_v3/experts/train/path_0001/
      desired_path.csv   # t,x,y,z
      expert_q.csv       # t,q1,q2,q3,q4,q5,q6

Outputs:
    diffusion_train.npz
    diffusion_test.npz
    diffusion_norm_stats.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


PATH_COLS = ["x", "y", "z"]
Q_COLS = ["q1", "q2", "q3", "q4", "q5", "q6"]
EXPECTED_TIMESTEPS = 100
STD_EPS = 1e-8


def missing_columns(df: pd.DataFrame, cols: List[str]) -> List[str]:
    return [col for col in cols if col not in df.columns]


def load_episode(path_dir: Path) -> Optional[Tuple[np.ndarray, np.ndarray, str]]:
    desired_csv = path_dir / "desired_path.csv"
    expert_q_csv = path_dir / "expert_q.csv"

    if not desired_csv.exists():
        print(f"[WARN] Skipping {path_dir.name}: missing desired_path.csv")
        return None

    if not expert_q_csv.exists():
        print(f"[WARN] Skipping {path_dir.name}: missing expert_q.csv")
        return None

    try:
        desired_df = pd.read_csv(desired_csv)
        q_df = pd.read_csv(expert_q_csv)
    except Exception as exc:
        print(f"[WARN] Skipping {path_dir.name}: failed to read CSV ({exc})")
        return None

    desired_missing = missing_columns(desired_df, ["t"] + PATH_COLS)
    if desired_missing:
        print(
            f"[WARN] Skipping {path_dir.name}: desired_path.csv missing columns "
            f"{desired_missing}"
        )
        return None

    q_missing = missing_columns(q_df, ["t"] + Q_COLS)
    if q_missing:
        print(f"[WARN] Skipping {path_dir.name}: expert_q.csv missing columns {q_missing}")
        return None

    desired_path = desired_df[PATH_COLS].to_numpy(dtype=np.float32)
    expert_q = q_df[Q_COLS].to_numpy(dtype=np.float32)

    if desired_path.shape != (EXPECTED_TIMESTEPS, len(PATH_COLS)):
        print(
            f"[WARN] Skipping {path_dir.name}: desired path shape "
            f"{desired_path.shape}, expected ({EXPECTED_TIMESTEPS}, {len(PATH_COLS)})"
        )
        return None

    if expert_q.shape != (EXPECTED_TIMESTEPS, len(Q_COLS)):
        print(
            f"[WARN] Skipping {path_dir.name}: expert_q shape "
            f"{expert_q.shape}, expected ({EXPECTED_TIMESTEPS}, {len(Q_COLS)})"
        )
        return None

    if np.isnan(desired_path).any():
        print(f"[WARN] Skipping {path_dir.name}: desired_path.csv contains NaN values")
        return None

    if np.isnan(expert_q).any():
        print(f"[WARN] Skipping {path_dir.name}: expert_q.csv contains NaN values")
        return None

    return desired_path, expert_q, path_dir.name


def load_split(split_dir: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    desired_paths = []
    expert_q = []
    path_names = []
    skipped_count = 0

    if not split_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {split_dir}")

    path_dirs = sorted(path for path in split_dir.iterdir() if path.is_dir())

    for path_dir in path_dirs:
        episode = load_episode(path_dir)
        if episode is None:
            skipped_count += 1
            continue

        desired_path, q_traj, path_name = episode
        desired_paths.append(desired_path)
        expert_q.append(q_traj)
        path_names.append(path_name)

    if not desired_paths:
        raise RuntimeError(f"No valid episodes found in {split_dir}")

    return (
        np.stack(desired_paths, axis=0),
        np.stack(expert_q, axis=0),
        np.asarray(path_names),
        skipped_count,
    )


def compute_norm_stats(
    desired_paths: np.ndarray,
    expert_q: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    path_mean = desired_paths.mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
    path_std = desired_paths.std(axis=(0, 1), dtype=np.float64).astype(np.float32) + STD_EPS
    q_mean = expert_q.mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
    q_std = expert_q.std(axis=(0, 1), dtype=np.float64).astype(np.float32) + STD_EPS
    return path_mean, path_std, q_mean, q_std


def normalize(
    desired_paths: np.ndarray,
    expert_q: np.ndarray,
    path_mean: np.ndarray,
    path_std: np.ndarray,
    q_mean: np.ndarray,
    q_std: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    desired_paths_norm = (desired_paths - path_mean.reshape(1, 1, -1)) / path_std.reshape(
        1, 1, -1
    )
    expert_q_norm = (expert_q - q_mean.reshape(1, 1, -1)) / q_std.reshape(1, 1, -1)
    return desired_paths_norm.astype(np.float32), expert_q_norm.astype(np.float32)


def save_npz(
    output_path: Path,
    desired_paths: np.ndarray,
    expert_q: np.ndarray,
    path_names: np.ndarray,
    desired_paths_norm: np.ndarray,
    expert_q_norm: np.ndarray,
) -> None:
    np.savez_compressed(
        output_path,
        desired_paths=desired_paths,
        expert_q=expert_q,
        path_names=path_names,
        desired_paths_norm=desired_paths_norm,
        expert_q_norm=expert_q_norm,
    )


def save_norm_stats(
    output_path: Path,
    path_mean: np.ndarray,
    path_std: np.ndarray,
    q_mean: np.ndarray,
    q_std: np.ndarray,
) -> None:
    stats = {
        "path_mean": path_mean.tolist(),
        "path_std": path_std.tolist(),
        "q_mean": q_mean.tolist(),
        "q_std": q_std.tolist(),
        "epsilon": STD_EPS,
        "path_columns": PATH_COLS,
        "q_columns": Q_COLS,
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
        f.write("\n")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build normalized trajectory diffusion train/test datasets."
    )
    parser.add_argument(
        "--train_dir",
        type=Path,
        default=Path("data/cartesian_expert_dataset_v3/experts/train"),
        help="Directory containing training path folders.",
    )
    parser.add_argument(
        "--test_dir",
        type=Path,
        default=Path("data/cartesian_expert_dataset_v3/experts/test"),
        help="Directory containing test path folders.",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/cartesian_expert_dataset_v3"),
        help="Directory where diffusion NPZ files and normalization stats are written.",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()

    train_desired, train_q, train_names, train_skipped = load_split(args.train_dir)
    test_desired, test_q, test_names, test_skipped = load_split(args.test_dir)

    path_mean, path_std, q_mean, q_std = compute_norm_stats(train_desired, train_q)
    train_desired_norm, train_q_norm = normalize(
        train_desired, train_q, path_mean, path_std, q_mean, q_std
    )
    test_desired_norm, test_q_norm = normalize(
        test_desired, test_q, path_mean, path_std, q_mean, q_std
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_npz = args.output_dir / "diffusion_train.npz"
    test_npz = args.output_dir / "diffusion_test.npz"
    stats_json = args.output_dir / "diffusion_norm_stats.json"

    save_npz(train_npz, train_desired, train_q, train_names, train_desired_norm, train_q_norm)
    save_npz(test_npz, test_desired, test_q, test_names, test_desired_norm, test_q_norm)
    save_norm_stats(stats_json, path_mean, path_std, q_mean, q_std)

    print()
    print("Diffusion trajectory dataset build complete.")
    print(f"Accepted train paths: {len(train_names)}")
    print(f"Skipped train paths:  {train_skipped}")
    print(f"Accepted test paths:  {len(test_names)}")
    print(f"Skipped test paths:   {test_skipped}")
    print()
    print(f"Train output: {train_npz}")
    print(f"Test output:  {test_npz}")
    print(f"Norm stats:   {stats_json}")
    print()
    print("Array shapes:")
    print(f"  train desired_paths:      {train_desired.shape}")
    print(f"  train expert_q:           {train_q.shape}")
    print(f"  train desired_paths_norm: {train_desired_norm.shape}")
    print(f"  train expert_q_norm:      {train_q_norm.shape}")
    print(f"  test desired_paths:       {test_desired.shape}")
    print(f"  test expert_q:            {test_q.shape}")
    print(f"  test desired_paths_norm:  {test_desired_norm.shape}")
    print(f"  test expert_q_norm:       {test_q_norm.shape}")


if __name__ == "__main__":
    main()
