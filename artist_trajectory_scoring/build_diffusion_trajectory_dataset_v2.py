#!/usr/bin/env python3
"""Build v2 conditional diffusion trajectory datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


EPS = 1e-8
EXPECTED_T = 100
Q_COLUMNS = [f"q{i}" for i in range(1, 7)]
XYZ_COLUMNS = ["x", "y", "z"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_dir", default="data/cartesian_expert_dataset_v3/experts/train")
    parser.add_argument("--test_dir", default="data/cartesian_expert_dataset_v3/experts/test")
    parser.add_argument("--output_dir", default="data/cartesian_expert_dataset_v3/diffusion_v2")
    return parser.parse_args()


def finite_difference_xyz(desired_df: pd.DataFrame, xyz: np.ndarray) -> np.ndarray:
    if "t" in desired_df.columns:
        t = desired_df["t"].to_numpy(dtype=np.float64)
        if t.shape[0] == xyz.shape[0] and np.all(np.isfinite(t)) and np.unique(t).shape[0] == t.shape[0]:
            return np.gradient(xyz, t, axis=0)
    return np.gradient(xyz, axis=0)


def load_path_folder(path_dir: Path) -> dict[str, np.ndarray | str] | None:
    desired_csv = path_dir / "desired_path.csv"
    expert_q_csv = path_dir / "expert_q.csv"

    if not desired_csv.exists() or not expert_q_csv.exists():
        print(f"WARNING: skipping {path_dir.name}: missing desired_path.csv or expert_q.csv")
        return None

    try:
        desired_df = pd.read_csv(desired_csv)
        expert_q_df = pd.read_csv(expert_q_csv)
    except Exception as exc:
        print(f"WARNING: skipping {path_dir.name}: failed to read CSV files: {exc}")
        return None

    missing_desired = [col for col in XYZ_COLUMNS if col not in desired_df.columns]
    missing_q = [col for col in Q_COLUMNS if col not in expert_q_df.columns]
    if missing_desired or missing_q:
        print(
            f"WARNING: skipping {path_dir.name}: missing columns "
            f"desired={missing_desired}, expert_q={missing_q}"
        )
        return None

    desired_path = desired_df[XYZ_COLUMNS].to_numpy(dtype=np.float32)
    expert_q = expert_q_df[Q_COLUMNS].to_numpy(dtype=np.float32)

    if desired_path.shape != (EXPECTED_T, 3) or expert_q.shape != (EXPECTED_T, 6):
        print(
            f"WARNING: skipping {path_dir.name}: wrong shapes "
            f"desired_path={desired_path.shape}, expert_q={expert_q.shape}"
        )
        return None

    if not np.all(np.isfinite(desired_path)) or not np.all(np.isfinite(expert_q)):
        print(f"WARNING: skipping {path_dir.name}: found NaN or inf values")
        return None

    q_start = expert_q[0].copy()
    delta_q = expert_q - q_start[None, :]
    xyz_velocity = finite_difference_xyz(desired_df, desired_path).astype(np.float32)
    normalized_t = np.linspace(0.0, 1.0, EXPECTED_T, dtype=np.float32)[:, None]
    q_start_repeated = np.repeat(q_start[None, :], EXPECTED_T, axis=0)
    condition_features = np.concatenate(
        [desired_path, xyz_velocity, normalized_t, q_start_repeated],
        axis=1,
    ).astype(np.float32)

    if condition_features.shape != (EXPECTED_T, 13):
        print(f"WARNING: skipping {path_dir.name}: condition shape is {condition_features.shape}")
        return None

    if not np.all(np.isfinite(condition_features)) or not np.all(np.isfinite(delta_q)):
        print(f"WARNING: skipping {path_dir.name}: derived features contain NaN or inf")
        return None

    return {
        "condition_features": condition_features,
        "delta_q": delta_q.astype(np.float32),
        "desired_paths": desired_path.astype(np.float32),
        "expert_q": expert_q.astype(np.float32),
        "q_start": q_start.astype(np.float32),
        "path_names": path_dir.name,
    }


def build_split(split_dir: Path) -> dict[str, np.ndarray]:
    records = []
    skipped = 0

    if not split_dir.exists():
        print(f"WARNING: split directory does not exist: {split_dir}")
        skipped += 1
    else:
        for path_dir in sorted(p for p in split_dir.iterdir() if p.is_dir()):
            record = load_path_folder(path_dir)
            if record is None:
                skipped += 1
                continue
            records.append(record)

    print(f"{split_dir}: accepted={len(records)}, skipped={skipped}")
    if not records:
        raise RuntimeError(f"No valid path folders found in {split_dir}")

    return {
        "condition_features": np.stack([r["condition_features"] for r in records]).astype(np.float32),
        "delta_q": np.stack([r["delta_q"] for r in records]).astype(np.float32),
        "desired_paths": np.stack([r["desired_paths"] for r in records]).astype(np.float32),
        "expert_q": np.stack([r["expert_q"] for r in records]).astype(np.float32),
        "q_start": np.stack([r["q_start"] for r in records]).astype(np.float32),
        "path_names": np.array([r["path_names"] for r in records], dtype=str),
    }


def compute_norm_stats(train_data: dict[str, np.ndarray]) -> dict[str, list[float]]:
    condition = train_data["condition_features"]
    delta_q = train_data["delta_q"]
    return {
        "condition_mean": condition.mean(axis=(0, 1)).astype(float).tolist(),
        "condition_std": (condition.std(axis=(0, 1)) + EPS).astype(float).tolist(),
        "delta_q_mean": delta_q.mean(axis=(0, 1)).astype(float).tolist(),
        "delta_q_std": (delta_q.std(axis=(0, 1)) + EPS).astype(float).tolist(),
        "condition_dim": 13,
        "target_dim": 6,
        "epsilon": EPS,
    }


def apply_normalization(data: dict[str, np.ndarray], stats: dict[str, list[float]]) -> dict[str, np.ndarray]:
    condition_mean = np.asarray(stats["condition_mean"], dtype=np.float32)
    condition_std = np.asarray(stats["condition_std"], dtype=np.float32)
    delta_q_mean = np.asarray(stats["delta_q_mean"], dtype=np.float32)
    delta_q_std = np.asarray(stats["delta_q_std"], dtype=np.float32)

    output = dict(data)
    output["condition_features_norm"] = (
        (data["condition_features"] - condition_mean[None, None, :]) / condition_std[None, None, :]
    ).astype(np.float32)
    output["delta_q_norm"] = (
        (data["delta_q"] - delta_q_mean[None, None, :]) / delta_q_std[None, None, :]
    ).astype(np.float32)
    return output


def save_npz(path: Path, data: dict[str, np.ndarray]) -> None:
    np.savez_compressed(path, **data)
    print(f"Wrote {path}")


def main() -> None:
    args = parse_args()
    train_dir = Path(args.train_dir)
    test_dir = Path(args.test_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_data = build_split(train_dir)
    test_data = build_split(test_dir)
    stats = compute_norm_stats(train_data)
    train_data = apply_normalization(train_data, stats)
    test_data = apply_normalization(test_data, stats)

    train_out = output_dir / "diffusion_train_v2.npz"
    test_out = output_dir / "diffusion_test_v2.npz"
    stats_out = output_dir / "diffusion_norm_stats_v2.json"

    save_npz(train_out, train_data)
    save_npz(test_out, test_data)
    with stats_out.open("w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print(f"Wrote {stats_out}")
    print(f"Summary: train={len(train_data['path_names'])}, test={len(test_data['path_names'])}")


if __name__ == "__main__":
    main()
