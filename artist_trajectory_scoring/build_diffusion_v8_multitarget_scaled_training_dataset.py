#!/usr/bin/env python3
"""Build the path-disjoint v8 multitarget, scale-conditioned dataset."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple, cast

import numpy as np
import pandas as pd


HORIZON = 32
JOINT_DIM = 6
NORMALIZATION_EPSILON = 1.0e-8
OUTPUT_FILENAMES = (
    "train_windows.npz",
    "validation_windows.npz",
    "normalization_stats.npz",
    "dataset_metadata.json",
    "train_rows.csv",
    "validation_rows.csv",
    "path_split.csv",
    "targets_per_window_summary.csv",
    "scale_distribution_by_split.csv",
    "dataset_build_summary.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a path-disjoint v8 scale-conditioned diffusion dataset."
    )
    parser.add_argument("--targets_npz", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--validation_path_count", type=int, default=20)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.validation_path_count < 1:
        raise ValueError("--validation_path_count must be at least 1")


def decode_strings(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            value.decode("utf-8", errors="strict")
            if isinstance(value, bytes)
            else str(value)
            for value in np.asarray(values).reshape(-1)
        ],
        dtype=str,
    )


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def load_targets(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        data = {key: np.asarray(archive[key]) for key in archive.files}
    required = (
        "path_name", "window_start", "base_target_id", "target_id",
        "candidate_method", "target_scale", "condition", "residual_target",
        "cartesian_improvement_m", "delta_score",
        "normalized_prefix_diversity_to_nearest",
        "normalized_full_diversity_to_nearest", "condition_feature_names",
    )
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"selected_targets.npz is missing keys: {missing}")
    for key in (
        "path_name", "base_target_id", "target_id", "candidate_method",
        "condition_feature_names",
    ):
        data[key] = decode_strings(data[key])
    count = len(data["path_name"])
    row_keys = [key for key, value in data.items() if value.ndim > 0 and key != "condition_feature_names"]
    inconsistent = {key: len(data[key]) for key in row_keys if len(data[key]) != count}
    if inconsistent:
        raise ValueError(f"Target NPZ row counts are inconsistent: {inconsistent}")
    if data["condition"].ndim != 3 or data["condition"].shape[1] != HORIZON:
        raise ValueError("condition must have shape (N,32,C)")
    if data["residual_target"].shape != (count, HORIZON, JOINT_DIM):
        raise ValueError("residual_target must have shape (N,32,6)")
    if len(data["condition_feature_names"]) != data["condition"].shape[-1]:
        raise ValueError("condition_feature_names does not match condition dimension")
    numeric = [value for value in data.values() if np.issubdtype(value.dtype, np.number)]
    if any(not np.all(np.isfinite(value)) for value in numeric):
        raise ValueError("Target NPZ contains NaN or infinity")
    if np.any(data["target_scale"] <= 0.0):
        raise ValueError("target_scale is missing or non-positive")
    if len(set(data["target_id"].tolist())) != count:
        raise ValueError("target_id values are not unique")
    return data


def deterministic_path_split(
    path_names: Sequence[str], validation_count: int, seed: int
) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    ordered = np.asarray(sorted(set(str(name) for name in path_names)), dtype=str)
    if validation_count >= len(ordered):
        raise ValueError(
            f"validation_path_count={validation_count} leaves no training paths "
            f"from {len(ordered)} total paths"
        )
    generator = np.random.default_rng(seed)
    permutation = generator.permutation(len(ordered))
    validation = tuple(sorted(ordered[permutation[:validation_count]].tolist()))
    training = tuple(sorted(ordered[permutation[validation_count:]].tolist()))
    if set(training) & set(validation):
        raise AssertionError("Training and validation path sets overlap")
    return training, validation


def append_scale_condition(
    condition: np.ndarray, target_scale: np.ndarray
) -> np.ndarray:
    repeated = np.repeat(
        np.asarray(target_scale, dtype=np.float64).reshape(-1, 1, 1),
        condition.shape[1],
        axis=1,
    )
    combined = np.concatenate((condition.astype(np.float64), repeated), axis=2)
    if not np.allclose(combined[:, :, -1], target_scale[:, None]):
        raise AssertionError("Appended target_scale condition is inconsistent")
    return combined


def training_normalization(
    condition: np.ndarray, residual: np.ndarray
) -> Dict[str, np.ndarray]:
    condition_mean = np.mean(condition, axis=(0, 1))
    condition_std_raw = np.std(condition, axis=(0, 1))
    residual_mean = np.mean(residual, axis=(0, 1))
    residual_std_raw = np.std(residual, axis=(0, 1))
    condition_std = np.where(
        condition_std_raw > NORMALIZATION_EPSILON, condition_std_raw, 1.0
    )
    residual_std = np.where(
        residual_std_raw > NORMALIZATION_EPSILON, residual_std_raw, 1.0
    )
    # Scale remains raw and interpretable in both condition and condition_norm.
    condition_mean[-1] = 0.0
    condition_std[-1] = 1.0
    return {
        "condition_mean": condition_mean,
        "condition_std": condition_std,
        "condition_std_before_floor": condition_std_raw,
        "residual_mean": residual_mean,
        "residual_std": residual_std,
        "residual_std_before_floor": residual_std_raw,
    }


def quality_weights(
    path_names: np.ndarray,
    window_starts: np.ndarray,
    improvement: np.ndarray,
    delta_score: np.ndarray,
) -> Dict[str, np.ndarray]:
    positive_improvement = np.maximum(np.asarray(improvement, dtype=np.float64), 0.0)
    score_gain = np.maximum(-np.asarray(delta_score, dtype=np.float64), 0.0)
    improvement_reference = max(float(np.median(positive_improvement)), 1.0e-12)
    score_reference = max(float(np.median(score_gain)), 1.0e-12)
    quality_raw = 0.5 * (
        positive_improvement / improvement_reference + score_gain / score_reference
    )
    quality_clipped = np.clip(quality_raw, 0.25, 4.0)
    quality = quality_clipped / max(float(np.mean(quality_clipped)), 1.0e-12)
    identities = [(str(path), int(start)) for path, start in zip(path_names, window_starts)]
    counts: Dict[Tuple[str, int], int] = {}
    for identity in identities:
        counts[identity] = counts.get(identity, 0) + 1
    window_balance = np.asarray([1.0 / counts[identity] for identity in identities])
    combined_raw = quality * window_balance
    combined = combined_raw / max(float(np.mean(combined_raw)), 1.0e-12)
    return {
        "quality_weight": quality,
        "window_balance_weight": window_balance,
        "combined_sample_weight": combined,
        "quality_improvement_reference_m": np.asarray(improvement_reference),
        "quality_delta_score_reference": np.asarray(score_reference),
    }


def split_arrays(
    source: Mapping[str, np.ndarray],
    indices: np.ndarray,
    condition: np.ndarray,
    normalization: Mapping[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    residual = source["residual_target"][indices].astype(np.float64)
    selected_condition = condition[indices].astype(np.float64)
    condition_norm = (
        selected_condition - normalization["condition_mean"].reshape(1, 1, -1)
    ) / normalization["condition_std"].reshape(1, 1, -1)
    residual_norm = (
        residual - normalization["residual_mean"].reshape(1, 1, JOINT_DIM)
    ) / normalization["residual_std"].reshape(1, 1, JOINT_DIM)
    weights = quality_weights(
        source["path_name"][indices],
        source["window_start"][indices],
        source["cartesian_improvement_m"][indices],
        source["delta_score"][indices],
    )
    arrays: Dict[str, np.ndarray] = {
        "condition": selected_condition.astype(np.float32),
        "condition_norm": condition_norm.astype(np.float32),
        "residual_q": residual.astype(np.float32),
        "residual_q_norm": residual_norm.astype(np.float32),
        "residual_target": residual.astype(np.float32),
        "path_names": source["path_name"][indices],
        "path_name": source["path_name"][indices],
        "window_start_indices": source["window_start"][indices].astype(np.int64),
        "window_start": source["window_start"][indices].astype(np.int64),
        "base_target_id": source["base_target_id"][indices],
        "target_id": source["target_id"][indices],
        "target_scale": source["target_scale"][indices].astype(np.float32),
        "candidate_method": source["candidate_method"][indices],
        "cartesian_improvement_m": source["cartesian_improvement_m"][indices].astype(np.float32),
        "delta_score": source["delta_score"][indices].astype(np.float32),
        "normalized_prefix_diversity_to_nearest": source["normalized_prefix_diversity_to_nearest"][indices].astype(np.float32),
        "normalized_full_diversity_to_nearest": source["normalized_full_diversity_to_nearest"][indices].astype(np.float32),
        "quality_weight": weights["quality_weight"].astype(np.float32),
        "window_balance_weight": weights["window_balance_weight"].astype(np.float32),
        "combined_sample_weight": weights["combined_sample_weight"].astype(np.float32),
        "sample_weight": weights["combined_sample_weight"].astype(np.float32),
        "condition_mean": normalization["condition_mean"].astype(np.float32),
        "condition_std": normalization["condition_std"].astype(np.float32),
        "residual_mean": normalization["residual_mean"].astype(np.float32),
        "residual_std": normalization["residual_std"].astype(np.float32),
    }
    for key in (
        "prior_q", "target_q", "desired_cartesian_window", "execution_horizon",
        "restart_index", "prior_prefix_cartesian_mean_error_m",
        "target_prefix_cartesian_mean_error_m", "prior_robot_aware_score",
        "target_robot_aware_score", "hard_safe",
    ):
        if key in source:
            arrays[key] = source[key][indices]
    return arrays


def row_frame(
    source: Mapping[str, np.ndarray], indices: np.ndarray, split: str,
    arrays: Mapping[str, np.ndarray],
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "split": split,
            "row_index_in_source": indices,
            "path_name": source["path_name"][indices],
            "window_start": source["window_start"][indices],
            "base_target_id": source["base_target_id"][indices],
            "target_id": source["target_id"][indices],
            "target_scale": source["target_scale"][indices],
            "candidate_method": source["candidate_method"][indices],
            "cartesian_improvement_m": source["cartesian_improvement_m"][indices],
            "delta_score": source["delta_score"][indices],
            "normalized_prefix_diversity_to_nearest": source["normalized_prefix_diversity_to_nearest"][indices],
            "normalized_full_diversity_to_nearest": source["normalized_full_diversity_to_nearest"][indices],
            "quality_weight": arrays["quality_weight"],
            "window_balance_weight": arrays["window_balance_weight"],
            "combined_sample_weight": arrays["combined_sample_weight"],
        }
    )


def validate_dataset(
    source: Mapping[str, np.ndarray],
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    train_arrays: Mapping[str, np.ndarray],
    validation_arrays: Mapping[str, np.ndarray],
    normalization: Mapping[str, np.ndarray],
) -> None:
    train_paths = set(source["path_name"][train_indices].tolist())
    validation_paths = set(source["path_name"][validation_indices].tolist())
    if train_paths & validation_paths:
        raise AssertionError("Train and validation paths overlap")
    split_by_window: Dict[Tuple[str, int], str] = {}
    for label, indices in (("train", train_indices), ("validation", validation_indices)):
        for index in indices:
            key = (str(source["path_name"][index]), int(source["window_start"][index]))
            previous = split_by_window.setdefault(key, label)
            if previous != label:
                raise AssertionError(f"Window {key} is split across datasets")
    for label, arrays in (("train", train_arrays), ("validation", validation_arrays)):
        count = len(arrays["path_name"])
        if arrays["condition"].shape[:2] != (count, HORIZON):
            raise AssertionError(f"{label} condition dimensions are inconsistent")
        if arrays["residual_q"].shape != (count, HORIZON, JOINT_DIM):
            raise AssertionError(f"{label} target dimensions are inconsistent")
        for key, value in arrays.items():
            if np.issubdtype(value.dtype, np.number) and not np.all(np.isfinite(value)):
                raise AssertionError(f"{label} key {key} contains NaN or infinity")
        if not np.allclose(arrays["condition"][:, :, -1], arrays["target_scale"][:, None]):
            raise AssertionError(f"{label} target_scale condition is inconsistent")
        if not np.allclose(arrays["condition_norm"][:, :, -1], arrays["target_scale"][:, None]):
            raise AssertionError(f"{label} normalized target_scale is not raw")
    expected_condition_mean = np.mean(train_arrays["condition"].astype(np.float64), axis=(0, 1))
    expected_condition_mean[-1] = 0.0
    if not np.allclose(normalization["condition_mean"], expected_condition_mean, atol=1e-6):
        raise AssertionError("Condition normalization was not derived from training rows")
    expected_residual_mean = np.mean(train_arrays["residual_q"].astype(np.float64), axis=(0, 1))
    if not np.allclose(normalization["residual_mean"], expected_residual_mean, atol=1e-6):
        raise AssertionError("Residual normalization was not derived from training rows")


def main() -> int:
    args = parse_args()
    validate_args(args)
    existing = [args.output_dir / name for name in OUTPUT_FILENAMES if (args.output_dir / name).exists()]
    if existing and not args.overwrite:
        raise FileExistsError(f"Outputs already exist: {existing}; pass --overwrite")
    source = load_targets(args.targets_npz)
    train_paths, validation_paths = deterministic_path_split(
        source["path_name"].tolist(), args.validation_path_count, args.split_seed
    )
    train_mask = np.isin(source["path_name"], np.asarray(train_paths))
    validation_mask = np.isin(source["path_name"], np.asarray(validation_paths))
    train_indices = np.flatnonzero(train_mask)
    validation_indices = np.flatnonzero(validation_mask)
    if len(train_indices) + len(validation_indices) != len(source["path_name"]):
        raise AssertionError("Some target rows were lost during path splitting")
    raw_condition = append_scale_condition(source["condition"], source["target_scale"])
    normalization = training_normalization(
        raw_condition[train_indices], source["residual_target"][train_indices]
    )
    feature_names = np.concatenate(
        (source["condition_feature_names"], np.asarray(["target_scale"]))
    )
    train_arrays = split_arrays(
        source, train_indices, raw_condition, normalization
    )
    validation_arrays = split_arrays(
        source, validation_indices, raw_condition, normalization
    )
    for arrays in (train_arrays, validation_arrays):
        arrays["condition_feature_names"] = feature_names
        arrays["residual_feature_names"] = np.asarray(
            [f"joint{index}" for index in range(1, JOINT_DIM + 1)]
        )
        arrays["condition_dim"] = np.asarray(raw_condition.shape[-1], dtype=np.int64)
        arrays["target_dim"] = np.asarray(JOINT_DIM, dtype=np.int64)
        arrays["horizon"] = np.asarray(HORIZON, dtype=np.int64)
    validate_dataset(
        source, train_indices, validation_indices,
        train_arrays, validation_arrays, normalization,
    )
    normalization_arrays = {
        **{key: value.astype(np.float32) for key, value in normalization.items()},
        "condition_feature_names": feature_names,
        "residual_feature_names": np.asarray([f"joint{index}" for index in range(1, 7)]),
        "condition_dim": np.asarray(raw_condition.shape[-1], dtype=np.int64),
        "v7_condition_dim": np.asarray(source["condition"].shape[-1], dtype=np.int64),
        "target_dim": np.asarray(JOINT_DIM, dtype=np.int64),
        "horizon": np.asarray(HORIZON, dtype=np.int64),
        "normalization_epsilon": np.asarray(NORMALIZATION_EPSILON),
        "target_scale_mean": np.asarray(0.0, dtype=np.float32),
        "target_scale_std": np.asarray(1.0, dtype=np.float32),
    }
    train_frame = row_frame(source, train_indices, "train", train_arrays)
    validation_frame = row_frame(
        source, validation_indices, "validation", validation_arrays
    )
    path_split_frame = pd.DataFrame(
        [
            {"path_name": path, "split": "train", "split_seed": args.split_seed}
            for path in train_paths
        ]
        + [
            {"path_name": path, "split": "validation", "split_seed": args.split_seed}
            for path in validation_paths
        ]
    ).sort_values("path_name").reset_index(drop=True)
    all_rows = pd.concat((train_frame, validation_frame), ignore_index=True)
    targets_per_window = (
        all_rows.groupby(["split", "path_name", "window_start"], sort=True)
        .agg(
            target_count=("target_id", "count"),
            distinct_base_target_count=("base_target_id", "nunique"),
            distinct_scale_count=("target_scale", "nunique"),
            best_cartesian_improvement_m=("cartesian_improvement_m", "max"),
            best_delta_score=("delta_score", "min"),
        )
        .reset_index()
    )
    scale_distribution = pd.DataFrame(
        [
            {
                "split": split,
                "target_scale": float(cast(Any, scale)),
                "target_count": len(group),
                "path_count": group["path_name"].nunique(),
                "window_count": group[["path_name", "window_start"]].drop_duplicates().shape[0],
                "mean_cartesian_improvement_m": float(group["cartesian_improvement_m"].mean()),
                "median_cartesian_improvement_m": float(group["cartesian_improvement_m"].median()),
                "mean_delta_score": float(group["delta_score"].mean()),
                "median_delta_score": float(group["delta_score"].median()),
            }
            for (split, scale), group in all_rows.groupby(
                ["split", "target_scale"], sort=True
            )
        ]
    )
    source_metadata_path = args.targets_npz.parent / "target_generation_summary.json"
    source_metadata: Mapping[str, Any] = {}
    if source_metadata_path.is_file():
        with source_metadata_path.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        if isinstance(loaded, Mapping):
            source_metadata = loaded
    metadata = {
        "classification": "READY_FOR_V8_TRAINING",
        "arguments": vars(args),
        "source_targets_npz": str(args.targets_npz.resolve()),
        "source_target_generation_metadata": source_metadata,
        "split": {
            "unit": "path_name",
            "seed": args.split_seed,
            "train_path_names": list(train_paths),
            "validation_path_names": list(validation_paths),
            "path_disjoint": True,
            "window_disjoint": True,
        },
        "condition": {
            "v7_condition_dim": int(source["condition"].shape[-1]),
            "v8_condition_dim": int(raw_condition.shape[-1]),
            "feature_names": feature_names.tolist(),
            "appended_feature": "target_scale",
            "target_scale_normalization": "raw value; mean=0 and std=1",
        },
        "target": {
            "name": "scaled residual_target",
            "shape": [HORIZON, JOINT_DIM],
        },
        "normalization": {
            "source_split": "training rows only",
            "axes": "row and horizon axes",
            "epsilon": NORMALIZATION_EPSILON,
            "zero_variance_replacement": 1.0,
            "statistics": {key: value.tolist() for key, value in normalization.items()},
        },
        "sample_weights": {
            "quality_weight": (
                "mean-normalized clip(0.5 * (cartesian_improvement / median_positive_"
                "improvement + -delta_score / median_score_gain), 0.25, 4.0)"
            ),
            "window_balance_weight": "1 / retained target count for path/window",
            "combined_sample_weight": (
                "mean-normalized quality_weight * window_balance_weight"
            ),
            "discard_policy": "weights never discard rows",
        },
        "counts": {
            "source_rows": len(source["path_name"]),
            "train_rows": len(train_indices),
            "validation_rows": len(validation_indices),
            "train_paths": len(train_paths),
            "validation_paths": len(validation_paths),
            "train_windows": train_frame[["path_name", "window_start"]].drop_duplicates().shape[0],
            "validation_windows": validation_frame[["path_name", "window_start"]].drop_duplicates().shape[0],
        },
        "compatibility": {
            "v7_loader_keys_preserved": [
                "condition", "condition_norm", "residual_q", "residual_q_norm",
                "path_names", "window_start_indices", "sample_weight",
            ],
            "training_script_modified": False,
            "required_model_condition_dim": int(raw_condition.shape[-1]),
        },
    }
    summary = {
        "classification": "READY_FOR_V8_TRAINING",
        "counts": metadata["counts"],
        "condition_dim": int(raw_condition.shape[-1]),
        "target_shape": [HORIZON, JOINT_DIM],
        "scale_distribution": scale_distribution.to_dict(orient="records"),
        "targets_per_window_distribution": (
            targets_per_window.groupby(["split", "target_count"]).size()
            .rename("window_count").reset_index().to_dict(orient="records")
        ),
        "integrity_checks": {
            "path_overlap_count": 0,
            "window_split_count": 0,
            "normalization_source": "train only",
            "nonfinite_count": 0,
            "metadata_counts_match": True,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_npz(args.output_dir / "train_windows.npz", train_arrays)
    atomic_npz(args.output_dir / "validation_windows.npz", validation_arrays)
    atomic_npz(args.output_dir / "normalization_stats.npz", normalization_arrays)
    atomic_json(args.output_dir / "dataset_metadata.json", metadata)
    atomic_csv(args.output_dir / "train_rows.csv", train_frame)
    atomic_csv(args.output_dir / "validation_rows.csv", validation_frame)
    atomic_csv(args.output_dir / "path_split.csv", path_split_frame)
    atomic_csv(args.output_dir / "targets_per_window_summary.csv", targets_per_window)
    atomic_csv(args.output_dir / "scale_distribution_by_split.csv", scale_distribution)
    atomic_json(args.output_dir / "dataset_build_summary.json", summary)
    print(
        f"train: paths={len(train_paths)}, rows={len(train_indices)}, "
        f"condition={train_arrays['condition'].shape}, target={train_arrays['residual_q'].shape}"
    )
    print(
        f"validation: paths={len(validation_paths)}, rows={len(validation_indices)}, "
        f"condition={validation_arrays['condition'].shape}, target={validation_arrays['residual_q'].shape}"
    )
    print(f"output directory: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
