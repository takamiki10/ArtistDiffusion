#!/usr/bin/env python3
"""Build the v7 cost-improving residual diffusion training dataset.

The selected-target archive has one row per target, so a conditional window can
appear more than once.  This builder deliberately retains that multiplicity,
while assigning every row 1 / targets_in_window training weight.

The 38-D condition is exactly the v6 strong-prior condition.  Although the v7
archive stores windows rather than complete paths, the overlapping windows
contain everything needed to recover q(t=0) and the desired-path finite
difference at every window boundary.  Inconsistent overlaps are rejected.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np


HORIZON = 32
JOINT_DIM = 6
CARTESIAN_DIM = 3
TRAJECTORY_LENGTH = 100
CONDITION_DIM = 38
NORMALIZATION_EPSILON = 1.0e-8
FLOAT_RTOL = 1.0e-5
FLOAT_ATOL = 1.0e-6
ZERO_ATOL = 1.0e-7

DEFAULT_INPUT = Path(
    "data/cartesian_expert_dataset_v3/"
    "diffusion_v7_cost_improving_residual_targets_100paths_fast/"
    "selected_targets.npz"
)
DEFAULT_OUTPUT = Path(
    "data/cartesian_expert_dataset_v3/"
    "diffusion_v7_cost_improving_training_dataset_100paths"
)
OUTPUT_FILENAMES = (
    "train_windows.npz",
    "validation_windows.npz",
    "normalization.npz",
    "split_manifest.json",
    "dataset_metadata.json",
    "dataset_summary.json",
)

CONDITION_FEATURE_LAYOUT = (
    "desired_xyz(3)",
    "desired_dxyz(3)",
    "progress(1)",
    "prior_q_start_repeated(6)",
    "prior_current_q_repeated(6)",
    "prior_q_window(6)",
    "prior_delta_from_start(6)",
    "prior_ee_xyz(3)",
    "prior_ee_error_xyz(3)",
    "prior_ee_error_norm(1)",
)
CONDITION_FEATURE_NAMES = (
    "desired_x", "desired_y", "desired_z",
    "desired_dx", "desired_dy", "desired_dz",
    "progress",
    *(f"prior_q_start_q{index}" for index in range(1, 7)),
    *(f"prior_current_q{index}" for index in range(1, 7)),
    *(f"prior_q{index}" for index in range(1, 7)),
    *(f"prior_delta_from_start_q{index}" for index in range(1, 7)),
    "prior_ee_x", "prior_ee_y", "prior_ee_z",
    "prior_ee_error_x", "prior_ee_error_y", "prior_ee_error_z",
    "prior_ee_error_norm",
)
RESIDUAL_FEATURE_NAMES = tuple(f"residual_q{index}" for index in range(1, 7))

REQUIRED_KEYS = (
    "path_names",
    "path_indices",
    "window_starts",
    "target_indices_within_window",
    "desired_path_window",
    "prior_q_window",
    "prior_ee_window",
    "residual_q_window",
    "candidate_q_window",
    "is_zero_residual",
    "improves_prior",
    "candidate_methods",
    "execution_horizon",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build path-disjoint v7 cost-improving residual windows."
    )
    parser.add_argument("--input_npz", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    counts = parser.add_mutually_exclusive_group()
    counts.add_argument("--train_path_count", type=int, default=None)
    counts.add_argument("--validation_path_count", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def decode_strings(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            item.decode("utf-8", errors="strict")
            if isinstance(item, bytes)
            else str(item)
            for item in np.asarray(values).reshape(-1)
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
    return value


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(json_safe(value), handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    temporary = path.with_name(f".{path.stem}.tmp.npz")
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def require_finite(name: str, values: np.ndarray) -> None:
    if not np.all(np.isfinite(values)):
        bad = np.argwhere(~np.isfinite(values))
        first = tuple(int(item) for item in bad[0]) if len(bad) else None
        raise ValueError(f"{name} contains NaN or infinity; first bad index={first}")


def load_and_validate(path: Path) -> Dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"Selected-target archive does not exist: {path}")
    with np.load(path, allow_pickle=True) as archive:
        missing = [key for key in REQUIRED_KEYS if key not in archive.files]
        if missing:
            raise KeyError(f"{path} is missing required arrays: {missing}")
        data = {key: np.asarray(archive[key]) for key in REQUIRED_KEYS}

    data["path_names"] = decode_strings(data["path_names"])
    data["candidate_methods"] = decode_strings(data["candidate_methods"])
    row_count = len(data["path_names"])
    if row_count == 0:
        raise ValueError("Selected-target archive contains no target rows")
    if np.any(np.char.str_len(data["path_names"]) == 0):
        raise ValueError("path_names contains an empty path name")
    if np.any(np.char.str_len(data["candidate_methods"]) == 0):
        raise ValueError("candidate_methods contains an empty method name")
    if np.any(np.isin(np.char.lower(data["path_names"]), ("nan", "none"))):
        raise ValueError("path_names contains a missing-value marker")
    if np.any(np.isin(np.char.lower(data["candidate_methods"]), ("nan", "none"))):
        raise ValueError("candidate_methods contains a missing-value marker")
    vector_keys = (
        "path_indices", "window_starts", "target_indices_within_window",
        "candidate_methods", "is_zero_residual", "improves_prior",
        "execution_horizon",
    )
    for key in vector_keys:
        if np.asarray(data[key]).reshape(-1).shape != (row_count,):
            raise ValueError(
                f"{key} has shape {data[key].shape}; expected ({row_count},)"
            )
        data[key] = np.asarray(data[key]).reshape(-1)

    expected_shapes = {
        "desired_path_window": (row_count, HORIZON, CARTESIAN_DIM),
        "prior_q_window": (row_count, HORIZON, JOINT_DIM),
        "prior_ee_window": (row_count, HORIZON, CARTESIAN_DIM),
        "residual_q_window": (row_count, HORIZON, JOINT_DIM),
        "candidate_q_window": (row_count, HORIZON, JOINT_DIM),
    }
    for key, expected in expected_shapes.items():
        if data[key].shape != expected:
            raise ValueError(f"{key} has shape {data[key].shape}; expected {expected}")
        data[key] = np.asarray(data[key], dtype=np.float64)
        require_finite(key, data[key])

    for key in ("path_indices", "window_starts", "target_indices_within_window", "execution_horizon"):
        numeric = np.asarray(data[key])
        require_finite(key, numeric.astype(np.float64))
        if not np.all(numeric == numeric.astype(np.int64)):
            raise ValueError(f"{key} contains non-integral values")
        data[key] = numeric.astype(np.int64)
    for key in ("is_zero_residual", "improves_prior"):
        raw_flags = np.asarray(data[key])
        if np.issubdtype(raw_flags.dtype, np.number):
            require_finite(key, raw_flags.astype(np.float64))
            if not np.all(np.isin(raw_flags, (0, 1))):
                raise ValueError(f"{key} must contain only boolean/0/1 values")
        elif not all(
            isinstance(value, (bool, np.bool_))
            for value in raw_flags.reshape(-1)
        ):
            raise ValueError(f"{key} contains a missing or non-boolean value")
        data[key] = raw_flags.astype(bool)

    reconstructed = data["candidate_q_window"] - data["prior_q_window"]
    max_residual_error = float(np.max(np.abs(data["residual_q_window"] - reconstructed)))
    if not np.allclose(
        data["residual_q_window"], reconstructed,
        rtol=FLOAT_RTOL, atol=FLOAT_ATOL,
    ):
        raise ValueError(
            "residual_q_window is not candidate_q_window - prior_q_window; "
            f"maximum absolute discrepancy={max_residual_error:.9g}"
        )

    numerically_zero = np.max(np.abs(data["residual_q_window"]), axis=(1, 2)) <= ZERO_ATOL
    flagged_zero = data["is_zero_residual"]
    if np.any(numerically_zero & ~flagged_zero):
        rows = np.flatnonzero(numerically_zero & ~flagged_zero).tolist()
        raise ValueError(f"Numerically zero residual rows are not flagged zero: {rows[:20]}")
    if np.any(flagged_zero & ~numerically_zero):
        rows = np.flatnonzero(flagged_zero & ~numerically_zero).tolist()
        raise ValueError(f"is_zero_residual rows contain nonzero targets: {rows[:20]}")
    nonzero = ~flagged_zero
    if np.any(nonzero & ~data["improves_prior"]):
        rows = np.flatnonzero(nonzero & ~data["improves_prior"]).tolist()
        raise ValueError(f"Nonzero rows without improves_prior=True: {rows[:20]}")
    if np.any(data["window_starts"] < 0) or np.any(
        data["window_starts"] + HORIZON > TRAJECTORY_LENGTH
    ):
        raise ValueError("window_starts do not describe 32-step windows inside a 100-step path")
    if np.any(data["execution_horizon"] <= 0) or np.any(
        data["execution_horizon"] > HORIZON
    ):
        raise ValueError("execution_horizon must lie within the 32-step window")
    return data


WindowKey = Tuple[str, int]


def group_windows(data: Mapping[str, np.ndarray]) -> Tuple[List[WindowKey], Dict[WindowKey, np.ndarray]]:
    grouped: Dict[WindowKey, List[int]] = defaultdict(list)
    for row, (name, start) in enumerate(zip(data["path_names"], data["window_starts"])):
        grouped[(str(name), int(start))].append(row)
    keys = sorted(grouped)
    return keys, {
        key: np.asarray(grouped[key], dtype=np.int64)
        for key in keys
    }


def maximum_abs_difference(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64))))


def verify_duplicate_conditioning(
    data: Mapping[str, np.ndarray], groups: Mapping[WindowKey, np.ndarray]
) -> float:
    maximum = 0.0
    conditioning_keys = (
        "desired_path_window", "prior_q_window", "prior_ee_window"
    )
    for window_key, rows in groups.items():
        reference = int(rows[0])
        for row_value in rows[1:]:
            row = int(row_value)
            for key in conditioning_keys:
                discrepancy = maximum_abs_difference(data[key][reference], data[key][row])
                maximum = max(maximum, discrepancy)
                if not np.allclose(
                    data[key][reference], data[key][row],
                    rtol=FLOAT_RTOL, atol=FLOAT_ATOL,
                ):
                    raise ValueError(
                        f"Duplicate-window conditioning is inconsistent for {window_key}, "
                        f"field={key}, rows=({reference},{row}), "
                        f"maximum absolute discrepancy={discrepancy:.9g}"
                    )
            if int(data["execution_horizon"][reference]) != int(data["execution_horizon"][row]):
                raise ValueError(
                    f"Duplicate-window execution_horizon is inconsistent for "
                    f"{window_key}, rows=({reference},{row})"
                )
    return maximum


def merge_path_windows(
    path_name: str,
    window_keys: Sequence[WindowKey],
    groups: Mapping[WindowKey, np.ndarray],
    data: Mapping[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    desired = np.full((TRAJECTORY_LENGTH, CARTESIAN_DIM), np.nan, dtype=np.float64)
    prior_q = np.full((TRAJECTORY_LENGTH, JOINT_DIM), np.nan, dtype=np.float64)
    desired_filled = np.zeros(TRAJECTORY_LENGTH, dtype=bool)
    prior_filled = np.zeros(TRAJECTORY_LENGTH, dtype=bool)
    for key in window_keys:
        if key[0] != path_name:
            continue
        row = int(groups[key][0])
        start = key[1]
        stop = start + HORIZON
        for label, destination, filled, source in (
            ("desired_path", desired, desired_filled, data["desired_path_window"][row]),
            ("prior_q", prior_q, prior_filled, data["prior_q_window"][row]),
        ):
            overlap = filled[start:stop]
            if np.any(overlap):
                existing = destination[start:stop][overlap]
                incoming = source[overlap]
                discrepancy = maximum_abs_difference(existing, incoming)
                if not np.allclose(existing, incoming, rtol=FLOAT_RTOL, atol=FLOAT_ATOL):
                    raise ValueError(
                        f"Overlapping {label} windows disagree for path={path_name}, "
                        f"start={start}, maximum absolute discrepancy={discrepancy:.9g}"
                    )
            destination[start:stop] = source
            filled[start:stop] = True
    if not prior_filled[0]:
        raise ValueError(
            f"Path {path_name} has no timestep-0 prior value; the exact v6 "
            "prior_q_start feature cannot be reconstructed"
        )
    return desired, prior_q


def build_v6_conditions(
    data: Mapping[str, np.ndarray],
    window_keys: Sequence[WindowKey],
    groups: Mapping[WindowKey, np.ndarray],
) -> Dict[WindowKey, np.ndarray]:
    by_path: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for path_name in sorted({key[0] for key in window_keys}):
        by_path[path_name] = merge_path_windows(path_name, window_keys, groups, data)

    result: Dict[WindowKey, np.ndarray] = {}
    for key in window_keys:
        path_name, start = key
        row = int(groups[key][0])
        desired_full, prior_full = by_path[path_name]
        stop = start + HORIZON
        desired_window = data["desired_path_window"][row]
        prior_q_window = data["prior_q_window"][row]
        prior_ee_window = data["prior_ee_window"][row]
        desired_delta = np.empty_like(desired_window)
        for offset, timestep in enumerate(range(start, stop)):
            if timestep == 0:
                desired_delta[offset] = 0.0
            else:
                if not np.all(np.isfinite(desired_full[timestep - 1])):
                    raise ValueError(
                        f"Missing desired timestep {timestep - 1} for path={path_name}; "
                        "the exact v6 desired_dxyz feature cannot be reconstructed"
                    )
                desired_delta[offset] = desired_full[timestep] - desired_full[timestep - 1]
        q_start = prior_full[0]
        current_q = prior_q_window[0]
        progress = np.arange(start, stop, dtype=np.float64) / (TRAJECTORY_LENGTH - 1)
        prior_error = prior_ee_window - desired_window
        condition = np.concatenate(
            (
                desired_window,
                desired_delta,
                progress[:, None],
                np.repeat(q_start[None, :], HORIZON, axis=0),
                np.repeat(current_q[None, :], HORIZON, axis=0),
                prior_q_window,
                prior_q_window - q_start[None, :],
                prior_ee_window,
                prior_error,
                np.linalg.norm(prior_error, axis=1, keepdims=True),
            ),
            axis=1,
        )
        if condition.shape != (HORIZON, CONDITION_DIM):
            raise AssertionError(f"Condition for {key} has unexpected shape {condition.shape}")
        require_finite(f"condition[{path_name},{start}]", condition)
        result[key] = condition
    return result


def stable_stats(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = np.mean(values, axis=(0, 1), keepdims=True)
    raw_std = np.std(values, axis=(0, 1), keepdims=True)
    return mean, np.where(raw_std < NORMALIZATION_EPSILON, 1.0, raw_std)


def weighted_stats(values: np.ndarray, sample_weight: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    weights = np.asarray(sample_weight, dtype=np.float64).reshape(-1, 1, 1)
    denominator = float(np.sum(weights) * values.shape[1])
    if denominator <= 0.0:
        raise ValueError("Training sample weights have zero total")
    mean = np.sum(values * weights, axis=(0, 1), keepdims=True) / denominator
    variance = np.sum(np.square(values - mean) * weights, axis=(0, 1), keepdims=True) / denominator
    raw_std = np.sqrt(np.maximum(variance, 0.0))
    return mean, np.where(raw_std < NORMALIZATION_EPSILON, 1.0, raw_std)


def path_split(
    path_names: Iterable[str], seed: int, train_count: int | None,
    validation_count: int | None,
) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    ordered = np.asarray(sorted(set(path_names)), dtype=str)
    total = len(ordered)
    if train_count is None and validation_count is None:
        train_count, validation_count = 80, 20
    elif train_count is None:
        assert validation_count is not None
        train_count = total - validation_count
    else:
        validation_count = total - train_count
    assert train_count is not None and validation_count is not None
    if train_count <= 0 or validation_count <= 0 or train_count + validation_count != total:
        raise ValueError(
            f"Requested train/validation counts {train_count}/{validation_count} "
            f"do not partition the {total} unique paths"
        )
    permutation = np.random.default_rng(seed).permutation(total)
    train = tuple(sorted(ordered[permutation[:train_count]].tolist()))
    validation = tuple(sorted(ordered[permutation[train_count:]].tolist()))
    return train, validation


def make_split_arrays(
    data: Mapping[str, np.ndarray], row_indices: np.ndarray,
    conditions: Mapping[WindowKey, np.ndarray], window_id: Mapping[WindowKey, int],
    target_count: Mapping[WindowKey, int], condition_mean: np.ndarray,
    condition_std: np.ndarray, residual_mean: np.ndarray, residual_std: np.ndarray,
) -> Dict[str, np.ndarray]:
    names = data["path_names"][row_indices]
    starts = data["window_starts"][row_indices]
    keys = [(str(name), int(start)) for name, start in zip(names, starts)]
    raw_condition = np.stack([conditions[key] for key in keys])
    targets_in_window = np.asarray([target_count[key] for key in keys], dtype=np.int64)
    sample_weight = 1.0 / targets_in_window.astype(np.float64)
    residual = data["residual_q_window"][row_indices]
    result: Dict[str, np.ndarray] = {
        "condition": raw_condition,
        "condition_features": raw_condition,
        "condition_norm": (raw_condition - condition_mean) / condition_std,
        "condition_features_norm": (raw_condition - condition_mean) / condition_std,
        "residual_q_window": residual,
        "residual_q_norm": (residual - residual_mean) / residual_std,
        "desired_path_window": data["desired_path_window"][row_indices],
        "prior_q_window": data["prior_q_window"][row_indices],
        "prior_ee_window": data["prior_ee_window"][row_indices],
        "candidate_q_window": data["candidate_q_window"][row_indices],
        "path_names": names,
        "path_indices": data["path_indices"][row_indices],
        "window_starts": starts,
        "target_indices_within_window": data["target_indices_within_window"][row_indices],
        "candidate_methods": data["candidate_methods"][row_indices],
        "is_zero_residual": data["is_zero_residual"][row_indices],
        "improves_prior": data["improves_prior"][row_indices],
        "execution_horizon": data["execution_horizon"][row_indices],
        "unique_window_id": np.asarray([window_id[key] for key in keys], dtype=np.int64),
        "targets_in_window": targets_in_window,
        "sample_weight": sample_weight,
    }
    for key, value in tuple(result.items()):
        if np.issubdtype(value.dtype, np.floating):
            result[key] = value.astype(np.float32)
    return result


def split_summary(arrays: Mapping[str, np.ndarray]) -> Dict[str, Any]:
    window_ids = np.asarray(arrays["unique_window_id"])
    weight_sums = np.asarray(
        [
            np.sum(arrays["sample_weight"][window_ids == identifier], dtype=np.float64)
            for identifier in np.unique(window_ids)
        ]
    )
    counts = Counter(np.asarray(arrays["targets_in_window"], dtype=np.int64).tolist())
    return {
        "unique_path_count": len(set(decode_strings(arrays["path_names"]).tolist())),
        "unique_window_count": len(np.unique(window_ids)),
        "target_row_count": len(window_ids),
        "zero_target_count": int(np.count_nonzero(arrays["is_zero_residual"])),
        "nonzero_target_count": int(len(window_ids) - np.count_nonzero(arrays["is_zero_residual"])),
        "targets_per_window_distribution": {
            str(count): int(frequency // count) for count, frequency in sorted(counts.items())
        },
        "sample_weight_sum_per_window": {
            "minimum": float(np.min(weight_sums)),
            "maximum": float(np.max(weight_sums)),
            "mean": float(np.mean(weight_sums)),
        },
    }


def main() -> int:
    args = parse_args()
    existing = [args.output_dir / name for name in OUTPUT_FILENAMES if (args.output_dir / name).exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            f"Output files already exist: {existing}; pass --overwrite to replace them"
        )
    data = load_and_validate(args.input_npz)
    window_keys, groups = group_windows(data)
    maximum_duplicate_discrepancy = verify_duplicate_conditioning(data, groups)
    conditions = build_v6_conditions(data, window_keys, groups)
    train_paths, validation_paths = path_split(
        data["path_names"], args.seed, args.train_path_count, args.validation_path_count
    )
    if set(train_paths) & set(validation_paths):
        raise AssertionError("A path appears in both training and validation")

    train_mask = np.isin(data["path_names"], np.asarray(train_paths))
    validation_mask = np.isin(data["path_names"], np.asarray(validation_paths))
    if not np.all(train_mask ^ validation_mask):
        raise AssertionError("Every target row must belong to exactly one split")
    train_rows = np.flatnonzero(train_mask)
    validation_rows = np.flatnonzero(validation_mask)
    window_id = {key: index for index, key in enumerate(window_keys)}
    target_count = {key: len(rows) for key, rows in groups.items()}

    training_window_keys = [key for key in window_keys if key[0] in set(train_paths)]
    unique_train_conditions = np.stack([conditions[key] for key in training_window_keys])
    train_row_keys = [
        (str(data["path_names"][row]), int(data["window_starts"][row]))
        for row in train_rows
    ]
    train_weights = np.asarray([1.0 / target_count[key] for key in train_row_keys])
    condition_mean, condition_std = stable_stats(unique_train_conditions)
    residual_mean, residual_std = weighted_stats(
        data["residual_q_window"][train_rows], train_weights
    )

    train_arrays = make_split_arrays(
        data, train_rows, conditions, window_id, target_count,
        condition_mean, condition_std, residual_mean, residual_std,
    )
    validation_arrays = make_split_arrays(
        data, validation_rows, conditions, window_id, target_count,
        condition_mean, condition_std, residual_mean, residual_std,
    )
    train_summary = split_summary(train_arrays)
    validation_summary = split_summary(validation_arrays)
    for label, summary in (("train", train_summary), ("validation", validation_summary)):
        weight_stats = summary["sample_weight_sum_per_window"]
        if not np.isclose(weight_stats["minimum"], 1.0, atol=1.0e-6) or not np.isclose(
            weight_stats["maximum"], 1.0, atol=1.0e-6
        ):
            raise AssertionError(f"{label} sample weights do not sum to one per window")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    normalization = {
        "condition_mean": condition_mean.astype(np.float32),
        "condition_std": condition_std.astype(np.float32),
        "residual_mean": residual_mean.astype(np.float32),
        "residual_std": residual_std.astype(np.float32),
        "condition_feature_names": np.asarray(CONDITION_FEATURE_NAMES),
        "condition_feature_layout": np.asarray(CONDITION_FEATURE_LAYOUT),
        "residual_feature_names": np.asarray(RESIDUAL_FEATURE_NAMES),
        "condition_dim": np.asarray(CONDITION_DIM, dtype=np.int64),
        "target_dim": np.asarray(JOINT_DIM, dtype=np.int64),
        "horizon": np.asarray(HORIZON, dtype=np.int64),
        "epsilon": np.asarray(NORMALIZATION_EPSILON, dtype=np.float64),
        "train_path_count": np.asarray(len(train_paths), dtype=np.int64),
        "train_window_count": np.asarray(len(training_window_keys), dtype=np.int64),
        "train_target_row_count": np.asarray(len(train_rows), dtype=np.int64),
        "validation_excluded": np.asarray(True, dtype=bool),
    }
    split_manifest = {
        "seed": args.seed,
        "split_unit": "path_name",
        "path_names_sorted_before_random_selection": True,
        "train_path_count": len(train_paths),
        "validation_path_count": len(validation_paths),
        "train_path_names": list(train_paths),
        "validation_path_names": list(validation_paths),
    }
    metadata = {
        "classification": "READY_FOR_V7_TRAINING",
        "source_npz": str(args.input_npz.resolve()),
        "source_target_row_count": len(data["path_names"]),
        "source_unique_path_count": len(set(data["path_names"].tolist())),
        "source_unique_window_count": len(window_keys),
        "horizon": HORIZON,
        "joint_dim": JOINT_DIM,
        "cartesian_dim": CARTESIAN_DIM,
        "condition_dim": CONDITION_DIM,
        "target_dim": JOINT_DIM,
        "condition_feature_layout": CONDITION_FEATURE_LAYOUT,
        "condition_feature_names": CONDITION_FEATURE_NAMES,
        "condition_schema_source": "v6 strong-prior residual window dataset",
        "omitted_v6_condition_features": [],
        "reconstruction_note": (
            "Overlapping per-path windows reconstruct q(t=0) and the preceding "
            "desired point, so all v6 condition features are retained exactly."
        ),
        "residual_target": "candidate_q_window_minus_prior_q_window",
        "target_multiplicity_preserved": True,
        "sample_weight_formula": "1 / targets_in_window",
        "condition_normalization_source": "one_row_per_unique_training_window",
        "residual_normalization_source": "training_target_rows_sample_weighted_by_inverse_window_multiplicity",
        "normalization_source_split": "training_paths_only",
        "validation_excluded_from_normalization": True,
        "normalization_standard_deviation_floor": NORMALIZATION_EPSILON,
        "normalization_floor_replacement": 1.0,
        "maximum_duplicate_conditioning_discrepancy": maximum_duplicate_discrepancy,
        "execution_horizons": sorted(
            set(int(value) for value in data["execution_horizon"].tolist())
        ),
        "split_seed": args.seed,
    }
    summary = {
        "train": train_summary,
        "validation": validation_summary,
        "condition_dim": CONDITION_DIM,
        "residual_dimensions": [HORIZON, JOINT_DIM],
        "maximum_duplicate_conditioning_discrepancy": maximum_duplicate_discrepancy,
        "normalization_source_split": "training_paths_only (validation excluded)",
    }

    atomic_npz(args.output_dir / "train_windows.npz", train_arrays)
    atomic_npz(args.output_dir / "validation_windows.npz", validation_arrays)
    atomic_npz(args.output_dir / "normalization.npz", normalization)
    atomic_json(args.output_dir / "split_manifest.json", split_manifest)
    atomic_json(args.output_dir / "dataset_metadata.json", metadata)
    atomic_json(args.output_dir / "dataset_summary.json", summary)

    for label, split in (("train", train_summary), ("validation", validation_summary)):
        print(
            f"{label}: paths={split['unique_path_count']}, "
            f"windows={split['unique_window_count']}, rows={split['target_row_count']}, "
            f"zero/nonzero={split['zero_target_count']}/{split['nonzero_target_count']}, "
            f"targets_per_window={split['targets_per_window_distribution']}, "
            f"weight_sums={split['sample_weight_sum_per_window']}"
        )
    print(f"condition_dim={CONDITION_DIM}, residual_dimensions=({HORIZON},{JOINT_DIM})")
    print(f"maximum duplicate-conditioning discrepancy={maximum_duplicate_discrepancy:.9g}")
    print("normalization source: training paths only; validation excluded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
