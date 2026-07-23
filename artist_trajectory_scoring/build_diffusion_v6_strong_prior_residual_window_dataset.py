#!/usr/bin/env python3
"""Build v6 strong-prior residual windows without supervised test targets."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


EXPECTED_TRAIN_PATHS = 418
EXPECTED_TEST_PATHS = 83
EXPECTED_ELIGIBLE_PATHS = 414
TRAJECTORY_LENGTH = 100
JOINT_DIM = 6
CARTESIAN_DIM = 3
CONDITION_DIM = 38
TARGET_DIM = 6
NORMALIZATION_EPSILON = 1.0e-8
DESIRED_PATH_RTOL = 1.0e-6
DESIRED_PATH_ATOL = 1.0e-7

OUTPUT_FILENAMES = (
    "train_windows.npz",
    "val_windows.npz",
    "test_inference_windows.npz",
    "test_reference_metadata.npz",
    "normalization_stats.npz",
    "split_manifest.csv",
    "split_manifest.json",
    "excluded_train_paths.csv",
    "dataset_summary.csv",
    "dataset_configuration.json",
)


@dataclass(frozen=True)
class PriorData:
    source: Path
    keys: Tuple[str, ...]
    names: Tuple[str, ...]
    desired: np.ndarray
    q: np.ndarray
    ee: np.ndarray
    generation_success: np.ndarray


@dataclass(frozen=True)
class ReferenceData:
    source: Path
    keys: Tuple[str, ...]
    names: Tuple[str, ...]
    desired: np.ndarray
    expert_q: Optional[np.ndarray]


@dataclass(frozen=True)
class AlignedData:
    names: Tuple[str, ...]
    source_indices: np.ndarray
    desired: np.ndarray
    prior_q: np.ndarray
    prior_ee: np.ndarray
    generation_success: np.ndarray
    expert_q: Optional[np.ndarray]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build path-disjoint v6 residual windows from the frozen Adaptive "
            "MLP+IK prior."
        )
    )
    parser.add_argument("--train_prior", type=Path, required=True)
    parser.add_argument("--test_prior", type=Path, required=True)
    parser.add_argument("--train_expert", type=Path, required=True)
    parser.add_argument("--test_reference", type=Path, required=True)
    parser.add_argument("--residual_audit_csv", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--val_count", type=int, default=42)
    parser.add_argument("--split_seed", type=int, default=42)
    parser.add_argument(
        "--include_train_classification",
        default="local_residual_candidate",
    )
    parser.add_argument(
        "--allow_eligible_count_mismatch",
        action="store_true",
        help="Allow a filtered train count other than the audited expectation of 414.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_cli(args: argparse.Namespace) -> None:
    if args.horizon <= 0 or args.horizon > TRAJECTORY_LENGTH:
        raise ValueError(
            f"--horizon must be in [1,{TRAJECTORY_LENGTH}], got {args.horizon}"
        )
    if args.stride <= 0:
        raise ValueError("--stride must be positive")
    if args.val_count <= 0:
        raise ValueError("--val_count must be positive")
    if not args.include_train_classification:
        raise ValueError("--include_train_classification cannot be empty")


def decode_names(values: np.ndarray) -> Tuple[str, ...]:
    names: List[str] = []
    for value in np.asarray(values).reshape(-1):
        if isinstance(value, bytes):
            names.append(value.decode("utf-8", errors="strict"))
        else:
            names.append(str(value))
    return tuple(names)


def require_keys(archive: Any, path: Path, keys: Sequence[str]) -> None:
    missing = [key for key in keys if key not in archive.files]
    if missing:
        raise KeyError(f"{path} is missing required keys {missing}")


def require_unique(names: Sequence[str], label: str) -> None:
    if len(set(names)) != len(names):
        duplicates = sorted(
            {name for name in names if list(names).count(name) > 1}
        )
        raise ValueError(f"{label} contains duplicate path names: {duplicates}")


def validate_joint_array(array: np.ndarray, count: int, label: str) -> None:
    expected = (count, TRAJECTORY_LENGTH, JOINT_DIM)
    if array.shape != expected:
        raise ValueError(f"{label} shape is {array.shape}, expected {expected}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} contains nonfinite values")


def validate_cartesian_array(array: np.ndarray, count: int, label: str) -> None:
    expected = (count, TRAJECTORY_LENGTH, CARTESIAN_DIM)
    if array.shape != expected:
        raise ValueError(f"{label} shape is {array.shape}, expected {expected}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} contains nonfinite values")


def load_prior(path: Path, expected_count: int, label: str) -> PriorData:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=True) as archive:
        require_keys(
            archive,
            path,
            (
                "path_names",
                "desired_paths",
                "prior_q",
                "prior_ee",
                "generation_success",
            ),
        )
        keys = tuple(archive.files)
        names = decode_names(archive["path_names"])
        desired = np.asarray(archive["desired_paths"], dtype=np.float64)
        q = np.asarray(archive["prior_q"], dtype=np.float64)
        ee = np.asarray(archive["prior_ee"], dtype=np.float64)
        success = np.asarray(archive["generation_success"], dtype=bool).reshape(-1)
    if len(names) != expected_count:
        raise ValueError(f"{label} has {len(names)} paths, expected {expected_count}")
    require_unique(names, label)
    validate_joint_array(q, expected_count, f"{label} prior_q")
    validate_cartesian_array(desired, expected_count, f"{label} desired_paths")
    validate_cartesian_array(ee, expected_count, f"{label} prior_ee")
    if success.shape != (expected_count,):
        raise ValueError(
            f"{label} generation_success shape is {success.shape}, "
            f"expected ({expected_count},)"
        )
    failed = [name for name, passed in zip(names, success) if not bool(passed)]
    if failed:
        raise ValueError(f"{label} contains failed prior paths: {failed}")
    return PriorData(path, keys, names, desired, q, ee, success)


def load_train_expert(path: Path) -> ReferenceData:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=True) as archive:
        require_keys(archive, path, ("path_names", "desired_paths", "expert_q"))
        keys = tuple(archive.files)
        names = decode_names(archive["path_names"])
        desired = np.asarray(archive["desired_paths"], dtype=np.float64)
        expert_q = np.asarray(archive["expert_q"], dtype=np.float64)
    if len(names) != EXPECTED_TRAIN_PATHS:
        raise ValueError(
            f"train expert has {len(names)} paths, expected {EXPECTED_TRAIN_PATHS}"
        )
    require_unique(names, "train expert")
    validate_cartesian_array(desired, len(names), "train expert desired_paths")
    validate_joint_array(expert_q, len(names), "train expert expert_q")
    return ReferenceData(path, keys, names, desired, expert_q)


def load_test_reference(path: Path) -> ReferenceData:
    """Load only names and desired paths; expert_q is deliberately not accessed."""
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=True) as archive:
        require_keys(archive, path, ("path_names", "desired_paths"))
        keys = tuple(archive.files)
        names = decode_names(archive["path_names"])
        desired = np.asarray(archive["desired_paths"], dtype=np.float64)
    if len(names) != EXPECTED_TEST_PATHS:
        raise ValueError(
            f"test reference has {len(names)} paths, expected {EXPECTED_TEST_PATHS}"
        )
    require_unique(names, "test reference")
    validate_cartesian_array(desired, len(names), "test reference desired_paths")
    return ReferenceData(path, keys, names, desired, None)


def align_prior_to_reference(
    prior: PriorData, reference: ReferenceData, label: str
) -> AlignedData:
    if set(prior.names) != set(reference.names):
        prior_only = sorted(set(prior.names) - set(reference.names))
        reference_only = sorted(set(reference.names) - set(prior.names))
        raise ValueError(
            f"{label} path-name sets differ: "
            f"prior_only={prior_only}, reference_only={reference_only}"
        )
    prior_index = {name: index for index, name in enumerate(prior.names)}
    order = np.asarray([prior_index[name] for name in reference.names], dtype=np.int64)
    prior_desired = prior.desired[order]
    maximum_difference = float(np.max(np.abs(prior_desired - reference.desired)))
    if not np.allclose(
        prior_desired,
        reference.desired,
        rtol=DESIRED_PATH_RTOL,
        atol=DESIRED_PATH_ATOL,
    ):
        raise ValueError(
            f"{label} desired paths differ after name alignment; "
            f"max abs difference={maximum_difference:.8g}"
        )
    return AlignedData(
        names=reference.names,
        source_indices=np.arange(len(reference.names), dtype=np.int64),
        desired=reference.desired,
        prior_q=prior.q[order],
        prior_ee=prior.ee[order],
        generation_success=prior.generation_success[order],
        expert_q=reference.expert_q,
    )


def load_audit(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    required = {
        "split",
        "path_name",
        "diagnostic_classification",
        "raw_rmse_rad",
        "wrapped_rmse_rad",
        "aligned_rmse_rad",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing audit columns {missing}")
    if frame.duplicated(["split", "path_name"]).any():
        duplicates = frame.loc[
            frame.duplicated(["split", "path_name"], keep=False),
            ["split", "path_name"],
        ].to_dict(orient="records")
        raise ValueError(f"Audit CSV contains duplicate split/path rows: {duplicates}")
    metric_columns = ("raw_rmse_rad", "wrapped_rmse_rad", "aligned_rmse_rad")
    metric_values = frame.loc[:, metric_columns].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(metric_values)):
        raise ValueError(f"{path} contains nonfinite residual RMSE values")
    return frame.copy()


def deterministic_path_split(
    eligible_names: Sequence[str], val_count: int, seed: int
) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    ordered = np.asarray(sorted(eligible_names), dtype=str)
    if val_count >= len(ordered):
        raise ValueError(
            f"--val_count {val_count} must be smaller than eligible count {len(ordered)}"
        )
    permutation = np.random.default_rng(seed).permutation(len(ordered))
    validation = tuple(sorted(ordered[permutation[:val_count]].tolist()))
    train = tuple(sorted(ordered[permutation[val_count:]].tolist()))
    return train, validation


def expected_window_starts(horizon: int, stride: int) -> np.ndarray:
    return np.arange(0, TRAJECTORY_LENGTH - horizon + 1, stride, dtype=np.int64)


def desired_finite_difference(desired: np.ndarray) -> np.ndarray:
    difference = np.zeros_like(desired, dtype=np.float64)
    difference[1:] = desired[1:] - desired[:-1]
    return difference


def build_condition(
    *,
    desired_window: np.ndarray,
    desired_delta_window: np.ndarray,
    progress_window: np.ndarray,
    q_start: np.ndarray,
    current_q: np.ndarray,
    prior_q_window: np.ndarray,
    prior_ee_window: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    horizon = prior_q_window.shape[0]
    prior_ee_error = prior_ee_window - desired_window
    prior_ee_error_norm = np.linalg.norm(prior_ee_error, axis=1, keepdims=True)
    q_start_repeated = np.repeat(q_start[None, :], horizon, axis=0)
    current_q_repeated = np.repeat(current_q[None, :], horizon, axis=0)
    prior_delta_from_start = prior_q_window - q_start[None, :]
    condition = np.concatenate(
        (
            desired_window,
            desired_delta_window,
            progress_window[:, None],
            q_start_repeated,
            current_q_repeated,
            prior_q_window,
            prior_delta_from_start,
            prior_ee_window,
            prior_ee_error,
            prior_ee_error_norm,
        ),
        axis=1,
    )
    if condition.shape != (horizon, CONDITION_DIM):
        raise AssertionError(
            f"Condition shape {condition.shape}, expected ({horizon},{CONDITION_DIM})"
        )
    return condition, prior_ee_error, prior_ee_error_norm


def build_windows(
    *,
    data: AlignedData,
    selected_names: Sequence[str],
    starts: np.ndarray,
    horizon: int,
    supervised: bool,
    assigned_split: str,
) -> Dict[str, np.ndarray]:
    index_by_name = {name: index for index, name in enumerate(data.names)}
    rows: Dict[str, List[np.ndarray]] = {
        "prior_q_window": [],
        "desired_path_window": [],
        "prior_ee_window": [],
        "prior_ee_error": [],
        "prior_ee_error_norm": [],
        "condition_features": [],
    }
    if supervised:
        if data.expert_q is None:
            raise ValueError("Supervised windows require expert_q")
        rows.update(
            {
                "expert_q_window": [],
                "residual_q_window": [],
            }
        )
    window_names: List[str] = []
    window_indices: List[int] = []
    window_starts: List[int] = []
    success_values: List[bool] = []
    full_progress = np.linspace(0.0, 1.0, TRAJECTORY_LENGTH, dtype=np.float64)

    for path_name in selected_names:
        path_index = index_by_name[path_name]
        prior_q = data.prior_q[path_index]
        desired = data.desired[path_index]
        prior_ee = data.prior_ee[path_index]
        desired_delta = desired_finite_difference(desired)
        q_start = prior_q[0]
        for start_value in starts:
            start = int(start_value)
            end = start + horizon
            prior_q_window = prior_q[start:end]
            desired_window = desired[start:end]
            prior_ee_window = prior_ee[start:end]
            condition, ee_error, ee_error_norm = build_condition(
                desired_window=desired_window,
                desired_delta_window=desired_delta[start:end],
                progress_window=full_progress[start:end],
                q_start=q_start,
                current_q=prior_q[start],
                prior_q_window=prior_q_window,
                prior_ee_window=prior_ee_window,
            )
            rows["prior_q_window"].append(prior_q_window)
            rows["desired_path_window"].append(desired_window)
            rows["prior_ee_window"].append(prior_ee_window)
            rows["prior_ee_error"].append(ee_error)
            rows["prior_ee_error_norm"].append(ee_error_norm)
            rows["condition_features"].append(condition)
            if supervised:
                assert data.expert_q is not None
                expert_window = data.expert_q[path_index, start:end]
                rows["expert_q_window"].append(expert_window)
                rows["residual_q_window"].append(expert_window - prior_q_window)
            window_names.append(path_name)
            window_indices.append(int(data.source_indices[path_index]))
            window_starts.append(start)
            success_values.append(bool(data.generation_success[path_index]))

    result = {key: np.stack(values) for key, values in rows.items()}
    result.update(
        {
            "path_names": np.asarray(window_names),
            "path_indices": np.asarray(window_indices, dtype=np.int64),
            "window_starts": np.asarray(window_starts, dtype=np.int64),
            "generation_success": np.asarray(success_values, dtype=bool),
            "split": np.asarray([assigned_split] * len(window_names)),
        }
    )
    return result


def stable_stats(values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = np.mean(values, axis=(0, 1), keepdims=True)
    raw_std = np.std(values, axis=(0, 1), keepdims=True)
    std = np.where(raw_std < NORMALIZATION_EPSILON, 1.0, raw_std)
    return mean, std


def normalize_windows(
    windows: Dict[str, np.ndarray],
    condition_mean: np.ndarray,
    condition_std: np.ndarray,
    residual_mean: Optional[np.ndarray],
    residual_std: Optional[np.ndarray],
) -> None:
    windows["condition_features_norm"] = (
        windows["condition_features"] - condition_mean
    ) / condition_std
    if "residual_q_window" in windows:
        if residual_mean is None or residual_std is None:
            raise ValueError("Residual statistics are required for supervised windows")
        windows["residual_q_norm"] = (
            windows["residual_q_window"] - residual_mean
        ) / residual_std


def validate_window_collection(
    *,
    label: str,
    windows: Mapping[str, np.ndarray],
    path_names: Sequence[str],
    starts: np.ndarray,
    condition_mean: np.ndarray,
    condition_std: np.ndarray,
    residual_mean: Optional[np.ndarray],
    residual_std: Optional[np.ndarray],
    supervised: bool,
) -> None:
    expected_windows = len(path_names) * len(starts)
    if len(windows["path_names"]) != expected_windows:
        raise AssertionError(
            f"{label} has {len(windows['path_names'])} windows, "
            f"expected {expected_windows}"
        )
    for path_name in path_names:
        selected_starts = windows["window_starts"][
            windows["path_names"] == path_name
        ]
        if not np.array_equal(selected_starts, starts):
            raise AssertionError(
                f"{label}/{path_name} starts {selected_starts.tolist()} "
                f"do not match {starts.tolist()}"
            )
    for key, value in windows.items():
        if np.issubdtype(value.dtype, np.number) and not np.all(np.isfinite(value)):
            raise AssertionError(f"{label}/{key} contains nonfinite values")
    condition_reconstructed = (
        windows["condition_features_norm"] * condition_std + condition_mean
    )
    if not np.allclose(
        condition_reconstructed,
        windows["condition_features"],
        rtol=1.0e-6,
        atol=1.0e-7,
    ):
        raise AssertionError(f"{label} condition normalization is not invertible")
    if supervised:
        if not np.allclose(
            windows["prior_q_window"] + windows["residual_q_window"],
            windows["expert_q_window"],
            rtol=1.0e-7,
            atol=1.0e-8,
        ):
            raise AssertionError(f"{label} residual reconstruction failed")
        assert residual_mean is not None and residual_std is not None
        residual_reconstructed = (
            windows["residual_q_norm"] * residual_std + residual_mean
        )
        if not np.allclose(
            residual_reconstructed,
            windows["residual_q_window"],
            rtol=1.0e-6,
            atol=1.0e-7,
        ):
            raise AssertionError(f"{label} residual normalization is not invertible")


def atomic_savez(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    temporary = path.with_name(f"{path.stem}.tmp.npz")
    try:
        np.savez_compressed(temporary, **arrays)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        frame.to_csv(temporary, index=False)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def atomic_json(value: Mapping[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(json_safe(value), handle, indent=2, sort_keys=True)
            handle.write("\n")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def manifest_row(
    *,
    split: str,
    name: str,
    source_index: int,
    assigned_split: str,
    audit_lookup: Mapping[Tuple[str, str], Mapping[str, Any]],
    split_seed: int,
) -> Dict[str, Any]:
    audit = audit_lookup[(split, name)]
    return {
        "source_split": split,
        "path_name": name,
        "source_dataset_index": source_index,
        "assigned_split": assigned_split,
        "diagnostic_classification": audit["diagnostic_classification"],
        "raw_rmse_rad": audit["raw_rmse_rad"],
        "wrapped_rmse_rad": audit["wrapped_rmse_rad"],
        "aligned_rmse_rad": audit["aligned_rmse_rad"],
        "split_seed": split_seed,
    }


def main() -> int:
    args = parse_args()
    validate_cli(args)
    existing = [args.output_dir / name for name in OUTPUT_FILENAMES]
    existing = [path for path in existing if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            f"Output files already exist: {existing}; pass --overwrite to replace them"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_prior = load_prior(
        args.train_prior, EXPECTED_TRAIN_PATHS, "train prior"
    )
    test_prior = load_prior(args.test_prior, EXPECTED_TEST_PATHS, "test prior")
    train_expert = load_train_expert(args.train_expert)
    test_reference = load_test_reference(args.test_reference)
    train = align_prior_to_reference(train_prior, train_expert, "train")
    test = align_prior_to_reference(test_prior, test_reference, "test")
    audit = load_audit(args.residual_audit_csv)

    expected_audit_names = {
        ("train", name) for name in train.names
    } | {("test", name) for name in test.names}
    actual_audit_names = set(
        zip(audit["split"].astype(str), audit["path_name"].astype(str))
    )
    if actual_audit_names != expected_audit_names:
        raise ValueError(
            "Audit CSV split/path set does not exactly match train and test sources"
        )
    audit_lookup: Dict[Tuple[str, str], Mapping[str, Any]] = {
        (str(row["split"]), str(row["path_name"])): row
        for row in audit.to_dict(orient="records")
    }
    eligible_names = tuple(
        sorted(
            row["path_name"]
            for row in audit.to_dict(orient="records")
            if row["split"] == "train"
            and row["diagnostic_classification"]
            == args.include_train_classification
        )
    )
    if (
        len(eligible_names) != EXPECTED_ELIGIBLE_PATHS
        and not args.allow_eligible_count_mismatch
    ):
        raise ValueError(
            f"Eligible train count is {len(eligible_names)}, expected "
            f"{EXPECTED_ELIGIBLE_PATHS}; pass --allow_eligible_count_mismatch "
            "only after reviewing the residual audit"
        )
    if not set(eligible_names).issubset(set(train.names)):
        raise ValueError("Eligible audit paths are not a subset of train paths")
    excluded_names = tuple(sorted(set(train.names) - set(eligible_names)))
    if not args.allow_eligible_count_mismatch:
        excluded_classifications = {
            str(audit_lookup[("train", name)]["diagnostic_classification"])
            for name in excluded_names
        }
        if excluded_classifications != {"requires_manual_review"}:
            raise ValueError(
                "The four excluded train paths are not all classified as "
                f"requires_manual_review: {sorted(excluded_classifications)}"
            )
    train_names, val_names = deterministic_path_split(
        eligible_names, args.val_count, args.split_seed
    )
    if set(train_names) & set(val_names):
        raise AssertionError("A path appears in both supervised train and validation")
    if set(train_names) | set(val_names) != set(eligible_names):
        raise AssertionError("Eligible paths were lost during path-level splitting")
    if set(excluded_names) & set(eligible_names):
        raise AssertionError("An excluded path appears in a supervised split")

    starts = expected_window_starts(args.horizon, args.stride)
    train_windows = build_windows(
        data=train,
        selected_names=train_names,
        starts=starts,
        horizon=args.horizon,
        supervised=True,
        assigned_split="train",
    )
    val_windows = build_windows(
        data=train,
        selected_names=val_names,
        starts=starts,
        horizon=args.horizon,
        supervised=True,
        assigned_split="val",
    )
    test_windows = build_windows(
        data=test,
        selected_names=test.names,
        starts=starts,
        horizon=args.horizon,
        supervised=False,
        assigned_split="test_inference",
    )

    condition_mean, condition_std = stable_stats(
        train_windows["condition_features"]
    )
    residual_mean, residual_std = stable_stats(
        train_windows["residual_q_window"]
    )
    normalize_windows(
        train_windows,
        condition_mean,
        condition_std,
        residual_mean,
        residual_std,
    )
    normalize_windows(
        val_windows,
        condition_mean,
        condition_std,
        residual_mean,
        residual_std,
    )
    normalize_windows(
        test_windows,
        condition_mean,
        condition_std,
        None,
        None,
    )

    if set(train_windows["path_names"]) != set(train_names):
        raise AssertionError("Normalization source paths differ from train manifest")
    if set(val_windows["path_names"]) & set(train_windows["path_names"]):
        raise AssertionError("Validation paths leaked into normalization train paths")
    validate_window_collection(
        label="train",
        windows=train_windows,
        path_names=train_names,
        starts=starts,
        condition_mean=condition_mean,
        condition_std=condition_std,
        residual_mean=residual_mean,
        residual_std=residual_std,
        supervised=True,
    )
    validate_window_collection(
        label="validation",
        windows=val_windows,
        path_names=val_names,
        starts=starts,
        condition_mean=condition_mean,
        condition_std=condition_std,
        residual_mean=residual_mean,
        residual_std=residual_std,
        supervised=True,
    )
    validate_window_collection(
        label="test_inference",
        windows=test_windows,
        path_names=test.names,
        starts=starts,
        condition_mean=condition_mean,
        condition_std=condition_std,
        residual_mean=None,
        residual_std=None,
        supervised=False,
    )

    forbidden_test_keys = {
        "residual_q_window",
        "residual_q_norm",
        "expert_q_window",
    }
    if forbidden_test_keys & set(test_windows):
        raise AssertionError("Test inference windows contain supervised target keys")

    train_index = {name: index for index, name in enumerate(train.names)}
    test_index = {name: index for index, name in enumerate(test.names)}
    manifest_rows: List[Dict[str, Any]] = []
    for name in train_names:
        manifest_rows.append(
            manifest_row(
                split="train",
                name=name,
                source_index=train_index[name],
                assigned_split="train",
                audit_lookup=audit_lookup,
                split_seed=args.split_seed,
            )
        )
    for name in val_names:
        manifest_rows.append(
            manifest_row(
                split="train",
                name=name,
                source_index=train_index[name],
                assigned_split="val",
                audit_lookup=audit_lookup,
                split_seed=args.split_seed,
            )
        )
    for name in excluded_names:
        manifest_rows.append(
            manifest_row(
                split="train",
                name=name,
                source_index=train_index[name],
                assigned_split="excluded_manual_review",
                audit_lookup=audit_lookup,
                split_seed=args.split_seed,
            )
        )
    for name in test.names:
        manifest_rows.append(
            manifest_row(
                split="test",
                name=name,
                source_index=test_index[name],
                assigned_split="test_inference",
                audit_lookup=audit_lookup,
                split_seed=args.split_seed,
            )
        )
    manifest = pd.DataFrame(manifest_rows).sort_values(
        ["assigned_split", "source_dataset_index", "path_name"]
    )
    if len(manifest) != EXPECTED_TRAIN_PATHS + EXPECTED_TEST_PATHS:
        raise AssertionError("Split manifest does not cover all source paths")
    if manifest.duplicated(["source_split", "path_name"]).any():
        raise AssertionError("A source split/path appears more than once in the manifest")
    test_manifest = manifest[manifest["source_split"] == "test"]
    if len(test_manifest) != EXPECTED_TEST_PATHS or set(
        test_manifest["assigned_split"]
    ) != {"test_inference"}:
        raise AssertionError(
            "Official test paths must appear exactly once and only in test_inference"
        )

    summary = pd.DataFrame(
        [
            {
                "split": "train",
                "path_count": len(train_names),
                "window_count": len(train_windows["path_names"]),
                "supervised_targets": True,
            },
            {
                "split": "val",
                "path_count": len(val_names),
                "window_count": len(val_windows["path_names"]),
                "supervised_targets": True,
            },
            {
                "split": "excluded_manual_review",
                "path_count": len(excluded_names),
                "window_count": 0,
                "supervised_targets": False,
            },
            {
                "split": "test_inference",
                "path_count": len(test.names),
                "window_count": len(test_windows["path_names"]),
                "supervised_targets": False,
            },
        ]
    )
    configuration = {
        "classification": "READY_FOR_V6_TRAINING",
        "input_files": {
            "train_prior": args.train_prior,
            "test_prior": args.test_prior,
            "train_expert": args.train_expert,
            "test_reference": args.test_reference,
            "residual_audit_csv": args.residual_audit_csv,
        },
        "discovered_npz_keys": {
            "train_prior": train_prior.keys,
            "test_prior": test_prior.keys,
            "train_expert": train_expert.keys,
            "test_reference": test_reference.keys,
        },
        "horizon": args.horizon,
        "stride": args.stride,
        "window_starts": starts,
        "windows_per_path": len(starts),
        "val_count": args.val_count,
        "split_seed": args.split_seed,
        "included_train_classification": args.include_train_classification,
        "eligible_train_path_count": len(eligible_names),
        "excluded_train_path_count": len(excluded_names),
        "supervised_train_path_count": len(train_names),
        "validation_path_count": len(val_names),
        "test_inference_path_count": len(test.names),
        "condition_feature_layout": [
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
        ],
        "condition_dim": CONDITION_DIM,
        "target_dim": TARGET_DIM,
        "q_start_source": "frozen_prior_q_at_timestep_0",
        "normalization_source": "supervised_train_paths_only",
        "test_expert_q_accessed": False,
        "residual_target": "raw_expert_q_minus_prior_q",
    }

    normalization_stats = {
        "condition_mean": condition_mean.astype(np.float32),
        "condition_std": condition_std.astype(np.float32),
        "residual_mean": residual_mean.astype(np.float32),
        "residual_std": residual_std.astype(np.float32),
        "epsilon": np.asarray(NORMALIZATION_EPSILON, dtype=np.float64),
        "train_path_count": np.asarray(len(train_names), dtype=np.int64),
        "train_window_count": np.asarray(
            len(train_windows["path_names"]), dtype=np.int64
        ),
        "condition_dim": np.asarray(CONDITION_DIM, dtype=np.int64),
        "target_dim": np.asarray(TARGET_DIM, dtype=np.int64),
    }
    test_reference_metadata = {
        "path_names": np.asarray(test.names),
        "path_indices": test.source_indices.astype(np.int64),
        "desired_paths": test.desired.astype(np.float32),
    }

    output_arrays = (
        ("train_windows.npz", train_windows),
        ("val_windows.npz", val_windows),
        ("test_inference_windows.npz", test_windows),
        ("test_reference_metadata.npz", test_reference_metadata),
        ("normalization_stats.npz", normalization_stats),
    )
    for filename, arrays in output_arrays:
        atomic_savez(
            args.output_dir / filename,
            {
                key: value.astype(np.float32)
                if np.issubdtype(value.dtype, np.floating)
                else value
                for key, value in arrays.items()
            },
        )
    atomic_csv(manifest, args.output_dir / "split_manifest.csv")
    atomic_json(
        {"rows": manifest.to_dict(orient="records")},
        args.output_dir / "split_manifest.json",
    )
    atomic_csv(
        manifest[manifest["assigned_split"] == "excluded_manual_review"],
        args.output_dir / "excluded_train_paths.csv",
    )
    atomic_csv(summary, args.output_dir / "dataset_summary.csv")
    atomic_json(
        configuration, args.output_dir / "dataset_configuration.json"
    )

    raw_residual_rmse = float(
        np.sqrt(np.mean(np.square(train_windows["residual_q_window"])))
    )
    normalized_residual_rmse = float(
        np.sqrt(np.mean(np.square(train_windows["residual_q_norm"])))
    )
    maximum_residual = float(
        np.max(np.abs(train_windows["residual_q_window"]))
    )
    print(
        f"paths: eligible={len(eligible_names)}, excluded={len(excluded_names)}, "
        f"train={len(train_names)}, val={len(val_names)}, test={len(test.names)}"
    )
    print(
        f"windows: train={len(train_windows['path_names'])}, "
        f"val={len(val_windows['path_names'])}, "
        f"test_inference={len(test_windows['path_names'])}"
    )
    print(f"condition_dim={CONDITION_DIM}, target_dim={TARGET_DIM}")
    print(
        f"train residual RMSE: raw={raw_residual_rmse:.8f}, "
        f"normalized={normalized_residual_rmse:.8f}"
    )
    print(f"maximum absolute train residual={maximum_residual:.8f} rad")
    for filename in OUTPUT_FILENAMES:
        print(f"output: {args.output_dir / filename}")
    print("classification: READY_FOR_V6_TRAINING")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
