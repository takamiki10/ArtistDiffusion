#!/usr/bin/env python3
"""Teacher-forced validation of v6 strong-prior residual diffusion.

This diagnostic deliberately loads only the supervised v6 validation windows.
Every candidate is reconstructed from the fixed prior window; corrections are
never propagated between windows and official test assets are never accessed.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from build_diffusion_v5b_residual_window_dataset_fk_condition import (
    fk_positions,
    load_fk_context,
)
from generate_ik_seed_path import (
    DEFAULT_JOINT_LIMIT_SAFETY_MARGIN_RAD,
    DEFAULT_JOINT_NAMES,
    DEFAULT_URDF_PATH,
    HARD_JOINT_LIMIT_TOLERANCE_RAD,
    check_joint_limits,
    get_joint_bounds,
)
from train_conditional_diffusion_trajectory_v6_strong_prior_residual_unet import (
    EXPECTED_CONDITION_DIM,
    EXPECTED_CONDITION_LAYOUT,
    EXPECTED_HORIZON,
    EXPECTED_TARGET_DIM,
    EXPECTED_VAL_WINDOWS,
    EXPECTED_WINDOWS_PER_PATH,
    EXPECTED_WINDOW_STARTS,
    PREDICTION_TARGET,
    build_schedule,
    instantiate_v5_model,
    predict_noise,
)


EXPECTED_PATHS = 42
DEFAULT_EXECUTION_HORIZON = 8
MEANINGFUL_ABSOLUTE_IMPROVEMENT_M = 5.0e-4
MEANINGFUL_RELATIVE_IMPROVEMENT = 0.01
SMOOTHNESS_RELATIVE_TOLERANCE = 0.10
BOUNDARY_ABSOLUTE_TOLERANCE_RAD = 0.01
EPS = 1.0e-12
BASELINE_CHECKPOINT = "validation_reference"
BASELINE_EPOCH = -1
REQUIRED_ARRAYS = (
    "prior_q_window",
    "desired_path_window",
    "prior_ee_window",
    "expert_q_window",
    "residual_q_window",
    "condition_features_norm",
    "path_names",
    "window_starts",
)
FORBIDDEN_VALIDATION_NAMES = (
    "test_inference_windows.npz",
    "test_prior.npz",
    "diffusion_test_v2.npz",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Teacher-forced v6 residual-diffusion validation only."
    )
    parser.add_argument("--val_npz", type=Path, required=True)
    parser.add_argument("--normalization_stats", type=Path, required=True)
    parser.add_argument("--dataset_configuration", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--checkpoint_labels", nargs="+", default=None)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--evaluate_raw_weights", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--evaluate_ema_weights", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--max_paths", type=int, default=None)
    parser.add_argument("--path_names", nargs="+", default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--sampler", choices=("ddpm", "ddim"), default="ddim")
    parser.add_argument("--sampling_steps", type=int, default=50)
    parser.add_argument("--ddim_eta", type=float, default=0.0)
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.05, 0.10, 0.25, 0.50, 1.00])
    parser.add_argument("--taper_lengths", type=int, nargs="+", default=[0, 4, 8])
    parser.add_argument("--execution_horizon", type=int, default=DEFAULT_EXECUTION_HORIZON)
    parser.add_argument("--max_joint_step_gate", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_cli(args: argparse.Namespace) -> None:
    if args.val_npz.name != "val_windows.npz" or any(
        token in str(args.val_npz) for token in FORBIDDEN_VALIDATION_NAMES
    ):
        raise ValueError("--val_npz must be the v6 val_windows.npz, never a test asset")
    if not args.evaluate_raw_weights and not args.evaluate_ema_weights:
        raise ValueError("At least one of raw or EMA weights must be enabled")
    if args.checkpoint_labels is not None and len(args.checkpoint_labels) != len(args.checkpoints):
        raise ValueError("--checkpoint_labels must have one label per checkpoint")
    if args.batch_size <= 0 or args.sampling_steps <= 0 or args.num_samples <= 0:
        raise ValueError("Batch size, sampling steps, and sample count must be positive")
    if not 1 <= args.execution_horizon < EXPECTED_HORIZON:
        raise ValueError(f"--execution_horizon must lie in [1,{EXPECTED_HORIZON - 1}]")
    if args.max_joint_step_gate != 0.20:
        raise ValueError("The validated maximum-joint-step gate must remain exactly 0.20 rad")
    if args.ddim_eta < 0.0:
        raise ValueError("--ddim_eta must be non-negative")
    if args.max_paths is not None and args.max_paths <= 0:
        raise ValueError("--max_paths must be positive")
    if any(alpha < 0.0 or not np.isfinite(alpha) for alpha in args.alphas):
        raise ValueError("All --alphas must be finite and non-negative")
    if any(length < 0 or length >= EXPECTED_HORIZON for length in args.taper_lengths):
        raise ValueError("Taper lengths must lie in [0,H-1]")
    if len(set(args.alphas)) != len(args.alphas) or len(set(args.taper_lengths)) != len(args.taper_lengths):
        raise ValueError("Alpha and taper lists may not contain duplicates")


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(value)


def load_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def decode_names(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [item.decode("utf-8", errors="strict") if isinstance(item, bytes) else str(item)
         for item in np.asarray(values).reshape(-1)],
        dtype=str,
    )


def finite(name: str, values: np.ndarray) -> None:
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} contains nonfinite values")


def authoritative_joint_limits(
    robot: Any, joint_names: Sequence[str]
) -> Tuple[np.ndarray, np.ndarray]:
    expected = tuple(DEFAULT_JOINT_NAMES)
    actual = tuple(str(name) for name in joint_names)
    if expected != tuple(f"joint{i}" for i in range(1, 7)):
        raise ValueError(f"Authoritative active-joint order is invalid: {expected}")
    if actual != expected:
        raise ValueError(
            f"Evaluator joint order {actual} differs from prior generator order {expected}"
        )
    bounds = get_joint_bounds(robot, expected, -np.pi, np.pi)
    return (
        np.asarray([bound[0] for bound in bounds], dtype=np.float64),
        np.asarray([bound[1] for bound in bounds], dtype=np.float64),
    )


def serialized_limit_columns(limits: Mapping[str, Any]) -> Dict[str, Any]:
    hard = list(limits["hard_violations"])
    return {
        "hard_joint_limit_violation_count": int(
            limits["hard_joint_limit_violation_count"]
        ),
        "hard_joint_limit_violation_magnitude": float(
            limits["hard_joint_limit_violation_magnitude"]
        ),
        "safety_margin_violation_count": int(
            limits["safety_margin_violation_count"]
        ),
        "minimum_joint_limit_margin_rad": float(
            limits["minimum_joint_limit_margin_rad"]
        ),
        "violating_joint_names": json.dumps(
            sorted({str(item["joint_name"]) for item in hard})
        ),
        "violating_timesteps": json.dumps(
            sorted({int(item["timestep"]) for item in hard})
        ),
        "violating_joint_values": json.dumps(
            [float(item["joint_value"]) for item in hard]
        ),
        "hard_lower_limits": json.dumps(limits["hard_lower_limits"]),
        "hard_upper_limits": json.dumps(limits["hard_upper_limits"]),
        # Compatibility names are explicitly hard-limit metrics.
        "joint_limit_violation_count": int(
            limits["hard_joint_limit_violation_count"]
        ),
        "joint_limit_violation_magnitude": float(
            limits["hard_joint_limit_violation_magnitude"]
        ),
    }


def hard_limit_consistency_audit(
    data: Mapping[str, np.ndarray],
    joint_names: Sequence[str],
    lower: np.ndarray,
    upper: np.ndarray,
    output_path: Path,
) -> Dict[str, Any]:
    hard_records: List[Dict[str, Any]] = []
    safety_records: List[Dict[str, Any]] = []
    legacy_exact_records: List[Dict[str, Any]] = []
    path_0067: Dict[Tuple[int, int], Dict[str, Any]] = {}
    alpha_zero_max_abs_difference = 0.0

    for index, (path_name_value, window_start_value) in enumerate(
        zip(data["path_names"], data["window_starts"])
    ):
        path_name = str(path_name_value)
        window_start = int(window_start_value)
        prior = np.asarray(data["prior_q_window"][index])
        alpha_zero_candidate = prior + np.zeros_like(prior)
        difference = np.abs(
            alpha_zero_candidate.astype(np.float64) - prior.astype(np.float64)
        )
        alpha_zero_max_abs_difference = max(
            alpha_zero_max_abs_difference,
            float(np.max(difference)) if difference.size else 0.0,
        )
        if not np.allclose(
            alpha_zero_candidate, prior, atol=1.0e-8, rtol=1.0e-7
        ):
            raise AssertionError(
                f"alpha=0 candidate differs from prior for {path_name} "
                f"window_start={window_start}"
            )
        limits = check_joint_limits(
            prior,
            lower,
            upper,
            joint_names,
            tolerance=HARD_JOINT_LIMIT_TOLERANCE_RAD,
            safety_margin=DEFAULT_JOINT_LIMIT_SAFETY_MARGIN_RAD,
        )

        def contextualize(item: Mapping[str, Any]) -> Dict[str, Any]:
            record = dict(item)
            offset = int(record.pop("timestep"))
            record.update({
                "path_name": path_name,
                "window_start": window_start,
                "window_offset": offset,
                "trajectory_timestep": window_start + offset,
            })
            return record

        hard_records.extend(
            contextualize(item) for item in limits["hard_violations"]
        )
        safety_records.extend(
            contextualize(item) for item in limits["safety_margin_violations"]
        )
        prior64 = prior.astype(np.float64, copy=False)
        exact_mask = (
            (prior64 < lower[None, :]) | (prior64 > upper[None, :])
        )
        for offset_value, joint_index_value in np.argwhere(exact_mask):
            offset = int(offset_value)
            joint_index = int(joint_index_value)
            value = float(prior64[offset, joint_index])
            record = {
                "path_name": path_name,
                "window_start": window_start,
                "window_offset": offset,
                "trajectory_timestep": window_start + offset,
                "joint_index": joint_index,
                "joint_name": str(joint_names[joint_index]),
                "joint_value": value,
                "lower_limit": float(lower[joint_index]),
                "upper_limit": float(upper[joint_index]),
                "value_minus_lower_limit": value - float(lower[joint_index]),
                "upper_limit_minus_value": float(upper[joint_index]) - value,
                "within_authoritative_tolerance": bool(
                    value >= float(lower[joint_index]) - HARD_JOINT_LIMIT_TOLERANCE_RAD
                    and value <= float(upper[joint_index]) + HARD_JOINT_LIMIT_TOLERANCE_RAD
                ),
            }
            legacy_exact_records.append(record)
        if path_name == "path_0067":
            for item in list(limits["hard_violations"]) + list(
                limits["safety_margin_violations"]
            ):
                offset = int(item["timestep"])
                key = (window_start + offset, int(item["joint_index"]))
                entry = path_0067.setdefault(key, {
                    **dict(item),
                    "trajectory_timestep": window_start + offset,
                    "window_starts": [],
                })
                entry["window_starts"].append(window_start)

    hard_keys = {
        (
            row["path_name"], row["window_start"], row["trajectory_timestep"],
            row["joint_name"],
        )
        for row in hard_records
    }
    hard_windows = {
        (row["path_name"], row["window_start"]) for row in hard_records
    }
    safety_keys = {
        (
            row["path_name"], row["window_start"], row["trajectory_timestep"],
            row["joint_name"],
        )
        for row in safety_records
    }
    hard_by_path = Counter(str(row[0]) for row in hard_keys)
    hard_by_joint = Counter(str(row[3]) for row in hard_keys)
    hard_by_window = Counter((str(row[0]), int(row[1])) for row in hard_keys)
    hard_by_timestep = Counter(
        (str(row[0]), int(row[2])) for row in hard_keys
    )
    audit = {
        "authoritative_source": (
            "generate_ik_seed_path.get_joint_bounds/check_joint_limits; "
            "also used by generate_adaptive_mlp_ik_bootstrap_prior"
        ),
        "authoritative_urdf": DEFAULT_URDF_PATH,
        "authoritative_joint_names": list(joint_names),
        "hard_lower_limits": lower.tolist(),
        "hard_upper_limits": upper.tolist(),
        "tolerance": HARD_JOINT_LIMIT_TOLERANCE_RAD,
        "safety_margin_rad": DEFAULT_JOINT_LIMIT_SAFETY_MARGIN_RAD,
        "policy_comparison": {
            "prior_generator_before_shared_checker": {
                "joint_names": list(DEFAULT_JOINT_NAMES),
                "hard_lower_limits": lower.tolist(),
                "hard_upper_limits": upper.tolist(),
                "input_precision": "generated trajectory retained as float64",
                "hard_limit_tolerance_rad": 0.0,
                "safety_margin_rad": DEFAULT_JOINT_LIMIT_SAFETY_MARGIN_RAD,
                "safety_margin_usage": (
                    "optional repair/margin gate only; not counted as a hard violation"
                ),
            },
            "teacher_forced_evaluator_before_shared_checker": {
                "joint_names": list(joint_names),
                "hard_lower_limits": lower.tolist(),
                "hard_upper_limits": upper.tolist(),
                "input_precision": "candidate forcibly converted to float32",
                "hard_limit_tolerance_rad": 0.0,
                "safety_margin_rad": None,
                "failure_mode": (
                    "exact comparison classified float32 serialization below "
                    "joint6 lower=-6.1082 as a hard violation"
                ),
            },
            "shared_authoritative_policy_after_fix": {
                "joint_names": list(joint_names),
                "hard_lower_limits": lower.tolist(),
                "hard_upper_limits": upper.tolist(),
                "hard_limit_tolerance_rad": HARD_JOINT_LIMIT_TOLERANCE_RAD,
                "safety_margin_rad": DEFAULT_JOINT_LIMIT_SAFETY_MARGIN_RAD,
                "safety_margin_is_hard_violation": False,
            },
        },
        "prior_window_count": int(len(data["path_names"])),
        "hard_violation_count": int(len(hard_keys)),
        "hard_violation_window_count": int(len(hard_windows)),
        "hard_violation_count_by_path": dict(sorted(hard_by_path.items())),
        "hard_violation_count_by_window": [
            {"path_name": path, "window_start": start, "count": count}
            for (path, start), count in sorted(hard_by_window.items())
        ],
        "hard_violation_count_by_timestep": [
            {"path_name": path, "trajectory_timestep": timestep, "count": count}
            for (path, timestep), count in sorted(hard_by_timestep.items())
        ],
        "hard_violation_count_by_joint": dict(sorted(hard_by_joint.items())),
        "safety_margin_violation_count": int(len(safety_keys)),
        "alpha_zero_atol": 1.0e-8,
        "alpha_zero_rtol": 1.0e-7,
        "alpha_zero_max_abs_difference": alpha_zero_max_abs_difference,
        "legacy_zero_tolerance_violation_count": int(len(legacy_exact_records)),
        "legacy_zero_tolerance_details": legacy_exact_records,
        "path_0067_legacy_zero_tolerance_details": [
            row for row in legacy_exact_records if row["path_name"] == "path_0067"
        ],
        "hard_violations": hard_records,
        "path_0067_values_and_limit_comparison": [
            {**value, "window_starts": sorted(set(value["window_starts"]))}
            for _, value in sorted(path_0067.items())
        ],
    }
    atomic_json(audit, output_path)
    if hard_records:
        first = hard_records[0]
        raise RuntimeError(
            "Frozen validation prior fails the authoritative hard-limit audit "
            f"before diffusion sampling: {len(hard_keys)} violations across "
            f"{len(hard_windows)} windows; first={first}"
        )
    return audit


def load_validation(path: Path) -> Dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=True) as archive:
        missing = sorted(set(REQUIRED_ARRAYS) - set(archive.files))
        if missing:
            raise KeyError(f"Validation archive is missing {missing}")
        data = {key: np.asarray(archive[key]) for key in archive.files}
    data["path_names"] = decode_names(data["path_names"])
    data["window_starts"] = np.asarray(data["window_starts"], dtype=np.int64)
    expected_joint = (EXPECTED_VAL_WINDOWS, EXPECTED_HORIZON, EXPECTED_TARGET_DIM)
    expected_xyz = (EXPECTED_VAL_WINDOWS, EXPECTED_HORIZON, 3)
    expected_condition = (EXPECTED_VAL_WINDOWS, EXPECTED_HORIZON, EXPECTED_CONDITION_DIM)
    for key in ("prior_q_window", "expert_q_window", "residual_q_window"):
        if data[key].shape != expected_joint:
            raise ValueError(f"{key} has shape {data[key].shape}, expected {expected_joint}")
    for key in ("desired_path_window", "prior_ee_window"):
        if data[key].shape != expected_xyz:
            raise ValueError(f"{key} has shape {data[key].shape}, expected {expected_xyz}")
    if data["condition_features_norm"].shape != expected_condition:
        raise ValueError("condition_features_norm does not have shape (756,32,38)")
    if data["path_names"].shape != (EXPECTED_VAL_WINDOWS,) or data["window_starts"].shape != (EXPECTED_VAL_WINDOWS,):
        raise ValueError("path_names/window_starts do not have 756 entries")
    for key, values in data.items():
        if np.issubdtype(values.dtype, np.number):
            finite(f"validation/{key}", values)
    if not np.allclose(
        data["expert_q_window"] - data["prior_q_window"],
        data["residual_q_window"], rtol=1e-6, atol=1e-7,
    ):
        raise ValueError("Validation residual_q_window is inconsistent with expert-prior")
    names = sorted(set(data["path_names"].tolist()))
    if len(names) != EXPECTED_PATHS:
        raise ValueError(f"Expected 42 validation paths, found {len(names)}")
    for name in names:
        starts = data["window_starts"][data["path_names"] == name]
        if len(starts) != EXPECTED_WINDOWS_PER_PATH or not np.array_equal(starts, EXPECTED_WINDOW_STARTS):
            raise ValueError(f"{name} does not have the validated 18 starts 0,4,...,68")
    return data


def load_stats(path: Path) -> Dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        stats = {key: np.asarray(archive[key]) for key in archive.files}
    required = ("condition_mean", "condition_std", "residual_mean", "residual_std")
    missing = sorted(set(required) - set(stats))
    if missing:
        raise KeyError(f"Normalization archive is missing {missing}")
    for key in required:
        finite(f"normalization/{key}", stats[key])
    for key, size in (("condition_mean", 38), ("condition_std", 38), ("residual_mean", 6), ("residual_std", 6)):
        stats[key] = np.asarray(stats[key], dtype=np.float32).reshape(-1)
        if stats[key].shape != (size,):
            raise ValueError(f"{key} must reduce to shape ({size},)")
    if np.any(stats["condition_std"] <= 0.0) or np.any(stats["residual_std"] <= 0.0):
        raise ValueError("Normalization standard deviations must be positive")
    for key, expected in (("train_path_count", 372), ("train_window_count", 6696),
                          ("condition_dim", 38), ("target_dim", 6)):
        if key not in stats or int(np.asarray(stats[key]).item()) != expected:
            raise ValueError(f"Normalization metadata {key} is not {expected}")
    return stats


def validate_dataset_configuration(config: Mapping[str, Any]) -> None:
    expected = {
        "classification": "READY_FOR_V6_TRAINING",
        "horizon": EXPECTED_HORIZON,
        "condition_dim": EXPECTED_CONDITION_DIM,
        "target_dim": EXPECTED_TARGET_DIM,
        "windows_per_path": EXPECTED_WINDOWS_PER_PATH,
        "validation_path_count": EXPECTED_PATHS,
        "normalization_source": "supervised_train_paths_only",
        "residual_target": "raw_expert_q_minus_prior_q",
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"Dataset configuration {key}={config.get(key)!r}, expected {value!r}")
    if list(config.get("window_starts", [])) != EXPECTED_WINDOW_STARTS.tolist():
        raise ValueError("Dataset configuration has incompatible window starts")
    if tuple(config.get("condition_feature_layout", ())) != tuple(EXPECTED_CONDITION_LAYOUT):
        raise ValueError("Dataset condition layout is not the validated 38-D v6 layout")


def select_paths(data: Dict[str, np.ndarray], requested: Optional[Sequence[str]], max_paths: Optional[int]) -> Dict[str, np.ndarray]:
    available = sorted(set(data["path_names"].tolist()))
    chosen = available
    if requested is not None:
        unknown = sorted(set(requested) - set(available))
        if unknown:
            raise ValueError(f"Requested validation paths are absent: {unknown}")
        chosen = [name for name in requested if name in available]
    if max_paths is not None:
        chosen = chosen[:max_paths]
    if not chosen:
        raise ValueError("Path filtering selected no validation paths")
    mask = np.isin(data["path_names"], np.asarray(chosen))
    return {key: values[mask] if values.shape[:1] == mask.shape else values for key, values in data.items()}


def torch_load(path: Path, device: torch.device) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        value = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        value = torch.load(path, map_location=device)
    if not isinstance(value, dict):
        raise TypeError(f"Checkpoint {path} is not a dictionary")
    return value


def checkpoint_schedule(checkpoint: Mapping[str, Any]) -> Dict[str, Any]:
    diffusion = checkpoint.get("diffusion_hyperparameters")
    if not isinstance(diffusion, Mapping):
        raise KeyError("Checkpoint lacks diffusion_hyperparameters")
    result = {
        "steps": int(diffusion.get("steps", -1)),
        "beta_schedule": str(diffusion.get("beta_schedule", "")),
        "beta_start": float(diffusion.get("beta_start", math.nan)),
        "beta_end": float(diffusion.get("beta_end", math.nan)),
    }
    if result["steps"] <= 0 or result["beta_schedule"] != "linear":
        raise ValueError(f"Unsupported checkpoint diffusion scheduler: {result}")
    if not np.isclose(result["beta_start"], 1e-4) or not np.isclose(result["beta_end"], 2e-2):
        raise ValueError(f"Checkpoint has incompatible beta schedule: {result}")
    return result


def validate_checkpoint(
    checkpoint: Mapping[str, Any], checkpoint_path: Path, val_path: Path,
    stats_path: Path, stats: Mapping[str, np.ndarray], dataset_config: Mapping[str, Any],
) -> Dict[str, Any]:
    for key, expected in (("horizon", 32), ("condition_dim", 38), ("target_dim", 6)):
        if int(checkpoint.get(key, -1)) != expected:
            raise ValueError(f"{checkpoint_path.name}: incompatible {key}")
    if checkpoint.get("prediction_target_type") != PREDICTION_TARGET:
        raise ValueError(f"{checkpoint_path.name}: prediction target is not epsilon")
    schedule = checkpoint_schedule(checkpoint)
    checkpoint_val = checkpoint.get("val_npz")
    if checkpoint_val is None or Path(str(checkpoint_val)).expanduser().resolve() != val_path.resolve():
        raise ValueError(f"{checkpoint_path.name}: validation provenance does not match --val_npz")
    checkpoint_stats = checkpoint.get("normalization_statistics")
    if not isinstance(checkpoint_stats, Mapping):
        raise KeyError(f"{checkpoint_path.name}: normalization_statistics are absent")
    for key in ("condition_mean", "condition_std", "residual_mean", "residual_std"):
        left = np.asarray(checkpoint_stats.get(key), dtype=np.float32).reshape(-1)
        right = np.asarray(stats[key], dtype=np.float32).reshape(-1)
        if left.shape != right.shape or not np.allclose(left, right, rtol=1e-6, atol=1e-7):
            raise ValueError(f"{checkpoint_path.name}: normalization field {key} is incompatible")
    source = checkpoint.get("normalization_source_path")
    if source is None or Path(str(source)).expanduser().resolve() != stats_path.resolve():
        raise ValueError(f"{checkpoint_path.name}: normalization source path is incompatible")
    embedded = checkpoint.get("dataset_configuration")
    if not isinstance(embedded, Mapping):
        raise KeyError(f"{checkpoint_path.name}: embedded dataset configuration is absent")
    for key in ("classification", "normalization_source", "horizon", "condition_dim", "target_dim", "residual_target"):
        if embedded.get(key) != dataset_config.get(key):
            raise ValueError(f"{checkpoint_path.name}: embedded dataset {key} is incompatible")
    return schedule


def checkpoint_states(checkpoint: Mapping[str, Any], raw: bool, ema: bool) -> Tuple[List[Tuple[str, Mapping[str, torch.Tensor]]], List[str]]:
    states: List[Tuple[str, Mapping[str, torch.Tensor]]] = []
    missing: List[str] = []
    selected = str(checkpoint.get("selected_model_state", ""))
    if raw:
        raw_state = checkpoint.get("raw_model_state_dict")
        if raw_state is None and selected == "raw":
            raw_state = checkpoint.get("model_state_dict")
        if isinstance(raw_state, Mapping):
            states.append(("raw", raw_state))
        else:
            missing.append("raw")
    if ema:
        ema_container = checkpoint.get("ema_state_dict")
        ema_state = ema_container.get("shadow") if isinstance(ema_container, Mapping) else None
        if ema_state is None and selected == "ema":
            ema_state = checkpoint.get("model_state_dict")
        if isinstance(ema_state, Mapping):
            states.append(("ema", ema_state))
        else:
            missing.append("ema")
    return states, missing


def stable_seed(global_seed: int, checkpoint_label: str, path_name: str, start: int, sample_index: int, extra: str = "initial") -> int:
    payload = json.dumps(
        [int(global_seed), checkpoint_label, path_name, int(start), int(sample_index), extra],
        separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def seeded_noise(shape: Sequence[int], seed: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.randn(tuple(shape), generator=generator, dtype=torch.float32).to(device)


@torch.no_grad()
def sample_batch(
    model: torch.nn.Module, condition: np.ndarray, seeds: Sequence[int], schedule: Any,
    sampling_steps: int, sampler: str, ddim_eta: float, device: torch.device,
) -> np.ndarray:
    total_steps = int(schedule.alpha_bars.shape[0])
    if sampling_steps > total_steps:
        raise ValueError(f"--sampling_steps={sampling_steps} exceeds checkpoint steps={total_steps}")
    indices = np.rint(np.linspace(total_steps - 1, 0, sampling_steps)).astype(np.int64)
    indices = np.asarray(list(dict.fromkeys(indices.tolist())), dtype=np.int64)
    if indices[-1] != 0:
        indices = np.concatenate([indices, np.asarray([0], dtype=np.int64)])
    condition_t = torch.from_numpy(condition.astype(np.float32)).to(device)
    x = torch.stack([seeded_noise((EXPECTED_HORIZON, EXPECTED_TARGET_DIM), seed, device) for seed in seeds])
    eta = 1.0 if sampler == "ddpm" else float(ddim_eta)
    model.eval()
    for position, timestep in enumerate(indices):
        t = torch.full((x.shape[0],), int(timestep), dtype=torch.long, device=device)
        predicted_epsilon = predict_noise(model, x, t, condition_t)
        alpha_bar_t = schedule.alpha_bars[int(timestep)]
        previous_t = int(indices[position + 1]) if position + 1 < len(indices) else -1
        alpha_bar_previous = (
            schedule.alpha_bars[previous_t] if previous_t >= 0
            else torch.ones((), dtype=torch.float32, device=device)
        )
        predicted_x0 = (x - torch.sqrt(1.0 - alpha_bar_t) * predicted_epsilon) / torch.sqrt(alpha_bar_t)
        sigma = eta * torch.sqrt(torch.clamp(
            (1.0 - alpha_bar_previous) / (1.0 - alpha_bar_t)
            * (1.0 - alpha_bar_t / alpha_bar_previous), min=0.0,
        ))
        direction = torch.sqrt(torch.clamp(1.0 - alpha_bar_previous - sigma * sigma, min=0.0)) * predicted_epsilon
        if previous_t >= 0 and float(sigma) > 0.0:
            noises = [
                seeded_noise((EXPECTED_HORIZON, EXPECTED_TARGET_DIM),
                             stable_seed(0, str(seed), "reverse", int(timestep), position, "noise"), device)
                for seed in seeds
            ]
            noise = torch.stack(noises)
        else:
            noise = torch.zeros_like(x)
        x = torch.sqrt(alpha_bar_previous) * predicted_x0 + direction + sigma * noise
    return x.detach().cpu().numpy().astype(np.float32)


def cosine_taper(length: int, horizon: int = EXPECTED_HORIZON) -> np.ndarray:
    if length == 0:
        return np.ones((horizon, 1), dtype=np.float32)
    values = np.ones(horizon, dtype=np.float32)
    if length == 1:
        values[0] = 0.0
    else:
        phase = np.arange(length, dtype=np.float64) / float(length - 1)
        values[:length] = (0.5 - 0.5 * np.cos(np.pi * phase)).astype(np.float32)
    return values[:, None]


def derivative_metrics(q: np.ndarray) -> Dict[str, float]:
    q64 = np.asarray(q, dtype=np.float64)
    differences = [np.diff(q64, n=order, axis=0) for order in (1, 2, 3)]
    velocity, acceleration, jerk = differences
    def cost(values: np.ndarray) -> float:
        return float(np.mean(np.sum(np.square(values), axis=1))) if values.size else 0.0
    return {
        "velocity_cost": cost(velocity),
        "acceleration_cost": cost(acceleration),
        "jerk_cost": cost(jerk),
        "max_absolute_joint_step_internal": float(np.max(np.abs(velocity))) if velocity.size else 0.0,
        "max_l2_joint_step_internal": float(np.max(np.linalg.norm(velocity, axis=1))) if velocity.size else 0.0,
    }


def cartesian_summary(errors: np.ndarray) -> Dict[str, float]:
    values = np.asarray(errors, dtype=np.float64)
    return {
        "mean_cartesian_error": float(np.mean(values)),
        "rms_cartesian_error": float(np.sqrt(np.mean(np.square(values)))),
        "median_cartesian_error": float(np.median(values)),
        "p95_cartesian_error": float(np.percentile(values, 95.0)),
        "max_cartesian_error": float(np.max(values)),
    }


def reconstruct_prior_paths(data: Mapping[str, np.ndarray]) -> Dict[str, Dict[int, np.ndarray]]:
    result: Dict[str, Dict[int, np.ndarray]] = {}
    for index, (name, start) in enumerate(zip(data["path_names"], data["window_starts"])):
        timeline = result.setdefault(str(name), {})
        for offset, q in enumerate(data["prior_q_window"][index]):
            timestep = int(start) + offset
            if timestep in timeline and not np.allclose(timeline[timestep], q, rtol=1e-6, atol=1e-7):
                raise ValueError(f"Prior overlap mismatch for {name} at timestep {timestep}")
            timeline[timestep] = np.asarray(q, dtype=np.float32)
    return result


def evaluate_candidate(
    q: np.ndarray, desired: np.ndarray, prior: np.ndarray, prior_previous: Optional[np.ndarray],
    execution_horizon: int, robot: Any, joint_names: Sequence[str], ee_link: str,
    lower: np.ndarray, upper: np.ndarray,
) -> Dict[str, Any]:
    q = np.asarray(q, dtype=np.float32)
    nonfinite_count = int(np.size(q) - np.count_nonzero(np.isfinite(q)))
    result: Dict[str, Any] = {"nonfinite_count": nonfinite_count}
    if nonfinite_count:
        for region in ("prefix", "full"):
            for key in (
                "mean_cartesian_error", "rms_cartesian_error", "median_cartesian_error",
                "p95_cartesian_error", "max_cartesian_error", "velocity_cost",
                "acceleration_cost", "jerk_cost", "max_absolute_joint_step_internal",
                "max_l2_joint_step_internal",
            ):
                result[f"{region}_{key}"] = math.inf
        result.update({
            "prefix_joint_limit_violation_count": 0,
            "prefix_joint_limit_violation_magnitude": 0.0,
            "full_joint_limit_violation_count": 0,
            "full_joint_limit_violation_magnitude": 0.0,
            "hard_joint_limit_violation_count": 0,
            "hard_joint_limit_violation_magnitude": 0.0,
            "safety_margin_violation_count": 0,
            "minimum_joint_limit_margin_rad": math.nan,
            "violating_joint_names": "[]",
            "violating_timesteps": "[]",
            "violating_joint_values": "[]",
            "hard_lower_limits": json.dumps(np.asarray(lower).tolist()),
            "hard_upper_limits": json.dumps(np.asarray(upper).tolist()),
            "joint_limit_violation_count": 0,
            "joint_limit_violation_magnitude": 0.0,
            "entry_boundary_available": int(prior_previous is not None),
            "entry_boundary_max_abs_step": math.inf,
            "entry_boundary_l2_step": math.inf,
            "exit_boundary_max_abs_step": math.inf,
            "exit_boundary_l2_step": math.inf,
            "max_boundary_step": math.inf,
            "max_boundary_l2_step": math.inf,
            "transition_acceleration_discontinuity_max_abs": math.inf,
            "transition_acceleration_discontinuity_l2": math.inf,
            "boundary_finite": 0,
            "gated_max_absolute_joint_step": math.inf,
            "gated_max_l2_joint_step": math.inf,
        })
        result["_prefix_errors"] = np.full(execution_horizon, np.inf, dtype=np.float64)
        result["_full_errors"] = np.full(EXPECTED_HORIZON, np.inf, dtype=np.float64)
        return result
    ee = fk_positions(robot, joint_names, ee_link, q)
    errors = np.linalg.norm(ee - desired, axis=1).astype(np.float64)
    prefix = q[:execution_horizon]
    prefix_errors = errors[:execution_horizon]
    prefix_limits = check_joint_limits(
        prefix,
        lower,
        upper,
        joint_names,
        tolerance=HARD_JOINT_LIMIT_TOLERANCE_RAD,
        safety_margin=DEFAULT_JOINT_LIMIT_SAFETY_MARGIN_RAD,
    )
    full_limits = check_joint_limits(
        q,
        lower,
        upper,
        joint_names,
        tolerance=HARD_JOINT_LIMIT_TOLERANCE_RAD,
        safety_margin=DEFAULT_JOINT_LIMIT_SAFETY_MARGIN_RAD,
    )
    entry_vector = None if prior_previous is None else prefix[0] - prior_previous
    exit_vector = prior[execution_horizon] - prefix[-1]
    entry_abs = math.nan if entry_vector is None else float(np.max(np.abs(entry_vector)))
    entry_l2 = math.nan if entry_vector is None else float(np.linalg.norm(entry_vector))
    exit_abs = float(np.max(np.abs(exit_vector)))
    exit_l2 = float(np.linalg.norm(exit_vector))
    internal = derivative_metrics(prefix)
    finite_boundaries = np.isfinite(exit_abs) and (prior_previous is None or np.isfinite(entry_abs))
    boundary_abs_values = [exit_abs] + ([] if entry_vector is None else [entry_abs])
    boundary_l2_values = [exit_l2] + ([] if entry_vector is None else [entry_l2])
    previous_velocity = prefix[-1] - prefix[-2] if execution_horizon >= 2 else np.zeros(6)
    acceleration_jump = exit_vector - previous_velocity
    result.update({f"prefix_{key}": value for key, value in cartesian_summary(prefix_errors).items()})
    result.update({f"full_{key}": value for key, value in cartesian_summary(errors).items()})
    result.update({f"prefix_{key}": value for key, value in internal.items()})
    result.update({f"full_{key}": value for key, value in derivative_metrics(q).items()})
    result.update({
        "prefix_joint_limit_violation_count": int(
            prefix_limits["hard_joint_limit_violation_count"]
        ),
        "prefix_joint_limit_violation_magnitude": float(
            prefix_limits["hard_joint_limit_violation_magnitude"]
        ),
        "full_joint_limit_violation_count": int(
            full_limits["hard_joint_limit_violation_count"]
        ),
        "full_joint_limit_violation_magnitude": float(
            full_limits["hard_joint_limit_violation_magnitude"]
        ),
        "entry_boundary_available": int(prior_previous is not None),
        "entry_boundary_max_abs_step": entry_abs,
        "entry_boundary_l2_step": entry_l2,
        "exit_boundary_max_abs_step": exit_abs,
        "exit_boundary_l2_step": exit_l2,
        "max_boundary_step": float(max(boundary_abs_values)),
        "max_boundary_l2_step": float(max(boundary_l2_values)),
        "transition_acceleration_discontinuity_max_abs": float(np.max(np.abs(acceleration_jump))),
        "transition_acceleration_discontinuity_l2": float(np.linalg.norm(acceleration_jump)),
        "boundary_finite": int(finite_boundaries),
        "gated_max_absolute_joint_step": float(max(internal["max_absolute_joint_step_internal"], *boundary_abs_values)),
        "gated_max_l2_joint_step": float(max(internal["max_l2_joint_step_internal"], *boundary_l2_values)),
        "_prefix_errors": prefix_errors,
        "_full_errors": errors,
    })
    result.update(serialized_limit_columns(full_limits))
    return result


def diagnostic_metrics(sample_residual: Optional[np.ndarray], candidate_q: np.ndarray,
                       expert_residual: np.ndarray, expert_q: np.ndarray) -> Dict[str, float]:
    result: Dict[str, float] = {}
    if sample_residual is None:
        residual_error = np.full_like(expert_residual, np.nan, dtype=np.float64)
    else:
        residual_error = np.asarray(sample_residual, dtype=np.float64) - expert_residual
    joint_error = np.asarray(candidate_q, dtype=np.float64) - expert_q
    result["sampled_residual_rmse"] = float(np.sqrt(np.nanmean(np.square(residual_error)))) if np.any(np.isfinite(residual_error)) else math.nan
    result["sampled_residual_mae"] = float(np.nanmean(np.abs(residual_error))) if np.any(np.isfinite(residual_error)) else math.nan
    result["candidate_joint_rmse"] = float(np.sqrt(np.mean(np.square(joint_error))))
    for joint in range(EXPECTED_TARGET_DIM):
        values = residual_error[:, joint]
        result[f"sampled_residual_q{joint + 1}_rmse"] = float(np.sqrt(np.nanmean(np.square(values)))) if np.any(np.isfinite(values)) else math.nan
        result[f"sampled_residual_q{joint + 1}_mae"] = float(np.nanmean(np.abs(values))) if np.any(np.isfinite(values)) else math.nan
        result[f"candidate_q{joint + 1}_rmse"] = float(np.sqrt(np.mean(np.square(joint_error[:, joint]))))
    return result


def rejection_reasons(metrics: Mapping[str, Any], prior_mean: float, step_gate: float) -> List[str]:
    reasons: List[str] = []
    if int(metrics.get("nonfinite_count", 0)):
        return ["nonfinite_values"]
    if int(metrics["prefix_joint_limit_violation_count"]) > 0:
        reasons.append("joint_limit_violation")
    if float(metrics["gated_max_absolute_joint_step"]) > step_gate:
        reasons.append("maximum_joint_step_gate")
    if not bool(metrics["boundary_finite"]):
        reasons.append("nonfinite_boundary")
    catastrophic_threshold = max(2.0 * prior_mean, prior_mean + 0.01)
    if float(metrics["prefix_mean_cartesian_error"]) > catastrophic_threshold:
        reasons.append("catastrophic_cartesian_degradation")
    return reasons


def row_identity(checkpoint_file: str, checkpoint_label: str, epoch: int, weight_state: str,
                 path_name: str, start: int, alpha: float, taper: int, mode: str) -> Dict[str, Any]:
    return {
        "checkpoint_file": checkpoint_file,
        "checkpoint": checkpoint_label,
        "checkpoint_epoch": int(epoch),
        "weight_state": weight_state,
        "path_name": path_name,
        "window_start": int(start),
        "alpha": float(alpha),
        "taper_length": int(taper),
        "selection_mode": mode,
    }


def baseline_rows(
    data: Mapping[str, np.ndarray], prior_paths: Mapping[str, Mapping[int, np.ndarray]],
    residual_mean: np.ndarray, execution_horizon: int, robot: Any,
    joint_names: Sequence[str], ee_link: str, lower: np.ndarray, upper: np.ndarray,
) -> Tuple[List[Dict[str, Any]], Dict[int, Dict[str, Any]]]:
    rows: List[Dict[str, Any]] = []
    prior_by_index: Dict[int, Dict[str, Any]] = {}
    definitions = (
        ("strong_prior", lambda i: data["prior_q_window"][i]),
        ("expert_oracle_ceiling", lambda i: data["expert_q_window"][i]),
        ("mean_residual_diagnostic", lambda i: data["prior_q_window"][i] + residual_mean.reshape(1, 6)),
    )
    for index in range(len(data["path_names"])):
        name = str(data["path_names"][index])
        start = int(data["window_starts"][index])
        previous = prior_paths[name].get(start - 1)
        for mode, constructor in definitions:
            candidate = np.asarray(constructor(index), dtype=np.float32)
            metrics = evaluate_candidate(
                candidate, data["desired_path_window"][index], data["prior_q_window"][index],
                previous, execution_horizon, robot, joint_names, ee_link, lower, upper,
            )
            row = row_identity(BASELINE_CHECKPOINT, BASELINE_CHECKPOINT, BASELINE_EPOCH,
                               "reference", name, start, 0.0, 0, mode)
            row.update(metrics)
            if mode == "strong_prior" and not np.allclose(
                candidate,
                data["prior_q_window"][index],
                atol=1.0e-8,
                rtol=1.0e-7,
            ):
                raise AssertionError(
                    f"alpha=0 strong-prior candidate differs from prior for "
                    f"{name} window_start={start}"
                )
            row.update(diagnostic_metrics(
                None if mode == "strong_prior" else
                (data["residual_q_window"][index] if mode == "expert_oracle_ceiling" else np.broadcast_to(residual_mean, candidate.shape)),
                candidate, data["residual_q_window"][index], data["expert_q_window"][index],
            ))
            row.update({
                "sample_index": -1,
                "sample_seed": -1,
                "safety_gate_pass": int(mode != "mean_residual_diagnostic" or not rejection_reasons(metrics, metrics["prefix_mean_cartesian_error"], 0.20)),
                "fallback_to_prior": 0,
                "rejection_reasons": "",
                "prior_prefix_mean_cartesian_error": math.nan,
                "cartesian_improvement": math.nan,
            })
            if mode == "strong_prior":
                prior_by_index[index] = row
            rows.append(row)
        prior_mean = float(prior_by_index[index]["prefix_mean_cartesian_error"])
        for row in rows[-3:]:
            row["prior_prefix_mean_cartesian_error"] = prior_mean
            row["cartesian_improvement"] = prior_mean - float(row["prefix_mean_cartesian_error"])
            reasons = rejection_reasons(row, prior_mean, 0.20)
            row["safety_gate_pass"] = int(not reasons)
            row["rejection_reasons"] = "|".join(reasons)
    return rows, prior_by_index


def candidate_sort_key(item: Tuple[int, Dict[str, Any]]) -> Tuple[float, float, float, float, float]:
    _, metrics = item
    return (
        float(metrics.get("prefix_mean_cartesian_error", math.inf)),
        float(metrics.get("prefix_max_cartesian_error", math.inf)),
        float(metrics.get("max_boundary_step", math.inf)),
        float(metrics.get("prefix_acceleration_cost", math.inf)),
        float(metrics.get("prefix_jerk_cost", math.inf)),
    )


def make_selected_row(
    identity: Mapping[str, Any], selected_index: int, selected: Mapping[str, Any],
    sampled_residual: Optional[np.ndarray], candidate_q: np.ndarray,
    expert_residual: np.ndarray, expert_q: np.ndarray, prior_mean: float,
    seed: int, gate_pass: bool, fallback: bool, reasons: Sequence[str],
) -> Dict[str, Any]:
    row = dict(identity)
    row.update(selected)
    row.update(diagnostic_metrics(sampled_residual, candidate_q, expert_residual, expert_q))
    row.update({
        "sample_index": int(selected_index),
        "sample_seed": int(seed),
        "safety_gate_pass": int(gate_pass),
        "fallback_to_prior": int(fallback),
        "rejection_reasons": "|".join(reasons),
        "prior_prefix_mean_cartesian_error": float(prior_mean),
        "cartesian_improvement": float(
            prior_mean - float(selected.get("prefix_mean_cartesian_error", math.inf))
        ),
    })
    return row


def evaluate_state(
    *, model: torch.nn.Module, schedule: Any, state_label: str, checkpoint_file: str,
    epoch: int, data: Mapping[str, np.ndarray], prior_paths: Mapping[str, Mapping[int, np.ndarray]],
    prior_by_index: Mapping[int, Mapping[str, Any]], residual_mean: np.ndarray,
    residual_std: np.ndarray, args: argparse.Namespace, device: torch.device,
    robot: Any, joint_names: Sequence[str], ee_link: str, lower: np.ndarray, upper: np.ndarray,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], np.ndarray, np.ndarray]:
    count = len(data["path_names"])
    sampled = np.empty((count, args.num_samples, EXPECTED_HORIZON, EXPECTED_TARGET_DIM), dtype=np.float32)
    sample_seeds = np.empty((count, args.num_samples), dtype=np.int64)
    expanded_conditions: List[np.ndarray] = []
    expanded_seeds: List[int] = []
    locations: List[Tuple[int, int]] = []
    for index, (name, start) in enumerate(zip(data["path_names"], data["window_starts"])):
        for sample_index in range(args.num_samples):
            seed = stable_seed(args.seed, state_label, str(name), int(start), sample_index)
            expanded_conditions.append(data["condition_features_norm"][index])
            expanded_seeds.append(seed)
            locations.append((index, sample_index))
            sample_seeds[index, sample_index] = seed
    step = args.batch_size * args.num_samples
    for begin in range(0, len(locations), step):
        end = min(begin + step, len(locations))
        normalized = sample_batch(
            model, np.stack(expanded_conditions[begin:end]), expanded_seeds[begin:end],
            schedule, args.sampling_steps, args.sampler, args.ddim_eta, device,
        )
        physical = normalized * residual_std.reshape(1, 1, 6) + residual_mean.reshape(1, 1, 6)
        for local, (window_index, sample_index) in enumerate(locations[begin:end]):
            sampled[window_index, sample_index] = physical[local]

    selected_rows: List[Dict[str, Any]] = []
    rejection_rows: List[Dict[str, Any]] = []
    for index in range(count):
        name = str(data["path_names"][index])
        start = int(data["window_starts"][index])
        prior = data["prior_q_window"][index]
        desired = data["desired_path_window"][index]
        expert_q = data["expert_q_window"][index]
        expert_residual = data["residual_q_window"][index]
        previous = prior_paths[name].get(start - 1)
        prior_metrics = prior_by_index[index]
        prior_mean = float(prior_metrics["prefix_mean_cartesian_error"])
        state_prior = copy.deepcopy(dict(prior_metrics))
        state_prior.update(row_identity(
            checkpoint_file, state_label, epoch, state_label.rsplit("/", 1)[-1],
            name, start, 0.0, 0, "strong_prior",
        ))
        selected_rows.append(state_prior)
        for alpha in args.alphas:
            for taper_length in args.taper_lengths:
                taper = cosine_taper(taper_length)
                candidates: List[np.ndarray] = []
                evaluated: List[Dict[str, Any]] = []
                reason_lists: List[List[str]] = []
                for sample_index in range(args.num_samples):
                    candidate = prior + taper * (float(alpha) * sampled[index, sample_index])
                    metrics = evaluate_candidate(
                        candidate, desired, prior, previous, args.execution_horizon,
                        robot, joint_names, ee_link, lower, upper,
                    )
                    reasons = rejection_reasons(metrics, prior_mean, args.max_joint_step_gate)
                    candidates.append(candidate)
                    evaluated.append(metrics)
                    reason_lists.append(reasons)
                    rejection_rows.append({
                        "checkpoint": state_label,
                        "checkpoint_file": checkpoint_file,
                        "checkpoint_epoch": epoch,
                        "weight_state": state_label.rsplit("/", 1)[-1],
                        "path_name": name,
                        "window_start": start,
                        "alpha": float(alpha),
                        "taper_length": taper_length,
                        "sample_index": sample_index,
                        "passed": int(not reasons),
                        "rejection_reasons": "|".join(reasons),
                    })
                base = {
                    "checkpoint_file": checkpoint_file,
                    "checkpoint": state_label,
                    "checkpoint_epoch": epoch,
                    "weight_state": state_label.rsplit("/", 1)[-1],
                    "path_name": name,
                    "window_start": start,
                    "alpha": float(alpha),
                    "taper_length": taper_length,
                }
                single_identity = {**base, "selection_mode": "single_sample"}
                selected_rows.append(make_selected_row(
                    single_identity, 0, evaluated[0], sampled[index, 0], candidates[0],
                    expert_residual, expert_q, prior_mean, int(sample_seeds[index, 0]),
                    not reason_lists[0], False, reason_lists[0],
                ))
                safe = [(sample_index, evaluated[sample_index]) for sample_index in range(args.num_samples) if not reason_lists[sample_index]]
                safe_identity = {**base, "selection_mode": "best_of_k_safe_cartesian"}
                if safe:
                    selected_index, selected_metrics = min(safe, key=candidate_sort_key)
                    selected_rows.append(make_selected_row(
                        safe_identity, selected_index, selected_metrics, sampled[index, selected_index],
                        candidates[selected_index], expert_residual, expert_q, prior_mean,
                        int(sample_seeds[index, selected_index]), True, False, (),
                    ))
                else:
                    fallback_metrics = {key: copy.deepcopy(value) for key, value in prior_metrics.items() if key not in row_identity("", "", 0, "", "", 0, 0, 0, "")}
                    combined = sorted(set(reason for reasons in reason_lists for reason in reasons))
                    selected_rows.append(make_selected_row(
                        safe_identity, -1, fallback_metrics, None, prior, expert_residual, expert_q,
                        prior_mean, -1, False, True, combined,
                    ))
                oracle_index, oracle_metrics = min(enumerate(evaluated), key=candidate_sort_key)
                oracle_identity = {**base, "selection_mode": "oracle_best_of_k_cartesian"}
                selected_rows.append(make_selected_row(
                    oracle_identity, oracle_index, oracle_metrics, sampled[index, oracle_index],
                    candidates[oracle_index], expert_residual, expert_q, prior_mean,
                    int(sample_seeds[index, oracle_index]), not reason_lists[oracle_index], False,
                    reason_lists[oracle_index],
                ))
    return selected_rows, rejection_rows, sampled, sample_seeds


GROUP_COLUMNS = ["checkpoint", "checkpoint_epoch", "weight_state", "alpha", "taper_length", "selection_mode"]


def aggregate_group(
    group: pd.DataFrame,
    prefix_errors: Mapping[int, np.ndarray],
    full_errors: Mapping[int, np.ndarray],
) -> Dict[str, Any]:
    # Baseline rows can be copied under checkpoint labels. Safety accounting is
    # always over unique physical windows, never duplicated bookkeeping rows.
    group = group.drop_duplicates(subset=["path_name", "window_start"], keep="first")
    pooled = np.concatenate([prefix_errors[int(index)] for index in group.index])
    pooled_full = np.concatenate([full_errors[int(index)] for index in group.index])
    improvements = group["cartesian_improvement"].to_numpy(dtype=float)
    prior = group["prior_prefix_mean_cartesian_error"].to_numpy(dtype=float)
    boundary = group["max_boundary_step"].to_numpy(dtype=float)
    return {
        "number_of_paths": int(group["path_name"].nunique()),
        "number_of_windows": int(
            group[["path_name", "window_start"]].drop_duplicates().shape[0]
        ),
        "mean_cartesian_error": float(group["prefix_mean_cartesian_error"].mean()),
        "median_cartesian_error": float(np.median(pooled)),
        "rms_cartesian_error": float(np.sqrt(np.mean(np.square(pooled)))),
        "p95_cartesian_error": float(np.percentile(pooled, 95.0)),
        "maximum_cartesian_error": float(np.max(pooled)),
        "percentage_windows_improving_over_prior": float(100.0 * np.mean(improvements > EPS)),
        "percentage_windows_worsening_over_prior": float(100.0 * np.mean(improvements < -EPS)),
        "mean_absolute_improvement": float(np.mean(improvements)),
        "mean_percentage_improvement": float(100.0 * np.mean(improvements / np.maximum(prior, EPS))),
        "safety_gate_pass_rate": float(group["safety_gate_pass"].mean()),
        "fallback_to_prior_rate": float(group["fallback_to_prior"].mean()),
        "joint_limit_violation_rate": float(np.mean(group["prefix_joint_limit_violation_count"] > 0)),
        "hard_joint_limit_violation_rate": float(
            np.mean(group["prefix_joint_limit_violation_count"] > 0)
        ),
        "hard_limit_safety_pass_rate": float(
            np.mean(group["prefix_joint_limit_violation_count"] == 0)
        ),
        "safety_margin_violation_rate": float(
            np.mean(group["safety_margin_violation_count"] > 0)
        ),
        "step_gate_violation_rate": float(np.mean(group["gated_max_absolute_joint_step"] > 0.20)),
        "mean_boundary_step": float(np.nanmean(boundary)),
        "maximum_boundary_step": float(np.nanmax(boundary)),
        "velocity_cost": float(group["prefix_velocity_cost"].mean()),
        "acceleration_cost": float(group["prefix_acceleration_cost"].mean()),
        "jerk_cost": float(group["prefix_jerk_cost"].mean()),
        "mean_transition_acceleration_discontinuity": float(group["transition_acceleration_discontinuity_l2"].mean()),
        "mean_sampled_residual_rmse": float(group["sampled_residual_rmse"].mean()),
        "mean_candidate_joint_rmse": float(group["candidate_joint_rmse"].mean()),
        "full_mean_cartesian_error": float(group["full_mean_cartesian_error"].mean()),
        "full_median_cartesian_error": float(np.median(pooled_full)),
        "full_rms_cartesian_error": float(np.sqrt(np.mean(np.square(pooled_full)))),
        "full_p95_cartesian_error": float(np.percentile(pooled_full, 95.0)),
        "full_maximum_cartesian_error": float(np.max(pooled_full)),
        "full_velocity_cost": float(group["full_velocity_cost"].mean()),
        "full_acceleration_cost": float(group["full_acceleration_cost"].mean()),
        "full_jerk_cost": float(group["full_jerk_cost"].mean()),
    }


def aggregate_results(rows: List[Dict[str, Any]]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prefix_error_arrays = {
        index: np.asarray(row.pop("_prefix_errors"), dtype=np.float64)
        for index, row in enumerate(rows)
    }
    full_error_arrays = {
        index: np.asarray(row.pop("_full_errors"), dtype=np.float64)
        for index, row in enumerate(rows)
    }
    frame = pd.DataFrame(rows)
    aggregate_rows: List[Dict[str, Any]] = []
    for keys, group in frame.groupby(GROUP_COLUMNS, dropna=False, sort=False):
        record = dict(zip(GROUP_COLUMNS, keys))
        record.update(aggregate_group(group, prefix_error_arrays, full_error_arrays))
        aggregate_rows.append(record)
    path_rows: List[Dict[str, Any]] = []
    for keys, group in frame.groupby(GROUP_COLUMNS + ["path_name"], dropna=False, sort=False):
        record = dict(zip(GROUP_COLUMNS + ["path_name"], keys))
        record.update(aggregate_group(group, prefix_error_arrays, full_error_arrays))
        path_rows.append(record)
    return frame, pd.DataFrame(path_rows), pd.DataFrame(aggregate_rows)


def rejection_summary(candidate_rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(candidate_rows)
    groups = ["checkpoint", "checkpoint_file", "checkpoint_epoch", "weight_state", "alpha", "taper_length"]
    records: List[Dict[str, Any]] = []
    for keys, group in frame.groupby(groups, sort=False):
        counts: Counter[str] = Counter()
        for value in group["rejection_reasons"]:
            counts.update(reason for reason in str(value).split("|") if reason)
        base = dict(zip(groups, keys))
        base.update({"candidate_count": len(group), "passed_count": int(group["passed"].sum()),
                     "safety_gate_pass_rate": float(group["passed"].mean())})
        for reason in ("nonfinite_values", "joint_limit_violation", "maximum_joint_step_gate",
                       "nonfinite_boundary", "catastrophic_cartesian_degradation"):
            base[f"rejected_{reason}"] = counts[reason]
        records.append(base)
    return pd.DataFrame(records)


def baseline_aggregate(aggregate: pd.DataFrame, mode: str) -> pd.Series:
    selected = aggregate[(aggregate["checkpoint"] == BASELINE_CHECKPOINT) & (aggregate["selection_mode"] == mode)]
    if len(selected) != 1:
        raise RuntimeError(f"Expected one aggregate row for baseline {mode}")
    return selected.iloc[0]


def recommend(path_frame: pd.DataFrame, aggregate: pd.DataFrame, args: argparse.Namespace) -> Tuple[Dict[str, Any], pd.DataFrame]:
    prior = baseline_aggregate(aggregate, "strong_prior")
    configs = aggregate[
        (aggregate["checkpoint"] != BASELINE_CHECKPOINT)
        & (aggregate["alpha"] > 0.0)
        & (aggregate["selection_mode"] == "best_of_k_safe_cartesian")
    ].copy()
    comparison_rows: List[Dict[str, Any]] = []
    for _, row in configs.iterrows():
        mask = np.ones(len(path_frame), dtype=bool)
        for key in GROUP_COLUMNS:
            mask &= path_frame[key].to_numpy() == row[key]
        paths = path_frame[mask]
        path_mean = float(paths["mean_cartesian_error"].mean())
        prior_paths = path_frame[(path_frame["checkpoint"] == BASELINE_CHECKPOINT) & (path_frame["selection_mode"] == "strong_prior")]
        prior_path_mean = float(prior_paths["mean_cartesian_error"].mean())
        improvement = prior_path_mean - path_mean
        acceleration_ok = float(row["acceleration_cost"]) <= float(prior["acceleration_cost"]) * (1.0 + SMOOTHNESS_RELATIVE_TOLERANCE) + EPS
        jerk_ok = float(row["jerk_cost"]) <= float(prior["jerk_cost"]) * (1.0 + SMOOTHNESS_RELATIVE_TOLERANCE) + EPS
        boundary_ok = float(row["mean_boundary_step"]) <= float(prior["mean_boundary_step"]) * (1.0 + SMOOTHNESS_RELATIVE_TOLERANCE) + BOUNDARY_ABSOLUTE_TOLERANCE_RAD
        safety_ok = (
            float(row["joint_limit_violation_rate"]) <= float(prior["joint_limit_violation_rate"]) + EPS
            and float(row["step_gate_violation_rate"]) <= float(prior["step_gate_violation_rate"]) + EPS
        )
        eligible = safety_ok and acceleration_ok and jerk_ok and boundary_ok and improvement > 0.0
        comparison_rows.append({
            **{key: row[key] for key in GROUP_COLUMNS},
            "path_level_mean_cartesian_error": path_mean,
            "prior_path_level_mean_cartesian_error": prior_path_mean,
            "absolute_improvement_m": improvement,
            "absolute_improvement_mm": 1000.0 * improvement,
            "percentage_improvement": 100.0 * improvement / max(prior_path_mean, EPS),
            "safety_eligible": int(safety_ok),
            "acceleration_eligible": int(acceleration_ok),
            "jerk_eligible": int(jerk_ok),
            "boundary_eligible": int(boundary_ok),
            "practical_eligible": int(eligible),
            "safety_pass_rate": row["safety_gate_pass_rate"],
            "fallback_rate": row["fallback_to_prior_rate"],
        })
    comparison = pd.DataFrame(comparison_rows)
    eligible = comparison[comparison["practical_eligible"] == 1]
    if eligible.empty:
        return {
            "recommendation": "retain_strong_prior_no_diffusion_gain",
            "reason": "No configuration met all path-level improvement, safety, smoothness, and boundary criteria.",
            "meaningful_absolute_threshold_m": MEANINGFUL_ABSOLUTE_IMPROVEMENT_M,
            "meaningful_relative_threshold_percent": 100.0 * MEANINGFUL_RELATIVE_IMPROVEMENT,
        }, comparison
    best = eligible.sort_values(
        ["path_level_mean_cartesian_error", "fallback_rate", "safety_pass_rate"],
        ascending=[True, True, False],
    ).iloc[0]
    meaningful = (
        float(best["absolute_improvement_m"]) >= MEANINGFUL_ABSOLUTE_IMPROVEMENT_M
        and float(best["percentage_improvement"]) >= 100.0 * MEANINGFUL_RELATIVE_IMPROVEMENT
    )
    recommendation = "proceed_to_small_recursive_validation" if meaningful else "retain_strong_prior_no_diffusion_gain"
    return {
        "recommendation": recommendation,
        "checkpoint": best["checkpoint"],
        "checkpoint_epoch": int(best["checkpoint_epoch"]),
        "weight_state": best["weight_state"],
        "alpha": float(best["alpha"]),
        "taper_length": int(best["taper_length"]),
        "selection_mode": best["selection_mode"],
        "sampling_method": args.sampler,
        "sampling_steps": args.sampling_steps,
        "ddim_eta": args.ddim_eta,
        "number_of_samples": args.num_samples,
        "prior_path_level_mean_cartesian_error_m": float(best["prior_path_level_mean_cartesian_error"]),
        "selected_path_level_mean_cartesian_error_m": float(best["path_level_mean_cartesian_error"]),
        "absolute_cartesian_improvement_m": float(best["absolute_improvement_m"]),
        "absolute_cartesian_improvement_mm": float(best["absolute_improvement_mm"]),
        "percentage_cartesian_improvement": float(best["percentage_improvement"]),
        "safety_pass_rate": float(best["safety_pass_rate"]),
        "fallback_to_prior_rate": float(best["fallback_rate"]),
        "meaningful_effect": bool(meaningful),
        "meaningful_absolute_threshold_m": MEANINGFUL_ABSOLUTE_IMPROVEMENT_M,
        "meaningful_relative_threshold_percent": 100.0 * MEANINGFUL_RELATIVE_IMPROVEMENT,
        "smoothness_relative_tolerance": SMOOTHNESS_RELATIVE_TOLERANCE,
        "boundary_absolute_tolerance_rad": BOUNDARY_ABSOLUTE_TOLERANCE_RAD,
    }, comparison


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


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
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def atomic_json(value: Mapping[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(value), handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def save_figure(fig: Any, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_plots(aggregate: pd.DataFrame, frame: pd.DataFrame, output_dir: Path) -> None:
    practical = aggregate[(aggregate["checkpoint"] != BASELINE_CHECKPOINT) & (aggregate["selection_mode"] == "best_of_k_safe_cartesian")]
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = practical["checkpoint"].astype(str) + "/a=" + practical["alpha"].astype(str) + "/t=" + practical["taper_length"].astype(str)
    ranked = practical.assign(_label=labels).sort_values("mean_cartesian_error").head(20)
    ax.barh(ranked["_label"], ranked["mean_cartesian_error"])
    ax.set_xlabel("Execution-prefix mean Cartesian error (m)")
    ax.set_title("Checkpoint validation comparison (best 20 configurations)")
    save_figure(fig, output_dir / "checkpoint_validation_comparison.png")

    for filename, x, xlabel in (
        ("alpha_cartesian_error.png", "alpha", "Residual scale alpha"),
        ("taper_cartesian_error.png", "taper_length", "Boundary taper length"),
    ):
        fig, ax = plt.subplots(figsize=(8, 5))
        for label, group in practical.groupby("checkpoint"):
            curve = group.groupby(x, as_index=False)["mean_cartesian_error"].mean()
            ax.plot(curve[x], curve["mean_cartesian_error"], marker="o", label=label)
        ax.set_xlabel(xlabel); ax.set_ylabel("Mean Cartesian error (m)"); ax.legend(fontsize=7)
        save_figure(fig, output_dir / filename)

    for filename, metric, ylabel in (
        ("safety_pass_rate.png", "safety_gate_pass_rate", "Safety-gate pass rate"),
        ("improvement_fraction.png", "percentage_windows_improving_over_prior", "Windows improving over prior (%)"),
    ):
        fig, ax = plt.subplots(figsize=(8, 5))
        points = practical.groupby("alpha", as_index=False)[metric].mean()
        ax.plot(points["alpha"], points[metric], marker="o")
        ax.set_xlabel("Residual scale alpha"); ax.set_ylabel(ylabel)
        save_figure(fig, output_dir / filename)

    selected = frame[(frame["checkpoint"] != BASELINE_CHECKPOINT) & (frame["selection_mode"] == "best_of_k_safe_cartesian")]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(selected["max_boundary_step"], selected["cartesian_improvement"], s=8, alpha=0.25)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xlabel("Maximum entry/exit boundary step (rad)"); ax.set_ylabel("Cartesian improvement (m)")
    save_figure(fig, output_dir / "cartesian_improvement_vs_boundary_step.png")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(selected["sampled_residual_rmse"], selected["cartesian_improvement"], s=8, alpha=0.25)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xlabel("Sampled residual RMSE vs expert residual (rad)"); ax.set_ylabel("Cartesian improvement (m)")
    save_figure(fig, output_dir / "residual_rmse_vs_cartesian_improvement.png")


def print_summary(aggregate: pd.DataFrame, comparison: pd.DataFrame, recommendation: Mapping[str, Any],
                  checkpoint_reports: Sequence[Mapping[str, Any]], paths: int, windows: int) -> None:
    print("\nValidation checkpoint states:")
    for report in checkpoint_reports:
        print(f"  {report['checkpoint_file']}: epoch={report.get('epoch')} checkpoint_states={report.get('checkpoint_available_states')} requested_available={report.get('available_states')} missing={report.get('missing_states')} status={report.get('status')}")
    print(f"Validation cohort: {paths} paths, {windows} windows")
    for mode, title in (("strong_prior", "Strong prior"), ("expert_oracle_ceiling", "Expert/oracle"), ("mean_residual_diagnostic", "Mean residual")):
        row = baseline_aggregate(aggregate, mode)
        print(f"{title}: prefix mean={row['mean_cartesian_error']:.6f} m, RMS={row['rms_cartesian_error']:.6f} m, max={row['maximum_cartesian_error']:.6f} m")
    if not comparison.empty:
        print("Best practical configuration per checkpoint:")
        for checkpoint, group in comparison.groupby("checkpoint"):
            best = group.sort_values("path_level_mean_cartesian_error").iloc[0]
            print(f"  {checkpoint}: alpha={best['alpha']:g}, taper={int(best['taper_length'])}, path mean={best['path_level_mean_cartesian_error']:.6f} m, safety={best['safety_pass_rate']:.3f}, fallback={best['fallback_rate']:.3f}")
    print("Best overall practical configuration:")
    print(json.dumps(json_safe(recommendation), indent=2, sort_keys=True))
    print(f"Final recommendation: {recommendation['recommendation']}")
    print("Official test data was not loaded; recursive rollout was not performed.")


def main() -> int:
    args = parse_args()
    validate_cli(args)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    required_outputs = (
        "teacher_forced_window_results.csv", "teacher_forced_path_results.csv",
        "teacher_forced_aggregate.csv", "teacher_forced_rejection_summary.csv",
        "teacher_forced_candidate_gate_results.csv",
        "checkpoint_comparison.csv", "recommended_validation_configuration.json",
        "evaluation_configuration.json", "worst_validation_windows.csv",
        "best_validation_improvements.csv",
        "hard_limit_consistency_audit.json",
    )
    if args.output_dir.exists() and not args.overwrite and any((args.output_dir / name).exists() for name in required_outputs):
        raise FileExistsError("Evaluation outputs already exist; pass --overwrite to replace them")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data_all = load_validation(args.val_npz)
    stats = load_stats(args.normalization_stats)
    dataset_config = load_json(args.dataset_configuration)
    validate_dataset_configuration(dataset_config)
    data = select_paths(data_all, args.path_names, args.max_paths)
    prior_paths = reconstruct_prior_paths(data)
    robot, joint_names, ee_link = load_fk_context(None, None)
    if tuple(joint_names) != tuple(f"joint{i}" for i in range(1, 7)):
        raise ValueError(f"FK active joints are not joint1..joint6: {joint_names}")
    if str(ee_link) != "xMateCR7_link6":
        raise ValueError(f"FK end-effector frame is not xMateCR7_link6: {ee_link}")
    lower, upper = authoritative_joint_limits(robot, joint_names)
    limit_audit = hard_limit_consistency_audit(
        data,
        joint_names,
        lower,
        upper,
        args.output_dir / "hard_limit_consistency_audit.json",
    )
    baseline, prior_by_index = baseline_rows(
        data, prior_paths, stats["residual_mean"], args.execution_horizon,
        robot, joint_names, ee_link, lower, upper,
    )
    all_rows = list(baseline)
    all_rejections: List[Dict[str, Any]] = []
    checkpoint_reports: List[Dict[str, Any]] = []
    residual_archives: Dict[str, np.ndarray] = {}

    for checkpoint_index, checkpoint_path in enumerate(args.checkpoints):
        custom = checkpoint_path.stem if args.checkpoint_labels is None else args.checkpoint_labels[checkpoint_index]
        report: Dict[str, Any] = {"checkpoint_file": str(checkpoint_path), "requested_label": custom}
        try:
            checkpoint = torch_load(checkpoint_path, device)
            schedule_config = validate_checkpoint(
                checkpoint, checkpoint_path, args.val_npz, args.normalization_stats, stats, dataset_config,
            )
            if args.sampling_steps > schedule_config["steps"]:
                raise ValueError("Requested sampling steps exceed checkpoint diffusion steps")
            epoch = int(checkpoint.get("epoch", -1))
            discovered_states, discovered_missing = checkpoint_states(checkpoint, True, True)
            states, missing_states = checkpoint_states(
                checkpoint, args.evaluate_raw_weights, args.evaluate_ema_weights,
            )
            report.update({"epoch": epoch, "available_states": [name for name, _ in states],
                           "missing_states": missing_states,
                           "checkpoint_available_states": [name for name, _ in discovered_states],
                           "checkpoint_missing_states": discovered_missing,
                           "schedule": schedule_config})
            if not states:
                report["status"] = "no_requested_weight_state_available"
                checkpoint_reports.append(report)
                print(f"[checkpoint] {checkpoint_path}: no requested state is available")
                continue
            evaluated_states: List[str] = []
            state_errors: Dict[str, str] = {}
            for state_name, state_dict in states:
                try:
                    label = f"{custom}|{checkpoint_path.name}|epoch={epoch}/{state_name}"
                    model, model_config = instantiate_v5_model(32, 38, 6, schedule_config["steps"])
                    model.load_state_dict(state_dict, strict=True)
                    model.to(device).eval()
                    schedule = build_schedule(schedule_config["steps"], device)
                    selected, rejections, samples, seeds = evaluate_state(
                        model=model, schedule=schedule, state_label=label,
                        checkpoint_file=str(checkpoint_path), epoch=epoch, data=data,
                        prior_paths=prior_paths, prior_by_index=prior_by_index,
                        residual_mean=stats["residual_mean"], residual_std=stats["residual_std"],
                        args=args, device=device, robot=robot, joint_names=joint_names,
                        ee_link=ee_link, lower=lower, upper=upper,
                    )
                    all_rows.extend(selected); all_rejections.extend(rejections)
                    archive_key = f"checkpoint_{checkpoint_index}_{state_name}"
                    residual_archives[f"{archive_key}_physical_residual"] = samples
                    residual_archives[f"{archive_key}_seeds"] = seeds
                    residual_archives[f"{archive_key}_label"] = np.asarray(label)
                    report.setdefault("model", {})[state_name] = model_config
                    evaluated_states.append(state_name)
                    del model
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                except Exception as state_error:
                    state_errors[state_name] = f"{type(state_error).__name__}: {state_error}"
                    print(f"[checkpoint state] {checkpoint_path}/{state_name}: {state_errors[state_name]}")
            report["evaluated_states"] = evaluated_states
            report["state_errors"] = state_errors
            report["status"] = "evaluated" if evaluated_states else "all_requested_states_failed"
        except Exception as error:
            report.update({"status": "incompatible_or_failed", "error": f"{type(error).__name__}: {error}"})
            print(f"[checkpoint] {checkpoint_path}: {report['error']}")
        checkpoint_reports.append(report)

    if not any(report.get("status") == "evaluated" for report in checkpoint_reports):
        raise RuntimeError("No compatible requested checkpoint state was evaluated")
    window_frame, path_frame, aggregate = aggregate_results(all_rows)
    rejection = rejection_summary(all_rejections)
    recommendation, comparison = recommend(path_frame, aggregate, args)
    recommended_label = recommendation.get("checkpoint")
    recommended = window_frame[
        (window_frame["checkpoint"] == recommended_label)
        & (window_frame["alpha"] == recommendation.get("alpha"))
        & (window_frame["taper_length"] == recommendation.get("taper_length"))
        & (window_frame["selection_mode"] == recommendation.get("selection_mode"))
    ].copy()
    if recommended.empty:
        recommended = window_frame[
            (window_frame["checkpoint"] == BASELINE_CHECKPOINT)
            & (window_frame["selection_mode"] == "strong_prior")
        ].copy()
    worst = recommended.sort_values("cartesian_improvement", ascending=True).head(50)
    best = recommended.sort_values("cartesian_improvement", ascending=False).head(50)

    atomic_csv(window_frame, args.output_dir / "teacher_forced_window_results.csv")
    atomic_csv(path_frame, args.output_dir / "teacher_forced_path_results.csv")
    atomic_csv(aggregate, args.output_dir / "teacher_forced_aggregate.csv")
    atomic_csv(rejection, args.output_dir / "teacher_forced_rejection_summary.csv")
    atomic_csv(pd.DataFrame(all_rejections), args.output_dir / "teacher_forced_candidate_gate_results.csv")
    atomic_csv(comparison, args.output_dir / "checkpoint_comparison.csv")
    atomic_csv(worst, args.output_dir / "worst_validation_windows.csv")
    atomic_csv(best, args.output_dir / "best_validation_improvements.csv")
    np.savez_compressed(args.output_dir / "sampled_residuals_physical.npz", **residual_archives)
    atomic_json(recommendation, args.output_dir / "recommended_validation_configuration.json")
    configuration = {
        "arguments": vars(args),
        "resolved_device": str(device),
        "validation_paths": sorted(set(data["path_names"].tolist())),
        "validation_path_count": len(set(data["path_names"].tolist())),
        "validation_window_count": len(data["path_names"]),
        "checkpoint_reports": checkpoint_reports,
        "scheduler_source": "checkpoint diffusion_hyperparameters",
        "prediction_target": "epsilon",
        "taper": "cosine ramp from zero to one across taper_length timesteps, then one",
        "candidate_construction": "fixed_prior + taper * alpha * sampled_physical_residual",
        "teacher_forced": True,
        "recursive_propagation": False,
        "official_test_data_loaded": False,
        "expert_used_for_sampling_conditioning_or_practical_ranking": False,
        "joint_step_gate_rad": args.max_joint_step_gate,
        "joint_limit_checker": limit_audit["authoritative_source"],
        "joint_limit_joint_names": limit_audit["authoritative_joint_names"],
        "hard_joint_lower_limits": limit_audit["hard_lower_limits"],
        "hard_joint_upper_limits": limit_audit["hard_upper_limits"],
        "hard_joint_limit_tolerance_rad": limit_audit["tolerance"],
        "joint_limit_safety_margin_rad": limit_audit["safety_margin_rad"],
        "joint_limit_safety_margin_is_rejection_gate": False,
        "meaningful_absolute_improvement_m": MEANINGFUL_ABSOLUTE_IMPROVEMENT_M,
        "meaningful_relative_improvement": MEANINGFUL_RELATIVE_IMPROVEMENT,
    }
    atomic_json(configuration, args.output_dir / "evaluation_configuration.json")
    save_plots(aggregate, window_frame, args.output_dir)
    print_summary(
        aggregate, comparison, recommendation, checkpoint_reports,
        len(set(data["path_names"].tolist())), len(data["path_names"]),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
