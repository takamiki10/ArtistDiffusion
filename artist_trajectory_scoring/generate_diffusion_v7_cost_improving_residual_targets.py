#!/usr/bin/env python3
"""Generate v7 cost-improving residual targets around the frozen strong prior.

This program is deliberately training-only and expert-free.  Targets are
candidate_q - prior_q for candidates selected solely by desired Cartesian path,
robot kinematics, hard safety checks, and motion-quality costs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.interpolate import CubicSpline, PchipInterpolator
from scipy.ndimage import gaussian_filter1d

from generate_ik_seed_path import (
    DEFAULT_EE_LINK,
    DEFAULT_JOINT_LIMIT_SAFETY_MARGIN_RAD,
    DEFAULT_JOINT_NAMES,
    DEFAULT_URDF_PATH,
    HARD_JOINT_LIMIT_TOLERANCE_RAD,
    check_joint_limits,
    get_joint_bounds,
    load_robot,
)


JOINT_DIM = 6
XYZ_DIM = 3
TRAJECTORY_LENGTH = 100
WINDOW_STRIDE = 4
EXPECTED_TRAIN_PATHS = 372
EXPECTED_WINDOW_STARTS = tuple(range(0, 69, WINDOW_STRIDE))
DEFAULT_METHODS = (
    "jacobian_dls",
    "sequential_ik",
    "spline_cem",
    "smooth_perturbation",
)
FORBIDDEN_INPUT_TOKENS = (
    "val_windows.npz",
    "test_inference_windows.npz",
    "test_prior.npz",
    "diffusion_test_v2.npz",
    "official_test",
)
WINDOW_KEYS = (
    "path_names",
    "window_starts",
    "prior_q_window",
    "desired_path_window",
    "prior_ee_window",
)
TARGET_ARRAY_KEYS = (
    "path_names",
    "path_indices",
    "window_starts",
    "target_indices_within_window",
    "candidate_methods",
    "target_types",
    "prior_q_window",
    "desired_path_window",
    "prior_ee_window",
    "candidate_q_window",
    "residual_q_window",
    "execution_horizon",
    "improves_prior",
    "is_zero_residual",
    "prior_prefix_cartesian_mean_error_m",
    "candidate_prefix_cartesian_mean_error_m",
    "absolute_cartesian_improvement_m",
    "relative_cartesian_improvement",
    "prior_prefix_cartesian_p95_error_m",
    "candidate_prefix_cartesian_p95_error_m",
    "prior_prefix_cartesian_max_error_m",
    "candidate_prefix_cartesian_max_error_m",
    "prior_acceleration_cost",
    "candidate_acceleration_cost",
    "prior_jerk_cost",
    "candidate_jerk_cost",
    "prior_boundary_step_rad",
    "candidate_boundary_step_rad",
    "prior_singularity_penalty",
    "candidate_singularity_penalty",
    "maximum_absolute_joint_step_rad",
    "hard_joint_limit_violation_count",
    "minimum_joint_limit_margin_rad",
    "delta_score",
    "pareto_rank",
    "residual_rms_rad",
    "residual_max_abs_rad",
)


@dataclass(frozen=True)
class ScoreWeights:
    cart_mean: float = 4.0
    cart_p95: float = 2.0
    cart_max: float = 1.0
    acceleration: float = 0.5
    jerk: float = 0.25
    boundary_step: float = 1.0
    boundary_acceleration: float = 0.5
    singularity: float = 0.25


@dataclass(frozen=True)
class MetricFloors:
    cartesian_m: float = 1.0e-4
    derivative: float = 1.0e-8
    boundary_rad: float = 1.0e-4
    singularity: float = 1.0e-4


@dataclass
class Candidate:
    method: str
    subtype: str
    residual: np.ndarray
    deterministic_seed: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    runtime_seconds: float = 0.0


@dataclass
class WindowContext:
    path_name: str
    path_index: int
    window_start: int
    prior_q: np.ndarray
    desired: np.ndarray
    prior_ee: np.ndarray
    previous_q: Optional[np.ndarray]
    previous_previous_q: Optional[np.ndarray]
    tail_q: np.ndarray
    tail_next_q: Optional[np.ndarray]


@dataclass
class RobotContext:
    robot: Any
    joint_names: Tuple[str, ...]
    ee_link: str
    lower: np.ndarray
    upper: np.ndarray


def parse_float_list(value: str) -> Tuple[float, ...]:
    result = tuple(float(item.strip()) for item in value.split(",") if item.strip())
    if not result or any(not np.isfinite(item) for item in result):
        raise argparse.ArgumentTypeError("Expected a comma-separated finite float list")
    return result


def parse_int_list(value: str) -> Tuple[int, ...]:
    result = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("Expected a comma-separated positive integer list")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate expert-free v7 cost-improving residual targets."
    )
    parser.add_argument("--train_prior", type=Path, required=True)
    parser.add_argument("--train_windows", type=Path, required=True)
    parser.add_argument("--split_manifest", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--path_names", nargs="+", default=None)
    parser.add_argument("--num_paths", type=int, default=20)
    parser.add_argument(
        "--path_selection", choices=("stratified_prior_error",),
        default="stratified_prior_error",
    )
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--execution_horizon", type=int, default=8)
    parser.add_argument("--targets_per_window", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max_windows", type=int, default=None)
    parser.add_argument(
        "--candidate_methods", nargs="+", choices=DEFAULT_METHODS,
        default=list(DEFAULT_METHODS),
    )
    parser.add_argument("--save_all_candidates", action="store_true")
    parser.add_argument("--robot_urdf", type=Path, default=Path(DEFAULT_URDF_PATH))
    parser.add_argument("--ee_link", default=DEFAULT_EE_LINK)
    parser.add_argument("--minimum_residual_distance", type=float, default=0.005)
    parser.add_argument("--min_cartesian_improvement_m", type=float, default=1.0e-5)
    parser.add_argument("--min_cartesian_improvement_fraction", type=float, default=0.005)
    parser.add_argument("--smoothness_relative_tolerance", type=float, default=0.10)
    parser.add_argument("--boundary_absolute_tolerance", type=float, default=0.01)
    parser.add_argument("--max_joint_step_gate", type=float, default=0.20)
    parser.add_argument("--joint_limit_safety_margin", type=float, default=DEFAULT_JOINT_LIMIT_SAFETY_MARGIN_RAD)
    parser.add_argument("--dls_damping", type=parse_float_list, default=(1e-4, 1e-3, 1e-2, 1e-1))
    parser.add_argument("--dls_scales", type=parse_float_list, default=(0.25, 0.50, 1.00))
    parser.add_argument("--ik_damping", type=parse_float_list, default=(1e-3, 1e-2, 1e-1))
    parser.add_argument("--ik_iteration_limits", type=parse_int_list, default=(8, 16))
    parser.add_argument("--cem_control_points", type=int, default=6)
    parser.add_argument("--cem_candidates", type=int, default=128)
    parser.add_argument("--cem_elites", type=int, default=16)
    parser.add_argument("--cem_iterations", type=int, default=20)
    parser.add_argument("--cem_restarts", type=int, default=2)
    parser.add_argument("--cem_initial_std", type=float, default=0.03)
    parser.add_argument("--cem_max_residual", type=float, default=0.15)
    parser.add_argument("--smooth_amplitudes", type=parse_float_list, default=(0.005, 0.01, 0.025, 0.05))
    parser.add_argument("--w_cart_mean", type=float, default=4.0)
    parser.add_argument("--w_cart_p95", type=float, default=2.0)
    parser.add_argument("--w_cart_max", type=float, default=1.0)
    parser.add_argument("--w_acceleration", type=float, default=0.5)
    parser.add_argument("--w_jerk", type=float, default=0.25)
    parser.add_argument("--w_boundary_step", type=float, default=1.0)
    parser.add_argument("--w_boundary_acceleration", type=float, default=0.5)
    parser.add_argument("--w_singularity", type=float, default=0.25)
    parser.add_argument("--floor_cartesian_m", type=float, default=1e-4)
    parser.add_argument("--floor_derivative", type=float, default=1e-8)
    parser.add_argument("--floor_boundary_rad", type=float, default=1e-4)
    parser.add_argument("--floor_singularity", type=float, default=1e-4)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.overwrite and args.resume:
        raise ValueError("--overwrite and --resume are mutually exclusive")
    for path in (args.train_prior, args.train_windows, args.split_manifest):
        lowered = str(path).lower()
        if any(token in lowered for token in FORBIDDEN_INPUT_TOKENS):
            raise ValueError(f"Forbidden validation/test input: {path}")
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.train_windows.name != "train_windows.npz":
        raise ValueError("--train_windows must be the v6 training archive")
    if args.train_prior.name != "train_prior.npz":
        raise ValueError("--train_prior must be the frozen training prior archive")
    if args.horizon != 32 or args.execution_horizon != 8:
        raise ValueError("The v7 pilot requires horizon=32 and execution_horizon=8")
    if args.max_joint_step_gate != 0.20:
        raise ValueError("The validated maximum joint-step gate must remain 0.20 rad")
    if args.num_paths <= 0 or args.targets_per_window <= 0:
        raise ValueError("Path and target counts must be positive")
    if args.max_windows is not None and args.max_windows <= 0:
        raise ValueError("--max_windows must be positive")
    if not 0.0 < args.minimum_residual_distance:
        raise ValueError("--minimum_residual_distance must be positive")
    if args.cem_elites > args.cem_candidates:
        raise ValueError("--cem_elites cannot exceed --cem_candidates")
    if args.cem_control_points < 4 or args.cem_restarts < 1:
        raise ValueError("CEM requires at least four control points and one restart")


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(value)


def decode_strings(values: np.ndarray) -> np.ndarray:
    return np.asarray([
        item.decode("utf-8", errors="strict") if isinstance(item, bytes) else str(item)
        for item in np.asarray(values).reshape(-1)
    ], dtype=str)


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(value), handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def stable_seed(base: int, *parts: Any) -> int:
    digest = hashlib.sha256("|".join(str(item) for item in (base, *parts)).encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**32 - 1)


def load_window_data(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as archive:
        missing = sorted(set(WINDOW_KEYS) - set(archive.files))
        if missing:
            raise KeyError(f"Training windows are missing {missing}")
        # Expert and legacy residual keys are intentionally never indexed.
        data = {key: np.asarray(archive[key]) for key in WINDOW_KEYS}
    data["path_names"] = decode_strings(data["path_names"])
    data["window_starts"] = np.asarray(data["window_starts"], dtype=np.int64)
    count = len(data["path_names"])
    for key in ("prior_q_window",):
        if data[key].shape != (count, 32, 6):
            raise ValueError(f"{key} must have shape (N,32,6), got {data[key].shape}")
    for key in ("desired_path_window", "prior_ee_window"):
        if data[key].shape != (count, 32, 3):
            raise ValueError(f"{key} must have shape (N,32,3), got {data[key].shape}")
    for key in ("prior_q_window", "desired_path_window", "prior_ee_window"):
        if not np.all(np.isfinite(data[key])):
            raise ValueError(f"{key} contains nonfinite values")
    return data


def load_prior_path_names(path: Path) -> Optional[set[str]]:
    with np.load(path, allow_pickle=True) as archive:
        for key in ("path_names", "names"):
            if key in archive.files:
                return set(decode_strings(np.asarray(archive[key])).tolist())
    return None


def load_train_manifest(path: Path) -> Tuple[pd.DataFrame, set[str]]:
    frame = pd.read_csv(path)
    if "path_name" not in frame.columns or "assigned_split" not in frame.columns:
        raise ValueError("split_manifest.csv requires path_name and assigned_split")
    frame = frame.copy()
    frame["path_name"] = frame["path_name"].astype(str)
    frame["assigned_split"] = frame["assigned_split"].astype(str).str.lower()
    train = frame[frame["assigned_split"] == "train"].copy()
    if train["path_name"].duplicated().any():
        raise ValueError("Training split manifest contains duplicate path names")
    if len(train) != EXPECTED_TRAIN_PATHS:
        raise ValueError(f"Expected 372 training paths, found {len(train)}")
    return frame, set(train["path_name"])


def reconstruct_timelines(data: Mapping[str, np.ndarray]) -> Dict[str, Dict[str, np.ndarray]]:
    timelines: Dict[str, Dict[str, np.ndarray]] = {}
    for path_name in sorted(set(data["path_names"].tolist())):
        q = np.full((TRAJECTORY_LENGTH, 6), np.nan, dtype=np.float64)
        desired = np.full((TRAJECTORY_LENGTH, 3), np.nan, dtype=np.float64)
        ee = np.full((TRAJECTORY_LENGTH, 3), np.nan, dtype=np.float64)
        mask = data["path_names"] == path_name
        starts = data["window_starts"][mask]
        if len(starts) != len(EXPECTED_WINDOW_STARTS) or set(starts.tolist()) != set(EXPECTED_WINDOW_STARTS):
            raise ValueError(f"{path_name} does not have starts 0,4,...,68")
        indices = np.flatnonzero(mask)
        for index in indices:
            start = int(data["window_starts"][index])
            for destination, source_key in ((q, "prior_q_window"), (desired, "desired_path_window"), (ee, "prior_ee_window")):
                values = np.asarray(data[source_key][index], dtype=np.float64)
                existing = destination[start:start + 32]
                known = np.isfinite(existing[:, 0])
                if np.any(known) and not np.allclose(existing[known], values[known], atol=1e-7, rtol=1e-6):
                    raise ValueError(f"Overlapping {source_key} differs for {path_name} at {start}")
                destination[start:start + 32] = values
        if not all(np.all(np.isfinite(item)) for item in (q, desired, ee)):
            raise ValueError(f"Could not reconstruct complete trajectory for {path_name}")
        timelines[path_name] = {"prior_q": q, "desired": desired, "prior_ee": ee}
    return timelines


def select_paths(
    timelines: Mapping[str, Mapping[str, np.ndarray]], train_names: set[str],
    requested: Optional[Sequence[str]], num_paths: int, seed: int,
) -> Tuple[List[str], pd.DataFrame]:
    available = sorted(set(timelines) & train_names)
    if len(available) != EXPECTED_TRAIN_PATHS:
        raise ValueError(f"Expected 372 window paths assigned to train, found {len(available)}")
    rows = []
    for name in available:
        values = timelines[name]
        errors = np.linalg.norm(values["prior_ee"] - values["desired"], axis=1)
        rows.append({
            "path_name": name,
            "prior_mean_cartesian_error_m": float(np.mean(errors)),
            "prior_rms_cartesian_error_m": float(np.sqrt(np.mean(np.square(errors)))),
            "prior_max_cartesian_error_m": float(np.max(errors)),
        })
    frame = pd.DataFrame(rows).sort_values(
        ["prior_mean_cartesian_error_m", "path_name"], ignore_index=True
    )
    groups = np.array_split(np.arange(len(frame)), 4)
    labels = ("low", "lower_middle", "upper_middle", "high")
    frame["difficulty_group"] = ""
    for label, group in zip(labels, groups):
        frame.loc[group, "difficulty_group"] = label
    if requested is not None:
        names = list(dict.fromkeys(str(name) for name in requested))
        unknown = sorted(set(names) - set(available))
        if unknown:
            raise ValueError(f"Requested paths are not training paths: {unknown}")
        if len(names) != len(requested):
            raise ValueError("--path_names contains duplicates")
    else:
        if num_paths > len(available):
            raise ValueError("--num_paths exceeds available training paths")
        rng = np.random.default_rng(seed)
        counts = [num_paths // 4 + int(group < num_paths % 4) for group in range(4)]
        names = []
        for indices, count in zip(groups, counts):
            chosen = rng.choice(indices, size=count, replace=False)
            names.extend(frame.loc[sorted(chosen.tolist()), "path_name"].tolist())
    selected = frame[frame["path_name"].isin(names)].copy()
    selected["selection_order"] = selected["path_name"].map({name: i for i, name in enumerate(names)})
    selected = selected.sort_values("selection_order", ignore_index=True)
    return names, selected


def resolve_project_path(path: Path) -> Path:
    if path.is_file():
        return path
    script_dir = Path(__file__).resolve().parent
    for candidate in (script_dir / path, script_dir.parent / path):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(path)


def make_robot_context(args: argparse.Namespace) -> RobotContext:
    joint_names = tuple(DEFAULT_JOINT_NAMES)
    if joint_names != tuple(f"joint{i}" for i in range(1, 7)):
        raise ValueError(f"Active joints must be joint1..joint6, got {joint_names}")
    if args.ee_link != DEFAULT_EE_LINK or args.ee_link != "xMateCR7_link6":
        raise ValueError("The v7 pilot requires ee_link=xMateCR7_link6")
    urdf = resolve_project_path(args.robot_urdf)
    robot = load_robot(urdf)
    bounds = get_joint_bounds(robot, joint_names, -np.pi, np.pi)
    lower = np.asarray([item[0] for item in bounds], dtype=np.float64)
    upper = np.asarray([item[1] for item in bounds], dtype=np.float64)
    return RobotContext(robot, joint_names, args.ee_link, lower, upper)


def transform_xyz(transform: Any) -> np.ndarray:
    if hasattr(transform, "translation"):
        value = np.asarray(transform.translation, dtype=np.float64).reshape(-1)
        if value.size >= 3:
            return value[:3]
    if hasattr(transform, "matrix"):
        value = np.asarray(transform.matrix, dtype=np.float64)
        if value.shape == (4, 4):
            return value[:3, 3]
    value = np.asarray(transform, dtype=np.float64)
    if value.shape == (4, 4):
        return value[:3, 3]
    raise ValueError(f"Unsupported FK transform representation: {type(transform)}")


def fk_one(context: RobotContext, q: np.ndarray) -> np.ndarray:
    values = np.asarray(q, dtype=np.float64).reshape(6)
    cfg = {name: float(values[index]) for index, name in enumerate(context.joint_names)}
    context.robot.update_cfg(cfg)
    try:
        transform = context.robot.get_transform(frame_to=context.ee_link)
    except TypeError:
        transform = context.robot.get_transform(context.ee_link)
    return transform_xyz(transform)


def fk_trajectory(context: RobotContext, q: np.ndarray) -> np.ndarray:
    return np.stack([fk_one(context, row) for row in np.asarray(q)], axis=0)


def positional_jacobian(
    context: RobotContext, q: np.ndarray, epsilon: float = 1.0e-5
) -> np.ndarray:
    values = np.asarray(q, dtype=np.float64).reshape(6)
    jacobian = np.empty((3, 6), dtype=np.float64)
    for joint in range(6):
        plus = values.copy()
        minus = values.copy()
        plus[joint] = min(plus[joint] + epsilon, context.upper[joint])
        minus[joint] = max(minus[joint] - epsilon, context.lower[joint])
        denominator = plus[joint] - minus[joint]
        if denominator <= 0.0:
            jacobian[:, joint] = 0.0
        else:
            jacobian[:, joint] = (fk_one(context, plus) - fk_one(context, minus)) / denominator
    if not np.all(np.isfinite(jacobian)):
        raise FloatingPointError("Numerical positional Jacobian is nonfinite")
    return jacobian


def dls_update(jacobian: np.ndarray, error: np.ndarray, damping: float) -> np.ndarray:
    matrix = jacobian @ jacobian.T + float(damping) * np.eye(3)
    try:
        solved = np.linalg.solve(matrix, np.asarray(error, dtype=np.float64))
    except np.linalg.LinAlgError:
        solved = np.linalg.pinv(matrix) @ np.asarray(error, dtype=np.float64)
    update = jacobian.T @ solved
    return np.asarray(update, dtype=np.float64)


def manipulability_and_penalty(jacobian: np.ndarray) -> Tuple[float, float]:
    singular_values = np.linalg.svd(jacobian, compute_uv=False)
    singular_values = np.maximum(np.asarray(singular_values, dtype=np.float64), 0.0)
    log_manipulability = float(np.sum(np.log(np.maximum(singular_values, 1.0e-12))))
    manipulability = float(np.exp(np.clip(log_manipulability, -60.0, 60.0)))
    penalty = float(1.0 / (manipulability + 1.0e-6))
    return manipulability, min(penalty, 1.0e12)


def derivative_cost(q: np.ndarray, order: int) -> float:
    values = np.diff(np.asarray(q, dtype=np.float64), n=order, axis=0)
    return float(np.mean(np.sum(np.square(values), axis=1))) if values.size else 0.0


def boundary_metrics(context: WindowContext, q: np.ndarray, execution_horizon: int) -> Dict[str, float]:
    prefix = np.asarray(q[:execution_horizon], dtype=np.float64)
    steps: List[np.ndarray] = []
    entry = None if context.previous_q is None else prefix[0] - context.previous_q
    exit_step = context.tail_q - prefix[-1]
    if entry is not None:
        steps.append(entry)
    steps.append(exit_step)
    max_abs = max(float(np.max(np.abs(item))) for item in steps)
    max_l2 = max(float(np.linalg.norm(item)) for item in steps)
    acceleration_vectors: List[np.ndarray] = []
    if context.previous_q is not None and context.previous_previous_q is not None:
        previous_velocity = context.previous_q - context.previous_previous_q
        entry_velocity = prefix[0] - context.previous_q
        acceleration_vectors.append(entry_velocity - previous_velocity)
    if execution_horizon >= 2:
        prefix_velocity = prefix[-1] - prefix[-2]
        exit_velocity = context.tail_q - prefix[-1]
        acceleration_vectors.append(exit_velocity - prefix_velocity)
    acceleration = (
        max(float(np.linalg.norm(item)) for item in acceleration_vectors)
        if acceleration_vectors else 0.0
    )
    return {
        "entry_boundary_available": float(entry is not None),
        "entry_boundary_step_max_abs_rad": (
            float(np.max(np.abs(entry))) if entry is not None else 0.0
        ),
        "entry_boundary_step_l2_rad": (
            float(np.linalg.norm(entry)) if entry is not None else 0.0
        ),
        "exit_boundary_step_max_abs_rad": float(np.max(np.abs(exit_step))),
        "exit_boundary_step_l2_rad": float(np.linalg.norm(exit_step)),
        "boundary_step_max_abs_rad": max_abs,
        "boundary_step_l2_rad": max_l2,
        "boundary_acceleration_discontinuity": acceleration,
        "boundary_finite": float(all(np.all(np.isfinite(item)) for item in steps)),
    }


def trajectory_metrics(
    robot: RobotContext, window: WindowContext, q: np.ndarray,
    execution_horizon: int, safety_margin: float,
) -> Dict[str, Any]:
    values = np.asarray(q, dtype=np.float64)
    result: Dict[str, Any] = {
        "finite": bool(values.shape == (len(window.desired), 6) and np.all(np.isfinite(values)))
    }
    if not result["finite"]:
        return result
    ee = fk_trajectory(robot, values)
    errors = np.linalg.norm(ee - window.desired, axis=1)
    limits = check_joint_limits(
        values, robot.lower, robot.upper, robot.joint_names,
        tolerance=HARD_JOINT_LIMIT_TOLERANCE_RAD,
        safety_margin=safety_margin,
    )
    prefix_limits = check_joint_limits(
        values[:execution_horizon], robot.lower, robot.upper, robot.joint_names,
        tolerance=HARD_JOINT_LIMIT_TOLERANCE_RAD,
        safety_margin=safety_margin,
    )
    internal_steps = np.diff(values, axis=0)
    boundary = boundary_metrics(window, values, execution_horizon)
    singularities = []
    manipulabilities = []
    for row in values:
        manipulability, penalty = manipulability_and_penalty(positional_jacobian(robot, row))
        manipulabilities.append(manipulability)
        singularities.append(penalty)
    for region, region_slice in (
        ("prefix", slice(0, execution_horizon)),
        ("full", slice(0, len(values))),
    ):
        region_errors = errors[region_slice]
        region_q = values[region_slice]
        region_steps = np.diff(region_q, axis=0)
        result.update({
            f"{region}_cartesian_mean_error_m": float(np.mean(region_errors)),
            f"{region}_cartesian_rms_error_m": float(np.sqrt(np.mean(np.square(region_errors)))),
            f"{region}_cartesian_median_error_m": float(np.median(region_errors)),
            f"{region}_cartesian_p95_error_m": float(np.percentile(region_errors, 95.0)),
            f"{region}_cartesian_max_error_m": float(np.max(region_errors)),
            f"{region}_velocity_cost": derivative_cost(region_q, 1),
            f"{region}_acceleration_cost": derivative_cost(region_q, 2),
            f"{region}_jerk_cost": derivative_cost(region_q, 3),
            f"{region}_maximum_absolute_joint_step_rad": (
                float(np.max(np.abs(region_steps))) if region_steps.size else 0.0
            ),
            f"{region}_maximum_l2_joint_step_rad": (
                float(np.max(np.linalg.norm(region_steps, axis=1))) if region_steps.size else 0.0
            ),
            f"{region}_singularity_penalty": float(np.mean(np.asarray(singularities)[region_slice])),
            f"{region}_minimum_manipulability": float(np.min(np.asarray(manipulabilities)[region_slice])),
        })
    result.update(boundary)
    result.update({
        "ee": ee,
        "maximum_absolute_joint_step_rad": max(
            float(np.max(np.abs(internal_steps))) if internal_steps.size else 0.0,
            boundary["boundary_step_max_abs_rad"],
        ),
        "maximum_l2_joint_step_rad": max(
            float(np.max(np.linalg.norm(internal_steps, axis=1))) if internal_steps.size else 0.0,
            boundary["boundary_step_l2_rad"],
        ),
        "hard_joint_limit_violation_count": int(limits["hard_joint_limit_violation_count"]),
        "hard_joint_limit_violation_magnitude": float(limits["hard_joint_limit_violation_magnitude"]),
        "safety_margin_violation_count": int(limits["safety_margin_violation_count"]),
        "minimum_joint_limit_margin_rad": float(limits["minimum_joint_limit_margin_rad"]),
        "hard_limit_violations": limits["hard_violations"],
        "prefix_hard_joint_limit_violation_count": int(
            prefix_limits["hard_joint_limit_violation_count"]
        ),
        "prefix_hard_joint_limit_violation_magnitude": float(
            prefix_limits["hard_joint_limit_violation_magnitude"]
        ),
        "prefix_safety_margin_violation_count": int(
            prefix_limits["safety_margin_violation_count"]
        ),
        "prefix_minimum_joint_limit_margin_rad": float(
            prefix_limits["minimum_joint_limit_margin_rad"]
        ),
        "full_hard_joint_limit_violation_count": int(
            limits["hard_joint_limit_violation_count"]
        ),
        "full_hard_joint_limit_violation_magnitude": float(
            limits["hard_joint_limit_violation_magnitude"]
        ),
        "full_safety_margin_violation_count": int(
            limits["safety_margin_violation_count"]
        ),
        "full_minimum_joint_limit_margin_rad": float(
            limits["minimum_joint_limit_margin_rad"]
        ),
        "branch_jump": bool(
            internal_steps.size
            and np.max(np.abs(internal_steps)) > 0.20 + HARD_JOINT_LIMIT_TOLERANCE_RAD
        ),
    })
    return result


def relative_delta(candidate: float, prior: float, floor: float) -> float:
    return float((candidate - prior) / max(abs(prior), floor))


def delta_score(
    candidate: Mapping[str, Any], prior: Mapping[str, Any],
    weights: ScoreWeights, floors: MetricFloors,
) -> float:
    terms = (
        (weights.cart_mean, "prefix_cartesian_mean_error_m", floors.cartesian_m),
        (weights.cart_p95, "prefix_cartesian_p95_error_m", floors.cartesian_m),
        (weights.cart_max, "prefix_cartesian_max_error_m", floors.cartesian_m),
        (weights.acceleration, "prefix_acceleration_cost", floors.derivative),
        (weights.jerk, "prefix_jerk_cost", floors.derivative),
        (weights.boundary_step, "boundary_step_max_abs_rad", floors.boundary_rad),
        (weights.boundary_acceleration, "boundary_acceleration_discontinuity", floors.boundary_rad),
        (weights.singularity, "prefix_singularity_penalty", floors.singularity),
    )
    return float(sum(
        weight * relative_delta(float(candidate[key]), float(prior[key]), floor)
        for weight, key, floor in terms
    ))


def acceptance_reasons(
    candidate: Mapping[str, Any], prior: Mapping[str, Any], args: argparse.Namespace,
) -> Tuple[List[str], float, float]:
    if not bool(candidate.get("finite", False)):
        return ["nonfinite_values"], -math.inf, -math.inf
    reasons: List[str] = []
    if int(candidate["hard_joint_limit_violation_count"]) > 0:
        reasons.append("hard_joint_limit_violation")
    if float(candidate["maximum_absolute_joint_step_rad"]) > args.max_joint_step_gate + HARD_JOINT_LIMIT_TOLERANCE_RAD:
        reasons.append("maximum_joint_step_gate")
    if not bool(candidate["boundary_finite"]):
        reasons.append("nonfinite_boundary")
    if bool(candidate["branch_jump"]):
        reasons.append("ik_branch_jump")
    catastrophic = max(
        2.0 * float(prior["full_cartesian_mean_error_m"]),
        float(prior["full_cartesian_mean_error_m"]) + 0.01,
    )
    if float(candidate["full_cartesian_mean_error_m"]) > catastrophic:
        reasons.append("catastrophic_full_window_degradation")
    improvement = float(prior["prefix_cartesian_mean_error_m"]) - float(
        candidate["prefix_cartesian_mean_error_m"]
    )
    relative = improvement / max(float(prior["prefix_cartesian_mean_error_m"]), 1e-12)
    required = max(
        args.min_cartesian_improvement_m,
        args.min_cartesian_improvement_fraction * float(prior["prefix_cartesian_mean_error_m"]),
    )
    if improvement < required:
        reasons.append("insufficient_cartesian_improvement")
    smooth = args.smoothness_relative_tolerance
    for key in ("prefix_acceleration_cost", "prefix_jerk_cost"):
        allowed = float(prior[key]) * (1.0 + smooth) + 1.0e-10
        if float(candidate[key]) > allowed:
            reasons.append(f"{key}_degradation")
    for key in ("boundary_step_max_abs_rad", "boundary_acceleration_discontinuity"):
        if float(candidate[key]) > float(prior[key]) + args.boundary_absolute_tolerance:
            reasons.append(f"{key}_degradation")
    return reasons, improvement, relative


def flatten_metrics(prefix: str, metrics: Mapping[str, Any]) -> Dict[str, Any]:
    excluded = {"ee", "hard_limit_violations", "finite", "branch_jump"}
    return {
        f"{prefix}_{key}": value
        for key, value in metrics.items()
        if key not in excluded and np.isscalar(value)
    }


def hard_safe_state(robot: RobotContext, q: np.ndarray) -> bool:
    if not np.all(np.isfinite(q)):
        return False
    check = check_joint_limits(
        np.asarray(q, dtype=np.float64).reshape(1, 6),
        robot.lower, robot.upper, robot.joint_names,
        tolerance=HARD_JOINT_LIMIT_TOLERANCE_RAD,
        safety_margin=0.0,
    )
    if int(check["hard_joint_limit_violation_count"]) != 0:
        return False
    try:
        return bool(np.all(np.isfinite(fk_one(robot, q))))
    except Exception:
        return False


def valid_local_transition(q: np.ndarray, neighbor: Optional[np.ndarray], gate: float) -> bool:
    return neighbor is None or float(np.max(np.abs(np.asarray(q) - np.asarray(neighbor)))) <= gate


def generate_dls_candidates(
    robot: RobotContext, window: WindowContext, args: argparse.Namespace,
) -> List[Candidate]:
    candidates: List[Candidate] = []
    for damping in args.dls_damping:
        raw_updates = np.zeros_like(window.prior_q)
        for timestep in range(args.horizon):
            q = window.prior_q[timestep]
            error = window.desired[timestep] - fk_one(robot, q)
            raw_updates[timestep] = dls_update(positional_jacobian(robot, q), error, damping)
        for scale in args.dls_scales:
            for variant in ("independent", "forward", "backward", "temporally_smoothed"):
                started = time.perf_counter()
                q_candidate = window.prior_q.copy()
                unresolved = 0
                if variant == "independent":
                    proposed = window.prior_q + float(scale) * raw_updates
                    for timestep in range(args.horizon):
                        if hard_safe_state(robot, proposed[timestep]):
                            q_candidate[timestep] = proposed[timestep]
                        else:
                            unresolved += 1
                elif variant in ("forward", "backward"):
                    order = range(args.horizon) if variant == "forward" else range(args.horizon - 1, -1, -1)
                    neighbor: Optional[np.ndarray] = window.previous_q if variant == "forward" else window.tail_next_q
                    for timestep in order:
                        seed_q = window.prior_q[timestep] if neighbor is None else neighbor
                        error = window.desired[timestep] - fk_one(robot, seed_q)
                        update = dls_update(positional_jacobian(robot, seed_q), error, damping)
                        proposed = seed_q + float(scale) * update
                        if (
                            hard_safe_state(robot, proposed)
                            and valid_local_transition(proposed, neighbor, args.max_joint_step_gate)
                            and np.max(np.abs(proposed - window.prior_q[timestep])) <= 0.15
                        ):
                            q_candidate[timestep] = proposed
                            neighbor = proposed
                        else:
                            unresolved += 1
                            neighbor = window.prior_q[timestep]
                else:
                    smoothed = gaussian_filter1d(raw_updates, sigma=1.5, axis=0, mode="nearest")
                    proposed = window.prior_q + float(scale) * smoothed
                    for timestep in range(args.horizon):
                        previous = window.previous_q if timestep == 0 else q_candidate[timestep - 1]
                        if hard_safe_state(robot, proposed[timestep]) and valid_local_transition(
                            proposed[timestep], previous, args.max_joint_step_gate
                        ):
                            q_candidate[timestep] = proposed[timestep]
                        else:
                            unresolved += 1
                candidates.append(Candidate(
                    method="jacobian_dls",
                    subtype=f"{variant}_lambda_{damping:g}_scale_{scale:g}",
                    residual=q_candidate - window.prior_q,
                    deterministic_seed=stable_seed(args.seed, window.path_name, window.window_start, "dls", damping, scale, variant),
                    metadata={"damping": damping, "correction_scale": scale, "variant": variant, "unresolved_timesteps": unresolved},
                    runtime_seconds=time.perf_counter() - started,
                ))
    return candidates


def solve_point_ik_dls(
    robot: RobotContext, desired: np.ndarray, seed_q: np.ndarray, prior_q: np.ndarray,
    damping: float, iterations: int, gate: float,
) -> Tuple[np.ndarray, bool, int]:
    q = np.asarray(seed_q, dtype=np.float64).copy()
    best = q.copy()
    best_error = float(np.linalg.norm(desired - fk_one(robot, q)))
    for iteration in range(iterations):
        error_vector = desired - fk_one(robot, q)
        if float(np.linalg.norm(error_vector)) <= 1.0e-5:
            return q, True, iteration + 1
        update = dls_update(positional_jacobian(robot, q), error_vector, damping)
        update -= 0.02 * (q - prior_q)
        max_update = float(np.max(np.abs(update)))
        if max_update > 0.05:
            update *= 0.05 / max_update
        proposed = q + update
        if not hard_safe_state(robot, proposed) or np.max(np.abs(proposed - prior_q)) > 0.20:
            break
        proposed_error = float(np.linalg.norm(desired - fk_one(robot, proposed)))
        if proposed_error < best_error:
            best, best_error = proposed.copy(), proposed_error
        if proposed_error <= float(np.linalg.norm(error_vector)):
            q = proposed
        else:
            q = 0.5 * (q + proposed)
    success = best_error < float(np.linalg.norm(desired - fk_one(robot, prior_q)))
    return best, success and hard_safe_state(robot, best), iterations


def sequential_ik_pass(
    robot: RobotContext, window: WindowContext, damping: float, iterations: int,
    direction: str, seed_perturbation: np.ndarray, gate: float,
) -> Tuple[np.ndarray, int, int]:
    result = window.prior_q.copy()
    unresolved = 0
    branch_rejections = 0
    order = range(32) if direction == "forward" else range(31, -1, -1)
    previous: Optional[np.ndarray] = window.previous_q if direction == "forward" else window.tail_next_q
    for timestep in order:
        principal = window.prior_q[timestep]
        seed = principal + seed_perturbation[timestep]
        if previous is not None:
            seed = 0.75 * previous + 0.25 * seed
        solved, success, _ = solve_point_ik_dls(
            robot, window.desired[timestep], seed, principal, damping, iterations, gate
        )
        if not success:
            unresolved += 1
            solved = principal
        if not valid_local_transition(solved, previous, gate) or np.max(np.abs(solved - principal)) > 0.20:
            branch_rejections += 1
            solved = principal
        result[timestep] = solved
        previous = solved
    return result, unresolved, branch_rejections


def generate_sequential_ik_candidates(
    robot: RobotContext, window: WindowContext, args: argparse.Namespace,
) -> List[Candidate]:
    candidates: List[Candidate] = []
    for damping in args.ik_damping:
        for iterations in args.ik_iteration_limits:
            for perturbation_index, amplitude in enumerate((0.0, 0.002)):
                seed = stable_seed(args.seed, window.path_name, window.window_start, "ik", damping, iterations, perturbation_index)
                rng = np.random.default_rng(seed)
                perturbation = smooth_control_residual(rng, 32, 6, amplitude, 6)
                forward, f_unresolved, f_branch = sequential_ik_pass(
                    robot, window, damping, iterations, "forward", perturbation, args.max_joint_step_gate
                )
                backward, b_unresolved, b_branch = sequential_ik_pass(
                    robot, window, damping, iterations, "backward", perturbation, args.max_joint_step_gate
                )
                variants = (
                    ("forward", forward, f_unresolved, f_branch),
                    ("backward", backward, b_unresolved, b_branch),
                    ("forward_backward_blend", 0.5 * (forward + backward), f_unresolved + b_unresolved, f_branch + b_branch),
                )
                for subtype, q_candidate, unresolved, branch in variants:
                    candidates.append(Candidate(
                        method="sequential_ik",
                        subtype=f"{subtype}_lambda_{damping:g}_iter_{iterations}_perturb_{amplitude:g}",
                        residual=q_candidate - window.prior_q,
                        deterministic_seed=seed,
                        metadata={
                            "damping": damping, "iteration_limit": iterations,
                            "seed_perturbation_amplitude": amplitude,
                            "unresolved_timesteps": unresolved,
                            "branch_rejections": branch,
                        },
                    ))
    return candidates


def interpolate_control_points(control: np.ndarray, horizon: int, pchip: bool = False) -> np.ndarray:
    control = np.asarray(control, dtype=np.float64)
    source = np.linspace(0.0, 1.0, control.shape[0])
    target = np.linspace(0.0, 1.0, horizon)
    interpolator = PchipInterpolator(source, control, axis=0) if pchip else CubicSpline(source, control, axis=0, bc_type="natural")
    return np.asarray(interpolator(target), dtype=np.float64)


def smooth_control_residual(
    rng: np.random.Generator, horizon: int, joints: int, amplitude: float,
    control_points: int,
) -> np.ndarray:
    control = rng.normal(0.0, amplitude, size=(control_points, joints))
    return interpolate_control_points(control, horizon, pchip=False)


def apply_boundary_variant(residual: np.ndarray, variant: str, execution_horizon: int) -> np.ndarray:
    result = np.asarray(residual, dtype=np.float64).copy()
    horizon = len(result)
    if variant in ("zero_start", "zero_both"):
        ramp = np.linspace(0.0, 1.0, min(6, horizon))
        result[:len(ramp)] *= ramp[:, None]
        result[0] = 0.0
    if variant in ("zero_execution", "zero_both"):
        center = min(execution_horizon, horizon - 1)
        radius = min(4, center, horizon - 1 - center)
        if radius > 0:
            weights = np.abs(np.arange(-radius, radius + 1)) / radius
            result[center - radius:center + radius + 1] *= weights[:, None]
        result[center] = 0.0
    if variant == "boundary_blend":
        start = np.linspace(0.0, 1.0, min(6, horizon))
        result[:len(start)] *= start[:, None]
        tail_start = max(execution_horizon - 3, 0)
        tail_end = min(execution_horizon + 4, horizon)
        if tail_end > tail_start:
            distances = np.abs(np.arange(tail_start, tail_end) - execution_horizon)
            result[tail_start:tail_end] *= np.minimum(distances / 3.0, 1.0)[:, None]
    return result


def generate_smooth_perturbation_candidates(
    window: WindowContext, args: argparse.Namespace,
) -> List[Candidate]:
    candidates: List[Candidate] = []
    variants = ("zero_start", "zero_execution", "zero_both", "boundary_blend")
    joint_scale = np.asarray([1.0, 0.8, 0.8, 0.7, 0.7, 0.5], dtype=np.float64)
    for amplitude in args.smooth_amplitudes:
        for sample_index in range(2):
            seed = stable_seed(args.seed, window.path_name, window.window_start, "smooth", amplitude, sample_index)
            rng = np.random.default_rng(seed)
            spline = smooth_control_residual(rng, 32, 6, amplitude, 6)
            filtered = gaussian_filter1d(rng.normal(size=(32, 6)), sigma=3.0, axis=0, mode="reflect")
            filtered /= max(float(np.max(np.abs(filtered))), 1e-12)
            base = (0.7 * spline + 0.3 * amplitude * filtered) * joint_scale[None, :]
            for variant in variants:
                residual = apply_boundary_variant(base, variant, args.execution_horizon)
                candidates.append(Candidate(
                    method="smooth_perturbation",
                    subtype=f"{variant}_amplitude_{amplitude:g}_sample_{sample_index}",
                    residual=residual,
                    deterministic_seed=seed,
                    metadata={"amplitude": amplitude, "sample_index": sample_index, "boundary_variant": variant},
                ))
    return candidates


def quick_cem_objective(
    robot: RobotContext, window: WindowContext, residual: np.ndarray,
    args: argparse.Namespace,
) -> float:
    q = window.prior_q + residual
    if not np.all(np.isfinite(q)):
        return 1.0e12
    limits = check_joint_limits(
        q, robot.lower, robot.upper, robot.joint_names,
        tolerance=HARD_JOINT_LIMIT_TOLERANCE_RAD, safety_margin=0.0,
    )
    if int(limits["hard_joint_limit_violation_count"]) > 0:
        return 1.0e10 + float(limits["hard_joint_limit_violation_magnitude"])
    boundaries = boundary_metrics(window, q, args.execution_horizon)
    steps = np.diff(q, axis=0)
    if steps.size and np.max(np.abs(steps)) > args.max_joint_step_gate:
        return 1.0e9 + float(np.max(np.abs(steps)))
    ee = fk_trajectory(robot, q[:args.execution_horizon])
    errors = np.linalg.norm(ee - window.desired[:args.execution_horizon], axis=1)
    return float(
        4.0 * np.mean(errors)
        + 2.0 * np.percentile(errors, 95.0)
        + np.max(errors)
        + 0.5 * derivative_cost(q[:args.execution_horizon], 2)
        + 0.25 * derivative_cost(q[:args.execution_horizon], 3)
        + boundaries["boundary_step_max_abs_rad"]
        + 0.5 * boundaries["boundary_acceleration_discontinuity"]
    )


def generate_cem_candidates(
    robot: RobotContext, window: WindowContext, args: argparse.Namespace,
) -> List[Candidate]:
    candidates: List[Candidate] = []
    shape = (args.cem_control_points, 6)
    for restart in range(args.cem_restarts):
        started = time.perf_counter()
        seed = stable_seed(args.seed, window.path_name, window.window_start, "cem", restart)
        rng = np.random.default_rng(seed)
        mean = np.zeros(shape, dtype=np.float64)
        std = np.full(shape, args.cem_initial_std, dtype=np.float64)
        best: List[Tuple[float, np.ndarray]] = []
        for _ in range(args.cem_iterations):
            controls = rng.normal(mean, std, size=(args.cem_candidates, *shape))
            controls = np.clip(controls, -args.cem_max_residual, args.cem_max_residual)
            scores = np.empty(args.cem_candidates, dtype=np.float64)
            residuals: List[np.ndarray] = []
            for candidate_index, control in enumerate(controls):
                residual = interpolate_control_points(control, args.horizon, pchip=True)
                residual = apply_boundary_variant(residual, "boundary_blend", args.execution_horizon)
                residual = np.clip(residual, -args.cem_max_residual, args.cem_max_residual)
                residuals.append(residual)
                scores[candidate_index] = quick_cem_objective(robot, window, residual, args)
            elite_indices = np.argsort(scores, kind="stable")[:args.cem_elites]
            elite_controls = controls[elite_indices]
            mean = 0.25 * mean + 0.75 * np.mean(elite_controls, axis=0)
            std = np.maximum(0.25 * std + 0.75 * np.std(elite_controls, axis=0), 1.0e-4)
            best.extend((float(scores[index]), residuals[index]) for index in elite_indices[:4])
            best = sorted(best, key=lambda item: item[0])[:8]
        final_residuals = [
            interpolate_control_points(mean, args.horizon, pchip=True),
            *[item[1] for item in best[:3]],
        ]
        for final_index, residual in enumerate(final_residuals):
            residual = apply_boundary_variant(residual, "boundary_blend", args.execution_horizon)
            residual = np.clip(residual, -args.cem_max_residual, args.cem_max_residual)
            candidates.append(Candidate(
                method="spline_cem",
                subtype=f"restart_{restart}_final_{final_index}",
                residual=residual,
                deterministic_seed=seed,
                metadata={
                    "cem_restart": restart, "cem_final_index": final_index,
                    "control_points": args.cem_control_points,
                    "candidates_per_iteration": args.cem_candidates,
                    "elites": args.cem_elites, "iterations": args.cem_iterations,
                    "initial_std": args.cem_initial_std,
                    "maximum_residual_amplitude": args.cem_max_residual,
                },
                runtime_seconds=(time.perf_counter() - started) / len(final_residuals),
            ))
    return candidates


def generate_candidates(
    robot: RobotContext, window: WindowContext, args: argparse.Namespace,
) -> List[Candidate]:
    result: List[Candidate] = []
    for method in args.candidate_methods:
        started = time.perf_counter()
        try:
            if method == "jacobian_dls":
                generated = generate_dls_candidates(robot, window, args)
            elif method == "sequential_ik":
                generated = generate_sequential_ik_candidates(robot, window, args)
            elif method == "spline_cem":
                generated = generate_cem_candidates(robot, window, args)
            elif method == "smooth_perturbation":
                generated = generate_smooth_perturbation_candidates(window, args)
            else:
                raise ValueError(f"Unknown candidate method: {method}")
        except Exception as error:
            generated = [Candidate(
                method=method,
                subtype="generation_failure",
                residual=np.zeros_like(window.prior_q),
                deterministic_seed=stable_seed(
                    args.seed, window.path_name, window.window_start, method, "failure"
                ),
                metadata={
                    "generation_error": f"{type(error).__name__}: {error}",
                    "unresolved_timesteps": args.horizon,
                },
            )]
        elapsed = time.perf_counter() - started
        if generated and all(item.runtime_seconds == 0.0 for item in generated):
            per_candidate = elapsed / len(generated)
            for item in generated:
                item.runtime_seconds = per_candidate
        result.extend(generated)
    return result


PARETO_METRICS = (
    "candidate_prefix_cartesian_mean_error_m",
    "candidate_prefix_cartesian_p95_error_m",
    "candidate_prefix_acceleration_cost",
    "candidate_prefix_jerk_cost",
    "candidate_boundary_step_max_abs_rad",
    "candidate_prefix_singularity_penalty",
)


def dominates(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    left_values = np.asarray([float(left[key]) for key in PARETO_METRICS])
    right_values = np.asarray([float(right[key]) for key in PARETO_METRICS])
    return bool(np.all(left_values <= right_values + 1.0e-15) and np.any(left_values < right_values - 1.0e-15))


def pareto_front(rows: Sequence[Mapping[str, Any]]) -> List[int]:
    result: List[int] = []
    for index, row in enumerate(rows):
        if not any(
            other_index != index and dominates(other, row)
            for other_index, other in enumerate(rows)
        ):
            result.append(index)
    return result


def residual_distance(left: Mapping[str, Any], right: Mapping[str, Any]) -> float:
    difference = np.asarray(left["residual_q_window"]) - np.asarray(right["residual_q_window"])
    return float(np.sqrt(np.mean(np.square(difference))))


def diverse_target_selection(
    pareto_rows: Sequence[Dict[str, Any]], limit: int, minimum_distance: float,
) -> List[Dict[str, Any]]:
    if not pareto_rows:
        return []
    smoothness_key = lambda row: float(row["candidate_prefix_acceleration_cost"]) + float(row["candidate_prefix_jerk_cost"])
    representatives = [
        min(pareto_rows, key=lambda row: float(row["candidate_prefix_cartesian_mean_error_m"])),
        min(pareto_rows, key=smoothness_key),
        min(pareto_rows, key=lambda row: float(row["candidate_boundary_step_max_abs_rad"])),
        min(pareto_rows, key=lambda row: float(row["delta_score"])),
    ]
    selected: List[Dict[str, Any]] = []
    seen_indices: set[int] = set()
    for row in representatives:
        index = int(row["candidate_index"])
        if index in seen_indices:
            continue
        if selected and min(residual_distance(row, item) for item in selected) < minimum_distance:
            continue
        selected.append(row)
        seen_indices.add(index)
        if len(selected) >= limit:
            return selected
    remaining = sorted(
        (row for row in pareto_rows if int(row["candidate_index"]) not in seen_indices),
        key=lambda row: float(row["delta_score"]),
    )
    while remaining and len(selected) < limit:
        if not selected:
            chosen = remaining.pop(0)
        else:
            distances = [min(residual_distance(row, item) for item in selected) for row in remaining]
            eligible = [index for index, distance in enumerate(distances) if distance >= minimum_distance]
            if not eligible:
                break
            chosen_index = max(eligible, key=lambda index: (distances[index], -float(remaining[index]["delta_score"])))
            chosen = remaining.pop(chosen_index)
        selected.append(chosen)
        seen_indices.add(int(chosen["candidate_index"]))
    return selected


def candidate_result_row(
    candidate: Candidate, candidate_index: int, window: WindowContext,
    prior_metrics: Mapping[str, Any], candidate_metrics: Mapping[str, Any],
    reasons: Sequence[str], improvement: float, relative_improvement: float,
    score: float,
) -> Dict[str, Any]:
    residual = np.asarray(candidate.residual, dtype=np.float64)
    row: Dict[str, Any] = {
        "path_name": window.path_name,
        "path_index": window.path_index,
        "window_start": window.window_start,
        "candidate_method": candidate.method,
        "candidate_subtype": candidate.subtype,
        "deterministic_seed": int(candidate.deterministic_seed),
        "candidate_index": int(candidate_index),
        "valid": int(not reasons),
        "hard_safe": int(
            bool(candidate_metrics.get("finite", False))
            and int(candidate_metrics.get("hard_joint_limit_violation_count", 1)) == 0
            and float(candidate_metrics.get("maximum_absolute_joint_step_rad", math.inf)) <= 0.20 + HARD_JOINT_LIMIT_TOLERANCE_RAD
        ),
        "accepted_as_target": 0,
        "rejection_reasons": "|".join(reasons),
        "absolute_cartesian_improvement_m": float(improvement),
        "relative_cartesian_improvement": float(relative_improvement),
        "delta_score": float(score),
        "residual_rms_rad": float(np.sqrt(np.mean(np.square(residual)))),
        "residual_max_abs_rad": float(np.max(np.abs(residual))),
        "hard_limit_violation_details": json.dumps(json_safe(candidate_metrics.get("hard_limit_violations", []))),
        "runtime_seconds": float(candidate.runtime_seconds),
        "candidate_metadata": json.dumps(json_safe(candidate.metadata), sort_keys=True),
        "unresolved_timestep_count": int(candidate.metadata.get("unresolved_timesteps", 0)),
        "branch_rejection_count": int(candidate.metadata.get("branch_rejections", 0)),
        "candidate_q_window": window.prior_q + residual,
        "residual_q_window": residual,
    }
    row.update(flatten_metrics("prior", prior_metrics))
    row.update(flatten_metrics("candidate", candidate_metrics))
    return row


def zero_target_row(
    window: WindowContext, prior_metrics: Mapping[str, Any], candidate_index: int = -1,
) -> Dict[str, Any]:
    zero = np.zeros_like(window.prior_q, dtype=np.float64)
    row: Dict[str, Any] = {
        "path_name": window.path_name,
        "path_index": window.path_index,
        "window_start": window.window_start,
        "candidate_method": "retain_prior",
        "candidate_subtype": "zero_residual_no_improvement",
        "deterministic_seed": -1,
        "candidate_index": candidate_index,
        "valid": 1,
        "hard_safe": 1,
        "accepted_as_target": 1,
        "rejection_reasons": "",
        "absolute_cartesian_improvement_m": 0.0,
        "relative_cartesian_improvement": 0.0,
        "delta_score": 0.0,
        "residual_rms_rad": 0.0,
        "residual_max_abs_rad": 0.0,
        "hard_limit_violation_details": "[]",
        "runtime_seconds": 0.0,
        "candidate_metadata": "{}",
        "unresolved_timestep_count": 0,
        "branch_rejection_count": 0,
        "pareto_rank": 0,
        "target_type": "zero_residual_no_improvement",
        "improves_prior": False,
        "is_zero_residual": True,
        "prior_q_window": window.prior_q,
        "desired_path_window": window.desired,
        "prior_ee_window": window.prior_ee,
        "candidate_q_window": window.prior_q.copy(),
        "residual_q_window": zero,
    }
    row.update(flatten_metrics("prior", prior_metrics))
    row.update(flatten_metrics("candidate", prior_metrics))
    return row


def evaluate_window(
    robot: RobotContext, window: WindowContext, args: argparse.Namespace,
    weights: ScoreWeights, floors: MetricFloors,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    prior_metrics = trajectory_metrics(
        robot, window, window.prior_q, args.execution_horizon,
        args.joint_limit_safety_margin,
    )
    if (
        not prior_metrics.get("finite", False)
        or int(prior_metrics["hard_joint_limit_violation_count"]) != 0
    ):
        raise RuntimeError(
            f"Frozen strong prior is hard-invalid for {window.path_name} "
            f"window_start={window.window_start}"
        )
    generated = generate_candidates(robot, window, args)
    all_rows: List[Dict[str, Any]] = []
    evaluated_rows: List[Dict[str, Any]] = []
    for candidate_index, candidate in enumerate(generated):
        candidate_q = window.prior_q + np.asarray(candidate.residual, dtype=np.float64)
        try:
            metrics = trajectory_metrics(
                robot, window, candidate_q, args.execution_horizon,
                args.joint_limit_safety_margin,
            )
        except Exception as error:
            metrics = {"finite": False}
            candidate.metadata["evaluation_error"] = f"{type(error).__name__}: {error}"
        reasons, improvement, relative = acceptance_reasons(metrics, prior_metrics, args)
        if "generation_error" in candidate.metadata:
            reasons.append("candidate_generation_failure")
        if "evaluation_error" in candidate.metadata:
            reasons.append("candidate_evaluation_failure")
        if int(candidate.metadata.get("unresolved_timesteps", 0)) > 0:
            reasons.append("unresolved_timesteps")
        if int(candidate.metadata.get("branch_rejections", 0)) > 0:
            reasons.append("branch_change_rejected_during_generation")
        score = delta_score(metrics, prior_metrics, weights, floors) if metrics.get("finite", False) else math.inf
        row = candidate_result_row(
            candidate, candidate_index, window, prior_metrics, metrics,
            sorted(set(reasons)), improvement, relative, score,
        )
        all_rows.append(row)
        if not reasons:
            row.update({
                "target_type": "cost_improving_residual",
                "improves_prior": True,
                "is_zero_residual": False,
                "prior_q_window": window.prior_q,
                "desired_path_window": window.desired,
                "prior_ee_window": window.prior_ee,
                "candidate_q_window": candidate_q,
                "residual_q_window": np.asarray(candidate.residual, dtype=np.float64),
            })
            evaluated_rows.append(row)
    pareto_indices = pareto_front(evaluated_rows)
    pareto_rows = [evaluated_rows[index] for index in pareto_indices]
    for row in pareto_rows:
        row["pareto_rank"] = 0
    selected = diverse_target_selection(
        pareto_rows, args.targets_per_window, args.minimum_residual_distance
    )
    if not selected:
        selected = [zero_target_row(window, prior_metrics)]
    else:
        for target_index, row in enumerate(selected):
            row["accepted_as_target"] = 1
            row["target_index_within_window"] = target_index
            all_rows[int(row["candidate_index"])]["accepted_as_target"] = 1
    for target_index, row in enumerate(selected):
        row["target_index_within_window"] = target_index
    summary = {
        "path_name": window.path_name,
        "path_index": window.path_index,
        "window_start": window.window_start,
        "candidate_count": len(all_rows),
        "hard_safe_candidate_count": sum(int(row["hard_safe"]) for row in all_rows),
        "cost_improving_candidate_count": len(evaluated_rows),
        "pareto_candidate_count": len(pareto_rows),
        "selected_target_count": len(selected),
        "has_improving_target": int(any(bool(row["improves_prior"]) for row in selected)),
        "used_zero_residual": int(bool(selected[0]["is_zero_residual"])),
        "prior_prefix_cartesian_mean_error_m": float(prior_metrics["prefix_cartesian_mean_error_m"]),
        "best_candidate_prefix_cartesian_mean_error_m": float(min(
            (row["candidate_prefix_cartesian_mean_error_m"] for row in evaluated_rows),
            default=prior_metrics["prefix_cartesian_mean_error_m"],
        )),
        "best_absolute_cartesian_improvement_m": float(max(
            (row["absolute_cartesian_improvement_m"] for row in evaluated_rows), default=0.0
        )),
    }
    return all_rows, selected, summary


def make_window_contexts(
    data: Mapping[str, np.ndarray], timelines: Mapping[str, Mapping[str, np.ndarray]],
    selected_names: Sequence[str],
) -> List[WindowContext]:
    path_indices = {name: index for index, name in enumerate(selected_names)}
    result: List[WindowContext] = []
    for name in selected_names:
        mask = data["path_names"] == name
        indices = np.flatnonzero(mask)
        indices = sorted(indices.tolist(), key=lambda index: int(data["window_starts"][index]))
        for index in indices:
            start = int(data["window_starts"][index])
            full_q = timelines[name]["prior_q"]
            tail_index = start + 8
            result.append(WindowContext(
                path_name=name,
                path_index=path_indices[name],
                window_start=start,
                prior_q=np.asarray(data["prior_q_window"][index], dtype=np.float64),
                desired=np.asarray(data["desired_path_window"][index], dtype=np.float64),
                prior_ee=np.asarray(data["prior_ee_window"][index], dtype=np.float64),
                previous_q=None if start == 0 else full_q[start - 1].copy(),
                previous_previous_q=None if start < 2 else full_q[start - 2].copy(),
                tail_q=full_q[tail_index].copy(),
                tail_next_q=None if tail_index + 1 >= TRAJECTORY_LENGTH else full_q[tail_index + 1].copy(),
            ))
    return result


def state_path(output_dir: Path, window: WindowContext) -> Path:
    return output_dir / "window_state" / window.path_name / f"window_{window.window_start:03d}.json"


def save_window_state(
    path: Path, all_rows: Sequence[Mapping[str, Any]], selected: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> None:
    payload = {
        "complete": True,
        "all_candidate_rows": [json_safe(row) for row in all_rows],
        "selected_target_rows": [json_safe(row) for row in selected],
        "summary": json_safe(summary),
    }
    atomic_json(path, payload)


def load_window_state(path: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not bool(payload.get("complete", False)):
        raise ValueError(f"Incomplete window state: {path}")
    return (
        list(payload["all_candidate_rows"]),
        list(payload["selected_target_rows"]),
        dict(payload["summary"]),
    )


ARRAY_COLUMNS = {
    "prior_q_window", "desired_path_window", "prior_ee_window",
    "candidate_q_window", "residual_q_window",
}


def tabular_rows(rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame([
        {key: value for key, value in row.items() if key not in ARRAY_COLUMNS}
        for row in rows
    ])


def selected_target_arrays(rows: Sequence[Mapping[str, Any]], execution_horizon: int) -> Dict[str, np.ndarray]:
    if not rows:
        raise ValueError("No selected targets are available")
    prior_q = np.stack([np.asarray(row["prior_q_window"], dtype=np.float32) for row in rows])
    residual_q = np.stack([np.asarray(row["residual_q_window"], dtype=np.float32) for row in rows])
    candidate_q = prior_q + residual_q
    arrays: Dict[str, np.ndarray] = {
        "path_names": np.asarray([str(row["path_name"]) for row in rows]),
        "path_indices": np.asarray([int(row["path_index"]) for row in rows], dtype=np.int64),
        "window_starts": np.asarray([int(row["window_start"]) for row in rows], dtype=np.int64),
        "target_indices_within_window": np.asarray([int(row["target_index_within_window"]) for row in rows], dtype=np.int64),
        "candidate_methods": np.asarray([str(row["candidate_method"]) for row in rows]),
        "target_types": np.asarray([str(row["target_type"]) for row in rows]),
        "prior_q_window": prior_q,
        "desired_path_window": np.stack([np.asarray(row["desired_path_window"], dtype=np.float32) for row in rows]),
        "prior_ee_window": np.stack([np.asarray(row["prior_ee_window"], dtype=np.float32) for row in rows]),
        "candidate_q_window": candidate_q,
        "residual_q_window": residual_q,
        "execution_horizon": np.full(len(rows), execution_horizon, dtype=np.int64),
        "improves_prior": np.asarray([bool(row["improves_prior"]) for row in rows], dtype=bool),
        "is_zero_residual": np.asarray([bool(row["is_zero_residual"]) for row in rows], dtype=bool),
    }
    source_keys = {
        "prior_prefix_cartesian_mean_error_m": "prior_prefix_cartesian_mean_error_m",
        "candidate_prefix_cartesian_mean_error_m": "candidate_prefix_cartesian_mean_error_m",
        "absolute_cartesian_improvement_m": "absolute_cartesian_improvement_m",
        "relative_cartesian_improvement": "relative_cartesian_improvement",
        "prior_prefix_cartesian_p95_error_m": "prior_prefix_cartesian_p95_error_m",
        "candidate_prefix_cartesian_p95_error_m": "candidate_prefix_cartesian_p95_error_m",
        "prior_prefix_cartesian_max_error_m": "prior_prefix_cartesian_max_error_m",
        "candidate_prefix_cartesian_max_error_m": "candidate_prefix_cartesian_max_error_m",
        "prior_acceleration_cost": "prior_prefix_acceleration_cost",
        "candidate_acceleration_cost": "candidate_prefix_acceleration_cost",
        "prior_jerk_cost": "prior_prefix_jerk_cost",
        "candidate_jerk_cost": "candidate_prefix_jerk_cost",
        "prior_boundary_step_rad": "prior_boundary_step_max_abs_rad",
        "candidate_boundary_step_rad": "candidate_boundary_step_max_abs_rad",
        "prior_singularity_penalty": "prior_prefix_singularity_penalty",
        "candidate_singularity_penalty": "candidate_prefix_singularity_penalty",
        "maximum_absolute_joint_step_rad": "candidate_maximum_absolute_joint_step_rad",
        "hard_joint_limit_violation_count": "candidate_hard_joint_limit_violation_count",
        "minimum_joint_limit_margin_rad": "candidate_minimum_joint_limit_margin_rad",
        "delta_score": "delta_score",
        "pareto_rank": "pareto_rank",
        "residual_rms_rad": "residual_rms_rad",
        "residual_max_abs_rad": "residual_max_abs_rad",
    }
    for output_key, source_key in source_keys.items():
        dtype = np.int64 if output_key in ("hard_joint_limit_violation_count", "pareto_rank") else np.float64
        arrays[output_key] = np.asarray([row[source_key] for row in rows], dtype=dtype)
    if set(arrays) != set(TARGET_ARRAY_KEYS):
        raise AssertionError(f"selected_targets keys differ: {sorted(set(TARGET_ARRAY_KEYS) ^ set(arrays))}")
    return arrays


def validate_selected_targets(
    arrays: Mapping[str, np.ndarray], selected_paths: Sequence[str], train_names: set[str],
    expected_windows: int, args: argparse.Namespace, robot: RobotContext,
    windows: Sequence[WindowContext],
) -> None:
    if not set(selected_paths) <= train_names:
        raise AssertionError("A selected path is not assigned to the training split")
    if tuple(robot.joint_names) != tuple(f"joint{i}" for i in range(1, 7)):
        raise AssertionError("Joint ordering changed")
    if robot.ee_link != "xMateCR7_link6":
        raise AssertionError("End-effector frame changed")
    if not np.array_equal(
        arrays["prior_q_window"] + arrays["residual_q_window"],
        arrays["candidate_q_window"],
    ):
        raise AssertionError("Selected residual reconstruction is not exact")
    zero = arrays["is_zero_residual"]
    if np.any(np.asarray(arrays["residual_q_window"])[zero] != 0.0):
        raise AssertionError("A zero-residual target is not exactly zero")
    nonzero = ~zero
    if np.any(np.asarray(arrays["absolute_cartesian_improvement_m"])[nonzero] <= 0.0):
        raise AssertionError("A nonzero selected target does not improve Cartesian mean error")
    if np.any(np.asarray(arrays["hard_joint_limit_violation_count"]) != 0):
        raise AssertionError("A selected target violates a hard joint limit")
    if np.any(np.asarray(arrays["maximum_absolute_joint_step_rad"]) > args.max_joint_step_gate + HARD_JOINT_LIMIT_TOLERANCE_RAD):
        raise AssertionError("A selected target violates the joint-step gate")
    identities = list(zip(
        arrays["path_names"].tolist(), arrays["window_starts"].tolist(),
        arrays["target_indices_within_window"].tolist(),
    ))
    if len(identities) != len(set(identities)):
        raise AssertionError("A target appears more than once")
    numeric_exclusions = {"path_names", "candidate_methods", "target_types"}
    for key, values in arrays.items():
        if key not in numeric_exclusions and np.issubdtype(values.dtype, np.number):
            if not np.all(np.isfinite(values)):
                raise AssertionError(f"Selected target array {key} contains nonfinite values")
    unique_windows = len(set(zip(arrays["path_names"].tolist(), arrays["window_starts"].tolist())))
    if unique_windows != expected_windows:
        raise AssertionError(f"Expected {expected_windows} unique windows, found {unique_windows}")
    window_lookup = {
        (window.path_name, window.window_start): window for window in windows
    }
    for index in range(len(arrays["path_names"])):
        identity = (
            str(arrays["path_names"][index]), int(arrays["window_starts"][index])
        )
        window = window_lookup[identity]
        prior_metrics = trajectory_metrics(
            robot, window, arrays["prior_q_window"][index],
            args.execution_horizon, args.joint_limit_safety_margin,
        )
        candidate_metrics = trajectory_metrics(
            robot, window, arrays["candidate_q_window"][index],
            args.execution_horizon, args.joint_limit_safety_margin,
        )
        if int(candidate_metrics["hard_joint_limit_violation_count"]) != 0:
            raise AssertionError(f"Saved alpha=1 target is hard-invalid: {identity}")
        if bool(arrays["is_zero_residual"][index]):
            continue
        reasons, improvement, _ = acceptance_reasons(
            candidate_metrics, prior_metrics, args
        )
        if reasons or improvement <= 0.0:
            raise AssertionError(
                f"Saved alpha=1 target fails acceptance for {identity}: {reasons}"
            )


def save_all_candidate_archive(
    output_dir: Path, window: WindowContext, rows: Sequence[Mapping[str, Any]],
) -> None:
    if not rows:
        return
    arrays = {
        "candidate_methods": np.asarray([str(row["candidate_method"]) for row in rows]),
        "candidate_subtypes": np.asarray([str(row["candidate_subtype"]) for row in rows]),
        "candidate_indices": np.asarray([int(row["candidate_index"]) for row in rows], dtype=np.int64),
        "residual_q_window": np.stack([np.asarray(row["residual_q_window"], dtype=np.float32) for row in rows]),
        "candidate_q_window": np.stack([np.asarray(row["candidate_q_window"], dtype=np.float32) for row in rows]),
    }
    path = output_dir / "all_candidates" / window.path_name / f"window_{window.window_start:03d}.npz"
    atomic_npz(path, arrays)


def grouped_summaries(
    all_frame: pd.DataFrame, selected_frame: pd.DataFrame, window_frame: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    path_rows = []
    for path_name, group in window_frame.groupby("path_name", sort=True):
        targets = selected_frame[selected_frame["path_name"] == path_name]
        path_rows.append({
            "path_name": path_name,
            "window_count": len(group),
            "windows_with_improvement": int(group["has_improving_target"].sum()),
            "windows_without_improvement": int(group["used_zero_residual"].sum()),
            "improving_window_ratio": float(group["has_improving_target"].mean()),
            "selected_target_count": len(targets),
            "zero_residual_target_fraction": float(targets["is_zero_residual"].mean()),
            "mean_cartesian_improvement_m": float(targets["absolute_cartesian_improvement_m"].mean()),
        })
    method_rows = []
    for method, group in all_frame.groupby("candidate_method", sort=True):
        method_rows.append({
            "candidate_method": method,
            "generated_count": len(group),
            "hard_safe_count": int(group["hard_safe"].sum()),
            "cost_improving_count": int(group["valid"].sum()),
            "selected_target_count": int(group["accepted_as_target"].sum()),
            "hard_safe_rate": float(group["hard_safe"].mean()),
            "cost_improving_rate": float(group["valid"].mean()),
            "selected_yield": float(group["accepted_as_target"].mean()),
            "mean_runtime_seconds": float(group["runtime_seconds"].mean()),
        })
    failed = window_frame[window_frame["used_zero_residual"] == 1].copy()
    pareto = window_frame[[
        "path_name", "path_index", "window_start", "candidate_count",
        "hard_safe_candidate_count", "cost_improving_candidate_count",
        "pareto_candidate_count", "selected_target_count",
    ]].copy()
    return pd.DataFrame(path_rows), pd.DataFrame(method_rows), failed, pareto


def save_plot(path: Path, draw: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    draw(ax)
    fig.tight_layout()
    temporary = path.with_name(path.stem + ".tmp" + path.suffix)
    fig.savefig(temporary, dpi=160)
    plt.close(fig)
    os.replace(temporary, path)


def save_plots(
    output_dir: Path, all_frame: pd.DataFrame, selected_frame: pd.DataFrame,
    window_frame: pd.DataFrame, method_frame: pd.DataFrame,
) -> None:
    plots = output_dir / "plots"
    improving = selected_frame[selected_frame["improves_prior"].astype(bool)]
    save_plot(plots / "prior_vs_best_cartesian_error.png", lambda ax: (
        ax.scatter(window_frame["prior_prefix_cartesian_mean_error_m"], window_frame["best_candidate_prefix_cartesian_mean_error_m"], s=12, alpha=0.6),
        ax.plot([0, window_frame["prior_prefix_cartesian_mean_error_m"].max()], [0, window_frame["prior_prefix_cartesian_mean_error_m"].max()], color="black", linewidth=1),
        ax.set(xlabel="Prior prefix mean Cartesian error (m)", ylabel="Best candidate error (m)"),
    ))
    save_plot(plots / "cartesian_improvement_histogram.png", lambda ax: (
        ax.hist(improving["absolute_cartesian_improvement_m"] if not improving.empty else [0.0], bins=30),
        ax.set(xlabel="Cartesian mean improvement (m)", ylabel="Selected targets"),
    ))
    save_plot(plots / "candidate_method_success_rate.png", lambda ax: (
        ax.bar(method_frame["candidate_method"], method_frame["cost_improving_rate"]),
        ax.tick_params(axis="x", rotation=25),
        ax.set(ylabel="Cost-improving candidate rate"),
    ))
    save_plot(plots / "residual_magnitude_vs_improvement.png", lambda ax: (
        ax.scatter(improving["residual_rms_rad"], improving["absolute_cartesian_improvement_m"], s=12, alpha=0.6),
        ax.set(xlabel="Residual RMS (rad)", ylabel="Cartesian improvement (m)"),
    ))
    save_plot(plots / "cartesian_vs_smoothness_tradeoff.png", lambda ax: (
        ax.scatter(all_frame["candidate_prefix_cartesian_mean_error_m"], all_frame["candidate_prefix_acceleration_cost"], s=6, alpha=0.25),
        ax.set(xlabel="Candidate Cartesian mean error (m)", ylabel="Acceleration cost"),
    ))
    save_plot(plots / "cartesian_vs_boundary_tradeoff.png", lambda ax: (
        ax.scatter(all_frame["candidate_prefix_cartesian_mean_error_m"], all_frame["candidate_boundary_step_max_abs_rad"], s=6, alpha=0.25),
        ax.set(xlabel="Candidate Cartesian mean error (m)", ylabel="Boundary step (rad)"),
    ))
    save_plot(plots / "targets_per_window.png", lambda ax: (
        ax.hist(window_frame["selected_target_count"], bins=np.arange(0.5, window_frame["selected_target_count"].max() + 1.5, 1.0)),
        ax.set(xlabel="Selected targets per window", ylabel="Windows"),
    ))
    zero_by_path = selected_frame.groupby("path_name")["is_zero_residual"].mean()
    save_plot(plots / "zero_residual_fraction_by_path.png", lambda ax: (
        ax.bar(zero_by_path.index, zero_by_path.values),
        ax.tick_params(axis="x", rotation=90, labelsize=7),
        ax.set(ylabel="Zero-residual target fraction"),
    ))


def pilot_statistics(
    selected_paths: Sequence[str], all_frame: pd.DataFrame,
    selected_frame: pd.DataFrame, window_frame: pd.DataFrame,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    improvements = selected_frame["absolute_cartesian_improvement_m"].to_numpy(float)
    acceleration_change = (
        selected_frame["candidate_prefix_acceleration_cost"].to_numpy(float)
        - selected_frame["prior_prefix_acceleration_cost"].to_numpy(float)
    )
    jerk_change = (
        selected_frame["candidate_prefix_jerk_cost"].to_numpy(float)
        - selected_frame["prior_prefix_jerk_cost"].to_numpy(float)
    )
    boundary_change = (
        selected_frame["candidate_boundary_step_max_abs_rad"].to_numpy(float)
        - selected_frame["prior_boundary_step_max_abs_rad"].to_numpy(float)
    )
    singularity_change = (
        selected_frame["candidate_prefix_singularity_penalty"].to_numpy(float)
        - selected_frame["prior_prefix_singularity_penalty"].to_numpy(float)
    )
    improving_window_ratio = float(window_frame["has_improving_target"].mean())
    nonzero = selected_frame[~selected_frame["is_zero_residual"].astype(bool)]
    prior_smooth = nonzero["prior_prefix_acceleration_cost"] + nonzero["prior_prefix_jerk_cost"]
    candidate_smooth = nonzero["candidate_prefix_acceleration_cost"] + nonzero["candidate_prefix_jerk_cost"]
    mean_smoothness_relative_change = float(np.mean(
        (candidate_smooth - prior_smooth) / np.maximum(np.abs(prior_smooth), 1e-8)
    )) if len(nonzero) else 0.0
    all_selected_safe = bool((selected_frame["candidate_hard_joint_limit_violation_count"] == 0).all())
    all_nonzero_improve = bool((nonzero["absolute_cartesian_improvement_m"] > 0.0).all())
    ready = (
        improving_window_ratio >= 0.60
        and all_selected_safe
        and all_nonzero_improve
        and mean_smoothness_relative_change <= args.smoothness_relative_tolerance
    )
    return {
        "classification": "READY_FOR_V7_DATASET_EXPANSION" if ready else "TARGET_GENERATION_NEEDS_REVISION",
        "selected_path_count": len(selected_paths),
        "total_window_count": len(window_frame),
        "total_candidates_generated": len(all_frame),
        "hard_safe_candidate_count": int(all_frame["hard_safe"].sum()),
        "cost_improving_candidate_count": int(all_frame["valid"].sum()),
        "windows_with_at_least_one_improvement": int(window_frame["has_improving_target"].sum()),
        "windows_with_no_improvement": int(window_frame["used_zero_residual"].sum()),
        "improving_window_ratio": improving_window_ratio,
        "zero_residual_target_fraction": float(selected_frame["is_zero_residual"].mean()),
        "mean_targets_per_window": float(window_frame["selected_target_count"].mean()),
        "candidate_yield_by_method": all_frame.groupby("candidate_method")["valid"].mean().to_dict(),
        "mean_cartesian_improvement_m": float(np.mean(improvements)),
        "median_cartesian_improvement_m": float(np.median(improvements)),
        "p95_cartesian_improvement_m": float(np.percentile(improvements, 95.0)),
        "maximum_cartesian_improvement_m": float(np.max(improvements)),
        "mean_residual_rms_rad": float(selected_frame["residual_rms_rad"].mean()),
        "maximum_residual_magnitude_rad": float(selected_frame["residual_max_abs_rad"].max()),
        "mean_acceleration_change": float(np.mean(acceleration_change)),
        "mean_jerk_change": float(np.mean(jerk_change)),
        "mean_boundary_step_change_rad": float(np.mean(boundary_change)),
        "mean_singularity_penalty_change": float(np.mean(singularity_change)),
        "mean_smoothness_relative_change": mean_smoothness_relative_change,
        "hard_limit_violation_count_among_selected_targets": int(selected_frame["candidate_hard_joint_limit_violation_count"].sum()),
        "all_selected_targets_pass_hard_gates": all_selected_safe,
        "all_selected_nonzero_targets_improve": all_nonzero_improve,
        "targets_evaluated_at_alpha": 1.0,
        "expert_q_loaded": False,
        "validation_or_test_data_loaded": False,
        "recursive_rollout_performed": False,
    }


def generation_signature(args: argparse.Namespace, selected_paths: Sequence[str]) -> str:
    relevant = {
        key: json_safe(value)
        for key, value in vars(args).items()
        if key not in {"resume", "overwrite", "device", "save_all_candidates"}
    }
    relevant["selected_paths"] = list(selected_paths)
    encoded = json.dumps(relevant, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def prepare_output_directory(args: argparse.Namespace) -> None:
    if args.output_dir.exists() and args.overwrite:
        shutil.rmtree(args.output_dir)
    if args.output_dir.exists() and not args.resume:
        if any(args.output_dir.iterdir()):
            raise FileExistsError(
                f"{args.output_dir} is not empty; pass --overwrite or --resume"
            )
    args.output_dir.mkdir(parents=True, exist_ok=True)


def main() -> int:
    args = parse_args()
    validate_args(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(args.device)
    prepare_output_directory(args)

    _manifest, train_names = load_train_manifest(args.split_manifest)
    data = load_window_data(args.train_windows)
    prior_names = load_prior_path_names(args.train_prior)
    timelines = reconstruct_timelines(data)
    selected_names, selected_paths_frame = select_paths(
        timelines, train_names, args.path_names, args.num_paths, args.seed
    )
    if args.path_names is None and len(selected_names) != args.num_paths:
        raise AssertionError("Pilot path selection did not produce --num_paths paths")
    if args.path_names is None and args.num_paths == 20:
        counts = selected_paths_frame["difficulty_group"].value_counts().to_dict()
        if any(counts.get(label, 0) != 5 for label in ("low", "lower_middle", "upper_middle", "high")):
            raise AssertionError(f"20-path stratification is not five per quartile: {counts}")
    if prior_names is not None and not set(selected_names) <= prior_names:
        raise ValueError("Frozen train_prior.npz does not contain every selected training path")

    signature = generation_signature(args, selected_names)
    configuration_path = args.output_dir / "pilot_configuration.json"
    if args.resume and configuration_path.is_file():
        with configuration_path.open("r", encoding="utf-8") as handle:
            prior_configuration = json.load(handle)
        if prior_configuration.get("generation_signature") != signature:
            raise ValueError("--resume configuration differs from the existing pilot")

    weights = ScoreWeights(
        args.w_cart_mean, args.w_cart_p95, args.w_cart_max,
        args.w_acceleration, args.w_jerk, args.w_boundary_step,
        args.w_boundary_acceleration, args.w_singularity,
    )
    floors = MetricFloors(
        args.floor_cartesian_m, args.floor_derivative,
        args.floor_boundary_rad, args.floor_singularity,
    )
    configuration = {
        "generation_signature": signature,
        "arguments": vars(args),
        "resolved_device": str(device),
        "scientific_target": "candidate_q_minus_frozen_strong_prior_q",
        "expert_q_loaded": False,
        "loaded_window_keys": list(WINDOW_KEYS),
        "allowed_split": "train",
        "manifest_train_path_count": len(train_names),
        "selected_paths": selected_names,
        "selected_path_count": len(selected_names),
        "trajectory_length": TRAJECTORY_LENGTH,
        "horizon": args.horizon,
        "execution_horizon": args.execution_horizon,
        "stride": WINDOW_STRIDE,
        "expected_window_starts": list(EXPECTED_WINDOW_STARTS),
        "robot": "ROKAE xMateCR7",
        "robot_urdf": str(resolve_project_path(args.robot_urdf)),
        "joint_names": list(DEFAULT_JOINT_NAMES),
        "ee_link": args.ee_link,
        "hard_joint_limit_tolerance_rad": HARD_JOINT_LIMIT_TOLERANCE_RAD,
        "safety_margin_is_hard_gate": False,
        "relative_score_weights": asdict(weights),
        "relative_score_floors": asdict(floors),
        "candidate_methods": list(args.candidate_methods),
        "alpha_used_for_target_validation": 1.0,
        "official_test_data_loaded": False,
        "recursive_rollout": False,
    }
    atomic_json(configuration_path, configuration)
    atomic_csv(args.output_dir / "selected_paths.csv", selected_paths_frame)

    robot = make_robot_context(args)
    configuration["hard_lower_limits"] = robot.lower.tolist()
    configuration["hard_upper_limits"] = robot.upper.tolist()
    atomic_json(configuration_path, configuration)

    windows = make_window_contexts(data, timelines, selected_names)
    expected_full_window_count = len(selected_names) * len(EXPECTED_WINDOW_STARTS)
    if len(windows) != expected_full_window_count:
        raise AssertionError(
            f"Expected {expected_full_window_count} selected windows, found {len(windows)}"
        )
    if args.max_windows is not None:
        windows = windows[:args.max_windows]

    all_candidate_rows: List[Dict[str, Any]] = []
    selected_target_rows: List[Dict[str, Any]] = []
    window_summaries: List[Dict[str, Any]] = []
    for window_number, window in enumerate(windows, start=1):
        state = state_path(args.output_dir, window)
        loaded = False
        if args.resume and state.is_file():
            try:
                all_rows, selected, summary = load_window_state(state)
                loaded = True
            except (OSError, ValueError, KeyError, json.JSONDecodeError):
                loaded = False
        if not loaded:
            all_rows, selected, summary = evaluate_window(
                robot, window, args, weights, floors
            )
            save_window_state(state, all_rows, selected, summary)
        if args.save_all_candidates:
            save_all_candidate_archive(args.output_dir, window, all_rows)
        all_candidate_rows.extend(all_rows)
        selected_target_rows.extend(selected)
        window_summaries.append(summary)
        print(
            f"[{window_number}/{len(windows)}] {window.path_name} "
            f"start={window.window_start} candidates={summary['candidate_count']} "
            f"improving={summary['cost_improving_candidate_count']} "
            f"targets={summary['selected_target_count']}"
            + (" [resumed]" if loaded else "")
        )

    all_frame = tabular_rows(all_candidate_rows)
    selected_frame = tabular_rows(selected_target_rows)
    window_frame = pd.DataFrame(window_summaries)
    arrays = selected_target_arrays(selected_target_rows, args.execution_horizon)
    validate_selected_targets(
        arrays, selected_names, train_names, len(windows), args, robot, windows
    )
    path_frame, method_frame, failed_frame, pareto_frame = grouped_summaries(
        all_frame, selected_frame, window_frame
    )
    summary = pilot_statistics(
        selected_names, all_frame, selected_frame, window_frame, args
    )
    complete_pilot = len(windows) == expected_full_window_count
    summary["complete_pilot"] = complete_pilot
    if not complete_pilot:
        summary["classification"] = "TARGET_GENERATION_NEEDS_REVISION"
        summary["classification_reason"] = "Pilot is incomplete because --max_windows was used"

    atomic_npz(args.output_dir / "selected_targets.npz", arrays)
    atomic_csv(args.output_dir / "all_candidate_results.csv", all_frame)
    atomic_csv(args.output_dir / "selected_target_results.csv", selected_frame)
    atomic_csv(args.output_dir / "per_window_summary.csv", window_frame)
    atomic_csv(args.output_dir / "per_path_summary.csv", path_frame)
    atomic_csv(args.output_dir / "candidate_method_summary.csv", method_frame)
    atomic_csv(args.output_dir / "failed_windows.csv", failed_frame)
    atomic_csv(args.output_dir / "pareto_set_summary.csv", pareto_frame)
    atomic_json(args.output_dir / "pilot_summary.json", summary)
    save_plots(args.output_dir, all_frame, selected_frame, window_frame, method_frame)

    print(json.dumps(json_safe(summary), indent=2, sort_keys=True))
    print(f"Final classification: {summary['classification']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
