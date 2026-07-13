#!/usr/bin/env python3
"""Teacher-forced audit of action-buffer diffusion candidate selection.

Every planning cycle is reconstructed from the buffer-only MLP trajectory.
Diffusion candidates are evaluated independently and are never propagated to a
later cycle. Expert joints are used only for columns prefixed with ``oracle_``.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import inspect
import json
import math
import random
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from build_diffusion_v5_residual_window_dataset import (
    read_predicted_q_csv,
    safe_path_name,
)
from build_diffusion_v5b_residual_window_dataset_fk_condition import (
    fk_positions,
    load_fk_context,
)
from diagnose_diffusion_v5_sampling_modes import reverse_noised_x0_batches
from evaluate_prior_refinement_fk_robot_costs import (
    Weights,
    drawing_fidelity_metrics,
    drawing_total_cost,
    shape_path_metrics,
    shape_total_cost,
    smoothness_costs,
    total_cost,
)
from sample_conditional_diffusion_trajectory_v5_residual_unet import (
    diffusion_config_from_checkpoint,
    instantiate_checkpoint_model,
    make_schedule,
    torch_load_checkpoint,
)


DEFAULT_CHECKPOINT = Path(
    "data/cartesian_expert_dataset_v3/"
    "diffusion_v5b_residual_unet_fk_condition/best_checkpoint.pt"
)
DEFAULT_TEST_NPZ = Path(
    "data/cartesian_expert_dataset_v3/diffusion_v2/diffusion_test_v2.npz"
)
DEFAULT_STATS_NPZ = Path(
    "data/cartesian_expert_dataset_v3/"
    "diffusion_v5b_residual_windows_fk_condition/normalization_stats.npz"
)
DEFAULT_PRIOR_DIR = Path(
    "data/cartesian_expert_dataset_v3/mlp_v3_test_predictions"
)
DEFAULT_OUTPUT_DIR = Path(
    "data/cartesian_expert_dataset_v3/"
    "action_buffer_candidate_selection_diagnostic"
)
JOINT_DIM = 6
CONDITION_DIM = 38
EPS = 1e-12


CONDITION_FEATURES: Tuple[Tuple[str, str], ...] = (
    ("desired_x", "desired_path"),
    ("desired_y", "desired_path"),
    ("desired_z", "desired_path"),
    ("desired_dx", "desired_path_delta"),
    ("desired_dy", "desired_path_delta"),
    ("desired_dz", "desired_path_delta"),
    ("progress", "time_index"),
    *((f"q_start_q{joint}", "q_start") for joint in range(1, 7)),
    *((f"current_q{joint}", "current_q") for joint in range(1, 7)),
    *((f"prior_q{joint}", "prior_joint_trajectory") for joint in range(1, 7)),
    *((f"prior_delta_q{joint}", "prior_delta_from_start") for joint in range(1, 7)),
    ("prior_ee_x", "prior_fk_position"),
    ("prior_ee_y", "prior_fk_position"),
    ("prior_ee_z", "prior_fk_position"),
    ("prior_ee_error_x", "cartesian_prior_error"),
    ("prior_ee_error_y", "cartesian_prior_error"),
    ("prior_ee_error_z", "cartesian_prior_error"),
    ("prior_ee_error_norm", "cartesian_prior_error_norm"),
)
if len(CONDITION_FEATURES) != CONDITION_DIM:
    raise RuntimeError("The v5b semantic condition map must contain 38 channels")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit teacher-forced action-buffer candidate selection."
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--test_npz", type=Path, default=DEFAULT_TEST_NPZ)
    parser.add_argument("--stats_npz", type=Path, default=DEFAULT_STATS_NPZ)
    parser.add_argument("--prior_dir", type=Path, default=DEFAULT_PRIOR_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--prediction_horizon", type=int, default=32)
    parser.add_argument("--execution_horizon", type=int, default=8)
    parser.add_argument("--t_init", type=int, default=10)
    parser.add_argument("--num_candidates", type=int, default=16)
    parser.add_argument("--max_paths", type=int, default=20)
    parser.add_argument("--ranking_discount", type=float, default=0.9)
    parser.add_argument("--allowed_boundary_step", type=float, default=0.25)
    parser.add_argument("--allowed_prefix_step", type=float, default=0.25)
    parser.add_argument("--boundary_ratio", type=float, default=2.0)
    parser.add_argument("--max_step_ratio", type=float, default=2.0)
    parser.add_argument("--continuity_weight", type=float, default=1.0)
    parser.add_argument("--num_diffusion_steps", type=int, default=None)
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda" and not torch.cuda.is_available():
        print("[device] CUDA requested but unavailable; using CPU")
        return torch.device("cpu")
    return torch.device(value)


def load_npz(path: Path, label: str) -> Dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def require_keys(data: Dict[str, np.ndarray], keys: Sequence[str], label: str) -> None:
    missing = [key for key in keys if key not in data]
    if missing:
        raise KeyError(f"{label} missing required key(s): {', '.join(missing)}")


def decode_names(values: np.ndarray) -> List[str]:
    return [
        value.decode("utf-8", errors="replace")
        if isinstance(value, bytes)
        else str(value)
        for value in np.asarray(values).reshape(-1)
    ]


def finite_array(values: np.ndarray, label: str) -> np.ndarray:
    array = np.asarray(values)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} contains NaN or infinity")
    return array


def load_stats(path: Path) -> Dict[str, np.ndarray]:
    stats = load_npz(path, "normalization stats")
    require_keys(
        stats,
        ("condition_mean", "condition_std", "residual_mean", "residual_std"),
        "normalization stats",
    )
    condition_mean = finite_array(stats["condition_mean"], "condition_mean").astype(np.float32)
    condition_std = finite_array(stats["condition_std"], "condition_std").astype(np.float32)
    residual_mean = finite_array(stats["residual_mean"], "residual_mean").astype(np.float32)
    residual_std = finite_array(stats["residual_std"], "residual_std").astype(np.float32)
    if condition_mean.shape != (CONDITION_DIM,) or condition_std.shape != (CONDITION_DIM,):
        raise ValueError("v5b condition statistics must have shape (38,)")
    if residual_mean.shape != (JOINT_DIM,) or residual_std.shape != (JOINT_DIM,):
        raise ValueError("v5b residual statistics must have shape (6,)")
    if np.any(condition_std <= 0.0) or np.any(residual_std <= 0.0):
        raise ValueError("Normalization standard deviations must be positive")
    return {
        "condition_mean": condition_mean,
        "condition_std": condition_std,
        "residual_mean": residual_mean,
        "residual_std": residual_std,
    }


def default_weights() -> Weights:
    return Weights(
        cart=1.0,
        max_cart=0.25,
        start=0.5,
        end=0.5,
        frechet=1.0,
        dtw=0.5,
        vel=0.01,
        acc=0.01,
        jerk=0.001,
        limit=10.0,
        tangent=0.5,
        progress=0.5,
        length_ratio=0.25,
        norm_shape=1.0,
    )


def pad_indices(start: int, horizon: int, length: int) -> np.ndarray:
    if not 0 <= start < length:
        raise ValueError(f"Planning start {start} is outside trajectory length {length}")
    return np.minimum(np.arange(start, start + horizon), length - 1)


def desired_differences(path: np.ndarray) -> np.ndarray:
    output = np.zeros_like(path, dtype=np.float32)
    if len(path) > 1:
        output[:-1] = path[1:] - path[:-1]
        output[-1] = output[-2]
    return output


def build_teacher_forced_buffer(
    prior_q: np.ndarray,
    start: int,
    horizon: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Shift/extend the buffer-only plan without propagating diffusion choices."""
    indices = pad_indices(start, horizon, prior_q.shape[0])
    buffer = np.asarray(prior_q[indices], dtype=np.float32).copy()
    finite_array(buffer, "teacher-forced action buffer")
    return buffer, indices


def build_v5b_condition(
    *,
    desired_path: np.ndarray,
    desired_delta: np.ndarray,
    indices: np.ndarray,
    q_start: np.ndarray,
    current_q: np.ndarray,
    buffer_q: np.ndarray,
    robot: Any,
    joint_names: Sequence[str],
    ee_link: str,
) -> Tuple[np.ndarray, np.ndarray]:
    horizon = buffer_q.shape[0]
    desired_window = desired_path[indices]
    desired_delta_window = desired_delta[indices]
    progress = indices.astype(np.float32) / max(desired_path.shape[0] - 1, 1)
    q_start_window = np.repeat(q_start.reshape(1, JOINT_DIM), horizon, axis=0)
    current_window = np.repeat(current_q.reshape(1, JOINT_DIM), horizon, axis=0)
    prior_delta = buffer_q - q_start_window
    prior_ee = fk_positions(robot, joint_names, ee_link, buffer_q)
    prior_error = prior_ee - desired_window
    prior_error_norm = np.linalg.norm(prior_error, axis=1, keepdims=True)
    condition = np.concatenate(
        (
            desired_window,
            desired_delta_window,
            progress[:, None],
            q_start_window,
            current_window,
            buffer_q,
            prior_delta,
            prior_ee,
            prior_error,
            prior_error_norm,
        ),
        axis=1,
    ).astype(np.float32)
    if condition.shape != (horizon, CONDITION_DIM):
        raise RuntimeError(f"Condition must have shape ({horizon},38), got {condition.shape}")
    finite_array(condition, "action-buffer condition")
    return condition, prior_ee


def condition_ood_rows(
    *,
    path_name: str,
    cycle_index: int,
    start_index: int,
    condition: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, float]]:
    z = ((condition - mean[None, :]) / std[None, :]).astype(np.float32)
    finite_array(z, "normalized action-buffer condition")
    rows: List[Dict[str, Any]] = []
    for channel, (feature, group) in enumerate(CONDITION_FEATURES):
        absolute = np.abs(z[:, channel])
        rows.append(
            {
                "path_name": path_name,
                "planning_cycle_index": cycle_index,
                "trajectory_start_index": start_index,
                "condition_channel": channel,
                "condition_feature": feature,
                "condition_group": group,
                "mean_abs_z": float(np.mean(absolute)),
                "max_abs_z": float(np.max(absolute)),
                "fraction_abs_z_gt_3": float(np.mean(absolute > 3.0)),
                "fraction_abs_z_gt_5": float(np.mean(absolute > 5.0)),
                "fraction_abs_z_gt_10": float(np.mean(absolute > 10.0)),
            }
        )
    absolute_all = np.abs(z)
    summary = {
        "condition_mean_abs_z": float(np.mean(absolute_all)),
        "condition_max_abs_z": float(np.max(absolute_all)),
        "condition_fraction_abs_z_gt_3": float(np.mean(absolute_all > 3.0)),
        "condition_fraction_abs_z_gt_5": float(np.mean(absolute_all > 5.0)),
        "condition_fraction_abs_z_gt_10": float(np.mean(absolute_all > 10.0)),
    }
    return z, rows, summary


def extract_joint_limits(robot: Any, joint_names: Sequence[str]) -> Tuple[np.ndarray, np.ndarray]:
    joints = getattr(getattr(robot, "robot", robot), "joints", [])
    by_name = {str(getattr(joint, "name", "")): joint for joint in joints}
    lower: List[float] = []
    upper: List[float] = []
    for name in joint_names:
        joint = by_name.get(str(name))
        limit = None if joint is None else getattr(joint, "limit", None)
        lower.append(float(getattr(limit, "lower", -np.inf)) if limit is not None else -np.inf)
        upper.append(float(getattr(limit, "upper", np.inf)) if limit is not None else np.inf)
    return np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)


def limit_metrics(q: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> Tuple[int, float]:
    below = np.where(np.isfinite(lower[None, :]), np.maximum(lower[None, :] - q, 0.0), 0.0)
    above = np.where(np.isfinite(upper[None, :]), np.maximum(q - upper[None, :], 0.0), 0.0)
    violation = below + above
    return int(np.count_nonzero(violation > 0.0)), float(np.mean(np.square(violation)))


def derivative_max_step(q: np.ndarray) -> float:
    differences = np.diff(q, axis=0)
    return float(np.max(np.abs(differences))) if differences.size else 0.0


def evaluate_region(
    *,
    q: np.ndarray,
    desired: np.ndarray,
    ee: np.ndarray,
    previous_q: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    weights: Weights,
) -> Dict[str, float]:
    finite_array(q, "candidate joint region")
    finite_array(ee, "candidate FK region")
    error = ee - desired
    distances = np.linalg.norm(error, axis=1)
    mean_cart = float(np.mean(distances))
    rms_cart = float(np.sqrt(np.mean(np.square(distances))))
    max_cart = float(np.max(distances))
    start_error, end_error, pred_length, desired_length, _, frechet, dtw = shape_path_metrics(ee, desired)
    _, tangent_weighted, progress_error, length_ratio_error, normalized_shape = drawing_fidelity_metrics(
        ee, desired, pred_length, desired_length
    )
    velocity_cost, acceleration_cost, jerk_cost = smoothness_costs(q)
    violation_count, violation_cost = limit_metrics(q, lower, upper)
    robot_cost = total_cost(
        weights, mean_cart, max_cart, velocity_cost, acceleration_cost, jerk_cost, violation_cost
    )
    shape_cost = shape_total_cost(
        weights,
        mean_cart,
        max_cart,
        start_error,
        end_error,
        frechet,
        dtw,
        velocity_cost,
        acceleration_cost,
        jerk_cost,
        violation_cost,
    )
    drawing_cost = drawing_total_cost(
        weights,
        shape_cost,
        tangent_weighted,
        progress_error,
        length_ratio_error,
        normalized_shape,
    )
    boundary = q[0] - previous_q
    return {
        "mean_cartesian_error": mean_cart,
        "rms_cartesian_error": rms_cart,
        "max_cartesian_error": max_cart,
        "drawing_cost": drawing_cost,
        "velocity_cost": velocity_cost,
        "acceleration_cost": acceleration_cost,
        "jerk_cost": jerk_cost,
        "max_joint_step": derivative_max_step(q),
        "joint_limit_violation_count": float(violation_count),
        "joint_limit_violation_cost": violation_cost,
        "continuity_cost": float(np.sum(np.square(boundary))),
    }


def prefixed(prefix: str, metrics: Dict[str, float]) -> Dict[str, float]:
    return {f"{prefix}_{key}": value for key, value in metrics.items()}


def discounted_score(
    *,
    q: np.ndarray,
    ee: np.ndarray,
    desired: np.ndarray,
    previous_q: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    discount: float,
    continuity_weight: float,
    weights: Weights,
) -> float:
    time_weights = np.power(discount, np.arange(q.shape[0], dtype=np.float64))
    time_weights /= np.sum(time_weights)
    distances = np.linalg.norm(ee - desired, axis=1)
    cart = float(np.sum(time_weights * distances))
    max_cart = float(np.max(distances))

    def weighted_square(values: np.ndarray, offset: int) -> float:
        if not values.size:
            return 0.0
        local_weights = time_weights[offset : offset + values.shape[0]]
        local_weights /= max(float(np.sum(local_weights)), EPS)
        return float(np.sum(local_weights * np.mean(np.square(values), axis=1)))

    velocity = np.diff(q, axis=0)
    acceleration = np.diff(q, n=2, axis=0)
    jerk = np.diff(q, n=3, axis=0)
    _, limit_cost = limit_metrics(q, lower, upper)
    continuity = float(np.sum(np.square(q[0] - previous_q)))
    return (
        weights.cart * cart
        + weights.max_cart * max_cart
        + weights.vel * weighted_square(velocity, 1)
        + weights.acc * weighted_square(acceleration, 2)
        + weights.jerk * weighted_square(jerk, 3)
        + weights.limit * limit_cost
        + continuity_weight * continuity
    )


def local_full_score(metrics: Dict[str, float], continuity_weight: float) -> float:
    return float(
        metrics["full_horizon_drawing_cost"]
        + continuity_weight * metrics["full_horizon_continuity_cost"]
    )


class CurrentRankingAdapter:
    """Call the rollout's ranking helper when it is exposed as a function."""

    NAMES = (
        "current_ranking_score",
        "candidate_ranking_score",
        "compute_candidate_ranking_score",
        "compute_ranking_score",
        "ranking_score",
        "ranking_cost",
        "score_candidate",
        "candidate_score",
        "calculate_candidate_score",
        "evaluate_candidate_score",
        "compute_candidate_cost",
        "candidate_cost",
        "candidate_selection_score",
    )

    def __init__(self, module: ModuleType, args: argparse.Namespace) -> None:
        self.module = module
        self.args = args
        self.function: Optional[Callable[..., Any]] = None
        self.source = "fallback_full_horizon_drawing_plus_continuity"
        for name in self.NAMES:
            candidate = getattr(module, name, None)
            if callable(candidate):
                self.function = candidate
                self.source = f"diagnose_warm_start_action_buffer_rollout.{name}"
                break
        self.fallback_used = self.function is None

    def score(self, context: Dict[str, Any]) -> float:
        if self.function is None:
            return local_full_score(context["metrics"], self.args.continuity_weight)
        signature = inspect.signature(self.function)
        aliases: Dict[str, Any] = {
            **context,
            **context["metrics"],
            "candidate_metrics": context["metrics"],
            "full_metrics": context["full_metrics"],
            "prefix_metrics": context["prefix_metrics"],
            "ranking_metrics": context["metrics"],
            "config": self.args,
            "args": self.args,
        }
        kwargs: Dict[str, Any] = {}
        for name, parameter in signature.parameters.items():
            if name in aliases:
                kwargs[name] = aliases[name]
            elif parameter.default is inspect.Parameter.empty:
                self.fallback_used = True
                self.source = (
                    "fallback_full_horizon_drawing_plus_continuity"
                    f" (unsupported required ranking parameter {name!r})"
                )
                return local_full_score(context["metrics"], self.args.continuity_weight)
        try:
            value = self.function(**kwargs)
            if isinstance(value, dict):
                for key in ("score", "ranking_score", "total_cost", "cost"):
                    if key in value:
                        value = value[key]
                        break
            if isinstance(value, (tuple, list)):
                value = value[0]
            result = float(value)
            if not np.isfinite(result):
                raise ValueError("ranking helper returned a non-finite score")
            return result
        except Exception as exc:
            self.fallback_used = True
            self.source = (
                "fallback_full_horizon_drawing_plus_continuity"
                f" (ranking helper failed: {type(exc).__name__})"
            )
            return local_full_score(context["metrics"], self.args.continuity_weight)


def rollout_initialization(
    module: ModuleType,
    horizon: int,
    residual_mean: np.ndarray,
    residual_std: np.ndarray,
) -> Tuple[np.ndarray, str]:
    """Recover an exposed rollout initializer; preserve its normalized-zero fallback."""
    shape = (horizon, JOINT_DIM)
    names = (
        "build_residual_initialization",
        "make_residual_initialization",
        "initial_residual_norm",
        "make_zero_residual_norm",
    )
    aliases: Dict[str, Any] = {
        "horizon": horizon,
        "prediction_horizon": horizon,
        "target_dim": JOINT_DIM,
        "joint_dim": JOINT_DIM,
        "residual_mean": residual_mean,
        "residual_std": residual_std,
        "shape": shape,
    }
    for name in names:
        function = getattr(module, name, None)
        if not callable(function):
            continue
        signature = inspect.signature(function)
        kwargs: Dict[str, Any] = {}
        compatible = True
        for parameter_name, parameter in signature.parameters.items():
            if parameter_name in aliases:
                kwargs[parameter_name] = aliases[parameter_name]
            elif parameter.default is inspect.Parameter.empty:
                compatible = False
                break
        if not compatible:
            continue
        try:
            result = function(**kwargs)
            if isinstance(result, torch.Tensor):
                result = result.detach().cpu().numpy()
            array = np.asarray(result, dtype=np.float32)
            if array.shape == (1, horizon, JOINT_DIM):
                array = array[0]
            elif array.shape == (JOINT_DIM, horizon):
                array = array.T
            elif array.shape == (1, JOINT_DIM, horizon):
                array = array[0].T
            if array.shape == shape and np.all(np.isfinite(array)):
                return array, f"diagnose_warm_start_action_buffer_rollout.{name}"
        except Exception:
            continue
    return np.zeros(shape, dtype=np.float32), "rollout_normalized_zero_fallback"


def array_stats(prefix: str, values: np.ndarray) -> Dict[str, float]:
    array = finite_array(values, prefix).astype(np.float64)
    return {
        f"{prefix}_mean": float(np.mean(array)),
        f"{prefix}_std": float(np.std(array)),
        f"{prefix}_min": float(np.min(array)),
        f"{prefix}_max": float(np.max(array)),
        f"{prefix}_rms": float(np.sqrt(np.mean(np.square(array)))),
    }


def rank_scores(scores: Sequence[float]) -> Tuple[List[int], int]:
    if not scores:
        raise ValueError("Cannot rank an empty candidate set")
    ordering = sorted(
        range(len(scores)),
        key=lambda index: (
            float(scores[index]) if np.isfinite(scores[index]) else float("inf"),
            index,
        ),
    )
    ranks = [0] * len(scores)
    for rank, index in enumerate(ordering, start=1):
        ranks[index] = rank
    if 0 not in ordering or ranks[0] <= 0:
        raise AssertionError("Safety candidate 0 was omitted from a ranking operation")
    return ranks, ordering[0]


def assert_metric_identity(first: Dict[str, float], second: Dict[str, float]) -> None:
    if first.keys() != second.keys():
        raise AssertionError("Candidate 0 and buffer metric keys differ")
    for key in first:
        if not np.isclose(first[key], second[key], rtol=1e-6, atol=1e-8, equal_nan=True):
            raise AssertionError(
                f"Candidate 0 metric {key!r} differs from separately evaluated buffer: "
                f"{first[key]} vs {second[key]}"
            )


def hard_gate_reason(
    row: Dict[str, Any],
    buffer_row: Dict[str, Any],
    args: argparse.Namespace,
) -> str:
    practical_finite_keys = (
        "boundary_joint_l2",
        "boundary_max_abs_joint_step",
        "prefix_mean_cartesian_error",
        "prefix_rms_cartesian_error",
        "prefix_max_cartesian_error",
        "prefix_drawing_cost",
        "prefix_velocity_cost",
        "prefix_acceleration_cost",
        "prefix_jerk_cost",
        "prefix_max_joint_step",
        "prefix_joint_limit_violation_cost",
        "prefix_continuity_cost",
        "full_horizon_mean_cartesian_error",
        "full_horizon_rms_cartesian_error",
        "full_horizon_max_cartesian_error",
        "full_horizon_drawing_cost",
        "full_horizon_velocity_cost",
        "full_horizon_acceleration_cost",
        "full_horizon_jerk_cost",
        "full_horizon_max_joint_step",
        "full_horizon_joint_limit_violation_cost",
        "full_horizon_continuity_cost",
        "current_ranking_score",
        "prefix_score",
        "full_horizon_score",
        "discounted_score",
    )
    numeric_values = [float(row[key]) for key in practical_finite_keys]
    if not np.all(np.isfinite(numeric_values)):
        return "non_finite"
    boundary_limit = min(
        args.allowed_boundary_step,
        max(
            float(buffer_row["boundary_max_abs_joint_step"]) * args.boundary_ratio,
            EPS,
        ),
    )
    prefix_step_limit = min(
        args.allowed_prefix_step,
        max(
            float(buffer_row["prefix_max_joint_step"]) * args.max_step_ratio,
            EPS,
        ),
    )
    if float(row["boundary_max_abs_joint_step"]) > boundary_limit:
        return "boundary_step"
    if float(row["prefix_max_joint_step"]) > prefix_step_limit:
        return "prefix_step"
    if float(row["prefix_joint_limit_violation_count"]) > 0.0:
        return "joint_limit"
    return ""


def write_dict_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    fields: List[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            output: Dict[str, Any] = {}
            for key in fields:
                value = row.get(key, "")
                if isinstance(value, (np.floating, float)):
                    output[key] = f"{float(value):.12e}"
                elif isinstance(value, (np.integer,)):
                    output[key] = int(value)
                else:
                    output[key] = value
            writer.writerow(output)


def aggregate_cycles(
    cycle_rows: Sequence[Dict[str, Any]],
    candidate_rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    groups: List[Tuple[str, Sequence[Dict[str, Any]]]] = [("all", cycle_rows)]
    for path_name in sorted({str(row["path_name"]) for row in cycle_rows}):
        groups.append(
            (
                path_name,
                [row for row in cycle_rows if str(row["path_name"]) == path_name],
            )
        )
    output: List[Dict[str, Any]] = []
    for scope, rows in groups:
        def mean_bool(key: str) -> float:
            return 100.0 * float(np.mean([bool(row[key]) for row in rows]))

        regrets_prefix = np.asarray(
            [float(row["ranking_regret_prefix_cartesian"]) for row in rows],
            dtype=np.float64,
        )
        regrets_drawing = np.asarray(
            [float(row["ranking_regret_prefix_drawing_cost"]) for row in rows],
            dtype=np.float64,
        )
        regrets_full = np.asarray(
            [float(row["ranking_regret_full_horizon"]) for row in rows],
            dtype=np.float64,
        )
        path_candidates = (
            candidate_rows
            if scope == "all"
            else [row for row in candidate_rows if str(row["path_name"]) == scope]
        )
        diffusion = [row for row in path_candidates if row["candidate_type"] == "diffusion_refined"]
        output.append(
            {
                "scope": "all" if scope == "all" else "path",
                "path_name": "" if scope == "all" else scope,
                "cycle_count": len(rows),
                "percent_cycles_prefix_improving_diffusion": mean_bool(
                    "any_diffusion_improved_prefix_cartesian"
                ),
                "percent_cycles_drawing_improving_diffusion": mean_bool(
                    "any_diffusion_improved_prefix_drawing_cost"
                ),
                "percent_cycles_improving_accuracy_and_continuity": mean_bool(
                    "any_candidate_improved_both_with_continuity"
                ),
                "percent_current_selected_best_prefix": mean_bool(
                    "current_selected_best_prefix_cartesian"
                ),
                "percent_current_selected_worse_than_buffer": mean_bool(
                    "current_selected_worse_than_buffer_prefix"
                ),
                "percent_current_selected_unsafe": mean_bool("current_selected_unsafe"),
                "mean_ranking_regret_prefix_cartesian": float(np.mean(regrets_prefix)),
                "median_ranking_regret_prefix_cartesian": float(np.median(regrets_prefix)),
                "mean_ranking_regret_prefix_drawing_cost": float(np.mean(regrets_drawing)),
                "median_ranking_regret_prefix_drawing_cost": float(np.median(regrets_drawing)),
                "mean_ranking_regret_full_horizon": float(np.mean(regrets_full)),
                "median_ranking_regret_full_horizon": float(np.median(regrets_full)),
                "mean_diffusion_residual_rms": float(
                    np.mean([float(row["generated_physical_residual_rms"]) for row in diffusion])
                ) if diffusion else 0.0,
                "mean_condition_fraction_abs_z_gt_3": float(
                    np.mean([float(row["condition_fraction_abs_z_gt_3"]) for row in rows])
                ),
                "mean_condition_fraction_abs_z_gt_5": float(
                    np.mean([float(row["condition_fraction_abs_z_gt_5"]) for row in rows])
                ),
                "mean_condition_fraction_abs_z_gt_10": float(
                    np.mean([float(row["condition_fraction_abs_z_gt_10"]) for row in rows])
                ),
                "mean_condition_max_abs_z": float(
                    np.mean([float(row["condition_max_abs_z"]) for row in rows])
                ),
            }
        )
    return output


def aggregate_condition_ood(
    rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str, int], List[Dict[str, Any]]] = {}
    for row in rows:
        channel = int(row["condition_channel"])
        groups.setdefault(("all", "", channel), []).append(row)
        groups.setdefault(("path", str(row["path_name"]), channel), []).append(row)
    output: List[Dict[str, Any]] = []
    for (scope, path_name, channel), group in sorted(
        groups.items(),
        key=lambda item: (item[0][0], item[0][1], item[0][2]),
    ):
        first = group[0]
        output.append(
            {
                "scope": scope,
                "path_name": path_name,
                "condition_channel": channel,
                "condition_feature": first["condition_feature"],
                "condition_group": first["condition_group"],
                "planning_cycle_count": len(group),
                "mean_abs_z": float(
                    np.mean([float(row["mean_abs_z"]) for row in group])
                ),
                "max_abs_z": float(
                    np.max([float(row["max_abs_z"]) for row in group])
                ),
                "mean_fraction_abs_z_gt_3": float(
                    np.mean(
                        [float(row["fraction_abs_z_gt_3"]) for row in group]
                    )
                ),
                "mean_fraction_abs_z_gt_5": float(
                    np.mean(
                        [float(row["fraction_abs_z_gt_5"]) for row in group]
                    )
                ),
                "mean_fraction_abs_z_gt_10": float(
                    np.mean(
                        [float(row["fraction_abs_z_gt_10"]) for row in group]
                    )
                ),
            }
        )
    return output


def save_plots(
    output_dir: Path,
    candidate_rows: Sequence[Dict[str, Any]],
    cycle_rows: Sequence[Dict[str, Any]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    diffusion = [row for row in candidate_rows if row["candidate_type"] == "diffusion_refined"]

    def scatter(x_key: str, y_key: str, filename: str, xlabel: str, ylabel: str) -> None:
        x = np.asarray([float(row[x_key]) for row in diffusion], dtype=np.float64)
        y = np.asarray([float(row[y_key]) for row in diffusion], dtype=np.float64)
        finite = np.isfinite(x) & np.isfinite(y)
        figure, axis = plt.subplots(figsize=(7, 5))
        axis.scatter(x[finite], y[finite], s=10, alpha=0.35)
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.25)
        figure.tight_layout()
        figure.savefig(plots_dir / filename, dpi=160)
        plt.close(figure)

    scatter(
        "current_ranking_score",
        "prefix_mean_cartesian_error",
        "current_score_vs_prefix_cartesian.png",
        "Current ranking score",
        "Executed-prefix mean Cartesian error",
    )
    scatter(
        "current_ranking_score",
        "prefix_drawing_cost",
        "current_score_vs_prefix_drawing_cost.png",
        "Current ranking score",
        "Executed-prefix drawing cost",
    )
    scatter(
        "prefix_cartesian_improvement_vs_buffer",
        "boundary_max_abs_joint_step",
        "prefix_improvement_vs_boundary_discontinuity.png",
        "Prefix Cartesian improvement vs buffer",
        "Boundary maximum absolute joint step",
    )
    scatter(
        "generated_physical_residual_rms",
        "prefix_cartesian_improvement_vs_buffer",
        "residual_magnitude_vs_cartesian_improvement.png",
        "Generated physical residual RMS",
        "Prefix Cartesian improvement vs buffer",
    )
    scatter(
        "condition_max_abs_z",
        "prefix_cartesian_improvement_vs_buffer",
        "condition_ood_vs_candidate_improvement.png",
        "Condition maximum absolute z-score",
        "Prefix Cartesian improvement vs buffer",
    )

    def histogram(values: Iterable[float], filename: str, xlabel: str) -> None:
        array = np.asarray(list(values), dtype=np.float64)
        array = array[np.isfinite(array)]
        figure, axis = plt.subplots(figsize=(7, 5))
        axis.hist(array, bins=30)
        axis.set_xlabel(xlabel)
        axis.set_ylabel("Count")
        axis.grid(True, alpha=0.25)
        figure.tight_layout()
        figure.savefig(plots_dir / filename, dpi=160)
        plt.close(figure)

    histogram(
        (float(row["ranking_regret_prefix_cartesian"]) for row in cycle_rows),
        "ranking_regret_histogram.png",
        "Current-selection regret: prefix Cartesian error",
    )
    histogram(
        (float(row["useful_prefix_candidate_count"]) for row in cycle_rows),
        "useful_candidates_per_cycle.png",
        "Useful diffusion candidates per cycle",
    )
    histogram(
        (float(row["buffer_rank_current"]) for row in cycle_rows),
        "buffer_rank_distribution.png",
        "Unrefined buffer rank under current score",
    )


def print_conclusion(
    aggregate: Dict[str, Any],
    normalization_mismatch: bool,
    ranking_fallback_used: bool,
) -> None:
    prefix_rate = float(aggregate["percent_cycles_prefix_improving_diffusion"])
    drawing_rate = float(aggregate["percent_cycles_drawing_improving_diffusion"])
    best_rate = float(aggregate["percent_current_selected_best_prefix"])
    worse_rate = float(aggregate["percent_current_selected_worse_than_buffer"])
    unsafe_rate = float(aggregate["percent_current_selected_unsafe"])
    ood_gt3 = 100.0 * float(aggregate["mean_condition_fraction_abs_z_gt_3"])
    ood_gt10 = 100.0 * float(aggregate["mean_condition_fraction_abs_z_gt_10"])
    print("\nFailure classification")
    assigned = False
    if prefix_rate < 25.0 and drawing_rate < 25.0:
        assigned = True
        print(
            "A. Candidate-generation failure: "
            f"prefix-improving cycles={prefix_rate:.1f}%, "
            f"drawing-improving cycles={drawing_rate:.1f}%."
        )
    if prefix_rate > 0.0 and (best_rate < 50.0 or worse_rate > 25.0):
        assigned = True
        suffix = " Ranking fallback was used." if ranking_fallback_used else ""
        print(
            "B. Ranking failure: "
            f"selected-best-prefix={best_rate:.1f}%, "
            f"selected-worse-than-buffer={worse_rate:.1f}%.{suffix}"
        )
    if unsafe_rate > 10.0:
        assigned = True
        print(
            "C. Continuity-gating failure: "
            f"current selector chose a hard-gate-unsafe candidate in {unsafe_rate:.1f}% of cycles."
        )
    if ood_gt3 > 5.0 or ood_gt10 > 1.0:
        assigned = True
        print(
            "D. Conditioning distribution shift: "
            f"|z|>3 rate={ood_gt3:.2f}%, |z|>10 rate={ood_gt10:.2f}%."
        )
    if normalization_mismatch:
        assigned = True
        print(
            "E. Residual normalization or initialization error: rollout initialization "
            "uses normalized zeros although physical zero maps to nonzero normalized values."
        )
    if not assigned:
        print("No category crossed the conservative automatic classification thresholds.")


def main() -> int:
    args = parse_args()
    if args.prediction_horizon <= 0 or args.execution_horizon <= 0:
        raise ValueError("Prediction and execution horizons must be positive")
    if args.execution_horizon > args.prediction_horizon:
        raise ValueError("execution_horizon cannot exceed prediction_horizon")
    if args.num_candidates <= 0 or args.max_paths <= 0:
        raise ValueError("num_candidates and max_paths must be positive")
    if not 0.0 < args.ranking_discount <= 1.0:
        raise ValueError("ranking_discount must be in (0,1]")
    for name in (
        "allowed_boundary_step",
        "allowed_prefix_step",
        "boundary_ratio",
        "max_step_ratio",
    ):
        if float(getattr(args, name)) <= 0.0:
            raise ValueError(f"--{name} must be positive")

    set_seed(args.seed)
    device = resolve_device(args.device)
    warm_rollout = importlib.import_module("diagnose_warm_start_action_buffer_rollout")
    ranking_adapter = CurrentRankingAdapter(warm_rollout, args)
    stats = load_stats(args.stats_npz)
    checkpoint = torch_load_checkpoint(args.checkpoint, device)
    if int(checkpoint["condition_dim"]) != CONDITION_DIM:
        raise ValueError(f"Best v5b checkpoint condition_dim must be 38, got {checkpoint['condition_dim']}")
    if int(checkpoint["target_dim"]) != JOINT_DIM:
        raise ValueError(f"Best v5b checkpoint target_dim must be 6, got {checkpoint['target_dim']}")
    if int(checkpoint["horizon"]) != args.prediction_horizon:
        raise ValueError(
            f"Checkpoint horizon {checkpoint['horizon']} differs from prediction_horizon "
            f"{args.prediction_horizon}"
        )
    model, call_variant, _ = instantiate_checkpoint_model(checkpoint, device)
    model.eval()
    diffusion_config = diffusion_config_from_checkpoint(checkpoint, args.num_diffusion_steps)
    num_steps = int(diffusion_config["num_steps"])
    if not 0 <= args.t_init < num_steps:
        raise ValueError(f"t_init must be in [0,{num_steps - 1}]")
    schedule = make_schedule(
        num_steps,
        float(diffusion_config["beta_start"]),
        float(diffusion_config["beta_end"]),
        device,
    )

    data = load_npz(args.test_npz, "test trajectories")
    require_keys(data, ("desired_paths", "expert_q", "q_start", "path_names"), "test trajectories")
    desired_paths = finite_array(data["desired_paths"], "desired_paths").astype(np.float32)
    expert_q = finite_array(data["expert_q"], "expert_q").astype(np.float32)
    q_start_all = finite_array(data["q_start"], "q_start").astype(np.float32)
    path_names = decode_names(data["path_names"])
    if desired_paths.ndim != 3 or desired_paths.shape[-1] != 3:
        raise ValueError("desired_paths must have shape (N,T,3)")
    if expert_q.shape != (desired_paths.shape[0], desired_paths.shape[1], JOINT_DIM):
        raise ValueError("expert_q must have shape (N,T,6) matching desired_paths")
    if q_start_all.shape != (desired_paths.shape[0], JOINT_DIM):
        raise ValueError("q_start must have shape (N,6)")
    if len(path_names) != desired_paths.shape[0] or len(set(path_names)) != len(path_names):
        raise ValueError("Test path_names must be unique and match N")

    robot, joint_names, ee_link = load_fk_context(None, None)
    if len(joint_names) != JOINT_DIM:
        raise ValueError("Exactly six active xMateCR7 joints are required")
    lower, upper = extract_joint_limits(robot, joint_names)
    weights = default_weights()
    residual_mean = stats["residual_mean"]
    residual_std = stats["residual_std"]
    physical_zero = np.zeros((args.prediction_horizon, JOINT_DIM), dtype=np.float32)
    normalized_physical_zero = (
        (physical_zero - residual_mean[None, :]) / residual_std[None, :]
    ).astype(np.float32)
    actual_initialization, initialization_source = rollout_initialization(
        warm_rollout,
        args.prediction_horizon,
        residual_mean,
        residual_std,
    )
    actual_initialization = finite_array(actual_initialization, "rollout residual initialization").astype(np.float32)
    denormalized_initialization = (
        actual_initialization * residual_std[None, :] + residual_mean[None, :]
    ).astype(np.float32)
    uses_normalized_zero = bool(np.allclose(actual_initialization, 0.0, atol=1e-8))
    physical_zero_is_nonzero_norm = bool(
        not np.allclose(normalized_physical_zero, 0.0, atol=1e-8)
    )
    normalization_mismatch = uses_normalized_zero and physical_zero_is_nonzero_norm
    if normalization_mismatch:
        print(
            "WARNING: rollout initialization is normalized zero, but physical zero "
            "has nonzero normalized coordinates because residual_mean is nonzero."
        )
    initialization_audit = {
        "residual_mean_per_joint": json.dumps(residual_mean.astype(float).tolist()),
        "residual_std_per_joint": json.dumps(residual_std.astype(float).tolist()),
        "normalized_physical_zero_per_joint": json.dumps(
            normalized_physical_zero[0].astype(float).tolist()
        ),
        "initialization_source": initialization_source,
        "normalized_zero_physical_zero_mismatch": int(normalization_mismatch),
        **array_stats("actual_normalized_initialization", actual_initialization),
        **array_stats("denormalized_initialization", denormalized_initialization),
    }

    candidate_rows: List[Dict[str, Any]] = []
    cycle_rows: List[Dict[str, Any]] = []
    ood_rows: List[Dict[str, Any]] = []
    max_paths = min(args.max_paths, len(path_names))
    with torch.no_grad():
        for path_index in range(max_paths):
            path_name = path_names[path_index]
            desired_path = desired_paths[path_index]
            desired_delta = desired_differences(desired_path)
            num_trajectory_steps = desired_path.shape[0]
            prior_q = read_predicted_q_csv(
                args.prior_dir / safe_path_name(path_name) / "predicted_q.csv",
                expected_steps=num_trajectory_steps,
            )
            finite_array(prior_q, f"{path_name} MLP prior")
            for cycle_index, start in enumerate(
                range(0, num_trajectory_steps, args.execution_horizon)
            ):
                buffer_q, indices = build_teacher_forced_buffer(
                    prior_q,
                    start,
                    args.prediction_horizon,
                )
                previous_q = q_start_all[path_index] if start == 0 else prior_q[start - 1]
                condition, _ = build_v5b_condition(
                    desired_path=desired_path,
                    desired_delta=desired_delta,
                    indices=indices,
                    q_start=q_start_all[path_index],
                    current_q=previous_q,
                    buffer_q=buffer_q,
                    robot=robot,
                    joint_names=joint_names,
                    ee_link=ee_link,
                )
                condition_norm, cycle_ood_rows, ood_summary = condition_ood_rows(
                    path_name=path_name,
                    cycle_index=cycle_index,
                    start_index=start,
                    condition=condition,
                    mean=stats["condition_mean"],
                    std=stats["condition_std"],
                )
                ood_rows.extend(cycle_ood_rows)
                execution_count = min(
                    args.execution_horizon,
                    num_trajectory_steps - start,
                )
                desired_window = desired_path[indices]
                expert_window = expert_q[path_index, indices]

                cycle_seed = args.seed + path_index * 100_000 + cycle_index * 1_000
                set_seed(cycle_seed)
                repeated_condition = np.repeat(
                    condition_norm[None, :, :],
                    args.num_candidates,
                    axis=0,
                )
                repeated_initialization = np.repeat(
                    actual_initialization[None, :, :],
                    args.num_candidates,
                    axis=0,
                )
                generated_norm = reverse_noised_x0_batches(
                    model=model,
                    call_variant=call_variant,
                    condition_bhc=repeated_condition,
                    x0_norm_bhc=repeated_initialization,
                    t_init=args.t_init,
                    schedule=schedule,
                    batch_size=args.num_candidates,
                    device=device,
                    deterministic=False,
                )
                finite_array(generated_norm, "generated normalized residuals")
                generated_physical = (
                    generated_norm * residual_std[None, None, :]
                    + residual_mean[None, None, :]
                ).astype(np.float32)
                finite_array(generated_physical, "generated physical residuals")
                candidates = [buffer_q.copy()]
                candidates.extend(
                    (buffer_q + generated_physical[index]).astype(np.float32)
                    for index in range(args.num_candidates)
                )
                if len(candidates) != args.num_candidates + 1:
                    raise AssertionError("Candidate set size is inconsistent")
                if not np.array_equal(candidates[0], buffer_q):
                    raise AssertionError("Candidate index 0 is not exactly the unrefined buffer")

                rows_this_cycle: List[Dict[str, Any]] = []
                separately_evaluated_buffer: Optional[Dict[str, float]] = None
                for candidate_index, candidate_q in enumerate(candidates):
                    finite_array(candidate_q, "candidate trajectory")
                    candidate_ee = fk_positions(
                        robot, joint_names, ee_link, candidate_q
                    )
                    prefix_metrics = evaluate_region(
                        q=candidate_q[:execution_count],
                        desired=desired_window[:execution_count],
                        ee=candidate_ee[:execution_count],
                        previous_q=previous_q,
                        lower=lower,
                        upper=upper,
                        weights=weights,
                    )
                    full_metrics = evaluate_region(
                        q=candidate_q,
                        desired=desired_window,
                        ee=candidate_ee,
                        previous_q=previous_q,
                        lower=lower,
                        upper=upper,
                        weights=weights,
                    )
                    metrics = {
                        **prefixed("prefix", prefix_metrics),
                        **prefixed("full_horizon", full_metrics),
                    }
                    boundary_delta = candidate_q[0] - previous_q
                    metrics.update(
                        {
                            "boundary_joint_l2": float(np.linalg.norm(boundary_delta)),
                            "boundary_max_abs_joint_step": float(
                                np.max(np.abs(boundary_delta))
                            ),
                        }
                    )
                    if candidate_index == 0:
                        buffer_ee = fk_positions(
                            robot, joint_names, ee_link, buffer_q.copy()
                        )
                        separate_prefix = evaluate_region(
                            q=buffer_q[:execution_count].copy(),
                            desired=desired_window[:execution_count].copy(),
                            ee=buffer_ee[:execution_count],
                            previous_q=previous_q.copy(),
                            lower=lower,
                            upper=upper,
                            weights=weights,
                        )
                        separate_full = evaluate_region(
                            q=buffer_q.copy(),
                            desired=desired_window.copy(),
                            ee=buffer_ee,
                            previous_q=previous_q.copy(),
                            lower=lower,
                            upper=upper,
                            weights=weights,
                        )
                        separately_evaluated_buffer = {
                            **prefixed("prefix", separate_prefix),
                            **prefixed("full_horizon", separate_full),
                            "boundary_joint_l2": float(
                                np.linalg.norm(buffer_q[0] - previous_q)
                            ),
                            "boundary_max_abs_joint_step": float(
                                np.max(np.abs(buffer_q[0] - previous_q))
                            ),
                        }
                        assert_metric_identity(metrics, separately_evaluated_buffer)

                    residual_norm = (
                        normalized_physical_zero
                        if candidate_index == 0
                        else generated_norm[candidate_index - 1]
                    )
                    residual_physical = (
                        physical_zero
                        if candidate_index == 0
                        else generated_physical[candidate_index - 1]
                    )
                    displacement = candidate_q - buffer_q
                    oracle_prefix_rmse = float(
                        np.sqrt(
                            np.mean(
                                np.square(
                                    candidate_q[:execution_count]
                                    - expert_window[:execution_count]
                                )
                            )
                        )
                    )
                    oracle_full_rmse = float(
                        np.sqrt(np.mean(np.square(candidate_q - expert_window)))
                    )
                    row: Dict[str, Any] = {
                        "path_name": path_name,
                        "planning_cycle_index": cycle_index,
                        "trajectory_start_index": start,
                        "execution_count": execution_count,
                        "candidate_index": candidate_index,
                        "candidate_type": (
                            "unrefined_buffer"
                            if candidate_index == 0
                            else "diffusion_refined"
                        ),
                        "candidate_seed": (
                            cycle_seed if candidate_index == 0 else cycle_seed + candidate_index
                        ),
                        **metrics,
                        **initialization_audit,
                        **array_stats("generated_normalized_residual", residual_norm),
                        **array_stats("generated_physical_residual", residual_physical),
                        **array_stats("candidate_joint_displacement", displacement),
                        **ood_summary,
                        "oracle_prefix_joint_rmse": oracle_prefix_rmse,
                        "oracle_full_horizon_joint_rmse": oracle_full_rmse,
                    }
                    row["prefix_score"] = float(
                        row["prefix_drawing_cost"]
                        + args.continuity_weight * row["prefix_continuity_cost"]
                    )
                    row["full_horizon_score"] = local_full_score(
                        row, args.continuity_weight
                    )
                    row["discounted_score"] = discounted_score(
                        q=candidate_q,
                        ee=candidate_ee,
                        desired=desired_window,
                        previous_q=previous_q,
                        lower=lower,
                        upper=upper,
                        discount=args.ranking_discount,
                        continuity_weight=args.continuity_weight,
                        weights=weights,
                    )
                    row["current_ranking_score"] = ranking_adapter.score(
                        {
                            "metrics": {
                                key: value
                                for key, value in row.items()
                                if not key.startswith("oracle_")
                            },
                            "prefix_metrics": prefix_metrics,
                            "full_metrics": full_metrics,
                            "candidate_q": candidate_q,
                            "q": candidate_q,
                            "desired_path": desired_window,
                            "desired": desired_window,
                            "previous_q": previous_q,
                            "condition": condition_norm,
                            "weights": weights,
                        }
                    )
                    rows_this_cycle.append(row)

                if separately_evaluated_buffer is None:
                    raise AssertionError("Safety buffer was not independently evaluated")
                buffer_row = rows_this_cycle[0]
                for row in rows_this_cycle:
                    reason = hard_gate_reason(row, buffer_row, args)
                    row["hard_gate_rejection_reason"] = reason
                    row["hard_gate_passed"] = int(reason == "")
                    row["hard_gated_prefix_score"] = (
                        float(row["prefix_score"])
                        if reason == ""
                        else float("inf")
                    )
                    row["prefix_cartesian_improvement_vs_buffer"] = float(
                        buffer_row["prefix_mean_cartesian_error"]
                        - row["prefix_mean_cartesian_error"]
                    )
                    row["prefix_drawing_improvement_vs_buffer"] = float(
                        buffer_row["prefix_drawing_cost"]
                        - row["prefix_drawing_cost"]
                    )

                ranking_specs = {
                    "current": "current_ranking_score",
                    "prefix": "prefix_score",
                    "full_horizon": "full_horizon_score",
                    "discounted": "discounted_score",
                    "hard_gated_prefix": "hard_gated_prefix_score",
                    "prefix_cartesian": "prefix_mean_cartesian_error",
                    "prefix_drawing": "prefix_drawing_cost",
                    "oracle_prefix_joint": "oracle_prefix_joint_rmse",
                }
                selected: Dict[str, int] = {}
                for method, score_key in ranking_specs.items():
                    scores = [float(row[score_key]) for row in rows_this_cycle]
                    ranks, selected_index = rank_scores(scores)
                    selected[method] = selected_index
                    for index, row in enumerate(rows_this_cycle):
                        row[f"rank_{method}"] = ranks[index]
                        row[f"selected_{method}"] = int(index == selected_index)
                if any(int(row["candidate_index"]) != index for index, row in enumerate(rows_this_cycle)):
                    raise AssertionError("Candidate indices changed before ranking")

                current_index = selected["current"]
                current_row = rows_this_cycle[current_index]
                best_prefix_index = selected["prefix_cartesian"]
                best_drawing_index = selected["prefix_drawing"]
                best_full_index = selected["full_horizon"]
                oracle_index = selected["oracle_prefix_joint"]
                for index, row in enumerate(rows_this_cycle):
                    row["oracle_best_prefix_joint_rmse"] = int(index == oracle_index)
                    row["current_ranking_source"] = ranking_adapter.source

                diffusion_rows = rows_this_cycle[1:]
                any_prefix = any(
                    float(row["prefix_mean_cartesian_error"])
                    < float(buffer_row["prefix_mean_cartesian_error"])
                    for row in diffusion_rows
                )
                any_drawing = any(
                    float(row["prefix_drawing_cost"])
                    < float(buffer_row["prefix_drawing_cost"])
                    for row in diffusion_rows
                )
                useful_rows = [
                    row
                    for row in diffusion_rows
                    if float(row["prefix_mean_cartesian_error"])
                    < float(buffer_row["prefix_mean_cartesian_error"])
                ]
                both_safe = [
                    row
                    for row in diffusion_rows
                    if float(row["prefix_mean_cartesian_error"])
                    < float(buffer_row["prefix_mean_cartesian_error"])
                    and float(row["prefix_drawing_cost"])
                    < float(buffer_row["prefix_drawing_cost"])
                    and int(row["hard_gate_passed"]) == 1
                ]
                any_oracle = any(
                    float(row["oracle_prefix_joint_rmse"])
                    < float(buffer_row["oracle_prefix_joint_rmse"])
                    for row in diffusion_rows
                )
                cycle_row: Dict[str, Any] = {
                    "path_name": path_name,
                    "planning_cycle_index": cycle_index,
                    "trajectory_start_index": start,
                    "execution_count": execution_count,
                    "candidate_count_including_buffer": len(rows_this_cycle),
                    "diffusion_candidate_count": args.num_candidates,
                    "useful_prefix_candidate_count": len(useful_rows),
                    "useful_both_safe_candidate_count": len(both_safe),
                    "any_diffusion_improved_prefix_cartesian": int(any_prefix),
                    "any_diffusion_improved_prefix_drawing_cost": int(any_drawing),
                    "any_candidate_improved_both_with_continuity": int(bool(both_safe)),
                    "current_selected_useful_both_candidate": int(current_row in both_safe),
                    "current_selected_best_prefix_cartesian": int(current_index == best_prefix_index),
                    "current_selected_best_prefix_drawing": int(current_index == best_drawing_index),
                    "current_selected_best_full_horizon": int(current_index == best_full_index),
                    "current_selected_worse_than_buffer_prefix": int(
                        float(current_row["prefix_mean_cartesian_error"])
                        > float(buffer_row["prefix_mean_cartesian_error"])
                    ),
                    "current_selected_unsafe": int(current_row["hard_gate_passed"] == 0),
                    "current_selected_candidate_index": current_index,
                    "current_selected_candidate_type": current_row["candidate_type"],
                    "best_prefix_candidate_index": best_prefix_index,
                    "best_prefix_drawing_candidate_index": best_drawing_index,
                    "best_full_horizon_candidate_index": best_full_index,
                    "ranking_regret_prefix_cartesian": float(
                        current_row["prefix_mean_cartesian_error"]
                        - rows_this_cycle[best_prefix_index]["prefix_mean_cartesian_error"]
                    ),
                    "ranking_regret_prefix_drawing_cost": float(
                        current_row["prefix_drawing_cost"]
                        - rows_this_cycle[best_drawing_index]["prefix_drawing_cost"]
                    ),
                    "ranking_regret_full_horizon": float(
                        current_row["full_horizon_score"]
                        - rows_this_cycle[best_full_index]["full_horizon_score"]
                    ),
                    "buffer_rank_current": int(buffer_row["rank_current"]),
                    "buffer_rank_prefix": int(buffer_row["rank_prefix"]),
                    "buffer_rank_full_horizon": int(buffer_row["rank_full_horizon"]),
                    "buffer_rank_discounted": int(buffer_row["rank_discounted"]),
                    "buffer_rank_hard_gated_prefix": int(
                        buffer_row["rank_hard_gated_prefix"]
                    ),
                    "oracle_best_candidate_index": oracle_index,
                    "oracle_any_diffusion_closer_than_buffer": int(any_oracle),
                    "mean_diffusion_residual_rms": float(
                        np.mean(
                            [
                                float(row["generated_physical_residual_rms"])
                                for row in diffusion_rows
                            ]
                        )
                    ),
                    **ood_summary,
                    **initialization_audit,
                    "current_ranking_source": ranking_adapter.source,
                }
                cycle_rows.append(cycle_row)
                candidate_rows.extend(rows_this_cycle)

            print(f"Audited {path_index + 1}/{max_paths} paths: {path_name}")

    if not candidate_rows or not cycle_rows or not ood_rows:
        raise RuntimeError("Diagnostic produced no rows")
    aggregate_rows = aggregate_cycles(cycle_rows, candidate_rows)
    ood_aggregate_rows = aggregate_condition_ood(ood_rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_dict_csv(args.output_dir / "candidate_details.csv", candidate_rows)
    write_dict_csv(args.output_dir / "cycle_summary.csv", cycle_rows)
    write_dict_csv(args.output_dir / "aggregate_summary.csv", aggregate_rows)
    write_dict_csv(args.output_dir / "condition_feature_ood.csv", ood_rows)
    write_dict_csv(
        args.output_dir / "condition_feature_ood_aggregate.csv",
        ood_aggregate_rows,
    )
    save_plots(args.output_dir, candidate_rows, cycle_rows)

    print(f"Saved candidate details: {args.output_dir / 'candidate_details.csv'}")
    print(f"Saved cycle summary: {args.output_dir / 'cycle_summary.csv'}")
    print(f"Saved aggregate summary: {args.output_dir / 'aggregate_summary.csv'}")
    print(f"Saved condition feature OOD audit: {args.output_dir / 'condition_feature_ood.csv'}")
    print(
        "Saved condition feature OOD aggregate: "
        f"{args.output_dir / 'condition_feature_ood_aggregate.csv'}"
    )
    print(f"Saved plots: {args.output_dir / 'plots'}")
    print(f"Current ranking source: {ranking_adapter.source}")
    if ranking_adapter.fallback_used:
        print(
            "WARNING: the rollout did not expose a compatible ranking helper; "
            "current score used the explicitly recorded full-horizon fallback."
        )
    print_conclusion(
        aggregate_rows[0],
        normalization_mismatch,
        ranking_adapter.fallback_used,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
