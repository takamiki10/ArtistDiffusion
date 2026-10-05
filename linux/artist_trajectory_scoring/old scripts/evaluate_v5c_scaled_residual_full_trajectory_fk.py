#!/usr/bin/env python3
"""Evaluate stitched v5c scaled-residual trajectories with FK drawing costs.

Predicted residual windows are denormalized, scaled in joint space, added to
their MLP prior windows, and averaged over overlapping timesteps. This script
does not train a model or run diffusion refinement.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

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
from diagnose_residual_predictor_v5c_alpha_sweep import (
    instantiate_model,
    load_checkpoint,
    load_npz,
    load_residual_stats,
    predict_all,
    resolve_device,
    set_seed,
    validate_test_data,
)
from evaluate_prior_refinement_fk_robot_costs import (
    Weights,
    drawing_fidelity_metrics,
    drawing_total_cost,
    shape_path_metrics,
    shape_total_cost,
    smoothness_costs,
    total_cost,
)


DEFAULT_PREDICTOR_CHECKPOINT = Path(
    "data/cartesian_expert_dataset_v3/"
    "residual_window_predictor_v5c_fk_condition/best_checkpoint.pt"
)
DEFAULT_WINDOWS_DIR = Path(
    "data/cartesian_expert_dataset_v3/"
    "diffusion_v5b_residual_windows_fk_condition"
)
DEFAULT_TEST_WINDOWS_NPZ = DEFAULT_WINDOWS_DIR / "test_windows.npz"
DEFAULT_STATS_NPZ = DEFAULT_WINDOWS_DIR / "normalization_stats.npz"
DEFAULT_TEST_NPZ = Path(
    "data/cartesian_expert_dataset_v3/"
    "diffusion_v2/diffusion_test_v2.npz"
)
DEFAULT_PRIOR_DIR = Path(
    "data/cartesian_expert_dataset_v3/mlp_v3_test_predictions"
)
DEFAULT_OUTPUT_DIR = Path(
    "data/cartesian_expert_dataset_v3/"
    "residual_window_predictor_v5c_fk_condition/"
    "full_trajectory_fk_evaluation"
)
DEFAULT_ALPHAS = ("0.05", "0.1", "0.25")
TARGET_DIM = 6
EPS = 1e-12

METRIC_FIELDS = (
    "mean_cartesian_error",
    "max_cartesian_error",
    "rmse_cartesian_error",
    "joint_rmse_to_expert",
    "mean_joint_velocity",
    "max_joint_velocity",
    "mean_joint_acceleration",
    "max_joint_acceleration",
    "mean_joint_jerk",
    "max_joint_jerk",
    "max_joint_step",
    "start_error",
    "end_error",
    "path_length_pred",
    "path_length_desired",
    "path_length_ratio",
    "frechet_distance",
    "dtw_distance",
    "tangent_cosine_error",
    "tangent_weighted_error",
    "progress_error",
    "length_ratio_error",
    "normalized_shape_error",
    "joint_velocity_cost",
    "joint_acceleration_cost",
    "joint_jerk_cost",
    "joint_limit_violation",
    "total_cost",
    "shape_total_cost",
    "drawing_total_cost",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate stitched v5c scaled-residual trajectories with FK."
    )
    parser.add_argument(
        "--predictor_checkpoint",
        type=Path,
        default=DEFAULT_PREDICTOR_CHECKPOINT,
    )
    parser.add_argument(
        "--test_windows_npz",
        type=Path,
        default=DEFAULT_TEST_WINDOWS_NPZ,
    )
    parser.add_argument("--stats_npz", type=Path, default=DEFAULT_STATS_NPZ)
    parser.add_argument("--test_npz", type=Path, default=DEFAULT_TEST_NPZ)
    parser.add_argument("--prior_dir", type=Path, default=DEFAULT_PRIOR_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--alphas",
        nargs="+",
        default=DEFAULT_ALPHAS,
        help=(
            "Positive residual scales. Accepts space-separated values "
            "(--alphas 0.05 0.1 0.25) or comma-separated values."
        ),
    )
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def parse_alphas(raw_values: Sequence[str]) -> List[float]:
    alphas: List[float] = []
    for raw in raw_values:
        for item in raw.split(","):
            item = item.strip()
            if not item:
                continue
            alpha = float(item)
            if not np.isfinite(alpha) or alpha <= 0.0:
                raise ValueError(
                    f"Alpha values must be finite and positive, got {item!r}"
                )
            if alpha in alphas:
                raise ValueError(f"Duplicate alpha value: {alpha}")
            alphas.append(alpha)
    if not alphas:
        raise ValueError("--alphas must contain at least one value")
    return alphas


def decode_names(values: np.ndarray) -> List[str]:
    names: List[str] = []
    for value in np.asarray(values).reshape(-1):
        if isinstance(value, bytes):
            names.append(value.decode("utf-8", errors="replace"))
        else:
            names.append(str(value))
    return names


def require_keys(
    data: Dict[str, np.ndarray],
    keys: Sequence[str],
    label: str,
) -> None:
    missing = [key for key in keys if key not in data]
    if missing:
        raise KeyError(f"{label} missing required key(s): {', '.join(missing)}")


def validate_original_test(
    data: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    require_keys(data, ("expert_q", "desired_paths", "path_names"), "test NPZ")
    expert_q = np.asarray(data["expert_q"], dtype=np.float32)
    desired_paths = np.asarray(data["desired_paths"], dtype=np.float32)
    names = decode_names(data["path_names"])

    if expert_q.ndim != 3 or expert_q.shape[-1] != TARGET_DIM:
        raise ValueError(
            f"expert_q must have shape (N,T,{TARGET_DIM}), got {expert_q.shape}"
        )
    if desired_paths.shape != (expert_q.shape[0], expert_q.shape[1], 3):
        raise ValueError(
            "desired_paths must match expert_q in N,T and have xyz features; "
            f"got {desired_paths.shape} and {expert_q.shape}"
        )
    if len(names) != expert_q.shape[0]:
        raise ValueError(
            f"path_names length {len(names)} does not match N={expert_q.shape[0]}"
        )
    if len(set(names)) != len(names):
        raise ValueError("Original test path_names must be unique")
    safe_names = [safe_path_name(name) for name in names]
    if len(set(safe_names)) != len(safe_names):
        raise ValueError("Original test path_names collide after sanitization")
    if not np.all(np.isfinite(expert_q)) or not np.all(np.isfinite(desired_paths)):
        raise ValueError("Original test NPZ contains non-finite trajectory values")
    return expert_q, desired_paths, names


def trajectory_times(
    data: Dict[str, np.ndarray],
    path_index: int,
    num_steps: int,
) -> np.ndarray:
    if "times" not in data:
        return np.linspace(0.0, 1.0, num_steps, dtype=np.float64)
    times = np.asarray(data["times"], dtype=np.float64)
    if times.shape == (num_steps,):
        return times
    if times.shape == (len(np.asarray(data["path_names"])), num_steps):
        return times[path_index]
    raise ValueError(
        f"times must have shape ({num_steps},) or (N,{num_steps}), got {times.shape}"
    )


def validate_stats_and_layout(
    stats_path: Path,
    checkpoint: Dict[str, Any],
    condition: np.ndarray,
    starts: np.ndarray,
) -> Tuple[int, int]:
    stats = load_npz(stats_path, "normalization stats")
    require_keys(
        stats,
        ("horizon", "stride", "condition_dim", "residual_dim"),
        "normalization stats",
    )
    horizon = int(np.asarray(stats["horizon"]).item())
    stride = int(np.asarray(stats["stride"]).item())
    condition_dim = int(np.asarray(stats["condition_dim"]).item())
    residual_dim = int(np.asarray(stats["residual_dim"]).item())

    if horizon != int(checkpoint["horizon"]):
        raise ValueError(
            f"Stats horizon {horizon} differs from checkpoint {checkpoint['horizon']}"
        )
    if condition_dim != int(checkpoint["condition_dim"]):
        raise ValueError(
            "Stats condition_dim differs from checkpoint: "
            f"{condition_dim} vs {checkpoint['condition_dim']}"
        )
    if residual_dim != int(checkpoint["target_dim"]) or residual_dim != TARGET_DIM:
        raise ValueError(
            "Stats residual_dim differs from checkpoint/expected target: "
            f"{residual_dim}, {checkpoint['target_dim']}, {TARGET_DIM}"
        )
    if condition.shape[1:] != (horizon, condition_dim):
        raise ValueError(
            f"Window condition shape {condition.shape[1:]} differs from stats "
            f"{(horizon, condition_dim)}"
        )
    if stride <= 0:
        raise ValueError(f"Stats stride must be positive, got {stride}")
    if np.any(starts < 0):
        raise ValueError("window_start_indices contains a negative start")
    return horizon, stride


def verify_checkpoint_stats(
    checkpoint: Dict[str, Any],
    residual_mean: np.ndarray,
    residual_std: np.ndarray,
) -> None:
    for key, expected in (
        ("residual_mean", residual_mean),
        ("residual_std", residual_std),
    ):
        if key not in checkpoint:
            continue
        actual = np.asarray(checkpoint[key], dtype=np.float32)
        if actual.shape != expected.shape or not np.allclose(
            actual,
            expected,
            rtol=1e-5,
            atol=1e-7,
        ):
            raise ValueError(f"Checkpoint {key} does not match stats NPZ")


def alpha_label(alpha: float) -> str:
    return f"{float(alpha):.12g}"


def expected_starts(num_steps: int, horizon: int, stride: int) -> List[int]:
    if horizon > num_steps:
        raise ValueError(
            f"Window horizon {horizon} exceeds trajectory length {num_steps}"
        )
    return list(range(0, num_steps - horizon + 1, stride))


def collect_path_windows(
    window_names: Sequence[str],
    window_starts: np.ndarray,
    original_names: Sequence[str],
    num_steps: int,
    horizon: int,
    stride: int,
) -> Dict[str, List[int]]:
    original_set = set(original_names)
    groups: Dict[str, List[int]] = {name: [] for name in original_names}
    for index, name in enumerate(window_names):
        if name not in original_set:
            raise ValueError(f"Window dataset contains unknown path_name {name!r}")
        groups[name].append(index)

    required_starts = expected_starts(num_steps, horizon, stride)
    for name in original_names:
        indices = groups[name]
        actual_starts = [int(window_starts[index]) for index in indices]
        if len(actual_starts) != len(set(actual_starts)):
            raise ValueError(f"{name}: duplicate window start indices")
        if sorted(actual_starts) != required_starts:
            raise ValueError(
                f"{name}: expected starts {required_starts}, got "
                f"{sorted(actual_starts)}"
            )
        groups[name] = sorted(indices, key=lambda index: int(window_starts[index]))
    return groups


def stitch_candidate(
    *,
    path_name: str,
    indices: Sequence[int],
    starts: np.ndarray,
    prior_windows: np.ndarray,
    predicted_residual_q: np.ndarray,
    full_prior_q: np.ndarray,
    alpha: float,
) -> Tuple[np.ndarray, np.ndarray]:
    num_steps = full_prior_q.shape[0]
    horizon = prior_windows.shape[1]
    accumulated = np.zeros_like(full_prior_q, dtype=np.float64)
    coverage = np.zeros(num_steps, dtype=np.int64)
    prior_accumulated = np.zeros_like(full_prior_q, dtype=np.float64)

    for index in indices:
        start = int(starts[index])
        end = start + horizon
        if end > num_steps:
            raise ValueError(
                f"{path_name}: window [{start}:{end}] exceeds T={num_steps}"
            )
        prior_window = np.asarray(prior_windows[index], dtype=np.float64)
        expected_prior = np.asarray(full_prior_q[start:end], dtype=np.float64)
        if not np.allclose(
            prior_window,
            expected_prior,
            rtol=1e-5,
            atol=1e-6,
        ):
            max_error = float(np.max(np.abs(prior_window - expected_prior)))
            raise ValueError(
                f"{path_name}: stored prior window at start {start} differs "
                f"from MLP predicted_q.csv; max error={max_error:.12e}"
            )
        candidate_window = (
            prior_window
            + float(alpha) * np.asarray(predicted_residual_q[index], dtype=np.float64)
        )
        accumulated[start:end] += candidate_window
        prior_accumulated[start:end] += prior_window
        coverage[start:end] += 1

    if np.any(coverage == 0):
        missing = np.flatnonzero(coverage == 0).tolist()
        raise ValueError(f"{path_name}: uncovered timesteps after stitching: {missing}")
    candidate_q = accumulated / coverage[:, None]
    stitched_prior = prior_accumulated / coverage[:, None]
    if not np.allclose(
        stitched_prior,
        full_prior_q,
        rtol=1e-5,
        atol=1e-6,
    ):
        max_error = float(np.max(np.abs(stitched_prior - full_prior_q)))
        raise RuntimeError(
            f"{path_name}: stitched prior does not reconstruct the MLP prior; "
            f"max error={max_error:.12e}"
        )
    return candidate_q.astype(np.float32), coverage


def extract_joint_limits(
    robot: Any,
    joint_names: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray]:
    joint_objects = getattr(getattr(robot, "robot", robot), "joints", [])
    by_name = {
        str(getattr(joint, "name", "")): joint
        for joint in joint_objects
    }
    lower: List[float] = []
    upper: List[float] = []
    for name in joint_names:
        joint = by_name.get(str(name))
        limit = None if joint is None else getattr(joint, "limit", None)
        lower.append(
            float(getattr(limit, "lower", -np.inf))
            if limit is not None
            else -np.inf
        )
        upper.append(
            float(getattr(limit, "upper", np.inf))
            if limit is not None
            else np.inf
        )
    return np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)


def joint_limit_violation(
    q: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> float:
    lower_row = lower.reshape(1, TARGET_DIM)
    upper_row = upper.reshape(1, TARGET_DIM)
    below = np.where(
        np.isfinite(lower_row),
        np.maximum(lower_row - q, 0.0),
        0.0,
    )
    above = np.where(
        np.isfinite(upper_row),
        np.maximum(q - upper_row, 0.0),
        0.0,
    )
    return float(np.mean(np.square(below + above)))


def derivative_metrics(q: np.ndarray) -> Dict[str, float]:
    velocity = np.diff(q, axis=0)
    acceleration = np.diff(q, n=2, axis=0)
    jerk = np.diff(q, n=3, axis=0)

    def mean_and_max_norm(values: np.ndarray) -> Tuple[float, float]:
        if values.size == 0:
            return 0.0, 0.0
        norms = np.linalg.norm(values, axis=1)
        return float(np.mean(norms)), float(np.max(norms))

    mean_vel, max_vel = mean_and_max_norm(velocity)
    mean_acc, max_acc = mean_and_max_norm(acceleration)
    mean_jerk, max_jerk = mean_and_max_norm(jerk)
    max_joint_step = (
        float(np.max(np.abs(velocity)))
        if velocity.size
        else 0.0
    )
    return {
        "mean_joint_velocity": mean_vel,
        "max_joint_velocity": max_vel,
        "mean_joint_acceleration": mean_acc,
        "max_joint_acceleration": max_acc,
        "mean_joint_jerk": mean_jerk,
        "max_joint_jerk": max_jerk,
        "max_joint_step": max_joint_step,
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


def evaluate_trajectory(
    *,
    q: np.ndarray,
    expert_q: np.ndarray,
    desired_path: np.ndarray,
    ee: np.ndarray,
    lower_limits: np.ndarray,
    upper_limits: np.ndarray,
    weights: Weights,
) -> Dict[str, float]:
    cartesian_error = ee - desired_path
    cartesian_distances = np.linalg.norm(cartesian_error, axis=1)
    mean_cartesian_error = float(np.mean(cartesian_distances))
    max_cartesian_error = float(np.max(cartesian_distances))
    rmse_cartesian_error = float(
        np.sqrt(np.mean(np.sum(np.square(cartesian_error), axis=1)))
    )
    joint_rmse = float(np.sqrt(np.mean(np.square(q - expert_q))))
    derivative = derivative_metrics(q)

    (
        start_error,
        end_error,
        path_length_pred,
        path_length_desired,
        path_length_ratio,
        frechet,
        dtw,
    ) = shape_path_metrics(ee, desired_path)
    (
        tangent_cosine,
        tangent_weighted,
        progress,
        length_ratio_error,
        normalized_shape,
    ) = drawing_fidelity_metrics(
        ee,
        desired_path,
        path_length_pred,
        path_length_desired,
    )
    velocity_cost, acceleration_cost, jerk_cost = smoothness_costs(q)
    limit_cost = joint_limit_violation(q, lower_limits, upper_limits)
    robot_cost = total_cost(
        weights,
        mean_cartesian_error,
        max_cartesian_error,
        velocity_cost,
        acceleration_cost,
        jerk_cost,
        limit_cost,
    )
    shape_cost = shape_total_cost(
        weights,
        mean_cartesian_error,
        max_cartesian_error,
        start_error,
        end_error,
        frechet,
        dtw,
        velocity_cost,
        acceleration_cost,
        jerk_cost,
        limit_cost,
    )
    drawing_cost = drawing_total_cost(
        weights,
        shape_cost,
        tangent_weighted,
        progress,
        length_ratio_error,
        normalized_shape,
    )

    return {
        "mean_cartesian_error": mean_cartesian_error,
        "max_cartesian_error": max_cartesian_error,
        "rmse_cartesian_error": rmse_cartesian_error,
        "joint_rmse_to_expert": joint_rmse,
        **derivative,
        "start_error": start_error,
        "end_error": end_error,
        "path_length_pred": path_length_pred,
        "path_length_desired": path_length_desired,
        "path_length_ratio": path_length_ratio,
        "frechet_distance": frechet,
        "dtw_distance": dtw,
        "tangent_cosine_error": tangent_cosine,
        "tangent_weighted_error": tangent_weighted,
        "progress_error": progress,
        "length_ratio_error": length_ratio_error,
        "normalized_shape_error": normalized_shape,
        "joint_velocity_cost": velocity_cost,
        "joint_acceleration_cost": acceleration_cost,
        "joint_jerk_cost": jerk_cost,
        "joint_limit_violation": limit_cost,
        "total_cost": robot_cost,
        "shape_total_cost": shape_cost,
        "drawing_total_cost": drawing_cost,
    }


def write_joint_csv(
    path: Path,
    times: np.ndarray,
    q: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["t", "q1", "q2", "q3", "q4", "q5", "q6"])
        for time, values in zip(times, q):
            writer.writerow(
                [f"{float(time):.12g}"]
                + [f"{float(value):.10f}" for value in values]
            )


def write_xyz_csv(
    path: Path,
    times: np.ndarray,
    xyz: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["t", "x", "y", "z"])
        for time, values in zip(times, xyz):
            writer.writerow(
                [f"{float(time):.12g}"]
                + [f"{float(value):.10f}" for value in values]
            )


def add_improvement_fields(
    row: Dict[str, Any],
    prior_metrics: Dict[str, float],
) -> None:
    for output_key, metric_key in (
        ("cartesian_improvement_vs_prior_percent", "mean_cartesian_error"),
        ("cartesian_rmse_improvement_vs_prior_percent", "rmse_cartesian_error"),
        ("joint_rmse_improvement_vs_prior_percent", "joint_rmse_to_expert"),
        ("drawing_cost_improvement_vs_prior_percent", "drawing_total_cost"),
    ):
        prior_value = float(prior_metrics[metric_key])
        candidate_value = float(row[metric_key])
        row[output_key] = (
            100.0 * (prior_value - candidate_value) / prior_value
            if abs(prior_value) > EPS
            else float("nan")
        )


def write_summary(
    path: Path,
    rows: Sequence[Dict[str, Any]],
) -> None:
    fields = [
        "path_name",
        "source",
        "alpha",
        *METRIC_FIELDS,
        "cartesian_improvement_vs_prior_percent",
        "cartesian_rmse_improvement_vs_prior_percent",
        "joint_rmse_improvement_vs_prior_percent",
        "drawing_cost_improvement_vs_prior_percent",
        "min_window_coverage",
        "max_window_coverage",
        "path_dir",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            output = dict(row)
            for key in fields:
                if key in {
                    "path_name",
                    "source",
                    "path_dir",
                    "min_window_coverage",
                    "max_window_coverage",
                }:
                    continue
                if output[key] == "":
                    continue
                output[key] = f"{float(output[key]):.12e}"
            writer.writerow({field: output[field] for field in fields})


def aggregate_rows(
    rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((str(row["source"]), str(row["alpha"])), []).append(row)

    prior_by_path = {
        str(row["path_name"]): row
        for row in rows
        if row["source"] == "prior_only"
    }
    aggregate: List[Dict[str, Any]] = []
    for (source, alpha), group in sorted(
        groups.items(),
        key=lambda item: (
            0 if item[0][0] == "prior_only" else 1,
            -1.0 if item[0][1] == "" else float(item[0][1]),
        ),
    ):
        result: Dict[str, Any] = {
            "source": source,
            "alpha": alpha,
            "path_count": len(group),
        }
        for metric in METRIC_FIELDS:
            values = np.asarray(
                [float(row[metric]) for row in group],
                dtype=np.float64,
            )
            result[f"mean_{metric}"] = float(np.nanmean(values))

        if source == "prior_only":
            comparable = len(group)
            cartesian_improved = 0
            joint_improved = 0
            drawing_improved = 0
        else:
            comparable = 0
            cartesian_improved = 0
            joint_improved = 0
            drawing_improved = 0
            for row in group:
                prior = prior_by_path.get(str(row["path_name"]))
                if prior is None:
                    continue
                comparable += 1
                cartesian_improved += int(
                    float(row["mean_cartesian_error"])
                    < float(prior["mean_cartesian_error"])
                )
                joint_improved += int(
                    float(row["joint_rmse_to_expert"])
                    < float(prior["joint_rmse_to_expert"])
                )
                drawing_improved += int(
                    float(row["drawing_total_cost"])
                    < float(prior["drawing_total_cost"])
                )
        result.update(
            {
                "comparable_path_count": comparable,
                "cartesian_improved_path_count": cartesian_improved,
                "cartesian_improved_path_ratio": (
                    cartesian_improved / max(comparable, 1)
                ),
                "joint_rmse_improved_path_count": joint_improved,
                "joint_rmse_improved_path_ratio": (
                    joint_improved / max(comparable, 1)
                ),
                "drawing_cost_improved_path_count": drawing_improved,
                "drawing_cost_improved_path_ratio": (
                    drawing_improved / max(comparable, 1)
                ),
            }
        )
        aggregate.append(result)
    return aggregate


def write_aggregate(
    path: Path,
    rows: Sequence[Dict[str, Any]],
) -> None:
    fields = [
        "source",
        "alpha",
        "path_count",
        *[f"mean_{metric}" for metric in METRIC_FIELDS],
        "comparable_path_count",
        "cartesian_improved_path_count",
        "cartesian_improved_path_ratio",
        "joint_rmse_improved_path_count",
        "joint_rmse_improved_path_ratio",
        "drawing_cost_improved_path_count",
        "drawing_cost_improved_path_ratio",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        integer_fields = {
            "path_count",
            "comparable_path_count",
            "cartesian_improved_path_count",
            "joint_rmse_improved_path_count",
            "drawing_cost_improved_path_count",
        }
        for row in rows:
            output = dict(row)
            for key in fields:
                if key in {"source", "alpha"} or key in integer_fields:
                    continue
                output[key] = f"{float(output[key]):.12e}"
            writer.writerow({field: output[field] for field in fields})


def main() -> int:
    args = parse_args()
    alphas = parse_alphas(args.alphas)
    set_seed(args.seed)
    device = resolve_device(args.device)

    checkpoint = load_checkpoint(args.predictor_checkpoint, device)
    model = instantiate_model(checkpoint, device)
    window_data = load_npz(args.test_windows_npz, "test windows")
    (
        condition,
        _,
        prior_windows,
        _,
        window_names,
        window_starts,
    ) = validate_test_data(window_data, checkpoint)
    residual_mean, residual_std = load_residual_stats(
        args.stats_npz,
        checkpoint,
    )
    verify_checkpoint_stats(checkpoint, residual_mean, residual_std)
    horizon, stride = validate_stats_and_layout(
        args.stats_npz,
        checkpoint,
        condition,
        window_starts,
    )

    original_data = load_npz(args.test_npz, "original test NPZ")
    expert_q, desired_paths, original_names = validate_original_test(
        original_data
    )
    num_paths, num_steps, _ = expert_q.shape
    groups = collect_path_windows(
        window_names,
        window_starts,
        original_names,
        num_steps,
        horizon,
        stride,
    )

    predicted_residual_norm = predict_all(model, condition, device)
    predicted_residual_q = (
        predicted_residual_norm
        * residual_std.reshape(1, 1, TARGET_DIM)
        + residual_mean.reshape(1, 1, TARGET_DIM)
    ).astype(np.float32)

    robot, joint_names, ee_link = load_fk_context(None, None)
    lower_limits, upper_limits = extract_joint_limits(robot, joint_names)
    weights = default_weights()
    summary_rows: List[Dict[str, Any]] = []

    print(
        f"Loaded {num_paths} full trajectories, {condition.shape[0]} windows, "
        f"horizon={horizon}, stride={stride}, device={device}"
    )
    for path_index, name in enumerate(original_names):
        path_dir = args.output_dir / safe_path_name(name)
        times = trajectory_times(original_data, path_index, num_steps)
        full_prior_q = read_predicted_q_csv(
            args.prior_dir / safe_path_name(name) / "predicted_q.csv",
            expected_steps=num_steps,
        )
        indices = groups[name]

        prior_ee = fk_positions(
            robot,
            joint_names,
            ee_link,
            full_prior_q,
        )
        prior_metrics = evaluate_trajectory(
            q=full_prior_q,
            expert_q=expert_q[path_index],
            desired_path=desired_paths[path_index],
            ee=prior_ee,
            lower_limits=lower_limits,
            upper_limits=upper_limits,
            weights=weights,
        )
        prior_row: Dict[str, Any] = {
            "path_name": name,
            "source": "prior_only",
            "alpha": "",
            **prior_metrics,
            "min_window_coverage": "",
            "max_window_coverage": "",
            "path_dir": str(path_dir),
        }
        add_improvement_fields(prior_row, prior_metrics)
        summary_rows.append(prior_row)

        write_joint_csv(path_dir / "prior_q.csv", times, full_prior_q)
        write_joint_csv(
            path_dir / "expert_q.csv",
            times,
            expert_q[path_index],
        )
        write_xyz_csv(
            path_dir / "desired_path.csv",
            times,
            desired_paths[path_index],
        )
        write_xyz_csv(path_dir / "prior_ee.csv", times, prior_ee)

        for alpha in alphas:
            label = alpha_label(alpha)
            candidate_q, coverage = stitch_candidate(
                path_name=name,
                indices=indices,
                starts=window_starts,
                prior_windows=prior_windows,
                predicted_residual_q=predicted_residual_q,
                full_prior_q=full_prior_q,
                alpha=alpha,
            )
            candidate_ee = fk_positions(
                robot,
                joint_names,
                ee_link,
                candidate_q,
            )
            metrics = evaluate_trajectory(
                q=candidate_q,
                expert_q=expert_q[path_index],
                desired_path=desired_paths[path_index],
                ee=candidate_ee,
                lower_limits=lower_limits,
                upper_limits=upper_limits,
                weights=weights,
            )
            row: Dict[str, Any] = {
                "path_name": name,
                "source": "scaled_residual",
                "alpha": alpha,
                **metrics,
                "min_window_coverage": int(np.min(coverage)),
                "max_window_coverage": int(np.max(coverage)),
                "path_dir": str(path_dir),
            }
            add_improvement_fields(row, prior_metrics)
            summary_rows.append(row)

            write_joint_csv(
                path_dir / f"candidate_q_alpha_{label}.csv",
                times,
                candidate_q,
            )
            write_xyz_csv(
                path_dir / f"candidate_ee_alpha_{label}.csv",
                times,
                candidate_ee,
            )

        if (path_index + 1) % 10 == 0 or path_index + 1 == num_paths:
            print(f"Evaluated {path_index + 1}/{num_paths} paths")

    aggregate_rows_output = aggregate_rows(summary_rows)
    summary_path = args.output_dir / "full_trajectory_fk_summary.csv"
    aggregate_path = args.output_dir / "full_trajectory_fk_aggregate.csv"
    write_summary(summary_path, summary_rows)
    write_aggregate(aggregate_path, aggregate_rows_output)

    print(f"Saved full-trajectory summary: {summary_path}")
    print(f"Saved aggregate summary: {aggregate_path}")
    for row in aggregate_rows_output:
        alpha = "-" if row["alpha"] == "" else row["alpha"]
        print(
            f"{row['source']} alpha={alpha} | "
            f"cartesian={float(row['mean_mean_cartesian_error']):.8e} | "
            f"joint_rmse={float(row['mean_joint_rmse_to_expert']):.8e} | "
            f"drawing_cost={float(row['mean_drawing_total_cost']):.8e} | "
            f"cartesian_improved="
            f"{int(row['cartesian_improved_path_count'])}/"
            f"{int(row['comparable_path_count'])}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
