#!/usr/bin/env python3
"""Evaluate globally anchored receding-horizon residual diffusion rollouts.

The fixed MLP trajectory is the only persistent global joint reference. Expert
joints are supplied only to the final evaluation function and are never used
for conditioning, candidate generation, selection, tail handling, or extension.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
from diagnose_action_buffer_candidate_selection import (
    CONDITION_DIM,
    JOINT_DIM,
    build_teacher_forced_buffer,
    build_v5b_condition,
    decode_names,
    default_weights,
    desired_differences,
    evaluate_region,
    extract_joint_limits,
    finite_array,
    limit_metrics,
    load_npz,
    load_stats,
    require_keys,
    resolve_device,
    set_seed,
)
from diagnose_diffusion_v5_sampling_modes import reverse_noised_x0_batches
from diagnose_scaled_tapered_action_buffer_candidates import (
    apply_scaled_taper,
    taper_values,
    weights_record,
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
    "global_anchored_receding_horizon_rollout"
)

TAIL_MODES = (
    "full_selected_tail",
    "base_tail",
    "decayed_selected_tail",
    "global_anchored_tail",
)
EPS = 1e-12


@dataclass(frozen=True)
class ScoreWeights:
    prefix: float
    tail: float
    reference_joint: float
    reference_cartesian: float
    terminal: float
    late_window: float
    shift_boundary: float
    extension: float
    safety: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate globally constrained action-buffer diffusion with "
            "independent tail-handling rollouts."
        )
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--test_npz", type=Path, default=DEFAULT_TEST_NPZ)
    parser.add_argument("--stats_npz", type=Path, default=DEFAULT_STATS_NPZ)
    parser.add_argument("--prior_dir", type=Path, default=DEFAULT_PRIOR_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--prediction_horizon", type=int, default=32)
    parser.add_argument("--execution_horizon", type=int, default=8)
    parser.add_argument("--t_init", type=int, default=10)
    parser.add_argument("--ramp_length", type=int, default=8)
    parser.add_argument(
        "--taper_mode", choices=("linear", "cosine"), default="linear"
    )
    parser.add_argument(
        "--alphas", nargs="+", type=float, default=(0.05, 0.10)
    )
    parser.add_argument("--num_base_samples", type=int, default=16)
    parser.add_argument(
        "--tail_modes", nargs="+", choices=TAIL_MODES, default=TAIL_MODES
    )
    parser.add_argument(
        "--tail_extension",
        choices=("constant_position", "constant_velocity", "bootstrap_prior"),
        default="bootstrap_prior",
    )
    parser.add_argument(
        "--selector",
        choices=("larger_picture", "discounted_hard_gate"),
        default="larger_picture",
    )
    parser.add_argument("--ranking_discount", type=float, default=0.9)
    parser.add_argument("--lookahead_points", type=int, default=8)

    parser.add_argument(
        "--tail_decay_mode",
        choices=("linear", "exponential", "cosine"),
        default="linear",
    )
    parser.add_argument("--tail_decay_length", type=int, default=8)
    parser.add_argument("--tail_decay_beta", type=float, default=0.35)
    parser.add_argument(
        "--global_anchor_mode",
        choices=("linear", "exponential", "cosine"),
        default="linear",
    )
    parser.add_argument("--global_anchor_length", type=int, default=8)
    parser.add_argument("--global_anchor_beta", type=float, default=0.35)

    parser.add_argument("--w_prefix", type=float, default=1.0)
    parser.add_argument("--w_tail", type=float, default=0.35)
    parser.add_argument("--w_reference_joint", type=float, default=0.50)
    parser.add_argument("--w_reference_cartesian", type=float, default=1.0)
    parser.add_argument("--w_terminal", type=float, default=0.75)
    parser.add_argument("--w_late_window", type=float, default=0.50)
    parser.add_argument("--w_shift_boundary", type=float, default=1.0)
    parser.add_argument("--w_extension", type=float, default=0.75)
    parser.add_argument("--w_safety", type=float, default=10.0)

    parser.add_argument("--boundary_ratio", type=float, default=2.0)
    parser.add_argument("--prefix_step_ratio", type=float, default=2.0)
    parser.add_argument("--shift_boundary_ratio", type=float, default=2.0)
    parser.add_argument("--absolute_boundary_limit", type=float, default=0.25)
    parser.add_argument("--absolute_prefix_step_limit", type=float, default=0.25)
    parser.add_argument(
        "--absolute_shift_boundary_limit", type=float, default=0.25
    )
    parser.add_argument("--max_extension_clipped_values", type=int, default=0)
    parser.add_argument("--max_extension_joint_step", type=float, default=0.25)
    parser.add_argument("--max_reference_joint_drift", type=float, default=0.50)
    parser.add_argument(
        "--max_reference_cartesian_drift", type=float, default=0.10
    )
    parser.add_argument(
        "--material_safety_ratio",
        type=float,
        default=1.10,
        help="Maximum final safety/continuity ratio used by pass criteria.",
    )
    parser.add_argument("--num_diffusion_steps", type=int, default=None)
    parser.add_argument("--max_paths", type=int, default=None)
    parser.add_argument("--save_candidate_details", action="store_true")
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.prediction_horizon <= 0 or args.execution_horizon <= 0:
        raise ValueError("Prediction and execution horizons must be positive")
    if args.execution_horizon > args.prediction_horizon:
        raise ValueError("execution_horizon cannot exceed prediction_horizon")
    if not 0 < args.lookahead_points <= args.prediction_horizon:
        raise ValueError("lookahead_points must be in [1,prediction_horizon]")
    if args.num_base_samples <= 0:
        raise ValueError("num_base_samples must be positive")
    if args.ramp_length < 0:
        raise ValueError("ramp_length must be non-negative")
    if not args.alphas or any(
        value <= 0.0 or not np.isfinite(value) for value in args.alphas
    ):
        raise ValueError("alphas must be finite and positive")
    if len(set(float(value) for value in args.alphas)) != len(args.alphas):
        raise ValueError("alphas contains duplicate values")
    if len(set(args.tail_modes)) != len(args.tail_modes):
        raise ValueError("tail_modes contains duplicate values")
    if args.max_paths is not None and args.max_paths <= 0:
        raise ValueError("max_paths must be positive when supplied")
    if not 0.0 < args.ranking_discount <= 1.0:
        raise ValueError("ranking_discount must be in (0,1]")
    if args.tail_decay_length <= 0 or args.global_anchor_length <= 0:
        raise ValueError("Tail decay and global anchor lengths must be positive")
    if args.tail_decay_beta <= 0.0 or args.global_anchor_beta <= 0.0:
        raise ValueError("Exponential tail beta values must be positive")
    if args.max_extension_clipped_values < 0:
        raise ValueError("max_extension_clipped_values must be non-negative")
    if args.material_safety_ratio < 1.0:
        raise ValueError("material_safety_ratio must be at least 1")
    positive = (
        "boundary_ratio",
        "prefix_step_ratio",
        "shift_boundary_ratio",
        "absolute_boundary_limit",
        "absolute_prefix_step_limit",
        "absolute_shift_boundary_limit",
        "max_extension_joint_step",
        "max_reference_joint_drift",
        "max_reference_cartesian_drift",
    )
    for name in positive:
        if float(getattr(args, name)) <= 0.0:
            raise ValueError(f"--{name} must be positive")
    weight_names = (
        "w_prefix",
        "w_tail",
        "w_reference_joint",
        "w_reference_cartesian",
        "w_terminal",
        "w_late_window",
        "w_shift_boundary",
        "w_extension",
        "w_safety",
    )
    if any(float(getattr(args, name)) < 0.0 for name in weight_names):
        raise ValueError("Larger-picture score weights must be non-negative")
    if not any(float(getattr(args, name)) > 0.0 for name in weight_names):
        raise ValueError("At least one larger-picture score weight must be positive")


def score_weights(args: argparse.Namespace) -> ScoreWeights:
    return ScoreWeights(
        prefix=float(args.w_prefix),
        tail=float(args.w_tail),
        reference_joint=float(args.w_reference_joint),
        reference_cartesian=float(args.w_reference_cartesian),
        terminal=float(args.w_terminal),
        late_window=float(args.w_late_window),
        shift_boundary=float(args.w_shift_boundary),
        extension=float(args.w_extension),
        safety=float(args.w_safety),
    )


def clip_to_joint_limits(
    raw_q: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    raw = np.asarray(raw_q, dtype=np.float32)
    if raw.ndim != 2 or raw.shape[1] != JOINT_DIM:
        raise ValueError(f"Joint trajectory must have shape (N,6), got {raw.shape}")
    finite = bool(np.all(np.isfinite(raw)))
    if not finite:
        return raw.copy(), {
            "finite_before_clipping": 0,
            "clipped_value_count": 0,
            "clipped_timestep_count": 0,
            "clipping_rms": float("inf"),
            "clipping_max_abs": float("inf"),
            "clipping_total_abs": float("inf"),
        }
    clipped = raw.copy()
    for joint in range(JOINT_DIM):
        if np.isfinite(lower[joint]):
            clipped[:, joint] = np.maximum(clipped[:, joint], lower[joint])
        if np.isfinite(upper[joint]):
            clipped[:, joint] = np.minimum(clipped[:, joint], upper[joint])
    delta = clipped.astype(np.float64) - raw.astype(np.float64)
    changed = np.abs(delta) > 1e-9
    return clipped, {
        "finite_before_clipping": 1,
        "clipped_value_count": int(np.count_nonzero(changed)),
        "clipped_timestep_count": int(np.count_nonzero(np.any(changed, axis=1))),
        "clipping_rms": float(np.sqrt(np.mean(np.square(delta)))) if delta.size else 0.0,
        "clipping_max_abs": float(np.max(np.abs(delta))) if delta.size else 0.0,
        "clipping_total_abs": float(np.sum(np.abs(delta))) if delta.size else 0.0,
    }


def padded_indices(start: int, length: int, trajectory_length: int) -> np.ndarray:
    if not 0 <= start <= trajectory_length:
        raise AssertionError(
            f"Global start index {start} is outside [0,{trajectory_length}]"
        )
    if length < 0:
        raise AssertionError("Padded segment length cannot be negative")
    indices = np.minimum(
        np.arange(start, start + length, dtype=np.int64),
        trajectory_length - 1,
    )
    if indices.size and (
        int(indices[0]) != min(start, trajectory_length - 1)
        or np.any(indices < 0)
        or np.any(indices >= trajectory_length)
    ):
        raise AssertionError("Padded global-reference indexing is inconsistent")
    return indices


def derivative_costs(q: np.ndarray) -> Dict[str, float]:
    q64 = np.asarray(q, dtype=np.float64)

    def values(order: int) -> np.ndarray:
        if q64.shape[0] <= order:
            return np.empty((0, JOINT_DIM), dtype=np.float64)
        return np.diff(q64, n=order, axis=0)

    velocity = values(1)
    acceleration = values(2)
    jerk = values(3)

    def rms(array: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.square(array)))) if array.size else 0.0

    def mean_norm(array: np.ndarray) -> float:
        return float(np.mean(np.linalg.norm(array, axis=1))) if array.size else 0.0

    def max_norm(array: np.ndarray) -> float:
        return float(np.max(np.linalg.norm(array, axis=1))) if array.size else 0.0

    return {
        "velocity_cost": float(np.mean(np.sum(np.square(velocity), axis=1)))
        if velocity.size
        else 0.0,
        "acceleration_cost": float(
            np.mean(np.sum(np.square(acceleration), axis=1))
        )
        if acceleration.size
        else 0.0,
        "jerk_cost": float(np.mean(np.sum(np.square(jerk), axis=1)))
        if jerk.size
        else 0.0,
        "velocity_rms": rms(velocity),
        "acceleration_rms": rms(acceleration),
        "jerk_rms": rms(jerk),
        "mean_joint_velocity": mean_norm(velocity),
        "max_joint_velocity": max_norm(velocity),
        "mean_joint_acceleration": mean_norm(acceleration),
        "max_joint_acceleration": max_norm(acceleration),
        "mean_joint_jerk": mean_norm(jerk),
        "max_joint_jerk": max_norm(jerk),
        "max_joint_step": float(np.max(np.abs(velocity))) if velocity.size else 0.0,
    }


def sequence_metrics(
    *,
    q: np.ndarray,
    desired: np.ndarray,
    previous_q: np.ndarray,
    robot: Any,
    joint_names: Sequence[str],
    ee_link: str,
    lower: np.ndarray,
    upper: np.ndarray,
    drawing_weights: Any,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    q = np.asarray(q, dtype=np.float32)
    desired = np.asarray(desired, dtype=np.float32)
    if q.shape[0] != desired.shape[0] or q.shape[1:] != (JOINT_DIM,):
        raise ValueError("Joint and desired sequence lengths do not match")
    if q.shape[0] == 0:
        metrics = {
            "mean_cartesian_error": 0.0,
            "rms_cartesian_error": 0.0,
            "max_cartesian_error": 0.0,
            "drawing_cost": 0.0,
            "joint_limit_violation_count": 0,
            "joint_limit_violation_magnitude": 0.0,
            **derivative_costs(q),
        }
        return metrics, np.empty((0, 3), dtype=np.float32), np.empty(0)
    finite_array(q, "practical candidate joint sequence")
    ee = fk_positions(robot, joint_names, ee_link, q)
    errors = np.linalg.norm(ee - desired, axis=1)
    drawing = evaluate_region(
        q=q,
        desired=desired,
        ee=ee,
        previous_q=previous_q,
        lower=lower,
        upper=upper,
        weights=drawing_weights,
    )
    limit_count, limit_magnitude = limit_metrics(q, lower, upper)
    metrics = {
        "mean_cartesian_error": float(np.mean(errors)),
        "rms_cartesian_error": float(np.sqrt(np.mean(np.square(errors)))),
        "max_cartesian_error": float(np.max(errors)),
        "drawing_cost": float(drawing["drawing_cost"]),
        "joint_limit_violation_count": int(limit_count),
        "joint_limit_violation_magnitude": float(limit_magnitude),
        **derivative_costs(q),
    }
    return metrics, ee, errors


def decay_weights(
    length: int,
    mode: str,
    transition_length: int,
    beta: float,
) -> np.ndarray:
    if length <= 0:
        return np.empty(0, dtype=np.float32)
    positions = np.arange(length, dtype=np.float64)
    if mode == "linear":
        weights = np.maximum(0.0, 1.0 - positions / float(transition_length))
    elif mode == "exponential":
        weights = np.exp(-float(beta) * positions)
    elif mode == "cosine":
        phase = np.clip(positions / float(transition_length), 0.0, 1.0)
        weights = 0.5 * (1.0 + np.cos(np.pi * phase))
    else:
        raise ValueError(f"Unknown decay mode: {mode}")
    weights[0] = 1.0
    return weights.astype(np.float32)


def handled_tail(
    *,
    candidate_q: np.ndarray,
    base_buffer: np.ndarray,
    global_reference: np.ndarray,
    start: int,
    execution_count: int,
    tail_mode: str,
    is_diffusion: bool,
    args: argparse.Namespace,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = candidate_q.shape[0] - execution_count
    global_indices = padded_indices(
        start + execution_count, count, global_reference.shape[0]
    )
    global_tail = global_reference[global_indices]
    base_tail = base_buffer[execution_count:].copy()
    selected_tail = candidate_q[execution_count:].copy()
    if not is_diffusion or tail_mode == "base_tail":
        weights = np.zeros(count, dtype=np.float32)
        return base_tail, weights, global_indices
    if tail_mode == "full_selected_tail":
        weights = np.ones(count, dtype=np.float32)
        return selected_tail, weights, global_indices
    if tail_mode == "decayed_selected_tail":
        weights = decay_weights(
            count,
            args.tail_decay_mode,
            args.tail_decay_length,
            args.tail_decay_beta,
        )
        tail = base_tail + weights[:, None] * (selected_tail - base_tail)
        return tail.astype(np.float32), weights, global_indices
    if tail_mode == "global_anchored_tail":
        weights = decay_weights(
            count,
            args.global_anchor_mode,
            args.global_anchor_length,
            args.global_anchor_beta,
        )
        tail = (
            weights[:, None] * selected_tail
            + (1.0 - weights[:, None]) * global_tail
        )
        return tail.astype(np.float32), weights, global_indices
    raise ValueError(f"Unknown tail mode: {tail_mode}")


def make_extension(
    *,
    retained_tail: np.ndarray,
    selected_candidate: np.ndarray,
    execution_count: int,
    extension_count: int,
    first_global_index: int,
    global_reference: np.ndarray,
    mode: str,
) -> Tuple[np.ndarray, List[str]]:
    if extension_count <= 0:
        return np.empty((0, JOINT_DIM), dtype=np.float32), []
    history: List[np.ndarray] = [row.copy() for row in retained_tail]
    seed_history = [
        row.copy()
        for row in selected_candidate[max(0, execution_count - 2) : execution_count]
    ]
    output: List[np.ndarray] = []
    sources: List[str] = []

    def last_values() -> List[np.ndarray]:
        combined = seed_history + history + output
        return combined[-2:]

    for offset in range(extension_count):
        global_index = first_global_index + offset
        if mode == "bootstrap_prior" and global_index < global_reference.shape[0]:
            value = global_reference[global_index].copy()
            source = "bootstrap_prior"
        elif mode in ("bootstrap_prior", "constant_velocity"):
            recent = last_values()
            if len(recent) >= 2:
                value = recent[-1] + (recent[-1] - recent[-2])
                source = (
                    "constant_velocity_fallback"
                    if mode == "bootstrap_prior"
                    else "constant_velocity"
                )
            elif recent:
                value = recent[-1].copy()
                source = "constant_position_fallback"
            else:
                raise AssertionError("No state is available for extension")
        elif mode == "constant_position":
            recent = last_values()
            if not recent:
                raise AssertionError("No state is available for extension")
            value = recent[-1].copy()
            source = "constant_position"
        else:
            raise ValueError(f"Unknown extension mode: {mode}")
        output.append(np.asarray(value, dtype=np.float32))
        sources.append(source)
    return np.stack(output, axis=0), sources


def extension_preview(
    *,
    candidate_q: np.ndarray,
    base_buffer: np.ndarray,
    global_reference: np.ndarray,
    desired_path: np.ndarray,
    start: int,
    execution_count: int,
    tail_mode: str,
    is_diffusion: bool,
    robot: Any,
    joint_names: Sequence[str],
    ee_link: str,
    lower: np.ndarray,
    upper: np.ndarray,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    horizon = candidate_q.shape[0]
    raw_retained, weights, retained_global_indices = handled_tail(
        candidate_q=candidate_q,
        base_buffer=base_buffer,
        global_reference=global_reference,
        start=start,
        execution_count=execution_count,
        tail_mode=tail_mode,
        is_diffusion=is_diffusion,
        args=args,
    )
    retained, retained_clipping = clip_to_joint_limits(
        raw_retained, lower, upper
    )
    next_start = start + execution_count
    extension_count = execution_count
    first_global_index = next_start + retained.shape[0]
    raw_extension, sources = make_extension(
        retained_tail=retained,
        selected_candidate=candidate_q,
        execution_count=execution_count,
        extension_count=extension_count,
        first_global_index=first_global_index,
        global_reference=global_reference,
        mode=args.tail_extension,
    )
    extension, clipping = clip_to_joint_limits(raw_extension, lower, upper)
    next_buffer = np.concatenate((retained, extension), axis=0).astype(np.float32)
    if next_buffer.shape != (horizon, JOINT_DIM):
        raise AssertionError(
            f"Previewed next buffer must have shape ({horizon},6), got {next_buffer.shape}"
        )
    preview_finite = bool(np.all(np.isfinite(next_buffer)))
    next_indices = padded_indices(next_start, horizon, global_reference.shape[0])
    if not np.array_equal(
        retained_global_indices, next_indices[: retained.shape[0]]
    ):
        raise AssertionError("Retained-tail global indices shifted unexpectedly")
    global_next = global_reference[next_indices]
    desired_next = desired_path[next_indices]
    if preview_finite:
        preview_ee = fk_positions(robot, joint_names, ee_link, next_buffer)
        global_next_ee = fk_positions(robot, joint_names, ee_link, global_next)
        preview_errors = np.linalg.norm(preview_ee - desired_next, axis=1)
        preview_reference_cart = np.linalg.norm(
            preview_ee - global_next_ee, axis=1
        )
    else:
        preview_errors = np.full(horizon, np.inf)
        preview_reference_cart = np.full(horizon, np.inf)
    preview_reference_joint = np.linalg.norm(next_buffer - global_next, axis=1)

    transition_seed = (
        retained[-1:]
        if retained.shape[0]
        else candidate_q[execution_count - 1 : execution_count]
    )
    extension_sequence = np.concatenate((transition_seed, extension), axis=0)
    extension_derivatives = derivative_costs(extension_sequence)
    if retained.shape[0] and execution_count > 0:
        retained_shift = retained[0] - candidate_q[execution_count - 1]
        retained_shift_l2 = float(np.linalg.norm(retained_shift))
        retained_shift_max = float(np.max(np.abs(retained_shift)))
    else:
        retained_shift_l2 = 0.0
        retained_shift_max = 0.0
    return {
        "preview_finite": int(preview_finite),
        "retained_tail_count": int(retained.shape[0]),
        "retained_tail_clipped_value_count": int(
            retained_clipping["clipped_value_count"]
        ),
        "retained_tail_clipping_max_abs": float(
            retained_clipping["clipping_max_abs"]
        ),
        "retained_tail_clipping_total_abs": float(
            retained_clipping["clipping_total_abs"]
        ),
        "tail_blending_weights": json.dumps([float(value) for value in weights]),
        "tail_weight_first": float(weights[0]) if weights.size else 0.0,
        "tail_weight_last": float(weights[-1]) if weights.size else 0.0,
        "retained_shift_boundary_joint_l2": retained_shift_l2,
        "retained_shift_boundary_max_abs_joint_step": retained_shift_max,
        "extension_count": int(extension_count),
        "extension_first_global_index": int(first_global_index),
        "extension_source": ";".join(sources),
        "extension_source_counts": json.dumps(dict(Counter(sources)), sort_keys=True),
        "extension_clipped_value_count": int(clipping["clipped_value_count"]),
        "extension_clipped_timestep_count": int(clipping["clipped_timestep_count"]),
        "extension_clipping_rms": float(clipping["clipping_rms"]),
        "extension_clipping_max_abs": float(clipping["clipping_max_abs"]),
        "extension_clipping_total_abs": float(clipping["clipping_total_abs"]),
        "extension_max_joint_step": extension_derivatives["max_joint_step"],
        "extension_velocity": extension_derivatives["velocity_rms"],
        "extension_acceleration": extension_derivatives["acceleration_rms"],
        "preview_next_buffer_reference_joint_drift": float(
            np.sqrt(np.mean(np.square(preview_reference_joint)))
        ),
        "preview_next_buffer_reference_cartesian_drift": float(
            np.sqrt(np.mean(np.square(preview_reference_cart)))
        ),
        "preview_next_buffer_mean_cartesian_error": float(
            np.mean(preview_errors)
        ),
        "_retained_tail": retained,
        "_next_buffer": next_buffer,
        "_tail_weights": weights,
    }


def generate_seeded_samples(
    *,
    model: torch.nn.Module,
    call_variant: str,
    condition_norm: np.ndarray,
    residual_norm_zero: np.ndarray,
    t_init: int,
    schedule: Dict[str, torch.Tensor],
    device: torch.device,
    global_seed: int,
    path_index: int,
    cycle_index: int,
    num_samples: int,
) -> Tuple[List[Optional[np.ndarray]], List[int]]:
    samples: List[Optional[np.ndarray]] = []
    seeds: List[int] = []
    for base_sample_index in range(num_samples):
        seed = (
            int(global_seed)
            + int(path_index) * 1_000_000
            + int(cycle_index) * 10_000
            + int(base_sample_index)
        )
        set_seed(seed)
        sample = reverse_noised_x0_batches(
            model=model,
            call_variant=call_variant,
            condition_bhc=condition_norm[None, :, :],
            x0_norm_bhc=residual_norm_zero[None, :, :],
            t_init=t_init,
            schedule=schedule,
            batch_size=1,
            device=device,
            deterministic=False,
        )[0].astype(np.float32)
        if sample.shape != residual_norm_zero.shape:
            raise ValueError(
                "Diffusion residual sample shape "
                f"{sample.shape} differs from {residual_norm_zero.shape}"
            )
        samples.append(sample if np.all(np.isfinite(sample)) else None)
        seeds.append(seed)
    return samples, seeds


def prefixed(source: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in source.items()}


def practical_candidate_metrics(
    *,
    candidate_q: np.ndarray,
    raw_candidate_q: np.ndarray,
    base_buffer: np.ndarray,
    desired_window: np.ndarray,
    global_segment: np.ndarray,
    global_reference: np.ndarray,
    desired_path: np.ndarray,
    start: int,
    execution_count: int,
    previous_executed_q: np.ndarray,
    tail_mode: str,
    is_diffusion: bool,
    clipping: Dict[str, Any],
    robot: Any,
    joint_names: Sequence[str],
    ee_link: str,
    lower: np.ndarray,
    upper: np.ndarray,
    drawing_weights: Any,
    score_config: ScoreWeights,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    preview = extension_preview(
        candidate_q=candidate_q,
        base_buffer=base_buffer,
        global_reference=global_reference,
        desired_path=desired_path,
        start=start,
        execution_count=execution_count,
        tail_mode=tail_mode,
        is_diffusion=is_diffusion,
        robot=robot,
        joint_names=joint_names,
        ee_link=ee_link,
        lower=lower,
        upper=upper,
        args=args,
    )
    prefix_q = candidate_q[:execution_count]
    prefix_desired = desired_window[:execution_count]
    prefix_metrics, _, prefix_errors = sequence_metrics(
        q=prefix_q,
        desired=prefix_desired,
        previous_q=previous_executed_q,
        robot=robot,
        joint_names=joint_names,
        ee_link=ee_link,
        lower=lower,
        upper=upper,
        drawing_weights=drawing_weights,
    )
    retained_tail = np.asarray(preview["_retained_tail"], dtype=np.float32)
    tail_desired = desired_window[execution_count:]
    tail_previous = (
        candidate_q[execution_count - 1]
        if execution_count > 0
        else previous_executed_q
    )
    tail_metrics, _, _ = sequence_metrics(
        q=retained_tail,
        desired=tail_desired,
        previous_q=tail_previous,
        robot=robot,
        joint_names=joint_names,
        ee_link=ee_link,
        lower=lower,
        upper=upper,
        drawing_weights=drawing_weights,
    )
    late_start = candidate_q.shape[0] - args.lookahead_points
    late_metrics, _, _ = sequence_metrics(
        q=candidate_q[late_start:],
        desired=desired_window[late_start:],
        previous_q=candidate_q[late_start - 1] if late_start else previous_executed_q,
        robot=robot,
        joint_names=joint_names,
        ee_link=ee_link,
        lower=lower,
        upper=upper,
        drawing_weights=drawing_weights,
    )
    candidate_ee = fk_positions(robot, joint_names, ee_link, candidate_q)
    global_ee = fk_positions(robot, joint_names, ee_link, global_segment)
    reference_joint_delta = candidate_q - global_segment
    reference_cart_delta = candidate_ee - global_ee
    base_delta = candidate_q - base_buffer
    reference_joint_cost = float(
        np.mean(np.sum(np.square(reference_joint_delta), axis=1))
    )
    reference_cartesian_cost = float(
        np.mean(np.sum(np.square(reference_cart_delta), axis=1))
    )
    reference_joint_rms = float(np.sqrt(np.mean(np.square(reference_joint_delta))))
    reference_cartesian_rms = float(
        np.sqrt(np.mean(np.square(reference_cart_delta)))
    )
    local_base_joint_rms = float(np.sqrt(np.mean(np.square(base_delta))))
    terminal_error = candidate_ee[-1] - desired_window[-1]
    terminal_cartesian_cost = float(np.dot(terminal_error, terminal_error))
    terminal_joint_delta = candidate_q[-1] - global_segment[-1]
    terminal_joint_reference_cost = float(
        np.dot(terminal_joint_delta, terminal_joint_delta)
    )
    current_boundary = candidate_q[0] - previous_executed_q
    current_boundary_l2 = float(np.linalg.norm(current_boundary))
    current_boundary_max = float(np.max(np.abs(current_boundary)))
    if execution_count < candidate_q.shape[0]:
        shift = candidate_q[execution_count] - candidate_q[execution_count - 1]
        shift_l2 = float(np.linalg.norm(shift))
        shift_max = float(np.max(np.abs(shift)))
    else:
        shift_l2 = 0.0
        shift_max = 0.0
    raw_limit_count, raw_limit_magnitude = limit_metrics(
        raw_candidate_q, lower, upper
    )
    actual_limit_count, actual_limit_magnitude = limit_metrics(
        candidate_q, lower, upper
    )
    raw_selected_tail_delta = candidate_q[execution_count:] - base_buffer[execution_count:]
    raw_selected_tail_global_delta = (
        candidate_q[execution_count:] - global_segment[execution_count:]
    )
    retained_tail_base_delta = (
        retained_tail - base_buffer[execution_count:]
    )
    retained_tail_global_delta = (
        retained_tail - global_segment[execution_count:]
    )

    j_prefix = float(
        prefix_metrics["drawing_cost"]
        + np.square(prefix_metrics["rms_cartesian_error"])
        + 0.01 * prefix_metrics["velocity_cost"]
        + 0.01 * prefix_metrics["acceleration_cost"]
        + 0.001 * prefix_metrics["jerk_cost"]
        + np.square(current_boundary_l2)
    )
    j_tail = float(
        tail_metrics["drawing_cost"]
        + np.square(tail_metrics["rms_cartesian_error"])
        + 0.01 * tail_metrics["velocity_cost"]
        + 0.01 * tail_metrics["acceleration_cost"]
        + 0.001 * tail_metrics["jerk_cost"]
    )
    j_terminal = terminal_cartesian_cost + 0.25 * terminal_joint_reference_cost
    j_late = float(
        late_metrics["drawing_cost"]
        + np.square(late_metrics["rms_cartesian_error"])
    )
    j_shift = float(
        np.square(shift_l2)
        + np.square(preview["retained_shift_boundary_joint_l2"])
    )
    j_extension = float(
        np.square(preview["extension_max_joint_step"])
        + np.square(preview["extension_velocity"])
        + np.square(preview["extension_acceleration"])
        + np.square(preview["preview_next_buffer_mean_cartesian_error"])
        + np.square(preview["preview_next_buffer_reference_joint_drift"])
        + np.square(preview["preview_next_buffer_reference_cartesian_drift"])
        + preview["extension_clipped_value_count"]
        + np.square(preview["extension_clipping_max_abs"])
    )
    j_safety = float(
        raw_limit_count
        + raw_limit_magnitude
        + actual_limit_count
        + actual_limit_magnitude
        + clipping["clipped_value_count"]
        + np.square(clipping["clipping_max_abs"])
        + preview["retained_tail_clipped_value_count"]
        + np.square(preview["retained_tail_clipping_max_abs"])
    )
    total_score = float(
        score_config.prefix * j_prefix
        + score_config.tail * j_tail
        + score_config.reference_joint * reference_joint_cost
        + score_config.reference_cartesian * reference_cartesian_cost
        + score_config.terminal * j_terminal
        + score_config.late_window * j_late
        + score_config.shift_boundary * j_shift
        + score_config.extension * j_extension
        + score_config.safety * j_safety
    )
    discounts = np.power(
        args.ranking_discount,
        np.arange(prefix_errors.shape[0], dtype=np.float64),
    )
    discounted_prefix_error = float(
        np.sum(discounts * prefix_errors) / max(np.sum(discounts), EPS)
    )
    discounted_score = float(
        discounted_prefix_error
        + 0.25 * late_metrics["mean_cartesian_error"]
        + 0.25 * reference_cartesian_rms
        + shift_max
        + preview["retained_shift_boundary_max_abs_joint_step"]
        + preview["extension_max_joint_step"]
        + score_config.safety * j_safety
    )
    return {
        **prefixed(prefix_metrics, "prefix"),
        **prefixed(tail_metrics, "tail"),
        **prefixed(late_metrics, "late_window"),
        "current_boundary_joint_l2": current_boundary_l2,
        "current_boundary_max_abs_joint_step": current_boundary_max,
        "shift_boundary_joint_l2": shift_l2,
        "shift_boundary_max_abs_joint_step": shift_max,
        "raw_selected_tail_base_deviation_rms": float(
            np.sqrt(np.mean(np.square(raw_selected_tail_delta)))
        )
        if raw_selected_tail_delta.size
        else 0.0,
        "raw_selected_tail_global_deviation_rms": float(
            np.sqrt(np.mean(np.square(raw_selected_tail_global_delta)))
        )
        if raw_selected_tail_global_delta.size
        else 0.0,
        "retained_tail_base_deviation_rms": float(
            np.sqrt(np.mean(np.square(retained_tail_base_delta)))
        )
        if retained_tail_base_delta.size
        else 0.0,
        "retained_tail_global_deviation_rms": float(
            np.sqrt(np.mean(np.square(retained_tail_global_delta)))
        )
        if retained_tail_global_delta.size
        else 0.0,
        "global_reference_joint_cost": reference_joint_cost,
        "global_reference_cartesian_cost": reference_cartesian_cost,
        "global_reference_joint_drift_rms": reference_joint_rms,
        "global_reference_cartesian_drift_rms": reference_cartesian_rms,
        "local_base_joint_drift_rms": local_base_joint_rms,
        "terminal_cartesian_error_squared": terminal_cartesian_cost,
        "terminal_cartesian_error": float(np.sqrt(terminal_cartesian_cost)),
        "terminal_joint_reference_deviation_squared": terminal_joint_reference_cost,
        "raw_joint_limit_violation_count": int(raw_limit_count),
        "raw_joint_limit_violation_magnitude": float(raw_limit_magnitude),
        "candidate_joint_limit_violation_count": int(actual_limit_count),
        "candidate_joint_limit_violation_magnitude": float(actual_limit_magnitude),
        **clipping,
        **{key: value for key, value in preview.items() if not key.startswith("_")},
        "j_prefix": j_prefix,
        "j_tail": j_tail,
        "j_reference_joint": reference_joint_cost,
        "j_reference_cartesian": reference_cartesian_cost,
        "j_terminal": j_terminal,
        "j_late_window": j_late,
        "j_shift_boundary": j_shift,
        "j_extension_preview": j_extension,
        "j_robot_safety": j_safety,
        "larger_picture_score": total_score,
        "discounted_hard_gate_score": discounted_score,
        "_next_buffer": preview["_next_buffer"],
        "_tail_weights": preview["_tail_weights"],
    }


def safety_status(
    row: Dict[str, Any],
    buffer: Dict[str, Any],
    args: argparse.Namespace,
) -> Tuple[int, str, int]:
    finite_keys = (
        "larger_picture_score",
        "discounted_hard_gate_score",
        "current_boundary_max_abs_joint_step",
        "prefix_max_joint_step",
        "shift_boundary_max_abs_joint_step",
        "retained_shift_boundary_max_abs_joint_step",
        "extension_max_joint_step",
        "preview_next_buffer_reference_joint_drift",
        "preview_next_buffer_reference_cartesian_drift",
    )
    if int(row.get("finite_before_clipping", 0)) == 0 or int(
        row.get("preview_finite", 0)
    ) == 0:
        return 0, "non_finite", 0
    if not np.all(np.isfinite([float(row[key]) for key in finite_keys])):
        return 0, "non_finite_metric", 0
    if int(row["raw_joint_limit_violation_count"]) > int(
        buffer["raw_joint_limit_violation_count"]
    ):
        return 0, "joint_limit_count_increase", 0
    if float(row["raw_joint_limit_violation_magnitude"]) > float(
        buffer["raw_joint_limit_violation_magnitude"]
    ) + EPS:
        return 0, "joint_limit_magnitude_increase", 0
    if int(row["candidate_joint_limit_violation_count"]) > int(
        buffer["candidate_joint_limit_violation_count"]
    ):
        return 0, "post_clip_joint_limit_count_increase", 0
    if int(row["retained_tail_clipped_value_count"]) > int(
        buffer["retained_tail_clipped_value_count"]
    ):
        return 0, "retained_tail_limit_clipping", 0

    def relative_limit(buffer_value: float, ratio: float, absolute: float) -> float:
        if buffer_value > absolute:
            return buffer_value + EPS
        return min(absolute, max(buffer_value * ratio, 1e-6))

    boundary_limit = relative_limit(
        float(buffer["current_boundary_max_abs_joint_step"]),
        args.boundary_ratio,
        args.absolute_boundary_limit,
    )
    prefix_limit = relative_limit(
        float(buffer["prefix_max_joint_step"]),
        args.prefix_step_ratio,
        args.absolute_prefix_step_limit,
    )
    shift_limit = relative_limit(
        max(
            float(buffer["shift_boundary_max_abs_joint_step"]),
            float(buffer["retained_shift_boundary_max_abs_joint_step"]),
        ),
        args.shift_boundary_ratio,
        args.absolute_shift_boundary_limit,
    )
    if float(row["current_boundary_max_abs_joint_step"]) > boundary_limit:
        return 0, "current_boundary_step", 0
    if float(row["prefix_max_joint_step"]) > prefix_limit:
        return 0, "executed_prefix_step", 0
    if max(
        float(row["shift_boundary_max_abs_joint_step"]),
        float(row["retained_shift_boundary_max_abs_joint_step"]),
    ) > shift_limit:
        return 0, "future_shift_boundary_step", 0
    extension_clip_limit = max(
        args.max_extension_clipped_values,
        int(buffer["extension_clipped_value_count"]),
    )
    if int(row["extension_clipped_value_count"]) > extension_clip_limit:
        return 0, "extension_clipping", 0
    extension_step_limit = relative_limit(
        float(buffer["extension_max_joint_step"]),
        args.prefix_step_ratio,
        args.max_extension_joint_step,
    )
    if float(row["extension_max_joint_step"]) > extension_step_limit:
        return 0, "extension_joint_step", 0
    if (
        float(row["global_reference_joint_drift_rms"])
        > args.max_reference_joint_drift
        or float(row["preview_next_buffer_reference_joint_drift"])
        > args.max_reference_joint_drift
    ):
        return 0, "global_reference_joint_drift", 0
    if (
        float(row["global_reference_cartesian_drift_rms"])
        > args.max_reference_cartesian_drift
        or float(row["preview_next_buffer_reference_cartesian_drift"])
        > args.max_reference_cartesian_drift
    ):
        return 0, "global_reference_cartesian_drift", 0
    safety_improved = int(
        float(row["current_boundary_max_abs_joint_step"])
        < float(buffer["current_boundary_max_abs_joint_step"]) - EPS
        or float(row["prefix_max_joint_step"])
        < float(buffer["prefix_max_joint_step"]) - EPS
        or float(row["retained_shift_boundary_max_abs_joint_step"])
        < float(buffer["retained_shift_boundary_max_abs_joint_step"]) - EPS
        or int(row["extension_clipped_value_count"])
        < int(buffer["extension_clipped_value_count"])
    )
    return 1, "", safety_improved


def failed_candidate_row(
    *,
    candidate_index: int,
    base_sample_index: int,
    candidate_seed: int,
    alpha: float,
    reason: str,
) -> Dict[str, Any]:
    return {
        "candidate_index": candidate_index,
        "candidate_type": "diffusion_refined",
        "base_sample_index": base_sample_index,
        "candidate_seed": candidate_seed,
        "alpha": float(alpha),
        "accepted": 0,
        "candidate_rejection_reason": reason,
        "safety_improving_candidate": 0,
        "larger_picture_score": float("inf"),
        "discounted_hard_gate_score": float("inf"),
        "finite_before_clipping": 0,
    }


def select_candidate(rows: Sequence[Dict[str, Any]], selector: str) -> int:
    if not rows or int(rows[0]["candidate_index"]) != 0:
        raise AssertionError("Candidate index zero must be the unrefined buffer")
    eligible = [
        index
        for index, row in enumerate(rows)
        if index == 0 or int(row.get("accepted", 0)) == 1
    ]
    score_key = (
        "larger_picture_score"
        if selector == "larger_picture"
        else "discounted_hard_gate_score"
    )
    return min(
        eligible,
        key=lambda index: (
            float(rows[index][score_key]),
            float(rows[index].get("larger_picture_score", np.inf)),
            index,
        ),
    )


def summarize_rejections(rows: Sequence[Dict[str, Any]]) -> str:
    reasons = Counter(
        str(row.get("candidate_rejection_reason", ""))
        for row in rows[1:]
        if int(row.get("accepted", 0)) == 0
    )
    reasons.pop("", None)
    return json.dumps(dict(reasons), sort_keys=True)


def run_method(
    *,
    method: str,
    path_index: int,
    path_name: str,
    desired_path: np.ndarray,
    expert_q_evaluation_only: np.ndarray,
    q_start: np.ndarray,
    global_reference: np.ndarray,
    model: torch.nn.Module,
    call_variant: str,
    schedule: Dict[str, torch.Tensor],
    residual_mean: np.ndarray,
    residual_std: np.ndarray,
    condition_mean: np.ndarray,
    condition_std: np.ndarray,
    robot: Any,
    joint_names: Sequence[str],
    ee_link: str,
    lower: np.ndarray,
    upper: np.ndarray,
    drawing_weights: Any,
    score_config: ScoreWeights,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    horizon = args.prediction_horizon
    execution_horizon = args.execution_horizon
    trajectory_length = desired_path.shape[0]
    tail_mode = "base_tail" if method == "buffer_only" else method
    buffer_q, _ = build_teacher_forced_buffer(
        global_reference, 0, horizon
    )
    buffer_q, initial_clipping = clip_to_joint_limits(buffer_q, lower, upper)
    if buffer_q.shape != (horizon, JOINT_DIM):
        raise AssertionError("Initial action buffer has an invalid shape")
    padded_indices(0, horizon, trajectory_length)
    desired_delta = desired_differences(desired_path)
    residual_zero = np.zeros((horizon, JOINT_DIM), dtype=np.float32)
    residual_norm_zero = (
        (residual_zero - residual_mean[None, :]) / residual_std[None, :]
    ).astype(np.float32)
    residual_round_trip = (
        residual_norm_zero * residual_std[None, :] + residual_mean[None, :]
    )
    if not np.allclose(residual_round_trip, residual_zero, rtol=1e-6, atol=1e-7):
        raise AssertionError("Physical-zero residual normalization failed")
    taper = taper_values(horizon, args.ramp_length, args.taper_mode)
    executed_chunks: List[np.ndarray] = []
    planning_rows: List[Dict[str, Any]] = []
    candidate_rows: List[Dict[str, Any]] = []

    for cycle_index, start in enumerate(
        range(0, trajectory_length, execution_horizon)
    ):
        execution_count = min(execution_horizon, trajectory_length - start)
        indices = padded_indices(start, horizon, trajectory_length)
        desired_window = desired_path[indices]
        global_segment = global_reference[indices]
        previous_executed_q = q_start if not executed_chunks else executed_chunks[-1][-1]
        condition, _ = build_v5b_condition(
            desired_path=desired_path,
            desired_delta=desired_delta,
            indices=indices,
            q_start=q_start,
            current_q=previous_executed_q,
            buffer_q=buffer_q,
            robot=robot,
            joint_names=joint_names,
            ee_link=ee_link,
        )
        if condition.shape != (horizon, CONDITION_DIM):
            raise AssertionError(
                f"Condition must have shape ({horizon},{CONDITION_DIM})"
            )
        condition_norm = (
            (condition - condition_mean[None, :]) / condition_std[None, :]
        ).astype(np.float32)
        finite_array(condition_norm, "global-rollout normalized condition")

        zero_clipping = {
            "finite_before_clipping": 1,
            "clipped_value_count": 0,
            "clipped_timestep_count": 0,
            "clipping_rms": 0.0,
            "clipping_max_abs": 0.0,
            "clipping_total_abs": 0.0,
        }
        buffer_metrics = practical_candidate_metrics(
            candidate_q=buffer_q,
            raw_candidate_q=buffer_q,
            base_buffer=buffer_q,
            desired_window=desired_window,
            global_segment=global_segment,
            global_reference=global_reference,
            desired_path=desired_path,
            start=start,
            execution_count=execution_count,
            previous_executed_q=previous_executed_q,
            tail_mode=tail_mode,
            is_diffusion=False,
            clipping=zero_clipping,
            robot=robot,
            joint_names=joint_names,
            ee_link=ee_link,
            lower=lower,
            upper=upper,
            drawing_weights=drawing_weights,
            score_config=score_config,
            args=args,
        )
        buffer_row: Dict[str, Any] = {
            "candidate_index": 0,
            "candidate_type": "unrefined_buffer",
            "base_sample_index": -1,
            "candidate_seed": -1,
            "alpha": 0.0,
            "accepted": 1,
            "candidate_rejection_reason": "",
            "safety_improving_candidate": 0,
            **buffer_metrics,
        }
        rows: List[Dict[str, Any]] = [buffer_row]

        if method != "buffer_only":
            generated_norm, sample_seeds = generate_seeded_samples(
                model=model,
                call_variant=call_variant,
                condition_norm=condition_norm,
                residual_norm_zero=residual_norm_zero,
                t_init=args.t_init,
                schedule=schedule,
                device=next(model.parameters()).device,
                global_seed=args.seed,
                path_index=path_index,
                cycle_index=cycle_index,
                num_samples=args.num_base_samples,
            )
            candidate_index = 1
            for base_sample_index, generated_sample in enumerate(generated_norm):
                for alpha in args.alphas:
                    if generated_sample is None:
                        rows.append(
                            failed_candidate_row(
                                candidate_index=candidate_index,
                                base_sample_index=base_sample_index,
                                candidate_seed=sample_seeds[base_sample_index],
                                alpha=float(alpha),
                                reason="non_finite_diffusion_sample",
                            )
                        )
                        candidate_index += 1
                        continue
                    generated_physical = (
                        generated_sample * residual_std[None, :]
                        + residual_mean[None, :]
                    ).astype(np.float32)
                    if not np.all(np.isfinite(generated_physical)):
                        rows.append(
                            failed_candidate_row(
                                candidate_index=candidate_index,
                                base_sample_index=base_sample_index,
                                candidate_seed=sample_seeds[base_sample_index],
                                alpha=float(alpha),
                                reason="non_finite_denormalized_residual",
                            )
                        )
                        candidate_index += 1
                        continue
                    scaled_residual = apply_scaled_taper(
                        generated_physical,
                        float(alpha),
                        taper,
                        anchor_first=False,
                        anchor_executed_prefix_start=False,
                        execution_horizon=execution_horizon,
                    )
                    raw_candidate = (buffer_q + scaled_residual).astype(np.float32)
                    candidate_q, clipping = clip_to_joint_limits(
                        raw_candidate, lower, upper
                    )
                    if int(clipping["finite_before_clipping"]) == 0:
                        rows.append(
                            failed_candidate_row(
                                candidate_index=candidate_index,
                                base_sample_index=base_sample_index,
                                candidate_seed=sample_seeds[base_sample_index],
                                alpha=float(alpha),
                                reason="non_finite_candidate",
                            )
                        )
                        candidate_index += 1
                        continue
                    metrics = practical_candidate_metrics(
                        candidate_q=candidate_q,
                        raw_candidate_q=raw_candidate,
                        base_buffer=buffer_q,
                        desired_window=desired_window,
                        global_segment=global_segment,
                        global_reference=global_reference,
                        desired_path=desired_path,
                        start=start,
                        execution_count=execution_count,
                        previous_executed_q=previous_executed_q,
                        tail_mode=tail_mode,
                        is_diffusion=True,
                        clipping=clipping,
                        robot=robot,
                        joint_names=joint_names,
                        ee_link=ee_link,
                        lower=lower,
                        upper=upper,
                        drawing_weights=drawing_weights,
                        score_config=score_config,
                        args=args,
                    )
                    row: Dict[str, Any] = {
                        "candidate_index": candidate_index,
                        "candidate_type": "diffusion_refined",
                        "base_sample_index": base_sample_index,
                        "candidate_seed": sample_seeds[base_sample_index],
                        "alpha": float(alpha),
                        "generated_residual_rms": float(
                            np.sqrt(np.mean(np.square(generated_physical)))
                        ),
                        "scaled_residual_rms": float(
                            np.sqrt(np.mean(np.square(scaled_residual)))
                        ),
                        **metrics,
                        "_candidate_q": candidate_q,
                    }
                    accepted, reason, safety_improved = safety_status(
                        row, buffer_row, args
                    )
                    row["accepted"] = accepted
                    row["candidate_rejection_reason"] = reason
                    row["safety_improving_candidate"] = safety_improved
                    rows.append(row)
                    candidate_index += 1

        selected_index = select_candidate(rows, args.selector)
        selected = rows[selected_index]
        selected_q = (
            buffer_q
            if selected_index == 0
            else np.asarray(selected["_candidate_q"], dtype=np.float32)
        )
        selected_prefix = selected_q[:execution_count].copy()
        if selected_prefix.shape != (execution_count, JOINT_DIM):
            raise AssertionError("Executed prefix has an invalid shape")
        executed_chunks.append(selected_prefix)
        selected_next_buffer = np.asarray(
            selected["_next_buffer"], dtype=np.float32
        )
        if selected_next_buffer.shape != (horizon, JOINT_DIM):
            raise AssertionError("Action buffer must remain length H after extension")
        finite_array(selected_next_buffer, "selected previewed next action buffer")

        accepted_diffusion = [
            row for row in rows[1:] if int(row.get("accepted", 0)) == 1
        ]
        fallback_reason = ""
        if method != "buffer_only" and selected_index == 0:
            fallback_reason = (
                "all_diffusion_candidates_rejected"
                if not accepted_diffusion
                else "buffer_has_best_practical_score"
            )
        selected_diffusion = selected_index != 0
        selected_unsafe = int(
            selected_diffusion and int(selected.get("accepted", 0)) != 1
        )
        planning_row: Dict[str, Any] = {
            "path_name": path_name,
            "rollout_method": method,
            "selected_tail_mode": tail_mode,
            "selector": args.selector,
            "planning_cycle_index": cycle_index,
            "current_index": start,
            "execution_count": execution_count,
            "global_window_first_index": int(indices[0]),
            "global_window_last_index": int(indices[-1]),
            "candidate_count": len(rows),
            "accepted_diffusion_candidate_count": len(accepted_diffusion),
            "selected_candidate_index": selected_index,
            "selected_candidate_type": selected["candidate_type"],
            "selected_diffusion": int(selected_diffusion),
            "selected_alpha": float(selected["alpha"]),
            "selected_base_sample_index": int(selected["base_sample_index"]),
            "selected_candidate_seed": int(selected["candidate_seed"]),
            "selected_diffusion_unsafe": selected_unsafe,
            "selected_safety_improving": int(
                selected.get("safety_improving_candidate", 0)
            ),
            "buffer_fallback": int(method != "buffer_only" and selected_index == 0),
            "buffer_fallback_reason": fallback_reason,
            "candidate_rejection_summary": summarize_rejections(rows),
            "buffer_larger_picture_score": buffer_row["larger_picture_score"],
            "selected_larger_picture_score": selected["larger_picture_score"],
            "selected_candidate_rejection_reason": selected.get(
                "candidate_rejection_reason", ""
            ),
            "initial_buffer_clipped_value_count": int(
                initial_clipping["clipped_value_count"]
            ),
            **{
                f"selected_{key}": value
                for key, value in selected.items()
                if not key.startswith("_")
                and key
                not in {
                    "candidate_index",
                    "candidate_type",
                    "base_sample_index",
                    "candidate_seed",
                    "alpha",
                    "accepted",
                    "candidate_rejection_reason",
                }
            },
        }
        planning_rows.append(planning_row)
        if args.save_candidate_details:
            for row in rows:
                candidate_rows.append(
                    {
                        "path_name": path_name,
                        "rollout_method": method,
                        "selected_tail_mode": tail_mode,
                        "planning_cycle_index": cycle_index,
                        "current_index": start,
                        **{
                            key: value
                            for key, value in row.items()
                            if not key.startswith("_")
                        },
                        "selected": int(row is selected),
                    }
                )
        buffer_q = selected_next_buffer

    executed_q = np.concatenate(executed_chunks, axis=0)
    if executed_q.shape != (trajectory_length, JOINT_DIM):
        raise AssertionError(
            f"Final trajectory must have shape ({trajectory_length},6), got {executed_q.shape}"
        )
    if sum(int(row["execution_count"]) for row in planning_rows) != trajectory_length:
        raise AssertionError("Rollout indexing did not execute exactly T states")
    metrics, ee, errors, joint_drift, cartesian_drift = full_trajectory_metrics(
        q=executed_q,
        desired_path=desired_path,
        expert_q_evaluation_only=expert_q_evaluation_only,
        q_start=q_start,
        global_reference=global_reference,
        robot=robot,
        joint_names=joint_names,
        ee_link=ee_link,
        lower=lower,
        upper=upper,
        drawing_weights=drawing_weights,
        planning_rows=planning_rows,
    )
    return {
        "method": method,
        "q": executed_q,
        "ee": ee,
        "cartesian_errors": errors,
        "global_joint_drift_over_time": joint_drift,
        "global_cartesian_drift_over_time": cartesian_drift,
        "planning_rows": planning_rows,
        "candidate_rows": candidate_rows,
        "metrics": metrics,
    }


def full_trajectory_metrics(
    *,
    q: np.ndarray,
    desired_path: np.ndarray,
    expert_q_evaluation_only: np.ndarray,
    q_start: np.ndarray,
    global_reference: np.ndarray,
    robot: Any,
    joint_names: Sequence[str],
    ee_link: str,
    lower: np.ndarray,
    upper: np.ndarray,
    drawing_weights: Any,
    planning_rows: Sequence[Dict[str, Any]],
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    region, ee, errors = sequence_metrics(
        q=q,
        desired=desired_path,
        previous_q=q_start,
        robot=robot,
        joint_names=joint_names,
        ee_link=ee_link,
        lower=lower,
        upper=upper,
        drawing_weights=drawing_weights,
    )
    global_ee = fk_positions(robot, joint_names, ee_link, global_reference)
    joint_drift = np.linalg.norm(q - global_reference, axis=1)
    cartesian_drift = np.linalg.norm(ee - global_ee, axis=1)
    planning_boundaries = np.asarray(
        [
            float(row["selected_current_boundary_max_abs_joint_step"])
            for row in planning_rows
            if int(row["planning_cycle_index"]) > 0
        ],
        dtype=np.float64,
    )
    future_boundaries = np.asarray(
        [
            max(
                float(row["selected_shift_boundary_max_abs_joint_step"]),
                float(row["selected_retained_shift_boundary_max_abs_joint_step"]),
            )
            for row in planning_rows
        ],
        dtype=np.float64,
    )
    selected_alphas = Counter(
        float(row["selected_alpha"])
        for row in planning_rows
        if int(row["selected_diffusion"]) == 1
    )
    limit_count, limit_magnitude = limit_metrics(q, lower, upper)
    metrics: Dict[str, Any] = {
        "mean_cartesian_error": region["mean_cartesian_error"],
        "rms_cartesian_error": region["rms_cartesian_error"],
        "max_cartesian_error": region["max_cartesian_error"],
        "drawing_total_cost": region["drawing_cost"],
        **{
            key: region[key]
            for key in (
                "mean_joint_velocity",
                "max_joint_velocity",
                "mean_joint_acceleration",
                "max_joint_acceleration",
                "mean_joint_jerk",
                "max_joint_jerk",
                "max_joint_step",
            )
        },
        "joint_limit_violation_count": int(limit_count),
        "joint_limit_violation_magnitude": float(limit_magnitude),
        "full_joint_rmse_vs_expert": float(
            np.sqrt(np.mean(np.square(q - expert_q_evaluation_only)))
        ),
        "mean_planning_boundary_discontinuity": float(
            np.mean(planning_boundaries)
        )
        if planning_boundaries.size
        else 0.0,
        "max_planning_boundary_discontinuity": float(
            np.max(planning_boundaries)
        )
        if planning_boundaries.size
        else 0.0,
        "mean_future_shift_boundary_discontinuity": float(
            np.mean(future_boundaries)
        )
        if future_boundaries.size
        else 0.0,
        "max_future_shift_boundary_discontinuity": float(
            np.max(future_boundaries)
        )
        if future_boundaries.size
        else 0.0,
        "total_global_reference_joint_drift": float(np.sum(joint_drift)),
        "mean_global_reference_joint_drift": float(np.mean(joint_drift)),
        "max_global_reference_joint_drift": float(np.max(joint_drift)),
        "total_global_reference_cartesian_drift": float(np.sum(cartesian_drift)),
        "mean_global_reference_cartesian_drift": float(np.mean(cartesian_drift)),
        "max_global_reference_cartesian_drift": float(np.max(cartesian_drift)),
        "extension_clipping_count": int(
            sum(
                int(row["selected_extension_clipped_value_count"])
                for row in planning_rows
            )
        ),
        "extension_clipping_magnitude": float(
            sum(
                float(row["selected_extension_clipping_total_abs"])
                for row in planning_rows
            )
        ),
        "planning_cycle_count": len(planning_rows),
        "diffusion_selection_fraction": float(
            np.mean([int(row["selected_diffusion"]) for row in planning_rows])
        ),
        "selected_alpha_distribution": json.dumps(
            {str(key): value for key, value in sorted(selected_alphas.items())},
            sort_keys=True,
        ),
        "unsafe_diffusion_selection_count": int(
            sum(int(row["selected_diffusion_unsafe"]) for row in planning_rows)
        ),
        "buffer_fallback_count": int(
            sum(int(row["buffer_fallback"]) for row in planning_rows)
        ),
        "safety_improving_selection_count": int(
            sum(int(row["selected_safety_improving"]) for row in planning_rows)
        ),
    }
    return metrics, ee, errors, joint_drift, cartesian_drift


FULL_METRICS = (
    "mean_cartesian_error",
    "rms_cartesian_error",
    "max_cartesian_error",
    "drawing_total_cost",
    "mean_joint_velocity",
    "max_joint_velocity",
    "mean_joint_acceleration",
    "max_joint_acceleration",
    "mean_joint_jerk",
    "max_joint_jerk",
    "max_joint_step",
    "joint_limit_violation_count",
    "joint_limit_violation_magnitude",
    "full_joint_rmse_vs_expert",
    "mean_planning_boundary_discontinuity",
    "max_planning_boundary_discontinuity",
    "mean_future_shift_boundary_discontinuity",
    "max_future_shift_boundary_discontinuity",
    "total_global_reference_joint_drift",
    "mean_global_reference_joint_drift",
    "max_global_reference_joint_drift",
    "total_global_reference_cartesian_drift",
    "mean_global_reference_cartesian_drift",
    "max_global_reference_cartesian_drift",
    "extension_clipping_count",
    "extension_clipping_magnitude",
    "planning_cycle_count",
    "diffusion_selection_fraction",
    "unsafe_diffusion_selection_count",
    "buffer_fallback_count",
    "safety_improving_selection_count",
)


def write_joint_csv(path: Path, q: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["t", "q1", "q2", "q3", "q4", "q5", "q6"])
        for index, row in enumerate(q):
            writer.writerow([index, *[f"{float(value):.10f}" for value in row]])


def write_xyz_csv(path: Path, xyz: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["t", "x", "y", "z"])
        for index, row in enumerate(xyz):
            writer.writerow([index, *[f"{float(value):.10f}" for value in row]])


def write_records_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def paired_comparisons(
    per_path_rows: Sequence[Dict[str, Any]],
    diffusion_methods: Sequence[str],
) -> List[Dict[str, Any]]:
    lookup = {
        (str(row["path_name"]), str(row["rollout_method"])): row
        for row in per_path_rows
    }
    paths = sorted({str(row["path_name"]) for row in per_path_rows})
    output: List[Dict[str, Any]] = []
    for path_name in paths:
        baseline = lookup[(path_name, "buffer_only")]
        for method in diffusion_methods:
            candidate = lookup[(path_name, method)]
            row: Dict[str, Any] = {
                "path_name": path_name,
                "rollout_method": method,
            }
            for metric in FULL_METRICS:
                base_value = float(baseline[metric])
                candidate_value = float(candidate[metric])
                difference = candidate_value - base_value
                row[f"buffer_{metric}"] = base_value
                row[f"candidate_{metric}"] = candidate_value
                row[f"absolute_difference_{metric}"] = difference
                row[f"percentage_difference_{metric}"] = (
                    100.0 * difference / abs(base_value)
                    if abs(base_value) > EPS
                    else float("nan")
                )
            output.append(row)
    return output


def aggregate_comparisons(
    paired_rows: Sequence[Dict[str, Any]],
    diffusion_methods: Sequence[str],
) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for method in diffusion_methods:
        group = [row for row in paired_rows if row["rollout_method"] == method]
        for metric in FULL_METRICS:
            changes = np.asarray(
                [float(row[f"absolute_difference_{metric}"]) for row in group],
                dtype=np.float64,
            )
            improved = changes < -EPS
            worsened = changes > EPS
            tied = ~(improved | worsened)
            output.append(
                {
                    "rollout_method": method,
                    "metric": metric,
                    "path_count": len(group),
                    "mean": float(np.mean(changes)),
                    "median": float(np.median(changes)),
                    "standard_deviation": float(np.std(changes)),
                    "minimum": float(np.min(changes)),
                    "maximum": float(np.max(changes)),
                    "improved_count": int(np.sum(improved)),
                    "worsened_count": int(np.sum(worsened)),
                    "tied_count": int(np.sum(tied)),
                    "percentage_improved": float(100.0 * np.mean(improved)),
                }
            )
    return output


def tail_mode_summary(
    per_path_rows: Sequence[Dict[str, Any]],
    diffusion_methods: Sequence[str],
) -> List[Dict[str, Any]]:
    lookup = {
        (str(row["path_name"]), str(row["rollout_method"])): row
        for row in per_path_rows
    }
    paths = sorted({str(row["path_name"]) for row in per_path_rows})
    output: List[Dict[str, Any]] = []
    for method in diffusion_methods:
        rows = [lookup[(path, method)] for path in paths]
        baseline = [lookup[(path, "buffer_only")] for path in paths]
        cart_changes = np.asarray(
            [
                float(row["mean_cartesian_error"])
                - float(base["mean_cartesian_error"])
                for row, base in zip(rows, baseline)
            ]
        )
        output.append(
            {
                "tail_mode": method,
                "path_count": len(rows),
                "mean_cartesian_error": float(
                    np.mean([float(row["mean_cartesian_error"]) for row in rows])
                ),
                "mean_drawing_total_cost": float(
                    np.mean([float(row["drawing_total_cost"]) for row in rows])
                ),
                "mean_max_joint_step": float(
                    np.mean([float(row["max_joint_step"]) for row in rows])
                ),
                "mean_joint_acceleration": float(
                    np.mean([float(row["mean_joint_acceleration"]) for row in rows])
                ),
                "mean_global_reference_joint_drift": float(
                    np.mean(
                        [float(row["mean_global_reference_joint_drift"]) for row in rows]
                    )
                ),
                "mean_global_reference_cartesian_drift": float(
                    np.mean(
                        [
                            float(row["mean_global_reference_cartesian_drift"])
                            for row in rows
                        ]
                    )
                ),
                "mean_planning_boundary_discontinuity": float(
                    np.mean(
                        [
                            float(row["mean_planning_boundary_discontinuity"])
                            for row in rows
                        ]
                    )
                ),
                "mean_future_shift_boundary_discontinuity": float(
                    np.mean(
                        [
                            float(row["mean_future_shift_boundary_discontinuity"])
                            for row in rows
                        ]
                    )
                ),
                "extension_clipping_count": int(
                    sum(int(row["extension_clipping_count"]) for row in rows)
                ),
                "paths_improved": int(np.sum(cart_changes < -EPS)),
                "paths_worsened": int(np.sum(cart_changes > EPS)),
                "paths_tied": int(np.sum(np.abs(cart_changes) <= EPS)),
                "diffusion_selection_fraction": float(
                    np.mean(
                        [float(row["diffusion_selection_fraction"]) for row in rows]
                    )
                ),
            }
        )
    return output


def method_colors() -> Dict[str, str]:
    return {
        "buffer_only": "black",
        "full_selected_tail": "tab:red",
        "base_tail": "tab:green",
        "decayed_selected_tail": "tab:blue",
        "global_anchored_tail": "tab:purple",
    }


def save_path_plots(
    path_dir: Path,
    desired: np.ndarray,
    results: Dict[str, Dict[str, Any]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = method_colors()
    figure = plt.figure(figsize=(9, 7))
    axis = figure.add_subplot(111, projection="3d")
    axis.plot(*desired.T, label="desired", color="tab:orange", linewidth=2)
    for method, result in results.items():
        axis.plot(*result["ee"].T, label=method, color=colors[method], alpha=0.8)
    axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(path_dir / "desired_vs_executed_cartesian.png", dpi=160)
    plt.close(figure)

    def line_plot(filename: str, ylabel: str, series: Dict[str, np.ndarray]) -> None:
        figure, axis = plt.subplots(figsize=(10, 5))
        for method, values in series.items():
            axis.plot(values, label=method, color=colors[method])
        axis.set_xlabel("Timestep or planning cycle")
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=7)
        figure.tight_layout()
        figure.savefig(path_dir / filename, dpi=160)
        plt.close(figure)

    line_plot(
        "cartesian_error_over_time.png",
        "Cartesian error",
        {method: result["cartesian_errors"] for method, result in results.items()},
    )
    baseline_error = results["buffer_only"]["cartesian_errors"]
    line_plot(
        "cumulative_cartesian_error_difference_vs_buffer.png",
        "Cumulative error difference",
        {
            method: np.cumsum(result["cartesian_errors"] - baseline_error)
            for method, result in results.items()
            if method != "buffer_only"
        },
    )
    line_plot(
        "maximum_joint_step_over_time.png",
        "Maximum absolute joint step",
        {
            method: np.concatenate(
                ([0.0], np.max(np.abs(np.diff(result["q"], axis=0)), axis=1))
            )
            for method, result in results.items()
        },
    )
    line_plot(
        "selected_alpha_by_cycle.png",
        "Selected alpha",
        {
            method: np.asarray(
                [float(row["selected_alpha"]) for row in result["planning_rows"]]
            )
            for method, result in results.items()
        },
    )
    line_plot(
        "global_reference_joint_drift_over_time.png",
        "Joint drift from global reference",
        {
            method: result["global_joint_drift_over_time"]
            for method, result in results.items()
        },
    )
    line_plot(
        "global_reference_cartesian_drift_over_time.png",
        "Cartesian drift from global reference",
        {
            method: result["global_cartesian_drift_over_time"]
            for method, result in results.items()
        },
    )
    line_plot(
        "extension_clipping_by_cycle.png",
        "Clipped extension values",
        {
            method: np.asarray(
                [
                    int(row["selected_extension_clipped_value_count"])
                    for row in result["planning_rows"]
                ]
            )
            for method, result in results.items()
        },
    )

    figure, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for method, result in results.items():
        current = [
            float(row["selected_current_boundary_max_abs_joint_step"])
            for row in result["planning_rows"]
        ]
        future = [
            max(
                float(row["selected_shift_boundary_max_abs_joint_step"]),
                float(row["selected_retained_shift_boundary_max_abs_joint_step"]),
            )
            for row in result["planning_rows"]
        ]
        axes[0].plot(current, label=method, color=colors[method])
        axes[1].plot(future, label=method, color=colors[method])
    axes[0].set_ylabel("Current boundary")
    axes[1].set_ylabel("Future shift boundary")
    axes[1].set_xlabel("Planning cycle")
    for axis in axes:
        axis.grid(True, alpha=0.25)
    axes[0].legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(path_dir / "current_and_future_shift_boundaries.png", dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(JOINT_DIM, 1, figsize=(11, 12), sharex=True)
    for joint, axis in enumerate(axes):
        for method, result in results.items():
            axis.plot(result["q"][:, joint], label=method, color=colors[method])
        axis.set_ylabel(f"q{joint + 1}")
        axis.grid(True, alpha=0.2)
    axes[0].legend(fontsize=7)
    axes[-1].set_xlabel("Timestep")
    figure.tight_layout()
    figure.savefig(path_dir / "joint_trajectory_comparison.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(10, 5))
    for method, result in results.items():
        if method == "buffer_only" or not result["planning_rows"]:
            continue
        weights = json.loads(result["planning_rows"][0]["selected_tail_blending_weights"])
        axis.plot(weights, label=method, color=colors[method])
    axis.set_xlabel("Retained-tail offset")
    axis.set_ylabel("Selected-tail weight")
    axis.grid(True, alpha=0.25)
    axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(path_dir / "tail_blending_weights.png", dpi=160)
    plt.close(figure)


def save_aggregate_plots(
    output_dir: Path,
    per_path_rows: Sequence[Dict[str, Any]],
    diffusion_methods: Sequence[str],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir = output_dir / "aggregate_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    colors = method_colors()
    lookup = {
        (str(row["path_name"]), str(row["rollout_method"])): row
        for row in per_path_rows
    }
    paths = sorted({str(row["path_name"]) for row in per_path_rows})

    def paired_scatter(metric: str, filename: str, label: str) -> None:
        figure, axis = plt.subplots(figsize=(7, 6))
        all_values: List[float] = []
        for method in diffusion_methods:
            x = [float(lookup[(path, "buffer_only")][metric]) for path in paths]
            y = [float(lookup[(path, method)][metric]) for path in paths]
            all_values.extend(x)
            all_values.extend(y)
            axis.scatter(x, y, label=method, color=colors[method], alpha=0.7)
        low, high = min(all_values), max(all_values)
        axis.plot([low, high], [low, high], "--", color="black")
        axis.set_xlabel(f"Buffer-only {label}")
        axis.set_ylabel(f"Tail-mode {label}")
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=7)
        figure.tight_layout()
        figure.savefig(plots_dir / filename, dpi=160)
        plt.close(figure)

    paired_scatter(
        "mean_cartesian_error",
        "paired_cartesian_error_scatter_by_tail_mode.png",
        "Cartesian error",
    )
    paired_scatter(
        "drawing_total_cost",
        "paired_drawing_cost_scatter_by_tail_mode.png",
        "drawing cost",
    )
    labels = list(diffusion_methods)
    improved: List[int] = []
    worsened: List[int] = []
    for method in diffusion_methods:
        changes = [
            float(lookup[(path, method)]["mean_cartesian_error"])
            - float(lookup[(path, "buffer_only")]["mean_cartesian_error"])
            for path in paths
        ]
        improved.append(sum(value < -EPS for value in changes))
        worsened.append(sum(value > EPS for value in changes))
    x = np.arange(len(labels))
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.bar(x - 0.18, improved, 0.36, label="improved")
    axis.bar(x + 0.18, worsened, 0.36, label="worsened")
    axis.set_xticks(x, labels, rotation=15, ha="right")
    axis.set_ylabel("Path count")
    axis.legend()
    figure.tight_layout()
    figure.savefig(plots_dir / "paths_improved_vs_worsened.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 6))
    for method in diffusion_methods:
        drift = [
            float(lookup[(path, method)]["mean_global_reference_cartesian_drift"])
            for path in paths
        ]
        improvement = [
            float(lookup[(path, "buffer_only")]["mean_cartesian_error"])
            - float(lookup[(path, method)]["mean_cartesian_error"])
            for path in paths
        ]
        axis.scatter(drift, improvement, label=method, color=colors[method], alpha=0.7)
    axis.axhline(0.0, color="black", linestyle="--")
    axis.set_xlabel("Global-reference Cartesian drift")
    axis.set_ylabel("Cartesian improvement vs buffer")
    axis.grid(True, alpha=0.25)
    axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(plots_dir / "global_drift_vs_cartesian_improvement.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 6))
    for method in diffusion_methods:
        clipping = [
            float(lookup[(path, method)]["extension_clipping_count"])
            for path in paths
        ]
        error = [
            float(lookup[(path, method)]["mean_cartesian_error"])
            for path in paths
        ]
        axis.scatter(clipping, error, label=method, color=colors[method], alpha=0.7)
    axis.set_xlabel("Extension clipped values")
    axis.set_ylabel("Full-trajectory Cartesian error")
    axis.grid(True, alpha=0.25)
    axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(plots_dir / "extension_clipping_vs_error.png", dpi=160)
    plt.close(figure)

    metrics = (
        "mean_cartesian_error",
        "drawing_total_cost",
        "mean_global_reference_cartesian_drift",
        "mean_planning_boundary_discontinuity",
    )
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    for axis, metric in zip(axes.flat, metrics):
        values = [
            np.mean([float(lookup[(path, method)][metric]) for path in paths])
            for method in diffusion_methods
        ]
        axis.bar(np.arange(len(diffusion_methods)), values, color=[colors[m] for m in diffusion_methods])
        axis.set_xticks(np.arange(len(diffusion_methods)), diffusion_methods, rotation=18, ha="right")
        axis.set_title(metric)
        axis.grid(True, axis="y", alpha=0.2)
    figure.tight_layout()
    figure.savefig(plots_dir / "tail_mode_metric_comparison.png", dpi=160)
    plt.close(figure)


def scientific_decisions(
    per_path_rows: Sequence[Dict[str, Any]],
    planning_rows: Sequence[Dict[str, Any]],
    diffusion_methods: Sequence[str],
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, Any]], str, List[str]]:
    lookup = {
        (str(row["path_name"]), str(row["rollout_method"])): row
        for row in per_path_rows
    }
    paths = sorted({str(row["path_name"]) for row in per_path_rows})
    decisions: List[Dict[str, Any]] = []
    for method in diffusion_methods:
        baseline = [lookup[(path, "buffer_only")] for path in paths]
        candidate = [lookup[(path, method)] for path in paths]
        mean = lambda rows, key: float(np.mean([float(row[key]) for row in rows]))
        improved_paths = sum(
            float(row["mean_cartesian_error"])
            < float(base["mean_cartesian_error"])
            or float(row["drawing_total_cost"])
            < float(base["drawing_total_cost"])
            for row, base in zip(candidate, baseline)
        )
        method_cycles = [
            row for row in planning_rows if row["rollout_method"] == method
        ]
        criteria = {
            "cartesian_improved": mean(candidate, "mean_cartesian_error")
            < mean(baseline, "mean_cartesian_error"),
            "drawing_improved": mean(candidate, "drawing_total_cost")
            < mean(baseline, "drawing_total_cost"),
            "majority_paths_improved": improved_paths > len(paths) / 2,
            "max_joint_step_safe": max(
                float(row["max_joint_step"]) for row in candidate
            )
            <= args.material_safety_ratio
            * max(max(float(row["max_joint_step"]) for row in baseline), EPS),
            "planning_boundary_safe": max(
                float(row["max_planning_boundary_discontinuity"])
                for row in candidate
            )
            <= args.material_safety_ratio
            * max(
                max(
                    float(row["max_planning_boundary_discontinuity"])
                    for row in baseline
                ),
                EPS,
            ),
            "future_shift_boundary_safe": max(
                float(row["max_future_shift_boundary_discontinuity"])
                for row in candidate
            )
            <= args.material_safety_ratio
            * max(
                max(
                    float(row["max_future_shift_boundary_discontinuity"])
                    for row in baseline
                ),
                EPS,
            ),
            "joint_limits_safe": sum(
                int(row["joint_limit_violation_count"]) for row in candidate
            )
            <= sum(int(row["joint_limit_violation_count"]) for row in baseline),
            "extension_clipping_safe": sum(
                int(row["extension_clipping_count"]) for row in candidate
            )
            <= max(
                sum(int(row["extension_clipping_count"]) for row in baseline),
                args.max_extension_clipped_values * len(method_cycles),
            ),
            "no_unsafe_selection": sum(
                int(row["selected_diffusion_unsafe"]) for row in method_cycles
            )
            == 0,
            "global_drift_bounded": mean(
                candidate, "mean_global_reference_joint_drift"
            )
            <= args.max_reference_joint_drift
            and mean(candidate, "mean_global_reference_cartesian_drift")
            <= args.max_reference_cartesian_drift,
        }
        decisions.append(
            {
                "tail_mode": method,
                "passed": int(all(criteria.values())),
                "failed_criteria": ";".join(
                    key for key, value in criteria.items() if not value
                ),
                **{key: int(value) for key, value in criteria.items()},
            }
        )

    passing = [row["tail_mode"] for row in decisions if int(row["passed"]) == 1]
    candidates = passing if passing else list(diffusion_methods)
    best = min(
        candidates,
        key=lambda method: (
            np.mean(
                [
                    float(lookup[(path, method)]["mean_cartesian_error"])
                    for path in paths
                ]
            ),
            np.mean(
                [
                    float(lookup[(path, method)]["drawing_total_cost"])
                    for path in paths
                ]
            ),
        ),
    )
    causes: List[str] = []
    if not passing:
        method_rows = [lookup[(path, best)] for path in paths]
        cycles = [row for row in planning_rows if row["rollout_method"] == best]
        selection_fraction = float(
            np.mean([int(row["selected_diffusion"]) for row in cycles])
        )
        if selection_fraction < 0.10:
            causes.append("diffusion prefix corrections are not large enough to matter globally")
        if selection_fraction >= 0.10 and np.mean(
            [float(row["mean_cartesian_error"]) for row in method_rows]
        ) >= np.mean(
            [float(lookup[(path, "buffer_only")]["mean_cartesian_error"]) for path in paths]
        ):
            causes.append("candidate scoring still lacks sufficient lookahead")
            causes.append("candidate generation is unsuitable for recursive use")
        if np.mean(
            [float(row["mean_global_reference_cartesian_drift"]) for row in method_rows]
        ) > args.max_reference_cartesian_drift:
            causes.append("global reference is too weak")
        elif selection_fraction < 0.05 and (
            args.w_reference_joint + args.w_reference_cartesian
        ) > args.w_prefix:
            causes.append("global reference is too restrictive")
        if sum(int(row["extension_clipping_count"]) for row in method_rows) > 0:
            causes.append("extension remains unstable")
        if selection_fraction > 0.0:
            causes.append(
                "diffusion must be retrained on recursively generated action-buffer states"
            )
    return decisions, best, list(dict.fromkeys(causes))


def main() -> int:
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)
    device = resolve_device(args.device)
    score_config = score_weights(args)
    drawing_weights = default_weights()
    stats = load_stats(args.stats_npz)
    residual_mean = np.asarray(stats["residual_mean"], dtype=np.float32)
    residual_std = np.asarray(stats["residual_std"], dtype=np.float32)
    condition_mean = np.asarray(stats["condition_mean"], dtype=np.float32)
    condition_std = np.asarray(stats["condition_std"], dtype=np.float32)
    if residual_mean.shape != (JOINT_DIM,) or residual_std.shape != (JOINT_DIM,):
        raise ValueError("Residual statistics must have shape (6,)")
    if condition_mean.shape != (CONDITION_DIM,) or condition_std.shape != (
        CONDITION_DIM,
    ):
        raise ValueError(f"Condition statistics must have shape ({CONDITION_DIM},)")
    if np.any(residual_std <= 0.0) or np.any(condition_std <= 0.0):
        raise ValueError("Normalization standard deviations must be positive")

    checkpoint = torch_load_checkpoint(args.checkpoint, device)
    if int(checkpoint["condition_dim"]) != CONDITION_DIM:
        raise ValueError(f"Checkpoint condition_dim must be {CONDITION_DIM}")
    if int(checkpoint["target_dim"]) != JOINT_DIM:
        raise ValueError("Checkpoint target_dim must be 6")
    if int(checkpoint["horizon"]) != args.prediction_horizon:
        raise ValueError("Checkpoint horizon differs from prediction_horizon")
    model, call_variant, _ = instantiate_checkpoint_model(checkpoint, device)
    model.eval()
    diffusion_config = diffusion_config_from_checkpoint(
        checkpoint, args.num_diffusion_steps
    )
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
    require_keys(
        data,
        ("desired_paths", "expert_q", "q_start", "path_names"),
        "test trajectories",
    )
    desired_paths = finite_array(data["desired_paths"], "desired_paths").astype(
        np.float32
    )
    expert_all = finite_array(data["expert_q"], "expert_q").astype(np.float32)
    q_start_all = finite_array(data["q_start"], "q_start").astype(np.float32)
    path_names = decode_names(data["path_names"])
    if desired_paths.ndim != 3 or desired_paths.shape[2] != 3:
        raise ValueError("desired_paths must have shape (N,T,3)")
    if expert_all.shape != (
        desired_paths.shape[0],
        desired_paths.shape[1],
        JOINT_DIM,
    ):
        raise ValueError("expert_q must have shape (N,T,6)")
    if q_start_all.shape != (desired_paths.shape[0], JOINT_DIM):
        raise ValueError("q_start must have shape (N,6)")
    if len(path_names) != desired_paths.shape[0]:
        raise ValueError("path_names length differs from trajectory count")

    robot, joint_names, ee_link = load_fk_context(None, None)
    if len(joint_names) != JOINT_DIM:
        raise ValueError("Exactly six active xMateCR7 joints are required")
    lower, upper = extract_joint_limits(robot, joint_names)
    max_paths = (
        len(path_names)
        if args.max_paths is None
        else min(args.max_paths, len(path_names))
    )
    methods = ("buffer_only", *tuple(args.tail_modes))
    per_path_rows: List[Dict[str, Any]] = []
    planning_rows: List[Dict[str, Any]] = []
    candidate_rows: List[Dict[str, Any]] = []

    print(f"Larger-picture score weights: {json.dumps(asdict(score_config), sort_keys=True)}")
    print(f"Drawing-aware weights: {json.dumps(weights_record(drawing_weights), sort_keys=True)}")
    print(
        f"Architecture: H={args.prediction_horizon}, E={args.execution_horizon}, "
        f"extension={args.tail_extension}, selector={args.selector}"
    )
    print(f"Tail modes: {', '.join(args.tail_modes)}")
    print(f"Global reference: fixed MLP predicted_q trajectory, T={desired_paths.shape[1]}")
    print("Residual initialization: normalized representation of physical zero")

    with torch.no_grad():
        for path_index in range(max_paths):
            path_name = path_names[path_index]
            global_reference = read_predicted_q_csv(
                args.prior_dir / safe_path_name(path_name) / "predicted_q.csv",
                expected_steps=desired_paths.shape[1],
            ).astype(np.float32)
            finite_array(global_reference, "fixed global bootstrap prior")
            if global_reference.shape != (
                desired_paths.shape[1],
                JOINT_DIM,
            ):
                raise ValueError("Global reference must have shape (T,6)")
            path_results: Dict[str, Dict[str, Any]] = {}
            for method in methods:
                result = run_method(
                    method=method,
                    path_index=path_index,
                    path_name=path_name,
                    desired_path=desired_paths[path_index],
                    expert_q_evaluation_only=expert_all[path_index],
                    q_start=q_start_all[path_index],
                    global_reference=global_reference,
                    model=model,
                    call_variant=call_variant,
                    schedule=schedule,
                    residual_mean=residual_mean,
                    residual_std=residual_std,
                    condition_mean=condition_mean,
                    condition_std=condition_std,
                    robot=robot,
                    joint_names=joint_names,
                    ee_link=ee_link,
                    lower=lower,
                    upper=upper,
                    drawing_weights=drawing_weights,
                    score_config=score_config,
                    args=args,
                )
                path_results[method] = result
                per_path_rows.append(
                    {
                        "path_name": path_name,
                        "rollout_method": method,
                        **result["metrics"],
                    }
                )
                planning_rows.extend(result["planning_rows"])
                candidate_rows.extend(result["candidate_rows"])

            path_dir = args.output_dir / "trajectories" / safe_path_name(path_name)
            write_xyz_csv(path_dir / "desired_path.csv", desired_paths[path_index])
            write_joint_csv(path_dir / "expert_q.csv", expert_all[path_index])
            write_joint_csv(path_dir / "global_reference_q.csv", global_reference)
            expert_ee = fk_positions(robot, joint_names, ee_link, expert_all[path_index])
            global_reference_ee = fk_positions(
                robot, joint_names, ee_link, global_reference
            )
            write_xyz_csv(path_dir / "expert_ee.csv", expert_ee)
            write_xyz_csv(path_dir / "global_reference_ee.csv", global_reference_ee)
            for method, result in path_results.items():
                write_joint_csv(path_dir / f"{method}_q.csv", result["q"])
                write_xyz_csv(path_dir / f"{method}_ee.csv", result["ee"])
            save_path_plots(path_dir, desired_paths[path_index], path_results)
            print(f"Completed {path_index + 1}/{max_paths} paths: {path_name}")

    paired_rows = paired_comparisons(per_path_rows, args.tail_modes)
    aggregate_rows = aggregate_comparisons(paired_rows, args.tail_modes)
    tail_rows = tail_mode_summary(per_path_rows, args.tail_modes)
    decisions, best_mode, failure_causes = scientific_decisions(
        per_path_rows, planning_rows, args.tail_modes, args
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_records_csv(args.output_dir / "per_path_summary.csv", per_path_rows)
    write_records_csv(args.output_dir / "paired_comparison.csv", paired_rows)
    write_records_csv(args.output_dir / "tail_mode_comparison.csv", tail_rows)
    write_records_csv(args.output_dir / "aggregate_summary.csv", aggregate_rows)
    write_records_csv(args.output_dir / "planning_cycle_details.csv", planning_rows)
    write_records_csv(args.output_dir / "scientific_decision.csv", decisions)
    if args.save_candidate_details:
        write_records_csv(args.output_dir / "candidate_details.csv", candidate_rows)
    save_aggregate_plots(args.output_dir, per_path_rows, args.tail_modes)

    lookup = {
        method: [row for row in per_path_rows if row["rollout_method"] == method]
        for method in methods
    }

    def mean(method: str, metric: str) -> float:
        return float(np.mean([float(row[metric]) for row in lookup[method]]))

    buffer_cart = mean("buffer_only", "mean_cartesian_error")
    print("\nGlobal-anchored recursive rollout summary")
    print(f"buffer-only aggregate Cartesian error: {buffer_cart:.8e}")
    for method in args.tail_modes:
        cart = mean(method, "mean_cartesian_error")
        relative = 100.0 * (cart - buffer_cart) / max(abs(buffer_cart), EPS)
        baseline_by_path = {
            row["path_name"]: row for row in lookup["buffer_only"]
        }
        improved = sum(
            float(row["mean_cartesian_error"])
            < float(baseline_by_path[row["path_name"]]["mean_cartesian_error"])
            - EPS
            for row in lookup[method]
        )
        worsened = sum(
            float(row["mean_cartesian_error"])
            > float(baseline_by_path[row["path_name"]]["mean_cartesian_error"])
            + EPS
            for row in lookup[method]
        )
        tied = max_paths - improved - worsened
        alpha_counter: Counter[float] = Counter(
            float(row["selected_alpha"])
            for row in planning_rows
            if row["rollout_method"] == method
            and int(row["selected_diffusion"]) == 1
        )
        method_cycles = [
            row for row in planning_rows if row["rollout_method"] == method
        ]
        print(f"\n{method}")
        print(f"  Cartesian error: {cart:.8e} ({relative:+.3f}% vs buffer)")
        print(
            "  drawing-cost change: "
            f"{mean(method, 'drawing_total_cost') - mean('buffer_only', 'drawing_total_cost'):+.8e}"
        )
        print(f"  paths improved/worsened/tied: {improved}/{worsened}/{tied}")
        print(
            "  max joint step: "
            f"{mean(method, 'max_joint_step'):.8e} "
            f"(buffer {mean('buffer_only', 'max_joint_step'):.8e})"
        )
        print(
            "  planning/future boundary: "
            f"{mean(method, 'mean_planning_boundary_discontinuity'):.8e} / "
            f"{mean(method, 'mean_future_shift_boundary_discontinuity'):.8e}"
        )
        print(
            "  global joint/Cartesian drift: "
            f"{mean(method, 'mean_global_reference_joint_drift'):.8e} / "
            f"{mean(method, 'mean_global_reference_cartesian_drift'):.8e}"
        )
        print(
            "  extension clipping count: "
            f"{sum(int(row['extension_clipping_count']) for row in lookup[method])}"
        )
        print(f"  selected-alpha distribution: {dict(sorted(alpha_counter.items()))}")
        print(
            "  diffusion-selection fraction: "
            f"{np.mean([int(row['selected_diffusion']) for row in method_cycles]):.4f}"
        )
        print(
            "  unsafe diffusion selections: "
            f"{sum(int(row['selected_diffusion_unsafe']) for row in method_cycles)}"
        )

    passing_modes = [row["tail_mode"] for row in decisions if int(row["passed"]) == 1]
    print(f"\nBest observed tail mode: {best_mode}")
    if passing_modes:
        print(f"Tail modes passing all criteria: {', '.join(passing_modes)}")
    else:
        print("No tail mode passes all scientific decision criteria.")
        print(
            "Remaining cause classification: "
            + (", ".join(failure_causes) if failure_causes else "unclassified")
        )
    for filename in (
        "per_path_summary.csv",
        "paired_comparison.csv",
        "tail_mode_comparison.csv",
        "aggregate_summary.csv",
        "planning_cycle_details.csv",
        "scientific_decision.csv",
    ):
        print(f"Saved: {args.output_dir / filename}")
    if args.save_candidate_details:
        print(f"Saved: {args.output_dir / 'candidate_details.csv'}")
    print(f"Saved trajectories and plots under: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
