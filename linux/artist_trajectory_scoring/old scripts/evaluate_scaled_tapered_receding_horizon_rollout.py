#!/usr/bin/env python3
"""Recursively evaluate scaled, tapered action-buffer diffusion rollouts.

The three rollout methods maintain independent action buffers. Expert joints
are used only for full-trajectory and ``oracle_*`` diagnostic metrics, never
for conditioning, candidate generation, safety gates, or selection.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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
    write_dict_csv,
)
from diagnose_diffusion_v5_sampling_modes import reverse_noised_x0_batches
from diagnose_scaled_tapered_action_buffer_candidates import (
    RepositoryFKAdapter,
    StrictRepositoryRanking,
    apply_scaled_taper,
    evaluate_candidate,
    rollout_default_namespace,
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
    "scaled_tapered_receding_horizon_rollout"
)
ROLLOUT_METHODS = (
    "buffer_only",
    "diffusion_lexicographic",
    "diffusion_discounted_hard_gate",
)
DIFFUSION_METHODS = ROLLOUT_METHODS[1:]
EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate recursive scaled/tapered action-buffer diffusion."
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
    parser.add_argument("--taper_mode", choices=("linear", "cosine"), default="linear")
    parser.add_argument(
        "--alphas",
        nargs="+",
        type=float,
        default=(0.05, 0.10, 0.25),
    )
    parser.add_argument("--num_base_samples", type=int, default=16)
    parser.add_argument(
        "--tail_extension",
        choices=("constant_position", "constant_velocity", "bootstrap_prior"),
        default="constant_velocity",
    )
    parser.add_argument("--boundary_ratio", type=float, default=2.0)
    parser.add_argument("--prefix_step_ratio", type=float, default=2.0)
    parser.add_argument("--absolute_boundary_limit", type=float, default=0.25)
    parser.add_argument("--absolute_prefix_step_limit", type=float, default=0.25)
    parser.add_argument("--ranking_discount", type=float, default=0.9)
    parser.add_argument(
        "--material_worsening_ratio",
        type=float,
        default=1.02,
        help="Maximum tracking-cost ratio accepted while repairing an unsafe buffer.",
    )
    parser.add_argument(
        "--material_safety_ratio",
        type=float,
        default=1.10,
        help="Maximum full-trajectory step/continuity ratio for the final decision.",
    )
    parser.add_argument("--continuity_weight", type=float, default=None)
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
    if args.num_base_samples <= 0:
        raise ValueError("num_base_samples must be positive")
    if args.ramp_length < 0:
        raise ValueError("ramp_length must be non-negative")
    if not args.alphas or any(value <= 0.0 or not np.isfinite(value) for value in args.alphas):
        raise ValueError("alphas must be finite and positive")
    if len(set(float(value) for value in args.alphas)) != len(args.alphas):
        raise ValueError("alphas contains duplicate values")
    if args.max_paths is not None and args.max_paths <= 0:
        raise ValueError("max_paths must be positive when supplied")
    if not 0.0 < args.ranking_discount <= 1.0:
        raise ValueError("ranking_discount must be in (0,1]")
    if args.material_worsening_ratio < 1.0 or args.material_safety_ratio < 1.0:
        raise ValueError("Material-worsening ratios must be at least 1")
    for name in (
        "boundary_ratio",
        "prefix_step_ratio",
        "absolute_boundary_limit",
        "absolute_prefix_step_limit",
    ):
        if float(getattr(args, name)) <= 0.0:
            raise ValueError(f"--{name} must be positive")


def clip_to_joint_limits(
    raw_q: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, float]]:
    finite_array(raw_q, "raw joint trajectory before limit clipping")
    clipped = np.asarray(raw_q, dtype=np.float32).copy()
    for joint in range(JOINT_DIM):
        if np.isfinite(lower[joint]):
            clipped[:, joint] = np.maximum(clipped[:, joint], lower[joint])
        if np.isfinite(upper[joint]):
            clipped[:, joint] = np.minimum(clipped[:, joint], upper[joint])
    delta = clipped.astype(np.float64) - np.asarray(raw_q, dtype=np.float64)
    changed = np.abs(delta) > 1e-9
    return clipped, {
        "clipped_value_count": int(np.count_nonzero(changed)),
        "clipped_timestep_count": int(np.count_nonzero(np.any(changed, axis=1))),
        "clipping_rms": float(np.sqrt(np.mean(np.square(delta)))),
        "clipping_max_abs": float(np.max(np.abs(delta))),
    }


def candidate_safety_status(
    row: Dict[str, Any],
    buffer: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    practical_keys = (
        "boundary_max_abs_joint_step",
        "prefix_max_joint_step",
        "prefix_mean_cartesian_error",
        "prefix_drawing_cost",
        "repository_ranking_score",
        "discounted_score",
        "raw_joint_limit_violation_magnitude",
    )
    if not np.all(np.isfinite([float(row[key]) for key in practical_keys])):
        return {
            "accepted": 0,
            "rejection_reason": "non_finite",
            "worsened_safety_vs_buffer": 1,
            "improved_safety_vs_buffer": 0,
        }

    raw_limit_worse = (
        int(row["raw_joint_limit_violation_count"])
        > int(buffer["raw_joint_limit_violation_count"])
        or float(row["raw_joint_limit_violation_magnitude"])
        > float(buffer["raw_joint_limit_violation_magnitude"]) + 1e-12
    )
    actual_limit_worse = (
        float(row["prefix_joint_limit_violation_count"])
        > float(buffer["prefix_joint_limit_violation_count"])
        or float(row["prefix_joint_limit_violation_cost"])
        > float(buffer["prefix_joint_limit_violation_cost"]) + 1e-12
    )
    boundary_worse = (
        float(row["boundary_max_abs_joint_step"])
        > float(buffer["boundary_max_abs_joint_step"]) + 1e-12
    )
    prefix_step_worse = (
        float(row["prefix_max_joint_step"])
        > float(buffer["prefix_max_joint_step"]) + 1e-12
    )
    safety_worse = raw_limit_worse or actual_limit_worse or boundary_worse or prefix_step_worse
    safety_improved = (
        int(row["raw_joint_limit_violation_count"])
        < int(buffer["raw_joint_limit_violation_count"])
        or float(row["raw_joint_limit_violation_magnitude"])
        < float(buffer["raw_joint_limit_violation_magnitude"]) - 1e-12
        or float(row["boundary_max_abs_joint_step"])
        < float(buffer["boundary_max_abs_joint_step"]) - 1e-12
        or float(row["prefix_max_joint_step"])
        < float(buffer["prefix_max_joint_step"]) - 1e-12
    )
    tracking_improved = (
        float(row["prefix_mean_cartesian_error"])
        < float(buffer["prefix_mean_cartesian_error"]) - 1e-12
        or float(row["prefix_drawing_cost"])
        < float(buffer["prefix_drawing_cost"]) - 1e-12
    )
    buffer_unsafe = bool(buffer["buffer_already_unsafe"])

    if buffer_unsafe:
        if raw_limit_worse or actual_limit_worse:
            reason = "joint_limit_worse_than_unsafe_buffer"
        elif boundary_worse:
            reason = "boundary_worse_than_unsafe_buffer"
        elif prefix_step_worse:
            reason = "prefix_step_worse_than_unsafe_buffer"
        elif (
            float(row["prefix_mean_cartesian_error"])
            > float(buffer["prefix_mean_cartesian_error"])
            * args.material_worsening_ratio
            or float(row["prefix_drawing_cost"])
            > float(buffer["prefix_drawing_cost"])
            * args.material_worsening_ratio
        ):
            reason = "tracking_materially_worse"
        elif not (safety_improved or tracking_improved):
            reason = "no_safety_or_tracking_improvement"
        else:
            reason = ""
    else:
        boundary_limit = min(
            args.absolute_boundary_limit,
            max(
                float(buffer["boundary_max_abs_joint_step"]) * args.boundary_ratio,
                EPS,
            ),
        )
        prefix_limit = min(
            args.absolute_prefix_step_limit,
            max(
                float(buffer["prefix_max_joint_step"]) * args.prefix_step_ratio,
                EPS,
            ),
        )
        if int(row["raw_joint_limit_violation_count"]) > 0 or float(
            row["prefix_joint_limit_violation_count"]
        ) > 0.0:
            reason = "joint_limit"
        elif float(row["boundary_max_abs_joint_step"]) > boundary_limit:
            reason = "boundary_step"
        elif float(row["prefix_max_joint_step"]) > prefix_limit:
            reason = "prefix_step"
        else:
            reason = ""
    return {
        "accepted": int(reason == ""),
        "rejection_reason": reason,
        "worsened_safety_vs_buffer": int(safety_worse),
        "improved_safety_vs_buffer": int(safety_improved),
    }


def buffer_is_unsafe(buffer: Dict[str, Any], args: argparse.Namespace) -> bool:
    return bool(
        int(buffer["raw_joint_limit_violation_count"]) > 0
        or float(buffer["prefix_joint_limit_violation_count"]) > 0.0
        or float(buffer["boundary_max_abs_joint_step"])
        > args.absolute_boundary_limit
        or float(buffer["prefix_max_joint_step"])
        > args.absolute_prefix_step_limit
        or not np.all(
            np.isfinite(
                [
                    float(buffer["prefix_mean_cartesian_error"]),
                    float(buffer["prefix_drawing_cost"]),
                ]
            )
        )
    )


def select_candidate(
    method: str,
    rows: Sequence[Dict[str, Any]],
    material_worsening_ratio: float,
) -> int:
    if not rows or int(rows[0]["candidate_index"]) != 0:
        raise AssertionError("Candidate 0 must be the current unrefined buffer")
    if method == "buffer_only":
        return 0
    buffer = rows[0]
    accepted = [
        index
        for index, row in enumerate(rows[1:], start=1)
        if int(row["accepted"]) == 1
    ]
    not_worse_both = [
        index
        for index in accepted
        if not (
            float(rows[index]["prefix_mean_cartesian_error"])
            > float(buffer["prefix_mean_cartesian_error"])
            and float(rows[index]["prefix_drawing_cost"])
            > float(buffer["prefix_drawing_cost"])
        )
    ]
    if method == "diffusion_discounted_hard_gate":
        eligible = [0, *not_worse_both]
        return min(
            eligible,
            key=lambda index: (float(rows[index]["discounted_score"]), index),
        )
    if method != "diffusion_lexicographic":
        raise ValueError(f"Unknown rollout method: {method}")

    both_improving = [
        index
        for index in accepted
        if float(rows[index]["prefix_mean_cartesian_error"])
        < float(buffer["prefix_mean_cartesian_error"])
        and float(rows[index]["prefix_drawing_cost"])
        < float(buffer["prefix_drawing_cost"])
    ]
    qualifying = both_improving
    if bool(buffer["buffer_already_unsafe"]) and not qualifying:
        qualifying = [
            index
            for index in accepted
            if int(rows[index]["improved_safety_vs_buffer"]) == 1
            and float(rows[index]["prefix_mean_cartesian_error"])
            <= float(buffer["prefix_mean_cartesian_error"])
            * material_worsening_ratio
            and float(rows[index]["prefix_drawing_cost"])
            <= float(buffer["prefix_drawing_cost"])
            * material_worsening_ratio
        ]
    if not qualifying:
        return 0
    return min(
        qualifying,
        key=lambda index: (
            float(rows[index]["prefix_drawing_cost"]),
            float(rows[index]["prefix_mean_cartesian_error"]),
            float(rows[index]["boundary_max_abs_joint_step"]),
            float(rows[index]["prefix_max_joint_step"]),
            index,
        ),
    )


def extend_buffer(
    *,
    selected_buffer: np.ndarray,
    execution_count: int,
    extension_count: int,
    next_start: int,
    bootstrap_prior: np.ndarray,
    mode: str,
    lower: np.ndarray,
    upper: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    retained = selected_buffer[execution_count:].copy()
    if extension_count <= 0:
        raw_extension = np.empty((0, JOINT_DIM), dtype=np.float32)
    elif mode == "constant_position":
        raw_extension = np.repeat(
            retained[-1:].copy(), extension_count, axis=0
        )
    elif mode == "constant_velocity":
        if retained.shape[0] >= 2:
            velocity = retained[-1] - retained[-2]
        elif selected_buffer.shape[0] >= 2:
            velocity = selected_buffer[-1] - selected_buffer[-2]
        else:
            velocity = np.zeros(JOINT_DIM, dtype=np.float32)
        raw_extension = np.stack(
            [retained[-1] + (offset + 1) * velocity for offset in range(extension_count)],
            axis=0,
        ).astype(np.float32)
    elif mode == "bootstrap_prior":
        first_index = next_start + retained.shape[0]
        indices = np.minimum(
            np.arange(first_index, first_index + extension_count),
            bootstrap_prior.shape[0] - 1,
        )
        raw_extension = bootstrap_prior[indices].astype(np.float32).copy()
    else:
        raise ValueError(f"Unknown tail extension mode: {mode}")
    clipped_extension, clipping = clip_to_joint_limits(
        raw_extension, lower, upper
    ) if extension_count else (raw_extension, {
        "clipped_value_count": 0,
        "clipped_timestep_count": 0,
        "clipping_rms": 0.0,
        "clipping_max_abs": 0.0,
    })
    next_buffer = np.concatenate((retained, clipped_extension), axis=0)
    if next_buffer.shape != selected_buffer.shape:
        raise AssertionError(
            f"Shifted buffer shape {next_buffer.shape} differs from {selected_buffer.shape}"
        )
    finite_array(next_buffer, "shifted and extended action buffer")
    return next_buffer.astype(np.float32), {
        "tail_extension_mode": mode,
        "retained_tail_count": int(retained.shape[0]),
        "extension_count": int(extension_count),
        "extension_first_index": int(next_start + retained.shape[0]),
        **{f"extension_{key}": value for key, value in clipping.items()},
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
) -> Tuple[np.ndarray, List[int]]:
    samples: List[np.ndarray] = []
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
        )[0]
        samples.append(sample)
        seeds.append(seed)
    output = np.stack(samples, axis=0).astype(np.float32)
    finite_array(output, "seeded normalized diffusion residual samples")
    return output, seeds


def derivative_summary(q: np.ndarray) -> Dict[str, float]:
    def summarize(values: np.ndarray, name: str) -> Dict[str, float]:
        if not values.size:
            return {f"mean_joint_{name}": 0.0, f"max_joint_{name}": 0.0}
        norms = np.linalg.norm(values, axis=1)
        return {
            f"mean_joint_{name}": float(np.mean(norms)),
            f"max_joint_{name}": float(np.max(norms)),
        }

    velocity = np.diff(q, axis=0)
    acceleration = np.diff(q, n=2, axis=0)
    jerk = np.diff(q, n=3, axis=0)
    return {
        **summarize(velocity, "velocity"),
        **summarize(acceleration, "acceleration"),
        **summarize(jerk, "jerk"),
        "max_joint_step": float(np.max(np.abs(velocity))) if velocity.size else 0.0,
    }


def full_trajectory_metrics(
    *,
    q: np.ndarray,
    desired_path: np.ndarray,
    expert_q: np.ndarray,
    q_start: np.ndarray,
    robot: Any,
    joint_names: Sequence[str],
    ee_link: str,
    lower: np.ndarray,
    upper: np.ndarray,
    weights: Any,
    boundary_steps: Sequence[float],
    cycle_rows: Sequence[Dict[str, Any]],
) -> Tuple[Dict[str, Any], np.ndarray, np.ndarray]:
    ee = fk_positions(robot, joint_names, ee_link, q)
    region = evaluate_region(
        q=q,
        desired=desired_path,
        ee=ee,
        previous_q=q_start,
        lower=lower,
        upper=upper,
        weights=weights,
    )
    errors = np.linalg.norm(ee - desired_path, axis=1)
    violation_count, violation_magnitude = limit_metrics(q, lower, upper)
    selected_diffusion = [
        row for row in cycle_rows if int(row["selected_diffusion"]) == 1
    ]
    metrics = {
        "mean_cartesian_error": float(np.mean(errors)),
        "rms_cartesian_error": float(np.sqrt(np.mean(np.square(errors)))),
        "max_cartesian_error": float(np.max(errors)),
        "drawing_total_cost": float(region["drawing_cost"]),
        **derivative_summary(q),
        "joint_limit_violation_count": int(violation_count),
        "joint_limit_violation_magnitude": float(violation_magnitude),
        "full_joint_rmse_vs_expert": float(
            np.sqrt(np.mean(np.square(q - expert_q)))
        ),
        "mean_planning_boundary_discontinuity": float(np.mean(boundary_steps)),
        "max_planning_boundary_discontinuity": float(np.max(boundary_steps)),
        "planning_cycle_count": len(cycle_rows),
        "diffusion_selection_fraction": float(
            len(selected_diffusion) / max(len(cycle_rows), 1)
        ),
        "unsafe_buffer_cycle_count": int(
            sum(int(row["buffer_already_unsafe"]) for row in cycle_rows)
        ),
        "unsafe_diffusion_selection_count": int(
            sum(int(row["selected_diffusion_unsafe"]) for row in cycle_rows)
        ),
        "safety_improving_selection_count": int(
            sum(int(row["selected_candidate_improved_safety_vs_buffer"]) for row in cycle_rows)
        ),
        "local_both_improving_cycle_count": int(
            sum(int(row["selected_improved_both_tracking_metrics"]) for row in cycle_rows)
        ),
    }
    return metrics, ee, errors


def run_method(
    *,
    method: str,
    path_index: int,
    path_name: str,
    desired_path: np.ndarray,
    expert_q: np.ndarray,
    q_start: np.ndarray,
    bootstrap_prior: np.ndarray,
    model: torch.nn.Module,
    call_variant: str,
    schedule: Dict[str, torch.Tensor],
    residual_mean: np.ndarray,
    residual_std: np.ndarray,
    condition_mean: np.ndarray,
    condition_std: np.ndarray,
    robot: Any,
    repository_fk: RepositoryFKAdapter,
    repository_ranking: StrictRepositoryRanking,
    joint_names: Sequence[str],
    ee_link: str,
    lower: np.ndarray,
    upper: np.ndarray,
    weights: Any,
    w_rms_cart: float,
    w_limit_count: float,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    horizon = args.prediction_horizon
    execution_horizon = args.execution_horizon
    trajectory_length = desired_path.shape[0]
    buffer_q, _ = build_teacher_forced_buffer(bootstrap_prior, 0, horizon)
    buffer_q, initial_clipping = clip_to_joint_limits(buffer_q, lower, upper)
    desired_delta = desired_differences(desired_path)
    residual_physical_zero = np.zeros((horizon, JOINT_DIM), dtype=np.float32)
    residual_norm_zero = (
        (residual_physical_zero - residual_mean[None, :])
        / residual_std[None, :]
    ).astype(np.float32)
    round_trip = residual_norm_zero * residual_std[None, :] + residual_mean[None, :]
    if not np.allclose(round_trip, residual_physical_zero, rtol=1e-6, atol=1e-7):
        raise AssertionError("Physical-zero residual normalization failed to round-trip")
    taper = taper_values(horizon, args.ramp_length, args.taper_mode)
    executed_chunks: List[np.ndarray] = []
    planning_rows: List[Dict[str, Any]] = []
    candidate_rows: List[Dict[str, Any]] = []
    boundary_steps: List[float] = []

    for cycle_index, start in enumerate(range(0, trajectory_length, execution_horizon)):
        execution_count = min(execution_horizon, trajectory_length - start)
        indices = np.minimum(
            np.arange(start, start + horizon), trajectory_length - 1
        )
        desired_window = desired_path[indices]
        expert_window = expert_q[indices]
        previous_executed_q = (
            q_start if not executed_chunks else executed_chunks[-1][-1]
        )
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
        condition_norm = (
            (condition - condition_mean[None, :]) / condition_std[None, :]
        ).astype(np.float32)
        finite_array(condition_norm, "recursive normalized condition")

        buffer_metrics = evaluate_candidate(
            candidate_q=buffer_q,
            candidate_index=0,
            candidate_type="unrefined_buffer",
            desired_window=desired_window,
            expert_window=expert_window,
            execution_count=execution_count,
            previous_q=previous_executed_q,
            repository_fk=repository_fk,
            robot=robot,
            joint_names=joint_names,
            ee_link=ee_link,
            lower=lower,
            upper=upper,
            weights=weights,
            w_rms_cart=w_rms_cart,
            w_limit_count=w_limit_count,
            args=args,
            repository_ranking=repository_ranking,
        )
        buffer_raw_count, buffer_raw_magnitude = limit_metrics(
            buffer_q, lower, upper
        )
        buffer_row: Dict[str, Any] = {
            "candidate_index": 0,
            "candidate_type": "unrefined_buffer",
            "base_sample_index": -1,
            "alpha": 0.0,
            **buffer_metrics,
            "raw_joint_limit_violation_count": int(buffer_raw_count),
            "raw_joint_limit_violation_magnitude": float(buffer_raw_magnitude),
            "clipped_value_count": 0,
            "clipping_max_abs": 0.0,
            "accepted": 1,
            "rejection_reason": "",
            "worsened_safety_vs_buffer": 0,
            "improved_safety_vs_buffer": 0,
        }
        buffer_row["buffer_already_unsafe"] = int(
            buffer_is_unsafe(buffer_row, args)
        )
        rows: List[Dict[str, Any]] = [buffer_row]

        if method in DIFFUSION_METHODS:
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
            generated_physical = (
                generated_norm * residual_std[None, None, :]
                + residual_mean[None, None, :]
            ).astype(np.float32)
            candidate_index = 1
            for base_sample_index in range(args.num_base_samples):
                for alpha in args.alphas:
                    scaled_residual = apply_scaled_taper(
                        generated_physical[base_sample_index],
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
                    raw_limit_count, raw_limit_magnitude = limit_metrics(
                        raw_candidate, lower, upper
                    )
                    metrics = evaluate_candidate(
                        candidate_q=candidate_q,
                        candidate_index=candidate_index,
                        candidate_type="diffusion_refined",
                        desired_window=desired_window,
                        expert_window=expert_window,
                        execution_count=execution_count,
                        previous_q=previous_executed_q,
                        repository_fk=repository_fk,
                        robot=robot,
                        joint_names=joint_names,
                        ee_link=ee_link,
                        lower=lower,
                        upper=upper,
                        weights=weights,
                        w_rms_cart=w_rms_cart,
                        w_limit_count=w_limit_count,
                        args=args,
                        repository_ranking=repository_ranking,
                    )
                    row: Dict[str, Any] = {
                        "candidate_index": candidate_index,
                        "candidate_type": "diffusion_refined",
                        "base_sample_index": base_sample_index,
                        "candidate_seed": sample_seeds[base_sample_index],
                        "alpha": float(alpha),
                        **metrics,
                        "raw_joint_limit_violation_count": int(raw_limit_count),
                        "raw_joint_limit_violation_magnitude": float(raw_limit_magnitude),
                        **clipping,
                        "generated_residual_rms": float(
                            np.sqrt(
                                np.mean(
                                    np.square(generated_physical[base_sample_index])
                                )
                            )
                        ),
                        "scaled_residual_rms": float(
                            np.sqrt(np.mean(np.square(scaled_residual)))
                        ),
                        "buffer_already_unsafe": buffer_row["buffer_already_unsafe"],
                        "_candidate_q": candidate_q,
                    }
                    row.update(candidate_safety_status(row, buffer_row, args))
                    rows.append(row)
                    candidate_index += 1

        selected_index = select_candidate(
            method,
            rows,
            args.material_worsening_ratio,
        )
        selected = rows[selected_index]
        selected_buffer = (
            buffer_q
            if selected_index == 0
            else np.asarray(selected["_candidate_q"], dtype=np.float32)
        )
        selected_prefix = selected_buffer[:execution_count].copy()
        if selected_prefix.shape != (execution_count, JOINT_DIM):
            raise AssertionError("Executed prefix has an unexpected shape")
        executed_chunks.append(selected_prefix)
        boundary_step = float(
            np.max(np.abs(selected_prefix[0] - previous_executed_q))
        )
        boundary_steps.append(boundary_step)

        selected_diffusion = selected_index != 0
        selected_worse_safety = bool(
            selected.get("worsened_safety_vs_buffer", 0)
        )
        selected_improved_safety = bool(
            selected.get("improved_safety_vs_buffer", 0)
        )
        selected_diffusion_unsafe = bool(
            selected_diffusion and int(selected.get("accepted", 0)) == 0
        )
        selected_buffer_unsafe = bool(
            not selected_diffusion and buffer_row["buffer_already_unsafe"]
        )
        selected_absolute_safe = not buffer_is_unsafe(selected, args)
        diffusion_repaired = bool(
            buffer_row["buffer_already_unsafe"]
            and selected_diffusion
            and selected_absolute_safe
        )

        if start + execution_count < trajectory_length:
            next_buffer, extension = extend_buffer(
                selected_buffer=selected_buffer,
                execution_count=execution_count,
                extension_count=execution_count,
                next_start=start + execution_count,
                bootstrap_prior=bootstrap_prior,
                mode=args.tail_extension,
                lower=lower,
                upper=upper,
            )
        else:
            next_buffer = selected_buffer
            extension = {
                "tail_extension_mode": args.tail_extension,
                "retained_tail_count": horizon - execution_count,
                "extension_count": 0,
                "extension_first_index": trajectory_length,
                "extension_clipped_value_count": 0,
                "extension_clipped_timestep_count": 0,
                "extension_clipping_rms": 0.0,
                "extension_clipping_max_abs": 0.0,
            }
        planning_row: Dict[str, Any] = {
            "path_name": path_name,
            "rollout_method": method,
            "planning_cycle_index": cycle_index,
            "current_index": start,
            "execution_count": execution_count,
            "candidate_count": len(rows),
            "selected_candidate_index": selected_index,
            "selected_candidate_type": selected["candidate_type"],
            "selected_diffusion": int(selected_diffusion),
            "selected_alpha": selected["alpha"],
            "selected_base_sample_index": selected["base_sample_index"],
            "buffer_already_unsafe": int(buffer_row["buffer_already_unsafe"]),
            "selected_buffer_unsafe": int(selected_buffer_unsafe),
            "selected_diffusion_unsafe": int(selected_diffusion_unsafe),
            "selected_candidate_worsened_safety_vs_buffer": int(
                selected_worse_safety
            ),
            "selected_candidate_improved_safety_vs_buffer": int(
                selected_improved_safety
            ),
            "diffusion_repaired_unsafe_buffer": int(diffusion_repaired),
            "selected_improved_prefix_cartesian": int(
                float(selected["prefix_mean_cartesian_error"])
                < float(buffer_row["prefix_mean_cartesian_error"])
            ),
            "selected_improved_prefix_drawing": int(
                float(selected["prefix_drawing_cost"])
                < float(buffer_row["prefix_drawing_cost"])
            ),
            "selected_improved_both_tracking_metrics": int(
                float(selected["prefix_mean_cartesian_error"])
                < float(buffer_row["prefix_mean_cartesian_error"])
                and float(selected["prefix_drawing_cost"])
                < float(buffer_row["prefix_drawing_cost"])
            ),
            "buffer_prefix_mean_cartesian_error": buffer_row[
                "prefix_mean_cartesian_error"
            ],
            "selected_prefix_mean_cartesian_error": selected[
                "prefix_mean_cartesian_error"
            ],
            "prefix_cartesian_change": float(
                selected["prefix_mean_cartesian_error"]
                - buffer_row["prefix_mean_cartesian_error"]
            ),
            "buffer_prefix_drawing_cost": buffer_row["prefix_drawing_cost"],
            "selected_prefix_drawing_cost": selected["prefix_drawing_cost"],
            "prefix_drawing_cost_change": float(
                selected["prefix_drawing_cost"]
                - buffer_row["prefix_drawing_cost"]
            ),
            "buffer_boundary_max_abs_joint_step": buffer_row[
                "boundary_max_abs_joint_step"
            ],
            "selected_boundary_max_abs_joint_step": selected[
                "boundary_max_abs_joint_step"
            ],
            "buffer_prefix_max_joint_step": buffer_row["prefix_max_joint_step"],
            "selected_prefix_max_joint_step": selected["prefix_max_joint_step"],
            "selected_raw_joint_limit_violation_count": selected[
                "raw_joint_limit_violation_count"
            ],
            "selected_raw_joint_limit_violation_magnitude": selected[
                "raw_joint_limit_violation_magnitude"
            ],
            "selected_clipped_value_count": selected.get("clipped_value_count", 0),
            "selected_clipping_max_abs": selected.get("clipping_max_abs", 0.0),
            "boundary_discontinuity": boundary_step,
            "useful_candidate_count": int(
                sum(
                    int(row.get("accepted", 0)) == 1
                    and float(row["prefix_mean_cartesian_error"])
                    < float(buffer_row["prefix_mean_cartesian_error"])
                    and float(row["prefix_drawing_cost"])
                    < float(buffer_row["prefix_drawing_cost"])
                    for row in rows[1:]
                )
            ),
            **extension,
            "initial_buffer_clipped_value_count": initial_clipping[
                "clipped_value_count"
            ],
        }
        planning_rows.append(planning_row)
        if args.save_candidate_details:
            for row in rows:
                candidate_rows.append(
                    {
                        "path_name": path_name,
                        "rollout_method": method,
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
        buffer_q = next_buffer

    executed_q = np.concatenate(executed_chunks, axis=0)
    if executed_q.shape != (trajectory_length, JOINT_DIM):
        raise AssertionError(
            f"Final trajectory must have shape ({trajectory_length},6), got {executed_q.shape}"
        )
    metrics, ee, cartesian_errors = full_trajectory_metrics(
        q=executed_q,
        desired_path=desired_path,
        expert_q=expert_q,
        q_start=q_start,
        robot=robot,
        joint_names=joint_names,
        ee_link=ee_link,
        lower=lower,
        upper=upper,
        weights=weights,
        boundary_steps=boundary_steps,
        cycle_rows=planning_rows,
    )
    return {
        "method": method,
        "q": executed_q,
        "ee": ee,
        "cartesian_errors": cartesian_errors,
        "planning_rows": planning_rows,
        "candidate_rows": candidate_rows,
        "metrics": metrics,
    }


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


COMPARISON_METRICS = (
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
)


def paired_rows(per_path: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_path_method = {
        (str(row["path_name"]), str(row["rollout_method"])): row
        for row in per_path
    }
    output: List[Dict[str, Any]] = []
    paths = sorted({str(row["path_name"]) for row in per_path})
    for path_name in paths:
        baseline = by_path_method[(path_name, "buffer_only")]
        for method in DIFFUSION_METHODS:
            candidate = by_path_method[(path_name, method)]
            row: Dict[str, Any] = {
                "path_name": path_name,
                "rollout_method": method,
            }
            for metric in COMPARISON_METRICS:
                base_value = float(baseline[metric])
                candidate_value = float(candidate[metric])
                change = candidate_value - base_value
                row[f"buffer_{metric}"] = base_value
                row[f"diffusion_{metric}"] = candidate_value
                row[f"absolute_change_{metric}"] = change
                row[f"percentage_change_{metric}"] = (
                    100.0 * change / abs(base_value)
                    if abs(base_value) > EPS
                    else float("nan")
                )
            output.append(row)
    return output


def aggregate_paired(paired: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for method in DIFFUSION_METHODS:
        group = [row for row in paired if row["rollout_method"] == method]
        for metric in COMPARISON_METRICS:
            changes = np.asarray(
                [float(row[f"absolute_change_{metric}"]) for row in group],
                dtype=np.float64,
            )
            tolerance = 1e-12
            improved = changes < -tolerance
            worsened = changes > tolerance
            tied = ~(improved | worsened)
            output.append(
                {
                    "rollout_method": method,
                    "metric": metric,
                    "path_count": len(group),
                    "mean_change": float(np.mean(changes)),
                    "median_change": float(np.median(changes)),
                    "std_change": float(np.std(changes)),
                    "min_change": float(np.min(changes)),
                    "max_change": float(np.max(changes)),
                    "improved_count": int(np.sum(improved)),
                    "worsened_count": int(np.sum(worsened)),
                    "tied_count": int(np.sum(tied)),
                    "percentage_improved": float(100.0 * np.mean(improved)),
                }
            )
    return output


def save_path_plots(
    path_dir: Path,
    desired: np.ndarray,
    results: Dict[str, Dict[str, Any]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "buffer_only": "black",
        "diffusion_lexicographic": "tab:green",
        "diffusion_discounted_hard_gate": "tab:blue",
    }

    figure = plt.figure(figsize=(8, 6))
    axis = figure.add_subplot(111, projection="3d")
    axis.plot(*desired.T, label="desired", color="tab:red", linewidth=2)
    for method, result in results.items():
        axis.plot(*result["ee"].T, label=method, color=colors[method], alpha=0.85)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path_dir / "cartesian_trajectory.png", dpi=160)
    plt.close(figure)

    def line_figure(filename: str, ylabel: str, series: Dict[str, np.ndarray]) -> None:
        figure, axis = plt.subplots(figsize=(9, 5))
        for method, values in series.items():
            axis.plot(values, label=method, color=colors[method])
        axis.set_xlabel("Timestep")
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=8)
        figure.tight_layout()
        figure.savefig(path_dir / filename, dpi=160)
        plt.close(figure)

    line_figure(
        "cartesian_error_over_time.png",
        "Cartesian error",
        {method: result["cartesian_errors"] for method, result in results.items()},
    )
    line_figure(
        "maximum_joint_step_over_time.png",
        "Maximum absolute joint step",
        {
            method: np.concatenate(
                ([0.0], np.max(np.abs(np.diff(result["q"], axis=0)), axis=1))
            )
            for method, result in results.items()
        },
    )

    figure, axes = plt.subplots(JOINT_DIM, 1, figsize=(10, 12), sharex=True)
    for joint, axis in enumerate(axes):
        for method, result in results.items():
            axis.plot(result["q"][:, joint], label=method, color=colors[method])
        axis.set_ylabel(f"q{joint + 1}")
        axis.grid(True, alpha=0.2)
    axes[0].legend(fontsize=8)
    axes[-1].set_xlabel("Timestep")
    figure.tight_layout()
    figure.savefig(path_dir / "joint_trajectory_comparison.png", dpi=160)
    plt.close(figure)

    line_figure(
        "boundary_discontinuity_by_cycle.png",
        "Boundary maximum joint step",
        {
            method: np.asarray(
                [float(row["boundary_discontinuity"]) for row in result["planning_rows"]]
            )
            for method, result in results.items()
        },
    )
    line_figure(
        "selected_alpha_by_cycle.png",
        "Selected alpha",
        {
            method: np.asarray(
                [float(row["selected_alpha"]) for row in result["planning_rows"]]
            )
            for method, result in results.items()
        },
    )
    buffer_error = results["buffer_only"]["cartesian_errors"]
    line_figure(
        "cumulative_cartesian_error_difference.png",
        "Cumulative error difference vs buffer",
        {
            method: np.cumsum(result["cartesian_errors"] - buffer_error)
            for method, result in results.items()
            if method != "buffer_only"
        },
    )


def save_aggregate_plots(
    output_dir: Path,
    per_path: Sequence[Dict[str, Any]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir = output_dir / "aggregate_plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    lookup = {
        (row["path_name"], row["rollout_method"]): row for row in per_path
    }
    paths = sorted({str(row["path_name"]) for row in per_path})

    def scatter(metric: str, filename: str, label: str) -> None:
        figure, axis = plt.subplots(figsize=(7, 6))
        all_values: List[float] = []
        for method, color in (
            ("diffusion_lexicographic", "tab:green"),
            ("diffusion_discounted_hard_gate", "tab:blue"),
        ):
            x = [float(lookup[(path, "buffer_only")][metric]) for path in paths]
            y = [float(lookup[(path, method)][metric]) for path in paths]
            all_values.extend(x)
            all_values.extend(y)
            axis.scatter(x, y, label=method, color=color, alpha=0.75)
        low, high = min(all_values), max(all_values)
        axis.plot([low, high], [low, high], linestyle="--", color="black")
        axis.set_xlabel(f"Buffer-only {label}")
        axis.set_ylabel(f"Diffusion {label}")
        axis.legend(fontsize=8)
        axis.grid(True, alpha=0.25)
        figure.tight_layout()
        figure.savefig(plots_dir / filename, dpi=160)
        plt.close(figure)

    scatter(
        "mean_cartesian_error",
        "paired_cartesian_error_scatter.png",
        "mean Cartesian error",
    )
    scatter(
        "drawing_total_cost",
        "paired_drawing_cost_scatter.png",
        "drawing total cost",
    )

    labels: List[str] = []
    improved_counts: List[int] = []
    worsened_counts: List[int] = []
    for method in DIFFUSION_METHODS:
        changes = [
            float(lookup[(path, method)]["mean_cartesian_error"])
            - float(lookup[(path, "buffer_only")]["mean_cartesian_error"])
            for path in paths
        ]
        labels.append(method)
        improved_counts.append(sum(value < -1e-12 for value in changes))
        worsened_counts.append(sum(value > 1e-12 for value in changes))
    x = np.arange(len(labels))
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.bar(x - 0.18, improved_counts, 0.36, label="improved")
    axis.bar(x + 0.18, worsened_counts, 0.36, label="worsened")
    axis.set_xticks(x, labels, rotation=12, ha="right")
    axis.set_ylabel("Path count")
    axis.legend()
    axis.grid(True, axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(plots_dir / "paths_improved_vs_worsened.png", dpi=160)
    plt.close(figure)


def scientific_decision(
    per_path: Sequence[Dict[str, Any]],
    planning_rows: Sequence[Dict[str, Any]],
    args: argparse.Namespace,
) -> Tuple[bool, List[str]]:
    buffer = [row for row in per_path if row["rollout_method"] == "buffer_only"]
    primary = [
        row
        for row in per_path
        if row["rollout_method"] == "diffusion_lexicographic"
    ]

    def mean(rows: Sequence[Dict[str, Any]], key: str) -> float:
        return float(np.mean([float(row[key]) for row in rows]))

    mean_cart_b = mean(buffer, "mean_cartesian_error")
    mean_cart_d = mean(primary, "mean_cartesian_error")
    mean_draw_b = mean(buffer, "drawing_total_cost")
    mean_draw_d = mean(primary, "drawing_total_cost")
    improved_paths = sum(
        float(candidate["mean_cartesian_error"])
        < float(base["mean_cartesian_error"])
        or float(candidate["drawing_total_cost"])
        < float(base["drawing_total_cost"])
        for base, candidate in zip(buffer, primary)
    )
    step_ratio = mean(primary, "max_joint_step") / max(
        mean(buffer, "max_joint_step"), EPS
    )
    boundary_ratio = mean(
        primary, "mean_planning_boundary_discontinuity"
    ) / max(mean(buffer, "mean_planning_boundary_discontinuity"), EPS)
    violations_b = sum(int(row["joint_limit_violation_count"]) for row in buffer)
    violations_d = sum(int(row["joint_limit_violation_count"]) for row in primary)
    primary_cycles = [
        row
        for row in planning_rows
        if row["rollout_method"] == "diffusion_lexicographic"
    ]
    unsafe_diffusion = sum(
        int(row["selected_diffusion_unsafe"]) for row in primary_cycles
    )
    criteria = (
        mean_cart_d < mean_cart_b,
        mean_draw_d < mean_draw_b,
        improved_paths > len(buffer) / 2,
        step_ratio <= args.material_safety_ratio,
        boundary_ratio <= args.material_safety_ratio,
        violations_d <= violations_b,
        unsafe_diffusion == 0,
    )
    if all(criteria):
        return True, []

    failures: List[str] = []
    useful_cycles = sum(
        int(row["useful_candidate_count"]) > 0 for row in primary_cycles
    )
    selected_both = sum(
        int(row["selected_improved_both_tracking_metrics"])
        for row in primary_cycles
    )
    extension_clips = sum(
        int(row["extension_clipped_value_count"]) for row in primary_cycles
    )
    baseline_unsafe = sum(int(row["buffer_already_unsafe"]) for row in primary_cycles)
    if useful_cycles < len(primary_cycles) * 0.25:
        failures.append("candidate generation")
    if useful_cycles > 0 and selected_both < useful_cycles * 0.5:
        failures.append("candidate selection")
    if extension_clips > 0 or args.tail_extension == "constant_velocity" and not criteria[0]:
        failures.append("buffer extension")
    if baseline_unsafe > 0 and violations_d > violations_b:
        failures.append("joint-limit inheritance")
    if boundary_ratio > args.material_safety_ratio:
        failures.append("boundary accumulation")
    if mean_cart_d >= mean_cart_b and useful_cycles >= len(primary_cycles) * 0.25:
        failures.append("recursive drift")
    relative_cart = abs(mean_cart_d - mean_cart_b) / max(mean_cart_b, EPS)
    if relative_cart < 0.01:
        failures.append("lack of meaningful improvement magnitude")
    return False, list(dict.fromkeys(failures))


def main() -> int:
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)
    device = resolve_device(args.device)
    rollout_module = __import__("diagnose_warm_start_action_buffer_rollout")
    repository_ranking = StrictRepositoryRanking(rollout_module)
    rollout_defaults = rollout_default_namespace(rollout_module)
    w_rms_cart = float(rollout_defaults.w_rms_cart)
    w_limit_count = float(rollout_defaults.w_limit_count)
    if args.continuity_weight is None:
        args.continuity_weight = float(rollout_defaults.continuity_weight)
    weights = default_weights()
    stats = load_stats(args.stats_npz)

    checkpoint = torch_load_checkpoint(args.checkpoint, device)
    if int(checkpoint["condition_dim"]) != CONDITION_DIM:
        raise ValueError("The v5b checkpoint must use condition_dim=38")
    if int(checkpoint["target_dim"]) != JOINT_DIM:
        raise ValueError("The v5b checkpoint must use target_dim=6")
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
    desired_paths = finite_array(data["desired_paths"], "desired_paths").astype(np.float32)
    expert_all = finite_array(data["expert_q"], "expert_q").astype(np.float32)
    q_start_all = finite_array(data["q_start"], "q_start").astype(np.float32)
    path_names = decode_names(data["path_names"])
    if expert_all.shape != (desired_paths.shape[0], desired_paths.shape[1], JOINT_DIM):
        raise ValueError("expert_q shape must be (N,T,6)")
    if q_start_all.shape != (desired_paths.shape[0], JOINT_DIM):
        raise ValueError("q_start shape must be (N,6)")
    if desired_paths.shape[1] != 100:
        raise ValueError(f"Expected T=100, got T={desired_paths.shape[1]}")

    robot, joint_names, ee_link = load_fk_context(None, None)
    if len(joint_names) != JOINT_DIM:
        raise ValueError("Exactly six active xMateCR7 joints are required")
    lower, upper = extract_joint_limits(robot, joint_names)
    repository_fk = RepositoryFKAdapter(
        robot, joint_names, ee_link, lower, upper
    )
    max_paths = len(path_names) if args.max_paths is None else min(args.max_paths, len(path_names))
    per_path_rows: List[Dict[str, Any]] = []
    planning_rows: List[Dict[str, Any]] = []
    candidate_rows: List[Dict[str, Any]] = []
    selected_alphas: Counter[float] = Counter()

    print(f"Repository ranking function: {repository_ranking.function_name}")
    print(f"Repository ranking weights: {json.dumps(weights_record(weights), sort_keys=True)}")
    print(
        f"w_rms_cart={w_rms_cart:g}, w_limit_count={w_limit_count:g}, "
        f"continuity_weight={args.continuity_weight:g}"
    )
    print(
        f"Recursive architecture: H={args.prediction_horizon}, "
        f"E={args.execution_horizon}, tail_extension={args.tail_extension}"
    )

    with torch.no_grad():
        for path_index in range(max_paths):
            path_name = path_names[path_index]
            bootstrap_prior = read_predicted_q_csv(
                args.prior_dir / safe_path_name(path_name) / "predicted_q.csv",
                expected_steps=desired_paths.shape[1],
            )
            path_results: Dict[str, Dict[str, Any]] = {}
            for method in ROLLOUT_METHODS:
                result = run_method(
                    method=method,
                    path_index=path_index,
                    path_name=path_name,
                    desired_path=desired_paths[path_index],
                    expert_q=expert_all[path_index],
                    q_start=q_start_all[path_index],
                    bootstrap_prior=bootstrap_prior,
                    model=model,
                    call_variant=call_variant,
                    schedule=schedule,
                    residual_mean=stats["residual_mean"],
                    residual_std=stats["residual_std"],
                    condition_mean=stats["condition_mean"],
                    condition_std=stats["condition_std"],
                    robot=robot,
                    repository_fk=repository_fk,
                    repository_ranking=repository_ranking,
                    joint_names=joint_names,
                    ee_link=ee_link,
                    lower=lower,
                    upper=upper,
                    weights=weights,
                    w_rms_cart=w_rms_cart,
                    w_limit_count=w_limit_count,
                    args=args,
                )
                path_results[method] = result
                row = {
                    "path_name": path_name,
                    "rollout_method": method,
                    **result["metrics"],
                }
                per_path_rows.append(row)
                planning_rows.extend(result["planning_rows"])
                candidate_rows.extend(result["candidate_rows"])
                for cycle in result["planning_rows"]:
                    if (
                        method == "diffusion_lexicographic"
                        and int(cycle["selected_diffusion"]) == 1
                    ):
                        selected_alphas[float(cycle["selected_alpha"])] += 1

            path_dir = args.output_dir / "trajectories" / safe_path_name(path_name)
            write_xyz_csv(path_dir / "desired_path.csv", desired_paths[path_index])
            write_joint_csv(path_dir / "expert_q.csv", expert_all[path_index])
            for method, result in path_results.items():
                write_joint_csv(path_dir / f"{method}_q.csv", result["q"])
                write_xyz_csv(path_dir / f"{method}_ee.csv", result["ee"])
            save_path_plots(path_dir, desired_paths[path_index], path_results)
            print(f"Completed {path_index + 1}/{max_paths} paths: {path_name}")

    paired = paired_rows(per_path_rows)
    aggregate = aggregate_paired(paired)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_dict_csv(args.output_dir / "per_path_summary.csv", per_path_rows)
    write_dict_csv(args.output_dir / "paired_comparison.csv", paired)
    write_dict_csv(args.output_dir / "aggregate_summary.csv", aggregate)
    write_dict_csv(args.output_dir / "planning_cycle_details.csv", planning_rows)
    if args.save_candidate_details:
        write_dict_csv(args.output_dir / "candidate_details.csv", candidate_rows)
    save_aggregate_plots(args.output_dir, per_path_rows)

    lookup = {
        method: [row for row in per_path_rows if row["rollout_method"] == method]
        for method in ROLLOUT_METHODS
    }

    def mean(method: str, key: str) -> float:
        return float(np.mean([float(row[key]) for row in lookup[method]]))

    buffer_cart = mean("buffer_only", "mean_cartesian_error")
    lex_cart = mean("diffusion_lexicographic", "mean_cartesian_error")
    relative_cart = 100.0 * (lex_cart - buffer_cart) / max(abs(buffer_cart), EPS)
    cart_improved = sum(
        float(diffusion["mean_cartesian_error"])
        < float(buffer["mean_cartesian_error"])
        for buffer, diffusion in zip(
            lookup["buffer_only"], lookup["diffusion_lexicographic"]
        )
    )
    lex_cycles = [
        row
        for row in planning_rows
        if row["rollout_method"] == "diffusion_lexicographic"
    ]
    print("\nRecursive rollout summary")
    print(f"buffer-only mean Cartesian error: {buffer_cart:.8e}")
    print(f"lexicographic diffusion mean Cartesian error: {lex_cart:.8e}")
    print(f"relative Cartesian change: {relative_cart:.3f}%")
    print(f"Cartesian-improved paths: {cart_improved}/{max_paths}")
    print(
        "drawing total cost: "
        f"buffer={mean('buffer_only', 'drawing_total_cost'):.8e}, "
        f"lexicographic={mean('diffusion_lexicographic', 'drawing_total_cost'):.8e}"
    )
    print(
        "maximum joint step: "
        f"buffer={mean('buffer_only', 'max_joint_step'):.8e}, "
        f"lexicographic={mean('diffusion_lexicographic', 'max_joint_step'):.8e}"
    )
    print(
        "mean boundary discontinuity: "
        f"buffer={mean('buffer_only', 'mean_planning_boundary_discontinuity'):.8e}, "
        f"lexicographic={mean('diffusion_lexicographic', 'mean_planning_boundary_discontinuity'):.8e}"
    )
    print(
        "joint-limit violations: "
        f"buffer={sum(int(row['joint_limit_violation_count']) for row in lookup['buffer_only'])}, "
        f"lexicographic={sum(int(row['joint_limit_violation_count']) for row in lookup['diffusion_lexicographic'])}"
    )
    print(
        "diffusion selection fraction: "
        f"{np.mean([int(row['selected_diffusion']) for row in lex_cycles]):.4f}"
    )
    print(f"selected-alpha distribution: {dict(sorted(selected_alphas.items()))}")
    print(
        "unsafe diffusion selections: "
        f"{sum(int(row['selected_diffusion_unsafe']) for row in lex_cycles)}"
    )
    print(
        "baseline-unsafe cycles: "
        f"{sum(int(row['buffer_already_unsafe']) for row in lex_cycles)}"
    )

    passed, failure_sources = scientific_decision(per_path_rows, planning_rows, args)
    if passed:
        print("Scientific decision: recursive diffusion PASSES all seven criteria.")
    else:
        print("Scientific decision: recursive diffusion FAILS one or more criteria.")
        print(
            "Likely failure sources: "
            + (", ".join(failure_sources) if failure_sources else "unclassified")
        )
    print(f"Saved per-path summary: {args.output_dir / 'per_path_summary.csv'}")
    print(f"Saved paired comparison: {args.output_dir / 'paired_comparison.csv'}")
    print(f"Saved aggregate summary: {args.output_dir / 'aggregate_summary.csv'}")
    print(f"Saved planning cycles: {args.output_dir / 'planning_cycle_details.csv'}")
    if args.save_candidate_details:
        print(f"Saved candidate details: {args.output_dir / 'candidate_details.csv'}")
    print(f"Saved trajectories and plots under: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
