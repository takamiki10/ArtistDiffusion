#!/usr/bin/env python3
"""Validated v8 anchored recursive receding-horizon rollout."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import multiprocessing
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

import evaluate_diffusion_v7_teacher_forced_validation as v7_evaluator
from evaluate_diffusion_v8_teacher_forced_all_windows import (
    DIFFICULT_PATHS,
    MAXIMUM_JOINT_STEP_RAD,
    PhysicalPathRecord,
    ValidatedInferenceBundle,
    build_recursive_condition_norm,
    compute_fk_positions,
    compute_full_trajectory_metrics,
    load_authoritative_physical_path_population,
    load_validated_inference_bundle,
    sample_ddim_candidates,
    sample_is_selectable,
    select_nested_candidate,
)
import generate_diffusion_v7_cost_improving_residual_targets as target_generator
from generate_ik_seed_path import DEFAULT_URDF_PATH


FROZEN_K_VALUES = (1, 4, 8)
FROZEN_CANDIDATE_COUNT = 8
FROZEN_PRIMARY_PATH_COUNT = 20
MAX_ALLOWED_SEGMENT_CORRECTION_GROWTH_RAD_PER_STEP = 1.0e-5
EPS = 1.0e-12


@dataclass
class RolloutResult:
    path: PhysicalPathRecord
    k: int
    rollout_q: np.ndarray
    rollout_ee: np.ndarray
    executed_source: np.ndarray
    accepted_step_mask: np.ndarray
    fallback_step_mask: np.ndarray
    selected_candidate_indices: np.ndarray
    applied_correction_norms: np.ndarray
    window_start_indices: np.ndarray
    executed_indices: np.ndarray
    decision_rows: List[Dict[str, Any]]
    candidate_rows: List[Dict[str, Any]]
    metrics: Dict[str, Any]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset_dir",
        type=Path,
        default=Path(
            "data/cartesian_expert_dataset_v3/"
            "diffusion_v8_multitarget_scaled_training_dataset_100paths"
        ),
    )
    parser.add_argument(
        "--target_generation_dir",
        type=Path,
        default=Path(
            "data/cartesian_expert_dataset_v3/"
            "diffusion_v8_multitarget_scaled_residual_targets_100paths"
        ),
    )
    parser.add_argument(
        "--model_dir",
        type=Path,
        default=Path(
            "models/"
            "diffusion_v8_multitarget_scaled_residual_unet_100paths_"
            "epsilon_only_seed42"
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/diffusion_v8_anchored_recursive_rollout"),
    )
    parser.add_argument("--checkpoint_state", default="raw_last_epoch187")
    parser.add_argument("--target_scale", type=float, default=1.0)
    parser.add_argument("--output_alpha", type=float, default=0.125)
    parser.add_argument("--k_values", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--sampling_seed", type=int, default=43)
    parser.add_argument("--ddim_steps", type=int, default=50)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--execution_horizon", type=int, default=8)
    parser.add_argument("--anchoring_horizon", type=int, default=8)
    parser.add_argument("--num_cpu_workers", type=int, default=8)
    parser.add_argument("--gpu_batch_size", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--include_difficult_paths",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--path_ids", nargs="*", default=None)
    parser.add_argument("--max_paths", type=int, default=None)
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.checkpoint_state != "raw_last_epoch187":
        raise ValueError("--checkpoint_state must be raw_last_epoch187")
    if args.target_scale != 1.0:
        raise ValueError("The frozen evaluation requires --target_scale 1.0")
    if args.output_alpha != 0.125:
        raise ValueError("The frozen evaluation requires --output_alpha 0.125")
    if tuple(args.k_values) != FROZEN_K_VALUES:
        raise ValueError("--k_values must be exactly 1 4 8")
    if args.ddim_steps != 50 or args.eta != 0.0:
        raise ValueError("The frozen sampler requires 50 DDIM steps and eta=0")
    if args.horizon != 32:
        raise ValueError("The frozen v8 horizon is 32")
    if args.execution_horizon != 8 or args.anchoring_horizon != 8:
        raise ValueError("Execution and anchoring horizons must both be 8")
    if args.num_cpu_workers < 1 or args.gpu_batch_size < 1:
        raise ValueError("CPU workers and GPU batch size must be positive")
    if args.max_paths is not None and args.max_paths < 1:
        raise ValueError("--max_paths must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")


def smoothstep(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, 0.0, 1.0)
    return clipped * clipped * (3.0 - 2.0 * clipped)


def padded_window(values: np.ndarray, start: int, horizon: int) -> np.ndarray:
    indices = np.minimum(np.arange(start, start + horizon), len(values) - 1)
    return np.asarray(values[indices], dtype=np.float64).copy()


def build_anchored_prior_window(
    strong_prior_q: np.ndarray,
    start_index: int,
    current_q: np.ndarray,
    horizon: int = 32,
    anchoring_horizon: int = 8,
    lower: Optional[np.ndarray] = None,
    upper: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Bridge the executed state exactly back to the frozen strong prior."""

    base = padded_window(strong_prior_q, start_index, horizon)
    current = np.asarray(current_q, dtype=np.float64).reshape(6)
    offset = current - base[0]
    phase = np.arange(horizon, dtype=np.float64) / anchoring_horizon
    weights = 1.0 - smoothstep(phase)
    weights[anchoring_horizon:] = 0.0
    if lower is not None and upper is not None:
        lower_bounds = np.asarray(lower, dtype=np.float64).reshape(6)
        upper_bounds = np.asarray(upper, dtype=np.float64).reshape(6)
        bounded_weights = weights[:, None] * np.ones((horizon, 6), dtype=np.float64)
        for joint in range(6):
            joint_offset = float(offset[joint])
            if abs(joint_offset) <= EPS:
                continue
            for timestep in range(1, horizon):
                if joint_offset > 0.0:
                    allowed = (upper_bounds[joint] - base[timestep, joint]) / joint_offset
                else:
                    allowed = (lower_bounds[joint] - base[timestep, joint]) / joint_offset
                bounded_weights[timestep, joint] = min(
                    bounded_weights[timestep, joint],
                    max(0.0, float(allowed)),
                )
            bounded_weights[1:, joint] = np.minimum.accumulate(
                bounded_weights[1:, joint]
            )
        bounded_weights[0, :] = 1.0
        bounded_weights[anchoring_horizon:, :] = 0.0
        anchored = base + bounded_weights * offset[None, :]
    else:
        anchored = base + weights[:, None] * offset
    anchored[0] = current
    anchored[anchoring_horizon:] = base[anchoring_horizon:]
    if not np.array_equal(anchored[0], current):
        raise AssertionError("Anchored prior does not start at current state")
    if not np.array_equal(
        anchored[anchoring_horizon:], base[anchoring_horizon:]
    ):
        raise AssertionError("Anchoring offset does not return exactly to zero")
    return anchored


def select_paths(
    population: Sequence[PhysicalPathRecord],
    args: argparse.Namespace,
) -> List[PhysicalPathRecord]:
    by_id = {path.path_id: path for path in population}
    ordinary = [path for path in population if path.population == "ordinary"]
    difficult = [path for path in population if path.population == "difficult"]
    if len(ordinary) != FROZEN_PRIMARY_PATH_COUNT:
        raise ValueError(f"Expected 20 ordinary validation paths, found {len(ordinary)}")
    if tuple(path.path_id for path in difficult) != tuple(DIFFICULT_PATHS):
        raise ValueError("Difficult population is not path_0306/path_0370")
    if args.path_ids is not None:
        requested = list(dict.fromkeys(args.path_ids))
        if len(requested) != len(args.path_ids):
            raise ValueError("--path_ids contains duplicates")
        missing = [path_id for path_id in requested if path_id not in by_id]
        if missing:
            raise ValueError(f"Unknown --path_ids: {missing}")
        selected = [by_id[path_id] for path_id in requested]
    elif args.smoke_test:
        selected = ordinary[:2]
    else:
        selected = list(ordinary)
        if args.include_difficult_paths:
            selected.extend(difficult)
    if args.max_paths is not None:
        selected = selected[: args.max_paths]
    return selected


def stable_candidate_seeds(
    global_seed: int,
    inference: ValidatedInferenceBundle,
    path_id: str,
    rollout_step: int,
    condition_norm: np.ndarray,
) -> Tuple[int, ...]:
    condition_hash = hashlib.sha256(
        np.ascontiguousarray(condition_norm).tobytes()
    ).hexdigest()
    seeds = []
    for candidate_index in range(FROZEN_CANDIDATE_COUNT):
        payload = json.dumps(
            [
                global_seed,
                inference.checkpoint_state_hash,
                path_id,
                rollout_step,
                condition_hash,
                candidate_index,
            ],
            separators=(",", ":"),
        ).encode("utf-8")
        seeds.append(
            int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
            & ((1 << 63) - 1)
        )
    return tuple(seeds)


def shifted_action_window(values: np.ndarray) -> np.ndarray:
    """Drop the already-executed boundary state and preserve length 32."""

    return np.concatenate((values[1:], values[-1:]), axis=0)


def make_action_context(
    path: PhysicalPathRecord,
    start_index: int,
    current_q: np.ndarray,
    previous_executed_q: Optional[np.ndarray],
    anchored_prior: np.ndarray,
    desired_window: np.ndarray,
    robot: target_generator.RobotContext,
    execution_count: int,
) -> target_generator.WindowContext:
    action_prior = shifted_action_window(anchored_prior)
    action_desired = shifted_action_window(desired_window)
    prior_ee = compute_fk_positions(robot, action_prior)
    tail_index = min(execution_count, len(action_prior) - 1)
    return target_generator.WindowContext(
        path_name=path.path_id,
        path_index=path.path_index,
        window_start=start_index + 1,
        prior_q=action_prior,
        desired=action_desired,
        prior_ee=prior_ee,
        previous_q=np.asarray(current_q, dtype=np.float64).copy(),
        previous_previous_q=(
            None
            if previous_executed_q is None
            else np.asarray(previous_executed_q, dtype=np.float64).copy()
        ),
        tail_q=action_prior[tail_index].copy(),
        tail_next_q=(
            action_prior[tail_index + 1].copy()
            if tail_index + 1 < len(action_prior)
            else None
        ),
    )


def candidate_row(
    result: v7_evaluator.CandidateEvaluationResult,
    *,
    path: PhysicalPathRecord,
    k: int,
    rollout_step: int,
    start_index: int,
    candidate_index: int,
    sampling_seed: int,
    selected: bool,
) -> Dict[str, Any]:
    decision = result.decision
    row: Dict[str, Any] = {
        "candidate_id": result.candidate_id,
        "path_id": path.path_id,
        "population": path.population,
        "k": k,
        "rollout_step": rollout_step,
        "window_start_index": start_index,
        "candidate_index": candidate_index,
        "candidate_sampling_seed": sampling_seed,
        "hard_safe": int(decision.hard_safe),
        "cartesian_improving": int(decision.cartesian_improving),
        "negative_delta_score": int(decision.delta_score < 0.0),
        "compatibility_gates_pass": int(decision.selectable),
        "selectable": int(sample_is_selectable(result)),
        "selected": int(selected),
        "delta_score": decision.delta_score,
        "improvement_m": decision.improvement_m,
        "acceptance_reasons": "|".join(decision.acceptance_reasons),
        "hard_safety_reasons": "|".join(decision.hard_safety_reasons),
        "evaluation_time_s": result.evaluation_time_s,
    }
    row.update(target_generator.flatten_metrics("candidate", result.metrics))
    return row


def recursive_executed_prefix_hard_safety_reasons(
    metrics: Mapping[str, Any],
) -> Tuple[str, ...]:
    """Hard gates for the segment that will actually be written to rollout_q.

    The validated v7 metrics intentionally include full 32-step lookahead
    checks for teacher-forced candidate selection. Recursive fallback only
    executes the first E action samples, so the fallback safety guard must not
    reject because of non-executed padded lookahead near the end of a path.
    """

    if not bool(metrics.get("finite", False)):
        return ("nonfinite_values",)
    reasons: List[str] = []
    if int(metrics.get("prefix_hard_joint_limit_violation_count", 1)) > 0:
        reasons.append("hard_joint_limit_violation")
    executed_step = max(
        float(metrics.get("prefix_maximum_absolute_joint_step_rad", 0.0)),
        float(metrics.get("entry_boundary_step_max_abs_rad", 0.0)),
    )
    if (
        executed_step
        > MAXIMUM_JOINT_STEP_RAD
        + target_generator.HARD_JOINT_LIMIT_TOLERANCE_RAD
    ):
        reasons.append("maximum_joint_step_gate")
    return tuple(reasons)


def evaluate_candidates(
    *,
    path: PhysicalPathRecord,
    k: int,
    rollout_step: int,
    start_index: int,
    current_q: np.ndarray,
    previous_executed_q: Optional[np.ndarray],
    anchored_prior: np.ndarray,
    desired_window: np.ndarray,
    residuals: np.ndarray,
    candidate_seeds: Sequence[int],
    output_alpha: float,
    execution_count: int,
    robot: target_generator.RobotContext,
    executor: Optional[concurrent.futures.ProcessPoolExecutor],
) -> Tuple[
    Optional[int],
    np.ndarray,
    Dict[str, Any],
    List[Dict[str, Any]],
]:
    context = make_action_context(
        path,
        start_index,
        current_q,
        previous_executed_q,
        anchored_prior,
        desired_window,
        robot,
        execution_count,
    )
    prior_metrics = v7_evaluator.evaluate_metrics(
        robot, context, context.prior_q, execution_count
    )
    prior_hard = recursive_executed_prefix_hard_safety_reasons(prior_metrics)
    if prior_hard:
        raise RuntimeError(
            f"{path.path_id}@{start_index}: anchored fallback executed prefix "
            f"is hard-unsafe: {list(prior_hard)}"
        )
    candidates = anchored_prior[None, :, :] + output_alpha * residuals
    candidates[:, 0, :] = current_q
    action_candidates = np.stack(
        [shifted_action_window(candidate) for candidate in candidates], axis=0
    )
    candidate_ids = [
        f"{path.path_id}::K={k}::step={rollout_step}::candidate={index}"
        for index in range(k)
    ]
    tasks = [
        v7_evaluator.CandidateEvaluationTask(
            candidate_id=candidate_ids[index],
            context=context,
            candidate_q=action_candidates[index].copy(),
            execution_horizon=execution_count,
            prior_metrics=dict(prior_metrics),
        )
        for index in range(k)
    ]
    results = v7_evaluator.evaluate_candidate_tasks(tasks, robot, executor)
    selected_index = select_nested_candidate(results, k)
    if selected_index is None:
        executed = anchored_prior[1 : execution_count + 1]
    else:
        if not sample_is_selectable(results[selected_index]):
            raise AssertionError("Selected candidate is not selectable")
        executed = candidates[selected_index, 1 : execution_count + 1]
    rows = [
        candidate_row(
            result,
            path=path,
            k=k,
            rollout_step=rollout_step,
            start_index=start_index,
            candidate_index=index,
            sampling_seed=candidate_seeds[index],
            selected=index == selected_index,
        )
        for index, result in enumerate(results)
    ]
    return selected_index, executed, dict(prior_metrics), rows


def validate_executed_indices(indices: Sequence[int], length: int) -> None:
    expected = list(range(length))
    actual = list(indices)
    if actual != expected:
        duplicates = sorted(
            index for index in set(actual) if actual.count(index) > 1
        )
        missing = sorted(set(expected) - set(actual))
        raise AssertionError(
            f"Execution indexing failed: duplicates={duplicates}, missing={missing}"
        )


def linear_slope(values: np.ndarray) -> float:
    if len(values) < 2:
        return 0.0
    return float(
        np.polyfit(
            np.arange(len(values), dtype=np.float64),
            np.asarray(values, dtype=np.float64),
            1,
        )[0]
    )


def run_rollout(
    path: PhysicalPathRecord,
    k: int,
    inference: ValidatedInferenceBundle,
    robot: target_generator.RobotContext,
    executor: Optional[concurrent.futures.ProcessPoolExecutor],
    args: argparse.Namespace,
    sample_cache: MutableMapping[Tuple[bytes, Tuple[int, ...]], np.ndarray],
) -> RolloutResult:
    length = len(path.strong_prior_q)
    rollout_q = np.empty_like(path.strong_prior_q)
    rollout_q[0] = path.strong_prior_q[0]
    executed_source = np.full(length, "", dtype="<U32")
    executed_source[0] = "initial_prior_state"
    correction_norms = np.zeros(length, dtype=np.float64)
    executed_indices = [0]
    decisions: List[Dict[str, Any]] = []
    candidate_rows: List[Dict[str, Any]] = []
    accepted_mask: List[bool] = []
    fallback_mask: List[bool] = []
    selected_indices: List[int] = []
    starts: List[int] = []
    boundary_offsets: List[float] = []
    segment_mean_corrections: List[float] = []
    segment_max_corrections: List[float] = []
    join_joint_step_norms: List[float] = []
    join_max_absolute_joint_steps: List[float] = []
    join_joint_acceleration_norms: List[float] = []
    cumulative_selected_local_delta_score = 0.0
    start = 0
    rollout_step = 0

    while start < length - 1:
        current_q = rollout_q[start].copy()
        previous_q = rollout_q[start - 1].copy() if start > 0 else None
        anchored = build_anchored_prior_window(
            path.strong_prior_q,
            start,
            current_q,
            args.horizon,
            args.anchoring_horizon,
            lower=robot.lower,
            upper=robot.upper,
        )
        desired = padded_window(path.desired_path, start, args.horizon)
        condition = build_recursive_condition_norm(
            inference,
            path,
            anchored,
            desired,
            current_q,
            start,
            args.target_scale,
            robot,
        )
        seeds = stable_candidate_seeds(
            args.sampling_seed, inference, path.path_id, rollout_step, condition
        )
        cache_key = (
            np.ascontiguousarray(condition).tobytes(),
            seeds,
        )
        residuals = sample_cache.get(cache_key)
        if residuals is None:
            residuals = sample_ddim_candidates(
                inference,
                condition,
                seeds,
                ddim_steps=args.ddim_steps,
                eta=args.eta,
                gpu_batch_size=args.gpu_batch_size,
            )
            sample_cache[cache_key] = residuals
        execution_count = min(args.execution_horizon, length - 1 - start)
        selected, executed, prior_metrics, rows = evaluate_candidates(
            path=path,
            k=k,
            rollout_step=rollout_step,
            start_index=start,
            current_q=current_q,
            previous_executed_q=previous_q,
            anchored_prior=anchored,
            desired_window=desired,
            residuals=residuals,
            candidate_seeds=seeds,
            output_alpha=args.output_alpha,
            execution_count=execution_count,
            robot=robot,
            executor=executor,
        )
        destination = list(range(start + 1, start + execution_count + 1))
        if len(destination) != len(executed):
            raise AssertionError("Execution prefix length mismatch")
        if set(destination) & set(executed_indices):
            raise AssertionError("A trajectory sample would be written twice")
        rollout_q[destination] = executed
        source = (
            "anchored_prior_fallback"
            if selected is None
            else "accepted_diffusion_candidate"
        )
        executed_source[destination] = source
        correction_norms[destination] = np.linalg.norm(
            executed - path.strong_prior_q[destination], axis=1
        )
        segment_correction = correction_norms[destination]
        boundary_offset = float(
            np.linalg.norm(current_q - path.strong_prior_q[start])
        )
        segment_mean_correction = float(np.mean(segment_correction))
        segment_max_correction = float(np.max(segment_correction))
        if rollout_step > 0:
            if previous_q is None:
                raise AssertionError("A recursive join lacks its previous sample")
            join_step = executed[0] - current_q
            join_acceleration = executed[0] - 2.0 * current_q + previous_q
            join_joint_step_norm = float(np.linalg.norm(join_step))
            join_max_absolute_joint_step = float(np.max(np.abs(join_step)))
            join_joint_acceleration_norm = float(
                np.linalg.norm(join_acceleration)
            )
            join_joint_step_norms.append(join_joint_step_norm)
            join_max_absolute_joint_steps.append(
                join_max_absolute_joint_step
            )
            join_joint_acceleration_norms.append(
                join_joint_acceleration_norm
            )
        else:
            join_joint_step_norm = float("nan")
            join_max_absolute_joint_step = float("nan")
            join_joint_acceleration_norm = float("nan")
        selected_local_delta_score = (
            0.0
            if selected is None
            else float(rows[selected]["delta_score"])
        )
        cumulative_selected_local_delta_score += selected_local_delta_score
        boundary_offsets.append(boundary_offset)
        segment_mean_corrections.append(segment_mean_correction)
        segment_max_corrections.append(segment_max_correction)
        executed_indices.extend(destination)
        accepted = selected is not None
        accepted_mask.append(accepted)
        fallback_mask.append(not accepted)
        selected_indices.append(-1 if selected is None else selected)
        starts.append(start)
        decisions.append(
            {
                "path_id": path.path_id,
                "population": path.population,
                "k": k,
                "rollout_step": rollout_step,
                "window_start_index": start,
                "execution_start_index": destination[0],
                "execution_end_index": destination[-1],
                "executed_count": execution_count,
                "generated_candidate_count": FROZEN_CANDIDATE_COUNT,
                "evaluated_candidate_count": k,
                "selected_candidate_index": (
                    -1 if selected is None else selected
                ),
                "accepted": int(accepted),
                "fallback": int(not accepted),
                "anchoring_offset_norm": boundary_offset,
                "boundary_anchoring_offset_norm": boundary_offset,
                "executed_segment_mean_correction_norm": (
                    segment_mean_correction
                ),
                "executed_segment_max_correction_norm": (
                    segment_max_correction
                ),
                "selected_local_delta_score": selected_local_delta_score,
                "cumulative_selected_local_delta_score": (
                    cumulative_selected_local_delta_score
                ),
                "join_joint_step_norm": join_joint_step_norm,
                "join_max_absolute_joint_step": (
                    join_max_absolute_joint_step
                ),
                "join_joint_acceleration_norm": (
                    join_joint_acceleration_norm
                ),
                "prior_prefix_cartesian_mean_error_m": float(
                    prior_metrics["prefix_cartesian_mean_error_m"]
                ),
                "fallback_prefix_hard_joint_limit_violation_count": int(
                    prior_metrics.get(
                        "prefix_hard_joint_limit_violation_count", 0
                    )
                ),
                "fallback_prefix_hard_joint_limit_violation_magnitude": float(
                    prior_metrics.get(
                        "prefix_hard_joint_limit_violation_magnitude", 0.0
                    )
                ),
                "fallback_prefix_maximum_absolute_joint_step_rad": float(
                    prior_metrics.get(
                        "prefix_maximum_absolute_joint_step_rad", 0.0
                    )
                ),
                "fallback_entry_boundary_step_max_abs_rad": float(
                    prior_metrics.get("entry_boundary_step_max_abs_rad", 0.0)
                ),
            }
        )
        candidate_rows.extend(rows)
        start = destination[-1]
        rollout_step += 1

    validate_executed_indices(executed_indices, length)
    if np.any(executed_source == ""):
        raise AssertionError("At least one trajectory sample has no executed source")
    full_metrics = compute_full_trajectory_metrics(
        robot=robot,
        strong_prior_q=path.strong_prior_q,
        rollout_q=rollout_q,
        desired_path=path.desired_path,
    )
    rollout_ee = np.asarray(full_metrics.pop("rollout_ee"), dtype=np.float64)
    prior_ee = np.asarray(full_metrics.pop("prior_ee"), dtype=np.float64)
    if not np.allclose(prior_ee, path.prior_ee, rtol=1.0e-5, atol=2.0e-5):
        raise ValueError(f"{path.path_id}: authoritative prior FK mismatch")
    cartesian_error_delta = (
        np.linalg.norm(rollout_ee - path.desired_path, axis=1)
        - np.linalg.norm(prior_ee - path.desired_path, axis=1)
    )
    full_metrics.update(
        {
            "path_id": path.path_id,
            "population": path.population,
            "k": k,
            "rollout_step_count": len(decisions),
            "accepted_rollout_step_count": int(np.sum(accepted_mask)),
            "accepted_rollout_step_rate": float(np.mean(accepted_mask)),
            "fallback_count": int(np.sum(fallback_mask)),
            "fallback_rate": float(np.mean(fallback_mask)),
            "correction_growth_slope": linear_slope(correction_norms),
            "boundary_anchoring_offset_growth_slope_per_step": linear_slope(
                np.asarray(boundary_offsets, dtype=np.float64)
            ),
            "segment_mean_correction_growth_slope_per_step": linear_slope(
                np.asarray(segment_mean_corrections, dtype=np.float64)
            ),
            "segment_max_correction_growth_slope_per_step": linear_slope(
                np.asarray(segment_max_corrections, dtype=np.float64)
            ),
            "cartesian_error_delta_growth_slope": linear_slope(
                cartesian_error_delta
            ),
            "sum_selected_local_delta_score": (
                cumulative_selected_local_delta_score
            ),
            "full_path_recomputed_delta_score": float(
                full_metrics["total_robot_aware_delta_score"]
            ),
            "local_vs_full_delta_score_gap": float(
                full_metrics["total_robot_aware_delta_score"]
                - cumulative_selected_local_delta_score
            ),
            "legacy_local_vs_full_delta_score_gap": float(
                full_metrics["legacy_full_path_robot_aware_delta_score"]
                - cumulative_selected_local_delta_score
            ),
            "internal_local_vs_full_delta_score_gap": float(
                full_metrics["internal_full_path_robot_aware_delta_score"]
                - cumulative_selected_local_delta_score
            ),
            "mean_join_joint_step_norm": float(
                np.mean(join_joint_step_norms)
                if join_joint_step_norms
                else 0.0
            ),
            "max_join_joint_step_norm": float(
                np.max(join_joint_step_norms)
                if join_joint_step_norms
                else 0.0
            ),
            "max_join_absolute_joint_step": float(
                np.max(join_max_absolute_joint_steps)
                if join_max_absolute_joint_steps
                else 0.0
            ),
            "mean_join_joint_acceleration_norm": float(
                np.mean(join_joint_acceleration_norms)
                if join_joint_acceleration_norms
                else 0.0
            ),
            "max_join_joint_acceleration_norm": float(
                np.max(join_joint_acceleration_norms)
                if join_joint_acceleration_norms
                else 0.0
            ),
            "path_improved_robot_aware": int(
                float(
                    full_metrics[
                        "internal_full_path_robot_aware_delta_score"
                    ]
                )
                < 0.0
            ),
        }
    )
    return RolloutResult(
        path=path,
        k=k,
        rollout_q=rollout_q,
        rollout_ee=rollout_ee,
        executed_source=executed_source,
        accepted_step_mask=np.asarray(accepted_mask, dtype=bool),
        fallback_step_mask=np.asarray(fallback_mask, dtype=bool),
        selected_candidate_indices=np.asarray(selected_indices, dtype=np.int64),
        applied_correction_norms=correction_norms,
        window_start_indices=np.asarray(starts, dtype=np.int64),
        executed_indices=np.asarray(executed_indices, dtype=np.int64),
        decision_rows=decisions,
        candidate_rows=candidate_rows,
        metrics=full_metrics,
    )


def save_trajectory_npz(output_dir: Path, result: RolloutResult) -> None:
    path_dir = output_dir / "trajectories" / result.path.path_id
    path_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path_dir / f"anchored_rollout_k{result.k}.npz",
        strong_prior_q=result.path.strong_prior_q,
        rollout_q=result.rollout_q,
        desired_path=result.path.desired_path,
        prior_ee=result.path.prior_ee,
        rollout_ee=result.rollout_ee,
        executed_source=result.executed_source,
        accepted_step_mask=result.accepted_step_mask,
        fallback_step_mask=result.fallback_step_mask,
        selected_candidate_indices=result.selected_candidate_indices,
        applied_correction_norms=result.applied_correction_norms,
        window_start_indices=result.window_start_indices,
        executed_indices=result.executed_indices,
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
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(json_safe(row.get(key)), separators=(",", ":"))
                        if isinstance(row.get(key), (dict, list, tuple, np.ndarray))
                        else json_safe(row.get(key))
                    )
                    for key in fields
                }
            )


def aggregate_rows(
    path_rows: Sequence[Mapping[str, Any]],
    population: str,
) -> List[Dict[str, Any]]:
    rows = [
        row
        for row in path_rows
        if population == "combined_diagnostic"
        or str(row["population"]) == population
    ]
    contribution_metrics = tuple(
        sorted(
            key
            for row in rows
            for key in row
            if "robot_score_contribution_" in key
        )
    )
    metrics = (
        "full_path_safety_pass",
        "accepted_rollout_step_rate",
        "fallback_rate",
        "total_robot_aware_delta_score",
        "legacy_full_path_robot_aware_delta_score",
        "internal_full_path_robot_aware_delta_score",
        "sum_selected_local_delta_score",
        "full_path_recomputed_delta_score",
        "local_vs_full_delta_score_gap",
        "legacy_local_vs_full_delta_score_gap",
        "internal_local_vs_full_delta_score_gap",
        "terminal_joint_deviation_norm_rad",
        "terminal_joint_deviation_max_rad",
        "terminal_cartesian_deviation_from_prior_m",
        "legacy_terminal_boundary_step_contribution",
        "legacy_terminal_boundary_acceleration_contribution",
        "mean_join_joint_step_norm",
        "max_join_joint_step_norm",
        "max_join_absolute_joint_step",
        "mean_join_joint_acceleration_norm",
        "max_join_joint_acceleration_norm",
        "maximum_actual_internal_joint_step_rad",
        "cartesian_mean_error_delta",
        "cartesian_rms_error_delta",
        "cartesian_max_error_delta",
        "velocity_cost_delta",
        "acceleration_cost_delta",
        "jerk_cost_delta",
        "joint_limit_cost_delta",
        "singularity_cost_delta",
        "minimum_manipulability_delta",
        "correction_growth_slope",
        "boundary_anchoring_offset_growth_slope_per_step",
        "segment_mean_correction_growth_slope_per_step",
        "segment_max_correction_growth_slope_per_step",
        "cartesian_error_delta_growth_slope",
        "path_improved_robot_aware",
        *contribution_metrics,
    )
    output: List[Dict[str, Any]] = []
    for k in FROZEN_K_VALUES:
        subset = [row for row in rows if int(row["k"]) == k]
        if not subset:
            continue
        aggregate: Dict[str, Any] = {
            "population": population,
            "k": k,
            "path_count": len(subset),
        }
        for metric in metrics:
            values = np.asarray(
                [float(row[metric]) for row in subset], dtype=np.float64
            )
            aggregate[f"mean_{metric}"] = float(np.mean(values))
            aggregate[f"std_{metric}"] = float(np.std(values))
            aggregate[f"median_{metric}"] = float(np.median(values))
            aggregate[f"max_{metric}"] = float(np.max(values))
        output.append(aggregate)
    return output


def save_path_plots(output_dir: Path, result: RolloutResult) -> None:
    plot_dir = output_dir / "plots" / result.path.path_id / f"k{result.k}"
    plot_dir.mkdir(parents=True, exist_ok=True)
    depth = np.arange(len(result.rollout_q))
    prior_error = np.linalg.norm(
        result.path.prior_ee - result.path.desired_path, axis=1
    )
    rollout_error = np.linalg.norm(
        result.rollout_ee - result.path.desired_path, axis=1
    )

    figure, axis = plt.subplots(figsize=(9, 4))
    axis.plot(depth, prior_error, label="strong prior")
    axis.plot(depth, rollout_error, label="anchored rollout")
    axis.set(xlabel="trajectory depth", ylabel="Cartesian error (m)")
    axis.legend()
    figure.tight_layout()
    figure.savefig(str(plot_dir / "cartesian_error_over_depth.png"), dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 4))
    axis.plot(depth, result.applied_correction_norms)
    axis.set(xlabel="trajectory depth", ylabel="joint correction norm (rad)")
    figure.tight_layout()
    figure.savefig(str(plot_dir / "applied_correction_norm.png"), dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 2.8))
    timeline = np.zeros(len(result.rollout_q), dtype=np.float64)
    timeline[result.executed_source == "accepted_diffusion_candidate"] = 1.0
    axis.step(depth, timeline, where="post")
    axis.set(
        xlabel="trajectory depth",
        ylabel="source",
        yticks=(0, 1),
        yticklabels=("fallback", "accepted"),
    )
    figure.tight_layout()
    figure.savefig(str(plot_dir / "accepted_fallback_timeline.png"), dpi=150)
    plt.close(figure)

    figure, axes = plt.subplots(3, 2, figsize=(11, 8), sharex=True)
    for joint, axis in enumerate(axes.flat):
        axis.plot(depth, result.path.strong_prior_q[:, joint], label="prior")
        axis.plot(depth, result.rollout_q[:, joint], label="rollout")
        axis.set_ylabel(f"q{joint + 1} (rad)")
    axes.flat[0].legend()
    axes.flat[-1].set_xlabel("trajectory depth")
    figure.tight_layout()
    figure.savefig(str(plot_dir / "six_joint_trajectories.png"), dpi=150)
    plt.close(figure)

    figure, axes = plt.subplots(3, 2, figsize=(11, 8), sharex=True)
    deviation = result.rollout_q - result.path.strong_prior_q
    for joint, axis in enumerate(axes.flat):
        axis.plot(depth, deviation[:, joint])
        axis.set_ylabel(f"dq{joint + 1} (rad)")
    axes.flat[-1].set_xlabel("trajectory depth")
    figure.tight_layout()
    figure.savefig(str(plot_dir / "joint_deviation_from_prior.png"), dpi=150)
    plt.close(figure)

    cartesian_component = np.cumsum(np.square(rollout_error))
    velocity_component = np.concatenate(
        (
            np.zeros(1),
            np.cumsum(np.sum(np.square(np.diff(result.rollout_q, axis=0)), axis=1)),
        )
    )
    acceleration = np.diff(result.rollout_q, n=2, axis=0)
    acceleration_component = np.concatenate(
        (
            np.zeros(2),
            np.cumsum(np.sum(np.square(acceleration), axis=1)),
        )
    )
    jerk = np.diff(result.rollout_q, n=3, axis=0)
    jerk_component = np.concatenate(
        (
            np.zeros(3),
            np.cumsum(np.sum(np.square(jerk), axis=1)),
        )
    )
    figure, axis = plt.subplots(figsize=(9, 4))
    axis.plot(depth, cartesian_component, label="Cartesian squared error")
    axis.plot(depth, velocity_component, label="velocity")
    axis.plot(depth, acceleration_component, label="acceleration")
    axis.plot(depth, jerk_component, label="jerk")
    axis.set(xlabel="trajectory depth", ylabel="cumulative cost component")
    axis.legend()
    figure.tight_layout()
    figure.savefig(str(plot_dir / "cumulative_cost_components.png"), dpi=150)
    plt.close(figure)


def save_manipulability_plot(
    output_dir: Path,
    result: RolloutResult,
    robot: target_generator.RobotContext,
) -> None:
    depth = np.arange(len(result.rollout_q))
    prior_values: List[float] = []
    rollout_values: List[float] = []
    for prior_q, rollout_q in zip(result.path.strong_prior_q, result.rollout_q):
        prior_values.append(
            target_generator.manipulability_and_penalty(
                target_generator.positional_jacobian(robot, prior_q)
            )[0]
        )
        rollout_values.append(
            target_generator.manipulability_and_penalty(
                target_generator.positional_jacobian(robot, rollout_q)
            )[0]
        )
    plot_dir = output_dir / "plots" / result.path.path_id / f"k{result.k}"
    figure, axis = plt.subplots(figsize=(9, 4))
    axis.plot(depth, prior_values, label="strong prior")
    axis.plot(depth, rollout_values, label="anchored rollout")
    axis.set(xlabel="trajectory depth", ylabel="positional manipulability")
    axis.legend()
    figure.tight_layout()
    figure.savefig(str(plot_dir / "manipulability_over_depth.png"), dpi=150)
    plt.close(figure)


def save_aggregate_plots(
    output_dir: Path,
    path_rows: Sequence[Mapping[str, Any]],
) -> None:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    ordinary_k8 = [
        row
        for row in path_rows
        if row["population"] == "ordinary" and int(row["k"]) == 8
    ]
    names = [str(row["path_id"]) for row in ordinary_k8]
    for metric, filename, ylabel in (
        (
            "total_robot_aware_delta_score",
            "per_path_total_robot_aware_delta_score.png",
            "robot-aware delta score",
        ),
        (
            "cartesian_mean_error_delta",
            "per_path_cartesian_mean_error_delta.png",
            "Cartesian mean-error delta (m)",
        ),
    ):
        figure, axis = plt.subplots(figsize=(12, 4))
        axis.bar(np.arange(len(names)), [float(row[metric]) for row in ordinary_k8])
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_xticks(np.arange(len(names)), names, rotation=90)
        axis.set_ylabel(ylabel)
        figure.tight_layout()
        figure.savefig(str(plot_dir / filename), dpi=150)
        plt.close(figure)
    figure, axis = plt.subplots(figsize=(7, 4))
    for k in FROZEN_K_VALUES:
        values = [
            float(row["total_robot_aware_delta_score"])
            for row in path_rows
            if row["population"] == "ordinary" and int(row["k"]) == k
        ]
        axis.scatter(np.full(len(values), k), values, alpha=0.7)
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set(
        xticks=FROZEN_K_VALUES,
        xlabel="nested K",
        ylabel="full-path robot-aware delta score",
    )
    figure.tight_layout()
    figure.savefig(str(plot_dir / "k_comparison.png"), dpi=150)
    plt.close(figure)

    contribution_fields = (
        "robot_score_contribution_cart_mean",
        "robot_score_contribution_cart_p95",
        "robot_score_contribution_cart_max",
        "robot_score_contribution_acceleration",
        "robot_score_contribution_jerk",
        "robot_score_contribution_boundary_step",
        "robot_score_contribution_boundary_acceleration",
        "robot_score_contribution_singularity",
    )
    figure, axis = plt.subplots(figsize=(15, 6))
    x = np.arange(len(ordinary_k8), dtype=np.float64)
    width = 0.8 / len(contribution_fields)
    for component_index, field in enumerate(contribution_fields):
        offset = (component_index - (len(contribution_fields) - 1) / 2.0) * width
        axis.bar(
            x + offset,
            [float(row[field]) for row in ordinary_k8],
            width=width,
            label=field.removeprefix("robot_score_contribution_"),
        )
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xticks(x, names, rotation=90)
    axis.set_ylabel("weighted normalized score contribution")
    axis.legend(ncol=4)
    figure.tight_layout()
    figure.savefig(
        str(plot_dir / "ordinary_k8_score_contributions.png"), dpi=150
    )
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(12, 4))
    axis.bar(
        x - 0.26,
        [
            float(row["sum_selected_local_delta_score"])
            for row in ordinary_k8
        ],
        width=0.26,
        label="sum selected local",
    )
    axis.bar(
        x,
        [
            float(row["internal_full_path_robot_aware_delta_score"])
            for row in ordinary_k8
        ],
        width=0.26,
        label="internal full path",
    )
    axis.bar(
        x + 0.26,
        [
            float(row["legacy_full_path_robot_aware_delta_score"])
            for row in ordinary_k8
        ],
        width=0.26,
        label="legacy full path",
    )
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xticks(x, names, rotation=90)
    axis.set_ylabel("robot-aware delta score")
    axis.legend()
    figure.tight_layout()
    figure.savefig(
        str(plot_dir / "local_sum_vs_full_path_score.png"), dpi=150
    )
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(12, 4))
    axis.bar(
        x,
        [
            float(row["terminal_joint_deviation_norm_rad"])
            for row in ordinary_k8
        ],
    )
    axis.set_xticks(x, names, rotation=90)
    axis.set_ylabel("terminal joint deviation norm (rad)")
    figure.tight_layout()
    figure.savefig(
        str(plot_dir / "terminal_joint_deviation.png"), dpi=150
    )
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(12, 4))
    axis.bar(
        x - 0.2,
        [
            float(
                row[
                    "internal_robot_score_contribution_boundary_step"
                ]
            )
            for row in ordinary_k8
        ],
        width=0.4,
        label="internal boundary contribution",
    )
    axis.bar(
        x + 0.2,
        [
            float(
                row[
                    "legacy_robot_score_contribution_boundary_step"
                ]
            )
            for row in ordinary_k8
        ],
        width=0.4,
        label="legacy terminal boundary contribution",
    )
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xticks(x, names, rotation=90)
    axis.set_ylabel("weighted normalized score contribution")
    axis.legend()
    figure.tight_layout()
    figure.savefig(
        str(plot_dir / "internal_vs_legacy_boundary_contribution.png"),
        dpi=150,
    )
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(12, 4))
    axis.bar(
        x,
        [
            float(row["max_join_absolute_joint_step"])
            for row in ordinary_k8
        ],
    )
    axis.axhline(
        MAXIMUM_JOINT_STEP_RAD,
        color="red",
        linestyle="--",
        label="hard-safety threshold",
    )
    axis.set_xticks(x, names, rotation=90)
    axis.set_ylabel("maximum actual join joint step (rad)")
    axis.legend()
    figure.tight_layout()
    figure.savefig(
        str(plot_dir / "actual_join_maximum_joint_step.png"), dpi=150
    )
    plt.close(figure)


def largest_positive_score_contribution(
    path_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    ordinary_k8 = [
        row
        for row in path_rows
        if row["population"] == "ordinary" and int(row["k"]) == 8
    ]
    result: Dict[str, Any] = {}
    for mode in ("internal", "legacy"):
        prefix = f"{mode}_robot_score_contribution_"
        fields = sorted(
            {
                key
                for row in ordinary_k8
                for key in row
                if key.startswith(prefix)
                and key
                not in {
                    f"{prefix}sum",
                    f"{mode}_robot_score_decomposition_residual",
                }
            }
        )
        means = {
            field: float(
                np.mean([float(row[field]) for row in ordinary_k8])
            )
            for field in fields
        }
        if not means:
            result[mode] = {
                "component": "",
                "mean_contribution": float("nan"),
            }
            continue
        component, value = max(means.items(), key=lambda item: item[1])
        result[mode] = {
            "component": component.removeprefix(prefix),
            "field": component,
            "mean_contribution": value,
        }
    return result


def single_seed_decision_inputs(
    path_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    rows = [
        row
        for row in path_rows
        if row["population"] == "ordinary" and int(row["k"]) == 8
    ]
    if len(rows) != FROZEN_PRIMARY_PATH_COUNT:
        return {
            "decision_population_complete": False,
            "note": "Final decision requires all 20 ordinary K=8 paths.",
        }
    return {
        "decision_population_complete": True,
        "ordinary_k8_path_count": len(rows),
        "all_finally_safe": all(
            int(row["full_path_safety_pass"]) == 1 for row in rows
        ),
        "fraction_paths_robot_aware_improved": float(
            np.mean(
                [
                    float(
                        row[
                            "internal_full_path_robot_aware_delta_score"
                        ]
                    )
                    < 0.0
                    for row in rows
                ]
            )
        ),
        "fraction_paths_internal_full_path_score_negative": float(
            np.mean(
                [
                    float(
                        row[
                            "internal_full_path_robot_aware_delta_score"
                        ]
                    )
                    < 0.0
                    for row in rows
                ]
            )
        ),
        "mean_internal_full_path_robot_aware_delta_score": float(
            np.mean(
                [
                    float(
                        row[
                            "internal_full_path_robot_aware_delta_score"
                        ]
                    )
                    for row in rows
                ]
            )
        ),
        "mean_legacy_full_path_robot_aware_delta_score": float(
            np.mean(
                [
                    float(
                        row[
                            "legacy_full_path_robot_aware_delta_score"
                        ]
                    )
                    for row in rows
                ]
            )
        ),
        "mean_total_robot_aware_delta_score": float(
            np.mean(
                [
                    float(
                        row[
                            "legacy_full_path_robot_aware_delta_score"
                        ]
                    )
                    for row in rows
                ]
            )
        ),
        "mean_sum_selected_local_delta_score": float(
            np.mean(
                [float(row["sum_selected_local_delta_score"]) for row in rows]
            )
        ),
        "mean_full_path_recomputed_delta_score": float(
            np.mean(
                [float(row["full_path_recomputed_delta_score"]) for row in rows]
            )
        ),
        "mean_local_vs_full_delta_score_gap": float(
            np.mean(
                [float(row["local_vs_full_delta_score_gap"]) for row in rows]
            )
        ),
        "mean_cartesian_mean_error_delta": float(
            np.mean([float(row["cartesian_mean_error_delta"]) for row in rows])
        ),
        "mean_boundary_anchoring_offset_growth_slope_per_step": float(
            np.mean(
                [
                    float(
                        row[
                            "boundary_anchoring_offset_growth_slope_per_step"
                        ]
                    )
                    for row in rows
                ]
            )
        ),
        "mean_segment_mean_correction_growth_slope_per_step": float(
            np.mean(
                [
                    float(
                        row[
                            "segment_mean_correction_growth_slope_per_step"
                        ]
                    )
                    for row in rows
                ]
            )
        ),
        "mean_segment_max_correction_growth_slope_per_step": float(
            np.mean(
                [
                    float(
                        row[
                            "segment_max_correction_growth_slope_per_step"
                        ]
                    )
                    for row in rows
                ]
            )
        ),
        "mean_cartesian_error_delta_growth_slope": float(
            np.mean(
                [
                    float(row["cartesian_error_delta_growth_slope"])
                    for row in rows
                ]
            )
        ),
        "maximum_allowed_segment_correction_growth_rad_per_step": (
            MAX_ALLOWED_SEGMENT_CORRECTION_GROWTH_RAD_PER_STEP
        ),
        "segment_correction_growth_tolerance_pass": bool(
            np.mean(
                [
                    float(
                        row[
                            "segment_max_correction_growth_slope_per_step"
                        ]
                    )
                    for row in rows
                ]
            )
            <= MAX_ALLOWED_SEGMENT_CORRECTION_GROWTH_RAD_PER_STEP
        ),
        "maximum_actual_internal_joint_step_rad": float(
            np.max(
                [
                    float(row["maximum_actual_internal_joint_step_rad"])
                    for row in rows
                ]
            )
        ),
        "maximum_actual_internal_joint_step_gate_rad": (
            MAXIMUM_JOINT_STEP_RAD
        ),
        "maximum_actual_internal_joint_step_gate_pass": bool(
            np.max(
                [
                    float(row["maximum_actual_internal_joint_step_rad"])
                    for row in rows
                ]
            )
            <= MAXIMUM_JOINT_STEP_RAD
        ),
        "note": (
            "This is one repeated stochastic evaluation seed. The advance rule "
            "is evaluated only by the five-seed summarizer."
        ),
    }


def main() -> int:
    args = parse_args()
    validate_args(args)
    started = time.perf_counter()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory is nonempty: {args.output_dir}; pass --overwrite"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    population = load_authoritative_physical_path_population(
        args.dataset_dir,
        args.target_generation_dir,
        include_difficult_paths=True,
    )
    selected_paths = select_paths(population, args)
    inference = load_validated_inference_bundle(
        args.dataset_dir,
        args.model_dir,
        args.checkpoint_state,
        args.device,
        args.ddim_steps,
    )
    robot = v7_evaluator.make_robot_context(Path(DEFAULT_URDF_PATH))
    executor: Optional[concurrent.futures.ProcessPoolExecutor] = None
    if args.num_cpu_workers > 1:
        executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=args.num_cpu_workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=v7_evaluator.initialize_candidate_worker,
            initargs=(str(Path(DEFAULT_URDF_PATH)),),
        )
    path_rows: List[Dict[str, Any]] = []
    decision_rows: List[Dict[str, Any]] = []
    candidate_rows: List[Dict[str, Any]] = []
    try:
        for path_index, path in enumerate(selected_paths, start=1):
            sample_cache: Dict[Tuple[bytes, Tuple[int, ...]], np.ndarray] = {}
            for k in FROZEN_K_VALUES:
                result = run_rollout(
                    path,
                    k,
                    inference,
                    robot,
                    executor,
                    args,
                    sample_cache,
                )
                save_trajectory_npz(args.output_dir, result)
                save_path_plots(args.output_dir, result)
                save_manipulability_plot(args.output_dir, result, robot)
                path_rows.append(result.metrics)
                decision_rows.extend(result.decision_rows)
                candidate_rows.extend(result.candidate_rows)
            print(f"Completed {path_index}/{len(selected_paths)} paths: {path.path_id}")
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=False)

    ordinary = aggregate_rows(path_rows, "ordinary")
    difficult = aggregate_rows(path_rows, "difficult")
    combined = aggregate_rows(path_rows, "combined_diagnostic")
    write_csv(args.output_dir / "anchored_rollout_decisions.csv", decision_rows)
    write_csv(args.output_dir / "anchored_candidate_results.csv", candidate_rows)
    write_csv(args.output_dir / "anchored_full_path_metrics.csv", path_rows)
    write_csv(args.output_dir / "anchored_ordinary_aggregate.csv", ordinary)
    write_csv(args.output_dir / "anchored_difficult_aggregate.csv", difficult)
    write_csv(
        args.output_dir / "anchored_combined_diagnostic_aggregate.csv", combined
    )
    save_aggregate_plots(args.output_dir, path_rows)
    decision_inputs = single_seed_decision_inputs(path_rows)
    largest_positive_contribution = largest_positive_score_contribution(path_rows)
    summary = {
        "status": "complete",
        "sampling_seed": args.sampling_seed,
        "checkpoint_state": inference.checkpoint_state,
        "checkpoint_epoch": inference.checkpoint_epoch,
        "checkpoint_state_hash": inference.checkpoint_state_hash,
        "path_ids": [path.path_id for path in selected_paths],
        "ordinary_path_count": sum(
            path.population == "ordinary" for path in selected_paths
        ),
        "difficult_path_count": sum(
            path.population == "difficult" for path in selected_paths
        ),
        "ordinary": ordinary,
        "difficult": difficult,
        "combined_diagnostic": combined,
        "score_semantics": {
            "legacy": (
                "Includes a diagnostic transition from the rollout endpoint "
                "to the strong-prior endpoint."
            ),
            "internal": (
                "Evaluates the physically executed complete trajectory "
                "without an artificial post-terminal transition."
            ),
        },
        "largest_mean_positive_score_contribution_ordinary_k8": (
            largest_positive_contribution
        ),
        "provisional_decision_inputs": decision_inputs,
        "wall_time_seconds": time.perf_counter() - started,
    }
    (args.output_dir / "anchored_rollout_summary.json").write_text(
        json.dumps(json_safe(summary), indent=2, sort_keys=True) + "\n"
    )
    report = [
        "Diffusion v8 anchored recursive rollout",
        f"sampling_seed: {args.sampling_seed}",
        f"checkpoint: {inference.checkpoint_state}",
        f"ordinary paths: {summary['ordinary_path_count']}",
        f"difficult paths: {summary['difficult_path_count']}",
        "decision population: ordinary K=8 only",
        (
            "legacy score: includes a diagnostic transition from rollout "
            "endpoint to strong-prior endpoint"
        ),
        (
            "internal score: evaluates the physically executed complete "
            "trajectory without that artificial post-terminal transition"
        ),
        (
            "largest mean positive score contribution (ordinary K=8): "
            f"{largest_positive_contribution}"
        ),
        json.dumps(json_safe(decision_inputs), indent=2, sort_keys=True),
    ]
    (args.output_dir / "anchored_rollout_report.txt").write_text(
        "\n".join(report) + "\n"
    )
    if args.smoke_test:
        print("V8_ANCHORED_RECURSIVE_SMOKE_TEST_COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
