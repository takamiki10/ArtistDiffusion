#!/usr/bin/env python3
"""v8.1 anchored recursive rollout with history-aware jerk guard.

This script is a separate development experiment.  It reuses the validated v8
anchored rollout machinery for model loading, condition construction, DDIM
sampling, anchoring, FK, hard-safety checks, compatibility gates, full-path
metrics, files, and plots.  The only selection change is an executed-history
jerk guard applied before choosing among otherwise selectable candidates.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import matplotlib
import numpy as np

matplotlib.use("Agg")

import evaluate_diffusion_v7_teacher_forced_validation as v7_evaluator
import evaluate_diffusion_v8_anchored_recursive_rollout as v8
from evaluate_diffusion_v8_teacher_forced_all_windows import (
    ValidatedInferenceBundle,
    build_recursive_condition_norm,
    compute_full_trajectory_metrics,
    load_authoritative_physical_path_population,
    load_validated_inference_bundle,
    sample_ddim_candidates,
    sample_is_selectable,
)
import generate_diffusion_v7_cost_improving_residual_targets as target_generator
from generate_ik_seed_path import DEFAULT_URDF_PATH


HISTORY_AWARE_JERK_TOLERANCE = 1.0e-12


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
        default=Path("results/diffusion_v8_1_anchored_recursive_jerk_guard"),
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
    parser.add_argument(
        "--disable_history_aware_jerk_guard",
        action="store_true",
        help=(
            "Diagnostic mode: reproduce v8 candidate eligibility and selection "
            "while still reporting history-aware jerk diagnostics."
        ),
    )
    return parser.parse_args()


def executed_history(rollout_q: np.ndarray, start_index: int) -> np.ndarray:
    first = max(0, int(start_index) - 2)
    return np.asarray(rollout_q[first : start_index + 1], dtype=np.float64).copy()


def history_aware_incremental_jerk_cost(
    history_q: np.ndarray,
    execution_prefix_q: np.ndarray,
) -> float:
    """Use the same np.diff(..., n=3) convention as v7 derivative_cost.

    Only jerk stencils whose newest sample belongs to the proposed execution
    prefix are included.  With up to three history states prepended, these are
    exactly the newly realized stencils caused by executing this prefix.
    """

    history = np.asarray(history_q, dtype=np.float64).reshape(-1, 6)
    prefix = np.asarray(execution_prefix_q, dtype=np.float64).reshape(-1, 6)
    if len(prefix) == 0:
        return 0.0
    sequence = np.concatenate((history, prefix), axis=0)
    jerk = np.diff(sequence, n=3, axis=0)
    if not jerk.size:
        return 0.0
    newest_indices = np.arange(3, 3 + len(jerk), dtype=np.int64)
    realized = jerk[newest_indices >= len(history)]
    if not realized.size:
        return 0.0
    return float(np.mean(np.sum(np.square(realized), axis=1)))


def select_v8_1_candidate(
    results: Sequence[v7_evaluator.CandidateEvaluationResult],
    jerk_deltas: Sequence[float],
    k: int,
    guard_disabled: bool,
) -> Optional[int]:
    if guard_disabled:
        return v8.select_nested_candidate(results, k)
    eligible = [
        index
        for index, result in enumerate(results[:k])
        if sample_is_selectable(result)
        and float(jerk_deltas[index]) <= HISTORY_AWARE_JERK_TOLERANCE
    ]
    if not eligible:
        return None
    minimum_score = min(float(results[index].decision.delta_score) for index in eligible)
    score_tied = [
        index
        for index in eligible
        if np.isclose(
            float(results[index].decision.delta_score),
            minimum_score,
            rtol=0.0,
            atol=HISTORY_AWARE_JERK_TOLERANCE,
        )
    ]
    return min(
        score_tied,
        key=lambda index: (float(jerk_deltas[index]), index),
    )


def candidate_row_v8_1(
    result: v7_evaluator.CandidateEvaluationResult,
    *,
    path: v8.PhysicalPathRecord,
    k: int,
    rollout_step: int,
    start_index: int,
    candidate_index: int,
    sampling_seed: int,
    selected: bool,
    candidate_jerk_cost: float,
    fallback_jerk_cost: float,
    jerk_delta: float,
    guard_pass: bool,
    guard_disabled: bool,
) -> Dict[str, Any]:
    row = v8.candidate_row(
        result,
        path=path,
        k=k,
        rollout_step=rollout_step,
        start_index=start_index,
        candidate_index=candidate_index,
        sampling_seed=sampling_seed,
        selected=selected,
    )
    base_selectable = sample_is_selectable(result)
    final_selectable = base_selectable and (guard_disabled or guard_pass)
    rejected_only_by_guard = (
        base_selectable
        and not guard_disabled
        and not guard_pass
    )
    row.update(
        {
            "v8_1_history_aware_jerk_guard_enabled": int(not guard_disabled),
            "v8_selectable_before_history_aware_jerk_guard": int(base_selectable),
            "v8_1_selectable_after_history_aware_jerk_guard": int(final_selectable),
            "history_aware_candidate_incremental_jerk_cost": candidate_jerk_cost,
            "history_aware_fallback_incremental_jerk_cost": fallback_jerk_cost,
            "history_aware_incremental_jerk_delta": jerk_delta,
            "history_aware_jerk_guard_pass": int(guard_pass),
            "rejected_only_by_history_aware_jerk_guard": int(
                rejected_only_by_guard
            ),
            "v8_1_rejection_reasons": (
                "history_aware_jerk_worsening"
                if rejected_only_by_guard
                else ""
            ),
        }
    )
    row["selectable"] = int(final_selectable)
    return row


def aggregate_rows_v8_1(
    path_rows: Sequence[Mapping[str, Any]],
    population: str,
) -> List[Dict[str, Any]]:
    rows = [
        row
        for row in path_rows
        if population == "combined_diagnostic"
        or str(row["population"]) == population
    ]
    output: List[Dict[str, Any]] = []
    for k in v8.FROZEN_K_VALUES:
        subset = [row for row in rows if int(row["k"]) == k]
        if not subset:
            continue
        numeric_fields = []
        for row in subset:
            for key, value in row.items():
                if key in {"path_id", "population", "k"}:
                    continue
                try:
                    float(value)
                except (TypeError, ValueError):
                    continue
                numeric_fields.append(key)
        aggregate: Dict[str, Any] = {
            "population": population,
            "k": k,
            "path_count": len(subset),
        }
        for metric in sorted(set(numeric_fields)):
            values = np.asarray(
                [float(row[metric]) for row in subset if metric in row],
                dtype=np.float64,
            )
            if not values.size:
                continue
            aggregate[f"mean_{metric}"] = float(np.mean(values))
            aggregate[f"std_{metric}"] = float(np.std(values))
            aggregate[f"median_{metric}"] = float(np.median(values))
            aggregate[f"max_{metric}"] = float(np.max(values))
        output.append(aggregate)
    return output


def evaluate_candidates_v8_1(
    *,
    path: v8.PhysicalPathRecord,
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
    history_q: np.ndarray,
    guard_disabled: bool,
) -> Tuple[
    Optional[int],
    np.ndarray,
    Dict[str, Any],
    List[Dict[str, Any]],
    Dict[str, Any],
]:
    context = v8.make_action_context(
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
    prior_hard = v8.recursive_executed_prefix_hard_safety_reasons(prior_metrics)
    if prior_hard:
        raise RuntimeError(
            f"{path.path_id}@{start_index}: anchored fallback executed prefix "
            f"is hard-unsafe: {list(prior_hard)}"
        )

    candidates = anchored_prior[None, :, :] + output_alpha * residuals
    candidates[:, 0, :] = current_q
    action_candidates = np.stack(
        [v8.shifted_action_window(candidate) for candidate in candidates], axis=0
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

    fallback_prefix = anchored_prior[1 : execution_count + 1]
    fallback_jerk_cost = history_aware_incremental_jerk_cost(
        history_q, fallback_prefix
    )
    candidate_jerk_costs: List[float] = []
    jerk_deltas: List[float] = []
    guard_passes: List[bool] = []
    for index in range(k):
        prefix = candidates[index, 1 : execution_count + 1]
        cost = history_aware_incremental_jerk_cost(history_q, prefix)
        delta = cost - fallback_jerk_cost
        candidate_jerk_costs.append(cost)
        jerk_deltas.append(delta)
        guard_passes.append(delta <= HISTORY_AWARE_JERK_TOLERANCE)

    selected_index = select_v8_1_candidate(
        results, jerk_deltas, k, guard_disabled
    )
    if selected_index is None:
        executed = fallback_prefix
        selected_jerk_delta = 0.0
        selected_jerk_cost = float(fallback_jerk_cost)
    else:
        if not sample_is_selectable(results[selected_index]):
            raise AssertionError("Selected candidate is not v8 selectable")
        if (
            not guard_disabled
            and jerk_deltas[selected_index] > HISTORY_AWARE_JERK_TOLERANCE
        ):
            raise AssertionError("Selected candidate failed v8.1 jerk guard")
        executed = candidates[selected_index, 1 : execution_count + 1]
        selected_jerk_delta = float(jerk_deltas[selected_index])
        selected_jerk_cost = float(candidate_jerk_costs[selected_index])

    rows = [
        candidate_row_v8_1(
            result,
            path=path,
            k=k,
            rollout_step=rollout_step,
            start_index=start_index,
            candidate_index=index,
            sampling_seed=candidate_seeds[index],
            selected=index == selected_index,
            candidate_jerk_cost=float(candidate_jerk_costs[index]),
            fallback_jerk_cost=float(fallback_jerk_cost),
            jerk_delta=float(jerk_deltas[index]),
            guard_pass=bool(guard_passes[index]),
            guard_disabled=guard_disabled,
        )
        for index, result in enumerate(results)
    ]
    step_info = {
        "history_aware_fallback_incremental_jerk_cost": float(fallback_jerk_cost),
        "history_aware_jerk_rejection_count": int(
            sum(
                sample_is_selectable(result)
                and not guard_disabled
                and not guard_passes[index]
                for index, result in enumerate(results)
            )
        ),
        "v8_selectable_candidate_count_before_jerk_guard": int(
            sum(sample_is_selectable(result) for result in results)
        ),
        "history_aware_jerk_guard_pass_count": int(sum(guard_passes)),
        "selected_history_aware_jerk_delta": selected_jerk_delta,
        "selected_history_aware_candidate_incremental_jerk_cost": (
            selected_jerk_cost
        ),
        "selected_history_aware_jerk_guard_pass": int(
            selected_index is None
            or guard_disabled
            or guard_passes[selected_index]
        ),
    }
    return selected_index, executed, dict(prior_metrics), rows, step_info


def run_rollout_v8_1(
    path: v8.PhysicalPathRecord,
    k: int,
    inference: ValidatedInferenceBundle,
    robot: target_generator.RobotContext,
    executor: Optional[concurrent.futures.ProcessPoolExecutor],
    args: argparse.Namespace,
    sample_cache: MutableMapping[Tuple[bytes, Tuple[int, ...]], np.ndarray],
) -> v8.RolloutResult:
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
    selected_jerk_deltas: List[float] = []
    selected_accepted_jerk_deltas: List[float] = []
    history_aware_jerk_rejection_count = 0
    v8_selectable_candidate_count = 0
    total_candidate_count = 0
    cumulative_selected_local_delta_score = 0.0
    start = 0
    rollout_step = 0

    while start < length - 1:
        current_q = rollout_q[start].copy()
        previous_q = rollout_q[start - 1].copy() if start > 0 else None
        anchored = v8.build_anchored_prior_window(
            path.strong_prior_q,
            start,
            current_q,
            args.horizon,
            args.anchoring_horizon,
        )
        desired = v8.padded_window(path.desired_path, start, args.horizon)
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
        seeds = v8.stable_candidate_seeds(
            args.sampling_seed, inference, path.path_id, rollout_step, condition
        )
        cache_key = (np.ascontiguousarray(condition).tobytes(), seeds)
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
        history_q = executed_history(rollout_q, start)
        selected, executed, prior_metrics, rows, step_info = evaluate_candidates_v8_1(
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
            history_q=history_q,
            guard_disabled=args.disable_history_aware_jerk_guard,
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
        boundary_offset = float(np.linalg.norm(current_q - path.strong_prior_q[start]))
        segment_mean_correction = float(np.mean(segment_correction))
        segment_max_correction = float(np.max(segment_correction))
        if rollout_step > 0:
            if previous_q is None:
                raise AssertionError("A recursive join lacks its previous sample")
            join_step = executed[0] - current_q
            join_acceleration = executed[0] - 2.0 * current_q + previous_q
            join_joint_step_norm = float(np.linalg.norm(join_step))
            join_max_absolute_joint_step = float(np.max(np.abs(join_step)))
            join_joint_acceleration_norm = float(np.linalg.norm(join_acceleration))
            join_joint_step_norms.append(join_joint_step_norm)
            join_max_absolute_joint_steps.append(join_max_absolute_joint_step)
            join_joint_acceleration_norms.append(join_joint_acceleration_norm)
        else:
            join_joint_step_norm = float("nan")
            join_max_absolute_joint_step = float("nan")
            join_joint_acceleration_norm = float("nan")

        selected_local_delta_score = 0.0 if selected is None else float(rows[selected]["delta_score"])
        cumulative_selected_local_delta_score += selected_local_delta_score
        selected_jerk_delta = float(step_info["selected_history_aware_jerk_delta"])
        selected_jerk_deltas.append(selected_jerk_delta)
        history_aware_jerk_rejection_count += int(
            step_info["history_aware_jerk_rejection_count"]
        )
        v8_selectable_candidate_count += int(
            step_info["v8_selectable_candidate_count_before_jerk_guard"]
        )
        total_candidate_count += k
        boundary_offsets.append(boundary_offset)
        segment_mean_corrections.append(segment_mean_correction)
        segment_max_corrections.append(segment_max_correction)
        executed_indices.extend(destination)
        accepted = selected is not None
        if accepted:
            selected_accepted_jerk_deltas.append(selected_jerk_delta)
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
                "generated_candidate_count": v8.FROZEN_CANDIDATE_COUNT,
                "evaluated_candidate_count": k,
                "selected_candidate_index": -1 if selected is None else selected,
                "accepted": int(accepted),
                "fallback": int(not accepted),
                "v8_1_history_aware_jerk_guard_enabled": int(
                    not args.disable_history_aware_jerk_guard
                ),
                "history_aware_fallback_incremental_jerk_cost": float(
                    step_info["history_aware_fallback_incremental_jerk_cost"]
                ),
                "history_aware_candidate_incremental_jerk_cost": float(
                    step_info[
                        "selected_history_aware_candidate_incremental_jerk_cost"
                    ]
                ),
                "history_aware_incremental_jerk_delta": selected_jerk_delta,
                "history_aware_jerk_guard_pass": int(
                    step_info["selected_history_aware_jerk_guard_pass"]
                ),
                "history_aware_jerk_rejection_count": int(
                    step_info["history_aware_jerk_rejection_count"]
                ),
                "v8_selectable_candidate_count_before_jerk_guard": int(
                    step_info["v8_selectable_candidate_count_before_jerk_guard"]
                ),
                "history_aware_jerk_guard_pass_count": int(
                    step_info["history_aware_jerk_guard_pass_count"]
                ),
                "selected_history_aware_jerk_delta": selected_jerk_delta,
                "selected_history_aware_jerk_guard_pass": int(
                    step_info["selected_history_aware_jerk_guard_pass"]
                ),
                "anchoring_offset_norm": boundary_offset,
                "boundary_anchoring_offset_norm": boundary_offset,
                "executed_segment_mean_correction_norm": segment_mean_correction,
                "executed_segment_max_correction_norm": segment_max_correction,
                "selected_local_delta_score": selected_local_delta_score,
                "cumulative_selected_local_delta_score": cumulative_selected_local_delta_score,
                "join_joint_step_norm": join_joint_step_norm,
                "join_max_absolute_joint_step": join_max_absolute_joint_step,
                "join_joint_acceleration_norm": join_joint_acceleration_norm,
                "prior_prefix_cartesian_mean_error_m": float(
                    prior_metrics["prefix_cartesian_mean_error_m"]
                ),
                "fallback_prefix_hard_joint_limit_violation_count": int(
                    prior_metrics.get("prefix_hard_joint_limit_violation_count", 0)
                ),
                "fallback_prefix_hard_joint_limit_violation_magnitude": float(
                    prior_metrics.get("prefix_hard_joint_limit_violation_magnitude", 0.0)
                ),
                "fallback_prefix_maximum_absolute_joint_step_rad": float(
                    prior_metrics.get("prefix_maximum_absolute_joint_step_rad", 0.0)
                ),
                "fallback_entry_boundary_step_max_abs_rad": float(
                    prior_metrics.get("entry_boundary_step_max_abs_rad", 0.0)
                ),
            }
        )
        candidate_rows.extend(rows)
        start = destination[-1]
        rollout_step += 1

    v8.validate_executed_indices(executed_indices, length)
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
    selected_delta_array = np.asarray(selected_jerk_deltas, dtype=np.float64)
    selected_accepted_delta_array = np.asarray(
        selected_accepted_jerk_deltas, dtype=np.float64
    )
    jerk_rejection_rate_all = float(
        history_aware_jerk_rejection_count / max(total_candidate_count, 1)
    )
    jerk_rejection_rate_selectable = (
        float(history_aware_jerk_rejection_count / v8_selectable_candidate_count)
        if v8_selectable_candidate_count > 0
        else float("nan")
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
            "v8_1_history_aware_jerk_guard_enabled": int(
                not args.disable_history_aware_jerk_guard
            ),
            "evaluated_candidate_count_total": int(total_candidate_count),
            "v8_selectable_candidate_count_before_jerk_guard": int(
                v8_selectable_candidate_count
            ),
            "history_aware_jerk_rejection_count": history_aware_jerk_rejection_count,
            "history_aware_jerk_rejection_rate_all_evaluated": (
                jerk_rejection_rate_all
            ),
            "history_aware_jerk_rejection_rate_among_v8_selectable": (
                jerk_rejection_rate_selectable
            ),
            "history_aware_jerk_rejection_rate": jerk_rejection_rate_selectable,
            "selected_history_aware_jerk_delta_sum": float(
                np.sum(selected_delta_array)
            ),
            "selected_history_aware_jerk_delta_max_including_fallback": float(
                np.max(selected_delta_array) if selected_delta_array.size else 0.0
            ),
            "selected_history_aware_jerk_delta_max": float(
                np.max(selected_delta_array) if selected_delta_array.size else 0.0
            ),
            "selected_history_aware_jerk_delta_max_accepted_only": float(
                np.max(selected_accepted_delta_array)
                if selected_accepted_delta_array.size
                else 0.0
            ),
            "selected_history_aware_jerk_delta_mean_accepted_only": float(
                np.mean(selected_accepted_delta_array)
                if selected_accepted_delta_array.size
                else 0.0
            ),
            "correction_growth_slope": v8.linear_slope(correction_norms),
            "boundary_anchoring_offset_growth_slope_per_step": v8.linear_slope(
                np.asarray(boundary_offsets, dtype=np.float64)
            ),
            "segment_mean_correction_growth_slope_per_step": v8.linear_slope(
                np.asarray(segment_mean_corrections, dtype=np.float64)
            ),
            "segment_max_correction_growth_slope_per_step": v8.linear_slope(
                np.asarray(segment_max_corrections, dtype=np.float64)
            ),
            "cartesian_error_delta_growth_slope": v8.linear_slope(cartesian_error_delta),
            "sum_selected_local_delta_score": cumulative_selected_local_delta_score,
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
                np.mean(join_joint_step_norms) if join_joint_step_norms else 0.0
            ),
            "max_join_joint_step_norm": float(
                np.max(join_joint_step_norms) if join_joint_step_norms else 0.0
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
                float(full_metrics["internal_full_path_robot_aware_delta_score"])
                < 0.0
            ),
        }
    )
    return v8.RolloutResult(
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


def main() -> int:
    args = parse_args()
    v8.validate_args(args)
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
    selected_paths = v8.select_paths(population, args)
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
            for k in v8.FROZEN_K_VALUES:
                result = run_rollout_v8_1(
                    path,
                    k,
                    inference,
                    robot,
                    executor,
                    args,
                    sample_cache,
                )
                v8.save_trajectory_npz(args.output_dir, result)
                v8.save_path_plots(args.output_dir, result)
                v8.save_manipulability_plot(args.output_dir, result, robot)
                path_rows.append(result.metrics)
                decision_rows.extend(result.decision_rows)
                candidate_rows.extend(result.candidate_rows)
            print(f"Completed {path_index}/{len(selected_paths)} paths: {path.path_id}")
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=False)

    ordinary = aggregate_rows_v8_1(path_rows, "ordinary")
    difficult = aggregate_rows_v8_1(path_rows, "difficult")
    combined = aggregate_rows_v8_1(path_rows, "combined_diagnostic")
    v8.write_csv(args.output_dir / "anchored_rollout_decisions.csv", decision_rows)
    v8.write_csv(args.output_dir / "anchored_candidate_results.csv", candidate_rows)
    v8.write_csv(args.output_dir / "anchored_full_path_metrics.csv", path_rows)
    v8.write_csv(args.output_dir / "anchored_ordinary_aggregate.csv", ordinary)
    v8.write_csv(args.output_dir / "anchored_difficult_aggregate.csv", difficult)
    v8.write_csv(
        args.output_dir / "anchored_combined_diagnostic_aggregate.csv", combined
    )
    v8.save_aggregate_plots(args.output_dir, path_rows)
    decision_inputs = v8.single_seed_decision_inputs(path_rows)
    largest_positive_contribution = v8.largest_positive_score_contribution(path_rows)
    summary = {
        "status": "complete",
        "experiment": "v8.1_history_aware_jerk_guard",
        "history_aware_jerk_guard_enabled": (
            not args.disable_history_aware_jerk_guard
        ),
        "history_aware_jerk_tolerance": HISTORY_AWARE_JERK_TOLERANCE,
        "sampling_seed": args.sampling_seed,
        "checkpoint_state": inference.checkpoint_state,
        "checkpoint_epoch": inference.checkpoint_epoch,
        "checkpoint_state_hash": inference.checkpoint_state_hash,
        "path_ids": [path.path_id for path in selected_paths],
        "ordinary_path_count": sum(path.population == "ordinary" for path in selected_paths),
        "difficult_path_count": sum(path.population == "difficult" for path in selected_paths),
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
        "regression_mode_note": (
            "When --disable_history_aware_jerk_guard is used, candidate "
            "seeds, trajectories, selected indices, rollout trajectories, "
            "local scores, and full-path metrics should match v8."
        ),
        "wall_time_seconds": time.perf_counter() - started,
    }
    (args.output_dir / "anchored_rollout_summary.json").write_text(
        json.dumps(v8.json_safe(summary), indent=2, sort_keys=True) + "\n"
    )
    report = [
        "Diffusion v8.1 anchored recursive rollout with history-aware jerk guard",
        f"sampling_seed: {args.sampling_seed}",
        f"checkpoint: {inference.checkpoint_state}",
        f"history-aware jerk guard enabled: {not args.disable_history_aware_jerk_guard}",
        f"history-aware jerk tolerance: {HISTORY_AWARE_JERK_TOLERANCE}",
        f"ordinary paths: {summary['ordinary_path_count']}",
        f"difficult paths: {summary['difficult_path_count']}",
        "decision population: ordinary K=8 only",
        (
            "Candidate generation, DDIM, anchoring, safety gates, v7 "
            "compatibility gates, and full-path internal score are reused "
            "from v8."
        ),
        json.dumps(v8.json_safe(decision_inputs), indent=2, sort_keys=True),
    ]
    (args.output_dir / "anchored_rollout_report.txt").write_text(
        "\n".join(report) + "\n"
    )
    print("classification: V8_1_ANCHORED_RECURSIVE_JERK_GUARD_COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
