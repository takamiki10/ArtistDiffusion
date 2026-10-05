#!/usr/bin/env python3
"""Audit scaled and boundary-tapered v5b action-buffer residual candidates.

This is a teacher-forced, offline diagnostic. Every planning cycle starts from
the buffer-only MLP plan; selected diffusion candidates are never propagated.
Expert joints are used only for columns prefixed with ``oracle_``.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import inspect
import json
import sys
from dataclasses import asdict, is_dataclass
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
from diagnose_action_buffer_candidate_selection import (
    CONDITION_DIM,
    JOINT_DIM,
    array_stats,
    build_teacher_forced_buffer,
    build_v5b_condition,
    decode_names,
    default_weights,
    desired_differences,
    discounted_score,
    evaluate_region,
    extract_joint_limits,
    finite_array,
    limit_metrics,
    load_npz,
    load_stats,
    prefixed,
    require_keys,
    resolve_device,
    set_seed,
    write_dict_csv,
)
from diagnose_diffusion_v5_sampling_modes import reverse_noised_x0_batches
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
    "scaled_tapered_action_buffer_candidate_diagnostic"
)
EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit scaled and tapered action-buffer diffusion candidates."
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--test_npz", type=Path, default=DEFAULT_TEST_NPZ)
    parser.add_argument("--stats_npz", type=Path, default=DEFAULT_STATS_NPZ)
    parser.add_argument("--prior_dir", type=Path, default=DEFAULT_PRIOR_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--prediction_horizon", type=int, default=32)
    parser.add_argument("--execution_horizon", type=int, default=8)
    parser.add_argument("--t_init", type=int, default=10)
    parser.add_argument("--num_base_samples", type=int, default=16)
    parser.add_argument("--max_paths", type=int, default=20)
    parser.add_argument(
        "--alphas",
        nargs="+",
        type=float,
        default=(0.05, 0.1, 0.25, 0.5, 1.0),
    )
    parser.add_argument(
        "--ramp_lengths",
        nargs="+",
        type=int,
        default=(0, 1, 4, 8),
    )
    parser.add_argument("--taper_mode", choices=("linear", "cosine"), default="linear")
    parser.add_argument(
        "--anchor_first",
        action="store_true",
        help="Force the residual at prediction timestep 0 to zero.",
    )
    parser.add_argument(
        "--anchor_executed_prefix_start",
        action="store_true",
        help="Force residuals at execution-block starts 0,E,2E,... to zero.",
    )
    parser.add_argument("--boundary_ratio", type=float, default=2.0)
    parser.add_argument("--prefix_step_ratio", type=float, default=2.0)
    parser.add_argument("--absolute_boundary_limit", type=float, default=0.25)
    parser.add_argument("--absolute_prefix_step_limit", type=float, default=0.25)
    parser.add_argument("--ranking_discount", type=float, default=0.9)
    parser.add_argument(
        "--continuity_weight",
        type=float,
        default=None,
        help="Override the rollout default; omitted uses the exact rollout value.",
    )
    parser.add_argument("--num_diffusion_steps", type=int, default=None)
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.prediction_horizon <= 0 or args.execution_horizon <= 0:
        raise ValueError("Prediction and execution horizons must be positive")
    if args.execution_horizon > args.prediction_horizon:
        raise ValueError("execution_horizon cannot exceed prediction_horizon")
    if args.num_base_samples <= 0 or args.max_paths <= 0:
        raise ValueError("num_base_samples and max_paths must be positive")
    if not args.alphas or any(not np.isfinite(value) or value <= 0.0 for value in args.alphas):
        raise ValueError("All --alphas values must be finite and positive")
    if len(set(float(value) for value in args.alphas)) != len(args.alphas):
        raise ValueError("--alphas contains duplicate values")
    if not args.ramp_lengths or any(value < 0 for value in args.ramp_lengths):
        raise ValueError("All --ramp_lengths values must be non-negative")
    if len(set(int(value) for value in args.ramp_lengths)) != len(args.ramp_lengths):
        raise ValueError("--ramp_lengths contains duplicate values")
    if not 0.0 < args.ranking_discount <= 1.0:
        raise ValueError("ranking_discount must be in (0,1]")
    if args.continuity_weight is not None and args.continuity_weight < 0.0:
        raise ValueError("continuity_weight must be non-negative")
    for name in (
        "boundary_ratio",
        "prefix_step_ratio",
        "absolute_boundary_limit",
        "absolute_prefix_step_limit",
    ):
        if float(getattr(args, name)) <= 0.0:
            raise ValueError(f"--{name} must be positive")


def taper_values(horizon: int, ramp_length: int, mode: str) -> np.ndarray:
    if ramp_length == 0:
        return np.ones(horizon, dtype=np.float32)
    progress = np.minimum(
        1.0,
        np.arange(horizon, dtype=np.float32) / float(ramp_length),
    )
    if mode == "linear":
        taper = progress
    elif mode == "cosine":
        taper = 0.5 * (1.0 - np.cos(np.pi * progress))
    else:
        raise ValueError(f"Unsupported taper mode: {mode}")
    taper = taper.astype(np.float32)
    if taper[0] != 0.0 or np.any(taper < 0.0) or np.any(taper > 1.0):
        raise RuntimeError("Nonzero-ramp taper must start at zero and remain in [0,1]")
    return taper


def apply_scaled_taper(
    residual: np.ndarray,
    alpha: float,
    taper: np.ndarray,
    anchor_first: bool,
    anchor_executed_prefix_start: bool,
    execution_horizon: int,
) -> np.ndarray:
    scaled = (
        float(alpha) * taper[:, None] * np.asarray(residual, dtype=np.float32)
    ).astype(np.float32)
    if anchor_first:
        scaled[0] = 0.0
    if anchor_executed_prefix_start:
        scaled[::execution_horizon] = 0.0
    finite_array(scaled, "scaled tapered physical residual")
    return scaled


class StrictRepositoryRanking:
    """Invoke the rollout's actual repository scoring helper without fallback."""

    FUNCTION_NAMES = (
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

    def __init__(self, rollout_module: ModuleType) -> None:
        self.function: Optional[Callable[..., Any]] = None
        for name in self.FUNCTION_NAMES:
            candidate = getattr(rollout_module, name, None)
            if callable(candidate):
                self.function = candidate
                break
        if self.function is None:
            available = sorted(
                name
                for name, value in vars(rollout_module).items()
                if callable(value) and ("score" in name or "cost" in name or "rank" in name)
            )
            raise RuntimeError(
                "Could not locate the warm-start rollout ranking function. "
                f"Searched={list(self.FUNCTION_NAMES)}, available={available}"
            )
        self.signature = inspect.signature(self.function)
        self.function_name = (
            f"{self.function.__module__}.{self.function.__qualname__}"
        )

    def score(self, supplied: Dict[str, Any]) -> float:
        required = [
            name
            for name, parameter in self.signature.parameters.items()
            if parameter.default is inspect.Parameter.empty
            and parameter.kind
            not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        ]
        missing = [name for name in required if name not in supplied]
        if missing:
            raise RuntimeError(
                "Repository ranking call is incomplete: "
                f"function={self.function_name}, required={required}, "
                f"missing={missing}, supplied={sorted(supplied)}"
            )
        positional: List[Any] = []
        keyword: Dict[str, Any] = {}
        for name, parameter in self.signature.parameters.items():
            if name not in supplied:
                continue
            if parameter.kind is inspect.Parameter.POSITIONAL_ONLY:
                positional.append(supplied[name])
            elif parameter.kind is not inspect.Parameter.VAR_POSITIONAL:
                keyword[name] = supplied[name]
        try:
            result = self.function(*positional, **keyword)
        except Exception as exc:
            raise RuntimeError(
                "Repository ranking function call failed: "
                f"function={self.function_name}, required={required}, "
                f"supplied={sorted(supplied)}, error={type(exc).__name__}: {exc}"
            ) from exc
        numeric_types = (float, int, np.floating, np.integer)
        score_fields = (
            "ranking_score",
            "ranking_cost",
            "score",
            "drawing_total_cost",
            "total_cost",
            "cost",
        )
        for _ in range(4):
            if isinstance(result, numeric_types):
                break
            previous = result
            if isinstance(result, dict):
                for key in score_fields:
                    if key in result:
                        result = result[key]
                        break
            elif isinstance(result, (tuple, list)) and result:
                result = result[0]
            else:
                for attribute in score_fields:
                    if hasattr(result, attribute):
                        result = getattr(result, attribute)
                        break
            if result is previous:
                break
        try:
            score = float(result)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Repository ranking function {self.function_name} returned "
                f"non-scalar result {type(result)!r}; "
                f"available_fields={sorted(vars(result)) if hasattr(result, '__dict__') else []}"
            ) from exc
        if not np.isfinite(score):
            raise RuntimeError(
                f"Repository ranking function {self.function_name} returned {score}"
            )
        return score


def rollout_default_namespace(module: ModuleType) -> argparse.Namespace:
    parse_function = getattr(module, "parse_args", None)
    if not callable(parse_function):
        raise RuntimeError(
            "diagnose_warm_start_action_buffer_rollout.parse_args is unavailable; "
            "cannot recover exact ranking weights"
        )
    signature = inspect.signature(parse_function)
    try:
        if len(signature.parameters) == 0:
            original_argv = sys.argv
            try:
                sys.argv = [str(getattr(module, "__file__", "rollout"))]
                defaults = parse_function()
            finally:
                sys.argv = original_argv
        else:
            defaults = parse_function([])
    except BaseException as exc:
        raise RuntimeError(
            "Could not recover exact ranking defaults from "
            "diagnose_warm_start_action_buffer_rollout.parse_args"
        ) from exc
    required = ("w_rms_cart", "w_limit_count", "continuity_weight")
    missing = [name for name in required if not hasattr(defaults, name)]
    if missing:
        raise RuntimeError(
            "Warm-start rollout arguments are missing exact ranking weights: "
            f"{missing}"
        )
    return defaults


class RepositoryFKAdapter:
    """Expose the FKComputer interface while preserving the project FK convention."""

    def __init__(
        self,
        robot: Any,
        joint_names: Sequence[str],
        ee_link: str,
        lower: np.ndarray,
        upper: np.ndarray,
    ) -> None:
        self.available = True
        self.robot = robot
        self.joint_names = list(joint_names)
        self.ee_link = ee_link
        self.lower = lower
        self.upper = upper

    def fk(self, raw_q: np.ndarray) -> np.ndarray:
        return fk_positions(
            self.robot,
            self.joint_names,
            self.ee_link,
            np.asarray(raw_q, dtype=np.float32),
        )

    def joint_limit_violation(self, raw_q: np.ndarray) -> float:
        _, violation_cost = limit_metrics(
            np.asarray(raw_q, dtype=np.float64),
            self.lower,
            self.upper,
        )
        return violation_cost


def repository_score_context(
    *,
    raw_q: np.ndarray,
    candidate_index: int,
    is_diffusion_refined: bool,
    desired_path: np.ndarray,
    ee_positions: np.ndarray,
    previous_q: np.ndarray,
    repository_fk: RepositoryFKAdapter,
    robot: Any,
    joint_names: Sequence[str],
    ee_link: str,
    lower: np.ndarray,
    upper: np.ndarray,
    weights: Any,
    w_rms_cart: float,
    w_limit_count: float,
    continuity_weight: float,
    args: argparse.Namespace,
    metrics: Dict[str, float],
) -> Dict[str, Any]:
    """Aliases cover the repository and rollout names without expert inputs."""
    return {
        "raw_q": raw_q,
        "candidate_index": int(candidate_index),
        "is_diffusion_refined": bool(is_diffusion_refined),
        "q": raw_q,
        "candidate_q": raw_q,
        "trajectory_q": raw_q,
        "desired_path": desired_path,
        "desired_window": desired_path,
        "desired": desired_path,
        "target_path": desired_path,
        "ee_positions": ee_positions,
        "candidate_ee": ee_positions,
        "ee_path": ee_positions,
        "fk_positions": ee_positions,
        "previous_q": previous_q,
        "previous_executed_q": previous_q,
        "current_q": previous_q,
        "fk": repository_fk,
        "robot": robot,
        "joint_names": joint_names,
        "ee_link": ee_link,
        "lower": lower,
        "upper": upper,
        "joint_lower": lower,
        "joint_upper": upper,
        "lower_limits": lower,
        "upper_limits": upper,
        "weights": weights,
        "cost_weights": weights,
        "ranking_weights": weights,
        "w_rms_cart": float(w_rms_cart),
        "w_limit_count": float(w_limit_count),
        "continuity_weight": float(continuity_weight),
        "horizon": int(raw_q.shape[0]),
        "prediction_horizon": int(raw_q.shape[0]),
        "args": args,
        "config": args,
        "metrics": metrics,
        "candidate_metrics": metrics,
        **metrics,
    }


def weights_record(weights: Any) -> Dict[str, float]:
    if is_dataclass(weights):
        raw = asdict(weights)
    elif hasattr(weights, "__dict__"):
        raw = vars(weights)
    else:
        return {"value": float(weights)}
    return {str(key): float(value) for key, value in raw.items()}


def practical_values_are_finite(row: Dict[str, Any]) -> bool:
    keys = (
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
        "repository_ranking_score",
        "discounted_score",
    )
    return bool(np.all(np.isfinite([float(row[key]) for key in keys])))


def safety_gate(
    row: Dict[str, Any],
    buffer_row: Dict[str, Any],
    args: argparse.Namespace,
) -> Tuple[bool, str]:
    if row["candidate_type"] == "unrefined_buffer":
        if not practical_values_are_finite(row):
            return False, "buffer_non_finite"
        if (
            float(row["prefix_joint_limit_violation_count"]) > 0.0
            or float(row["full_horizon_joint_limit_violation_count"]) > 0.0
        ):
            return False, "buffer_joint_limit"
        if float(row["boundary_max_abs_joint_step"]) > args.absolute_boundary_limit:
            return False, "buffer_boundary_step"
        if float(row["prefix_max_joint_step"]) > args.absolute_prefix_step_limit:
            return False, "buffer_prefix_step"
        return True, ""
    if not practical_values_are_finite(row):
        return False, "non_finite"
    if (
        float(row["prefix_joint_limit_violation_count"]) > 0.0
        or float(row["full_horizon_joint_limit_violation_count"]) > 0.0
    ):
        return False, "joint_limit"
    boundary_limit = min(
        args.absolute_boundary_limit,
        max(
            float(buffer_row["boundary_max_abs_joint_step"]) * args.boundary_ratio,
            EPS,
        ),
    )
    prefix_limit = min(
        args.absolute_prefix_step_limit,
        max(
            float(buffer_row["prefix_max_joint_step"]) * args.prefix_step_ratio,
            EPS,
        ),
    )
    if float(row["boundary_max_abs_joint_step"]) > boundary_limit:
        return False, "boundary_step"
    if float(row["prefix_max_joint_step"]) > prefix_limit:
        return False, "prefix_step"
    return True, ""


def select_minimum(
    rows: Sequence[Dict[str, Any]],
    score_key: str,
) -> int:
    eligible = [
        index
        for index, row in enumerate(rows)
        if row["candidate_type"] == "unrefined_buffer" or int(row["hard_gate_passed"]) == 1
    ]
    if 0 not in eligible:
        raise AssertionError("The unrefined buffer was omitted from practical selection")
    return min(eligible, key=lambda index: (float(rows[index][score_key]), index))


def select_lexicographic(rows: Sequence[Dict[str, Any]]) -> int:
    buffer = rows[0]
    eligible = [
        index
        for index, row in enumerate(rows)
        if row["candidate_type"] == "unrefined_buffer" or int(row["hard_gate_passed"]) == 1
    ]
    if 0 not in eligible:
        raise AssertionError("The unrefined buffer was omitted from lexicographic selection")

    def key(index: int) -> Tuple[int, int, float, float, int]:
        row = rows[index]
        cart_improved = (
            float(row["prefix_mean_cartesian_error"])
            < float(buffer["prefix_mean_cartesian_error"])
        )
        drawing_improved = (
            float(row["prefix_drawing_cost"])
            < float(buffer["prefix_drawing_cost"])
        )
        return (
            0 if cart_improved else 1,
            0 if drawing_improved else 1,
            float(row["prefix_drawing_cost"]),
            float(row["boundary_max_abs_joint_step"]),
            index,
        )

    return min(eligible, key=key)


def evaluate_candidate(
    *,
    candidate_q: np.ndarray,
    candidate_index: int,
    candidate_type: str,
    desired_window: np.ndarray,
    expert_window: np.ndarray,
    execution_count: int,
    previous_q: np.ndarray,
    repository_fk: RepositoryFKAdapter,
    robot: Any,
    joint_names: Sequence[str],
    ee_link: str,
    lower: np.ndarray,
    upper: np.ndarray,
    weights: Any,
    w_rms_cart: float,
    w_limit_count: float,
    args: argparse.Namespace,
    repository_ranking: StrictRepositoryRanking,
) -> Dict[str, Any]:
    finite_array(candidate_q, "candidate q")
    candidate_ee = fk_positions(robot, joint_names, ee_link, candidate_q)
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
    practical_metrics = {
        **prefixed("prefix", prefix_metrics),
        **prefixed("full_horizon", full_metrics),
    }
    boundary = candidate_q[0] - previous_q
    practical_metrics.update(
        {
            "boundary_joint_l2": float(np.linalg.norm(boundary)),
            "boundary_max_abs_joint_step": float(np.max(np.abs(boundary))),
        }
    )
    repository_score = repository_ranking.score(
        repository_score_context(
            raw_q=candidate_q,
            candidate_index=candidate_index,
            is_diffusion_refined=candidate_type == "diffusion_refined",
            desired_path=desired_window,
            ee_positions=candidate_ee,
            previous_q=previous_q,
            repository_fk=repository_fk,
            robot=robot,
            joint_names=joint_names,
            ee_link=ee_link,
            lower=lower,
            upper=upper,
            weights=weights,
            w_rms_cart=w_rms_cart,
            w_limit_count=w_limit_count,
            continuity_weight=args.continuity_weight,
            args=args,
            metrics=practical_metrics,
        )
    )
    prefix_repository_score = repository_ranking.score(
        repository_score_context(
            raw_q=candidate_q[:execution_count],
            candidate_index=candidate_index,
            is_diffusion_refined=candidate_type == "diffusion_refined",
            desired_path=desired_window[:execution_count],
            ee_positions=candidate_ee[:execution_count],
            previous_q=previous_q,
            repository_fk=repository_fk,
            robot=robot,
            joint_names=joint_names,
            ee_link=ee_link,
            lower=lower,
            upper=upper,
            weights=weights,
            w_rms_cart=w_rms_cart,
            w_limit_count=w_limit_count,
            continuity_weight=args.continuity_weight,
            args=args,
            metrics=prefix_metrics,
        )
    )
    discounted = discounted_score(
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
    oracle_prefix = float(
        np.sqrt(
            np.mean(
                np.square(
                    candidate_q[:execution_count] - expert_window[:execution_count]
                )
            )
        )
    )
    oracle_full = float(
        np.sqrt(np.mean(np.square(candidate_q - expert_window)))
    )
    return {
        "candidate_type": candidate_type,
        **practical_metrics,
        "repository_prefix_ranking_score": prefix_repository_score,
        "repository_ranking_score": repository_score,
        "discounted_score": discounted,
        "oracle_prefix_joint_rmse": oracle_prefix,
        "oracle_full_horizon_joint_rmse": oracle_full,
    }


def selector_cycle_rows(
    path_name: str,
    cycle_index: int,
    start: int,
    rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    buffer = rows[0]
    selectors = {
        "selector_prefix_hard_gate": select_minimum(
            rows, "repository_prefix_ranking_score"
        ),
        "selector_discounted_hard_gate": select_minimum(rows, "discounted_score"),
        "selector_lexicographic": select_lexicographic(rows),
    }
    safe_prefix_best = min(
        (
            index
            for index, row in enumerate(rows)
            if row["candidate_type"] == "unrefined_buffer"
            or int(row["hard_gate_passed"]) == 1
        ),
        key=lambda index: (float(rows[index]["prefix_mean_cartesian_error"]), index),
    )
    output: List[Dict[str, Any]] = []
    for selector, index in selectors.items():
        selected = rows[index]
        output.append(
            {
                "path_name": path_name,
                "planning_cycle_index": cycle_index,
                "trajectory_start_index": start,
                "selector": selector,
                "selected_candidate_index": index,
                "selected_candidate_type": selected["candidate_type"],
                "selected_alpha": selected["alpha"],
                "selected_ramp_length": selected["ramp_length"],
                "selected_base_sample_index": selected["base_sample_index"],
                "selected_diffusion": int(selected["candidate_type"] == "diffusion_refined"),
                "cartesian_improved": int(
                    float(selected["prefix_mean_cartesian_error"])
                    < float(buffer["prefix_mean_cartesian_error"])
                ),
                "drawing_cost_improved": int(
                    float(selected["prefix_drawing_cost"])
                    < float(buffer["prefix_drawing_cost"])
                ),
                "both_improved": int(
                    float(selected["prefix_mean_cartesian_error"])
                    < float(buffer["prefix_mean_cartesian_error"])
                    and float(selected["prefix_drawing_cost"])
                    < float(buffer["prefix_drawing_cost"])
                ),
                "unsafe_selection": int(selected["hard_gate_passed"] == 0),
                "cartesian_change": float(
                    selected["prefix_mean_cartesian_error"]
                    - buffer["prefix_mean_cartesian_error"]
                ),
                "drawing_cost_change": float(
                    selected["prefix_drawing_cost"]
                    - buffer["prefix_drawing_cost"]
                ),
                "boundary_max_abs_joint_step": float(
                    selected["boundary_max_abs_joint_step"]
                ),
                "prefix_max_joint_step": float(selected["prefix_max_joint_step"]),
                "selection_regret_prefix_cartesian": float(
                    selected["prefix_mean_cartesian_error"]
                    - rows[safe_prefix_best]["prefix_mean_cartesian_error"]
                ),
            }
        )
    return output


def aggregate_alpha_ramp(
    details: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    diffusion = [row for row in details if row["candidate_type"] == "diffusion_refined"]
    groups: Dict[Tuple[float, int, str], List[Dict[str, Any]]] = {}
    for row in diffusion:
        key = (float(row["alpha"]), int(row["ramp_length"]), str(row["taper_mode"]))
        groups.setdefault(key, []).append(row)
    output: List[Dict[str, Any]] = []
    for (alpha, ramp, mode), rows in sorted(groups.items()):
        cycles: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}
        for row in rows:
            cycles.setdefault(
                (str(row["path_name"]), int(row["planning_cycle_index"])), []
            ).append(row)
        cycle_groups = list(cycles.values())

        def cycle_rate(predicate: Callable[[Dict[str, Any]], bool]) -> float:
            return float(np.mean([any(predicate(row) for row in group) for group in cycle_groups]))

        cart_predicate = lambda row: float(row["cartesian_improvement_vs_buffer"]) > 0.0
        draw_predicate = lambda row: float(row["drawing_improvement_vs_buffer"]) > 0.0
        both_predicate = lambda row: cart_predicate(row) and draw_predicate(row)
        safe_cart = lambda row: int(row["hard_gate_passed"]) == 1 and cart_predicate(row)
        safe_both = lambda row: int(row["hard_gate_passed"]) == 1 and both_predicate(row)

        selected_rows: List[Dict[str, Any]] = []
        for group in cycle_groups:
            buffer = group[0]["_buffer_row"]
            eligible = [row for row in group if int(row["hard_gate_passed"]) == 1]
            selected_rows.append(
                min(eligible, key=lambda row: float(row["prefix_drawing_cost"]))
                if eligible
                else buffer
            )
        output.append(
            {
                "alpha": alpha,
                "ramp_length": ramp,
                "taper_mode": mode,
                "cycle_count": len(cycle_groups),
                "candidate_count": len(rows),
                "cycles_with_cartesian_improving_candidate": cycle_rate(cart_predicate),
                "cycles_with_drawing_improving_candidate": cycle_rate(draw_predicate),
                "cycles_with_both_improving_candidate": cycle_rate(both_predicate),
                "cycles_with_safe_cartesian_improving_candidate": cycle_rate(safe_cart),
                "cycles_with_safe_both_improving_candidate": cycle_rate(safe_both),
                "mean_boundary_max_abs_joint_step": float(
                    np.mean([float(row["boundary_max_abs_joint_step"]) for row in rows])
                ),
                "mean_prefix_max_joint_step": float(
                    np.mean([float(row["prefix_max_joint_step"]) for row in rows])
                ),
                "mean_cartesian_improvement": float(
                    np.mean([float(row["cartesian_improvement_vs_buffer"]) for row in rows])
                ),
                "mean_drawing_cost_improvement": float(
                    np.mean([float(row["drawing_improvement_vs_buffer"]) for row in rows])
                ),
                "hard_gate_pass_rate": float(
                    np.mean([int(row["hard_gate_passed"]) for row in rows])
                ),
                "oracle_improvement_rate": float(
                    np.mean([int(row["oracle_improved_vs_buffer"]) for row in rows])
                ),
                "selected_diffusion_rate": float(
                    np.mean(
                        [row["candidate_type"] == "diffusion_refined" for row in selected_rows]
                    )
                ),
                "selected_unsafe_count": int(
                    sum(
                        int(row["hard_gate_passed"]) == 0
                        for row in selected_rows
                    )
                ),
                "selected_joint_limit_violation_count": int(
                    sum(
                        float(row["prefix_joint_limit_violation_count"]) > 0.0
                        or float(row["full_horizon_joint_limit_violation_count"]) > 0.0
                        for row in selected_rows
                    )
                ),
                "selected_mean_prefix_cartesian_error": float(
                    np.mean([float(row["prefix_mean_cartesian_error"]) for row in selected_rows])
                ),
                "selected_mean_prefix_drawing_cost": float(
                    np.mean([float(row["prefix_drawing_cost"]) for row in selected_rows])
                ),
                "buffer_mean_prefix_cartesian_error": float(
                    np.mean(
                        [float(group[0]["_buffer_row"]["prefix_mean_cartesian_error"]) for group in cycle_groups]
                    )
                ),
                "buffer_mean_prefix_drawing_cost": float(
                    np.mean(
                        [float(group[0]["_buffer_row"]["prefix_drawing_cost"]) for group in cycle_groups]
                    )
                ),
            }
        )
    return output


def aggregate_selectors(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for selector in sorted({str(row["selector"]) for row in rows}):
        group = [row for row in rows if row["selector"] == selector]
        output.append(
            {
                "selector": selector,
                "cycle_count": len(group),
                "fraction_selecting_diffusion": float(
                    np.mean([int(row["selected_diffusion"]) for row in group])
                ),
                "cartesian_improvement_rate": float(
                    np.mean([int(row["cartesian_improved"]) for row in group])
                ),
                "drawing_cost_improvement_rate": float(
                    np.mean([int(row["drawing_cost_improved"]) for row in group])
                ),
                "both_improved_rate": float(
                    np.mean([int(row["both_improved"]) for row in group])
                ),
                "unsafe_selection_rate": float(
                    np.mean([int(row["unsafe_selection"]) for row in group])
                ),
                "mean_cartesian_change": float(
                    np.mean([float(row["cartesian_change"]) for row in group])
                ),
                "mean_drawing_cost_change": float(
                    np.mean([float(row["drawing_cost_change"]) for row in group])
                ),
                "mean_boundary_step": float(
                    np.mean([float(row["boundary_max_abs_joint_step"]) for row in group])
                ),
                "mean_prefix_maximum_step": float(
                    np.mean([float(row["prefix_max_joint_step"]) for row in group])
                ),
                "mean_selection_regret": float(
                    np.mean(
                        [float(row["selection_regret_prefix_cartesian"]) for row in group]
                    )
                ),
            }
        )
    return output


def strip_private_fields(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {key: value for key, value in row.items() if not key.startswith("_")}
        for row in rows
    ]


def save_plots(
    output_dir: Path,
    details: Sequence[Dict[str, Any]],
    alpha_ramp: Sequence[Dict[str, Any]],
    selectors: Sequence[Dict[str, Any]],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    def line_plot(field: str, filename: str, ylabel: str) -> None:
        figure, axis = plt.subplots(figsize=(7, 5))
        for ramp in sorted({int(row["ramp_length"]) for row in alpha_ramp}):
            group = sorted(
                (row for row in alpha_ramp if int(row["ramp_length"]) == ramp),
                key=lambda row: float(row["alpha"]),
            )
            axis.plot(
                [float(row["alpha"]) for row in group],
                [float(row[field]) for row in group],
                marker="o",
                label=f"ramp={ramp}",
            )
        axis.set_xlabel("Alpha")
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.25)
        axis.legend()
        figure.tight_layout()
        figure.savefig(plot_dir / filename, dpi=160)
        plt.close(figure)

    line_plot(
        "cycles_with_safe_cartesian_improving_candidate",
        "safe_cartesian_improvement_rate_vs_alpha.png",
        "Safe Cartesian-improving cycle rate",
    )
    line_plot(
        "cycles_with_safe_both_improving_candidate",
        "safe_both_improvement_rate_vs_alpha.png",
        "Safe both-improving cycle rate",
    )
    line_plot(
        "mean_boundary_max_abs_joint_step",
        "boundary_step_vs_alpha.png",
        "Mean boundary maximum joint step",
    )
    line_plot(
        "mean_prefix_max_joint_step",
        "prefix_max_step_vs_alpha.png",
        "Mean prefix maximum joint step",
    )

    diffusion = [row for row in details if row["candidate_type"] == "diffusion_refined"]
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.scatter(
        [float(row["scaled_residual_rms"]) for row in diffusion],
        [float(row["cartesian_improvement_vs_buffer"]) for row in diffusion],
        s=8,
        alpha=0.3,
    )
    axis.set_xlabel("Scaled residual RMS")
    axis.set_ylabel("Prefix Cartesian improvement vs buffer")
    axis.grid(True, alpha=0.25)
    figure.tight_layout()
    figure.savefig(plot_dir / "cartesian_improvement_vs_scaled_residual_rms.png", dpi=160)
    plt.close(figure)

    alphas = sorted({float(row["alpha"]) for row in alpha_ramp})
    ramps = sorted({int(row["ramp_length"]) for row in alpha_ramp})

    def heatmap(field: str, filename: str, title: str) -> None:
        matrix = np.full((len(ramps), len(alphas)), np.nan, dtype=np.float64)
        for row in alpha_ramp:
            matrix[ramps.index(int(row["ramp_length"])), alphas.index(float(row["alpha"]))] = float(row[field])
        figure, axis = plt.subplots(figsize=(8, 5))
        image = axis.imshow(matrix, aspect="auto", origin="lower")
        axis.set_xticks(range(len(alphas)), [f"{value:g}" for value in alphas])
        axis.set_yticks(range(len(ramps)), [str(value) for value in ramps])
        axis.set_xlabel("Alpha")
        axis.set_ylabel("Ramp length")
        axis.set_title(title)
        figure.colorbar(image, ax=axis)
        figure.tight_layout()
        figure.savefig(plot_dir / filename, dpi=160)
        plt.close(figure)

    heatmap(
        "cycles_with_safe_cartesian_improving_candidate",
        "safe_improvement_alpha_ramp_heatmap.png",
        "Safe Cartesian-improving cycle rate",
    )
    heatmap(
        "cycles_with_drawing_improving_candidate",
        "drawing_improvement_alpha_ramp_heatmap.png",
        "Drawing-cost-improving cycle rate",
    )

    metrics = (
        "cartesian_improvement_rate",
        "drawing_cost_improvement_rate",
        "both_improved_rate",
        "fraction_selecting_diffusion",
    )
    x = np.arange(len(selectors), dtype=np.float64)
    width = 0.18
    figure, axis = plt.subplots(figsize=(10, 5))
    for metric_index, metric in enumerate(metrics):
        axis.bar(
            x + (metric_index - 1.5) * width,
            [float(row[metric]) for row in selectors],
            width,
            label=metric,
        )
    axis.set_xticks(x, [str(row["selector"]) for row in selectors], rotation=15, ha="right")
    axis.set_ylabel("Fraction of cycles")
    axis.legend(fontsize=8)
    axis.grid(True, axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(plot_dir / "selector_comparison.png", dpi=160)
    plt.close(figure)


def main() -> int:
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)
    device = resolve_device(args.device)

    rollout_module = importlib.import_module("diagnose_warm_start_action_buffer_rollout")
    repository_ranking = StrictRepositoryRanking(rollout_module)
    rollout_defaults = rollout_default_namespace(rollout_module)
    w_rms_cart = float(rollout_defaults.w_rms_cart)
    w_limit_count = float(rollout_defaults.w_limit_count)
    if args.continuity_weight is None:
        args.continuity_weight = float(rollout_defaults.continuity_weight)
    weights = default_weights()
    stats = load_stats(args.stats_npz)
    residual_mean = stats["residual_mean"]
    residual_std = stats["residual_std"]

    residual_physical_zero = np.zeros(
        (args.prediction_horizon, JOINT_DIM), dtype=np.float32
    )
    residual_norm_zero = (
        (residual_physical_zero - residual_mean[None, :])
        / residual_std[None, :]
    ).astype(np.float32)
    denormalized_initialization = (
        residual_norm_zero * residual_std[None, :] + residual_mean[None, :]
    ).astype(np.float32)
    initialization_assertion = bool(
        np.allclose(
            denormalized_initialization,
            residual_physical_zero,
            rtol=1e-6,
            atol=1e-7,
        )
    )
    if not initialization_assertion:
        max_error = float(
            np.max(np.abs(denormalized_initialization - residual_physical_zero))
        )
        raise AssertionError(
            "Physical-zero residual normalization did not round-trip; "
            f"max_error={max_error:.12e}"
        )
    normalization_row = {
        "residual_mean_per_joint": json.dumps(residual_mean.astype(float).tolist()),
        "residual_std_per_joint": json.dumps(residual_std.astype(float).tolist()),
        "normalized_physical_zero_per_joint": json.dumps(
            residual_norm_zero[0].astype(float).tolist()
        ),
        "actual_normalized_initialization_per_joint": json.dumps(
            residual_norm_zero[0].astype(float).tolist()
        ),
        "denormalized_initialization_per_joint": json.dumps(
            denormalized_initialization[0].astype(float).tolist()
        ),
        "physical_zero_round_trip_assertion": int(initialization_assertion),
        "physical_zero_round_trip_max_abs_error": float(
            np.max(np.abs(denormalized_initialization - residual_physical_zero))
        ),
        **array_stats("actual_normalized_initialization", residual_norm_zero),
        **array_stats("denormalized_initialization", denormalized_initialization),
    }

    checkpoint = torch_load_checkpoint(args.checkpoint, device)
    if int(checkpoint["condition_dim"]) != CONDITION_DIM:
        raise ValueError("The best v5b checkpoint must use condition_dim=38")
    if int(checkpoint["target_dim"]) != JOINT_DIM:
        raise ValueError("The best v5b checkpoint must use target_dim=6")
    if int(checkpoint["horizon"]) != args.prediction_horizon:
        raise ValueError(
            f"Checkpoint horizon {checkpoint['horizon']} differs from "
            f"prediction_horizon {args.prediction_horizon}"
        )
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
    expert_q = finite_array(data["expert_q"], "expert_q").astype(np.float32)
    q_start_all = finite_array(data["q_start"], "q_start").astype(np.float32)
    path_names = decode_names(data["path_names"])
    if desired_paths.ndim != 3 or desired_paths.shape[-1] != 3:
        raise ValueError("desired_paths must have shape (N,T,3)")
    if expert_q.shape != (desired_paths.shape[0], desired_paths.shape[1], JOINT_DIM):
        raise ValueError("expert_q must have shape (N,T,6)")
    if q_start_all.shape != (desired_paths.shape[0], JOINT_DIM):
        raise ValueError("q_start must have shape (N,6)")

    robot, joint_names, ee_link = load_fk_context(None, None)
    if len(joint_names) != JOINT_DIM:
        raise ValueError("Exactly six active xMateCR7 joints are required")
    lower, upper = extract_joint_limits(robot, joint_names)
    repository_fk = RepositoryFKAdapter(
        robot,
        joint_names,
        ee_link,
        lower,
        upper,
    )
    max_paths = min(args.max_paths, len(path_names))
    details: List[Dict[str, Any]] = []
    selector_cycles: List[Dict[str, Any]] = []

    print(f"Repository ranking function: {repository_ranking.function_name}")
    print(f"Repository ranking signature: {repository_ranking.signature}")
    print(f"Repository ranking weights: {json.dumps(weights_record(weights), sort_keys=True)}")
    print(
        "Repository ranking scalar weights: "
        f"w_rms_cart={w_rms_cart:g}, "
        f"w_limit_count={w_limit_count:g}, "
        f"continuity_weight={args.continuity_weight:g}"
    )
    print("Residual initialization: normalized representation of physical zero (round-trip verified)")

    with torch.no_grad():
        for path_index in range(max_paths):
            path_name = path_names[path_index]
            desired_path = desired_paths[path_index]
            desired_delta = desired_differences(desired_path)
            trajectory_length = desired_path.shape[0]
            prior_q = read_predicted_q_csv(
                args.prior_dir / safe_path_name(path_name) / "predicted_q.csv",
                expected_steps=trajectory_length,
            )
            for cycle_index, start in enumerate(
                range(0, trajectory_length, args.execution_horizon)
            ):
                buffer_q, indices = build_teacher_forced_buffer(
                    prior_q, start, args.prediction_horizon
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
                condition_norm = (
                    (condition - stats["condition_mean"][None, :])
                    / stats["condition_std"][None, :]
                ).astype(np.float32)
                finite_array(condition_norm, "normalized action-buffer condition")
                desired_window = desired_path[indices]
                expert_window = expert_q[path_index, indices]
                execution_count = min(
                    args.execution_horizon, trajectory_length - start
                )

                buffer_metrics = evaluate_candidate(
                    candidate_q=buffer_q,
                    candidate_index=0,
                    candidate_type="unrefined_buffer",
                    desired_window=desired_window,
                    expert_window=expert_window,
                    execution_count=execution_count,
                    previous_q=previous_q,
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
                buffer_row: Dict[str, Any] = {
                    "path_name": path_name,
                    "planning_cycle_index": cycle_index,
                    "trajectory_start_index": start,
                    "execution_count": execution_count,
                    "candidate_index": 0,
                    "base_sample_index": -1,
                    "alpha": 0.0,
                    "ramp_length": 0,
                    "taper_mode": "none",
                    "anchor_first": int(args.anchor_first),
                    "anchor_executed_prefix_start": int(
                        args.anchor_executed_prefix_start
                    ),
                    **buffer_metrics,
                    "physical_residual_rms_before_scaling": 0.0,
                    "scaled_residual_rms": 0.0,
                    "first_step_residual": json.dumps([0.0] * JOINT_DIM),
                    "first_step_residual_l2": 0.0,
                    "first_step_residual_max_abs": 0.0,
                    "cartesian_improvement_vs_buffer": 0.0,
                    "drawing_improvement_vs_buffer": 0.0,
                    "oracle_improved_vs_buffer": 0,
                }
                buffer_passed, buffer_reason = safety_gate(
                    buffer_row, buffer_row, args
                )
                buffer_row["hard_gate_passed"] = int(buffer_passed)
                buffer_row["hard_gate_rejection_reason"] = buffer_reason
                cycle_rows: List[Dict[str, Any]] = [buffer_row]

                cycle_seed = args.seed + path_index * 100_000 + cycle_index * 1_000
                set_seed(cycle_seed)
                generated_norm = reverse_noised_x0_batches(
                    model=model,
                    call_variant=call_variant,
                    condition_bhc=np.repeat(
                        condition_norm[None, :, :],
                        args.num_base_samples,
                        axis=0,
                    ),
                    x0_norm_bhc=np.repeat(
                        residual_norm_zero[None, :, :],
                        args.num_base_samples,
                        axis=0,
                    ),
                    t_init=args.t_init,
                    schedule=schedule,
                    batch_size=args.num_base_samples,
                    device=device,
                    deterministic=False,
                )
                finite_array(generated_norm, "generated normalized residual")
                generated_physical = (
                    generated_norm * residual_std[None, None, :]
                    + residual_mean[None, None, :]
                ).astype(np.float32)
                finite_array(generated_physical, "generated physical residual")

                candidate_index = 1
                for base_sample_index in range(args.num_base_samples):
                    base_residual = generated_physical[base_sample_index]
                    base_rms = float(np.sqrt(np.mean(np.square(base_residual))))
                    for alpha in args.alphas:
                        for ramp_length in args.ramp_lengths:
                            taper = taper_values(
                                args.prediction_horizon,
                                int(ramp_length),
                                args.taper_mode,
                            )
                            scaled_residual = apply_scaled_taper(
                                base_residual,
                                float(alpha),
                                taper,
                                args.anchor_first,
                                args.anchor_executed_prefix_start,
                                args.execution_horizon,
                            )
                            candidate_q = (buffer_q + scaled_residual).astype(np.float32)
                            if int(ramp_length) > 0 and not np.array_equal(
                                candidate_q[0], buffer_q[0]
                            ):
                                raise AssertionError(
                                    "A nonzero taper ramp changed the first buffer configuration"
                                )
                            if args.anchor_first and not np.array_equal(
                                candidate_q[0], buffer_q[0]
                            ):
                                raise AssertionError(
                                    "--anchor_first did not preserve the first buffer configuration"
                                )
                            if args.anchor_executed_prefix_start:
                                anchor_indices = np.arange(
                                    0,
                                    args.prediction_horizon,
                                    args.execution_horizon,
                                )
                                if not np.array_equal(
                                    candidate_q[anchor_indices],
                                    buffer_q[anchor_indices],
                                ):
                                    raise AssertionError(
                                        "--anchor_executed_prefix_start did not preserve "
                                        "all execution-block boundary configurations"
                                    )
                            metrics = evaluate_candidate(
                                candidate_q=candidate_q,
                                candidate_index=candidate_index,
                                candidate_type="diffusion_refined",
                                desired_window=desired_window,
                                expert_window=expert_window,
                                execution_count=execution_count,
                                previous_q=previous_q,
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
                                "path_name": path_name,
                                "planning_cycle_index": cycle_index,
                                "trajectory_start_index": start,
                                "execution_count": execution_count,
                                "candidate_index": candidate_index,
                                "base_sample_index": base_sample_index,
                                "candidate_seed": cycle_seed,
                                "alpha": float(alpha),
                                "ramp_length": int(ramp_length),
                                "taper_mode": args.taper_mode,
                                "anchor_first": int(args.anchor_first),
                                "anchor_executed_prefix_start": int(
                                    args.anchor_executed_prefix_start
                                ),
                                **metrics,
                                "physical_residual_rms_before_scaling": base_rms,
                                "scaled_residual_rms": float(
                                    np.sqrt(np.mean(np.square(scaled_residual)))
                                ),
                                "first_step_residual": json.dumps(
                                    scaled_residual[0].astype(float).tolist()
                                ),
                                "first_step_residual_l2": float(
                                    np.linalg.norm(scaled_residual[0])
                                ),
                                "first_step_residual_max_abs": float(
                                    np.max(np.abs(scaled_residual[0]))
                                ),
                                "cartesian_improvement_vs_buffer": float(
                                    buffer_row["prefix_mean_cartesian_error"]
                                    - metrics["prefix_mean_cartesian_error"]
                                ),
                                "drawing_improvement_vs_buffer": float(
                                    buffer_row["prefix_drawing_cost"]
                                    - metrics["prefix_drawing_cost"]
                                ),
                                "oracle_improved_vs_buffer": int(
                                    metrics["oracle_prefix_joint_rmse"]
                                    < buffer_row["oracle_prefix_joint_rmse"]
                                ),
                                "_buffer_row": buffer_row,
                            }
                            passed, reason = safety_gate(row, buffer_row, args)
                            row["hard_gate_passed"] = int(passed)
                            row["hard_gate_rejection_reason"] = reason
                            cycle_rows.append(row)
                            candidate_index += 1

                if cycle_rows[0] is not buffer_row or cycle_rows[0]["candidate_index"] != 0:
                    raise AssertionError("Candidate 0 is not the original unrefined buffer")
                selector_cycles.extend(
                    selector_cycle_rows(
                        path_name,
                        cycle_index,
                        start,
                        cycle_rows,
                    )
                )
                details.extend(cycle_rows)
            print(f"Audited {path_index + 1}/{max_paths} paths: {path_name}")

    alpha_ramp = aggregate_alpha_ramp(details)
    selector_comparison = aggregate_selectors(selector_cycles)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_dict_csv(
        args.output_dir / "scaled_candidate_details.csv",
        strip_private_fields(details),
    )
    write_dict_csv(args.output_dir / "alpha_ramp_aggregate.csv", alpha_ramp)
    write_dict_csv(args.output_dir / "selector_comparison.csv", selector_comparison)
    write_dict_csv(args.output_dir / "normalization_audit.csv", [normalization_row])
    save_plots(args.output_dir, details, alpha_ramp, selector_comparison)

    valid = [
        row
        for row in alpha_ramp
        if int(row["selected_unsafe_count"]) == 0
        and int(row["selected_joint_limit_violation_count"]) == 0
        and float(row["selected_mean_prefix_cartesian_error"])
        < float(row["buffer_mean_prefix_cartesian_error"])
        and float(row["selected_mean_prefix_drawing_cost"])
        <= float(row["buffer_mean_prefix_drawing_cost"]) + 1e-12
    ]
    print(f"Saved candidate details: {args.output_dir / 'scaled_candidate_details.csv'}")
    print(f"Saved alpha/ramp aggregate: {args.output_dir / 'alpha_ramp_aggregate.csv'}")
    print(f"Saved selector comparison: {args.output_dir / 'selector_comparison.csv'}")
    print(f"Saved normalization audit: {args.output_dir / 'normalization_audit.csv'}")
    print(f"Saved plots: {args.output_dir / 'plots'}")
    if valid:
        best = min(
            valid,
            key=lambda row: (
                float(row["selected_mean_prefix_cartesian_error"]),
                float(row["selected_mean_prefix_drawing_cost"]),
            ),
        )
        print(
            "Best safe useful combination: "
            f"alpha={float(best['alpha']):g}, "
            f"ramp_length={int(best['ramp_length'])}, "
            f"taper_mode={best['taper_mode']}, "
            f"prefix_cartesian={float(best['selected_mean_prefix_cartesian_error']):.8e}, "
            f"prefix_drawing={float(best['selected_mean_prefix_drawing_cost']):.8e}."
        )
    else:
        print(
            "No alpha/ramp combination met every safety and quality requirement; "
            "the existing v5b diffusion residuals cannot be made practically useful "
            "through scaling and tapering alone."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
