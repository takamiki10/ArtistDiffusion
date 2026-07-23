#!/usr/bin/env python3
"""Generate diverse, scale-conditioned v8 residual targets.

The v7 target generator remains the scientific authority. This script adds
deterministic restarts, residual-space diversity retention, independent scale
reevaluation, and spawn-based CPU multiprocessing without changing v7 FK,
safety, acceptance, or robot-aware scoring formulas.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import multiprocessing
import os
import pickle
import random
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

import generate_diffusion_v7_cost_improving_residual_targets as v7
import build_diffusion_v7_cost_improving_training_dataset as v7_dataset_builder
from generate_ik_seed_path import (
    DEFAULT_EE_LINK,
    DEFAULT_JOINT_LIMIT_SAFETY_MARGIN_RAD,
    DEFAULT_JOINT_NAMES,
    DEFAULT_URDF_PATH,
    HARD_JOINT_LIMIT_TOLERANCE_RAD,
)


HORIZON = 32
JOINT_DIM = 6
DEFAULT_EXECUTION_HORIZON = 8
DEFAULT_SCALES = (0.125, 0.25, 0.50, 0.75, 1.00)
DEFAULT_OUTPUT_DIR = Path(
    "data/cartesian_expert_dataset_v3/"
    "diffusion_v8_multitarget_scaled_residual_targets"
)
DEFAULT_TRAIN_PRIOR = Path(
    "data/cartesian_expert_dataset_v3/"
    "adaptive_mlp_ik_bootstrap_prior/train_prior.npz"
)
DEFAULT_TRAIN_WINDOWS = Path(
    "data/cartesian_expert_dataset_v3/"
    "diffusion_v6_strong_prior_residual_windows/train_windows.npz"
)
DEFAULT_SPLIT_MANIFEST = Path(
    "data/cartesian_expert_dataset_v3/"
    "diffusion_v6_strong_prior_residual_windows/split_manifest.csv"
)
DEFAULT_PRIMARY_METHODS = ("jacobian_dls", "sequential_ik")
DEFAULT_FALLBACK_METHODS = ("smooth_perturbation", "spline_cem")
ADAPTIVE_STATE_JSON = "adaptive_generation_state.json"
# Candidate counts produced by one v7 method restart with the default method grids.
V7_CANDIDATES_PER_METHOD_RESTART = {
    "jacobian_dls": 48,
    "sequential_ik": 36,
    "smooth_perturbation": 32,
    "spline_cem": 8,
}
V7_METHOD_ORDER: Dict[str, int] = {
    str(method): index for index, method in enumerate(v7.DEFAULT_METHODS)
}
OUTPUT_FILENAMES = (
    "selected_targets.npz",
    "candidate_results.csv",
    "selected_target_summary.csv",
    "per_window_summary.csv",
    "per_path_summary.csv",
    "scale_summary.csv",
    "candidate_method_summary.csv",
    "candidate_method_window_contribution.csv",
    "adaptive_stage_summary.csv",
    "rejection_reason_summary.csv",
    "diversity_summary.csv",
    "target_generation_summary.json",
)


@dataclass(frozen=True)
class SourceWindow:
    canonical_index: int
    path_name: str
    window_start: int
    condition: np.ndarray
    prior_q: np.ndarray
    desired: np.ndarray
    execution_horizon: int
    context: v7.WindowContext


@dataclass(frozen=True)
class BaseGenerationTask:
    work_id: str
    window: SourceWindow
    restarts_per_method: int
    candidate_seed: int
    v7_arguments: Dict[str, Any]
    generation_stage: str


@dataclass(frozen=True)
class BaseGenerationResult:
    work_id: str
    window: SourceWindow
    prior_metrics: Dict[str, Any]
    candidates: Tuple[Dict[str, Any], ...]
    generation_time_s: float
    scoring_time_s: float
    generation_stage: str


@dataclass(frozen=True)
class ScaleEvaluationTask:
    work_id: str
    window: SourceWindow
    prior_metrics: Dict[str, Any]
    base_targets: Tuple[Dict[str, Any], ...]
    scales: Tuple[float, ...]
    v7_arguments: Dict[str, Any]
    generation_stage: str


@dataclass(frozen=True)
class ScaleEvaluationResult:
    work_id: str
    rows: Tuple[Dict[str, Any], ...]
    scoring_time_s: float
    generation_stage: str


_WORKER_ROBOT: Optional[v7.RobotContext] = None


def decode_strings(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            value.decode("utf-8", errors="strict")
            if isinstance(value, bytes)
            else str(value)
            for value in np.asarray(values).reshape(-1)
        ],
        dtype=str,
    )


def stable_seed(*parts: Any) -> int:
    payload = json.dumps(
        list(parts), ensure_ascii=False, separators=(",", ":"), sort_keys=False
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & 0x7FFFFFFF


def stable_identifier(prefix: str, *parts: Any) -> str:
    payload = json.dumps(
        list(parts), ensure_ascii=False, separators=(",", ":"), sort_keys=False
    ).encode("utf-8")
    return f"{prefix}_{hashlib.sha256(payload).hexdigest()[:20]}"


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.npz")
    np.savez_compressed(temporary, **arrays)
    os.replace(temporary, path)


def atomic_pickle(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        pickle.dump(dict(payload), handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_pickle(path: Path) -> Dict[str, Any]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Adaptive state at {path} is not a dictionary")
    return payload


def v7_generation_defaults() -> Dict[str, Any]:
    """Mirror v7 parser defaults needed by its candidate-generation APIs."""
    return {
        "path_names": None,
        "num_paths": 20,
        "path_selection": "stratified_prior_error",
        "horizon": 32,
        "execution_horizon": 8,
        "targets_per_window": 8,
        "seed": 42,
        "device": "cpu",
        "overwrite": False,
        "resume": False,
        "max_windows": None,
        "candidate_methods": list(v7.DEFAULT_METHODS),
        "save_all_candidates": False,
        "robot_urdf": Path(DEFAULT_URDF_PATH),
        "ee_link": DEFAULT_EE_LINK,
        "minimum_residual_distance": 0.005,
        "min_cartesian_improvement_m": 1.0e-5,
        "min_cartesian_improvement_fraction": 0.005,
        "smoothness_relative_tolerance": 0.10,
        "boundary_absolute_tolerance": 0.01,
        "max_joint_step_gate": 0.20,
        "joint_limit_safety_margin": DEFAULT_JOINT_LIMIT_SAFETY_MARGIN_RAD,
        "dls_damping": (1e-4, 1e-3, 1e-2, 1e-1),
        "dls_scales": (0.25, 0.50, 1.00),
        "ik_damping": (1e-3, 1e-2, 1e-1),
        "ik_iteration_limits": (8, 16),
        "cem_control_points": 6,
        "cem_candidates": 128,
        "cem_elites": 16,
        "cem_iterations": 20,
        "cem_restarts": 2,
        "cem_initial_std": 0.03,
        "cem_max_residual": 0.15,
        "smooth_amplitudes": (0.005, 0.01, 0.025, 0.05),
        "w_cart_mean": 4.0,
        "w_cart_p95": 2.0,
        "w_cart_max": 1.0,
        "w_acceleration": 0.5,
        "w_jerk": 0.25,
        "w_boundary_step": 1.0,
        "w_boundary_acceleration": 0.5,
        "w_singularity": 0.25,
        "floor_cartesian_m": 1e-4,
        "floor_derivative": 1e-8,
        "floor_boundary_rad": 1e-4,
        "floor_singularity": 1e-4,
    }


def parse_args() -> argparse.Namespace:
    defaults = v7_generation_defaults()
    parser = argparse.ArgumentParser(
        description="Generate diverse scale-conditioned v8 residual targets."
    )
    parser.add_argument("--train_prior", type=Path, default=DEFAULT_TRAIN_PRIOR)
    parser.add_argument("--train_windows", type=Path, default=DEFAULT_TRAIN_WINDOWS)
    parser.add_argument("--split_manifest", type=Path, default=DEFAULT_SPLIT_MANIFEST)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--robot_urdf",
        type=Path,
        default=Path(DEFAULT_URDF_PATH),
    )
    parser.add_argument("--horizon", type=int, default=HORIZON)
    parser.add_argument(
        "--execution_horizon", type=int, default=DEFAULT_EXECUTION_HORIZON
    )
    parser.add_argument("--restarts_per_method", type=int, default=4)
    parser.add_argument("--candidate_seed", type=int, default=42)
    parser.add_argument("--max_base_targets_per_window", type=int, default=4)
    parser.add_argument("--min_prefix_diversity_rms", type=float, default=0.10)
    parser.add_argument("--min_full_diversity_rms", type=float, default=0.05)
    parser.add_argument(
        "--scales", nargs="+", type=float, default=list(DEFAULT_SCALES)
    )
    parser.add_argument("--max_targets_per_window", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--max_paths", type=int, default=None)
    parser.add_argument("--max_windows", type=int, default=None)
    parser.add_argument("--path_names", nargs="+", default=None)
    parser.add_argument(
        "--generation_policy",
        choices=("exhaustive", "adaptive"),
        default="exhaustive",
    )
    parser.add_argument(
        "--primary_methods",
        nargs="+",
        choices=tuple(v7.DEFAULT_METHODS),
        default=list(DEFAULT_PRIMARY_METHODS),
    )
    parser.add_argument(
        "--fallback_methods",
        nargs="+",
        choices=tuple(v7.DEFAULT_METHODS),
        default=list(DEFAULT_FALLBACK_METHODS),
    )
    parser.add_argument("--primary_restarts_per_method", type=int, default=2)
    parser.add_argument("--fallback_restarts_per_method", type=int, default=4)
    parser.add_argument("--minimum_base_targets_before_stop", type=int, default=4)
    parser.add_argument("--minimum_final_targets_before_stop", type=int, default=8)
    parser.add_argument("--fallback_trigger_final_target_count", type=int, default=4)
    parser.add_argument(
        "--enable_early_stop",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--candidate_results_mode",
        choices=("all", "retained_and_summary", "none"),
        default="all",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    args.v7_default_arguments = defaults
    return args


def validate_args(args: argparse.Namespace) -> None:
    for label, path in (
        ("train_prior", args.train_prior),
        ("train_windows", args.train_windows),
        ("split_manifest", args.split_manifest),
    ):
        resolved = v7.resolve_project_path(path)
        if any(token in str(resolved).lower() for token in v7.FORBIDDEN_INPUT_TOKENS):
            raise ValueError(f"Forbidden validation/test {label}: {resolved}")
    if args.train_prior.name != "train_prior.npz":
        raise ValueError("--train_prior must name train_prior.npz")
    if args.train_windows.name != "train_windows.npz":
        raise ValueError("--train_windows must name the v6 train_windows.npz")
    if args.split_manifest.name != "split_manifest.csv":
        raise ValueError("--split_manifest must name split_manifest.csv")
    if args.horizon != HORIZON:
        raise ValueError(f"v8 currently requires --horizon {HORIZON}")
    if args.execution_horizon != DEFAULT_EXECUTION_HORIZON:
        raise ValueError(
            f"v8 currently requires --execution_horizon {DEFAULT_EXECUTION_HORIZON}"
        )
    integer_fields = (
        "restarts_per_method",
        "primary_restarts_per_method",
        "fallback_restarts_per_method",
        "minimum_base_targets_before_stop",
        "minimum_final_targets_before_stop",
        "fallback_trigger_final_target_count",
        "max_base_targets_per_window",
        "max_targets_per_window",
        "num_workers",
    )
    for name in integer_fields:
        if int(getattr(args, name)) < 1:
            raise ValueError(f"--{name} must be at least 1")
    for name in ("max_paths", "max_windows"):
        value = getattr(args, name)
        if value is not None and value < 1:
            raise ValueError(f"--{name} must be positive")
    if args.min_prefix_diversity_rms < 0.0:
        raise ValueError("--min_prefix_diversity_rms must be non-negative")
    if args.min_full_diversity_rms < 0.0:
        raise ValueError("--min_full_diversity_rms must be non-negative")
    if not args.scales or any(
        not np.isfinite(scale) or scale <= 0.0 for scale in args.scales
    ):
        raise ValueError("--scales must contain positive finite values")
    if len(set(float(scale) for scale in args.scales)) != len(args.scales):
        raise ValueError("--scales cannot contain duplicates")
    for option_name in ("primary_methods", "fallback_methods"):
        methods = [str(method) for method in getattr(args, option_name)]
        if not methods:
            raise ValueError(f"--{option_name} cannot be empty")
        if len(methods) != len(set(methods)):
            raise ValueError(f"--{option_name} cannot contain duplicates")
    overlap = set(args.primary_methods) & set(args.fallback_methods)
    if overlap:
        raise ValueError(
            "Primary and fallback methods must be disjoint; overlap="
            f"{sorted(overlap)}"
        )
    unsupported_primary = set(args.primary_methods) - {
        "jacobian_dls",
        "sequential_ik",
    }
    if unsupported_primary:
        raise ValueError(
            "Adaptive primary stages support jacobian_dls followed by "
            f"sequential_ik; unsupported={sorted(unsupported_primary)}"
        )
    if args.primary_methods[0] != "jacobian_dls":
        raise ValueError("--primary_methods must start with jacobian_dls")
    if len(args.primary_methods) > 2:
        raise ValueError("--primary_methods accepts at most two staged methods")
    if args.resume and args.generation_policy != "adaptive":
        raise ValueError("--resume is supported only with --generation_policy adaptive")
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")


def make_robot_context(robot_urdf: Path) -> v7.RobotContext:
    return v7.make_robot_context(
        argparse.Namespace(
            robot_urdf=robot_urdf,
            ee_link=DEFAULT_EE_LINK,
        )
    )


def initialize_worker(robot_urdf: str) -> None:
    global _WORKER_ROBOT
    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[variable] = "1"
    torch.set_num_threads(1)
    _WORKER_ROBOT = make_robot_context(Path(robot_urdf))


def assert_cpu_payload(value: Any, location: str = "payload") -> None:
    if torch.is_tensor(value):
        raise AssertionError(
            f"{location} contains a Torch tensor; worker payloads must be CPU NumPy data"
        )
    if isinstance(value, Mapping):
        for key, item in value.items():
            assert_cpu_payload(item, f"{location}.{key}")
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            assert_cpu_payload(item, f"{location}[{index}]")
    elif hasattr(value, "__dataclass_fields__"):
        for key, item in vars(value).items():
            assert_cpu_payload(item, f"{location}.{key}")


def load_source_windows(args: argparse.Namespace) -> List[SourceWindow]:
    _manifest, train_names = v7.load_train_manifest(
        v7.resolve_project_path(args.split_manifest)
    )
    data = v7.load_window_data(v7.resolve_project_path(args.train_windows))
    prior_names = v7.load_prior_path_names(v7.resolve_project_path(args.train_prior))
    timelines = v7.reconstruct_timelines(data)
    available_paths = sorted(set(timelines) & train_names)
    if len(available_paths) != v7.EXPECTED_TRAIN_PATHS:
        raise ValueError(
            f"Expected {v7.EXPECTED_TRAIN_PATHS} authoritative training paths, "
            f"found {len(available_paths)}"
        )
    if args.path_names:
        requested = [str(name) for name in args.path_names]
        if len(requested) != len(set(requested)):
            raise ValueError("--path_names contains duplicates")
        missing = sorted(set(requested) - set(available_paths))
        if missing:
            raise ValueError(f"Requested paths are not training paths: {missing}")
        selected_paths = sorted(requested)
    else:
        selected_paths = list(available_paths)
    if args.max_paths is not None:
        selected_paths = selected_paths[: args.max_paths]
    if not selected_paths:
        raise ValueError("No training paths remain after filtering")
    if prior_names is not None and not set(selected_paths) <= prior_names:
        missing = sorted(set(selected_paths) - prior_names)
        raise ValueError(f"Frozen train prior is missing paths: {missing}")

    # Reuse the exact v7 context and v7-dataset condition reconstruction.
    contexts = v7.make_window_contexts(data, timelines, selected_paths)
    selected_mask = np.isin(data["path_names"], np.asarray(selected_paths))
    selected_data = {
        key: np.asarray(value)[selected_mask]
        for key, value in data.items()
    }
    window_keys, groups = v7_dataset_builder.group_windows(selected_data)
    conditions = v7_dataset_builder.build_v6_conditions(
        selected_data, window_keys, groups
    )
    contexts = sorted(contexts, key=lambda item: (item.path_name, item.window_start))
    if args.max_windows is not None:
        contexts = contexts[: args.max_windows]
    if not contexts:
        raise ValueError("No training windows remain after filtering")

    result: List[SourceWindow] = []
    for canonical_index, context in enumerate(contexts):
        key = (context.path_name, context.window_start)
        condition = np.asarray(conditions[key], dtype=np.float64)
        if condition.shape != (args.horizon, len(v7_dataset_builder.CONDITION_FEATURE_NAMES)):
            raise ValueError(f"Condition for {key} has unexpected shape {condition.shape}")
        result.append(
            SourceWindow(
                canonical_index=canonical_index,
                path_name=context.path_name,
                window_start=context.window_start,
                condition=condition,
                prior_q=np.asarray(context.prior_q, dtype=np.float64),
                desired=np.asarray(context.desired, dtype=np.float64),
                execution_horizon=args.execution_horizon,
                context=context,
            )
        )
    return result


def worker_arguments(args: argparse.Namespace) -> Dict[str, Any]:
    values = dict(args.v7_default_arguments)
    values.update(
        {
            "horizon": args.horizon,
            "execution_horizon": args.execution_horizon,
            "seed": args.candidate_seed,
            "max_joint_step_gate": float(
                values.get("max_joint_step_gate", 0.20)
            ),
            "joint_limit_safety_margin": float(
                values.get(
                    "joint_limit_safety_margin",
                    DEFAULT_JOINT_LIMIT_SAFETY_MARGIN_RAD,
                )
            ),
            "min_cartesian_improvement_m": float(
                values.get("min_cartesian_improvement_m", 1.0e-5)
            ),
            "min_cartesian_improvement_fraction": float(
                values.get("min_cartesian_improvement_fraction", 0.005)
            ),
            "smoothness_relative_tolerance": float(
                values.get("smoothness_relative_tolerance", 0.10)
            ),
            "boundary_absolute_tolerance": float(
                values.get("boundary_absolute_tolerance", 0.01)
            ),
        }
    )
    return values


def hard_safe_from_v7_row(row: Mapping[str, Any]) -> bool:
    return bool(int(row.get("hard_safe", 0)))


def evaluate_v7_candidate(
    robot: v7.RobotContext,
    context: v7.WindowContext,
    prior_metrics: Mapping[str, Any],
    candidate: v7.Candidate,
    candidate_index: int,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    evaluation_started = time.perf_counter()
    candidate_q = context.prior_q + np.asarray(candidate.residual, dtype=np.float64)
    try:
        metrics = v7.trajectory_metrics(
            robot,
            context,
            candidate_q,
            args.execution_horizon,
            args.joint_limit_safety_margin,
        )
    except Exception as error:
        metrics = {"finite": False}
        candidate.metadata["evaluation_error"] = f"{type(error).__name__}: {error}"
    reasons, improvement, relative = v7.acceptance_reasons(
        metrics, prior_metrics, args
    )
    if "generation_error" in candidate.metadata:
        reasons.append("candidate_generation_failure")
    if "evaluation_error" in candidate.metadata:
        reasons.append("candidate_evaluation_failure")
    if int(candidate.metadata.get("unresolved_timesteps", 0)) > 0:
        reasons.append("unresolved_timesteps")
    if int(candidate.metadata.get("branch_rejections", 0)) > 0:
        reasons.append("branch_change_rejected_during_generation")
    reasons = sorted(set(reasons))
    weights = v7.ScoreWeights(
        args.w_cart_mean,
        args.w_cart_p95,
        args.w_cart_max,
        args.w_acceleration,
        args.w_jerk,
        args.w_boundary_step,
        args.w_boundary_acceleration,
        args.w_singularity,
    )
    floors = v7.MetricFloors(
        args.floor_cartesian_m,
        args.floor_derivative,
        args.floor_boundary_rad,
        args.floor_singularity,
    )
    score = (
        v7.delta_score(metrics, prior_metrics, weights, floors)
        if bool(metrics.get("finite", False)) else math.inf
    )
    row = v7.candidate_result_row(
        candidate,
        candidate_index,
        context,
        prior_metrics,
        metrics,
        reasons,
        improvement,
        relative,
        score,
    )
    valid = (
        hard_safe_from_v7_row(row)
        and improvement > 0.0
        and score < 0.0
        and not reasons
    )
    row.update(
        {
            "candidate_q": candidate_q,
            "residual": np.asarray(candidate.residual, dtype=np.float64),
            "metrics": metrics,
            "hard_safe": int(hard_safe_from_v7_row(row)),
            "cartesian_improving": int(improvement > 0.0),
            "negative_delta_score": int(score < 0.0),
            "valid_target": int(valid),
            "rejection_reasons": "|".join(reasons),
            "v8_fk_scoring_time_s": time.perf_counter() - evaluation_started,
        }
    )
    return row


def generate_base_candidates(
    task: BaseGenerationTask, robot: v7.RobotContext
) -> BaseGenerationResult:
    assert_cpu_payload(task)
    args = argparse.Namespace(**task.v7_arguments)
    prior_ee = v7.fk_trajectory(robot, task.window.prior_q)
    context = replace(task.window.context, prior_ee=prior_ee)
    window = replace(task.window, context=context)
    prior_metrics = v7.trajectory_metrics(
        robot,
        context,
        context.prior_q,
        task.window.execution_horizon,
        args.joint_limit_safety_margin,
    )
    if (
        not bool(prior_metrics.get("finite", False))
        or int(prior_metrics.get("hard_joint_limit_violation_count", 1)) != 0
    ):
        raise RuntimeError(
            f"Frozen prior is hard-invalid for {window.path_name}@{window.window_start}"
        )

    generated: List[Tuple[int, int, v7.Candidate]] = []
    generation_time = 0.0
    for restart_index in range(task.restarts_per_method):
        configured_methods = list(args.candidate_methods)
        method_batches = (
            [configured_methods]
            if task.generation_stage == "exhaustive"
            else [[method] for method in configured_methods]
        )
        for method_batch in method_batches:
            method_seed_part = (
                "exhaustive"
                if task.generation_stage == "exhaustive"
                else method_batch[0]
            )
            restart_seed_parts: Tuple[Any, ...] = (
                task.candidate_seed,
                window.path_name,
                window.window_start,
                "v7_candidate_generation",
            )
            if task.generation_stage != "exhaustive":
                restart_seed_parts += (method_seed_part,)
            restart_seed = stable_seed(*restart_seed_parts, restart_index)
            args.candidate_methods = list(method_batch)
            args.seed = restart_seed
            random.seed(restart_seed)
            np.random.seed(restart_seed)
            torch.manual_seed(restart_seed)
            started = time.perf_counter()
            restart_candidates = v7.generate_candidates(robot, context, args)
            generation_time += time.perf_counter() - started
            for local_index, original in enumerate(restart_candidates):
                canonical_local_index = local_index
                if task.generation_stage != "exhaustive":
                    original_method_order = V7_METHOD_ORDER[str(original.method)]
                    canonical_local_index += sum(
                        V7_CANDIDATES_PER_METHOD_RESTART.get(method, 0)
                        for method, method_order in V7_METHOD_ORDER.items()
                        if method_order < original_method_order
                    )
                method_key = f"{original.method}:{original.subtype}"
                candidate_seed = stable_seed(
                    task.candidate_seed,
                    window.path_name,
                    window.window_start,
                    method_key,
                    restart_index,
                )
                metadata = dict(original.metadata)
                metadata.update(
                    {
                        "restart_index": restart_index,
                        "v8_candidate_seed": candidate_seed,
                        "v7_restart_seed": restart_seed,
                        "generation_stage": task.generation_stage,
                    }
                )
                candidate = v7.Candidate(
                    method=original.method,
                    subtype=original.subtype,
                    residual=np.asarray(original.residual, dtype=np.float64),
                    deterministic_seed=candidate_seed,
                    metadata=metadata,
                    runtime_seconds=float(original.runtime_seconds),
                )
                generated.append((restart_index, canonical_local_index, candidate))

    generated.sort(
        key=lambda item: (
            item[2].method,
            item[2].subtype,
            item[0],
            item[1],
            item[2].deterministic_seed,
        )
    )
    scoring_started = time.perf_counter()
    rows: List[Dict[str, Any]] = []
    for candidate_index, (restart_index, local_index, candidate) in enumerate(generated):
        row = evaluate_v7_candidate(
            robot, context, prior_metrics, candidate, candidate_index, args
        )
        row.update(
            {
                "stage": "base",
                "generation_stage": task.generation_stage,
                "work_id": task.work_id,
                "candidate_id": stable_identifier(
                    "candidate",
                    window.path_name,
                    window.window_start,
                    candidate.method,
                    candidate.subtype,
                    restart_index,
                    local_index,
                ),
                "restart_index": restart_index,
                "local_candidate_index": local_index,
                "target_scale": 1.0,
                "retained": 0,
                "base_retention_rejection_reason": "",
                "v8_candidate_generation_time_s": float(candidate.runtime_seconds),
            }
        )
        rows.append(row)
    scoring_time = time.perf_counter() - scoring_started
    return BaseGenerationResult(
        work_id=task.work_id,
        window=window,
        prior_metrics=dict(prior_metrics),
        candidates=tuple(rows),
        generation_time_s=generation_time,
        scoring_time_s=scoring_time,
        generation_stage=task.generation_stage,
    )


def base_worker_entry(task: BaseGenerationTask) -> BaseGenerationResult:
    if _WORKER_ROBOT is None:
        raise RuntimeError("Worker robot is not initialized")
    return generate_base_candidates(task, _WORKER_ROBOT)


def robust_joint_residual_std(results: Sequence[BaseGenerationResult]) -> np.ndarray:
    residuals = [
        np.asarray(row["residual"], dtype=np.float64)
        for result in results
        for row in result.candidates
        if int(row["valid_target"]) == 1
    ]
    if not residuals:
        raise RuntimeError("No valid base candidates exist; diversity scale is undefined")
    values = np.concatenate(residuals, axis=0)
    median = np.median(values, axis=0)
    mad = np.median(np.abs(values - median), axis=0)
    robust = 1.4826 * mad
    conventional = np.std(values, axis=0)
    scale = np.where(robust > 1.0e-6, robust, conventional)
    scale = np.where(scale > 1.0e-6, scale, 1.0)
    if scale.shape != (JOINT_DIM,) or not np.all(np.isfinite(scale)):
        raise RuntimeError(f"Invalid diversity residual scale: {scale}")
    return scale


def diversity_distance(
    first: np.ndarray,
    second: np.ndarray,
    joint_std: np.ndarray,
    steps: int,
) -> float:
    normalized = (first[:steps] - second[:steps]) / joint_std.reshape(1, JOINT_DIM)
    return float(np.sqrt(np.mean(np.square(normalized))))


def retain_diverse_base_targets(
    result: BaseGenerationResult,
    joint_std: np.ndarray,
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows = [dict(row) for row in result.candidates]
    valid_indices = [index for index, row in enumerate(rows) if int(row["valid_target"])]
    pareto_indices: set[int] = set()
    if valid_indices:
        valid_rows = [rows[index] for index in valid_indices]
        local_pareto = set(v7.pareto_front(valid_rows))
        pareto_indices = {valid_indices[index] for index in local_pareto}
    retained: List[Dict[str, Any]] = []
    ordered = sorted(
        valid_indices,
        key=lambda index: (
            float(rows[index]["delta_score"]),
            -float(rows[index]["absolute_cartesian_improvement_m"]),
            str(rows[index]["candidate_id"]),
        ),
    )
    for index in ordered:
        row = rows[index]
        if index not in pareto_indices:
            row["base_retention_rejection_reason"] = "dominated_by_better_candidate"
            continue
        distances = [
            (
                diversity_distance(
                    np.asarray(row["residual"]),
                    np.asarray(previous["residual"]),
                    joint_std,
                    args.execution_horizon,
                ),
                diversity_distance(
                    np.asarray(row["residual"]),
                    np.asarray(previous["residual"]),
                    joint_std,
                    args.horizon,
                ),
            )
            for previous in retained
        ]
        prefix_nearest = min((item[0] for item in distances), default=-1.0)
        full_nearest = min((item[1] for item in distances), default=-1.0)
        if distances and any(
            prefix < args.min_prefix_diversity_rms
            or full < args.min_full_diversity_rms
            for prefix, full in distances
        ):
            row["base_retention_rejection_reason"] = "duplicate_or_near_duplicate"
            row["normalized_prefix_diversity_to_nearest"] = prefix_nearest
            row["normalized_full_diversity_to_nearest"] = full_nearest
            continue
        if len(retained) >= args.max_base_targets_per_window:
            row["base_retention_rejection_reason"] = "base_target_limit_reached"
            row["normalized_prefix_diversity_to_nearest"] = prefix_nearest
            row["normalized_full_diversity_to_nearest"] = full_nearest
            continue
        row["retained"] = 1
        row["base_target_id"] = stable_identifier(
            "base", result.window.path_name, result.window.window_start, row["candidate_id"]
        )
        row["normalized_prefix_diversity_to_nearest"] = prefix_nearest
        row["normalized_full_diversity_to_nearest"] = full_nearest
        retained.append(row)
    return rows, retained


def evaluate_scaled_targets(
    task: ScaleEvaluationTask, robot: v7.RobotContext
) -> ScaleEvaluationResult:
    assert_cpu_payload(task)
    args = argparse.Namespace(**task.v7_arguments)
    rows: List[Dict[str, Any]] = []
    started = time.perf_counter()
    candidate_index = 0
    for base in sorted(task.base_targets, key=lambda row: str(row["base_target_id"])):
        base_residual = np.asarray(base["residual"], dtype=np.float64)
        for scale in sorted(task.scales):
            scaled_residual = float(scale) * base_residual
            candidate = v7.Candidate(
                method=str(base["candidate_method"]),
                subtype=f"scaled_{base['candidate_subtype']}",
                residual=scaled_residual,
                deterministic_seed=stable_seed(
                    base["deterministic_seed"], "scale", format(float(scale), ".12g")
                ),
                metadata={
                    "base_target_id": base["base_target_id"],
                    "target_scale": float(scale),
                    "restart_index": int(base["restart_index"]),
                },
                runtime_seconds=0.0,
            )
            row = evaluate_v7_candidate(
                robot,
                task.window.context,
                task.prior_metrics,
                candidate,
                candidate_index,
                args,
            )
            target_id = stable_identifier(
                "target", base["base_target_id"], format(float(scale), ".12g")
            )
            row.update(
                {
                    "stage": "scaled",
                    "generation_stage": task.generation_stage,
                    "work_id": task.work_id,
                    "candidate_id": target_id,
                    "target_id": target_id,
                    "base_target_id": base["base_target_id"],
                    "base_candidate_id": base["candidate_id"],
                    "base_residual": base_residual,
                    "target_scale": float(scale),
                    "restart_index": int(base["restart_index"]),
                    "retained": 0,
                    "normalized_prefix_diversity_to_nearest": float(
                        base["normalized_prefix_diversity_to_nearest"]
                    ),
                    "normalized_full_diversity_to_nearest": float(
                        base["normalized_full_diversity_to_nearest"]
                    ),
                }
            )
            rows.append(row)
            candidate_index += 1
    return ScaleEvaluationResult(
        work_id=task.work_id,
        rows=tuple(rows),
        scoring_time_s=time.perf_counter() - started,
        generation_stage=task.generation_stage,
    )


def scale_worker_entry(task: ScaleEvaluationTask) -> ScaleEvaluationResult:
    if _WORKER_ROBOT is None:
        raise RuntimeError("Worker robot is not initialized")
    return evaluate_scaled_targets(task, _WORKER_ROBOT)


def ordered_map(
    tasks: Sequence[Any],
    serial_function: Any,
    worker_function: Any,
    robot: Optional[v7.RobotContext],
    executor: Optional[concurrent.futures.ProcessPoolExecutor],
) -> List[Any]:
    work_ids = [str(task.work_id) for task in tasks]
    if len(work_ids) != len(set(work_ids)):
        raise AssertionError("Submitted work IDs are not unique")
    for task in tasks:
        assert_cpu_payload(task)
    if executor is None:
        if robot is None:
            raise AssertionError("Serial evaluation requires a robot context")
        unordered = [serial_function(task, robot) for task in tasks]
    else:
        future_to_id = {
            executor.submit(worker_function, task): str(task.work_id) for task in tasks
        }
        unordered = []
        for future in concurrent.futures.as_completed(future_to_id):
            expected = future_to_id[future]
            result = future.result()
            if str(result.work_id) != expected:
                raise AssertionError(
                    f"Worker returned work_id={result.work_id!r}, expected {expected!r}"
                )
            unordered.append(result)
    by_id: Dict[str, Any] = {}
    for result in unordered:
        identifier = str(result.work_id)
        if identifier in by_id:
            raise AssertionError(f"work_id={identifier!r} returned more than once")
        by_id[identifier] = result
    if set(by_id) != set(work_ids):
        raise AssertionError("Worker result IDs do not match submitted work IDs")
    ordered = [by_id[identifier] for identifier in work_ids]
    if [str(result.work_id) for result in ordered] != work_ids:
        raise AssertionError("Canonical worker result ordering was not restored")
    return ordered


def retain_scaled_targets(
    rows: Sequence[Dict[str, Any]], max_targets: int
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    updated = [dict(row) for row in rows]
    valid = [row for row in updated if int(row["valid_target"]) == 1]
    selected: List[Dict[str, Any]] = []
    remaining = sorted(
        valid,
        key=lambda row: (
            float(row["delta_score"]),
            -float(row["absolute_cartesian_improvement_m"]),
            str(row["target_id"]),
        ),
    )
    represented_scales: set[float] = set()
    represented_bases: set[str] = set()
    while remaining and len(selected) < max_targets:
        best_index = min(
            range(len(remaining)),
            key=lambda index: (
                -int(float(remaining[index]["target_scale"]) not in represented_scales),
                -int(str(remaining[index]["base_target_id"]) not in represented_bases),
                float(remaining[index]["delta_score"]),
                -float(remaining[index]["absolute_cartesian_improvement_m"]),
                str(remaining[index]["target_id"]),
            ),
        )
        chosen = remaining.pop(best_index)
        chosen["retained"] = 1
        chosen["scaled_retention_rejection_reason"] = ""
        selected.append(chosen)
        represented_scales.add(float(chosen["target_scale"]))
        represented_bases.add(str(chosen["base_target_id"]))
    selected_ids = {str(row["target_id"]) for row in selected}
    for row in updated:
        identifier = str(row.get("target_id", ""))
        if identifier in selected_ids:
            selected_row = next(item for item in selected if str(item["target_id"]) == identifier)
            row.update(selected_row)
        elif int(row["valid_target"]) == 1:
            row["scaled_retention_rejection_reason"] = "target_limit_reached"
        else:
            row["scaled_retention_rejection_reason"] = "failed_independent_scale_validation"
    selected.sort(
        key=lambda row: (
            float(row["target_scale"]),
            str(row["base_target_id"]),
            float(row["delta_score"]),
            str(row["target_id"]),
        )
    )
    return updated, selected


def make_base_tasks(
    windows_by_work: Mapping[str, SourceWindow],
    work_ids: Sequence[str],
    methods: Sequence[str],
    restarts_per_method: int,
    generation_stage: str,
    args: argparse.Namespace,
    v7_arguments: Mapping[str, Any],
) -> List[BaseGenerationTask]:
    stage_arguments = dict(v7_arguments)
    stage_arguments["candidate_methods"] = list(methods)
    return [
        BaseGenerationTask(
            work_id=work_id,
            window=windows_by_work[work_id],
            restarts_per_method=restarts_per_method,
            candidate_seed=args.candidate_seed,
            v7_arguments=dict(stage_arguments),
            generation_stage=generation_stage,
        )
        for work_id in work_ids
    ]


def combine_base_results(
    work_id: str, results: Sequence[BaseGenerationResult]
) -> BaseGenerationResult:
    if not results:
        raise ValueError(f"Cannot combine an empty result list for {work_id}")
    candidates = sorted(
        [dict(row) for result in results for row in result.candidates],
        key=lambda row: (
            str(row["candidate_method"]),
            str(row["candidate_subtype"]),
            int(row["restart_index"]),
            int(row["local_candidate_index"]),
            str(row["candidate_id"]),
        ),
    )
    candidate_ids = [str(row["candidate_id"]) for row in candidates]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise AssertionError(f"Duplicate adaptive candidate IDs for {work_id}")
    first = results[0]
    return BaseGenerationResult(
        work_id=work_id,
        window=first.window,
        prior_metrics=dict(first.prior_metrics),
        candidates=tuple(candidates),
        generation_time_s=sum(result.generation_time_s for result in results),
        scoring_time_s=sum(result.scoring_time_s for result in results),
        generation_stage=results[-1].generation_stage,
    )


def flatten_base_results(
    base_results_by_work: Mapping[str, Sequence[BaseGenerationResult]],
    canonical_work_ids: Sequence[str],
) -> List[BaseGenerationResult]:
    return [
        result
        for work_id in canonical_work_ids
        for result in base_results_by_work.get(work_id, ())
    ]


def adaptive_diversity_scale(
    base_results_by_work: Mapping[str, Sequence[BaseGenerationResult]],
    canonical_work_ids: Sequence[str],
) -> np.ndarray:
    results = flatten_base_results(base_results_by_work, canonical_work_ids)
    try:
        return robust_joint_residual_std(results)
    except RuntimeError:
        # This keeps the adaptive pipeline resumable even if Stage A finds no
        # valid base. No invalid candidate is retained by this fallback scale.
        return np.ones(JOINT_DIM, dtype=np.float64)


def rerank_and_scale_windows(
    work_ids: Sequence[str],
    generation_stage: str,
    base_results_by_work: Mapping[str, Sequence[BaseGenerationResult]],
    joint_std: np.ndarray,
    base_rows_by_work: Dict[str, List[Dict[str, Any]]],
    scaled_rows_by_work: Dict[str, List[Dict[str, Any]]],
    scaled_history: List[Dict[str, Any]],
    selected_by_work: Dict[str, List[Dict[str, Any]]],
    prior_metrics_by_key: Dict[Tuple[str, int], Mapping[str, Any]],
    args: argparse.Namespace,
    v7_arguments: Mapping[str, Any],
    serial_robot: Optional[v7.RobotContext],
    executor: Optional[concurrent.futures.ProcessPoolExecutor],
) -> List[ScaleEvaluationResult]:
    combined_results: List[BaseGenerationResult] = []
    retained_by_work: Dict[str, List[Dict[str, Any]]] = {}
    for work_id in work_ids:
        combined = combine_base_results(work_id, base_results_by_work[work_id])
        combined_results.append(combined)
        updated, retained = retain_diverse_base_targets(combined, joint_std, args)
        base_rows_by_work[work_id] = updated
        retained_by_work[work_id] = retained
        prior_metrics_by_key[
            (combined.window.path_name, combined.window.window_start)
        ] = combined.prior_metrics

    scale_tasks = [
        ScaleEvaluationTask(
            work_id=result.work_id,
            window=result.window,
            prior_metrics=result.prior_metrics,
            base_targets=tuple(retained_by_work[result.work_id]),
            scales=tuple(sorted(float(scale) for scale in args.scales)),
            v7_arguments=dict(v7_arguments),
            generation_stage=generation_stage,
        )
        for result in combined_results
        if retained_by_work[result.work_id]
    ]
    scale_results = (
        ordered_map(
            scale_tasks,
            evaluate_scaled_targets,
            scale_worker_entry,
            serial_robot,
            executor,
        )
        if scale_tasks
        else []
    )
    result_by_work = {result.work_id: result for result in scale_results}
    reranked = set(work_ids)
    for row in scaled_history:
        if str(row["work_id"]) in reranked and int(row.get("retained", 0)):
            row["retained"] = 0
            row["superseded_by_later_stage"] = 1
    for work_id in work_ids:
        result = result_by_work.get(work_id)
        if result is None:
            scaled_rows_by_work[work_id] = []
            selected_by_work[work_id] = []
            continue
        updated, selected = retain_scaled_targets(
            result.rows, args.max_targets_per_window
        )
        for row in updated:
            row["superseded_by_later_stage"] = 0
        scaled_rows_by_work[work_id] = updated
        selected_by_work[work_id] = selected
        scaled_history.extend(updated)
    return scale_results


def adaptive_window_complete(
    work_id: str,
    base_rows_by_work: Mapping[str, Sequence[Mapping[str, Any]]],
    selected_by_work: Mapping[str, Sequence[Mapping[str, Any]]],
    args: argparse.Namespace,
) -> bool:
    retained_bases = sum(
        int(row.get("retained", 0)) for row in base_rows_by_work.get(work_id, ())
    )
    final_targets = len(selected_by_work.get(work_id, ()))
    return (
        retained_bases >= args.minimum_base_targets_before_stop
        and final_targets >= args.minimum_final_targets_before_stop
    )


def adaptive_signature(
    args: argparse.Namespace, windows: Sequence[SourceWindow]
) -> Dict[str, Any]:
    scientific_arguments = {
        "train_prior": str(v7.resolve_project_path(args.train_prior)),
        "train_windows": str(v7.resolve_project_path(args.train_windows)),
        "split_manifest": str(v7.resolve_project_path(args.split_manifest)),
        "robot_urdf": str(v7.resolve_project_path(args.robot_urdf)),
        "horizon": args.horizon,
        "execution_horizon": args.execution_horizon,
        "candidate_seed": args.candidate_seed,
        "primary_methods": list(args.primary_methods),
        "fallback_methods": list(args.fallback_methods),
        "primary_restarts_per_method": args.primary_restarts_per_method,
        "fallback_restarts_per_method": args.fallback_restarts_per_method,
        "minimum_base_targets_before_stop": args.minimum_base_targets_before_stop,
        "minimum_final_targets_before_stop": args.minimum_final_targets_before_stop,
        "fallback_trigger_final_target_count": args.fallback_trigger_final_target_count,
        "enable_early_stop": bool(args.enable_early_stop),
        "max_base_targets_per_window": args.max_base_targets_per_window,
        "max_targets_per_window": args.max_targets_per_window,
        "min_prefix_diversity_rms": args.min_prefix_diversity_rms,
        "min_full_diversity_rms": args.min_full_diversity_rms,
        "scales": [float(scale) for scale in args.scales],
        "window_keys": [
            [window.path_name, int(window.window_start)] for window in windows
        ],
    }
    encoded = json.dumps(
        scientific_arguments, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return {
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "scientific_arguments": scientific_arguments,
    }


def save_adaptive_state(
    output_dir: Path,
    signature: Mapping[str, Any],
    state: Mapping[str, Any],
) -> None:
    stage_count = len(state["completed_stages"])
    stage_suffix = (
        str(state["completed_stages"][-1]) if stage_count else "initial"
    )
    pickle_path = output_dir / (
        f".adaptive_generation_state_{stage_count:02d}_{stage_suffix}.pkl"
    )
    json_path = output_dir / ADAPTIVE_STATE_JSON
    atomic_pickle(pickle_path, state)
    atomic_json(
        json_path,
        {
            "format_version": 1,
            "signature": signature,
            "completed_stages": list(state["completed_stages"]),
            "intermediate_canonical_data": pickle_path.name,
            "candidate_count": sum(
                len(result.candidates)
                for results in state["base_results_by_work"].values()
                for result in results
            ),
            "retained_target_count": sum(
                len(rows) for rows in state["selected_by_work"].values()
            ),
            "updated_at_unix_seconds": time.time(),
        },
    )


def load_adaptive_state(
    output_dir: Path, expected_signature: Mapping[str, Any]
) -> Dict[str, Any]:
    json_path = output_dir / ADAPTIVE_STATE_JSON
    if not json_path.exists():
        raise FileNotFoundError(
            "--resume requires adaptive_generation_state.json"
        )
    with json_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    pickle_name = str(metadata.get("intermediate_canonical_data", ""))
    pickle_path = output_dir / pickle_name
    if not pickle_name or not pickle_path.exists():
        raise FileNotFoundError(
            "Adaptive state JSON does not reference an existing intermediate file"
        )
    if metadata.get("signature", {}).get("sha256") != expected_signature["sha256"]:
        raise ValueError(
            "Adaptive resume arguments or canonical windows differ from the saved state"
        )
    state = load_pickle(pickle_path)
    if list(state.get("completed_stages", ())) != list(
        metadata.get("completed_stages", ())
    ):
        raise ValueError("Adaptive JSON and intermediate state disagree on completed stages")
    return state


def stage_record(
    stage_name: str,
    entering_work_ids: Sequence[str],
    stage_results: Sequence[BaseGenerationResult],
    base_rows_by_work: Mapping[str, Sequence[Mapping[str, Any]]],
    scaled_rows_by_work: Mapping[str, Sequence[Mapping[str, Any]]],
    selected_by_work: Mapping[str, Sequence[Mapping[str, Any]]],
    canonical_work_ids: Sequence[str],
    args: argparse.Namespace,
    cumulative_wall_time_s: float,
) -> Dict[str, Any]:
    completed = {
        work_id
        for work_id in canonical_work_ids
        if adaptive_window_complete(
            work_id, base_rows_by_work, selected_by_work, args
        )
    }
    entering = set(entering_work_ids)
    base_rows = [
        row for work_id in entering_work_ids for row in base_rows_by_work.get(work_id, ())
    ]
    scaled_rows = [
        row for work_id in entering_work_ids for row in scaled_rows_by_work.get(work_id, ())
    ]
    return {
        "generation_stage": stage_name,
        "windows_entering_stage": len(entering_work_ids),
        "windows_attempted": len({result.work_id for result in stage_results}),
        "candidates_generated": sum(len(result.candidates) for result in stage_results),
        "valid_base_candidates": sum(int(row["valid_target"]) for row in base_rows),
        "retained_diverse_bases": sum(int(row["retained"]) for row in base_rows),
        "valid_scaled_targets": sum(int(row["valid_target"]) for row in scaled_rows),
        "retained_final_targets": sum(
            len(selected_by_work.get(work_id, ())) for work_id in entering_work_ids
        ),
        "windows_completed_after_stage": len(completed),
        "windows_remaining_after_stage": len(canonical_work_ids) - len(completed),
        "zero_target_windows_after_stage": sum(
            int(not selected_by_work.get(work_id)) for work_id in canonical_work_ids
        ),
        "cumulative_wall_time_s": cumulative_wall_time_s,
    }


def print_stage_record(record: Mapping[str, Any], stage_wall_time_s: float) -> None:
    print(
        f"stage={record['generation_stage']}: "
        f"entering={record['windows_entering_stage']}, "
        f"completed={record['windows_completed_after_stage']}, "
        f"remaining={record['windows_remaining_after_stage']}, "
        f"candidates={record['candidates_generated']}, "
        f"retained_targets={record['retained_final_targets']}, "
        f"stage_wall={stage_wall_time_s:.3f}s, "
        f"cumulative_wall={record['cumulative_wall_time_s']:.3f}s"
    )


def run_exhaustive_pipeline(
    windows: Sequence[SourceWindow],
    windows_by_work: Mapping[str, SourceWindow],
    args: argparse.Namespace,
    v7_arguments: Mapping[str, Any],
    serial_robot: Optional[v7.RobotContext],
    executor: Optional[concurrent.futures.ProcessPoolExecutor],
) -> Dict[str, Any]:
    canonical_work_ids = list(windows_by_work)
    tasks = make_base_tasks(
        windows_by_work,
        canonical_work_ids,
        list(v7_arguments["candidate_methods"]),
        args.restarts_per_method,
        "exhaustive",
        args,
        v7_arguments,
    )
    stage_started = time.perf_counter()
    base_results = ordered_map(
        tasks, generate_base_candidates, base_worker_entry, serial_robot, executor
    )
    joint_std = robust_joint_residual_std(base_results)
    base_rows_by_work: Dict[str, List[Dict[str, Any]]] = {}
    scaled_rows_by_work: Dict[str, List[Dict[str, Any]]] = {}
    selected_by_work: Dict[str, List[Dict[str, Any]]] = {}
    prior_metrics_by_key: Dict[Tuple[str, int], Mapping[str, Any]] = {}
    base_results_by_work = {
        result.work_id: [result] for result in base_results
    }
    scaled_history: List[Dict[str, Any]] = []
    scale_results = rerank_and_scale_windows(
        canonical_work_ids,
        "exhaustive",
        base_results_by_work,
        joint_std,
        base_rows_by_work,
        scaled_rows_by_work,
        scaled_history,
        selected_by_work,
        prior_metrics_by_key,
        args,
        v7_arguments,
        serial_robot,
        executor,
    )
    stage_wall = time.perf_counter() - stage_started
    record = stage_record(
        "exhaustive",
        canonical_work_ids,
        base_results,
        base_rows_by_work,
        scaled_rows_by_work,
        selected_by_work,
        canonical_work_ids,
        args,
        stage_wall,
    )
    print_stage_record(record, stage_wall)
    return {
        "base_results": base_results,
        "base_rows": [
            row for work_id in canonical_work_ids for row in base_rows_by_work[work_id]
        ],
        "scaled_rows": scaled_history,
        "selected_rows": [
            row for work_id in canonical_work_ids for row in selected_by_work.get(work_id, ())
        ],
        "prior_metrics_by_key": prior_metrics_by_key,
        "joint_std": joint_std,
        "stage_records": [record],
        "generation_time_s": sum(result.generation_time_s for result in base_results),
        "scoring_time_s": sum(result.scoring_time_s for result in base_results)
        + sum(result.scoring_time_s for result in scale_results),
        "fallback_windows_attempted": 0,
        "fallback_windows_rescued": 0,
    }


def run_adaptive_pipeline(
    windows: Sequence[SourceWindow],
    windows_by_work: Mapping[str, SourceWindow],
    args: argparse.Namespace,
    v7_arguments: Mapping[str, Any],
    serial_robot: Optional[v7.RobotContext],
    executor: Optional[concurrent.futures.ProcessPoolExecutor],
) -> Dict[str, Any]:
    canonical_work_ids = list(windows_by_work)
    signature = adaptive_signature(args, windows)
    if args.resume:
        state = load_adaptive_state(args.output_dir, signature)
        print(
            "resuming adaptive generation after stages: "
            + ", ".join(state["completed_stages"])
        )
    else:
        state = {
            "completed_stages": [],
            "base_results_by_work": {work_id: [] for work_id in canonical_work_ids},
            "base_rows_by_work": {},
            "scaled_rows_by_work": {},
            "scaled_history": [],
            "selected_by_work": {},
            "prior_metrics_by_key": {},
            "joint_std": None,
            "stage_records": [],
            "cumulative_stage_wall_time_s": 0.0,
            "generation_time_s": 0.0,
            "scoring_time_s": 0.0,
            "fallback_windows_attempted": 0,
            "fallback_windows_rescued": 0,
        }

    base_results_by_work = state["base_results_by_work"]
    base_rows_by_work = state["base_rows_by_work"]
    scaled_rows_by_work = state["scaled_rows_by_work"]
    scaled_history = state["scaled_history"]
    selected_by_work = state["selected_by_work"]
    prior_metrics_by_key = state["prior_metrics_by_key"]

    for stage_name in (
        "primary_jacobian",
        "primary_sequential_ik",
        "fallback",
    ):
        if stage_name in state["completed_stages"]:
            continue
        if stage_name == "primary_jacobian":
            methods = [str(args.primary_methods[0])]
            entering_work_ids = list(canonical_work_ids)
            restarts = args.primary_restarts_per_method
        elif stage_name == "primary_sequential_ik":
            methods = [str(method) for method in args.primary_methods[1:]]
            entering_work_ids = [
                work_id
                for work_id in canonical_work_ids
                if not args.enable_early_stop
                or not adaptive_window_complete(
                    work_id, base_rows_by_work, selected_by_work, args
                )
            ]
            restarts = args.primary_restarts_per_method
        else:
            methods = list(args.fallback_methods)
            entering_work_ids = [
                work_id
                for work_id in canonical_work_ids
                if len(selected_by_work.get(work_id, ()))
                < args.fallback_trigger_final_target_count
            ]
            restarts = args.fallback_restarts_per_method

        before_final_counts = {
            work_id: len(selected_by_work.get(work_id, ()))
            for work_id in entering_work_ids
        }
        stage_started = time.perf_counter()
        tasks = make_base_tasks(
            windows_by_work,
            entering_work_ids,
            methods,
            restarts,
            stage_name,
            args,
            v7_arguments,
        )
        stage_results = (
            ordered_map(
                tasks,
                generate_base_candidates,
                base_worker_entry,
                serial_robot,
                executor,
            )
            if tasks
            else []
        )
        for result in stage_results:
            base_results_by_work[result.work_id].append(result)
        if stage_results or state["joint_std"] is None:
            state["joint_std"] = adaptive_diversity_scale(
                base_results_by_work, canonical_work_ids
            )
        joint_std = np.asarray(state["joint_std"], dtype=np.float64)
        reranked_work_ids = entering_work_ids if stage_results else []
        scale_results = rerank_and_scale_windows(
            reranked_work_ids,
            stage_name,
            base_results_by_work,
            joint_std,
            base_rows_by_work,
            scaled_rows_by_work,
            scaled_history,
            selected_by_work,
            prior_metrics_by_key,
            args,
            v7_arguments,
            serial_robot,
            executor,
        )
        stage_wall = time.perf_counter() - stage_started
        state["cumulative_stage_wall_time_s"] += stage_wall
        state["generation_time_s"] += sum(
            result.generation_time_s for result in stage_results
        )
        state["scoring_time_s"] += sum(
            result.scoring_time_s for result in stage_results
        ) + sum(result.scoring_time_s for result in scale_results)
        if stage_name == "fallback":
            state["fallback_windows_attempted"] = len(entering_work_ids)
            state["fallback_windows_rescued"] = sum(
                int(before_final_counts[work_id] == 0)
                and int(len(selected_by_work.get(work_id, ())) > 0)
                for work_id in entering_work_ids
            )
        record = stage_record(
            stage_name,
            entering_work_ids,
            stage_results,
            base_rows_by_work,
            scaled_rows_by_work,
            selected_by_work,
            canonical_work_ids,
            args,
            state["cumulative_stage_wall_time_s"],
        )
        state["stage_records"].append(record)
        state["completed_stages"].append(stage_name)
        save_adaptive_state(args.output_dir, signature, state)
        print_stage_record(record, stage_wall)

    base_results = flatten_base_results(base_results_by_work, canonical_work_ids)
    return {
        "base_results": base_results,
        "base_rows": [
            row
            for work_id in canonical_work_ids
            for row in base_rows_by_work.get(work_id, ())
        ],
        "scaled_rows": list(scaled_history),
        "selected_rows": [
            row
            for work_id in canonical_work_ids
            for row in selected_by_work.get(work_id, ())
        ],
        "prior_metrics_by_key": prior_metrics_by_key,
        "joint_std": np.asarray(state["joint_std"], dtype=np.float64),
        "stage_records": list(state["stage_records"]),
        "generation_time_s": float(state["generation_time_s"]),
        "scoring_time_s": float(state["scoring_time_s"]),
        "fallback_windows_attempted": int(state["fallback_windows_attempted"]),
        "fallback_windows_rescued": int(state["fallback_windows_rescued"]),
    }


def candidate_results_for_storage(
    candidate_frame: pd.DataFrame,
    window_frame: pd.DataFrame,
    mode: str,
) -> Optional[pd.DataFrame]:
    if mode == "none":
        return None
    if mode == "all":
        return candidate_frame
    zero_window_rows = window_frame.loc[
        window_frame["zero_valid_target"].astype("int64") == 1,
        ["path_name", "window_start"],
    ]
    zero_windows = set(
        zip(
            zero_window_rows["path_name"].astype(str).tolist(),
            zero_window_rows["window_start"].astype("int64").tolist(),
        )
    )
    candidate_paths = candidate_frame["path_name"].astype(str).tolist()
    candidate_starts = candidate_frame["window_start"].astype("int64").tolist()
    in_zero_window = np.asarray(
        [
            (path_name, window_start) in zero_windows
            for path_name, window_start in zip(candidate_paths, candidate_starts)
        ],
        dtype=bool,
    )
    retained = candidate_frame["retained"].astype(int).to_numpy() == 1
    valid_diversity_rejected = (
        (candidate_frame["stage"].to_numpy() == "base")
        & (candidate_frame["valid_target"].astype(int).to_numpy() == 1)
        & ~retained
    )
    failed_zero_window = (
        in_zero_window
        & (candidate_frame["valid_target"].astype(int).to_numpy() == 0)
    )
    return candidate_frame.loc[
        retained | valid_diversity_rejected | failed_zero_window
    ].copy()


def csv_safe_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    excluded = {
        "candidate_q", "residual", "metrics", "base_residual",
        "prior_q", "target_q", "condition", "desired",
    }
    result: Dict[str, Any] = {}
    for key, value in row.items():
        if key in excluded:
            continue
        if isinstance(value, np.ndarray):
            continue
        if isinstance(value, (Mapping, list, tuple)):
            result[key] = json.dumps(json_safe(value), sort_keys=True)
        else:
            result[key] = value
    return result


def selected_arrays(
    selected: Sequence[Dict[str, Any]],
    windows_by_key: Mapping[Tuple[str, int], SourceWindow],
    prior_metrics_by_key: Mapping[Tuple[str, int], Mapping[str, Any]],
) -> Dict[str, np.ndarray]:
    if not selected:
        raise RuntimeError("No scaled targets were retained")
    ordered = sorted(
        selected,
        key=lambda row: (
            str(row["path_name"]), int(row["window_start"]),
            str(row["target_id"]),
        ),
    )
    arrays: Dict[str, np.ndarray] = {
        "path_name": np.asarray([str(row["path_name"]) for row in ordered]),
        "window_start": np.asarray([int(row["window_start"]) for row in ordered], dtype=np.int64),
        "base_target_id": np.asarray([str(row["base_target_id"]) for row in ordered]),
        "target_id": np.asarray([str(row["target_id"]) for row in ordered]),
        "candidate_method": np.asarray([str(row["candidate_method"]) for row in ordered]),
        "restart_index": np.asarray([int(row["restart_index"]) for row in ordered], dtype=np.int64),
        "target_scale": np.asarray([float(row["target_scale"]) for row in ordered], dtype=np.float64),
        "prior_q": np.stack([
            windows_by_key[(str(row["path_name"]), int(row["window_start"]))].prior_q
            for row in ordered
        ]).astype(np.float64),
        "residual_target": np.stack([np.asarray(row["residual"]) for row in ordered]).astype(np.float64),
        "target_q": np.stack([np.asarray(row["candidate_q"]) for row in ordered]).astype(np.float64),
        "base_residual": np.stack([np.asarray(row["base_residual"]) for row in ordered]).astype(np.float64),
        "condition": np.stack([
            windows_by_key[(str(row["path_name"]), int(row["window_start"]))].condition
            for row in ordered
        ]).astype(np.float64),
        "desired_cartesian_window": np.stack([
            windows_by_key[(str(row["path_name"]), int(row["window_start"]))].desired
            for row in ordered
        ]).astype(np.float64),
        "execution_horizon": np.asarray([
            windows_by_key[(str(row["path_name"]), int(row["window_start"]))].execution_horizon
            for row in ordered
        ], dtype=np.int64),
        "prior_prefix_cartesian_mean_error_m": np.asarray([
            float(prior_metrics_by_key[(str(row["path_name"]), int(row["window_start"]))]["prefix_cartesian_mean_error_m"])
            for row in ordered
        ]),
        "target_prefix_cartesian_mean_error_m": np.asarray([
            float(row["metrics"]["prefix_cartesian_mean_error_m"]) for row in ordered
        ]),
        "cartesian_improvement_m": np.asarray([
            float(row["absolute_cartesian_improvement_m"]) for row in ordered
        ]),
        "prior_robot_aware_score": np.zeros(len(ordered), dtype=np.float64),
        "target_robot_aware_score": np.asarray([float(row["delta_score"]) for row in ordered]),
        "delta_score": np.asarray([float(row["delta_score"]) for row in ordered]),
        "hard_safe": np.ones(len(ordered), dtype=np.int8),
        "normalized_prefix_diversity_to_nearest": np.asarray([
            float(row["normalized_prefix_diversity_to_nearest"]) for row in ordered
        ]),
        "normalized_full_diversity_to_nearest": np.asarray([
            float(row["normalized_full_diversity_to_nearest"]) for row in ordered
        ]),
    }
    condition_dim = arrays["condition"].shape[-1]
    feature_names = tuple(
        str(name) for name in getattr(v7_dataset_builder, "CONDITION_FEATURE_NAMES", ())
    )
    if len(feature_names) != condition_dim:
        raise ValueError(
            "The v7 condition feature-name schema does not match the source "
            f"condition dimension: names={len(feature_names)}, dim={condition_dim}"
        )
    arrays["condition_feature_names"] = np.asarray(feature_names)
    prior_scalar_keys = sorted(
        key for key, value in prior_metrics_by_key[
            (str(ordered[0]["path_name"]), int(ordered[0]["window_start"]))
        ].items()
        if np.isscalar(value) and key not in {"finite", "branch_jump"}
    )
    target_scalar_keys = sorted(
        key for key, value in ordered[0]["metrics"].items()
        if np.isscalar(value) and key not in {"finite", "branch_jump"}
    )
    for key in prior_scalar_keys:
        arrays[f"prior_{key}"] = np.asarray([
            prior_metrics_by_key[(str(row["path_name"]), int(row["window_start"]))][key]
            for row in ordered
        ])
    for key in target_scalar_keys:
        arrays[f"target_{key}"] = np.asarray([row["metrics"][key] for row in ordered])
    return arrays


def validate_selected_arrays(
    arrays: Mapping[str, np.ndarray], scales: Sequence[float]
) -> None:
    count = len(arrays["target_id"])
    if len(set(arrays["target_id"].tolist())) != count:
        raise AssertionError("Retained target IDs are not unique")
    composite = list(zip(
        arrays["path_name"].tolist(), arrays["window_start"].tolist(),
        arrays["target_id"].tolist(),
    ))
    if len(set(composite)) != count:
        raise AssertionError("Retained path/window/target identities are duplicated")
    numeric_arrays = [
        value for value in arrays.values()
        if np.issubdtype(value.dtype, np.number)
    ]
    if any(not np.all(np.isfinite(value)) for value in numeric_arrays):
        raise AssertionError("Retained output contains NaN or infinity")
    if arrays["prior_q"].shape[1:] != (HORIZON, JOINT_DIM):
        raise AssertionError("Retained prior_q shape is inconsistent")
    if arrays["residual_target"].shape[1:] != (HORIZON, JOINT_DIM):
        raise AssertionError("Retained residual target shape is inconsistent")
    if not np.allclose(
        arrays["prior_q"] + arrays["residual_target"],
        arrays["target_q"], rtol=1e-7, atol=1e-8,
    ):
        raise AssertionError("prior_q + residual_target does not reproduce target_q")
    if np.any(arrays["hard_safe"] != 1):
        raise AssertionError("A retained target is not hard-safe")
    if np.any(arrays["cartesian_improvement_m"] <= 0.0):
        raise AssertionError("A retained target does not improve Cartesian error")
    if np.any(arrays["delta_score"] >= 0.0):
        raise AssertionError("A retained target has non-negative delta_score")
    requested = np.asarray(scales, dtype=np.float64)
    for index, scale in enumerate(arrays["target_scale"]):
        if not np.any(np.isclose(scale, requested, rtol=1e-10, atol=1e-12)):
            raise AssertionError(f"Unexpected target scale {scale}")
        if not np.allclose(
            arrays["residual_target"][index],
            scale * arrays["base_residual"][index],
            rtol=1e-7, atol=1e-8,
        ):
            raise AssertionError("target_scale is inconsistent with the base residual")


def split_reasons(value: Any) -> List[str]:
    return [reason for reason in str(value or "").split("|") if reason]


def build_summaries(
    candidate_rows: Sequence[Dict[str, Any]],
    selected_rows: Sequence[Dict[str, Any]],
    windows: Sequence[SourceWindow],
) -> Dict[str, pd.DataFrame]:
    candidate_frame = pd.DataFrame([csv_safe_row(row) for row in candidate_rows])
    selected_frame = pd.DataFrame([csv_safe_row(row) for row in selected_rows])
    window_records: List[Dict[str, Any]] = []
    for window in windows:
        mask = (
            (candidate_frame["path_name"] == window.path_name)
            & (candidate_frame["window_start"] == window.window_start)
        )
        group = candidate_frame.loc[mask]
        base = group[group["stage"] == "base"]
        scaled = group[group["stage"] == "scaled"]
        selected = scaled[scaled["retained"] == 1]
        all_reasons = [
            reason
            for value in group.get("rejection_reasons", pd.Series(dtype=str))
            for reason in split_reasons(value)
        ]
        window_records.append(
            {
                "path_name": window.path_name,
                "window_start": window.window_start,
                "generated_candidate_count": len(base),
                "hard_safe_candidate_count": int(base["hard_safe"].sum()) if len(base) else 0,
                "cartesian_improving_candidate_count": int(base["cartesian_improving"].sum()) if len(base) else 0,
                "negative_delta_score_candidate_count": int(base["negative_delta_score"].sum()) if len(base) else 0,
                "valid_base_candidate_count": int(base["valid_target"].sum()) if len(base) else 0,
                "diverse_base_target_count": int(base["retained"].sum()) if len(base) else 0,
                "valid_scaled_target_count": int(scaled["valid_target"].sum()) if len(scaled) else 0,
                "retained_target_count": len(selected),
                "distinct_retained_scale_count": selected["target_scale"].nunique() if len(selected) else 0,
                "best_cartesian_improvement_m": float(group["absolute_cartesian_improvement_m"].max()) if len(group) else math.nan,
                "best_delta_score": float(group["delta_score"].min()) if len(group) else math.nan,
                "zero_valid_target": int(len(selected) == 0),
                "rejection_count": len(all_reasons),
                "rejection_reasons": "|".join(sorted(set(all_reasons))),
            }
        )
    window_frame = pd.DataFrame(window_records)
    path_records = []
    for path_name, group in window_frame.groupby("path_name", sort=True):
        path_records.append(
            {
                "path_name": path_name,
                "window_count": len(group),
                "generated_candidate_count": int(group["generated_candidate_count"].sum()),
                "valid_base_candidate_count": int(group["valid_base_candidate_count"].sum()),
                "diverse_base_target_count": int(group["diverse_base_target_count"].sum()),
                "valid_scaled_target_count": int(group["valid_scaled_target_count"].sum()),
                "retained_target_count": int(group["retained_target_count"].sum()),
                "covered_window_count": int((group["retained_target_count"] > 0).sum()),
                "window_coverage": float((group["retained_target_count"] > 0).mean()),
                "mean_targets_per_window": float(group["retained_target_count"].mean()),
                "zero_target_window_count": int(group["zero_valid_target"].sum()),
            }
        )
    path_frame = pd.DataFrame(path_records)
    scaled_frame = candidate_frame[candidate_frame["stage"] == "scaled"]
    scale_records = []
    for scale, group in scaled_frame.groupby("target_scale", sort=True):
        retained = group[group["retained"] == 1]
        evaluated_count = len(group)
        hard_safe_count = int(group["hard_safe"].sum())
        cartesian_improving_count = int(group["cartesian_improving"].sum())
        negative_delta_score_count = int(group["negative_delta_score"].sum())
        independently_valid_count = int(group["valid_target"].sum())
        scale_records.append(
            {
                "target_scale": float(scale),
                "evaluated_count": evaluated_count,
                "hard_safe_count": hard_safe_count,
                "cartesian_improving_count": cartesian_improving_count,
                "negative_delta_score_count": negative_delta_score_count,
                "independently_valid_count": independently_valid_count,
                "retained_count": len(retained),
                "hard_safe_rate": hard_safe_count / max(evaluated_count, 1),
                "cartesian_improving_rate": (
                    cartesian_improving_count / max(evaluated_count, 1)
                ),
                "robot_aware_score_improving_rate": (
                    negative_delta_score_count / max(evaluated_count, 1)
                ),
                "independently_valid_rate": (
                    independently_valid_count / max(evaluated_count, 1)
                ),
                "retained_window_coverage": float(
                    retained[["path_name", "window_start"]].drop_duplicates().shape[0]
                    / max(len(windows), 1)
                ),
                "mean_cartesian_improvement_m": float(group["absolute_cartesian_improvement_m"].mean()),
                "median_cartesian_improvement_m": float(group["absolute_cartesian_improvement_m"].median()),
                "mean_delta_score": float(group["delta_score"].replace([np.inf, -np.inf], np.nan).mean()),
                "median_delta_score": float(group["delta_score"].replace([np.inf, -np.inf], np.nan).median()),
            }
        )
    scale_frame = pd.DataFrame(scale_records)
    method_records = []
    base_frame = candidate_frame[candidate_frame["stage"] == "base"]
    retained_methods_by_window: Dict[Tuple[str, int], set[str]] = {}
    for (path_name, window_start), group in base_frame.groupby(
        ["path_name", "window_start"], sort=True
    ):
        retained_methods_by_window[(str(path_name), int(window_start))] = set(
            group.loc[group["retained"] == 1, "candidate_method"].astype(str)
        )
    contribution_records: List[Dict[str, Any]] = []
    for method, group in base_frame.groupby("candidate_method", sort=True):
        generated_count = len(group)
        cumulative_generation_time = float(
            group["v8_candidate_generation_time_s"].fillna(0.0).sum()
        )
        cumulative_scoring_time = float(
            group["v8_fk_scoring_time_s"].fillna(0.0).sum()
        )
        valid_base_count = int(group["valid_target"].sum())
        retained_base_count = int(group["retained"].sum())
        attempted_windows = group[["path_name", "window_start"]].drop_duplicates()
        valid_windows = group.loc[
            group["valid_target"] == 1, ["path_name", "window_start"]
        ].drop_duplicates()
        retained_windows = group.loc[
            group["retained"] == 1, ["path_name", "window_start"]
        ].drop_duplicates()
        uniquely_covered = sum(
            retained_methods_by_window.get((str(row.path_name), int(row.window_start)))
            == {str(method)}
            for row in retained_windows.itertuples(index=False)
        )
        common = {
            "candidate_method": method,
            "generated_count": generated_count,
            "generation_stage": "|".join(
                sorted(set(group["generation_stage"].astype(str)))
            ),
            "windows_attempted": len(attempted_windows),
            "windows_with_valid_base": len(valid_windows),
            "windows_with_retained_base": len(retained_windows),
            "windows_uniquely_covered_by_method": int(uniquely_covered),
            "retained_base_count": retained_base_count,
            "valid_base_rate": valid_base_count / max(generated_count, 1),
            "retained_per_1000_candidates": (
                1000.0 * retained_base_count / max(generated_count, 1)
            ),
            "cumulative_generation_time_s": cumulative_generation_time,
            "cumulative_fk_scoring_time_s": cumulative_scoring_time,
            "mean_generation_time_per_candidate_s": (
                cumulative_generation_time / max(generated_count, 1)
            ),
            "mean_scoring_time_per_candidate_s": (
                cumulative_scoring_time / max(generated_count, 1)
            ),
        }
        contribution_records.append(common)
        method_records.append(
            {
                **common,
                "hard_safe_count": int(group["hard_safe"].sum()),
                "cartesian_improving_count": int(group["cartesian_improving"].sum()),
                "negative_delta_score_count": int(group["negative_delta_score"].sum()),
                "valid_base_count": valid_base_count,
                "mean_delta_score": float(group["delta_score"].replace([np.inf, -np.inf], np.nan).mean()),
            }
        )
    reason_counts: Dict[Tuple[str, str], int] = {}
    for row in candidate_rows:
        reasons = split_reasons(row.get("rejection_reasons", ""))
        for key in ("base_retention_rejection_reason", "scaled_retention_rejection_reason"):
            if row.get(key):
                reasons.append(str(row[key]))
        for reason in sorted(set(reasons)):
            identity = (str(row["stage"]), reason)
            reason_counts[identity] = reason_counts.get(identity, 0) + 1
    rejection_frame = pd.DataFrame(
        [
            {"stage": stage, "rejection_reason": reason, "count": count}
            for (stage, reason), count in sorted(reason_counts.items())
        ]
    )
    diversity_frame = base_frame[
        [
            "path_name", "window_start", "candidate_id", "candidate_method",
            "restart_index", "valid_target", "retained",
            "base_retention_rejection_reason",
            "normalized_prefix_diversity_to_nearest",
            "normalized_full_diversity_to_nearest",
        ]
    ].copy()
    return {
        "candidate": candidate_frame,
        "selected": selected_frame,
        "window": window_frame,
        "path": path_frame,
        "scale": scale_frame,
        "method": pd.DataFrame(method_records),
        "method_contribution": pd.DataFrame(contribution_records),
        "rejection": rejection_frame,
        "diversity": diversity_frame,
    }


def main() -> int:
    wall_started = time.perf_counter()
    args = parse_args()
    validate_args(args)
    existing = [
        args.output_dir / name
        for name in OUTPUT_FILENAMES
        if (args.output_dir / name).exists()
    ]
    if existing and not args.overwrite and not args.resume:
        raise FileExistsError(f"Outputs already exist: {existing}; pass --overwrite")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.overwrite:
        state_json = args.output_dir / ADAPTIVE_STATE_JSON
        if state_json.exists():
            state_json.unlink()
        for state_path in args.output_dir.glob(".adaptive_generation_state_*.pkl"):
            state_path.unlink()
    windows = load_source_windows(args)
    windows_by_key = {(window.path_name, window.window_start): window for window in windows}
    windows_by_work = {
        f"{window.path_name}::{window.window_start}": window for window in windows
    }
    v7_arguments = worker_arguments(args)
    logical_cpus = os.cpu_count() or 1
    active_workers = args.num_workers
    print(f"detected logical CPU count: {logical_cpus}")
    print(f"requested workers: {args.num_workers}")
    print(f"active workers: {active_workers}")
    print(f"generation policy: {args.generation_policy}")
    serial_robot = make_robot_context(args.robot_urdf) if args.num_workers == 1 else None
    executor: Optional[concurrent.futures.ProcessPoolExecutor] = None
    try:
        if args.num_workers > 1:
            executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=args.num_workers,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=initialize_worker,
                initargs=(str(v7.resolve_project_path(args.robot_urdf)),),
            )
        if args.generation_policy == "adaptive":
            pipeline = run_adaptive_pipeline(
                windows,
                windows_by_work,
                args,
                v7_arguments,
                serial_robot,
                executor,
            )
        else:
            pipeline = run_exhaustive_pipeline(
                windows,
                windows_by_work,
                args,
                v7_arguments,
                serial_robot,
                executor,
            )
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    base_results = pipeline["base_results"]
    base_rows = pipeline["base_rows"]
    scaled_rows = pipeline["scaled_rows"]
    selected_rows = pipeline["selected_rows"]
    prior_metrics_by_key = pipeline["prior_metrics_by_key"]
    joint_std = pipeline["joint_std"]
    generation_stage_order = {
        "exhaustive": 0,
        "primary_jacobian": 0,
        "primary_sequential_ik": 1,
        "fallback": 2,
    }
    candidate_rows = sorted(
        [*base_rows, *scaled_rows],
        key=lambda row: (
            str(row["path_name"]), int(row["window_start"]),
            0 if row["stage"] == "base" else 1,
            generation_stage_order.get(str(row.get("generation_stage", "")), 99),
            str(row["candidate_id"]),
        ),
    )
    selected_rows.sort(
        key=lambda row: (
            str(row["path_name"]), int(row["window_start"]), str(row["target_id"])
        )
    )
    arrays = selected_arrays(selected_rows, windows_by_key, prior_metrics_by_key)
    validate_selected_arrays(arrays, args.scales)
    summaries = build_summaries(candidate_rows, selected_rows, windows)
    scale_evaluated = int(summaries["scale"]["evaluated_count"].sum())
    scale_valid = int(summaries["scale"]["independently_valid_count"].sum())
    scale_retained = int(summaries["scale"]["retained_count"].sum())
    if scale_evaluated != len(scaled_rows):
        raise AssertionError(
            "scale_summary evaluated_count does not equal all scaled evaluations"
        )
    if scale_valid != sum(int(row["valid_target"]) for row in scaled_rows):
        raise AssertionError(
            "scale_summary independently_valid_count does not equal valid scaled targets"
        )
    if scale_retained != len(selected_rows):
        raise AssertionError(
            "scale_summary retained_count does not equal retained final targets"
        )
    atomic_npz(args.output_dir / "selected_targets.npz", arrays)
    stored_candidate_frame = candidate_results_for_storage(
        summaries["candidate"], summaries["window"], args.candidate_results_mode
    )
    candidate_results_path = args.output_dir / "candidate_results.csv"
    if stored_candidate_frame is None:
        if candidate_results_path.exists():
            candidate_results_path.unlink()
    else:
        atomic_csv(candidate_results_path, stored_candidate_frame)
    atomic_csv(args.output_dir / "selected_target_summary.csv", summaries["selected"])
    atomic_csv(args.output_dir / "per_window_summary.csv", summaries["window"])
    atomic_csv(args.output_dir / "per_path_summary.csv", summaries["path"])
    atomic_csv(args.output_dir / "scale_summary.csv", summaries["scale"])
    atomic_csv(args.output_dir / "candidate_method_summary.csv", summaries["method"])
    atomic_csv(
        args.output_dir / "candidate_method_window_contribution.csv",
        summaries["method_contribution"],
    )
    stage_frame = pd.DataFrame(pipeline["stage_records"])
    atomic_csv(args.output_dir / "adaptive_stage_summary.csv", stage_frame)
    atomic_csv(args.output_dir / "rejection_reason_summary.csv", summaries["rejection"])
    atomic_csv(args.output_dir / "diversity_summary.csv", summaries["diversity"])
    generation_time = float(pipeline["generation_time_s"])
    scoring_time = float(pipeline["scoring_time_s"])
    exhaustive_candidate_count = (
        len(windows)
        * args.restarts_per_method
        * sum(
            V7_CANDIDATES_PER_METHOD_RESTART.get(str(method), 0)
            for method in v7_arguments["candidate_methods"]
        )
    )
    exhaustive_candidates_avoided = max(
        0, exhaustive_candidate_count - len(base_rows)
    ) if args.generation_policy == "adaptive" else 0
    estimated_reduction_percent = (
        100.0 * exhaustive_candidates_avoided / max(exhaustive_candidate_count, 1)
    )
    total_wall = time.perf_counter() - wall_started
    metadata = {
        "classification": "V8_TARGET_GENERATION_COMPLETE",
        "arguments": vars(args),
        "source_paths": {
            "train_prior": str(v7.resolve_project_path(args.train_prior)),
            "train_windows": str(v7.resolve_project_path(args.train_windows)),
            "split_manifest": str(v7.resolve_project_path(args.split_manifest)),
            "robot_urdf": str(v7.resolve_project_path(args.robot_urdf)),
        },
        "schema": {key: list(value.shape) for key, value in arrays.items()},
        "counts": {
            "paths_processed": len({window.path_name for window in windows}),
            "windows_processed": len(windows),
            "candidates_evaluated": len(base_rows) + len(scaled_rows),
            "base_candidates_evaluated": len(base_rows),
            "hard_safe_base_candidates": sum(int(row["hard_safe"]) for row in base_rows),
            "cartesian_improving_base_candidates": sum(int(row["cartesian_improving"]) for row in base_rows),
            "negative_delta_score_base_candidates": sum(int(row["negative_delta_score"]) for row in base_rows),
            "valid_base_candidates": sum(int(row["valid_target"]) for row in base_rows),
            "retained_diverse_base_targets": sum(int(row["retained"]) for row in base_rows),
            "valid_scaled_targets": sum(int(row["valid_target"]) for row in scaled_rows),
            "retained_final_targets": len(selected_rows),
            "zero_target_windows": int(summaries["window"]["zero_valid_target"].sum()),
            "exhaustive_base_candidate_count_estimate": exhaustive_candidate_count,
            "exhaustive_candidates_avoided": exhaustive_candidates_avoided,
            "estimated_candidate_reduction_percent": estimated_reduction_percent,
            "fallback_windows_attempted": pipeline["fallback_windows_attempted"],
            "fallback_windows_rescued": pipeline["fallback_windows_rescued"],
            "candidate_results_rows_written": (
                0 if stored_candidate_frame is None else len(stored_candidate_frame)
            ),
        },
        "generation_policy": args.generation_policy,
        "adaptive_stage_summary": stage_frame.to_dict(orient="records"),
        "candidate_results_mode": args.candidate_results_mode,
        "metric_definitions": {
            "implementation": "generate_diffusion_v7_cost_improving_residual_targets",
            "valid_target_rule": (
                "all v7 acceptance reasons pass AND hard_safe AND execution-prefix "
                "Cartesian improvement > 0 AND robot-aware delta_score < 0"
            ),
            "robot_aware_score_weights": vars(v7.ScoreWeights()),
            "robot_aware_score_floors": vars(v7.MetricFloors()),
            "prior_robot_aware_score": "zero reference under v7 relative delta_score",
        },
        "robot_conventions": {
            "robot": "ROKAE xMateCR7",
            "joint_names": list(DEFAULT_JOINT_NAMES),
            "end_effector_frame": DEFAULT_EE_LINK,
            "fk_calls": ["robot.update_cfg(cfg)", "robot.get_transform(frame_to='xMateCR7_link6')"],
            "hard_joint_limit_tolerance_rad": HARD_JOINT_LIMIT_TOLERANCE_RAD,
            "joint_limit_safety_margin_rad": v7_arguments["joint_limit_safety_margin"],
        },
        "diversity": {
            "joint_residual_scale": joint_std.tolist(),
            "scale_estimator": "1.4826 * MAD with conventional-std then 1.0 fallback",
            "prefix_steps": args.execution_horizon,
            "minimum_prefix_normalized_rms": args.min_prefix_diversity_rms,
            "minimum_full_normalized_rms": args.min_full_diversity_rms,
        },
        "determinism": {
            "policy": "SHA-256 stable seeds and IDs from global seed/path/window/method/restart/scale",
            "candidate_seed": args.candidate_seed,
            "canonical_order": "path_name, window_start, artifact stage, generation stage, stable candidate/target ID",
            "worker_completion_order_restored": True,
            "adaptive_method_streams_independent": args.generation_policy == "adaptive",
        },
        "multiprocessing": {
            "start_method": "serial" if args.num_workers == 1 else "spawn",
            "logical_cpu_count": logical_cpus,
            "requested_workers": args.num_workers,
            "active_workers": active_workers,
            "robot_initialization": "one reusable xMateCR7 model per worker",
        },
        "timing": {
            "cumulative_candidate_generation_time_s": generation_time,
            "cumulative_fk_scoring_time_s": scoring_time,
            "total_wall_time_s": total_wall,
        },
        "scale_distribution": summaries["scale"].to_dict(orient="records"),
        "candidate_method_contribution": summaries["method_contribution"].to_dict(orient="records"),
        "targets_per_window_distribution": summaries["window"]["retained_target_count"].value_counts().sort_index().to_dict(),
    }
    atomic_json(args.output_dir / "target_generation_summary.json", metadata)
    print(f"paths processed: {metadata['counts']['paths_processed']}")
    print(f"windows processed: {metadata['counts']['windows_processed']}")
    print(f"candidates evaluated: {metadata['counts']['candidates_evaluated']}")
    print(f"valid base candidates: {metadata['counts']['valid_base_candidates']}")
    print(f"retained diverse base targets: {metadata['counts']['retained_diverse_base_targets']}")
    print(f"valid scaled targets: {metadata['counts']['valid_scaled_targets']}")
    print(f"retained final targets: {metadata['counts']['retained_final_targets']}")
    print(f"exhaustive candidate count avoided: {exhaustive_candidates_avoided}")
    print(f"estimated candidate reduction: {estimated_reduction_percent:.3f}%")
    print(f"fallback windows attempted: {pipeline['fallback_windows_attempted']}")
    print(f"fallback windows rescued: {pipeline['fallback_windows_rescued']}")
    print(f"candidate-results output mode: {args.candidate_results_mode}")
    print(f"cumulative candidate generation time: {generation_time:.3f} s")
    print(f"cumulative FK/scoring time: {scoring_time:.3f} s")
    print(f"total wall time: {total_wall:.3f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
