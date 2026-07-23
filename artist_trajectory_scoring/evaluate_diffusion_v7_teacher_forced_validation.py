#!/usr/bin/env python3
"""Teacher-forced validation for v7 cost-improving residual diffusion.

Every unique validation window is evaluated once from its stored strong prior.
Diffusion samples start from Gaussian noise and traverse the complete DDIM
reverse process; retained targets are used only for the separately reported
oracle ceiling.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import multiprocessing
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import evaluate_diffusion_v6_teacher_forced_validation as v6_evaluator
import generate_diffusion_v7_cost_improving_residual_targets as target_generator
import train_conditional_diffusion_trajectory_v6_strong_prior_residual_unet as v6_trainer
from generate_ik_seed_path import (
    DEFAULT_EE_LINK,
    DEFAULT_JOINT_LIMIT_SAFETY_MARGIN_RAD,
    DEFAULT_JOINT_NAMES,
    DEFAULT_URDF_PATH,
    HARD_JOINT_LIMIT_TOLERANCE_RAD,
    get_joint_bounds,
    load_robot,
)


HORIZON = 32
CONDITION_DIM = 38
TARGET_DIM = 6
EXPECTED_EXECUTION_HORIZON = 8
MAXIMUM_JOINT_STEP_RAD = 0.20
FLOAT_RTOL = 1.0e-5
FLOAT_ATOL = 1.0e-6
# The dataset builder normalizes in float64, then independently stores the raw
# conditions, statistics, and normalized conditions as float32. Reconstructing
# from those rounded values needs a small, normalization-specific tolerance.
NORMALIZATION_RECONSTRUCTION_ATOL = 2.0e-5
MEANINGFUL_GAIN_THRESHOLD_M = 0.000075
MEANINGFUL_IMPROVED_WINDOW_FRACTION = 0.50
SYSTEMATIC_HARD_UNSAFE_SAMPLE_RATE_THRESHOLD = 0.05

# This exactly matches candidate_result_row(...)["hard_safe"] in the v7 target
# generator. Its remaining acceptance reasons are candidate-selection
# constraints, not evidence that a trajectory is physically unsafe.
TARGET_HARD_GATE_REASONS = frozenset(
    {
        "nonfinite_values",
        "hard_joint_limit_violation",
        "maximum_joint_step_gate",
    }
)
TARGET_REJECTION_REASONS = (
    "nonfinite_values",
    "hard_joint_limit_violation",
    "maximum_joint_step_gate",
    "nonfinite_boundary",
    "ik_branch_jump",
    "catastrophic_full_window_degradation",
    "insufficient_cartesian_improvement",
    "prefix_acceleration_cost_degradation",
    "prefix_jerk_cost_degradation",
    "boundary_step_max_abs_rad_degradation",
    "boundary_acceleration_discontinuity_degradation",
)

DEFAULT_DATASET_DIR = Path(
    "data/cartesian_expert_dataset_v3/"
    "diffusion_v7_cost_improving_training_dataset_100paths"
)
DEFAULT_CHECKPOINT_DIR = Path(
    "models/diffusion_v7_cost_improving_residual_unet_100paths_seed42_full"
)
DEFAULT_OUTPUT_DIR = Path("results/diffusion_v7_teacher_forced_validation_seed42")

OUTPUT_FILES = (
    "evaluation_configuration.json",
    "evaluation_summary.json",
    "checkpoint_summary.csv",
    "best_of_k_summary.csv",
    "alpha_summary.csv",
    "per_window_results.csv",
    "per_path_summary.csv",
    "per_sample_results.csv",
    "safety_and_fallback_summary.csv",
    "runtime_summary.csv",
)
PLOT_FILES = (
    "prior_vs_diffusion_cartesian_error_scatter.png",
    "cartesian_improvement_histogram.png",
    "best_of_k_gain_curve.png",
    "alpha_sweep_cartesian_improvement.png",
    "safety_pass_and_fallback_rate.png",
    "raw_vs_ema_comparison.png",
    "oracle_gap.png",
    "example_cartesian_trajectories.png",
    "example_error_over_time.png",
    "example_joint_trajectories.png",
)


@dataclass(frozen=True)
class ValidationWindow:
    index: int
    path_name: str
    path_index: int
    window_start: int
    condition: np.ndarray
    condition_norm: np.ndarray
    desired: np.ndarray
    prior_q: np.ndarray
    prior_ee: np.ndarray
    execution_horizon: int
    retained_candidates: np.ndarray
    retained_zero_flags: np.ndarray
    retained_methods: Tuple[str, ...]


@dataclass
class CheckpointRuntime:
    label: str
    method_prefix: str
    path: Path
    diagnostic_only: bool
    checkpoint: Dict[str, Any]
    model: torch.nn.Module
    schedule: Any
    condition_mean: np.ndarray
    condition_std: np.ndarray
    residual_mean: np.ndarray
    residual_std: np.ndarray
    selected_state: str
    epoch: int
    report: Dict[str, Any]


@dataclass(frozen=True)
class PlotArtifact:
    path_name: str
    window_start: int
    desired: np.ndarray
    prior_q: np.ndarray
    candidate_q: np.ndarray
    prior_ee: np.ndarray
    candidate_ee: np.ndarray
    execution_horizon: int


@dataclass(frozen=True)
class CandidateDecision:
    acceptance_reasons: Tuple[str, ...]
    hard_safety_reasons: Tuple[str, ...]
    improvement_m: float
    relative_improvement: float
    delta_score: float

    @property
    def hard_safe(self) -> bool:
        return not self.hard_safety_reasons

    @property
    def selectable(self) -> bool:
        return not self.acceptance_reasons

    @property
    def cartesian_improving(self) -> bool:
        return self.improvement_m > 0.0

    @property
    def sample_category(self) -> str:
        if not self.hard_safe:
            return "hard_unsafe"
        if not self.cartesian_improving:
            return "safe_but_nonimproving"
        return "safe_improving_but_not_selected"


@dataclass(frozen=True)
class CandidateEvaluationTask:
    candidate_id: str
    context: target_generator.WindowContext
    candidate_q: np.ndarray
    execution_horizon: int
    prior_metrics: Dict[str, Any]


@dataclass(frozen=True)
class CandidateEvaluationResult:
    candidate_id: str
    metrics: Dict[str, Any]
    decision: CandidateDecision
    evaluation_time_s: float


_WORKER_ROBOT: Optional[target_generator.RobotContext] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate v7 diffusion on unique teacher-forced validation windows."
    )
    parser.add_argument("--dataset_dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument(
        "--best_raw_checkpoint", type=Path,
        default=DEFAULT_CHECKPOINT_DIR / "best_raw_checkpoint.pt",
    )
    parser.add_argument(
        "--best_ema_checkpoint", type=Path,
        default=DEFAULT_CHECKPOINT_DIR / "best_ema_checkpoint.pt",
    )
    parser.add_argument(
        "--last_checkpoint", type=Path,
        default=DEFAULT_CHECKPOINT_DIR / "last_checkpoint.pt",
    )
    parser.add_argument("--v6_checkpoint", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ddim_steps", type=int, default=50)
    parser.add_argument("--k_values", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--num_cpu_workers", type=int, default=1)
    parser.add_argument("--gpu_batch_size", type=int, default=8)
    parser.add_argument(
        "--alphas", type=float, nargs="+", default=[0.25, 0.5, 0.75, 1.0]
    )
    parser.add_argument(
        "--primary_alpha",
        type=float,
        default=None,
        help=(
            "Alpha used for scientific ranking and primary plots. Defaults to "
            "1.0 when present, otherwise the largest requested alpha."
        ),
    )
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--max_windows", type=int, default=None)
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument(
        "--save_per_sample_results",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--plot_example_count", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--robot_urdf", type=Path, default=Path(DEFAULT_URDF_PATH))
    return parser.parse_args()


def validate_cli(args: argparse.Namespace) -> None:
    if args.ddim_steps <= 0:
        raise ValueError("--ddim_steps must be positive")
    if args.eta < 0.0 or not np.isfinite(args.eta):
        raise ValueError("--eta must be finite and non-negative")
    if not args.k_values or any(value <= 0 for value in args.k_values):
        raise ValueError("--k_values must contain positive integers")
    if 1 not in args.k_values:
        raise ValueError("--k_values must include 1 for the required K=1 result")
    if len(set(args.k_values)) != len(args.k_values):
        raise ValueError("--k_values cannot contain duplicates")
    if args.num_cpu_workers < 1:
        raise ValueError("--num_cpu_workers must be at least 1")
    if args.gpu_batch_size < 1:
        raise ValueError("--gpu_batch_size must be at least 1")
    if not args.alphas or any(
        value <= 0.0 or not np.isfinite(value) for value in args.alphas
    ):
        raise ValueError("--alphas must contain positive finite values")
    if len(set(args.alphas)) != len(args.alphas):
        raise ValueError("--alphas cannot contain duplicates")
    if args.primary_alpha is None:
        args.primary_alpha = (
            1.0
            if any(np.isclose(value, 1.0) for value in args.alphas)
            else max(args.alphas)
        )
    if args.primary_alpha <= 0.0 or not np.isfinite(args.primary_alpha):
        raise ValueError("--primary_alpha must be positive and finite")
    matching_alphas = [
        value for value in args.alphas
        if np.isclose(value, args.primary_alpha)
    ]
    if not matching_alphas:
        raise ValueError("--primary_alpha must be included in --alphas")
    args.primary_alpha = float(matching_alphas[0])
    if args.max_windows is not None and args.max_windows <= 0:
        raise ValueError("--max_windows must be positive")
    if args.plot_example_count < 0:
        raise ValueError("--plot_example_count must be non-negative")


def apply_smoke_mode(args: argparse.Namespace) -> None:
    if not args.smoke_test:
        return
    args.ddim_steps = 10
    args.k_values = [1, 2]
    args.alphas = [1.0]
    args.primary_alpha = 1.0
    args.max_windows = 5
    args.last_checkpoint = None
    args.v6_checkpoint = None
    args.output_dir = args.output_dir / "smoke_test"


def set_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(value)


def decode_strings(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [
            item.decode("utf-8", errors="strict")
            if isinstance(item, bytes)
            else str(item)
            for item in np.asarray(values).reshape(-1)
        ],
        dtype=str,
    )


def load_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def load_normalization(path: Path) -> Dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        values = {key: np.asarray(archive[key]) for key in archive.files}
    required = (
        "condition_mean", "condition_std", "residual_mean", "residual_std",
        "condition_feature_names", "condition_feature_layout",
        "condition_dim", "target_dim", "horizon", "validation_excluded",
    )
    missing = [key for key in required if key not in values]
    if missing:
        raise KeyError(f"{path} is missing normalization fields: {missing}")
    for key in ("condition_mean", "condition_std", "residual_mean", "residual_std"):
        if not np.all(np.isfinite(values[key])):
            raise ValueError(f"{path}/{key} contains NaN or infinity")
    if np.any(values["condition_std"] <= 0.0) or np.any(values["residual_std"] <= 0.0):
        raise ValueError("Normalization standard deviations must be positive")
    for key, expected in (
        ("condition_dim", CONDITION_DIM), ("target_dim", TARGET_DIM),
        ("horizon", HORIZON),
    ):
        if int(np.asarray(values[key]).item()) != expected:
            raise ValueError(f"Normalization {key} is incompatible")
    if not bool(np.asarray(values["validation_excluded"]).item()):
        raise ValueError("Normalization does not confirm validation exclusion")
    return values


def maximum_discrepancy(reference: np.ndarray, values: np.ndarray) -> float:
    return float(
        np.max(
            np.abs(
                np.asarray(reference, dtype=np.float64)
                - np.asarray(values, dtype=np.float64)
            )
        )
    )


def load_unique_validation_windows(
    path: Path,
    normalization: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> Tuple[List[ValidationWindow], Dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=True) as archive:
        required = (
            "condition", "condition_features_norm", "desired_path_window",
            "prior_q_window", "prior_ee_window", "candidate_q_window",
            "residual_q_window", "path_names", "path_indices", "window_starts",
            "candidate_methods", "is_zero_residual", "execution_horizon",
            "unique_window_id", "targets_in_window",
        )
        missing = [key for key in required if key not in archive.files]
        if missing:
            raise KeyError(f"{path} is missing required arrays: {missing}")
        data = {key: np.asarray(archive[key]) for key in required}
    data["path_names"] = decode_strings(data["path_names"])
    data["candidate_methods"] = decode_strings(data["candidate_methods"])
    count = len(data["path_names"])
    expected = {
        "condition": (count, HORIZON, CONDITION_DIM),
        "condition_features_norm": (count, HORIZON, CONDITION_DIM),
        "desired_path_window": (count, HORIZON, 3),
        "prior_q_window": (count, HORIZON, TARGET_DIM),
        "prior_ee_window": (count, HORIZON, 3),
        "candidate_q_window": (count, HORIZON, TARGET_DIM),
        "residual_q_window": (count, HORIZON, TARGET_DIM),
    }
    for key, shape in expected.items():
        if data[key].shape != shape:
            raise ValueError(f"{path}/{key} has shape {data[key].shape}; expected {shape}")
        if not np.all(np.isfinite(data[key])):
            raise ValueError(f"{path}/{key} contains NaN or infinity")
    reconstructed_norm = (
        data["condition"] - np.asarray(normalization["condition_mean"])
    ) / np.asarray(normalization["condition_std"])
    if not np.allclose(
        reconstructed_norm, data["condition_features_norm"],
        rtol=FLOAT_RTOL, atol=NORMALIZATION_RECONSTRUCTION_ATOL,
    ):
        difference = maximum_discrepancy(
            reconstructed_norm, data["condition_features_norm"]
        )
        raise ValueError(
            "Stored normalized conditions do not match normalization.npz; "
            f"maximum_difference={difference:.9g}"
        )
    if not np.allclose(
        data["prior_q_window"] + data["residual_q_window"],
        data["candidate_q_window"], rtol=FLOAT_RTOL, atol=FLOAT_ATOL,
    ):
        raise ValueError("Validation candidate_q_window is not prior + residual")

    validation_names = tuple(sorted(str(name) for name in manifest.get("validation_path_names", ())))
    if len(validation_names) != 20:
        raise ValueError(f"split_manifest.json must contain 20 validation paths, found {len(validation_names)}")
    if set(data["path_names"].tolist()) != set(validation_names):
        raise ValueError("Validation archive path set differs from split_manifest.json")
    if metadata.get("classification") != "READY_FOR_V7_TRAINING":
        raise ValueError("dataset_metadata.json is not READY_FOR_V7_TRAINING")
    if tuple(metadata.get("condition_feature_layout", ())) != tuple(
        v6_trainer.EXPECTED_CONDITION_LAYOUT
    ):
        raise ValueError("Dataset condition layout differs from the established v6 layout")
    dataset_features = tuple(str(value) for value in metadata.get("condition_feature_names", ()))
    normalization_features = tuple(
        str(value) for value in normalization["condition_feature_names"].tolist()
    )
    if dataset_features != normalization_features or len(dataset_features) != CONDITION_DIM:
        raise ValueError("Dataset and normalization feature ordering differ")

    groups: Dict[Tuple[str, int], List[int]] = {}
    for row, (name, start) in enumerate(zip(data["path_names"], data["window_starts"])):
        groups.setdefault((str(name), int(start)), []).append(row)
    maximum_duplicate_difference = 0.0
    windows: List[ValidationWindow] = []
    repeated_keys = (
        "condition", "condition_features_norm", "desired_path_window",
        "prior_q_window", "prior_ee_window",
    )
    for window_index, key in enumerate(sorted(groups)):
        rows = np.asarray(groups[key], dtype=np.int64)
        reference = int(rows[0])
        for row_value in rows[1:]:
            row = int(row_value)
            for array_key in repeated_keys:
                difference = maximum_discrepancy(data[array_key][reference], data[array_key][row])
                maximum_duplicate_difference = max(maximum_duplicate_difference, difference)
                if not np.allclose(
                    data[array_key][reference], data[array_key][row],
                    rtol=FLOAT_RTOL, atol=FLOAT_ATOL,
                ):
                    raise ValueError(
                        f"Repeated validation condition differs for {key}, "
                        f"field={array_key}, max_difference={difference:.9g}"
                    )
            if int(data["execution_horizon"][reference]) != int(data["execution_horizon"][row]):
                raise ValueError(f"Repeated execution_horizon differs for {key}")
        execution_horizon = int(data["execution_horizon"][reference])
        if execution_horizon != EXPECTED_EXECUTION_HORIZON:
            raise ValueError(
                f"{key} execution_horizon={execution_horizon}; expected "
                f"{EXPECTED_EXECUTION_HORIZON}"
            )
        if np.any(np.asarray(data["targets_in_window"])[rows] != len(rows)):
            raise ValueError(f"targets_in_window is inconsistent for {key}")
        windows.append(
            ValidationWindow(
                index=window_index,
                path_name=key[0],
                path_index=int(data["path_indices"][reference]),
                window_start=key[1],
                condition=np.asarray(data["condition"][reference], dtype=np.float32),
                condition_norm=np.asarray(
                    data["condition_features_norm"][reference], dtype=np.float32
                ),
                desired=np.asarray(data["desired_path_window"][reference], dtype=np.float64),
                prior_q=np.asarray(data["prior_q_window"][reference], dtype=np.float64),
                prior_ee=np.asarray(data["prior_ee_window"][reference], dtype=np.float64),
                execution_horizon=execution_horizon,
                retained_candidates=np.asarray(data["candidate_q_window"][rows], dtype=np.float64),
                retained_zero_flags=np.asarray(data["is_zero_residual"][rows], dtype=bool),
                retained_methods=tuple(str(value) for value in data["candidate_methods"][rows]),
            )
        )
    expected_windows = len(validation_names) * 18
    if len(windows) != expected_windows:
        raise ValueError(
            f"Expected {expected_windows} unique validation windows, found {len(windows)}"
        )
    return windows, {
        "target_row_count": count,
        "unique_window_count": len(windows),
        "unique_path_count": len(validation_names),
        "maximum_duplicate_conditioning_discrepancy": maximum_duplicate_difference,
        "condition_feature_names": list(dataset_features),
    }


def reconstruct_prior_timelines(
    windows: Sequence[ValidationWindow],
) -> Dict[str, np.ndarray]:
    timelines: Dict[str, np.ndarray] = {}
    filled: Dict[str, np.ndarray] = {}
    for window in windows:
        timeline = timelines.setdefault(
            window.path_name, np.full((100, TARGET_DIM), np.nan, dtype=np.float64)
        )
        mask = filled.setdefault(window.path_name, np.zeros(100, dtype=bool))
        start = window.window_start
        stop = start + HORIZON
        overlap = mask[start:stop]
        if np.any(overlap) and not np.allclose(
            timeline[start:stop][overlap], window.prior_q[overlap],
            rtol=FLOAT_RTOL, atol=FLOAT_ATOL,
        ):
            raise ValueError(f"Overlapping prior windows differ for {window.path_name}")
        timeline[start:stop] = window.prior_q
        mask[start:stop] = True
    for name, values in timelines.items():
        if not np.all(np.isfinite(values)):
            missing = np.flatnonzero(~np.all(np.isfinite(values), axis=1)).tolist()
            raise ValueError(f"Prior timeline for {name} is incomplete: {missing}")
    return timelines


def make_robot_context(robot_urdf: Path) -> target_generator.RobotContext:
    joint_names = tuple(str(value) for value in DEFAULT_JOINT_NAMES)
    if joint_names != tuple(f"joint{index}" for index in range(1, 7)):
        raise ValueError(f"Unexpected active-joint order: {joint_names}")
    if DEFAULT_EE_LINK != "xMateCR7_link6":
        raise ValueError(f"Unexpected end-effector frame: {DEFAULT_EE_LINK}")
    urdf = target_generator.resolve_project_path(robot_urdf)
    robot = load_robot(urdf)
    bounds = get_joint_bounds(robot, joint_names, -np.pi, np.pi)
    lower = np.asarray([value[0] for value in bounds], dtype=np.float64)
    upper = np.asarray([value[1] for value in bounds], dtype=np.float64)
    return target_generator.RobotContext(
        robot=robot,
        joint_names=joint_names,
        ee_link=DEFAULT_EE_LINK,
        lower=lower,
        upper=upper,
    )


def make_window_context(
    window: ValidationWindow,
    timelines: Mapping[str, np.ndarray],
) -> target_generator.WindowContext:
    full_prior = timelines[window.path_name]
    start = window.window_start
    tail_index = start + window.execution_horizon
    return target_generator.WindowContext(
        path_name=window.path_name,
        path_index=window.path_index,
        window_start=start,
        prior_q=window.prior_q,
        desired=window.desired,
        prior_ee=window.prior_ee,
        previous_q=None if start == 0 else full_prior[start - 1],
        previous_previous_q=None if start < 2 else full_prior[start - 2],
        tail_q=full_prior[tail_index],
        tail_next_q=(
            None if tail_index + 1 >= len(full_prior) else full_prior[tail_index + 1]
        ),
    )


def target_acceptance_arguments() -> argparse.Namespace:
    return argparse.Namespace(
        max_joint_step_gate=MAXIMUM_JOINT_STEP_RAD,
        min_cartesian_improvement_m=1.0e-5,
        min_cartesian_improvement_fraction=0.005,
        smoothness_relative_tolerance=0.10,
        boundary_absolute_tolerance=0.01,
    )


def scalar_metrics(prefix: str, metrics: Mapping[str, Any]) -> Dict[str, Any]:
    return target_generator.flatten_metrics(prefix, metrics)


def candidate_decision(
    metrics: Mapping[str, Any],
    prior: Mapping[str, Any],
    acceptance_args: argparse.Namespace,
    weights: target_generator.ScoreWeights,
    floors: target_generator.MetricFloors,
) -> CandidateDecision:
    """Apply the target generator's gates and robot-aware score verbatim."""
    reasons, improvement, relative = target_generator.acceptance_reasons(
        metrics, prior, acceptance_args
    )
    hard_reasons = tuple(
        reason for reason in reasons if reason in TARGET_HARD_GATE_REASONS
    )
    score = (
        target_generator.delta_score(metrics, prior, weights, floors)
        if bool(metrics.get("finite", False))
        else math.inf
    )
    return CandidateDecision(
        acceptance_reasons=tuple(reasons),
        hard_safety_reasons=hard_reasons,
        improvement_m=float(improvement),
        relative_improvement=float(relative),
        delta_score=float(score),
    )


def hard_safety_reasons(
    metrics: Mapping[str, Any], prior: Mapping[str, Any]
) -> Tuple[str, ...]:
    decision = candidate_decision(
        metrics,
        prior,
        target_acceptance_arguments(),
        target_generator.ScoreWeights(),
        target_generator.MetricFloors(),
    )
    return decision.hard_safety_reasons


def assert_cpu_pool_payload(value: Any, location: str = "payload") -> None:
    """Reject Torch objects before process-pool submission."""
    if torch.is_tensor(value):
        device = getattr(value, "device", "unknown")
        raise AssertionError(
            f"Process-pool {location} contains a Torch tensor on {device}; "
            "workers accept CPU NumPy data only"
        )
    if isinstance(value, Mapping):
        for key, item in value.items():
            assert_cpu_pool_payload(item, f"{location}.{key}")
        return
    if isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            assert_cpu_pool_payload(item, f"{location}[{index}]")
        return
    if hasattr(value, "__dataclass_fields__"):
        for key, item in vars(value).items():
            assert_cpu_pool_payload(item, f"{location}.{key}")


def initialize_candidate_worker(robot_urdf: str) -> None:
    """Construct one reusable robot per spawned CPU worker."""
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


def evaluate_candidate_task(
    task: CandidateEvaluationTask,
    robot: target_generator.RobotContext,
) -> CandidateEvaluationResult:
    """Shared serial/worker implementation for FK, costs, gates, and score."""
    started = time.perf_counter()
    metrics = evaluate_metrics(
        robot,
        task.context,
        np.asarray(task.candidate_q, dtype=np.float64),
        task.execution_horizon,
    )
    decision = candidate_decision(
        metrics,
        task.prior_metrics,
        target_acceptance_arguments(),
        target_generator.ScoreWeights(),
        target_generator.MetricFloors(),
    )
    return CandidateEvaluationResult(
        candidate_id=task.candidate_id,
        metrics=metrics,
        decision=decision,
        evaluation_time_s=time.perf_counter() - started,
    )


def candidate_worker_entry(task: CandidateEvaluationTask) -> CandidateEvaluationResult:
    if _WORKER_ROBOT is None:
        raise RuntimeError("Candidate worker robot was not initialized")
    assert_cpu_pool_payload(task)
    return evaluate_candidate_task(task, _WORKER_ROBOT)


def evaluate_candidate_tasks(
    tasks: Sequence[CandidateEvaluationTask],
    robot: target_generator.RobotContext,
    executor: Optional[concurrent.futures.ProcessPoolExecutor],
) -> List[CandidateEvaluationResult]:
    candidate_ids = [task.candidate_id for task in tasks]
    if len(candidate_ids) != len(set(candidate_ids)):
        raise AssertionError("Submitted candidate_id values are not unique")
    for task in tasks:
        assert_cpu_pool_payload(task)

    if executor is None:
        unordered_results = [
            evaluate_candidate_task(task, robot) for task in tasks
        ]
    else:
        future_to_id = {
            executor.submit(candidate_worker_entry, task): task.candidate_id
            for task in tasks
        }
        unordered_results = []
        for future in concurrent.futures.as_completed(future_to_id):
            expected_id = future_to_id[future]
            result = future.result()
            if result.candidate_id != expected_id:
                raise AssertionError(
                    f"Worker returned candidate_id={result.candidate_id!r}; "
                    f"expected {expected_id!r}"
                )
            unordered_results.append(result)

    results_by_id: Dict[str, CandidateEvaluationResult] = {}
    for result in unordered_results:
        if result.candidate_id in results_by_id:
            raise AssertionError(
                f"candidate_id={result.candidate_id!r} was returned more than once"
            )
        results_by_id[result.candidate_id] = result
    if set(results_by_id) != set(candidate_ids):
        missing = sorted(set(candidate_ids) - set(results_by_id))
        unexpected = sorted(set(results_by_id) - set(candidate_ids))
        raise AssertionError(
            f"Candidate result mismatch: missing={missing}, unexpected={unexpected}"
        )

    ordered_results = [results_by_id[candidate_id] for candidate_id in candidate_ids]
    if [result.candidate_id for result in ordered_results] != candidate_ids:
        raise AssertionError("Canonical candidate ordering was not restored")
    return ordered_results


def stable_window_seed(
    global_seed: int,
    checkpoint_identity: str,
    path_name: str,
    window_start: int,
    alpha: float,
    sample_index: int,
) -> int:
    payload = json.dumps(
        [
            int(global_seed), checkpoint_identity, path_name, int(window_start),
            format(float(alpha), ".12g"), int(sample_index),
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & (
        (1 << 63) - 1
    )


def torch_load(path: Path, device: torch.device) -> Dict[str, Any]:
    return v6_evaluator.torch_load(path, device)


def normalization_from_checkpoint(
    checkpoint: Mapping[str, Any], checkpoint_path: Path
) -> Dict[str, np.ndarray]:
    embedded = checkpoint.get("normalization_statistics")
    if not isinstance(embedded, Mapping):
        raise KeyError(f"{checkpoint_path} lacks normalization_statistics")
    result: Dict[str, np.ndarray] = {}
    for key, width in (
        ("condition_mean", CONDITION_DIM), ("condition_std", CONDITION_DIM),
        ("residual_mean", TARGET_DIM), ("residual_std", TARGET_DIM),
    ):
        values = np.asarray(embedded.get(key), dtype=np.float32).reshape(-1)
        if values.shape != (width,) or not np.all(np.isfinite(values)):
            raise ValueError(f"{checkpoint_path}/{key} is incompatible")
        if key.endswith("std") and np.any(values <= 0.0):
            raise ValueError(f"{checkpoint_path}/{key} must be positive")
        result[key] = values
    return result


def state_dicts_equal(
    left: Mapping[str, Any], right: Mapping[str, Any]
) -> bool:
    if set(left) != set(right):
        return False
    return all(
        isinstance(left[key], torch.Tensor)
        and isinstance(right[key], torch.Tensor)
        and torch.equal(left[key], right[key])
        for key in left
    )


def load_checkpoint_runtime(
    *, label: str, method_prefix: str, path: Path, expected_state: Optional[str],
    diagnostic_only: bool, device: torch.device,
    dataset_normalization: Mapping[str, np.ndarray],
    dataset_features: Sequence[str], manifest: Mapping[str, Any],
    v7_checkpoint: bool,
) -> CheckpointRuntime:
    checkpoint = torch_load(path, device)
    for key, expected in (
        ("horizon", HORIZON), ("condition_dim", CONDITION_DIM),
        ("target_dim", TARGET_DIM),
    ):
        if int(checkpoint.get(key, -1)) != expected:
            raise ValueError(f"{path.name}: checkpoint {key} is incompatible")
    if checkpoint.get("prediction_target_type") != "epsilon":
        raise ValueError(f"{path.name}: prediction target must be epsilon")
    schedule_config = v6_evaluator.checkpoint_schedule(checkpoint)
    selected_state = str(checkpoint.get("selected_model_state", ""))
    if v7_checkpoint and selected_state not in ("raw", "ema"):
        raise ValueError(f"{path.name}: v7 selected_model_state is missing or invalid")
    if expected_state is not None and selected_state != expected_state:
        raise ValueError(
            f"{path.name}: selected_model_state={selected_state!r}, "
            f"expected {expected_state!r}"
        )
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, Mapping):
        raise KeyError(f"{path.name}: model_state_dict is absent")
    if selected_state == "raw":
        raw_state = checkpoint.get("raw_model_state_dict")
        if isinstance(raw_state, Mapping) and not state_dicts_equal(state, raw_state):
            raise ValueError(f"{path.name}: selected raw state differs from raw_model_state_dict")
    if selected_state == "ema":
        ema_container = checkpoint.get("ema_state_dict")
        ema_shadow = ema_container.get("shadow") if isinstance(ema_container, Mapping) else None
        if not isinstance(ema_shadow, Mapping) or not state_dicts_equal(state, ema_shadow):
            raise ValueError(f"{path.name}: selected EMA state differs from EMA shadow")

    checkpoint_normalization = normalization_from_checkpoint(checkpoint, path)
    checkpoint_features = tuple(
        str(value) for value in checkpoint.get("condition_feature_ordering", ())
    )
    checkpoint_layout = tuple(checkpoint.get("condition_feature_layout", ()))
    if v7_checkpoint and checkpoint_features != tuple(dataset_features):
        raise ValueError(f"{path.name}: v7 condition feature ordering is incompatible")
    if not v7_checkpoint and checkpoint_features and checkpoint_features != tuple(dataset_features):
        raise ValueError(f"{path.name}: condition feature ordering is incompatible")
    if checkpoint_layout and checkpoint_layout != tuple(v6_trainer.EXPECTED_CONDITION_LAYOUT):
        raise ValueError(f"{path.name}: condition feature layout is incompatible")
    if not v7_checkpoint and not checkpoint_layout:
        embedded_dataset = checkpoint.get("dataset_configuration")
        embedded_layout = (
            tuple(embedded_dataset.get("condition_feature_layout", ()))
            if isinstance(embedded_dataset, Mapping) else ()
        )
        if embedded_layout != tuple(v6_trainer.EXPECTED_CONDITION_LAYOUT):
            raise ValueError(f"{path.name}: v6 condition feature layout is incompatible")
    if v7_checkpoint:
        for key in ("condition_mean", "condition_std", "residual_mean", "residual_std"):
            dataset_values = np.asarray(dataset_normalization[key], dtype=np.float32).reshape(-1)
            if not np.allclose(
                checkpoint_normalization[key], dataset_values,
                rtol=1.0e-6, atol=1.0e-7,
            ):
                raise ValueError(f"{path.name}: v7 normalization {key} is incompatible")
        checkpoint_manifest = checkpoint.get("split_manifest")
        if not isinstance(checkpoint_manifest, Mapping):
            raise KeyError(f"{path.name}: embedded split_manifest is absent")
        if set(checkpoint_manifest.get("validation_path_names", ())) != set(
            manifest.get("validation_path_names", ())
        ):
            raise ValueError(f"{path.name}: embedded validation split is incompatible")

    model, model_config = v6_trainer.instantiate_v5_model(
        int(checkpoint["horizon"]), int(checkpoint["condition_dim"]),
        int(checkpoint["target_dim"]), schedule_config["steps"],
    )
    embedded_model = checkpoint.get("model_hyperparameters")
    if v7_checkpoint and not isinstance(embedded_model, Mapping):
        raise KeyError(f"{path.name}: v7 model_hyperparameters are absent")
    if isinstance(embedded_model, Mapping):
        expected_class = embedded_model.get("class_path")
        if expected_class is not None and expected_class != model_config.get("class_path"):
            raise ValueError(f"{path.name}: reconstructed model class differs from checkpoint")
    model.load_state_dict(cast(Mapping[str, torch.Tensor], state), strict=True)
    model.to(device).eval()
    schedule = v6_trainer.build_schedule(schedule_config["steps"], device)
    report = {
        "checkpoint": label,
        "checkpoint_path": str(path.resolve()),
        "status": "loaded",
        "selected_state": selected_state,
        "epoch": int(checkpoint.get("epoch", -1)),
        "diagnostic_only": diagnostic_only,
        "diffusion_steps": schedule_config["steps"],
        "model_class": model_config.get("class_path"),
        "v7_checkpoint": v7_checkpoint,
    }
    return CheckpointRuntime(
        label=label,
        method_prefix=method_prefix,
        path=path,
        diagnostic_only=diagnostic_only,
        checkpoint=checkpoint,
        model=model,
        schedule=schedule,
        condition_mean=checkpoint_normalization["condition_mean"],
        condition_std=checkpoint_normalization["condition_std"],
        residual_mean=checkpoint_normalization["residual_mean"],
        residual_std=checkpoint_normalization["residual_std"],
        selected_state=selected_state,
        epoch=int(checkpoint.get("epoch", -1)),
        report=report,
    )


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def evaluate_metrics(
    robot: target_generator.RobotContext,
    context: target_generator.WindowContext,
    q: np.ndarray,
    execution_horizon: int,
) -> Dict[str, Any]:
    return target_generator.trajectory_metrics(
        robot, context, q, execution_horizon,
        DEFAULT_JOINT_LIMIT_SAFETY_MARGIN_RAD,
    )


def result_row(
    *, result_id: str, window: ValidationWindow, checkpoint: str, method: str,
    alpha: float, k_value: int, diagnostic_only: bool,
    prior_metrics: Mapping[str, Any], selected_metrics: Mapping[str, Any],
    selected_sample_index: int, selected_seed: int, fallback: bool,
    selectable_found: bool, pre_fallback_safety_pass: bool,
    rejection_reasons: Sequence[str], inference_per_sample: float,
    total_time: float, oracle_improvement: float,
) -> Dict[str, Any]:
    prior_mean = float(prior_metrics["prefix_cartesian_mean_error_m"])
    selected_mean = float(selected_metrics["prefix_cartesian_mean_error_m"])
    improvement = prior_mean - selected_mean
    row: Dict[str, Any] = {
        "result_id": result_id,
        "path_name": window.path_name,
        "path_index": window.path_index,
        "window_start": window.window_start,
        "execution_horizon": window.execution_horizon,
        "checkpoint": checkpoint,
        "method": method,
        "alpha": float(alpha),
        "K": int(k_value),
        "diagnostic_only": int(diagnostic_only),
        "selected_sample_index": int(selected_sample_index),
        "selected_seed": int(selected_seed),
        "fallback": int(fallback),
        "selectable_found": int(selectable_found),
        "pre_fallback_safety_pass": int(pre_fallback_safety_pass),
        "final_safety_pass": int(
            bool(selected_metrics.get("finite", False))
            and not hard_safety_reasons(selected_metrics, prior_metrics)
        ),
        "prior_hard_safety_pass": int(
            bool(prior_metrics.get("finite", False))
            and not hard_safety_reasons(prior_metrics, prior_metrics)
        ),
        "rejection_reasons": "|".join(rejection_reasons),
        "absolute_cartesian_improvement_m": improvement,
        "absolute_cartesian_improvement_mm": 1000.0 * improvement,
        "relative_cartesian_improvement_percent": 100.0 * improvement / max(prior_mean, 1.0e-12),
        "final_output_improved": int(improvement > 0.0),
        "inference_time_per_sample_s": float(inference_per_sample),
        "total_time_per_window_s": float(total_time),
        "oracle_improvement_m": float(oracle_improvement),
        "oracle_improvement_recovered_percent": (
            100.0 * improvement / oracle_improvement
            if oracle_improvement > 1.0e-12 else 0.0
        ),
    }
    row.update(scalar_metrics("prior", prior_metrics))
    row.update(scalar_metrics("selected", selected_metrics))
    return row


def evaluate_prior_and_oracle(
    windows: Sequence[ValidationWindow],
    timelines: Mapping[str, np.ndarray],
    robot: target_generator.RobotContext,
) -> Tuple[
    Dict[Tuple[str, int], Dict[str, Any]],
    Dict[Tuple[str, int], float],
    List[Dict[str, Any]],
    Dict[str, PlotArtifact],
    Dict[Tuple[str, int], target_generator.WindowContext],
]:
    prior_metrics_by_key: Dict[Tuple[str, int], Dict[str, Any]] = {}
    oracle_improvement: Dict[Tuple[str, int], float] = {}
    rows: List[Dict[str, Any]] = []
    artifacts: Dict[str, PlotArtifact] = {}
    contexts: Dict[Tuple[str, int], target_generator.WindowContext] = {}
    weights = target_generator.ScoreWeights()
    floors = target_generator.MetricFloors()
    acceptance_args = target_acceptance_arguments()
    for window in windows:
        key = (window.path_name, window.window_start)
        context = make_window_context(window, timelines)
        contexts[key] = context
        prior_metrics = evaluate_metrics(
            robot, context, window.prior_q, window.execution_horizon
        )
        prior_decision = candidate_decision(
            prior_metrics, prior_metrics, acceptance_args, weights, floors
        )
        if not prior_decision.hard_safe:
            raise RuntimeError(
                f"Stored strong prior is unsafe for {key}: "
                f"{list(prior_decision.hard_safety_reasons)}"
            )
        computed_prior_ee = np.asarray(prior_metrics["ee"], dtype=np.float64)
        if not np.allclose(
            computed_prior_ee, window.prior_ee, rtol=1.0e-5, atol=2.0e-5
        ):
            difference = maximum_discrepancy(computed_prior_ee, window.prior_ee)
            raise ValueError(
                f"Stored prior FK differs from authoritative FK for {key}; "
                f"max_difference={difference:.9g}"
            )
        prior_metrics_by_key[key] = prior_metrics
        selectable: List[Tuple[int, Dict[str, Any], float]] = []
        oracle_decisions: List[CandidateDecision] = []
        for index, candidate in enumerate(window.retained_candidates):
            metrics = evaluate_metrics(
                robot, context, candidate, window.execution_horizon
            )
            decision = candidate_decision(
                metrics, prior_metrics, acceptance_args, weights, floors
            )
            oracle_decisions.append(decision)
            if decision.selectable:
                if not np.isfinite(decision.delta_score):
                    raise RuntimeError(
                        f"Selectable retained target has nonfinite score for {key}, "
                        f"candidate_index={index}"
                    )
                selectable.append(
                    (index, metrics, decision.delta_score)
                )
        if selectable:
            selected_index, oracle_metrics, _ = min(selectable, key=lambda item: item[2])
            if not oracle_decisions[selected_index].selectable:
                raise AssertionError("Selected oracle candidate is not selectable")
            oracle_q = window.retained_candidates[selected_index]
        else:
            selected_index = -1
            oracle_metrics = prior_metrics
            oracle_q = window.prior_q
        improvement = float(prior_metrics["prefix_cartesian_mean_error_m"]) - float(
            oracle_metrics["prefix_cartesian_mean_error_m"]
        )
        oracle_improvement[key] = improvement
        oracle_audit = {
            "oracle_retained_candidate_count": len(oracle_decisions),
            "oracle_hard_safe_candidate_count": sum(
                int(decision.hard_safe) for decision in oracle_decisions
            ),
            "oracle_safe_improving_candidate_count": sum(
                int(decision.hard_safe and decision.cartesian_improving)
                for decision in oracle_decisions
            ),
            "oracle_selectable_candidate_count": sum(
                int(decision.selectable) for decision in oracle_decisions
            ),
        }
        prior_id = f"prior::{window.index}"
        prior_row = result_row(
            result_id=prior_id, window=window, checkpoint="strong_prior",
            method="strong_prior", alpha=1.0, k_value=0,
            diagnostic_only=False, prior_metrics=prior_metrics,
            selected_metrics=prior_metrics, selected_sample_index=-1,
            selected_seed=-1, fallback=False, selectable_found=True,
            pre_fallback_safety_pass=True, rejection_reasons=(),
            inference_per_sample=0.0, total_time=0.0,
            oracle_improvement=improvement,
        )
        prior_row.update(oracle_audit)
        rows.append(prior_row)
        oracle_id = f"oracle::{window.index}"
        oracle_row = result_row(
            result_id=oracle_id, window=window, checkpoint="oracle",
            method="oracle_retained_v7_target", alpha=1.0, k_value=0,
            diagnostic_only=False, prior_metrics=prior_metrics,
            selected_metrics=oracle_metrics, selected_sample_index=selected_index,
            selected_seed=-1, fallback=selected_index < 0,
            selectable_found=selected_index >= 0,
            pre_fallback_safety_pass=True, rejection_reasons=(),
            inference_per_sample=0.0, total_time=0.0,
            oracle_improvement=improvement,
        )
        oracle_row.update(oracle_audit)
        rows.append(oracle_row)
        for result_id, candidate_q, metrics in (
            (prior_id, window.prior_q, prior_metrics),
            (oracle_id, oracle_q, oracle_metrics),
        ):
            artifacts[result_id] = PlotArtifact(
                path_name=window.path_name,
                window_start=window.window_start,
                desired=window.desired,
                prior_q=window.prior_q,
                candidate_q=np.asarray(candidate_q),
                prior_ee=np.asarray(prior_metrics["ee"]),
                candidate_ee=np.asarray(metrics["ee"]),
                execution_horizon=window.execution_horizon,
            )
    return prior_metrics_by_key, oracle_improvement, rows, artifacts, contexts


def method_name(checkpoint: CheckpointRuntime, k_value: int) -> str:
    if checkpoint.method_prefix == "v7_last_checkpoint":
        return "v7_last_checkpoint"
    if checkpoint.method_prefix == "v6_checkpoint":
        return "v6_checkpoint_k1" if k_value == 1 else "v6_checkpoint_best_of_k"
    suffix = "k1" if k_value == 1 else "best_of_k"
    return f"{checkpoint.method_prefix}_{suffix}"


def evaluate_checkpoint(
    checkpoint: CheckpointRuntime,
    windows: Sequence[ValidationWindow],
    contexts: Mapping[Tuple[str, int], target_generator.WindowContext],
    prior_metrics_by_key: Mapping[Tuple[str, int], Dict[str, Any]],
    oracle_improvements: Mapping[Tuple[str, int], float],
    robot: target_generator.RobotContext,
    args: argparse.Namespace,
    device: torch.device,
    executor: Optional[concurrent.futures.ProcessPoolExecutor],
) -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    Dict[str, PlotArtifact],
    Dict[str, float],
]:
    selected_rows: List[Dict[str, Any]] = []
    sample_rows: List[Dict[str, Any]] = []
    artifacts: Dict[str, PlotArtifact] = {}
    max_k = max(args.k_values)
    generated_candidate_count = 0
    cumulative_gpu_sampling_time = 0.0
    cumulative_cpu_scoring_time = 0.0
    cumulative_cpu_scoring_wall_time = 0.0
    checkpoint_identity = f"{checkpoint.path.resolve()}::{checkpoint.selected_state}"
    for window in windows:
        key = (window.path_name, window.window_start)
        context = contexts[key]
        prior_metrics = prior_metrics_by_key[key]
        condition_norm = (
            window.condition.astype(np.float32)
            - checkpoint.condition_mean.reshape(1, CONDITION_DIM)
        ) / checkpoint.condition_std.reshape(1, CONDITION_DIM)
        for alpha in args.alphas:
            window_started = time.perf_counter()
            seeds = [
                stable_window_seed(
                    args.seed, checkpoint_identity, window.path_name,
                    window.window_start, alpha, sample_index,
                )
                for sample_index in range(max_k)
            ]
            normalized_batches: List[np.ndarray] = []
            sampling_time = 0.0
            for batch_start in range(0, max_k, args.gpu_batch_size):
                batch_end = min(batch_start + args.gpu_batch_size, max_k)
                batch_size = batch_end - batch_start
                repeated_condition = np.repeat(
                    condition_norm[None, :, :], batch_size, axis=0
                )
                synchronize(device)
                sampling_started = time.perf_counter()
                sampled_batch = v6_evaluator.sample_batch(
                    checkpoint.model,
                    repeated_condition,
                    seeds[batch_start:batch_end],
                    checkpoint.schedule,
                    args.ddim_steps,
                    "ddim",
                    args.eta,
                    device,
                )
                synchronize(device)
                sampling_time += time.perf_counter() - sampling_started
                normalized_batches.append(np.asarray(sampled_batch))
            normalized = np.concatenate(normalized_batches, axis=0)
            if normalized.shape[0] != max_k:
                raise AssertionError(
                    f"GPU batching returned {normalized.shape[0]} samples; expected {max_k}"
                )
            time_per_sample = sampling_time / max_k
            physical = (
                normalized * checkpoint.residual_std.reshape(1, 1, TARGET_DIM)
                + checkpoint.residual_mean.reshape(1, 1, TARGET_DIM)
            )
            candidates = np.asarray(
                window.prior_q[None, :, :] + float(alpha) * physical,
                dtype=np.float64,
            )
            candidate_ids = [
                (
                    f"{checkpoint.label}::{checkpoint.selected_state}::"
                    f"{window.path_name}::{window.window_start}::"
                    f"a={float(alpha):.12g}::sample={sample_index}"
                )
                for sample_index in range(max_k)
            ]
            tasks = [
                CandidateEvaluationTask(
                    candidate_id=candidate_ids[sample_index],
                    context=context,
                    candidate_q=candidates[sample_index].copy(),
                    execution_horizon=window.execution_horizon,
                    prior_metrics=dict(prior_metrics),
                )
                for sample_index in range(max_k)
            ]
            cpu_scoring_started = time.perf_counter()
            candidate_results = evaluate_candidate_tasks(tasks, robot, executor)
            cpu_scoring_wall_time = time.perf_counter() - cpu_scoring_started
            generated_candidate_count += max_k
            cumulative_gpu_sampling_time += sampling_time
            cumulative_cpu_scoring_time += sum(
                result.evaluation_time_s for result in candidate_results
            )
            cumulative_cpu_scoring_wall_time += cpu_scoring_wall_time
            evaluated: List[Dict[str, Any]] = []
            decisions: List[CandidateDecision] = []
            all_reasons: List[List[str]] = []
            all_safety_reasons: List[List[str]] = []
            scores: List[float] = []
            window_sample_rows: List[Dict[str, Any]] = []
            for sample_index, candidate_result in enumerate(candidate_results):
                if candidate_result.candidate_id != candidate_ids[sample_index]:
                    raise AssertionError("Candidate result order is not deterministic")
                metrics = candidate_result.metrics
                decision = candidate_result.decision
                evaluated.append(metrics)
                decisions.append(decision)
                all_reasons.append(list(decision.acceptance_reasons))
                all_safety_reasons.append(list(decision.hard_safety_reasons))
                scores.append(decision.delta_score)
                sample_row: Dict[str, Any] = {
                    "checkpoint": checkpoint.label,
                    "checkpoint_path": str(checkpoint.path.resolve()),
                    "checkpoint_epoch": checkpoint.epoch,
                    "selected_state": checkpoint.selected_state,
                    "diagnostic_only": int(checkpoint.diagnostic_only),
                    "path_name": window.path_name,
                    "path_index": window.path_index,
                    "window_start": window.window_start,
                    "execution_horizon": window.execution_horizon,
                    "alpha": float(alpha),
                    "sample_index": sample_index,
                    "candidate_id": candidate_result.candidate_id,
                    "sample_seed": seeds[sample_index],
                    "candidate_finite": int(bool(metrics.get("finite", False))),
                    "selectable": int(decision.selectable),
                    "hard_safety_pass": int(decision.hard_safe),
                    "cartesian_improving": int(decision.cartesian_improving),
                    "sample_category": decision.sample_category,
                    "selected_for_k_values": "",
                    "rejection_reasons": "|".join(decision.acceptance_reasons),
                    "safety_rejection_reasons": "|".join(
                        decision.hard_safety_reasons
                    ),
                    "selection_rejection_reasons": "|".join(
                        reason
                        for reason in decision.acceptance_reasons
                        if reason not in TARGET_HARD_GATE_REASONS
                    ),
                    "delta_score": decision.delta_score,
                    "absolute_cartesian_improvement_m": decision.improvement_m,
                    "relative_cartesian_improvement_percent": (
                        100.0 * decision.relative_improvement
                    ),
                    "inference_time_per_sample_s": time_per_sample,
                    "total_sampling_time_s": sampling_time,
                    "cpu_evaluation_time_s": candidate_result.evaluation_time_s,
                    "residual_rms_rad": float(np.sqrt(np.mean(np.square(float(alpha) * physical[sample_index])))),
                    "residual_max_abs_rad": float(np.max(np.abs(float(alpha) * physical[sample_index]))),
                    "candidate_hard_joint_limit_violation_count": int(
                        metrics.get("hard_joint_limit_violation_count", 0)
                    ),
                    "candidate_minimum_joint_limit_margin_rad": float(
                        metrics.get("minimum_joint_limit_margin_rad", -math.inf)
                    ),
                    "candidate_maximum_absolute_joint_step_rad": float(
                        metrics.get("maximum_absolute_joint_step_rad", math.inf)
                    ),
                }
                sample_row.update(scalar_metrics("prior", prior_metrics))
                sample_row.update(scalar_metrics("candidate", metrics))
                window_sample_rows.append(sample_row)
                sample_rows.append(sample_row)

            total_window_time = time.perf_counter() - window_started
            for k_value in sorted(args.k_values):
                selectable = [
                    index for index in range(k_value) if not all_reasons[index]
                ]
                finite_indices = [
                    index for index in range(k_value)
                    if bool(evaluated[index].get("finite", False))
                ]
                ungated_index = (
                    min(finite_indices, key=lambda index: scores[index])
                    if finite_indices else 0
                )
                pre_fallback_safety = not all_safety_reasons[ungated_index]
                if selectable:
                    selected_index = min(selectable, key=lambda index: scores[index])
                    if not decisions[selected_index].selectable:
                        raise AssertionError("Selected candidate is not selectable")
                    selected_k_values = window_sample_rows[selected_index][
                        "selected_for_k_values"
                    ]
                    window_sample_rows[selected_index]["selected_for_k_values"] = (
                        f"{selected_k_values}|{k_value}"
                        if selected_k_values else str(k_value)
                    )
                    window_sample_rows[selected_index]["sample_category"] = "selected"
                    selected_metrics = evaluated[selected_index]
                    selected_q = candidates[selected_index]
                    fallback = False
                    selected_reasons: Sequence[str] = ()
                    selected_seed = seeds[selected_index]
                else:
                    selected_index = -1
                    selected_metrics = prior_metrics
                    selected_q = window.prior_q
                    fallback = True
                    selected_reasons = tuple(
                        sorted({reason for reasons in all_reasons[:k_value] for reason in reasons})
                    )
                    selected_seed = -1
                result_id = (
                    f"diffusion::{checkpoint.label}::{window.index}::"
                    f"a={float(alpha):.12g}::k={k_value}"
                )
                selected_rows.append(
                    result_row(
                        result_id=result_id,
                        window=window,
                        checkpoint=checkpoint.label,
                        method=method_name(checkpoint, k_value),
                        alpha=alpha,
                        k_value=k_value,
                        diagnostic_only=checkpoint.diagnostic_only,
                        prior_metrics=prior_metrics,
                        selected_metrics=selected_metrics,
                        selected_sample_index=selected_index,
                        selected_seed=selected_seed,
                        fallback=fallback,
                        selectable_found=bool(selectable),
                        pre_fallback_safety_pass=pre_fallback_safety,
                        rejection_reasons=selected_reasons,
                        inference_per_sample=time_per_sample,
                        total_time=total_window_time,
                        oracle_improvement=oracle_improvements[key],
                    )
                )
                artifacts[result_id] = PlotArtifact(
                    path_name=window.path_name,
                    window_start=window.window_start,
                    desired=window.desired,
                    prior_q=window.prior_q,
                    candidate_q=np.asarray(selected_q),
                    prior_ee=np.asarray(prior_metrics["ee"]),
                    candidate_ee=np.asarray(selected_metrics["ee"]),
                    execution_horizon=window.execution_horizon,
                )
    return selected_rows, sample_rows, artifacts, {
        "generated_candidate_count": float(generated_candidate_count),
        "gpu_sampling_time_s": cumulative_gpu_sampling_time,
        "cpu_scoring_time_s": cumulative_cpu_scoring_time,
        "cpu_scoring_wall_time_s": cumulative_cpu_scoring_wall_time,
    }


def aggregate_summary(
    window_frame: pd.DataFrame,
    sample_frame: pd.DataFrame,
) -> pd.DataFrame:
    evaluated_methods = window_frame.copy()
    rows: List[Dict[str, Any]] = []
    group_columns = ["checkpoint", "method", "diagnostic_only", "alpha", "K"]
    for keys, group in evaluated_methods.groupby(group_columns, dropna=False):
        key_values = cast(Tuple[Any, Any, Any, Any, Any], keys)
        checkpoint = str(key_values[0])
        method = str(key_values[1])
        diagnostic_only = int(key_values[2])
        alpha = float(key_values[3])
        k_value = int(key_values[4])
        samples = sample_frame[
            (sample_frame["checkpoint"] == checkpoint)
            & np.isclose(sample_frame["alpha"], alpha)
            & (sample_frame["sample_index"] < k_value)
        ]
        improvement = group["absolute_cartesian_improvement_m"].to_numpy(dtype=float)
        oracle = group["oracle_improvement_m"].to_numpy(dtype=float)
        prior_mean = group["prior_prefix_cartesian_mean_error_m"].to_numpy(dtype=float)
        selected_mean = group["selected_prefix_cartesian_mean_error_m"].to_numpy(dtype=float)
        sample_count = len(samples)
        hard_safe_count = int(samples["hard_safety_pass"].sum()) if sample_count else 0
        hard_unsafe_count = sample_count - hard_safe_count
        safe_improving_count = int(
            (
                (samples["hard_safety_pass"] == 1)
                & (samples["cartesian_improving"] == 1)
            ).sum()
        ) if sample_count else 0
        safe_nonimproving_count = int(
            (
                (samples["hard_safety_pass"] == 1)
                & (samples["cartesian_improving"] == 0)
            ).sum()
        ) if sample_count else 0
        rejection_counts = {
            reason: int(
                samples["rejection_reasons"].fillna("").astype(str).map(
                    lambda value, expected=reason: expected in value.split("|")
                ).sum()
            )
            if sample_count else 0
            for reason in TARGET_REJECTION_REASONS
        }
        row: Dict[str, Any] = {
            "checkpoint": checkpoint,
            "method": method,
            "diagnostic_only": diagnostic_only,
            "alpha": alpha,
            "K": k_value,
            "window_count": len(group),
            "path_count": group["path_name"].nunique(),
            "prior_cartesian_mean_error_m": float(np.mean(prior_mean)),
            "prior_cartesian_rms_error_m": float(group["prior_prefix_cartesian_rms_error_m"].mean()),
            "prior_cartesian_p95_error_m": float(group["prior_prefix_cartesian_p95_error_m"].mean()),
            "prior_cartesian_max_error_m": float(group["prior_prefix_cartesian_max_error_m"].mean()),
            "selected_cartesian_mean_error_m": float(np.mean(selected_mean)),
            "selected_cartesian_rms_error_m": float(group["selected_prefix_cartesian_rms_error_m"].mean()),
            "selected_cartesian_p95_error_m": float(group["selected_prefix_cartesian_p95_error_m"].mean()),
            "selected_cartesian_max_error_m": float(group["selected_prefix_cartesian_max_error_m"].mean()),
            "full_window_prior_cartesian_mean_error_m": float(group["prior_full_cartesian_mean_error_m"].mean()),
            "full_window_selected_cartesian_mean_error_m": float(group["selected_full_cartesian_mean_error_m"].mean()),
            "absolute_cartesian_improvement_m": float(np.mean(improvement)),
            "absolute_cartesian_improvement_mm": float(1000.0 * np.mean(improvement)),
            "relative_cartesian_improvement_percent": float(
                100.0 * np.mean(improvement) / max(float(np.mean(prior_mean)), 1.0e-12)
            ),
            "percentage_windows_improved_before_fallback": float(100.0 * group["selectable_found"].mean()),
            "percentage_final_outputs_improved": float(100.0 * group["final_output_improved"].mean()),
            "total_sample_count": sample_count,
            "hard_unsafe_sample_count": hard_unsafe_count,
            "hard_safe_sample_count": hard_safe_count,
            "safe_but_nonimproving_sample_count": safe_nonimproving_count,
            "safe_improving_sample_count": safe_improving_count,
            "hard_unsafe_sample_rate": hard_unsafe_count / sample_count if sample_count else math.nan,
            "hard_safe_sample_rate": hard_safe_count / sample_count if sample_count else math.nan,
            "safe_but_nonimproving_rate": safe_nonimproving_count / sample_count if sample_count else math.nan,
            "safe_improving_sample_rate": safe_improving_count / sample_count if sample_count else math.nan,
            "sample_safety_pass_rate": hard_safe_count / sample_count if sample_count else math.nan,
            "selected_window_count": int(group["selectable_found"].sum()),
            "fallback_window_count": int(group["fallback"].sum()),
            "final_safe_output_count": int(group["final_safety_pass"].sum()),
            "selectable_window_rate": float(group["selectable_found"].mean()),
            "final_safety_pass_rate_before_fallback": float(group["pre_fallback_safety_pass"].mean()),
            "final_safety_pass_rate_after_fallback": float(group["final_safety_pass"].mean()),
            "final_output_safety_pass_rate": float(group["final_safety_pass"].mean()),
            "prior_safety_pass_rate": float(group["prior_hard_safety_pass"].mean()),
            "fallback_rate": float(group["fallback"].mean()),
            "hard_limit_violation_count": int((samples["candidate_hard_joint_limit_violation_count"] > 0).sum()) if len(samples) else 0,
            "maximum_joint_step_violation_count": int((samples["candidate_maximum_absolute_joint_step_rad"] > MAXIMUM_JOINT_STEP_RAD + HARD_JOINT_LIMIT_TOLERANCE_RAD).sum()) if len(samples) else 0,
            "velocity_cost_change": float((group["selected_prefix_velocity_cost"] - group["prior_prefix_velocity_cost"]).mean()),
            "acceleration_cost_change": float((group["selected_prefix_acceleration_cost"] - group["prior_prefix_acceleration_cost"]).mean()),
            "jerk_cost_change": float((group["selected_prefix_jerk_cost"] - group["prior_prefix_jerk_cost"]).mean()),
            "boundary_step_change_rad": float((group["selected_boundary_step_max_abs_rad"] - group["prior_boundary_step_max_abs_rad"]).mean()),
            "boundary_acceleration_change": float((group["selected_boundary_acceleration_discontinuity"] - group["prior_boundary_acceleration_discontinuity"]).mean()),
            "singularity_penalty_change": float((group["selected_prefix_singularity_penalty"] - group["prior_prefix_singularity_penalty"]).mean()),
            "mean_inference_time_per_sample_s": float(samples["inference_time_per_sample_s"].mean()) if len(samples) else math.nan,
            "p95_inference_time_per_sample_s": float(samples["inference_time_per_sample_s"].quantile(0.95)) if len(samples) else math.nan,
            "mean_total_time_per_window_s": float(group["total_time_per_window_s"].mean()),
            "oracle_improvement_m": float(np.mean(oracle)),
            "oracle_improvement_recovered_percent": (
                float(100.0 * np.sum(improvement) / np.sum(oracle))
                if np.sum(oracle) > 1.0e-12 else 0.0
            ),
            "rejection_counts_by_reason": json.dumps(
                rejection_counts, sort_keys=True, separators=(",", ":")
            ),
        }
        row.update(
            {
                f"rejection_{reason}_count": count
                for reason, count in rejection_counts.items()
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def aggregate_paths(window_frame: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    group_columns = [
        "path_name", "checkpoint", "method", "diagnostic_only", "alpha", "K"
    ]
    for keys, group in window_frame.groupby(group_columns, dropna=False):
        key_values = cast(Tuple[Any, Any, Any, Any, Any, Any], keys)
        path_name = str(key_values[0])
        checkpoint = str(key_values[1])
        method = str(key_values[2])
        diagnostic_only = int(key_values[3])
        alpha = float(key_values[4])
        k_value = int(key_values[5])
        rows.append(
            {
                "path_name": path_name,
                "checkpoint": checkpoint,
                "method": method,
                "diagnostic_only": diagnostic_only,
                "alpha": alpha,
                "K": k_value,
                "window_count": len(group),
                "prior_cartesian_mean_error_m": float(group["prior_prefix_cartesian_mean_error_m"].mean()),
                "prior_cartesian_rms_error_m": float(group["prior_prefix_cartesian_rms_error_m"].mean()),
                "prior_cartesian_p95_error_m": float(group["prior_prefix_cartesian_p95_error_m"].mean()),
                "prior_cartesian_max_error_m": float(group["prior_prefix_cartesian_max_error_m"].mean()),
                "selected_cartesian_mean_error_m": float(group["selected_prefix_cartesian_mean_error_m"].mean()),
                "selected_cartesian_rms_error_m": float(group["selected_prefix_cartesian_rms_error_m"].mean()),
                "selected_cartesian_p95_error_m": float(group["selected_prefix_cartesian_p95_error_m"].mean()),
                "selected_cartesian_max_error_m": float(group["selected_prefix_cartesian_max_error_m"].mean()),
                "absolute_cartesian_improvement_m": float(group["absolute_cartesian_improvement_m"].mean()),
                "absolute_cartesian_improvement_mm": float(group["absolute_cartesian_improvement_mm"].mean()),
                "relative_cartesian_improvement_percent": float(
                    100.0 * group["absolute_cartesian_improvement_m"].mean()
                    / max(group["prior_prefix_cartesian_mean_error_m"].mean(), 1.0e-12)
                ),
                "percentage_windows_improved_before_fallback": float(100.0 * group["selectable_found"].mean()),
                "percentage_final_outputs_improved": float(100.0 * group["final_output_improved"].mean()),
                "fallback_rate": float(group["fallback"].mean()),
                "pre_fallback_safety_pass_rate": float(group["pre_fallback_safety_pass"].mean()),
                "final_safety_pass_rate": float(group["final_safety_pass"].mean()),
                "hard_limit_violation_count": int((group["selected_hard_joint_limit_violation_count"] > 0).sum()),
                "maximum_joint_step_violation_count": int((group["selected_maximum_absolute_joint_step_rad"] > MAXIMUM_JOINT_STEP_RAD + HARD_JOINT_LIMIT_TOLERANCE_RAD).sum()),
                "velocity_cost_change": float((group["selected_prefix_velocity_cost"] - group["prior_prefix_velocity_cost"]).mean()),
                "acceleration_cost_change": float((group["selected_prefix_acceleration_cost"] - group["prior_prefix_acceleration_cost"]).mean()),
                "jerk_cost_change": float((group["selected_prefix_jerk_cost"] - group["prior_prefix_jerk_cost"]).mean()),
                "boundary_step_change_rad": float((group["selected_boundary_step_max_abs_rad"] - group["prior_boundary_step_max_abs_rad"]).mean()),
                "boundary_acceleration_change": float((group["selected_boundary_acceleration_discontinuity"] - group["prior_boundary_acceleration_discontinuity"]).mean()),
                "singularity_penalty_change": float((group["selected_prefix_singularity_penalty"] - group["prior_prefix_singularity_penalty"]).mean()),
                "mean_inference_time_per_sample_s": float(group["inference_time_per_sample_s"].mean()),
                "mean_total_time_per_window_s": float(group["total_time_per_window_s"].mean()),
                "oracle_improvement_m": float(group["oracle_improvement_m"].mean()),
                "oracle_improvement_recovered_percent": (
                    float(
                        100.0 * group["absolute_cartesian_improvement_m"].sum()
                        / group["oracle_improvement_m"].sum()
                    )
                    if group["oracle_improvement_m"].sum() > 1.0e-12 else 0.0
                ),
            }
        )
    return pd.DataFrame(rows)


def primary_ranking(
    summary: pd.DataFrame,
    primary_alpha: float,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    primary_mask = (
        (summary["diagnostic_only"] == 0)
        & summary["checkpoint"].isin(("v7_best_raw", "v7_best_ema"))
        & np.isclose(summary["alpha"], primary_alpha)
        & (summary["K"] == summary["K"].max())
    )
    primary = cast(
        pd.DataFrame,
        summary.loc[cast(Any, primary_mask), :].copy(),
    )
    if primary.empty:
        raise RuntimeError(
            f"No primary alpha={primary_alpha:g} best-raw/best-EMA result is available"
        )
    primary = cast(
        pd.DataFrame,
        primary.sort_values(
            by=cast(
                Any,
                [
                    "absolute_cartesian_improvement_m",
                    "percentage_final_outputs_improved",
                    "fallback_rate",
                    "acceleration_cost_change",
                    "boundary_step_change_rad",
                    "mean_total_time_per_window_s",
                ],
            ),
            ascending=cast(Any, [False, False, True, True, True, True]),
        ),
    ).reset_index(drop=True)
    primary.insert(0, "scientific_rank", np.arange(1, len(primary) + 1))
    best_row = cast(pd.Series, primary.iloc[0])
    return primary, {
        str(column): value for column, value in best_row.to_dict().items()
    }


def classify(primary: Mapping[str, Any]) -> str:
    final_output_safety = float(primary["final_output_safety_pass_rate"])
    prior_safety = float(primary["prior_safety_pass_rate"])
    hard_unsafe_sample_rate = float(primary["hard_unsafe_sample_rate"])
    systematic_hard_failure = (
        np.isfinite(hard_unsafe_sample_rate)
        and hard_unsafe_sample_rate
        > SYSTEMATIC_HARD_UNSAFE_SAMPLE_RATE_THRESHOLD
    )
    if (
        final_output_safety < 1.0 - FLOAT_ATOL
        or prior_safety < 1.0 - FLOAT_ATOL
        or systematic_hard_failure
    ):
        return "V7_TEACHER_FORCED_UNSAFE"
    improvement = float(primary["absolute_cartesian_improvement_m"])
    improved_fraction = float(primary["percentage_final_outputs_improved"]) / 100.0
    if improvement <= 0.0:
        return "V7_TEACHER_FORCED_NO_GAIN"
    if (
        improvement > MEANINGFUL_GAIN_THRESHOLD_M
        and improved_fraction >= MEANINGFUL_IMPROVED_WINDOW_FRACTION
    ):
        return "V7_TEACHER_FORCED_MEANINGFUL_GAIN"
    return "V7_TEACHER_FORCED_SMALL_GAIN"


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


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


def atomic_json(value: Mapping[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(value), handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def save_figure(figure: Any, path: Path) -> None:
    figure.tight_layout()
    figure.savefig(str(path), dpi=180, bbox_inches="tight")
    plt.close(figure)


def placeholder_plot(path: Path, title: str, message: str) -> None:
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.axis("off")
    axis.set_title(title)
    axis.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    save_figure(figure, path)


def select_examples(primary_rows: pd.DataFrame, count: int) -> List[Tuple[str, pd.Series]]:
    if count <= 0 or primary_rows.empty:
        return []
    candidates: List[Tuple[str, pd.Series]] = []
    candidates.append(("best diffusion improvement", primary_rows.loc[primary_rows["absolute_cartesian_improvement_m"].idxmax()]))
    median = float(primary_rows["absolute_cartesian_improvement_m"].median())
    median_index = (primary_rows["absolute_cartesian_improvement_m"] - median).abs().idxmin()
    candidates.append(("median improvement", primary_rows.loc[median_index]))
    nonfallback = primary_rows[primary_rows["fallback"] == 0]
    if not nonfallback.empty:
        candidates.append(("worst non-fallback result", nonfallback.loc[nonfallback["absolute_cartesian_improvement_m"].idxmin()]))
    fallback = primary_rows[primary_rows["fallback"] == 1]
    if not fallback.empty:
        candidates.append(("fallback case", fallback.iloc[0]))
    oracle_gap = primary_rows["oracle_improvement_m"] - primary_rows["absolute_cartesian_improvement_m"]
    candidates.append(("high oracle gap", primary_rows.loc[oracle_gap.idxmax()]))
    selected: List[Tuple[str, pd.Series]] = []
    seen: set[str] = set()
    for label, row in candidates:
        result_id = str(row["result_id"])
        if result_id not in seen:
            selected.append((label, row))
            seen.add(result_id)
        if len(selected) >= count:
            break
    return selected


def save_plots(
    summary: pd.DataFrame,
    window_frame: pd.DataFrame,
    primary: Mapping[str, Any],
    artifacts: Mapping[str, PlotArtifact],
    output_dir: Path,
    example_count: int,
) -> None:
    primary_mask = (
        (window_frame["checkpoint"] == primary["checkpoint"])
        & np.isclose(window_frame["alpha"], float(primary["alpha"]))
        & (window_frame["K"] == int(primary["K"]))
    )
    primary_rows = cast(
        pd.DataFrame,
        window_frame.loc[cast(Any, primary_mask), :].copy(),
    )
    figure, axis = plt.subplots(figsize=(6, 6))
    axis.scatter(
        primary_rows["prior_prefix_cartesian_mean_error_m"],
        primary_rows["selected_prefix_cartesian_mean_error_m"], alpha=0.5, s=18,
    )
    limits = axis.get_xlim()
    upper = max(limits[1], axis.get_ylim()[1])
    axis.plot([0.0, upper], [0.0, upper], "k--", linewidth=1)
    axis.set_xlabel("Strong-prior execution-prefix mean error (m)")
    axis.set_ylabel("Selected diffusion execution-prefix mean error (m)")
    axis.set_title("Prior vs diffusion Cartesian error")
    save_figure(figure, output_dir / PLOT_FILES[0])

    figure, axis = plt.subplots(figsize=(8, 5))
    axis.hist(1000.0 * primary_rows["absolute_cartesian_improvement_m"], bins=30)
    axis.axvline(0.0, color="black", linewidth=1)
    axis.set_xlabel("Safe final Cartesian improvement (mm)")
    axis.set_ylabel("Unique windows")
    axis.set_title("Cartesian improvement distribution")
    save_figure(figure, output_dir / PLOT_FILES[1])

    figure, axis = plt.subplots(figsize=(8, 5))
    primary_models_mask = (
        summary["checkpoint"].isin(("v7_best_raw", "v7_best_ema"))
        & np.isclose(summary["alpha"], float(primary["alpha"]))
    )
    primary_models = cast(
        pd.DataFrame,
        summary.loc[cast(Any, primary_models_mask), :],
    )
    for checkpoint, group in primary_models.groupby("checkpoint"):
        group_frame = cast(pd.DataFrame, group)
        curve = cast(
            pd.DataFrame,
            group_frame.sort_values(by=cast(Any, "K")),
        )
        axis.plot(curve["K"], curve["absolute_cartesian_improvement_mm"], marker="o", label=checkpoint)
    axis.set_xlabel("K")
    axis.set_ylabel("Mean safe improvement (mm)")
    axis.set_title("Nested best-of-K gain")
    axis.legend()
    save_figure(figure, output_dir / PLOT_FILES[2])

    figure, axis = plt.subplots(figsize=(8, 5))
    alpha_mask = summary["K"] == summary["K"].max()
    alpha_data = cast(
        pd.DataFrame,
        summary.loc[cast(Any, alpha_mask), :],
    )
    for checkpoint, group in alpha_data.groupby("checkpoint"):
        if checkpoint in ("strong_prior", "oracle"):
            continue
        group_frame = cast(pd.DataFrame, group)
        curve = cast(
            pd.DataFrame,
            group_frame.sort_values(by=cast(Any, "alpha")),
        )
        axis.plot(curve["alpha"], curve["absolute_cartesian_improvement_mm"], marker="o", label=checkpoint)
    axis.set_xlabel("Residual scale alpha")
    axis.set_ylabel("Mean safe improvement (mm)")
    axis.set_title("Alpha sweep")
    axis.legend(fontsize=8)
    save_figure(figure, output_dir / PLOT_FILES[3])

    figure, axis = plt.subplots(figsize=(9, 5))
    safety_mask = np.isclose(summary["alpha"], float(primary["alpha"])) & (
        summary["K"] == summary["K"].max()
    )
    safety_data = cast(
        pd.DataFrame,
        summary.loc[cast(Any, safety_mask), :],
    )
    x = np.arange(len(safety_data))
    axis.bar(x - 0.2, safety_data["final_safety_pass_rate_before_fallback"], width=0.4, label="pre-fallback safety")
    axis.bar(x + 0.2, safety_data["fallback_rate"], width=0.4, label="fallback")
    axis.set_xticks(x)
    axis.set_xticklabels(
        safety_data["checkpoint"].astype(str).tolist(), rotation=25, ha="right"
    )
    axis.set_ylim(0.0, 1.05)
    axis.set_title("Safety pass and fallback rates")
    axis.legend()
    save_figure(figure, output_dir / PLOT_FILES[4])

    raw_ema_mask = safety_data["checkpoint"].isin(("v7_best_raw", "v7_best_ema"))
    raw_ema = cast(
        pd.DataFrame,
        safety_data.loc[cast(Any, raw_ema_mask), :],
    )
    figure, axis = plt.subplots(figsize=(7, 5))
    axis.bar(raw_ema["checkpoint"], raw_ema["absolute_cartesian_improvement_mm"])
    axis.set_ylabel("Mean safe improvement (mm)")
    axis.set_title("Raw vs EMA at alpha=1, maximum K")
    save_figure(figure, output_dir / PLOT_FILES[5])

    figure, axis = plt.subplots(figsize=(8, 5))
    gap = 1000.0 * (
        primary_rows["oracle_improvement_m"]
        - primary_rows["absolute_cartesian_improvement_m"]
    )
    axis.hist(gap, bins=30)
    axis.set_xlabel("Oracle improvement minus diffusion improvement (mm)")
    axis.set_ylabel("Unique windows")
    axis.set_title("Retained-target oracle gap")
    save_figure(figure, output_dir / PLOT_FILES[6])

    examples = select_examples(primary_rows, example_count)
    if not examples:
        for filename, title in zip(PLOT_FILES[7:], ("Example Cartesian trajectories", "Example errors", "Example joints")):
            placeholder_plot(output_dir / filename, title, "No representative examples available")
        return
    figure = plt.figure(figsize=(6 * len(examples), 5))
    for index, (label, row) in enumerate(examples, start=1):
        artifact = artifacts[str(row["result_id"])]
        axis = figure.add_subplot(1, len(examples), index, projection="3d")
        axis.plot(*artifact.desired.T, label="desired")
        axis.plot(*artifact.prior_ee.T, label="prior")
        axis.plot(*artifact.candidate_ee.T, label="selected")
        axis.set_title(f"{label}\n{artifact.path_name}@{artifact.window_start}")
        axis.legend(fontsize=7)
    save_figure(figure, output_dir / PLOT_FILES[7])

    figure, axes = plt.subplots(1, len(examples), figsize=(6 * len(examples), 4), squeeze=False)
    for axis, (label, row) in zip(axes[0], examples):
        artifact = artifacts[str(row["result_id"])]
        prior_error = np.linalg.norm(artifact.prior_ee - artifact.desired, axis=1)
        selected_error = np.linalg.norm(artifact.candidate_ee - artifact.desired, axis=1)
        axis.plot(prior_error, label="prior")
        axis.plot(selected_error, label="selected")
        axis.axvline(artifact.execution_horizon - 0.5, color="black", linestyle="--", linewidth=1)
        axis.set_title(label)
        axis.set_xlabel("Window timestep")
        axis.set_ylabel("Cartesian error (m)")
        axis.legend(fontsize=7)
    save_figure(figure, output_dir / PLOT_FILES[8])

    figure, axes = plt.subplots(len(examples), TARGET_DIM, figsize=(18, 3 * len(examples)), squeeze=False)
    for row_index, (label, row) in enumerate(examples):
        artifact = artifacts[str(row["result_id"])]
        for joint in range(TARGET_DIM):
            axis = axes[row_index, joint]
            axis.plot(artifact.prior_q[:, joint], label="prior")
            axis.plot(artifact.candidate_q[:, joint], label="selected")
            axis.axvline(artifact.execution_horizon - 0.5, color="black", linestyle="--", linewidth=0.8)
            axis.set_title(f"{label}: q{joint + 1}", fontsize=8)
            if row_index == 0 and joint == 0:
                axis.legend(fontsize=7)
    save_figure(figure, output_dir / PLOT_FILES[9])


def run_evaluation(args: argparse.Namespace) -> int:
    evaluation_wall_started = time.perf_counter()
    apply_smoke_mode(args)
    validate_cli(args)
    set_reproducibility(args.seed)
    device = resolve_device(args.device)
    logical_cpu_count = os.cpu_count() or 1
    active_cpu_workers = args.num_cpu_workers
    print(f"detected logical CPU count: {logical_cpu_count}")
    print(f"requested CPU workers: {args.num_cpu_workers}")
    print(f"active CPU workers: {active_cpu_workers}")
    print(f"GPU batch size: {args.gpu_batch_size}")
    dataset_files = {
        "validation": args.dataset_dir / "validation_windows.npz",
        "normalization": args.dataset_dir / "normalization.npz",
        "metadata": args.dataset_dir / "dataset_metadata.json",
        "manifest": args.dataset_dir / "split_manifest.json",
    }
    existing = [
        args.output_dir / name for name in (*OUTPUT_FILES, *PLOT_FILES)
        if (args.output_dir / name).exists()
    ]
    if existing and not args.overwrite:
        raise FileExistsError(
            f"Evaluation outputs already exist: {existing}; pass --overwrite"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    normalization = load_normalization(dataset_files["normalization"])
    metadata = load_json(dataset_files["metadata"])
    manifest = load_json(dataset_files["manifest"])
    all_windows, validation_report = load_unique_validation_windows(
        dataset_files["validation"], normalization, metadata, manifest
    )
    windows = all_windows
    if args.max_windows is not None:
        windows = windows[: args.max_windows]
    timelines = reconstruct_prior_timelines(all_windows)
    robot = make_robot_context(args.robot_urdf)
    (
        prior_metrics_by_key, oracle_improvements, baseline_rows,
        artifacts, contexts,
    ) = evaluate_prior_and_oracle(windows, timelines, robot)

    feature_names = [
        str(value) for value in normalization["condition_feature_names"].tolist()
    ]
    checkpoint_requests: List[Tuple[str, str, Optional[Path], Optional[str], bool, bool]] = [
        ("v7_best_raw", "v7_best_raw", args.best_raw_checkpoint, "raw", False, True),
        ("v7_best_ema", "v7_best_ema", args.best_ema_checkpoint, "ema", False, True),
    ]
    if not args.smoke_test:
        checkpoint_requests.append(
            ("v7_last_checkpoint", "v7_last_checkpoint", args.last_checkpoint, None, True, True)
        )
        if args.v6_checkpoint is not None:
            checkpoint_requests.append(
                ("v6_checkpoint", "v6_checkpoint", args.v6_checkpoint, None, False, False)
            )
    reports: List[Dict[str, Any]] = []
    selected_rows = list(baseline_rows)
    sample_rows: List[Dict[str, Any]] = []
    successful_checkpoints = 0
    total_generated_candidates = 0
    cumulative_gpu_sampling_time = 0.0
    cumulative_cpu_scoring_time = 0.0
    cumulative_cpu_scoring_wall_time = 0.0
    executor: Optional[concurrent.futures.ProcessPoolExecutor] = None
    try:
        if args.num_cpu_workers > 1:
            executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=args.num_cpu_workers,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=initialize_candidate_worker,
                initargs=(str(target_generator.resolve_project_path(args.robot_urdf)),),
            )
        for label, method_prefix, path, expected_state, diagnostic, is_v7 in checkpoint_requests:
            if path is None:
                continue
            runtime: Optional[CheckpointRuntime] = None
            try:
                runtime = load_checkpoint_runtime(
                    label=label, method_prefix=method_prefix, path=path,
                    expected_state=expected_state, diagnostic_only=diagnostic,
                    device=device, dataset_normalization=normalization,
                    dataset_features=feature_names, manifest=manifest,
                    v7_checkpoint=is_v7,
                )
                if args.ddim_steps > int(runtime.schedule.alpha_bars.shape[0]):
                    raise ValueError(
                        f"{path}: --ddim_steps exceeds checkpoint diffusion steps"
                    )
                (
                    checkpoint_selected,
                    checkpoint_samples,
                    checkpoint_artifacts,
                    checkpoint_timing,
                ) = evaluate_checkpoint(
                    runtime, windows, contexts, prior_metrics_by_key,
                    oracle_improvements, robot, args, device, executor,
                )
                selected_rows.extend(checkpoint_selected)
                sample_rows.extend(checkpoint_samples)
                artifacts.update(checkpoint_artifacts)
                runtime.report.update(checkpoint_timing)
                runtime.report["status"] = "evaluated"
                reports.append(runtime.report)
                total_generated_candidates += int(
                    checkpoint_timing["generated_candidate_count"]
                )
                cumulative_gpu_sampling_time += checkpoint_timing[
                    "gpu_sampling_time_s"
                ]
                cumulative_cpu_scoring_time += checkpoint_timing[
                    "cpu_scoring_time_s"
                ]
                cumulative_cpu_scoring_wall_time += checkpoint_timing[
                    "cpu_scoring_wall_time_s"
                ]
                successful_checkpoints += 1
            except Exception as error:
                reports.append(
                    {
                        "checkpoint": label,
                        "checkpoint_path": str(path),
                        "status": "failed",
                        "error": f"{type(error).__name__}: {error}",
                        "diagnostic_only": diagnostic,
                    }
                )
                if label in ("v7_best_raw", "v7_best_ema"):
                    raise
            finally:
                if runtime is not None:
                    runtime.checkpoint.clear()
                    runtime.model.to("cpu")
                    del runtime
                if device.type == "cuda":
                    torch.cuda.empty_cache()
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
    if successful_checkpoints == 0:
        raise RuntimeError("No compatible checkpoints were loaded")

    window_frame = pd.DataFrame(selected_rows)
    sample_frame = pd.DataFrame(sample_rows)
    best_of_k = aggregate_summary(window_frame, sample_frame)
    path_frame = aggregate_paths(window_frame)
    ranked, primary = primary_ranking(best_of_k, args.primary_alpha)
    classification = classify(primary)
    alpha_summary_mask = best_of_k["K"] == best_of_k["K"].max()
    alpha_summary = cast(
        pd.DataFrame,
        best_of_k.loc[cast(Any, alpha_summary_mask), :].copy(),
    )
    alpha_summary = cast(
        pd.DataFrame,
        alpha_summary.sort_values(
            by=cast(Any, ["checkpoint", "alpha"]),
        ),
    )
    rejection_count_columns = [
        f"rejection_{reason}_count" for reason in TARGET_REJECTION_REASONS
    ]
    safety_summary = best_of_k[
        [
            "checkpoint", "method", "alpha", "K", "total_sample_count",
            "hard_unsafe_sample_count", "hard_safe_sample_count",
            "safe_but_nonimproving_sample_count", "safe_improving_sample_count",
            "hard_unsafe_sample_rate", "hard_safe_sample_rate",
            "safe_but_nonimproving_rate", "safe_improving_sample_rate",
            "sample_safety_pass_rate", "selectable_window_rate",
            "selected_window_count", "fallback_window_count",
            "final_safe_output_count", "prior_safety_pass_rate",
            "final_safety_pass_rate_before_fallback",
            "final_safety_pass_rate_after_fallback",
            "final_output_safety_pass_rate", "fallback_rate",
            "hard_limit_violation_count", "maximum_joint_step_violation_count",
            "rejection_counts_by_reason", *rejection_count_columns,
        ]
    ].copy()
    runtime_summary = best_of_k[
        [
            "checkpoint", "method", "alpha", "K",
            "mean_inference_time_per_sample_s", "p95_inference_time_per_sample_s",
            "mean_total_time_per_window_s",
        ]
    ].copy()
    runtime_summary["num_cpu_workers"] = args.num_cpu_workers
    runtime_summary["gpu_batch_size"] = args.gpu_batch_size
    runtime_summary["total_generated_candidates"] = total_generated_candidates
    runtime_summary["cumulative_gpu_sampling_time_s"] = cumulative_gpu_sampling_time
    runtime_summary["cumulative_cpu_scoring_time_s"] = cumulative_cpu_scoring_time
    runtime_summary["cumulative_cpu_scoring_wall_time_s"] = (
        cumulative_cpu_scoring_wall_time
    )

    checkpoint_frame = pd.DataFrame(reports)
    atomic_csv(checkpoint_frame, args.output_dir / "checkpoint_summary.csv")
    atomic_csv(best_of_k, args.output_dir / "best_of_k_summary.csv")
    atomic_csv(alpha_summary, args.output_dir / "alpha_summary.csv")
    atomic_csv(window_frame, args.output_dir / "per_window_results.csv")
    atomic_csv(path_frame, args.output_dir / "per_path_summary.csv")
    if args.save_per_sample_results:
        atomic_csv(sample_frame, args.output_dir / "per_sample_results.csv")
    else:
        atomic_csv(sample_frame.iloc[0:0], args.output_dir / "per_sample_results.csv")
    atomic_csv(safety_summary, args.output_dir / "safety_and_fallback_summary.csv")
    atomic_csv(runtime_summary, args.output_dir / "runtime_summary.csv")

    configuration = {
        "arguments": vars(args),
        "resolved_device": str(device),
        "parallel_execution": {
            "logical_cpu_count": logical_cpu_count,
            "requested_cpu_workers": args.num_cpu_workers,
            "active_cpu_workers": active_cpu_workers,
            "process_start_method": (
                "serial" if args.num_cpu_workers == 1 else "spawn"
            ),
            "worker_robot_initialization": "one xMateCR7 model per worker",
            "gpu_batch_size": args.gpu_batch_size,
            "cuda_process": "main process only",
            "total_generated_candidates": total_generated_candidates,
            "cumulative_gpu_sampling_time_s": cumulative_gpu_sampling_time,
            "cumulative_cpu_scoring_time_s": cumulative_cpu_scoring_time,
            "cumulative_cpu_scoring_wall_time_s": (
                cumulative_cpu_scoring_wall_time
            ),
        },
        "teacher_forced": True,
        "recursive_propagation": False,
        "sampling_initialization": "independent Gaussian noise",
        "sampling_process": "full DDIM reverse diffusion",
        "ddim_steps": args.ddim_steps,
        "ddim_eta": args.eta,
        "nested_k_values": sorted(args.k_values),
        "alphas": args.alphas,
        "primary_alpha": args.primary_alpha,
        "validation_report": validation_report,
        "evaluated_unique_window_count": len(windows),
        "validation_path_names": sorted({window.path_name for window in windows}),
        "checkpoint_reports": reports,
        "condition_feature_ordering": feature_names,
        "reused_components": {
            "model_loader": "train_conditional_diffusion_trajectory_v6_strong_prior_residual_unet.instantiate_v5_model",
            "diffusion_schedule": "train_conditional_diffusion_trajectory_v6_strong_prior_residual_unet.build_schedule",
            "ddim_sampler": "evaluate_diffusion_v6_teacher_forced_validation.sample_batch",
            "robot_costs_and_gates": "generate_diffusion_v7_cost_improving_residual_targets",
            "joint_limits": "generate_ik_seed_path.get_joint_bounds/check_joint_limits",
        },
        "fk_convention": {
            "robot": "ROKAE xMateCR7",
            "active_joints": list(DEFAULT_JOINT_NAMES),
            "end_effector_frame": DEFAULT_EE_LINK,
            "calls": ["robot.update_cfg(cfg)", "robot.get_transform(frame_to='xMateCR7_link6')"],
            "robot_link_fk_used": False,
        },
        "safety_convention": {
            "hard_joint_limit_tolerance_rad": HARD_JOINT_LIMIT_TOLERANCE_RAD,
            "reporting_safety_margin_rad": DEFAULT_JOINT_LIMIT_SAFETY_MARGIN_RAD,
            "maximum_absolute_joint_step_gate_rad": MAXIMUM_JOINT_STEP_RAD,
            "execution_horizon_source": "validation_windows.npz",
            "expected_execution_horizon": EXPECTED_EXECUTION_HORIZON,
            "hard_unsafe_reasons": sorted(TARGET_HARD_GATE_REASONS),
            "systematic_hard_unsafe_sample_rate_threshold": (
                SYSTEMATIC_HARD_UNSAFE_SAMPLE_RATE_THRESHOLD
            ),
        },
        "target_generator_acceptance_semantics": {
            "minimum_cartesian_improvement_m": 1.0e-5,
            "minimum_cartesian_improvement_fraction": 0.005,
            "smoothness_relative_tolerance": 0.10,
            "boundary_absolute_tolerance": 0.01,
            "catastrophic_full_window_threshold": (
                "max(2 * prior_full_mean_error, prior_full_mean_error + 0.01 m)"
            ),
            "soft_score_weights": vars(target_generator.ScoreWeights()),
            "soft_score_floors": vars(target_generator.MetricFloors()),
            "soft_score_terms": [
                "prefix_cartesian_mean_error_m",
                "prefix_cartesian_p95_error_m",
                "prefix_cartesian_max_error_m",
                "prefix_acceleration_cost",
                "prefix_jerk_cost",
                "boundary_step_max_abs_rad",
                "boundary_acceleration_discontinuity",
                "prefix_singularity_penalty",
            ],
            "classification_note": (
                "Nonfinite-boundary, branch-jump, smoothness, boundary, "
                "catastrophic-error, and improvement acceptance failures prevent "
                "selection but do not change the generator-compatible hard_safe flag."
            ),
        },
        "selection_rule": (
            "Retain candidates passing the v7 target-generator acceptance gates "
            "and improving execution-prefix Cartesian mean error; choose minimum "
            "robot-aware delta_score, otherwise fall back to the stored prior."
        ),
        "oracle_use": "reporting only; never used for diffusion sample selection",
        "classification_thresholds": {
            "meaningful_gain_must_exceed_m": MEANINGFUL_GAIN_THRESHOLD_M,
            "minimum_improved_window_fraction": MEANINGFUL_IMPROVED_WINDOW_FRACTION,
            "required_final_output_safety_pass_rate": 1.0,
            "required_stored_prior_safety_pass_rate": 1.0,
            "systematic_hard_unsafe_sample_rate_must_exceed": (
                SYSTEMATIC_HARD_UNSAFE_SAMPLE_RATE_THRESHOLD
            ),
        },
        "assumptions": [
            "Validation execution_horizon is 8, as produced by the v7 target generator.",
            "The v6 optional comparison uses the raw 38-D condition normalized by its own checkpoint statistics.",
            "Safety-margin proximity is reported but is not promoted to a hard joint-limit gate.",
        ],
        "compatibility_concerns": [
            "A v6 checkpoint is skipped unless explicitly supplied; its training split and residual distribution differ from v7.",
            "Checkpoint model class metadata must resolve to the established local Conditional 1D U-Net implementation.",
            "Stored-prior and retained-target oracle evaluation remains serial in the main process; raw diffusion-candidate FK and scoring use the configured workers.",
        ],
    }
    summary_json = {
        "classification": classification,
        "classification_is_provisional": bool(args.smoke_test),
        "primary_result": dict(primary),
        "primary_rejection_counts_by_reason": json.loads(
            str(primary["rejection_counts_by_reason"])
        ),
        "primary_ranking": ranked.to_dict(orient="records"),
        "raw_measurements_preserved": True,
        "smoke_test": bool(args.smoke_test),
    }
    atomic_json(configuration, args.output_dir / "evaluation_configuration.json")
    atomic_json(summary_json, args.output_dir / "evaluation_summary.json")
    save_plots(
        best_of_k, window_frame, primary, artifacts,
        args.output_dir, args.plot_example_count,
    )
    missing_outputs = [
        name for name in (*OUTPUT_FILES, *PLOT_FILES)
        if not (args.output_dir / name).is_file()
    ]
    if missing_outputs:
        raise RuntimeError(f"Evaluation did not write required outputs: {missing_outputs}")
    total_wall_time = time.perf_counter() - evaluation_wall_started
    print(f"file created: {Path(__file__).resolve()}")
    print(
        "reused v6/v7 components: Conditional 1D U-Net loader, linear diffusion "
        "schedule, full Gaussian DDIM sampler, FK, robot-aware costs, and safety gates"
    )
    print(
        "FK/safety: joint1..joint6; xMateCR7_link6 via update_cfg/get_transform; "
        f"hard-limit tolerance={HARD_JOINT_LIMIT_TOLERANCE_RAD:g}; "
        f"maximum joint step={MAXIMUM_JOINT_STEP_RAD:.2f} rad"
    )
    print(
        "selection: minimum v7 robot-aware delta_score among fully gated, "
        "execution-prefix-improving samples; otherwise unchanged-prior fallback; "
        f"primary alpha={args.primary_alpha:g}"
    )
    print(
        "assumptions/compatibility: execution horizon 8; optional v6 uses its own "
        "normalization and is omitted when not supplied"
    )
    print(
        "summary: "
        f"total samples={int(primary['total_sample_count'])}, "
        f"hard-safe samples={int(primary['hard_safe_sample_count'])}, "
        f"safe improving samples={int(primary['safe_improving_sample_count'])}, "
        f"selected windows={int(primary['selected_window_count'])}, "
        f"fallback windows={int(primary['fallback_window_count'])}, "
        f"final safe outputs={int(primary['final_safe_output_count'])}"
    )
    print(f"total generated candidates: {total_generated_candidates}")
    print(f"cumulative GPU sampling time: {cumulative_gpu_sampling_time:.3f} s")
    print(f"cumulative CPU FK/scoring time: {cumulative_cpu_scoring_time:.3f} s")
    print(f"total wall time: {total_wall_time:.3f} s")
    if args.smoke_test:
        print("V7_TEACHER_FORCED_SMOKE_TEST_COMPLETE")
        print(f"provisional scientific classification: {classification}")
    else:
        print(f"classification: {classification}")
    return 0


def main() -> int:
    args = parse_args()
    try:
        return run_evaluation(args)
    except Exception:
        print("classification: V7_TEACHER_FORCED_EVALUATION_FAILED")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
