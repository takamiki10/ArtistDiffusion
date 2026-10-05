#!/usr/bin/env python3
"""Final multi-seed benchmark for the frozen base-tail diffusion rollout.

This file deliberately reuses ``evaluate_global_anchored_receding_horizon_rollout``
for dataset/model loading, v5b conditioning, residual normalization, diffusion
sampling, tapering, candidate scoring, safety gates, FK, robot costs, and the
recursive buffer update.  It adds only experiment orchestration, runtime
invariants, compatible external-baseline loading, paired statistics, and plots.

Expert joints are passed to the practical rollout only through the existing
``expert_q_evaluation_only`` argument and are consumed solely by final metric
calculation.  They never enter conditioning, sampling, gating, or ranking.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import csv
import hashlib
import inspect
import json
import math
import os
import time
import warnings
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import torch

import diagnose_action_buffer_candidate_selection as action_buffer_diagnostic
import evaluate_global_anchored_receding_horizon_rollout as rollout


DEFAULT_TEST_NPZ = Path(
    "data/cartesian_expert_dataset_v3/diffusion_v2/diffusion_test_v2.npz"
)
DEFAULT_WINDOW_NPZ = Path(
    "data/cartesian_expert_dataset_v3/"
    "diffusion_v5b_residual_windows_fk_condition/test_windows.npz"
)
DEFAULT_STATS_NPZ = Path(
    "data/cartesian_expert_dataset_v3/"
    "diffusion_v5b_residual_windows_fk_condition/normalization_stats.npz"
)
DEFAULT_CHECKPOINT = Path(
    "data/cartesian_expert_dataset_v3/"
    "diffusion_v5b_residual_unet_fk_condition/best_checkpoint.pt"
)
DEFAULT_PRIOR_DIR = Path(
    "data/cartesian_expert_dataset_v3/mlp_v3_test_predictions"
)
DEFAULT_OUTPUT_DIR = Path(
    "data/cartesian_expert_dataset_v3/base_tail_final_benchmark"
)
DEFAULT_BASELINE_ROOT = Path("data/cartesian_expert_dataset_v3")

FROZEN_ARCHITECTURE = {
    "prediction_horizon": 32,
    "execution_horizon": 8,
    "t_init": 10,
    "num_base_samples": 16,
    "alphas": (0.05, 0.10),
    "ramp_length": 8,
    "taper_mode": "linear",
    "tail_mode": "base_tail",
    "tail_extension": "bootstrap_prior",
    "selector": "larger_picture",
}

FROZEN_ROLLOUT_SETTINGS = {
    "ranking_discount": 0.9,
    "lookahead_points": 8,
    "w_prefix": 1.0,
    "w_tail": 0.35,
    "w_reference_joint": 0.50,
    "w_reference_cartesian": 1.0,
    "w_terminal": 0.75,
    "w_late_window": 0.50,
    "w_shift_boundary": 1.0,
    "w_extension": 0.75,
    "w_safety": 10.0,
    "boundary_ratio": 2.0,
    "prefix_step_ratio": 2.0,
    "shift_boundary_ratio": 2.0,
    "absolute_boundary_limit": 0.25,
    "absolute_prefix_step_limit": 0.25,
    "absolute_shift_boundary_limit": 0.25,
    "max_extension_clipped_values": 0,
    "max_extension_joint_step": 0.25,
    "max_reference_joint_drift": 0.50,
    "max_reference_cartesian_drift": 0.10,
}

EPS = 1e-12
TIE_ABS_TOL = 1e-12
TIE_REL_TOL = 1e-9

# All practical metrics are lower-is-better.  Descriptive counters and runtime
# are retained in raw outputs but are not silently interpreted as quality.
QUALITY_METRICS = tuple(
    metric
    for metric in rollout.FULL_METRICS
    if metric
    not in {
        "planning_cycle_count",
        "diffusion_selection_fraction",
        "unsafe_diffusion_selection_count",
        "buffer_fallback_count",
        "safety_improving_selection_count",
    }
)

RUN_METRICS = tuple(
    dict.fromkeys(
        (
            *rollout.FULL_METRICS,
            "runtime_seconds",
            "num_fk_evaluations",
            "num_fk_batches",
            "num_fk_configurations",
            "num_generated_candidates",
            "num_generated_residual_samples",
        )
    )
)

REQUIRED_EXTERNAL_COMPARISON_METRICS = (
    "mean_cartesian_error",
    "rms_cartesian_error",
    "max_cartesian_error",
    "drawing_total_cost",
    "max_joint_step",
    "max_joint_acceleration",
    "max_joint_jerk",
    "joint_limit_violation_count",
    "joint_limit_violation_magnitude",
)

EXTERNAL_PLANNING_ONLY_METRICS = {
    "mean_planning_boundary_discontinuity",
    "max_planning_boundary_discontinuity",
    "mean_future_shift_boundary_discontinuity",
    "max_future_shift_boundary_discontinuity",
    "extension_clipping_count",
    "extension_clipping_magnitude",
    "planning_cycle_count",
    "diffusion_selection_fraction",
    "unsafe_diffusion_selection_count",
    "buffer_fallback_count",
    "safety_improving_selection_count",
    "num_generated_candidates",
    "num_generated_residual_samples",
}

EXTERNAL_AVAILABLE_STATUSES = {
    "source_csv",
    "computed_from_compatible_trajectory",
}

REQUIRED_PLOT_COUNT = 13
COMPLETION_MANIFEST_NAME = "benchmark_completion_manifest.json"
_PUBLISHED_PLOT_PATHS: List[Path] = []

# Predeclared validation-only tolerances.  These do not affect rollout scoring,
# safety gates, selection, or any existing benchmark metric.
FK_HOMOGENEOUS_ATOL = 1e-8
FK_ROTATION_ATOL = 1e-6
FK_RANGE_DENOMINATOR_ATOL = 1e-12
FK_CLASSIFICATION_MATERIAL_IMPROVEMENT_FACTOR = 0.50
FK_CLASSIFICATIONS = (
    "PASS_DIRECT",
    "POSSIBLE_FIXED_TRANSLATION",
    "POSSIBLE_RIGID_FRAME_MISMATCH",
    "POSSIBLE_TEMPORAL_MISALIGNMENT",
    "EXPERT_SHAPE_MISMATCH",
    "INCONCLUSIVE",
)

# The singular acceleration/jerk/boundary requests are resolved to their
# maximum full-trajectory versions, the conservative and directly auditable
# convention used by the scientific decision layer.
PRIMARY_METRICS = (
    "mean_cartesian_error",
    "rms_cartesian_error",
    "max_cartesian_error",
    "drawing_total_cost",
    "max_joint_step",
    "max_joint_acceleration",
    "max_joint_jerk",
    "max_planning_boundary_discontinuity",
)

PLOT_COLORS = {
    "buffer_only": "#4C78A8",
    "base_tail_diffusion": "#F58518",
    "adaptive_mlp_ik": "#54A24B",
    "diffusion_v1_best_of_k": "#E45756",
    "expert_ik": "#B279A2",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the frozen H32/E8/t10 scaled-tapered base-tail rollout "
            "on held-out paths with independent seeds and paired statistics."
        )
    )
    parser.add_argument("--test_npz", type=Path, default=DEFAULT_TEST_NPZ)
    parser.add_argument("--window_npz", type=Path, default=DEFAULT_WINDOW_NPZ)
    parser.add_argument(
        "--normalization_stats", type=Path, default=DEFAULT_STATS_NPZ
    )
    parser.add_argument(
        "--diffusion_checkpoint", type=Path, default=DEFAULT_CHECKPOINT
    )
    parser.add_argument("--prior_dir", type=Path, default=DEFAULT_PRIOR_DIR)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)

    parser.add_argument("--prediction_horizon", type=int, default=32)
    parser.add_argument("--execution_horizon", type=int, default=8)
    parser.add_argument("--t_init", type=int, default=10)
    parser.add_argument("--num_base_samples", type=int, default=16)
    parser.add_argument("--alphas", nargs="+", type=float, default=(0.05, 0.10))
    parser.add_argument("--ramp_length", type=int, default=8)
    parser.add_argument("--taper_mode", choices=("linear", "cosine"), default="linear")
    parser.add_argument(
        "--tail_extension",
        choices=("bootstrap_prior", "constant_velocity", "constant_position"),
        default="bootstrap_prior",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=(0, 1, 2, 3, 4))
    parser.add_argument(
        "--max_paths",
        type=int,
        default=0,
        help="Zero evaluates every available held-out path.",
    )
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    parser.add_argument("--num_diffusion_steps", type=int, default=None)

    # Validation-only controls. These helpers are intentionally not wired into
    # the benchmark path yet; phase 1 establishes their public CLI contract.
    parser.add_argument(
        "--run_fk_validation",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--diagnostic_only", action="store_true")
    parser.add_argument("--fail_on_expert_fk_mismatch", action="store_true")
    parser.add_argument(
        "--expert_mean_error_threshold",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--expert_max_error_threshold",
        type=float,
        default=0.03,
    )
    parser.add_argument("--tool_transform", type=Path, default=None)
    parser.add_argument(
        "--tool_offset_xyz",
        nargs=3,
        type=float,
        default=None,
        metavar=("X", "Y", "Z"),
    )
    parser.add_argument(
        "--tool_offset_rpy",
        nargs=3,
        type=float,
        default=None,
        metavar=("ROLL", "PITCH", "YAW"),
    )
    parser.add_argument(
        "--plot_equal_axes",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--save_fk_pointwise_csv",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--validation_max_paths",
        type=int,
        default=None,
        help=(
            "Explicit FK-validation-only path limit. When omitted, --max_paths "
            "is used; zero means every available validation path."
        ),
    )

    # Exact existing larger-picture score and gates.
    parser.add_argument("--ranking_discount", type=float, default=0.9)
    parser.add_argument("--lookahead_points", type=int, default=8)
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
    parser.add_argument("--absolute_shift_boundary_limit", type=float, default=0.25)
    parser.add_argument("--max_extension_clipped_values", type=int, default=0)
    parser.add_argument("--max_extension_joint_step", type=float, default=0.25)
    parser.add_argument("--max_reference_joint_drift", type=float, default=0.50)
    parser.add_argument("--max_reference_cartesian_drift", type=float, default=0.10)

    # External result discovery.  Explicit CLI paths always win.
    parser.add_argument("--baseline_root", type=Path, default=DEFAULT_BASELINE_ROOT)
    parser.add_argument("--adaptive_mlp_ik_csv", type=Path, default=None)
    parser.add_argument("--diffusion_v1_best_of_k_csv", type=Path, default=None)
    parser.add_argument("--expert_ik_csv", type=Path, default=None)
    parser.add_argument(
        "--adaptive_mlp_ik_q_root",
        type=Path,
        default=Path("data/cartesian_expert_dataset_v3/experts/test"),
    )
    parser.add_argument(
        "--no_baseline_trajectory_augmentation",
        action="store_true",
        help=(
            "Do not calculate missing benchmark metrics from already-generated "
            "external trajectory CSVs. Existing summary metrics are still reused."
        ),
    )
    parser.add_argument("--baseline_metric_atol", type=float, default=1e-5)
    parser.add_argument("--baseline_metric_rtol", type=float, default=1e-4)

    # Inference and reproducibility.
    parser.add_argument("--bootstrap_resamples", type=int, default=10_000)
    parser.add_argument("--analysis_seed", type=int, default=20260714)
    parser.add_argument(
        "--determinism_check",
        choices=("all", "first_path", "none"),
        default="all",
        help=(
            "Rerun seeds in reverse order. 'all' is the prespecified final "
            "benchmark behavior; reduced modes are diagnostic only."
        ),
    )
    parser.add_argument("--unstable_cv_threshold", type=float, default=0.25)
    parser.add_argument("--max_seed_contribution_share", type=float, default=0.50)

    # Prespecified scientific decision thresholds.
    parser.add_argument("--max_joint_step_relative_limit", type=float, default=1.10)
    parser.add_argument("--max_joint_step_absolute_limit", type=float, default=0.25)
    parser.add_argument("--smoothness_relative_limit", type=float, default=1.10)
    parser.add_argument("--max_joint_acceleration_limit", type=float, default=0.25)
    parser.add_argument("--max_joint_jerk_limit", type=float, default=0.25)
    parser.add_argument("--min_practical_improvement_percent", type=float, default=1.0)
    parser.add_argument("--min_benefit_rank_biserial", type=float, default=0.10)
    parser.add_argument("--max_runtime_ratio", type=float, default=1000.0)
    parser.add_argument(
        "--min_improvement_per_extra_second", type=float, default=0.0
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not math.isfinite(args.expert_mean_error_threshold):
        raise ValueError("--expert_mean_error_threshold must be finite")
    if args.expert_mean_error_threshold <= 0.0:
        raise ValueError("--expert_mean_error_threshold must be > 0")
    if not math.isfinite(args.expert_max_error_threshold):
        raise ValueError("--expert_max_error_threshold must be finite")
    if args.expert_max_error_threshold <= 0.0:
        raise ValueError("--expert_max_error_threshold must be > 0")
    if (
        args.validation_max_paths is not None
        and args.validation_max_paths < 0
    ):
        raise ValueError("--validation_max_paths must be >= 0")
    if args.diagnostic_only and not args.run_fk_validation:
        raise ValueError("--diagnostic_only requires --run_fk_validation")

    uses_numeric_tool_offset = (
        args.tool_offset_xyz is not None or args.tool_offset_rpy is not None
    )
    if args.tool_transform is not None and uses_numeric_tool_offset:
        raise ValueError(
            "--tool_transform is mutually exclusive with --tool_offset_xyz "
            "and --tool_offset_rpy"
        )
    for option_name, values in (
        ("--tool_offset_xyz", args.tool_offset_xyz),
        ("--tool_offset_rpy", args.tool_offset_rpy),
    ):
        if values is None:
            continue
        if len(values) != 3:
            raise ValueError(f"{option_name} requires exactly three values")
        if not np.all(np.isfinite(np.asarray(values, dtype=np.float64))):
            raise ValueError(f"{option_name} values must all be finite")

    # Resolve here so malformed transform files and invalid homogeneous
    # matrices fail at argument validation time.
    resolve_tool_transform(args)

    observed_rollout_settings = {
        key: getattr(args, key) for key in FROZEN_ROLLOUT_SETTINGS
    }
    if observed_rollout_settings != FROZEN_ROLLOUT_SETTINGS:
        differences = {
            key: {
                "required": FROZEN_ROLLOUT_SETTINGS[key],
                "observed": observed_rollout_settings[key],
            }
            for key in FROZEN_ROLLOUT_SETTINGS
            if observed_rollout_settings[key] != FROZEN_ROLLOUT_SETTINGS[key]
        }
        raise ValueError(
            "This final benchmark has frozen rollout scoring and safety settings; "
            f"incompatible overrides were supplied: {json.dumps(differences, sort_keys=True)}"
        )
    if args.num_diffusion_steps is not None:
        raise ValueError(
            "--num_diffusion_steps cannot override the frozen checkpoint diffusion schedule"
        )
    observed = {
        "prediction_horizon": args.prediction_horizon,
        "execution_horizon": args.execution_horizon,
        "t_init": args.t_init,
        "num_base_samples": args.num_base_samples,
        "alphas": tuple(float(value) for value in args.alphas),
        "ramp_length": args.ramp_length,
        "taper_mode": args.taper_mode,
        "tail_mode": "base_tail",
        "tail_extension": args.tail_extension,
        "selector": "larger_picture",
    }
    if observed != FROZEN_ARCHITECTURE:
        differences = {
            key: {"required": FROZEN_ARCHITECTURE[key], "observed": value}
            for key, value in observed.items()
            if value != FROZEN_ARCHITECTURE[key]
        }
        raise ValueError(
            "This final benchmark has a frozen architecture; incompatible "
            f"overrides were supplied: {json.dumps(differences, sort_keys=True)}"
        )
    if args.determinism_check != "all":
        raise ValueError(
            "The final benchmark requires --determinism_check all so every "
            "path and seed is rerun and verified"
        )
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("--seeds must contain at least one unique seed")
    if any(seed < 0 for seed in args.seeds):
        raise ValueError("--seeds must be non-negative")
    if args.max_paths < 0:
        raise ValueError("--max_paths must be zero or positive")
    if args.bootstrap_resamples < 10_000:
        raise ValueError("Statistical analysis requires at least 10,000 bootstrap resamples")
    positive = (
        "baseline_metric_atol",
        "baseline_metric_rtol",
        "unstable_cv_threshold",
        "max_seed_contribution_share",
        "max_joint_step_relative_limit",
        "max_joint_step_absolute_limit",
        "smoothness_relative_limit",
        "max_joint_acceleration_limit",
        "max_joint_jerk_limit",
        "max_runtime_ratio",
    )
    if any(float(getattr(args, name)) <= 0.0 for name in positive):
        raise ValueError("Configured tolerances and safety limits must be positive")
    if not 0.0 < args.max_seed_contribution_share <= 1.0:
        raise ValueError("--max_seed_contribution_share must be in (0,1]")
    if args.min_practical_improvement_percent < 0.0:
        raise ValueError("--min_practical_improvement_percent cannot be negative")
    if (
        not math.isfinite(float(args.min_benefit_rank_biserial))
        or not 0.0 <= float(args.min_benefit_rank_biserial) <= 1.0
    ):
        raise ValueError("--min_benefit_rank_biserial must be finite and in [0,1]")
    if args.min_improvement_per_extra_second < 0.0:
        raise ValueError("--min_improvement_per_extra_second cannot be negative")


def resolve_fk_validation_path_selection(
    args: argparse.Namespace,
    available_path_names: Sequence[str],
) -> Tuple[List[str], str, int]:
    """Resolve the FK cohort and retain the exact CLI option that selected it."""

    available_names = [str(name) for name in available_path_names]
    if not available_names:
        raise ValueError("FK validation requires at least one available path")
    if len(set(available_names)) != len(available_names):
        raise ValueError("Available FK validation path names must be unique")

    explicit_validation_limit = getattr(args, "validation_max_paths", None)
    if explicit_validation_limit is None:
        determining_option = "--max_paths"
        requested_limit = int(args.max_paths)
    else:
        determining_option = "--validation_max_paths"
        requested_limit = int(explicit_validation_limit)
    if requested_limit < 0:
        raise ValueError(f"{determining_option} must be zero or positive")

    resolved_count = (
        len(available_names)
        if requested_limit == 0
        else min(requested_limit, len(available_names))
    )
    return (
        available_names[:resolved_count],
        determining_option,
        requested_limit,
    )


def print_fk_validation_path_selection(
    *,
    selected_count: int,
    available_count: int,
    determining_option: str,
    requested_limit: int,
) -> None:
    """Print an auditable FK path-limit resolution."""

    if selected_count < 0 or selected_count > available_count:
        raise ValueError(
            "selected FK path count must be between zero and available_count"
        )
    if determining_option not in {"--max_paths", "--validation_max_paths"}:
        raise ValueError(f"Unknown FK path-limit option {determining_option!r}")
    print(
        "[FK validation path selection] "
        f"selected={selected_count} available={available_count} "
        f"determined_by={determining_option} value={requested_limit} "
        "zero_means_all=True"
    )


def rollout_namespace(args: argparse.Namespace, seed: int) -> argparse.Namespace:
    """Build the exact namespace expected by the reused rollout module."""
    namespace = argparse.Namespace(
        prediction_horizon=args.prediction_horizon,
        execution_horizon=args.execution_horizon,
        t_init=args.t_init,
        ramp_length=args.ramp_length,
        taper_mode=args.taper_mode,
        alphas=tuple(float(value) for value in args.alphas),
        num_base_samples=args.num_base_samples,
        tail_modes=("base_tail",),
        tail_extension=args.tail_extension,
        selector="larger_picture",
        ranking_discount=args.ranking_discount,
        lookahead_points=args.lookahead_points,
        tail_decay_mode="linear",
        tail_decay_length=8,
        tail_decay_beta=0.35,
        global_anchor_mode="linear",
        global_anchor_length=8,
        global_anchor_beta=0.35,
        w_prefix=args.w_prefix,
        w_tail=args.w_tail,
        w_reference_joint=args.w_reference_joint,
        w_reference_cartesian=args.w_reference_cartesian,
        w_terminal=args.w_terminal,
        w_late_window=args.w_late_window,
        w_shift_boundary=args.w_shift_boundary,
        w_extension=args.w_extension,
        w_safety=args.w_safety,
        boundary_ratio=args.boundary_ratio,
        prefix_step_ratio=args.prefix_step_ratio,
        shift_boundary_ratio=args.shift_boundary_ratio,
        absolute_boundary_limit=args.absolute_boundary_limit,
        absolute_prefix_step_limit=args.absolute_prefix_step_limit,
        absolute_shift_boundary_limit=args.absolute_shift_boundary_limit,
        max_extension_clipped_values=args.max_extension_clipped_values,
        max_extension_joint_step=args.max_extension_joint_step,
        max_reference_joint_drift=args.max_reference_joint_drift,
        max_reference_cartesian_drift=args.max_reference_cartesian_drift,
        material_safety_ratio=args.max_joint_step_relative_limit,
        num_diffusion_steps=args.num_diffusion_steps,
        max_paths=None,
        save_candidate_details=True,
        device=args.device,
        seed=int(seed),
    )
    rollout.validate_args(namespace)
    return namespace


def finite_float(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def csv_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    if isinstance(value, Path):
        return str(value)
    return value


def write_records_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty required output: {path}")
    fieldnames: List[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key, "")) for key in fieldnames})


def read_records_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Existing baseline CSV is empty: {path}")
    return rows


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_analysis_seed(base_seed: int, label: str) -> int:
    payload = f"base-tail-final-benchmark-v1|{base_seed}|{label}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**32)


def candidate_noise_seed(
    global_seed: int,
    path_index: int,
    cycle_index: int,
    sample_index: int,
    num_paths: int,
    num_cycles: int,
    num_samples: int,
) -> int:
    """Injectively encode the required candidate-noise key into uint32.

    The repository sampler accepts a single integer seed.  Its historical
    additive encoding makes global seed ``s``, sample ``i+1`` identical to
    global seed ``s+1``, sample ``i``.  This mixed-radix encoding preserves the
    existing per-candidate sampling routine while making every requested
    (global seed, path, cycle, sample) stream distinct.
    """
    if not (
        global_seed >= 0
        and 0 <= path_index < num_paths
        and 0 <= cycle_index < num_cycles
        and 0 <= sample_index < num_samples
    ):
        raise ValueError("Candidate-noise key is outside the declared benchmark dimensions")
    value = (
        (
            (int(global_seed) * int(num_paths) + int(path_index))
            * int(num_cycles)
            + int(cycle_index)
        )
        * int(num_samples)
        + int(sample_index)
    )
    if value > np.iinfo(np.uint32).max:
        raise ValueError(
            "Candidate-noise tuple cannot be represented by the repository's "
            "uint32-compatible seeding convention; use smaller non-negative global seeds"
        )
    return value


def outcome(candidate: float, baseline: float) -> str:
    tolerance = TIE_ABS_TOL + TIE_REL_TOL * max(abs(candidate), abs(baseline))
    difference = candidate - baseline
    if difference < -tolerance:
        return "improved"
    if difference > tolerance:
        return "worsened"
    return "tied"


def percentage_difference(candidate: float, baseline: float) -> float:
    if abs(baseline) <= EPS:
        return float("nan")
    return 100.0 * (candidate - baseline) / abs(baseline)


def configure_determinism() -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


class FKCounter:
    """Count canonical FK calls/configurations, including per planning cycle."""

    def __init__(self, original: Any) -> None:
        self.original = original
        self.call_count = 0
        self.configuration_count = 0
        self.current_cycle: Optional[int] = None
        self.cycle_calls: Counter[int] = Counter()
        self.cycle_configurations: Counter[int] = Counter()

    def __call__(
        self,
        robot: Any,
        joint_names: Sequence[str],
        ee_link: str,
        q_traj: np.ndarray,
    ) -> np.ndarray:
        q_array = np.asarray(q_traj)
        count = int(q_array.shape[0]) if q_array.ndim >= 1 else 0
        self.call_count += 1
        self.configuration_count += count
        if self.current_cycle is not None:
            self.cycle_calls[self.current_cycle] += 1
            self.cycle_configurations[self.current_cycle] += count
        return self.original(robot, joint_names, ee_link, q_traj)


def _update_length_prefixed_hash(digest: Any, payload: bytes) -> None:
    digest.update(len(payload).to_bytes(8, "little", signed=False))
    digest.update(payload)


class GeneratedSampleRecorder:
    """Stable fingerprint of every normalized residual returned by the sampler."""

    def __init__(self) -> None:
        self._digest = hashlib.sha256()
        self.sample_count = 0

    def record(
        self,
        *,
        global_seed: int,
        path_index: int,
        cycle_index: int,
        sample_index: int,
        encoded_seed: int,
        sample: Optional[np.ndarray],
    ) -> None:
        metadata = json.dumps(
            {
                "global_seed": int(global_seed),
                "path_index": int(path_index),
                "cycle_index": int(cycle_index),
                "sample_index": int(sample_index),
                "encoded_seed": int(encoded_seed),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        _update_length_prefixed_hash(self._digest, metadata)
        if sample is None:
            _update_length_prefixed_hash(self._digest, b"none")
        else:
            array = np.ascontiguousarray(np.asarray(sample))
            descriptor = json.dumps(
                {"dtype": array.dtype.str, "shape": list(array.shape)},
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            _update_length_prefixed_hash(self._digest, descriptor)
            _update_length_prefixed_hash(self._digest, array.tobytes(order="C"))
        self.sample_count += 1

    def hexdigest(self) -> str:
        return self._digest.hexdigest()


class BaseTailGuard:
    """Runtime proof that no unexecuted diffusion tail reaches the next buffer."""

    def __init__(self, original: Any) -> None:
        self.original = original
        self.check_count = 0

    def __call__(self, **kwargs: Any) -> Dict[str, Any]:
        base_snapshot = np.asarray(kwargs["base_buffer"], dtype=np.float32).copy()
        candidate_snapshot = np.asarray(kwargs["candidate_q"], dtype=np.float32).copy()
        result = self.original(**kwargs)
        if kwargs["tail_mode"] != "base_tail":
            raise AssertionError("The final benchmark must only execute base_tail")
        execution_count = int(kwargs["execution_count"])
        base_buffer = np.asarray(kwargs["base_buffer"], dtype=np.float32)
        candidate_input = np.asarray(kwargs["candidate_q"], dtype=np.float32)
        candidate = candidate_snapshot
        if not np.array_equal(base_buffer, base_snapshot):
            raise AssertionError("Candidate evaluation mutated the preserved base_buffer")
        lower = np.asarray(kwargs["lower"], dtype=np.float32)
        upper = np.asarray(kwargs["upper"], dtype=np.float32)
        expected, _ = rollout.clip_to_joint_limits(
            base_buffer[execution_count:].copy(), lower, upper
        )
        retained = np.asarray(result["_retained_tail"], dtype=np.float32)
        next_buffer = np.asarray(result["_next_buffer"], dtype=np.float32)
        if not np.array_equal(retained, expected):
            raise AssertionError(
                "Base-tail invariant failed: retained tail is not base_buffer[E:H]"
            )
        if not np.array_equal(next_buffer[: expected.shape[0]], expected):
            raise AssertionError(
                "Base-tail invariant failed: next-buffer prefix is not retained base tail"
            )
        if np.shares_memory(next_buffer, candidate_input):
            raise AssertionError("Next buffer aliases storage from the selected candidate")

        # Metamorphic proof: replacing every unexecuted selected-tail value by
        # a sentinel must leave both tail handling and extension bit-identical.
        probe = candidate.copy()
        if probe.shape[0] > execution_count:
            sentinel = np.linspace(
                0.1234567,
                0.9876543,
                probe[execution_count:].size,
                dtype=np.float32,
            ).reshape(probe[execution_count:].shape)
            probe[execution_count:] = probe[execution_count:] + sentinel
        probe_tail, _, _ = rollout.handled_tail(
            candidate_q=probe,
            base_buffer=base_buffer,
            global_reference=kwargs["global_reference"],
            start=int(kwargs["start"]),
            execution_count=execution_count,
            tail_mode="base_tail",
            is_diffusion=bool(kwargs["is_diffusion"]),
            args=kwargs["args"],
        )
        if not np.array_equal(probe_tail, base_buffer[execution_count:]):
            raise AssertionError("Sentinel diffusion tail changed the retained base tail")

        extension_count = execution_count
        first_global_index = (
            int(kwargs["start"]) + execution_count + expected.shape[0]
        )
        extension_a, sources_a = rollout.make_extension(
            retained_tail=expected,
            selected_candidate=candidate,
            execution_count=execution_count,
            extension_count=extension_count,
            first_global_index=first_global_index,
            global_reference=kwargs["global_reference"],
            mode=kwargs["args"].tail_extension,
        )
        extension_b, sources_b = rollout.make_extension(
            retained_tail=expected,
            selected_candidate=probe,
            execution_count=execution_count,
            extension_count=extension_count,
            first_global_index=first_global_index,
            global_reference=kwargs["global_reference"],
            mode=kwargs["args"].tail_extension,
        )
        if not np.array_equal(extension_a, extension_b) or sources_a != sources_b:
            raise AssertionError("Unexecuted selected tail changed buffer extension")
        expected_extension, _ = rollout.clip_to_joint_limits(extension_a, lower, upper)
        if not np.array_equal(next_buffer[expected.shape[0] :], expected_extension):
            raise AssertionError(
                "Next-buffer suffix does not equal the candidate-tail-independent extension"
            )

        self.check_count += 1
        result["base_tail_tail_exclusion_asserted"] = 1
        return result


@contextlib.contextmanager
def rollout_instrumentation(
    *,
    num_paths: int,
    num_cycles: int,
    num_samples: int,
) -> Iterator[Tuple[FKCounter, BaseTailGuard, GeneratedSampleRecorder]]:
    original_fk_rollout = rollout.fk_positions
    original_fk_diagnostic = action_buffer_diagnostic.fk_positions
    original_condition = rollout.build_v5b_condition
    original_full_metrics = rollout.full_trajectory_metrics
    original_preview = rollout.extension_preview
    original_generate = rollout.generate_seeded_samples
    counter = FKCounter(original_fk_rollout)
    guard = BaseTailGuard(original_preview)
    sample_recorder = GeneratedSampleRecorder()

    def instrumented_condition(*args: Any, **kwargs: Any) -> Any:
        indices = np.asarray(kwargs.get("indices"), dtype=np.int64)
        if indices.size == 0:
            raise AssertionError("Condition construction received no global indices")
        counter.current_cycle = int(indices[0]) // FROZEN_ARCHITECTURE["execution_horizon"]
        return original_condition(*args, **kwargs)

    def instrumented_full_metrics(*args: Any, **kwargs: Any) -> Any:
        counter.current_cycle = None
        return original_full_metrics(*args, **kwargs)

    def instrumented_generate(*args: Any, **kwargs: Any) -> Any:
        if args:
            raise AssertionError("Repository candidate sampler is expected to use keyword arguments")
        requested_samples = int(kwargs["num_samples"])
        if requested_samples != num_samples or requested_samples <= 0:
            raise AssertionError(
                "Repository sampler sample count differs from the frozen benchmark"
            )
        global_seed = int(kwargs["global_seed"])
        path_index = int(kwargs["path_index"])
        cycle_index = int(kwargs["cycle_index"])
        base_seed = candidate_noise_seed(
            global_seed,
            path_index,
            cycle_index,
            0,
            num_paths,
            num_cycles,
            num_samples,
        )
        final_seed = candidate_noise_seed(
            global_seed,
            path_index,
            cycle_index,
            requested_samples - 1,
            num_paths,
            num_cycles,
            num_samples,
        )
        encoded = dict(kwargs)
        encoded.update(
            {
                "global_seed": base_seed,
                "path_index": 0,
                "cycle_index": 0,
                "num_samples": requested_samples,
            }
        )
        generated, generated_seeds = original_generate(**encoded)
        expected_seeds = list(range(base_seed, final_seed + 1))
        if len(generated) != requested_samples or generated_seeds != expected_seeds:
            raise AssertionError(
                "Repository sampler did not preserve the encoded candidate-seed block"
            )
        for sample_index, (sample, encoded_seed) in enumerate(
            zip(generated, generated_seeds)
        ):
            sample_recorder.record(
                global_seed=int(kwargs["global_seed"]),
                path_index=int(kwargs["path_index"]),
                cycle_index=int(kwargs["cycle_index"]),
                sample_index=sample_index,
                encoded_seed=int(encoded_seed),
                sample=sample,
            )
        return generated, generated_seeds

    rollout.fk_positions = counter
    action_buffer_diagnostic.fk_positions = counter
    rollout.build_v5b_condition = instrumented_condition
    rollout.full_trajectory_metrics = instrumented_full_metrics
    rollout.extension_preview = guard
    rollout.generate_seeded_samples = instrumented_generate
    try:
        yield counter, guard, sample_recorder
    finally:
        rollout.fk_positions = original_fk_rollout
        action_buffer_diagnostic.fk_positions = original_fk_diagnostic
        rollout.build_v5b_condition = original_condition
        rollout.full_trajectory_metrics = original_full_metrics
        rollout.extension_preview = original_preview
        rollout.generate_seeded_samples = original_generate


def pristine_model_state(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {
        key: value.detach().clone()
        for key, value in model.state_dict().items()
    }


def reset_model(model: torch.nn.Module, pristine: Mapping[str, torch.Tensor]) -> None:
    model.load_state_dict(pristine, strict=True)
    model.eval()
    for parameter in model.parameters():
        parameter.grad = None


def assert_model_unchanged(
    model: torch.nn.Module, pristine: Mapping[str, torch.Tensor]
) -> None:
    current = model.state_dict()
    if current.keys() != pristine.keys():
        raise AssertionError("Model state keys changed during evaluation")
    for key, expected in pristine.items():
        if not torch.equal(current[key], expected):
            raise AssertionError(f"Model state changed during evaluation: {key}")


def _positive_integer_metadata_scalar(value: Any, *, label: str) -> int:
    """Read a required positive integer without silently truncating metadata."""

    array = np.asarray(value)
    if array.size != 1 or not np.issubdtype(array.dtype, np.integer):
        raise ValueError(
            f"{label} must be a scalar integer, got shape={array.shape}, "
            f"dtype={array.dtype}"
        )
    result = int(array.reshape(-1)[0])
    if result <= 0:
        raise ValueError(f"{label} must be positive, got {result}")
    return result


def resolve_window_stride(
    window_metadata: Mapping[str, Any],
    normalization_metadata: Mapping[str, Any],
) -> Tuple[int, str]:
    """Resolve stride from persisted window/statistics metadata and cross-check it."""

    candidates: List[Tuple[str, int]] = []
    if "stride" in window_metadata:
        candidates.append(
            (
                "window_npz",
                _positive_integer_metadata_scalar(
                    window_metadata["stride"],
                    label="window NPZ stride",
                ),
            )
        )
    if "stride" in normalization_metadata:
        candidates.append(
            (
                "normalization_stats",
                _positive_integer_metadata_scalar(
                    normalization_metadata["stride"],
                    label="normalization metadata stride",
                ),
            )
        )
    if not candidates:
        raise KeyError(
            "Window stride is absent from both the window NPZ and normalization "
            "metadata; reconstruction auditing will not assume stride one"
        )
    distinct_values = {value for _, value in candidates}
    if len(distinct_values) != 1:
        raise ValueError(
            "Window NPZ and normalization metadata disagree on stride: "
            + ", ".join(f"{source}={value}" for source, value in candidates)
        )
    return candidates[0][1], "+".join(source for source, _ in candidates)


def validate_window_artifact(
    window_path: Path,
    stats: Mapping[str, np.ndarray],
    expected_names: Sequence[str],
    desired_paths: np.ndarray,
    expert_q: np.ndarray,
    horizon: int,
    *,
    normalization_metadata: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Validate the held-out v5b window artifact used to define this model."""
    windows = rollout.load_npz(window_path, "v5b held-out windows")
    stride, stride_source = resolve_window_stride(
        windows,
        stats if normalization_metadata is None else normalization_metadata,
    )
    rollout.require_keys(
        windows,
        (
            "condition",
            "path_names",
            "window_start_indices",
            "prior_q_window",
            "desired_path_window",
            "expert_q_window",
        ),
        "v5b held-out windows",
    )
    condition = np.asarray(windows["condition"])
    if condition.ndim != 3 or condition.shape[1:] != (
        horizon,
        action_buffer_diagnostic.CONDITION_DIM,
    ):
        raise ValueError(
            "v5b window condition must have shape "
            f"(W,{horizon},{action_buffer_diagnostic.CONDITION_DIM}), got {condition.shape}"
        )
    window_count = int(condition.shape[0])
    names = rollout.decode_names(np.asarray(windows["path_names"]))
    starts = np.asarray(windows["window_start_indices"], dtype=np.int64)
    prior_windows = np.asarray(windows["prior_q_window"], dtype=np.float32)
    desired_windows = np.asarray(windows["desired_path_window"], dtype=np.float32)
    expert_windows = np.asarray(windows["expert_q_window"], dtype=np.float32)
    expected_shapes = {
        "path_names": (window_count,),
        "window_start_indices": (window_count,),
        "prior_q_window": (window_count, horizon, rollout.JOINT_DIM),
        "desired_path_window": (window_count, horizon, 3),
        "expert_q_window": (window_count, horizon, rollout.JOINT_DIM),
    }
    observed_shapes = {
        "path_names": (len(names),),
        "window_start_indices": starts.shape,
        "prior_q_window": prior_windows.shape,
        "desired_path_window": desired_windows.shape,
        "expert_q_window": expert_windows.shape,
    }
    for key, expected_shape in expected_shapes.items():
        if observed_shapes[key] != expected_shape:
            raise ValueError(
                f"v5b held-out windows {key} must have shape {expected_shape}, "
                f"got {observed_shapes[key]}"
            )
    for key, values in (
        ("condition", condition),
        ("prior_q_window", prior_windows),
        ("desired_path_window", desired_windows),
        ("expert_q_window", expert_windows),
    ):
        rollout.finite_array(values, f"v5b held-out windows {key}")

    name_to_index = {str(name): index for index, name in enumerate(expected_names)}
    unknown = sorted(set(names) - set(name_to_index))
    if unknown:
        raise ValueError(
            "The v5b test-window artifact contains unknown path identifiers: "
            + ", ".join(unknown[:10])
        )
    missing = sorted(set(expected_names) - set(names))
    if missing:
        raise ValueError(
            "The v5b test-window artifact does not cover held-out paths: "
            + ", ".join(missing[:10])
        )
    seen_windows: set[Tuple[str, int]] = set()
    trajectory_length = int(desired_paths.shape[1])
    for window_index, (path_name, start_value) in enumerate(zip(names, starts)):
        start = int(start_value)
        identity = (path_name, start)
        if identity in seen_windows:
            raise ValueError(
                f"The v5b test-window artifact duplicates {path_name} at start {start}"
            )
        seen_windows.add(identity)
        if start < 0 or start + horizon > trajectory_length:
            raise ValueError(
                f"The v5b test window {path_name} start={start} cannot provide H={horizon} "
                f"within trajectory length {trajectory_length}"
            )
        path_index = name_to_index[path_name]
        expected_desired = desired_paths[path_index, start : start + horizon]
        expected_expert = expert_q[path_index, start : start + horizon]
        if not np.array_equal(desired_windows[window_index], expected_desired):
            raise ValueError(
                f"v5b desired_path_window disagrees with the held-out test trajectory "
                f"for {path_name} at start {start}"
            )
        if not np.array_equal(expert_windows[window_index], expected_expert):
            raise ValueError(
                f"v5b expert_q_window disagrees with the held-out test trajectory "
                f"for {path_name} at start {start}"
            )
    for key in ("condition_mean", "condition_std", "residual_mean", "residual_std"):
        if key not in windows:
            raise KeyError(f"v5b held-out windows missing normalization field {key}")
        window_value = np.asarray(windows[key], dtype=np.float32)
        stats_value = np.asarray(stats[key], dtype=np.float32)
        if window_value.shape != stats_value.shape:
            raise ValueError(
                f"{window_path} and normalization stats have different shapes for "
                f"{key}: {window_value.shape} vs {stats_value.shape}"
            )
        if not np.allclose(window_value, stats_value, rtol=1e-6, atol=1e-7):
            raise ValueError(
                f"{window_path} and normalization stats disagree for {key}"
            )
    return {
        "path_names": names,
        "expected_path_names": [str(name) for name in expected_names],
        "window_start_indices": starts,
        "prior_q_window": prior_windows,
        "desired_path_window": desired_windows,
        "expert_q_window": expert_windows,
        "stride": stride,
        "stride_source": stride_source,
    }


def validate_prior_window_reconstruction(
    window_artifact: Mapping[str, Any],
    global_references: Mapping[str, np.ndarray],
    horizon: int,
) -> None:
    """Prove every stored training/evaluation prior window comes from the fixed CSV prior."""
    names = list(window_artifact["path_names"])
    starts = np.asarray(window_artifact["window_start_indices"], dtype=np.int64)
    prior_windows = np.asarray(window_artifact["prior_q_window"], dtype=np.float32)
    for window_index, (path_name, start_value) in enumerate(zip(names, starts)):
        if path_name not in global_references:
            raise KeyError(
                f"No fixed bootstrap prior CSV was loaded for window path {path_name}"
            )
        start = int(start_value)
        reference = np.asarray(global_references[path_name], dtype=np.float32)
        expected = reference[start : start + horizon]
        if expected.shape != (horizon, rollout.JOINT_DIM):
            raise ValueError(
                f"Fixed bootstrap prior for {path_name} cannot reconstruct start={start}, H={horizon}"
            )
        if not np.array_equal(prior_windows[window_index], expected):
            raise ValueError(
                f"v5b prior_q_window does not equal the fixed bootstrap prior CSV slice "
                f"for {path_name} at start {start}"
            )


def load_benchmark_assets(args: argparse.Namespace) -> Dict[str, Any]:
    device = rollout.resolve_device(args.device)
    normalization_metadata = rollout.load_npz(
        args.normalization_stats,
        "normalization metadata",
    )
    stats = rollout.load_stats(args.normalization_stats)
    residual_mean = np.asarray(stats["residual_mean"], dtype=np.float32)
    residual_std = np.asarray(stats["residual_std"], dtype=np.float32)
    condition_mean = np.asarray(stats["condition_mean"], dtype=np.float32)
    condition_std = np.asarray(stats["condition_std"], dtype=np.float32)
    if residual_mean.shape != (rollout.JOINT_DIM,) or residual_std.shape != (
        rollout.JOINT_DIM,
    ):
        raise ValueError("Residual normalization statistics must have shape (6,)")
    if condition_mean.shape != (action_buffer_diagnostic.CONDITION_DIM,) or condition_std.shape != (
        action_buffer_diagnostic.CONDITION_DIM,
    ):
        raise ValueError(
            "Condition normalization statistics must have shape "
            f"({action_buffer_diagnostic.CONDITION_DIM},)"
        )
    if np.any(residual_std <= 0.0) or np.any(condition_std <= 0.0):
        raise ValueError("Normalization standard deviations must be positive")

    data = rollout.load_npz(args.test_npz, "held-out test trajectories")
    rollout.require_keys(
        data,
        ("desired_paths", "expert_q", "q_start", "path_names"),
        "held-out test trajectories",
    )
    desired_paths = rollout.finite_array(
        data["desired_paths"], "desired_paths"
    ).astype(np.float32)
    expert_q = rollout.finite_array(data["expert_q"], "expert_q").astype(
        np.float32
    )
    q_start = rollout.finite_array(data["q_start"], "q_start").astype(
        np.float32
    )
    path_names = rollout.decode_names(data["path_names"])
    if desired_paths.ndim != 3 or desired_paths.shape[2] != 3:
        raise ValueError("desired_paths must have shape (N,T,3)")
    if expert_q.shape != (
        desired_paths.shape[0],
        desired_paths.shape[1],
        rollout.JOINT_DIM,
    ):
        raise ValueError("expert_q must have shape (N,T,6)")
    if q_start.shape != (desired_paths.shape[0], rollout.JOINT_DIM):
        raise ValueError("q_start must have shape (N,6)")
    if len(path_names) != desired_paths.shape[0] or len(set(path_names)) != len(path_names):
        raise ValueError("Test path identifiers must be unique and match trajectory count")
    window_artifact = validate_window_artifact(
        args.window_npz,
        stats,
        path_names,
        desired_paths,
        expert_q,
        args.prediction_horizon,
        normalization_metadata=normalization_metadata,
    )

    checkpoint = rollout.torch_load_checkpoint(args.diffusion_checkpoint, device)
    for key in ("condition_mean", "condition_std", "residual_mean", "residual_std"):
        if key not in checkpoint:
            raise KeyError(f"Diffusion checkpoint missing normalization field {key}")
        checkpoint_value = np.asarray(checkpoint[key], dtype=np.float32)
        stats_value = np.asarray(stats[key], dtype=np.float32)
        if checkpoint_value.shape != stats_value.shape:
            raise ValueError(
                f"Checkpoint and normalization stats have different shapes for {key}: "
                f"{checkpoint_value.shape} vs {stats_value.shape}"
            )
        rollout.finite_array(checkpoint_value, f"checkpoint {key}")
        if not np.allclose(checkpoint_value, stats_value, rtol=1e-6, atol=1e-7):
            raise ValueError(
                f"Checkpoint normalization field {key} disagrees with "
                f"{args.normalization_stats}"
            )
    if "test_npz" not in checkpoint:
        raise KeyError("Diffusion checkpoint missing test_npz provenance")
    checkpoint_test_npz = Path(str(checkpoint["test_npz"])).expanduser().resolve()
    requested_window_npz = args.window_npz.expanduser().resolve()
    if checkpoint_test_npz != requested_window_npz:
        raise ValueError(
            "Diffusion checkpoint test_npz provenance does not match --window_npz: "
            f"{checkpoint_test_npz} vs {requested_window_npz}"
        )
    if int(checkpoint["condition_dim"]) != action_buffer_diagnostic.CONDITION_DIM:
        raise ValueError(
            f"Checkpoint condition_dim must be {action_buffer_diagnostic.CONDITION_DIM}"
        )
    if int(checkpoint["target_dim"]) != rollout.JOINT_DIM:
        raise ValueError("Checkpoint target_dim must be six")
    if int(checkpoint["horizon"]) != args.prediction_horizon:
        raise ValueError("Checkpoint horizon differs from the frozen horizon")
    diffusion_metadata = checkpoint.get("diffusion_config")
    if not isinstance(diffusion_metadata, Mapping):
        raise ValueError("Checkpoint diffusion_config must be a mapping")
    expected_diffusion_metadata = {
        "prediction_type": "epsilon",
        "beta_schedule": "linear",
        "target_key": "residual_q_norm",
        "condition_key": "condition_norm",
    }
    for key, expected_value in expected_diffusion_metadata.items():
        if diffusion_metadata.get(key) != expected_value:
            raise ValueError(
                f"Checkpoint diffusion_config {key} must be {expected_value!r}, "
                f"got {diffusion_metadata.get(key)!r}"
            )
    if "num_diffusion_steps" not in checkpoint or "num_diffusion_steps" not in diffusion_metadata:
        raise KeyError("Checkpoint must record num_diffusion_steps consistently")
    checkpoint_steps = int(checkpoint["num_diffusion_steps"])
    metadata_steps = int(diffusion_metadata["num_diffusion_steps"])
    if checkpoint_steps <= args.t_init or metadata_steps != checkpoint_steps:
        raise ValueError(
            "Checkpoint diffusion step metadata is inconsistent or cannot support "
            f"frozen t_init={args.t_init}: checkpoint={checkpoint_steps}, config={metadata_steps}"
        )
    if args.num_diffusion_steps is not None and int(args.num_diffusion_steps) != checkpoint_steps:
        raise ValueError(
            "--num_diffusion_steps cannot override the checkpoint scheduler in the final benchmark"
        )
    for key in ("beta_start", "beta_end"):
        if key not in diffusion_metadata:
            raise KeyError(f"Checkpoint diffusion_config missing {key}")
        value = float(diffusion_metadata[key])
        if not math.isfinite(value) or not 0.0 < value < 1.0:
            raise ValueError(f"Checkpoint diffusion_config {key} must be finite and in (0,1)")
    if float(diffusion_metadata["beta_start"]) >= float(diffusion_metadata["beta_end"]):
        raise ValueError("Checkpoint beta_start must be smaller than beta_end")
    model, call_variant, _ = rollout.instantiate_checkpoint_model(checkpoint, device)
    model.eval()
    diffusion_config = rollout.diffusion_config_from_checkpoint(
        checkpoint, args.num_diffusion_steps
    )
    num_steps = int(diffusion_config["num_steps"])
    if not 0 <= args.t_init < num_steps:
        raise ValueError(f"t_init must be in [0,{num_steps - 1}]")
    schedule = rollout.make_schedule(
        num_steps,
        float(diffusion_config["beta_start"]),
        float(diffusion_config["beta_end"]),
        device,
    )

    robot, joint_names, ee_link = rollout.load_fk_context(None, None)
    if len(joint_names) != rollout.JOINT_DIM:
        raise ValueError("Exactly six active xMateCR7 joints are required")
    lower, upper = rollout.extract_joint_limits(robot, joint_names)
    return {
        "device": device,
        "stats": stats,
        "residual_mean": residual_mean,
        "residual_std": residual_std,
        "condition_mean": condition_mean,
        "condition_std": condition_std,
        "desired_paths": desired_paths,
        "expert_q": expert_q,
        "q_start": q_start,
        "path_names": path_names,
        "window_artifact": window_artifact,
        "model": model,
        "call_variant": call_variant,
        "schedule": schedule,
        "robot": robot,
        "joint_names": joint_names,
        "ee_link": ee_link,
        "lower": lower,
        "upper": upper,
        "drawing_weights": rollout.default_weights(),
        "resolved_urdf_descriptor": resolve_loaded_urdf_descriptor(robot),
    }


def load_global_reference(
    args: argparse.Namespace,
    path_name: str,
    trajectory_length: int,
) -> np.ndarray:
    reference = rollout.read_predicted_q_csv(
        args.prior_dir / rollout.safe_path_name(path_name) / "predicted_q.csv",
        expected_steps=trajectory_length,
    ).astype(np.float32)
    rollout.finite_array(reference, "fixed global bootstrap prior")
    if reference.shape != (trajectory_length, rollout.JOINT_DIM):
        raise ValueError(
            f"Global reference for {path_name} must have shape ({trajectory_length},6)"
        )
    reference.setflags(write=False)
    return reference


def _canonical_signature_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return {"__float__": "nan"}
        return {"__float__": "inf" if value > 0.0 else "-inf"}
    if isinstance(value, Mapping):
        return {
            str(key): _canonical_signature_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_signature_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(
        f"Unsupported candidate-signature value type: {type(value).__name__}"
    )


def _is_timing_field(name: str) -> bool:
    lowered = name.lower()
    return lowered in {"runtime_seconds", "elapsed_seconds"} or lowered.endswith(
        ("_runtime_ms", "_runtime_seconds", "_time_ms", "_time_seconds")
    )


def candidate_rows_signature(rows: Sequence[Mapping[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row_index, row in enumerate(rows):
        canonical = {
            str(key): _canonical_signature_value(value)
            for key, value in row.items()
            if not str(key).startswith("_") and not _is_timing_field(str(key))
        }
        payload = json.dumps(
            {"row_index": row_index, "fields": canonical},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        _update_length_prefixed_hash(digest, payload)
    return digest.hexdigest()


def run_signature(result: Mapping[str, Any]) -> Dict[str, Any]:
    planning = result["planning_rows"]
    return {
        "q": np.asarray(result["q"]),
        "ee": np.asarray(result["ee"]),
        "metrics": {
            key: result["metrics"].get(key)
            for key in rollout.FULL_METRICS
        },
        "generated_sample_count": int(result["generated_sample_count"]),
        "generated_sample_signature": str(result["generated_sample_signature"]),
        "candidate_row_count": int(result["candidate_row_count"]),
        "candidate_rows_signature": str(result["candidate_rows_signature"]),
        "cycles": [
            (
                int(row["planning_cycle_index"]),
                int(row["selected_candidate_index"]),
                float(row["selected_alpha"]),
                int(row["selected_candidate_seed"]),
                int(row["selected_diffusion"]),
                str(row["buffer_fallback_reason"]),
                int(row["accepted_diffusion_candidate_count"]),
                str(row["candidate_rejection_summary"]),
                float(row["buffer_larger_picture_score"]),
                float(row["selected_larger_picture_score"]),
            )
            for row in planning
        ],
    }


def assert_identical_rerun(
    first: Mapping[str, Any],
    repeated: Mapping[str, Any],
    path_name: str,
    seed: int,
) -> None:
    a = run_signature(first)
    b = run_signature(repeated)
    if not np.array_equal(a["q"], b["q"]):
        raise AssertionError(
            f"Determinism failure for {path_name}, seed {seed}: joint rollout differs"
        )
    if not np.array_equal(a["ee"], b["ee"]):
        raise AssertionError(
            f"Determinism failure for {path_name}, seed {seed}: FK rollout differs"
        )
    if (
        a["generated_sample_count"] != b["generated_sample_count"]
        or a["generated_sample_signature"] != b["generated_sample_signature"]
    ):
        raise AssertionError(
            f"Determinism failure for {path_name}, seed {seed}: "
            "generated residual/sample signature differs"
        )
    if (
        a["candidate_row_count"] != b["candidate_row_count"]
        or a["candidate_rows_signature"] != b["candidate_rows_signature"]
    ):
        raise AssertionError(
            f"Determinism failure for {path_name}, seed {seed}: "
            "full scored-candidate signature differs"
        )
    if a["metrics"] != b["metrics"] or a["cycles"] != b["cycles"]:
        raise AssertionError(
            f"Determinism failure for {path_name}, seed {seed}: metrics/selection differ"
        )


def run_one_rollout(
    *,
    args: argparse.Namespace,
    assets: Mapping[str, Any],
    pristine: Mapping[str, torch.Tensor],
    method: str,
    seed: int,
    path_index: int,
    path_name: str,
    desired_path: np.ndarray,
    expert_q_evaluation_only: np.ndarray,
    q_start: np.ndarray,
    global_reference: np.ndarray,
) -> Dict[str, Any]:
    reset_model(assets["model"], pristine)
    rollout.set_seed(seed)
    namespace = rollout_namespace(args, seed)
    score_config = rollout.score_weights(namespace)
    num_cycles = int(
        math.ceil(desired_path.shape[0] / args.execution_horizon)
    )
    with rollout_instrumentation(
        num_paths=len(assets["path_names"]),
        num_cycles=num_cycles,
        num_samples=args.num_base_samples,
    ) as (fk_counter, tail_guard, sample_recorder):
        start_time = time.perf_counter()
        with torch.no_grad():
            result = rollout.run_method(
                method=method,
                path_index=path_index,
                path_name=path_name,
                desired_path=desired_path,
                expert_q_evaluation_only=expert_q_evaluation_only,
                q_start=q_start,
                global_reference=global_reference,
                model=assets["model"],
                call_variant=assets["call_variant"],
                schedule=assets["schedule"],
                residual_mean=assets["residual_mean"],
                residual_std=assets["residual_std"],
                condition_mean=assets["condition_mean"],
                condition_std=assets["condition_std"],
                robot=assets["robot"],
                joint_names=assets["joint_names"],
                ee_link=assets["ee_link"],
                lower=assets["lower"],
                upper=assets["upper"],
                drawing_weights=assets["drawing_weights"],
                score_config=score_config,
                args=namespace,
            )
        elapsed = time.perf_counter() - start_time
    assert_model_unchanged(assets["model"], pristine)
    result["generated_sample_count"] = int(sample_recorder.sample_count)
    result["generated_sample_signature"] = sample_recorder.hexdigest()
    result["candidate_row_count"] = len(result["candidate_rows"])
    result["candidate_rows_signature"] = candidate_rows_signature(
        result["candidate_rows"]
    )
    planning_rows = result["planning_rows"]
    generated_candidates = int(
        sum(max(0, int(row["candidate_count"]) - 1) for row in planning_rows)
    )
    generated_samples = (
        args.num_base_samples * len(planning_rows) if method != "buffer_only" else 0
    )
    result["metrics"].update(
        {
            "runtime_seconds": float(elapsed),
            "num_fk_evaluations": int(fk_counter.configuration_count),
            "num_fk_batches": int(fk_counter.call_count),
            "num_fk_configurations": int(fk_counter.configuration_count),
            "num_generated_candidates": generated_candidates,
            "num_generated_residual_samples": generated_samples,
            "base_tail_invariant_check_count": int(tail_guard.check_count),
            "model_state_reset_verified": 1,
            "action_buffer_reset_verified": 1,
        }
    )
    output_method = "base_tail_diffusion" if method == "base_tail" else method
    for row in planning_rows:
        cycle = int(row["planning_cycle_index"])
        row["rollout_method"] = output_method
        row["seed"] = int(seed) if method != "buffer_only" else -1
        row["global_seed"] = int(seed)
        row["path_index"] = int(path_index)
        row["fk_evaluations"] = int(fk_counter.cycle_configurations[cycle])
        row["fk_batches"] = int(fk_counter.cycle_calls[cycle])
        row["fk_configurations"] = int(fk_counter.cycle_configurations[cycle])
        row["generated_candidate_count"] = max(
            0, int(row["candidate_count"]) - 1
        )
        row["generated_residual_sample_count"] = (
            args.num_base_samples if method != "buffer_only" else 0
        )
        row["candidate_seed_formula"] = (
            "(((global_seed*num_test_paths)+path_index)*num_planning_cycles+"
            "planning_cycle_index)*num_base_samples+candidate_sample_index"
        )
        row["frozen_rollout_settings"] = json.dumps(
            FROZEN_ROLLOUT_SETTINGS, sort_keys=True
        )
        row["base_tail_tail_exclusion_asserted"] = int(
            row.get("selected_base_tail_tail_exclusion_asserted", 0)
        )
    result["method"] = output_method
    return result


def computed_result_row(
    *,
    result: Mapping[str, Any],
    path_name: str,
    path_index: int,
    seed: int,
    determinism_verified: bool,
    order_independence_verified: bool,
) -> Dict[str, Any]:
    return {
        "path_name": path_name,
        "path_index": int(path_index),
        "seed": int(seed),
        "method": str(result["method"]),
        "result_source": "computed_by_final_benchmark",
        "source_csv": "",
        "prediction_horizon": FROZEN_ARCHITECTURE["prediction_horizon"],
        "execution_horizon": FROZEN_ARCHITECTURE["execution_horizon"],
        "t_init": FROZEN_ARCHITECTURE["t_init"],
        "num_base_samples": FROZEN_ARCHITECTURE["num_base_samples"],
        "alphas": json.dumps(FROZEN_ARCHITECTURE["alphas"]),
        "ramp_length": FROZEN_ARCHITECTURE["ramp_length"],
        "taper_mode": FROZEN_ARCHITECTURE["taper_mode"],
        "tail_mode": FROZEN_ARCHITECTURE["tail_mode"],
        "tail_extension": FROZEN_ARCHITECTURE["tail_extension"],
        "selector": FROZEN_ARCHITECTURE["selector"],
        "frozen_rollout_settings": json.dumps(
            FROZEN_ROLLOUT_SETTINGS, sort_keys=True
        ),
        "determinism_rerun_verified": int(determinism_verified),
        "seed_order_independence_verified": int(order_independence_verified),
        **result["metrics"],
    }


def require_columns(
    rows: Sequence[Mapping[str, str]],
    path: Path,
    required: Sequence[str],
) -> None:
    columns = set(rows[0])
    missing = [name for name in required if name not in columns]
    if missing:
        raise ValueError(
            f"Existing baseline {path} is missing required columns: {missing}"
        )


def discover_adaptive_csv(args: argparse.Namespace) -> Path:
    if args.adaptive_mlp_ik_csv is not None:
        if not args.adaptive_mlp_ik_csv.exists():
            raise FileNotFoundError(args.adaptive_mlp_ik_csv)
        return args.adaptive_mlp_ik_csv
    preferred = args.baseline_root / "mlp_ik_refine_test_summary_adaptive.csv"
    if preferred.exists():
        return preferred
    candidates = sorted(args.baseline_root.glob("mlp_ik_refine_test_summary*.csv"))
    compatible: List[Path] = []
    for candidate in candidates:
        rows = read_records_csv(candidate)
        if {
            "path_id",
            "after_path_error",
            "after_mean_error",
            "after_max_error",
        }.issubset(rows[0]):
            compatible.append(candidate)
    if len(compatible) != 1:
        raise ValueError(
            "Adaptive MLP+IK baseline discovery is ambiguous or empty. "
            "Supply --adaptive_mlp_ik_csv. Candidates: "
            + ", ".join(str(path) for path in compatible)
        )
    return compatible[0]


def inspect_diffusion_v1_candidate(path: Path) -> Dict[str, Any]:
    """Validate and summarize one existing diffusion-v1 per-path K result."""
    rows = read_records_csv(path)
    required = (
        "path_name",
        "path_error",
        "mean_error",
        "max_error",
        "total_cost",
        "output_folder",
    )
    require_columns(rows, path, required)
    seen: set[str] = set()
    total_costs: List[float] = []
    for row_index, row in enumerate(rows, start=2):
        path_name = str(row.get("path_name", ""))
        if not path_name or path_name != path_name.strip():
            raise ValueError(
                f"Diffusion-v1 candidate {path} has an empty or non-exact path_name "
                f"on CSV row {row_index}"
            )
        if path_name in seen:
            raise ValueError(
                f"Diffusion-v1 candidate {path} duplicates exact path ID {path_name}"
            )
        seen.add(path_name)
        total_cost = finite_float(row.get("total_cost"))
        if total_cost is None:
            raise ValueError(
                f"Diffusion-v1 candidate {path} has non-finite total_cost for {path_name}"
            )
        total_costs.append(float(total_cost))
    if not total_costs:
        raise ValueError(f"Diffusion-v1 candidate {path} has no path-level total_cost values")
    return {
        "path": path,
        "k": path.parent.name,
        "path_count": len(rows),
        "mean_total_cost": float(np.mean(np.asarray(total_costs, dtype=np.float64))),
        "sha256": file_sha256(path),
    }


def discover_diffusion_v1_csv(args: argparse.Namespace) -> Tuple[Path, Dict[str, Any]]:
    if args.diffusion_v1_best_of_k_csv is not None:
        if not args.diffusion_v1_best_of_k_csv.exists():
            raise FileNotFoundError(args.diffusion_v1_best_of_k_csv)
        try:
            selected = inspect_diffusion_v1_candidate(args.diffusion_v1_best_of_k_csv)
        except (KeyError, ValueError) as exc:
            raise ValueError(
                "Explicit --diffusion_v1_best_of_k_csv is not a compatible "
                "path-level best-of-K result"
            ) from exc
        inventory = [
            {
                key: str(value) if key == "path" else value
                for key, value in selected.items()
            }
        ]
        return args.diffusion_v1_best_of_k_csv, {
            "selection_criterion": "explicit_cli_override",
            "selected_k": selected["k"],
            "selected_mean_total_cost": selected["mean_total_cost"],
            "selected_sha256": selected["sha256"],
            "candidate_inventory": inventory,
        }

    ranked_root = args.baseline_root / "diffusion_v1_ranked_samples"
    candidate_paths = sorted(
        ranked_root.glob("k*/diffusion_v1_best_per_path.csv"),
        key=lambda value: str(value),
    )
    if not candidate_paths:
        raise ValueError(
            "No diffusion-v1 k*/diffusion_v1_best_per_path.csv candidates were found; "
            "supply --diffusion_v1_best_of_k_csv"
        )
    candidates: List[Dict[str, Any]] = []
    for candidate_path in candidate_paths:
        try:
            candidates.append(inspect_diffusion_v1_candidate(candidate_path))
        except (KeyError, ValueError) as exc:
            raise ValueError(
                f"Automatic diffusion-v1 discovery found an incompatible candidate "
                f"{candidate_path}; supply --diffusion_v1_best_of_k_csv to resolve ambiguity"
            ) from exc
    best_mean = min(float(candidate["mean_total_cost"]) for candidate in candidates)
    selected_candidates = [
        candidate
        for candidate in candidates
        if math.isclose(
            float(candidate["mean_total_cost"]),
            best_mean,
            rel_tol=1e-12,
            abs_tol=1e-15,
        )
    ]
    if len(selected_candidates) != 1:
        tied = ", ".join(
            f"{candidate['k']}={float(candidate['mean_total_cost']):.12g}"
            for candidate in selected_candidates
        )
        raise ValueError(
            "Diffusion-v1 best-of-K discovery has a lowest-mean total_cost tie "
            f"({tied}); supply --diffusion_v1_best_of_k_csv"
        )
    selected = selected_candidates[0]
    inventory = [
        {
            key: str(value) if key == "path" else value
            for key, value in candidate.items()
        }
        for candidate in candidates
    ]
    return Path(selected["path"]), {
        "selection_criterion": "unique_lowest_mean_existing_path_total_cost",
        "selected_k": selected["k"],
        "selected_mean_total_cost": selected["mean_total_cost"],
        "selected_sha256": selected["sha256"],
        "candidate_inventory": inventory,
    }


def discover_expert_csv(args: argparse.Namespace) -> Path:
    if args.expert_ik_csv is not None:
        if not args.expert_ik_csv.exists():
            raise FileNotFoundError(args.expert_ik_csv)
        return args.expert_ik_csv
    preferred = args.baseline_root / "experts/ik_generation_summary.csv"
    if preferred.exists():
        return preferred
    candidates = sorted(args.baseline_root.glob("**/ik_generation_summary.csv"))
    compatible: List[Path] = []
    for candidate in candidates:
        rows = read_records_csv(candidate)
        if {"path_id", "mean_error", "max_error", "path_error"}.issubset(rows[0]):
            compatible.append(candidate)
    if len(compatible) != 1:
        raise ValueError(
            "Expert-IK baseline discovery is ambiguous or empty. Supply "
            "--expert_ik_csv. Candidates: "
            + ", ".join(str(path) for path in compatible)
        )
    return compatible[0]


def source_cartesian_metrics(
    row: Mapping[str, str], method: str
) -> Dict[str, float]:
    if method == "adaptive_mlp_ik":
        mean_value = finite_float(row.get("after_mean_error"))
        squared_value = finite_float(row.get("after_path_error"))
        max_value = finite_float(row.get("after_max_error"))
    else:
        mean_value = finite_float(row.get("mean_error"))
        squared_value = finite_float(row.get("path_error"))
        max_value = finite_float(row.get("max_error"))
    if mean_value is None or squared_value is None or max_value is None:
        raise ValueError(f"{method} baseline has non-finite Cartesian metrics")
    if squared_value < 0.0:
        raise ValueError(f"{method} path_error cannot be negative")
    return {
        "mean_cartesian_error": mean_value,
        "rms_cartesian_error": float(math.sqrt(squared_value)),
        "max_cartesian_error": max_value,
    }


def baseline_source_rows(
    args: argparse.Namespace,
) -> Tuple[Dict[str, List[Dict[str, Any]]], Dict[str, Dict[str, Any]]]:
    adaptive_path = discover_adaptive_csv(args)
    diffusion_path, diffusion_discovery = discover_diffusion_v1_csv(args)
    expert_path = discover_expert_csv(args)
    paths = {
        "adaptive_mlp_ik": adaptive_path,
        "diffusion_v1_best_of_k": diffusion_path,
        "expert_ik": expert_path,
    }
    output: Dict[str, List[Dict[str, Any]]] = {}
    provenance: Dict[str, Dict[str, Any]] = {}
    for method, path in paths.items():
        raw_rows = read_records_csv(path)
        if method == "adaptive_mlp_ik":
            require_columns(
                raw_rows,
                path,
                (
                    "path_id",
                    "after_path_error",
                    "after_mean_error",
                    "after_max_error",
                ),
            )
            path_key = "path_id"
        elif method == "diffusion_v1_best_of_k":
            require_columns(
                raw_rows,
                path,
                ("path_name", "path_error", "mean_error", "max_error", "output_folder"),
            )
            path_key = "path_name"
        else:
            require_columns(
                raw_rows,
                path,
                ("path_id", "path_error", "mean_error", "max_error"),
            )
            path_key = "path_id"
            if "split" in raw_rows[0]:
                raw_rows = [
                    row for row in raw_rows if str(row.get("split", "")).strip() == "test"
                ]
                if not raw_rows:
                    raise ValueError(f"Expert baseline {path} contains no test rows")
        canonical: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for raw in raw_rows:
            path_name = str(raw.get(path_key, "")).strip()
            if not path_name:
                raise ValueError(f"{path} contains an empty path identifier")
            if path_name in seen:
                raise ValueError(f"{path} contains duplicate path identifier {path_name}")
            seen.add(path_name)
            canonical.append(
                {
                    "path_name": path_name,
                    "method": method,
                    "source_row": raw,
                    "source_metrics": source_cartesian_metrics(raw, method),
                }
            )
        output[method] = canonical
        provenance[method] = {
            "source_type": "previously_generated_csv",
            "source_csv": str(path),
            "source_sha256": file_sha256(path),
            "source_path_count": len(canonical),
            "selection": "explicit_cli" if getattr(
                args,
                {
                    "adaptive_mlp_ik": "adaptive_mlp_ik_csv",
                    "diffusion_v1_best_of_k": "diffusion_v1_best_of_k_csv",
                    "expert_ik": "expert_ik_csv",
                }[method],
            ) is not None else "automatic",
            "diffusion_k": diffusion_discovery["selected_k"]
            if method == "diffusion_v1_best_of_k"
            else "",
            "diffusion_selection_criterion": diffusion_discovery["selection_criterion"]
            if method == "diffusion_v1_best_of_k"
            else "",
            "diffusion_selected_mean_total_cost": diffusion_discovery[
                "selected_mean_total_cost"
            ]
            if method == "diffusion_v1_best_of_k"
            else "",
            "diffusion_selected_sha256": diffusion_discovery["selected_sha256"]
            if method == "diffusion_v1_best_of_k"
            else "",
            "diffusion_candidate_inventory": json.dumps(
                diffusion_discovery["candidate_inventory"], sort_keys=True
            )
            if method == "diffusion_v1_best_of_k"
            else "",
        }
    return output, provenance


def external_trajectory_metrics(
    *,
    q: np.ndarray,
    desired_path: np.ndarray,
    expert_q: np.ndarray,
    q_start: np.ndarray,
    global_reference: np.ndarray,
    assets: Mapping[str, Any],
) -> Dict[str, Any]:
    region, ee, _ = rollout.sequence_metrics(
        q=q,
        desired=desired_path,
        previous_q=q_start,
        robot=assets["robot"],
        joint_names=assets["joint_names"],
        ee_link=assets["ee_link"],
        lower=assets["lower"],
        upper=assets["upper"],
        drawing_weights=assets["drawing_weights"],
    )
    global_ee = rollout.fk_positions(
        assets["robot"],
        assets["joint_names"],
        assets["ee_link"],
        global_reference,
    )
    joint_drift = np.linalg.norm(q - global_reference, axis=1)
    cartesian_drift = np.linalg.norm(ee - global_ee, axis=1)
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
                "joint_limit_violation_count",
                "joint_limit_violation_magnitude",
            )
        },
        "full_joint_rmse_vs_expert": float(
            np.sqrt(np.mean(np.square(q - expert_q)))
        ),
        "mean_planning_boundary_discontinuity": float("nan"),
        "max_planning_boundary_discontinuity": float("nan"),
        "mean_future_shift_boundary_discontinuity": float("nan"),
        "max_future_shift_boundary_discontinuity": float("nan"),
        "total_global_reference_joint_drift": float(np.sum(joint_drift)),
        "mean_global_reference_joint_drift": float(np.mean(joint_drift)),
        "max_global_reference_joint_drift": float(np.max(joint_drift)),
        "total_global_reference_cartesian_drift": float(np.sum(cartesian_drift)),
        "mean_global_reference_cartesian_drift": float(np.mean(cartesian_drift)),
        "max_global_reference_cartesian_drift": float(np.max(cartesian_drift)),
        "extension_clipping_count": float("nan"),
        "extension_clipping_magnitude": float("nan"),
        "planning_cycle_count": float("nan"),
        "diffusion_selection_fraction": float("nan"),
        "unsafe_diffusion_selection_count": float("nan"),
        "buffer_fallback_count": float("nan"),
        "safety_improving_selection_count": float("nan"),
        "num_fk_evaluations": float("nan"),
        "num_fk_configurations": float("nan"),
        "num_generated_candidates": float("nan"),
        "num_generated_residual_samples": float("nan"),
    }
    return metrics


def trajectory_csv_for_baseline(
    method: str,
    record: Mapping[str, Any],
    args: argparse.Namespace,
) -> Optional[Path]:
    raw = record["source_row"]
    path_name = str(record["path_name"])
    if method == "adaptive_mlp_ik":
        return args.adaptive_mlp_ik_q_root / path_name / "refined_mlp_ik_q.csv"
    if method == "diffusion_v1_best_of_k":
        folder = str(raw.get("output_folder", "")).strip()
        return Path(folder) / "diffusion_pred_q.csv" if folder else None
    return None


def source_matches_recomputed(
    source: Mapping[str, float],
    computed: Mapping[str, Any],
    atol: float,
    rtol: float,
) -> Tuple[bool, str]:
    mismatches: List[str] = []
    for metric in (
        "mean_cartesian_error",
        "rms_cartesian_error",
        "max_cartesian_error",
    ):
        if not np.isclose(
            float(source[metric]),
            float(computed[metric]),
            atol=atol,
            rtol=rtol,
        ):
            mismatches.append(
                f"{metric}:csv={source[metric]:.9g},trajectory={float(computed[metric]):.9g}"
            )
    return not mismatches, ";".join(mismatches)


def build_external_result_rows(
    *,
    args: argparse.Namespace,
    assets: Mapping[str, Any],
    selected_names: Sequence[str],
    global_references: Mapping[str, np.ndarray],
    source_records: Mapping[str, Sequence[Mapping[str, Any]]],
    provenance: Mapping[str, Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    index_by_name = {name: index for index, name in enumerate(assets["path_names"])}
    selected = set(selected_names)
    per_seed_rows: List[Dict[str, Any]] = []
    method_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for method in ("adaptive_mlp_ik", "diffusion_v1_best_of_k", "expert_ik"):
        records = {str(row["path_name"]): row for row in source_records[method]}
        unknown = sorted(set(records) - set(index_by_name))
        if unknown:
            warnings.warn(
                f"{method} contains {len(unknown)} paths outside the held-out manifest; ignored",
                RuntimeWarning,
            )
        for path_name in selected_names:
            if path_name not in records:
                continue
            record = records[path_name]
            path_index = index_by_name[path_name]
            source_metrics = dict(record["source_metrics"])
            metrics: Dict[str, Any] = {
                metric: float("nan") for metric in RUN_METRICS
            }
            metrics.update(source_metrics)
            metric_status = {
                metric: (
                    "not_applicable"
                    if metric in EXTERNAL_PLANNING_ONLY_METRICS
                    else "missing/incompatible"
                )
                for metric in RUN_METRICS
            }
            metric_provenance = {
                metric: (
                    "not_applicable_to_external_method"
                    if metric in EXTERNAL_PLANNING_ONLY_METRICS
                    else "unavailable_or_incompatible_external_trajectory"
                )
                for metric in RUN_METRICS
            }
            for metric in source_metrics:
                metric_status[metric] = "source_csv"
                metric_provenance[metric] = provenance[method]["source_csv"]
            raw = record["source_row"]
            if method == "adaptive_mlp_ik":
                runtime = finite_float(raw.get("solve_time_sec"))
                if runtime is not None:
                    metrics["runtime_seconds"] = runtime
                    metric_status["runtime_seconds"] = "source_csv"
                    metric_provenance["runtime_seconds"] = provenance[method][
                        "source_csv"
                    ]
            compatibility = "csv_metrics_only"
            augmentation_note = ""
            q: Optional[np.ndarray] = None
            q_csv: Optional[Path] = None
            if method == "expert_ik":
                q = np.asarray(assets["expert_q"][path_index], dtype=np.float32)
                compatibility = "test_npz_expert_trajectory"
            elif not args.no_baseline_trajectory_augmentation:
                q_csv = trajectory_csv_for_baseline(method, record, args)
                if q_csv is not None and q_csv.exists():
                    q = rollout.read_predicted_q_csv(
                        q_csv,
                        expected_steps=assets["desired_paths"].shape[1],
                    ).astype(np.float32)
                else:
                    augmentation_note = f"trajectory CSV unavailable: {q_csv}"
            if q is not None:
                computed = external_trajectory_metrics(
                    q=q,
                    desired_path=assets["desired_paths"][path_index],
                    expert_q=assets["expert_q"][path_index],
                    q_start=assets["q_start"][path_index],
                    global_reference=global_references[path_name],
                    assets=assets,
                )
                compatible, mismatch = source_matches_recomputed(
                    source_metrics,
                    computed,
                    args.baseline_metric_atol,
                    args.baseline_metric_rtol,
                )
                if compatible:
                    metrics.update(computed)
                    # Preserve the exact values already reported by the source CSV
                    # while filling only metrics it did not contain.
                    metrics.update(source_metrics)
                    trajectory_provenance = (
                        str(q_csv) if q_csv is not None else "test_npz:expert_q"
                    )
                    for metric in RUN_METRICS:
                        if metric in source_metrics:
                            continue
                        if metric in EXTERNAL_PLANNING_ONLY_METRICS:
                            metric_status[metric] = "not_applicable"
                            metric_provenance[metric] = (
                                "not_applicable_to_external_method"
                            )
                        elif finite_float(computed.get(metric)) is not None:
                            metric_status[metric] = (
                                "computed_from_compatible_trajectory"
                            )
                            metric_provenance[metric] = trajectory_provenance
                    compatibility = "verified_and_augmented_from_existing_trajectory"
                else:
                    augmentation_note = (
                        "trajectory was not merged because source metrics differ: " + mismatch
                    )
                    compatibility = "csv_metrics_only_trajectory_incompatible"
            row = {
                "path_name": path_name,
                "path_index": path_index,
                "seed": -1,
                "method": method,
                "result_source": "previously_generated_csv",
                "source_csv": provenance[method]["source_csv"],
                "source_sha256": provenance[method]["source_sha256"],
                "baseline_selection_criterion": provenance[method].get(
                    "diffusion_selection_criterion", ""
                ),
                "baseline_candidate_inventory": provenance[method].get(
                    "diffusion_candidate_inventory", ""
                ),
                "baseline_selected_k": provenance[method].get("diffusion_k", ""),
                "baseline_selected_mean_total_cost": provenance[method].get(
                    "diffusion_selected_mean_total_cost", ""
                ),
                "source_q_csv": str(q_csv) if q_csv is not None else "test_npz:expert_q",
                "baseline_compatibility": compatibility,
                "baseline_augmentation_note": augmentation_note,
                "accepted": raw.get("accepted", ""),
                "selected_alpha_distribution": "",
                "selected_alpha_distribution_available": 0,
                "selected_alpha_distribution_availability_status": "not_applicable",
                "selected_alpha_distribution_provenance": (
                    "not_applicable_to_external_method"
                ),
                **metrics,
                **{
                    f"{metric}_available": int(
                        metric_status[metric] in EXTERNAL_AVAILABLE_STATUSES
                        and finite_float(metrics.get(metric)) is not None
                    )
                    for metric in RUN_METRICS
                },
                **{
                    f"{metric}_availability_status": metric_status[metric]
                    for metric in RUN_METRICS
                },
                **{
                    f"{metric}_provenance": metric_provenance[metric]
                    for metric in RUN_METRICS
                },
            }
            per_seed_rows.append(row)
            method_rows[method].append(row)
    missing_methods = [
        method
        for method in ("adaptive_mlp_ik", "diffusion_v1_best_of_k", "expert_ik")
        if not method_rows[method]
    ]
    if missing_methods:
        raise ValueError(
            "Required external baselines have no held-out path-level rows: "
            + ", ".join(missing_methods)
        )
    return per_seed_rows, method_rows


def validate_external_baseline_completeness(
    method_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    selected_names: Sequence[str],
) -> List[str]:
    methods = ("adaptive_mlp_ik", "diffusion_v1_best_of_k", "expert_ik")
    lookup = {
        method: {str(row["path_name"]): row for row in method_rows[method]}
        for method in methods
    }
    common_paths = [
        path_name
        for path_name in selected_names
        if all(path_name in lookup[method] for method in methods)
    ]
    if not common_paths:
        raise ValueError(
            "No exact held-out path identifiers are common to all external baselines"
        )

    failures: List[str] = []
    for method in methods:
        for path_name in common_paths:
            row = lookup[method][path_name]
            for metric in REQUIRED_EXTERNAL_COMPARISON_METRICS:
                status = str(row.get(f"{metric}_availability_status", ""))
                value = finite_float(row.get(metric))
                if status not in EXTERNAL_AVAILABLE_STATUSES or value is None:
                    provenance_value = row.get(f"{metric}_provenance", "")
                    failures.append(
                        f"method={method}, path={path_name}, metric={metric}, "
                        f"status={status or 'missing'}, provenance="
                        f"{provenance_value or 'missing'}"
                    )
    if failures:
        preview = "; ".join(failures[:20])
        remainder = len(failures) - min(len(failures), 20)
        suffix = f"; ... and {remainder} more" if remainder else ""
        raise ValueError(
            "External baseline comparison is incomplete on the exact common "
            f"held-out cohort ({len(common_paths)} paths): {preview}{suffix}"
        )
    return common_paths


def parse_alpha_distribution(value: Any) -> Counter[float]:
    counts: Counter[float] = Counter()
    if value in (None, ""):
        return counts
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid selected-alpha distribution: {value}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("Selected-alpha distribution must be a JSON object")
    for key, count in parsed.items():
        counts[float(key)] += int(count)
    return counts


def build_paired_path_summary(
    computed_rows: Sequence[Mapping[str, Any]],
    expected_seeds: Sequence[int],
) -> List[Dict[str, Any]]:
    buffers = {
        str(row["path_name"]): row
        for row in computed_rows
        if row["method"] == "buffer_only"
    }
    diffusion: Dict[str, Dict[int, Mapping[str, Any]]] = defaultdict(dict)
    for row in computed_rows:
        if row["method"] != "base_tail_diffusion":
            continue
        path_name = str(row["path_name"])
        seed = int(row["seed"])
        if seed in diffusion[path_name]:
            raise ValueError(f"Duplicate diffusion result for {path_name}, seed {seed}")
        diffusion[path_name][seed] = row
    if set(buffers) != set(diffusion):
        raise ValueError("Buffer and diffusion path sets differ")
    expected = list(int(seed) for seed in expected_seeds)
    output: List[Dict[str, Any]] = []
    for path_name, buffer in buffers.items():
        seed_rows = diffusion[path_name]
        if sorted(seed_rows) != sorted(expected):
            raise ValueError(
                f"Incomplete seed set for {path_name}: expected {expected}, got {sorted(seed_rows)}"
            )
        ordered = [seed_rows[seed] for seed in expected]
        row: Dict[str, Any] = {
            "path_name": path_name,
            "path_index": int(buffer["path_index"]),
            "seed_count": len(ordered),
            "expected_seeds": json.dumps(expected),
            "complete_seed_set": 1,
        }
        for metric in RUN_METRICS:
            baseline_value = finite_float(buffer.get(metric))
            candidate_values = [finite_float(seed_row.get(metric)) for seed_row in ordered]
            if baseline_value is None or any(value is None for value in candidate_values):
                continue
            values = np.asarray([float(value) for value in candidate_values], dtype=np.float64)
            mean_value = float(np.mean(values))
            median_value = float(np.median(values))
            std_value = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
            outcomes = [outcome(float(value), baseline_value) for value in values]
            counts = Counter(outcomes)
            worst_value = float(np.max(values))
            worst_outcome = outcome(worst_value, baseline_value)
            mean_outcome = outcome(mean_value, baseline_value)
            prefix = metric
            row.update(
                {
                    f"{prefix}_buffer": baseline_value,
                    f"{prefix}_seed_mean": mean_value,
                    f"{prefix}_seed_median": median_value,
                    f"{prefix}_seed_std": std_value,
                    f"{prefix}_seed_min": float(np.min(values)),
                    f"{prefix}_seed_max": float(np.max(values)),
                    f"{prefix}_mean_paired_difference": mean_value - baseline_value,
                    f"{prefix}_percentage_paired_difference": percentage_difference(
                        mean_value, baseline_value
                    ),
                    f"{prefix}_improved": int(mean_outcome == "improved"),
                    f"{prefix}_worsened": int(mean_outcome == "worsened"),
                    f"{prefix}_tied": int(mean_outcome == "tied"),
                    f"{prefix}_improved_seed_count": int(counts["improved"]),
                    f"{prefix}_worsened_seed_count": int(counts["worsened"]),
                    f"{prefix}_tied_seed_count": int(counts["tied"]),
                    f"{prefix}_worst_seed_improved": int(worst_outcome == "improved"),
                    f"{prefix}_worst_seed_worsened": int(worst_outcome == "worsened"),
                    f"{prefix}_all_seeds_improved": int(counts["improved"] == len(values)),
                    f"{prefix}_at_least_one_seed_improved": int(counts["improved"] > 0),
                    f"{prefix}_seed_improvement_sign_changes": int(
                        counts["improved"] > 0 and counts["worsened"] > 0
                    ),
                }
            )

        alpha_counts: Counter[float] = Counter()
        fallback_fractions: List[float] = []
        for seed_row in ordered:
            alpha_counts.update(
                parse_alpha_distribution(seed_row.get("selected_alpha_distribution", "{}"))
            )
            cycles = max(float(seed_row.get("planning_cycle_count", 0)), 1.0)
            fallback_fractions.append(
                float(seed_row.get("buffer_fallback_count", 0)) / cycles
            )
        row.update(
            {
                "selected_alpha_distribution": json.dumps(
                    {str(key): value for key, value in sorted(alpha_counts.items())},
                    sort_keys=True,
                ),
                "diffusion_selection_fraction_seed_mean": float(
                    np.mean(
                        [float(seed_row["diffusion_selection_fraction"]) for seed_row in ordered]
                    )
                ),
                "fallback_fraction_seed_mean": float(np.mean(fallback_fractions)),
                "extension_clipping_count_seed_total": int(
                    sum(int(seed_row["extension_clipping_count"]) for seed_row in ordered)
                ),
                "unsafe_diffusion_selection_count_seed_total": int(
                    sum(
                        int(seed_row["unsafe_diffusion_selection_count"])
                        for seed_row in ordered
                    )
                ),
                "safety_improving_selection_count_seed_total": int(
                    sum(
                        int(seed_row["safety_improving_selection_count"])
                        for seed_row in ordered
                    )
                ),
            }
        )
        output.append(row)
    return sorted(output, key=lambda item: int(item["path_index"]))


def diffusion_path_rows(
    paired_rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for paired in paired_rows:
        row: Dict[str, Any] = {
            "path_name": paired["path_name"],
            "path_index": paired["path_index"],
            "method": "base_tail_diffusion",
            "result_source": "seed_mean_computed_by_final_benchmark",
            "source_csv": "",
            "selected_alpha_distribution": paired["selected_alpha_distribution"],
            "fallback_fraction": paired["fallback_fraction_seed_mean"],
        }
        for metric in RUN_METRICS:
            key = f"{metric}_seed_mean"
            if key in paired:
                row[metric] = paired[key]
        output.append(row)
    return output


def buffer_path_rows(
    computed_rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    return [dict(row) for row in computed_rows if row["method"] == "buffer_only"]


def bootstrap_intervals(
    differences: np.ndarray,
    resamples: int,
    seed: int,
) -> Tuple[float, float, float, float]:
    if differences.ndim != 1 or differences.size == 0:
        raise ValueError("Paired bootstrap requires at least one path difference")
    if differences.size == 1:
        value = float(differences[0])
        return value, value, value, value
    rng = np.random.default_rng(seed)
    mean_samples = np.empty(resamples, dtype=np.float64)
    median_samples = np.empty(resamples, dtype=np.float64)
    batch = 1000
    cursor = 0
    while cursor < resamples:
        count = min(batch, resamples - cursor)
        indices = rng.integers(
            0, differences.size, size=(count, differences.size), endpoint=False
        )
        sampled = differences[indices]
        mean_samples[cursor : cursor + count] = np.mean(sampled, axis=1)
        median_samples[cursor : cursor + count] = np.median(sampled, axis=1)
        cursor += count
    mean_low, mean_high = np.percentile(mean_samples, (2.5, 97.5))
    median_low, median_high = np.percentile(median_samples, (2.5, 97.5))
    return (
        float(mean_low),
        float(mean_high),
        float(median_low),
        float(median_high),
    )


def benefit_rank_biserial(differences: np.ndarray, scipy_stats: Any) -> Tuple[float, float, float]:
    benefit = -np.asarray(differences, dtype=np.float64)
    nonzero = np.abs(benefit) > TIE_ABS_TOL
    if not np.any(nonzero):
        return 0.0, 0.0, 0.0
    ranks = scipy_stats.rankdata(np.abs(benefit[nonzero]), method="average")
    values = benefit[nonzero]
    improved_ranks = float(np.sum(ranks[values > 0.0]))
    worsened_ranks = float(np.sum(ranks[values < 0.0]))
    denominator = improved_ranks + worsened_ranks
    effect = (improved_ranks - worsened_ranks) / denominator if denominator else 0.0
    return effect, improved_ranks, worsened_ranks


def holm_adjust(rows: List[Dict[str, Any]]) -> None:
    finite = [
        (index, float(row["wilcoxon_p_value"]))
        for index, row in enumerate(rows)
        if finite_float(row.get("wilcoxon_p_value")) is not None
    ]
    ordered = sorted(finite, key=lambda item: item[1])
    running = 0.0
    total = len(ordered)
    for rank, (index, p_value) in enumerate(ordered):
        adjusted = min(1.0, (total - rank) * p_value)
        running = max(running, adjusted)
        rows[index]["wilcoxon_p_value_holm"] = running


def statistical_tests(
    paired_rows: Sequence[Mapping[str, Any]],
    args: argparse.Namespace,
    scipy_module: Any,
    scipy_stats: Any,
) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for metric in PRIMARY_METRICS:
        differences = np.asarray(
            [float(row[f"{metric}_mean_paired_difference"]) for row in paired_rows],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(differences)):
            raise ValueError(f"Non-finite paired differences for primary metric {metric}")
        analysis_seed = stable_analysis_seed(args.analysis_seed, metric)
        mean_low, mean_high, median_low, median_high = bootstrap_intervals(
            differences, args.bootstrap_resamples, analysis_seed
        )
        zeroed = differences.copy()
        zeroed[np.abs(zeroed) <= TIE_ABS_TOL] = 0.0
        if np.all(zeroed == 0.0):
            wilcoxon_statistic = 0.0
            wilcoxon_p = 1.0
            wilcoxon_status = "all_ties"
        else:
            try:
                test = scipy_stats.wilcoxon(
                    zeroed,
                    zero_method="pratt",
                    alternative="two-sided",
                    correction=False,
                    method="auto",
                )
            except Exception as exc:
                raise RuntimeError(
                    f"SciPy Wilcoxon failed for required metric {metric}"
                ) from exc
            wilcoxon_statistic = float(test.statistic)
            wilcoxon_p = float(test.pvalue)
            wilcoxon_status = "ok"
        effect, improved_rank_sum, worsened_rank_sum = benefit_rank_biserial(
            zeroed, scipy_stats
        )
        outcomes = [
            outcome(
                float(row[f"{metric}_seed_mean"]),
                float(row[f"{metric}_buffer"]),
            )
            for row in paired_rows
        ]
        counts = Counter(outcomes)
        sample_std = (
            float(np.std(differences, ddof=1)) if differences.size > 1 else 0.0
        )
        output.append(
            {
                "metric": metric,
                "difference_definition": "base_tail_seed_mean_minus_buffer",
                "negative_is_improvement": 1,
                "path_count": int(differences.size),
                "paired_mean_difference": float(np.mean(differences)),
                "paired_median_difference": float(np.median(differences)),
                "standard_deviation": sample_std,
                "standard_error": sample_std / math.sqrt(differences.size),
                "bootstrap_resamples": args.bootstrap_resamples,
                "bootstrap_seed": analysis_seed,
                "bootstrap_method": "paired_path_percentile",
                "bootstrap_mean_ci_low": mean_low,
                "bootstrap_mean_ci_high": mean_high,
                "bootstrap_median_ci_low": median_low,
                "bootstrap_median_ci_high": median_high,
                "wilcoxon_statistic": wilcoxon_statistic,
                "wilcoxon_p_value": wilcoxon_p,
                "wilcoxon_p_value_holm": float("nan"),
                "wilcoxon_zero_method": "pratt",
                "wilcoxon_status": wilcoxon_status,
                "scipy_version": scipy_module.__version__,
                "paired_rank_biserial_effect_size": effect,
                "effect_size_sign": "positive_favors_base_tail_diffusion",
                "improved_rank_sum": improved_rank_sum,
                "worsened_rank_sum": worsened_rank_sum,
                "improved_path_count": int(counts["improved"]),
                "worsened_path_count": int(counts["worsened"]),
                "tied_path_count": int(counts["tied"]),
                "percentage_improved": 100.0 * counts["improved"] / differences.size,
            }
        )
    holm_adjust(output)
    return output


def seed_stability_rows(
    computed_rows: Sequence[Mapping[str, Any]],
    expected_seeds: Sequence[int],
    args: argparse.Namespace,
    scipy_stats: Any,
) -> List[Dict[str, Any]]:
    buffers = {
        str(row["path_name"]): row
        for row in computed_rows
        if row["method"] == "buffer_only"
    }
    diffusion: Dict[int, Dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in computed_rows:
        if row["method"] == "base_tail_diffusion":
            diffusion[int(row["seed"])][str(row["path_name"])] = row
    metrics = tuple(dict.fromkeys((*QUALITY_METRICS, "runtime_seconds")))
    output: List[Dict[str, Any]] = []
    aggregate_cache: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for metric in metrics:
        for path_name, buffer in buffers.items():
            values = np.asarray(
                [float(diffusion[int(seed)][path_name][metric]) for seed in expected_seeds],
                dtype=np.float64,
            )
            baseline = float(buffer[metric])
            differences = values - baseline
            outcomes = [outcome(float(value), baseline) for value in values]
            counts = Counter(outcomes)
            mean_value = float(np.mean(values))
            std_value = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
            cv_defined = abs(mean_value) > EPS
            cv = std_value / abs(mean_value) if cv_defined else float("nan")
            sign_changes = counts["improved"] > 0 and counts["worsened"] > 0
            unstable_reasons: List[str] = []
            if sign_changes:
                unstable_reasons.append("improvement_sign_changes")
            if cv_defined and cv > args.unstable_cv_threshold:
                unstable_reasons.append("cv_threshold")
            path_row = {
                "record_type": "path_metric",
                "metric": metric,
                "path_name": path_name,
                "seed_count": len(values),
                "within_path_seed_mean": mean_value,
                "within_path_seed_std": std_value,
                "within_path_seed_min": float(np.min(values)),
                "within_path_seed_max": float(np.max(values)),
                "within_path_coefficient_of_variation": cv,
                "coefficient_of_variation_defined": int(cv_defined),
                "improvement_sign_changes": int(sign_changes),
                "all_seeds_improved": int(counts["improved"] == len(values)),
                "all_seeds_worsened": int(counts["worsened"] == len(values)),
                "unstable_path": int(bool(unstable_reasons)),
                "instability_reasons": ";".join(unstable_reasons),
                "seed_mean_paired_difference": float(np.mean(differences)),
                "max_absolute_seed_deviation": float(
                    np.max(np.abs(values - mean_value))
                ),
            }
            output.append(path_row)
            aggregate_cache[metric].append(path_row)

        metric_paths = aggregate_cache[metric]
        stds = np.asarray(
            [float(row["within_path_seed_std"]) for row in metric_paths],
            dtype=np.float64,
        )
        cvs = np.asarray(
            [
                float(row["within_path_coefficient_of_variation"])
                for row in metric_paths
                if int(row["coefficient_of_variation_defined"]) == 1
            ],
            dtype=np.float64,
        )
        output.append(
            {
                "record_type": "aggregate_metric",
                "metric": metric,
                "path_name": "__aggregate__",
                "path_count": len(metric_paths),
                "mean_within_path_seed_std": float(np.mean(stds)),
                "max_within_path_seed_std": float(np.max(stds)),
                "mean_within_path_coefficient_of_variation": float(np.mean(cvs))
                if cvs.size
                else float("nan"),
                "improvement_sign_change_path_count": int(
                    sum(int(row["improvement_sign_changes"]) for row in metric_paths)
                ),
                "improvement_sign_change_path_fraction": float(
                    np.mean([int(row["improvement_sign_changes"]) for row in metric_paths])
                ),
                "all_seeds_improved_path_fraction": float(
                    np.mean([int(row["all_seeds_improved"]) for row in metric_paths])
                ),
                "all_seeds_worsened_path_fraction": float(
                    np.mean([int(row["all_seeds_worsened"]) for row in metric_paths])
                ),
                "unstable_path_count": int(
                    sum(int(row["unstable_path"]) for row in metric_paths)
                ),
                "unstable_paths": json.dumps(
                    [row["path_name"] for row in metric_paths if int(row["unstable_path"])],
                    sort_keys=True,
                ),
            }
        )

        for index_a, seed_a in enumerate(expected_seeds):
            for seed_b in expected_seeds[index_a + 1 :]:
                paths = list(buffers)
                delta_a = np.asarray(
                    [
                        float(diffusion[int(seed_a)][name][metric])
                        - float(buffers[name][metric])
                        for name in paths
                    ],
                    dtype=np.float64,
                )
                delta_b = np.asarray(
                    [
                        float(diffusion[int(seed_b)][name][metric])
                        - float(buffers[name][metric])
                        for name in paths
                    ],
                    dtype=np.float64,
                )
                if len(paths) < 3 or np.std(delta_a) <= EPS or np.std(delta_b) <= EPS:
                    pearson = float("nan")
                    spearman = float("nan")
                    status = "undefined_constant_or_too_few_paths"
                else:
                    pearson = float(np.corrcoef(delta_a, delta_b)[0, 1])
                    spearman_result = scipy_stats.spearmanr(delta_a, delta_b)
                    spearman_value = getattr(spearman_result, "statistic", None)
                    if spearman_value is None:
                        spearman_value = getattr(spearman_result, "correlation", None)
                    if spearman_value is None:
                        try:
                            spearman_value = spearman_result[0]
                        except (IndexError, KeyError, TypeError) as exc:
                            raise RuntimeError(
                                "SciPy Spearman result exposes neither statistic nor correlation"
                            ) from exc
                    spearman = float(spearman_value)
                    if not math.isfinite(spearman):
                        raise RuntimeError(
                            "SciPy Spearman correlation was non-finite for non-constant seed deltas"
                        )
                    status = "ok"
                output.append(
                    {
                        "record_type": "seed_pair_correlation",
                        "metric": metric,
                        "path_name": "__aggregate__",
                        "seed_a": int(seed_a),
                        "seed_b": int(seed_b),
                        "common_path_count": len(paths),
                        "pearson_correlation_of_paired_differences": pearson,
                        "spearman_correlation_of_paired_differences": spearman,
                        "correlation_status": status,
                    }
                )
    return output


def finite_metric_values(
    rows_by_path: Mapping[str, Mapping[str, Any]],
    metric: str,
    paths: Sequence[str],
) -> Tuple[List[str], np.ndarray]:
    valid_paths: List[str] = []
    values: List[float] = []
    for path_name in paths:
        row = rows_by_path.get(path_name)
        if row is None:
            continue
        value = finite_float(row.get(metric))
        if value is None:
            continue
        valid_paths.append(path_name)
        values.append(value)
    return valid_paths, np.asarray(values, dtype=np.float64)


def aggregate_method_outputs(
    method_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    selected_order: Sequence[str],
    provenance: Mapping[str, Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
    required_methods = (
        "buffer_only",
        "base_tail_diffusion",
        "adaptive_mlp_ik",
        "diffusion_v1_best_of_k",
        "expert_ik",
    )
    lookup: Dict[str, Dict[str, Mapping[str, Any]]] = {
        method: {str(row["path_name"]): row for row in method_rows[method]}
        for method in required_methods
    }
    common = [
        name
        for name in selected_order
        if all(name in lookup[method] for method in required_methods)
    ]
    if not common:
        raise ValueError("No held-out path identifiers are common to all five methods")
    aggregate_rows: List[Dict[str, Any]] = []
    for method in required_methods:
        available_paths = [name for name in selected_order if name in lookup[method]]
        source = provenance.get(method, {})
        row: Dict[str, Any] = {
            "method": method,
            "path_count": len(available_paths),
            "common_path_count": len(common),
            "result_source": source.get(
                "source_type",
                "computed_by_final_benchmark",
            ),
            "source_csv": source.get("source_csv", ""),
        }
        for metric in RUN_METRICS:
            _, available_values = finite_metric_values(
                lookup[method], metric, available_paths
            )
            _, common_values = finite_metric_values(lookup[method], metric, common)
            common_complete = common_values.size == len(common)
            summary_values = (
                common_values
                if common_complete
                else np.asarray([], dtype=np.float64)
            )
            row[f"{metric}_available_path_count"] = int(available_values.size)
            row[f"{metric}_available_mean"] = (
                float(np.mean(available_values))
                if available_values.size
                else float("nan")
            )
            row[f"{metric}_available_median"] = (
                float(np.median(available_values))
                if available_values.size
                else float("nan")
            )
            row[f"{metric}_available_std"] = (
                float(np.std(available_values, ddof=1))
                if available_values.size > 1
                else 0.0
                if available_values.size
                else float("nan")
            )
            row[f"{metric}_available_min"] = (
                float(np.min(available_values))
                if available_values.size
                else float("nan")
            )
            row[f"{metric}_available_max"] = (
                float(np.max(available_values))
                if available_values.size
                else float("nan")
            )
            row[f"{metric}_path_count"] = int(summary_values.size)
            row[f"{metric}_mean"] = (
                float(np.mean(summary_values)) if summary_values.size else float("nan")
            )
            row[f"{metric}_median"] = (
                float(np.median(summary_values)) if summary_values.size else float("nan")
            )
            row[f"{metric}_std"] = (
                float(np.std(summary_values, ddof=1))
                if summary_values.size > 1
                else 0.0
                if summary_values.size
                else float("nan")
            )
            row[f"{metric}_min"] = (
                float(np.min(summary_values)) if summary_values.size else float("nan")
            )
            row[f"{metric}_max"] = (
                float(np.max(summary_values)) if summary_values.size else float("nan")
            )
            row[f"{metric}_common_path_count"] = int(common_values.size)
            row[f"{metric}_common_cohort_complete"] = int(common_complete)
            row[f"{metric}_common_mean"] = (
                float(np.mean(summary_values)) if summary_values.size else float("nan")
            )
            row[f"{metric}_common_median"] = (
                float(np.median(summary_values)) if summary_values.size else float("nan")
            )
        aggregate_rows.append(row)

    unified: List[Dict[str, Any]] = []
    for metric in RUN_METRICS:
        for method in required_methods:
            method_valid, method_values = finite_metric_values(lookup[method], metric, common)
            complete_common_metric_cohort = len(method_valid) == len(common)
            if (
                not complete_common_metric_cohort
                and metric in REQUIRED_EXTERNAL_COMPARISON_METRICS
            ):
                missing_paths = [
                    path_name for path_name in common if path_name not in method_valid
                ]
                raise ValueError(
                    "Required unified comparison metric is incomplete on the exact "
                    f"common cohort: method={method}, metric={metric}, "
                    f"missing_paths={missing_paths}"
                )
            if not complete_common_metric_cohort:
                method_valid = []
                method_values = np.asarray([], dtype=np.float64)
            method_map = {
                path: float(lookup[method][path][metric]) for path in method_valid
            }

            def relative_to(reference_method: str) -> Tuple[int, float, float, float]:
                reference_complete = all(
                    finite_float(lookup[reference_method][path].get(metric)) is not None
                    for path in common
                )
                if len(method_map) != len(common) or not reference_complete:
                    return 0, float("nan"), float("nan"), float("nan")
                pair_paths = list(common)
                candidate = np.asarray([method_map[path] for path in pair_paths])
                reference = np.asarray(
                    [float(lookup[reference_method][path][metric]) for path in pair_paths]
                )
                candidate_mean = float(np.mean(candidate))
                reference_mean = float(np.mean(reference))
                return (
                    len(pair_paths),
                    reference_mean,
                    candidate_mean - reference_mean,
                    percentage_difference(candidate_mean, reference_mean),
                )

            buffer_n, buffer_mean, buffer_diff, buffer_pct = relative_to("buffer_only")
            adaptive_n, adaptive_mean, adaptive_diff, adaptive_pct = relative_to(
                "adaptive_mlp_ik"
            )
            source = provenance.get(method, {})
            unified.append(
                {
                    "method": method,
                    "metric": metric,
                    "common_identity_path_count": len(common),
                    "complete_common_metric_cohort": int(
                        complete_common_metric_cohort
                    ),
                    "metric_availability_status": (
                        "available_complete_common_cohort"
                        if complete_common_metric_cohort
                        else "unavailable_partial_common_cohort"
                    ),
                    "metric_path_count": int(method_values.size),
                    "method_mean": float(np.mean(method_values))
                    if method_values.size
                    else float("nan"),
                    "method_median": float(np.median(method_values))
                    if method_values.size
                    else float("nan"),
                    "buffer_pair_path_count": buffer_n,
                    "buffer_mean_on_pair": buffer_mean,
                    "mean_difference_vs_buffer_only": buffer_diff,
                    "relative_change_vs_buffer_only_percent": buffer_pct,
                    "adaptive_pair_path_count": adaptive_n,
                    "adaptive_mean_on_pair": adaptive_mean,
                    "mean_difference_vs_adaptive_mlp_ik": adaptive_diff,
                    "relative_change_vs_adaptive_mlp_ik_percent": adaptive_pct,
                    "negative_change_is_improvement": int(metric in QUALITY_METRICS),
                    "result_source": source.get(
                        "source_type", "computed_by_final_benchmark"
                    ),
                    "source_csv": source.get("source_csv", ""),
                    "compatibility_note": (
                        "blank metric values mean the existing CSV/trajectory did not "
                        "provide a definition compatible with the final benchmark"
                    ),
                }
            )
    return aggregate_rows, unified, common


def failure_path_rankings(
    paired_rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    configurations = (
        (
            "largest_cartesian_improvements",
            "mean_cartesian_error",
            lambda row: float(row["mean_cartesian_error_mean_paired_difference"]),
            False,
        ),
        (
            "largest_cartesian_degradations",
            "mean_cartesian_error",
            lambda row: float(row["mean_cartesian_error_mean_paired_difference"]),
            True,
        ),
        (
            "largest_drawing_cost_improvements",
            "drawing_total_cost",
            lambda row: float(row["drawing_total_cost_mean_paired_difference"]),
            False,
        ),
        (
            "largest_drawing_cost_degradations",
            "drawing_total_cost",
            lambda row: float(row["drawing_total_cost_mean_paired_difference"]),
            True,
        ),
        (
            "largest_max_joint_step_increases",
            "max_joint_step",
            lambda row: float(row["max_joint_step_mean_paired_difference"]),
            True,
        ),
        (
            "highest_seed_variances",
            "mean_cartesian_error",
            lambda row: float(row["mean_cartesian_error_seed_std"]),
            True,
        ),
    )
    output: List[Dict[str, Any]] = []
    for category, metric, key, reverse in configurations:
        ranked = sorted(paired_rows, key=lambda row: (key(row), str(row["path_name"])), reverse=reverse)
        for rank, row in enumerate(ranked[:5], start=1):
            cycles = max(float(row.get("planning_cycle_count_seed_mean", 0.0)), 1.0)
            output.append(
                {
                    "ranking_category": category,
                    "rank": rank,
                    "path_name": row["path_name"],
                    "relevant_metric": metric,
                    "buffer_metric": row[f"{metric}_buffer"],
                    "diffusion_seed_mean": row[f"{metric}_seed_mean"],
                    "diffusion_seed_std": row[f"{metric}_seed_std"],
                    "paired_difference": row[f"{metric}_mean_paired_difference"],
                    "ranking_value": key(row),
                    "selected_alpha_distribution": row["selected_alpha_distribution"],
                    "diffusion_selection_fraction": row[
                        "diffusion_selection_fraction_seed_mean"
                    ],
                    "fallback_fraction": row["fallback_fraction_seed_mean"],
                    "extension_clipping_count": row[
                        "extension_clipping_count_seed_total"
                    ],
                    "unsafe_diffusion_selection_count": row[
                        "unsafe_diffusion_selection_count_seed_total"
                    ],
                    "safety_improving_selection_count": row[
                        "safety_improving_selection_count_seed_total"
                    ],
                    "max_joint_step_buffer": row["max_joint_step_buffer"],
                    "max_joint_step_diffusion_mean": row["max_joint_step_seed_mean"],
                    "joint_limit_violation_count_buffer": row[
                        "joint_limit_violation_count_buffer"
                    ],
                    "joint_limit_violation_count_diffusion_mean": row[
                        "joint_limit_violation_count_seed_mean"
                    ],
                    "planning_cycle_count_seed_mean": cycles,
                }
            )
    unsafe = sorted(
        (
            row
            for row in paired_rows
            if int(row["unsafe_diffusion_selection_count_seed_total"]) > 0
        ),
        key=lambda row: (
            int(row["unsafe_diffusion_selection_count_seed_total"]),
            str(row["path_name"]),
        ),
        reverse=True,
    )
    for rank, row in enumerate(unsafe[:5], start=1):
        output.append(
            {
                "ranking_category": "unsafe_cases",
                "rank": rank,
                "path_name": row["path_name"],
                "relevant_metric": "unsafe_diffusion_selection_count",
                "ranking_value": row["unsafe_diffusion_selection_count_seed_total"],
                "selected_alpha_distribution": row["selected_alpha_distribution"],
                "diffusion_selection_fraction": row[
                    "diffusion_selection_fraction_seed_mean"
                ],
                "fallback_fraction": row["fallback_fraction_seed_mean"],
                "extension_clipping_count": row[
                    "extension_clipping_count_seed_total"
                ],
                "unsafe_diffusion_selection_count": row[
                    "unsafe_diffusion_selection_count_seed_total"
                ],
                "safety_improving_selection_count": row[
                    "safety_improving_selection_count_seed_total"
                ],
            }
        )
    return output


def decision_record(
    order: int,
    criterion: str,
    measured: Any,
    threshold: str,
    status: str,
    explanation: str,
) -> Dict[str, Any]:
    if status not in {"PASS", "NOT_MET", "INCONCLUSIVE"}:
        raise ValueError(f"Unknown decision status {status}")
    return {
        "criterion_order": order,
        "criterion": criterion,
        "measured_result": measured,
        "threshold_or_interpretation": threshold,
        "status": status,
        "passed": 1 if status == "PASS" else 0 if status == "NOT_MET" else "",
        "explanation": explanation,
    }


def scientific_decision_rows(
    *,
    paired_rows: Sequence[Mapping[str, Any]],
    computed_rows: Sequence[Mapping[str, Any]],
    statistical_rows: Sequence[Mapping[str, Any]],
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    stats = {str(row["metric"]): row for row in statistical_rows}
    decisions: List[Dict[str, Any]] = []

    cart_diff = float(stats["mean_cartesian_error"]["paired_mean_difference"])
    cost_diff = float(stats["drawing_total_cost"]["paired_mean_difference"])
    decisions.append(
        decision_record(
            1,
            "mean Cartesian error is lower than buffer_only",
            cart_diff,
            "paired path mean difference < 0 (diffusion - buffer)",
            "PASS" if cart_diff < -TIE_ABS_TOL else "NOT_MET",
            "Negative differences indicate lower Cartesian error.",
        )
    )
    decisions.append(
        decision_record(
            2,
            "mean drawing total cost is lower than buffer_only",
            cost_diff,
            "paired path mean difference < 0 (diffusion - buffer)",
            "PASS" if cost_diff < -TIE_ABS_TOL else "NOT_MET",
            "Negative differences indicate lower drawing-aware cost.",
        )
    )
    for order, metric, description in (
        (3, "mean_cartesian_error", "more than 50% of paths improve in Cartesian error"),
        (4, "drawing_total_cost", "more than 50% of paths improve in drawing cost"),
    ):
        fraction = float(np.mean([int(row[f"{metric}_improved"]) for row in paired_rows]))
        decisions.append(
            decision_record(
                order,
                description,
                fraction,
                "strictly greater than 0.50; ties remain in denominator",
                "PASS" if fraction > 0.50 else "NOT_MET",
                f"Evaluated on {len(paired_rows)} paired held-out paths.",
            )
        )

    supported = [
        metric
        for metric, row in stats.items()
        if float(row["bootstrap_mean_ci_high"]) < 0.0
    ]
    decisions.append(
        decision_record(
            5,
            "95% bootstrap interval supports improvement in at least one primary metric",
            json.dumps(supported),
            "at least one primary mean-difference CI has upper bound < 0",
            "PASS" if supported else "NOT_MET",
            "Intervals resample paired paths after seed aggregation.",
        )
    )

    buffers = {
        str(row["path_name"]): row
        for row in computed_rows
        if row["method"] == "buffer_only"
    }
    diffusion = [row for row in computed_rows if row["method"] == "base_tail_diffusion"]
    violation_increases = sum(
        int(row["joint_limit_violation_count"])
        > int(buffers[str(row["path_name"])]["joint_limit_violation_count"])
        for row in diffusion
    )
    decisions.append(
        decision_record(
            6,
            "no increase in joint-limit violations",
            violation_increases,
            "zero path-seed observations with an increased violation count",
            "PASS" if violation_increases == 0 else "NOT_MET",
            "Compared every diffusion seed with its deterministic path buffer.",
        )
    )

    unsafe = int(sum(int(row["unsafe_diffusion_selection_count"]) for row in diffusion))
    selections = int(
        sum(
            round(
                float(row["diffusion_selection_fraction"])
                * float(row["planning_cycle_count"])
            )
            for row in diffusion
        )
    )
    safety_status = (
        "INCONCLUSIVE" if selections == 0 else "PASS" if unsafe == 0 else "NOT_MET"
    )
    decisions.append(
        decision_record(
            7,
            "no unsafe diffusion candidate selections",
            json.dumps({"unsafe": unsafe, "diffusion_selections": selections}),
            "unsafe count = 0 with at least one eligible diffusion selection",
            safety_status,
            "The existing candidate gates define unsafe selection status.",
        )
    )

    step_failures = 0
    maximum_step_ratio = 0.0
    for row in diffusion:
        baseline = float(buffers[str(row["path_name"])]["max_joint_step"])
        value = float(row["max_joint_step"])
        relative_limit = args.max_joint_step_relative_limit * max(abs(baseline), EPS)
        maximum_step_ratio = max(maximum_step_ratio, value / max(abs(baseline), EPS))
        if value > args.max_joint_step_absolute_limit or value > relative_limit:
            step_failures += 1
    decisions.append(
        decision_record(
            8,
            "maximum joint step remains within configured relative and absolute limits",
            json.dumps(
                {
                    "failed_path_seed_observations": step_failures,
                    "maximum_ratio_vs_buffer": maximum_step_ratio,
                }
            ),
            (
                f"all observations <= {args.max_joint_step_absolute_limit:g} and "
                f"<= {args.max_joint_step_relative_limit:g} * paired buffer"
            ),
            "PASS" if step_failures == 0 else "NOT_MET",
            "Both bounds are required for every path and seed.",
        )
    )

    smoothness_failures = 0
    for row in diffusion:
        baseline = buffers[str(row["path_name"])]
        acceleration = float(row["max_joint_acceleration"])
        jerk = float(row["max_joint_jerk"])
        if (
            acceleration > args.max_joint_acceleration_limit
            or jerk > args.max_joint_jerk_limit
            or acceleration
            > args.smoothness_relative_limit
            * max(abs(float(baseline["max_joint_acceleration"])), EPS)
            or jerk
            > args.smoothness_relative_limit
            * max(abs(float(baseline["max_joint_jerk"])), EPS)
        ):
            smoothness_failures += 1
    decisions.append(
        decision_record(
            9,
            "acceleration and jerk remain within configured limits",
            smoothness_failures,
            (
                f"zero observations above absolute acceleration={args.max_joint_acceleration_limit:g}, "
                f"jerk={args.max_joint_jerk_limit:g}, or relative factor="
                f"{args.smoothness_relative_limit:g}"
            ),
            "PASS" if smoothness_failures == 0 else "NOT_MET",
            "Both maximum acceleration and maximum jerk are checked per path/seed.",
        )
    )

    seed_metrics = [
        metric
        for metric, difference in (
            ("mean_cartesian_error", cart_diff),
            ("drawing_total_cost", cost_diff),
        )
        if difference < -TIE_ABS_TOL
    ]
    seed_stability_details: Dict[str, Any] = {}
    reproducibility_verified = all(
        int(row.get("determinism_rerun_verified", 0)) == 1
        and int(row.get("seed_order_independence_verified", 0)) == 1
        for row in diffusion
    )
    seed_stable = bool(seed_metrics) and len(args.seeds) >= 2 and reproducibility_verified
    for metric in seed_metrics if len(args.seeds) >= 2 else ():
        by_seed = {
            int(seed): float(
                np.mean(
                    [
                        float(row[metric])
                        - float(buffers[str(row["path_name"])][metric])
                        for row in diffusion
                        if int(row["seed"]) == int(seed)
                    ]
                )
            )
            for seed in args.seeds
        }
        leave_one_out = {
            int(seed): float(
                np.mean([value for other, value in by_seed.items() if other != int(seed)])
            )
            for seed in args.seeds
        }
        benefits = {seed: max(-value, 0.0) for seed, value in by_seed.items()}
        total_benefit = sum(benefits.values())
        contribution = (
            max(benefits.values()) / total_benefit if total_benefit > 0.0 else float("inf")
        )
        metric_stable = (
            all(value < -TIE_ABS_TOL for value in leave_one_out.values())
            and contribution <= args.max_seed_contribution_share
        )
        seed_stable = seed_stable and metric_stable
        seed_stability_details[metric] = {
            "per_seed_mean_differences": by_seed,
            "leave_one_seed_out_differences": leave_one_out,
            "maximum_positive_benefit_share": contribution,
            "passed": metric_stable,
        }
    seed_stability_details["full_reverse_order_rerun_verified"] = reproducibility_verified
    seed_decision_status = (
        "INCONCLUSIVE"
        if len(args.seeds) < 2 or not reproducibility_verified
        else "PASS"
        if seed_stable
        else "NOT_MET"
    )
    decisions.append(
        decision_record(
            10,
            "results are not dominated by one random seed",
            json.dumps(seed_stability_details, sort_keys=True),
            (
                "all improving decision metrics retain improvement when any seed is removed; "
                f"maximum benefit share <= {args.max_seed_contribution_share:g}"
            ),
            seed_decision_status,
            "Seed means use the same paired paths; seeds are not inferential units.",
        )
    )

    buffer_runtime = float(np.mean([float(row["runtime_seconds"]) for row in buffers.values()]))
    diffusion_runtime = float(np.mean([float(row["runtime_seconds"]) for row in diffusion]))
    runtime_ratio = diffusion_runtime / max(buffer_runtime, EPS)
    cart_buffer_mean = float(
        np.mean([float(row["mean_cartesian_error"]) for row in buffers.values()])
    )
    cost_buffer_mean = float(
        np.mean([float(row["drawing_total_cost"]) for row in buffers.values()])
    )
    cart_improvement_pct = -100.0 * cart_diff / max(abs(cart_buffer_mean), EPS)
    cost_improvement_pct = -100.0 * cost_diff / max(abs(cost_buffer_mean), EPS)
    practical_improvement = max(cart_improvement_pct, cost_improvement_pct)
    raw_improvement = max(
        -cart_diff if cart_improvement_pct >= cost_improvement_pct else -cost_diff,
        0.0,
    )
    extra_seconds = max(diffusion_runtime - buffer_runtime, 0.0)
    improvement_per_second = (
        raw_improvement / extra_seconds if extra_seconds > 0.0 else float("inf")
    )
    target_evidence: Dict[str, Dict[str, Any]] = {}
    qualifying_target_metrics: List[str] = []
    for metric in ("mean_cartesian_error", "drawing_total_cost"):
        metric_stats = stats[metric]
        ci_high = float(metric_stats["bootstrap_mean_ci_high"])
        benefit_effect = float(
            metric_stats["paired_rank_biserial_effect_size"]
        )
        qualifies = (
            ci_high < 0.0
            and benefit_effect >= args.min_benefit_rank_biserial
        )
        target_evidence[metric] = {
            "bootstrap_mean_ci_high": ci_high,
            "benefit_oriented_paired_rank_biserial": benefit_effect,
            "minimum_effect_threshold": args.min_benefit_rank_biserial,
            "qualifies": qualifies,
        }
        if qualifies:
            qualifying_target_metrics.append(metric)
    value_passed = (
        practical_improvement >= args.min_practical_improvement_percent
        and runtime_ratio <= args.max_runtime_ratio
        and improvement_per_second >= args.min_improvement_per_extra_second
        and selections > 0
        and bool(qualifying_target_metrics)
    )
    decisions.append(
        decision_record(
            11,
            "base_tail_diffusion adds measurable value relative to computational cost",
            json.dumps(
                {
                    "cartesian_improvement_percent": cart_improvement_pct,
                    "drawing_cost_improvement_percent": cost_improvement_pct,
                    "runtime_ratio": runtime_ratio,
                    "improvement_per_extra_second": improvement_per_second,
                    "diffusion_selection_count": selections,
                    "qualifying_target_metrics": qualifying_target_metrics,
                    "target_evidence": target_evidence,
                }
            ),
            (
                f"best practical improvement >= {args.min_practical_improvement_percent:g}%, "
                f"runtime ratio <= {args.max_runtime_ratio:g}, improvement/extra-second >= "
                f"{args.min_improvement_per_extra_second:g}, at least one diffusion selection, "
                "and mean Cartesian error or drawing cost has bootstrap CI upper bound < 0 "
                f"with benefit rank-biserial >= {args.min_benefit_rank_biserial:g}"
            ),
            "PASS" if value_passed else "NOT_MET",
            (
                "Computational value requires prespecified practical, runtime, confidence, "
                "and non-negligible benefit-effect evidence on a target metric."
            ),
        )
    )

    safety_failed = any(
        row["status"] == "NOT_MET" and int(row["criterion_order"]) in {6, 7, 8, 9}
        for row in decisions
    )
    any_not_met = any(row["status"] == "NOT_MET" for row in decisions)
    all_passed = all(row["status"] == "PASS" for row in decisions)
    overall = (
        "REJECT_SAFETY"
        if safety_failed
        else "RECOMMEND"
        if all_passed
        else "NOT_SUPPORTED"
        if any_not_met
        else "INCONCLUSIVE"
    )
    decisions.append(
        {
            "criterion_order": 12,
            "criterion": "OVERALL",
            "measured_result": overall,
            "threshold_or_interpretation": "all 11 prespecified criteria",
            "status": overall,
            "passed": 1 if overall == "RECOMMEND" else 0,
            "explanation": (
                "No superiority claim is warranted unless prespecified confidence, "
                "effect, stability, safety, and cost criteria all pass."
            ),
        }
    )
    return decisions


def _save_figure_impl(fig: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    # Close through pyplot imported by the caller; Figure.clear alone can leave
    # pyplot references alive during an all-path benchmark.
    import matplotlib.pyplot as plt

    plt.close(fig)


def save_figure(fig: Any, path: Path) -> None:
    _save_figure_impl(fig, path)
    _PUBLISHED_PLOT_PATHS.append(Path(path))


def paired_scatter_plot(
    plt: Any,
    paired_rows: Sequence[Mapping[str, Any]],
    metric: str,
    title: str,
    ylabel: str,
    output: Path,
) -> None:
    x = np.asarray([float(row[f"{metric}_buffer"]) for row in paired_rows])
    y = np.asarray([float(row[f"{metric}_seed_mean"]) for row in paired_rows])
    low = np.asarray([float(row[f"{metric}_seed_min"]) for row in paired_rows])
    high = np.asarray([float(row[f"{metric}_seed_max"]) for row in paired_rows])
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    ax.errorbar(
        x,
        y,
        yerr=np.vstack((y - low, high - y)),
        fmt="o",
        color=PLOT_COLORS["base_tail_diffusion"],
        ecolor="#BBBBBB",
        alpha=0.75,
        markersize=4,
        linewidth=0.8,
    )
    lower = float(min(np.min(x), np.min(low)))
    upper = float(max(np.max(x), np.max(high)))
    ax.plot([lower, upper], [lower, upper], "--", color="black", linewidth=1)
    ax.set_xlabel(f"buffer_only {ylabel}")
    ax.set_ylabel(f"base_tail_diffusion seed mean {ylabel}")
    ax.set_title(title)
    ax.grid(alpha=0.2)
    save_figure(fig, output)


def sorted_difference_plot(
    plt: Any,
    paired_rows: Sequence[Mapping[str, Any]],
    metric: str,
    title: str,
    output: Path,
) -> None:
    ordered = sorted(
        paired_rows,
        key=lambda row: float(row[f"{metric}_mean_paired_difference"]),
    )
    differences = np.asarray(
        [float(row[f"{metric}_mean_paired_difference"]) for row in ordered]
    )
    colors = ["#54A24B" if value < 0.0 else "#E45756" if value > 0.0 else "#999999" for value in differences]
    fig, ax = plt.subplots(figsize=(max(9.0, len(ordered) * 0.11), 5.2))
    ax.bar(np.arange(len(ordered)), differences, color=colors, width=0.9)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xlabel("held-out path (sorted best to worst)")
    ax.set_ylabel("diffusion seed mean - buffer (negative is better)")
    ax.set_title(title)
    ax.set_xticks([])
    ax.grid(axis="y", alpha=0.2)
    save_figure(fig, output)


def lookup_aggregate(
    aggregate_rows: Sequence[Mapping[str, Any]], method: str, metric: str
) -> float:
    row = next(item for item in aggregate_rows if item["method"] == method)
    value = finite_float(row.get(f"{metric}_mean"))
    return float(value) if value is not None else float("nan")


def save_required_plots(
    *,
    output_dir: Path,
    paired_rows: Sequence[Mapping[str, Any]],
    computed_rows: Sequence[Mapping[str, Any]],
    planning_rows: Sequence[Mapping[str, Any]],
    statistical_rows: Sequence[Mapping[str, Any]],
    aggregate_rows: Sequence[Mapping[str, Any]],
    seeds: Sequence[int],
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    paired_scatter_plot(
        plt,
        paired_rows,
        "mean_cartesian_error",
        "Paired mean Cartesian error",
        "error (m)",
        output_dir / "paired_cartesian_error_scatter.png",
    )
    paired_scatter_plot(
        plt,
        paired_rows,
        "drawing_total_cost",
        "Paired drawing-aware total cost",
        "cost",
        output_dir / "paired_drawing_cost_scatter.png",
    )

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
    for ax, metric, label in (
        (axes[0], "mean_cartesian_error", "Cartesian error difference (m)"),
        (axes[1], "drawing_total_cost", "drawing-cost difference"),
    ):
        values = [float(row[f"{metric}_mean_paired_difference"]) for row in paired_rows]
        ax.hist(values, bins=min(20, max(5, len(values) // 4)), color=PLOT_COLORS["base_tail_diffusion"], alpha=0.8)
        ax.axvline(0.0, color="black", linewidth=1)
        ax.set_xlabel(label + "\n(diffusion - buffer; negative is better)")
        ax.set_ylabel("path count")
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Paired path-difference distributions")
    save_figure(fig, output_dir / "paired_difference_distribution.png")

    stat_lookup = {str(row["metric"]): row for row in statistical_rows}
    relative_centers: List[float] = []
    relative_low: List[float] = []
    relative_high: List[float] = []
    labels: List[str] = []
    for metric in PRIMARY_METRICS:
        baseline = float(np.mean([float(row[f"{metric}_buffer"]) for row in paired_rows]))
        scale = max(abs(baseline), EPS)
        stat = stat_lookup[metric]
        relative_centers.append(100.0 * float(stat["paired_mean_difference"]) / scale)
        relative_low.append(100.0 * float(stat["bootstrap_mean_ci_low"]) / scale)
        relative_high.append(100.0 * float(stat["bootstrap_mean_ci_high"]) / scale)
        labels.append(metric.replace("_", " "))
    y = np.arange(len(labels))
    centers = np.asarray(relative_centers)
    fig, ax = plt.subplots(figsize=(8.4, 6.0))
    ax.errorbar(
        centers,
        y,
        xerr=np.vstack((centers - np.asarray(relative_low), np.asarray(relative_high) - centers)),
        fmt="o",
        color=PLOT_COLORS["base_tail_diffusion"],
        ecolor="#777777",
        capsize=3,
    )
    ax.axvline(0.0, color="black", linewidth=1)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("paired mean difference / buffer mean (%)\nnegative is better")
    ax.set_title("95% paired-path bootstrap confidence intervals")
    ax.grid(axis="x", alpha=0.2)
    save_figure(fig, output_dir / "primary_metric_confidence_intervals.png")

    sorted_difference_plot(
        plt,
        paired_rows,
        "mean_cartesian_error",
        "Per-path Cartesian difference",
        output_dir / "per_path_cartesian_difference_sorted.png",
    )
    sorted_difference_plot(
        plt,
        paired_rows,
        "drawing_total_cost",
        "Per-path drawing-cost difference",
        output_dir / "per_path_drawing_cost_difference_sorted.png",
    )

    variance_order = sorted(
        paired_rows,
        key=lambda row: float(row["mean_cartesian_error_seed_std"]),
        reverse=True,
    )
    fig, ax = plt.subplots(figsize=(max(9.0, len(variance_order) * 0.11), 4.8))
    ax.bar(
        np.arange(len(variance_order)),
        [float(row["mean_cartesian_error_seed_std"]) for row in variance_order],
        color="#72B7B2",
    )
    ax.set_xticks([])
    ax.set_xlabel("held-out path (sorted by seed standard deviation)")
    ax.set_ylabel("within-path Cartesian-error SD (m)")
    ax.set_title("Seed variance by path")
    ax.grid(axis="y", alpha=0.2)
    save_figure(fig, output_dir / "seed_variance.png")

    methods = [str(row["method"]) for row in aggregate_rows]
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    for ax, metric, label in (
        (axes[0], "mean_cartesian_error", "mean Cartesian error (m)"),
        (axes[1], "drawing_total_cost", "drawing total cost"),
    ):
        values = [lookup_aggregate(aggregate_rows, method, metric) for method in methods]
        positions = [index for index, value in enumerate(values) if math.isfinite(value)]
        ax.bar(
            positions,
            [values[index] for index in positions],
            color=[PLOT_COLORS.get(methods[index], "#999999") for index in positions],
        )
        ax.set_xticks(positions, [methods[index] for index in positions], rotation=35, ha="right")
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Method comparison (available compatible metrics)")
    save_figure(fig, output_dir / "method_comparison.png")

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    step_values = [lookup_aggregate(aggregate_rows, method, "max_joint_step") for method in methods]
    positions = [index for index, value in enumerate(step_values) if math.isfinite(value)]
    ax.bar(
        positions,
        [step_values[index] for index in positions],
        color=[PLOT_COLORS.get(methods[index], "#999999") for index in positions],
    )
    ax.set_xticks(positions, [methods[index] for index in positions], rotation=35, ha="right")
    ax.set_ylabel("mean of path maximum joint step")
    ax.set_title("Maximum-joint-step comparison")
    ax.grid(axis="y", alpha=0.2)
    save_figure(fig, output_dir / "maximum_joint_step_comparison.png")

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    for ax, metric, label in (
        (axes[0], "max_joint_acceleration", "path maximum acceleration"),
        (axes[1], "max_joint_jerk", "path maximum jerk"),
    ):
        values = [lookup_aggregate(aggregate_rows, method, metric) for method in methods]
        positions = [index for index, value in enumerate(values) if math.isfinite(value)]
        ax.bar(
            positions,
            [values[index] for index in positions],
            color=[PLOT_COLORS.get(methods[index], "#999999") for index in positions],
        )
        ax.set_xticks(positions, [methods[index] for index in positions], rotation=35, ha="right")
        ax.set_ylabel(label)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Acceleration and jerk comparison")
    save_figure(fig, output_dir / "acceleration_and_jerk_comparison.png")

    selection_order = sorted(
        paired_rows,
        key=lambda row: float(row["diffusion_selection_fraction_seed_mean"]),
        reverse=True,
    )
    fig, ax = plt.subplots(figsize=(max(9.0, len(selection_order) * 0.11), 4.8))
    ax.bar(
        np.arange(len(selection_order)),
        [float(row["diffusion_selection_fraction_seed_mean"]) for row in selection_order],
        color=PLOT_COLORS["base_tail_diffusion"],
    )
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks([])
    ax.set_xlabel("held-out path (sorted)")
    ax.set_ylabel("diffusion-selection fraction")
    ax.set_title("Diffusion-selection fraction by path")
    ax.grid(axis="y", alpha=0.2)
    save_figure(fig, output_dir / "diffusion_selection_fraction_by_path.png")

    alpha_counts: Counter[float] = Counter()
    for row in planning_rows:
        if row["rollout_method"] == "base_tail_diffusion" and int(row["selected_diffusion"]) == 1:
            alpha_counts[float(row["selected_alpha"])] += 1
    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    alpha_keys = sorted(float(value) for value in FROZEN_ARCHITECTURE["alphas"])
    ax.bar(
        [str(value) for value in alpha_keys],
        [alpha_counts[value] for value in alpha_keys],
        color="#B279A2",
    )
    ax.set_xlabel("selected residual alpha")
    ax.set_ylabel("planning-cycle count")
    ax.set_title("Selected-alpha distribution")
    ax.grid(axis="y", alpha=0.2)
    save_figure(fig, output_dir / "selected_alpha_distribution.png")

    buffer_lookup = {
        str(row["path_name"]): row
        for row in computed_rows
        if row["method"] == "buffer_only"
    }
    diffusion_lookup = {
        (str(row["path_name"]), int(row["seed"])): row
        for row in computed_rows
        if row["method"] == "base_tail_diffusion"
    }
    paths = [str(row["path_name"]) for row in paired_rows]
    matrix = np.zeros((len(seeds), len(paths)), dtype=np.float32)
    for seed_index, seed in enumerate(seeds):
        for path_index, path_name in enumerate(paths):
            status = outcome(
                float(diffusion_lookup[(path_name, int(seed))]["mean_cartesian_error"]),
                float(buffer_lookup[path_name]["mean_cartesian_error"]),
            )
            matrix[seed_index, path_index] = {
                "improved": -1.0,
                "tied": 0.0,
                "worsened": 1.0,
            }[status]
    fig, ax = plt.subplots(figsize=(max(10.0, len(paths) * 0.12), 4.5))
    image = ax.imshow(matrix, aspect="auto", cmap="RdYlGn_r", vmin=-1.0, vmax=1.0)
    ax.set_yticks(np.arange(len(seeds)), [str(seed) for seed in seeds])
    ax.set_xticks([])
    ax.set_xlabel("held-out path")
    ax.set_ylabel("global seed")
    ax.set_title("Cartesian improvement consistency (green improves, red worsens)")
    colorbar = fig.colorbar(image, ax=ax, ticks=(-1, 0, 1))
    colorbar.ax.set_yticklabels(("improved", "tied", "worsened"))
    save_figure(fig, output_dir / "improvement_consistency_across_seeds.png")


def require_scipy() -> Tuple[Any, Any]:
    try:
        import scipy
        from scipy import stats as scipy_stats
    except Exception as exc:
        raise RuntimeError(
            "SciPy is required for the prespecified Wilcoxon and correlation analyses"
        ) from exc
    return scipy, scipy_stats


def main() -> int:
    args = parse_args()
    validate_args(args)
    if args.diagnostic_only:
        return run_fk_diagnostic_only(args)
    configure_full_fk_rollout_capture(bool(args.run_fk_validation))
    configure_determinism()
    scipy_module, scipy_stats = require_scipy()
    assets = load_benchmark_assets(args)
    selected_count = (
        len(assets["path_names"])
        if args.max_paths == 0
        else min(args.max_paths, len(assets["path_names"]))
    )
    selected_names = list(assets["path_names"][:selected_count])
    if not selected_names:
        raise ValueError("No held-out test paths were selected")
    total_cycles = int(
        math.ceil(assets["desired_paths"].shape[1] / args.execution_horizon)
    )
    candidate_noise_seed(
        max(int(seed) for seed in args.seeds),
        len(assets["path_names"]) - 1,
        total_cycles - 1,
        args.num_base_samples - 1,
        len(assets["path_names"]),
        total_cycles,
        args.num_base_samples,
    )

    source_records, external_provenance = baseline_source_rows(args)
    trajectory_length = int(assets["desired_paths"].shape[1])
    global_references = {
        path_name: load_global_reference(args, path_name, trajectory_length)
        for path_name in assets["path_names"]
    }
    validate_prior_window_reconstruction(
        assets["window_artifact"],
        global_references,
        args.prediction_horizon,
    )

    fk_validation_names: List[str] = []
    fk_tool_transform: Optional[np.ndarray] = None
    fk_prior_detail_rows: List[Dict[str, Any]] = []
    fk_prior_audit_rows: List[Dict[str, Any]] = []
    fk_preflight_records: List[Dict[str, Any]] = []
    fk_resolved_urdf_descriptor = ""
    if not args.run_fk_validation:
        print("[FK validation startup] enabled=False mode=DISABLED")
    else:
        (
            fk_validation_names,
            fk_limit_option,
            fk_requested_limit,
        ) = resolve_fk_validation_path_selection(
            args,
            assets["path_names"],
        )
        if not fk_validation_names:
            raise ValueError("No full-mode paths were selected for FK validation")
        print_fk_validation_path_selection(
            selected_count=len(fk_validation_names),
            available_count=len(assets["path_names"]),
            determining_option=fk_limit_option,
            requested_limit=fk_requested_limit,
        )
        fk_tool_transform = resolve_tool_transform(args)
        fk_prior_detail_rows = list(
            audit_prior_window_path_alignment(
                assets["window_artifact"],
                assets["path_names"],
                assets["desired_paths"],
                assets["expert_q"],
                global_references,
                tolerance=PRIOR_RECONSTRUCTION_AUDIT_ATOL,
            )
        )
        fk_prior_audit_rows = list(
            aggregate_prior_reconstruction_audit_rows(
                fk_prior_detail_rows,
                assets["window_artifact"],
                fk_validation_names,
                trajectory_length,
                args.prediction_horizon,
            )
        )
        fk_resolved_urdf_descriptor = str(
            assets["resolved_urdf_descriptor"]
        )
        print_fk_validation_startup(
            enabled=True,
            diagnostic_only=False,
            resolved_urdf_descriptor=fk_resolved_urdf_descriptor,
            active_joint_names=assets["joint_names"],
            fk_frame=assets["ee_link"],
            tool_transform=fk_tool_transform,
            mean_threshold=args.expert_mean_error_threshold,
            max_threshold=args.expert_max_error_threshold,
            selected_path_count=len(fk_validation_names),
        )

        if args.fail_on_expert_fk_mismatch:
            path_index_by_name = {
                str(path_name): path_index
                for path_index, path_name in enumerate(assets["path_names"])
            }
            with torch.no_grad():
                for path_name in fk_validation_names:
                    path_index = path_index_by_name[path_name]
                    fk_preflight_records.append(
                        build_fk_validation_path_record(
                            path_name=path_name,
                            path_index=path_index,
                            desired=assets["desired_paths"][path_index],
                            expert_q=assets["expert_q"][path_index],
                            global_q=global_references[path_name],
                            diffusion_by_seed={},
                            robot=assets["robot"],
                            joint_names=assets["joint_names"],
                            ee_link=assets["ee_link"],
                            mean_threshold=args.expert_mean_error_threshold,
                            max_threshold=args.expert_max_error_threshold,
                            buffer_q=None,
                            tool_transform=fk_tool_transform,
                        )
                    )
            preflight_mismatches = [
                record
                for record in fk_preflight_records
                if str(
                    record["expert"]["classification_result"]["classification"]
                )
                != "PASS_DIRECT"
            ]
            if preflight_mismatches:
                stale_completion_manifest = (
                    args.output_dir / COMPLETION_MANIFEST_NAME
                )
                if stale_completion_manifest.exists():
                    stale_completion_manifest.unlink()
                try:
                    publish_fk_validation_outputs(
                        args,
                        fk_preflight_records,
                        fk_prior_audit_rows,
                        fk_resolved_urdf_descriptor,
                    )
                finally:
                    configure_full_fk_rollout_capture(False)
                mismatch_descriptions = [
                    str(record["path_name"])
                    + "="
                    + str(
                        record["expert"]["classification_result"][
                            "classification"
                        ]
                    )
                    for record in preflight_mismatches
                ]
                raise RuntimeError(
                    "Expert FK preflight mismatch before benchmark rollouts; "
                    "partial validation outputs were published: "
                    + ", ".join(mismatch_descriptions)
                )
    pristine = pristine_model_state(assets["model"])
    computed_rows: List[Dict[str, Any]] = []
    planning_rows: List[Dict[str, Any]] = []

    architecture_text = json.dumps(FROZEN_ARCHITECTURE, sort_keys=True)
    print(f"Frozen base-tail architecture: {architecture_text}")
    print(f"Held-out paths: {selected_count}; independent seeds: {list(args.seeds)}")
    print(
        "Candidate noise key (injective mixed-radix encoding): global seed, path "
        "index, planning-cycle index, candidate sample index"
    )
    print("Expert joints are evaluation-only and excluded from practical planning/ranking.")

    for selected_index, path_name in enumerate(selected_names):
        path_index = int(assets["path_names"].index(path_name))
        common_kwargs = {
            "args": args,
            "assets": assets,
            "pristine": pristine,
            "path_index": path_index,
            "path_name": path_name,
            "desired_path": assets["desired_paths"][path_index],
            "expert_q_evaluation_only": assets["expert_q"][path_index],
            "q_start": assets["q_start"][path_index],
            "global_reference": global_references[path_name],
        }
        buffer_result = run_one_rollout(method="buffer_only", seed=0, **common_kwargs)
        planning_rows.extend(buffer_result["planning_rows"])
        computed_rows.append(
            computed_result_row(
                result=buffer_result,
                path_name=path_name,
                path_index=path_index,
                seed=-1,
                determinism_verified=True,
                order_independence_verified=True,
            )
        )

        first_results: Dict[int, Dict[str, Any]] = {}
        first_rows: Dict[int, Dict[str, Any]] = {}
        for seed in args.seeds:
            result = run_one_rollout(method="base_tail", seed=int(seed), **common_kwargs)
            first_results[int(seed)] = result
            planning_rows.extend(result["planning_rows"])
            row = computed_result_row(
                result=result,
                path_name=path_name,
                path_index=path_index,
                seed=int(seed),
                determinism_verified=False,
                order_independence_verified=False,
            )
            first_rows[int(seed)] = row
            computed_rows.append(row)

        check_path = args.determinism_check == "all" or (
            args.determinism_check == "first_path" and selected_index == 0
        )
        if check_path:
            for seed in reversed(args.seeds):
                repeated = run_one_rollout(
                    method="base_tail", seed=int(seed), **common_kwargs
                )
                assert_identical_rerun(
                    first_results[int(seed)], repeated, path_name, int(seed)
                )
                first_rows[int(seed)]["determinism_rerun_verified"] = 1
                first_rows[int(seed)]["seed_order_independence_verified"] = 1
        print(f"Completed {selected_index + 1}/{selected_count}: {path_name}")

    if args.determinism_check != "all":
        warnings.warn(
            "Reduced determinism checking was requested; criterion 10 must not be "
            "interpreted as a full final-benchmark reproducibility verification.",
            RuntimeWarning,
        )

    external_rows, external_method_rows = build_external_result_rows(
        args=args,
        assets=assets,
        selected_names=selected_names,
        global_references=global_references,
        source_records=source_records,
        provenance=external_provenance,
    )
    validate_external_baseline_completeness(
        external_method_rows,
        selected_names,
    )
    per_seed_rows = [*computed_rows, *external_rows]
    paired_rows = build_paired_path_summary(computed_rows, args.seeds)
    stat_rows = statistical_tests(
        paired_rows, args, scipy_module, scipy_stats
    )
    stability_rows = seed_stability_rows(
        computed_rows, args.seeds, args, scipy_stats
    )

    method_rows: Dict[str, Sequence[Mapping[str, Any]]] = {
        "buffer_only": buffer_path_rows(computed_rows),
        "base_tail_diffusion": diffusion_path_rows(paired_rows),
        **external_method_rows,
    }
    provenance: Dict[str, Mapping[str, Any]] = {
        "buffer_only": {"source_type": "computed_by_final_benchmark", "source_csv": ""},
        "base_tail_diffusion": {
            "source_type": "computed_by_final_benchmark_seed_mean",
            "source_csv": "",
        },
        **external_provenance,
    }
    aggregate_rows, unified_rows, common_paths = aggregate_method_outputs(
        method_rows, selected_names, provenance
    )
    failure_rows = failure_path_rankings(paired_rows)
    decision_rows = scientific_decision_rows(
        paired_rows=paired_rows,
        computed_rows=computed_rows,
        statistical_rows=stat_rows,
        args=args,
    )

    fk_validation_records: List[Dict[str, Any]] = []
    fk_validation_publication: Optional[Dict[str, Any]] = None
    try:
        if args.run_fk_validation:
            if fk_tool_transform is None:
                raise RuntimeError(
                    "full-mode FK validation tool transform was not initialized"
                )
            fk_validation_records = build_full_fk_validation_records_from_capture(
                args,
                assets,
                fk_validation_names,
                global_references,
                fk_tool_transform,
            )
    finally:
        # Capture must never remain active during publication or after main exits.
        configure_full_fk_rollout_capture(False)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    completion_manifest = args.output_dir / COMPLETION_MANIFEST_NAME
    if completion_manifest.exists():
        completion_manifest.unlink()
    required_outputs = {
        "per_seed_path_results.csv": per_seed_rows,
        "paired_path_summary.csv": paired_rows,
        "aggregate_method_summary.csv": aggregate_rows,
        "statistical_tests.csv": stat_rows,
        "seed_stability_summary.csv": stability_rows,
        "unified_baseline_comparison.csv": unified_rows,
        "failure_path_ranking.csv": failure_rows,
        "planning_cycle_details.csv": planning_rows,
        "scientific_decision.csv": decision_rows,
    }
    for filename, rows in required_outputs.items():
        write_records_csv(args.output_dir / filename, rows)

    _PUBLISHED_PLOT_PATHS.clear()
    save_required_plots(
        output_dir=args.output_dir,
        paired_rows=paired_rows,
        computed_rows=computed_rows,
        planning_rows=planning_rows,
        statistical_rows=stat_rows,
        aggregate_rows=aggregate_rows,
        seeds=args.seeds,
    )

    missing_csvs = [
        filename
        for filename in required_outputs
        if not (args.output_dir / filename).is_file()
    ]
    if missing_csvs:
        raise RuntimeError(
            f"Required benchmark CSVs are missing after publication: {missing_csvs}"
        )
    published_plot_paths = list(_PUBLISHED_PLOT_PATHS)
    plot_files = [path.name for path in published_plot_paths]
    invalid_plot_paths = [
        str(path)
        for path in published_plot_paths
        if path.parent != args.output_dir
        or path.suffix.lower() != ".png"
        or not path.is_file()
    ]
    if (
        len(published_plot_paths) != REQUIRED_PLOT_COUNT
        or len(set(plot_files)) != REQUIRED_PLOT_COUNT
        or invalid_plot_paths
    ):
        raise RuntimeError(
            "Required plot publication is incomplete: "
            f"expected exactly {REQUIRED_PLOT_COUNT} unique PNG files generated "
            f"in {args.output_dir}, recorded={plot_files}, "
            f"invalid_paths={invalid_plot_paths}"
        )
    plot_files = sorted(plot_files)

    if args.run_fk_validation:
        fk_validation_publication = publish_fk_validation_outputs(
            args,
            fk_validation_records,
            fk_prior_audit_rows,
            fk_resolved_urdf_descriptor,
        )
        fk_mismatch_rows = [
            row
            for row in fk_validation_publication[
                "expert_fk_validation_summary_rows"
            ]
            if str(row["classification"]) != "PASS_DIRECT"
        ]
        if fk_mismatch_rows and not args.fail_on_expert_fk_mismatch:
            print(
                "*** WARNING: EXPERT FK VALIDATION MISMATCH; BENCHMARK DECISION "
                "IS UNCHANGED, BUT DIFFUSION TRACKING METRICS MAY NOT BE "
                "SCIENTIFICALLY INTERPRETABLE. ***"
            )
            print(
                "*** FK mismatch paths: "
                + ", ".join(
                    str(row["path_name"])
                    + "="
                    + str(row["classification"])
                    for row in fk_mismatch_rows
                )
                + " ***"
            )

    if fk_validation_publication is None:
        fk_top_level_csv_names: List[str] = []
        fk_pointwise_status = {
            "requested": bool(args.save_fk_pointwise_csv),
            "written": False,
            "filename": None,
        }
        fk_trajectory_file_counts: Dict[str, int] = {}
        fk_plot_file_counts: Dict[str, int] = {}
        fk_all_relative_paths: List[str] = []
        fk_conclusion = "NOT_RUN"
        fk_conclusions = ["NOT_RUN"]
        fk_decision_statuses = {
            "coordinate_fk_validity": "NOT_PERFORMED",
            "expert_dataset_validity": "NOT_PERFORMED",
            "prior_reconstruction_validity": "NOT_PERFORMED",
            "prior_trajectory_quality": "NOT_PERFORMED",
            "diffusion_tracking_validity": "NOT_PERFORMED",
        }
        fk_classification_counts: Dict[str, int] = {}
        fk_scientifically_valid = False
    else:
        fk_top_level_csv_names = sorted(
            path.name
            for path in fk_validation_publication["top_level_csv_paths"]
        )
        pointwise_output_path = fk_validation_publication["pointwise_csv_path"]
        fk_pointwise_status = {
            "requested": bool(fk_validation_publication["pointwise_csv_enabled"]),
            "written": pointwise_output_path is not None,
            "filename": (
                pointwise_output_path.name
                if pointwise_output_path is not None
                else None
            ),
        }
        fk_trajectory_file_counts = dict(
            sorted(
                Counter(
                    path.parent.name
                    for path in fk_validation_publication[
                        "per_path_trajectory_paths"
                    ]
                ).items()
            )
        )
        fk_plot_file_counts = dict(
            sorted(
                Counter(
                    path.parent.name
                    for path in fk_validation_publication["plot_paths"]
                ).items()
            )
        )
        fk_all_relative_paths = [
            str(path.relative_to(args.output_dir))
            for path in fk_validation_publication["all_output_paths"]
        ]
        fk_interpretation = fk_validation_publication["interpretation"]
        fk_conclusion = str(fk_interpretation["conclusion"])
        fk_conclusions = [
            str(value) for value in fk_interpretation["conclusions"]
        ]
        fk_decision_statuses = {
            key: str(fk_interpretation[key])
            for key in (
                "coordinate_fk_validity",
                "expert_dataset_validity",
                "prior_reconstruction_validity",
                "prior_trajectory_quality",
                "diffusion_tracking_validity",
            )
        }
        fk_classification_counts = dict(
            fk_interpretation["category_counts"]
        )
        fk_scientifically_valid = bool(
            fk_interpretation[
                "scientifically_valid_to_interpret_diffusion_tracking_metrics"
            ]
        )
    manifest_payload = {
        "status": "complete",
        "required_csv_count": len(required_outputs),
        "required_csv_files": list(required_outputs),
        "required_plot_count": REQUIRED_PLOT_COUNT,
        "published_plot_count": len(plot_files),
        "published_plot_files": plot_files,
        "held_out_path_count": selected_count,
        "seeds": [int(seed) for seed in args.seeds],
        "frozen_architecture": FROZEN_ARCHITECTURE,
        "frozen_rollout_settings": FROZEN_ROLLOUT_SETTINGS,
        "fk_validation_enabled": bool(args.run_fk_validation),
        "fk_validation_selected_path_count": int(len(fk_validation_names)),
        "fk_validation_top_level_csv_files": fk_top_level_csv_names,
        "fk_validation_pointwise_csv": fk_pointwise_status,
        "fk_validation_per_path_trajectory_file_counts": (
            fk_trajectory_file_counts
        ),
        "fk_validation_per_path_plot_file_counts": fk_plot_file_counts,
        "fk_validation_all_output_paths_relative": fk_all_relative_paths,
        "fk_validation_conclusion": fk_conclusion,
        "fk_validation_conclusions": fk_conclusions,
        "fk_validation_decision_statuses": fk_decision_statuses,
        "fk_validation_classification_counts": fk_classification_counts,
        "fk_validation_scientifically_valid_to_interpret_diffusion_tracking_metrics": (
            fk_scientifically_valid
        ),
        "fk_validation_resolved_urdf_descriptor": (
            fk_resolved_urdf_descriptor
            if args.run_fk_validation
            else str(assets["resolved_urdf_descriptor"])
        ),
        "fk_validation_fk_frame": str(assets["ee_link"]),
        "fk_validation_tool_transform": (
            fk_tool_transform.tolist()
            if fk_tool_transform is not None
            else None
        ),
    }
    temporary_manifest = args.output_dir / f".{COMPLETION_MANIFEST_NAME}.tmp"
    with temporary_manifest.open("w", encoding="utf-8") as handle:
        json.dump(manifest_payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary_manifest, completion_manifest)

    print(f"Common-path baseline cohort: {len(common_paths)}")
    for method in ("adaptive_mlp_ik", "diffusion_v1_best_of_k", "expert_ik"):
        print(
            f"Reused {method}: {external_provenance[method]['source_csv']} "
            f"(sha256={external_provenance[method]['source_sha256'][:12]}...)"
        )
    overall = next(row for row in decision_rows if row["criterion"] == "OVERALL")
    print(f"Scientific decision: {overall['measured_result']}")
    if overall["measured_result"] != "RECOMMEND":
        print(
            "The prespecified evidence does not support a superiority claim; see "
            "scientific_decision.csv and confidence intervals."
        )
    for filename in required_outputs:
        print(f"Saved: {args.output_dir / filename}")
    print(f"Saved required noninteractive plots under: {args.output_dir}")
    print(f"Saved completion manifest: {completion_manifest}")
    return 0


def validate_homogeneous_transform(
    transform: Any,
    *,
    label: str = "transform",
) -> np.ndarray:
    """Return a validated proper 4x4 homogeneous transform."""

    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"{label} must have shape (4, 4), got {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{label} must contain only finite values")
    if not np.allclose(
        matrix[3, :],
        np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64),
        rtol=0.0,
        atol=FK_HOMOGENEOUS_ATOL,
    ):
        raise ValueError(
            f"{label} must have homogeneous bottom row [0, 0, 0, 1]"
        )

    rotation = matrix[:3, :3]
    if not np.allclose(
        rotation.T @ rotation,
        np.eye(3, dtype=np.float64),
        rtol=0.0,
        atol=FK_ROTATION_ATOL,
    ):
        raise ValueError(f"{label} rotation block must be orthonormal")
    determinant = float(np.linalg.det(rotation))
    if not math.isclose(
        determinant,
        1.0,
        rel_tol=0.0,
        abs_tol=FK_ROTATION_ATOL,
    ):
        raise ValueError(
            f"{label} rotation block must be proper (determinant +1), "
            f"got {determinant:.9g}"
        )
    return matrix.copy()


def _load_tool_transform_file(path: Path) -> np.ndarray:
    path = Path(path).expanduser()
    if not path.is_file():
        raise ValueError(f"tool transform file does not exist: {path}")

    suffix = path.suffix.lower()
    try:
        if suffix == ".npy":
            raw_transform = np.load(path, allow_pickle=False)
        elif suffix == ".npz":
            with np.load(path, allow_pickle=False) as archive:
                if "tool_transform" in archive.files:
                    raw_transform = archive["tool_transform"]
                elif len(archive.files) == 1:
                    raw_transform = archive[archive.files[0]]
                else:
                    raise ValueError(
                        "NPZ tool transform must contain a 'tool_transform' "
                        "array or exactly one array"
                    )
        elif suffix == ".json":
            with path.open("r", encoding="utf-8") as handle:
                raw_transform = json.load(handle)
        else:
            delimiter = "," if suffix == ".csv" else None
            raw_transform = np.loadtxt(path, delimiter=delimiter)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError(f"failed to read tool transform from {path}: {exc}") from exc

    return validate_homogeneous_transform(
        raw_transform,
        label=f"tool transform from {path}",
    )


def tool_transform_from_xyz_rpy(
    xyz: Optional[Sequence[float]] = None,
    rpy: Optional[Sequence[float]] = None,
) -> np.ndarray:
    """Build radians XYZ/RPY using Rz(yaw) @ Ry(pitch) @ Rx(roll)."""

    translation = np.zeros(3, dtype=np.float64)
    if xyz is not None:
        translation = np.asarray(xyz, dtype=np.float64)
        if translation.shape != (3,) or not np.all(np.isfinite(translation)):
            raise ValueError("tool XYZ offset must be exactly three finite values")
    angles = np.zeros(3, dtype=np.float64)
    if rpy is not None:
        angles = np.asarray(rpy, dtype=np.float64)
        if angles.shape != (3,) or not np.all(np.isfinite(angles)):
            raise ValueError("tool RPY offset must be exactly three finite values")

    roll, pitch, yaw = (float(value) for value in angles)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rotation_x = np.array(
        [[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float64
    )
    rotation_y = np.array(
        [[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]], dtype=np.float64
    )
    rotation_z = np.array(
        [[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation_z @ rotation_y @ rotation_x
    transform[:3, 3] = translation
    return validate_homogeneous_transform(transform, label="numeric tool transform")


def resolve_tool_transform(args: argparse.Namespace) -> np.ndarray:
    """Resolve file or numeric tool transform; absent options mean identity."""

    uses_numeric_offset = (
        args.tool_offset_xyz is not None or args.tool_offset_rpy is not None
    )
    if args.tool_transform is not None and uses_numeric_offset:
        raise ValueError(
            "--tool_transform is mutually exclusive with --tool_offset_xyz "
            "and --tool_offset_rpy"
        )
    if args.tool_transform is not None:
        return _load_tool_transform_file(args.tool_transform)
    return tool_transform_from_xyz_rpy(args.tool_offset_xyz, args.tool_offset_rpy)


def validate_joint_trajectory(
    trajectory: Any,
    *,
    label: str = "joint trajectory",
    expected_steps: Optional[int] = None,
) -> np.ndarray:
    array = np.asarray(trajectory, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 6:
        raise ValueError(f"{label} must have shape [T, 6], got {array.shape}")
    if array.shape[0] == 0:
        raise ValueError(f"{label} must contain at least one timestep")
    if expected_steps is not None and array.shape[0] != expected_steps:
        raise ValueError(
            f"{label} must contain {expected_steps} timesteps, got {array.shape[0]}"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must contain only finite values")
    return array


def validate_cartesian_trajectory(
    trajectory: Any,
    *,
    label: str = "Cartesian trajectory",
    expected_steps: Optional[int] = None,
) -> np.ndarray:
    array = np.asarray(trajectory, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"{label} must have shape [T, 3], got {array.shape}")
    if array.shape[0] == 0:
        raise ValueError(f"{label} must contain at least one timestep")
    if expected_steps is not None and array.shape[0] != expected_steps:
        raise ValueError(
            f"{label} must contain {expected_steps} timesteps, got {array.shape[0]}"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must contain only finite values")
    return array


def validate_cartesian_pair(
    reference: Any,
    observed: Any,
) -> Tuple[np.ndarray, np.ndarray]:
    reference_array = validate_cartesian_trajectory(
        reference, label="reference Cartesian trajectory"
    )
    observed_array = validate_cartesian_trajectory(
        observed, label="observed Cartesian trajectory"
    )
    if observed_array.shape != reference_array.shape:
        raise ValueError(
            "Cartesian trajectory shape mismatch: reference has shape "
            f"{reference_array.shape}, observed has shape {observed_array.shape}"
        )
    return reference_array, observed_array


def transform_to_homogeneous_matrix(
    transform: Any,
    *,
    label: str = "robot transform",
) -> np.ndarray:
    candidate = transform
    if hasattr(candidate, "matrix"):
        candidate = candidate.matrix
        candidate = candidate() if callable(candidate) else candidate
    elif hasattr(candidate, "A"):
        candidate = candidate.A
        candidate = candidate() if callable(candidate) else candidate
    return validate_homogeneous_transform(candidate, label=label)


def validation_fk_positions(
    robot: Any,
    joint_names: Sequence[str],
    ee_link: str,
    q_trajectory: Any,
    tool_transform: Optional[Any] = None,
) -> np.ndarray:
    """Canonical validation FK with a postmultiplied EE-to-tool transform."""

    if len(joint_names) != 6 or len(set(joint_names)) != 6:
        raise ValueError("validation FK requires exactly six unique joint names")
    if not isinstance(ee_link, str) or not ee_link:
        raise ValueError("validation FK EE link must be a non-empty string")
    q_array = validate_joint_trajectory(q_trajectory)
    ee_to_tool = (
        np.eye(4, dtype=np.float64)
        if tool_transform is None
        else validate_homogeneous_transform(
            tool_transform, label="validation FK tool transform"
        )
    )
    positions = np.empty((q_array.shape[0], 3), dtype=np.float64)
    for timestep, joint_values in enumerate(q_array):
        cfg = {
            name: float(value)
            for name, value in zip(joint_names, joint_values)
        }
        robot.update_cfg(cfg)
        world_to_ee = transform_to_homogeneous_matrix(
            robot.get_transform(frame_to=ee_link),
            label=f"FK transform at timestep {timestep}",
        )
        world_to_tool = world_to_ee @ ee_to_tool
        if not np.all(np.isfinite(world_to_tool)):
            raise ValueError(f"non-finite FK result at timestep {timestep}")
        positions[timestep] = world_to_tool[:3, 3]
    return validate_cartesian_trajectory(
        positions,
        label="validation FK Cartesian trajectory",
        expected_steps=q_array.shape[0],
    )


def _trajectory_arc_length(trajectory: np.ndarray) -> float:
    if trajectory.shape[0] < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(trajectory, axis=0), axis=1)))


def _safe_ratio(numerator: float, denominator: float) -> float:
    if abs(float(denominator)) <= FK_RANGE_DENOMINATOR_ATOL:
        return float("nan")
    return float(numerator) / float(denominator)


def cartesian_error_metrics(reference: Any, observed: Any) -> Dict[str, Any]:
    """Absolute pointwise, per-axis, extent, and path-length diagnostics."""

    reference_array, observed_array = validate_cartesian_pair(reference, observed)
    error = observed_array - reference_array
    distance = np.linalg.norm(error, axis=1)
    reference_centroid = np.mean(reference_array, axis=0)
    observed_centroid = np.mean(observed_array, axis=0)
    reference_range = np.ptp(reference_array, axis=0)
    observed_range = np.ptp(observed_array, axis=0)
    range_ratio = [
        _safe_ratio(observed_range[index], reference_range[index])
        for index in range(3)
    ]
    reference_arc_length = _trajectory_arc_length(reference_array)
    observed_arc_length = _trajectory_arc_length(observed_array)
    axis_names = ("x", "y", "z")
    return {
        "num_points": int(reference_array.shape[0]),
        "mean_distance": float(np.mean(distance)),
        "rms_distance": float(np.sqrt(np.mean(np.square(distance)))),
        "max_distance": float(np.max(distance)),
        "median_distance": float(np.median(distance)),
        "p95_distance": float(np.percentile(distance, 95.0)),
        "start_distance": float(distance[0]),
        "end_distance": float(distance[-1]),
        "axis_signed_error": {
            axis: float(np.mean(error[:, index]))
            for index, axis in enumerate(axis_names)
        },
        "axis_mae": {
            axis: float(np.mean(np.abs(error[:, index])))
            for index, axis in enumerate(axis_names)
        },
        "axis_rms": {
            axis: float(np.sqrt(np.mean(np.square(error[:, index]))))
            for index, axis in enumerate(axis_names)
        },
        "reference_centroid": reference_centroid.tolist(),
        "observed_centroid": observed_centroid.tolist(),
        "centroid_delta_observed_minus_reference": (
            observed_centroid - reference_centroid
        ).tolist(),
        "reference_range": reference_range.tolist(),
        "observed_range": observed_range.tolist(),
        "range_ratio_observed_to_reference": range_ratio,
        "reference_arc_length": reference_arc_length,
        "observed_arc_length": observed_arc_length,
        "arc_length_ratio_observed_to_reference": _safe_ratio(
            observed_arc_length, reference_arc_length
        ),
    }


def translation_alignment_diagnostics(
    reference: Any,
    observed: Any,
) -> Dict[str, Any]:
    reference_array, observed_array = validate_cartesian_pair(reference, observed)
    centroid_translation = (
        np.mean(reference_array, axis=0) - np.mean(observed_array, axis=0)
    )
    start_translation = reference_array[0] - observed_array[0]
    return {
        "centroid_alignment": {
            "translation_applied_to_observed": centroid_translation.tolist(),
            "metrics": cartesian_error_metrics(
                reference_array, observed_array + centroid_translation
            ),
        },
        "start_alignment": {
            "translation_applied_to_observed": start_translation.tolist(),
            "metrics": cartesian_error_metrics(
                reference_array, observed_array + start_translation
            ),
        },
    }


def kabsch_alignment_diagnostics(
    reference: Any,
    observed: Any,
) -> Dict[str, Any]:
    """Align observed to reference with a proper rigid transform and no scale."""

    reference_array, observed_array = validate_cartesian_pair(reference, observed)
    reference_centroid = np.mean(reference_array, axis=0)
    observed_centroid = np.mean(observed_array, axis=0)
    reference_centered = reference_array - reference_centroid
    observed_centered = observed_array - observed_centroid
    covariance = observed_centered.T @ reference_centered
    u_matrix, singular_values, vt_matrix = np.linalg.svd(covariance)
    rotation = vt_matrix.T @ u_matrix.T
    determinant_before = float(np.linalg.det(rotation))
    reflection_corrected = determinant_before < 0.0
    if reflection_corrected:
        vt_matrix = vt_matrix.copy()
        vt_matrix[-1, :] *= -1.0
        rotation = vt_matrix.T @ u_matrix.T
    determinant_after = float(np.linalg.det(rotation))
    translation = reference_centroid - rotation @ observed_centroid
    aligned_observed = observed_array @ rotation.T + translation
    return {
        "metrics": cartesian_error_metrics(reference_array, aligned_observed),
        "rotation_observed_to_reference": rotation.tolist(),
        "translation_observed_to_reference": translation.tolist(),
        "scale": 1.0,
        "reflection_corrected": bool(reflection_corrected),
        "determinant_before_reflection_correction": determinant_before,
        "determinant_after_reflection_correction": determinant_after,
        "covariance_rank": int(np.linalg.matrix_rank(covariance)),
        "singular_values": singular_values.tolist(),
    }


def _metric_order_key(metrics: Dict[str, Any]) -> Tuple[float, float, float]:
    return (
        float(metrics["mean_distance"]),
        float(metrics["max_distance"]),
        float(metrics["rms_distance"]),
    )


def temporal_alignment_diagnostics(
    reference: Any,
    observed: Any,
) -> Dict[str, Any]:
    """Compare direct, reversed, and every circularly shifted correspondence."""

    reference_array, observed_array = validate_cartesian_pair(reference, observed)
    circular_shifts = []
    for shift in range(observed_array.shape[0]):
        circular_shifts.append(
            {
                "shift": int(shift),
                "metrics": cartesian_error_metrics(
                    reference_array,
                    np.roll(observed_array, shift=shift, axis=0),
                ),
            }
        )
    best_circular = min(
        circular_shifts,
        key=lambda item: (
            *_metric_order_key(item["metrics"]),
            abs(int(item["shift"])),
            int(item["shift"]),
        ),
    )
    return {
        "direct": {"metrics": cartesian_error_metrics(reference_array, observed_array)},
        "reversed": {
            "metrics": cartesian_error_metrics(reference_array, observed_array[::-1])
        },
        "circular_shifts": circular_shifts,
        "best_circular_shift": best_circular,
    }


def _passes_error_thresholds(
    metrics: Dict[str, Any],
    *,
    mean_threshold: float,
    max_threshold: float,
) -> bool:
    return (
        float(metrics["mean_distance"]) <= mean_threshold
        and float(metrics["max_distance"]) <= max_threshold
    )


def _is_material_improvement(
    candidate: Dict[str, Any],
    direct: Dict[str, Any],
    *,
    mean_threshold: float,
    max_threshold: float,
    material_improvement_factor: float,
) -> bool:
    return (
        _passes_error_thresholds(
            candidate,
            mean_threshold=mean_threshold,
            max_threshold=max_threshold,
        )
        and float(candidate["mean_distance"])
        <= material_improvement_factor * float(direct["mean_distance"])
        and float(candidate["max_distance"])
        <= material_improvement_factor * float(direct["max_distance"])
    )


def classify_expert_fk_mismatch(
    reference: Any,
    observed: Any,
    *,
    mean_threshold: float,
    max_threshold: float,
    material_improvement_factor: float = FK_CLASSIFICATION_MATERIAL_IMPROVEMENT_FACTOR,
) -> Dict[str, Any]:
    """Classify an expert/FK comparison with explicit quantitative rules.

    Direct correspondence passes only when both configured thresholds pass.
    A mismatch hypothesis must pass those thresholds after alignment and reduce
    both direct mean and maximum error by the material-improvement factor.
    Translation is considered before rigid-frame and temporal hypotheses.
    """

    if not math.isfinite(mean_threshold) or mean_threshold <= 0.0:
        raise ValueError("mean_threshold must be finite and > 0")
    if not math.isfinite(max_threshold) or max_threshold <= 0.0:
        raise ValueError("max_threshold must be finite and > 0")
    if (
        not math.isfinite(material_improvement_factor)
        or not 0.0 < material_improvement_factor < 1.0
    ):
        raise ValueError("material_improvement_factor must be finite and in (0, 1)")

    result: Dict[str, Any] = {
        "classification": "INCONCLUSIVE",
        "classification_rule": "no hypothesis satisfied its quantitative rule",
        "mean_error_threshold": float(mean_threshold),
        "max_error_threshold": float(max_threshold),
        "material_improvement_factor": float(material_improvement_factor),
    }
    try:
        reference_array, observed_array = validate_cartesian_pair(reference, observed)
    except (TypeError, ValueError) as exc:
        result.update(
            {
                "classification": "EXPERT_SHAPE_MISMATCH",
                "classification_rule": str(exc),
                "diagnostics": None,
            }
        )
        return result

    direct_metrics = cartesian_error_metrics(reference_array, observed_array)
    translation = translation_alignment_diagnostics(reference_array, observed_array)
    rigid = kabsch_alignment_diagnostics(reference_array, observed_array)
    temporal = temporal_alignment_diagnostics(reference_array, observed_array)
    result["diagnostics"] = {
        "direct": direct_metrics,
        "translation": translation,
        "rigid": rigid,
        "temporal": temporal,
    }

    if _passes_error_thresholds(
        direct_metrics,
        mean_threshold=mean_threshold,
        max_threshold=max_threshold,
    ):
        result.update(
            {
                "classification": "PASS_DIRECT",
                "classification_rule": (
                    "direct mean and maximum distances are within configured "
                    "thresholds"
                ),
            }
        )
        return result

    translation_candidates = (
        ("centroid", translation["centroid_alignment"]["metrics"]),
        ("start", translation["start_alignment"]["metrics"]),
    )
    best_translation_name, best_translation_metrics = min(
        translation_candidates,
        key=lambda item: (*_metric_order_key(item[1]), item[0]),
    )
    if _is_material_improvement(
        best_translation_metrics,
        direct_metrics,
        mean_threshold=mean_threshold,
        max_threshold=max_threshold,
        material_improvement_factor=material_improvement_factor,
    ):
        result.update(
            {
                "classification": "POSSIBLE_FIXED_TRANSLATION",
                "classification_rule": (
                    f"{best_translation_name} translation passes thresholds and "
                    "materially improves direct mean and maximum error"
                ),
            }
        )
        return result

    rigid_passes_thresholds = _passes_error_thresholds(
        rigid["metrics"],
        mean_threshold=mean_threshold,
        max_threshold=max_threshold,
    )
    rigid_rank = int(rigid["covariance_rank"])
    if rigid_passes_thresholds and rigid_rank < 2:
        result.update(
            {
                "classification": "INCONCLUSIVE",
                "classification_rule": (
                    "rigid alignment passes configured thresholds, but Kabsch "
                    f"covariance rank={rigid_rank} is below two; the rigid-frame "
                    "hypothesis is not geometrically identifiable"
                ),
            }
        )
        return result
    if rigid_rank >= 2 and _is_material_improvement(
        rigid["metrics"],
        direct_metrics,
        mean_threshold=mean_threshold,
        max_threshold=max_threshold,
        material_improvement_factor=material_improvement_factor,
    ):
        result.update(
            {
                "classification": "POSSIBLE_RIGID_FRAME_MISMATCH",
                "classification_rule": (
                    "proper no-scale Kabsch alignment passes thresholds and "
                    "materially improves direct mean and maximum error"
                ),
            }
        )
        return result

    temporal_candidates = [("reversed", None, temporal["reversed"]["metrics"])]
    temporal_candidates.extend(
        ("circular_shift", int(item["shift"]), item["metrics"])
        for item in temporal["circular_shifts"]
        if int(item["shift"]) != 0
    )
    best_temporal_kind, best_temporal_shift, best_temporal_metrics = min(
        temporal_candidates,
        key=lambda item: (
            *_metric_order_key(item[2]),
            item[0],
            -1 if item[1] is None else abs(int(item[1])),
        ),
    )
    if _is_material_improvement(
        best_temporal_metrics,
        direct_metrics,
        mean_threshold=mean_threshold,
        max_threshold=max_threshold,
        material_improvement_factor=material_improvement_factor,
    ):
        temporal_description = best_temporal_kind
        if best_temporal_shift is not None:
            temporal_description += f"={best_temporal_shift}"
        result.update(
            {
                "classification": "POSSIBLE_TEMPORAL_MISALIGNMENT",
                "classification_rule": (
                    f"{temporal_description} passes thresholds and materially "
                    "improves direct mean and maximum error"
                ),
            }
        )
        return result

    if result["classification"] not in FK_CLASSIFICATIONS:
        raise RuntimeError(f"unknown FK classification: {result['classification']}")
    return result


def cartesian_validation_diagnostics(
    reference: Any,
    observed: Any,
    *,
    mean_threshold: float,
    max_threshold: float,
) -> Dict[str, Any]:
    """Convenience entry point for the complete phase-1 diagnostic set."""

    return classify_expert_fk_mismatch(
        reference,
        observed,
        mean_threshold=mean_threshold,
        max_threshold=max_threshold,
    )


PRIOR_RECONSTRUCTION_AUDIT_ATOL = 1.0e-8


def audit_prior_window_path_alignment(
    window_artifact: Mapping[str, Any],
    path_names: Sequence[str],
    desired_paths: np.ndarray,
    expert_q: np.ndarray,
    global_references: Mapping[str, np.ndarray],
    *,
    tolerance: float = PRIOR_RECONSTRUCTION_AUDIT_ATOL,
) -> Sequence[Dict[str, Any]]:
    """Audit window-to-full-path alignment without resolving overlap conflicts.

    One row is returned for every expected ``(path_name, global_timestep)``.
    Multiple prior observations of the same global timestep remain separate for
    discrepancy calculation; the audit never selects or averages one into a
    reconstructed prior value. Structural, name, finite-value, and coverage
    failures raise ``ValueError``. Finite prior disagreements are retained as
    rows whose ``passed`` field is false.
    """

    required_keys = {
        "path_names",
        "window_start_indices",
        "prior_q_window",
        "desired_path_window",
        "expert_q_window",
    }
    missing_keys = sorted(required_keys.difference(window_artifact.keys()))
    if missing_keys:
        raise ValueError(
            "Validated window artifact is missing audit-required keys: "
            f"{missing_keys}"
        )

    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError(
            "Prior reconstruction audit tolerance must be finite and "
            f"non-negative, got {tolerance!r}"
        )

    full_names = [str(name) for name in path_names]
    if not full_names:
        raise ValueError("Prior reconstruction audit requires at least one path")
    if len(set(full_names)) != len(full_names):
        raise ValueError(f"Full path_names must be unique, got {full_names}")

    desired = np.asarray(desired_paths)
    expert = np.asarray(expert_q)
    if desired.ndim != 3 or desired.shape[0] != len(full_names) or desired.shape[2] != 3:
        raise ValueError(
            "desired_paths must have shape [N,T,3] aligned with path_names, "
            f"got {desired.shape} for N={len(full_names)}"
        )
    if expert.ndim != 3 or expert.shape[0] != len(full_names) or expert.shape[2] != 6:
        raise ValueError(
            "expert_q must have shape [N,T,6] aligned with path_names, "
            f"got {expert.shape} for N={len(full_names)}"
        )
    if desired.shape[1] != expert.shape[1]:
        raise ValueError(
            "desired_paths and expert_q must have the same full trajectory "
            f"length, got {desired.shape[1]} and {expert.shape[1]}"
        )
    if not np.all(np.isfinite(desired)):
        raise ValueError("desired_paths contains non-finite values")
    if not np.all(np.isfinite(expert)):
        raise ValueError("expert_q contains non-finite values")

    artifact_names_array = np.asarray(window_artifact["path_names"])
    starts = np.asarray(window_artifact["window_start_indices"])
    prior_windows = np.asarray(window_artifact["prior_q_window"])
    desired_windows = np.asarray(window_artifact["desired_path_window"])
    expert_windows = np.asarray(window_artifact["expert_q_window"])

    if artifact_names_array.ndim != 1:
        raise ValueError(
            "window artifact path_names must be one-dimensional, got "
            f"{artifact_names_array.shape}"
        )
    artifact_names = [str(name) for name in artifact_names_array.tolist()]
    window_count = len(artifact_names)
    if starts.shape != (window_count,):
        raise ValueError(
            "window_start_indices must have shape [W], got "
            f"{starts.shape} for W={window_count}"
        )
    if not np.issubdtype(starts.dtype, np.integer):
        raise ValueError(
            "window_start_indices must retain an integer dtype, got "
            f"{starts.dtype}"
        )
    if prior_windows.ndim != 3 or prior_windows.shape[0] != window_count or prior_windows.shape[2] != 6:
        raise ValueError(
            "prior_q_window must have shape [W,H,6], got "
            f"{prior_windows.shape} for W={window_count}"
        )
    horizon = int(prior_windows.shape[1])
    if horizon <= 0:
        raise ValueError(f"Window horizon must be positive, got {horizon}")
    expected_desired_shape = (window_count, horizon, 3)
    expected_expert_shape = (window_count, horizon, 6)
    if desired_windows.shape != expected_desired_shape:
        raise ValueError(
            "desired_path_window must have shape [W,H,3], got "
            f"{desired_windows.shape}; expected {expected_desired_shape}"
        )
    if expert_windows.shape != expected_expert_shape:
        raise ValueError(
            "expert_q_window must have shape [W,H,6], got "
            f"{expert_windows.shape}; expected {expected_expert_shape}"
        )
    for label, values in (
        ("prior_q_window", prior_windows),
        ("desired_path_window", desired_windows),
        ("expert_q_window", expert_windows),
    ):
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{label} contains non-finite values")

    artifact_name_set = set(artifact_names)
    full_name_set = set(full_names)
    if artifact_name_set != full_name_set:
        raise ValueError(
            "Window/full path names are irreconcilable: "
            f"missing={sorted(full_name_set - artifact_name_set)}, "
            f"unexpected={sorted(artifact_name_set - full_name_set)}"
        )

    try:
        reference_names = [str(name) for name in global_references.keys()]
    except AttributeError as exc:
        raise ValueError(
            "global_references must be a mapping keyed by exact path name"
        ) from exc
    reference_name_set = set(reference_names)
    if len(reference_names) != len(reference_name_set):
        raise ValueError("global_references contains duplicate path names")
    if reference_name_set != full_name_set:
        raise ValueError(
            "Global-reference/full path names are irreconcilable: "
            f"missing={sorted(full_name_set - reference_name_set)}, "
            f"unexpected={sorted(reference_name_set - full_name_set)}"
        )

    trajectory_length = int(desired.shape[1])
    if horizon > trajectory_length:
        raise ValueError(
            f"Window horizon H={horizon} exceeds full length T={trajectory_length}"
        )
    name_to_index = {name: index for index, name in enumerate(full_names)}
    references: Dict[str, np.ndarray] = {}
    for name in full_names:
        reference = np.asarray(global_references[name])
        if reference.shape != (trajectory_length, 6):
            raise ValueError(
                f"global_references[{name!r}] must have shape "
                f"[{trajectory_length},6], got {reference.shape}"
            )
        if not np.all(np.isfinite(reference)):
            raise ValueError(
                f"global_references[{name!r}] contains non-finite values"
            )
        references[name] = reference

    coverage: Dict[str, Sequence[Any]] = {
        name: [[] for _ in range(trajectory_length)] for name in full_names
    }
    previous_start: Dict[str, int] = {}
    for window_row, (name, raw_start) in enumerate(zip(artifact_names, starts.tolist())):
        start = int(raw_start)
        if start < 0 or start + horizon > trajectory_length:
            raise ValueError(
                f"Window row {window_row} for path {name!r} has out-of-range "
                f"start={start} for H={horizon}, T={trajectory_length}"
            )
        if name in previous_start and start <= previous_start[name]:
            raise ValueError(
                f"Window starts for path {name!r} must be strictly monotonic; "
                f"encountered {start} after {previous_start[name]} at row {window_row}"
            )
        previous_start[name] = start

        path_index = name_to_index[name]
        expected_desired = desired[path_index, start : start + horizon]
        expected_expert = expert[path_index, start : start + horizon]
        if not np.array_equal(desired_windows[window_row], expected_desired):
            max_difference = float(
                np.max(np.abs(desired_windows[window_row] - expected_desired))
            )
            raise ValueError(
                f"desired_path_window row {window_row} does not exactly match "
                f"full path {name!r} at timesteps [{start}, {start + horizon}); "
                f"max_abs_difference={max_difference:.17g}"
            )
        if not np.array_equal(expert_windows[window_row], expected_expert):
            max_difference = float(
                np.max(np.abs(expert_windows[window_row] - expected_expert))
            )
            raise ValueError(
                f"expert_q_window row {window_row} does not exactly match full "
                f"path {name!r} at timesteps [{start}, {start + horizon}); "
                f"max_abs_difference={max_difference:.17g}"
            )

        for local_offset in range(horizon):
            global_timestep = start + local_offset
            coverage[name][global_timestep].append(
                {
                    "window_row": int(window_row),
                    "window_start_index": start,
                    "window_local_offset": int(local_offset),
                    "prior_q": prior_windows[window_row, local_offset],
                }
            )

    missing_coverage = {
        name: [
            timestep
            for timestep, observations in enumerate(coverage[name])
            if not observations
        ]
        for name in full_names
    }
    missing_coverage = {
        name: timesteps for name, timesteps in missing_coverage.items() if timesteps
    }
    if missing_coverage:
        raise ValueError(
            "Window artifact does not provide full global-index coverage: "
            f"{missing_coverage}"
        )

    audit_rows = []
    for name in full_names:
        reference = references[name]
        for global_timestep, observations in enumerate(coverage[name]):
            observed = np.stack(
                [np.asarray(item["prior_q"], dtype=np.float64) for item in observations],
                axis=0,
            )
            reference_abs = np.abs(observed - reference[global_timestep][None, :])
            max_reference_discrepancy = float(np.max(reference_abs))
            mean_reference_discrepancy = float(np.mean(reference_abs))

            pairwise_parts = []
            for left in range(observed.shape[0]):
                for right in range(left + 1, observed.shape[0]):
                    pairwise_parts.append(np.abs(observed[left] - observed[right]))
            if pairwise_parts:
                pairwise_abs = np.stack(pairwise_parts, axis=0)
                max_overlap_discrepancy = float(np.max(pairwise_abs))
                mean_overlap_discrepancy = float(np.mean(pairwise_abs))
            else:
                max_overlap_discrepancy = 0.0
                mean_overlap_discrepancy = 0.0

            passed = bool(
                max_reference_discrepancy <= tolerance
                and max_overlap_discrepancy <= tolerance
            )
            audit_rows.append(
                {
                    "path_name": name,
                    "global_timestep": int(global_timestep),
                    "observation_count": int(len(observations)),
                    "duplicate_index_count": int(len(observations) - 1),
                    "is_overlap": bool(len(observations) > 1),
                    "window_rows": [
                        int(item["window_row"]) for item in observations
                    ],
                    "window_start_indices": [
                        int(item["window_start_index"]) for item in observations
                    ],
                    "window_local_offsets": [
                        int(item["window_local_offset"]) for item in observations
                    ],
                    "tolerance": float(tolerance),
                    "max_abs_discrepancy_vs_global_reference": (
                        max_reference_discrepancy
                    ),
                    "mean_abs_discrepancy_vs_global_reference": (
                        mean_reference_discrepancy
                    ),
                    "max_abs_discrepancy_between_overlaps": (
                        max_overlap_discrepancy
                    ),
                    "mean_abs_discrepancy_between_overlaps": (
                        mean_overlap_discrepancy
                    ),
                    "passed": passed,
                    "details": (
                        ""
                        if passed
                        else "Finite prior observations disagree beyond the "
                        "declared absolute tolerance; no observation was selected."
                    ),
                }
            )

    return audit_rows


def _full_fk_cartesian_diagnostics(
    desired: Any,
    observed_ee: Any,
) -> Dict[str, Any]:
    desired_array, observed_array = validate_cartesian_pair(desired, observed_ee)
    return {
        "direct": cartesian_error_metrics(desired_array, observed_array),
        "translation": translation_alignment_diagnostics(
            desired_array, observed_array
        ),
        "rigid": kabsch_alignment_diagnostics(desired_array, observed_array),
        "temporal": temporal_alignment_diagnostics(desired_array, observed_array),
    }


def _build_fk_validation_method_entry(
    *,
    desired: np.ndarray,
    q_trajectory: Any,
    robot: Any,
    joint_names: Sequence[str],
    ee_link: str,
    tool_transform: np.ndarray,
) -> Dict[str, Any]:
    q_array = validate_joint_trajectory(
        q_trajectory,
        expected_steps=desired.shape[0],
    )
    ee_array = validation_fk_positions(
        robot,
        joint_names,
        ee_link,
        q_array,
        tool_transform,
    )
    return {
        "q": q_array,
        "ee": ee_array,
        "diagnostics": _full_fk_cartesian_diagnostics(desired, ee_array),
    }


def _select_diffusion_medoid_seed_by_direct_mean_error(
    diffusion_entries: Mapping[int, Mapping[str, Any]],
) -> int:
    """Select an actual seed as the one-dimensional medoid of direct mean error."""

    if not diffusion_entries:
        raise ValueError("at least one diffusion seed trajectory is required")
    errors = {
        int(seed): float(entry["diagnostics"]["direct"]["mean_distance"])
        for seed, entry in diffusion_entries.items()
    }
    return min(
        errors,
        key=lambda seed: (
            sum(abs(errors[seed] - other_error) for other_error in errors.values()),
            errors[seed],
            seed,
        ),
    )


def build_fk_validation_path_record(
    *,
    path_name: str,
    path_index: int,
    desired: Any,
    expert_q: Any,
    global_q: Any,
    diffusion_by_seed: Mapping[int, Any],
    robot: Any,
    joint_names: Sequence[str],
    ee_link: str,
    mean_threshold: float,
    max_threshold: float,
    buffer_q: Optional[Any] = None,
    tool_transform: Optional[Any] = None,
) -> Dict[str, Any]:
    """Build standalone FK-validation data without affecting benchmark rollout state."""

    if not isinstance(path_name, str) or not path_name:
        raise ValueError("path_name must be a non-empty string")
    if path_index < 0:
        raise ValueError("path_index must be >= 0")
    desired_array = validate_cartesian_trajectory(
        desired,
        label=f"desired Cartesian trajectory for {path_name}",
    )
    resolved_tool = (
        np.eye(4, dtype=np.float64)
        if tool_transform is None
        else validate_homogeneous_transform(
            tool_transform,
            label=f"tool transform for {path_name}",
        )
    )

    expert_entry = _build_fk_validation_method_entry(
        desired=desired_array,
        q_trajectory=expert_q,
        robot=robot,
        joint_names=joint_names,
        ee_link=ee_link,
        tool_transform=resolved_tool,
    )
    expert_classification = classify_expert_fk_mismatch(
        desired_array,
        expert_entry["ee"],
        mean_threshold=mean_threshold,
        max_threshold=max_threshold,
    )
    # Use the classifier's diagnostic object as the single authoritative expert
    # diagnostic payload rather than independently recomputing it downstream.
    expert_entry["classification_result"] = expert_classification
    expert_entry["diagnostics"] = expert_classification["diagnostics"]

    global_entry = _build_fk_validation_method_entry(
        desired=desired_array,
        q_trajectory=global_q,
        robot=robot,
        joint_names=joint_names,
        ee_link=ee_link,
        tool_transform=resolved_tool,
    )
    buffer_entry = None
    if buffer_q is not None:
        buffer_entry = _build_fk_validation_method_entry(
            desired=desired_array,
            q_trajectory=buffer_q,
            robot=robot,
            joint_names=joint_names,
            ee_link=ee_link,
            tool_transform=resolved_tool,
        )

    diffusion_entries: Dict[int, Dict[str, Any]] = {}
    for raw_seed, q_trajectory in diffusion_by_seed.items():
        if isinstance(raw_seed, bool):
            raise ValueError("diffusion seed identifiers must be integers, not bool")
        seed = int(raw_seed)
        if seed in diffusion_entries:
            raise ValueError(f"duplicate diffusion seed after integer conversion: {seed}")
        diffusion_entries[seed] = _build_fk_validation_method_entry(
            desired=desired_array,
            q_trajectory=q_trajectory,
            robot=robot,
            joint_names=joint_names,
            ee_link=ee_link,
            tool_transform=resolved_tool,
        )
    diffusion_entries = dict(sorted(diffusion_entries.items()))
    if diffusion_entries:
        medoid_seed: Optional[int] = (
            _select_diffusion_medoid_seed_by_direct_mean_error(diffusion_entries)
        )
        diffusion_ee_stack = np.stack(
            [entry["ee"] for entry in diffusion_entries.values()], axis=0
        )
        descriptive_ee_seed_mean: Optional[np.ndarray] = np.mean(
            diffusion_ee_stack, axis=0
        )
        descriptive_ee_seed_mean_label = (
            "DESCRIPTIVE_ONLY_EE_SEED_MEAN_NOT_A_ROBOT_TRAJECTORY"
        )
    else:
        medoid_seed = None
        descriptive_ee_seed_mean = None
        descriptive_ee_seed_mean_label = "NOT_PERFORMED"

    return {
        "path_name": path_name,
        "path_index": int(path_index),
        "desired": desired_array,
        "robot_ee_link": ee_link,
        "joint_names": tuple(str(name) for name in joint_names),
        "tool_transform": resolved_tool,
        "fk_convention": (
            "robot.update_cfg(cfg); robot.get_transform(frame_to=ee_link); "
            "world_T_tool=world_T_ee@ee_T_tool"
        ),
        "mean_threshold": float(mean_threshold),
        "max_threshold": float(max_threshold),
        "expert": expert_entry,
        "global_reference": global_entry,
        "buffer_only": buffer_entry,
        "base_tail_diffusion_by_seed": diffusion_entries,
        "base_tail_diffusion_medoid_seed": medoid_seed,
        "base_tail_diffusion_ee_seed_mean_descriptive_only": (
            descriptive_ee_seed_mean
        ),
        "base_tail_diffusion_ee_seed_mean_label": descriptive_ee_seed_mean_label,
    }


def _flatten_fk_method_row(
    record: Mapping[str, Any],
    method: str,
    entry: Mapping[str, Any],
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Flatten one actual FK trajectory and its diagnostics into a CSV-ready row."""

    diagnostics = entry.get("diagnostics")
    if not isinstance(diagnostics, Mapping):
        raise ValueError(
            f"FK validation entry for {record['path_name']} / {method} "
            "does not contain diagnostic mappings"
        )
    direct = diagnostics["direct"]
    translation = diagnostics["translation"]
    rigid = diagnostics["rigid"]
    temporal = diagnostics["temporal"]
    centroid_aligned = translation["centroid_alignment"]
    start_aligned = translation["start_alignment"]
    reversed_metrics = temporal["reversed"]["metrics"]
    best_circular = temporal["best_circular_shift"]
    best_circular_metrics = best_circular["metrics"]

    tool_transform = validate_homogeneous_transform(
        record["tool_transform"],
        label=f"stored tool transform for {record['path_name']}",
    )
    tool_applied = not np.allclose(
        tool_transform,
        np.eye(4, dtype=np.float64),
        rtol=0.0,
        atol=FK_HOMOGENEOUS_ATOL,
    )
    mean_threshold = float(record["mean_threshold"])
    max_threshold = float(record["max_threshold"])
    is_expert = method == "expert_fk"
    classification_result = entry.get("classification_result") if is_expert else None
    if is_expert and not isinstance(classification_result, Mapping):
        raise ValueError(
            f"expert FK entry for {record['path_name']} lacks classification_result"
        )

    row: Dict[str, Any] = {
        "path_name": str(record["path_name"]),
        "path_index": int(record["path_index"]),
        "method": method,
        "seed": "" if seed is None else int(seed),
        "is_actual_seed_trajectory": int(seed is not None),
        "is_medoid_diffusion_seed": int(
            seed is not None
            and int(seed) == int(record["base_tail_diffusion_medoid_seed"])
        ),
        "num_points": int(direct["num_points"]),
        "compared_point_count": int(direct["num_points"]),
        "correspondence_mode": "DIRECT_SAME_TIMESTEP_NO_RESAMPLING",
        "direct_metric_role": "ABSOLUTE_FRAME_FK_VALIDATION",
        "temporal_role": "DIAGNOSTIC_CIRCULAR_REINDEXING_ONLY",
        "aligned_metric_role": "DIAGNOSTIC_ONLY_NOT_MODEL_RANKING",
        "aligned_claim_eligible": False,
        "mean_error_threshold": mean_threshold,
        "max_error_threshold": max_threshold,
        "direct_pass": int(
            float(direct["mean_distance"]) <= mean_threshold
            and float(direct["max_distance"]) <= max_threshold
        ),
        "expert_classification": (
            str(classification_result["classification"]) if is_expert else ""
        ),
        "expert_classification_rule": (
            str(classification_result["classification_rule"]) if is_expert else ""
        ),
        "expert_direct_pass": (
            int(classification_result["classification"] == "PASS_DIRECT")
            if is_expert
            else ""
        ),
        "fk_frame": str(record["robot_ee_link"]),
        "fk_convention": str(record["fk_convention"]),
        "joint_names_json": json.dumps(list(record["joint_names"])),
        "tool_transform_applied": int(tool_applied),
        "tool_transform_direction": "ee_T_tool_postmultiply",
        "tool_transform_matrix_json": json.dumps(tool_transform.tolist()),
        "direct_mean_distance": float(direct["mean_distance"]),
        "direct_rms_distance": float(direct["rms_distance"]),
        "direct_max_distance": float(direct["max_distance"]),
        "direct_median_distance": float(direct["median_distance"]),
        "direct_p95_distance": float(direct["p95_distance"]),
        "direct_start_distance": float(direct["start_distance"]),
        "direct_end_distance": float(direct["end_distance"]),
        "desired_arc_length": float(direct["reference_arc_length"]),
        "ee_arc_length": float(direct["observed_arc_length"]),
        "arc_length_ratio_ee_to_desired": float(
            direct["arc_length_ratio_observed_to_reference"]
        ),
        "centroid_aligned_mean_distance": float(
            centroid_aligned["metrics"]["mean_distance"]
        ),
        "centroid_aligned_rms_distance": float(
            centroid_aligned["metrics"]["rms_distance"]
        ),
        "centroid_aligned_max_distance": float(
            centroid_aligned["metrics"]["max_distance"]
        ),
        "start_aligned_mean_distance": float(
            start_aligned["metrics"]["mean_distance"]
        ),
        "start_aligned_rms_distance": float(
            start_aligned["metrics"]["rms_distance"]
        ),
        "start_aligned_max_distance": float(
            start_aligned["metrics"]["max_distance"]
        ),
        "rigid_aligned_mean_distance": float(rigid["metrics"]["mean_distance"]),
        "rigid_aligned_rms_distance": float(rigid["metrics"]["rms_distance"]),
        "rigid_aligned_max_distance": float(rigid["metrics"]["max_distance"]),
        "rigid_scale": float(rigid["scale"]),
        "rigid_reflection_corrected": int(rigid["reflection_corrected"]),
        "rigid_determinant_before_reflection_correction": float(
            rigid["determinant_before_reflection_correction"]
        ),
        "rigid_determinant_after_reflection_correction": float(
            rigid["determinant_after_reflection_correction"]
        ),
        "rigid_covariance_rank": int(rigid["covariance_rank"]),
        "rigid_singular_values_json": json.dumps(rigid["singular_values"]),
        "reversed_mean_distance": float(reversed_metrics["mean_distance"]),
        "reversed_rms_distance": float(reversed_metrics["rms_distance"]),
        "reversed_max_distance": float(reversed_metrics["max_distance"]),
        "best_circular_shift": int(best_circular["shift"]),
        "best_circular_mean_distance": float(
            best_circular_metrics["mean_distance"]
        ),
        "best_circular_rms_distance": float(
            best_circular_metrics["rms_distance"]
        ),
        "best_circular_max_distance": float(
            best_circular_metrics["max_distance"]
        ),
        "best_circular_mean_improvement": float(
            direct["mean_distance"] - best_circular_metrics["mean_distance"]
        ),
        "best_circular_mean_improvement_fraction": _safe_ratio(
            float(direct["mean_distance"])
            - float(best_circular_metrics["mean_distance"]),
            float(direct["mean_distance"]),
        ),
        "best_circular_max_improvement": float(
            direct["max_distance"] - best_circular_metrics["max_distance"]
        ),
        "best_circular_max_improvement_fraction": _safe_ratio(
            float(direct["max_distance"])
            - float(best_circular_metrics["max_distance"]),
            float(direct["max_distance"]),
        ),
    }

    axis_names = ("x", "y", "z")
    for axis_index, axis in enumerate(axis_names):
        row[f"direct_signed_error_{axis}"] = float(
            direct["axis_signed_error"][axis]
        )
        row[f"direct_mae_{axis}"] = float(direct["axis_mae"][axis])
        row[f"direct_rms_{axis}"] = float(direct["axis_rms"][axis])
        row[f"desired_centroid_{axis}"] = float(
            direct["reference_centroid"][axis_index]
        )
        row[f"ee_centroid_{axis}"] = float(
            direct["observed_centroid"][axis_index]
        )
        row[f"centroid_delta_ee_minus_desired_{axis}"] = float(
            direct["centroid_delta_observed_minus_reference"][axis_index]
        )
        row[f"desired_range_{axis}"] = float(
            direct["reference_range"][axis_index]
        )
        row[f"ee_range_{axis}"] = float(
            direct["observed_range"][axis_index]
        )
        row[f"range_ratio_ee_to_desired_{axis}"] = float(
            direct["range_ratio_observed_to_reference"][axis_index]
        )
        row[f"centroid_alignment_translation_{axis}"] = float(
            centroid_aligned["translation_applied_to_observed"][axis_index]
        )
        row[f"start_alignment_translation_{axis}"] = float(
            start_aligned["translation_applied_to_observed"][axis_index]
        )
        row[f"rigid_translation_{axis}"] = float(
            rigid["translation_observed_to_reference"][axis_index]
        )
    rotation = rigid["rotation_observed_to_reference"]
    for row_index in range(3):
        for column_index in range(3):
            row[f"rigid_rotation_r{row_index}{column_index}"] = float(
                rotation[row_index][column_index]
            )
    for row_index in range(4):
        for column_index in range(4):
            row[f"tool_transform_m{row_index}{column_index}"] = float(
                tool_transform[row_index, column_index]
            )
    return row


def fk_validation_per_path_rows(
    records: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Emit deterministic path-by-method rows, preserving every actual seed."""

    rows: List[Dict[str, Any]] = []
    ordered_records = sorted(
        records,
        key=lambda record: (
            int(record["path_index"]),
            str(record["path_name"]),
        ),
    )
    for record in ordered_records:
        rows.append(
            _flatten_fk_method_row(
                record,
                "expert_fk",
                record["expert"],
            )
        )
        rows.append(
            _flatten_fk_method_row(
                record,
                "global_reference",
                record["global_reference"],
            )
        )
        if record.get("buffer_only") is not None:
            rows.append(
                _flatten_fk_method_row(
                    record,
                    "buffer_only",
                    record["buffer_only"],
                )
            )
        diffusion_entries = record["base_tail_diffusion_by_seed"]
        for seed in sorted(int(value) for value in diffusion_entries):
            rows.append(
                _flatten_fk_method_row(
                    record,
                    "base_tail_diffusion",
                    diffusion_entries[seed],
                    seed=seed,
                )
            )
    return rows


def expert_fk_validation_summary_rows(
    records: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Return one thresholded expert-FK validation summary row per path."""

    rows: List[Dict[str, Any]] = []
    for record in sorted(
        records,
        key=lambda item: (int(item["path_index"]), str(item["path_name"])),
    ):
        expert_entry = record["expert"]
        diagnostics = expert_entry["diagnostics"]
        classification = expert_entry["classification_result"]
        direct = diagnostics["direct"]
        centroid = diagnostics["translation"]["centroid_alignment"]["metrics"]
        start = diagnostics["translation"]["start_alignment"]["metrics"]
        rigid = diagnostics["rigid"]["metrics"]
        reversed_metrics = diagnostics["temporal"]["reversed"]["metrics"]
        best_shift = diagnostics["temporal"]["best_circular_shift"]
        best_shift_metrics = best_shift["metrics"]
        mean_threshold = float(record["mean_threshold"])
        max_threshold = float(record["max_threshold"])
        direct_mean_pass = float(direct["mean_distance"]) <= mean_threshold
        direct_max_pass = float(direct["max_distance"]) <= max_threshold
        direct_pass = direct_mean_pass and direct_max_pass
        if direct_pass:
            explanation = (
                "PASS: direct expert FK mean and maximum Cartesian errors both "
                "satisfy their configured thresholds"
            )
        else:
            failed_parts = []
            if not direct_mean_pass:
                failed_parts.append(
                    "mean "
                    f"{float(direct['mean_distance']):.9g} > {mean_threshold:.9g}"
                )
            if not direct_max_pass:
                failed_parts.append(
                    "max "
                    f"{float(direct['max_distance']):.9g} > {max_threshold:.9g}"
                )
            explanation = (
                "FAIL: direct expert FK threshold mismatch ("
                + "; ".join(failed_parts)
                + f"); diagnostic classification={classification['classification']}: "
                + str(classification["classification_rule"])
            )
        rows.append(
            {
                "path_name": str(record["path_name"]),
                "path_index": int(record["path_index"]),
                "mean_error_threshold": mean_threshold,
                "max_error_threshold": max_threshold,
                "direct_mean_distance": float(direct["mean_distance"]),
                "direct_rms_distance": float(direct["rms_distance"]),
                "direct_max_distance": float(direct["max_distance"]),
                "centroid_aligned_mean_distance": float(
                    centroid["mean_distance"]
                ),
                "centroid_aligned_rms_distance": float(
                    centroid["rms_distance"]
                ),
                "centroid_aligned_max_distance": float(
                    centroid["max_distance"]
                ),
                "start_aligned_mean_distance": float(start["mean_distance"]),
                "start_aligned_rms_distance": float(start["rms_distance"]),
                "start_aligned_max_distance": float(start["max_distance"]),
                "rigid_aligned_mean_distance": float(rigid["mean_distance"]),
                "rigid_aligned_rms_distance": float(rigid["rms_distance"]),
                "rigid_aligned_max_distance": float(rigid["max_distance"]),
                "reversed_mean_distance": float(
                    reversed_metrics["mean_distance"]
                ),
                "best_circular_shift": int(best_shift["shift"]),
                "best_circular_mean_distance": float(
                    best_shift_metrics["mean_distance"]
                ),
                "best_circular_rms_distance": float(
                    best_shift_metrics["rms_distance"]
                ),
                "best_circular_max_distance": float(
                    best_shift_metrics["max_distance"]
                ),
                "classification": str(classification["classification"]),
                "classification_rule": str(classification["classification_rule"]),
                "direct_mean_pass": int(direct_mean_pass),
                "direct_max_pass": int(direct_max_pass),
                "direct_pass": int(direct_pass),
                "validation_status": "PASS" if direct_pass else "FAIL",
                "validation_pass": int(direct_pass),
                "validation_explanation": explanation,
            }
        )
    return rows


def fk_validation_aggregate_rows(
    per_path_rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Aggregate method metrics with seed-then-path reduction and equal path weight."""

    if not per_path_rows:
        return []
    metric_specs: List[Tuple[str, str]] = [
        ("direct_mean_distance", "direct_absolute"),
        ("direct_rms_distance", "direct_absolute"),
        ("direct_max_distance", "direct_absolute"),
        ("direct_median_distance", "direct_absolute"),
        ("direct_p95_distance", "direct_absolute"),
        ("direct_start_distance", "direct_absolute"),
        ("direct_end_distance", "direct_absolute"),
    ]
    for axis in ("x", "y", "z"):
        metric_specs.extend(
            [
                (f"direct_signed_error_{axis}", "direct_per_axis_signed"),
                (f"direct_mae_{axis}", "direct_per_axis_absolute"),
                (f"direct_rms_{axis}", "direct_per_axis_absolute"),
            ]
        )
    metric_specs.extend(
        [
            ("centroid_aligned_mean_distance", "centroid_aligned"),
            ("centroid_aligned_rms_distance", "centroid_aligned"),
            ("centroid_aligned_max_distance", "centroid_aligned"),
            ("start_aligned_mean_distance", "start_aligned"),
            ("start_aligned_rms_distance", "start_aligned"),
            ("start_aligned_max_distance", "start_aligned"),
            ("rigid_aligned_mean_distance", "rigid_aligned"),
            ("rigid_aligned_rms_distance", "rigid_aligned"),
            ("rigid_aligned_max_distance", "rigid_aligned"),
            ("reversed_mean_distance", "temporal_reversed"),
            ("reversed_rms_distance", "temporal_reversed"),
            ("reversed_max_distance", "temporal_reversed"),
            ("best_circular_mean_distance", "temporal_circular"),
            ("best_circular_rms_distance", "temporal_circular"),
            ("best_circular_max_distance", "temporal_circular"),
            ("best_circular_mean_improvement", "temporal_improvement"),
            (
                "best_circular_mean_improvement_fraction",
                "temporal_improvement",
            ),
            ("best_circular_max_improvement", "temporal_improvement"),
            (
                "best_circular_max_improvement_fraction",
                "temporal_improvement",
            ),
        ]
    )

    grouped: Dict[Tuple[str, str], List[Mapping[str, Any]]] = defaultdict(list)
    for row in per_path_rows:
        grouped[(str(row["method"]), str(row["path_name"]))].append(row)
    method_order = {
        "expert_fk": 0,
        "global_reference": 1,
        "buffer_only": 2,
        "base_tail_diffusion": 3,
    }
    methods = sorted(
        {method for method, _ in grouped},
        key=lambda method: (method_order.get(method, len(method_order)), method),
    )
    output_rows: List[Dict[str, Any]] = []
    for method in methods:
        path_groups = {
            path_name: grouped[(method, path_name)]
            for grouped_method, path_name in grouped
            if grouped_method == method
        }
        for metric_name, metric_family in metric_specs:
            allow_undefined_values = metric_name in {
                "best_circular_mean_improvement_fraction",
                "best_circular_max_improvement_fraction",
            }
            within_path_values: List[float] = []
            within_path_worst_seed_values: List[float] = []
            source_row_count = 0
            undefined_source_value_count = 0
            seed_counts: List[int] = []
            for path_name in sorted(path_groups):
                source_rows = path_groups[path_name]
                if method != "base_tail_diffusion" and len(source_rows) != 1:
                    raise ValueError(
                        f"non-diffusion FK aggregate has {len(source_rows)} rows "
                        f"for method={method}, path={path_name}; expected one"
                    )
                values = [float(row[metric_name]) for row in source_rows]
                finite_values = [value for value in values if math.isfinite(value)]
                if len(finite_values) != len(values) and not allow_undefined_values:
                    raise ValueError(
                        f"non-finite FK aggregate metric {metric_name} for "
                        f"method={method}, path={path_name}"
                    )
                undefined_source_value_count += len(values) - len(finite_values)
                if finite_values:
                    within_path_values.append(float(np.mean(finite_values)))
                if (
                    method == "base_tail_diffusion"
                    and metric_name == "direct_max_distance"
                ):
                    within_path_worst_seed_values.append(
                        float(np.max(finite_values))
                    )
                source_row_count += len(source_rows)
                seed_counts.append(len(source_rows))
            values_array = np.asarray(within_path_values, dtype=np.float64)
            path_count = int(len(path_groups))
            value_defined_count = int(values_array.size)
            undefined_count = path_count - value_defined_count
            sample_std = (
                float(np.std(values_array, ddof=1))
                if value_defined_count > 1
                else float("nan")
            )
            if value_defined_count:
                aggregate_mean: Any = float(np.mean(values_array))
                aggregate_median: Any = float(np.median(values_array))
                aggregate_minimum: Any = float(np.min(values_array))
                aggregate_maximum: Any = float(np.max(values_array))
            else:
                aggregate_mean = ""
                aggregate_median = ""
                aggregate_minimum = ""
                aggregate_maximum = ""
            if metric_family.startswith("direct_"):
                alignment = "DIRECT_SAME_TIMESTEP_NO_RESAMPLING"
                metric_role = "ABSOLUTE_FRAME_FK_VALIDATION"
                claim_eligible = True
            elif metric_family == "centroid_aligned":
                alignment = "CENTROID_TRANSLATION_ALIGNED"
                metric_role = "DIAGNOSTIC_ONLY_NOT_MODEL_RANKING"
                claim_eligible = False
            elif metric_family == "start_aligned":
                alignment = "START_TRANSLATION_ALIGNED"
                metric_role = "DIAGNOSTIC_ONLY_NOT_MODEL_RANKING"
                claim_eligible = False
            elif metric_family == "rigid_aligned":
                alignment = "PROPER_NO_SCALE_KABSCH_ALIGNED"
                metric_role = "DIAGNOSTIC_ONLY_NOT_MODEL_RANKING"
                claim_eligible = False
            else:
                alignment = "DIAGNOSTIC_TEMPORAL_REINDEXING"
                metric_role = "DIAGNOSTIC_ONLY_NOT_MODEL_RANKING"
                claim_eligible = False
            seed_reduction = (
                "ARITHMETIC_MEAN_ACROSS_ACTUAL_SEEDS_WITHIN_PATH"
                if method == "base_tail_diffusion"
                else "SINGLE_DETERMINISTIC_TRAJECTORY_PER_PATH"
            )
            if within_path_worst_seed_values:
                worst_seed_value: Any = float(
                    np.max(within_path_worst_seed_values)
                )
                worst_seed_equal_path_mean: Any = float(
                    np.mean(within_path_worst_seed_values)
                )
                worst_seed_reduction = (
                    "MAX_ACROSS_ACTUAL_SEEDS_WITHIN_PATH; OVERALL VALUE IS MAX_"
                    "ACROSS_PATHS; EQUAL_PATH_MEAN_REPORTED_SEPARATELY"
                )
            else:
                worst_seed_value = ""
                worst_seed_equal_path_mean = ""
                worst_seed_reduction = "NOT_APPLICABLE"
            output_rows.append(
                {
                    "method": method,
                    "metric_family": metric_family,
                    "metric": metric_name,
                    "alignment": alignment,
                    "metric_role": metric_role,
                    "claim_eligible": claim_eligible,
                    "seed_reduction": seed_reduction,
                    "worst_seed_reduction": worst_seed_reduction,
                    "worst_seed_value": worst_seed_value,
                    "worst_seed_equal_path_mean": worst_seed_equal_path_mean,
                    "aggregation_stage_1": (
                        "arithmetic_mean_across_actual_seeds_within_each_path"
                        if method == "base_tail_diffusion"
                        else "single_trajectory_value_within_each_path"
                    ),
                    "aggregation_stage_2": "equal_weight_across_paths",
                    "source_method_row_count": int(source_row_count),
                    "within_path_seed_count_min": int(min(seed_counts)),
                    "within_path_seed_count_max": int(max(seed_counts)),
                    "path_count": path_count,
                    "value_defined_count": value_defined_count,
                    "undefined_count": undefined_count,
                    "undefined_source_value_count": int(
                        undefined_source_value_count
                    ),
                    "mean": aggregate_mean,
                    "median": aggregate_median,
                    "sample_std": sample_std,
                    "min": aggregate_minimum,
                    "max": aggregate_maximum,
                }
            )
    return output_rows


def coordinate_frame_audit_rows(
    records: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Summarize coordinate extents and rigid-frame evidence once per path."""

    axis_names = ("x", "y", "z")

    def add_trajectory_fields(
        row: Dict[str, Any],
        prefix: str,
        trajectory: Any,
        desired_centroid: np.ndarray,
        rigid: Optional[Mapping[str, Any]],
    ) -> None:
        array = validate_cartesian_trajectory(
            trajectory,
            label=f"coordinate-frame audit {prefix}",
        )
        minimum = np.min(array, axis=0)
        maximum = np.max(array, axis=0)
        extent = maximum - minimum
        centroid = np.mean(array, axis=0)
        for axis_index, axis in enumerate(axis_names):
            row[f"{prefix}_min_{axis}"] = float(minimum[axis_index])
            row[f"{prefix}_max_{axis}"] = float(maximum[axis_index])
            row[f"{prefix}_range_{axis}"] = float(extent[axis_index])
            row[f"{prefix}_centroid_{axis}"] = float(centroid[axis_index])
            row[f"{prefix}_centroid_minus_desired_{axis}"] = float(
                centroid[axis_index] - desired_centroid[axis_index]
            )
        row[f"{prefix}_arc_length"] = _trajectory_arc_length(array)
        if rigid is not None:
            rotation = rigid["rotation_observed_to_reference"]
            translation = rigid["translation_observed_to_reference"]
            for rotation_row in range(3):
                for rotation_column in range(3):
                    row[
                        f"{prefix}_rigid_rotation_r{rotation_row}{rotation_column}"
                    ] = float(rotation[rotation_row][rotation_column])
            for axis_index, axis in enumerate(axis_names):
                row[f"{prefix}_rigid_translation_{axis}"] = float(
                    translation[axis_index]
                )
            row[f"{prefix}_rigid_reflection_corrected"] = int(
                rigid["reflection_corrected"]
            )
            row[f"{prefix}_rigid_determinant"] = float(
                rigid["determinant_after_reflection_correction"]
            )

    rows: List[Dict[str, Any]] = []
    for record in sorted(
        records,
        key=lambda item: (int(item["path_index"]), str(item["path_name"])),
    ):
        desired = validate_cartesian_trajectory(
            record["desired"],
            label=f"coordinate-frame audit desired {record['path_name']}",
        )
        desired_centroid = np.mean(desired, axis=0)
        diffusion_entries = record["base_tail_diffusion_by_seed"]
        diffusion_available = bool(diffusion_entries)
        raw_medoid_seed = record.get("base_tail_diffusion_medoid_seed")
        medoid_seed = int(raw_medoid_seed) if diffusion_available else None
        if diffusion_available and medoid_seed not in diffusion_entries:
            raise ValueError(
                f"stored diffusion medoid seed {medoid_seed} is absent for "
                f"{record['path_name']}"
            )
        if diffusion_available:
            diffusion_direct_means = {
                int(seed): float(
                    entry["diagnostics"]["direct"]["mean_distance"]
                )
                for seed, entry in diffusion_entries.items()
            }
            medoid_direct_mean: Any = diffusion_direct_means[medoid_seed]
            medoid_selection_score: Any = sum(
                abs(medoid_direct_mean - other_mean)
                for other_mean in diffusion_direct_means.values()
            )
        else:
            medoid_direct_mean = ""
            medoid_selection_score = ""
        expert_classification = record["expert"]["classification_result"]
        row: Dict[str, Any] = {
            "path_name": str(record["path_name"]),
            "path_index": int(record["path_index"]),
            "coordinate_units": "metres",
            "fk_frame": str(record["robot_ee_link"]),
            "fk_convention": str(record["fk_convention"]),
            "tool_transform_matrix_json": json.dumps(
                np.asarray(record["tool_transform"], dtype=np.float64).tolist()
            ),
            "expert_fk_available": 1,
            "global_reference_available": 1,
            "buffer_only_available": int(record.get("buffer_only") is not None),
            "buffer_only_status": (
                "AVAILABLE"
                if record.get("buffer_only") is not None
                else "NOT_PERFORMED"
            ),
            "base_tail_diffusion_available": int(diffusion_available),
            "diffusion_available": int(diffusion_available),
            "diffusion_representative": (
                "ACTUAL_MEDOID_SEED"
                if diffusion_available
                else "NOT_PERFORMED"
            ),
            "diffusion_seed_count": int(len(diffusion_entries)),
            "diffusion_medoid_selection_metric": (
                "ONE_DIMENSIONAL_MEDOID_OF_DIRECT_MEAN_DISTANCE"
            ),
            "diffusion_medoid_tie_rule": (
                "MIN_TOTAL_ABSOLUTE_ERROR_DISTANCE_THEN_LOWER_DIRECT_MEAN_"
                "THEN_LOWER_SEED"
            ),
            "diffusion_medoid_selection_score": medoid_selection_score,
            "diffusion_medoid_direct_mean_distance": medoid_direct_mean,
            "base_tail_diffusion_status": (
                "AVAILABLE"
                if diffusion_available
                else "NOT_PERFORMED"
            ),
            "base_tail_diffusion_medoid_seed": (
                medoid_seed if medoid_seed is not None else ""
            ),
            "base_tail_diffusion_medoid_is_actual_seed": int(diffusion_available),
            "base_tail_diffusion_medoid_selection": (
                "actual_seed_medoid_of_direct_mean_cartesian_error"
                if diffusion_available
                else "NOT_PERFORMED"
            ),
            "expert_classification": str(
                expert_classification["classification"]
            ),
            "expert_classification_rule": str(
                expert_classification["classification_rule"]
            ),
            "expert_direct_pass": int(
                expert_classification["classification"] == "PASS_DIRECT"
            ),
        }
        add_trajectory_fields(
            row,
            "desired",
            desired,
            desired_centroid,
            rigid=None,
        )
        for prefix, entry in (
            ("expert_fk", record["expert"]),
            ("global_reference", record["global_reference"]),
            ("buffer_only", record.get("buffer_only")),
            (
                "base_tail_diffusion_medoid",
                (
                    diffusion_entries[medoid_seed]
                    if medoid_seed is not None
                    else None
                ),
            ),
        ):
            if entry is None:
                for axis in axis_names:
                    for field in (
                        "min",
                        "max",
                        "range",
                        "centroid",
                        "centroid_minus_desired",
                        "rigid_translation",
                    ):
                        row[f"{prefix}_{field}_{axis}"] = ""
                for rotation_row in range(3):
                    for rotation_column in range(3):
                        row[
                            f"{prefix}_rigid_rotation_r{rotation_row}{rotation_column}"
                        ] = ""
                row[f"{prefix}_arc_length"] = ""
                row[f"{prefix}_rigid_reflection_corrected"] = ""
                row[f"{prefix}_rigid_determinant"] = ""
                continue
            add_trajectory_fields(
                row,
                prefix,
                entry["ee"],
                desired_centroid,
                rigid=entry["diagnostics"]["rigid"],
            )
        rows.append(row)
    return rows


def fk_pointwise_error_rows(
    records: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    """Emit desired-versus-FK errors for actual trajectories only."""

    rows: List[Dict[str, Any]] = []
    ordered_records = sorted(
        records,
        key=lambda item: (int(item["path_index"]), str(item["path_name"])),
    )
    for record in ordered_records:
        desired = validate_cartesian_trajectory(
            record["desired"],
            label=f"pointwise desired trajectory {record['path_name']}",
        )
        method_entries: List[Tuple[str, Optional[int], Mapping[str, Any]]] = [
            ("expert_fk", None, record["expert"]),
            ("global_reference", None, record["global_reference"]),
        ]
        if record.get("buffer_only") is not None:
            method_entries.append(("buffer_only", None, record["buffer_only"]))
        diffusion_entries = record["base_tail_diffusion_by_seed"]
        method_entries.extend(
            ("base_tail_diffusion", seed, diffusion_entries[seed])
            for seed in sorted(int(value) for value in diffusion_entries)
        )
        for method, seed, entry in method_entries:
            _, observed = validate_cartesian_pair(desired, entry["ee"])
            signed_error = observed - desired
            error_norm = np.linalg.norm(signed_error, axis=1)
            for timestep in range(desired.shape[0]):
                rows.append(
                    {
                        "path_name": str(record["path_name"]),
                        "path_index": int(record["path_index"]),
                        "method": method,
                        "seed": "" if seed is None else int(seed),
                        "seed_scope": (
                            "individual_seed" if seed is not None else "deterministic"
                        ),
                        "correspondence_mode": (
                            "DIRECT_SAME_TIMESTEP_NO_RESAMPLING"
                        ),
                        "is_actual_seed_trajectory": int(seed is not None),
                        "is_medoid_diffusion_seed": int(
                            seed is not None
                            and int(seed)
                            == int(record["base_tail_diffusion_medoid_seed"])
                        ),
                        "global_timestep": int(timestep),
                        "coordinate_units": "metres",
                        "desired_x": float(desired[timestep, 0]),
                        "desired_y": float(desired[timestep, 1]),
                        "desired_z": float(desired[timestep, 2]),
                        "fk_x": float(observed[timestep, 0]),
                        "fk_y": float(observed[timestep, 1]),
                        "fk_z": float(observed[timestep, 2]),
                        "signed_error_observed_minus_desired_x": float(
                            signed_error[timestep, 0]
                        ),
                        "signed_error_observed_minus_desired_y": float(
                            signed_error[timestep, 1]
                        ),
                        "signed_error_observed_minus_desired_z": float(
                            signed_error[timestep, 2]
                        ),
                        "error_norm": float(error_norm[timestep]),
                    }
                )
    return rows


def aggregate_prior_reconstruction_audit_rows(
    detail_rows: Sequence[Mapping[str, Any]],
    window_artifact: Mapping[str, Any],
    selected_path_names: Sequence[str],
    trajectory_length: int,
    horizon: int,
    stride: Optional[int] = None,
) -> List[dict]:
    """Reduce stride-aware reconstruction evidence to one row per path."""

    selected_names = [str(name) for name in selected_path_names]
    if len(set(selected_names)) != len(selected_names):
        raise ValueError(
            "selected_path_names must be unique for prior audit aggregation"
        )
    if trajectory_length <= 0:
        raise ValueError(
            f"trajectory_length must be positive, got {trajectory_length}"
        )
    if horizon <= 0 or horizon > trajectory_length:
        raise ValueError(
            f"horizon must be in [1, {trajectory_length}], got {horizon}"
        )
    if stride is None:
        if "stride" not in window_artifact:
            raise KeyError(
                "Prior reconstruction audit requires stride from normalization "
                "metadata or the window NPZ"
            )
        resolved_stride = _positive_integer_metadata_scalar(
            window_artifact["stride"],
            label="validated window artifact stride",
        )
        stride_source = str(
            window_artifact.get("stride_source", "validated_window_artifact")
        )
    else:
        resolved_stride = _positive_integer_metadata_scalar(
            stride,
            label="prior reconstruction audit stride",
        )
        stride_source = "explicit_function_argument"

    artifact_names = [
        str(name) for name in np.asarray(window_artifact["path_names"]).tolist()
    ]
    artifact_starts = np.asarray(window_artifact["window_start_indices"])
    if artifact_starts.shape != (len(artifact_names),):
        raise ValueError(
            "window artifact names/starts length mismatch: "
            f"{len(artifact_names)} names versus shape {artifact_starts.shape}"
        )

    artifact_name_set = set(artifact_names)
    if "expected_path_names" in window_artifact:
        expected_artifact_names = [
            str(name) for name in window_artifact["expected_path_names"]
        ]
        if len(set(expected_artifact_names)) != len(expected_artifact_names):
            raise ValueError("expected_path_names in window artifact must be unique")
        expected_artifact_name_set = set(expected_artifact_names)
    else:
        # A legacy/synthetic artifact cannot establish whether non-selected
        # names are unexpected. It can still prove whether each selected name
        # is present without manufacturing a disagreement for a selected subset.
        expected_artifact_name_set = artifact_name_set.union(selected_names)
    missing_artifact_path_names = sorted(
        expected_artifact_name_set.difference(artifact_name_set)
    )
    unexpected_artifact_path_names = sorted(
        artifact_name_set.difference(expected_artifact_name_set)
    )
    path_name_disagreement = bool(
        missing_artifact_path_names or unexpected_artifact_path_names
    )

    detail_by_path: Dict[str, List[Mapping[str, Any]]] = {
        name: [] for name in selected_names
    }
    for detail in detail_rows:
        name = str(detail.get("path_name", ""))
        if name in detail_by_path:
            detail_by_path[name].append(detail)

    expected_start_indices = list(
        range(0, trajectory_length - horizon + 1, resolved_stride)
    )
    expected_window_count = len(expected_start_indices)
    expected_starts = set(expected_start_indices)
    aggregate_rows: List[dict] = []

    for name in selected_names:
        source_starts = [
            int(start)
            for artifact_name, start in zip(artifact_names, artifact_starts.tolist())
            if artifact_name == name
        ]
        valid_starts = [
            start
            for start in source_starts
            if 0 <= start and start + horizon <= trajectory_length
        ]
        out_of_range_starts = [
            start
            for start in source_starts
            if start < 0 or start + horizon > trajectory_length
        ]
        start_counts = Counter(source_starts)
        duplicate_start_indices = sorted(
            start for start, count in start_counts.items() if count > 1
        )
        monotonic_start_pass = bool(
            source_starts
            and all(
                right > left
                for left, right in zip(source_starts[:-1], source_starts[1:])
            )
        )
        missing_window_starts = sorted(expected_starts.difference(valid_starts))
        unexpected_start_indices = sorted(
            set(source_starts).difference(expected_starts)
        )

        coverage_counts = np.zeros(trajectory_length, dtype=np.int64)
        for start in valid_starts:
            coverage_counts[start : start + horizon] += 1
        missing_global_indices = np.flatnonzero(coverage_counts == 0).astype(int).tolist()
        overlap_global_indices = np.flatnonzero(coverage_counts > 1).astype(int).tolist()

        path_details = detail_by_path[name]
        detail_timestep_counts: Dict[int, int] = {}
        for detail in path_details:
            timestep = int(detail.get("global_timestep", -1))
            detail_timestep_counts[timestep] = detail_timestep_counts.get(timestep, 0) + 1
        duplicate_detail_timesteps = sorted(
            timestep
            for timestep, count in detail_timestep_counts.items()
            if count > 1
        )
        absent_detail_timesteps = sorted(
            set(range(trajectory_length)).difference(detail_timestep_counts)
        )
        if absent_detail_timesteps:
            missing_global_indices = sorted(
                set(missing_global_indices).union(absent_detail_timesteps)
            )

        tolerance_values = sorted(
            {
                float(detail["tolerance"])
                for detail in path_details
                if detail.get("tolerance") is not None
            }
        )
        tolerance_consistent = len(tolerance_values) <= 1
        tolerance = (
            tolerance_values[0]
            if len(tolerance_values) == 1
            else (
                float(PRIOR_RECONSTRUCTION_AUDIT_ATOL)
                if not tolerance_values
                else None
            )
        )

        max_pairwise_overlap_discrepancy = 0.0
        pairwise_weighted_sum = 0.0
        pairwise_comparison_count = 0
        max_reference_discrepancy = 0.0
        reference_weighted_sum = 0.0
        reference_observation_count = 0
        worst_discrepancy = -1.0
        affected_worst_timestep = None
        affected_worst_kind = ""
        finite_discrepancies = True
        failed_detail_timesteps = []
        overlap_disagreement_timesteps: List[int] = []
        global_reference_disagreement_timesteps: List[int] = []

        for detail in path_details:
            timestep = int(detail.get("global_timestep", -1))
            observation_count = int(detail.get("observation_count", 0))
            pair_count = observation_count * (observation_count - 1) // 2
            pair_max = float(
                detail.get("max_abs_discrepancy_between_overlaps", 0.0)
            )
            pair_mean = float(
                detail.get("mean_abs_discrepancy_between_overlaps", 0.0)
            )
            reference_max = float(
                detail.get("max_abs_discrepancy_vs_global_reference", 0.0)
            )
            reference_mean = float(
                detail.get("mean_abs_discrepancy_vs_global_reference", 0.0)
            )
            values = (pair_max, pair_mean, reference_max, reference_mean)
            if not all(np.isfinite(value) for value in values):
                finite_discrepancies = False

            detail_tolerance = float(
                detail.get("tolerance", PRIOR_RECONSTRUCTION_AUDIT_ATOL)
            )
            if np.isfinite(pair_max) and pair_max > detail_tolerance:
                overlap_disagreement_timesteps.append(timestep)
            if np.isfinite(reference_max) and reference_max > detail_tolerance:
                global_reference_disagreement_timesteps.append(timestep)

            max_pairwise_overlap_discrepancy = max(
                max_pairwise_overlap_discrepancy, pair_max
            )
            if pair_count > 0:
                pairwise_weighted_sum += pair_mean * pair_count
                pairwise_comparison_count += pair_count
            max_reference_discrepancy = max(
                max_reference_discrepancy, reference_max
            )
            if observation_count > 0:
                reference_weighted_sum += reference_mean * observation_count
                reference_observation_count += observation_count

            for discrepancy, kind in (
                (pair_max, "pairwise_overlap"),
                (reference_max, "versus_global_reference"),
            ):
                if (
                    discrepancy > worst_discrepancy
                    or (
                        discrepancy == worst_discrepancy
                        and affected_worst_timestep is not None
                        and timestep < affected_worst_timestep
                    )
                ):
                    worst_discrepancy = discrepancy
                    affected_worst_timestep = timestep
                    affected_worst_kind = kind

            if not bool(detail.get("passed", False)):
                failed_detail_timesteps.append(timestep)

        overlap_disagreement_timesteps = sorted(
            set(overlap_disagreement_timesteps)
        )
        global_reference_disagreement_timesteps = sorted(
            set(global_reference_disagreement_timesteps)
        )

        mean_pairwise_overlap_discrepancy = (
            pairwise_weighted_sum / pairwise_comparison_count
            if pairwise_comparison_count > 0
            else 0.0
        )
        mean_reference_discrepancy = (
            reference_weighted_sum / reference_observation_count
            if reference_observation_count > 0
            else 0.0
        )

        failure_reasons = []
        if len(source_starts) != expected_window_count:
            failure_reasons.append(
                f"source window count {len(source_starts)} != expected "
                f"{expected_window_count}"
            )
        if missing_window_starts:
            failure_reasons.append(
                f"missing expected window starts {missing_window_starts}"
            )
        if unexpected_start_indices:
            failure_reasons.append(
                f"unexpected window starts {unexpected_start_indices}"
            )
        if duplicate_start_indices:
            failure_reasons.append(
                f"duplicate window starts {duplicate_start_indices}"
            )
        if path_name_disagreement:
            failure_reasons.append(
                "path-name disagreement "
                f"missing={missing_artifact_path_names}, "
                f"unexpected={unexpected_artifact_path_names}"
            )
        if not monotonic_start_pass:
            failure_reasons.append("window starts are not strictly monotonic")
        if missing_global_indices:
            failure_reasons.append(
                "missing reconstructed full-trajectory timesteps "
                f"{missing_global_indices}"
            )
        if duplicate_detail_timesteps:
            failure_reasons.append(
                f"duplicate detail timesteps {duplicate_detail_timesteps}"
            )
        if not tolerance_consistent:
            failure_reasons.append(
                f"inconsistent detail tolerances {tolerance_values}"
            )
        if not finite_discrepancies:
            failure_reasons.append("non-finite discrepancy in detail rows")
        if overlap_disagreement_timesteps:
            failure_reasons.append(
                "overlap disagreement at timesteps "
                f"{overlap_disagreement_timesteps}"
            )
        if global_reference_disagreement_timesteps:
            failure_reasons.append(
                "global-reference disagreement at timesteps "
                f"{global_reference_disagreement_timesteps}"
            )
        otherwise_failed_detail_timesteps = sorted(
            set(failed_detail_timesteps)
            .difference(overlap_disagreement_timesteps)
            .difference(global_reference_disagreement_timesteps)
        )
        if otherwise_failed_detail_timesteps:
            failure_reasons.append(
                "other failed reconstruction details at timesteps "
                f"{otherwise_failed_detail_timesteps}"
            )

        reconstruction_pass = not failure_reasons
        explanation = (
            "PASS: complete stride-aware expected-window and full-trajectory "
            "coverage; path names agree; all overlapping prior observations and "
            "global-reference comparisons agree within tolerance."
            if reconstruction_pass
            else "FAIL: " + "; ".join(failure_reasons)
        )
        aggregate_rows.append(
            {
                "path_name": name,
                "trajectory_length": int(trajectory_length),
                "horizon": int(horizon),
                "stride": int(resolved_stride),
                "stride_source": stride_source,
                "number_source_windows": int(len(source_starts)),
                "expected_windows": int(expected_window_count),
                "expected_start_indices": expected_start_indices,
                "missing_expected_start_indices": missing_window_starts,
                "unexpected_start_indices": unexpected_start_indices,
                "out_of_range_start_indices": sorted(set(out_of_range_starts)),
                "duplicate_start_indices": duplicate_start_indices,
                "duplicate_start_index_count": int(
                    len(duplicate_start_indices)
                ),
                "missing_artifact_path_names": missing_artifact_path_names,
                "unexpected_artifact_path_names": unexpected_artifact_path_names,
                "path_name_disagreement": path_name_disagreement,
                "path_name_present_in_window_artifact": bool(
                    name in artifact_name_set
                ),
                "missing_reconstructed_full_trajectory_timesteps": (
                    missing_global_indices
                ),
                "missing_global_indices": missing_global_indices,
                "duplicate_overlap_global_indices": overlap_global_indices,
                "duplicate_overlap_global_index_count": int(
                    len(overlap_global_indices)
                ),
                "overlap_disagreement_timesteps": (
                    overlap_disagreement_timesteps
                ),
                "overlap_disagreement": bool(
                    overlap_disagreement_timesteps
                ),
                "global_reference_disagreement_timesteps": (
                    global_reference_disagreement_timesteps
                ),
                "maximum_pairwise_overlap_discrepancy": float(
                    max_pairwise_overlap_discrepancy
                ),
                "mean_pairwise_overlap_discrepancy": float(
                    mean_pairwise_overlap_discrepancy
                ),
                "maximum_versus_global_reference_discrepancy": float(
                    max_reference_discrepancy
                ),
                "mean_versus_global_reference_discrepancy": float(
                    mean_reference_discrepancy
                ),
                "affected_worst_timestep": affected_worst_timestep,
                "affected_worst_discrepancy_kind": affected_worst_kind,
                "tolerance": tolerance,
                "monotonic_start_pass": monotonic_start_pass,
                "reconstruction_pass": reconstruction_pass,
                "explanation": explanation,
            }
        )

    return aggregate_rows


def _numeric_array_csv_rows(
    values: Any,
    column_names: Sequence[str],
    *,
    label: str,
) -> List[Dict[str, Any]]:
    """Convert a finite two-dimensional numeric array to timestep CSV rows."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[0] == 0:
        raise ValueError(f"{label} must be a non-empty two-dimensional array")
    if array.shape[1] != len(column_names):
        raise ValueError(
            f"{label} has {array.shape[1]} columns but "
            f"{len(column_names)} column names were supplied"
        )
    if len(set(column_names)) != len(column_names):
        raise ValueError(f"{label} column names must be unique")
    if "timestep" in column_names:
        raise ValueError(f"{label} data columns cannot be named 'timestep'")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} must contain only finite values")
    return [
        {
            "timestep": int(timestep),
            **{
                str(column_name): float(array[timestep, column_index])
                for column_index, column_name in enumerate(column_names)
            },
        }
        for timestep in range(array.shape[0])
    ]


def write_fk_validation_path_trajectories(
    output_dir: Path,
    records: Sequence[Mapping[str, Any]],
) -> List[Path]:
    """Write standalone per-path FK trajectories and return every written path."""

    written_paths: List[Path] = []
    seen_safe_names: Dict[str, str] = {}
    for record in sorted(
        records,
        key=lambda item: (int(item["path_index"]), str(item["path_name"])),
    ):
        path_name = str(record["path_name"])
        safe_name = rollout.safe_path_name(path_name)
        if safe_name in seen_safe_names:
            raise ValueError(
                "FK validation output path-name collision after sanitization: "
                f"{seen_safe_names[safe_name]!r} and {path_name!r} -> {safe_name!r}"
            )
        seen_safe_names[safe_name] = path_name
        path_output_dir = Path(output_dir) / "fk_validation_paths" / safe_name
        joint_names = tuple(str(name) for name in record["joint_names"])
        if len(joint_names) != 6 or len(set(joint_names)) != 6:
            raise ValueError(
                f"FK trajectory writer requires six unique joint names for {path_name}"
            )

        def write_numeric_csv(
            filename: str,
            values: Any,
            column_names: Sequence[str],
        ) -> None:
            path = path_output_dir / filename
            write_records_csv(
                path,
                _numeric_array_csv_rows(
                    values,
                    column_names,
                    label=f"{path_name} {filename}",
                ),
            )
            written_paths.append(path)

        write_numeric_csv(
            "desired_path.csv",
            record["desired"],
            ("x", "y", "z"),
        )
        write_numeric_csv("expert_q.csv", record["expert"]["q"], joint_names)
        write_numeric_csv(
            "expert_ee.csv",
            record["expert"]["ee"],
            ("x", "y", "z"),
        )
        write_numeric_csv(
            "global_reference_q.csv",
            record["global_reference"]["q"],
            joint_names,
        )
        write_numeric_csv(
            "global_reference_ee.csv",
            record["global_reference"]["ee"],
            ("x", "y", "z"),
        )
        if record.get("buffer_only") is not None:
            write_numeric_csv(
                "buffer_only_q.csv",
                record["buffer_only"]["q"],
                joint_names,
            )
            write_numeric_csv(
                "buffer_only_ee.csv",
                record["buffer_only"]["ee"],
                ("x", "y", "z"),
            )

        diffusion_entries = record["base_tail_diffusion_by_seed"]
        if not diffusion_entries:
            # Diagnostic-only mode, and full-mode FK paths outside the benchmark
            # cohort, intentionally have no rollout trajectory. Desired/expert/
            # global files above remain complete; skip diffusion-specific files.
            continue
        seeds = sorted(int(value) for value in diffusion_entries)
        for seed in seeds:
            entry = diffusion_entries[seed]
            write_numeric_csv(
                f"base_tail_diffusion_seed_{seed}_q.csv",
                entry["q"],
                joint_names,
            )
            write_numeric_csv(
                f"base_tail_diffusion_seed_{seed}_ee.csv",
                entry["ee"],
                ("x", "y", "z"),
            )

        medoid_seed = int(record["base_tail_diffusion_medoid_seed"])
        if medoid_seed not in diffusion_entries:
            raise ValueError(
                f"diffusion medoid seed {medoid_seed} is absent for {path_name}"
            )
        medoid_entry = diffusion_entries[medoid_seed]
        # These unsuffixed files are copies of one actual seed, never averaged q.
        write_numeric_csv(
            "base_tail_diffusion_q.csv",
            medoid_entry["q"],
            joint_names,
        )
        write_numeric_csv(
            "base_tail_diffusion_ee.csv",
            medoid_entry["ee"],
            ("x", "y", "z"),
        )

        desired = validate_cartesian_trajectory(
            record["desired"],
            label=f"desired trajectory for seed aggregate {path_name}",
        )
        ee_stack = np.stack(
            [
                validate_cartesian_trajectory(
                    diffusion_entries[seed]["ee"],
                    label=f"diffusion seed {seed} EE for {path_name}",
                    expected_steps=desired.shape[0],
                )
                for seed in seeds
            ],
            axis=0,
        )
        error_norms = np.linalg.norm(
            ee_stack - desired[np.newaxis, :, :],
            axis=2,
        )
        ee_mean = np.mean(ee_stack, axis=0)
        ee_std = np.std(ee_stack, axis=0, ddof=0)
        ee_min = np.min(ee_stack, axis=0)
        ee_max = np.max(ee_stack, axis=0)
        error_mean = np.mean(error_norms, axis=0)
        error_std = np.std(error_norms, axis=0, ddof=0)
        error_max = np.max(error_norms, axis=0)
        aggregate_rows = []
        for timestep in range(desired.shape[0]):
            row: Dict[str, Any] = {
                "timestep": int(timestep),
                "seed_count": int(len(seeds)),
                "medoid_seed": medoid_seed,
                "error_norm_mean": float(error_mean[timestep]),
                "error_norm_std": float(error_std[timestep]),
                "error_norm_max": float(error_max[timestep]),
            }
            for axis_index, axis in enumerate(("x", "y", "z")):
                row[f"ee_mean_{axis}"] = float(ee_mean[timestep, axis_index])
                row[f"ee_std_{axis}"] = float(ee_std[timestep, axis_index])
                row[f"ee_min_{axis}"] = float(ee_min[timestep, axis_index])
                row[f"ee_max_{axis}"] = float(ee_max[timestep, axis_index])
            aggregate_rows.append(row)
        aggregate_path = (
            path_output_dir / "base_tail_diffusion_seed_aggregate_summary.csv"
        )
        write_records_csv(aggregate_path, aggregate_rows)
        written_paths.append(aggregate_path)
    return written_paths


def set_equal_3d_axes_from_trajectories(
    ax: Any,
    trajectories: Sequence[Any],
    *,
    degenerate_span_floor: float = 1e-6,
) -> Dict[str, Any]:
    """Set physically equal XYZ limits from the combined finite trajectories."""

    if not math.isfinite(degenerate_span_floor) or degenerate_span_floor <= 0.0:
        raise ValueError("degenerate_span_floor must be finite and > 0")
    validated = [
        validate_cartesian_trajectory(
            trajectory,
            label=f"equal-axis trajectory {index}",
        )
        for index, trajectory in enumerate(trajectories)
    ]
    if not validated:
        raise ValueError("at least one trajectory is required for equal 3D axes")
    combined = np.concatenate(validated, axis=0)
    minimum = np.min(combined, axis=0)
    maximum = np.max(combined, axis=0)
    center = 0.5 * (minimum + maximum)
    spans = maximum - minimum
    equal_span = max(float(np.max(spans)), float(degenerate_span_floor))
    half_span = 0.5 * equal_span
    ax.set_xlim(center[0] - half_span, center[0] + half_span)
    ax.set_ylim(center[1] - half_span, center[1] + half_span)
    ax.set_zlim(center[2] - half_span, center[2] + half_span)
    ax.set_box_aspect((1.0, 1.0, 1.0))
    return {
        "combined_min": minimum.tolist(),
        "combined_max": maximum.tolist(),
        "combined_center": center.tolist(),
        "axis_spans_before_equalization": spans.tolist(),
        "equal_physical_axis_span": equal_span,
        "degenerate_span_floor": float(degenerate_span_floor),
    }


def publish_fk_validation_spatial_plots(
    output_dir: Path,
    record: Mapping[str, Any],
) -> List[Path]:
    """Publish five standalone spatial diagnostics for one validation path."""

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    path_name = str(record["path_name"])
    path_output_dir = (
        Path(output_dir)
        / "fk_validation_paths"
        / rollout.safe_path_name(path_name)
    )
    desired = validate_cartesian_trajectory(
        record["desired"],
        label=f"spatial plot desired {path_name}",
    )
    expert = validate_cartesian_trajectory(
        record["expert"]["ee"],
        label=f"spatial plot expert {path_name}",
        expected_steps=desired.shape[0],
    )
    global_reference = validate_cartesian_trajectory(
        record["global_reference"]["ee"],
        label=f"spatial plot global reference {path_name}",
        expected_steps=desired.shape[0],
    )
    buffer_only = None
    if record.get("buffer_only") is not None:
        buffer_only = validate_cartesian_trajectory(
            record["buffer_only"]["ee"],
            label=f"spatial plot buffer-only {path_name}",
            expected_steps=desired.shape[0],
        )
    diffusion_entries = record["base_tail_diffusion_by_seed"]
    seeds = sorted(int(value) for value in diffusion_entries)
    diffusion_by_seed = {
        seed: validate_cartesian_trajectory(
            diffusion_entries[seed]["ee"],
            label=f"spatial plot diffusion seed {seed} {path_name}",
            expected_steps=desired.shape[0],
        )
        for seed in seeds
    }
    medoid_seed = (
        int(record["base_tail_diffusion_medoid_seed"]) if seeds else None
    )
    if medoid_seed is not None and medoid_seed not in diffusion_by_seed:
        raise ValueError(
            f"diffusion medoid seed {medoid_seed} is absent for {path_name}"
        )
    medoid = diffusion_by_seed[medoid_seed] if medoid_seed is not None else None

    def plot_3d_trajectories(ax: Any, *, show_legend: bool) -> None:
        ax.plot(
            desired[:, 0],
            desired[:, 1],
            desired[:, 2],
            color="#111111",
            linewidth=2.5,
            label="desired",
            zorder=8,
        )
        ax.plot(
            expert[:, 0],
            expert[:, 1],
            expert[:, 2],
            color="#54A24B",
            linewidth=1.8,
            label="expert FK",
            zorder=6,
        )
        ax.plot(
            global_reference[:, 0],
            global_reference[:, 1],
            global_reference[:, 2],
            color="#4C78A8",
            linewidth=1.8,
            label="global prior FK",
            zorder=5,
        )
        if buffer_only is not None:
            ax.plot(
                buffer_only[:, 0],
                buffer_only[:, 1],
                buffer_only[:, 2],
                color="#F2CF5B",
                linewidth=1.5,
                label="buffer-only FK",
                zorder=4,
            )
        for seed_index, seed in enumerate(seeds):
            trajectory = diffusion_by_seed[seed]
            ax.plot(
                trajectory[:, 0],
                trajectory[:, 1],
                trajectory[:, 2],
                color="#E45756",
                linewidth=0.8,
                alpha=0.16,
                label="diffusion seeds" if seed_index == 0 else None,
                zorder=2,
            )
        if medoid is not None:
            ax.plot(
                medoid[:, 0],
                medoid[:, 1],
                medoid[:, 2],
                color="#B22222",
                linewidth=2.4,
                label=f"diffusion medoid (seed {medoid_seed})",
                zorder=7,
            )
        ax.set_xlabel("x (m)")
        ax.set_ylabel("y (m)")
        ax.set_zlabel("z (m)")
        ax.grid(alpha=0.2)
        if show_legend:
            ax.legend(loc="best", fontsize=8)

    all_trajectories = [desired, expert, global_reference]
    output_trajectories = [expert, global_reference]
    if buffer_only is not None:
        all_trajectories.append(buffer_only)
        output_trajectories.append(buffer_only)
    all_trajectories.extend(diffusion_by_seed[seed] for seed in seeds)
    output_trajectories.extend(diffusion_by_seed[seed] for seed in seeds)
    written_paths: List[Path] = []

    equal_path = path_output_dir / (
        "desired_expert_prior_diffusion_3d_equal_axes.png"
    )
    fig = plt.figure(figsize=(9.2, 7.8))
    equal_ax = fig.add_subplot(111, projection="3d")
    plot_3d_trajectories(equal_ax, show_legend=True)
    set_equal_3d_axes_from_trajectories(equal_ax, all_trajectories)
    equal_ax.set_title(f"{path_name}: full-frame equal physical XYZ scales")
    _save_figure_impl(fig, equal_path)
    written_paths.append(equal_path)

    auto_path = path_output_dir / (
        "desired_expert_prior_diffusion_3d_auto_axes.png"
    )
    fig = plt.figure(figsize=(15.0, 6.8))
    full_ax = fig.add_subplot(121, projection="3d")
    close_ax = fig.add_subplot(122, projection="3d")
    plot_3d_trajectories(full_ax, show_legend=True)
    full_ax.set_title("Full frame (desired + all FK outputs)")
    plot_3d_trajectories(close_ax, show_legend=False)
    combined_output = np.concatenate(output_trajectories, axis=0)
    output_minimum = np.min(combined_output, axis=0)
    output_maximum = np.max(combined_output, axis=0)
    output_span = output_maximum - output_minimum
    output_center = 0.5 * (output_minimum + output_maximum)
    close_half_spans = np.maximum(0.55 * output_span, 0.5e-6)
    close_ax.set_xlim(
        output_center[0] - close_half_spans[0],
        output_center[0] + close_half_spans[0],
    )
    close_ax.set_ylim(
        output_center[1] - close_half_spans[1],
        output_center[1] + close_half_spans[1],
    )
    close_ax.set_zlim(
        output_center[2] - close_half_spans[2],
        output_center[2] + close_half_spans[2],
    )
    desired_centroid = np.mean(desired, axis=0)
    desired_range = np.ptp(desired, axis=0)
    close_ax.scatter(
        [desired_centroid[0]],
        [desired_centroid[1]],
        [desired_centroid[2]],
        marker="x",
        s=55,
        linewidths=2,
        color="#111111",
        label="desired centroid",
    )
    close_ax.text2D(
        0.02,
        0.98,
        (
            "Desired centroid (m): "
            f"({desired_centroid[0]:.4g}, {desired_centroid[1]:.4g}, "
            f"{desired_centroid[2]:.4g})\n"
            "Desired range XYZ (m): "
            f"({desired_range[0]:.4g}, {desired_range[1]:.4g}, "
            f"{desired_range[2]:.4g})\n"
            "Centroid/range reported even when outside this FK-output close-up."
        ),
        transform=close_ax.transAxes,
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "#777777"},
    )
    close_ax.set_title("FK-output close-up (auto limits)")
    fig.suptitle(f"{path_name}: full-frame and FK-output close-up")
    _save_figure_impl(fig, auto_path)
    written_paths.append(auto_path)

    projection_specs = (
        ("xy", 0, 1, "x (m)", "y (m)"),
        ("xz", 0, 2, "x (m)", "z (m)"),
        ("yz", 1, 2, "y (m)", "z (m)"),
    )
    for projection_name, horizontal, vertical, xlabel, ylabel in projection_specs:
        projection_path = path_output_dir / (
            f"desired_expert_prior_diffusion_{projection_name}.png"
        )
        fig, ax = plt.subplots(figsize=(8.2, 7.2))
        ax.plot(
            desired[:, horizontal],
            desired[:, vertical],
            color="#111111",
            linewidth=2.5,
            label="desired",
            zorder=8,
        )
        ax.plot(
            expert[:, horizontal],
            expert[:, vertical],
            color="#54A24B",
            linewidth=1.8,
            label="expert FK",
            zorder=6,
        )
        ax.plot(
            global_reference[:, horizontal],
            global_reference[:, vertical],
            color="#4C78A8",
            linewidth=1.8,
            label="global prior FK",
            zorder=5,
        )
        if buffer_only is not None:
            ax.plot(
                buffer_only[:, horizontal],
                buffer_only[:, vertical],
                color="#F2CF5B",
                linewidth=1.5,
                label="buffer-only FK",
                zorder=4,
            )
        for seed_index, seed in enumerate(seeds):
            trajectory = diffusion_by_seed[seed]
            ax.plot(
                trajectory[:, horizontal],
                trajectory[:, vertical],
                color="#E45756",
                linewidth=0.8,
                alpha=0.16,
                label="diffusion seeds" if seed_index == 0 else None,
                zorder=2,
            )
        if medoid is not None:
            ax.plot(
                medoid[:, horizontal],
                medoid[:, vertical],
                color="#B22222",
                linewidth=2.4,
                label=f"diffusion medoid (seed {medoid_seed})",
                zorder=7,
            )
        combined_projection = np.concatenate(
            [
                trajectory[:, (horizontal, vertical)]
                for trajectory in all_trajectories
            ],
            axis=0,
        )
        projection_minimum = np.min(combined_projection, axis=0)
        projection_maximum = np.max(combined_projection, axis=0)
        projection_center = 0.5 * (projection_minimum + projection_maximum)
        projection_span = max(
            float(np.max(projection_maximum - projection_minimum)),
            1e-6,
        )
        projection_half_span = 0.5 * projection_span
        ax.set_xlim(
            projection_center[0] - projection_half_span,
            projection_center[0] + projection_half_span,
        )
        ax.set_ylim(
            projection_center[1] - projection_half_span,
            projection_center[1] + projection_half_span,
        )
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{path_name}: {projection_name.upper()} equal-scale projection")
        ax.grid(alpha=0.2)
        ax.legend(loc="best", fontsize=8)
        _save_figure_impl(fig, projection_path)
        written_paths.append(projection_path)
    return written_paths


def publish_fk_validation_diagnostic_plots(
    output_dir: Path,
    record: Mapping[str, Any],
    *,
    plot_equal_axes: bool,
) -> List[Path]:
    """Publish the four non-spatial standalone diagnostics for one path."""

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    path_name = str(record["path_name"])
    path_output_dir = (
        Path(output_dir)
        / "fk_validation_paths"
        / rollout.safe_path_name(path_name)
    )
    desired = validate_cartesian_trajectory(
        record["desired"],
        label=f"diagnostic plot desired {path_name}",
    )
    expert = validate_cartesian_trajectory(
        record["expert"]["ee"],
        label=f"diagnostic plot expert {path_name}",
        expected_steps=desired.shape[0],
    )
    global_reference = validate_cartesian_trajectory(
        record["global_reference"]["ee"],
        label=f"diagnostic plot global reference {path_name}",
        expected_steps=desired.shape[0],
    )
    buffer_only = None
    if record.get("buffer_only") is not None:
        buffer_only = validate_cartesian_trajectory(
            record["buffer_only"]["ee"],
            label=f"diagnostic plot buffer-only {path_name}",
            expected_steps=desired.shape[0],
        )
    diffusion_entries = record["base_tail_diffusion_by_seed"]
    seeds = sorted(int(value) for value in diffusion_entries)
    diffusion_by_seed = {
        seed: validate_cartesian_trajectory(
            diffusion_entries[seed]["ee"],
            label=f"diagnostic plot diffusion seed {seed} {path_name}",
            expected_steps=desired.shape[0],
        )
        for seed in seeds
    }
    medoid_seed = (
        int(record["base_tail_diffusion_medoid_seed"]) if seeds else None
    )
    if medoid_seed is not None and medoid_seed not in diffusion_by_seed:
        raise ValueError(
            f"diffusion medoid seed {medoid_seed} is absent for {path_name}"
        )
    medoid = diffusion_by_seed[medoid_seed] if medoid_seed is not None else None
    option_label = f"plot_equal_axes={bool(plot_equal_axes)}"
    timesteps = np.arange(desired.shape[0], dtype=np.int64)
    written_paths: List[Path] = []

    error_path = path_output_dir / "cartesian_error_over_time_all_methods.png"
    fig, ax = plt.subplots(figsize=(11.5, 5.8))
    ax.plot(
        timesteps,
        np.linalg.norm(expert - desired, axis=1),
        color="#54A24B",
        linewidth=1.8,
        label="expert FK",
        zorder=6,
    )
    ax.plot(
        timesteps,
        np.linalg.norm(global_reference - desired, axis=1),
        color="#4C78A8",
        linewidth=1.8,
        label="global prior FK",
        zorder=5,
    )
    if buffer_only is not None:
        ax.plot(
            timesteps,
            np.linalg.norm(buffer_only - desired, axis=1),
            color="#F2CF5B",
            linewidth=1.5,
            label="buffer-only FK",
            zorder=4,
        )
    for seed_index, seed in enumerate(seeds):
        ax.plot(
            timesteps,
            np.linalg.norm(diffusion_by_seed[seed] - desired, axis=1),
            color="#E45756",
            linewidth=0.8,
            alpha=0.16,
            label="diffusion seeds" if seed_index == 0 else None,
            zorder=2,
        )
    if medoid is not None:
        ax.plot(
            timesteps,
            np.linalg.norm(medoid - desired, axis=1),
            color="#B22222",
            linewidth=2.3,
            label=f"diffusion medoid (seed {medoid_seed})",
            zorder=7,
        )
    ax.text(
        0.01,
        0.98,
        (
            "Trajectory-level mean-error criterion: "
            f"{float(record['mean_threshold']):.6g} m "
            "(annotation only; not a pointwise bound)"
        ),
        transform=ax.transAxes,
        va="top",
        fontsize=8,
        color="#6F42A1",
        bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "#9467BD"},
    )
    ax.axhline(
        float(record["max_threshold"]),
        color="#8C564B",
        linestyle=":",
        linewidth=1.4,
        label="configured max-error threshold",
    )
    ax.set_xlabel("global timestep")
    ax.set_ylabel("Cartesian error norm (m)")
    ax.set_title(f"{path_name}: Cartesian error over time ({option_label})")
    ax.grid(alpha=0.2)
    ax.legend(loc="best", fontsize=8, ncol=2)
    _save_figure_impl(fig, error_path)
    written_paths.append(error_path)

    xyz_path = path_output_dir / "cartesian_error_xyz_over_time.png"
    fig, axes = plt.subplots(3, 1, figsize=(11.5, 9.0), sharex=True)
    for axis_index, (ax, axis_name) in enumerate(zip(axes, ("x", "y", "z"))):
        ax.plot(
            timesteps,
            expert[:, axis_index] - desired[:, axis_index],
            color="#54A24B",
            linewidth=1.6,
            label="expert FK",
            zorder=6,
        )
        ax.plot(
            timesteps,
            global_reference[:, axis_index] - desired[:, axis_index],
            color="#4C78A8",
            linewidth=1.6,
            label="global prior FK",
            zorder=5,
        )
        if buffer_only is not None:
            ax.plot(
                timesteps,
                buffer_only[:, axis_index] - desired[:, axis_index],
                color="#F2CF5B",
                linewidth=1.4,
                label="buffer-only FK",
                zorder=4,
            )
        for seed_index, seed in enumerate(seeds):
            ax.plot(
                timesteps,
                diffusion_by_seed[seed][:, axis_index] - desired[:, axis_index],
                color="#E45756",
                linewidth=0.7,
                alpha=0.16,
                label="diffusion seeds" if seed_index == 0 else None,
                zorder=2,
            )
        if medoid is not None:
            ax.plot(
                timesteps,
                medoid[:, axis_index] - desired[:, axis_index],
                color="#B22222",
                linewidth=2.1,
                label=f"diffusion medoid (seed {medoid_seed})",
                zorder=7,
            )
        ax.axhline(0.0, color="black", linewidth=0.9)
        ax.set_ylabel(f"signed {axis_name} error (m)")
        ax.grid(alpha=0.2)
    axes[-1].set_xlabel("global timestep")
    axes[0].legend(loc="best", fontsize=8, ncol=2)
    fig.suptitle(
        f"{path_name}: observed FK minus desired by coordinate ({option_label})"
    )
    _save_figure_impl(fig, xyz_path)
    written_paths.append(xyz_path)

    expert_diagnostics = record["expert"]["diagnostics"]
    centroid_shift = np.asarray(
        expert_diagnostics["translation"]["centroid_alignment"][
            "translation_applied_to_observed"
        ],
        dtype=np.float64,
    )
    start_shift = np.asarray(
        expert_diagnostics["translation"]["start_alignment"][
            "translation_applied_to_observed"
        ],
        dtype=np.float64,
    )
    rigid_diagnostics = expert_diagnostics["rigid"]
    rigid_rotation = np.asarray(
        rigid_diagnostics["rotation_observed_to_reference"], dtype=np.float64
    )
    rigid_translation = np.asarray(
        rigid_diagnostics["translation_observed_to_reference"], dtype=np.float64
    )
    centroid_aligned = expert + centroid_shift
    start_aligned = expert + start_shift
    rigid_aligned = expert @ rigid_rotation.T + rigid_translation
    alignment_path = path_output_dir / "expert_alignment_comparison.png"
    fig = plt.figure(figsize=(10.0, 8.0))
    ax = fig.add_subplot(111, projection="3d")
    for trajectory, color, linewidth, linestyle, label in (
        (desired, "#111111", 2.6, "-", "desired"),
        (expert, "#E45756", 1.8, "-", "original expert FK"),
        (
            centroid_aligned,
            "#4C78A8",
            1.5,
            "--",
            "centroid-aligned expert (diagnostic)",
        ),
        (
            start_aligned,
            "#F2CF5B",
            1.5,
            "--",
            "start-aligned expert (diagnostic)",
        ),
        (
            rigid_aligned,
            "#54A24B",
            1.8,
            "-.",
            "rigid-aligned expert (diagnostic)",
        ),
    ):
        ax.plot(
            trajectory[:, 0],
            trajectory[:, 1],
            trajectory[:, 2],
            color=color,
            linewidth=linewidth,
            linestyle=linestyle,
            label=label,
        )
    if plot_equal_axes:
        set_equal_3d_axes_from_trajectories(
            ax,
            [desired, expert, centroid_aligned, start_aligned, rigid_aligned],
        )
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.set_title(
        f"DIAGNOSTIC expert alignment comparison — {path_name} ({option_label})"
    )
    ax.text2D(
        0.02,
        0.98,
        "DIAGNOSTIC ALIGNMENTS ONLY — NOT SCORED TRAJECTORIES",
        transform=ax.transAxes,
        va="top",
        color="#A00000",
        weight="bold",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "#A00000"},
    )
    ax.grid(alpha=0.2)
    ax.legend(loc="best", fontsize=8)
    _save_figure_impl(fig, alignment_path)
    written_paths.append(alignment_path)

    range_path = path_output_dir / "coordinate_range_comparison.png"
    diffusion_combined = (
        np.concatenate([diffusion_by_seed[seed] for seed in seeds], axis=0)
        if seeds
        else None
    )
    range_series: List[Tuple[str, np.ndarray, str]] = [
        ("desired", desired, "#111111"),
        ("expert FK", expert, "#54A24B"),
        ("global prior FK", global_reference, "#4C78A8"),
    ]
    if buffer_only is not None:
        range_series.append(("buffer-only FK", buffer_only, "#F2CF5B"))
    if diffusion_combined is not None and medoid is not None:
        range_series.extend(
            [
                ("diffusion seed envelope", diffusion_combined, "#E45756"),
                (f"diffusion medoid seed {medoid_seed}", medoid, "#B22222"),
            ]
        )
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 6.4), sharey=True)
    y_positions = np.arange(len(range_series))
    for axis_index, (ax, axis_name) in enumerate(zip(axes, ("x", "y", "z"))):
        for y_position, (label, trajectory, color) in enumerate(range_series):
            coordinate = trajectory[:, axis_index]
            minimum = float(np.min(coordinate))
            maximum = float(np.max(coordinate))
            centroid = float(np.mean(coordinate))
            ax.hlines(
                y_position,
                minimum,
                maximum,
                color=color,
                linewidth=4.0 if "envelope" in label else 2.5,
                alpha=0.65 if "envelope" in label else 0.9,
            )
            ax.scatter(
                [centroid],
                [y_position],
                color=color,
                edgecolors="black",
                linewidths=0.4,
                s=36,
                zorder=3,
            )
        ax.set_xlabel(f"{axis_name} coordinate (m)")
        ax.set_title(f"{axis_name.upper()} min—centroid—max")
        ax.set_yticks(y_positions, [item[0] for item in range_series])
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.2)
    fig.suptitle(
        f"{path_name}: coordinate ranges and centroids ({option_label})"
    )
    _save_figure_impl(fig, range_path)
    written_paths.append(range_path)
    return written_paths


def publish_all_fk_validation_plots(
    output_dir: Path,
    records: Sequence[Mapping[str, Any]],
    plot_equal_axes: bool,
) -> List[Path]:
    """Publish and verify exactly nine standalone diagnostic PNGs per path."""

    expected_filenames = {
        "desired_expert_prior_diffusion_3d_equal_axes.png",
        "desired_expert_prior_diffusion_3d_auto_axes.png",
        "desired_expert_prior_diffusion_xy.png",
        "desired_expert_prior_diffusion_xz.png",
        "desired_expert_prior_diffusion_yz.png",
        "cartesian_error_over_time_all_methods.png",
        "cartesian_error_xyz_over_time.png",
        "expert_alignment_comparison.png",
        "coordinate_range_comparison.png",
    }
    ordered_records = sorted(
        records,
        key=lambda item: (int(item["path_index"]), str(item["path_name"])),
    )
    safe_name_owners: Dict[str, str] = {}
    for record in ordered_records:
        path_name = str(record["path_name"])
        safe_name = rollout.safe_path_name(path_name)
        if safe_name in safe_name_owners:
            raise ValueError(
                "FK validation plot path-name collision after sanitization: "
                f"{safe_name_owners[safe_name]!r} and {path_name!r} -> "
                f"{safe_name!r}"
            )
        safe_name_owners[safe_name] = path_name

    all_paths: List[Path] = []
    for record in ordered_records:
        spatial_paths = publish_fk_validation_spatial_plots(output_dir, record)
        diagnostic_paths = publish_fk_validation_diagnostic_plots(
            output_dir,
            record,
            plot_equal_axes=bool(plot_equal_axes),
        )
        record_paths = spatial_paths + diagnostic_paths
        actual_filenames = [path.name for path in record_paths]
        if (
            len(record_paths) != 9
            or len(set(actual_filenames)) != 9
            or set(actual_filenames) != expected_filenames
        ):
            raise RuntimeError(
                f"FK validation plot publication for {record['path_name']} must "
                "produce exactly the nine required filenames; got "
                f"{actual_filenames}"
            )
        expected_parent = (
            Path(output_dir)
            / "fk_validation_paths"
            / rollout.safe_path_name(str(record["path_name"]))
        )
        if any(path.parent != expected_parent for path in record_paths):
            raise RuntimeError(
                f"FK validation plots for {record['path_name']} were not all "
                f"published under {expected_parent}"
            )
        all_paths.extend(record_paths)
    return all_paths


FK_VALID_FRAME_PASS_FRACTION = 0.90


def interpret_fk_validation(
    expert_summary_rows: Sequence[Mapping[str, Any]],
    prior_audit_rows: Sequence[Mapping[str, Any]],
    records: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    """Interpret independent FK, dataset, reconstruction, and tracking decisions."""

    if not expert_summary_rows:
        raise ValueError("expert FK interpretation requires at least one path row")

    def pass_value(row: Mapping[str, Any], candidate_keys: Sequence[str]) -> bool:
        for key in candidate_keys:
            if key not in row:
                continue
            value = row[key]
            if isinstance(value, str):
                normalized = value.strip().lower()
                if normalized in {"1", "true", "pass", "passed", "yes"}:
                    return True
                if normalized in {"0", "false", "fail", "failed", "no", ""}:
                    return False
            return bool(value)
        raise KeyError(
            "audit/summary row lacks a pass field; expected one of "
            + ", ".join(candidate_keys)
        )

    category_counts = {classification: 0 for classification in FK_CLASSIFICATIONS}
    expert_paths: List[str] = []
    direct_means: List[float] = []
    rigid_means: List[float] = []
    pass_count = 0
    mismatch_families: set[str] = set()
    expert_dataset_failure_paths: List[str] = []
    for row in expert_summary_rows:
        path_name = str(row["path_name"])
        if path_name in expert_paths:
            raise ValueError(f"duplicate expert FK summary path: {path_name}")
        expert_paths.append(path_name)
        classification = str(row["classification"])
        if classification not in category_counts:
            raise ValueError(f"unknown expert FK classification: {classification}")
        category_counts[classification] += 1
        direct_mean = float(row["direct_mean_distance"])
        rigid_mean = float(row["rigid_aligned_mean_distance"])
        if not math.isfinite(direct_mean) or not math.isfinite(rigid_mean):
            raise ValueError(f"non-finite expert FK summary metric for {path_name}")
        direct_means.append(direct_mean)
        rigid_means.append(rigid_mean)
        direct_pass = pass_value(
            row,
            ("direct_pass", "validation_pass"),
        )
        if direct_pass:
            pass_count += 1
        if classification in {
            "POSSIBLE_FIXED_TRANSLATION",
            "POSSIBLE_RIGID_FRAME_MISMATCH",
        }:
            mismatch_families.add("fixed_frame")
        elif classification == "POSSIBLE_TEMPORAL_MISALIGNMENT":
            mismatch_families.add("temporal")
        elif classification == "EXPERT_SHAPE_MISMATCH":
            mismatch_families.add("dataset")
            expert_dataset_failure_paths.append(path_name)
        elif classification == "INCONCLUSIVE":
            mismatch_families.add("inconclusive")

    prior_paths: List[str] = []
    prior_pass_count = 0
    for row in prior_audit_rows:
        path_name = str(row["path_name"])
        if path_name in prior_paths:
            raise ValueError(f"duplicate prior reconstruction audit path: {path_name}")
        prior_paths.append(path_name)
        if pass_value(
            row,
            (
                "pass",
                "audit_pass",
                "reconstruction_pass",
                "validation_pass",
            ),
        ):
            prior_pass_count += 1
    expert_path_set = set(expert_paths)
    prior_path_set = set(prior_paths)
    prior_coverage_complete = (
        bool(prior_audit_rows)
        and prior_path_set == expert_path_set
        and len(prior_paths) == len(expert_paths)
    )
    prior_audit_performed = bool(prior_audit_rows)
    all_prior_audits_pass: Optional[bool] = (
        None
        if not prior_audit_performed
        else bool(
            prior_coverage_complete and prior_pass_count == len(prior_paths)
        )
    )
    prior_reconstruction_validity = (
        "NOT_PERFORMED"
        if not prior_audit_performed
        else "PASS"
        if all_prior_audits_pass
        else "FAIL"
    )

    expert_path_count = len(expert_summary_rows)
    pass_fraction = pass_count / expert_path_count
    all_expert_rows_pass_direct = bool(
        pass_count == expert_path_count
        and category_counts["PASS_DIRECT"] == expert_path_count
    )
    coordinate_fk_validity = (
        "PASS" if all_expert_rows_pass_direct else "FAIL"
    )
    if expert_dataset_failure_paths:
        expert_dataset_validity = "FAIL"
    elif all_expert_rows_pass_direct:
        expert_dataset_validity = "PASS"
    else:
        expert_dataset_validity = "INDETERMINATE"

    validation_records = [] if records is None else list(records)
    record_paths: List[str] = []
    global_prior_pass_paths: List[str] = []
    global_prior_failure_paths: List[str] = []
    global_prior_unperformed_paths: List[str] = []
    global_prior_direct_means: List[float] = []
    diffusion_pass_paths: List[str] = []
    diffusion_failure_paths: List[str] = []
    diffusion_unperformed_paths: List[str] = []
    diffusion_trajectory_count = 0
    for record in validation_records:
        path_name = str(record["path_name"])
        if path_name in record_paths:
            raise ValueError(f"duplicate FK validation record path: {path_name}")
        record_paths.append(path_name)
        mean_threshold = float(record["mean_threshold"])
        max_threshold = float(record["max_threshold"])

        global_entry = record.get("global_reference")
        if not isinstance(global_entry, Mapping):
            global_prior_unperformed_paths.append(path_name)
        else:
            global_direct = global_entry.get("diagnostics", {}).get("direct")
            if not isinstance(global_direct, Mapping):
                global_prior_unperformed_paths.append(path_name)
            else:
                global_mean = float(global_direct["mean_distance"])
                global_max = float(global_direct["max_distance"])
                if not math.isfinite(global_mean) or not math.isfinite(global_max):
                    raise ValueError(
                        f"non-finite global-prior FK metric for {path_name}"
                    )
                global_prior_direct_means.append(global_mean)
                if global_mean <= mean_threshold and global_max <= max_threshold:
                    global_prior_pass_paths.append(path_name)
                else:
                    global_prior_failure_paths.append(path_name)

        diffusion_entries = record.get("base_tail_diffusion_by_seed")
        if not isinstance(diffusion_entries, Mapping) or not diffusion_entries:
            diffusion_unperformed_paths.append(path_name)
            continue
        path_diffusion_pass = True
        for entry in diffusion_entries.values():
            if not isinstance(entry, Mapping):
                raise ValueError(
                    f"invalid diffusion FK entry for validation path {path_name}"
                )
            diffusion_direct = entry.get("diagnostics", {}).get("direct")
            if not isinstance(diffusion_direct, Mapping):
                raise ValueError(
                    f"diffusion FK entry lacks direct metrics for {path_name}"
                )
            diffusion_mean = float(diffusion_direct["mean_distance"])
            diffusion_max = float(diffusion_direct["max_distance"])
            if not math.isfinite(diffusion_mean) or not math.isfinite(diffusion_max):
                raise ValueError(
                    f"non-finite diffusion FK metric for {path_name}"
                )
            diffusion_trajectory_count += 1
            path_diffusion_pass = bool(
                path_diffusion_pass
                and diffusion_mean <= mean_threshold
                and diffusion_max <= max_threshold
            )
        if path_diffusion_pass:
            diffusion_pass_paths.append(path_name)
        else:
            diffusion_failure_paths.append(path_name)

    global_prior_performed_count = (
        len(global_prior_pass_paths) + len(global_prior_failure_paths)
    )
    if global_prior_performed_count == 0:
        prior_trajectory_quality = "NOT_PERFORMED"
    elif global_prior_failure_paths:
        prior_trajectory_quality = "FAIL"
    elif global_prior_unperformed_paths:
        prior_trajectory_quality = "PARTIAL"
    else:
        prior_trajectory_quality = "PASS"

    diffusion_performed_count = len(diffusion_pass_paths) + len(
        diffusion_failure_paths
    )
    if diffusion_performed_count == 0:
        diffusion_tracking_validity = "NOT_PERFORMED"
    elif diffusion_failure_paths:
        diffusion_tracking_validity = "FAIL"
    elif diffusion_unperformed_paths:
        diffusion_tracking_validity = "PARTIAL"
    else:
        diffusion_tracking_validity = "PASS"

    scientifically_valid = (
        coordinate_fk_validity == "PASS"
        and expert_dataset_validity == "PASS"
        and prior_reconstruction_validity == "PASS"
    )

    conclusions: List[str] = []
    if (
        coordinate_fk_validity == "PASS"
        and pass_fraction >= FK_VALID_FRAME_PASS_FRACTION
    ):
        conclusions.append("VALID_EVALUATION_FRAME")
    elif expert_dataset_validity == "FAIL":
        conclusions.append("EXPERT_DATASET_INCONSISTENCY")
    else:
        conclusions.append("MIXED_RESULT")
    if prior_reconstruction_validity == "FAIL":
        conclusions.append("PRIOR_RECONSTRUCTION_FAILURE")
    if prior_trajectory_quality == "FAIL":
        conclusions.append("GLOBAL_PRIOR_QUALITY_FAILURE")

    decision_explanations = {
        "coordinate_fk_validity": (
            "All selected expert trajectories are PASS_DIRECT."
            if coordinate_fk_validity == "PASS"
            else "One or more selected expert trajectories do not pass direct FK."
        ),
        "expert_dataset_validity": (
            "No expert trajectory has an EXPERT_SHAPE_MISMATCH classification."
            if expert_dataset_validity == "PASS"
            else (
                "Expert shape mismatch was found for "
                + ", ".join(expert_dataset_failure_paths)
                if expert_dataset_validity == "FAIL"
                else "Coordinate/temporal evidence prevents a dataset conclusion."
            )
        ),
        "prior_reconstruction_validity": (
            "The prior reconstruction audit was not performed."
            if prior_reconstruction_validity == "NOT_PERFORMED"
            else (
                "All requested prior reconstruction audits pass."
                if prior_reconstruction_validity == "PASS"
                else "At least one requested prior reconstruction audit fails or is missing."
            )
        ),
        "prior_trajectory_quality": (
            "Global-prior Cartesian quality was not evaluated."
            if prior_trajectory_quality == "NOT_PERFORMED"
            else (
                "Global-prior Cartesian quality exceeds the configured FK error "
                "threshold on: " + ", ".join(global_prior_failure_paths)
                if prior_trajectory_quality == "FAIL"
                else (
                    "Global-prior quality was evaluated for only part of the cohort."
                    if prior_trajectory_quality == "PARTIAL"
                    else "Global-prior Cartesian quality passes on every evaluated path."
                )
            )
        ),
        "diffusion_tracking_validity": (
            "Diffusion tracking was not performed for this validation cohort."
            if diffusion_tracking_validity == "NOT_PERFORMED"
            else (
                "Diffusion tracking exceeds configured FK error thresholds on: "
                + ", ".join(diffusion_failure_paths)
                if diffusion_tracking_validity == "FAIL"
                else (
                    "Diffusion tracking passed where performed; other selected "
                    "paths were not benchmarked."
                    if diffusion_tracking_validity == "PARTIAL"
                    else "Diffusion tracking passes on every evaluated path and seed."
                )
            )
        ),
    }

    explanation = " ".join(decision_explanations.values())

    scope_disclaimer = (
        " Coordinate/FK and expert-dataset conclusions are independent of prior "
        "reconstruction, global-prior quality, and diffusion tracking. An "
        "unperformed audit remains NOT_PERFORMED and is never converted to FAIL. "
        "This validation does not endorse or alter scientific_decision.csv."
    )
    explanation += scope_disclaimer

    return {
        "conclusion": conclusions[0],
        "conclusions": conclusions,
        "additional_conclusions": conclusions[1:],
        "conclusion_explanation": explanation,
        "decision_explanations": decision_explanations,
        "coordinate_fk_validity": coordinate_fk_validity,
        "expert_dataset_validity": expert_dataset_validity,
        "prior_reconstruction_validity": prior_reconstruction_validity,
        "prior_trajectory_quality": prior_trajectory_quality,
        "diffusion_tracking_validity": diffusion_tracking_validity,
        "interpretation_scope": (
            "SEPARATE_COORDINATE_DATASET_RECONSTRUCTION_PRIOR_AND_DIFFUSION_DECISIONS"
        ),
        "not_evidence_of_diffusion_or_buffer_success": True,
        "does_not_endorse_scientific_decision_csv": True,
        "valid_frame_pass_fraction_threshold": FK_VALID_FRAME_PASS_FRACTION,
        "expert_path_count": expert_path_count,
        "expert_pass_count": int(pass_count),
        "expert_pass_fraction": float(pass_fraction),
        "category_counts": category_counts,
        "expert_mismatch_families": sorted(mismatch_families),
        "aggregate_expert_direct_mean_distance": float(np.mean(direct_means)),
        "aggregate_expert_rigid_aligned_mean_distance": float(
            np.mean(rigid_means)
        ),
        "prior_audit_path_count": int(len(prior_audit_rows)),
        "prior_audit_pass_count": int(prior_pass_count),
        "prior_audit_coverage_complete": bool(prior_coverage_complete),
        "prior_audit_performed": prior_audit_performed,
        "all_prior_audits_pass": all_prior_audits_pass,
        "all_expert_rows_pass_direct": bool(all_expert_rows_pass_direct),
        "expert_dataset_failure_paths": expert_dataset_failure_paths,
        "global_prior_evaluated_path_count": int(global_prior_performed_count),
        "global_prior_pass_paths": global_prior_pass_paths,
        "global_prior_failure_paths": global_prior_failure_paths,
        "global_prior_unperformed_paths": global_prior_unperformed_paths,
        "aggregate_global_prior_direct_mean_distance": (
            float(np.mean(global_prior_direct_means))
            if global_prior_direct_means
            else None
        ),
        "diffusion_evaluated_path_count": int(diffusion_performed_count),
        "diffusion_evaluated_trajectory_count": int(diffusion_trajectory_count),
        "diffusion_pass_paths": diffusion_pass_paths,
        "diffusion_failure_paths": diffusion_failure_paths,
        "diffusion_unperformed_paths": diffusion_unperformed_paths,
        "scientifically_valid_to_interpret_diffusion_tracking_metrics": bool(
            scientifically_valid
        ),
    }


def print_fk_validation_startup(
    *,
    enabled: bool,
    diagnostic_only: bool,
    resolved_urdf_descriptor: str,
    active_joint_names: Sequence[str],
    fk_frame: str,
    tool_transform: Any,
    mean_threshold: float,
    max_threshold: float,
    selected_path_count: int,
) -> None:
    """Print explicit FK-validation configuration without guessing URDF provenance."""

    if not isinstance(resolved_urdf_descriptor, str) or not resolved_urdf_descriptor:
        raise ValueError("resolved_urdf_descriptor must be supplied explicitly")
    validated_tool = validate_homogeneous_transform(
        tool_transform,
        label="startup tool transform",
    )
    if selected_path_count < 0:
        raise ValueError("selected_path_count must be >= 0")
    mode = (
        "DISABLED"
        if not enabled
        else "DIAGNOSTIC_ONLY"
        if diagnostic_only
        else "BENCHMARK_WITH_FK_VALIDATION"
    )
    print(
        "[FK validation startup] "
        f"enabled={bool(enabled)} mode={mode} selected_paths={selected_path_count}"
    )
    print(f"[FK validation startup] URDF={resolved_urdf_descriptor}")
    print(
        "[FK validation startup] active_joints="
        + json.dumps([str(name) for name in active_joint_names])
        + f" FK_frame={fk_frame}"
    )
    print(
        "[FK validation startup] tool_matrix="
        + json.dumps(validated_tool.tolist())
    )
    print(
        "[FK validation startup] thresholds_m="
        f"mean<={float(mean_threshold):.9g}, max<={float(max_threshold):.9g}"
    )


def print_fk_validation_completion(
    records: Sequence[Mapping[str, Any]],
    expert_summary_rows: Sequence[Mapping[str, Any]],
    prior_audit_rows: Sequence[Mapping[str, Any]],
    interpretation: Optional[Mapping[str, Any]] = None,
) -> None:
    """Print compact per-path FK results and the conservative interpretation."""

    interpreted = (
        interpret_fk_validation(expert_summary_rows, prior_audit_rows, records)
        if interpretation is None
        else dict(interpretation)
    )
    summary_by_path: Dict[str, Mapping[str, Any]] = {}
    for row in expert_summary_rows:
        path_name = str(row["path_name"])
        if path_name in summary_by_path:
            raise ValueError(f"duplicate expert summary path: {path_name}")
        summary_by_path[path_name] = row
    for record in sorted(
        records,
        key=lambda item: (int(item["path_index"]), str(item["path_name"])),
    ):
        path_name = str(record["path_name"])
        if path_name not in summary_by_path:
            raise KeyError(f"missing expert summary row for {path_name}")
        summary = summary_by_path[path_name]
        global_mean = float(
            record["global_reference"]["diagnostics"]["direct"]["mean_distance"]
        )
        buffer_entry = record.get("buffer_only")
        buffer_text = (
            "MISSING"
            if buffer_entry is None
            else f"{float(buffer_entry['diagnostics']['direct']['mean_distance']):.6g}m"
        )
        diffusion_entries = record["base_tail_diffusion_by_seed"]
        seed_means = [
            float(entry["diagnostics"]["direct"]["mean_distance"])
            for entry in diffusion_entries.values()
        ]
        diffusion_text = (
            f"{float(np.mean(seed_means)):.6g}m"
            if seed_means
            else "NOT_PERFORMED"
        )
        print(
            "[FK validation path] "
            f"path={path_name} "
            f"expert_mean={float(summary['direct_mean_distance']):.6g}m "
            f"expert_max={float(summary['direct_max_distance']):.6g}m "
            f"expert_rigid_mean="
            f"{float(summary['rigid_aligned_mean_distance']):.6g}m "
            f"global_mean={global_mean:.6g}m buffer_mean={buffer_text} "
            f"diffusion_seed_reduced_mean={diffusion_text} "
            f"classification={summary['classification']}"
        )
    print(
        "[FK validation completion] category_counts="
        + json.dumps(interpreted["category_counts"], sort_keys=True)
    )
    print(
        "[FK validation completion] "
        f"expert_pass={interpreted['expert_pass_count']}/"
        f"{interpreted['expert_path_count']} "
        f"pass_fraction={float(interpreted['expert_pass_fraction']):.3f} "
        f"aggregate_direct_mean="
        f"{float(interpreted['aggregate_expert_direct_mean_distance']):.6g}m "
        f"aggregate_rigid_mean="
        f"{float(interpreted['aggregate_expert_rigid_aligned_mean_distance']):.6g}m"
    )
    print(
        "[FK validation completion] "
        f"prior_audits={interpreted['prior_audit_pass_count']}/"
        f"{interpreted['prior_audit_path_count']} "
        "scientifically_valid_to_interpret_diffusion_tracking_metrics="
        f"{bool(interpreted['scientifically_valid_to_interpret_diffusion_tracking_metrics'])}"
    )
    print(
        "[FK validation completion] decisions "
        f"coordinate_fk={interpreted['coordinate_fk_validity']} "
        f"expert_dataset={interpreted['expert_dataset_validity']} "
        f"prior_reconstruction={interpreted['prior_reconstruction_validity']} "
        f"prior_quality={interpreted['prior_trajectory_quality']} "
        f"diffusion_tracking={interpreted['diffusion_tracking_validity']}"
    )
    print(
        "[FK validation completion] "
        "conclusions="
        + json.dumps(interpreted["conclusions"])
        + " "
        f"explanation={interpreted['conclusion_explanation']}"
    )


def publish_fk_validation_outputs(
    args: argparse.Namespace,
    records: Sequence[Mapping[str, Any]],
    prior_audit_rows: Sequence[Mapping[str, Any]],
    resolved_urdf_descriptor: str,
) -> Dict[str, Any]:
    """Publish the standalone FK-validation artifact family only."""

    if not records:
        raise ValueError("FK validation publication requires at least one path record")
    if not isinstance(resolved_urdf_descriptor, str) or not resolved_urdf_descriptor:
        raise ValueError("resolved_urdf_descriptor must be supplied explicitly")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    per_path_rows = fk_validation_per_path_rows(records)
    expert_summary_rows = expert_fk_validation_summary_rows(records)
    aggregate_rows = fk_validation_aggregate_rows(per_path_rows)
    prior_rows = [dict(row) for row in prior_audit_rows]
    frame_audit_rows = coordinate_frame_audit_rows(records)
    required_rows = {
        "fk_validation_per_path.csv": per_path_rows,
        "expert_fk_validation_summary.csv": expert_summary_rows,
        "fk_validation_aggregate.csv": aggregate_rows,
        "prior_reconstruction_audit.csv": prior_rows,
        "coordinate_frame_audit.csv": frame_audit_rows,
    }
    for filename, rows in required_rows.items():
        if not rows:
            raise ValueError(
                f"required FK validation output {filename} has no rows"
            )

    top_level_csv_paths: List[Path] = []
    for filename, rows in required_rows.items():
        path = output_dir / filename
        write_records_csv(path, rows)
        top_level_csv_paths.append(path)

    pointwise_rows: List[Dict[str, Any]] = []
    pointwise_path: Optional[Path] = None
    if bool(args.save_fk_pointwise_csv):
        pointwise_rows = fk_pointwise_error_rows(records)
        if not pointwise_rows:
            raise ValueError("requested fk_pointwise_errors.csv has no rows")
        pointwise_path = output_dir / "fk_pointwise_errors.csv"
        write_records_csv(pointwise_path, pointwise_rows)
        top_level_csv_paths.append(pointwise_path)

    trajectory_paths = write_fk_validation_path_trajectories(output_dir, records)
    plot_paths = publish_all_fk_validation_plots(
        output_dir,
        records,
        bool(args.plot_equal_axes),
    )
    expected_plot_count = 9 * len(records)
    if len(plot_paths) != expected_plot_count:
        raise RuntimeError(
            "FK validation plot publication returned the wrong total count: "
            f"expected {expected_plot_count}, got {len(plot_paths)}"
        )
    expected_plot_parents = {
        output_dir
        / "fk_validation_paths"
        / rollout.safe_path_name(str(record["path_name"]))
        for record in records
    }
    plot_counts_by_parent = Counter(path.parent for path in plot_paths)
    if set(plot_counts_by_parent) != expected_plot_parents or any(
        count != 9 for count in plot_counts_by_parent.values()
    ):
        raise RuntimeError(
            "FK validation publication must contain exactly nine plots under "
            "each selected safe per-path directory"
        )

    interpretation = interpret_fk_validation(
        expert_summary_rows,
        prior_rows,
        records,
    )
    print_fk_validation_completion(
        records,
        expert_summary_rows,
        prior_rows,
        interpretation,
    )

    deterministic_paths = sorted(
        top_level_csv_paths + trajectory_paths + plot_paths,
        key=lambda path: str(path),
    )
    if len(deterministic_paths) != len(set(deterministic_paths)):
        raise RuntimeError("FK validation publication returned duplicate output paths")
    return {
        "resolved_urdf_descriptor": resolved_urdf_descriptor,
        "output_dir": output_dir,
        "fk_validation_per_path_rows": per_path_rows,
        "expert_fk_validation_summary_rows": expert_summary_rows,
        "fk_validation_aggregate_rows": aggregate_rows,
        "prior_reconstruction_audit_rows": prior_rows,
        "coordinate_frame_audit_rows": frame_audit_rows,
        "fk_pointwise_error_rows": pointwise_rows,
        "interpretation": interpretation,
        "top_level_csv_paths": list(top_level_csv_paths),
        "per_path_trajectory_paths": list(trajectory_paths),
        "plot_paths": list(plot_paths),
        "all_output_paths": deterministic_paths,
        "counts": {
            "path_records": int(len(records)),
            "fk_validation_per_path_rows": int(len(per_path_rows)),
            "expert_fk_validation_summary_rows": int(len(expert_summary_rows)),
            "fk_validation_aggregate_rows": int(len(aggregate_rows)),
            "prior_reconstruction_audit_rows": int(len(prior_rows)),
            "coordinate_frame_audit_rows": int(len(frame_audit_rows)),
            "fk_pointwise_error_rows": int(len(pointwise_rows)),
            "top_level_csv_files": int(len(top_level_csv_paths)),
            "per_path_trajectory_files": int(len(trajectory_paths)),
            "plots": int(len(plot_paths)),
            "all_output_files": int(len(deterministic_paths)),
        },
        "pointwise_csv_enabled": bool(args.save_fk_pointwise_csv),
        "pointwise_csv_path": pointwise_path,
    }


def resolve_loaded_urdf_descriptor(robot: Any) -> str:
    """Resolve explicit loaded URDF provenance without guessing or filesystem search."""

    candidate_sources: List[Tuple[str, Path]] = []

    def inspect_attributes(owner_label: str, attributes: Mapping[str, Any]) -> None:
        for attribute_name, value in attributes.items():
            if "urdf" not in str(attribute_name).lower():
                continue
            if not isinstance(value, (str, Path)):
                continue
            candidate = Path(value).expanduser()
            if candidate.suffix.lower() != ".urdf" or not candidate.is_file():
                continue
            candidate_sources.append(
                (f"{owner_label}.{attribute_name}", candidate.resolve())
            )

    robot_attributes = getattr(robot, "__dict__", None)
    if isinstance(robot_attributes, Mapping):
        inspect_attributes("robot", robot_attributes)
    for module_name in ("rollout", "diagnostic", "action_buffer_diagnostic"):
        module = globals().get(module_name)
        if module is None:
            continue
        module_attributes = getattr(module, "__dict__", None)
        if isinstance(module_attributes, Mapping):
            inspect_attributes(module_name, module_attributes)

    unique_paths = sorted(
        {path for _, path in candidate_sources},
        key=lambda path: str(path),
    )
    if len(unique_paths) > 1:
        provenance = ", ".join(
            f"{source}={path}" for source, path in candidate_sources
        )
        raise ValueError(
            "Canonical FK loader exposes multiple distinct existing URDF paths; "
            f"refusing to guess: {provenance}"
        )
    if len(unique_paths) == 1:
        return str(unique_paths[0])
    return (
        "WARNING:CANONICAL_LOADER_URDF_PATH_UNAVAILABLE; "
        "robot=rollout.load_fk_context(None,None); no explicit URDF-named "
        "robot/module attribute contained an existing .urdf path; no filesystem "
        "search or path guess was performed"
    )


def load_fk_diagnostic_assets(args: argparse.Namespace) -> Dict[str, Any]:
    """Load only test/statistics/window/FK assets needed by diagnostic-only mode."""

    normalization_metadata = rollout.load_npz(
        args.normalization_stats,
        "normalization metadata",
    )
    stats = rollout.load_stats(args.normalization_stats)
    residual_mean = np.asarray(stats["residual_mean"], dtype=np.float32)
    residual_std = np.asarray(stats["residual_std"], dtype=np.float32)
    condition_mean = np.asarray(stats["condition_mean"], dtype=np.float32)
    condition_std = np.asarray(stats["condition_std"], dtype=np.float32)
    if residual_mean.shape != (rollout.JOINT_DIM,) or residual_std.shape != (
        rollout.JOINT_DIM,
    ):
        raise ValueError("Residual normalization statistics must have shape (6,)")
    if condition_mean.shape != (
        action_buffer_diagnostic.CONDITION_DIM,
    ) or condition_std.shape != (action_buffer_diagnostic.CONDITION_DIM,):
        raise ValueError(
            "Condition normalization statistics must have shape "
            f"({action_buffer_diagnostic.CONDITION_DIM},)"
        )
    if np.any(residual_std <= 0.0) or np.any(condition_std <= 0.0):
        raise ValueError("Normalization standard deviations must be positive")

    data = rollout.load_npz(args.test_npz, "held-out test trajectories")
    rollout.require_keys(
        data,
        ("desired_paths", "expert_q", "q_start", "path_names"),
        "held-out test trajectories",
    )
    desired_paths = rollout.finite_array(
        data["desired_paths"], "desired_paths"
    ).astype(np.float32)
    expert_q = rollout.finite_array(data["expert_q"], "expert_q").astype(
        np.float32
    )
    q_start = rollout.finite_array(data["q_start"], "q_start").astype(
        np.float32
    )
    path_names = rollout.decode_names(data["path_names"])
    if desired_paths.ndim != 3 or desired_paths.shape[2] != 3:
        raise ValueError("desired_paths must have shape (N,T,3)")
    if expert_q.shape != (
        desired_paths.shape[0],
        desired_paths.shape[1],
        rollout.JOINT_DIM,
    ):
        raise ValueError("expert_q must have shape (N,T,6)")
    if q_start.shape != (desired_paths.shape[0], rollout.JOINT_DIM):
        raise ValueError("q_start must have shape (N,6)")
    if len(path_names) != desired_paths.shape[0] or len(set(path_names)) != len(
        path_names
    ):
        raise ValueError("Test path identifiers must be unique and match trajectory count")

    window_artifact = validate_window_artifact(
        args.window_npz,
        stats,
        path_names,
        desired_paths,
        expert_q,
        args.prediction_horizon,
        normalization_metadata=normalization_metadata,
    )
    robot, joint_names, ee_link = rollout.load_fk_context(None, None)
    if rollout.JOINT_DIM != 6 or len(joint_names) != 6:
        raise ValueError("Exactly six active xMateCR7 joints are required")
    if len(set(joint_names)) != 6:
        raise ValueError("Active xMateCR7 joint names must be unique")
    lower, upper = rollout.extract_joint_limits(robot, joint_names)
    return {
        "stats": stats,
        "residual_mean": residual_mean,
        "residual_std": residual_std,
        "condition_mean": condition_mean,
        "condition_std": condition_std,
        "desired_paths": desired_paths,
        "expert_q": expert_q,
        "q_start": q_start,
        "path_names": path_names,
        "window_artifact": window_artifact,
        "robot": robot,
        "joint_names": joint_names,
        "ee_link": ee_link,
        "lower": lower,
        "upper": upper,
        "drawing_weights": rollout.default_weights(),
        "resolved_urdf_descriptor": resolve_loaded_urdf_descriptor(robot),
    }


def run_fk_diagnostic_only(args: argparse.Namespace) -> int:
    """Run lightweight expert/global FK diagnostics without benchmark assets."""

    if not args.diagnostic_only or not args.run_fk_validation:
        raise ValueError(
            "run_fk_diagnostic_only requires diagnostic-only FK validation mode"
        )
    assets = load_fk_diagnostic_assets(args)
    available_path_count = len(assets["path_names"])
    (
        selected_names,
        limit_option,
        requested_limit,
    ) = resolve_fk_validation_path_selection(
        args,
        assets["path_names"],
    )
    selected_count = len(selected_names)
    if not selected_names:
        raise ValueError("No held-out paths were selected for FK diagnostic-only mode")
    print_fk_validation_path_selection(
        selected_count=selected_count,
        available_count=available_path_count,
        determining_option=limit_option,
        requested_limit=requested_limit,
    )

    trajectory_length = int(assets["desired_paths"].shape[1])
    # Reconstruction provenance is validated against every held-out path before
    # the independently limited diagnostic reporting cohort is selected.
    global_references = {
        path_name: load_global_reference(args, path_name, trajectory_length)
        for path_name in assets["path_names"]
    }
    validate_prior_window_reconstruction(
        assets["window_artifact"],
        global_references,
        args.prediction_horizon,
    )
    prior_detail_rows = audit_prior_window_path_alignment(
        assets["window_artifact"],
        assets["path_names"],
        assets["desired_paths"],
        assets["expert_q"],
        global_references,
        tolerance=PRIOR_RECONSTRUCTION_AUDIT_ATOL,
    )
    prior_audit_rows = aggregate_prior_reconstruction_audit_rows(
        prior_detail_rows,
        assets["window_artifact"],
        selected_names,
        trajectory_length,
        args.prediction_horizon,
    )

    tool_transform = resolve_tool_transform(args)
    resolved_urdf_descriptor = str(assets["resolved_urdf_descriptor"])
    print_fk_validation_startup(
        enabled=True,
        diagnostic_only=True,
        resolved_urdf_descriptor=resolved_urdf_descriptor,
        active_joint_names=assets["joint_names"],
        fk_frame=assets["ee_link"],
        tool_transform=tool_transform,
        mean_threshold=args.expert_mean_error_threshold,
        max_threshold=args.expert_max_error_threshold,
        selected_path_count=selected_count,
    )

    records: List[Dict[str, Any]] = []
    with torch.no_grad():
        for path_index, path_name in enumerate(selected_names):
            records.append(
                build_fk_validation_path_record(
                    path_name=path_name,
                    path_index=path_index,
                    desired=assets["desired_paths"][path_index],
                    expert_q=assets["expert_q"][path_index],
                    global_q=global_references[path_name],
                    diffusion_by_seed={},
                    robot=assets["robot"],
                    joint_names=assets["joint_names"],
                    ee_link=assets["ee_link"],
                    mean_threshold=args.expert_mean_error_threshold,
                    max_threshold=args.expert_max_error_threshold,
                    buffer_q=None,
                    tool_transform=tool_transform,
                )
            )

    publication = publish_fk_validation_outputs(
        args,
        records,
        prior_audit_rows,
        resolved_urdf_descriptor,
    )
    mismatched_rows = [
        row
        for row in publication["expert_fk_validation_summary_rows"]
        if str(row["classification"]) != "PASS_DIRECT"
    ]
    # The requested fail gate is deliberately after complete diagnostic
    # publication so mismatch evidence is never lost.
    if args.fail_on_expert_fk_mismatch and mismatched_rows:
        mismatched_paths = [str(row["path_name"]) for row in mismatched_rows]
        raise RuntimeError(
            "Expert FK mismatch after complete diagnostic publication for paths: "
            + ", ".join(mismatched_paths)
        )
    return 0


_BASE_TAIL_BENCHMARK_RUN_ONE_ROLLOUT = run_one_rollout
_BASE_TAIL_BENCHMARK_RUN_ONE_ROLLOUT_SIGNATURE = inspect.signature(
    _BASE_TAIL_BENCHMARK_RUN_ONE_ROLLOUT
)
_FULL_FK_ROLLOUT_CAPTURE_ENABLED = False
_FULL_FK_ROLLOUT_CAPTURE_ARGS: Optional[argparse.Namespace] = None
_FULL_FK_ROLLOUT_CAPTURE: Dict[
    Tuple[int, str, str, Optional[int]], np.ndarray
] = {}


def configure_full_fk_rollout_capture(enabled: bool) -> None:
    """Reset rollout capture and explicitly enable or disable first-result capture."""

    global _FULL_FK_ROLLOUT_CAPTURE_ENABLED
    global _FULL_FK_ROLLOUT_CAPTURE_ARGS
    _FULL_FK_ROLLOUT_CAPTURE.clear()
    _FULL_FK_ROLLOUT_CAPTURE_ARGS = None
    _FULL_FK_ROLLOUT_CAPTURE_ENABLED = bool(enabled)


def run_one_rollout(*call_args: Any, **call_kwargs: Any) -> Any:
    """Transparent call-through with optional post-success FK trajectory capture."""

    global _FULL_FK_ROLLOUT_CAPTURE_ARGS
    if not _FULL_FK_ROLLOUT_CAPTURE_ENABLED:
        return _BASE_TAIL_BENCHMARK_RUN_ONE_ROLLOUT(*call_args, **call_kwargs)

    bound = _BASE_TAIL_BENCHMARK_RUN_ONE_ROLLOUT_SIGNATURE.bind(
        *call_args, **call_kwargs
    )
    bound.apply_defaults()
    arguments = bound.arguments
    benchmark_args = arguments.get("args")
    path_index_value = arguments.get("path_index")
    path_name_value = arguments.get("path_name")
    method_value = arguments.get("method", arguments.get("rollout_method"))
    seed_value = arguments.get("seed", arguments.get("global_seed"))

    # Preserve original execution, exception, and return behavior before capture
    # performs any result inspection or copying.
    result = _BASE_TAIL_BENCHMARK_RUN_ONE_ROLLOUT(*call_args, **call_kwargs)
    rollout_method = str(method_value) if method_value is not None else ""
    if rollout_method not in {"buffer_only", "base_tail"}:
        return result
    method = (
        "base_tail_diffusion"
        if rollout_method == "base_tail"
        else rollout_method
    )
    if benchmark_args is None:
        raise KeyError("captured run_one_rollout call did not bind an 'args' argument")
    if path_index_value is None or path_name_value is None:
        raise KeyError(
            "captured run_one_rollout call lacks path_index or path_name"
        )
    if method == "base_tail_diffusion" and seed_value is None:
        raise KeyError("base_tail_diffusion capture requires a bound seed")
    if _FULL_FK_ROLLOUT_CAPTURE_ARGS is None:
        _FULL_FK_ROLLOUT_CAPTURE_ARGS = benchmark_args
    elif _FULL_FK_ROLLOUT_CAPTURE_ARGS is not benchmark_args:
        raise RuntimeError("rollout capture observed multiple benchmark args objects")
    if not isinstance(result, Mapping) or "q" not in result:
        raise KeyError("captured run_one_rollout result must contain result['q']")

    path_index = int(path_index_value)
    path_name = str(path_name_value)
    seed = None if seed_value is None else int(seed_value)
    capture_key = (path_index, path_name, method, seed)
    if capture_key not in _FULL_FK_ROLLOUT_CAPTURE:
        _FULL_FK_ROLLOUT_CAPTURE[capture_key] = np.asarray(result["q"]).copy()
    return result


run_one_rollout.__signature__ = _BASE_TAIL_BENCHMARK_RUN_ONE_ROLLOUT_SIGNATURE
run_one_rollout.__wrapped__ = _BASE_TAIL_BENCHMARK_RUN_ONE_ROLLOUT


def build_full_fk_validation_records_from_capture(
    args: argparse.Namespace,
    assets: Mapping[str, Any],
    selected_validation_names: Sequence[str],
    global_references: Mapping[str, np.ndarray],
    tool_transform: Any,
) -> List[Dict[str, Any]]:
    """Build validation records from first-occurrence full benchmark trajectories."""

    if not selected_validation_names:
        raise ValueError("full FK validation requires at least one selected path")
    if len(set(selected_validation_names)) != len(selected_validation_names):
        raise ValueError("selected FK validation path names must be unique")
    if _FULL_FK_ROLLOUT_CAPTURE_ARGS is not args:
        raise RuntimeError(
            "full FK rollout capture was not configured for this benchmark args object"
        )
    all_path_names = [str(name) for name in assets["path_names"]]
    if len(set(all_path_names)) != len(all_path_names):
        raise ValueError("benchmark asset path names must be unique")
    path_index_by_name = {
        path_name: path_index
        for path_index, path_name in enumerate(all_path_names)
    }
    missing_selected = [
        path_name
        for path_name in selected_validation_names
        if path_name not in path_index_by_name
    ]
    if missing_selected:
        raise KeyError(
            "selected FK validation paths are absent from benchmark assets: "
            + ", ".join(missing_selected)
        )
    configured_seeds = tuple(int(seed) for seed in args.seeds)
    if len(set(configured_seeds)) != len(configured_seeds):
        raise ValueError("configured benchmark seeds must be unique")
    resolved_tool = validate_homogeneous_transform(
        tool_transform,
        label="full benchmark FK validation tool transform",
    )

    records: List[Dict[str, Any]] = []
    with torch.no_grad():
        for path_name_value in selected_validation_names:
            path_name = str(path_name_value)
            path_index = path_index_by_name[path_name]
            captured_for_path = [
                (key, q_trajectory)
                for key, q_trajectory in _FULL_FK_ROLLOUT_CAPTURE.items()
                if key[0] == path_index and key[1] == path_name
            ]
            if captured_for_path:
                buffer_trajectories = [
                    q_trajectory
                    for key, q_trajectory in captured_for_path
                    if key[2] == "buffer_only"
                ]
                if len(buffer_trajectories) != 1:
                    raise RuntimeError(
                        f"expected exactly one captured buffer_only trajectory for "
                        f"{path_name}, got {len(buffer_trajectories)}"
                    )
                buffer_trajectory: Optional[np.ndarray] = buffer_trajectories[0]
                diffusion_by_seed = {
                    int(key[3]): q_trajectory
                    for key, q_trajectory in captured_for_path
                    if key[2] == "base_tail_diffusion" and key[3] is not None
                }
                captured_seed_set = set(diffusion_by_seed)
                configured_seed_set = set(configured_seeds)
                if captured_seed_set != configured_seed_set:
                    raise RuntimeError(
                        f"captured diffusion seeds for {path_name} do not match the "
                        f"configured seeds: captured={sorted(captured_seed_set)}, "
                        f"configured={sorted(configured_seed_set)}"
                    )
            else:
                # An explicit --validation_max_paths may select paths outside the
                # --max_paths benchmark cohort. Expert/global FK remains fully
                # auditable there; practical rollout methods are genuinely
                # unperformed and must not be fabricated or treated as failures.
                buffer_trajectory = None
                diffusion_by_seed = {}
            if path_name not in global_references:
                raise KeyError(f"missing global reference for {path_name}")
            records.append(
                build_fk_validation_path_record(
                    path_name=path_name,
                    path_index=path_index,
                    desired=assets["desired_paths"][path_index],
                    expert_q=assets["expert_q"][path_index],
                    global_q=global_references[path_name],
                    diffusion_by_seed=diffusion_by_seed,
                    robot=assets["robot"],
                    joint_names=assets["joint_names"],
                    ee_link=assets["ee_link"],
                    mean_threshold=args.expert_mean_error_threshold,
                    max_threshold=args.expert_max_error_threshold,
                    buffer_q=buffer_trajectory,
                    tool_transform=resolved_tool,
                )
            )
    return records


if __name__ == "__main__":
    raise SystemExit(main())
