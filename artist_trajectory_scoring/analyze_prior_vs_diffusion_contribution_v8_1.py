#!/usr/bin/env python3
"""Frozen diffusion v8.1 prior-ablation analysis.

This script evaluates the marginal effect of frozen diffusion v8.1 relative
to the exact strong MLP + adaptive-IK prior used to initialize each accepted
deployment trajectory.

This v8.1 prior-ablation analysis does not compare v8.1 against MLP-only
generation, sequential IK, deterministic residual models, v7 or other
diffusion versions, numerical cost-optimized targets, rejected deployment
attempts, runtime, or method-level acceptance rate. It measures marginal
contribution only; it does not independently certify deployment safety or
weaken any deployment validation threshold.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

import evaluate_diffusion_v7_teacher_forced_validation as v7_evaluator
import generate_diffusion_v7_cost_improving_residual_targets as target_generator
import validate_diffusion_v8_1_deployment_output as deployment_validator
from generate_ik_seed_path import (
    DEFAULT_EE_LINK,
    DEFAULT_JOINT_NAMES,
    HARD_JOINT_LIMIT_TOLERANCE_RAD,
)
from orientation_aware_adaptive_ik import (
    orientation_error_trajectory,
    trajectory_full_transform_fk,
)


ACCEPTED_VERDICT = "V8_1_DEPLOYMENT_TRAJECTORY_ACCEPTED"
ANALYSIS_TYPE = "v8_1_prior_ablation"
DERIVATIVE_INTERIOR_TRIM_SAMPLES = 3
REQUIRED_CASE_NAMES = {
    "stroke_01",
    "stroke_02",
    "stroke_03_branch_continuous",
}
JOINT_DIM = 6
CHANGE_ABSOLUTE_TOLERANCE = 1.0e-12
CHANGE_RELATIVE_TOLERANCE = 1.0e-6
TIMESTAMP_ABSOLUTE_TOLERANCE_S = 1.0e-9
TIMESTAMP_RELATIVE_TOLERANCE = 1.0e-7
SVD_ABSOLUTE_TOLERANCE = 1.0e-12
JACOBIAN_FINITE_DIFFERENCE_STEP_RAD = 1.0e-5

OUTPUT_NAMES = (
    "per_stroke_summary.csv",
    "per_joint_summary.csv",
    "per_sample_metrics.csv",
    "aggregate_summary.csv",
    "primary_metric_group_summary.csv",
    "contribution_report.json",
    "contribution_report.md",
    "joint_difference_over_time.png",
    "cartesian_error_comparison.png",
    "acceleration_comparison.png",
    "jerk_comparison.png",
    "interior_acceleration_comparison.png",
    "interior_jerk_comparison.png",
    "singularity_margin_comparison.png",
    "joint_limit_margin_comparison.png",
)

REQUIRED_DEPLOYMENT_ARTIFACTS = (
    "deployment_input_copy.npz",
    "deployment_trajectory_full.npz",
    "deployment_trajectory.csv",
    "deployment_joint_positions.csv",
    "deployment_joint_dynamics.csv",
    "deployment_cartesian_tracking.csv",
    "deployment_segment_decisions.csv",
    "deployment_candidate_results.csv",
    "deployment_metrics.json",
    "deployment_report.txt",
    "approved_simulation_trajectory.csv",
    "approved_simulation_trajectory.npz",
)


class AnalysisError(RuntimeError):
    """The requested contribution analysis cannot be performed faithfully."""


@dataclass(frozen=True)
class MetricSpec:
    key: str
    column_stem: str
    unit_suffix: str
    direction: str
    pooled: bool = True


METRIC_SPECS: Tuple[MetricSpec, ...] = (
    MetricSpec("cartesian_mean_error_m", "cartesian_mean_error", "m", "lower"),
    MetricSpec("cartesian_rms_error_m", "cartesian_rms_error", "m", "lower"),
    MetricSpec("cartesian_max_error_m", "cartesian_max_error", "m", "lower"),
    MetricSpec(
        "cartesian_final_error_m",
        "cartesian_final_error",
        "m",
        "lower",
        pooled=False,
    ),
    MetricSpec(
        "orientation_mean_error_rad",
        "orientation_mean_error",
        "rad",
        "lower",
    ),
    MetricSpec(
        "orientation_rms_error_rad",
        "orientation_rms_error",
        "rad",
        "lower",
    ),
    MetricSpec(
        "orientation_max_error_rad",
        "orientation_max_error",
        "rad",
        "lower",
    ),
    MetricSpec(
        "orientation_final_error_rad",
        "orientation_final_error",
        "rad",
        "lower",
        pooled=False,
    ),
    MetricSpec(
        "velocity_max_abs_rad_s",
        "velocity_max_abs",
        "rad_s",
        "lower",
    ),
    MetricSpec("velocity_rms_rad_s", "velocity_rms", "rad_s", "lower"),
    MetricSpec(
        "acceleration_max_abs_rad_s2",
        "acceleration_max_abs",
        "rad_s2",
        "lower",
    ),
    MetricSpec(
        "acceleration_rms_rad_s2",
        "acceleration_rms",
        "rad_s2",
        "lower",
    ),
    MetricSpec(
        "integrated_squared_acceleration_rad2_s3",
        "integrated_squared_acceleration",
        "rad2_s3",
        "lower",
    ),
    MetricSpec(
        "interior_acceleration_max_abs_rad_s2",
        "interior_acceleration_max_abs",
        "rad_s2",
        "lower",
    ),
    MetricSpec(
        "interior_acceleration_rms_rad_s2",
        "interior_acceleration_rms",
        "rad_s2",
        "lower",
    ),
    MetricSpec(
        "interior_integrated_squared_acceleration_rad2_s3",
        "interior_integrated_squared_acceleration",
        "rad2_s3",
        "lower",
    ),
    MetricSpec("jerk_max_abs_rad_s3", "jerk_max_abs", "rad_s3", "lower"),
    MetricSpec("jerk_rms_rad_s3", "jerk_rms", "rad_s3", "lower"),
    MetricSpec(
        "integrated_squared_jerk_rad2_s5",
        "integrated_squared_jerk",
        "rad2_s5",
        "lower",
    ),
    MetricSpec(
        "interior_jerk_max_abs_rad_s3",
        "interior_jerk_max_abs",
        "rad_s3",
        "lower",
    ),
    MetricSpec(
        "interior_jerk_rms_rad_s3",
        "interior_jerk_rms",
        "rad_s3",
        "lower",
    ),
    MetricSpec(
        "interior_integrated_squared_jerk_rad2_s5",
        "interior_integrated_squared_jerk",
        "rad2_s5",
        "lower",
    ),
    MetricSpec(
        "minimum_joint_limit_margin_rad",
        "minimum_joint_limit_margin",
        "rad",
        "higher",
    ),
    MetricSpec(
        "minimum_normalized_joint_limit_margin",
        "minimum_normalized_joint_limit_margin",
        "",
        "higher",
    ),
    MetricSpec(
        "joint_limit_violation_count",
        "joint_limit_violation_count",
        "",
        "lower",
    ),
    MetricSpec(
        "translational_jacobian_min_singular_value_m_per_rad",
        "translational_jacobian_min_singular_value",
        "m_per_rad",
        "higher",
    ),
    MetricSpec(
        "translational_jacobian_max_condition_number",
        "translational_jacobian_max_condition_number",
        "",
        "lower",
    ),
    MetricSpec(
        "translational_jacobian_min_manipulability_m3_per_rad3",
        "translational_jacobian_min_manipulability",
        "m3_per_rad3",
        "higher",
    ),
)

SPEC_BY_KEY = {spec.key: spec for spec in METRIC_SPECS}

PRIMARY_METRIC_GROUPS: Dict[str, Dict[str, Any]] = {
    "cartesian_position_accuracy": {
        "primary_metric": "cartesian_rms_error_m",
        "supporting_metrics": [
            "cartesian_mean_error_m",
            "cartesian_max_error_m",
            "cartesian_final_error_m",
        ],
    },
    "orientation_accuracy": {
        "primary_metric": "orientation_rms_error_rad",
        "supporting_metrics": [
            "orientation_mean_error_rad",
            "orientation_max_error_rad",
            "orientation_final_error_rad",
        ],
    },
    "velocity_magnitude": {
        "primary_metric": "velocity_rms_rad_s",
        "supporting_metrics": ["velocity_max_abs_rad_s"],
    },
    "acceleration_smoothness": {
        "primary_metric": (
            "interior_integrated_squared_acceleration_rad2_s3"
        ),
        "supporting_metrics": [
            "interior_acceleration_rms_rad_s2",
            "interior_acceleration_max_abs_rad_s2",
            "integrated_squared_acceleration_rad2_s3",
            "acceleration_rms_rad_s2",
            "acceleration_max_abs_rad_s2",
        ],
    },
    "jerk_smoothness": {
        "primary_metric": "interior_integrated_squared_jerk_rad2_s5",
        "supporting_metrics": [
            "interior_jerk_rms_rad_s3",
            "interior_jerk_max_abs_rad_s3",
            "integrated_squared_jerk_rad2_s5",
            "jerk_rms_rad_s3",
            "jerk_max_abs_rad_s3",
        ],
    },
    "joint_limit_margin": {
        "primary_metric": "minimum_normalized_joint_limit_margin",
        "supporting_metrics": [
            "minimum_joint_limit_margin_rad",
            "joint_limit_violation_count",
        ],
    },
    "translational_singularity_margin": {
        "primary_metric": (
            "translational_jacobian_min_singular_value_m_per_rad"
        ),
        "supporting_metrics": [
            "translational_jacobian_max_condition_number",
            "translational_jacobian_min_manipulability_m3_per_rad3",
        ],
    },
}

SCOPE_EXCLUSIONS = (
    "MLP-only generation",
    "sequential IK",
    "deterministic residual models",
    "v7 and other diffusion versions",
    "cost-optimized targets",
    "rejected deployment attempts",
    "runtime comparison",
    "method-level acceptance-rate comparison",
)


@dataclass
class LoadedCase:
    name: str
    input_path: Path
    output_dir: Path
    full_path: Path
    metrics_path: Path
    approved_path: Path
    prior_q: np.ndarray
    diffusion_q: np.ndarray
    desired_position_m: np.ndarray
    target_rotation_matrix: np.ndarray
    timestamps_s: np.ndarray
    timestep_s: float
    joint_order: Tuple[str, ...]
    joint_order_sources: Dict[str, str]
    robot: target_generator.RobotContext
    lower_limits_rad: np.ndarray
    upper_limits_rad: np.ndarray
    metadata: Dict[str, Any]
    validation: Dict[str, Any]


@dataclass
class TrajectoryMetrics:
    trajectory_type: str
    q_rad: np.ndarray
    fk_position_m: np.ndarray
    position_error_m: np.ndarray
    orientation_error_rad: np.ndarray
    velocity_rad_s: np.ndarray
    acceleration_rad_s2: np.ndarray
    jerk_rad_s3: np.ndarray
    joint_limit_margin_rad: np.ndarray
    normalized_joint_limit_margin: np.ndarray
    joint_limit_violation: np.ndarray
    jacobian_min_singular_value: np.ndarray
    jacobian_condition_number: np.ndarray
    jacobian_manipulability: np.ndarray
    scalar: Dict[str, float]
    indices: Dict[str, Any]
    per_joint: Dict[str, np.ndarray]


@dataclass
class CaseAnalysis:
    loaded: LoadedCase
    prior: TrajectoryMetrics
    diffusion: TrajectoryMetrics
    difference_q_rad: np.ndarray
    difference_velocity_rad_s: np.ndarray
    difference_acceleration_rad_s2: np.ndarray
    difference_jerk_rad_s3: np.ndarray
    difference: Dict[str, Any]
    comparison: Dict[str, Dict[str, Any]]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Frozen diffusion v8.1 prior-ablation analysis: evaluate the "
            "marginal effect of v8.1 relative to each exact strong prior."
        )
    )
    parser.add_argument(
        "--case",
        dest="cases",
        action="append",
        nargs=3,
        metavar=("CASE_NAME", "DEPLOYMENT_INPUT_NPZ", "DIFFUSION_OUTPUT_DIR"),
        required=True,
        help="Repeat once for each accepted stroke.",
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def strict_json(path: Path) -> Dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise AnalysisError(f"{path}: non-strict JSON constant {value}")

    try:
        value = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    except (OSError, json.JSONDecodeError) as exc:
        raise AnalysisError(f"Cannot read strict JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise AnalysisError(f"{path} must contain a JSON object")
    return value


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            return {key: np.array(archive[key], copy=True) for key in archive.files}
    except (OSError, ValueError) as exc:
        raise AnalysisError(f"Cannot read NPZ {path}: {exc}") from exc


def scalar_text(value: Any, label: str) -> str:
    array = np.asarray(value)
    if array.size != 1:
        raise AnalysisError(f"{label} must contain exactly one scalar")
    item = array.reshape(-1)[0]
    if isinstance(item, bytes):
        return item.decode("utf-8")
    return str(item)


def scalar_bool(value: Any, label: str) -> bool:
    array = np.asarray(value)
    if array.size != 1:
        raise AnalysisError(f"{label} must contain exactly one boolean")
    item = array.reshape(-1)[0]
    if not isinstance(item, (bool, np.bool_)):
        raise AnalysisError(f"{label} must be stored as a boolean")
    return bool(item)


def finite_array(
    value: Any,
    label: str,
    *,
    shape: Tuple[int, ...] | None = None,
    ndim: int | None = None,
) -> np.ndarray:
    raw = np.asarray(value)
    if not np.issubdtype(raw.dtype, np.number):
        raise AnalysisError(f"{label} must be numeric")
    array = np.asarray(raw, dtype=np.float64)
    if shape is not None and array.shape != shape:
        raise AnalysisError(f"{label} must have shape {shape}, got {array.shape}")
    if ndim is not None and array.ndim != ndim:
        raise AnalysisError(f"{label} must have {ndim} dimensions, got {array.ndim}")
    if not np.all(np.isfinite(array)):
        raise AnalysisError(f"{label} contains non-finite values")
    return array


def require_keys(data: Mapping[str, Any], keys: Sequence[str], label: str) -> None:
    missing = [key for key in keys if key not in data]
    if missing:
        raise AnalysisError(f"{label} lacks required keys: {missing}")


def string_tuple(value: Any, label: str) -> Tuple[str, ...]:
    array = np.asarray(value)
    if array.ndim != 1:
        raise AnalysisError(f"{label} must be one-dimensional")
    values = tuple(
        item.decode("utf-8") if isinstance(item, bytes) else str(item)
        for item in array.tolist()
    )
    if len(values) != JOINT_DIM or len(set(values)) != JOINT_DIM:
        raise AnalysisError(f"{label} must contain six unique joint names")
    return values


def resolve_joint_order(
    source: Mapping[str, Any],
    label: str,
) -> Tuple[Tuple[str, ...], str]:
    for key in ("joint_order", "joint_ordering", "joint_names"):
        if key in source:
            return string_tuple(source[key], f"{label}:{key}"), key
    # The v8.1 deployment format defines q1..q6 in this fixed repository order.
    return tuple(str(name) for name in DEFAULT_JOINT_NAMES), (
        "fixed_v8_1_q1_to_q6_repository_order"
    )


def verify_timestamps(
    timestamps: np.ndarray,
    label: str,
) -> float:
    if timestamps.ndim != 1 or len(timestamps) < 4:
        raise AnalysisError(f"{label} must be a one-dimensional array with at least 4 samples")
    differences = np.diff(timestamps)
    if not np.all(np.isfinite(differences)) or np.any(differences <= 0.0):
        raise AnalysisError(f"{label} must be finite and strictly increasing")
    timestep = float(differences[0])
    if not np.allclose(
        differences,
        timestep,
        rtol=TIMESTAMP_RELATIVE_TOLERANCE,
        atol=TIMESTAMP_ABSOLUTE_TOLERANCE_S,
    ):
        maximum_deviation = float(np.max(np.abs(differences - timestep)))
        raise AnalysisError(
            f"{label} does not define one verified sample period; "
            f"maximum interval deviation is {maximum_deviation:.12g} s"
        )
    return timestep


def require_exact_array(actual: np.ndarray, expected: np.ndarray, label: str) -> None:
    if actual.shape != expected.shape or not np.array_equal(actual, expected):
        maximum = (
            float(np.max(np.abs(actual - expected)))
            if actual.shape == expected.shape and actual.size
            else math.inf
        )
        raise AnalysisError(
            f"{label} is not an exact match; maximum absolute difference={maximum}"
        )


def validate_accepted_metrics(metrics: Mapping[str, Any], label: str) -> None:
    if not isinstance(metrics.get("accepted"), bool) or not metrics["accepted"]:
        raise AnalysisError(f"{label}: accepted must be JSON boolean true")
    if metrics.get("verdict") != ACCEPTED_VERDICT:
        raise AnalysisError(f"{label}: verdict is not {ACCEPTED_VERDICT}")
    rejection_reasons = metrics.get("rejection_reasons", [])
    if not isinstance(rejection_reasons, list) or rejection_reasons:
        raise AnalysisError(f"{label}: accepted artifact has rejection reasons")

    exact_one_fields = (
        "full_path_safety_pass",
        "independent_full_path_safety_pass",
        "independent_executed_prefix_safety_pass",
        "independent_joint_limit_pass",
        "independent_finite_joint_pass",
        "independent_finite_fk_pass",
        "independent_timestamp_pass",
        "independent_finite_orientation_pass",
        "independent_prior_orientation_pass",
        "independent_final_orientation_pass",
        "independent_finite_z_pass",
        "independent_prior_z_pass",
        "independent_final_z_pass",
    )
    for field in exact_one_fields:
        if field not in metrics or int(metrics[field]) != 1:
            raise AnalysisError(f"{label}: required accepted-validation field {field} != 1")
    if int(metrics.get("rollout_full_hard_joint_limit_violation_count", -1)) != 0:
        raise AnalysisError(f"{label}: recorded hard joint-limit violation")
    if float(metrics.get("rollout_full_hard_joint_limit_violation_magnitude", math.inf)) != 0.0:
        raise AnalysisError(f"{label}: nonzero hard joint-limit violation magnitude")
    if float(metrics.get("maximum_actual_internal_joint_step_rad", math.inf)) > 0.20:
        raise AnalysisError(f"{label}: recorded maximum internal joint step exceeds 0.20 rad")
    for enforced_key in ("orientation_constraint_enforced", "z_constraint_enforced"):
        if not isinstance(metrics.get(enforced_key), bool) or not metrics[enforced_key]:
            raise AnalysisError(f"{label}: {enforced_key} must be JSON boolean true")
    for frame_key in ("orientation_fk_frame", "z_fk_frame"):
        if metrics.get(frame_key) != DEFAULT_EE_LINK:
            raise AnalysisError(f"{label}: {frame_key} must be {DEFAULT_EE_LINK}")
    if float(metrics["maximum_prior_orientation_error_rad"]) > float(
        metrics["maximum_orientation_error_gate_rad"]
    ):
        raise AnalysisError(f"{label}: prior orientation gate failed")
    if float(metrics["maximum_final_orientation_error_rad"]) > float(
        metrics["maximum_orientation_error_gate_rad"]
    ):
        raise AnalysisError(f"{label}: diffusion orientation gate failed")
    if float(metrics["maximum_prior_z_error_m"]) > float(metrics["maximum_z_error_gate_m"]):
        raise AnalysisError(f"{label}: prior fixed-Z gate failed")
    if float(metrics["maximum_final_z_error_m"]) > float(metrics["maximum_z_error_gate_m"]):
        raise AnalysisError(f"{label}: diffusion fixed-Z gate failed")


def run_existing_artifact_consistency_checks(
    output_dir: Path,
    metrics: Mapping[str, Any],
    full: Mapping[str, Any],
) -> Dict[str, Any]:
    """Reuse the deployment validator's artifact-level consistency checks."""
    try:
        deployment_validator.require_files(output_dir)
        deployment_validator.validate_verdict(output_dir, metrics)
        segment_count = deployment_validator.validate_segment_arrays(full, metrics)
        deployment_validator.validate_frozen_config(metrics, full)
        deployment_validator.validate_full_npz_metadata(metrics, full)
        deployment_validator.validate_csv_agreement(output_dir, full)
        deployment_validator.validate_decision_candidate_rows(
            output_dir,
            metrics,
            full,
            segment_count,
        )
        deployment_validator.validate_approved_exports(output_dir, metrics, full)
        deployment_validator.validate_input_copy(output_dir, metrics, full)
        deployment_validator.validate_provenance(metrics)
    except Exception as exc:
        raise AnalysisError(
            f"{output_dir}: existing v8.1 deployment artifact validation failed: {exc}"
        ) from exc
    return {
        "existing_validator_artifact_checks_passed": True,
        "segment_count": int(segment_count),
        "accepted_verdict": ACCEPTED_VERDICT,
    }


def load_case(
    name: str,
    input_path_value: str,
    output_dir_value: str,
) -> LoadedCase:
    input_path = Path(input_path_value).expanduser().resolve()
    output_dir = Path(output_dir_value).expanduser().resolve()
    if not input_path.is_file():
        raise AnalysisError(f"{name}: deployment input does not exist: {input_path}")
    if not output_dir.is_dir():
        raise AnalysisError(f"{name}: diffusion output directory does not exist: {output_dir}")
    missing = [
        artifact
        for artifact in REQUIRED_DEPLOYMENT_ARTIFACTS
        if not (output_dir / artifact).is_file()
    ]
    if missing:
        raise AnalysisError(f"{name}: diffusion output lacks required artifacts: {missing}")

    full_path = output_dir / "deployment_trajectory_full.npz"
    metrics_path = output_dir / "deployment_metrics.json"
    approved_path = output_dir / "approved_simulation_trajectory.npz"
    input_data = load_npz(input_path)
    full = load_npz(full_path)
    approved = load_npz(approved_path)
    copied = load_npz(output_dir / "deployment_input_copy.npz")
    metrics = strict_json(metrics_path)

    require_keys(
        input_data,
        (
            "strong_prior_q",
            "desired_path",
            "timestamps",
            "target_rotation_matrix",
        ),
        f"{name}:deployment input",
    )
    require_keys(
        full,
        (
            "strong_prior_q",
            "final_q",
            "strong_prior_ee",
            "final_ee",
            "desired_path",
            "timestamps",
            "target_rotation_matrix",
            "strong_prior_orientation_error_rad",
            "final_orientation_error_rad",
            "verdict",
            "input_sha256",
            "input_file",
            "urdf_path",
            "urdf_sha256",
            "orientation_fk_frame",
        ),
        f"{name}:deployment_trajectory_full.npz",
    )
    require_keys(approved, ("q", "timestamps", "verdict"), f"{name}:approved NPZ")

    prior_q = finite_array(input_data["strong_prior_q"], f"{name}:input strong_prior_q", ndim=2)
    diffusion_q = finite_array(full["final_q"], f"{name}:output final_q", ndim=2)
    if prior_q.shape[1:] != (JOINT_DIM,):
        raise AnalysisError(
            f"{name}: input strong_prior_q must have shape [T,6], got {prior_q.shape}"
        )
    if diffusion_q.shape[1:] != (JOINT_DIM,):
        raise AnalysisError(
            f"{name}: output final_q must have shape [T,6], got {diffusion_q.shape}"
        )
    if len(prior_q) != len(diffusion_q):
        raise AnalysisError(
            f"{name}: prior and diffusion lengths differ: {len(prior_q)} != {len(diffusion_q)}"
        )
    sample_count = len(prior_q)
    desired = finite_array(
        input_data["desired_path"],
        f"{name}:input desired_path",
        shape=(sample_count, 3),
    )
    timestamps = finite_array(
        input_data["timestamps"],
        f"{name}:input timestamps",
        shape=(sample_count,),
    )
    target_rotation = finite_array(
        input_data["target_rotation_matrix"],
        f"{name}:input target_rotation_matrix",
        shape=(3, 3),
    )
    timestep_s = verify_timestamps(timestamps, f"{name}:input timestamps")

    full_prior = finite_array(
        full["strong_prior_q"],
        f"{name}:full strong_prior_q",
        shape=prior_q.shape,
    )
    full_desired = finite_array(
        full["desired_path"],
        f"{name}:full desired_path",
        shape=desired.shape,
    )
    full_timestamps = finite_array(
        full["timestamps"],
        f"{name}:full timestamps",
        shape=timestamps.shape,
    )
    full_target_rotation = finite_array(
        full["target_rotation_matrix"],
        f"{name}:full target_rotation_matrix",
        shape=(3, 3),
    )
    require_exact_array(full_prior, prior_q, f"{name}:output prior vs input prior")
    require_exact_array(full_desired, desired, f"{name}:output desired path vs input")
    require_exact_array(full_timestamps, timestamps, f"{name}:output timestamps vs input")
    require_exact_array(
        full_target_rotation,
        target_rotation,
        f"{name}:output target rotation vs input",
    )

    copied_prior = finite_array(
        copied["strong_prior_q"],
        f"{name}:deployment input copy strong_prior_q",
        shape=prior_q.shape,
    )
    require_exact_array(copied_prior, prior_q, f"{name}:deployment input copy prior")

    approved_q = finite_array(approved["q"], f"{name}:approved q", shape=diffusion_q.shape)
    approved_timestamps = finite_array(
        approved["timestamps"],
        f"{name}:approved timestamps",
        shape=timestamps.shape,
    )
    require_exact_array(approved_q, diffusion_q, f"{name}:approved q vs final_q")
    require_exact_array(
        approved_timestamps,
        timestamps,
        f"{name}:approved timestamps vs input",
    )

    validate_accepted_metrics(metrics, f"{name}:deployment_metrics.json")
    if scalar_text(full["verdict"], f"{name}:full verdict") != ACCEPTED_VERDICT:
        raise AnalysisError(f"{name}: full NPZ is not accepted")
    if scalar_text(approved["verdict"], f"{name}:approved verdict") != ACCEPTED_VERDICT:
        raise AnalysisError(f"{name}: approved NPZ is not accepted")
    if "accepted" in approved and not scalar_bool(approved["accepted"], f"{name}:approved accepted"):
        raise AnalysisError(f"{name}: approved NPZ accepted flag is false")

    input_sha256 = sha256_file(input_path)
    if scalar_text(full["input_sha256"], f"{name}:full input_sha256") != input_sha256:
        raise AnalysisError(f"{name}: full NPZ input SHA-256 does not match supplied input")
    if str(metrics.get("input_sha256")) != input_sha256:
        raise AnalysisError(f"{name}: metrics input SHA-256 does not match supplied input")
    provenance = metrics.get("provenance")
    if not isinstance(provenance, Mapping):
        raise AnalysisError(f"{name}: metrics provenance must be an object")
    if str(provenance.get("input_npz_sha256")) != input_sha256:
        raise AnalysisError(f"{name}: provenance input SHA-256 does not match supplied input")
    recorded_input = Path(str(metrics.get("input_file", ""))).expanduser().resolve()
    if recorded_input != input_path:
        raise AnalysisError(
            f"{name}: supplied input path {input_path} differs from recorded input path "
            f"{recorded_input}"
        )

    input_order, input_order_source = resolve_joint_order(input_data, f"{name}:input")
    full_order, full_order_source = resolve_joint_order(full, f"{name}:full")
    metrics_order, metrics_order_source = resolve_joint_order(metrics, f"{name}:metrics")
    approved_order, approved_order_source = resolve_joint_order(approved, f"{name}:approved")
    orders = (input_order, full_order, metrics_order, approved_order)
    authoritative_order = tuple(str(value) for value in DEFAULT_JOINT_NAMES)
    if any(order != authoritative_order for order in orders):
        raise AnalysisError(
            f"{name}: joint order mismatch; input={input_order}, full={full_order}, "
            f"metrics={metrics_order}, approved={approved_order}, "
            f"authoritative={authoritative_order}"
        )
    if scalar_text(full["orientation_fk_frame"], f"{name}:orientation_fk_frame") != DEFAULT_EE_LINK:
        raise AnalysisError(f"{name}: FK frame is not authoritative {DEFAULT_EE_LINK}")

    urdf_path = Path(scalar_text(full["urdf_path"], f"{name}:urdf_path")).expanduser().resolve()
    if not urdf_path.is_file():
        raise AnalysisError(f"{name}: recorded URDF does not exist: {urdf_path}")
    urdf_sha256 = sha256_file(urdf_path)
    if scalar_text(full["urdf_sha256"], f"{name}:urdf_sha256") != urdf_sha256:
        raise AnalysisError(f"{name}: full NPZ URDF SHA-256 mismatch")
    if str(metrics.get("urdf_sha256")) != urdf_sha256:
        raise AnalysisError(f"{name}: metrics URDF SHA-256 mismatch")

    validation = run_existing_artifact_consistency_checks(output_dir, metrics, full)
    robot = v7_evaluator.make_robot_context(urdf_path)
    if tuple(robot.joint_names) != authoritative_order:
        raise AnalysisError(f"{name}: robot context joint order is not authoritative")
    if robot.ee_link != DEFAULT_EE_LINK:
        raise AnalysisError(f"{name}: robot context end-effector frame is not authoritative")

    prior_position, prior_rotation, _ = trajectory_full_transform_fk(
        robot.robot,
        prior_q,
        robot.joint_names,
        robot.ee_link,
    )
    diffusion_position, diffusion_rotation, _ = trajectory_full_transform_fk(
        robot.robot,
        diffusion_q,
        robot.joint_names,
        robot.ee_link,
    )
    for key, recomputed in (
        ("strong_prior_ee", prior_position),
        ("final_ee", diffusion_position),
    ):
        stored = finite_array(
            full[key],
            f"{name}:stored {key}",
            shape=(sample_count, 3),
        )
        if not np.allclose(stored, recomputed, rtol=1.0e-5, atol=2.0e-5):
            raise AnalysisError(
                f"{name}: stored {key} disagrees with authoritative full-transform FK"
            )
    recomputed_prior_orientation_error = orientation_error_trajectory(
        target_rotation,
        prior_rotation,
    )
    recomputed_diffusion_orientation_error = orientation_error_trajectory(
        target_rotation,
        diffusion_rotation,
    )
    for key, recomputed in (
        ("strong_prior_orientation_error_rad", recomputed_prior_orientation_error),
        ("final_orientation_error_rad", recomputed_diffusion_orientation_error),
    ):
        stored = finite_array(
            full[key],
            f"{name}:stored {key}",
            shape=(sample_count,),
        )
        if not np.allclose(stored, recomputed, rtol=1.0e-7, atol=1.0e-9):
            raise AnalysisError(
                f"{name}: stored {key} disagrees with validator orientation error"
            )

    if "sampling_seed" not in metrics:
        raise AnalysisError(f"{name}: metrics lack sampling_seed")
    sampling_seed = int(metrics["sampling_seed"])
    metadata = {
        "case_name": name,
        "deployment_input_npz": str(input_path),
        "deployment_input_sha256": input_sha256,
        "diffusion_output_dir": str(output_dir),
        "deployment_trajectory_full_npz": str(full_path),
        "deployment_trajectory_full_sha256": sha256_file(full_path),
        "deployment_metrics_json": str(metrics_path),
        "deployment_metrics_sha256": sha256_file(metrics_path),
        "approved_simulation_trajectory_npz": str(approved_path),
        "approved_simulation_trajectory_sha256": sha256_file(approved_path),
        "deployment_path_id": str(metrics.get("deployment_path_id")),
        "input_path_name": str(metrics.get("input_path_name")),
        "input_file_recorded": str(metrics.get("input_file")),
        "source_method": str(metrics.get("source_method")),
        "source_checkpoint": str(metrics.get("source_checkpoint")),
        "source_checkpoint_sha256": str(metrics.get("source_checkpoint_sha256")),
        "checkpoint_state": str(metrics.get("checkpoint_state")),
        "checkpoint_state_hash": str(metrics.get("checkpoint_state_hash")),
        "sampling_seed": sampling_seed,
        "urdf_path": str(urdf_path),
        "urdf_sha256": urdf_sha256,
        "end_effector_frame": robot.ee_link,
        "artifact_keys": {
            "prior_joint_trajectory": "deployment input NPZ:strong_prior_q",
            "accepted_diffusion_trajectory": (
                "diffusion output:deployment_trajectory_full.npz:final_q"
            ),
            "approved_diffusion_cross_check": (
                "diffusion output:approved_simulation_trajectory.npz:q"
            ),
            "desired_cartesian_position": "deployment input NPZ:desired_path",
            "desired_cartesian_orientation": (
                "deployment input NPZ:target_rotation_matrix"
            ),
            "timestamps": "deployment input NPZ:timestamps",
        },
    }
    return LoadedCase(
        name=name,
        input_path=input_path,
        output_dir=output_dir,
        full_path=full_path,
        metrics_path=metrics_path,
        approved_path=approved_path,
        prior_q=prior_q,
        diffusion_q=diffusion_q,
        desired_position_m=desired,
        target_rotation_matrix=target_rotation,
        timestamps_s=timestamps,
        timestep_s=timestep_s,
        joint_order=authoritative_order,
        joint_order_sources={
            "input": input_order_source,
            "full_output": full_order_source,
            "metrics": metrics_order_source,
            "approved_output": approved_order_source,
        },
        robot=robot,
        lower_limits_rad=np.asarray(robot.lower, dtype=np.float64),
        upper_limits_rad=np.asarray(robot.upper, dtype=np.float64),
        metadata=metadata,
        validation=validation,
    )


def rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def derivatives(
    q_rad: np.ndarray,
    timestamps_s: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    velocity = np.gradient(q_rad, timestamps_s, axis=0, edge_order=2)
    acceleration = np.gradient(velocity, timestamps_s, axis=0, edge_order=2)
    jerk = np.gradient(acceleration, timestamps_s, axis=0, edge_order=2)
    for label, values in (
        ("velocity", velocity),
        ("acceleration", acceleration),
        ("jerk", jerk),
    ):
        if not np.all(np.isfinite(values)):
            raise AnalysisError(f"Finite-difference {label} contains non-finite values")
    return velocity, acceleration, jerk


def jacobian_diagnostics(
    robot: target_generator.RobotContext,
    q_rad: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    minimum_singular_values = np.empty(len(q_rad), dtype=np.float64)
    condition_numbers = np.empty(len(q_rad), dtype=np.float64)
    manipulability = np.empty(len(q_rad), dtype=np.float64)
    for sample_index, row in enumerate(q_rad):
        jacobian = target_generator.positional_jacobian(
            robot,
            row,
            epsilon=JACOBIAN_FINITE_DIFFERENCE_STEP_RAD,
        )
        singular_values = np.linalg.svd(jacobian, compute_uv=False)
        if singular_values.shape != (3,) or not np.all(np.isfinite(singular_values)):
            raise AnalysisError(
                f"Invalid translational Jacobian SVD at sample {sample_index}"
            )
        singular_values = np.maximum(singular_values, 0.0)
        minimum = float(singular_values[-1])
        largest = float(singular_values[0])
        tolerance = max(
            SVD_ABSOLUTE_TOLERANCE,
            max(jacobian.shape) * np.finfo(np.float64).eps * largest,
        )
        minimum_singular_values[sample_index] = minimum
        condition_numbers[sample_index] = (
            math.inf if minimum <= tolerance else largest / minimum
        )
        # For a 3x6 translational Jacobian, Yoshikawa's measure is the
        # product of its three singular values, equivalent to sqrt(det(JJ^T)).
        manipulability[sample_index] = float(np.prod(singular_values))
    if not np.all(np.isfinite(minimum_singular_values)):
        raise AnalysisError("Jacobian minimum singular values are non-finite")
    if not np.all(np.isfinite(manipulability)):
        raise AnalysisError("Jacobian manipulability values are non-finite")
    return minimum_singular_values, condition_numbers, manipulability


def compute_trajectory_metrics(
    loaded: LoadedCase,
    trajectory_type: str,
    q_rad: np.ndarray,
) -> TrajectoryMetrics:
    positions, rotations, _ = trajectory_full_transform_fk(
        loaded.robot.robot,
        q_rad,
        loaded.robot.joint_names,
        loaded.robot.ee_link,
    )
    positions = finite_array(
        positions,
        f"{loaded.name}:{trajectory_type} FK positions",
        shape=loaded.desired_position_m.shape,
    )
    rotations = finite_array(
        rotations,
        f"{loaded.name}:{trajectory_type} FK rotations",
        shape=(len(q_rad), 3, 3),
    )
    position_error = np.linalg.norm(positions - loaded.desired_position_m, axis=1)
    orientation_error = orientation_error_trajectory(
        loaded.target_rotation_matrix,
        rotations,
    )
    velocity, acceleration, jerk = derivatives(q_rad, loaded.timestamps_s)
    if len(q_rad) <= 2 * DERIVATIVE_INTERIOR_TRIM_SAMPLES:
        raise AnalysisError(
            f"{loaded.name}:{trajectory_type} requires more than "
            f"{2 * DERIVATIVE_INTERIOR_TRIM_SAMPLES} samples for interior "
            "derivative metrics"
        )
    interior_slice = slice(
        DERIVATIVE_INTERIOR_TRIM_SAMPLES,
        -DERIVATIVE_INTERIOR_TRIM_SAMPLES,
    )
    interior_acceleration = acceleration[interior_slice]
    interior_jerk = jerk[interior_slice]

    lower_distance = q_rad - loaded.lower_limits_rad[None, :]
    upper_distance = loaded.upper_limits_rad[None, :] - q_rad
    margin = np.minimum(lower_distance, upper_distance)
    ranges = loaded.upper_limits_rad - loaded.lower_limits_rad
    normalized_margin = margin / ranges[None, :]
    violation = np.logical_or(
        q_rad < loaded.lower_limits_rad[None, :] - HARD_JOINT_LIMIT_TOLERANCE_RAD,
        q_rad > loaded.upper_limits_rad[None, :] + HARD_JOINT_LIMIT_TOLERANCE_RAD,
    )
    min_singular, condition, manipulability = jacobian_diagnostics(
        loaded.robot,
        q_rad,
    )

    min_margin_flat = int(np.argmin(margin))
    min_margin_sample, min_margin_joint = np.unravel_index(
        min_margin_flat,
        margin.shape,
    )
    min_normalized_flat = int(np.argmin(normalized_margin))
    min_normalized_sample, min_normalized_joint = np.unravel_index(
        min_normalized_flat,
        normalized_margin.shape,
    )
    scalar = {
        "cartesian_mean_error_m": float(np.mean(position_error)),
        "cartesian_rms_error_m": rms(position_error),
        "cartesian_max_error_m": float(np.max(position_error)),
        "cartesian_final_error_m": float(position_error[-1]),
        "orientation_mean_error_rad": float(np.mean(orientation_error)),
        "orientation_rms_error_rad": rms(orientation_error),
        "orientation_max_error_rad": float(np.max(orientation_error)),
        "orientation_final_error_rad": float(orientation_error[-1]),
        "velocity_max_abs_rad_s": float(np.max(np.abs(velocity))),
        "velocity_rms_rad_s": rms(velocity),
        "acceleration_max_abs_rad_s2": float(np.max(np.abs(acceleration))),
        "acceleration_rms_rad_s2": rms(acceleration),
        "integrated_squared_acceleration_rad2_s3": float(
            np.sum(np.square(acceleration)) * loaded.timestep_s
        ),
        "interior_acceleration_max_abs_rad_s2": float(
            np.max(np.abs(interior_acceleration))
        ),
        "interior_acceleration_rms_rad_s2": rms(interior_acceleration),
        "interior_integrated_squared_acceleration_rad2_s3": float(
            np.sum(np.square(interior_acceleration)) * loaded.timestep_s
        ),
        "jerk_max_abs_rad_s3": float(np.max(np.abs(jerk))),
        "jerk_rms_rad_s3": rms(jerk),
        "integrated_squared_jerk_rad2_s5": float(
            np.sum(np.square(jerk)) * loaded.timestep_s
        ),
        "interior_jerk_max_abs_rad_s3": float(
            np.max(np.abs(interior_jerk))
        ),
        "interior_jerk_rms_rad_s3": rms(interior_jerk),
        "interior_integrated_squared_jerk_rad2_s5": float(
            np.sum(np.square(interior_jerk)) * loaded.timestep_s
        ),
        "minimum_joint_limit_margin_rad": float(np.min(margin)),
        "minimum_normalized_joint_limit_margin": float(np.min(normalized_margin)),
        "joint_limit_violation_count": float(np.count_nonzero(violation)),
        "translational_jacobian_min_singular_value_m_per_rad": float(
            np.min(min_singular)
        ),
        "translational_jacobian_max_condition_number": float(np.max(condition)),
        "translational_jacobian_min_manipulability_m3_per_rad3": float(
            np.min(manipulability)
        ),
    }
    per_joint = {
        "velocity_max_abs_rad_s": np.max(np.abs(velocity), axis=0),
        "velocity_rms_rad_s": np.sqrt(np.mean(np.square(velocity), axis=0)),
        "acceleration_max_abs_rad_s2": np.max(np.abs(acceleration), axis=0),
        "acceleration_rms_rad_s2": np.sqrt(np.mean(np.square(acceleration), axis=0)),
        "integrated_squared_acceleration_rad2_s3": (
            np.sum(np.square(acceleration), axis=0) * loaded.timestep_s
        ),
        "interior_acceleration_max_abs_rad_s2": np.max(
            np.abs(interior_acceleration),
            axis=0,
        ),
        "interior_acceleration_rms_rad_s2": np.sqrt(
            np.mean(np.square(interior_acceleration), axis=0)
        ),
        "interior_integrated_squared_acceleration_rad2_s3": (
            np.sum(np.square(interior_acceleration), axis=0)
            * loaded.timestep_s
        ),
        "jerk_max_abs_rad_s3": np.max(np.abs(jerk), axis=0),
        "jerk_rms_rad_s3": np.sqrt(np.mean(np.square(jerk), axis=0)),
        "integrated_squared_jerk_rad2_s5": (
            np.sum(np.square(jerk), axis=0) * loaded.timestep_s
        ),
        "interior_jerk_max_abs_rad_s3": np.max(
            np.abs(interior_jerk),
            axis=0,
        ),
        "interior_jerk_rms_rad_s3": np.sqrt(
            np.mean(np.square(interior_jerk), axis=0)
        ),
        "interior_integrated_squared_jerk_rad2_s5": (
            np.sum(np.square(interior_jerk), axis=0) * loaded.timestep_s
        ),
        "minimum_joint_limit_margin_rad": np.min(margin, axis=0),
        "minimum_normalized_joint_limit_margin": np.min(
            normalized_margin,
            axis=0,
        ),
        "joint_limit_violation_count": np.count_nonzero(violation, axis=0),
    }
    indices = {
        "maximum_cartesian_error_sample_index": int(np.argmax(position_error)),
        "maximum_orientation_error_sample_index": int(np.argmax(orientation_error)),
        "minimum_joint_limit_margin_sample_index": int(min_margin_sample),
        "minimum_joint_limit_margin_joint_index": int(min_margin_joint),
        "minimum_joint_limit_margin_joint_name": loaded.joint_order[min_margin_joint],
        "minimum_normalized_joint_limit_margin_sample_index": int(
            min_normalized_sample
        ),
        "minimum_normalized_joint_limit_margin_joint_index": int(
            min_normalized_joint
        ),
        "minimum_normalized_joint_limit_margin_joint_name": loaded.joint_order[
            min_normalized_joint
        ],
        "minimum_translational_jacobian_singular_value_sample_index": int(
            np.argmin(min_singular)
        ),
        "maximum_translational_jacobian_condition_number_sample_index": int(
            np.argmax(condition)
        ),
        "minimum_translational_jacobian_manipulability_sample_index": int(
            np.argmin(manipulability)
        ),
    }
    return TrajectoryMetrics(
        trajectory_type=trajectory_type,
        q_rad=q_rad,
        fk_position_m=positions,
        position_error_m=position_error,
        orientation_error_rad=orientation_error,
        velocity_rad_s=velocity,
        acceleration_rad_s2=acceleration,
        jerk_rad_s3=jerk,
        joint_limit_margin_rad=margin,
        normalized_joint_limit_margin=normalized_margin,
        joint_limit_violation=violation,
        jacobian_min_singular_value=min_singular,
        jacobian_condition_number=condition,
        jacobian_manipulability=manipulability,
        scalar=scalar,
        indices=indices,
        per_joint=per_joint,
    )


def interpretation(
    prior: float,
    diffusion: float,
    direction: str,
) -> str:
    delta = diffusion - prior
    tolerance = CHANGE_ABSOLUTE_TOLERANCE + CHANGE_RELATIVE_TOLERANCE * max(
        abs(prior),
        abs(diffusion),
    )
    if math.isinf(prior) or math.isinf(diffusion):
        if prior == diffusion:
            return "unchanged"
        if direction == "lower":
            return "improved" if diffusion < prior else "worsened"
        return "improved" if diffusion > prior else "worsened"
    if abs(delta) <= tolerance:
        return "unchanged"
    if direction == "lower":
        return "improved" if delta < 0.0 else "worsened"
    if direction == "higher":
        return "improved" if delta > 0.0 else "worsened"
    return "descriptive"


def scalar_delta(prior: float, diffusion: float) -> float | str:
    """Return diffusion-prior without silently manufacturing a NaN.

    Equal signed infinities can occur when both trajectories have a singular
    Jacobian.  Their arithmetic difference is undefined, so it is labeled
    explicitly while the direction-aware interpretation remains unchanged.
    """
    if math.isinf(prior) and prior == diffusion:
        return "undefined_infinity_minus_infinity"
    return diffusion - prior


def analyze_case(loaded: LoadedCase) -> CaseAnalysis:
    prior = compute_trajectory_metrics(loaded, "prior", loaded.prior_q)
    diffusion = compute_trajectory_metrics(
        loaded,
        "diffusion",
        loaded.diffusion_q,
    )
    difference_q = loaded.diffusion_q - loaded.prior_q
    difference_velocity, difference_acceleration, difference_jerk = derivatives(
        difference_q,
        loaded.timestamps_s,
    )
    interior_slice = slice(
        DERIVATIVE_INTERIOR_TRIM_SAMPLES,
        -DERIVATIVE_INTERIOR_TRIM_SAMPLES,
    )
    difference_interior_acceleration = difference_acceleration[interior_slice]
    difference_interior_jerk = difference_jerk[interior_slice]
    global_flat = int(np.argmax(np.abs(difference_q)))
    global_sample, global_joint = np.unravel_index(global_flat, difference_q.shape)
    difference: Dict[str, Any] = {
        "rms_joint_difference_rad": rms(difference_q),
        "maximum_absolute_joint_difference_rad": float(
            np.max(np.abs(difference_q))
        ),
        "mean_absolute_joint_difference_rad": float(
            np.mean(np.abs(difference_q))
        ),
        "per_joint_rms_difference_rad": np.sqrt(
            np.mean(np.square(difference_q), axis=0)
        ),
        "per_joint_maximum_absolute_difference_rad": np.max(
            np.abs(difference_q),
            axis=0,
        ),
        "per_joint_mean_absolute_difference_rad": np.mean(
            np.abs(difference_q),
            axis=0,
        ),
        "difference_first_sample_rad": difference_q[0],
        "difference_last_sample_rad": difference_q[-1],
        "global_maximum_sample_index": int(global_sample),
        "global_maximum_joint_index": int(global_joint),
        "global_maximum_joint_name": loaded.joint_order[global_joint],
        "correction_velocity_max_abs_rad_s": float(
            np.max(np.abs(difference_velocity))
        ),
        "correction_velocity_rms_rad_s": rms(difference_velocity),
        "correction_acceleration_max_abs_rad_s2": float(
            np.max(np.abs(difference_acceleration))
        ),
        "correction_acceleration_rms_rad_s2": rms(difference_acceleration),
        "correction_integrated_squared_acceleration_rad2_s3": float(
            np.sum(np.square(difference_acceleration)) * loaded.timestep_s
        ),
        "correction_interior_acceleration_max_abs_rad_s2": float(
            np.max(np.abs(difference_interior_acceleration))
        ),
        "correction_interior_acceleration_rms_rad_s2": rms(
            difference_interior_acceleration
        ),
        "correction_interior_integrated_squared_acceleration_rad2_s3": float(
            np.sum(np.square(difference_interior_acceleration))
            * loaded.timestep_s
        ),
        "correction_jerk_max_abs_rad_s3": float(
            np.max(np.abs(difference_jerk))
        ),
        "correction_jerk_rms_rad_s3": rms(difference_jerk),
        "correction_integrated_squared_jerk_rad2_s5": float(
            np.sum(np.square(difference_jerk)) * loaded.timestep_s
        ),
        "correction_interior_jerk_max_abs_rad_s3": float(
            np.max(np.abs(difference_interior_jerk))
        ),
        "correction_interior_jerk_rms_rad_s3": rms(
            difference_interior_jerk
        ),
        "correction_interior_integrated_squared_jerk_rad2_s5": float(
            np.sum(np.square(difference_interior_jerk))
            * loaded.timestep_s
        ),
    }
    comparison: Dict[str, Dict[str, Any]] = {}
    for spec in METRIC_SPECS:
        prior_value = prior.scalar[spec.key]
        diffusion_value = diffusion.scalar[spec.key]
        comparison[spec.key] = {
            "prior": prior_value,
            "diffusion": diffusion_value,
            "delta": scalar_delta(prior_value, diffusion_value),
            "directionality": spec.direction,
            "interpretation": interpretation(
                prior_value,
                diffusion_value,
                spec.direction,
            ),
        }
    return CaseAnalysis(
        loaded=loaded,
        prior=prior,
        diffusion=diffusion,
        difference_q_rad=difference_q,
        difference_velocity_rad_s=difference_velocity,
        difference_acceleration_rad_s2=difference_acceleration,
        difference_jerk_rad_s3=difference_jerk,
        difference=difference,
        comparison=comparison,
    )


def metric_column(spec: MetricSpec, suffix: str) -> str:
    if spec.unit_suffix:
        return f"{spec.column_stem}_{suffix}_{spec.unit_suffix}"
    return f"{spec.column_stem}_{suffix}"


def per_stroke_rows(analyses: Sequence[CaseAnalysis]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for analysis in analyses:
        loaded = analysis.loaded
        row: Dict[str, Any] = {
            "case_name": loaded.name,
            "deployment_input_npz": str(loaded.input_path),
            "diffusion_output_dir": str(loaded.output_dir),
            "deployment_trajectory_full_npz": str(loaded.full_path),
            "deployment_metrics_json": str(loaded.metrics_path),
            "approved_simulation_trajectory_npz": str(loaded.approved_path),
            "sample_count": len(loaded.timestamps_s),
            "timestep_s": loaded.timestep_s,
            "joint_order": "|".join(loaded.joint_order),
            "accepted_verdict": ACCEPTED_VERDICT,
        }
        for spec in METRIC_SPECS:
            comparison = analysis.comparison[spec.key]
            row[metric_column(spec, "prior")] = comparison["prior"]
            row[metric_column(spec, "diffusion")] = comparison["diffusion"]
            row[metric_column(spec, "delta")] = comparison["delta"]
            row[f"{spec.column_stem}_interpretation"] = comparison[
                "interpretation"
            ]
        row.update(
            {
                "joint_difference_rms_rad": analysis.difference[
                    "rms_joint_difference_rad"
                ],
                "joint_difference_max_abs_rad": analysis.difference[
                    "maximum_absolute_joint_difference_rad"
                ],
                "joint_difference_mean_abs_rad": analysis.difference[
                    "mean_absolute_joint_difference_rad"
                ],
                "joint_difference_global_max_sample_index": analysis.difference[
                    "global_maximum_sample_index"
                ],
                "joint_difference_global_max_joint_name": analysis.difference[
                    "global_maximum_joint_name"
                ],
                "prior_maximum_cartesian_error_sample_index": analysis.prior.indices[
                    "maximum_cartesian_error_sample_index"
                ],
                "diffusion_maximum_cartesian_error_sample_index": (
                    analysis.diffusion.indices[
                        "maximum_cartesian_error_sample_index"
                    ]
                ),
                "prior_maximum_orientation_error_sample_index": (
                    analysis.prior.indices[
                        "maximum_orientation_error_sample_index"
                    ]
                ),
                "diffusion_maximum_orientation_error_sample_index": (
                    analysis.diffusion.indices[
                        "maximum_orientation_error_sample_index"
                    ]
                ),
                "prior_minimum_joint_limit_margin_sample_index": (
                    analysis.prior.indices[
                        "minimum_joint_limit_margin_sample_index"
                    ]
                ),
                "prior_minimum_joint_limit_margin_joint_name": (
                    analysis.prior.indices[
                        "minimum_joint_limit_margin_joint_name"
                    ]
                ),
                "diffusion_minimum_joint_limit_margin_sample_index": (
                    analysis.diffusion.indices[
                        "minimum_joint_limit_margin_sample_index"
                    ]
                ),
                "diffusion_minimum_joint_limit_margin_joint_name": (
                    analysis.diffusion.indices[
                        "minimum_joint_limit_margin_joint_name"
                    ]
                ),
                "prior_minimum_jacobian_singular_value_sample_index": (
                    analysis.prior.indices[
                        "minimum_translational_jacobian_singular_value_sample_index"
                    ]
                ),
                "diffusion_minimum_jacobian_singular_value_sample_index": (
                    analysis.diffusion.indices[
                        "minimum_translational_jacobian_singular_value_sample_index"
                    ]
                ),
                "prior_minimum_manipulability_sample_index": (
                    analysis.prior.indices[
                        "minimum_translational_jacobian_manipulability_sample_index"
                    ]
                ),
                "diffusion_minimum_manipulability_sample_index": (
                    analysis.diffusion.indices[
                        "minimum_translational_jacobian_manipulability_sample_index"
                    ]
                ),
            }
        )
        rows.append(row)
    return rows


def per_joint_rows(analyses: Sequence[CaseAnalysis]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for analysis in analyses:
        loaded = analysis.loaded
        for trajectory_type, metrics in (
            ("prior", analysis.prior),
            ("diffusion", analysis.diffusion),
        ):
            for joint_index, joint_name in enumerate(loaded.joint_order):
                rows.append(
                    {
                        "case_name": loaded.name,
                        "trajectory_type": trajectory_type,
                        "joint_index": joint_index,
                        "joint_name": joint_name,
                        "velocity_max_abs_rad_s": float(
                            metrics.per_joint["velocity_max_abs_rad_s"][
                                joint_index
                            ]
                        ),
                        "velocity_rms_rad_s": float(
                            metrics.per_joint["velocity_rms_rad_s"][
                                joint_index
                            ]
                        ),
                        "acceleration_max_abs_rad_s2": float(
                            metrics.per_joint[
                                "acceleration_max_abs_rad_s2"
                            ][joint_index]
                        ),
                        "acceleration_rms_rad_s2": float(
                            metrics.per_joint["acceleration_rms_rad_s2"][
                                joint_index
                            ]
                        ),
                        "integrated_squared_acceleration_rad2_s3": float(
                            metrics.per_joint[
                                "integrated_squared_acceleration_rad2_s3"
                            ][joint_index]
                        ),
                        "interior_acceleration_max_abs_rad_s2": float(
                            metrics.per_joint[
                                "interior_acceleration_max_abs_rad_s2"
                            ][joint_index]
                        ),
                        "interior_acceleration_rms_rad_s2": float(
                            metrics.per_joint[
                                "interior_acceleration_rms_rad_s2"
                            ][joint_index]
                        ),
                        "interior_integrated_squared_acceleration_rad2_s3": float(
                            metrics.per_joint[
                                "interior_integrated_squared_acceleration_rad2_s3"
                            ][joint_index]
                        ),
                        "jerk_max_abs_rad_s3": float(
                            metrics.per_joint["jerk_max_abs_rad_s3"][
                                joint_index
                            ]
                        ),
                        "jerk_rms_rad_s3": float(
                            metrics.per_joint["jerk_rms_rad_s3"][joint_index]
                        ),
                        "integrated_squared_jerk_rad2_s5": float(
                            metrics.per_joint[
                                "integrated_squared_jerk_rad2_s5"
                            ][joint_index]
                        ),
                        "interior_jerk_max_abs_rad_s3": float(
                            metrics.per_joint[
                                "interior_jerk_max_abs_rad_s3"
                            ][joint_index]
                        ),
                        "interior_jerk_rms_rad_s3": float(
                            metrics.per_joint["interior_jerk_rms_rad_s3"][
                                joint_index
                            ]
                        ),
                        "interior_integrated_squared_jerk_rad2_s5": float(
                            metrics.per_joint[
                                "interior_integrated_squared_jerk_rad2_s5"
                            ][joint_index]
                        ),
                        "minimum_joint_limit_margin_rad": float(
                            metrics.per_joint[
                                "minimum_joint_limit_margin_rad"
                            ][joint_index]
                        ),
                        "minimum_normalized_joint_limit_margin": float(
                            metrics.per_joint[
                                "minimum_normalized_joint_limit_margin"
                            ][joint_index]
                        ),
                        "joint_limit_violation_count": int(
                            metrics.per_joint[
                                "joint_limit_violation_count"
                            ][joint_index]
                        ),
                    }
                )

        correction_interior_slice = slice(
            DERIVATIVE_INTERIOR_TRIM_SAMPLES,
            -DERIVATIVE_INTERIOR_TRIM_SAMPLES,
        )
        correction_interior_acceleration = (
            analysis.difference_acceleration_rad_s2[
                correction_interior_slice
            ]
        )
        correction_interior_jerk = analysis.difference_jerk_rad_s3[
            correction_interior_slice
        ]
        for joint_index, joint_name in enumerate(loaded.joint_order):
            correction_acceleration = analysis.difference_acceleration_rad_s2[
                :,
                joint_index,
            ]
            correction_jerk = analysis.difference_jerk_rad_s3[
                :,
                joint_index,
            ]
            rows.append(
                {
                    "case_name": loaded.name,
                    "trajectory_type": "correction",
                    "joint_index": joint_index,
                    "joint_name": joint_name,
                    "correction_joint_difference_rms_rad": float(
                        analysis.difference[
                            "per_joint_rms_difference_rad"
                        ][joint_index]
                    ),
                    "correction_joint_difference_max_abs_rad": float(
                        analysis.difference[
                            "per_joint_maximum_absolute_difference_rad"
                        ][joint_index]
                    ),
                    "correction_joint_difference_mean_abs_rad": float(
                        analysis.difference[
                            "per_joint_mean_absolute_difference_rad"
                        ][joint_index]
                    ),
                    "correction_joint_difference_first_sample_rad": float(
                        analysis.difference[
                            "difference_first_sample_rad"
                        ][joint_index]
                    ),
                    "correction_joint_difference_last_sample_rad": float(
                        analysis.difference[
                            "difference_last_sample_rad"
                        ][joint_index]
                    ),
                    "correction_velocity_max_abs_rad_s": float(
                        np.max(
                            np.abs(
                                analysis.difference_velocity_rad_s[
                                    :,
                                    joint_index,
                                ]
                            )
                        )
                    ),
                    "correction_velocity_rms_rad_s": rms(
                        analysis.difference_velocity_rad_s[:, joint_index]
                    ),
                    "correction_acceleration_max_abs_rad_s2": float(
                        np.max(np.abs(correction_acceleration))
                    ),
                    "correction_acceleration_rms_rad_s2": rms(
                        correction_acceleration
                    ),
                    "correction_integrated_squared_acceleration_rad2_s3": float(
                        np.sum(np.square(correction_acceleration))
                        * loaded.timestep_s
                    ),
                    "correction_interior_acceleration_max_abs_rad_s2": float(
                        np.max(
                            np.abs(
                                correction_interior_acceleration[
                                    :,
                                    joint_index,
                                ]
                            )
                        )
                    ),
                    "correction_interior_acceleration_rms_rad_s2": rms(
                        correction_interior_acceleration[:, joint_index]
                    ),
                    "correction_interior_integrated_squared_acceleration_rad2_s3": float(
                        np.sum(
                            np.square(
                                correction_interior_acceleration[
                                    :,
                                    joint_index,
                                ]
                            )
                        )
                        * loaded.timestep_s
                    ),
                    "correction_jerk_max_abs_rad_s3": float(
                        np.max(np.abs(correction_jerk))
                    ),
                    "correction_jerk_rms_rad_s3": rms(correction_jerk),
                    "correction_integrated_squared_jerk_rad2_s5": float(
                        np.sum(np.square(correction_jerk))
                        * loaded.timestep_s
                    ),
                    "correction_interior_jerk_max_abs_rad_s3": float(
                        np.max(
                            np.abs(
                                correction_interior_jerk[:, joint_index]
                            )
                        )
                    ),
                    "correction_interior_jerk_rms_rad_s3": rms(
                        correction_interior_jerk[:, joint_index]
                    ),
                    "correction_interior_integrated_squared_jerk_rad2_s5": float(
                        np.sum(
                            np.square(
                                correction_interior_jerk[:, joint_index]
                            )
                        )
                        * loaded.timestep_s
                    ),
                }
            )

            prior_joint = analysis.prior.per_joint
            diffusion_joint = analysis.diffusion.per_joint
            rows.append(
                {
                    "case_name": loaded.name,
                    "trajectory_type": "comparison",
                    "joint_index": joint_index,
                    "joint_name": joint_name,
                    "velocity_max_abs_delta_rad_s": float(
                        diffusion_joint["velocity_max_abs_rad_s"][joint_index]
                        - prior_joint["velocity_max_abs_rad_s"][joint_index]
                    ),
                    "velocity_rms_delta_rad_s": float(
                        diffusion_joint["velocity_rms_rad_s"][joint_index]
                        - prior_joint["velocity_rms_rad_s"][joint_index]
                    ),
                    "acceleration_max_abs_delta_rad_s2": float(
                        diffusion_joint["acceleration_max_abs_rad_s2"][
                            joint_index
                        ]
                        - prior_joint["acceleration_max_abs_rad_s2"][
                            joint_index
                        ]
                    ),
                    "acceleration_rms_delta_rad_s2": float(
                        diffusion_joint["acceleration_rms_rad_s2"][joint_index]
                        - prior_joint["acceleration_rms_rad_s2"][joint_index]
                    ),
                    "integrated_squared_acceleration_delta_rad2_s3": float(
                        diffusion_joint[
                            "integrated_squared_acceleration_rad2_s3"
                        ][joint_index]
                        - prior_joint[
                            "integrated_squared_acceleration_rad2_s3"
                        ][joint_index]
                    ),
                    "interior_acceleration_max_abs_delta_rad_s2": float(
                        diffusion_joint[
                            "interior_acceleration_max_abs_rad_s2"
                        ][joint_index]
                        - prior_joint[
                            "interior_acceleration_max_abs_rad_s2"
                        ][joint_index]
                    ),
                    "interior_acceleration_rms_delta_rad_s2": float(
                        diffusion_joint[
                            "interior_acceleration_rms_rad_s2"
                        ][joint_index]
                        - prior_joint[
                            "interior_acceleration_rms_rad_s2"
                        ][joint_index]
                    ),
                    "interior_integrated_squared_acceleration_delta_rad2_s3": float(
                        diffusion_joint[
                            "interior_integrated_squared_acceleration_rad2_s3"
                        ][joint_index]
                        - prior_joint[
                            "interior_integrated_squared_acceleration_rad2_s3"
                        ][joint_index]
                    ),
                    "jerk_max_abs_delta_rad_s3": float(
                        diffusion_joint["jerk_max_abs_rad_s3"][joint_index]
                        - prior_joint["jerk_max_abs_rad_s3"][joint_index]
                    ),
                    "jerk_rms_delta_rad_s3": float(
                        diffusion_joint["jerk_rms_rad_s3"][joint_index]
                        - prior_joint["jerk_rms_rad_s3"][joint_index]
                    ),
                    "integrated_squared_jerk_delta_rad2_s5": float(
                        diffusion_joint["integrated_squared_jerk_rad2_s5"][
                            joint_index
                        ]
                        - prior_joint["integrated_squared_jerk_rad2_s5"][
                            joint_index
                        ]
                    ),
                    "interior_jerk_max_abs_delta_rad_s3": float(
                        diffusion_joint["interior_jerk_max_abs_rad_s3"][
                            joint_index
                        ]
                        - prior_joint["interior_jerk_max_abs_rad_s3"][
                            joint_index
                        ]
                    ),
                    "interior_jerk_rms_delta_rad_s3": float(
                        diffusion_joint["interior_jerk_rms_rad_s3"][
                            joint_index
                        ]
                        - prior_joint["interior_jerk_rms_rad_s3"][
                            joint_index
                        ]
                    ),
                    "interior_integrated_squared_jerk_delta_rad2_s5": float(
                        diffusion_joint[
                            "interior_integrated_squared_jerk_rad2_s5"
                        ][joint_index]
                        - prior_joint[
                            "interior_integrated_squared_jerk_rad2_s5"
                        ][joint_index]
                    ),
                    "minimum_joint_limit_margin_delta_rad": float(
                        diffusion_joint["minimum_joint_limit_margin_rad"][
                            joint_index
                        ]
                        - prior_joint["minimum_joint_limit_margin_rad"][
                            joint_index
                        ]
                    ),
                    "minimum_normalized_joint_limit_margin_delta": float(
                        diffusion_joint[
                            "minimum_normalized_joint_limit_margin"
                        ][joint_index]
                        - prior_joint[
                            "minimum_normalized_joint_limit_margin"
                        ][joint_index]
                    ),
                    "joint_limit_violation_count_delta": int(
                        diffusion_joint["joint_limit_violation_count"][
                            joint_index
                        ]
                        - prior_joint["joint_limit_violation_count"][
                            joint_index
                        ]
                    ),
                }
            )
    return rows


def per_sample_rows(analyses: Sequence[CaseAnalysis]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for analysis in analyses:
        loaded = analysis.loaded
        for sample_index, time_s in enumerate(loaded.timestamps_s):
            for trajectory_type, metrics in (
                ("prior", analysis.prior),
                ("diffusion", analysis.diffusion),
            ):
                margin_row = metrics.joint_limit_margin_rad[sample_index]
                margin_joint_index = int(np.argmin(margin_row))
                row: Dict[str, Any] = {
                    "case_name": loaded.name,
                    "trajectory_type": trajectory_type,
                    "sample_index": sample_index,
                    "time_s": float(time_s),
                    "cartesian_position_error_m": float(
                        metrics.position_error_m[sample_index]
                    ),
                    "orientation_error_rad": float(
                        metrics.orientation_error_rad[sample_index]
                    ),
                    "minimum_joint_limit_margin_rad": float(
                        margin_row[margin_joint_index]
                    ),
                    "minimum_joint_limit_margin_joint_name": loaded.joint_order[
                        margin_joint_index
                    ],
                    "minimum_normalized_joint_limit_margin": float(
                        np.min(
                            metrics.normalized_joint_limit_margin[sample_index]
                        )
                    ),
                    "joint_limit_violation_count": int(
                        np.count_nonzero(
                            metrics.joint_limit_violation[sample_index]
                        )
                    ),
                    "translational_jacobian_min_singular_value_m_per_rad": float(
                        metrics.jacobian_min_singular_value[sample_index]
                    ),
                    "translational_jacobian_condition_number": float(
                        metrics.jacobian_condition_number[sample_index]
                    ),
                    "translational_jacobian_manipulability_m3_per_rad3": float(
                        metrics.jacobian_manipulability[sample_index]
                    ),
                }
                for joint_index, joint_name in enumerate(loaded.joint_order):
                    row[f"{joint_name}_velocity_rad_s"] = float(
                        metrics.velocity_rad_s[sample_index, joint_index]
                    )
                    row[f"{joint_name}_acceleration_rad_s2"] = float(
                        metrics.acceleration_rad_s2[sample_index, joint_index]
                    )
                    row[f"{joint_name}_jerk_rad_s3"] = float(
                        metrics.jerk_rad_s3[sample_index, joint_index]
                    )
                rows.append(row)

            prior_margin_row = analysis.prior.joint_limit_margin_rad[
                sample_index
            ]
            diffusion_margin_row = (
                analysis.diffusion.joint_limit_margin_rad[sample_index]
            )
            prior_margin_joint_index = int(np.argmin(prior_margin_row))
            diffusion_margin_joint_index = int(
                np.argmin(diffusion_margin_row)
            )
            prior_minimum_margin = float(
                prior_margin_row[prior_margin_joint_index]
            )
            diffusion_minimum_margin = float(
                diffusion_margin_row[diffusion_margin_joint_index]
            )
            prior_minimum_normalized_margin = float(
                np.min(
                    analysis.prior.normalized_joint_limit_margin[
                        sample_index
                    ]
                )
            )
            diffusion_minimum_normalized_margin = float(
                np.min(
                    analysis.diffusion.normalized_joint_limit_margin[
                        sample_index
                    ]
                )
            )
            prior_condition = float(
                analysis.prior.jacobian_condition_number[sample_index]
            )
            diffusion_condition = float(
                analysis.diffusion.jacobian_condition_number[sample_index]
            )
            both_condition_infinite = math.isinf(
                prior_condition
            ) and math.isinf(diffusion_condition)
            comparison_row: Dict[str, Any] = {
                "case_name": loaded.name,
                "trajectory_type": "comparison",
                "sample_index": sample_index,
                "time_s": float(time_s),
                "cartesian_position_error_delta_m": float(
                    analysis.diffusion.position_error_m[sample_index]
                    - analysis.prior.position_error_m[sample_index]
                ),
                "orientation_error_delta_rad": float(
                    analysis.diffusion.orientation_error_rad[sample_index]
                    - analysis.prior.orientation_error_rad[sample_index]
                ),
                "prior_minimum_margin_rad": prior_minimum_margin,
                "prior_minimum_margin_joint_name": loaded.joint_order[
                    prior_margin_joint_index
                ],
                "diffusion_minimum_margin_rad": diffusion_minimum_margin,
                "diffusion_minimum_margin_joint_name": loaded.joint_order[
                    diffusion_margin_joint_index
                ],
                "minimum_margin_delta_rad": (
                    diffusion_minimum_margin - prior_minimum_margin
                ),
                "prior_minimum_normalized_margin": (
                    prior_minimum_normalized_margin
                ),
                "diffusion_minimum_normalized_margin": (
                    diffusion_minimum_normalized_margin
                ),
                "minimum_normalized_margin_delta": (
                    diffusion_minimum_normalized_margin
                    - prior_minimum_normalized_margin
                ),
                "joint_limit_violation_count_delta": int(
                    np.count_nonzero(
                        analysis.diffusion.joint_limit_violation[sample_index]
                    )
                    - np.count_nonzero(
                        analysis.prior.joint_limit_violation[sample_index]
                    )
                ),
                "translational_jacobian_min_singular_value_delta_m_per_rad": float(
                    analysis.diffusion.jacobian_min_singular_value[sample_index]
                    - analysis.prior.jacobian_min_singular_value[sample_index]
                ),
                "translational_jacobian_condition_number_delta": (
                    float(
                        diffusion_condition - prior_condition
                    )
                    if not both_condition_infinite
                    else None
                ),
                "translational_jacobian_condition_number_delta_status": (
                    "undefined_infinity_minus_infinity"
                    if both_condition_infinite
                    else "defined"
                ),
                "translational_jacobian_manipulability_delta_m3_per_rad3": float(
                    analysis.diffusion.jacobian_manipulability[sample_index]
                    - analysis.prior.jacobian_manipulability[sample_index]
                ),
            }
            for joint_index, joint_name in enumerate(loaded.joint_order):
                comparison_row[
                    f"q{joint_index + 1}_correction_velocity_rad_s"
                ] = float(
                    analysis.difference_velocity_rad_s[
                        sample_index,
                        joint_index,
                    ]
                )
                comparison_row[
                    f"q{joint_index + 1}_correction_acceleration_rad_s2"
                ] = float(
                    analysis.difference_acceleration_rad_s2[
                        sample_index,
                        joint_index,
                    ]
                )
                comparison_row[
                    f"q{joint_index + 1}_correction_jerk_rad_s3"
                ] = float(
                    analysis.difference_jerk_rad_s3[
                        sample_index,
                        joint_index,
                    ]
                )
            rows.append(comparison_row)
    return rows


def select_method_extreme(
    records: Sequence[Mapping[str, Any]],
    field: str,
    direction: str,
    *,
    worst: bool,
) -> Mapping[str, Any]:
    if direction == "lower":
        selector = max if worst else min
    else:
        selector = min if worst else max
    return selector(records, key=lambda record: float(record[field]))


def paired_delta_order_value(value: float | str) -> float:
    if isinstance(value, str):
        if value != "undefined_infinity_minus_infinity":
            raise AnalysisError(f"Unsupported paired delta value: {value}")
        # Equal infinite condition numbers are explicitly undefined arithmetically
        # but unchanged directionally; use zero only for deterministic ordering.
        return 0.0
    return float(value)


def select_paired_delta_extreme(
    records: Sequence[Mapping[str, Any]],
    direction: str,
    *,
    worst: bool,
) -> Mapping[str, Any]:
    if direction == "lower":
        selector = max if worst else min
    else:
        selector = min if worst else max
    return selector(
        records,
        key=lambda record: paired_delta_order_value(record["delta"]),
    )


def pooled_metric(
    analyses: Sequence[CaseAnalysis],
    trajectory_type: str,
    key: str,
) -> float:
    metrics = [
        analysis.prior if trajectory_type == "prior" else analysis.diffusion
        for analysis in analyses
    ]
    if key == "cartesian_mean_error_m":
        return float(np.mean(np.concatenate([item.position_error_m for item in metrics])))
    if key == "cartesian_rms_error_m":
        return rms(np.concatenate([item.position_error_m for item in metrics]))
    if key == "cartesian_max_error_m":
        return float(np.max(np.concatenate([item.position_error_m for item in metrics])))
    if key == "orientation_mean_error_rad":
        return float(
            np.mean(np.concatenate([item.orientation_error_rad for item in metrics]))
        )
    if key == "orientation_rms_error_rad":
        return rms(np.concatenate([item.orientation_error_rad for item in metrics]))
    if key == "orientation_max_error_rad":
        return float(
            np.max(np.concatenate([item.orientation_error_rad for item in metrics]))
        )
    if key == "velocity_max_abs_rad_s":
        return float(
            np.max(np.abs(np.concatenate([item.velocity_rad_s for item in metrics])))
        )
    if key == "velocity_rms_rad_s":
        return rms(np.concatenate([item.velocity_rad_s for item in metrics]))
    if key == "acceleration_max_abs_rad_s2":
        return float(
            np.max(
                np.abs(np.concatenate([item.acceleration_rad_s2 for item in metrics]))
            )
        )
    if key == "acceleration_rms_rad_s2":
        return rms(np.concatenate([item.acceleration_rad_s2 for item in metrics]))
    if key == "integrated_squared_acceleration_rad2_s3":
        return float(sum(item.scalar[key] for item in metrics))
    if key == "interior_acceleration_max_abs_rad_s2":
        return float(
            np.max(
                np.abs(
                    np.concatenate(
                        [
                            item.acceleration_rad_s2[
                                DERIVATIVE_INTERIOR_TRIM_SAMPLES:
                                -DERIVATIVE_INTERIOR_TRIM_SAMPLES
                            ]
                            for item in metrics
                        ]
                    )
                )
            )
        )
    if key == "interior_acceleration_rms_rad_s2":
        return rms(
            np.concatenate(
                [
                    item.acceleration_rad_s2[
                        DERIVATIVE_INTERIOR_TRIM_SAMPLES:
                        -DERIVATIVE_INTERIOR_TRIM_SAMPLES
                    ]
                    for item in metrics
                ]
            )
        )
    if key == "interior_integrated_squared_acceleration_rad2_s3":
        return float(sum(item.scalar[key] for item in metrics))
    if key == "jerk_max_abs_rad_s3":
        return float(
            np.max(np.abs(np.concatenate([item.jerk_rad_s3 for item in metrics])))
        )
    if key == "jerk_rms_rad_s3":
        return rms(np.concatenate([item.jerk_rad_s3 for item in metrics]))
    if key == "integrated_squared_jerk_rad2_s5":
        return float(sum(item.scalar[key] for item in metrics))
    if key == "interior_jerk_max_abs_rad_s3":
        return float(
            np.max(
                np.abs(
                    np.concatenate(
                        [
                            item.jerk_rad_s3[
                                DERIVATIVE_INTERIOR_TRIM_SAMPLES:
                                -DERIVATIVE_INTERIOR_TRIM_SAMPLES
                            ]
                            for item in metrics
                        ]
                    )
                )
            )
        )
    if key == "interior_jerk_rms_rad_s3":
        return rms(
            np.concatenate(
                [
                    item.jerk_rad_s3[
                        DERIVATIVE_INTERIOR_TRIM_SAMPLES:
                        -DERIVATIVE_INTERIOR_TRIM_SAMPLES
                    ]
                    for item in metrics
                ]
            )
        )
    if key == "interior_integrated_squared_jerk_rad2_s5":
        return float(sum(item.scalar[key] for item in metrics))
    if key == "minimum_joint_limit_margin_rad":
        return float(
            np.min(
                np.concatenate(
                    [item.joint_limit_margin_rad.reshape(-1) for item in metrics]
                )
            )
        )
    if key == "minimum_normalized_joint_limit_margin":
        return float(
            np.min(
                np.concatenate(
                    [
                        item.normalized_joint_limit_margin.reshape(-1)
                        for item in metrics
                    ]
                )
            )
        )
    if key == "joint_limit_violation_count":
        return float(sum(np.count_nonzero(item.joint_limit_violation) for item in metrics))
    if key == "translational_jacobian_min_singular_value_m_per_rad":
        return float(
            np.min(
                np.concatenate(
                    [item.jacobian_min_singular_value for item in metrics]
                )
            )
        )
    if key == "translational_jacobian_max_condition_number":
        return float(
            np.max(
                np.concatenate([item.jacobian_condition_number for item in metrics])
            )
        )
    if key == "translational_jacobian_min_manipulability_m3_per_rad3":
        return float(
            np.min(
                np.concatenate([item.jacobian_manipulability for item in metrics])
            )
        )
    raise AnalysisError(f"No pooled definition for metric {key}")


def aggregate_rows(
    analyses: Sequence[CaseAnalysis],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    report: Dict[str, Any] = {
        "metrics": {},
        "paired_aggregation_definition": (
            "All deltas are formed within a stroke as diffusion-prior before "
            "mean, best, or worst paired aggregation. Independently selected "
            "method extrema are reported without subtraction. Equal infinite "
            "condition numbers retain an explicit undefined delta and use zero "
            "only as a deterministic ordering value because their directional "
            "interpretation is unchanged."
        ),
        "pooled_label": "pooled_trajectory_samples",
    }
    for spec in METRIC_SPECS:
        paired_records: List[Dict[str, Any]] = []
        for analysis in analyses:
            prior = analysis.prior.scalar[spec.key]
            diffusion = analysis.diffusion.scalar[spec.key]
            paired_records.append(
                {
                    "case_name": analysis.loaded.name,
                    "prior": prior,
                    "diffusion": diffusion,
                    "delta": scalar_delta(prior, diffusion),
                    "interpretation": interpretation(
                        prior,
                        diffusion,
                        spec.direction,
                    ),
                }
            )

        mean_prior = float(
            np.mean([float(record["prior"]) for record in paired_records])
        )
        mean_diffusion = float(
            np.mean(
                [float(record["diffusion"]) for record in paired_records]
            )
        )
        numeric_paired_deltas = [
            record["delta"]
            for record in paired_records
            if not isinstance(record["delta"], str)
        ]
        mean_paired_delta: float | str
        if len(numeric_paired_deltas) == len(paired_records):
            mean_paired_delta = float(
                np.mean(
                    [
                        float(delta)
                        for delta in numeric_paired_deltas
                    ]
                )
            )
        else:
            mean_paired_delta = "undefined_infinity_minus_infinity"

        worst_paired = select_paired_delta_extreme(
            paired_records,
            spec.direction,
            worst=True,
        )
        best_paired = select_paired_delta_extreme(
            paired_records,
            spec.direction,
            worst=False,
        )
        worst_prior = select_method_extreme(
            paired_records,
            "prior",
            spec.direction,
            worst=True,
        )
        worst_diffusion = select_method_extreme(
            paired_records,
            "diffusion",
            spec.direction,
            worst=True,
        )
        best_prior = select_method_extreme(
            paired_records,
            "prior",
            spec.direction,
            worst=False,
        )
        best_diffusion = select_method_extreme(
            paired_records,
            "diffusion",
            spec.direction,
            worst=False,
        )

        if spec.pooled:
            pooled_prior_value = pooled_metric(
                analyses,
                "prior",
                spec.key,
            )
            pooled_diffusion_value = pooled_metric(
                analyses,
                "diffusion",
                spec.key,
            )
            pooled_prior: float | None = pooled_prior_value
            pooled_diffusion: float | None = pooled_diffusion_value
            pooled_delta: float | str | None = scalar_delta(
                pooled_prior_value,
                pooled_diffusion_value,
            )
            pooled_interpretation: str | None = interpretation(
                pooled_prior_value,
                pooled_diffusion_value,
                spec.direction,
            )
            pooled_status = "pooled_trajectory_samples"
        else:
            pooled_prior = None
            pooled_diffusion = None
            pooled_delta = None
            pooled_interpretation = None
            pooled_status = (
                "not_applicable: this stroke-level endpoint metric has no "
                "valid pooled-sample definition"
            )

        item: Dict[str, Any] = {
            "metric": spec.key,
            "directionality": spec.direction,
            "stroke_count": len(analyses),
            "mean_prior": mean_prior,
            "mean_diffusion": mean_diffusion,
            "mean_paired_delta": mean_paired_delta,
            "mean_interpretation": interpretation(
                mean_prior,
                mean_diffusion,
                spec.direction,
            ),
            "worst_paired_delta": worst_paired["delta"],
            "worst_paired_delta_case_name": worst_paired["case_name"],
            "worst_paired_delta_prior": worst_paired["prior"],
            "worst_paired_delta_diffusion": worst_paired["diffusion"],
            "worst_paired_delta_interpretation": worst_paired[
                "interpretation"
            ],
            "best_paired_delta": best_paired["delta"],
            "best_paired_delta_case_name": best_paired["case_name"],
            "best_paired_delta_prior": best_paired["prior"],
            "best_paired_delta_diffusion": best_paired["diffusion"],
            "best_paired_delta_interpretation": best_paired[
                "interpretation"
            ],
            "worst_prior_value": worst_prior["prior"],
            "worst_prior_case_name": worst_prior["case_name"],
            "worst_diffusion_value": worst_diffusion["diffusion"],
            "worst_diffusion_case_name": worst_diffusion["case_name"],
            "best_prior_value": best_prior["prior"],
            "best_prior_case_name": best_prior["case_name"],
            "best_diffusion_value": best_diffusion["diffusion"],
            "best_diffusion_case_name": best_diffusion["case_name"],
            "pooled_prior": pooled_prior,
            "pooled_diffusion": pooled_diffusion,
            "pooled_delta": pooled_delta,
            "pooled_interpretation": pooled_interpretation,
            "pooled_status": pooled_status,
        }
        rows.append(item)
        report["metrics"][spec.key] = {
            **item,
            "paired_stroke_records": paired_records,
            "method_extrema_note": (
                "Prior and diffusion extrema are selected independently and "
                "are never subtracted or directionally interpreted as a pair."
            ),
        }

    report["joint_difference"] = {
        "unweighted_mean_rms_joint_difference_rad": float(
            np.mean(
                [
                    analysis.difference["rms_joint_difference_rad"]
                    for analysis in analyses
                ]
            )
        ),
        "global_maximum_absolute_joint_difference_rad": float(
            max(
                analysis.difference["maximum_absolute_joint_difference_rad"]
                for analysis in analyses
            )
        ),
        "pooled_rms_joint_difference_rad": rms(
            np.concatenate(
                [analysis.difference_q_rad.reshape(-1) for analysis in analyses]
            )
        ),
        "pooled_mean_absolute_joint_difference_rad": float(
            np.mean(
                np.abs(
                    np.concatenate(
                        [
                            analysis.difference_q_rad.reshape(-1)
                            for analysis in analyses
                        ]
                    )
                )
            )
        ),
        "interpretation": "descriptive",
    }
    return rows, report


def summarize_primary_metric_groups(
    aggregate_report: Mapping[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
    aggregate_metrics = aggregate_report["metrics"]
    rows: List[Dict[str, Any]] = []
    group_summaries: Dict[str, Any] = {}
    counts = {"improved": 0, "worsened": 0, "unchanged": 0}
    for group_name, definition in PRIMARY_METRIC_GROUPS.items():
        primary_metric = str(definition["primary_metric"])
        supporting_metric_names = [
            str(value) for value in definition["supporting_metrics"]
        ]
        primary = aggregate_metrics[primary_metric]
        group_interpretation = str(primary["mean_interpretation"])
        counts[group_interpretation] += 1
        supporting_metrics = [
            {
                "metric": metric,
                "prior": aggregate_metrics[metric]["mean_prior"],
                "diffusion": aggregate_metrics[metric]["mean_diffusion"],
                "delta": aggregate_metrics[metric]["mean_paired_delta"],
                "directionality": aggregate_metrics[metric][
                    "directionality"
                ],
                "interpretation": aggregate_metrics[metric][
                    "mean_interpretation"
                ],
            }
            for metric in supporting_metric_names
        ]
        summary = {
            "group_name": group_name,
            "primary_metric": primary_metric,
            "prior": primary["mean_prior"],
            "diffusion": primary["mean_diffusion"],
            "delta": primary["mean_paired_delta"],
            "directionality": primary["directionality"],
            "interpretation": group_interpretation,
            "supporting_metrics": supporting_metrics,
            "aggregation_basis": (
                "unweighted mean prior and diffusion values across the three "
                "paired strokes; interpretation comes only from the primary metric"
            ),
        }
        group_summaries[group_name] = summary
        rows.append(
            {
                **summary,
                "supporting_metrics": "|".join(supporting_metric_names),
            }
        )

    summary_statement = (
        "This v8.1 prior-ablation analysis found that across seven primary "
        f"metric groups, frozen diffusion v8.1 improved {counts['improved']}, "
        f"worsened {counts['worsened']}, and left {counts['unchanged']} "
        "effectively unchanged. These results characterize the marginal effect "
        "of diffusion on the three accepted deployment strokes. They do not "
        "establish superiority over the project's other trajectory-generation "
        "pipelines."
    )
    conclusion = {
        "primary_groups_improved": counts["improved"],
        "primary_groups_worsened": counts["worsened"],
        "primary_groups_unchanged": counts["unchanged"],
        "primary_group_count": len(PRIMARY_METRIC_GROUPS),
        "summary_statement": summary_statement,
        "limitations": analysis_limitations(),
    }
    return rows, group_summaries, conclusion


def analysis_limitations() -> List[str]:
    return [
        "Only three accepted deployment strokes are analyzed.",
        "Rejected deployment attempts are excluded.",
        "No method-level acceptance-rate comparison is made.",
        "No runtime comparison is made.",
        (
            "No comparison is made against MLP-only generation, sequential IK, "
            "deterministic residual models, v7 or other diffusion versions, "
            "or numerical cost-optimized target pipelines."
        ),
        (
            "These metrics characterize translational kinematic conditioning "
            "only. They do not fully characterize orientation singularities "
            "or full-pose 6D Jacobian conditioning."
        ),
        (
            "Repeated finite differences can amplify endpoint effects. "
            "Full-trajectory derivative metrics include endpoint "
            "finite-difference effects."
        ),
        (
            "Interior-only derivative metrics exclude the first and last "
            f"{DERIVATIVE_INTERIOR_TRIM_SAMPLES} samples."
        ),
        (
            "This v8.1 prior-ablation analysis establishes only the marginal "
            "effect of diffusion relative to its exact strong MLP + adaptive-IK "
            "prior."
        ),
    ]


def atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise AnalysisError(f"Refusing to write empty CSV: {path}")
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value):
            raise AnalysisError("NaN cannot be serialized in a successful report")
        if math.isinf(value):
            return "Infinity" if value > 0.0 else "-Infinity"
    return value


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    text = json.dumps(
        json_safe(value),
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    atomic_write_text(path, text + "\n")


def save_plot(
    path: Path,
    analyses: Sequence[CaseAnalysis],
    value_getter: Any,
    ylabel: str,
    *,
    include_prior: bool,
    include_diffusion: bool,
) -> None:
    figure, axis = plt.subplots(figsize=(12, 5))
    for analysis in analyses:
        if include_prior:
            axis.plot(
                analysis.loaded.timestamps_s,
                value_getter(analysis, "prior"),
                label=f"{analysis.loaded.name} prior",
                linewidth=1.4,
            )
        if include_diffusion:
            axis.plot(
                analysis.loaded.timestamps_s,
                value_getter(analysis, "diffusion"),
                "--",
                label=f"{analysis.loaded.name} diffusion",
                linewidth=1.4,
            )
    axis.set_xlabel("time (s)")
    axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.25)
    axis.legend(ncol=2, fontsize=8)
    figure.tight_layout()
    figure.savefig(str(path), dpi=160)
    plt.close(figure)


def save_interior_derivative_plot(
    path: Path,
    analyses: Sequence[CaseAnalysis],
    derivative_name: str,
    ylabel: str,
) -> None:
    interior_slice = slice(
        DERIVATIVE_INTERIOR_TRIM_SAMPLES,
        -DERIVATIVE_INTERIOR_TRIM_SAMPLES,
    )
    figure, axis = plt.subplots(figsize=(12, 5))
    for analysis in analyses:
        for trajectory_type, metrics, linestyle in (
            ("prior", analysis.prior, "-"),
            ("diffusion", analysis.diffusion, "--"),
        ):
            values = (
                metrics.acceleration_rad_s2
                if derivative_name == "acceleration"
                else metrics.jerk_rad_s3
            )
            axis.plot(
                analysis.loaded.timestamps_s[interior_slice],
                np.max(np.abs(values[interior_slice]), axis=1),
                linestyle,
                label=f"{analysis.loaded.name} {trajectory_type}",
                linewidth=1.4,
            )
    axis.set_xlabel("time (s)")
    axis.set_ylabel(ylabel)
    axis.set_title(
        "Interior-only derivative comparison "
        f"(first/last {DERIVATIVE_INTERIOR_TRIM_SAMPLES} samples excluded)"
    )
    axis.grid(True, alpha=0.25)
    axis.legend(ncol=2, fontsize=8)
    figure.tight_layout()
    figure.savefig(str(path), dpi=160)
    plt.close(figure)


def save_plots(output_dir: Path, analyses: Sequence[CaseAnalysis]) -> None:
    figure, axis = plt.subplots(figsize=(12, 5))
    for analysis in analyses:
        axis.plot(
            analysis.loaded.timestamps_s,
            np.sqrt(np.mean(np.square(analysis.difference_q_rad), axis=1)),
            label=analysis.loaded.name,
            linewidth=1.4,
        )
    axis.set_xlabel("time (s)")
    axis.set_ylabel("joint difference RMS across joints (rad)")
    axis.grid(True, alpha=0.25)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(
        str(output_dir / "joint_difference_over_time.png"),
        dpi=160,
    )
    plt.close(figure)
    save_plot(
        output_dir / "cartesian_error_comparison.png",
        analyses,
        lambda analysis, kind: (
            analysis.prior.position_error_m
            if kind == "prior"
            else analysis.diffusion.position_error_m
        ),
        "Cartesian position error (m)",
        include_prior=True,
        include_diffusion=True,
    )
    save_plot(
        output_dir / "acceleration_comparison.png",
        analyses,
        lambda analysis, kind: np.max(
            np.abs(
                analysis.prior.acceleration_rad_s2
                if kind == "prior"
                else analysis.diffusion.acceleration_rad_s2
            ),
            axis=1,
        ),
        "maximum absolute joint acceleration (rad/s²)",
        include_prior=True,
        include_diffusion=True,
    )
    save_plot(
        output_dir / "jerk_comparison.png",
        analyses,
        lambda analysis, kind: np.max(
            np.abs(
                analysis.prior.jerk_rad_s3
                if kind == "prior"
                else analysis.diffusion.jerk_rad_s3
            ),
            axis=1,
        ),
        "maximum absolute joint jerk (rad/s³)",
        include_prior=True,
        include_diffusion=True,
    )
    save_interior_derivative_plot(
        output_dir / "interior_acceleration_comparison.png",
        analyses,
        "acceleration",
        "interior maximum absolute joint acceleration (rad/s²)",
    )
    save_interior_derivative_plot(
        output_dir / "interior_jerk_comparison.png",
        analyses,
        "jerk",
        "interior maximum absolute joint jerk (rad/s³)",
    )
    save_plot(
        output_dir / "singularity_margin_comparison.png",
        analyses,
        lambda analysis, kind: (
            analysis.prior.jacobian_min_singular_value
            if kind == "prior"
            else analysis.diffusion.jacobian_min_singular_value
        ),
        (
            "translational Jacobian singularity margin: "
            "minimum singular value (m/rad)"
        ),
        include_prior=True,
        include_diffusion=True,
    )
    save_plot(
        output_dir / "joint_limit_margin_comparison.png",
        analyses,
        lambda analysis, kind: np.min(
            (
                analysis.prior.joint_limit_margin_rad
                if kind == "prior"
                else analysis.diffusion.joint_limit_margin_rad
            ),
            axis=1,
        ),
        "minimum joint-limit margin (rad)",
        include_prior=True,
        include_diffusion=True,
    )


def markdown_number(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        if math.isinf(value):
            return "∞" if value > 0.0 else "−∞"
        return f"{value:.6g}"
    return str(value)


def build_markdown_report(
    analyses: Sequence[CaseAnalysis],
    aggregate_report: Mapping[str, Any],
    primary_group_summaries: Mapping[str, Any],
    conclusion: Mapping[str, Any],
) -> str:
    lines = [
        "# Frozen diffusion v8.1 prior-ablation report",
        "",
        "## 1. Objective and scope",
        "",
        (
            "This v8.1 prior-ablation analysis evaluates the marginal effect of "
            "frozen diffusion v8.1 relative to the exact strong MLP + adaptive-IK "
            "prior used to initialize each accepted deployment trajectory. It is "
            "not a new safety approval or a comparison against other generation "
            "pipelines."
        ),
        "",
        "## 2. Artifact provenance and validation",
        "",
    ]
    for analysis in analyses:
        metadata = analysis.loaded.metadata
        lines.extend(
            [
                (
                    f"- `{analysis.loaded.name}`: input "
                    f"`{metadata['deployment_input_npz']}`; output "
                    f"`{metadata['diffusion_output_dir']}`; deployment ID "
                    f"`{metadata['deployment_path_id']}`; accepted verdict "
                    f"`{ACCEPTED_VERDICT}`."
                ),
                (
                    f"  URDF: `{metadata['urdf_path']}` "
                    f"(`{metadata['urdf_sha256']}`); dt="
                    f"{analysis.loaded.timestep_s:.12g} s."
                ),
            ]
        )
    lines.extend(
        [
            "",
            "## 3. Metric definitions",
            "",
            (
                "Derivatives use `numpy.gradient(q, timestamps, edge_order=2)` "
                "recursively for velocity, acceleration, and jerk, matching the "
                "deployment generator and validator. Integrated squared derivative "
                "metrics are `sum(value²) × dt` over samples and joints."
            ),
            "",
            (
                "Full-trajectory derivative metrics include endpoint "
                "finite-difference effects. Interior-only metrics exclude the "
                f"first and last {DERIVATIVE_INTERIOR_TRIM_SAMPLES} samples."
            ),
            "",
            (
                "Orientation error is the repository validator's rotation-matrix "
                "geodesic angle `acos(clip((trace(R_targetᵀR_actual)-1)/2,-1,1))`."
            ),
            "",
            (
                "Joint-limit margin is `min(q-lower, upper-q)` and is negative "
                "outside a limit. Normalized margin divides by the joint's full "
                "permitted range. Violations use the repository hard-limit "
                f"tolerance of {HARD_JOINT_LIMIT_TOLERANCE_RAD:g} rad."
            ),
            "",
            (
                "The Jacobian is the deployment stack's central finite-difference "
                "3×6 translational Jacobian on the recorded kinematic chain, using "
                f"a {JACOBIAN_FINITE_DIFFERENCE_STEP_RAD:g} rad step. Condition "
                "number is infinite when the minimum singular value is no larger "
                "than max(1e-12, max(J.shape)×eps×largest singular value). "
                "Manipulability is the product of the three singular values. These "
                "metrics characterize translational kinematic conditioning only. "
                "They do not fully characterize orientation singularities or "
                "full-pose 6D Jacobian conditioning."
            ),
            "",
            (
                "A scalar delta is diffusion minus prior. Interpretations use "
                f"absolute tolerance {CHANGE_ABSOLUTE_TOLERANCE:g} plus relative "
                f"tolerance {CHANGE_RELATIVE_TOLERANCE:g} times the larger "
                "magnitude. Joint differences are descriptive."
            ),
            "",
            "## 4. Primary metric-group results",
            "",
            "| Group | Primary metric | Prior | Diffusion | Delta | Interpretation |",
            "|---|---|---:|---:|---:|---|",
        ]
    )
    for group_name, summary in primary_group_summaries.items():
        lines.append(
            f"| {group_name} | {summary['primary_metric']} | "
            f"{markdown_number(summary['prior'])} | "
            f"{markdown_number(summary['diffusion'])} | "
            f"{markdown_number(summary['delta'])} | "
            f"{summary['interpretation']} |"
        )

    lines.extend(
        [
            "",
            "## 5. Per-stroke results",
            "",
            "| Stroke | Primary metric | Prior | Diffusion | Delta | Interpretation |",
            "|---|---|---:|---:|---:|---|",
        ]
    )
    for analysis in analyses:
        for definition in PRIMARY_METRIC_GROUPS.values():
            metric = str(definition["primary_metric"])
            item = analysis.comparison[metric]
            lines.append(
                f"| {analysis.loaded.name} | {metric} | "
                f"{markdown_number(item['prior'])} | "
                f"{markdown_number(item['diffusion'])} | "
                f"{markdown_number(item['delta'])} | "
                f"{item['interpretation']} |"
            )

    lines.extend(
        [
            "",
            "## 6. Paired aggregate results",
            "",
            (
                "Best and worst deltas select paired prior/diffusion values from "
                "the same stroke. Prior and diffusion extrema are listed "
                "separately and are never subtracted."
            ),
            "",
            (
                "| Metric | Mean paired delta | Best paired delta (case) | "
                "Worst paired delta (case) | Prior extrema (best / worst) | "
                "Diffusion extrema (best / worst) |"
            ),
            "|---|---:|---:|---:|---|---|",
        ]
    )
    aggregate_metrics = aggregate_report["metrics"]
    for definition in PRIMARY_METRIC_GROUPS.values():
        metric = str(definition["primary_metric"])
        item = aggregate_metrics[metric]
        lines.append(
            f"| {metric} | {markdown_number(item['mean_paired_delta'])} | "
            f"{markdown_number(item['best_paired_delta'])} "
            f"({item['best_paired_delta_case_name']}) | "
            f"{markdown_number(item['worst_paired_delta'])} "
            f"({item['worst_paired_delta_case_name']}) | "
            f"{markdown_number(item['best_prior_value'])} "
            f"({item['best_prior_case_name']}) / "
            f"{markdown_number(item['worst_prior_value'])} "
            f"({item['worst_prior_case_name']}) | "
            f"{markdown_number(item['best_diffusion_value'])} "
            f"({item['best_diffusion_case_name']}) / "
            f"{markdown_number(item['worst_diffusion_value'])} "
            f"({item['worst_diffusion_case_name']}) |"
        )

    lines.extend(
        [
            "",
            "## 7. Correction magnitude introduced by diffusion",
            "",
        ]
    )
    for analysis in sorted(
        analyses,
        key=lambda item: item.difference[
            "maximum_absolute_joint_difference_rad"
        ],
        reverse=True,
    ):
        difference = analysis.difference
        lines.append(
            f"- `{analysis.loaded.name}`: RMS difference "
            f"{difference['rms_joint_difference_rad']:.6g} rad; maximum "
            f"{difference['maximum_absolute_joint_difference_rad']:.6g} rad at "
            f"sample {difference['global_maximum_sample_index']}, "
            f"{difference['global_maximum_joint_name']}."
        )

    categorized: Dict[str, List[str]] = {
        "improved": [],
        "worsened": [],
        "unchanged": [],
    }
    for key, item in aggregate_metrics.items():
        categorized[str(item["mean_interpretation"])].append(
            f"`{key}` (mean paired delta "
            f"{markdown_number(item['mean_paired_delta'])})"
        )
    for number, category, title in (
        (8, "improved", "Metrics improved"),
        (9, "worsened", "Metrics worsened"),
        (10, "unchanged", "Metrics effectively unchanged"),
    ):
        lines.extend(["", f"## {number}. {title}", ""])
        if categorized[category]:
            lines.extend(f"- {item}" for item in categorized[category])
        else:
            lines.append("- None under the documented tolerance.")

    lines.extend(
        [
            "",
            "## 11. Limitations",
            "",
        ]
    )
    lines.extend(f"- {limitation}" for limitation in conclusion["limitations"])
    lines.extend(
        [
            "",
            "## 12. Evidence-based conclusion",
            "",
            str(conclusion["summary_statement"]),
            "",
        ]
    )
    return "\n".join(lines)


def trajectory_json(metrics: TrajectoryMetrics) -> Dict[str, Any]:
    return {
        "scalar_metrics": metrics.scalar,
        "extreme_sample_indices": metrics.indices,
        "per_joint_metrics": metrics.per_joint,
        "per_sample": {
            "cartesian_position_error_m": metrics.position_error_m,
            "orientation_error_rad": metrics.orientation_error_rad,
            "minimum_joint_limit_margin_rad": np.min(
                metrics.joint_limit_margin_rad,
                axis=1,
            ),
            "minimum_normalized_joint_limit_margin": np.min(
                metrics.normalized_joint_limit_margin,
                axis=1,
            ),
            "joint_limit_violation_count": np.count_nonzero(
                metrics.joint_limit_violation,
                axis=1,
            ),
            "translational_jacobian_min_singular_value_m_per_rad": (
                metrics.jacobian_min_singular_value
            ),
            "translational_jacobian_condition_number": (
                metrics.jacobian_condition_number
            ),
            "translational_jacobian_manipulability_m3_per_rad3": (
                metrics.jacobian_manipulability
            ),
        },
    }


def prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = [output_dir / name for name in OUTPUT_NAMES if (output_dir / name).exists()]
    if existing and not overwrite:
        raise AnalysisError(
            "Analysis outputs already exist; pass --overwrite to replace them: "
            + ", ".join(str(path) for path in existing)
        )
    if overwrite:
        for path in existing:
            if not path.is_file() and not path.is_symlink():
                raise AnalysisError(f"Refusing to replace non-file output: {path}")
            path.unlink()
    failure_path = output_dir / "contribution_analysis_failure.json"
    if failure_path.exists():
        if not overwrite:
            raise AnalysisError(
                f"Failure report exists; pass --overwrite to replace it: {failure_path}"
            )
        failure_path.unlink()


def remove_success_outputs(output_dir: Path) -> None:
    for name in OUTPUT_NAMES:
        path = output_dir / name
        if path.is_file() or path.is_symlink():
            path.unlink()


def validate_case_arguments(raw_cases: Sequence[Sequence[str]]) -> None:
    names = [case[0] for case in raw_cases]
    if len(names) != len(set(names)):
        raise AnalysisError("--case names must be unique")
    actual = set(names)
    if actual != REQUIRED_CASE_NAMES:
        raise AnalysisError(
            "Exactly the three scoped cases are required; "
            f"missing={sorted(REQUIRED_CASE_NAMES - actual)}, "
            f"unexpected={sorted(actual - REQUIRED_CASE_NAMES)}"
        )


def run(args: argparse.Namespace) -> None:
    output_dir = args.output_dir.expanduser().resolve()
    prepare_output_dir(output_dir, args.overwrite)
    validate_case_arguments(args.cases)

    loaded_cases = [
        load_case(case_name, input_npz, diffusion_output_dir)
        for case_name, input_npz, diffusion_output_dir in args.cases
    ]
    reference_order = loaded_cases[0].joint_order
    reference_urdf_hash = loaded_cases[0].metadata["urdf_sha256"]
    reference_frame = loaded_cases[0].metadata["end_effector_frame"]
    for loaded in loaded_cases[1:]:
        if loaded.joint_order != reference_order:
            raise AnalysisError("Joint order differs between accepted cases")
        if loaded.metadata["urdf_sha256"] != reference_urdf_hash:
            raise AnalysisError("Recorded URDF differs between accepted cases")
        if loaded.metadata["end_effector_frame"] != reference_frame:
            raise AnalysisError("End-effector frame differs between accepted cases")

    analyses = [analyze_case(loaded) for loaded in loaded_cases]
    stroke_rows = per_stroke_rows(analyses)
    joint_rows = per_joint_rows(analyses)
    sample_rows = per_sample_rows(analyses)
    aggregate_csv_rows, aggregate_report = aggregate_rows(analyses)
    (
        primary_group_rows,
        primary_group_summaries,
        conclusion,
    ) = summarize_primary_metric_groups(aggregate_report)

    report: Dict[str, Any] = {
        "generation_status": "passed",
        "analysis_type": ANALYSIS_TYPE,
        "analysis_objective": (
            "This v8.1 prior-ablation analysis evaluates the marginal effect "
            "of frozen diffusion v8.1 relative to the exact strong MLP + "
            "adaptive-IK prior used to initialize each accepted deployment "
            "trajectory."
        ),
        "scope_exclusions": list(SCOPE_EXCLUSIONS),
        "safety_scope": (
            "Prior-ablation measurement only; accepted deployment artifacts "
            "are validated for consistency, but this is not an independent "
            "safety approval."
        ),
        "artifact_keys_used": {
            "prior_joint_trajectory": "deployment input NPZ:strong_prior_q",
            "accepted_diffusion_trajectory": (
                "deployment_trajectory_full.npz:final_q"
            ),
            "accepted_diffusion_cross_check": (
                "approved_simulation_trajectory.npz:q"
            ),
            "desired_cartesian_position": "deployment input NPZ:desired_path",
            "desired_cartesian_orientation": (
                "deployment input NPZ:target_rotation_matrix"
            ),
            "timestamps": "deployment input NPZ:timestamps",
        },
        "joint_order": list(reference_order),
        "joint_limits_rad": {
            joint_name: {
                "lower": float(loaded_cases[0].lower_limits_rad[joint_index]),
                "upper": float(loaded_cases[0].upper_limits_rad[joint_index]),
                "full_range": float(
                    loaded_cases[0].upper_limits_rad[joint_index]
                    - loaded_cases[0].lower_limits_rad[joint_index]
                ),
            }
            for joint_index, joint_name in enumerate(reference_order)
        },
        "joint_limit_source": (
            "Recorded hash-verified URDF loaded through "
            "evaluate_diffusion_v7_teacher_forced_validation.make_robot_context, "
            "which uses generate_ik_seed_path.get_joint_bounds in authoritative "
            "joint order."
        ),
        "end_effector_frame": reference_frame,
        "timestep": {
            loaded.name: loaded.timestep_s for loaded in loaded_cases
        },
        "derivative_method": (
            "numpy.gradient with the verified timestamps, axis=0, edge_order=2; "
            "applied recursively for velocity, acceleration, and jerk. "
            "Full-trajectory derivative metrics include endpoint "
            "finite-difference effects. Interior-only metrics exclude the "
            f"first and last {DERIVATIVE_INTERIOR_TRIM_SAMPLES} samples."
        ),
        "time_integral_method": (
            "sum of squared sample values over time and joints multiplied by "
            "the verified uniform timestep"
        ),
        "orientation_error_definition": (
            "Repository deployment-validator geodesic rotation angle: "
            "acos(clip((trace(R_target.T @ R_actual)-1)/2,-1,1))"
        ),
        "joint_limit_margin_definition": (
            "min(q-lower, upper-q); negative outside a hard limit. Normalized "
            "margin divides by the joint's full permitted range."
        ),
        "jacobian_definition": {
            "reported": (
                "Translational Jacobian singularity margin from the 3x6 "
                "translational finite-difference Jacobian in "
                "generate_diffusion_v7_cost_improving_residual_targets."
                "positional_jacobian on the accepted deployment robot context"
            ),
            "finite_difference_step_rad": JACOBIAN_FINITE_DIFFERENCE_STEP_RAD,
            "full_6d_reported": False,
            "full_6d_reason": (
                "The inspected deployment generator and validator provide an "
                "authoritative translational Jacobian but no authoritative full "
                "6D Jacobian; no conflicting implementation was invented."
            ),
            "condition_number_definition": (
                "largest singular value / smallest singular value; Infinity "
                "when the smallest is at or below the documented SVD tolerance"
            ),
            "manipulability_definition": (
                "Product of the three translational Jacobian singular values, "
                "equivalent to sqrt(det(J J.T))"
            ),
            "scope_warning": (
                "These metrics characterize translational kinematic "
                "conditioning only. They do not fully characterize orientation "
                "singularities or full-pose 6D Jacobian conditioning."
            ),
        },
        "numerical_tolerances": {
            "change_absolute": CHANGE_ABSOLUTE_TOLERANCE,
            "change_relative": CHANGE_RELATIVE_TOLERANCE,
            "timestamp_absolute_s": TIMESTAMP_ABSOLUTE_TOLERANCE_S,
            "timestamp_relative": TIMESTAMP_RELATIVE_TOLERANCE,
            "hard_joint_limit_rad": HARD_JOINT_LIMIT_TOLERANCE_RAD,
            "derivative_interior_trim_samples_each_endpoint": (
                DERIVATIVE_INTERIOR_TRIM_SAMPLES
            ),
            "svd_absolute": SVD_ABSOLUTE_TOLERANCE,
            "svd_effective": (
                "max(svd_absolute, max(J.shape) * float64_eps * largest_singular_value)"
            ),
            "infinite_json_encoding": (
                "Infinite condition numbers are encoded as the explicit string "
                "'Infinity'; NaN is never silently replaced."
            ),
        },
        "metric_directionality": {
            spec.key: spec.direction for spec in METRIC_SPECS
        },
        "source_paths": [analysis.loaded.metadata for analysis in analyses],
        "validation_results": {
            analysis.loaded.name: {
                **analysis.loaded.validation,
                "input_output_prior_exact_match": True,
                "desired_path_exact_match": True,
                "timestamps_exact_match": True,
                "approved_final_q_exact_match": True,
                "joint_order_match": True,
                "urdf_hash_match": True,
                "provenance_consistent": True,
            }
            for analysis in analyses
        },
        "per_stroke_results": {
            analysis.loaded.name: {
                "prior": trajectory_json(analysis.prior),
                "diffusion": trajectory_json(analysis.diffusion),
                "paired_comparison_scalar_metrics": analysis.comparison,
                "correction_metrics": analysis.difference,
                "joint_order_sources": analysis.loaded.joint_order_sources,
                "timestep_s": analysis.loaded.timestep_s,
            }
            for analysis in analyses
        },
        "aggregate_results": aggregate_report,
        "primary_metric_groups": primary_group_summaries,
        "conclusion": conclusion,
        "warnings": [
            (
                "These metrics characterize translational kinematic "
                "conditioning only. They do not fully characterize orientation "
                "singularities or full-pose 6D Jacobian conditioning."
            ),
            (
                "Correction rows describe correction_q=diffusion_q-prior_q and "
                "its derivatives. Comparison rows contain paired scalar metric "
                "differences computed as diffusion metric minus prior metric. "
                "Derivative-of-correction values are never presented as paired "
                "metric comparisons."
            )
        ],
        "provenance": {
            "analysis_script": str(Path(__file__).resolve()),
            "analysis_script_sha256": sha256_file(Path(__file__).resolve()),
            "repository_utility_modules": {
                "deployment_validator": str(
                    Path(deployment_validator.__file__).resolve()
                ),
                "robot_context": str(Path(v7_evaluator.__file__).resolve()),
                "jacobian": str(Path(target_generator.__file__).resolve()),
            },
            "deterministic": True,
            "input_artifacts_read_only": True,
        },
    }

    atomic_write_csv(output_dir / "per_stroke_summary.csv", stroke_rows)
    atomic_write_csv(output_dir / "per_joint_summary.csv", joint_rows)
    atomic_write_csv(output_dir / "per_sample_metrics.csv", sample_rows)
    atomic_write_csv(output_dir / "aggregate_summary.csv", aggregate_csv_rows)
    atomic_write_csv(
        output_dir / "primary_metric_group_summary.csv",
        primary_group_rows,
    )
    save_plots(output_dir, analyses)
    atomic_write_json(output_dir / "contribution_report.json", report)
    atomic_write_text(
        output_dir / "contribution_report.md",
        build_markdown_report(
            analyses,
            aggregate_report,
            primary_group_summaries,
            conclusion,
        ),
    )


def main(argv: Sequence[str] | None = None) -> int:
    args: argparse.Namespace | None = None
    output_dir: Path | None = None
    success_outputs_preexisting = False
    failure_report_preexisting = False
    try:
        args = parse_args(argv)
        resolved_output_dir = Path(args.output_dir).expanduser().resolve()
        output_dir = resolved_output_dir
        success_outputs_preexisting = any(
            (resolved_output_dir / name).exists() for name in OUTPUT_NAMES
        )
        failure_report_preexisting = (
            resolved_output_dir / "contribution_analysis_failure.json"
        ).exists()
        run(args)
    except Exception as exc:
        if output_dir is None and args is not None:
            output_dir = Path(args.output_dir).expanduser().resolve()
        if output_dir is not None:
            try:
                output_dir.mkdir(parents=True, exist_ok=True)
                if not success_outputs_preexisting or (
                    args is not None and args.overwrite
                ):
                    remove_success_outputs(output_dir)
                if not failure_report_preexisting or (
                    args is not None and args.overwrite
                ):
                    atomic_write_json(
                        output_dir / "contribution_analysis_failure.json",
                        {
                            "generation_status": "failed",
                            "error_type": type(exc).__name__,
                            "error": str(exc),
                        },
                    )
            except Exception as failure_write_exc:
                print(
                    f"CONTRIBUTION_ANALYSIS_FAILED: {exc}; "
                    f"could not write failure report: {failure_write_exc}",
                    file=sys.stderr,
                )
                return 1
        print(f"CONTRIBUTION_ANALYSIS_FAILED: {exc}", file=sys.stderr)
        return 1
    print("PRIOR_VS_DIFFUSION_CONTRIBUTION_ANALYSIS_PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
