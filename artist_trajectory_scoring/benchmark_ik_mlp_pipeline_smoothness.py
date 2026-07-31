#!/usr/bin/env python3
"""Thesis-ready multi-path, multi-seed trajectory smoothness benchmark.

Only existing trajectory artifacts are read.  No generation, inference,
candidate scoring, rollout, training, or robot execution is performed.
The statistical unit is one complete Cartesian path.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import os
import re
import shlex
import subprocess
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib
import numpy as np
from matplotlib.markers import MarkerStyle
from scipy import stats
from scipy.interpolate import CubicSpline

_INITIAL_BACKEND = str(matplotlib.get_backend())
comparison = importlib.import_module("compare_ik_mlp_pipeline_jerk_over_time")

METHODS = ("ik", "mlp", "pipeline")
METHOD_LABELS = {"ik": "IK", "mlp": "MLP", "pipeline": "Proposed pipeline"}
JOINT_COUNT = 6
DEFAULT_MEAN_CARTESIAN_THRESHOLD_SOURCE = (
    "generate_deployment_input_from_cartesian_csv."
    "MAXIMUM_ALLOWED_MEAN_ERROR_GATE_M"
)
DEFAULT_MAX_CARTESIAN_THRESHOLD_SOURCE = (
    "rerank_diffusion_candidates.ACCEPTANCE_MAX_ERROR"
)
DEFAULT_ORIENTATION_THRESHOLD_RAD = 0.05
DEFAULT_ORIENTATION_THRESHOLD_SOURCE = (
    "validate_diffusion_v8_1_deployment_output."
    "MAXIMUM_ALLOWED_ORIENTATION_ERROR_GATE_RAD"
)
BOOTSTRAP_SEED = 20260728
BOOTSTRAP_ITERATIONS = 4000
TIE_RTOL = 1.0e-9
TIE_ATOL = 1.0e-12
PRIMARY_METRICS = (
    "rms_acceleration_rad_s2",
    "rms_jerk_rad_s3",
    "integrated_squared_jerk_rad2_s5",
    "max_abs_jerk_rad_s3",
    "mean_high_frequency_energy_ratio",
    "boundary_jerk_energy_enrichment",
)
SUMMARY_METRICS = (
    "rms_velocity_rad_s",
    "max_abs_velocity_rad_s",
    "integrated_squared_velocity_rad2_s",
    "mean_abs_acceleration_rad_s2",
    "rms_acceleration_rad_s2",
    "interior_mean_abs_acceleration_rad_s2",
    "interior_rms_acceleration_rad_s2",
    "max_abs_acceleration_rad_s2",
    "interior_max_abs_acceleration_rad_s2",
    "integrated_absolute_acceleration_rad_s",
    "integrated_squared_acceleration_rad2_s3",
    "interior_integrated_absolute_acceleration_rad_s",
    "interior_integrated_squared_acceleration_rad2_s3",
    "mean_abs_jerk_rad_s3",
    "median_abs_jerk_rad_s3",
    "rms_jerk_rad_s3",
    "interior_mean_abs_jerk_rad_s3",
    "interior_median_abs_jerk_rad_s3",
    "interior_rms_jerk_rad_s3",
    "max_abs_jerk_rad_s3",
    "interior_max_abs_jerk_rad_s3",
    "integrated_absolute_jerk_rad_s2",
    "integrated_squared_jerk_rad2_s5",
    "interior_integrated_absolute_jerk_rad_s2",
    "interior_integrated_squared_jerk_rad2_s5",
    "endpoint_jerk_energy_fraction",
    "position_total_variation_rad",
    "acceleration_total_variation_rad_s2",
    "max_joint_step_l2_rad",
    "max_velocity_step_l2_rad_s",
    "max_acceleration_step_l2_rad_s2",
    "boundary_jerk_energy_fraction",
    "boundary_sample_fraction",
    "boundary_time_fraction",
    "boundary_jerk_energy_enrichment",
    "boundary_jerk_energy_density_rad2_s6",
    "nonboundary_jerk_energy_density_rad2_s6",
    "boundary_to_nonboundary_energy_density_ratio",
    "boundary_to_nonboundary_jerk_ratio",
    "mean_high_frequency_energy_ratio",
    "max_high_frequency_energy_ratio",
    "dominant_frequency_hz",
    "spectral_centroid_hz",
    "mean_cartesian_error_m",
    "rms_cartesian_error_m",
    "max_cartesian_error_m",
)
PATH_LEVEL_ALIASES = {
    "rms_velocity": "rms_velocity_rad_s",
    "max_abs_velocity": "max_abs_velocity_rad_s",
    "rms_acceleration": "rms_acceleration_rad_s2",
    "interior_rms_acceleration": "interior_rms_acceleration_rad_s2",
    "max_abs_acceleration": "max_abs_acceleration_rad_s2",
    "interior_max_abs_acceleration": "interior_max_abs_acceleration_rad_s2",
    "integrated_squared_acceleration": "integrated_squared_acceleration_rad2_s3",
    "mean_abs_jerk": "mean_abs_jerk_rad_s3",
    "rms_jerk": "rms_jerk_rad_s3",
    "interior_rms_jerk": "interior_rms_jerk_rad_s3",
    "max_abs_jerk": "max_abs_jerk_rad_s3",
    "interior_max_abs_jerk": "interior_max_abs_jerk_rad_s3",
    "integrated_squared_jerk": "integrated_squared_jerk_rad2_s5",
    "position_total_variation": "position_total_variation_rad",
    "acceleration_total_variation": "acceleration_total_variation_rad_s2",
    "max_joint_step": "max_joint_step_l2_rad",
    "max_velocity_step": "max_velocity_step_l2_rad_s",
    "max_acceleration_step": "max_acceleration_step_l2_rad_s2",
}
REQUIRED_FIGURE_STEMS = (
    "rms_jerk_by_method",
    "integrated_squared_jerk_by_method",
    "rms_acceleration_by_method",
    "full_vs_interior_rms_jerk",
    "boundary_jerk_energy_ratio",
    "boundary_vs_nonboundary_jerk",
    "boundary_energy_enrichment_by_radius",
    "high_frequency_energy_ratio",
    "cartesian_error_vs_rms_jerk",
    "pipeline_seed_rms_jerk_variability",
    "smoothness_win_rates",
)


class BenchmarkError(RuntimeError):
    """The requested benchmark cannot be completed faithfully."""


@dataclass(frozen=True)
class PipelineArtifact:
    """One saved pipeline trajectory plus its seed-level acceptance evidence."""

    path_id: str
    seed: int
    path: Path
    accepted: bool
    acceptance_source: str
    runtime_s: float


@dataclass
class EvaluatedTrajectory:
    """Metrics and aligned arrays retained for aggregation and plots."""

    row: dict[str, Any]
    q: np.ndarray
    timestamps: np.ndarray
    velocity: np.ndarray
    acceleration: np.ndarray
    jerk: np.ndarray


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the benchmark command-line interface."""
    parser = argparse.ArgumentParser(
        description="Benchmark saved IK, MLP, and pipeline trajectory smoothness."
    )
    parser.add_argument("--ik_root", type=Path, required=True)
    parser.add_argument("--mlp_root", type=Path, required=True)
    parser.add_argument("--pipeline_root", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--path_ids", nargs="*")
    parser.add_argument("--path_list_file", type=Path)
    parser.add_argument("--path_glob", default="path_*")
    parser.add_argument("--pipeline_seed_glob", default="*seed*")
    parser.add_argument("--pipeline_seeds", nargs="*", type=int)
    parser.add_argument("--execution_horizon", type=int, default=8)
    parser.add_argument("--boundary_radius", type=int, default=2)
    parser.add_argument(
        "--boundary_sensitivity_radii",
        nargs="+",
        type=int,
        default=[0, 1, 2],
    )
    parser.add_argument("--endpoint_exclusion", type=int, default=3)
    parser.add_argument("--common_duration_s", type=float, default=10.0)
    parser.add_argument("--common_samples", type=int, default=100)
    parser.add_argument(
        "--timing_policy",
        choices=("require_equal", "common_duration", "shared_interval_diagnostic"),
        default="common_duration",
    )
    parser.add_argument(
        "--cartesian_error_threshold_m",
        type=float,
        help="Deprecated alias that sets both Cartesian tracking thresholds.",
    )
    parser.add_argument("--mean_cartesian_error_threshold_m", type=float)
    parser.add_argument("--max_cartesian_error_threshold_m", type=float)
    parser.add_argument("--orientation_error_threshold_rad", type=float)
    parser.add_argument(
        "--require_accepted_pipeline",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--minimum_pipeline_seeds", type=int, default=1)
    parser.add_argument(
        "--spectral_signal",
        choices=("position", "velocity", "acceleration"),
        default="acceleration",
    )
    parser.add_argument("--high_frequency_fraction", type=float, default=0.25)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    """Validate policy and numerical arguments before discovery."""
    if args.common_duration_s <= 0.0 or not math.isfinite(args.common_duration_s):
        raise BenchmarkError("--common_duration_s must be finite and positive")
    if args.common_samples < 4:
        raise BenchmarkError("--common_samples must be at least 4")
    if args.execution_horizon <= 0:
        raise BenchmarkError("--execution_horizon must be positive")
    if args.boundary_radius < 0:
        raise BenchmarkError("--boundary_radius cannot be negative")
    if any(radius < 0 for radius in args.boundary_sensitivity_radii):
        raise BenchmarkError(
            "--boundary_sensitivity_radii values must be nonnegative integers"
        )
    args.boundary_sensitivity_radii = sorted(
        set(args.boundary_sensitivity_radii) | {args.boundary_radius}
    )
    if args.endpoint_exclusion < 0:
        raise BenchmarkError("--endpoint_exclusion cannot be negative")
    if args.endpoint_exclusion * 2 >= args.common_samples - 1:
        raise BenchmarkError("--endpoint_exclusion leaves too few interior samples")
    if args.minimum_pipeline_seeds < 1:
        raise BenchmarkError("--minimum_pipeline_seeds must be positive")
    if not 0.0 < args.high_frequency_fraction <= 1.0:
        raise BenchmarkError("--high_frequency_fraction must be in (0,1]")
    if args.timing_policy == "shared_interval_diagnostic":
        raise BenchmarkError(
            "shared_interval_diagnostic is claim_eligible=false and cannot produce "
            "primary benchmark rankings"
        )
    if args.cartesian_error_threshold_m is not None and (
        args.mean_cartesian_error_threshold_m is not None
        or args.max_cartesian_error_threshold_m is not None
    ):
        raise BenchmarkError(
            "--cartesian_error_threshold_m is deprecated and cannot be combined "
            "with either separate Cartesian threshold"
        )
    for name in (
        "cartesian_error_threshold_m",
        "mean_cartesian_error_threshold_m",
        "max_cartesian_error_threshold_m",
        "orientation_error_threshold_rad",
    ):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value <= 0.0):
            raise BenchmarkError(f"--{name} must be finite and positive")


def authoritative_mean_cartesian_threshold() -> tuple[float, str]:
    """Load the repository's accepted mean Cartesian-error gate."""
    try:
        module = importlib.import_module(
            "generate_deployment_input_from_cartesian_csv"
        )
        value = float(module.MAXIMUM_ALLOWED_MEAN_ERROR_GATE_M)
    except (AttributeError, ImportError, TypeError, ValueError) as exc:
        raise BenchmarkError(
            "No defensible authoritative mean Cartesian-error gate was found; "
            "supply --mean_cartesian_error_threshold_m"
        ) from exc
    if not math.isfinite(value) or value <= 0.0:
        raise BenchmarkError(
            "Authoritative mean Cartesian-error gate is invalid; supply "
            "--mean_cartesian_error_threshold_m"
        )
    return value, DEFAULT_MEAN_CARTESIAN_THRESHOLD_SOURCE


def authoritative_max_cartesian_threshold() -> tuple[float, str]:
    """Load the repository's accepted maximum Cartesian-error gate."""
    try:
        module = importlib.import_module("rerank_diffusion_candidates")
        value = float(module.ACCEPTANCE_MAX_ERROR)
    except (AttributeError, ImportError, TypeError, ValueError) as exc:
        raise BenchmarkError(
            "No defensible authoritative maximum Cartesian-error gate was found; "
            "supply --max_cartesian_error_threshold_m"
        ) from exc
    if not math.isfinite(value) or value <= 0.0:
        raise BenchmarkError(
            "Authoritative maximum Cartesian-error gate is invalid; supply "
            "--max_cartesian_error_threshold_m"
        )
    return value, DEFAULT_MAX_CARTESIAN_THRESHOLD_SOURCE


def resolve_cartesian_thresholds(
    args: argparse.Namespace,
) -> tuple[str, str]:
    """Resolve separate gates, preserving the deprecated equal-gate alias."""
    if args.cartesian_error_threshold_m is not None and (
        args.mean_cartesian_error_threshold_m is not None
        or args.max_cartesian_error_threshold_m is not None
    ):
        raise BenchmarkError(
            "--cartesian_error_threshold_m is deprecated and cannot be combined "
            "with either separate Cartesian threshold"
        )
    if args.cartesian_error_threshold_m is not None:
        value = float(args.cartesian_error_threshold_m)
        args.mean_cartesian_error_threshold_m = value
        args.max_cartesian_error_threshold_m = value
        return "deprecated_alias:--cartesian_error_threshold_m", (
            "deprecated_alias:--cartesian_error_threshold_m"
        )
    if args.mean_cartesian_error_threshold_m is None:
        (
            args.mean_cartesian_error_threshold_m,
            mean_source,
        ) = authoritative_mean_cartesian_threshold()
    else:
        mean_source = "command_line:--mean_cartesian_error_threshold_m"
    if args.max_cartesian_error_threshold_m is None:
        (
            args.max_cartesian_error_threshold_m,
            max_source,
        ) = authoritative_max_cartesian_threshold()
    else:
        max_source = "command_line:--max_cartesian_error_threshold_m"
    return mean_source, max_source


def atomic_write_text(path: Path, text: str) -> None:
    """Atomically write one UTF-8 text artifact."""
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Atomically write union-column CSV rows in deterministic key order."""
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    if not columns:
        columns = ["empty"]
        rows = [{"empty": ""}]
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def prepare_output(path: Path, overwrite: bool) -> Path:
    """Protect existing nonempty benchmark directories."""
    output = path.expanduser().resolve()
    if output.exists() and not output.is_dir():
        raise BenchmarkError(f"Output path is not a directory: {output}")
    if output.is_dir() and any(output.iterdir()) and not overwrite:
        raise BenchmarkError(
            f"Output directory is nonempty: {output}; pass --overwrite"
        )
    output.mkdir(parents=True, exist_ok=True)
    return output


def path_id_from_text(value: str) -> str | None:
    """Extract canonical ``path_XXXX`` identity."""
    match = re.search(r"path[_-]?(\d+)", value.lower())
    return f"path_{int(match.group(1)):04d}" if match else None


def read_path_list(path: Path) -> list[str]:
    """Read path IDs from plain text or any CSV field."""
    if not path.is_file():
        raise BenchmarkError(f"Path list does not exist: {path}")
    ids: list[str] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle):
            for value in row:
                identity = path_id_from_text(value)
                if identity:
                    ids.append(identity)
                    break
    return ids


def discover_requested_paths(args: argparse.Namespace) -> list[str]:
    """Resolve explicit path IDs or infer their intersection from input roots."""
    explicit: list[str] = []
    for value in args.path_ids or []:
        identity = path_id_from_text(value)
        if not identity:
            raise BenchmarkError(f"Invalid path ID: {value}")
        explicit.append(identity)
    if args.path_list_file:
        explicit.extend(read_path_list(args.path_list_file))
    if explicit:
        return sorted(set(explicit))
    ik = {
        identity
        for path in args.ik_root.glob(args.path_glob)
        if (identity := path_id_from_text(path.name))
    }
    mlp = {
        identity
        for path in args.mlp_root.glob(args.path_glob)
        if (identity := path_id_from_text(path.name))
    }
    return sorted(ik & mlp)


def read_metrics_index(seed_dir: Path) -> dict[tuple[str, int], dict[str, str]]:
    """Index full-path acceptance rows for one saved rollout seed."""
    path = seed_dir / "anchored_full_path_metrics.csv"
    if not path.is_file():
        return {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result: dict[tuple[str, int], dict[str, str]] = {}
    for row in rows:
        identity = path_id_from_text(str(row.get("path_id", "")))
        try:
            k = int(float(row.get("k", "")))
        except (TypeError, ValueError):
            continue
        if identity:
            result[(identity, k)] = row
    return result


def discover_pipeline_artifacts(
    root: Path,
    seed_glob: str,
    selected_seeds: Sequence[int] | None,
) -> dict[str, list[PipelineArtifact]]:
    """Discover accepted/final deployment and K=8 rollout artifacts."""
    resolved = root.expanduser().resolve()
    if not resolved.is_dir():
        raise BenchmarkError(f"Pipeline root does not exist: {resolved}")
    seed_filter = set(selected_seeds or [])
    discovered: dict[str, list[PipelineArtifact]] = defaultdict(list)
    seed_dirs = [path for path in resolved.glob(seed_glob) if path.is_dir()]
    if not seed_dirs:
        seed_dirs = [resolved]
    for seed_dir in sorted(seed_dirs):
        seed_match = re.search(r"seed[_-]?(\d+)", str(seed_dir))
        seed = int(seed_match.group(1)) if seed_match else 0
        if seed_filter and seed not in seed_filter:
            continue
        index = read_metrics_index(seed_dir)
        for artifact in sorted(seed_dir.glob("trajectories/*/anchored_rollout_k8.npz")):
            identity = path_id_from_text(artifact.parent.name)
            if not identity:
                continue
            row = index.get((identity, 8))
            accepted = bool(
                row
                and int(float(row.get("full_path_safety_pass", "0"))) == 1
                and not str(row.get("failed_hard_safety_gate", "")).strip()
            )
            runtime = math.nan
            if row and row.get("runtime_s"):
                try:
                    runtime = float(row["runtime_s"])
                except ValueError:
                    pass
            discovered[identity].append(
                PipelineArtifact(
                    identity,
                    seed,
                    artifact.resolve(),
                    accepted,
                    str(seed_dir / "anchored_full_path_metrics.csv"),
                    runtime,
                )
            )
        for artifact in sorted(seed_dir.rglob("deployment_trajectory_full.npz")):
            identity = path_id_from_text(str(artifact.parent))
            if not identity:
                continue
            try:
                loaded = comparison.load_trajectory(artifact, "pipeline")
                accepted = "recorded verdict" in str(
                    loaded.metadata.get("pipeline_selection_evidence", "")
                )
            except Exception:
                accepted = False
            discovered[identity].append(
                PipelineArtifact(
                    identity,
                    seed,
                    artifact.resolve(),
                    accepted,
                    str(artifact),
                    math.nan,
                )
            )
    for identity in discovered:
        unique = {(item.seed, item.path): item for item in discovered[identity]}
        discovered[identity] = sorted(
            unique.values(), key=lambda item: (item.seed, str(item.path))
        )
    return dict(discovered)


def load_method_trajectory(path: Path, method: str) -> Any:
    """Load a deterministic method trajectory using single-path conventions."""
    return comparison.load_trajectory(path, method)


def load_pipeline_trajectory(artifact: PipelineArtifact) -> Any:
    """Load pipeline output, using index progress only when timing is absent."""
    try:
        return comparison.load_trajectory(artifact.path, "pipeline")
    except comparison.ComparisonError as exc:
        if "no timestamps or saved sample interval" not in str(exc):
            raise
        loaded = comparison.load_trajectory(
            artifact.path, "pipeline", dt_override=1.0
        )
        loaded.timestamps = np.linspace(0.0, 1.0, len(loaded.q))
        loaded.timestamp_source = "normalized_sample_index_fallback"
        return loaded


def target_from_trajectory(item: Any) -> np.ndarray:
    """Require one finite Cartesian target path."""
    if item.desired_path is None:
        raise BenchmarkError(f"No desired Cartesian path found for {item.selected_file}")
    target = np.asarray(item.desired_path, dtype=np.float64)
    if target.ndim != 2 or target.shape[1] != 3 or not np.all(np.isfinite(target)):
        raise BenchmarkError(f"Malformed Cartesian target in {item.selected_file}")
    return target


def verify_targets(
    path_id: str, deterministic: Sequence[Any], pipeline_items: Sequence[Any]
) -> dict[str, Any]:
    """Validate every available target against the IK target."""
    reference = target_from_trajectory(deterministic[0])
    compared = 0
    maximum = 0.0
    sources = ["path_id", "IK desired_path"]
    for item in [*deterministic[1:], *pipeline_items]:
        if item.desired_path is None:
            continue
        target = np.asarray(item.desired_path, dtype=np.float64)
        if target.shape != reference.shape:
            raise BenchmarkError(
                f"{path_id}: target shape mismatch {reference.shape} != {target.shape}"
            )
        difference = float(np.max(np.abs(reference - target)))
        maximum = max(maximum, difference)
        if not np.allclose(
            reference,
            target,
            rtol=comparison.TARGET_RTOL,
            atol=comparison.TARGET_ATOL_M,
        ):
            raise BenchmarkError(
                f"{path_id}: Cartesian targets disagree; max difference={difference}"
            )
        compared += 1
        sources.append(str(item.selected_file))
    return {
        "path_identity_verified": True,
        "verification_sources": ";".join(sources),
        "cartesian_target_compared": compared > 0,
        "target_max_difference_m": maximum,
    }


def resample_target(target: np.ndarray, sample_count: int) -> np.ndarray:
    """Map a complete target path to the benchmark progress grid."""
    source = np.linspace(0.0, 1.0, len(target))
    destination = np.linspace(0.0, 1.0, sample_count)
    values = np.asarray(
        CubicSpline(source, target, axis=0, extrapolate=False)(destination),
        dtype=np.float64,
    )
    values[0] = target[0]
    values[-1] = target[-1]
    return values


def make_robot_context() -> Any:
    """Construct the authoritative existing xMateCR7 robot context."""
    evaluator = importlib.import_module(
        "evaluate_diffusion_v7_teacher_forced_validation"
    )
    ik_module = importlib.import_module("generate_ik_seed_path")
    return evaluator.make_robot_context(Path(ik_module.DEFAULT_URDF_PATH))


def authoritative_fk(context: Any, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reuse the canonical full-transform FK implementation."""
    helper = importlib.import_module("orientation_aware_adaptive_ik")
    positions, rotations, _ = helper.trajectory_full_transform_fk(
        context.robot, q, context.joint_names, context.ee_link
    )
    return positions, rotations


def orientation_errors(
    target_rotation: np.ndarray, actual_rotations: np.ndarray
) -> np.ndarray:
    """Reuse the repository geodesic orientation error."""
    helper = importlib.import_module("orientation_aware_adaptive_ik")
    return helper.orientation_error_trajectory(target_rotation, actual_rotations)


def integration_weights(timestamps: np.ndarray) -> np.ndarray:
    """Return trapezoidal integration weights for a strictly increasing grid."""
    times = np.asarray(timestamps, dtype=np.float64)
    differences = np.diff(times)
    weights = np.empty_like(times)
    weights[0] = differences[0] / 2.0
    weights[-1] = differences[-1] / 2.0
    weights[1:-1] = (differences[:-1] + differences[1:]) / 2.0
    return weights


def rms(values: np.ndarray) -> float:
    """Return pooled root mean square."""
    return float(np.sqrt(np.mean(np.square(values))))


def safe_ratio(numerator: float, denominator: float) -> float:
    """Divide finite metrics without returning infinity."""
    return math.nan if denominator == 0.0 else float(numerator / denominator)


def boundary_indices(sample_count: int, execution_horizon: int) -> np.ndarray:
    """Return rollout boundaries excluding sample zero."""
    return np.arange(execution_horizon, sample_count, execution_horizon, dtype=int)


def boundary_mask(
    sample_count: int, boundaries: np.ndarray, radius: int
) -> np.ndarray:
    """Return samples within ``radius`` of any rollout boundary."""
    mask = np.zeros(sample_count, dtype=bool)
    for boundary in np.asarray(boundaries, dtype=int):
        start = max(0, int(boundary) - radius)
        stop = min(sample_count, int(boundary) + radius + 1)
        mask[start:stop] = True
    return mask


def boundary_energy_metrics(
    jerk_energy_samples: np.ndarray,
    integration_weights_s: np.ndarray,
    vicinity: np.ndarray,
) -> dict[str, float]:
    """Normalize integrated jerk energy by covered samples and physical time."""
    energy = np.asarray(jerk_energy_samples, dtype=np.float64)
    weights = np.asarray(integration_weights_s, dtype=np.float64)
    mask = np.asarray(vicinity, dtype=bool)
    if (
        energy.ndim != 1
        or weights.shape != energy.shape
        or mask.shape != energy.shape
        or not np.all(np.isfinite(energy))
        or not np.all(np.isfinite(weights))
        or np.any(weights < 0.0)
    ):
        raise BenchmarkError("Boundary energy inputs must be finite aligned vectors")
    total_energy = float(np.sum(energy))
    boundary_energy = float(np.sum(energy[mask]))
    nonboundary_energy = float(np.sum(energy[~mask]))
    total_time = float(np.sum(weights))
    boundary_time = float(np.sum(weights[mask]))
    nonboundary_time = float(np.sum(weights[~mask]))
    sample_fraction = float(np.mean(mask))
    time_fraction = safe_ratio(boundary_time, total_time)
    energy_fraction = safe_ratio(boundary_energy, total_energy)
    enrichment = safe_ratio(energy_fraction, time_fraction)
    boundary_density = safe_ratio(boundary_energy, boundary_time)
    nonboundary_density = safe_ratio(nonboundary_energy, nonboundary_time)
    density_ratio = safe_ratio(boundary_density, nonboundary_density)
    return {
        "boundary_sample_fraction": sample_fraction,
        "boundary_time_fraction": time_fraction,
        "boundary_jerk_energy_fraction": energy_fraction,
        "boundary_jerk_energy_enrichment": enrichment,
        "boundary_jerk_energy_density_rad2_s6": boundary_density,
        "nonboundary_jerk_energy_density_rad2_s6": nonboundary_density,
        "boundary_to_nonboundary_energy_density_ratio": density_ratio,
    }


def boundary_sensitivity_metrics(
    jerk_energy_samples: np.ndarray,
    integration_weights_s: np.ndarray,
    sample_count: int,
    boundaries: np.ndarray,
    radii: Sequence[int],
) -> dict[str, float]:
    """Calculate coverage-normalized boundary diagnostics for unique radii."""
    output: dict[str, float] = {}
    for radius in sorted(set(int(value) for value in radii)):
        if radius < 0:
            raise BenchmarkError("Boundary sensitivity radii must be nonnegative")
        metrics = boundary_energy_metrics(
            jerk_energy_samples,
            integration_weights_s,
            boundary_mask(sample_count, boundaries, radius),
        )
        prefix = f"boundary_r{radius}_"
        output.update(
            {
                prefix + "sample_fraction": metrics["boundary_sample_fraction"],
                prefix + "time_fraction": metrics["boundary_time_fraction"],
                prefix
                + "jerk_energy_fraction": metrics[
                    "boundary_jerk_energy_fraction"
                ],
                prefix
                + "jerk_energy_enrichment": metrics[
                    "boundary_jerk_energy_enrichment"
                ],
                prefix
                + "energy_density_ratio": metrics[
                    "boundary_to_nonboundary_energy_density_ratio"
                ],
            }
        )
    return output


def spectral_metrics(
    signal: np.ndarray,
    timestamps: np.ndarray,
    high_frequency_fraction: float,
) -> dict[str, Any]:
    """Calculate per-joint and aggregate real-FFT energy diagnostics."""
    values = np.asarray(signal, dtype=np.float64)
    dt = float(np.median(np.diff(timestamps)))
    sampling_frequency = 1.0 / dt
    frequencies = np.fft.rfftfreq(len(values), d=dt)
    nonzero = frequencies > 0.0
    nonzero_indices = np.flatnonzero(nonzero)
    high_count = max(1, int(math.ceil(len(nonzero_indices) * high_frequency_fraction)))
    high_indices = nonzero_indices[-high_count:]
    cutoff = float(frequencies[high_indices[0]])
    ratios: list[float] = []
    dominant: list[float] = []
    centroids: list[float] = []
    aggregate_power = np.zeros_like(frequencies)
    for joint_index in range(JOINT_COUNT):
        centered = values[:, joint_index] - np.mean(values[:, joint_index])
        power = np.square(np.abs(np.fft.rfft(centered)))
        aggregate_power += power
        total = float(np.sum(power[nonzero]))
        ratios.append(
            math.nan if total == 0.0 else float(np.sum(power[high_indices]) / total)
        )
        if total == 0.0:
            dominant.append(math.nan)
            centroids.append(math.nan)
        else:
            local = power[nonzero_indices]
            dominant.append(float(frequencies[nonzero_indices[np.argmax(local)]]))
            centroids.append(
                float(np.sum(frequencies[nonzero] * power[nonzero]) / total)
            )
    aggregate_total = float(np.sum(aggregate_power[nonzero]))
    aggregate_dominant = (
        math.nan
        if aggregate_total == 0.0
        else float(
            frequencies[
                nonzero_indices[np.argmax(aggregate_power[nonzero_indices])]
            ]
        )
    )
    aggregate_centroid = (
        math.nan
        if aggregate_total == 0.0
        else float(
            np.sum(frequencies[nonzero] * aggregate_power[nonzero])
            / aggregate_total
        )
    )
    finite_ratios = np.asarray([value for value in ratios if math.isfinite(value)])
    return {
        "high_frequency_energy_ratio_per_joint": json.dumps(ratios),
        "mean_high_frequency_energy_ratio": (
            float(np.mean(finite_ratios)) if finite_ratios.size else math.nan
        ),
        "max_high_frequency_energy_ratio": (
            float(np.max(finite_ratios)) if finite_ratios.size else math.nan
        ),
        "dominant_frequency_per_joint_hz": json.dumps(dominant),
        "dominant_frequency_hz": aggregate_dominant,
        "spectral_centroid_per_joint_hz": json.dumps(centroids),
        "spectral_centroid_hz": aggregate_centroid,
        "spectral_sampling_frequency_hz": sampling_frequency,
        "spectral_nyquist_frequency_hz": sampling_frequency / 2.0,
        "spectral_frequency_cutoff_hz": cutoff,
        "spectral_method": "mean-removed real FFT squared magnitude",
        "spectral_window": "none",
    }


def smoothness_metrics(
    q: np.ndarray,
    timestamps: np.ndarray,
    *,
    endpoint_exclusion: int,
    execution_horizon: int,
    boundary_radius: int,
    boundary_sensitivity_radii: Sequence[int] = (0, 1, 2),
    spectral_signal: str,
    high_frequency_fraction: float,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    """Calculate primary unfiltered derivative, continuity, boundary, and FFT metrics."""
    if endpoint_exclusion * 2 >= len(q) - 1:
        raise BenchmarkError(
            "Endpoint exclusion leaves fewer than two interior time intervals"
        )
    velocity, acceleration, jerk = comparison.compute_derivatives(q, timestamps)
    weights = integration_weights(timestamps)
    interior = slice(endpoint_exclusion, len(q) - endpoint_exclusion)
    interior_acceleration = acceleration[interior]
    interior_jerk = jerk[interior]
    jerk_energy_samples = np.sum(np.square(jerk), axis=1) * weights
    total_jerk_energy = float(np.sum(jerk_energy_samples))
    endpoint_mask = np.ones(len(q), dtype=bool)
    endpoint_mask[interior] = False

    q_steps = np.diff(q, axis=0)
    velocity_steps = np.diff(velocity, axis=0)
    acceleration_steps = np.diff(acceleration, axis=0)
    boundaries = boundary_indices(len(q), execution_horizon)
    vicinity = boundary_mask(len(q), boundaries, boundary_radius)
    jerk_norm = np.linalg.norm(jerk, axis=1)
    position_boundary = np.linalg.norm(q[boundaries] - q[boundaries - 1], axis=1)
    velocity_boundary = np.linalg.norm(
        velocity[boundaries] - velocity[boundaries - 1], axis=1
    )
    acceleration_boundary = np.linalg.norm(
        acceleration[boundaries] - acceleration[boundaries - 1], axis=1
    )
    outside = ~vicinity
    primary_boundary_metrics = boundary_energy_metrics(
        jerk_energy_samples, weights, vicinity
    )
    sensitivity_boundary_metrics = boundary_sensitivity_metrics(
        jerk_energy_samples,
        weights,
        len(q),
        boundaries,
        (*boundary_sensitivity_radii, boundary_radius),
    )
    sign_changes = np.sum(
        np.diff(np.signbit(acceleration), axis=0) != 0, axis=0
    ).astype(int)
    signal = {
        "position": q,
        "velocity": velocity,
        "acceleration": acceleration,
    }[spectral_signal]

    def integrate_per_joint(
        values: np.ndarray, integration_times: np.ndarray = timestamps
    ) -> list[float]:
        return [
            float(comparison.integrate(values[:, joint], integration_times))
            for joint in range(JOINT_COUNT)
        ]

    def integrate_columns(values: np.ndarray) -> float:
        return float(sum(integrate_per_joint(values)))

    def encoded(values: np.ndarray | Sequence[float]) -> str:
        return json.dumps(np.asarray(values, dtype=np.float64).tolist())

    def empty_safe_stat(values: np.ndarray, operation: str) -> float:
        if values.size == 0:
            return math.nan
        if operation == "mean":
            return float(np.mean(values))
        return float(np.max(values))

    velocity_squared_integrals = integrate_per_joint(np.square(velocity))
    acceleration_absolute_integrals = integrate_per_joint(np.abs(acceleration))
    acceleration_squared_integrals = integrate_per_joint(np.square(acceleration))
    jerk_absolute_integrals = integrate_per_joint(np.abs(jerk))
    jerk_squared_integrals = integrate_per_joint(np.square(jerk))
    interior_times = timestamps[interior]
    interior_acceleration_absolute_integrals = integrate_per_joint(
        np.abs(interior_acceleration), interior_times
    )
    interior_acceleration_squared_integrals = integrate_per_joint(
        np.square(interior_acceleration), interior_times
    )
    interior_jerk_absolute_integrals = integrate_per_joint(
        np.abs(interior_jerk), interior_times
    )
    interior_jerk_squared_integrals = integrate_per_joint(
        np.square(interior_jerk), interior_times
    )

    metrics: dict[str, Any] = {
        "sample_count": len(q),
        "duration_s": float(timestamps[-1] - timestamps[0]),
        "rms_velocity_rad_s": rms(velocity),
        "max_abs_velocity_rad_s": float(np.max(np.abs(velocity))),
        "integrated_squared_velocity_rad2_s": integrate_columns(
            np.square(velocity)
        ),
        "velocity_rms_per_joint_rad_s": encoded(
            np.sqrt(np.mean(np.square(velocity), axis=0))
        ),
        "velocity_max_abs_per_joint_rad_s": encoded(
            np.max(np.abs(velocity), axis=0)
        ),
        "velocity_integrated_squared_per_joint_rad2_s": encoded(
            velocity_squared_integrals
        ),
        "mean_abs_acceleration_rad_s2": float(np.mean(np.abs(acceleration))),
        "rms_acceleration_rad_s2": rms(acceleration),
        "interior_rms_acceleration_rad_s2": rms(interior_acceleration),
        "max_abs_acceleration_rad_s2": float(np.max(np.abs(acceleration))),
        "interior_max_abs_acceleration_rad_s2": float(
            np.max(np.abs(interior_acceleration))
        ),
        "interior_mean_abs_acceleration_rad_s2": float(
            np.mean(np.abs(interior_acceleration))
        ),
        "interior_integrated_absolute_acceleration_rad_s": float(
            sum(interior_acceleration_absolute_integrals)
        ),
        "interior_integrated_squared_acceleration_rad2_s3": float(
            sum(interior_acceleration_squared_integrals)
        ),
        "integrated_absolute_acceleration_rad_s": integrate_columns(
            np.abs(acceleration)
        ),
        "integrated_squared_acceleration_rad2_s3": integrate_columns(
            np.square(acceleration)
        ),
        "acceleration_mean_abs_per_joint_rad_s2": encoded(
            np.mean(np.abs(acceleration), axis=0)
        ),
        "acceleration_rms_per_joint_rad_s2": encoded(
            np.sqrt(np.mean(np.square(acceleration), axis=0))
        ),
        "acceleration_max_abs_per_joint_rad_s2": encoded(
            np.max(np.abs(acceleration), axis=0)
        ),
        "acceleration_integrated_absolute_per_joint_rad_s": encoded(
            acceleration_absolute_integrals
        ),
        "acceleration_integrated_squared_per_joint_rad2_s3": encoded(
            acceleration_squared_integrals
        ),
        "interior_acceleration_mean_abs_per_joint_rad_s2": encoded(
            np.mean(np.abs(interior_acceleration), axis=0)
        ),
        "interior_acceleration_rms_per_joint_rad_s2": encoded(
            np.sqrt(np.mean(np.square(interior_acceleration), axis=0))
        ),
        "interior_acceleration_max_abs_per_joint_rad_s2": encoded(
            np.max(np.abs(interior_acceleration), axis=0)
        ),
        "interior_acceleration_integrated_absolute_per_joint_rad_s": encoded(
            interior_acceleration_absolute_integrals
        ),
        "interior_acceleration_integrated_squared_per_joint_rad2_s3": encoded(
            interior_acceleration_squared_integrals
        ),
        "mean_abs_jerk_rad_s3": float(np.mean(np.abs(jerk))),
        "median_abs_jerk_rad_s3": float(np.median(np.abs(jerk))),
        "rms_jerk_rad_s3": rms(jerk),
        "interior_rms_jerk_rad_s3": rms(interior_jerk),
        "max_abs_jerk_rad_s3": float(np.max(np.abs(jerk))),
        "interior_max_abs_jerk_rad_s3": float(np.max(np.abs(interior_jerk))),
        "interior_mean_abs_jerk_rad_s3": float(np.mean(np.abs(interior_jerk))),
        "interior_median_abs_jerk_rad_s3": float(
            np.median(np.abs(interior_jerk))
        ),
        "interior_integrated_absolute_jerk_rad_s2": float(
            sum(interior_jerk_absolute_integrals)
        ),
        "interior_integrated_squared_jerk_rad2_s5": float(
            sum(interior_jerk_squared_integrals)
        ),
        "integrated_absolute_jerk_rad_s2": integrate_columns(np.abs(jerk)),
        "integrated_squared_jerk_rad2_s5": total_jerk_energy,
        "jerk_mean_abs_per_joint_rad_s3": encoded(
            np.mean(np.abs(jerk), axis=0)
        ),
        "jerk_median_abs_per_joint_rad_s3": encoded(
            np.median(np.abs(jerk), axis=0)
        ),
        "jerk_rms_per_joint_rad_s3": encoded(
            np.sqrt(np.mean(np.square(jerk), axis=0))
        ),
        "jerk_max_abs_per_joint_rad_s3": encoded(
            np.max(np.abs(jerk), axis=0)
        ),
        "jerk_integrated_absolute_per_joint_rad_s2": encoded(
            jerk_absolute_integrals
        ),
        "jerk_integrated_squared_per_joint_rad2_s5": encoded(
            jerk_squared_integrals
        ),
        "interior_jerk_mean_abs_per_joint_rad_s3": encoded(
            np.mean(np.abs(interior_jerk), axis=0)
        ),
        "interior_jerk_median_abs_per_joint_rad_s3": encoded(
            np.median(np.abs(interior_jerk), axis=0)
        ),
        "interior_jerk_rms_per_joint_rad_s3": encoded(
            np.sqrt(np.mean(np.square(interior_jerk), axis=0))
        ),
        "interior_jerk_max_abs_per_joint_rad_s3": encoded(
            np.max(np.abs(interior_jerk), axis=0)
        ),
        "interior_jerk_integrated_absolute_per_joint_rad_s2": encoded(
            interior_jerk_absolute_integrals
        ),
        "interior_jerk_integrated_squared_per_joint_rad2_s5": encoded(
            interior_jerk_squared_integrals
        ),
        "endpoint_jerk_energy_fraction": safe_ratio(
            float(np.sum(jerk_energy_samples[endpoint_mask])), total_jerk_energy
        ),
        "max_joint_step_l2_rad": float(
            np.max(np.linalg.norm(q_steps, axis=1))
        ),
        "max_joint_step_per_joint_rad": json.dumps(
            np.max(np.abs(q_steps), axis=0).tolist()
        ),
        "max_velocity_step_l2_rad_s": float(
            np.max(np.linalg.norm(velocity_steps, axis=1))
        ),
        "max_acceleration_step_l2_rad_s2": float(
            np.max(np.linalg.norm(acceleration_steps, axis=1))
        ),
        "position_total_variation_rad": float(np.sum(np.abs(q_steps))),
        "acceleration_total_variation_rad_s2": float(
            np.sum(np.abs(acceleration_steps))
        ),
        "acceleration_direction_change_count_per_joint": json.dumps(
            sign_changes.tolist()
        ),
        "rollout_boundary_indices": json.dumps(boundaries.tolist()),
        "boundary_control_type": "pipeline rollout boundaries or equivalent control indices",
        "mean_boundary_position_discontinuity_l2_rad": float(
            empty_safe_stat(position_boundary, "mean")
        ),
        "max_boundary_position_discontinuity_l2_rad": float(
            empty_safe_stat(position_boundary, "max")
        ),
        "mean_boundary_velocity_discontinuity_l2_rad_s": float(
            empty_safe_stat(velocity_boundary, "mean")
        ),
        "max_boundary_velocity_discontinuity_l2_rad_s": float(
            empty_safe_stat(velocity_boundary, "max")
        ),
        "mean_boundary_acceleration_discontinuity_l2_rad_s2": float(
            empty_safe_stat(acceleration_boundary, "mean")
        ),
        "max_boundary_acceleration_discontinuity_l2_rad_s2": float(
            empty_safe_stat(acceleration_boundary, "max")
        ),
        "mean_jerk_norm_at_boundaries_rad_s3": float(
            empty_safe_stat(jerk_norm[boundaries], "mean")
        ),
        "mean_jerk_norm_outside_boundaries_rad_s3": (
            float(np.mean(jerk_norm[outside])) if np.any(outside) else math.nan
        ),
        "boundary_to_nonboundary_jerk_ratio": safe_ratio(
            empty_safe_stat(jerk_norm[boundaries], "mean"),
            float(np.mean(jerk_norm[outside])) if np.any(outside) else 0.0,
        ),
        **primary_boundary_metrics,
        **sensitivity_boundary_metrics,
    }
    metrics.update(
        spectral_metrics(signal, timestamps, high_frequency_fraction)
    )
    return metrics, velocity, acceleration, jerk


def tracking_metrics(
    q: np.ndarray,
    target: np.ndarray,
    context: Any,
    *,
    mean_cartesian_threshold_m: float,
    max_cartesian_threshold_m: float,
    orientation_threshold_rad: float | None,
    target_rotation: np.ndarray | None = None,
    fk_override: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[dict[str, Any], bool, str]:
    """Evaluate authoritative FK tracking and eligibility."""
    positions, rotations = (
        fk_override if fk_override is not None else authoritative_fk(context, q)
    )
    errors = np.linalg.norm(positions - target, axis=1)
    mean_error = float(np.mean(errors))
    maximum_error = float(np.max(errors))
    eligible = (
        mean_error <= mean_cartesian_threshold_m
        and maximum_error <= max_cartesian_threshold_m
    )
    reasons: list[str] = []
    if mean_error > mean_cartesian_threshold_m:
        reasons.append("mean_cartesian_error_threshold")
    if maximum_error > max_cartesian_threshold_m:
        reasons.append("max_cartesian_error_threshold")
    orientation = {
        "orientation_target_available": target_rotation is not None,
        "mean_orientation_error_rad": math.nan,
        "rms_orientation_error_rad": math.nan,
        "max_orientation_error_rad": math.nan,
        "endpoint_orientation_error_rad": math.nan,
    }
    if target_rotation is not None:
        orientation_values = orientation_errors(target_rotation, rotations)
        orientation.update(
            {
                "mean_orientation_error_rad": float(np.mean(orientation_values)),
                "rms_orientation_error_rad": rms(orientation_values),
                "max_orientation_error_rad": float(np.max(orientation_values)),
                "endpoint_orientation_error_rad": float(orientation_values[-1]),
            }
        )
        if (
            orientation_threshold_rad is not None
            and float(np.max(orientation_values)) > orientation_threshold_rad
        ):
            eligible = False
            reasons.append("orientation_error_threshold")
    metrics = {
        "mean_cartesian_error_m": mean_error,
        "rms_cartesian_error_m": rms(errors),
        "median_cartesian_error_m": float(np.median(errors)),
        "max_cartesian_error_m": maximum_error,
        "endpoint_cartesian_error_m": float(errors[-1]),
        "fraction_samples_below_cartesian_threshold": float(
            np.mean(errors <= max_cartesian_threshold_m)
        ),
        "fraction_samples_below_mean_cartesian_threshold": float(
            np.mean(errors <= mean_cartesian_threshold_m)
        ),
        "fraction_samples_below_max_cartesian_threshold": float(
            np.mean(errors <= max_cartesian_threshold_m)
        ),
        "mean_cartesian_error_threshold_m": mean_cartesian_threshold_m,
        "max_cartesian_error_threshold_m": max_cartesian_threshold_m,
        **orientation,
    }
    return metrics, eligible, ";".join(reasons)


def target_rotation_from_items(items: Sequence[Any]) -> np.ndarray | None:
    """Return one consistent saved target rotation when available."""
    found: list[np.ndarray] = []
    for item in items:
        embedded = item.metadata.get("embedded", {})
        for key in ("target_rotation_matrix", "fixed_rotation_matrix"):
            if key in embedded:
                value = np.asarray(embedded[key], dtype=np.float64)
                if value.shape == (3, 3):
                    found.append(value)
    if not found:
        return None
    if any(not np.allclose(found[0], value, rtol=1e-7, atol=1e-9) for value in found[1:]):
        raise BenchmarkError("Saved target orientation matrices disagree")
    return found[0]


def evaluate_aligned(
    *,
    path_id: str,
    method: str,
    seed: int | None,
    selected_file: Path,
    accepted: bool,
    acceptance_source: str,
    runtime_s: float,
    q: np.ndarray,
    timestamps: np.ndarray,
    target: np.ndarray,
    context: Any,
    args: argparse.Namespace,
    verification: Mapping[str, Any],
    target_rotation: np.ndarray | None,
) -> EvaluatedTrajectory:
    """Evaluate one aligned trajectory without suppressing unfavorable results."""
    smooth, velocity, acceleration, jerk = smoothness_metrics(
        q,
        timestamps,
        endpoint_exclusion=args.endpoint_exclusion,
        execution_horizon=args.execution_horizon,
        boundary_radius=args.boundary_radius,
        boundary_sensitivity_radii=args.boundary_sensitivity_radii,
        spectral_signal=args.spectral_signal,
        high_frequency_fraction=args.high_frequency_fraction,
    )
    tracking, tracking_eligible, tracking_reason = tracking_metrics(
        q,
        target,
        context,
        mean_cartesian_threshold_m=args.mean_cartesian_error_threshold_m,
        max_cartesian_threshold_m=args.max_cartesian_error_threshold_m,
        orientation_threshold_rad=args.orientation_error_threshold_rad,
        target_rotation=target_rotation,
    )
    accepted_required_failure = (
        method == "pipeline" and args.require_accepted_pipeline and not accepted
    )
    reasons = [value for value in (tracking_reason,) if value]
    if accepted_required_failure:
        reasons.append("pipeline_artifact_not_accepted")
    claim_eligible = bool(
        tracking_eligible
        and not accepted_required_failure
        and args.timing_policy != "shared_interval_diagnostic"
    )
    row: dict[str, Any] = {
        "path_id": path_id,
        "method": method,
        "seed": "" if seed is None else seed,
        "selected_file": str(selected_file),
        "accepted": accepted,
        "acceptance_source": acceptance_source,
        "trajectory_valid": True,
        "tracking_eligible": tracking_eligible,
        "smoothness_claim_eligible": claim_eligible,
        "exclusion_reason": ";".join(reasons),
        "timing_policy": args.timing_policy,
        "standardized_duration_s": float(timestamps[-1] - timestamps[0]),
        "runtime_s": runtime_s,
        **verification,
        **tracking,
        **smooth,
    }
    row.update({alias: row[source] for alias, source in PATH_LEVEL_ALIASES.items()})
    return EvaluatedTrajectory(
        row, q, timestamps, velocity, acceleration, jerk
    )


def numeric_metric_keys(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Return aggregate-safe numeric metric columns."""
    excluded = {
        "seed",
        "accepted",
        "trajectory_valid",
        "tracking_eligible",
        "smoothness_claim_eligible",
        "path_identity_verified",
        "cartesian_target_compared",
    }
    keys: list[str] = []
    sensitivity_keys = sorted(
        {
            key
            for row in rows
            for key in row
            if re.fullmatch(
                r"boundary_r\d+_(?:sample_fraction|time_fraction|"
                r"jerk_energy_fraction|jerk_energy_enrichment|"
                r"energy_density_ratio)",
                key,
            )
        }
    )
    for key in (*SUMMARY_METRICS, *sensitivity_keys):
        if key in excluded:
            continue
        if any(
            isinstance(row.get(key), (int, float, np.integer, np.floating))
            for row in rows
        ):
            keys.append(key)
    return keys


def aggregate_pipeline_path(
    path_id: str,
    seed_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], dict[str, Any] | None]:
    """Aggregate eligible seeds by median and retain metric-specific best/worst."""
    eligible = [
        row for row in seed_rows if bool(row.get("smoothness_claim_eligible"))
    ]
    accepted_count = sum(bool(row.get("accepted")) for row in seed_rows)
    selection_rows: list[dict[str, Any]] = []
    if not eligible:
        return None, selection_rows, None
    metric_keys = numeric_metric_keys(eligible)
    aggregate: dict[str, Any] = {
        "path_id": path_id,
        "method": "pipeline",
        "aggregation": "median_across_eligible_accepted_seeds",
        "num_pipeline_seeds": len(seed_rows),
        "trajectory_valid": True,
        "tracking_eligible": True,
        "smoothness_claim_eligible": True,
        "accepted": True,
        "runtime_s": math.nan,
    }
    for key in metric_keys:
        values = np.asarray([float(row[key]) for row in eligible], dtype=np.float64)
        finite = values[np.isfinite(values)]
        aggregate[key] = float(np.median(finite)) if finite.size else math.nan
    for alias, source in PATH_LEVEL_ALIASES.items():
        aggregate[alias] = aggregate.get(source, math.nan)
    finite_runtimes = np.asarray(
        [
            float(row.get("runtime_s", math.nan))
            for row in eligible
            if math.isfinite(float(row.get("runtime_s", math.nan)))
        ],
        dtype=np.float64,
    )
    if finite_runtimes.size:
        aggregate["runtime_s"] = float(np.median(finite_runtimes))
    aggregate["path_identity_verified"] = all(
        bool(row.get("path_identity_verified")) for row in eligible
    )
    aggregate["cartesian_target_compared"] = any(
        bool(row.get("cartesian_target_compared")) for row in eligible
    )
    aggregate["verification_sources"] = ";".join(
        sorted({str(row.get("verification_sources", "")) for row in eligible})
    )
    aggregate["target_max_difference_m"] = max(
        float(row.get("target_max_difference_m", 0.0)) for row in eligible
    )
    for metric in metric_keys:
        candidates = [
            row for row in eligible if math.isfinite(float(row.get(metric, math.nan)))
        ]
        if not candidates:
            continue
        ordered = sorted(candidates, key=lambda row: float(row[metric]))
        values = np.asarray([float(row[metric]) for row in candidates])
        selection_rows.append(
            {
                "path_id": path_id,
                "metric": metric,
                "primary_aggregation": "median",
                "median_value": float(np.median(values)),
                "mean_value": float(np.mean(values)),
                "std_value": float(np.std(values)),
                "q1_value": float(np.quantile(values, 0.25)),
                "q3_value": float(np.quantile(values, 0.75)),
                "iqr_value": float(np.quantile(values, 0.75) - np.quantile(values, 0.25)),
                "minimum": float(np.min(values)),
                "maximum": float(np.max(values)),
                "best_seed": ordered[0]["seed"],
                "best_value": float(ordered[0][metric]),
                "worst_seed": ordered[-1]["seed"],
                "worst_value": float(ordered[-1][metric]),
                "eligible_seed_count": len(eligible),
                "accepted_seed_count": accepted_count,
                "discovered_seed_count": len(seed_rows),
            }
        )
    rms_candidates = sorted(
        eligible, key=lambda row: float(row["rms_jerk_rad_s3"])
    )
    best_seed_row = dict(rms_candidates[0])
    best_seed_row["aggregation"] = "secondary_best_seed_by_rms_jerk"
    return aggregate, selection_rows, best_seed_row


def deterministic_path_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Convert one IK/MLP trajectory row to path-level form."""
    result = dict(row)
    result["aggregation"] = "deterministic_single_trajectory"
    result["num_pipeline_seeds"] = 0
    return result


def method_summaries(path_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Summarize path-level values without treating samples or seeds as replicates."""
    output: list[dict[str, Any]] = []
    for method in METHODS:
        method_rows = [row for row in path_rows if row["method"] == method]
        eligible = [
            row for row in method_rows if bool(row.get("smoothness_claim_eligible"))
        ]
        sensitivity_enrichment_metrics = sorted(
            {
                key
                for row in path_rows
                for key in row
                if re.fullmatch(r"boundary_r\d+_jerk_energy_enrichment", key)
            }
        )
        for metric in (*SUMMARY_METRICS, *sensitivity_enrichment_metrics):
            values = np.asarray(
                [
                    float(row.get(metric, math.nan))
                    for row in eligible
                    if math.isfinite(float(row.get(metric, math.nan)))
                ]
            )
            output.append(
                {
                    "method": method,
                    "metric": metric,
                    "eligible_path_count": len(values),
                    "total_path_count": len(method_rows),
                    "mean": float(np.mean(values)) if len(values) else math.nan,
                    "median": float(np.median(values)) if len(values) else math.nan,
                    "std": float(np.std(values)) if len(values) else math.nan,
                    "q1": float(np.quantile(values, 0.25)) if len(values) else math.nan,
                    "q3": float(np.quantile(values, 0.75)) if len(values) else math.nan,
                    "iqr": (
                        float(np.quantile(values, 0.75) - np.quantile(values, 0.25))
                        if len(values)
                        else math.nan
                    ),
                    "minimum": float(np.min(values)) if len(values) else math.nan,
                    "maximum": float(np.max(values)) if len(values) else math.nan,
                }
            )
    return output


def matched_values(
    path_rows: Sequence[Mapping[str, Any]],
    methods: Sequence[str],
    metric: str,
) -> tuple[list[str], list[np.ndarray]]:
    """Return one metric value per matched eligible path and method."""
    lookup = {
        (str(row["path_id"]), str(row["method"])): row
        for row in path_rows
        if bool(row.get("smoothness_claim_eligible"))
    }
    path_ids = sorted(
        {
            path_id
            for path_id, method in lookup
            if all((path_id, candidate) in lookup for candidate in methods)
            and all(
                math.isfinite(float(lookup[(path_id, candidate)].get(metric, math.nan)))
                for candidate in methods
            )
        }
    )
    arrays = [
        np.asarray([float(lookup[(path_id, method)][metric]) for path_id in path_ids])
        for method in methods
    ]
    return path_ids, arrays


def win_rates(path_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Calculate lower-is-better path-level matched win rates."""
    output: list[dict[str, Any]] = []
    for reference, candidate in (("ik", "pipeline"), ("mlp", "pipeline"), ("ik", "mlp")):
        for metric in PRIMARY_METRICS:
            _, (reference_values, candidate_values) = matched_values(
                path_rows, (reference, candidate), metric
            )
            differences = candidate_values - reference_values
            tolerance = TIE_ATOL + TIE_RTOL * np.abs(reference_values)
            ties = np.abs(differences) <= tolerance
            wins = differences < -tolerance
            losses = differences > tolerance
            improvements = np.asarray(
                [
                    comparison.percent_improvement(float(ref), float(comp))
                    for ref, comp in zip(reference_values, candidate_values)
                ]
            )
            finite_improvements = improvements[np.isfinite(improvements)]
            output.append(
                {
                    "reference_method": reference,
                    "comparison_method": candidate,
                    "metric": metric,
                    "matched_path_count": len(reference_values),
                    "comparison_wins": int(np.sum(wins)),
                    "reference_wins": int(np.sum(losses)),
                    "ties": int(np.sum(ties)),
                    "comparison_win_rate": (
                        float(np.mean(wins)) if len(wins) else math.nan
                    ),
                    "median_percent_improvement_lower_is_better": (
                        float(np.median(finite_improvements))
                        if len(finite_improvements)
                        else math.nan
                    ),
                }
            )
    return output


def paired_bootstrap(
    left: np.ndarray,
    right: np.ndarray,
    *,
    seed: int = BOOTSTRAP_SEED,
    iterations: int = BOOTSTRAP_ITERATIONS,
) -> tuple[float, float, float, float]:
    """Bootstrap median difference and lower-is-better improvement by path."""
    if len(left) == 0:
        return math.nan, math.nan, math.nan, math.nan
    rng = np.random.default_rng(seed)
    differences = np.empty(iterations)
    improvements = np.empty(iterations)
    for index in range(iterations):
        selected = rng.integers(0, len(left), len(left))
        l_values = left[selected]
        r_values = right[selected]
        differences[index] = np.median(r_values - l_values)
        valid = l_values != 0.0
        improvements[index] = (
            np.median(100.0 * (l_values[valid] - r_values[valid]) / l_values[valid])
            if np.any(valid)
            else math.nan
        )
    return (
        float(np.quantile(differences, 0.025)),
        float(np.quantile(differences, 0.975)),
        float(np.nanquantile(improvements, 0.025)),
        float(np.nanquantile(improvements, 0.975)),
    )


def holm_bonferroni(p_values: Sequence[float]) -> list[float]:
    """Return monotone Holm–Bonferroni adjusted p-values."""
    values = np.asarray(p_values, dtype=np.float64)
    adjusted = np.full(len(values), np.nan)
    finite_indices = np.flatnonzero(np.isfinite(values))
    order = finite_indices[np.argsort(values[finite_indices])]
    running = 0.0
    count = len(order)
    for rank, index in enumerate(order):
        running = max(running, (count - rank) * values[index])
        adjusted[index] = min(1.0, running)
    return adjusted.tolist()


def statistical_tests(path_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Run path-level Friedman, Wilcoxon, bootstrap, and Holm correction."""
    output: list[dict[str, Any]] = []
    for metric_index, metric in enumerate(PRIMARY_METRICS):
        diagnostic_note = (
            "; values below 1 indicate reduced boundary concentration, not "
            "necessarily globally smoother motion"
            if metric == "boundary_jerk_energy_enrichment"
            else ""
        )
        _, arrays = matched_values(path_rows, METHODS, metric)
        if arrays and len(arrays[0]) >= 3:
            statistic, p_value = stats.friedmanchisquare(*arrays)
            output.append(
                {
                    "metric": metric,
                    "test": "Friedman",
                    "method_a": "ik",
                    "method_b": "mlp,pipeline",
                    "sample_count": len(arrays[0]),
                    "statistic": float(statistic),
                    "p_value": float(p_value),
                    "effect_estimate": math.nan,
                    "ci_lower": math.nan,
                    "ci_upper": math.nan,
                    "notes": "one value per complete matched path" + diagnostic_note,
                }
            )
        else:
            output.append(
                {
                    "metric": metric,
                    "test": "Friedman",
                    "method_a": "ik",
                    "method_b": "mlp,pipeline",
                    "sample_count": len(arrays[0]) if arrays else 0,
                    "statistic": math.nan,
                    "p_value": math.nan,
                    "effect_estimate": math.nan,
                    "ci_lower": math.nan,
                    "ci_upper": math.nan,
                    "notes": (
                        "insufficient matched paths; no significance claim"
                        + diagnostic_note
                    ),
                }
            )
        for pair_index, (method_a, method_b) in enumerate(
            (("ik", "pipeline"), ("mlp", "pipeline"), ("ik", "mlp"))
        ):
            _, (left, right) = matched_values(
                path_rows, (method_a, method_b), metric
            )
            statistic = p_value = math.nan
            notes = "one value per complete matched path" + diagnostic_note
            if len(left) >= 2:
                try:
                    statistic, p_value = stats.wilcoxon(
                        left, right, zero_method="wilcox"
                    )
                    statistic, p_value = float(statistic), float(p_value)
                except ValueError:
                    notes += "; all paired differences were zero"
            else:
                notes += "; insufficient matched paths"
            lower, upper, improvement_lower, improvement_upper = paired_bootstrap(
                left,
                right,
                seed=BOOTSTRAP_SEED + metric_index * 10 + pair_index,
            )
            output.append(
                {
                    "metric": metric,
                    "test": "Wilcoxon signed-rank with paired bootstrap",
                    "method_a": method_a,
                    "method_b": method_b,
                    "sample_count": len(left),
                    "statistic": statistic,
                    "p_value": p_value,
                    "effect_estimate": (
                        float(np.median(right - left)) if len(left) else math.nan
                    ),
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "percent_improvement_ci_lower": improvement_lower,
                    "percent_improvement_ci_upper": improvement_upper,
                    "notes": notes,
                }
            )
    for metric in PRIMARY_METRICS:
        indices = [index for index, row in enumerate(output) if row["metric"] == metric]
        adjusted = holm_bonferroni([float(output[index]["p_value"]) for index in indices])
        for index, value in zip(indices, adjusted):
            output[index]["adjusted_p_value"] = value
            output[index]["significant_at_0_05"] = bool(
                math.isfinite(value) and value < 0.05
            )
    return output


def save_figure(fig: Any, output: Path, stem: str, show: bool, plt: Any) -> None:
    """Save PNG/PDF figure pair and close it."""
    fig.savefig(output / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output / f"{stem}.pdf", bbox_inches="tight")
    if show:
        plt.show(block=False)
    plt.close(fig)


def eligible_method_values(
    path_rows: Sequence[Mapping[str, Any]], metric: str
) -> list[np.ndarray]:
    """Return finite eligible path arrays in method order."""
    return [
        np.asarray(
            [
                float(row[metric])
                for row in path_rows
                if row["method"] == method
                and bool(row.get("smoothness_claim_eligible"))
                and math.isfinite(float(row.get(metric, math.nan)))
            ]
        )
        for method in METHODS
    ]


def plot_distributions(
    output: Path,
    path_rows: Sequence[Mapping[str, Any]],
    args: argparse.Namespace,
    show: bool,
) -> None:
    """Generate required path-level distribution and Pareto plots."""
    if show:
        matplotlib.use(_INITIAL_BACKEND, force=True)
    else:
        matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    colors = ("#0072B2", "#D55E00", "#009E73")
    labels = [METHOD_LABELS[method] for method in METHODS]

    def boxplot(metric: str, ylabel: str, stem: str) -> None:
        values = eligible_method_values(path_rows, metric)
        fig, axis = plt.subplots(figsize=(8.0, 5.5))
        box_artists = axis.boxplot(values, labels=labels)
        for flier in box_artists["fliers"]:
            flier.set_visible(False)
        for index, (array, color) in enumerate(zip(values, colors), start=1):
            if len(array):
                offsets = np.linspace(-0.07, 0.07, len(array))
                axis.scatter(
                    np.full(len(array), index) + offsets,
                    array,
                    color=color,
                    s=24,
                    alpha=0.8,
                    zorder=3,
                )
        axis.set_ylabel(ylabel)
        axis.set_title(
            f"Path-level {ylabel}\nStandardized complete duration: "
            f"{args.common_duration_s:g} s"
        )
        axis.grid(True, axis="y", alpha=0.25)
        fig.tight_layout()
        save_figure(fig, output, stem, show, plt)

    boxplot("rms_jerk_rad_s3", "RMS jerk (rad/s³)", "rms_jerk_by_method")
    boxplot(
        "integrated_squared_jerk_rad2_s5",
        "Integrated squared jerk (rad²/s⁵)",
        "integrated_squared_jerk_by_method",
    )
    boxplot(
        "rms_acceleration_rad_s2",
        "RMS acceleration (rad/s²)",
        "rms_acceleration_by_method",
    )
    boxplot(
        "mean_high_frequency_energy_ratio",
        "Mean high-frequency energy ratio",
        "high_frequency_energy_ratio",
    )

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.4))
    boundary_panels = (
        ("boundary_time_fraction", "Covered time fraction"),
        ("boundary_jerk_energy_fraction", "Jerk-energy fraction"),
        ("boundary_jerk_energy_enrichment", "Energy enrichment"),
    )
    for panel_index, (metric, title) in enumerate(boundary_panels):
        axis = axes[panel_index]
        values = eligible_method_values(path_rows, metric)
        axis.boxplot(values, labels=labels, showfliers=False)
        for index, (array, color) in enumerate(zip(values, colors), start=1):
            if len(array):
                axis.scatter(
                    np.full(len(array), index),
                    array,
                    color=color,
                    s=22,
                    alpha=0.8,
                )
        if metric == "boundary_jerk_energy_enrichment":
            axis.set_title(title, fontweight="bold")
        else:
            axis.set_title(title)
        axis.grid(True, axis="y", alpha=0.25)
        if metric == "boundary_jerk_energy_enrichment":
            axis.axhline(
                1.0,
                color="black",
                linestyle=":",
                label="Energy proportional to covered time",
            )
            axis.legend(fontsize=7)
    fig.suptitle(
        f"Boundary coverage and energy at radius {args.boundary_radius}; "
        "IK/MLP are control indices"
    )
    fig.tight_layout()
    save_figure(fig, output, "boundary_jerk_energy_ratio", show, plt)

    sensitivity_radii = [
        radius for radius in (0, 1, 2) if radius in args.boundary_sensitivity_radii
    ]
    if not sensitivity_radii:
        sensitivity_radii = list(args.boundary_sensitivity_radii)
    fig, axis = plt.subplots(figsize=(11.0, 6.0))
    group_centers = np.arange(len(METHODS), dtype=np.float64)
    width = 0.22
    for radius_index, radius in enumerate(sensitivity_radii):
        values = eligible_method_values(
            path_rows, f"boundary_r{radius}_jerk_energy_enrichment"
        )
        positions = (
            group_centers
            + (radius_index - (len(sensitivity_radii) - 1) / 2.0) * width
        )
        for method_index, array in enumerate(values):
            if len(array):
                box_artists = axis.boxplot(
                    [array],
                    positions=[positions[method_index]],
                    widths=width * 0.8,
                )
                for flier in box_artists["fliers"]:
                    flier.set_visible(False)
                axis.scatter(
                    np.full(len(array), positions[method_index]),
                    array,
                    s=18,
                    alpha=0.75,
                    label=(
                        f"radius {radius}"
                        if method_index == 0
                        else None
                    ),
                )
    axis.axhline(
        1.0,
        color="black",
        linestyle=":",
        label="Energy proportional to covered time",
    )
    axis.set_xticks(
        group_centers,
        ("IK control indices", "MLP control indices", "Pipeline rollout boundaries"),
    )
    axis.set_ylabel("Boundary jerk-energy enrichment")
    axis.set_title("Boundary-energy enrichment sensitivity to mask radius")
    axis.grid(True, axis="y", alpha=0.25)
    axis.legend(fontsize=8)
    fig.tight_layout()
    save_figure(
        fig, output, "boundary_energy_enrichment_by_radius", show, plt
    )

    fig, axis = plt.subplots(figsize=(9.0, 5.8))
    x = np.arange(3)
    width = 0.35
    full = [
        np.median(values) if len(values) else math.nan
        for values in eligible_method_values(path_rows, "rms_jerk_rad_s3")
    ]
    interior = [
        np.median(values) if len(values) else math.nan
        for values in eligible_method_values(
            path_rows, "interior_rms_jerk_rad_s3"
        )
    ]
    axis.bar(x - width / 2, full, width, label="Full")
    axis.bar(x + width / 2, interior, width, label="Interior")
    axis.set_xticks(x, labels)
    axis.set_ylabel("Median path RMS jerk (rad/s³)")
    axis.set_title("Full versus endpoint-excluded jerk")
    axis.legend()
    axis.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    save_figure(fig, output, "full_vs_interior_rms_jerk", show, plt)

    fig, axis = plt.subplots(figsize=(9.0, 5.8))
    boundary = [
        np.median(values) if len(values) else math.nan
        for values in eligible_method_values(
            path_rows, "boundary_jerk_energy_density_rad2_s6"
        )
    ]
    outside = [
        np.median(values) if len(values) else math.nan
        for values in eligible_method_values(
            path_rows, "nonboundary_jerk_energy_density_rad2_s6"
        )
    ]
    axis.bar(x - width / 2, boundary, width, label="Boundary-mask density")
    axis.bar(x + width / 2, outside, width, label="Nonboundary density")
    axis.set_xticks(x, labels)
    axis.set_ylabel("Median jerk-energy density (rad²/s⁶)")
    axis.set_title(
        "Boundary versus nonboundary energy density; IK/MLP use control indices"
    )
    axis.legend()
    axis.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    save_figure(fig, output, "boundary_vs_nonboundary_jerk", show, plt)

    def draw_accuracy_smoothness_scatter(
        target_axis: Any, *, include_legend_labels: bool
    ) -> None:
        """Draw identical method and eligibility encodings on either axis."""
        for method, color in zip(METHODS, colors):
            method_rows = [row for row in path_rows if row["method"] == method]
            for eligible in (True, False):
                subset = [
                    row
                    for row in method_rows
                    if bool(row.get("smoothness_claim_eligible")) == eligible
                ]
                if not subset:
                    continue
                target_axis.scatter(
                    [float(row["rms_cartesian_error_m"]) for row in subset],
                    [float(row["rms_jerk_rad_s3"]) for row in subset],
                    color=color,
                    marker=MarkerStyle("o" if eligible else "x"),
                    alpha=0.85,
                    s=42,
                    label=(
                        f"{METHOD_LABELS[method]} "
                        f"({'eligible' if eligible else 'ineligible'})"
                        if include_legend_labels
                        else None
                    ),
                )

    fig, axis = plt.subplots(figsize=(10.5, 6.4))
    draw_accuracy_smoothness_scatter(axis, include_legend_labels=True)
    axis.set_xlabel("Cartesian RMS error (m)")
    axis.set_ylabel("RMS joint jerk (rad/s³)")
    axis.set_title(
        "Eligible trajectories satisfy both mean- and maximum-error criteria",
        fontsize=10,
        pad=8,
    )
    fig.suptitle(
        "Cartesian Tracking Accuracy–Joint Smoothness Trade-off",
        fontsize=15,
        fontweight="semibold",
    )
    axis.grid(True, alpha=0.25)
    axis.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        fontsize=8,
        frameon=True,
    )
    fig.subplots_adjust(left=0.09, right=0.78, bottom=0.11, top=0.84)

    inset = axis.inset_axes([0.06, 0.52, 0.43, 0.42])
    draw_accuracy_smoothness_scatter(inset, include_legend_labels=False)
    inset.set_xlim(0.0, 0.015)
    low_error_jerk = np.asarray(
        [
            float(row["rms_jerk_rad_s3"])
            for row in path_rows
            if 0.0 <= float(row.get("rms_cartesian_error_m", math.inf)) <= 0.015
            and math.isfinite(float(row.get("rms_jerk_rad_s3", math.nan)))
        ],
        dtype=np.float64,
    )
    main_y_limits = axis.get_ylim()
    if (
        main_y_limits[1] > 6.0
        and low_error_jerk.size
        and float(np.mean(low_error_jerk <= 6.0)) >= 0.8
    ):
        inset.set_ylim(0.0, 6.0)
    else:
        inset.set_ylim(float(main_y_limits[0]), float(main_y_limits[1]))
    inset.set_title("High-accuracy region", fontsize=9)
    inset.tick_params(labelsize=8)
    inset.grid(True, alpha=0.2)
    axis.indicate_inset_zoom(
        inset,
        edgecolor="0.45",
        linewidth=0.8,
    )

    save_figure(fig, output, "cartesian_error_vs_rms_jerk", show, plt)


def plot_seed_variability(
    output: Path,
    trajectory_rows: Sequence[Mapping[str, Any]],
    show: bool,
) -> None:
    """Plot every eligible pipeline seed grouped by path."""
    import matplotlib.pyplot as plt

    rows = [
        row
        for row in trajectory_rows
        if row["method"] == "pipeline"
        and bool(row.get("smoothness_claim_eligible"))
    ]
    path_ids = sorted({str(row["path_id"]) for row in rows})
    fig, axis = plt.subplots(figsize=(max(9.0, len(path_ids) * 0.42), 5.8))
    for index, path_id in enumerate(path_ids):
        values = [
            float(row["rms_jerk_rad_s3"])
            for row in rows
            if row["path_id"] == path_id
        ]
        axis.scatter(np.full(len(values), index), values, color="#009E73", alpha=0.8)
        if values:
            axis.plot(
                [index - 0.2, index + 0.2],
                [np.median(values)] * 2,
                color="black",
                linewidth=1.4,
            )
    axis.set_xticks(np.arange(len(path_ids)), path_ids, rotation=60, ha="right")
    axis.set_ylabel("Pipeline RMS jerk (rad/s³)")
    axis.set_title("Pipeline diffusion-seed variability; line = seed median")
    axis.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    save_figure(fig, output, "pipeline_seed_rms_jerk_variability", show, plt)


def plot_win_rates(output: Path, rows: Sequence[Mapping[str, Any]], show: bool) -> None:
    """Plot pipeline path-level win rates versus IK and MLP."""
    import matplotlib.pyplot as plt

    metrics = list(PRIMARY_METRICS)
    labels = [
        "RMS acc.",
        "RMS jerk",
        "IS jerk",
        "Max jerk",
        "HF ratio",
        "Boundary\nenrichment",
    ]
    fig, axis = plt.subplots(figsize=(11.0, 5.5))
    x = np.arange(len(metrics))
    width = 0.35
    for offset, reference, color in (
        (-width / 2, "ik", "#0072B2"),
        (width / 2, "mlp", "#D55E00"),
    ):
        values = []
        for metric in metrics:
            match = next(
                (
                    row
                    for row in rows
                    if row["reference_method"] == reference
                    and row["comparison_method"] == "pipeline"
                    and row["metric"] == metric
                ),
                None,
            )
            values.append(
                float(match["comparison_win_rate"]) if match else math.nan
            )
        axis.bar(
            x + offset,
            values,
            width,
            label=f"Pipeline vs {METHOD_LABELS[reference]}",
            color=color,
        )
    axis.set_xticks(x, labels)
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Path-level pipeline win rate")
    axis.set_title(
        "Lower-is-better metrics; boundary enrichment is a stitching diagnostic"
    )
    axis.grid(True, axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    save_figure(fig, output, "smoothness_win_rates", show, plt)


def representative_paths(
    path_rows: Sequence[Mapping[str, Any]],
) -> dict[str, str]:
    """Select strongest, median, and worst pipeline-vs-IK RMS-jerk changes."""
    lookup = {
        (str(row["path_id"]), str(row["method"])): row
        for row in path_rows
        if bool(row.get("smoothness_claim_eligible"))
    }
    changes: list[tuple[str, float]] = []
    for path_id, method in lookup:
        if method != "pipeline" or (path_id, "ik") not in lookup:
            continue
        reference = float(lookup[(path_id, "ik")]["rms_jerk_rad_s3"])
        candidate = float(lookup[(path_id, "pipeline")]["rms_jerk_rad_s3"])
        changes.append(
            (path_id, comparison.percent_improvement(reference, candidate))
        )
    if not changes:
        return {}
    ordered = sorted(changes, key=lambda item: item[1], reverse=True)
    median_value = float(np.median([value for _, value in changes]))
    return {
        "best": ordered[0][0],
        "median": min(changes, key=lambda item: abs(item[1] - median_value))[0],
        "worst": ordered[-1][0],
    }


def plot_representatives(
    output: Path,
    selected: Mapping[str, str],
    series: Mapping[tuple[str, str, int | None], EvaluatedTrajectory],
    trajectory_rows: Sequence[Mapping[str, Any]],
    show: bool,
) -> dict[str, Any]:
    """Plot all methods for representative paths using median-nearest pipeline seed."""
    import matplotlib.pyplot as plt

    metadata: dict[str, Any] = {}
    for category, path_id in selected.items():
        pipeline_rows = [
            row
            for row in trajectory_rows
            if row["path_id"] == path_id
            and row["method"] == "pipeline"
            and bool(row.get("smoothness_claim_eligible"))
        ]
        median = float(np.median([float(row["rms_jerk_rad_s3"]) for row in pipeline_rows]))
        pipeline_row = min(
            pipeline_rows,
            key=lambda row: abs(float(row["rms_jerk_rad_s3"]) - median),
        )
        seed = int(pipeline_row["seed"])
        selected_series = {
            "ik": series[(path_id, "ik", None)],
            "mlp": series[(path_id, "mlp", None)],
            "pipeline": series[(path_id, "pipeline", seed)],
        }
        metadata[category] = {"path_id": path_id, "pipeline_seed": seed}
        for name, attribute, ylabel in (
            ("joint_jerk_over_time", "jerk", "Jerk (rad/s³)"),
            (
                "joint_acceleration_over_time",
                "acceleration",
                "Acceleration (rad/s²)",
            ),
            ("joint_positions_over_time", "q", "Position (rad)"),
        ):
            fig, axes = plt.subplots(6, 1, figsize=(10.0, 12.5), sharex=True)
            for joint, axis in enumerate(axes):
                for method, color in zip(METHODS, ("#0072B2", "#D55E00", "#009E73")):
                    item = selected_series[method]
                    axis.plot(
                        item.timestamps,
                        getattr(item, attribute)[:, joint],
                        color=color,
                        label=METHOD_LABELS[method],
                        linewidth=1.1,
                    )
                axis.set_ylabel(f"q{joint + 1}\n{ylabel}")
                axis.grid(True, alpha=0.2)
            axes[0].legend(ncol=3)
            axes[-1].set_xlabel("Time (s)")
            fig.suptitle(
                f"{category.title()} representative: {path_id}; "
                f"pipeline seed {seed}"
            )
            fig.tight_layout(rect=(0, 0, 1, 0.98))
            save_figure(
                fig,
                output,
                f"representative_{category}_{name}",
                show,
                plt,
            )
    return metadata


def thesis_summary(
    path_rows: Sequence[Mapping[str, Any]],
    wins: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Build the compact thesis-facing summary table."""
    rows: list[dict[str, Any]] = []

    def values(method: str, metric: str) -> np.ndarray:
        return np.asarray(
            [
                float(row[metric])
                for row in path_rows
                if row["method"] == method
                and bool(row.get("smoothness_claim_eligible"))
                and math.isfinite(float(row.get(metric, math.nan)))
            ]
        )

    def median(method: str, metric: str) -> float:
        array = values(method, metric)
        return float(np.median(array)) if len(array) else math.nan

    def iqr(method: str, metric: str) -> float:
        array = values(method, metric)
        return (
            float(np.quantile(array, 0.75) - np.quantile(array, 0.25))
            if len(array)
            else math.nan
        )

    def win(method: str, reference: str) -> float | str:
        if method == reference:
            return ""
        row = next(
            (
                item
                for item in wins
                if item["reference_method"] == reference
                and item["comparison_method"] == method
                and item["metric"] == "rms_jerk_rad_s3"
            ),
            None,
        )
        if row:
            return float(row["comparison_win_rate"])
        reverse = next(
            (
                item
                for item in wins
                if item["reference_method"] == method
                and item["comparison_method"] == reference
                and item["metric"] == "rms_jerk_rad_s3"
            ),
            None,
        )
        if reverse and int(reverse["matched_path_count"]) > 0:
            return float(reverse["reference_wins"]) / int(
                reverse["matched_path_count"]
            )
        return math.nan

    for method in METHODS:
        eligible_paths = {
            str(row["path_id"])
            for row in path_rows
            if row["method"] == method
            and bool(row.get("smoothness_claim_eligible"))
        }
        rows.append(
            {
                "method": method,
                "eligible_paths": len(eligible_paths),
                "median_rms_acceleration": median(
                    method, "rms_acceleration_rad_s2"
                ),
                "iqr_rms_acceleration": iqr(
                    method, "rms_acceleration_rad_s2"
                ),
                "median_rms_jerk": median(method, "rms_jerk_rad_s3"),
                "iqr_rms_jerk": iqr(method, "rms_jerk_rad_s3"),
                "median_integrated_squared_jerk": median(
                    method, "integrated_squared_jerk_rad2_s5"
                ),
                "iqr_integrated_squared_jerk": iqr(
                    method, "integrated_squared_jerk_rad2_s5"
                ),
                "median_max_abs_jerk": median(
                    method, "max_abs_jerk_rad_s3"
                ),
                "median_interior_rms_jerk": median(
                    method, "interior_rms_jerk_rad_s3"
                ),
                "median_boundary_jerk_energy_fraction": median(
                    method, "boundary_jerk_energy_fraction"
                ),
                "median_boundary_time_fraction": median(
                    method, "boundary_time_fraction"
                ),
                "median_boundary_jerk_energy_enrichment": median(
                    method, "boundary_jerk_energy_enrichment"
                ),
                "median_boundary_to_nonboundary_energy_density_ratio": median(
                    method, "boundary_to_nonboundary_energy_density_ratio"
                ),
                "median_high_frequency_energy_ratio": median(
                    method, "mean_high_frequency_energy_ratio"
                ),
                "median_cartesian_rms_error_m": median(
                    method, "rms_cartesian_error_m"
                ),
                "smoothness_win_rate_vs_ik": win(method, "ik"),
                "smoothness_win_rate_vs_mlp": win(method, "mlp"),
            }
        )
    return rows


def repository_commit() -> str | None:
    """Return Git HEAD when available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def json_safe(value: Any) -> Any:
    """Convert arrays and nonfinite floats to strict JSON values."""
    if isinstance(value, np.ndarray):
        return [json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def record_failure(
    args: argparse.Namespace,
    excluded_paths: list[dict[str, Any]],
    path_id: str,
    reason: str,
) -> None:
    """Record or raise one path-level discovery/evaluation failure."""
    excluded_paths.append({"path_id": path_id, "reason": reason})
    if args.strict:
        raise BenchmarkError(f"{path_id}: {reason}")


def run_benchmark(
    args: argparse.Namespace,
    *,
    robot_context: Any | None = None,
) -> dict[str, Any]:
    """Discover, evaluate, aggregate, and write the saved-artifact benchmark."""
    if args.orientation_error_threshold_rad is None:
        args.orientation_error_threshold_rad = DEFAULT_ORIENTATION_THRESHOLD_RAD
        orientation_source = DEFAULT_ORIENTATION_THRESHOLD_SOURCE
    else:
        orientation_source = "command_line"
    validate_args(args)
    mean_cartesian_source, max_cartesian_source = resolve_cartesian_thresholds(
        args
    )
    assert args.mean_cartesian_error_threshold_m is not None
    assert args.max_cartesian_error_threshold_m is not None
    requested = discover_requested_paths(args)
    if not requested:
        raise BenchmarkError("No requested or discoverable matched paths")
    pipeline_index = discover_pipeline_artifacts(
        args.pipeline_root,
        args.pipeline_seed_glob,
        args.pipeline_seeds,
    )
    output = prepare_output(args.output_dir, args.overwrite)
    context = robot_context if robot_context is not None else make_robot_context()

    trajectory_rows: list[dict[str, Any]] = []
    path_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []
    best_seed_rows: list[dict[str, Any]] = []
    excluded_trajectories: list[dict[str, Any]] = []
    excluded_paths: list[dict[str, Any]] = []
    series: dict[tuple[str, str, int | None], EvaluatedTrajectory] = {}
    evaluated_paths: list[str] = []
    seeds_discovered: dict[str, list[int]] = {}
    seeds_accepted: dict[str, list[int]] = {}

    for path_id in requested:
        ik_path = args.ik_root / path_id
        mlp_path = args.mlp_root / path_id
        artifacts = pipeline_index.get(path_id, [])
        seeds_discovered[path_id] = [item.seed for item in artifacts]
        seeds_accepted[path_id] = [item.seed for item in artifacts if item.accepted]
        missing = [
            label
            for label, path in (("IK", ik_path), ("MLP", mlp_path))
            if not path.is_dir()
        ]
        if not artifacts:
            missing.append("pipeline")
        if missing:
            record_failure(
                args,
                excluded_paths,
                path_id,
                "missing inputs: " + ",".join(missing),
            )
            continue
        try:
            ik = load_method_trajectory(ik_path, "ik")
            mlp = load_method_trajectory(mlp_path, "mlp")
        except Exception as exc:
            record_failure(args, excluded_paths, path_id, f"deterministic load: {exc}")
            continue

        loaded_pipeline: list[tuple[PipelineArtifact, Any, dict[str, Any]]] = []
        for artifact in artifacts:
            try:
                item = load_pipeline_trajectory(artifact)
                compatibility = comparison.validate_path_compatibility(
                    {"ik": ik, "mlp": mlp, "pipeline": item},
                    path_id=path_id,
                )
                verification = verify_targets(path_id, (ik, mlp), (item,))
                verification["verification_sources"] += (
                    ";compare_ik_mlp_pipeline_jerk_over_time."
                    "validate_path_compatibility:"
                    + json.dumps(compatibility, sort_keys=True)
                )
                loaded_pipeline.append((artifact, item, verification))
            except Exception as exc:
                trajectory_rows.append(
                    {
                        "path_id": path_id,
                        "method": "pipeline",
                        "seed": artifact.seed,
                        "selected_file": str(artifact.path),
                        "accepted": artifact.accepted,
                        "trajectory_valid": False,
                        "tracking_eligible": False,
                        "smoothness_claim_eligible": False,
                        "exclusion_reason": str(exc),
                    }
                )
                excluded_trajectories.append(
                    {
                        "path_id": path_id,
                        "method": "pipeline",
                        "seed": artifact.seed,
                        "selected_file": str(artifact.path),
                        "reason": str(exc),
                    }
                )
                if args.strict:
                    raise BenchmarkError(
                        f"{path_id} seed {artifact.seed}: {exc}"
                    ) from exc
        accepted_loaded = [
            entry for entry in loaded_pipeline if entry[0].accepted
        ]
        if (
            args.require_accepted_pipeline
            and len(accepted_loaded) < args.minimum_pipeline_seeds
        ):
            record_failure(
                args,
                excluded_paths,
                path_id,
                "insufficient accepted pipeline seeds: "
                f"{len(accepted_loaded)} < {args.minimum_pipeline_seeds}",
            )
        if not loaded_pipeline:
            continue

        source_target = target_from_trajectory(ik)
        target_rotation = target_rotation_from_items(
            [ik, mlp, *[item for _, item, _ in loaded_pipeline]]
        )
        deterministic_evaluated = False
        pipeline_evaluated: list[EvaluatedTrajectory] = []
        for artifact, pipeline_item, verification in loaded_pipeline:
            try:
                aligned = comparison.align_trajectories(
                    {"ik": ik, "mlp": mlp, "pipeline": pipeline_item},
                    timing_policy=args.timing_policy,
                    common_duration_s=(
                        args.common_duration_s
                        if args.timing_policy == "common_duration"
                        else None
                    ),
                    common_samples=(
                        args.common_samples
                        if args.timing_policy == "common_duration"
                        else None
                    ),
                )
                if not aligned.complete_trajectory_used:
                    raise BenchmarkError("timing policy cropped a trajectory")
                target = resample_target(source_target, len(aligned.timestamps))
                if not deterministic_evaluated:
                    for method, item in (("ik", ik), ("mlp", mlp)):
                        evaluated = evaluate_aligned(
                            path_id=path_id,
                            method=method,
                            seed=None,
                            selected_file=item.selected_file,
                            accepted=True,
                            acceptance_source="deterministic saved method trajectory",
                            runtime_s=math.nan,
                            q=aligned.q[method],
                            timestamps=aligned.timestamps,
                            target=target,
                            context=context,
                            args=args,
                            verification=verification,
                            target_rotation=target_rotation,
                        )
                        trajectory_rows.append(evaluated.row)
                        series[(path_id, method, None)] = evaluated
                        if not evaluated.row["smoothness_claim_eligible"]:
                            excluded_trajectories.append(
                                {
                                    "path_id": path_id,
                                    "method": method,
                                    "seed": "",
                                    "selected_file": str(item.selected_file),
                                    "reason": evaluated.row["exclusion_reason"],
                                }
                            )
                    deterministic_evaluated = True
                evaluated_pipeline = evaluate_aligned(
                    path_id=path_id,
                    method="pipeline",
                    seed=artifact.seed,
                    selected_file=artifact.path,
                    accepted=artifact.accepted,
                    acceptance_source=artifact.acceptance_source,
                    runtime_s=artifact.runtime_s,
                    q=aligned.q["pipeline"],
                    timestamps=aligned.timestamps,
                    target=target,
                    context=context,
                    args=args,
                    verification=verification,
                    target_rotation=target_rotation,
                )
                pipeline_evaluated.append(evaluated_pipeline)
                trajectory_rows.append(evaluated_pipeline.row)
                series[(path_id, "pipeline", artifact.seed)] = evaluated_pipeline
                if not evaluated_pipeline.row["smoothness_claim_eligible"]:
                    excluded_trajectories.append(
                        {
                            "path_id": path_id,
                            "method": "pipeline",
                            "seed": artifact.seed,
                            "selected_file": str(artifact.path),
                            "reason": evaluated_pipeline.row["exclusion_reason"],
                        }
                    )
            except Exception as exc:
                if not any(
                    row.get("path_id") == path_id
                    and row.get("method") == "pipeline"
                    and row.get("seed") == artifact.seed
                    for row in trajectory_rows
                ):
                    trajectory_rows.append(
                        {
                            "path_id": path_id,
                            "method": "pipeline",
                            "seed": artifact.seed,
                            "selected_file": str(artifact.path),
                            "accepted": artifact.accepted,
                            "trajectory_valid": False,
                            "tracking_eligible": False,
                            "smoothness_claim_eligible": False,
                            "exclusion_reason": str(exc),
                        }
                    )
                excluded_trajectories.append(
                    {
                        "path_id": path_id,
                        "method": "pipeline",
                        "seed": artifact.seed,
                        "selected_file": str(artifact.path),
                        "reason": str(exc),
                    }
                )
                if args.strict:
                    raise BenchmarkError(
                        f"{path_id} seed {artifact.seed}: {exc}"
                    ) from exc

        pipeline_primary, path_selection, best_seed = aggregate_pipeline_path(
            path_id, [item.row for item in pipeline_evaluated]
        )
        deterministic_rows = [
            row
            for row in trajectory_rows
            if row["path_id"] == path_id and row["method"] in ("ik", "mlp")
        ]
        eligible_seed_count = sum(
            bool(item.row["smoothness_claim_eligible"])
            for item in pipeline_evaluated
        )
        if pipeline_primary is None or eligible_seed_count < args.minimum_pipeline_seeds:
            record_failure(
                args,
                excluded_paths,
                path_id,
                "insufficient tracking-eligible accepted pipeline seeds: "
                f"{eligible_seed_count} < {args.minimum_pipeline_seeds}",
            )
            continue
        path_rows.extend(deterministic_path_row(row) for row in deterministic_rows)
        path_rows.append(pipeline_primary)
        selection_rows.extend(path_selection)
        if best_seed is not None:
            best_seed_rows.append(best_seed)
        evaluated_paths.append(path_id)

    if not evaluated_paths:
        raise BenchmarkError("No paths had sufficient eligible matched trajectories")

    summaries = method_summaries(path_rows)
    wins = win_rates(path_rows)
    statistical = statistical_tests(path_rows)
    thesis = thesis_summary(path_rows, wins)
    selected_representatives = representative_paths(path_rows)

    write_csv(output / "per_trajectory_smoothness.csv", trajectory_rows)
    write_csv(output / "per_path_method_smoothness.csv", path_rows)
    write_csv(output / "pipeline_seed_selection.csv", selection_rows)
    write_csv(output / "pipeline_best_seed_smoothness.csv", best_seed_rows)
    write_csv(output / "method_smoothness_summary.csv", summaries)
    write_csv(output / "method_win_rates.csv", wins)
    write_csv(output / "path_level_statistical_tests.csv", statistical)
    write_csv(output / "thesis_smoothness_summary.csv", thesis)
    write_csv(output / "excluded_trajectories.csv", excluded_trajectories)
    write_csv(output / "excluded_paths.csv", excluded_paths)

    plot_distributions(output, path_rows, args, args.show)
    plot_seed_variability(output, trajectory_rows, args.show)
    plot_win_rates(output, wins, args.show)
    representative_metadata = plot_representatives(
        output,
        selected_representatives,
        series,
        trajectory_rows,
        args.show,
    )
    metadata = {
        "analysis": "multi-path multi-seed IK/MLP/pipeline smoothness benchmark",
        "repository_commit": repository_commit(),
        "command": " ".join(shlex.quote(value) for value in sys.argv),
        "creation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "roots": {
            "ik": str(args.ik_root.resolve()),
            "mlp": str(args.mlp_root.resolve()),
            "pipeline": str(args.pipeline_root.resolve()),
        },
        "path_discovery_rules": {
            "path_glob": args.path_glob,
            "pipeline_seed_glob": args.pipeline_seed_glob,
            "pipeline_artifact": (
                "deployment_trajectory_full.npz:final_q or "
                "trajectories/*/anchored_rollout_k8.npz:rollout_q with "
                "anchored_full_path_metrics full_path_safety_pass"
            ),
        },
        "path_ids_requested": requested,
        "paths_discovered": sorted(
            path_id
            for path_id in requested
            if (args.ik_root / path_id).is_dir()
            and (args.mlp_root / path_id).is_dir()
            and path_id in pipeline_index
        ),
        "paths_evaluated": evaluated_paths,
        "paths_skipped": excluded_paths,
        "exclusion_reasons": {
            "paths": excluded_paths,
            "trajectories": excluded_trajectories,
        },
        "seeds_discovered_per_path": seeds_discovered,
        "seeds_accepted_per_path": seeds_accepted,
        "timing_policy": args.timing_policy,
        "claim_eligible": args.timing_policy != "shared_interval_diagnostic",
        "complete_trajectory_required": True,
        "common_duration_s": args.common_duration_s,
        "common_samples": args.common_samples,
        "derivative_implementation": (
            "analyze_prior_vs_diffusion_contribution_v8_1.derivatives; "
            "three successive numpy.gradient(..., timestamps, axis=0, edge_order=2)"
        ),
        "endpoint_exclusion": args.endpoint_exclusion,
        "execution_horizon": args.execution_horizon,
        "boundary_radius": args.boundary_radius,
        "boundary_primary_radius": args.boundary_radius,
        "boundary_sensitivity_radii": args.boundary_sensitivity_radii,
        "boundary_energy_normalization": (
            "integrated squared jerk fraction divided by trapezoidal "
            "boundary-time fraction"
        ),
        "boundary_enrichment_interpretation": (
            ">1 concentrated near boundaries; approximately 1 proportional to "
            "covered time; <1 depleted near boundaries"
        ),
        "spectral_method": "mean-removed real FFT squared magnitude",
        "spectral_signal": args.spectral_signal,
        "high_frequency_fraction": args.high_frequency_fraction,
        "spectral_frequency_cutoff_hz": next(
            (
                float(row["spectral_frequency_cutoff_hz"])
                for row in trajectory_rows
                if row.get("trajectory_valid")
                and math.isfinite(
                    float(row.get("spectral_frequency_cutoff_hz", math.nan))
                )
            ),
            None,
        ),
        "mean_cartesian_error_threshold_m": (
            args.mean_cartesian_error_threshold_m
        ),
        "max_cartesian_error_threshold_m": args.max_cartesian_error_threshold_m,
        "mean_cartesian_threshold_source": mean_cartesian_source,
        "max_cartesian_threshold_source": max_cartesian_source,
        "orientation_error_threshold_rad": args.orientation_error_threshold_rad,
        "orientation_threshold_source": orientation_source,
        "fk_implementation": (
            "evaluate_diffusion_v7_teacher_forced_validation.make_robot_context "
            "+ orientation_aware_adaptive_ik.trajectory_full_transform_fk"
        ),
        "statistical_methods": (
            "path-level Friedman, Wilcoxon signed-rank, deterministic paired "
            "bootstrap, Holm-Bonferroni"
        ),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        "claim_eligibility_rules": (
            "finite/malformed validation, accepted pipeline when required, "
            "mean and maximum Cartesian FK error within threshold, orientation "
            "threshold when a target orientation exists, complete claim-eligible timing"
        ),
        "pipeline_primary_aggregation": (
            "median across eligible accepted diffusion seeds per complete path"
        ),
        "pipeline_best_seed_analysis": (
            "secondary; one seed selected by minimum eligible RMS jerk and "
            "metric-specific best/worst seeds recorded"
        ),
        "representative_paths_and_seeds": representative_metadata,
        "warnings": [
            "Low jerk alone does not establish a superior trajectory if "
            "Cartesian tracking requirements are not satisfied.",
            "Primary pipeline path-level results use the median across eligible "
            "accepted diffusion seeds.",
            "Statistical unit is one complete path, never an individual sample.",
            "Boundary jerk-energy fraction must be interpreted relative to "
            "boundary time coverage. Enrichment near 1 indicates no concentration.",
            "Boundary enrichment below 1 does not necessarily imply a globally "
            "smoother trajectory; it only indicates reduced concentration near "
            "the chosen boundary indices.",
        ],
    }
    atomic_write_text(
        output / "benchmark_metadata.json",
        json.dumps(json_safe(metadata), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
    )
    return {
        "output_dir": output,
        "metadata": metadata,
        "trajectory_rows": trajectory_rows,
        "path_rows": path_rows,
        "selection_rows": selection_rows,
        "best_seed_rows": best_seed_rows,
        "summaries": summaries,
        "wins": wins,
        "statistical": statistical,
        "thesis": thesis,
        "excluded_paths": excluded_paths,
        "excluded_trajectories": excluded_trajectories,
    }


def print_summary(result: Mapping[str, Any]) -> None:
    """Print the required benchmark summary without overstating conclusions."""
    metadata = result["metadata"]
    trajectory_rows = result["trajectory_rows"]
    path_rows = result["path_rows"]
    thesis = {row["method"]: row for row in result["thesis"]}
    wins = result["wins"]

    def pipeline_win(reference: str) -> float:
        row = next(
            (
                value
                for value in wins
                if value["reference_method"] == reference
                and value["comparison_method"] == "pipeline"
                and value["metric"] == "rms_jerk_rad_s3"
            ),
            None,
        )
        return float(row["comparison_win_rate"]) if row else math.nan

    print(f"Requested paths: {len(metadata['path_ids_requested'])}")
    print(f"Discovered matched paths: {len(metadata['paths_discovered'])}")
    print(f"Evaluated paths: {len(metadata['paths_evaluated'])}")
    print(f"Excluded paths: {len(result['excluded_paths'])}")
    pipeline_rows = [row for row in trajectory_rows if row["method"] == "pipeline"]
    discovered_seed_count = sum(
        len(values)
        for values in metadata.get("seeds_discovered_per_path", {}).values()
    )
    accepted_seed_count = sum(
        len(values)
        for values in metadata.get("seeds_accepted_per_path", {}).values()
    )
    if not metadata.get("seeds_discovered_per_path"):
        discovered_seed_count = len(pipeline_rows)
    if not metadata.get("seeds_accepted_per_path"):
        accepted_seed_count = sum(
            bool(row.get("accepted")) for row in pipeline_rows
        )
    print(f"Pipeline seeds discovered: {discovered_seed_count}")
    print(f"Pipeline seeds accepted: {accepted_seed_count}")
    print(
        "Pipeline seeds tracking-eligible: "
        f"{sum(bool(row['tracking_eligible']) for row in pipeline_rows)}"
    )
    print(
        f"Common duration/samples: {metadata['common_duration_s']} s / "
        f"{metadata['common_samples']}"
    )
    print(
        "Mean Cartesian eligibility threshold: "
        f"{metadata['mean_cartesian_error_threshold_m']} m"
    )
    print(
        "Maximum Cartesian eligibility threshold: "
        f"{metadata['max_cartesian_error_threshold_m']} m"
    )
    for method in METHODS:
        print(
            f"{METHOD_LABELS[method]} median RMS jerk: "
            f"{thesis[method]['median_rms_jerk']}; median integrated squared jerk: "
            f"{thesis[method]['median_integrated_squared_jerk']}"
        )
    print(f"Pipeline RMS-jerk win rate vs IK: {pipeline_win('ik')}")
    print(f"Pipeline RMS-jerk win rate vs MLP: {pipeline_win('mlp')}")
    pipeline_path_rows = [
        row for row in path_rows if row["method"] == "pipeline"
    ]
    print(
        "Pipeline median boundary time fraction: "
        f"{np.median([float(row['boundary_time_fraction']) for row in pipeline_path_rows])}"
    )
    print(
        "Pipeline median boundary jerk-energy fraction: "
        f"{np.median([float(row['boundary_jerk_energy_fraction']) for row in pipeline_path_rows])}"
    )
    print(
        "Pipeline median boundary jerk-energy enrichment: "
        f"{np.median([float(row['boundary_jerk_energy_enrichment']) for row in pipeline_path_rows])}"
    )
    print(
        "Pipeline median boundary/nonboundary energy-density ratio: "
        f"{np.median([float(row['boundary_to_nonboundary_energy_density_ratio']) for row in pipeline_path_rows])}"
    )
    print(
        "Pipeline median high-frequency energy ratio: "
        f"{np.median([float(row['mean_high_frequency_energy_ratio']) for row in pipeline_path_rows])}"
    )
    print(
        "Low jerk alone does not establish a superior trajectory if Cartesian "
        "tracking requirements are not satisfied."
    )
    print(
        "Primary pipeline path-level results use the median across eligible "
        "accepted diffusion seeds."
    )
    print(
        "Boundary jerk-energy fraction must be interpreted relative to boundary "
        "time coverage. Enrichment near 1 indicates no concentration."
    )
    print(f"Output directory: {result['output_dir']}")
    print("IK_MLP_PIPELINE_SMOOTHNESS_BENCHMARK_COMPLETE")


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    try:
        args = parse_args(argv)
        result = run_benchmark(args)
        print_summary(result)
        return 0
    except BenchmarkError as exc:
        print(f"IK_MLP_PIPELINE_SMOOTHNESS_BENCHMARK_FAILED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
