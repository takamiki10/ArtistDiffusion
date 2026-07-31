#!/usr/bin/env python3
"""Compare time-domain joint jerk for saved IK, MLP, and pipeline trajectories.

This module is deliberately read-only with respect to its inputs.  It loads
already generated joint trajectories, verifies that they describe the same
path, aligns valid time grids when necessary, and writes descriptive
trajectory-level comparisons.  It never invokes trajectory generation,
inference, IK, scoring, or robot execution.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shlex
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import matplotlib
import numpy as np
from scipy.interpolate import CubicSpline

_INITIAL_MATPLOTLIB_BACKEND = str(matplotlib.get_backend())

JOINT_COUNT = 6
JOINT_NAMES = tuple(f"q{index}" for index in range(1, JOINT_COUNT + 1))
METHODS = ("ik", "mlp", "pipeline")
METHOD_LABELS = {
    "ik": "IK",
    "mlp": "MLP",
    "pipeline": "Proposed pipeline",
}
TIMESTAMP_KEYS = ("timestamps", "time_s", "time_seconds", "time", "t")
TIMESTAMP_COLUMNS = ("time_s", "time_seconds", "time", "t", "Timestamp")
TARGET_KEYS = ("desired_path", "target_path", "cartesian_path")
IDENTITY_KEYS = (
    "path_id",
    "path_name",
    "input_path_name",
    "deployment_path_id",
    "stroke_id",
    "stroke_index",
    "source_input",
    "input_file",
    "input_csv",
)
METADATA_FILENAMES = (
    "comparison_metadata.json",
    "deployment_metrics.json",
    "path_meta.json",
    "metrics.json",
    "evaluation_metadata.json",
)
DIRECTORY_CANDIDATE_TIERS = {
    "ik": (
        ("ik_seed_q.csv", "expert_q.csv"),
        ("ik_trajectory.npz", "ik_trajectory.npy"),
    ),
    "mlp": (
        ("predicted_q.csv", "canonical_mlp_q.csv", "path_conditioned_pred_q.csv"),
        ("mlp_trajectory.npz", "mlp_trajectory.npy"),
    ),
    "pipeline": (
        ("deployment_trajectory_full.npz",),
        ("approved_simulation_trajectory.npz",),
        ("deployment_joint_positions.csv",),
        ("approved_simulation_trajectory.csv",),
        ("selected_trajectory.npz", "final_trajectory.npz"),
    ),
}
ARRAY_KEY_TIERS = {
    "ik": (
        ("ik_q", "expert_q"),
        ("q", "joint_positions", "trajectory"),
    ),
    "mlp": (
        ("canonical_mlp_q", "mlp_q", "q_pred", "predicted_q"),
        ("q", "joint_positions", "trajectory"),
    ),
    "pipeline": (
        ("final_q",),
        ("selected_q", "rollout_q"),
        ("q", "joint_positions", "trajectory"),
    ),
}
TIME_RTOL = 1.0e-7
TIME_ATOL_S = 1.0e-9
TARGET_RTOL = 1.0e-6
TARGET_ATOL_M = 5.0e-8


class ComparisonError(RuntimeError):
    """Raised when the requested comparison cannot be performed faithfully."""


@dataclass
class Trajectory:
    """A validated saved trajectory and its discovered provenance."""

    method: str
    provided_path: Path
    selected_file: Path
    selected_array_key: str | None
    original_shape: tuple[int, ...]
    q: np.ndarray
    timestamps: np.ndarray
    timestamp_source: str
    metadata: dict[str, Any] = field(default_factory=dict)
    identities: set[str] = field(default_factory=set)
    desired_path: np.ndarray | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def duration_s(self) -> float:
        return float(self.timestamps[-1] - self.timestamps[0])


@dataclass
class AlignedData:
    """Three trajectories represented on one shared time grid."""

    timestamps: np.ndarray
    progress: np.ndarray
    q: dict[str, np.ndarray]
    timing_policy: str
    claim_eligible: bool
    original_timing_preserved: bool
    complete_trajectory_used: bool
    common_duration_s: float
    common_sample_count: int
    progress_grid_source: str
    progress_interpolation_used: bool
    shared_interval_crop_used: bool
    duration_standardized: bool
    timing_methodology: str
    method_alignment: dict[str, dict[str, Any]]
    interpolation_method: str
    interpolation_used: bool
    warnings: list[str]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the comparison command-line interface."""
    parser = argparse.ArgumentParser(
        description="Compare saved IK, MLP, and proposed-pipeline joint jerk."
    )
    parser.add_argument("--ik_path", type=Path, required=True)
    parser.add_argument("--mlp_path", type=Path, required=True)
    parser.add_argument("--pipeline_path", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--dt", type=float)
    parser.add_argument(
        "--timing_policy",
        choices=(
            "require_equal",
            "common_duration",
            "shared_interval_diagnostic",
        ),
        default="require_equal",
    )
    parser.add_argument("--common_duration_s", type=float)
    parser.add_argument("--common_samples", type=int)
    parser.add_argument("--inspect_only", action="store_true")
    parser.add_argument("--path_id")
    parser.add_argument("--stroke_id")
    parser.add_argument("--title")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--smoothing",
        choices=("none",),
        default="none",
        help="No smoothing is the repository trajectory-scoring convention.",
    )
    return parser.parse_args(argv)


def _json_value(value: Any) -> Any:
    """Convert NumPy values to strict JSON-compatible Python values."""
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return _json_value(value.item())
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set)):
        return [_json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def scalar_text(value: Any) -> str | None:
    """Return text for a scalar metadata value, or ``None`` for non-scalars."""
    array = np.asarray(value)
    if array.size != 1:
        return None
    item = array.reshape(-1)[0]
    if isinstance(item, bytes):
        return item.decode("utf-8")
    return str(item)


def sha256_file(path: Path) -> str:
    """Return the SHA-256 of an input file without modifying it."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json_object(path: Path) -> dict[str, Any]:
    """Read one JSON object with a useful source-specific error."""
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ComparisonError(f"Cannot read metadata JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ComparisonError(f"Metadata file {path} must contain a JSON object")
    return value


def directory_candidates(directory: Path, method: str) -> Path:
    """Select a trajectory file using existing method-specific conventions."""
    for tier in DIRECTORY_CANDIDATE_TIERS[method]:
        matches = [directory / name for name in tier if (directory / name).is_file()]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            names = ", ".join(str(path) for path in matches)
            raise ComparisonError(
                f"{method}: ambiguous trajectory files at the same convention "
                f"priority in {directory}: {names}; pass one file directly"
            )

    supported = sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in {".npz", ".npy", ".csv"}
    )
    if len(supported) == 1:
        return supported[0]
    if not supported:
        raise ComparisonError(
            f"{method}: no supported saved trajectory file found in {directory}"
        )
    raise ComparisonError(
        f"{method}: no recognized final-trajectory convention and multiple "
        f"supported files exist in {directory}: "
        + ", ".join(path.name for path in supported)
    )


def resolve_trajectory_file(path: Path, method: str) -> tuple[Path, Path]:
    """Resolve a direct file or result directory to one exact source file."""
    provided = path.expanduser().resolve()
    if not provided.exists():
        raise ComparisonError(f"{method}: input path does not exist: {provided}")
    if provided.is_dir():
        return provided, directory_candidates(provided, method)
    if not provided.is_file():
        raise ComparisonError(f"{method}: input is not a regular file: {provided}")
    if provided.suffix.lower() not in {".npz", ".npy", ".csv"}:
        raise ComparisonError(
            f"{method}: unsupported trajectory extension {provided.suffix!r}"
        )
    return provided, provided


def select_array_key(data: Mapping[str, Any], method: str, source: Path) -> str:
    """Select one method-appropriate joint array from a saved NPZ."""
    for tier in ARRAY_KEY_TIERS[method]:
        matches = [key for key in tier if key in data]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise ComparisonError(
                f"{method}: {source} has ambiguous trajectory arrays at the "
                f"same priority: {matches}"
            )
    raise ComparisonError(
        f"{method}: {source} contains none of the accepted joint-position keys "
        f"{ARRAY_KEY_TIERS[method]}"
    )


def normalize_joint_array(value: Any, label: str) -> tuple[np.ndarray, tuple[int, ...]]:
    """Validate a `(T,6)` or `(6,T)` array and normalize it to `(T,6)`."""
    raw = np.asarray(value)
    original_shape = tuple(int(size) for size in raw.shape)
    if raw.ndim != 2:
        raise ComparisonError(
            f"{label} must be two-dimensional `(T,6)` or `(6,T)`, got {raw.shape}"
        )
    if raw.shape == (JOINT_COUNT, JOINT_COUNT):
        raise ComparisonError(
            f"{label} has ambiguous shape (6,6); sample and joint axes cannot "
            "be identified safely"
        )
    if raw.shape[1] == JOINT_COUNT:
        normalized = raw
    elif raw.shape[0] == JOINT_COUNT:
        normalized = raw.T
    else:
        raise ComparisonError(
            f"{label} must have exactly six active joints, got {raw.shape}"
        )
    if len(normalized) < 4:
        raise ComparisonError(
            f"{label} needs at least four samples for third derivatives"
        )
    if not np.issubdtype(normalized.dtype, np.number):
        raise ComparisonError(f"{label} must be numeric")
    q = np.asarray(normalized, dtype=np.float64)
    if not np.all(np.isfinite(q)):
        raise ComparisonError(f"{label} contains NaN or infinity")
    return np.array(q, copy=True), original_shape


def validate_timestamps(value: Any, sample_count: int, label: str) -> np.ndarray:
    """Validate finite, strictly increasing timestamps."""
    times = np.asarray(value, dtype=np.float64)
    if times.shape != (sample_count,):
        raise ComparisonError(
            f"{label} must have shape ({sample_count},), got {times.shape}"
        )
    if not np.all(np.isfinite(times)):
        raise ComparisonError(f"{label} contains NaN or infinity")
    if np.any(np.diff(times) <= 0.0):
        raise ComparisonError(f"{label} must be strictly increasing")
    return np.array(times, copy=True)


def extract_csv(path: Path, method: str) -> tuple[np.ndarray, tuple[int, ...], np.ndarray | None, str | None]:
    """Load a repository-style joint CSV without silently inferring columns."""
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            columns = reader.fieldnames
            if not columns:
                raise ComparisonError(f"{method}: CSV has no header: {path}")
            rows = list(reader)
    except OSError as exc:
        raise ComparisonError(f"Cannot read CSV {path}: {exc}") from exc
    if not rows:
        raise ComparisonError(f"{method}: CSV is empty: {path}")

    column_sets = (
        tuple(f"q{index}" for index in range(1, 7)),
        tuple(f"joint{index}" for index in range(1, 7)),
        tuple(f"final_q{index}" for index in range(1, 7)),
    )
    joint_columns = next(
        (candidate for candidate in column_sets if all(name in columns for name in candidate)),
        None,
    )
    if joint_columns is None:
        raise ComparisonError(
            f"{method}: {path} must contain q1..q6, joint1..joint6, or "
            "final_q1..final_q6 columns"
        )
    try:
        raw = np.asarray(
            [[float(row[name]) for name in joint_columns] for row in rows],
            dtype=np.float64,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ComparisonError(f"{method}: invalid joint value in {path}: {exc}") from exc
    q, shape = normalize_joint_array(raw, f"{method}:{path}")

    time_column = next((name for name in TIMESTAMP_COLUMNS if name in columns), None)
    timestamps = None
    if time_column is not None:
        try:
            timestamps = validate_timestamps(
                [float(row[time_column]) for row in rows],
                len(q),
                f"{method}:{path}:{time_column}",
            )
        except (TypeError, ValueError) as exc:
            raise ComparisonError(
                f"{method}: invalid timestamp in {path}:{time_column}: {exc}"
            ) from exc
    return q, shape, timestamps, time_column


def load_sidecar_metadata(selected: Path, root: Path) -> dict[str, Any]:
    """Load non-conflicting metadata sidecars next to a selected trajectory."""
    directory = root if root.is_dir() else selected.parent
    metadata: dict[str, Any] = {}
    for filename in METADATA_FILENAMES:
        candidate = directory / filename
        if candidate.is_file():
            metadata[filename] = read_json_object(candidate)
    return metadata


def flatten_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten nested source documents only for metadata lookup."""
    flattened: dict[str, Any] = {}

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            for key, item in value.items():
                flattened.setdefault(str(key), item)
                visit(item)

    visit(metadata)
    return flattened


def identity_tokens(value: Any) -> set[str]:
    """Extract stable path/stroke tokens from one metadata value."""
    text = scalar_text(value)
    if not text:
        return set()
    lowered = text.lower()
    tokens = set(re.findall(r"(?:path|stroke)[_-]?\d+", lowered))
    normalized = {
        re.sub(r"^(path|stroke)_?(\d+)$", r"\1_\2", token)
        for token in tokens
    }
    return normalized


def infer_identities(
    provided: Path,
    selected: Path,
    embedded: Mapping[str, Any],
    sidecars: Mapping[str, Any],
) -> set[str]:
    """Collect path/stroke identity tokens from files, directories, and metadata."""
    identities: set[str] = set()
    for part in (*selected.parts, *provided.parts):
        identities.update(identity_tokens(part))
    flat = flatten_metadata({"embedded": embedded, "sidecars": sidecars})
    for key in IDENTITY_KEYS:
        if key in flat:
            identities.update(identity_tokens(flat[key]))
    return identities


def discover_desired_path(
    selected: Path,
    provided: Path,
    embedded: Mapping[str, Any],
) -> np.ndarray | None:
    """Load an available Cartesian target solely for cross-method matching."""
    for key in TARGET_KEYS:
        if key in embedded:
            raw = np.asarray(embedded[key], dtype=np.float64)
            if raw.ndim == 2 and raw.shape[1] == 3 and np.all(np.isfinite(raw)):
                return np.array(raw, copy=True)
    directory = provided if provided.is_dir() else selected.parent
    candidate = directory / "desired_path.csv"
    if not candidate.is_file():
        return None
    try:
        with candidate.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or not all(
                name in reader.fieldnames for name in ("x", "y", "z")
            ):
                return None
            values = np.asarray(
                [[float(row[name]) for name in ("x", "y", "z")] for row in reader],
                dtype=np.float64,
            )
    except (OSError, TypeError, ValueError):
        return None
    if values.ndim != 2 or values.shape[1] != 3 or not np.all(np.isfinite(values)):
        return None
    return values


def metadata_timestep(metadata: Mapping[str, Any]) -> tuple[float | None, str | None]:
    """Find an explicitly stored positive sample interval in metadata."""
    flat = flatten_metadata(metadata)
    for key in ("dt", "timestep_s", "sample_interval_s", "sampling_interval_s"):
        if key not in flat:
            continue
        try:
            value = float(np.asarray(flat[key]).reshape(-1)[0])
        except (TypeError, ValueError, IndexError):
            continue
        if math.isfinite(value) and value > 0.0:
            return value, f"metadata:{key}"
    return None, None


def validate_pipeline_selection(
    selected: Path,
    embedded: Mapping[str, Any],
    sidecars: Mapping[str, Any],
) -> str:
    """Reject an explicitly recorded rejected pipeline result."""
    flat = flatten_metadata({"embedded": embedded, "sidecars": sidecars})
    accepted_value = flat.get("accepted")
    if accepted_value is not None:
        array = np.asarray(accepted_value)
        if array.size == 1 and not bool(array.reshape(-1)[0]):
            raise ComparisonError(
                f"pipeline: selected artifact is explicitly marked unaccepted: {selected}"
            )
    verdict = scalar_text(flat["verdict"]) if "verdict" in flat else None
    if verdict is not None:
        upper = verdict.upper()
        if "REJECT" in upper or "FAIL" in upper:
            raise ComparisonError(
                f"pipeline: selected artifact has rejected verdict {verdict!r}: "
                f"{selected}"
            )
        if "ACCEPT" in upper or "SELECT" in upper or "COMPLETE" in upper:
            return f"recorded verdict: {verdict}"
    if selected.name in {
        "deployment_trajectory_full.npz",
        "approved_simulation_trajectory.npz",
    }:
        raise ComparisonError(
            f"pipeline: canonical deployment artifact lacks an accepted verdict: {selected}"
        )
    return "directly supplied saved trajectory; no rejection marker recorded"


def load_trajectory(path: Path, method: str, dt_override: float | None = None) -> Trajectory:
    """Load and validate one existing method trajectory."""
    if method not in METHODS:
        raise ComparisonError(f"Unknown method {method!r}")
    provided, selected = resolve_trajectory_file(path, method)
    suffix = selected.suffix.lower()
    embedded: dict[str, Any] = {}
    key: str | None = None
    times: np.ndarray | None = None
    time_source: str | None = None

    if suffix == ".csv":
        q, original_shape, times, time_column = extract_csv(selected, method)
        if time_column:
            time_source = f"{selected}:{time_column}"
    elif suffix == ".npy":
        try:
            value = np.load(selected, allow_pickle=False)
        except (OSError, ValueError) as exc:
            raise ComparisonError(f"Cannot read NPY {selected}: {exc}") from exc
        q, original_shape = normalize_joint_array(value, f"{method}:{selected}")
    else:
        try:
            with np.load(selected, allow_pickle=False) as archive:
                embedded = {name: np.array(archive[name], copy=True) for name in archive.files}
        except (OSError, ValueError) as exc:
            raise ComparisonError(f"Cannot read NPZ {selected}: {exc}") from exc
        key = select_array_key(embedded, method, selected)
        q, original_shape = normalize_joint_array(
            embedded[key], f"{method}:{selected}:{key}"
        )
        for timestamp_key in TIMESTAMP_KEYS:
            if timestamp_key in embedded:
                times = validate_timestamps(
                    embedded[timestamp_key],
                    len(q),
                    f"{method}:{selected}:{timestamp_key}",
                )
                time_source = f"{selected}:{timestamp_key}"
                break

    sidecars = load_sidecar_metadata(selected, provided)
    metadata: dict[str, Any] = {
        "embedded": embedded,
        "sidecars": sidecars,
    }
    warnings: list[str] = []
    if method == "pipeline":
        metadata["pipeline_selection_evidence"] = validate_pipeline_selection(
            selected, embedded, sidecars
        )
    if dt_override is not None:
        if not math.isfinite(dt_override) or dt_override <= 0.0:
            raise ComparisonError("--dt must be finite and greater than zero")
        times = np.arange(len(q), dtype=np.float64) * dt_override
        time_source = "command_line:--dt"
    elif times is None:
        stored_dt, stored_source = metadata_timestep(metadata)
        if stored_dt is None:
            raise ComparisonError(
                f"{method}: no timestamps or saved sample interval found for "
                f"{selected}; supply --dt explicitly"
            )
        times = np.arange(len(q), dtype=np.float64) * stored_dt
        time_source = stored_source

    assert times is not None
    assert time_source is not None
    identities = infer_identities(provided, selected, embedded, sidecars)
    desired = discover_desired_path(selected, provided, embedded)
    return Trajectory(
        method=method,
        provided_path=provided,
        selected_file=selected,
        selected_array_key=key,
        original_shape=original_shape,
        q=q,
        timestamps=times,
        timestamp_source=time_source,
        metadata=metadata,
        identities=identities,
        desired_path=desired,
        warnings=warnings,
    )


def normalize_expected_identity(value: str | None) -> str | None:
    """Normalize a CLI identity to the same token form used for metadata."""
    if value is None:
        return None
    tokens = identity_tokens(value)
    if not tokens:
        normalized = value.strip().lower()
        if normalized:
            return normalized
    if len(tokens) != 1:
        raise ComparisonError(f"Cannot normalize path/stroke identifier {value!r}")
    return next(iter(tokens))


def validate_path_compatibility(
    trajectories: Mapping[str, Trajectory],
    path_id: str | None = None,
    stroke_id: str | None = None,
) -> dict[str, Any]:
    """Verify that all methods refer to the same Cartesian path or stroke."""
    expected = {
        token
        for token in (
            normalize_expected_identity(path_id),
            normalize_expected_identity(stroke_id),
        )
        if token is not None
    }
    identity_sets = {method: set(item.identities) for method, item in trajectories.items()}
    if expected:
        for method, identities in identity_sets.items():
            if identities and not expected.intersection(identities):
                raise ComparisonError(
                    f"{method}: discovered identities {sorted(identities)} do not "
                    f"match requested identity {sorted(expected)}"
                )

    informative = [values for values in identity_sets.values() if values]
    if len(informative) >= 2:
        common = set.intersection(*informative)
        if not common and not expected:
            raise ComparisonError(
                "Trajectory path metadata disagree: "
                + ", ".join(
                    f"{method}={sorted(values)}"
                    for method, values in identity_sets.items()
                )
            )
    else:
        common = set()

    targets = [
        (method, item.desired_path)
        for method, item in trajectories.items()
        if item.desired_path is not None
    ]
    if len(targets) >= 2:
        reference_method, reference = targets[0]
        assert reference is not None
        for method, target in targets[1:]:
            assert target is not None
            if reference.shape != target.shape or not np.allclose(
                reference,
                target,
                rtol=TARGET_RTOL,
                atol=TARGET_ATOL_M,
            ):
                maximum = (
                    float(np.max(np.abs(reference - target)))
                    if reference.shape == target.shape
                    else math.inf
                )
                raise ComparisonError(
                    f"Cartesian targets differ for {reference_method} and {method}; "
                    f"shapes={reference.shape}/{target.shape}, "
                    f"maximum_difference={maximum}"
                )

    counts = {method: len(item.q) for method, item in trajectories.items()}
    if len(set(counts.values())) > 1 and not (expected or common or len(targets) >= 2):
        raise ComparisonError(
            "Trajectory lengths differ and no matching path metadata or Cartesian "
            f"target is available: {counts}"
        )
    return {
        "requested_path_id": path_id,
        "requested_stroke_id": stroke_id,
        "discovered_identities": {
            method: sorted(values) for method, values in identity_sets.items()
        },
        "common_identities": sorted(common),
        "cartesian_target_comparisons": len(targets),
    }


def timestamps_equal(left: np.ndarray, right: np.ndarray) -> bool:
    """Return whether two timestamp grids are equivalent under project tolerance."""
    return left.shape == right.shape and bool(
        np.allclose(left, right, rtol=TIME_RTOL, atol=TIME_ATOL_S)
    )


def relative_timestamps(item: Trajectory) -> np.ndarray:
    """Return one trajectory's validated time grid normalized to start at zero."""
    return np.asarray(item.timestamps - item.timestamps[0], dtype=np.float64)


def timing_description(trajectories: Mapping[str, Trajectory]) -> str:
    """Format all original timing facts for a strict-policy failure."""
    lines = []
    for method in METHODS:
        item = trajectories[method]
        lines.append(
            f"{method}: sample_count={len(item.q)}, "
            f"start_time_s={item.timestamps[0]:.12g}, "
            f"end_time_s={item.timestamps[-1]:.12g}, "
            f"duration_s={item.duration_s:.12g}, "
            f"median_dt_s={np.median(np.diff(item.timestamps)):.12g}, "
            f"timestamp_source={item.timestamp_source}"
        )
    return "\n".join(lines)


def base_method_alignment(item: Trajectory) -> dict[str, Any]:
    """Create per-method timing audit fields shared by all policies."""
    return {
        "original_sample_count": len(item.q),
        "original_duration_s": item.duration_s,
        "original_timestamp_source": item.timestamp_source,
        "original_progress_range": [0.0, 1.0],
    }


def align_require_equal(trajectories: Mapping[str, Trajectory]) -> AlignedData:
    """Retain complete trajectories only when relative timestamp grids agree."""
    reference = relative_timestamps(trajectories["ik"])
    counts_equal = len({len(item.q) for item in trajectories.values()}) == 1
    grids_equal = counts_equal and all(
        timestamps_equal(reference, relative_timestamps(trajectories[method]))
        for method in METHODS[1:]
    )
    if not counts_equal or not grids_equal:
        raise ComparisonError(
            "require_equal timing policy rejected unequal sample counts, durations, "
            "or relative timestamp grids.\n"
            + timing_description(trajectories)
            + "\nSupply a correct common --dt when samples correspond one-to-one, "
            "or use --timing_policy common_duration with --common_duration_s."
        )
    progress = reference / reference[-1]
    method_alignment = {
        method: {
            **base_method_alignment(item),
            "interpolation_sample_count": len(item.q),
            "first_joint_position_difference_after_alignment": [0.0] * JOINT_COUNT,
            "last_joint_position_difference_after_alignment": [0.0] * JOINT_COUNT,
        }
        for method, item in trajectories.items()
    }
    return AlignedData(
        timestamps=np.array(reference, copy=True),
        progress=np.array(progress, copy=True),
        q={method: np.array(item.q, copy=True) for method, item in trajectories.items()},
        timing_policy="require_equal",
        claim_eligible=True,
        original_timing_preserved=True,
        complete_trajectory_used=True,
        common_duration_s=float(reference[-1]),
        common_sample_count=len(reference),
        progress_grid_source="common equivalent relative timestamps",
        progress_interpolation_used=False,
        shared_interval_crop_used=False,
        duration_standardized=False,
        timing_methodology=(
            "All methods used complete trajectories with equivalent original "
            "relative timestamp grids; absolute starts were normalized to zero."
        ),
        method_alignment=method_alignment,
        interpolation_method="none",
        interpolation_used=False,
        warnings=[],
    )


def align_common_duration(
    trajectories: Mapping[str, Trajectory],
    common_duration_s: float | None,
    common_samples: int | None,
) -> AlignedData:
    """Standardize every complete trajectory to one progress and physical timeline."""
    if common_duration_s is None or not math.isfinite(common_duration_s) or common_duration_s <= 0:
        raise ComparisonError(
            "common_duration timing policy requires finite positive "
            "--common_duration_s"
        )
    counts = {method: len(item.q) for method, item in trajectories.items()}
    if common_samples is None:
        if len(set(counts.values())) != 1:
            raise ComparisonError(
                "common_duration inputs have unequal sample counts "
                f"{counts}; supply explicit --common_samples"
            )
        common_samples = next(iter(counts.values()))
    if common_samples < 4:
        raise ComparisonError("--common_samples must be at least 4")

    shared_progress = np.linspace(0.0, 1.0, common_samples, dtype=np.float64)
    common_times = np.linspace(
        0.0, common_duration_s, common_samples, dtype=np.float64
    )
    aligned: dict[str, np.ndarray] = {}
    method_alignment: dict[str, dict[str, Any]] = {}
    for method, item in trajectories.items():
        source_progress = relative_timestamps(item) / item.duration_s
        spline = CubicSpline(source_progress, item.q, axis=0, extrapolate=False)
        values = np.asarray(spline(shared_progress), dtype=np.float64)
        if values.shape != (common_samples, JOINT_COUNT) or not np.all(
            np.isfinite(values)
        ):
            raise ComparisonError(
                f"{method}: normalized-progress interpolation produced invalid values"
            )
        values[0] = item.q[0]
        values[-1] = item.q[-1]
        first_difference = values[0] - item.q[0]
        last_difference = values[-1] - item.q[-1]
        if not np.allclose(first_difference, 0.0, rtol=0.0, atol=1.0e-12):
            raise ComparisonError(f"{method}: first endpoint was not preserved")
        if not np.allclose(last_difference, 0.0, rtol=0.0, atol=1.0e-12):
            raise ComparisonError(f"{method}: final endpoint was not preserved")
        aligned[method] = values
        method_alignment[method] = {
            **base_method_alignment(item),
            "progress_source": "relative saved timestamps",
            "interpolation_sample_count": common_samples,
            "first_joint_position_difference_after_interpolation": (
                first_difference.tolist()
            ),
            "last_joint_position_difference_after_interpolation": (
                last_difference.tolist()
            ),
        }
    return AlignedData(
        timestamps=common_times,
        progress=shared_progress,
        q=aligned,
        timing_policy="common_duration",
        claim_eligible=True,
        original_timing_preserved=False,
        complete_trajectory_used=True,
        common_duration_s=float(common_duration_s),
        common_sample_count=common_samples,
        progress_grid_source=(
            "per-method relative saved timestamps mapped to shared linspace(0,1)"
        ),
        progress_interpolation_used=True,
        shared_interval_crop_used=False,
        duration_standardized=True,
        timing_methodology=(
            "Complete trajectories were interpolated over normalized progress and "
            f"evaluated on a standardized {common_duration_s:.12g} s physical timeline."
        ),
        method_alignment=method_alignment,
        interpolation_method=(
            "SciPy CubicSpline over normalized progress, component-wise, "
            "no extrapolation; exact endpoints restored"
        ),
        interpolation_used=True,
        warnings=[
            "This comparison evaluates smoothness under a standardized execution "
            "duration, not the methods' original execution times."
        ],
    )


def align_shared_interval_diagnostic(
    trajectories: Mapping[str, Trajectory],
) -> AlignedData:
    """Preserve the legacy shared-time crop solely as a diagnostic."""
    first = trajectories["ik"].timestamps
    grids_equal = all(
        timestamps_equal(first, trajectories[method].timestamps)
        for method in METHODS[1:]
    )
    if grids_equal:
        common = np.array(first, copy=True)
        aligned = {
            method: np.array(item.q, copy=True)
            for method, item in trajectories.items()
        }
        interpolation_used = False
        crop_used = False
        complete_used = True
        interpolation_method = "none; original timestamp grids were equal"
    else:
        start = max(float(item.timestamps[0]) for item in trajectories.values())
        end = min(float(item.timestamps[-1]) for item in trajectories.values())
        if not end > start:
            raise ComparisonError(
                f"Trajectory timestamp intervals do not overlap: shared=[{start}, {end}]"
            )
        grid_dt = max(
            float(np.median(np.diff(item.timestamps)))
            for item in trajectories.values()
        )
        sample_count = int(math.floor((end - start) / grid_dt + 1.0e-12)) + 1
        common = start + np.arange(sample_count, dtype=np.float64) * grid_dt
        if common[-1] < end - max(TIME_ATOL_S, grid_dt * TIME_RTOL):
            common = np.append(common, end)
        else:
            common[-1] = min(common[-1], end)
        if len(common) < 4:
            raise ComparisonError(
                "Shared interval produces fewer than four derivative samples"
            )
        aligned = {}
        for method, item in trajectories.items():
            values = np.asarray(
                CubicSpline(
                    item.timestamps, item.q, axis=0, extrapolate=False
                )(common),
                dtype=np.float64,
            )
            if values.shape != (len(common), JOINT_COUNT) or not np.all(
                np.isfinite(values)
            ):
                raise ComparisonError(
                    f"{method}: shared-interval interpolation produced invalid values"
                )
            aligned[method] = values
        interpolation_used = True
        crop_used = any(
            not np.isclose(common[0], item.timestamps[0], rtol=TIME_RTOL, atol=TIME_ATOL_S)
            or not np.isclose(
                common[-1], item.timestamps[-1], rtol=TIME_RTOL, atol=TIME_ATOL_S
            )
            for item in trajectories.values()
        )
        complete_used = not crop_used
        interpolation_method = (
            "SciPy CubicSpline on shared absolute-time intersection, "
            "component-wise, no extrapolation"
        )
    progress = (common - common[0]) / (common[-1] - common[0])
    method_alignment = {
        method: {
            **base_method_alignment(item),
            "interpolation_sample_count": len(common),
        }
        for method, item in trajectories.items()
    }
    return AlignedData(
        timestamps=common,
        progress=progress,
        q=aligned,
        timing_policy="shared_interval_diagnostic",
        claim_eligible=False,
        original_timing_preserved=True,
        complete_trajectory_used=complete_used,
        common_duration_s=float(common[-1] - common[0]),
        common_sample_count=len(common),
        progress_grid_source="retained shared interval normalized to [0,1]",
        progress_interpolation_used=interpolation_used,
        shared_interval_crop_used=crop_used,
        duration_standardized=False,
        timing_methodology=(
            "DIAGNOSTIC ONLY: trajectories were evaluated only on their shared "
            "absolute-time interval and may be partial."
        ),
        method_alignment=method_alignment,
        interpolation_method=interpolation_method,
        interpolation_used=interpolation_used,
        warnings=[
            "SHARED_INTERVAL_DIAGNOSTIC_ONLY: this mode may compare partial "
            "trajectories and must not be used for primary ranking or thesis claims."
        ],
    )


def align_trajectories(
    trajectories: Mapping[str, Trajectory],
    timing_policy: str = "require_equal",
    common_duration_s: float | None = None,
    common_samples: int | None = None,
) -> AlignedData:
    """Apply one explicit timing policy to matched saved trajectories."""
    if timing_policy == "require_equal":
        if common_duration_s is not None or common_samples is not None:
            raise ComparisonError(
                "--common_duration_s and --common_samples are only valid with "
                "--timing_policy common_duration"
            )
        return align_require_equal(trajectories)
    if timing_policy == "common_duration":
        return align_common_duration(
            trajectories, common_duration_s, common_samples
        )
    if timing_policy == "shared_interval_diagnostic":
        if common_duration_s is not None or common_samples is not None:
            raise ComparisonError(
                "Common-duration options are incompatible with "
                "shared_interval_diagnostic"
            )
        return align_shared_interval_diagnostic(trajectories)
    raise ComparisonError(f"Unknown timing policy {timing_policy!r}")


def compute_derivatives(
    q: np.ndarray, timestamps: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reuse the repository's validated three-stage gradient convention."""
    from analyze_prior_vs_diffusion_contribution_v8_1 import (
        derivatives as repository_derivatives,
    )

    q_valid, _ = normalize_joint_array(q, "derivative input")
    times = validate_timestamps(timestamps, len(q_valid), "derivative timestamps")
    return repository_derivatives(q_valid, times)


def integrate(values: np.ndarray, timestamps: np.ndarray) -> float:
    """Integrate one sampled scalar series using timestamp-aware trapezoids."""
    trapezoid = getattr(np, "trapezoid", np.trapz)
    return float(trapezoid(values, x=timestamps))


def per_joint_summary(
    derivatives_by_method: Mapping[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    timestamps: np.ndarray,
) -> list[dict[str, Any]]:
    """Calculate one descriptive jerk row per method and joint."""
    rows: list[dict[str, Any]] = []
    duration = float(timestamps[-1] - timestamps[0])
    for method in METHODS:
        jerk = derivatives_by_method[method][2]
        for joint_index, joint in enumerate(JOINT_NAMES):
            values = jerk[:, joint_index]
            absolute = np.abs(values)
            rows.append(
                {
                    "method": method,
                    "joint": joint,
                    "num_samples": len(values),
                    "duration_s": duration,
                    "mean_jerk_rad_s3": float(np.mean(values)),
                    "mean_abs_jerk_rad_s3": float(np.mean(absolute)),
                    "median_abs_jerk_rad_s3": float(np.median(absolute)),
                    "rms_jerk_rad_s3": float(np.sqrt(np.mean(np.square(values)))),
                    "std_jerk_rad_s3": float(np.std(values)),
                    "max_abs_jerk_rad_s3": float(np.max(absolute)),
                    "integrated_abs_jerk_rad_s2": integrate(absolute, timestamps),
                    "integrated_squared_jerk_rad2_s5": integrate(
                        np.square(values), timestamps
                    ),
                }
            )
    return rows


def method_summary(
    derivatives_by_method: Mapping[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    timestamps: np.ndarray,
) -> list[dict[str, Any]]:
    """Calculate pooled and integrated method-level jerk metrics."""
    rows: list[dict[str, Any]] = []
    duration = float(timestamps[-1] - timestamps[0])
    for method in METHODS:
        jerk = derivatives_by_method[method][2]
        rows.append(
            {
                "method": method,
                "num_samples": len(jerk),
                "duration_s": duration,
                "mean_abs_jerk_all_joints": float(np.mean(np.abs(jerk))),
                "rms_jerk_all_joints": float(np.sqrt(np.mean(np.square(jerk)))),
                "max_abs_jerk_all_joints": float(np.max(np.abs(jerk))),
                "sum_integrated_abs_jerk": float(
                    sum(integrate(np.abs(jerk[:, index]), timestamps) for index in range(6))
                ),
                "sum_integrated_squared_jerk": float(
                    sum(
                        integrate(np.square(jerk[:, index]), timestamps)
                        for index in range(6)
                    )
                ),
            }
        )
    return rows


def percent_improvement(reference: float, comparison: float) -> float:
    """Return lower-is-better improvement, using NaN for a zero reference."""
    if reference == 0.0:
        return math.nan
    return 100.0 * (reference - comparison) / reference


def pairwise_summary(method_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Build pairwise lower-is-better comparisons for every aggregate metric."""
    by_method = {str(row["method"]): row for row in method_rows}
    metrics = (
        "mean_abs_jerk_all_joints",
        "rms_jerk_all_joints",
        "max_abs_jerk_all_joints",
        "sum_integrated_abs_jerk",
        "sum_integrated_squared_jerk",
    )
    pairs = (("ik", "pipeline"), ("mlp", "pipeline"), ("ik", "mlp"))
    rows: list[dict[str, Any]] = []
    for reference_method, comparison_method in pairs:
        for metric in metrics:
            reference = float(by_method[reference_method][metric])
            comparison = float(by_method[comparison_method][metric])
            rows.append(
                {
                    "reference_method": reference_method,
                    "comparison_method": comparison_method,
                    "metric": metric,
                    "reference_value": reference,
                    "comparison_value": comparison,
                    "absolute_difference": comparison - reference,
                    "percent_change": (
                        math.nan
                        if reference == 0.0
                        else 100.0 * (comparison - reference) / reference
                    ),
                    "percent_improvement_lower_is_better": percent_improvement(
                        reference, comparison
                    ),
                }
            )
    return rows


def prepare_output_directory(path: Path, overwrite: bool) -> Path:
    """Create an output directory while protecting existing non-empty results."""
    output = path.expanduser().resolve()
    if output.exists() and not output.is_dir():
        raise ComparisonError(f"Output path exists and is not a directory: {output}")
    if output.is_dir() and any(output.iterdir()) and not overwrite:
        raise ComparisonError(
            f"Output directory is non-empty: {output}; pass --overwrite to replace outputs"
        )
    output.mkdir(parents=True, exist_ok=True)
    return output


def atomic_write_text(path: Path, text: str) -> None:
    """Atomically replace a UTF-8 text file in its destination directory."""
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


def write_csv(path: Path, columns: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    """Atomically write a deterministic CSV."""
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="raise")
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def time_series_rows(
    timestamps: np.ndarray,
    progress: np.ndarray,
    arrays: Mapping[str, np.ndarray],
    suffix: str,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Create wide time-series CSV rows in the required method/joint order."""
    if progress.shape != timestamps.shape:
        raise ComparisonError("Progress grid must match the aligned timestamp grid")
    columns = ["time_s", "progress_0_1"] + [
        f"{method}_{joint}{suffix}" for method in METHODS for joint in JOINT_NAMES
    ]
    rows: list[dict[str, Any]] = []
    for sample_index, time_s in enumerate(timestamps):
        row: dict[str, Any] = {
            "time_s": float(time_s),
            "progress_0_1": float(progress[sample_index]),
        }
        for method in METHODS:
            for joint_index, joint in enumerate(JOINT_NAMES):
                row[f"{method}_{joint}{suffix}"] = float(
                    arrays[method][sample_index, joint_index]
                )
        rows.append(row)
    return columns, rows


def save_figure(fig: Any, output_dir: Path, stem: str, show: bool, plt: Any) -> None:
    """Save a figure as publication-resolution PNG and vector PDF."""
    fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    if show:
        plt.show(block=False)
    plt.close(fig)


def plot_stacked(
    timestamps: np.ndarray,
    arrays: Mapping[str, np.ndarray],
    ylabel: str,
    title: str,
    stem: str,
    output_dir: Path,
    show: bool,
    *,
    absolute: bool = False,
    zero_line: bool = False,
    robust_limit: float | None = None,
) -> None:
    """Plot six stacked joint time histories."""
    import matplotlib.pyplot as plt

    styles = {
        "ik": {"color": "#0072B2", "linestyle": "-", "linewidth": 1.2},
        "mlp": {"color": "#D55E00", "linestyle": "--", "linewidth": 1.2},
        "pipeline": {"color": "#009E73", "linestyle": "-", "linewidth": 1.5},
    }
    fig, axes = plt.subplots(6, 1, figsize=(10.5, 13.0), sharex=True)
    for joint_index, axis in enumerate(axes):
        for method in METHODS:
            values = arrays[method][:, joint_index]
            if absolute:
                values = np.abs(values)
            axis.plot(
                timestamps,
                values,
                label=METHOD_LABELS[method],
                **styles[method],
            )
        if zero_line:
            axis.axhline(0.0, color="0.45", linewidth=0.7, alpha=0.7)
        if robust_limit is not None:
            if absolute:
                axis.set_ylim(0.0, robust_limit)
            else:
                axis.set_ylim(-robust_limit, robust_limit)
        axis.set_ylabel(f"{JOINT_NAMES[joint_index]}\n{ylabel}")
        axis.grid(True, alpha=0.25)
    axes[0].legend(loc="upper right", ncol=3, frameon=True)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle(title)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.98))
    save_figure(fig, output_dir, stem, show, plt)


def generate_plots(
    output_dir: Path,
    timestamps: np.ndarray,
    aligned_q: Mapping[str, np.ndarray],
    derivatives_by_method: Mapping[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    joint_rows: Sequence[Mapping[str, Any]],
    title: str,
    timing_caption: str,
    show: bool,
) -> list[str]:
    """Generate all required full-range figures and any warranted robust jerk view."""
    if show:
        matplotlib.use(_INITIAL_MATPLOTLIB_BACKEND, force=True)
    else:
        matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    velocity = {method: derivatives_by_method[method][0] for method in METHODS}
    acceleration = {method: derivatives_by_method[method][1] for method in METHODS}
    jerk = {method: derivatives_by_method[method][2] for method in METHODS}
    del velocity  # Velocity is exported as CSV but has no separately required figure.
    titled = f"{title}\n{timing_caption}"

    plot_stacked(
        timestamps,
        jerk,
        "Jerk (rad/s³)",
        f"{titled}: joint jerk over time",
        "joint_jerk_over_time",
        output_dir,
        show,
        zero_line=True,
    )
    plot_stacked(
        timestamps,
        jerk,
        "|Jerk| (rad/s³)",
        f"{titled}: absolute joint jerk over time",
        "joint_absolute_jerk_over_time",
        output_dir,
        show,
        absolute=True,
        zero_line=True,
    )
    plot_stacked(
        timestamps,
        aligned_q,
        "Position (rad)",
        f"{titled}: joint positions over time",
        "joint_positions_over_time",
        output_dir,
        show,
        zero_line=True,
    )
    plot_stacked(
        timestamps,
        acceleration,
        "Acceleration (rad/s²)",
        f"{titled}: joint acceleration over time",
        "joint_acceleration_over_time",
        output_dir,
        show,
        zero_line=True,
    )

    styles = {
        "ik": ("#0072B2", "-"),
        "mlp": ("#D55E00", "--"),
        "pipeline": ("#009E73", "-"),
    }
    fig, axis = plt.subplots(figsize=(10.0, 5.5))
    for method in METHODS:
        aggregate = np.linalg.norm(jerk[method], axis=1)
        color, linestyle = styles[method]
        axis.plot(
            timestamps,
            aggregate,
            label=METHOD_LABELS[method],
            color=color,
            linestyle=linestyle,
            linewidth=1.4,
        )
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Joint jerk L2 norm (rad/s³)")
    axis.set_title(f"{titled}: aggregate jerk over time")
    axis.grid(True, alpha=0.25)
    axis.legend()
    fig.tight_layout()
    save_figure(fig, output_dir, "aggregate_jerk_over_time", show, plt)

    by_method_joint = {
        (str(row["method"]), str(row["joint"])): float(row["rms_jerk_rad_s3"])
        for row in joint_rows
    }
    fig, axis = plt.subplots(figsize=(9.5, 5.5))
    positions = np.arange(6, dtype=np.float64)
    width = 0.25
    for method_index, method in enumerate(METHODS):
        values = [by_method_joint[(method, joint)] for joint in JOINT_NAMES]
        axis.bar(
            positions + (method_index - 1) * width,
            values,
            width,
            label=METHOD_LABELS[method],
            color=styles[method][0],
        )
    axis.set_xticks(positions, JOINT_NAMES)
    axis.set_xlabel("Joint")
    axis.set_ylabel("RMS jerk (rad/s³)")
    axis.set_title(f"{titled}: per-joint RMS jerk")
    axis.grid(True, axis="y", alpha=0.25)
    axis.legend()
    fig.tight_layout()
    save_figure(fig, output_dir, "per_joint_rms_jerk", show, plt)

    warnings: list[str] = []
    absolute_values = np.concatenate(
        [np.abs(jerk[method]).reshape(-1) for method in METHODS]
    )
    maximum = float(np.max(absolute_values))
    robust = float(np.percentile(absolute_values, 99.0))
    if robust > 0.0 and maximum > 20.0 * robust:
        limit = 1.1 * robust
        plot_stacked(
            timestamps,
            jerk,
            "Jerk (rad/s³)",
            f"{titled}: joint jerk over time (robust ±99th-percentile scale)",
            "joint_jerk_over_time_robust_scale",
            output_dir,
            show,
            zero_line=True,
            robust_limit=limit,
        )
        warnings.append(
            "A robust-scale jerk figure was added because the full maximum "
            f"({maximum:.12g}) exceeded 20 times the pooled 99th percentile "
            f"({robust:.12g}); full-range figures were retained."
        )
    return warnings


def repository_commit() -> str | None:
    """Return the current Git commit when the repository exposes one."""
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
    value = result.stdout.strip()
    return value or None


def create_metadata(
    trajectories: Mapping[str, Trajectory],
    aligned: AlignedData,
    identity: Mapping[str, Any],
    args: argparse.Namespace,
    warnings: Sequence[str],
) -> dict[str, Any]:
    """Create an auditable comparison metadata document."""
    dt_values = np.diff(aligned.timestamps)
    effective_dt = float(np.median(dt_values))
    methods: dict[str, Any] = {}
    for method in METHODS:
        item = trajectories[method]
        methods[method] = {
            "provided_source_path": str(item.provided_path),
            "selected_trajectory_file": str(item.selected_file),
            "selected_array_key": item.selected_array_key,
            "selected_file_sha256": sha256_file(item.selected_file),
            "source_input_or_stroke_identifiers": sorted(item.identities),
            "original_trajectory_shape": list(item.original_shape),
            "normalized_trajectory_shape": list(item.q.shape),
            "original_sample_count": len(item.q),
            "aligned_sample_count": len(aligned.timestamps),
            "original_duration_s": item.duration_s,
            "aligned_duration_s": float(
                aligned.timestamps[-1] - aligned.timestamps[0]
            ),
            "timestamp_source": item.timestamp_source,
            "original_effective_dt_s": float(np.median(np.diff(item.timestamps))),
            "cartesian_target_available": item.desired_path is not None,
            "pipeline_selection_evidence": item.metadata.get(
                "pipeline_selection_evidence"
            ),
            "alignment": aligned.method_alignment[method],
            "warnings": item.warnings,
        }
    return {
        "analysis": "saved IK vs MLP vs proposed-pipeline joint jerk comparison",
        "timing_policy": aligned.timing_policy,
        "claim_eligible": aligned.claim_eligible,
        "original_timing_preserved": aligned.original_timing_preserved,
        "complete_trajectory_used": aligned.complete_trajectory_used,
        "common_duration_s": aligned.common_duration_s,
        "common_sample_count": aligned.common_sample_count,
        "progress_grid_source": aligned.progress_grid_source,
        "progress_interpolation_used": aligned.progress_interpolation_used,
        "shared_interval_crop_used": aligned.shared_interval_crop_used,
        "duration_standardized": aligned.duration_standardized,
        "timing_methodology": aligned.timing_methodology,
        "methods": methods,
        "path_validation": identity,
        "aligned_sample_count": len(aligned.timestamps),
        "aligned_time_start_s": float(aligned.timestamps[0]),
        "aligned_time_end_s": float(aligned.timestamps[-1]),
        "aligned_duration_s": float(aligned.timestamps[-1] - aligned.timestamps[0]),
        "effective_dt_s": effective_dt,
        "aligned_grid_uniform": bool(
            np.allclose(dt_values, effective_dt, rtol=TIME_RTOL, atol=TIME_ATOL_S)
        ),
        "dt_override_s": getattr(args, "dt", None),
        "derivative_implementation_reused": (
            "analyze_prior_vs_diffusion_contribution_v8_1.derivatives: "
            "three successive numpy.gradient calls with timestamps, axis=0, "
            "edge_order=2"
        ),
        "interpolation_used": aligned.interpolation_used,
        "interpolation_method": aligned.interpolation_method,
        "smoothing_or_filtering": "none",
        "statistical_scope": (
            "trajectory-level descriptive comparison only; time samples are not "
            "treated as independent experimental replicates"
        ),
        "repository_commit_hash": repository_commit(),
        "creation_command": " ".join(shlex.quote(value) for value in sys.argv),
        "creation_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "warnings": list(warnings),
    }


def timing_plot_caption(aligned: AlignedData) -> str:
    """Return the mandatory timing-policy label for every figure."""
    if aligned.timing_policy == "common_duration":
        return (
            "Standardized complete-trajectory duration: "
            f"{aligned.common_duration_s:.12g} s"
        )
    if aligned.timing_policy == "require_equal":
        return "Original equivalent timing retained"
    return "DIAGNOSTIC ONLY — PARTIAL SHARED INTERVAL"


def load_matched_trajectories(
    args: argparse.Namespace,
) -> tuple[dict[str, Trajectory], dict[str, Any]]:
    """Load all methods and validate path identity without aligning them."""
    trajectories = {
        "ik": load_trajectory(args.ik_path, "ik", args.dt),
        "mlp": load_trajectory(args.mlp_path, "mlp", args.dt),
        "pipeline": load_trajectory(args.pipeline_path, "pipeline", args.dt),
    }
    identity = validate_path_compatibility(
        trajectories, path_id=args.path_id, stroke_id=args.stroke_id
    )
    return trajectories, identity


def timing_policy_availability(
    trajectories: Mapping[str, Trajectory],
    common_duration_s: float | None,
    common_samples: int | None,
) -> dict[str, str]:
    """Describe policy usability without performing interpolation or derivatives."""
    relative = {method: relative_timestamps(item) for method, item in trajectories.items()}
    counts = {method: len(item.q) for method, item in trajectories.items()}
    equal_counts = len(set(counts.values())) == 1
    equal_grids = equal_counts and all(
        timestamps_equal(relative["ik"], relative[method])
        for method in METHODS[1:]
    )
    require_status = (
        "available: equivalent relative timestamp grids"
        if equal_grids
        else "unavailable: sample counts, durations, or relative grids differ"
    )

    if common_duration_s is None:
        common_status = (
            "available after supplying finite positive --common_duration_s"
            if equal_counts or common_samples is not None
            else "available after supplying --common_duration_s and --common_samples"
        )
    elif not math.isfinite(common_duration_s) or common_duration_s <= 0.0:
        common_status = "unavailable: --common_duration_s must be finite and positive"
    elif not equal_counts and common_samples is None:
        common_status = "unavailable until explicit --common_samples is supplied"
    elif common_samples is not None and common_samples < 4:
        common_status = "unavailable: --common_samples must be at least 4"
    else:
        common_status = "available: complete trajectories will be standardized"

    start = max(float(item.timestamps[0]) for item in trajectories.values())
    end = min(float(item.timestamps[-1]) for item in trajectories.values())
    diagnostic_status = (
        "available but claim_eligible=false"
        if end > start
        else "unavailable: timestamp intervals do not overlap"
    )
    return {
        "require_equal": require_status,
        "common_duration": common_status,
        "shared_interval_diagnostic": diagnostic_status,
    }


def print_timing_inspection(
    trajectories: Mapping[str, Trajectory],
    identity: Mapping[str, Any],
    args: argparse.Namespace,
) -> None:
    """Print a preflight table without creating analysis outputs."""
    headers = (
        "method",
        "selected_file",
        "array_key",
        "sample_count",
        "start_time_s",
        "end_time_s",
        "duration_s",
        "median_dt_s",
        "timestamp_source",
        "path_identity",
    )
    print("\t".join(headers))
    for method in METHODS:
        item = trajectories[method]
        values = (
            method,
            str(item.selected_file),
            item.selected_array_key or "",
            str(len(item.q)),
            f"{item.timestamps[0]:.12g}",
            f"{item.timestamps[-1]:.12g}",
            f"{item.duration_s:.12g}",
            f"{np.median(np.diff(item.timestamps)):.12g}",
            item.timestamp_source,
            ",".join(sorted(item.identities)) or "not_recorded",
        )
        print("\t".join(values))
    print("Timing policy availability:")
    availability = timing_policy_availability(
        trajectories, args.common_duration_s, args.common_samples
    )
    for policy, status in availability.items():
        print(f"  {policy}: {status}")
    print(
        "Validated path identity: "
        f"{identity.get('common_identities') or identity.get('requested_path_id')}"
    )
    print("IK_MLP_PIPELINE_TIMING_INSPECTION_COMPLETE")


def run_comparison(args: argparse.Namespace) -> dict[str, Any]:
    """Run the read-only comparison and write all requested outputs."""
    trajectories, identity = load_matched_trajectories(args)
    aligned = align_trajectories(
        trajectories,
        timing_policy=getattr(args, "timing_policy", "require_equal"),
        common_duration_s=getattr(args, "common_duration_s", None),
        common_samples=getattr(args, "common_samples", None),
    )
    derivatives_by_method = {
        method: compute_derivatives(aligned.q[method], aligned.timestamps)
        for method in METHODS
    }
    joint_rows = per_joint_summary(derivatives_by_method, aligned.timestamps)
    method_rows = method_summary(derivatives_by_method, aligned.timestamps)
    pairwise_rows = pairwise_summary(method_rows)
    output_dir = prepare_output_directory(args.output_dir, args.overwrite)

    columns, rows = time_series_rows(
        aligned.timestamps, aligned.progress, aligned.q, ""
    )
    write_csv(output_dir / "aligned_joint_positions.csv", columns, rows)
    for filename, derivative_index, suffix in (
        ("joint_velocity_over_time.csv", 0, "_velocity"),
        ("joint_acceleration_over_time.csv", 1, "_acceleration"),
        ("joint_jerk_over_time.csv", 2, "_jerk"),
    ):
        arrays = {
            method: derivatives_by_method[method][derivative_index]
            for method in METHODS
        }
        columns, rows = time_series_rows(
            aligned.timestamps, aligned.progress, arrays, suffix
        )
        write_csv(output_dir / filename, columns, rows)

    joint_columns = (
        "method",
        "joint",
        "num_samples",
        "duration_s",
        "mean_jerk_rad_s3",
        "mean_abs_jerk_rad_s3",
        "median_abs_jerk_rad_s3",
        "rms_jerk_rad_s3",
        "std_jerk_rad_s3",
        "max_abs_jerk_rad_s3",
        "integrated_abs_jerk_rad_s2",
        "integrated_squared_jerk_rad2_s5",
    )
    write_csv(output_dir / "joint_jerk_summary.csv", joint_columns, joint_rows)
    method_columns = (
        "method",
        "num_samples",
        "duration_s",
        "mean_abs_jerk_all_joints",
        "rms_jerk_all_joints",
        "max_abs_jerk_all_joints",
        "sum_integrated_abs_jerk",
        "sum_integrated_squared_jerk",
    )
    write_csv(output_dir / "method_jerk_summary.csv", method_columns, method_rows)
    pairwise_columns = (
        "reference_method",
        "comparison_method",
        "metric",
        "reference_value",
        "comparison_value",
        "absolute_difference",
        "percent_change",
        "percent_improvement_lower_is_better",
    )
    write_csv(
        output_dir / "jerk_pairwise_comparison.csv",
        pairwise_columns,
        pairwise_rows,
    )

    title = args.title or "IK, MLP, and proposed-pipeline comparison"
    plot_warnings = generate_plots(
        output_dir,
        aligned.timestamps,
        aligned.q,
        derivatives_by_method,
        joint_rows,
        title,
        timing_plot_caption(aligned),
        args.show,
    )
    warnings = [*aligned.warnings, *plot_warnings]
    metadata = create_metadata(trajectories, aligned, identity, args, warnings)
    atomic_write_text(
        output_dir / "comparison_metadata.json",
        json.dumps(_json_value(metadata), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
    )
    return {
        "output_dir": output_dir,
        "trajectories": trajectories,
        "aligned": aligned,
        "method_rows": method_rows,
        "pairwise_rows": pairwise_rows,
        "identity": identity,
    }


def print_summary(result: Mapping[str, Any]) -> None:
    """Print the required concise console summary."""
    trajectories: Mapping[str, Trajectory] = result["trajectories"]
    aligned: AlignedData = result["aligned"]
    method_rows = {row["method"]: row for row in result["method_rows"]}
    pairwise = {
        (row["reference_method"], row["comparison_method"], row["metric"]): row
        for row in result["pairwise_rows"]
    }
    print("Selected saved trajectories:")
    for method in METHODS:
        item = trajectories[method]
        print(
            f"  {METHOD_LABELS[method]}: {item.selected_file} "
            f"(samples={len(item.q)}, duration={item.duration_s:.12g}s, "
            f"identity={sorted(item.identities) or ['not recorded']})"
        )
    print(
        f"Timing policy={aligned.timing_policy}, "
        f"claim_eligible={aligned.claim_eligible}, "
        f"complete_trajectory_used={aligned.complete_trajectory_used}"
    )
    print(
        f"Common samples={aligned.common_sample_count}, "
        f"aligned duration={aligned.common_duration_s:.12g}s, "
        f"effective_dt={np.median(np.diff(aligned.timestamps)):.12g}s, "
        f"progress_interpolation={aligned.progress_interpolation_used}, "
        f"trajectory_cropped={aligned.shared_interval_crop_used}"
    )
    if aligned.timing_policy == "common_duration":
        print(
            "All methods were evaluated over their complete trajectories using "
            "a standardized execution duration of "
            f"{aligned.common_duration_s:.12g} seconds."
        )
    elif aligned.timing_policy == "require_equal":
        print("All methods used equivalent original relative timestamp grids.")
    else:
        print("WARNING: SHARED_INTERVAL_DIAGNOSTIC_ONLY")
        print(
            "This mode may compare partial trajectories and must not be used "
            "for primary method ranking or thesis claims."
        )
    for method in METHODS:
        row = method_rows[method]
        print(
            f"  {METHOD_LABELS[method]}: RMS jerk="
            f"{row['rms_jerk_all_joints']:.12g} rad/s^3, max |jerk|="
            f"{row['max_abs_jerk_all_joints']:.12g} rad/s^3, integrated "
            f"squared jerk={row['sum_integrated_squared_jerk']:.12g} rad^2/s^5"
        )
    for reference in ("ik", "mlp"):
        row = pairwise[(reference, "pipeline", "rms_jerk_all_joints")]
        print(
            f"Pipeline RMS-jerk improvement relative to "
            f"{METHOD_LABELS[reference]}: "
            f"{row['percent_improvement_lower_is_better']:.12g}%"
        )
    print(f"Output directory: {result['output_dir']}")
    print("IK_MLP_PIPELINE_JERK_COMPARISON_COMPLETE")


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point."""
    try:
        args = parse_args(argv)
        if args.inspect_only:
            trajectories, identity = load_matched_trajectories(args)
            print_timing_inspection(trajectories, identity, args)
            return 0
        result = run_comparison(args)
        print_summary(result)
        return 0
    except ComparisonError as exc:
        print(f"IK_MLP_PIPELINE_JERK_COMPARISON_FAILED: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
