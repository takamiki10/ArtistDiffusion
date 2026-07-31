#!/usr/bin/env python3
"""Validate artifacts produced by build_multistroke_full_pose_execution.py."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np

import evaluate_diffusion_v7_teacher_forced_validation as v7_evaluator
from generate_ik_seed_path import (
    HARD_JOINT_LIMIT_TOLERANCE_RAD,
    get_joint_bounds,
)
from orientation_aware_adaptive_ik import (  # pyright: ignore[reportMissingImports]
    orientation_error_trajectory,
    trajectory_full_transform_fk,
)


ACCEPTED_SOURCE_VERDICT = "V8_1_DEPLOYMENT_TRAJECTORY_ACCEPTED"
ACCEPTED_VERDICT = "MULTISTROKE_EXECUTION_ACCEPTED"
REJECTED_VERDICT = "MULTISTROKE_EXECUTION_REJECTED"

FULL_CSV_NAME = "multistroke_execution_trajectory.csv"
FULL_NPZ_NAME = "multistroke_execution_full.npz"
MANIFEST_NAME = "multistroke_segment_manifest.csv"
METRICS_NAME = "multistroke_execution_metrics.json"
REPORT_NAME = "multistroke_execution_report.txt"
APPROVED_CSV_NAME = "approved_multistroke_execution_trajectory.csv"
APPROVED_NPZ_NAME = "approved_multistroke_execution_trajectory.npz"
LEGACY_CSV_NAME = "SmartJoint_Data_diffusion.csv"

SOURCE_FULL_NPZ_NAME = "deployment_trajectory_full.npz"
SOURCE_METRICS_NAME = "deployment_metrics.json"
SOURCE_APPROVED_CSV_NAME = "approved_simulation_trajectory.csv"
SOURCE_APPROVED_NPZ_NAME = "approved_simulation_trajectory.npz"

JOINT_DIM = 6
EXPECTED_FK_FRAME = "xMateCR7_link6"
ATOL = 1.0e-9
RTOL = 1.0e-7

FULL_NPZ_FIELDS = {
    "timestamps",
    "q",
    "fk_position",
    "fk_rotation_matrix",
    "fk_quaternion",
    "target_rotation_matrix",
    "target_quaternion",
    "target_rpy",
    "drawing_z",
    "hover_z",
    "desired_position",
    "cartesian_error_m",
    "orientation_error_rad",
    "z_tracking_error_m",
    "segment_index_per_sample",
    "segment_type_per_sample",
    "stroke_index_per_sample",
    "joint_velocity",
    "joint_acceleration",
    "joint_jerk",
    "urdf_path",
    "urdf_sha256",
    "joint_order",
    "fk_frame",
    "source_stroke_directories",
    "source_stroke_input_sha256",
    "source_stroke_deployment_path_id",
    "maximum_joint_step_rad",
    "maximum_cartesian_error_m",
    "maximum_orientation_error_rad",
    "maximum_z_tracking_error_m",
    "accepted",
    "verdict",
}

FULL_CSV_COLUMNS = [
    "sample_index",
    "time_seconds",
    "q1",
    "q2",
    "q3",
    "q4",
    "q5",
    "q6",
    "desired_x",
    "desired_y",
    "desired_z",
    "fk_x",
    "fk_y",
    "fk_z",
    "cartesian_error_m",
    "orientation_error_rad",
    "z_tracking_error_m",
    "segment_index",
    "segment_type",
    "stroke_index",
]

APPROVED_CSV_COLUMNS = [
    "time_seconds",
    "q1",
    "q2",
    "q3",
    "q4",
    "q5",
    "q6",
]

LEGACY_CSV_COLUMNS = [
    "Timestamp",
    "TouchType",
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
    "OriginalStatus",
]

MANIFEST_COLUMNS = [
    "segment_index",
    "segment_type",
    "stroke_index",
    "start_sample",
    "end_sample",
    "start_time_seconds",
    "end_time_seconds",
    "sample_count",
    "duration_seconds",
    "planned_duration_seconds",
    "duplicated_first_boundary_removed",
]

SOURCE_FULL_FIELDS = {
    "final_q",
    "final_ee",
    "desired_path",
    "timestamps",
    "target_rpy",
    "target_quaternion",
    "target_rotation_matrix",
    "target_z",
    "maximum_orientation_error_gate_rad",
    "maximum_z_error_gate_m",
    "urdf_path",
    "urdf_sha256",
    "input_sha256",
    "deployment_path_id",
    "orientation_fk_frame",
    "z_fk_frame",
}

SEGMENT_TYPES = (
    "initial_hover",
    "initial_descent",
    "drawing_stroke",
    "lift",
    "hover_travel",
    "descent",
    "final_lift",
)

TRANSITION_TYPES = {
    "initial_descent",
    "lift",
    "hover_travel",
    "descent",
    "final_lift",
}


class ValidationError(RuntimeError):
    """The artifact is malformed or inconsistent."""


def require_file(path: Path) -> None:
    if not path.is_file():
        raise ValidationError(f"Required file is missing: {path}")


def load_json(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationError(f"Cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValidationError(f"JSON root must be an object: {path}")
    return value


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    try:
        with np.load(  # pyright: ignore[reportArgumentType]
            str(path),
            allow_pickle=False,
        ) as archive:
            return {
                name: np.array(archive[name], copy=True)
                for name in archive.files
            }
    except (OSError, ValueError) as exc:
        raise ValidationError(f"Cannot read NPZ {path}: {exc}") from exc


def require_fields(
    data: Mapping[str, Any],
    fields: set[str],
    label: str,
    *,
    exact: bool,
) -> None:
    names = set(data)
    missing = sorted(fields - names)
    extra = sorted(names - fields) if exact else []
    if missing or extra:
        raise ValidationError(
            f"{label} schema mismatch; missing={missing}, extra={extra}"
        )


def scalar(value: Any, label: str) -> Any:
    array = np.asarray(value)
    if array.size != 1:
        raise ValidationError(f"{label} must contain exactly one scalar")
    return array.reshape(-1)[0].item()


def text_scalar(value: Any, label: str) -> str:
    value = scalar(value, label)
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def bool_scalar(value: Any, label: str) -> bool:
    value = scalar(value, label)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)) and int(value) in (0, 1):
        return bool(value)
    raise ValidationError(f"{label} must be a boolean")


def finite_scalar(value: Any, label: str) -> float:
    result = float(scalar(value, label))
    if not math.isfinite(result):
        raise ValidationError(f"{label} must be finite")
    return result


def finite_array(
    value: Any,
    label: str,
    *,
    shape: Tuple[int, ...] | None = None,
    ndim: int | None = None,
) -> np.ndarray:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number):
        raise ValidationError(f"{label} must be numeric")
    result = np.asarray(array, dtype=np.float64)
    if shape is not None and result.shape != shape:
        raise ValidationError(
            f"{label} must have shape {shape}, got {result.shape}"
        )
    if ndim is not None and result.ndim != ndim:
        raise ValidationError(
            f"{label} must have {ndim} dimensions, got {result.ndim}"
        )
    if not np.all(np.isfinite(result)):
        raise ValidationError(f"{label} contains non-finite values")
    return result


def string_array(value: Any, label: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 1:
        raise ValidationError(f"{label} must be one-dimensional")
    return np.asarray(
        [
            item.decode("utf-8") if isinstance(item, bytes) else str(item)
            for item in array.tolist()
        ],
        dtype=str,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise ValidationError(f"Cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def assert_close(
    actual: np.ndarray,
    expected: np.ndarray,
    label: str,
    *,
    atol: float = ATOL,
    rtol: float = RTOL,
) -> None:
    if actual.shape != expected.shape or not np.allclose(
        actual,
        expected,
        atol=atol,
        rtol=rtol,
    ):
        maximum = (
            float(np.max(np.abs(actual - expected)))
            if actual.shape == expected.shape and actual.size
            else math.inf
        )
        raise ValidationError(
            f"{label} mismatch; shapes={actual.shape}/{expected.shape}, "
            f"maximum absolute difference={maximum}"
        )


def assert_exact(actual: np.ndarray, expected: np.ndarray, label: str) -> None:
    if actual.shape != expected.shape or not np.array_equal(actual, expected):
        raise ValidationError(f"{label} is not an exact match")


def parse_json_verdict(
    metrics: Mapping[str, Any],
    *,
    accepted_verdict: str,
    rejected_verdict: str | None,
    label: str,
) -> bool:
    accepted = metrics.get("accepted")
    verdict = metrics.get("verdict")
    if not isinstance(accepted, bool):
        raise ValidationError(f"{label}.accepted must be a JSON boolean")
    if not isinstance(verdict, str):
        raise ValidationError(f"{label}.verdict must be a string")
    allowed = {accepted_verdict}
    if rejected_verdict is not None:
        allowed.add(rejected_verdict)
    if verdict not in allowed:
        raise ValidationError(
            f"{label}.verdict has unsupported value {verdict!r}"
        )
    if accepted != (verdict == accepted_verdict):
        raise ValidationError(f"{label} accepted and verdict disagree")
    return accepted


def validate_rotation(rotation: np.ndarray, label: str) -> None:
    assert_close(
        rotation.T @ rotation,
        np.eye(3),
        f"{label} orthonormality",
        atol=1.0e-8,
        rtol=1.0e-8,
    )
    if not math.isclose(
        float(np.linalg.det(rotation)),
        1.0,
        rel_tol=0.0,
        abs_tol=1.0e-8,
    ):
        raise ValidationError(f"{label} determinant must be +1")


def read_csv(path: Path, columns: Sequence[str]) -> List[Dict[str, str]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != list(columns):
                raise ValidationError(
                    f"{path} columns must be exactly {list(columns)}, "
                    f"got {reader.fieldnames}"
                )
            rows = list(reader)
    except OSError as exc:
        raise ValidationError(f"Cannot read CSV {path}: {exc}") from exc
    if not rows:
        raise ValidationError(f"CSV contains no data rows: {path}")
    return rows


@dataclass(frozen=True)
class SourceStroke:
    directory: Path
    final_q: np.ndarray
    final_ee: np.ndarray
    desired_path: np.ndarray
    timestamps: np.ndarray
    target_rpy: np.ndarray
    target_quaternion: np.ndarray
    target_rotation_matrix: np.ndarray
    target_z: float
    orientation_gate: float
    z_gate: float
    urdf_path: Path
    urdf_sha256: str
    input_sha256: str
    deployment_path_id: str
    fk_frame: str


def load_source_stroke(directory: Path) -> SourceStroke:
    directory = directory.expanduser().resolve()
    paths = {
        "full": directory / SOURCE_FULL_NPZ_NAME,
        "metrics": directory / SOURCE_METRICS_NAME,
        "csv": directory / SOURCE_APPROVED_CSV_NAME,
        "npz": directory / SOURCE_APPROVED_NPZ_NAME,
    }
    for path in paths.values():
        require_file(path)
    full = load_npz(paths["full"])
    require_fields(full, SOURCE_FULL_FIELDS, str(paths["full"]), exact=False)
    metrics = load_json(paths["metrics"])
    if not parse_json_verdict(
        metrics,
        accepted_verdict=ACCEPTED_SOURCE_VERDICT,
        rejected_verdict=None,
        label=str(paths["metrics"]),
    ):
        raise ValidationError(f"Source stroke is not accepted: {directory}")

    final_q = finite_array(
        full["final_q"], "source final_q", shape=(100, JOINT_DIM)
    )
    final_ee = finite_array(
        full["final_ee"], "source final_ee", shape=(100, 3)
    )
    desired_path = finite_array(
        full["desired_path"], "source desired_path", shape=(100, 3)
    )
    timestamps = finite_array(
        full["timestamps"], "source timestamps", shape=(100,)
    )
    if np.any(np.diff(timestamps) <= 0.0):
        raise ValidationError("Source timestamps must be strictly increasing")
    target_rpy = finite_array(full["target_rpy"], "source target_rpy", shape=(3,))
    target_quaternion = finite_array(
        full["target_quaternion"],
        "source target_quaternion",
        shape=(4,),
    )
    target_rotation = finite_array(
        full["target_rotation_matrix"],
        "source target_rotation_matrix",
        shape=(3, 3),
    )
    validate_rotation(target_rotation, "source target rotation")
    target_z = finite_scalar(full["target_z"], "source target_z")
    if not np.allclose(
        desired_path[:, 2], target_z, rtol=0.0, atol=ATOL
    ):
        raise ValidationError("Source desired_path does not enforce target_z")
    orientation_gate = finite_scalar(
        full["maximum_orientation_error_gate_rad"],
        "source maximum_orientation_error_gate_rad",
    )
    z_gate = finite_scalar(
        full["maximum_z_error_gate_m"],
        "source maximum_z_error_gate_m",
    )
    if orientation_gate <= 0.0 or z_gate <= 0.0:
        raise ValidationError("Source pose gates must be positive")
    urdf_path = Path(
        text_scalar(full["urdf_path"], "source urdf_path")
    ).expanduser().resolve()
    require_file(urdf_path)
    urdf_hash = text_scalar(full["urdf_sha256"], "source urdf_sha256")
    if sha256_file(urdf_path) != urdf_hash:
        raise ValidationError(f"Source URDF hash mismatch: {directory}")
    input_sha = text_scalar(full["input_sha256"], "source input_sha256")
    path_id = text_scalar(
        full["deployment_path_id"], "source deployment_path_id"
    )
    orientation_frame = text_scalar(
        full["orientation_fk_frame"], "source orientation_fk_frame"
    )
    z_frame = text_scalar(full["z_fk_frame"], "source z_fk_frame")
    if orientation_frame != z_frame:
        raise ValidationError("Source orientation and Z FK frames differ")

    csv_rows = read_csv(paths["csv"], APPROVED_CSV_COLUMNS)
    if len(csv_rows) != 100:
        raise ValidationError("Source approved CSV must contain 100 rows")
    try:
        csv_timestamps = np.asarray(
            [float(row["time_seconds"]) for row in csv_rows]
        )
        csv_q = np.asarray(
            [
                [float(row[f"q{joint + 1}"]) for joint in range(JOINT_DIM)]
                for row in csv_rows
            ]
        )
    except (TypeError, ValueError) as exc:
        raise ValidationError("Source approved CSV contains invalid values") from exc
    assert_exact(csv_timestamps, timestamps, "source approved CSV timestamps")
    assert_exact(csv_q, final_q, "source approved CSV q")

    approved = load_npz(paths["npz"])
    if "timestamps" in approved:
        assert_exact(
            finite_array(
                approved["timestamps"],
                "source approved NPZ timestamps",
                shape=(100,),
            ),
            timestamps,
            "source approved NPZ timestamps",
        )
    q_key = "q" if "q" in approved else "final_q"
    if q_key not in approved:
        raise ValidationError("Source approved NPZ has no q/final_q field")
    assert_exact(
        finite_array(
            approved[q_key],
            f"source approved NPZ {q_key}",
            shape=(100, JOINT_DIM),
        ),
        final_q,
        "source approved NPZ q",
    )

    return SourceStroke(
        directory=directory,
        final_q=final_q,
        final_ee=final_ee,
        desired_path=desired_path,
        timestamps=timestamps,
        target_rpy=target_rpy,
        target_quaternion=target_quaternion,
        target_rotation_matrix=target_rotation,
        target_z=target_z,
        orientation_gate=orientation_gate,
        z_gate=z_gate,
        urdf_path=urdf_path,
        urdf_sha256=urdf_hash,
        input_sha256=input_sha,
        deployment_path_id=path_id,
        fk_frame=orientation_frame,
    )


@dataclass(frozen=True)
class ManifestRow:
    segment_index: int
    segment_type: str
    stroke_index: int
    start_sample: int
    end_sample: int
    start_time: float
    end_time: float
    sample_count: int
    duration: float
    planned_duration: float
    boundary_removed: int


def read_manifest(path: Path) -> List[ManifestRow]:
    rows = read_csv(path, MANIFEST_COLUMNS)
    result: List[ManifestRow] = []
    for row_number, row in enumerate(rows):
        try:
            item = ManifestRow(
                segment_index=int(row["segment_index"]),
                segment_type=row["segment_type"],
                stroke_index=int(row["stroke_index"]),
                start_sample=int(row["start_sample"]),
                end_sample=int(row["end_sample"]),
                start_time=float(row["start_time_seconds"]),
                end_time=float(row["end_time_seconds"]),
                sample_count=int(row["sample_count"]),
                duration=float(row["duration_seconds"]),
                planned_duration=float(row["planned_duration_seconds"]),
                boundary_removed=int(row["duplicated_first_boundary_removed"]),
            )
        except (TypeError, ValueError) as exc:
            raise ValidationError(
                f"Manifest row {row_number} contains invalid values"
            ) from exc
        if item.segment_type not in SEGMENT_TYPES:
            raise ValidationError(
                f"Manifest row {row_number} has invalid segment type"
            )
        if not all(
            math.isfinite(value)
            for value in (
                item.start_time,
                item.end_time,
                item.duration,
                item.planned_duration,
            )
        ):
            raise ValidationError(
                f"Manifest row {row_number} contains non-finite values"
            )
        result.append(item)
    return result


def expected_segment_types(stroke_count: int) -> List[str]:
    result = ["initial_hover", "initial_descent", "drawing_stroke"]
    for _ in range(1, stroke_count):
        result.extend(["lift", "hover_travel", "descent", "drawing_stroke"])
    result.append("final_lift")
    return result


def validate_manifest(
    rows: Sequence[ManifestRow],
    timestamps: np.ndarray,
    segment_indices: np.ndarray,
    segment_types: np.ndarray,
    stroke_indices: np.ndarray,
    stroke_count: int,
) -> None:
    actual_types = [row.segment_type for row in rows]
    expected_types = expected_segment_types(stroke_count)
    if actual_types != expected_types:
        raise ValidationError(
            f"Segment sequence mismatch: expected {expected_types}, "
            f"got {actual_types}"
        )
    next_sample = 0
    for expected_index, row in enumerate(rows):
        if row.segment_index != expected_index:
            raise ValidationError("Manifest segment indices are not contiguous")
        if row.start_sample != next_sample:
            raise ValidationError(
                f"Manifest segment {row.segment_index} is not contiguous"
            )
        if row.end_sample < row.start_sample:
            raise ValidationError("Manifest contains an inverted range")
        count = row.end_sample - row.start_sample + 1
        if row.sample_count != count:
            raise ValidationError(
                f"Manifest segment {row.segment_index} sample count is wrong"
            )
        if row.end_sample >= len(timestamps):
            raise ValidationError("Manifest range exceeds trajectory")
        expected_boundary_removed = int(
            expected_index != 0 and row.segment_type != "drawing_stroke"
        )
        if row.boundary_removed != expected_boundary_removed:
            raise ValidationError(
                f"Manifest segment {row.segment_index} boundary flag is wrong"
            )
        selection = slice(row.start_sample, row.end_sample + 1)
        if not np.all(segment_indices[selection] == row.segment_index):
            raise ValidationError("Per-sample segment index disagrees with manifest")
        if not np.all(segment_types[selection] == row.segment_type):
            raise ValidationError("Per-sample segment type disagrees with manifest")
        if not np.all(stroke_indices[selection] == row.stroke_index):
            raise ValidationError("Per-sample stroke index disagrees with manifest")
        if not math.isclose(
            row.start_time,
            float(timestamps[row.start_sample]),
            rel_tol=0.0,
            abs_tol=ATOL,
        ) or not math.isclose(
            row.end_time,
            float(timestamps[row.end_sample]),
            rel_tol=0.0,
            abs_tol=ATOL,
        ):
            raise ValidationError("Manifest timestamps disagree with trajectory")
        expected_duration = row.end_time - row.start_time
        if not math.isclose(
            row.duration,
            expected_duration,
            rel_tol=RTOL,
            abs_tol=ATOL,
        ):
            raise ValidationError("Manifest duration is inconsistent")
        if row.segment_type != "initial_hover" and row.planned_duration <= 0.0:
            raise ValidationError("Planned transition/drawing duration must be positive")
        next_sample = row.end_sample + 1
    if next_sample != len(timestamps):
        raise ValidationError("Manifest does not cover every trajectory sample")


def exact_subsequence_starts(
    combined: np.ndarray,
    source: np.ndarray,
) -> List[int]:
    if source.shape[1:] != combined.shape[1:] or len(source) > len(combined):
        return []
    return [
        start
        for start in range(len(combined) - len(source) + 1)
        if np.array_equal(combined[start : start + len(source)], source)
    ]


def validate_drawing_preservation(
    strokes: Sequence[SourceStroke],
    q: np.ndarray,
    timestamps: np.ndarray,
    desired_position: np.ndarray,
    manifest: Sequence[ManifestRow],
    recorded_input_sha: Sequence[str],
    recorded_path_ids: Sequence[str],
) -> None:
    if list(recorded_input_sha) != [stroke.input_sha256 for stroke in strokes]:
        raise ValidationError("Recorded source input SHA list is inconsistent")
    if list(recorded_path_ids) != [
        stroke.deployment_path_id for stroke in strokes
    ]:
        raise ValidationError("Recorded source deployment path IDs are inconsistent")
    drawing_rows = [
        row for row in manifest if row.segment_type == "drawing_stroke"
    ]
    if len(drawing_rows) != len(strokes):
        raise ValidationError("Drawing segment count differs from stroke count")
    previous_end = -1
    for stroke_index, (stroke, row) in enumerate(zip(strokes, drawing_rows)):
        starts = exact_subsequence_starts(q, stroke.final_q)
        if len(starts) != 1:
            raise ValidationError(
                f"Stroke {stroke_index} must occur exactly once; starts={starts}"
            )
        start = starts[0]
        end = start + len(stroke.final_q) - 1
        if start <= previous_end:
            raise ValidationError("Source strokes do not appear in supplied order")
        if row.stroke_index != stroke_index:
            raise ValidationError("Drawing manifest stroke index is wrong")
        if row.start_sample != start or row.end_sample != end:
            raise ValidationError(
                "Drawing segment does not own its complete source range"
            )
        if row.sample_count != len(stroke.final_q):
            raise ValidationError("Drawing segment must contain 100 stored rows")
        assert_close(
            timestamps[start : end + 1] - timestamps[start],
            stroke.timestamps - stroke.timestamps[0],
            f"stroke {stroke_index} relative timestamps",
            atol=1.0e-10,
            rtol=1.0e-10,
        )
        assert_exact(
            desired_position[start : end + 1],
            stroke.desired_path,
            f"stroke {stroke_index} desired positions",
        )
        previous_end = end


def quintic(u: np.ndarray) -> np.ndarray:
    return 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5


def transition_endpoint(
    segment_type: str,
    occurrence: int,
    strokes: Sequence[SourceStroke],
    hover_z: float,
) -> np.ndarray:
    if segment_type == "initial_descent":
        return np.array(strokes[0].desired_path[0], copy=True)
    if segment_type == "lift":
        point = np.array(strokes[occurrence].desired_path[-1], copy=True)
        point[2] = hover_z
        return point
    if segment_type == "hover_travel":
        point = np.array(strokes[occurrence + 1].desired_path[0], copy=True)
        point[2] = hover_z
        return point
    if segment_type == "descent":
        return np.array(strokes[occurrence + 1].desired_path[0], copy=True)
    if segment_type == "final_lift":
        point = np.array(strokes[-1].desired_path[-1], copy=True)
        point[2] = hover_z
        return point
    raise ValidationError(f"Unsupported transition type: {segment_type}")


def validate_transition_geometry(
    strokes: Sequence[SourceStroke],
    desired_position: np.ndarray,
    timestamps: np.ndarray,
    manifest: Sequence[ManifestRow],
    hover_z: float,
) -> None:
    first = manifest[0]
    if (
        first.segment_type != "initial_hover"
        or first.start_sample != 0
        or first.end_sample != 0
    ):
        raise ValidationError("Initial hover must contain exactly one sample")
    expected_hover = np.array(strokes[0].desired_path[0], copy=True)
    expected_hover[2] = hover_z
    assert_close(
        desired_position[0],
        expected_hover,
        "initial hover desired position",
        atol=ATOL,
        rtol=0.0,
    )

    occurrences = {name: 0 for name in TRANSITION_TYPES}
    for row_index, row in enumerate(manifest):
        if row.segment_type not in TRANSITION_TYPES:
            continue
        if row.start_sample == 0:
            raise ValidationError("Transition has no preceding boundary sample")
        preceding = row.start_sample - 1
        occurrence = occurrences[row.segment_type]
        endpoint = transition_endpoint(
            row.segment_type,
            occurrence,
            strokes,
            hover_z,
        )
        occurrences[row.segment_type] += 1
        validation_end = row.end_sample
        if (
            row.segment_type in {"initial_descent", "descent"}
            and row_index + 1 < len(manifest)
            and manifest[row_index + 1].segment_type == "drawing_stroke"
        ):
            validation_end += 1
        u = (
            timestamps[row.start_sample : validation_end + 1]
            - timestamps[preceding]
        ) / row.planned_duration
        if np.any(u <= 0.0) or np.any(u > 1.0 + 1.0e-10):
            raise ValidationError("Transition normalized times are invalid")
        if not math.isclose(
            float(u[-1]), 1.0, rel_tol=0.0, abs_tol=1.0e-9
        ):
            raise ValidationError("Transition does not end at planned duration")
        start_position = desired_position[preceding]
        expected = start_position + quintic(u)[:, None] * (
            endpoint - start_position
        )
        assert_close(
            desired_position[row.start_sample : validation_end + 1],
            expected,
            f"{row.segment_type} quintic desired positions",
            atol=1.0e-9,
            rtol=1.0e-9,
        )
        points = desired_position[preceding : validation_end + 1]
        if row.segment_type in {"lift", "final_lift"}:
            if not np.allclose(
                points[:, :2], points[0, :2], atol=ATOL, rtol=0.0
            ) or np.any(np.diff(points[:, 2]) < -ATOL):
                raise ValidationError(
                    f"{row.segment_type} must keep XY fixed and increase Z"
                )
        elif row.segment_type in {"initial_descent", "descent"}:
            if not np.allclose(
                points[:, :2], points[0, :2], atol=ATOL, rtol=0.0
            ) or np.any(np.diff(points[:, 2]) > ATOL):
                raise ValidationError(
                    f"{row.segment_type} must keep XY fixed and decrease Z"
                )
        elif row.segment_type == "hover_travel":
            if not np.allclose(
                points[:, 2], hover_z, atol=ATOL, rtol=0.0
            ):
                raise ValidationError("hover_travel must remain at hover_z")


def make_authoritative_robot(urdf_path: Path) -> Any:
    """Load the authoritative project robot model."""
    try:
        context = v7_evaluator.make_robot_context(urdf_path)
    except Exception as exc:
        raise ValidationError(
            f"make_robot_context failed for {urdf_path}: {exc}"
        ) from exc

    if not hasattr(context, "robot"):
        raise ValidationError(
            "make_robot_context did not return the expected RobotContext"
        )

    return context.robot


def joint_limit_arrays(
    robot: Any,
    joint_order: Sequence[str],
) -> Tuple[np.ndarray, np.ndarray]:
    try:
        bounds = np.asarray(
            get_joint_bounds(
                robot,
                tuple(joint_order),
                -np.pi,
                np.pi,
            ),
            dtype=np.float64,
        )
        lower = bounds[:, 0]
        upper = bounds[:, 1]
    except (TypeError, ValueError, IndexError) as exc:
        raise ValidationError(f"Cannot obtain authoritative joint limits: {exc}") from exc
    if (
        lower.shape != (JOINT_DIM,)
        or upper.shape != (JOINT_DIM,)
        or not np.all(np.isfinite(lower))
        or not np.all(np.isfinite(upper))
        or np.any(lower > upper)
    ):
        raise ValidationError("Authoritative joint limits are invalid")
    return lower, upper


def validate_full_csv(
    path: Path,
    timestamps: np.ndarray,
    q: np.ndarray,
    desired: np.ndarray,
    fk_position: np.ndarray,
    cartesian_error: np.ndarray,
    orientation_error: np.ndarray,
    z_error: np.ndarray,
    segment_indices: np.ndarray,
    segment_types: np.ndarray,
    stroke_indices: np.ndarray,
) -> None:
    rows = read_csv(path, FULL_CSV_COLUMNS)
    if len(rows) != len(timestamps):
        raise ValidationError("Full CSV row count differs from full NPZ")
    try:
        sample_index = np.asarray([int(row["sample_index"]) for row in rows])
        csv_timestamps = np.asarray(
            [float(row["time_seconds"]) for row in rows]
        )
        csv_q = np.asarray(
            [
                [float(row[f"q{joint + 1}"]) for joint in range(JOINT_DIM)]
                for row in rows
            ]
        )
        csv_desired = np.asarray(
            [
                [
                    float(row["desired_x"]),
                    float(row["desired_y"]),
                    float(row["desired_z"]),
                ]
                for row in rows
            ]
        )
        csv_fk = np.asarray(
            [
                [
                    float(row["fk_x"]),
                    float(row["fk_y"]),
                    float(row["fk_z"]),
                ]
                for row in rows
            ]
        )
        csv_cartesian = np.asarray(
            [float(row["cartesian_error_m"]) for row in rows]
        )
        csv_orientation = np.asarray(
            [float(row["orientation_error_rad"]) for row in rows]
        )
        csv_z = np.asarray(
            [float(row["z_tracking_error_m"]) for row in rows]
        )
        csv_segment_index = np.asarray(
            [int(row["segment_index"]) for row in rows]
        )
        csv_segment_type = np.asarray(
            [row["segment_type"] for row in rows], dtype=str
        )
        csv_stroke_index = np.asarray(
            [int(row["stroke_index"]) for row in rows]
        )
    except (TypeError, ValueError) as exc:
        raise ValidationError("Full CSV contains invalid values") from exc
    assert_exact(sample_index, np.arange(len(rows)), "CSV sample_index")
    for actual, expected, label in (
        (csv_timestamps, timestamps, "CSV timestamps"),
        (csv_q, q, "CSV q"),
        (csv_desired, desired, "CSV desired_position"),
        (csv_fk, fk_position, "CSV fk_position"),
        (csv_cartesian, cartesian_error, "CSV cartesian_error_m"),
        (csv_orientation, orientation_error, "CSV orientation_error_rad"),
        (csv_z, z_error, "CSV z_tracking_error_m"),
        (csv_segment_index, segment_indices, "CSV segment_index"),
        (csv_segment_type, segment_types, "CSV segment_type"),
        (csv_stroke_index, stroke_indices, "CSV stroke_index"),
    ):
        assert_exact(actual, expected, label)


def validate_approved_outputs(
    output_dir: Path,
    accepted: bool,
    timestamps: np.ndarray,
    q: np.ndarray,
    urdf_path: Path,
    urdf_sha256: str,
) -> None:
    csv_path = output_dir / APPROVED_CSV_NAME
    npz_path = output_dir / APPROVED_NPZ_NAME
    if not accepted:
        if csv_path.exists() or npz_path.exists():
            raise ValidationError(
                "Rejected execution must not have approved exports"
            )
        return
    require_file(csv_path)
    require_file(npz_path)
    rows = read_csv(csv_path, APPROVED_CSV_COLUMNS)
    if len(rows) != len(timestamps):
        raise ValidationError("Approved CSV row count differs from full trajectory")
    try:
        approved_timestamps = np.asarray(
            [float(row["time_seconds"]) for row in rows]
        )
        approved_q = np.asarray(
            [
                [float(row[f"q{joint + 1}"]) for joint in range(JOINT_DIM)]
                for row in rows
            ]
        )
    except (TypeError, ValueError) as exc:
        raise ValidationError("Approved CSV contains invalid values") from exc
    assert_exact(approved_timestamps, timestamps, "approved CSV timestamps")
    assert_exact(approved_q, q, "approved CSV q")

    approved = load_npz(npz_path)
    approved_fields = {
        "timestamps",
        "q",
        "accepted",
        "verdict",
        "urdf_path",
        "urdf_sha256",
    }
    require_fields(approved, approved_fields, str(npz_path), exact=True)
    assert_exact(
        finite_array(
            approved["timestamps"],
            "approved NPZ timestamps",
            shape=timestamps.shape,
        ),
        timestamps,
        "approved NPZ timestamps",
    )
    assert_exact(
        finite_array(approved["q"], "approved NPZ q", shape=q.shape),
        q,
        "approved NPZ q",
    )
    if not bool_scalar(approved["accepted"], "approved NPZ accepted"):
        raise ValidationError("Approved NPZ accepted must be true")
    if text_scalar(approved["verdict"], "approved NPZ verdict") != ACCEPTED_VERDICT:
        raise ValidationError("Approved NPZ verdict is invalid")
    if (
        Path(text_scalar(approved["urdf_path"], "approved NPZ urdf_path"))
        .expanduser()
        .resolve()
        != urdf_path
    ):
        raise ValidationError("Approved NPZ URDF path differs from full NPZ")
    if (
        text_scalar(approved["urdf_sha256"], "approved NPZ urdf_sha256")
        != urdf_sha256
    ):
        raise ValidationError("Approved NPZ URDF hash differs from full NPZ")


def validate_legacy_smartjoint_csv(
    output_dir: Path,
    accepted: bool,
    timestamps: np.ndarray,
    q: np.ndarray,
    segment_types: np.ndarray,
    stroke_indices: np.ndarray,
    manifest: Sequence[ManifestRow],
    metrics: Mapping[str, Any],
    stroke_count: int,
) -> None:
    legacy_path = output_dir / LEGACY_CSV_NAME
    required_metric_fields = (
        "legacy_smartjoint_csv",
        "legacy_smartjoint_row_count",
        "legacy_drawing_stroke_ids",
    )
    missing_metrics = [
        name for name in required_metric_fields if name not in metrics
    ]
    if missing_metrics:
        raise ValidationError(
            f"Metrics are missing legacy SmartJoint fields: {missing_metrics}"
        )

    recorded_path = metrics["legacy_smartjoint_csv"]
    recorded_row_count = metrics["legacy_smartjoint_row_count"]
    recorded_drawing_ids = metrics["legacy_drawing_stroke_ids"]
    if not accepted:
        if legacy_path.exists():
            raise ValidationError(
                "Rejected execution must not contain SmartJoint_Data_diffusion.csv"
            )
        if recorded_path is not None:
            raise ValidationError(
                "Rejected metrics legacy_smartjoint_csv must be null"
            )
        if recorded_row_count != 0:
            raise ValidationError(
                "Rejected metrics legacy_smartjoint_row_count must be 0"
            )
        if recorded_drawing_ids != []:
            raise ValidationError(
                "Rejected metrics legacy_drawing_stroke_ids must be empty"
            )
        return

    require_file(legacy_path)
    if not isinstance(recorded_path, str) or not recorded_path:
        raise ValidationError(
            "Accepted metrics legacy_smartjoint_csv must be a path string"
        )
    if Path(recorded_path).expanduser().resolve() != legacy_path.resolve():
        raise ValidationError(
            "Metrics legacy_smartjoint_csv does not resolve to the legacy CSV"
        )
    if (
        isinstance(recorded_row_count, bool)
        or not isinstance(recorded_row_count, int)
        or recorded_row_count != len(timestamps)
    ):
        raise ValidationError(
            "Metrics legacy_smartjoint_row_count is inconsistent"
        )
    expected_drawing_ids = list(range(1, stroke_count + 1))
    if recorded_drawing_ids != expected_drawing_ids:
        raise ValidationError(
            "Metrics legacy_drawing_stroke_ids is inconsistent"
        )

    rows = read_csv(legacy_path, LEGACY_CSV_COLUMNS)
    if len(rows) != len(timestamps):
        raise ValidationError(
            "Legacy SmartJoint row count differs from full trajectory"
        )
    for row_index, row in enumerate(rows):
        empty_columns = [
            name
            for name in LEGACY_CSV_COLUMNS
            if row.get(name) is None or row[name] == ""
        ]
        if empty_columns:
            raise ValidationError(
                f"Legacy SmartJoint row {row_index} has empty values: "
                f"{empty_columns}"
            )
    try:
        legacy_timestamps = np.asarray(
            [float(row["Timestamp"]) for row in rows],
            dtype=np.float64,
        )
        legacy_q = np.asarray(
            [
                [
                    float(row[f"joint{joint + 1}"])
                    for joint in range(JOINT_DIM)
                ]
                for row in rows
            ],
            dtype=np.float64,
        )
    except (TypeError, ValueError) as exc:
        raise ValidationError(
            "Legacy SmartJoint CSV contains invalid numeric values"
        ) from exc
    if not np.all(np.isfinite(legacy_timestamps)) or not np.all(
        np.isfinite(legacy_q)
    ):
        raise ValidationError(
            "Legacy SmartJoint CSV contains non-finite timestamps or joints"
        )
    assert_exact(
        legacy_timestamps,
        timestamps,
        "legacy SmartJoint timestamps",
    )
    assert_exact(legacy_q, q, "legacy SmartJoint q")
    assert_exact(
        legacy_q[0],
        q[0],
        "legacy SmartJoint first joint vector",
    )
    assert_exact(
        legacy_q[-1],
        q[-1],
        "legacy SmartJoint final joint vector",
    )
    if np.any(np.diff(legacy_timestamps) <= 0.0):
        raise ValidationError(
            "Legacy SmartJoint timestamps must be strictly increasing"
        )

    expected_touch_types: List[str] = []
    expected_statuses: List[str] = []
    for row_index, (segment_type, stroke_index_value) in enumerate(
        zip(segment_types.tolist(), stroke_indices.tolist())
    ):
        segment_type = str(segment_type)
        if segment_type not in SEGMENT_TYPES:
            raise ValidationError(
                f"Legacy source row {row_index} has unexpected segment type "
                f"{segment_type!r}"
            )
        stroke_index = int(stroke_index_value)
        if segment_type == "initial_hover":
            touch_type = "Air"
            status = "RECORDING_START"
        elif segment_type == "drawing_stroke":
            if stroke_index < 0 or stroke_index >= stroke_count:
                raise ValidationError(
                    f"Drawing row {row_index} has invalid stroke index"
                )
            touch_type = "Pen"
            status = f"DRAWING_STROKE_{stroke_index + 1}"
        elif segment_type in {"lift", "final_lift"}:
            if stroke_index < 0 or stroke_index >= stroke_count:
                raise ValidationError(
                    f"Lift row {row_index} has invalid stroke index"
                )
            touch_type = "Air"
            status = f"END_STROKE_{stroke_index + 1}"
        else:
            touch_type = "Air"
            status = "MOVING_FAST"
        expected_touch_types.append(touch_type)
        expected_statuses.append(status)

    actual_touch_types = [row["TouchType"] for row in rows]
    actual_statuses = [row["OriginalStatus"] for row in rows]
    for row_index, (actual, expected) in enumerate(
        zip(actual_touch_types, expected_touch_types)
    ):
        if actual != expected:
            raise ValidationError(
                f"Legacy SmartJoint TouchType mismatch at row {row_index}: "
                f"expected {expected!r}, got {actual!r}"
            )
    for row_index, (actual, expected) in enumerate(
        zip(actual_statuses, expected_statuses)
    ):
        if actual != expected:
            raise ValidationError(
                f"Legacy SmartJoint OriginalStatus mismatch at row {row_index}: "
                f"expected {expected!r}, got {actual!r}"
            )

    drawing_manifest = [
        row for row in manifest if row.segment_type == "drawing_stroke"
    ]
    if len(drawing_manifest) != stroke_count:
        raise ValidationError(
            "Legacy validation drawing manifest count differs from stroke count"
        )
    for stroke_number, row in enumerate(drawing_manifest, start=1):
        if row.stroke_index != stroke_number - 1:
            raise ValidationError(
                "Legacy validation drawing manifest stroke order is invalid"
            )
        if row.sample_count != 100:
            raise ValidationError(
                f"DRAWING_STROKE_{stroke_number} manifest must contain 100 rows"
            )
        label = f"DRAWING_STROKE_{stroke_number}"
        selection = slice(row.start_sample, row.end_sample + 1)
        if actual_touch_types[row.start_sample] != "Pen":
            raise ValidationError(
                f"{label} manifest start sample must be Pen"
            )
        if actual_touch_types[row.end_sample] != "Pen":
            raise ValidationError(
                f"{label} manifest end sample must be Pen"
            )
        if not all(
            value == "Pen" for value in actual_touch_types[selection]
        ) or not all(
            value == label for value in actual_statuses[selection]
        ):
            raise ValidationError(
                f"{label} manifest range has incorrect legacy ownership"
            )
        labelled_rows = [
            index
            for index, status in enumerate(actual_statuses)
            if status == label
        ]
        expected_rows = list(range(row.start_sample, row.end_sample + 1))
        if labelled_rows != expected_rows or len(labelled_rows) != 100:
            raise ValidationError(
                f"{label} must occur exactly within its 100-row manifest range"
            )

    report_path = output_dir / REPORT_NAME
    try:
        report = report_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ValidationError(f"Cannot read report {report_path}: {exc}") from exc
    expected_report_lines = (
        f"legacy_smartjoint_csv: {recorded_path}",
        f"legacy_smartjoint_rows: {recorded_row_count}",
        f"legacy_drawing_stroke_ids: {recorded_drawing_ids}",
    )
    for line in expected_report_lines:
        if line not in report:
            raise ValidationError(
                f"Execution report is missing or disagrees with {line!r}"
            )


def validate_metrics(
    metrics: Mapping[str, Any],
    accepted: bool,
    sample_count: int,
    stroke_count: int,
    segment_count: int,
    duration: float,
) -> None:
    metrics_accepted = parse_json_verdict(
        metrics,
        accepted_verdict=ACCEPTED_VERDICT,
        rejected_verdict=REJECTED_VERDICT,
        label="metrics",
    )
    if metrics_accepted != accepted:
        raise ValidationError("Metrics verdict differs from full NPZ")
    expected_values: Tuple[Tuple[str, Any], ...] = (
        ("sample_count", sample_count),
        ("stroke_count", stroke_count),
        ("segment_count", segment_count),
    )
    for key, expected in expected_values:
        if key in metrics and metrics[key] != expected:
            raise ValidationError(f"metrics.{key} is inconsistent")
    if "duration_seconds" in metrics:
        value = metrics["duration_seconds"]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isclose(
                float(value), duration, rel_tol=RTOL, abs_tol=ATOL
            )
        ):
            raise ValidationError("metrics.duration_seconds is inconsistent")


def validate_artifact(args: argparse.Namespace) -> bool:
    output_dir = args.output_dir.expanduser().resolve()
    if not output_dir.is_dir():
        raise ValidationError(f"Output directory does not exist: {output_dir}")
    paths = {
        "csv": output_dir / FULL_CSV_NAME,
        "npz": output_dir / FULL_NPZ_NAME,
        "manifest": output_dir / MANIFEST_NAME,
        "metrics": output_dir / METRICS_NAME,
        "report": output_dir / REPORT_NAME,
    }
    for path in paths.values():
        require_file(path)
    if paths["report"].stat().st_size == 0:
        raise ValidationError("Execution report is empty")

    full = load_npz(paths["npz"])
    require_fields(full, FULL_NPZ_FIELDS, str(paths["npz"]), exact=True)
    timestamps = finite_array(full["timestamps"], "timestamps", ndim=1)
    q = finite_array(full["q"], "q", ndim=2)
    sample_count = len(timestamps)
    if sample_count < 3 or q.shape != (sample_count, JOINT_DIM):
        raise ValidationError("Combined q/timestamps shape is invalid")
    if timestamps[0] != 0.0 or np.any(np.diff(timestamps) <= 0.0):
        raise ValidationError(
            "Combined timestamps must start at zero and strictly increase"
        )
    fk_position = finite_array(
        full["fk_position"], "fk_position", shape=(sample_count, 3)
    )
    fk_rotation = finite_array(
        full["fk_rotation_matrix"],
        "fk_rotation_matrix",
        shape=(sample_count, 3, 3),
    )
    fk_quaternion = finite_array(
        full["fk_quaternion"],
        "fk_quaternion",
        shape=(sample_count, 4),
    )
    target_rotation = finite_array(
        full["target_rotation_matrix"],
        "target_rotation_matrix",
        shape=(3, 3),
    )
    target_quaternion = finite_array(
        full["target_quaternion"], "target_quaternion", shape=(4,)
    )
    target_rpy = finite_array(full["target_rpy"], "target_rpy", shape=(3,))
    validate_rotation(target_rotation, "target_rotation_matrix")
    desired_rotation = np.repeat(
        target_rotation[np.newaxis, :, :],
        sample_count,
        axis=0,
    )
    desired_position = finite_array(
        full["desired_position"],
        "desired_position",
        shape=(sample_count, 3),
    )
    cartesian_error = finite_array(
        full["cartesian_error_m"],
        "cartesian_error_m",
        shape=(sample_count,),
    )
    orientation_error = finite_array(
        full["orientation_error_rad"],
        "orientation_error_rad",
        shape=(sample_count,),
    )
    z_error = finite_array(
        full["z_tracking_error_m"],
        "z_tracking_error_m",
        shape=(sample_count,),
    )
    segment_indices = np.asarray(full["segment_index_per_sample"])
    stroke_indices = np.asarray(full["stroke_index_per_sample"])
    if not np.issubdtype(segment_indices.dtype, np.integer):
        raise ValidationError("segment_index_per_sample must be integer")
    if not np.issubdtype(stroke_indices.dtype, np.integer):
        raise ValidationError("stroke_index_per_sample must be integer")
    segment_indices = np.asarray(segment_indices, dtype=np.int64)
    stroke_indices = np.asarray(stroke_indices, dtype=np.int64)
    segment_types = string_array(
        full["segment_type_per_sample"],
        "segment_type_per_sample",
    )
    if (
        segment_indices.shape != (sample_count,)
        or stroke_indices.shape != (sample_count,)
        or segment_types.shape != (sample_count,)
    ):
        raise ValidationError("Per-sample segment/stroke arrays have wrong shape")

    joint_order = string_array(full["joint_order"], "joint_order").tolist()
    if len(joint_order) != JOINT_DIM or len(set(joint_order)) != JOINT_DIM:
        raise ValidationError("joint_order must contain six unique names")
    fk_frame = text_scalar(full["fk_frame"], "fk_frame")
    if fk_frame != EXPECTED_FK_FRAME:
        raise ValidationError(
            f"fk_frame must be {EXPECTED_FK_FRAME}, got {fk_frame}"
        )
    urdf_path = Path(
        text_scalar(full["urdf_path"], "urdf_path")
    ).expanduser().resolve()
    require_file(urdf_path)
    urdf_hash = text_scalar(full["urdf_sha256"], "urdf_sha256")
    if sha256_file(urdf_path) != urdf_hash:
        raise ValidationError("Full NPZ URDF SHA-256 mismatch")
    source_directories = string_array(
        full["source_stroke_directories"],
        "source_stroke_directories",
    ).tolist()
    recorded_input_sha = string_array(
        full["source_stroke_input_sha256"],
        "source_stroke_input_sha256",
    ).tolist()
    recorded_path_ids = string_array(
        full["source_stroke_deployment_path_id"],
        "source_stroke_deployment_path_id",
    ).tolist()
    if len(source_directories) < 2:
        raise ValidationError("At least two source strokes are required")
    if (
        len(recorded_input_sha) != len(source_directories)
        or len(recorded_path_ids) != len(source_directories)
    ):
        raise ValidationError("Recorded source metadata list lengths differ")
    strokes = [load_source_stroke(Path(path)) for path in source_directories]
    first = strokes[0]
    for stroke_index, stroke in enumerate(strokes):
        if (
            stroke.urdf_path != urdf_path
            or stroke.urdf_sha256 != urdf_hash
            or stroke.fk_frame != fk_frame
        ):
            raise ValidationError(
                f"Source stroke {stroke_index} robot/FK contract differs"
            )
        assert_exact(
            stroke.target_rotation_matrix,
            target_rotation,
            f"source stroke {stroke_index} target rotation",
        )
        assert_exact(
            stroke.target_quaternion,
            target_quaternion,
            f"source stroke {stroke_index} target quaternion",
        )
        assert_exact(
            stroke.target_rpy,
            target_rpy,
            f"source stroke {stroke_index} target RPY",
        )
        if stroke.target_z != first.target_z:
            raise ValidationError("Source strokes have different drawing Z")
    drawing_z = finite_scalar(full["drawing_z"], "drawing_z")
    hover_z = finite_scalar(full["hover_z"], "hover_z")
    if drawing_z != first.target_z or hover_z <= drawing_z:
        raise ValidationError("drawing_z/hover_z values are inconsistent")

    maximum_joint_step = finite_scalar(
        full["maximum_joint_step_rad"], "maximum_joint_step_rad"
    )
    maximum_cartesian_error = finite_scalar(
        full["maximum_cartesian_error_m"],
        "maximum_cartesian_error_m",
    )
    maximum_orientation_error = finite_scalar(
        full["maximum_orientation_error_rad"],
        "maximum_orientation_error_rad",
    )
    maximum_z_error = finite_scalar(
        full["maximum_z_tracking_error_m"],
        "maximum_z_tracking_error_m",
    )
    if (
        maximum_joint_step <= 0.0
        or maximum_joint_step > 0.20
        or maximum_cartesian_error <= 0.0
        or maximum_orientation_error <= 0.0
        or maximum_z_error <= 0.0
    ):
        raise ValidationError("Acceptance gates are invalid")
    for stroke_index, stroke in enumerate(strokes):
        if maximum_orientation_error > stroke.orientation_gate:
            raise ValidationError(
                f"Orientation gate is weaker than stroke {stroke_index}"
            )
        if maximum_z_error > stroke.z_gate:
            raise ValidationError(
                f"Z gate is weaker than stroke {stroke_index}"
            )

    npz_accepted = bool_scalar(full["accepted"], "full NPZ accepted")
    npz_verdict = text_scalar(full["verdict"], "full NPZ verdict")
    if npz_verdict not in {ACCEPTED_VERDICT, REJECTED_VERDICT}:
        raise ValidationError("Full NPZ verdict is unsupported")
    if npz_accepted != (npz_verdict == ACCEPTED_VERDICT):
        raise ValidationError("Full NPZ accepted and verdict disagree")

    manifest = read_manifest(paths["manifest"])
    validate_manifest(
        manifest,
        timestamps,
        segment_indices,
        segment_types,
        stroke_indices,
        len(strokes),
    )
    validate_drawing_preservation(
        strokes,
        q,
        timestamps,
        desired_position,
        manifest,
        recorded_input_sha,
        recorded_path_ids,
    )
    validate_transition_geometry(
        strokes,
        desired_position,
        timestamps,
        manifest,
        hover_z,
    )

    robot = make_authoritative_robot(urdf_path)
    recomputed_position, recomputed_rotation, recomputed_quaternion = (
        trajectory_full_transform_fk(
            robot,
            q,
            tuple(joint_order),
            fk_frame,
        )
    )
    recomputed_position = finite_array(
        recomputed_position,
        "recomputed fk_position",
        shape=(sample_count, 3),
    )
    recomputed_rotation = finite_array(
        recomputed_rotation,
        "recomputed fk_rotation_matrix",
        shape=(sample_count, 3, 3),
    )
    recomputed_quaternion = finite_array(
        recomputed_quaternion,
        "recomputed fk_quaternion",
        shape=(sample_count, 4),
    )
    assert_close(fk_position, recomputed_position, "stored fk_position")
    assert_close(fk_rotation, recomputed_rotation, "stored fk_rotation_matrix")
    assert_close(fk_quaternion, recomputed_quaternion, "stored fk_quaternion")
    recomputed_cartesian = np.linalg.norm(
        recomputed_position - desired_position,
        axis=1,
    )
    recomputed_orientation = orientation_error_trajectory(
        target_rotation,
        recomputed_rotation,
    )
    recomputed_z = np.abs(
        recomputed_position[:, 2] - desired_position[:, 2]
    )
    assert_close(
        cartesian_error,
        recomputed_cartesian,
        "stored cartesian_error_m",
    )
    assert_close(
        orientation_error,
        recomputed_orientation,
        "stored orientation_error_rad",
    )
    assert_close(z_error, recomputed_z, "stored z_tracking_error_m")

    velocity = np.gradient(q, timestamps, axis=0, edge_order=2)
    acceleration = np.gradient(
        velocity,
        timestamps,
        axis=0,
        edge_order=2,
    )
    jerk = np.gradient(
        acceleration,
        timestamps,
        axis=0,
        edge_order=2,
    )
    assert_close(
        finite_array(
            full["joint_velocity"],
            "joint_velocity",
            shape=q.shape,
        ),
        velocity,
        "stored joint_velocity",
    )
    assert_close(
        finite_array(
            full["joint_acceleration"],
            "joint_acceleration",
            shape=q.shape,
        ),
        acceleration,
        "stored joint_acceleration",
    )
    assert_close(
        finite_array(full["joint_jerk"], "joint_jerk", shape=q.shape),
        jerk,
        "stored joint_jerk",
    )
    observed_joint_step = float(np.max(np.abs(np.diff(q, axis=0))))
    lower, upper = joint_limit_arrays(robot, joint_order)
    limit_violation = bool(
        np.any(q < lower[None, :] - HARD_JOINT_LIMIT_TOLERANCE_RAD)
        or np.any(q > upper[None, :] + HARD_JOINT_LIMIT_TOLERANCE_RAD)
    )

    recomputed_accepted = bool(
        not limit_violation
        and observed_joint_step <= maximum_joint_step
        and float(np.max(recomputed_cartesian)) <= maximum_cartesian_error
        and float(np.max(recomputed_orientation)) <= maximum_orientation_error
        and float(np.max(recomputed_z)) <= maximum_z_error
    )
    if recomputed_accepted != npz_accepted:
        raise ValidationError(
            "Full NPZ verdict disagrees with independently recomputed gates"
        )

    validate_full_csv(
        paths["csv"],
        timestamps,
        q,
        desired_position,
        fk_position,
        cartesian_error,
        orientation_error,
        z_error,
        segment_indices,
        segment_types,
        stroke_indices,
    )
    metrics = load_json(paths["metrics"])
    validate_metrics(
        metrics,
        npz_accepted,
        sample_count,
        len(strokes),
        len(manifest),
        float(timestamps[-1]),
    )
    validate_legacy_smartjoint_csv(
        output_dir,
        npz_accepted,
        timestamps,
        q,
        segment_types,
        stroke_indices,
        manifest,
        metrics,
        len(strokes),
    )
    validate_approved_outputs(
        output_dir,
        npz_accepted,
        timestamps,
        q,
        urdf_path,
        urdf_hash,
    )
    return npz_accepted


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--require_accepted",
        action="store_true",
        help="Return exit code 2 when the validated execution is rejected.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        accepted = validate_artifact(args)
    except ValidationError as exc:
        print(f"MULTISTROKE_EXECUTION_VALIDATION_FAILED: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(
            f"MULTISTROKE_EXECUTION_VALIDATION_RUNTIME_FAILURE: {exc}",
            file=sys.stderr,
        )
        return 1
    print("MULTISTROKE_EXECUTION_VALIDATION_PASSED")
    if not accepted:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
