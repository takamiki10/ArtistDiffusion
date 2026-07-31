#!/usr/bin/env python3
# Builds an auditable multi-stroke full-pose execution artifact.
"""Combine approved full-pose drawing strokes with safe hover transitions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from generate_adaptive_mlp_ik_bootstrap_prior import (
    SEQUENTIAL_IK_DEFAULT_RETRIES,
)
from generate_ik_seed_path import (
    DEFAULT_EE_LINK,
    DEFAULT_JOINT_NAMES,
    HARD_JOINT_LIMIT_TOLERANCE_RAD,
    check_joint_limits,
    get_joint_bounds,
    load_robot,
)
from orientation_aware_adaptive_ik import (  # pyright: ignore[reportMissingImports]
    adaptive_refine_full_pose_path,
    orientation_error_trajectory,
    target_orientation_from_rpy,
    trajectory_full_transform_fk,
)


ACCEPTED_INPUT_VERDICT = "V8_1_DEPLOYMENT_TRAJECTORY_ACCEPTED"
ACCEPTED_VERDICT = "MULTISTROKE_EXECUTION_ACCEPTED"
REJECTED_VERDICT = "MULTISTROKE_EXECUTION_REJECTED"
TRAJECTORY_LENGTH = 100
JOINT_DIM = 6
XYZ_DIM = 3
MAXIMUM_ALLOWED_JOINT_STEP_RAD = 0.20
ENDPOINT_ATOL = 1.0e-9
ARRAY_RTOL = 1.0e-7
ARRAY_ATOL = 1.0e-9
ALLOWED_SEGMENT_TYPES = (
    "initial_hover",
    "initial_descent",
    "drawing_stroke",
    "lift",
    "hover_travel",
    "descent",
    "final_lift",
)
LEGACY_SMARTJOINT_COLUMNS = (
    "Timestamp",
    "TouchType",
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
    "OriginalStatus",
)
LEGACY_SMARTJOINT_FILENAME = "SmartJoint_Data_diffusion.csv"

TRANSITION_DIAGNOSTICS: List[Dict[str, Any]] = []


@dataclass(frozen=True)
class Stroke:
    output_dir: Path
    full_path: Path
    metrics_path: Path
    final_q: np.ndarray
    final_ee: np.ndarray
    desired_path: np.ndarray
    timestamps: np.ndarray
    target_rpy: np.ndarray
    target_quaternion: np.ndarray
    target_rotation_matrix: np.ndarray
    target_z: float
    maximum_orientation_error_gate_rad: float
    maximum_z_error_gate_m: float
    urdf_path: Path
    urdf_sha256: str
    input_sha256: str
    deployment_path_id: str
    joint_order: Tuple[str, ...]
    fk_frame: str


@dataclass
class LocalSegment:
    segment_type: str
    stroke_index: int
    local_times: np.ndarray
    q: np.ndarray
    desired_position: np.ndarray
    planned_duration_seconds: float


PENDING_COUPLED_DESCENTS: Dict[
    int,
    Tuple[np.ndarray, np.ndarray, LocalSegment, Dict[str, Any]],
] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stroke_output_dirs",
        nargs="+",
        type=Path,
        required=True,
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--hover_offset_z", type=float, default=0.05)
    parser.add_argument(
        "--transition_sample_period",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--initial_descent_duration",
        type=float,
        default=1.5,
    )
    parser.add_argument("--lift_duration", type=float, default=1.0)
    parser.add_argument("--travel_duration", type=float, default=2.0)
    parser.add_argument("--descent_duration", type=float, default=1.0)
    parser.add_argument("--final_lift_duration", type=float, default=1.0)
    parser.add_argument(
        "--maximum_joint_step_rad",
        type=float,
        default=0.20,
    )
    parser.add_argument(
        "--maximum_cartesian_error_m",
        type=float,
        default=0.001,
    )
    parser.add_argument(
        "--maximum_orientation_error_rad",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--maximum_z_tracking_error_m",
        type=float,
        default=0.001,
    )
    parser.add_argument(
        "--retry_profile",
        choices=("standard", "robust"),
        default="robust",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def scalar_text(value: Any) -> str:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError("Expected scalar string value")
    item = array.reshape(-1)[0]
    return item.decode("utf-8") if isinstance(item, bytes) else str(item)


def scalar_bool(value: Any, name: str) -> bool:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"{name} must be a scalar boolean")
    item = array.reshape(-1)[0]
    if not isinstance(item, (bool, np.bool_)):
        raise ValueError(f"{name} must be stored as a real boolean")
    return bool(item)


def finite_array(
    name: str,
    value: Any,
    shape: Tuple[int, ...],
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains nonfinite values")
    return array


def strict_json(path: Path) -> Mapping[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"{path}: non-strict JSON constant {value}")

    value = json.loads(path.read_text(), parse_constant=reject_constant)
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def recorded_joint_order(
    full: Mapping[str, Any],
    metrics: Mapping[str, Any],
) -> Tuple[str, ...]:
    for source in (full, metrics):
        for key in ("joint_order", "joint_ordering", "joint_names"):
            if key not in source:
                continue
            values = np.asarray(source[key]).reshape(-1)
            return tuple(
                item.decode("utf-8") if isinstance(item, bytes) else str(item)
                for item in values
            )
    # The deployment format fixes q1..q6 to the repository's active order.
    return tuple(str(name) for name in DEFAULT_JOINT_NAMES)


def read_approved_csv(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        expected = ["time_seconds", *[f"q{i}" for i in range(1, 7)]]
        if reader.fieldnames != expected:
            raise ValueError(
                f"{path} columns must be exactly {expected}; "
                f"found {reader.fieldnames}"
            )
        rows = list(reader)
    if len(rows) != TRAJECTORY_LENGTH:
        raise ValueError(f"{path} must contain 100 rows")
    timestamps = np.asarray(
        [float(row["time_seconds"]) for row in rows],
        dtype=np.float64,
    )
    q = np.asarray(
        [
            [float(row[f"q{i}"]) for i in range(1, 7)]
            for row in rows
        ],
        dtype=np.float64,
    )
    return timestamps, q


def load_stroke(output_dir: Path) -> Stroke:
    directory = output_dir.expanduser().resolve()
    required = (
        "deployment_trajectory_full.npz",
        "deployment_metrics.json",
        "approved_simulation_trajectory.csv",
        "approved_simulation_trajectory.npz",
    )
    missing = [name for name in required if not (directory / name).is_file()]
    if missing:
        raise FileNotFoundError(f"{directory} lacks required files: {missing}")
    full_path = directory / required[0]
    metrics_path = directory / required[1]
    metrics = strict_json(metrics_path)
    if str(metrics.get("verdict")) != ACCEPTED_INPUT_VERDICT:
        raise ValueError(f"{directory} is not an accepted deployment")
    if not isinstance(metrics.get("accepted"), bool) or not metrics["accepted"]:
        raise ValueError(f"{directory} metrics accepted must be boolean True")

    with np.load(full_path, allow_pickle=False) as archive:
        final_q = finite_array(
            f"{directory}:final_q",
            archive["final_q"],
            (TRAJECTORY_LENGTH, JOINT_DIM),
        )
        final_ee = finite_array(
            f"{directory}:final_ee",
            archive["final_ee"],
            (TRAJECTORY_LENGTH, XYZ_DIM),
        )
        desired_path = finite_array(
            f"{directory}:desired_path",
            archive["desired_path"],
            (TRAJECTORY_LENGTH, XYZ_DIM),
        )
        timestamps = finite_array(
            f"{directory}:timestamps",
            archive["timestamps"],
            (TRAJECTORY_LENGTH,),
        )
        target_rpy = finite_array(
            f"{directory}:target_rpy",
            archive["target_rpy"],
            (3,),
        )
        target_quaternion = finite_array(
            f"{directory}:target_quaternion",
            archive["target_quaternion"],
            (4,),
        )
        target_rotation = finite_array(
            f"{directory}:target_rotation_matrix",
            archive["target_rotation_matrix"],
            (3, 3),
        )
        target_z = float(np.asarray(archive["target_z"]).item())
        orientation_gate = float(
            np.asarray(
                archive["maximum_orientation_error_gate_rad"]
            ).item()
        )
        z_gate = float(
            np.asarray(archive["maximum_z_error_gate_m"]).item()
        )
        if not scalar_bool(
            archive["orientation_constraint_enforced"],
            "orientation_constraint_enforced",
        ):
            raise ValueError(f"{directory} does not enforce orientation")
        if not scalar_bool(
            archive["z_constraint_enforced"],
            "z_constraint_enforced",
        ):
            raise ValueError(f"{directory} does not enforce fixed Z")
        orientation_frame = scalar_text(archive["orientation_fk_frame"])
        z_frame = scalar_text(archive["z_fk_frame"])
        urdf_path = Path(scalar_text(archive["urdf_path"])).resolve()
        urdf_sha256 = scalar_text(archive["urdf_sha256"])
        input_sha256 = scalar_text(archive["input_sha256"])
        deployment_path_id = scalar_text(archive["deployment_path_id"])
        joint_order = recorded_joint_order(archive, metrics)

    if orientation_frame != DEFAULT_EE_LINK or z_frame != DEFAULT_EE_LINK:
        raise ValueError(
            f"{directory} FK frames must both be {DEFAULT_EE_LINK}"
        )
    if not np.all(np.diff(timestamps) > 0.0):
        raise ValueError(f"{directory} timestamps are not strictly increasing")
    if not np.allclose(
        desired_path[:, 2],
        target_z,
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError(f"{directory} drawing Z is not fixed at target_z")
    expected_quaternion, expected_rotation = target_orientation_from_rpy(
        *target_rpy.tolist()
    )
    if not np.allclose(
        target_quaternion,
        expected_quaternion,
        rtol=0.0,
        atol=1.0e-10,
    ) or not np.allclose(
        target_rotation,
        expected_rotation,
        rtol=0.0,
        atol=1.0e-10,
    ):
        raise ValueError(f"{directory} target orientation representations differ")
    if not urdf_path.is_file():
        raise FileNotFoundError(f"Recorded URDF does not exist: {urdf_path}")
    if sha256_file(urdf_path) != urdf_sha256:
        raise ValueError(f"{directory} recorded URDF SHA-256 does not match")
    approved_csv_t, approved_csv_q = read_approved_csv(
        directory / "approved_simulation_trajectory.csv"
    )
    if not np.array_equal(approved_csv_q, final_q) or not np.allclose(
        approved_csv_t,
        timestamps,
        rtol=0.0,
        atol=1.0e-9,
    ):
        raise ValueError(f"{directory} approved CSV differs from full NPZ")
    with np.load(
        directory / "approved_simulation_trajectory.npz",
        allow_pickle=False,
    ) as approved:
        approved_q = finite_array(
            f"{directory}:approved q",
            approved["q"],
            (TRAJECTORY_LENGTH, JOINT_DIM),
        )
        approved_t = finite_array(
            f"{directory}:approved timestamps",
            approved["timestamps"],
            (TRAJECTORY_LENGTH,),
        )
        if not np.array_equal(approved_q, final_q) or not np.allclose(
            approved_t,
            timestamps,
            rtol=0.0,
            atol=1.0e-9,
        ):
            raise ValueError(f"{directory} approved NPZ differs from full NPZ")
    return Stroke(
        output_dir=directory,
        full_path=full_path,
        metrics_path=metrics_path,
        final_q=final_q,
        final_ee=final_ee,
        desired_path=desired_path,
        timestamps=timestamps,
        target_rpy=target_rpy,
        target_quaternion=target_quaternion,
        target_rotation_matrix=target_rotation,
        target_z=target_z,
        maximum_orientation_error_gate_rad=orientation_gate,
        maximum_z_error_gate_m=z_gate,
        urdf_path=urdf_path,
        urdf_sha256=urdf_sha256,
        input_sha256=input_sha256,
        deployment_path_id=deployment_path_id,
        joint_order=joint_order,
        fk_frame=orientation_frame,
    )


def require_compatible(strokes: Sequence[Stroke]) -> None:
    first = strokes[0]
    for index, stroke in enumerate(strokes[1:], start=2):
        scalar_pairs = (
            ("target_z", stroke.target_z, first.target_z),
            (
                "maximum_orientation_error_gate_rad",
                stroke.maximum_orientation_error_gate_rad,
                first.maximum_orientation_error_gate_rad,
            ),
            (
                "maximum_z_error_gate_m",
                stroke.maximum_z_error_gate_m,
                first.maximum_z_error_gate_m,
            ),
        )
        for name, actual, expected in scalar_pairs:
            if actual != expected:
                raise ValueError(
                    f"Stroke {index} {name}={actual} differs from {expected}"
                )
        for name, actual, expected in (
            ("target_rpy", stroke.target_rpy, first.target_rpy),
            (
                "target_quaternion",
                stroke.target_quaternion,
                first.target_quaternion,
            ),
            (
                "target_rotation_matrix",
                stroke.target_rotation_matrix,
                first.target_rotation_matrix,
            ),
        ):
            if not np.array_equal(actual, expected):
                raise ValueError(f"Stroke {index} {name} is incompatible")
        if (
            stroke.urdf_path != first.urdf_path
            or stroke.urdf_sha256 != first.urdf_sha256
            or stroke.joint_order != first.joint_order
            or stroke.fk_frame != first.fk_frame
        ):
            raise ValueError(
                f"Stroke {index} robot identity/order/frame is incompatible"
            )


def validate_args(args: argparse.Namespace, strokes: Sequence[Stroke]) -> None:
    if len(args.stroke_output_dirs) < 2:
        raise ValueError("--stroke_output_dirs requires at least two directories")
    numeric = {
        "hover_offset_z": args.hover_offset_z,
        "transition_sample_period": args.transition_sample_period,
        "initial_descent_duration": args.initial_descent_duration,
        "lift_duration": args.lift_duration,
        "travel_duration": args.travel_duration,
        "descent_duration": args.descent_duration,
        "final_lift_duration": args.final_lift_duration,
        "maximum_joint_step_rad": args.maximum_joint_step_rad,
        "maximum_cartesian_error_m": args.maximum_cartesian_error_m,
        "maximum_orientation_error_rad": args.maximum_orientation_error_rad,
        "maximum_z_tracking_error_m": args.maximum_z_tracking_error_m,
    }
    invalid = [name for name, value in numeric.items() if not np.isfinite(value)]
    if invalid:
        raise ValueError(f"CLI values must be finite: {invalid}")
    positive = [
        name for name, value in numeric.items() if value <= 0.0
    ]
    if positive:
        raise ValueError(f"CLI values must be positive: {positive}")
    if args.maximum_joint_step_rad > MAXIMUM_ALLOWED_JOINT_STEP_RAD:
        raise ValueError("--maximum_joint_step_rad may not exceed 0.20")
    if (
        args.maximum_orientation_error_rad
        > strokes[0].maximum_orientation_error_gate_rad
    ):
        raise ValueError(
            "--maximum_orientation_error_rad may not weaken the recorded "
            "stroke orientation gate"
        )
    if (
        args.maximum_z_tracking_error_m
        > strokes[0].maximum_z_error_gate_m
    ):
        raise ValueError(
            "--maximum_z_tracking_error_m may not weaken the recorded "
            "stroke Z gate"
        )


def time_grid(duration: float, sample_period: float) -> np.ndarray:
    intervals = max(1, int(math.ceil(duration / sample_period)))
    return np.linspace(0.0, duration, intervals + 1, dtype=np.float64)


def quintic_scale(times: np.ndarray) -> np.ndarray:
    if len(times) == 1:
        return np.zeros(1, dtype=np.float64)
    u = times / float(times[-1])
    return 10.0 * u**3 - 15.0 * u**4 + 6.0 * u**5


def interpolate_position(
    start: np.ndarray,
    end: np.ndarray,
    times: np.ndarray,
) -> np.ndarray:
    scale = quintic_scale(times)[:, None]
    return np.asarray(start)[None, :] + scale * (
        np.asarray(end)[None, :] - np.asarray(start)[None, :]
    )


def solve_pose_path(
    *,
    robot: Any,
    desired: np.ndarray,
    start_q: np.ndarray,
    end_q: np.ndarray,
    joint_names: Sequence[str],
    bounds: Sequence[Tuple[float, float]],
    lower: np.ndarray,
    upper: np.ndarray,
    target_rotation: np.ndarray,
    args: argparse.Namespace,
    seed: int,
) -> np.ndarray:
    length = len(desired)
    blend = np.linspace(0.0, 1.0, length, dtype=np.float64)[:, None]
    canonical = start_q[None, :] + blend * (end_q[None, :] - start_q[None, :])
    result = adaptive_refine_full_pose_path(
        robot=robot,
        desired_path=desired,
        target_rotation=target_rotation,
        canonical_mlp_q=canonical,
        q_start=start_q,
        joint_names=joint_names,
        ee_link=DEFAULT_EE_LINK,
        bounds=bounds,
        lower=lower,
        upper=upper,
        mean_error_gate=args.maximum_cartesian_error_m,
        max_joint_step_gate=args.maximum_joint_step_rad,
        maximum_orientation_error_gate_rad=args.maximum_orientation_error_rad,
        num_ik_retries=SEQUENTIAL_IK_DEFAULT_RETRIES,
        random_seed=seed,
        retry_profile=args.retry_profile,
    )
    if int(result.adaptive_metadata["valid_candidate_count"]) == 0:
        raise RuntimeError(
            "Full-pose transition IK produced no gate-valid candidate: "
            f"{result.candidate_table}"
        )
    if result.unresolved_timestep_count != 0:
        raise RuntimeError("Full-pose transition IK has unresolved timesteps")
    q = np.asarray(result.q, dtype=np.float64).copy()
    # Exact endpoints are assigned only by transition_segment, which performs
    # authoritative post-pinning validation before returning the candidate.
    return q


def solve_hover(
    *,
    robot: Any,
    position: np.ndarray,
    seed_q: np.ndarray,
    joint_names: Sequence[str],
    bounds: Sequence[Tuple[float, float]],
    lower: np.ndarray,
    upper: np.ndarray,
    target_rotation: np.ndarray,
    args: argparse.Namespace,
    seed: int,
) -> np.ndarray:
    result = adaptive_refine_full_pose_path(
        robot=robot,
        desired_path=np.asarray(
            position,
            dtype=np.float64,
        )[None, :].copy(),
        target_rotation=target_rotation,
        canonical_mlp_q=np.asarray(
            seed_q,
            dtype=np.float64,
        )[None, :].copy(),
        q_start=np.asarray(seed_q, dtype=np.float64).copy(),
        joint_names=joint_names,
        ee_link=DEFAULT_EE_LINK,
        bounds=bounds,
        lower=lower,
        upper=upper,
        mean_error_gate=args.maximum_cartesian_error_m,
        max_joint_step_gate=args.maximum_joint_step_rad,
        maximum_orientation_error_gate_rad=args.maximum_orientation_error_rad,
        num_ik_retries=SEQUENTIAL_IK_DEFAULT_RETRIES,
        random_seed=seed,
        retry_profile=args.retry_profile,
    )
    if (
        int(result.adaptive_metadata["valid_candidate_count"]) == 0
        or result.unresolved_timestep_count != 0
    ):
        raise RuntimeError(
            "Could not solve a gate-valid full-pose hover state"
        )
    return np.asarray(result.q[0], dtype=np.float64).copy()


def _transition_segment_one_direction(
    *,
    segment_type: str,
    stroke_index: int,
    start_position: np.ndarray,
    end_position: np.ndarray,
    start_q: np.ndarray,
    end_q: np.ndarray,
    duration: float,
    robot: Any,
    joint_names: Sequence[str],
    bounds: Sequence[Tuple[float, float]],
    lower: np.ndarray,
    upper: np.ndarray,
    target_rotation: np.ndarray,
    args: argparse.Namespace,
    seed: int,
) -> LocalSegment:
    times = time_grid(duration, args.transition_sample_period)
    desired = interpolate_position(start_position, end_position, times)
    q = solve_pose_path(
        robot=robot,
        desired=desired,
        start_q=start_q,
        end_q=end_q,
        joint_names=joint_names,
        bounds=bounds,
        lower=lower,
        upper=upper,
        target_rotation=target_rotation,
        args=args,
        seed=seed,
    )
    return LocalSegment(
        segment_type=segment_type,
        stroke_index=stroke_index,
        local_times=times,
        q=q,
        desired_position=desired,
        planned_duration_seconds=duration,
    )


def drawing_segment(stroke: Stroke, stroke_index: int) -> LocalSegment:
    relative = stroke.timestamps - stroke.timestamps[0]
    return LocalSegment(
        segment_type="drawing_stroke",
        stroke_index=stroke_index,
        local_times=relative,
        q=stroke.final_q.copy(),
        desired_position=stroke.desired_path.copy(),
        planned_duration_seconds=float(relative[-1]),
    )


def prepare_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"{path} exists; pass --overwrite to replace it")
        shutil.rmtree(path)
    path.mkdir(parents=True)
    (path / "plots").mkdir()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


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
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    atomic_write_text(
        path,
        json.dumps(
            json_safe(value),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
    )


def atomic_write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        newline="",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _required_legacy_stroke_number(
    value: Any,
    *,
    segment_type: str,
    row_index: int,
) -> int:
    if isinstance(value, (bool, np.bool_)) or value is None:
        raise ValueError(
            f"Legacy export row {row_index} segment {segment_type!r} "
            "requires a finite integer-like stroke_index"
        )
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Legacy export row {row_index} segment {segment_type!r} "
            "requires a finite integer-like stroke_index"
        ) from exc
    if not math.isfinite(numeric) or not numeric.is_integer() or numeric < 0.0:
        raise ValueError(
            f"Legacy export row {row_index} segment {segment_type!r} "
            "requires a finite nonnegative integer-like stroke_index"
        )
    return int(numeric) + 1


def export_legacy_smartjoint_csv(
    trajectory_rows: Sequence[Mapping[str, Any]],
    output_csv_path: Path,
) -> Tuple[int, List[int]]:
    """Export accepted detailed trajectory rows for legacy ROKAE playback."""
    if not trajectory_rows:
        raise ValueError("Legacy SmartJoint export requires trajectory rows")
    required_source_columns = {
        "time_seconds",
        "q1",
        "q2",
        "q3",
        "q4",
        "q5",
        "q6",
        "segment_type",
        "stroke_index",
    }
    missing_by_row = [
        (row_index, sorted(required_source_columns - set(row)))
        for row_index, row in enumerate(trajectory_rows)
        if required_source_columns - set(row)
    ]
    if missing_by_row:
        row_index, missing = missing_by_row[0]
        raise ValueError(
            f"Legacy export source row {row_index} is missing required "
            f"columns: {missing}"
        )

    source_timestamps: List[float] = []
    source_joint_rows: List[List[float]] = []
    legacy_rows: List[Dict[str, Any]] = []
    drawing_stroke_ids: set[int] = set()
    for row_index, source in enumerate(trajectory_rows):
        try:
            timestamp = float(source["time_seconds"])
            joints = [
                float(source[f"q{joint}"])
                for joint in range(1, JOINT_DIM + 1)
            ]
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Legacy export row {row_index} contains invalid timestamp "
                "or joint values"
            ) from exc
        if not math.isfinite(timestamp) or not all(
            math.isfinite(value) for value in joints
        ):
            raise ValueError(
                f"Legacy export row {row_index} contains non-finite "
                "timestamp or joint values"
            )
        segment_type = str(source["segment_type"])
        if segment_type not in ALLOWED_SEGMENT_TYPES:
            raise ValueError(
                f"Legacy export row {row_index} has unexpected "
                f"segment_type {segment_type!r}"
            )

        if segment_type == "initial_hover":
            touch_type = "Air"
            original_status = "RECORDING_START"
        elif segment_type == "drawing_stroke":
            stroke_number = _required_legacy_stroke_number(
                source["stroke_index"],
                segment_type=segment_type,
                row_index=row_index,
            )
            drawing_stroke_ids.add(stroke_number)
            touch_type = "Pen"
            original_status = f"DRAWING_STROKE_{stroke_number}"
        elif segment_type in {"lift", "final_lift"}:
            stroke_number = _required_legacy_stroke_number(
                source["stroke_index"],
                segment_type=segment_type,
                row_index=row_index,
            )
            touch_type = "Air"
            original_status = f"END_STROKE_{stroke_number}"
        else:
            touch_type = "Air"
            original_status = "MOVING_FAST"

        source_timestamps.append(timestamp)
        source_joint_rows.append(joints)
        legacy_rows.append(
            {
                "Timestamp": source["time_seconds"],
                "TouchType": touch_type,
                **{
                    f"joint{joint}": source[f"q{joint}"]
                    for joint in range(1, JOINT_DIM + 1)
                },
                "OriginalStatus": original_status,
            }
        )

    timestamps = np.asarray(source_timestamps, dtype=np.float64)
    source_joints = np.asarray(source_joint_rows, dtype=np.float64)
    if np.any(np.diff(timestamps) < 0.0):
        raise ValueError(
            "Legacy SmartJoint timestamps must be monotonically nondecreasing"
        )
    if len(legacy_rows) != len(trajectory_rows):
        raise ValueError("Legacy SmartJoint row count differs from source")
    if tuple(legacy_rows[0].keys()) != LEGACY_SMARTJOINT_COLUMNS:
        raise ValueError("Legacy SmartJoint output columns are not exact")
    output_joints = np.asarray(
        [
            [
                float(row[f"joint{joint}"])
                for joint in range(1, JOINT_DIM + 1)
            ]
            for row in legacy_rows
        ],
        dtype=np.float64,
    )
    if not np.array_equal(output_joints, source_joints):
        raise ValueError(
            "Legacy SmartJoint joints differ numerically from source"
        )
    if not np.array_equal(output_joints[0], source_joints[0]):
        raise ValueError(
            "Legacy SmartJoint first joint vector differs from source"
        )
    if not np.array_equal(output_joints[-1], source_joints[-1]):
        raise ValueError(
            "Legacy SmartJoint last joint vector differs from source"
        )
    for row_index, (source, output) in enumerate(
        zip(trajectory_rows, legacy_rows)
    ):
        drawing = str(source["segment_type"]) == "drawing_stroke"
        expected_touch_type = "Pen" if drawing else "Air"
        if output["TouchType"] != expected_touch_type:
            raise ValueError(
                f"Legacy SmartJoint row {row_index} TouchType is invalid"
            )
        if any(
            value is None
            or (
                isinstance(value, (float, np.floating))
                and not math.isfinite(float(value))
            )
            for value in output.values()
        ):
            raise ValueError(
                f"Legacy SmartJoint row {row_index} contains an empty/NaN cell"
            )

    atomic_write_csv(output_csv_path, legacy_rows)
    return len(legacy_rows), sorted(drawing_stroke_ids)


def atomic_save_npz(path: Path, **arrays: Any) -> None:
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".npz",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


class TransitionBranchIncompatible(ValueError):
    """No endpoint-exact IK branch satisfies the complete transition gates."""


def _maximum_step_details(q: np.ndarray) -> Tuple[float, int, int]:
    if len(q) < 2:
        return 0.0, 0, 0
    absolute_steps = np.abs(np.diff(q, axis=0))
    flat_index = int(np.argmax(absolute_steps))
    step_index, joint_index = np.unravel_index(
        flat_index,
        absolute_steps.shape,
    )
    return (
        float(absolute_steps[step_index, joint_index]),
        int(step_index + 1),
        int(joint_index),
    )


def _interior_maximum_step(q: np.ndarray) -> float:
    """Report solver continuity before exact endpoint assignment."""
    if len(q) < 2:
        return 0.0
    return float(np.max(np.abs(np.diff(q, axis=0))))


def _evaluate_pinned_transition(
    segment: LocalSegment,
    *,
    start_q: np.ndarray,
    end_q: np.ndarray,
    robot: Any,
    joint_names: Sequence[str],
    lower: np.ndarray,
    upper: np.ndarray,
    target_rotation: np.ndarray,
    args: argparse.Namespace,
) -> Tuple[LocalSegment, Dict[str, Any]]:
    candidate_q = np.asarray(segment.q, dtype=np.float64).copy()
    maximum_before = _interior_maximum_step(candidate_q)
    candidate_q[0] = np.asarray(start_q, dtype=np.float64)
    candidate_q[-1] = np.asarray(end_q, dtype=np.float64)
    pinned = LocalSegment(
        segment_type=segment.segment_type,
        stroke_index=segment.stroke_index,
        local_times=np.asarray(segment.local_times, dtype=np.float64).copy(),
        q=candidate_q,
        desired_position=np.asarray(
            segment.desired_position,
            dtype=np.float64,
        ).copy(),
        planned_duration_seconds=segment.planned_duration_seconds,
    )
    fk_position, fk_rotation, _ = trajectory_full_transform_fk(
        robot,
        pinned.q,
        tuple(joint_names),
        DEFAULT_EE_LINK,
    )
    cartesian_error = np.linalg.norm(
        fk_position - pinned.desired_position,
        axis=1,
    )
    orientation_error = orientation_error_trajectory(
        target_rotation,
        fk_rotation,
    )
    z_error = np.abs(
        fk_position[:, 2] - pinned.desired_position[:, 2]
    )
    maximum_step, failing_sample, failing_joint = _maximum_step_details(
        pinned.q
    )
    joint_limit_violation = bool(
        np.any(
            pinned.q
            < lower[None, :] - HARD_JOINT_LIMIT_TOLERANCE_RAD
        )
        or np.any(
            pinned.q
            > upper[None, :] + HARD_JOINT_LIMIT_TOLERANCE_RAD
        )
    )
    endpoint_start_exact = bool(np.array_equal(pinned.q[0], start_q))
    endpoint_end_exact = bool(np.array_equal(pinned.q[-1], end_q))
    maximum_cartesian = float(np.max(cartesian_error))
    maximum_orientation = float(np.max(orientation_error))
    maximum_z = float(np.max(z_error))
    total_joint_travel = float(np.sum(np.abs(np.diff(pinned.q, axis=0))))
    gate_pass = bool(
        endpoint_start_exact
        and endpoint_end_exact
        and not joint_limit_violation
        and maximum_step <= args.maximum_joint_step_rad
        and maximum_cartesian <= args.maximum_cartesian_error_m
        and maximum_orientation <= args.maximum_orientation_error_rad
        and maximum_z <= args.maximum_z_tracking_error_m
    )
    return pinned, {
        "gate_pass": gate_pass,
        "maximum_joint_step_before_endpoint_pinning": maximum_before,
        "maximum_joint_step_after_endpoint_pinning": maximum_step,
        "maximum_cartesian_error_m": maximum_cartesian,
        "maximum_orientation_error_rad": maximum_orientation,
        "maximum_z_error_m": maximum_z,
        "total_joint_travel_rad": total_joint_travel,
        "joint_limit_violation": joint_limit_violation,
        "endpoint_start_exact": endpoint_start_exact,
        "endpoint_end_exact": endpoint_end_exact,
        "failing_sample_index": failing_sample,
        "failing_joint_index": failing_joint,
    }


def _candidate_order(evaluation: Mapping[str, Any]) -> Tuple[float, ...]:
    return (
        float(evaluation["maximum_joint_step_after_endpoint_pinning"]),
        float(evaluation["maximum_cartesian_error_m"]),
        float(evaluation["maximum_orientation_error_rad"]),
        float(evaluation["maximum_z_error_m"]),
        float(evaluation["total_joint_travel_rad"]),
    )


def transition_segment(
    *,
    segment_type: str,
    stroke_index: int,
    start_position: np.ndarray,
    end_position: np.ndarray,
    start_q: np.ndarray,
    end_q: np.ndarray,
    duration: float,
    robot: Any,
    joint_names: Sequence[str],
    bounds: Any,
    lower: np.ndarray,
    upper: np.ndarray,
    target_rotation: np.ndarray,
    args: argparse.Namespace,
    seed: int,
) -> LocalSegment:
    pending = PENDING_COUPLED_DESCENTS.get(stroke_index)
    if (
        segment_type == "descent"
        and pending is not None
        and np.array_equal(pending[0], start_q)
        and np.array_equal(pending[1], end_q)
    ):
        del PENDING_COUPLED_DESCENTS[stroke_index]
        TRANSITION_DIAGNOSTICS.append(dict(pending[3]))
        return pending[2]
    candidates: List[
        Tuple[str, LocalSegment, Dict[str, Any]]
    ] = []
    failures: Dict[str, str] = {}
    forward_evaluation: Dict[str, Any] = {
        "gate_pass": False,
        "maximum_joint_step_after_endpoint_pinning": math.inf,
        "failing_sample_index": -1,
        "failing_joint_index": -1,
    }
    reverse_evaluation: Dict[str, Any] = dict(forward_evaluation)
    try:
        forward_raw = _transition_segment_one_direction(
            segment_type=segment_type,
            stroke_index=stroke_index,
            start_position=start_position,
            end_position=end_position,
            start_q=start_q,
            end_q=end_q,
            duration=duration,
            robot=robot,
            joint_names=joint_names,
            bounds=bounds,
            lower=lower,
            upper=upper,
            target_rotation=target_rotation,
            args=args,
            seed=seed,
        )
        forward, forward_evaluation = _evaluate_pinned_transition(
            forward_raw,
            start_q=start_q,
            end_q=end_q,
            robot=robot,
            joint_names=joint_names,
            lower=lower,
            upper=upper,
            target_rotation=target_rotation,
            args=args,
        )
        if bool(forward_evaluation["gate_pass"]):
            candidates.append(("forward", forward, forward_evaluation))
    except Exception as exc:
        failures["forward"] = str(exc)

    try:
        reverse_raw = _transition_segment_one_direction(
            segment_type=segment_type,
            stroke_index=stroke_index,
            start_position=end_position,
            end_position=start_position,
            start_q=end_q,
            end_q=start_q,
            duration=duration,
            robot=robot,
            joint_names=joint_names,
            bounds=bounds,
            lower=lower,
            upper=upper,
            target_rotation=target_rotation,
            args=args,
            seed=seed + 1000003,
        )
        reverse_execution = LocalSegment(
            segment_type=segment_type,
            stroke_index=stroke_index,
            local_times=(
                duration
                - np.asarray(reverse_raw.local_times, dtype=np.float64)[::-1]
            ),
            q=np.asarray(reverse_raw.q, dtype=np.float64)[::-1].copy(),
            desired_position=np.asarray(
                reverse_raw.desired_position,
                dtype=np.float64,
            )[::-1].copy(),
            planned_duration_seconds=duration,
        )
        reverse_execution.local_times[0] = 0.0
        reverse_execution.local_times[-1] = duration
        reverse, reverse_evaluation = _evaluate_pinned_transition(
            reverse_execution,
            start_q=start_q,
            end_q=end_q,
            robot=robot,
            joint_names=joint_names,
            lower=lower,
            upper=upper,
            target_rotation=target_rotation,
            args=args,
        )
        if bool(reverse_evaluation["gate_pass"]):
            candidates.append(("reverse", reverse, reverse_evaluation))
    except Exception as exc:
        failures["reverse"] = str(exc)

    if not candidates:
        forward_step = float(
            forward_evaluation[
                "maximum_joint_step_after_endpoint_pinning"
            ]
        )
        reverse_step = float(
            reverse_evaluation[
                "maximum_joint_step_after_endpoint_pinning"
            ]
        )
        failing = (
            forward_evaluation
            if forward_step <= reverse_step
            else reverse_evaluation
        )
        raise TransitionBranchIncompatible(
            "transition_endpoint_ik_branch_incompatible "
            + json.dumps(
                {
                    "segment_type": segment_type,
                    "stroke_index": stroke_index,
                    "start_q": np.asarray(start_q).tolist(),
                    "end_q": np.asarray(end_q).tolist(),
                    "forward_maximum_joint_step_rad": forward_step,
                    "reverse_maximum_joint_step_rad": reverse_step,
                    "failing_sample_index": int(
                        failing.get("failing_sample_index", -1)
                    ),
                    "failing_joint_index": int(
                        failing.get("failing_joint_index", -1)
                    ),
                    "solver_failures": failures,
                },
                sort_keys=True,
            )
        )

    direction, selected, selected_evaluation = min(
        candidates,
        key=lambda item: _candidate_order(item[2]),
    )
    TRANSITION_DIAGNOSTICS.append(
        {
            "segment_type": segment_type,
            "stroke_index": stroke_index,
            "selected_direction": direction,
            "maximum_joint_step_before_endpoint_pinning": float(
                selected_evaluation[
                    "maximum_joint_step_before_endpoint_pinning"
                ]
            ),
            "maximum_joint_step_after_endpoint_pinning": float(
                selected_evaluation[
                    "maximum_joint_step_after_endpoint_pinning"
                ]
            ),
            "forward_gate_pass": bool(
                forward_evaluation.get("gate_pass", False)
            ),
            "reverse_gate_pass": bool(
                reverse_evaluation.get("gate_pass", False)
            ),
            "selected_gate_pass": bool(
                selected_evaluation["gate_pass"]
            ),
            "endpoint_start_exact": bool(
                selected_evaluation["endpoint_start_exact"]
            ),
            "endpoint_end_exact": bool(
                selected_evaluation["endpoint_end_exact"]
            ),
        }
    )
    return selected


def select_coupled_hover_pair(
    *,
    stroke_index: int,
    current_hover_position: np.ndarray,
    next_hover_position: np.ndarray,
    next_drawing_position: np.ndarray,
    current_hover_q: np.ndarray,
    next_drawing_q: np.ndarray,
    independently_solved_hover_q: np.ndarray,
    robot: Any,
    joint_names: Sequence[str],
    bounds: Any,
    lower: np.ndarray,
    upper: np.ndarray,
    target_rotation: np.ndarray,
    args: argparse.Namespace,
    seed: int,
) -> Tuple[np.ndarray, LocalSegment]:
    """Select a next-hover IK branch only after travel and descent pass."""
    hover_candidates: List[Tuple[str, np.ndarray]] = []

    def add_hover_candidate(label: str, candidate: np.ndarray) -> None:
        value = np.asarray(candidate, dtype=np.float64)
        if value.shape != (JOINT_DIM,) or not np.all(np.isfinite(value)):
            return
        if any(np.array_equal(value, existing) for _, existing in hover_candidates):
            return
        hover_candidates.append((label, value.copy()))

    try:
        add_hover_candidate(
            "previous_lifted_hover_seed",
            solve_hover(
                position=next_hover_position,
                seed_q=current_hover_q,
                robot=robot,
                joint_names=joint_names,
                bounds=bounds,
                lower=lower,
                upper=upper,
                target_rotation=target_rotation,
                args=args,
                seed=seed + 2000003,
            ),
        )
    except Exception:
        pass
    try:
        add_hover_candidate(
            "next_drawing_start_seed",
            independently_solved_hover_q,
        )
    except Exception:
        pass

    preliminary_seed = independently_solved_hover_q
    try:
        preliminary = _transition_segment_one_direction(
            segment_type="hover_travel",
            stroke_index=stroke_index,
            start_position=current_hover_position,
            end_position=next_hover_position,
            start_q=current_hover_q,
            end_q=independently_solved_hover_q,
            duration=args.travel_duration,
            robot=robot,
            joint_names=joint_names,
            bounds=bounds,
            lower=lower,
            upper=upper,
            target_rotation=target_rotation,
            args=args,
            seed=seed + 3000017,
        )
        preliminary_seed = np.asarray(preliminary.q[-1]).copy()
    except Exception:
        pass
    try:
        add_hover_candidate(
            "preliminary_hover_travel_final_state_seed",
            solve_hover(
                position=next_hover_position,
                seed_q=preliminary_seed,
                robot=robot,
                joint_names=joint_names,
                bounds=bounds,
                lower=lower,
                upper=upper,
                target_rotation=target_rotation,
                args=args,
                seed=seed + 4000037,
            ),
        )
    except Exception:
        pass

    valid_pairs: List[
        Tuple[
            Tuple[float, ...],
            np.ndarray,
            LocalSegment,
            LocalSegment,
            List[Dict[str, Any]],
        ]
    ] = []
    trial_failures: List[Dict[str, Any]] = []
    for candidate_number, (candidate_label, hover_q) in enumerate(
        hover_candidates
    ):
        diagnostic_start = len(TRANSITION_DIAGNOSTICS)
        try:
            travel = transition_segment(
                segment_type="hover_travel",
                stroke_index=stroke_index,
                start_position=current_hover_position,
                end_position=next_hover_position,
                start_q=current_hover_q,
                end_q=hover_q,
                duration=args.travel_duration,
                robot=robot,
                joint_names=joint_names,
                bounds=bounds,
                lower=lower,
                upper=upper,
                target_rotation=target_rotation,
                args=args,
                seed=seed + candidate_number * 101,
            )
            descent = transition_segment(
                segment_type="descent",
                stroke_index=stroke_index,
                start_position=next_hover_position,
                end_position=next_drawing_position,
                start_q=hover_q,
                end_q=next_drawing_q,
                duration=args.descent_duration,
                robot=robot,
                joint_names=joint_names,
                bounds=bounds,
                lower=lower,
                upper=upper,
                target_rotation=target_rotation,
                args=args,
                seed=seed + candidate_number * 101 + 1,
            )
            pair_diagnostics = [
                dict(item)
                for item in TRANSITION_DIAGNOSTICS[diagnostic_start:]
            ]
            if len(pair_diagnostics) != 2:
                raise TransitionBranchIncompatible(
                    "transition_endpoint_ik_branch_incompatible "
                    "coupled pair did not produce two diagnostics"
                )
            maximum_step = max(
                float(
                    item[
                        "maximum_joint_step_after_endpoint_pinning"
                    ]
                )
                for item in pair_diagnostics
            )
            travel_evaluation = _evaluate_pinned_transition(
                travel,
                start_q=current_hover_q,
                end_q=hover_q,
                robot=robot,
                joint_names=joint_names,
                lower=lower,
                upper=upper,
                target_rotation=target_rotation,
                args=args,
            )[1]
            descent_evaluation = _evaluate_pinned_transition(
                descent,
                start_q=hover_q,
                end_q=next_drawing_q,
                robot=robot,
                joint_names=joint_names,
                lower=lower,
                upper=upper,
                target_rotation=target_rotation,
                args=args,
            )[1]
            score = (
                maximum_step,
                max(
                    float(travel_evaluation["maximum_cartesian_error_m"]),
                    float(descent_evaluation["maximum_cartesian_error_m"]),
                ),
                max(
                    float(
                        travel_evaluation[
                            "maximum_orientation_error_rad"
                        ]
                    ),
                    float(
                        descent_evaluation[
                            "maximum_orientation_error_rad"
                        ]
                    ),
                ),
                max(
                    float(travel_evaluation["maximum_z_error_m"]),
                    float(descent_evaluation["maximum_z_error_m"]),
                ),
                float(travel_evaluation["total_joint_travel_rad"])
                + float(descent_evaluation["total_joint_travel_rad"]),
            )
            for item in pair_diagnostics:
                item["coupled_hover_candidate"] = candidate_label
            valid_pairs.append(
                (
                    score,
                    hover_q.copy(),
                    travel,
                    descent,
                    pair_diagnostics,
                )
            )
        except Exception as exc:
            trial_failures.append(
                {
                    "candidate": candidate_label,
                    "error": str(exc),
                }
            )
        finally:
            del TRANSITION_DIAGNOSTICS[diagnostic_start:]

    if not valid_pairs:
        maximum, failing_sample, failing_joint = _maximum_step_details(
            np.vstack((current_hover_q, independently_solved_hover_q))
        )
        raise TransitionBranchIncompatible(
            "transition_endpoint_ik_branch_incompatible "
            + json.dumps(
                {
                    "segment_type": "hover_travel",
                    "stroke_index": stroke_index,
                    "start_q": np.asarray(current_hover_q).tolist(),
                    "end_q": np.asarray(
                        independently_solved_hover_q
                    ).tolist(),
                    "forward_maximum_joint_step_rad": maximum,
                    "reverse_maximum_joint_step_rad": maximum,
                    "failing_sample_index": failing_sample,
                    "failing_joint_index": failing_joint,
                    "coupled_candidate_failures": trial_failures,
                },
                sort_keys=True,
            )
        )

    (
        _,
        selected_hover_q,
        selected_travel,
        selected_descent,
        selected_diagnostics,
    ) = min(
        valid_pairs,
        key=lambda item: item[0],
    )
    TRANSITION_DIAGNOSTICS.extend(selected_diagnostics[:1])
    PENDING_COUPLED_DESCENTS[stroke_index] = (
        selected_hover_q.copy(),
        np.asarray(next_drawing_q, dtype=np.float64).copy(),
        selected_descent,
        dict(selected_diagnostics[1]),
    )
    return selected_hover_q, selected_travel


def dynamics(
    q: np.ndarray,
    timestamps: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    return velocity, acceleration, jerk


def append_segments(
    segments: Sequence[LocalSegment],
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    List[Dict[str, Any]],
]:
    timestamps: List[np.ndarray] = []
    q_parts: List[np.ndarray] = []
    desired_parts: List[np.ndarray] = []
    segment_indices: List[np.ndarray] = []
    stroke_indices: List[np.ndarray] = []
    type_parts: List[np.ndarray] = []
    manifest: List[Dict[str, Any]] = []
    sample_cursor = 0
    time_cursor = 0.0
    previous_q: np.ndarray | None = None
    for segment_index, segment in enumerate(segments):
        if segment.segment_type not in ALLOWED_SEGMENT_TYPES:
            raise ValueError(f"Unsupported segment type: {segment.segment_type}")
        if previous_q is None:
            keep = slice(None)
            global_times = segment.local_times.copy()
        else:
            if not np.allclose(
                segment.q[0],
                previous_q,
                rtol=0.0,
                atol=ENDPOINT_ATOL,
            ):
                raise ValueError(
                    f"Boundary before {segment.segment_type} is discontinuous"
                )
            if segment.segment_type == "drawing_stroke":
                if (
                    not timestamps
                    or len(timestamps[-1]) <= 1
                    or not manifest
                ):
                    raise ValueError(
                        "Cannot transfer drawing boundary ownership from an "
                        "empty or single-sample preceding segment"
                    )
                boundary_time = time_cursor
                timestamps[-1] = timestamps[-1][:-1]
                q_parts[-1] = q_parts[-1][:-1]
                desired_parts[-1] = desired_parts[-1][:-1]
                segment_indices[-1] = segment_indices[-1][:-1]
                stroke_indices[-1] = stroke_indices[-1][:-1]
                type_parts[-1] = type_parts[-1][:-1]
                previous_manifest = manifest[-1]
                previous_manifest["end_sample"] = (
                    int(previous_manifest["end_sample"]) - 1
                )
                previous_manifest["sample_count"] = (
                    int(previous_manifest["sample_count"]) - 1
                )
                previous_manifest["end_time_seconds"] = float(
                    timestamps[-1][-1]
                )
                previous_manifest["duration_seconds"] = float(
                    timestamps[-1][-1] - timestamps[-1][0]
                )
                sample_cursor -= 1
                keep = slice(None)
                global_times = boundary_time + segment.local_times
            else:
                keep = slice(1, None)
                global_times = time_cursor + segment.local_times[1:]
        q_kept = segment.q[keep]
        desired_kept = segment.desired_position[keep]
        if len(q_kept) == 0:
            raise ValueError(f"Segment became empty: {segment.segment_type}")
        if timestamps and global_times[0] <= timestamps[-1][-1]:
            raise ValueError("Segment concatenation produced duplicate time")
        start = sample_cursor
        end = start + len(q_kept) - 1
        timestamps.append(global_times)
        q_parts.append(q_kept)
        desired_parts.append(desired_kept)
        segment_indices.append(
            np.full(len(q_kept), segment_index, dtype=np.int64)
        )
        stroke_indices.append(
            np.full(len(q_kept), segment.stroke_index, dtype=np.int64)
        )
        type_parts.append(
            np.full(
                len(q_kept),
                segment.segment_type,
                dtype=f"<U{max(len(name) for name in ALLOWED_SEGMENT_TYPES)}",
            )
        )
        manifest.append(
            {
                "segment_index": segment_index,
                "segment_type": segment.segment_type,
                "stroke_index": segment.stroke_index,
                "start_sample": start,
                "end_sample": end,
                "start_time_seconds": float(global_times[0]),
                "end_time_seconds": float(global_times[-1]),
                "sample_count": len(q_kept),
                "duration_seconds": float(global_times[-1] - global_times[0]),
                "planned_duration_seconds": float(
                    segment.planned_duration_seconds
                ),
                "duplicated_first_boundary_removed": int(
                    previous_q is not None
                    and segment.segment_type != "drawing_stroke"
                ),
            }
        )
        sample_cursor = end + 1
        time_cursor = float(global_times[-1])
        previous_q = segment.q[-1].copy()
    return (
        np.concatenate(timestamps),
        np.concatenate(q_parts),
        np.concatenate(desired_parts),
        np.concatenate(segment_indices),
        np.concatenate(type_parts),
        np.concatenate(stroke_indices),
        manifest,
    )


def find_exact_subsequence(haystack: np.ndarray, needle: np.ndarray) -> List[int]:
    starts: List[int] = []
    for start in range(len(haystack) - len(needle) + 1):
        if np.array_equal(haystack[start : start + len(needle)], needle):
            starts.append(start)
    return starts


def validate_drawing_segment_ownership(
    strokes: Sequence[Stroke],
    q: np.ndarray,
    segment_types: np.ndarray,
    stroke_indices: np.ndarray,
    manifest: Sequence[Mapping[str, Any]],
) -> None:
    """Verify every authoritative drawing segment owns all source waypoints."""
    for stroke_index, stroke in enumerate(strokes):
        drawing_rows = [
            row
            for row in manifest
            if row["segment_type"] == "drawing_stroke"
            and int(row["stroke_index"]) == stroke_index
        ]
        if len(drawing_rows) != 1:
            raise ValueError(
                f"Stroke {stroke_index} must have exactly one drawing manifest "
                f"row, found {len(drawing_rows)}"
            )
        row = drawing_rows[0]
        start = int(row["start_sample"])
        end = int(row["end_sample"])
        sample_count = int(row["sample_count"])
        if end - start + 1 != TRAJECTORY_LENGTH:
            raise ValueError(
                f"Stroke {stroke_index} drawing range has "
                f"{end - start + 1} samples, expected {TRAJECTORY_LENGTH}"
            )
        if sample_count != TRAJECTORY_LENGTH:
            raise ValueError(
                f"Stroke {stroke_index} drawing manifest sample_count is "
                f"{sample_count}, expected {TRAJECTORY_LENGTH}"
            )
        drawing_q = q[start : end + 1]
        if not np.array_equal(drawing_q, stroke.final_q):
            raise ValueError(
                f"Stroke {stroke_index} drawing q does not exactly match source"
            )
        if not np.array_equal(drawing_q[0], stroke.final_q[0]):
            raise ValueError(
                f"Stroke {stroke_index} first drawing sample differs from source"
            )
        if not np.array_equal(drawing_q[-1], stroke.final_q[-1]):
            raise ValueError(
                f"Stroke {stroke_index} final drawing sample differs from source"
            )
        if not np.all(segment_types[start : end + 1] == "drawing_stroke"):
            raise ValueError(
                f"Stroke {stroke_index} drawing segment types are inconsistent"
            )
        if not np.all(stroke_indices[start : end + 1] == stroke_index):
            raise ValueError(
                f"Stroke {stroke_index} drawing stroke indices are inconsistent"
            )


def save_plots(
    output_dir: Path,
    timestamps: np.ndarray,
    q: np.ndarray,
    velocity: np.ndarray,
    acceleration: np.ndarray,
    jerk: np.ndarray,
    desired: np.ndarray,
    fk_position: np.ndarray,
    orientation_error: np.ndarray,
    cartesian_error: np.ndarray,
    segment_types: np.ndarray,
) -> None:
    plot_dir = output_dir / "plots"
    for values, filename, ylabel in (
        (q, "joint_positions.png", "joint position (rad)"),
        (velocity, "joint_velocities.png", "joint velocity (rad/s)"),
        (
            acceleration,
            "joint_accelerations.png",
            "joint acceleration (rad/s²)",
        ),
        (jerk, "joint_jerks.png", "joint jerk (rad/s³)"),
    ):
        figure, axis = plt.subplots(figsize=(11, 5))
        for joint in range(JOINT_DIM):
            axis.plot(timestamps, values[:, joint], label=f"q{joint + 1}")
        axis.set(xlabel="time (s)", ylabel=ylabel)
        axis.legend(ncol=3)
        figure.tight_layout()
        figure.savefig(str(plot_dir / filename), dpi=150)
        plt.close(figure)
    figure, axis = plt.subplots(figsize=(8, 7))
    drawing = segment_types == "drawing_stroke"
    axis.plot(desired[:, 0], desired[:, 1], color="0.75", label="desired")
    axis.scatter(
        fk_position[~drawing, 0],
        fk_position[~drawing, 1],
        s=5,
        label="transition",
    )
    axis.scatter(
        fk_position[drawing, 0],
        fk_position[drawing, 1],
        s=5,
        label="drawing",
    )
    axis.set(xlabel="x (m)", ylabel="y (m)")
    axis.legend()
    figure.tight_layout()
    figure.savefig(str(plot_dir / "cartesian_xy_path.png"), dpi=150)
    plt.close(figure)
    for values, filename, ylabel in (
        (fk_position[:, 2], "cartesian_z_vs_time.png", "z (m)"),
        (
            orientation_error,
            "orientation_error.png",
            "orientation error (rad)",
        ),
        (cartesian_error, "cartesian_error.png", "position error (m)"),
    ):
        figure, axis = plt.subplots(figsize=(11, 4))
        axis.plot(timestamps, values)
        axis.set(xlabel="time (s)", ylabel=ylabel)
        figure.tight_layout()
        figure.savefig(str(plot_dir / filename), dpi=150)
        plt.close(figure)
    figure, axis = plt.subplots(figsize=(12, 3))
    names: List[str] = list(ALLOWED_SEGMENT_TYPES)
    encoded = np.asarray([names.index(str(value)) for value in segment_types])
    axis.step(timestamps, encoded, where="post")
    axis.set(
        xlabel="time (s)",
        yticks=np.arange(len(names)),
        yticklabels=names,
    )
    figure.tight_layout()
    figure.savefig(str(plot_dir / "segment_timeline.png"), dpi=150)
    plt.close(figure)


def main() -> int:
    args = parse_args()
    TRANSITION_DIAGNOSTICS.clear()
    PENDING_COUPLED_DESCENTS.clear()
    try:
        if len(args.stroke_output_dirs) < 2:
            raise ValueError(
                "--stroke_output_dirs requires two or more directories"
            )
        strokes = [load_stroke(path) for path in args.stroke_output_dirs]
        require_compatible(strokes)
        validate_args(args, strokes)
        prepare_output_dir(args.output_dir, args.overwrite)
        first = strokes[0]
        joint_names = tuple(DEFAULT_JOINT_NAMES)
        if first.joint_order != joint_names:
            raise ValueError(
                f"Stroke joint order {first.joint_order} is not {joint_names}"
            )
        robot = load_robot(first.urdf_path)
        bounds = get_joint_bounds(robot, joint_names, -np.pi, np.pi)
        lower = np.asarray([value[0] for value in bounds], dtype=np.float64)
        upper = np.asarray([value[1] for value in bounds], dtype=np.float64)
        drawing_z = first.target_z
        hover_z = drawing_z + float(args.hover_offset_z)
        target_rotation = first.target_rotation_matrix
        segments: List[LocalSegment] = []
        seed_counter = 10_000

        first_hover_position = np.asarray(
            [
                first.desired_path[0, 0],
                first.desired_path[0, 1],
                hover_z,
            ],
            dtype=np.float64,
        )
        first_hover_q = solve_hover(
            robot=robot,
            position=first_hover_position,
            seed_q=first.final_q[0],
            joint_names=joint_names,
            bounds=bounds,
            lower=lower,
            upper=upper,
            target_rotation=target_rotation,
            args=args,
            seed=seed_counter,
        )
        seed_counter += 1
        segments.append(
            LocalSegment(
                segment_type="initial_hover",
                stroke_index=0,
                local_times=np.asarray([0.0], dtype=np.float64),
                q=first_hover_q[None, :],
                desired_position=first_hover_position[None, :],
                planned_duration_seconds=0.0,
            )
        )
        segments.append(
            transition_segment(
                segment_type="initial_descent",
                stroke_index=0,
                start_position=first_hover_position,
                end_position=first.desired_path[0],
                start_q=first_hover_q,
                end_q=first.final_q[0],
                duration=args.initial_descent_duration,
                robot=robot,
                joint_names=joint_names,
                bounds=bounds,
                lower=lower,
                upper=upper,
                target_rotation=target_rotation,
                args=args,
                seed=seed_counter,
            )
        )
        seed_counter += 1

        for stroke_index, stroke in enumerate(strokes):
            segments.append(drawing_segment(stroke, stroke_index))
            end_drawing_position = stroke.desired_path[-1].copy()
            end_hover_position = end_drawing_position.copy()
            end_hover_position[2] = hover_z
            end_hover_q = solve_hover(
                robot=robot,
                position=end_hover_position,
                seed_q=stroke.final_q[-1],
                joint_names=joint_names,
                bounds=bounds,
                lower=lower,
                upper=upper,
                target_rotation=target_rotation,
                args=args,
                seed=seed_counter,
            )
            seed_counter += 1
            is_final = stroke_index == len(strokes) - 1
            segments.append(
                transition_segment(
                    segment_type="final_lift" if is_final else "lift",
                    stroke_index=stroke_index,
                    start_position=end_drawing_position,
                    end_position=end_hover_position,
                    start_q=stroke.final_q[-1],
                    end_q=end_hover_q,
                    duration=(
                        args.final_lift_duration
                        if is_final
                        else args.lift_duration
                    ),
                    robot=robot,
                    joint_names=joint_names,
                    bounds=bounds,
                    lower=lower,
                    upper=upper,
                    target_rotation=target_rotation,
                    args=args,
                    seed=seed_counter,
                )
            )
            seed_counter += 1
            if is_final:
                continue
            next_stroke = strokes[stroke_index + 1]
            next_hover_position = next_stroke.desired_path[0].copy()
            next_hover_position[2] = hover_z
            next_hover_q = solve_hover(
                robot=robot,
                position=next_hover_position,
                seed_q=next_stroke.final_q[0],
                joint_names=joint_names,
                bounds=bounds,
                lower=lower,
                upper=upper,
                target_rotation=target_rotation,
                args=args,
                seed=seed_counter,
            )
            seed_counter += 1
            next_hover_q, coupled_travel = select_coupled_hover_pair(
                stroke_index=stroke_index + 1,
                current_hover_position=end_hover_position,
                next_hover_position=next_hover_position,
                next_drawing_position=next_stroke.desired_path[0],
                current_hover_q=end_hover_q,
                next_drawing_q=next_stroke.final_q[0],
                independently_solved_hover_q=next_hover_q,
                robot=robot,
                joint_names=joint_names,
                bounds=bounds,
                lower=lower,
                upper=upper,
                target_rotation=target_rotation,
                args=args,
                seed=seed_counter,
            )
            segments.append(coupled_travel)
            seed_counter += 1
            segments.append(
                transition_segment(
                    segment_type="descent",
                    stroke_index=stroke_index + 1,
                    start_position=next_hover_position,
                    end_position=next_stroke.desired_path[0],
                    start_q=next_hover_q,
                    end_q=next_stroke.final_q[0],
                    duration=args.descent_duration,
                    robot=robot,
                    joint_names=joint_names,
                    bounds=bounds,
                    lower=lower,
                    upper=upper,
                    target_rotation=target_rotation,
                    args=args,
                    seed=seed_counter,
                )
            )
            seed_counter += 1

        (
            timestamps,
            q,
            desired_position,
            segment_index,
            segment_type,
            stroke_index_per_sample,
            manifest,
        ) = append_segments(segments)
        if not np.all(np.diff(timestamps) > 0.0):
            raise ValueError("Combined timestamps are not strictly increasing")
        validate_drawing_segment_ownership(
            strokes,
            q,
            segment_type,
            stroke_index_per_sample,
            manifest,
        )
        fk_position, fk_rotation, fk_quaternion = trajectory_full_transform_fk(
            robot,
            q,
            joint_names,
            DEFAULT_EE_LINK,
        )
        cartesian_error = np.linalg.norm(
            fk_position - desired_position,
            axis=1,
        )
        orientation_error = orientation_error_trajectory(
            target_rotation,
            fk_rotation,
        )
        z_tracking_error = np.abs(
            fk_position[:, 2] - desired_position[:, 2]
        )
        velocity, acceleration, jerk = dynamics(q, timestamps)
        limits = check_joint_limits(
            q,
            lower,
            upper,
            joint_names,
            tolerance=HARD_JOINT_LIMIT_TOLERANCE_RAD,
        )
        maximum_joint_step = float(
            np.max(np.abs(np.diff(q, axis=0)))
        )
        preservation_rows: List[Dict[str, Any]] = []
        preservation_pass = True
        previous_start = -1
        for index, stroke in enumerate(strokes):
            starts = find_exact_subsequence(q, stroke.final_q)
            ordered = [start for start in starts if start > previous_start]
            passed = len(starts) == 1 and len(ordered) == 1
            start = ordered[0] if ordered else -1
            preservation_pass = preservation_pass and passed
            previous_start = start
            preservation_rows.append(
                {
                    "stroke_index": index,
                    "source_directory": str(stroke.output_dir),
                    "combined_start_sample": start,
                    "combined_end_sample": (
                        -1 if start < 0 else start + TRAJECTORY_LENGTH - 1
                    ),
                    "exactly_once_and_ordered": passed,
                }
            )
        rejection_reasons: List[str] = []
        if not np.all(np.isfinite(q)):
            rejection_reasons.append("nonfinite_q")
        if not (
            np.all(np.isfinite(fk_position))
            and np.all(np.isfinite(fk_rotation))
            and np.all(np.isfinite(fk_quaternion))
        ):
            rejection_reasons.append("nonfinite_full_transform_fk")
        if int(limits["hard_joint_limit_violation_count"]) != 0:
            rejection_reasons.append("hard_joint_limit_violation")
        if maximum_joint_step > args.maximum_joint_step_rad:
            rejection_reasons.append("maximum_joint_step_gate")
        if float(np.max(cartesian_error)) > args.maximum_cartesian_error_m:
            rejection_reasons.append("maximum_cartesian_error_gate")
        if float(np.max(orientation_error)) > args.maximum_orientation_error_rad:
            rejection_reasons.append("maximum_orientation_error_gate")
        if float(np.max(z_tracking_error)) > args.maximum_z_tracking_error_m:
            rejection_reasons.append("maximum_z_tracking_error_gate")
        if not preservation_pass:
            rejection_reasons.append("approved_drawing_not_preserved")
        if manifest[0]["segment_type"] != "initial_hover":
            rejection_reasons.append("first_segment_not_initial_hover")
        if manifest[-1]["segment_type"] != "final_lift":
            rejection_reasons.append("last_segment_not_final_lift")
        for expected, row in enumerate(manifest):
            if (
                int(row["segment_index"]) != expected
                or int(row["start_sample"])
                != (0 if expected == 0 else int(manifest[expected - 1]["end_sample"]) + 1)
            ):
                rejection_reasons.append("invalid_segment_ranges")
                break
        accepted = not rejection_reasons
        verdict = ACCEPTED_VERDICT if accepted else REJECTED_VERDICT
        metrics: Dict[str, Any] = {
            "accepted": accepted,
            "verdict": verdict,
            "rejection_reasons": rejection_reasons,
            "sample_count": len(q),
            "segment_count": len(manifest),
            "stroke_count": len(strokes),
            "drawing_z": drawing_z,
            "hover_z": hover_z,
            "hover_offset_z": args.hover_offset_z,
            "transition_sample_period": args.transition_sample_period,
            "maximum_consecutive_absolute_joint_step_rad": maximum_joint_step,
            "maximum_joint_step_rad": args.maximum_joint_step_rad,
            "maximum_observed_cartesian_error_m": float(
                np.max(cartesian_error)
            ),
            "maximum_cartesian_error_m": args.maximum_cartesian_error_m,
            "maximum_observed_orientation_error_rad": float(
                np.max(orientation_error)
            ),
            "maximum_orientation_error_rad": (
                args.maximum_orientation_error_rad
            ),
            "maximum_observed_z_tracking_error_m": float(
                np.max(z_tracking_error)
            ),
            "maximum_z_tracking_error_m": (
                args.maximum_z_tracking_error_m
            ),
            "hard_joint_limit_violation_count": int(
                limits["hard_joint_limit_violation_count"]
            ),
            "target_rpy": first.target_rpy,
            "target_quaternion": first.target_quaternion,
            "target_rotation_matrix": target_rotation,
            "urdf_path": str(first.urdf_path),
            "urdf_sha256": first.urdf_sha256,
            "joint_order": list(joint_names),
            "fk_frame": DEFAULT_EE_LINK,
            "source_stroke_directories": [
                str(stroke.output_dir) for stroke in strokes
            ],
            "source_stroke_input_sha256": [
                stroke.input_sha256 for stroke in strokes
            ],
            "source_stroke_deployment_path_id": [
                stroke.deployment_path_id for stroke in strokes
            ],
            "drawing_preservation": preservation_rows,
            "transition_diagnostics": TRANSITION_DIAGNOSTICS,
            "operator_start_requirement": (
                "The operator must move the robot to q[0], the solved initial "
                "hover state, before starting. No motion from an arbitrary "
                "physical robot state is planned."
            ),
        }
        trajectory_rows: List[Dict[str, Any]] = []
        approved_rows: List[Dict[str, Any]] = []
        for sample in range(len(q)):
            row: Dict[str, Any] = {
                "sample_index": sample,
                "time_seconds": float(timestamps[sample]),
                **{
                    f"q{joint + 1}": float(q[sample, joint])
                    for joint in range(JOINT_DIM)
                },
                "desired_x": float(desired_position[sample, 0]),
                "desired_y": float(desired_position[sample, 1]),
                "desired_z": float(desired_position[sample, 2]),
                "fk_x": float(fk_position[sample, 0]),
                "fk_y": float(fk_position[sample, 1]),
                "fk_z": float(fk_position[sample, 2]),
                "cartesian_error_m": float(cartesian_error[sample]),
                "orientation_error_rad": float(orientation_error[sample]),
                "z_tracking_error_m": float(z_tracking_error[sample]),
                "segment_index": int(segment_index[sample]),
                "segment_type": str(segment_type[sample]),
                "stroke_index": int(stroke_index_per_sample[sample]),
            }
            trajectory_rows.append(row)
            approved_rows.append(
                {
                    "time_seconds": float(timestamps[sample]),
                    **{
                        f"q{joint + 1}": float(q[sample, joint])
                        for joint in range(JOINT_DIM)
                    },
                }
            )
        atomic_write_csv(
            args.output_dir / "multistroke_execution_trajectory.csv",
            trajectory_rows,
        )
        legacy_csv_path = args.output_dir / LEGACY_SMARTJOINT_FILENAME
        legacy_row_count = 0
        legacy_drawing_stroke_ids: List[int] = []
        if accepted:
            (
                legacy_row_count,
                legacy_drawing_stroke_ids,
            ) = export_legacy_smartjoint_csv(
                trajectory_rows,
                legacy_csv_path,
            )
            print(f"Legacy SmartJoint CSV written: {legacy_csv_path}")
            print(f"Legacy SmartJoint rows: {legacy_row_count}")
            print(
                "Legacy drawing stroke IDs: "
                f"{legacy_drawing_stroke_ids}"
            )
        metrics["legacy_smartjoint_csv"] = (
            str(legacy_csv_path) if accepted else None
        )
        metrics["legacy_smartjoint_row_count"] = legacy_row_count
        metrics["legacy_drawing_stroke_ids"] = (
            legacy_drawing_stroke_ids
        )
        atomic_write_csv(
            args.output_dir / "multistroke_segment_manifest.csv",
            manifest,
        )
        atomic_save_npz(
            args.output_dir / "multistroke_execution_full.npz",
            timestamps=timestamps,
            q=q,
            fk_position=fk_position,
            fk_rotation_matrix=fk_rotation,
            fk_quaternion=fk_quaternion,
            target_rotation_matrix=target_rotation,
            target_quaternion=first.target_quaternion,
            target_rpy=first.target_rpy,
            drawing_z=drawing_z,
            hover_z=hover_z,
            desired_position=desired_position,
            cartesian_error_m=cartesian_error,
            orientation_error_rad=orientation_error,
            z_tracking_error_m=z_tracking_error,
            segment_index_per_sample=segment_index,
            segment_type_per_sample=segment_type,
            stroke_index_per_sample=stroke_index_per_sample,
            joint_velocity=velocity,
            joint_acceleration=acceleration,
            joint_jerk=jerk,
            urdf_path=str(first.urdf_path),
            urdf_sha256=first.urdf_sha256,
            joint_order=np.asarray(joint_names),
            fk_frame=DEFAULT_EE_LINK,
            source_stroke_directories=np.asarray(
                [str(stroke.output_dir) for stroke in strokes]
            ),
            source_stroke_input_sha256=np.asarray(
                [stroke.input_sha256 for stroke in strokes]
            ),
            source_stroke_deployment_path_id=np.asarray(
                [stroke.deployment_path_id for stroke in strokes]
            ),
            maximum_joint_step_rad=args.maximum_joint_step_rad,
            maximum_cartesian_error_m=args.maximum_cartesian_error_m,
            maximum_orientation_error_rad=args.maximum_orientation_error_rad,
            maximum_z_tracking_error_m=args.maximum_z_tracking_error_m,
            accepted=accepted,
            verdict=verdict,
        )
        atomic_write_json(
            args.output_dir / "multistroke_execution_metrics.json",
            metrics,
        )
        report = [
            "Multistroke full-pose execution",
            f"verdict: {verdict}",
            f"stroke_count: {len(strokes)}",
            f"sample_count: {len(q)}",
            f"segment_count: {len(manifest)}",
            f"rejection_reasons: {rejection_reasons}",
            (
                "maximum_consecutive_absolute_joint_step_rad: "
                f"{maximum_joint_step:.9f}"
            ),
            (
                "maximum_observed_cartesian_error_m: "
                f"{float(np.max(cartesian_error)):.9f}"
            ),
            (
                "maximum_observed_orientation_error_rad: "
                f"{float(np.max(orientation_error)):.9f}"
            ),
            (
                "maximum_observed_z_tracking_error_m: "
                f"{float(np.max(z_tracking_error)):.9f}"
            ),
            "",
            "Operator start requirement:",
            str(metrics["operator_start_requirement"]),
        ]
        if accepted:
            report.extend(
                [
                    "",
                    f"legacy_smartjoint_csv: {legacy_csv_path}",
                    f"legacy_smartjoint_rows: {legacy_row_count}",
                    (
                        "legacy_drawing_stroke_ids: "
                        f"{legacy_drawing_stroke_ids}"
                    ),
                ]
            )
        atomic_write_text(
            args.output_dir / "multistroke_execution_report.txt",
            "\n".join(report) + "\n",
        )
        save_plots(
            args.output_dir,
            timestamps,
            q,
            velocity,
            acceleration,
            jerk,
            desired_position,
            fk_position,
            orientation_error,
            cartesian_error,
            segment_type,
        )
        if accepted:
            atomic_write_csv(
                args.output_dir
                / "approved_multistroke_execution_trajectory.csv",
                approved_rows,
            )
            atomic_save_npz(
                args.output_dir
                / "approved_multistroke_execution_trajectory.npz",
                timestamps=timestamps,
                q=q,
                accepted=True,
                verdict=verdict,
                urdf_path=str(first.urdf_path),
                urdf_sha256=first.urdf_sha256,
            )
        print(verdict)
        report_path = (
            args.output_dir / "multistroke_execution_report.txt"
        )
        with report_path.open("a", encoding="utf-8") as report_handle:
            report_handle.write("\ntransition_diagnostics:\n")
            for diagnostic in TRANSITION_DIAGNOSTICS:
                report_handle.write(
                    json.dumps(diagnostic, sort_keys=True) + "\n"
                )
        return 0 if accepted else 2
    except Exception as exc:
        print(f"MULTISTROKE_EXECUTION_FAILED: {type(exc).__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
