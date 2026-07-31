#!/usr/bin/env python3
"""Generate a deployment input constrained to a previous stroke's IK branch."""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import math
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Mapping, Sequence, Tuple

import numpy as np

import generate_deployment_input_from_cartesian_csv as base_generator  # pyright: ignore[reportMissingImports]
from generate_ik_seed_path import (
    DEFAULT_EE_LINK,
    DEFAULT_JOINT_NAMES,
    HARD_JOINT_LIMIT_TOLERANCE_RAD,
    get_joint_bounds,
    load_robot,
)
from orientation_aware_adaptive_ik import (  # pyright: ignore[reportMissingImports]
    orientation_error_trajectory,
    trajectory_full_transform_fk,
)


ACCEPTED_SOURCE_VERDICT = "V8_1_DEPLOYMENT_TRAJECTORY_ACCEPTED"
SOURCE_FULL_NAME = "deployment_trajectory_full.npz"
SOURCE_METRICS_NAME = "deployment_metrics.json"
SOURCE_APPROVED_CSV_NAME = "approved_simulation_trajectory.csv"
SOURCE_APPROVED_NPZ_NAME = "approved_simulation_trajectory.npz"
FAILURE_SUFFIX = ".branch_continuity_failure.json"
EXPECTED_SAMPLE_COUNT = 100
EXPECTED_JOINT_COUNT = 6
TWO_PI = 2.0 * math.pi
TIMESTAMP_RTOL = 1.0e-7
TIMESTAMP_ATOL = 1.0e-9


class ConfigurationError(RuntimeError):
    """The requested generation configuration is invalid."""


@dataclass(frozen=True)
class PreviousStroke:
    directory: Path
    final_q: np.ndarray
    desired_path: np.ndarray
    timestamps: np.ndarray
    branch_seed_q: np.ndarray
    branch_seed_sample: int
    urdf_path: Path
    urdf_sha256: str
    joint_order: Tuple[str, ...]
    joint_order_source: str
    fk_frame: str
    target_rpy: np.ndarray
    target_rotation: np.ndarray
    target_quaternion: np.ndarray
    target_z: float
    orientation_gate: float
    z_gate: float
    input_sha256: str
    deployment_path_id: str


@dataclass
class CandidateResult:
    name: str
    generation_status: str
    output_data: Dict[str, np.ndarray] | None
    diagnostics: Dict[str, Any]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_csv", type=Path, required=True)
    parser.add_argument("--output_npz", type=Path, required=True)
    parser.add_argument("--path_name", required=True)
    parser.add_argument("--trajectory_duration_seconds", type=float)
    parser.add_argument("--roll", type=float)
    parser.add_argument("--pitch", type=float)
    parser.add_argument("--yaw", type=float)
    parser.add_argument("--mean_error_gate", type=float)
    parser.add_argument("--max_joint_step_gate", type=float)
    parser.add_argument("--maximum_orientation_error_gate_rad", type=float)
    parser.add_argument("--maximum_z_error_gate_m", type=float)
    parser.add_argument("--retry_profile")
    parser.add_argument("--device")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--previous_stroke_output_dir",
        type=Path,
        required=True,
    )
    parser.add_argument("--branch_seed_sample", type=int, default=-1)
    parser.add_argument(
        "--maximum_branch_seed_difference_rad",
        type=float,
        default=1.0,
    )
    return parser.parse_args(argv)


def scalar(value: Any, label: str) -> Any:
    array = np.asarray(value)
    if array.size != 1:
        raise ConfigurationError(f"{label} must contain one scalar")
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
    raise ConfigurationError(f"{label} must be a boolean")


def finite_scalar(value: Any, label: str) -> float:
    result = float(scalar(value, label))
    if not math.isfinite(result):
        raise ConfigurationError(f"{label} must be finite")
    return result


def finite_array(
    value: Any,
    label: str,
    shape: Tuple[int, ...],
) -> np.ndarray:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number):
        raise ConfigurationError(f"{label} must be numeric")
    result = np.asarray(array, dtype=np.float64)
    if result.shape != shape:
        raise ConfigurationError(
            f"{label} must have shape {shape}, got {result.shape}"
        )
    if not np.all(np.isfinite(result)):
        raise ConfigurationError(f"{label} contains non-finite values")
    return result


def string_tuple(value: Any, label: str) -> Tuple[str, ...]:
    array = np.asarray(value)
    if array.ndim != 1:
        raise ConfigurationError(f"{label} must be one-dimensional")
    return tuple(
        item.decode("utf-8") if isinstance(item, bytes) else str(item)
        for item in array.tolist()
    )


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
        raise ConfigurationError(f"Cannot read NPZ {path}: {exc}") from exc


def load_json(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigurationError(f"Cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ConfigurationError(f"JSON root must be an object: {path}")
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise ConfigurationError(f"Cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def require_file(path: Path) -> None:
    if not path.is_file():
        raise ConfigurationError(f"Required file is missing: {path}")


def require_field(
    data: Mapping[str, Any],
    name: str,
    label: str,
) -> Any:
    if name not in data:
        raise ConfigurationError(f"{label} is missing {name!r}")
    return data[name]


def load_previous_stroke(
    directory: Path,
    branch_seed_sample: int,
) -> PreviousStroke:
    directory = directory.expanduser().resolve()
    full_path = directory / SOURCE_FULL_NAME
    metrics_path = directory / SOURCE_METRICS_NAME
    approved_csv = directory / SOURCE_APPROVED_CSV_NAME
    approved_npz = directory / SOURCE_APPROVED_NPZ_NAME
    for path in (full_path, metrics_path, approved_csv, approved_npz):
        require_file(path)
    full = load_npz(full_path)
    metrics = load_json(metrics_path)
    if metrics.get("accepted") is not True:
        raise ConfigurationError("Previous stroke metrics accepted must be true")
    if metrics.get("verdict") != ACCEPTED_SOURCE_VERDICT:
        raise ConfigurationError(
            "Previous stroke verdict must be "
            f"{ACCEPTED_SOURCE_VERDICT}"
        )
    final_q = finite_array(
        require_field(full, "final_q", "previous full NPZ"),
        "previous final_q",
        (EXPECTED_SAMPLE_COUNT, EXPECTED_JOINT_COUNT),
    )
    desired_path = finite_array(
        require_field(full, "desired_path", "previous full NPZ"),
        "previous desired_path",
        (EXPECTED_SAMPLE_COUNT, 3),
    )
    timestamps = finite_array(
        require_field(full, "timestamps", "previous full NPZ"),
        "previous timestamps",
        (EXPECTED_SAMPLE_COUNT,),
    )
    if np.any(np.diff(timestamps) <= 0.0):
        raise ConfigurationError(
            "Previous stroke timestamps must strictly increase"
        )
    try:
        with approved_csv.open(
            "r",
            encoding="utf-8",
            newline="",
        ) as handle:
            reader = csv.DictReader(handle)
            expected_columns = [
                "time_seconds",
                "q1",
                "q2",
                "q3",
                "q4",
                "q5",
                "q6",
            ]
            if reader.fieldnames != expected_columns:
                raise ConfigurationError(
                    "Previous approved CSV columns must be exactly "
                    f"{expected_columns}, got {reader.fieldnames}"
                )
            approved_rows = list(reader)
    except OSError as exc:
        raise ConfigurationError(
            f"Cannot read previous approved CSV {approved_csv}: {exc}"
        ) from exc
    if len(approved_rows) != EXPECTED_SAMPLE_COUNT:
        raise ConfigurationError(
            "Previous approved CSV must contain exactly 100 rows"
        )
    try:
        approved_csv_timestamps = np.asarray(
            [float(row["time_seconds"]) for row in approved_rows],
            dtype=np.float64,
        )
        approved_csv_q = np.asarray(
            [
                [
                    float(row[f"q{joint + 1}"])
                    for joint in range(EXPECTED_JOINT_COUNT)
                ]
                for row in approved_rows
            ],
            dtype=np.float64,
        )
    except (TypeError, ValueError) as exc:
        raise ConfigurationError(
            "Previous approved CSV contains invalid numeric values"
        ) from exc
    if not np.allclose(
        approved_csv_timestamps,
        timestamps,
        rtol=TIMESTAMP_RTOL,
        atol=TIMESTAMP_ATOL,
    ):
        raise ConfigurationError(
            "Previous approved CSV timestamps differ from the full artifact"
        )
    if not np.array_equal(approved_csv_q, final_q):
        raise ConfigurationError(
            "Previous approved CSV q differs from the full artifact"
        )

    approved_data = load_npz(approved_npz)
    approved_npz_timestamps = finite_array(
        require_field(
            approved_data,
            "timestamps",
            "previous approved NPZ",
        ),
        "previous approved NPZ timestamps",
        (EXPECTED_SAMPLE_COUNT,),
    )
    approved_npz_q = finite_array(
        require_field(approved_data, "q", "previous approved NPZ"),
        "previous approved NPZ q",
        (EXPECTED_SAMPLE_COUNT, EXPECTED_JOINT_COUNT),
    )
    if not np.allclose(
        approved_npz_timestamps,
        timestamps,
        rtol=TIMESTAMP_RTOL,
        atol=TIMESTAMP_ATOL,
    ):
        raise ConfigurationError(
            "Previous approved NPZ timestamps differ from the full artifact"
        )
    if not np.array_equal(approved_npz_q, final_q):
        raise ConfigurationError(
            "Previous approved NPZ q differs from the full artifact"
        )
    normalized_index = branch_seed_sample
    if normalized_index < 0:
        normalized_index += EXPECTED_SAMPLE_COUNT
    if normalized_index < 0 or normalized_index >= EXPECTED_SAMPLE_COUNT:
        raise ConfigurationError(
            f"branch_seed_sample {branch_seed_sample} is out of range"
        )
    branch_seed_q = np.array(final_q[normalized_index], copy=True)
    urdf_path = Path(
        text_scalar(
            require_field(full, "urdf_path", "previous full NPZ"),
            "previous urdf_path",
        )
    ).expanduser().resolve()
    require_file(urdf_path)
    urdf_hash = text_scalar(
        require_field(full, "urdf_sha256", "previous full NPZ"),
        "previous urdf_sha256",
    )
    if sha256_file(urdf_path) != urdf_hash:
        raise ConfigurationError("Previous stroke URDF SHA-256 mismatch")
    joint_order_value = full.get("joint_order", metrics.get("joint_order"))
    joint_order = (
        string_tuple(
            joint_order_value,
            "previous joint_order",
        )
        if joint_order_value is not None
        else tuple(str(name) for name in DEFAULT_JOINT_NAMES)
    )
    default_joint_order = tuple(
        str(name) for name in DEFAULT_JOINT_NAMES
    )
    if len(joint_order) != EXPECTED_JOINT_COUNT:
        raise ConfigurationError("Previous joint_order must contain six names")
    if joint_order != default_joint_order:
        raise ConfigurationError(
            "Previous stroke joint order differs from DEFAULT_JOINT_NAMES"
        )
    joint_order_source = (
        "artifact_metadata"
        if joint_order_value is not None
        else "repository_default_fallback"
    )
    orientation_frame = text_scalar(
        require_field(
            full,
            "orientation_fk_frame",
            "previous full NPZ",
        ),
        "previous orientation_fk_frame",
    )
    z_frame = text_scalar(
        require_field(full, "z_fk_frame", "previous full NPZ"),
        "previous z_fk_frame",
    )
    if orientation_frame != z_frame:
        raise ConfigurationError(
            "Previous orientation and Z FK frames differ"
        )
    target_rpy = finite_array(
        require_field(full, "target_rpy", "previous full NPZ"),
        "previous target_rpy",
        (3,),
    )
    target_rotation = finite_array(
        require_field(
            full,
            "target_rotation_matrix",
            "previous full NPZ",
        ),
        "previous target_rotation_matrix",
        (3, 3),
    )
    target_quaternion = finite_array(
        require_field(
            full,
            "target_quaternion",
            "previous full NPZ",
        ),
        "previous target_quaternion",
        (4,),
    )
    target_z = finite_scalar(
        require_field(full, "target_z", "previous full NPZ"),
        "previous target_z",
    )
    orientation_gate = finite_scalar(
        require_field(
            full,
            "maximum_orientation_error_gate_rad",
            "previous full NPZ",
        ),
        "previous maximum_orientation_error_gate_rad",
    )
    z_gate = finite_scalar(
        require_field(
            full,
            "maximum_z_error_gate_m",
            "previous full NPZ",
        ),
        "previous maximum_z_error_gate_m",
    )
    return PreviousStroke(
        directory=directory,
        final_q=final_q,
        desired_path=desired_path,
        timestamps=timestamps,
        branch_seed_q=branch_seed_q,
        branch_seed_sample=branch_seed_sample,
        urdf_path=urdf_path,
        urdf_sha256=urdf_hash,
        joint_order=joint_order,
        joint_order_source=joint_order_source,
        fk_frame=orientation_frame,
        target_rpy=target_rpy,
        target_rotation=target_rotation,
        target_quaternion=target_quaternion,
        target_z=target_z,
        orientation_gate=orientation_gate,
        z_gate=z_gate,
        input_sha256=text_scalar(
            require_field(full, "input_sha256", "previous full NPZ"),
            "previous input_sha256",
        ),
        deployment_path_id=text_scalar(
            require_field(
                full,
                "deployment_path_id",
                "previous full NPZ",
            ),
            "previous deployment_path_id",
        ),
    )


def base_cli_arguments(args: argparse.Namespace, output_npz: Path) -> List[str]:
    result = [
        "generate_deployment_input_from_cartesian_csv.py",
        "--input_csv",
        str(args.input_csv),
        "--output_npz",
        str(output_npz),
        "--path_name",
        args.path_name,
        "--overwrite",
    ]
    optional = (
        ("trajectory_duration_seconds", "--trajectory_duration_seconds"),
        ("roll", "--roll"),
        ("pitch", "--pitch"),
        ("yaw", "--yaw"),
        ("mean_error_gate", "--mean_error_gate"),
        ("max_joint_step_gate", "--max_joint_step_gate"),
        (
            "maximum_orientation_error_gate_rad",
            "--maximum_orientation_error_gate_rad",
        ),
        ("maximum_z_error_gate_m", "--maximum_z_error_gate_m"),
        ("retry_profile", "--retry_profile"),
        ("device", "--device"),
    )
    for attribute, option in optional:
        value = getattr(args, attribute)
        if value is not None:
            result.extend((option, str(value)))
    return result


def find_argument_name(
    arguments: Mapping[str, Any],
    preferred: Sequence[str],
    expected_shape: Tuple[int, ...],
) -> str:
    for name in preferred:
        if name in arguments:
            return name
    matches = [
        name
        for name, value in arguments.items()
        if isinstance(value, np.ndarray) and value.shape == expected_shape
    ]
    if len(matches) != 1:
        raise ConfigurationError(
            f"Cannot identify adaptive IK argument with shape {expected_shape}; "
            f"matches={matches}"
        )
    return matches[0]


def extract_limits(
    arguments: Mapping[str, Any],
    canonical: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray] | None:
    lower_names = ("lower", "joint_lower", "lower_limits")
    upper_names = ("upper", "joint_upper", "upper_limits")
    lower = next(
        (
            np.asarray(arguments[name], dtype=np.float64)
            for name in lower_names
            if name in arguments
        ),
        None,
    )
    upper = next(
        (
            np.asarray(arguments[name], dtype=np.float64)
            for name in upper_names
            if name in arguments
        ),
        None,
    )
    if (
        lower is not None
        and upper is not None
        and lower.shape == (canonical.shape[1],)
        and upper.shape == (canonical.shape[1],)
    ):
        return lower, upper
    for name in ("bounds", "joint_bounds"):
        if name in arguments:
            bounds = np.asarray(arguments[name], dtype=np.float64)
            if bounds.shape == (canonical.shape[1], 2):
                return bounds[:, 0], bounds[:, 1]
    return None


def align_periodic_canonical(
    canonical: np.ndarray,
    branch_seed_q: np.ndarray,
    limits: Tuple[np.ndarray, np.ndarray] | None,
) -> np.ndarray:
    aligned = np.array(canonical, dtype=np.float64, copy=True)
    if limits is None:
        return aligned
    lower, upper = limits
    for joint in range(aligned.shape[1]):
        shift_count = int(
            np.rint(
                (branch_seed_q[joint] - aligned[0, joint]) / TWO_PI
            )
        )
        proposed = aligned[:, joint] + shift_count * TWO_PI
        if (
            np.all(proposed >= lower[joint])
            and np.all(proposed <= upper[joint])
        ):
            aligned[:, joint] = proposed
    return aligned


@contextmanager
def patched_adaptive_ik(
    candidate_name: str,
    branch_seed_q: np.ndarray,
) -> Iterator[None]:
    if not hasattr(base_generator, "adaptive_refine_full_pose_path"):
        raise ConfigurationError(
            "Existing generator does not expose adaptive_refine_full_pose_path"
        )
    original: Callable[..., Any] = getattr(
        base_generator,
        "adaptive_refine_full_pose_path",
    )
    signature = inspect.signature(original)

    def branch_constrained(*call_args: Any, **call_kwargs: Any) -> Any:
        bound = signature.bind_partial(*call_args, **call_kwargs)
        arguments = bound.arguments
        start_name = find_argument_name(
            arguments,
            ("q_start", "start_q", "initial_q"),
            (EXPECTED_JOINT_COUNT,),
        )
        canonical_name = find_argument_name(
            arguments,
            (
                "canonical_mlp_q",
                "canonical_q",
                "canonical_seed_q",
                "canonical_seed_path",
            ),
            (EXPECTED_SAMPLE_COUNT, EXPECTED_JOINT_COUNT),
        )
        canonical = np.asarray(
            arguments[canonical_name],
            dtype=np.float64,
        )
        arguments[start_name] = np.array(branch_seed_q, copy=True)
        if candidate_name == "mlp_canonical_previous_branch_start":
            replacement = canonical
        elif candidate_name == "repeated_branch_seed":
            replacement = np.repeat(
                branch_seed_q[np.newaxis, :],
                EXPECTED_SAMPLE_COUNT,
                axis=0,
            )
        elif candidate_name == "branch_aligned_mlp_canonical":
            replacement = align_periodic_canonical(
                canonical,
                branch_seed_q,
                extract_limits(arguments, canonical),
            )
        else:
            raise ConfigurationError(
                f"Unsupported candidate name {candidate_name}"
            )
        arguments[canonical_name] = replacement
        return original(*bound.args, **bound.kwargs)

    setattr(
        base_generator,
        "adaptive_refine_full_pose_path",
        branch_constrained,
    )
    try:
        yield
    finally:
        setattr(
            base_generator,
            "adaptive_refine_full_pose_path",
            original,
        )


@contextmanager
def replaced_argv(arguments: Sequence[str]) -> Iterator[None]:
    original = sys.argv
    sys.argv = list(arguments)
    try:
        yield
    finally:
        sys.argv = original


def invoke_base_generator(
    args: argparse.Namespace,
    output_npz: Path,
    candidate_name: str,
    branch_seed_q: np.ndarray,
) -> Tuple[int, str]:
    try:
        with patched_adaptive_ik(candidate_name, branch_seed_q):
            with replaced_argv(base_cli_arguments(args, output_npz)):
                result = base_generator.main()
        return int(result), f"base_generator_exit_{int(result)}"
    except SystemExit as exc:
        code = int(exc.code) if isinstance(exc.code, int) else 1
        return code, f"base_generator_system_exit_{code}"
    except Exception as exc:
        return 1, f"candidate_generation_exception: {exc}"


def output_number(
    data: Mapping[str, Any],
    names: Sequence[str],
    cli_value: float | None,
    label: str,
) -> float:
    for name in names:
        if name in data:
            value = finite_scalar(data[name], label)
            if value <= 0.0:
                raise ConfigurationError(f"{label} must be positive")
            return value
    if cli_value is not None and math.isfinite(cli_value) and cli_value > 0.0:
        return float(cli_value)
    raise ConfigurationError(f"Candidate output is missing {label}")


def unresolved_count(data: Mapping[str, Any]) -> int:
    for name in (
        "unresolved_timestep_count",
        "unresolved_timesteps_count",
    ):
        if name in data:
            value = int(scalar(data[name], name))
            return max(value, 0)
    for name in ("unresolved_timesteps", "unresolved_mask"):
        if name in data:
            return int(np.count_nonzero(np.asarray(data[name])))
    return 0


def compatibility_errors(
    previous: PreviousStroke,
    data: Mapping[str, Any],
) -> List[str]:
    errors: List[str] = []
    candidate_urdf = Path(
        text_scalar(
            require_field(data, "urdf_path", "candidate"),
            "candidate urdf_path",
        )
    ).expanduser().resolve()
    candidate_hash = text_scalar(
        require_field(data, "urdf_sha256", "candidate"),
        "candidate urdf_sha256",
    )
    candidate_order = (
        string_tuple(data["joint_order"], "candidate joint order")
        if "joint_order" in data
        else tuple(str(name) for name in DEFAULT_JOINT_NAMES)
    )
    default_order = tuple(str(name) for name in DEFAULT_JOINT_NAMES)
    orientation_frame = text_scalar(
        require_field(data, "orientation_fk_frame", "candidate"),
        "candidate orientation_fk_frame",
    )
    z_frame = text_scalar(
        require_field(data, "z_fk_frame", "candidate"),
        "candidate z_fk_frame",
    )
    comparisons = (
        (candidate_urdf == previous.urdf_path, "URDF path"),
        (candidate_hash == previous.urdf_sha256, "URDF SHA-256"),
        (
            candidate_order == previous.joint_order,
            "previous joint order",
        ),
        (
            candidate_order == default_order,
            "default joint order",
        ),
        (
            orientation_frame == previous.fk_frame
            and z_frame == previous.fk_frame,
            "FK frame",
        ),
        (
            np.array_equal(
                finite_array(data["target_rpy"], "target_rpy", (3,)),
                previous.target_rpy,
            ),
            "target RPY",
        ),
        (
            np.array_equal(
                finite_array(
                    data["target_rotation_matrix"],
                    "target_rotation_matrix",
                    (3, 3),
                ),
                previous.target_rotation,
            ),
            "target rotation",
        ),
        (
            np.array_equal(
                finite_array(
                    data["target_quaternion"],
                    "target_quaternion",
                    (4,),
                ),
                previous.target_quaternion,
            ),
            "target quaternion",
        ),
        (
            finite_scalar(data["target_z"], "target_z")
            == previous.target_z,
            "target Z",
        ),
        (
            finite_scalar(
                data["maximum_orientation_error_gate_rad"],
                "maximum_orientation_error_gate_rad",
            )
            == previous.orientation_gate,
            "orientation gate",
        ),
        (
            finite_scalar(
                data["maximum_z_error_gate_m"],
                "maximum_z_error_gate_m",
            )
            == previous.z_gate,
            "Z gate",
        ),
    )
    for matches, label in comparisons:
        if not matches:
            errors.append(f"incompatible_{label.lower().replace(' ', '_')}")
    return errors


def evaluate_candidate(
    name: str,
    generation_status: str,
    data: Dict[str, np.ndarray],
    previous: PreviousStroke,
    args: argparse.Namespace,
) -> CandidateResult:
    rejection_reasons = compatibility_errors(previous, data)
    desired_path = finite_array(
        require_field(data, "desired_path", "candidate"),
        "candidate desired_path",
        (EXPECTED_SAMPLE_COUNT, 3),
    )
    q = finite_array(
        require_field(data, "strong_prior_q", "candidate"),
        "candidate strong_prior_q",
        (EXPECTED_SAMPLE_COUNT, EXPECTED_JOINT_COUNT),
    )
    timestamps = finite_array(
        require_field(data, "timestamps", "candidate"),
        "candidate timestamps",
        (EXPECTED_SAMPLE_COUNT,),
    )
    if np.any(np.diff(timestamps) <= 0.0):
        rejection_reasons.append("timestamps_not_strictly_increasing")
    target_z = finite_scalar(data["target_z"], "candidate target_z")
    if not np.allclose(
        desired_path[:, 2],
        target_z,
        rtol=0.0,
        atol=1.0e-12,
    ):
        rejection_reasons.append("desired_path_target_z_not_constant")

    urdf_path = Path(
        text_scalar(data["urdf_path"], "candidate urdf_path")
    ).expanduser().resolve()
    require_file(urdf_path)
    if sha256_file(urdf_path) != text_scalar(
        data["urdf_sha256"],
        "candidate urdf_sha256",
    ):
        rejection_reasons.append("candidate_urdf_sha256_mismatch")
    joint_order = (
        string_tuple(data["joint_order"], "candidate joint order")
        if "joint_order" in data
        else tuple(str(name) for name in DEFAULT_JOINT_NAMES)
    )
    if len(joint_order) != EXPECTED_JOINT_COUNT:
        rejection_reasons.append("candidate_joint_order_length_invalid")
    if joint_order != previous.joint_order:
        rejection_reasons.append(
            "candidate_joint_order_differs_from_previous_stroke"
        )
    fk_frame = text_scalar(
        data["orientation_fk_frame"],
        "candidate orientation_fk_frame",
    )
    robot = load_robot(urdf_path)
    fk_position, fk_rotation, _ = trajectory_full_transform_fk(
        robot,
        q,
        joint_order,
        fk_frame,
    )
    target_rotation = finite_array(
        data["target_rotation_matrix"],
        "candidate target_rotation_matrix",
        (3, 3),
    )
    cartesian_error = np.linalg.norm(fk_position - desired_path, axis=1)
    orientation_error = orientation_error_trajectory(
        target_rotation,
        fk_rotation,
    )
    z_error = np.abs(fk_position[:, 2] - desired_path[:, 2])
    internal_step = float(np.max(np.abs(np.diff(q, axis=0))))
    branch_difference = float(
        np.max(np.abs(q[0] - previous.branch_seed_q))
    )
    bounds = np.asarray(
        get_joint_bounds(
            robot,
            joint_order,
            -np.pi,
            np.pi,
        ),
        dtype=np.float64,
    )
    if bounds.shape != (EXPECTED_JOINT_COUNT, 2):
        raise ConfigurationError(
            f"Joint bounds have unexpected shape {bounds.shape}"
        )
    violations = np.logical_or(
        q < bounds[:, 0][None, :] - HARD_JOINT_LIMIT_TOLERANCE_RAD,
        q > bounds[:, 1][None, :] + HARD_JOINT_LIMIT_TOLERANCE_RAD,
    )
    violation_count = int(np.count_nonzero(violations))
    unresolved = unresolved_count(data)
    mean_gate = output_number(
        data,
        ("mean_error_gate", "mean_cartesian_error_gate"),
        args.mean_error_gate,
        "mean Cartesian error gate",
    )
    step_gate = output_number(
        data,
        ("max_joint_step_gate", "maximum_joint_step_gate_rad"),
        args.max_joint_step_gate,
        "maximum joint-step gate",
    )
    orientation_gate = output_number(
        data,
        ("maximum_orientation_error_gate_rad",),
        args.maximum_orientation_error_gate_rad,
        "maximum orientation-error gate",
    )
    z_gate = output_number(
        data,
        ("maximum_z_error_gate_m",),
        args.maximum_z_error_gate_m,
        "maximum Z-error gate",
    )
    mean_cartesian = float(np.mean(cartesian_error))
    maximum_cartesian = float(np.max(cartesian_error))
    mean_orientation = float(np.mean(orientation_error))
    maximum_orientation = float(np.max(orientation_error))
    mean_z = float(np.mean(z_error))
    maximum_z = float(np.max(z_error))
    all_finite = bool(
        np.all(np.isfinite(q))
        and np.all(np.isfinite(fk_position))
        and np.all(np.isfinite(fk_rotation))
        and np.all(np.isfinite(cartesian_error))
        and np.all(np.isfinite(orientation_error))
        and np.all(np.isfinite(z_error))
    )
    generation_success = (
        bool_scalar(data["generation_success"], "generation_success")
        if "generation_success" in data
        else True
    )
    if not generation_success:
        rejection_reasons.append("base_generation_unsuccessful")
    if not all_finite:
        rejection_reasons.append("non_finite_values")
    if unresolved:
        rejection_reasons.append("unresolved_timesteps")
    if violation_count:
        rejection_reasons.append("hard_joint_limit_violation")
    if internal_step > step_gate:
        rejection_reasons.append("maximum_internal_joint_step_exceeded")
    if mean_cartesian > mean_gate:
        rejection_reasons.append("mean_cartesian_error_exceeded")
    if maximum_orientation > orientation_gate:
        rejection_reasons.append("maximum_orientation_error_exceeded")
    if maximum_z > z_gate:
        rejection_reasons.append("maximum_z_error_exceeded")
    if branch_difference > args.maximum_branch_seed_difference_rad:
        rejection_reasons.append("whole_arm_branch_seed_difference_exceeded")

    diagnostics: Dict[str, Any] = {
        "candidate_name": name,
        "candidate_generation_status": generation_status,
        "branch_seed_difference_rad": branch_difference,
        "maximum_internal_joint_step_rad": internal_step,
        "mean_cartesian_error_m": mean_cartesian,
        "maximum_cartesian_error_m": maximum_cartesian,
        "mean_orientation_error_rad": mean_orientation,
        "maximum_orientation_error_rad": maximum_orientation,
        "mean_z_error_m": mean_z,
        "maximum_z_error_m": maximum_z,
        "joint_limit_violation_count": violation_count,
        "unresolved_timestep_count": unresolved,
        "rejection_reasons": sorted(set(rejection_reasons)),
        "gate_pass": not rejection_reasons,
    }
    return CandidateResult(
        name=name,
        generation_status=generation_status,
        output_data=data,
        diagnostics=diagnostics,
    )


def candidate_order(candidate: CandidateResult) -> Tuple[float, ...]:
    diagnostic = candidate.diagnostics
    return (
        float(diagnostic["branch_seed_difference_rad"]),
        float(diagnostic["maximum_internal_joint_step_rad"]),
        float(diagnostic["mean_cartesian_error_m"]),
        float(diagnostic["maximum_orientation_error_rad"]),
        float(diagnostic["maximum_z_error_m"]),
    )


def json_safe(value: Any) -> Any:
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(
                json_safe(dict(value)),
                handle,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_write_npz(path: Path, data: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **data)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def failure_path(output_npz: Path) -> Path:
    return output_npz.with_suffix(output_npz.suffix + FAILURE_SUFFIX)


def validate_cli(args: argparse.Namespace) -> None:
    require_file(args.input_csv.expanduser().resolve())
    if (
        not math.isfinite(args.maximum_branch_seed_difference_rad)
        or args.maximum_branch_seed_difference_rad <= 0.0
    ):
        raise ConfigurationError(
            "maximum_branch_seed_difference_rad must be positive"
        )
    for name in (
        "trajectory_duration_seconds",
        "mean_error_gate",
        "max_joint_step_gate",
        "maximum_orientation_error_gate_rad",
        "maximum_z_error_gate_m",
    ):
        value = getattr(args, name)
        if value is not None and (
            not math.isfinite(value) or value <= 0.0
        ):
            raise ConfigurationError(f"{name} must be positive")
    output = args.output_npz.expanduser().resolve()
    if output.exists() and not args.overwrite:
        raise ConfigurationError(
            f"Output exists; pass --overwrite to replace it: {output}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_npz = args.output_npz.expanduser().resolve()
    diagnostic_path = failure_path(output_npz)
    previous: PreviousStroke | None = None
    try:
        validate_cli(args)
        previous = load_previous_stroke(
            args.previous_stroke_output_dir,
            args.branch_seed_sample,
        )
        candidate_names = (
            "mlp_canonical_previous_branch_start",
            "repeated_branch_seed",
            "branch_aligned_mlp_canonical",
        )
        results: List[CandidateResult] = []
        with tempfile.TemporaryDirectory(
            prefix="branch_continuous_deployment_"
        ) as temporary_directory:
            temporary_root = Path(temporary_directory)
            for candidate_number, candidate_name in enumerate(candidate_names):
                candidate_path = (
                    temporary_root / f"candidate_{candidate_number}.npz"
                )
                exit_code, status = invoke_base_generator(
                    args,
                    candidate_path,
                    candidate_name,
                    previous.branch_seed_q,
                )
                if not candidate_path.is_file():
                    results.append(
                        CandidateResult(
                            name=candidate_name,
                            generation_status=status,
                            output_data=None,
                            diagnostics={
                                "candidate_name": candidate_name,
                                "candidate_generation_status": status,
                                "branch_seed_difference_rad": None,
                                "maximum_internal_joint_step_rad": None,
                                "mean_cartesian_error_m": None,
                                "maximum_cartesian_error_m": None,
                                "mean_orientation_error_rad": None,
                                "maximum_orientation_error_rad": None,
                                "mean_z_error_m": None,
                                "maximum_z_error_m": None,
                                "joint_limit_violation_count": None,
                                "unresolved_timestep_count": None,
                                "rejection_reasons": [
                                    f"candidate_output_missing_exit_{exit_code}"
                                ],
                                "gate_pass": False,
                            },
                        )
                    )
                    continue
                try:
                    results.append(
                        evaluate_candidate(
                            candidate_name,
                            status,
                            load_npz(candidate_path),
                            previous,
                            args,
                        )
                    )
                except Exception as exc:
                    results.append(
                        CandidateResult(
                            name=candidate_name,
                            generation_status=status,
                            output_data=None,
                            diagnostics={
                                "candidate_name": candidate_name,
                                "candidate_generation_status": status,
                                "branch_seed_difference_rad": None,
                                "maximum_internal_joint_step_rad": None,
                                "mean_cartesian_error_m": None,
                                "maximum_cartesian_error_m": None,
                                "mean_orientation_error_rad": None,
                                "maximum_orientation_error_rad": None,
                                "mean_z_error_m": None,
                                "maximum_z_error_m": None,
                                "joint_limit_violation_count": None,
                                "unresolved_timestep_count": None,
                                "rejection_reasons": [
                                    f"candidate_evaluation_failure: {exc}"
                                ],
                                "gate_pass": False,
                            },
                        )
                    )

        passing = [
            result
            for result in results
            if result.diagnostics.get("gate_pass") is True
            and result.output_data is not None
        ]
        diagnostics = [result.diagnostics for result in results]
        if not passing:
            atomic_write_json(
                diagnostic_path,
                {
                    "generation_status": "all_candidates_rejected",
                    "branch_seed_source_directory": str(previous.directory),
                    "branch_seed_joint_order_source": (
                        previous.joint_order_source
                    ),
                    "branch_seed_sample": previous.branch_seed_sample,
                    "branch_seed_q": previous.branch_seed_q,
                    "maximum_branch_seed_difference_rad": (
                        args.maximum_branch_seed_difference_rad
                    ),
                    "candidate_diagnostics": diagnostics,
                },
            )
            if output_npz.exists():
                output_npz.unlink()
            return 2

        selected = min(passing, key=candidate_order)
        assert selected.output_data is not None
        output_data: Dict[str, Any] = dict(selected.output_data)
        output_data.update(
            {
                "joint_order": np.asarray(
                    tuple(str(name) for name in DEFAULT_JOINT_NAMES)
                ),
                "branch_continuity_enforced": np.asarray(True),
                "branch_seed_q": previous.branch_seed_q,
                "branch_seed_sample": np.asarray(
                    previous.branch_seed_sample,
                    dtype=np.int64,
                ),
                "branch_seed_source_directory": np.asarray(
                    str(previous.directory)
                ),
                "branch_seed_source_input_sha256": np.asarray(
                    previous.input_sha256
                ),
                "branch_seed_source_deployment_path_id": np.asarray(
                    previous.deployment_path_id
                ),
                "branch_seed_joint_order_source": np.asarray(
                    previous.joint_order_source
                ),
                "maximum_branch_seed_difference_rad": np.asarray(
                    args.maximum_branch_seed_difference_rad,
                    dtype=np.float64,
                ),
                "observed_branch_seed_difference_rad": np.asarray(
                    selected.diagnostics[
                        "branch_seed_difference_rad"
                    ],
                    dtype=np.float64,
                ),
                "selected_branch_candidate": np.asarray(selected.name),
                "branch_candidate_diagnostics_json": np.asarray(
                    json.dumps(
                        json_safe(diagnostics),
                        sort_keys=True,
                        allow_nan=False,
                    )
                ),
            }
        )
        atomic_write_npz(output_npz, output_data)
        if diagnostic_path.exists():
            diagnostic_path.unlink()
        print("BRANCH_CONTINUOUS_DEPLOYMENT_INPUT_GENERATION_PASSED")
        return 0
    except ConfigurationError as exc:
        atomic_write_json(
            diagnostic_path,
            {
                "generation_status": "configuration_failure",
                "error": str(exc),
                "branch_seed_joint_order_source": (
                    previous.joint_order_source
                    if previous is not None
                    else "unavailable"
                ),
            },
        )
        return 1
    except Exception as exc:
        atomic_write_json(
            diagnostic_path,
            {
                "generation_status": "runtime_failure",
                "error": str(exc),
                "branch_seed_joint_order_source": (
                    previous.joint_order_source
                    if previous is not None
                    else "unavailable"
                ),
            },
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
