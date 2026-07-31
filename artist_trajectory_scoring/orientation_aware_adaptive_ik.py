#!/usr/bin/env python3
"""Orientation-aware adaptive sequential IK for xMateCR7 deployment priors.

This non-frozen helper preserves the repository's established numerical IK
method (SciPy L-BFGS-B, URDF bounds, continuity regularization, canonical-MLP
retry seeds, and robust restarts) while extending the objective and acceptance
checks from position-only IK to full-pose IK.

Quaternions use ROS/tf ``quaternion_from_euler(roll, pitch, yaw)`` ordering:
``[x, y, z, w]``. The corresponding rotation convention is the fixed-axis
roll-pitch-yaw composition ``Rz(yaw) @ Ry(pitch) @ Rx(roll)``.
"""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

from generate_adaptive_mlp_ik_bootstrap_prior import (
    ADAPTIVE_STAGE1_MAX_ITERS,
    ADAPTIVE_STAGE1_SMOOTH_WEIGHT,
    ADAPTIVE_STAGE2_MAX_ITERS,
    ADAPTIVE_STAGE2_MAX_ERROR_THRESHOLD,
    ADAPTIVE_STAGE2_SMOOTH_WEIGHT,
    IK_CARTESIAN_TOLERANCE,
    IK_FTOL,
    JOINT_DIM,
    retry_seeds,
)
from generate_ik_seed_path import (
    DEFAULT_JOINT_LIMIT_SAFETY_MARGIN_RAD,
    HARD_JOINT_LIMIT_TOLERANCE_RAD,
    check_joint_limits,
    clip_to_bounds,
)


QUATERNION_ORDER = "xyzw"
ORIENTATION_OBJECTIVE_WEIGHT = 1.0
ROTATION_VALIDATION_ATOL = 1.0e-6


@dataclass(frozen=True)
class FullTransform:
    position: np.ndarray
    rotation_matrix: np.ndarray
    quaternion: np.ndarray


@dataclass(frozen=True)
class PoseSolverParameters:
    stage: int
    smooth_weight: float
    max_iters: int
    ftol: float
    position_tolerance_m: float
    orientation_tolerance_rad: float
    orientation_objective_weight: float


@dataclass
class PoseIkAttempt:
    q: np.ndarray
    transform: FullTransform
    position_error_m: float
    orientation_error_rad: float
    solver_success: bool
    nit: int
    message: str


@dataclass
class PoseStageResult:
    q: np.ndarray
    positions: np.ndarray
    rotations: np.ndarray
    quaternions: np.ndarray
    position_errors_m: np.ndarray
    orientation_errors_rad: np.ndarray
    step_records: List[Dict[str, Any]]
    failed_timestep_count: int
    retry_count: int
    runtime_seconds: float
    parameters: PoseSolverParameters


@dataclass(frozen=True)
class AdaptivePoseResult:
    q: np.ndarray
    positions: np.ndarray
    rotations: np.ndarray
    quaternions: np.ndarray
    position_errors_m: np.ndarray
    orientation_errors_rad: np.ndarray
    unresolved_timestep_count: int
    selected_candidate_name: str
    candidate_table: Tuple[Mapping[str, Any], ...]
    adaptive_metadata: Mapping[str, Any]


def quaternion_from_euler_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Return the tf-compatible fixed-axis RPY quaternion in XYZW order."""
    rpy = np.asarray([roll, pitch, yaw], dtype=np.float64)
    if rpy.shape != (3,) or not np.all(np.isfinite(rpy)):
        raise ValueError("roll, pitch, and yaw must be finite scalars")
    half_roll = 0.5 * float(roll)
    half_pitch = 0.5 * float(pitch)
    half_yaw = 0.5 * float(yaw)
    sin_roll, cos_roll = math.sin(half_roll), math.cos(half_roll)
    sin_pitch, cos_pitch = math.sin(half_pitch), math.cos(half_pitch)
    sin_yaw, cos_yaw = math.sin(half_yaw), math.cos(half_yaw)
    # This is the formula used by tf.transformations.quaternion_from_euler
    # for its default static XYZ axes, returned in [x, y, z, w] order.
    quaternion = np.asarray(
        [
            sin_roll * cos_pitch * cos_yaw
            - cos_roll * sin_pitch * sin_yaw,
            cos_roll * sin_pitch * cos_yaw
            + sin_roll * cos_pitch * sin_yaw,
            cos_roll * cos_pitch * sin_yaw
            - sin_roll * sin_pitch * cos_yaw,
            cos_roll * cos_pitch * cos_yaw
            + sin_roll * sin_pitch * sin_yaw,
        ],
        dtype=np.float64,
    )
    return canonicalize_quaternion(quaternion)


def canonicalize_quaternion(quaternion: np.ndarray) -> np.ndarray:
    value = np.asarray(quaternion, dtype=np.float64).reshape(4)
    if not np.all(np.isfinite(value)):
        raise ValueError("Quaternion contains nonfinite values")
    norm = float(np.linalg.norm(value))
    if not np.isfinite(norm) or norm <= 0.0:
        raise ValueError("Quaternion norm must be finite and positive")
    value = value / norm
    largest = int(np.argmax(np.abs(value)))
    if value[largest] < 0.0:
        value = -value
    return value


def rotation_matrix_from_quaternion(quaternion: np.ndarray) -> np.ndarray:
    value = canonicalize_quaternion(quaternion)
    matrix = Rotation.from_quat(value).as_matrix()
    return validate_rotation_matrix(matrix, "quaternion rotation matrix")


def target_orientation_from_rpy(
    roll: float,
    pitch: float,
    yaw: float,
) -> Tuple[np.ndarray, np.ndarray]:
    quaternion = quaternion_from_euler_rpy(roll, pitch, yaw)
    return quaternion, rotation_matrix_from_quaternion(quaternion)


def validate_rotation_matrix(matrix: np.ndarray, label: str) -> np.ndarray:
    value = np.asarray(matrix, dtype=np.float64)
    if value.shape != (3, 3):
        raise ValueError(f"{label} must have shape (3,3), got {value.shape}")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{label} contains nonfinite values")
    if not np.allclose(
        value.T @ value,
        np.eye(3, dtype=np.float64),
        rtol=0.0,
        atol=ROTATION_VALIDATION_ATOL,
    ):
        raise ValueError(f"{label} is not orthonormal")
    determinant = float(np.linalg.det(value))
    if not math.isclose(
        determinant,
        1.0,
        rel_tol=0.0,
        abs_tol=ROTATION_VALIDATION_ATOL,
    ):
        raise ValueError(f"{label} determinant must be +1, got {determinant}")
    return value


def quaternion_from_rotation_matrix(matrix: np.ndarray) -> np.ndarray:
    rotation = np.array(
        validate_rotation_matrix(matrix, "FK rotation matrix"),
        dtype=np.float64,
        copy=True,
    )
    return canonicalize_quaternion(Rotation.from_matrix(rotation).as_quat())


def _transform_matrix(robot: Any, ee_link: str) -> np.ndarray:
    try:
        transform = robot.get_transform(frame_to=ee_link)
    except TypeError:
        transform = robot.get_transform(ee_link)
    if hasattr(transform, "matrix"):
        matrix = np.asarray(transform.matrix, dtype=np.float64)
    else:
        matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        raise ValueError(
            f"FK transform for {ee_link} must be a finite 4x4 matrix; "
            f"got {matrix.shape}"
        )
    return matrix


def full_transform_fk(
    robot: Any,
    q: np.ndarray,
    joint_names: Sequence[str],
    ee_link: str,
) -> FullTransform:
    values = np.asarray(q, dtype=np.float64)
    if values.shape != (len(joint_names),):
        raise ValueError(
            f"FK q must have shape ({len(joint_names)},), got {values.shape}"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("FK q contains nonfinite values")
    robot.update_cfg(
        {
            str(joint_name): float(joint_value)
            for joint_name, joint_value in zip(joint_names, values)
        }
    )
    matrix = _transform_matrix(robot, ee_link)
    rotation = validate_rotation_matrix(matrix[:3, :3], "FK rotation matrix")
    position = np.asarray(matrix[:3, 3], dtype=np.float64)
    if position.shape != (3,) or not np.all(np.isfinite(position)):
        raise ValueError("FK position must be a finite shape-(3,) vector")
    quaternion = quaternion_from_rotation_matrix(rotation)
    return FullTransform(position, rotation, quaternion)


def trajectory_full_transform_fk(
    robot: Any,
    q: np.ndarray,
    joint_names: Sequence[str],
    ee_link: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(q, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(joint_names):
        raise ValueError(
            f"Trajectory FK q must have shape (T,{len(joint_names)}), "
            f"got {values.shape}"
        )
    positions = np.empty((values.shape[0], 3), dtype=np.float64)
    rotations = np.empty((values.shape[0], 3, 3), dtype=np.float64)
    quaternions = np.empty((values.shape[0], 4), dtype=np.float64)
    for index, row in enumerate(values):
        transform = full_transform_fk(robot, row, joint_names, ee_link)
        positions[index] = transform.position
        rotations[index] = transform.rotation_matrix
        quaternions[index] = transform.quaternion
    if not (
        np.all(np.isfinite(positions))
        and np.all(np.isfinite(rotations))
        and np.all(np.isfinite(quaternions))
    ):
        raise ValueError("Full-transform trajectory FK produced nonfinite values")
    return positions, rotations, quaternions


def orientation_geodesic_angle(
    target_rotation: np.ndarray,
    actual_rotation: np.ndarray,
) -> float:
    target = validate_rotation_matrix(target_rotation, "target rotation")
    actual = validate_rotation_matrix(actual_rotation, "actual rotation")
    relative = target.T @ actual
    cosine = float((np.trace(relative) - 1.0) / 2.0)
    return float(math.acos(float(np.clip(cosine, -1.0, 1.0))))


def orientation_error_trajectory(
    target_rotation: np.ndarray,
    actual_rotations: np.ndarray,
) -> np.ndarray:
    rotations = np.asarray(actual_rotations, dtype=np.float64)
    if rotations.ndim != 3 or rotations.shape[1:] != (3, 3):
        raise ValueError(
            "actual_rotations must have shape (T,3,3), "
            f"got {rotations.shape}"
        )
    errors = np.asarray(
        [
            orientation_geodesic_angle(target_rotation, rotation)
            for rotation in rotations
        ],
        dtype=np.float64,
    )
    if errors.shape != (rotations.shape[0],) or not np.all(np.isfinite(errors)):
        raise ValueError("Orientation-error trajectory is invalid or nonfinite")
    return errors


def solve_full_pose_ik_from_initial_guess(
    *,
    robot: Any,
    target_position: np.ndarray,
    target_rotation: np.ndarray,
    q_init: np.ndarray,
    q_ref: Optional[np.ndarray],
    joint_names: Sequence[str],
    ee_link: str,
    bounds: Sequence[Tuple[float, float]],
    parameters: PoseSolverParameters,
) -> PoseIkAttempt:
    position = np.asarray(target_position, dtype=np.float64).reshape(3)
    rotation = validate_rotation_matrix(target_rotation, "target rotation")
    if not np.all(np.isfinite(position)):
        raise ValueError("Target position contains nonfinite values")
    q0 = clip_to_bounds(np.asarray(q_init, dtype=np.float64), bounds)
    reference = (
        None
        if q_ref is None
        else clip_to_bounds(np.asarray(q_ref, dtype=np.float64), bounds)
    )

    def objective(q_candidate: np.ndarray) -> float:
        transform = full_transform_fk(
            robot,
            np.asarray(q_candidate, dtype=np.float64),
            joint_names,
            ee_link,
        )
        position_residual = transform.position - position
        relative_rotation = rotation.T @ transform.rotation_matrix
        rotation_vector = Rotation.from_matrix(relative_rotation).as_rotvec()
        cost = float(np.dot(position_residual, position_residual))
        cost += float(
            parameters.orientation_objective_weight
            * np.dot(rotation_vector, rotation_vector)
        )
        if reference is not None and parameters.smooth_weight > 0.0:
            continuity_residual = np.asarray(q_candidate) - reference
            cost += float(
                parameters.smooth_weight
                * np.dot(continuity_residual, continuity_residual)
            )
        return cost

    result = minimize(
        objective,
        q0,
        method="L-BFGS-B",
        bounds=bounds,
        options={
            "maxiter": parameters.max_iters,
            "ftol": parameters.ftol,
        },
    )
    q_solution = clip_to_bounds(
        np.asarray(result.x, dtype=np.float64),
        bounds,
    )
    transform = full_transform_fk(robot, q_solution, joint_names, ee_link)
    return PoseIkAttempt(
        q=q_solution,
        transform=transform,
        position_error_m=float(np.linalg.norm(transform.position - position)),
        orientation_error_rad=orientation_geodesic_angle(
            rotation,
            transform.rotation_matrix,
        ),
        solver_success=bool(result.success),
        nit=int(result.nit),
        message=str(result.message),
    )


def _attempt_record(
    attempt: PoseIkAttempt,
    source: str,
    previous_q: Optional[np.ndarray],
    max_joint_step_gate: float,
    parameters: PoseSolverParameters,
) -> Dict[str, Any]:
    if previous_q is None:
        step_l2 = 0.0
        step_max = 0.0
        step_safe = True
    else:
        delta = attempt.q - previous_q
        step_l2 = float(np.linalg.norm(delta))
        step_max = float(np.max(np.abs(delta)))
        step_safe = step_max <= max_joint_step_gate + 1.0e-12
    finite = bool(
        np.all(np.isfinite(attempt.q))
        and np.all(np.isfinite(attempt.transform.position))
        and np.all(np.isfinite(attempt.transform.rotation_matrix))
        and np.all(np.isfinite(attempt.transform.quaternion))
        and np.isfinite(attempt.position_error_m)
        and np.isfinite(attempt.orientation_error_rad)
    )
    return {
        "source": source,
        "attempt": attempt,
        "finite": finite,
        "step_l2": step_l2,
        "maximum_absolute_joint_step_rad": step_max,
        "step_safe": step_safe,
        "position_safe": bool(
            finite
            and attempt.position_error_m <= parameters.position_tolerance_m
        ),
        "orientation_safe": bool(
            finite
            and attempt.orientation_error_rad
            <= parameters.orientation_tolerance_rad
        ),
    }


def _record_acceptable(record: Mapping[str, Any]) -> bool:
    return bool(
        record["finite"]
        and record["step_safe"]
        and record["position_safe"]
        and record["orientation_safe"]
    )


def _choose_attempt(
    records: Sequence[Mapping[str, Any]],
    previous_q: Optional[np.ndarray],
) -> Optional[Mapping[str, Any]]:
    finite = [record for record in records if bool(record["finite"])]
    if not finite:
        return None
    acceptable = [record for record in finite if _record_acceptable(record)]
    if not acceptable:
        return None
    if previous_q is None:
        return min(
            acceptable,
            key=lambda record: (
                float(record["attempt"].position_error_m),
                float(record["attempt"].orientation_error_rad),
                not bool(record["attempt"].solver_success),
            ),
        )
    return min(
        acceptable,
        key=lambda record: (
            float(record["step_l2"]),
            float(record["attempt"].position_error_m),
            float(record["attempt"].orientation_error_rad),
            not bool(record["attempt"].solver_success),
        ),
    )


def solve_pose_stage(
    *,
    robot: Any,
    desired_path: np.ndarray,
    target_rotation: np.ndarray,
    canonical_mlp_q: np.ndarray,
    q_start: np.ndarray,
    joint_names: Sequence[str],
    ee_link: str,
    bounds: Sequence[Tuple[float, float]],
    parameters: PoseSolverParameters,
    max_joint_step_gate: float,
    num_ik_retries: int,
    random_seed: int,
    retry_profile: str,
) -> PoseStageResult:
    desired = np.asarray(desired_path, dtype=np.float64)
    canonical = np.asarray(canonical_mlp_q, dtype=np.float64)
    if desired.ndim != 2 or desired.shape[1] != 3:
        raise ValueError(f"desired_path must have shape (T,3), got {desired.shape}")
    if canonical.shape != (desired.shape[0], JOINT_DIM):
        raise ValueError(
            f"canonical_mlp_q must have shape ({desired.shape[0]},{JOINT_DIM}), "
            f"got {canonical.shape}"
        )
    target = validate_rotation_matrix(target_rotation, "target rotation")
    length = desired.shape[0]
    q_output = np.empty((length, JOINT_DIM), dtype=np.float64)
    positions = np.empty((length, 3), dtype=np.float64)
    rotations = np.empty((length, 3, 3), dtype=np.float64)
    quaternions = np.empty((length, 4), dtype=np.float64)
    position_errors = np.empty(length, dtype=np.float64)
    orientation_errors = np.empty(length, dtype=np.float64)
    step_records: List[Dict[str, Any]] = []
    accepted_history: List[np.ndarray] = []
    previous_q: Optional[np.ndarray] = None
    rng = np.random.default_rng(random_seed)
    failed_count = 0
    retry_count = 0
    started = time.perf_counter()

    for timestep, target_position in enumerate(desired):
        primary_source = (
            "canonical_mlp_initial"
            if previous_q is None
            else "previous_accepted_refined"
        )
        primary_seed = canonical[timestep] if previous_q is None else previous_q
        records: List[Mapping[str, Any]] = []
        exceptions: List[str] = []

        def attempt_seed(source: str, seed: np.ndarray) -> None:
            nonlocal retry_count
            try:
                attempt = solve_full_pose_ik_from_initial_guess(
                    robot=robot,
                    target_position=target_position,
                    target_rotation=target,
                    q_init=seed,
                    q_ref=previous_q,
                    joint_names=joint_names,
                    ee_link=ee_link,
                    bounds=bounds,
                    parameters=parameters,
                )
                records.append(
                    _attempt_record(
                        attempt,
                        source,
                        previous_q,
                        max_joint_step_gate,
                        parameters,
                    )
                )
            except Exception as exc:
                exceptions.append(f"{source}: {type(exc).__name__}: {exc}")

        attempt_seed(primary_source, primary_seed)
        if not records or not _record_acceptable(records[0]):
            alternatives = retry_seeds(
                previous_q=previous_q,
                previous_previous_q=(
                    accepted_history[-2]
                    if len(accepted_history) >= 2
                    else None
                ),
                mlp_q=canonical[timestep],
                q_start=np.asarray(q_start, dtype=np.float64),
                rng=rng,
                bounds=bounds,
                count=num_ik_retries,
                retry_profile=retry_profile,
            )
            for source, seed in alternatives:
                retry_count += 1
                attempt_seed(source, seed)

        selected = _choose_attempt(records, previous_q)
        unresolved = selected is None
        if unresolved:
            failed_count += 1
            if previous_q is None:
                fallback_q = clip_to_bounds(canonical[timestep], bounds)
            else:
                fallback_q = np.asarray(
                    previous_q,
                    dtype=np.float64,
                ).copy()
            fallback_transform = full_transform_fk(
                robot,
                fallback_q,
                joint_names,
                ee_link,
            )
            selected_attempt = PoseIkAttempt(
                q=fallback_q,
                transform=fallback_transform,
                position_error_m=float(
                    np.linalg.norm(fallback_transform.position - target_position)
                ),
                orientation_error_rad=orientation_geodesic_angle(
                    target,
                    fallback_transform.rotation_matrix,
                ),
                solver_success=False,
                nit=0,
                message="no acceptable full-pose IK attempt",
            )
            selected_source = "retained_unresolved_fallback"
        else:
            selected_attempt = selected["attempt"]
            selected_source = str(selected["source"])

        q_output[timestep] = selected_attempt.q
        positions[timestep] = selected_attempt.transform.position
        rotations[timestep] = selected_attempt.transform.rotation_matrix
        quaternions[timestep] = selected_attempt.transform.quaternion
        position_errors[timestep] = selected_attempt.position_error_m
        orientation_errors[timestep] = selected_attempt.orientation_error_rad
        accepted_q = np.asarray(
            selected_attempt.q,
            dtype=np.float64,
        ).copy()
        previous_q = accepted_q
        accepted_history.append(accepted_q.copy())
        step_records.append(
            {
                "timestep": timestep,
                "selected_source": selected_source,
                "unresolved": unresolved,
                "position_error_m": selected_attempt.position_error_m,
                "orientation_error_rad": selected_attempt.orientation_error_rad,
                "solver_success": selected_attempt.solver_success,
                "maximum_absolute_joint_step_rad": (
                    0.0
                    if timestep == 0
                    else float(
                        np.max(np.abs(q_output[timestep] - q_output[timestep - 1]))
                    )
                ),
                "attempt_count": len(records),
                "solver_exceptions": exceptions,
                "attempts": [
                    {
                        "source": str(record["source"]),
                        "finite": bool(record["finite"]),
                        "step_safe": bool(record["step_safe"]),
                        "position_safe": bool(record["position_safe"]),
                        "orientation_safe": bool(record["orientation_safe"]),
                        "position_error_m": float(
                            record["attempt"].position_error_m
                        ),
                        "orientation_error_rad": float(
                            record["attempt"].orientation_error_rad
                        ),
                        "maximum_absolute_joint_step_rad": float(
                            record["maximum_absolute_joint_step_rad"]
                        ),
                        "solver_success": bool(
                            record["attempt"].solver_success
                        ),
                    }
                    for record in records
                ],
            }
        )

    return PoseStageResult(
        q=q_output,
        positions=positions,
        rotations=rotations,
        quaternions=quaternions,
        position_errors_m=position_errors,
        orientation_errors_rad=orientation_errors,
        step_records=step_records,
        failed_timestep_count=failed_count,
        retry_count=retry_count,
        runtime_seconds=float(time.perf_counter() - started),
        parameters=parameters,
    )


def _evaluate_stage(
    *,
    name: str,
    result: PoseStageResult,
    desired_path: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    joint_names: Sequence[str],
    mean_error_gate: float,
    max_joint_step_gate: float,
    maximum_orientation_error_gate_rad: float,
) -> Dict[str, Any]:
    q = np.asarray(result.q, dtype=np.float64)
    positions = np.asarray(result.positions, dtype=np.float64)
    rotations = np.asarray(result.rotations, dtype=np.float64)
    quaternions = np.asarray(result.quaternions, dtype=np.float64)
    position_errors = np.linalg.norm(positions - desired_path, axis=1)
    orientation_errors = np.asarray(
        result.orientation_errors_rad,
        dtype=np.float64,
    )
    finite = bool(
        np.all(np.isfinite(q))
        and np.all(np.isfinite(positions))
        and np.all(np.isfinite(rotations))
        and np.all(np.isfinite(quaternions))
        and np.all(np.isfinite(position_errors))
        and np.all(np.isfinite(orientation_errors))
    )
    maximum_step = (
        float(np.max(np.abs(np.diff(q, axis=0)))) if len(q) > 1 else 0.0
    )
    limits = check_joint_limits(
        q,
        lower,
        upper,
        joint_names,
        tolerance=HARD_JOINT_LIMIT_TOLERANCE_RAD,
        safety_margin=DEFAULT_JOINT_LIMIT_SAFETY_MARGIN_RAD,
    )
    mean_position_error = float(np.mean(position_errors))
    maximum_position_error = float(np.max(position_errors))
    mean_orientation_error = float(np.mean(orientation_errors))
    maximum_orientation_error = float(np.max(orientation_errors))
    rejection_reasons: List[str] = []
    if not finite:
        rejection_reasons.append("nonfinite_pose_values")
    if result.failed_timestep_count != 0:
        rejection_reasons.append("unresolved_timesteps")
    if mean_position_error > mean_error_gate:
        rejection_reasons.append("mean_cartesian_error_gate")
    if maximum_step > max_joint_step_gate + 1.0e-12:
        rejection_reasons.append("maximum_absolute_joint_step_gate")
    if int(limits["hard_joint_limit_violation_count"]) != 0:
        rejection_reasons.append("hard_joint_limit_violation")
    if maximum_orientation_error > maximum_orientation_error_gate_rad:
        rejection_reasons.append("maximum_orientation_error_gate")
    return {
        "candidate_name": name,
        "valid": not rejection_reasons,
        "rejection_reasons": rejection_reasons,
        "unresolved_timestep_count": int(result.failed_timestep_count),
        "mean_cartesian_error_m": mean_position_error,
        "maximum_cartesian_error_m": maximum_position_error,
        "mean_orientation_error_rad": mean_orientation_error,
        "maximum_orientation_error_rad": maximum_orientation_error,
        "maximum_absolute_joint_step_rad": maximum_step,
        "hard_joint_limit_violation_count": int(
            limits["hard_joint_limit_violation_count"]
        ),
        "retry_count": int(result.retry_count),
        "runtime_seconds": float(result.runtime_seconds),
        "parameters": asdict(result.parameters),
        "_result": result,
    }


def adaptive_refine_full_pose_path(
    *,
    robot: Any,
    desired_path: np.ndarray,
    target_rotation: np.ndarray,
    canonical_mlp_q: np.ndarray,
    q_start: np.ndarray,
    joint_names: Sequence[str],
    ee_link: str,
    bounds: Sequence[Tuple[float, float]],
    lower: np.ndarray,
    upper: np.ndarray,
    mean_error_gate: float,
    max_joint_step_gate: float,
    maximum_orientation_error_gate_rad: float,
    num_ik_retries: int,
    random_seed: int,
    retry_profile: str,
) -> AdaptivePoseResult:
    if (
        not np.isfinite(mean_error_gate)
        or mean_error_gate <= 0.0
        or not np.isfinite(max_joint_step_gate)
        or max_joint_step_gate <= 0.0
        or not np.isfinite(maximum_orientation_error_gate_rad)
        or maximum_orientation_error_gate_rad <= 0.0
    ):
        raise ValueError(
            "Position, joint-step, and orientation gates must be finite "
            "and positive"
        )
    if num_ik_retries < 0:
        raise ValueError("num_ik_retries must be non-negative")
    if retry_profile not in {"standard", "robust"}:
        raise ValueError(
            "retry_profile must be 'standard' or 'robust'"
        )
    stage1_parameters = PoseSolverParameters(
        stage=1,
        smooth_weight=ADAPTIVE_STAGE1_SMOOTH_WEIGHT,
        max_iters=ADAPTIVE_STAGE1_MAX_ITERS,
        ftol=IK_FTOL,
        position_tolerance_m=IK_CARTESIAN_TOLERANCE,
        orientation_tolerance_rad=maximum_orientation_error_gate_rad,
        orientation_objective_weight=ORIENTATION_OBJECTIVE_WEIGHT,
    )
    stage1 = solve_pose_stage(
        robot=robot,
        desired_path=desired_path,
        target_rotation=target_rotation,
        canonical_mlp_q=canonical_mlp_q,
        q_start=q_start,
        joint_names=joint_names,
        ee_link=ee_link,
        bounds=bounds,
        parameters=stage1_parameters,
        max_joint_step_gate=max_joint_step_gate,
        num_ik_retries=num_ik_retries,
        random_seed=random_seed,
        retry_profile="standard",
    )
    results: List[Tuple[str, PoseStageResult]] = [
        ("full_pose_adaptive_stage1", stage1)
    ]
    stage2_triggered = bool(
        stage1.failed_timestep_count
        or float(np.max(stage1.position_errors_m))
        > ADAPTIVE_STAGE2_MAX_ERROR_THRESHOLD
        or float(np.max(stage1.orientation_errors_rad))
        > maximum_orientation_error_gate_rad
    )
    if stage2_triggered:
        stage2_parameters = PoseSolverParameters(
            stage=2,
            smooth_weight=ADAPTIVE_STAGE2_SMOOTH_WEIGHT,
            max_iters=ADAPTIVE_STAGE2_MAX_ITERS,
            ftol=IK_FTOL,
            position_tolerance_m=IK_CARTESIAN_TOLERANCE,
            orientation_tolerance_rad=maximum_orientation_error_gate_rad,
            orientation_objective_weight=ORIENTATION_OBJECTIVE_WEIGHT,
        )
        results.append(
            (
                "full_pose_adaptive_stage2",
                solve_pose_stage(
                    robot=robot,
                    desired_path=desired_path,
                    target_rotation=target_rotation,
                    canonical_mlp_q=canonical_mlp_q,
                    q_start=q_start,
                    joint_names=joint_names,
                    ee_link=ee_link,
                    bounds=bounds,
                    parameters=stage2_parameters,
                    max_joint_step_gate=max_joint_step_gate,
                    num_ik_retries=num_ik_retries,
                    random_seed=random_seed + 1_000_003,
                    retry_profile="standard",
                ),
            )
        )
    if retry_profile == "robust":
        robust_retry_count = min(max(num_ik_retries * 2, 8), 24)
        robust_settings = (
            ("full_pose_robust_continuity", 101, 0.05, 500),
            ("full_pose_robust_long_iteration", 102, 0.01, 750),
            ("full_pose_robust_accuracy", 103, 0.001, 750),
        )
        for offset, (
            name,
            stage_number,
            smooth_weight,
            max_iters,
        ) in enumerate(robust_settings, start=1):
            parameters = PoseSolverParameters(
                stage=stage_number,
                smooth_weight=smooth_weight,
                max_iters=max_iters,
                ftol=IK_FTOL,
                position_tolerance_m=IK_CARTESIAN_TOLERANCE,
                orientation_tolerance_rad=maximum_orientation_error_gate_rad,
                orientation_objective_weight=ORIENTATION_OBJECTIVE_WEIGHT,
            )
            results.append(
                (
                    name,
                    solve_pose_stage(
                        robot=robot,
                        desired_path=desired_path,
                        target_rotation=target_rotation,
                        canonical_mlp_q=canonical_mlp_q,
                        q_start=q_start,
                        joint_names=joint_names,
                        ee_link=ee_link,
                        bounds=bounds,
                        parameters=parameters,
                        max_joint_step_gate=max_joint_step_gate,
                        num_ik_retries=robust_retry_count,
                        random_seed=random_seed + offset * 2_000_003,
                        retry_profile="robust",
                    ),
                )
            )

    evaluated = [
        _evaluate_stage(
            name=name,
            result=result,
            desired_path=np.asarray(desired_path, dtype=np.float64),
            lower=np.asarray(lower, dtype=np.float64),
            upper=np.asarray(upper, dtype=np.float64),
            joint_names=joint_names,
            mean_error_gate=mean_error_gate,
            max_joint_step_gate=max_joint_step_gate,
            maximum_orientation_error_gate_rad=(
                maximum_orientation_error_gate_rad
            ),
        )
        for name, result in results
    ]
    selected = min(
        evaluated,
        key=lambda candidate: (
            not bool(candidate["valid"]),
            float(candidate["mean_cartesian_error_m"]),
            float(candidate["maximum_orientation_error_rad"]),
            float(candidate["maximum_absolute_joint_step_rad"]),
            str(candidate["candidate_name"]),
        ),
    )
    selected_result: PoseStageResult = selected["_result"]
    public_table = tuple(
        {
            key: value
            for key, value in candidate.items()
            if not str(key).startswith("_")
        }
        for candidate in evaluated
    )
    metadata = {
        "stage2_triggered": stage2_triggered,
        "retry_profile": retry_profile,
        "candidate_names": [name for name, _ in results],
        "valid_candidate_count": int(
            sum(bool(candidate["valid"]) for candidate in evaluated)
        ),
        "selected_candidate": str(selected["candidate_name"]),
        "orientation_objective_weight": ORIENTATION_OBJECTIVE_WEIGHT,
        "quaternion_order": QUATERNION_ORDER,
    }
    return AdaptivePoseResult(
        q=selected_result.q,
        positions=selected_result.positions,
        rotations=selected_result.rotations,
        quaternions=selected_result.quaternions,
        position_errors_m=selected_result.position_errors_m,
        orientation_errors_rad=selected_result.orientation_errors_rad,
        unresolved_timestep_count=selected_result.failed_timestep_count,
        selected_candidate_name=str(selected["candidate_name"]),
        candidate_table=public_table,
        adaptive_metadata=metadata,
    )
