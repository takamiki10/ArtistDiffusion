#!/usr/bin/env python3
"""GPU-accelerated entry point for v7 cost-improving target generation.

The original CPU implementation remains unchanged.  This module imports it and
replaces only the kinematics/Jacobian and CEM scoring hot paths with batched
Torch operations.  Robot origins and axes are read from the same loaded URDF,
then checked against the authoritative yourdfpy FK before generation starts.
"""

from __future__ import annotations

import json
import hashlib
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

import generate_diffusion_v7_cost_improving_residual_targets as cpu
from generate_ik_seed_path import find_urdf_joint


ORIGINAL_MAKE_ROBOT_CONTEXT = cpu.make_robot_context
ORIGINAL_FK_ONE = cpu.fk_one
ORIGINAL_PREPARE_OUTPUT_DIRECTORY = cpu.prepare_output_directory
ORIGINAL_ATOMIC_JSON = cpu.atomic_json
LAST_OUTPUT_DIR: Optional[Path] = None
LAST_ACCELERATION_REPORT: Dict[str, Any] = {}
GPU_IMPLEMENTATION_VERSION = "gpu_v2"
GPU_BACKEND_IDENTIFIER = "batched_torch_analytic_urdf_chain"
GPU_EQUIVALENCE_SELF_CHECK = False


class BatchedTorchKinematics:
    """Batched revolute-chain FK and positional Jacobians from URDF data."""

    def __init__(
        self,
        origins: np.ndarray,
        axes: np.ndarray,
        device: torch.device,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        if origins.shape != (6, 4, 4) or axes.shape != (6, 3):
            raise ValueError(
                f"Expected origins (6,4,4) and axes (6,3), got "
                f"{origins.shape} and {axes.shape}"
            )
        self.device = device
        self.dtype = dtype
        self.origins = torch.as_tensor(origins, dtype=dtype, device=device)
        axis_tensor = torch.as_tensor(axes, dtype=dtype, device=device)
        self.axes = axis_tensor / torch.linalg.vector_norm(
            axis_tensor, dim=-1, keepdim=True
        ).clamp_min(torch.finfo(dtype).eps)

    def _axis_angle_transform(
        self, axis: torch.Tensor, angle: torch.Tensor
    ) -> torch.Tensor:
        batch = angle.shape[0]
        x, y, z = axis.unbind()
        zero = torch.zeros((), dtype=self.dtype, device=self.device)
        skew = torch.stack((
            torch.stack((zero, -z, y)),
            torch.stack((z, zero, -x)),
            torch.stack((-y, x, zero)),
        ))
        identity3 = torch.eye(3, dtype=self.dtype, device=self.device)
        outer = axis[:, None] * axis[None, :]
        cosine = torch.cos(angle)[:, None, None]
        sine = torch.sin(angle)[:, None, None]
        rotation = cosine * identity3 + (1.0 - cosine) * outer + sine * skew
        transform = torch.eye(
            4, dtype=self.dtype, device=self.device
        ).expand(batch, 4, 4).clone()
        transform[:, :3, :3] = rotation
        return transform

    @torch.inference_mode()
    def fk_and_jacobian(
        self, q: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        values = torch.as_tensor(q, dtype=self.dtype, device=self.device)
        if values.shape[-1] != 6:
            raise ValueError(f"Expected q[...,6], got {tuple(values.shape)}")
        leading_shape = values.shape[:-1]
        flat = values.reshape(-1, 6)
        batch = flat.shape[0]
        transform = torch.eye(
            4, dtype=self.dtype, device=self.device
        ).expand(batch, 4, 4).clone()
        joint_positions: List[torch.Tensor] = []
        joint_axes: List[torch.Tensor] = []
        for joint_index in range(6):
            transform = transform @ self.origins[joint_index]
            joint_positions.append(transform[:, :3, 3])
            joint_axes.append(
                torch.einsum(
                    "bij,j->bi", transform[:, :3, :3], self.axes[joint_index]
                )
            )
            transform = transform @ self._axis_angle_transform(
                self.axes[joint_index], flat[:, joint_index]
            )
        ee = transform[:, :3, 3]
        positions = torch.stack(joint_positions, dim=1)
        axes = torch.stack(joint_axes, dim=1)
        jacobian = torch.cross(
            axes, ee[:, None, :] - positions, dim=-1
        ).transpose(1, 2)
        return (
            ee.reshape(*leading_shape, 3),
            jacobian.reshape(*leading_shape, 3, 6),
        )

    @torch.inference_mode()
    def numpy_fk_and_jacobian(
        self, q: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        tensor = torch.as_tensor(q, dtype=self.dtype, device=self.device)
        ee, jacobian = self.fk_and_jacobian(tensor)
        return ee.cpu().numpy(), jacobian.cpu().numpy()


def _backend(context: cpu.RobotContext) -> BatchedTorchKinematics:
    value = getattr(context, "gpu_kinematics", None)
    if not isinstance(value, BatchedTorchKinematics):
        raise RuntimeError("GPU kinematics backend is not initialized")
    return value


def _joint_arrays(robot_context: cpu.RobotContext) -> Tuple[np.ndarray, np.ndarray]:
    origins = []
    axes = []
    for name in robot_context.joint_names:
        joint = find_urdf_joint(robot_context.robot, name)
        origin = np.asarray(joint.origin, dtype=np.float64)
        axis = np.asarray(joint.axis, dtype=np.float64).reshape(3)
        if origin.shape != (4, 4) or not np.all(np.isfinite(origin)):
            raise ValueError(f"Invalid URDF origin for {name}")
        if not np.all(np.isfinite(axis)) or np.linalg.norm(axis) <= 0.0:
            raise ValueError(f"Invalid URDF axis for {name}")
        origins.append(origin)
        axes.append(axis)
    return np.stack(origins), np.stack(axes)


def _cache_jacobians(
    context: cpu.RobotContext, q: np.ndarray, jacobians: np.ndarray
) -> None:
    cache: Dict[bytes, np.ndarray] = getattr(
        context, "_gpu_jacobian_cache", {}
    )
    values = np.asarray(q, dtype=np.float64).reshape(-1, 6)
    jacobian_values = np.asarray(jacobians, dtype=np.float64).reshape(-1, 3, 6)
    for row, jacobian in zip(values, jacobian_values):
        cache[row.tobytes()] = jacobian
    if len(cache) > 4096:
        cache = dict(list(cache.items())[-2048:])
    setattr(context, "_gpu_jacobian_cache", cache)


def _reference_positional_jacobian(
    context: cpu.RobotContext,
    q: np.ndarray,
    epsilon: float = 1.0e-5,
) -> np.ndarray:
    """Central finite differences using the unpatched yourdfpy FK reference.

    The CPU Jacobian helper resolves ``cpu.fk_one`` dynamically.  Once this
    wrapper installs its overrides, calling that helper would accidentally
    compare the analytic backend with itself instead of with yourdfpy.
    """
    values = np.asarray(q, dtype=np.float64).reshape(6)
    jacobian = np.empty((3, 6), dtype=np.float64)
    for joint_index in range(6):
        plus = values.copy()
        minus = values.copy()
        plus[joint_index] = min(
            plus[joint_index] + epsilon, context.upper[joint_index]
        )
        minus[joint_index] = max(
            minus[joint_index] - epsilon, context.lower[joint_index]
        )
        denominator = plus[joint_index] - minus[joint_index]
        if denominator <= 0.0:
            jacobian[:, joint_index] = 0.0
        else:
            jacobian[:, joint_index] = (
                ORIGINAL_FK_ONE(context, plus)
                - ORIGINAL_FK_ONE(context, minus)
            ) / denominator
    if not np.all(np.isfinite(jacobian)):
        raise FloatingPointError(
            "Reference finite-difference positional Jacobian is nonfinite"
        )
    return jacobian


def _equivalence_metrics(
    context: cpu.RobotContext, backend: BatchedTorchKinematics
) -> Dict[str, Any]:
    rng = np.random.default_rng(20260717)
    span = context.upper - context.lower
    samples = np.vstack((
        np.zeros(6, dtype=np.float64),
        context.lower + 0.25 * span,
        context.lower + 0.75 * span,
        rng.uniform(context.lower + 0.1 * span, context.upper - 0.1 * span),
    ))
    torch_ee, torch_jacobian = backend.numpy_fk_and_jacobian(samples)
    reference_ee = np.stack([ORIGINAL_FK_ONE(context, row) for row in samples])
    fk_max_error = float(np.max(np.abs(torch_ee - reference_ee)))
    jacobian_errors = []
    for index in range(len(samples)):
        reference_jacobian = _reference_positional_jacobian(
            context, samples[index]
        )
        jacobian_errors.append(
            float(np.max(np.abs(torch_jacobian[index] - reference_jacobian)))
        )
    jacobian_max_error = max(jacobian_errors)
    fk_tolerance = 2.0e-6 if backend.dtype == torch.float32 else 1.0e-9
    jacobian_tolerance = 2.0e-5
    passed = bool(
        fk_max_error <= fk_tolerance
        and jacobian_max_error <= jacobian_tolerance
    )
    return {
        "backend": GPU_BACKEND_IDENTIFIER,
        "implementation_version": GPU_IMPLEMENTATION_VERSION,
        "device": str(backend.device),
        "device_class": backend.device.type,
        "dtype": str(backend.dtype).replace("torch.", ""),
        "fk_reference_max_abs_error_m": fk_max_error,
        "jacobian_reference_max_abs_error": jacobian_max_error,
        "fk_validation_tolerance_m": fk_tolerance,
        "jacobian_validation_tolerance": jacobian_tolerance,
        "equivalence_validation_passed": passed,
        "urdf_derived_origins_and_axes": True,
        "cem_scoring_vectorized": True,
        "trajectory_fk_and_jacobians_batched": True,
        "reference_fk_backend": "yourdfpy",
        "reference_jacobian_backend": (
            "central_finite_difference_using_original_yourdfpy_fk"
        ),
        "cem_penalty_semantics_match_cpu": True,
        "mixed_backend_resume_guard": True,
        "signature_includes_backend_and_dtype": True,
    }


def _assert_equivalence(report: Dict[str, Any]) -> None:
    if float(report["fk_reference_max_abs_error_m"]) > float(
        report["fk_validation_tolerance_m"]
    ):
        raise RuntimeError(
            "Torch FK does not match authoritative yourdfpy FK: "
            f"max_abs_error={report['fk_reference_max_abs_error_m']:.3e}"
        )
    if float(report["jacobian_reference_max_abs_error"]) > float(
        report["jacobian_validation_tolerance"]
    ):
        raise RuntimeError(
            "Torch analytic Jacobian does not match finite differences: "
            f"max_abs_error={report['jacobian_reference_max_abs_error']:.3e}"
        )


def _validate_backend(
    context: cpu.RobotContext, backend: BatchedTorchKinematics
) -> Dict[str, Any]:
    report = _equivalence_metrics(context, backend)
    _assert_equivalence(report)
    return report


def _run_requested_self_check(
    context: cpu.RobotContext, backend: BatchedTorchKinematics
) -> None:
    report = _equivalence_metrics(context, backend)
    self_check = {
        "maximum_fk_absolute_error": report["fk_reference_max_abs_error_m"],
        "maximum_jacobian_absolute_error": report[
            "jacobian_reference_max_abs_error"
        ],
        "dtype": report["dtype"],
        "device": report["device"],
        "pass": report["equivalence_validation_passed"],
    }
    print(json.dumps(self_check, indent=2, sort_keys=True))
    _assert_equivalence(report)


def _gpu_dtype_name() -> str:
    dtype_name = os.environ.get("ARTISTDIFFUSION_GPU_DTYPE", "float32").strip().lower()
    if dtype_name not in {"float32", "float64"}:
        raise ValueError(
            "ARTISTDIFFUSION_GPU_DTYPE must be float32 or float64"
        )
    return dtype_name


def _runtime_descriptor(args: Any) -> Dict[str, str]:
    resolved = cpu.resolve_device(args.device)
    return {
        "backend": GPU_BACKEND_IDENTIFIER,
        "implementation_version": GPU_IMPLEMENTATION_VERSION,
        "requested_device": str(args.device),
        "resolved_device": str(resolved),
        "device_class": resolved.type,
        "dtype": _gpu_dtype_name(),
    }


def gpu_generation_signature(
    args: Any, selected_paths: Sequence[str]
) -> str:
    relevant = {
        key: cpu.json_safe(value)
        for key, value in vars(args).items()
        if key not in {"resume", "overwrite", "save_all_candidates"}
    }
    relevant["gpu_acceleration"] = _runtime_descriptor(args)
    relevant["selected_paths"] = list(selected_paths)
    encoded = json.dumps(
        relevant, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def gpu_prepare_output_directory(args: Any) -> None:
    """Reject mixed CPU/GPU state before the CPU pipeline can load it."""
    if args.resume:
        configuration_path = Path(args.output_dir) / "pilot_configuration.json"
        if configuration_path.is_file():
            with configuration_path.open("r", encoding="utf-8") as handle:
                existing = json.load(handle)
            acceleration = existing.get("gpu_acceleration")
            if not isinstance(acceleration, dict):
                raise ValueError(
                    "Cannot resume GPU generation from this output directory: "
                    "pilot_configuration.json has no gpu_acceleration metadata. "
                    "CPU and GPU window states must not be mixed."
                )
            expected = _runtime_descriptor(args)
            mismatches = {
                key: {"existing": acceleration.get(key), "requested": expected[key]}
                for key in ("backend", "implementation_version", "dtype", "device_class")
                if acceleration.get(key) != expected[key]
            }
            if mismatches:
                raise ValueError(
                    "Cannot resume GPU generation with a different backend, "
                    "implementation, device class, or dtype: "
                    + json.dumps(mismatches, sort_keys=True)
                )
    ORIGINAL_PREPARE_OUTPUT_DIRECTORY(args)


def gpu_atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = dict(value)
    if (
        path.name in {"pilot_configuration.json", "pilot_summary.json"}
        and LAST_ACCELERATION_REPORT
    ):
        payload["gpu_acceleration"] = dict(LAST_ACCELERATION_REPORT)
    ORIGINAL_ATOMIC_JSON(path, payload)


def gpu_make_robot_context(args: Any) -> cpu.RobotContext:
    global LAST_OUTPUT_DIR, LAST_ACCELERATION_REPORT
    context = ORIGINAL_MAKE_ROBOT_CONTEXT(args)
    requested = cpu.resolve_device(args.device)
    origins, axes = _joint_arrays(context)
    dtype_name = _gpu_dtype_name()
    dtype = torch.float32 if dtype_name == "float32" else torch.float64
    backend = BatchedTorchKinematics(
        origins, axes, requested, dtype=dtype
    )
    # Attach first: validation calls the immutable yourdfpy reference, but any
    # monkey-patched helper reached during startup must still see a valid backend.
    setattr(context, "gpu_kinematics", backend)
    setattr(context, "_gpu_jacobian_cache", {})
    report = _validate_backend(context, backend)
    report.update(_runtime_descriptor(args))
    if GPU_EQUIVALENCE_SELF_CHECK:
        _run_requested_self_check(context, backend)
    LAST_OUTPUT_DIR = Path(args.output_dir)
    LAST_ACCELERATION_REPORT = report
    print(
        "GPU kinematics backend: "
        f"device={report['device']} dtype={report['dtype']} "
        f"FK_error={report['fk_reference_max_abs_error_m']:.3e} "
        f"Jacobian_error={report['jacobian_reference_max_abs_error']:.3e}"
    )
    return context


def gpu_fk_one(context: cpu.RobotContext, q: np.ndarray) -> np.ndarray:
    backend = _backend(context)
    ee, jacobian = backend.numpy_fk_and_jacobian(
        np.asarray(q, dtype=np.float64).reshape(1, 6)
    )
    _cache_jacobians(context, q, jacobian)
    return ee[0]


def gpu_fk_trajectory(context: cpu.RobotContext, q: np.ndarray) -> np.ndarray:
    backend = _backend(context)
    values = np.asarray(q, dtype=np.float64)
    ee, jacobian = backend.numpy_fk_and_jacobian(values)
    _cache_jacobians(context, values, jacobian)
    return ee


def gpu_positional_jacobian(
    context: cpu.RobotContext, q: np.ndarray, epsilon: float = 1.0e-5
) -> np.ndarray:
    del epsilon
    values = np.asarray(q, dtype=np.float64).reshape(6)
    cache: Dict[bytes, np.ndarray] = getattr(context, "_gpu_jacobian_cache", {})
    cached = cache.get(values.tobytes())
    if cached is not None:
        return np.asarray(cached, dtype=np.float64)
    backend = _backend(context)
    _, jacobian = backend.numpy_fk_and_jacobian(values.reshape(1, 6))
    _cache_jacobians(context, values, jacobian)
    return jacobian[0]


def _interpolate_control_points_batch(
    controls: np.ndarray, horizon: int
) -> np.ndarray:
    values = np.asarray(controls, dtype=np.float64)
    source = np.linspace(0.0, 1.0, values.shape[1])
    target = np.linspace(0.0, 1.0, horizon)
    interpolator = cpu.PchipInterpolator(source, values, axis=1)
    return np.asarray(interpolator(target), dtype=np.float64)


def _boundary_blend_batch(
    residuals: np.ndarray, execution_horizon: int
) -> np.ndarray:
    result = np.asarray(residuals, dtype=np.float64).copy()
    horizon = result.shape[1]
    start = np.linspace(0.0, 1.0, min(6, horizon))
    result[:, :len(start)] *= start[None, :, None]
    tail_start = max(execution_horizon - 3, 0)
    tail_end = min(execution_horizon + 4, horizon)
    if tail_end > tail_start:
        distances = np.abs(
            np.arange(tail_start, tail_end) - execution_horizon
        )
        weights = np.minimum(distances / 3.0, 1.0)
        result[:, tail_start:tail_end] *= weights[None, :, None]
    return result


def _torch_derivative_cost(values: torch.Tensor, order: int) -> torch.Tensor:
    differences = values
    for _ in range(order):
        differences = torch.diff(differences, dim=1)
    if differences.shape[1] == 0:
        return torch.zeros(
            values.shape[0], dtype=values.dtype, device=values.device
        )
    return torch.mean(torch.sum(differences.square(), dim=-1), dim=1)


@torch.inference_mode()
def gpu_quick_cem_objective_batch(
    robot: cpu.RobotContext,
    window: cpu.WindowContext,
    residuals: np.ndarray,
    args: Any,
) -> np.ndarray:
    backend = _backend(robot)
    residual_policy = torch.as_tensor(
        residuals, dtype=torch.float64, device=backend.device
    )
    prior_policy = torch.as_tensor(
        window.prior_q, dtype=torch.float64, device=backend.device
    )
    q_policy = prior_policy.unsqueeze(0) + residual_policy
    q = q_policy.to(dtype=backend.dtype)
    batch = q_policy.shape[0]
    lower = torch.as_tensor(
        robot.lower, dtype=torch.float64, device=backend.device
    )
    upper = torch.as_tensor(
        robot.upper, dtype=torch.float64, device=backend.device
    )
    hard_element_mask = (
        (q_policy < lower - cpu.HARD_JOINT_LIMIT_TOLERANCE_RAD)
        | (q_policy > upper + cpu.HARD_JOINT_LIMIT_TOLERANCE_RAD)
    )
    hard_invalid = torch.any(hard_element_mask, dim=(1, 2))
    raw_hard_magnitude = (
        torch.relu(lower - q_policy) + torch.relu(q_policy - upper)
    )
    hard_violation_magnitude = torch.sum(
        torch.where(
            hard_element_mask,
            raw_hard_magnitude,
            torch.zeros_like(raw_hard_magnitude),
        ),
        dim=(1, 2),
    )
    internal_max = torch.amax(
        torch.abs(torch.diff(q_policy, dim=1)), dim=(1, 2)
    )
    previous_steps: List[torch.Tensor] = []
    if window.previous_q is not None:
        previous = torch.as_tensor(
            window.previous_q, dtype=torch.float64, device=backend.device
        )
        previous_steps.append(q_policy[:, 0] - previous)
    tail = torch.as_tensor(
        window.tail_q, dtype=torch.float64, device=backend.device
    )
    exit_step = tail - q_policy[:, args.execution_horizon - 1]
    previous_steps.append(exit_step)
    boundary_max = torch.stack([
        torch.amax(torch.abs(step), dim=1) for step in previous_steps
    ], dim=1).amax(dim=1)
    # Match cpu.quick_cem_objective exactly: only internal np.diff(q) steps
    # trigger its quick rejection. Boundary steps remain part of the objective.
    step_invalid = internal_max > args.max_joint_step_gate

    prefix = q_policy[:, :args.execution_horizon]
    ee, _ = backend.fk_and_jacobian(
        q[:, :args.execution_horizon]
    )
    desired = torch.as_tensor(
        window.desired[:args.execution_horizon],
        dtype=backend.dtype,
        device=backend.device,
    )
    errors = torch.linalg.vector_norm(ee - desired.unsqueeze(0), dim=-1)
    cart_mean = torch.mean(errors, dim=1)
    cart_p95 = torch.quantile(errors, 0.95, dim=1)
    cart_max = torch.amax(errors, dim=1)
    acceleration = _torch_derivative_cost(prefix, 2)
    jerk = _torch_derivative_cost(prefix, 3)

    acceleration_vectors: List[torch.Tensor] = []
    if window.previous_q is not None and window.previous_previous_q is not None:
        previous = torch.as_tensor(
            window.previous_q, dtype=torch.float64, device=backend.device
        )
        previous_previous = torch.as_tensor(
            window.previous_previous_q,
            dtype=torch.float64,
            device=backend.device,
        )
        acceleration_vectors.append(
            (q_policy[:, 0] - previous) - (previous - previous_previous)
        )
    prefix_velocity = prefix[:, -1] - prefix[:, -2]
    acceleration_vectors.append(exit_step - prefix_velocity)
    boundary_acceleration = torch.stack([
        torch.linalg.vector_norm(item, dim=1)
        for item in acceleration_vectors
    ], dim=1).amax(dim=1)

    score = (
        4.0 * cart_mean
        + 2.0 * cart_p95
        + cart_max
        + 0.5 * acceleration
        + 0.25 * jerk
        + boundary_max
        + 0.5 * boundary_acceleration
    )
    result = torch.where(step_invalid, 1.0e9 + internal_max, score)
    # Hard-limit invalidity takes precedence over an internal-step violation.
    result = torch.where(
        hard_invalid, 1.0e10 + hard_violation_magnitude, result
    )
    nonfinite = (
        ~torch.all(torch.isfinite(q_policy), dim=(1, 2))
        | ~torch.isfinite(result)
    )
    result = torch.where(
        nonfinite,
        torch.full(
            (batch,), 1.0e12, dtype=result.dtype, device=backend.device
        ),
        result,
    )
    return result.cpu().numpy()


def gpu_generate_cem_candidates(
    robot: cpu.RobotContext, window: cpu.WindowContext, args: Any
) -> List[cpu.Candidate]:
    candidates: List[cpu.Candidate] = []
    shape = (args.cem_control_points, 6)
    for restart in range(args.cem_restarts):
        started = time.perf_counter()
        seed = cpu.stable_seed(
            args.seed, window.path_name, window.window_start, "cem", restart
        )
        rng = np.random.default_rng(seed)
        mean = np.zeros(shape, dtype=np.float64)
        std = np.full(shape, args.cem_initial_std, dtype=np.float64)
        best: List[Tuple[float, np.ndarray]] = []
        for _ in range(args.cem_iterations):
            controls = rng.normal(
                mean, std, size=(args.cem_candidates, *shape)
            )
            controls = np.clip(
                controls, -args.cem_max_residual, args.cem_max_residual
            )
            residuals = _interpolate_control_points_batch(
                controls, args.horizon
            )
            residuals = _boundary_blend_batch(
                residuals, args.execution_horizon
            )
            residuals = np.clip(
                residuals, -args.cem_max_residual, args.cem_max_residual
            )
            scores = gpu_quick_cem_objective_batch(
                robot, window, residuals, args
            )
            elite_indices = np.argsort(scores, kind="stable")[:args.cem_elites]
            elite_controls = controls[elite_indices]
            mean = 0.25 * mean + 0.75 * np.mean(elite_controls, axis=0)
            std = np.maximum(
                0.25 * std + 0.75 * np.std(elite_controls, axis=0),
                1.0e-4,
            )
            best.extend(
                (float(scores[index]), residuals[index].copy())
                for index in elite_indices[:4]
            )
            best = sorted(best, key=lambda item: item[0])[:8]
        mean_residual = _interpolate_control_points_batch(
            mean[None, ...], args.horizon
        )[0]
        final_residuals = [mean_residual, *[item[1] for item in best[:3]]]
        final_batch = _boundary_blend_batch(
            np.stack(final_residuals), args.execution_horizon
        )
        final_batch = np.clip(
            final_batch, -args.cem_max_residual, args.cem_max_residual
        )
        elapsed = time.perf_counter() - started
        for final_index, residual in enumerate(final_batch):
            candidates.append(cpu.Candidate(
                method="spline_cem",
                subtype=f"restart_{restart}_final_{final_index}_gpu",
                residual=residual,
                deterministic_seed=seed,
                metadata={
                    "cem_restart": restart,
                    "cem_final_index": final_index,
                    "control_points": args.cem_control_points,
                    "candidates_per_iteration": args.cem_candidates,
                    "elites": args.cem_elites,
                    "iterations": args.cem_iterations,
                    "initial_std": args.cem_initial_std,
                    "maximum_residual_amplitude": args.cem_max_residual,
                    "scoring_backend": "batched_torch",
                    "scoring_device": str(_backend(robot).device),
                },
                runtime_seconds=elapsed / len(final_batch),
            ))
    return candidates


def install_gpu_overrides() -> None:
    cpu.make_robot_context = gpu_make_robot_context
    cpu.prepare_output_directory = gpu_prepare_output_directory
    cpu.generation_signature = gpu_generation_signature
    cpu.atomic_json = gpu_atomic_json
    cpu.fk_one = gpu_fk_one
    cpu.fk_trajectory = gpu_fk_trajectory
    cpu.positional_jacobian = gpu_positional_jacobian
    cpu.generate_cem_candidates = gpu_generate_cem_candidates


def _enrich_output_metadata() -> None:
    if LAST_OUTPUT_DIR is None or not LAST_ACCELERATION_REPORT:
        return
    for filename in ("pilot_configuration.json", "pilot_summary.json"):
        path = LAST_OUTPUT_DIR / filename
        if not path.is_file():
            continue
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        payload["gpu_acceleration"] = LAST_ACCELERATION_REPORT
        cpu.atomic_json(path, payload)


def _consume_self_check_flag() -> None:
    global GPU_EQUIVALENCE_SELF_CHECK
    flag = "--gpu_equivalence_self_check"
    requested = flag in sys.argv[1:]
    if requested:
        sys.argv[:] = [argument for argument in sys.argv if argument != flag]
    GPU_EQUIVALENCE_SELF_CHECK = requested


def main() -> int:
    _consume_self_check_flag()
    install_gpu_overrides()
    result = cpu.main()
    _enrich_output_metadata()
    return result


if __name__ == "__main__":
    raise SystemExit(main())
