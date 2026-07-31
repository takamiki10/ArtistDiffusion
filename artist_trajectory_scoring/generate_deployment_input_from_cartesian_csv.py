#!/usr/bin/env python3
"""Build a validated diffusion-v8.1 deployment input from one Cartesian CSV.

The strong prior is generated without expert/test joint labels:

    canonical path-conditioned MLP -> adaptive sequential IK -> safety gates

The input CSV time coordinate is treated as a normalized trajectory parameter.
It is normalized to [0, 1] for the canonical MLP and mapped independently to
[0, trajectory_duration_seconds] for deployment timestamps.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
import traceback
from pathlib import Path
from typing import Any, Dict, Mapping, NoReturn, Tuple

import matplotlib

# Project imports can transitively import pyplot. Select the backend first.
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import torch

from generate_adaptive_mlp_ik_bootstrap_prior import (
    EXPECTED_STEPS,
    JOINT_DIM,
    SEQUENTIAL_IK_DEFAULT_RETRIES,
    canonical_mlp_full_q,
    resolve_device,
)
from generate_ik_seed_path import (
    DEFAULT_EE_LINK,
    DEFAULT_JOINT_NAMES,
    DEFAULT_URDF_PATH,
    HARD_JOINT_LIMIT_TOLERANCE_RAD,
    check_joint_limits,
    get_joint_bounds,
    load_robot,
)
from predict_path_conditioned_mlp import PathConditionedMLP, load_model
from refine_mlp_predictions_with_ik import resolve_urdf_path
from orientation_aware_adaptive_ik import (  # pyright: ignore[reportMissingImports]
    QUATERNION_ORDER,
    adaptive_refine_full_pose_path,
    orientation_error_trajectory,
    target_orientation_from_rpy,
    trajectory_full_transform_fk,
)


DEFAULT_MLP_CHECKPOINT = Path(
    "data/cartesian_expert_dataset_v3/path_conditioned_mlp_v3.pt"
)
SOURCE_METHOD = "canonical_path_conditioned_mlp_plus_adaptive_sequential_ik"
MAXIMUM_ALLOWED_MEAN_ERROR_GATE_M = 0.01
MAXIMUM_ALLOWED_JOINT_STEP_GATE_RAD = 0.20
MAXIMUM_ALLOWED_ORIENTATION_ERROR_GATE_RAD = 0.05
MAXIMUM_ALLOWED_Z_ERROR_GATE_M = 0.001
CSV_COLUMNS = ("t", "x", "y", "z")
FIXED_RPY_ORDER = ("roll", "pitch", "yaw")
RANDOM_SEED = 42


class GenerationFailure(RuntimeError):
    """A generation failure with JSON-serializable diagnostic context."""

    def __init__(self, message: str, diagnostics: Mapping[str, Any]) -> None:
        super().__init__(message)
        self.diagnostics = dict(diagnostics)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_csv", type=Path, required=True)
    parser.add_argument("--output_npz", type=Path, required=True)
    parser.add_argument("--path_name", required=True)
    parser.add_argument(
        "--mlp_checkpoint",
        type=Path,
        default=DEFAULT_MLP_CHECKPOINT,
    )
    parser.add_argument(
        "--urdf_path",
        type=Path,
        default=Path(DEFAULT_URDF_PATH),
    )
    parser.add_argument("--roll", type=float, default=math.pi)
    parser.add_argument("--pitch", type=float, default=0.0)
    parser.add_argument("--yaw", type=float, default=0.0)
    parser.add_argument(
        "--trajectory_duration_seconds",
        type=float,
        default=10.0,
    )
    parser.add_argument(
        "--mean_error_gate",
        type=float,
        default=MAXIMUM_ALLOWED_MEAN_ERROR_GATE_M,
    )
    parser.add_argument(
        "--max_joint_step_gate",
        type=float,
        default=MAXIMUM_ALLOWED_JOINT_STEP_GATE_RAD,
    )
    parser.add_argument(
        "--maximum_orientation_error_gate_rad",
        type=float,
        default=MAXIMUM_ALLOWED_ORIENTATION_ERROR_GATE_RAD,
    )
    parser.add_argument(
        "--maximum_z_error_gate_m",
        type=float,
        default=MAXIMUM_ALLOWED_Z_ERROR_GATE_M,
    )
    parser.add_argument(
        "--retry_profile",
        choices=("standard", "robust"),
        default="robust",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_existing_file(path: Path, label: str) -> Path:
    candidate = path.expanduser()
    if not candidate.is_file():
        script_relative = Path(__file__).resolve().parent / candidate
        if script_relative.is_file():
            candidate = script_relative
    if not candidate.is_file():
        raise FileNotFoundError(f"{label} does not exist or is not a file: {path}")
    return candidate.resolve(strict=True)


def validate_args(args: argparse.Namespace) -> None:
    if not str(args.path_name).strip():
        raise ValueError("--path_name must contain non-whitespace text")
    if args.output_npz.suffix.lower() != ".npz":
        raise ValueError("--output_npz must have a .npz suffix")
    finite_cli_values = {
        "roll": args.roll,
        "pitch": args.pitch,
        "yaw": args.yaw,
        "trajectory_duration_seconds": args.trajectory_duration_seconds,
        "mean_error_gate": args.mean_error_gate,
        "max_joint_step_gate": args.max_joint_step_gate,
        "maximum_orientation_error_gate_rad": (
            args.maximum_orientation_error_gate_rad
        ),
        "maximum_z_error_gate_m": args.maximum_z_error_gate_m,
    }
    nonfinite = [
        name for name, value in finite_cli_values.items() if not np.isfinite(value)
    ]
    if nonfinite:
        raise ValueError(f"CLI values must be finite; invalid: {nonfinite}")
    if args.trajectory_duration_seconds <= 0.0:
        raise ValueError("--trajectory_duration_seconds must be positive")
    if not 0.0 < args.mean_error_gate <= MAXIMUM_ALLOWED_MEAN_ERROR_GATE_M:
        raise ValueError(
            "--mean_error_gate must be positive and may not weaken the "
            f"established {MAXIMUM_ALLOWED_MEAN_ERROR_GATE_M:.6f} m gate"
        )
    if not 0.0 < args.max_joint_step_gate <= MAXIMUM_ALLOWED_JOINT_STEP_GATE_RAD:
        raise ValueError(
            "--max_joint_step_gate must be positive and may not weaken the "
            f"established {MAXIMUM_ALLOWED_JOINT_STEP_GATE_RAD:.6f} rad gate"
        )
    if not (
        0.0
        < args.maximum_orientation_error_gate_rad
        <= MAXIMUM_ALLOWED_ORIENTATION_ERROR_GATE_RAD
    ):
        raise ValueError(
            "--maximum_orientation_error_gate_rad must be positive and may "
            "not weaken the deployment orientation gate beyond "
            f"{MAXIMUM_ALLOWED_ORIENTATION_ERROR_GATE_RAD:.6f} rad"
        )
    if not (
        0.0
        < args.maximum_z_error_gate_m
        <= MAXIMUM_ALLOWED_Z_ERROR_GATE_M
    ):
        raise ValueError(
            "--maximum_z_error_gate_m must be positive and may not weaken "
            "the fixed-Z deployment gate beyond "
            f"{MAXIMUM_ALLOWED_Z_ERROR_GATE_M:.6f} m"
        )
    if args.output_npz.exists() and not args.overwrite:
        raise FileExistsError(
            f"{args.output_npz} already exists; pass --overwrite to replace it"
        )


def read_cartesian_csv(
    input_csv: Path,
    trajectory_duration_seconds: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    try:
        frame = pd.read_csv(input_csv)
    except Exception as exc:
        raise ValueError(f"Could not parse Cartesian CSV {input_csv}: {exc}") from exc
    actual_columns = tuple(str(column) for column in frame.columns)
    if actual_columns != CSV_COLUMNS:
        raise ValueError(
            f"{input_csv} columns must be exactly {list(CSV_COLUMNS)} in that "
            f"order; found {list(actual_columns)}"
        )
    if len(frame) != EXPECTED_STEPS:
        raise ValueError(
            f"{input_csv} must contain exactly {EXPECTED_STEPS} data rows; "
            f"found {len(frame)}"
        )
    try:
        values = frame.loc[:, list(CSV_COLUMNS)].to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{input_csv} columns t,x,y,z must contain only numeric values"
        ) from exc
    if values.shape != (EXPECTED_STEPS, 4):
        raise ValueError(f"Unexpected CSV numeric array shape: {values.shape}")
    if not np.all(np.isfinite(values)):
        locations = np.argwhere(~np.isfinite(values))
        preview = [
            {"row": int(row), "column": CSV_COLUMNS[int(column)]}
            for row, column in locations[:10]
        ]
        raise ValueError(f"{input_csv} contains nonfinite values at {preview}")

    source_t = values[:, 0]
    if not np.all(np.diff(source_t) > 0.0):
        raise ValueError(f"{input_csv} t values must be strictly increasing")
    source_span = float(source_t[-1] - source_t[0])
    if not np.isfinite(source_span) or source_span <= 0.0:
        raise ValueError(f"{input_csv} t range must be finite and positive")

    normalized_t = (source_t - source_t[0]) / source_span
    normalized_t[0] = 0.0
    normalized_t[-1] = 1.0
    timestamps = normalized_t * float(trajectory_duration_seconds)
    desired_path = values[:, 1:4]
    if desired_path.shape != (EXPECTED_STEPS, 3):
        raise ValueError(
            f"desired_path must have shape ({EXPECTED_STEPS},3), "
            f"got {desired_path.shape}"
        )
    if not np.all(np.diff(timestamps) > 0.0):
        raise ValueError("Mapped deployment timestamps are not strictly increasing")
    if not np.allclose(
        desired_path[:, 2],
        desired_path[0, 2],
        rtol=0.0,
        atol=1.0e-12,
    ):
        raise ValueError(
            f"{input_csv} z must be constant across all {EXPECTED_STEPS} samples"
        )
    return (
        desired_path.astype(np.float64, copy=False),
        normalized_t.astype(np.float64, copy=False),
        timestamps.astype(np.float64, copy=False),
    )


def normalization_array(
    checkpoint: Mapping[str, Any],
    key: str,
    expected_size: int,
    *,
    positive: bool,
) -> np.ndarray:
    if key not in checkpoint:
        raise KeyError(f"MLP checkpoint is missing required normalization array {key!r}")
    value = np.asarray(checkpoint[key], dtype=np.float32)
    if value.size != expected_size or value.ndim not in (1, 2):
        raise ValueError(
            f"Checkpoint {key} must contain {expected_size} values in a 1-D or "
            f"2-D array; got shape {value.shape}"
        )
    if not np.all(np.isfinite(value)):
        raise ValueError(f"Checkpoint {key} contains nonfinite values")
    if positive and np.any(value <= 0.0):
        raise ValueError(f"Checkpoint {key} must be strictly positive")
    return value


def validate_checkpoint_contract(
    checkpoint: Mapping[str, Any],
    model: PathConditionedMLP,
) -> None:
    required_scalars = ("input_dim", "hidden_dim", "num_layers", "num_steps")
    missing = [key for key in required_scalars if key not in checkpoint]
    if missing:
        raise KeyError(f"MLP checkpoint is missing architecture keys: {missing}")
    if int(checkpoint["num_steps"]) != EXPECTED_STEPS:
        raise ValueError(
            f"MLP checkpoint num_steps must be {EXPECTED_STEPS}, "
            f"got {checkpoint['num_steps']}"
        )
    if int(checkpoint.get("output_dim", JOINT_DIM)) != JOINT_DIM:
        raise ValueError(
            f"MLP checkpoint output_dim must be {JOINT_DIM}, "
            f"got {checkpoint.get('output_dim')}"
        )
    include_current_point = bool(checkpoint.get("include_current_point", True))
    expected_input_dim = EXPECTED_STEPS * 3 + 1 + (
        3 if include_current_point else 0
    )
    input_dim = int(checkpoint["input_dim"])
    if input_dim != expected_input_dim:
        raise ValueError(
            "Checkpoint input_dim does not match the canonical feature layout "
            f"[desired_path_flat,t{',x_t,y_t,z_t' if include_current_point else ''}]: "
            f"expected {expected_input_dim}, got {input_dim}"
        )
    if not isinstance(model, PathConditionedMLP):
        raise TypeError(
            "Canonical checkpoint loader did not return PathConditionedMLP"
        )
    if str(checkpoint.get("model_type", "path_conditioned_mlp")) != (
        "path_conditioned_mlp"
    ):
        raise ValueError(
            "Checkpoint model_type must be path_conditioned_mlp when present"
        )
    normalization_array(
        checkpoint, "x_mean", input_dim, positive=False
    )
    normalization_array(
        checkpoint, "x_std", input_dim, positive=True
    )
    normalization_array(
        checkpoint, "y_mean", JOINT_DIM, positive=False
    )
    normalization_array(
        checkpoint, "y_std", JOINT_DIM, positive=True
    )

    expected_joint_order = tuple(DEFAULT_JOINT_NAMES)
    for key in ("joint_names", "joint_order", "joint_ordering"):
        if key not in checkpoint:
            continue
        actual = tuple(
            item.decode("utf-8") if isinstance(item, bytes) else str(item)
            for item in np.asarray(checkpoint[key]).reshape(-1)
        )
        if actual != expected_joint_order:
            raise ValueError(
                f"Checkpoint {key} must be {expected_joint_order}, got {actual}"
            )


def finite_array(name: str, value: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains nonfinite values")
    return array


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
        if not math.isfinite(value):
            return str(value)
        return value
    if value is None or isinstance(value, (str, int, bool)):
        return value
    return str(value)


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            json.dump(
                json_safe(payload),
                handle,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def atomic_save_npz(
    path: Path,
    arrays: Mapping[str, Any],
    *,
    overwrite: bool,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".npz",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
        np.savez_compressed(temporary_path, **arrays)
        with temporary_path.open("rb+") as handle:
            os.fsync(handle.fileno())
        if overwrite:
            os.replace(temporary_path, path)
        else:
            # A same-directory hard link makes the no-clobber publication
            # atomic: it either creates path or raises FileExistsError.
            os.link(temporary_path, path)
            temporary_path.unlink()
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def failure_json_path(output_npz: Path) -> Path:
    return output_npz.with_name(f"{output_npz.stem}_failure.json")


def fail_generation(message: str, diagnostics: Mapping[str, Any]) -> NoReturn:
    raise GenerationFailure(message, diagnostics)


def generate(args: argparse.Namespace) -> Dict[str, Any]:
    validate_args(args)
    input_csv = resolve_existing_file(args.input_csv, "Input CSV")
    checkpoint_path = resolve_existing_file(
        args.mlp_checkpoint, "MLP checkpoint"
    )
    resolved_urdf_candidate = resolve_urdf_path(args.urdf_path.expanduser())
    urdf_path = resolve_existing_file(resolved_urdf_candidate, "Robot URDF")

    desired_path, normalized_t, timestamps = read_cartesian_csv(
        input_csv,
        args.trajectory_duration_seconds,
    )
    target_z = float(desired_path[0, 2])
    if not np.isfinite(target_z):
        raise ValueError("target_z derived from the CSV must be finite")
    fixed_rpy = finite_array(
        "fixed_rpy",
        np.asarray([args.roll, args.pitch, args.yaw], dtype=np.float64),
        (3,),
    )
    target_quaternion, target_rotation_matrix = target_orientation_from_rpy(
        float(fixed_rpy[0]),
        float(fixed_rpy[1]),
        float(fixed_rpy[2]),
    )
    target_quaternion = finite_array(
        "target_quaternion",
        target_quaternion,
        (4,),
    )
    target_rotation_matrix = finite_array(
        "target_rotation_matrix",
        target_rotation_matrix,
        (3, 3),
    )

    device = resolve_device(args.device)
    model, checkpoint = load_model(checkpoint_path, device)
    validate_checkpoint_contract(checkpoint, model)

    # canonical_mlp_full_q delegates feature construction, x normalization,
    # model decoding, and y_mean/y_std output denormalization to the validated
    # canonical predictor. q_start is unused for a canonical full-q checkpoint.
    canonical_q, interpretation = canonical_mlp_full_q(
        model=model,
        checkpoint=checkpoint,
        checkpoint_path=checkpoint_path,
        times=normalized_t.astype(np.float32),
        desired_path=desired_path.astype(np.float32),
        q_start=np.zeros(JOINT_DIM, dtype=np.float64),
        device=device,
    )
    if interpretation["checkpoint_target_interpretation"] != "full_q":
        raise ValueError(
            "Deployment prior requires canonical full-q MLP output; checkpoint "
            f"declares {interpretation['checkpoint_target_interpretation']!r}"
        )
    canonical_q = finite_array(
        "canonical_mlp_q",
        canonical_q,
        (EXPECTED_STEPS, JOINT_DIM),
    )

    joint_names = tuple(DEFAULT_JOINT_NAMES)
    expected_joint_names = tuple(f"joint{index}" for index in range(1, 7))
    if joint_names != expected_joint_names:
        raise ValueError(
            f"Authoritative joint order must be {expected_joint_names}, "
            f"got {joint_names}"
        )
    if DEFAULT_EE_LINK != "xMateCR7_link6":
        raise ValueError(
            "Authoritative end-effector frame must be xMateCR7_link6, "
            f"got {DEFAULT_EE_LINK}"
        )

    robot = load_robot(urdf_path)
    bounds = get_joint_bounds(robot, joint_names, -np.pi, np.pi)
    lower = np.asarray([bound[0] for bound in bounds], dtype=np.float64)
    upper = np.asarray([bound[1] for bound in bounds], dtype=np.float64)
    (
        canonical_ee,
        canonical_rotations,
        canonical_quaternions,
    ) = trajectory_full_transform_fk(
        robot,
        canonical_q,
        joint_names,
        DEFAULT_EE_LINK,
    )
    canonical_ee = finite_array(
        "canonical_mlp_ee",
        canonical_ee,
        (EXPECTED_STEPS, 3),
    )
    canonical_rotations = finite_array(
        "canonical_mlp_rotation_matrix",
        canonical_rotations,
        (EXPECTED_STEPS, 3, 3),
    )
    canonical_quaternions = finite_array(
        "canonical_mlp_quaternion",
        canonical_quaternions,
        (EXPECTED_STEPS, 4),
    )
    canonical_orientation_error = finite_array(
        "canonical_mlp_orientation_error_rad",
        orientation_error_trajectory(
            target_rotation_matrix,
            canonical_rotations,
        ),
        (EXPECTED_STEPS,),
    )
    canonical_z_error = finite_array(
        "canonical_mlp_z_error_m",
        np.abs(canonical_ee[:, 2] - target_z),
        (EXPECTED_STEPS,),
    )

    # For the first sample the adaptive implementation uses canonical_mlp_q[0].
    # Thereafter its primary seed is the previous accepted refined state; this
    # value is also the canonical first-point retry/known-start seed.
    q_start = canonical_q[0].copy()
    pose_result = adaptive_refine_full_pose_path(
        robot=robot,
        desired_path=desired_path,
        target_rotation=target_rotation_matrix,
        canonical_mlp_q=canonical_q,
        q_start=q_start,
        joint_names=joint_names,
        ee_link=DEFAULT_EE_LINK,
        bounds=bounds,
        lower=lower,
        upper=upper,
        mean_error_gate=float(args.mean_error_gate),
        max_joint_step_gate=float(args.max_joint_step_gate),
        maximum_orientation_error_gate_rad=float(
            args.maximum_orientation_error_gate_rad
        ),
        num_ik_retries=SEQUENTIAL_IK_DEFAULT_RETRIES,
        random_seed=RANDOM_SEED,
        retry_profile=args.retry_profile,
    )
    candidate_table = list(pose_result.candidate_table)
    if int(pose_result.adaptive_metadata["valid_candidate_count"]) == 0:
        fail_generation(
            "Orientation-aware adaptive sequential IK produced no candidate "
            "satisfying every deployment acceptance gate",
            {
                "generation_status": "failed_acceptance_gates",
                "candidate_table": candidate_table,
                "adaptive_metadata": pose_result.adaptive_metadata,
                "maximum_orientation_error_gate_rad": float(
                    args.maximum_orientation_error_gate_rad
                ),
            },
        )

    strong_prior_q = finite_array(
        "strong_prior_q",
        pose_result.q,
        (EXPECTED_STEPS, JOINT_DIM),
    )
    (
        strong_prior_ee,
        strong_prior_rotations,
        strong_prior_quaternions,
    ) = trajectory_full_transform_fk(
        robot,
        strong_prior_q,
        joint_names,
        DEFAULT_EE_LINK,
    )
    strong_prior_ee = finite_array(
        "strong_prior_ee",
        strong_prior_ee,
        (EXPECTED_STEPS, 3),
    )
    strong_prior_rotations = finite_array(
        "strong_prior_rotation_matrix",
        strong_prior_rotations,
        (EXPECTED_STEPS, 3, 3),
    )
    strong_prior_quaternions = finite_array(
        "strong_prior_quaternion",
        strong_prior_quaternions,
        (EXPECTED_STEPS, 4),
    )
    strong_prior_orientation_error = finite_array(
        "strong_prior_orientation_error_rad",
        orientation_error_trajectory(
            target_rotation_matrix,
            strong_prior_rotations,
        ),
        (EXPECTED_STEPS,),
    )
    strong_prior_z_error = finite_array(
        "strong_prior_z_error_m",
        np.abs(strong_prior_ee[:, 2] - target_z),
        (EXPECTED_STEPS,),
    )
    if not np.allclose(
        strong_prior_ee,
        np.asarray(pose_result.positions, dtype=np.float64),
        rtol=1.0e-7,
        atol=1.0e-9,
    ):
        fail_generation(
            "Selected full-pose IK positions do not match authoritative FK",
            {
                "generation_status": "failed_authoritative_fk_check",
                "selected_candidate": pose_result.selected_candidate_name,
            },
        )
    if not np.allclose(
        strong_prior_quaternions,
        np.asarray(pose_result.quaternions, dtype=np.float64),
        rtol=1.0e-7,
        atol=1.0e-9,
    ):
        fail_generation(
            "Selected full-pose IK quaternions do not match authoritative FK",
            {
                "generation_status": "failed_authoritative_orientation_fk_check",
                "selected_candidate": pose_result.selected_candidate_name,
            },
        )

    cartesian_error = np.linalg.norm(strong_prior_ee - desired_path, axis=1)
    mean_error = float(np.mean(cartesian_error))
    rms_error = float(np.sqrt(np.mean(np.square(cartesian_error))))
    maximum_error = float(np.max(cartesian_error))
    maximum_absolute_joint_step = float(
        np.max(np.abs(np.diff(strong_prior_q, axis=0)))
    )
    mean_orientation_error = float(
        np.mean(strong_prior_orientation_error)
    )
    maximum_orientation_error = float(
        np.max(strong_prior_orientation_error)
    )
    mean_strong_prior_z_error = float(
        np.mean(strong_prior_z_error)
    )
    maximum_strong_prior_z_error = float(
        np.max(strong_prior_z_error)
    )
    limit_result = check_joint_limits(
        strong_prior_q,
        lower,
        upper,
        joint_names,
        tolerance=HARD_JOINT_LIMIT_TOLERANCE_RAD,
    )
    violation_count = int(
        limit_result["hard_joint_limit_violation_count"]
    )
    unresolved_count = int(pose_result.unresolved_timestep_count)
    final_rejection_reasons = []
    if unresolved_count != 0:
        final_rejection_reasons.append("unresolved_timesteps")
    if mean_error > float(args.mean_error_gate):
        final_rejection_reasons.append("mean_cartesian_error_gate")
    if maximum_absolute_joint_step > float(args.max_joint_step_gate) + 1.0e-12:
        final_rejection_reasons.append("maximum_absolute_joint_step_gate")
    if violation_count != 0:
        final_rejection_reasons.append("hard_joint_limit_violation")
    if maximum_orientation_error > float(
        args.maximum_orientation_error_gate_rad
    ):
        final_rejection_reasons.append("maximum_orientation_error_gate")
    if maximum_strong_prior_z_error > float(
        args.maximum_z_error_gate_m
    ):
        final_rejection_reasons.append("maximum_strong_prior_z_error_gate")
    if final_rejection_reasons:
        fail_generation(
            "Final authoritative validation rejected the selected strong prior",
            {
                "generation_status": "failed_final_validation",
                "rejection_reasons": final_rejection_reasons,
                "mean_cartesian_error": mean_error,
                "maximum_cartesian_error": maximum_error,
                "maximum_absolute_joint_step_rad": maximum_absolute_joint_step,
                "joint_limit_violation_count": violation_count,
                "unresolved_timestep_count": unresolved_count,
                "mean_orientation_error_rad": mean_orientation_error,
                "maximum_orientation_error_rad": maximum_orientation_error,
                "maximum_orientation_error_gate_rad": float(
                    args.maximum_orientation_error_gate_rad
                ),
                "target_z": target_z,
                "mean_strong_prior_z_error_m": (
                    mean_strong_prior_z_error
                ),
                "maximum_strong_prior_z_error_m": (
                    maximum_strong_prior_z_error
                ),
                "maximum_z_error_gate_m": float(
                    args.maximum_z_error_gate_m
                ),
            },
        )

    checkpoint_sha256 = sha256_file(checkpoint_path)
    input_csv_sha256 = sha256_file(input_csv)
    urdf_sha256 = sha256_file(urdf_path)
    source_description = (
        "Canonical path-conditioned full-q MLP decoded with checkpoint "
        "x_mean/x_std and y_mean/y_std, followed by the repository adaptive "
        "sequential IK with previous accepted refined q as the primary seed "
        "and canonical MLP q as a retry seed. Full-pose IK enforces the fixed "
        f"target RPY {FIXED_RPY_ORDER}="
        f"({fixed_rpy[0]:.17g},{fixed_rpy[1]:.17g},{fixed_rpy[2]:.17g}) rad; "
        "orientation is represented as an XYZW quaternion using the "
        "quaternion_from_euler roll-pitch-yaw convention and checked with "
        "authoritative xMateCR7_link6 full-transform FK. The same FK enforces "
        f"fixed target_z={target_z:.17g} m with a non-weakening maximum "
        f"Z-error gate of {float(args.maximum_z_error_gate_m):.17g} m."
    )
    arrays: Dict[str, Any] = {
        "desired_path": desired_path,
        "strong_prior_q": strong_prior_q,
        "strong_prior_ee": strong_prior_ee,
        "timestamps": timestamps,
        "canonical_mlp_q": canonical_q,
        "canonical_mlp_ee": canonical_ee,
        "fixed_rpy": fixed_rpy,
        "target_rpy": fixed_rpy,
        "target_quaternion": target_quaternion,
        "target_rotation_matrix": target_rotation_matrix,
        "canonical_mlp_quaternion": canonical_quaternions,
        "strong_prior_quaternion": strong_prior_quaternions,
        "canonical_mlp_orientation_error_rad": (
            canonical_orientation_error
        ),
        "strong_prior_orientation_error_rad": (
            strong_prior_orientation_error
        ),
        "mean_orientation_error_rad": np.asarray(
            mean_orientation_error,
            dtype=np.float64,
        ),
        "maximum_orientation_error_rad": np.asarray(
            maximum_orientation_error,
            dtype=np.float64,
        ),
        "maximum_orientation_error_gate_rad": np.asarray(
            args.maximum_orientation_error_gate_rad,
            dtype=np.float64,
        ),
        "orientation_constraint_enforced": np.asarray(
            True,
            dtype=np.bool_,
        ),
        "orientation_fk_frame": np.asarray(DEFAULT_EE_LINK),
        "target_z": np.asarray(target_z, dtype=np.float64),
        "maximum_z_error_gate_m": np.asarray(
            args.maximum_z_error_gate_m,
            dtype=np.float64,
        ),
        "canonical_mlp_z_error_m": canonical_z_error,
        "strong_prior_z_error_m": strong_prior_z_error,
        "mean_strong_prior_z_error_m": np.asarray(
            mean_strong_prior_z_error,
            dtype=np.float64,
        ),
        "maximum_strong_prior_z_error_m": np.asarray(
            maximum_strong_prior_z_error,
            dtype=np.float64,
        ),
        "z_constraint_enforced": np.asarray(True, dtype=np.bool_),
        "z_fk_frame": np.asarray(DEFAULT_EE_LINK),
        "path_name": np.asarray(str(args.path_name)),
        "source_method": np.asarray(SOURCE_METHOD),
        "source_checkpoint": np.asarray(str(checkpoint_path)),
        "source_checkpoint_sha256": np.asarray(checkpoint_sha256),
        "source_description": np.asarray(source_description),
        "input_csv": np.asarray(str(input_csv)),
        "input_csv_sha256": np.asarray(input_csv_sha256),
        "urdf_path": np.asarray(str(urdf_path)),
        "urdf_sha256": np.asarray(urdf_sha256),
        "mean_cartesian_error": np.asarray(mean_error, dtype=np.float64),
        "rms_cartesian_error": np.asarray(rms_error, dtype=np.float64),
        "maximum_cartesian_error": np.asarray(
            maximum_error, dtype=np.float64
        ),
        "maximum_absolute_joint_step_rad": np.asarray(
            maximum_absolute_joint_step, dtype=np.float64
        ),
        "joint_limit_violation_count": np.asarray(
            violation_count, dtype=np.int64
        ),
        "generation_success": np.asarray(True, dtype=np.bool_),
        "generation_status": np.asarray("success"),
    }
    atomic_save_npz(args.output_npz, arrays, overwrite=args.overwrite)
    return {
        "path_name": str(args.path_name),
        "mean_cartesian_error": mean_error,
        "rms_cartesian_error": rms_error,
        "maximum_cartesian_error": maximum_error,
        "maximum_absolute_joint_step_rad": maximum_absolute_joint_step,
        "joint_limit_violation_count": violation_count,
        "output_npz": str(args.output_npz.resolve()),
        "source_checkpoint_sha256": checkpoint_sha256,
        "mean_orientation_error_rad": mean_orientation_error,
        "maximum_orientation_error_rad": maximum_orientation_error,
        "maximum_orientation_error_gate_rad": float(
            args.maximum_orientation_error_gate_rad
        ),
        "maximum_z_error_gate_m": float(args.maximum_z_error_gate_m),
        "target_z": target_z,
        "mean_strong_prior_z_error_m": mean_strong_prior_z_error,
        "maximum_strong_prior_z_error_m": maximum_strong_prior_z_error,
        "maximum_z_error_gate_m": float(args.maximum_z_error_gate_m),
        "quaternion_order": QUATERNION_ORDER,
        "selected_candidate": pose_result.selected_candidate_name,
    }


def base_failure_payload(
    args: argparse.Namespace,
    exc: BaseException,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "generation_success": False,
        "generation_status": "failed",
        "error_type": type(exc).__name__,
        "error": str(exc),
        "path_name": str(args.path_name),
        "input_csv": str(args.input_csv.expanduser().absolute()),
        "output_npz": str(args.output_npz.expanduser().absolute()),
        "mlp_checkpoint": str(args.mlp_checkpoint.expanduser().absolute()),
        "urdf_path": str(args.urdf_path.expanduser().absolute()),
        "fixed_rpy": [float(args.roll), float(args.pitch), float(args.yaw)],
        "fixed_rpy_order": list(FIXED_RPY_ORDER),
        "trajectory_duration_seconds": float(
            args.trajectory_duration_seconds
        ),
        "mean_error_gate": float(args.mean_error_gate),
        "max_joint_step_gate": float(args.max_joint_step_gate),
        "maximum_orientation_error_gate_rad": float(
            args.maximum_orientation_error_gate_rad
        ),
        "maximum_z_error_gate_m": float(args.maximum_z_error_gate_m),
        "retry_profile": str(args.retry_profile),
        "device": str(args.device),
        "traceback": traceback.format_exc(),
    }
    if isinstance(exc, GenerationFailure):
        payload.update(exc.diagnostics)
        payload["generation_success"] = False
    return payload


def main() -> int:
    args = parse_args()
    try:
        result = generate(args)
    except Exception as exc:
        diagnostic_path = failure_json_path(args.output_npz)
        payload = base_failure_payload(args, exc)
        try:
            atomic_write_json(diagnostic_path, payload)
        except Exception as diagnostic_exc:
            print(
                "ERROR: generation failed and the diagnostic JSON could not "
                f"be written: {diagnostic_exc}"
            )
        print(f"DEPLOYMENT_INPUT_GENERATION_FAILED: {type(exc).__name__}: {exc}")
        print(f"diagnostic JSON: {diagnostic_path}")
        return 1

    print("DEPLOYMENT_INPUT_GENERATION_PASSED")
    print(f"path_name: {result['path_name']}")
    print(
        "mean Cartesian error: "
        f"{result['mean_cartesian_error']:.9f} m"
    )
    print(
        "maximum Cartesian error: "
        f"{result['maximum_cartesian_error']:.9f} m"
    )
    print(
        "maximum absolute joint step: "
        f"{result['maximum_absolute_joint_step_rad']:.9f} rad"
    )
    print(
        "maximum orientation error: "
        f"{result['maximum_orientation_error_rad']:.9f} rad"
    )
    print(
        "maximum strong-prior Z error: "
        f"{result['maximum_strong_prior_z_error_m']:.9f} m"
    )
    print(
        "joint-limit violation count: "
        f"{result['joint_limit_violation_count']}"
    )
    print(f"output NPZ: {result['output_npz']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
