#!/usr/bin/env python3
"""Deployment-oriented frozen diffusion v8.1 joint trajectory generator."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import hashlib
import json
import multiprocessing
import os
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

import evaluate_diffusion_v7_teacher_forced_validation as v7_evaluator
import evaluate_diffusion_v8_1_anchored_recursive_jerk_guard as v81
import evaluate_diffusion_v8_anchored_recursive_rollout as v8
import generate_diffusion_v7_cost_improving_residual_targets as target_generator
from generate_ik_seed_path import DEFAULT_URDF_PATH


TRAJECTORY_LENGTH = 100
JOINT_DIM = 6
XYZ_DIM = 3
FROZEN_TARGET_SCALE = 1.0
FROZEN_OUTPUT_ALPHA = 0.125
FROZEN_K = 8
FROZEN_DDIM_STEPS = 50
FROZEN_ETA = 0.0
FROZEN_HORIZON = 32
FROZEN_EXECUTION_HORIZON = 8
FROZEN_ANCHORING_HORIZON = 8
FROZEN_CHECKPOINT_STATE = "raw_last_epoch187"
MAXIMUM_INTERNAL_JOINT_STEP_RAD = 0.20
FK_RTOL = 1.0e-5
FK_ATOL = 2.0e-5
REQUIRED_SAFETY_METRICS = (
    "full_path_safety_pass",
    "maximum_actual_internal_joint_step_rad",
    "rollout_full_hard_joint_limit_violation_count",
    "rollout_full_hard_joint_limit_violation_magnitude",
    "internal_full_path_robot_aware_delta_score",
    "cartesian_mean_error_delta",
    "internal_robot_score_contribution_jerk",
)


@dataclass(frozen=True)
class DeploymentInput:
    desired_path: np.ndarray
    strong_prior_q: np.ndarray
    strong_prior_ee: np.ndarray
    timestamps: np.ndarray
    input_path_name: str
    source_method: str
    source_checkpoint: str
    source_description: str
    input_sha256: str
    deployment_path_id: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_npz", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument(
        "--training_dataset_dir",
        type=Path,
        default=Path(
            "data/cartesian_expert_dataset_v3/"
            "diffusion_v8_multitarget_scaled_training_dataset_100paths"
        ),
    )
    parser.add_argument(
        "--model_dir",
        type=Path,
        default=Path(
            "models/"
            "diffusion_v8_multitarget_scaled_residual_unet_100paths_"
            "epsilon_only_seed42"
        ),
    )
    parser.add_argument("--checkpoint_state", default=FROZEN_CHECKPOINT_STATE)
    parser.add_argument("--target_scale", type=float, default=FROZEN_TARGET_SCALE)
    parser.add_argument("--output_alpha", type=float, default=FROZEN_OUTPUT_ALPHA)
    parser.add_argument("--k", type=int, default=FROZEN_K)
    parser.add_argument("--sampling_seed", type=int, default=53)
    parser.add_argument("--ddim_steps", type=int, default=FROZEN_DDIM_STEPS)
    parser.add_argument("--eta", type=float, default=FROZEN_ETA)
    parser.add_argument("--horizon", type=int, default=FROZEN_HORIZON)
    parser.add_argument("--execution_horizon", type=int, default=FROZEN_EXECUTION_HORIZON)
    parser.add_argument("--anchoring_horizon", type=int, default=FROZEN_ANCHORING_HORIZON)
    parser.add_argument("--trajectory_duration_seconds", type=float, default=10.0)
    parser.add_argument("--num_cpu_workers", type=int, default=8)
    parser.add_argument("--gpu_batch_size", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--path_id", default=None)
    parser.add_argument("--urdf_path", type=Path, default=Path(DEFAULT_URDF_PATH))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def require_frozen_args(args: argparse.Namespace) -> None:
    expected = {
        "target_scale": FROZEN_TARGET_SCALE,
        "output_alpha": FROZEN_OUTPUT_ALPHA,
        "k": FROZEN_K,
        "ddim_steps": FROZEN_DDIM_STEPS,
        "eta": FROZEN_ETA,
        "horizon": FROZEN_HORIZON,
        "execution_horizon": FROZEN_EXECUTION_HORIZON,
        "anchoring_horizon": FROZEN_ANCHORING_HORIZON,
    }
    for name, value in expected.items():
        actual = getattr(args, name)
        if actual != value:
            raise ValueError(f"--{name} is frozen at {value!r}; got {actual!r}")
    if args.checkpoint_state != FROZEN_CHECKPOINT_STATE:
        raise ValueError(
            f"--checkpoint_state is frozen at {FROZEN_CHECKPOINT_STATE!r}; "
            f"got {args.checkpoint_state!r}"
        )
    if args.trajectory_duration_seconds <= 0.0:
        raise ValueError("--trajectory_duration_seconds must be positive")
    if args.num_cpu_workers < 1 or args.gpu_batch_size < 1:
        raise ValueError("--num_cpu_workers and --gpu_batch_size must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_numeric_metric(
    metrics: Mapping[str, Any],
    key: str,
    *,
    integer: bool = False,
) -> float:
    if key not in metrics:
        raise KeyError(f"Required deployment safety metric is missing: {key}")
    value = metrics[key]
    if not np.isscalar(value):
        raise TypeError(f"Required deployment safety metric is not scalar: {key}")
    numeric = float(value)
    if not np.isfinite(numeric):
        raise ValueError(f"Required deployment safety metric is nonfinite: {key}")
    if integer and int(numeric) != numeric:
        raise ValueError(f"Required deployment safety metric is not integer-like: {key}")
    return numeric


def required_safety_metrics(metrics: Mapping[str, Any]) -> Dict[str, float]:
    return {
        "full_path_safety_pass": require_numeric_metric(
            metrics, "full_path_safety_pass", integer=True
        ),
        "maximum_actual_internal_joint_step_rad": require_numeric_metric(
            metrics, "maximum_actual_internal_joint_step_rad"
        ),
        "rollout_full_hard_joint_limit_violation_count": require_numeric_metric(
            metrics, "rollout_full_hard_joint_limit_violation_count", integer=True
        ),
        "rollout_full_hard_joint_limit_violation_magnitude": require_numeric_metric(
            metrics, "rollout_full_hard_joint_limit_violation_magnitude"
        ),
        "internal_full_path_robot_aware_delta_score": require_numeric_metric(
            metrics, "internal_full_path_robot_aware_delta_score"
        ),
        "cartesian_mean_error_delta": require_numeric_metric(
            metrics, "cartesian_mean_error_delta"
        ),
        "internal_robot_score_contribution_jerk": require_numeric_metric(
            metrics, "internal_robot_score_contribution_jerk"
        ),
    }


def sha256_arrays(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def sanitize_identifier(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", value.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_").lower()
    return cleaned or "trajectory"


def scalar_text(data: Mapping[str, Any], key: str, default: str = "") -> str:
    if key not in data:
        return default
    value = data[key]
    if isinstance(value, np.ndarray):
        if value.shape == ():
            value = value.item()
        elif value.size == 1:
            value = value.reshape(-1)[0]
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def finite_array(name: str, value: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f"{name} has shape {array.shape}; expected {shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains nonfinite values")
    return array


def compute_fk(robot: target_generator.RobotContext, q: np.ndarray) -> np.ndarray:
    return v8.compute_fk_positions(robot, np.asarray(q, dtype=np.float64))


def load_input(args: argparse.Namespace, robot: target_generator.RobotContext) -> DeploymentInput:
    input_sha = sha256_file(args.input_npz)
    with np.load(args.input_npz, allow_pickle=False) as data:
        if "desired_path" not in data or "strong_prior_q" not in data:
            raise KeyError("--input_npz must contain desired_path and strong_prior_q")
        desired_path = finite_array(
            "desired_path", data["desired_path"], (TRAJECTORY_LENGTH, XYZ_DIM)
        )
        strong_prior_q = finite_array(
            "strong_prior_q", data["strong_prior_q"], (TRAJECTORY_LENGTH, JOINT_DIM)
        )
        authoritative_prior_ee = compute_fk(robot, strong_prior_q)
        if "strong_prior_ee" in data:
            supplied_prior_ee = finite_array(
                "strong_prior_ee",
                data["strong_prior_ee"],
                (TRAJECTORY_LENGTH, XYZ_DIM),
            )
            if not np.allclose(
                supplied_prior_ee, authoritative_prior_ee, rtol=FK_RTOL, atol=FK_ATOL
            ):
                raise ValueError("strong_prior_ee does not match authoritative FK")
        if "timestamps" in data:
            timestamps = finite_array("timestamps", data["timestamps"], (TRAJECTORY_LENGTH,))
        else:
            timestamps = np.linspace(
                0.0,
                float(args.trajectory_duration_seconds),
                TRAJECTORY_LENGTH,
                dtype=np.float64,
            )
        if not np.all(np.diff(timestamps) > 0.0):
            raise ValueError("timestamps must be strictly increasing")
        path_name = scalar_text(data, "path_name", args.input_npz.stem)
        source_method = scalar_text(data, "source_method", "")
        source_checkpoint = scalar_text(data, "source_checkpoint", "")
        source_description = scalar_text(data, "source_description", "")
    base_id = sanitize_identifier(args.path_id or path_name)
    path_hash = sha256_arrays(desired_path, strong_prior_q)[:8]
    deployment_path_id = f"deployment__{base_id}__{path_hash}"
    return DeploymentInput(
        desired_path=desired_path,
        strong_prior_q=strong_prior_q,
        strong_prior_ee=authoritative_prior_ee,
        timestamps=timestamps,
        input_path_name=path_name,
        source_method=source_method,
        source_checkpoint=source_checkpoint,
        source_description=source_description,
        input_sha256=input_sha,
        deployment_path_id=deployment_path_id,
    )


def validate_prior_fallback(
    record: v8.PhysicalPathRecord,
    robot: target_generator.RobotContext,
    args: argparse.Namespace,
) -> None:
    start = 0
    while start < len(record.strong_prior_q) - 1:
        current_q = record.strong_prior_q[start]
        previous_q = record.strong_prior_q[start - 1] if start > 0 else None
        execution_count = min(args.execution_horizon, len(record.strong_prior_q) - 1 - start)
        anchored = v8.build_anchored_prior_window(
            record.strong_prior_q,
            start,
            current_q,
            args.horizon,
            args.anchoring_horizon,
        )
        desired = v8.padded_window(record.desired_path, start, args.horizon)
        context = v8.make_action_context(
            record,
            start,
            current_q,
            previous_q,
            anchored,
            desired,
            robot,
            execution_count,
        )
        metrics = v7_evaluator.evaluate_metrics(
            robot, context, context.prior_q, execution_count
        )
        reasons = v8.recursive_executed_prefix_hard_safety_reasons(metrics)
        if reasons:
            raise ValueError(f"prior fallback hard-safety failed at {start}: {reasons}")
        start += execution_count


def validate_segment_arrays(result: v8.RolloutResult) -> Dict[str, int]:
    starts = np.asarray(result.window_start_indices)
    accepted_raw = np.asarray(result.accepted_step_mask)
    fallback_raw = np.asarray(result.fallback_step_mask)
    selected = np.asarray(result.selected_candidate_indices)
    executed_indices = np.asarray(result.executed_indices)
    for name, mask in (
        ("accepted_step_mask", accepted_raw),
        ("fallback_step_mask", fallback_raw),
    ):
        if not np.all((mask == 0) | (mask == 1)):
            raise ValueError(f"{name} values must be boolean or exactly 0/1")
    accepted = accepted_raw.astype(bool)
    fallback = fallback_raw.astype(bool)
    if starts.ndim != 1:
        raise ValueError("window_start_indices must be one-dimensional")
    if accepted.ndim != 1 or fallback.ndim != 1 or selected.ndim != 1:
        raise ValueError("segment masks and selected indices must be one-dimensional")
    if not (len(starts) == len(accepted) == len(fallback) == len(selected)):
        raise ValueError("segment arrays must have equal length")
    if not np.array_equal(accepted, np.logical_not(fallback)):
        raise ValueError("accepted_step_mask and fallback_step_mask must be complements")
    if len(starts) == 0 or int(starts[0]) != 0:
        raise ValueError("window_start_indices must begin at 0")
    if np.any(np.diff(starts) <= 0):
        raise ValueError("window_start_indices must be strictly increasing")
    if np.any(starts < 0) or np.any(starts >= TRAJECTORY_LENGTH):
        raise ValueError("window_start_indices contain out-of-bounds starts")
    if np.any((selected < -1) | (selected >= FROZEN_K)):
        raise ValueError("selected_candidate_indices must be -1 or 0..7")
    if not np.all(selected[fallback] == -1):
        raise ValueError("fallback selected_candidate_indices must be -1")
    if not np.all((selected[accepted] >= 0) & (selected[accepted] < FROZEN_K)):
        raise ValueError("accepted selected_candidate_indices must be 0..7")
    if not np.array_equal(executed_indices, np.arange(TRAJECTORY_LENGTH)):
        raise ValueError("executed_indices must be exactly 0..99")
    if np.asarray(result.applied_correction_norms).shape != (TRAJECTORY_LENGTH,):
        raise ValueError("applied_correction_norms must have shape (100,)")
    if np.asarray(result.executed_source).shape != (TRAJECTORY_LENGTH,):
        raise ValueError("executed_source must have shape (100,)")
    selected_count = int(np.sum(accepted))
    fallback_count = int(np.sum(fallback))
    segment_count = int(len(starts))
    if selected_count + fallback_count != segment_count:
        raise ValueError("selected + fallback segment counts must equal segment count")
    return {
        "selected_diffusion_segment_count": selected_count,
        "fallback_segment_count": fallback_count,
        "rollout_segment_count": segment_count,
    }


def prepare_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f"Output directory is nonempty: {path}; pass --overwrite")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    (path / "plots").mkdir(parents=True, exist_ok=True)


def atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    atomic_write_text(
        path,
        json.dumps(json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
    )


def atomic_write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    if not rows:
        tmp.write_text("")
        tmp.replace(path)
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with tmp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def atomic_save_npz(path: Path, **arrays: Any) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def dynamics(q: np.ndarray, timestamps: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    velocity = np.gradient(q, timestamps, axis=0, edge_order=2)
    acceleration = np.gradient(velocity, timestamps, axis=0, edge_order=2)
    jerk = np.gradient(acceleration, timestamps, axis=0, edge_order=2)
    return velocity, acceleration, jerk


def rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def manipulability_diagnostics(
    robot: target_generator.RobotContext,
    q: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    manipulability: List[float] = []
    min_singular: List[float] = []
    for row in q:
        jacobian = target_generator.positional_jacobian(robot, row)
        singular = np.linalg.svd(jacobian, compute_uv=False)
        min_singular.append(float(np.min(singular)))
        manipulability.append(target_generator.manipulability_and_penalty(jacobian)[0])
    return np.asarray(manipulability), np.asarray(min_singular)


def script_hash(module: Any) -> Optional[str]:
    module_file = getattr(module, "__file__", None)
    return sha256_file(Path(module_file)) if module_file else None


def model_dir_hashes(path: Path) -> Dict[str, str]:
    hashes: Dict[str, str] = {}
    if not path.is_dir():
        return hashes
    for item in sorted(p for p in path.rglob("*") if p.is_file()):
        hashes[str(item.relative_to(path))] = sha256_file(item)
    return hashes


def git_metadata() -> Dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=5,
        ).strip()
        return {"commit": commit, "error": None}
    except Exception as exc:  # noqa: BLE001 - provenance must not block deployment.
        return {"commit": None, "error": str(exc)}


def annotate_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    sampling_seed: int,
    deployment_path_id: str,
    input_path_name: str,
    input_sha256: str,
    verdict: str,
) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for row in rows:
        annotated = dict(row)
        annotated.update(
            {
                "sampling_seed": int(sampling_seed),
                "deployment_path_id": deployment_path_id,
                "input_path_name": input_path_name,
                "input_sha256": input_sha256,
                "verdict": verdict,
            }
        )
        output.append(annotated)
    return output


def final_safety_checks(
    result: v8.RolloutResult,
    recomputed_metrics: Mapping[str, Any],
    timestamps: np.ndarray,
    robot: target_generator.RobotContext,
) -> Tuple[bool, List[str], Dict[str, Any]]:
    reasons: List[str] = []
    required = required_safety_metrics(recomputed_metrics)
    prefix_failure_reasons: List[str] = []
    prefix_check_count = 0
    if result.rollout_q.shape != (TRAJECTORY_LENGTH, JOINT_DIM):
        reasons.append("final_q_shape")
    if result.rollout_ee.shape != (TRAJECTORY_LENGTH, XYZ_DIM):
        reasons.append("final_ee_shape")
    if not np.all(np.isfinite(result.rollout_q)):
        reasons.append("nonfinite_final_q")
    if not np.all(np.isfinite(result.rollout_ee)):
        reasons.append("nonfinite_final_ee")
    if not np.all(np.diff(timestamps) > 0.0):
        reasons.append("invalid_timestamps")
    if int(required["full_path_safety_pass"]) != 1:
        reasons.append("full_path_safety_pass_failed")
    if required["maximum_actual_internal_joint_step_rad"] > MAXIMUM_INTERNAL_JOINT_STEP_RAD:
        reasons.append("maximum_internal_joint_step")
    if int(required["rollout_full_hard_joint_limit_violation_count"]) > 0:
        reasons.append("joint_limit_violation")
    if required["rollout_full_hard_joint_limit_violation_magnitude"] > 0.0:
        reasons.append("joint_limit_violation_magnitude")
    start = 0
    while start < TRAJECTORY_LENGTH - 1:
        prefix_check_count += 1
        execution_count = min(FROZEN_EXECUTION_HORIZON, TRAJECTORY_LENGTH - 1 - start)
        current_q = result.rollout_q[start]
        previous_q = result.rollout_q[start - 1] if start > 0 else None
        window_q = v8.padded_window(result.rollout_q, start, FROZEN_HORIZON)
        desired = v8.padded_window(result.path.desired_path, start, FROZEN_HORIZON)
        context = v8.make_action_context(
            result.path,
            start,
            current_q,
            previous_q,
            window_q,
            desired,
            robot,
            execution_count,
        )
        metrics = v7_evaluator.evaluate_metrics(
            robot, context, context.prior_q, execution_count
        )
        prefix_reasons = v8.recursive_executed_prefix_hard_safety_reasons(metrics)
        if prefix_reasons:
            reason = f"executed_prefix_hard_safety@{start}:{'|'.join(prefix_reasons)}"
            prefix_failure_reasons.append(reason)
            reasons.append(reason)
        start += execution_count
    details = {
        "independent_full_path_safety_pass": int(required["full_path_safety_pass"] == 1),
        "independent_executed_prefix_safety_pass": int(not prefix_failure_reasons),
        "independent_executed_prefix_check_count": prefix_check_count,
        "independent_executed_prefix_failure_count": len(prefix_failure_reasons),
        "independent_executed_prefix_failure_reasons": prefix_failure_reasons,
        "independent_joint_limit_pass": int(
            required["rollout_full_hard_joint_limit_violation_count"] == 0
            and required["rollout_full_hard_joint_limit_violation_magnitude"] == 0.0
        ),
        "independent_timestamp_pass": int(np.all(np.diff(timestamps) > 0.0)),
        "independent_finite_joint_pass": int(np.all(np.isfinite(result.rollout_q))),
        "independent_finite_fk_pass": int(np.all(np.isfinite(result.rollout_ee))),
    }
    return not reasons and all(
        bool(details[key])
        for key in (
            "independent_full_path_safety_pass",
            "independent_executed_prefix_safety_pass",
            "independent_joint_limit_pass",
            "independent_timestamp_pass",
            "independent_finite_joint_pass",
            "independent_finite_fk_pass",
        )
    ), reasons, details


def trajectory_rows(
    data: DeploymentInput,
    result: v8.RolloutResult,
    cartesian_error: np.ndarray,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for index in range(TRAJECTORY_LENGTH):
        row: Dict[str, Any] = {
            "sample_index": index,
            "time_seconds": float(data.timestamps[index]),
            "desired_x": float(data.desired_path[index, 0]),
            "desired_y": float(data.desired_path[index, 1]),
            "desired_z": float(data.desired_path[index, 2]),
            "prior_x": float(data.strong_prior_ee[index, 0]),
            "prior_y": float(data.strong_prior_ee[index, 1]),
            "prior_z": float(data.strong_prior_ee[index, 2]),
            "final_x": float(result.rollout_ee[index, 0]),
            "final_y": float(result.rollout_ee[index, 1]),
            "final_z": float(result.rollout_ee[index, 2]),
            "cartesian_error": float(cartesian_error[index]),
            "execution_source": str(result.executed_source[index]),
            "accepted_diffusion": int(result.executed_source[index] == "accepted_diffusion_candidate"),
            "fallback": int(result.executed_source[index] == "anchored_prior_fallback"),
        }
        for joint in range(JOINT_DIM):
            row[f"prior_q{joint + 1}"] = float(data.strong_prior_q[index, joint])
            row[f"final_q{joint + 1}"] = float(result.rollout_q[index, joint])
        rows.append(row)
    return rows


def position_rows(timestamps: np.ndarray, q: np.ndarray) -> List[Dict[str, Any]]:
    return [
        {
            "sample_index": index,
            "time_seconds": float(timestamps[index]),
            **{f"q{joint + 1}": float(q[index, joint]) for joint in range(JOINT_DIM)},
        }
        for index in range(TRAJECTORY_LENGTH)
    ]


def dynamics_rows(
    timestamps: np.ndarray,
    velocity: np.ndarray,
    acceleration: np.ndarray,
    jerk: np.ndarray,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for index in range(TRAJECTORY_LENGTH):
        row: Dict[str, Any] = {
            "sample_index": index,
            "time_seconds": float(timestamps[index]),
        }
        for joint in range(JOINT_DIM):
            row[f"dq{joint + 1}"] = float(velocity[index, joint])
        for joint in range(JOINT_DIM):
            row[f"ddq{joint + 1}"] = float(acceleration[index, joint])
        for joint in range(JOINT_DIM):
            row[f"dddq{joint + 1}"] = float(jerk[index, joint])
        rows.append(row)
    return rows


def tracking_rows(
    timestamps: np.ndarray,
    desired: np.ndarray,
    final_ee: np.ndarray,
    error: np.ndarray,
    manipulability: np.ndarray,
    min_singular: np.ndarray,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for index in range(TRAJECTORY_LENGTH):
        rows.append(
            {
                "sample_index": index,
                "time_seconds": float(timestamps[index]),
                "desired_x": float(desired[index, 0]),
                "desired_y": float(desired[index, 1]),
                "desired_z": float(desired[index, 2]),
                "final_x": float(final_ee[index, 0]),
                "final_y": float(final_ee[index, 1]),
                "final_z": float(final_ee[index, 2]),
                "cartesian_error": float(error[index]),
                "manipulability": float(manipulability[index]),
                "minimum_translational_jacobian_singular_value": float(min_singular[index]),
            }
        )
    return rows


def approved_rows(timestamps: np.ndarray, q: np.ndarray) -> List[Dict[str, Any]]:
    return [
        {
            "time_seconds": float(timestamps[index]),
            **{f"q{joint + 1}": float(q[index, joint]) for joint in range(JOINT_DIM)},
        }
        for index in range(TRAJECTORY_LENGTH)
    ]


def save_plots(
    output_dir: Path,
    data: DeploymentInput,
    result: v8.RolloutResult,
    velocity: np.ndarray,
    acceleration: np.ndarray,
    jerk: np.ndarray,
    cartesian_error: np.ndarray,
    manipulability: np.ndarray,
) -> None:
    plot_dir = output_dir / "plots"
    t = data.timestamps
    labels = [f"q{joint + 1}" for joint in range(JOINT_DIM)]
    for values, name, ylabel in (
        (result.rollout_q, "joint_positions.png", "joint position (rad)"),
        (velocity, "joint_velocities.png", "joint velocity (rad/s)"),
        (acceleration, "joint_accelerations.png", "joint acceleration (rad/s^2)"),
        (jerk, "joint_jerks.png", "joint jerk (rad/s^3)"),
    ):
        figure, axis = plt.subplots(figsize=(10, 5))
        for joint, label in enumerate(labels):
            axis.plot(t, values[:, joint], label=label)
        axis.set(xlabel="time (s)", ylabel=ylabel)
        axis.legend(ncol=3)
        figure.tight_layout()
        figure.savefig(str(plot_dir / name), dpi=150)
        plt.close(figure)
    figure, axis = plt.subplots(figsize=(8, 6))
    axis.plot(data.desired_path[:, 0], data.desired_path[:, 1], label="desired")
    axis.plot(result.rollout_ee[:, 0], result.rollout_ee[:, 1], label="final FK")
    axis.set(xlabel="x", ylabel="y")
    axis.legend()
    figure.tight_layout()
    figure.savefig(str(plot_dir / "cartesian_tracking.png"), dpi=150)
    plt.close(figure)
    for values, name, ylabel in (
        (cartesian_error, "cartesian_error.png", "Cartesian error (m)"),
        (manipulability, "manipulability.png", "manipulability"),
        (result.applied_correction_norms, "correction_norms.png", "correction norm (rad)"),
    ):
        figure, axis = plt.subplots(figsize=(10, 4))
        axis.plot(t, values)
        axis.set(xlabel="time (s)", ylabel=ylabel)
        figure.tight_layout()
        figure.savefig(str(plot_dir / name), dpi=150)
        plt.close(figure)
    figure, axis = plt.subplots(figsize=(10, 3))
    source = np.zeros(TRAJECTORY_LENGTH)
    source[result.executed_source == "accepted_diffusion_candidate"] = 1.0
    axis.step(t, source, where="post")
    axis.set(yticks=(0, 1), yticklabels=("fallback/prior", "diffusion"), xlabel="time (s)")
    figure.tight_layout()
    figure.savefig(str(plot_dir / "segment_source_timeline.png"), dpi=150)
    plt.close(figure)


def main() -> int:
    args = parse_args()
    require_frozen_args(args)
    prepare_output_dir(args.output_dir, args.overwrite)
    if v81.run_rollout_v8_1.__module__ != "evaluate_diffusion_v8_1_anchored_recursive_jerk_guard":
        raise AssertionError("run_rollout_v8_1 is not imported from frozen v8.1 module")
    if v81.HISTORY_AWARE_JERK_TOLERANCE != 1.0e-12:
        raise AssertionError("Frozen v8.1 jerk tolerance changed")
    if tuple(v8.FROZEN_K_VALUES) != (1, 4, 8):
        raise AssertionError("Frozen K values changed")
    resolved_urdf = args.urdf_path.resolve()
    urdf_sha256 = sha256_file(resolved_urdf)
    robot = v7_evaluator.make_robot_context(resolved_urdf)
    data = load_input(args, robot)
    atomic_save_npz(
        args.output_dir / "deployment_input_copy.npz",
        desired_path=data.desired_path,
        strong_prior_q=data.strong_prior_q,
        strong_prior_ee=data.strong_prior_ee,
        timestamps=data.timestamps,
        path_name=data.input_path_name,
        source_method=data.source_method,
        source_checkpoint=data.source_checkpoint,
        source_description=data.source_description,
        input_sha256=data.input_sha256,
        deployment_path_id=data.deployment_path_id,
        urdf_path=str(resolved_urdf),
        urdf_sha256=urdf_sha256,
    )
    record = v8.PhysicalPathRecord(
        path_id=data.deployment_path_id,
        path_index=0,
        population="ordinary",
        desired_path=data.desired_path,
        strong_prior_q=data.strong_prior_q,
        prior_ee=data.strong_prior_ee,
    )
    validate_prior_fallback(record, robot, args)
    inference = v81.load_validated_inference_bundle(
        args.training_dataset_dir,
        args.model_dir,
        args.checkpoint_state,
        args.device,
        args.ddim_steps,
    )
    args.dataset_dir = args.training_dataset_dir
    args.disable_history_aware_jerk_guard = False
    executor: Optional[concurrent.futures.ProcessPoolExecutor] = None
    sample_cache: Dict[Tuple[bytes, Tuple[int, ...]], np.ndarray] = {}
    try:
        if args.num_cpu_workers > 1:
            executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=args.num_cpu_workers,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=v7_evaluator.initialize_candidate_worker,
                initargs=(str(resolved_urdf),),
            )
        result = v81.run_rollout_v8_1(
            record,
            FROZEN_K,
            inference,
            robot,
            executor,
            args,
            sample_cache,
        )
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=False)
    recomputed_metrics = v8.compute_full_trajectory_metrics(
        robot=robot,
        strong_prior_q=data.strong_prior_q,
        rollout_q=result.rollout_q,
        desired_path=data.desired_path,
    )
    recomputed_ee = np.asarray(recomputed_metrics.pop("rollout_ee"), dtype=np.float64)
    recomputed_prior_ee = np.asarray(recomputed_metrics.pop("prior_ee"), dtype=np.float64)
    if not np.allclose(recomputed_ee, result.rollout_ee, rtol=FK_RTOL, atol=FK_ATOL):
        raise AssertionError("Frozen rollout FK and independent FK recomputation differ")
    if not np.allclose(recomputed_prior_ee, data.strong_prior_ee, rtol=FK_RTOL, atol=FK_ATOL):
        raise AssertionError("Prior FK changed during deployment")
    required_metrics = required_safety_metrics(recomputed_metrics)
    segment_counts = validate_segment_arrays(result)
    velocity, acceleration, jerk = dynamics(result.rollout_q, data.timestamps)
    manipulability, min_singular = manipulability_diagnostics(robot, result.rollout_q)
    cartesian_error = np.linalg.norm(result.rollout_ee - data.desired_path, axis=1)
    safe, rejection_reasons, independent_safety = final_safety_checks(
        result, recomputed_metrics, data.timestamps, robot
    )
    verdict = (
        "V8_1_DEPLOYMENT_TRAJECTORY_ACCEPTED"
        if safe
        else "V8_1_DEPLOYMENT_TRAJECTORY_REJECTED"
    )
    decision_rows = annotate_rows(
        result.decision_rows,
        sampling_seed=args.sampling_seed,
        deployment_path_id=data.deployment_path_id,
        input_path_name=data.input_path_name,
        input_sha256=data.input_sha256,
        verdict=verdict,
    )
    candidate_rows = annotate_rows(
        result.candidate_rows,
        sampling_seed=args.sampling_seed,
        deployment_path_id=data.deployment_path_id,
        input_path_name=data.input_path_name,
        input_sha256=data.input_sha256,
        verdict=verdict,
    )
    metrics: Dict[str, Any] = {
        **{key: value for key, value in recomputed_metrics.items() if np.isscalar(value)},
        **independent_safety,
        "verdict": verdict,
        "accepted": safe,
        "rejection_reasons": rejection_reasons,
        "deployment_path_id": data.deployment_path_id,
        "input_path_name": data.input_path_name,
        "input_file": str(args.input_npz),
        "input_sha256": data.input_sha256,
        "urdf_path": str(resolved_urdf),
        "urdf_sha256": urdf_sha256,
        "sampling_seed": int(args.sampling_seed),
        "maximum_absolute_joint_velocity_per_joint": np.max(np.abs(velocity), axis=0),
        "maximum_absolute_joint_acceleration_per_joint": np.max(np.abs(acceleration), axis=0),
        "maximum_absolute_joint_jerk_per_joint": np.max(np.abs(jerk), axis=0),
        "trajectory_rms_velocity": rms(velocity),
        "trajectory_rms_acceleration": rms(acceleration),
        "trajectory_rms_jerk": rms(jerk),
        "minimum_manipulability": float(np.min(manipulability)),
        "mean_manipulability": float(np.mean(manipulability)),
        "minimum_translational_jacobian_singular_value": float(np.min(min_singular)),
        "mean_cartesian_error": float(np.mean(cartesian_error)),
        "rms_cartesian_error": float(np.sqrt(np.mean(np.square(cartesian_error)))),
        "maximum_cartesian_error": float(np.max(cartesian_error)),
        "robot_aware_full_path_delta_score": float(
            required_metrics["internal_full_path_robot_aware_delta_score"]
        ),
        "cartesian_mean_error_delta_relative_to_prior": float(
            required_metrics["cartesian_mean_error_delta"]
        ),
        "jerk_contribution_relative_to_prior": float(
            required_metrics["internal_robot_score_contribution_jerk"]
        ),
        "accepted_rollout_step_rate": float(result.metrics["accepted_rollout_step_rate"]),
        "fallback_rate": float(result.metrics["fallback_rate"]),
        "jerk_rejection_rate_among_original_v8_selectable_candidates": float(
            result.metrics.get("history_aware_jerk_rejection_rate_among_v8_selectable", np.nan)
        ),
        **segment_counts,
        "model_dir": str(args.model_dir),
        "training_dataset_dir": str(args.training_dataset_dir),
        "checkpoint_state": inference.checkpoint_state,
        "checkpoint_state_hash": inference.checkpoint_state_hash,
        "target_scale": FROZEN_TARGET_SCALE,
        "output_alpha": FROZEN_OUTPUT_ALPHA,
        "k": FROZEN_K,
        "ddim_steps": FROZEN_DDIM_STEPS,
        "eta": FROZEN_ETA,
        "horizon": FROZEN_HORIZON,
        "execution_horizon": FROZEN_EXECUTION_HORIZON,
        "anchoring_horizon": FROZEN_ANCHORING_HORIZON,
        "history_aware_jerk_tolerance": v81.HISTORY_AWARE_JERK_TOLERANCE,
        "source_method": data.source_method,
        "source_checkpoint": data.source_checkpoint,
        "source_description": data.source_description,
        "provenance": {
            "generator_script_sha256": sha256_file(Path(__file__).resolve()),
            "frozen_v8_1_script_sha256": script_hash(v81),
            "frozen_v8_script_sha256": script_hash(v8),
            "training_dataset_manifest_or_path": str(args.training_dataset_dir),
            "model_directory_file_hashes": model_dir_hashes(args.model_dir),
            "checkpoint_state_hash": inference.checkpoint_state_hash,
            "input_npz_sha256": data.input_sha256,
            "urdf_path": str(resolved_urdf),
            "urdf_sha256": urdf_sha256,
            "python_version": sys.version,
            "platform": platform.platform(),
            "numpy_version": np.__version__,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "git": git_metadata(),
        },
    }
    atomic_write_csv(args.output_dir / "deployment_trajectory.csv", trajectory_rows(data, result, cartesian_error))
    atomic_write_csv(args.output_dir / "deployment_joint_positions.csv", position_rows(data.timestamps, result.rollout_q))
    atomic_write_csv(args.output_dir / "deployment_joint_dynamics.csv", dynamics_rows(data.timestamps, velocity, acceleration, jerk))
    atomic_write_csv(args.output_dir / "deployment_cartesian_tracking.csv", tracking_rows(data.timestamps, data.desired_path, result.rollout_ee, cartesian_error, manipulability, min_singular))
    atomic_write_csv(args.output_dir / "deployment_segment_decisions.csv", decision_rows)
    atomic_write_csv(args.output_dir / "deployment_candidate_results.csv", candidate_rows)
    atomic_save_npz(
        args.output_dir / "deployment_trajectory_full.npz",
        timestamps=data.timestamps,
        desired_path=data.desired_path,
        strong_prior_q=data.strong_prior_q,
        strong_prior_ee=data.strong_prior_ee,
        final_q=result.rollout_q,
        final_ee=result.rollout_ee,
        joint_velocity=velocity,
        joint_acceleration=acceleration,
        joint_jerk=jerk,
        manipulability=manipulability,
        minimum_translational_jacobian_singular_value=min_singular,
        executed_source=result.executed_source,
        accepted_step_mask=result.accepted_step_mask,
        fallback_step_mask=result.fallback_step_mask,
        selected_candidate_indices=result.selected_candidate_indices,
        applied_correction_norms=result.applied_correction_norms,
        window_start_indices=result.window_start_indices,
        executed_indices=result.executed_indices,
        sampling_seed=int(args.sampling_seed),
        deployment_path_id=data.deployment_path_id,
        input_path_name=data.input_path_name,
        input_file=str(args.input_npz),
        input_sha256=data.input_sha256,
        urdf_path=str(resolved_urdf),
        urdf_sha256=urdf_sha256,
        model_dir=str(args.model_dir),
        training_dataset_dir=str(args.training_dataset_dir),
        checkpoint_state=inference.checkpoint_state,
        checkpoint_state_hash=inference.checkpoint_state_hash,
        target_scale=FROZEN_TARGET_SCALE,
        output_alpha=FROZEN_OUTPUT_ALPHA,
        k=FROZEN_K,
        ddim_steps=FROZEN_DDIM_STEPS,
        eta=FROZEN_ETA,
        horizon=FROZEN_HORIZON,
        execution_horizon=FROZEN_EXECUTION_HORIZON,
        anchoring_horizon=FROZEN_ANCHORING_HORIZON,
        history_aware_jerk_tolerance=v81.HISTORY_AWARE_JERK_TOLERANCE,
        verdict=verdict,
    )
    atomic_write_json(args.output_dir / "deployment_metrics.json", metrics)
    report = [
        "Diffusion v8.1 deployment trajectory generator",
        f"verdict: {verdict}",
        f"deployment_path_id: {data.deployment_path_id}",
        f"input_path_name: {data.input_path_name}",
        f"sampling_seed: {args.sampling_seed}",
        f"rejection_reasons: {rejection_reasons}",
        "Robot-aware score, Cartesian error delta, accepted-step rate, and fallback rate are reported but not independent rejection gates.",
    ]
    atomic_write_text(args.output_dir / "deployment_report.txt", "\n".join(report) + "\n")
    save_plots(args.output_dir, data, result, velocity, acceleration, jerk, cartesian_error, manipulability)
    if safe:
        atomic_write_csv(
            args.output_dir / "approved_simulation_trajectory.csv",
            approved_rows(data.timestamps, result.rollout_q),
        )
        atomic_save_npz(
            args.output_dir / "approved_simulation_trajectory.npz",
            timestamps=data.timestamps,
            q=result.rollout_q,
            deployment_path_id=data.deployment_path_id,
            verdict=verdict,
            urdf_path=str(resolved_urdf),
            urdf_sha256=urdf_sha256,
        )
    print(verdict)
    return 0 if safe else 2


if __name__ == "__main__":
    raise SystemExit(main())
