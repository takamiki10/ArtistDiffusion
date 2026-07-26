#!/usr/bin/env python3
"""Validate outputs from generate_joint_trajectory_diffusion_v8_1.py."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import numpy as np

import evaluate_diffusion_v8_1_anchored_recursive_jerk_guard as v81
import evaluate_diffusion_v8_anchored_recursive_rollout as v8
import evaluate_diffusion_v7_teacher_forced_validation as v7_evaluator


TRAJECTORY_LENGTH = 100
JOINT_DIM = 6
XYZ_DIM = 3
FROZEN = {
    "checkpoint_state": "raw_last_epoch187",
    "target_scale": 1.0,
    "output_alpha": 0.125,
    "k": 8,
    "ddim_steps": 50,
    "eta": 0.0,
    "horizon": 32,
    "execution_horizon": 8,
    "anchoring_horizon": 8,
    "history_aware_jerk_tolerance": 1.0e-12,
}
ACCEPTED = "V8_1_DEPLOYMENT_TRAJECTORY_ACCEPTED"
REJECTED = "V8_1_DEPLOYMENT_TRAJECTORY_REJECTED"
REQUIRED_SAFETY_KEYS = (
    "full_path_safety_pass",
    "maximum_actual_internal_joint_step_rad",
    "rollout_full_hard_joint_limit_violation_count",
    "rollout_full_hard_joint_limit_violation_magnitude",
    "internal_full_path_robot_aware_delta_score",
    "cartesian_mean_error_delta",
    "internal_robot_score_contribution_jerk",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--require_accepted", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def strict_json(path: Path) -> Mapping[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"{path}: non-strict JSON constant {value}")

    return json.loads(path.read_text(), parse_constant=reject_constant)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def module_file(module: Any, label: str) -> Path:
    module_file_value = getattr(module, "__file__", None)
    if module_file_value is None:
        raise ValueError(f"Cannot resolve {label} module file for provenance validation")
    return Path(str(module_file_value)).resolve()


def require_files(output_dir: Path) -> None:
    required = (
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
    )
    missing = [name for name in required if not (output_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing deployment output files: {missing}")


def finite_array(name: str, array: np.ndarray, shape: Sequence[int]) -> np.ndarray:
    values = np.asarray(array)
    if values.shape != tuple(shape):
        raise ValueError(f"{name} has shape {values.shape}; expected {tuple(shape)}")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} contains nonfinite values")
    return values


def string_value(value: Any) -> str:
    array = np.asarray(value)
    if array.shape == ():
        item = array.item()
        if isinstance(item, bytes):
            return item.decode("utf-8")
        return str(item)
    if array.size == 1:
        item = array.reshape(-1)[0]
        if isinstance(item, bytes):
            return item.decode("utf-8")
        return str(item)
    return str(value)


def number(value: Any) -> float:
    return float(value)


def list_value(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, tuple):
        return [str(item) for item in value]
    if value in (None, ""):
        return []
    return [str(value)]


def require_metric(metrics: Mapping[str, Any], key: str, *, integer: bool = False) -> float:
    if key not in metrics:
        raise KeyError(f"Missing required safety metric: {key}")
    value = number(metrics[key])
    if not np.isfinite(value):
        raise ValueError(f"Required safety metric is nonfinite: {key}")
    if integer and int(value) != value:
        raise ValueError(f"Required safety metric is not integer-like: {key}")
    return value


def validate_frozen_config(metrics: Mapping[str, Any], npz: Mapping[str, Any]) -> None:
    for key, expected in FROZEN.items():
        if isinstance(expected, str):
            metric_value = str(metrics[key])
            npz_value = string_value(npz[key])
        else:
            metric_value = number(metrics[key])
            npz_value = number(np.asarray(npz[key]).item())
        if metric_value != expected:
            raise ValueError(f"metrics {key}={metric_value}; expected {expected}")
        if npz_value != expected:
            raise ValueError(f"npz {key}={npz_value}; expected {expected}")


def validate_csv_agreement(output_dir: Path, npz: Mapping[str, Any]) -> None:
    trajectory = read_csv(output_dir / "deployment_trajectory.csv")
    positions = read_csv(output_dir / "deployment_joint_positions.csv")
    dynamics = read_csv(output_dir / "deployment_joint_dynamics.csv")
    tracking = read_csv(output_dir / "deployment_cartesian_tracking.csv")
    if len(trajectory) != TRAJECTORY_LENGTH or len(positions) != TRAJECTORY_LENGTH:
        raise ValueError("Trajectory/position CSV row count must be 100")
    if len(dynamics) != TRAJECTORY_LENGTH or len(tracking) != TRAJECTORY_LENGTH:
        raise ValueError("Joint dynamics/tracking CSV row count must be 100")
    timestamps = np.asarray(npz["timestamps"], dtype=np.float64)
    desired = np.asarray(npz["desired_path"], dtype=np.float64)
    prior_q = np.asarray(npz["strong_prior_q"], dtype=np.float64)
    prior_ee = np.asarray(npz["strong_prior_ee"], dtype=np.float64)
    final_q = np.asarray(npz["final_q"], dtype=np.float64)
    final_ee = np.asarray(npz["final_ee"], dtype=np.float64)
    velocity = np.asarray(npz["joint_velocity"], dtype=np.float64)
    acceleration = np.asarray(npz["joint_acceleration"], dtype=np.float64)
    jerk = np.asarray(npz["joint_jerk"], dtype=np.float64)
    manipulability = np.asarray(npz["manipulability"], dtype=np.float64)
    min_singular = np.asarray(
        npz["minimum_translational_jacobian_singular_value"], dtype=np.float64
    )
    cartesian_error = np.linalg.norm(final_ee - desired, axis=1)
    executed_source = np.asarray(npz["executed_source"]).astype(str)
    for index in range(TRAJECTORY_LENGTH):
        for label, rows in (
            ("deployment_trajectory.csv", trajectory),
            ("deployment_joint_positions.csv", positions),
            ("deployment_joint_dynamics.csv", dynamics),
            ("deployment_cartesian_tracking.csv", tracking),
        ):
            if int(rows[index]["sample_index"]) != index:
                raise ValueError(f"{label} sample_index must be exactly 0..99")
            if abs(number(rows[index]["time_seconds"]) - timestamps[index]) > 1.0e-9:
                raise ValueError(f"{label} timestamps disagree with NPZ")
        if abs(number(trajectory[index]["time_seconds"]) - timestamps[index]) > 1.0e-9:
            raise ValueError("deployment_trajectory.csv timestamps disagree with NPZ")
        if abs(number(tracking[index]["time_seconds"]) - timestamps[index]) > 1.0e-9:
            raise ValueError("deployment_cartesian_tracking.csv timestamps disagree with NPZ")
        for joint in range(JOINT_DIM):
            if abs(number(trajectory[index][f"prior_q{joint + 1}"]) - prior_q[index, joint]) > 1.0e-8:
                raise ValueError("deployment_trajectory.csv prior_q disagrees with NPZ")
            if abs(number(trajectory[index][f"final_q{joint + 1}"]) - final_q[index, joint]) > 1.0e-8:
                raise ValueError("deployment_trajectory.csv final_q disagrees with NPZ")
            if abs(number(positions[index][f"q{joint + 1}"]) - final_q[index, joint]) > 1.0e-8:
                raise ValueError("deployment_joint_positions.csv q disagrees with NPZ")
            if abs(number(dynamics[index][f"dq{joint + 1}"]) - velocity[index, joint]) > 1.0e-8:
                raise ValueError("deployment_joint_dynamics.csv dq disagrees with NPZ")
            if abs(number(dynamics[index][f"ddq{joint + 1}"]) - acceleration[index, joint]) > 1.0e-8:
                raise ValueError("deployment_joint_dynamics.csv ddq disagrees with NPZ")
            if abs(number(dynamics[index][f"dddq{joint + 1}"]) - jerk[index, joint]) > 1.0e-8:
                raise ValueError("deployment_joint_dynamics.csv dddq disagrees with NPZ")
        for axis, column in enumerate(("desired_x", "desired_y", "desired_z")):
            if abs(number(trajectory[index][column]) - desired[index, axis]) > 1.0e-8:
                raise ValueError("deployment_trajectory.csv desired path disagrees with NPZ")
            if abs(number(tracking[index][column]) - desired[index, axis]) > 1.0e-8:
                raise ValueError("deployment_cartesian_tracking.csv desired path disagrees with NPZ")
        for axis, column in enumerate(("prior_x", "prior_y", "prior_z")):
            if abs(number(trajectory[index][column]) - prior_ee[index, axis]) > 1.0e-8:
                raise ValueError("deployment_trajectory.csv prior FK disagrees with NPZ")
        for axis, column in enumerate(("final_x", "final_y", "final_z")):
            if abs(number(trajectory[index][column]) - final_ee[index, axis]) > 1.0e-8:
                raise ValueError("deployment_trajectory.csv final FK disagrees with NPZ")
            if abs(number(tracking[index][column]) - final_ee[index, axis]) > 1.0e-8:
                raise ValueError("deployment_cartesian_tracking.csv final FK disagrees with NPZ")
        if abs(number(trajectory[index]["cartesian_error"]) - cartesian_error[index]) > 1.0e-8:
            raise ValueError("deployment_trajectory.csv Cartesian error disagrees with NPZ")
        if abs(number(tracking[index]["cartesian_error"]) - cartesian_error[index]) > 1.0e-8:
            raise ValueError("deployment_cartesian_tracking.csv Cartesian error disagrees with NPZ")
        if abs(number(tracking[index]["manipulability"]) - manipulability[index]) > 1.0e-8:
            raise ValueError("deployment_cartesian_tracking.csv manipulability disagrees with NPZ")
        if abs(number(tracking[index]["minimum_translational_jacobian_singular_value"]) - min_singular[index]) > 1.0e-8:
            raise ValueError("deployment_cartesian_tracking.csv singular value disagrees with NPZ")
        if str(trajectory[index]["execution_source"]) != executed_source[index]:
            raise ValueError("deployment_trajectory.csv execution_source disagrees with NPZ")
        accepted = int(executed_source[index] == "accepted_diffusion_candidate")
        fallback = int(executed_source[index] == "anchored_prior_fallback")
        if int(trajectory[index]["accepted_diffusion"]) != accepted:
            raise ValueError("deployment_trajectory.csv accepted flag disagrees with NPZ")
        if int(trajectory[index]["fallback"]) != fallback:
            raise ValueError("deployment_trajectory.csv fallback flag disagrees with NPZ")


def validate_verdict(output_dir: Path, metrics: Mapping[str, Any]) -> None:
    verdict = str(metrics["verdict"])
    if verdict not in {ACCEPTED, REJECTED}:
        raise ValueError(f"Unexpected verdict: {verdict}")
    approved_csv = output_dir / "approved_simulation_trajectory.csv"
    approved_npz = output_dir / "approved_simulation_trajectory.npz"
    if verdict == ACCEPTED:
        if not approved_csv.is_file() or not approved_npz.is_file():
            raise ValueError("Accepted verdict requires approved simulation exports")
    else:
        if approved_csv.exists() or approved_npz.exists():
            raise ValueError("Rejected verdict must not produce approved simulation exports")
    safety_flags = (
        int(metrics.get("full_path_safety_pass", 0)) == 1,
        float(metrics.get("maximum_actual_internal_joint_step_rad", 1.0)) <= 0.20,
    )
    if verdict == ACCEPTED and not all(safety_flags):
        raise ValueError("Accepted verdict is inconsistent with safety flags")


def validate_binary_mask(name: str, value: Any) -> np.ndarray:
    raw = np.asarray(value)
    if not np.all((raw == 0) | (raw == 1)):
        raise ValueError(f"{name} values must be boolean or exactly 0/1")
    return raw.astype(bool)


def validate_segment_arrays(npz: Mapping[str, Any], metrics: Mapping[str, Any]) -> int:
    starts = np.asarray(npz["window_start_indices"], dtype=np.int64)
    accepted = validate_binary_mask("accepted_step_mask", npz["accepted_step_mask"])
    fallback = validate_binary_mask("fallback_step_mask", npz["fallback_step_mask"])
    selected = np.asarray(npz["selected_candidate_indices"], dtype=np.int64)
    executed_indices = np.asarray(npz["executed_indices"], dtype=np.int64)
    finite_array("applied_correction_norms", npz["applied_correction_norms"], (TRAJECTORY_LENGTH,))
    if np.asarray(npz["executed_source"]).shape != (TRAJECTORY_LENGTH,):
        raise ValueError("executed_source must have shape (100,)")
    if starts.ndim != 1 or accepted.ndim != 1 or fallback.ndim != 1 or selected.ndim != 1:
        raise ValueError("segment arrays must be one-dimensional")
    if not (len(starts) == len(accepted) == len(fallback) == len(selected)):
        raise ValueError("segment arrays must have equal length")
    if len(starts) == 0 or starts[0] != 0:
        raise ValueError("window_start_indices must start at 0")
    if np.any(np.diff(starts) <= 0):
        raise ValueError("window_start_indices must be strictly increasing")
    if np.any(starts < 0) or np.any(starts >= TRAJECTORY_LENGTH):
        raise ValueError("window_start_indices out of bounds")
    if not np.array_equal(accepted, np.logical_not(fallback)):
        raise ValueError("accepted/fallback masks must be complements")
    if np.any((selected < -1) | (selected >= 8)):
        raise ValueError("selected_candidate_indices must be -1 or 0..7")
    if not np.all(selected[fallback] == -1):
        raise ValueError("fallback segments must have selected index -1")
    if not np.all((selected[accepted] >= 0) & (selected[accepted] <= 7)):
        raise ValueError("accepted segments must have selected index 0..7")
    if not np.array_equal(executed_indices, np.arange(TRAJECTORY_LENGTH)):
        raise ValueError("executed_indices must be exactly 0..99")
    segment_count = int(len(starts))
    accepted_count = int(np.sum(accepted))
    fallback_count = int(np.sum(fallback))
    if accepted_count + fallback_count != segment_count:
        raise ValueError("selected diffusion + fallback segment counts must equal segment count")
    expected_counts = {
        "rollout_segment_count": segment_count,
        "selected_diffusion_segment_count": accepted_count,
        "fallback_segment_count": fallback_count,
    }
    for key, expected in expected_counts.items():
        if int(metrics[key]) != expected:
            raise ValueError(f"metrics {key}={metrics[key]}; expected {expected}")
    expected_rates = {
        "accepted_rollout_step_rate": accepted_count / segment_count,
        "fallback_rate": fallback_count / segment_count,
    }
    for key, expected in expected_rates.items():
        if abs(float(metrics[key]) - expected) > 1.0e-12:
            raise ValueError(f"metrics {key}={metrics[key]}; expected {expected}")
    return segment_count


def validate_decision_candidate_rows(
    output_dir: Path,
    metrics: Mapping[str, Any],
    npz: Mapping[str, Any],
    segment_count: int,
) -> None:
    decisions = read_csv(output_dir / "deployment_segment_decisions.csv")
    candidates = read_csv(output_dir / "deployment_candidate_results.csv")
    if len(decisions) != segment_count:
        raise ValueError("decision row count must equal segment count")
    starts = np.asarray(npz["window_start_indices"], dtype=np.int64)
    accepted_mask = validate_binary_mask("accepted_step_mask", npz["accepted_step_mask"])
    fallback_mask = validate_binary_mask("fallback_step_mask", npz["fallback_step_mask"])
    selected_indices = np.asarray(npz["selected_candidate_indices"], dtype=np.int64)
    executed_source = np.asarray(npz["executed_source"]).astype(str)
    if executed_source[0] != "initial_prior_state":
        raise ValueError("sample zero executed_source must be initial_prior_state")
    common = {
        "sampling_seed": str(int(metrics["sampling_seed"])),
        "deployment_path_id": str(metrics["deployment_path_id"]),
        "input_path_name": str(metrics["input_path_name"]),
        "input_sha256": str(metrics["input_sha256"]),
        "verdict": str(metrics["verdict"]),
    }
    for row in [*decisions, *candidates]:
        for field, expected in common.items():
            if field not in row:
                raise ValueError(f"Missing provenance field in row: {field}")
            if str(row[field]) != expected:
                raise ValueError(f"Row provenance field {field} disagrees with metrics")
    decision_by_step: Dict[int, Mapping[str, str]] = {}
    for row in decisions:
        step = int(row["rollout_step"])
        if step in decision_by_step:
            raise ValueError(f"Duplicate decision rollout_step: {step}")
        decision_by_step[step] = row
    expected_steps = set(range(segment_count))
    if set(decision_by_step) != expected_steps:
        raise ValueError("Decision rollout_step set must be exactly 0..segment_count-1")
    by_segment: Dict[int, List[Mapping[str, str]]] = {}
    for row in candidates:
        if int(row["k"]) != 8:
            raise ValueError("Deployment candidate rows must have k=8")
        candidate_index = int(row["candidate_index"])
        if candidate_index < 0 or candidate_index > 7:
            raise ValueError("candidate_index must be 0..7")
        by_segment.setdefault(int(row["rollout_step"]), []).append(row)
    if set(by_segment) != expected_steps:
        raise ValueError("Candidate rollout_step set must be exactly 0..segment_count-1")
    covered_samples: List[int] = []
    for step in range(segment_count):
        decision = decision_by_step[step]
        if int(decision["window_start_index"]) != int(starts[step]):
            raise ValueError("decision window_start_index disagrees with NPZ")
        selected = int(decision["selected_candidate_index"])
        accepted = int(decision["accepted"])
        fallback = int(decision["fallback"])
        if accepted not in (0, 1) or fallback not in (0, 1):
            raise ValueError("decision accepted/fallback flags must be exactly 0/1")
        if accepted + fallback != 1:
            raise ValueError("decision accepted + fallback must equal 1")
        if bool(accepted) != bool(accepted_mask[step]):
            raise ValueError("decision accepted flag disagrees with NPZ")
        if bool(fallback) != bool(fallback_mask[step]):
            raise ValueError("decision fallback flag disagrees with NPZ")
        if selected != int(selected_indices[step]):
            raise ValueError("decision selected index disagrees with NPZ")
        if accepted and not (0 <= selected <= 7):
            raise ValueError("accepted decision selected_candidate_index must be 0..7")
        if fallback and selected != -1:
            raise ValueError("fallback decision selected_candidate_index must be -1")
        execution_start = int(decision["execution_start_index"])
        execution_end = int(decision["execution_end_index"])
        executed_count = int(decision["executed_count"])
        expected_start = int(starts[step]) + 1
        expected_count = min(int(metrics["execution_horizon"]), TRAJECTORY_LENGTH - 1 - int(starts[step]))
        if execution_start != expected_start:
            raise ValueError("decision execution_start_index disagrees with window_start_index")
        if executed_count != expected_count:
            raise ValueError("decision executed_count disagrees with frozen execution horizon")
        if execution_end != execution_start + executed_count - 1:
            raise ValueError("decision execution_end_index disagrees with executed_count")
        covered_samples.extend(range(execution_start, execution_end + 1))
        expected_source = (
            "accepted_diffusion_candidate" if accepted else "anchored_prior_fallback"
        )
        for sample_index in range(execution_start, execution_end + 1):
            if executed_source[sample_index] != expected_source:
                raise ValueError("executed_source disagrees with decision segment source")
        rows = by_segment.get(step, [])
        if len(rows) != 8:
            raise ValueError(f"rollout_step {step} must have exactly eight candidate rows")
        candidate_indices = sorted(int(row["candidate_index"]) for row in rows)
        if candidate_indices != list(range(8)):
            raise ValueError(f"rollout_step {step} candidate indices must be exactly 0..7")
        for row in rows:
            if int(row["selected"]) not in (0, 1):
                raise ValueError("candidate selected flags must be exactly 0/1")
        selected_rows = [row for row in rows if int(row["selected"]) == 1]
        if selected == -1:
            if selected_rows:
                raise ValueError("fallback segment has selected candidate row")
        else:
            if len(selected_rows) != 1 or int(selected_rows[0]["candidate_index"]) != selected:
                raise ValueError("candidate selected flags disagree with decision/NPZ")
    if covered_samples != list(range(1, TRAJECTORY_LENGTH)):
        raise ValueError("decision execution ranges must cover samples 1..99 contiguously")


def validate_approved_exports(output_dir: Path, metrics: Mapping[str, Any], npz: Mapping[str, Any]) -> None:
    verdict = str(metrics["verdict"])
    csv_path = output_dir / "approved_simulation_trajectory.csv"
    npz_path = output_dir / "approved_simulation_trajectory.npz"
    timestamps = np.asarray(npz["timestamps"], dtype=np.float64)
    final_q = np.asarray(npz["final_q"], dtype=np.float64)
    if verdict == ACCEPTED:
        rows = read_csv(csv_path)
        if len(rows) != TRAJECTORY_LENGTH:
            raise ValueError("approved_simulation_trajectory.csv must have 100 rows")
        for index, row in enumerate(rows):
            if abs(number(row["time_seconds"]) - timestamps[index]) > 1.0e-9:
                raise ValueError("approved CSV timestamps disagree with NPZ")
            for joint in range(JOINT_DIM):
                if abs(number(row[f"q{joint + 1}"]) - final_q[index, joint]) > 1.0e-8:
                    raise ValueError("approved CSV q disagrees with NPZ")
        with np.load(npz_path, allow_pickle=False) as approved:
            finite_array("approved timestamps", approved["timestamps"], (TRAJECTORY_LENGTH,))
            finite_array("approved q", approved["q"], (TRAJECTORY_LENGTH, JOINT_DIM))
            if not np.allclose(approved["timestamps"], timestamps):
                raise ValueError("approved NPZ timestamps disagree with full NPZ")
            if not np.allclose(approved["q"], final_q):
                raise ValueError("approved NPZ q disagrees with full NPZ")
            for field in ("deployment_path_id", "verdict", "urdf_path", "urdf_sha256"):
                if string_value(approved[field]) != str(metrics[field]):
                    raise ValueError(f"approved NPZ {field} disagrees with metrics")
    else:
        if csv_path.exists() or npz_path.exists():
            raise ValueError("Rejected verdict must not include approved exports")


def recompute_dynamics(final_q: np.ndarray, timestamps: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    velocity = np.gradient(final_q, timestamps, axis=0, edge_order=2)
    acceleration = np.gradient(velocity, timestamps, axis=0, edge_order=2)
    jerk = np.gradient(acceleration, timestamps, axis=0, edge_order=2)
    return velocity, acceleration, jerk


def compare_metric(metrics: Mapping[str, Any], recomputed: Mapping[str, Any], key: str, *, integer: bool = False) -> None:
    reported = require_metric(metrics, key, integer=integer)
    actual = require_metric(recomputed, key, integer=integer)
    if integer:
        if int(reported) != int(actual):
            raise ValueError(f"{key} mismatch: reported={reported}, recomputed={actual}")
    elif abs(reported - actual) > 1.0e-8 * max(1.0, abs(actual)):
        raise ValueError(f"{key} mismatch: reported={reported}, recomputed={actual}")


def independent_prefix_safety(
    robot: Any,
    desired: np.ndarray,
    final_q: np.ndarray,
    final_ee: np.ndarray,
) -> Tuple[int, List[str]]:
    record = v8.PhysicalPathRecord(
        path_id="deployment_validation",
        path_index=0,
        population="ordinary",
        desired_path=desired,
        strong_prior_q=final_q,
        prior_ee=final_ee,
    )
    failures: List[str] = []
    count = 0
    start = 0
    while start < TRAJECTORY_LENGTH - 1:
        count += 1
        execution_count = min(8, TRAJECTORY_LENGTH - 1 - start)
        current_q = final_q[start]
        previous_q = final_q[start - 1] if start > 0 else None
        window_q = v8.padded_window(final_q, start, 32)
        desired_window = v8.padded_window(desired, start, 32)
        context = v8.make_action_context(
            record,
            start,
            current_q,
            previous_q,
            window_q,
            desired_window,
            robot,
            execution_count,
        )
        prefix_metrics = v7_evaluator.evaluate_metrics(
            robot, context, context.prior_q, execution_count
        )
        reasons = v8.recursive_executed_prefix_hard_safety_reasons(prefix_metrics)
        if reasons:
            failures.append(
                f"executed_prefix_hard_safety@{start}:{'|'.join(reasons)}"
            )
        start += execution_count
    return count, failures


def validate_input_copy(output_dir: Path, metrics: Mapping[str, Any], full: Mapping[str, Any]) -> None:
    with np.load(output_dir / "deployment_input_copy.npz", allow_pickle=False) as copied:
        for key, shape in (
            ("desired_path", (TRAJECTORY_LENGTH, XYZ_DIM)),
            ("strong_prior_q", (TRAJECTORY_LENGTH, JOINT_DIM)),
            ("strong_prior_ee", (TRAJECTORY_LENGTH, XYZ_DIM)),
            ("timestamps", (TRAJECTORY_LENGTH,)),
        ):
            values = finite_array(f"input copy {key}", copied[key], shape)
            if not np.allclose(values, np.asarray(full[key], dtype=np.float64)):
                raise ValueError(f"deployment_input_copy.npz {key} disagrees with full NPZ")
        if string_value(copied["path_name"]) != str(metrics["input_path_name"]):
            raise ValueError("deployment_input_copy.npz path_name disagrees with metrics")
        for field in ("input_sha256", "deployment_path_id", "urdf_path", "urdf_sha256"):
            if string_value(copied[field]) != str(metrics[field]):
                raise ValueError(f"deployment_input_copy.npz {field} disagrees with metrics")


def validate_provenance(metrics: Mapping[str, Any]) -> None:
    provenance = metrics.get("provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("deployment_metrics.json lacks provenance dictionary")
    required = (
        "generator_script_sha256",
        "frozen_v8_1_script_sha256",
        "frozen_v8_script_sha256",
        "training_dataset_manifest_or_path",
        "model_directory_file_hashes",
        "checkpoint_state_hash",
        "input_npz_sha256",
        "urdf_path",
        "urdf_sha256",
        "python_version",
        "numpy_version",
        "torch_version",
        "cuda_available",
        "git",
    )
    missing = [field for field in required if field not in provenance]
    if missing:
        raise ValueError(f"Missing provenance fields: {missing}")
    provenance_matches = {
        "training_dataset_manifest_or_path": "training_dataset_dir",
        "checkpoint_state_hash": "checkpoint_state_hash",
        "input_npz_sha256": "input_sha256",
        "urdf_path": "urdf_path",
        "urdf_sha256": "urdf_sha256",
    }
    for provenance_field, metrics_field in provenance_matches.items():
        if str(provenance[provenance_field]) != str(metrics[metrics_field]):
            raise ValueError(
                f"provenance {provenance_field} disagrees with metrics {metrics_field}"
            )
    for path_field, hash_field in (
        ("urdf_path", "urdf_sha256"),
    ):
        path = Path(str(provenance[path_field]))
        if path.is_file() and sha256_file(path) != str(provenance[hash_field]):
            raise ValueError(f"Provenance hash mismatch for {path_field}")
    script_checks = (
        (
            Path(__file__).resolve().with_name(
                "generate_joint_trajectory_diffusion_v8_1.py"
            ),
            "generator_script_sha256",
        ),
        (module_file(v81, "frozen v8.1"), "frozen_v8_1_script_sha256"),
        (module_file(v8, "frozen v8"), "frozen_v8_script_sha256"),
    )
    for path, field in script_checks:
        if path.is_file() and sha256_file(path) != str(provenance[field]):
            raise ValueError(f"Provenance script hash mismatch: {field}")
    model_hashes = provenance["model_directory_file_hashes"]
    if not isinstance(model_hashes, Mapping) or not model_hashes:
        raise ValueError("model_directory_file_hashes must be a nonempty dictionary")
    model_root = Path(str(metrics["model_dir"]))
    if not model_root.is_dir():
        raise FileNotFoundError(f"Recorded model_dir does not exist: {model_root}")
    for relative, expected in model_hashes.items():
        model_file = model_root / str(relative)
        if not model_file.is_file():
            raise FileNotFoundError(f"Model file listed in provenance is missing: {model_file}")
        if sha256_file(model_file) != str(expected):
            raise ValueError(f"Model file hash mismatch: {model_file}")


def validate_full_npz_metadata(metrics: Mapping[str, Any], npz: Mapping[str, Any]) -> None:
    scalar_fields = (
        "sampling_seed",
        "deployment_path_id",
        "input_path_name",
        "input_file",
        "input_sha256",
        "urdf_path",
        "urdf_sha256",
        "model_dir",
        "training_dataset_dir",
        "checkpoint_state",
        "checkpoint_state_hash",
        "target_scale",
        "output_alpha",
        "k",
        "ddim_steps",
        "eta",
        "horizon",
        "execution_horizon",
        "anchoring_horizon",
        "history_aware_jerk_tolerance",
        "verdict",
    )
    string_fields = {
        "deployment_path_id",
        "input_path_name",
        "input_file",
        "input_sha256",
        "urdf_path",
        "urdf_sha256",
        "model_dir",
        "training_dataset_dir",
        "checkpoint_state",
        "checkpoint_state_hash",
        "verdict",
    }
    for field in scalar_fields:
        if field not in metrics or field not in npz:
            raise ValueError(f"Missing scalar metadata field: {field}")
        if field in string_fields:
            if string_value(npz[field]) != str(metrics[field]):
                raise ValueError(f"full NPZ {field} disagrees with metrics JSON")
        else:
            if number(np.asarray(npz[field]).item()) != number(metrics[field]):
                raise ValueError(f"full NPZ {field} disagrees with metrics JSON")
    if string_value(npz["verdict"]) != str(metrics["verdict"]):
        raise ValueError("full NPZ verdict disagrees with metrics verdict")


def validate_independent_safety_fields(
    metrics: Mapping[str, Any],
    recomputed_metrics: Mapping[str, Any],
    timestamps: np.ndarray,
    final_q: np.ndarray,
    final_ee: np.ndarray,
    prefix_count: int,
    prefix_failures: Sequence[str],
) -> None:
    hard_count = require_metric(
        recomputed_metrics, "rollout_full_hard_joint_limit_violation_count", integer=True
    )
    hard_magnitude = require_metric(
        recomputed_metrics, "rollout_full_hard_joint_limit_violation_magnitude"
    )
    expected = {
        "independent_full_path_safety_pass": int(
            require_metric(recomputed_metrics, "full_path_safety_pass", integer=True) == 1
        ),
        "independent_executed_prefix_safety_pass": int(not prefix_failures),
        "independent_executed_prefix_check_count": int(prefix_count),
        "independent_executed_prefix_failure_count": int(len(prefix_failures)),
        "independent_joint_limit_pass": int(hard_count == 0 and hard_magnitude == 0.0),
        "independent_timestamp_pass": int(np.all(np.diff(timestamps) > 0.0)),
        "independent_finite_joint_pass": int(np.all(np.isfinite(final_q))),
        "independent_finite_fk_pass": int(np.all(np.isfinite(final_ee))),
    }
    for field, value in expected.items():
        if int(metrics[field]) != value:
            raise ValueError(f"{field} disagrees with independent recomputation")
    stored_reasons = list_value(metrics.get("independent_executed_prefix_failure_reasons"))
    recomputed_reasons = [str(reason) for reason in prefix_failures]
    if stored_reasons != recomputed_reasons:
        raise ValueError("independent_executed_prefix_failure_reasons mismatch")
    if str(metrics["verdict"]) == ACCEPTED and recomputed_reasons:
        raise ValueError("accepted trajectory has executed-prefix failure reasons")


def main() -> int:
    args = parse_args()
    require_files(args.output_dir)
    metrics = strict_json(args.output_dir / "deployment_metrics.json")
    with np.load(args.output_dir / "deployment_trajectory_full.npz", allow_pickle=False) as data:
        timestamps = finite_array("timestamps", data["timestamps"], (TRAJECTORY_LENGTH,))
        desired = finite_array("desired_path", data["desired_path"], (TRAJECTORY_LENGTH, XYZ_DIM))
        prior_q = finite_array("strong_prior_q", data["strong_prior_q"], (TRAJECTORY_LENGTH, JOINT_DIM))
        prior_ee = finite_array("strong_prior_ee", data["strong_prior_ee"], (TRAJECTORY_LENGTH, XYZ_DIM))
        final_q = finite_array("final_q", data["final_q"], (TRAJECTORY_LENGTH, JOINT_DIM))
        final_ee = finite_array("final_ee", data["final_ee"], (TRAJECTORY_LENGTH, XYZ_DIM))
        velocity = finite_array("joint_velocity", data["joint_velocity"], (TRAJECTORY_LENGTH, JOINT_DIM))
        acceleration = finite_array("joint_acceleration", data["joint_acceleration"], (TRAJECTORY_LENGTH, JOINT_DIM))
        jerk = finite_array("joint_jerk", data["joint_jerk"], (TRAJECTORY_LENGTH, JOINT_DIM))
        finite_array("manipulability", data["manipulability"], (TRAJECTORY_LENGTH,))
        finite_array(
            "minimum_translational_jacobian_singular_value",
            data["minimum_translational_jacobian_singular_value"],
            (TRAJECTORY_LENGTH,),
        )
        if not np.all(np.diff(timestamps) > 0.0):
            raise ValueError("timestamps are not strictly increasing")
        segment_count = validate_segment_arrays(data, metrics)
        if string_value(data["urdf_path"]) != str(metrics["urdf_path"]):
            raise ValueError("URDF path differs between metrics and full NPZ")
        if string_value(data["urdf_sha256"]) != str(metrics["urdf_sha256"]):
            raise ValueError("URDF hash differs between metrics and full NPZ")
        urdf_path = Path(str(metrics["urdf_path"])).resolve()
        if not urdf_path.is_file():
            raise FileNotFoundError(f"Recorded URDF does not exist: {urdf_path}")
        if sha256_file(urdf_path) != str(metrics["urdf_sha256"]):
            raise ValueError("Recorded URDF hash does not match local file")
        robot = v7_evaluator.make_robot_context(urdf_path)
        recomputed_prior_ee = v8.compute_fk_positions(robot, prior_q)
        if not np.allclose(recomputed_prior_ee, prior_ee, rtol=1.0e-5, atol=2.0e-5):
            raise ValueError("Stored prior FK does not match recomputation")
        recomputed_ee = v8.compute_fk_positions(robot, final_q)
        if not np.allclose(recomputed_ee, final_ee, rtol=1.0e-5, atol=2.0e-5):
            raise ValueError("Stored final FK does not match recomputation")
        recomputed_metrics = v8.compute_full_trajectory_metrics(
            robot=robot,
            strong_prior_q=prior_q,
            rollout_q=final_q,
            desired_path=desired,
        )
        recomputed_rollout_ee = np.asarray(recomputed_metrics.pop("rollout_ee"), dtype=np.float64)
        recomputed_metric_prior_ee = np.asarray(recomputed_metrics.pop("prior_ee"), dtype=np.float64)
        if not np.allclose(recomputed_rollout_ee, final_ee, rtol=1.0e-5, atol=2.0e-5):
            raise ValueError("Frozen full-path metric final FK disagrees with stored FK")
        if not np.allclose(recomputed_metric_prior_ee, prior_ee, rtol=1.0e-5, atol=2.0e-5):
            raise ValueError("Frozen full-path metric prior FK disagrees with stored FK")
        for key in REQUIRED_SAFETY_KEYS:
            compare_metric(
                metrics,
                recomputed_metrics,
                key,
                integer=key.endswith("_count") or key == "full_path_safety_pass",
            )
        max_step = float(np.max(np.abs(np.diff(final_q, axis=0))))
        if abs(max_step - float(metrics["maximum_actual_internal_joint_step_rad"])) > 1.0e-8:
            raise ValueError("Reported maximum internal joint step disagrees with recomputation")
        recomputed_velocity, recomputed_acceleration, recomputed_jerk = recompute_dynamics(final_q, timestamps)
        if not np.allclose(recomputed_velocity, velocity, rtol=1.0e-8, atol=1.0e-8):
            raise ValueError("Stored joint_velocity disagrees with recomputation")
        if not np.allclose(recomputed_acceleration, acceleration, rtol=1.0e-8, atol=1.0e-8):
            raise ValueError("Stored joint_acceleration disagrees with recomputation")
        if not np.allclose(recomputed_jerk, jerk, rtol=1.0e-8, atol=1.0e-8):
            raise ValueError("Stored joint_jerk disagrees with recomputation")
        prefix_count, prefix_failures = independent_prefix_safety(robot, desired, final_q, final_ee)
        validate_independent_safety_fields(
            metrics,
            recomputed_metrics,
            timestamps,
            final_q,
            final_ee,
            prefix_count,
            prefix_failures,
        )
        validate_frozen_config(metrics, data)
        validate_full_npz_metadata(metrics, data)
        validate_csv_agreement(args.output_dir, data)
        validate_decision_candidate_rows(args.output_dir, metrics, data, segment_count)
        validate_approved_exports(args.output_dir, metrics, data)
        validate_input_copy(args.output_dir, metrics, data)
        validate_provenance(metrics)
        for field in (
            "deployment_path_id",
            "input_path_name",
            "input_file",
            "input_sha256",
            "model_dir",
            "training_dataset_dir",
            "checkpoint_state",
            "checkpoint_state_hash",
            "sampling_seed",
        ):
            if field not in data or field not in metrics:
                raise ValueError(f"Missing provenance field: {field}")
    validate_verdict(args.output_dir, metrics)
    accepted_condition = (
        int(metrics["full_path_safety_pass"]) == 1
        and float(metrics["maximum_actual_internal_joint_step_rad"]) <= 0.20
        and int(metrics["rollout_full_hard_joint_limit_violation_count"]) == 0
        and float(metrics["rollout_full_hard_joint_limit_violation_magnitude"]) == 0.0
        and int(metrics["independent_full_path_safety_pass"]) == 1
        and int(metrics["independent_executed_prefix_safety_pass"]) == 1
        and int(metrics["independent_joint_limit_pass"]) == 1
        and int(metrics["independent_finite_joint_pass"]) == 1
        and int(metrics["independent_finite_fk_pass"]) == 1
        and int(metrics["independent_timestamp_pass"]) == 1
    )
    metrics_accepted = bool(metrics["accepted"])
    verdict_accepted = str(metrics["verdict"]) == ACCEPTED
    if verdict_accepted != metrics_accepted:
        raise ValueError("metrics accepted flag disagrees with verdict")
    if metrics_accepted != accepted_condition:
        raise ValueError("Stored accepted flag disagrees with independently derived safety")
    if args.require_accepted and str(metrics["verdict"]) != ACCEPTED:
        raise ValueError("--require_accepted was set but verdict is rejected")
    print("V8_1_DEPLOYMENT_OUTPUT_VALIDATION_PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
