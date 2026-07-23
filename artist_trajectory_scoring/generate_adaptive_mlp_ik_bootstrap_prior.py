#!/usr/bin/env python3
"""Generate a method-specific canonical-MLP plus adaptive-IK bootstrap prior.

Practical generation uses only the desired Cartesian path, canonical MLP
checkpoint, known starting configuration, robot model, and previous accepted IK
state. Expert joints are opened only after generation when the optional expert
evaluation flag is enabled.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from generate_ik_seed_path import (
    DEFAULT_EE_LINK,
    DEFAULT_JOINT_NAMES,
    DEFAULT_URDF_PATH,
    DEFAULT_JOINT_LIMIT_SAFETY_MARGIN_RAD,
    HARD_JOINT_LIMIT_TOLERANCE_RAD,
    IkAttempt,
    check_joint_limits,
    get_joint_bounds,
    load_robot,
    sample_uniform_q,
    solve_ik_from_initial_guess,
)
from predict_path_conditioned_mlp import (
    PathConditionedMLP,
    load_model,
    predict_q,
)
from refine_mlp_predictions_with_ik import resolve_urdf_path


DATA_ROOT = Path("data/cartesian_expert_dataset_v3")
DEFAULT_CHECKPOINT = DATA_ROOT / "path_conditioned_mlp_v3.pt"
DEFAULT_OUTPUT_ROOT = DATA_ROOT / "adaptive_mlp_ik_bootstrap_prior"
DEFAULT_TRAIN_NPZ = DATA_ROOT / "diffusion_v2/diffusion_train_v2.npz"
DEFAULT_TEST_NPZ = DATA_ROOT / "diffusion_v2/diffusion_test_v2.npz"
DEFAULT_OLD_TEST_ROOT = DATA_ROOT / "experts/test"

JOINT_DIM = 6
EXPECTED_STEPS = 100
JOINT_COLUMNS = ("q1", "q2", "q3", "q4", "q5", "q6")
XYZ_COLUMNS = ("x", "y", "z")
RANGE_EPS = 1.0e-10

# These are the exact two stages defined by adaptive_refine_mlp_predictions_with_ik.py.
ADAPTIVE_STAGE1_SMOOTH_WEIGHT = 0.01
ADAPTIVE_STAGE1_MAX_ITERS = 200
ADAPTIVE_STAGE2_SMOOTH_WEIGHT = 0.001
ADAPTIVE_STAGE2_MAX_ITERS = 500
ADAPTIVE_STAGE2_MAX_ERROR_THRESHOLD = 0.03

# The adaptive wrapper does not define these. They come from the repository's
# sequential IK implementation and are recorded as such in output metadata.
IK_FTOL = 1.0e-10
IK_CARTESIAN_TOLERANCE = 0.02
SEQUENTIAL_IK_DEFAULT_RETRIES = 8


@dataclass(frozen=True)
class StageParameters:
    stage: int
    smooth_weight: float
    max_iters: int
    ftol: float
    cartesian_tolerance: float


@dataclass(frozen=True)
class GenerationDataset:
    source: Path
    names: Tuple[str, ...]
    desired_paths: np.ndarray
    times: np.ndarray
    q_start: np.ndarray
    times_source: str


@dataclass
class StageResult:
    q: np.ndarray
    ee: np.ndarray
    step_records: List[Dict[str, Any]]
    ik_success_count: int
    ik_retry_count: int
    failed_timestep_count: int
    ik_fallback_timestep_count: int
    strict_step_failure_count: int
    runtime_sec: float
    parameters: StageParameters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate canonical full-q MLP trajectories and refine them with "
            "adaptive sequential IK."
        )
    )
    parser.add_argument("--split", choices=("train", "test"), required=True)
    parser.add_argument(
        "--input_npz",
        type=Path,
        default=None,
        help="Defaults to the diffusion-v2 NPZ for the selected split.",
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--path_indices", nargs="+", type=int, default=None)
    parser.add_argument("--path_names", nargs="+", default=None)
    parser.add_argument(
        "--max_paths",
        type=int,
        default=0,
        help="Maximum selected paths; zero means every selected path.",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--urdf", type=Path, default=Path(DEFAULT_URDF_PATH))
    parser.add_argument("--ee_link", default=DEFAULT_EE_LINK)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--overwrite_selected",
        action="store_true",
        help="Regenerate only explicitly selected paths while preserving all others.",
    )
    parser.add_argument(
        "--retry_profile",
        choices=("standard", "robust"),
        default="standard",
    )
    parser.add_argument("--mean_error_gate", type=float, default=0.01)
    parser.add_argument("--max_joint_step_gate", type=float, default=0.20)
    parser.add_argument("--local_repair", action="store_true")
    parser.add_argument("--local_repair_radius", type=int, default=4)
    parser.add_argument("--local_repair_max_passes", type=int, default=3)
    parser.add_argument("--bridge_step_target", type=float, default=0.18)
    parser.add_argument("--joint_limit_repair", action="store_true")
    parser.add_argument(
        "--joint_limit_margin",
        type=float,
        default=DEFAULT_JOINT_LIMIT_SAFETY_MARGIN_RAD,
    )
    parser.add_argument("--joint_limit_repair_radius", type=int, default=4)
    parser.add_argument("--joint_limit_repair_passes", type=int, default=3)
    parser.add_argument("--save_expert_evaluation", action="store_true")
    parser.add_argument("--max_allowed_joint_step", type=float, default=0.2)
    parser.add_argument("--strict_joint_step", action="store_true")
    parser.add_argument(
        "--num_ik_retries",
        type=int,
        default=None,
        help=(
            "Additional seeds after a failed primary solve. The resolved default "
            "is 8 from generate_ik_seed_path.py; the adaptive wrapper has no value."
        ),
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")
    if args.overwrite_selected and args.path_names is None and args.path_indices is None:
        raise ValueError(
            "--overwrite_selected requires --path_names or --path_indices"
        )
    if args.max_paths < 0:
        raise ValueError("--max_paths must be zero or positive")
    if args.max_allowed_joint_step <= 0.0:
        raise ValueError("--max_allowed_joint_step must be positive")
    if args.mean_error_gate <= 0.0:
        raise ValueError("--mean_error_gate must be positive")
    if args.max_joint_step_gate <= 0.0:
        raise ValueError("--max_joint_step_gate must be positive")
    if args.local_repair_radius < 0:
        raise ValueError("--local_repair_radius must be non-negative")
    if args.local_repair_max_passes <= 0:
        raise ValueError("--local_repair_max_passes must be positive")
    if args.bridge_step_target <= 0.0:
        raise ValueError("--bridge_step_target must be positive")
    if args.bridge_step_target > args.max_joint_step_gate:
        raise ValueError(
            "--bridge_step_target must not exceed --max_joint_step_gate"
        )
    if args.joint_limit_margin < 0.0:
        raise ValueError("--joint_limit_margin must be non-negative")
    if args.joint_limit_repair_radius < 0:
        raise ValueError("--joint_limit_repair_radius must be non-negative")
    if args.joint_limit_repair_passes <= 0:
        raise ValueError("--joint_limit_repair_passes must be positive")
    if args.num_ik_retries is not None and args.num_ik_retries < 0:
        raise ValueError("--num_ik_retries must be non-negative")
    if args.path_indices is not None and len(set(args.path_indices)) != len(
        args.path_indices
    ):
        raise ValueError("--path_indices contains duplicate values")
    if args.path_names is not None and len(set(args.path_names)) != len(
        args.path_names
    ):
        raise ValueError("--path_names contains duplicate values")


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(requested)


def decode_names(values: np.ndarray) -> List[str]:
    names: List[str] = []
    for value in np.asarray(values).reshape(-1):
        if isinstance(value, bytes):
            names.append(value.decode("utf-8", errors="replace"))
        else:
            names.append(str(value))
    return names


def safe_path_name(name: str) -> str:
    result = Path(str(name)).name.replace("/", "_").replace("\\", "_")
    return result or "unnamed_path"


def default_input_npz(split: str) -> Path:
    return DEFAULT_TRAIN_NPZ if split == "train" else DEFAULT_TEST_NPZ


def load_generation_dataset(path: Path) -> GenerationDataset:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=True) as archive:
        missing = [
            key
            for key in ("desired_paths", "path_names", "q_start")
            if key not in archive.files
        ]
        if missing:
            raise KeyError(f"{path} is missing generation keys {missing}")
        desired = np.asarray(archive["desired_paths"], dtype=np.float32)
        names = decode_names(archive["path_names"])
        q_start = np.asarray(archive["q_start"], dtype=np.float64)
        if "times" in archive.files:
            raw_times = np.asarray(archive["times"], dtype=np.float32)
            if raw_times.ndim == 1 and desired.ndim == 3:
                times = np.repeat(raw_times[None, :], desired.shape[0], axis=0)
            else:
                times = raw_times
            times_source = f"{path}:times"
        else:
            if desired.ndim != 3:
                raise ValueError(f"desired_paths must be [N,T,3], got {desired.shape}")
            unit_time = np.linspace(0.0, 1.0, desired.shape[1], dtype=np.float32)
            times = np.repeat(unit_time[None, :], desired.shape[0], axis=0)
            times_source = "np.linspace(0,1,T); input NPZ has no times key"
    if desired.ndim != 3 or desired.shape[2] != 3:
        raise ValueError(f"desired_paths must be [N,T,3], got {desired.shape}")
    if desired.shape[1] != EXPECTED_STEPS:
        raise ValueError(f"Expected T={EXPECTED_STEPS}, got T={desired.shape[1]}")
    if len(names) != desired.shape[0]:
        raise ValueError("path_names length does not match desired_paths")
    if len(set(names)) != len(names):
        raise ValueError("path_names contains duplicates")
    if times.shape != desired.shape[:2]:
        raise ValueError(f"times must have shape {desired.shape[:2]}, got {times.shape}")
    if q_start.shape != (desired.shape[0], JOINT_DIM):
        raise ValueError(f"q_start must have shape ({desired.shape[0]},6)")
    if not np.all(np.isfinite(desired)) or not np.all(np.isfinite(times)):
        raise ValueError("desired_paths or times contains nonfinite values")
    if not np.all(np.isfinite(q_start)):
        raise ValueError("q_start contains nonfinite values")
    return GenerationDataset(
        source=path,
        names=tuple(names),
        desired_paths=desired,
        times=times,
        q_start=q_start,
        times_source=times_source,
    )


def select_indices(dataset: GenerationDataset, args: argparse.Namespace) -> List[int]:
    selected = list(range(len(dataset.names)))
    if args.path_indices is not None:
        invalid = [
            index
            for index in args.path_indices
            if index < 0 or index >= len(dataset.names)
        ]
        if invalid:
            raise IndexError(f"--path_indices out of range: {invalid}")
        requested = set(args.path_indices)
        selected = [index for index in selected if index in requested]
    if args.path_names is not None:
        unknown = sorted(set(args.path_names) - set(dataset.names))
        if unknown:
            raise KeyError(f"Unknown --path_names: {unknown}")
        requested_names = set(args.path_names)
        selected = [index for index in selected if dataset.names[index] in requested_names]
    if args.max_paths > 0:
        selected = selected[: args.max_paths]
    if not selected:
        raise ValueError("Path filters selected no trajectories")
    return selected


def checkpoint_target_interpretation(
    checkpoint: Mapping[str, Any], checkpoint_path: Path
) -> Tuple[str, str]:
    explicit_delta: Optional[bool] = None
    for key in ("predicts_delta_q", "predict_delta_q", "is_delta_target"):
        if key in checkpoint:
            explicit_delta = bool(checkpoint[key])
            break
    for key in ("target_type", "target_convention", "prediction_target"):
        if key not in checkpoint:
            continue
        value = str(checkpoint[key]).strip().lower()
        if value in {"delta", "delta_q", "joint_delta", "residual"}:
            explicit_delta = True
        elif value in {"q", "full_q", "joint_position", "actions"}:
            explicit_delta = False
        else:
            raise ValueError(f"Unrecognized checkpoint {key}={checkpoint[key]!r}")
        break
    if explicit_delta is True:
        interpretation = "delta_q"
        provenance = "explicit checkpoint target metadata"
    elif explicit_delta is False:
        interpretation = "full_q"
        provenance = "explicit checkpoint target metadata"
    else:
        if str(checkpoint.get("model_type", "")) != "path_conditioned_mlp":
            raise ValueError(
                "Checkpoint has no target metadata and is not model_type=path_conditioned_mlp"
            )
        interpretation = "full_q"
        provenance = (
            "canonical train_path_conditioned_mlp.py target is actions/full q; "
            "predict_path_conditioned_mlp.py directly applies y_std/y_mean"
        )
    if checkpoint_path.name == "path_conditioned_mlp_v3.pt" and interpretation != "full_q":
        raise AssertionError(
            "Known path_conditioned_mlp_v3.pt must use canonical full-q interpretation"
        )
    return interpretation, provenance


def canonical_mlp_full_q(
    *,
    model: PathConditionedMLP,
    checkpoint: Mapping[str, Any],
    checkpoint_path: Path,
    times: np.ndarray,
    desired_path: np.ndarray,
    q_start: np.ndarray,
    device: torch.device,
) -> Tuple[np.ndarray, Dict[str, str]]:
    denormalized_target = predict_q(
        model,
        dict(checkpoint),
        times.astype(np.float32),
        desired_path.astype(np.float32),
        device,
    ).astype(np.float64)
    interpretation, provenance = checkpoint_target_interpretation(
        checkpoint, checkpoint_path
    )
    if interpretation == "delta_q":
        q_mlp = q_start[None, :] + denormalized_target
        formula = "q_start + (model_output_normalized * y_std + y_mean)"
    else:
        q_mlp = denormalized_target
        formula = "model_output_normalized * y_std + y_mean"
    if q_mlp.shape != (int(checkpoint["num_steps"]), JOINT_DIM):
        raise ValueError(f"Canonical MLP output has invalid shape {q_mlp.shape}")
    if not np.all(np.isfinite(q_mlp)):
        raise ValueError("Canonical MLP output contains nonfinite values")
    return q_mlp, {
        "checkpoint_target_interpretation": interpretation,
        "checkpoint_target_interpretation_provenance": provenance,
        "canonical_output_formula": formula,
        "delta_export_reused": "false",
    }


def trajectory_fk(
    robot: Any,
    q: np.ndarray,
    joint_names: Sequence[str],
    ee_link: str,
) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != JOINT_DIM:
        raise ValueError(f"FK input must be [T,6], got {q.shape}")
    result = np.empty((q.shape[0], 3), dtype=np.float64)
    for index, row in enumerate(q):
        cfg = {
            joint_name: float(value)
            for joint_name, value in zip(joint_names, row)
        }
        robot.update_cfg(cfg)
        transform = robot.get_transform(frame_to=ee_link)
        matrix = np.asarray(transform, dtype=np.float64)
        if matrix.shape != (4, 4):
            raise ValueError(f"FK transform must be 4x4, got {matrix.shape}")
        result[index] = matrix[:3, 3]
    return result


def clip_seed(
    q: np.ndarray, bounds: Sequence[Tuple[float, float]]
) -> np.ndarray:
    lower = np.asarray([bound[0] for bound in bounds], dtype=np.float64)
    upper = np.asarray([bound[1] for bound in bounds], dtype=np.float64)
    return np.clip(np.asarray(q, dtype=np.float64), lower, upper)


def attempt_record(
    attempt: IkAttempt,
    source: str,
    previous_q: Optional[np.ndarray],
    max_allowed_joint_step: float,
) -> Dict[str, Any]:
    finite = bool(
        np.all(np.isfinite(attempt.q))
        and np.all(np.isfinite(attempt.ee))
        and np.isfinite(attempt.error)
    )
    if previous_q is None:
        delta = np.zeros(JOINT_DIM, dtype=np.float64)
        step_l2 = 0.0
        step_max = 0.0
        step_safe = True
    else:
        delta = attempt.q - previous_q
        step_l2 = float(np.linalg.norm(delta))
        step_max = float(np.max(np.abs(delta)))
        step_safe = step_max <= max_allowed_joint_step + 1.0e-12
    return {
        "source": source,
        "q": np.asarray(attempt.q, dtype=np.float64),
        "ee": np.asarray(attempt.ee, dtype=np.float64),
        "error": float(attempt.error),
        "solver_success": bool(attempt.success),
        "nit": int(attempt.nit),
        "message": str(attempt.message),
        "finite": finite,
        "joint_step_l2": step_l2,
        "joint_step_max_abs": step_max,
        "step_safe": bool(step_safe),
        "cartesian_tolerance_satisfied": bool(
            finite and attempt.error <= IK_CARTESIAN_TOLERANCE
        ),
    }


def unique_seeds(seeds: Iterable[Tuple[str, np.ndarray]]) -> List[Tuple[str, np.ndarray]]:
    result: List[Tuple[str, np.ndarray]] = []
    for source, seed in seeds:
        value = np.asarray(seed, dtype=np.float64)
        if any(np.allclose(value, existing, rtol=0.0, atol=1.0e-10) for _, existing in result):
            continue
        result.append((source, value))
    return result


def retry_seeds(
    *,
    previous_q: Optional[np.ndarray],
    previous_previous_q: Optional[np.ndarray],
    mlp_q: np.ndarray,
    q_start: np.ndarray,
    rng: np.random.Generator,
    bounds: Sequence[Tuple[float, float]],
    count: int,
    retry_profile: str,
) -> List[Tuple[str, np.ndarray]]:
    seeds: List[Tuple[str, np.ndarray]] = []
    if previous_q is None:
        seeds.extend(
            (
                ("known_q_start", q_start),
                ("zero_seed", np.zeros(JOINT_DIM, dtype=np.float64)),
            )
        )
    else:
        if retry_profile == "robust" and previous_previous_q is not None:
            seeds.extend(
                (
                    (
                        "local_constant_velocity",
                        previous_q + (previous_q - previous_previous_q),
                    ),
                    ("backtrack_recent_accepted", previous_previous_q),
                )
            )
        seeds.extend(
            (
                ("canonical_mlp_fallback", mlp_q),
                ("previous_mlp_blend_0.25", 0.75 * previous_q + 0.25 * mlp_q),
                ("previous_mlp_blend_0.50", 0.50 * previous_q + 0.50 * mlp_q),
            )
        )
    seeds = unique_seeds(seeds)
    while len(seeds) < count:
        seeds.append(("uniform_restart", sample_uniform_q(rng, bounds)))
    return seeds[:count]


def solve_attempt(
    *,
    robot: Any,
    desired_point: np.ndarray,
    seed: np.ndarray,
    previous_q: Optional[np.ndarray],
    joint_names: Sequence[str],
    ee_link: str,
    bounds: Sequence[Tuple[float, float]],
    parameters: StageParameters,
) -> IkAttempt:
    return solve_ik_from_initial_guess(
        robot=robot,
        p_des=desired_point,
        q_init=seed,
        q_ref=previous_q,
        joint_names=joint_names,
        ee_link=ee_link,
        bounds=bounds,
        smooth_weight=parameters.smooth_weight,
        maxiter=parameters.max_iters,
        ftol=parameters.ftol,
    )


def fallback_attempt(
    *,
    robot: Any,
    desired_point: np.ndarray,
    q: np.ndarray,
    joint_names: Sequence[str],
    ee_link: str,
    message: str,
) -> IkAttempt:
    ee = trajectory_fk(robot, q[None, :], joint_names, ee_link)[0]
    return IkAttempt(
        q=np.asarray(q, dtype=np.float64),
        ee=ee,
        error=float(np.linalg.norm(ee - desired_point)),
        success=False,
        nit=0,
        message=message,
    )


def choose_attempt(
    records: Sequence[Dict[str, Any]],
    previous_q: Optional[np.ndarray],
) -> Tuple[Optional[Dict[str, Any]], str]:
    finite = [record for record in records if bool(record["finite"])]
    if not finite:
        return None, "no_finite_ik_attempt"
    if previous_q is None:
        return min(
            finite,
            key=lambda record: (
                not bool(record["cartesian_tolerance_satisfied"]),
                float(record["error"]),
                not bool(record["solver_success"]),
            ),
        ), "first_point_best_cartesian"
    acceptable = [
        record
        for record in finite
        if bool(record["step_safe"])
        and bool(record["cartesian_tolerance_satisfied"])
    ]
    if acceptable:
        return min(
            acceptable,
            key=lambda record: (
                float(record["joint_step_l2"]),
                float(record["error"]),
                not bool(record["solver_success"]),
            ),
        ), "safe_cartesian_tolerance_solution"
    safe = [record for record in finite if bool(record["step_safe"])]
    if safe:
        return min(
            safe,
            key=lambda record: (
                float(record["error"]),
                float(record["joint_step_l2"]),
            ),
        ), "best_safe_solution_outside_cartesian_tolerance"
    return None, "no_step_safe_ik_solution"


def solve_adaptive_stage(
    *,
    robot: Any,
    desired_path: np.ndarray,
    canonical_mlp_q: np.ndarray,
    q_start: np.ndarray,
    joint_names: Sequence[str],
    ee_link: str,
    bounds: Sequence[Tuple[float, float]],
    parameters: StageParameters,
    max_allowed_joint_step: float,
    strict_joint_step: bool,
    num_ik_retries: int,
    random_seed: int,
    retry_profile: str,
) -> StageResult:
    trajectory_length = desired_path.shape[0]
    q_output = np.empty((trajectory_length, JOINT_DIM), dtype=np.float64)
    ee_output = np.empty((trajectory_length, 3), dtype=np.float64)
    records: List[Dict[str, Any]] = []
    rng = np.random.default_rng(random_seed)
    previous_q: Optional[np.ndarray] = None
    accepted_history: List[np.ndarray] = []
    ik_success_count = 0
    retry_count = 0
    failed_count = 0
    fallback_count = 0
    strict_failure_count = 0
    start_time = time.perf_counter()

    for timestep, desired_point in enumerate(desired_path):
        primary_source = (
            "canonical_mlp_initial" if previous_q is None else "previous_accepted_refined"
        )
        primary_seed = canonical_mlp_q[timestep] if previous_q is None else previous_q
        attempt_records: List[Dict[str, Any]] = []
        solver_exceptions: List[str] = []
        try:
            primary = solve_attempt(
                robot=robot,
                desired_point=desired_point,
                seed=primary_seed,
                previous_q=previous_q,
                joint_names=joint_names,
                ee_link=ee_link,
                bounds=bounds,
                parameters=parameters,
            )
            attempt_records.append(
                attempt_record(
                    primary,
                    primary_source,
                    previous_q,
                    max_allowed_joint_step,
                )
            )
        except Exception as exc:
            solver_exceptions.append(f"{primary_source}: {type(exc).__name__}: {exc}")

        primary_acceptable = bool(
            attempt_records
            and attempt_records[0]["finite"]
            and attempt_records[0]["step_safe"]
            and attempt_records[0]["cartesian_tolerance_satisfied"]
        )
        if not primary_acceptable and num_ik_retries > 0:
            alternatives = retry_seeds(
                previous_q=previous_q,
                previous_previous_q=(
                    accepted_history[-2] if len(accepted_history) >= 2 else None
                ),
                mlp_q=canonical_mlp_q[timestep],
                q_start=q_start,
                rng=rng,
                bounds=bounds,
                count=num_ik_retries,
                retry_profile=retry_profile,
            )
            for source, seed in alternatives:
                retry_count += 1
                try:
                    attempt = solve_attempt(
                        robot=robot,
                        desired_point=desired_point,
                        seed=seed,
                        previous_q=previous_q,
                        joint_names=joint_names,
                        ee_link=ee_link,
                        bounds=bounds,
                        parameters=parameters,
                    )
                    attempt_records.append(
                        attempt_record(
                            attempt,
                            source,
                            previous_q,
                            max_allowed_joint_step,
                        )
                    )
                except Exception as exc:
                    solver_exceptions.append(f"{source}: {type(exc).__name__}: {exc}")

        selected, selection_reason = choose_attempt(attempt_records, previous_q)
        retained_safe_fallback = False
        if selected is None:
            retained_safe_fallback = True
            strict_failure_count += int(previous_q is not None)
            fallback_q = (
                previous_q.copy()
                if previous_q is not None
                else clip_seed(canonical_mlp_q[timestep], bounds)
            )
            fallback = fallback_attempt(
                robot=robot,
                desired_point=desired_point,
                q=fallback_q,
                joint_names=joint_names,
                ee_link=ee_link,
                message=selection_reason,
            )
            selected = attempt_record(
                fallback,
                "retained_safe_fallback",
                previous_q,
                max_allowed_joint_step,
            )
            selection_reason += "; retained previous accepted configuration"

        q_selected = np.asarray(selected["q"], dtype=np.float64)
        ee_selected = np.asarray(selected["ee"], dtype=np.float64)
        q_output[timestep] = q_selected
        ee_output[timestep] = ee_selected
        previous_q = q_selected.copy()
        accepted_history.append(q_selected.copy())
        if bool(selected["solver_success"]):
            ik_success_count += 1
        failed_timestep = bool(
            retained_safe_fallback
            or not bool(selected["finite"])
            or not bool(selected["cartesian_tolerance_satisfied"])
        )
        failed_count += int(failed_timestep)
        selected_from_fallback = selected["source"] not in {
            "canonical_mlp_initial",
            "previous_accepted_refined",
        }
        fallback_count += int(selected_from_fallback)
        records.append(
            {
                "timestep": timestep,
                "selected_source": selected["source"],
                "selection_reason": selection_reason,
                "selected_error": float(selected["error"]),
                "selected_solver_success": bool(selected["solver_success"]),
                "selected_joint_step_l2": float(selected["joint_step_l2"]),
                "selected_joint_step_max_abs": float(
                    selected["joint_step_max_abs"]
                ),
                "selected_step_safe": bool(selected["step_safe"]),
                "retained_safe_fallback": retained_safe_fallback,
                "selected_from_fallback_seed": selected_from_fallback,
                "failed_timestep": failed_timestep,
                "strict_joint_step_failure": bool(
                    strict_joint_step and retained_safe_fallback and timestep > 0
                ),
                "attempt_count": len(attempt_records),
                "solver_exceptions": solver_exceptions,
                "attempts": [
                    {
                        key: value
                        for key, value in attempt.items()
                        if key not in {"q", "ee"}
                    }
                    for attempt in attempt_records
                ],
            }
        )

    runtime_sec = float(time.perf_counter() - start_time)
    return StageResult(
        q=q_output,
        ee=ee_output,
        step_records=records,
        ik_success_count=ik_success_count,
        ik_retry_count=retry_count,
        failed_timestep_count=failed_count,
        ik_fallback_timestep_count=fallback_count,
        strict_step_failure_count=strict_failure_count,
        runtime_sec=runtime_sec,
        parameters=parameters,
    )


def cartesian_errors(ee: np.ndarray, desired: np.ndarray) -> np.ndarray:
    if ee.shape != desired.shape:
        raise ValueError(f"EE/desired shape mismatch: {ee.shape} vs {desired.shape}")
    return np.linalg.norm(ee - desired, axis=1)


def adaptive_refine_path(
    *,
    robot: Any,
    desired_path: np.ndarray,
    canonical_mlp_q: np.ndarray,
    q_start: np.ndarray,
    joint_names: Sequence[str],
    ee_link: str,
    bounds: Sequence[Tuple[float, float]],
    max_allowed_joint_step: float,
    strict_joint_step: bool,
    num_ik_retries: int,
    random_seed: int,
    retry_profile: str,
) -> Tuple[StageResult, Dict[str, Any], List[Tuple[str, StageResult]]]:
    stage1_parameters = StageParameters(
        stage=1,
        smooth_weight=ADAPTIVE_STAGE1_SMOOTH_WEIGHT,
        max_iters=ADAPTIVE_STAGE1_MAX_ITERS,
        ftol=IK_FTOL,
        cartesian_tolerance=IK_CARTESIAN_TOLERANCE,
    )
    stage1 = solve_adaptive_stage(
        robot=robot,
        desired_path=desired_path,
        canonical_mlp_q=canonical_mlp_q,
        q_start=q_start,
        joint_names=joint_names,
        ee_link=ee_link,
        bounds=bounds,
        parameters=stage1_parameters,
        max_allowed_joint_step=max_allowed_joint_step,
        strict_joint_step=strict_joint_step,
        num_ik_retries=num_ik_retries,
        random_seed=random_seed,
        retry_profile="standard",
    )
    stage1_max_error = float(np.max(cartesian_errors(stage1.ee, desired_path)))
    stage2_triggered = stage1_max_error > ADAPTIVE_STAGE2_MAX_ERROR_THRESHOLD
    selected = stage1
    stage2: Optional[StageResult] = None
    if stage2_triggered:
        stage2_parameters = StageParameters(
            stage=2,
            smooth_weight=ADAPTIVE_STAGE2_SMOOTH_WEIGHT,
            max_iters=ADAPTIVE_STAGE2_MAX_ITERS,
            ftol=IK_FTOL,
            cartesian_tolerance=IK_CARTESIAN_TOLERANCE,
        )
        stage2 = solve_adaptive_stage(
            robot=robot,
            desired_path=desired_path,
            canonical_mlp_q=canonical_mlp_q,
            q_start=q_start,
            joint_names=joint_names,
            ee_link=ee_link,
            bounds=bounds,
            parameters=stage2_parameters,
            max_allowed_joint_step=max_allowed_joint_step,
            strict_joint_step=strict_joint_step,
            num_ik_retries=num_ik_retries,
            random_seed=random_seed + 1_000_003,
            retry_profile="standard",
        )
        if np.all(np.isfinite(stage2.q)) and np.all(np.isfinite(stage2.ee)):
            selected = stage2
    candidate_results: List[Tuple[str, StageResult]] = [
        ("standard_adaptive", selected),
        ("adaptive_stage1", stage1),
    ]
    if stage2 is not None:
        candidate_results.append(("adaptive_stage2", stage2))

    robust_results: List[Tuple[str, StageResult]] = []
    if retry_profile == "robust":
        robust_retry_count = min(max(num_ik_retries * 2, 8), 24)
        robust_settings = (
            ("robust_continuity", 101, 0.05, 500),
            ("robust_long_iteration", 102, 0.01, 750),
            ("robust_accuracy", 103, 0.001, 750),
        )
        for offset, (name, stage_number, smooth_weight, max_iters) in enumerate(
            robust_settings, start=1
        ):
            parameters = StageParameters(
                stage=stage_number,
                smooth_weight=smooth_weight,
                max_iters=max_iters,
                ftol=IK_FTOL,
                cartesian_tolerance=IK_CARTESIAN_TOLERANCE,
            )
            robust_result = solve_adaptive_stage(
                robot=robot,
                desired_path=desired_path,
                canonical_mlp_q=canonical_mlp_q,
                q_start=q_start,
                joint_names=joint_names,
                ee_link=ee_link,
                bounds=bounds,
                parameters=parameters,
                max_allowed_joint_step=max_allowed_joint_step,
                strict_joint_step=strict_joint_step,
                num_ik_retries=robust_retry_count,
                random_seed=random_seed + offset * 2_000_003,
                retry_profile="robust",
            )
            robust_results.append((name, robust_result))
        candidate_results.extend(robust_results)

    metadata = {
        "stage1_parameters": asdict(stage1.parameters),
        "stage1_max_cartesian_error": stage1_max_error,
        "stage1_runtime_sec": stage1.runtime_sec,
        "stage1_ik_retry_count": stage1.ik_retry_count,
        "stage1_failed_timestep_count": stage1.failed_timestep_count,
        "stage2_trigger_threshold": ADAPTIVE_STAGE2_MAX_ERROR_THRESHOLD,
        "stage2_triggered": stage2_triggered,
        "selected_adaptive_stage": selected.parameters.stage,
        "selected_stage_parameters": asdict(selected.parameters),
        "total_runtime_sec": stage1.runtime_sec
        + (0.0 if stage2 is None else stage2.runtime_sec),
        "total_ik_retry_count": stage1.ik_retry_count
        + (0 if stage2 is None else stage2.ik_retry_count),
        "retry_profile": retry_profile,
        "candidate_stage_names": [name for name, _ in candidate_results],
        "candidate_pool_runtime_sec": float(
            stage1.runtime_sec
            + (0.0 if stage2 is None else stage2.runtime_sec)
            + sum(result.runtime_sec for _, result in robust_results)
        ),
    }
    if stage2 is not None:
        metadata.update(
            {
                "stage2_parameters": asdict(stage2.parameters),
                "stage2_max_cartesian_error": float(
                    np.max(cartesian_errors(stage2.ee, desired_path))
                ),
                "stage2_runtime_sec": stage2.runtime_sec,
                "stage2_ik_retry_count": stage2.ik_retry_count,
                "stage2_failed_timestep_count": stage2.failed_timestep_count,
            }
        )
    return selected, metadata, candidate_results


def safe_ratio(numerator: float, denominator: float) -> float:
    if denominator > RANGE_EPS:
        return float(numerator / denominator)
    if numerator <= RANGE_EPS:
        return 1.0
    return float("nan")


def arc_length(points: np.ndarray) -> float:
    if points.shape[0] < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))


def derivative_cost(q: np.ndarray, order: int) -> float:
    if q.shape[0] <= order:
        return 0.0
    derivative = np.diff(q, n=order, axis=0)
    return float(np.mean(np.sum(np.square(derivative), axis=1)))


def branch_diagnostics(
    q: np.ndarray, joint_names: Sequence[str]
) -> Dict[str, Any]:
    delta = np.diff(q, axis=0)
    if not delta.size:
        return {
            "maximum_absolute_joint_step": 0.0,
            "maximum_l2_joint_step": 0.0,
            "timestep_of_maximum_step": 0,
            "joint_responsible_for_maximum_step": "",
            "timesteps_above_0_1_rad": 0,
            "timesteps_above_0_2_rad": 0,
            "timesteps_above_0_5_rad": 0,
            "timesteps_above_1_0_rad": 0,
        }
    absolute = np.abs(delta)
    flat_index = int(np.argmax(absolute))
    row_index_raw, joint_index_raw = np.unravel_index(flat_index, absolute.shape)
    row_index = int(row_index_raw)
    joint_index = int(joint_index_raw)
    l2 = np.linalg.norm(delta, axis=1)
    per_timestep_max = np.max(absolute, axis=1)
    return {
        "maximum_absolute_joint_step": float(absolute[row_index, joint_index]),
        "maximum_l2_joint_step": float(np.max(l2)),
        "timestep_of_maximum_step": int(row_index + 1),
        "joint_responsible_for_maximum_step": str(joint_names[joint_index]),
        "timesteps_above_0_1_rad": int(np.sum(per_timestep_max > 0.1)),
        "timesteps_above_0_2_rad": int(np.sum(per_timestep_max > 0.2)),
        "timesteps_above_0_5_rad": int(np.sum(per_timestep_max > 0.5)),
        "timesteps_above_1_0_rad": int(np.sum(per_timestep_max > 1.0)),
    }


def trajectory_metrics(
    *,
    q: np.ndarray,
    ee: np.ndarray,
    desired: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    joint_names: Sequence[str],
) -> Dict[str, Any]:
    error = cartesian_errors(ee, desired)
    desired_range = np.ptp(desired, axis=0)
    actual_range = np.ptp(ee, axis=0)
    steps = np.linalg.norm(np.diff(q, axis=0), axis=1)
    limits = check_joint_limits(
        q,
        lower,
        upper,
        joint_names,
        tolerance=HARD_JOINT_LIMIT_TOLERANCE_RAD,
        safety_margin=DEFAULT_JOINT_LIMIT_SAFETY_MARGIN_RAD,
    )
    return {
        "mean_cartesian_error": float(np.mean(error)),
        "rms_cartesian_error": float(np.sqrt(np.mean(np.square(error)))),
        "maximum_cartesian_error": float(np.max(error)),
        "median_cartesian_error": float(np.median(error)),
        "p95_cartesian_error": float(np.percentile(error, 95.0)),
        "cartesian_arc_length_ratio": safe_ratio(
            arc_length(ee), arc_length(desired)
        ),
        "x_range_ratio": safe_ratio(float(actual_range[0]), float(desired_range[0])),
        "y_range_ratio": safe_ratio(float(actual_range[1]), float(desired_range[1])),
        "z_range_ratio": safe_ratio(float(actual_range[2]), float(desired_range[2])),
        "mean_joint_step": float(np.mean(steps)) if steps.size else 0.0,
        "maximum_joint_step": float(np.max(steps)) if steps.size else 0.0,
        "velocity_cost": derivative_cost(q, 1),
        "acceleration_cost": derivative_cost(q, 2),
        "jerk_cost": derivative_cost(q, 3),
        "joint_limit_violation_count": int(
            limits["hard_joint_limit_violation_count"]
        ),
        "joint_limit_violation_magnitude": float(
            limits["hard_joint_limit_violation_magnitude"]
        ),
        **branch_diagnostics(q, joint_names),
    }


def joint_limit_diagnostics(
    *,
    q: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    joint_names: Sequence[str],
    margin: float = 0.0,
) -> Tuple[List[Dict[str, Any]], float]:
    if q.shape[1:] != (JOINT_DIM,):
        return [], float("-inf")
    limits = check_joint_limits(
        q,
        lower,
        upper,
        joint_names,
        tolerance=HARD_JOINT_LIMIT_TOLERANCE_RAD,
        safety_margin=margin,
    )
    details = list(limits["hard_violations"])
    details.extend(limits["safety_margin_violations"])
    return details, float(limits["minimum_joint_limit_margin_rad"])


def evaluate_complete_candidate(
    *,
    name: str,
    q: np.ndarray,
    ee: np.ndarray,
    desired: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    joint_names: Sequence[str],
    mean_error_gate: float,
    max_joint_step_gate: float,
    stage_result: Optional[StageResult],
) -> Dict[str, Any]:
    q = np.asarray(q, dtype=np.float64)
    ee = np.asarray(ee, dtype=np.float64)
    expected_q_shape = (desired.shape[0], JOINT_DIM)
    expected_ee_shape = desired.shape
    shape_valid = q.shape == expected_q_shape and ee.shape == expected_ee_shape
    nonfinite_count = int(np.count_nonzero(~np.isfinite(q))) + int(
        np.count_nonzero(~np.isfinite(ee))
    )
    if shape_valid and nonfinite_count == 0:
        metrics = trajectory_metrics(
            q=q,
            ee=ee,
            desired=desired,
            lower=lower,
            upper=upper,
            joint_names=joint_names,
        )
    else:
        metrics = {
            "mean_cartesian_error": float("inf"),
            "maximum_cartesian_error": float("inf"),
            "rms_cartesian_error": float("inf"),
            "maximum_absolute_joint_step": float("inf"),
            "joint_limit_violation_count": -1,
            "velocity_cost": float("inf"),
            "acceleration_cost": float("inf"),
            "jerk_cost": float("inf"),
        }
    unresolved_count = (
        0 if stage_result is None else int(stage_result.failed_timestep_count)
    )
    fallback_count = (
        0 if stage_result is None else int(stage_result.ik_fallback_timestep_count)
    )
    rejection_reasons: List[str] = []
    if not shape_valid:
        rejection_reasons.append("invalid_shape")
    if nonfinite_count > 0:
        rejection_reasons.append("nonfinite_values")
    if unresolved_count > 0:
        rejection_reasons.append("unresolved_timesteps")
    if float(metrics["mean_cartesian_error"]) > mean_error_gate:
        rejection_reasons.append("mean_cartesian_error_gate")
    if float(metrics["maximum_absolute_joint_step"]) > max_joint_step_gate:
        rejection_reasons.append("max_abs_joint_step_gate")
    if int(metrics["joint_limit_violation_count"]) != 0:
        rejection_reasons.append("joint_limit_violation")
    valid = len(rejection_reasons) == 0
    hard_limit_details, minimum_joint_limit_margin = joint_limit_diagnostics(
        q=q,
        lower=lower,
        upper=upper,
        joint_names=joint_names,
    )
    return {
        "candidate_name": name,
        "valid": valid,
        "mean_cartesian_error_m": float(metrics["mean_cartesian_error"]),
        "max_cartesian_error_m": float(metrics["maximum_cartesian_error"]),
        "rms_cartesian_error_m": float(metrics["rms_cartesian_error"]),
        "max_abs_joint_step_rad": float(metrics["maximum_absolute_joint_step"]),
        "joint_limit_violation_count": int(
            metrics["joint_limit_violation_count"]
        ),
        "joint_limit_violation_magnitude": float(
            metrics.get("joint_limit_violation_magnitude", float("inf"))
        ),
        "joint_limit_violations": hard_limit_details,
        "minimum_joint_limit_margin_rad": minimum_joint_limit_margin,
        "nonfinite_count": nonfinite_count,
        "unresolved_timestep_count": unresolved_count,
        "ik_fallback_timestep_count": fallback_count,
        "velocity_cost": float(metrics["velocity_cost"]),
        "acceleration_cost": float(metrics["acceleration_cost"]),
        "jerk_cost": float(metrics["jerk_cost"]),
        "ik_success_count": (
            0 if stage_result is None else int(stage_result.ik_success_count)
        ),
        "ik_retry_count": (
            0 if stage_result is None else int(stage_result.ik_retry_count)
        ),
        "runtime_sec": 0.0 if stage_result is None else stage_result.runtime_sec,
        "stage_parameters": (
            None if stage_result is None else asdict(stage_result.parameters)
        ),
        "rejection_reasons": rejection_reasons,
        "_q": q,
        "_ee": ee,
        "_metrics": metrics,
        "_stage_result": stage_result,
    }


def select_complete_candidate(
    candidates: Sequence[Dict[str, Any]],
) -> Tuple[Dict[str, Any], int]:
    if not candidates:
        raise ValueError("Candidate pool is empty")

    def finite_sort_value(value: Any) -> float:
        numeric = float(value)
        return numeric if np.isfinite(numeric) else float("inf")

    def descending_finite_sort_value(value: Any) -> float:
        numeric = float(value)
        return -numeric if np.isfinite(numeric) else float("inf")

    ordered = sorted(
        candidates,
        key=lambda candidate: (
            not bool(candidate["valid"]),
            finite_sort_value(candidate["mean_cartesian_error_m"]),
            finite_sort_value(candidate["max_cartesian_error_m"]),
            descending_finite_sort_value(
                candidate["minimum_joint_limit_margin_rad"]
            ),
            finite_sort_value(candidate["max_abs_joint_step_rad"]),
            finite_sort_value(candidate["acceleration_cost"]),
            finite_sort_value(candidate["jerk_cost"]),
            str(candidate["candidate_name"]),
        ),
    )
    return ordered[0], sum(bool(candidate["valid"]) for candidate in candidates)


def public_candidate_row(candidate: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        key: value
        for key, value in candidate.items()
        if not str(key).startswith("_")
    }


LOCAL_REPAIR_METHODS = (
    "forward_ik",
    "backward_ik",
    "bridge_seeded_ik",
    "canonical_mlp_local",
    "blended_bidirectional",
)


def contiguous_intervals(indices: Sequence[int]) -> List[Tuple[int, int]]:
    ordered = sorted(set(int(index) for index in indices))
    if not ordered:
        return []
    intervals: List[Tuple[int, int]] = []
    start = ordered[0]
    end = ordered[0]
    for index in ordered[1:]:
        if index == end + 1:
            end = index
        else:
            intervals.append((start, end))
            start = end = index
    intervals.append((start, end))
    return intervals


def expand_and_merge_intervals(
    intervals: Sequence[Tuple[int, int]], radius: int, length: int
) -> List[Tuple[int, int]]:
    expanded = sorted(
        (max(0, start - radius), min(length - 1, end + radius))
        for start, end in intervals
    )
    merged: List[Tuple[int, int]] = []
    for start, end in expanded:
        if not merged or start > merged[-1][1] + 1:
            merged.append((start, end))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
    return merged


def minimum_jerk_weight(progress: float) -> float:
    value = float(np.clip(progress, 0.0, 1.0))
    return value**3 * (10.0 - 15.0 * value + 6.0 * value**2)


def minimum_jerk_bridge(
    left_q: np.ndarray, right_q: np.ndarray, count: int
) -> np.ndarray:
    if count <= 0:
        return np.empty((0, JOINT_DIM), dtype=np.float64)
    progress = np.arange(1, count + 1, dtype=np.float64) / float(count + 1)
    weights = np.asarray([minimum_jerk_weight(value) for value in progress])
    return left_q[None, :] + weights[:, None] * (right_q - left_q)[None, :]


def candidate_unresolved_indices(candidate: Mapping[str, Any]) -> List[int]:
    stage_result = candidate.get("_stage_result")
    if stage_result is None:
        return []
    return [
        int(record["timestep"])
        for record in stage_result.step_records
        if bool(record.get("failed_timestep", False))
    ]


def local_repair_intervals(
    *,
    candidate: Mapping[str, Any],
    desired: np.ndarray,
    mean_error_gate: float,
    max_joint_step_gate: float,
    radius: int,
) -> List[Tuple[int, int]]:
    q = np.asarray(candidate["_q"], dtype=np.float64)
    ee = np.asarray(candidate["_ee"], dtype=np.float64)
    raw_intervals = contiguous_intervals(candidate_unresolved_indices(candidate))

    if q.shape == (desired.shape[0], JOINT_DIM) and np.all(np.isfinite(q)):
        violating_transitions = np.flatnonzero(
            np.max(np.abs(np.diff(q, axis=0)), axis=1) > max_joint_step_gate
        )
        raw_intervals.extend(
            (int(index), int(index + 1)) for index in violating_transitions
        )

    if (
        ee.shape == desired.shape
        and np.all(np.isfinite(ee))
        and float(candidate["mean_cartesian_error_m"]) > mean_error_gate
    ):
        errors = cartesian_errors(ee, desired)
        peak_index = int(np.argmax(errors))
        region_threshold = max(mean_error_gate, float(np.percentile(errors, 75.0)))
        start = peak_index
        end = peak_index
        while start > 0 and errors[start - 1] >= region_threshold:
            start -= 1
        while end + 1 < len(errors) and errors[end + 1] >= region_threshold:
            end += 1
        raw_intervals.append((start, end))

    return expand_and_merge_intervals(raw_intervals, radius, desired.shape[0])


def local_ik_polish(
    *,
    robot: Any,
    desired_point: np.ndarray,
    seed: np.ndarray,
    continuity_q: Optional[np.ndarray],
    joint_names: Sequence[str],
    ee_link: str,
    bounds: Sequence[Tuple[float, float]],
    parameters: StageParameters,
    bridge_step_target: float,
) -> Optional[np.ndarray]:
    try:
        attempt = solve_attempt(
            robot=robot,
            desired_point=desired_point,
            seed=seed,
            previous_q=continuity_q,
            joint_names=joint_names,
            ee_link=ee_link,
            bounds=bounds,
            parameters=parameters,
        )
        record = attempt_record(
            attempt,
            "local_repair_seed",
            continuity_q,
            bridge_step_target,
        )
    except Exception:
        return None
    if not bool(record["finite"]) or not bool(record["step_safe"]):
        return None
    return np.asarray(record["q"], dtype=np.float64)


def canonical_local_replacement(
    q: np.ndarray,
    canonical_q: np.ndarray,
    intervals: Sequence[Tuple[int, int]],
    radius: int,
) -> Tuple[np.ndarray, Dict[int, str]]:
    repaired = q.copy()
    provenance: Dict[int, str] = {}
    for start, end in intervals:
        count = end - start + 1
        for offset, timestep in enumerate(range(start, end + 1)):
            entry = 1.0 if start == 0 else minimum_jerk_weight((offset + 1) / (radius + 1))
            exit_distance = count - offset
            exit_weight = (
                1.0
                if end == q.shape[0] - 1
                else minimum_jerk_weight(exit_distance / (radius + 1))
            )
            canonical_weight = min(1.0, entry, exit_weight)
            repaired[timestep] = (
                (1.0 - canonical_weight) * q[timestep]
                + canonical_weight * canonical_q[timestep]
            )
            provenance[timestep] = "canonical_mlp_local"
    return repaired, provenance


def directional_local_repair(
    *,
    q: np.ndarray,
    canonical_q: np.ndarray,
    desired: np.ndarray,
    intervals: Sequence[Tuple[int, int]],
    direction: str,
    robot: Any,
    joint_names: Sequence[str],
    ee_link: str,
    bounds: Sequence[Tuple[float, float]],
    parameters: StageParameters,
    bridge_step_target: float,
) -> Tuple[np.ndarray, Dict[int, str]]:
    repaired = q.copy()
    provenance: Dict[int, str] = {}
    for start, end in intervals:
        if direction == "forward":
            timesteps = range(start, end + 1)
            continuity_q = repaired[start - 1].copy() if start > 0 else None
            fallback_source = "canonical_mlp_local"
            source = "forward_ik"
        else:
            timesteps = range(end, start - 1, -1)
            continuity_q = repaired[end + 1].copy() if end + 1 < len(repaired) else None
            fallback_source = "canonical_mlp_local"
            source = "backward_ik"
        for timestep in timesteps:
            seed = canonical_q[timestep] if continuity_q is None else continuity_q
            polished = local_ik_polish(
                robot=robot,
                desired_point=desired[timestep],
                seed=seed,
                continuity_q=continuity_q,
                joint_names=joint_names,
                ee_link=ee_link,
                bounds=bounds,
                parameters=parameters,
                bridge_step_target=bridge_step_target,
            )
            if polished is None:
                repaired[timestep] = canonical_q[timestep]
                provenance[timestep] = fallback_source
            else:
                repaired[timestep] = polished
                provenance[timestep] = source
            continuity_q = repaired[timestep].copy()
    return repaired, provenance


def bridge_seeded_local_repair(
    *,
    q: np.ndarray,
    desired: np.ndarray,
    intervals: Sequence[Tuple[int, int]],
    robot: Any,
    joint_names: Sequence[str],
    ee_link: str,
    bounds: Sequence[Tuple[float, float]],
    parameters: StageParameters,
    bridge_step_target: float,
) -> Optional[Tuple[np.ndarray, Dict[int, str]]]:
    if not all(start > 0 and end + 1 < len(q) for start, end in intervals):
        return None
    repaired = q.copy()
    provenance: Dict[int, str] = {}
    for start, end in intervals:
        bridge = minimum_jerk_bridge(
            repaired[start - 1], repaired[end + 1], end - start + 1
        )
        continuity_q = repaired[start - 1].copy()
        for offset, timestep in enumerate(range(start, end + 1)):
            polished = local_ik_polish(
                robot=robot,
                desired_point=desired[timestep],
                seed=bridge[offset],
                continuity_q=continuity_q,
                joint_names=joint_names,
                ee_link=ee_link,
                bounds=bounds,
                parameters=parameters,
                bridge_step_target=bridge_step_target,
            )
            if polished is None:
                repaired[timestep] = bridge[offset]
                provenance[timestep] = "interpolated_bridge"
            else:
                repaired[timestep] = polished
                provenance[timestep] = "bridge_seeded_ik"
            continuity_q = repaired[timestep].copy()
    return repaired, provenance


def blended_bidirectional_repair(
    forward_q: np.ndarray,
    backward_q: np.ndarray,
    intervals: Sequence[Tuple[int, int]],
) -> Tuple[np.ndarray, Dict[int, str]]:
    blended = forward_q.copy()
    provenance: Dict[int, str] = {}
    for start, end in intervals:
        count = end - start + 1
        for offset, timestep in enumerate(range(start, end + 1)):
            weight = minimum_jerk_weight((offset + 1) / float(count + 1))
            blended[timestep] = (
                (1.0 - weight) * forward_q[timestep]
                + weight * backward_q[timestep]
            )
            provenance[timestep] = "blended_bidirectional"
    return blended, provenance


def evaluate_local_repair_candidate(
    *,
    name: str,
    q: np.ndarray,
    provenance: Mapping[int, str],
    intervals: Sequence[Tuple[int, int]],
    base_candidate: Mapping[str, Any],
    repair_method: str,
    repair_pass: int,
    desired: np.ndarray,
    robot: Any,
    joint_names: Sequence[str],
    ee_link: str,
    lower: np.ndarray,
    upper: np.ndarray,
    mean_error_gate: float,
    max_joint_step_gate: float,
) -> Dict[str, Any]:
    ee = trajectory_fk(robot, q, joint_names, ee_link)
    candidate = evaluate_complete_candidate(
        name=name,
        q=q,
        ee=ee,
        desired=desired,
        lower=lower,
        upper=upper,
        joint_names=joint_names,
        mean_error_gate=mean_error_gate,
        max_joint_step_gate=max_joint_step_gate,
        stage_result=None,
    )
    candidate["_repair_metadata"] = {
        "base_candidate": str(base_candidate["candidate_name"]),
        "repair_method": repair_method,
        "repair_pass": repair_pass,
        "original_unresolved_timestep_count": int(
            base_candidate["unresolved_timestep_count"]
        ),
        "repair_intervals": [
            {"start": int(start), "end": int(end)} for start, end in intervals
        ],
        "repaired_timestep_count": len(provenance),
        "repaired_timestep_provenance": [
            {"timestep": int(timestep), "source": source}
            for timestep, source in sorted(provenance.items())
        ],
        "pre_repair_mean_cartesian_error_m": float(
            base_candidate["mean_cartesian_error_m"]
        ),
        "post_repair_mean_cartesian_error_m": float(
            candidate["mean_cartesian_error_m"]
        ),
        "pre_repair_max_abs_joint_step_rad": float(
            base_candidate["max_abs_joint_step_rad"]
        ),
        "post_repair_max_abs_joint_step_rad": float(
            candidate["max_abs_joint_step_rad"]
        ),
    }
    candidate["repair_method"] = repair_method
    candidate["repair_pass"] = repair_pass
    candidate["repair_intervals"] = candidate["_repair_metadata"][
        "repair_intervals"
    ]
    candidate["repaired_timestep_count"] = len(provenance)
    candidate["repaired_timestep_provenance"] = candidate["_repair_metadata"][
        "repaired_timestep_provenance"
    ]
    return candidate


def generate_local_repair_candidates(
    *,
    base_candidate: Mapping[str, Any],
    canonical_q: np.ndarray,
    desired: np.ndarray,
    robot: Any,
    joint_names: Sequence[str],
    ee_link: str,
    bounds: Sequence[Tuple[float, float]],
    lower: np.ndarray,
    upper: np.ndarray,
    mean_error_gate: float,
    max_joint_step_gate: float,
    local_repair_radius: int,
    local_repair_max_passes: int,
    bridge_step_target: float,
) -> List[Dict[str, Any]]:
    if bool(base_candidate["valid"]):
        return []
    eligible = (
        int(base_candidate["unresolved_timestep_count"]) > 0
        or float(base_candidate["mean_cartesian_error_m"]) > mean_error_gate
        or float(base_candidate["max_abs_joint_step_rad"]) > max_joint_step_gate
    )
    if not eligible:
        return []

    parameters = StageParameters(
        stage=201,
        smooth_weight=ADAPTIVE_STAGE1_SMOOTH_WEIGHT,
        max_iters=max(ADAPTIVE_STAGE2_MAX_ITERS, 500),
        ftol=IK_FTOL,
        cartesian_tolerance=IK_CARTESIAN_TOLERANCE,
    )
    generated: List[Dict[str, Any]] = []
    pass_base: Mapping[str, Any] = base_candidate
    for repair_pass in range(1, local_repair_max_passes + 1):
        radius = local_repair_radius + repair_pass - 1
        intervals = local_repair_intervals(
            candidate=pass_base,
            desired=desired,
            mean_error_gate=mean_error_gate,
            max_joint_step_gate=max_joint_step_gate,
            radius=radius,
        )
        if not intervals:
            break
        input_q = np.asarray(pass_base["_q"], dtype=np.float64)
        method_outputs: List[Tuple[str, np.ndarray, Dict[int, str]]] = []

        forward_q, forward_provenance = directional_local_repair(
            q=input_q,
            canonical_q=canonical_q,
            desired=desired,
            intervals=intervals,
            direction="forward",
            robot=robot,
            joint_names=joint_names,
            ee_link=ee_link,
            bounds=bounds,
            parameters=parameters,
            bridge_step_target=bridge_step_target,
        )
        method_outputs.append(("forward_ik", forward_q, forward_provenance))

        backward_q, backward_provenance = directional_local_repair(
            q=input_q,
            canonical_q=canonical_q,
            desired=desired,
            intervals=intervals,
            direction="backward",
            robot=robot,
            joint_names=joint_names,
            ee_link=ee_link,
            bounds=bounds,
            parameters=parameters,
            bridge_step_target=bridge_step_target,
        )
        method_outputs.append(("backward_ik", backward_q, backward_provenance))

        bridge_output = bridge_seeded_local_repair(
            q=input_q,
            desired=desired,
            intervals=intervals,
            robot=robot,
            joint_names=joint_names,
            ee_link=ee_link,
            bounds=bounds,
            parameters=parameters,
            bridge_step_target=bridge_step_target,
        )
        if bridge_output is not None:
            method_outputs.append(
                ("bridge_seeded_ik", bridge_output[0], bridge_output[1])
            )

        canonical_q_local, canonical_provenance = canonical_local_replacement(
            input_q, canonical_q, intervals, radius
        )
        method_outputs.append(
            ("canonical_mlp_local", canonical_q_local, canonical_provenance)
        )

        if np.all(np.isfinite(forward_q)) and np.all(np.isfinite(backward_q)):
            blended_q, blended_provenance = blended_bidirectional_repair(
                forward_q, backward_q, intervals
            )
            method_outputs.append(
                ("blended_bidirectional", blended_q, blended_provenance)
            )

        pass_candidates: List[Dict[str, Any]] = []
        for method, repaired_q, provenance in method_outputs:
            combined_provenance: Dict[int, str] = {}
            combined_intervals: List[Tuple[int, int]] = []
            previous_repair = pass_base.get("_repair_metadata")
            if previous_repair is not None:
                combined_provenance.update(
                    {
                        int(record["timestep"]): str(record["source"])
                        for record in previous_repair[
                            "repaired_timestep_provenance"
                        ]
                    }
                )
                combined_intervals.extend(
                    (int(interval["start"]), int(interval["end"]))
                    for interval in previous_repair["repair_intervals"]
                )
            combined_provenance.update(provenance)
            combined_intervals.extend(intervals)
            combined_intervals = expand_and_merge_intervals(
                combined_intervals, 0, desired.shape[0]
            )
            repaired_candidate = evaluate_local_repair_candidate(
                name=(
                    f"{base_candidate['candidate_name']}__local_{method}"
                    f"__pass_{repair_pass}"
                ),
                q=repaired_q,
                provenance=combined_provenance,
                intervals=combined_intervals,
                base_candidate=base_candidate,
                repair_method=method,
                repair_pass=repair_pass,
                desired=desired,
                robot=robot,
                joint_names=joint_names,
                ee_link=ee_link,
                lower=lower,
                upper=upper,
                mean_error_gate=mean_error_gate,
                max_joint_step_gate=max_joint_step_gate,
            )
            pass_candidates.append(repaired_candidate)
            generated.append(repaired_candidate)

        pass_base, _ = select_complete_candidate(pass_candidates)
        if bool(pass_base["valid"]):
            break
    return generated


JOINT_LIMIT_REPAIR_METHODS = (
    "projected_seed_ik",
    "forward_bound_aware_ik",
    "backward_bound_aware_ik",
    "minimum_jerk_bridge_ik",
    "canonical_mlp_bound_aware",
    "reduced_window_projected_ik",
)


def project_to_safe_joint_limits(
    q: np.ndarray, lower: np.ndarray, upper: np.ndarray, margin: float
) -> np.ndarray:
    return np.clip(q, lower + margin, upper - margin)


def inside_safe_joint_limits(
    q: np.ndarray, lower: np.ndarray, upper: np.ndarray, margin: float
) -> bool:
    return bool(
        np.all(np.isfinite(q))
        and np.all(q >= lower + margin)
        and np.all(q <= upper - margin)
    )


def joint_limit_repair_intervals(
    *,
    q: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    joint_names: Sequence[str],
    margin: float,
    radius: int,
) -> Tuple[List[Tuple[int, int]], List[Dict[str, Any]]]:
    violations, _ = joint_limit_diagnostics(
        q=q,
        lower=lower,
        upper=upper,
        joint_names=joint_names,
        margin=margin,
    )
    intervals = expand_and_merge_intervals(
        contiguous_intervals([int(item["timestep"]) for item in violations]),
        radius,
        q.shape[0],
    )
    return intervals, violations


def bound_aware_directional_repair(
    *,
    q: np.ndarray,
    canonical_q: np.ndarray,
    desired: np.ndarray,
    intervals: Sequence[Tuple[int, int]],
    direction: str,
    seed_mode: str,
    robot: Any,
    joint_names: Sequence[str],
    ee_link: str,
    bounds: Sequence[Tuple[float, float]],
    lower: np.ndarray,
    upper: np.ndarray,
    margin: float,
    parameters: StageParameters,
    bridge_step_target: float,
    blend_radius: int,
) -> np.ndarray:
    repaired = q.copy()
    if seed_mode == "canonical":
        seed_trajectory, _ = canonical_local_replacement(
            q, canonical_q, intervals, blend_radius
        )
        seed_trajectory = project_to_safe_joint_limits(
            seed_trajectory, lower, upper, margin
        )
    else:
        seed_trajectory = project_to_safe_joint_limits(q, lower, upper, margin)

    for start, end in intervals:
        if direction == "backward":
            timesteps = range(end, start - 1, -1)
            continuity_q = repaired[end + 1].copy() if end + 1 < len(q) else None
        else:
            timesteps = range(start, end + 1)
            continuity_q = repaired[start - 1].copy() if start > 0 else None
        if continuity_q is not None:
            continuity_q = project_to_safe_joint_limits(
                continuity_q, lower, upper, margin
            )
        for timestep in timesteps:
            if seed_mode == "boundary" and continuity_q is not None:
                seed = continuity_q
            else:
                seed = seed_trajectory[timestep]
            polished = local_ik_polish(
                robot=robot,
                desired_point=desired[timestep],
                seed=seed,
                continuity_q=continuity_q,
                joint_names=joint_names,
                ee_link=ee_link,
                bounds=bounds,
                parameters=parameters,
                bridge_step_target=bridge_step_target,
            )
            if polished is not None and inside_safe_joint_limits(
                polished, lower, upper, margin
            ):
                repaired[timestep] = polished
            else:
                repaired[timestep] = seed_trajectory[timestep]
            continuity_q = repaired[timestep].copy()
    return repaired


def bound_aware_bridge_repair(
    *,
    q: np.ndarray,
    desired: np.ndarray,
    intervals: Sequence[Tuple[int, int]],
    robot: Any,
    joint_names: Sequence[str],
    ee_link: str,
    bounds: Sequence[Tuple[float, float]],
    lower: np.ndarray,
    upper: np.ndarray,
    margin: float,
    parameters: StageParameters,
    bridge_step_target: float,
) -> Optional[np.ndarray]:
    if not all(start > 0 and end + 1 < len(q) for start, end in intervals):
        return None
    repaired = q.copy()
    for start, end in intervals:
        left_anchor = project_to_safe_joint_limits(
            repaired[start - 1], lower, upper, margin
        )
        right_anchor = project_to_safe_joint_limits(
            repaired[end + 1], lower, upper, margin
        )
        bridge = project_to_safe_joint_limits(
            minimum_jerk_bridge(left_anchor, right_anchor, end - start + 1),
            lower,
            upper,
            margin,
        )
        continuity_q = left_anchor
        for offset, timestep in enumerate(range(start, end + 1)):
            polished = local_ik_polish(
                robot=robot,
                desired_point=desired[timestep],
                seed=bridge[offset],
                continuity_q=continuity_q,
                joint_names=joint_names,
                ee_link=ee_link,
                bounds=bounds,
                parameters=parameters,
                bridge_step_target=bridge_step_target,
            )
            if polished is not None and inside_safe_joint_limits(
                polished, lower, upper, margin
            ):
                repaired[timestep] = polished
            else:
                repaired[timestep] = bridge[offset]
            continuity_q = repaired[timestep].copy()
    return repaired


def evaluate_joint_limit_repair_candidate(
    *,
    name: str,
    q: np.ndarray,
    method: str,
    repair_pass: int,
    intervals: Sequence[Tuple[int, int]],
    base_candidate: Mapping[str, Any],
    desired: np.ndarray,
    robot: Any,
    joint_names: Sequence[str],
    ee_link: str,
    lower: np.ndarray,
    upper: np.ndarray,
    margin: float,
    mean_error_gate: float,
    max_joint_step_gate: float,
) -> Dict[str, Any]:
    ee = trajectory_fk(robot, q, joint_names, ee_link)
    candidate = evaluate_complete_candidate(
        name=name,
        q=q,
        ee=ee,
        desired=desired,
        lower=lower,
        upper=upper,
        joint_names=joint_names,
        mean_error_gate=mean_error_gate,
        max_joint_step_gate=max_joint_step_gate,
        stage_result=None,
    )
    safe_violations, minimum_margin = joint_limit_diagnostics(
        q=q,
        lower=lower,
        upper=upper,
        joint_names=joint_names,
        margin=margin,
    )
    if safe_violations:
        candidate["rejection_reasons"].append("joint_limit_margin_gate")
        candidate["valid"] = False
    base_hard_violations = list(base_candidate["joint_limit_violations"])
    repaired_timesteps = sorted(
        {
            timestep
            for start, end in intervals
            for timestep in range(start, end + 1)
        }
    )
    repair_metadata = {
        "base_candidate": str(base_candidate["candidate_name"]),
        "repair_method": method,
        "repair_pass": repair_pass,
        "repair_intervals": [
            {"start": int(start), "end": int(end)} for start, end in intervals
        ],
        "repaired_timesteps": repaired_timesteps,
        "pre_repair_joint_limit_violations": base_hard_violations,
        "post_repair_joint_limit_violations": list(
            candidate["joint_limit_violations"]
        ),
        "safe_margin_violations": safe_violations,
        "pre_repair_joint_limit_violation_count": len(base_hard_violations),
        "post_repair_joint_limit_violation_count": int(
            candidate["joint_limit_violation_count"]
        ),
        "minimum_joint_limit_margin_rad": minimum_margin,
    }
    candidate["minimum_joint_limit_margin_rad"] = minimum_margin
    candidate["joint_limit_margin_violations"] = safe_violations
    candidate["joint_limit_repair_method"] = method
    candidate["joint_limit_repair_pass"] = repair_pass
    candidate["joint_limit_repair_intervals"] = repair_metadata[
        "repair_intervals"
    ]
    candidate["_joint_limit_repair_metadata"] = repair_metadata
    return candidate


def generate_joint_limit_repair_candidates(
    *,
    base_candidate: Mapping[str, Any],
    canonical_q: np.ndarray,
    desired: np.ndarray,
    robot: Any,
    joint_names: Sequence[str],
    ee_link: str,
    bounds: Sequence[Tuple[float, float]],
    lower: np.ndarray,
    upper: np.ndarray,
    margin: float,
    repair_radius: int,
    repair_passes: int,
    bridge_step_target: float,
    mean_error_gate: float,
    max_joint_step_gate: float,
) -> List[Dict[str, Any]]:
    input_q = np.asarray(base_candidate["_q"], dtype=np.float64)
    _, initial_safe_violations = joint_limit_repair_intervals(
        q=input_q,
        lower=lower,
        upper=upper,
        joint_names=joint_names,
        margin=margin,
        radius=0,
    )
    if not initial_safe_violations:
        return []

    parameters = StageParameters(
        stage=301,
        smooth_weight=ADAPTIVE_STAGE1_SMOOTH_WEIGHT,
        max_iters=max(ADAPTIVE_STAGE2_MAX_ITERS, 500),
        ftol=IK_FTOL,
        cartesian_tolerance=IK_CARTESIAN_TOLERANCE,
    )
    generated: List[Dict[str, Any]] = []
    pass_base: Mapping[str, Any] = base_candidate
    for repair_pass in range(1, repair_passes + 1):
        pass_q = np.asarray(pass_base["_q"], dtype=np.float64)
        radius = repair_radius + repair_pass - 1
        intervals, safe_violations = joint_limit_repair_intervals(
            q=pass_q,
            lower=lower,
            upper=upper,
            joint_names=joint_names,
            margin=margin,
            radius=radius,
        )
        if not safe_violations:
            break
        violating_timesteps = sorted(
            {int(item["timestep"]) for item in safe_violations}
        )
        reduced_radius = max(0, repair_radius // 2)
        reduced_intervals = expand_and_merge_intervals(
            [(timestep, timestep) for timestep in violating_timesteps],
            reduced_radius,
            pass_q.shape[0],
        )

        outputs: List[Tuple[str, np.ndarray, Sequence[Tuple[int, int]]]] = []
        outputs.append(
            (
                "projected_seed_ik",
                bound_aware_directional_repair(
                    q=pass_q,
                    canonical_q=canonical_q,
                    desired=desired,
                    intervals=intervals,
                    direction="forward",
                    seed_mode="projected",
                    robot=robot,
                    joint_names=joint_names,
                    ee_link=ee_link,
                    bounds=bounds,
                    lower=lower,
                    upper=upper,
                    margin=margin,
                    parameters=parameters,
                    bridge_step_target=bridge_step_target,
                    blend_radius=radius,
                ),
                intervals,
            )
        )
        for method, direction in (
            ("forward_bound_aware_ik", "forward"),
            ("backward_bound_aware_ik", "backward"),
        ):
            outputs.append(
                (
                    method,
                    bound_aware_directional_repair(
                        q=pass_q,
                        canonical_q=canonical_q,
                        desired=desired,
                        intervals=intervals,
                        direction=direction,
                        seed_mode="boundary",
                        robot=robot,
                        joint_names=joint_names,
                        ee_link=ee_link,
                        bounds=bounds,
                        lower=lower,
                        upper=upper,
                        margin=margin,
                        parameters=parameters,
                        bridge_step_target=bridge_step_target,
                        blend_radius=radius,
                    ),
                    intervals,
                )
            )
        bridge_q = bound_aware_bridge_repair(
            q=pass_q,
            desired=desired,
            intervals=intervals,
            robot=robot,
            joint_names=joint_names,
            ee_link=ee_link,
            bounds=bounds,
            lower=lower,
            upper=upper,
            margin=margin,
            parameters=parameters,
            bridge_step_target=bridge_step_target,
        )
        if bridge_q is not None:
            outputs.append(("minimum_jerk_bridge_ik", bridge_q, intervals))
        outputs.append(
            (
                "canonical_mlp_bound_aware",
                bound_aware_directional_repair(
                    q=pass_q,
                    canonical_q=canonical_q,
                    desired=desired,
                    intervals=intervals,
                    direction="forward",
                    seed_mode="canonical",
                    robot=robot,
                    joint_names=joint_names,
                    ee_link=ee_link,
                    bounds=bounds,
                    lower=lower,
                    upper=upper,
                    margin=margin,
                    parameters=parameters,
                    bridge_step_target=bridge_step_target,
                    blend_radius=radius,
                ),
                intervals,
            )
        )
        outputs.append(
            (
                "reduced_window_projected_ik",
                bound_aware_directional_repair(
                    q=pass_q,
                    canonical_q=canonical_q,
                    desired=desired,
                    intervals=reduced_intervals,
                    direction="forward",
                    seed_mode="projected",
                    robot=robot,
                    joint_names=joint_names,
                    ee_link=ee_link,
                    bounds=bounds,
                    lower=lower,
                    upper=upper,
                    margin=margin,
                    parameters=parameters,
                    bridge_step_target=bridge_step_target,
                    blend_radius=reduced_radius,
                ),
                reduced_intervals,
            )
        )

        pass_candidates: List[Dict[str, Any]] = []
        for method, repaired_q, method_intervals in outputs:
            combined_intervals = list(method_intervals)
            previous_repair = pass_base.get("_joint_limit_repair_metadata")
            if previous_repair is not None:
                combined_intervals.extend(
                    (int(interval["start"]), int(interval["end"]))
                    for interval in previous_repair["repair_intervals"]
                )
            combined_intervals = expand_and_merge_intervals(
                combined_intervals, 0, desired.shape[0]
            )
            candidate = evaluate_joint_limit_repair_candidate(
                name=(
                    f"{base_candidate['candidate_name']}__joint_limit_{method}"
                    f"__pass_{repair_pass}"
                ),
                q=repaired_q,
                method=method,
                repair_pass=repair_pass,
                intervals=combined_intervals,
                base_candidate=base_candidate,
                desired=desired,
                robot=robot,
                joint_names=joint_names,
                ee_link=ee_link,
                lower=lower,
                upper=upper,
                margin=margin,
                mean_error_gate=mean_error_gate,
                max_joint_step_gate=max_joint_step_gate,
            )
            generated.append(candidate)
            pass_candidates.append(candidate)
        pass_base, _ = select_complete_candidate(pass_candidates)
        if bool(pass_base["valid"]):
            break
    return generated


def write_joint_csv(path: Path, times: np.ndarray, q: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("t",) + JOINT_COLUMNS)
        for time_value, row in zip(times, q):
            writer.writerow(
                [f"{float(time_value):.10f}"]
                + [f"{float(value):.12g}" for value in row]
            )


def write_xyz_csv(path: Path, times: np.ndarray, xyz: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("t",) + XYZ_COLUMNS)
        for time_value, row in zip(times, xyz):
            writer.writerow(
                [f"{float(time_value):.10f}"]
                + [f"{float(value):.12g}" for value in row]
            )


def read_numeric_csv(path: Path, columns: Sequence[str]) -> np.ndarray:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        missing = [column for column in columns if column not in fieldnames]
        if missing:
            raise ValueError(f"{path} missing columns {missing}")
        rows = [[float(row[column]) for column in columns] for row in reader]
    values = np.asarray(rows, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(columns):
        raise ValueError(f"{path} has invalid data shape {values.shape}")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{path} contains nonfinite values")
    return values


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)


def write_records_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fallback_fields: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: List[str] = []
    seen = set()
    for field in fallback_fields:
        if field not in seen:
            seen.add(field)
            fields.append(field)
    for row in rows:
        for field in row:
            if field not in seen:
                seen.add(field)
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def existing_path_outputs(path_dir: Path) -> List[Path]:
    names = (
        "desired_path.csv",
        "canonical_mlp_q.csv",
        "canonical_mlp_ee.csv",
        "adaptive_mlp_ik_q.csv",
        "adaptive_mlp_ik_ee.csv",
        "generation_metadata.json",
    )
    return [path_dir / name for name in names if (path_dir / name).exists()]


def load_resumed_result(
    *,
    path_dir: Path,
    desired: np.ndarray,
    times: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    joint_names: Sequence[str],
    mean_error_gate: float,
    max_joint_step_gate: float,
) -> Dict[str, Any]:
    required = (
        path_dir / "desired_path.csv",
        path_dir / "canonical_mlp_q.csv",
        path_dir / "canonical_mlp_ee.csv",
        path_dir / "adaptive_mlp_ik_q.csv",
        path_dir / "adaptive_mlp_ik_ee.csv",
        path_dir / "generation_metadata.json",
    )
    missing = [path for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Cannot resume incomplete path {path_dir}; missing {missing}"
        )
    saved_desired = read_numeric_csv(path_dir / "desired_path.csv", XYZ_COLUMNS)
    if saved_desired.shape != desired.shape or not np.allclose(
        saved_desired, desired, rtol=1.0e-7, atol=1.0e-9
    ):
        raise ValueError(f"Resume desired path differs from input for {path_dir}")
    canonical_q = read_numeric_csv(path_dir / "canonical_mlp_q.csv", JOINT_COLUMNS)
    canonical_ee = read_numeric_csv(path_dir / "canonical_mlp_ee.csv", XYZ_COLUMNS)
    prior_q = read_numeric_csv(path_dir / "adaptive_mlp_ik_q.csv", JOINT_COLUMNS)
    prior_ee = read_numeric_csv(path_dir / "adaptive_mlp_ik_ee.csv", XYZ_COLUMNS)
    metadata = read_json(path_dir / "generation_metadata.json")
    candidate = evaluate_complete_candidate(
        name="existing_saved_adaptive_mlp_ik",
        q=prior_q,
        ee=prior_ee,
        desired=desired,
        lower=lower,
        upper=upper,
        joint_names=joint_names,
        mean_error_gate=mean_error_gate,
        max_joint_step_gate=max_joint_step_gate,
        stage_result=None,
    )
    unresolved_count = int(
        metadata.get(
            "unresolved_timestep_count",
            metadata.get("selected_stage_failed_timestep_count", 0),
        )
    )
    fallback_count = int(
        metadata.get(
            "ik_fallback_timestep_count",
            metadata.get("selected_stage_ik_fallback_timestep_count", 0),
        )
    )
    candidate["unresolved_timestep_count"] = unresolved_count
    candidate["ik_fallback_timestep_count"] = fallback_count
    if unresolved_count > 0 and "unresolved_timesteps" not in candidate["rejection_reasons"]:
        candidate["rejection_reasons"].append("unresolved_timesteps")
    candidate["valid"] = len(candidate["rejection_reasons"]) == 0
    generation_success = bool(candidate["valid"])
    metadata.update(
        {
            "selected_candidate": candidate["candidate_name"],
            "candidate_count": 1,
            "valid_candidate_count": int(generation_success),
            "mean_error_gate": mean_error_gate,
            "max_joint_step_gate": max_joint_step_gate,
            "mean_cartesian_error_m": candidate["mean_cartesian_error_m"],
            "max_cartesian_error_m": candidate["max_cartesian_error_m"],
            "rms_cartesian_error_m": candidate["rms_cartesian_error_m"],
            "max_abs_joint_step_rad": candidate["max_abs_joint_step_rad"],
            "joint_limit_violation_count": candidate[
                "joint_limit_violation_count"
            ],
            "nonfinite_count": candidate["nonfinite_count"],
            "unresolved_timestep_count": unresolved_count,
            "ik_fallback_timestep_count": fallback_count,
            "generation_success": generation_success,
            "generation_status": (
                "success" if generation_success else "failed_acceptance_gates"
            ),
            "rejection_reasons": candidate["rejection_reasons"],
            "candidate_table": [public_candidate_row(candidate)],
        }
    )
    metrics = dict(candidate["_metrics"])
    return {
        "canonical_q": canonical_q,
        "canonical_ee": canonical_ee,
        "prior_q": prior_q,
        "prior_ee": prior_ee,
        "metadata": metadata,
        "metrics": metrics,
        "resumed": True,
    }


def generate_path(
    *,
    split: str,
    dataset_index: int,
    path_name: str,
    desired: np.ndarray,
    times: np.ndarray,
    q_start: np.ndarray,
    path_dir: Path,
    model: PathConditionedMLP,
    checkpoint: Mapping[str, Any],
    checkpoint_path: Path,
    checkpoint_hash: str,
    times_source: str,
    device: torch.device,
    robot: Any,
    urdf_path: Path,
    joint_names: Sequence[str],
    ee_link: str,
    bounds: Sequence[Tuple[float, float]],
    lower: np.ndarray,
    upper: np.ndarray,
    max_allowed_joint_step: float,
    strict_joint_step: bool,
    num_ik_retries: int,
    random_seed: int,
    overwrite: bool,
    retry_profile: str,
    mean_error_gate: float,
    max_joint_step_gate: float,
    local_repair: bool,
    local_repair_radius: int,
    local_repair_max_passes: int,
    bridge_step_target: float,
    joint_limit_repair: bool,
    joint_limit_margin: float,
    joint_limit_repair_radius: int,
    joint_limit_repair_passes: int,
) -> Dict[str, Any]:
    canonical_q, interpretation = canonical_mlp_full_q(
        model=model,
        checkpoint=checkpoint,
        checkpoint_path=checkpoint_path,
        times=times,
        desired_path=desired,
        q_start=q_start,
        device=device,
    )
    canonical_ee = trajectory_fk(robot, canonical_q, joint_names, ee_link)
    canonical_metrics = trajectory_metrics(
        q=canonical_q,
        ee=canonical_ee,
        desired=desired,
        lower=lower,
        upper=upper,
        joint_names=joint_names,
    )
    standard_selected_stage, adaptive_metadata, stage_candidates = adaptive_refine_path(
        robot=robot,
        desired_path=desired,
        canonical_mlp_q=canonical_q,
        q_start=q_start,
        joint_names=joint_names,
        ee_link=ee_link,
        bounds=bounds,
        max_allowed_joint_step=max_allowed_joint_step,
        strict_joint_step=strict_joint_step,
        num_ik_retries=num_ik_retries,
        random_seed=random_seed,
        retry_profile=retry_profile,
    )
    del standard_selected_stage
    candidate_pool = [
        evaluate_complete_candidate(
            name=name,
            q=stage_result.q,
            ee=stage_result.ee,
            desired=desired,
            lower=lower,
            upper=upper,
            joint_names=joint_names,
            mean_error_gate=mean_error_gate,
            max_joint_step_gate=max_joint_step_gate,
            stage_result=stage_result,
        )
        for name, stage_result in stage_candidates
    ]
    candidate_pool.append(
        evaluate_complete_candidate(
            name="canonical_mlp_fallback",
            q=canonical_q,
            ee=canonical_ee,
            desired=desired,
            lower=lower,
            upper=upper,
            joint_names=joint_names,
            mean_error_gate=mean_error_gate,
            max_joint_step_gate=max_joint_step_gate,
            stage_result=None,
        )
    )
    repair_candidates: List[Dict[str, Any]] = []
    if local_repair:
        for base_candidate in list(candidate_pool):
            repair_candidates.extend(
                generate_local_repair_candidates(
                    base_candidate=base_candidate,
                    canonical_q=canonical_q,
                    desired=desired,
                    robot=robot,
                    joint_names=joint_names,
                    ee_link=ee_link,
                    bounds=bounds,
                    lower=lower,
                    upper=upper,
                    mean_error_gate=mean_error_gate,
                    max_joint_step_gate=max_joint_step_gate,
                    local_repair_radius=local_repair_radius,
                    local_repair_max_passes=local_repair_max_passes,
                    bridge_step_target=bridge_step_target,
                )
            )
        candidate_pool.extend(repair_candidates)
    joint_limit_repair_candidates: List[Dict[str, Any]] = []
    if joint_limit_repair:
        for base_candidate in list(candidate_pool):
            safe_margin_violations, minimum_margin = joint_limit_diagnostics(
                q=np.asarray(base_candidate["_q"], dtype=np.float64),
                lower=lower,
                upper=upper,
                joint_names=joint_names,
                margin=joint_limit_margin,
            )
            base_candidate["minimum_joint_limit_margin_rad"] = minimum_margin
            base_candidate["joint_limit_margin_violations"] = (
                safe_margin_violations
            )
            if safe_margin_violations:
                if (
                    "joint_limit_margin_gate"
                    not in base_candidate["rejection_reasons"]
                ):
                    base_candidate["rejection_reasons"].append(
                        "joint_limit_margin_gate"
                    )
                base_candidate["valid"] = False
            joint_limit_repair_candidates.extend(
                generate_joint_limit_repair_candidates(
                    base_candidate=base_candidate,
                    canonical_q=canonical_q,
                    desired=desired,
                    robot=robot,
                    joint_names=joint_names,
                    ee_link=ee_link,
                    bounds=bounds,
                    lower=lower,
                    upper=upper,
                    margin=joint_limit_margin,
                    repair_radius=joint_limit_repair_radius,
                    repair_passes=joint_limit_repair_passes,
                    bridge_step_target=bridge_step_target,
                    mean_error_gate=mean_error_gate,
                    max_joint_step_gate=max_joint_step_gate,
                )
            )
        candidate_pool.extend(joint_limit_repair_candidates)
    selected_candidate, valid_candidate_count = select_complete_candidate(
        candidate_pool
    )
    prior_q = np.asarray(selected_candidate["_q"], dtype=np.float64)
    prior_ee = np.asarray(selected_candidate["_ee"], dtype=np.float64)
    metrics = dict(selected_candidate["_metrics"])
    selected_stage = selected_candidate["_stage_result"]
    generation_success = bool(selected_candidate["valid"])
    status = "success" if generation_success else "failed_acceptance_gates"
    selected_repair = selected_candidate.get("_repair_metadata")
    repair_methods_attempted = sorted(
        {
            str(candidate["_repair_metadata"]["repair_method"])
            for candidate in repair_candidates
            if candidate.get("_repair_metadata") is not None
        }
    )
    local_repair_pass_count = max(
        (
            int(candidate["_repair_metadata"]["repair_pass"])
            for candidate in repair_candidates
            if candidate.get("_repair_metadata") is not None
        ),
        default=0,
    )
    selected_joint_limit_repair = selected_candidate.get(
        "_joint_limit_repair_metadata"
    )
    joint_limit_repair_methods_attempted = sorted(
        {
            str(candidate["_joint_limit_repair_metadata"]["repair_method"])
            for candidate in joint_limit_repair_candidates
            if candidate.get("_joint_limit_repair_metadata") is not None
        }
    )
    joint_limit_repair_pass_count = max(
        (
            int(candidate["_joint_limit_repair_metadata"]["repair_pass"])
            for candidate in joint_limit_repair_candidates
            if candidate.get("_joint_limit_repair_metadata") is not None
        ),
        default=0,
    )
    selected_pre_limit_violations = (
        list(selected_candidate["joint_limit_violations"])
        if selected_joint_limit_repair is None
        else list(
            selected_joint_limit_repair[
                "pre_repair_joint_limit_violations"
            ]
        )
    )
    metadata: Dict[str, Any] = {
        "split": split,
        "dataset_index": dataset_index,
        "path_name": path_name,
        "source_method": "canonical_path_conditioned_mlp_plus_adaptive_sequential_ik",
        "source_checkpoint": str(checkpoint_path),
        "source_checkpoint_sha256": checkpoint_hash,
        **interpretation,
        "times_source": times_source,
        "urdf": str(urdf_path),
        "end_effector_link": ee_link,
        "joint_ordering": list(joint_names),
        "joint_bounds": [list(bound) for bound in bounds],
        "expert_used_during_generation": False,
        "generation_success": generation_success,
        "generation_status": status,
        "random_seed": random_seed,
        "max_allowed_joint_step": max_allowed_joint_step,
        "mean_error_gate": mean_error_gate,
        "max_joint_step_gate": max_joint_step_gate,
        "strict_joint_step": strict_joint_step,
        "num_ik_retries": num_ik_retries,
        "retry_profile": retry_profile,
        "local_repair_enabled": local_repair,
        "local_repair_radius": local_repair_radius,
        "local_repair_max_passes": local_repair_max_passes,
        "bridge_step_target": bridge_step_target,
        "local_repair_attempt_count": len(repair_candidates),
        "local_repair_pass_count": local_repair_pass_count,
        "original_unresolved_timestep_count": (
            int(selected_candidate["unresolved_timestep_count"])
            if selected_repair is None
            else int(selected_repair["original_unresolved_timestep_count"])
        ),
        "repaired_timestep_count": (
            0
            if selected_repair is None
            else int(selected_repair["repaired_timestep_count"])
        ),
        "repair_intervals": (
            [] if selected_repair is None else selected_repair["repair_intervals"]
        ),
        "repair_methods_attempted": repair_methods_attempted,
        "selected_repair_method": (
            "none" if selected_repair is None else selected_repair["repair_method"]
        ),
        "pre_repair_mean_cartesian_error_m": (
            float(selected_candidate["mean_cartesian_error_m"])
            if selected_repair is None
            else float(selected_repair["pre_repair_mean_cartesian_error_m"])
        ),
        "post_repair_mean_cartesian_error_m": float(
            selected_candidate["mean_cartesian_error_m"]
        ),
        "pre_repair_max_abs_joint_step_rad": (
            float(selected_candidate["max_abs_joint_step_rad"])
            if selected_repair is None
            else float(selected_repair["pre_repair_max_abs_joint_step_rad"])
        ),
        "post_repair_max_abs_joint_step_rad": float(
            selected_candidate["max_abs_joint_step_rad"]
        ),
        "repaired_timestep_provenance": (
            []
            if selected_repair is None
            else selected_repair["repaired_timestep_provenance"]
        ),
        "joint_limit_repair_enabled": joint_limit_repair,
        "joint_limit_margin": joint_limit_margin,
        "joint_limit_repair_radius": joint_limit_repair_radius,
        "joint_limit_repair_passes": joint_limit_repair_passes,
        "pre_repair_joint_limit_violation_count": len(
            selected_pre_limit_violations
        ),
        "post_repair_joint_limit_violation_count": int(
            selected_candidate["joint_limit_violation_count"]
        ),
        "violating_joint_names": sorted(
            {
                str(item["joint_name"])
                for item in selected_pre_limit_violations
            }
        ),
        "violating_timesteps": sorted(
            {
                int(item["timestep"])
                for item in selected_pre_limit_violations
            }
        ),
        "maximum_pre_repair_violation_magnitude": max(
            (
                float(item["violation_magnitude"])
                for item in selected_pre_limit_violations
            ),
            default=0.0,
        ),
        "minimum_joint_limit_margin_rad": float(
            selected_candidate["minimum_joint_limit_margin_rad"]
        ),
        "joint_limit_repair_attempt_count": len(
            joint_limit_repair_candidates
        ),
        "joint_limit_repair_pass_count": joint_limit_repair_pass_count,
        "joint_limit_repair_intervals": (
            []
            if selected_joint_limit_repair is None
            else selected_joint_limit_repair["repair_intervals"]
        ),
        "joint_limit_repair_methods_attempted": (
            joint_limit_repair_methods_attempted
        ),
        "selected_joint_limit_repair_method": (
            "none"
            if selected_joint_limit_repair is None
            else selected_joint_limit_repair["repair_method"]
        ),
        "pre_repair_joint_limit_violations": selected_pre_limit_violations,
        "post_repair_joint_limit_violations": list(
            selected_candidate["joint_limit_violations"]
        ),
        "selected_candidate": selected_candidate["candidate_name"],
        "candidate_count": len(candidate_pool),
        "valid_candidate_count": valid_candidate_count,
        "mean_cartesian_error_m": selected_candidate["mean_cartesian_error_m"],
        "max_cartesian_error_m": selected_candidate["max_cartesian_error_m"],
        "rms_cartesian_error_m": selected_candidate["rms_cartesian_error_m"],
        "max_abs_joint_step_rad": selected_candidate["max_abs_joint_step_rad"],
        "joint_limit_violation_count": selected_candidate[
            "joint_limit_violation_count"
        ],
        "joint_limit_violation_magnitude": selected_candidate[
            "joint_limit_violation_magnitude"
        ],
        "nonfinite_count": selected_candidate["nonfinite_count"],
        "unresolved_timestep_count": selected_candidate[
            "unresolved_timestep_count"
        ],
        "ik_fallback_timestep_count": selected_candidate[
            "ik_fallback_timestep_count"
        ],
        "rejection_reasons": selected_candidate["rejection_reasons"],
        "selected_stage_ik_success_count": (
            0 if selected_stage is None else selected_stage.ik_success_count
        ),
        "selected_stage_ik_retry_count": (
            0 if selected_stage is None else selected_stage.ik_retry_count
        ),
        "selected_stage_failed_timestep_count": (
            0 if selected_stage is None else selected_stage.failed_timestep_count
        ),
        "selected_stage_strict_step_failure_count": (
            0 if selected_stage is None else selected_stage.strict_step_failure_count
        ),
        "selected_stage_step_records": (
            [] if selected_stage is None else selected_stage.step_records
        ),
        "candidate_table": [
            public_candidate_row(candidate) for candidate in candidate_pool
        ],
        "canonical_mlp_metrics": canonical_metrics,
        "adaptive_parameter_provenance_ambiguities": {
            "num_ik_retries": (
                "adaptive wrapper has no per-timestep retry count; inherited from "
                "generate_ik_seed_path.py unless explicitly supplied"
            ),
            "damping": (
                "repository L-BFGS-B IK exposes no damping parameter; smooth_weight "
                "is the available continuity regularizer"
            ),
        },
        "metrics": metrics,
        **adaptive_metadata,
        "selected_adaptive_stage": (
            "canonical_mlp_fallback"
            if selected_stage is None
            else selected_stage.parameters.stage
        ),
    }
    if existing_path_outputs(path_dir) and not overwrite:
        raise FileExistsError(
            f"Method-specific output already exists for {path_dir}; "
            "pass --resume or --overwrite"
        )
    write_xyz_csv(path_dir / "desired_path.csv", times, desired)
    write_joint_csv(path_dir / "canonical_mlp_q.csv", times, canonical_q)
    write_xyz_csv(path_dir / "canonical_mlp_ee.csv", times, canonical_ee)
    write_joint_csv(path_dir / "adaptive_mlp_ik_q.csv", times, prior_q)
    write_xyz_csv(path_dir / "adaptive_mlp_ik_ee.csv", times, prior_ee)
    write_json(path_dir / "generation_metadata.json", metadata)
    return {
        "canonical_q": canonical_q,
        "canonical_ee": canonical_ee,
        "prior_q": prior_q,
        "prior_ee": prior_ee,
        "metadata": metadata,
        "metrics": metrics,
        "resumed": False,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def summary_row(
    *,
    split: str,
    dataset_index: int,
    path_name: str,
    result: Mapping[str, Any],
    checkpoint: Path,
) -> Dict[str, Any]:
    metadata = result["metadata"]
    metrics = result["metrics"]
    row: Dict[str, Any] = {
        "split": split,
        "dataset_index": dataset_index,
        "path_name": path_name,
        "generation_success": bool(metadata.get("generation_success", False)),
        "generation_status": str(metadata.get("generation_status", "unknown")),
        "source_method": "canonical_path_conditioned_mlp_plus_adaptive_sequential_ik",
        "source_checkpoint": str(checkpoint),
        "selected_candidate": metadata.get("selected_candidate", ""),
        "candidate_count": metadata.get("candidate_count", 0),
        "valid_candidate_count": metadata.get("valid_candidate_count", 0),
        "retry_profile": metadata.get("retry_profile", "standard"),
        "mean_cartesian_error_m": metadata.get(
            "mean_cartesian_error_m", metrics.get("mean_cartesian_error", "")
        ),
        "max_cartesian_error_m": metadata.get(
            "max_cartesian_error_m", metrics.get("maximum_cartesian_error", "")
        ),
        "rms_cartesian_error_m": metadata.get(
            "rms_cartesian_error_m", metrics.get("rms_cartesian_error", "")
        ),
        "max_abs_joint_step_rad": metadata.get(
            "max_abs_joint_step_rad", metrics.get("maximum_absolute_joint_step", "")
        ),
        "nonfinite_count": metadata.get("nonfinite_count", 0),
        "unresolved_timestep_count": metadata.get(
            "unresolved_timestep_count", 0
        ),
        "ik_fallback_timestep_count": metadata.get(
            "ik_fallback_timestep_count", 0
        ),
        "rejection_reasons": json.dumps(metadata.get("rejection_reasons", [])),
        "local_repair_enabled": bool(metadata.get("local_repair_enabled", False)),
        "local_repair_attempt_count": metadata.get("local_repair_attempt_count", 0),
        "local_repair_pass_count": metadata.get("local_repair_pass_count", 0),
        "original_unresolved_timestep_count": metadata.get(
            "original_unresolved_timestep_count",
            metadata.get("unresolved_timestep_count", 0),
        ),
        "repaired_timestep_count": metadata.get("repaired_timestep_count", 0),
        "repair_intervals": json.dumps(metadata.get("repair_intervals", [])),
        "repair_methods_attempted": json.dumps(
            metadata.get("repair_methods_attempted", [])
        ),
        "selected_repair_method": metadata.get("selected_repair_method", "none"),
        "pre_repair_mean_cartesian_error_m": metadata.get(
            "pre_repair_mean_cartesian_error_m",
            metrics.get("mean_cartesian_error", ""),
        ),
        "post_repair_mean_cartesian_error_m": metadata.get(
            "post_repair_mean_cartesian_error_m",
            metrics.get("mean_cartesian_error", ""),
        ),
        "pre_repair_max_abs_joint_step_rad": metadata.get(
            "pre_repair_max_abs_joint_step_rad",
            metrics.get("maximum_absolute_joint_step", ""),
        ),
        "post_repair_max_abs_joint_step_rad": metadata.get(
            "post_repair_max_abs_joint_step_rad",
            metrics.get("maximum_absolute_joint_step", ""),
        ),
        "joint_limit_repair_enabled": bool(
            metadata.get("joint_limit_repair_enabled", False)
        ),
        "pre_repair_joint_limit_violation_count": metadata.get(
            "pre_repair_joint_limit_violation_count",
            metrics.get("joint_limit_violation_count", 0),
        ),
        "post_repair_joint_limit_violation_count": metadata.get(
            "post_repair_joint_limit_violation_count",
            metrics.get("joint_limit_violation_count", 0),
        ),
        "violating_joint_names": json.dumps(
            metadata.get("violating_joint_names", [])
        ),
        "violating_timesteps": json.dumps(
            metadata.get("violating_timesteps", [])
        ),
        "maximum_pre_repair_violation_magnitude": metadata.get(
            "maximum_pre_repair_violation_magnitude", 0.0
        ),
        "minimum_joint_limit_margin_rad": metadata.get(
            "minimum_joint_limit_margin_rad", ""
        ),
        "joint_limit_repair_attempt_count": metadata.get(
            "joint_limit_repair_attempt_count", 0
        ),
        "joint_limit_repair_pass_count": metadata.get(
            "joint_limit_repair_pass_count", 0
        ),
        "joint_limit_repair_intervals": json.dumps(
            metadata.get("joint_limit_repair_intervals", [])
        ),
        "joint_limit_repair_methods_attempted": json.dumps(
            metadata.get("joint_limit_repair_methods_attempted", [])
        ),
        "selected_joint_limit_repair_method": metadata.get(
            "selected_joint_limit_repair_method", "none"
        ),
        "checkpoint_target_interpretation": metadata.get(
            "checkpoint_target_interpretation", ""
        ),
        "adaptive_stage": metadata.get("selected_adaptive_stage", ""),
        "canonical_mlp_mean_cartesian_error": metadata.get(
            "canonical_mlp_metrics", {}
        ).get("mean_cartesian_error", ""),
        "canonical_mlp_maximum_cartesian_error": metadata.get(
            "canonical_mlp_metrics", {}
        ).get("maximum_cartesian_error", ""),
        "ik_success_count": metadata.get("selected_stage_ik_success_count", ""),
        "ik_retry_count": metadata.get("total_ik_retry_count", ""),
        "failed_timestep_count": metadata.get(
            "selected_stage_failed_timestep_count", ""
        ),
        "runtime_sec": metadata.get("total_runtime_sec", ""),
        "resumed": bool(result.get("resumed", False)),
        **metrics,
    }
    if "joint_rmse_vs_expert" in metadata:
        row["joint_rmse_vs_expert"] = metadata["joint_rmse_vs_expert"]
    return row


def branch_row(
    split: str,
    dataset_index: int,
    path_name: str,
    result: Mapping[str, Any],
) -> Dict[str, Any]:
    metrics = result["metrics"]
    return {
        "split": split,
        "dataset_index": dataset_index,
        "path_name": path_name,
        "generation_success": result["metadata"].get("generation_success", False),
        "generation_status": result["metadata"].get("generation_status", ""),
        "max_allowed_joint_step": result["metadata"].get(
            "max_allowed_joint_step", ""
        ),
        "maximum_absolute_joint_step": metrics["maximum_absolute_joint_step"],
        "maximum_l2_joint_step": metrics["maximum_l2_joint_step"],
        "timestep_of_maximum_step": metrics["timestep_of_maximum_step"],
        "joint_responsible_for_maximum_step": metrics[
            "joint_responsible_for_maximum_step"
        ],
        "timesteps_above_0_1_rad": metrics["timesteps_above_0_1_rad"],
        "timesteps_above_0_2_rad": metrics["timesteps_above_0_2_rad"],
        "timesteps_above_0_5_rad": metrics["timesteps_above_0_5_rad"],
        "timesteps_above_1_0_rad": metrics["timesteps_above_1_0_rad"],
    }


def load_expert_after_generation(
    input_npz: Path, expected_shape: Tuple[int, int, int]
) -> np.ndarray:
    with np.load(input_npz, allow_pickle=True) as archive:
        if "expert_q" not in archive.files:
            raise KeyError(f"{input_npz} has no expert_q for optional evaluation")
        expert = np.asarray(archive["expert_q"], dtype=np.float64)
    if expert.shape != expected_shape:
        raise ValueError(f"expert_q has shape {expert.shape}, expected {expected_shape}")
    if not np.all(np.isfinite(expert)):
        raise ValueError("expert_q contains nonfinite values")
    return expert


def reproduction_row(
    *,
    path_name: str,
    new_q: np.ndarray,
    new_ee: np.ndarray,
    desired: np.ndarray,
    old_root: Path,
    robot: Any,
    joint_names: Sequence[str],
    ee_link: str,
) -> Dict[str, Any]:
    old_path = old_root / safe_path_name(path_name) / "refined_mlp_ik_q.csv"
    base: Dict[str, Any] = {
        "split": "test",
        "path_name": path_name,
        "old_trajectory_path": str(old_path),
        "old_trajectory_exists": old_path.exists(),
    }
    if not old_path.exists():
        return {
            **base,
            "maximum_joint_difference": "",
            "mean_joint_difference": "",
            "new_mean_cartesian_error": float(
                np.mean(cartesian_errors(new_ee, desired))
            ),
            "old_mean_cartesian_error": "",
            "cartesian_metric_difference": "",
            "old_trajectory_exactly_reproducible": False,
            "old_trajectory_allclose": False,
            "difference_explanation": "old refined_mlp_ik_q.csv is absent",
        }
    old_q = read_numeric_csv(old_path, JOINT_COLUMNS)
    if old_q.shape != new_q.shape:
        return {
            **base,
            "maximum_joint_difference": "",
            "mean_joint_difference": "",
            "new_mean_cartesian_error": float(
                np.mean(cartesian_errors(new_ee, desired))
            ),
            "old_mean_cartesian_error": "",
            "cartesian_metric_difference": "",
            "old_trajectory_exactly_reproducible": False,
            "old_trajectory_allclose": False,
            "difference_explanation": f"old shape {old_q.shape} differs from {new_q.shape}",
        }
    difference = new_q - old_q
    old_ee = trajectory_fk(robot, old_q, joint_names, ee_link)
    new_mean = float(np.mean(cartesian_errors(new_ee, desired)))
    old_mean = float(np.mean(cartesian_errors(old_ee, desired)))
    exact = bool(np.array_equal(new_q, old_q))
    allclose = bool(np.allclose(new_q, old_q, rtol=1.0e-7, atol=1.0e-9))
    explanation = ""
    if not exact:
        explanation = (
            "legacy file provenance is ambiguous because fixed and adaptive runs shared "
            "refined_mlp_ik_q.csv; the new method also adds explicit branch-safe retries"
        )
    return {
        **base,
        "maximum_joint_difference": float(np.max(np.abs(difference))),
        "mean_joint_difference": float(np.mean(np.abs(difference))),
        "new_mean_cartesian_error": new_mean,
        "old_mean_cartesian_error": old_mean,
        "cartesian_metric_difference": new_mean - old_mean,
        "old_trajectory_exactly_reproducible": exact,
        "old_trajectory_allclose": allclose,
        "difference_explanation": explanation,
    }


def merge_split_rows(
    path: Path,
    split: str,
    new_rows: Sequence[Mapping[str, Any]],
    fallback_fields: Sequence[str],
) -> None:
    retained: List[Dict[str, Any]] = []
    if path.exists():
        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            retained = [
                dict(row)
                for row in csv.DictReader(handle)
                if str(row.get("split", "")) != split
            ]
    write_records_csv(path, [*retained, *new_rows], fallback_fields)


def save_prior_npz(
    *,
    path: Path,
    names: Sequence[str],
    desired_paths: np.ndarray,
    results: Sequence[Mapping[str, Any]],
    checkpoint: Path,
    overwrite_allowed: bool,
) -> None:
    if path.exists() and not overwrite_allowed:
        raise FileExistsError(f"{path} exists; pass --resume or --overwrite")
    prior_q = np.stack([np.asarray(result["prior_q"]) for result in results])
    prior_ee = np.stack([np.asarray(result["prior_ee"]) for result in results])
    canonical_q = np.stack(
        [np.asarray(result["canonical_q"]) for result in results]
    )
    canonical_ee = np.stack(
        [np.asarray(result["canonical_ee"]) for result in results]
    )
    expected_n = len(names)
    expected_q_shape = (expected_n, EXPECTED_STEPS, JOINT_DIM)
    expected_ee_shape = (expected_n, EXPECTED_STEPS, 3)
    if prior_q.shape != expected_q_shape or canonical_q.shape != expected_q_shape:
        raise AssertionError("Prior or canonical MLP NPZ trajectory shape is invalid")
    if prior_ee.shape != expected_ee_shape or canonical_ee.shape != expected_ee_shape:
        raise AssertionError("Prior or canonical MLP NPZ FK shape is invalid")
    if desired_paths.shape != expected_ee_shape:
        raise AssertionError("Selected desired path NPZ shape is invalid")
    success = np.asarray(
        [bool(result["metadata"].get("generation_success", False)) for result in results],
        dtype=bool,
    )
    status = np.asarray(
        [str(result["metadata"].get("generation_status", "unknown")) for result in results]
    )
    source_method = np.asarray(
        ["canonical_path_conditioned_mlp_plus_adaptive_sequential_ik"] * expected_n
    )
    source_checkpoint = np.asarray([str(checkpoint)] * expected_n)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(
        temporary,
        path_names=np.asarray(names),
        desired_paths=desired_paths.astype(np.float32),
        prior_q=prior_q.astype(np.float32),
        prior_ee=prior_ee.astype(np.float32),
        canonical_mlp_q=canonical_q.astype(np.float32),
        canonical_mlp_ee=canonical_ee.astype(np.float32),
        generation_success=success,
        generation_status=status,
        source_method=source_method,
        source_checkpoint=source_checkpoint,
    )
    temporary.replace(path)


def update_configuration(path: Path, split: str, run: Mapping[str, Any]) -> None:
    configuration: Dict[str, Any] = {"runs": {}}
    if path.exists():
        existing = read_json(path)
        if isinstance(existing, dict):
            configuration.update(existing)
            configuration.setdefault("runs", {})
    configuration["runs"][split] = dict(run)
    configuration["latest_split"] = split
    write_json(path, configuration)


def classification(
    rows: Sequence[Mapping[str, Any]], requested_count: int
) -> str:
    if len(rows) != requested_count or any(
        not bool(row["generation_success"]) for row in rows
    ):
        return "GENERATION_FAILURE"
    mean_error = float(
        np.mean([float(row["mean_cartesian_error"]) for row in rows])
    )
    if mean_error > 0.01:
        return "INSUFFICIENT_CARTESIAN_ACCURACY"
    if any(float(row["maximum_absolute_joint_step"]) > 0.2 for row in rows):
        return "ACCURATE_BUT_DISCONTINUOUS"
    if sum(int(row["joint_limit_violation_count"]) for row in rows) > 0:
        return "GENERATION_FAILURE"
    return "ACCEPTABLE_STRONG_PRIOR"


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
        temporary_path.replace(path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _atomic_write_records_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fallback_fields: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    records_to_write: List[Mapping[str, Any]] = list(rows)
    if (
        path.exists()
        and "generation_summary" in path.name
        and all("path_name" in record for record in records_to_write)
    ):
        with path.open("r", encoding="utf-8", newline="") as handle:
            previous_records = list(csv.DictReader(handle))
        merged_by_name: Dict[str, Mapping[str, Any]] = {
            str(record.get("path_name", "")): record
            for record in previous_records
            if record.get("path_name")
        }
        for record in records_to_write:
            merged_by_name[str(record["path_name"])] = record
        records_to_write = list(merged_by_name.values())

        def path_order(record: Mapping[str, Any]) -> Tuple[int, str]:
            try:
                index = record.get("dataset_index", record.get("path_index", 10**9))
                return int(index), str(record.get("path_name", ""))
            except (TypeError, ValueError):
                return 10**9, str(record.get("path_name", ""))

        records_to_write.sort(key=path_order)

    fieldnames = list(fallback_fields)
    for record in records_to_write:
        for key in record:
            if key not in fieldnames:
                fieldnames.append(key)

    temporary_path = path.with_name(f".{path.name}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(records_to_write)
        temporary_path.replace(path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


# Summary and split artifacts are replaced atomically so interrupted selective
# regeneration cannot leave a partially written aggregate dataset.
write_json = _atomic_write_json
write_records_csv = _atomic_write_records_csv


def main() -> int:
    args = parse_args()
    validate_args(args)
    args.input_npz = args.input_npz or default_input_npz(args.split)
    num_ik_retries = (
        SEQUENTIAL_IK_DEFAULT_RETRIES
        if args.num_ik_retries is None
        else args.num_ik_retries
    )
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = resolve_device(args.device)
    dataset = load_generation_dataset(args.input_npz)
    indices = select_indices(dataset, args)
    checkpoint_path = args.checkpoint.resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    checkpoint_hash = sha256_file(checkpoint_path)
    model, checkpoint = load_model(checkpoint_path, device)
    if int(checkpoint.get("output_dim", JOINT_DIM)) != JOINT_DIM:
        raise ValueError("Canonical MLP checkpoint output_dim must be 6")
    if int(checkpoint["num_steps"]) != dataset.desired_paths.shape[1]:
        raise ValueError("Checkpoint num_steps differs from the input dataset")
    interpretation, interpretation_provenance = checkpoint_target_interpretation(
        checkpoint, checkpoint_path
    )
    model.eval()

    joint_names = tuple(DEFAULT_JOINT_NAMES)
    if len(joint_names) != JOINT_DIM or len(set(joint_names)) != JOINT_DIM:
        raise ValueError("Exactly six unique xMateCR7 joints are required")
    urdf_path = resolve_urdf_path(args.urdf)
    robot = load_robot(urdf_path)
    bounds = get_joint_bounds(robot, joint_names, -np.pi, np.pi)
    lower = np.asarray([bound[0] for bound in bounds], dtype=np.float64)
    upper = np.asarray([bound[1] for bound in bounds], dtype=np.float64)
    if args.joint_limit_repair and np.any(
        lower + args.joint_limit_margin > upper - args.joint_limit_margin
    ):
        raise ValueError(
            "--joint_limit_margin leaves an empty safe interval for at least one joint"
        )

    split_root = args.output_root / args.split
    summary_root = args.output_root / "summaries"
    metadata_root = args.output_root / "metadata"
    prior_npz = args.output_root / f"{args.split}_prior.npz"
    if prior_npz.exists() and not (
        args.resume or args.overwrite or args.overwrite_selected
    ):
        raise FileExistsError(
            f"{prior_npz} exists; pass --resume or --overwrite"
        )
    results: List[Dict[str, Any]] = []
    selected_names: List[str] = []
    selected_desired: List[np.ndarray] = []

    with torch.no_grad():
        for position, dataset_index in enumerate(indices, start=1):
            path_name = dataset.names[dataset_index]
            selected_names.append(path_name)
            desired = dataset.desired_paths[dataset_index].astype(np.float64)
            times = dataset.times[dataset_index].astype(np.float64)
            selected_desired.append(desired)
            path_dir = split_root / safe_path_name(path_name)
            existing = existing_path_outputs(path_dir)
            if existing and args.resume and not args.overwrite_selected:
                result = load_resumed_result(
                    path_dir=path_dir,
                    desired=desired,
                    times=times,
                    lower=lower,
                    upper=upper,
                    joint_names=joint_names,
                    mean_error_gate=args.mean_error_gate,
                    max_joint_step_gate=args.max_joint_step_gate,
                )
            else:
                if existing and not (args.overwrite or args.overwrite_selected):
                    raise FileExistsError(
                        f"{path_dir} already contains method-specific output; "
                        "pass --resume or --overwrite"
                    )
                result = generate_path(
                    split=args.split,
                    dataset_index=dataset_index,
                    path_name=path_name,
                    desired=desired,
                    times=times,
                    q_start=dataset.q_start[dataset_index],
                    path_dir=path_dir,
                    model=model,
                    checkpoint=checkpoint,
                    checkpoint_path=checkpoint_path,
                    checkpoint_hash=checkpoint_hash,
                    times_source=dataset.times_source,
                    device=device,
                    robot=robot,
                    urdf_path=urdf_path,
                    joint_names=joint_names,
                    ee_link=args.ee_link,
                    bounds=bounds,
                    lower=lower,
                    upper=upper,
                    max_allowed_joint_step=args.max_allowed_joint_step,
                    strict_joint_step=args.strict_joint_step,
                    num_ik_retries=num_ik_retries,
                    random_seed=args.seed + dataset_index * 100_003,
                    overwrite=args.overwrite or args.overwrite_selected,
                    retry_profile=args.retry_profile,
                    mean_error_gate=args.mean_error_gate,
                    max_joint_step_gate=args.max_joint_step_gate,
                    local_repair=args.local_repair,
                    local_repair_radius=args.local_repair_radius,
                    local_repair_max_passes=args.local_repair_max_passes,
                    bridge_step_target=args.bridge_step_target,
                    joint_limit_repair=args.joint_limit_repair,
                    joint_limit_margin=args.joint_limit_margin,
                    joint_limit_repair_radius=args.joint_limit_repair_radius,
                    joint_limit_repair_passes=args.joint_limit_repair_passes,
                )
            results.append(result)
            print(
                f"[{position}/{len(indices)}] {path_name}: "
                f"status={result['metadata'].get('generation_status', 'unknown')} "
                f"mean_cart={result['metrics']['mean_cartesian_error']:.8e} "
                f"max_abs_step={result['metrics']['maximum_absolute_joint_step']:.6f}"
            )

    # This is deliberately after every practical trajectory has been generated.
    if args.save_expert_evaluation:
        expert_all = load_expert_after_generation(
            args.input_npz,
            (
                len(dataset.names),
                dataset.desired_paths.shape[1],
                JOINT_DIM,
            ),
        )
        for dataset_index, path_name, result in zip(indices, selected_names, results):
            joint_rmse = float(
                np.sqrt(
                    np.mean(
                        np.square(result["prior_q"] - expert_all[dataset_index])
                    )
                )
            )
            result["metadata"]["joint_rmse_vs_expert"] = joint_rmse
            write_json(
                split_root / safe_path_name(path_name) / "generation_metadata.json",
                result["metadata"],
            )

    selected_results_by_index = dict(zip(indices, results))
    aggregate_indices = list(range(len(dataset.names)))
    aggregate_names = [dataset.names[index] for index in aggregate_indices]
    aggregate_desired = [
        dataset.desired_paths[index].astype(np.float64)
        for index in aggregate_indices
    ]
    aggregate_results: List[Dict[str, Any]] = []
    for dataset_index in aggregate_indices:
        selected_result = selected_results_by_index.get(dataset_index)
        if selected_result is not None:
            aggregate_results.append(selected_result)
            continue
        path_name = dataset.names[dataset_index]
        path_dir = split_root / safe_path_name(path_name)
        if not existing_path_outputs(path_dir):
            raise FileNotFoundError(
                f"Cannot rebuild complete {args.split} artifacts: unselected path "
                f"{path_name} has no existing output in {path_dir}"
            )
        aggregate_results.append(
            load_resumed_result(
                path_dir=path_dir,
                desired=dataset.desired_paths[dataset_index].astype(np.float64),
                times=dataset.times[dataset_index].astype(np.float64),
                lower=lower,
                upper=upper,
                joint_names=joint_names,
                mean_error_gate=args.mean_error_gate,
                max_joint_step_gate=args.max_joint_step_gate,
            )
        )

    if len(aggregate_results) != len(dataset.names):
        raise RuntimeError(
            f"Expected {len(dataset.names)} aggregate paths, got "
            f"{len(aggregate_results)}"
        )

    summary_rows = [
        summary_row(
            split=args.split,
            dataset_index=dataset_index,
            path_name=path_name,
            result=result,
            checkpoint=checkpoint_path,
        )
        for dataset_index, path_name, result in zip(
            aggregate_indices, aggregate_names, aggregate_results
        )
    ]
    branch_rows = [
        branch_row(args.split, dataset_index, path_name, result)
        for dataset_index, path_name, result in zip(
            aggregate_indices, aggregate_names, aggregate_results
        )
    ]
    failed_rows = [
        {
            "split": row["split"],
            "dataset_index": row["dataset_index"],
            "path_name": row["path_name"],
            "generation_status": row["generation_status"],
            "failed_timestep_count": row["failed_timestep_count"],
            "maximum_absolute_joint_step": row["maximum_absolute_joint_step"],
        }
        for row in summary_rows
        if not bool(row["generation_success"])
    ]
    reproduction_rows: List[Dict[str, Any]] = []
    if args.split == "test":
        reproduction_rows = [
            reproduction_row(
                path_name=path_name,
                new_q=result["prior_q"],
                new_ee=result["prior_ee"],
                desired=desired,
                old_root=DEFAULT_OLD_TEST_ROOT,
                robot=robot,
                joint_names=joint_names,
                ee_link=args.ee_link,
            )
            for path_name, result, desired in zip(
                aggregate_names, aggregate_results, aggregate_desired
            )
        ]

    summary_path = summary_root / f"generation_summary_{args.split}.csv"
    write_records_csv(
        summary_path,
        summary_rows,
        ("split", "dataset_index", "path_name", "generation_success", "generation_status"),
    )
    merge_split_rows(
        summary_root / "branch_jump_summary.csv",
        args.split,
        branch_rows,
        ("split", "dataset_index", "path_name", "maximum_absolute_joint_step"),
    )
    merge_split_rows(
        summary_root / "failed_paths.csv",
        args.split,
        failed_rows,
        ("split", "dataset_index", "path_name", "generation_status"),
    )
    reproduction_path = summary_root / "reproduction_audit_test.csv"
    if args.split == "test":
        write_records_csv(
            reproduction_path,
            reproduction_rows,
            ("split", "path_name", "old_trajectory_exists"),
        )

    save_prior_npz(
        path=prior_npz,
        names=aggregate_names,
        desired_paths=np.stack(aggregate_desired),
        results=aggregate_results,
        checkpoint=checkpoint_path,
        overwrite_allowed=args.overwrite or args.resume or args.overwrite_selected,
    )

    adaptive_ambiguities = {
        "num_ik_retries": (
            "adaptive_refine_mlp_predictions_with_ik.py defines no per-timestep "
            "retry count; resolved from generate_ik_seed_path.py default=8"
        ),
        "damping": (
            "no damping parameter exists in the repository L-BFGS-B IK solver; "
            "smooth_weight is the available continuity regularizer"
        ),
        "legacy_trajectory_provenance": (
            "fixed and adaptive runs shared refined_mlp_ik_q.csv, so the old "
            "test trajectory's last writer is not provable"
        ),
    }
    configuration = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "split": args.split,
        "dataset_path": str(args.input_npz),
        "selected_dataset_indices": indices,
        "selected_path_names": selected_names,
        "resume": args.resume,
        "overwrite": args.overwrite,
        "overwrite_selected": args.overwrite_selected,
        "retry_profile": args.retry_profile,
        "mean_error_gate": args.mean_error_gate,
        "max_joint_step_gate": args.max_joint_step_gate,
        "local_repair": args.local_repair,
        "local_repair_radius": args.local_repair_radius,
        "local_repair_max_passes": args.local_repair_max_passes,
        "bridge_step_target": args.bridge_step_target,
        "joint_limit_repair": args.joint_limit_repair,
        "joint_limit_margin": args.joint_limit_margin,
        "joint_limit_repair_radius": args.joint_limit_repair_radius,
        "joint_limit_repair_passes": args.joint_limit_repair_passes,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_target_interpretation": interpretation,
        "checkpoint_target_interpretation_provenance": interpretation_provenance,
        "canonical_output_formula": (
            "model_output_normalized * checkpoint_y_std + checkpoint_y_mean"
            if interpretation == "full_q"
            else "q_start + model_output_normalized * checkpoint_y_std + checkpoint_y_mean"
        ),
        "repository_scripts_reused": [
            "predict_path_conditioned_mlp.py",
            "train_path_conditioned_mlp.py",
            "generate_ik_seed_path.py",
            "refine_mlp_predictions_with_ik.py",
            "adaptive_refine_mlp_predictions_with_ik.py",
        ],
        "ik_parameters": {
            "stage1_smooth_weight": ADAPTIVE_STAGE1_SMOOTH_WEIGHT,
            "stage1_max_iters": ADAPTIVE_STAGE1_MAX_ITERS,
            "stage2_smooth_weight": ADAPTIVE_STAGE2_SMOOTH_WEIGHT,
            "stage2_max_iters": ADAPTIVE_STAGE2_MAX_ITERS,
            "stage2_max_error_threshold": ADAPTIVE_STAGE2_MAX_ERROR_THRESHOLD,
            "ftol": IK_FTOL,
            "cartesian_tolerance": IK_CARTESIAN_TOLERANCE,
            "num_ik_retries": num_ik_retries,
            "max_allowed_joint_step": args.max_allowed_joint_step,
            "strict_joint_step": args.strict_joint_step,
        },
        "adaptive_rules": {
            "primary_seed": "previous accepted refined solution",
            "first_seed": "canonical MLP q[0]",
            "fallback_seeds": (
                "corresponding canonical MLP point, previous/MLP blends, then "
                "repository uniform restarts"
            ),
            "selection": (
                "among Cartesian-tolerant and step-safe attempts prefer smallest "
                "joint displacement; otherwise retain best safe attempt"
            ),
            "stage2": (
                "rerun full path with smooth_weight=0.001,max_iters=500 when "
                "stage1 maximum Cartesian error exceeds 0.03 m"
            ),
            "expert_information": "not loaded until optional post-generation evaluation",
        },
        "adaptive_parameter_provenance_ambiguities": adaptive_ambiguities,
        "urdf": str(urdf_path),
        "end_effector_link": args.ee_link,
        "joint_ordering": list(joint_names),
        "joint_bounds": [list(bound) for bound in bounds],
        "requested_device": args.device,
        "mlp_device": str(device),
        "ik_device": "cpu/scipy",
        "times_source": dataset.times_source,
        "random_seed": args.seed,
    }
    configuration_path = metadata_root / "generation_configuration.json"
    update_configuration(configuration_path, args.split, configuration)

    successful = sum(bool(row["generation_success"]) for row in summary_rows)
    aggregate_mean = float(
        np.mean([float(row["mean_cartesian_error"]) for row in summary_rows])
    )
    aggregate_max = float(
        np.max([float(row["maximum_cartesian_error"]) for row in summary_rows])
    )
    above_02 = sum(
        float(row["maximum_absolute_joint_step"]) > 0.2 for row in summary_rows
    )
    above_05 = sum(
        float(row["maximum_absolute_joint_step"]) > 0.5 for row in summary_rows
    )
    above_10 = sum(
        float(row["maximum_absolute_joint_step"]) > 1.0 for row in summary_rows
    )
    violations = sum(
        int(row["joint_limit_violation_count"]) for row in summary_rows
    )
    result_class = classification(summary_rows, len(dataset.names))
    print("\nAdaptive MLP + IK bootstrap-prior summary")
    print(f"selected/regenerated paths: {len(indices)}")
    print(f"aggregate paths: {len(summary_rows)}")
    print(f"successful paths: {successful}")
    print(f"failed paths: {len(summary_rows) - successful}")
    print(f"aggregate mean Cartesian error: {aggregate_mean:.8e} m")
    print(f"aggregate maximum Cartesian error: {aggregate_max:.8e} m")
    print(f"paths exceeding 0.2-rad maximum joint step: {above_02}")
    print(f"paths exceeding 0.5 rad: {above_05}")
    print(f"paths exceeding 1.0 rad: {above_10}")
    print(f"joint-limit violations: {violations}")
    print(f"classification: {result_class}")
    print(f"prior NPZ: {prior_npz}")
    print(f"generation summary: {summary_path}")
    print(f"branch-jump summary: {summary_root / 'branch_jump_summary.csv'}")
    print(f"failed paths: {summary_root / 'failed_paths.csv'}")
    if args.split == "test":
        print(f"reproduction audit: {reproduction_path}")
    print(f"configuration: {configuration_path}")
    print(f"per-path outputs: {split_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
