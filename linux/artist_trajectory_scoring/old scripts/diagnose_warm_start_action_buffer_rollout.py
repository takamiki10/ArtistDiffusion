#!/usr/bin/env python3
"""Diagnose action-buffer warm starts for receding-horizon xMateCR7 rollouts.

The rollout convention used here is intentionally simple:

* A candidate chunk has shape ``(H, 6)`` and represents the next commanded
  samples beginning at ``current_index``.
* The already executed sample at ``current_index - 1`` is not duplicated in the
  next chunk. Boundary consistency is handled by the ranking penalty
  ``||q_candidate[0] - q_previous_executed||_2``.
* Each cycle executes only the first ``min(E, T - current_index)`` samples, then
  shifts the unused tail of the selected chunk and extends it back to ``H``.

Expert joint trajectories are used only after rollout for diagnostics.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from build_diffusion_v5b_residual_window_dataset_fk_condition import (
    CONDITION_DIM as V5B_CONDITION_DIM,
    RESIDUAL_DIM,
    build_condition_window as build_v5b_condition_window,
    desired_finite_difference,
)
from diagnose_diffusion_v5_sampling_modes import (
    forward_noise_at_t,
    reverse_from_initial,
)
from evaluate_prior_refinement_fk_robot_costs import (
    FKComputer,
    Weights,
    drawing_fidelity_metrics,
    drawing_total_cost,
    path_length,
    q_error_metrics,
    shape_path_metrics,
    shape_total_cost,
    smoothness_costs,
)
from sample_conditional_diffusion_trajectory_v5_residual_unet import (
    diffusion_config_from_checkpoint,
    instantiate_checkpoint_model,
    load_residual_stats,
    make_schedule,
    safe_path_name,
    torch_load_checkpoint,
)


DEFAULT_TEST_NPZ = Path("data/cartesian_expert_dataset_v3/diffusion_v2/diffusion_test_v2.npz")
DEFAULT_WINDOW_NPZ = Path(
    "data/cartesian_expert_dataset_v3/diffusion_v5b_residual_windows_fk_condition/test_windows.npz"
)
DEFAULT_STATS_NPZ = Path(
    "data/cartesian_expert_dataset_v3/diffusion_v5b_residual_windows_fk_condition/normalization_stats.npz"
)
DEFAULT_DIFFUSION_CHECKPOINT = Path(
    "data/cartesian_expert_dataset_v3/diffusion_v5b_residual_unet_fk_condition/best_checkpoint.pt"
)
DEFAULT_OUTPUT_DIR = Path("data/cartesian_expert_dataset_v3/warm_start_action_buffer_diagnostic")
DEFAULT_PRIOR_DIR = Path("data/cartesian_expert_dataset_v3/mlp_v3_test_predictions")
JOINT_COLUMNS = ("q1", "q2", "q3", "q4", "q5", "q6")
XYZ_COLUMNS = ("x", "y", "z")
METHODS = ("bootstrap_prior", "buffer_only", "buffer_plus_diffusion", "expert")
MAIN_DELTA_METRICS = (
    "mean_cartesian_error",
    "rms_cartesian_error",
    "max_cartesian_error",
    "drawing_total_cost",
    "mean_joint_velocity_magnitude",
    "mean_joint_acceleration_magnitude",
    "mean_joint_jerk_magnitude",
    "max_joint_step",
    "mean_boundary_discontinuity",
    "joint_rmse_vs_expert",
)
PRIMARY_AGGREGATE_METRICS = (
    "mean_cartesian_error",
    "drawing_total_cost",
    "max_joint_step",
    "mean_boundary_discontinuity",
)
EPS = 1e-12


@dataclass(frozen=True)
class NormalizationStats:
    condition_mean: np.ndarray
    condition_std: np.ndarray
    residual_mean: np.ndarray
    residual_std: np.ndarray


@dataclass(frozen=True)
class RolloutConfig:
    prediction_horizon: int
    execution_horizon: int
    t_init: int
    num_candidates: int
    tail_extension: str
    no_diffusion: bool


@dataclass(frozen=True)
class CandidateScore:
    candidate_index: int
    is_diffusion_refined: bool
    ranking_score: float
    mean_cartesian_error: float
    rms_cartesian_error: float
    max_cartesian_error: float
    joint_velocity_cost: float
    joint_acceleration_cost: float
    joint_jerk_cost: float
    max_abs_joint_step: float
    joint_limit_violation_count: int
    joint_limit_violation_magnitude: float
    raw_joint_limit_violation_count: int
    raw_joint_limit_violation_magnitude: float
    boundary_discontinuity: float
    drawing_total_cost: float


@dataclass
class ScoredCandidate:
    raw_q: np.ndarray
    q: np.ndarray
    score: CandidateScore


@dataclass
class RolloutResult:
    q: np.ndarray
    ee: np.ndarray
    cycle_rows: List[Dict[str, Any]]
    boundary_discontinuities: List[float]
    selected_diffusion_flags: List[bool]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Receding-horizon action-buffer warm-start diagnostic for the 6-DoF xMateCR7."
    )
    parser.add_argument("--test_npz", type=Path, default=DEFAULT_TEST_NPZ)
    parser.add_argument("--window_npz", type=Path, default=DEFAULT_WINDOW_NPZ)
    parser.add_argument("--normalization_stats", type=Path, default=DEFAULT_STATS_NPZ)
    parser.add_argument("--diffusion_checkpoint", type=Path, default=DEFAULT_DIFFUSION_CHECKPOINT)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--prediction_horizon", type=int, default=32, choices=(16, 32))
    parser.add_argument("--execution_horizon", type=int, default=4, choices=(4, 8))
    parser.add_argument("--t_init", type=int, default=5, choices=(5, 10))
    parser.add_argument("--num_candidates", type=int, default=8)
    parser.add_argument("--max_paths", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--tail_extension",
        choices=("constant_position", "constant_velocity", "bootstrap_prior"),
        default="constant_velocity",
    )
    parser.add_argument("--continuity_weight", type=float, default=1.0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--path_indices", type=str, default="")
    parser.add_argument("--urdf", type=Path, default=None)
    parser.add_argument("--ee_link", type=str, default=None)
    parser.add_argument("--save_candidate_details", action="store_true")
    parser.add_argument("--no_diffusion", action="store_true")
    parser.add_argument(
        "--prior_dir",
        type=Path,
        default=DEFAULT_PRIOR_DIR,
        help="Fallback MLP prior directory used by the v5/v5b builders when no full prior key is present.",
    )
    parser.add_argument("--w_cart", type=float, default=1.0)
    parser.add_argument("--w_rms_cart", type=float, default=0.0)
    parser.add_argument("--w_max", type=float, default=0.25)
    parser.add_argument("--w_vel", type=float, default=0.01)
    parser.add_argument("--w_acc", type=float, default=0.01)
    parser.add_argument("--w_jerk", type=float, default=0.001)
    parser.add_argument("--w_limit", type=float, default=10.0)
    parser.add_argument("--w_limit_count", type=float, default=10.0)
    parser.add_argument("--w_start", type=float, default=0.5)
    parser.add_argument("--w_end", type=float, default=0.5)
    parser.add_argument("--w_frechet", type=float, default=1.0)
    parser.add_argument("--w_dtw", type=float, default=0.5)
    parser.add_argument("--w_tangent", type=float, default=0.5)
    parser.add_argument("--w_progress", type=float, default=0.5)
    parser.add_argument("--w_length_ratio", type=float, default=0.25)
    parser.add_argument("--w_norm_shape", type=float, default=1.0)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def set_candidate_seed(seed: int, path_index: int, cycle_index: int, candidate_index: int) -> None:
    candidate_seed = (
        int(seed)
        + 1_000_003 * int(path_index)
        + 9_176 * int(cycle_index)
        + 37 * int(candidate_index)
    ) % (2**31 - 1)
    torch.manual_seed(candidate_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(candidate_seed)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        print("[device] CUDA requested but unavailable; using CPU")
        return torch.device("cpu")
    return torch.device(device_arg)


def load_npz(path: Path, label: str) -> Dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    with np.load(path, allow_pickle=True) as data:
        out = {key: data[key] for key in data.files}
    print_npz_summary(label, path, out)
    return out


def print_npz_summary(label: str, path: Path, data: Mapping[str, np.ndarray]) -> None:
    print(f"[{label}] {path}")
    for key in sorted(data):
        value = data[key]
        print(f"  {key}: shape={value.shape}, dtype={value.dtype}")


def decode_names(raw: np.ndarray, count: int) -> List[str]:
    if raw.shape[0] != count:
        raise ValueError(f"path_names length {raw.shape[0]} does not match path count {count}")
    names: List[str] = []
    for item in np.asarray(raw):
        if isinstance(item, bytes):
            names.append(item.decode("utf-8", errors="replace"))
        else:
            names.append(str(item))
    return names


def require_finite(name: str, values: np.ndarray) -> None:
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} contains NaN or infinity")


def resolve_array_key(
    data: Mapping[str, np.ndarray],
    candidates: Sequence[str],
    label: str,
    ndim: Optional[int] = None,
    last_dim: Optional[int] = None,
) -> Tuple[str, np.ndarray]:
    for key in candidates:
        if key not in data:
            continue
        arr = np.asarray(data[key])
        if ndim is not None and arr.ndim != ndim:
            continue
        if last_dim is not None and (arr.ndim == 0 or arr.shape[-1] != last_dim):
            continue
        return key, arr
    raise KeyError(
        f"Could not resolve {label}. Tried keys: {', '.join(candidates)}"
    )


def load_full_dataset(data: Mapping[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], List[str]]:
    desired_key, desired = resolve_array_key(
        data,
        ("desired_paths", "desired_path", "cartesian_paths", "cartesian_path", "paths"),
        "desired Cartesian paths",
        ndim=3,
        last_dim=3,
    )
    expert_key, expert_q = resolve_array_key(
        data,
        ("expert_q", "q_expert", "expert_joint_trajectories", "joint_trajectories"),
        "expert joint trajectories",
        ndim=3,
        last_dim=RESIDUAL_DIM,
    )
    desired = np.asarray(desired, dtype=np.float64)
    expert_q = np.asarray(expert_q, dtype=np.float64)
    if desired.shape[:2] != expert_q.shape[:2]:
        raise ValueError(
            f"{desired_key} and {expert_key} must share (N,T), got "
            f"{desired.shape[:2]} vs {expert_q.shape[:2]}"
        )

    q_start: Optional[np.ndarray] = None
    if "q_start" in data:
        q_start = np.asarray(data["q_start"], dtype=np.float64)
        if q_start.shape != (desired.shape[0], RESIDUAL_DIM):
            raise ValueError(f"q_start must have shape {(desired.shape[0], RESIDUAL_DIM)}, got {q_start.shape}")
        require_finite("q_start", q_start)

    if "path_names" in data:
        names = decode_names(np.asarray(data["path_names"]), desired.shape[0])
    else:
        names = [f"path_{idx:04d}" for idx in range(desired.shape[0])]

    require_finite(desired_key, desired)
    require_finite(expert_key, expert_q)
    return desired, expert_q, q_start, names


def parse_path_indices(raw: str, num_paths: int, max_paths: int) -> List[int]:
    if raw.strip():
        indices: List[int] = []
        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue
            if "-" in part:
                start_s, end_s = part.split("-", 1)
                start = int(start_s)
                end = int(end_s)
                step = 1 if end >= start else -1
                indices.extend(range(start, end + step, step))
            else:
                indices.append(int(part))
        out = []
        seen = set()
        for idx in indices:
            if idx < 0 or idx >= num_paths:
                raise IndexError(f"path index {idx} is outside [0,{num_paths})")
            if idx not in seen:
                out.append(idx)
                seen.add(idx)
        return out

    if max_paths < 0:
        raise ValueError("--max_paths must be non-negative")
    if max_paths == 0:
        return list(range(num_paths))
    return list(range(min(max_paths, num_paths)))


def load_normalization_stats(stats_npz: Path, checkpoint: Mapping[str, Any]) -> NormalizationStats:
    stats = load_npz(stats_npz, "normalization stats")
    missing = [
        key
        for key in ("condition_mean", "condition_std", "residual_mean", "residual_std")
        if key not in stats and key not in checkpoint
    ]
    if missing:
        raise KeyError(f"Missing normalization stat(s): {', '.join(missing)}")

    def get_stat(key: str) -> np.ndarray:
        source = stats if key in stats else checkpoint
        return np.asarray(source[key], dtype=np.float32)

    condition_mean = get_stat("condition_mean")
    condition_std = get_stat("condition_std")
    residual_mean = get_stat("residual_mean")
    residual_std = get_stat("residual_std")
    if condition_mean.shape != (V5B_CONDITION_DIM,) or condition_std.shape != (V5B_CONDITION_DIM,):
        raise ValueError(
            f"condition stats must have shape ({V5B_CONDITION_DIM},), got "
            f"{condition_mean.shape}/{condition_std.shape}"
        )
    if residual_mean.shape != (RESIDUAL_DIM,) or residual_std.shape != (RESIDUAL_DIM,):
        raise ValueError(
            f"residual stats must have shape ({RESIDUAL_DIM},), got "
            f"{residual_mean.shape}/{residual_std.shape}"
        )
    if np.any(condition_std <= 0.0) or np.any(residual_std <= 0.0):
        raise ValueError("normalization std arrays must be strictly positive")
    print(
        "[normalization] "
        f"condition_mean={condition_mean.shape}, condition_std={condition_std.shape}, "
        f"residual_mean={residual_mean.shape}, residual_std={residual_std.shape}"
    )
    return NormalizationStats(condition_mean, condition_std, residual_mean, residual_std)


def try_full_prior_from_dataset(
    data: Mapping[str, np.ndarray],
    expected_shape: Tuple[int, int, int],
) -> Optional[Tuple[str, np.ndarray]]:
    for key in (
        "prior_q",
        "bootstrap_prior_q",
        "mlp_prior_q",
        "predicted_q",
        "prior_trajectories",
        "predicted_q_paths",
        "q_prior",
    ):
        if key not in data:
            continue
        arr = np.asarray(data[key], dtype=np.float64)
        if arr.shape == expected_shape:
            require_finite(key, arr)
            return key, arr
    return None


def reconstruct_prior_from_windows(
    window_data: Mapping[str, np.ndarray],
    full_names: Sequence[str],
    trajectory_length: int,
) -> Optional[np.ndarray]:
    required = ("prior_q_window", "path_names", "window_start_indices")
    if any(key not in window_data for key in required):
        return None

    prior_windows = np.asarray(window_data["prior_q_window"], dtype=np.float64)
    starts = np.asarray(window_data["window_start_indices"], dtype=np.int64)
    window_names = decode_names(np.asarray(window_data["path_names"]), prior_windows.shape[0])
    if prior_windows.ndim != 3 or prior_windows.shape[-1] != RESIDUAL_DIM:
        raise ValueError(f"prior_q_window must have shape (W,H,{RESIDUAL_DIM}), got {prior_windows.shape}")
    if starts.shape != (prior_windows.shape[0],):
        raise ValueError(
            f"window_start_indices shape must be {(prior_windows.shape[0],)}, got {starts.shape}"
        )

    out = np.full((len(full_names), trajectory_length, RESIDUAL_DIM), np.nan, dtype=np.float64)
    counts = np.zeros((len(full_names), trajectory_length, 1), dtype=np.float64)
    name_to_index = {name: idx for idx, name in enumerate(full_names)}
    safe_to_index = {safe_path_name(name): idx for idx, name in enumerate(full_names)}

    for window, name, start in zip(prior_windows, window_names, starts):
        idx = name_to_index.get(name)
        if idx is None:
            idx = safe_to_index.get(safe_path_name(name))
        if idx is None:
            continue
        if start < 0:
            raise ValueError(f"negative window_start_index for {name}: {start}")
        end = min(int(start) + window.shape[0], trajectory_length)
        if end <= start:
            continue
        segment = window[: end - int(start)]
        existing = out[idx, int(start):end]
        mask = counts[idx, int(start):end] > 0
        existing[~mask[:, 0]] = segment[~mask[:, 0]]
        if np.any(mask):
            mismatch = np.max(np.abs(existing[mask[:, 0]] - segment[mask[:, 0]]))
            if mismatch > 1e-4:
                raise ValueError(
                    f"Overlapping prior windows disagree for {name} near start={start}; max mismatch={mismatch:.6e}"
                )
            existing[mask[:, 0]] = 0.5 * (existing[mask[:, 0]] + segment[mask[:, 0]])
        out[idx, int(start):end] = existing
        counts[idx, int(start):end] += 1.0

    missing_paths = [
        name for idx, name in enumerate(full_names) if np.any(counts[idx, :, 0] == 0.0)
    ]
    if missing_paths:
        print(
            "[prior] v5b window prior did not cover all samples for "
            f"{len(missing_paths)} path(s); trying other sources"
        )
        return None
    require_finite("reconstructed prior_q from window_npz", out)
    return out


def read_predicted_q_csv(path: Path, expected_steps: int) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header")
        field_map = {field.strip().lower(): field for field in reader.fieldnames if field is not None}
        missing = [column for column in JOINT_COLUMNS if column not in field_map]
        if missing:
            raise ValueError(f"{path} missing joint columns: {', '.join(missing)}")
        rows = [
            [float(row[field_map[column]]) for column in JOINT_COLUMNS]
            for row in reader
        ]
    arr = np.asarray(rows, dtype=np.float64)
    expected_shape = (expected_steps, RESIDUAL_DIM)
    if arr.shape != expected_shape:
        raise ValueError(f"{path} must contain shape {expected_shape}, got {arr.shape}")
    require_finite(str(path), arr)
    return arr


def load_prior_from_csv_dir(
    prior_dir: Path,
    names: Sequence[str],
    trajectory_length: int,
) -> Optional[np.ndarray]:
    if prior_dir is None or not prior_dir.exists():
        return None
    rows: List[np.ndarray] = []
    try:
        for name in names:
            rows.append(read_predicted_q_csv(prior_dir / safe_path_name(name) / "predicted_q.csv", trajectory_length))
    except FileNotFoundError:
        return None
    return np.stack(rows, axis=0)


def resolve_bootstrap_prior(
    *,
    test_data: Mapping[str, np.ndarray],
    window_data: Mapping[str, np.ndarray],
    names: Sequence[str],
    trajectory_length: int,
    prior_dir: Path,
) -> Tuple[np.ndarray, str]:
    expected_shape = (len(names), trajectory_length, RESIDUAL_DIM)
    full_prior = try_full_prior_from_dataset(test_data, expected_shape)
    if full_prior is not None:
        key, values = full_prior
        print(f"[prior] using full prior from test_npz key: {key}")
        return values, f"test_npz:{key}"

    reconstructed = reconstruct_prior_from_windows(window_data, names, trajectory_length)
    if reconstructed is not None:
        print("[prior] reconstructed full bootstrap prior from window_npz prior_q_window")
        return reconstructed, "window_npz:prior_q_window"

    csv_prior = load_prior_from_csv_dir(prior_dir, names, trajectory_length)
    if csv_prior is not None:
        print(f"[prior] using MLP prior CSVs from {prior_dir}")
        return csv_prior, f"prior_dir:{prior_dir}"

    raise KeyError(
        "Could not resolve a full bootstrap prior. Expected a full prior key in test_npz, "
        "covering prior_q_window data in window_npz, or predicted_q.csv files under --prior_dir."
    )


def finite_limits(fk: FKComputer) -> Tuple[np.ndarray, np.ndarray]:
    if fk.lower is None or fk.upper is None or len(fk.lower) < RESIDUAL_DIM:
        lower = np.full(RESIDUAL_DIM, -np.inf, dtype=np.float64)
        upper = np.full(RESIDUAL_DIM, np.inf, dtype=np.float64)
    else:
        lower = np.asarray(fk.lower[:RESIDUAL_DIM], dtype=np.float64)
        upper = np.asarray(fk.upper[:RESIDUAL_DIM], dtype=np.float64)
    return lower, upper


def require_fk_ready(fk: FKComputer) -> None:
    if not fk.available:
        raise RuntimeError(
            "FK is required for this diagnostic. Pass --urdf/--ee_link or install the existing xMateCR7 URDF context."
        )
    if len(fk.joint_names) < RESIDUAL_DIM:
        raise RuntimeError(f"Expected {RESIDUAL_DIM} active joints, got {len(fk.joint_names)}")


def clip_to_limits(q: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return np.minimum(np.maximum(q, lower.reshape(1, RESIDUAL_DIM)), upper.reshape(1, RESIDUAL_DIM))


def joint_limit_metrics(
    q: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> Tuple[int, float]:
    finite_lower = np.isfinite(lower).reshape(1, RESIDUAL_DIM)
    finite_upper = np.isfinite(upper).reshape(1, RESIDUAL_DIM)
    below = np.maximum(lower.reshape(1, RESIDUAL_DIM) - q, 0.0) * finite_lower
    above = np.maximum(q - upper.reshape(1, RESIDUAL_DIM), 0.0) * finite_upper
    violation = below + above
    return int(np.sum(violation > 0.0)), float(np.sum(violation))


def extend_joint_window(
    partial_q: np.ndarray,
    *,
    horizon: int,
    mode: str,
    bootstrap_prior: np.ndarray,
    current_index: int,
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    if partial_q.ndim != 2 or partial_q.shape[-1] != RESIDUAL_DIM:
        raise ValueError(f"partial_q must have shape (K,{RESIDUAL_DIM}), got {partial_q.shape}")
    if partial_q.shape[0] == 0:
        if current_index < bootstrap_prior.shape[0]:
            partial_q = bootstrap_prior[current_index: current_index + 1]
        else:
            partial_q = bootstrap_prior[-1:]
    values = [row.astype(np.float64).copy() for row in partial_q[:horizon]]

    def append_constant_position() -> None:
        values.append(values[-1].copy())

    def append_constant_velocity() -> None:
        if len(values) < 2:
            append_constant_position()
            return
        dq = values[-1] - values[-2]
        values.append(np.minimum(np.maximum(values[-1] + dq, lower), upper))

    while len(values) < horizon:
        if mode == "bootstrap_prior":
            future_idx = current_index + len(values)
            if future_idx < bootstrap_prior.shape[0]:
                values.append(bootstrap_prior[future_idx].astype(np.float64).copy())
                continue
            if len(values) >= 2:
                append_constant_velocity()
            else:
                append_constant_position()
        elif mode == "constant_velocity":
            append_constant_velocity()
        elif mode == "constant_position":
            append_constant_position()
        else:
            raise ValueError(f"Unknown tail extension mode: {mode}")

    out = np.stack(values[:horizon], axis=0)
    return clip_to_limits(out, lower, upper)


def slice_extend_array(values: np.ndarray, start: int, horizon: int) -> np.ndarray:
    end = min(start + horizon, values.shape[0])
    segment = values[start:end]
    if segment.shape[0] == 0:
        segment = values[-1:]
    rows = [row.copy() for row in segment]
    while len(rows) < horizon:
        rows.append(rows[-1].copy())
    return np.stack(rows, axis=0)


def build_condition_for_buffer(
    *,
    desired_path: np.ndarray,
    desired_diff: np.ndarray,
    progress: np.ndarray,
    q_start: np.ndarray,
    current_index: int,
    buffer_prior_q: np.ndarray,
    fk: FKComputer,
    stats: NormalizationStats,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    desired_window = slice_extend_array(desired_path, current_index, buffer_prior_q.shape[0]).astype(np.float32)
    desired_diff_window = slice_extend_array(desired_diff, current_index, buffer_prior_q.shape[0]).astype(np.float32)
    progress_window = slice_extend_array(progress.reshape(-1, 1), current_index, buffer_prior_q.shape[0])[:, 0].astype(np.float32)
    prior_ee = fk.fk(buffer_prior_q)
    if prior_ee is None:
        raise RuntimeError("FK unexpectedly unavailable while building condition")
    prior_ee = np.asarray(prior_ee, dtype=np.float32)
    prior_ee_error = (prior_ee - desired_window).astype(np.float32)
    condition = build_v5b_condition_window(
        desired_window=desired_window,
        desired_diff_window=desired_diff_window,
        progress_window=progress_window,
        q_start=q_start.astype(np.float32),
        current_q=buffer_prior_q[0].astype(np.float32),
        prior_q_window=buffer_prior_q.astype(np.float32),
        prior_ee_window=prior_ee,
        prior_ee_error=prior_ee_error,
    )
    condition_norm = (
        (condition - stats.condition_mean.reshape(1, V5B_CONDITION_DIM))
        / stats.condition_std.reshape(1, V5B_CONDITION_DIM)
    ).astype(np.float32)
    require_finite("condition_norm", condition_norm)
    return condition_norm, desired_window.astype(np.float64), prior_ee.astype(np.float64)


def normalize_residual(residual_q: np.ndarray, stats: NormalizationStats) -> np.ndarray:
    return (
        (residual_q - stats.residual_mean.reshape(1, RESIDUAL_DIM))
        / stats.residual_std.reshape(1, RESIDUAL_DIM)
    ).astype(np.float32)


def denormalize_residual(residual_norm: np.ndarray, stats: NormalizationStats) -> np.ndarray:
    return (
        residual_norm * stats.residual_std.reshape(1, RESIDUAL_DIM)
        + stats.residual_mean.reshape(1, RESIDUAL_DIM)
    ).astype(np.float64)


def refine_buffer_with_diffusion(
    *,
    model: torch.nn.Module,
    call_variant: str,
    schedule: Mapping[str, torch.Tensor],
    condition_norm_hc: np.ndarray,
    buffer_prior_q: np.ndarray,
    stats: NormalizationStats,
    t_init: int,
    device: torch.device,
) -> np.ndarray:
    zero_residual_q = np.zeros_like(buffer_prior_q, dtype=np.float64)
    residual_norm = normalize_residual(zero_residual_q, stats)
    condition = torch.from_numpy(condition_norm_hc[None]).to(device=device, dtype=torch.float32)
    x0 = torch.from_numpy(residual_norm[None]).to(device=device, dtype=torch.float32)
    condition_cf = condition.permute(0, 2, 1).contiguous()
    x0_cf = x0.permute(0, 2, 1).contiguous()
    x_t = forward_noise_at_t(x0_cf, t_init, dict(schedule))
    sampled = reverse_from_initial(
        model=model,
        call_variant=call_variant,
        x_init_cf=x_t,
        condition_cf=condition_cf,
        start_step=t_init,
        schedule=dict(schedule),
        deterministic=False,
    )
    residual_refined_norm = sampled.permute(0, 2, 1).detach().cpu().numpy()[0].astype(np.float32)
    residual_refined_q = denormalize_residual(residual_refined_norm, stats)
    candidate_q = buffer_prior_q + residual_refined_q
    require_finite("diffusion candidate", candidate_q)
    return candidate_q


def trajectory_derivative_magnitudes(q: np.ndarray) -> Tuple[float, float, float, float]:
    velocity = np.diff(q, axis=0)
    acceleration = np.diff(q, n=2, axis=0)
    jerk = np.diff(q, n=3, axis=0)
    mean_velocity = float(np.mean(np.linalg.norm(velocity, axis=1))) if velocity.size else 0.0
    mean_acceleration = float(np.mean(np.linalg.norm(acceleration, axis=1))) if acceleration.size else 0.0
    mean_jerk = float(np.mean(np.linalg.norm(jerk, axis=1))) if jerk.size else 0.0
    max_step = float(np.max(np.abs(velocity))) if velocity.size else 0.0
    return mean_velocity, mean_acceleration, mean_jerk, max_step


def cartesian_error_metrics(ee: np.ndarray, desired: np.ndarray) -> Tuple[float, float, float]:
    error = ee - desired
    distances = np.linalg.norm(error, axis=1)
    return (
        float(np.mean(distances)),
        float(np.sqrt(np.mean(np.square(distances)))),
        float(np.max(distances)),
    )


def compute_drawing_cost(
    *,
    q: np.ndarray,
    ee: np.ndarray,
    desired: np.ndarray,
    weights: Weights,
    limit_magnitude: float,
) -> float:
    mean_cart, _, max_cart = cartesian_error_metrics(ee, desired)
    start_error, end_error, path_length_pred, path_length_desired, _, frechet, dtw = shape_path_metrics(ee, desired)
    tangent_cosine, tangent_weighted, progress_error, length_ratio_error, norm_shape = drawing_fidelity_metrics(
        ee,
        desired,
        path_length_pred,
        path_length_desired,
    )
    del tangent_cosine
    vel_cost, acc_cost, jerk_cost = smoothness_costs(q)
    shape_cost = shape_total_cost(
        weights,
        mean_cart,
        max_cart,
        start_error,
        end_error,
        frechet,
        dtw,
        vel_cost,
        acc_cost,
        jerk_cost,
        limit_magnitude,
    )
    return drawing_total_cost(
        weights,
        shape_cost,
        tangent_weighted,
        progress_error,
        length_ratio_error,
        norm_shape,
    )


def score_candidate(
    *,
    raw_q: np.ndarray,
    candidate_index: int,
    is_diffusion_refined: bool,
    desired_window: np.ndarray,
    previous_executed_q: Optional[np.ndarray],
    fk: FKComputer,
    lower: np.ndarray,
    upper: np.ndarray,
    weights: Weights,
    w_rms_cart: float,
    w_limit_count: float,
    continuity_weight: float,
) -> ScoredCandidate:
    if raw_q.ndim != 2 or raw_q.shape[-1] != RESIDUAL_DIM:
        raise ValueError(f"candidate q must have shape (H,{RESIDUAL_DIM}), got {raw_q.shape}")

    raw_limit_count, raw_limit_magnitude = joint_limit_metrics(raw_q, lower, upper)
    q = clip_to_limits(raw_q.astype(np.float64), lower, upper)
    limit_count, limit_magnitude = joint_limit_metrics(q, lower, upper)
    if not np.all(np.isfinite(q)):
        score = CandidateScore(
            candidate_index=candidate_index,
            is_diffusion_refined=is_diffusion_refined,
            ranking_score=float("inf"),
            mean_cartesian_error=float("inf"),
            rms_cartesian_error=float("inf"),
            max_cartesian_error=float("inf"),
            joint_velocity_cost=float("inf"),
            joint_acceleration_cost=float("inf"),
            joint_jerk_cost=float("inf"),
            max_abs_joint_step=float("inf"),
            joint_limit_violation_count=limit_count,
            joint_limit_violation_magnitude=limit_magnitude,
            raw_joint_limit_violation_count=raw_limit_count,
            raw_joint_limit_violation_magnitude=raw_limit_magnitude,
            boundary_discontinuity=float("inf"),
            drawing_total_cost=float("inf"),
        )
        return ScoredCandidate(raw_q=raw_q, q=q, score=score)

    ee = fk.fk(q)
    if ee is None:
        raise RuntimeError("FK unexpectedly unavailable during candidate scoring")
    mean_cart, rms_cart, max_cart = cartesian_error_metrics(ee, desired_window)
    vel_cost, acc_cost, jerk_cost = smoothness_costs(q)
    velocity = np.diff(q, axis=0)
    max_abs_step = float(np.max(np.abs(velocity))) if velocity.size else 0.0
    if previous_executed_q is None:
        boundary = 0.0
    else:
        boundary = float(np.linalg.norm(q[0] - previous_executed_q))
    drawing_cost = compute_drawing_cost(
        q=q,
        ee=ee,
        desired=desired_window,
        weights=weights,
        limit_magnitude=limit_magnitude + raw_limit_magnitude,
    )
    ranking = (
        weights.cart * mean_cart
        + w_rms_cart * rms_cart
        + weights.max_cart * max_cart
        + weights.vel * vel_cost
        + weights.acc * acc_cost
        + weights.jerk * jerk_cost
        + weights.limit * (limit_magnitude + raw_limit_magnitude)
        + w_limit_count * float(limit_count + raw_limit_count)
        + continuity_weight * boundary
    )
    score = CandidateScore(
        candidate_index=candidate_index,
        is_diffusion_refined=is_diffusion_refined,
        ranking_score=float(ranking),
        mean_cartesian_error=mean_cart,
        rms_cartesian_error=rms_cart,
        max_cartesian_error=max_cart,
        joint_velocity_cost=float(vel_cost),
        joint_acceleration_cost=float(acc_cost),
        joint_jerk_cost=float(jerk_cost),
        max_abs_joint_step=max_abs_step,
        joint_limit_violation_count=limit_count,
        joint_limit_violation_magnitude=limit_magnitude,
        raw_joint_limit_violation_count=raw_limit_count,
        raw_joint_limit_violation_magnitude=raw_limit_magnitude,
        boundary_discontinuity=boundary,
        drawing_total_cost=float(drawing_cost),
    )
    return ScoredCandidate(raw_q=raw_q, q=q, score=score)


def candidate_score_to_row(score: CandidateScore) -> Dict[str, Any]:
    return {
        "candidate_index": score.candidate_index,
        "is_diffusion_refined": int(score.is_diffusion_refined),
        "candidate_ranking_score": score.ranking_score,
        "candidate_mean_cartesian_error": score.mean_cartesian_error,
        "candidate_rms_cartesian_error": score.rms_cartesian_error,
        "candidate_max_cartesian_error": score.max_cartesian_error,
        "candidate_joint_velocity_cost": score.joint_velocity_cost,
        "candidate_joint_acceleration_cost": score.joint_acceleration_cost,
        "candidate_joint_jerk_cost": score.joint_jerk_cost,
        "candidate_max_abs_joint_step": score.max_abs_joint_step,
        "candidate_joint_limit_violation_count": score.joint_limit_violation_count,
        "candidate_joint_limit_violation_magnitude": score.joint_limit_violation_magnitude,
        "candidate_raw_joint_limit_violation_count": score.raw_joint_limit_violation_count,
        "candidate_raw_joint_limit_violation_magnitude": score.raw_joint_limit_violation_magnitude,
        "candidate_boundary_discontinuity": score.boundary_discontinuity,
        "candidate_drawing_total_cost": score.drawing_total_cost,
    }


def run_rollout(
    *,
    method: str,
    path_index: int,
    path_name: str,
    desired_path: np.ndarray,
    q_start: np.ndarray,
    bootstrap_prior_q: np.ndarray,
    fk: FKComputer,
    lower: np.ndarray,
    upper: np.ndarray,
    stats: NormalizationStats,
    model: Optional[torch.nn.Module],
    call_variant: Optional[str],
    schedule: Optional[Mapping[str, torch.Tensor]],
    device: torch.device,
    config: RolloutConfig,
    weights: Weights,
    w_rms_cart: float,
    w_limit_count: float,
    continuity_weight: float,
    seed: int,
    save_candidate_details: bool,
) -> RolloutResult:
    horizon = config.prediction_horizon
    execution_horizon = config.execution_horizon
    desired_diff = desired_finite_difference(desired_path.astype(np.float32)).astype(np.float64)
    progress = np.linspace(0.0, 1.0, desired_path.shape[0], dtype=np.float64)
    current_index = 0
    cycle_index = 0
    previous_executed_q: Optional[np.ndarray] = None
    executed_rows: List[np.ndarray] = []
    cycle_rows: List[Dict[str, Any]] = []
    boundaries: List[float] = []
    selected_diffusion: List[bool] = []
    buffer_q = extend_joint_window(
        bootstrap_prior_q[:horizon],
        horizon=horizon,
        mode=config.tail_extension,
        bootstrap_prior=bootstrap_prior_q,
        current_index=0,
        lower=lower,
        upper=upper,
    )

    while current_index < desired_path.shape[0]:
        execute_count = min(execution_horizon, desired_path.shape[0] - current_index)
        condition_norm, desired_window, _ = build_condition_for_buffer(
            desired_path=desired_path,
            desired_diff=desired_diff,
            progress=progress,
            q_start=q_start,
            current_index=current_index,
            buffer_prior_q=buffer_q,
            fk=fk,
            stats=stats,
        )

        raw_candidates: List[Tuple[np.ndarray, bool]] = [(buffer_q.copy(), False)]
        if method == "buffer_plus_diffusion" and not config.no_diffusion:
            if model is None or call_variant is None or schedule is None:
                raise RuntimeError("Diffusion rollout requested without a loaded model")
            for candidate_offset in range(config.num_candidates):
                candidate_index = candidate_offset + 1
                set_candidate_seed(seed, path_index, cycle_index, candidate_index)
                with torch.no_grad():
                    raw_candidates.append(
                        (
                            refine_buffer_with_diffusion(
                                model=model,
                                call_variant=call_variant,
                                schedule=schedule,
                                condition_norm_hc=condition_norm,
                                buffer_prior_q=buffer_q,
                                stats=stats,
                                t_init=config.t_init,
                                device=device,
                            ),
                            True,
                        )
                    )

        scored = [
            score_candidate(
                raw_q=raw_q,
                candidate_index=idx,
                is_diffusion_refined=is_diffusion,
                desired_window=desired_window,
                previous_executed_q=previous_executed_q,
                fk=fk,
                lower=lower,
                upper=upper,
                weights=weights,
                w_rms_cart=w_rms_cart,
                w_limit_count=w_limit_count,
                continuity_weight=continuity_weight,
            )
            for idx, (raw_q, is_diffusion) in enumerate(raw_candidates)
        ]
        selected = min(scored, key=lambda item: item.score.ranking_score)
        selected_score = selected.score
        selected_diffusion.append(selected_score.is_diffusion_refined)
        boundaries.append(selected_score.boundary_discontinuity)

        base_row = {
            "path_index": path_index,
            "path_name": path_name,
            "method": method,
            "cycle_index": cycle_index,
            "rollout_start_index": current_index,
            "executed_step_count": execute_count,
            "candidate_count": len(scored),
            "selected_candidate_index": selected_score.candidate_index,
            "selected_is_diffusion_refined": int(selected_score.is_diffusion_refined),
            "buffer_extension_mode": config.tail_extension,
            "H": horizon,
            "E": execution_horizon,
            "t_init": config.t_init,
        }
        if save_candidate_details:
            for candidate in scored:
                row = dict(base_row)
                row.update(candidate_score_to_row(candidate.score))
                row["selected_candidate_row"] = int(candidate.score.candidate_index == selected_score.candidate_index)
                cycle_rows.append(row)
        else:
            row = dict(base_row)
            row.update(candidate_score_to_row(selected_score))
            row["selected_candidate_row"] = 1
            cycle_rows.append(row)

        executed_rows.extend(row.copy() for row in selected.q[:execute_count])
        previous_executed_q = selected.q[execute_count - 1].copy()
        current_index += execute_count
        if current_index >= desired_path.shape[0]:
            break

        unused_tail = selected.q[execute_count:]
        buffer_q = extend_joint_window(
            unused_tail,
            horizon=horizon,
            mode=config.tail_extension,
            bootstrap_prior=bootstrap_prior_q,
            current_index=current_index,
            lower=lower,
            upper=upper,
        )
        cycle_index += 1

    q = np.stack(executed_rows, axis=0).astype(np.float64)
    if q.shape != (desired_path.shape[0], RESIDUAL_DIM):
        raise RuntimeError(f"{method} rollout produced q shape {q.shape}, expected {(desired_path.shape[0], RESIDUAL_DIM)}")
    ee = fk.fk(q)
    if ee is None:
        raise RuntimeError("FK unexpectedly unavailable for rollout trajectory")
    return RolloutResult(
        q=q,
        ee=np.asarray(ee, dtype=np.float64),
        cycle_rows=cycle_rows,
        boundary_discontinuities=boundaries,
        selected_diffusion_flags=selected_diffusion,
    )


def full_trajectory_metrics(
    *,
    method: str,
    path_index: int,
    path_name: str,
    q: np.ndarray,
    ee: np.ndarray,
    desired: np.ndarray,
    expert_q: np.ndarray,
    weights: Weights,
    lower: np.ndarray,
    upper: np.ndarray,
    boundary_discontinuities: Sequence[float],
    selected_diffusion_flags: Sequence[bool],
    num_cycles: int,
    config: RolloutConfig,
    prior_source: str,
) -> Dict[str, Any]:
    mean_cart, rms_cart, max_cart = cartesian_error_metrics(ee, desired)
    mean_vel, mean_acc, mean_jerk, max_step = trajectory_derivative_magnitudes(q)
    vel_cost, acc_cost, jerk_cost = smoothness_costs(q)
    limit_count, limit_magnitude = joint_limit_metrics(q, lower, upper)
    q_rmse, max_q_error = q_error_metrics(q, expert_q)
    drawing_cost = compute_drawing_cost(
        q=q,
        ee=ee,
        desired=desired,
        weights=weights,
        limit_magnitude=limit_magnitude,
    )
    all_boundaries = np.asarray(boundary_discontinuities, dtype=np.float64)
    boundaries = all_boundaries[1:] if all_boundaries.size > 1 else np.asarray([], dtype=np.float64)
    selected = np.asarray(selected_diffusion_flags, dtype=np.float64)
    return {
        "path_index": path_index,
        "path_name": path_name,
        "method": method,
        "prediction_horizon": config.prediction_horizon,
        "execution_horizon": config.execution_horizon,
        "t_init": config.t_init,
        "num_candidates": config.num_candidates,
        "tail_extension": config.tail_extension,
        "prior_source": prior_source,
        "mean_cartesian_error": mean_cart,
        "rms_cartesian_error": rms_cart,
        "max_cartesian_error": max_cart,
        "drawing_total_cost": drawing_cost,
        "mean_joint_velocity_magnitude": mean_vel,
        "mean_joint_acceleration_magnitude": mean_acc,
        "mean_joint_jerk_magnitude": mean_jerk,
        "joint_velocity_cost": float(vel_cost),
        "joint_acceleration_cost": float(acc_cost),
        "joint_jerk_cost": float(jerk_cost),
        "max_joint_step": max_step,
        "joint_limit_violation_count": limit_count,
        "joint_limit_violation_magnitude": limit_magnitude,
        "joint_rmse_vs_expert": q_rmse,
        "max_joint_abs_error_vs_expert": max_q_error,
        "num_planning_cycles": num_cycles,
        "fraction_cycles_selected_diffusion": float(np.mean(selected)) if selected.size else 0.0,
        "mean_boundary_discontinuity": float(np.mean(boundaries)) if boundaries.size else 0.0,
        "max_boundary_discontinuity": float(np.max(boundaries)) if boundaries.size else 0.0,
    }


def format_value(value: Any) -> Any:
    if isinstance(value, (float, np.floating)):
        if math.isfinite(float(value)):
            return f"{float(value):.12e}"
        return str(float(value))
    if isinstance(value, (int, np.integer)):
        return int(value)
    return value


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fields: Optional[Sequence[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        ordered: List[str] = []
        for row in rows:
            for key in row:
                if key not in ordered:
                    ordered.append(key)
        fields = ordered
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: format_value(row.get(field, "")) for field in fields})


def write_matrix_csv(path: Path, values: np.ndarray, columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["t", *columns])
        for idx, row in enumerate(values):
            writer.writerow([idx, *[f"{float(value):.12e}" for value in row]])


def make_paired_row(buffer_row: Mapping[str, Any], diffusion_row: Mapping[str, Any]) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "path_index": buffer_row["path_index"],
        "path_name": buffer_row["path_name"],
    }
    for metric in MAIN_DELTA_METRICS:
        base = float(buffer_row[metric])
        refined = float(diffusion_row[metric])
        delta = refined - base
        pct = 100.0 * delta / abs(base) if abs(base) > EPS else float("nan")
        row[f"{metric}_buffer_only"] = base
        row[f"{metric}_buffer_plus_diffusion"] = refined
        row[f"{metric}_delta"] = delta
        row[f"{metric}_percent_change"] = pct
        row[f"{metric}_improved"] = int(delta < -EPS)
    return row


def aggregate_summary_rows(
    per_path_rows: Sequence[Mapping[str, Any]],
    paired_rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    metrics = [
        "mean_cartesian_error",
        "rms_cartesian_error",
        "max_cartesian_error",
        "drawing_total_cost",
        "mean_joint_velocity_magnitude",
        "mean_joint_acceleration_magnitude",
        "mean_joint_jerk_magnitude",
        "max_joint_step",
        "mean_boundary_discontinuity",
        "joint_rmse_vs_expert",
    ]
    rows: List[Dict[str, Any]] = []
    for method in METHODS:
        group = [row for row in per_path_rows if row["method"] == method]
        if not group:
            continue
        for metric in metrics:
            values = np.asarray([float(row[metric]) for row in group], dtype=np.float64)
            rows.append(
                {
                    "section": "method",
                    "method": method,
                    "metric": metric,
                    "count": values.size,
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "paths_improved": "",
                    "paths_worsened": "",
                    "paths_tied": "",
                    "percentage_improved": "",
                }
            )

    for metric in PRIMARY_AGGREGATE_METRICS:
        deltas = np.asarray([float(row[f"{metric}_delta"]) for row in paired_rows], dtype=np.float64)
        improved = int(np.sum(deltas < -EPS))
        worsened = int(np.sum(deltas > EPS))
        tied = int(deltas.size - improved - worsened)
        rows.append(
            {
                "section": "paired",
                "method": "buffer_plus_diffusion - buffer_only",
                "metric": metric,
                "count": int(deltas.size),
                "mean": float(np.mean(deltas)) if deltas.size else float("nan"),
                "median": float(np.median(deltas)) if deltas.size else float("nan"),
                "std": float(np.std(deltas)) if deltas.size else float("nan"),
                "min": float(np.min(deltas)) if deltas.size else float("nan"),
                "max": float(np.max(deltas)) if deltas.size else float("nan"),
                "paths_improved": improved,
                "paths_worsened": worsened,
                "paths_tied": tied,
                "percentage_improved": 100.0 * improved / max(int(deltas.size), 1),
            }
        )
    return rows


def save_path_outputs(
    *,
    path_dir: Path,
    desired: np.ndarray,
    expert_q: np.ndarray,
    bootstrap_prior_q: np.ndarray,
    buffer_only: RolloutResult,
    diffusion: RolloutResult,
    bootstrap_ee: np.ndarray,
    expert_ee: np.ndarray,
) -> None:
    write_matrix_csv(path_dir / "desired_path.csv", desired, XYZ_COLUMNS)
    write_matrix_csv(path_dir / "expert_q.csv", expert_q, JOINT_COLUMNS)
    write_matrix_csv(path_dir / "bootstrap_prior_q.csv", bootstrap_prior_q, JOINT_COLUMNS)
    write_matrix_csv(path_dir / "buffer_only_q.csv", buffer_only.q, JOINT_COLUMNS)
    write_matrix_csv(path_dir / "buffer_plus_diffusion_q.csv", diffusion.q, JOINT_COLUMNS)
    write_matrix_csv(path_dir / "bootstrap_prior_ee.csv", bootstrap_ee, XYZ_COLUMNS)
    write_matrix_csv(path_dir / "buffer_only_ee.csv", buffer_only.ee, XYZ_COLUMNS)
    write_matrix_csv(path_dir / "buffer_plus_diffusion_ee.csv", diffusion.ee, XYZ_COLUMNS)
    write_matrix_csv(path_dir / "expert_ee.csv", expert_ee, XYZ_COLUMNS)


def save_path_plots(
    *,
    path_dir: Path,
    path_name: str,
    desired: np.ndarray,
    expert_q: np.ndarray,
    bootstrap_q: np.ndarray,
    buffer_only: RolloutResult,
    diffusion: RolloutResult,
    bootstrap_ee: np.ndarray,
    expert_ee: np.ndarray,
) -> None:
    path_dir.mkdir(parents=True, exist_ok=True)
    t = np.arange(desired.shape[0])

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(desired[:, 0], desired[:, 1], desired[:, 2], label="desired", linewidth=2)
    ax.plot(expert_ee[:, 0], expert_ee[:, 1], expert_ee[:, 2], label="expert", alpha=0.8)
    ax.plot(bootstrap_ee[:, 0], bootstrap_ee[:, 1], bootstrap_ee[:, 2], label="bootstrap_prior", alpha=0.8)
    ax.plot(buffer_only.ee[:, 0], buffer_only.ee[:, 1], buffer_only.ee[:, 2], label="buffer_only", alpha=0.9)
    ax.plot(diffusion.ee[:, 0], diffusion.ee[:, 1], diffusion.ee[:, 2], label="buffer_plus_diffusion", alpha=0.9)
    ax.set_title(path_name)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path_dir / "cartesian_3d.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for label, ee in (
        ("expert", expert_ee),
        ("bootstrap_prior", bootstrap_ee),
        ("buffer_only", buffer_only.ee),
        ("buffer_plus_diffusion", diffusion.ee),
    ):
        err = np.linalg.norm(ee - desired, axis=1)
        ax.plot(t, err, label=label)
    ax.set_xlabel("t")
    ax.set_ylabel("Cartesian error")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path_dir / "cartesian_error_over_time.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(3, 2, figsize=(10, 8), sharex=True)
    for joint_idx, ax in enumerate(axes.ravel()):
        ax.plot(t, expert_q[:, joint_idx], label="expert", linewidth=1.5)
        ax.plot(t, bootstrap_q[:, joint_idx], label="bootstrap_prior", alpha=0.7)
        ax.plot(t, buffer_only.q[:, joint_idx], label="buffer_only", alpha=0.8)
        ax.plot(t, diffusion.q[:, joint_idx], label="buffer_plus_diffusion", alpha=0.8)
        ax.set_ylabel(f"q{joint_idx + 1}")
    axes[-1, 0].set_xlabel("t")
    axes[-1, 1].set_xlabel("t")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path_dir / "joint_trajectories.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(3, 2, figsize=(10, 8), sharex=True)
    tv = np.arange(max(desired.shape[0] - 1, 0))
    velocity_series = {
        "expert": np.diff(expert_q, axis=0),
        "bootstrap_prior": np.diff(bootstrap_q, axis=0),
        "buffer_only": np.diff(buffer_only.q, axis=0),
        "buffer_plus_diffusion": np.diff(diffusion.q, axis=0),
    }
    for joint_idx, ax in enumerate(axes.ravel()):
        for label, values in velocity_series.items():
            ax.plot(tv, values[:, joint_idx], label=label, alpha=0.8)
        ax.set_ylabel(f"dq{joint_idx + 1}")
    axes[-1, 0].set_xlabel("t")
    axes[-1, 1].set_xlabel("t")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path_dir / "joint_velocity_over_time.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(np.arange(len(buffer_only.boundary_discontinuities)), buffer_only.boundary_discontinuities, label="buffer_only")
    ax.plot(
        np.arange(len(diffusion.boundary_discontinuities)),
        diffusion.boundary_discontinuities,
        label="buffer_plus_diffusion",
    )
    ax.set_xlabel("cycle")
    ax.set_ylabel("boundary discontinuity")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path_dir / "planning_boundary_discontinuity.png", dpi=160)
    plt.close(fig)


def save_aggregate_plots(output_dir: Path, paired_rows: Sequence[Mapping[str, Any]], per_path_rows: Sequence[Mapping[str, Any]]) -> None:
    plot_dir = output_dir / "aggregate_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    def paired_values(metric: str) -> Tuple[np.ndarray, np.ndarray]:
        base = np.asarray([float(row[f"{metric}_buffer_only"]) for row in paired_rows], dtype=np.float64)
        refined = np.asarray([float(row[f"{metric}_buffer_plus_diffusion"]) for row in paired_rows], dtype=np.float64)
        return base, refined

    for metric, filename, label in (
        ("mean_cartesian_error", "paired_mean_cartesian_error_scatter.png", "Mean Cartesian error"),
        ("drawing_total_cost", "paired_drawing_cost_scatter.png", "Drawing total cost"),
    ):
        base, refined = paired_values(metric)
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        ax.scatter(base, refined, alpha=0.8)
        lo = float(np.nanmin(np.concatenate([base, refined]))) if base.size else 0.0
        hi = float(np.nanmax(np.concatenate([base, refined]))) if base.size else 1.0
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1)
        ax.set_xlabel("buffer_only")
        ax.set_ylabel("buffer_plus_diffusion")
        ax.set_title(label)
        fig.tight_layout()
        fig.savefig(plot_dir / filename, dpi=160)
        plt.close(fig)

    pct = np.asarray([float(row["mean_cartesian_error_percent_change"]) for row in paired_rows], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(pct[np.isfinite(pct)], bins=20)
    ax.set_xlabel("Cartesian error percent change")
    ax.set_ylabel("paths")
    fig.tight_layout()
    fig.savefig(plot_dir / "histogram_percentage_cartesian_error_change.png", dpi=160)
    plt.close(fig)

    drawing_delta = np.asarray([float(row["drawing_total_cost_delta"]) for row in paired_rows], dtype=np.float64)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(drawing_delta[np.isfinite(drawing_delta)], bins=20)
    ax.set_xlabel("Drawing cost change")
    ax.set_ylabel("paths")
    fig.tight_layout()
    fig.savefig(plot_dir / "histogram_drawing_cost_change.png", dpi=160)
    plt.close(fig)

    main_delta = np.asarray([float(row["mean_cartesian_error_delta"]) for row in paired_rows], dtype=np.float64)
    improved = int(np.sum(main_delta < -EPS))
    worsened = int(np.sum(main_delta > EPS))
    tied = int(main_delta.size - improved - worsened)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.bar(["improved", "worsened", "tied"], [improved, worsened, tied])
    ax.set_ylabel("paths")
    fig.tight_layout()
    fig.savefig(plot_dir / "paths_improved_vs_worsened.png", dpi=160)
    plt.close(fig)

    diffusion_rows = [row for row in per_path_rows if row["method"] == "buffer_plus_diffusion"]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(
        [str(row["path_index"]) for row in diffusion_rows],
        [float(row["fraction_cycles_selected_diffusion"]) for row in diffusion_rows],
    )
    ax.set_xlabel("path index")
    ax.set_ylabel("diffusion selection fraction")
    fig.tight_layout()
    fig.savefig(plot_dir / "diffusion_selection_fraction_by_path.png", dpi=160)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    if args.num_candidates < 1:
        raise ValueError("--num_candidates must be >= 1")

    set_seed(args.seed)
    device = resolve_device(args.device)
    test_data = load_npz(args.test_npz, "test npz")
    window_data = load_npz(args.window_npz, "window npz")
    desired_all, expert_all, q_start_all, names = load_full_dataset(test_data)
    if desired_all.shape[-1] != 3 or expert_all.shape[-1] != RESIDUAL_DIM:
        raise RuntimeError("Resolved trajectories must have desired shape [...,3] and q shape [...,6]")
    num_paths, trajectory_length, _ = desired_all.shape
    path_indices = parse_path_indices(args.path_indices, num_paths, args.max_paths)
    if not path_indices:
        raise ValueError("No paths selected")

    bootstrap_prior_all, prior_source = resolve_bootstrap_prior(
        test_data=test_data,
        window_data=window_data,
        names=names,
        trajectory_length=trajectory_length,
        prior_dir=args.prior_dir,
    )
    if bootstrap_prior_all.shape != expert_all.shape:
        raise ValueError(f"bootstrap prior shape {bootstrap_prior_all.shape} must match expert shape {expert_all.shape}")
    require_finite("bootstrap_prior", bootstrap_prior_all)
    if q_start_all is None:
        q_start_all = bootstrap_prior_all[:, 0, :].copy()
        print("[dataset] q_start missing; using bootstrap_prior[:,0] for condition q_start")

    fk = FKComputer(args.urdf, args.ee_link)
    require_fk_ready(fk)
    lower, upper = finite_limits(fk)

    checkpoint: Optional[Dict[str, Any]] = None
    model: Optional[torch.nn.Module] = None
    call_variant: Optional[str] = None
    schedule: Optional[Mapping[str, torch.Tensor]] = None
    diffusion_config: Dict[str, Any] = {"num_steps": 0, "beta_start": float("nan"), "beta_end": float("nan")}
    if not args.no_diffusion:
        checkpoint = torch_load_checkpoint(args.diffusion_checkpoint, device)
        model, call_variant, model_config = instantiate_checkpoint_model(checkpoint, device)
        model.eval()
        diffusion_config = diffusion_config_from_checkpoint(checkpoint, None)
        num_steps = int(diffusion_config["num_steps"])
        if args.t_init >= num_steps:
            raise ValueError(f"t_init={args.t_init} must be < num_diffusion_steps={num_steps}")
        if int(checkpoint.get("target_dim", RESIDUAL_DIM)) != RESIDUAL_DIM:
            raise ValueError(f"checkpoint target_dim must be {RESIDUAL_DIM}")
        if int(checkpoint.get("condition_dim", V5B_CONDITION_DIM)) != V5B_CONDITION_DIM:
            raise ValueError(f"checkpoint condition_dim must be {V5B_CONDITION_DIM}")
        checkpoint_horizon = int(checkpoint.get("horizon", args.prediction_horizon))
        if checkpoint_horizon != args.prediction_horizon:
            print(
                "[checkpoint] requested prediction_horizon="
                f"{args.prediction_horizon}, checkpoint metadata horizon={checkpoint_horizon}; "
                "using the requested rollout horizon with the convolutional denoiser."
            )
        schedule = make_schedule(
            num_steps,
            float(diffusion_config["beta_start"]),
            float(diffusion_config["beta_end"]),
            device,
        )
        print(
            f"[checkpoint] {args.diffusion_checkpoint} | epoch={checkpoint.get('epoch', '')} | "
            f"model={model_config.get('model_class', type(model).__name__)} | call_variant={call_variant}"
        )
    else:
        print("[diffusion] --no_diffusion set; buffer_plus_diffusion will contain only the safety buffer candidate")
        checkpoint = {}

    stats = load_normalization_stats(args.normalization_stats, checkpoint or {})
    residual_mean_ckpt, residual_std_ckpt = load_residual_stats(args.normalization_stats, checkpoint or {})
    if not np.allclose(stats.residual_mean, residual_mean_ckpt) or not np.allclose(stats.residual_std, residual_std_ckpt):
        raise ValueError("Residual stats from normalization_stats and checkpoint loader disagree")

    weights = Weights(
        cart=args.w_cart,
        max_cart=args.w_max,
        start=args.w_start,
        end=args.w_end,
        frechet=args.w_frechet,
        dtw=args.w_dtw,
        vel=args.w_vel,
        acc=args.w_acc,
        jerk=args.w_jerk,
        limit=args.w_limit,
        tangent=args.w_tangent,
        progress=args.w_progress,
        length_ratio=args.w_length_ratio,
        norm_shape=args.w_norm_shape,
    )
    config = RolloutConfig(
        prediction_horizon=args.prediction_horizon,
        execution_horizon=args.execution_horizon,
        t_init=args.t_init,
        num_candidates=args.num_candidates,
        tail_extension=args.tail_extension,
        no_diffusion=args.no_diffusion,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    print(
        "[startup] "
        f"device={device}, seed={args.seed}, selected_paths={len(path_indices)}, "
        f"H={args.prediction_horizon}, E={args.execution_horizon}, t_init={args.t_init}, "
        f"extension={args.tail_extension}, num_diffusion_candidates={args.num_candidates}, "
        f"condition_dim={V5B_CONDITION_DIM}, prior_source={prior_source}"
    )

    per_path_rows: List[Dict[str, Any]] = []
    paired_rows: List[Dict[str, Any]] = []
    cycle_rows: List[Dict[str, Any]] = []

    for ordinal, path_index in enumerate(path_indices):
        path_name = names[path_index]
        desired = desired_all[path_index]
        expert_q = expert_all[path_index]
        bootstrap_q = clip_to_limits(bootstrap_prior_all[path_index], lower, upper)
        q_start = q_start_all[path_index]
        expert_ee = fk.fk(expert_q)
        bootstrap_ee = fk.fk(bootstrap_q)
        if expert_ee is None or bootstrap_ee is None:
            raise RuntimeError("FK unexpectedly unavailable for reference trajectories")
        expert_ee = np.asarray(expert_ee, dtype=np.float64)
        bootstrap_ee = np.asarray(bootstrap_ee, dtype=np.float64)

        buffer_only = run_rollout(
            method="buffer_only",
            path_index=path_index,
            path_name=path_name,
            desired_path=desired,
            q_start=q_start,
            bootstrap_prior_q=bootstrap_q,
            fk=fk,
            lower=lower,
            upper=upper,
            stats=stats,
            model=None,
            call_variant=None,
            schedule=None,
            device=device,
            config=config,
            weights=weights,
            w_rms_cart=args.w_rms_cart,
            w_limit_count=args.w_limit_count,
            continuity_weight=args.continuity_weight,
            seed=args.seed,
            save_candidate_details=args.save_candidate_details,
        )
        diffusion = run_rollout(
            method="buffer_plus_diffusion",
            path_index=path_index,
            path_name=path_name,
            desired_path=desired,
            q_start=q_start,
            bootstrap_prior_q=bootstrap_q,
            fk=fk,
            lower=lower,
            upper=upper,
            stats=stats,
            model=model,
            call_variant=call_variant,
            schedule=schedule,
            device=device,
            config=config,
            weights=weights,
            w_rms_cart=args.w_rms_cart,
            w_limit_count=args.w_limit_count,
            continuity_weight=args.continuity_weight,
            seed=args.seed,
            save_candidate_details=args.save_candidate_details,
        )

        bootstrap_metrics = full_trajectory_metrics(
            method="bootstrap_prior",
            path_index=path_index,
            path_name=path_name,
            q=bootstrap_q,
            ee=bootstrap_ee,
            desired=desired,
            expert_q=expert_q,
            weights=weights,
            lower=lower,
            upper=upper,
            boundary_discontinuities=[],
            selected_diffusion_flags=[],
            num_cycles=0,
            config=config,
            prior_source=prior_source,
        )
        buffer_metrics = full_trajectory_metrics(
            method="buffer_only",
            path_index=path_index,
            path_name=path_name,
            q=buffer_only.q,
            ee=buffer_only.ee,
            desired=desired,
            expert_q=expert_q,
            weights=weights,
            lower=lower,
            upper=upper,
            boundary_discontinuities=buffer_only.boundary_discontinuities,
            selected_diffusion_flags=buffer_only.selected_diffusion_flags,
            num_cycles=len(buffer_only.boundary_discontinuities),
            config=config,
            prior_source=prior_source,
        )
        diffusion_metrics = full_trajectory_metrics(
            method="buffer_plus_diffusion",
            path_index=path_index,
            path_name=path_name,
            q=diffusion.q,
            ee=diffusion.ee,
            desired=desired,
            expert_q=expert_q,
            weights=weights,
            lower=lower,
            upper=upper,
            boundary_discontinuities=diffusion.boundary_discontinuities,
            selected_diffusion_flags=diffusion.selected_diffusion_flags,
            num_cycles=len(diffusion.boundary_discontinuities),
            config=config,
            prior_source=prior_source,
        )
        expert_metrics = full_trajectory_metrics(
            method="expert",
            path_index=path_index,
            path_name=path_name,
            q=expert_q,
            ee=expert_ee,
            desired=desired,
            expert_q=expert_q,
            weights=weights,
            lower=lower,
            upper=upper,
            boundary_discontinuities=[],
            selected_diffusion_flags=[],
            num_cycles=0,
            config=config,
            prior_source=prior_source,
        )

        per_path_rows.extend([bootstrap_metrics, buffer_metrics, diffusion_metrics, expert_metrics])
        paired_rows.append(make_paired_row(buffer_metrics, diffusion_metrics))
        cycle_rows.extend(buffer_only.cycle_rows)
        cycle_rows.extend(diffusion.cycle_rows)

        path_dir = args.output_dir / f"path_{path_index:04d}_{safe_path_name(path_name)}"
        save_path_outputs(
            path_dir=path_dir,
            desired=desired,
            expert_q=expert_q,
            bootstrap_prior_q=bootstrap_q,
            buffer_only=buffer_only,
            diffusion=diffusion,
            bootstrap_ee=bootstrap_ee,
            expert_ee=expert_ee,
        )
        save_path_plots(
            path_dir=path_dir,
            path_name=path_name,
            desired=desired,
            expert_q=expert_q,
            bootstrap_q=bootstrap_q,
            buffer_only=buffer_only,
            diffusion=diffusion,
            bootstrap_ee=bootstrap_ee,
            expert_ee=expert_ee,
        )

        delta = paired_rows[-1]["mean_cartesian_error_delta"]
        print(
            f"[{ordinal + 1}/{len(path_indices)}] {path_name}: "
            f"buffer_mean={buffer_metrics['mean_cartesian_error']:.6e}, "
            f"diffusion_mean={diffusion_metrics['mean_cartesian_error']:.6e}, "
            f"delta={delta:.6e}, "
            f"diffusion_selected={diffusion_metrics['fraction_cycles_selected_diffusion']:.3f}"
        )

    aggregate_rows = aggregate_summary_rows(per_path_rows, paired_rows)
    per_path_path = args.output_dir / "per_path_summary.csv"
    paired_path = args.output_dir / "paired_comparison.csv"
    aggregate_path = args.output_dir / "aggregate_summary.csv"
    cycle_path = args.output_dir / "planning_cycle_details.csv"
    write_csv(per_path_path, per_path_rows)
    write_csv(paired_path, paired_rows)
    write_csv(aggregate_path, aggregate_rows)
    write_csv(cycle_path, cycle_rows)
    save_aggregate_plots(args.output_dir, paired_rows, per_path_rows)

    def method_mean(method: str, metric: str) -> float:
        values = [float(row[metric]) for row in per_path_rows if row["method"] == method]
        return float(np.mean(values)) if values else float("nan")

    buffer_mean = method_mean("buffer_only", "mean_cartesian_error")
    diffusion_mean = method_mean("buffer_plus_diffusion", "mean_cartesian_error")
    relative = 100.0 * (diffusion_mean - buffer_mean) / abs(buffer_mean) if abs(buffer_mean) > EPS else float("nan")
    cart_deltas = np.asarray([float(row["mean_cartesian_error_delta"]) for row in paired_rows], dtype=np.float64)
    improved = int(np.sum(cart_deltas < -EPS))
    diffusion_selection = method_mean("buffer_plus_diffusion", "fraction_cycles_selected_diffusion")
    print("[complete]")
    print(f"  buffer-only aggregate mean Cartesian error: {buffer_mean:.8e}")
    print(f"  buffer-plus-diffusion aggregate mean Cartesian error: {diffusion_mean:.8e}")
    print(f"  relative change: {relative:.3f}%")
    print(f"  paths improved: {improved}/{len(paired_rows)} ({100.0 * improved / max(len(paired_rows), 1):.1f}%)")
    print(
        "  drawing cost: "
        f"buffer={method_mean('buffer_only', 'drawing_total_cost'):.8e}, "
        f"diffusion={method_mean('buffer_plus_diffusion', 'drawing_total_cost'):.8e}"
    )
    print(
        "  max joint step: "
        f"buffer={method_mean('buffer_only', 'max_joint_step'):.8e}, "
        f"diffusion={method_mean('buffer_plus_diffusion', 'max_joint_step'):.8e}"
    )
    print(
        "  boundary continuity: "
        f"buffer={method_mean('buffer_only', 'mean_boundary_discontinuity'):.8e}, "
        f"diffusion={method_mean('buffer_plus_diffusion', 'mean_boundary_discontinuity'):.8e}"
    )
    print(f"  fraction of cycles selecting diffusion: {diffusion_selection:.6f}")
    print(f"  per-path summary: {per_path_path}")
    print(f"  paired comparison: {paired_path}")
    print(f"  aggregate summary: {aggregate_path}")
    print(f"  planning-cycle details: {cycle_path}")
    print(f"  output directory: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
