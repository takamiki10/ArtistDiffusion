#!/usr/bin/env python3
"""Build v5b residual-window datasets with FK prior-error conditioning.

This extends the v5 residual-window dataset by appending prior end-effector FK
features to each condition timestep:

    prior_ee_xyz, prior_ee_error_xyz, ||prior_ee_error||

FK is computed with the project convention:

    robot.update_cfg(cfg)
    transform = robot.get_transform(frame_to=ee_link)

Do not use robot.link_fk(...) here.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


DEFAULT_TRAIN_NPZ = Path("data/cartesian_expert_dataset_v3/diffusion_v2/diffusion_train_v2.npz")
DEFAULT_TEST_NPZ = Path("data/cartesian_expert_dataset_v3/diffusion_v2/diffusion_test_v2.npz")
DEFAULT_TRAIN_PRIOR_DIR = Path("data/cartesian_expert_dataset_v3/mlp_v3_train_predictions")
DEFAULT_TEST_PRIOR_DIR = Path("data/cartesian_expert_dataset_v3/mlp_v3_test_predictions")
DEFAULT_OUTPUT_DIR = Path("data/cartesian_expert_dataset_v3/diffusion_v5b_residual_windows_fk_condition")
PREDICTED_Q_NAME = "predicted_q.csv"
JOINT_COLUMNS = ("q1", "q2", "q3", "q4", "q5", "q6")
BASE_CONDITION_DIM = 31
CONDITION_DIM = 38
RESIDUAL_DIM = 6
EPS = 1e-8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build v5b residual-window train/test NPZ datasets with FK prior-error condition features."
    )
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--train_npz", type=Path, default=DEFAULT_TRAIN_NPZ)
    parser.add_argument("--test_npz", type=Path, default=DEFAULT_TEST_NPZ)
    parser.add_argument("--train_prior_dir", type=Path, default=DEFAULT_TRAIN_PRIOR_DIR)
    parser.add_argument("--test_prior_dir", type=Path, default=DEFAULT_TEST_PRIOR_DIR)
    parser.add_argument(
        "--urdf_path",
        type=Path,
        default=None,
        help="Optional URDF path. Defaults to generate_ik_seed_path.DEFAULT_URDF_PATH.",
    )
    parser.add_argument(
        "--ee_link",
        type=str,
        default=None,
        help="Optional end-effector link. Defaults to generate_ik_seed_path.DEFAULT_EE_LINK.",
    )
    return parser.parse_args()


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Missing NPZ dataset: {path}")
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def require_keys(data: Dict[str, np.ndarray], keys: Sequence[str], label: str) -> None:
    missing = [key for key in keys if key not in data]
    if missing:
        raise KeyError(f"{label} missing required key(s): {', '.join(missing)}")


def path_names(data: Dict[str, np.ndarray]) -> List[str]:
    raw = np.asarray(data["path_names"])
    names: List[str] = []
    for item in raw:
        names.append(item.decode("utf-8", errors="replace") if isinstance(item, bytes) else str(item))
    return names


def safe_path_name(name: str) -> str:
    return Path(str(name)).name.replace("/", "_").replace("\\", "_")


def check_safe_name_collisions(names: Sequence[str], label: str) -> None:
    safe_names = [safe_path_name(name) for name in names]
    if len(set(safe_names)) == len(safe_names):
        return

    seen: Dict[str, str] = {}
    for original, safe in zip(names, safe_names):
        if safe in seen:
            raise ValueError(
                f"{label}: path_names collide after filesystem sanitization: "
                f"{seen[safe]!r} and {original!r} both map to {safe!r}"
            )
        seen[safe] = original


def read_predicted_q_csv(path: Path, expected_steps: int) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing MLP prior predicted_q.csv: {path}")

    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no CSV header")
        field_map = {field.strip().lower(): field for field in reader.fieldnames if field is not None}
        missing = [name for name in JOINT_COLUMNS if name not in field_map]
        if missing:
            raise ValueError(f"{path} missing joint column(s): {', '.join(missing)}")

        rows: List[List[float]] = []
        for row_idx, row in enumerate(reader):
            try:
                rows.append([float(row[field_map[name]]) for name in JOINT_COLUMNS])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path} has non-numeric joint value on data row {row_idx}") from exc

    arr = np.asarray(rows, dtype=np.float32)
    expected_shape = (expected_steps, RESIDUAL_DIM)
    if arr.shape != expected_shape:
        raise ValueError(f"{path} must contain shape {expected_shape}, got {arr.shape}")
    return arr


def validate_split_arrays(
    data: Dict[str, np.ndarray],
    label: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    require_keys(data, ("desired_paths", "expert_q", "q_start", "path_names"), label)

    desired_paths = np.asarray(data["desired_paths"], dtype=np.float32)
    expert_q = np.asarray(data["expert_q"], dtype=np.float32)
    q_start = np.asarray(data["q_start"], dtype=np.float32)
    names = path_names(data)

    if desired_paths.ndim != 3 or desired_paths.shape[-1] != 3:
        raise ValueError(f"{label}: desired_paths must have shape (N,T,3), got {desired_paths.shape}")
    if expert_q.ndim != 3 or expert_q.shape[-1] != RESIDUAL_DIM:
        raise ValueError(f"{label}: expert_q must have shape (N,T,6), got {expert_q.shape}")
    if q_start.shape != (desired_paths.shape[0], RESIDUAL_DIM):
        raise ValueError(
            f"{label}: q_start must have shape {(desired_paths.shape[0], RESIDUAL_DIM)}, got {q_start.shape}"
        )
    if expert_q.shape[:2] != desired_paths.shape[:2]:
        raise ValueError(
            f"{label}: expert_q and desired_paths must share (N,T), got "
            f"{expert_q.shape[:2]} vs {desired_paths.shape[:2]}"
        )
    if len(names) != desired_paths.shape[0]:
        raise ValueError(f"{label}: path_names length {len(names)} does not match N={desired_paths.shape[0]}")

    check_safe_name_collisions(names, label)
    progress = np.linspace(0.0, 1.0, desired_paths.shape[1], dtype=np.float32)
    return desired_paths, expert_q, q_start, progress, names


def desired_finite_difference(desired_path: np.ndarray) -> np.ndarray:
    diff = np.zeros_like(desired_path, dtype=np.float32)
    if desired_path.shape[0] <= 1:
        return diff
    diff[:-1] = desired_path[1:] - desired_path[:-1]
    diff[-1] = diff[-2]
    return diff


def make_window_starts(num_steps: int, horizon: int, stride: int) -> List[int]:
    if horizon <= 0:
        raise ValueError("--horizon must be positive")
    if stride <= 0:
        raise ValueError("--stride must be positive")
    if horizon > num_steps:
        raise ValueError(f"--horizon={horizon} exceeds trajectory length T={num_steps}")
    return list(range(0, num_steps - horizon + 1, stride))


def resolve_project_path(path: Path) -> Path:
    if path.exists():
        return path
    script_dir = Path(__file__).resolve().parent
    for candidate in (script_dir / path, script_dir.parent / path):
        if candidate.exists():
            return candidate
    return path


def load_fk_context(
    urdf_path: Optional[Path],
    ee_link: Optional[str],
) -> Tuple[Any, Sequence[str], str]:
    try:
        from generate_ik_seed_path import (
            DEFAULT_EE_LINK,
            DEFAULT_JOINT_NAMES,
            DEFAULT_URDF_PATH,
            load_robot,
        )
    except Exception as exc:
        raise ImportError(
            "Could not import FK helpers from generate_ik_seed_path.py. "
            "v5b requires the existing xMateCR7 kinematics helper."
        ) from exc

    resolved_urdf = resolve_project_path(Path(DEFAULT_URDF_PATH) if urdf_path is None else urdf_path)
    resolved_ee_link = DEFAULT_EE_LINK if ee_link is None else ee_link
    joint_names = tuple(DEFAULT_JOINT_NAMES)
    if len(joint_names) != RESIDUAL_DIM:
        raise ValueError(
            f"Expected {RESIDUAL_DIM} joint names for q columns, got {len(joint_names)}: {joint_names}"
        )

    try:
        robot = load_robot(resolved_urdf)
    except TypeError:
        robot = load_robot()
    return robot, joint_names, str(resolved_ee_link)


def transform_xyz(transform: Any) -> np.ndarray:
    if hasattr(transform, "translation"):
        xyz = np.asarray(transform.translation, dtype=np.float64)
        if xyz.shape == (3,):
            return xyz.astype(np.float32)
    if hasattr(transform, "pos"):
        xyz = np.asarray(transform.pos, dtype=np.float64)
        if xyz.shape == (3,):
            return xyz.astype(np.float32)
    if hasattr(transform, "matrix"):
        matrix = np.asarray(transform.matrix, dtype=np.float64)
        if matrix.shape == (4, 4):
            return matrix[:3, 3].astype(np.float32)

    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape == (4, 4):
        return matrix[:3, 3].astype(np.float32)
    if matrix.shape == (3,):
        return matrix.astype(np.float32)
    raise ValueError(f"Unsupported transform object for FK xyz extraction: shape={matrix.shape}")


def q_to_cfg(q: np.ndarray, joint_names: Sequence[str]) -> Dict[str, float]:
    if q.shape != (RESIDUAL_DIM,):
        raise ValueError(f"q must have shape ({RESIDUAL_DIM},), got {q.shape}")
    return {joint_name: float(value) for joint_name, value in zip(joint_names, q)}


def fk_positions(
    robot: Any,
    joint_names: Sequence[str],
    ee_link: str,
    q_traj: np.ndarray,
) -> np.ndarray:
    positions: List[np.ndarray] = []
    for q in q_traj:
        cfg = q_to_cfg(np.asarray(q, dtype=np.float32), joint_names)
        robot.update_cfg(cfg)
        transform = robot.get_transform(frame_to=ee_link)
        positions.append(transform_xyz(transform))
    return np.stack(positions, axis=0).astype(np.float32)


def build_condition_window(
    desired_window: np.ndarray,
    desired_diff_window: np.ndarray,
    progress_window: np.ndarray,
    q_start: np.ndarray,
    current_q: np.ndarray,
    prior_q_window: np.ndarray,
    prior_ee_window: np.ndarray,
    prior_ee_error: np.ndarray,
) -> np.ndarray:
    horizon = desired_window.shape[0]
    q_start_window = np.repeat(q_start.reshape(1, RESIDUAL_DIM), horizon, axis=0)
    current_q_window = np.repeat(current_q.reshape(1, RESIDUAL_DIM), horizon, axis=0)
    prior_delta_from_start = prior_q_window - q_start_window
    prior_ee_error_norm = np.linalg.norm(prior_ee_error, axis=1, keepdims=True)

    base_condition = np.concatenate(
        [
            desired_window,
            desired_diff_window,
            progress_window.reshape(horizon, 1),
            q_start_window,
            current_q_window,
            prior_q_window,
            prior_delta_from_start,
        ],
        axis=1,
    ).astype(np.float32)
    if base_condition.shape != (horizon, BASE_CONDITION_DIM):
        raise RuntimeError(
            f"Base condition window must have shape ({horizon},{BASE_CONDITION_DIM}), "
            f"got {base_condition.shape}"
        )

    condition = np.concatenate(
        [
            base_condition,
            prior_ee_window,
            prior_ee_error,
            prior_ee_error_norm.astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)
    if condition.shape != (horizon, CONDITION_DIM):
        raise RuntimeError(f"Condition window must have shape ({horizon},{CONDITION_DIM}), got {condition.shape}")
    return condition


def build_split_windows(
    *,
    label: str,
    npz_path: Path,
    prior_dir: Path,
    horizon: int,
    stride: int,
    robot: Any,
    joint_names: Sequence[str],
    ee_link: str,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    data = load_npz(npz_path)
    desired_paths, expert_q, q_start_all, progress, names = validate_split_arrays(data, label)
    num_paths, num_steps, _ = desired_paths.shape
    starts = make_window_starts(num_steps, horizon, stride)

    conditions: List[np.ndarray] = []
    residuals: List[np.ndarray] = []
    expert_windows: List[np.ndarray] = []
    prior_windows: List[np.ndarray] = []
    desired_windows: List[np.ndarray] = []
    prior_ee_windows: List[np.ndarray] = []
    prior_ee_errors: List[np.ndarray] = []
    window_names: List[str] = []
    window_starts: List[int] = []
    window_q_start: List[np.ndarray] = []

    for path_idx, name in enumerate(names):
        prior_csv = prior_dir / safe_path_name(name) / PREDICTED_Q_NAME
        prior_q = read_predicted_q_csv(prior_csv, expected_steps=num_steps)
        prior_ee = fk_positions(robot, joint_names, ee_link, prior_q)
        prior_ee_error_full = prior_ee - desired_paths[path_idx]
        desired_diff = desired_finite_difference(desired_paths[path_idx])

        for start in starts:
            end = start + horizon
            prior_q_window = prior_q[start:end]
            expert_q_window = expert_q[path_idx, start:end]
            desired_path_window = desired_paths[path_idx, start:end]
            prior_ee_window = prior_ee[start:end]
            prior_ee_error = prior_ee_error_full[start:end]
            condition = build_condition_window(
                desired_window=desired_path_window,
                desired_diff_window=desired_diff[start:end],
                progress_window=progress[start:end],
                q_start=q_start_all[path_idx],
                current_q=prior_q[start],
                prior_q_window=prior_q_window,
                prior_ee_window=prior_ee_window,
                prior_ee_error=prior_ee_error,
            )

            conditions.append(condition)
            residuals.append((expert_q_window - prior_q_window).astype(np.float32))
            expert_windows.append(expert_q_window.astype(np.float32))
            prior_windows.append(prior_q_window.astype(np.float32))
            desired_windows.append(desired_path_window.astype(np.float32))
            prior_ee_windows.append(prior_ee_window.astype(np.float32))
            prior_ee_errors.append(prior_ee_error.astype(np.float32))
            window_names.append(name)
            window_starts.append(start)
            window_q_start.append(q_start_all[path_idx].astype(np.float32))

    split = {
        "condition": np.stack(conditions, axis=0).astype(np.float32),
        "residual_q": np.stack(residuals, axis=0).astype(np.float32),
        "expert_q_window": np.stack(expert_windows, axis=0).astype(np.float32),
        "prior_q_window": np.stack(prior_windows, axis=0).astype(np.float32),
        "desired_path_window": np.stack(desired_windows, axis=0).astype(np.float32),
        "prior_ee_window": np.stack(prior_ee_windows, axis=0).astype(np.float32),
        "prior_ee_error": np.stack(prior_ee_errors, axis=0).astype(np.float32),
        "path_names": np.asarray(window_names),
        "window_start_indices": np.asarray(window_starts, dtype=np.int64),
        "q_start": np.stack(window_q_start, axis=0).astype(np.float32),
    }
    validate_window_dataset(split, label, horizon)

    summary = {
        "split": label,
        "source_npz": str(npz_path),
        "prior_dir": str(prior_dir),
        "num_paths": num_paths,
        "trajectory_length": num_steps,
        "horizon": horizon,
        "stride": stride,
        "windows_per_path": len(starts),
        "num_windows": split["condition"].shape[0],
    }
    return split, summary


def validate_window_dataset(split: Dict[str, np.ndarray], label: str, horizon: int) -> None:
    condition = split["condition"]
    residual_q = split["residual_q"]
    expert_q_window = split["expert_q_window"]
    prior_q_window = split["prior_q_window"]
    desired_path_window = split["desired_path_window"]
    prior_ee_window = split["prior_ee_window"]
    prior_ee_error = split["prior_ee_error"]
    path_names_arr = split["path_names"]
    window_start_indices = split["window_start_indices"]
    q_start = split["q_start"]

    if condition.ndim != 3 or condition.shape[1:] != (horizon, CONDITION_DIM):
        raise RuntimeError(f"{label}: condition must have shape (W,{horizon},{CONDITION_DIM}), got {condition.shape}")
    if residual_q.shape != (condition.shape[0], horizon, RESIDUAL_DIM):
        raise RuntimeError(
            f"{label}: residual_q must have shape {(condition.shape[0], horizon, RESIDUAL_DIM)}, got {residual_q.shape}"
        )
    if expert_q_window.shape != residual_q.shape:
        raise RuntimeError(f"{label}: expert_q_window shape {expert_q_window.shape} != residual_q {residual_q.shape}")
    if prior_q_window.shape != residual_q.shape:
        raise RuntimeError(f"{label}: prior_q_window shape {prior_q_window.shape} != residual_q {residual_q.shape}")
    if desired_path_window.shape != (condition.shape[0], horizon, 3):
        raise RuntimeError(
            f"{label}: desired_path_window must have shape {(condition.shape[0], horizon, 3)}, "
            f"got {desired_path_window.shape}"
        )
    if prior_ee_window.shape != desired_path_window.shape:
        raise RuntimeError(f"{label}: prior_ee_window shape {prior_ee_window.shape} != desired_path {desired_path_window.shape}")
    if prior_ee_error.shape != desired_path_window.shape:
        raise RuntimeError(f"{label}: prior_ee_error shape {prior_ee_error.shape} != desired_path {desired_path_window.shape}")
    if path_names_arr.shape != (condition.shape[0],):
        raise RuntimeError(f"{label}: path_names must have shape {(condition.shape[0],)}, got {path_names_arr.shape}")
    if window_start_indices.shape != (condition.shape[0],):
        raise RuntimeError(
            f"{label}: window_start_indices must have shape {(condition.shape[0],)}, got {window_start_indices.shape}"
        )
    if q_start.shape != (condition.shape[0], RESIDUAL_DIM):
        raise RuntimeError(f"{label}: q_start must have shape {(condition.shape[0], RESIDUAL_DIM)}, got {q_start.shape}")

    residual_check = expert_q_window - prior_q_window
    if not np.allclose(residual_q, residual_check, rtol=1e-6, atol=1e-7):
        max_error = float(np.max(np.abs(residual_q - residual_check)))
        raise RuntimeError(f"{label}: residual_q must equal expert_q_window - prior_q_window; max error={max_error:.12e}")

    prior_ee_check = prior_ee_window - desired_path_window
    if not np.allclose(prior_ee_error, prior_ee_check, rtol=1e-6, atol=1e-7):
        max_error = float(np.max(np.abs(prior_ee_error - prior_ee_check)))
        raise RuntimeError(f"{label}: prior_ee_error must equal prior_ee_window - desired_path_window; max error={max_error:.12e}")


def train_stats(values: np.ndarray, feature_dim: int) -> Tuple[np.ndarray, np.ndarray]:
    flat = values.reshape(-1, feature_dim).astype(np.float64)
    mean = flat.mean(axis=0).astype(np.float32)
    std = flat.std(axis=0).astype(np.float32)
    std = np.maximum(std, EPS).astype(np.float32)
    return mean, std


def apply_normalization(
    split: Dict[str, np.ndarray],
    condition_mean: np.ndarray,
    condition_std: np.ndarray,
    residual_mean: np.ndarray,
    residual_std: np.ndarray,
) -> Dict[str, np.ndarray]:
    out = dict(split)
    out["condition_norm"] = (
        (out["condition"] - condition_mean.reshape(1, 1, CONDITION_DIM))
        / condition_std.reshape(1, 1, CONDITION_DIM)
    ).astype(np.float32)
    out["residual_q_norm"] = (
        (out["residual_q"] - residual_mean.reshape(1, 1, RESIDUAL_DIM))
        / residual_std.reshape(1, 1, RESIDUAL_DIM)
    ).astype(np.float32)
    out["condition_mean"] = condition_mean.astype(np.float32)
    out["condition_std"] = condition_std.astype(np.float32)
    out["residual_mean"] = residual_mean.astype(np.float32)
    out["residual_std"] = residual_std.astype(np.float32)
    return out


def save_split(path: Path, split: Dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        condition=split["condition"],
        condition_norm=split["condition_norm"],
        residual_q=split["residual_q"],
        residual_q_norm=split["residual_q_norm"],
        expert_q_window=split["expert_q_window"],
        prior_q_window=split["prior_q_window"],
        desired_path_window=split["desired_path_window"],
        prior_ee_window=split["prior_ee_window"],
        prior_ee_error=split["prior_ee_error"],
        path_names=split["path_names"],
        window_start_indices=split["window_start_indices"],
        q_start=split["q_start"],
        condition_mean=split["condition_mean"],
        condition_std=split["condition_std"],
        residual_mean=split["residual_mean"],
        residual_std=split["residual_std"],
    )


def save_normalization_stats(
    path: Path,
    condition_mean: np.ndarray,
    condition_std: np.ndarray,
    residual_mean: np.ndarray,
    residual_std: np.ndarray,
    horizon: int,
    stride: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        condition_mean=condition_mean.astype(np.float32),
        condition_std=condition_std.astype(np.float32),
        residual_mean=residual_mean.astype(np.float32),
        residual_std=residual_std.astype(np.float32),
        horizon=np.asarray(horizon, dtype=np.int64),
        stride=np.asarray(stride, dtype=np.int64),
        condition_dim=np.asarray(CONDITION_DIM, dtype=np.int64),
        residual_dim=np.asarray(RESIDUAL_DIM, dtype=np.int64),
    )


def residual_metrics(residual_q: np.ndarray) -> Dict[str, float]:
    return {
        "residual_rmse": float(np.sqrt(np.mean(np.square(residual_q)))),
        "residual_mean_abs": float(np.mean(np.abs(residual_q))),
        "residual_max_abs": float(np.max(np.abs(residual_q))),
    }


def prior_fk_error_metrics(prior_ee_error: np.ndarray) -> Dict[str, float]:
    norm = np.linalg.norm(prior_ee_error, axis=-1)
    return {
        "prior_fk_error_mean_norm": float(np.mean(norm)),
        "prior_fk_error_rms_norm": float(np.sqrt(np.mean(np.square(norm)))),
        "prior_fk_error_max_norm": float(np.max(norm)),
        "prior_fk_error_x_rmse": float(np.sqrt(np.mean(np.square(prior_ee_error[..., 0])))),
        "prior_fk_error_y_rmse": float(np.sqrt(np.mean(np.square(prior_ee_error[..., 1])))),
        "prior_fk_error_z_rmse": float(np.sqrt(np.mean(np.square(prior_ee_error[..., 2])))),
    }


def shape_string(values: np.ndarray) -> str:
    return "x".join(str(dim) for dim in values.shape)


def summary_row(summary: Dict[str, Any], output_npz: Path, split: Dict[str, np.ndarray]) -> Dict[str, Any]:
    row = dict(summary)
    row["output_npz"] = str(output_npz)
    row["condition_shape"] = shape_string(split["condition"])
    row["condition_norm_shape"] = shape_string(split["condition_norm"])
    row["condition_dim"] = CONDITION_DIM
    row["residual_shape"] = shape_string(split["residual_q"])
    row["residual_dim"] = RESIDUAL_DIM
    row.update({key: f"{value:.12e}" for key, value in residual_metrics(split["residual_q"]).items()})
    row.update({key: f"{value:.12e}" for key, value in prior_fk_error_metrics(split["prior_ee_error"]).items()})

    for idx, value in enumerate(split["residual_mean"]):
        row[f"residual_mean_q{idx + 1}"] = f"{float(value):.12e}"
    for idx, value in enumerate(split["residual_std"]):
        row[f"residual_std_q{idx + 1}"] = f"{float(value):.12e}"
    return row


def write_summary_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    fields = [
        "split",
        "source_npz",
        "prior_dir",
        "output_npz",
        "num_paths",
        "trajectory_length",
        "horizon",
        "stride",
        "windows_per_path",
        "num_windows",
        "condition_shape",
        "condition_norm_shape",
        "condition_dim",
        "residual_shape",
        "residual_dim",
        "residual_rmse",
        "residual_mean_abs",
        "residual_max_abs",
        "prior_fk_error_mean_norm",
        "prior_fk_error_rms_norm",
        "prior_fk_error_max_norm",
        "prior_fk_error_x_rmse",
        "prior_fk_error_y_rmse",
        "prior_fk_error_z_rmse",
    ]
    fields += [f"residual_mean_q{idx + 1}" for idx in range(RESIDUAL_DIM)]
    fields += [f"residual_std_q{idx + 1}" for idx in range(RESIDUAL_DIM)]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def print_split_summary(label: str, split: Dict[str, np.ndarray], summary: Dict[str, Any]) -> None:
    residual = residual_metrics(split["residual_q"])
    fk = prior_fk_error_metrics(split["prior_ee_error"])
    print(
        f"[{label}] paths={summary['num_paths']}, windows={summary['num_windows']}, "
        f"T={summary['trajectory_length']}, H={summary['horizon']}, stride={summary['stride']}, "
        f"condition={split['condition'].shape}, residual_rmse={residual['residual_rmse']:.6e}, "
        f"prior_fk_mean={fk['prior_fk_error_mean_norm']:.6e}, prior_fk_max={fk['prior_fk_error_max_norm']:.6e}"
    )


def main() -> int:
    args = parse_args()
    robot, joint_names, ee_link = load_fk_context(args.urdf_path, args.ee_link)

    train_split_raw, train_summary = build_split_windows(
        label="train",
        npz_path=args.train_npz,
        prior_dir=args.train_prior_dir,
        horizon=args.horizon,
        stride=args.stride,
        robot=robot,
        joint_names=joint_names,
        ee_link=ee_link,
    )
    test_split_raw, test_summary = build_split_windows(
        label="test",
        npz_path=args.test_npz,
        prior_dir=args.test_prior_dir,
        horizon=args.horizon,
        stride=args.stride,
        robot=robot,
        joint_names=joint_names,
        ee_link=ee_link,
    )

    condition_mean, condition_std = train_stats(train_split_raw["condition"], CONDITION_DIM)
    residual_mean, residual_std = train_stats(train_split_raw["residual_q"], RESIDUAL_DIM)

    train_split = apply_normalization(
        train_split_raw,
        condition_mean,
        condition_std,
        residual_mean,
        residual_std,
    )
    test_split = apply_normalization(
        test_split_raw,
        condition_mean,
        condition_std,
        residual_mean,
        residual_std,
    )

    train_path = args.output_dir / "train_windows.npz"
    test_path = args.output_dir / "test_windows.npz"
    stats_path = args.output_dir / "normalization_stats.npz"
    summary_path = args.output_dir / "dataset_summary.csv"

    save_split(train_path, train_split)
    save_split(test_path, test_split)
    save_normalization_stats(
        stats_path,
        condition_mean,
        condition_std,
        residual_mean,
        residual_std,
        args.horizon,
        args.stride,
    )
    write_summary_csv(
        summary_path,
        [
            summary_row(train_summary, train_path, train_split),
            summary_row(test_summary, test_path, test_split),
        ],
    )

    print(f"FK ee_link: {ee_link}")
    print(f"Condition dim: {CONDITION_DIM}")
    print_split_summary("train", train_split, train_summary)
    print_split_summary("test", test_split, test_summary)
    print(f"Saved train windows: {train_path}")
    print(f"Saved test windows: {test_path}")
    print(f"Saved normalization stats: {stats_path}")
    print(f"Saved summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
