"""Trajectory scoring and cost helpers."""

from __future__ import annotations

import numpy as np


def _as_float_array(name: str, value: np.ndarray, ndim: int, width: int | None = None) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions, got shape {arr.shape}")
    if width is not None and arr.shape[-1] != width:
        raise ValueError(f"{name} must have last dimension {width}, got shape {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or inf values")
    return arr


def compute_xyz_axis_loss(
    desired_ee: np.ndarray,
    pred_ee: np.ndarray,
    w_x: float = 1.0,
    w_y: float = 1.0,
    w_z: float = 1.0,
) -> dict[str, float]:
    desired = _as_float_array("desired_ee", desired_ee, 2, 3)
    pred = _as_float_array("pred_ee", pred_ee, 2, 3)
    if desired.shape != pred.shape:
        raise ValueError(f"desired_ee and pred_ee shapes must match, got {desired.shape} and {pred.shape}")

    squared = (pred - desired) ** 2
    loss_x = float(np.mean(squared[:, 0]))
    loss_y = float(np.mean(squared[:, 1]))
    loss_z = float(np.mean(squared[:, 2]))
    weighted_xyz_loss = float(w_x * loss_x + w_y * loss_y + w_z * loss_z)
    return {
        "loss_x": loss_x,
        "loss_y": loss_y,
        "loss_z": loss_z,
        "weighted_xyz_loss": weighted_xyz_loss,
    }


def compute_cartesian_error_metrics(desired_ee: np.ndarray, pred_ee: np.ndarray) -> dict[str, float]:
    desired = _as_float_array("desired_ee", desired_ee, 2, 3)
    pred = _as_float_array("pred_ee", pred_ee, 2, 3)
    if desired.shape != pred.shape:
        raise ValueError(f"desired_ee and pred_ee shapes must match, got {desired.shape} and {pred.shape}")

    diff = pred - desired
    euclidean = np.linalg.norm(diff, axis=1)
    return {
        "path_error": float(np.mean(np.sum(diff**2, axis=1))),
        "mean_error": float(np.mean(euclidean)),
        "max_error": float(np.max(euclidean)),
    }


def compute_joint_velocity_cost(q: np.ndarray) -> float:
    q_arr = _as_float_array("q", q, 2)
    if q_arr.shape[0] < 2:
        return 0.0
    velocity = np.diff(q_arr, axis=0)
    return float(np.mean(velocity**2))


def compute_joint_acceleration_cost(q: np.ndarray) -> float:
    q_arr = _as_float_array("q", q, 2)
    if q_arr.shape[0] < 3:
        return 0.0
    acceleration = np.diff(q_arr, n=2, axis=0)
    return float(np.mean(acceleration**2))


def compute_stanford_style_trajectory_cost(
    desired_ee: np.ndarray,
    pred_ee: np.ndarray,
    q: np.ndarray,
    w_path: float = 1.0,
    w_x: float = 1.0,
    w_y: float = 1.0,
    w_z: float = 1.0,
    w_vel: float = 0.0,
    w_accel: float = 0.01,
) -> dict[str, float]:
    cartesian_metrics = compute_cartesian_error_metrics(desired_ee, pred_ee)
    axis_losses = compute_xyz_axis_loss(desired_ee, pred_ee, w_x=w_x, w_y=w_y, w_z=w_z)
    joint_velocity_cost = compute_joint_velocity_cost(q)
    joint_acceleration_cost = compute_joint_acceleration_cost(q)
    total_cost = (
        w_path * cartesian_metrics["path_error"]
        + axis_losses["weighted_xyz_loss"]
        + w_vel * joint_velocity_cost
        + w_accel * joint_acceleration_cost
    )

    return {
        **cartesian_metrics,
        **axis_losses,
        "joint_velocity_cost": float(joint_velocity_cost),
        "joint_acceleration_cost": float(joint_acceleration_cost),
        "total_cost": float(total_cost),
    }
