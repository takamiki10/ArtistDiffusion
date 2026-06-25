#!/usr/bin/env python3
"""
Optimize a joint trajectory for a desired Cartesian drawing path using CEM.

This is the first expert-demonstration generator for the artist trajectory project.
It avoids differentiating through yourdfpy by using a derivative-free Cross-Entropy
Method (CEM) optimizer over a low-dimensional set of joint waypoints.

Input:
    desired_path.csv with columns: t,x,y,z

Output:
    optimized_q.csv  with columns: t,q1,q2,q3,q4,q5,q6
    optimized_ee.csv with columns: t,x,y,z

Example:
    python optimize_cartesian_path.py \
      --path_csv data/cartesian_test_paths/line_001/desired_path.csv \
      --output_q_csv data/cartesian_test_paths/line_001/optimized_q.csv \
      --output_ee_csv data/cartesian_test_paths/line_001/optimized_ee.csv \
      --iterations 100 \
      --num_candidates 128 \
      --num_elites 16

Recommended if you already have a poor time-conditioned MLP prediction:
    python optimize_cartesian_path.py \
      --path_csv data/cartesian_test_paths/arc_001/desired_path.csv \
      --init_q_csv data/cartesian_test_paths/arc_001/time_conditioned_pred_q.csv \
      --output_q_csv data/cartesian_test_paths/arc_001/optimized_q.csv \
      --output_ee_csv data/cartesian_test_paths/arc_001/optimized_ee.csv
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from yourdfpy import URDF

try:
    from scipy.interpolate import CubicSpline, PchipInterpolator
except ImportError:  # pragma: no cover - optional runtime dependency
    CubicSpline = None
    PchipInterpolator = None


DEFAULT_URDF_PATH = (
    "robot_model/rokae_ros_ws/rokae_ros_pkg/src/"
    "rokae_xMateCR7_moveit_config/config/gazebo_xMateCR7.urdf"
)
DEFAULT_JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
DEFAULT_EE_LINK = "xMateCR7_link6"


@dataclass
class ScoreBreakdown:
    total: float
    path_error: float
    mean_error: float
    max_error: float
    max_error_time: float
    vel_cost: float
    accel_cost: float
    limit_cost: float
    z_error: float = 0.0
    topk_error: float = 0.0
    shape_cost: float = 0.0
    cusp_cost: float = 0.0
    waypoint_accel_cost: float = 0.0
    progress_cost: float = 0.0
    tangent_cost: float = 0.0
    segment_cost: float = 0.0


def read_desired_path(path_csv: str) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path_csv)
    required = {"t", "x", "y", "z"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path_csv} missing required columns: {sorted(missing)}")
    t = df["t"].to_numpy(dtype=np.float64)
    p = df[["x", "y", "z"]].to_numpy(dtype=np.float64)
    if len(t) < 3:
        raise ValueError("desired_path.csv must contain at least 3 timesteps")
    return t, p


def read_q_csv(q_csv: str, joint_names: List[str]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    df = pd.read_csv(q_csv)
    missing = set(joint_names) - set(df.columns)
    if missing:
        # Also allow q1..q6 if the file does not use joint names.
        q_cols = [f"q{i + 1}" for i in range(len(joint_names))]
        missing_q = set(q_cols) - set(df.columns)
        if missing_q:
            raise ValueError(
                f"{q_csv} missing joint columns {sorted(missing)} and q-columns {sorted(missing_q)}"
            )
        q = df[q_cols].to_numpy(dtype=np.float64)
    else:
        q = df[joint_names].to_numpy(dtype=np.float64)
    t = df["t"].to_numpy(dtype=np.float64) if "t" in df.columns else None
    return q, t


def save_q_csv(path: str, t: np.ndarray, q: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df = pd.DataFrame(q, columns=[f"q{i + 1}" for i in range(q.shape[1])])
    df.insert(0, "t", t)
    df.to_csv(path, index=False)


def save_ee_csv(path: str, t: np.ndarray, ee: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df = pd.DataFrame({"t": t, "x": ee[:, 0], "y": ee[:, 1], "z": ee[:, 2]})
    df.to_csv(path, index=False)


def save_metrics_json(path: str, score: ScoreBreakdown, args: argparse.Namespace) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    metrics = {
        "total_score": score.total,
        "path_error": score.path_error,
        "mean_error": score.mean_error,
        "max_error": score.max_error,
        "max_error_time": score.max_error_time,
        "vel_cost": score.vel_cost,
        "accel_cost": score.accel_cost,
        "limit_cost": score.limit_cost,
        "z_error": score.z_error,
        "topk_error": score.topk_error,
        "shape_cost": score.shape_cost,
        "cusp_cost": score.cusp_cost,
        "waypoint_accel_cost": score.waypoint_accel_cost,
        "progress_cost": score.progress_cost,
        "tangent_cost": score.tangent_cost,
        "segment_cost": score.segment_cost,
        "weights": {
            "w_path": args.w_path,
            "w_vel": args.w_vel,
            "w_accel": args.w_accel,
            "w_limit": args.w_limit,
            "w_z": args.w_z,
            "w_topk": args.w_topk,
            "w_max": args.w_max,
            "w_shape": args.w_shape,
            "w_cusp": args.w_cusp,
            "w_wp_accel": args.w_wp_accel,
            "w_progress": args.w_progress,
            "w_tangent": args.w_tangent,
            "w_segment": args.w_segment,
        },
        "optimizer": {
            "iterations": args.iterations,
            "num_candidates": args.num_candidates,
            "num_elites": args.num_elites,
            "num_waypoints": args.num_waypoints,
            "num_restarts": args.num_restarts,
            "interp": args.interp,
            "seed": args.seed,
        },
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)


def interpolate_waypoints(
    waypoints: np.ndarray,
    num_steps: int,
    method: str = "linear",
) -> np.ndarray:
    """Interpolate K joint waypoints into a T-step trajectory."""
    k, dof = waypoints.shape
    src = np.linspace(0.0, 1.0, k)
    dst = np.linspace(0.0, 1.0, num_steps)
    q = np.empty((num_steps, dof), dtype=np.float64)

    if method == "linear":
        for j in range(dof):
            q[:, j] = np.interp(dst, src, waypoints[:, j])
        return q

    if method == "cubic":
        if CubicSpline is None:
            raise ImportError("--interp cubic requires scipy")
        for j in range(dof):
            q[:, j] = CubicSpline(src, waypoints[:, j], bc_type="natural")(dst)
        return q

    if method == "pchip":
        if PchipInterpolator is None:
            raise ImportError("--interp pchip requires scipy")
        for j in range(dof):
            q[:, j] = PchipInterpolator(src, waypoints[:, j])(dst)
        return q

    raise ValueError(f"Unknown interpolation method: {method}")


def downsample_to_waypoints(q: np.ndarray, num_waypoints: int) -> np.ndarray:
    t_src = np.linspace(0.0, 1.0, q.shape[0])
    t_wp = np.linspace(0.0, 1.0, num_waypoints)
    wp = np.empty((num_waypoints, q.shape[1]), dtype=np.float64)
    for j in range(q.shape[1]):
        wp[:, j] = np.interp(t_wp, t_src, q[:, j])
    return wp


def load_robot(urdf_path: str) -> URDF:
    if not os.path.exists(urdf_path):
        raise FileNotFoundError(
            f"URDF not found: {urdf_path}\n"
            "Run this from /workspace/artist_trajectory_scoring or pass --urdf_path explicitly."
        )
    return URDF.load(urdf_path, load_meshes=False)


def fk_trajectory(
    robot: URDF,
    q: np.ndarray,
    joint_names: List[str],
    ee_link: str,
) -> np.ndarray:
    """Compute end-effector positions using the yourdfpy API available in this environment.

    Important: this installed yourdfpy version does not provide ``link_fk``.
    It updates the robot configuration with ``update_cfg`` and then queries the
    transform of the end-effector link with ``get_transform``.
    """
    ee = np.empty((q.shape[0], 3), dtype=np.float64)

    update_cfg_fn: Callable[..., Any] = getattr(robot, "update_cfg")
    get_transform_fn: Callable[..., Any] = getattr(robot, "get_transform")

    for i in range(q.shape[0]):
        cfg: Dict[str, float] = {name: float(val) for name, val in zip(joint_names, q[i])}

        # yourdfpy updates its internal scene graph, then get_transform reads
        # the current world/base-to-link transform.
        update_cfg_fn(cfg)

        try:
            transform = get_transform_fn(frame_to=ee_link)
        except TypeError:
            # Older variants may not accept keyword arguments.
            transform = get_transform_fn(ee_link)
        except Exception as exc:
            raise RuntimeError(
                f"Could not get transform for end-effector link '{ee_link}'. "
                "Check --ee_link and the URDF link names."
            ) from exc

        ee[i] = np.asarray(transform, dtype=np.float64)[:3, 3]

    return ee


def joint_limit_penalty(q: np.ndarray, q_min: np.ndarray, q_max: np.ndarray) -> float:
    below = np.maximum(q_min - q, 0.0)
    above = np.maximum(q - q_max, 0.0)
    return float(np.mean(below * below + above * above))


def desired_segment_geometry(desired_xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return segment starts, vectors, lengths, and cumulative arc-length starts."""
    starts = desired_xyz[:-1]
    vectors = desired_xyz[1:] - desired_xyz[:-1]
    lengths = np.linalg.norm(vectors, axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    return starts, vectors, lengths, cumulative[:-1]


def desired_path_tangents(desired_xyz: np.ndarray) -> np.ndarray:
    """Compute normalized per-segment desired tangents from desired_path[:, :3]."""
    _, vectors, lengths, _ = desired_segment_geometry(desired_xyz)
    tangents = np.zeros_like(vectors)
    valid = lengths > 1e-12
    tangents[valid] = vectors[valid] / lengths[valid, None]
    return tangents


def fk_path_tangents(ee_xyz: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute normalized FK step tangents and raw FK step lengths."""
    steps = ee_xyz[1:] - ee_xyz[:-1]
    lengths = np.linalg.norm(steps, axis=1)
    tangents = np.zeros_like(steps)
    valid = lengths > 1e-12
    tangents[valid] = steps[valid] / lengths[valid, None]
    return tangents, lengths


def nearest_desired_segment_progress(
    points_xyz: np.ndarray,
    desired_xyz: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project points to the nearest desired segment.

    Returns:
        segment_distance_sq: squared 3D distance to nearest segment
        progress: approximate arc-length progress along desired path
        segment_index: nearest desired segment index
    """
    starts, vectors, lengths, segment_progress0 = desired_segment_geometry(desired_xyz)
    if len(starts) == 0:
        zeros = np.zeros(points_xyz.shape[0], dtype=np.float64)
        return zeros, zeros, zeros.astype(np.int64)

    best_dist_sq = np.full(points_xyz.shape[0], np.inf, dtype=np.float64)
    best_progress = np.zeros(points_xyz.shape[0], dtype=np.float64)
    best_segment = np.zeros(points_xyz.shape[0], dtype=np.int64)

    for seg_idx, (start, vec, length) in enumerate(zip(starts, vectors, lengths)):
        rel = points_xyz - start[None, :]
        if length > 1e-12:
            u = np.clip((rel @ vec) / (length * length), 0.0, 1.0)
        else:
            u = np.zeros(points_xyz.shape[0], dtype=np.float64)
        projection = start[None, :] + u[:, None] * vec[None, :]
        dist_sq = np.sum((points_xyz - projection) ** 2, axis=1)
        better = dist_sq < best_dist_sq
        best_dist_sq[better] = dist_sq[better]
        best_progress[better] = segment_progress0[seg_idx] + u[better] * length
        best_segment[better] = seg_idx

    return best_dist_sq, best_progress, best_segment


def score_trajectory_arrays(
    ee: np.ndarray,
    desired: np.ndarray,
    t: np.ndarray,
    q: np.ndarray,
    waypoints: np.ndarray,
    q_min: np.ndarray,
    q_max: np.ndarray,
    w_path: float,
    w_vel: float,
    w_accel: float,
    w_limit: float,
    w_z: float,
    w_topk: float,
    topk_frac: float,
    w_max: float,
    w_shape: float,
    w_cusp: float,
    w_wp_accel: float,
    w_progress: float,
    w_tangent: float,
    w_segment: float,
) -> ScoreBreakdown:
    delta = ee - desired
    err_sq = np.sum(delta * delta, axis=1)
    err = np.sqrt(err_sq)
    path_error = float(np.mean(err_sq))
    mean_error = float(np.mean(err))
    max_idx = int(np.argmax(err))
    max_error = float(err[max_idx])
    max_error_time = float(t[max_idx])

    z_error = float(np.mean(delta[:, 2] * delta[:, 2]))
    if w_z > 0.0 and float(np.ptp(desired[:, 2])) > 1e-4:
        # If the target intentionally changes z, avoid over-constraining z
        # relative to the full Cartesian path term.
        z_error *= 0.25

    topk_error = 0.0
    if w_topk > 0.0:
        k = max(1, int(math.ceil(len(err_sq) * topk_frac)))
        topk_error = float(np.mean(np.partition(err_sq, -k)[-k:]))

    shape_cost = 0.0
    cusp_cost = 0.0
    progress_cost = 0.0
    tangent_cost = 0.0
    segment_cost = 0.0

    nearest_seg_idx: Optional[np.ndarray] = None
    if w_progress > 0.0 or w_tangent > 0.0 or w_segment > 0.0:
        segment_dist_sq, desired_progress, nearest_seg_idx = nearest_desired_segment_progress(ee, desired)
        segment_cost = float(np.mean(segment_dist_sq))

        if len(desired_progress) >= 2:
            backward = np.maximum(desired_progress[:-1] - desired_progress[1:], 0.0)
            progress_cost = float(np.mean(backward * backward))

        if len(ee) >= 2:
            desired_tangent = desired_path_tangents(desired)
            fk_tangent, fk_step_len = fk_path_tangents(ee)
            if len(desired_tangent) and nearest_seg_idx is not None:
                # Align each FK motion step with the tangent of the desired
                # segment nearest to the FK point at the start of that step.
                seg_for_step = np.clip(nearest_seg_idx[:-1], 0, len(desired_tangent) - 1)
                active = fk_step_len > 1e-12
                if np.any(active):
                    dots = np.sum(fk_tangent[active] * desired_tangent[seg_for_step[active]], axis=1)
                    dots = np.clip(dots, -1.0, 1.0)
                    poor_alignment = np.maximum(1.0 - dots, 0.0)
                    tangent_cost = float(np.mean(poor_alignment * poor_alignment))

    if len(ee) >= 2:
        ee_step = np.diff(ee, axis=0)
        desired_step = np.diff(desired, axis=0)
        shape_cost = float(np.mean(np.sum((ee_step - desired_step) ** 2, axis=1)))

        if w_cusp > 0.0:
            desired_norm = np.linalg.norm(desired_step, axis=1)
            active = desired_norm > 1e-9
            if np.any(active):
                tangent = desired_step[active] / desired_norm[active, None]
                progress = np.sum(ee_step[active] * tangent, axis=1)
                desired_progress = desired_norm[active]
                backward = np.maximum(-progress, 0.0)
                overspeed = np.maximum(np.abs(progress) - 2.5 * desired_progress, 0.0)
                cusp_cost = float(np.mean(backward * backward + overspeed * overspeed))

    vel = np.diff(q, axis=0)
    accel = q[2:] - 2.0 * q[1:-1] + q[:-2]
    vel_cost = float(np.mean(vel * vel)) if len(vel) else 0.0
    accel_cost = float(np.mean(accel * accel)) if len(accel) else 0.0
    limit_cost = joint_limit_penalty(q, q_min, q_max)

    wp_accel = waypoints[2:] - 2.0 * waypoints[1:-1] + waypoints[:-2]
    waypoint_accel_cost = float(np.mean(wp_accel * wp_accel)) if len(wp_accel) else 0.0

    total = (
        w_path * path_error
        + w_vel * vel_cost
        + w_accel * accel_cost
        + w_limit * limit_cost
        + w_z * z_error
        + w_topk * topk_error
        + w_max * float(np.max(err_sq))
        + w_shape * shape_cost
        + w_cusp * cusp_cost
        + w_wp_accel * waypoint_accel_cost
        + w_progress * progress_cost
        + w_tangent * tangent_cost
        + w_segment * segment_cost
    )
    return ScoreBreakdown(
        total=total,
        path_error=path_error,
        mean_error=mean_error,
        max_error=max_error,
        max_error_time=max_error_time,
        vel_cost=vel_cost,
        accel_cost=accel_cost,
        limit_cost=limit_cost,
        z_error=z_error,
        topk_error=topk_error,
        shape_cost=shape_cost,
        cusp_cost=cusp_cost,
        waypoint_accel_cost=waypoint_accel_cost,
        progress_cost=progress_cost,
        tangent_cost=tangent_cost,
        segment_cost=segment_cost,
    )


def make_initial_waypoints(
    num_waypoints: int,
    num_steps: int,
    dof: int,
    init_q_csv: Optional[str],
    joint_names: List[str],
    initial_q: Optional[List[float]],
) -> np.ndarray:
    if init_q_csv:
        q_init, _ = read_q_csv(init_q_csv, joint_names)
        if q_init.shape[1] != dof:
            raise ValueError(f"init_q_csv has {q_init.shape[1]} joints, expected {dof}")
        return downsample_to_waypoints(q_init, num_waypoints)

    if initial_q is not None:
        if len(initial_q) != dof:
            raise ValueError(f"--initial_q must contain {dof} comma-separated values")
        q0 = np.array(initial_q, dtype=np.float64)
    else:
        q0 = np.zeros(dof, dtype=np.float64)
    return np.repeat(q0[None, :], num_waypoints, axis=0)


def parse_float_list(text: Optional[str]) -> Optional[List[float]]:
    if text is None:
        return None
    vals = [v.strip() for v in text.split(",") if v.strip()]
    return [float(v) for v in vals]


def run_cem_restart(
    args: argparse.Namespace,
    restart_idx: int,
    base_mean_wp: np.ndarray,
    t: np.ndarray,
    desired: np.ndarray,
    joint_names: List[str],
    q_min: np.ndarray,
    q_max: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, ScoreBreakdown]:
    rng = np.random.default_rng(args.seed + restart_idx)
    robot = load_robot(args.urdf_path)
    num_steps = len(t)

    mean_wp = base_mean_wp.copy()
    if restart_idx > 0 and args.restart_jitter > 0.0:
        mean_wp = mean_wp + rng.normal(0.0, args.restart_jitter, size=mean_wp.shape)
        mean_wp = np.clip(mean_wp, q_min, q_max)
    std_wp = np.full_like(mean_wp, args.init_std, dtype=np.float64)

    fixed_first = mean_wp[0].copy() if args.fix_first_waypoint else None
    fixed_last = mean_wp[-1].copy() if args.fix_last_waypoint else None

    best_q: Optional[np.ndarray] = None
    best_ee: Optional[np.ndarray] = None
    best_score: Optional[ScoreBreakdown] = None

    for it in range(1, args.iterations + 1):
        candidate_scores: List[float] = []
        candidate_wps: List[np.ndarray] = []
        candidate_qs: List[np.ndarray] = []
        candidate_ees: List[np.ndarray] = []
        candidate_breakdowns: List[ScoreBreakdown] = []

        for _ in range(args.num_candidates):
            wp = rng.normal(loc=mean_wp, scale=std_wp)
            if fixed_first is not None:
                wp[0] = fixed_first
            if fixed_last is not None:
                wp[-1] = fixed_last
            wp = np.clip(wp, q_min, q_max)
            q = interpolate_waypoints(wp, num_steps, method=args.interp)
            ee = fk_trajectory(robot, q, joint_names, args.ee_link)
            sc = score_trajectory_arrays(
                ee=ee,
                desired=desired,
                t=t,
                q=q,
                waypoints=wp,
                q_min=q_min,
                q_max=q_max,
                w_path=args.w_path,
                w_vel=args.w_vel,
                w_accel=args.w_accel,
                w_limit=args.w_limit,
                w_z=args.w_z,
                w_topk=args.w_topk,
                topk_frac=args.topk_frac,
                w_max=args.w_max,
                w_shape=args.w_shape,
                w_cusp=args.w_cusp,
                w_wp_accel=args.w_wp_accel,
                w_progress=args.w_progress,
                w_tangent=args.w_tangent,
                w_segment=args.w_segment,
            )
            candidate_scores.append(sc.total)
            candidate_wps.append(wp)
            candidate_qs.append(q)
            candidate_ees.append(ee)
            candidate_breakdowns.append(sc)

        order = np.argsort(candidate_scores)
        elite_idx = order[: args.num_elites]
        elite_wps = np.stack([candidate_wps[i] for i in elite_idx], axis=0)

        elite_mean = elite_wps.mean(axis=0)
        elite_std = elite_wps.std(axis=0) + args.min_std

        mean_wp = args.alpha * mean_wp + (1.0 - args.alpha) * elite_mean
        std_wp = args.alpha * std_wp + (1.0 - args.alpha) * elite_std
        std_wp = np.maximum(std_wp * args.std_decay, args.min_std)

        iter_best = int(order[0])
        iter_score = candidate_breakdowns[iter_best]
        if best_score is None or iter_score.total < best_score.total:
            best_score = iter_score
            best_q = candidate_qs[iter_best]
            best_ee = candidate_ees[iter_best]

        if it == 1 or it % args.print_every == 0 or it == args.iterations:
            assert best_score is not None
            print(
                f"restart {restart_idx + 1}/{args.num_restarts} | "
                f"iter {it:04d} | "
                f"iter_total={iter_score.total:.8e} | "
                f"best_total={best_score.total:.8e} | "
                f"path_error={best_score.path_error:.8e} | "
                f"mean={best_score.mean_error:.6f} m | "
                f"max={best_score.max_error:.6f} m @ t={best_score.max_error_time:.3f} | "
                f"z={best_score.z_error:.3e} | "
                f"topk={best_score.topk_error:.3e} | "
                f"vel={best_score.vel_cost:.3e} | "
                f"accel={best_score.accel_cost:.3e} | "
                f"std_mean={float(std_wp.mean()):.4f}"
            )

    assert best_q is not None and best_ee is not None and best_score is not None
    return best_q, best_ee, best_score


def cem_optimize(args: argparse.Namespace) -> Tuple[np.ndarray, np.ndarray, ScoreBreakdown]:
    t, desired = read_desired_path(args.path_csv)

    joint_names = args.joint_names.split(",") if args.joint_names else DEFAULT_JOINT_NAMES
    dof = len(joint_names)
    num_steps = len(t)

    q_min = np.full(dof, args.q_min, dtype=np.float64)
    q_max = np.full(dof, args.q_max, dtype=np.float64)

    base_mean_wp = make_initial_waypoints(
        num_waypoints=args.num_waypoints,
        num_steps=num_steps,
        dof=dof,
        init_q_csv=args.init_q_csv,
        joint_names=joint_names,
        initial_q=parse_float_list(args.initial_q),
    )

    best_q: Optional[np.ndarray] = None
    best_ee: Optional[np.ndarray] = None
    best_score: Optional[ScoreBreakdown] = None

    for restart_idx in range(args.num_restarts):
        restart_q, restart_ee, restart_score = run_cem_restart(
            args=args,
            restart_idx=restart_idx,
            base_mean_wp=base_mean_wp,
            t=t,
            desired=desired,
            joint_names=joint_names,
            q_min=q_min,
            q_max=q_max,
        )
        if best_score is None or restart_score.total < best_score.total:
            best_q = restart_q
            best_ee = restart_ee
            best_score = restart_score

        print(
            f"restart {restart_idx + 1} done | "
            f"best_total={restart_score.total:.8e} | "
            f"mean={restart_score.mean_error:.6f} m | "
            f"max={restart_score.max_error:.6f} m | "
            f"path_error={restart_score.path_error:.8e}"
        )

    assert best_q is not None and best_ee is not None and best_score is not None
    return best_q, best_ee, best_score


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CEM optimizer for desired Cartesian path -> q trajectory")
    p.add_argument("--path_csv", required=True, help="desired_path.csv with columns t,x,y,z")
    p.add_argument("--output_q_csv", required=True, help="where to save optimized q trajectory")
    p.add_argument("--output_ee_csv", required=True, help="where to save FK end-effector trajectory")
    p.add_argument("--metrics_json", default=None, help="where to save optimizer metrics JSON")
    p.add_argument("--urdf_path", default=DEFAULT_URDF_PATH)
    p.add_argument("--ee_link", default=DEFAULT_EE_LINK)
    p.add_argument("--joint_names", default=",".join(DEFAULT_JOINT_NAMES))

    p.add_argument("--init_q_csv", default=None, help="optional warm-start q CSV, e.g. time_conditioned_pred_q.csv")
    p.add_argument("--initial_q", default=None, help="optional comma-separated q start, e.g. 0,0,0,0,0,0")

    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--num_candidates", type=int, default=128)
    p.add_argument("--num_elites", type=int, default=16)
    p.add_argument("--num_waypoints", type=int, default=10)
    p.add_argument("--init_std", type=float, default=0.45)
    p.add_argument("--min_std", type=float, default=0.02)
    p.add_argument("--std_decay", type=float, default=0.98)
    p.add_argument("--alpha", type=float, default=0.15, help="smoothing for CEM mean/std update")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num_restarts", type=int, default=1, help="number of independent CEM restarts")
    p.add_argument("--restart_jitter", type=float, default=0.15, help="waypoint mean perturbation for restarts after the first")
    p.add_argument(
        "--interp",
        choices=["linear", "cubic", "pchip"],
        default="linear",
        help="waypoint interpolation method; cubic/pchip require scipy",
    )

    p.add_argument("--w_path", type=float, default=1.0)
    p.add_argument("--w_vel", type=float, default=0.001)
    p.add_argument("--w_accel", type=float, default=0.01)
    p.add_argument("--w_limit", type=float, default=10.0)
    p.add_argument("--w_z", type=float, default=0.0, help="extra squared z tracking penalty")
    p.add_argument("--w_topk", type=float, default=0.0, help="penalty on worst tracking errors")
    p.add_argument("--topk_frac", type=float, default=0.10, help="fraction of timesteps used by --w_topk")
    p.add_argument("--w_max", type=float, default=0.0, help="penalty on single worst squared tracking error")
    p.add_argument("--w_shape", type=float, default=0.0, help="penalty matching local Cartesian step vectors")
    p.add_argument("--w_cusp", type=float, default=0.0, help="penalty for backward/excessive progress along desired tangent")
    p.add_argument("--w_wp_accel", type=float, default=0.0, help="waypoint-level acceleration penalty")
    p.add_argument("--w_progress", type=float, default=0.0, help="penalty for decreasing nearest-segment progress along desired path")
    p.add_argument("--w_tangent", type=float, default=0.0, help="penalty for poor FK tangent alignment with nearest desired-path tangent")
    p.add_argument("--w_segment", type=float, default=0.0, help="penalty for squared distance to nearest desired-path segment")

    # Conservative generic limits. Replace with exact xMateCR7 limits later if needed.
    p.add_argument("--q_min", type=float, default=-3.141592653589793)
    p.add_argument("--q_max", type=float, default=3.141592653589793)

    p.add_argument("--fix_first_waypoint", action="store_true", help="keep first waypoint fixed to initialization")
    p.add_argument("--fix_last_waypoint", action="store_true", help="keep last waypoint fixed to initialization")
    p.add_argument("--print_every", type=int, default=5)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    if args.num_elites >= args.num_candidates:
        raise ValueError("--num_elites must be smaller than --num_candidates")
    if args.num_waypoints < 2:
        raise ValueError("--num_waypoints must be at least 2")
    if args.num_restarts < 1:
        raise ValueError("--num_restarts must be at least 1")
    if not (0.0 < args.topk_frac <= 1.0):
        raise ValueError("--topk_frac must be in (0, 1]")

    t, _ = read_desired_path(args.path_csv)
    best_q, best_ee, best_score = cem_optimize(args)
    save_q_csv(args.output_q_csv, t, best_q)
    save_ee_csv(args.output_ee_csv, t, best_ee)
    metrics_path = args.metrics_json or os.path.splitext(args.output_ee_csv)[0] + "_metrics.json"
    save_metrics_json(metrics_path, best_score, args)

    print("\nSaved optimized q trajectory to:", args.output_q_csv)
    print("Saved optimized FK end-effector trajectory to:", args.output_ee_csv)
    print("Saved optimizer metrics to:", metrics_path)
    print("Final score:")
    print(f"  total_score:         {best_score.total:.10e}")
    print(f"  path_error:          {best_score.path_error:.10e}")
    print(f"  mean_error:          {best_score.mean_error:.6f} m")
    print(f"  max_error:           {best_score.max_error:.6f} m")
    print(f"  max_error_time:      {best_score.max_error_time:.6f}")
    print(f"  RMS error:           {math.sqrt(best_score.path_error):.6f} m")
    print(f"  z_error:             {best_score.z_error:.10e}")
    print(f"  topk_error:          {best_score.topk_error:.10e}")
    print(f"  shape_cost:          {best_score.shape_cost:.10e}")
    print(f"  cusp_cost:           {best_score.cusp_cost:.10e}")
    print(f"  progress_cost:       {best_score.progress_cost:.10e}")
    print(f"  tangent_cost:        {best_score.tangent_cost:.10e}")
    print(f"  segment_cost:        {best_score.segment_cost:.10e}")
    print(f"  vel_cost:            {best_score.vel_cost:.10e}")
    print(f"  accel_cost:          {best_score.accel_cost:.10e}")
    print(f"  waypoint_accel_cost: {best_score.waypoint_accel_cost:.10e}")
    print(f"  limit_cost:          {best_score.limit_cost:.10e}")


if __name__ == "__main__":
    main()
