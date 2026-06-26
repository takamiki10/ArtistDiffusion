#!/usr/bin/env python3
"""
Generate a joint trajectory seed for a desired Cartesian path using sequential numerical IK.

This script is intentionally independent from the CEM optimizer.  It walks along
the desired Cartesian path and solves one position-only IK problem per timestep,
using the previous timestep's joint solution as the next initial guess.

Example:
    python generate_ik_seed_path.py \
      --path_csv data/cartesian_test_paths/arc_001/desired_path.csv \
      --output_q_csv data/cartesian_test_paths/arc_001/ik_seed_q.csv \
      --output_ee_csv data/cartesian_test_paths/arc_001/ik_seed_ee.csv \
      --smooth_weight 0.01
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from yourdfpy import URDF


DEFAULT_URDF_PATH = (
    "robot_model/rokae_ros_ws/rokae_ros_pkg/src/"
    "rokae_xMateCR7_moveit_config/config/gazebo_xMateCR7.urdf"
)
DEFAULT_JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
DEFAULT_EE_LINK = "xMateCR7_link6"


def read_desired_path(path_csv: Path) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path_csv)
    required = {"t", "x", "y", "z"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path_csv} missing required columns: {sorted(missing)}")
    t = df["t"].to_numpy(dtype=np.float64)
    desired = df[["x", "y", "z"]].to_numpy(dtype=np.float64)
    if len(t) == 0:
        raise ValueError(f"{path_csv} is empty")
    return t, desired


def save_q_csv(path: Path, t: np.ndarray, q: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(q, columns=[f"q{i + 1}" for i in range(q.shape[1])])
    df.insert(0, "t", t)
    df.to_csv(path, index=False)


def save_ee_csv(path: Path, t: np.ndarray, ee: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({"t": t, "x": ee[:, 0], "y": ee[:, 1], "z": ee[:, 2]})
    df.to_csv(path, index=False)


def save_metrics_json(
    path: Path,
    mean_error: float,
    max_error: float,
    max_error_time: float,
    path_error: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metrics = {
        "mean_error": mean_error,
        "max_error": max_error,
        "max_error_time": max_error_time,
        "path_error": path_error,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)


def parse_start_q(text: Optional[str], dof: int) -> Optional[np.ndarray]:
    if text is None:
        return None
    vals = [float(v.strip()) for v in text.split(",") if v.strip()]
    if len(vals) != dof:
        raise ValueError(f"--start_q must contain {dof} comma-separated joint values")
    return np.asarray(vals, dtype=np.float64)


def read_start_q_csv(q_csv: Path, joint_names: Sequence[str]) -> np.ndarray:
    df = pd.read_csv(q_csv)
    if len(df) == 0:
        raise ValueError(f"{q_csv} is empty")

    if set(joint_names).issubset(df.columns):
        return df[list(joint_names)].iloc[0].to_numpy(dtype=np.float64)

    q_cols = [f"q{i + 1}" for i in range(len(joint_names))]
    if set(q_cols).issubset(df.columns):
        return df[q_cols].iloc[0].to_numpy(dtype=np.float64)

    raise ValueError(
        f"{q_csv} must contain either joint columns {list(joint_names)} "
        f"or q-columns {q_cols}"
    )


def load_robot(urdf_path: Path) -> URDF:
    if not urdf_path.exists():
        raise FileNotFoundError(
            f"URDF not found: {urdf_path}\n"
            "Run from artist_trajectory_scoring or pass --urdf_path explicitly."
        )
    return URDF.load(str(urdf_path), load_meshes=False)


def fk_position(
    robot: URDF,
    q: np.ndarray,
    joint_names: Sequence[str],
    ee_link: str,
) -> np.ndarray:
    """Compute one end-effector position using the same yourdfpy FK method as the optimizer."""
    cfg: Dict[str, float] = {name: float(val) for name, val in zip(joint_names, q)}

    get_transform_fn: Callable[..., Any] = getattr(robot, "get_transform")

    robot.update_cfg(cfg)

    try:
        transform = get_transform_fn(frame_to=ee_link)
    except TypeError:
        transform = get_transform_fn(ee_link)
    except Exception as exc:
        raise RuntimeError(
            f"Could not get transform for end-effector link '{ee_link}'. "
            "Check --ee_link and the URDF link names."
        ) from exc

    return np.asarray(transform, dtype=np.float64)[:3, 3]


def find_urdf_joint(robot: URDF, joint_name: str) -> Any:
    for joint in robot.robot.joints:
        if joint.name == joint_name:
            return joint
    raise KeyError(f"Joint '{joint_name}' not found in URDF")


def get_joint_bounds(
    robot: URDF,
    joint_names: Sequence[str],
    fallback_min: float,
    fallback_max: float,
) -> List[Tuple[float, float]]:
    bounds: List[Tuple[float, float]] = []
    for joint_name in joint_names:
        joint = find_urdf_joint(robot, joint_name)
        lower = fallback_min
        upper = fallback_max

        limit = getattr(joint, "limit", None)
        if limit is not None:
            limit_lower = getattr(limit, "lower", None)
            limit_upper = getattr(limit, "upper", None)
            if limit_lower is not None:
                lower = float(limit_lower)
            if limit_upper is not None:
                upper = float(limit_upper)

        if lower >= upper:
            raise ValueError(f"Invalid bounds for {joint_name}: lower={lower}, upper={upper}")
        bounds.append((lower, upper))
    return bounds


def clip_to_bounds(q: np.ndarray, bounds: Sequence[Tuple[float, float]]) -> np.ndarray:
    lower = np.asarray([b[0] for b in bounds], dtype=np.float64)
    upper = np.asarray([b[1] for b in bounds], dtype=np.float64)
    return np.clip(q, lower, upper)


def sample_uniform_q(
    rng: np.random.Generator,
    bounds: Sequence[Tuple[float, float]],
) -> np.ndarray:
    lower = np.asarray([b[0] for b in bounds], dtype=np.float64)
    upper = np.asarray([b[1] for b in bounds], dtype=np.float64)
    return rng.uniform(lower, upper)


@dataclass
class IkAttempt:
    q: np.ndarray
    ee: np.ndarray
    error: float
    success: bool
    nit: int
    message: str


def solve_ik_from_initial_guess(
    robot: URDF,
    p_des: np.ndarray,
    q_init: np.ndarray,
    q_ref: Optional[np.ndarray],
    joint_names: Sequence[str],
    ee_link: str,
    bounds: Sequence[Tuple[float, float]],
    smooth_weight: float,
    maxiter: int,
    ftol: float,
) -> IkAttempt:
    q0 = clip_to_bounds(q_init, bounds)

    def objective(q_candidate: np.ndarray) -> float:
        p_fk = fk_position(robot, q_candidate, joint_names, ee_link)
        pos_err = p_fk - p_des
        cost = float(np.dot(pos_err, pos_err))
        if q_ref is not None and smooth_weight > 0.0:
            smooth_err = q_candidate - q_ref
            cost += float(smooth_weight * np.dot(smooth_err, smooth_err))
        return cost

    result = minimize(
        objective,
        q0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": maxiter, "ftol": ftol},
    )
    q_sol = clip_to_bounds(np.asarray(result.x, dtype=np.float64), bounds)
    p_fk = fk_position(robot, q_sol, joint_names, ee_link)
    err = float(np.linalg.norm(p_fk - p_des))

    return IkAttempt(
        q=q_sol,
        ee=p_fk,
        error=err,
        success=bool(result.success),
        nit=int(result.nit),
        message=str(result.message),
    )


def best_ik_attempt(
    robot: URDF,
    p_des: np.ndarray,
    initial_guesses: Sequence[np.ndarray],
    q_ref: Optional[np.ndarray],
    joint_names: Sequence[str],
    ee_link: str,
    bounds: Sequence[Tuple[float, float]],
    smooth_weight: float,
    maxiter: int,
    ftol: float,
) -> Tuple[IkAttempt, int]:
    if not initial_guesses:
        raise ValueError("best_ik_attempt requires at least one initial guess")

    attempts = [
        solve_ik_from_initial_guess(
            robot=robot,
            p_des=p_des,
            q_init=q0,
            q_ref=q_ref,
            joint_names=joint_names,
            ee_link=ee_link,
            bounds=bounds,
            smooth_weight=smooth_weight,
            maxiter=maxiter,
            ftol=ftol,
        )
        for q0 in initial_guesses
    ]
    best = min(attempts, key=lambda attempt: attempt.error)
    return best, len(attempts)


def solve_sequential_ik(
    robot: URDF,
    t: np.ndarray,
    desired: np.ndarray,
    joint_names: Sequence[str],
    ee_link: str,
    bounds: Sequence[Tuple[float, float]],
    start_q: np.ndarray,
    start_q_csv_first: Optional[np.ndarray],
    smooth_weight: float,
    maxiter: int,
    first_point_maxiter: int,
    ftol: float,
    num_restarts: int,
    retry_error_threshold: float,
    random_seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    dof = len(joint_names)
    q_traj = np.empty((len(t), dof), dtype=np.float64)
    ee_traj = np.empty((len(t), 3), dtype=np.float64)

    rng = np.random.default_rng(random_seed)
    q_prev = clip_to_bounds(start_q, bounds)
    q_zero = np.zeros(dof, dtype=np.float64)

    for i, p_des in enumerate(desired):
        if i == 0:
            initial_guesses = [q_zero]
            if start_q_csv_first is not None:
                initial_guesses.append(start_q_csv_first)
            if np.linalg.norm(start_q - q_zero) > 1e-12:
                initial_guesses.append(start_q)
            initial_guesses.extend(sample_uniform_q(rng, bounds) for _ in range(num_restarts))
            best, restarts_used = best_ik_attempt(
                robot=robot,
                p_des=p_des,
                initial_guesses=initial_guesses,
                q_ref=None,
                joint_names=joint_names,
                ee_link=ee_link,
                bounds=bounds,
                smooth_weight=smooth_weight,
                maxiter=first_point_maxiter,
                ftol=ftol,
            )
        else:
            q_ref = q_prev.copy()
            best, restarts_used = best_ik_attempt(
                robot=robot,
                p_des=p_des,
                initial_guesses=[q_ref],
                q_ref=q_ref,
                joint_names=joint_names,
                ee_link=ee_link,
                bounds=bounds,
                smooth_weight=smooth_weight,
                maxiter=maxiter,
                ftol=ftol,
            )
            if best.error > retry_error_threshold and num_restarts > 0:
                random_guesses = [sample_uniform_q(rng, bounds) for _ in range(num_restarts)]
                retry_best, retry_count = best_ik_attempt(
                    robot=robot,
                    p_des=p_des,
                    initial_guesses=random_guesses,
                    q_ref=q_ref,
                    joint_names=joint_names,
                    ee_link=ee_link,
                    bounds=bounds,
                    smooth_weight=smooth_weight,
                    maxiter=maxiter,
                    ftol=ftol,
                )
                restarts_used += retry_count
                if retry_best.error < best.error:
                    best = retry_best

        q_sol = best.q
        p_fk = best.ee

        q_traj[i] = q_sol
        ee_traj[i] = p_fk

        q_delta = float(np.linalg.norm(q_sol - q_prev))
        q_prev = q_sol

        status = "ok" if best.success else f"warn:{best.message}"
        print(
            f"step {i:04d} | "
            f"target=({p_des[0]:.6f},{p_des[1]:.6f},{p_des[2]:.6f}) | "
            f"best_error={best.error:.6f} m | "
            f"success={best.success} | "
            f"restarts_used={restarts_used} | "
            f"iters={best.nit} | "
            f"{status}"
        )
        if i > 0 and q_delta < 1e-7 and best.error > retry_error_threshold:
            print(
                "  WARNING: joint values barely changed from previous timestep "
                f"(||dq||={q_delta:.3e}) while Cartesian error is high "
                f"({best.error:.6f} m > {retry_error_threshold:.6f} m)."
            )

    return q_traj, ee_traj


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate a sequential numerical IK seed trajectory for a desired Cartesian path."
    )
    parser.add_argument("--path_csv", type=Path, required=True, help="desired_path.csv with columns t,x,y,z")
    parser.add_argument("--output_q_csv", type=Path, required=True)
    parser.add_argument("--output_ee_csv", type=Path, required=True)
    parser.add_argument("--metrics_json", type=Path, default=None)
    parser.add_argument("--urdf_path", type=Path, default=Path(DEFAULT_URDF_PATH))
    parser.add_argument("--ee_link", default=DEFAULT_EE_LINK)
    parser.add_argument("--joint_names", default=",".join(DEFAULT_JOINT_NAMES))
    parser.add_argument("--start_q", default=None, help="comma-separated initial q, e.g. 0,0,0,0,0,0")
    parser.add_argument("--start_q_csv", type=Path, default=None, help="optional q CSV; first row is tried as a first-point seed")
    parser.add_argument("--smooth_weight", type=float, default=0.01)
    parser.add_argument("--maxiter", type=int, default=100)
    parser.add_argument("--first_point_maxiter", type=int, default=500)
    parser.add_argument("--ftol", type=float, default=1e-10)
    parser.add_argument("--num_restarts", type=int, default=8, help="number of random IK restarts when multi-start is used")
    parser.add_argument("--retry_error_threshold", type=float, default=0.02, help="retry later timesteps when FK error exceeds this many meters")
    parser.add_argument("--random_seed", type=int, default=0)
    parser.add_argument("--print_every", type=int, default=10)
    parser.add_argument("--fallback_q_min", type=float, default=-np.pi)
    parser.add_argument("--fallback_q_max", type=float, default=np.pi)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    joint_names = [name.strip() for name in args.joint_names.split(",") if name.strip()]
    if len(joint_names) == 0:
        raise ValueError("--joint_names must contain at least one joint")
    if args.smooth_weight < 0.0:
        raise ValueError("--smooth_weight must be non-negative")
    if args.maxiter < 1:
        raise ValueError("--maxiter must be at least 1")
    if args.first_point_maxiter < 1:
        raise ValueError("--first_point_maxiter must be at least 1")
    if args.num_restarts < 0:
        raise ValueError("--num_restarts must be non-negative")
    if args.retry_error_threshold < 0.0:
        raise ValueError("--retry_error_threshold must be non-negative")

    t, desired = read_desired_path(args.path_csv)
    robot = load_robot(args.urdf_path)
    bounds = get_joint_bounds(robot, joint_names, args.fallback_q_min, args.fallback_q_max)
    start_q = parse_start_q(args.start_q, len(joint_names))
    if start_q is None:
        start_q = np.zeros(len(joint_names), dtype=np.float64)
    start_q_csv_first = None
    if args.start_q_csv is not None:
        start_q_csv_first = clip_to_bounds(read_start_q_csv(args.start_q_csv, joint_names), bounds)

    print("Sequential IK seed generation")
    print(f"  path_csv:      {args.path_csv}")
    print(f"  urdf_path:     {args.urdf_path}")
    print(f"  ee_link:       {args.ee_link}")
    print(f"  joints:        {','.join(joint_names)}")
    print(f"  smooth_weight: {args.smooth_weight}")
    print(f"  maxiter:       {args.maxiter}")
    print(f"  first maxiter: {args.first_point_maxiter}")
    print(f"  ftol:          {args.ftol}")
    print(f"  num_restarts:  {args.num_restarts}")
    print(f"  retry thresh:  {args.retry_error_threshold}")
    print(f"  random_seed:   {args.random_seed}")
    if args.start_q_csv is not None:
        print(f"  start_q_csv:   {args.start_q_csv}")
    print("  bounds:")
    for joint_name, (lower, upper) in zip(joint_names, bounds):
        print(f"    {joint_name}: [{lower:.6f}, {upper:.6f}]")

    q_traj, ee_traj = solve_sequential_ik(
        robot=robot,
        t=t,
        desired=desired,
        joint_names=joint_names,
        ee_link=args.ee_link,
        bounds=bounds,
        start_q=start_q,
        start_q_csv_first=start_q_csv_first,
        smooth_weight=args.smooth_weight,
        maxiter=args.maxiter,
        first_point_maxiter=args.first_point_maxiter,
        ftol=args.ftol,
        num_restarts=args.num_restarts,
        retry_error_threshold=args.retry_error_threshold,
        random_seed=args.random_seed,
    )

    save_q_csv(args.output_q_csv, t, q_traj)
    save_ee_csv(args.output_ee_csv, t, ee_traj)

    error = np.linalg.norm(ee_traj - desired, axis=1)
    mean_error = float(np.mean(error))
    max_idx = int(np.argmax(error))
    max_error = float(error[max_idx])
    max_error_time = float(t[max_idx])
    path_error = float(np.mean(error * error))
    metrics_path = args.metrics_json or args.output_ee_csv.with_name(
        args.output_ee_csv.stem + "_metrics.json"
    )
    save_metrics_json(metrics_path, mean_error, max_error, max_error_time, path_error)

    print("\nSaved IK seed q trajectory to:", args.output_q_csv)
    print("Saved IK seed FK end-effector trajectory to:", args.output_ee_csv)
    print("Saved IK seed metrics to:", metrics_path)
    print("Final IK seed metrics:")
    print(f"  mean_error: {mean_error:.8f} m")
    print(f"  max_error:  {max_error:.8f} m")
    print(f"  max_error_time: {max_error_time:.8f}")
    print(f"  max_error_index: {max_idx}")
    print(f"  path_error: {path_error:.8e}")


if __name__ == "__main__":
    main()
