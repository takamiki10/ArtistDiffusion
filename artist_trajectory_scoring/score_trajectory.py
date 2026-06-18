#!/usr/bin/env python3

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from yourdfpy import URDF

URDF_PATH = "robot_model/rokae_ros_ws/rokae_ros_pkg/src/rokae_xMateCR7_moveit_config/config/gazebo_xMateCR7.urdf"

ROKAE_JOINT_NAMES = [
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
]

ROKAE_EE_LINK = "xMateCR7_link6"

def rokae_forward_kinematics(q: np.ndarray) -> np.ndarray:
    """
    Compute end-effector Cartesian positions for the xMateCR7.

    Input:
        q: shape (T, 6), joint angles in radians

    Output:
        p_ee: shape (T, 3), end-effector positions
    """
    if q.shape[1] != len(ROKAE_JOINT_NAMES):
        raise ValueError(
            f"Expected {len(ROKAE_JOINT_NAMES)} joints, got {q.shape[1]}."
        )

    robot = URDF.load(URDF_PATH, load_meshes=False)

    positions = []

    for q_t in q:
        cfg = {
            joint_name: float(joint_value)
            for joint_name, joint_value in zip(ROKAE_JOINT_NAMES, q_t)
        }

        robot.update_cfg(cfg)

        transform = robot.get_transform(
            frame_to=ROKAE_EE_LINK,
            frame_from="world"
        )

        pos = transform[:3, 3]
        positions.append(pos)

    return np.asarray(positions)

def load_joint_trajectory(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load candidate joint trajectory CSV.

    Expected columns:
        t,q1,q2,q3,q4,q5,q6

    Returns:
        t: shape (T,)
        q: shape (T, DOF)
    """
    df = pd.read_csv(path)

    if "t" not in df.columns:
        raise ValueError("Joint trajectory CSV must contain column 't'.")

    joint_cols = [c for c in df.columns if c.startswith("q")]
    if len(joint_cols) == 0:
        raise ValueError("Joint trajectory CSV must contain q columns, e.g. q1,q2,...")

    t = df["t"].to_numpy(dtype=float)
    q = df[joint_cols].to_numpy(dtype=float)

    return t, q


def load_desired_path(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load desired Cartesian path CSV.

    Expected columns:
        t,x,y,z

    Returns:
        t: shape (T,)
        p_des: shape (T, 3)
    """
    df = pd.read_csv(path)

    required = ["t", "x", "y", "z"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Desired path CSV must contain column '{col}'.")

    t = df["t"].to_numpy(dtype=float)
    p_des = df[["x", "y", "z"]].to_numpy(dtype=float)

    return t, p_des


def placeholder_forward_kinematics(q: np.ndarray) -> np.ndarray:
    """
    Temporary placeholder FK.

    Input:
        q: shape (T, DOF)

    Output:
        p_ee: shape (T, 3)

    Replace this later with real ROKAE forward kinematics.

    Current placeholder:
        Uses first three joints as a fake Cartesian position.
        This is only for testing the scoring pipeline.
    """
    if q.shape[1] < 3:
        raise ValueError("Need at least 3 joint columns for placeholder FK.")

    p_ee = q[:, :3].copy()
    return p_ee


def compute_path_error(p_ee: np.ndarray, p_des: np.ndarray) -> float:
    """
    Mean squared Cartesian path error.

    J_path = mean_t ||p_ee[t] - p_des[t]||^2
    """
    if p_ee.shape != p_des.shape:
        raise ValueError(f"Shape mismatch: p_ee {p_ee.shape}, p_des {p_des.shape}")

    err = p_ee - p_des
    squared_dist = np.sum(err ** 2, axis=1)
    return float(np.mean(squared_dist))


def compute_smoothness_cost(q: np.ndarray) -> float:
    """
    Joint acceleration smoothness cost.

    Uses second finite difference:
        Δ²q[t] = q[t+1] - 2q[t] + q[t-1]

    J_smooth = mean_t ||Δ²q[t]||^2
    """
    if q.shape[0] < 3:
        return 0.0

    ddq = q[2:] - 2.0 * q[1:-1] + q[:-2]
    squared_accel = np.sum(ddq ** 2, axis=1)
    return float(np.mean(squared_accel))


def score_trajectory(
    q_csv: Path,
    path_csv: Path,
    w_path: float,
    w_smooth: float,
) -> dict:
    """
    Compute total trajectory score.

    Lower score is better.
    """
    t_q, q = load_joint_trajectory(q_csv)
    t_des, p_des = load_desired_path(path_csv)

    if len(t_q) != len(t_des):
        raise ValueError(
            f"Length mismatch: joint trajectory has {len(t_q)} steps, "
            f"desired path has {len(t_des)} steps."
        )

    # Later replace this with real ROKAE FK.
    p_ee = rokae_forward_kinematics(q)

    j_path = compute_path_error(p_ee, p_des)
    j_smooth = compute_smoothness_cost(q)

    total = w_path * j_path + w_smooth * j_smooth

    return {
        "total_score": total,
        "path_error": j_path,
        "smoothness_cost": j_smooth,
        "w_path": w_path,
        "w_smooth": w_smooth,
        "num_steps": len(t_q),
        "num_joints": q.shape[1],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--q_csv", required=True)
    parser.add_argument("--path_csv", required=True)
    parser.add_argument("--w_path", type=float, default=1.0)
    parser.add_argument("--w_smooth", type=float, default=0.01)

    parser.add_argument(
        "--save_ee_csv",
        type=str,
        default=None,
        help="Optional output CSV path for FK-computed end-effector trajectory."
    )

    args = parser.parse_args()

    q_df = pd.read_csv(args.q_csv)
    path_df = pd.read_csv(args.path_csv)

    q = q_df[["q1", "q2", "q3", "q4", "q5", "q6"]].to_numpy()
    desired_path = path_df[["x", "y", "z"]].to_numpy()

    ee_positions = rokae_forward_kinematics(q)

    if args.save_ee_csv is not None:
        ee_df = pd.DataFrame({
            "t": q_df["t"].to_numpy(),
            "x": ee_positions[:, 0],
            "y": ee_positions[:, 1],
            "z": ee_positions[:, 2],
        })
        ee_df.to_csv(args.save_ee_csv, index=False)
        print(f"Saved FK end-effector trajectory to: {args.save_ee_csv}")

    path_error = compute_path_error(ee_positions, desired_path)
    smoothness_cost = compute_smoothness_cost(q)

    total_score = args.w_path * path_error + args.w_smooth * smoothness_cost

    print(f"total_score: {total_score}")
    print(f"path_error: {path_error}")
    print(f"smoothness_cost: {smoothness_cost}")
    print(f"w_path: {args.w_path}")
    print(f"w_smooth: {args.w_smooth}")
    print(f"num_steps: {q.shape[0]}")
    print(f"num_joints: {q.shape[1]}")


if __name__ == "__main__":
    main()