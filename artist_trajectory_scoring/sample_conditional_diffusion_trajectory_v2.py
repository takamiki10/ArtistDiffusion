#!/usr/bin/env python3
"""Sample and score diffusion-only v2 trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from train_conditional_diffusion_trajectory_v2 import ConditionalTrajectoryDenoiser, make_beta_schedule
from trajectory_costs import compute_stanford_style_trajectory_cost


Q_COLUMNS = [f"q{i}" for i in range(1, 7)]
XYZ_COLUMNS = ["x", "y", "z"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="data/cartesian_expert_dataset_v3/diffusion_v2/conditional_diffusion_v2.pt")
    parser.add_argument("--test_npz", default="data/cartesian_expert_dataset_v3/diffusion_v2/diffusion_test_v2.npz")
    parser.add_argument("--norm_stats", default="data/cartesian_expert_dataset_v3/diffusion_v2/diffusion_norm_stats_v2.json")
    parser.add_argument("--dataset_dir", default="data/cartesian_expert_dataset_v3/experts/test")
    parser.add_argument("--output_dir", default="data/cartesian_expert_dataset_v3/diffusion_v2/samples")
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--max_paths", type=int, default=10)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--w_path", type=float, default=1.0)
    parser.add_argument("--w_x", type=float, default=1.0)
    parser.add_argument("--w_y", type=float, default=1.0)
    parser.add_argument("--w_z", type=float, default=1.0)
    parser.add_argument("--w_vel", type=float, default=0.0)
    parser.add_argument("--w_accel", type=float, default=0.01)
    parser.add_argument("--ee_link", default=None)
    return parser.parse_args()


def select_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def reverse_sample(
    model: ConditionalTrajectoryDenoiser,
    condition: torch.Tensor,
    num_steps: int,
    device: torch.device,
) -> torch.Tensor:
    betas, alphas, alpha_bars = make_beta_schedule(num_steps, device)
    x = torch.randn((1, condition.shape[1], 6), device=device)
    model.eval()

    with torch.no_grad():
        for step in reversed(range(num_steps)):
            t = torch.full((1,), step, device=device, dtype=torch.long)
            pred_noise = model(x, condition, t)
            beta_t = betas[step]
            alpha_t = alphas[step]
            alpha_bar_t = alpha_bars[step]
            coef = beta_t / torch.sqrt(1.0 - alpha_bar_t)
            mean = (x - coef * pred_noise) / torch.sqrt(alpha_t)
            if step > 0:
                noise = torch.randn_like(x)
                x = mean + torch.sqrt(beta_t) * noise
            else:
                x = mean

    return x.squeeze(0).cpu()


def load_model(model_path: Path, device: torch.device) -> tuple[ConditionalTrajectoryDenoiser, int]:
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(model_path, map_location=device)
    hidden_dim = int(checkpoint.get("hidden_dim", 256))
    condition_dim = int(checkpoint.get("condition_dim", 13))
    target_dim = int(checkpoint.get("target_dim", 6))
    num_steps = int(checkpoint.get("num_diffusion_steps", 100))
    model = ConditionalTrajectoryDenoiser(
        condition_dim=condition_dim,
        target_dim=target_dim,
        hidden_dim=hidden_dim,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, num_steps


def denormalize_delta_q(delta_q_norm: np.ndarray, stats: dict) -> np.ndarray:
    mean = np.asarray(stats["delta_q_mean"], dtype=np.float32)
    std = np.asarray(stats["delta_q_std"], dtype=np.float32)
    return delta_q_norm * std[None, :] + mean[None, :]


def load_robot():
    try:
        import score_trajectory
    except ImportError as exc:
        raise RuntimeError("Could not import score_trajectory.py for project FK setup") from exc

    for fn_name in ("load_robot", "create_robot", "make_robot", "get_robot"):
        fn = getattr(score_trajectory, fn_name, None)
        if callable(fn):
            return fn()

    for attr_name in ("robot", "ROBOT"):
        robot = getattr(score_trajectory, attr_name, None)
        if robot is not None:
            return robot

    urdf_path_value = getattr(score_trajectory, "URDF_PATH", None)
    if urdf_path_value is not None:
        try:
            from yourdfpy import URDF
        except ImportError as exc:
            raise RuntimeError("score_trajectory.py exposes URDF_PATH, but yourdfpy is not installed") from exc

        urdf_path = Path(urdf_path_value)
        if not urdf_path.exists():
            urdf_path = Path(__file__).resolve().parent / urdf_path
        if not urdf_path.exists():
            raise FileNotFoundError(f"URDF not found: {urdf_path}")
        return URDF.load(str(urdf_path), load_meshes=False)

    raise RuntimeError(
        "score_trajectory.py was found, but no robot factory/global was recognized. "
        "Expected one of load_robot/create_robot/make_robot/get_robot, robot/ROBOT, or URDF_PATH."
    )


def infer_ee_link(robot, explicit_ee_link: str | None = None) -> str:
    if explicit_ee_link:
        return explicit_ee_link
    for attr_name in ("ee_link", "EE_LINK", "end_effector_link", "END_EFFECTOR_LINK"):
        value = getattr(robot, attr_name, None)
        if isinstance(value, str):
            return value
    for attr_name in ("ee_link", "EE_LINK", "end_effector_link", "END_EFFECTOR_LINK", "ROKAE_EE_LINK"):
        try:
            import score_trajectory

            value = getattr(score_trajectory, attr_name, None)
        except ImportError:
            value = None
        if isinstance(value, str):
            return value
    return "tool0"


def infer_joint_names(robot) -> list[str]:
    try:
        import score_trajectory
    except ImportError:
        score_trajectory = None

    if score_trajectory is not None:
        for attr_name in (
            "joint_names",
            "JOINT_NAMES",
            "active_joint_names",
            "ACTIVE_JOINT_NAMES",
            "ROKAE_JOINT_NAMES",
        ):
            value = getattr(score_trajectory, attr_name, None)
            if isinstance(value, (list, tuple)) and len(value) == 6:
                return [str(name) for name in value]

    for attr_name in ("joint_names", "active_joint_names"):
        value = getattr(robot, attr_name, None)
        if isinstance(value, (list, tuple)) and len(value) == 6:
            return [str(name) for name in value]

    return Q_COLUMNS


def fk_trajectory(robot, q: np.ndarray, ee_link: str, joint_names: list[str]) -> np.ndarray:
    ee_points = []
    for q_row in q:
        cfg = {name: float(value) for name, value in zip(joint_names, q_row)}
        robot.update_cfg(cfg)
        try:
            transform = robot.get_transform(frame_to=ee_link, frame_from="world")
        except TypeError:
            transform = robot.get_transform(frame_to=ee_link)
        ee_points.append(np.asarray(transform)[:3, 3])
    return np.asarray(ee_points, dtype=np.float64)


def save_q_csv(path: Path, q: np.ndarray) -> None:
    df = pd.DataFrame(q, columns=Q_COLUMNS)
    df.insert(0, "t", np.arange(q.shape[0], dtype=float))
    df.to_csv(path, index=False)


def save_ee_csv(path: Path, ee: np.ndarray) -> None:
    df = pd.DataFrame(ee, columns=XYZ_COLUMNS)
    df.insert(0, "t", np.arange(ee.shape[0], dtype=float))
    df.to_csv(path, index=False)


def copy_or_write_csv(source: Path, dest: Path, fallback: np.ndarray, columns: list[str]) -> None:
    if source.exists():
        pd.read_csv(source).to_csv(dest, index=False)
        return
    df = pd.DataFrame(fallback, columns=columns)
    df.insert(0, "t", np.arange(fallback.shape[0], dtype=float))
    df.to_csv(dest, index=False)


def save_overlay(path: Path, desired: np.ndarray, pred: np.ndarray, expert_ee_csv: Path) -> None:
    plt.figure(figsize=(6, 6))
    plt.plot(desired[:, 0], desired[:, 1], label="desired path", linewidth=2)
    if expert_ee_csv.exists():
        expert_ee = pd.read_csv(expert_ee_csv)[XYZ_COLUMNS].to_numpy(dtype=float)
        plt.plot(expert_ee[:, 0], expert_ee[:, 1], label="expert IK FK path", linewidth=2)
    plt.plot(pred[:, 0], pred[:, 1], label="diffusion predicted FK path", linewidth=2)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = select_device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(args.test_npz, allow_pickle=True)
    with Path(args.norm_stats).open("r", encoding="utf-8") as f:
        stats = json.load(f)

    model, num_steps = load_model(Path(args.model), device)
    robot = load_robot()
    ee_link = infer_ee_link(robot, args.ee_link)
    joint_names = infer_joint_names(robot)
    condition_all = data["condition_features_norm"].astype(np.float32)
    desired_all = data["desired_paths"].astype(np.float32)
    expert_q_all = data["expert_q"].astype(np.float32)
    q_start_all = data["q_start"].astype(np.float32)
    path_names = [str(name) for name in data["path_names"]]
    max_paths = min(args.max_paths, len(path_names)) if args.max_paths > 0 else len(path_names)
    summary_rows = []

    for path_index in range(max_paths):
        path_name = path_names[path_index]
        try:
            condition = torch.from_numpy(condition_all[path_index : path_index + 1]).to(device)
            desired = desired_all[path_index]
            expert_q = expert_q_all[path_index]
            q_start = q_start_all[path_index]
            source_folder = Path(args.dataset_dir) / path_name

            for sample_index in range(args.num_samples):
                sample_folder = output_dir / path_name / f"sample_{sample_index:03d}"
                sample_folder.mkdir(parents=True, exist_ok=True)

                delta_q_norm = reverse_sample(model, condition, num_steps, device).numpy()
                delta_q = denormalize_delta_q(delta_q_norm, stats)
                q_pred = q_start[None, :] + delta_q
                pred_ee = fk_trajectory(robot, q_pred, ee_link, joint_names)

                metrics = compute_stanford_style_trajectory_cost(
                    desired,
                    pred_ee,
                    q_pred,
                    w_path=args.w_path,
                    w_x=args.w_x,
                    w_y=args.w_y,
                    w_z=args.w_z,
                    w_vel=args.w_vel,
                    w_accel=args.w_accel,
                )
                metrics["accepted"] = bool(metrics["mean_error"] <= 0.010 and metrics["max_error"] <= 0.030)

                copy_or_write_csv(source_folder / "desired_path.csv", sample_folder / "desired_path.csv", desired, XYZ_COLUMNS)
                copy_or_write_csv(source_folder / "expert_q.csv", sample_folder / "expert_q.csv", expert_q, Q_COLUMNS)
                save_q_csv(sample_folder / "diffusion_pred_q.csv", q_pred)
                save_ee_csv(sample_folder / "diffusion_pred_ee.csv", pred_ee)
                with (sample_folder / "diffusion_metrics.json").open("w", encoding="utf-8") as f:
                    json.dump(metrics, f, indent=2)
                save_overlay(sample_folder / "diffusion_overlay.png", desired, pred_ee, source_folder / "expert_ee.csv")

                summary_rows.append(
                    {
                        "path_name": path_name,
                        "sample_index": sample_index,
                        **metrics,
                        "output_folder": str(sample_folder),
                    }
                )
        except Exception as exc:
            print(f"WARNING: failed path {path_name}: {exc}")
            continue

    summary_path = output_dir / "diffusion_v2_sample_summary.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
