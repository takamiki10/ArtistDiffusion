#!/usr/bin/env python3
"""Diagnose conditional diffusion v2 reconstruction from partially noised expert deltas."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from train_conditional_diffusion_trajectory_v2 import (
    ConditionalTrajectoryDenoiser,
    make_beta_schedule,
)


Q_COLUMNS = [f"q{i}" for i in range(1, 7)]
XYZ_COLUMNS = ["x", "y", "z"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="data/cartesian_expert_dataset_v3/diffusion_v2/conditional_diffusion_v2.pt")
    parser.add_argument("--test_npz", default="data/cartesian_expert_dataset_v3/diffusion_v2/diffusion_test_v2.npz")
    parser.add_argument("--norm_stats", default="data/cartesian_expert_dataset_v3/diffusion_v2/diffusion_norm_stats_v2.json")
    parser.add_argument("--dataset_dir", default="data/cartesian_expert_dataset_v3/experts/test")
    parser.add_argument("--output_csv", default="data/cartesian_expert_dataset_v3/diffusion_v2/reconstruction_diagnostic_v2.csv")
    parser.add_argument("--output_dir", default="data/cartesian_expert_dataset_v3/diffusion_v2/reconstruction_diagnostics")
    parser.add_argument("--timesteps", default="0,10,25,50,75,99")
    parser.add_argument("--max_paths", type=int, default=83)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ee_link", default=None)
    return parser.parse_args()


def select_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_timesteps(text: str) -> list[int]:
    timesteps = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not timesteps:
        raise ValueError("--timesteps must contain at least one integer")
    return timesteps


def load_checkpoint_model(model_path: Path, device: torch.device) -> tuple[ConditionalTrajectoryDenoiser, int]:
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
    model.eval()
    return model, num_steps


def load_norm_stats(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8") as f:
        stats = json.load(f)
    mean = np.asarray(stats["delta_q_mean"], dtype=np.float32)
    std = np.asarray(stats["delta_q_std"], dtype=np.float32)
    if mean.shape != (6,) or std.shape != (6,):
        raise ValueError(f"Expected delta_q_mean/delta_q_std shape (6,), got {mean.shape}/{std.shape}")
    return mean, std


def load_required_npz(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    required = [
        "condition_features_norm",
        "delta_q_norm",
        "delta_q",
        "desired_paths",
        "expert_q",
        "q_start",
        "path_names",
    ]
    missing = [key for key in required if key not in data.files]
    if missing:
        raise KeyError(f"Missing keys in {path}: {missing}. Available keys: {data.files}")
    return {key: data[key] for key in required}


def load_robot():
    try:
        import score_trajectory
    except ImportError as exc:
        raise RuntimeError("Could not import score_trajectory.py for project FK setup") from exc

    for fn_name in ("load_robot", "create_robot", "make_robot", "get_robot"):
        fn = getattr(score_trajectory, fn_name, None)
        if callable(fn):
            try:
                return fn()
            except TypeError:
                pass

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

    try:
        import score_trajectory
    except ImportError:
        score_trajectory = None

    if score_trajectory is not None:
        for attr_name in ("ee_link", "EE_LINK", "end_effector_link", "END_EFFECTOR_LINK", "ROKAE_EE_LINK"):
            value = getattr(score_trajectory, attr_name, None)
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
        transform = robot.get_transform(frame_to=ee_link)
        ee_points.append(np.asarray(transform, dtype=np.float64)[:3, 3])
    return np.asarray(ee_points, dtype=np.float64)


def denormalize_delta_q(delta_q_norm: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return delta_q_norm * std[None, :] + mean[None, :]


def compute_metrics(desired: np.ndarray, pred_ee: np.ndarray, q_pred: np.ndarray, expert_q: np.ndarray) -> dict:
    q_delta = q_pred - expert_q
    q_rmse = float(np.sqrt(np.mean(np.square(q_delta))))
    q_mae = float(np.mean(np.abs(q_delta)))

    try:
        from trajectory_costs import compute_stanford_style_trajectory_cost

        cost_metrics = compute_stanford_style_trajectory_cost(
            desired,
            pred_ee,
            q_pred,
            w_path=1.0,
            w_x=1.0,
            w_y=1.0,
            w_z=1.0,
            w_vel=0.0,
            w_accel=0.0,
        )
        path_error = float(cost_metrics["path_error"])
        mean_error = float(cost_metrics["mean_error"])
        max_error = float(cost_metrics["max_error"])
    except Exception:
        if desired.shape != pred_ee.shape:
            raise ValueError(f"Shape mismatch: desired {desired.shape}, pred_ee {pred_ee.shape}")
        errors = np.linalg.norm(pred_ee - desired, axis=1)
        path_error = float(np.mean(np.square(errors)))
        mean_error = float(np.mean(errors))
        max_error = float(np.max(errors))

    return {
        "q_rmse": q_rmse,
        "q_mae": q_mae,
        "path_error": path_error,
        "mean_error": mean_error,
        "max_error": max_error,
        "accepted": bool(mean_error <= 0.010 and max_error <= 0.030),
    }


def reconstruct_from_noisy_expert(
    model: ConditionalTrajectoryDenoiser,
    condition_norm: np.ndarray,
    x0_norm: np.ndarray,
    timestep: int,
    alpha_bars: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    condition = torch.from_numpy(condition_norm.astype(np.float32)).unsqueeze(0).to(device)
    x0 = torch.from_numpy(x0_norm.astype(np.float32)).unsqueeze(0).to(device)
    noise = torch.randn_like(x0)

    alpha_bar_t = alpha_bars[timestep]
    sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
    sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - alpha_bar_t)

    x_t = sqrt_alpha_bar_t * x0 + sqrt_one_minus_alpha_bar_t * noise
    t = torch.full((1,), timestep, device=device, dtype=torch.long)

    with torch.no_grad():
        pred_noise = model(x_t, condition, t)
        x0_pred = (x_t - sqrt_one_minus_alpha_bar_t * pred_noise) / sqrt_alpha_bar_t

    return x0_pred.squeeze(0).cpu().numpy().astype(np.float32)


def save_overlay(path: Path, desired: np.ndarray, pred: np.ndarray, expert_ee_csv: Path) -> None:
    plt.figure(figsize=(6, 6))
    plt.plot(desired[:, 0], desired[:, 1], label="desired path", linewidth=2)

    if expert_ee_csv.exists():
        try:
            expert_ee = pd.read_csv(expert_ee_csv)
            if all(col in expert_ee.columns for col in XYZ_COLUMNS):
                plt.plot(
                    expert_ee["x"].to_numpy(dtype=float),
                    expert_ee["y"].to_numpy(dtype=float),
                    label="expert IK FK path",
                    linewidth=2,
                )
        except Exception as exc:
            print(f"WARNING: could not read expert EE overlay {expert_ee_csv}: {exc}")

    plt.plot(pred[:, 0], pred[:, 1], label="reconstructed diffusion FK path", linewidth=2)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.legend()
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=160)
    plt.close()


def save_t50_plots(rows: list[dict], output_dir: Path, dataset_dir: Path) -> None:
    t50_rows = [row for row in rows if row["diagnostic_timestep"] == 50 and "pred_ee" in row and "desired_path" in row]
    if not t50_rows:
        print("WARNING: no timestep 50 rows available for optional plots")
        return

    sorted_rows = sorted(t50_rows, key=lambda row: row["mean_error"])
    selected = {
        "best": sorted_rows[0],
        "median": sorted_rows[len(sorted_rows) // 2],
        "worst": sorted_rows[-1],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    for label, row in selected.items():
        save_overlay(
            output_dir / f"reconstruction_t50_{label}.png",
            row["desired_path"],
            row["pred_ee"],
            dataset_dir / row["path_name"] / "expert_ee.csv",
        )


def print_grouped_summary(summary_df: pd.DataFrame) -> None:
    if summary_df.empty:
        print("No diagnostic rows were produced.")
        return

    print("\nGrouped summary by diagnostic_timestep:")
    grouped = summary_df.groupby("diagnostic_timestep", sort=True)
    for timestep, group in grouped:
        print(
            f"t={int(timestep):3d} "
            f"count={len(group):3d} "
            f"accepted={int(group['accepted'].sum()):3d} "
            f"mean_q_rmse={group['q_rmse'].mean():.6f} "
            f"mean_q_mae={group['q_mae'].mean():.6f} "
            f"mean_mean_error={group['mean_error'].mean():.6f} "
            f"mean_max_error={group['max_error'].mean():.6f}"
        )


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = select_device(args.device)
    timesteps = parse_timesteps(args.timesteps)
    model, num_steps = load_checkpoint_model(Path(args.model), device)
    _, _, alpha_bars = make_beta_schedule(num_steps, device)

    invalid_timesteps = [t for t in timesteps if t < 0 or t >= num_steps]
    if invalid_timesteps:
        raise ValueError(f"Diagnostic timesteps must be in [0, {num_steps - 1}], got {invalid_timesteps}")

    data = load_required_npz(Path(args.test_npz))
    delta_q_mean, delta_q_std = load_norm_stats(Path(args.norm_stats))
    dataset_dir = Path(args.dataset_dir)
    output_csv = Path(args.output_csv)
    output_dir = Path(args.output_dir)

    robot = load_robot()
    ee_link = infer_ee_link(robot, args.ee_link)
    joint_names = infer_joint_names(robot)

    condition_all = data["condition_features_norm"].astype(np.float32)
    delta_q_norm_all = data["delta_q_norm"].astype(np.float32)
    desired_all = data["desired_paths"].astype(np.float32)
    expert_q_all = data["expert_q"].astype(np.float32)
    q_start_all = data["q_start"].astype(np.float32)
    path_names = [str(name) for name in data["path_names"]]
    max_paths = min(args.max_paths, len(path_names)) if args.max_paths > 0 else len(path_names)

    rows_for_csv: list[dict] = []
    plot_rows: list[dict] = []

    for path_index in range(max_paths):
        path_name = path_names[path_index]
        condition_norm = condition_all[path_index]
        x0_norm = delta_q_norm_all[path_index]
        desired = desired_all[path_index]
        expert_q = expert_q_all[path_index]
        q_start = q_start_all[path_index]

        for timestep in timesteps:
            try:
                x0_pred_norm = reconstruct_from_noisy_expert(
                    model=model,
                    condition_norm=condition_norm,
                    x0_norm=x0_norm,
                    timestep=timestep,
                    alpha_bars=alpha_bars,
                    device=device,
                )
                delta_q_pred = denormalize_delta_q(x0_pred_norm, delta_q_mean, delta_q_std)
                q_pred = q_start[None, :] + delta_q_pred
                pred_ee = fk_trajectory(robot, q_pred, ee_link, joint_names)
                metrics = compute_metrics(desired, pred_ee, q_pred, expert_q)

                csv_row = {
                    "path_name": path_name,
                    "diagnostic_timestep": timestep,
                    **metrics,
                }
                rows_for_csv.append(csv_row)

                if timestep == 50:
                    plot_rows.append(
                        {
                            **csv_row,
                            "desired_path": desired,
                            "pred_ee": pred_ee,
                        }
                    )
            except Exception as exc:
                print(f"WARNING: failed {path_name} at diagnostic timestep {timestep}: {exc}")
                continue

    summary_df = pd.DataFrame(
        rows_for_csv,
        columns=[
            "path_name",
            "diagnostic_timestep",
            "q_rmse",
            "q_mae",
            "path_error",
            "mean_error",
            "max_error",
            "accepted",
        ],
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_csv, index=False)

    print_grouped_summary(summary_df)
    save_t50_plots(plot_rows, output_dir, dataset_dir)

    print(f"\nWrote output CSV: {output_csv}")
    print(f"Wrote output plot folder: {output_dir}")


if __name__ == "__main__":
    main()
