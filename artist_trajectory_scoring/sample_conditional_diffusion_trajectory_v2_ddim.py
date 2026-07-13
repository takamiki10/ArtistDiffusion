#!/usr/bin/env python3
"""Sample and score conditional diffusion v2 trajectories with DDIM."""

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
from trajectory_costs import compute_stanford_style_trajectory_cost


Q_COLUMNS = [f"q{i}" for i in range(1, 7)]
XYZ_COLUMNS = ["x", "y", "z"]
NUM_TRAJECTORY_STEPS = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="data/cartesian_expert_dataset_v3/diffusion_v2/conditional_diffusion_v2.pt")
    parser.add_argument("--test_npz", default="data/cartesian_expert_dataset_v3/diffusion_v2/diffusion_test_v2.npz")
    parser.add_argument("--norm_stats", default="data/cartesian_expert_dataset_v3/diffusion_v2/diffusion_norm_stats_v2.json")
    parser.add_argument("--dataset_dir", default="data/cartesian_expert_dataset_v3/experts/test")
    parser.add_argument("--output_dir", default="data/cartesian_expert_dataset_v3/diffusion_v2/ddim_samples")
    parser.add_argument("--max_paths", type=int, default=83)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--num_sampling_steps", type=int, default=50)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--clip_x0", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--clip_value", type=float, default=3.0)
    parser.add_argument("--rank_by_cost", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--w_path", type=float, default=1.0)
    parser.add_argument("--w_x", type=float, default=1.0)
    parser.add_argument("--w_y", type=float, default=1.0)
    parser.add_argument("--w_z", type=float, default=2.0)
    parser.add_argument("--w_vel", type=float, default=0.0)
    parser.add_argument("--w_accel", type=float, default=0.01)
    parser.add_argument("--ee_link", default=None)
    return parser.parse_args()


def select_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def make_ddim_timesteps(num_diffusion_steps: int, num_sampling_steps: int) -> list[int]:
    if num_sampling_steps <= 0:
        raise ValueError("--num_sampling_steps must be positive")
    if num_sampling_steps > num_diffusion_steps:
        num_sampling_steps = num_diffusion_steps

    timesteps = np.linspace(num_diffusion_steps - 1, 0, num_sampling_steps)
    timesteps = np.rint(timesteps).astype(int).tolist()

    deduped: list[int] = []
    for timestep in timesteps:
        if not deduped or deduped[-1] != timestep:
            deduped.append(timestep)
    if deduped[-1] != 0:
        deduped.append(0)
    return deduped


def ddim_sample(
    model: ConditionalTrajectoryDenoiser,
    condition: torch.Tensor,
    alpha_bars: torch.Tensor,
    timesteps: list[int],
    eta: float,
    clip_x0: bool,
    clip_value: float,
    device: torch.device,
) -> torch.Tensor:
    x = torch.randn((1, condition.shape[1], 6), device=device)

    with torch.no_grad():
        for step_index, timestep in enumerate(timesteps):
            t = torch.full((1,), timestep, device=device, dtype=torch.long)
            eps_pred = model(x, condition, t)

            alpha_bar_t = alpha_bars[timestep]
            sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
            sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - alpha_bar_t)
            x0_pred = (x - sqrt_one_minus_alpha_bar_t * eps_pred) / sqrt_alpha_bar_t

            if clip_x0:
                x0_pred = torch.clamp(x0_pred, -clip_value, clip_value)

            t_prev = timesteps[step_index + 1] if step_index + 1 < len(timesteps) else -1
            if t_prev < 0:
                x = x0_pred
                continue

            alpha_bar_prev = alpha_bars[t_prev]
            sigma_t = eta * torch.sqrt((1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t))
            sigma_t = sigma_t * torch.sqrt(1.0 - alpha_bar_t / alpha_bar_prev)
            direction_scale = torch.clamp(1.0 - alpha_bar_prev - sigma_t * sigma_t, min=0.0)
            direction = torch.sqrt(direction_scale) * eps_pred
            x = torch.sqrt(alpha_bar_prev) * x0_pred + direction

            if eta > 0.0:
                x = x + sigma_t * torch.randn_like(x)

    return x.squeeze(0).cpu()


def denormalize_delta_q(delta_q_norm: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return delta_q_norm * std[None, :] + mean[None, :]


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


def load_times(dataset_dir: Path, path_name: str, num_steps: int) -> np.ndarray:
    desired_csv = dataset_dir / path_name / "desired_path.csv"
    if desired_csv.exists():
        try:
            df = pd.read_csv(desired_csv)
            if "t" in df.columns and len(df) == num_steps:
                return df["t"].to_numpy(dtype=np.float64)
        except Exception as exc:
            print(f"WARNING: could not read time column from {desired_csv}: {exc}")
    return np.linspace(0.0, 1.0, num_steps, dtype=np.float64)


def save_q_csv(path: Path, times: np.ndarray, q: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(q, columns=Q_COLUMNS)
    df.insert(0, "t", times)
    df.to_csv(path, index=False)


def save_ee_csv(path: Path, times: np.ndarray, ee: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(ee, columns=XYZ_COLUMNS)
    df.insert(0, "t", times)
    df.to_csv(path, index=False)


def save_desired_csv(path: Path, times: np.ndarray, desired: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(desired, columns=XYZ_COLUMNS)
    df.insert(0, "t", times)
    df.to_csv(path, index=False)


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

    plt.plot(pred[:, 0], pred[:, 1], label="diffusion DDIM FK path", linewidth=2)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.legend()
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=160)
    plt.close()


def compute_metrics(
    desired: np.ndarray,
    pred_ee: np.ndarray,
    q_pred: np.ndarray,
    args: argparse.Namespace,
) -> dict:
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
    return metrics


def save_sample_outputs(
    folder: Path,
    times: np.ndarray,
    desired: np.ndarray,
    expert_q: np.ndarray,
    q_pred: np.ndarray,
    pred_ee: np.ndarray,
    metrics: dict,
    expert_ee_csv: Path,
) -> None:
    metric_keys = [
        "path_error",
        "mean_error",
        "max_error",
        "loss_x",
        "loss_y",
        "loss_z",
        "weighted_xyz_loss",
        "joint_velocity_cost",
        "joint_acceleration_cost",
        "total_cost",
        "accepted",
    ]
    metrics_json = {key: metrics[key] for key in metric_keys if key in metrics}
    folder.mkdir(parents=True, exist_ok=True)
    save_desired_csv(folder / "desired_path.csv", times, desired)
    save_q_csv(folder / "expert_q.csv", times, expert_q)
    save_q_csv(folder / "diffusion_pred_q.csv", times, q_pred)
    save_ee_csv(folder / "diffusion_pred_ee.csv", times, pred_ee)
    with (folder / "diffusion_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics_json, f, indent=2)
        f.write("\n")
    save_overlay(folder / "diffusion_overlay.png", desired, pred_ee, expert_ee_csv)


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = select_device(args.device)
    data = load_required_npz(Path(args.test_npz))
    delta_q_mean, delta_q_std = load_norm_stats(Path(args.norm_stats))
    model, num_diffusion_steps = load_model(Path(args.model), device)
    _, _, alpha_bars = make_beta_schedule(num_diffusion_steps, device)
    timesteps = make_ddim_timesteps(num_diffusion_steps, args.num_sampling_steps)

    robot = load_robot()
    ee_link = infer_ee_link(robot, args.ee_link)
    joint_names = infer_joint_names(robot)

    output_dir = Path(args.output_dir)
    dataset_dir = Path(args.dataset_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    condition_all = data["condition_features_norm"].astype(np.float32)
    desired_all = data["desired_paths"].astype(np.float32)
    expert_q_all = data["expert_q"].astype(np.float32)
    q_start_all = data["q_start"].astype(np.float32)
    path_names = [str(name) for name in data["path_names"]]
    max_paths = min(args.max_paths, len(path_names)) if args.max_paths > 0 else len(path_names)

    summary_rows = []

    for path_index in range(max_paths):
        path_name = path_names[path_index]
        condition_np = condition_all[path_index]
        desired = desired_all[path_index]
        expert_q = expert_q_all[path_index]
        q_start = q_start_all[path_index]
        times = load_times(dataset_dir, path_name, desired.shape[0])
        expert_ee_csv = dataset_dir / path_name / "expert_ee.csv"
        path_results = []

        try:
            condition = torch.from_numpy(condition_np).unsqueeze(0).to(device)
        except Exception as exc:
            print(f"WARNING: failed to prepare condition for {path_name}: {exc}")
            continue

        for sample_index in range(args.num_samples):
            sample_folder = output_dir / path_name / f"sample_{sample_index:03d}"
            try:
                delta_q_norm = ddim_sample(
                    model=model,
                    condition=condition,
                    alpha_bars=alpha_bars,
                    timesteps=timesteps,
                    eta=args.eta,
                    clip_x0=args.clip_x0,
                    clip_value=args.clip_value,
                    device=device,
                ).numpy()
                delta_q_pred = denormalize_delta_q(delta_q_norm, delta_q_mean, delta_q_std)
                q_pred = q_start[None, :] + delta_q_pred
                pred_ee = fk_trajectory(robot, q_pred, ee_link, joint_names)
                metrics = compute_metrics(desired, pred_ee, q_pred, args)

                save_sample_outputs(
                    sample_folder,
                    times,
                    desired,
                    expert_q,
                    q_pred,
                    pred_ee,
                    metrics,
                    expert_ee_csv,
                )

                result = {
                    "path_name": path_name,
                    "sample_index": sample_index,
                    "best_for_path": False,
                    **metrics,
                    "num_sampling_steps": len(timesteps),
                    "eta": args.eta,
                    "clip_x0": args.clip_x0,
                    "clip_value": args.clip_value,
                    "output_folder": str(sample_folder),
                    "times": times,
                    "desired": desired,
                    "expert_q": expert_q,
                    "q_pred": q_pred,
                    "pred_ee": pred_ee,
                    "expert_ee_csv": expert_ee_csv,
                }
                path_results.append(result)
            except Exception as exc:
                print(f"WARNING: failed {path_name} sample {sample_index}: {exc}")
                continue

        if not path_results:
            continue

        if args.rank_by_cost:
            best_index = min(range(len(path_results)), key=lambda idx: path_results[idx]["total_cost"])
        else:
            best_index = 0

        for result_index, result in enumerate(path_results):
            result["best_for_path"] = result_index == best_index
            if result["best_for_path"]:
                save_sample_outputs(
                    output_dir / path_name / "best",
                    result["times"],
                    result["desired"],
                    result["expert_q"],
                    result["q_pred"],
                    result["pred_ee"],
                    result,
                    result["expert_ee_csv"],
                )

            summary_rows.append(
                {
                    "path_name": result["path_name"],
                    "sample_index": result["sample_index"],
                    "best_for_path": result["best_for_path"],
                    "path_error": result["path_error"],
                    "mean_error": result["mean_error"],
                    "max_error": result["max_error"],
                    "loss_x": result["loss_x"],
                    "loss_y": result["loss_y"],
                    "loss_z": result["loss_z"],
                    "weighted_xyz_loss": result["weighted_xyz_loss"],
                    "joint_velocity_cost": result["joint_velocity_cost"],
                    "joint_acceleration_cost": result["joint_acceleration_cost"],
                    "total_cost": result["total_cost"],
                    "accepted": result["accepted"],
                    "num_sampling_steps": result["num_sampling_steps"],
                    "eta": result["eta"],
                    "clip_x0": result["clip_x0"],
                    "clip_value": result["clip_value"],
                    "output_folder": result["output_folder"],
                }
            )

    summary_path = output_dir / "diffusion_v2_ddim_sample_summary.csv"
    pd.DataFrame(
        summary_rows,
        columns=[
            "path_name",
            "sample_index",
            "best_for_path",
            "path_error",
            "mean_error",
            "max_error",
            "loss_x",
            "loss_y",
            "loss_z",
            "weighted_xyz_loss",
            "joint_velocity_cost",
            "joint_acceleration_cost",
            "total_cost",
            "accepted",
            "num_sampling_steps",
            "eta",
            "clip_x0",
            "clip_value",
            "output_folder",
        ],
    ).to_csv(summary_path, index=False)
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
