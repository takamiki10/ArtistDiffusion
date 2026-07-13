#!/usr/bin/env python3
"""Generate best-of-K ranked v1 diffusion trajectory candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from train_conditional_diffusion_trajectory import (
    ConditionalTrajectoryDenoiser,
    DDPMSchedule,
    resolve_device,
    set_seed,
)


Q_COLUMNS = [f"q{i}" for i in range(1, 7)]
XYZ_COLUMNS = ["x", "y", "z"]
NUM_TRAJECTORY_STEPS = 100


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="data/cartesian_expert_dataset_v3/conditional_diffusion_v1.pt")
    parser.add_argument("--test_npz", default="data/cartesian_expert_dataset_v3/diffusion_test.npz")
    parser.add_argument("--norm_stats", default="data/cartesian_expert_dataset_v3/diffusion_norm_stats.json")
    parser.add_argument("--dataset_dir", default="data/cartesian_expert_dataset_v3/experts/test")
    parser.add_argument("--output_dir", default="data/cartesian_expert_dataset_v3/diffusion_v1_ranked_samples/k8")
    parser.add_argument("--max_paths", type=int, default=83)
    parser.add_argument("--num_samples", type=int, default=8)
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


def load_required_npz(path: Path) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True)
    required = ["desired_paths", "expert_q", "path_names", "desired_paths_norm", "expert_q_norm"]
    missing = [key for key in required if key not in data.files]
    if missing:
        raise KeyError(f"Missing keys in {path}: {missing}. Available keys: {data.files}")
    return {key: data[key] for key in required}


def load_norm_stats(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8") as f:
        stats = json.load(f)
    q_mean = np.asarray(stats["q_mean"], dtype=np.float32)
    q_std = np.asarray(stats["q_std"], dtype=np.float32)
    if q_mean.shape != (6,) or q_std.shape != (6,):
        raise ValueError(f"Expected q_mean/q_std shape (6,), got {q_mean.shape}/{q_std.shape}")
    return q_mean, q_std


def load_model(model_path: Path, device: torch.device) -> tuple[ConditionalTrajectoryDenoiser, DDPMSchedule]:
    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(model_path, map_location=device)

    hidden_dim = int(checkpoint.get("hidden_dim", 256))
    num_diffusion_steps = int(checkpoint.get("num_diffusion_steps", 100))
    model = ConditionalTrajectoryDenoiser(hidden_dim=hidden_dim).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    schedule = DDPMSchedule(num_diffusion_steps).to(device)
    return model, schedule


def reverse_ddpm_sample(
    model: ConditionalTrajectoryDenoiser,
    schedule: DDPMSchedule,
    condition: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    x = torch.randn((condition.shape[0], NUM_TRAJECTORY_STEPS, 6), device=device)

    with torch.no_grad():
        for step in reversed(range(schedule.num_diffusion_steps)):
            timesteps = torch.full((condition.shape[0],), step, dtype=torch.long, device=device)
            pred_noise = model(x, condition, timesteps)

            beta_t = schedule.beta[step].view(1, 1, 1)
            alpha_t = schedule.alpha[step].view(1, 1, 1)
            sqrt_one_minus_alpha_bar_t = schedule.sqrt_one_minus_alpha_bar[step].view(1, 1, 1)
            mean = (x - beta_t * pred_noise / sqrt_one_minus_alpha_bar_t) / torch.sqrt(alpha_t)

            if step > 0:
                x = mean + torch.sqrt(beta_t) * torch.randn_like(x)
            else:
                x = mean

    return x


def denormalize_q(q_norm: np.ndarray, q_mean: np.ndarray, q_std: np.ndarray) -> np.ndarray:
    return q_norm * q_std.reshape(1, 6) + q_mean.reshape(1, 6)


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

    plt.plot(pred[:, 0], pred[:, 1], label="diffusion predicted FK path", linewidth=2)
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
    try:
        from trajectory_costs import compute_stanford_style_trajectory_cost

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
    except Exception:
        errors = pred_ee - desired
        loss_x = float(np.mean(errors[:, 0] ** 2))
        loss_y = float(np.mean(errors[:, 1] ** 2))
        loss_z = float(np.mean(errors[:, 2] ** 2))
        weighted_xyz_loss = args.w_x * loss_x + args.w_y * loss_y + args.w_z * loss_z
        euclidean_error = np.linalg.norm(errors, axis=1)
        if q_pred.shape[0] > 1:
            velocity = np.diff(q_pred, axis=0)
            joint_velocity_cost = float(np.mean(np.sum(velocity * velocity, axis=1)))
        else:
            joint_velocity_cost = 0.0
        if q_pred.shape[0] > 2:
            accel = q_pred[2:] - 2.0 * q_pred[1:-1] + q_pred[:-2]
            joint_acceleration_cost = float(np.mean(np.sum(accel * accel, axis=1)))
        else:
            joint_acceleration_cost = 0.0
        path_error = float(np.mean(euclidean_error * euclidean_error))
        metrics = {
            "path_error": path_error,
            "mean_error": float(np.mean(euclidean_error)),
            "max_error": float(np.max(euclidean_error)),
            "loss_x": loss_x,
            "loss_y": loss_y,
            "loss_z": loss_z,
            "weighted_xyz_loss": weighted_xyz_loss,
            "joint_velocity_cost": joint_velocity_cost,
            "joint_acceleration_cost": joint_acceleration_cost,
            "total_cost": args.w_path * path_error
            + weighted_xyz_loss
            + args.w_vel * joint_velocity_cost
            + args.w_accel * joint_acceleration_cost,
        }

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


def summary_row(result: dict) -> dict:
    return {
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
        "output_folder": result["output_folder"],
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    output_dir = Path(args.output_dir)
    dataset_dir = Path(args.dataset_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_required_npz(Path(args.test_npz))
    q_mean, q_std = load_norm_stats(Path(args.norm_stats))
    model, schedule = load_model(Path(args.model), device)
    robot = load_robot()
    ee_link = infer_ee_link(robot, args.ee_link)
    joint_names = infer_joint_names(robot)

    desired_all = data["desired_paths"].astype(np.float32)
    expert_q_all = data["expert_q"].astype(np.float32)
    condition_all = data["desired_paths_norm"].astype(np.float32)
    path_names = [str(name) for name in data["path_names"]]
    max_paths = min(args.max_paths, len(path_names)) if args.max_paths > 0 else len(path_names)

    all_rows = []
    best_rows = []

    for path_index in range(max_paths):
        path_name = path_names[path_index]
        desired = desired_all[path_index]
        expert_q = expert_q_all[path_index]
        condition_np = condition_all[path_index]
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
                q_norm = reverse_ddpm_sample(model, schedule, condition, device).squeeze(0).cpu().numpy().astype(np.float32)
                q_pred = denormalize_q(q_norm, q_mean, q_std)
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

                path_results.append(
                    {
                        "path_name": path_name,
                        "sample_index": sample_index,
                        "best_for_path": False,
                        **metrics,
                        "output_folder": str(sample_folder),
                        "times": times,
                        "desired": desired,
                        "expert_q": expert_q,
                        "q_pred": q_pred,
                        "pred_ee": pred_ee,
                        "expert_ee_csv": expert_ee_csv,
                    }
                )
            except Exception as exc:
                print(f"WARNING: failed {path_name} sample {sample_index}: {exc}")
                continue

        if not path_results:
            continue

        best_index = min(range(len(path_results)), key=lambda idx: path_results[idx]["total_cost"])
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
                best_rows.append(summary_row(result))
            all_rows.append(summary_row(result))

    columns = [
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
        "output_folder",
    ]
    all_candidates_path = output_dir / "diffusion_v1_all_candidates.csv"
    best_per_path_path = output_dir / "diffusion_v1_best_per_path.csv"
    all_df = pd.DataFrame(all_rows, columns=columns)
    best_df = pd.DataFrame(best_rows, columns=columns)
    all_df.to_csv(all_candidates_path, index=False)
    best_df.to_csv(best_per_path_path, index=False)

    total_candidates = len(all_df)
    total_paths = len(best_df)
    accepted_candidates = int(all_df["accepted"].sum()) if total_candidates else 0
    accepted_best = int(best_df["accepted"].sum()) if total_paths else 0
    best_mean_error = float(best_df["mean_error"].mean()) if total_paths else float("nan")
    best_max_error = float(best_df["max_error"].mean()) if total_paths else float("nan")
    best_worst_max_error = float(best_df["max_error"].max()) if total_paths else float("nan")
    best_total_cost = float(best_df["total_cost"].mean()) if total_paths else float("nan")

    print(f"All-candidates CSV: {all_candidates_path}")
    print(f"Best-per-path CSV: {best_per_path_path}")
    print(f"Number of paths evaluated: {total_paths}")
    print(f"Total samples generated: {total_candidates}")
    print(f"Accepted candidates / total candidates: {accepted_candidates} / {total_candidates}")
    print(f"Accepted best paths / total paths: {accepted_best} / {total_paths}")
    print(f"Best-path mean_error mean: {best_mean_error:.6f}")
    print(f"Best-path max_error mean: {best_max_error:.6f}")
    print(f"Best-path worst max_error: {best_worst_max_error:.6f}")
    print(f"Best-path total_cost mean: {best_total_cost:.6f}")


if __name__ == "__main__":
    main()
