#!/usr/bin/env python3
"""
Sample joint trajectories from a trained conditional trajectory DDPM.

This is diffusion-only generation: no IK refinement is applied.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from yourdfpy import URDF

from score_trajectory import (
    ROKAE_EE_LINK,
    ROKAE_JOINT_NAMES,
    URDF_PATH,
    compute_path_error,
    compute_smoothness_cost,
)
from train_conditional_diffusion_trajectory import (
    ConditionalTrajectoryDenoiser,
    DDPMSchedule,
    resolve_device,
    set_seed,
)


Q_COLS = ["q1", "q2", "q3", "q4", "q5", "q6"]
PATH_COLS = ["x", "y", "z"]
NUM_STEPS = 100


def load_test_npz(npz_path: Path) -> Dict[str, np.ndarray]:
    data = np.load(npz_path, allow_pickle=True)
    required = [
        "desired_paths",
        "desired_paths_norm",
        "expert_q",
        "expert_q_norm",
        "path_names",
    ]
    missing = [key for key in required if key not in data.files]
    if missing:
        raise KeyError(f"Missing keys in {npz_path}: {missing}. Available: {data.files}")
    return {key: data[key] for key in required}


def load_norm_stats(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8") as f:
        stats = json.load(f)

    q_mean = np.asarray(stats["q_mean"], dtype=np.float32)
    q_std = np.asarray(stats["q_std"], dtype=np.float32)
    if q_mean.shape != (6,) or q_std.shape != (6,):
        raise ValueError(f"Expected q_mean/q_std shape (6,), got {q_mean.shape}/{q_std.shape}")
    return q_mean, q_std


def load_checkpoint_model(
    model_path: Path,
    device: torch.device,
) -> Tuple[ConditionalTrajectoryDenoiser, DDPMSchedule, Dict[str, object]]:
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    hidden_dim = int(checkpoint.get("hidden_dim", 256))
    num_diffusion_steps = int(checkpoint.get("num_diffusion_steps", 100))

    model = ConditionalTrajectoryDenoiser(hidden_dim=hidden_dim).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    schedule = DDPMSchedule(num_diffusion_steps).to(device)
    return model, schedule, checkpoint


def reverse_ddpm_sample(
    model: ConditionalTrajectoryDenoiser,
    schedule: DDPMSchedule,
    condition: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    x = torch.randn((condition.shape[0], NUM_STEPS, 6), device=device)

    with torch.no_grad():
        for step in reversed(range(schedule.num_diffusion_steps)):
            timesteps = torch.full(
                (condition.shape[0],),
                step,
                dtype=torch.long,
                device=device,
            )
            pred_noise = model(x, condition, timesteps)

            beta_t = schedule.beta[step].view(1, 1, 1)
            alpha_t = schedule.alpha[step].view(1, 1, 1)
            sqrt_one_minus_alpha_bar_t = schedule.sqrt_one_minus_alpha_bar[step].view(
                1, 1, 1
            )

            mean = (x - beta_t * pred_noise / sqrt_one_minus_alpha_bar_t) / torch.sqrt(
                alpha_t
            )

            if step > 0:
                noise = torch.randn_like(x)
                x = mean + torch.sqrt(beta_t) * noise
            else:
                x = mean

    return x


def denormalize_q(q_norm: np.ndarray, q_mean: np.ndarray, q_std: np.ndarray) -> np.ndarray:
    return q_norm * q_std.reshape(1, 6) + q_mean.reshape(1, 6)


def load_times_from_dataset(dataset_dir: Path, path_name: str) -> np.ndarray:
    desired_csv = dataset_dir / path_name / "desired_path.csv"
    if desired_csv.exists():
        try:
            df = pd.read_csv(desired_csv)
            if "t" in df.columns and len(df) == NUM_STEPS:
                return df["t"].to_numpy(dtype=np.float32)
        except Exception as exc:
            print(f"[WARN] Could not read original times for {path_name}: {exc}")
    return np.linspace(0.0, 1.0, NUM_STEPS, dtype=np.float32)


def save_path_csv(path: Path, times: np.ndarray, xyz: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "t": times,
            "x": xyz[:, 0],
            "y": xyz[:, 1],
            "z": xyz[:, 2],
        }
    )
    df.to_csv(path, index=False)


def save_q_csv(path: Path, times: np.ndarray, q: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        np.concatenate([times.reshape(-1, 1), q], axis=1),
        columns=["t"] + Q_COLS,
    )
    df.to_csv(path, index=False)


def rokae_forward_kinematics(robot: URDF, q: np.ndarray, ee_link: str = ROKAE_EE_LINK) -> np.ndarray:
    if q.shape[1] != len(ROKAE_JOINT_NAMES):
        raise ValueError(f"Expected {len(ROKAE_JOINT_NAMES)} joints, got {q.shape[1]}")

    positions = []
    for q_t in q:
        cfg = {
            joint_name: float(joint_value)
            for joint_name, joint_value in zip(ROKAE_JOINT_NAMES, q_t)
        }
        robot.update_cfg(cfg)
        transform = robot.get_transform(frame_to=ee_link)
        positions.append(transform[:3, 3])

    return np.asarray(positions, dtype=np.float32)


def compute_metrics(pred_ee: np.ndarray, desired_path: np.ndarray, q: np.ndarray) -> Dict[str, object]:
    if pred_ee.shape != desired_path.shape:
        raise ValueError(f"Shape mismatch: pred_ee {pred_ee.shape}, desired {desired_path.shape}")

    euclidean_error = np.linalg.norm(pred_ee - desired_path, axis=1)
    path_error = compute_path_error(pred_ee, desired_path)
    mean_error = float(np.mean(euclidean_error))
    max_error = float(np.max(euclidean_error))
    smoothness = compute_smoothness_cost(q)
    accepted = bool(mean_error <= 0.010 and max_error <= 0.030)

    return {
        "path_error": float(path_error),
        "mean_error": mean_error,
        "max_error": max_error,
        "smoothness": float(smoothness),
        "accepted": accepted,
    }


def save_metrics_json(path: Path, metrics: Dict[str, object]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
        f.write("\n")


def save_overlay_plot(
    path: Path,
    desired_path: np.ndarray,
    pred_ee: np.ndarray,
    expert_ee_csv: Path,
) -> None:
    plt.figure(figsize=(6, 6))
    plt.plot(desired_path[:, 0], desired_path[:, 1], label="desired path", linewidth=2)

    if expert_ee_csv.exists():
        try:
            expert_ee = pd.read_csv(expert_ee_csv)
            if all(col in expert_ee.columns for col in PATH_COLS):
                plt.plot(
                    expert_ee["x"],
                    expert_ee["y"],
                    label="expert IK FK path",
                    linewidth=1.5,
                )
            else:
                print(f"[WARN] {expert_ee_csv} missing x,y,z columns; skipping expert overlay")
        except Exception as exc:
            print(f"[WARN] Could not read expert EE path {expert_ee_csv}: {exc}")
    else:
        print(f"[WARN] Missing expert_ee.csv for overlay: {expert_ee_csv}")

    plt.plot(pred_ee[:, 0], pred_ee[:, 1], label="diffusion predicted FK path", linewidth=1.5)
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.axis("equal")
    plt.legend()
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=200)
    plt.close()


def selected_indices(num_paths: int, max_paths: int) -> range:
    if max_paths <= 0:
        return range(num_paths)
    return range(min(num_paths, max_paths))


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Sample diffusion-only joint trajectories conditioned on test paths."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("data/cartesian_expert_dataset_v3/conditional_diffusion_v1.pt"),
    )
    parser.add_argument(
        "--test_npz",
        type=Path,
        default=Path("data/cartesian_expert_dataset_v3/diffusion_test.npz"),
    )
    parser.add_argument(
        "--norm_stats",
        type=Path,
        default=Path("data/cartesian_expert_dataset_v3/diffusion_norm_stats.json"),
    )
    parser.add_argument(
        "--dataset_dir",
        type=Path,
        default=Path("data/cartesian_expert_dataset_v3/experts/test"),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/cartesian_expert_dataset_v3/diffusion_v1_samples"),
    )
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--max_paths", type=int, default=10)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    data = load_test_npz(args.test_npz)
    q_mean, q_std = load_norm_stats(args.norm_stats)
    model, schedule, _ = load_checkpoint_model(args.model, device)

    robot = URDF.load(str(Path(URDF_PATH)), load_meshes=False)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = args.output_dir / "diffusion_sample_summary.csv"

    print(f"Model: {args.model}")
    print(f"Test NPZ: {args.test_npz}")
    print(f"Output dir: {args.output_dir}")
    print(f"Device: {device}")

    summary_rows: List[Dict[str, object]] = []
    path_names = [str(name) for name in data["path_names"]]

    for idx in selected_indices(len(path_names), args.max_paths):
        path_name = path_names[idx]
        desired_path = data["desired_paths"][idx].astype(np.float32)
        expert_q = data["expert_q"][idx].astype(np.float32)
        condition_np = data["desired_paths_norm"][idx].astype(np.float32)
        times = load_times_from_dataset(args.dataset_dir, path_name)

        for sample_index in range(args.num_samples):
            sample_folder = args.output_dir / path_name
            if args.num_samples > 1:
                sample_folder = sample_folder / f"sample_{sample_index:03d}"

            try:
                condition = torch.from_numpy(condition_np).unsqueeze(0).to(device)
                pred_q_norm = reverse_ddpm_sample(model, schedule, condition, device)
                pred_q_norm_np = pred_q_norm.squeeze(0).cpu().numpy().astype(np.float32)
                pred_q = denormalize_q(pred_q_norm_np, q_mean, q_std)

                pred_ee = rokae_forward_kinematics(robot, pred_q)
                metrics = compute_metrics(pred_ee, desired_path, pred_q)

                save_path_csv(sample_folder / "desired_path.csv", times, desired_path)
                save_q_csv(sample_folder / "expert_q.csv", times, expert_q)
                save_q_csv(sample_folder / "diffusion_pred_q.csv", times, pred_q)
                save_path_csv(sample_folder / "diffusion_pred_ee.csv", times, pred_ee)
                save_metrics_json(sample_folder / "diffusion_metrics.json", metrics)
                save_overlay_plot(
                    sample_folder / "diffusion_overlay.png",
                    desired_path,
                    pred_ee,
                    args.dataset_dir / path_name / "expert_ee.csv",
                )

                summary_rows.append(
                    {
                        "path_name": path_name,
                        "sample_index": sample_index,
                        "path_error": metrics["path_error"],
                        "mean_error": metrics["mean_error"],
                        "max_error": metrics["max_error"],
                        "smoothness": metrics["smoothness"],
                        "accepted": metrics["accepted"],
                        "output_folder": str(sample_folder),
                    }
                )
                print(
                    f"[OK] {path_name} sample {sample_index}: "
                    f"mean_error={metrics['mean_error']:.6f}, "
                    f"max_error={metrics['max_error']:.6f}, "
                    f"accepted={metrics['accepted']}"
                )
            except Exception as exc:
                print(f"[WARN] Failed {path_name} sample {sample_index}: {exc}")

    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "path_name",
            "sample_index",
            "path_error",
            "mean_error",
            "max_error",
            "smoothness",
            "accepted",
            "output_folder",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    print()
    print(f"Saved summary: {summary_csv}")
    print(f"Completed samples: {len(summary_rows)}")


if __name__ == "__main__":
    main()
