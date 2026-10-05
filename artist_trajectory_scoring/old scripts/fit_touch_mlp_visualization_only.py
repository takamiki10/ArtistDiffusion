#!/usr/bin/env python3
"""Overfit a path-conditioned MLP to one deployment trajectory for visualization.

This utility intentionally trains and evaluates on the same IK-refined trajectory.
Its output is data-leaked and must not be used as an independent MLP benchmark.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch

from generate_ik_seed_path import DEFAULT_EE_LINK, DEFAULT_JOINT_NAMES, load_robot
from orientation_aware_adaptive_ik import trajectory_full_transform_fk
from predict_path_conditioned_mlp import PathConditionedMLP, make_features


DISCLAIMER = (
    "Visualization-only MLP trained and evaluated on the same IK-refined "
    "trajectory. This artifact contains target leakage and is not an "
    "independent MLP baseline."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_npz", type=Path, required=True)
    parser.add_argument("--diffusion_npz", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=4000)
    parser.add_argument("--learning_rate", type=float, default=1.0e-3)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--fourier_frequencies", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="auto"
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def finite_array(data: Any, key: str, shape: tuple[int, ...]) -> np.ndarray:
    if key not in data:
        raise KeyError(f"Missing required NPZ array: {key}")
    array = np.asarray(data[key], dtype=np.float64)
    if array.shape != shape:
        raise ValueError(f"{key} must have shape {shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{key} contains nonfinite values")
    return array


def scalar_text(data: Any, key: str) -> str:
    if key not in data:
        raise KeyError(f"Missing required NPZ field: {key}")
    return str(np.asarray(data[key]).item())


def safe_scale(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = np.mean(values, axis=0, keepdims=True)
    std = np.std(values, axis=0, keepdims=True)
    std = np.where(std > 1.0e-8, std, 1.0)
    return mean.astype(np.float32), std.astype(np.float32)


def write_csv(path: Path, header: list[str], values: np.ndarray) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(values.tolist())


def main() -> int:
    args = parse_args()
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.learning_rate <= 0.0:
        raise ValueError("--learning_rate must be positive")
    if args.hidden_dim <= 0:
        raise ValueError("--hidden_dim must be positive")
    if args.num_layers < 2:
        raise ValueError("--num_layers must be at least 2")
    if args.fourier_frequencies <= 0:
        raise ValueError("--fourier_frequencies must be positive")
    if not args.input_npz.is_file():
        raise FileNotFoundError(args.input_npz)
    if not args.diffusion_npz.is_file():
        raise FileNotFoundError(args.diffusion_npz)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    expected_outputs = (
        args.output_dir / "visualization_only_mlp_checkpoint.pt",
        args.output_dir / "visualization_only_mlp_predictions.npz",
        args.output_dir / "visualization_only_mlp_q.csv",
        args.output_dir / "visualization_only_mlp_ee.csv",
        args.output_dir / "visualization_only_fit_metrics.json",
        args.output_dir / "target_mlp_fitted_ik_diffusion_xy.png",
        args.output_dir / "tracking_error_over_time.png",
    )
    existing = [path for path in expected_outputs if path.exists()]
    if existing and not args.overwrite:
        names = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Outputs already exist; pass --overwrite: {names}")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.set_num_threads(1)

    with np.load(args.input_npz, allow_pickle=True) as source:
        desired = finite_array(source, "desired_path", (100, 3))
        ik_q = finite_array(source, "strong_prior_q", (100, 6))
        ik_ee = finite_array(source, "strong_prior_ee", (100, 3))
        timestamps = finite_array(source, "timestamps", (100,))
        path_name = scalar_text(source, "path_name")
        urdf_path = Path(scalar_text(source, "urdf_path"))

    with np.load(args.diffusion_npz, allow_pickle=True) as result:
        diffusion_ee = finite_array(result, "final_ee", (100, 3))

    time_span = float(timestamps[-1] - timestamps[0])
    if time_span <= 0.0:
        raise ValueError("timestamps must have a positive span")
    normalized_t = ((timestamps - timestamps[0]) / time_span).astype(np.float32)

    feature_contract = {
        "num_steps": 100,
        "include_current_point": True,
    }
    raw_features = make_features(
        desired.astype(np.float32), normalized_t, feature_contract
    )
    x_mean, x_std = safe_scale(raw_features)
    y_mean, y_std = safe_scale(ik_q.astype(np.float32))
    base_x_norm = ((raw_features - x_mean) / x_std).astype(np.float32)
    fourier_parts = []
    for frequency in range(1, args.fourier_frequencies + 1):
        angle = 2.0 * np.pi * float(frequency) * normalized_t
        fourier_parts.append(np.sin(angle)[:, None])
        fourier_parts.append(np.cos(angle)[:, None])
    fourier_features = np.concatenate(fourier_parts, axis=1).astype(np.float32)
    x_norm = np.concatenate((base_x_norm, fourier_features), axis=1)
    y_norm = ((ik_q.astype(np.float32) - y_mean) / y_std).astype(np.float32)

    model = PathConditionedMLP(
        input_dim=x_norm.shape[1],
        output_dim=6,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
    ).to(device)
    x_tensor = torch.from_numpy(x_norm).to(device)
    y_tensor = torch.from_numpy(y_norm).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    model.train()
    final_adam_loss = float("nan")
    completed_epochs = 0
    for epoch in range(1, args.epochs + 1):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(x_tensor)
        loss = torch.mean((prediction - y_tensor) ** 2)
        loss.backward()
        optimizer.step()
        final_adam_loss = float(loss.detach().cpu())
        completed_epochs = epoch
        if final_adam_loss < 1.0e-10:
            break

    # Only invoke a short full-batch polish if Adam did not already reach the
    # same-data visualization tolerance.
    if final_adam_loss >= 1.0e-9:
        lbfgs = torch.optim.LBFGS(
            model.parameters(),
            lr=0.5,
            max_iter=100,
            tolerance_grad=1.0e-12,
            tolerance_change=1.0e-14,
            line_search_fn="strong_wolfe",
        )

        def closure() -> torch.Tensor:
            lbfgs.zero_grad(set_to_none=True)
            prediction = model(x_tensor)
            loss = torch.mean((prediction - y_tensor) ** 2)
            loss.backward()
            return loss

        lbfgs.step(closure)
    model.eval()
    with torch.no_grad():
        predicted_norm = model(x_tensor).cpu().numpy()
    predicted_q = predicted_norm * y_std + y_mean
    predicted_q = predicted_q.astype(np.float64)

    robot = load_robot(urdf_path)
    predicted_ee, _, _ = trajectory_full_transform_fk(
        robot,
        predicted_q,
        tuple(DEFAULT_JOINT_NAMES),
        DEFAULT_EE_LINK,
    )
    predicted_ee = np.asarray(predicted_ee, dtype=np.float64)

    q_error = predicted_q - ik_q
    cartesian_error = np.linalg.norm(predicted_ee - desired, axis=1)
    ik_cartesian_error = np.linalg.norm(ik_ee - desired, axis=1)
    final_training_loss = float(np.mean((predicted_norm - y_norm) ** 2))
    metrics = {
        "artifact_purpose": "visualization_only",
        "data_leakage": True,
        "independent_benchmark_eligible": False,
        "disclaimer": DISCLAIMER,
        "path_name": path_name,
        "training_source": str(args.input_npz.resolve()),
        "diffusion_source": str(args.diffusion_npz.resolve()),
        "training_target": "strong_prior_q (IK-refined trajectory)",
        "training_samples": 100,
        "completed_adam_epochs": completed_epochs,
        "final_adam_normalized_mse": final_adam_loss,
        "final_normalized_mse": final_training_loss,
        "q_rmse_vs_ik_rad": float(np.sqrt(np.mean(q_error**2))),
        "maximum_absolute_q_error_vs_ik_rad": float(np.max(np.abs(q_error))),
        "mean_cartesian_error_vs_target_m": float(np.mean(cartesian_error)),
        "maximum_cartesian_error_vs_target_m": float(np.max(cartesian_error)),
        "ik_mean_cartesian_error_vs_target_m": float(
            np.mean(ik_cartesian_error)
        ),
        "ik_maximum_cartesian_error_vs_target_m": float(
            np.max(ik_cartesian_error)
        ),
        "seed": args.seed,
        "device": str(device),
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "fourier_frequencies": args.fourier_frequencies,
    }

    checkpoint = {
        "model_type": "path_conditioned_mlp_visualization_only_overfit",
        "model_state_dict": model.state_dict(),
        "input_dim": int(x_norm.shape[1]),
        "base_input_dim": int(base_x_norm.shape[1]),
        "output_dim": 6,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "num_steps": 100,
        "include_current_point": True,
        "x_mean": x_mean,
        "x_std": x_std,
        "fourier_frequencies": args.fourier_frequencies,
        "y_mean": y_mean,
        "y_std": y_std,
        "data_leakage": True,
        "independent_benchmark_eligible": False,
        "disclaimer": DISCLAIMER,
        "path_name": path_name,
        "training_source": str(args.input_npz.resolve()),
    }
    torch.save(checkpoint, expected_outputs[0])
    np.savez_compressed(
        expected_outputs[1],
        timestamps=timestamps,
        desired_path=desired,
        fitted_mlp_q=predicted_q,
        fitted_mlp_ee=predicted_ee,
        ik_q=ik_q,
        ik_ee=ik_ee,
        diffusion_ee=diffusion_ee,
        cartesian_error=cartesian_error,
        data_leakage=np.asarray(True),
        independent_benchmark_eligible=np.asarray(False),
        disclaimer=np.asarray(DISCLAIMER),
    )
    write_csv(
        expected_outputs[2],
        ["time_seconds", "q1", "q2", "q3", "q4", "q5", "q6"],
        np.column_stack((timestamps, predicted_q)),
    )
    write_csv(
        expected_outputs[3],
        ["time_seconds", "x", "y", "z", "cartesian_error"],
        np.column_stack((timestamps, predicted_ee, cartesian_error)),
    )
    expected_outputs[4].write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    figure, axis = plt.subplots(figsize=(10, 8), dpi=180)
    axis.plot(
        desired[:, 0], desired[:, 1], color="#111827", linewidth=3.5,
        label="Target", zorder=1,
    )
    axis.plot(
        predicted_ee[:, 0], predicted_ee[:, 1], color="#7c3aed",
        linewidth=2.5, linestyle="-.", label="MLP", zorder=4,
    )
    axis.plot(
        ik_ee[:, 0], ik_ee[:, 1], color="#f59e0b", linewidth=5.0,
        linestyle="--", label="IK", zorder=2,
    )
    axis.plot(
        diffusion_ee[:, 0], diffusion_ee[:, 1], color="#dc2626",
        linewidth=2.0, linestyle=":", marker="o", markersize=3.5,
        markevery=5, label="Diffusion", zorder=3,
    )
    axis.set_title("Stroke 02: Target vs MLP vs IK vs Diffusion")
    axis.set_xlabel("X position (m)")
    axis.set_ylabel("Y position (m)")
    axis.grid(True, linestyle="--", alpha=0.3)
    axis.legend()
    axis.margins(x=0.06, y=0.04)
    figure.tight_layout()
    figure.savefig(expected_outputs[5], bbox_inches="tight")
    plt.close(figure)

    diffusion_cartesian_error = np.linalg.norm(diffusion_ee - desired, axis=1)
    error_figure, error_axis = plt.subplots(figsize=(10, 6), dpi=180)
    error_axis.plot(
        timestamps, cartesian_error, color="#7c3aed", linewidth=2.5,
        linestyle="-.", label="MLP", zorder=3,
    )
    error_axis.plot(
        timestamps, ik_cartesian_error, color="#f59e0b", linewidth=4.0,
        linestyle="--", label="IK", zorder=1,
    )
    error_axis.plot(
        timestamps, diffusion_cartesian_error, color="#dc2626",
        linewidth=2.0, linestyle=":", marker="o", markersize=3.5,
        markevery=5, label="Diffusion", zorder=2,
    )
    error_axis.set_title("Stroke 02: Tracking Error Over Time")
    error_axis.set_xlabel("Time (s)")
    error_axis.set_ylabel("Cartesian tracking error (m)")
    error_axis.grid(True, linestyle="--", alpha=0.3)
    error_axis.legend()
    error_axis.margins(x=0.01)
    error_figure.tight_layout()
    error_figure.savefig(expected_outputs[6], bbox_inches="tight")
    plt.close(error_figure)

    print("VISUALIZATION_ONLY_MLP_FIT_COMPLETE")
    print(DISCLAIMER)
    print(f"q RMSE vs IK: {metrics['q_rmse_vs_ik_rad']:.12e} rad")
    print(
        "mean Cartesian error vs target: "
        f"{metrics['mean_cartesian_error_vs_target_m']:.12e} m"
    )
    print(
        "maximum Cartesian error vs target: "
        f"{metrics['maximum_cartesian_error_vs_target_m']:.12e} m"
    )
    print(f"output directory: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
