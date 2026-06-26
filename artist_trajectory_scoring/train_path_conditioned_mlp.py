#!/usr/bin/env python3
"""
Train a path-conditioned MLP for Cartesian expert trajectories.

Unlike train_time_conditioned_mlp.py, each timestep input includes the full
desired path flattened into one vector.  This gives the model global drawing
context while still predicting one q[t] at a time.

Input per timestep:
    [desired_path_flat, t, x_t, y_t, z_t] by default

Target per timestep:
    [q1, q2, q3, q4, q5, q6]

Expected NPZ keys:
    desired_paths: (N, T, 3)
    actions:       (N, T, 6)
    times:         (N, T)
    path_ids:      (N,) optional
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class PathConditionedMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 6,
        hidden_dim: int = 512,
        num_layers: int = 4,
    ) -> None:
        super().__init__()
        if num_layers < 2:
            raise ValueError("--num_layers must be at least 2")

        layers = []
        in_dim = input_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_npz(npz_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(npz_path, allow_pickle=True)
    required = ["desired_paths", "actions", "times"]
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(f"Missing keys in {npz_path}: {missing}. Available: {list(data.keys())}")

    desired_paths = data["desired_paths"].astype(np.float32)
    actions = data["actions"].astype(np.float32)
    times = data["times"].astype(np.float32)
    path_ids = data["path_ids"] if "path_ids" in data.files else np.arange(len(desired_paths))

    if desired_paths.ndim != 3 or desired_paths.shape[-1] != 3:
        raise ValueError(f"desired_paths must have shape (N,T,3), got {desired_paths.shape}")
    if actions.ndim != 3 or actions.shape[-1] != 6:
        raise ValueError(f"actions must have shape (N,T,6), got {actions.shape}")
    if times.shape != desired_paths.shape[:2]:
        raise ValueError(
            f"times must have shape {desired_paths.shape[:2]}, got {times.shape}"
        )
    if actions.shape[:2] != desired_paths.shape[:2]:
        raise ValueError(
            f"actions and desired_paths must share (N,T), got {actions.shape} vs {desired_paths.shape}"
        )

    return desired_paths, actions, times, path_ids


def make_timestep_dataset(
    desired_paths: np.ndarray,
    actions: np.ndarray,
    times: np.ndarray,
    include_current_point: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    n, steps, _ = desired_paths.shape
    path_flat = desired_paths.reshape(n, 1, steps * 3)
    path_context = np.repeat(path_flat, steps, axis=1)
    features = [path_context, times[..., None]]
    if include_current_point:
        features.append(desired_paths)
    x = np.concatenate(features, axis=-1).astype(np.float32)
    y = actions.astype(np.float32)
    return x.reshape(n * steps, -1), y.reshape(n * steps, 6)


def standardize(
    x: np.ndarray,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0, keepdims=True).astype(np.float32)
    std = x.std(axis=0, keepdims=True).astype(np.float32)
    std = np.maximum(std, eps)
    return ((x - mean) / std).astype(np.float32), mean, std


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train path-conditioned Cartesian expert MLP.")
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--output_model", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument(
        "--no_current_point",
        action="store_true",
        help="Use only full desired path + t, without current [x,y,z].",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    desired_paths, actions, times, path_ids = load_npz(args.npz)
    include_current_point = not args.no_current_point
    x_raw, y_raw = make_timestep_dataset(desired_paths, actions, times, include_current_point)
    x, x_mean, x_std = standardize(x_raw)
    y, y_mean, y_std = standardize(y_raw)

    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

    model = PathConditionedMLP(
        input_dim=x.shape[1],
        output_dim=6,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()

    print(f"Loaded: {args.npz}")
    print(f"Paths: {desired_paths.shape[0]} | T: {desired_paths.shape[1]}")
    print(f"Samples: {len(dataset)} timestep samples")
    print(f"Input dim: {x.shape[1]} | include_current_point={include_current_point}")
    print(f"Device: {device}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_count = 0
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            loss = loss_fn(pred, yb)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * xb.shape[0]
            total_count += xb.shape[0]

        mean_loss = total_loss / max(total_count, 1)
        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(f"epoch {epoch:05d} | train_mse_normalized={mean_loss:.8e}")

    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_type": "path_conditioned_mlp",
            "input_dim": int(x.shape[1]),
            "output_dim": 6,
            "hidden_dim": int(args.hidden_dim),
            "num_layers": int(args.num_layers),
            "num_steps": int(desired_paths.shape[1]),
            "path_dim": int(desired_paths.shape[1] * 3),
            "include_current_point": bool(include_current_point),
            "x_mean": x_mean,
            "x_std": x_std,
            "y_mean": y_mean,
            "y_std": y_std,
            "npz": str(args.npz),
            "path_ids": path_ids,
        },
        args.output_model,
    )
    print(f"Saved model: {args.output_model}")


if __name__ == "__main__":
    main()
