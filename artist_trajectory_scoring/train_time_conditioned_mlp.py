#!/usr/bin/env python3
"""
Train a time-conditioned MLP baseline.

Input per timestep:
    [x, y, z, t]

Target per timestep:
    [q1, q2, q3, q4, q5, q6]

Expected NPZ keys:
    desired_paths: (N, T, 3)
    actions:       (N, T, 6)
    times:         (N, T)

Example:
python train_time_conditioned_mlp.py \
  --npz data/synthetic_paths_train_2000/multipath_episodes.npz \
  --epochs 2000 \
  --batch_size 4096 \
  --hidden_dim 256 \
  --num_layers 4 \
  --output_model data/synthetic_paths_train_2000/time_conditioned_mlp.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class TimeConditionedMLP(nn.Module):
    def __init__(
        self,
        input_dim: int = 4,
        output_dim: int = 6,
        hidden_dim: int = 256,
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


def load_npz_as_timestep_dataset(npz_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(npz_path, allow_pickle=True)

    required = ["desired_paths", "actions", "times"]
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(
            f"Missing keys in {npz_path}: {missing}. "
            f"Available keys: {list(data.keys())}"
        )

    desired_paths = data["desired_paths"].astype(np.float32)  # (N, T, 3)
    actions = data["actions"].astype(np.float32)              # (N, T, 6)
    times = data["times"].astype(np.float32)                  # (N, T)

    if desired_paths.ndim != 3 or desired_paths.shape[-1] != 3:
        raise ValueError(f"desired_paths must have shape (N,T,3), got {desired_paths.shape}")
    if actions.ndim != 3 or actions.shape[-1] != 6:
        raise ValueError(f"actions must have shape (N,T,6), got {actions.shape}")
    if times.ndim != 2:
        raise ValueError(f"times must have shape (N,T), got {times.shape}")
    if desired_paths.shape[:2] != actions.shape[:2] or times.shape != desired_paths.shape[:2]:
        raise ValueError(
            "Shape mismatch: "
            f"desired_paths={desired_paths.shape}, actions={actions.shape}, times={times.shape}"
        )

    time_col = times[..., None]                               # (N, T, 1)
    inputs = np.concatenate([desired_paths, time_col], axis=-1) # (N, T, 4)

    x = inputs.reshape(-1, 4)
    y = actions.reshape(-1, 6)
    return x, y


def standardize(
    x: np.ndarray,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0, keepdims=True).astype(np.float32)
    std = x.std(axis=0, keepdims=True).astype(np.float32)
    std = np.maximum(std, eps)
    return ((x - mean) / std).astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=4096)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--output_model", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--log_every", type=int, default=50)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    x_raw, y_raw = load_npz_as_timestep_dataset(args.npz)

    x, x_mean, x_std = standardize(x_raw)
    y, y_mean, y_std = standardize(y_raw)

    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

    model = TimeConditionedMLP(
        input_dim=4,
        output_dim=6,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()

    print(f"Loaded: {args.npz}")
    print(f"Samples: {len(dataset)} timestep samples")
    print(f"Device: {device}")
    print(f"Model: hidden_dim={args.hidden_dim}, num_layers={args.num_layers}")

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

            batch_size = xb.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_count += batch_size

        mean_loss = total_loss / max(total_count, 1)

        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(f"epoch {epoch:05d} | train_mse_normalized={mean_loss:.8e}")

    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": 4,
            "output_dim": 6,
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "x_mean": x_mean,
            "x_std": x_std,
            "y_mean": y_mean,
            "y_std": y_std,
            "npz": str(args.npz),
        },
        args.output_model,
    )

    print(f"Saved model: {args.output_model}")


if __name__ == "__main__":
    main()
