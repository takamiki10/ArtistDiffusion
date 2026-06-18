#!/usr/bin/env python3
"""
Train a GRU sequence baseline for artist trajectory generation.

Goal:
    desired Cartesian path sequence -> joint trajectory sequence

Input:
    desired_paths: (N, T, 3)
    times:         (N, T)

Model input:
    [x, y, z, t], shape (N, T, 4)

Target:
    [q1, q2, q3, q4, q5, q6], shape (N, T, 6)

Example:
    cd /workspace/artist_trajectory_scoring

    source /opt/conda/etc/profile.d/conda.sh
    conda activate robodiff

    python train_sequence_gru.py \
      --npz data/synthetic_paths_train_2000/multipath_episodes.npz \
      --epochs 2000 \
      --batch_size 64 \
      --hidden_dim 256 \
      --num_layers 2 \
      --bidirectional \
      --output_model data/synthetic_paths_train_2000/sequence_gru.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.data import DataLoader, TensorDataset


class SequenceGRU(nn.Module):
    def __init__(
        self,
        input_dim: int = 4,
        output_dim: int = 6,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.0,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()

        if num_layers < 1:
            raise ValueError("--num_layers must be at least 1")

        # PyTorch GRU only applies dropout between layers.
        gru_dropout = dropout if num_layers > 1 else 0.0

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=gru_dropout,
            bidirectional=bidirectional,
        )

        gru_out_dim = hidden_dim * (2 if bidirectional else 1)

        self.head = nn.Sequential(
            nn.Linear(gru_out_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:
                input sequence, shape (B, T, 4)

        Returns:
            predicted joint sequence, shape (B, T, 6)
        """
        h, _ = self.gru(x)
        y = self.head(h)
        return y


def load_npz_as_sequences(npz_path: Path) -> Tuple[np.ndarray, np.ndarray]:
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

    if desired_paths.shape[:2] != actions.shape[:2] or desired_paths.shape[:2] != times.shape:
        raise ValueError(
            "Shape mismatch: "
            f"desired_paths={desired_paths.shape}, actions={actions.shape}, times={times.shape}"
        )

    x = np.concatenate([desired_paths, times[..., None]], axis=-1).astype(np.float32)  # (N,T,4)
    y = actions.astype(np.float32)                                                     # (N,T,6)

    return x, y


def standardize_sequence_features(
    x: np.ndarray,
    eps: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Standardize over both episode and timestep dimensions.

    x shape:
        (N, T, D)

    mean/std shape:
        (1, 1, D)
    """
    mean = x.mean(axis=(0, 1), keepdims=True).astype(np.float32)
    std = x.std(axis=(0, 1), keepdims=True).astype(np.float32)
    std = np.maximum(std, eps)

    x_std = ((x - mean) / std).astype(np.float32)
    return x_std, mean, std


def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    x_raw, y_raw = load_npz_as_sequences(args.npz)

    x, x_mean, x_std = standardize_sequence_features(x_raw)
    y, y_mean, y_std = standardize_sequence_features(y_raw)

    dataset = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )

    model = SequenceGRU(
        input_dim=x.shape[-1],
        output_dim=y.shape[-1],
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        bidirectional=args.bidirectional,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    loss_fn = nn.MSELoss()

    print(f"Loaded: {args.npz}")
    print(f"Sequences: {x.shape[0]}")
    print(f"Timesteps: {x.shape[1]}")
    print(f"Input dim: {x.shape[-1]}")
    print(f"Output dim: {y.shape[-1]}")
    print(f"Device: {device}")
    print(
        f"Model: GRU hidden_dim={args.hidden_dim}, "
        f"num_layers={args.num_layers}, "
        f"bidirectional={args.bidirectional}, "
        f"dropout={args.dropout}"
    )

    best_loss = float("inf")

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

            if args.grad_clip > 0:
                clip_grad_norm_(model.parameters(), args.grad_clip)

            optimizer.step()

            batch_size = xb.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_count += batch_size

        mean_loss = total_loss / max(total_count, 1)
        best_loss = min(best_loss, mean_loss)

        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(f"epoch {epoch:05d} | train_mse_normalized={mean_loss:.8e} | best={best_loss:.8e}")

    args.output_model.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": int(x.shape[-1]),
            "output_dim": int(y.shape[-1]),
            "hidden_dim": args.hidden_dim,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "bidirectional": args.bidirectional,
            "x_mean": x_mean,
            "x_std": x_std,
            "y_mean": y_mean,
            "y_std": y_std,
            "npz": str(args.npz),
            "seed": args.seed,
            "best_train_mse_normalized": best_loss,
        },
        args.output_model,
    )

    print(f"Saved model: {args.output_model}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--npz", type=Path, required=True)
    parser.add_argument("--output_model", type=Path, required=True)

    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=64)

    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--bidirectional", action="store_true")

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--log_every", type=int, default=50)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()
