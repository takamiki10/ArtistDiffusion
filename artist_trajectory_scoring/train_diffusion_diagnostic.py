#!/usr/bin/env python3
"""
Minimal diagnostic conditional diffusion model for artist trajectory generation.

Goal:
    desired Cartesian path  ->  joint trajectory

This is NOT the final Stanford Diffusion Policy integration.
This is only a diagnostic run to verify that:

    1. Our NPZ dataset loads.
    2. A conditional diffusion training loop runs.
    3. A model checkpoint saves.
    4. Later, we can sample q trajectories and score them with FK.

Expected NPZ keys:
    desired_paths: (N, T, 3)
    actions:       (N, T, 6)
    times:         (N, T)

Training formulation:
    condition: flattened desired path, shape (T*3)
    target:    flattened joint trajectory, shape (T*6)

Diffusion:
    q trajectory is corrupted with Gaussian noise.
    model learns to predict the added noise.

Example:
    cd /workspace/artist_trajectory_scoring

    source /opt/conda/etc/profile.d/conda.sh
    conda activate robodiff

    python train_diffusion_diagnostic.py \
      --npz data/synthetic_paths_train_2000/multipath_episodes.npz \
      --epochs 20 \
      --batch_size 32 \
      --hidden_dim 1024 \
      --num_diffusion_steps 100 \
      --output_model data/synthetic_paths_train_2000/diffusion_diagnostic.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.clip_grad import clip_grad_norm_


# ------------------------------------------------------------
# Dataset loading
# ------------------------------------------------------------

def load_artist_npz(npz_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load multipath dataset.

    Returns:
        cond_raw:
            flattened desired paths, shape (N, T*3)

        action_raw:
            flattened joint trajectories, shape (N, T*6)
    """
    data = np.load(npz_path, allow_pickle=True)

    required = ["desired_paths", "actions"]
    missing = [key for key in required if key not in data]
    if missing:
        raise KeyError(
            f"Missing keys in {npz_path}: {missing}. "
            f"Available keys: {list(data.keys())}"
        )

    desired_paths = data["desired_paths"].astype(np.float32)  # (N, T, 3)
    actions = data["actions"].astype(np.float32)              # (N, T, 6)

    if desired_paths.ndim != 3 or desired_paths.shape[-1] != 3:
        raise ValueError(
            f"desired_paths must have shape (N, T, 3), got {desired_paths.shape}"
        )

    if actions.ndim != 3 or actions.shape[-1] != 6:
        raise ValueError(
            f"actions must have shape (N, T, 6), got {actions.shape}"
        )

    if desired_paths.shape[0] != actions.shape[0]:
        raise ValueError(
            f"N mismatch: desired_paths has {desired_paths.shape[0]}, "
            f"actions has {actions.shape[0]}"
        )

    if desired_paths.shape[1] != actions.shape[1]:
        raise ValueError(
            f"T mismatch: desired_paths has {desired_paths.shape[1]}, "
            f"actions has {actions.shape[1]}"
        )

    n, t, _ = desired_paths.shape

    cond_raw = desired_paths.reshape(n, t * 3)
    action_raw = actions.reshape(n, t * 6)

    return cond_raw, action_raw


def standardize(x: np.ndarray, eps: float = 1e-8):
    """
    Standardize data and return standardized values plus statistics.
    """
    mean = x.mean(axis=0, keepdims=True).astype(np.float32)
    std = x.std(axis=0, keepdims=True).astype(np.float32)
    std = np.maximum(std, eps)

    x_std = ((x - mean) / std).astype(np.float32)
    return x_std, mean, std


# ------------------------------------------------------------
# Diffusion schedule
# ------------------------------------------------------------

class DiffusionSchedule:
    """
    Basic DDPM-style linear beta schedule.

    For training:
        x_t = sqrt(alpha_bar_t) * x_0
              + sqrt(1 - alpha_bar_t) * noise

    The model predicts noise.
    """

    def __init__(
        self,
        num_steps: int,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        device: torch.device | str = "cpu",
    ) -> None:
        self.num_steps = int(num_steps)
        self.device = torch.device(device)

        betas = torch.linspace(
            beta_start,
            beta_end,
            self.num_steps,
            dtype=torch.float32,
            device=self.device,
        )

        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alpha_bars = alpha_bars
        self.sqrt_alpha_bars = torch.sqrt(alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - alpha_bars)

    def to(self, device: torch.device | str):
        device = torch.device(device)
        self.device = device

        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_bars = self.alpha_bars.to(device)
        self.sqrt_alpha_bars = self.sqrt_alpha_bars.to(device)
        self.sqrt_one_minus_alpha_bars = self.sqrt_one_minus_alpha_bars.to(device)
        return self

    def add_noise(
        self,
        x0: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x0:
                clean action trajectory, shape (B, action_dim)

            timesteps:
                integer diffusion timestep, shape (B,)

            noise:
                Gaussian noise, same shape as x0
        """
        sqrt_ab = self.sqrt_alpha_bars[timesteps].view(-1, 1)
        sqrt_omab = self.sqrt_one_minus_alpha_bars[timesteps].view(-1, 1)

        return sqrt_ab * x0 + sqrt_omab * noise


# ------------------------------------------------------------
# Model
# ------------------------------------------------------------

class SinusoidalTimestepEmbedding(nn.Module):
    """
    Standard sinusoidal embedding for diffusion timestep.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()

        if dim % 2 != 0:
            raise ValueError("Timestep embedding dimension must be even.")

        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t:
                diffusion timestep, shape (B,)

        Returns:
            embedding, shape (B, dim)
        """
        device = t.device
        half_dim = self.dim // 2

        freqs = torch.exp(
            -np.log(10000.0)
            * torch.arange(half_dim, device=device, dtype=torch.float32)
            / max(half_dim - 1, 1)
        )

        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)

        return torch.cat([torch.sin(args), torch.cos(args)], dim=1)


class ConditionalNoisePredictor(nn.Module):
    """
    Simple MLP that predicts diffusion noise.

    Inputs:
        noisy_action:
            noisy q trajectory, shape (B, T*6)

        condition:
            desired path, shape (B, T*3)

        diffusion timestep:
            integer timestep, shape (B,)

    Output:
        predicted noise, shape (B, T*6)
    """

    def __init__(
        self,
        action_dim: int,
        cond_dim: int,
        hidden_dim: int = 1024,
        timestep_embed_dim: int = 128,
        num_layers: int = 4,
    ) -> None:
        super().__init__()

        if num_layers < 2:
            raise ValueError("--num_layers must be at least 2")

        self.action_dim = action_dim
        self.cond_dim = cond_dim
        self.hidden_dim = hidden_dim
        self.timestep_embed_dim = timestep_embed_dim
        self.num_layers = num_layers

        self.t_embed = SinusoidalTimestepEmbedding(timestep_embed_dim)

        input_dim = action_dim + cond_dim + timestep_embed_dim

        layers = []
        in_dim = input_dim

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, action_dim))

        self.net = nn.Sequential(*layers)

    def forward(
        self,
        noisy_action: torch.Tensor,
        condition: torch.Tensor,
        diffusion_timestep: torch.Tensor,
    ) -> torch.Tensor:
        t_emb = self.t_embed(diffusion_timestep)

        x = torch.cat(
            [
                noisy_action,
                condition,
                t_emb,
            ],
            dim=1,
        )

        return self.net(x)


# ------------------------------------------------------------
# Training
# ------------------------------------------------------------

def train(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Device: {device}")

    cond_raw, action_raw = load_artist_npz(args.npz)

    cond, cond_mean, cond_std = standardize(cond_raw)
    action, action_mean, action_std = standardize(action_raw)

    n = cond.shape[0]
    cond_dim = cond.shape[1]
    action_dim = action.shape[1]

    print(f"Loaded dataset: {args.npz}")
    print(f"N episodes: {n}")
    print(f"condition dim: {cond_dim}")
    print(f"action dim: {action_dim}")

    dataset = TensorDataset(
        torch.from_numpy(cond),
        torch.from_numpy(action),
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )

    schedule = DiffusionSchedule(
        num_steps=args.num_diffusion_steps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        device=device,
    )

    model = ConditionalNoisePredictor(
        action_dim=action_dim,
        cond_dim=cond_dim,
        hidden_dim=args.hidden_dim,
        timestep_embed_dim=args.timestep_embed_dim,
        num_layers=args.num_layers,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    loss_fn = nn.MSELoss()

    print("Starting diagnostic diffusion training...")
    print(
        "Note: this diagnostic model is only meant to verify the pipeline, "
        "not to produce high-quality trajectories yet."
    )

    for epoch in range(1, args.epochs + 1):
        model.train()

        total_loss = 0.0
        total_count = 0

        for cond_batch, action_batch in loader:
            cond_batch = cond_batch.to(device)
            action_batch = action_batch.to(device)

            batch_size = action_batch.shape[0]

            diffusion_t = torch.randint(
                low=0,
                high=args.num_diffusion_steps,
                size=(batch_size,),
                device=device,
                dtype=torch.long,
            )

            noise = torch.randn_like(action_batch)

            noisy_action = schedule.add_noise(
                x0=action_batch,
                timesteps=diffusion_t,
                noise=noise,
            )

            pred_noise = model(
                noisy_action=noisy_action,
                condition=cond_batch,
                diffusion_timestep=diffusion_t,
            )

            loss = loss_fn(pred_noise, noise)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            if args.grad_clip > 0:
                clip_grad_norm_(model.parameters(), args.grad_clip)

            optimizer.step()

            total_loss += float(loss.item()) * batch_size
            total_count += batch_size

        mean_loss = total_loss / max(total_count, 1)

        if epoch == 1 or epoch % args.log_every == 0 or epoch == args.epochs:
            print(f"epoch {epoch:05d} | noise_pred_mse={mean_loss:.8e}")

    args.output_model.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),

        "cond_dim": cond_dim,
        "action_dim": action_dim,
        "hidden_dim": args.hidden_dim,
        "timestep_embed_dim": args.timestep_embed_dim,
        "num_layers": args.num_layers,

        "num_diffusion_steps": args.num_diffusion_steps,
        "beta_start": args.beta_start,
        "beta_end": args.beta_end,

        "cond_mean": cond_mean,
        "cond_std": cond_std,
        "action_mean": action_mean,
        "action_std": action_std,

        "npz": str(args.npz),
        "seed": args.seed,
    }

    torch.save(checkpoint, args.output_model)

    print(f"Saved diagnostic diffusion model: {args.output_model}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--npz",
        type=Path,
        required=True,
        help="Path to multipath_episodes.npz",
    )

    parser.add_argument(
        "--output_model",
        type=Path,
        required=True,
        help="Where to save the diagnostic diffusion checkpoint",
    )

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)

    parser.add_argument("--hidden_dim", type=int, default=1024)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--timestep_embed_dim", type=int, default=128)

    parser.add_argument("--num_diffusion_steps", type=int, default=100)
    parser.add_argument("--beta_start", type=float, default=1e-4)
    parser.add_argument("--beta_end", type=float, default=2e-2)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
    )

    parser.add_argument("--log_every", type=int, default=1)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":
    main()