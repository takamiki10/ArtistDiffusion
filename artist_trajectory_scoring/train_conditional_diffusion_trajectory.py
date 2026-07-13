#!/usr/bin/env python3
"""
Train a conditional DDPM-style trajectory diffusion model.

Condition:
    desired_paths_norm: (N, 100, 3)

Target:
    expert_q_norm:      (N, 100, 6)
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def load_diffusion_npz(npz_path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
    data = np.load(npz_path, allow_pickle=True)
    required = ["desired_paths_norm", "expert_q_norm"]
    missing = [key for key in required if key not in data.files]
    if missing:
        raise KeyError(f"Missing keys in {npz_path}: {missing}. Available: {data.files}")

    desired_paths = data["desired_paths_norm"].astype(np.float32)
    expert_q = data["expert_q_norm"].astype(np.float32)

    if desired_paths.ndim != 3 or desired_paths.shape[1:] != (100, 3):
        raise ValueError(f"desired_paths_norm must have shape (N,100,3), got {desired_paths.shape}")
    if expert_q.ndim != 3 or expert_q.shape[1:] != (100, 6):
        raise ValueError(f"expert_q_norm must have shape (N,100,6), got {expert_q.shape}")
    if desired_paths.shape[0] != expert_q.shape[0]:
        raise ValueError(
            f"desired_paths_norm and expert_q_norm must share N, got "
            f"{desired_paths.shape[0]} and {expert_q.shape[0]}"
        )

    return torch.from_numpy(desired_paths), torch.from_numpy(expert_q)


def sinusoidal_timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half_dim = dim // 2
    exponent = -math.log(10000.0) * torch.arange(
        half_dim, device=timesteps.device, dtype=torch.float32
    )
    exponent = exponent / max(half_dim - 1, 1)
    freqs = torch.exp(exponent)
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)

    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))

    return emb


class ResidualTemporalBlock(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups=8, num_channels=hidden_dim)
        self.conv1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.GroupNorm(num_groups=8, num_channels=hidden_dim)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(F.silu(time_emb)).unsqueeze(-1)
        h = self.conv2(F.silu(self.norm2(h)))
        return residual + h


class ConditionalTrajectoryDenoiser(nn.Module):
    def __init__(
        self,
        q_dim: int = 6,
        path_dim: int = 3,
        hidden_dim: int = 256,
        num_blocks: int = 6,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_proj = nn.Conv1d(q_dim + path_dim, hidden_dim, kernel_size=3, padding=1)
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.blocks = nn.ModuleList(
            ResidualTemporalBlock(hidden_dim) for _ in range(num_blocks)
        )
        self.output_norm = nn.GroupNorm(num_groups=8, num_channels=hidden_dim)
        self.output_proj = nn.Conv1d(hidden_dim, q_dim, kernel_size=3, padding=1)

    def forward(
        self,
        noisy_q: torch.Tensor,
        desired_path: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([noisy_q, desired_path], dim=-1).transpose(1, 2)
        time_emb = sinusoidal_timestep_embedding(timesteps, self.hidden_dim)
        time_emb = self.time_mlp(time_emb)

        h = self.input_proj(x)
        h = h + time_emb.unsqueeze(-1)
        for block in self.blocks:
            h = block(h, time_emb)

        pred_noise = self.output_proj(F.silu(self.output_norm(h))).transpose(1, 2)
        return pred_noise


class DDPMSchedule(nn.Module):
    def __init__(self, num_diffusion_steps: int) -> None:
        super().__init__()
        beta = torch.linspace(1e-4, 0.02, num_diffusion_steps, dtype=torch.float32)
        alpha = 1.0 - beta
        alpha_bar = torch.cumprod(alpha, dim=0)

        self.num_diffusion_steps = num_diffusion_steps
        self.register_buffer("beta", beta)
        self.register_buffer("alpha", alpha)
        self.register_buffer("alpha_bar", alpha_bar)
        self.register_buffer("sqrt_alpha_bar", torch.sqrt(alpha_bar))
        self.register_buffer("sqrt_one_minus_alpha_bar", torch.sqrt(1.0 - alpha_bar))

    def add_noise(
        self,
        x0: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        sqrt_alpha_bar_t = self.sqrt_alpha_bar[timesteps].view(-1, 1, 1)
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bar[timesteps].view(-1, 1, 1)
        return sqrt_alpha_bar_t * x0 + sqrt_one_minus_alpha_bar_t * noise


def make_dataloader(
    condition: torch.Tensor,
    target: torch.Tensor,
    batch_size: int,
    shuffle: bool,
    seed: int,
    device: torch.device,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    dataset = TensorDataset(condition, target)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        generator=generator if shuffle else None,
        pin_memory=(device.type == "cuda"),
    )


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(
    model: nn.Module,
    schedule: DDPMSchedule,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
) -> float:
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_count = 0

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for condition, x0 in loader:
            condition = condition.to(device, non_blocking=True)
            x0 = x0.to(device, non_blocking=True)
            batch_size = x0.shape[0]

            timesteps = torch.randint(
                low=0,
                high=schedule.num_diffusion_steps,
                size=(batch_size,),
                device=device,
            )
            noise = torch.randn_like(x0)
            noisy_q = schedule.add_noise(x0, noise, timesteps)
            pred_noise = model(noisy_q, condition, timesteps)
            loss = F.mse_loss(pred_noise, noise)

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            total_loss += float(loss.item()) * batch_size
            total_count += batch_size

    return total_loss / max(total_count, 1)


def checkpoint_payload(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_loss: float,
    val_loss: float,
    args: argparse.Namespace,
) -> Dict[str, object]:
    args_dict = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "args": args_dict,
        "num_diffusion_steps": int(args.num_diffusion_steps),
        "hidden_dim": int(args.hidden_dim),
    }


def latest_checkpoint_path(output_model: Path) -> Path:
    suffix = output_model.suffix or ".pt"
    return output_model.with_name(f"{output_model.stem}_latest{suffix}")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train conditional DDPM trajectory diffusion model."
    )
    parser.add_argument(
        "--train_npz",
        type=Path,
        default=Path("data/cartesian_expert_dataset_v3/diffusion_train.npz"),
    )
    parser.add_argument(
        "--test_npz",
        type=Path,
        default=Path("data/cartesian_expert_dataset_v3/diffusion_test.npz"),
    )
    parser.add_argument(
        "--output_model",
        type=Path,
        default=Path("data/cartesian_expert_dataset_v3/conditional_diffusion_v1.pt"),
    )
    parser.add_argument(
        "--log_csv",
        type=Path,
        default=Path("data/cartesian_expert_dataset_v3/conditional_diffusion_v1_train_log.csv"),
    )
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_diffusion_steps", type=int, default=100)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    train_condition, train_target = load_diffusion_npz(args.train_npz)
    test_condition, test_target = load_diffusion_npz(args.test_npz)

    train_loader = make_dataloader(
        train_condition,
        train_target,
        batch_size=args.batch_size,
        shuffle=True,
        seed=args.seed,
        device=device,
    )
    val_loader = make_dataloader(
        test_condition,
        test_target,
        batch_size=args.batch_size,
        shuffle=False,
        seed=args.seed,
        device=device,
    )

    model = ConditionalTrajectoryDenoiser(hidden_dim=args.hidden_dim).to(device)
    schedule = DDPMSchedule(args.num_diffusion_steps).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    args.log_csv.parent.mkdir(parents=True, exist_ok=True)
    latest_model = latest_checkpoint_path(args.output_model)

    print(f"Train NPZ: {args.train_npz}")
    print(f"Test NPZ:  {args.test_npz}")
    print(f"Train samples: {train_target.shape[0]} | Test samples: {test_target.shape[0]}")
    print(f"Device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    best_val_loss = float("inf")

    with args.log_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "lr"])
        writer.writeheader()

        for epoch in range(1, args.epochs + 1):
            train_loss = run_epoch(model, schedule, train_loader, optimizer, device)
            val_loss = run_epoch(model, schedule, val_loader, optimizer=None, device=device)
            lr = optimizer.param_groups[0]["lr"]

            writer.writerow(
                {
                    "epoch": epoch,
                    "train_loss": f"{train_loss:.10f}",
                    "val_loss": f"{val_loss:.10f}",
                    "lr": f"{lr:.10g}",
                }
            )
            f.flush()

            payload = checkpoint_payload(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                train_loss=train_loss,
                val_loss=val_loss,
                args=args,
            )

            torch.save(payload, latest_model)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(payload, args.output_model)

            if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
                print(
                    f"epoch {epoch:05d} | train_loss={train_loss:.8e} "
                    f"| val_loss={val_loss:.8e} | lr={lr:.3e}"
                )

    print()
    print(f"Best checkpoint:   {args.output_model}")
    print(f"Latest checkpoint: {latest_model}")
    print(f"Training log:      {args.log_csv}")


if __name__ == "__main__":
    main()
