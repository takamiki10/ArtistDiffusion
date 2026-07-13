#!/usr/bin/env python3
"""Train conditional diffusion v3 with direct x0 prediction."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_npz", default="data/cartesian_expert_dataset_v3/diffusion_v2/diffusion_train_v2.npz")
    parser.add_argument("--test_npz", default="data/cartesian_expert_dataset_v3/diffusion_v2/diffusion_test_v2.npz")
    parser.add_argument("--output_model", default="data/cartesian_expert_dataset_v3/diffusion_v3_x0/conditional_diffusion_v3_x0.pt")
    parser.add_argument("--latest_model", default="data/cartesian_expert_dataset_v3/diffusion_v3_x0/conditional_diffusion_v3_x0_latest.pt")
    parser.add_argument("--log_csv", default="data/cartesian_expert_dataset_v3/diffusion_v3_x0/conditional_diffusion_v3_x0_train_log.csv")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_diffusion_steps", type=int, default=100)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--accel_loss_weight", type=float, default=0.0)
    return parser.parse_args()


def select_device(device_arg: str) -> torch.device:
    if device_arg != "auto":
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_beta_schedule(
    num_steps: int,
    device: torch.device,
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    betas = torch.linspace(beta_start, beta_end, num_steps, device=device)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_bars


def sinusoidal_timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half_dim = dim // 2
    exponent = -math.log(10000.0) * torch.arange(half_dim, device=timesteps.device) / max(half_dim - 1, 1)
    frequencies = torch.exp(exponent)
    args = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        embedding = F.pad(embedding, (0, 1))
    return embedding


class ResidualTemporalBlock(nn.Module):
    def __init__(self, hidden_dim: int, time_dim: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, hidden_dim)
        self.norm2 = nn.GroupNorm(8, hidden_dim)
        self.time_proj = nn.Linear(time_dim, hidden_dim)

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(time_emb).unsqueeze(-1)
        h = self.conv2(F.silu(self.norm2(h)))
        return residual + h


class ConditionalTrajectoryX0Denoiser(nn.Module):
    def __init__(
        self,
        condition_dim: int = 13,
        target_dim: int = 6,
        hidden_dim: int = 256,
        num_blocks: int = 6,
    ) -> None:
        super().__init__()
        self.condition_dim = condition_dim
        self.target_dim = target_dim
        self.hidden_dim = hidden_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.input_proj = nn.Conv1d(condition_dim + target_dim, hidden_dim, kernel_size=1)
        self.blocks = nn.ModuleList(
            ResidualTemporalBlock(hidden_dim=hidden_dim, time_dim=hidden_dim)
            for _ in range(num_blocks)
        )
        self.output_norm = nn.GroupNorm(8, hidden_dim)
        self.output_proj = nn.Conv1d(hidden_dim, target_dim, kernel_size=1)

    def forward(self, x_t: torch.Tensor, condition: torch.Tensor, timestep: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x_t, condition], dim=-1).transpose(1, 2)
        time_emb = sinusoidal_timestep_embedding(timestep, self.hidden_dim)
        time_emb = self.time_mlp(time_emb)
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h, time_emb)
        out = self.output_proj(F.silu(self.output_norm(h)))
        return out.transpose(1, 2)


def load_npz_dataset(path: Path) -> TensorDataset:
    data = np.load(path, allow_pickle=True)
    required = ["condition_features_norm", "delta_q_norm"]
    missing = [key for key in required if key not in data.files]
    if missing:
        raise KeyError(f"Missing keys in {path}: {missing}. Available keys: {data.files}")
    condition = torch.from_numpy(data["condition_features_norm"].astype(np.float32))
    target = torch.from_numpy(data["delta_q_norm"].astype(np.float32))
    return TensorDataset(condition, target)


def sample_noisy_x0(
    x0: torch.Tensor,
    alpha_bars: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = x0.shape[0]
    timesteps = torch.randint(0, alpha_bars.shape[0], (batch_size,), device=x0.device)
    noise = torch.randn_like(x0)
    alpha_bar_t = alpha_bars[timesteps].view(batch_size, 1, 1)
    x_t = torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1.0 - alpha_bar_t) * noise
    return x_t, timesteps


def acceleration_loss(pred_x0: torch.Tensor) -> torch.Tensor:
    if pred_x0.shape[1] < 3:
        return pred_x0.new_tensor(0.0)
    accel = pred_x0[:, 2:, :] - 2.0 * pred_x0[:, 1:-1, :] + pred_x0[:, :-2, :]
    return torch.mean(accel * accel)


def evaluate(
    model: ConditionalTrajectoryX0Denoiser,
    loader: DataLoader,
    alpha_bars: torch.Tensor,
    device: torch.device,
) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for condition, x0 in loader:
            condition = condition.to(device)
            x0 = x0.to(device)
            x_t, timesteps = sample_noisy_x0(x0, alpha_bars)
            pred_x0 = model(x_t, condition, timesteps)
            losses.append(F.mse_loss(pred_x0, x0).item())
    model.train()
    return float(np.mean(losses)) if losses else float("inf")


def save_checkpoint(
    path: Path,
    model: ConditionalTrajectoryX0Denoiser,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_loss: float,
    val_loss: float,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "args": vars(args),
            "num_diffusion_steps": args.num_diffusion_steps,
            "hidden_dim": args.hidden_dim,
            "condition_dim": 13,
            "target_dim": 6,
            "prediction_type": "x0",
        },
        path,
    )


def append_log(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["epoch", "train_loss", "train_x0_loss", "train_accel_loss", "val_loss", "lr"]
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = select_device(args.device)

    train_dataset = load_npz_dataset(Path(args.train_npz))
    test_dataset = load_npz_dataset(Path(args.test_npz))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    model = ConditionalTrajectoryX0Denoiser(hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    _, _, alpha_bars = make_beta_schedule(args.num_diffusion_steps, device)
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_losses = []
        train_x0_losses = []
        train_accel_losses = []

        for condition, x0 in train_loader:
            condition = condition.to(device)
            x0 = x0.to(device)
            x_t, timesteps = sample_noisy_x0(x0, alpha_bars)
            pred_x0 = model(x_t, condition, timesteps)

            loss_x0 = F.mse_loss(pred_x0, x0)
            loss_accel = acceleration_loss(pred_x0) if args.accel_loss_weight > 0.0 else pred_x0.new_tensor(0.0)
            loss = loss_x0 + args.accel_loss_weight * loss_accel

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())
            train_x0_losses.append(loss_x0.item())
            train_accel_losses.append(loss_accel.item())

        train_loss = float(np.mean(train_losses)) if train_losses else float("inf")
        train_x0_loss = float(np.mean(train_x0_losses)) if train_x0_losses else float("inf")
        train_accel_loss = float(np.mean(train_accel_losses)) if train_accel_losses else 0.0

        if epoch == 1 or epoch % 10 == 0:
            val_loss = evaluate(model, val_loader, alpha_bars, device)
        else:
            val_loss = best_val_loss

        append_log(
            Path(args.log_csv),
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_x0_loss": train_x0_loss,
                "train_accel_loss": train_accel_loss,
                "val_loss": val_loss,
                "lr": optimizer.param_groups[0]["lr"],
            },
        )

        save_checkpoint(Path(args.latest_model), model, optimizer, epoch, train_loss, val_loss, args)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(Path(args.output_model), model, optimizer, epoch, train_loss, val_loss, args)

        print(
            f"epoch={epoch} train_loss={train_loss:.6f} "
            f"train_x0_loss={train_x0_loss:.6f} train_accel_loss={train_accel_loss:.6f} "
            f"val_loss={val_loss:.6f}"
        )


if __name__ == "__main__":
    main()
