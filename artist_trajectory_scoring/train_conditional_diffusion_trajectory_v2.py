#!/usr/bin/env python3
"""Train conditional DDPM v2 for delta-q trajectory generation."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train_npz", default="data/cartesian_expert_dataset_v3/diffusion_v2/diffusion_train_v2.npz")
    parser.add_argument("--test_npz", default="data/cartesian_expert_dataset_v3/diffusion_v2/diffusion_test_v2.npz")
    parser.add_argument("--output_model", default="data/cartesian_expert_dataset_v3/diffusion_v2/conditional_diffusion_v2.pt")
    parser.add_argument("--latest_model", default="data/cartesian_expert_dataset_v3/diffusion_v2/conditional_diffusion_v2_latest.pt")
    parser.add_argument("--log_csv", default="data/cartesian_expert_dataset_v3/diffusion_v2/conditional_diffusion_v2_train_log.csv")
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


def sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    half_dim = dim // 2
    exponent = -math.log(10000.0) * torch.arange(half_dim, device=timesteps.device) / max(half_dim - 1, 1)
    freqs = torch.exp(exponent)
    angles = timesteps.float()[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
    if dim % 2 == 1:
        emb = torch.nn.functional.pad(emb, (0, 1))
    return emb


def choose_group_count(channels: int, preferred: int = 8) -> int:
    for groups in range(min(preferred, channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


class ResidualConvBlock(nn.Module):
    def __init__(self, hidden_dim: int, time_dim: int, groups: int = 8) -> None:
        super().__init__()
        groups = choose_group_count(hidden_dim, groups)
        self.norm1 = nn.GroupNorm(groups, hidden_dim)
        self.conv1 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(groups, hidden_dim)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_dim, hidden_dim)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        residual = x
        h = self.conv1(self.act(self.norm1(x)))
        h = h + self.time_proj(time_emb)[:, :, None]
        h = self.conv2(self.act(self.norm2(h)))
        return residual + h


class ConditionalTrajectoryDenoiser(nn.Module):
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
        self.input_proj = nn.Conv1d(condition_dim + target_dim, hidden_dim, kernel_size=1)
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        )
        self.blocks = nn.ModuleList([ResidualConvBlock(hidden_dim, hidden_dim) for _ in range(num_blocks)])
        self.out_norm = nn.GroupNorm(choose_group_count(hidden_dim), hidden_dim)
        self.out = nn.Conv1d(hidden_dim, target_dim, kernel_size=1)
        self.act = nn.SiLU()

    def forward(self, noisy_delta_q: torch.Tensor, condition: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        x = torch.cat([noisy_delta_q, condition], dim=-1).transpose(1, 2)
        time_emb = self.time_mlp(sinusoidal_embedding(timesteps, self.hidden_dim))
        h = self.input_proj(x)
        for block in self.blocks:
            h = block(h, time_emb)
        return self.out(self.act(self.out_norm(h))).transpose(1, 2)


def make_beta_schedule(num_steps: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    betas = torch.linspace(1e-4, 0.02, num_steps, device=device)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_bars


def load_dataset(path: Path) -> TensorDataset:
    data = np.load(path, allow_pickle=True)
    condition = torch.from_numpy(data["condition_features_norm"].astype(np.float32))
    target = torch.from_numpy(data["delta_q_norm"].astype(np.float32))
    return TensorDataset(condition, target)


def acceleration_loss(x0_pred: torch.Tensor) -> torch.Tensor:
    if x0_pred.shape[1] < 3:
        return x0_pred.new_tensor(0.0)
    accel = x0_pred[:, 2:, :] - 2.0 * x0_pred[:, 1:-1, :] + x0_pred[:, :-2, :]
    return torch.mean(accel**2)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    alpha_bars: torch.Tensor,
    device: torch.device,
    accel_loss_weight: float,
) -> tuple[float, float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss_sum = 0.0
    noise_loss_sum = 0.0
    accel_loss_sum = 0.0
    count = 0

    for condition, x0 in loader:
        condition = condition.to(device)
        x0 = x0.to(device)
        batch_size = x0.shape[0]
        t = torch.randint(0, alpha_bars.shape[0], (batch_size,), device=device)
        noise = torch.randn_like(x0)
        alpha_bar_t = alpha_bars[t].view(batch_size, 1, 1)
        sqrt_alpha_bar_t = torch.sqrt(alpha_bar_t)
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1.0 - alpha_bar_t)
        x_t = sqrt_alpha_bar_t * x0 + sqrt_one_minus_alpha_bar_t * noise

        with torch.set_grad_enabled(training):
            pred_noise = model(x_t, condition, t)
            loss_noise = torch.mean((pred_noise - noise) ** 2)
            loss_accel = x0.new_tensor(0.0)
            if accel_loss_weight > 0.0:
                x0_pred = (x_t - sqrt_one_minus_alpha_bar_t * pred_noise) / sqrt_alpha_bar_t
                loss_accel = acceleration_loss(x0_pred)
            loss = loss_noise + accel_loss_weight * loss_accel

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        total_loss_sum += float(loss.detach().cpu()) * batch_size
        noise_loss_sum += float(loss_noise.detach().cpu()) * batch_size
        accel_loss_sum += float(loss_accel.detach().cpu()) * batch_size
        count += batch_size

    return total_loss_sum / count, noise_loss_sum / count, accel_loss_sum / count


def save_checkpoint(
    path: Path,
    model: nn.Module,
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
        },
        path,
    )


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = select_device(args.device)

    train_dataset = load_dataset(Path(args.train_npz))
    test_dataset = load_dataset(Path(args.test_npz))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    model = ConditionalTrajectoryDenoiser(hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    _, _, alpha_bars = make_beta_schedule(args.num_diffusion_steps, device)

    log_path = Path(args.log_csv)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    last_val_loss = float("nan")

    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["epoch", "train_loss", "train_noise_loss", "train_accel_loss", "val_loss", "lr"],
        )
        writer.writeheader()

        for epoch in range(1, args.epochs + 1):
            train_loss, train_noise_loss, train_accel_loss = run_epoch(
                model, train_loader, optimizer, alpha_bars, device, args.accel_loss_weight
            )

            if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
                val_loss, _, _ = run_epoch(model, val_loader, None, alpha_bars, device, args.accel_loss_weight)
                last_val_loss = val_loss
            else:
                val_loss = last_val_loss

            save_checkpoint(Path(args.latest_model), model, optimizer, epoch, train_loss, val_loss, args)
            if np.isfinite(val_loss) and val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(Path(args.output_model), model, optimizer, epoch, train_loss, val_loss, args)

            writer.writerow(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_noise_loss": train_noise_loss,
                    "train_accel_loss": train_accel_loss,
                    "val_loss": val_loss,
                    "lr": optimizer.param_groups[0]["lr"],
                }
            )
            f.flush()


if __name__ == "__main__":
    main()
