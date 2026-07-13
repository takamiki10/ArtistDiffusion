#!/usr/bin/env python3
"""Train a deterministic v5c residual-window predictor.

This supervised residual-prior model maps:

    condition_norm -> residual_q_norm

and evaluates reconstructed candidates:

    predicted_residual_q = residual_q_norm * residual_std + residual_mean
    q_candidate_window = prior_q_window + predicted_residual_q

No diffusion, FK scoring, or receding-horizon stitching is performed here.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


DEFAULT_DATASET_DIR = Path("data/cartesian_expert_dataset_v3/diffusion_v5b_residual_windows_fk_condition")
DEFAULT_TRAIN_NPZ = DEFAULT_DATASET_DIR / "train_windows.npz"
DEFAULT_TEST_NPZ = DEFAULT_DATASET_DIR / "test_windows.npz"
DEFAULT_STATS_NPZ = DEFAULT_DATASET_DIR / "normalization_stats.npz"
DEFAULT_OUTPUT_DIR = Path("data/cartesian_expert_dataset_v3/diffusion_v5c_residual_predictor_fk_condition")
TARGET_DIM = 6
EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train deterministic v5c residual-window predictor."
    )
    parser.add_argument("--train_npz", type=Path, default=DEFAULT_TRAIN_NPZ)
    parser.add_argument("--test_npz", type=Path, default=DEFAULT_TEST_NPZ)
    parser.add_argument("--stats_npz", type=Path, default=DEFAULT_STATS_NPZ)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--w_vel", type=float, default=0.0)
    parser.add_argument("--w_acc", type=float, default=0.0)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--num_workers", type=int, default=0)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        print("[device] CUDA requested but unavailable; using CPU")
        return torch.device("cpu")
    return torch.device(device_arg)


def load_npz(path: Path, label: str) -> Dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def require_keys(data: Dict[str, np.ndarray], keys: Sequence[str], label: str) -> None:
    missing = [key for key in keys if key not in data]
    if missing:
        raise KeyError(f"{label} missing required key(s): {', '.join(missing)}")


def validate_split(
    data: Dict[str, np.ndarray],
    label: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    require_keys(
        data,
        ("condition_norm", "residual_q_norm", "prior_q_window", "expert_q_window"),
        label,
    )
    condition = np.asarray(data["condition_norm"], dtype=np.float32)
    target = np.asarray(data["residual_q_norm"], dtype=np.float32)
    prior_q = np.asarray(data["prior_q_window"], dtype=np.float32)
    expert_q = np.asarray(data["expert_q_window"], dtype=np.float32)

    if condition.ndim != 3:
        raise ValueError(f"{label}: condition_norm must have shape (N,H,C), got {condition.shape}")
    if target.ndim != 3 or target.shape[-1] != TARGET_DIM:
        raise ValueError(f"{label}: residual_q_norm must have shape (N,H,{TARGET_DIM}), got {target.shape}")
    if condition.shape[:2] != target.shape[:2]:
        raise ValueError(
            f"{label}: condition and target must share N,H, got "
            f"{condition.shape[:2]} vs {target.shape[:2]}"
        )
    if prior_q.shape != target.shape:
        raise ValueError(f"{label}: prior_q_window shape {prior_q.shape} must match target {target.shape}")
    if expert_q.shape != target.shape:
        raise ValueError(f"{label}: expert_q_window shape {expert_q.shape} must match target {target.shape}")
    for name, values in (
        ("condition_norm", condition),
        ("residual_q_norm", target),
        ("prior_q_window", prior_q),
        ("expert_q_window", expert_q),
    ):
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{label}: {name} contains non-finite values")
    return condition, target, prior_q, expert_q


def load_residual_stats(stats_npz: Path, train_data: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    stats: Dict[str, np.ndarray] = {}
    if stats_npz.exists():
        stats = load_npz(stats_npz, "normalization stats")
    if "residual_mean" in stats and "residual_std" in stats:
        mean = np.asarray(stats["residual_mean"], dtype=np.float32)
        std = np.asarray(stats["residual_std"], dtype=np.float32)
    elif "residual_mean" in train_data and "residual_std" in train_data:
        mean = np.asarray(train_data["residual_mean"], dtype=np.float32)
        std = np.asarray(train_data["residual_std"], dtype=np.float32)
    else:
        raise KeyError("Missing residual_mean/residual_std in stats_npz and train dataset")
    if mean.shape != (TARGET_DIM,) or std.shape != (TARGET_DIM,):
        raise ValueError(f"residual stats must have shape ({TARGET_DIM},), got {mean.shape}/{std.shape}")
    if np.any(std <= 0.0):
        raise ValueError("residual_std must be strictly positive")
    return mean, std


class ResidualWindowDataset(Dataset):
    def __init__(
        self,
        condition: np.ndarray,
        target: np.ndarray,
        prior_q: np.ndarray,
        expert_q: np.ndarray,
    ) -> None:
        self.condition = torch.from_numpy(condition.astype(np.float32))
        self.target = torch.from_numpy(target.astype(np.float32))
        self.prior_q = torch.from_numpy(prior_q.astype(np.float32))
        self.expert_q = torch.from_numpy(expert_q.astype(np.float32))

    def __len__(self) -> int:
        return int(self.target.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.condition[idx], self.target[idx], self.prior_q[idx], self.expert_q[idx]


def group_count(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class TemporalResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        padding = dilation
        self.net = nn.Sequential(
            nn.GroupNorm(group_count(channels), channels),
            nn.SiLU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=padding, dilation=dilation),
            nn.GroupNorm(group_count(channels), channels),
            nn.SiLU(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=padding, dilation=dilation),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class TemporalCNNResidualPredictor(nn.Module):
    def __init__(
        self,
        condition_dim: int,
        target_dim: int,
        hidden_dim: int,
        num_layers: int,
    ) -> None:
        super().__init__()
        if num_layers <= 0:
            raise ValueError("--num_layers must be positive")
        self.condition_dim = int(condition_dim)
        self.target_dim = int(target_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)

        self.input = nn.Conv1d(condition_dim, hidden_dim, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList(
            TemporalResidualBlock(hidden_dim, dilation=2 ** (idx % 4))
            for idx in range(num_layers)
        )
        self.output = nn.Sequential(
            nn.GroupNorm(group_count(hidden_dim), hidden_dim),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, target_dim, kernel_size=3, padding=1),
        )

    def forward(self, condition_bhc: torch.Tensor) -> torch.Tensor:
        x = condition_bhc.permute(0, 2, 1).contiguous()
        x = self.input(x)
        for block in self.blocks:
            x = block(x)
        return self.output(x).permute(0, 2, 1).contiguous()


def denormalize_residual_torch(
    residual_norm: torch.Tensor,
    residual_mean: torch.Tensor,
    residual_std: torch.Tensor,
) -> torch.Tensor:
    return residual_norm * residual_std.reshape(1, 1, TARGET_DIM) + residual_mean.reshape(1, 1, TARGET_DIM)


def smoothness_losses(residual_q: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    if residual_q.shape[1] < 2:
        zero = residual_q.new_tensor(0.0)
        return zero, zero
    velocity = residual_q[:, 1:, :] - residual_q[:, :-1, :]
    vel_loss = torch.mean(torch.square(velocity))
    if residual_q.shape[1] < 3:
        acc_loss = residual_q.new_tensor(0.0)
    else:
        acceleration = residual_q[:, 2:, :] - 2.0 * residual_q[:, 1:-1, :] + residual_q[:, :-2, :]
        acc_loss = torch.mean(torch.square(acceleration))
    return vel_loss, acc_loss


def train_epoch(
    *,
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    residual_mean: torch.Tensor,
    residual_std: torch.Tensor,
    w_vel: float,
    w_acc: float,
) -> float:
    model.train()
    total_loss = 0.0
    total_count = 0
    for condition, target, _, _ in loader:
        condition = condition.to(device=device, dtype=torch.float32)
        target = target.to(device=device, dtype=torch.float32)
        pred_norm = model(condition)
        mse_loss = F.mse_loss(pred_norm, target)
        loss = mse_loss
        if w_vel != 0.0 or w_acc != 0.0:
            pred_residual_q = denormalize_residual_torch(pred_norm, residual_mean, residual_std)
            vel_loss, acc_loss = smoothness_losses(pred_residual_q)
            loss = loss + float(w_vel) * vel_loss + float(w_acc) * acc_loss

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        batch_size = int(condition.shape[0])
        total_loss += float(loss.detach().cpu()) * batch_size
        total_count += batch_size
    return total_loss / max(total_count, 1)


def evaluate(
    *,
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    residual_mean: torch.Tensor,
    residual_std: torch.Tensor,
) -> Dict[str, float]:
    model.eval()
    total_mse = 0.0
    total_count = 0
    prior_rmse_values: List[torch.Tensor] = []
    candidate_rmse_values: List[torch.Tensor] = []
    residual_rmse_values: List[torch.Tensor] = []
    improved_values: List[torch.Tensor] = []
    improvement_values: List[torch.Tensor] = []

    with torch.no_grad():
        for condition, target, prior_q, expert_q in loader:
            condition = condition.to(device=device, dtype=torch.float32)
            target = target.to(device=device, dtype=torch.float32)
            prior_q = prior_q.to(device=device, dtype=torch.float32)
            expert_q = expert_q.to(device=device, dtype=torch.float32)
            pred_norm = model(condition)
            mse_loss = F.mse_loss(pred_norm, target)

            pred_residual_q = denormalize_residual_torch(pred_norm, residual_mean, residual_std)
            oracle_residual_q = expert_q - prior_q
            candidate_q = prior_q + pred_residual_q

            prior_rmse = torch.sqrt(torch.mean(torch.square(prior_q - expert_q), dim=(1, 2)))
            candidate_rmse = torch.sqrt(torch.mean(torch.square(candidate_q - expert_q), dim=(1, 2)))
            residual_rmse = torch.sqrt(torch.mean(torch.square(pred_residual_q - oracle_residual_q), dim=(1, 2)))
            improvement = 100.0 * (prior_rmse - candidate_rmse) / torch.clamp(prior_rmse, min=EPS)
            improved = candidate_rmse < prior_rmse

            batch_size = int(condition.shape[0])
            total_mse += float(mse_loss.detach().cpu()) * batch_size
            total_count += batch_size
            prior_rmse_values.append(prior_rmse.detach().cpu())
            candidate_rmse_values.append(candidate_rmse.detach().cpu())
            residual_rmse_values.append(residual_rmse.detach().cpu())
            improvement_values.append(improvement.detach().cpu())
            improved_values.append(improved.detach().cpu())

    prior_all = torch.cat(prior_rmse_values)
    candidate_all = torch.cat(candidate_rmse_values)
    residual_all = torch.cat(residual_rmse_values)
    improvement_all = torch.cat(improvement_values)
    improved_all = torch.cat(improved_values)
    improved_count = int(torch.sum(improved_all).item())
    total_windows = int(improved_all.numel())

    return {
        "test_loss": total_mse / max(total_count, 1),
        "prior_rmse": float(torch.mean(prior_all).item()),
        "candidate_rmse_to_expert": float(torch.mean(candidate_all).item()),
        "residual_rmse_to_oracle": float(torch.mean(residual_all).item()),
        "improvement_vs_prior_percent": float(torch.mean(improvement_all).item()),
        "improved_window_count": float(improved_count),
        "improved_window_ratio": float(improved_count / max(total_windows, 1)),
    }


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_loss: float,
    metrics: Dict[str, float],
    condition_dim: int,
    target_dim: int,
    horizon: int,
    model_config: Dict[str, Any],
    residual_mean: np.ndarray,
    residual_std: np.ndarray,
    train_npz: Path,
    test_npz: Path,
    stats_npz: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": int(epoch),
            "train_loss": float(train_loss),
            "test_loss": float(metrics["test_loss"]),
            "condition_dim": int(condition_dim),
            "target_dim": int(target_dim),
            "horizon": int(horizon),
            "model_config": model_config,
            "metrics": dict(metrics),
            "residual_mean": residual_mean.astype(np.float32),
            "residual_std": residual_std.astype(np.float32),
            "train_npz": str(train_npz),
            "test_npz": str(test_npz),
            "stats_npz": str(stats_npz),
        },
        path,
    )


def write_training_config(path: Path, config: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)


def init_log(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "epoch",
                "train_loss",
                "test_loss",
                "prior_rmse",
                "candidate_rmse_to_expert",
                "residual_rmse_to_oracle",
                "improvement_vs_prior_percent",
                "improved_window_count",
                "improved_window_ratio",
                "best_candidate_rmse_to_expert",
                "is_best",
                "lr",
            ]
        )


def append_log(
    path: Path,
    epoch: int,
    train_loss: float,
    metrics: Dict[str, float],
    best_candidate_rmse: float,
    is_best: bool,
    lr: float,
) -> None:
    with path.open("a", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                epoch,
                f"{train_loss:.12e}",
                f"{metrics['test_loss']:.12e}",
                f"{metrics['prior_rmse']:.12e}",
                f"{metrics['candidate_rmse_to_expert']:.12e}",
                f"{metrics['residual_rmse_to_oracle']:.12e}",
                f"{metrics['improvement_vs_prior_percent']:.12e}",
                int(metrics["improved_window_count"]),
                f"{metrics['improved_window_ratio']:.12e}",
                f"{best_candidate_rmse:.12e}",
                int(is_best),
                f"{lr:.12e}",
            ]
        )


def main() -> int:
    args = parse_args()
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive")
    if args.num_layers <= 0:
        raise ValueError("--num_layers must be positive")

    set_seed(args.seed)
    device = resolve_device(args.device)

    train_data = load_npz(args.train_npz, "train windows")
    test_data = load_npz(args.test_npz, "test windows")
    train_condition, train_target, train_prior, train_expert = validate_split(train_data, "train")
    test_condition, test_target, test_prior, test_expert = validate_split(test_data, "test")

    if train_condition.shape[1:] != test_condition.shape[1:]:
        raise ValueError(
            f"train/test condition shapes must share H,C, got {train_condition.shape[1:]} vs {test_condition.shape[1:]}"
        )
    if train_target.shape[1:] != test_target.shape[1:]:
        raise ValueError(
            f"train/test target shapes must share H,D, got {train_target.shape[1:]} vs {test_target.shape[1:]}"
        )

    residual_mean_np, residual_std_np = load_residual_stats(args.stats_npz, train_data)
    residual_mean = torch.from_numpy(residual_mean_np).to(device=device, dtype=torch.float32)
    residual_std = torch.from_numpy(residual_std_np).to(device=device, dtype=torch.float32)

    horizon = int(train_target.shape[1])
    target_dim = int(train_target.shape[2])
    condition_dim = int(train_condition.shape[2])

    train_loader = DataLoader(
        ResidualWindowDataset(train_condition, train_target, train_prior, train_expert),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        ResidualWindowDataset(test_condition, test_target, test_prior, test_expert),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = TemporalCNNResidualPredictor(
        condition_dim=condition_dim,
        target_dim=target_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    model_config = {
        "model_class": "TemporalCNNResidualPredictor",
        "condition_dim": condition_dim,
        "target_dim": target_dim,
        "hidden_dim": int(args.hidden_dim),
        "num_layers": int(args.num_layers),
        "architecture": "temporal_cnn_residual_blocks",
    }

    best_path = args.output_dir / "best_checkpoint.pt"
    latest_path = args.output_dir / "latest_checkpoint.pt"
    log_path = args.output_dir / "train_log.csv"
    config_path = args.output_dir / "training_config.json"

    write_training_config(
        config_path,
        {
            "train_npz": str(args.train_npz),
            "test_npz": str(args.test_npz),
            "stats_npz": str(args.stats_npz),
            "output_dir": str(args.output_dir),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "seed": int(args.seed),
            "device": str(device),
            "w_vel": float(args.w_vel),
            "w_acc": float(args.w_acc),
            "horizon": horizon,
            "condition_dim": condition_dim,
            "target_dim": target_dim,
            "train_shape": {
                "condition_norm": list(train_condition.shape),
                "residual_q_norm": list(train_target.shape),
            },
            "test_shape": {
                "condition_norm": list(test_condition.shape),
                "residual_q_norm": list(test_target.shape),
            },
            "model_config": model_config,
        },
    )
    init_log(log_path)

    best_candidate_rmse = float("inf")
    print(
        f"Loaded v5c residual windows: train={train_target.shape}, test={test_target.shape}, "
        f"condition_dim={condition_dim}, target_dim={target_dim}, horizon={horizon}, device={device}"
    )

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            residual_mean=residual_mean,
            residual_std=residual_std,
            w_vel=args.w_vel,
            w_acc=args.w_acc,
        )
        metrics = evaluate(
            model=model,
            loader=test_loader,
            device=device,
            residual_mean=residual_mean,
            residual_std=residual_std,
        )
        candidate_rmse = float(metrics["candidate_rmse_to_expert"])
        is_best = candidate_rmse < best_candidate_rmse
        if is_best:
            best_candidate_rmse = candidate_rmse

        save_checkpoint(
            latest_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            train_loss=train_loss,
            metrics=metrics,
            condition_dim=condition_dim,
            target_dim=target_dim,
            horizon=horizon,
            model_config=model_config,
            residual_mean=residual_mean_np,
            residual_std=residual_std_np,
            train_npz=args.train_npz,
            test_npz=args.test_npz,
            stats_npz=args.stats_npz,
        )
        if is_best:
            save_checkpoint(
                best_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                train_loss=train_loss,
                metrics=metrics,
                condition_dim=condition_dim,
                target_dim=target_dim,
                horizon=horizon,
                model_config=model_config,
                residual_mean=residual_mean_np,
                residual_std=residual_std_np,
                train_npz=args.train_npz,
                test_npz=args.test_npz,
                stats_npz=args.stats_npz,
            )

        append_log(log_path, epoch, train_loss, metrics, best_candidate_rmse, is_best, args.lr)
        print(
            f"epoch {epoch:04d} | train_loss={train_loss:.8e} | "
            f"test_loss={metrics['test_loss']:.8e} | "
            f"candidate_rmse={metrics['candidate_rmse_to_expert']:.8e} | "
            f"improvement={metrics['improvement_vs_prior_percent']:.3f}% | "
            f"best_candidate_rmse={best_candidate_rmse:.8e}"
        )

    print(f"Saved best checkpoint: {best_path}")
    print(f"Saved latest checkpoint: {latest_path}")
    print(f"Saved train log: {log_path}")
    print(f"Saved training config: {config_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
