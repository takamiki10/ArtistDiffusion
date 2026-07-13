#!/usr/bin/env python3
"""Train v5 residual-window conditional diffusion with a 1D U-Net.

This is the v5 residual receding-horizon model. The diffusion target is
`residual_q_norm` windows, not full q, expert q, or delta_q.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import inspect
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


DEFAULT_DATASET_DIR = Path("data/cartesian_expert_dataset_v3/diffusion_v5_residual_windows")
DEFAULT_TRAIN_NPZ = DEFAULT_DATASET_DIR / "train_windows.npz"
DEFAULT_TEST_NPZ = DEFAULT_DATASET_DIR / "test_windows.npz"
DEFAULT_OUTPUT_DIR = DEFAULT_DATASET_DIR / "residual_unet"
DEFAULT_CONDITION_DIM = 31
EXPECTED_CONDITION_DIM = DEFAULT_CONDITION_DIM
EXPECTED_TARGET_DIM = 6
PROJECT_MODEL_MODULE = "conditional_unet1d_artist"
PROJECT_MODEL_CLASS_NAMES = (
    "ConditionalUNet1D",
    "ConditionalUnet1D",
    "ConditionalUNet1DArtist",
    "ConditionalUnet1DArtist",
    "ArtistConditionalUNet1D",
)
CALL_VARIANTS = (
    "cf_x_cond_t",
    "cf_x_t_cond",
    "cf_x_t_condition_kw",
    "cf_x_t_cond_kw",
    "cf_x_t_context_kw",
    "tf_x_cond_t",
    "tf_x_t_cond",
    "tf_x_t_condition_kw",
    "tf_x_t_cond_kw",
    "tf_x_t_context_kw",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train v5 residual-window conditional U-Net diffusion."
    )
    parser.add_argument("--train_npz", type=Path, default=DEFAULT_TRAIN_NPZ)
    parser.add_argument("--test_npz", type=Path, default=DEFAULT_TEST_NPZ)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--num_diffusion_steps", type=int, default=1000)
    parser.add_argument("--base_channels", "--model_channels", dest="base_channels", type=int, default=64)
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--weight_decay", type=float, default=1e-6)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--beta_start", type=float, default=1e-4)
    parser.add_argument("--beta_end", type=float, default=2e-2)
    parser.add_argument(
        "--force_local_model",
        action="store_true",
        help="Skip conditional_unet1d_artist import and use the local v5 U-Net.",
    )
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


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Missing residual-window dataset: {path}")
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def require_keys(data: Dict[str, np.ndarray], keys: Sequence[str], label: str) -> None:
    missing = [key for key in keys if key not in data]
    if missing:
        raise KeyError(f"{label} missing required key(s): {', '.join(missing)}")


def load_split(path: Path, label: str) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    data = load_npz(path)
    require_keys(data, ("condition_norm", "residual_q_norm"), label)
    condition = np.asarray(data["condition_norm"], dtype=np.float32)
    target = np.asarray(data["residual_q_norm"], dtype=np.float32)
    validate_shapes(condition, target, label)
    if not np.all(np.isfinite(condition)):
        raise ValueError(f"{label}: condition_norm contains non-finite values")
    if not np.all(np.isfinite(target)):
        raise ValueError(f"{label}: residual_q_norm contains non-finite values")
    return condition, target, data


def validate_shapes(condition: np.ndarray, target: np.ndarray, label: str) -> None:
    if condition.ndim != 3:
        raise ValueError(f"{label}: condition_norm must have shape (N,H,C), got {condition.shape}")
    if target.ndim != 3:
        raise ValueError(f"{label}: residual_q_norm must have shape (N,H,6), got {target.shape}")
    if condition.shape[:2] != target.shape[:2]:
        raise ValueError(
            f"{label}: condition and residual must share N,H, got "
            f"{condition.shape[:2]} vs {target.shape[:2]}"
        )
    if condition.shape[-1] <= 0:
        raise ValueError(f"{label}: condition_dim must be positive, got {condition.shape[-1]}")
    if target.shape[-1] != EXPECTED_TARGET_DIM:
        raise ValueError(f"{label}: target_dim should be {EXPECTED_TARGET_DIM}, got {target.shape[-1]}")


class ResidualWindowDataset(Dataset):
    def __init__(self, condition: np.ndarray, target: np.ndarray) -> None:
        self.condition = torch.from_numpy(condition.astype(np.float32))
        self.target = torch.from_numpy(target.astype(np.float32))

    def __len__(self) -> int:
        return int(self.target.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.target[idx], self.condition[idx]


class DiffusionSchedule:
    def __init__(
        self,
        num_steps: int,
        beta_start: float,
        beta_end: float,
        device: torch.device,
    ) -> None:
        if num_steps <= 0:
            raise ValueError("--num_diffusion_steps must be positive")
        betas = torch.linspace(beta_start, beta_end, num_steps, dtype=torch.float32, device=device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.num_steps = int(num_steps)
        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    def q_sample(self, x0: torch.Tensor, timesteps: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        sqrt_alpha = extract(self.sqrt_alphas_cumprod, timesteps, x0)
        sqrt_one_minus_alpha = extract(self.sqrt_one_minus_alphas_cumprod, timesteps, x0)
        return sqrt_alpha * x0 + sqrt_one_minus_alpha * noise


def extract(values: torch.Tensor, timesteps: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    gathered = values.gather(0, timesteps)
    return gathered.reshape(timesteps.shape[0], *([1] * (target.ndim - 1)))


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        if half_dim == 0:
            return timesteps.float().unsqueeze(1)
        scale = math.log(10000.0) / max(half_dim - 1, 1)
        freqs = torch.exp(torch.arange(half_dim, device=timesteps.device, dtype=torch.float32) * -scale)
        args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


def group_count(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(group_count(in_channels), in_channels)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(group_count(out_channels), out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_dim, out_channels))
        self.skip = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_mlp(time_emb).unsqueeze(-1)
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class Downsample1D(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample1D(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose1d(channels, channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class LocalResidualConditionalUNet1D(nn.Module):
    """Compact conditional 1D U-Net for residual windows.

    Inputs and outputs use channel-first tensors:
        x:         (B, target_dim, H)
        condition: (B, condition_dim, H)
        timesteps: (B,)
    """

    def __init__(
        self,
        target_dim: int,
        condition_dim: int,
        base_channels: int,
        horizon: int,
    ) -> None:
        super().__init__()
        if horizon % 8 != 0:
            raise ValueError(f"Local U-Net expects horizon divisible by 8, got {horizon}")

        time_dim = base_channels * 4
        self.target_dim = int(target_dim)
        self.condition_dim = int(condition_dim)
        self.base_channels = int(base_channels)
        self.horizon = int(horizon)

        self.time_embedding = nn.Sequential(
            SinusoidalTimeEmbedding(base_channels),
            nn.Linear(base_channels, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        in_channels = target_dim + condition_dim
        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4

        self.init_conv = nn.Conv1d(in_channels, c1, kernel_size=3, padding=1)
        self.down1 = ResidualBlock1D(c1, c1, time_dim)
        self.downsample1 = Downsample1D(c1)
        self.down2 = ResidualBlock1D(c1, c2, time_dim)
        self.downsample2 = Downsample1D(c2)
        self.down3 = ResidualBlock1D(c2, c3, time_dim)
        self.downsample3 = Downsample1D(c3)

        self.mid1 = ResidualBlock1D(c3, c3, time_dim)
        self.mid2 = ResidualBlock1D(c3, c3, time_dim)

        self.upsample3 = Upsample1D(c3)
        self.up3 = ResidualBlock1D(c3 + c3, c2, time_dim)
        self.upsample2 = Upsample1D(c2)
        self.up2 = ResidualBlock1D(c2 + c2, c1, time_dim)
        self.upsample1 = Upsample1D(c1)
        self.up1 = ResidualBlock1D(c1 + c1, c1, time_dim)

        self.final = nn.Sequential(
            nn.GroupNorm(group_count(c1), c1),
            nn.SiLU(),
            nn.Conv1d(c1, target_dim, kernel_size=3, padding=1),
        )

    def forward(
        self,
        x: torch.Tensor,
        condition: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        if x.shape[-1] != condition.shape[-1]:
            raise ValueError(f"x and condition horizon mismatch: {x.shape} vs {condition.shape}")
        time_emb = self.time_embedding(timesteps)
        h = self.init_conv(torch.cat([x, condition], dim=1))

        skip1 = self.down1(h, time_emb)
        h = self.downsample1(skip1)
        skip2 = self.down2(h, time_emb)
        h = self.downsample2(skip2)
        skip3 = self.down3(h, time_emb)
        h = self.downsample3(skip3)

        h = self.mid1(h, time_emb)
        h = self.mid2(h, time_emb)

        h = self.upsample3(h)
        h = self.up3(torch.cat([h, skip3], dim=1), time_emb)
        h = self.upsample2(h)
        h = self.up2(torch.cat([h, skip2], dim=1), time_emb)
        h = self.upsample1(h)
        h = self.up1(torch.cat([h, skip1], dim=1), time_emb)
        return self.final(h)


def project_model_classes() -> Iterable[type]:
    module = importlib.import_module(PROJECT_MODEL_MODULE)
    yielded = set()
    for name in PROJECT_MODEL_CLASS_NAMES:
        cls = getattr(module, name, None)
        if inspect.isclass(cls) and issubclass(cls, nn.Module):
            yielded.add(cls)
            yield cls
    for _, cls in inspect.getmembers(module, inspect.isclass):
        if cls in yielded:
            continue
        if issubclass(cls, nn.Module) and "UNet" in cls.__name__:
            yielded.add(cls)
            yield cls


def model_init_attempts(
    condition_dim: int,
    target_dim: int,
    horizon: int,
    base_channels: int,
    num_diffusion_steps: int,
) -> List[Dict[str, Any]]:
    return [
        {
            "target_dim": target_dim,
            "condition_dim": condition_dim,
            "horizon": horizon,
            "base_channels": base_channels,
        },
        {
            "target_dim": target_dim,
            "condition_dim": condition_dim,
            "horizon": horizon,
            "model_channels": base_channels,
        },
        {
            "in_channels": target_dim,
            "condition_channels": condition_dim,
            "horizon": horizon,
            "base_channels": base_channels,
        },
        {
            "input_channels": target_dim,
            "condition_channels": condition_dim,
            "model_channels": base_channels,
        },
        {
            "input_dim": target_dim,
            "condition_dim": condition_dim,
            "model_channels": base_channels,
        },
        {
            "trajectory_dim": target_dim,
            "condition_dim": condition_dim,
            "base_channels": base_channels,
            "num_diffusion_steps": num_diffusion_steps,
        },
    ]


def call_model_variant(
    model: nn.Module,
    variant: str,
    x_cf: torch.Tensor,
    condition_cf: torch.Tensor,
    timesteps: torch.Tensor,
) -> torch.Tensor:
    if variant.startswith("cf_"):
        x = x_cf
        condition = condition_cf
    elif variant.startswith("tf_"):
        x = x_cf.permute(0, 2, 1).contiguous()
        condition = condition_cf.permute(0, 2, 1).contiguous()
    else:
        raise ValueError(f"Unknown model call variant: {variant}")

    if variant.endswith("x_cond_t"):
        output = model(x, condition, timesteps)
    elif variant.endswith("x_t_cond"):
        output = model(x, timesteps, condition)
    elif variant.endswith("x_t_condition_kw"):
        output = model(x, timesteps, condition=condition)
    elif variant.endswith("x_t_cond_kw"):
        output = model(x, timesteps, cond=condition)
    elif variant.endswith("x_t_context_kw"):
        output = model(x, timesteps, context=condition)
    else:
        raise ValueError(f"Unknown model call variant: {variant}")

    if isinstance(output, tuple):
        output = output[0]
    if not torch.is_tensor(output):
        raise TypeError(f"Model output must be a tensor, got {type(output)!r}")
    if output.shape == x_cf.shape:
        return output
    if output.shape == x.shape and variant.startswith("tf_"):
        return output.permute(0, 2, 1).contiguous()
    raise ValueError(f"Model output shape {tuple(output.shape)} does not match target {tuple(x_cf.shape)}")


def resolve_call_variant(
    model: nn.Module,
    condition_dim: int,
    target_dim: int,
    horizon: int,
    device: torch.device,
) -> str:
    model = model.to(device)
    x = torch.randn(2, target_dim, horizon, device=device)
    condition = torch.randn(2, condition_dim, horizon, device=device)
    timesteps = torch.randint(0, 10, (2,), device=device)
    model.eval()
    with torch.no_grad():
        for variant in CALL_VARIANTS:
            try:
                output = call_model_variant(model, variant, x, condition, timesteps)
            except Exception:
                continue
            if output.shape == x.shape:
                return variant
    raise RuntimeError("Could not find a compatible forward signature for the project U-Net")


def build_model(
    *,
    condition_dim: int,
    target_dim: int,
    horizon: int,
    base_channels: int,
    num_diffusion_steps: int,
    device: torch.device,
    force_local_model: bool,
) -> Tuple[nn.Module, str, Dict[str, Any]]:
    if not force_local_model:
        try:
            for cls in project_model_classes():
                for kwargs in model_init_attempts(
                    condition_dim,
                    target_dim,
                    horizon,
                    base_channels,
                    num_diffusion_steps,
                ):
                    try:
                        model = cls(**kwargs)
                    except Exception:
                        continue
                    try:
                        call_variant = resolve_call_variant(
                            model,
                            condition_dim,
                            target_dim,
                            horizon,
                            device,
                        )
                    except Exception:
                        continue
                    config = {
                        "model_source": "project",
                        "model_module": cls.__module__,
                        "model_class": cls.__name__,
                        "init_kwargs": kwargs,
                        "call_variant": call_variant,
                    }
                    print(f"[model] using {cls.__module__}.{cls.__name__} ({call_variant})")
                    return model.to(device), call_variant, config
        except Exception as exc:
            print(f"[model] project U-Net unavailable; using local fallback: {exc}")

    model = LocalResidualConditionalUNet1D(
        target_dim=target_dim,
        condition_dim=condition_dim,
        base_channels=base_channels,
        horizon=horizon,
    )
    config = {
        "model_source": "local",
        "model_module": __name__,
        "model_class": "LocalResidualConditionalUNet1D",
        "init_kwargs": {
            "target_dim": target_dim,
            "condition_dim": condition_dim,
            "base_channels": base_channels,
            "horizon": horizon,
        },
        "call_variant": "cf_x_cond_t",
    }
    print("[model] using local LocalResidualConditionalUNet1D")
    return model.to(device), "cf_x_cond_t", config


def batch_to_channel_first(
    target_bhw: torch.Tensor,
    condition_bhw: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    target = target_bhw.to(device=device, dtype=torch.float32).permute(0, 2, 1).contiguous()
    condition = condition_bhw.to(device=device, dtype=torch.float32).permute(0, 2, 1).contiguous()
    return target, condition


def run_epoch(
    *,
    model: nn.Module,
    loader: DataLoader,
    schedule: DiffusionSchedule,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    call_variant: str,
) -> float:
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_count = 0

    grad_context = torch.enable_grad() if is_train else torch.no_grad()
    with grad_context:
        for target_bhw, condition_bhw in loader:
            x0, condition = batch_to_channel_first(target_bhw, condition_bhw, device)
            batch_size = x0.shape[0]
            timesteps = torch.randint(0, schedule.num_steps, (batch_size,), device=device)
            noise = torch.randn_like(x0)
            x_t = schedule.q_sample(x0, timesteps, noise)
            pred_noise = call_model_variant(model, call_variant, x_t, condition, timesteps)
            loss = F.mse_loss(pred_noise, noise)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            total_loss += float(loss.detach().cpu()) * batch_size
            total_count += batch_size

    return total_loss / max(total_count, 1)


def stats_for_checkpoint(data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    out: Dict[str, np.ndarray] = {}
    for key in ("condition_mean", "condition_std", "residual_mean", "residual_std"):
        if key in data:
            out[key] = np.asarray(data[key], dtype=np.float32)
    return out


def save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    train_loss: float,
    test_loss: float,
    best_test_loss: float,
    condition_dim: int,
    target_dim: int,
    horizon: int,
    num_diffusion_steps: int,
    model_config: Dict[str, Any],
    diffusion_config: Dict[str, Any],
    train_npz: Path,
    test_npz: Path,
    normalization_stats: Dict[str, np.ndarray],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": int(epoch),
        "train_loss": float(train_loss),
        "test_loss": float(test_loss),
        "best_test_loss": float(best_test_loss),
        "condition_dim": int(condition_dim),
        "target_dim": int(target_dim),
        "horizon": int(horizon),
        "num_diffusion_steps": int(num_diffusion_steps),
        "model_config": model_config,
        "diffusion_config": diffusion_config,
        "train_npz": str(train_npz),
        "test_npz": str(test_npz),
        **normalization_stats,
    }
    torch.save(checkpoint, path)


def write_training_config(path: Path, config: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, sort_keys=True)


def init_log(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "train_loss", "test_loss", "best_test_loss", "is_best", "lr"])


def append_log(
    path: Path,
    epoch: int,
    train_loss: float,
    test_loss: float,
    best_test_loss: float,
    is_best: bool,
    lr: float,
) -> None:
    with path.open("a", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                epoch,
                f"{train_loss:.12e}",
                f"{test_loss:.12e}",
                f"{best_test_loss:.12e}",
                int(is_best),
                f"{lr:.12e}",
            ]
        )


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    device = resolve_device(args.device)

    train_condition, train_target, train_data = load_split(args.train_npz, "train")
    test_condition, test_target, test_data = load_split(args.test_npz, "test")
    if train_condition.shape[1:] != test_condition.shape[1:]:
        raise ValueError(
            f"train/test condition windows must share H,D, got "
            f"{train_condition.shape[1:]} vs {test_condition.shape[1:]}"
        )
    if train_target.shape[1:] != test_target.shape[1:]:
        raise ValueError(
            f"train/test residual windows must share H,D, got "
            f"{train_target.shape[1:]} vs {test_target.shape[1:]}"
        )

    horizon = int(train_target.shape[1])
    target_dim = int(train_target.shape[2])
    condition_dim = int(train_condition.shape[2])

    train_loader = DataLoader(
        ResidualWindowDataset(train_condition, train_target),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        ResidualWindowDataset(test_condition, test_target),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model, call_variant, model_config = build_model(
        condition_dim=condition_dim,
        target_dim=target_dim,
        horizon=horizon,
        base_channels=args.base_channels,
        num_diffusion_steps=args.num_diffusion_steps,
        device=device,
        force_local_model=args.force_local_model,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    schedule = DiffusionSchedule(
        num_steps=args.num_diffusion_steps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        device=device,
    )
    diffusion_config = {
        "prediction_type": "epsilon",
        "beta_schedule": "linear",
        "beta_start": float(args.beta_start),
        "beta_end": float(args.beta_end),
        "num_diffusion_steps": int(args.num_diffusion_steps),
        "target_key": "residual_q_norm",
        "condition_key": "condition_norm",
    }

    best_path = args.output_dir / "best_checkpoint.pt"
    latest_path = args.output_dir / "latest_checkpoint.pt"
    log_path = args.output_dir / "train_log.csv"
    config_path = args.output_dir / "training_config.json"
    normalization_stats = stats_for_checkpoint(train_data)

    write_training_config(
        config_path,
        {
            "train_npz": str(args.train_npz),
            "test_npz": str(args.test_npz),
            "output_dir": str(args.output_dir),
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "weight_decay": float(args.weight_decay),
            "seed": int(args.seed),
            "device": str(device),
            "train_shape": {
                "condition_norm": list(train_condition.shape),
                "residual_q_norm": list(train_target.shape),
            },
            "test_shape": {
                "condition_norm": list(test_condition.shape),
                "residual_q_norm": list(test_target.shape),
            },
            "condition_dim": condition_dim,
            "target_dim": target_dim,
            "horizon": horizon,
            "model_config": model_config,
            "diffusion_config": diffusion_config,
        },
    )
    init_log(log_path)

    best_test_loss = float("inf")
    print(
        f"Loaded v5 residual windows: train={train_target.shape}, test={test_target.shape}, "
        f"condition_dim={condition_dim}, target_dim={target_dim}, horizon={horizon}, device={device}"
    )

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(
            model=model,
            loader=train_loader,
            schedule=schedule,
            optimizer=optimizer,
            device=device,
            call_variant=call_variant,
        )
        test_loss = run_epoch(
            model=model,
            loader=test_loader,
            schedule=schedule,
            optimizer=None,
            device=device,
            call_variant=call_variant,
        )
        is_best = test_loss < best_test_loss
        if is_best:
            best_test_loss = test_loss

        save_checkpoint(
            latest_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            train_loss=train_loss,
            test_loss=test_loss,
            best_test_loss=best_test_loss,
            condition_dim=condition_dim,
            target_dim=target_dim,
            horizon=horizon,
            num_diffusion_steps=args.num_diffusion_steps,
            model_config=model_config,
            diffusion_config=diffusion_config,
            train_npz=args.train_npz,
            test_npz=args.test_npz,
            normalization_stats=normalization_stats,
        )
        if is_best:
            save_checkpoint(
                best_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                train_loss=train_loss,
                test_loss=test_loss,
                best_test_loss=best_test_loss,
                condition_dim=condition_dim,
                target_dim=target_dim,
                horizon=horizon,
                num_diffusion_steps=args.num_diffusion_steps,
                model_config=model_config,
                diffusion_config=diffusion_config,
                train_npz=args.train_npz,
                test_npz=args.test_npz,
                normalization_stats=normalization_stats,
            )

        append_log(log_path, epoch, train_loss, test_loss, best_test_loss, is_best, args.lr)
        print(
            f"epoch {epoch:04d} | train_loss={train_loss:.8e} | "
            f"test_loss={test_loss:.8e} | best_test_loss={best_test_loss:.8e}"
        )

    print(f"Saved best checkpoint: {best_path}")
    print(f"Saved latest checkpoint: {latest_path}")
    print(f"Saved train log: {log_path}")
    print(f"Saved training config: {config_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
