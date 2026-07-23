#!/usr/bin/env python3
"""Train v6 conditional diffusion on strong-prior residual windows only."""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib
import inspect
import json
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


EXPECTED_TRAIN_WINDOWS = 6696
EXPECTED_VAL_WINDOWS = 756
EXPECTED_HORIZON = 32
EXPECTED_TARGET_DIM = 6
EXPECTED_CONDITION_DIM = 38
EXPECTED_WINDOWS_PER_PATH = 18
EXPECTED_WINDOW_STARTS = np.arange(0, 69, 4, dtype=np.int64)
PREDICTION_TARGET = "epsilon"
EXPECTED_CONDITION_LAYOUT = (
    "desired_xyz(3)",
    "desired_dxyz(3)",
    "progress(1)",
    "prior_q_start_repeated(6)",
    "prior_current_q_repeated(6)",
    "prior_q_window(6)",
    "prior_delta_from_start(6)",
    "prior_ee_xyz(3)",
    "prior_ee_error_xyz(3)",
    "prior_ee_error_norm(1)",
)


@dataclass(frozen=True)
class WindowArrays:
    source: Path
    keys: Tuple[str, ...]
    condition: np.ndarray
    target: np.ndarray
    path_names: Tuple[str, ...]
    window_starts: np.ndarray


@dataclass(frozen=True)
class DiffusionSchedule:
    betas: torch.Tensor
    alphas: torch.Tensor
    alpha_bars: torch.Tensor

    def to(self, device: torch.device) -> "DiffusionSchedule":
        return DiffusionSchedule(
            betas=self.betas.to(device),
            alphas=self.alphas.to(device),
            alpha_bars=self.alpha_bars.to(device),
        )


class ResidualWindowDataset(Dataset[Tuple[torch.Tensor, torch.Tensor, int]]):
    def __init__(self, arrays: WindowArrays) -> None:
        self.condition = torch.from_numpy(arrays.condition.astype(np.float32))
        self.target = torch.from_numpy(arrays.target.astype(np.float32))

    def __len__(self) -> int:
        return int(self.target.shape[0])

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        return self.condition[index], self.target[index], index


class ExponentialMovingAverage:
    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = float(decay)
        self.shadow = {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        current = model.state_dict()
        for name, shadow_value in self.shadow.items():
            model_value = current[name].detach()
            if torch.is_floating_point(shadow_value):
                shadow_value.mul_(self.decay).add_(
                    model_value, alpha=1.0 - self.decay
                )
            else:
                shadow_value.copy_(model_value)

    def state_dict(self) -> Dict[str, Any]:
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.decay = float(state["decay"])
        self.shadow = {
            str(name): value.detach().clone()
            for name, value in state["shadow"].items()
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train v6 strong-prior conditional residual diffusion."
    )
    parser.add_argument("--train_npz", type=Path, required=True)
    parser.add_argument("--val_npz", type=Path, required=True)
    parser.add_argument("--normalization_stats", type=Path, required=True)
    parser.add_argument("--dataset_configuration", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=1.0e-4)
    parser.add_argument("--diffusion_steps", type=int, default=1000)
    parser.add_argument("--validation_interval", type=int, default=1)
    parser.add_argument("--checkpoint_interval", type=int, default=25)
    parser.add_argument("--early_stopping_patience", type=int, default=75)
    parser.add_argument("--gradient_clip_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--resume_checkpoint", type=Path, default=None)
    parser.add_argument("--max_train_batches", type=int, default=0)
    parser.add_argument("--max_val_batches", type=int, default=0)
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument(
        "--use_ema", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument(
        "--save_last", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--deterministic", action="store_true")
    return parser.parse_args()


def validate_cli(args: argparse.Namespace) -> None:
    positive_integer_fields = (
        "epochs",
        "batch_size",
        "diffusion_steps",
        "validation_interval",
        "checkpoint_interval",
        "early_stopping_patience",
    )
    for field in positive_integer_fields:
        if int(getattr(args, field)) <= 0:
            raise ValueError(f"--{field} must be positive")
    if args.learning_rate <= 0.0:
        raise ValueError("--learning_rate must be positive")
    if args.gradient_clip_norm < 0.0:
        raise ValueError("--gradient_clip_norm must be non-negative")
    if args.num_workers < 0:
        raise ValueError("--num_workers must be non-negative")
    if args.weight_decay < 0.0:
        raise ValueError("--weight_decay must be non-negative")
    if args.max_train_batches < 0 or args.max_val_batches < 0:
        raise ValueError("Maximum batch counts must be non-negative")
    if not 0.0 < args.ema_decay < 1.0:
        raise ValueError("--ema_decay must lie in (0,1)")
    if args.train_npz.resolve() == args.val_npz.resolve():
        raise ValueError("Training and validation NPZ files must be different")


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(requested)


def set_reproducibility(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True


def decode_names(values: np.ndarray) -> Tuple[str, ...]:
    result: List[str] = []
    for value in np.asarray(values).reshape(-1):
        if isinstance(value, bytes):
            result.append(value.decode("utf-8", errors="strict"))
        else:
            result.append(str(value))
    return tuple(result)


def print_npz_schema(path: Path, archive: Any) -> None:
    print(f"{path} keys and shapes:")
    for key in archive.files:
        print(f"  {key}: shape={archive[key].shape}, dtype={archive[key].dtype}")


def load_windows(path: Path, expected_count: int, label: str) -> WindowArrays:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=True) as archive:
        print_npz_schema(path, archive)
        required = (
            "condition_features_norm",
            "residual_q_norm",
            "path_names",
            "window_starts",
        )
        missing = [key for key in required if key not in archive.files]
        if missing:
            raise KeyError(f"{path} is missing required keys {missing}")
        arrays = WindowArrays(
            source=path,
            keys=tuple(archive.files),
            condition=np.asarray(
                archive["condition_features_norm"], dtype=np.float32
            ),
            target=np.asarray(archive["residual_q_norm"], dtype=np.float32),
            path_names=decode_names(archive["path_names"]),
            window_starts=np.asarray(archive["window_starts"], dtype=np.int64),
        )
    if arrays.condition.shape != (
        expected_count,
        EXPECTED_HORIZON,
        EXPECTED_CONDITION_DIM,
    ):
        raise ValueError(
            f"{label} condition shape {arrays.condition.shape}, expected "
            f"({expected_count},{EXPECTED_HORIZON},{EXPECTED_CONDITION_DIM})"
        )
    if arrays.target.shape != (
        expected_count,
        EXPECTED_HORIZON,
        EXPECTED_TARGET_DIM,
    ):
        raise ValueError(
            f"{label} target shape {arrays.target.shape}, expected "
            f"({expected_count},{EXPECTED_HORIZON},{EXPECTED_TARGET_DIM})"
        )
    if len(arrays.path_names) != expected_count:
        raise ValueError(f"{label} path_names length does not match window count")
    if arrays.window_starts.shape != (expected_count,):
        raise ValueError(f"{label} window_starts must have shape ({expected_count},)")
    if not np.all(np.isfinite(arrays.condition)):
        raise ValueError(f"{label} condition contains nonfinite values")
    if not np.all(np.isfinite(arrays.target)):
        raise ValueError(f"{label} residual target contains nonfinite values")
    validate_path_windows(arrays, label)
    return arrays


def validate_path_windows(arrays: WindowArrays, label: str) -> None:
    unique_names = sorted(set(arrays.path_names))
    for path_name in unique_names:
        indices = np.flatnonzero(np.asarray(arrays.path_names) == path_name)
        if len(indices) != EXPECTED_WINDOWS_PER_PATH:
            raise ValueError(
                f"{label}/{path_name} has {len(indices)} windows, "
                f"expected {EXPECTED_WINDOWS_PER_PATH}"
            )
        starts = arrays.window_starts[indices]
        if not np.array_equal(starts, EXPECTED_WINDOW_STARTS):
            raise ValueError(
                f"{label}/{path_name} starts {starts.tolist()}, expected "
                f"{EXPECTED_WINDOW_STARTS.tolist()}"
            )


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def load_normalization(path: Path) -> Dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        required = ("condition_mean", "condition_std", "residual_mean", "residual_std")
        missing = [key for key in required if key not in archive.files]
        if missing:
            raise KeyError(f"{path} is missing normalization keys {missing}")
        values = {key: np.asarray(archive[key]) for key in archive.files}
    for key in required:
        if not np.all(np.isfinite(values[key])):
            raise ValueError(f"{path}/{key} contains nonfinite values")
    if np.any(values["condition_std"] <= 0.0) or np.any(
        values["residual_std"] <= 0.0
    ):
        raise ValueError("Normalization standard deviations must be positive")
    return values


def validate_dataset_metadata(
    configuration: Mapping[str, Any], normalization: Mapping[str, np.ndarray]
) -> None:
    if configuration.get("classification") != "READY_FOR_V6_TRAINING":
        raise ValueError("Dataset classification is not READY_FOR_V6_TRAINING")
    if configuration.get("normalization_source") != "supervised_train_paths_only":
        raise ValueError(
            "Dataset normalization_source is not supervised_train_paths_only"
        )
    expected = {
        "horizon": EXPECTED_HORIZON,
        "condition_dim": EXPECTED_CONDITION_DIM,
        "target_dim": EXPECTED_TARGET_DIM,
    }
    for key, value in expected.items():
        if int(configuration.get(key, -1)) != value:
            raise ValueError(
                f"Dataset configuration {key}={configuration.get(key)}, expected {value}"
            )
    if tuple(configuration.get("condition_feature_layout", ())) != (
        EXPECTED_CONDITION_LAYOUT
    ):
        raise ValueError("Dataset condition feature layout is not the v5b 38-D layout")
    scalar_checks = {
        "train_path_count": 372,
        "train_window_count": EXPECTED_TRAIN_WINDOWS,
        "condition_dim": EXPECTED_CONDITION_DIM,
        "target_dim": EXPECTED_TARGET_DIM,
    }
    for key, expected_value in scalar_checks.items():
        if key not in normalization:
            raise KeyError(f"Normalization metadata is missing {key}")
        if int(np.asarray(normalization[key]).item()) != expected_value:
            raise ValueError(
                f"Normalization metadata {key} is not {expected_value}"
            )


def path_set_hash(names: Iterable[str]) -> str:
    payload = "\n".join(sorted(set(names))).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def distribution_metrics(values: np.ndarray) -> Dict[str, float]:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    absolute = np.abs(flat)
    return {
        "mean": float(np.mean(flat)),
        "standard_deviation": float(np.std(flat)),
        "median_absolute_value": float(np.median(absolute)),
        "p90_absolute_value": float(np.percentile(absolute, 90.0)),
        "p95_absolute_value": float(np.percentile(absolute, 95.0)),
        "p99_absolute_value": float(np.percentile(absolute, 99.0)),
        "maximum_absolute_value": float(np.max(absolute)),
    }


def save_residual_statistics(
    train: WindowArrays,
    validation: WindowArrays,
    residual_mean: np.ndarray,
    residual_std: np.ndarray,
    output_dir: Path,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    statistics: Dict[str, Any] = {}
    joint_rows: List[Dict[str, Any]] = []
    for split, normalized in (("train", train.target), ("validation", validation.target)):
        physical = normalized * residual_std + residual_mean
        statistics[split] = {
            "normalized": distribution_metrics(normalized),
            "physical_radians": distribution_metrics(physical),
            "path_count": len(set(train.path_names if split == "train" else validation.path_names)),
            "window_count": int(normalized.shape[0]),
        }
        for joint_index in range(EXPECTED_TARGET_DIM):
            for representation, values in (
                ("normalized", normalized[:, :, joint_index]),
                ("physical_radians", physical[:, :, joint_index]),
            ):
                joint_rows.append(
                    {
                        "split": split,
                        "representation": representation,
                        "joint_name": f"q{joint_index + 1}",
                        "joint_index": joint_index,
                        **distribution_metrics(values),
                    }
                )
    atomic_json(statistics, output_dir / "residual_distribution_statistics.json")
    joint_frame = pd.DataFrame(joint_rows)
    atomic_csv(joint_frame, output_dir / "residual_distribution_per_joint.csv")
    return statistics, joint_frame


def build_schedule(steps: int, device: torch.device) -> DiffusionSchedule:
    betas = torch.linspace(1.0e-4, 2.0e-2, steps, dtype=torch.float32)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return DiffusionSchedule(betas, alphas, alpha_bars).to(device)


def extract(values: torch.Tensor, timesteps: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    selected = values.gather(0, timesteps)
    return selected.reshape(timesteps.shape[0], *([1] * (target.ndim - 1)))


def add_noise(
    x0: torch.Tensor,
    noise: torch.Tensor,
    timesteps: torch.Tensor,
    schedule: DiffusionSchedule,
) -> torch.Tensor:
    alpha_bar = extract(schedule.alpha_bars, timesteps, x0)
    return torch.sqrt(alpha_bar) * x0 + torch.sqrt(1.0 - alpha_bar) * noise


def reconstruct_x0(
    noisy: torch.Tensor,
    predicted_noise: torch.Tensor,
    timesteps: torch.Tensor,
    schedule: DiffusionSchedule,
) -> torch.Tensor:
    alpha_bar = extract(schedule.alpha_bars, timesteps, noisy)
    return (noisy - torch.sqrt(1.0 - alpha_bar) * predicted_noise) / torch.sqrt(
        alpha_bar
    )


def locate_v5_model_class() -> Tuple[type[nn.Module], str]:
    candidates = (
        (
            "train_conditional_diffusion_trajectory_v5_residual_unet",
            "LocalResidualConditionalUNet1D",
        ),
        ("conditional_unet1d_artist", "LocalResidualConditionalUNet1D"),
        ("conditional_unet1d_artist", "ConditionalUNet1DArtist"),
        ("conditional_unet1d_artist", "ConditionalUNet1D"),
    )
    errors: List[str] = []
    for module_name, class_name in candidates:
        try:
            module = importlib.import_module(module_name)
        except Exception as error:
            errors.append(f"{module_name}: {type(error).__name__}: {error}")
            continue
        candidate = getattr(module, class_name, None)
        if isinstance(candidate, type) and issubclass(candidate, nn.Module):
            return candidate, f"{module_name}.{class_name}"
    raise ImportError(
        "Could not locate the established v5 residual U-Net class. "
        + "; ".join(errors)
    )


def instantiate_v5_model(
    horizon: int, condition_dim: int, target_dim: int, diffusion_steps: int
) -> Tuple[nn.Module, Dict[str, Any]]:
    model_class, class_path = locate_v5_model_class()
    signature = inspect.signature(model_class)
    known_values: Dict[str, Any] = {
        "horizon": horizon,
        "sequence_length": horizon,
        "condition_dim": condition_dim,
        "cond_dim": condition_dim,
        "conditioning_dim": condition_dim,
        "global_cond_dim": condition_dim,
        "target_dim": target_dim,
        "input_dim": target_dim,
        "action_dim": target_dim,
        "trajectory_dim": target_dim,
        "in_channels": target_dim,
        "output_dim": target_dim,
        "out_channels": target_dim,
        "num_diffusion_steps": diffusion_steps,
        "diffusion_steps": diffusion_steps,
        "num_steps": diffusion_steps,
        "base_channels": 64,
        "model_channels": 64,
        "time_embed_dim": 128,
    }
    constructor_kwargs: Dict[str, Any] = {}
    unresolved_required: List[str] = []
    dimensional_parameters = {
        "horizon",
        "sequence_length",
        "condition_dim",
        "cond_dim",
        "conditioning_dim",
        "global_cond_dim",
        "target_dim",
        "input_dim",
        "action_dim",
        "trajectory_dim",
        "in_channels",
        "output_dim",
        "out_channels",
        "num_diffusion_steps",
        "diffusion_steps",
        "num_steps",
    }
    for name, parameter in signature.parameters.items():
        if name in ("self", "args", "kwargs"):
            continue
        if name in known_values and (
            name in dimensional_parameters
            or parameter.default is inspect.Parameter.empty
        ):
            constructor_kwargs[name] = known_values[name]
        elif parameter.default is inspect.Parameter.empty:
            unresolved_required.append(name)
    if unresolved_required:
        raise TypeError(
            f"Cannot instantiate {class_path}; unsupported required constructor "
            f"parameters={unresolved_required}, signature={signature}"
        )
    model = model_class(**constructor_kwargs)
    return model, {
        "class_path": class_path,
        "constructor_kwargs": constructor_kwargs,
        "constructor_signature": str(signature),
    }


def predict_noise(
    model: nn.Module,
    noisy: torch.Tensor,
    timesteps: torch.Tensor,
    condition: torch.Tensor,
) -> torch.Tensor:
    noisy_channel_first = noisy.permute(0, 2, 1).contiguous()
    condition_channel_first = condition.permute(0, 2, 1).contiguous()
    prediction_channel_first = model(
        noisy_channel_first,
        condition_channel_first,
        timesteps,
    )
    if prediction_channel_first.shape != noisy_channel_first.shape:
        raise ValueError(
            f"Model output shape {prediction_channel_first.shape}, expected "
            f"{noisy_channel_first.shape}"
        )
    return prediction_channel_first.permute(0, 2, 1).contiguous()


def dataloader_worker_init(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def create_loaders(
    train: WindowArrays,
    validation: WindowArrays,
    args: argparse.Namespace,
) -> Tuple[DataLoader[Any], DataLoader[Any]]:
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    common = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": args.device != "cpu" and torch.cuda.is_available(),
        "worker_init_fn": dataloader_worker_init,
    }
    train_loader = DataLoader(
        ResidualWindowDataset(train),
        shuffle=True,
        generator=generator,
        **common,
    )
    validation_loader = DataLoader(
        ResidualWindowDataset(validation), shuffle=False, **common
    )
    return train_loader, validation_loader


def make_validation_bank(
    count: int, horizon: int, target_dim: int, diffusion_steps: int, seed: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 1_000_003)
    timesteps = torch.randint(
        0, diffusion_steps, (count,), generator=generator, dtype=torch.long
    )
    noise = torch.randn(
        count, horizon, target_dim, generator=generator, dtype=torch.float32
    )
    return timesteps, noise


def train_epoch(
    *,
    model: nn.Module,
    loader: DataLoader[Any],
    optimizer: torch.optim.Optimizer,
    schedule: DiffusionSchedule,
    device: torch.device,
    diffusion_steps: int,
    gradient_clip_norm: float,
    max_batches: int,
    ema: Optional[ExponentialMovingAverage],
) -> Tuple[float, int]:
    model.train()
    loss_sum = 0.0
    example_count = 0
    batches = 0
    for condition, target, _ in loader:
        if max_batches and batches >= max_batches:
            break
        condition = condition.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        timesteps = torch.randint(
            0, diffusion_steps, (target.shape[0],), device=device
        )
        noise = torch.randn_like(target)
        noisy = add_noise(target, noise, timesteps, schedule)
        optimizer.zero_grad(set_to_none=True)
        predicted_noise = predict_noise(model, noisy, timesteps, condition)
        loss = torch.mean(torch.square(predicted_noise - noise))
        loss.backward()
        if gradient_clip_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        optimizer.step()
        if ema is not None:
            ema.update(model)
        batch_size = int(target.shape[0])
        loss_sum += float(loss.detach()) * batch_size
        example_count += batch_size
        batches += 1
    if example_count == 0:
        raise RuntimeError("No training batches were processed")
    return loss_sum / example_count, batches


@torch.no_grad()
def validate_model(
    *,
    model: nn.Module,
    loader: DataLoader[Any],
    schedule: DiffusionSchedule,
    device: torch.device,
    validation_timesteps: torch.Tensor,
    validation_noise: torch.Tensor,
    residual_std: torch.Tensor,
    max_batches: int,
) -> Dict[str, Any]:
    model.eval()
    squared_epsilon_sum = 0.0
    squared_x0_sum = 0.0
    squared_physical_sum = 0.0
    per_joint_squared = torch.zeros(EXPECTED_TARGET_DIM, dtype=torch.float64)
    element_count = 0
    joint_element_count = 0
    batches = 0
    for condition, target, indices in loader:
        if max_batches and batches >= max_batches:
            break
        condition = condition.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        indices = indices.to(dtype=torch.long)
        timesteps = validation_timesteps[indices].to(device)
        noise = validation_noise[indices].to(device)
        noisy = add_noise(target, noise, timesteps, schedule)
        predicted_noise = predict_noise(model, noisy, timesteps, condition)
        predicted_x0 = reconstruct_x0(noisy, predicted_noise, timesteps, schedule)
        epsilon_error = predicted_noise - noise
        x0_error = predicted_x0 - target
        physical_error = x0_error * residual_std
        squared_epsilon_sum += float(torch.sum(torch.square(epsilon_error)))
        squared_x0_sum += float(torch.sum(torch.square(x0_error)))
        squared_physical_sum += float(torch.sum(torch.square(physical_error)))
        per_joint_squared += torch.sum(
            torch.square(physical_error).double(), dim=(0, 1)
        ).cpu()
        element_count += int(target.numel())
        joint_element_count += int(target.shape[0] * target.shape[1])
        batches += 1
    if element_count == 0:
        raise RuntimeError("No validation batches were processed")
    diffusion_loss = squared_epsilon_sum / element_count
    return {
        "diffusion_loss": diffusion_loss,
        "epsilon_rmse": float(np.sqrt(diffusion_loss)),
        "x0_rmse_normalized": float(np.sqrt(squared_x0_sum / element_count)),
        "x0_rmse_physical_rad": float(
            np.sqrt(squared_physical_sum / element_count)
        ),
        "per_joint_x0_rmse_physical_rad": np.sqrt(
            per_joint_squared.numpy() / joint_element_count
        ).tolist(),
        "batch_count": batches,
    }


def validate_with_ema(
    *,
    model: nn.Module,
    ema: ExponentialMovingAverage,
    validation_kwargs: Mapping[str, Any],
) -> Dict[str, Any]:
    raw_state = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    try:
        model.load_state_dict(ema.shadow, strict=True)
        return validate_model(model=model, **validation_kwargs)
    finally:
        model.load_state_dict(raw_state, strict=True)


def atomic_json(value: Mapping[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(json_safe(value), handle, indent=2, sort_keys=True)
            handle.write("\n")
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        frame.to_csv(temporary, index=False)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_torch_save(value: Mapping[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        torch.save(dict(value), temporary)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def git_commit_hash() -> Optional[str]:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() or None


def checkpoint_payload(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    ema: Optional[ExponentialMovingAverage],
    selected_state: str,
    epoch: int,
    global_step: int,
    best_validation_loss: float,
    best_epoch: int,
    model_config: Mapping[str, Any],
    args: argparse.Namespace,
    normalization: Mapping[str, np.ndarray],
    dataset_configuration: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
    git_hash: Optional[str],
) -> Dict[str, Any]:
    raw_state = copy.deepcopy(model.state_dict())
    if selected_state == "ema":
        if ema is None:
            raise ValueError("EMA state selected without EMA enabled")
        selected_model_state = copy.deepcopy(ema.shadow)
    else:
        selected_model_state = raw_state
    return {
        "model_state_dict": selected_model_state,
        "raw_model_state_dict": raw_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": None,
        "ema_state_dict": None if ema is None else ema.state_dict(),
        "selected_model_state": selected_state,
        "epoch": epoch,
        "global_step": global_step,
        "best_validation_loss": best_validation_loss,
        "best_epoch": best_epoch,
        "model_hyperparameters": dict(model_config),
        "diffusion_hyperparameters": {
            "steps": args.diffusion_steps,
            "beta_schedule": "linear",
            "beta_start": 1.0e-4,
            "beta_end": 2.0e-2,
        },
        "horizon": EXPECTED_HORIZON,
        "condition_dim": EXPECTED_CONDITION_DIM,
        "target_dim": EXPECTED_TARGET_DIM,
        "prediction_target_type": PREDICTION_TARGET,
        "seed": args.seed,
        "train_npz": str(args.train_npz.resolve()),
        "val_npz": str(args.val_npz.resolve()),
        "normalization_source_path": str(args.normalization_stats.resolve()),
        "normalization_statistics": {
            key: value for key, value in normalization.items()
        },
        "dataset_configuration": dict(dataset_configuration),
        "git_commit_hash": git_hash,
        "training_history": list(history),
    }


def save_history(history: Sequence[Mapping[str, Any]], output_dir: Path) -> None:
    frame = pd.DataFrame(history)
    atomic_csv(frame, output_dir / "training_history.csv")
    atomic_json({"epochs": list(history)}, output_dir / "training_history.json")


def load_resume_checkpoint(
    *,
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    ema: Optional[ExponentialMovingAverage],
    device: torch.device,
    args: argparse.Namespace,
) -> Tuple[int, int, float, int, List[Dict[str, Any]]]:
    if not path.exists():
        raise FileNotFoundError(path)
    checkpoint = torch.load(path, map_location=device)
    for key, expected in (
        ("horizon", EXPECTED_HORIZON),
        ("condition_dim", EXPECTED_CONDITION_DIM),
        ("target_dim", EXPECTED_TARGET_DIM),
    ):
        if int(checkpoint.get(key, -1)) != expected:
            raise ValueError(f"Resume checkpoint {key} is incompatible")
    if checkpoint.get("prediction_target_type") != PREDICTION_TARGET:
        raise ValueError("Resume checkpoint prediction target is not epsilon")
    raw_state = checkpoint.get("raw_model_state_dict", checkpoint["model_state_dict"])
    model.load_state_dict(raw_state, strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if ema is not None:
        if checkpoint.get("ema_state_dict") is None:
            raise ValueError("Resume checkpoint has no EMA state")
        ema.load_state_dict(checkpoint["ema_state_dict"])
    elif checkpoint.get("ema_state_dict") is not None and args.use_ema:
        raise ValueError("EMA resume state could not be restored")
    history = [dict(row) for row in checkpoint.get("training_history", [])]
    return (
        int(checkpoint["epoch"]) + 1,
        int(checkpoint.get("global_step", 0)),
        float(checkpoint.get("best_validation_loss", float("inf"))),
        int(checkpoint.get("best_epoch", 0)),
        history,
    )


def verify_checkpoint_reload(
    path: Path,
    device: torch.device,
    args: argparse.Namespace,
) -> None:
    checkpoint = torch.load(path, map_location=device)
    reloaded, _ = instantiate_v5_model(
        EXPECTED_HORIZON,
        EXPECTED_CONDITION_DIM,
        EXPECTED_TARGET_DIM,
        int(checkpoint["diffusion_hyperparameters"]["steps"]),
    )
    reloaded = reloaded.to(device)
    reloaded_optimizer = torch.optim.AdamW(
        reloaded.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    reloaded_ema = (
        ExponentialMovingAverage(reloaded, args.ema_decay)
        if args.use_ema
        else None
    )
    load_resume_checkpoint(
        path=path,
        model=reloaded,
        optimizer=reloaded_optimizer,
        ema=reloaded_ema,
        device=device,
        args=args,
    )
    del reloaded, reloaded_optimizer, reloaded_ema


def main() -> int:
    args = parse_args()
    validate_cli(args)
    if args.smoke_test:
        args.epochs = min(args.epochs, 2)
        args.max_train_batches = (
            5 if args.max_train_batches == 0 else min(args.max_train_batches, 5)
        )
        args.max_val_batches = (
            3 if args.max_val_batches == 0 else min(args.max_val_batches, 3)
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.resume_checkpoint is None and any(
        (args.output_dir / name).exists()
        for name in ("best_checkpoint.pt", "last_checkpoint.pt", "training_history.csv")
    ):
        raise FileExistsError(
            "Training outputs already exist; use a new output directory or "
            "--resume_checkpoint"
        )

    set_reproducibility(args.seed, args.deterministic)
    device = resolve_device(args.device)
    train_arrays = load_windows(
        args.train_npz, EXPECTED_TRAIN_WINDOWS, "train"
    )
    val_arrays = load_windows(args.val_npz, EXPECTED_VAL_WINDOWS, "validation")
    train_paths = set(train_arrays.path_names)
    val_paths = set(val_arrays.path_names)
    if train_paths & val_paths:
        raise ValueError("Training and validation path-name sets overlap")
    if len(train_paths) != 372 or len(val_paths) != 42:
        raise ValueError(
            f"Expected 372 train paths and 42 validation paths, got "
            f"{len(train_paths)} and {len(val_paths)}"
        )
    normalization = load_normalization(args.normalization_stats)
    dataset_configuration = load_json(args.dataset_configuration)
    validate_dataset_metadata(dataset_configuration, normalization)

    residual_mean_np = np.asarray(
        normalization["residual_mean"], dtype=np.float32
    ).reshape(1, 1, EXPECTED_TARGET_DIM)
    residual_std_np = np.asarray(
        normalization["residual_std"], dtype=np.float32
    ).reshape(1, 1, EXPECTED_TARGET_DIM)
    residual_statistics, _ = save_residual_statistics(
        train_arrays,
        val_arrays,
        residual_mean_np,
        residual_std_np,
        args.output_dir,
    )

    model, model_config = instantiate_v5_model(
        EXPECTED_HORIZON,
        EXPECTED_CONDITION_DIM,
        EXPECTED_TARGET_DIM,
        args.diffusion_steps,
    )
    model = model.to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    ema = ExponentialMovingAverage(model, args.ema_decay) if args.use_ema else None
    schedule = build_schedule(args.diffusion_steps, device)
    train_loader, val_loader = create_loaders(train_arrays, val_arrays, args)
    validation_timesteps, validation_noise = make_validation_bank(
        len(val_arrays.path_names),
        EXPECTED_HORIZON,
        EXPECTED_TARGET_DIM,
        args.diffusion_steps,
        args.seed,
    )
    residual_std = torch.from_numpy(residual_std_np).to(device)
    git_hash = git_commit_hash()

    training_configuration = {
        "classification": "V6_TRAINING_IN_PROGRESS",
        "arguments": vars(args),
        "device": str(device),
        "prediction_target_type": PREDICTION_TARGET,
        "model": model_config,
        "model_parameter_count": parameter_count,
        "horizon": EXPECTED_HORIZON,
        "condition_dim": EXPECTED_CONDITION_DIM,
        "target_dim": EXPECTED_TARGET_DIM,
        "condition_feature_layout": dataset_configuration.get(
            "condition_feature_layout"
        ),
        "train_path_names": sorted(train_paths),
        "validation_path_names": sorted(val_paths),
        "train_path_names_sha256": path_set_hash(train_paths),
        "validation_path_names_sha256": path_set_hash(val_paths),
        "validation_noise_bank_seed": args.seed + 1_000_003,
        "official_test_data_loaded": False,
        "normalization_source": "supervised_train_paths_only",
        "git_commit_hash": git_hash,
    }
    atomic_json(
        training_configuration, args.output_dir / "training_configuration.json"
    )
    with (args.output_dir / "model_summary.txt").open("w", encoding="utf-8") as handle:
        handle.write(f"{model}\n\n")
        handle.write(f"model_class: {model_config['class_path']}\n")
        handle.write(f"parameter_count: {parameter_count}\n")
        handle.write(f"prediction_target: {PREDICTION_TARGET}\n")

    start_epoch = 1
    global_step = 0
    best_validation_loss = float("inf")
    best_epoch = 0
    best_physical_rmse = float("inf")
    history: List[Dict[str, Any]] = []
    if args.resume_checkpoint is not None:
        (
            start_epoch,
            global_step,
            best_validation_loss,
            best_epoch,
            history,
        ) = load_resume_checkpoint(
            path=args.resume_checkpoint,
            model=model,
            optimizer=optimizer,
            ema=ema,
            device=device,
            args=args,
        )
        if history:
            best_rows = [
                row for row in history if int(row.get("epoch", -1)) == best_epoch
            ]
            if best_rows:
                best_physical_rmse = float(
                    best_rows[-1].get(
                        "selected_x0_rmse_physical_rad", float("inf")
                    )
                )

    print(f"device: {device}")
    print(
        f"train shape: condition={train_arrays.condition.shape}, "
        f"target={train_arrays.target.shape}"
    )
    print(
        f"validation shape: condition={val_arrays.condition.shape}, "
        f"target={val_arrays.target.shape}"
    )
    print(f"path counts: train={len(train_paths)}, validation={len(val_paths)}")
    print(f"model parameters: {parameter_count}")
    print(f"prediction target: {PREDICTION_TARGET}")
    print(f"diffusion steps: {args.diffusion_steps}")
    print(f"residual distribution summary: {residual_statistics}")

    early_stopping_counter = sum(
        1
        for row in history
        if int(row.get("epoch", 0)) > best_epoch
        and np.isfinite(float(row.get("selected_validation_loss", float("nan"))))
    )
    last_checkpoint_path = args.output_dir / "last_checkpoint.pt"
    best_checkpoint_path = args.output_dir / "best_checkpoint.pt"
    last_payload: Optional[Dict[str, Any]] = None
    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, train_batches = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            schedule=schedule,
            device=device,
            diffusion_steps=args.diffusion_steps,
            gradient_clip_norm=args.gradient_clip_norm,
            max_batches=args.max_train_batches,
            ema=ema,
        )
        global_step += train_batches
        raw_metrics: Optional[Dict[str, Any]] = None
        ema_metrics: Optional[Dict[str, Any]] = None
        selected_metrics: Optional[Dict[str, Any]] = None
        selected_state = "raw"
        improved = False
        if epoch % args.validation_interval == 0:
            validation_kwargs = {
                "loader": val_loader,
                "schedule": schedule,
                "device": device,
                "validation_timesteps": validation_timesteps,
                "validation_noise": validation_noise,
                "residual_std": residual_std,
                "max_batches": args.max_val_batches,
            }
            raw_metrics = validate_model(model=model, **validation_kwargs)
            selected_metrics = raw_metrics
            if ema is not None:
                ema_metrics = validate_with_ema(
                    model=model, ema=ema, validation_kwargs=validation_kwargs
                )
                if ema_metrics["diffusion_loss"] < raw_metrics["diffusion_loss"]:
                    selected_metrics = ema_metrics
                    selected_state = "ema"
            selected_loss = float(selected_metrics["diffusion_loss"])
            improved = selected_loss < best_validation_loss
            if improved:
                best_validation_loss = selected_loss
                best_epoch = epoch
                best_physical_rmse = float(
                    selected_metrics["x0_rmse_physical_rad"]
                )
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1

        row: Dict[str, Any] = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": train_loss,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "raw_validation_loss": (
                float("nan") if raw_metrics is None else raw_metrics["diffusion_loss"]
            ),
            "ema_validation_loss": (
                float("nan") if ema_metrics is None else ema_metrics["diffusion_loss"]
            ),
            "raw_epsilon_rmse": (
                float("nan") if raw_metrics is None else raw_metrics["epsilon_rmse"]
            ),
            "raw_x0_rmse_normalized": (
                float("nan")
                if raw_metrics is None
                else raw_metrics["x0_rmse_normalized"]
            ),
            "raw_x0_rmse_physical_rad": (
                float("nan")
                if raw_metrics is None
                else raw_metrics["x0_rmse_physical_rad"]
            ),
            "ema_epsilon_rmse": (
                float("nan") if ema_metrics is None else ema_metrics["epsilon_rmse"]
            ),
            "ema_x0_rmse_normalized": (
                float("nan")
                if ema_metrics is None
                else ema_metrics["x0_rmse_normalized"]
            ),
            "ema_x0_rmse_physical_rad": (
                float("nan")
                if ema_metrics is None
                else ema_metrics["x0_rmse_physical_rad"]
            ),
            "selected_model_state": selected_state if selected_metrics else "none",
            "selected_validation_loss": (
                float("nan")
                if selected_metrics is None
                else selected_metrics["diffusion_loss"]
            ),
            "selected_epsilon_rmse": (
                float("nan")
                if selected_metrics is None
                else selected_metrics["epsilon_rmse"]
            ),
            "selected_x0_rmse_normalized": (
                float("nan")
                if selected_metrics is None
                else selected_metrics["x0_rmse_normalized"]
            ),
            "selected_x0_rmse_physical_rad": (
                float("nan")
                if selected_metrics is None
                else selected_metrics["x0_rmse_physical_rad"]
            ),
            "best_validation_loss": best_validation_loss,
            "best_epoch": best_epoch,
            "early_stopping_counter": early_stopping_counter,
        }
        if selected_metrics is not None:
            for joint_index, value in enumerate(
                selected_metrics["per_joint_x0_rmse_physical_rad"]
            ):
                row[f"selected_q{joint_index + 1}_x0_rmse_physical_rad"] = value
        history.append(row)
        save_history(history, args.output_dir)

        payload = checkpoint_payload(
            model=model,
            optimizer=optimizer,
            ema=ema,
            selected_state=selected_state,
            epoch=epoch,
            global_step=global_step,
            best_validation_loss=best_validation_loss,
            best_epoch=best_epoch,
            model_config=model_config,
            args=args,
            normalization=normalization,
            dataset_configuration=dataset_configuration,
            history=history,
            git_hash=git_hash,
        )
        last_payload = payload
        if improved:
            atomic_torch_save(payload, best_checkpoint_path)
        if epoch % args.checkpoint_interval == 0:
            atomic_torch_save(
                payload, args.output_dir / f"checkpoint_epoch_{epoch:04d}.pt"
            )
        if args.save_last:
            atomic_torch_save(payload, last_checkpoint_path)

        validation_loss_text = (
            "nan"
            if selected_metrics is None
            else f"{selected_metrics['diffusion_loss']:.8f}"
        )
        normalized_text = (
            "nan"
            if selected_metrics is None
            else f"{selected_metrics['x0_rmse_normalized']:.8f}"
        )
        physical_text = (
            "nan"
            if selected_metrics is None
            else f"{selected_metrics['x0_rmse_physical_rad']:.8f}"
        )
        print(
            f"epoch={epoch} train_loss={train_loss:.8f} "
            f"val_loss={validation_loss_text} val_x0_norm={normalized_text} "
            f"val_x0_rad={physical_text} "
            f"lr={optimizer.param_groups[0]['lr']:.3e} best_epoch={best_epoch} "
            f"early_stop={early_stopping_counter}"
        )
        if (
            selected_metrics is not None
            and early_stopping_counter >= args.early_stopping_patience
        ):
            print(f"early stopping at epoch {epoch}")
            break

    if not best_checkpoint_path.exists():
        raise RuntimeError("No best checkpoint was saved")
    if args.save_last and not last_checkpoint_path.exists():
        raise RuntimeError("No last checkpoint was saved")
    if args.smoke_test:
        smoke_checkpoint_path = args.output_dir / "smoke_checkpoint.pt"
        if last_payload is None:
            loaded_payload = torch.load(best_checkpoint_path, map_location=device)
            if not isinstance(loaded_payload, dict):
                raise TypeError("Best checkpoint payload must be a dictionary")
            smoke_payload: Mapping[str, Any] = loaded_payload
        else:
            smoke_payload = last_payload
        atomic_torch_save(smoke_payload, smoke_checkpoint_path)
        verify_checkpoint_reload(smoke_checkpoint_path, device, args)

    training_configuration["classification"] = "V6_TRAINING_COMPLETE"
    training_configuration["best_epoch"] = best_epoch
    training_configuration["best_validation_loss"] = best_validation_loss
    training_configuration["best_physical_validation_x0_rmse_rad"] = (
        best_physical_rmse
    )
    atomic_json(
        training_configuration, args.output_dir / "training_configuration.json"
    )
    print(f"best epoch: {best_epoch}")
    print(f"best validation loss: {best_validation_loss:.8f}")
    print(f"best physical validation x0 RMSE: {best_physical_rmse:.8f} rad")
    print(f"best checkpoint: {best_checkpoint_path}")
    print(
        f"last checkpoint: {last_checkpoint_path if args.save_last else 'disabled'}"
    )
    print("classification: V6_TRAINING_COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
