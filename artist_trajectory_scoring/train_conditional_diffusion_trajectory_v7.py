#!/usr/bin/env python3
"""Train v7 conditional diffusion on cost-improving residual targets.

This module intentionally reuses the v6 diffusion schedule, established
Conditional 1D U-Net loader, noise/denoising equations, EMA implementation,
normalization convention, and atomic checkpoint helpers.  The v7-specific
change is row multiplicity: the default loss weights each target row by the
inverse number of targets for its conditional window.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import platform
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

import train_conditional_diffusion_trajectory_v6_strong_prior_residual_unet as v6


HORIZON = 32
CONDITION_DIM = 38
TARGET_DIM = 6
PREDICTION_TARGET = "epsilon"
DEFAULT_DATASET_DIR = Path(
    "data/cartesian_expert_dataset_v3/"
    "diffusion_v7_cost_improving_training_dataset_100paths"
)


@dataclass(frozen=True)
class WindowArrays:
    source: Path
    condition: np.ndarray
    target: np.ndarray
    sample_weight: np.ndarray
    path_names: Tuple[str, ...]
    window_starts: np.ndarray
    unique_window_id: np.ndarray
    prior_q_window: np.ndarray


class WeightedResidualDataset(
    Dataset[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]]
):
    def __init__(self, arrays: WindowArrays) -> None:
        self.condition = torch.from_numpy(arrays.condition.astype(np.float32))
        self.target = torch.from_numpy(arrays.target.astype(np.float32))
        self.sample_weight = torch.from_numpy(
            arrays.sample_weight.astype(np.float32)
        )

    def __len__(self) -> int:
        return int(self.target.shape[0])

    def __getitem__(
        self, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        return (
            self.condition[index],
            self.target[index],
            self.sample_weight[index],
            index,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train v7 cost-improving residual conditional diffusion."
    )
    parser.add_argument("--dataset_dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--train_npz", type=Path, default=None)
    parser.add_argument("--validation_npz", type=Path, default=None)
    parser.add_argument("--normalization", type=Path, default=None)
    parser.add_argument("--dataset_metadata", type=Path, default=None)
    parser.add_argument("--split_manifest", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--learning_rate", type=float, default=1.0e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--diffusion_steps", type=int, default=1000)
    parser.add_argument("--validation_interval", type=int, default=1)
    parser.add_argument("--checkpoint_interval", type=int, default=25)
    parser.add_argument("--early_stopping_patience", type=int, default=75)
    parser.add_argument("--gradient_clip_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--resume_checkpoint", type=Path, default=None)
    parser.add_argument("--max_train_batches", type=int, default=0)
    parser.add_argument("--max_val_batches", type=int, default=0)
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument(
        "--sampling_mode",
        choices=("weighted_loss", "uniform_rows"),
        default="weighted_loss",
        help=(
            "weighted_loss gives each unique window total weight one; "
            "uniform_rows is an unweighted row-level ablation"
        ),
    )
    parser.add_argument(
        "--use_ema", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--deterministic", action="store_true")
    return parser.parse_args()


def resolve_dataset_paths(args: argparse.Namespace) -> None:
    defaults = {
        "train_npz": "train_windows.npz",
        "validation_npz": "validation_windows.npz",
        "normalization": "normalization.npz",
        "dataset_metadata": "dataset_metadata.json",
        "split_manifest": "split_manifest.json",
    }
    for attribute, filename in defaults.items():
        if getattr(args, attribute) is None:
            setattr(args, attribute, args.dataset_dir / filename)


def validate_cli(args: argparse.Namespace) -> None:
    for name in (
        "epochs", "batch_size", "diffusion_steps", "validation_interval",
        "checkpoint_interval", "early_stopping_patience",
    ):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name} must be positive")
    if args.learning_rate <= 0.0:
        raise ValueError("--learning_rate must be positive")
    if args.weight_decay < 0.0 or args.gradient_clip_norm < 0.0:
        raise ValueError("Weight decay and gradient clipping must be non-negative")
    if args.num_workers < 0 or args.max_train_batches < 0 or args.max_val_batches < 0:
        raise ValueError("Worker and maximum-batch counts must be non-negative")
    if not 0.0 < args.ema_decay < 1.0:
        raise ValueError("--ema_decay must lie in (0,1)")
    if args.train_npz.resolve() == args.validation_npz.resolve():
        raise ValueError("Training and validation archives must be different")


def decode_names(values: np.ndarray) -> Tuple[str, ...]:
    return tuple(
        item.decode("utf-8", errors="strict")
        if isinstance(item, bytes)
        else str(item)
        for item in np.asarray(values).reshape(-1)
    )


def load_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def load_normalization(path: Path) -> Dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        required = (
            "condition_mean", "condition_std", "residual_mean", "residual_std",
            "condition_feature_names", "condition_dim", "target_dim", "horizon",
        )
        missing = [key for key in required if key not in archive.files]
        if missing:
            raise KeyError(f"{path} is missing normalization arrays: {missing}")
        result = {key: np.asarray(archive[key]) for key in archive.files}
    for key in ("condition_mean", "condition_std", "residual_mean", "residual_std"):
        if not np.all(np.isfinite(result[key])):
            raise ValueError(f"{path}/{key} contains NaN or infinity")
    if np.any(result["condition_std"] <= 0.0) or np.any(result["residual_std"] <= 0.0):
        raise ValueError("Normalization standard deviations must be positive")
    expected_shapes = {
        "condition_mean": (1, 1, CONDITION_DIM),
        "condition_std": (1, 1, CONDITION_DIM),
        "residual_mean": (1, 1, TARGET_DIM),
        "residual_std": (1, 1, TARGET_DIM),
    }
    for key, expected in expected_shapes.items():
        if result[key].shape != expected:
            raise ValueError(f"{key} has shape {result[key].shape}; expected {expected}")
    for key, expected in (("condition_dim", CONDITION_DIM), ("target_dim", TARGET_DIM), ("horizon", HORIZON)):
        if int(np.asarray(result[key]).item()) != expected:
            raise ValueError(f"Normalization {key} is incompatible with v7 trainer")
    return result


def load_windows(path: Path, label: str) -> WindowArrays:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=True) as archive:
        required = (
            "condition_features_norm", "residual_q_norm", "sample_weight",
            "path_names", "window_starts", "unique_window_id", "prior_q_window",
            "residual_q_window", "candidate_q_window", "is_zero_residual",
        )
        missing = [key for key in required if key not in archive.files]
        if missing:
            raise KeyError(f"{path} is missing required arrays: {missing}")
        condition = np.asarray(archive["condition_features_norm"], dtype=np.float32)
        target = np.asarray(archive["residual_q_norm"], dtype=np.float32)
        weights = np.asarray(archive["sample_weight"], dtype=np.float32).reshape(-1)
        names = decode_names(archive["path_names"])
        starts = np.asarray(archive["window_starts"], dtype=np.int64).reshape(-1)
        window_ids = np.asarray(archive["unique_window_id"], dtype=np.int64).reshape(-1)
        prior = np.asarray(archive["prior_q_window"], dtype=np.float32)
        physical_residual = np.asarray(archive["residual_q_window"], dtype=np.float32)
        candidate = np.asarray(archive["candidate_q_window"], dtype=np.float32)
        zero_flags = np.asarray(archive["is_zero_residual"], dtype=bool).reshape(-1)
    count = condition.shape[0]
    if condition.shape != (count, HORIZON, CONDITION_DIM):
        raise ValueError(f"{label} condition shape is {condition.shape}")
    if target.shape != (count, HORIZON, TARGET_DIM):
        raise ValueError(f"{label} target shape is {target.shape}")
    if prior.shape != (count, HORIZON, TARGET_DIM):
        raise ValueError(f"{label} prior shape is {prior.shape}")
    if physical_residual.shape != (count, HORIZON, TARGET_DIM):
        raise ValueError(f"{label} physical residual shape is {physical_residual.shape}")
    if candidate.shape != (count, HORIZON, TARGET_DIM):
        raise ValueError(f"{label} candidate shape is {candidate.shape}")
    for name, values in (
        ("condition", condition), ("target", target), ("sample_weight", weights),
        ("prior", prior), ("residual", physical_residual), ("candidate", candidate),
    ):
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{label} {name} contains NaN or infinity")
    for key, values in (
        ("sample_weight", weights), ("path_names", names), ("window_starts", starts),
        ("unique_window_id", window_ids), ("is_zero_residual", zero_flags),
    ):
        if len(values) != count:
            raise ValueError(f"{label} {key} length does not match row count")
    if np.any(weights <= 0.0):
        raise ValueError(f"{label} sample_weight must be positive")
    if not np.allclose(prior + physical_residual, candidate, rtol=1.0e-5, atol=1.0e-6):
        raise ValueError(
            f"{label} candidate trajectory is not prior_q_window + residual_q_window"
        )
    numerical_zero = np.max(np.abs(physical_residual), axis=(1, 2)) <= 1.0e-7
    if not np.array_equal(numerical_zero, zero_flags):
        raise ValueError(f"{label} zero-residual flags do not match residual values")
    for identifier in np.unique(window_ids):
        total = float(np.sum(weights[window_ids == identifier], dtype=np.float64))
        if not np.isclose(total, 1.0, atol=1.0e-5):
            raise ValueError(
                f"{label} sample weights for unique_window_id={identifier} sum to {total}"
            )
    return WindowArrays(path, condition, target, weights, names, starts, window_ids, prior)


def validate_metadata(
    metadata: Mapping[str, Any], normalization: Mapping[str, np.ndarray]
) -> None:
    if metadata.get("classification") != "READY_FOR_V7_TRAINING":
        raise ValueError("Dataset metadata is not classified READY_FOR_V7_TRAINING")
    if not bool(metadata.get("validation_excluded_from_normalization", False)):
        raise ValueError("Dataset does not confirm validation exclusion from normalization")
    for key, expected in (
        ("horizon", HORIZON), ("condition_dim", CONDITION_DIM), ("target_dim", TARGET_DIM)
    ):
        if int(metadata.get(key, -1)) != expected:
            raise ValueError(f"Dataset metadata {key} is incompatible")
    if tuple(metadata.get("condition_feature_layout", ())) != tuple(v6.EXPECTED_CONDITION_LAYOUT):
        raise ValueError("Dataset condition ordering is not the v6 38-D condition layout")
    names = tuple(str(value) for value in normalization["condition_feature_names"].tolist())
    if len(names) != CONDITION_DIM or len(set(names)) != CONDITION_DIM:
        raise ValueError("Normalization condition feature names must be 38 unique names")
    if tuple(metadata.get("condition_feature_names", ())) != names:
        raise ValueError("Dataset and normalization condition feature names differ")


def path_set_hash(names: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(sorted(set(names))).encode("utf-8")).hexdigest()


def make_loaders(
    train: WindowArrays, validation: WindowArrays, args: argparse.Namespace
) -> Tuple[DataLoader[Any], DataLoader[Any]]:
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    common = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": str(args.device) != "cpu" and torch.cuda.is_available(),
        "worker_init_fn": v6.dataloader_worker_init,
    }
    return (
        DataLoader(
            WeightedResidualDataset(train), shuffle=True,
            generator=generator, **common,
        ),
        DataLoader(WeightedResidualDataset(validation), shuffle=False, **common),
    )


def weighted_average(per_sample: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    weight = weight.reshape(-1).to(dtype=per_sample.dtype)
    denominator = torch.sum(weight)
    if not bool(denominator > 0.0):
        raise ValueError("A batch has zero total sample weight")
    return torch.sum(per_sample * weight) / denominator


def train_epoch(
    *, model: nn.Module, loader: DataLoader[Any], optimizer: torch.optim.Optimizer,
    schedule: v6.DiffusionSchedule, device: torch.device, diffusion_steps: int,
    gradient_clip_norm: float, max_batches: int,
    ema: Optional[v6.ExponentialMovingAverage], sampling_mode: str,
) -> Tuple[float, int]:
    model.train()
    numerator = 0.0
    denominator = 0.0
    batches = 0
    for condition, target, sample_weight, _ in loader:
        if max_batches and batches >= max_batches:
            break
        condition = condition.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        sample_weight = sample_weight.to(device, non_blocking=True)
        effective_weight = (
            sample_weight if sampling_mode == "weighted_loss"
            else torch.ones_like(sample_weight)
        )
        timesteps = torch.randint(0, diffusion_steps, (target.shape[0],), device=device)
        noise = torch.randn_like(target)
        noisy = v6.add_noise(target, noise, timesteps, schedule)
        optimizer.zero_grad(set_to_none=True)
        predicted_noise = v6.predict_noise(model, noisy, timesteps, condition)
        per_sample_loss = torch.mean(
            torch.square(predicted_noise - noise), dim=(1, 2)
        )
        loss = weighted_average(per_sample_loss, effective_weight)
        loss.backward()
        if gradient_clip_norm > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        optimizer.step()
        if ema is not None:
            ema.update(model)
        numerator += float(torch.sum(per_sample_loss.detach() * effective_weight))
        denominator += float(torch.sum(effective_weight))
        batches += 1
    if denominator <= 0.0:
        raise RuntimeError("No training batches were processed")
    return numerator / denominator, batches


@torch.no_grad()
def validate_model(
    *, model: nn.Module, loader: DataLoader[Any], schedule: v6.DiffusionSchedule,
    device: torch.device, validation_timesteps: torch.Tensor,
    validation_noise: torch.Tensor, residual_std: torch.Tensor,
    max_batches: int,
) -> Dict[str, Any]:
    model.eval()
    epsilon_sum = 0.0
    x0_sum = 0.0
    physical_sum = 0.0
    joint_sum = torch.zeros(TARGET_DIM, dtype=torch.float64)
    weight_sum = 0.0
    batches = 0
    for condition, target, sample_weight, indices in loader:
        if max_batches and batches >= max_batches:
            break
        condition = condition.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        weights = sample_weight.to(device, non_blocking=True)
        indices = indices.to(dtype=torch.long)
        timesteps = validation_timesteps[indices].to(device)
        noise = validation_noise[indices].to(device)
        noisy = v6.add_noise(target, noise, timesteps, schedule)
        predicted_noise = v6.predict_noise(model, noisy, timesteps, condition)
        predicted_x0 = v6.reconstruct_x0(noisy, predicted_noise, timesteps, schedule)
        epsilon_per_sample = torch.mean(torch.square(predicted_noise - noise), dim=(1, 2))
        x0_error = predicted_x0 - target
        x0_per_sample = torch.mean(torch.square(x0_error), dim=(1, 2))
        physical_error = x0_error * residual_std
        physical_per_sample = torch.mean(torch.square(physical_error), dim=(1, 2))
        per_sample_joint = torch.mean(torch.square(physical_error), dim=1)
        epsilon_sum += float(torch.sum(epsilon_per_sample * weights))
        x0_sum += float(torch.sum(x0_per_sample * weights))
        physical_sum += float(torch.sum(physical_per_sample * weights))
        joint_sum += torch.sum(
            per_sample_joint.double() * weights.double().reshape(-1, 1), dim=0
        ).cpu()
        weight_sum += float(torch.sum(weights))
        batches += 1
    if weight_sum <= 0.0:
        raise RuntimeError("No validation batches were processed")
    diffusion_loss = epsilon_sum / weight_sum
    return {
        "diffusion_loss": diffusion_loss,
        "epsilon_rmse": float(np.sqrt(diffusion_loss)),
        "x0_rmse_normalized": float(np.sqrt(x0_sum / weight_sum)),
        "x0_rmse_physical_rad": float(np.sqrt(physical_sum / weight_sum)),
        "per_joint_x0_rmse_physical_rad": np.sqrt(
            joint_sum.numpy() / weight_sum
        ).tolist(),
        "sample_weight_sum": weight_sum,
        "batch_count": batches,
    }


def validate_with_ema(
    model: nn.Module, ema: v6.ExponentialMovingAverage,
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


def software_device_information(device: torch.device) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "torch": torch.__version__,
        "matplotlib": __import__("matplotlib").__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
    }
    if device.type == "cuda":
        result["cuda_device_name"] = torch.cuda.get_device_name(device)
        result["cuda_device_capability"] = torch.cuda.get_device_capability(device)
    return result


def checkpoint_payload(
    *, model: nn.Module, optimizer: torch.optim.Optimizer,
    ema: Optional[v6.ExponentialMovingAverage], selected_state: str,
    epoch: int, global_step: int, best_raw_loss: float, best_raw_epoch: int,
    best_ema_loss: float, best_ema_epoch: int, model_config: Mapping[str, Any],
    args: argparse.Namespace, normalization: Mapping[str, np.ndarray],
    dataset_metadata: Mapping[str, Any], split_manifest: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]], environment: Mapping[str, Any],
) -> Dict[str, Any]:
    raw_state = copy.deepcopy(model.state_dict())
    if selected_state == "ema":
        if ema is None:
            raise ValueError("Cannot create an EMA checkpoint when EMA is disabled")
        selected_model_state = copy.deepcopy(ema.shadow)
    elif selected_state == "raw":
        selected_model_state = raw_state
    else:
        raise ValueError(f"Unknown selected_state={selected_state}")
    feature_names = [str(value) for value in normalization["condition_feature_names"].tolist()]
    selected_best_loss = best_ema_loss if selected_state == "ema" else best_raw_loss
    selected_best_epoch = best_ema_epoch if selected_state == "ema" else best_raw_epoch
    return {
        "model_state_dict": selected_model_state,
        "raw_model_state_dict": raw_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": None,
        "ema_state_dict": None if ema is None else ema.state_dict(),
        "selected_model_state": selected_state,
        "epoch": epoch,
        "global_step": global_step,
        "best_raw_validation_loss": best_raw_loss,
        "best_raw_epoch": best_raw_epoch,
        "best_ema_validation_loss": best_ema_loss,
        "best_ema_epoch": best_ema_epoch,
        "best_validation_loss": selected_best_loss,
        "best_epoch": selected_best_epoch,
        "model_hyperparameters": dict(model_config),
        "diffusion_hyperparameters": {
            "steps": args.diffusion_steps,
            "beta_schedule": "linear",
            "beta_start": 1.0e-4,
            "beta_end": 2.0e-2,
        },
        "optimizer_hyperparameters": {
            "class": "torch.optim.AdamW",
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "learning_rate_scheduler": None,
        },
        "horizon": HORIZON,
        "condition_dim": CONDITION_DIM,
        "target_dim": TARGET_DIM,
        "prediction_target_type": PREDICTION_TARGET,
        "diffusion_target": "normalized residual_q_window",
        "candidate_reconstruction": "prior_q_window + denormalized predicted_residual_q_window",
        "sampling_mode": args.sampling_mode,
        "sample_weighted_validation": True,
        "seed": args.seed,
        "condition_feature_ordering": feature_names,
        "condition_feature_layout": list(v6.EXPECTED_CONDITION_LAYOUT),
        "normalization_statistics": {
            key: value for key, value in normalization.items()
        },
        "dataset_metadata": dict(dataset_metadata),
        "dataset_configuration": dict(dataset_metadata),
        "split_manifest": dict(split_manifest),
        "software_device_information": dict(environment),
        "train_npz": str(args.train_npz.resolve()),
        "validation_npz": str(args.validation_npz.resolve()),
        "val_npz": str(args.validation_npz.resolve()),
        "normalization_source_path": str(args.normalization.resolve()),
        "training_history": list(history),
    }


def save_history_and_plot(
    history: Sequence[Mapping[str, Any]], output_dir: Path
) -> None:
    frame = pd.DataFrame(history)
    v6.atomic_csv(frame, output_dir / "training_history.csv")
    v6.atomic_json({"epochs": list(history)}, output_dir / "training_history.json")
    figure, axis = plt.subplots(figsize=(9, 5))
    axis.plot(frame["epoch"], frame["train_loss"], label="train")
    for column, label in (
        ("raw_validation_loss", "validation raw"),
        ("ema_validation_loss", "validation EMA"),
    ):
        finite = np.isfinite(frame[column].to_numpy(dtype=np.float64))
        if np.any(finite):
            axis.plot(frame.loc[finite, "epoch"], frame.loc[finite, column], label=label)
    positive_values = frame[
        ["train_loss", "raw_validation_loss", "ema_validation_loss"]
    ].to_numpy(dtype=np.float64)
    if np.all(positive_values[np.isfinite(positive_values)] > 0.0):
        axis.set_yscale("log")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Weighted epsilon MSE")
    axis.set_title("V7 diffusion training and validation loss")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    output_path = output_dir / "training_and_validation_loss.png"
    temporary = output_dir / ".training_and_validation_loss.tmp.png"
    try:
        figure.savefig(str(temporary), dpi=160)
        os.replace(temporary, output_path)
    finally:
        plt.close(figure)
        if temporary.exists():
            temporary.unlink()


def load_resume(
    *, path: Path, model: nn.Module, optimizer: torch.optim.Optimizer,
    ema: Optional[v6.ExponentialMovingAverage], device: torch.device,
    args: argparse.Namespace,
) -> Tuple[int, int, float, int, float, int, List[Dict[str, Any]]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    checkpoint = torch.load(path, map_location=device)
    for key, expected in (
        ("horizon", HORIZON), ("condition_dim", CONDITION_DIM), ("target_dim", TARGET_DIM)
    ):
        if int(checkpoint.get(key, -1)) != expected:
            raise ValueError(f"Resume checkpoint {key} is incompatible")
    if checkpoint.get("prediction_target_type") != PREDICTION_TARGET:
        raise ValueError("Resume checkpoint prediction target is not epsilon")
    if checkpoint.get("sampling_mode") != args.sampling_mode:
        raise ValueError("Resume checkpoint sampling_mode differs from requested mode")
    model.load_state_dict(checkpoint["raw_model_state_dict"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if ema is not None:
        if checkpoint.get("ema_state_dict") is None:
            raise ValueError("EMA is enabled but the resume checkpoint has no EMA state")
        ema.load_state_dict(checkpoint["ema_state_dict"])
    return (
        int(checkpoint["epoch"]) + 1,
        int(checkpoint.get("global_step", 0)),
        float(checkpoint.get("best_raw_validation_loss", float("inf"))),
        int(checkpoint.get("best_raw_epoch", 0)),
        float(checkpoint.get("best_ema_validation_loss", float("inf"))),
        int(checkpoint.get("best_ema_epoch", 0)),
        [dict(row) for row in checkpoint.get("training_history", [])],
    )


def copy_dataset_records(args: argparse.Namespace) -> None:
    copies = (
        (args.split_manifest, args.output_dir / "copied_split_manifest.json"),
        (args.normalization, args.output_dir / "copied_normalization.npz"),
        (args.dataset_metadata, args.output_dir / "copied_dataset_metadata.json"),
    )
    for source, destination in copies:
        if not source.is_file():
            raise FileNotFoundError(source)
        shutil.copy2(source, destination)


def main() -> int:
    args = parse_args()
    resolve_dataset_paths(args)
    validate_cli(args)
    if args.smoke_test:
        args.epochs = min(args.epochs, 2)
        args.max_train_batches = min(args.max_train_batches or 5, 5)
        args.max_val_batches = min(args.max_val_batches or 3, 3)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    protected_outputs = (
        "last_checkpoint.pt", "best_raw_checkpoint.pt", "best_ema_checkpoint.pt",
        "training_history.csv", "configuration.json",
    )
    if args.resume_checkpoint is None and any(
        (args.output_dir / name).exists() for name in protected_outputs
    ):
        raise FileExistsError(
            "Training outputs already exist; choose a new output directory or "
            "pass --resume_checkpoint"
        )

    v6.set_reproducibility(args.seed, args.deterministic)
    device = v6.resolve_device(args.device)
    train = load_windows(args.train_npz, "train")
    validation = load_windows(args.validation_npz, "validation")
    train_paths = set(train.path_names)
    validation_paths = set(validation.path_names)
    if train_paths & validation_paths:
        raise ValueError("Training and validation path names overlap")
    normalization = load_normalization(args.normalization)
    metadata = load_json(args.dataset_metadata)
    manifest = load_json(args.split_manifest)
    validate_metadata(metadata, normalization)
    if set(manifest.get("train_path_names", ())) != train_paths:
        raise ValueError("Training archive paths differ from split_manifest.json")
    if set(manifest.get("validation_path_names", ())) != validation_paths:
        raise ValueError("Validation archive paths differ from split_manifest.json")

    model, model_config = v6.instantiate_v5_model(
        HORIZON, CONDITION_DIM, TARGET_DIM, args.diffusion_steps
    )
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    ema = v6.ExponentialMovingAverage(model, args.ema_decay) if args.use_ema else None
    schedule = v6.build_schedule(args.diffusion_steps, device)
    train_loader, validation_loader = make_loaders(train, validation, args)
    validation_timesteps, validation_noise = v6.make_validation_bank(
        len(validation.path_names), HORIZON, TARGET_DIM, args.diffusion_steps, args.seed
    )
    residual_std = torch.from_numpy(
        np.asarray(normalization["residual_std"], dtype=np.float32)
    ).to(device)
    environment = software_device_information(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    copy_dataset_records(args)
    configuration: Dict[str, Any] = {
        "classification": "V7_TRAINING_IN_PROGRESS",
        "arguments": vars(args),
        "random_seed": args.seed,
        "reproducibility": {
            "deterministic_algorithms": args.deterministic,
            "validation_noise_bank_seed": args.seed + 1_000_003,
        },
        "software_device_information": environment,
        "model": model_config,
        "model_parameter_count": parameter_count,
        "horizon": HORIZON,
        "condition_dim": CONDITION_DIM,
        "target_dim": TARGET_DIM,
        "condition_feature_ordering": [
            str(value) for value in normalization["condition_feature_names"].tolist()
        ],
        "condition_feature_layout": list(v6.EXPECTED_CONDITION_LAYOUT),
        "diffusion_target": "residual_q_window",
        "prediction_target_type": PREDICTION_TARGET,
        "candidate_reconstruction": "prior_q_window + predicted_residual_q_window",
        "sampling_mode": args.sampling_mode,
        "training_loss_reduction": (
            "sum(sample_weight * per_sample_epsilon_mse) / sum(sample_weight)"
            if args.sampling_mode == "weighted_loss"
            else "mean(per_sample_epsilon_mse); ablation only"
        ),
        "validation_loss_reduction": "sample-weighted for every sampling mode",
        "optimizer": "AdamW (v6 settings)",
        "learning_rate_scheduler": "none (matches v6 trainer)",
        "train_path_names_sha256": path_set_hash(train_paths),
        "validation_path_names_sha256": path_set_hash(validation_paths),
        "train_target_row_count": len(train.path_names),
        "validation_target_row_count": len(validation.path_names),
        "train_unique_window_count": len(np.unique(train.unique_window_id)),
        "validation_unique_window_count": len(np.unique(validation.unique_window_id)),
        "normalization_source": "training paths only; validation excluded",
    }
    v6.atomic_json(configuration, args.output_dir / "configuration.json")
    with (args.output_dir / "model_architecture_summary.txt").open(
        "w", encoding="utf-8"
    ) as handle:
        handle.write(f"{model}\n\n")
        handle.write(f"model_class: {model_config['class_path']}\n")
        handle.write(f"constructor_kwargs: {model_config['constructor_kwargs']}\n")
        handle.write(f"parameter_count: {parameter_count}\n")
        handle.write(f"condition_dim: {CONDITION_DIM}\n")
        handle.write(f"target_dim: {TARGET_DIM}\n")
        handle.write(f"prediction_target: {PREDICTION_TARGET}\n")

    start_epoch = 1
    global_step = 0
    best_raw_loss = float("inf")
    best_raw_epoch = 0
    best_ema_loss = float("inf")
    best_ema_epoch = 0
    history: List[Dict[str, Any]] = []
    if args.resume_checkpoint is not None:
        (
            start_epoch, global_step, best_raw_loss, best_raw_epoch,
            best_ema_loss, best_ema_epoch, history,
        ) = load_resume(
            path=args.resume_checkpoint, model=model, optimizer=optimizer,
            ema=ema, device=device, args=args,
        )

    raw_checkpoint_dir = args.output_dir / "raw_checkpoints"
    ema_checkpoint_dir = args.output_dir / "ema_checkpoints"
    raw_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    if ema is not None:
        ema_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    early_stopping_counter = 0
    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, train_batches = train_epoch(
            model=model, loader=train_loader, optimizer=optimizer,
            schedule=schedule, device=device, diffusion_steps=args.diffusion_steps,
            gradient_clip_norm=args.gradient_clip_norm,
            max_batches=args.max_train_batches, ema=ema,
            sampling_mode=args.sampling_mode,
        )
        global_step += train_batches
        raw_metrics: Optional[Dict[str, Any]] = None
        ema_metrics: Optional[Dict[str, Any]] = None
        raw_improved = False
        ema_improved = False
        if epoch % args.validation_interval == 0:
            validation_kwargs = {
                "loader": validation_loader,
                "schedule": schedule,
                "device": device,
                "validation_timesteps": validation_timesteps,
                "validation_noise": validation_noise,
                "residual_std": residual_std,
                "max_batches": args.max_val_batches,
            }
            raw_metrics = validate_model(model=model, **validation_kwargs)
            raw_improved = raw_metrics["diffusion_loss"] < best_raw_loss
            if raw_improved:
                best_raw_loss = float(raw_metrics["diffusion_loss"])
                best_raw_epoch = epoch
            if ema is not None:
                ema_metrics = validate_with_ema(model, ema, validation_kwargs)
                ema_improved = ema_metrics["diffusion_loss"] < best_ema_loss
                if ema_improved:
                    best_ema_loss = float(ema_metrics["diffusion_loss"])
                    best_ema_epoch = epoch
            if raw_improved or ema_improved:
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1

        row: Dict[str, Any] = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": train_loss,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "raw_validation_loss": float("nan") if raw_metrics is None else raw_metrics["diffusion_loss"],
            "ema_validation_loss": float("nan") if ema_metrics is None else ema_metrics["diffusion_loss"],
            "raw_x0_rmse_normalized": float("nan") if raw_metrics is None else raw_metrics["x0_rmse_normalized"],
            "raw_x0_rmse_physical_rad": float("nan") if raw_metrics is None else raw_metrics["x0_rmse_physical_rad"],
            "ema_x0_rmse_normalized": float("nan") if ema_metrics is None else ema_metrics["x0_rmse_normalized"],
            "ema_x0_rmse_physical_rad": float("nan") if ema_metrics is None else ema_metrics["x0_rmse_physical_rad"],
            "best_raw_validation_loss": best_raw_loss,
            "best_raw_epoch": best_raw_epoch,
            "best_ema_validation_loss": best_ema_loss,
            "best_ema_epoch": best_ema_epoch,
            "early_stopping_counter": early_stopping_counter,
        }
        history.append(row)
        save_history_and_plot(history, args.output_dir)

        raw_payload = checkpoint_payload(
            model=model, optimizer=optimizer, ema=ema, selected_state="raw",
            epoch=epoch, global_step=global_step, best_raw_loss=best_raw_loss,
            best_raw_epoch=best_raw_epoch, best_ema_loss=best_ema_loss,
            best_ema_epoch=best_ema_epoch, model_config=model_config, args=args,
            normalization=normalization, dataset_metadata=metadata,
            split_manifest=manifest, history=history, environment=environment,
        )
        if raw_improved:
            v6.atomic_torch_save(raw_payload, args.output_dir / "best_raw_checkpoint.pt")
        ema_payload: Optional[Dict[str, Any]] = None
        if ema is not None:
            ema_payload = checkpoint_payload(
                model=model, optimizer=optimizer, ema=ema, selected_state="ema",
                epoch=epoch, global_step=global_step, best_raw_loss=best_raw_loss,
                best_raw_epoch=best_raw_epoch, best_ema_loss=best_ema_loss,
                best_ema_epoch=best_ema_epoch, model_config=model_config, args=args,
                normalization=normalization, dataset_metadata=metadata,
                split_manifest=manifest, history=history, environment=environment,
            )
            if ema_improved:
                v6.atomic_torch_save(ema_payload, args.output_dir / "best_ema_checkpoint.pt")
        if epoch % args.checkpoint_interval == 0:
            v6.atomic_torch_save(
                raw_payload, raw_checkpoint_dir / f"raw_checkpoint_epoch_{epoch:04d}.pt"
            )
            if ema_payload is not None:
                v6.atomic_torch_save(
                    ema_payload, ema_checkpoint_dir / f"ema_checkpoint_epoch_{epoch:04d}.pt"
                )
        v6.atomic_torch_save(raw_payload, args.output_dir / "last_checkpoint.pt")

        raw_text = "nan" if raw_metrics is None else f"{raw_metrics['diffusion_loss']:.8f}"
        ema_text = "nan" if ema_metrics is None else f"{ema_metrics['diffusion_loss']:.8f}"
        print(
            f"epoch={epoch} train_loss={train_loss:.8f} raw_val={raw_text} "
            f"ema_val={ema_text} best_raw_epoch={best_raw_epoch} "
            f"best_ema_epoch={best_ema_epoch}"
        )
        if raw_metrics is not None and early_stopping_counter >= args.early_stopping_patience:
            print(f"early stopping at epoch {epoch}")
            break

    if not (args.output_dir / "best_raw_checkpoint.pt").is_file():
        raise RuntimeError("No best raw checkpoint was saved")
    if ema is not None and not (args.output_dir / "best_ema_checkpoint.pt").is_file():
        raise RuntimeError("No best EMA checkpoint was saved")
    if not (args.output_dir / "last_checkpoint.pt").is_file():
        raise RuntimeError("No last checkpoint was saved")

    configuration.update(
        {
            "classification": "V7_TRAINING_COMPLETE",
            "best_raw_epoch": best_raw_epoch,
            "best_raw_validation_loss": best_raw_loss,
            "best_ema_epoch": best_ema_epoch if ema is not None else None,
            "best_ema_validation_loss": best_ema_loss if ema is not None else None,
        }
    )
    v6.atomic_json(configuration, args.output_dir / "configuration.json")
    print(f"best raw checkpoint: {args.output_dir / 'best_raw_checkpoint.pt'}")
    if ema is not None:
        print(f"best EMA checkpoint: {args.output_dir / 'best_ema_checkpoint.pt'}")
    print(f"last checkpoint: {args.output_dir / 'last_checkpoint.pt'}")
    print("classification: V7_TRAINING_COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
