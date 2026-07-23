#!/usr/bin/env python3
"""Train v8 scale-conditioned diffusion on diverse residual targets.

The model, epsilon-prediction convention, linear schedule, tensor ordering,
AdamW optimizer, and EMA implementation are inherited from the v7/v6
trainers. V8 adds deterministic row balancing and physical reconstructed-x0
auxiliary losses without changing the U-Net capacity.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
import math
import os
import platform
import random
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import (
    DataLoader,
    Dataset,
    RandomSampler,
    WeightedRandomSampler,
)

import train_conditional_diffusion_trajectory_v6_strong_prior_residual_unet as v6


HORIZON = 32
CONDITION_DIM = 39
TARGET_DIM = 6
PREDICTION_TARGET = "epsilon"
EXPECTED_SCALES = (0.125, 0.25, 0.50, 0.75, 1.0)
AUXILIARY_SCALE_EPSILON = 1.0e-8
FLOAT32_NORMALIZATION_AUDIT_ATOL = 1.0e-5
AUXILIARY_SCALE_NAMES = (
    "position_scale",
    "velocity_scale",
    "acceleration_scale",
    "jerk_scale",
)
OUTPUT_FILES = (
    "best_raw_total_loss_checkpoint.pt",
    "best_ema_total_loss_checkpoint.pt",
    "best_raw_epsilon_loss_checkpoint.pt",
    "best_ema_epsilon_loss_checkpoint.pt",
    "last_checkpoint.pt",
    "training_history.csv",
    "training_history.json",
    "training_and_validation_loss.png",
    "loss_component_history.png",
    "validation_loss_by_scale.csv",
    "validation_loss_by_timestep_bin.csv",
    "sampling_diagnostics.csv",
    "training_metadata.json",
    "auxiliary_loss_normalization.npz",
    "dataset_integrity_report.json",
)
LOSS_NAMES = (
    "total_loss",
    "epsilon_loss",
    "x0_loss",
    "velocity_loss",
    "acceleration_loss",
    "jerk_loss",
    "boundary_loss",
)
RMSE_NAMES = (
    "physical_residual_rmse_rad",
    "execution_prefix_residual_rmse_rad",
    "full_window_residual_rmse_rad",
)


@dataclass(frozen=True)
class WindowArrays:
    source: Path
    condition_norm: np.ndarray
    residual_norm: np.ndarray
    residual_physical: np.ndarray
    path_names: Tuple[str, ...]
    window_starts: np.ndarray
    target_scales: np.ndarray
    quality_weight: np.ndarray
    window_balance_weight: np.ndarray
    combined_sample_weight: np.ndarray
    target_ids: Tuple[str, ...]
    condition_feature_names: Tuple[str, ...]
    keys: Tuple[str, ...]


class ResidualTargetDataset(
    Dataset[Tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]]
):
    def __init__(self, arrays: WindowArrays) -> None:
        self.condition = torch.from_numpy(arrays.condition_norm)
        self.target = torch.from_numpy(arrays.residual_norm)
        self.scale = torch.from_numpy(arrays.target_scales.astype(np.float32))

    def __len__(self) -> int:
        return int(self.target.shape[0])

    def __getitem__(
        self, index: int
    ) -> Tuple[torch.Tensor, torch.Tensor, int, torch.Tensor]:
        return self.condition[index], self.target[index], index, self.scale[index]


class MetricAccumulator:
    def __init__(self) -> None:
        self.sums = {name: 0.0 for name in (*LOSS_NAMES, *RMSE_NAMES)}
        self.count = 0

    def add(self, values: Mapping[str, torch.Tensor], mask: torch.Tensor) -> None:
        selected_count = int(mask.sum().item())
        if selected_count == 0:
            return
        for name in LOSS_NAMES:
            self.sums[name] += float(values[name][mask].detach().double().sum().cpu())
        self.sums["physical_residual_rmse_rad"] += float(
            values["physical_residual_mse_rad2"][mask].detach().double().sum().cpu()
        )
        self.sums["execution_prefix_residual_rmse_rad"] += float(
            values["execution_prefix_residual_mse_rad2"][mask]
            .detach().double().sum().cpu()
        )
        self.sums["full_window_residual_rmse_rad"] += float(
            values["full_window_residual_mse_rad2"][mask]
            .detach().double().sum().cpu()
        )
        self.count += selected_count

    def finalize(self) -> Dict[str, float]:
        if self.count <= 0:
            return {name: math.nan for name in (*LOSS_NAMES, *RMSE_NAMES)}
        result = {
            name: self.sums[name] / self.count for name in LOSS_NAMES
        }
        for name in RMSE_NAMES:
            result[name] = math.sqrt(max(self.sums[name] / self.count, 0.0))
        return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train v8 multitarget scale-conditioned residual diffusion."
    )
    parser.add_argument("--dataset_dir", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=1.0e-4)
    parser.add_argument("--weight_decay", type=float, default=1.0e-6)
    parser.add_argument("--num_diffusion_steps", type=int, default=1000)
    parser.add_argument("--num_data_workers", type=int, default=6)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument(
        "--persistent_workers", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument(
        "--pin_memory", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--amp_initial_scale", type=float, default=128.0)
    parser.add_argument("--gradient_clip_norm", type=float, default=1.0)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--early_stopping_patience", type=int, default=80)
    parser.add_argument("--checkpoint_interval", type=int, default=25)
    parser.add_argument("--resume_checkpoint", type=Path, default=None)
    parser.add_argument("--init_checkpoint", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max_train_batches", type=int, default=0)
    parser.add_argument("--max_validation_batches", type=int, default=0)
    parser.add_argument(
        "--deterministic_algorithms",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--sampling_mode", choices=("balanced", "uniform"), default="balanced"
    )
    parser.add_argument("--window_balance_power", type=float, default=1.0)
    parser.add_argument("--scale_balance_power", type=float, default=1.0)
    parser.add_argument("--quality_weight_power", type=float, default=0.25)
    parser.add_argument("--sampler_weight_clip_min", type=float, default=0.1)
    parser.add_argument("--sampler_weight_clip_max", type=float, default=10.0)
    parser.add_argument("--lambda_epsilon", type=float, default=1.0)
    parser.add_argument("--lambda_x0", type=float, default=0.25)
    parser.add_argument("--lambda_velocity", type=float, default=0.05)
    parser.add_argument("--lambda_acceleration", type=float, default=0.10)
    parser.add_argument("--lambda_jerk", type=float, default=0.05)
    parser.add_argument("--lambda_boundary", type=float, default=0.10)
    parser.add_argument("--execution_horizon", type=int, default=8)
    parser.add_argument("--execution_prefix_weight", type=float, default=2.0)
    parser.add_argument(
        "--early_stopping_metric",
        choices=(
            "ema_total_loss",
            "raw_total_loss",
            "ema_epsilon_loss",
            "raw_epsilon_loss",
        ),
        default="ema_total_loss",
    )
    return parser.parse_args()


def validate_cli(args: argparse.Namespace) -> None:
    for name in (
        "epochs",
        "batch_size",
        "num_diffusion_steps",
        "prefetch_factor",
        "early_stopping_patience",
        "checkpoint_interval",
        "execution_horizon",
    ):
        if int(getattr(args, name)) < 1:
            raise ValueError(f"--{name} must be at least 1")
    for name in ("num_data_workers", "max_train_batches", "max_validation_batches"):
        if int(getattr(args, name)) < 0:
            raise ValueError(f"--{name} must be non-negative")
    if args.execution_horizon > HORIZON:
        raise ValueError(f"--execution_horizon cannot exceed {HORIZON}")
    if args.learning_rate <= 0.0 or args.weight_decay < 0.0:
        raise ValueError("Learning rate must be positive and weight decay non-negative")
    if args.gradient_clip_norm < 0.0:
        raise ValueError("--gradient_clip_norm must be non-negative")
    if not np.isfinite(args.amp_initial_scale) or args.amp_initial_scale <= 0.0:
        raise ValueError("--amp_initial_scale must be positive and finite")
    if not 0.0 < args.ema_decay < 1.0:
        raise ValueError("--ema_decay must lie in (0,1)")
    for name in (
        "window_balance_power",
        "scale_balance_power",
        "quality_weight_power",
        "lambda_epsilon",
        "lambda_x0",
        "lambda_velocity",
        "lambda_acceleration",
        "lambda_jerk",
        "lambda_boundary",
    ):
        if float(getattr(args, name)) < 0.0:
            raise ValueError(f"--{name} must be non-negative")
    if not any(
        float(getattr(args, name)) > 0.0
        for name in (
            "lambda_epsilon",
            "lambda_x0",
            "lambda_velocity",
            "lambda_acceleration",
            "lambda_jerk",
            "lambda_boundary",
        )
    ):
        raise ValueError("At least one loss weight must be positive")
    if args.execution_prefix_weight <= 0.0:
        raise ValueError("--execution_prefix_weight must be positive")
    if not (
        0.0 < args.sampler_weight_clip_min <= 1.0
        <= args.sampler_weight_clip_max
    ):
        raise ValueError("Sampler clipping must satisfy 0 < min <= 1 <= max")
    if args.resume_checkpoint is not None and args.init_checkpoint is not None:
        raise ValueError("--resume_checkpoint and --init_checkpoint are exclusive")
    if args.resume_checkpoint is not None and args.overwrite:
        raise ValueError("--resume_checkpoint cannot be combined with --overwrite")


def decode_strings(values: np.ndarray) -> Tuple[str, ...]:
    return tuple(
        value.decode("utf-8", errors="strict")
        if isinstance(value, bytes)
        else str(value)
        for value in np.asarray(values).reshape(-1)
    )


def load_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return payload


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def atomic_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    temporary = path.with_name(f".{path.name}.tmp.npz")
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_torch_checkpoint(path: Path, device: torch.device) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    try:
        payload = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location=device)
    if not isinstance(payload, dict):
        raise TypeError(f"Checkpoint {path} must contain a dictionary")
    return payload


def load_normalization(path: Path) -> Dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        required = (
            "condition_mean",
            "condition_std",
            "residual_mean",
            "residual_std",
            "condition_feature_names",
            "condition_dim",
            "target_dim",
            "horizon",
            "target_scale_mean",
            "target_scale_std",
        )
        missing = [key for key in required if key not in archive.files]
        if missing:
            raise KeyError(f"{path} is missing normalization keys: {missing}")
        result = {key: np.asarray(archive[key]) for key in archive.files}
    result["condition_mean"] = np.asarray(
        result["condition_mean"], dtype=np.float64
    ).reshape(-1)
    result["condition_std"] = np.asarray(
        result["condition_std"], dtype=np.float64
    ).reshape(-1)
    result["residual_mean"] = np.asarray(
        result["residual_mean"], dtype=np.float64
    ).reshape(-1)
    result["residual_std"] = np.asarray(
        result["residual_std"], dtype=np.float64
    ).reshape(-1)
    expected_shapes = {
        "condition_mean": (CONDITION_DIM,),
        "condition_std": (CONDITION_DIM,),
        "residual_mean": (TARGET_DIM,),
        "residual_std": (TARGET_DIM,),
    }
    for key, expected in expected_shapes.items():
        if result[key].shape != expected:
            raise ValueError(f"{path}/{key} has shape {result[key].shape}, expected {expected}")
        if not np.all(np.isfinite(result[key])):
            raise ValueError(f"{path}/{key} contains NaN or infinity")
    if np.any(result["condition_std"] <= 0.0) or np.any(result["residual_std"] <= 0.0):
        raise ValueError("Normalization standard deviations must be positive")
    for key, expected in (
        ("condition_dim", CONDITION_DIM),
        ("target_dim", TARGET_DIM),
        ("horizon", HORIZON),
    ):
        if int(np.asarray(result[key]).item()) != expected:
            raise ValueError(f"Normalization {key} is incompatible with v8")
    names = decode_strings(result["condition_feature_names"])
    if len(names) != CONDITION_DIM or len(set(names)) != CONDITION_DIM:
        raise ValueError("Normalization must contain 39 unique condition features")
    if names[-1] != "target_scale":
        raise ValueError("target_scale must be the final condition feature")
    if not np.isclose(float(result["target_scale_mean"]), 0.0) or not np.isclose(
        float(result["target_scale_std"]), 1.0
    ):
        raise ValueError("target_scale normalization must be the raw identity transform")
    return result


def load_windows(path: Path, label: str) -> WindowArrays:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        required = (
            "condition",
            "condition_norm",
            "residual_q",
            "residual_q_norm",
            "path_names",
            "window_start_indices",
            "target_scale",
            "quality_weight",
            "window_balance_weight",
            "combined_sample_weight",
            "target_id",
            "condition_feature_names",
            "condition_dim",
            "target_dim",
            "horizon",
        )
        missing = [key for key in required if key not in archive.files]
        if missing:
            raise KeyError(f"{path} is missing required keys: {missing}")
        keys = tuple(archive.files)
        condition = np.asarray(archive["condition_norm"], dtype=np.float32)
        condition_raw = np.asarray(archive["condition"], dtype=np.float32)
        residual_norm = np.asarray(archive["residual_q_norm"], dtype=np.float32)
        residual = np.asarray(archive["residual_q"], dtype=np.float32)
        names = decode_strings(archive["path_names"])
        starts = np.asarray(archive["window_start_indices"], dtype=np.int64).reshape(-1)
        scales = np.asarray(archive["target_scale"], dtype=np.float64).reshape(-1)
        quality = np.asarray(archive["quality_weight"], dtype=np.float64).reshape(-1)
        window_weight = np.asarray(
            archive["window_balance_weight"], dtype=np.float64
        ).reshape(-1)
        combined = np.asarray(
            archive["combined_sample_weight"], dtype=np.float64
        ).reshape(-1)
        target_ids = decode_strings(archive["target_id"])
        feature_names = decode_strings(archive["condition_feature_names"])
        dimensions = {
            "condition_dim": int(np.asarray(archive["condition_dim"]).item()),
            "target_dim": int(np.asarray(archive["target_dim"]).item()),
            "horizon": int(np.asarray(archive["horizon"]).item()),
        }
    count = len(names)
    if condition.shape != (count, HORIZON, CONDITION_DIM):
        raise ValueError(f"{label} condition_norm shape is {condition.shape}")
    if condition_raw.shape != condition.shape:
        raise ValueError(f"{label} raw and normalized conditions differ in shape")
    if residual_norm.shape != (count, HORIZON, TARGET_DIM):
        raise ValueError(f"{label} residual_q_norm shape is {residual_norm.shape}")
    if residual.shape != residual_norm.shape:
        raise ValueError(f"{label} physical and normalized residuals differ in shape")
    if dimensions != {
        "condition_dim": CONDITION_DIM,
        "target_dim": TARGET_DIM,
        "horizon": HORIZON,
    }:
        raise ValueError(f"{label} dimensions are incompatible: {dimensions}")
    row_values: Iterable[Tuple[str, Any]] = (
        ("window_start_indices", starts),
        ("target_scale", scales),
        ("quality_weight", quality),
        ("window_balance_weight", window_weight),
        ("combined_sample_weight", combined),
        ("target_id", target_ids),
    )
    for key, values in row_values:
        if len(values) != count:
            raise ValueError(f"{label} {key} row count differs from conditions")
    numeric = (condition, condition_raw, residual_norm, residual, scales, quality, window_weight, combined)
    if any(not np.all(np.isfinite(value)) for value in numeric):
        raise ValueError(f"{label} contains nonfinite numeric values")
    if np.any(scales <= 0.0) or np.any(quality <= 0.0) or np.any(window_weight <= 0.0) or np.any(combined <= 0.0):
        raise ValueError(f"{label} scale and weight arrays must be positive")
    if len(set(target_ids)) != count:
        raise ValueError(f"{label} target_id values are not unique")
    if len(feature_names) != CONDITION_DIM or feature_names[-1] != "target_scale":
        raise ValueError(f"{label} condition feature ordering is incompatible")
    if not np.allclose(condition_raw[:, :, -1], scales[:, None], atol=1.0e-7):
        raise ValueError(f"{label} raw target_scale channel is inconsistent")
    if not np.allclose(condition[:, :, -1], scales[:, None], atol=1.0e-7):
        raise ValueError(f"{label} normalized target_scale must remain raw")
    return WindowArrays(
        source=path,
        condition_norm=condition,
        residual_norm=residual_norm,
        residual_physical=residual,
        path_names=names,
        window_starts=starts,
        target_scales=scales,
        quality_weight=quality,
        window_balance_weight=window_weight,
        combined_sample_weight=combined,
        target_ids=target_ids,
        condition_feature_names=feature_names,
        keys=keys,
    )


def window_identities(arrays: WindowArrays) -> List[Tuple[str, int]]:
    return [
        (path_name, int(start))
        for path_name, start in zip(arrays.path_names, arrays.window_starts)
    ]


def validate_row_csv(path: Path, arrays: WindowArrays, label: str) -> None:
    frame = pd.read_csv(path)
    required = (
        "path_name",
        "window_start",
        "target_id",
        "target_scale",
        "cartesian_improvement_m",
        "delta_score",
        "quality_weight",
        "window_balance_weight",
        "combined_sample_weight",
    )
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise KeyError(f"{path} is missing columns: {missing}")
    if len(frame) != len(arrays.path_names):
        raise ValueError(f"{label} row CSV count differs from its NPZ")
    if frame["path_name"].astype(str).tolist() != list(arrays.path_names):
        raise ValueError(f"{label} row CSV path ordering differs from its NPZ")
    if not np.array_equal(
        frame["window_start"].to_numpy(dtype=np.int64), arrays.window_starts
    ):
        raise ValueError(f"{label} row CSV window ordering differs from its NPZ")
    if frame["target_id"].astype(str).tolist() != list(arrays.target_ids):
        raise ValueError(f"{label} row CSV target IDs differ from its NPZ")
    for column, expected in (
        ("target_scale", arrays.target_scales),
        ("quality_weight", arrays.quality_weight),
        ("window_balance_weight", arrays.window_balance_weight),
        ("combined_sample_weight", arrays.combined_sample_weight),
    ):
        if not np.allclose(
            frame[column].to_numpy(dtype=np.float64), expected, atol=2.0e-6
        ):
            raise ValueError(f"{label} row CSV {column} differs from its NPZ")
    improvement = np.maximum(
        frame["cartesian_improvement_m"].to_numpy(dtype=np.float64), 0.0
    )
    score_gain = np.maximum(
        -frame["delta_score"].to_numpy(dtype=np.float64), 0.0
    )
    improvement_reference = max(float(np.median(improvement)), 1.0e-12)
    score_reference = max(float(np.median(score_gain)), 1.0e-12)
    quality_raw = 0.5 * (
        improvement / improvement_reference + score_gain / score_reference
    )
    quality_clipped = np.clip(quality_raw, 0.25, 4.0)
    expected_quality = quality_clipped / np.mean(quality_clipped)
    if not np.allclose(expected_quality, arrays.quality_weight, atol=2.0e-6):
        raise ValueError(f"{label} quality_weight definition is inconsistent")


def validate_dataset(
    train: WindowArrays,
    validation: WindowArrays,
    normalization: Mapping[str, np.ndarray],
    metadata: Mapping[str, Any],
    normalization_path: Path,
    metadata_path: Path,
) -> Dict[str, Any]:
    if metadata.get("classification") != "READY_FOR_V8_TRAINING":
        raise ValueError("Dataset is not classified READY_FOR_V8_TRAINING")
    condition_metadata = metadata.get("condition", {})
    if int(condition_metadata.get("v8_condition_dim", -1)) != CONDITION_DIM:
        raise ValueError("Dataset metadata does not declare condition_dim=39")
    if condition_metadata.get("appended_feature") != "target_scale":
        raise ValueError("Dataset metadata does not identify target_scale as appended")
    if "raw" not in str(condition_metadata.get("target_scale_normalization", "")).lower():
        raise ValueError("Dataset target_scale normalization is ambiguous")
    target_metadata = metadata.get("target", {})
    if list(target_metadata.get("shape", ())) != [HORIZON, TARGET_DIM]:
        raise ValueError("Dataset metadata target shape is incompatible")
    normalization_metadata = metadata.get("normalization", {})
    if normalization_metadata.get("source_split") != "training rows only":
        raise ValueError("Normalization is not explicitly training-only")
    feature_names = tuple(str(value) for value in condition_metadata.get("feature_names", ()))
    normalization_names = decode_strings(normalization["condition_feature_names"])
    if not (
        train.condition_feature_names
        == validation.condition_feature_names
        == normalization_names
        == feature_names
    ):
        raise ValueError("Condition feature ordering differs across dataset artifacts")
    train_paths = set(train.path_names)
    validation_paths = set(validation.path_names)
    path_overlap = train_paths & validation_paths
    train_windows = set(window_identities(train))
    validation_windows = set(window_identities(validation))
    window_overlap = train_windows & validation_windows
    if path_overlap or window_overlap:
        raise ValueError(
            f"Train/validation overlap: paths={sorted(path_overlap)}, "
            f"windows={sorted(window_overlap)[:10]}"
        )
    condition_mean = np.asarray(normalization["condition_mean"]).reshape(1, 1, -1)
    condition_std = np.asarray(normalization["condition_std"]).reshape(1, 1, -1)
    residual_mean = np.asarray(normalization["residual_mean"]).reshape(1, 1, -1)
    residual_std = np.asarray(normalization["residual_std"]).reshape(1, 1, -1)
    condition_normalization_max_abs_error: Dict[str, float] = {}
    for label, arrays in (("train", train), ("validation", validation)):
        reconstructed_residual = (
            arrays.residual_norm.astype(np.float64) * residual_std + residual_mean
        )
        if not np.allclose(
            reconstructed_residual,
            arrays.residual_physical.astype(np.float64),
            rtol=2.0e-5,
            atol=2.0e-7,
        ):
            raise ValueError(f"{label} residual normalization is inconsistent")
        with np.load(arrays.source, allow_pickle=False) as archive:
            raw_condition = np.asarray(archive["condition"], dtype=np.float64)
        expected_condition = (raw_condition - condition_mean) / condition_std
        condition_normalization_max_abs_error[label] = float(
            np.max(
                np.abs(
                    expected_condition
                    - arrays.condition_norm.astype(np.float64)
                )
            )
        )
        if not np.allclose(
            expected_condition,
            arrays.condition_norm.astype(np.float64),
            rtol=2.0e-5,
            atol=FLOAT32_NORMALIZATION_AUDIT_ATOL,
        ):
            raise ValueError(f"{label} condition normalization is inconsistent")
    expected_residual_mean = np.mean(
        train.residual_physical.astype(np.float64), axis=(0, 1)
    )
    expected_residual_std = np.std(
        train.residual_physical.astype(np.float64), axis=(0, 1)
    )
    residual_normalization_epsilon = float(
        np.asarray(normalization.get("normalization_epsilon", 1.0e-8)).item()
    )
    expected_residual_std = np.where(
        expected_residual_std > residual_normalization_epsilon,
        expected_residual_std,
        1.0,
    )
    if not np.allclose(expected_residual_mean, normalization["residual_mean"], atol=1.0e-7):
        raise ValueError("Residual mean was not derived from training rows")
    if not np.allclose(expected_residual_std, normalization["residual_std"], atol=1.0e-7):
        raise ValueError("Residual std was not derived from training rows")
    with np.load(train.source, allow_pickle=False) as archive:
        train_raw_condition = np.asarray(archive["condition"], dtype=np.float64)
    expected_condition_mean = np.mean(train_raw_condition, axis=(0, 1))
    expected_condition_std = np.std(train_raw_condition, axis=(0, 1))
    normalization_epsilon = float(
        np.asarray(normalization.get("normalization_epsilon", 1.0e-8)).item()
    )
    expected_condition_std = np.where(
        expected_condition_std > normalization_epsilon,
        expected_condition_std,
        1.0,
    )
    expected_condition_mean[-1] = 0.0
    expected_condition_std[-1] = 1.0
    if not np.allclose(
        expected_condition_mean, normalization["condition_mean"], atol=1.0e-7
    ):
        raise ValueError("Condition mean was not derived from training rows")
    if not np.allclose(
        expected_condition_std, normalization["condition_std"], atol=1.0e-7
    ):
        raise ValueError("Condition std was not derived from training rows")
    metadata_stats = normalization_metadata.get("statistics", {})
    for key in ("condition_mean", "condition_std", "residual_mean", "residual_std"):
        if key not in metadata_stats or not np.allclose(
            np.asarray(metadata_stats[key], dtype=np.float64).reshape(-1),
            np.asarray(normalization[key], dtype=np.float64).reshape(-1),
            atol=1.0e-7,
        ):
            raise ValueError(f"Metadata and normalization_stats disagree for {key}")
    counts = metadata.get("counts", {})
    actual_counts = {
        "train_rows": len(train.path_names),
        "validation_rows": len(validation.path_names),
        "train_paths": len(train_paths),
        "validation_paths": len(validation_paths),
        "train_windows": len(train_windows),
        "validation_windows": len(validation_windows),
    }
    for key, value in actual_counts.items():
        if int(counts.get(key, -1)) != value:
            raise ValueError(f"Metadata count {key} differs: {counts.get(key)} != {value}")
    for label, arrays in (("train", train), ("validation", validation)):
        identities = window_identities(arrays)
        multiplicity: Dict[Tuple[str, int], int] = {}
        for identity in identities:
            multiplicity[identity] = multiplicity.get(identity, 0) + 1
        expected_window_weight = np.asarray(
            [1.0 / multiplicity[identity] for identity in identities]
        )
        if not np.allclose(arrays.window_balance_weight, expected_window_weight, atol=1.0e-7):
            raise ValueError(f"{label} builder window_balance_weight is inconsistent")
        combined_raw = arrays.quality_weight * arrays.window_balance_weight
        expected_combined = combined_raw / np.mean(combined_raw)
        if not np.allclose(arrays.combined_sample_weight, expected_combined, atol=2.0e-6):
            raise ValueError(f"{label} builder combined_sample_weight is inconsistent")
    observed_train_scales = set(np.round(train.target_scales, 6).tolist())
    observed_validation_scales = set(np.round(validation.target_scales, 6).tolist())
    expected_scales = set(EXPECTED_SCALES)
    if observed_train_scales != expected_scales or observed_validation_scales != expected_scales:
        raise ValueError(
            "Expected all five target scales in both splits; "
            f"train={sorted(observed_train_scales)}, validation={sorted(observed_validation_scales)}"
        )
    return {
        "classification": "V8_DATASET_INTEGRITY_CONFIRMED",
        "dataset_directory": str(train.source.parent.resolve()),
        "train_npz_keys": list(train.keys),
        "validation_npz_keys": list(validation.keys),
        "fields_consumed": [
            "condition_norm",
            "residual_q_norm",
            "residual_q",
            "path_names",
            "window_start_indices",
            "target_scale",
            "quality_weight",
            "window_balance_weight",
            "combined_sample_weight",
            "target_id",
            "condition_feature_names",
            "condition_dim",
            "target_dim",
            "horizon",
        ],
        "row_csv_fields_consumed": [
            "path_name",
            "window_start",
            "target_id",
            "target_scale",
            "cartesian_improvement_m",
            "delta_score",
            "quality_weight",
            "window_balance_weight",
            "combined_sample_weight",
        ],
        "counts": actual_counts,
        "condition_shape": [HORIZON, CONDITION_DIM],
        "target_shape": [HORIZON, TARGET_DIM],
        "condition_feature_names": list(feature_names),
        "target_scale_is_final_feature": True,
        "target_scale_identity_normalization": True,
        "path_overlap_count": 0,
        "window_overlap_count": 0,
        "nonfinite_count": 0,
        "normalization_source": "training rows only",
        "normalization_applied_by_trainer": False,
        "float32_condition_normalization_audit_atol": (
            FLOAT32_NORMALIZATION_AUDIT_ATOL
        ),
        "condition_normalization_max_abs_error": (
            condition_normalization_max_abs_error
        ),
        "normalization_stats_sha256": file_sha256(normalization_path),
        "dataset_metadata_sha256": file_sha256(metadata_path),
        "builder_weight_definitions_verified": True,
        "row_csvs_verified_against_npz": True,
        "compatibility_note": (
            "dataset_metadata.compatibility.v7_loader_keys_preserved is not "
            "literal compatibility with the current v7 trainer: v8 stores "
            "condition_norm/window_start_indices and has no v7 unique_window_id; "
            "this trainer consumes the actual v8 schema explicitly"
        ),
    }


def robust_joint_scale(values: np.ndarray) -> Dict[str, np.ndarray]:
    flattened = np.asarray(values, dtype=np.float64).reshape(-1, TARGET_DIM)
    standard = np.std(flattened, axis=0)
    median = np.median(flattened, axis=0)
    mad_scale = 1.4826 * np.median(np.abs(flattened - median), axis=0)
    use_standard = np.isfinite(standard) & (standard > AUXILIARY_SCALE_EPSILON)
    use_mad = np.isfinite(mad_scale) & (mad_scale > AUXILIARY_SCALE_EPSILON)
    scale = np.where(use_standard, standard, np.where(use_mad, mad_scale, AUXILIARY_SCALE_EPSILON))
    return {
        "scale": scale.astype(np.float64),
        "standard_deviation": standard.astype(np.float64),
        "mad_scale": mad_scale.astype(np.float64),
    }


def build_auxiliary_normalization(residual: np.ndarray) -> Dict[str, np.ndarray]:
    current = np.asarray(residual, dtype=np.float64)
    result: Dict[str, np.ndarray] = {
        "epsilon_floor": np.asarray(AUXILIARY_SCALE_EPSILON, dtype=np.float64),
        "source_row_count": np.asarray(current.shape[0], dtype=np.int64),
        "source": np.asarray("training residual_q only"),
    }
    for order, name in enumerate(("position", "velocity", "acceleration", "jerk")):
        if order > 0:
            current = np.diff(current, axis=1)
        statistics = robust_joint_scale(current)
        result[f"{name}_scale"] = statistics["scale"]
        result[f"{name}_standard_deviation"] = statistics["standard_deviation"]
        result[f"{name}_mad_scale"] = statistics["mad_scale"]
    return result


def bounded_mean_one_weights(
    raw: np.ndarray, lower: float, upper: float
) -> np.ndarray:
    values = np.asarray(raw, dtype=np.float64)
    if np.any(~np.isfinite(values)) or np.any(values <= 0.0):
        raise ValueError("Raw sampler weights must be finite and positive")
    low_scale = 0.0
    high_scale = 1.0 / max(float(np.min(values)), np.finfo(np.float64).tiny)
    for _ in range(100):
        middle = 0.5 * (low_scale + high_scale)
        mean = float(np.mean(np.clip(values * middle, lower, upper)))
        if mean < 1.0:
            low_scale = middle
        else:
            high_scale = middle
    result = np.clip(values * high_scale, lower, upper)
    if not np.isclose(float(np.mean(result)), 1.0, rtol=1.0e-10, atol=1.0e-10):
        raise RuntimeError("Could not normalize clipped sampler weights to mean one")
    return result


def build_sampler_weights(
    train: WindowArrays, args: argparse.Namespace
) -> Tuple[np.ndarray, pd.DataFrame, Dict[str, Any]]:
    identities = window_identities(train)
    window_counts: Dict[Tuple[str, int], int] = {}
    for identity in identities:
        window_counts[identity] = window_counts.get(identity, 0) + 1
    rounded_scales = np.round(train.target_scales, 6)
    unique_scales, counts = np.unique(rounded_scales, return_counts=True)
    scale_counts = {
        float(scale): int(count) for scale, count in zip(unique_scales, counts)
    }
    window_factor = np.asarray(
        [window_counts[identity] ** (-args.window_balance_power) for identity in identities],
        dtype=np.float64,
    )
    scale_factor = np.asarray(
        [scale_counts[float(scale)] ** (-args.scale_balance_power) for scale in rounded_scales],
        dtype=np.float64,
    )
    quality_factor = np.power(train.quality_weight, args.quality_weight_power)
    raw = window_factor * scale_factor * quality_factor
    balanced = bounded_mean_one_weights(
        raw, args.sampler_weight_clip_min, args.sampler_weight_clip_max
    )
    effective = balanced if args.sampling_mode == "balanced" else np.ones_like(balanced)
    total_weight = float(np.sum(effective))
    row_count = len(effective)
    records: List[Dict[str, Any]] = []
    for scale in sorted(scale_counts):
        mask = rounded_scales == scale
        expected = row_count * float(np.sum(effective[mask])) / total_weight
        records.append(
            {
                "summary_type": "target_scale",
                "target_scale": scale,
                "target_count_per_window": math.nan,
                "row_count": int(mask.sum()),
                "window_count": len({identities[index] for index in np.flatnonzero(mask)}),
                "original_row_fraction": float(mask.mean()),
                "expected_sampled_rows": expected,
                "expected_sampled_fraction": expected / row_count,
                "mean_sampler_weight": float(np.mean(effective[mask])),
                "minimum_sampler_weight": float(np.min(effective[mask])),
                "maximum_sampler_weight": float(np.max(effective[mask])),
            }
        )
    expected_by_window: Dict[Tuple[str, int], float] = {}
    for index, identity in enumerate(identities):
        expected_by_window[identity] = expected_by_window.get(identity, 0.0) + (
            row_count * effective[index] / total_weight
        )
    windows_by_count: Dict[int, List[Tuple[str, int]]] = {}
    for identity, count in window_counts.items():
        windows_by_count.setdefault(count, []).append(identity)
    for count, window_keys in sorted(windows_by_count.items()):
        window_key_set = set(window_keys)
        expected_values = np.asarray(
            [expected_by_window[identity] for identity in window_keys], dtype=np.float64
        )
        records.append(
            {
                "summary_type": "window_target_count",
                "target_scale": math.nan,
                "target_count_per_window": count,
                "row_count": count * len(window_keys),
                "window_count": len(window_keys),
                "original_row_fraction": count * len(window_keys) / row_count,
                "expected_sampled_rows": float(np.sum(expected_values)),
                "expected_sampled_fraction": float(np.sum(expected_values)) / row_count,
                "mean_sampler_weight": float(
                    np.mean([
                        effective[index]
                        for index, identity in enumerate(identities)
                        if identity in window_key_set
                    ])
                ),
                "minimum_sampler_weight": math.nan,
                "maximum_sampler_weight": math.nan,
                "mean_expected_rows_per_window": float(np.mean(expected_values)),
                "minimum_expected_rows_per_window": float(np.min(expected_values)),
                "maximum_expected_rows_per_window": float(np.max(expected_values)),
            }
        )
    percentiles = np.percentile(effective, [1, 5, 25, 50, 75, 95, 99])
    diagnostics = {
        "sampling_mode": args.sampling_mode,
        "formula": (
            "window_count^-window_balance_power * scale_count^-scale_balance_power "
            "* quality_weight^quality_weight_power"
        ),
        "weights_applied_through_sampler_only": True,
        "row_count": row_count,
        "minimum": float(np.min(effective)),
        "maximum": float(np.max(effective)),
        "mean": float(np.mean(effective)),
        "median": float(np.median(effective)),
        "percentiles": {
            str(percentile): float(value)
            for percentile, value in zip((1, 5, 25, 50, 75, 95, 99), percentiles)
        },
        "effective_sample_size": float(total_weight**2 / np.sum(np.square(effective))),
        "original_scale_distribution": {
            format(scale, ".6g"): count for scale, count in sorted(scale_counts.items())
        },
        "expected_sampled_scale_distribution": {
            format(scale, ".6g"): float(
                row_count * np.sum(effective[rounded_scales == scale]) / total_weight
            )
            for scale in sorted(scale_counts)
        },
        "expected_rows_sampled_per_window": {
            f"{path_name}::{window_start}": value
            for (path_name, window_start), value in sorted(expected_by_window.items())
        },
    }
    return effective, pd.DataFrame(records), diagnostics


def data_worker_init(worker_id: int) -> None:
    del worker_id
    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[variable] = "1"
    torch.set_num_threads(1)
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def make_loaders(
    train: WindowArrays,
    validation: WindowArrays,
    sampler_weights: np.ndarray,
    args: argparse.Namespace,
    pin_memory: bool,
) -> Tuple[
    DataLoader[Any],
    DataLoader[Any],
    DataLoader[Any],
    torch.Generator,
    torch.Generator,
]:
    train_dataset = ResidualTargetDataset(train)
    validation_dataset = ResidualTargetDataset(validation)
    sampler_generator = torch.Generator(device="cpu")
    sampler_generator.manual_seed(args.seed + 101)
    loader_generator = torch.Generator(device="cpu")
    loader_generator.manual_seed(args.seed + 202)
    if args.sampling_mode == "balanced":
        sampler: Any = WeightedRandomSampler(
            sampler_weights.astype(np.float64).tolist(),
            num_samples=len(train_dataset),
            replacement=True,
            generator=sampler_generator,
        )
    else:
        sampler = RandomSampler(
            train_dataset, replacement=False, generator=sampler_generator
        )
    worker_options: Dict[str, Any] = {
        "num_workers": args.num_data_workers,
        "pin_memory": pin_memory,
        "worker_init_fn": data_worker_init,
        "persistent_workers": bool(args.persistent_workers),
    }
    if args.num_data_workers > 0:
        worker_options["prefetch_factor"] = args.prefetch_factor
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=sampler,
        generator=loader_generator,
        **worker_options,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        generator=loader_generator,
        **worker_options,
    )
    # A deterministic, unweighted pass for EMA training diagnostics. Keeping
    # this loader in the main process avoids creating a third worker pool.
    train_evaluation_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=pin_memory,
    )
    return (
        train_loader,
        validation_loader,
        train_evaluation_loader,
        sampler_generator,
        loader_generator,
    )


def temporal_weights(
    length: int,
    prefix_length: int,
    prefix_weight: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    weights = torch.ones(length, device=device, dtype=dtype)
    weights[: max(0, min(prefix_length, length))] = prefix_weight
    return weights / weights.mean()


def weighted_temporal_joint_mse(
    error: torch.Tensor, scale: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    normalized_square = torch.square(error / scale.reshape(1, 1, TARGET_DIM))
    per_time = torch.mean(normalized_square, dim=2)
    return torch.sum(per_time * weights.reshape(1, -1), dim=1) / torch.sum(weights)


def loss_weights(args: argparse.Namespace) -> Dict[str, float]:
    return {
        "epsilon_loss": args.lambda_epsilon,
        "x0_loss": args.lambda_x0,
        "velocity_loss": args.lambda_velocity,
        "acceleration_loss": args.lambda_acceleration,
        "jerk_loss": args.lambda_jerk,
        "boundary_loss": args.lambda_boundary,
    }


def compute_loss_components(
    *,
    target_norm: torch.Tensor,
    noisy: torch.Tensor,
    true_noise: torch.Tensor,
    predicted_noise: torch.Tensor,
    timesteps: torch.Tensor,
    schedule: v6.DiffusionSchedule,
    residual_mean: torch.Tensor,
    residual_std: torch.Tensor,
    auxiliary_scales: Mapping[str, torch.Tensor],
    args: argparse.Namespace,
) -> Dict[str, torch.Tensor]:
    predicted_x0_norm = v6.reconstruct_x0(
        noisy, predicted_noise, timesteps, schedule
    )
    predicted_physical = predicted_x0_norm * residual_std + residual_mean
    target_physical = target_norm * residual_std + residual_mean
    position_error = predicted_physical - target_physical
    velocity_error = torch.diff(predicted_physical, dim=1) - torch.diff(
        target_physical, dim=1
    )
    acceleration_error = torch.diff(predicted_physical, n=2, dim=1) - torch.diff(
        target_physical, n=2, dim=1
    )
    jerk_error = torch.diff(predicted_physical, n=3, dim=1) - torch.diff(
        target_physical, n=3, dim=1
    )
    position_weights = temporal_weights(
        HORIZON,
        args.execution_horizon,
        args.execution_prefix_weight,
        target_norm.device,
        target_norm.dtype,
    )
    velocity_weights = temporal_weights(
        HORIZON - 1,
        args.execution_horizon - 1,
        args.execution_prefix_weight,
        target_norm.device,
        target_norm.dtype,
    )
    acceleration_weights = temporal_weights(
        HORIZON - 2,
        args.execution_horizon - 2,
        args.execution_prefix_weight,
        target_norm.device,
        target_norm.dtype,
    )
    jerk_weights = temporal_weights(
        HORIZON - 3,
        args.execution_horizon - 3,
        args.execution_prefix_weight,
        target_norm.device,
        target_norm.dtype,
    )
    epsilon_per_sample = torch.mean(
        torch.square(predicted_noise - true_noise), dim=(1, 2)
    )
    x0_per_sample = weighted_temporal_joint_mse(
        position_error, auxiliary_scales["position_scale"], position_weights
    )
    velocity_per_sample = weighted_temporal_joint_mse(
        velocity_error, auxiliary_scales["velocity_scale"], velocity_weights
    )
    acceleration_per_sample = weighted_temporal_joint_mse(
        acceleration_error,
        auxiliary_scales["acceleration_scale"],
        acceleration_weights,
    )
    jerk_per_sample = weighted_temporal_joint_mse(
        jerk_error, auxiliary_scales["jerk_scale"], jerk_weights
    )
    boundary_terms = torch.stack(
        (
            torch.mean(
                torch.square(
                    position_error[:, 0] / auxiliary_scales["position_scale"]
                ),
                dim=1,
            ),
            torch.mean(
                torch.square(
                    position_error[:, -1] / auxiliary_scales["position_scale"]
                ),
                dim=1,
            ),
            torch.mean(
                torch.square(
                    velocity_error[:, 0] / auxiliary_scales["velocity_scale"]
                ),
                dim=1,
            ),
            torch.mean(
                torch.square(
                    velocity_error[:, -1] / auxiliary_scales["velocity_scale"]
                ),
                dim=1,
            ),
        ),
        dim=1,
    )
    boundary_weights = torch.as_tensor(
        [args.execution_prefix_weight, 1.0, args.execution_prefix_weight, 1.0],
        device=target_norm.device,
        dtype=target_norm.dtype,
    )
    boundary_weights = boundary_weights / boundary_weights.mean()
    boundary_per_sample = torch.mean(
        boundary_terms * boundary_weights.reshape(1, 4), dim=1
    )
    components = {
        "epsilon_loss": epsilon_per_sample,
        "x0_loss": x0_per_sample,
        "velocity_loss": velocity_per_sample,
        "acceleration_loss": acceleration_per_sample,
        "jerk_loss": jerk_per_sample,
        "boundary_loss": boundary_per_sample,
    }
    total = torch.zeros_like(epsilon_per_sample)
    for name, weight in loss_weights(args).items():
        total = total + weight * components[name]
    physical_mse = torch.mean(torch.square(position_error), dim=(1, 2))
    prefix_mse = torch.mean(
        torch.square(position_error[:, : args.execution_horizon]), dim=(1, 2)
    )
    return {
        "total_loss": total,
        **components,
        "physical_residual_mse_rad2": physical_mse,
        "execution_prefix_residual_mse_rad2": prefix_mse,
        "full_window_residual_mse_rad2": physical_mse,
    }


def amp_context(enabled: bool) -> Any:
    if enabled:
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def make_grad_scaler(enabled: bool, initial_scale: float) -> Any:
    scaler_class = getattr(torch.amp, "GradScaler", None)
    if scaler_class is not None:
        return scaler_class(
            "cuda", enabled=enabled, init_scale=initial_scale
        )
    return torch.cuda.amp.GradScaler(
        enabled=enabled, init_scale=initial_scale
    )


def verify_component_gradients(
    components: Mapping[str, torch.Tensor],
    model: nn.Module,
    args: argparse.Namespace,
) -> Dict[str, float]:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    checks: Dict[str, float] = {}
    for name, weight in loss_weights(args).items():
        if weight <= 0.0:
            continue
        gradients = torch.autograd.grad(
            components[name].mean(),
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
        magnitude = sum(
            float(gradient.detach().abs().sum().cpu())
            for gradient in gradients
            if gradient is not None
        )
        if not np.isfinite(magnitude) or magnitude <= 0.0:
            raise RuntimeError(f"Enabled loss component {name} has no finite gradient")
        checks[name] = magnitude
    return checks


def train_epoch(
    *,
    model: nn.Module,
    loader: DataLoader[Any],
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    ema: v6.ExponentialMovingAverage,
    schedule: v6.DiffusionSchedule,
    device: torch.device,
    residual_mean: torch.Tensor,
    residual_std: torch.Tensor,
    auxiliary_scales: Mapping[str, torch.Tensor],
    args: argparse.Namespace,
    amp_enabled: bool,
    verify_gradients: bool,
) -> Tuple[
    Dict[str, float],
    int,
    int,
    float,
    Dict[str, int],
    Dict[str, float],
    int,
]:
    model.train()
    accumulator = MetricAccumulator()
    batch_count = 0
    optimizer_step_count = 0
    amp_overflow_count = 0
    gradient_norm_sum = 0.0
    sampled_scale_counts: Dict[str, int] = {}
    gradient_checks: Dict[str, float] = {}
    for condition, target, _, target_scale in loader:
        if args.max_train_batches and batch_count >= args.max_train_batches:
            break
        condition = condition.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        timesteps = torch.randint(
            0, args.num_diffusion_steps, (target.shape[0],), device=device
        )
        noise = torch.randn_like(target)
        noisy = v6.add_noise(target, noise, timesteps, schedule)
        optimizer.zero_grad(set_to_none=True)
        with amp_context(amp_enabled):
            predicted_noise = v6.predict_noise(model, noisy, timesteps, condition)
            components = compute_loss_components(
                target_norm=target,
                noisy=noisy,
                true_noise=noise,
                predicted_noise=predicted_noise,
                timesteps=timesteps,
                schedule=schedule,
                residual_mean=residual_mean,
                residual_std=residual_std,
                auxiliary_scales=auxiliary_scales,
                args=args,
            )
            loss = components["total_loss"].mean()
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("Training produced a nonfinite total loss")
        if verify_gradients and not gradient_checks:
            print(
                "smoke tensor shapes: "
                f"condition={tuple(condition.shape)}, target={tuple(target.shape)}, "
                f"timesteps={tuple(timesteps.shape)}, noise={tuple(noise.shape)}, "
                f"noisy={tuple(noisy.shape)}, predicted_noise={tuple(predicted_noise.shape)}"
            )
            gradient_checks = verify_component_gradients(components, model, args)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.grad is not None
        ]
        gradients_finite = all(
            bool(torch.all(torch.isfinite(gradient))) for gradient in gradients
        )
        if not gradients_finite:
            if not amp_enabled:
                raise RuntimeError("Training produced nonfinite unscaled gradients")
            scale_before = float(scaler.get_scale())
            scaler.step(optimizer)
            scaler.update()
            scale_after = float(scaler.get_scale())
            optimizer.zero_grad(set_to_none=True)
            if scale_after >= scale_before:
                raise RuntimeError(
                    "AMP found nonfinite gradients but did not reduce its loss scale"
                )
            amp_overflow_count += 1
            gradient_norm_value = math.nan
        elif args.gradient_clip_norm > 0.0:
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), args.gradient_clip_norm
            )
            gradient_norm_value = float(gradient_norm.detach().cpu())
        else:
            squared = sum(
                float(torch.sum(torch.square(parameter.grad.detach())).cpu())
                for parameter in model.parameters()
                if parameter.grad is not None
            )
            gradient_norm_value = math.sqrt(max(squared, 0.0))
        if gradients_finite:
            if not np.isfinite(gradient_norm_value):
                raise RuntimeError("Training produced a nonfinite gradient norm")
            scaler.step(optimizer)
            scaler.update()
            ema.update(model)
            optimizer_step_count += 1
            gradient_norm_sum += gradient_norm_value
        mask = torch.ones(target.shape[0], dtype=torch.bool, device=device)
        accumulator.add(components, mask)
        for scale in target_scale.numpy().tolist():
            key = format(float(scale), ".6g")
            sampled_scale_counts[key] = sampled_scale_counts.get(key, 0) + 1
        batch_count += 1
    if batch_count == 0:
        raise RuntimeError("No training batches were processed")
    if optimizer_step_count == 0:
        raise RuntimeError(
            "Every optimizer step overflowed under AMP; rerun with a smaller "
            "--amp_initial_scale or use --no-amp"
        )
    return (
        accumulator.finalize(),
        batch_count,
        optimizer_step_count,
        gradient_norm_sum / optimizer_step_count,
        sampled_scale_counts,
        gradient_checks,
        amp_overflow_count,
    )


@torch.no_grad()
def evaluate_model(
    *,
    model: nn.Module,
    loader: DataLoader[Any],
    schedule: v6.DiffusionSchedule,
    validation_timesteps: torch.Tensor,
    validation_noise: torch.Tensor,
    device: torch.device,
    residual_mean: torch.Tensor,
    residual_std: torch.Tensor,
    auxiliary_scales: Mapping[str, torch.Tensor],
    args: argparse.Namespace,
    amp_enabled: bool,
    max_batches: int,
) -> Dict[str, Any]:
    model.eval()
    overall = MetricAccumulator()
    by_scale: Dict[str, MetricAccumulator] = {
        format(scale, ".6g"): MetricAccumulator() for scale in EXPECTED_SCALES
    }
    by_timestep = {
        "early": MetricAccumulator(),
        "middle": MetricAccumulator(),
        "late": MetricAccumulator(),
    }
    first_boundary = args.num_diffusion_steps // 3
    second_boundary = 2 * args.num_diffusion_steps // 3
    batches = 0
    for condition, target, indices, target_scale in loader:
        if max_batches and batches >= max_batches:
            break
        condition = condition.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        index_tensor = torch.as_tensor(indices, dtype=torch.long)
        timesteps = validation_timesteps[index_tensor].to(device, non_blocking=True)
        noise = validation_noise[index_tensor].to(device, non_blocking=True)
        noisy = v6.add_noise(target, noise, timesteps, schedule)
        with amp_context(amp_enabled):
            predicted_noise = v6.predict_noise(model, noisy, timesteps, condition)
            components = compute_loss_components(
                target_norm=target,
                noisy=noisy,
                true_noise=noise,
                predicted_noise=predicted_noise,
                timesteps=timesteps,
                schedule=schedule,
                residual_mean=residual_mean,
                residual_std=residual_std,
                auxiliary_scales=auxiliary_scales,
                args=args,
            )
        if any(not bool(torch.all(torch.isfinite(value))) for value in components.values()):
            raise RuntimeError("Validation produced nonfinite metrics")
        all_mask = torch.ones(target.shape[0], dtype=torch.bool, device=device)
        overall.add(components, all_mask)
        scale_device = target_scale.to(device)
        for scale_key, accumulator in by_scale.items():
            accumulator.add(
                components,
                torch.isclose(
                    scale_device,
                    torch.tensor(float(scale_key), device=device),
                    rtol=1.0e-5,
                    atol=1.0e-7,
                ),
            )
        by_timestep["early"].add(components, timesteps < first_boundary)
        by_timestep["middle"].add(
            components,
            (timesteps >= first_boundary) & (timesteps < second_boundary),
        )
        by_timestep["late"].add(components, timesteps >= second_boundary)
        batches += 1
    if overall.count == 0:
        raise RuntimeError("No validation batches were processed")
    return {
        "overall": overall.finalize(),
        "by_scale": {key: value.finalize() for key, value in by_scale.items()},
        "by_timestep": {
            key: value.finalize() for key, value in by_timestep.items()
        },
        "row_count": overall.count,
        "batch_count": batches,
        "timestep_bins": {
            "early": [0, first_boundary - 1],
            "middle": [first_boundary, second_boundary - 1],
            "late": [second_boundary, args.num_diffusion_steps - 1],
        },
    }


def evaluate_with_ema(
    model: nn.Module,
    ema: v6.ExponentialMovingAverage,
    evaluation_kwargs: Mapping[str, Any],
) -> Dict[str, Any]:
    raw_state = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    try:
        model.load_state_dict(ema.shadow, strict=True)
        return evaluate_model(model=model, **evaluation_kwargs)
    finally:
        model.load_state_dict(raw_state, strict=True)


def model_environment(device: torch.device) -> Dict[str, Any]:
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
        result["gpu_name"] = torch.cuda.get_device_name(device)
        result["gpu_capability"] = torch.cuda.get_device_capability(device)
    return result


def capture_random_states(
    sampler_generator: torch.Generator, loader_generator: torch.Generator
) -> Dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "sampler_generator": sampler_generator.get_state(),
        "loader_generator": loader_generator.get_state(),
    }


def restore_random_states(
    states: Mapping[str, Any],
    sampler_generator: torch.Generator,
    loader_generator: torch.Generator,
) -> None:
    random.setstate(states["python"])
    np.random.set_state(states["numpy"])
    torch.set_rng_state(states["torch_cpu"].cpu())
    if torch.cuda.is_available() and states.get("torch_cuda") is not None:
        torch.cuda.set_rng_state_all(
            [state.cpu() for state in states["torch_cuda"]]
        )
    sampler_generator.set_state(states["sampler_generator"].cpu())
    loader_generator.set_state(states["loader_generator"].cpu())


def sampler_configuration(
    args: argparse.Namespace, diagnostics: Mapping[str, Any]
) -> Dict[str, Any]:
    return {
        "mode": args.sampling_mode,
        "window_balance_power": args.window_balance_power,
        "scale_balance_power": args.scale_balance_power,
        "quality_weight_power": args.quality_weight_power,
        "clip_min": args.sampler_weight_clip_min,
        "clip_max": args.sampler_weight_clip_max,
        "replacement": args.sampling_mode == "balanced",
        "num_samples": diagnostics["row_count"],
        "weights_are_not_applied_to_loss": True,
    }


def checkpoint_payload(
    *,
    model: nn.Module,
    ema: v6.ExponentialMovingAverage,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    epoch: int,
    global_step: int,
    best_metrics: Mapping[str, Mapping[str, Any]],
    early_stopping_counter: int,
    model_config: Mapping[str, Any],
    args: argparse.Namespace,
    normalization: Mapping[str, np.ndarray],
    auxiliary_normalization: Mapping[str, np.ndarray],
    sampler_config: Mapping[str, Any],
    metadata: Mapping[str, Any],
    integrity_report: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
    validation_scale_history: Sequence[Mapping[str, Any]],
    validation_timestep_history: Sequence[Mapping[str, Any]],
    environment: Mapping[str, Any],
    sampler_generator: torch.Generator,
    loader_generator: torch.Generator,
) -> Dict[str, Any]:
    raw_state = copy.deepcopy(model.state_dict())
    ema_state = copy.deepcopy(ema.state_dict())
    return {
        "model_state_dict": raw_state,
        "raw_model_state_dict": raw_state,
        "ema_model_state_dict": ema_state["shadow"],
        "ema_state_dict": ema_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": None,
        "amp_scaler_state_dict": scaler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_metrics": copy.deepcopy(dict(best_metrics)),
        "early_stopping_state": {
            "metric": args.early_stopping_metric,
            "counter": early_stopping_counter,
            "patience": args.early_stopping_patience,
        },
        "model_configuration": dict(model_config),
        "model_hyperparameters": dict(model_config),
        "condition_dim": CONDITION_DIM,
        "condition_feature_names": list(
            decode_strings(normalization["condition_feature_names"])
        ),
        "condition_feature_ordering": list(
            decode_strings(normalization["condition_feature_names"])
        ),
        "target_shape": [HORIZON, TARGET_DIM],
        "horizon": HORIZON,
        "target_dim": TARGET_DIM,
        "prediction_target_type": PREDICTION_TARGET,
        "diffusion_schedule": {
            "steps": args.num_diffusion_steps,
            "beta_schedule": "linear",
            "beta_start": 1.0e-4,
            "beta_end": 2.0e-2,
        },
        "diffusion_hyperparameters": {
            "steps": args.num_diffusion_steps,
            "beta_schedule": "linear",
            "beta_start": 1.0e-4,
            "beta_end": 2.0e-2,
        },
        "normalization_statistics": {
            key: value for key, value in normalization.items()
        },
        "normalization_source_path": str(
            (args.dataset_dir / "normalization_stats.npz").resolve()
        ),
        "normalization_sha256": integrity_report["normalization_stats_sha256"],
        "dataset_metadata_sha256": integrity_report["dataset_metadata_sha256"],
        "auxiliary_loss_normalization": {
            key: value for key, value in auxiliary_normalization.items()
        },
        "loss_weights": loss_weights(args),
        "execution_horizon": args.execution_horizon,
        "execution_prefix_weight": args.execution_prefix_weight,
        "sampler_configuration": dict(sampler_config),
        "dataset_metadata": dict(metadata),
        "dataset_integrity_report": dict(integrity_report),
        "random_generator_states": capture_random_states(
            sampler_generator, loader_generator
        ),
        "training_arguments": vars(args),
        "training_history": list(history),
        "validation_scale_history": list(validation_scale_history),
        "validation_timestep_history": list(validation_timestep_history),
        "optimizer_configuration": {
            "class": "torch.optim.AdamW",
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
        },
        "scheduler_configuration": None,
        "software_device_information": dict(environment),
    }


def load_initial_checkpoint(
    path: Path,
    model: nn.Module,
    model_config: Mapping[str, Any],
    feature_names: Sequence[str],
    normalization_hash: str,
    device: torch.device,
) -> None:
    checkpoint = load_torch_checkpoint(path, device)
    if int(checkpoint.get("condition_dim", -1)) != CONDITION_DIM:
        raise ValueError("Initialization checkpoint condition_dim is not 39")
    if list(checkpoint.get("target_shape", ())) != [HORIZON, TARGET_DIM]:
        raise ValueError("Initialization checkpoint target shape is incompatible")
    if checkpoint.get("model_configuration", checkpoint.get("model_hyperparameters")) != dict(model_config):
        raise ValueError("Initialization checkpoint model configuration differs")
    checkpoint_features = checkpoint.get(
        "condition_feature_names", checkpoint.get("condition_feature_ordering", ())
    )
    if list(checkpoint_features) != list(feature_names):
        raise ValueError("Initialization checkpoint feature ordering differs")
    if checkpoint.get("normalization_sha256") != normalization_hash:
        raise ValueError("Initialization checkpoint normalization differs")
    state = checkpoint.get("model_state_dict", checkpoint.get("raw_model_state_dict"))
    if not isinstance(state, Mapping):
        raise KeyError("Initialization checkpoint has no model state")
    model.load_state_dict(state, strict=True)


def load_resume_checkpoint(
    *,
    path: Path,
    model: nn.Module,
    ema: v6.ExponentialMovingAverage,
    optimizer: torch.optim.Optimizer,
    scaler: Any,
    model_config: Mapping[str, Any],
    feature_names: Sequence[str],
    normalization_hash: str,
    dataset_metadata_hash: str,
    auxiliary_normalization: Mapping[str, np.ndarray],
    sampler_config: Mapping[str, Any],
    args: argparse.Namespace,
    sampler_generator: torch.Generator,
    loader_generator: torch.Generator,
    device: torch.device,
) -> Dict[str, Any]:
    checkpoint = load_torch_checkpoint(path, device)
    for key, expected in (
        ("condition_dim", CONDITION_DIM),
        ("horizon", HORIZON),
        ("target_dim", TARGET_DIM),
    ):
        if int(checkpoint.get(key, -1)) != expected:
            raise ValueError(f"Resume checkpoint {key} is incompatible")
    if list(checkpoint.get("target_shape", ())) != [HORIZON, TARGET_DIM]:
        raise ValueError("Resume checkpoint target_shape is incompatible")
    if checkpoint.get("model_configuration") != dict(model_config):
        raise ValueError("Resume checkpoint model configuration differs")
    if list(checkpoint.get("condition_feature_names", ())) != list(feature_names):
        raise ValueError("Resume checkpoint condition feature ordering differs")
    if checkpoint.get("normalization_sha256") != normalization_hash:
        raise ValueError("Resume checkpoint normalization differs")
    if checkpoint.get("dataset_metadata_sha256") != dataset_metadata_hash:
        raise ValueError("Resume checkpoint dataset metadata differs")
    if checkpoint.get("sampler_configuration") != dict(sampler_config):
        raise ValueError("Resume checkpoint sampler configuration differs")
    if checkpoint.get("loss_weights") != loss_weights(args):
        raise ValueError("Resume checkpoint loss weights differ")
    checkpoint_auxiliary = checkpoint.get("auxiliary_loss_normalization", {})
    for key in AUXILIARY_SCALE_NAMES:
        if key not in checkpoint_auxiliary or not np.array_equal(
            np.asarray(checkpoint_auxiliary[key], dtype=np.float64),
            np.asarray(auxiliary_normalization[key], dtype=np.float64),
        ):
            raise ValueError(f"Resume checkpoint auxiliary scale {key} differs")
    if int(checkpoint.get("execution_horizon", -1)) != args.execution_horizon or not np.isclose(
        float(checkpoint.get("execution_prefix_weight", math.nan)),
        args.execution_prefix_weight,
    ):
        raise ValueError("Resume checkpoint execution-prefix configuration differs")
    schedule = checkpoint.get("diffusion_schedule", {})
    if int(schedule.get("steps", -1)) != args.num_diffusion_steps:
        raise ValueError("Resume checkpoint diffusion schedule differs")
    optimizer_configuration = checkpoint.get("optimizer_configuration", {})
    expected_optimizer = {
        "class": "torch.optim.AdamW",
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
    }
    if optimizer_configuration != expected_optimizer:
        raise ValueError("Resume checkpoint optimizer configuration differs")
    checkpoint_arguments = checkpoint.get("training_arguments", {})
    if bool(checkpoint_arguments.get("amp", False)) != bool(args.amp):
        raise ValueError("Resume checkpoint AMP setting differs")
    for argument_name in (
        "seed",
        "batch_size",
        "gradient_clip_norm",
        "amp_initial_scale",
    ):
        if checkpoint_arguments.get(argument_name) != getattr(args, argument_name):
            raise ValueError(
                f"Resume checkpoint {argument_name} setting differs"
            )
    if bool(checkpoint_arguments.get("deterministic_algorithms", False)) != bool(
        args.deterministic_algorithms
    ):
        raise ValueError("Resume checkpoint deterministic-algorithm setting differs")
    checkpoint_ema = checkpoint.get("ema_state_dict", {})
    if not np.isclose(float(checkpoint_ema.get("decay", math.nan)), args.ema_decay):
        raise ValueError("Resume checkpoint EMA decay differs")
    early_stopping_state = checkpoint.get("early_stopping_state", {})
    if (
        early_stopping_state.get("metric") != args.early_stopping_metric
        or int(early_stopping_state.get("patience", -1))
        != args.early_stopping_patience
    ):
        raise ValueError("Resume checkpoint early-stopping configuration differs")
    model.load_state_dict(checkpoint["raw_model_state_dict"], strict=True)
    ema.load_state_dict(checkpoint["ema_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scaler.load_state_dict(checkpoint.get("amp_scaler_state_dict", {}))
    if checkpoint.get("scheduler_state_dict") is not None:
        raise ValueError("V8 uses no scheduler but resume checkpoint contains one")
    random_states = checkpoint.get("random_generator_states")
    if not isinstance(random_states, Mapping):
        raise ValueError("Resume checkpoint has no random-generator state")
    restore_random_states(random_states, sampler_generator, loader_generator)
    return {
        "start_epoch": int(checkpoint["epoch"]) + 1,
        "global_step": int(checkpoint.get("global_step", 0)),
        "best_metrics": copy.deepcopy(checkpoint.get("best_metrics", {})),
        "early_stopping_counter": int(
            checkpoint.get("early_stopping_state", {}).get("counter", 0)
        ),
        "history": [dict(row) for row in checkpoint.get("training_history", ())],
        "validation_scale_history": [
            dict(row) for row in checkpoint.get("validation_scale_history", ())
        ],
        "validation_timestep_history": [
            dict(row) for row in checkpoint.get("validation_timestep_history", ())
        ],
    }


def save_checkpoint_with_category(
    payload: Mapping[str, Any], path: Path, category: str
) -> None:
    categorized = dict(payload)
    categorized["checkpoint_category"] = category
    if category.startswith("ema_"):
        categorized["model_state_dict"] = categorized["ema_model_state_dict"]
        categorized["selected_model_state"] = "ema"
    else:
        categorized["model_state_dict"] = categorized["raw_model_state_dict"]
        categorized["selected_model_state"] = "raw"
    v6.atomic_torch_save(categorized, path)


def add_prefixed_metrics(
    row: Dict[str, Any], prefix: str, metrics: Mapping[str, float]
) -> None:
    for name, value in metrics.items():
        row[f"{prefix}_{name}"] = value


def append_group_records(
    records: List[Dict[str, Any]],
    *,
    epoch: int,
    model_state: str,
    grouping_name: str,
    groups: Mapping[str, Mapping[str, float]],
) -> None:
    for group_value, metrics in groups.items():
        records.append(
            {
                "epoch": epoch,
                "model_state": model_state,
                grouping_name: group_value,
                **dict(metrics),
            }
        )


def atomic_plot(fig: Any, destination: Path) -> None:
    temporary = destination.with_name(f".{destination.stem}.tmp{destination.suffix}")
    try:
        fig.savefig(temporary, dpi=160)
        os.replace(temporary, destination)
    finally:
        plt.close(fig)
        if temporary.exists():
            temporary.unlink()


def save_diagnostics(
    history: Sequence[Mapping[str, Any]],
    scale_records: Sequence[Mapping[str, Any]],
    timestep_records: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> None:
    history_frame = pd.DataFrame(history)
    v6.atomic_csv(history_frame, output_dir / "training_history.csv")
    v6.atomic_json({"epochs": list(history)}, output_dir / "training_history.json")
    v6.atomic_csv(pd.DataFrame(scale_records), output_dir / "validation_loss_by_scale.csv")
    v6.atomic_csv(
        pd.DataFrame(timestep_records),
        output_dir / "validation_loss_by_timestep_bin.csv",
    )
    figure, axis = plt.subplots(figsize=(10, 5.5))
    for column, label in (
        ("train_raw_total_loss", "train raw"),
        ("train_ema_total_loss", "train EMA"),
        ("validation_raw_total_loss", "validation raw"),
        ("validation_ema_total_loss", "validation EMA"),
    ):
        if column in history_frame:
            axis.plot(history_frame["epoch"], history_frame[column], label=label)
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Total loss")
    axis.set_title("V8 training and deterministic validation")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    atomic_plot(figure, output_dir / "training_and_validation_loss.png")
    figure, axes = plt.subplots(2, 3, figsize=(14, 8), sharex=True)
    for axis, component in zip(
        axes.reshape(-1),
        ("epsilon", "x0", "velocity", "acceleration", "jerk", "boundary"),
    ):
        for state, label in (("raw", "raw"), ("ema", "EMA")):
            column = f"validation_{state}_{component}_loss"
            axis.plot(history_frame["epoch"], history_frame[column], label=label)
        axis.set_title(component)
        axis.grid(True, alpha=0.25)
    axes[0, 0].legend()
    for axis in axes[-1]:
        axis.set_xlabel("Epoch")
    figure.suptitle("V8 validation loss components")
    figure.tight_layout()
    atomic_plot(figure, output_dir / "loss_component_history.png")


def prepare_output_directory(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    existing = [args.output_dir / name for name in OUTPUT_FILES if (args.output_dir / name).exists()]
    if existing and args.resume_checkpoint is None and not args.overwrite:
        raise FileExistsError(
            f"Training outputs already exist: {existing}; pass --overwrite or resume"
        )
    if args.overwrite:
        for path in existing:
            path.unlink()
        checkpoint_dir = args.output_dir / "checkpoints"
        if checkpoint_dir.exists():
            shutil.rmtree(checkpoint_dir)


def set_reproducibility(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic, warn_only=True)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = deterministic
        torch.backends.cudnn.benchmark = not deterministic


def main() -> int:
    wall_started = time.perf_counter()
    args = parse_args()
    validate_cli(args)
    dataset_paths = {
        "train": args.dataset_dir / "train_windows.npz",
        "validation": args.dataset_dir / "validation_windows.npz",
        "normalization": args.dataset_dir / "normalization_stats.npz",
        "metadata": args.dataset_dir / "dataset_metadata.json",
        "train_rows": args.dataset_dir / "train_rows.csv",
        "validation_rows": args.dataset_dir / "validation_rows.csv",
    }
    for path in dataset_paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    prepare_output_directory(args)
    set_reproducibility(args.seed, args.deterministic_algorithms)
    device = v6.resolve_device(args.device)
    args.persistent_workers = (
        args.num_data_workers > 0
        if args.persistent_workers is None
        else bool(args.persistent_workers and args.num_data_workers > 0)
    )
    args.pin_memory = (
        device.type == "cuda" if args.pin_memory is None else bool(args.pin_memory)
    )
    args.amp = device.type == "cuda" if args.amp is None else bool(args.amp)
    if args.amp and device.type != "cuda":
        raise ValueError("AMP is supported only with CUDA in this trainer")
    train = load_windows(dataset_paths["train"], "train")
    validation = load_windows(dataset_paths["validation"], "validation")
    validate_row_csv(dataset_paths["train_rows"], train, "train")
    validate_row_csv(dataset_paths["validation_rows"], validation, "validation")
    normalization = load_normalization(dataset_paths["normalization"])
    metadata = load_json(dataset_paths["metadata"])
    integrity_report = validate_dataset(
        train,
        validation,
        normalization,
        metadata,
        dataset_paths["normalization"],
        dataset_paths["metadata"],
    )
    v6.atomic_json(integrity_report, args.output_dir / "dataset_integrity_report.json")
    auxiliary_normalization = build_auxiliary_normalization(train.residual_physical)
    atomic_npz(
        args.output_dir / "auxiliary_loss_normalization.npz",
        auxiliary_normalization,
    )
    sampler_weights, sampling_frame, sampling_info = build_sampler_weights(train, args)
    v6.atomic_csv(sampling_frame, args.output_dir / "sampling_diagnostics.csv")
    (
        train_loader,
        validation_loader,
        train_evaluation_loader,
        sampler_generator,
        loader_generator,
    ) = make_loaders(train, validation, sampler_weights, args, args.pin_memory)
    model, model_config = v6.instantiate_v5_model(
        HORIZON, CONDITION_DIM, TARGET_DIM, args.num_diffusion_steps
    )
    model = model.to(device)
    feature_names = decode_strings(normalization["condition_feature_names"])
    if args.init_checkpoint is not None:
        load_initial_checkpoint(
            args.init_checkpoint,
            model,
            model_config,
            feature_names,
            integrity_report["normalization_stats_sha256"],
            device,
        )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    ema = v6.ExponentialMovingAverage(model, args.ema_decay)
    scaler = make_grad_scaler(args.amp, args.amp_initial_scale)
    schedule = v6.build_schedule(args.num_diffusion_steps, device)
    residual_mean = torch.as_tensor(
        normalization["residual_mean"], dtype=torch.float32, device=device
    ).reshape(1, 1, TARGET_DIM)
    residual_std = torch.as_tensor(
        normalization["residual_std"], dtype=torch.float32, device=device
    ).reshape(1, 1, TARGET_DIM)
    auxiliary_scales = {
        key: torch.as_tensor(
            auxiliary_normalization[key], dtype=torch.float32, device=device
        )
        for key in AUXILIARY_SCALE_NAMES
    }
    validation_timesteps, validation_noise = v6.make_validation_bank(
        len(validation.path_names),
        HORIZON,
        TARGET_DIM,
        args.num_diffusion_steps,
        args.seed,
    )
    train_validation_timesteps, train_validation_noise = v6.make_validation_bank(
        len(train.path_names),
        HORIZON,
        TARGET_DIM,
        args.num_diffusion_steps,
        args.seed + 2_000_003,
    )
    environment = model_environment(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    sampler_config = sampler_configuration(args, sampling_info)
    smoke_mode = bool(args.max_train_batches or args.max_validation_batches)
    training_metadata: Dict[str, Any] = {
        "classification": "V8_TRAINING_IN_PROGRESS",
        "arguments": vars(args),
        "dataset_integrity": integrity_report,
        "dataset_metadata": metadata,
        "model_configuration": model_config,
        "model_parameter_count": parameter_count,
        "condition_dim": CONDITION_DIM,
        "condition_feature_names": list(feature_names),
        "target_shape": [HORIZON, TARGET_DIM],
        "normalization_behavior": (
            "condition_norm and residual_q_norm are consumed directly; trainer "
            "does not normalize them again"
        ),
        "sampler": {**sampler_config, **sampling_info},
        "loss_weights": loss_weights(args),
        "auxiliary_loss_normalization": {
            key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in auxiliary_normalization.items()
        },
        "seed_policy": {
            "global_seed": args.seed,
            "python_numpy_torch_cpu_cuda_seeded": True,
            "sampler_generator_seed": args.seed + 101,
            "dataloader_generator_seed": args.seed + 202,
            "validation_noise_bank_seed": args.seed + 1_000_003,
            "train_diagnostic_noise_bank_seed": args.seed + 3_000_006,
            "dataloader_worker_seed": "torch.initial_seed modulo 2**32",
            "deterministic_algorithms": args.deterministic_algorithms,
            "bitwise_cuda_reproducibility_guaranteed": bool(
                args.deterministic_algorithms
            ),
        },
        "dataloader": {
            "num_data_workers": args.num_data_workers,
            "prefetch_factor": args.prefetch_factor,
            "persistent_workers": args.persistent_workers,
            "pin_memory": args.pin_memory,
            "non_blocking_transfers": True,
            "worker_torch_threads": 1,
        },
        "optimizer": "AdamW",
        "scheduler": None,
        "amp_enabled": args.amp,
        "amp_initial_scale": args.amp_initial_scale,
        "environment": environment,
        "scientific_limit": (
            "Training losses do not establish robot-aware or generative improvement; "
            "all-window FK evaluation is required"
        ),
    }
    v6.atomic_json(training_metadata, args.output_dir / "training_metadata.json")
    print(f"dataset directory: {args.dataset_dir}")
    print(f"output directory: {args.output_dir}")
    print(f"device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"condition dimension: {CONDITION_DIM}")
    print(f"condition feature names: {list(feature_names)}")
    print(f"target shape: ({HORIZON}, {TARGET_DIM})")
    print(
        f"train tensors: condition={train.condition_norm.shape}, "
        f"target={train.residual_norm.shape}"
    )
    print(
        f"validation tensors: condition={validation.condition_norm.shape}, "
        f"target={validation.residual_norm.shape}"
    )
    print(
        f"train: paths={len(set(train.path_names))}, "
        f"windows={len(set(window_identities(train)))}, rows={len(train.path_names)}"
    )
    print(
        f"validation: paths={len(set(validation.path_names))}, "
        f"windows={len(set(window_identities(validation)))}, rows={len(validation.path_names)}"
    )
    print(f"scale distribution: {sampling_info['original_scale_distribution']}")
    print(f"sampling mode: {args.sampling_mode}")
    print(f"DataLoader workers: {args.num_data_workers}")
    print(f"AMP enabled: {args.amp}")
    print(f"model parameter count: {parameter_count}")
    print(f"diffusion steps: {args.num_diffusion_steps}")
    print(f"loss weights: {loss_weights(args)}")
    print(
        f"execution-prefix weighting: horizon={args.execution_horizon}, "
        f"weight={args.execution_prefix_weight}"
    )
    print(
        "auxiliary normalization scales: "
        + json.dumps(
            {
                key: value.tolist()
                for key, value in auxiliary_normalization.items()
                if key in AUXILIARY_SCALE_NAMES
            },
            sort_keys=True,
        )
    )

    best_metrics: Dict[str, Dict[str, Any]] = {
        name: {"value": math.inf, "epoch": 0}
        for name in (
            "raw_total_loss",
            "ema_total_loss",
            "raw_epsilon_loss",
            "ema_epsilon_loss",
        )
    }
    start_epoch = 1
    global_step = 0
    early_stopping_counter = 0
    history: List[Dict[str, Any]] = []
    scale_records: List[Dict[str, Any]] = []
    timestep_records: List[Dict[str, Any]] = []
    if args.resume_checkpoint is not None:
        resumed = load_resume_checkpoint(
            path=args.resume_checkpoint,
            model=model,
            ema=ema,
            optimizer=optimizer,
            scaler=scaler,
            model_config=model_config,
            feature_names=feature_names,
            normalization_hash=integrity_report["normalization_stats_sha256"],
            dataset_metadata_hash=integrity_report["dataset_metadata_sha256"],
            auxiliary_normalization=auxiliary_normalization,
            sampler_config=sampler_config,
            args=args,
            sampler_generator=sampler_generator,
            loader_generator=loader_generator,
            device=device,
        )
        start_epoch = resumed["start_epoch"]
        global_step = resumed["global_step"]
        best_metrics = resumed["best_metrics"]
        early_stopping_counter = resumed["early_stopping_counter"]
        history = resumed["history"]
        scale_records = resumed["validation_scale_history"]
        timestep_records = resumed["validation_timestep_history"]
    if start_epoch > args.epochs:
        raise ValueError(
            f"Resume starts at epoch {start_epoch}, beyond requested epochs={args.epochs}"
        )
    checkpoint_dir = args.output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    cumulative_wall = (
        float(history[-1].get("cumulative_wall_time_s", 0.0)) if history else 0.0
    )
    gradient_checks: Dict[str, float] = {}
    last_epoch = start_epoch - 1
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_started = time.perf_counter()
        (
            train_metrics,
            train_batches,
            optimizer_steps,
            mean_gradient_norm,
            sampled_scales,
            checks,
            amp_overflow_batches,
        ) = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            ema=ema,
            schedule=schedule,
            device=device,
            residual_mean=residual_mean,
            residual_std=residual_std,
            auxiliary_scales=auxiliary_scales,
            args=args,
            amp_enabled=args.amp,
            verify_gradients=smoke_mode and not gradient_checks,
        )
        gradient_checks.update(checks)
        global_step += optimizer_steps
        validation_kwargs = {
            "loader": validation_loader,
            "schedule": schedule,
            "validation_timesteps": validation_timesteps,
            "validation_noise": validation_noise,
            "device": device,
            "residual_mean": residual_mean,
            "residual_std": residual_std,
            "auxiliary_scales": auxiliary_scales,
            "args": args,
            "amp_enabled": args.amp,
            "max_batches": args.max_validation_batches,
        }
        raw_validation = evaluate_model(model=model, **validation_kwargs)
        ema_validation = evaluate_with_ema(model, ema, validation_kwargs)
        train_evaluation_kwargs = {
            **validation_kwargs,
            "loader": train_evaluation_loader,
            "validation_timesteps": train_validation_timesteps,
            "validation_noise": train_validation_noise,
            "max_batches": args.max_train_batches,
        }
        raw_train = evaluate_model(model=model, **train_evaluation_kwargs)
        ema_train = evaluate_with_ema(model, ema, train_evaluation_kwargs)
        current_best_values = {
            "raw_total_loss": raw_validation["overall"]["total_loss"],
            "ema_total_loss": ema_validation["overall"]["total_loss"],
            "raw_epsilon_loss": raw_validation["overall"]["epsilon_loss"],
            "ema_epsilon_loss": ema_validation["overall"]["epsilon_loss"],
        }
        improvement_flags = {
            name: value < float(best_metrics[name]["value"])
            for name, value in current_best_values.items()
        }
        monitor_improved = improvement_flags[args.early_stopping_metric]
        for name, improved in improvement_flags.items():
            if improved:
                best_metrics[name] = {
                    "value": float(current_best_values[name]),
                    "epoch": epoch,
                }
        early_stopping_counter = 0 if monitor_improved else early_stopping_counter + 1
        epoch_wall = time.perf_counter() - epoch_started
        cumulative_wall += epoch_wall
        row: Dict[str, Any] = {
            "epoch": epoch,
            "global_step": global_step,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "gradient_norm": mean_gradient_norm,
            "training_batch_count": train_batches,
            "optimizer_step_count": optimizer_steps,
            "amp_overflow_batch_count": amp_overflow_batches,
            "amp_loss_scale": float(scaler.get_scale()),
            "epoch_wall_time_s": epoch_wall,
            "cumulative_wall_time_s": cumulative_wall,
            "early_stopping_counter": early_stopping_counter,
            "sampled_scale_counts": json.dumps(sampled_scales, sort_keys=True),
        }
        add_prefixed_metrics(row, "optimization_raw", train_metrics)
        add_prefixed_metrics(row, "train_raw", raw_train["overall"])
        add_prefixed_metrics(row, "train_ema", ema_train["overall"])
        add_prefixed_metrics(row, "validation_raw", raw_validation["overall"])
        add_prefixed_metrics(row, "validation_ema", ema_validation["overall"])
        for scale in EXPECTED_SCALES:
            key = format(scale, ".6g")
            slug = key.replace(".", "_")
            row[f"sampled_scale_{slug}_count"] = sampled_scales.get(key, 0)
        for name, state in best_metrics.items():
            row[f"best_{name}"] = state["value"]
            row[f"best_{name}_epoch"] = state["epoch"]
        history.append(row)
        append_group_records(
            scale_records,
            epoch=epoch,
            model_state="raw",
            grouping_name="target_scale",
            groups=raw_validation["by_scale"],
        )
        append_group_records(
            scale_records,
            epoch=epoch,
            model_state="ema",
            grouping_name="target_scale",
            groups=ema_validation["by_scale"],
        )
        append_group_records(
            timestep_records,
            epoch=epoch,
            model_state="raw",
            grouping_name="timestep_bin",
            groups=raw_validation["by_timestep"],
        )
        append_group_records(
            timestep_records,
            epoch=epoch,
            model_state="ema",
            grouping_name="timestep_bin",
            groups=ema_validation["by_timestep"],
        )
        save_diagnostics(history, scale_records, timestep_records, args.output_dir)
        payload = checkpoint_payload(
            model=model,
            ema=ema,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch,
            global_step=global_step,
            best_metrics=best_metrics,
            early_stopping_counter=early_stopping_counter,
            model_config=model_config,
            args=args,
            normalization=normalization,
            auxiliary_normalization=auxiliary_normalization,
            sampler_config=sampler_config,
            metadata=metadata,
            integrity_report=integrity_report,
            history=history,
            validation_scale_history=scale_records,
            validation_timestep_history=timestep_records,
            environment=environment,
            sampler_generator=sampler_generator,
            loader_generator=loader_generator,
        )
        checkpoint_paths = {
            "raw_total_loss": args.output_dir / "best_raw_total_loss_checkpoint.pt",
            "ema_total_loss": args.output_dir / "best_ema_total_loss_checkpoint.pt",
            "raw_epsilon_loss": args.output_dir / "best_raw_epsilon_loss_checkpoint.pt",
            "ema_epsilon_loss": args.output_dir / "best_ema_epsilon_loss_checkpoint.pt",
        }
        for name, improved in improvement_flags.items():
            if improved:
                save_checkpoint_with_category(payload, checkpoint_paths[name], name)
        if epoch % args.checkpoint_interval == 0:
            save_checkpoint_with_category(
                payload,
                checkpoint_dir / f"epoch_{epoch:04d}.pt",
                "periodic",
            )
        save_checkpoint_with_category(
            payload, args.output_dir / "last_checkpoint.pt", "last"
        )
        last_epoch = epoch
        print(
            f"epoch={epoch} train_total={train_metrics['total_loss']:.7f} "
            f"val_raw_total={raw_validation['overall']['total_loss']:.7f} "
            f"val_ema_total={ema_validation['overall']['total_loss']:.7f} "
            f"val_ema_eps={ema_validation['overall']['epsilon_loss']:.7f} "
            f"val_ema_x0={ema_validation['overall']['x0_loss']:.7f} "
            f"val_ema_acc={ema_validation['overall']['acceleration_loss']:.7f} "
            f"val_ema_jerk={ema_validation['overall']['jerk_loss']:.7f} "
            f"lr={optimizer.param_groups[0]['lr']:.3e} "
            f"amp_overflow={amp_overflow_batches} "
            f"amp_scale={float(scaler.get_scale()):.1f} "
            f"time={epoch_wall:.2f}s early_stop={early_stopping_counter}"
        )
        if early_stopping_counter >= args.early_stopping_patience:
            print(f"early stopping at epoch {epoch}")
            break

    required_checkpoints = (
        "best_raw_total_loss_checkpoint.pt",
        "best_ema_total_loss_checkpoint.pt",
        "best_raw_epsilon_loss_checkpoint.pt",
        "best_ema_epsilon_loss_checkpoint.pt",
        "last_checkpoint.pt",
    )
    missing_checkpoints = [
        name for name in required_checkpoints if not (args.output_dir / name).is_file()
    ]
    if missing_checkpoints:
        raise RuntimeError(f"Required checkpoints were not saved: {missing_checkpoints}")
    if smoke_mode:
        enabled_components = {
            name for name, weight in loss_weights(args).items() if weight > 0.0
        }
        if set(gradient_checks) != enabled_components:
            raise RuntimeError(
                "Smoke test did not verify every enabled component gradient: "
                f"verified={sorted(gradient_checks)}, expected={sorted(enabled_components)}"
            )
    classification = (
        "V8_TRAINING_SMOKE_TEST_COMPLETE" if smoke_mode else "V8_TRAINING_COMPLETE"
    )
    total_wall = time.perf_counter() - wall_started
    training_metadata.update(
        {
            "classification": classification,
            "best_metrics": best_metrics,
            "last_epoch": last_epoch,
            "total_wall_time_s": total_wall,
            "smoke_test_component_gradient_magnitudes": gradient_checks,
            "checkpoint_paths": {
                name: str((args.output_dir / name).resolve())
                for name in required_checkpoints
            },
        }
    )
    v6.atomic_json(training_metadata, args.output_dir / "training_metadata.json")
    print(
        f"best raw total-loss epoch: {best_metrics['raw_total_loss']['epoch']}"
    )
    print(
        f"best EMA total-loss epoch: {best_metrics['ema_total_loss']['epoch']}"
    )
    print(
        f"best raw epsilon-loss epoch: {best_metrics['raw_epsilon_loss']['epoch']}"
    )
    print(
        f"best EMA epsilon-loss epoch: {best_metrics['ema_epsilon_loss']['epoch']}"
    )
    print(f"last epoch: {last_epoch}")
    print(f"total wall time: {total_wall:.3f} s")
    for name in required_checkpoints:
        print(f"checkpoint: {args.output_dir / name}")
    print(f"classification: {classification}")
    print(classification)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
