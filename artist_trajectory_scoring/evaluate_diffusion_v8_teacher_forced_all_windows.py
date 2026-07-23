#!/usr/bin/env python3
"""All-window teacher-forced evaluation for v8 residual diffusion.

The evaluation population is reconstructed from the authoritative v6 strong-
prior windows. Retained v8 targets are used only to label post-hoc coverage
subsets; they never enter sampling, candidate scoring, or selection.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import multiprocessing
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Mapping, Optional, Sequence, Tuple, cast

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import build_diffusion_v7_cost_improving_training_dataset as v7_dataset_builder
import evaluate_diffusion_v6_teacher_forced_validation as v6_evaluator
import evaluate_diffusion_v7_teacher_forced_validation as v7_evaluator
import generate_diffusion_v7_cost_improving_residual_targets as v7_target_generator
import train_conditional_diffusion_trajectory_v6_strong_prior_residual_unet as v6_trainer
import train_conditional_diffusion_trajectory_v8 as v8_trainer
from generate_ik_seed_path import (
    DEFAULT_EE_LINK,
    DEFAULT_JOINT_LIMIT_SAFETY_MARGIN_RAD,
    DEFAULT_JOINT_NAMES,
    DEFAULT_URDF_PATH,
    HARD_JOINT_LIMIT_TOLERANCE_RAD,
)


HORIZON = 32
EXECUTION_HORIZON = 8
V7_CONDITION_DIM = 38
CONDITION_DIM = 39
TARGET_DIM = 6
MAXIMUM_JOINT_STEP_RAD = 0.20
FLOAT_RTOL = 1.0e-6
FLOAT_ATOL = 1.0e-7
NORMALIZATION_ATOL = 2.0e-5
V7_HISTORICAL_ACCEPTED_RATE = 0.361
DIFFICULT_PATHS = ("path_0306", "path_0370")

DEFAULT_DATASET_DIR = Path(
    "data/cartesian_expert_dataset_v3/"
    "diffusion_v8_multitarget_scaled_training_dataset_100paths"
)
DEFAULT_TARGET_DIR = Path(
    "data/cartesian_expert_dataset_v3/"
    "diffusion_v8_multitarget_scaled_residual_targets_100paths"
)
DEFAULT_MODEL_DIR = Path(
    "models/diffusion_v8_multitarget_scaled_residual_unet_100paths_seed42"
)
DEFAULT_OUTPUT_DIR = Path(
    "results/diffusion_v8_teacher_forced_all_windows_seed42"
)
DEFAULT_SOURCE_WINDOWS = Path(
    "data/cartesian_expert_dataset_v3/"
    "diffusion_v6_strong_prior_residual_windows/train_windows.npz"
)

OUTPUT_FILES = (
    "evaluation_summary.json",
    "per_sample_results.csv",
    "per_window_results.csv",
    "configuration_summary.csv",
    "per_path_summary.csv",
    "scale_summary.csv",
    "checkpoint_summary.csv",
    "checkpoint_state_manifest.csv",
    "target_coverage_subset_summary.csv",
    "difficult_path_summary.csv",
    "bootstrap_confidence_intervals.csv",
    "v7_v8_comparison_summary.csv",
    "timing_summary.json",
    "evaluation_metadata.json",
)
PLOT_FILES = (
    "accepted_rate_by_scale_and_k.png",
    "hard_safe_rate_by_scale.png",
    "improvement_by_scale_and_k.png",
    "accepted_windows_by_path.png",
    "fallback_rate_by_configuration.png",
    "selected_example_trajectories.png",
)
SUBSET_ORDER = (
    "primary_all",
    "primary_target_covered",
    "primary_zero_target",
    "difficult_no_target",
    "combined_diagnostic",
)
FORBIDDEN_SELECTION_FIELDS = frozenset(
    {
        "target_q",
        "target_residual",
        "residual_target",
        "target_cartesian_improvement",
        "target_delta_score",
        "nearest_target_distance",
        "oracle_target_id",
    }
)
REPORT_REJECTION_REASONS = (
    *v7_evaluator.TARGET_REJECTION_REASONS,
    "nonnegative_delta_score",
)


@dataclass(frozen=True)
class EvaluationWindow:
    canonical_index: int
    path_name: str
    path_index: int
    window_start: int
    population: str
    target_covered: bool
    condition_v7: np.ndarray
    context: v7_target_generator.WindowContext

    @property
    def prior_q(self) -> np.ndarray:
        return self.context.prior_q

    @property
    def desired(self) -> np.ndarray:
        return self.context.desired

    @property
    def prior_ee(self) -> np.ndarray:
        return self.context.prior_ee


@dataclass
class ModelVariant:
    label: str
    source_argument: str
    checkpoint_path: Path
    state_type: str
    epoch: int
    state_hash: str
    state_dict: Mapping[str, torch.Tensor]
    diffusion_steps: int
    model_configuration: Mapping[str, Any]


@dataclass(frozen=True)
class PlotRecord:
    path_name: str
    window_start: int
    desired: np.ndarray
    prior_ee: np.ndarray
    selected_ee: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate v8 Gaussian DDIM residual generation on every physical "
            "window of the path-disjoint validation paths."
        )
    )
    parser.add_argument("--dataset_dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--target_generation_dir", type=Path, default=DEFAULT_TARGET_DIR)
    parser.add_argument("--source_windows_npz", type=Path, default=DEFAULT_SOURCE_WINDOWS)
    parser.add_argument(
        "--best_raw_total_checkpoint", type=Path,
        default=DEFAULT_MODEL_DIR / "best_raw_total_loss_checkpoint.pt",
    )
    parser.add_argument(
        "--best_ema_total_checkpoint", type=Path,
        default=DEFAULT_MODEL_DIR / "best_ema_total_loss_checkpoint.pt",
    )
    parser.add_argument(
        "--best_raw_epsilon_checkpoint", type=Path,
        default=DEFAULT_MODEL_DIR / "best_raw_epsilon_loss_checkpoint.pt",
    )
    parser.add_argument(
        "--best_ema_epsilon_checkpoint", type=Path,
        default=DEFAULT_MODEL_DIR / "best_ema_epsilon_loss_checkpoint.pt",
    )
    parser.add_argument(
        "--last_checkpoint", type=Path,
        default=DEFAULT_MODEL_DIR / "last_checkpoint.pt",
    )
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--target_scales", type=float, nargs="+",
        default=[0.125, 0.25, 0.50, 0.75, 1.00],
    )
    parser.add_argument(
        "--output_alphas", type=float, nargs="+", default=[1.0],
        help="Post-generation residual multipliers; 1.0 is the scientific result.",
    )
    parser.add_argument("--ddim_steps", type=int, default=50)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--k_values", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Legacy reproducibility seed and default for --sampling_seed.",
    )
    parser.add_argument(
        "--sampling_seed",
        type=int,
        default=None,
        help=(
            "Seed used only for stochastic diffusion inference. When omitted, "
            "--seed preserves the historical behavior."
        ),
    )
    parser.add_argument(
        "--checkpoint_states",
        type=str,
        nargs="+",
        default=None,
        help="Optional exact variant labels to evaluate after state deduplication.",
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--num_cpu_workers", type=int, default=1)
    parser.add_argument("--gpu_batch_size", type=int, default=8)
    parser.add_argument("--bootstrap_samples", type=int, default=2000)
    parser.add_argument("--bootstrap_seed", type=int, default=42)
    parser.add_argument("--max_primary_paths", type=int, default=None)
    parser.add_argument("--max_primary_windows", type=int, default=None)
    parser.add_argument(
        "--include_difficult_paths", dest="include_difficult_paths",
        action="store_true",
    )
    parser.add_argument(
        "--no-include-difficult-paths", "--no_include_difficult_paths",
        dest="include_difficult_paths", action="store_false",
    )
    parser.set_defaults(include_difficult_paths=True)
    parser.add_argument(
        "--save_per_sample_results",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--plot_example_count", type=int, default=5)
    parser.add_argument("--robot_urdf", type=Path, default=Path(DEFAULT_URDF_PATH))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_cli(args: argparse.Namespace) -> None:
    if args.ddim_steps < 1:
        raise ValueError("--ddim_steps must be at least 1")
    if not np.isfinite(args.eta) or args.eta < 0.0:
        raise ValueError("--eta must be finite and non-negative")
    if not args.k_values or any(value < 1 for value in args.k_values):
        raise ValueError("--k_values must contain positive integers")
    if 1 not in args.k_values:
        raise ValueError("--k_values must include 1")
    if len(set(args.k_values)) != len(args.k_values):
        raise ValueError("--k_values cannot contain duplicates")
    for name in ("target_scales", "output_alphas"):
        values = [float(value) for value in getattr(args, name)]
        if not values or any(not np.isfinite(value) or value <= 0.0 for value in values):
            raise ValueError(f"--{name} must contain positive finite values")
        if len(set(values)) != len(values):
            raise ValueError(f"--{name} cannot contain duplicates")
    if args.checkpoint_states is not None:
        states = [str(value).strip() for value in args.checkpoint_states]
        if not states or any(not value for value in states):
            raise ValueError("--checkpoint_states cannot contain empty labels")
        if len(set(states)) != len(states):
            raise ValueError("--checkpoint_states cannot contain duplicates")
    if args.num_cpu_workers < 1:
        raise ValueError("--num_cpu_workers must be at least 1")
    if args.gpu_batch_size < 1:
        raise ValueError("--gpu_batch_size must be at least 1")
    if args.bootstrap_samples < 1:
        raise ValueError("--bootstrap_samples must be at least 1")
    for name in ("max_primary_paths", "max_primary_windows"):
        value = getattr(args, name)
        if value is not None and value < 1:
            raise ValueError(f"--{name} must be positive")
    if args.plot_example_count < 0:
        raise ValueError("--plot_example_count must be non-negative")


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(value)


def set_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_project_path(path: Path) -> Path:
    value = Path(path).expanduser()
    if value.is_absolute():
        return value
    script_dir = Path(__file__).resolve().parent
    candidates = (script_dir / value, Path.cwd() / value, script_dir.parent / value)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    # New output directories need not exist yet; project-local is authoritative.
    return (script_dir / value).resolve()


def load_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return payload


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def stable_seed(*parts: Any) -> int:
    payload = json.dumps(
        parts, ensure_ascii=False, separators=(",", ":"), sort_keys=False
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & (
        (1 << 63) - 1
    )


def finite_array(label: str, values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{label} contains NaN or infinity")
    return array


def load_target_coverage(path: Path) -> set[Tuple[str, int]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    # NumPy's overloads can misclassify pathlib.Path as the file-object form.
    with cast(Any, np.load)(path, allow_pickle=False) as archive:
        for key in ("path_name", "window_start"):
            if key not in archive.files:
                raise KeyError(f"{path} is missing {key}")
        names = v8_trainer.decode_strings(np.asarray(archive["path_name"]))
        starts = np.asarray(archive["window_start"], dtype=np.int64)
    return {(str(name), int(start)) for name, start in zip(names, starts)}


def validate_dataset_contract(
    dataset_dir: Path,
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray], Tuple[str, ...], Tuple[str, ...]]:
    metadata_path = dataset_dir / "dataset_metadata.json"
    normalization_path = dataset_dir / "normalization_stats.npz"
    metadata = load_json(metadata_path)
    normalization = v8_trainer.load_normalization(normalization_path)
    if metadata.get("classification") != "READY_FOR_V8_TRAINING":
        raise ValueError("dataset_metadata.json is not READY_FOR_V8_TRAINING")
    feature_names = tuple(
        str(value) for value in v8_trainer.decode_strings(
            normalization["condition_feature_names"]
        )
    )
    metadata_features = tuple(
        str(value) for value in metadata.get("condition", {}).get("feature_names", ())
    )
    expected_v7 = tuple(str(value) for value in v7_dataset_builder.CONDITION_FEATURE_NAMES)
    expected = (*expected_v7, "target_scale")
    if len(feature_names) != CONDITION_DIM or feature_names != expected:
        raise ValueError(
            "V8 condition feature order must be the exact v7 38-D order followed "
            "by target_scale"
        )
    if metadata_features != feature_names:
        raise ValueError("Dataset metadata and normalization feature orders differ")
    if feature_names.index("target_scale") != CONDITION_DIM - 1:
        raise ValueError("target_scale must be feature index 38")
    condition_mean = np.asarray(normalization["condition_mean"], dtype=np.float64)
    condition_std = np.asarray(normalization["condition_std"], dtype=np.float64)
    if not np.isclose(condition_mean[-1], 0.0) or not np.isclose(condition_std[-1], 1.0):
        raise ValueError("target_scale must use the identity normalization")
    return metadata, normalization, feature_names, expected_v7


def load_split_and_manifest(
    dataset_dir: Path,
) -> Tuple[Tuple[str, ...], Tuple[str, ...], Tuple[str, ...]]:
    split = pd.read_csv(dataset_dir / "path_split.csv")
    required = {"path_name", "split"}
    if not required <= set(split.columns):
        raise ValueError("path_split.csv requires path_name and split")
    split = split.copy()
    split["path_name"] = pd.Series(
        [str(value) for value in split["path_name"].tolist()],
        index=split.index,
        dtype="string",
    )
    split["split"] = pd.Series(
        [str(value).lower() for value in split["split"].tolist()],
        index=split.index,
        dtype="string",
    )
    if split["path_name"].duplicated().any():
        raise ValueError("path_split.csv contains duplicate paths")
    training: Tuple[str, ...] = tuple(sorted(
        str(value)
        for value in split.loc[split["split"] == "train", "path_name"].tolist()
    ))
    validation: Tuple[str, ...] = tuple(sorted(
        str(value)
        for value in split.loc[
            split["split"] == "validation", "path_name"
        ].tolist()
    ))
    if len(validation) != 20:
        raise ValueError(f"Expected 20 validation paths, found {len(validation)}")
    if set(training) & set(validation):
        raise ValueError("A validation path appears in training")

    original = pd.read_csv(dataset_dir / "original_100path_evaluation_manifest.csv")
    if not {"path_name", "has_v8_target"} <= set(original.columns):
        raise ValueError("original_100path_evaluation_manifest.csv has an invalid schema")
    original["path_name"] = pd.Series(
        [str(value) for value in original["path_name"].tolist()],
        index=original.index,
        dtype="string",
    )
    if original["path_name"].duplicated().any() or len(original) != 100:
        raise ValueError("The original evaluation manifest must contain 100 unique paths")
    has_target = pd.Series(
        [
            str(value).strip().lower() in {"true", "1"}
            for value in original["has_v8_target"].tolist()
        ],
        index=original.index,
        dtype=bool,
    )
    difficult: Tuple[str, ...] = tuple(sorted(
        str(value)
        for value in original.loc[~has_target, "path_name"].tolist()
    ))
    if difficult != DIFFICULT_PATHS:
        raise ValueError(
            f"Expected difficult no-target paths {DIFFICULT_PATHS}, found {difficult}"
        )
    split_paths = {str(value) for value in split["path_name"].tolist()}
    target_paths = {
        str(value)
        for value in original.loc[has_target, "path_name"].tolist()
    }
    if split_paths != target_paths:
        raise ValueError("The v8 split does not exactly cover the 98 target-bearing paths")
    return training, validation, difficult


def build_evaluation_windows(
    source_path: Path,
    validation_paths: Sequence[str],
    difficult_paths: Sequence[str],
    target_coverage: set[Tuple[str, int]],
    include_difficult: bool,
    max_primary_paths: Optional[int],
    max_primary_windows: Optional[int],
) -> Tuple[List[EvaluationWindow], Dict[str, Any]]:
    data = v7_target_generator.load_window_data(resolve_project_path(source_path))
    timelines = v7_target_generator.reconstruct_timelines(data)
    primary_names = list(sorted(validation_paths))
    if max_primary_paths is not None:
        primary_names = primary_names[:max_primary_paths]
    missing = sorted(set((*primary_names, *difficult_paths)) - set(timelines))
    if missing:
        raise ValueError(f"Authoritative source is missing evaluation paths: {missing}")
    selected_names = list(primary_names)
    if include_difficult:
        selected_names.extend(sorted(difficult_paths))
    selected_names = list(dict.fromkeys(selected_names))

    selected_mask = np.isin(data["path_names"], np.asarray(selected_names))
    selected_data = {
        key: np.asarray(value)[selected_mask] for key, value in data.items()
    }
    keys, groups = v7_dataset_builder.group_windows(selected_data)
    conditions = v7_dataset_builder.build_v6_conditions(selected_data, keys, groups)
    contexts = v7_target_generator.make_window_contexts(
        data, timelines, selected_names
    )
    contexts = sorted(contexts, key=lambda item: (item.path_name, item.window_start))

    by_path: Dict[str, List[int]] = {}
    for context in contexts:
        by_path.setdefault(context.path_name, []).append(context.window_start)
    expected_starts = list(v7_target_generator.EXPECTED_WINDOW_STARTS)
    for path_name in selected_names:
        starts = sorted(by_path.get(path_name, ()))
        if starts != expected_starts:
            raise ValueError(
                f"Primary/difficult source for {path_name} does not contain all "
                f"18 starts 0,4,...,68: {starts}"
            )

    primary_contexts = [item for item in contexts if item.path_name in primary_names]
    if max_primary_windows is not None:
        primary_contexts = primary_contexts[:max_primary_windows]
    difficult_contexts = [
        item for item in contexts
        if include_difficult and item.path_name in set(difficult_paths)
    ]
    kept_contexts = [*primary_contexts, *difficult_contexts]
    identities = [(item.path_name, item.window_start) for item in kept_contexts]
    if len(identities) != len(set(identities)):
        raise ValueError("Duplicate physical windows exist in the evaluation population")

    windows: List[EvaluationWindow] = []
    path_indices = {name: index for index, name in enumerate(selected_names)}
    for index, context in enumerate(kept_contexts):
        key = (context.path_name, context.window_start)
        condition = finite_array(
            f"condition[{key}]", np.asarray(conditions[key], dtype=np.float64)
        )
        if condition.shape != (HORIZON, V7_CONDITION_DIM):
            raise ValueError(f"Condition for {key} has shape {condition.shape}")
        population = "primary" if context.path_name in primary_names else "difficult"
        covered = key in target_coverage
        if population == "difficult" and covered:
            raise ValueError(f"Difficult no-target window unexpectedly has a target: {key}")
        windows.append(
            EvaluationWindow(
                canonical_index=index,
                path_name=context.path_name,
                path_index=path_indices[context.path_name],
                window_start=context.window_start,
                population=population,
                target_covered=covered,
                condition_v7=condition,
                context=context,
            )
        )

    primary_keys = {
        (window.path_name, window.window_start)
        for window in windows if window.population == "primary"
    }
    expected_primary = len(primary_names) * len(expected_starts)
    if max_primary_windows is None and len(primary_keys) != expected_primary:
        raise ValueError(
            f"A primary validation window is missing: expected {expected_primary}, "
            f"found {len(primary_keys)}"
        )
    return windows, {
        "primary_paths": primary_names,
        "primary_window_count": len(primary_keys),
        "primary_target_covered_window_count": sum(
            int(window.population == "primary" and window.target_covered)
            for window in windows
        ),
        "primary_zero_target_window_count": sum(
            int(window.population == "primary" and not window.target_covered)
            for window in windows
        ),
        "difficult_paths": list(difficult_paths) if include_difficult else [],
        "difficult_window_count": sum(
            int(window.population == "difficult") for window in windows
        ),
        "authoritative_window_starts": expected_starts,
    }


def audit_condition_reconstruction(
    windows: Sequence[EvaluationWindow],
    dataset_dir: Path,
    normalization: Mapping[str, np.ndarray],
) -> Dict[str, Any]:
    source_by_key = {
        (window.path_name, window.window_start): window.condition_v7
        for window in windows if window.population == "primary"
    }
    path = dataset_dir / "validation_windows.npz"
    with cast(Any, np.load)(path, allow_pickle=False) as archive:
        required = (
            "path_names", "window_start_indices", "target_scale",
            "condition", "condition_norm",
        )
        missing = [key for key in required if key not in archive.files]
        if missing:
            raise KeyError(f"{path} is missing condition-audit keys: {missing}")
        names = v8_trainer.decode_strings(np.asarray(archive["path_names"]))
        starts = np.asarray(archive["window_start_indices"], dtype=np.int64)
        scales = np.asarray(archive["target_scale"], dtype=np.float64)
        raw = np.asarray(archive["condition"], dtype=np.float64)
        normalized = np.asarray(archive["condition_norm"], dtype=np.float64)
    maximum_source_difference = 0.0
    audited_rows = 0
    mean = np.asarray(normalization["condition_mean"], dtype=np.float64)
    std = np.asarray(normalization["condition_std"], dtype=np.float64)
    for row, (name, start, scale) in enumerate(zip(names, starts, scales)):
        key = (str(name), int(start))
        if key not in source_by_key:
            continue
        expected_raw = np.concatenate(
            (
                source_by_key[key],
                np.full((HORIZON, 1), float(scale), dtype=np.float64),
            ),
            axis=1,
        )
        maximum_source_difference = max(
            maximum_source_difference,
            float(np.max(np.abs(expected_raw - raw[row]))),
        )
        if not np.allclose(expected_raw, raw[row], rtol=FLOAT_RTOL, atol=FLOAT_ATOL):
            raise ValueError(f"Reconstructed source condition differs for {key}")
        expected_norm = (expected_raw - mean) / std
        if not np.allclose(
            expected_norm, normalized[row],
            rtol=1.0e-5, atol=NORMALIZATION_ATOL,
        ):
            raise ValueError(f"Condition normalization is inconsistent for {key}")
        if not np.allclose(normalized[row, :, -1], scale, rtol=0.0, atol=FLOAT_ATOL):
            raise ValueError(f"Normalized target_scale is not raw for {key}")
        audited_rows += 1
    return {
        "audited_target_rows": audited_rows,
        "maximum_reconstructed_raw_condition_difference": maximum_source_difference,
        "selection_uses_target_rows": False,
        "audit_note": (
            "No retained-target row matched this pilot population; every evaluated "
            "condition is still checked by the normalization round-trip."
            if audited_rows == 0 else "Matched retained rows agree with reconstruction."
        ),
    }


def state_dict_hash(state: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name]
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"State entry {name!r} is not a tensor")
        cpu = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(cpu.dtype).encode("ascii"))
        digest.update(np.asarray(cpu.shape, dtype=np.int64).tobytes())
        digest.update(cpu.view(torch.uint8).numpy().tobytes())
    result = digest.hexdigest()
    if result != digest.hexdigest():
        raise AssertionError("Checkpoint state hash is inconsistent")
    return result


def extract_state(
    checkpoint: Mapping[str, Any], state_type: str, path: Path
) -> Mapping[str, torch.Tensor]:
    if state_type == "raw":
        state = checkpoint.get("raw_model_state_dict", checkpoint.get("model_state_dict"))
    elif state_type == "ema":
        state = checkpoint.get("ema_model_state_dict")
        if not isinstance(state, Mapping):
            container = checkpoint.get("ema_state_dict")
            state = container.get("shadow") if isinstance(container, Mapping) else None
    else:
        raise ValueError(f"Unknown state type {state_type}")
    if not isinstance(state, Mapping):
        raise KeyError(f"{path} has no {state_type} model state")
    return cast(Mapping[str, torch.Tensor], state)


def variant_label(source_tag: str, state_type: str, epoch: int) -> str:
    preferred = {
        "raw_total": "raw",
        "ema_total": "ema",
        "raw_epsilon": "raw",
        "ema_epsilon": "ema",
        "last": state_type,
    }
    if preferred[source_tag] == state_type:
        return f"{state_type}_{source_tag.split('_', 1)[-1]}_epoch{epoch}"
    return f"{state_type}_from_{source_tag}_epoch{epoch}"


def checkpoint_normalization(
    checkpoint: Mapping[str, Any], path: Path
) -> Dict[str, np.ndarray]:
    embedded = checkpoint.get("normalization_statistics")
    if not isinstance(embedded, Mapping):
        raise KeyError(f"{path} lacks normalization_statistics")
    result: Dict[str, np.ndarray] = {}
    for key, width in (
        ("condition_mean", CONDITION_DIM),
        ("condition_std", CONDITION_DIM),
        ("residual_mean", TARGET_DIM),
        ("residual_std", TARGET_DIM),
    ):
        values = finite_array(
            f"{path}/{key}", np.asarray(embedded.get(key), dtype=np.float64).reshape(-1)
        )
        if values.shape != (width,):
            raise ValueError(f"{path}/{key} has shape {values.shape}")
        if key.endswith("std") and np.any(values <= 0.0):
            raise ValueError(f"{path}/{key} must be positive")
        result[key] = values
    return result


def validate_checkpoint_contract(
    checkpoint: Mapping[str, Any],
    path: Path,
    normalization: Mapping[str, np.ndarray],
    feature_names: Sequence[str],
    normalization_hash: str,
    metadata_hash: str,
    expected_model_configuration: Mapping[str, Any],
) -> int:
    for key, expected in (
        ("horizon", HORIZON),
        ("condition_dim", CONDITION_DIM),
        ("target_dim", TARGET_DIM),
    ):
        if int(checkpoint.get(key, -1)) != expected:
            raise ValueError(f"{path}: checkpoint {key} is incompatible")
    if list(checkpoint.get("target_shape", ())) != [HORIZON, TARGET_DIM]:
        raise ValueError(f"{path}: checkpoint target_shape is incompatible")
    if checkpoint.get("prediction_target_type") != "epsilon":
        raise ValueError(f"{path}: checkpoint prediction target is not epsilon")
    checkpoint_features = checkpoint.get(
        "condition_feature_names", checkpoint.get("condition_feature_ordering", ())
    )
    if tuple(str(value) for value in checkpoint_features) != tuple(feature_names):
        raise ValueError(f"{path}: checkpoint condition feature order is incompatible")
    if checkpoint.get("model_configuration") != dict(expected_model_configuration):
        raise ValueError(f"{path}: checkpoint model configuration is incompatible")
    if checkpoint.get("normalization_sha256") != normalization_hash:
        raise ValueError(f"{path}: checkpoint normalization hash is incompatible")
    if checkpoint.get("dataset_metadata_sha256") != metadata_hash:
        raise ValueError(f"{path}: checkpoint dataset metadata hash is incompatible")
    embedded = checkpoint_normalization(checkpoint, path)
    for key in embedded:
        expected = np.asarray(normalization[key], dtype=np.float64).reshape(-1)
        if not np.array_equal(embedded[key], expected):
            if not np.allclose(embedded[key], expected, rtol=1.0e-7, atol=1.0e-9):
                raise ValueError(f"{path}: checkpoint {key} differs from the dataset")
    schedule = checkpoint.get("diffusion_schedule", checkpoint.get("diffusion_hyperparameters"))
    if not isinstance(schedule, Mapping):
        raise KeyError(f"{path}: checkpoint diffusion schedule is absent")
    steps = int(schedule.get("steps", -1))
    if steps < 1 or schedule.get("beta_schedule") != "linear":
        raise ValueError(f"{path}: checkpoint diffusion schedule is incompatible")
    if not np.isclose(float(schedule.get("beta_start", math.nan)), 1.0e-4) or not np.isclose(
        float(schedule.get("beta_end", math.nan)), 2.0e-2
    ):
        raise ValueError(f"{path}: checkpoint beta endpoints are incompatible")
    return steps


def load_checkpoint_variants(
    args: argparse.Namespace,
    normalization: Mapping[str, np.ndarray],
    feature_names: Sequence[str],
) -> Tuple[List[ModelVariant], pd.DataFrame]:
    normalization_path = args.dataset_dir / "normalization_stats.npz"
    metadata_path = args.dataset_dir / "dataset_metadata.json"
    normalization_hash = sha256_file(normalization_path)
    metadata_hash = sha256_file(metadata_path)
    expected_model, expected_configuration = v6_trainer.instantiate_v5_model(
        HORIZON, CONDITION_DIM, TARGET_DIM, 1000
    )
    del expected_model
    specifications = (
        ("best_raw_total_checkpoint", "raw_total", "raw"),
        ("best_ema_total_checkpoint", "ema_total", "ema"),
        ("best_raw_epsilon_checkpoint", "raw_epsilon", "raw"),
        ("best_ema_epsilon_checkpoint", "ema_epsilon", "ema"),
        ("last_checkpoint", "last", "raw"),
    )
    state_records: Dict[
        Tuple[str, str],
        Tuple[Path, str, int, int, Mapping[str, Any], Mapping[str, torch.Tensor]],
    ] = {}
    for argument, source_tag, _preferred in specifications:
        path = resolve_project_path(Path(getattr(args, argument)))
        checkpoint = v8_trainer.load_torch_checkpoint(path, torch.device("cpu"))
        steps = validate_checkpoint_contract(
            checkpoint,
            path,
            normalization,
            feature_names,
            normalization_hash,
            metadata_hash,
            expected_configuration,
        )
        if args.ddim_steps > steps:
            raise ValueError(f"--ddim_steps exceeds the schedule in {path}")
        epoch = int(checkpoint.get("epoch", -1))
        model_configuration = dict(checkpoint["model_configuration"])
        for state_type in ("raw", "ema"):
            source_state = extract_state(checkpoint, state_type, path)
            cpu_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in source_state.items()
            }
            state_records[(argument, state_type)] = (
                path,
                source_tag,
                steps,
                epoch,
                model_configuration,
                cpu_state,
            )
        checkpoint.clear()
        del checkpoint

    pending: List[Tuple[str, str]] = []
    for argument, _source_tag, preferred in specifications:
        pending.append((argument, preferred))
    for argument, _source_tag, preferred in specifications:
        pending.append((argument, "ema" if preferred == "raw" else "raw"))

    representatives: Dict[str, ModelVariant] = {}
    manifest_rows: List[Dict[str, Any]] = []
    for argument, state_type in pending:
        (
            path,
            source_tag,
            steps,
            epoch,
            model_configuration,
            state,
        ) = state_records[(argument, state_type)]
        state_hash = state_dict_hash(state)
        label = variant_label(source_tag, state_type, epoch)
        duplicate_of = ""
        evaluated = 0
        if state_hash in representatives:
            representative = representatives[state_hash]
            if not v7_evaluator.state_dicts_equal(state, representative.state_dict):
                raise RuntimeError(
                    f"Checkpoint state hash collision for {label} and {representative.label}"
                )
            duplicate_of = representative.label
        else:
            representative = ModelVariant(
                label=label,
                source_argument=argument,
                checkpoint_path=path,
                state_type=state_type,
                epoch=epoch,
                state_hash=state_hash,
                state_dict=state,
                diffusion_steps=steps,
                model_configuration=model_configuration,
            )
            representatives[state_hash] = representative
            evaluated = 1
        manifest_rows.append(
            {
                "input_checkpoint_argument": argument,
                "input_checkpoint_path": str(path.resolve()),
                "variant_name": label,
                "state_type": state_type,
                "epoch": epoch,
                "deterministic_state_hash": state_hash,
                "duplicate_of": duplicate_of,
                "evaluated": evaluated,
            }
        )
    variants = list(representatives.values())
    state_records.clear()
    if not variants:
        raise RuntimeError("No unique checkpoint states were found")
    available_labels = {variant.label for variant in variants}
    if args.checkpoint_states is not None:
        requested_labels = {str(value) for value in args.checkpoint_states}
        missing_labels = sorted(requested_labels - available_labels)
        if missing_labels:
            raise ValueError(
                "Requested --checkpoint_states were not found after checkpoint "
                f"deduplication: {missing_labels}; available={sorted(available_labels)}"
            )
        variants = [
            variant for variant in variants if variant.label in requested_labels
        ]
    selected_labels = {variant.label for variant in variants}
    manifest = pd.DataFrame(manifest_rows)
    manifest["selected_for_evaluation"] = (
        manifest["variant_name"].isin(selected_labels)
        & (manifest["evaluated"] == 1)
    ).astype(int)
    return variants, manifest


def make_condition(
    window: EvaluationWindow,
    target_scale: float,
    normalization: Mapping[str, np.ndarray],
) -> np.ndarray:
    raw = np.concatenate(
        (
            window.condition_v7,
            np.full((HORIZON, 1), float(target_scale), dtype=np.float64),
        ),
        axis=1,
    )
    if raw.shape != (HORIZON, CONDITION_DIM):
        raise AssertionError(f"V8 condition has shape {raw.shape}")
    mean = np.asarray(normalization["condition_mean"], dtype=np.float64)
    std = np.asarray(normalization["condition_std"], dtype=np.float64)
    normalized = finite_array("normalized v8 condition", (raw - mean) / std)
    reconstructed = normalized * std + mean
    if not np.allclose(reconstructed, raw, rtol=FLOAT_RTOL, atol=FLOAT_ATOL):
        raise ValueError("Condition normalization round-trip failed")
    if not np.allclose(
        normalized[:, -1], target_scale, rtol=0.0, atol=FLOAT_ATOL
    ):
        raise ValueError("target_scale was not inserted before normalization")
    return normalized.astype(np.float32)


def metric_values_are_finite(metrics: Mapping[str, Any]) -> bool:
    if not bool(metrics.get("finite", False)):
        return False
    for key, value in metrics.items():
        if key == "hard_limit_violations":
            continue
        if isinstance(value, np.ndarray) and not np.all(np.isfinite(value)):
            return False
        if isinstance(value, (float, np.floating)) and not np.isfinite(value):
            return False
    return True


def evaluate_priors(
    windows: Sequence[EvaluationWindow],
    robot: v7_target_generator.RobotContext,
) -> Dict[Tuple[str, int], Dict[str, Any]]:
    result: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for window in windows:
        key = (window.path_name, window.window_start)
        metrics = v7_evaluator.evaluate_metrics(
            robot, window.context, window.prior_q, EXECUTION_HORIZON
        )
        if not metric_values_are_finite(metrics):
            raise RuntimeError(f"Stored strong-prior metrics are nonfinite for {key}")
        hard_reasons = v7_evaluator.hard_safety_reasons(metrics, metrics)
        if hard_reasons:
            raise RuntimeError(
                f"Stored strong prior is hard-unsafe for {key}: {list(hard_reasons)}"
            )
        computed_ee = np.asarray(metrics["ee"], dtype=np.float64)
        if not np.allclose(
            computed_ee, window.prior_ee, rtol=1.0e-5, atol=2.0e-5
        ):
            discrepancy = float(np.max(np.abs(computed_ee - window.prior_ee)))
            raise ValueError(
                f"Stored prior FK differs from authoritative FK for {key}; "
                f"maximum_difference={discrepancy:.9g}"
            )
        result[key] = metrics
    return result


def sample_is_selectable(result: v7_evaluator.CandidateEvaluationResult) -> bool:
    decision = result.decision
    return bool(
        decision.hard_safe
        and decision.cartesian_improving
        and decision.delta_score < 0.0
        and decision.selectable
    )


def sample_selection_reasons(
    result: v7_evaluator.CandidateEvaluationResult,
) -> Tuple[str, ...]:
    reasons = list(result.decision.acceptance_reasons)
    if result.decision.delta_score >= 0.0:
        reasons.append("nonnegative_delta_score")
    return tuple(dict.fromkeys(reasons))


def select_nested_candidate(
    results: Sequence[v7_evaluator.CandidateEvaluationResult],
    k_value: int,
) -> Optional[int]:
    if any(field in FORBIDDEN_SELECTION_FIELDS for field in vars(results[0])):
        raise AssertionError("Target data reached candidate selection")
    eligible = [
        index for index, result in enumerate(results[:k_value])
        if sample_is_selectable(result)
    ]
    if not eligible:
        return None
    selected = min(eligible, key=lambda index: results[index].decision.delta_score)
    if not sample_is_selectable(results[selected]):
        raise AssertionError("Selected candidate is not selectable")
    return selected


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def configuration_id(
    checkpoint_state: str, target_scale: float, output_alpha: float, k_value: int
) -> str:
    return (
        f"{checkpoint_state}::target_scale={target_scale:.12g}::"
        f"output_alpha={output_alpha:.12g}::K={k_value}"
    )


def sample_base_id(
    variant: ModelVariant,
    window: EvaluationWindow,
    target_scale: float,
    sample_index: int,
) -> str:
    return (
        f"{variant.state_hash[:16]}::{window.path_name}::{window.window_start}::"
        f"scale={target_scale:.12g}::sample={sample_index}"
    )


def flatten_metrics(prefix: str, metrics: Mapping[str, Any]) -> Dict[str, Any]:
    return v7_evaluator.scalar_metrics(prefix, metrics)


def evaluate_variant(
    variant: ModelVariant,
    windows: Sequence[EvaluationWindow],
    prior_metrics: Mapping[Tuple[str, int], Dict[str, Any]],
    normalization: Mapping[str, np.ndarray],
    robot: v7_target_generator.RobotContext,
    executor: Optional[concurrent.futures.ProcessPoolExecutor],
    args: argparse.Namespace,
    device: torch.device,
) -> Tuple[
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    Dict[str, PlotRecord],
    Dict[str, float],
]:
    model, model_configuration = v6_trainer.instantiate_v5_model(
        HORIZON, CONDITION_DIM, TARGET_DIM, variant.diffusion_steps
    )
    if dict(model_configuration) != dict(variant.model_configuration):
        raise ValueError(f"Reconstructed model configuration differs for {variant.label}")
    model.load_state_dict(variant.state_dict, strict=True)
    model.to(device).eval()
    schedule = v6_trainer.build_schedule(variant.diffusion_steps, device)
    residual_mean = np.asarray(normalization["residual_mean"], dtype=np.float64)
    residual_std = np.asarray(normalization["residual_std"], dtype=np.float64)
    max_k = max(args.k_values)

    window_rows: List[Dict[str, Any]] = []
    sample_rows: List[Dict[str, Any]] = []
    artifacts: Dict[str, PlotRecord] = {}
    generated_count = 0
    scored_count = 0
    gpu_sampling_time = 0.0
    cpu_scoring_time = 0.0
    cpu_scoring_wall_time = 0.0

    try:
        for target_scale in args.target_scales:
            for window in windows:
                key = (window.path_name, window.window_start)
                prior = prior_metrics[key]
                condition_norm = make_condition(window, target_scale, normalization)
                seeds = [
                    stable_seed(
                        args.sampling_seed,
                        variant.state_hash,
                        window.path_name,
                        window.window_start,
                        format(float(target_scale), ".12g"),
                        sample_index,
                    )
                    for sample_index in range(max_k)
                ]
                sampled_batches: List[np.ndarray] = []
                window_gpu_time = 0.0
                for batch_start in range(0, max_k, args.gpu_batch_size):
                    batch_end = min(batch_start + args.gpu_batch_size, max_k)
                    batch_size = batch_end - batch_start
                    repeated_condition = np.repeat(
                        condition_norm[None, :, :], batch_size, axis=0
                    )
                    synchronize(device)
                    started = time.perf_counter()
                    sampled = v6_evaluator.sample_batch(
                        model,
                        repeated_condition,
                        seeds[batch_start:batch_end],
                        schedule,
                        args.ddim_steps,
                        "ddim",
                        args.eta,
                        device,
                    )
                    synchronize(device)
                    window_gpu_time += time.perf_counter() - started
                    sampled_batches.append(np.asarray(sampled, dtype=np.float64))
                sampled_norm = finite_array(
                    "sampled residual_q_norm", np.concatenate(sampled_batches, axis=0)
                )
                if sampled_norm.shape != (max_k, HORIZON, TARGET_DIM):
                    raise ValueError(
                        f"DDIM returned {sampled_norm.shape}; expected "
                        f"{(max_k, HORIZON, TARGET_DIM)}"
                    )
                base_residuals = finite_array(
                    "sampled residual_q",
                    sampled_norm * residual_std.reshape(1, 1, TARGET_DIM)
                    + residual_mean.reshape(1, 1, TARGET_DIM),
                )
                generated_count += max_k
                gpu_sampling_time += window_gpu_time

                for output_alpha in args.output_alphas:
                    candidates = finite_array(
                        "candidate_q",
                        window.prior_q[None, :, :]
                        + float(output_alpha) * base_residuals,
                    ).astype(np.float64)
                    candidate_ids = [
                        (
                            f"{sample_base_id(variant, window, target_scale, index)}::"
                            f"output_alpha={float(output_alpha):.12g}"
                        )
                        for index in range(max_k)
                    ]
                    tasks = [
                        v7_evaluator.CandidateEvaluationTask(
                            candidate_id=candidate_ids[index],
                            context=window.context,
                            candidate_q=candidates[index].copy(),
                            execution_horizon=EXECUTION_HORIZON,
                            prior_metrics=dict(prior),
                        )
                        for index in range(max_k)
                    ]
                    for task in tasks:
                        v7_evaluator.assert_cpu_pool_payload(task)
                    cpu_started = time.perf_counter()
                    results = v7_evaluator.evaluate_candidate_tasks(
                        tasks, robot, executor
                    )
                    cpu_scoring_wall_time += time.perf_counter() - cpu_started
                    cpu_scoring_time += sum(
                        result.evaluation_time_s for result in results
                    )
                    scored_count += max_k
                    if [result.candidate_id for result in results] != candidate_ids:
                        raise AssertionError(
                            "Candidate ordering changed because of worker completion order"
                        )

                    local_sample_rows: List[Dict[str, Any]] = []
                    for sample_index, result in enumerate(results):
                        metrics = result.metrics
                        if not metric_values_are_finite(metrics):
                            raise RuntimeError(
                                "Nonfinite generated-candidate metrics for "
                                f"{window.path_name}@{window.window_start}, "
                                f"sample={sample_index}"
                            )
                        decision = result.decision
                        selectable = sample_is_selectable(result)
                        row: Dict[str, Any] = {
                            "candidate_id": result.candidate_id,
                            "base_sample_id": sample_base_id(
                                variant, window, target_scale, sample_index
                            ),
                            "sample_seed": seeds[sample_index],
                            "sampling_seed": int(args.sampling_seed),
                            "sample_index": sample_index,
                            "checkpoint_state": variant.label,
                            "checkpoint_state_hash": variant.state_hash,
                            "checkpoint_epoch": variant.epoch,
                            "state_type": variant.state_type,
                            "path_name": window.path_name,
                            "path_index": window.path_index,
                            "window_start": window.window_start,
                            "population": window.population,
                            "target_covered": int(window.target_covered),
                            "target_scale": float(target_scale),
                            "output_alpha": float(output_alpha),
                            "hard_safe": int(decision.hard_safe),
                            "cartesian_improving": int(decision.cartesian_improving),
                            "negative_delta_score": int(decision.delta_score < 0.0),
                            "compatibility_gates_pass": int(decision.selectable),
                            "selectable": int(selectable),
                            "selected_for_k_values": "",
                            "rejection_reasons": "|".join(
                                sample_selection_reasons(result)
                            ),
                            "hard_safety_reasons": "|".join(decision.hard_safety_reasons),
                            "delta_score": float(decision.delta_score),
                            "absolute_cartesian_improvement_m": float(decision.improvement_m),
                            "relative_cartesian_improvement_percent": float(
                                100.0 * decision.relative_improvement
                            ),
                            "sampled_residual_norm_rms": float(
                                np.sqrt(np.mean(np.square(sampled_norm[sample_index])))
                            ),
                            "sampled_residual_rms_rad": float(
                                np.sqrt(np.mean(np.square(base_residuals[sample_index])))
                            ),
                            "applied_residual_rms_rad": float(
                                np.sqrt(
                                    np.mean(
                                        np.square(
                                            float(output_alpha)
                                            * base_residuals[sample_index]
                                        )
                                    )
                                )
                            ),
                            "gpu_sampling_time_per_sample_s": window_gpu_time / max_k,
                            "cpu_scoring_time_s": result.evaluation_time_s,
                        }
                        row.update(flatten_metrics("prior", prior))
                        row.update(flatten_metrics("candidate", metrics))
                        local_sample_rows.append(row)
                        sample_rows.append(row)

                    previous_subset: Tuple[str, ...] = ()
                    for k_value in sorted(args.k_values):
                        current_subset = tuple(
                            result.candidate_id for result in results[:k_value]
                        )
                        if previous_subset and current_subset[: len(previous_subset)] != previous_subset:
                            raise AssertionError("Nested K candidate subsets are inconsistent")
                        previous_subset = current_subset
                        selected_index = select_nested_candidate(results, k_value)
                        accepted = selected_index is not None
                        if accepted:
                            assert selected_index is not None
                            selected_result = results[selected_index]
                            selected_metrics = selected_result.metrics
                            selected_q = candidates[selected_index]
                            selected_seed = seeds[selected_index]
                            selected_score = selected_result.decision.delta_score
                            existing = local_sample_rows[selected_index]["selected_for_k_values"]
                            local_sample_rows[selected_index]["selected_for_k_values"] = (
                                f"{existing}|{k_value}" if existing else str(k_value)
                            )
                        else:
                            selected_index = -1
                            selected_metrics = prior
                            selected_q = window.prior_q.copy()
                            selected_seed = -1
                            selected_score = 0.0
                            if not np.array_equal(selected_q, window.prior_q):
                                raise AssertionError("Fallback output differs from the strong prior")
                        final_hard_reasons = v7_evaluator.hard_safety_reasons(
                            selected_metrics, prior
                        )
                        final_safe = metric_values_are_finite(selected_metrics) and not final_hard_reasons
                        if not final_safe:
                            raise RuntimeError(
                                f"Final selected/fallback output is unsafe for {key}"
                            )
                        prior_mean = float(prior["prefix_cartesian_mean_error_m"])
                        selected_mean = float(
                            selected_metrics["prefix_cartesian_mean_error_m"]
                        )
                        improvement = prior_mean - selected_mean
                        result_id = (
                            f"{configuration_id(variant.label, target_scale, output_alpha, k_value)}::"
                            f"{window.path_name}::{window.window_start}"
                        )
                        rejection_reasons = sorted(
                            {
                                reason
                                for result in results[:k_value]
                                for reason in sample_selection_reasons(result)
                            }
                        ) if not accepted else []
                        window_row: Dict[str, Any] = {
                            "result_id": result_id,
                            "configuration_id": configuration_id(
                                variant.label, target_scale, output_alpha, k_value
                            ),
                            "checkpoint_state": variant.label,
                            "checkpoint_state_hash": variant.state_hash,
                            "checkpoint_epoch": variant.epoch,
                            "state_type": variant.state_type,
                            "sampling_seed": int(args.sampling_seed),
                            "target_scale": float(target_scale),
                            "output_alpha": float(output_alpha),
                            "K": int(k_value),
                            "path_name": window.path_name,
                            "path_index": window.path_index,
                            "window_start": window.window_start,
                            "population": window.population,
                            "target_covered": int(window.target_covered),
                            "accepted": int(accepted),
                            "fallback": int(not accepted),
                            "final_safe": int(final_safe),
                            "selected_sample_index": int(selected_index),
                            "selected_seed": int(selected_seed),
                            "selected_delta_score": float(selected_score),
                            "absolute_cartesian_improvement_m": float(improvement),
                            "absolute_cartesian_improvement_mm": float(1000.0 * improvement),
                            "relative_cartesian_improvement_percent": float(
                                100.0 * improvement / max(prior_mean, 1.0e-12)
                            ),
                            "rejection_reasons": "|".join(rejection_reasons),
                            "gpu_sampling_time_per_sample_s": window_gpu_time / max_k,
                            "cpu_scoring_worker_time_sum_per_alpha_s": (
                                sum(item.evaluation_time_s for item in results)
                            ),
                        }
                        window_row.update(flatten_metrics("prior", prior))
                        window_row.update(flatten_metrics("selected", selected_metrics))
                        window_rows.append(window_row)
                        artifacts[result_id] = PlotRecord(
                            path_name=window.path_name,
                            window_start=window.window_start,
                            desired=window.desired.copy(),
                            prior_ee=np.asarray(prior["ee"], dtype=np.float64).copy(),
                            selected_ee=np.asarray(
                                selected_metrics["ee"], dtype=np.float64
                            ).copy(),
                        )
    finally:
        model.to("cpu")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return window_rows, sample_rows, artifacts, {
        "generated_candidate_count": float(generated_count),
        "cpu_scored_candidate_count": float(scored_count),
        "gpu_sampling_time_s": gpu_sampling_time,
        "cpu_scoring_time_s": cpu_scoring_time,
        "cpu_scoring_wall_time_s": cpu_scoring_wall_time,
    }


def subset_mask(frame: pd.DataFrame, subset: str) -> pd.Series:
    if subset == "primary_all":
        return frame["population"] == "primary"
    if subset == "primary_target_covered":
        return (frame["population"] == "primary") & (frame["target_covered"] == 1)
    if subset == "primary_zero_target":
        return (frame["population"] == "primary") & (frame["target_covered"] == 0)
    if subset == "difficult_no_target":
        return frame["population"] == "difficult"
    if subset == "combined_diagnostic":
        return pd.Series(True, index=frame.index, dtype=bool)
    raise ValueError(f"Unknown evaluation subset {subset}")


def raw_generator_classification(
    hard_safe_rate: float, selectable_rate: float
) -> str:
    if not np.isfinite(hard_safe_rate) or hard_safe_rate <= 0.0:
        return "RAW_GENERATOR_UNSAFE"
    if hard_safe_rate >= 1.0 - FLOAT_ATOL and selectable_rate > 0.0:
        return "RAW_GENERATOR_SAFE"
    return "RAW_GENERATOR_PARTIALLY_SAFE"


def gated_system_classification(
    final_safe_rate: float,
    accepted_rate: float,
    mean_improvement_m: float,
    accepted_rate_ci_lower: Optional[float] = None,
) -> str:
    if not np.isfinite(final_safe_rate) or final_safe_rate < 1.0 - FLOAT_ATOL:
        return "GATED_SYSTEM_UNSAFE"
    if accepted_rate <= 0.0 or mean_improvement_m <= 0.0:
        return "GATED_SYSTEM_NO_GAIN"
    lower = accepted_rate if accepted_rate_ci_lower is None else accepted_rate_ci_lower
    if accepted_rate > V7_HISTORICAL_ACCEPTED_RATE and lower > V7_HISTORICAL_ACCEPTED_RATE:
        return "GATED_SYSTEM_MEANINGFUL_GAIN"
    return "GATED_SYSTEM_SMALL_GAIN"


def contains_rejection_reason(value: Any, expected: str) -> bool:
    return expected in str(value).split("|")


def sort_frame(
    frame: pd.DataFrame,
    by: str | Sequence[str],
    ascending: bool | Sequence[bool] = True,
    *,
    na_position: Literal["first", "last"] = "last",
) -> pd.DataFrame:
    """Sort a frame while containing pandas-stub overload ambiguity."""
    return cast(
        pd.DataFrame,
        frame.sort_values(  # pyright: ignore[reportCallIssue]
            by=by,
            ascending=ascending,
            na_position=na_position,
        ),
    )


def rejection_counts(samples: pd.DataFrame) -> Dict[str, int]:
    return {
        reason: int(
            samples["rejection_reasons"].fillna("").astype(str).map(
                lambda value, expected=reason: contains_rejection_reason(
                    value, expected
                )
            ).sum()
        )
        for reason in REPORT_REJECTION_REASONS
    }


def aggregate_configurations(
    window_frame: pd.DataFrame, sample_frame: pd.DataFrame
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    configuration_columns = (
        "configuration_id",
        "checkpoint_state",
        "checkpoint_state_hash",
        "checkpoint_epoch",
        "state_type",
        "sampling_seed",
        "target_scale",
        "output_alpha",
        "K",
    )
    for keys, all_windows in window_frame.groupby(list(configuration_columns), sort=True):
        all_windows_frame = cast(pd.DataFrame, all_windows)
        key_values = cast(Tuple[Any, ...], keys)
        configuration = dict(zip(configuration_columns, key_values))
        k_value = int(configuration["K"])
        base_sample_mask = (
            (sample_frame["checkpoint_state"] == configuration["checkpoint_state"])
            & (sample_frame["sampling_seed"] == configuration["sampling_seed"])
            & np.isclose(sample_frame["target_scale"], float(configuration["target_scale"]))
            & np.isclose(sample_frame["output_alpha"], float(configuration["output_alpha"]))
            & (sample_frame["sample_index"] < k_value)
        )
        base_samples = cast(
            pd.DataFrame,
            sample_frame.loc[cast(Any, base_sample_mask), :].copy(),
        )
        for subset in SUBSET_ORDER:
            windows = cast(
                pd.DataFrame,
                all_windows_frame.loc[
                    cast(Any, subset_mask(all_windows_frame, subset)), :
                ].copy(),
            )
            if windows.empty:
                continue
            samples = cast(
                pd.DataFrame,
                base_samples.loc[
                    cast(Any, subset_mask(base_samples, subset)), :
                ].copy(),
            )
            accepted = cast(
                pd.DataFrame,
                windows.loc[cast(Any, windows["accepted"] == 1), :].copy(),
            )
            sample_count = len(samples)
            hard_safe_count = int(samples["hard_safe"].sum())
            improving_count = int(samples["cartesian_improving"].sum())
            negative_score_count = int(samples["negative_delta_score"].sum())
            selectable_count = int(samples["selectable"].sum())
            accepted_count = int(windows["accepted"].sum())
            final_safe_count = int(windows["final_safe"].sum())
            path_acceptance = windows.groupby("path_name", sort=True)["accepted"].sum()
            counts = rejection_counts(samples)
            hard_safe_rate = hard_safe_count / sample_count if sample_count else math.nan
            selectable_sample_rate = selectable_count / sample_count if sample_count else math.nan
            accepted_rate = accepted_count / len(windows)
            final_safe_rate = final_safe_count / len(windows)
            improvement = windows["absolute_cartesian_improvement_m"].to_numpy(dtype=float)
            accepted_improvement = accepted["absolute_cartesian_improvement_m"].to_numpy(dtype=float)
            accepted_scores = accepted["selected_delta_score"].to_numpy(dtype=float)
            row: Dict[str, Any] = {
                **configuration,
                "evaluation_subset": subset,
                "generated_sample_count": sample_count,
                "hard_safe_sample_count": hard_safe_count,
                "hard_safe_sample_rate": hard_safe_rate,
                "cartesian_improving_sample_count": improving_count,
                "cartesian_improving_sample_rate": (
                    improving_count / sample_count if sample_count else math.nan
                ),
                "negative_delta_score_sample_count": negative_score_count,
                "negative_delta_score_sample_rate": (
                    negative_score_count / sample_count if sample_count else math.nan
                ),
                "selectable_sample_count": selectable_count,
                "selectable_sample_rate": selectable_sample_rate,
                "total_window_count": len(windows),
                "path_count": windows["path_name"].nunique(),
                "accepted_window_count": accepted_count,
                "accepted_window_rate": accepted_rate,
                "fallback_window_count": int(windows["fallback"].sum()),
                "fallback_rate": float(windows["fallback"].mean()),
                "final_safe_window_count": final_safe_count,
                "final_safe_window_rate": final_safe_rate,
                "mean_cartesian_improvement_all_windows_m": float(np.mean(improvement)),
                "median_cartesian_improvement_all_windows_m": float(np.median(improvement)),
                "mean_cartesian_improvement_accepted_windows_m": (
                    float(np.mean(accepted_improvement))
                    if len(accepted_improvement) else math.nan
                ),
                "median_cartesian_improvement_accepted_windows_m": (
                    float(np.median(accepted_improvement))
                    if len(accepted_improvement) else math.nan
                ),
                "mean_robot_aware_delta_score_accepted_windows": (
                    float(np.mean(accepted_scores)) if len(accepted_scores) else math.nan
                ),
                "median_robot_aware_delta_score_accepted_windows": (
                    float(np.median(accepted_scores)) if len(accepted_scores) else math.nan
                ),
                "mean_robot_aware_improvement_accepted_windows": (
                    float(np.mean(-accepted_scores)) if len(accepted_scores) else math.nan
                ),
                "total_cartesian_improvement_m": float(np.sum(improvement)),
                "maximum_cartesian_improvement_m": float(np.max(improvement)),
                "paths_with_at_least_one_accepted_window": int((path_acceptance > 0).sum()),
                "fraction_paths_with_at_least_one_accepted_window": float(
                    (path_acceptance > 0).mean()
                ),
                "accepted_windows_per_path_mean": float(path_acceptance.mean()),
                "accepted_windows_per_path_median": float(path_acceptance.median()),
                "accepted_windows_per_path_minimum": int(path_acceptance.min()),
                "accepted_windows_per_path_maximum": int(path_acceptance.max()),
                "accepted_windows_per_path": json.dumps(
                    {str(name): int(value) for name, value in path_acceptance.items()},
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                "raw_generator_classification": raw_generator_classification(
                    hard_safe_rate, selectable_sample_rate
                ),
                "gated_system_classification": gated_system_classification(
                    final_safe_rate,
                    accepted_rate,
                    float(np.mean(improvement)),
                ),
                "rejection_counts_by_reason": json.dumps(
                    counts, sort_keys=True, separators=(",", ":")
                ),
            }
            row.update(
                {f"rejection_{reason}_count": count for reason, count in counts.items()}
            )
            rows.append(row)
    return pd.DataFrame(rows)


def aggregate_paths(window_frame: pd.DataFrame) -> pd.DataFrame:
    group_columns = (
        "configuration_id",
        "checkpoint_state",
        "checkpoint_state_hash",
        "checkpoint_epoch",
        "state_type",
        "sampling_seed",
        "target_scale",
        "output_alpha",
        "K",
        "population",
        "path_name",
    )
    rows: List[Dict[str, Any]] = []
    for keys, group in window_frame.groupby(list(group_columns), sort=True):
        group_frame = cast(pd.DataFrame, group)
        values = dict(zip(group_columns, cast(Tuple[Any, ...], keys)))
        accepted = cast(
            pd.DataFrame,
            group_frame.loc[cast(Any, group_frame["accepted"] == 1), :].copy(),
        )
        rows.append(
            {
                **values,
                "target_covered_window_count": int(group_frame["target_covered"].sum()),
                "zero_target_window_count": int((group_frame["target_covered"] == 0).sum()),
                "window_count": len(group_frame),
                "accepted_window_count": int(group_frame["accepted"].sum()),
                "accepted_window_rate": float(group_frame["accepted"].mean()),
                "fallback_window_count": int(group_frame["fallback"].sum()),
                "fallback_rate": float(group_frame["fallback"].mean()),
                "final_safe_window_rate": float(group_frame["final_safe"].mean()),
                "mean_cartesian_improvement_all_windows_m": float(
                    group_frame["absolute_cartesian_improvement_m"].mean()
                ),
                "median_cartesian_improvement_all_windows_m": float(
                    group_frame["absolute_cartesian_improvement_m"].median()
                ),
                "mean_cartesian_improvement_accepted_windows_m": (
                    float(accepted["absolute_cartesian_improvement_m"].mean())
                    if len(accepted) else math.nan
                ),
                "mean_robot_aware_delta_score_accepted_windows": (
                    float(accepted["selected_delta_score"].mean())
                    if len(accepted) else math.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def ranked_primary_configurations(
    summary: pd.DataFrame,
    *,
    native_only: bool,
) -> pd.DataFrame:
    primary_mask = cast(
        pd.Series,
        (summary["evaluation_subset"] == "primary_all")
        & (summary["output_alpha"] > 0.0),
    )
    if native_only:
        primary_mask = cast(
            pd.Series,
            primary_mask & np.isclose(summary["output_alpha"], 1.0),
        )
    primary = cast(
        pd.DataFrame,
        summary.loc[cast(Any, primary_mask), :].copy(),
    )
    if primary.empty:
        return primary
    return sort_frame(
        primary,
        [
            "accepted_window_rate",
            "mean_cartesian_improvement_all_windows_m",
            "hard_safe_sample_rate",
            "mean_robot_aware_delta_score_accepted_windows",
            "K",
            "checkpoint_state",
            "target_scale",
            "output_alpha",
        ],
        [False, False, False, True, False, True, True, True],
        na_position="last",
    )


def best_native_configuration(
    summary: pd.DataFrame,
) -> Optional[Dict[str, Any]]:
    ranked = ranked_primary_configurations(summary, native_only=True)
    if ranked.empty:
        return None
    return {str(key): value for key, value in ranked.iloc[0].to_dict().items()}


def best_diagnostic_configuration(
    summary: pd.DataFrame,
) -> Optional[Dict[str, Any]]:
    ranked = ranked_primary_configurations(summary, native_only=False)
    if ranked.empty:
        return None
    return {str(key): value for key, value in ranked.iloc[0].to_dict().items()}


def best_configuration(summary: pd.DataFrame) -> Dict[str, Any]:
    """Backward-compatible native-alpha optimum accessor."""
    best = best_native_configuration(summary)
    if best is None:
        raise RuntimeError(
            "No native primary output_alpha=1.0 configuration is available; "
            "use best_diagnostic_configuration() for diagnostic-only evaluations"
        )
    return best


def best_rows_by(
    summary: pd.DataFrame,
    group_columns: Sequence[str],
    *,
    optimum_scope: Literal["native", "diagnostic"],
) -> pd.DataFrame:
    primary = ranked_primary_configurations(
        summary,
        native_only=optimum_scope == "native",
    )
    if primary.empty:
        empty = primary.copy()
        empty["optimum_scope"] = pd.Series(dtype="string")
        return empty
    primary = sort_frame(
        primary,
        [
            *group_columns,
            "accepted_window_rate",
            "mean_cartesian_improvement_all_windows_m",
            "hard_safe_sample_rate",
        ],
        [*([True] * len(group_columns)), False, False, False],
    )
    result = cast(
        pd.DataFrame,
        primary.groupby(list(group_columns), as_index=False, sort=True).head(1),
    )
    result = result.copy()
    result.insert(0, "optimum_scope", optimum_scope)
    return result


def characterize_optimum(
    optimum: Optional[Mapping[str, Any]],
    bootstrap_frame: pd.DataFrame,
) -> Optional[Dict[str, Any]]:
    if optimum is None:
        return None
    result = {str(key): value for key, value in optimum.items()}
    interval_rows = bootstrap_frame[
        (bootstrap_frame["configuration_id"] == result["configuration_id"])
        & (bootstrap_frame["metric"] == "accepted_window_rate")
    ]
    if len(interval_rows) != 1:
        raise RuntimeError(
            "Optimum accepted-rate bootstrap row is missing or duplicated for "
            f"{result['configuration_id']}"
        )
    lower = float(interval_rows.iloc[0]["ci_95_lower"])
    upper = float(interval_rows.iloc[0]["ci_95_upper"])
    result["accepted_window_rate_ci_95_lower"] = lower
    result["accepted_window_rate_ci_95_upper"] = upper
    result["raw_generator_classification"] = raw_generator_classification(
        float(result["hard_safe_sample_rate"]),
        float(result["selectable_sample_rate"]),
    )
    result["gated_system_classification"] = gated_system_classification(
        float(result["final_safe_window_rate"]),
        float(result["accepted_window_rate"]),
        float(result["mean_cartesian_improvement_all_windows_m"]),
        lower,
    )
    return result


def v7_comparison_row(
    optimum_scope: Literal["native", "diagnostic"],
    optimum: Mapping[str, Any],
) -> Dict[str, Any]:
    accepted_rate = float(optimum["accepted_window_rate"])
    lower = float(optimum["accepted_window_rate_ci_95_lower"])
    upper = float(optimum["accepted_window_rate_ci_95_upper"])
    return {
        "optimum_scope": optimum_scope,
        "v7_best_accepted_window_rate": V7_HISTORICAL_ACCEPTED_RATE,
        "v7_reference_source": "externally supplied historical reference",
        "v8_best_accepted_window_rate": accepted_rate,
        "absolute_percentage_point_change": float(
            100.0 * (accepted_rate - V7_HISTORICAL_ACCEPTED_RATE)
        ),
        "relative_change_percent": float(
            100.0
            * (accepted_rate - V7_HISTORICAL_ACCEPTED_RATE)
            / V7_HISTORICAL_ACCEPTED_RATE
        ),
        "v8_accepted_window_rate_ci_95_lower": lower,
        "v8_accepted_window_rate_ci_95_upper": upper,
        "uncertainty_supports_rate_above_v7": int(
            lower > V7_HISTORICAL_ACCEPTED_RATE
        ),
        "v8_raw_hard_safe_rate": float(optimum["hard_safe_sample_rate"]),
        "v8_final_safe_rate": float(optimum["final_safe_window_rate"]),
        "v8_fallback_rate": float(optimum["fallback_rate"]),
        "sampling_seed": int(optimum["sampling_seed"]),
        "best_checkpoint_state": optimum["checkpoint_state"],
        "best_target_scale": float(optimum["target_scale"]),
        "best_output_alpha": float(optimum["output_alpha"]),
        "best_K": int(optimum["K"]),
        "significance_claim": (
            "path-bootstrap interval exceeds v7 historical rate"
            if lower > V7_HISTORICAL_ACCEPTED_RATE
            else "no significant-improvement claim"
        ),
    }


def bootstrap_configuration(
    rows: pd.DataFrame,
    bootstrap_samples: int,
    seed: int,
) -> List[Dict[str, Any]]:
    path_rows: List[Dict[str, float]] = []
    for _path_name, group in rows.groupby("path_name", sort=True):
        group_frame = cast(pd.DataFrame, group)
        accepted = group_frame["accepted"].to_numpy(dtype=float)
        selected_scores = cast(
            pd.Series,
            group_frame.loc[
                cast(Any, group_frame["accepted"] == 1),
                "selected_delta_score",
            ],
        )
        scores = -np.asarray(selected_scores, dtype=np.float64)
        path_rows.append(
            {
                "window_count": float(len(group_frame)),
                "accepted_count": float(np.sum(accepted)),
                "improvement_sum": float(
                    group_frame["absolute_cartesian_improvement_m"].sum()
                ),
                "robot_improvement_sum": float(np.sum(scores)),
            }
        )
    if not path_rows:
        raise ValueError("Cannot bootstrap an empty configuration")
    matrix = np.asarray(
        [
            [
                row["window_count"],
                row["accepted_count"],
                row["improvement_sum"],
                row["robot_improvement_sum"],
            ]
            for row in path_rows
        ],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    distributions = np.full((bootstrap_samples, 3), np.nan, dtype=np.float64)
    for sample_index in range(bootstrap_samples):
        selected = matrix[rng.integers(0, len(matrix), size=len(matrix))]
        totals = np.sum(selected, axis=0)
        distributions[sample_index, 0] = totals[1] / totals[0]
        distributions[sample_index, 1] = totals[2] / totals[0]
        if totals[1] > 0.0:
            distributions[sample_index, 2] = totals[3] / totals[1]
    estimates = (
        float(rows["accepted"].mean()),
        float(rows["absolute_cartesian_improvement_m"].mean()),
        (
            float(
                -rows.loc[rows["accepted"] == 1, "selected_delta_score"].mean()
            )
            if int(rows["accepted"].sum()) else math.nan
        ),
    )
    names = (
        "accepted_window_rate",
        "mean_all_window_cartesian_improvement_m",
        "mean_accepted_robot_aware_improvement",
    )
    result: List[Dict[str, Any]] = []
    for index, (name, estimate) in enumerate(zip(names, estimates)):
        finite = distributions[np.isfinite(distributions[:, index]), index]
        lower, upper = (
            np.quantile(finite, (0.025, 0.975))
            if len(finite) else (math.nan, math.nan)
        )
        result.append(
            {
                "metric": name,
                "estimate": estimate,
                "ci_95_lower": float(lower),
                "ci_95_upper": float(upper),
                "bootstrap_samples": bootstrap_samples,
                "bootstrap_seed": seed,
                "resampling_unit": "path_name",
                "paths_per_resample": len(matrix),
            }
        )
    return result


def bootstrap_all_configurations(
    window_frame: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    primary = window_frame[window_frame["population"] == "primary"].copy()
    columns = (
        "configuration_id",
        "checkpoint_state",
        "checkpoint_state_hash",
        "sampling_seed",
        "target_scale",
        "output_alpha",
        "K",
    )
    rows: List[Dict[str, Any]] = []
    for keys, group in primary.groupby(list(columns), sort=True):
        group_frame = cast(pd.DataFrame, group)
        values = dict(zip(columns, cast(Tuple[Any, ...], keys)))
        seed = stable_seed(args.bootstrap_seed, values["configuration_id"])
        for metric in bootstrap_configuration(
            group_frame, args.bootstrap_samples, seed
        ):
            rows.append(
                {
                    **values,
                    **metric,
                    "bootstrap_base_seed": args.bootstrap_seed,
                    "configuration_bootstrap_seed": seed,
                }
            )
    return pd.DataFrame(rows)


def save_figure(figure: Any, path: Path) -> None:
    figure.tight_layout()
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def placeholder_plot(path: Path, title: str, message: str) -> None:
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.axis("off")
    axis.set_title(title)
    axis.text(0.5, 0.5, message, ha="center", va="center", wrap=True)
    save_figure(figure, path)


def save_plots(
    configuration_summary: pd.DataFrame,
    per_window: pd.DataFrame,
    per_path: pd.DataFrame,
    diagnostic_best: Mapping[str, Any],
    artifacts: Mapping[str, PlotRecord],
    output_dir: Path,
    example_count: int,
) -> None:
    checkpoint = str(diagnostic_best["checkpoint_state"])
    diagnostic_alpha = float(diagnostic_best["output_alpha"])
    primary = configuration_summary[
        (configuration_summary["evaluation_subset"] == "primary_all")
        & (configuration_summary["checkpoint_state"] == checkpoint)
        & np.isclose(configuration_summary["output_alpha"], diagnostic_alpha)
    ].copy()

    figure, axis = plt.subplots(figsize=(8, 5))
    for k_value, group in primary.groupby("K", sort=True):
        group_frame = cast(pd.DataFrame, group)
        curve = sort_frame(group_frame, "target_scale")
        axis.plot(
            curve["target_scale"],
            100.0 * curve["accepted_window_rate"],
            marker="o",
            label=f"K={int(cast(Any, k_value))}",
        )
    axis.set_xlabel("Target scale")
    axis.set_ylabel("Accepted unique windows (%)")
    axis.set_title(
        f"Diagnostic accepted rate: {checkpoint}, alpha={diagnostic_alpha:g}"
    )
    axis.legend()
    save_figure(figure, output_dir / PLOT_FILES[0])

    figure, axis = plt.subplots(figsize=(8, 5))
    for k_value, group in primary.groupby("K", sort=True):
        group_frame = cast(pd.DataFrame, group)
        curve = sort_frame(group_frame, "target_scale")
        axis.plot(
            curve["target_scale"],
            100.0 * curve["hard_safe_sample_rate"],
            marker="o",
            label=f"K={int(cast(Any, k_value))}",
        )
    axis.set_xlabel("Target scale")
    axis.set_ylabel("Raw hard-safe samples (%)")
    axis.set_ylim(0.0, 105.0)
    axis.set_title(
        f"Diagnostic hard-safe rate: {checkpoint}, alpha={diagnostic_alpha:g}"
    )
    axis.legend()
    save_figure(figure, output_dir / PLOT_FILES[1])

    figure, axis = plt.subplots(figsize=(8, 5))
    for k_value, group in primary.groupby("K", sort=True):
        group_frame = cast(pd.DataFrame, group)
        curve = sort_frame(group_frame, "target_scale")
        axis.plot(
            curve["target_scale"],
            1000.0 * curve["mean_cartesian_improvement_all_windows_m"],
            marker="o",
            label=f"K={int(cast(Any, k_value))}",
        )
    axis.axhline(0.0, color="black", linewidth=1)
    axis.set_xlabel("Target scale")
    axis.set_ylabel("Mean all-window improvement (mm)")
    axis.set_title(
        f"Diagnostic gated improvement: {checkpoint}, alpha={diagnostic_alpha:g}"
    )
    axis.legend()
    save_figure(figure, output_dir / PLOT_FILES[2])

    best_path_mask = (
        (per_path["configuration_id"] == diagnostic_best["configuration_id"])
        & (per_path["population"] == "primary")
    )
    best_paths = sort_frame(
        cast(pd.DataFrame, per_path.loc[cast(Any, best_path_mask), :]),
        "path_name",
    )
    figure, axis = plt.subplots(figsize=(12, 5))
    axis.bar(best_paths["path_name"], best_paths["accepted_window_count"])
    axis.set_xlabel("Validation path")
    axis.set_ylabel("Accepted windows")
    axis.set_title("Accepted windows by validation path")
    axis.tick_params(axis="x", rotation=65)
    save_figure(figure, output_dir / PLOT_FILES[3])

    configuration_mask = (
        (configuration_summary["evaluation_subset"] == "primary_all")
        & (configuration_summary["output_alpha"] > 0.0)
    )
    configurations = cast(
        pd.DataFrame,
        configuration_summary.loc[cast(Any, configuration_mask), :].copy(),
    )
    configurations = sort_frame(
        configurations,
        ["accepted_window_rate", "fallback_rate"],
        [False, True],
    ).head(24)
    labels = [
        f"{row.checkpoint_state}\ns={row.target_scale:g}, "
        f"a={row.output_alpha:g}, K={int(cast(Any, row.K))}"
        for row in configurations.itertuples()
    ]
    figure, axis = plt.subplots(figsize=(14, 6))
    axis.bar(np.arange(len(configurations)), 100.0 * configurations["fallback_rate"])
    axis.set_xticks(np.arange(len(configurations)))
    axis.set_xticklabels(labels, rotation=65, ha="right", fontsize=7)
    axis.set_ylabel("Fallback windows (%)")
    axis.set_title("Fallback rate by leading configuration")
    save_figure(figure, output_dir / PLOT_FILES[4])

    best_window_mask = (
        (per_window["configuration_id"] == diagnostic_best["configuration_id"])
        & (per_window["population"] == "primary")
        & (per_window["accepted"] == 1)
    )
    best_windows = sort_frame(
        cast(pd.DataFrame, per_window.loc[cast(Any, best_window_mask), :]),
        "absolute_cartesian_improvement_m",
        False,
    )
    examples = best_windows.head(example_count)
    if examples.empty:
        placeholder_plot(
            output_dir / PLOT_FILES[5],
            "Selected example trajectories",
            "No selectable generated trajectory was found; every final output used the prior.",
        )
        return
    figure = plt.figure(figsize=(5 * len(examples), 5))
    for plot_index, row in enumerate(examples.itertuples(), start=1):
        artifact = artifacts[str(row.result_id)]
        axis = figure.add_subplot(1, len(examples), plot_index, projection="3d")
        axis.plot(*artifact.desired.T, label="desired")
        axis.plot(*artifact.prior_ee.T, label="prior")
        axis.plot(*artifact.selected_ee.T, label="selected")
        axis.set_title(
            f"{artifact.path_name}@{artifact.window_start}\n"
            f"gain={float(row.absolute_cartesian_improvement_mm):.3f} mm"
        )
        axis.legend(fontsize=7)
    save_figure(figure, output_dir / PLOT_FILES[5])


def prepare_output_directory(args: argparse.Namespace) -> None:
    existing = [
        args.output_dir / name
        for name in (*OUTPUT_FILES, *PLOT_FILES)
        if (args.output_dir / name).exists()
    ]
    if existing and not args.overwrite:
        raise FileExistsError(
            f"Evaluation outputs already exist: {existing}; pass --overwrite"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)


def json_safe(value: Any) -> Any:
    return v7_evaluator.json_safe(value)


def write_outputs(
    args: argparse.Namespace,
    sample_frame: pd.DataFrame,
    window_frame: pd.DataFrame,
    configuration_summary: pd.DataFrame,
    path_summary: pd.DataFrame,
    scale_summary: pd.DataFrame,
    checkpoint_summary: pd.DataFrame,
    checkpoint_manifest: pd.DataFrame,
    bootstrap_frame: pd.DataFrame,
    comparison: pd.DataFrame,
    evaluation_summary: Mapping[str, Any],
    timing_summary: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> None:
    v7_evaluator.atomic_csv(window_frame, args.output_dir / "per_window_results.csv")
    v7_evaluator.atomic_csv(
        sample_frame if args.save_per_sample_results else sample_frame.iloc[0:0],
        args.output_dir / "per_sample_results.csv",
    )
    v7_evaluator.atomic_csv(
        configuration_summary, args.output_dir / "configuration_summary.csv"
    )
    v7_evaluator.atomic_csv(path_summary, args.output_dir / "per_path_summary.csv")
    v7_evaluator.atomic_csv(scale_summary, args.output_dir / "scale_summary.csv")
    v7_evaluator.atomic_csv(
        checkpoint_summary, args.output_dir / "checkpoint_summary.csv"
    )
    v7_evaluator.atomic_csv(
        checkpoint_manifest, args.output_dir / "checkpoint_state_manifest.csv"
    )
    target_subsets = configuration_summary[
        configuration_summary["evaluation_subset"].isin(
            ("primary_target_covered", "primary_zero_target")
        )
    ].copy()
    difficult = configuration_summary[
        configuration_summary["evaluation_subset"] == "difficult_no_target"
    ].copy()
    v7_evaluator.atomic_csv(
        target_subsets, args.output_dir / "target_coverage_subset_summary.csv"
    )
    v7_evaluator.atomic_csv(
        difficult, args.output_dir / "difficult_path_summary.csv"
    )
    v7_evaluator.atomic_csv(
        bootstrap_frame, args.output_dir / "bootstrap_confidence_intervals.csv"
    )
    v7_evaluator.atomic_csv(
        comparison, args.output_dir / "v7_v8_comparison_summary.csv"
    )
    v7_evaluator.atomic_json(
        dict(evaluation_summary), args.output_dir / "evaluation_summary.json"
    )
    v7_evaluator.atomic_json(
        dict(timing_summary), args.output_dir / "timing_summary.json"
    )
    v7_evaluator.atomic_json(
        dict(metadata), args.output_dir / "evaluation_metadata.json"
    )


def run_evaluation(args: argparse.Namespace) -> int:
    wall_started = time.perf_counter()
    if args.sampling_seed is None:
        args.sampling_seed = int(args.seed)
    else:
        args.sampling_seed = int(args.sampling_seed)
    validate_cli(args)
    set_reproducibility(args.sampling_seed)
    device = resolve_device(args.device)
    args.dataset_dir = resolve_project_path(args.dataset_dir).resolve()
    args.target_generation_dir = resolve_project_path(
        args.target_generation_dir
    ).resolve()
    args.source_windows_npz = resolve_project_path(args.source_windows_npz).resolve()
    args.output_dir = resolve_project_path(args.output_dir).resolve()
    args.robot_urdf = resolve_project_path(args.robot_urdf).resolve()
    prepare_output_directory(args)

    metadata, normalization, feature_names, v7_features = validate_dataset_contract(
        args.dataset_dir
    )
    source_arguments = metadata.get("source_target_generation_metadata", {}).get(
        "arguments", {}
    )
    recorded_source = source_arguments.get("train_windows")
    if not recorded_source:
        raise KeyError("Dataset metadata does not record the authoritative train_windows")
    if resolve_project_path(Path(str(recorded_source))) != args.source_windows_npz:
        raise ValueError(
            "--source_windows_npz differs from the authoritative source recorded "
            "by v8 target generation"
        )
    recorded_targets = metadata.get("arguments", {}).get("targets_npz")
    requested_targets = args.target_generation_dir / "selected_targets.npz"
    if not recorded_targets or resolve_project_path(
        Path(str(recorded_targets))
    ) != requested_targets:
        raise ValueError(
            "--target_generation_dir differs from the target source recorded by "
            "the v8 dataset builder"
        )
    training_paths, validation_paths, difficult_paths = load_split_and_manifest(
        args.dataset_dir
    )
    if set(training_paths) & set(validation_paths):
        raise AssertionError("A validation path appears in training")
    target_coverage = load_target_coverage(
        args.target_generation_dir / "selected_targets.npz"
    )
    windows, population_report = build_evaluation_windows(
        args.source_windows_npz,
        validation_paths,
        difficult_paths,
        target_coverage,
        args.include_difficult_paths,
        args.max_primary_paths,
        args.max_primary_windows,
    )
    if args.max_primary_paths is None and args.max_primary_windows is None:
        if int(population_report["primary_window_count"]) != 360:
            raise ValueError("Complete v8 primary evaluation must contain 360 windows")
        expected_covered = int(metadata.get("counts", {}).get("validation_windows", -1))
        if int(population_report["primary_target_covered_window_count"]) != expected_covered:
            raise ValueError(
                "Target-covered primary-window count differs from dataset metadata"
            )
        if args.include_difficult_paths and int(
            population_report["difficult_window_count"]
        ) != 36:
            raise ValueError("Difficult evaluation must contain 36 windows")
    condition_audit = audit_condition_reconstruction(
        windows, args.dataset_dir, normalization
    )
    variants, checkpoint_manifest = load_checkpoint_variants(
        args, normalization, feature_names
    )

    logical_cpu_count = os.cpu_count() or 1
    active_cpu_workers = int(args.num_cpu_workers)
    gpu_name = torch.cuda.get_device_name(device) if device.type == "cuda" else "not used"
    print(f"device: {device}")
    print(f"GPU: {gpu_name}")
    print(f"logical CPU count: {logical_cpu_count}")
    print(f"requested CPU workers: {args.num_cpu_workers}")
    print(f"active CPU workers: {active_cpu_workers}")
    print(f"GPU batch size: {args.gpu_batch_size}")
    print(f"primary validation paths: {len(population_report['primary_paths'])}")
    print(f"primary validation windows: {population_report['primary_window_count']}")
    print(
        "target-covered and zero-target primary windows: "
        f"{population_report['primary_target_covered_window_count']} / "
        f"{population_report['primary_zero_target_window_count']}"
    )
    print(
        "difficult evaluation paths and windows: "
        f"{population_report['difficult_paths']} / "
        f"{population_report['difficult_window_count']}"
    )
    print(
        "checkpoint states and deduplicated hashes: "
        + ", ".join(
            f"{variant.label}={variant.state_hash[:12]}" for variant in variants
        )
    )
    print(f"target scales: {[float(value) for value in args.target_scales]}")
    print(f"output alphas: {[float(value) for value in args.output_alphas]}")
    print(f"diffusion sampling seed: {args.sampling_seed}")
    print(f"K values: {sorted(args.k_values)}")
    print(f"DDIM: steps={args.ddim_steps}, eta={args.eta}, Gaussian initialization")

    robot = v7_evaluator.make_robot_context(args.robot_urdf)
    prior_metrics = evaluate_priors(windows, robot)
    all_window_rows: List[Dict[str, Any]] = []
    all_sample_rows: List[Dict[str, Any]] = []
    artifacts: Dict[str, PlotRecord] = {}
    total_generated_candidates = 0
    total_cpu_scored_candidates = 0
    cumulative_gpu_sampling_time = 0.0
    cumulative_cpu_scoring_time = 0.0
    cumulative_cpu_scoring_wall_time = 0.0

    executor: Optional[concurrent.futures.ProcessPoolExecutor] = None
    try:
        if args.num_cpu_workers > 1:
            executor = concurrent.futures.ProcessPoolExecutor(
                max_workers=args.num_cpu_workers,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=v7_evaluator.initialize_candidate_worker,
                initargs=(str(args.robot_urdf),),
            )
        for variant_index, variant in enumerate(variants, start=1):
            print(
                f"evaluating checkpoint state {variant_index}/{len(variants)}: "
                f"{variant.label}"
            )
            rows, samples, variant_artifacts, timing = evaluate_variant(
                variant,
                windows,
                prior_metrics,
                normalization,
                robot,
                executor,
                args,
                device,
            )
            all_window_rows.extend(rows)
            all_sample_rows.extend(samples)
            artifacts.update(variant_artifacts)
            total_generated_candidates += int(timing["generated_candidate_count"])
            total_cpu_scored_candidates += int(timing["cpu_scored_candidate_count"])
            cumulative_gpu_sampling_time += timing["gpu_sampling_time_s"]
            cumulative_cpu_scoring_time += timing["cpu_scoring_time_s"]
            cumulative_cpu_scoring_wall_time += timing["cpu_scoring_wall_time_s"]
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    window_frame = pd.DataFrame(all_window_rows)
    sample_frame = pd.DataFrame(all_sample_rows)
    if window_frame.empty or sample_frame.empty:
        raise RuntimeError("Evaluation produced no rows")
    configuration_summary = aggregate_configurations(window_frame, sample_frame)
    path_summary = aggregate_paths(window_frame)
    bootstrap_frame = bootstrap_all_configurations(window_frame, args)
    native_best = characterize_optimum(
        best_native_configuration(configuration_summary),
        bootstrap_frame,
    )
    diagnostic_best = characterize_optimum(
        best_diagnostic_configuration(configuration_summary),
        bootstrap_frame,
    )
    if diagnostic_best is None:
        raise RuntimeError("No positive-alpha primary diagnostic configuration exists")
    best = diagnostic_best
    accepted_ci_lower = float(best["accepted_window_rate_ci_95_lower"])
    accepted_ci_upper = float(best["accepted_window_rate_ci_95_upper"])
    raw_classification = str(best["raw_generator_classification"])
    gated_classification = str(best["gated_system_classification"])

    configuration_summary["is_best_native_configuration"] = 0
    configuration_summary["is_best_diagnostic_configuration"] = 0
    for scope, optimum in (
        ("native", native_best),
        ("diagnostic", diagnostic_best),
    ):
        if optimum is None:
            continue
        matching_best = (
            (configuration_summary["configuration_id"] == optimum["configuration_id"])
            & (configuration_summary["evaluation_subset"] == "primary_all")
        )
        configuration_summary.loc[
            matching_best, f"is_best_{scope}_configuration"
        ] = 1
        configuration_summary.loc[
            matching_best, "gated_system_classification"
        ] = optimum["gated_system_classification"]

    checkpoint_summary = pd.concat(
        [
            best_rows_by(
                configuration_summary,
                ("checkpoint_state",),
                optimum_scope="native",
            ),
            best_rows_by(
                configuration_summary,
                ("checkpoint_state",),
                optimum_scope="diagnostic",
            ),
        ],
        ignore_index=True,
    )
    scale_summary = pd.concat(
        [
            best_rows_by(
                configuration_summary,
                ("target_scale", "output_alpha", "K"),
                optimum_scope="native",
            ),
            best_rows_by(
                configuration_summary,
                ("target_scale", "output_alpha", "K"),
                optimum_scope="diagnostic",
            ),
        ],
        ignore_index=True,
    )
    comparison_rows = [
        v7_comparison_row("diagnostic", diagnostic_best),
    ]
    if native_best is not None:
        comparison_rows.insert(0, v7_comparison_row("native", native_best))
    comparison = pd.DataFrame(comparison_rows)

    best_window_rows = window_frame[
        window_frame["configuration_id"] == best["configuration_id"]
    ]
    target_covered_best = best_window_rows[
        (best_window_rows["population"] == "primary")
        & (best_window_rows["target_covered"] == 1)
    ]
    zero_target_best = best_window_rows[
        (best_window_rows["population"] == "primary")
        & (best_window_rows["target_covered"] == 0)
    ]
    difficult_best = best_window_rows[best_window_rows["population"] == "difficult"]
    limited_population = bool(
        args.max_primary_paths is not None or args.max_primary_windows is not None
    )
    total_wall_time = time.perf_counter() - wall_started
    timing_summary = {
        "logical_cpu_count": logical_cpu_count,
        "requested_cpu_workers": args.num_cpu_workers,
        "active_cpu_workers": active_cpu_workers,
        "gpu_batch_size": args.gpu_batch_size,
        "generated_base_candidate_count": total_generated_candidates,
        "cpu_scored_candidate_count": total_cpu_scored_candidates,
        "cumulative_gpu_sampling_time_s": cumulative_gpu_sampling_time,
        "cumulative_cpu_fk_scoring_time_s": cumulative_cpu_scoring_time,
        "cumulative_cpu_fk_scoring_wall_time_s": cumulative_cpu_scoring_wall_time,
        "total_wall_time_s": total_wall_time,
    }
    diagnostic_headline = {
        "optimum_scope": "diagnostic",
        "accepted_unique_validation_windows": int(best["accepted_window_count"]),
        "all_unique_validation_windows": int(best["total_window_count"]),
        "accepted_window_rate": float(best["accepted_window_rate"]),
        "accepted_window_rate_ci_95": [accepted_ci_lower, accepted_ci_upper],
    }
    native_headline = (
        {
            "optimum_scope": "native",
            "accepted_unique_validation_windows": int(
                native_best["accepted_window_count"]
            ),
            "all_unique_validation_windows": int(native_best["total_window_count"]),
            "accepted_window_rate": float(native_best["accepted_window_rate"]),
            "accepted_window_rate_ci_95": [
                float(native_best["accepted_window_rate_ci_95_lower"]),
                float(native_best["accepted_window_rate_ci_95_upper"]),
            ],
        }
        if native_best is not None
        else None
    )
    evaluation_summary = {
        "raw_generator_classification": raw_classification,
        "gated_system_classification": gated_classification,
        "classification_is_provisional": limited_population,
        "sampling_seed": int(args.sampling_seed),
        "best_configuration": native_best if native_best is not None else best,
        "best_configuration_scope": (
            "native" if native_best is not None else "diagnostic"
        ),
        "best_native_configuration": native_best,
        "best_diagnostic_configuration": diagnostic_best,
        "primary_headline": diagnostic_headline,
        "native_primary_headline": native_headline,
        "diagnostic_primary_headline": diagnostic_headline,
        "target_covered_primary": {
            "accepted_windows": int(target_covered_best["accepted"].sum()),
            "total_windows": len(target_covered_best),
            "accepted_window_rate": (
                float(target_covered_best["accepted"].mean())
                if len(target_covered_best) else math.nan
            ),
        },
        "zero_target_primary": {
            "accepted_windows": int(zero_target_best["accepted"].sum()),
            "total_windows": len(zero_target_best),
            "accepted_window_rate": (
                float(zero_target_best["accepted"].mean())
                if len(zero_target_best) else math.nan
            ),
        },
        "difficult_no_target_paths": {
            "accepted_windows": int(difficult_best["accepted"].sum()),
            "total_windows": len(difficult_best),
            "accepted_window_rate": (
                float(difficult_best["accepted"].mean())
                if len(difficult_best) else math.nan
            ),
        },
        "v7_historical_reference": {
            "accepted_window_rate": V7_HISTORICAL_ACCEPTED_RATE,
            "comparison_rows": comparison.to_dict(orient="records"),
        },
        "target_leakage": False,
        "recursive_rollout_performed": False,
    }
    evaluation_metadata = {
        "arguments": vars(args),
        "device": str(device),
        "gpu_name": gpu_name,
        "population": population_report,
        "limited_pilot_population": limited_population,
        "condition_audit": condition_audit,
        "condition_dimension": CONDITION_DIM,
        "condition_feature_names": list(feature_names),
        "v7_condition_feature_names": list(v7_features),
        "target_scale_feature_index": feature_names.index("target_scale"),
        "condition_scale_handling": (
            "append raw target_scale as feature 39 before applying the saved "
            "39-D training normalization"
        ),
        "residual_application": "candidate_q = prior_q + output_alpha * generated_residual",
        "primary_output_alpha": 1.0,
        "sampling": {
            "initialization": "independent Gaussian noise",
            "algorithm": "DDIM reverse sampling",
            "sampling_seed": int(args.sampling_seed),
            "seed_scope": (
                "Python, NumPy, PyTorch CPU, PyTorch CUDA, and stable per-sample "
                "DDIM seeds only; population, path order, priors, gates, and scoring "
                "are unchanged"
            ),
            "nested_k": sorted(args.k_values),
            "max_k_generated_once_per_state_scale_window": True,
            "same_base_samples_reused_across_output_alphas": True,
        },
        "checkpoint_state_deduplication": {
            "hash": "SHA-256 over sorted names, dtype, shape, and tensor bytes",
            "input_state_count": len(checkpoint_manifest),
            "unique_evaluated_state_count": len(variants),
        },
        "selection_rule": (
            "Using generated data only: require v7 hard safety, lower execution-prefix "
            "Cartesian mean error, negative v7 delta_score, and every v7 compatibility "
            "gate; select minimum delta_score within nested first-K; otherwise use the "
            "unchanged strong prior."
        ),
        "target_data_use": (
            "retained target identities label post-hoc coverage subsets and audit the "
            "condition contract only; no target values enter generation or selection"
        ),
        "reused_v7_components": {
            "robot": "evaluate_diffusion_v7_teacher_forced_validation.make_robot_context",
            "metrics": "evaluate_diffusion_v7_teacher_forced_validation.evaluate_metrics",
            "gates_and_score": "evaluate_diffusion_v7_teacher_forced_validation.candidate_decision",
            "candidate_task": "evaluate_diffusion_v7_teacher_forced_validation.evaluate_candidate_task",
            "multiprocessing": "evaluate_diffusion_v7_teacher_forced_validation.evaluate_candidate_tasks",
            "worker_initializer": "evaluate_diffusion_v7_teacher_forced_validation.initialize_candidate_worker",
            "ddim": "evaluate_diffusion_v6_teacher_forced_validation.sample_batch",
        },
        "robot_convention": {
            "robot": "ROKAE xMateCR7",
            "joint_names": list(DEFAULT_JOINT_NAMES),
            "end_effector": DEFAULT_EE_LINK,
            "fk_calls": [
                "robot.update_cfg(cfg)",
                "robot.get_transform(frame_to='xMateCR7_link6')",
            ],
            "hard_joint_limit_tolerance_rad": HARD_JOINT_LIMIT_TOLERANCE_RAD,
            "safety_margin_rad": DEFAULT_JOINT_LIMIT_SAFETY_MARGIN_RAD,
            "maximum_joint_step_rad": MAXIMUM_JOINT_STEP_RAD,
        },
        "parallel_architecture": {
            "main_process": "checkpoint/model/CUDA schedule/DDIM/denormalization",
            "worker_processes": "CPU NumPy FK, safety, costs, gates, and score",
            "start_method": "serial" if args.num_cpu_workers == 1 else "spawn",
            "robot_lifetime": "one reusable xMateCR7 model per worker",
            "canonical_candidate_order_restored": True,
        },
        "classification_rules": {
            "raw": (
                "UNSAFE when no raw sample is hard-safe; SAFE only when all raw "
                "samples are hard-safe and some are selectable; PARTIALLY_SAFE otherwise"
            ),
            "gated": (
                "UNSAFE only when a final selected/fallback output is unsafe; NO_GAIN "
                "for no positive gated gain; MEANINGFUL_GAIN only when the accepted-rate "
                "95% path-bootstrap lower bound exceeds the historical v7 rate; "
                "SMALL_GAIN otherwise"
            ),
        },
        "bootstrap": {
            "unit": "path_name",
            "samples": args.bootstrap_samples,
            "seed": args.bootstrap_seed,
            "confidence_level": 0.95,
            "windows_within_path_kept_together": True,
        },
        "advancement": "No anchored or recursive rollout is performed automatically.",
        "dataset_metadata_classification": metadata.get("classification"),
    }

    write_outputs(
        args,
        sample_frame,
        window_frame,
        configuration_summary,
        path_summary,
        scale_summary,
        checkpoint_summary,
        checkpoint_manifest,
        bootstrap_frame,
        comparison,
        evaluation_summary,
        timing_summary,
        evaluation_metadata,
    )
    save_plots(
        configuration_summary,
        window_frame,
        path_summary,
        best,
        artifacts,
        args.output_dir,
        args.plot_example_count,
    )
    total_wall_time = time.perf_counter() - wall_started
    timing_summary["total_wall_time_s"] = total_wall_time
    v7_evaluator.atomic_json(
        timing_summary, args.output_dir / "timing_summary.json"
    )
    missing_outputs = [
        name for name in (*OUTPUT_FILES, *PLOT_FILES)
        if not (args.output_dir / name).is_file()
    ]
    if missing_outputs:
        raise RuntimeError(f"Evaluation did not write required outputs: {missing_outputs}")

    if native_best is None:
        print(
            "best native configuration: unavailable "
            "(output_alpha=1.0 was not evaluated)"
        )
    else:
        print(
            "best native configuration: "
            f"{native_best['configuration_id']} "
            f"(rate={float(native_best['accepted_window_rate']):.6f})"
        )
    print(
        "best diagnostic configuration: "
        f"{diagnostic_best['configuration_id']} "
        f"(rate={float(diagnostic_best['accepted_window_rate']):.6f})"
    )
    print(
        "diagnostic-optimum accepted primary windows and rate: "
        f"{int(best['accepted_window_count'])}/{int(best['total_window_count'])} "
        f"({100.0 * float(best['accepted_window_rate']):.2f}%)"
    )
    print(
        "accepted target-covered primary windows and rate: "
        f"{int(target_covered_best['accepted'].sum())}/{len(target_covered_best)} "
        f"({100.0 * float(target_covered_best['accepted'].mean()):.2f}%)"
        if len(target_covered_best) else "accepted target-covered primary windows: none"
    )
    print(
        "accepted zero-target primary windows and rate: "
        f"{int(zero_target_best['accepted'].sum())}/{len(zero_target_best)} "
        f"({100.0 * float(zero_target_best['accepted'].mean()):.2f}%)"
        if len(zero_target_best) else "accepted zero-target primary windows: none"
    )
    print(
        "difficult-path accepted windows and rate: "
        f"{int(difficult_best['accepted'].sum())}/{len(difficult_best)} "
        f"({100.0 * float(difficult_best['accepted'].mean()):.2f}%)"
        if len(difficult_best) else "difficult-path evaluation: not included"
    )
    print(f"raw hard-safe rate: {float(best['hard_safe_sample_rate']):.6f}")
    print(f"fallback rate: {float(best['fallback_rate']):.6f}")
    print(f"final safety rate: {float(best['final_safe_window_rate']):.6f}")
    print(
        "mean and median all-window improvements: "
        f"{1000.0 * float(best['mean_cartesian_improvement_all_windows_m']):.6f} / "
        f"{1000.0 * float(best['median_cartesian_improvement_all_windows_m']):.6f} mm"
    )
    print(
        "accepted-rate 95% path-bootstrap interval: "
        f"[{accepted_ci_lower:.6f}, {accepted_ci_upper:.6f}]"
    )
    print(
        "v7 comparison: historical=36.10%, "
        f"v8 diagnostic={100.0 * float(best['accepted_window_rate']):.2f}%"
    )
    print(f"total generated candidates: {total_generated_candidates}")
    print(f"total CPU-scored candidates: {total_cpu_scored_candidates}")
    print(f"GPU sampling time: {cumulative_gpu_sampling_time:.3f} s")
    print(f"CPU FK/scoring time: {cumulative_cpu_scoring_time:.3f} s")
    print(f"total wall time: {total_wall_time:.3f} s")
    classification_prefix = "provisional " if limited_population else ""
    print(f"{classification_prefix}raw-generator classification: {raw_classification}")
    print(f"{classification_prefix}gated-system classification: {gated_classification}")
    return 0


def main() -> int:
    return run_evaluation(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
