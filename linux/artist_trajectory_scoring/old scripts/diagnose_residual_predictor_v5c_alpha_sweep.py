#!/usr/bin/env python3
"""Diagnose whether the v5c residual predictor has useful direction but poor scale.

The deterministic model is evaluated once, then its denormalized residual is
scaled by each alpha:

    q_candidate = prior_q_window + alpha * predicted_residual_q

This script performs no training, FK evaluation, diffusion, or window stitching.
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch

from train_residual_window_predictor_v5c import TemporalCNNResidualPredictor


DEFAULT_CHECKPOINT = Path(
    "data/cartesian_expert_dataset_v3/"
    "residual_window_predictor_v5c_fk_condition/best_checkpoint.pt"
)
DEFAULT_DATASET_DIR = Path(
    "data/cartesian_expert_dataset_v3/"
    "diffusion_v5b_residual_windows_fk_condition"
)
DEFAULT_TEST_NPZ = DEFAULT_DATASET_DIR / "test_windows.npz"
DEFAULT_STATS_NPZ = DEFAULT_DATASET_DIR / "normalization_stats.npz"
DEFAULT_OUTPUT_DIR = Path(
    "data/cartesian_expert_dataset_v3/"
    "residual_window_predictor_v5c_fk_condition/alpha_sweep_diagnostic"
)
ALPHAS = (0.0, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0)
TARGET_DIM = 6
EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate scaled v5c deterministic residual predictions."
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--test_npz", type=Path, default=DEFAULT_TEST_NPZ)
    parser.add_argument("--stats_npz", type=Path, default=DEFAULT_STATS_NPZ)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--num_windows",
        type=int,
        default=0,
        help="Number of leading test windows to evaluate; 0 evaluates all windows.",
    )
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    parser.add_argument("--seed", type=int, default=0)
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


def load_checkpoint(path: Path, device: torch.device) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {path}")
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"Checkpoint does not contain model_state_dict: {path}")
    return checkpoint


def decode_names(values: np.ndarray) -> List[str]:
    names: List[str] = []
    for value in np.asarray(values).reshape(-1):
        if isinstance(value, bytes):
            names.append(value.decode("utf-8"))
        else:
            names.append(str(value))
    return names


def require_keys(data: Dict[str, np.ndarray], keys: Sequence[str], label: str) -> None:
    missing = [key for key in keys if key not in data]
    if missing:
        raise KeyError(f"{label} missing required key(s): {', '.join(missing)}")


def validate_test_data(
    data: Dict[str, np.ndarray],
    checkpoint: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], np.ndarray]:
    require_keys(
        data,
        (
            "condition_norm",
            "residual_q_norm",
            "prior_q_window",
            "expert_q_window",
            "path_names",
            "window_start_indices",
        ),
        "test windows",
    )
    condition = np.asarray(data["condition_norm"], dtype=np.float32)
    target_norm = np.asarray(data["residual_q_norm"], dtype=np.float32)
    prior_q = np.asarray(data["prior_q_window"], dtype=np.float32)
    expert_q = np.asarray(data["expert_q_window"], dtype=np.float32)
    names = decode_names(data["path_names"])
    starts = np.asarray(data["window_start_indices"], dtype=np.int64).reshape(-1)

    if condition.ndim != 3:
        raise ValueError(
            f"condition_norm must have shape (N,H,C), got {condition.shape}"
        )
    if target_norm.ndim != 3:
        raise ValueError(
            f"residual_q_norm must have shape (N,H,D), got {target_norm.shape}"
        )
    if condition.shape[:2] != target_norm.shape[:2]:
        raise ValueError(
            "condition_norm and residual_q_norm must share N,H, got "
            f"{condition.shape[:2]} and {target_norm.shape[:2]}"
        )
    if prior_q.shape != target_norm.shape or expert_q.shape != target_norm.shape:
        raise ValueError(
            "prior_q_window and expert_q_window must match residual_q_norm; got "
            f"{prior_q.shape}, {expert_q.shape}, and {target_norm.shape}"
        )

    expected_condition_dim = int(checkpoint["condition_dim"])
    expected_target_dim = int(checkpoint["target_dim"])
    expected_horizon = int(checkpoint["horizon"])
    if condition.shape[1:] != (expected_horizon, expected_condition_dim):
        raise ValueError(
            "Test condition shape does not match checkpoint: "
            f"{condition.shape[1:]} vs {(expected_horizon, expected_condition_dim)}"
        )
    if target_norm.shape[1:] != (expected_horizon, expected_target_dim):
        raise ValueError(
            "Test target shape does not match checkpoint: "
            f"{target_norm.shape[1:]} vs {(expected_horizon, expected_target_dim)}"
        )
    if expected_target_dim != TARGET_DIM:
        raise ValueError(
            f"Expected a {TARGET_DIM}-joint checkpoint, got target_dim={expected_target_dim}"
        )
    if len(names) != condition.shape[0]:
        raise ValueError(
            f"path_names length {len(names)} does not match N={condition.shape[0]}"
        )
    if starts.shape != (condition.shape[0],):
        raise ValueError(
            f"window_start_indices must have shape {(condition.shape[0],)}, "
            f"got {starts.shape}"
        )
    for name, values in (
        ("condition_norm", condition),
        ("residual_q_norm", target_norm),
        ("prior_q_window", prior_q),
        ("expert_q_window", expert_q),
    ):
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} contains non-finite values")
    return condition, target_norm, prior_q, expert_q, names, starts


def load_residual_stats(
    path: Path,
    checkpoint: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    stats = load_npz(path, "normalization stats")
    if "residual_mean" in stats and "residual_std" in stats:
        mean = np.asarray(stats["residual_mean"], dtype=np.float32)
        std = np.asarray(stats["residual_std"], dtype=np.float32)
    elif "residual_mean" in checkpoint and "residual_std" in checkpoint:
        mean = np.asarray(checkpoint["residual_mean"], dtype=np.float32)
        std = np.asarray(checkpoint["residual_std"], dtype=np.float32)
    else:
        raise KeyError("Missing residual_mean/residual_std in normalization stats")
    if mean.shape != (TARGET_DIM,) or std.shape != (TARGET_DIM,):
        raise ValueError(
            f"Residual stats must have shape ({TARGET_DIM},), got "
            f"{mean.shape} and {std.shape}"
        )
    if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)):
        raise ValueError("Residual normalization stats contain non-finite values")
    if np.any(std <= 0.0):
        raise ValueError("residual_std must be strictly positive")
    return mean, std


def instantiate_model(
    checkpoint: Dict[str, Any],
    device: torch.device,
) -> TemporalCNNResidualPredictor:
    model_config = checkpoint.get("model_config")
    if not isinstance(model_config, dict):
        raise ValueError("Checkpoint is missing model_config")
    model_class = str(model_config.get("model_class", ""))
    if model_class and model_class != "TemporalCNNResidualPredictor":
        raise ValueError(f"Unsupported checkpoint model_class: {model_class}")

    condition_dim = int(checkpoint["condition_dim"])
    target_dim = int(checkpoint["target_dim"])
    config_condition_dim = int(model_config.get("condition_dim", condition_dim))
    config_target_dim = int(model_config.get("target_dim", target_dim))
    if config_condition_dim != condition_dim or config_target_dim != target_dim:
        raise ValueError("Checkpoint model_config dimensions are inconsistent")

    model = TemporalCNNResidualPredictor(
        condition_dim=condition_dim,
        target_dim=target_dim,
        hidden_dim=int(model_config["hidden_dim"]),
        num_layers=int(model_config["num_layers"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model


def predict_all(
    model: torch.nn.Module,
    condition: np.ndarray,
    device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    predictions: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, condition.shape[0], batch_size):
            batch = torch.from_numpy(condition[start : start + batch_size]).to(
                device=device,
                dtype=torch.float32,
            )
            prediction = model(batch)
            predictions.append(
                prediction.detach().cpu().numpy().astype(np.float32)
            )
    output = np.concatenate(predictions, axis=0)
    if output.shape[:2] != condition.shape[:2] or output.shape[-1] != TARGET_DIM:
        raise RuntimeError(
            f"Unexpected model output shape {output.shape} for input {condition.shape}"
        )
    if not np.all(np.isfinite(output)):
        raise RuntimeError("Model prediction contains non-finite values")
    return output


def velocity_metrics(q: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    num_windows = q.shape[0]
    if q.shape[1] < 2:
        zeros = np.zeros(num_windows, dtype=np.float64)
        return zeros, zeros, zeros
    velocity = q[:, 1:, :] - q[:, :-1, :]
    velocity_rms = np.sqrt(np.mean(np.square(velocity), axis=(1, 2)))
    max_joint_step = np.max(np.abs(velocity), axis=(1, 2))
    if q.shape[1] < 3:
        acceleration_rms = np.zeros(num_windows, dtype=np.float64)
    else:
        acceleration = q[:, 2:, :] - 2.0 * q[:, 1:-1, :] + q[:, :-2, :]
        acceleration_rms = np.sqrt(
            np.mean(np.square(acceleration), axis=(1, 2))
        )
    return velocity_rms, acceleration_rms, max_joint_step


def make_rows(
    *,
    alphas: Sequence[float],
    predicted_residual_q: np.ndarray,
    prior_q: np.ndarray,
    expert_q: np.ndarray,
    names: Sequence[str],
    starts: np.ndarray,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    oracle_residual = expert_q - prior_q
    prior_rmse = np.sqrt(
        np.mean(np.square(prior_q - expert_q), axis=(1, 2))
    )

    for alpha in alphas:
        scaled_residual = float(alpha) * predicted_residual_q
        candidate_q = prior_q + scaled_residual
        candidate_rmse = np.sqrt(
            np.mean(np.square(candidate_q - expert_q), axis=(1, 2))
        )
        residual_rmse = np.sqrt(
            np.mean(
                np.square(scaled_residual - oracle_residual),
                axis=(1, 2),
            )
        )
        improvement = 100.0 * (prior_rmse - candidate_rmse) / np.maximum(
            prior_rmse,
            EPS,
        )
        improved = candidate_rmse < prior_rmse
        velocity_rms, acceleration_rms, max_joint_step = velocity_metrics(
            candidate_q
        )

        for idx in range(prior_q.shape[0]):
            rows.append(
                {
                    "sample_index": idx,
                    "path_name": names[idx],
                    "window_start_index": int(starts[idx]),
                    "mode": "prior_only" if alpha == 0.0 else "scaled_prediction",
                    "alpha": float(alpha),
                    "prior_rmse": float(prior_rmse[idx]),
                    "candidate_rmse_to_expert": float(candidate_rmse[idx]),
                    "improvement_vs_prior_percent": float(improvement[idx]),
                    "improved": int(improved[idx]),
                    "residual_rmse_to_oracle": float(residual_rmse[idx]),
                    "velocity_rms": float(velocity_rms[idx]),
                    "acceleration_rms": float(acceleration_rms[idx]),
                    "max_joint_step": float(max_joint_step[idx]),
                }
            )
    return rows


def aggregate_rows(
    rows: Sequence[Dict[str, Any]],
    alphas: Sequence[float],
) -> List[Dict[str, Any]]:
    aggregate: List[Dict[str, Any]] = []
    for alpha in alphas:
        group = [row for row in rows if float(row["alpha"]) == float(alpha)]
        if not group:
            continue

        def values(key: str) -> np.ndarray:
            return np.asarray([float(row[key]) for row in group], dtype=np.float64)

        improved_count = int(sum(int(row["improved"]) for row in group))
        count = len(group)
        aggregate.append(
            {
                "mode": "prior_only" if alpha == 0.0 else "scaled_prediction",
                "alpha": float(alpha),
                "count": count,
                "prior_rmse": float(np.mean(values("prior_rmse"))),
                "candidate_rmse_to_expert": float(
                    np.mean(values("candidate_rmse_to_expert"))
                ),
                "improvement_vs_prior_percent": float(
                    np.mean(values("improvement_vs_prior_percent"))
                ),
                "improved_window_count": improved_count,
                "improved_window_ratio": float(improved_count / max(count, 1)),
                "residual_rmse_to_oracle": float(
                    np.mean(values("residual_rmse_to_oracle"))
                ),
                "velocity_rms": float(np.mean(values("velocity_rms"))),
                "acceleration_rms": float(
                    np.mean(values("acceleration_rms"))
                ),
                "max_joint_step": float(np.mean(values("max_joint_step"))),
            }
        )
    return aggregate


def write_summary(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    fields = [
        "sample_index",
        "path_name",
        "window_start_index",
        "mode",
        "alpha",
        "prior_rmse",
        "candidate_rmse_to_expert",
        "improvement_vs_prior_percent",
        "improved",
        "residual_rmse_to_oracle",
        "velocity_rms",
        "acceleration_rms",
        "max_joint_step",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            output = dict(row)
            output["alpha"] = f"{float(output['alpha']):.12g}"
            for key in (
                "prior_rmse",
                "candidate_rmse_to_expert",
                "improvement_vs_prior_percent",
                "residual_rmse_to_oracle",
                "velocity_rms",
                "acceleration_rms",
                "max_joint_step",
            ):
                output[key] = f"{float(output[key]):.12e}"
            writer.writerow({field: output[field] for field in fields})


def write_aggregate(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    fields = [
        "mode",
        "alpha",
        "count",
        "prior_rmse",
        "candidate_rmse_to_expert",
        "improvement_vs_prior_percent",
        "improved_window_count",
        "improved_window_ratio",
        "residual_rmse_to_oracle",
        "velocity_rms",
        "acceleration_rms",
        "max_joint_step",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            output = dict(row)
            output["alpha"] = f"{float(output['alpha']):.12g}"
            for key in (
                "prior_rmse",
                "candidate_rmse_to_expert",
                "improvement_vs_prior_percent",
                "improved_window_ratio",
                "residual_rmse_to_oracle",
                "velocity_rms",
                "acceleration_rms",
                "max_joint_step",
            ):
                output[key] = f"{float(output[key]):.12e}"
            writer.writerow({field: output[field] for field in fields})


def main() -> int:
    args = parse_args()
    if args.num_windows < 0:
        raise ValueError("--num_windows must be non-negative")

    set_seed(args.seed)
    device = resolve_device(args.device)
    checkpoint = load_checkpoint(args.checkpoint, device)
    model = instantiate_model(checkpoint, device)
    test_data = load_npz(args.test_npz, "test windows")
    condition, _, prior_q, expert_q, names, starts = validate_test_data(
        test_data,
        checkpoint,
    )
    residual_mean, residual_std = load_residual_stats(
        args.stats_npz,
        checkpoint,
    )

    count = (
        condition.shape[0]
        if args.num_windows == 0
        else min(args.num_windows, condition.shape[0])
    )
    condition = condition[:count]
    prior_q = prior_q[:count]
    expert_q = expert_q[:count]
    names = names[:count]
    starts = starts[:count]

    predicted_norm = predict_all(model, condition, device)
    predicted_residual_q = (
        predicted_norm * residual_std.reshape(1, 1, TARGET_DIM)
        + residual_mean.reshape(1, 1, TARGET_DIM)
    ).astype(np.float32)

    rows = make_rows(
        alphas=ALPHAS,
        predicted_residual_q=predicted_residual_q,
        prior_q=prior_q,
        expert_q=expert_q,
        names=names,
        starts=starts,
    )
    aggregate = aggregate_rows(rows, ALPHAS)

    summary_path = args.output_dir / "alpha_sweep_summary.csv"
    aggregate_path = args.output_dir / "alpha_sweep_aggregate.csv"
    write_summary(summary_path, rows)
    write_aggregate(aggregate_path, aggregate)

    checkpoint_epoch = int(checkpoint.get("epoch", -1))
    print(
        f"Evaluated {count} windows from checkpoint epoch {checkpoint_epoch} "
        f"on {device}"
    )
    for row in aggregate:
        print(
            f"alpha={float(row['alpha']):.3g} | "
            f"candidate_rmse={float(row['candidate_rmse_to_expert']):.8e} | "
            f"improvement={float(row['improvement_vs_prior_percent']):.3f}% | "
            f"improved={int(row['improved_window_count'])}/{int(row['count'])}"
        )
    print(f"Saved per-window summary: {summary_path}")
    print(f"Saved aggregate summary: {aggregate_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
