#!/usr/bin/env python3
"""Generate MLP v3 train and test predictions with explicit delta-q export.

The MLP v3 checkpoint predicts normalized residual joint motion:

    predicted_delta_q_norm -> predicted_delta_q -> predicted_q

where:

    predicted_delta_q = predicted_delta_q_norm * train_delta_std + train_delta_mean
    predicted_q = q_start[:, None, :] + predicted_delta_q

This script only runs inference/export. It does not train models or shell out to
other commands.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch

from generate_mlp_v3_test_predictions import (
    DEFAULT_CHECKPOINT,
    DEFAULT_TEST_NPZ,
    DEFAULT_TRAIN_NPZ,
    forward_batches,
    infer_input_array,
    infer_times,
    load_mlp_model,
    load_train_stats,
    path_names,
    require_keys,
    reshape_model_output,
    rmse,
    safe_path_name,
    try_fk_metrics,
    write_summary_csv,
)


DEFAULT_TRAIN_OUTPUT_DIR = Path("data/cartesian_expert_dataset_v3/mlp_v3_train_predictions")
DEFAULT_TEST_OUTPUT_DIR = Path("data/cartesian_expert_dataset_v3/mlp_v3_test_predictions")
DEFAULT_SUMMARY_NAME = "summary.csv"
REQUIRED_KEYS = (
    "condition_features",
    "condition_features_norm",
    "delta_q",
    "delta_q_norm",
    "desired_paths",
    "expert_q",
    "path_names",
    "q_start",
)
JOINT_COLUMNS = ("t", "q1", "q2", "q3", "q4", "q5", "q6")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate path-conditioned MLP v3 train and test predictions."
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--train_npz", type=Path, default=DEFAULT_TRAIN_NPZ)
    parser.add_argument("--test_npz", type=Path, default=DEFAULT_TEST_NPZ)
    parser.add_argument("--train_output_dir", type=Path, default=DEFAULT_TRAIN_OUTPUT_DIR)
    parser.add_argument("--test_output_dir", type=Path, default=DEFAULT_TEST_OUTPUT_DIR)
    parser.add_argument("--summary_name", type=str, default=DEFAULT_SUMMARY_NAME)
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    parser.add_argument("--batch_size", type=int, default=256)
    return parser.parse_args()


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset: {path}")
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def write_joint_csv(path: Path, values: np.ndarray, times: np.ndarray) -> None:
    if values.ndim != 2 or values.shape[1] != 6:
        raise ValueError(f"{path.name} values must have shape (T,6), got {values.shape}")
    if times.ndim != 1 or times.shape[0] != values.shape[0]:
        raise ValueError(f"{path.name} times must have shape ({values.shape[0]},), got {times.shape}")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(JOINT_COLUMNS)
        for t_value, row in zip(times, values):
            writer.writerow([f"{float(t_value):.10f}"] + [f"{float(value):.10f}" for value in row])


def assert_dataset_convention(split_name: str, data: Dict[str, np.ndarray]) -> None:
    expert_q = np.asarray(data["expert_q"], dtype=np.float64)
    delta_q = np.asarray(data["delta_q"], dtype=np.float64)
    q_start = np.asarray(data["q_start"], dtype=np.float64)
    expected_delta_q = expert_q - q_start[:, None, :]
    if delta_q.shape != expert_q.shape:
        raise RuntimeError(
            f"{split_name}: delta_q shape must match expert_q shape, got "
            f"{delta_q.shape} vs {expert_q.shape}"
        )
    if not np.allclose(delta_q, expected_delta_q, rtol=1e-5, atol=1e-7):
        max_error = float(np.max(np.abs(delta_q - expected_delta_q)))
        raise RuntimeError(
            f"{split_name}: delta_q does not match expert_q - q_start; max error={max_error:.12e}"
        )


def sanity_check_predictions(
    split_name: str,
    predicted_delta_q_norm: np.ndarray,
    predicted_delta_q: np.ndarray,
    predicted_q: np.ndarray,
    data: Dict[str, np.ndarray],
) -> None:
    expert_q = np.asarray(data["expert_q"], dtype=np.float64)
    q_start = np.asarray(data["q_start"], dtype=np.float64)

    if predicted_q.shape != expert_q.shape:
        raise RuntimeError(
            f"{split_name}: predicted_q shape must match expert_q shape, got "
            f"{predicted_q.shape} vs {expert_q.shape}"
        )
    if predicted_delta_q_norm.shape != predicted_q.shape:
        raise RuntimeError(
            f"{split_name}: predicted_delta_q_norm shape must match predicted_q shape, got "
            f"{predicted_delta_q_norm.shape} vs {predicted_q.shape}"
        )
    if predicted_delta_q.shape != predicted_q.shape:
        raise RuntimeError(
            f"{split_name}: predicted_delta_q shape must match predicted_q shape, got "
            f"{predicted_delta_q.shape} vs {predicted_q.shape}"
        )
    if np.allclose(predicted_q, predicted_delta_q_norm, rtol=1e-6, atol=1e-8):
        raise RuntimeError(
            f"{split_name}: predicted_q unexpectedly equals predicted_delta_q_norm; "
            "this usually means normalized residuals were exported as full q."
        )

    reconstructed_q = q_start[:, None, :] + predicted_delta_q
    if not np.allclose(predicted_q, reconstructed_q, rtol=1e-7, atol=1e-9):
        max_error = float(np.max(np.abs(predicted_q - reconstructed_q)))
        raise RuntimeError(
            f"{split_name}: predicted_q must equal q_start + predicted_delta_q; "
            f"max error={max_error:.12e}"
        )


def summary_row(
    path_name: str,
    predicted_q: np.ndarray,
    expert_q: np.ndarray,
    fk_mean: np.ndarray | None,
    fk_max: np.ndarray | None,
    idx: int,
) -> Dict[str, Any]:
    error = predicted_q - expert_q
    per_joint = np.sqrt(np.mean(np.square(error), axis=0))
    row: Dict[str, Any] = {
        "path_name": path_name,
        "q_rmse_vs_expert": f"{rmse(error):.12e}",
        "max_q_error_vs_expert": f"{float(np.max(np.abs(error))):.12e}",
    }
    for joint_idx, value in enumerate(per_joint):
        row[f"q{joint_idx + 1}_rmse_vs_expert"] = f"{float(value):.12e}"
    if fk_mean is not None and fk_max is not None:
        row["mean_cartesian_error"] = f"{float(fk_mean[idx]):.12e}"
        row["max_cartesian_error"] = f"{float(fk_max[idx]):.12e}"
    return row


def export_split_predictions(
    *,
    split_name: str,
    npz_path: Path,
    output_dir: Path,
    summary_name: str,
    model: torch.nn.Module,
    state: Dict[str, torch.Tensor],
    checkpoint: Dict[str, Any],
    train_delta_mean: np.ndarray,
    train_delta_std: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> None:
    data = load_npz(npz_path)
    require_keys(data, REQUIRED_KEYS)
    assert_dataset_convention(split_name, data)

    names = path_names(data)
    if len(names) != np.asarray(data["expert_q"]).shape[0]:
        raise RuntimeError(
            f"{split_name}: path_names count {len(names)} does not match "
            f"expert_q batch size {np.asarray(data['expert_q']).shape[0]}"
        )

    input_name, inputs = infer_input_array(data, state, checkpoint)
    times = infer_times(data, checkpoint)
    raw_pred = forward_batches(model, inputs, batch_size, device)
    raw_model_output = reshape_model_output(raw_pred, data)

    predicted_delta_q_norm = raw_model_output
    predicted_delta_q = predicted_delta_q_norm * train_delta_std + train_delta_mean
    q_start = np.asarray(data["q_start"], dtype=np.float64)
    predicted_q = q_start[:, None, :] + predicted_delta_q

    sanity_check_predictions(
        split_name,
        predicted_delta_q_norm,
        predicted_delta_q,
        predicted_q,
        data,
    )

    expert_q = np.asarray(data["expert_q"], dtype=np.float64)
    fk_mean, fk_max = try_fk_metrics(predicted_q, data)
    include_fk = fk_mean is not None and fk_max is not None

    summary_rows: List[Dict[str, Any]] = []
    for idx, name in enumerate(names):
        path_dir = output_dir / safe_path_name(name)
        write_joint_csv(path_dir / "raw_model_output.csv", raw_model_output[idx], times[idx])
        write_joint_csv(path_dir / "predicted_delta_q_norm.csv", predicted_delta_q_norm[idx], times[idx])
        write_joint_csv(path_dir / "predicted_delta_q.csv", predicted_delta_q[idx], times[idx])
        write_joint_csv(path_dir / "predicted_q.csv", predicted_q[idx], times[idx])
        summary_rows.append(
            summary_row(
                path_name=name,
                predicted_q=predicted_q[idx],
                expert_q=expert_q[idx],
                fk_mean=fk_mean,
                fk_max=fk_max,
                idx=idx,
            )
        )

    summary_csv = output_dir / summary_name
    write_summary_csv(summary_csv, summary_rows, include_fk)
    print(
        f"[{split_name}] input={input_name}, paths={len(names)}, "
        f"output_dir={output_dir}, summary={summary_csv}"
    )


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        print("[device] CUDA requested but unavailable; using CPU")
        return torch.device("cpu")
    return torch.device(device_arg)


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)

    model, checkpoint, state = load_mlp_model(args.checkpoint, device)
    train_delta_mean, train_delta_std, _, _ = load_train_stats(args.train_npz)

    export_split_predictions(
        split_name="train",
        npz_path=args.train_npz,
        output_dir=args.train_output_dir,
        summary_name=args.summary_name,
        model=model,
        state=state,
        checkpoint=checkpoint,
        train_delta_mean=train_delta_mean,
        train_delta_std=train_delta_std,
        batch_size=args.batch_size,
        device=device,
    )
    export_split_predictions(
        split_name="test",
        npz_path=args.test_npz,
        output_dir=args.test_output_dir,
        summary_name=args.summary_name,
        model=model,
        state=state,
        checkpoint=checkpoint,
        train_delta_mean=train_delta_mean,
        train_delta_std=train_delta_std,
        batch_size=args.batch_size,
        device=device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
