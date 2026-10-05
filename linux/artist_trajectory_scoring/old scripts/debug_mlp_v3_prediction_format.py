#!/usr/bin/env python3
"""Diagnose the numeric format of generated MLP v3 prediction CSVs.

This script does not modify diffusion, MLP training, or prediction files. It
loads saved predicted_q.csv files and evaluates several possible meanings:
  A. full q
  B. raw delta_q
  C. normalized delta_q
  D. normalized full q
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


DEFAULT_DATASET_DIR = Path("data/cartesian_expert_dataset_v3/diffusion_v2")
DEFAULT_PRED_DIR = Path("data/cartesian_expert_dataset_v3/mlp_v3_test_predictions")
JOINT_NAMES = ("q1", "q2", "q3", "q4", "q5", "q6")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug MLP v3 predicted_q.csv numeric format.")
    parser.add_argument("--dataset_dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--pred_dir", type=Path, default=DEFAULT_PRED_DIR)
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--max_paths", type=int, default=None)
    return parser.parse_args()


def split_path(dataset_dir: Path, split: str) -> Path:
    return dataset_dir / f"diffusion_{split}_v2.npz"


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset file: {path}")
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def require_keys(data: Dict[str, np.ndarray], keys: Sequence[str], label: str) -> None:
    missing = [key for key in keys if key not in data]
    if missing:
        raise KeyError(f"{label} missing required key(s): {', '.join(missing)}")


def subset_data(data: Dict[str, np.ndarray], max_paths: Optional[int]) -> Dict[str, np.ndarray]:
    if max_paths is None:
        return data
    if max_paths <= 0:
        raise ValueError("--max_paths must be positive")
    out: Dict[str, np.ndarray] = {}
    for key, value in data.items():
        if value.ndim > 0 and value.shape[0] >= max_paths:
            out[key] = value[:max_paths]
        else:
            out[key] = value
    return out


def path_names(data: Dict[str, np.ndarray]) -> List[str]:
    raw = np.asarray(data["path_names"])
    out: List[str] = []
    for value in raw:
        out.append(value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value))
    return out


def safe_path_name(name: str) -> str:
    return Path(str(name)).name.replace("/", "_").replace("\\", "_")


def pred_csv_path(pred_dir: Path, path_name: str) -> Path:
    return pred_dir / safe_path_name(path_name) / "predicted_q.csv"


def read_pred_csv(path: Path) -> np.ndarray:
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no CSV header")
        lowered = [field.strip().lower() for field in reader.fieldnames]
        if not all(name in lowered for name in JOINT_NAMES):
            raise ValueError(f"{path} must contain q1...q6 columns; found {reader.fieldnames}")
        rows: List[List[float]] = []
        for row in reader:
            row_lower = {key.strip().lower(): value for key, value in row.items() if key is not None}
            rows.append([float(row_lower[name]) for name in JOINT_NAMES])
    arr = np.asarray(rows, dtype=np.float64)
    if arr.shape != (100, 6):
        raise ValueError(f"{path} must have shape (100,6) from q columns, got {arr.shape}")
    return arr


def load_predictions(pred_dir: Path, names: Sequence[str]) -> Tuple[np.ndarray, List[str]]:
    preds: List[np.ndarray] = []
    missing: List[str] = []
    for name in names:
        path = pred_csv_path(pred_dir, name)
        if not path.exists():
            missing.append(name)
            preds.append(np.full((100, 6), np.nan, dtype=np.float64))
            continue
        preds.append(read_pred_csv(path))
    return np.stack(preds, axis=0), missing


def channel_stats(values: np.ndarray) -> List[Tuple[float, float, float, float]]:
    stats: List[Tuple[float, float, float, float]] = []
    for idx in range(values.shape[-1]):
        channel = values[..., idx]
        finite = channel[np.isfinite(channel)]
        if finite.size == 0:
            stats.append((float("nan"), float("nan"), float("nan"), float("nan")))
        else:
            stats.append(
                (
                    float(np.mean(finite)),
                    float(np.std(finite)),
                    float(np.min(finite)),
                    float(np.max(finite)),
                )
            )
    return stats


def print_channel_stats(label: str, values: np.ndarray) -> None:
    print(f"\n[{label}] stats mean/std/min/max per joint")
    for idx, (mean, std, min_value, max_value) in enumerate(channel_stats(values)):
        q6_marker = "  <-- q6" if idx == 5 else ""
        print(
            f"  q{idx + 1}: mean={mean:.12e}, std={std:.12e}, "
            f"min={min_value:.12e}, max={max_value:.12e}{q6_marker}"
        )


def finite_rows_mask(pred: np.ndarray) -> np.ndarray:
    return np.all(np.isfinite(pred), axis=(1, 2))


def metric_bundle(q_candidate: np.ndarray, expert_q: np.ndarray, names: Sequence[str]) -> Dict[str, object]:
    mask = finite_rows_mask(q_candidate)
    if not np.any(mask):
        return {
            "q_rmse": float("inf"),
            "max_q_error": float("inf"),
            "per_joint_rmse": np.full(6, np.inf),
            "path_mean_rmse": float("inf"),
            "worst_path": "<none>",
            "num_valid": 0,
        }

    error = q_candidate[mask] - expert_q[mask]
    per_path = np.sqrt(np.mean(np.square(error), axis=(1, 2)))
    valid_names = [name for name, keep in zip(names, mask) if keep]
    return {
        "q_rmse": float(np.sqrt(np.mean(np.square(error)))),
        "max_q_error": float(np.max(np.abs(error))),
        "per_joint_rmse": np.sqrt(np.mean(np.square(error), axis=(0, 1))),
        "path_mean_rmse": float(np.mean(per_path)),
        "worst_path": valid_names[int(np.argmax(per_path))],
        "num_valid": int(np.sum(mask)),
    }


def print_first_path_debug(
    names: Sequence[str],
    pred: np.ndarray,
    expert_q: np.ndarray,
    q_start: np.ndarray,
    delta_q: np.ndarray,
) -> None:
    if not names:
        return
    print("\n[first path debug]")
    print(f"  first path_name: {names[0]}")
    print(f"  expected prediction file path component: {safe_path_name(names[0])}")
    print("  first 3 rows pred:")
    print(np.array2string(pred[0, :3], precision=8, suppress_small=False))
    print("  first 3 rows expert_q:")
    print(np.array2string(expert_q[0, :3], precision=8, suppress_small=False))
    print("  q_start:")
    print(np.array2string(q_start[0], precision=8, suppress_small=False))
    print("  first 3 rows delta_q:")
    print(np.array2string(delta_q[0, :3], precision=8, suppress_small=False))
    print("  q6 first path diagnostics:")
    print(f"    pred q6 first 3:     {pred[0, :3, 5]}")
    print(f"    expert_q q6 first 3: {expert_q[0, :3, 5]}")
    print(f"    q_start q6:          {q_start[0, 5]}")
    print(f"    delta_q q6 first 3:  {delta_q[0, :3, 5]}")


def make_interpretations(
    pred: np.ndarray,
    q_start: np.ndarray,
    train_data: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    train_delta = np.asarray(train_data["delta_q"], dtype=np.float64)
    train_expert = np.asarray(train_data["expert_q"], dtype=np.float64)
    delta_mean = train_delta.mean(axis=(0, 1)).reshape(1, 1, 6)
    delta_std = train_delta.std(axis=(0, 1)).reshape(1, 1, 6)
    delta_std = np.maximum(delta_std, 1e-12)
    q_mean = train_expert.mean(axis=(0, 1)).reshape(1, 1, 6)
    q_std = train_expert.std(axis=(0, 1)).reshape(1, 1, 6)
    q_std = np.maximum(q_std, 1e-12)

    return {
        "full_q": pred,
        "raw_delta_q": q_start[:, None, :] + pred,
        "normalized_delta_q": q_start[:, None, :] + pred * delta_std + delta_mean,
        "normalized_full_q": pred * q_std + q_mean,
    }


def print_ranking(results: Dict[str, Dict[str, object]]) -> str:
    ranked = sorted(results.items(), key=lambda item: float(item[1]["q_rmse"]))
    print("\ninterpretation | q_rmse | max_q_error | path_mean_rmse | valid_paths | q1_rmse | q2_rmse | q3_rmse | q4_rmse | q5_rmse | q6_rmse")
    for name, metrics in ranked:
        per_joint = np.asarray(metrics["per_joint_rmse"], dtype=np.float64)
        print(
            f"{name} | "
            f"{float(metrics['q_rmse']):.12e} | "
            f"{float(metrics['max_q_error']):.12e} | "
            f"{float(metrics['path_mean_rmse']):.12e} | "
            f"{int(metrics['num_valid'])} | "
            + " | ".join(f"{value:.12e}" for value in per_joint)
        )
        print(f"  worst path: {metrics['worst_path']}")
    return ranked[0][0]


def print_path_alignment_report(pred_dir: Path, names: Sequence[str], missing: Sequence[str]) -> None:
    print("\n[path alignment]")
    print(f"  pred_dir: {pred_dir}")
    print(f"  dataset path count: {len(names)}")
    print(f"  missing prediction files: {len(missing)}")
    if missing:
        preview = ", ".join(str(name) for name in missing[:10])
        suffix = "..." if len(missing) > 10 else ""
        print(f"  missing path_names: {preview}{suffix}")
    if names:
        first = names[0]
        print(f"  first expected file: {pred_csv_path(pred_dir, first)}")


def main() -> int:
    args = parse_args()
    train_data = load_npz(split_path(args.dataset_dir, "train"))
    split_data = load_npz(split_path(args.dataset_dir, args.split))
    split_data = subset_data(split_data, args.max_paths)

    required = (
        "expert_q",
        "delta_q",
        "delta_q_norm",
        "q_start",
        "path_names",
        "condition_features",
        "condition_features_norm",
    )
    require_keys(train_data, ("expert_q", "delta_q"), "train split")
    require_keys(split_data, required, f"{args.split} split")

    names = path_names(split_data)
    expert_q = np.asarray(split_data["expert_q"], dtype=np.float64)
    delta_q = np.asarray(split_data["delta_q"], dtype=np.float64)
    delta_q_norm = np.asarray(split_data["delta_q_norm"], dtype=np.float64)
    q_start = np.asarray(split_data["q_start"], dtype=np.float64)

    pred, missing = load_predictions(args.pred_dir, names)
    print(f"Dataset directory: {args.dataset_dir}")
    print(f"Split: {args.split}")
    print(f"Prediction directory: {args.pred_dir}")
    print(f"Loaded predictions shape: {pred.shape}")
    print_path_alignment_report(args.pred_dir, names, missing)

    print_channel_stats("predicted raw CSV values", pred)
    print_channel_stats("expert_q", expert_q)
    print_channel_stats("delta_q", delta_q)
    print_channel_stats("delta_q_norm", delta_q_norm)
    print_first_path_debug(names, pred, expert_q, q_start, delta_q)

    print("\n[q6 focused stats]")
    for label, values in (
        ("pred raw", pred),
        ("expert_q", expert_q),
        ("delta_q", delta_q),
        ("delta_q_norm", delta_q_norm),
    ):
        finite = values[..., 5][np.isfinite(values[..., 5])]
        print(
            f"  {label}: mean={float(np.mean(finite)):.12e}, std={float(np.std(finite)):.12e}, "
            f"min={float(np.min(finite)):.12e}, max={float(np.max(finite)):.12e}"
        )

    interpretations = make_interpretations(pred, q_start, train_data)
    results = {
        name: metric_bundle(candidate, expert_q, names)
        for name, candidate in interpretations.items()
    }
    best = print_ranking(results)
    print(f'\nMost likely prediction format: {best}')

    if missing:
        print("\nNOTE: Missing prediction files can distort global metrics; rerun after generating all paths.")
    if best not in {"full_q", "raw_delta_q", "normalized_delta_q", "normalized_full_q"}:
        print("NOTE: No known interpretation matched cleanly; suspect wrong model/input convention or path alignment.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
