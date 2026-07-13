#!/usr/bin/env python3
"""Debug arbitrary prior trajectory file formats before diffusion refinement.

The script tests whether prior files are full q, raw delta_q, normalized
delta_q, or normalized full q, and reports path alignment issues.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


DEFAULT_DATASET_DIR = Path("data/cartesian_expert_dataset_v3/diffusion_v2")
JOINT_NAMES = ("q1", "q2", "q3", "q4", "q5", "q6")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug arbitrary prior prediction trajectory format.")
    parser.add_argument("--dataset_dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--pred_dir", type=Path, required=True)
    parser.add_argument("--pred_filename", type=Path, required=True)
    parser.add_argument("--split", choices=("test", "train"), default="test")
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
    names: List[str] = []
    for value in raw:
        names.append(value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value))
    return names


def safe_path_name(name: str) -> str:
    return Path(str(name)).name.replace("/", "_").replace("\\", "_")


def pred_path(pred_dir: Path, path_name: str, pred_filename: Path) -> Path:
    return pred_dir / safe_path_name(path_name) / pred_filename


def read_prediction_csv(path: Path) -> np.ndarray:
    with path.open("r", newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        try:
            has_header = csv.Sniffer().has_header(sample) if sample.strip() else False
        except csv.Error:
            has_header = True

        if has_header:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"{path} has no CSV header")
            field_map = {field.strip().lower(): field for field in reader.fieldnames if field is not None}
            rows: List[List[float]] = []
            if all(name in field_map for name in JOINT_NAMES):
                for row in reader:
                    rows.append([float(row[field_map[name]]) for name in JOINT_NAMES])
            else:
                for row in reader:
                    numeric: List[float] = []
                    for key, value in row.items():
                        if key is None or value is None or value == "":
                            continue
                        if key.strip().lower() == "t":
                            continue
                        try:
                            numeric.append(float(value))
                        except ValueError:
                            continue
                    if len(numeric) >= 6:
                        rows.append(numeric[-6:])
        else:
            reader = csv.reader(handle)
            rows = []
            for row in reader:
                if not row:
                    continue
                numeric = [float(value) for value in row if value.strip()]
                if len(numeric) == 7:
                    numeric = numeric[1:]
                if len(numeric) >= 6:
                    rows.append(numeric[-6:])

    arr = np.asarray(rows, dtype=np.float64)
    if arr.shape != (100, 6):
        raise ValueError(f"{path} must contain trajectory shape (100,6), got {arr.shape}")
    return arr


def load_predictions(
    pred_dir: Path,
    pred_filename: Path,
    names: Sequence[str],
) -> Tuple[np.ndarray, List[str], List[str]]:
    preds: List[np.ndarray] = []
    missing: List[str] = []
    read_errors: List[str] = []
    for name in names:
        path = pred_path(pred_dir, name, pred_filename)
        if not path.exists():
            missing.append(name)
            preds.append(np.full((100, 6), np.nan, dtype=np.float64))
            continue
        try:
            preds.append(read_prediction_csv(path))
        except Exception as exc:
            read_errors.append(f"{name}: {exc}")
            preds.append(np.full((100, 6), np.nan, dtype=np.float64))
    return np.stack(preds, axis=0), missing, read_errors


def channel_stats(values: np.ndarray) -> List[Tuple[float, float, float, float]]:
    out: List[Tuple[float, float, float, float]] = []
    for idx in range(values.shape[-1]):
        channel = values[..., idx]
        finite = channel[np.isfinite(channel)]
        if finite.size == 0:
            out.append((float("nan"), float("nan"), float("nan"), float("nan")))
        else:
            out.append(
                (
                    float(np.mean(finite)),
                    float(np.std(finite)),
                    float(np.min(finite)),
                    float(np.max(finite)),
                )
            )
    return out


def print_channel_stats(label: str, values: np.ndarray) -> None:
    print(f"\n[{label}] stats mean/std/min/max per joint")
    for idx, (mean, std, min_value, max_value) in enumerate(channel_stats(values)):
        marker = "  <-- q6" if idx == 5 else ""
        print(
            f"  q{idx + 1}: mean={mean:.12e}, std={std:.12e}, "
            f"min={min_value:.12e}, max={max_value:.12e}{marker}"
        )


def finite_path_mask(values: np.ndarray) -> np.ndarray:
    return np.all(np.isfinite(values), axis=(1, 2))


def metric_bundle(q_candidate: np.ndarray, expert_q: np.ndarray, names: Sequence[str]) -> Dict[str, object]:
    mask = finite_path_mask(q_candidate)
    if not np.any(mask):
        return {
            "q_rmse": float("inf"),
            "max_q_error": float("inf"),
            "path_mean_rmse": float("inf"),
            "per_joint_rmse": np.full(6, np.inf),
            "worst_path": "<none>",
            "num_valid": 0,
        }
    error = q_candidate[mask] - expert_q[mask]
    per_path = np.sqrt(np.mean(np.square(error), axis=(1, 2)))
    valid_names = [name for name, keep in zip(names, mask) if keep]
    return {
        "q_rmse": float(np.sqrt(np.mean(np.square(error)))),
        "max_q_error": float(np.max(np.abs(error))),
        "path_mean_rmse": float(np.mean(per_path)),
        "per_joint_rmse": np.sqrt(np.mean(np.square(error), axis=(0, 1))),
        "worst_path": valid_names[int(np.argmax(per_path))],
        "num_valid": int(np.sum(mask)),
    }


def make_interpretations(
    pred: np.ndarray,
    q_start: np.ndarray,
    train_data: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    train_delta = np.asarray(train_data["delta_q"], dtype=np.float64)
    train_expert = np.asarray(train_data["expert_q"], dtype=np.float64)
    delta_mean = train_delta.mean(axis=(0, 1)).reshape(1, 1, 6)
    delta_std = np.maximum(train_delta.std(axis=(0, 1)).reshape(1, 1, 6), 1e-12)
    q_mean = train_expert.mean(axis=(0, 1)).reshape(1, 1, 6)
    q_std = np.maximum(train_expert.std(axis=(0, 1)).reshape(1, 1, 6), 1e-12)
    return {
        "full_q": pred,
        "raw_delta_q": q_start[:, None, :] + pred,
        "normalized_delta_q": q_start[:, None, :] + pred * delta_std + delta_mean,
        "normalized_full_q": pred * q_std + q_mean,
    }


def print_first_path_debug(
    names: Sequence[str],
    pred: np.ndarray,
    expert_q: np.ndarray,
    q_start: np.ndarray,
    delta_q: np.ndarray,
    pred_dir: Path,
    pred_filename: Path,
) -> None:
    if not names:
        return
    print("\n[first path debug]")
    print(f"  first path_name: {names[0]}")
    print(f"  first expected file: {pred_path(pred_dir, names[0], pred_filename)}")
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


def print_q6_focus(pred: np.ndarray, expert_q: np.ndarray, delta_q: np.ndarray, delta_q_norm: np.ndarray) -> None:
    print("\n[q6 focused diagnostics]")
    for label, values in (
        ("raw pred", pred),
        ("expert_q", expert_q),
        ("delta_q", delta_q),
        ("delta_q_norm", delta_q_norm),
    ):
        q6 = values[..., 5]
        finite = q6[np.isfinite(q6)]
        print(
            f"  {label}: mean={float(np.mean(finite)):.12e}, std={float(np.std(finite)):.12e}, "
            f"min={float(np.min(finite)):.12e}, max={float(np.max(finite)):.12e}"
        )


def find_extra_prediction_files(
    pred_dir: Path,
    pred_filename: Path,
    dataset_names: Sequence[str],
) -> List[Path]:
    expected = {str(pred_path(pred_dir, name, pred_filename).resolve()) for name in dataset_names}
    if not pred_dir.exists():
        return []
    target_name = pred_filename.name
    matches = [path for path in pred_dir.rglob(target_name) if path.is_file()]
    extras = [path for path in matches if str(path.resolve()) not in expected]
    return extras[:10]


def print_path_alignment(
    pred_dir: Path,
    pred_filename: Path,
    names: Sequence[str],
    missing: Sequence[str],
    read_errors: Sequence[str],
) -> None:
    print("\n[path alignment]")
    print("  first 10 dataset path_names and file existence:")
    for name in names[:10]:
        path = pred_path(pred_dir, name, pred_filename)
        print(f"    {name}: {'FOUND' if path.exists() else 'MISSING'} -> {path}")
    print(f"  total missing files: {len(missing)}")
    if missing:
        print("  missing examples: " + ", ".join(str(name) for name in missing[:10]))
    print(f"  read errors: {len(read_errors)}")
    for error in read_errors[:5]:
        print(f"    {error}")
    extras = find_extra_prediction_files(pred_dir, pred_filename, names)
    print(f"  extra prediction files not matched to dataset path_names: {len(extras)} shown")
    for path in extras:
        print(f"    {path}")


def print_sorted_results(results: Dict[str, Dict[str, object]]) -> str:
    ranked = sorted(results.items(), key=lambda item: float(item[1]["q_rmse"]))
    print(
        "\ninterpretation | q_rmse | max_q_error | path_mean_rmse | "
        "q1_rmse | q2_rmse | q3_rmse | q4_rmse | q5_rmse | q6_rmse"
    )
    for name, metrics in ranked:
        per_joint = np.asarray(metrics["per_joint_rmse"], dtype=np.float64)
        print(
            f"{name} | "
            f"{float(metrics['q_rmse']):.12e} | "
            f"{float(metrics['max_q_error']):.12e} | "
            f"{float(metrics['path_mean_rmse']):.12e} | "
            + " | ".join(f"{value:.12e}" for value in per_joint)
        )
        print(f"  valid paths={int(metrics['num_valid'])}, worst path={metrics['worst_path']}")
    return ranked[0][0]


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

    pred, missing, read_errors = load_predictions(args.pred_dir, args.pred_filename, names)
    print(f"Dataset directory: {args.dataset_dir}")
    print(f"Split: {args.split}")
    print(f"Prediction directory: {args.pred_dir}")
    print(f"Prediction filename: {args.pred_filename}")
    print(f"Loaded prediction tensor shape: {pred.shape}")

    print_path_alignment(args.pred_dir, args.pred_filename, names, missing, read_errors)
    print_channel_stats("raw prediction", pred)
    print_channel_stats("expert_q", expert_q)
    print_channel_stats("delta_q", delta_q)
    print_channel_stats("delta_q_norm", delta_q_norm)
    print_q6_focus(pred, expert_q, delta_q, delta_q_norm)
    print_first_path_debug(names, pred, expert_q, q_start, delta_q, args.pred_dir, args.pred_filename)

    interpretations = make_interpretations(pred, q_start, train_data)
    results = {
        name: metric_bundle(candidate, expert_q, names)
        for name, candidate in interpretations.items()
    }
    best = print_sorted_results(results)
    print(f"\nMost likely prediction format: {best}")
    if missing or read_errors:
        print("NOTE: Missing or unreadable files can distort metrics; fix path alignment before trusting rankings.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
