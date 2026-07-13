#!/usr/bin/env python3
"""Sanity checks for v4 diffusion dataset and DDPM reconstruction math.

This script intentionally does not load or use the diffusion model. It verifies
that the dataset deltas, normalization, and DDPM forward/oracle inversion math
are internally consistent before judging Conditional U-Net behavior.
"""

from __future__ import annotations

import argparse
import importlib
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch


DEFAULT_DATASET_DIR = Path("data/cartesian_expert_dataset_v3/diffusion_v2")
STAT_NAME_PAIRS = (
    ("delta_q_mean", "delta_q_std"),
    ("target_mean", "target_std"),
    ("action_mean", "action_std"),
    ("y_mean", "y_std"),
)
SUMMARY_KEYS = (
    "raw_delta_reconstruction",
    "normalized_delta_reconstruction",
    "ddpm_oracle_reconstruction",
    "optional_fk_expert_check",
    "optional_fk_reconstructed_check",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify diffusion v4 dataset reconstruction and DDPM oracle math."
    )
    parser.add_argument(
        "--dataset_dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help=f"Directory containing diffusion_train_v2.npz and diffusion_test_v2.npz. "
        f"Default: {DEFAULT_DATASET_DIR}",
    )
    parser.add_argument(
        "--split",
        choices=("test", "train"),
        default="test",
        help="Dataset split to check. Default: test",
    )
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default="cuda",
        help="Torch device for DDPM oracle math. Falls back to CPU if CUDA is unavailable.",
    )
    parser.add_argument(
        "--timesteps",
        default="0,10,25,50,75,99",
        help="Comma-separated DDPM timesteps to test. Default: 0,10,25,50,75,99",
    )
    parser.add_argument(
        "--max_paths",
        type=int,
        default=None,
        help="Optional maximum number of paths to evaluate.",
    )
    parser.add_argument(
        "--num_diffusion_steps",
        type=int,
        default=100,
        help="Number of DDPM steps in the beta/alpha schedule. Default: 100",
    )
    return parser.parse_args()


def split_path(dataset_dir: Path, split: str) -> Path:
    return dataset_dir / f"diffusion_{split}_v2.npz"


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Missing expected dataset file: {path}")
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def print_keys_and_shapes(label: str, data: Dict[str, np.ndarray]) -> None:
    print(f"\n[{label}] keys and shapes")
    for key in sorted(data.keys()):
        arr = data[key]
        dtype = getattr(arr, "dtype", "unknown")
        print(f"  {key}: shape={arr.shape}, dtype={dtype}")


def subset_data(data: Dict[str, np.ndarray], max_paths: Optional[int]) -> Dict[str, np.ndarray]:
    if max_paths is None:
        return data
    if max_paths <= 0:
        raise ValueError("--max_paths must be positive when provided")
    out: Dict[str, np.ndarray] = {}
    for key, value in data.items():
        if hasattr(value, "shape") and len(value.shape) > 0 and value.shape[0] >= max_paths:
            out[key] = value[:max_paths]
        else:
            out[key] = value
    return out


def require_keys(data: Dict[str, np.ndarray], keys: Sequence[str], check_name: str) -> bool:
    missing = [key for key in keys if key not in data]
    if missing:
        print(f"[{check_name}] FAIL: missing required key(s): {', '.join(missing)}")
        return False
    return True


def as_float64(name: str, data: Dict[str, np.ndarray]) -> np.ndarray:
    return np.asarray(data[name], dtype=np.float64)


def path_names(data: Dict[str, np.ndarray], count: int) -> List[str]:
    if "path_names" not in data:
        return [f"path_{idx}" for idx in range(count)]
    raw_names = np.asarray(data["path_names"])
    names: List[str] = []
    for idx in range(count):
        value = raw_names[idx]
        if isinstance(value, bytes):
            names.append(value.decode("utf-8", errors="replace"))
        else:
            names.append(str(value))
    return names


def rmse(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def per_path_rmse(values: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(np.square(values), axis=tuple(range(1, values.ndim))))


def print_error_stats(
    check_name: str,
    error: np.ndarray,
    names: Sequence[str],
    tolerance: float,
) -> bool:
    mean_rmse = rmse(error)
    max_abs = float(np.max(np.abs(error)))
    worst_idx = int(np.argmax(per_path_rmse(error)))
    passed = bool(np.isfinite(mean_rmse) and np.isfinite(max_abs) and max_abs <= tolerance)
    status = "PASS" if passed else "FAIL"
    print(
        f"[{check_name}] {status}: mean RMSE={mean_rmse:.12e}, "
        f"max abs error={max_abs:.12e}, worst path={names[worst_idx]}"
    )
    return passed


def raw_delta_reconstruction(data: Dict[str, np.ndarray]) -> Tuple[bool, Optional[np.ndarray]]:
    check_name = "raw_delta_reconstruction"
    if not require_keys(data, ("q_start", "delta_q", "expert_q"), check_name):
        print("  likely broke: dataset alignment or q_start + delta_q logic")
        return False, None

    q_start = as_float64("q_start", data)
    delta_q = as_float64("delta_q", data)
    expert_q = as_float64("expert_q", data)
    q_recon = q_start[:, None, :] + delta_q

    if q_recon.shape != expert_q.shape:
        print(f"[{check_name}] FAIL: reconstructed shape {q_recon.shape} != expert_q {expert_q.shape}")
        print("  likely broke: dataset alignment or q_start + delta_q logic")
        return False, q_recon

    passed = print_error_stats(check_name, q_recon - expert_q, path_names(data, expert_q.shape[0]), 1e-8)
    if not passed:
        print("  likely broke: dataset alignment or q_start + delta_q logic")
    return passed, q_recon


def find_stats_in_npz(data: Dict[str, np.ndarray]) -> Optional[Tuple[np.ndarray, np.ndarray, str]]:
    for mean_key, std_key in STAT_NAME_PAIRS:
        if mean_key in data and std_key in data:
            mean = np.asarray(data[mean_key], dtype=np.float64)
            std = np.asarray(data[std_key], dtype=np.float64)
            return mean, std, f"{mean_key}/{std_key}"
    return None


def normalize_stat_shape(stat: np.ndarray, target_ndim: int) -> np.ndarray:
    stat = np.asarray(stat, dtype=np.float64)
    stat = np.squeeze(stat)
    while stat.ndim < target_ndim:
        stat = stat[None, ...]
    return stat


def get_delta_stats(
    selected_data: Dict[str, np.ndarray],
    train_data: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, str]:
    selected_stats = find_stats_in_npz(selected_data)
    if selected_stats is not None:
        return selected_stats

    train_stats = find_stats_in_npz(train_data)
    if train_stats is not None:
        return train_stats

    if "delta_q" not in train_data:
        raise KeyError("Cannot recompute normalization stats because train split is missing delta_q")

    train_delta = as_float64("delta_q", train_data)
    mean = train_delta.mean(axis=(0, 1))
    std = train_delta.std(axis=(0, 1))
    return mean, std, "recomputed from train delta_q"


def unnormalize_delta_q_norm(
    delta_q_norm: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    mean = normalize_stat_shape(mean, delta_q_norm.ndim)
    std = normalize_stat_shape(std, delta_q_norm.ndim)
    safe_std = np.where(np.abs(std) < 1e-12, 1.0, std)
    return delta_q_norm * safe_std + mean


def print_joint_std(delta_q: np.ndarray, std: np.ndarray) -> None:
    recomputed = delta_q.std(axis=(0, 1))
    std_flat = np.squeeze(std)
    print("[normalized_delta_reconstruction] per-joint std")
    for idx, value in enumerate(recomputed):
        stored = std_flat[idx] if idx < std_flat.shape[0] else float("nan")
        marker = "  <-- q6" if idx == 5 else ""
        print(f"  q{idx + 1}: train/stored std={stored:.12e}, selected raw std={value:.12e}{marker}")


def normalized_delta_reconstruction(
    data: Dict[str, np.ndarray],
    train_data: Dict[str, np.ndarray],
) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray], Optional[Tuple[np.ndarray, np.ndarray]]]:
    check_name = "normalized_delta_reconstruction"
    if not require_keys(data, ("q_start", "delta_q", "delta_q_norm", "expert_q"), check_name):
        print("  likely broke: normalization / unnormalization")
        return False, None, None, None

    delta_q = as_float64("delta_q", data)
    delta_q_norm = as_float64("delta_q_norm", data)
    q_start = as_float64("q_start", data)
    expert_q = as_float64("expert_q", data)

    try:
        mean, std, source = get_delta_stats(data, train_data)
    except Exception as exc:
        print(f"[{check_name}] FAIL: could not obtain normalization stats: {exc}")
        print("  likely broke: normalization / unnormalization")
        return False, None, delta_q_norm, None

    print(f"[{check_name}] normalization stats source: {source}")
    print_joint_std(delta_q, std)

    delta_q_recon = unnormalize_delta_q_norm(delta_q_norm, mean, std)
    if delta_q_recon.shape != delta_q.shape:
        print(f"[{check_name}] FAIL: unnormalized shape {delta_q_recon.shape} != delta_q {delta_q.shape}")
        print("  likely broke: normalization / unnormalization")
        return False, delta_q_recon, delta_q_norm, (mean, std)

    names = path_names(data, delta_q.shape[0])
    delta_passed = print_error_stats(f"{check_name}: delta_q_norm_to_delta_q", delta_q_recon - delta_q, names, 1e-6)
    q_recon = q_start[:, None, :] + delta_q_recon

    if q_recon.shape != expert_q.shape:
        print(f"[{check_name}] FAIL: reconstructed shape {q_recon.shape} != expert_q {expert_q.shape}")
        print("  likely broke: normalization / unnormalization")
        return False, q_recon, delta_q_norm, (mean, std)

    q_passed = print_error_stats(f"{check_name}: q_reconstruction", q_recon - expert_q, names, 1e-6)
    passed = delta_passed and q_passed
    if not passed:
        print("  likely broke: normalization / unnormalization")
    return passed, q_recon, delta_q_norm, (mean, std)


def parse_timesteps(raw: str, num_steps: int) -> List[int]:
    timesteps: List[int] = []
    for piece in raw.split(","):
        piece = piece.strip()
        if not piece:
            continue
        timestep = int(piece)
        if timestep < 0 or timestep >= num_steps:
            raise ValueError(f"Requested timestep {timestep} is outside [0, {num_steps - 1}]")
        timesteps.append(timestep)
    if not timesteps:
        raise ValueError("At least one timestep is required")
    return timesteps


def diffusion_beta_schedule(num_steps: int, device: torch.device) -> torch.Tensor:
    """Mirror the common v4 linear DDPM schedule, with import fallback.

    The v4 scripts in this project have historically used a 100-step linear
    schedule. If a helper is exposed by the train/sample scripts, use it;
    otherwise mirror that schedule directly.
    """

    candidate_modules = (
        "train_conditional_diffusion_trajectory_v4_unet",
        "sample_conditional_diffusion_trajectory_v4_unet",
    )
    candidate_fns = (
        "get_beta_schedule",
        "make_beta_schedule",
        "linear_beta_schedule",
        "create_beta_schedule",
    )

    for module_name in candidate_modules:
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        for fn_name in candidate_fns:
            fn = getattr(module, fn_name, None)
            if fn is None:
                continue
            try:
                betas = fn(num_steps)
            except TypeError:
                try:
                    betas = fn(timesteps=num_steps)
                except Exception:
                    continue
            except Exception:
                continue
            betas_tensor = torch.as_tensor(betas, dtype=torch.float32, device=device)
            if betas_tensor.numel() == num_steps:
                print(f"[ddpm_oracle_reconstruction] beta schedule source: {module_name}.{fn_name}")
                return betas_tensor.reshape(num_steps)

    print("[ddpm_oracle_reconstruction] beta schedule source: mirrored linear 1e-4..2e-2")
    return torch.linspace(1e-4, 2e-2, num_steps, dtype=torch.float32, device=device)


def ddpm_oracle_reconstruction(
    data: Dict[str, np.ndarray],
    delta_q_norm: Optional[np.ndarray],
    stats: Optional[Tuple[np.ndarray, np.ndarray]],
    timesteps: Sequence[int],
    num_steps: int,
    device_name: str,
) -> Tuple[bool, Optional[np.ndarray]]:
    check_name = "ddpm_oracle_reconstruction"
    if delta_q_norm is None or stats is None:
        print(f"[{check_name}] FAIL: skipped because normalized reconstruction did not produce inputs")
        print("  likely broke: DDPM schedule or reconstruction formula")
        return False, None
    if not require_keys(data, ("q_start", "expert_q"), check_name):
        print("  likely broke: DDPM schedule or reconstruction formula")
        return False, None

    if device_name == "cuda" and not torch.cuda.is_available():
        print("[ddpm_oracle_reconstruction] CUDA requested but unavailable; using CPU")
        device_name = "cpu"
    device = torch.device(device_name)

    x0 = torch.as_tensor(delta_q_norm, dtype=torch.float32, device=device)
    betas = diffusion_beta_schedule(num_steps, device)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)

    mean, std = stats
    q_start = as_float64("q_start", data)
    expert_q = as_float64("expert_q", data)
    names = path_names(data, expert_q.shape[0])

    all_passed = True
    last_q_recon: Optional[np.ndarray] = None
    with torch.no_grad():
        epsilon = torch.randn_like(x0)
        for timestep in timesteps:
            alpha_bar_t = alpha_bars[timestep]
            sqrt_alpha_bar = torch.sqrt(alpha_bar_t)
            sqrt_one_minus_alpha_bar = torch.sqrt(1.0 - alpha_bar_t)
            x_t = sqrt_alpha_bar * x0 + sqrt_one_minus_alpha_bar * epsilon
            oracle_x0 = (x_t - sqrt_one_minus_alpha_bar * epsilon) / sqrt_alpha_bar

            x0_error = (oracle_x0 - x0).detach().cpu().numpy().astype(np.float64)
            x0_passed = print_error_stats(f"{check_name}: t={timestep} x0", x0_error, names, 5e-6)

            delta_q_recon = unnormalize_delta_q_norm(
                oracle_x0.detach().cpu().numpy().astype(np.float64),
                mean,
                std,
            )
            q_recon = q_start[:, None, :] + delta_q_recon
            last_q_recon = q_recon
            q_error = q_recon - expert_q
            q_passed = print_error_stats(f"{check_name}: t={timestep} q", q_error, names, 5e-5)
            all_passed = all_passed and x0_passed and q_passed

    if not all_passed:
        print("  likely broke: DDPM schedule or reconstruction formula")
    return all_passed, last_q_recon


def import_fk_callable() -> Tuple[Optional[Any], Optional[str]]:
    """Best-effort discovery of existing FK helpers without making FK mandatory."""

    candidates = (
        "score_trajectory",
        "validate_expert_dataset",
    )
    names = (
        "compute_fk_positions",
        "compute_fk_path",
        "forward_kinematics",
        "get_fk_positions",
        "fk_positions",
    )
    for module_name in candidates:
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        for name in names:
            fn = getattr(module, name, None)
            if callable(fn):
                return fn, f"{module_name}.{name}"
    return None, None


def try_call_fk(fn: Any, q: np.ndarray, data: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
    """Try common local FK helper signatures.

    This is intentionally defensive because FK is optional and helper APIs can
    differ across project revisions.
    """

    attempts = (
        (q,),
        (q, data),
        (q, data.get("q_start")),
    )
    for args in attempts:
        try:
            result = fn(*args)
        except Exception:
            continue
        arr = np.asarray(result, dtype=np.float64)
        if arr.ndim >= 2 and arr.shape[-1] >= 3:
            return arr[..., :3]
    return None


def optional_fk_check(
    label: str,
    data: Dict[str, np.ndarray],
    q_values: Optional[np.ndarray],
) -> Optional[bool]:
    summary_key = f"optional_fk_{label}_check"
    if q_values is None:
        print(f"[{summary_key}] SKIP: no reconstructed q available")
        return None
    if "desired_paths" not in data:
        print(f"[{summary_key}] SKIP: desired_paths key not available")
        return None

    fn, source = import_fk_callable()
    if fn is None:
        print(
            f"[{summary_key}] SKIP: no clean FK helper import found in "
            "score_trajectory.py or validate_expert_dataset.py"
        )
        return None

    fk_positions = try_call_fk(fn, q_values, data)
    if fk_positions is None:
        print(f"[{summary_key}] SKIP: FK helper {source} could not be called with known signatures")
        return None

    desired = np.asarray(data["desired_paths"], dtype=np.float64)
    desired = desired[..., :3]
    if fk_positions.shape != desired.shape:
        print(f"[{summary_key}] SKIP: FK shape {fk_positions.shape} != desired_paths shape {desired.shape}")
        return None

    error = np.linalg.norm(fk_positions - desired, axis=-1)
    mean_error = float(np.mean(error))
    max_error = float(np.max(error))
    passed = bool(np.isfinite(mean_error) and np.isfinite(max_error))
    status = "PASS" if passed else "FAIL"
    print(
        f"[{summary_key}] {status}: FK source={source}, "
        f"mean Cartesian error={mean_error:.12e}, max Cartesian error={max_error:.12e}"
    )
    if passed and max_error > 1e-2:
        print("  note: FK numeric reconstruction passes but FK error is bad -> "
              "FK convention, robot config, or desired_path alignment")
    return passed


def print_summary(results: Dict[str, Optional[bool]]) -> None:
    print("\nPASS/FAIL summary")
    for key in SUMMARY_KEYS:
        value = results.get(key)
        if value is True:
            status = "PASS"
        elif value is False:
            status = "FAIL"
        else:
            status = "SKIP"
        print(f"  {key}: {status}")


def main() -> int:
    args = parse_args()
    train_path = split_path(args.dataset_dir, "train")
    test_path = split_path(args.dataset_dir, "test")
    selected_path = split_path(args.dataset_dir, args.split)

    print(f"Dataset directory: {args.dataset_dir}")
    train_data = load_npz(train_path)
    test_data = load_npz(test_path)
    print_keys_and_shapes("train", train_data)
    print_keys_and_shapes("test", test_data)

    selected_data = train_data if args.split == "train" else test_data
    selected_data = subset_data(selected_data, args.max_paths)
    print(f"\nUsing split: {args.split}")
    if args.max_paths is not None:
        print(f"Using first {args.max_paths} paths from selected split")

    timesteps = parse_timesteps(args.timesteps, args.num_diffusion_steps)
    results: Dict[str, Optional[bool]] = {key: None for key in SUMMARY_KEYS}

    raw_passed, raw_q_recon = raw_delta_reconstruction(selected_data)
    results["raw_delta_reconstruction"] = raw_passed

    norm_passed, norm_q_recon, delta_q_norm, stats = normalized_delta_reconstruction(
        selected_data,
        train_data,
    )
    results["normalized_delta_reconstruction"] = norm_passed

    ddpm_passed, ddpm_q_recon = ddpm_oracle_reconstruction(
        selected_data,
        delta_q_norm,
        stats,
        timesteps,
        args.num_diffusion_steps,
        args.device,
    )
    results["ddpm_oracle_reconstruction"] = ddpm_passed

    expert_q = as_float64("expert_q", selected_data) if "expert_q" in selected_data else None
    results["optional_fk_expert_check"] = optional_fk_check("expert", selected_data, expert_q)
    if ddpm_q_recon is not None:
        fk_recon_q = ddpm_q_recon
    elif norm_q_recon is not None:
        fk_recon_q = norm_q_recon
    else:
        fk_recon_q = raw_q_recon
    results["optional_fk_reconstructed_check"] = optional_fk_check(
        "reconstructed",
        selected_data,
        fk_recon_q,
    )

    print_summary(results)

    required_passed = raw_passed and norm_passed and ddpm_passed
    return 0 if required_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
