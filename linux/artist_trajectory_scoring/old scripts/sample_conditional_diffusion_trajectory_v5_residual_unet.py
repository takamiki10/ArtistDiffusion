#!/usr/bin/env python3
"""Sample v5 residual U-Net windows and compare against MLP prior windows.

This is a window-level diagnostic only. It does not perform FK scoring or
receding-horizon stitching.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import random
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from train_conditional_diffusion_trajectory_v5_residual_unet import (
    EXPECTED_TARGET_DIM,
    LocalResidualConditionalUNet1D,
    call_model_variant,
)


DEFAULT_DATASET_DIR = Path("data/cartesian_expert_dataset_v3/diffusion_v5_residual_windows")
DEFAULT_CHECKPOINT = Path("data/cartesian_expert_dataset_v3/diffusion_v5_residual_unet/best_checkpoint.pt")
DEFAULT_TEST_NPZ = DEFAULT_DATASET_DIR / "test_windows.npz"
DEFAULT_STATS_NPZ = DEFAULT_DATASET_DIR / "normalization_stats.npz"
DEFAULT_OUTPUT_DIR = Path("data/cartesian_expert_dataset_v3/diffusion_v5_residual_unet/window_diagnostics")
JOINT_COLUMNS = ("q1", "q2", "q3", "q4", "q5", "q6")
DESIRED_COLUMNS = ("x", "y", "z")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample v5 residual diffusion windows and reconstruct candidate q windows."
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--test_npz", type=Path, default=DEFAULT_TEST_NPZ)
    parser.add_argument("--stats_npz", type=Path, default=DEFAULT_STATS_NPZ)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num_samples", type=int, default=32)
    parser.add_argument(
        "--num_diffusion_steps",
        type=int,
        default=None,
        help="Reverse diffusion steps. Defaults to the checkpoint training value.",
    )
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use posterior means only during reverse diffusion after the initial noise sample.",
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


def torch_load_checkpoint(path: Path, device: torch.device) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {path}")
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint must be a dict, got {type(checkpoint)!r}")
    return checkpoint


def load_npz(path: Path, label: str) -> Dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def require_keys(data: Dict[str, np.ndarray], keys: Sequence[str], label: str) -> None:
    missing = [key for key in keys if key not in data]
    if missing:
        raise KeyError(f"{label} missing required key(s): {', '.join(missing)}")


def decode_names(raw: np.ndarray) -> List[str]:
    names: List[str] = []
    for item in np.asarray(raw):
        names.append(item.decode("utf-8", errors="replace") if isinstance(item, bytes) else str(item))
    return names


def safe_path_name(name: str) -> str:
    return Path(str(name)).name.replace("/", "_").replace("\\", "_")


def validate_test_windows(
    data: Dict[str, np.ndarray],
    expected_condition_dim: int | None = None,
    expected_target_dim: int = EXPECTED_TARGET_DIM,
    expected_horizon: int | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str], np.ndarray]:
    require_keys(
        data,
        (
            "condition_norm",
            "residual_q_norm",
            "prior_q_window",
            "expert_q_window",
            "desired_path_window",
            "path_names",
            "window_start_indices",
        ),
        "test windows",
    )
    condition = np.asarray(data["condition_norm"], dtype=np.float32)
    target = np.asarray(data["residual_q_norm"], dtype=np.float32)
    prior_q = np.asarray(data["prior_q_window"], dtype=np.float32)
    expert_q = np.asarray(data["expert_q_window"], dtype=np.float32)
    desired = np.asarray(data["desired_path_window"], dtype=np.float32)
    names = decode_names(data["path_names"])
    starts = np.asarray(data["window_start_indices"], dtype=np.int64)

    if condition.ndim != 3:
        raise ValueError(f"condition shape must be (N,H,C), got {condition.shape}")
    if expected_condition_dim is not None and condition.shape[-1] != expected_condition_dim:
        raise ValueError(f"condition shape must be (N,H,{expected_condition_dim}), got {condition.shape}")
    if expected_horizon is not None and condition.shape[1] != expected_horizon:
        raise ValueError(f"condition horizon must be {expected_horizon}, got {condition.shape[1]}")
    if target.ndim != 3 or target.shape[-1] != expected_target_dim:
        raise ValueError(f"residual target shape must be (N,H,{expected_target_dim}), got {target.shape}")
    if condition.shape[:2] != target.shape[:2]:
        raise ValueError(f"condition and residual target must share N,H, got {condition.shape[:2]} vs {target.shape[:2]}")
    if prior_q.shape != target.shape:
        raise ValueError(f"prior_q_window shape must match residual target shape, got {prior_q.shape} vs {target.shape}")
    if expert_q.shape != target.shape:
        raise ValueError(f"expert_q_window shape must match residual target shape, got {expert_q.shape} vs {target.shape}")
    if desired.shape != (condition.shape[0], condition.shape[1], len(DESIRED_COLUMNS)):
        raise ValueError(f"desired_path_window shape must be (N,H,3), got {desired.shape}")
    if len(names) != condition.shape[0]:
        raise ValueError(f"path_names length {len(names)} does not match N={condition.shape[0]}")
    if starts.shape != (condition.shape[0],):
        raise ValueError(f"window_start_indices shape must be {(condition.shape[0],)}, got {starts.shape}")
    for label, values in (
        ("condition_norm", condition),
        ("residual_q_norm", target),
        ("prior_q_window", prior_q),
        ("expert_q_window", expert_q),
        ("desired_path_window", desired),
    ):
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{label} contains non-finite values")

    return condition, target, prior_q, expert_q, desired, names, starts


def load_residual_stats(stats_npz: Path, checkpoint: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    stats: Dict[str, np.ndarray] = {}
    if stats_npz.exists():
        stats = load_npz(stats_npz, "normalization stats")
    if "residual_mean" in stats and "residual_std" in stats:
        mean = np.asarray(stats["residual_mean"], dtype=np.float32)
        std = np.asarray(stats["residual_std"], dtype=np.float32)
    elif "residual_mean" in checkpoint and "residual_std" in checkpoint:
        mean = np.asarray(checkpoint["residual_mean"], dtype=np.float32)
        std = np.asarray(checkpoint["residual_std"], dtype=np.float32)
    else:
        raise KeyError(
            f"Missing residual_mean/residual_std in {stats_npz} and checkpoint"
        )
    if mean.shape != (EXPECTED_TARGET_DIM,) or std.shape != (EXPECTED_TARGET_DIM,):
        raise ValueError(
            f"residual stats must have shape ({EXPECTED_TARGET_DIM},), got "
            f"{mean.shape}/{std.shape}"
        )
    if np.any(std <= 0.0):
        raise ValueError("residual_std must be strictly positive")
    return mean, std


def instantiate_checkpoint_model(
    checkpoint: Dict[str, Any],
    device: torch.device,
) -> Tuple[nn.Module, str, Dict[str, Any]]:
    required = ("model_state_dict", "condition_dim", "target_dim", "horizon")
    missing = [key for key in required if key not in checkpoint]
    if missing:
        raise KeyError(f"Checkpoint missing required key(s): {', '.join(missing)}")

    condition_dim = int(checkpoint["condition_dim"])
    target_dim = int(checkpoint["target_dim"])
    horizon = int(checkpoint["horizon"])
    model_config = dict(checkpoint.get("model_config", {}))
    init_kwargs = dict(model_config.get("init_kwargs", {}))
    model_class_name = str(model_config.get("model_class", ""))
    model_source = str(model_config.get("model_source", ""))

    if model_source == "local" or model_class_name == "LocalResidualConditionalUNet1D":
        init_kwargs.setdefault("condition_dim", condition_dim)
        init_kwargs.setdefault("target_dim", target_dim)
        init_kwargs.setdefault("horizon", horizon)
        init_kwargs.setdefault("base_channels", int(init_kwargs.get("base_channels", 64)))
        model: nn.Module = LocalResidualConditionalUNet1D(**init_kwargs)
    else:
        module_name = str(model_config.get("model_module", ""))
        if not module_name:
            raise KeyError("Checkpoint model_config missing model_module for project model")
        module = importlib.import_module(module_name)
        cls = getattr(module, model_class_name)
        model = cls(**init_kwargs)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    call_variant = str(model_config.get("call_variant", "cf_x_cond_t"))
    return model, call_variant, model_config


def diffusion_config_from_checkpoint(
    checkpoint: Dict[str, Any],
    num_steps_override: int | None,
) -> Dict[str, Any]:
    diffusion_config = dict(checkpoint.get("diffusion_config", {}))
    checkpoint_steps = int(
        checkpoint.get(
            "num_diffusion_steps",
            diffusion_config.get("num_diffusion_steps", 1000),
        )
    )
    num_steps = checkpoint_steps if num_steps_override is None else int(num_steps_override)
    if num_steps <= 0:
        raise ValueError("--num_diffusion_steps must be positive")
    if num_steps != checkpoint_steps:
        print(
            f"[diffusion] using {num_steps} reverse steps; checkpoint was trained with {checkpoint_steps}"
        )
    return {
        "num_steps": num_steps,
        "beta_start": float(diffusion_config.get("beta_start", 1e-4)),
        "beta_end": float(diffusion_config.get("beta_end", 2e-2)),
    }


def make_schedule(num_steps: int, beta_start: float, beta_end: float, device: torch.device) -> Dict[str, torch.Tensor]:
    betas = torch.linspace(beta_start, beta_end, num_steps, dtype=torch.float32, device=device)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    alpha_bars_prev = torch.cat([torch.ones(1, device=device), alpha_bars[:-1]], dim=0)
    posterior_variance = betas * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars)
    posterior_variance = torch.clamp(posterior_variance, min=1e-20)
    return {
        "betas": betas,
        "alphas": alphas,
        "alpha_bars": alpha_bars,
        "posterior_variance": posterior_variance,
    }


def extract(values: torch.Tensor, timesteps: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    gathered = values.gather(0, timesteps)
    return gathered.reshape(timesteps.shape[0], *([1] * (target.ndim - 1)))


def sample_residual_norm_windows(
    *,
    model: nn.Module,
    call_variant: str,
    condition_bhc: np.ndarray,
    target_dim: int,
    num_steps: int,
    beta_start: float,
    beta_end: float,
    device: torch.device,
    deterministic: bool,
) -> np.ndarray:
    condition = torch.from_numpy(condition_bhc).to(device=device, dtype=torch.float32)
    condition_cf = condition.permute(0, 2, 1).contiguous()
    batch_size, horizon, _ = condition_bhc.shape
    x = torch.randn(batch_size, target_dim, horizon, device=device)
    schedule = make_schedule(num_steps, beta_start, beta_end, device)

    with torch.no_grad():
        for step in reversed(range(num_steps)):
            timesteps = torch.full((batch_size,), step, device=device, dtype=torch.long)
            pred_noise = call_model_variant(model, call_variant, x, condition_cf, timesteps)

            beta_t = extract(schedule["betas"], timesteps, x)
            alpha_t = extract(schedule["alphas"], timesteps, x)
            alpha_bar_t = extract(schedule["alpha_bars"], timesteps, x)
            posterior_variance_t = extract(schedule["posterior_variance"], timesteps, x)

            model_mean = (x - beta_t * pred_noise / torch.sqrt(1.0 - alpha_bar_t)) / torch.sqrt(alpha_t)
            if step > 0 and not deterministic:
                x = model_mean + torch.sqrt(posterior_variance_t) * torch.randn_like(x)
            else:
                x = model_mean

    return x.permute(0, 2, 1).detach().cpu().numpy().astype(np.float32)


def rmse(error: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(error))))


def write_joint_csv(path: Path, values: np.ndarray) -> None:
    if values.ndim != 2 or values.shape[1] != len(JOINT_COLUMNS):
        raise ValueError(f"{path.name} values must have shape (H,6), got {values.shape}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["t", *JOINT_COLUMNS])
        for idx, row in enumerate(values):
            writer.writerow([idx] + [f"{float(value):.10f}" for value in row])


def write_desired_csv(path: Path, values: np.ndarray) -> None:
    if values.ndim != 2 or values.shape[1] != len(DESIRED_COLUMNS):
        raise ValueError(f"{path.name} values must have shape (H,3), got {values.shape}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["t", *DESIRED_COLUMNS])
        for idx, row in enumerate(values):
            writer.writerow([idx] + [f"{float(value):.10f}" for value in row])


def write_summary(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    fields = [
        "sample_index",
        "path_name",
        "window_start_index",
        "prior_rmse",
        "sampled_residual_rmse",
        "candidate_rmse_to_expert",
        "improvement_vs_prior_percent",
        "oracle_candidate_rmse_to_expert",
        "window_dir",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def export_samples(
    *,
    output_dir: Path,
    sampled_residual_norm: np.ndarray,
    sampled_residual: np.ndarray,
    prior_q: np.ndarray,
    expert_q: np.ndarray,
    desired_path: np.ndarray,
    names: Sequence[str],
    starts: np.ndarray,
) -> None:
    rows: List[Dict[str, Any]] = []
    for idx, name in enumerate(names):
        start = int(starts[idx])
        window_dir = output_dir / f"window_{idx:04d}_{safe_path_name(name)}_start_{start:03d}"

        candidate_q = prior_q[idx] + sampled_residual[idx]
        oracle_residual = expert_q[idx] - prior_q[idx]
        oracle_candidate = prior_q[idx] + oracle_residual

        prior_rmse = rmse(prior_q[idx] - expert_q[idx])
        sampled_residual_rmse = rmse(sampled_residual[idx] - oracle_residual)
        candidate_rmse = rmse(candidate_q - expert_q[idx])
        oracle_rmse = rmse(oracle_candidate - expert_q[idx])
        improvement = (
            100.0 * (prior_rmse - candidate_rmse) / prior_rmse
            if prior_rmse > 1e-12
            else float("nan")
        )

        write_joint_csv(window_dir / "sampled_residual_q_norm.csv", sampled_residual_norm[idx])
        write_joint_csv(window_dir / "sampled_residual_q.csv", sampled_residual[idx])
        write_joint_csv(window_dir / "prior_q_window.csv", prior_q[idx])
        write_joint_csv(window_dir / "expert_q_window.csv", expert_q[idx])
        write_joint_csv(window_dir / "q_candidate_window.csv", candidate_q)
        write_desired_csv(window_dir / "desired_path_window.csv", desired_path[idx])

        rows.append(
            {
                "sample_index": idx,
                "path_name": name,
                "window_start_index": start,
                "prior_rmse": f"{prior_rmse:.12e}",
                "sampled_residual_rmse": f"{sampled_residual_rmse:.12e}",
                "candidate_rmse_to_expert": f"{candidate_rmse:.12e}",
                "improvement_vs_prior_percent": f"{improvement:.12e}",
                "oracle_candidate_rmse_to_expert": f"{oracle_rmse:.12e}",
                "window_dir": str(window_dir),
            }
        )

    write_summary(output_dir / "sample_summary.csv", rows)


def main() -> int:
    args = parse_args()
    if args.num_samples <= 0:
        raise ValueError("--num_samples must be positive")

    set_seed(args.seed)
    device = resolve_device(args.device)
    checkpoint = torch_load_checkpoint(args.checkpoint, device)
    model, call_variant, model_config = instantiate_checkpoint_model(checkpoint, device)
    diffusion_config = diffusion_config_from_checkpoint(checkpoint, args.num_diffusion_steps)
    residual_mean, residual_std = load_residual_stats(args.stats_npz, checkpoint)

    test_data = load_npz(args.test_npz, "test windows")
    condition_dim = int(checkpoint["condition_dim"])
    target_dim = int(checkpoint["target_dim"])
    horizon = int(checkpoint["horizon"])
    condition, target, prior_q, expert_q, desired_path, names, starts = validate_test_windows(
        test_data,
        expected_condition_dim=condition_dim,
        expected_target_dim=target_dim,
        expected_horizon=horizon,
    )

    if target_dim != EXPECTED_TARGET_DIM or target.shape[-1] != target_dim:
        raise ValueError(f"checkpoint/test target_dim mismatch: {target_dim} vs {target.shape[-1]}")
    if horizon != target.shape[1]:
        raise ValueError(f"checkpoint/test horizon mismatch: {horizon} vs {target.shape[1]}")

    sample_count = min(args.num_samples, condition.shape[0])
    if sample_count < args.num_samples:
        print(f"[samples] requested {args.num_samples}, but test set has {condition.shape[0]}; using {sample_count}")

    sampled_residual_norm = sample_residual_norm_windows(
        model=model,
        call_variant=call_variant,
        condition_bhc=condition[:sample_count],
        target_dim=target_dim,
        num_steps=int(diffusion_config["num_steps"]),
        beta_start=float(diffusion_config["beta_start"]),
        beta_end=float(diffusion_config["beta_end"]),
        device=device,
        deterministic=args.deterministic,
    )
    sampled_residual = (
        sampled_residual_norm * residual_std.reshape(1, 1, target_dim)
        + residual_mean.reshape(1, 1, target_dim)
    ).astype(np.float32)

    export_samples(
        output_dir=args.output_dir,
        sampled_residual_norm=sampled_residual_norm,
        sampled_residual=sampled_residual,
        prior_q=prior_q[:sample_count],
        expert_q=expert_q[:sample_count],
        desired_path=desired_path[:sample_count],
        names=names[:sample_count],
        starts=starts[:sample_count],
    )

    candidate_q = prior_q[:sample_count] + sampled_residual
    prior_rmse = rmse(prior_q[:sample_count] - expert_q[:sample_count])
    candidate_rmse = rmse(candidate_q - expert_q[:sample_count])
    improvement = 100.0 * (prior_rmse - candidate_rmse) / prior_rmse if prior_rmse > 1e-12 else float("nan")
    oracle_rmse = rmse((prior_q[:sample_count] + (expert_q[:sample_count] - prior_q[:sample_count])) - expert_q[:sample_count])

    print(
        f"Sampled {sample_count} v5 residual windows with {diffusion_config['num_steps']} DDPM steps "
        f"(deterministic={args.deterministic})"
    )
    print(f"Model: {model_config.get('model_class', type(model).__name__)} | call_variant={call_variant}")
    print(
        f"prior_rmse={prior_rmse:.8e} | candidate_rmse={candidate_rmse:.8e} | "
        f"improvement={improvement:.3f}% | oracle_rmse={oracle_rmse:.8e}"
    )
    print(f"Saved window diagnostics: {args.output_dir}")
    print(f"Saved summary CSV: {args.output_dir / 'sample_summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
