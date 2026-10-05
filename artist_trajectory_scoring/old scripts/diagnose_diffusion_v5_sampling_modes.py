#!/usr/bin/env python3
"""Compare v5 residual diffusion sampling and refinement modes.

This is a window-level diagnostic for the v5 residual U-Net. It does not run
FK, does not train, and does not stitch receding-horizon windows.
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch

from sample_conditional_diffusion_trajectory_v5_residual_unet import (
    DEFAULT_CHECKPOINT,
    DEFAULT_STATS_NPZ,
    DEFAULT_TEST_NPZ,
    diffusion_config_from_checkpoint,
    instantiate_checkpoint_model,
    load_residual_stats,
    make_schedule,
    rmse,
    safe_path_name,
    torch_load_checkpoint,
    validate_test_windows,
)
from train_conditional_diffusion_trajectory_v5_residual_unet import (
    EXPECTED_TARGET_DIM,
    call_model_variant,
)


DEFAULT_OUTPUT_DIR = Path("data/cartesian_expert_dataset_v3/diffusion_v5_residual_unet/sampling_mode_diagnostics")
DEFAULT_T_INIT_VALUES = "5,10,25,50"
DEFAULT_K_VALUES = "4,8,16"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose v5 residual diffusion sampling/refinement modes."
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--test_npz", type=Path, default=DEFAULT_TEST_NPZ)
    parser.add_argument("--stats_npz", type=Path, default=DEFAULT_STATS_NPZ)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num_windows", "--num_samples", dest="num_windows", type=int, default=200)
    parser.add_argument("--num_diffusion_steps", type=int, default=None)
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--t_init_values", type=str, default=DEFAULT_T_INIT_VALUES)
    parser.add_argument("--k_values", type=str, default=DEFAULT_K_VALUES)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument(
        "--example_count",
        type=int,
        default=3,
        help="Save this many best and worst per-window examples. Use 0 to disable.",
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


def load_npz(path: Path, label: str) -> Dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Missing {label}: {path}")
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def parse_int_list(raw: str, label: str) -> List[int]:
    values: List[int] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value < 0:
            raise ValueError(f"{label} values must be non-negative, got {value}")
        values.append(value)
    if not values:
        raise ValueError(f"{label} must contain at least one integer")
    return values


def normalize_residual(values: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((values - mean.reshape(1, 1, EXPECTED_TARGET_DIM)) / std.reshape(1, 1, EXPECTED_TARGET_DIM)).astype(np.float32)


def denormalize_residual(values: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (values * std.reshape(1, 1, EXPECTED_TARGET_DIM) + mean.reshape(1, 1, EXPECTED_TARGET_DIM)).astype(np.float32)


def forward_noise_at_t(
    x0_cf: torch.Tensor,
    t_init: int,
    schedule: Dict[str, torch.Tensor],
) -> torch.Tensor:
    alpha_bar = schedule["alpha_bars"][t_init].reshape(1, 1, 1)
    noise = torch.randn_like(x0_cf)
    return torch.sqrt(alpha_bar) * x0_cf + torch.sqrt(1.0 - alpha_bar) * noise


def reverse_from_initial(
    *,
    model: torch.nn.Module,
    call_variant: str,
    x_init_cf: torch.Tensor,
    condition_cf: torch.Tensor,
    start_step: int,
    schedule: Dict[str, torch.Tensor],
    deterministic: bool,
) -> torch.Tensor:
    x = x_init_cf
    batch_size = x.shape[0]
    device = x.device
    with torch.no_grad():
        for step in reversed(range(start_step + 1)):
            timesteps = torch.full((batch_size,), step, device=device, dtype=torch.long)
            pred_noise = call_model_variant(model, call_variant, x, condition_cf, timesteps)

            beta_t = schedule["betas"][step].reshape(1, 1, 1)
            alpha_t = schedule["alphas"][step].reshape(1, 1, 1)
            alpha_bar_t = schedule["alpha_bars"][step].reshape(1, 1, 1)
            posterior_variance_t = schedule["posterior_variance"][step].reshape(1, 1, 1)

            model_mean = (x - beta_t * pred_noise / torch.sqrt(1.0 - alpha_bar_t)) / torch.sqrt(alpha_t)
            if step > 0 and not deterministic:
                x = model_mean + torch.sqrt(posterior_variance_t) * torch.randn_like(x)
            else:
                x = model_mean
    return x


def reverse_gaussian_batches(
    *,
    model: torch.nn.Module,
    call_variant: str,
    condition_bhc: np.ndarray,
    schedule: Dict[str, torch.Tensor],
    batch_size: int,
    device: torch.device,
    deterministic: bool,
) -> np.ndarray:
    outputs: List[np.ndarray] = []
    num_steps = int(schedule["betas"].shape[0])
    for start in range(0, condition_bhc.shape[0], batch_size):
        condition = torch.from_numpy(condition_bhc[start:start + batch_size]).to(device=device, dtype=torch.float32)
        condition_cf = condition.permute(0, 2, 1).contiguous()
        x_init = torch.randn(condition_cf.shape[0], EXPECTED_TARGET_DIM, condition_cf.shape[-1], device=device)
        sampled = reverse_from_initial(
            model=model,
            call_variant=call_variant,
            x_init_cf=x_init,
            condition_cf=condition_cf,
            start_step=num_steps - 1,
            schedule=schedule,
            deterministic=deterministic,
        )
        outputs.append(sampled.permute(0, 2, 1).detach().cpu().numpy().astype(np.float32))
    return np.concatenate(outputs, axis=0)


def reverse_noised_x0_batches(
    *,
    model: torch.nn.Module,
    call_variant: str,
    condition_bhc: np.ndarray,
    x0_norm_bhc: np.ndarray,
    t_init: int,
    schedule: Dict[str, torch.Tensor],
    batch_size: int,
    device: torch.device,
    deterministic: bool,
) -> np.ndarray:
    outputs: List[np.ndarray] = []
    for start in range(0, condition_bhc.shape[0], batch_size):
        condition = torch.from_numpy(condition_bhc[start:start + batch_size]).to(device=device, dtype=torch.float32)
        x0 = torch.from_numpy(x0_norm_bhc[start:start + batch_size]).to(device=device, dtype=torch.float32)
        condition_cf = condition.permute(0, 2, 1).contiguous()
        x0_cf = x0.permute(0, 2, 1).contiguous()
        x_t = forward_noise_at_t(x0_cf, t_init, schedule)
        sampled = reverse_from_initial(
            model=model,
            call_variant=call_variant,
            x_init_cf=x_t,
            condition_cf=condition_cf,
            start_step=t_init,
            schedule=schedule,
            deterministic=deterministic,
        )
        outputs.append(sampled.permute(0, 2, 1).detach().cpu().numpy().astype(np.float32))
    return np.concatenate(outputs, axis=0)


def velocity_acceleration_metrics(q: np.ndarray) -> Tuple[float, float, float]:
    if q.shape[0] < 2:
        return 0.0, 0.0, 0.0
    velocity = q[1:] - q[:-1]
    velocity_rms = float(np.sqrt(np.mean(np.square(velocity))))
    max_joint_step = float(np.max(np.abs(velocity)))
    if q.shape[0] < 3:
        acceleration_rms = 0.0
    else:
        acceleration = q[2:] - 2.0 * q[1:-1] + q[:-2]
        acceleration_rms = float(np.sqrt(np.mean(np.square(acceleration))))
    return velocity_rms, acceleration_rms, max_joint_step


def metric_row(
    *,
    sample_index: int,
    path_name: str,
    window_start_index: int,
    mode: str,
    t_init: Optional[int],
    k_value: Optional[int],
    prior_q: np.ndarray,
    expert_q: np.ndarray,
    residual_q: np.ndarray,
) -> Tuple[Dict[str, Any], np.ndarray]:
    oracle_residual = expert_q - prior_q
    candidate_q = prior_q + residual_q
    prior_rmse = rmse(prior_q - expert_q)
    candidate_rmse = rmse(candidate_q - expert_q)
    residual_rmse_to_oracle = rmse(residual_q - oracle_residual)
    improvement = 100.0 * (prior_rmse - candidate_rmse) / prior_rmse if prior_rmse > 1e-12 else float("nan")
    velocity_rms, acceleration_rms, max_joint_step = velocity_acceleration_metrics(candidate_q)
    return (
        {
            "sample_index": sample_index,
            "path_name": path_name,
            "window_start_index": int(window_start_index),
            "mode": mode,
            "t_init": "" if t_init is None else int(t_init),
            "K": "" if k_value is None else int(k_value),
            "prior_rmse": prior_rmse,
            "candidate_rmse_to_expert": candidate_rmse,
            "improvement_vs_prior_percent": improvement,
            "residual_rmse_to_oracle": residual_rmse_to_oracle,
            "velocity_rms": velocity_rms,
            "acceleration_rms": acceleration_rms,
            "max_joint_step": max_joint_step,
        },
        candidate_q,
    )


def append_mode_rows(
    *,
    rows: List[Dict[str, Any]],
    examples: List[Dict[str, Any]],
    mode: str,
    t_init: Optional[int],
    k_value: Optional[int],
    residual_norm: np.ndarray,
    residual_q: np.ndarray,
    prior_q: np.ndarray,
    expert_q: np.ndarray,
    desired_path: np.ndarray,
    names: Sequence[str],
    starts: np.ndarray,
) -> None:
    for idx, name in enumerate(names):
        row, candidate_q = metric_row(
            sample_index=idx,
            path_name=name,
            window_start_index=int(starts[idx]),
            mode=mode,
            t_init=t_init,
            k_value=k_value,
            prior_q=prior_q[idx],
            expert_q=expert_q[idx],
            residual_q=residual_q[idx],
        )
        rows.append(row)
        examples.append(
            {
                "row": row,
                "sampled_residual_q_norm": residual_norm[idx],
                "sampled_residual_q": residual_q[idx],
                "prior_q_window": prior_q[idx],
                "expert_q_window": expert_q[idx],
                "q_candidate_window": candidate_q,
                "desired_path_window": desired_path[idx],
            }
        )


def best_of_k_oracle(
    *,
    sampled_norm: np.ndarray,
    residual_mean: np.ndarray,
    residual_std: np.ndarray,
    prior_q: np.ndarray,
    expert_q: np.ndarray,
    k_value: int,
) -> Tuple[np.ndarray, np.ndarray]:
    num_windows, horizon, target_dim = prior_q.shape
    sampled_norm = sampled_norm.reshape(num_windows, k_value, horizon, target_dim)
    sampled_q = denormalize_residual(
        sampled_norm.reshape(num_windows * k_value, horizon, target_dim),
        residual_mean,
        residual_std,
    ).reshape(num_windows, k_value, horizon, target_dim)

    candidate = prior_q[:, None, :, :] + sampled_q
    errors = candidate - expert_q[:, None, :, :]
    candidate_rmse = np.sqrt(np.mean(np.square(errors), axis=(2, 3)))
    best_indices = np.argmin(candidate_rmse, axis=1)
    selected_norm = sampled_norm[np.arange(num_windows), best_indices]
    selected_q = sampled_q[np.arange(num_windows), best_indices]
    return selected_norm.astype(np.float32), selected_q.astype(np.float32)


def write_metric_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    fields = [
        "sample_index",
        "path_name",
        "window_start_index",
        "mode",
        "t_init",
        "K",
        "prior_rmse",
        "candidate_rmse_to_expert",
        "improvement_vs_prior_percent",
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
            out = dict(row)
            for key in (
                "prior_rmse",
                "candidate_rmse_to_expert",
                "improvement_vs_prior_percent",
                "residual_rmse_to_oracle",
                "velocity_rms",
                "acceleration_rms",
                "max_joint_step",
            ):
                out[key] = f"{float(out[key]):.12e}"
            writer.writerow({field: out.get(field, "") for field in fields})


def aggregate_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    for row in rows:
        key = (str(row["mode"]), str(row["t_init"]), str(row["K"]))
        groups.setdefault(key, []).append(row)

    out: List[Dict[str, Any]] = []
    for (mode, t_init, k_value), group in sorted(groups.items()):
        candidate = np.asarray([float(row["candidate_rmse_to_expert"]) for row in group], dtype=np.float64)
        prior = np.asarray([float(row["prior_rmse"]) for row in group], dtype=np.float64)
        improvement = np.asarray([float(row["improvement_vs_prior_percent"]) for row in group], dtype=np.float64)
        residual = np.asarray([float(row["residual_rmse_to_oracle"]) for row in group], dtype=np.float64)
        velocity = np.asarray([float(row["velocity_rms"]) for row in group], dtype=np.float64)
        acceleration = np.asarray([float(row["acceleration_rms"]) for row in group], dtype=np.float64)
        max_step = np.asarray([float(row["max_joint_step"]) for row in group], dtype=np.float64)
        improved = candidate < prior
        out.append(
            {
                "mode": mode,
                "t_init": t_init,
                "K": k_value,
                "count": len(group),
                "mean_prior_rmse": float(np.mean(prior)),
                "mean_candidate_rmse_to_expert": float(np.mean(candidate)),
                "median_candidate_rmse_to_expert": float(np.median(candidate)),
                "mean_improvement_vs_prior_percent": float(np.mean(improvement)),
                "median_improvement_vs_prior_percent": float(np.median(improvement)),
                "improved_count": int(np.sum(improved)),
                "improved_percent": float(100.0 * np.mean(improved)),
                "mean_residual_rmse_to_oracle": float(np.mean(residual)),
                "mean_velocity_rms": float(np.mean(velocity)),
                "mean_acceleration_rms": float(np.mean(acceleration)),
                "mean_max_joint_step": float(np.mean(max_step)),
            }
        )
    return out


def write_aggregate_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    fields = [
        "mode",
        "t_init",
        "K",
        "count",
        "mean_prior_rmse",
        "mean_candidate_rmse_to_expert",
        "median_candidate_rmse_to_expert",
        "mean_improvement_vs_prior_percent",
        "median_improvement_vs_prior_percent",
        "improved_count",
        "improved_percent",
        "mean_residual_rmse_to_oracle",
        "mean_velocity_rms",
        "mean_acceleration_rms",
        "mean_max_joint_step",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            out = dict(row)
            for key in fields:
                if key in {"mode", "t_init", "K", "count", "improved_count"}:
                    continue
                out[key] = f"{float(out[key]):.12e}"
            writer.writerow({field: out.get(field, "") for field in fields})


def write_joint_csv(path: Path, values: np.ndarray, start_index: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["t", "q1", "q2", "q3", "q4", "q5", "q6"])
        for offset, row in enumerate(values):
            writer.writerow([start_index + offset] + [f"{float(value):.10f}" for value in row])


def write_desired_csv(path: Path, values: np.ndarray, start_index: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["t", "x", "y", "z"])
        for offset, row in enumerate(values):
            writer.writerow([start_index + offset] + [f"{float(value):.10f}" for value in row])


def write_example_metadata(path: Path, row: Dict[str, Any], rank_label: str) -> None:
    fields = ["rank_label", *row.keys()]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        out = {"rank_label": rank_label, **row}
        for key, value in list(out.items()):
            if isinstance(value, float):
                out[key] = f"{value:.12e}"
        writer.writerow(out)


def save_examples(output_dir: Path, examples: Sequence[Dict[str, Any]], count: int) -> None:
    if count <= 0 or not examples:
        return
    ranked = sorted(examples, key=lambda item: float(item["row"]["improvement_vs_prior_percent"]))
    selected: List[Tuple[str, Dict[str, Any]]] = []
    selected.extend((f"worst_{idx + 1:02d}", item) for idx, item in enumerate(ranked[:count]))
    selected.extend((f"best_{idx + 1:02d}", item) for idx, item in enumerate(reversed(ranked[-count:])))

    for rank_label, item in selected:
        row = item["row"]
        start = int(row["window_start_index"])
        path_component = safe_path_name(str(row["path_name"]))
        mode_component = str(row["mode"])
        if row["t_init"] != "":
            mode_component += f"_t{row['t_init']}"
        if row["K"] != "":
            mode_component += f"_K{row['K']}"
        example_dir = output_dir / "examples" / f"{rank_label}_{mode_component}_{path_component}_start_{start:03d}"
        write_example_metadata(example_dir / "metrics.csv", row, rank_label)
        write_joint_csv(example_dir / "sampled_residual_q_norm.csv", item["sampled_residual_q_norm"], start)
        write_joint_csv(example_dir / "sampled_residual_q.csv", item["sampled_residual_q"], start)
        write_joint_csv(example_dir / "prior_q_window.csv", item["prior_q_window"], start)
        write_joint_csv(example_dir / "expert_q_window.csv", item["expert_q_window"], start)
        write_joint_csv(example_dir / "q_candidate_window.csv", item["q_candidate_window"], start)
        write_desired_csv(example_dir / "desired_path_window.csv", item["desired_path_window"], start)


def print_top_aggregates(rows: Sequence[Dict[str, Any]]) -> None:
    ranked = sorted(rows, key=lambda row: float(row["mean_candidate_rmse_to_expert"]))
    print("\nTop modes by mean candidate RMSE")
    for row in ranked[:8]:
        label = str(row["mode"])
        if row["t_init"] != "":
            label += f" t={row['t_init']}"
        if row["K"] != "":
            label += f" K={row['K']}"
        print(
            f"  {label}: candidate_rmse={float(row['mean_candidate_rmse_to_expert']):.8e}, "
            f"improvement={float(row['mean_improvement_vs_prior_percent']):.3f}%, "
            f"improved={int(row['improved_count'])}/{int(row['count'])}"
        )


def main() -> int:
    args = parse_args()
    if args.num_windows <= 0:
        raise ValueError("--num_windows must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive")

    set_seed(args.seed)
    device = resolve_device(args.device)
    t_init_values = parse_int_list(args.t_init_values, "--t_init_values")
    k_values = parse_int_list(args.k_values, "--k_values")

    checkpoint = torch_load_checkpoint(args.checkpoint, device)
    model, call_variant, model_config = instantiate_checkpoint_model(checkpoint, device)
    diffusion_config = diffusion_config_from_checkpoint(checkpoint, args.num_diffusion_steps)
    num_steps = int(diffusion_config["num_steps"])
    for t_init in t_init_values:
        if t_init >= num_steps:
            raise ValueError(f"t_init={t_init} must be < num_diffusion_steps={num_steps}")

    residual_mean, residual_std = load_residual_stats(args.stats_npz, checkpoint)
    test_data = load_npz(args.test_npz, "test windows")
    condition_dim = int(checkpoint["condition_dim"])
    target_dim = int(checkpoint["target_dim"])
    horizon = int(checkpoint["horizon"])
    condition, target_norm, prior_q, expert_q, desired_path, names, starts = validate_test_windows(
        test_data,
        expected_condition_dim=condition_dim,
        expected_target_dim=target_dim,
        expected_horizon=horizon,
    )
    if target_dim != EXPECTED_TARGET_DIM or target_norm.shape[-1] != target_dim:
        raise RuntimeError("Unexpected residual target dimensions after validation")

    count = min(args.num_windows, condition.shape[0])
    condition = condition[:count]
    target_norm = target_norm[:count]
    prior_q = prior_q[:count]
    expert_q = expert_q[:count]
    desired_path = desired_path[:count]
    names = names[:count]
    starts = starts[:count]

    schedule = make_schedule(
        num_steps,
        float(diffusion_config["beta_start"]),
        float(diffusion_config["beta_end"]),
        device,
    )
    oracle_residual = expert_q - prior_q
    oracle_residual_norm = normalize_residual(oracle_residual, residual_mean, residual_std)
    zero_residual = np.zeros_like(oracle_residual, dtype=np.float32)
    zero_residual_norm = normalize_residual(zero_residual, residual_mean, residual_std)

    rows: List[Dict[str, Any]] = []
    examples: List[Dict[str, Any]] = []

    append_mode_rows(
        rows=rows,
        examples=examples,
        mode="prior_only",
        t_init=None,
        k_value=None,
        residual_norm=zero_residual_norm,
        residual_q=zero_residual,
        prior_q=prior_q,
        expert_q=expert_q,
        desired_path=desired_path,
        names=names,
        starts=starts,
    )

    pure_norm = reverse_gaussian_batches(
        model=model,
        call_variant=call_variant,
        condition_bhc=condition,
        schedule=schedule,
        batch_size=args.batch_size,
        device=device,
        deterministic=args.deterministic,
    )
    append_mode_rows(
        rows=rows,
        examples=examples,
        mode="pure_gaussian_sample",
        t_init=None,
        k_value=None,
        residual_norm=pure_norm,
        residual_q=denormalize_residual(pure_norm, residual_mean, residual_std),
        prior_q=prior_q,
        expert_q=expert_q,
        desired_path=desired_path,
        names=names,
        starts=starts,
    )

    for t_init in t_init_values:
        sampled_norm = reverse_noised_x0_batches(
            model=model,
            call_variant=call_variant,
            condition_bhc=condition,
            x0_norm_bhc=oracle_residual_norm,
            t_init=t_init,
            schedule=schedule,
            batch_size=args.batch_size,
            device=device,
            deterministic=args.deterministic,
        )
        append_mode_rows(
            rows=rows,
            examples=examples,
            mode="noised_oracle_residual_t",
            t_init=t_init,
            k_value=None,
            residual_norm=sampled_norm,
            residual_q=denormalize_residual(sampled_norm, residual_mean, residual_std),
            prior_q=prior_q,
            expert_q=expert_q,
            desired_path=desired_path,
            names=names,
            starts=starts,
        )

    for t_init in t_init_values:
        sampled_norm = reverse_noised_x0_batches(
            model=model,
            call_variant=call_variant,
            condition_bhc=condition,
            x0_norm_bhc=zero_residual_norm,
            t_init=t_init,
            schedule=schedule,
            batch_size=args.batch_size,
            device=device,
            deterministic=args.deterministic,
        )
        append_mode_rows(
            rows=rows,
            examples=examples,
            mode="zero_residual_refinement_t",
            t_init=t_init,
            k_value=None,
            residual_norm=sampled_norm,
            residual_q=denormalize_residual(sampled_norm, residual_mean, residual_std),
            prior_q=prior_q,
            expert_q=expert_q,
            desired_path=desired_path,
            names=names,
            starts=starts,
        )

    for k_value in k_values:
        repeated_condition = np.repeat(condition, k_value, axis=0)
        sampled_norm_all = reverse_gaussian_batches(
            model=model,
            call_variant=call_variant,
            condition_bhc=repeated_condition,
            schedule=schedule,
            batch_size=args.batch_size,
            device=device,
            deterministic=args.deterministic,
        )
        selected_norm, selected_q = best_of_k_oracle(
            sampled_norm=sampled_norm_all,
            residual_mean=residual_mean,
            residual_std=residual_std,
            prior_q=prior_q,
            expert_q=expert_q,
            k_value=k_value,
        )
        append_mode_rows(
            rows=rows,
            examples=examples,
            mode="multi_sample_best_of_K_oracle",
            t_init=None,
            k_value=k_value,
            residual_norm=selected_norm,
            residual_q=selected_q,
            prior_q=prior_q,
            expert_q=expert_q,
            desired_path=desired_path,
            names=names,
            starts=starts,
        )

    aggregate = aggregate_rows(rows)
    summary_path = args.output_dir / "sampling_modes_summary.csv"
    aggregate_path = args.output_dir / "sampling_modes_aggregate.csv"
    write_metric_csv(summary_path, rows)
    write_aggregate_csv(aggregate_path, aggregate)
    save_examples(args.output_dir, examples, args.example_count)

    print(
        f"Evaluated {count} windows with model={model_config.get('model_class', type(model).__name__)}, "
        f"steps={num_steps}, deterministic={args.deterministic}"
    )
    print(f"Saved per-window summary: {summary_path}")
    print(f"Saved aggregate summary: {aggregate_path}")
    if args.example_count > 0:
        print(f"Saved examples under: {args.output_dir / 'examples'}")
    print_top_aggregates(aggregate)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
