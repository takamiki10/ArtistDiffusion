#!/usr/bin/env python3
"""Diagnose v4 multi-step reverse diffusion rollout from partially noised experts.

This complements diagnose_diffusion_reconstruction_v4_unet.py:
  - one-step diagnostic: predict epsilon at a single t and reconstruct x0
  - this script: start from expert x_t and run reverse DDPM t_start -> 0

No training or sampling files are modified.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from diagnose_diffusion_reconstruction_v4_unet import (
    DEFAULT_DATASET_DIR,
    as_float64,
    beta_schedule,
    get_delta_stats,
    load_model,
    load_npz,
    max_abs,
    parse_timesteps,
    path_names,
    print_joint_std,
    print_keys_and_shapes,
    predict_epsilon,
    reconstruct_q,
    require_keys,
    rmse,
    split_path,
    subset_data,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run v4 reverse DDPM rollout from partially noised expert trajectories."
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="v4 U-Net checkpoint")
    parser.add_argument(
        "--dataset_dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help=f"Directory containing diffusion_train_v2.npz and diffusion_test_v2.npz. Default: {DEFAULT_DATASET_DIR}",
    )
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--start_timesteps", default="10,25,50,75,99")
    parser.add_argument("--max_paths", type=int, default=None)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--num_diffusion_steps", type=int, default=100)
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Do not add random reverse-process noise during rollout.",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def q_metrics(q_pred: np.ndarray, expert_q: np.ndarray, names: Sequence[str]) -> Dict[str, Any]:
    assert q_pred.shape == expert_q.shape, f"q_pred {q_pred.shape} != expert_q {expert_q.shape}"
    error = q_pred - expert_q
    per_path = np.sqrt(np.mean(np.square(error), axis=(1, 2)))
    per_joint = np.sqrt(np.mean(np.square(error), axis=(0, 1)))
    return {
        "q_rmse": rmse(error),
        "max_q_error": max_abs(error),
        "per_joint_q_rmse": per_joint,
        "worst_path": names[int(np.argmax(per_path))],
    }


def print_q_detail(label: str, metrics: Dict[str, Any]) -> None:
    print(
        f"[{label}] q RMSE={metrics['q_rmse']:.12e}, "
        f"max q error={metrics['max_q_error']:.12e}, worst path={metrics['worst_path']}"
    )
    values = metrics["per_joint_q_rmse"]
    print("  per-joint q RMSE: " + "  ".join(f"q{idx + 1}={value:.6e}" for idx, value in enumerate(values)))


def reconstruct_from_norm(
    x_norm: torch.Tensor,
    q_start: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    assert tuple(x_norm.shape[1:]) == (100, 6), f"x_norm must be (B, 100, 6), got {tuple(x_norm.shape)}"
    return reconstruct_q(q_start, x_norm.detach().cpu().numpy().astype(np.float64), mean, std)


def ddpm_reverse_step(
    x_t: torch.Tensor,
    eps_pred: torch.Tensor,
    t_index: int,
    betas: torch.Tensor,
    alphas: torch.Tensor,
    alpha_bars: torch.Tensor,
    deterministic: bool,
) -> torch.Tensor:
    assert x_t.shape == eps_pred.shape, f"x_t {x_t.shape} != eps_pred {eps_pred.shape}"
    beta_t = betas[t_index]
    alpha_t = alphas[t_index]
    alpha_bar_t = alpha_bars[t_index]

    mean = (x_t - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * eps_pred) / torch.sqrt(alpha_t)
    if deterministic or t_index == 0:
        return mean

    alpha_bar_prev = alpha_bars[t_index - 1]
    posterior_var = beta_t * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)
    noise = torch.randn_like(x_t)
    return mean + torch.sqrt(torch.clamp(posterior_var, min=0.0)) * noise


def rollout_from_timestep(
    model: torch.nn.Module,
    x_t_start: torch.Tensor,
    cond: torch.Tensor,
    t_start: int,
    betas: torch.Tensor,
    alphas: torch.Tensor,
    alpha_bars: torch.Tensor,
    deterministic: bool,
    layout: Optional[Tuple[bool, bool]],
) -> Tuple[torch.Tensor, Tuple[bool, bool]]:
    x_t = x_t_start
    assert tuple(x_t.shape[1:]) == (100, 6), f"x_t must be (B, 100, 6), got {tuple(x_t.shape)}"
    assert tuple(cond.shape[1:]) == (100, 13), (
        f"condition_features_norm must be (B, 100, 13), got {tuple(cond.shape)}"
    )

    with torch.no_grad():
        for t_index in range(t_start, -1, -1):
            eps_pred, layout = predict_epsilon(model, x_t, t_index, cond, layout)
            assert eps_pred.shape == x_t.shape, f"model output must be (B, 100, 6), got {tuple(eps_pred.shape)}"
            x_t = ddpm_reverse_step(
                x_t,
                eps_pred,
                t_index,
                betas,
                alphas,
                alpha_bars,
                deterministic,
            )
            assert tuple(x_t.shape[1:]) == (100, 6), f"reverse step produced bad shape {tuple(x_t.shape)}"
    return x_t, layout


def print_summary_table(rows: Sequence[Dict[str, Any]]) -> None:
    print("\nReverse rollout comparison")
    print(
        "t_start | rollout_q_rmse | one_step_q_rmse | no_denoise_q_rmse | "
        "rollout_beats_one_step | rollout_beats_no_denoise"
    )
    for row in rows:
        print(
            f"{row['t_start']:>7d} | "
            f"{row['rollout_q_rmse']:.6e} | "
            f"{row['one_step_q_rmse']:.6e} | "
            f"{row['no_denoise_q_rmse']:.6e} | "
            f"{'yes' if row['rollout_beats_one_step'] else 'no':>22s} | "
            f"{'yes' if row['rollout_beats_no_denoise'] else 'no'}"
        )


def main() -> int:
    args = parse_args()
    set_seed(args.seed)
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[device] CUDA requested but unavailable; using CPU")
        args.device = "cpu"
    device = torch.device(args.device)

    train_data = load_npz(split_path(args.dataset_dir, "train"))
    test_data = load_npz(split_path(args.dataset_dir, "test"))
    print(f"Dataset directory: {args.dataset_dir}")
    print_keys_and_shapes("train", train_data)
    print_keys_and_shapes("test", test_data)

    selected = train_data if args.split == "train" else test_data
    selected = subset_data(selected, args.max_paths)
    print(f"\nUsing split: {args.split}")
    if args.max_paths is not None:
        print(f"Using first {args.max_paths} paths")
    print(f"Reverse rollout mode: {'deterministic' if args.deterministic else 'stochastic'}")
    print(f"Seed: {args.seed}")

    require_keys(
        selected,
        ("condition_features_norm", "delta_q_norm", "delta_q", "expert_q", "q_start", "path_names"),
        "selected split",
    )
    x0_np = as_float64(selected, "delta_q_norm")
    cond_np = as_float64(selected, "condition_features_norm")
    q_start = as_float64(selected, "q_start")
    expert_q = as_float64(selected, "expert_q")
    names = path_names(selected, expert_q.shape[0])

    assert x0_np.shape[1:] == (100, 6), f"delta_q_norm must be (B, 100, 6), got {x0_np.shape}"
    assert cond_np.shape == (x0_np.shape[0], 100, 13), (
        f"condition_features_norm must be (B, 100, 13), got {cond_np.shape}"
    )
    assert expert_q.shape == x0_np.shape, f"expert_q {expert_q.shape} != delta_q_norm {x0_np.shape}"
    assert q_start.shape == (x0_np.shape[0], 6), f"q_start must be (B, 6), got {q_start.shape}"

    mean, std, stats_source = get_delta_stats(selected, train_data)
    print(f"[normalization] stats source: {stats_source}")
    print_joint_std(as_float64(selected, "delta_q"), std)

    start_timesteps = parse_timesteps(args.start_timesteps, args.num_diffusion_steps)
    betas = beta_schedule(args.num_diffusion_steps, device)
    assert betas.shape == (args.num_diffusion_steps,), (
        f"betas must be ({args.num_diffusion_steps},), got {tuple(betas.shape)}"
    )
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)

    model = load_model(
        args.checkpoint,
        device,
        condition_dim=cond_np.shape[-1],
        trajectory_dim=x0_np.shape[-1],
        horizon=x0_np.shape[1],
        num_steps=args.num_diffusion_steps,
    )

    x0 = torch.as_tensor(x0_np, dtype=torch.float32, device=device)
    cond = torch.as_tensor(cond_np, dtype=torch.float32, device=device)
    assert tuple(x0.shape[1:]) == (100, 6), f"x0 must be (B, 100, 6), got {tuple(x0.shape)}"
    assert tuple(cond.shape[1:]) == (100, 13), (
        f"condition_features_norm must be (B, 100, 13), got {tuple(cond.shape)}"
    )

    rows: List[Dict[str, Any]] = []
    layout: Optional[Tuple[bool, bool]] = None

    for t_start in start_timesteps:
        print(f"\n[t_start={t_start}]")
        eps_true = torch.randn_like(x0)
        alpha_bar_t = alpha_bars[t_start]
        sqrt_ab = torch.sqrt(alpha_bar_t)
        sqrt_omab = torch.sqrt(1.0 - alpha_bar_t)
        x_t = sqrt_ab * x0 + sqrt_omab * eps_true
        assert tuple(x_t.shape[1:]) == (100, 6), f"x_t must be (B, 100, 6), got {tuple(x_t.shape)}"

        no_denoise_q = reconstruct_from_norm(x_t, q_start, mean, std)
        no_denoise_metrics = q_metrics(no_denoise_q, expert_q, names)
        print_q_detail(f"no denoising baseline t={t_start}", no_denoise_metrics)

        with torch.no_grad():
            eps_pred, layout = predict_epsilon(model, x_t, t_start, cond, layout)
            assert eps_pred.shape == x_t.shape, f"model output must be (B, 100, 6), got {tuple(eps_pred.shape)}"
            x0_one_step = (x_t - sqrt_omab * eps_pred) / sqrt_ab
        one_step_q = reconstruct_from_norm(x0_one_step, q_start, mean, std)
        one_step_metrics = q_metrics(one_step_q, expert_q, names)
        print_q_detail(f"one-step x0 baseline t={t_start}", one_step_metrics)

        rollout_x0, layout = rollout_from_timestep(
            model,
            x_t,
            cond,
            t_start,
            betas,
            alphas,
            alpha_bars,
            args.deterministic,
            layout,
        )
        rollout_q = reconstruct_from_norm(rollout_x0, q_start, mean, std)
        rollout_metrics = q_metrics(rollout_q, expert_q, names)
        print_q_detail(f"reverse rollout t={t_start}", rollout_metrics)

        rows.append(
            {
                "t_start": t_start,
                "rollout_q_rmse": rollout_metrics["q_rmse"],
                "one_step_q_rmse": one_step_metrics["q_rmse"],
                "no_denoise_q_rmse": no_denoise_metrics["q_rmse"],
                "rollout_beats_one_step": rollout_metrics["q_rmse"] < one_step_metrics["q_rmse"],
                "rollout_beats_no_denoise": rollout_metrics["q_rmse"] < no_denoise_metrics["q_rmse"],
            }
        )

    print_summary_table(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
