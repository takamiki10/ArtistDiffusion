#!/usr/bin/env python3
"""Compare v4 reverse rollout from noised experts versus pure Gaussian starts."""

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
from diagnose_diffusion_v4_reverse_rollout import ddpm_reverse_step


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare pure Gaussian sampling to noised-expert initialization for v4 diffusion."
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="v4 U-Net checkpoint")
    parser.add_argument(
        "--dataset_dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help=f"Directory containing diffusion_train_v2.npz and diffusion_test_v2.npz. Default: {DEFAULT_DATASET_DIR}",
    )
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--start_timesteps", default="99")
    parser.add_argument("--max_paths", type=int, default=None)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--num_diffusion_steps", type=int, default=100)
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Do not add random reverse-process noise.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_samples_per_path", type=int, default=1)
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


def channel_stats(values: np.ndarray) -> List[Tuple[float, float, float, float]]:
    assert values.ndim == 3, f"expected (B, T, C), got {values.shape}"
    out: List[Tuple[float, float, float, float]] = []
    for idx in range(values.shape[-1]):
        channel = values[..., idx]
        out.append(
            (
                float(np.mean(channel)),
                float(np.std(channel)),
                float(np.min(channel)),
                float(np.max(channel)),
            )
        )
    return out


def print_channel_stats(label: str, values: np.ndarray, prefix: str = "q") -> None:
    print(f"[{label}] channel stats mean/std/min/max")
    for idx, (mean, std, min_value, max_value) in enumerate(channel_stats(values)):
        print(
            f"  {prefix}{idx + 1}: mean={mean:.6e}, std={std:.6e}, "
            f"min={min_value:.6e}, max={max_value:.6e}"
        )


def print_comparative_stats(
    start_type: str,
    sample_norm: np.ndarray,
    q_pred: np.ndarray,
    x0_np: np.ndarray,
    expert_q: np.ndarray,
) -> None:
    print_channel_stats(f"{start_type} final sample normalized", sample_norm, prefix="x")
    print_channel_stats(f"{start_type} raw reconstructed q", q_pred, prefix="q")
    print_channel_stats("expert delta_q_norm reference", x0_np, prefix="x")
    print_channel_stats("expert_q reference", expert_q, prefix="q")


def print_result_detail(start_type: str, t_start: int, metrics: Dict[str, Any]) -> None:
    print(
        f"[{start_type} t={t_start}] q RMSE={metrics['q_rmse']:.12e}, "
        f"max q error={metrics['max_q_error']:.12e}, worst path={metrics['worst_path']}"
    )
    print(
        "  per-joint q RMSE: "
        + "  ".join(f"q{idx + 1}={value:.6e}" for idx, value in enumerate(metrics["per_joint_q_rmse"]))
    )


def reconstruct_from_norm(
    x_norm: torch.Tensor,
    q_start: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    assert tuple(x_norm.shape[1:]) == (100, 6), f"trajectory must be (B,100,6), got {tuple(x_norm.shape)}"
    return reconstruct_q(q_start, x_norm.detach().cpu().numpy().astype(np.float64), mean, std)


def reverse_rollout(
    model: torch.nn.Module,
    x_start: torch.Tensor,
    cond: torch.Tensor,
    t_start: int,
    betas: torch.Tensor,
    alphas: torch.Tensor,
    alpha_bars: torch.Tensor,
    deterministic: bool,
    layout: Optional[Tuple[bool, bool]],
) -> Tuple[torch.Tensor, Tuple[bool, bool]]:
    x_t = x_start
    assert tuple(x_t.shape[1:]) == (100, 6), f"trajectory must be (B,100,6), got {tuple(x_t.shape)}"
    assert tuple(cond.shape[1:]) == (100, 13), f"condition must be (B,100,13), got {tuple(cond.shape)}"

    with torch.no_grad():
        for t_index in range(t_start, -1, -1):
            eps_pred, layout = predict_epsilon(model, x_t, t_index, cond, layout)
            assert eps_pred.shape == x_t.shape, f"model output must be (B,100,6), got {tuple(eps_pred.shape)}"
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


def repeat_for_samples(
    array: np.ndarray,
    num_samples_per_path: int,
) -> np.ndarray:
    if num_samples_per_path == 1:
        return array
    return np.repeat(array, num_samples_per_path, axis=0)


def repeated_names(names: Sequence[str], num_samples_per_path: int) -> List[str]:
    if num_samples_per_path == 1:
        return list(names)
    out: List[str] = []
    for name in names:
        for sample_idx in range(num_samples_per_path):
            out.append(f"{name}#sample{sample_idx}")
    return out


def print_summary_table(rows: Sequence[Dict[str, Any]]) -> None:
    print("\nPure sampling versus noised expert summary")
    print("start_type | t_start | q_rmse | max_q_error | q1_rmse | q2_rmse | q3_rmse | q4_rmse | q5_rmse | q6_rmse")
    for row in rows:
        joints = row["per_joint_q_rmse"]
        print(
            f"{row['start_type']} | "
            f"{row['t_start']:>7d} | "
            f"{row['q_rmse']:.6e} | "
            f"{row['max_q_error']:.6e} | "
            + " | ".join(f"{value:.6e}" for value in joints)
        )


def main() -> int:
    args = parse_args()
    if args.num_samples_per_path <= 0:
        raise ValueError("--num_samples_per_path must be positive")
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
    print(f"Samples per path: {args.num_samples_per_path}")

    require_keys(
        selected,
        ("condition_features_norm", "delta_q_norm", "delta_q", "expert_q", "q_start", "desired_paths", "path_names"),
        "selected split",
    )

    base_x0_np = as_float64(selected, "delta_q_norm")
    base_cond_np = as_float64(selected, "condition_features_norm")
    base_q_start = as_float64(selected, "q_start")
    base_expert_q = as_float64(selected, "expert_q")
    base_names = path_names(selected, base_expert_q.shape[0])

    assert base_x0_np.shape[1:] == (100, 6), f"delta_q_norm must be (B,100,6), got {base_x0_np.shape}"
    assert base_cond_np.shape == (base_x0_np.shape[0], 100, 13), (
        f"condition_features_norm must be (B,100,13), got {base_cond_np.shape}"
    )
    assert base_expert_q.shape == base_x0_np.shape, f"expert_q {base_expert_q.shape} != delta_q_norm {base_x0_np.shape}"
    assert base_q_start.shape == (base_x0_np.shape[0], 6), f"q_start must be (B,6), got {base_q_start.shape}"

    mean, std, stats_source = get_delta_stats(selected, train_data)
    print(f"[normalization] stats source: {stats_source}")
    print_joint_std(as_float64(selected, "delta_q"), std)

    x0_np = repeat_for_samples(base_x0_np, args.num_samples_per_path)
    cond_np = repeat_for_samples(base_cond_np, args.num_samples_per_path)
    q_start = repeat_for_samples(base_q_start, args.num_samples_per_path)
    expert_q = repeat_for_samples(base_expert_q, args.num_samples_per_path)
    names = repeated_names(base_names, args.num_samples_per_path)

    assert cond_np.shape == (x0_np.shape[0], 100, 13), f"condition must be (B,100,13), got {cond_np.shape}"
    assert q_start.shape == (x0_np.shape[0], 6), f"q_start must be (B,6), got {q_start.shape}"
    assert expert_q.shape == x0_np.shape, f"expert_q {expert_q.shape} != x0 {x0_np.shape}"

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
    assert tuple(x0.shape[1:]) == (100, 6), f"trajectory must be (B,100,6), got {tuple(x0.shape)}"
    assert tuple(cond.shape[1:]) == (100, 13), f"condition must be (B,100,13), got {tuple(cond.shape)}"

    rows: List[Dict[str, Any]] = []
    layout: Optional[Tuple[bool, bool]] = None

    for t_start in start_timesteps:
        print(f"\n[t_start={t_start}]")
        alpha_bar_t = alpha_bars[t_start]
        sqrt_ab = torch.sqrt(alpha_bar_t)
        sqrt_omab = torch.sqrt(1.0 - alpha_bar_t)

        eps = torch.randn_like(x0)
        noised_expert_start = sqrt_ab * x0 + sqrt_omab * eps
        pure_gaussian_start = torch.randn_like(x0)

        for start_type, x_start in (
            ("noised_expert", noised_expert_start),
            ("pure_gaussian", pure_gaussian_start),
        ):
            print(f"\n[{start_type} t={t_start}]")
            assert tuple(x_start.shape[1:]) == (100, 6), (
                f"{start_type} trajectory must be (B,100,6), got {tuple(x_start.shape)}"
            )
            final_norm, layout = reverse_rollout(
                model,
                x_start,
                cond,
                t_start,
                betas,
                alphas,
                alpha_bars,
                args.deterministic,
                layout,
            )
            assert tuple(final_norm.shape[1:]) == (100, 6), (
                f"final sample trajectory must be (B,100,6), got {tuple(final_norm.shape)}"
            )
            q_pred = reconstruct_from_norm(final_norm, q_start, mean, std)
            metrics = q_metrics(q_pred, expert_q, names)
            print_result_detail(start_type, t_start, metrics)
            final_norm_np = final_norm.detach().cpu().numpy().astype(np.float64)
            print_comparative_stats(start_type, final_norm_np, q_pred, x0_np, expert_q)
            rows.append(
                {
                    "start_type": start_type,
                    "t_start": t_start,
                    "q_rmse": metrics["q_rmse"],
                    "max_q_error": metrics["max_q_error"],
                    "per_joint_q_rmse": metrics["per_joint_q_rmse"],
                }
            )

    print_summary_table(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
