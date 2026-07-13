#!/usr/bin/env python3
"""Diagnose v4 diffusion as a prior-initialized deterministic refiner.

This script compares:
  1. prior trajectory alone
  2. prior + deterministic diffusion refinement
  3. optional pure Gaussian deterministic sampling
  4. optional noised-expert deterministic upper-bound reference

It does not modify training, sampling, or existing evaluation files.
"""

from __future__ import annotations

import argparse
import csv
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


DEFAULT_SUMMARY_CSV = Path(
    "data/cartesian_expert_dataset_v3/diffusion_v4_unet/prior_refinement_diagnostic.csv"
)
DEFAULT_OUTPUT_TRAJECTORY_DIR = Path(
    "data/cartesian_expert_dataset_v3/diffusion_v4_unet/prior_refinement_outputs"
)
PRIOR_FILENAME_FALLBACKS = ("predicted_q.csv", "mlp_pred_q.csv", "q_pred.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test v4 diffusion as a deterministic refiner initialized from rough trajectory priors."
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="v4 U-Net checkpoint")
    parser.add_argument(
        "--dataset_dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help=f"Directory containing diffusion_train_v2.npz and diffusion_test_v2.npz. Default: {DEFAULT_DATASET_DIR}",
    )
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--prior_q_dir", type=Path, default=None)
    parser.add_argument("--prior_q_filename", default="predicted_q.csv")
    parser.add_argument(
        "--prior_format",
        choices=("full_q", "raw_delta_q", "normalized_delta_q", "normalized_full_q", "auto"),
        default="full_q",
        help=(
            "Format of prior CSV values when --prior_q_dir is provided. "
            "Use auto only as a diagnostic because it selects by expert_q RMSE."
        ),
    )
    parser.add_argument("--start_timesteps", default="10,25,50,75")
    parser.add_argument("--max_paths", type=int, default=None)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--num_diffusion_steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_candidates", type=int, default=1)
    parser.add_argument("--prior_noise_scale", type=float, default=0.25)
    parser.add_argument("--include_pure_gaussian", action="store_true")
    parser.add_argument("--include_noised_expert", action="store_true")
    parser.add_argument("--experiment_name", default="prior_refinement")
    parser.add_argument("--output_trajectories_dir", type=Path, default=DEFAULT_OUTPUT_TRAJECTORY_DIR)
    parser.add_argument(
        "--save_outputs",
        action="store_true",
        help="Compatibility flag; candidate trajectories are saved by default.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_name(name: str) -> str:
    return Path(str(name)).stem


def prior_candidates(prior_q_dir: Path, path_name: str, override_filename: str) -> List[Path]:
    name = str(path_name)
    stem = normalize_name(name)
    filenames: List[str] = []
    for filename in (override_filename,) + PRIOR_FILENAME_FALLBACKS:
        if filename not in filenames:
            filenames.append(filename)

    candidates: List[Path] = []
    for path_key in (name, stem):
        for filename in filenames:
            candidates.append(prior_q_dir / path_key / filename)
        candidates.append(prior_q_dir / f"{path_key}.csv")
    return candidates


def read_prior_q_csv(path: Path) -> np.ndarray:
    with path.open("r", newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        try:
            has_header = csv.Sniffer().has_header(sample) if sample.strip() else False
        except csv.Error:
            has_header = False
        if has_header:
            reader = csv.DictReader(handle)
            rows: List[List[float]] = []
            for row in reader:
                lowered = {key.strip().lower(): value for key, value in row.items() if key is not None}
                if all(f"q{idx}" in lowered for idx in range(1, 7)):
                    rows.append([float(lowered[f"q{idx}"]) for idx in range(1, 7)])
                else:
                    numeric = []
                    for key, value in lowered.items():
                        if key == "t" or value == "":
                            continue
                        try:
                            numeric.append(float(value))
                        except ValueError:
                            continue
                    if len(numeric) >= 6:
                        rows.append(numeric[-6:])
            arr = np.asarray(rows, dtype=np.float64)
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
        raise ValueError(f"{path} must contain q trajectory shape (100, 6), got {arr.shape}")
    return arr


def load_prior_pred_batch(
    prior_q_dir: Optional[Path],
    names: Sequence[str],
    expert_q: np.ndarray,
    override_filename: str,
) -> Tuple[np.ndarray, str]:
    if prior_q_dir is None:
        print(
            "[prior] --prior_q_dir was not provided; using synthetic prior "
            "prior_x0 = x0 + prior_noise_scale * noise."
        )
        return expert_q.copy(), "synthetic"

    if not prior_q_dir.exists():
        print(
            f"[prior] prior_q_dir does not exist: {prior_q_dir}; using synthetic prior "
            "prior_x0 = x0 + prior_noise_scale * noise."
        )
        return expert_q.copy(), "synthetic"

    prior_pred = np.zeros_like(expert_q, dtype=np.float64)
    missing: List[str] = []
    for idx, name in enumerate(names):
        found: Optional[Path] = None
        for candidate in prior_candidates(prior_q_dir, name, override_filename):
            if candidate.exists():
                found = candidate
                break
        if found is None:
            missing.append(str(name))
            prior_pred[idx] = expert_q[idx]
            continue
        prior_pred[idx] = read_prior_q_csv(found)

    if missing:
        preview = ", ".join(missing[:5])
        suffix = "..." if len(missing) > 5 else ""
        print(
            f"[prior] missing prior CSV for {len(missing)} path(s); using expert_q fallback for those paths: "
            f"{preview}{suffix}"
        )
    print(f"[prior] loaded CSV priors from {prior_q_dir}")
    return prior_pred, "csv"


def train_expert_q_stats(train_data: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    expert_q = np.asarray(train_data["expert_q"], dtype=np.float64)
    mean = expert_q.mean(axis=(0, 1)).reshape(1, 1, 6)
    std = expert_q.std(axis=(0, 1)).reshape(1, 1, 6)
    std = np.where(np.abs(std) < 1e-12, 1e-12, std)
    return mean, std


def train_delta_q_stats(train_data: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    delta_q = np.asarray(train_data["delta_q"], dtype=np.float64)
    mean = delta_q.mean(axis=(0, 1))
    std = delta_q.std(axis=(0, 1))
    std = np.where(np.abs(std) < 1e-12, 1e-12, std)
    return mean, std


def convert_prior_pred_to_q(
    raw_pred: np.ndarray,
    prior_format: str,
    q_start: np.ndarray,
    expert_q: np.ndarray,
    train_delta_mean: np.ndarray,
    train_delta_std: np.ndarray,
    train_q_mean: Optional[np.ndarray],
    train_q_std: Optional[np.ndarray],
) -> Tuple[np.ndarray, str]:
    delta_mean = np.squeeze(np.asarray(train_delta_mean, dtype=np.float64)).reshape(1, 1, 6)
    delta_std = np.squeeze(np.asarray(train_delta_std, dtype=np.float64)).reshape(1, 1, 6)
    delta_std = np.where(np.abs(delta_std) < 1e-12, 1e-12, delta_std)

    def as_q(fmt: str) -> np.ndarray:
        if fmt == "full_q":
            return raw_pred
        if fmt == "raw_delta_q":
            return q_start[:, None, :] + raw_pred
        if fmt == "normalized_delta_q":
            delta_q = raw_pred * delta_std + delta_mean
            return q_start[:, None, :] + delta_q
        if fmt == "normalized_full_q":
            if train_q_mean is None or train_q_std is None:
                raise ValueError("normalized_full_q requires train expert_q mean/std")
            return raw_pred * train_q_std + train_q_mean
        raise ValueError(f"Unknown prior_format: {fmt}")

    if prior_format != "auto":
        q_prior = as_q(prior_format)
        print(f"[prior] prior_format used: {prior_format}")
        return q_prior, prior_format

    candidates: Dict[str, np.ndarray] = {
        "full_q": as_q("full_q"),
        "raw_delta_q": as_q("raw_delta_q"),
        "normalized_delta_q": as_q("normalized_delta_q"),
        "normalized_full_q": as_q("normalized_full_q"),
    }
    scored: List[Tuple[float, str, np.ndarray]] = []
    print("[prior] prior_format auto diagnostics")
    for fmt, q_candidate in candidates.items():
        error = q_candidate - expert_q
        q_rmse = rmse(error)
        max_error = float(np.max(np.abs(error)))
        per_joint = np.sqrt(np.mean(np.square(error), axis=(0, 1)))
        print(
            f"  {fmt}: q_rmse={q_rmse:.12e}, max_q_error={max_error:.12e}, "
            + "  ".join(f"q{idx + 1}_rmse={value:.6e}" for idx, value in enumerate(per_joint))
        )
        scored.append((q_rmse, fmt, q_candidate))
    scored.sort(key=lambda item: item[0])
    _, chosen_format, q_prior = scored[0]
    print(f"[prior] prior_format used: {chosen_format}  (auto-selected for diagnostics)")
    return q_prior, chosen_format


def repeat_array(array: np.ndarray, num_candidates: int) -> np.ndarray:
    if num_candidates == 1:
        return array
    return np.repeat(array, num_candidates, axis=0)


def repeat_names(names: Sequence[str], num_candidates: int) -> List[str]:
    if num_candidates == 1:
        return list(names)
    out: List[str] = []
    for name in names:
        for candidate_idx in range(num_candidates):
            out.append(f"{name}#candidate{candidate_idx}")
    return out


def normalize_delta(delta_q: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    mean = np.squeeze(np.asarray(mean, dtype=np.float64))
    std = np.squeeze(np.asarray(std, dtype=np.float64))
    safe_std = np.where(np.abs(std) < 1e-12, 1.0, std)
    return (delta_q - mean.reshape(1, 1, -1)) / safe_std.reshape(1, 1, -1)


def q_metric_bundle(q_pred: np.ndarray, expert_q: np.ndarray, names: Sequence[str]) -> Dict[str, Any]:
    assert q_pred.shape == expert_q.shape, f"q_pred {q_pred.shape} != expert_q {expert_q.shape}"
    error = q_pred - expert_q
    path_rmse = np.sqrt(np.mean(np.square(error), axis=(1, 2)))
    return {
        "q_rmse": rmse(error),
        "max_q_error": max_abs(error),
        "per_joint_q_rmse": np.sqrt(np.mean(np.square(error), axis=(0, 1))),
        "path_rmse": path_rmse,
        "worst_path": names[int(np.argmax(path_rmse))],
    }


def print_metric_detail(source: str, t_start: Optional[int], metrics: Dict[str, Any], improved: Optional[int]) -> None:
    t_label = "prior" if t_start is None else str(t_start)
    improved_text = "" if improved is None else f", improved paths={improved}/{metrics['path_rmse'].shape[0]}"
    print(
        f"[{source} t={t_label}] q RMSE={metrics['q_rmse']:.12e}, "
        f"max q error={metrics['max_q_error']:.12e}, worst path={metrics['worst_path']}{improved_text}"
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


def reverse_rollout_deterministic(
    model: torch.nn.Module,
    x_start: torch.Tensor,
    cond: torch.Tensor,
    t_start: int,
    betas: torch.Tensor,
    alphas: torch.Tensor,
    alpha_bars: torch.Tensor,
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
                deterministic=True,
            )
    return x_t, layout


def summary_row(
    source: str,
    t_start: Optional[int],
    metrics: Dict[str, Any],
    improved: Optional[int],
) -> Dict[str, Any]:
    return {
        "source": source,
        "t_start": "" if t_start is None else t_start,
        "q_rmse": metrics["q_rmse"],
        "max_q_error": metrics["max_q_error"],
        "improved_paths": "" if improved is None else improved,
        "num_paths": metrics["path_rmse"].shape[0],
        "q1_rmse": metrics["per_joint_q_rmse"][0],
        "q2_rmse": metrics["per_joint_q_rmse"][1],
        "q3_rmse": metrics["per_joint_q_rmse"][2],
        "q4_rmse": metrics["per_joint_q_rmse"][3],
        "q5_rmse": metrics["per_joint_q_rmse"][4],
        "q6_rmse": metrics["per_joint_q_rmse"][5],
    }


def print_summary(rows: Sequence[Dict[str, Any]]) -> None:
    print("\nPrior-initialized refinement summary")
    print(
        "source | t_start | q_rmse | max_q_error | improved_paths | "
        "q1_rmse | q2_rmse | q3_rmse | q4_rmse | q5_rmse | q6_rmse"
    )
    for row in rows:
        improved = row["improved_paths"]
        improved_text = "" if improved == "" else f"{improved}/{row['num_paths']}"
        print(
            f"{row['source']} | "
            f"{row['t_start']} | "
            f"{row['q_rmse']:.6e} | "
            f"{row['max_q_error']:.6e} | "
            f"{improved_text} | "
            f"{row['q1_rmse']:.6e} | "
            f"{row['q2_rmse']:.6e} | "
            f"{row['q3_rmse']:.6e} | "
            f"{row['q4_rmse']:.6e} | "
            f"{row['q5_rmse']:.6e} | "
            f"{row['q6_rmse']:.6e}"
        )


def save_summary_csv(rows: Sequence[Dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "source",
        "t_start",
        "q_rmse",
        "max_q_error",
        "improved_paths",
        "num_paths",
        "q1_rmse",
        "q2_rmse",
        "q3_rmse",
        "q4_rmse",
        "q5_rmse",
        "q6_rmse",
    )
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})
    print(f"\nSaved CSV summary: {path}")


def safe_path_name(name: str) -> str:
    return Path(str(name)).name.replace("/", "_").replace("\\", "_")


def save_q_csv(path: Path, q: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["t", "q1", "q2", "q3", "q4", "q5", "q6"])
        for timestep, row in enumerate(q):
            writer.writerow([timestep] + [f"{float(value):.10f}" for value in row])


def save_candidate_batch(
    root: Path,
    experiment_name: str,
    source: str,
    t_start: Optional[int],
    names: Sequence[str],
    q_batch: np.ndarray,
) -> None:
    assert q_batch.shape[0] == len(names), f"q_batch paths {q_batch.shape[0]} != names {len(names)}"
    if t_start is None:
        base_dir = root / experiment_name / source
    else:
        base_dir = root / experiment_name / source / f"t_{t_start}"
    for name, q in zip(names, q_batch):
        save_q_csv(base_dir / safe_path_name(name) / "predicted_q.csv", q)
    print(f"[save] wrote {len(names)} {source} trajectories to {base_dir}")


def main() -> int:
    args = parse_args()
    if args.num_candidates <= 0:
        raise ValueError("--num_candidates must be positive")
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
    print(f"Seed: {args.seed}")
    print(f"Candidates per path: {args.num_candidates}")

    require_keys(
        selected,
        ("condition_features_norm", "delta_q_norm", "delta_q", "expert_q", "q_start", "desired_paths", "path_names"),
        "selected split",
    )

    base_x0 = as_float64(selected, "delta_q_norm")
    base_cond = as_float64(selected, "condition_features_norm")
    base_q_start = as_float64(selected, "q_start")
    base_expert_q = as_float64(selected, "expert_q")
    base_names = path_names(selected, base_expert_q.shape[0])

    assert base_x0.shape[1:] == (100, 6), f"delta_q_norm must be (B,100,6), got {base_x0.shape}"
    assert base_cond.shape == (base_x0.shape[0], 100, 13), (
        f"condition_features_norm must be (B,100,13), got {base_cond.shape}"
    )
    assert base_expert_q.shape == base_x0.shape, f"expert_q {base_expert_q.shape} != delta_q_norm {base_x0.shape}"
    assert base_q_start.shape == (base_x0.shape[0], 6), f"q_start must be (B,6), got {base_q_start.shape}"

    mean, std = train_delta_q_stats(train_data)
    print("[normalization] stats source: recomputed from train delta_q")
    print_joint_std(as_float64(selected, "delta_q"), std)

    prior_pred_base, prior_source = load_prior_pred_batch(
        args.prior_q_dir,
        base_names,
        base_expert_q,
        args.prior_q_filename,
    )

    x0_np = repeat_array(base_x0, args.num_candidates)
    cond_np = repeat_array(base_cond, args.num_candidates)
    q_start = repeat_array(base_q_start, args.num_candidates)
    expert_q = repeat_array(base_expert_q, args.num_candidates)
    prior_pred = repeat_array(prior_pred_base, args.num_candidates)
    names = repeat_names(base_names, args.num_candidates)

    assert x0_np.shape[1:] == (100, 6), f"trajectory must be (B,100,6), got {x0_np.shape}"
    assert cond_np.shape == (x0_np.shape[0], 100, 13), f"condition must be (B,100,13), got {cond_np.shape}"
    assert q_start.shape == (x0_np.shape[0], 6), f"q_start must be (B,6), got {q_start.shape}"
    assert expert_q.shape == x0_np.shape, f"expert_q {expert_q.shape} != x0 {x0_np.shape}"
    assert prior_pred.shape == expert_q.shape, f"prior_pred {prior_pred.shape} != expert_q {expert_q.shape}"

    if prior_source == "csv":
        train_q_mean: Optional[np.ndarray] = None
        train_q_std: Optional[np.ndarray] = None
        if args.prior_format in ("normalized_full_q", "auto"):
            train_q_mean, train_q_std = train_expert_q_stats(train_data)
        prior_q, used_prior_format = convert_prior_pred_to_q(
            prior_pred,
            args.prior_format,
            q_start,
            expert_q,
            mean,
            std,
            train_q_mean,
            train_q_std,
        )
        print(f"[prior] raw prior CSV interpreted as: {used_prior_format}")
        prior_delta_q = prior_q - q_start[:, None, :]
        prior_x0_np = normalize_delta(prior_delta_q, mean, std)
    else:
        x0_for_prior = torch.as_tensor(x0_np, dtype=torch.float32)
        prior_x0_np = (
            x0_for_prior + args.prior_noise_scale * torch.randn_like(x0_for_prior)
        ).cpu().numpy().astype(np.float64)
        prior_q = reconstruct_q(
            q_start,
            prior_x0_np.astype(np.float64),
            mean,
            std,
        )
        print(f"[prior] synthetic prior noise scale: {args.prior_noise_scale}")

    assert prior_x0_np.shape == x0_np.shape, f"prior_x0 {prior_x0_np.shape} != x0 {x0_np.shape}"

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
    prior_x0 = torch.as_tensor(prior_x0_np, dtype=torch.float32, device=device)
    cond = torch.as_tensor(cond_np, dtype=torch.float32, device=device)
    assert tuple(x0.shape[1:]) == (100, 6), f"x0 must be (B,100,6), got {tuple(x0.shape)}"
    assert tuple(prior_x0.shape[1:]) == (100, 6), f"prior_x0 must be (B,100,6), got {tuple(prior_x0.shape)}"
    assert tuple(cond.shape[1:]) == (100, 13), f"condition must be (B,100,13), got {tuple(cond.shape)}"

    rows: List[Dict[str, Any]] = []
    prior_metrics = q_metric_bundle(prior_q, expert_q, names)
    print_metric_detail("prior_only", None, prior_metrics, None)
    rows.append(summary_row("prior_only", None, prior_metrics, None))
    save_candidate_batch(
        args.output_trajectories_dir,
        args.experiment_name,
        "prior_only",
        None,
        names,
        prior_q,
    )

    layout: Optional[Tuple[bool, bool]] = None
    for t_start in start_timesteps:
        print(f"\n[t_start={t_start}]")
        alpha_bar_t = alpha_bars[t_start]
        sqrt_ab = torch.sqrt(alpha_bar_t)
        sqrt_omab = torch.sqrt(1.0 - alpha_bar_t)

        eps_prior = torch.randn_like(prior_x0)
        prior_start = sqrt_ab * prior_x0 + sqrt_omab * eps_prior
        refined_norm, layout = reverse_rollout_deterministic(
            model,
            prior_start,
            cond,
            t_start,
            betas,
            alphas,
            alpha_bars,
            layout,
        )
        refined_q = reconstruct_from_norm(refined_norm, q_start, mean, std)
        refined_metrics = q_metric_bundle(refined_q, expert_q, names)
        improved = int(np.sum(refined_metrics["path_rmse"] < prior_metrics["path_rmse"]))
        print_metric_detail("prior_refined", t_start, refined_metrics, improved)
        rows.append(summary_row("prior_refined", t_start, refined_metrics, improved))
        save_candidate_batch(
            args.output_trajectories_dir,
            args.experiment_name,
            "prior_refined",
            t_start,
            names,
            refined_q,
        )

        if args.include_pure_gaussian:
            pure_start = torch.randn_like(x0)
            pure_norm, layout = reverse_rollout_deterministic(
                model,
                pure_start,
                cond,
                t_start,
                betas,
                alphas,
                alpha_bars,
                layout,
            )
            pure_q = reconstruct_from_norm(pure_norm, q_start, mean, std)
            pure_metrics = q_metric_bundle(pure_q, expert_q, names)
            pure_improved = int(np.sum(pure_metrics["path_rmse"] < prior_metrics["path_rmse"]))
            print_metric_detail("pure_gaussian", t_start, pure_metrics, pure_improved)
            rows.append(summary_row("pure_gaussian", t_start, pure_metrics, pure_improved))
            save_candidate_batch(
                args.output_trajectories_dir,
                args.experiment_name,
                "pure_gaussian",
                t_start,
                names,
                pure_q,
            )

        if args.include_noised_expert:
            eps_expert = torch.randn_like(x0)
            noised_expert_start = sqrt_ab * x0 + sqrt_omab * eps_expert
            noised_norm, layout = reverse_rollout_deterministic(
                model,
                noised_expert_start,
                cond,
                t_start,
                betas,
                alphas,
                alpha_bars,
                layout,
            )
            noised_q = reconstruct_from_norm(noised_norm, q_start, mean, std)
            noised_metrics = q_metric_bundle(noised_q, expert_q, names)
            noised_improved = int(np.sum(noised_metrics["path_rmse"] < prior_metrics["path_rmse"]))
            print_metric_detail("noised_expert", t_start, noised_metrics, noised_improved)
            rows.append(summary_row("noised_expert", t_start, noised_metrics, noised_improved))
            save_candidate_batch(
                args.output_trajectories_dir,
                args.experiment_name,
                "noised_expert",
                t_start,
                names,
                noised_q,
            )

    print_summary(rows)
    save_summary_csv(rows, DEFAULT_SUMMARY_CSV)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
