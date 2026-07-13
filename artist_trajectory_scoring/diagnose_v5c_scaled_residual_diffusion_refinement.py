#!/usr/bin/env python3
"""Refine scaled v5c residual predictions with the v5b residual diffusion model.

This window-level diagnostic compares:

    scaled_residual_only:
        r = alpha * r_pred

    scaled_residual_diffusion_refined:
        r_init = alpha * r_pred
        r_init_norm = normalize(r_init)
        r_t = forward_noise(r_init_norm, t_init)
        r_refined_norm = reverse_diffusion(r_t, t_init -> 0)

No training, FK evaluation, or receding-horizon stitching is performed.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from diagnose_diffusion_v5_sampling_modes import reverse_noised_x0_batches
from diagnose_residual_predictor_v5c_alpha_sweep import (
    instantiate_model as instantiate_predictor,
    load_checkpoint as load_predictor_checkpoint,
    load_npz,
    load_residual_stats,
    predict_all,
    resolve_device,
    set_seed,
    validate_test_data,
)
from sample_conditional_diffusion_trajectory_v5_residual_unet import (
    diffusion_config_from_checkpoint,
    instantiate_checkpoint_model,
    make_schedule,
    torch_load_checkpoint,
)


DEFAULT_PREDICTOR_CHECKPOINT = Path(
    "data/cartesian_expert_dataset_v3/"
    "residual_window_predictor_v5c_fk_condition/best_checkpoint.pt"
)
DEFAULT_DIFFUSION_CHECKPOINT = Path(
    "data/cartesian_expert_dataset_v3/"
    "diffusion_v5b_residual_unet_fk_condition/best_checkpoint.pt"
)
DEFAULT_DATASET_DIR = Path(
    "data/cartesian_expert_dataset_v3/"
    "diffusion_v5b_residual_windows_fk_condition"
)
DEFAULT_TEST_NPZ = DEFAULT_DATASET_DIR / "test_windows.npz"
DEFAULT_STATS_NPZ = DEFAULT_DATASET_DIR / "normalization_stats.npz"
DEFAULT_OUTPUT_DIR = Path(
    "data/cartesian_expert_dataset_v3/"
    "diffusion_v5b_residual_unet_fk_condition/"
    "v5c_scaled_residual_refinement_diagnostic"
)
ALPHA_VALUES = (0.05, 0.1, 0.25)
T_INIT_VALUES = (0, 5, 10, 25)
TARGET_DIM = 6
EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refine scaled v5c residual predictions with v5b diffusion."
    )
    parser.add_argument(
        "--predictor_checkpoint",
        type=Path,
        default=DEFAULT_PREDICTOR_CHECKPOINT,
    )
    parser.add_argument(
        "--diffusion_checkpoint",
        type=Path,
        default=DEFAULT_DIFFUSION_CHECKPOINT,
    )
    parser.add_argument("--test_npz", type=Path, default=DEFAULT_TEST_NPZ)
    parser.add_argument("--stats_npz", type=Path, default=DEFAULT_STATS_NPZ)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--num_windows",
        type=int,
        default=200,
        help="Number of leading test windows to evaluate; 0 evaluates all windows.",
    )
    parser.add_argument(
        "--num_diffusion_steps",
        type=int,
        default=None,
        help="Diffusion schedule length; defaults to the checkpoint value.",
    )
    parser.add_argument("--device", choices=("cuda", "cpu", "auto"), default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def validate_checkpoint_compatibility(
    predictor_checkpoint: Dict[str, Any],
    diffusion_checkpoint: Dict[str, Any],
) -> Tuple[int, int, int]:
    required = ("condition_dim", "target_dim", "horizon")
    for label, checkpoint in (
        ("predictor", predictor_checkpoint),
        ("diffusion", diffusion_checkpoint),
    ):
        missing = [key for key in required if key not in checkpoint]
        if missing:
            raise KeyError(
                f"{label} checkpoint missing required key(s): {', '.join(missing)}"
            )

    predictor_shape = tuple(
        int(predictor_checkpoint[key]) for key in required
    )
    diffusion_shape = tuple(
        int(diffusion_checkpoint[key]) for key in required
    )
    if predictor_shape != diffusion_shape:
        raise ValueError(
            "Predictor and diffusion checkpoint dimensions differ: "
            f"{predictor_shape} vs {diffusion_shape}"
        )
    condition_dim, target_dim, horizon = predictor_shape
    if target_dim != TARGET_DIM:
        raise ValueError(
            f"Expected target_dim={TARGET_DIM}, got target_dim={target_dim}"
        )
    return condition_dim, target_dim, horizon


def verify_checkpoint_stats(
    checkpoint: Dict[str, Any],
    residual_mean: np.ndarray,
    residual_std: np.ndarray,
    label: str,
) -> None:
    if "residual_mean" in checkpoint:
        checkpoint_mean = np.asarray(
            checkpoint["residual_mean"],
            dtype=np.float32,
        )
        if checkpoint_mean.shape != residual_mean.shape or not np.allclose(
            checkpoint_mean,
            residual_mean,
            rtol=1e-5,
            atol=1e-7,
        ):
            raise ValueError(
                f"{label} checkpoint residual_mean does not match stats NPZ"
            )
    if "residual_std" in checkpoint:
        checkpoint_std = np.asarray(
            checkpoint["residual_std"],
            dtype=np.float32,
        )
        if checkpoint_std.shape != residual_std.shape or not np.allclose(
            checkpoint_std,
            residual_std,
            rtol=1e-5,
            atol=1e-7,
        ):
            raise ValueError(
                f"{label} checkpoint residual_std does not match stats NPZ"
            )


def normalize_residual(
    residual_q: np.ndarray,
    residual_mean: np.ndarray,
    residual_std: np.ndarray,
) -> np.ndarray:
    return (
        (residual_q - residual_mean.reshape(1, 1, TARGET_DIM))
        / residual_std.reshape(1, 1, TARGET_DIM)
    ).astype(np.float32)


def denormalize_residual(
    residual_q_norm: np.ndarray,
    residual_mean: np.ndarray,
    residual_std: np.ndarray,
) -> np.ndarray:
    return (
        residual_q_norm * residual_std.reshape(1, 1, TARGET_DIM)
        + residual_mean.reshape(1, 1, TARGET_DIM)
    ).astype(np.float32)


def velocity_metrics(
    candidate_q: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    num_windows = candidate_q.shape[0]
    if candidate_q.shape[1] < 2:
        zeros = np.zeros(num_windows, dtype=np.float64)
        return zeros, zeros, zeros

    velocity = candidate_q[:, 1:, :] - candidate_q[:, :-1, :]
    velocity_rms = np.sqrt(np.mean(np.square(velocity), axis=(1, 2)))
    max_joint_step = np.max(np.abs(velocity), axis=(1, 2))
    if candidate_q.shape[1] < 3:
        acceleration_rms = np.zeros(num_windows, dtype=np.float64)
    else:
        acceleration = (
            candidate_q[:, 2:, :]
            - 2.0 * candidate_q[:, 1:-1, :]
            + candidate_q[:, :-2, :]
        )
        acceleration_rms = np.sqrt(
            np.mean(np.square(acceleration), axis=(1, 2))
        )
    return velocity_rms, acceleration_rms, max_joint_step


def append_metric_rows(
    *,
    rows: List[Dict[str, Any]],
    mode: str,
    alpha: float,
    t_init: Optional[int],
    residual_q: np.ndarray,
    prior_q: np.ndarray,
    expert_q: np.ndarray,
    names: Sequence[str],
    starts: np.ndarray,
) -> None:
    oracle_residual = expert_q - prior_q
    candidate_q = prior_q + residual_q
    prior_rmse = np.sqrt(
        np.mean(np.square(prior_q - expert_q), axis=(1, 2))
    )
    candidate_rmse = np.sqrt(
        np.mean(np.square(candidate_q - expert_q), axis=(1, 2))
    )
    residual_rmse = np.sqrt(
        np.mean(np.square(residual_q - oracle_residual), axis=(1, 2))
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
                "mode": mode,
                "alpha": float(alpha),
                "t_init": "" if t_init is None else int(t_init),
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


def aggregate_rows(
    rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, float, str], List[Dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row["mode"]),
            float(row["alpha"]),
            str(row["t_init"]),
        )
        groups.setdefault(key, []).append(row)

    aggregates: List[Dict[str, Any]] = []
    for (mode, alpha, t_init), group in sorted(
        groups.items(),
        key=lambda item: (
            float(item[0][1]),
            0 if item[0][0] == "scaled_residual_only" else 1,
            -1 if item[0][2] == "" else int(item[0][2]),
        ),
    ):
        def values(key: str) -> np.ndarray:
            return np.asarray(
                [float(row[key]) for row in group],
                dtype=np.float64,
            )

        improved_count = int(sum(int(row["improved"]) for row in group))
        count = len(group)
        aggregates.append(
            {
                "mode": mode,
                "alpha": alpha,
                "t_init": t_init,
                "count": count,
                "prior_rmse": float(np.mean(values("prior_rmse"))),
                "candidate_rmse_to_expert": float(
                    np.mean(values("candidate_rmse_to_expert"))
                ),
                "improvement_vs_prior_percent": float(
                    np.mean(values("improvement_vs_prior_percent"))
                ),
                "improved_window_count": improved_count,
                "improved_window_ratio": float(
                    improved_count / max(count, 1)
                ),
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
    return aggregates


def write_summary(
    path: Path,
    rows: Sequence[Dict[str, Any]],
) -> None:
    fields = [
        "sample_index",
        "path_name",
        "window_start_index",
        "mode",
        "alpha",
        "t_init",
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


def write_aggregate(
    path: Path,
    rows: Sequence[Dict[str, Any]],
) -> None:
    fields = [
        "mode",
        "alpha",
        "t_init",
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
    predictor_checkpoint = load_predictor_checkpoint(
        args.predictor_checkpoint,
        device,
    )
    diffusion_checkpoint = torch_load_checkpoint(
        args.diffusion_checkpoint,
        device,
    )
    condition_dim, target_dim, horizon = validate_checkpoint_compatibility(
        predictor_checkpoint,
        diffusion_checkpoint,
    )

    predictor = instantiate_predictor(predictor_checkpoint, device)
    diffusion_model, call_variant, diffusion_model_config = (
        instantiate_checkpoint_model(diffusion_checkpoint, device)
    )
    diffusion_config = diffusion_config_from_checkpoint(
        diffusion_checkpoint,
        args.num_diffusion_steps,
    )
    num_steps = int(diffusion_config["num_steps"])
    for t_init in T_INIT_VALUES:
        if t_init >= num_steps:
            raise ValueError(
                f"t_init={t_init} must be < num_diffusion_steps={num_steps}"
            )

    test_data = load_npz(args.test_npz, "test windows")
    condition, _, prior_q, expert_q, names, starts = validate_test_data(
        test_data,
        predictor_checkpoint,
    )
    if condition.shape[1:] != (horizon, condition_dim):
        raise RuntimeError(
            f"Validated condition shape unexpectedly changed: {condition.shape}"
        )
    if prior_q.shape[-1] != target_dim:
        raise RuntimeError(
            f"Validated target dimension unexpectedly changed: {prior_q.shape}"
        )

    residual_mean, residual_std = load_residual_stats(
        args.stats_npz,
        predictor_checkpoint,
    )
    verify_checkpoint_stats(
        predictor_checkpoint,
        residual_mean,
        residual_std,
        "predictor",
    )
    verify_checkpoint_stats(
        diffusion_checkpoint,
        residual_mean,
        residual_std,
        "diffusion",
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

    predicted_residual_norm = predict_all(
        predictor,
        condition,
        device,
    )
    predicted_residual_q = denormalize_residual(
        predicted_residual_norm,
        residual_mean,
        residual_std,
    )
    schedule = make_schedule(
        num_steps,
        float(diffusion_config["beta_start"]),
        float(diffusion_config["beta_end"]),
        device,
    )

    rows: List[Dict[str, Any]] = []
    for alpha in ALPHA_VALUES:
        scaled_residual_q = (float(alpha) * predicted_residual_q).astype(
            np.float32
        )
        append_metric_rows(
            rows=rows,
            mode="scaled_residual_only",
            alpha=alpha,
            t_init=None,
            residual_q=scaled_residual_q,
            prior_q=prior_q,
            expert_q=expert_q,
            names=names,
            starts=starts,
        )

        scaled_residual_norm = normalize_residual(
            scaled_residual_q,
            residual_mean,
            residual_std,
        )
        for t_init in T_INIT_VALUES:
            refined_residual_norm = reverse_noised_x0_batches(
                model=diffusion_model,
                call_variant=call_variant,
                condition_bhc=condition,
                x0_norm_bhc=scaled_residual_norm,
                t_init=t_init,
                schedule=schedule,
                batch_size=256,
                device=device,
                deterministic=False,
            )
            refined_residual_q = denormalize_residual(
                refined_residual_norm,
                residual_mean,
                residual_std,
            )
            append_metric_rows(
                rows=rows,
                mode="scaled_residual_diffusion_refined",
                alpha=alpha,
                t_init=t_init,
                residual_q=refined_residual_q,
                prior_q=prior_q,
                expert_q=expert_q,
                names=names,
                starts=starts,
            )

    aggregates = aggregate_rows(rows)
    summary_path = (
        args.output_dir / "scaled_residual_refinement_summary.csv"
    )
    aggregate_path = (
        args.output_dir / "scaled_residual_refinement_aggregate.csv"
    )
    write_summary(summary_path, rows)
    write_aggregate(aggregate_path, aggregates)

    predictor_epoch = int(predictor_checkpoint.get("epoch", -1))
    diffusion_epoch = int(diffusion_checkpoint.get("epoch", -1))
    diffusion_name = diffusion_model_config.get(
        "model_class",
        type(diffusion_model).__name__,
    )
    print(
        f"Evaluated {count} windows on {device}; "
        f"predictor_epoch={predictor_epoch}, "
        f"diffusion_epoch={diffusion_epoch}, "
        f"diffusion_model={diffusion_name}, steps={num_steps}"
    )
    for row in aggregates:
        t_label = (
            "-"
            if row["t_init"] == ""
            else str(row["t_init"])
        )
        print(
            f"{row['mode']} | alpha={float(row['alpha']):.3g} | "
            f"t_init={t_label} | "
            f"candidate_rmse={float(row['candidate_rmse_to_expert']):.8e} | "
            f"improvement={float(row['improvement_vs_prior_percent']):.3f}% | "
            f"improved={int(row['improved_window_count'])}/{int(row['count'])}"
        )
    print(f"Saved per-window summary: {summary_path}")
    print(f"Saved aggregate summary: {aggregate_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
