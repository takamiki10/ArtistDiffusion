#!/usr/bin/env python3
"""Audit residual branch compatibility between adaptive priors and experts.

This script is diagnostic only. It never modifies the frozen prior or expert
datasets and does not construct residual-window training data.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXPECTED_STEPS = 100
EXPECTED_JOINTS = 6
EXPECTED_CARTESIAN_DIM = 3
EXPECTED_COUNTS = {"train": 418, "test": 83}
JOINT_LABELS = tuple(f"q{index}" for index in range(1, EXPECTED_JOINTS + 1))

PATH_NAME_ALIASES = ("path_names", "trajectory_names")
PRIOR_Q_ALIASES = ("prior_q", "adaptive_prior_q")
EXPERT_Q_ALIASES = ("expert_q", "expert_joint_trajectories")
DESIRED_PATH_ALIASES = ("desired_paths", "desired_cartesian_paths")
PRIOR_EE_ALIASES = ("prior_ee", "prior_end_effector_positions")
SUCCESS_ALIASES = ("generation_success", "prior_generation_success")
JOINT_NAME_ALIASES = ("joint_names", "joint_ordering")

RESIDUAL_MODES = ("raw", "wrapped", "unwrapped", "aligned")
METRIC_SUFFIXES = (
    "mean_rad",
    "mean_abs_rad",
    "rmse_rad",
    "median_abs_rad",
    "p90_abs_rad",
    "p95_abs_rad",
    "p99_abs_rad",
    "max_abs_rad",
    "rms_residual_step_rad",
    "max_residual_step_rad",
    "fraction_above_0_25_rad",
    "fraction_above_0_50_rad",
    "fraction_above_1_00_rad",
    "fraction_above_pi",
)


class AuditValidationError(ValueError):
    """Raised when a source artifact is unsafe to compare."""


@dataclass(frozen=True)
class LoadedPrior:
    source: Path
    keys: Tuple[str, ...]
    names: Tuple[str, ...]
    q: np.ndarray
    desired: np.ndarray
    ee: Optional[np.ndarray]
    generation_success: np.ndarray
    joint_names: Optional[Tuple[str, ...]]


@dataclass(frozen=True)
class LoadedExpert:
    source: Path
    keys: Tuple[str, ...]
    names: Tuple[str, ...]
    q: np.ndarray
    desired: np.ndarray
    joint_names: Optional[Tuple[str, ...]]


@dataclass(frozen=True)
class AlignedSplit:
    split: str
    names: Tuple[str, ...]
    prior_q: np.ndarray
    expert_q: np.ndarray
    desired: np.ndarray
    prior_ee: Optional[np.ndarray]
    generation_success: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit whether Adaptive MLP+IK priors and expert trajectories use "
            "compatible joint-space branches."
        )
    )
    parser.add_argument("--train_prior", type=Path, required=True)
    parser.add_argument("--test_prior", type=Path, required=True)
    parser.add_argument("--train_expert", type=Path, required=True)
    parser.add_argument("--test_expert", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--local_rmse_threshold", type=float, default=0.50)
    parser.add_argument("--branch_rmse_threshold", type=float, default=1.00)
    parser.add_argument(
        "--cartesian_accuracy_threshold", type=float, default=0.01
    )
    parser.add_argument("--top_k", type=int, default=20)
    return parser.parse_args()


def validate_cli(args: argparse.Namespace) -> None:
    if args.local_rmse_threshold <= 0.0:
        raise ValueError("--local_rmse_threshold must be positive")
    if args.branch_rmse_threshold <= 0.0:
        raise ValueError("--branch_rmse_threshold must be positive")
    if args.cartesian_accuracy_threshold <= 0.0:
        raise ValueError("--cartesian_accuracy_threshold must be positive")
    if args.top_k <= 0:
        raise ValueError("--top_k must be positive")


def decode_strings(values: np.ndarray) -> Tuple[str, ...]:
    decoded: List[str] = []
    for value in np.asarray(values).reshape(-1):
        if isinstance(value, bytes):
            decoded.append(value.decode("utf-8", errors="strict"))
        else:
            decoded.append(str(value))
    return tuple(decoded)


def load_named_array(
    archive: Any,
    aliases: Sequence[str],
    semantic_name: str,
    *,
    required: bool,
) -> Optional[np.ndarray]:
    present = [alias for alias in aliases if alias in archive]
    if not present:
        if required:
            raise AuditValidationError(
                f"Missing {semantic_name}; expected one of {list(aliases)}"
            )
        return None
    selected = np.asarray(archive[present[0]])
    for alias in present[1:]:
        alternative = np.asarray(archive[alias])
        if selected.shape != alternative.shape or not np.array_equal(
            selected, alternative
        ):
            raise AuditValidationError(
                f"Ambiguous {semantic_name}: keys {present} contain different arrays"
            )
    return selected


def load_prior(path: Path) -> LoadedPrior:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=True) as archive:
        keys = tuple(archive.files)
        print(f"{path}: discovered keys={list(keys)}")
        names_raw = load_named_array(
            archive, PATH_NAME_ALIASES, "prior path names", required=True
        )
        q_raw = load_named_array(
            archive, PRIOR_Q_ALIASES, "prior joint trajectories", required=True
        )
        desired_raw = load_named_array(
            archive, DESIRED_PATH_ALIASES, "prior desired paths", required=True
        )
        ee_raw = load_named_array(
            archive, PRIOR_EE_ALIASES, "prior end-effector paths", required=False
        )
        success_raw = load_named_array(
            archive, SUCCESS_ALIASES, "prior generation success", required=False
        )
        joint_names_raw = load_named_array(
            archive, JOINT_NAME_ALIASES, "prior joint ordering", required=False
        )
        assert names_raw is not None and q_raw is not None and desired_raw is not None
        names = decode_strings(names_raw)
        success = (
            np.ones(len(names), dtype=bool)
            if success_raw is None
            else np.asarray(success_raw, dtype=bool).reshape(-1)
        )
        joint_names = (
            None if joint_names_raw is None else decode_strings(joint_names_raw)
        )
        return LoadedPrior(
            source=path,
            keys=keys,
            names=names,
            q=np.asarray(q_raw, dtype=np.float64),
            desired=np.asarray(desired_raw, dtype=np.float64),
            ee=None if ee_raw is None else np.asarray(ee_raw, dtype=np.float64),
            generation_success=success,
            joint_names=joint_names,
        )


def load_expert(path: Path) -> LoadedExpert:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=True) as archive:
        keys = tuple(archive.files)
        print(f"{path}: discovered keys={list(keys)}")
        names_raw = load_named_array(
            archive, PATH_NAME_ALIASES, "expert path names", required=True
        )
        q_raw = load_named_array(
            archive, EXPERT_Q_ALIASES, "expert joint trajectories", required=True
        )
        desired_raw = load_named_array(
            archive, DESIRED_PATH_ALIASES, "expert desired paths", required=True
        )
        joint_names_raw = load_named_array(
            archive, JOINT_NAME_ALIASES, "expert joint ordering", required=False
        )
        assert names_raw is not None and q_raw is not None and desired_raw is not None
        return LoadedExpert(
            source=path,
            keys=keys,
            names=decode_strings(names_raw),
            q=np.asarray(q_raw, dtype=np.float64),
            desired=np.asarray(desired_raw, dtype=np.float64),
            joint_names=(
                None
                if joint_names_raw is None
                else decode_strings(joint_names_raw)
            ),
        )


def require_unique_names(names: Sequence[str], label: str) -> None:
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise AuditValidationError(f"{label} contains duplicate names: {duplicates}")


def validate_joint_array(array: np.ndarray, count: int, label: str) -> None:
    expected = (count, EXPECTED_STEPS, EXPECTED_JOINTS)
    if array.shape != expected:
        raise AuditValidationError(f"{label} shape {array.shape}, expected {expected}")
    if not np.all(np.isfinite(array)):
        raise AuditValidationError(f"{label} contains nonfinite values")
    maximum_absolute = float(np.max(np.abs(array)))
    if maximum_absolute > 8.0 * np.pi:
        raise AuditValidationError(
            f"{label} does not appear to be radians: max abs={maximum_absolute:.6g}"
        )


def validate_cartesian_array(array: np.ndarray, count: int, label: str) -> None:
    expected = (count, EXPECTED_STEPS, EXPECTED_CARTESIAN_DIM)
    if array.shape != expected:
        raise AuditValidationError(f"{label} shape {array.shape}, expected {expected}")
    if not np.all(np.isfinite(array)):
        raise AuditValidationError(f"{label} contains nonfinite values")


def normalize_joint_names(names: Sequence[str]) -> Tuple[str, ...]:
    normalized: List[str] = []
    for name in names:
        compact = str(name).lower().replace("_", "").replace("-", "")
        if compact.startswith("joint"):
            compact = "q" + compact[len("joint") :]
        normalized.append(compact)
    return tuple(normalized)


def validate_joint_order(
    prior_names: Optional[Sequence[str]],
    expert_names: Optional[Sequence[str]],
    split: str,
) -> str:
    expected = normalize_joint_names(JOINT_LABELS)
    if prior_names is not None and normalize_joint_names(prior_names) != expected:
        raise AuditValidationError(
            f"{split} prior joint ordering {list(prior_names)} is not q1..q6"
        )
    if expert_names is not None and normalize_joint_names(expert_names) != expected:
        raise AuditValidationError(
            f"{split} expert joint ordering {list(expert_names)} is not q1..q6"
        )
    if prior_names is not None and expert_names is not None:
        if normalize_joint_names(prior_names) != normalize_joint_names(expert_names):
            raise AuditValidationError(f"{split} prior/expert joint ordering differs")
        return "explicit_prior_and_expert_joint_names_match_q1_through_q6"
    return "validated_by_repository_Nx100x6_q1_through_q6_schema"


def align_and_validate_split(
    split: str, prior: LoadedPrior, expert: LoadedExpert
) -> Tuple[AlignedSplit, Dict[str, Any]]:
    expected_count = EXPECTED_COUNTS[split]
    if len(prior.names) != expected_count:
        raise AuditValidationError(
            f"{split} prior has {len(prior.names)} paths, expected {expected_count}"
        )
    if len(expert.names) != expected_count:
        raise AuditValidationError(
            f"{split} expert has {len(expert.names)} paths, expected {expected_count}"
        )
    require_unique_names(prior.names, f"{split} prior")
    require_unique_names(expert.names, f"{split} expert")
    prior_set = set(prior.names)
    expert_set = set(expert.names)
    if prior_set != expert_set:
        raise AuditValidationError(
            f"{split} path-name sets differ; "
            f"prior_only={sorted(prior_set - expert_set)}, "
            f"expert_only={sorted(expert_set - prior_set)}"
        )
    validate_joint_array(prior.q, expected_count, f"{split} prior_q")
    validate_joint_array(expert.q, expected_count, f"{split} expert_q")
    validate_cartesian_array(prior.desired, expected_count, f"{split} prior desired")
    validate_cartesian_array(expert.desired, expected_count, f"{split} expert desired")
    if prior.ee is not None:
        validate_cartesian_array(prior.ee, expected_count, f"{split} prior_ee")
    if prior.generation_success.shape != (expected_count,):
        raise AuditValidationError(
            f"{split} generation_success shape {prior.generation_success.shape}, "
            f"expected ({expected_count},)"
        )
    joint_order_validation = validate_joint_order(
        prior.joint_names, expert.joint_names, split
    )

    prior_index = {name: index for index, name in enumerate(prior.names)}
    order = np.asarray([prior_index[name] for name in expert.names], dtype=np.int64)
    aligned_prior_q = prior.q[order]
    aligned_prior_desired = prior.desired[order]
    aligned_prior_ee = None if prior.ee is None else prior.ee[order]
    aligned_success = prior.generation_success[order]
    desired_difference = float(
        np.max(np.abs(aligned_prior_desired - expert.desired))
    )
    if not np.allclose(
        aligned_prior_desired, expert.desired, rtol=1.0e-6, atol=1.0e-7
    ):
        raise AuditValidationError(
            f"{split} desired paths differ after path-name alignment; "
            f"max abs difference={desired_difference:.8g}"
        )

    validation = {
        "expected_path_count": expected_count,
        "matched_path_count": len(expert.names),
        "unique_path_names": True,
        "path_name_sets_match": True,
        "source_orders_match": list(prior.names) == list(expert.names),
        "joint_shape": list(expert.q.shape),
        "cartesian_shape": list(expert.desired.shape),
        "all_required_arrays_finite": True,
        "joint_units_validation": "radian_range_heuristic_passed",
        "joint_order_validation": joint_order_validation,
        "maximum_desired_path_difference_after_alignment": desired_difference,
        "prior_ee_available": prior.ee is not None,
        "generation_success_available": any(
            alias in prior.keys for alias in SUCCESS_ALIASES
        ),
    }
    return (
        AlignedSplit(
            split=split,
            names=expert.names,
            prior_q=aligned_prior_q,
            expert_q=expert.q,
            desired=expert.desired,
            prior_ee=aligned_prior_ee,
            generation_success=aligned_success,
        ),
        validation,
    )


def continuity_preserving_alignment(
    prior_q: np.ndarray, expert_q: np.ndarray
) -> np.ndarray:
    aligned = np.empty_like(expert_q, dtype=np.float64)
    two_pi = 2.0 * np.pi
    for joint_index in range(expert_q.shape[1]):
        previous: Optional[float] = None
        for timestep in range(expert_q.shape[0]):
            expert_value = float(expert_q[timestep, joint_index])
            prior_value = float(prior_q[timestep, joint_index])
            prior_turn = int(np.rint((prior_value - expert_value) / two_pi))
            turn_candidates = set(range(prior_turn - 2, prior_turn + 3))
            if previous is not None:
                continuity_turn = int(np.rint((previous - expert_value) / two_pi))
                turn_candidates.update(
                    range(continuity_turn - 2, continuity_turn + 3)
                )
            equivalent_values = [
                expert_value + two_pi * turn for turn in sorted(turn_candidates)
            ]
            if previous is None:
                selected = min(
                    equivalent_values, key=lambda value: abs(value - prior_value)
                )
            else:
                selected = min(
                    equivalent_values,
                    key=lambda value: (
                        (value - prior_value) ** 2 + (value - previous) ** 2,
                        abs(value - prior_value),
                    ),
                )
            aligned[timestep, joint_index] = selected
            previous = selected
    return aligned


def residual_metrics(residual: np.ndarray) -> Dict[str, float]:
    residual_array = np.asarray(residual, dtype=np.float64)
    flat = residual_array.reshape(-1)
    absolute = np.abs(flat)
    time_axis = 1 if residual_array.ndim >= 3 else 0
    changes = np.diff(residual_array, axis=time_axis).reshape(-1)
    return {
        "mean_rad": float(np.mean(flat)),
        "mean_abs_rad": float(np.mean(absolute)),
        "rmse_rad": float(np.sqrt(np.mean(np.square(flat)))),
        "median_abs_rad": float(np.median(absolute)),
        "p90_abs_rad": float(np.percentile(absolute, 90.0)),
        "p95_abs_rad": float(np.percentile(absolute, 95.0)),
        "p99_abs_rad": float(np.percentile(absolute, 99.0)),
        "max_abs_rad": float(np.max(absolute)),
        "rms_residual_step_rad": (
            float(np.sqrt(np.mean(np.square(changes)))) if changes.size else 0.0
        ),
        "max_residual_step_rad": (
            float(np.max(np.abs(changes))) if changes.size else 0.0
        ),
        "fraction_above_0_25_rad": float(np.mean(absolute > 0.25)),
        "fraction_above_0_50_rad": float(np.mean(absolute > 0.50)),
        "fraction_above_1_00_rad": float(np.mean(absolute > 1.00)),
        "fraction_above_pi": float(np.mean(absolute > np.pi)),
    }


def maximum_joint_step(q: np.ndarray) -> float:
    delta = np.diff(q, axis=0)
    return float(np.max(np.abs(delta))) if delta.size else 0.0


def safe_reduction(original: float, reduced: float) -> float:
    if original <= np.finfo(np.float64).eps:
        return 0.0 if reduced <= np.finfo(np.float64).eps else float("-inf")
    return float((original - reduced) / original)


def classify_path(
    *,
    generation_success: bool,
    prior_cartesian_mean_error: float,
    raw_rmse: float,
    wrapped_rmse: float,
    aligned_rmse: float,
    local_threshold: float,
    branch_threshold: float,
    cartesian_threshold: float,
) -> str:
    if not generation_success or (
        np.isfinite(prior_cartesian_mean_error)
        and prior_cartesian_mean_error > cartesian_threshold
    ):
        return "prior_generation_failure"
    if min(wrapped_rmse, aligned_rmse) <= local_threshold:
        return "local_residual_candidate"
    reduction = max(
        safe_reduction(raw_rmse, wrapped_rmse),
        safe_reduction(raw_rmse, aligned_rmse),
    )
    if raw_rmse > branch_threshold and reduction >= 0.70:
        return "likely_two_pi_representation_offset"
    if wrapped_rmse > branch_threshold and aligned_rmse > branch_threshold:
        return "likely_ik_branch_mismatch"
    return "requires_manual_review"


def audit_split(
    data: AlignedSplit,
    args: argparse.Namespace,
) -> Tuple[List[Dict[str, Any]], Dict[str, np.ndarray], List[np.ndarray]]:
    rows: List[Dict[str, Any]] = []
    residual_collections: Dict[str, List[np.ndarray]] = {
        mode: [] for mode in RESIDUAL_MODES
    }
    aligned_experts: List[np.ndarray] = []
    for index, path_name in enumerate(data.names):
        prior_q = data.prior_q[index]
        expert_q = data.expert_q[index]
        aligned_expert_q = continuity_preserving_alignment(prior_q, expert_q)
        aligned_experts.append(aligned_expert_q)
        residuals = {
            "raw": expert_q - prior_q,
            "wrapped": (expert_q - prior_q + np.pi) % (2.0 * np.pi) - np.pi,
            "unwrapped": np.unwrap(expert_q, axis=0)
            - np.unwrap(prior_q, axis=0),
            "aligned": aligned_expert_q - prior_q,
        }
        for mode, residual in residuals.items():
            residual_collections[mode].append(residual)
        metrics_by_mode = {
            mode: residual_metrics(residual) for mode, residual in residuals.items()
        }
        prior_cartesian_error = (
            float("nan")
            if data.prior_ee is None
            else float(
                np.mean(
                    np.linalg.norm(data.prior_ee[index] - data.desired[index], axis=1)
                )
            )
        )
        raw_rmse = metrics_by_mode["raw"]["rmse_rad"]
        wrapped_rmse = metrics_by_mode["wrapped"]["rmse_rad"]
        aligned_rmse = metrics_by_mode["aligned"]["rmse_rad"]
        row: Dict[str, Any] = {
            "split": data.split,
            "path_name": path_name,
            "generation_success": bool(data.generation_success[index]),
            "prior_cartesian_mean_error_m": prior_cartesian_error,
            "prior_max_joint_step_rad": maximum_joint_step(prior_q),
            "expert_max_joint_step_rad": maximum_joint_step(expert_q),
            "aligned_expert_max_joint_step_rad": maximum_joint_step(
                aligned_expert_q
            ),
            "raw_to_wrapped_rmse_reduction_fraction": safe_reduction(
                raw_rmse, wrapped_rmse
            ),
            "raw_to_aligned_rmse_reduction_fraction": safe_reduction(
                raw_rmse, aligned_rmse
            ),
        }
        for mode, metrics in metrics_by_mode.items():
            row.update({f"{mode}_{key}": value for key, value in metrics.items()})
        row["diagnostic_classification"] = classify_path(
            generation_success=bool(data.generation_success[index]),
            prior_cartesian_mean_error=prior_cartesian_error,
            raw_rmse=raw_rmse,
            wrapped_rmse=wrapped_rmse,
            aligned_rmse=aligned_rmse,
            local_threshold=args.local_rmse_threshold,
            branch_threshold=args.branch_rmse_threshold,
            cartesian_threshold=args.cartesian_accuracy_threshold,
        )
        rows.append(row)
    stacked = {
        mode: np.stack(residual_collections[mode]) for mode in RESIDUAL_MODES
    }
    return rows, stacked, aligned_experts


def aggregate_rows(
    residuals_by_split: Mapping[str, Mapping[str, np.ndarray]],
    per_path: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    aggregate_records: List[Dict[str, Any]] = []
    joint_records: List[Dict[str, Any]] = []
    split_arrays: Dict[str, Dict[str, np.ndarray]] = {
        split: dict(values) for split, values in residuals_by_split.items()
    }
    split_arrays["combined"] = {
        mode: np.concatenate(
            [residuals_by_split["train"][mode], residuals_by_split["test"][mode]],
            axis=0,
        )
        for mode in RESIDUAL_MODES
    }
    for split, mode_arrays in split_arrays.items():
        path_frame = (
            per_path
            if split == "combined"
            else per_path[per_path["split"] == split]
        )
        for mode, residual in mode_arrays.items():
            metrics = residual_metrics(residual)
            path_rmse = path_frame[f"{mode}_rmse_rad"].to_numpy(dtype=float)
            aggregate_records.append(
                {
                    "split": split,
                    "residual_mode": mode,
                    "path_count": int(residual.shape[0]),
                    **metrics,
                    "mean_path_rmse_rad": float(np.mean(path_rmse)),
                    "median_path_rmse_rad": float(np.median(path_rmse)),
                    "p95_path_rmse_rad": float(np.percentile(path_rmse, 95.0)),
                }
            )
            for joint_index, joint_name in enumerate(JOINT_LABELS):
                joint_metrics = residual_metrics(
                    residual[:, :, joint_index : joint_index + 1]
                )
                joint_records.append(
                    {
                        "split": split,
                        "joint_name": joint_name,
                        "joint_index": joint_index,
                        "residual_mode": mode,
                        "path_count": int(residual.shape[0]),
                        **joint_metrics,
                    }
                )
    return pd.DataFrame(joint_records), pd.DataFrame(aggregate_records)


def save_worst_paths(
    frame: pd.DataFrame, mode: str, top_k: int, output_dir: Path
) -> None:
    ordered = frame.sort_values(f"{mode}_rmse_rad", ascending=False).head(top_k)
    ordered.to_csv(output_dir / f"worst_paths_{mode}_residual.csv", index=False)


def save_histogram(values: np.ndarray, title: str, path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 5))
    axis.hist(values.reshape(-1), bins=100)
    axis.set_title(title)
    axis.set_xlabel("Residual (rad)")
    axis.set_ylabel("Count")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def save_plots(
    per_path: pd.DataFrame,
    combined_residuals: Mapping[str, np.ndarray],
    output_dir: Path,
) -> None:
    save_histogram(
        combined_residuals["raw"],
        "Raw residual distribution",
        output_dir / "raw_residual_histogram.png",
    )
    save_histogram(
        combined_residuals["wrapped"],
        "Wrapped residual distribution",
        output_dir / "wrapped_residual_histogram.png",
    )
    save_histogram(
        combined_residuals["aligned"],
        "Aligned residual distribution",
        output_dir / "aligned_residual_histogram.png",
    )

    figure, axis = plt.subplots(figsize=(9, 5))
    axis.boxplot(
        [
            combined_residuals["wrapped"][:, :, joint].reshape(-1)
            for joint in range(EXPECTED_JOINTS)
        ],
        labels=JOINT_LABELS,
        showfliers=False,
    )
    axis.set_title("Wrapped residuals by joint")
    axis.set_ylabel("Residual (rad)")
    figure.tight_layout()
    figure.savefig(output_dir / "per_joint_wrapped_residual_boxplot.png", dpi=150)
    plt.close(figure)

    scatter_specs = (
        (
            "raw_rmse_rad",
            "wrapped_rmse_rad",
            "Raw RMSE (rad)",
            "Wrapped RMSE (rad)",
            "raw_vs_wrapped_rmse_scatter.png",
        ),
        (
            "wrapped_rmse_rad",
            "aligned_rmse_rad",
            "Wrapped RMSE (rad)",
            "Aligned RMSE (rad)",
            "wrapped_vs_aligned_rmse_scatter.png",
        ),
        (
            "wrapped_rmse_rad",
            "prior_cartesian_mean_error_m",
            "Wrapped RMSE (rad)",
            "Prior Cartesian mean error (m)",
            "wrapped_rmse_vs_cartesian_error_scatter.png",
        ),
    )
    for x_column, y_column, x_label, y_label, filename in scatter_specs:
        figure, axis = plt.subplots(figsize=(7, 6))
        axis.scatter(per_path[x_column], per_path[y_column], alpha=0.7)
        axis.set_xlabel(x_label)
        axis.set_ylabel(y_label)
        figure.tight_layout()
        figure.savefig(output_dir / filename, dpi=150)
        plt.close(figure)

    counts = per_path["diagnostic_classification"].value_counts().sort_index()
    figure, axis = plt.subplots(figsize=(10, 5))
    axis.bar(counts.index, counts.values)
    axis.set_ylabel("Path count")
    axis.tick_params(axis="x", rotation=25)
    figure.tight_layout()
    figure.savefig(output_dir / "diagnostic_classification_counts.png", dpi=150)
    plt.close(figure)


def classification_counts(frame: pd.DataFrame) -> Dict[str, Dict[str, int]]:
    result: Dict[str, Dict[str, int]] = {}
    for split in ("train", "test", "combined"):
        selected = frame if split == "combined" else frame[frame["split"] == split]
        counts = selected["diagnostic_classification"].value_counts()
        result[split] = {str(key): int(value) for key, value in counts.items()}
    return result


def choose_recommendation(
    frame: pd.DataFrame, local_rmse_threshold: float
) -> str:
    usable = frame[
        frame["diagnostic_classification"] != "prior_generation_failure"
    ]
    if usable.empty:
        return "audit_failed_validation"
    counts = usable["diagnostic_classification"].value_counts()
    offset_count = int(counts.get("likely_two_pi_representation_offset", 0))
    branch_count = int(counts.get("likely_ik_branch_mismatch", 0))
    manual_count = int(counts.get("requires_manual_review", 0))
    raw_local_fraction = float(
        np.mean(usable["raw_rmse_rad"] <= local_rmse_threshold)
    )
    if branch_count > 0 and (offset_count > 0 or manual_count > 0):
        return "mixed_result_requires_path_level_handling"
    if branch_count > 0:
        return "generate_branch_aligned_expert_targets"
    if offset_count > 0:
        return "use_two_pi_aligned_expert_representation"
    if raw_local_fraction >= 0.95 and manual_count == 0:
        return "raw_residuals_are_local"
    return "mixed_result_requires_path_level_handling"


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main() -> int:
    args = parse_args()
    validate_cli(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    prior_by_split = {
        "train": load_prior(args.train_prior),
        "test": load_prior(args.test_prior),
    }
    expert_by_split = {
        "train": load_expert(args.train_expert),
        "test": load_expert(args.test_expert),
    }
    aligned_by_split: Dict[str, AlignedSplit] = {}
    validations: Dict[str, Dict[str, Any]] = {}
    for split in ("train", "test"):
        aligned, validation = align_and_validate_split(
            split, prior_by_split[split], expert_by_split[split]
        )
        aligned_by_split[split] = aligned
        validations[split] = validation
        print(f"{split}: matched paths={len(aligned.names)}")

    path_rows: List[Dict[str, Any]] = []
    residuals_by_split: Dict[str, Dict[str, np.ndarray]] = {}
    for split in ("train", "test"):
        rows, residuals, aligned_experts = audit_split(
            aligned_by_split[split], args
        )
        path_rows.extend(rows)
        residuals_by_split[split] = residuals
        np.savez_compressed(
            args.output_dir / f"aligned_expert_q_{split}.npz",
            path_names=np.asarray(aligned_by_split[split].names),
            aligned_expert_q=np.stack(aligned_experts).astype(np.float32),
        )

    per_path = pd.DataFrame(path_rows)
    per_joint, aggregate = aggregate_rows(residuals_by_split, per_path)
    per_path.to_csv(args.output_dir / "residual_audit_per_path.csv", index=False)
    per_joint.to_csv(args.output_dir / "residual_audit_per_joint.csv", index=False)
    aggregate.to_csv(args.output_dir / "residual_audit_aggregate.csv", index=False)
    for mode in ("raw", "wrapped", "aligned"):
        save_worst_paths(per_path, mode, args.top_k, args.output_dir)

    combined_residuals = {
        mode: np.concatenate(
            [residuals_by_split["train"][mode], residuals_by_split["test"][mode]],
            axis=0,
        )
        for mode in RESIDUAL_MODES
    }
    save_plots(per_path, combined_residuals, args.output_dir)

    counts = classification_counts(per_path)
    recommendation = choose_recommendation(
        per_path, args.local_rmse_threshold
    )
    worst_wrapped = (
        per_path.sort_values("wrapped_rmse_rad", ascending=False)
        .head(10)[["split", "path_name", "wrapped_rmse_rad"]]
        .to_dict(orient="records")
    )
    worst_aligned = (
        per_path.sort_values("aligned_rmse_rad", ascending=False)
        .head(10)[["split", "path_name", "aligned_rmse_rad"]]
        .to_dict(orient="records")
    )
    aggregate_records = aggregate.to_dict(orient="records")
    summary = {
        "input_files": {
            "train_prior": args.train_prior,
            "test_prior": args.test_prior,
            "train_expert": args.train_expert,
            "test_expert": args.test_expert,
        },
        "discovered_npz_keys": {
            "train_prior": prior_by_split["train"].keys,
            "test_prior": prior_by_split["test"].keys,
            "train_expert": expert_by_split["train"].keys,
            "test_expert": expert_by_split["test"].keys,
        },
        "validation_results": validations,
        "path_counts": {
            "train": len(aligned_by_split["train"].names),
            "test": len(aligned_by_split["test"].names),
            "combined": len(per_path),
        },
        "classification_counts": counts,
        "aggregate_residual_statistics": aggregate_records,
        "thresholds": {
            "local_rmse_threshold": args.local_rmse_threshold,
            "branch_rmse_threshold": args.branch_rmse_threshold,
            "cartesian_accuracy_threshold": args.cartesian_accuracy_threshold,
        },
        "ten_worst_paths_by_wrapped_rmse": worst_wrapped,
        "ten_worst_paths_by_aligned_rmse": worst_aligned,
        "overall_recommendation": recommendation,
    }
    with (args.output_dir / "residual_audit_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(json_safe(summary), handle, indent=2, sort_keys=True)
        handle.write("\n")

    print("classification counts:")
    for split, split_counts in counts.items():
        print(f"  {split}: {split_counts}")
    for split in ("train", "test", "combined"):
        selected = aggregate[aggregate["split"] == split].set_index("residual_mode")
        print(
            f"{split}: mean path RMSE "
            f"raw={selected.loc['raw', 'mean_path_rmse_rad']:.6f}, "
            f"wrapped={selected.loc['wrapped', 'mean_path_rmse_rad']:.6f}, "
            f"aligned={selected.loc['aligned', 'mean_path_rmse_rad']:.6f}"
        )
    print("ten worst paths by wrapped RMSE:")
    for record in worst_wrapped:
        print(
            f"  {record['split']}/{record['path_name']}: "
            f"{record['wrapped_rmse_rad']:.6f} rad"
        )
    print(f"final recommendation: {recommendation}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AuditValidationError, FileNotFoundError) as error:
        failure_args = parse_args()
        failure_args.output_dir.mkdir(parents=True, exist_ok=True)
        failure_summary = {
            "input_files": {
                "train_prior": str(failure_args.train_prior),
                "test_prior": str(failure_args.test_prior),
                "train_expert": str(failure_args.train_expert),
                "test_expert": str(failure_args.test_expert),
            },
            "discovered_npz_keys": {},
            "validation_results": {
                "passed": False,
                "error": f"{type(error).__name__}: {error}",
            },
            "path_counts": {},
            "classification_counts": {},
            "aggregate_residual_statistics": [],
            "thresholds": {
                "local_rmse_threshold": failure_args.local_rmse_threshold,
                "branch_rmse_threshold": failure_args.branch_rmse_threshold,
                "cartesian_accuracy_threshold": (
                    failure_args.cartesian_accuracy_threshold
                ),
            },
            "ten_worst_paths_by_wrapped_rmse": [],
            "ten_worst_paths_by_aligned_rmse": [],
            "overall_recommendation": "audit_failed_validation",
        }
        with (
            failure_args.output_dir / "residual_audit_summary.json"
        ).open("w", encoding="utf-8") as handle:
            json.dump(failure_summary, handle, indent=2, sort_keys=True)
            handle.write("\n")
        raise SystemExit(f"Audit validation failed: {error}") from error
