#!/usr/bin/env python3
"""Rerank diffusion candidate trajectories with alternative objectives."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


DEFAULT_ROOTS = [
    "data/cartesian_expert_dataset_v3/diffusion_v1_ranked_samples/k4",
    "data/cartesian_expert_dataset_v3/diffusion_v1_ranked_samples/k8",
    "data/cartesian_expert_dataset_v3/diffusion_v1_ranked_samples/k16",
]
DEFAULT_OUTPUT_CSV = (
    "data/cartesian_expert_dataset_v3/diffusion_v1_ranked_samples/rerank_summary.csv"
)
ALL_CANDIDATES_CSV = "diffusion_v1_all_candidates.csv"

ACCEPTANCE_MEAN_ERROR = 0.010
ACCEPTANCE_MAX_ERROR = 0.030

REQUIRED_BASE_COLUMNS = ["path_name", "mean_error", "max_error"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rerank already generated diffusion candidate trajectories using "
            "alternative per-path objectives."
        )
    )
    parser.add_argument(
        "--roots",
        nargs="+",
        default=DEFAULT_ROOTS,
        help="One or more diffusion ranked-sample root folders.",
    )
    parser.add_argument(
        "--output_csv",
        default=DEFAULT_OUTPUT_CSV,
        help="Path for the combined rerank summary CSV.",
    )
    return parser.parse_args()


def infer_k_value(root: Path) -> str:
    return root.name if root.name.lower().startswith("k") else ""


def numeric_column(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[column], errors="coerce").fillna(default)


def require_columns(df: pd.DataFrame, columns: list[str], csv_path: Path) -> bool:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        print(
            f"WARNING: Skipping {csv_path} because required columns are missing: "
            f"{', '.join(missing)}"
        )
        return False
    return True


def objective_functions() -> dict[str, Callable[[pd.DataFrame], pd.Series]]:
    return {
        "original_total_cost": lambda df: numeric_column(df, "total_cost"),
        "mean_error_only": lambda df: numeric_column(df, "mean_error"),
        "max_error_only": lambda df: numeric_column(df, "max_error"),
        "mean_plus_max": lambda df: (
            numeric_column(df, "mean_error") + numeric_column(df, "max_error")
        ),
        "path_plus_xyz": lambda df: (
            numeric_column(df, "path_error") + numeric_column(df, "weighted_xyz_loss")
        ),
        "path_xyz_accel": lambda df: (
            numeric_column(df, "path_error")
            + numeric_column(df, "weighted_xyz_loss")
            + 0.01 * numeric_column(df, "joint_acceleration_cost")
        ),
        "acceptance_focused": lambda df: (
            numeric_column(df, "mean_error")
            + 2.0 * numeric_column(df, "max_error")
            + 0.01 * numeric_column(df, "joint_acceleration_cost")
        ),
        "squared_acceptance_focused": lambda df: (
            numeric_column(df, "mean_error") ** 2
            + numeric_column(df, "max_error") ** 2
            + 0.01 * numeric_column(df, "joint_acceleration_cost")
        ),
        "max_threshold_focused": lambda df: (
            numeric_column(df, "mean_error")
            + 5.0
            * np.maximum(0.0, numeric_column(df, "max_error") - ACCEPTANCE_MAX_ERROR)
            + 0.01 * numeric_column(df, "joint_acceleration_cost")
        ),
    }


def select_best_per_path(df: pd.DataFrame, rerank_score: pd.Series) -> pd.DataFrame:
    ranked = df.copy()
    ranked["rerank_score"] = pd.to_numeric(rerank_score, errors="coerce")
    ranked["rerank_score"] = ranked["rerank_score"].replace([np.inf, -np.inf], np.nan)
    ranked["rerank_score"] = ranked["rerank_score"].fillna(np.inf)
    best_indices = ranked.groupby("path_name", sort=False)["rerank_score"].idxmin()
    return ranked.loc[best_indices].copy()


def mean_of_column(df: pd.DataFrame, column: str) -> float:
    return float(numeric_column(df, column).mean()) if len(df) else float("nan")


def max_of_column(df: pd.DataFrame, column: str) -> float:
    return float(numeric_column(df, column).max()) if len(df) else float("nan")


def summarize_selection(
    selected: pd.DataFrame,
    root: Path,
    objective: str,
) -> dict[str, object]:
    mean_error = numeric_column(selected, "mean_error")
    max_error = numeric_column(selected, "max_error")
    accepted = (mean_error <= ACCEPTANCE_MEAN_ERROR) & (
        max_error <= ACCEPTANCE_MAX_ERROR
    )

    return {
        "root": str(root),
        "k_value": infer_k_value(root),
        "objective": objective,
        "evaluated_paths": int(selected["path_name"].nunique()),
        "accepted_paths": int(accepted.sum()),
        "mean_path_error": mean_of_column(selected, "path_error"),
        "mean_mean_error": mean_of_column(selected, "mean_error"),
        "mean_max_error": mean_of_column(selected, "max_error"),
        "worst_max_error": max_of_column(selected, "max_error"),
        "mean_total_cost": mean_of_column(selected, "total_cost"),
        "mean_joint_acceleration_cost": mean_of_column(
            selected, "joint_acceleration_cost"
        ),
        "mean_rerank_score": mean_of_column(selected, "rerank_score"),
    }


def save_selected_candidates(
    selected: pd.DataFrame,
    root: Path,
    objective: str,
) -> None:
    output_path = root / f"reranked_best_{objective}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(output_path, index=False)


def rerank_root(root: Path) -> list[dict[str, object]]:
    csv_path = root / ALL_CANDIDATES_CSV
    if not csv_path.exists():
        print(f"WARNING: Missing candidates file: {csv_path}")
        return []

    df = pd.read_csv(csv_path)
    if not require_columns(df, REQUIRED_BASE_COLUMNS, csv_path):
        return []

    summaries = []
    for objective, score_fn in objective_functions().items():
        selected = select_best_per_path(df, score_fn(df))
        selected["accepted"] = (
            (numeric_column(selected, "mean_error") <= ACCEPTANCE_MEAN_ERROR)
            & (numeric_column(selected, "max_error") <= ACCEPTANCE_MAX_ERROR)
        )
        save_selected_candidates(selected, root, objective)
        summaries.append(summarize_selection(selected, root, objective))

    return summaries


def print_best(
    summary_df: pd.DataFrame,
    label: str,
    metric: str,
    maximize: bool = False,
) -> None:
    if summary_df.empty or metric not in summary_df.columns:
        return

    values = pd.to_numeric(summary_df[metric], errors="coerce")
    if values.isna().all():
        return

    best_index = values.idxmax() if maximize else values.idxmin()
    row = summary_df.loc[best_index]
    print(
        f"{label}: root={row['root']} objective={row['objective']} "
        f"{metric}={row[metric]}"
    )


def print_summary(summary_df: pd.DataFrame) -> None:
    if summary_df.empty:
        print("No rerank summaries were produced.")
        return

    print("\nRerank summary")
    print("==============")
    print_best(summary_df, "Best objective by mean_mean_error", "mean_mean_error")
    print_best(summary_df, "Best objective by mean_max_error", "mean_max_error")
    print_best(
        summary_df,
        "Best objective by accepted_paths",
        "accepted_paths",
        maximize=True,
    )
    print_best(summary_df, "Best objective by worst_max_error", "worst_max_error")

    display_columns = [
        "k_value",
        "objective",
        "evaluated_paths",
        "accepted_paths",
        "mean_mean_error",
        "mean_max_error",
        "worst_max_error",
    ]
    print("\nAll objectives")
    print(summary_df[display_columns].to_string(index=False))


def main() -> None:
    args = parse_args()
    summaries = []

    for root_arg in args.roots:
        root = Path(root_arg)
        summaries.extend(rerank_root(root))

    summary_df = pd.DataFrame(summaries)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_csv, index=False)

    print(f"Saved combined summary to: {output_csv}")
    print_summary(summary_df)


if __name__ == "__main__":
    main()
