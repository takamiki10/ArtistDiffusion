#!/usr/bin/env python3
"""Create compact research-results tables for v4 diffusion refinement."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np


DEFAULT_INPUT_CSV = Path(
    "data/cartesian_expert_dataset_v3/diffusion_v4_unet/prior_refinement_fk_robot_costs_drawing.csv"
)
DEFAULT_KEY_CSV = Path(
    "data/cartesian_expert_dataset_v3/diffusion_v4_unet/v4_refinement_key_results.csv"
)
DEFAULT_FULL_CSV = Path(
    "data/cartesian_expert_dataset_v3/diffusion_v4_unet/v4_refinement_full_summary.csv"
)
METRIC_FIELDS = (
    "q_rmse",
    "mean_cartesian_error",
    "max_cartesian_error",
    "frechet_distance",
    "dtw_distance",
    "tangent_weighted_error",
    "progress_error",
    "length_ratio_error",
    "normalized_shape_error",
    "total_cost",
    "shape_total_cost",
    "drawing_total_cost",
)
IMPROVEMENT_METRICS = (
    "mean_cartesian_error",
    "dtw_distance",
    "tangent_weighted_error",
    "drawing_total_cost",
)
KEY_ROWS = (
    ("MLP prior_only", "mlp_prior_v4_refine", "prior_only", "prior_only"),
    ("MLP prior_refined t=25", "mlp_prior_v4_refine", "prior_refined", "25"),
    ("MLP pure_gaussian t=25", "mlp_prior_v4_refine", "pure_gaussian", "25"),
    ("MLP noised_expert t=25", "mlp_prior_v4_refine", "noised_expert", "25"),
    ("v1 prior_only", "v1_prior_v4_refine", "prior_only", "prior_only"),
    ("v1 prior_refined t=25", "v1_prior_v4_refine", "prior_refined", "25"),
    ("v1 pure_gaussian t=25", "v1_prior_v4_refine", "pure_gaussian", "25"),
    ("v1 noised_expert t=25", "v1_prior_v4_refine", "noised_expert", "25"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize v4 diffusion refinement evaluation results.")
    parser.add_argument("--input_csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--key_csv", type=Path, default=DEFAULT_KEY_CSV)
    parser.add_argument("--full_csv", type=Path, default=DEFAULT_FULL_CSV)
    return parser.parse_args()


def read_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing input CSV: {path}")
    with path.open("r", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"experiment_name", "source", "t_start", "path_name", *METRIC_FIELDS}
    missing = sorted(required - set(rows[0].keys())) if rows else sorted(required)
    if missing:
        raise KeyError(f"{path} missing required columns: {missing}")
    return rows


def finite_mean(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    return float(np.nanmean(arr)) if arr.size else float("nan")


def group_rows(rows: Sequence[Dict[str, str]]) -> Dict[Tuple[str, str, str], List[Dict[str, str]]]:
    groups: Dict[Tuple[str, str, str], List[Dict[str, str]]] = {}
    for row in rows:
        key = (row["experiment_name"], row["source"], row["t_start"])
        groups.setdefault(key, []).append(row)
    return groups


def make_group_summary(rows: Sequence[Dict[str, str]]) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    groups = group_rows(rows)
    prior_by_experiment_path = {
        (row["experiment_name"], row["path_name"]): float(row["drawing_total_cost"])
        for row in rows
        if row["source"] == "prior_only"
    }
    summary: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for key, group in groups.items():
        experiment_name, source, t_start = key
        out: Dict[str, Any] = {
            "experiment_name": experiment_name,
            "source": source,
            "t_start": t_start,
            "num_paths": len(group),
        }
        for metric in METRIC_FIELDS:
            out[metric] = finite_mean([float(row[metric]) for row in group])

        improved = 0
        comparable = 0
        for row in group:
            baseline = prior_by_experiment_path.get((experiment_name, row["path_name"]))
            if baseline is None:
                continue
            comparable += 1
            if float(row["drawing_total_cost"]) < baseline:
                improved += 1
        out["drawing_total_improved_paths"] = improved
        out["drawing_total_comparable_paths"] = comparable
        summary[key] = out
    return summary


def percent_improvement(prior_value: float, refined_value: float) -> float:
    if not np.isfinite(prior_value) or not np.isfinite(refined_value) or abs(prior_value) < 1e-12:
        return float("nan")
    return 100.0 * (prior_value - refined_value) / prior_value


def add_prior_refined_improvements(summary: Dict[Tuple[str, str, str], Dict[str, Any]]) -> None:
    experiments = sorted({key[0] for key in summary})
    for experiment in experiments:
        prior = summary.get((experiment, "prior_only", "prior_only"))
        refined = summary.get((experiment, "prior_refined", "25"))
        if prior is None or refined is None:
            continue
        for metric in IMPROVEMENT_METRICS:
            refined[f"{metric}_improvement_pct_vs_prior_only"] = percent_improvement(
                float(prior[metric]),
                float(refined[metric]),
            )


def format_value(value: Any) -> Any:
    if isinstance(value, float):
        return f"{value:.12e}"
    return value


def write_csv(path: Path, rows: Sequence[Dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: format_value(row.get(field, "")) for field in fields})


def full_summary_rows(summary: Dict[Tuple[str, str, str], Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [summary[key] for key in sorted(summary)]


def key_rows(summary: Dict[Tuple[str, str, str], Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for label, experiment, source, t_start in KEY_ROWS:
        row = summary.get((experiment, source, t_start))
        if row is None:
            continue
        out = dict(row)
        out["label"] = label
        rows.append(out)
    return rows


def print_table(title: str, rows: Sequence[Dict[str, Any]]) -> None:
    print(f"\n{title}")
    print(
        "label/source | t_start | mean_cart | dtw | tangent | norm_shape | drawing_total | improved"
    )
    for row in rows:
        label = row.get("label", f"{row['experiment_name']}:{row['source']}")
        improved = f"{row.get('drawing_total_improved_paths', '')}/{row.get('drawing_total_comparable_paths', '')}"
        print(
            f"{label} | {row['t_start']} | "
            f"{float(row['mean_cartesian_error']):.6e} | "
            f"{float(row['dtw_distance']):.6e} | "
            f"{float(row['tangent_weighted_error']):.6e} | "
            f"{float(row['normalized_shape_error']):.6e} | "
            f"{float(row['drawing_total_cost']):.6e} | "
            f"{improved}"
        )


def print_interpretation(summary: Dict[Tuple[str, str, str], Dict[str, Any]]) -> None:
    print("\nInterpretation")
    for experiment in sorted({key[0] for key in summary}):
        prior = summary.get((experiment, "prior_only", "prior_only"))
        refined = summary.get((experiment, "prior_refined", "25"))
        gaussian = summary.get((experiment, "pure_gaussian", "25"))
        noised = summary.get((experiment, "noised_expert", "25"))
        if prior is None or refined is None:
            print(f"  {experiment}: missing prior_only or prior_refined t=25 rows.")
            continue
        cart_pct = percent_improvement(float(prior["mean_cartesian_error"]), float(refined["mean_cartesian_error"]))
        dtw_pct = percent_improvement(float(prior["dtw_distance"]), float(refined["dtw_distance"]))
        tangent_pct = percent_improvement(float(prior["tangent_weighted_error"]), float(refined["tangent_weighted_error"]))
        drawing_pct = percent_improvement(float(prior["drawing_total_cost"]), float(refined["drawing_total_cost"]))
        print(
            f"  {experiment}: prior_refined t=25 vs prior_only improves mean Cartesian by "
            f"{cart_pct:.2f}%, DTW by {dtw_pct:.2f}%, tangent by {tangent_pct:.2f}%, "
            f"and drawing_total_cost by {drawing_pct:.2f}%."
        )
        if gaussian is not None:
            comparison = "lower" if float(refined["drawing_total_cost"]) < float(gaussian["drawing_total_cost"]) else "higher"
            print(f"    refined drawing_total_cost is {comparison} than pure_gaussian t=25.")
        if noised is not None:
            gap = float(refined["drawing_total_cost"]) - float(noised["drawing_total_cost"])
            print(f"    noised_expert t=25 reference gap in drawing_total_cost: {gap:.6e}.")


def main() -> int:
    args = parse_args()
    rows = read_rows(args.input_csv)
    summary = make_group_summary(rows)
    add_prior_refined_improvements(summary)

    improvement_fields = tuple(f"{metric}_improvement_pct_vs_prior_only" for metric in IMPROVEMENT_METRICS)
    common_fields = (
        "experiment_name",
        "source",
        "t_start",
        "num_paths",
        *METRIC_FIELDS,
        "drawing_total_improved_paths",
        "drawing_total_comparable_paths",
        *improvement_fields,
    )
    key_fields = ("label", *common_fields)

    full_rows = full_summary_rows(summary)
    selected_key_rows = key_rows(summary)
    write_csv(args.full_csv, full_rows, common_fields)
    write_csv(args.key_csv, selected_key_rows, key_fields)

    print(f"Saved full summary: {args.full_csv}")
    print(f"Saved key results: {args.key_csv}")
    print_table("Key Comparison Table", selected_key_rows)
    print_interpretation(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
