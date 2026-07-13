#!/usr/bin/env python3
"""Build unified ArtistDiffusion research-results tables.

The script aggregates whatever project summaries are available under
data/cartesian_expert_dataset_v3 and writes a full comparison table plus a
smaller main-paper table. Missing metrics are left blank.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


DEFAULT_ROOT = Path("data/cartesian_expert_dataset_v3")
DEFAULT_FULL_OUT = DEFAULT_ROOT / "unified_artistdiffusion_results_full.csv"
DEFAULT_MAIN_OUT = DEFAULT_ROOT / "unified_artistdiffusion_results_main.csv"

OUTPUT_FIELDS = (
    "method",
    "role",
    "mean_cartesian_error",
    "max_cartesian_error",
    "path_error",
    "q_rmse",
    "dtw_distance",
    "frechet_distance",
    "tangent_weighted_error",
    "drawing_total_cost",
    "joint_velocity_cost",
    "joint_acceleration_cost",
    "joint_jerk_cost",
    "accepted_count",
    "generation_or_solve_time_sec",
    "notes",
)

MAIN_METHODS = (
    "MLP-only",
    "Adaptive MLP + IK",
    "Diffusion v1 best-of-K",
    "Diffusion v4 MLP-prior refinement t=25",
    "Diffusion v4 v1-prior refinement t=25",
    "IK expert",
    "Noised-expert v4 reference t=25",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create unified ArtistDiffusion result tables.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--full_out", type=Path, default=DEFAULT_FULL_OUT)
    parser.add_argument("--main_out", type=Path, default=DEFAULT_MAIN_OUT)
    return parser.parse_args()


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="") as handle:
        return list(csv.DictReader(handle))


def to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        value_float = float(text)
    except ValueError:
        return None
    return value_float if np.isfinite(value_float) else None


def mean_values(values: Iterable[Any]) -> Optional[float]:
    parsed = [value for value in (to_float(item) for item in values) if value is not None]
    if not parsed:
        return None
    return float(np.mean(parsed))


def sum_true(values: Iterable[Any]) -> Optional[int]:
    seen = False
    count = 0
    for value in values:
        text = str(value).strip().lower()
        if text == "":
            continue
        seen = True
        if text in ("true", "1", "yes", "y", "accepted"):
            count += 1
    return count if seen else None


def first_matching_row(rows: Sequence[Dict[str, str]], *needles: str) -> Optional[Dict[str, str]]:
    lowered = [needle.lower() for needle in needles]
    for row in rows:
        haystack = " ".join(str(value).lower() for value in row.values())
        if all(needle in haystack for needle in lowered):
            return row
    return None


def best_numeric_row(rows: Sequence[Dict[str, str]], field: str) -> Optional[Dict[str, str]]:
    candidates: List[Tuple[float, Dict[str, str]]] = []
    for row in rows:
        value = to_float(row.get(field))
        if value is not None:
            candidates.append((value, row))
    if not candidates:
        return None
    return min(candidates, key=lambda item: item[0])[1]


def result_row(method: str, role: str, notes: str = "", **metrics: Any) -> Dict[str, Any]:
    row: Dict[str, Any] = {field: "" for field in OUTPUT_FIELDS}
    row["method"] = method
    row["role"] = role
    row["notes"] = notes
    for key, value in metrics.items():
        if key in row and value is not None:
            row[key] = value
    return row


def format_cell(value: Any) -> Any:
    if isinstance(value, float):
        return f"{value:.12e}"
    return value


def write_table(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: format_cell(row.get(field, "")) for field in OUTPUT_FIELDS})


def discover_likely_files(root: Path) -> List[Path]:
    patterns = ("*summary*.csv", "*results*.csv", "*comparison*.csv", "*.json")
    found: List[Path] = []
    for pattern in patterns:
        found.extend(root.rglob(pattern))
    useful = [
        path for path in sorted(set(found))
        if any(token in str(path).lower() for token in ("summary", "result", "comparison", "metrics"))
    ]
    return useful


def legacy_comparison_rows(root: Path) -> List[Dict[str, str]]:
    filled = root / "results_comparison_table_filled.csv"
    plain = root / "results_comparison_table.csv"
    return read_csv_rows(filled) or read_csv_rows(plain)


def build_ik_expert(root: Path) -> Dict[str, Any]:
    comparison = legacy_comparison_rows(root)
    cold_row = first_matching_row(comparison, "cold", "ik")
    timed_rows = read_csv_rows(root / "cold_ik_test_timed/ik_generation_summary.csv")
    accepted_rows = [row for row in timed_rows if str(row.get("accepted", "")).lower() == "true"]
    rows_for_mean = accepted_rows or timed_rows

    return result_row(
        "IK expert",
        "teacher/reference",
        "Cold IK teacher baseline; Cartesian metrics from accepted test paths when available.",
        mean_cartesian_error=(
            mean_values(row.get("mean_error") for row in rows_for_mean)
            or to_float(cold_row.get("mean_cartesian_error_m") if cold_row else None)
        ),
        max_cartesian_error=(
            mean_values(row.get("max_error") for row in rows_for_mean)
            or to_float(cold_row.get("mean_max_error_m") if cold_row else None)
        ),
        path_error=(
            mean_values(row.get("path_error") for row in rows_for_mean)
            or to_float(cold_row.get("mean_path_error") if cold_row else None)
        ),
        accepted_count=sum_true(row.get("accepted") for row in timed_rows)
        or to_float(cold_row.get("accepted") if cold_row else None),
        generation_or_solve_time_sec=to_float(cold_row.get("mean_solve_time_sec") if cold_row else None),
    )


def build_mlp_only(root: Path) -> Dict[str, Any]:
    comparison = legacy_comparison_rows(root)
    mlp_row = first_matching_row(comparison, "mlp", "only")
    mlp_summary = read_csv_rows(root / "mlp_v3_test_predictions_summary.csv")
    return result_row(
        "MLP-only",
        "learned prior",
        "Standalone path-conditioned MLP prediction; q_rmse from regenerated prediction summary.",
        mean_cartesian_error=to_float(mlp_row.get("mean_cartesian_error_m") if mlp_row else None),
        max_cartesian_error=to_float(mlp_row.get("mean_max_error_m") if mlp_row else None),
        path_error=to_float(mlp_row.get("mean_path_error") if mlp_row else None),
        q_rmse=mean_values(row.get("q_rmse_vs_expert") for row in mlp_summary),
        accepted_count=to_float(mlp_row.get("accepted") if mlp_row else None),
        generation_or_solve_time_sec=to_float(mlp_row.get("mean_solve_time_sec") if mlp_row else None),
    )


def build_adaptive_mlp_ik(root: Path) -> Dict[str, Any]:
    rows = read_csv_rows(root / "mlp_ik_refine_test_summary_smooth001.csv")
    comparison = legacy_comparison_rows(root)
    comparison_row = first_matching_row(comparison, "mlp", "ik")
    return result_row(
        "Adaptive MLP + IK",
        "robot-aware optimizer",
        "Adaptive IK refinement initialized from MLP; smooth=0.01 summary when available.",
        mean_cartesian_error=mean_values(row.get("after_mean_error") for row in rows)
        or to_float(comparison_row.get("mean_cartesian_error_m") if comparison_row else None),
        max_cartesian_error=mean_values(row.get("after_max_error") for row in rows)
        or to_float(comparison_row.get("mean_max_error_m") if comparison_row else None),
        path_error=mean_values(row.get("after_path_error") for row in rows)
        or to_float(comparison_row.get("mean_path_error") if comparison_row else None),
        accepted_count=sum_true(row.get("accepted") for row in rows)
        or to_float(comparison_row.get("accepted") if comparison_row else None),
        generation_or_solve_time_sec=mean_values(row.get("solve_time_sec") for row in rows)
        or to_float(comparison_row.get("mean_solve_time_sec") if comparison_row else None),
    )


def build_v1_single(root: Path) -> Dict[str, Any]:
    rows = read_csv_rows(root / "diffusion_v1_samples/diffusion_sample_summary.csv")
    return result_row(
        "Diffusion v1 single sample",
        "generative baseline",
        "Single v1 diffusion sample; smoothness column mapped to joint_acceleration_cost.",
        mean_cartesian_error=mean_values(row.get("mean_error") for row in rows),
        max_cartesian_error=mean_values(row.get("max_error") for row in rows),
        path_error=mean_values(row.get("path_error") for row in rows),
        joint_acceleration_cost=mean_values(row.get("smoothness") for row in rows),
        accepted_count=sum_true(row.get("accepted") for row in rows),
    )


def build_v1_best_of_k(root: Path) -> Dict[str, Any]:
    rows = read_csv_rows(root / "diffusion_v1_ranked_samples/best_of_k_summary.csv")
    best = best_numeric_row(rows, "mean_total_cost")
    if best is None:
        best = best_numeric_row(rows, "mean_mean_error")
    root_name = Path(best.get("root", "")).name if best else ""
    return result_row(
        "Diffusion v1 best-of-K",
        "reranked generative baseline",
        f"Best aggregate row from best_of_k_summary.csv ({root_name}); mean_total_cost is stored in drawing_total_cost for compact comparison.",
        mean_cartesian_error=to_float(best.get("mean_mean_error") if best else None),
        max_cartesian_error=to_float(best.get("mean_max_error") if best else None),
        path_error=to_float(best.get("mean_path_error") if best else None),
        drawing_total_cost=to_float(best.get("mean_total_cost") if best else None),
        joint_acceleration_cost=to_float(best.get("mean_joint_acceleration_cost") if best else None),
        accepted_count=to_float(best.get("accepted_paths") if best else None),
    )


def aggregate_rows(rows: Sequence[Dict[str, str]], group_key: Tuple[str, str, str]) -> Optional[Dict[str, Any]]:
    experiment, source, t_start = group_key
    group = [
        row for row in rows
        if row.get("experiment_name") == experiment
        and row.get("source") == source
        and row.get("t_start") == t_start
    ]
    if not group:
        return None
    metrics = (
        "q_rmse",
        "mean_cartesian_error",
        "max_cartesian_error",
        "path_error",
        "dtw_distance",
        "frechet_distance",
        "tangent_weighted_error",
        "drawing_total_cost",
        "joint_velocity_cost",
        "joint_acceleration_cost",
        "joint_jerk_cost",
        "drawing_total_improved_paths",
    )
    out: Dict[str, Any] = {}
    for metric in metrics:
        values = [row.get(metric) for row in group if row.get(metric, "") != ""]
        if metric == "drawing_total_improved_paths":
            out[metric] = to_float(values[0]) if len(values) == 1 else mean_values(values)
        else:
            out[metric] = mean_values(values)
    return out


def load_v4_groups(root: Path) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    detailed = read_csv_rows(root / "diffusion_v4_unet/prior_refinement_fk_robot_costs_drawing.csv")
    full = read_csv_rows(root / "diffusion_v4_unet/v4_refinement_full_summary.csv")
    key = read_csv_rows(root / "diffusion_v4_unet/v4_refinement_key_results.csv")
    all_keys = {
        (row.get("experiment_name", ""), row.get("source", ""), row.get("t_start", ""))
        for rows in (detailed, full, key)
        for row in rows
    }
    groups: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for group_key in all_keys:
        if not all(group_key):
            continue
        merged: Dict[str, Any] = {}
        for rows in (detailed, full, key):
            agg = aggregate_rows(rows, group_key)
            if agg:
                for metric, value in agg.items():
                    if value is not None:
                        merged[metric] = value
        if merged:
            groups[group_key] = merged
    return groups


def build_v4_row(
    method: str,
    role: str,
    groups: Dict[Tuple[str, str, str], Dict[str, Any]],
    group_key: Tuple[str, str, str],
    notes: str,
) -> Dict[str, Any]:
    metrics = groups.get(group_key, {})
    return result_row(
        method,
        role,
        notes,
        mean_cartesian_error=metrics.get("mean_cartesian_error"),
        max_cartesian_error=metrics.get("max_cartesian_error"),
        path_error=metrics.get("path_error"),
        q_rmse=metrics.get("q_rmse"),
        dtw_distance=metrics.get("dtw_distance"),
        frechet_distance=metrics.get("frechet_distance"),
        tangent_weighted_error=metrics.get("tangent_weighted_error"),
        drawing_total_cost=metrics.get("drawing_total_cost"),
        joint_velocity_cost=metrics.get("joint_velocity_cost"),
        joint_acceleration_cost=metrics.get("joint_acceleration_cost"),
        joint_jerk_cost=metrics.get("joint_jerk_cost"),
    )


def build_all_rows(root: Path) -> List[Dict[str, Any]]:
    v4_groups = load_v4_groups(root)
    rows = [
        build_ik_expert(root),
        build_mlp_only(root),
        build_adaptive_mlp_ik(root),
        build_v1_single(root),
        build_v1_best_of_k(root),
        build_v4_row(
            "Diffusion v4 pure Gaussian t=25",
            "sampler baseline",
            v4_groups,
            ("mlp_prior_v4_refine", "pure_gaussian", "25"),
            "Pure Gaussian deterministic v4 rollout at t=25; MLP experiment seed/group used when duplicate references exist.",
        ),
        build_v4_row(
            "Diffusion v4 MLP-prior refinement t=25",
            "deterministic diffusion refiner",
            v4_groups,
            ("mlp_prior_v4_refine", "prior_refined", "25"),
            "v4 deterministic refinement initialized from MLP prior.",
        ),
        build_v4_row(
            "Diffusion v4 v1-prior refinement t=25",
            "deterministic diffusion refiner",
            v4_groups,
            ("v1_prior_v4_refine", "prior_refined", "25"),
            "v4 deterministic refinement initialized from v1 best prior.",
        ),
        build_v4_row(
            "Noised-expert v4 reference t=25",
            "near-manifold reference",
            v4_groups,
            ("mlp_prior_v4_refine", "noised_expert", "25"),
            "Upper-bound-style v4 reference initialized from a noised expert trajectory.",
        ),
    ]
    return rows


def select_main_rows(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_method = {row["method"]: row for row in rows}
    return [by_method[method] for method in MAIN_METHODS if method in by_method]


def metric_value(row: Dict[str, Any], field: str) -> Optional[float]:
    return to_float(row.get(field))


def best_by(rows: Sequence[Dict[str, Any]], field: str, exclude_reference: bool = False) -> Optional[Dict[str, Any]]:
    candidates: List[Tuple[float, Dict[str, Any]]] = []
    for row in rows:
        if exclude_reference and "reference" in str(row.get("role", "")).lower():
            continue
        value = metric_value(row, field)
        if value is not None:
            candidates.append((value, row))
    if not candidates:
        return None
    return min(candidates, key=lambda item: item[0])[1]


def pct_improvement(before: Optional[float], after: Optional[float]) -> Optional[float]:
    if before is None or after is None or abs(before) < 1e-12:
        return None
    return 100.0 * (before - after) / before


def print_interpretation(
    rows: Sequence[Dict[str, Any]],
    v4_groups: Dict[Tuple[str, str, str], Dict[str, Any]],
) -> None:
    print("\nInterpretation")
    best_accuracy = best_by(rows, "mean_cartesian_error")
    if best_accuracy:
        print(
            f"  Most accurate by mean Cartesian error: {best_accuracy['method']} "
            f"({metric_value(best_accuracy, 'mean_cartesian_error'):.6e})."
        )

    diffusion_rows = [
        row for row in rows
        if "diffusion" in row["method"].lower()
        and "reference" not in str(row.get("role", "")).lower()
    ]
    strongest_diffusion = best_by(diffusion_rows, "mean_cartesian_error")
    if strongest_diffusion:
        print(
            f"  Strongest diffusion method by mean Cartesian error: {strongest_diffusion['method']} "
            f"({metric_value(strongest_diffusion, 'mean_cartesian_error'):.6e})."
        )

    v4_rows = [
        row for row in rows
        if row["method"].startswith("Diffusion v4")
        and "reference" not in str(row.get("role", "")).lower()
    ]
    strongest_v4 = best_by(v4_rows, "drawing_total_cost")
    if strongest_v4:
        print(
            f"  Strongest drawing-aware v4 row by drawing_total_cost: {strongest_v4['method']} "
            f"({metric_value(strongest_v4, 'drawing_total_cost'):.6e})."
        )

    for label, experiment in (
        ("MLP-prior", "mlp_prior_v4_refine"),
        ("v1-prior", "v1_prior_v4_refine"),
    ):
        prior = v4_groups.get((experiment, "prior_only", "prior_only"))
        refined = v4_groups.get((experiment, "prior_refined", "25"))
        if not prior or not refined:
            continue
        cart_pct = pct_improvement(
            to_float(prior.get("mean_cartesian_error")),
            to_float(refined.get("mean_cartesian_error")),
        )
        drawing_pct = pct_improvement(
            to_float(prior.get("drawing_total_cost")),
            to_float(refined.get("drawing_total_cost")),
        )
        if cart_pct is not None and drawing_pct is not None:
            print(
                f"  v4 {label} refinement t=25 improves its prior_only baseline by "
                f"{cart_pct:.2f}% mean Cartesian and {drawing_pct:.2f}% drawing_total_cost."
            )

    print(
        "  Remaining unsolved: v4 refinement is useful near plausible priors, but pure Gaussian sampling "
        "remains weak and drawing-fidelity metrics are needed because mean Cartesian error alone can hide poor stroke shape."
    )


def print_found_files(root: Path) -> None:
    files = discover_likely_files(root)
    print(f"Found {len(files)} likely summary/result files under {root}")
    for path in files[:20]:
        print(f"  {path}")
    if len(files) > 20:
        print(f"  ... {len(files) - 20} more")


def main() -> int:
    args = parse_args()
    print_found_files(args.root)
    rows = build_all_rows(args.root)
    v4_groups = load_v4_groups(args.root)
    main_rows = select_main_rows(rows)
    write_table(args.full_out, rows)
    write_table(args.main_out, main_rows)
    print(f"\nSaved full unified table: {args.full_out}")
    print(f"Saved main-paper table: {args.main_out}")
    print_interpretation(rows, v4_groups)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
