#!/usr/bin/env python3
"""Summarize frozen v8 anchored rollout seeds without post-hoc K selection."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple


FROZEN_SEEDS = (43, 44, 45, 46, 47)
DECISION_K = 8
ORDINARY_PATH_COUNT = 20
MAX_ALLOWED_SEGMENT_CORRECTION_GROWTH_RAD_PER_STEP = 1.0e-5
MAXIMUM_INTERNAL_JOINT_STEP_RAD = 0.20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input_root",
        type=Path,
        default=Path("results/diffusion_v8_anchored_recursive_multiseed"),
    )
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--allow_incomplete", action="store_true")
    return parser.parse_args()


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def number(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return math.nan
    return result


def finite(values: Iterable[float]) -> List[float]:
    return [value for value in values if math.isfinite(value)]


def mean_std(values: Iterable[float]) -> Tuple[float, float]:
    clean = finite(values)
    if not clean:
        return math.nan, math.nan
    return mean(clean), stdev(clean) if len(clean) > 1 else 0.0


def load_runs(
    input_root: Path,
    allow_incomplete: bool,
) -> Tuple[List[Dict[str, Any]], List[int]]:
    rows: List[Dict[str, Any]] = []
    missing: List[int] = []
    for seed in FROZEN_SEEDS:
        path = input_root / f"seed_{seed}" / "anchored_full_path_metrics.csv"
        if not path.is_file():
            missing.append(seed)
            continue
        rows.extend({"sampling_seed": seed, **row} for row in read_csv(path))
    if missing and not allow_incomplete:
        raise FileNotFoundError(f"Missing required anchored seeds: {missing}")
    if not rows:
        raise FileNotFoundError("No completed anchored seed metrics were found")
    return rows, missing


def per_seed_rows(
    rows: Sequence[Mapping[str, Any]],
    completed_seeds: Sequence[int],
) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for seed in completed_seeds:
        decision_rows = [
            row
            for row in rows
            if int(row["sampling_seed"]) == seed
            and str(row["population"]) == "ordinary"
            and int(row["k"]) == DECISION_K
        ]
        if len(decision_rows) != ORDINARY_PATH_COUNT:
            raise ValueError(
                f"Seed {seed} has {len(decision_rows)} ordinary K=8 paths; "
                f"expected {ORDINARY_PATH_COUNT}"
            )
        safety_rate = mean(
            number(row["full_path_safety_pass"]) for row in decision_rows
        )
        improved_fraction = mean(
            number(row["internal_full_path_robot_aware_delta_score"]) < 0.0
            for row in decision_rows
        )
        output.append(
            {
                "sampling_seed": seed,
                "population": "ordinary",
                "k": DECISION_K,
                "path_count": len(decision_rows),
                "final_output_safety_pass_rate": safety_rate,
                "fraction_paths_internal_full_path_score_negative": (
                    improved_fraction
                ),
                "mean_total_robot_aware_delta_score": mean(
                    number(row["total_robot_aware_delta_score"])
                    for row in decision_rows
                ),
                "mean_legacy_full_path_robot_aware_delta_score": mean(
                    number(row["legacy_full_path_robot_aware_delta_score"])
                    for row in decision_rows
                ),
                "std_legacy_full_path_robot_aware_delta_score": stdev(
                    number(row["legacy_full_path_robot_aware_delta_score"])
                    for row in decision_rows
                ),
                "mean_internal_full_path_robot_aware_delta_score": mean(
                    number(row["internal_full_path_robot_aware_delta_score"])
                    for row in decision_rows
                ),
                "std_internal_full_path_robot_aware_delta_score": stdev(
                    number(row["internal_full_path_robot_aware_delta_score"])
                    for row in decision_rows
                ),
                "mean_sum_selected_local_delta_score": mean(
                    number(row["sum_selected_local_delta_score"])
                    for row in decision_rows
                ),
                "std_sum_selected_local_delta_score": stdev(
                    number(row["sum_selected_local_delta_score"])
                    for row in decision_rows
                ),
                "mean_full_path_recomputed_delta_score": mean(
                    number(row["full_path_recomputed_delta_score"])
                    for row in decision_rows
                ),
                "mean_local_vs_full_delta_score_gap": mean(
                    number(row["local_vs_full_delta_score_gap"])
                    for row in decision_rows
                ),
                "mean_legacy_local_vs_full_delta_score_gap": mean(
                    number(row["legacy_local_vs_full_delta_score_gap"])
                    for row in decision_rows
                ),
                "std_legacy_local_vs_full_delta_score_gap": stdev(
                    number(row["legacy_local_vs_full_delta_score_gap"])
                    for row in decision_rows
                ),
                "mean_internal_local_vs_full_delta_score_gap": mean(
                    number(row["internal_local_vs_full_delta_score_gap"])
                    for row in decision_rows
                ),
                "std_internal_local_vs_full_delta_score_gap": stdev(
                    number(row["internal_local_vs_full_delta_score_gap"])
                    for row in decision_rows
                ),
                "mean_terminal_joint_deviation_norm_rad": mean(
                    number(row["terminal_joint_deviation_norm_rad"])
                    for row in decision_rows
                ),
                "std_terminal_joint_deviation_norm_rad": stdev(
                    number(row["terminal_joint_deviation_norm_rad"])
                    for row in decision_rows
                ),
                "mean_terminal_joint_deviation_max_rad": mean(
                    number(row["terminal_joint_deviation_max_rad"])
                    for row in decision_rows
                ),
                "std_terminal_joint_deviation_max_rad": stdev(
                    number(row["terminal_joint_deviation_max_rad"])
                    for row in decision_rows
                ),
                "mean_max_join_absolute_joint_step": mean(
                    number(row["max_join_absolute_joint_step"])
                    for row in decision_rows
                ),
                "std_max_join_absolute_joint_step": stdev(
                    number(row["max_join_absolute_joint_step"])
                    for row in decision_rows
                ),
                "maximum_actual_internal_joint_step_rad": max(
                    number(row["maximum_actual_internal_joint_step_rad"])
                    for row in decision_rows
                ),
                "mean_max_join_joint_acceleration_norm": mean(
                    number(row["max_join_joint_acceleration_norm"])
                    for row in decision_rows
                ),
                "std_max_join_joint_acceleration_norm": stdev(
                    number(row["max_join_joint_acceleration_norm"])
                    for row in decision_rows
                ),
                "mean_cartesian_mean_error_delta": mean(
                    number(row["cartesian_mean_error_delta"])
                    for row in decision_rows
                ),
                "mean_velocity_cost_delta": mean(
                    number(row["velocity_cost_delta"]) for row in decision_rows
                ),
                "mean_acceleration_cost_delta": mean(
                    number(row["acceleration_cost_delta"]) for row in decision_rows
                ),
                "mean_jerk_cost_delta": mean(
                    number(row["jerk_cost_delta"]) for row in decision_rows
                ),
                "mean_joint_limit_cost_delta": mean(
                    number(row["joint_limit_cost_delta"]) for row in decision_rows
                ),
                "mean_singularity_cost_delta": mean(
                    number(row["singularity_cost_delta"]) for row in decision_rows
                ),
                "mean_minimum_manipulability_delta": mean(
                    number(row["minimum_manipulability_delta"])
                    for row in decision_rows
                ),
                "mean_boundary_anchoring_offset_growth_slope_per_step": mean(
                    number(
                        row[
                            "boundary_anchoring_offset_growth_slope_per_step"
                        ]
                    )
                    for row in decision_rows
                ),
                "mean_segment_mean_correction_growth_slope_per_step": mean(
                    number(
                        row[
                            "segment_mean_correction_growth_slope_per_step"
                        ]
                    )
                    for row in decision_rows
                ),
                "mean_segment_max_correction_growth_slope_per_step": mean(
                    number(
                        row[
                            "segment_max_correction_growth_slope_per_step"
                        ]
                    )
                    for row in decision_rows
                ),
                "mean_cartesian_error_delta_growth_slope": mean(
                    number(row["cartesian_error_delta_growth_slope"])
                    for row in decision_rows
                ),
            }
        )
        contribution_fields = sorted(
            {
                key
                for row in decision_rows
                for key in row
                if "robot_score_contribution_" in key
            }
        )
        for field in contribution_fields:
            output[-1][f"mean_{field}"] = mean(
                number(row[field]) for row in decision_rows
            )
    return output


def per_path_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    identities = sorted(
        {
            (str(row["population"]), int(row["k"]), str(row["path_id"]))
            for row in rows
        }
    )
    contribution_metrics = tuple(
        sorted(
            {
                key
                for row in rows
                for key in row
                if "robot_score_contribution_" in key
            }
        )
    )
    metrics = (
        "full_path_safety_pass",
        "total_robot_aware_delta_score",
        "legacy_full_path_robot_aware_delta_score",
        "internal_full_path_robot_aware_delta_score",
        "sum_selected_local_delta_score",
        "full_path_recomputed_delta_score",
        "local_vs_full_delta_score_gap",
        "legacy_local_vs_full_delta_score_gap",
        "internal_local_vs_full_delta_score_gap",
        "terminal_joint_deviation_norm_rad",
        "terminal_joint_deviation_max_rad",
        "terminal_cartesian_deviation_from_prior_m",
        "legacy_terminal_boundary_step_contribution",
        "legacy_terminal_boundary_acceleration_contribution",
        "mean_join_joint_step_norm",
        "max_join_joint_step_norm",
        "max_join_absolute_joint_step",
        "mean_join_joint_acceleration_norm",
        "max_join_joint_acceleration_norm",
        "maximum_actual_internal_joint_step_rad",
        "cartesian_mean_error_delta",
        "velocity_cost_delta",
        "acceleration_cost_delta",
        "jerk_cost_delta",
        "joint_limit_cost_delta",
        "singularity_cost_delta",
        "minimum_manipulability_delta",
        "correction_growth_slope",
        "boundary_anchoring_offset_growth_slope_per_step",
        "segment_mean_correction_growth_slope_per_step",
        "segment_max_correction_growth_slope_per_step",
        "cartesian_error_delta_growth_slope",
        "accepted_rollout_step_rate",
        "fallback_rate",
        *contribution_metrics,
    )
    for population, k, path_id in identities:
        subset = [
            row
            for row in rows
            if str(row["population"]) == population
            and int(row["k"]) == k
            and str(row["path_id"]) == path_id
        ]
        result: Dict[str, Any] = {
            "path_id": path_id,
            "population": population,
            "k": k,
            "stochastic_seed_count": len(subset),
            "sampling_unit_note": (
                "Repeated stochastic evaluations of one fixed physical path; "
                "not independent path samples."
            ),
        }
        for metric in metrics:
            average, deviation = mean_std(number(row[metric]) for row in subset)
            result[f"mean_{metric}"] = average
            result[f"std_{metric}"] = deviation
        result["negative_robot_aware_delta_seed_rate"] = mean(
            number(row["internal_full_path_robot_aware_delta_score"]) < 0.0
            for row in subset
        )
        output.append(result)
    return output


def aggregate_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    populations = ("ordinary", "difficult", "combined_diagnostic")
    contribution_metrics = tuple(
        sorted(
            {
                key
                for row in rows
                for key in row
                if "robot_score_contribution_" in key
            }
        )
    )
    metrics = (
        "full_path_safety_pass",
        "total_robot_aware_delta_score",
        "legacy_full_path_robot_aware_delta_score",
        "internal_full_path_robot_aware_delta_score",
        "sum_selected_local_delta_score",
        "full_path_recomputed_delta_score",
        "local_vs_full_delta_score_gap",
        "legacy_local_vs_full_delta_score_gap",
        "internal_local_vs_full_delta_score_gap",
        "terminal_joint_deviation_norm_rad",
        "terminal_joint_deviation_max_rad",
        "terminal_cartesian_deviation_from_prior_m",
        "legacy_terminal_boundary_step_contribution",
        "legacy_terminal_boundary_acceleration_contribution",
        "mean_join_joint_step_norm",
        "max_join_joint_step_norm",
        "max_join_absolute_joint_step",
        "mean_join_joint_acceleration_norm",
        "max_join_joint_acceleration_norm",
        "maximum_actual_internal_joint_step_rad",
        "cartesian_mean_error_delta",
        "cartesian_rms_error_delta",
        "cartesian_max_error_delta",
        "velocity_cost_delta",
        "acceleration_cost_delta",
        "jerk_cost_delta",
        "joint_limit_cost_delta",
        "singularity_cost_delta",
        "minimum_manipulability_delta",
        "correction_growth_slope",
        "boundary_anchoring_offset_growth_slope_per_step",
        "segment_mean_correction_growth_slope_per_step",
        "segment_max_correction_growth_slope_per_step",
        "cartesian_error_delta_growth_slope",
        "accepted_rollout_step_rate",
        "fallback_rate",
        *contribution_metrics,
    )
    for population in populations:
        for k in (1, 4, 8):
            subset = [
                row
                for row in rows
                if int(row["k"]) == k
                and (
                    population == "combined_diagnostic"
                    or str(row["population"]) == population
                )
            ]
            if not subset:
                continue
            result: Dict[str, Any] = {
                "population": population,
                "k": k,
                "path_seed_run_count": len(subset),
                "unique_physical_path_count": len(
                    {str(row["path_id"]) for row in subset}
                ),
                "completed_seed_count": len(
                    {int(row["sampling_seed"]) for row in subset}
                ),
                "sampling_unit_note": (
                    "Seeds are repeated stochastic evaluations of the same "
                    "fixed path population."
                ),
            }
            for metric in metrics:
                average, deviation = mean_std(
                    number(row[metric]) for row in subset
                )
                result[f"mean_{metric}"] = average
                result[f"std_{metric}"] = deviation
            result["fraction_runs_robot_aware_improved"] = mean(
                number(row["internal_full_path_robot_aware_delta_score"]) < 0.0
                for row in subset
            )
            output.append(result)
    return output


def engineering_decision(
    seed_rows: Sequence[Mapping[str, Any]],
    missing_seeds: Sequence[int],
) -> Dict[str, Any]:
    complete = not missing_seeds and len(seed_rows) == len(FROZEN_SEEDS)
    every_seed_safe = complete and all(
        number(row["final_output_safety_pass_rate"]) == 1.0
        for row in seed_rows
    )
    mean_improved_fraction = mean(
        number(row["fraction_paths_internal_full_path_score_negative"])
        for row in seed_rows
    )
    mean_internal_score_delta = mean(
        number(row["mean_internal_full_path_robot_aware_delta_score"])
        for row in seed_rows
    )
    mean_legacy_score_delta = mean(
        number(row["mean_legacy_full_path_robot_aware_delta_score"])
        for row in seed_rows
    )
    mean_local_score_sum = mean(
        number(row["mean_sum_selected_local_delta_score"])
        for row in seed_rows
    )
    mean_full_recomputed_score = mean(
        number(row["mean_full_path_recomputed_delta_score"])
        for row in seed_rows
    )
    mean_local_full_gap = mean(
        number(row["mean_local_vs_full_delta_score_gap"])
        for row in seed_rows
    )
    mean_cartesian_delta = mean(
        number(row["mean_cartesian_mean_error_delta"]) for row in seed_rows
    )
    mean_boundary_offset_slope = mean(
        number(
            row[
                "mean_boundary_anchoring_offset_growth_slope_per_step"
            ]
        )
        for row in seed_rows
    )
    mean_segment_mean_slope = mean(
        number(
            row[
                "mean_segment_mean_correction_growth_slope_per_step"
            ]
        )
        for row in seed_rows
    )
    mean_segment_max_slope = mean(
        number(
            row[
                "mean_segment_max_correction_growth_slope_per_step"
            ]
        )
        for row in seed_rows
    )
    mean_error_delta_slope = mean(
        number(row["mean_cartesian_error_delta_growth_slope"])
        for row in seed_rows
    )
    maximum_actual_internal_joint_step = max(
        number(row["maximum_actual_internal_joint_step_rad"])
        for row in seed_rows
    )
    advance = bool(
        complete
        and every_seed_safe
        and mean_improved_fraction >= 12.0 / 20.0
        and mean_internal_score_delta < 0.0
        and mean_cartesian_delta <= 0.0
        and mean_segment_max_slope
        <= MAX_ALLOWED_SEGMENT_CORRECTION_GROWTH_RAD_PER_STEP
        and mean_error_delta_slope <= 0.0
        and maximum_actual_internal_joint_step
        <= MAXIMUM_INTERNAL_JOINT_STEP_RAD
    )
    return {
        "decision_population": "ordinary",
        "decision_k": DECISION_K,
        "required_seeds": list(FROZEN_SEEDS),
        "complete_five_seed_evaluation": complete,
        "every_seed_has_100_percent_finally_safe_ordinary_paths": every_seed_safe,
        "mean_fraction_ordinary_paths_robot_aware_improved": mean_improved_fraction,
        "mean_fraction_ordinary_paths_internal_full_path_score_negative": (
            mean_improved_fraction
        ),
        "required_fraction_ordinary_paths_robot_aware_improved": 12.0 / 20.0,
        "mean_legacy_full_path_robot_aware_delta_score": (
            mean_legacy_score_delta
        ),
        "mean_internal_full_path_robot_aware_delta_score": (
            mean_internal_score_delta
        ),
        "mean_total_robot_aware_delta_score": mean_legacy_score_delta,
        "mean_sum_selected_local_delta_score": mean_local_score_sum,
        "mean_full_path_recomputed_delta_score": mean_full_recomputed_score,
        "mean_local_vs_full_delta_score_gap": mean_local_full_gap,
        "mean_cartesian_mean_error_delta": mean_cartesian_delta,
        "mean_boundary_anchoring_offset_growth_slope_per_step": (
            mean_boundary_offset_slope
        ),
        "mean_segment_mean_correction_growth_slope_per_step": (
            mean_segment_mean_slope
        ),
        "mean_segment_max_correction_growth_slope_per_step": (
            mean_segment_max_slope
        ),
        "maximum_allowed_segment_correction_growth_rad_per_step": (
            MAX_ALLOWED_SEGMENT_CORRECTION_GROWTH_RAD_PER_STEP
        ),
        "mean_cartesian_error_delta_growth_slope": mean_error_delta_slope,
        "maximum_actual_internal_joint_step_rad": (
            maximum_actual_internal_joint_step
        ),
        "maximum_allowed_internal_joint_step_rad": (
            MAXIMUM_INTERNAL_JOINT_STEP_RAD
        ),
        "maximum_actual_internal_joint_step_gate_pass": bool(
            maximum_actual_internal_joint_step
            <= MAXIMUM_INTERNAL_JOINT_STEP_RAD
        ),
        "advance_to_closed_loop": advance,
        "classification": (
            "V8_ANCHORED_ADVANCE_TO_CLOSED_LOOP"
            if advance
            else (
                "V8_ANCHORED_INCOMPLETE"
                if not complete
                else "V8_ANCHORED_ENGINEERING_HOLD"
            )
        ),
        "decision_note": (
            "Provisional engineering rule over repeated stochastic evaluations "
            "of the fixed 20-path ordinary validation population; not a formal "
            "statistical test."
        ),
    }


def largest_positive_score_contribution(
    rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    decision_rows = [
        row
        for row in rows
        if str(row["population"]) == "ordinary" and int(row["k"]) == DECISION_K
    ]
    result: Dict[str, Any] = {}
    for mode in ("internal", "legacy"):
        prefix = f"{mode}_robot_score_contribution_"
        fields = sorted(
            {
                key
                for row in decision_rows
                for key in row
                if key.startswith(prefix)
                and key
                not in {
                    f"{prefix}sum",
                    f"{mode}_robot_score_decomposition_residual",
                }
            }
        )
        means = {
            field: mean(number(row[field]) for row in decision_rows)
            for field in fields
        }
        if not means:
            result[mode] = {
                "component": "",
                "mean_contribution": math.nan,
            }
            continue
        component, value = max(means.items(), key=lambda item: item[1])
        result[mode] = {
            "component": component.removeprefix(prefix),
            "field": component,
            "mean_contribution": value,
        }
    return result


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir or args.input_root / "summary"
    output_dir.mkdir(parents=True, exist_ok=True)
    run_rows, missing = load_runs(args.input_root, args.allow_incomplete)
    completed = sorted({int(row["sampling_seed"]) for row in run_rows})
    seed_rows = per_seed_rows(run_rows, completed)
    path_rows = per_path_rows(run_rows)
    aggregate = aggregate_rows(run_rows)
    decision = engineering_decision(seed_rows, missing)
    largest_contribution = largest_positive_score_contribution(run_rows)
    write_csv(output_dir / "anchored_multiseed_per_seed.csv", seed_rows)
    write_csv(output_dir / "anchored_multiseed_per_path.csv", path_rows)
    write_csv(output_dir / "anchored_multiseed_aggregate.csv", aggregate)
    summary = {
        "completed_seeds": completed,
        "missing_seeds": missing,
        "population": {
            "ordinary_path_count": ORDINARY_PATH_COUNT,
            "difficult_paths": ["path_0306", "path_0370"],
            "difficult_paths_affect_decision": False,
        },
        "score_semantics": {
            "legacy": (
                "Includes a diagnostic transition from the rollout endpoint "
                "to the strong-prior endpoint."
            ),
            "internal": (
                "Evaluates the physically executed complete trajectory "
                "without an artificial post-terminal transition."
            ),
        },
        "largest_mean_positive_score_contribution_ordinary_k8": (
            largest_contribution
        ),
        "engineering_decision": decision,
    }
    (output_dir / "anchored_multiseed_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    report_lines = [
        "Diffusion v8 anchored recursive multiseed summary",
        "",
        "Sampling unit:",
        "  Five stochastic evaluations of the same fixed 20 ordinary paths.",
        "  These are not 100 independent physical path samples.",
        "",
        "Decision population: ordinary paths, K=8",
        (
            "Legacy score: includes a diagnostic transition from rollout "
            "endpoint to strong-prior endpoint."
        ),
        (
            "Internal score: evaluates the physically executed complete "
            "trajectory without that artificial post-terminal transition."
        ),
        f"Completed seeds: {completed}",
        f"Missing seeds: {missing}",
        (
            "Largest mean positive score contribution (ordinary K=8): "
            f"{largest_contribution}"
        ),
        json.dumps(decision, indent=2, sort_keys=True),
        "",
        "Difficult paths path_0306/path_0370 are diagnostic only.",
    ]
    (output_dir / "anchored_multiseed_report.txt").write_text(
        "\n".join(report_lines) + "\n"
    )
    print(f"classification: {decision['classification']}")
    print(f"advance_to_closed_loop: {decision['advance_to_closed_loop']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
