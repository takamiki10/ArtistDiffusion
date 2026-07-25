#!/usr/bin/env python3
"""Summarize v8.1 anchored jerk-guard seeds with paired v8 comparison."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt


FROZEN_SEEDS = (43, 44, 45, 46, 47)
DECISION_K = 8
ORDINARY_PATH_COUNT = 20
MAXIMUM_INTERNAL_JOINT_STEP_RAD = 0.20
PAIRED_FIELDS = (
    "internal_full_path_robot_aware_delta_score",
    "internal_robot_score_contribution_jerk",
    "cartesian_mean_error_delta",
    "accepted_rollout_step_rate",
    "fallback_rate",
    "maximum_actual_internal_joint_step_rad",
)


def row_key(row: Mapping[str, Any]) -> Tuple[int, str, int, str]:
    return (
        int(row["sampling_seed"]),
        str(row["population"]),
        int(row["k"]),
        str(row["path_id"]),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input_root",
        type=Path,
        default=Path("results/diffusion_v8_1_anchored_recursive_multiseed"),
    )
    parser.add_argument(
        "--baseline_root",
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
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def finite(values: Iterable[float]) -> List[float]:
    return [value for value in values if math.isfinite(value)]


def mean_std(values: Iterable[float]) -> Tuple[float, float]:
    clean = finite(values)
    if not clean:
        return math.nan, math.nan
    return mean(clean), stdev(clean) if len(clean) > 1 else 0.0


def percentile(values: Iterable[float], q: float) -> float:
    clean = finite(values)
    return float(np.percentile(np.asarray(clean, dtype=np.float64), q)) if clean else math.nan


def load_runs(
    input_root: Path,
    allow_incomplete: bool,
    label: str,
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
        raise FileNotFoundError(f"Missing required {label} seeds: {missing}")
    if not rows:
        raise FileNotFoundError(f"No completed {label} seed metrics were found")
    return rows, missing


def decision_rows(rows: Sequence[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    return [
        row
        for row in rows
        if str(row["population"]) == "ordinary" and int(row["k"]) == DECISION_K
    ]


def assert_unique_keys(
    rows: Sequence[Mapping[str, Any]],
    label: str,
) -> Dict[Tuple[int, str, int, str], Mapping[str, Any]]:
    output: Dict[Tuple[int, str, int, str], Mapping[str, Any]] = {}
    duplicates: List[Tuple[int, str, int, str]] = []
    for row in rows:
        key = row_key(row)
        if key in output:
            duplicates.append(key)
        output[key] = row
    if duplicates:
        raise ValueError(f"{label} contains duplicate keys: {duplicates[:10]}")
    return output


def assert_decision_population_integrity(
    v8_1_rows: Sequence[Mapping[str, Any]],
    baseline_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    v8_1_decision = decision_rows(v8_1_rows)
    baseline_decision = decision_rows(baseline_rows)
    v8_1_by_key = assert_unique_keys(v8_1_decision, "v8.1 decision rows")
    baseline_by_key = assert_unique_keys(
        baseline_decision, "baseline v8 decision rows"
    )
    expected_seeds = set(FROZEN_SEEDS)
    expected_count = len(FROZEN_SEEDS) * ORDINARY_PATH_COUNT
    for label, by_key in (("v8.1", v8_1_by_key), ("baseline v8", baseline_by_key)):
        seeds = {key[0] for key in by_key}
        populations = {key[1] for key in by_key}
        k_values = {key[2] for key in by_key}
        if seeds != expected_seeds:
            raise ValueError(
                f"{label} decision seeds are {sorted(seeds)}, "
                f"expected {list(FROZEN_SEEDS)}"
            )
        if populations != {"ordinary"}:
            raise ValueError(f"{label} decision populations are {sorted(populations)}")
        if k_values != {DECISION_K}:
            raise ValueError(f"{label} decision K values are {sorted(k_values)}")
        if len(by_key) != expected_count:
            raise ValueError(
                f"{label} has {len(by_key)} decision keys, expected {expected_count}"
            )
        for seed in FROZEN_SEEDS:
            paths = {key[3] for key in by_key if key[0] == seed}
            if len(paths) != ORDINARY_PATH_COUNT:
                raise ValueError(
                    f"{label} seed {seed} has {len(paths)} unique paths, "
                    f"expected {ORDINARY_PATH_COUNT}"
                )
    v8_1_keys = set(v8_1_by_key)
    baseline_keys = set(baseline_by_key)
    if v8_1_keys != baseline_keys:
        missing_in_v8_1 = sorted(baseline_keys - v8_1_keys)
        missing_in_baseline = sorted(v8_1_keys - baseline_keys)
        raise ValueError(
            "Decision key sets differ: "
            f"missing_in_v8_1={missing_in_v8_1[:10]}, "
            f"missing_in_baseline={missing_in_baseline[:10]}"
        )
    return {
        "decision_population": "ordinary",
        "decision_k": DECISION_K,
        "required_seeds": list(FROZEN_SEEDS),
        "unique_v8_1_key_count": len(v8_1_keys),
        "unique_baseline_v8_key_count": len(baseline_keys),
        "paired_row_count": len(v8_1_keys),
        "unique_paths_per_seed": ORDINARY_PATH_COUNT,
        "exact_key_set_match": True,
        "duplicate_keys": False,
    }


def paired_rows(
    v8_1_rows: Sequence[Mapping[str, Any]],
    baseline_rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    assert_decision_population_integrity(v8_1_rows, baseline_rows)
    v8_1_by_key = assert_unique_keys(v8_1_rows, "v8.1 all rows")
    baseline_by_key = assert_unique_keys(baseline_rows, "baseline v8 all rows")
    if set(v8_1_by_key) != set(baseline_by_key):
        missing_in_v8_1 = sorted(set(baseline_by_key) - set(v8_1_by_key))
        missing_in_baseline = sorted(set(v8_1_by_key) - set(baseline_by_key))
        raise ValueError(
            "Full paired key sets differ: "
            f"missing_in_v8_1={missing_in_v8_1[:10]}, "
            f"missing_in_baseline={missing_in_baseline[:10]}"
        )
    output: List[Dict[str, Any]] = []
    for key in sorted(v8_1_by_key):
        row = v8_1_by_key[key]
        baseline = baseline_by_key[key]
        result: Dict[str, Any] = {
            "sampling_seed": key[0],
            "population": key[1],
            "k": key[2],
            "path_id": key[3],
            "sampling_unit_note": (
                "Paired repeated stochastic evaluation of one fixed physical "
                "path; seeds are not independent physical paths."
            ),
        }
        for field in PAIRED_FIELDS:
            new_value = number(row[field])
            old_value = number(baseline[field])
            result[f"v8_1_{field}"] = new_value
            result[f"baseline_v8_{field}"] = old_value
            result[f"paired_change_{field}"] = new_value - old_value
        result["v8_1_full_path_safety_pass"] = number(row["full_path_safety_pass"])
        result["baseline_v8_full_path_safety_pass"] = number(
            baseline["full_path_safety_pass"]
        )
        output.append(result)
    return output


def per_seed_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for seed in sorted({int(row["sampling_seed"]) for row in rows}):
        subset = [
            row
            for row in decision_rows(rows)
            if int(row["sampling_seed"]) == seed
        ]
        if not subset:
            continue
        result: Dict[str, Any] = {
            "sampling_seed": seed,
            "population": "ordinary",
            "k": DECISION_K,
            "path_count": len(subset),
            "final_output_safety_pass_rate": mean(
                number(row["full_path_safety_pass"]) for row in subset
            ),
            "fraction_paths_internal_full_path_score_negative": mean(
                number(row["internal_full_path_robot_aware_delta_score"]) < 0.0
                for row in subset
            ),
        }
        for field in (
            "internal_full_path_robot_aware_delta_score",
            "internal_robot_score_contribution_jerk",
            "cartesian_mean_error_delta",
            "accepted_rollout_step_rate",
            "fallback_rate",
            "maximum_actual_internal_joint_step_rad",
            "history_aware_jerk_rejection_rate",
            "history_aware_jerk_rejection_rate_all_evaluated",
            "history_aware_jerk_rejection_rate_among_v8_selectable",
            "selected_history_aware_jerk_delta_sum",
            "selected_history_aware_jerk_delta_max_including_fallback",
            "selected_history_aware_jerk_delta_max_accepted_only",
            "selected_history_aware_jerk_delta_mean_accepted_only",
        ):
            average, deviation = mean_std(number(row[field]) for row in subset)
            result[f"mean_{field}"] = average
            result[f"std_{field}"] = deviation
        result["maximum_actual_internal_joint_step_rad"] = max(
            number(row["maximum_actual_internal_joint_step_rad"]) for row in subset
        )
        output.append(result)
    return output


def aggregate_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for population in ("ordinary", "difficult", "combined_diagnostic"):
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
                "unique_physical_path_count": len({str(row["path_id"]) for row in subset}),
                "completed_seed_count": len({int(row["sampling_seed"]) for row in subset}),
                "sampling_unit_note": (
                    "Seeds are repeated stochastic evaluations of the same "
                    "fixed path population."
                ),
            }
            for field in (
                "full_path_safety_pass",
                "internal_full_path_robot_aware_delta_score",
                "internal_robot_score_contribution_jerk",
                "cartesian_mean_error_delta",
                "accepted_rollout_step_rate",
                "fallback_rate",
                "maximum_actual_internal_joint_step_rad",
                "history_aware_jerk_rejection_rate",
                "history_aware_jerk_rejection_rate_all_evaluated",
                "history_aware_jerk_rejection_rate_among_v8_selectable",
                "selected_history_aware_jerk_delta_sum",
                "selected_history_aware_jerk_delta_max_including_fallback",
                "selected_history_aware_jerk_delta_max_accepted_only",
                "selected_history_aware_jerk_delta_mean_accepted_only",
                "correction_growth_slope",
                "boundary_anchoring_offset_growth_slope_per_step",
                "segment_mean_correction_growth_slope_per_step",
                "segment_max_correction_growth_slope_per_step",
                "cartesian_error_delta_growth_slope",
            ):
                average, deviation = mean_std(number(row[field]) for row in subset)
                result[f"mean_{field}"] = average
                result[f"std_{field}"] = deviation
                result[f"p95_{field}"] = percentile(
                    (number(row[field]) for row in subset), 95.0
                )
            output.append(result)
    return output


def paired_aggregate(pairs: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    ordinary = [
        row
        for row in pairs
        if str(row["population"]) == "ordinary" and int(row["k"]) == DECISION_K
    ]
    if not ordinary:
        return {"paired_row_count": 0}
    internal_changes = [
        number(row["paired_change_internal_full_path_robot_aware_delta_score"])
        for row in ordinary
    ]
    jerk_changes = [
        number(row["paired_change_internal_robot_score_contribution_jerk"])
        for row in ordinary
    ]
    new_internal = [
        number(row["v8_1_internal_full_path_robot_aware_delta_score"])
        for row in ordinary
    ]
    old_internal = [
        number(row["baseline_v8_internal_full_path_robot_aware_delta_score"])
        for row in ordinary
    ]
    path_ids = sorted({str(row["path_id"]) for row in ordinary})
    new_path_means = {
        path_id: mean(
            number(row["v8_1_internal_full_path_robot_aware_delta_score"])
            for row in ordinary
            if str(row["path_id"]) == path_id
        )
        for path_id in path_ids
    }
    old_path_means = {
        path_id: mean(
            number(row["baseline_v8_internal_full_path_robot_aware_delta_score"])
            for row in ordinary
            if str(row["path_id"]) == path_id
        )
        for path_id in path_ids
    }
    return {
        "population": "ordinary",
        "k": DECISION_K,
        "paired_path_seed_count": len(ordinary),
        "unique_physical_path_count": len({str(row["path_id"]) for row in ordinary}),
        "completed_seed_count": len({int(row["sampling_seed"]) for row in ordinary}),
        "fraction_path_seed_pairs_with_lower_internal_score": mean(
            change < 0.0 for change in internal_changes
        ),
        "fraction_path_seed_pairs_with_lower_jerk_contribution": mean(
            change < 0.0 for change in jerk_changes
        ),
        "mean_paired_internal_score_change": mean(internal_changes),
        "mean_paired_jerk_contribution_change": mean(jerk_changes),
        "worst_path_mean_internal_score": max(new_path_means.values()),
        "baseline_worst_path_mean_internal_score": max(old_path_means.values()),
        "worst_path_seed_internal_score": max(new_internal),
        "baseline_worst_path_seed_internal_score": max(old_internal),
        "p95_internal_score": percentile(new_internal, 95.0),
        "baseline_p95_internal_score": percentile(old_internal, 95.0),
        "mean_paired_cartesian_mean_error_delta_change": mean(
            number(row["paired_change_cartesian_mean_error_delta"]) for row in ordinary
        ),
        "mean_cartesian_mean_error_delta": mean(
            number(row["v8_1_cartesian_mean_error_delta"]) for row in ordinary
        ),
        "maximum_actual_internal_joint_step_rad": max(
            number(row["v8_1_maximum_actual_internal_joint_step_rad"])
            for row in ordinary
        ),
        "mean_paired_accepted_rollout_step_rate_change": mean(
            number(row["paired_change_accepted_rollout_step_rate"])
            for row in ordinary
        ),
        "mean_paired_fallback_rate_change": mean(
            number(row["paired_change_fallback_rate"]) for row in ordinary
        ),
    }


def engineering_decision(
    v8_1_rows: Sequence[Mapping[str, Any]],
    pairs: Sequence[Mapping[str, Any]],
    missing_v8_1: Sequence[int],
    missing_baseline: Sequence[int],
) -> Dict[str, Any]:
    ordinary = decision_rows(v8_1_rows)
    paired = paired_aggregate(pairs)
    complete = (
        not missing_v8_1
        and not missing_baseline
        and len({int(row["sampling_seed"]) for row in ordinary}) == len(FROZEN_SEEDS)
        and len(ordinary) == len(FROZEN_SEEDS) * ORDINARY_PATH_COUNT
        and number(paired.get("paired_path_seed_count")) == len(FROZEN_SEEDS) * ORDINARY_PATH_COUNT
    )
    all_safe = complete and all(
        number(row["full_path_safety_pass"]) == 1.0 for row in ordinary
    )
    max_step = max(
        number(row["maximum_actual_internal_joint_step_rad"]) for row in ordinary
    ) if ordinary else math.nan
    mean_internal_change = number(paired.get("mean_paired_internal_score_change"))
    mean_jerk_change = number(paired.get("mean_paired_jerk_contribution_change"))
    p95_new = number(paired.get("p95_internal_score"))
    p95_old = number(paired.get("baseline_p95_internal_score"))
    mean_cartesian_delta = mean(
        number(row["cartesian_mean_error_delta"]) for row in ordinary
    ) if ordinary else math.nan
    success = bool(
        complete
        and all_safe
        and max_step <= MAXIMUM_INTERNAL_JOINT_STEP_RAD
        and mean_internal_change < 0.0
        and mean_jerk_change < 0.0
        and p95_new < p95_old
        and mean_cartesian_delta <= 0.0
    )
    return {
        "decision_population": "ordinary",
        "decision_k": DECISION_K,
        "required_development_seeds": list(FROZEN_SEEDS),
        "complete_paired_development_evaluation": complete,
        "paired_integrity_checks_passed": bool(complete),
        "all_final_trajectories_safe": all_safe,
        "maximum_actual_internal_joint_step_rad": max_step,
        "maximum_allowed_internal_joint_step_rad": MAXIMUM_INTERNAL_JOINT_STEP_RAD,
        "mean_paired_internal_score_change": mean_internal_change,
        "mean_paired_jerk_contribution_change": mean_jerk_change,
        "p95_internal_score": p95_new,
        "baseline_p95_internal_score": p95_old,
        "mean_cartesian_mean_error_delta": mean_cartesian_delta,
        "successful_engineering_improvement": success,
        "classification": (
            "V8_1_JERK_GUARD_ENGINEERING_IMPROVEMENT"
            if success
            else (
                "V8_1_JERK_GUARD_INCOMPLETE"
                if not complete
                else "V8_1_JERK_GUARD_ENGINEERING_HOLD"
            )
        ),
        "development_data_note": (
            "Seeds 43-47 became development data after inspecting v8 results. "
            "A final frozen v8.1 method needs fresh stochastic seeds such as "
            "48-52 and preferably additional path-disjoint trajectories."
        ),
    }


def save_paired_plots(output_dir: Path, pairs: Sequence[Mapping[str, Any]]) -> None:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    ordinary = [
        row
        for row in pairs
        if str(row["population"]) == "ordinary" and int(row["k"]) == DECISION_K
    ]
    if not ordinary:
        return
    labels = [f"{row['path_id']}:{row['sampling_seed']}" for row in ordinary]
    x = np.arange(len(ordinary), dtype=np.float64)
    for field, title, filename in (
        (
            "internal_full_path_robot_aware_delta_score",
            "internal full-path score",
            "paired_internal_full_path_score.png",
        ),
        (
            "internal_robot_score_contribution_jerk",
            "internal jerk contribution",
            "paired_internal_jerk_contribution.png",
        ),
        (
            "cartesian_mean_error_delta",
            "Cartesian mean-error delta",
            "paired_cartesian_mean_error_delta.png",
        ),
        (
            "accepted_rollout_step_rate",
            "accepted rollout-step rate",
            "paired_accepted_rollout_step_rate.png",
        ),
        (
            "fallback_rate",
            "fallback rate",
            "paired_fallback_rate.png",
        ),
    ):
        figure, axis = plt.subplots(figsize=(15, 4.5))
        axis.plot(
            x,
            [number(row[f"baseline_v8_{field}"]) for row in ordinary],
            marker="o",
            linewidth=1,
            label="v8 baseline",
        )
        axis.plot(
            x,
            [number(row[f"v8_1_{field}"]) for row in ordinary],
            marker="o",
            linewidth=1,
            label="v8.1 jerk guard",
        )
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_xticks(x, labels, rotation=90, fontsize=7)
        axis.set_ylabel(title)
        axis.legend()
        figure.tight_layout()
        figure.savefig(str(plot_dir / filename), dpi=150)
        plt.close(figure)

    aggregate = paired_aggregate(pairs)
    figure, axis = plt.subplots(figsize=(8, 4))
    names = (
        "internal score",
        "jerk contribution",
        "Cartesian delta",
        "accepted rate",
        "fallback rate",
    )
    fields = (
        "mean_paired_internal_score_change",
        "mean_paired_jerk_contribution_change",
        "mean_paired_cartesian_mean_error_delta_change",
        "mean_paired_accepted_rollout_step_rate_change",
        "mean_paired_fallback_rate_change",
    )
    axis.bar(np.arange(len(fields)), [number(aggregate.get(field)) for field in fields])
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xticks(np.arange(len(fields)), names, rotation=30, ha="right")
    axis.set_ylabel("v8.1 - v8 paired mean change")
    figure.tight_layout()
    figure.savefig(str(plot_dir / "paired_aggregate_changes.png"), dpi=150)
    plt.close(figure)


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir or args.input_root / "summary"
    output_dir.mkdir(parents=True, exist_ok=True)
    v8_1_rows, missing_v8_1 = load_runs(args.input_root, args.allow_incomplete, "v8.1")
    baseline_rows, missing_baseline = load_runs(
        args.baseline_root, args.allow_incomplete, "baseline v8"
    )
    paired_integrity = assert_decision_population_integrity(
        v8_1_rows, baseline_rows
    )
    pairs = paired_rows(v8_1_rows, baseline_rows)
    seed_rows = per_seed_rows(v8_1_rows)
    aggregate = aggregate_rows(v8_1_rows)
    paired = paired_aggregate(pairs)
    decision = engineering_decision(v8_1_rows, pairs, missing_v8_1, missing_baseline)
    write_csv(output_dir / "v8_1_anchored_multiseed_per_seed.csv", seed_rows)
    write_csv(output_dir / "v8_1_anchored_multiseed_aggregate.csv", aggregate)
    write_csv(output_dir / "v8_1_vs_v8_paired_path_seed_comparison.csv", pairs)
    write_csv(output_dir / "v8_1_vs_v8_paired_aggregate.csv", [paired])
    save_paired_plots(output_dir, pairs)
    summary = {
        "completed_v8_1_seeds": sorted({int(row["sampling_seed"]) for row in v8_1_rows}),
        "missing_v8_1_seeds": missing_v8_1,
        "missing_baseline_v8_seeds": missing_baseline,
        "paired_integrity": paired_integrity,
        "paired_comparison": paired,
        "engineering_decision": decision,
        "baseline_v8_result_preserved": (
            "The completed v8 engineering-hold result is not reclassified. "
            "v8.1 is a separate development comparison."
        ),
        "fresh_confirmation_required": (
            "Seeds 43-47 are development data after v8 inspection; confirm a "
            "frozen v8.1 with fresh seeds such as 48-52."
        ),
    }
    (output_dir / "v8_1_anchored_multiseed_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    report_lines = [
        "Diffusion v8.1 anchored recursive jerk-guard multiseed summary",
        "",
        "Sampling unit:",
        "  Paired repeated stochastic evaluations of fixed physical paths.",
        "  The five seeds are not independent physical path samples.",
        "",
        f"v8.1 missing seeds: {missing_v8_1}",
        f"baseline v8 missing seeds: {missing_baseline}",
        "Decision population: ordinary paths, K=8",
        "Baseline v8 engineering-hold result is preserved.",
        json.dumps(decision, indent=2, sort_keys=True),
    ]
    (output_dir / "v8_1_anchored_multiseed_report.txt").write_text(
        "\n".join(report_lines) + "\n"
    )
    print(f"classification: {decision['classification']}")
    print(f"successful_engineering_improvement: {decision['successful_engineering_improvement']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
