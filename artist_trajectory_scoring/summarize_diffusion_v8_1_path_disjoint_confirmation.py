#!/usr/bin/env python3
"""Summarize frozen v8.1 path-disjoint confirmation seeds."""

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


DEFAULT_INPUT_ROOT = Path(
    "results/diffusion_v8_1_path_disjoint_confirmation_test30_seeds53_57"
)
DEFAULT_SEEDS = (53, 54, 55, 56, 57)
DECISION_K = 8
DEFAULT_EXPECTED_PATH_COUNT = 30
MAXIMUM_INTERNAL_JOINT_STEP_RAD = 0.20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--expected_path_count", type=int, default=DEFAULT_EXPECTED_PATH_COUNT)
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


def json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


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
    if not clean:
        return math.nan
    return float(np.percentile(np.asarray(clean, dtype=np.float64), q))


def load_runs(
    input_root: Path,
    seeds: Sequence[int],
    allow_incomplete: bool,
) -> Tuple[List[Dict[str, Any]], List[int], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    missing: List[int] = []
    reference: Optional[Dict[str, Any]] = None
    summary_integrity: Dict[str, Any] = {
        "summary_files_checked": [],
        "summary_cross_seed_fields_identical": True,
    }
    for seed in seeds:
        path = input_root / f"seed_{seed}" / "anchored_full_path_metrics.csv"
        if not path.is_file():
            missing.append(seed)
            continue
        summary_path = input_root / f"seed_{seed}" / "anchored_rollout_summary.json"
        if not summary_path.is_file():
            raise FileNotFoundError(f"Missing summary for seed {seed}: {summary_path}")
        summary = json.loads(summary_path.read_text())
        if int(summary.get("sampling_seed", -1)) != int(seed):
            raise ValueError(
                f"{summary_path}: summary sampling_seed={summary.get('sampling_seed')} "
                f"does not match directory seed {seed}"
            )
        if int(summary.get("diagnostic_subset_run", 1)) != 0:
            raise ValueError(f"{summary_path}: diagnostic_subset_run must be 0")
        if int(summary.get("path_disjoint_confirmation_population", 0)) != 1:
            raise ValueError(
                f"{summary_path}: path_disjoint_confirmation_population must be 1"
            )
        comparable = {
            key: summary.get(key)
            for key in (
                "checkpoint_state",
                "checkpoint_state_hash",
                "training_dataset_dir",
                "model_dir",
                "test_prior_npz",
                "path_manifest",
                "source_path_names",
                "evaluator_path_ids",
            )
        }
        if reference is None:
            reference = comparable
        elif comparable != reference:
            raise ValueError(
                f"{summary_path}: cross-seed artifact metadata differs from the first seed"
            )
        summary_integrity["summary_files_checked"].append(str(summary_path))
        for row_index, row in enumerate(read_csv(path)):
            if "sampling_seed" not in row:
                raise ValueError(f"{path}: row {row_index} lacks sampling_seed")
            recorded_seed = int(float(row["sampling_seed"]))
            if recorded_seed != int(seed):
                raise ValueError(
                    f"{path}: row {row_index} sampling_seed={recorded_seed} "
                    f"does not match directory seed {seed}"
                )
            rows.append(dict(row))
    if missing and not allow_incomplete:
        raise FileNotFoundError(f"Missing required path-disjoint seeds: {missing}")
    if not rows:
        raise FileNotFoundError("No completed path-disjoint metrics were found")
    summary_integrity["reference"] = reference or {}
    return rows, missing, summary_integrity


def key(row: Mapping[str, Any]) -> Tuple[int, str, str, int]:
    return (
        int(row["sampling_seed"]),
        str(row["source_split"]),
        str(row["source_path_name"]),
        int(row["k"]),
    )


def decision_rows(rows: Sequence[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    return [row for row in rows if int(row["k"]) == DECISION_K]


def assert_confirmation_integrity(
    rows: Sequence[Mapping[str, Any]],
    seeds: Sequence[int],
    expected_path_count: int,
    missing_seeds: Sequence[int],
    allow_incomplete: bool,
) -> Dict[str, Any]:
    expected_seeds = tuple(seeds)
    if tuple(expected_seeds) != DEFAULT_SEEDS and not allow_incomplete:
        raise ValueError(f"Formal confirmation requires seeds {list(DEFAULT_SEEDS)}")
    if expected_path_count != DEFAULT_EXPECTED_PATH_COUNT and not allow_incomplete:
        raise ValueError("Formal confirmation requires 30 paths per seed")
    selected = decision_rows(rows)
    by_key: Dict[Tuple[int, str, str, int], Mapping[str, Any]] = {}
    duplicates: List[Tuple[int, str, str, int]] = []
    for row in selected:
        row_key = key(row)
        if row_key in by_key:
            duplicates.append(row_key)
        by_key[row_key] = row
    if duplicates:
        raise ValueError(f"Duplicate confirmation keys: {duplicates[:10]}")
    if missing_seeds and not allow_incomplete:
        raise ValueError(f"Missing seeds prevent formal confirmation: {missing_seeds}")
    seeds_present = {item[0] for item in by_key}
    if seeds_present != set(expected_seeds) and not allow_incomplete:
        raise ValueError(
            f"Observed seeds {sorted(seeds_present)} do not match {list(expected_seeds)}"
        )
    if any(item[1] != "test" for item in by_key):
        raise ValueError("All decision rows must have source_split=test")
    if any(item[3] != DECISION_K for item in by_key):
        raise ValueError("All decision rows must be K=8")
    for row in selected:
        if int(float(row["path_disjoint_confirmation_population"])) != 1:
            raise ValueError("All formal rows must be path_disjoint_confirmation_population=1")
        if int(float(row["diagnostic_subset_run"])) != 0:
            raise ValueError("Diagnostic subset rows cannot be formally classified")
    path_sets = {}
    for seed in seeds_present:
        paths = {item[2] for item in by_key if item[0] == seed}
        path_sets[seed] = paths
        if len(paths) != expected_path_count and not allow_incomplete:
            raise ValueError(
                f"Seed {seed} has {len(paths)} source paths; expected {expected_path_count}"
            )
    unique_path_sets = {tuple(sorted(paths)) for paths in path_sets.values()}
    if len(unique_path_sets) != 1 and not allow_incomplete:
        raise ValueError("Source path sets differ across seeds")
    expected_keys = len(expected_seeds) * expected_path_count
    if len(by_key) != expected_keys and not allow_incomplete:
        raise ValueError(f"Observed {len(by_key)} keys; expected {expected_keys}")
    return {
        "required_seeds": list(expected_seeds),
        "observed_seeds": sorted(seeds_present),
        "expected_path_count_per_seed": expected_path_count,
        "unique_key_count": len(by_key),
        "expected_unique_key_count": expected_keys,
        "duplicate_keys": bool(duplicates),
        "identical_source_path_set_across_seeds": len(unique_path_sets) == 1,
        "all_source_split_test": all(item[1] == "test" for item in by_key),
        "all_population_flags_formal": all(
            int(float(row["path_disjoint_confirmation_population"])) == 1
            for row in selected
        ),
        "all_diagnostic_subset_flags_zero": all(
            int(float(row["diagnostic_subset_run"])) == 0 for row in selected
        ),
        "complete_five_seed_30_path_population": (
            not missing_seeds
            and tuple(expected_seeds) == DEFAULT_SEEDS
            and expected_path_count == DEFAULT_EXPECTED_PATH_COUNT
            and len(by_key) == len(DEFAULT_SEEDS) * DEFAULT_EXPECTED_PATH_COUNT
            and len(unique_path_sets) == 1
            and not duplicates
        ),
    }


def per_seed_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for seed in sorted({int(row["sampling_seed"]) for row in decision_rows(rows)}):
        subset = [row for row in decision_rows(rows) if int(row["sampling_seed"]) == seed]
        result: Dict[str, Any] = {
            "sampling_seed": seed,
            "source_split": "test",
            "k": DECISION_K,
            "path_count": len(subset),
        }
        for field in (
            "full_path_safety_pass",
            "internal_full_path_robot_aware_delta_score",
            "cartesian_mean_error_delta",
            "internal_robot_score_contribution_jerk",
            "accepted_rollout_step_rate",
            "fallback_rate",
            "history_aware_jerk_rejection_rate_among_v8_selectable",
            "maximum_actual_internal_joint_step_rad",
        ):
            average, deviation = mean_std(number(row[field]) for row in subset)
            result[f"mean_{field}"] = average
            result[f"std_{field}"] = deviation
        result["maximum_actual_internal_joint_step_rad"] = max(
            number(row["maximum_actual_internal_joint_step_rad"]) for row in subset
        )
        result["fraction_runs_internal_score_negative"] = mean(
            number(row["internal_full_path_robot_aware_delta_score"]) < 0.0
            for row in subset
        )
        output.append(result)
    return output


def per_path_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    selected = decision_rows(rows)
    for path_name in sorted({str(row["source_path_name"]) for row in selected}):
        subset = [row for row in selected if str(row["source_path_name"]) == path_name]
        output.append(
            {
                "source_split": "test",
                "source_path_name": path_name,
                "k": DECISION_K,
                "seed_count": len(subset),
                "mean_internal_score": mean(
                    number(row["internal_full_path_robot_aware_delta_score"])
                    for row in subset
                ),
                "maximum_internal_score": max(
                    number(row["internal_full_path_robot_aware_delta_score"])
                    for row in subset
                ),
                "negative_score_seed_rate": mean(
                    number(row["internal_full_path_robot_aware_delta_score"]) < 0.0
                    for row in subset
                ),
                "mean_cartesian_delta": mean(
                    number(row["cartesian_mean_error_delta"]) for row in subset
                ),
                "mean_jerk_contribution": mean(
                    number(row["internal_robot_score_contribution_jerk"])
                    for row in subset
                ),
                "mean_acceptance_rate": mean(
                    number(row["accepted_rollout_step_rate"]) for row in subset
                ),
                "maximum_internal_joint_step": max(
                    number(row["maximum_actual_internal_joint_step_rad"])
                    for row in subset
                ),
            }
        )
    return output


def aggregate_metrics(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    selected = decision_rows(rows)
    internal_scores = [
        number(row["internal_full_path_robot_aware_delta_score"]) for row in selected
    ]
    finite_jerk_rates = finite(
        number(row["history_aware_jerk_rejection_rate_among_v8_selectable"])
        for row in selected
    )
    return {
        "path_seed_run_count": len(selected),
        "unique_source_path_count": len({str(row["source_path_name"]) for row in selected}),
        "completed_seed_count": len({int(row["sampling_seed"]) for row in selected}),
        "all_final_trajectories_safe": all(
            number(row["full_path_safety_pass"]) == 1.0 for row in selected
        ),
        "maximum_actual_internal_joint_step_rad": max(
            number(row["maximum_actual_internal_joint_step_rad"]) for row in selected
        ),
        "mean_internal_full_path_robot_aware_delta_score": mean(internal_scores),
        "p95_internal_full_path_robot_aware_delta_score": percentile(internal_scores, 95.0),
        "worst_path_seed_internal_full_path_robot_aware_delta_score": max(internal_scores),
        "mean_cartesian_mean_error_delta": mean(
            number(row["cartesian_mean_error_delta"]) for row in selected
        ),
        "fraction_path_seed_runs_with_negative_internal_score": mean(
            score < 0.0 for score in internal_scores
        ),
        "mean_internal_jerk_contribution": mean(
            number(row["internal_robot_score_contribution_jerk"]) for row in selected
        ),
        "mean_accepted_rollout_step_rate": mean(
            number(row["accepted_rollout_step_rate"]) for row in selected
        ),
        "mean_fallback_rate": mean(number(row["fallback_rate"]) for row in selected),
        "mean_jerk_rejection_rate_among_v8_selectable_candidates": (
            mean(finite_jerk_rates) if finite_jerk_rates else math.nan
        ),
        "jerk_rejection_rate_nan_policy": (
            "Nonfinite per-path rates are excluded from this aggregate denominator; "
            "NaN is reported only if no finite values exist."
        ),
    }


def classify(integrity: Mapping[str, Any], aggregate: Mapping[str, Any]) -> Dict[str, Any]:
    gated_metrics = (
        "maximum_actual_internal_joint_step_rad",
        "mean_internal_full_path_robot_aware_delta_score",
        "mean_cartesian_mean_error_delta",
        "p95_internal_full_path_robot_aware_delta_score",
        "worst_path_seed_internal_full_path_robot_aware_delta_score",
    )
    nonfinite = [
        metric for metric in gated_metrics if not math.isfinite(number(aggregate[metric]))
    ]
    if nonfinite:
        raise ValueError(f"Cannot classify with nonfinite gated metrics: {nonfinite}")
    pass_criteria = bool(
        integrity["complete_five_seed_30_path_population"]
        and bool(aggregate["all_final_trajectories_safe"])
        and number(aggregate["maximum_actual_internal_joint_step_rad"])
        <= MAXIMUM_INTERNAL_JOINT_STEP_RAD
        and number(aggregate["mean_internal_full_path_robot_aware_delta_score"]) < 0.0
        and number(aggregate["mean_cartesian_mean_error_delta"]) <= 0.0
        and number(aggregate["p95_internal_full_path_robot_aware_delta_score"]) <= 0.0
        and number(aggregate["worst_path_seed_internal_full_path_robot_aware_delta_score"]) <= 0.0
    )
    return {
        "classification": (
            "V8_1_PATH_DISJOINT_CONFIRMATION_PASS"
            if pass_criteria
            else "V8_1_PATH_DISJOINT_CONFIRMATION_HOLD"
        ),
        "pass": pass_criteria,
        "criteria": {
            "complete_five_seed_30_path_population": bool(
                integrity["complete_five_seed_30_path_population"]
            ),
            "all_final_trajectories_safe": bool(
                aggregate["all_final_trajectories_safe"]
            ),
            "maximum_actual_internal_joint_step_le_0_20_rad": (
                number(aggregate["maximum_actual_internal_joint_step_rad"])
                <= MAXIMUM_INTERNAL_JOINT_STEP_RAD
            ),
            "mean_internal_score_lt_0": (
                number(aggregate["mean_internal_full_path_robot_aware_delta_score"])
                < 0.0
            ),
            "mean_cartesian_delta_le_0": (
                number(aggregate["mean_cartesian_mean_error_delta"]) <= 0.0
            ),
            "p95_internal_score_le_0": (
                number(aggregate["p95_internal_full_path_robot_aware_delta_score"])
                <= 0.0
            ),
            "worst_path_seed_internal_score_le_0": (
                number(aggregate["worst_path_seed_internal_full_path_robot_aware_delta_score"])
                <= 0.0
            ),
        },
        "correction_growth_slope_is_reported_not_gated": True,
    }


def save_plots(output_dir: Path, rows: Sequence[Mapping[str, Any]], paths: Sequence[Mapping[str, Any]]) -> None:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    selected = decision_rows(rows)
    seeds = sorted({int(row["sampling_seed"]) for row in selected})

    def per_seed_mean(field: str) -> List[float]:
        return [
            mean(number(row[field]) for row in selected if int(row["sampling_seed"]) == seed)
            for seed in seeds
        ]

    for field, ylabel, filename in (
        (
            "internal_full_path_robot_aware_delta_score",
            "mean internal full-path score",
            "per_seed_mean_internal_score.png",
        ),
        (
            "cartesian_mean_error_delta",
            "mean Cartesian delta (m)",
            "per_seed_mean_cartesian_delta.png",
        ),
    ):
        figure, axis = plt.subplots(figsize=(7, 4))
        axis.bar(np.arange(len(seeds)), per_seed_mean(field))
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_xticks(np.arange(len(seeds)), [str(seed) for seed in seeds])
        axis.set_xlabel("sampling seed")
        axis.set_ylabel(ylabel)
        figure.tight_layout()
        figure.savefig(str(plot_dir / filename), dpi=150)
        plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 4))
    x = np.arange(len(seeds))
    axis.plot(x, per_seed_mean("accepted_rollout_step_rate"), marker="o", label="accepted")
    axis.plot(x, per_seed_mean("fallback_rate"), marker="o", label="fallback")
    axis.set_xticks(x, [str(seed) for seed in seeds])
    axis.set_xlabel("sampling seed")
    axis.set_ylabel("rate")
    axis.legend()
    figure.tight_layout()
    figure.savefig(str(plot_dir / "per_seed_acceptance_fallback_rates.png"), dpi=150)
    plt.close(figure)

    path_names = [str(row["source_path_name"]) for row in paths]
    x_path = np.arange(len(paths))
    figure, axis = plt.subplots(figsize=(12, 4))
    axis.bar(x_path - 0.2, [number(row["mean_internal_score"]) for row in paths], width=0.4, label="mean")
    axis.bar(x_path + 0.2, [number(row["maximum_internal_score"]) for row in paths], width=0.4, label="max")
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xticks(x_path, path_names, rotation=90)
    axis.set_ylabel("internal score")
    axis.legend()
    figure.tight_layout()
    figure.savefig(str(plot_dir / "per_path_mean_and_max_internal_score.png"), dpi=150)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(12, 4))
    axis.bar(x_path, [number(row["negative_score_seed_rate"]) for row in paths])
    axis.set_xticks(x_path, path_names, rotation=90)
    axis.set_ylabel("negative-score seed rate")
    figure.tight_layout()
    figure.savefig(str(plot_dir / "per_path_negative_score_seed_rate.png"), dpi=150)
    plt.close(figure)

    for field, xlabel, filename in (
        (
            "internal_full_path_robot_aware_delta_score",
            "internal full-path score",
            "internal_score_distribution.png",
        ),
        (
            "cartesian_mean_error_delta",
            "Cartesian mean-error delta (m)",
            "cartesian_delta_distribution.png",
        ),
        (
            "internal_robot_score_contribution_jerk",
            "internal jerk contribution",
            "jerk_contribution_distribution.png",
        ),
    ):
        figure, axis = plt.subplots(figsize=(7, 4))
        axis.hist([number(row[field]) for row in selected], bins=24)
        axis.axvline(0.0, color="black", linewidth=0.8)
        axis.set_xlabel(xlabel)
        axis.set_ylabel("path-seed count")
        figure.tight_layout()
        figure.savefig(str(plot_dir / filename), dpi=150)
        plt.close(figure)


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir or args.input_root / "summary"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, missing, summary_integrity = load_runs(
        args.input_root, args.seeds, args.allow_incomplete
    )
    integrity = assert_confirmation_integrity(
        rows, args.seeds, args.expected_path_count, missing, args.allow_incomplete
    )
    per_seed = per_seed_rows(rows)
    per_path = per_path_rows(rows)
    aggregate = aggregate_metrics(rows)
    decision = classify(integrity, aggregate)
    write_csv(output_dir / "path_disjoint_per_seed.csv", per_seed)
    write_csv(output_dir / "path_disjoint_per_path.csv", per_path)
    write_csv(output_dir / "path_disjoint_aggregate.csv", [aggregate])
    save_plots(output_dir, rows, per_path)
    summary = {
        "input_root": str(args.input_root),
        "missing_seeds": missing,
        "integrity": integrity,
        "summary_integrity": summary_integrity,
        "aggregate": aggregate,
        "decision": decision,
        "reported_not_gated": [
            "fraction path-seed runs with negative internal score",
            "mean internal jerk contribution",
            "mean accepted rollout-step rate",
            "mean fallback rate",
            "mean jerk rejection rate among v8-selectable candidates",
            "correction-growth slopes",
        ],
    }
    (output_dir / "path_disjoint_confirmation_summary.json").write_text(
        json.dumps(json_safe(summary), indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    report = [
        "Diffusion v8.1 path-disjoint confirmation summary",
        f"classification: {decision['classification']}",
        f"missing seeds: {missing}",
        json.dumps(decision, indent=2, sort_keys=True),
    ]
    (output_dir / "path_disjoint_confirmation_report.txt").write_text(
        "\n".join(report) + "\n"
    )
    print(f"classification: {decision['classification']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
