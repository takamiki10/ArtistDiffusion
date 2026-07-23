#!/usr/bin/env python3
"""Aggregate focused v8 teacher-forced evaluations across sampling seeds."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, cast

import numpy as np
import pandas as pd


PER_SEED_FILE = "focused_multiseed_per_seed.csv"
AGGREGATE_FILE = "focused_multiseed_aggregate.csv"
AGGREGATE_JSON_FILE = "focused_multiseed_aggregate.json"
PER_PATH_FILE = "focused_multiseed_per_path.csv"
REPORT_FILE = "focused_multiseed_report.txt"
EXPECTED_K_VALUES = (1, 4, 8)
EXPECTED_FULL_PRIMARY_WINDOWS = 360
EXPECTED_FULL_DIFFICULT_WINDOWS = 36
DECISION_MIN_K8_RATE = 0.40
DECISION_REQUIRED_EXCEEDING_SEEDS = 4
DECISION_EXPECTED_SEED_COUNT = 5
DECISION_MIN_PATH_FRACTION = 0.75
FLOAT_ATOL = 1.0e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize focused v8 teacher-forced evaluations by sampling seed."
    )
    parser.add_argument("--results_root", type=Path, required=True)
    parser.add_argument("--checkpoint_state", type=str, required=True)
    parser.add_argument("--target_scale", type=float, required=True)
    parser.add_argument("--output_alpha", type=float, required=True)
    parser.add_argument("--k_values", type=int, nargs="+", required=True)
    parser.add_argument("--sampling_seeds", type=int, nargs="+", required=True)
    parser.add_argument("--historical_v7_rate", type=float, required=True)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.checkpoint_state.strip():
        raise ValueError("--checkpoint_state cannot be empty")
    for name in ("target_scale", "output_alpha"):
        value = float(getattr(args, name))
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"--{name} must be positive and finite")
    k_values = tuple(sorted(int(value) for value in args.k_values))
    if k_values != EXPECTED_K_VALUES or len(set(args.k_values)) != len(args.k_values):
        raise ValueError("--k_values must be exactly 1 4 8")
    if not args.sampling_seeds:
        raise ValueError("--sampling_seeds cannot be empty")
    if len(set(args.sampling_seeds)) != len(args.sampling_seeds):
        raise ValueError("--sampling_seeds cannot contain duplicates")
    historical = float(args.historical_v7_rate)
    if not np.isfinite(historical) or not 0.0 <= historical <= 1.0:
        raise ValueError("--historical_v7_rate must be finite and in [0, 1]")


def read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"{path} is empty")
    return frame


def sort_frame(frame: pd.DataFrame, by: str | Sequence[str]) -> pd.DataFrame:
    return cast(
        pd.DataFrame,
        frame.sort_values(by=by),  # pyright: ignore[reportCallIssue]
    )


def require_columns(
    frame: pd.DataFrame,
    required: Sequence[str],
    path: Path,
) -> None:
    missing = sorted(set(required) - set(str(column) for column in frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")


def configuration_mask(
    frame: pd.DataFrame,
    args: argparse.Namespace,
    k_value: int,
    subset: str,
) -> pd.Series:
    return cast(
        pd.Series,
        (frame["checkpoint_state"].astype(str) == args.checkpoint_state)
        & np.isclose(frame["target_scale"].astype(float), args.target_scale)
        & np.isclose(frame["output_alpha"].astype(float), args.output_alpha)
        & (frame["K"].astype(int) == int(k_value))
        & (frame["evaluation_subset"].astype(str) == subset),
    )


def unique_configuration_row(
    frame: pd.DataFrame,
    args: argparse.Namespace,
    k_value: int,
    subset: str,
    *,
    required: bool,
) -> Optional[Dict[str, Any]]:
    rows = cast(
        pd.DataFrame,
        frame.loc[
            cast(Any, configuration_mask(frame, args, k_value, subset)),
            :,
        ],
    )
    if rows.empty and not required:
        return None
    if len(rows) != 1:
        raise ValueError(
            f"Expected exactly one {subset} row for K={k_value}, found {len(rows)}"
        )
    return {str(key): value for key, value in rows.iloc[0].to_dict().items()}


def optional_rate(row: Optional[Mapping[str, Any]], column: str) -> float:
    if row is None:
        return math.nan
    return float(row[column])


def load_seed_rows(
    args: argparse.Namespace,
    sampling_seed: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], bool]:
    seed_dir = args.results_root / f"seed_{sampling_seed}"
    configuration_path = seed_dir / "configuration_summary.csv"
    path_path = seed_dir / "per_path_summary.csv"
    summary_path = seed_dir / "evaluation_summary.json"
    configuration = read_csv(configuration_path)
    per_path = read_csv(path_path)
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    with summary_path.open("r", encoding="utf-8") as handle:
        evaluation_summary = json.load(handle)
    if int(evaluation_summary.get("sampling_seed", sampling_seed)) != sampling_seed:
        raise ValueError(f"{summary_path} records the wrong sampling seed")
    require_columns(
        configuration,
        (
            "checkpoint_state",
            "sampling_seed",
            "target_scale",
            "output_alpha",
            "K",
            "evaluation_subset",
            "accepted_window_count",
            "total_window_count",
            "accepted_window_rate",
            "hard_safe_sample_rate",
            "selectable_sample_rate",
            "fallback_rate",
            "final_safe_window_rate",
            "mean_cartesian_improvement_all_windows_m",
            "mean_cartesian_improvement_accepted_windows_m",
            "mean_robot_aware_delta_score_accepted_windows",
            "paths_with_at_least_one_accepted_window",
            "fraction_paths_with_at_least_one_accepted_window",
        ),
        configuration_path,
    )
    require_columns(
        per_path,
        (
            "checkpoint_state",
            "sampling_seed",
            "target_scale",
            "output_alpha",
            "K",
            "population",
            "path_name",
            "window_count",
            "accepted_window_count",
            "accepted_window_rate",
        ),
        path_path,
    )
    configuration_seeds = {
        int(value) for value in configuration["sampling_seed"].tolist()
    }
    path_seeds = {int(value) for value in per_path["sampling_seed"].tolist()}
    if configuration_seeds != {sampling_seed} or path_seeds != {sampling_seed}:
        raise ValueError(
            f"{seed_dir} contains rows for unexpected sampling seeds: "
            f"configuration={sorted(configuration_seeds)}, "
            f"per_path={sorted(path_seeds)}"
        )

    seed_rows: List[Dict[str, Any]] = []
    path_rows: List[Dict[str, Any]] = []
    full_population = True
    for k_value in sorted(args.k_values):
        primary = unique_configuration_row(
            configuration, args, k_value, "primary_all", required=True
        )
        assert primary is not None
        target_covered = unique_configuration_row(
            configuration,
            args,
            k_value,
            "primary_target_covered",
            required=False,
        )
        zero_target = unique_configuration_row(
            configuration,
            args,
            k_value,
            "primary_zero_target",
            required=False,
        )
        difficult = unique_configuration_row(
            configuration,
            args,
            k_value,
            "difficult_no_target",
            required=False,
        )
        primary_total = int(primary["total_window_count"])
        difficult_total = int(difficult["total_window_count"]) if difficult else 0
        full_population &= (
            primary_total == EXPECTED_FULL_PRIMARY_WINDOWS
            and difficult_total == EXPECTED_FULL_DIFFICULT_WINDOWS
        )
        seed_rows.append(
            {
                "sampling_seed": sampling_seed,
                "checkpoint_state": args.checkpoint_state,
                "target_scale": float(args.target_scale),
                "output_alpha": float(args.output_alpha),
                "K": int(k_value),
                "accepted_window_count": int(primary["accepted_window_count"]),
                "total_window_count": primary_total,
                "accepted_window_rate": float(primary["accepted_window_rate"]),
                "target_covered_accepted_rate": optional_rate(
                    target_covered, "accepted_window_rate"
                ),
                "zero_target_accepted_rate": optional_rate(
                    zero_target, "accepted_window_rate"
                ),
                "difficult_path_accepted_rate": optional_rate(
                    difficult, "accepted_window_rate"
                ),
                "hard_safe_sample_rate": float(primary["hard_safe_sample_rate"]),
                "selectable_sample_rate": float(primary["selectable_sample_rate"]),
                "fallback_rate": float(primary["fallback_rate"]),
                "final_safe_window_rate": float(primary["final_safe_window_rate"]),
                "mean_cartesian_improvement_all_windows_m": float(
                    primary["mean_cartesian_improvement_all_windows_m"]
                ),
                "mean_cartesian_improvement_accepted_windows_m": float(
                    primary["mean_cartesian_improvement_accepted_windows_m"]
                ),
                "mean_robot_aware_delta_score_accepted_windows": float(
                    primary["mean_robot_aware_delta_score_accepted_windows"]
                ),
                "paths_with_at_least_one_accepted_window": int(
                    primary["paths_with_at_least_one_accepted_window"]
                ),
                "fraction_paths_with_at_least_one_accepted_window": float(
                    primary["fraction_paths_with_at_least_one_accepted_window"]
                ),
                "primary_population_complete": int(
                    primary_total == EXPECTED_FULL_PRIMARY_WINDOWS
                ),
                "difficult_population_complete": int(
                    difficult_total == EXPECTED_FULL_DIFFICULT_WINDOWS
                ),
            }
        )

        path_mask = (
            (per_path["checkpoint_state"].astype(str) == args.checkpoint_state)
            & np.isclose(per_path["target_scale"].astype(float), args.target_scale)
            & np.isclose(per_path["output_alpha"].astype(float), args.output_alpha)
            & (per_path["K"].astype(int) == int(k_value))
        )
        selected_paths = cast(
            pd.DataFrame,
            per_path.loc[cast(Any, path_mask), :],
        )
        if selected_paths.empty:
            raise ValueError(
                f"{path_path} has no rows for {args.checkpoint_state}, K={k_value}"
            )
        for row in selected_paths.itertuples(index=False):
            accepted_count = int(cast(Any, row.accepted_window_count))
            path_rows.append(
                {
                    "sampling_seed": sampling_seed,
                    "checkpoint_state": args.checkpoint_state,
                    "target_scale": float(args.target_scale),
                    "output_alpha": float(args.output_alpha),
                    "K": int(k_value),
                    "population": str(row.population),
                    "path_name": str(row.path_name),
                    "accepted_window_count": accepted_count,
                    "total_window_count": int(cast(Any, row.window_count)),
                    "accepted_window_rate": float(
                        cast(Any, row.accepted_window_rate)
                    ),
                    "has_at_least_one_accepted_window": int(accepted_count > 0),
                }
            )
    return seed_rows, path_rows, full_population


def finite_mean(values: pd.Series) -> float:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    return float(np.mean(finite)) if len(finite) else math.nan


def aggregate_seed_rows(
    per_seed: pd.DataFrame,
    args: argparse.Namespace,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for k_value, group in per_seed.groupby("K", sort=True):
        frame = cast(pd.DataFrame, group)
        rates = np.asarray(frame["accepted_window_rate"], dtype=np.float64)
        accepted = np.asarray(frame["accepted_window_count"], dtype=np.int64)
        totals = np.asarray(frame["total_window_count"], dtype=np.int64)
        completed = len(frame)
        exceeding = int(np.sum(rates > float(args.historical_v7_rate)))
        safe = int(
            np.sum(
                np.isclose(
                    np.asarray(frame["final_safe_window_rate"], dtype=np.float64),
                    1.0,
                    rtol=0.0,
                    atol=FLOAT_ATOL,
                )
            )
        )
        rows.append(
            {
                "checkpoint_state": args.checkpoint_state,
                "target_scale": float(args.target_scale),
                "output_alpha": float(args.output_alpha),
                "K": int(cast(Any, k_value)),
                "completed_seed_count": completed,
                "mean_accepted_window_rate": float(np.mean(rates)),
                "sample_std_accepted_window_rate": (
                    float(np.std(rates, ddof=1)) if completed > 1 else math.nan
                ),
                "minimum_accepted_window_rate": float(np.min(rates)),
                "maximum_accepted_window_rate": float(np.max(rates)),
                "median_accepted_window_rate": float(np.median(rates)),
                "mean_accepted_window_count": float(np.mean(accepted)),
                "pooled_accepted_window_count": int(np.sum(accepted)),
                "pooled_total_window_count": int(np.sum(totals)),
                "pooled_accepted_window_rate": float(
                    np.sum(accepted) / np.sum(totals)
                ),
                "historical_v7_rate": float(args.historical_v7_rate),
                "seeds_exceeding_historical_v7_count": exceeding,
                "seeds_exceeding_historical_v7_fraction": exceeding / completed,
                "seeds_with_final_safe_rate_one_count": safe,
                "seeds_with_final_safe_rate_one_fraction": safe / completed,
                "mean_target_covered_accepted_rate": finite_mean(
                    frame["target_covered_accepted_rate"]
                ),
                "mean_zero_target_accepted_rate": finite_mean(
                    frame["zero_target_accepted_rate"]
                ),
                "mean_difficult_path_accepted_rate": finite_mean(
                    frame["difficult_path_accepted_rate"]
                ),
                "mean_hard_safe_sample_rate": finite_mean(
                    frame["hard_safe_sample_rate"]
                ),
                "mean_selectable_sample_rate": finite_mean(
                    frame["selectable_sample_rate"]
                ),
                "mean_fallback_rate": finite_mean(frame["fallback_rate"]),
                "mean_final_safe_window_rate": finite_mean(
                    frame["final_safe_window_rate"]
                ),
                "mean_fraction_paths_with_at_least_one_accepted_window": finite_mean(
                    frame["fraction_paths_with_at_least_one_accepted_window"]
                ),
            }
        )
    return pd.DataFrame(rows)


def add_path_stability(path_rows: pd.DataFrame) -> pd.DataFrame:
    stability = (
        path_rows.groupby(["K", "population", "path_name"], sort=True)
        .agg(
            mean_accepted_rate_across_seeds=("accepted_window_rate", "mean"),
            minimum_accepted_rate_across_seeds=("accepted_window_rate", "min"),
            maximum_accepted_rate_across_seeds=("accepted_window_rate", "max"),
            seeds_with_at_least_one_accepted_window=(
                "has_at_least_one_accepted_window",
                "sum",
            ),
            completed_seed_count=("sampling_seed", "nunique"),
        )
        .reset_index()
    )
    return cast(
        pd.DataFrame,
        path_rows.merge(
            stability,
            on=["K", "population", "path_name"],
            how="left",
            validate="many_to_one",
        ),
    )


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    return value


def atomic_json(value: Mapping[str, Any], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(json_safe(value), handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def format_rate(value: float) -> str:
    return "not available" if not np.isfinite(value) else f"{100.0 * value:.2f}%"


def build_report(
    per_seed: pd.DataFrame,
    aggregate: pd.DataFrame,
    args: argparse.Namespace,
    full_population: bool,
) -> Tuple[str, Dict[str, Any]]:
    k8_rows = cast(pd.DataFrame, aggregate.loc[aggregate["K"] == 8, :])
    k1_rows = cast(pd.DataFrame, aggregate.loc[aggregate["K"] == 1, :])
    if len(k8_rows) != 1 or len(k1_rows) != 1:
        raise ValueError("Focused report requires exactly one aggregate K=1 and K=8 row")
    k8 = k8_rows.iloc[0].to_dict()
    k1 = k1_rows.iloc[0].to_dict()
    completed = int(k8["completed_seed_count"])
    k8_mean = float(k8["mean_accepted_window_rate"])
    exceeding = int(k8["seeds_exceeding_historical_v7_count"])
    every_seed_safe = (
        int(k8["seeds_with_final_safe_rate_one_count"]) == completed
        and completed > 0
    )
    path_fraction = float(
        k8["mean_fraction_paths_with_at_least_one_accepted_window"]
    )
    broad_path_improvement = path_fraction >= DECISION_MIN_PATH_FRACTION
    complete_five_seed_confirmation = completed == DECISION_EXPECTED_SEED_COUNT
    advance = bool(
        complete_five_seed_confirmation
        and k8_mean >= DECISION_MIN_K8_RATE
        and exceeding >= DECISION_REQUIRED_EXCEEDING_SEEDS
        and every_seed_safe
        and path_fraction >= DECISION_MIN_PATH_FRACTION
    )
    decision = {
        "rule_type": "provisional engineering decision rule",
        "advance_to_anchored_rollout": advance,
        "complete_five_seed_confirmation": complete_five_seed_confirmation,
        "criteria": {
            "K8_mean_accepted_rate_at_least_0_40": k8_mean >= DECISION_MIN_K8_RATE,
            "at_least_4_of_5_seeds_exceed_historical_v7": (
                complete_five_seed_confirmation
                and exceeding >= DECISION_REQUIRED_EXCEEDING_SEEDS
            ),
            "every_seed_final_safe_window_rate_is_1": every_seed_safe,
            "mean_path_coverage_at_least_0_75": (
                broad_path_improvement
            ),
        },
    }
    target_rate = float(k8["mean_target_covered_accepted_rate"])
    zero_rate = float(k8["mean_zero_target_accepted_rate"])
    difficult_rate = float(k8["mean_difficult_path_accepted_rate"])
    k1_mean = float(k1["mean_accepted_window_rate"])
    lines = [
        "Diffusion v8 focused multi-seed confirmation",
        "",
        f"Checkpoint state: {args.checkpoint_state}",
        f"Target scale: {float(args.target_scale):g}",
        f"Output alpha: {float(args.output_alpha):g}",
        f"Sampling seeds: {' '.join(str(value) for value in args.sampling_seeds)}",
        f"Population: {'full 360 primary + 36 difficult windows' if full_population else 'smoke-test/limited population'}",
        "",
        "K=8 findings",
        f"1. Mean accepted rate: {format_rate(k8_mean)}; historical v7: "
        f"{format_rate(float(args.historical_v7_rate))}; exceeds v7: "
        f"{k8_mean > float(args.historical_v7_rate)}.",
        f"2. Seeds exceeding v7: {exceeding}/{completed}.",
        f"3. Final safe-window rate is 100% for every seed: {every_seed_safe}.",
        f"4. Mean fraction of paths with an accepted window: "
        f"{format_rate(path_fraction)}; improvements appear across many paths "
        f"under the 75% engineering threshold: {broad_path_improvement}.",
        f"5. K=1 mean accepted rate is {format_rate(k1_mean)} versus "
        f"K=8 {format_rate(k8_mean)}.",
        f"6. Target-covered mean accepted rate is {format_rate(target_rate)}; "
        f"zero-target mean accepted rate is {format_rate(zero_rate)}.",
        f"7. Difficult-path mean accepted rate is {format_rate(difficult_rate)}. "
        "The difficult population is a separate stress test.",
        f"8. Advance to anchored recursive rollout under the provisional "
        f"engineering rule: {advance}.",
        "",
        "Decision rule",
        "advance_to_anchored_rollout = K8 mean >= 0.40 AND at least 4 of 5 "
        f"seeds exceed {float(args.historical_v7_rate):.6f} AND every seed is "
        "finally safe AND mean path coverage >= 0.75.",
        "This is an engineering decision rule, not a formal statistical "
        "significance test.",
        "",
        "Uncertainty note",
        "Windows from different sampling seeds are repeated evaluations of the "
        "same physical windows. They are not treated as statistically independent. "
        "The pooled accepted rate is descriptive only.",
    ]
    return "\n".join(lines) + "\n", decision


def summarize(args: argparse.Namespace) -> Dict[str, Any]:
    validate_args(args)
    args.results_root = args.results_root.expanduser().resolve()
    args.results_root.mkdir(parents=True, exist_ok=True)
    all_seed_rows: List[Dict[str, Any]] = []
    all_path_rows: List[Dict[str, Any]] = []
    population_flags: List[bool] = []
    for sampling_seed in args.sampling_seeds:
        seed_rows, path_rows, full_population = load_seed_rows(
            args, int(sampling_seed)
        )
        all_seed_rows.extend(seed_rows)
        all_path_rows.extend(path_rows)
        population_flags.append(full_population)
    per_seed = sort_frame(
        pd.DataFrame(all_seed_rows), ["sampling_seed", "K"]
    ).reset_index(drop=True)
    expected_rows = len(args.sampling_seeds) * len(args.k_values)
    if len(per_seed) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} per-seed rows, found {len(per_seed)}"
        )
    aggregate = sort_frame(
        aggregate_seed_rows(per_seed, args), "K"
    ).reset_index(drop=True)
    per_path = sort_frame(
        add_path_stability(pd.DataFrame(all_path_rows)),
        ["K", "population", "path_name", "sampling_seed"],
    ).reset_index(drop=True)
    full_population = all(population_flags)
    report, decision = build_report(per_seed, aggregate, args, full_population)

    atomic_csv(per_seed, args.results_root / PER_SEED_FILE)
    atomic_csv(aggregate, args.results_root / AGGREGATE_FILE)
    atomic_csv(per_path, args.results_root / PER_PATH_FILE)
    payload = {
        "configuration": {
            "checkpoint_state": args.checkpoint_state,
            "target_scale": float(args.target_scale),
            "output_alpha": float(args.output_alpha),
            "k_values": sorted(int(value) for value in args.k_values),
            "sampling_seeds": [int(value) for value in args.sampling_seeds],
            "historical_v7_rate": float(args.historical_v7_rate),
        },
        "full_population": full_population,
        "aggregate_by_K": aggregate.to_dict(orient="records"),
        "decision": decision,
        "pooled_rate_is_descriptive_only": True,
        "independence_warning": (
            "Repeated sampling seeds reuse the same physical windows; windows "
            "across seeds are not independent observations."
        ),
    }
    atomic_json(payload, args.results_root / AGGREGATE_JSON_FILE)
    report_path = args.results_root / REPORT_FILE
    temporary_report = report_path.with_suffix(report_path.suffix + ".tmp")
    temporary_report.write_text(report, encoding="utf-8")
    temporary_report.replace(report_path)
    return payload


def main() -> int:
    args = parse_args()
    payload = summarize(args)
    print(f"wrote focused multi-seed summary to {args.results_root}")
    print(
        "advance_to_anchored_rollout: "
        f"{payload['decision']['advance_to_anchored_rollout']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
