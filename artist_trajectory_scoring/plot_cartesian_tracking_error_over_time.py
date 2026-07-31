#!/usr/bin/env python3
"""Plot Cartesian position tracking error over time for IK, MLP, and pipeline."""

from __future__ import annotations

import argparse
import csv
import importlib
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

comparison = importlib.import_module(
    "compare_ik_mlp_pipeline_jerk_over_time"
)
benchmark = importlib.import_module(
    "benchmark_ik_mlp_pipeline_smoothness"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path_id", required=True)

    parser.add_argument(
        "--ik_root",
        type=Path,
        default=Path(
            "data/cartesian_expert_dataset_v3/"
            "cold_ik_test_timed/test"
        ),
    )
    parser.add_argument(
        "--mlp_root",
        type=Path,
        default=Path(
            "data/cartesian_expert_dataset_v3/"
            "mlp_v3_test_predictions"
        ),
    )
    parser.add_argument("--pipeline_root", type=Path, required=True)
    parser.add_argument(
        "--pipeline_seeds",
        nargs="+",
        type=int,
        default=[53, 54, 55, 56, 57],
    )

    parser.add_argument("--common_duration_s", type=float, default=10.0)
    parser.add_argument("--common_samples", type=int, default=100)

    parser.add_argument(
        "--mean_cartesian_threshold_m",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--max_cartesian_threshold_m",
        type=float,
        default=0.03,
    )

    parser.add_argument("--output_dir", type=Path, required=True)
    return parser.parse_args()


def tracking_error(
    context: object,
    q: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    positions, _ = benchmark.authoritative_fk(context, q)
    return np.linalg.norm(positions - target, axis=1)


def summarize(error: np.ndarray) -> dict[str, float]:
    return {
        "mean_error_m": float(np.mean(error)),
        "rms_error_m": float(np.sqrt(np.mean(np.square(error)))),
        "max_error_m": float(np.max(error)),
        "endpoint_error_m": float(error[-1]),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    ik_path = args.ik_root / args.path_id
    mlp_path = args.mlp_root / args.path_id

    if not ik_path.is_dir():
        raise FileNotFoundError(f"IK path not found: {ik_path}")
    if not mlp_path.is_dir():
        raise FileNotFoundError(f"MLP path not found: {mlp_path}")

    ik = comparison.load_trajectory(ik_path, "ik")
    mlp = comparison.load_trajectory(mlp_path, "mlp")

    discovered = benchmark.discover_pipeline_artifacts(
        args.pipeline_root,
        "*seed*",
        args.pipeline_seeds,
    )

    artifacts = discovered.get(args.path_id, [])
    artifacts = [
        artifact
        for artifact in artifacts
        if artifact.seed in set(args.pipeline_seeds)
        and artifact.accepted
    ]

    if not artifacts:
        raise RuntimeError(
            f"No accepted pipeline artifacts found for {args.path_id}"
        )

    found_seeds = {artifact.seed for artifact in artifacts}
    missing_seeds = sorted(set(args.pipeline_seeds) - found_seeds)

    if missing_seeds:
        print(f"Warning: missing accepted seeds: {missing_seeds}")

    context = benchmark.make_robot_context()

    ik_error: np.ndarray | None = None
    mlp_error: np.ndarray | None = None
    timestamps: np.ndarray | None = None
    pipeline_errors: list[np.ndarray] = []
    pipeline_seed_summaries: list[dict[str, float | int]] = []

    source_target = benchmark.target_from_trajectory(ik)

    for artifact in artifacts:
        pipeline = benchmark.load_pipeline_trajectory(artifact)

        comparison.validate_path_compatibility(
            {
                "ik": ik,
                "mlp": mlp,
                "pipeline": pipeline,
            },
            path_id=args.path_id,
        )

        benchmark.verify_targets(
            args.path_id,
            (ik, mlp),
            (pipeline,),
        )

        aligned = comparison.align_trajectories(
            {
                "ik": ik,
                "mlp": mlp,
                "pipeline": pipeline,
            },
            timing_policy="common_duration",
            common_duration_s=args.common_duration_s,
            common_samples=args.common_samples,
        )

        if not aligned.complete_trajectory_used:
            raise RuntimeError("A complete trajectory was not used")

        target = benchmark.resample_target(
            source_target,
            len(aligned.timestamps),
        )

        if timestamps is None:
            timestamps = aligned.timestamps

            ik_error = tracking_error(
                context,
                aligned.q["ik"],
                target,
            )
            mlp_error = tracking_error(
                context,
                aligned.q["mlp"],
                target,
            )

        current_pipeline_error = tracking_error(
            context,
            aligned.q["pipeline"],
            target,
        )

        pipeline_errors.append(current_pipeline_error)

        seed_summary = summarize(current_pipeline_error)
        seed_summary["seed"] = artifact.seed
        pipeline_seed_summaries.append(seed_summary)

    assert timestamps is not None
    assert ik_error is not None
    assert mlp_error is not None

    pipeline_matrix = np.vstack(pipeline_errors)

    pipeline_median = np.median(pipeline_matrix, axis=0)
    pipeline_q1 = np.quantile(pipeline_matrix, 0.25, axis=0)
    pipeline_q3 = np.quantile(pipeline_matrix, 0.75, axis=0)
    pipeline_min = np.min(pipeline_matrix, axis=0)
    pipeline_max = np.max(pipeline_matrix, axis=0)

    ik_summary = summarize(ik_error)
    mlp_summary = summarize(mlp_error)

    pipeline_path_summary = {
        key: float(
            np.median(
                [
                    float(row[key])
                    for row in pipeline_seed_summaries
                ]
            )
        )
        for key in (
            "mean_error_m",
            "rms_error_m",
            "max_error_m",
            "endpoint_error_m",
        )
    }

    # Save time-series data.
    csv_path = (
        args.output_dir
        / f"{args.path_id}_cartesian_error_over_time.csv"
    )

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "time_s",
                "ik_error_m",
                "mlp_error_m",
                "pipeline_median_error_m",
                "pipeline_q1_error_m",
                "pipeline_q3_error_m",
                "pipeline_min_error_m",
                "pipeline_max_error_m",
            ]
        )

        for index, time_s in enumerate(timestamps):
            writer.writerow(
                [
                    float(time_s),
                    float(ik_error[index]),
                    float(mlp_error[index]),
                    float(pipeline_median[index]),
                    float(pipeline_q1[index]),
                    float(pipeline_q3[index]),
                    float(pipeline_min[index]),
                    float(pipeline_max[index]),
                ]
            )

    # Save summary values.
    summary_path = (
        args.output_dir
        / f"{args.path_id}_cartesian_error_summary.csv"
    )

    with summary_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "method",
            "aggregation",
            "mean_error_m",
            "rms_error_m",
            "max_error_m",
            "endpoint_error_m",
            "tracking_eligible",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for method, aggregation, summary in (
            ("ik", "single trajectory", ik_summary),
            ("mlp", "single trajectory", mlp_summary),
            (
                "pipeline",
                "median metric across accepted seeds",
                pipeline_path_summary,
            ),
        ):
            eligible = (
                summary["mean_error_m"]
                <= args.mean_cartesian_threshold_m
                and summary["max_error_m"]
                <= args.max_cartesian_threshold_m
            )

            writer.writerow(
                {
                    "method": method,
                    "aggregation": aggregation,
                    **summary,
                    "tracking_eligible": eligible,
                }
            )

    # Convert to millimeters for presentation.
    ik_mm = 1000.0 * ik_error
    mlp_mm = 1000.0 * mlp_error
    pipeline_median_mm = 1000.0 * pipeline_median
    pipeline_q1_mm = 1000.0 * pipeline_q1
    pipeline_q3_mm = 1000.0 * pipeline_q3

    # Full-scale graph.
    fig, axis = plt.subplots(figsize=(10, 6))

    ik_line, = axis.plot(
        timestamps,
        ik_mm,
        linewidth=1.8,
        label=(
            f"IK "
            f"(RMS {1000 * ik_summary['rms_error_m']:.2f} mm)"
        ),
    )

    mlp_line, = axis.plot(
        timestamps,
        mlp_mm,
        linewidth=1.8,
        label=(
            f"MLP "
            f"(RMS {1000 * mlp_summary['rms_error_m']:.2f} mm)"
        ),
    )

    pipeline_line, = axis.plot(
        timestamps,
        pipeline_median_mm,
        linewidth=2.0,
        label=(
            "Proposed pipeline median "
            f"(RMS {1000 * pipeline_path_summary['rms_error_m']:.2f} mm)"
        ),
    )

    axis.fill_between(
        timestamps,
        pipeline_q1_mm,
        pipeline_q3_mm,
        alpha=0.20,
        color=pipeline_line.get_color(),
        label="Pipeline seed IQR",
    )

    # The maximum-error threshold is pointwise and can be shown directly.
    axis.axhline(
        1000.0 * args.max_cartesian_threshold_m,
        linestyle="--",
        linewidth=1.3,
        label=(
            "Maximum-error gate "
            f"({1000 * args.max_cartesian_threshold_m:.0f} mm)"
        ),
    )

    axis.set_xlabel("Standardized time (s)")
    axis.set_ylabel("Cartesian position error (mm)")
    axis.set_title(
        f"{args.path_id}: Cartesian Tracking Error Over Time\n"
        f"Complete trajectories standardized to "
        f"{args.common_duration_s:g} s"
    )
    axis.set_ylim(bottom=0)
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize=9)
    fig.tight_layout()

    full_png = (
        args.output_dir
        / f"{args.path_id}_cartesian_error_over_time.png"
    )
    full_pdf = (
        args.output_dir
        / f"{args.path_id}_cartesian_error_over_time.pdf"
    )

    fig.savefig(full_png, dpi=300, bbox_inches="tight")
    fig.savefig(full_pdf, bbox_inches="tight")
    plt.close(fig)

    # Zoomed graph for IK versus pipeline when MLP dominates the scale.
    zoom_limit_mm = max(
        1.2 * 1000.0 * args.max_cartesian_threshold_m,
        1.1 * float(
            np.quantile(
                np.concatenate(
                    [ik_mm, pipeline_median_mm, pipeline_q3_mm]
                ),
                0.99,
            )
        ),
    )

    fig, axis = plt.subplots(figsize=(10, 6))

    axis.plot(
        timestamps,
        ik_mm,
        linewidth=1.8,
        label="IK",
    )
    axis.plot(
        timestamps,
        mlp_mm,
        linewidth=1.4,
        label="MLP",
    )
    pipeline_line, = axis.plot(
        timestamps,
        pipeline_median_mm,
        linewidth=2.0,
        label="Proposed pipeline median",
    )
    axis.fill_between(
        timestamps,
        pipeline_q1_mm,
        pipeline_q3_mm,
        alpha=0.20,
        color=pipeline_line.get_color(),
        label="Pipeline seed IQR",
    )
    axis.axhline(
        1000.0 * args.max_cartesian_threshold_m,
        linestyle="--",
        linewidth=1.3,
        label="Maximum-error gate",
    )

    axis.set_xlabel("Standardized time (s)")
    axis.set_ylabel("Cartesian position error (mm)")
    axis.set_title(
        f"{args.path_id}: Cartesian Tracking Error Over Time "
        "(Tracking-Threshold Zoom)"
    )
    axis.set_ylim(0, zoom_limit_mm)
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize=9)
    fig.tight_layout()

    zoom_png = (
        args.output_dir
        / f"{args.path_id}_cartesian_error_over_time_zoom.png"
    )
    zoom_pdf = (
        args.output_dir
        / f"{args.path_id}_cartesian_error_over_time_zoom.pdf"
    )

    fig.savefig(zoom_png, dpi=300, bbox_inches="tight")
    fig.savefig(zoom_pdf, bbox_inches="tight")
    plt.close(fig)

    print("\nCartesian tracking summary")
    print("----------------------------")
    print(
        f"IK:       mean={1000 * ik_summary['mean_error_m']:.3f} mm, "
        f"RMS={1000 * ik_summary['rms_error_m']:.3f} mm, "
        f"max={1000 * ik_summary['max_error_m']:.3f} mm"
    )
    print(
        f"MLP:      mean={1000 * mlp_summary['mean_error_m']:.3f} mm, "
        f"RMS={1000 * mlp_summary['rms_error_m']:.3f} mm, "
        f"max={1000 * mlp_summary['max_error_m']:.3f} mm"
    )
    print(
        "Pipeline: "
        f"median seed mean={1000 * pipeline_path_summary['mean_error_m']:.3f} mm, "
        f"median seed RMS={1000 * pipeline_path_summary['rms_error_m']:.3f} mm, "
        f"median seed max={1000 * pipeline_path_summary['max_error_m']:.3f} mm"
    )

    print("\nPipeline seeds used:")
    for row in sorted(
        pipeline_seed_summaries,
        key=lambda value: int(value["seed"]),
    ):
        print(
            f"  seed {int(row['seed'])}: "
            f"RMS={1000 * float(row['rms_error_m']):.3f} mm, "
            f"max={1000 * float(row['max_error_m']):.3f} mm"
        )

    print("\nSaved:")
    print(full_png)
    print(full_pdf)
    print(zoom_png)
    print(zoom_pdf)
    print(csv_path)
    print(summary_path)


if __name__ == "__main__":
    main()
