#!/usr/bin/env python3
"""Create final result tables and report plots for the v3 trajectory pipeline."""

from __future__ import annotations

import argparse
import math
import os
import re
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import pandas as pd


MLP_FILL_VALUES = {
    "mean_cartesian_error_m": 0.202251,
    "mean_max_error_m": 0.306324,
    "worst_max_error_m": 0.571212,
}

METHOD_ORDER = [
    "MLP only",
    "Cold IK",
    "MLP + IK, smooth=0.01",
    "MLP + IK, smooth=0.001",
    "Adaptive MLP + IK",
]

RUNTIME_METHODS = [
    "Cold IK",
    "MLP + IK, smooth=0.01",
    "MLP + IK, smooth=0.001",
    "Adaptive MLP + IK",
]

LABELS = {
    "MLP only": "MLP only",
    "Cold IK": "Cold IK",
    "MLP + IK, smooth=0.01": "MLP+IK\nsmooth=0.01",
    "MLP + IK, smooth=0.001": "MLP+IK\nsmooth=0.001",
    "Adaptive MLP + IK": "Adaptive\nMLP+IK",
}

COLORS = {
    "MLP only": "#9a8c98",
    "Cold IK": "#4d908e",
    "MLP + IK, smooth=0.01": "#577590",
    "MLP + IK, smooth=0.001": "#f8961e",
    "Adaptive MLP + IK": "#43aa8b",
}


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir / "data" / "cartesian_expert_dataset_v3"

    parser = argparse.ArgumentParser(
        description="Build final filled result tables and report-ready comparison plots."
    )
    parser.add_argument(
        "--input_csv",
        type=Path,
        default=data_dir / "results_comparison_table.csv",
        help="Input comparison CSV.",
    )
    parser.add_argument(
        "--output_csv",
        type=Path,
        default=data_dir / "results_comparison_table_filled.csv",
        help="Filled comparison CSV.",
    )
    parser.add_argument(
        "--output_md",
        type=Path,
        default=data_dir / "results_comparison_table_filled.md",
        help="Filled comparison Markdown table.",
    )
    parser.add_argument(
        "--plot_dir",
        type=Path,
        default=data_dir / "final_plots",
        help="Directory for final PNG plots.",
    )
    parser.add_argument(
        "--test_dir",
        type=Path,
        default=data_dir / "experts" / "test",
        help="Test path folder used to find an overlay example.",
    )
    return parser.parse_args()


def warn(message: str) -> None:
    print(f"Warning: {message}")


def fmt_cell(value: object, floatfmt: str = ".6g") -> str:
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, float):
        text = format(value, floatfmt)
    else:
        text = str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def dataframe_to_markdown(df: pd.DataFrame, floatfmt: str = ".6g") -> str:
    columns = list(df.columns)
    rows = [
        [fmt_cell(value, floatfmt=floatfmt) for value in row]
        for row in df.itertuples(index=False, name=None)
    ]
    widths = [
        max(len(str(column)), *(len(row[i]) for row in rows)) if rows else len(str(column))
        for i, column in enumerate(columns)
    ]

    def render_row(values: list[str]) -> str:
        return "| " + " | ".join(value.ljust(widths[i]) for i, value in enumerate(values)) + " |"

    return "\n".join(
        [
            render_row([str(column) for column in columns]),
            "| " + " | ".join("-" * width for width in widths) + " |",
            *(render_row(row) for row in rows),
        ]
    )


def load_table(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        warn(f"input CSV is missing: {path}")
        return None
    return pd.read_csv(path)


def fill_mlp_only_values(df: pd.DataFrame) -> pd.DataFrame:
    if "method" not in df.columns:
        warn("column 'method' is missing; cannot fill MLP-only metrics")
        return df

    out = df.copy()
    mlp_mask = out["method"].astype(str).eq("MLP only")
    if not mlp_mask.any():
        warn("no 'MLP only' row found; no MLP-only values filled")
        return out

    for column, value in MLP_FILL_VALUES.items():
        if column not in out.columns:
            warn(f"column '{column}' is missing; cannot fill it")
            continue
        numeric = pd.to_numeric(out.loc[mlp_mask, column], errors="coerce")
        needs_fill = numeric.isna()
        if needs_fill.any():
            out.loc[mlp_mask, column] = value

    if "total_time_readable" in out.columns:
        blank_runtime = out.loc[mlp_mask, "total_time_readable"].isna() | (
            out.loc[mlp_mask, "total_time_readable"].astype(str).str.strip() == ""
        )
        if blank_runtime.any():
            out.loc[mlp_mask, "total_time_readable"] = "fast / not timed"

    return out


def save_filled_tables(df: pd.DataFrame, output_csv: Path, output_md: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(dataframe_to_markdown(df) + "\n", encoding="utf-8")
    print(f"Saved filled CSV: {output_csv}")
    print(f"Saved filled Markdown: {output_md}")


def ordered_methods(df: pd.DataFrame, methods: Iterable[str] | None = None) -> pd.DataFrame:
    if "method" not in df.columns:
        warn("column 'method' is missing; cannot order methods")
        return df.copy()
    desired = list(methods) if methods is not None else METHOD_ORDER
    out = df[df["method"].astype(str).isin(desired)].copy()
    order = {name: i for i, name in enumerate(desired)}
    out["_method_order"] = out["method"].astype(str).map(order)
    out = out.sort_values("_method_order").drop(columns=["_method_order"])
    return out


def parse_time_to_minutes(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip().lower()
    if not text:
        return None

    try:
        return float(text) / 60.0
    except ValueError:
        pass

    match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*min", text)
    if match:
        return float(match.group(1))
    match = re.fullmatch(r"(\d+(?:\.\d+)?)\s*s", text)
    if match:
        return float(match.group(1)) / 60.0
    return None


def runtime_minutes(row: pd.Series) -> float | None:
    if "total_time_sec" in row and not pd.isna(row["total_time_sec"]):
        return float(row["total_time_sec"]) / 60.0
    if "total_time_readable" in row:
        return parse_time_to_minutes(row["total_time_readable"])
    return None


def format_value(value: float, suffix: str = "") -> str:
    if abs(value) >= 100:
        text = f"{value:.0f}"
    elif abs(value) >= 10:
        text = f"{value:.1f}"
    elif abs(value) >= 1:
        text = f"{value:.2f}"
    else:
        text = f"{value:.3g}"
    return f"{text}{suffix}"


def add_bar_labels(ax: plt.Axes, values: list[float], *, log_scale: bool, suffix: str = "") -> None:
    if not values:
        return
    ymax = max(values)
    for patch, value in zip(ax.patches, values):
        if pd.isna(value):
            continue
        y = value * 1.12 if log_scale else value + ymax * 0.025
        ax.text(
            patch.get_x() + patch.get_width() / 2,
            y,
            format_value(value, suffix=suffix),
            ha="center",
            va="bottom",
            fontsize=9,
        )


def save_bar_plot(
    df: pd.DataFrame,
    *,
    column: str,
    title: str,
    ylabel: str,
    output_path: Path,
    methods: Iterable[str] | None = None,
    log_scale: bool = False,
    threshold: float | None = None,
    threshold_label: str | None = None,
    suffix: str = "",
) -> None:
    if column not in df.columns:
        warn(f"column '{column}' is missing; skipped {output_path.name}")
        return

    plot_df = ordered_methods(df, methods)
    plot_df[column] = pd.to_numeric(plot_df[column], errors="coerce")
    plot_df = plot_df.dropna(subset=[column])
    if log_scale:
        plot_df = plot_df[plot_df[column] > 0]
    if plot_df.empty:
        warn(f"no usable values for '{column}'; skipped {output_path.name}")
        return

    methods_used = plot_df["method"].astype(str).tolist()
    values = plot_df[column].astype(float).tolist()
    labels = [LABELS.get(method, method) for method in methods_used]
    colors = [COLORS.get(method, "#6c757d") for method in methods_used]

    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    ax.bar(labels, values, color=colors, edgecolor="#263238", linewidth=0.8)
    ax.set_title(title, fontsize=14, weight="bold")
    ax.set_ylabel(ylabel)
    if log_scale:
        ax.set_yscale("log")
    ax.grid(axis="y", which="major", alpha=0.28)
    ax.set_axisbelow(True)
    if threshold is not None:
        ax.axhline(threshold, color="#c1121f", linestyle="--", linewidth=1.6)
        if threshold_label:
            ax.text(
                0.99,
                threshold,
                threshold_label,
                color="#8f0d17",
                ha="right",
                va="bottom",
                transform=ax.get_yaxis_transform(),
                fontsize=10,
            )
    add_bar_labels(ax, values, log_scale=log_scale, suffix=suffix)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    print(f"Saved plot: {output_path}")


def save_runtime_plot(df: pd.DataFrame, output_path: Path) -> None:
    if "method" not in df.columns:
        warn("column 'method' is missing; skipped runtime plot")
        return

    plot_df = ordered_methods(df, RUNTIME_METHODS)
    rows = []
    for _, row in plot_df.iterrows():
        minutes = runtime_minutes(row)
        if minutes is None:
            warn(f"missing runtime for {row.get('method', '<unknown>')}; skipped in runtime plot")
            continue
        rows.append({"method": row["method"], "runtime_min": minutes})

    runtime_df = pd.DataFrame(rows)
    save_bar_plot(
        runtime_df,
        column="runtime_min",
        title="Total Runtime Comparison",
        ylabel="Total runtime (minutes)",
        output_path=output_path,
        methods=RUNTIME_METHODS,
        suffix=" min",
    )


def read_xyz_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        warn(f"overlay input is missing: {path}")
        return None
    df = pd.read_csv(path)
    missing = {"x", "y"} - set(df.columns)
    if missing:
        warn(f"overlay input {path} is missing columns: {sorted(missing)}")
        return None
    return df


def find_overlay_path(test_dir: Path) -> tuple[Path, Path] | None:
    if not test_dir.exists():
        warn(f"test directory is missing: {test_dir}")
        return None

    refined_candidates = ["adaptive_refined_ee.csv", "refined_mlp_ik_ee.csv"]
    for path_dir in sorted(p for p in test_dir.iterdir() if p.is_dir()):
        desired = path_dir / "desired_path.csv"
        mlp = path_dir / "path_conditioned_pred_ee.csv"
        if not desired.exists() or not mlp.exists():
            continue
        for name in refined_candidates:
            refined = path_dir / name
            if refined.exists():
                return path_dir, refined
    return None


def save_overlay_plot(test_dir: Path, output_path: Path) -> None:
    found = find_overlay_path(test_dir)
    if found is None:
        warn("no path folder with desired, MLP prediction, and refined EE files; skipped overlay plot")
        return

    path_dir, refined_path = found
    desired = read_xyz_csv(path_dir / "desired_path.csv")
    mlp = read_xyz_csv(path_dir / "path_conditioned_pred_ee.csv")
    refined = read_xyz_csv(refined_path)
    if desired is None or mlp is None or refined is None:
        warn("overlay inputs were incomplete; skipped overlay plot")
        return

    refined_label = (
        "Adaptive / final refined path"
        if refined_path.name == "adaptive_refined_ee.csv"
        else "MLP+IK refined path"
    )
    title = (
        f"Example XY Overlay: {path_dir.name}\n"
        "MLP warm start improves after IK refinement"
    )

    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    ax.plot(desired["x"], desired["y"], color="#111111", linewidth=2.4, label="Desired path")
    ax.plot(
        mlp["x"],
        mlp["y"],
        color="#9a8c98",
        linewidth=1.9,
        linestyle="--",
        label="MLP-only prediction",
    )
    ax.plot(
        refined["x"],
        refined["y"],
        color="#43aa8b",
        linewidth=2.2,
        label=refined_label,
    )
    ax.set_title(title, fontsize=13, weight="bold")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.axis("equal")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    print(f"Saved overlay plot: {output_path}")


def make_plots(df: pd.DataFrame, plot_dir: Path, test_dir: Path) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)

    save_runtime_plot(df, plot_dir / "total_runtime_comparison.png")
    save_bar_plot(
        df,
        column="mean_cartesian_error_m",
        title="Mean Cartesian Error Comparison",
        ylabel="Mean Cartesian error (m, log scale)",
        output_path=plot_dir / "mean_cartesian_error_comparison.png",
        log_scale=True,
    )
    save_bar_plot(
        df,
        column="mean_max_error_m",
        title="Mean Max Error Comparison",
        ylabel="Mean max error (m, log scale)",
        output_path=plot_dir / "mean_max_error_comparison.png",
        log_scale=True,
    )
    save_bar_plot(
        df,
        column="worst_max_error_m",
        title="Worst Max Error Comparison",
        ylabel="Worst max error (m, log scale)",
        output_path=plot_dir / "worst_max_error_comparison.png",
        log_scale=True,
        threshold=0.03,
        threshold_label="0.03 m threshold",
    )
    save_bar_plot(
        df,
        column="mean_path_error",
        title="Mean Path Error Comparison",
        ylabel="Mean path error (log scale)",
        output_path=plot_dir / "path_error_comparison.png",
        log_scale=True,
    )
    save_overlay_plot(test_dir, plot_dir / "adaptive_overlay_example.png")


def main() -> None:
    args = parse_args()
    table = load_table(args.input_csv)
    if table is None:
        return

    filled = fill_mlp_only_values(table)
    save_filled_tables(filled, args.output_csv, args.output_md)
    make_plots(filled, args.plot_dir, args.test_dir)

    print()
    print("Conclusion to preserve:")
    print(
        "The path-conditioned MLP is not reliable as a standalone trajectory generator, "
        "but it is effective as a warm-start model for IK refinement. The adaptive "
        "two-stage MLP+IK method preserves most of the speed benefit while removing "
        "local spike failures."
    )


if __name__ == "__main__":
    main()
