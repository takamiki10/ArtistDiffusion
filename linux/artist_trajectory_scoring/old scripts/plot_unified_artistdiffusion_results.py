#!/usr/bin/env python3
"""Create presentation plots from unified ArtistDiffusion results."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_INPUT_CSV = Path("data/cartesian_expert_dataset_v3/unified_artistdiffusion_results_main.csv")
DEFAULT_OUTPUT_DIR = Path("data/cartesian_expert_dataset_v3/final_result_plots")

DIFFUSION_METHOD_KEYWORDS = (
    "Diffusion v1 best-of-K",
    "Diffusion v4 pure Gaussian",
    "Diffusion v4 MLP-prior refinement",
    "Diffusion v4 v1-prior refinement",
    "Noised-expert v4 reference",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot unified ArtistDiffusion result summaries.")
    parser.add_argument("--input_csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def readable_label(method: str) -> str:
    replacements = {
        "Diffusion v4 MLP-prior refinement t=25": "v4 MLP-prior refine",
        "Diffusion v4 v1-prior refinement t=25": "v4 v1-prior refine",
        "Diffusion v4 pure Gaussian t=25": "v4 pure Gaussian",
        "Noised-expert v4 reference t=25": "v4 noised-expert ref",
        "Diffusion v1 best-of-K": "v1 best-of-K",
        "Adaptive MLP + IK": "Adaptive MLP+IK",
    }
    label = replacements.get(method, method)
    return label.replace("_", " ")


def load_results(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing input CSV: {path}")
    df = pd.read_csv(path)
    if "method" not in df.columns:
        raise KeyError(f"{path} must contain a method column")
    for column in df.columns:
        if column not in ("method", "role", "notes"):
            df[column] = pd.to_numeric(df[column], errors="coerce")
    df["plot_label"] = df["method"].map(readable_label)
    return df


def filter_metric(df: pd.DataFrame, metric: str, plot_name: str) -> pd.DataFrame:
    if metric not in df.columns:
        print(f"WARNING: {plot_name}: missing column {metric}; skipping plot.")
        return pd.DataFrame()
    out = df[df[metric].notna()].copy()
    skipped = len(df) - len(out)
    if skipped:
        print(f"WARNING: {plot_name}: skipped {skipped} method(s) with missing/NaN {metric}.")
    return out


def annotate_bars(ax: plt.Axes, values: Sequence[float]) -> None:
    if not values:
        return
    ymax = max(values)
    offset = max(ymax * 0.015, 1e-6)
    for patch, value in zip(ax.patches, values):
        ax.text(
            patch.get_x() + patch.get_width() / 2.0,
            patch.get_height() + offset,
            f"{value:.3g}",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=0,
        )


def annotate_bars_log(ax: plt.Axes, values: Sequence[float]) -> None:
    for patch, value in zip(ax.patches, values):
        ax.text(
            patch.get_x() + patch.get_width() / 2.0,
            value * 1.08,
            f"{value:.3g}",
            ha="center",
            va="bottom",
            fontsize=8,
            rotation=0,
        )


def bar_plot(
    df: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    output_path: Path,
    log_scale: bool = False,
    color: str = "#4C78A8",
) -> Optional[Path]:
    plot_df = filter_metric(df, metric, output_path.name)
    if plot_df.empty:
        return None
    plot_df = plot_df.sort_values(metric, ascending=True)
    values = plot_df[metric].tolist()

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(plot_df["plot_label"], values, color=color)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if log_scale:
        positive = plot_df[plot_df[metric] > 0].copy()
        if len(positive) != len(plot_df):
            print(f"WARNING: {output_path.name}: skipped non-positive values for log scale.")
            plot_df = positive
            values = plot_df[metric].tolist()
            ax.clear()
            ax.bar(plot_df["plot_label"], values, color=color)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
        ax.set_yscale("log")
    ax.tick_params(axis="x", rotation=35)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    if log_scale:
        annotate_bars_log(ax, values)
    else:
        annotate_bars(ax, values)
        ax.margins(y=0.15)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def diffusion_subset(df: pd.DataFrame) -> pd.DataFrame:
    mask = pd.Series(False, index=df.index)
    for keyword in DIFFUSION_METHOD_KEYWORDS:
        mask = mask | df["method"].str.contains(keyword, regex=False, na=False)
    out = df[mask].copy()
    if out.empty:
        print("WARNING: no diffusion-related methods found for diffusion comparison plot.")
    return out


def drawing_cost_diffusion_subset(df: pd.DataFrame) -> pd.DataFrame:
    out = diffusion_subset(df)
    if "drawing_total_cost" not in out.columns:
        print("WARNING: missing drawing_total_cost column for diffusion drawing-cost plot.")
        return pd.DataFrame()
    out = out[out["drawing_total_cost"].notna()].copy()
    if out.empty:
        print("WARNING: no diffusion methods with drawing_total_cost available.")
    return out


def method_group(method: str) -> str:
    if method == "Adaptive MLP + IK":
        return "IK-assisted"
    if method == "IK expert":
        return "IK / reference"
    if method == "MLP-only":
        return "Learning baseline"
    if method == "Diffusion v1 best-of-K":
        return "Diffusion"
    if method == "Noised-expert v4 reference t=25":
        return "Diffusion reference"
    if method in (
        "Diffusion v4 MLP-prior refinement t=25",
        "Diffusion v4 v1-prior refinement t=25",
    ):
        return "Diffusion refinement"
    if "Diffusion" in method:
        return "Diffusion"
    return "Other"


def grouped_mean_cartesian_plot(df: pd.DataFrame, output_path: Path) -> Optional[Path]:
    plot_df = filter_metric(df, "mean_cartesian_error", output_path.name)
    if plot_df.empty:
        return None
    plot_df["group"] = plot_df["method"].map(method_group)
    plot_df = plot_df.sort_values(["group", "mean_cartesian_error"], ascending=[True, True])
    colors = {
        "IK / reference": "#54A24B",
        "IK-assisted": "#72B7B2",
        "Learning baseline": "#B279A2",
        "Diffusion": "#4C78A8",
        "Diffusion refinement": "#F58518",
        "Diffusion reference": "#E45756",
        "Other": "#9D755D",
    }
    bar_colors = [colors.get(group, "#9D755D") for group in plot_df["group"]]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(plot_df["plot_label"], plot_df["mean_cartesian_error"], color=bar_colors)
    ax.set_ylabel("Mean Cartesian Error [m]")
    ax.set_title("Mean Cartesian Tracking Error by Method Group")
    ax.tick_params(axis="x", rotation=35)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=color)
        for group, color in colors.items()
        if group in set(plot_df["group"])
    ]
    labels = [group for group in colors if group in set(plot_df["group"])]
    ax.legend(handles, labels, loc="best", fontsize=8)
    annotate_bars(ax, plot_df["mean_cartesian_error"].tolist())
    ax.margins(y=0.15)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def scatter_plot(df: pd.DataFrame, output_path: Path) -> Optional[Path]:
    required = ("mean_cartesian_error", "drawing_total_cost")
    for column in required:
        if column not in df.columns:
            print(f"WARNING: {output_path.name}: missing column {column}; skipping plot.")
            return None
    plot_df = df[df["mean_cartesian_error"].notna() & df["drawing_total_cost"].notna()].copy()
    skipped = len(df) - len(plot_df)
    if skipped:
        print(f"WARNING: {output_path.name}: skipped {skipped} method(s) with missing scatter metrics.")
    if plot_df.empty:
        return None

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(plot_df["mean_cartesian_error"], plot_df["drawing_total_cost"], s=70, color="#F58518")
    for _, row in plot_df.iterrows():
        ax.annotate(
            row["plot_label"],
            (row["mean_cartesian_error"], row["drawing_total_cost"]),
            xytext=(6, 4),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_xlabel("Mean Cartesian Error [m]")
    ax.set_ylabel("Drawing-Aware Cost")
    ax.set_title("Cartesian Accuracy vs Drawing-Aware Cost")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def clean_scatter_plot(df: pd.DataFrame, output_path: Path) -> Optional[Path]:
    required = ("mean_cartesian_error", "drawing_total_cost")
    for column in required:
        if column not in df.columns:
            print(f"WARNING: {output_path.name}: missing column {column}; skipping plot.")
            return None
    plot_df = df[df["mean_cartesian_error"].notna() & df["drawing_total_cost"].notna()].copy()
    skipped = len(df) - len(plot_df)
    if skipped:
        print(f"WARNING: {output_path.name}: skipped {skipped} method(s) with missing scatter metrics.")
    if plot_df.empty:
        return None
    plot_df = plot_df.sort_values(["mean_cartesian_error", "drawing_total_cost"]).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(12, 6.5))
    ax.scatter(plot_df["mean_cartesian_error"], plot_df["drawing_total_cost"], s=80, color="#F58518")
    offsets = [(7, 7), (7, -13), (7, 19), (7, -25), (7, 31), (7, -37)]
    previous_points: List[Tuple[float, float]] = []
    x_range = max(plot_df["mean_cartesian_error"].max() - plot_df["mean_cartesian_error"].min(), 1e-12)
    y_range = max(plot_df["drawing_total_cost"].max() - plot_df["drawing_total_cost"].min(), 1e-12)
    for idx, row in plot_df.iterrows():
        x = float(row["mean_cartesian_error"])
        y = float(row["drawing_total_cost"])
        close_count = sum(
            abs(x - px) / x_range < 0.04 and abs(y - py) / y_range < 0.08
            for px, py in previous_points
        )
        xytext = offsets[(idx + close_count) % len(offsets)]
        ax.annotate(
            row["plot_label"],
            (x, y),
            xytext=xytext,
            textcoords="offset points",
            fontsize=8,
            arrowprops={"arrowstyle": "-", "color": "0.55", "lw": 0.6} if close_count else None,
        )
        previous_points.append((x, y))
    ax.set_xlabel("Mean Cartesian Error [m]")
    ax.set_ylabel("Drawing-Aware Cost")
    ax.set_title("Cartesian Accuracy vs Drawing-Aware Cost")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return output_path


def save_plotting_data(df: pd.DataFrame, output_path: Path) -> Path:
    columns = [
        column
        for column in (
            "method",
            "plot_label",
            "role",
            "mean_cartesian_error",
            "max_cartesian_error",
            "drawing_total_cost",
            "dtw_distance",
            "frechet_distance",
            "tangent_weighted_error",
            "notes",
        )
        if column in df.columns
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df[columns].to_csv(output_path, index=False)
    return output_path


def best_method(df: pd.DataFrame, metric: str, subset: Optional[pd.DataFrame] = None) -> Optional[Tuple[str, float]]:
    source = df if subset is None else subset
    if metric not in source.columns:
        return None
    valid = source[source[metric].notna()].copy()
    if valid.empty:
        return None
    row = valid.loc[valid[metric].idxmin()]
    return str(row["method"]), float(row[metric])


def pct_improvement(before: Optional[float], after: Optional[float]) -> Optional[float]:
    if before is None or after is None or abs(before) < 1e-12:
        return None
    return 100.0 * (before - after) / before


def row_metric(df: pd.DataFrame, method_contains: str, metric: str) -> Optional[float]:
    if metric not in df.columns:
        return None
    rows = df[df["method"].str.contains(method_contains, regex=False, na=False)]
    if rows.empty:
        return None
    value = rows.iloc[0][metric]
    if pd.isna(value):
        return None
    return float(value)


def print_interpretation(df: pd.DataFrame) -> None:
    print("\nInterpretation")
    best_cart = best_method(df, "mean_cartesian_error")
    if best_cart:
        print(f"  Best Cartesian method: {best_cart[0]} ({best_cart[1]:.6e} m mean error).")

    diff_df = diffusion_subset(df)
    best_diff = best_method(df, "mean_cartesian_error", subset=diff_df)
    if best_diff:
        print(f"  Best diffusion method by mean Cartesian error: {best_diff[0]} ({best_diff[1]:.6e} m).")

    mlp_prior = row_metric(df, "MLP-only", "mean_cartesian_error")
    mlp_refined = row_metric(df, "Diffusion v4 MLP-prior refinement", "mean_cartesian_error")
    v1_prior = row_metric(df, "Diffusion v1 best-of-K", "mean_cartesian_error")
    v1_refined = row_metric(df, "Diffusion v4 v1-prior refinement", "mean_cartesian_error")
    mlp_pct = pct_improvement(mlp_prior, mlp_refined)
    v1_pct = pct_improvement(v1_prior, v1_refined)
    if mlp_pct is not None:
        print(f"  v4 MLP-prior refinement vs MLP-only mean Cartesian change: {mlp_pct:.2f}%.")
    if v1_pct is not None:
        print(f"  v4 v1-prior refinement vs v1 best-of-K mean Cartesian change: {v1_pct:.2f}%.")


def main() -> int:
    args = parse_args()
    df = load_results(args.input_csv)
    output_dir = args.output_dir
    generated: List[Path] = []

    plotting_data = save_plotting_data(df, output_dir / "plotting_data.csv")

    for path in (
        bar_plot(
            df,
            "mean_cartesian_error",
            "Mean Cartesian Error [m]",
            "Mean Cartesian Tracking Error by Method",
            output_dir / "mean_cartesian_error_bar.png",
        ),
        bar_plot(
            df,
            "mean_cartesian_error",
            "Mean Cartesian Error [m, log scale]",
            "Mean Cartesian Tracking Error by Method",
            output_dir / "mean_cartesian_error_bar_log.png",
            log_scale=True,
            color="#4C78A8",
        ),
        bar_plot(
            df,
            "max_cartesian_error",
            "Max Cartesian Error [m]",
            "Maximum Cartesian Tracking Error by Method",
            output_dir / "max_cartesian_error_bar.png",
        ),
        bar_plot(
            df,
            "max_cartesian_error",
            "Max Cartesian Error [m, log scale]",
            "Maximum Cartesian Tracking Error by Method",
            output_dir / "max_cartesian_error_bar_log.png",
            log_scale=True,
            color="#4C78A8",
        ),
        grouped_mean_cartesian_plot(
            df,
            output_dir / "mean_cartesian_error_grouped.png",
        ),
        bar_plot(
            diffusion_subset(df),
            "mean_cartesian_error",
            "Mean Cartesian Error [m]",
            "Diffusion Method Comparison",
            output_dir / "diffusion_methods_mean_cartesian_error_bar.png",
        ),
        scatter_plot(df, output_dir / "accuracy_vs_drawing_cost_scatter.png"),
        clean_scatter_plot(df, output_dir / "accuracy_vs_drawing_cost_scatter_clean.png"),
        # v1 best-of-K drawing_total_cost may not be directly equivalent if it was copied from mean_total_cost.
        bar_plot(
            drawing_cost_diffusion_subset(df),
            "drawing_total_cost",
            "Drawing-Aware Cost",
            "Drawing-Aware Cost for Diffusion Methods",
            output_dir / "diffusion_drawing_cost_bar.png",
            color="#F58518",
        ),
    ):
        if path is not None:
            generated.append(path)

    print("Generated plot files:")
    for path in generated:
        print(f"  {path}")
    print(f"Generated plotting data CSV:\n  {plotting_data}")
    print_interpretation(df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
