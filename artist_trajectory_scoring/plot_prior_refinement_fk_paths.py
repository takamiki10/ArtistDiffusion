#!/usr/bin/env python3
"""Plot desired Cartesian path against FK trajectories for refinement outputs."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from evaluate_prior_refinement_fk_robot_costs import FKComputer, read_q_csv, safe_path_name


DEFAULT_DATASET_DIR = Path("data/cartesian_expert_dataset_v3/diffusion_v2")
DEFAULT_RESULTS_DIR = Path("data/cartesian_expert_dataset_v3/diffusion_v4_unet/prior_refinement_outputs")
DEFAULT_METRICS_CSV = Path("data/cartesian_expert_dataset_v3/diffusion_v4_unet/prior_refinement_fk_robot_costs.csv")
DEFAULT_OUTPUT_DIR = Path("data/cartesian_expert_dataset_v3/diffusion_v4_unet/prior_refinement_fk_plots")
SOURCES = ("prior_only", "prior_refined", "pure_gaussian", "noised_expert")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot desired path vs FK paths for selected refinement cases.")
    parser.add_argument("--dataset_dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--results_dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--metrics_csv", type=Path, default=DEFAULT_METRICS_CSV)
    parser.add_argument("--split", choices=("test", "train"), default="test")
    parser.add_argument("--experiment_name", default="all")
    parser.add_argument("--t_start", default="25")
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--urdf", type=Path, default=None)
    parser.add_argument("--ee_link", default=None)
    return parser.parse_args()


def split_path(dataset_dir: Path, split: str) -> Path:
    return dataset_dir / f"diffusion_{split}_v2.npz"


def load_dataset(dataset_dir: Path, split: str) -> Tuple[List[str], np.ndarray]:
    path = split_path(dataset_dir, split)
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset file: {path}")
    with np.load(path, allow_pickle=True) as data:
        raw_names = data["path_names"]
        desired = np.asarray(data["desired_paths"], dtype=np.float64)
    names = [
        value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value)
        for value in raw_names
    ]
    return names, desired


def load_metrics(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Missing metrics CSV: {path}")
    with path.open("r", newline="") as handle:
        return list(csv.DictReader(handle))


def candidate_q_path(
    results_dir: Path,
    experiment_name: str,
    source: str,
    t_start: str,
    path_name: str,
) -> Path:
    if source == "prior_only":
        return results_dir / experiment_name / source / safe_path_name(path_name) / "predicted_q.csv"
    return results_dir / experiment_name / source / f"t_{t_start}" / safe_path_name(path_name) / "predicted_q.csv"


def row_key(row: Dict[str, str]) -> Tuple[str, str, str, str]:
    return row["experiment_name"], row["source"], row["t_start"], row["path_name"]


def select_paths(
    rows: Sequence[Dict[str, str]],
    experiment_name: str,
    t_start: str,
) -> List[Tuple[str, str, float]]:
    prior = {
        row["path_name"]: row
        for row in rows
        if row["experiment_name"] == experiment_name
        and row["source"] == "prior_only"
        and row["t_start"] == "prior_only"
    }
    refined = {
        row["path_name"]: row
        for row in rows
        if row["experiment_name"] == experiment_name
        and row["source"] == "prior_refined"
        and row["t_start"] == t_start
    }
    improvements: List[Tuple[str, float]] = []
    for path_name, refined_row in refined.items():
        prior_row = prior.get(path_name)
        if prior_row is None:
            continue
        improvement = float(prior_row["mean_cartesian_error"]) - float(refined_row["mean_cartesian_error"])
        improvements.append((path_name, improvement))
    if not improvements:
        raise RuntimeError(f"No comparable prior_only/prior_refined t={t_start} rows for {experiment_name}")

    improvements_sorted = sorted(improvements, key=lambda item: item[1], reverse=True)
    positive = [item for item in improvements_sorted if item[1] > 0.0]
    median_pool = positive if positive else improvements_sorted
    median_item = sorted(median_pool, key=lambda item: item[1])[len(median_pool) // 2]
    return [
        ("best_improved", improvements_sorted[0][0], improvements_sorted[0][1]),
        ("median_improved", median_item[0], median_item[1]),
        ("worst_path", improvements_sorted[-1][0], improvements_sorted[-1][1]),
    ]


def set_equal_3d_axes(ax: Any, points: Sequence[np.ndarray]) -> None:
    all_points = np.concatenate(points, axis=0)
    mins = np.min(all_points, axis=0)
    maxs = np.max(all_points, axis=0)
    centers = (mins + maxs) / 2.0
    radius = float(np.max(maxs - mins) / 2.0)
    radius = max(radius, 1e-6)
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)


def plot_case(
    output_path: Path,
    experiment_name: str,
    label: str,
    path_name: str,
    improvement: float,
    desired_path: np.ndarray,
    fk_paths: Dict[str, np.ndarray],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(desired_path[:, 0], desired_path[:, 1], desired_path[:, 2], color="black", linewidth=3, label="desired_path")
    styles = {
        "prior_only": ("#1f77b4", "-"),
        "prior_refined": ("#2ca02c", "-"),
        "pure_gaussian": ("#d62728", "--"),
        "noised_expert": ("#9467bd", "-."),
    }
    for source in SOURCES:
        path = fk_paths.get(source)
        if path is None:
            continue
        color, linestyle = styles[source]
        ax.plot(path[:, 0], path[:, 1], path[:, 2], color=color, linestyle=linestyle, linewidth=2, label=f"FK {source}")
    ax.scatter(desired_path[0, 0], desired_path[0, 1], desired_path[0, 2], color="black", s=35, marker="o", label="start")
    ax.scatter(desired_path[-1, 0], desired_path[-1, 1], desired_path[-1, 2], color="black", s=45, marker="x", label="end")
    set_equal_3d_axes(ax, [desired_path] + list(fk_paths.values()))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(f"{experiment_name} | {label} | {path_name} | cart improvement={improvement:.6e}")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    names, desired_paths = load_dataset(args.dataset_dir, args.split)
    name_to_idx = {name: idx for idx, name in enumerate(names)}
    rows = load_metrics(args.metrics_csv)
    experiments = sorted({row["experiment_name"] for row in rows})
    if args.experiment_name != "all":
        experiments = [args.experiment_name]

    fk = FKComputer(args.urdf, args.ee_link)
    if not fk.available:
        raise RuntimeError("FK is unavailable; cannot plot FK paths")

    saved: List[Path] = []
    for experiment in experiments:
        selected = select_paths(rows, experiment, args.t_start)
        print(f"\n[{experiment}] selected paths")
        for label, path_name, improvement in selected:
            print(f"  {label}: {path_name} cartesian improvement={improvement:.12e}")
            idx = name_to_idx[path_name]
            fk_paths: Dict[str, np.ndarray] = {}
            for source in SOURCES:
                q_path = candidate_q_path(args.results_dir, experiment, source, args.t_start, path_name)
                if not q_path.exists():
                    print(f"    missing {source}: {q_path}")
                    continue
                q = read_q_csv(q_path)
                fk_path = fk.fk(q)
                if fk_path is None:
                    raise RuntimeError(f"FK failed for {q_path}")
                fk_paths[source] = fk_path
            output_path = args.output_dir / experiment / f"{label}_{safe_path_name(path_name)}_t_{args.t_start}.png"
            plot_case(output_path, experiment, label, path_name, improvement, desired_paths[idx], fk_paths)
            saved.append(output_path)

    print("\nSaved plots")
    for path in saved:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
