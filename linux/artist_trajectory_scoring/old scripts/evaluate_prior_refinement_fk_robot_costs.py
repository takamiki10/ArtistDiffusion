#!/usr/bin/env python3
"""Evaluate prior/refinement trajectory outputs with FK and robot-aware costs."""

from __future__ import annotations

import argparse
import csv
import importlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


DEFAULT_DATASET_DIR = Path("data/cartesian_expert_dataset_v3/diffusion_v2")
DEFAULT_RESULTS_DIR = Path("data/cartesian_expert_dataset_v3/diffusion_v4_unet/prior_refinement_outputs")
DEFAULT_OUTPUT_CSV = Path(
    "data/cartesian_expert_dataset_v3/diffusion_v4_unet/prior_refinement_fk_robot_costs.csv"
)
SOURCES = ("prior_only", "prior_refined", "pure_gaussian", "noised_expert")
JOINT_COLUMNS = ("q1", "q2", "q3", "q4", "q5", "q6")


@dataclass
class Candidate:
    experiment_name: str
    source: str
    t_start: str
    path_name: str
    csv_path: Path


@dataclass
class Weights:
    cart: float
    max_cart: float
    start: float
    end: float
    frechet: float
    dtw: float
    vel: float
    acc: float
    jerk: float
    limit: float
    tangent: float
    progress: float
    length_ratio: float
    norm_shape: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate saved prior-refinement trajectories with FK costs.")
    parser.add_argument("--dataset_dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--results_dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--split", choices=("test", "train"), default="test")
    parser.add_argument("--max_paths", type=int, default=None)
    parser.add_argument("--urdf", type=Path, default=None)
    parser.add_argument("--ee_link", default=None)
    parser.add_argument("--output_csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--w_cart", type=float, default=1.0)
    parser.add_argument("--w_max", type=float, default=0.25)
    parser.add_argument("--w_start", type=float, default=0.5)
    parser.add_argument("--w_end", type=float, default=0.5)
    parser.add_argument("--w_frechet", type=float, default=1.0)
    parser.add_argument("--w_dtw", type=float, default=0.5)
    parser.add_argument("--w_vel", type=float, default=0.01)
    parser.add_argument("--w_acc", type=float, default=0.01)
    parser.add_argument("--w_jerk", type=float, default=0.001)
    parser.add_argument("--w_limit", type=float, default=10.0)
    parser.add_argument("--w_tangent", type=float, default=0.5)
    parser.add_argument("--w_progress", type=float, default=0.5)
    parser.add_argument("--w_length_ratio", type=float, default=0.25)
    parser.add_argument("--w_norm_shape", type=float, default=1.0)
    return parser.parse_args()


def split_path(dataset_dir: Path, split: str) -> Path:
    return dataset_dir / f"diffusion_{split}_v2.npz"


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset file: {path}")
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def require_keys(data: Dict[str, np.ndarray], keys: Sequence[str]) -> None:
    missing = [key for key in keys if key not in data]
    if missing:
        raise KeyError(f"Dataset missing required key(s): {', '.join(missing)}")


def subset_data(data: Dict[str, np.ndarray], max_paths: Optional[int]) -> Dict[str, np.ndarray]:
    if max_paths is None:
        return data
    if max_paths <= 0:
        raise ValueError("--max_paths must be positive")
    out: Dict[str, np.ndarray] = {}
    for key, value in data.items():
        if value.ndim > 0 and value.shape[0] >= max_paths:
            out[key] = value[:max_paths]
        else:
            out[key] = value
    return out


def path_names(data: Dict[str, np.ndarray]) -> List[str]:
    raw = np.asarray(data["path_names"])
    names: List[str] = []
    for value in raw:
        names.append(value.decode("utf-8", errors="replace") if isinstance(value, bytes) else str(value))
    return names


def safe_path_name(name: str) -> str:
    return Path(str(name)).name.replace("/", "_").replace("\\", "_")


def read_q_csv(path: Path) -> np.ndarray:
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no header")
        field_map = {field.strip().lower(): field for field in reader.fieldnames if field is not None}
        if not all(column in field_map for column in JOINT_COLUMNS):
            raise ValueError(f"{path} must contain q1...q6 columns; found {reader.fieldnames}")
        rows: List[List[float]] = []
        for row in reader:
            rows.append([float(row[field_map[column]]) for column in JOINT_COLUMNS])
    q = np.asarray(rows, dtype=np.float64)
    if q.shape != (100, 6):
        raise ValueError(f"{path} must contain q shape (100,6), got {q.shape}")
    return q


def find_default_urdf() -> Optional[Path]:
    cwd = Path.cwd()
    candidates = [
        cwd / "xmate_description/urdf/xmatecr7.urdf",
        cwd / "xmate_description/urdf/xMateCR7.urdf",
        cwd / "urdf/xmatecr7.urdf",
        cwd / "urdf/xMateCR7.urdf",
        cwd.parent / "xmate_description/urdf/xmatecr7.urdf",
        cwd.parent / "xmate_description/urdf/xMateCR7.urdf",
    ]
    for path in candidates:
        if path.exists():
            return path
    patterns = ("*xmate*cr7*.urdf", "*xMate*CR7*.urdf", "*xmate*.urdf")
    for pattern in patterns:
        for root in (cwd, cwd.parent):
            try:
                matches = list(root.rglob(pattern))
            except Exception:
                matches = []
            if matches:
                return sorted(matches)[0]
    return None


def import_yourdfpy() -> Optional[Any]:
    try:
        module = importlib.import_module("yourdfpy")
    except Exception:
        return None
    return getattr(module, "URDF", None)


class FKComputer:
    def __init__(self, urdf_path: Optional[Path], ee_link: Optional[str]) -> None:
        self.available = False
        self.robot: Optional[Any] = None
        self.ee_link = ee_link
        self.joint_names: List[str] = []
        self.lower: Optional[np.ndarray] = None
        self.upper: Optional[np.ndarray] = None

        if urdf_path is None:
            urdf_path = find_default_urdf()
        if urdf_path is None:
            print("[fk] no URDF found; Cartesian and joint-limit metrics will be NaN")
            return

        URDF = import_yourdfpy()
        if URDF is None:
            print("[fk] yourdfpy is unavailable; Cartesian and joint-limit metrics will be NaN")
            return

        try:
            self.robot = URDF.load(str(urdf_path))
            self.joint_names, self.lower, self.upper = self._extract_joint_info(self.robot)
            if self.ee_link is None:
                self.ee_link = self._infer_ee_link(self.robot)
            self.available = self.robot is not None and self.ee_link is not None and len(self.joint_names) >= 6
        except Exception as exc:
            print(f"[fk] failed to load URDF {urdf_path}: {exc}")
            return

        if self.available:
            print(f"[fk] URDF: {urdf_path}")
            print(f"[fk] ee_link: {self.ee_link}")
            print(f"[fk] using joints: {self.joint_names[:6]}")
        else:
            print("[fk] URDF loaded, but joint names or ee_link could not be inferred")

    @staticmethod
    def _extract_joint_info(robot: Any) -> Tuple[List[str], np.ndarray, np.ndarray]:
        joint_objs = getattr(getattr(robot, "robot", robot), "joints", [])
        names: List[str] = []
        lower: List[float] = []
        upper: List[float] = []
        for joint in joint_objs:
            joint_type = getattr(joint, "type", "")
            if joint_type in ("fixed", "floating", "planar"):
                continue
            names.append(str(getattr(joint, "name")))
            limit = getattr(joint, "limit", None)
            lower.append(float(getattr(limit, "lower", -np.inf)) if limit is not None else -np.inf)
            upper.append(float(getattr(limit, "upper", np.inf)) if limit is not None else np.inf)
            if len(names) == 6:
                break
        return names, np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)

    @staticmethod
    def _infer_ee_link(robot: Any) -> Optional[str]:
        link_objs = getattr(getattr(robot, "robot", robot), "links", [])
        names = [str(getattr(link, "name")) for link in link_objs]
        preferred = ("ee_link", "tool0", "flange", "link6", "Link6", "xmate_link6")
        for candidate in preferred:
            if candidate in names:
                return candidate
        return names[-1] if names else None

    def fk(self, q: np.ndarray) -> Optional[np.ndarray]:
        if not self.available or self.robot is None or self.ee_link is None:
            return None
        positions: List[np.ndarray] = []
        for cfg in q:
            cfg_map = {name: float(value) for name, value in zip(self.joint_names[:6], cfg[:6])}
            self.robot.update_cfg(cfg_map)
            transform = self.robot.get_transform(frame_to=self.ee_link)
            positions.append(np.asarray(transform, dtype=np.float64)[:3, 3])
        return np.stack(positions, axis=0)

    def joint_limit_violation(self, q: np.ndarray) -> float:
        if self.lower is None or self.upper is None or len(self.lower) < 6:
            return 0.0
        lower = self.lower[:6].reshape(1, 6)
        upper = self.upper[:6].reshape(1, 6)
        below = np.maximum(lower - q, 0.0)
        above = np.maximum(q - upper, 0.0)
        finite_mask = np.isfinite(lower) & np.isfinite(upper)
        violation = (below + above) * finite_mask
        return float(np.mean(np.square(violation)))


def discover_experiment_dirs(results_dir: Path) -> List[Tuple[str, Path]]:
    if not results_dir.exists():
        raise FileNotFoundError(f"Missing results_dir: {results_dir}")
    source_children = [source for source in SOURCES if (results_dir / source).is_dir()]
    if source_children:
        return [(results_dir.name, results_dir)]
    experiments = [(path.name, path) for path in sorted(results_dir.iterdir()) if path.is_dir()]
    if not experiments:
        raise FileNotFoundError(f"No experiment directories found under {results_dir}")
    return experiments


def resolve_dataset_path_name(path_dir_name: str, safe_to_original: Dict[str, str]) -> Optional[str]:
    if path_dir_name in safe_to_original:
        return safe_to_original[path_dir_name]
    for marker in ("#candidate", "#sample"):
        if marker in path_dir_name:
            base = path_dir_name.split(marker, 1)[0]
            if base in safe_to_original:
                return safe_to_original[base]
    return None


def discover_candidates(results_dir: Path, dataset_names: Sequence[str]) -> List[Candidate]:
    safe_to_original = {safe_path_name(name): name for name in dataset_names}
    candidates: List[Candidate] = []
    for experiment_name, experiment_dir in discover_experiment_dirs(results_dir):
        for source in SOURCES:
            source_dir = experiment_dir / source
            if not source_dir.is_dir():
                continue
            if source == "prior_only":
                for path_dir in sorted(source_dir.iterdir()):
                    if not path_dir.is_dir():
                        continue
                    csv_path = path_dir / "predicted_q.csv"
                    dataset_name = resolve_dataset_path_name(path_dir.name, safe_to_original)
                    if csv_path.exists() and dataset_name is not None:
                        candidates.append(
                            Candidate(experiment_name, source, "prior_only", dataset_name, csv_path)
                        )
                continue
            for t_dir in sorted(source_dir.iterdir()):
                if not t_dir.is_dir() or not t_dir.name.startswith("t_"):
                    continue
                t_start = t_dir.name[2:]
                for path_dir in sorted(t_dir.iterdir()):
                    if not path_dir.is_dir():
                        continue
                    csv_path = path_dir / "predicted_q.csv"
                    dataset_name = resolve_dataset_path_name(path_dir.name, safe_to_original)
                    if csv_path.exists() and dataset_name is not None:
                        candidates.append(
                            Candidate(experiment_name, source, t_start, dataset_name, csv_path)
                        )
    return candidates


def q_error_metrics(q: np.ndarray, expert_q: np.ndarray) -> Tuple[float, float]:
    error = q - expert_q
    return float(np.sqrt(np.mean(np.square(error)))), float(np.max(np.abs(error)))


def smoothness_costs(q: np.ndarray) -> Tuple[float, float, float]:
    velocity = np.diff(q, axis=0)
    acceleration = np.diff(q, n=2, axis=0)
    jerk = np.diff(q, n=3, axis=0)
    return (
        float(np.mean(np.square(velocity))) if velocity.size else 0.0,
        float(np.mean(np.square(acceleration))) if acceleration.size else 0.0,
        float(np.mean(np.square(jerk))) if jerk.size else 0.0,
    )


def cartesian_metrics(fk_positions: Optional[np.ndarray], desired_path: np.ndarray) -> Tuple[float, float, float]:
    if fk_positions is None:
        return float("nan"), float("nan"), float("nan")
    error = fk_positions - desired_path
    distances = np.linalg.norm(error, axis=1)
    path_error = float(np.mean(np.sum(np.square(error), axis=1)))
    return float(np.mean(distances)), float(np.max(distances)), path_error


def path_length(path: np.ndarray) -> float:
    if path.shape[0] < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1)))


def discrete_frechet_distance(path_a: np.ndarray, path_b: np.ndarray) -> float:
    n = path_a.shape[0]
    m = path_b.shape[0]
    if n == 0 or m == 0:
        return float("nan")
    distances = np.linalg.norm(path_a[:, None, :] - path_b[None, :, :], axis=2)
    ca = np.full((n, m), np.inf, dtype=np.float64)
    for i in range(n):
        for j in range(m):
            if i == 0 and j == 0:
                ca[i, j] = distances[i, j]
            elif i == 0:
                ca[i, j] = max(ca[i, j - 1], distances[i, j])
            elif j == 0:
                ca[i, j] = max(ca[i - 1, j], distances[i, j])
            else:
                ca[i, j] = max(
                    min(ca[i - 1, j], ca[i - 1, j - 1], ca[i, j - 1]),
                    distances[i, j],
                )
    return float(ca[-1, -1])


def dtw_distance(path_a: np.ndarray, path_b: np.ndarray) -> float:
    """Dynamic time warping distance normalized by the longer path length in samples."""

    n = path_a.shape[0]
    m = path_b.shape[0]
    if n == 0 or m == 0:
        return float("nan")
    dp = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    dp[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = float(np.linalg.norm(path_a[i - 1] - path_b[j - 1]))
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])
    return float(dp[n, m] / max(n, m))


def cumulative_arc_length(path: np.ndarray) -> np.ndarray:
    if path.shape[0] == 0:
        return np.asarray([], dtype=np.float64)
    if path.shape[0] == 1:
        return np.asarray([0.0], dtype=np.float64)
    segment_lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(segment_lengths)])


def normalized_progress(path: np.ndarray) -> np.ndarray:
    cumulative = cumulative_arc_length(path)
    if cumulative.size == 0:
        return cumulative
    total = cumulative[-1]
    if total <= 1e-12:
        return np.zeros_like(cumulative)
    return cumulative / total


def tangent_errors(
    pred_path: np.ndarray,
    desired_path: np.ndarray,
    eps: float = 1e-12,
) -> Tuple[float, float]:
    pred_segments = np.diff(pred_path, axis=0)
    desired_segments = np.diff(desired_path, axis=0)
    if pred_segments.size == 0 or desired_segments.size == 0:
        return float("nan"), float("nan")
    pred_norm = np.linalg.norm(pred_segments, axis=1)
    desired_norm = np.linalg.norm(desired_segments, axis=1)
    pred_unit = pred_segments / np.maximum(pred_norm[:, None], eps)
    desired_unit = desired_segments / np.maximum(desired_norm[:, None], eps)
    cosine = np.sum(pred_unit * desired_unit, axis=1)
    cosine = np.clip(cosine, -1.0, 1.0)
    cosine_error = 1.0 - cosine
    tangent_cosine_error = float(np.mean(cosine_error))
    weight_sum = float(np.sum(desired_norm))
    if weight_sum <= eps:
        tangent_weighted_error = tangent_cosine_error
    else:
        tangent_weighted_error = float(np.sum(cosine_error * desired_norm) / weight_sum)
    return tangent_cosine_error, tangent_weighted_error


def progress_error(pred_path: np.ndarray, desired_path: np.ndarray) -> float:
    pred_progress = normalized_progress(pred_path)
    desired_progress = normalized_progress(desired_path)
    if pred_progress.shape != desired_progress.shape:
        count = min(pred_progress.shape[0], desired_progress.shape[0])
        pred_progress = pred_progress[:count]
        desired_progress = desired_progress[:count]
    if pred_progress.size == 0:
        return float("nan")
    return float(np.mean(np.abs(pred_progress - desired_progress)))


def resample_by_normalized_arc_length(path: np.ndarray, num_points: int = 100) -> np.ndarray:
    if path.shape[0] == 0:
        return np.zeros((num_points, 3), dtype=np.float64)
    if path.shape[0] == 1:
        return np.repeat(path.astype(np.float64), num_points, axis=0)
    progress = normalized_progress(path)
    sample_points = np.linspace(0.0, 1.0, num_points)
    out = np.zeros((num_points, path.shape[1]), dtype=np.float64)
    for dim in range(path.shape[1]):
        out[:, dim] = np.interp(sample_points, progress, path[:, dim])
    return out


def normalized_shape_error(pred_path: np.ndarray, desired_path: np.ndarray, num_points: int = 100) -> float:
    pred_resampled = resample_by_normalized_arc_length(pred_path, num_points=num_points)
    desired_resampled = resample_by_normalized_arc_length(desired_path, num_points=num_points)
    pred_centered = pred_resampled - pred_resampled[0]
    desired_centered = desired_resampled - desired_resampled[0]
    desired_length = path_length(desired_path)
    scale = desired_length if desired_length > 1e-12 else 1.0
    distances = np.linalg.norm((pred_centered - desired_centered) / scale, axis=1)
    return float(np.mean(distances))


def drawing_fidelity_metrics(
    fk_positions: Optional[np.ndarray],
    desired_path: np.ndarray,
    path_length_pred: float,
    path_length_desired: float,
) -> Tuple[float, float, float, float, float]:
    if fk_positions is None:
        return float("nan"), float("nan"), float("nan"), float("nan"), float("nan")
    tangent_cosine, tangent_weighted = tangent_errors(fk_positions, desired_path)
    progress = progress_error(fk_positions, desired_path)
    if path_length_pred <= 1e-12 or path_length_desired <= 1e-12:
        length_ratio_err = float("nan")
    else:
        length_ratio_err = float(abs(np.log(path_length_pred / path_length_desired)))
    norm_shape = normalized_shape_error(fk_positions, desired_path, num_points=100)
    return tangent_cosine, tangent_weighted, progress, length_ratio_err, norm_shape


def shape_path_metrics(
    fk_positions: Optional[np.ndarray],
    desired_path: np.ndarray,
) -> Tuple[float, float, float, float, float, float, float]:
    if fk_positions is None:
        desired_length = path_length(desired_path)
        return (
            float("nan"),
            float("nan"),
            float("nan"),
            desired_length,
            float("nan"),
            float("nan"),
            float("nan"),
        )
    start_error = float(np.linalg.norm(fk_positions[0] - desired_path[0]))
    end_error = float(np.linalg.norm(fk_positions[-1] - desired_path[-1]))
    pred_length = path_length(fk_positions)
    desired_length = path_length(desired_path)
    length_ratio = pred_length / desired_length if desired_length > 1e-12 else float("nan")
    frechet = discrete_frechet_distance(desired_path, fk_positions)
    dtw = dtw_distance(desired_path, fk_positions)
    return start_error, end_error, pred_length, desired_length, length_ratio, frechet, dtw


def total_cost(
    weights: Weights,
    mean_cart: float,
    max_cart: float,
    vel: float,
    acc: float,
    jerk: float,
    limit: float,
) -> float:
    cart = 0.0 if np.isnan(mean_cart) else mean_cart
    max_value = 0.0 if np.isnan(max_cart) else max_cart
    return (
        weights.cart * cart
        + weights.max_cart * max_value
        + weights.vel * vel
        + weights.acc * acc
        + weights.jerk * jerk
        + weights.limit * limit
    )


def finite_or_zero(value: float) -> float:
    return 0.0 if not np.isfinite(value) else value


def shape_total_cost(
    weights: Weights,
    mean_cart: float,
    max_cart: float,
    start_error: float,
    end_error: float,
    frechet: float,
    dtw: float,
    vel: float,
    acc: float,
    jerk: float,
    limit: float,
) -> float:
    return (
        weights.cart * finite_or_zero(mean_cart)
        + weights.max_cart * finite_or_zero(max_cart)
        + weights.start * finite_or_zero(start_error)
        + weights.end * finite_or_zero(end_error)
        + weights.frechet * finite_or_zero(frechet)
        + weights.dtw * finite_or_zero(dtw)
        + weights.vel * vel
        + weights.acc * acc
        + weights.jerk * jerk
        + weights.limit * limit
    )


def drawing_total_cost(
    weights: Weights,
    shape_cost: float,
    tangent_weighted_error: float,
    progress: float,
    length_ratio_error: float,
    norm_shape: float,
) -> float:
    return (
        shape_cost
        + weights.tangent * finite_or_zero(tangent_weighted_error)
        + weights.progress * finite_or_zero(progress)
        + weights.length_ratio * finite_or_zero(length_ratio_error)
        + weights.norm_shape * finite_or_zero(norm_shape)
    )


def write_rows(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    fields = (
        "experiment_name",
        "source",
        "t_start",
        "path_name",
        "q_rmse",
        "max_q_error",
        "mean_cartesian_error",
        "max_cartesian_error",
        "path_error",
        "start_error",
        "end_error",
        "path_length_pred",
        "path_length_desired",
        "path_length_ratio",
        "frechet_distance",
        "dtw_distance",
        "tangent_cosine_error",
        "tangent_weighted_error",
        "progress_error",
        "length_ratio_error",
        "normalized_shape_error",
        "joint_velocity_cost",
        "joint_acceleration_cost",
        "joint_jerk_cost",
        "joint_limit_violation",
        "total_cost",
        "shape_total_cost",
        "drawing_total_cost",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row[field] for field in fields})


def grouped_summary(rows: Sequence[Dict[str, Any]]) -> None:
    prior_by_experiment_path: Dict[Tuple[str, str], Dict[str, float]] = {}
    for row in rows:
        if row["source"] != "prior_only":
            continue
        prior_by_experiment_path[(row["experiment_name"], row["path_name"])] = {
            "mean_cartesian_error": float(row["mean_cartesian_error"]),
            "frechet_distance": float(row["frechet_distance"]),
            "dtw_distance": float(row["dtw_distance"]),
            "total_cost": float(row["total_cost"]),
            "shape_total_cost": float(row["shape_total_cost"]),
            "drawing_total_cost": float(row["drawing_total_cost"]),
            "tangent_weighted_error": float(row["tangent_weighted_error"]),
            "progress_error": float(row["progress_error"]),
            "normalized_shape_error": float(row["normalized_shape_error"]),
            "q_rmse": float(row["q_rmse"]),
        }

    groups: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["experiment_name"], row["source"], row["t_start"]), []).append(row)
    print("\nGrouped summary")
    print(
        "experiment_name | source | t_start | mean_cart_error | frechet | dtw | tangent | "
        "progress | length_ratio_error | norm_shape | shape_total_cost | drawing_total_cost | "
        "drawing_total_improved_paths | tangent_improved_paths | progress_improved_paths | "
        "normalized_shape_improved_paths"
    )
    for (experiment_name, source, t_start), group in sorted(groups.items()):
        def mean_field(field: str) -> float:
            values = np.asarray([float(row[field]) for row in group], dtype=np.float64)
            return float(np.nanmean(values))

        drawing_total_improved = 0
        tangent_improved = 0
        progress_improved = 0
        normalized_shape_improved = 0
        comparable = 0
        for row in group:
            baseline = prior_by_experiment_path.get((experiment_name, row["path_name"]))
            if baseline is None:
                continue
            comparable += 1
            if float(row["drawing_total_cost"]) < baseline["drawing_total_cost"]:
                drawing_total_improved += 1
            row_tangent = float(row["tangent_weighted_error"])
            base_tangent = baseline["tangent_weighted_error"]
            if np.isfinite(row_tangent) and np.isfinite(base_tangent) and row_tangent < base_tangent:
                tangent_improved += 1
            row_progress = float(row["progress_error"])
            base_progress = baseline["progress_error"]
            if np.isfinite(row_progress) and np.isfinite(base_progress) and row_progress < base_progress:
                progress_improved += 1
            row_norm_shape = float(row["normalized_shape_error"])
            base_norm_shape = baseline["normalized_shape_error"]
            if np.isfinite(row_norm_shape) and np.isfinite(base_norm_shape) and row_norm_shape < base_norm_shape:
                normalized_shape_improved += 1

        print(
            f"{experiment_name} | {source} | {t_start} | "
            f"{mean_field('mean_cartesian_error'):.6e} | "
            f"{mean_field('frechet_distance'):.6e} | "
            f"{mean_field('dtw_distance'):.6e} | "
            f"{mean_field('tangent_weighted_error'):.6e} | "
            f"{mean_field('progress_error'):.6e} | "
            f"{mean_field('length_ratio_error'):.6e} | "
            f"{mean_field('normalized_shape_error'):.6e} | "
            f"{mean_field('shape_total_cost'):.6e} | "
            f"{mean_field('drawing_total_cost'):.6e} | "
            f"{drawing_total_improved}/{comparable} | "
            f"{tangent_improved}/{comparable} | "
            f"{progress_improved}/{comparable} | "
            f"{normalized_shape_improved}/{comparable}"
        )


def main() -> int:
    args = parse_args()
    data = subset_data(load_npz(split_path(args.dataset_dir, args.split)), args.max_paths)
    require_keys(data, ("expert_q", "desired_paths", "path_names"))
    names = path_names(data)
    name_to_idx = {name: idx for idx, name in enumerate(names)}
    expert_q = np.asarray(data["expert_q"], dtype=np.float64)
    desired_paths = np.asarray(data["desired_paths"], dtype=np.float64)

    fk = FKComputer(args.urdf, args.ee_link)
    weights = Weights(
        args.w_cart,
        args.w_max,
        args.w_start,
        args.w_end,
        args.w_frechet,
        args.w_dtw,
        args.w_vel,
        args.w_acc,
        args.w_jerk,
        args.w_limit,
        args.w_tangent,
        args.w_progress,
        args.w_length_ratio,
        args.w_norm_shape,
    )
    candidates = discover_candidates(args.results_dir, names)
    if not candidates:
        raise FileNotFoundError(f"No candidate predicted_q.csv files found under {args.results_dir}")
    print(f"Found {len(candidates)} candidate trajectories")

    rows: List[Dict[str, Any]] = []
    for candidate in candidates:
        idx = name_to_idx[candidate.path_name]
        q = read_q_csv(candidate.csv_path)
        q_rmse, max_q_error = q_error_metrics(q, expert_q[idx])
        fk_positions = fk.fk(q)
        mean_cart, max_cart, path_error = cartesian_metrics(fk_positions, desired_paths[idx])
        (
            start_error,
            end_error,
            path_length_pred,
            path_length_desired,
            path_length_ratio,
            frechet,
            dtw,
        ) = shape_path_metrics(fk_positions, desired_paths[idx])
        (
            tangent_cosine,
            tangent_weighted,
            progress,
            length_ratio_error,
            norm_shape,
        ) = drawing_fidelity_metrics(
            fk_positions,
            desired_paths[idx],
            path_length_pred,
            path_length_desired,
        )
        vel, acc, jerk = smoothness_costs(q)
        limit = fk.joint_limit_violation(q)
        cost = total_cost(weights, mean_cart, max_cart, vel, acc, jerk, limit)
        shape_cost = shape_total_cost(
            weights,
            mean_cart,
            max_cart,
            start_error,
            end_error,
            frechet,
            dtw,
            vel,
            acc,
            jerk,
            limit,
        )
        drawing_cost = drawing_total_cost(
            weights,
            shape_cost,
            tangent_weighted,
            progress,
            length_ratio_error,
            norm_shape,
        )
        rows.append(
            {
                "experiment_name": candidate.experiment_name,
                "source": candidate.source,
                "t_start": candidate.t_start,
                "path_name": candidate.path_name,
                "q_rmse": f"{q_rmse:.12e}",
                "max_q_error": f"{max_q_error:.12e}",
                "mean_cartesian_error": f"{mean_cart:.12e}",
                "max_cartesian_error": f"{max_cart:.12e}",
                "path_error": f"{path_error:.12e}",
                "start_error": f"{start_error:.12e}",
                "end_error": f"{end_error:.12e}",
                "path_length_pred": f"{path_length_pred:.12e}",
                "path_length_desired": f"{path_length_desired:.12e}",
                "path_length_ratio": f"{path_length_ratio:.12e}",
                "frechet_distance": f"{frechet:.12e}",
                "dtw_distance": f"{dtw:.12e}",
                "tangent_cosine_error": f"{tangent_cosine:.12e}",
                "tangent_weighted_error": f"{tangent_weighted:.12e}",
                "progress_error": f"{progress:.12e}",
                "length_ratio_error": f"{length_ratio_error:.12e}",
                "normalized_shape_error": f"{norm_shape:.12e}",
                "joint_velocity_cost": f"{vel:.12e}",
                "joint_acceleration_cost": f"{acc:.12e}",
                "joint_jerk_cost": f"{jerk:.12e}",
                "joint_limit_violation": f"{limit:.12e}",
                "total_cost": f"{cost:.12e}",
                "shape_total_cost": f"{shape_cost:.12e}",
                "drawing_total_cost": f"{drawing_cost:.12e}",
            }
        )

    write_rows(args.output_csv, rows)
    print(f"Saved evaluation CSV: {args.output_csv}")
    grouped_summary(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
