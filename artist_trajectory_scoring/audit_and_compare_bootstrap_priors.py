#!/usr/bin/env python3
"""Audit the v5b bootstrap-prior lineage and compare full joint trajectories.

The audit is intentionally read-only with respect to datasets, models, and
checkpoints.  It writes only reports, plots, and reconstructed copies beneath
--output_dir.

Supported repository-native candidate formats are deliberately narrow:

* diffusion-v2 NPZ files with path_names/expert_q (oracle-only);
* v5/v5b window NPZ files with prior_q_window, path_names, and starts;
* joint CSVs with t,q1,...,q6;
* path trees containing the repository's known joint-CSV filenames; and
* diffusion-v1 best-of-K manifests with path_name/output_folder rows.

The xMateCR7 FK convention is fixed to six active joints and exactly:

    robot.update_cfg(cfg)
    transform = robot.get_transform(frame_to=ee_link)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np


DATA_ROOT = Path("data/cartesian_expert_dataset_v3")
DEFAULT_TEST_NPZ = DATA_ROOT / "diffusion_v2/diffusion_test_v2.npz"
DEFAULT_TRAIN_NPZ = DATA_ROOT / "diffusion_v2/diffusion_train_v2.npz"
DEFAULT_WINDOW_TEST_NPZ = (
    DATA_ROOT / "diffusion_v5b_residual_windows_fk_condition/test_windows.npz"
)
DEFAULT_WINDOW_TRAIN_NPZ = (
    DATA_ROOT / "diffusion_v5b_residual_windows_fk_condition/train_windows.npz"
)
DEFAULT_OUTPUT_DIR = DATA_ROOT / "bootstrap_prior_audit"

JOINT_COLUMNS = ("q1", "q2", "q3", "q4", "q5", "q6")
XYZ_COLUMNS = ("x", "y", "z")
JOINT_DIM = 6
RANGE_EPS = 1.0e-10
OVERLAP_ATOL = 1.0e-7
OVERLAP_RTOL = 1.0e-6

KNOWN_Q_FILENAMES = (
    "predicted_q.csv",
    "path_conditioned_pred_q.csv",
    "refined_mlp_ik_q.csv",
    "diffusion_pred_q.csv",
    "expert_q.csv",
    "bootstrap_prior_q.csv",
    "buffer_only_q.csv",
    "buffer_plus_diffusion_q.csv",
    "diffusion_lexicographic_q.csv",
    "diffusion_discounted_hard_gate_q.csv",
    "global_reference_q.csv",
    "global_anchored_tail_q.csv",
    "base_tail_q.csv",
)

METRIC_FIELDS = (
    "mean_cartesian_error",
    "rms_cartesian_error",
    "maximum_cartesian_error",
    "median_cartesian_error",
    "p95_cartesian_error",
    "starting_point_error",
    "ending_point_error",
    "x_range_ratio",
    "y_range_ratio",
    "z_range_ratio",
    "cartesian_arc_length_ratio",
    "mean_joint_step",
    "maximum_joint_step",
    "velocity_cost",
    "acceleration_cost",
    "jerk_cost",
    "joint_limit_violation_count",
    "joint_limit_violation_magnitude",
    "joint_rmse_vs_expert",
)

AGGREGATE_STATS = ("mean", "median", "std", "minimum", "maximum")


@dataclass(frozen=True)
class SplitDataset:
    label: str
    source: Path
    names: Tuple[str, ...]
    desired: Mapping[str, np.ndarray]
    expert_q: Mapping[str, np.ndarray]
    trajectory_length: int


@dataclass
class Candidate:
    name: str
    method_class: str
    train: Dict[str, np.ndarray] = field(default_factory=dict)
    test: Dict[str, np.ndarray] = field(default_factory=dict)
    source_paths: List[str] = field(default_factory=list)
    source_checkpoint: str = ""
    expert_dependence_status: str = "unknown"
    practical_at_inference: str = "unknown"
    generation_uses_expert_information: str = "unknown"
    loader_format: str = ""
    notes: List[str] = field(default_factory=list)
    duplicate_train_names: List[str] = field(default_factory=list)
    duplicate_test_names: List[str] = field(default_factory=list)
    invalid_shapes: List[str] = field(default_factory=list)
    load_errors: List[str] = field(default_factory=list)
    unavailable_reason: str = ""

    @property
    def available(self) -> bool:
        return bool(self.train or self.test)

    def add_source(self, path: Path | str) -> None:
        text = str(path)
        if text and text not in self.source_paths:
            self.source_paths.append(text)

    def add_trajectory(
        self,
        split: str,
        name: str,
        q: np.ndarray,
        source: Path | str,
    ) -> None:
        target = self.train if split == "train" else self.test
        duplicates = (
            self.duplicate_train_names if split == "train" else self.duplicate_test_names
        )
        if name in target:
            duplicates.append(name)
            self.notes.append(
                f"duplicate {split} trajectory for {name} ignored from {source}"
            )
            return
        target[name] = np.asarray(q, dtype=np.float64)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test_npz", type=Path, default=DEFAULT_TEST_NPZ)
    parser.add_argument("--train_npz", type=Path, default=DEFAULT_TRAIN_NPZ)
    parser.add_argument("--window_test_npz", type=Path, default=DEFAULT_WINDOW_TEST_NPZ)
    parser.add_argument("--window_train_npz", type=Path, default=DEFAULT_WINDOW_TRAIN_NPZ)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--max_paths",
        type=int,
        default=0,
        help="Maximum test paths to FK-evaluate; zero evaluates every test path.",
    )
    parser.add_argument(
        "--candidate_prior",
        action="append",
        default=[],
        metavar="NAME=PATH",
        help="Additional or replacement candidate. Repeat for multiple candidates.",
    )
    parser.add_argument("--urdf_path", type=Path, default=None)
    parser.add_argument("--ee_link", type=str, default=None)
    parser.add_argument(
        "--joint_names",
        type=str,
        default=None,
        help="Six comma-separated URDF joints; defaults to the project xMateCR7 list.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
        help="The repository yourdfpy FK backend is CPU-only; the request is recorded.",
    )
    parser.add_argument("--fallback_joint_min", type=float, default=-math.pi)
    parser.add_argument("--fallback_joint_max", type=float, default=math.pi)
    parser.add_argument("--overlap_atol", type=float, default=OVERLAP_ATOL)
    parser.add_argument("--overlap_rtol", type=float, default=OVERLAP_RTOL)
    args = parser.parse_args(argv)
    if args.max_paths < 0:
        parser.error("--max_paths must be zero or positive")
    if args.overlap_atol < 0.0 or args.overlap_rtol < 0.0:
        parser.error("overlap tolerances must be non-negative")
    return args


def load_npz_dict(path: Path) -> Dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=True) as archive:
        return {key: archive[key] for key in archive.files}


def decode_names(values: np.ndarray) -> List[str]:
    result: List[str] = []
    for value in np.asarray(values).reshape(-1):
        if isinstance(value, bytes):
            result.append(value.decode("utf-8", errors="replace"))
        else:
            result.append(str(value))
    return result


def require_keys(
    data: Mapping[str, np.ndarray],
    keys: Sequence[str],
    label: str,
) -> None:
    missing = [key for key in keys if key not in data]
    if missing:
        raise KeyError(f"{label} missing keys {missing}; available={sorted(data)}")


def load_split_dataset(path: Path, label: str) -> SplitDataset:
    data = load_npz_dict(path)
    require_keys(data, ("desired_paths", "expert_q", "path_names"), str(path))
    desired = np.asarray(data["desired_paths"], dtype=np.float64)
    expert_q = np.asarray(data["expert_q"], dtype=np.float64)
    names = decode_names(data["path_names"])
    if desired.ndim != 3 or desired.shape[-1] != 3:
        raise ValueError(f"{path}: desired_paths must be [N,T,3], got {desired.shape}")
    if expert_q.shape != (desired.shape[0], desired.shape[1], JOINT_DIM):
        raise ValueError(
            f"{path}: expert_q must be {(desired.shape[0], desired.shape[1], JOINT_DIM)}, "
            f"got {expert_q.shape}"
        )
    if len(names) != desired.shape[0]:
        raise ValueError(
            f"{path}: path_names has {len(names)} entries for N={desired.shape[0]}"
        )
    duplicate_names = sorted(name for name, count in Counter(names).items() if count > 1)
    if duplicate_names:
        raise ValueError(f"{path}: duplicated path_names: {duplicate_names}")
    if not np.all(np.isfinite(desired)) or not np.all(np.isfinite(expert_q)):
        raise ValueError(f"{path}: desired_paths/expert_q contain non-finite values")
    return SplitDataset(
        label=label,
        source=path,
        names=tuple(names),
        desired={name: desired[idx] for idx, name in enumerate(names)},
        expert_q={name: expert_q[idx] for idx, name in enumerate(names)},
        trajectory_length=int(desired.shape[1]),
    )


def safe_path_name(name: str) -> str:
    safe = Path(str(name)).name.replace("/", "_").replace("\\", "_")
    return safe or "unnamed_path"


def scalar_int(value: np.ndarray, label: str) -> int:
    arr = np.asarray(value)
    if arr.size != 1:
        raise ValueError(f"{label} must be scalar, got shape {arr.shape}")
    return int(arr.reshape(-1)[0])


def resolve_window_stride(
    window_path: Path,
    window_data: Mapping[str, np.ndarray],
) -> Tuple[int, str, Optional[Dict[str, np.ndarray]]]:
    for key in ("stride", "window_stride"):
        if key in window_data:
            stride = scalar_int(window_data[key], f"{window_path}:{key}")
            if stride <= 0:
                raise ValueError(f"{window_path}:{key} must be positive")
            return stride, f"{window_path}:{key}", None

    stats_path = window_path.parent / "normalization_stats.npz"
    if not stats_path.exists():
        raise KeyError(
            f"{window_path} has no stride key and {stats_path} is missing; "
            "the reconstruction will not assume stride one"
        )
    stats = load_npz_dict(stats_path)
    if "stride" not in stats:
        raise KeyError(f"{stats_path} is missing required stride metadata")
    stride = scalar_int(stats["stride"], f"{stats_path}:stride")
    if stride <= 0:
        raise ValueError(f"{stats_path}:stride must be positive")
    return stride, f"{stats_path}:stride", stats


def write_joint_csv(path: Path, q: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    times = np.linspace(0.0, 1.0, q.shape[0], dtype=np.float64)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("t",) + JOINT_COLUMNS)
        for time_value, row in zip(times, q):
            writer.writerow(
                [f"{float(time_value):.10f}"]
                + ["" if not np.isfinite(value) else f"{float(value):.12g}" for value in row]
            )


def join_ints(values: Iterable[int]) -> str:
    return ";".join(str(int(value)) for value in values)


def join_names(values: Iterable[str]) -> str:
    return ";".join(str(value) for value in values)


def reconstruct_current_prior(
    *,
    split: SplitDataset,
    window_path: Path,
    output_dir: Path,
    overlap_atol: float,
    overlap_rtol: float,
    save_reconstructions: bool = True,
) -> Tuple[Dict[str, np.ndarray], List[Dict[str, Any]], Dict[str, Any]]:
    data = load_npz_dict(window_path)
    require_keys(
        data,
        ("prior_q_window", "path_names", "window_start_indices"),
        str(window_path),
    )
    windows = np.asarray(data["prior_q_window"], dtype=np.float64)
    names = decode_names(data["path_names"])
    starts = np.asarray(data["window_start_indices"], dtype=np.int64).reshape(-1)
    if windows.ndim != 3 or windows.shape[-1] != JOINT_DIM:
        raise ValueError(f"{window_path}: prior_q_window must be [W,H,6], got {windows.shape}")
    if windows.shape[0] != len(names) or windows.shape[0] != starts.shape[0]:
        raise ValueError(
            f"{window_path}: W mismatch windows={windows.shape[0]}, "
            f"path_names={len(names)}, starts={starts.shape[0]}"
        )
    if not np.all(np.isfinite(windows)):
        raise ValueError(f"{window_path}: prior_q_window contains non-finite values")

    horizon = int(windows.shape[1])
    stride, stride_source, stats = resolve_window_stride(window_path, data)
    if "horizon" in data and scalar_int(data["horizon"], f"{window_path}:horizon") != horizon:
        raise ValueError(f"{window_path}: horizon metadata disagrees with prior_q_window")
    if stats is not None and "horizon" in stats:
        stats_horizon = scalar_int(
            stats["horizon"], f"{window_path.parent / 'normalization_stats.npz'}:horizon"
        )
        if stats_horizon != horizon:
            raise ValueError(
                f"{window_path}: H={horizon} disagrees with normalization H={stats_horizon}"
            )
    if horizon > split.trajectory_length:
        raise ValueError(
            f"{window_path}: horizon {horizon} exceeds T={split.trajectory_length}"
        )

    expected_starts = list(
        range(0, split.trajectory_length - horizon + 1, stride)
    )
    rows_by_name: Dict[str, List[int]] = defaultdict(list)
    for row_index, name in enumerate(names):
        rows_by_name[name].append(row_index)

    split_name_set = set(split.names)
    window_name_set = set(names)
    unexpected_names = sorted(window_name_set - split_name_set)
    missing_names = sorted(split_name_set - window_name_set)
    reconstructions: Dict[str, np.ndarray] = {}
    audit_rows: List[Dict[str, Any]] = []

    names_to_audit = list(split.names) + unexpected_names
    for name in names_to_audit:
        row_indices = rows_by_name.get(name, [])
        observed_starts = [int(starts[index]) for index in row_indices]
        counts = Counter(observed_starts)
        duplicate_starts = sorted(start for start, count in counts.items() if count > 1)
        path_in_split = name in split_name_set
        missing_expected = (
            sorted(set(expected_starts) - set(observed_starts)) if path_in_split else []
        )
        unexpected_starts = (
            sorted(set(observed_starts) - set(expected_starts))
            if path_in_split
            else sorted(set(observed_starts))
        )

        accumulated = np.zeros((split.trajectory_length, JOINT_DIM), dtype=np.float64)
        observation_count = np.zeros(split.trajectory_length, dtype=np.int64)
        first_observation = np.full(
            (split.trajectory_length, JOINT_DIM), np.nan, dtype=np.float64
        )
        max_overlap_abs = 0.0
        overlap_disagreement_count = 0
        out_of_bounds_windows = 0

        if path_in_split:
            for row_index in row_indices:
                start = int(starts[row_index])
                end = start + horizon
                if start < 0 or end > split.trajectory_length:
                    out_of_bounds_windows += 1
                    continue
                window = windows[row_index]
                for local_index in range(horizon):
                    time_index = start + local_index
                    value = window[local_index]
                    if observation_count[time_index] > 0:
                        reference = first_observation[time_index]
                        difference = np.abs(value - reference)
                        max_overlap_abs = max(
                            max_overlap_abs, float(np.max(difference))
                        )
                        if not np.allclose(
                            value,
                            reference,
                            rtol=overlap_rtol,
                            atol=overlap_atol,
                        ):
                            overlap_disagreement_count += 1
                    else:
                        first_observation[time_index] = value
                    accumulated[time_index] += value
                    observation_count[time_index] += 1

            reconstructed = np.full_like(accumulated, np.nan)
            covered = observation_count > 0
            reconstructed[covered] = (
                accumulated[covered] / observation_count[covered, None]
            )
            reconstructions[name] = reconstructed
            if save_reconstructions:
                write_joint_csv(
                    output_dir
                    / "current_v5b_prior_reconstructed"
                    / split.label
                    / safe_path_name(name)
                    / "prior_q.csv",
                    reconstructed,
                )
        else:
            reconstructed = np.empty((0, JOINT_DIM), dtype=np.float64)
            covered = np.zeros(split.trajectory_length, dtype=bool)

        missing_timesteps = np.flatnonzero(~covered).tolist() if path_in_split else []
        audit_pass = bool(
            path_in_split
            and row_indices
            and not missing_expected
            and not unexpected_starts
            and not duplicate_starts
            and not missing_timesteps
            and not out_of_bounds_windows
            and overlap_disagreement_count == 0
        )
        audit_rows.append(
            {
                "split": split.label,
                "path_name": name,
                "path_name_in_split": path_in_split,
                "path_name_disagreement": (not path_in_split) or not bool(row_indices),
                "window_npz": str(window_path),
                "trajectory_length": split.trajectory_length if path_in_split else "",
                "horizon": horizon,
                "stride": stride,
                "stride_source": stride_source,
                "expected_window_count": len(expected_starts) if path_in_split else "",
                "observed_window_count": len(row_indices),
                "expected_start_indices": join_ints(expected_starts) if path_in_split else "",
                "observed_start_indices": join_ints(sorted(observed_starts)),
                "missing_expected_start_indices": join_ints(missing_expected),
                "unexpected_start_indices": join_ints(unexpected_starts),
                "duplicate_start_indices": join_ints(duplicate_starts),
                "missing_reconstructed_timesteps": join_ints(missing_timesteps),
                "missing_reconstructed_timestep_count": len(missing_timesteps),
                "out_of_bounds_window_count": out_of_bounds_windows,
                "overlap_observation_count": int(
                    np.sum(np.maximum(observation_count - 1, 0))
                ),
                "overlap_disagreement_count": overlap_disagreement_count,
                "maximum_overlap_absolute_disagreement": max_overlap_abs,
                "audit_pass": audit_pass,
                "saved_reconstruction": (
                    str(
                        output_dir
                        / "current_v5b_prior_reconstructed"
                        / split.label
                        / safe_path_name(name)
                        / "prior_q.csv"
                    )
                    if path_in_split and save_reconstructions
                    else ""
                ),
            }
        )

    metadata = {
        "split": split.label,
        "window_npz": str(window_path),
        "horizon": horizon,
        "stride": stride,
        "stride_source": stride_source,
        "expected_starts": expected_starts,
        "window_path_names_match_split": not unexpected_names and not missing_names,
        "unexpected_window_path_names": unexpected_names,
        "missing_window_path_names": missing_names,
        "audit_pass": all(bool(row["audit_pass"]) for row in audit_rows),
    }
    return reconstructions, audit_rows, metadata


def read_joint_csv(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path}: missing CSV header")
        field_map = {
            str(field).strip().lower(): field
            for field in reader.fieldnames
            if field is not None
        }
        missing = [column for column in JOINT_COLUMNS if column not in field_map]
        if missing:
            raise ValueError(
                f"{path}: not a repository joint CSV; missing columns {missing}"
            )
        rows: List[List[float]] = []
        for row_number, row in enumerate(reader, start=2):
            try:
                rows.append(
                    [float(row[field_map[column]]) for column in JOINT_COLUMNS]
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{path}: non-numeric joint value on CSV row {row_number}"
                ) from exc
    q = np.asarray(rows, dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != JOINT_DIM or q.shape[0] == 0:
        raise ValueError(f"{path}: expected a non-empty [T,6] CSV, got {q.shape}")
    if not np.all(np.isfinite(q)):
        raise ValueError(f"{path}: joint CSV contains non-finite values")
    return q


def split_for_name(
    name: str,
    train_names: set[str],
    test_names: set[str],
    source_hint: str = "",
) -> Optional[str]:
    in_train = name in train_names
    in_test = name in test_names
    if in_train and not in_test:
        return "train"
    if in_test and not in_train:
        return "test"
    hint_parts = {part.lower() for part in Path(source_hint).parts}
    train_hint = any("train" in part for part in hint_parts)
    test_hint = any("test" in part for part in hint_parts)
    if in_train and train_hint and not test_hint:
        return "train"
    if in_test and test_hint and not train_hint:
        return "test"
    return None


def add_csv_to_candidate(
    candidate: Candidate,
    q_path: Path,
    path_name: str,
    train_names: set[str],
    test_names: set[str],
    split_hint: Optional[str] = None,
) -> None:
    try:
        q = read_joint_csv(q_path)
    except Exception as exc:
        candidate.load_errors.append(str(exc))
        candidate.invalid_shapes.append(
            f"{split_hint or 'unassigned'}:{path_name}:unloadable:{q_path}"
        )
        return
    split = split_hint or split_for_name(
        path_name, train_names, test_names, str(q_path)
    )
    if split is None:
        candidate.notes.append(
            f"{q_path}: path {path_name} could not be assigned to train or test"
        )
        return
    candidate.add_trajectory(split, path_name, q, q_path)


def load_known_path_tree(
    candidate: Candidate,
    root: Path,
    split: str,
    q_filename: str,
    train_names: set[str],
    test_names: set[str],
) -> None:
    candidate.add_source(root)
    if not root.exists():
        candidate.notes.append(f"missing {split} tree: {root}")
        return
    found = 0
    for path_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        q_path = path_dir / q_filename
        if not q_path.exists():
            continue
        found += 1
        add_csv_to_candidate(
            candidate,
            q_path,
            path_dir.name,
            train_names,
            test_names,
            split_hint=split,
        )
    if found == 0:
        candidate.notes.append(f"no {q_filename} files directly under {root}/<path>")


def load_indexed_test_tree(
    candidate: Candidate,
    root: Path,
    q_filename: str,
    train_names: set[str],
    test_names: set[str],
    ordered_test_names: Sequence[str],
) -> None:
    """Load a tree whose path_### folder is a test-row index, not a path ID."""
    candidate.add_source(root)
    if not root.exists():
        candidate.notes.append(f"missing indexed test tree: {root}")
        return
    found = 0
    for path_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        match = re.fullmatch(r"path_(\d+)", path_dir.name)
        if match is None:
            continue
        index = int(match.group(1))
        q_path = path_dir / q_filename
        if not q_path.exists():
            continue
        found += 1
        if index >= len(ordered_test_names):
            candidate.invalid_shapes.append(
                f"test:{path_dir.name}:dataset-index-out-of-range:{q_path}"
            )
            continue
        add_csv_to_candidate(
            candidate,
            q_path,
            str(ordered_test_names[index]),
            train_names,
            test_names,
            split_hint="test",
        )
    if found == 0:
        candidate.notes.append(f"no {q_filename} files under indexed folders in {root}")


def resolve_row_path(raw: str, manifest_path: Path) -> Path:
    path = Path(raw).expanduser()
    candidates = (
        path,
        Path.cwd() / path,
        manifest_path.parent / path,
        manifest_path.parent.parent / path,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return path


def load_diffusion_manifest(
    candidate: Candidate,
    manifest_path: Path,
    train_names: set[str],
    test_names: set[str],
    split_hint: str = "test",
) -> None:
    candidate.add_source(manifest_path)
    if not manifest_path.exists():
        candidate.notes.append(f"missing manifest: {manifest_path}")
        return
    with manifest_path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        headers = set(reader.fieldnames or [])
        required = {"path_name", "output_folder"}
        if not required.issubset(headers):
            raise ValueError(
                f"{manifest_path}: expected diffusion manifest columns {sorted(required)}, "
                f"got {sorted(headers)}"
            )
        for row in reader:
            name = str(row["path_name"])
            folder = resolve_row_path(str(row["output_folder"]), manifest_path)
            q_path = folder / "diffusion_pred_q.csv"
            if not q_path.exists():
                best_path = (
                    manifest_path.parent / safe_path_name(name) / "best/diffusion_pred_q.csv"
                )
                if best_path.exists():
                    q_path = best_path
            add_csv_to_candidate(
                candidate,
                q_path,
                name,
                train_names,
                test_names,
                split_hint=split_hint,
            )


def candidate_template(name: str) -> Candidate:
    normalized = name.lower()
    if normalized == "current_v5b_prior":
        return Candidate(
            name=name,
            method_class="mlp_v3_delta_export_used_by_v5b",
            source_checkpoint=str(DATA_ROOT / "path_conditioned_mlp_v3.pt"),
            expert_dependence_status="independent",
            practical_at_inference="yes",
            generation_uses_expert_information="no_at_generation",
            loader_format="stride-aware prior_q_window reconstruction",
        )
    if normalized == "mlp_only":
        return Candidate(
            name=name,
            method_class="canonical_path_conditioned_mlp_full_q",
            source_checkpoint=str(DATA_ROOT / "path_conditioned_mlp_v3.pt"),
            expert_dependence_status="independent",
            practical_at_inference="yes",
            generation_uses_expert_information="no_at_generation",
            loader_format="path tree: path_conditioned_pred_q.csv",
        )
    if normalized in {"mlp_v3_delta_export", "current_v5b_source_csv"}:
        return Candidate(
            name=name,
            method_class="mlp_v3_delta_export",
            source_checkpoint=str(DATA_ROOT / "path_conditioned_mlp_v3.pt"),
            expert_dependence_status="independent",
            practical_at_inference="yes",
            generation_uses_expert_information="no_at_generation",
            loader_format="path tree: predicted_q.csv",
        )
    if normalized == "adaptive_mlp":
        return Candidate(
            name=name,
            method_class="adaptive_mlp_without_ik",
            expert_dependence_status="unknown",
            practical_at_inference="unknown",
            generation_uses_expert_information="unknown",
        )
    if normalized == "adaptive_mlp_ik":
        return Candidate(
            name=name,
            method_class="adaptive_mlp_plus_sequential_ik",
            expert_dependence_status="independent",
            practical_at_inference="yes",
            generation_uses_expert_information="no_at_generation",
            loader_format="path tree: refined_mlp_ik_q.csv",
        )
    if normalized.startswith("mlp_ik_fixed_smooth_"):
        return Candidate(
            name=name,
            method_class="fixed_mlp_plus_sequential_ik",
            expert_dependence_status="independent",
            practical_at_inference="yes",
            generation_uses_expert_information="no_at_generation",
            loader_format="summary-only artifact; no distinct trajectory snapshot",
        )
    if normalized in {"sequential_ik", "expert_ceiling"}:
        return Candidate(
            name=name,
            method_class=(
                "stored_sequential_ik_expert"
                if normalized == "sequential_ik"
                else "expert_fk_reference_ceiling"
            ),
            expert_dependence_status="oracle_or_expert_dependent",
            practical_at_inference="no",
            generation_uses_expert_information="yes_or_stored_as_expert_label",
            loader_format=(
                "path tree: expert_q.csv"
                if normalized == "sequential_ik"
                else "diffusion-v2 expert_q"
            ),
        )
    if "diffusion_v1" in normalized:
        return Candidate(
            name=name,
            method_class=(
                "diffusion_v1_best_of_k_or_reranked"
                if "best_of_k" in normalized or "reranked" in normalized
                else "diffusion_v1_single_sample"
            ),
            source_checkpoint=str(DATA_ROOT / "conditional_diffusion_v1.pt"),
            expert_dependence_status="evaluation_test_label_dependent",
            practical_at_inference="yes",
            generation_uses_expert_information=(
                "no_at_runtime_or_ranking_but_benchmark_test_labels_used_for_checkpoint_selection"
            ),
            loader_format=(
                "diffusion-v1 best-per-path manifest"
                if "best_of_k" in normalized or "reranked" in normalized
                else "path tree: diffusion_pred_q.csv"
            ),
        )
    if normalized.startswith("diffusion_v2_"):
        return Candidate(
            name=name,
            method_class=(
                "trajectory_diffusion_v2_ddim"
                if "ddim" in normalized
                else "trajectory_diffusion_v2_ddpm"
            ),
            source_checkpoint=str(
                DATA_ROOT / "diffusion_v2/conditional_diffusion_v2.pt"
            ),
            expert_dependence_status="evaluation_test_label_dependent",
            practical_at_inference="yes",
            generation_uses_expert_information=(
                "no_at_runtime_but_benchmark_test_labels_used_for_checkpoint_selection"
            ),
            loader_format="diffusion-v2 path manifest",
        )
    if normalized.startswith("diffusion_v3_"):
        return Candidate(
            name=name,
            method_class="trajectory_diffusion_v3_x0",
            source_checkpoint=str(
                DATA_ROOT / "diffusion_v3_x0/conditional_diffusion_v3_x0.pt"
            ),
            expert_dependence_status="evaluation_test_label_dependent",
            practical_at_inference="yes",
            generation_uses_expert_information=(
                "no_at_runtime_but_benchmark_test_labels_used_for_checkpoint_selection"
            ),
            loader_format="diffusion-v3 path manifest",
        )
    if normalized.startswith("diffusion_v4_"):
        if "noised_expert" in normalized:
            return Candidate(
                name=name,
                method_class="diffusion_v4_noised_expert_oracle_refinement",
                source_checkpoint=str(DATA_ROOT / "diffusion_v4_unet/best_model.pt"),
                expert_dependence_status="oracle_or_expert_dependent",
                practical_at_inference="no",
                generation_uses_expert_information="yes",
                loader_format="oracle inventory only",
            )
        if "v1_prior_" in normalized:
            return Candidate(
                name=name,
                method_class="diffusion_v4_v1_prior_benchmark_contaminated",
                source_checkpoint=str(DATA_ROOT / "diffusion_v4_unet/best_model.pt"),
                expert_dependence_status="oracle_or_expert_dependent",
                practical_at_inference="yes_but_not_unbiased_for_this_benchmark",
                generation_uses_expert_information=(
                    "expert_fallback_or_expert_rmse_source_selection_permitted"
                ),
                loader_format="path tree: predicted_q.csv",
            )
        return Candidate(
            name=name,
            method_class=(
                "diffusion_v4_pure_gaussian"
                if "pure_gaussian" in normalized
                else "diffusion_v4_mlp_prior_refinement"
                if "mlp_prior_refined" in normalized
                else "trajectory_diffusion_v4_single_sample"
            ),
            source_checkpoint=str(DATA_ROOT / "diffusion_v4_unet/best_model.pt"),
            expert_dependence_status="evaluation_test_label_dependent",
            practical_at_inference="yes",
            generation_uses_expert_information=(
                "no_at_runtime_but_benchmark_test_labels_used_for_checkpoint_selection"
            ),
            loader_format=(
                "indexed test tree: diffusion_v4_pred_q.csv"
                if "single_sample" in normalized
                else "path tree: predicted_q.csv"
            ),
        )
    if normalized.startswith("residual_v5c_"):
        return Candidate(
            name=name,
            method_class="residual_window_v5c_full_trajectory_refinement",
            source_checkpoint=str(
                DATA_ROOT / "residual_window_predictor_v5c_fk_condition/best_checkpoint.pt"
            ),
            expert_dependence_status="evaluation_test_label_dependent",
            practical_at_inference="yes",
            generation_uses_expert_information=(
                "no_at_runtime_but_test_windows_and_expert_rmse_used_for_checkpoint_selection"
            ),
            loader_format="path tree: candidate_q_alpha_*.csv",
        )
    if normalized.startswith(("scaled_tapered_", "global_anchored_")):
        return Candidate(
            name=name,
            method_class="documented_v5b_full_trajectory_rollout",
            source_checkpoint=str(
                DATA_ROOT / "diffusion_v5b_residual_unet_fk_condition/best_checkpoint.pt"
            ),
            expert_dependence_status="evaluation_test_label_dependent",
            practical_at_inference="yes",
            generation_uses_expert_information=(
                "no_at_runtime_or_ranking_but_benchmark_test_windows_used_for_checkpoint_selection"
            ),
            loader_format="documented rollout path tree",
        )
    if "expert" in normalized or "oracle" in normalized:
        return Candidate(
            name=name,
            method_class="explicit_oracle_candidate",
            expert_dependence_status="oracle_or_expert_dependent",
            practical_at_inference="no",
            generation_uses_expert_information="yes",
        )
    return Candidate(
        name=name,
        method_class="explicit_candidate_unknown_lineage",
        expert_dependence_status="unknown",
        practical_at_inference="unknown",
        generation_uses_expert_information="unknown",
    )


def make_current_candidate(
    train_reconstruction: Mapping[str, np.ndarray],
    test_reconstruction: Mapping[str, np.ndarray],
    train_audit_rows: Sequence[Mapping[str, Any]],
    test_audit_rows: Sequence[Mapping[str, Any]],
    train_window_path: Path,
    test_window_path: Path,
) -> Candidate:
    candidate = candidate_template("current_v5b_prior")
    valid_train_names = {
        str(row["path_name"])
        for row in train_audit_rows
        if bool(row.get("audit_pass"))
    }
    valid_test_names = {
        str(row["path_name"])
        for row in test_audit_rows
        if bool(row.get("audit_pass"))
    }
    candidate.train.update(
        {
            name: q
            for name, q in train_reconstruction.items()
            if name in valid_train_names
            and q.shape[1:] == (JOINT_DIM,)
            and np.all(np.isfinite(q))
        }
    )
    candidate.test.update(
        {
            name: q
            for name, q in test_reconstruction.items()
            if name in valid_test_names
            and q.shape[1:] == (JOINT_DIM,)
            and np.all(np.isfinite(q))
        }
    )
    candidate.add_source(train_window_path)
    candidate.add_source(test_window_path)
    candidate.notes.append(
        "Reconstructed from raw-radian prior_q_window values; overlapping windows are averaged "
        "only after agreement is audited."
    )
    failed_train = sorted(set(train_reconstruction) - set(candidate.train))
    failed_test = sorted(set(test_reconstruction) - set(candidate.test))
    if failed_train or failed_test:
        candidate.notes.append(
            "Reconstruction-audit failures were retained on disk but excluded from evaluation "
            f"and recommendation (train={join_names(failed_train)}, test={join_names(failed_test)})."
        )
    return candidate


def auto_discover_candidates(
    train: SplitDataset,
    test: SplitDataset,
    current: Candidate,
) -> List[Candidate]:
    train_names = set(train.names)
    test_names = set(test.names)
    candidates: List[Candidate] = [current]

    mlp_only = candidate_template("mlp_only")
    load_known_path_tree(
        mlp_only,
        DATA_ROOT / "experts/train",
        "train",
        "path_conditioned_pred_q.csv",
        train_names,
        test_names,
    )
    load_known_path_tree(
        mlp_only,
        DATA_ROOT / "experts/test",
        "test",
        "path_conditioned_pred_q.csv",
        train_names,
        test_names,
    )
    candidates.append(mlp_only)

    delta_export = candidate_template("mlp_v3_delta_export")
    load_known_path_tree(
        delta_export,
        DATA_ROOT / "mlp_v3_train_predictions",
        "train",
        "predicted_q.csv",
        train_names,
        test_names,
    )
    load_known_path_tree(
        delta_export,
        DATA_ROOT / "mlp_v3_test_predictions",
        "test",
        "predicted_q.csv",
        train_names,
        test_names,
    )
    delta_export.notes.append(
        "This is the full-CSV source of current_v5b_prior and is retained separately "
        "to expose reconstruction/source mismatches."
    )
    candidates.append(delta_export)

    adaptive_mlp = candidate_template("adaptive_mlp")
    adaptive_mlp.unavailable_reason = (
        "No distinct full-trajectory Adaptive-MLP-without-IK artifact was found; "
        "the repository contains Adaptive MLP + IK outputs only."
    )
    candidates.append(adaptive_mlp)

    adaptive_ik = candidate_template("adaptive_mlp_ik")
    load_known_path_tree(
        adaptive_ik,
        DATA_ROOT / "experts/train",
        "train",
        "refined_mlp_ik_q.csv",
        train_names,
        test_names,
    )
    load_known_path_tree(
        adaptive_ik,
        DATA_ROOT / "experts/test",
        "test",
        "refined_mlp_ik_q.csv",
        train_names,
        test_names,
    )
    adaptive_ik.add_source(DATA_ROOT / "mlp_ik_refine_test_summary_adaptive.csv")
    adaptive_ik.notes.append(
        "refined_mlp_ik_q.csv is shared by fixed and adaptive refinement runs; "
        "the last writer cannot be proven from the trajectory CSV alone."
    )
    candidates.append(adaptive_ik)

    fixed_ik_specs = (
        (
            "mlp_ik_fixed_smooth_0_01",
            DATA_ROOT / "mlp_ik_refine_test_summary.csv",
        ),
        (
            "mlp_ik_fixed_smooth_0_001",
            DATA_ROOT / "mlp_ik_refine_test_summary_smooth001.csv",
        ),
    )
    for method_name, summary_path in fixed_ik_specs:
        fixed_ik = candidate_template(method_name)
        fixed_ik.add_source(summary_path)
        fixed_ik.unavailable_reason = (
            "A per-path metric summary exists, but fixed and adaptive runs share "
            "refined_mlp_ik_q.csv and no method-specific full-trajectory snapshot exists."
        )
        candidates.append(fixed_ik)

    sequential = candidate_template("sequential_ik")
    load_known_path_tree(
        sequential,
        DATA_ROOT / "experts/train",
        "train",
        "expert_q.csv",
        train_names,
        test_names,
    )
    load_known_path_tree(
        sequential,
        DATA_ROOT / "experts/test",
        "test",
        "expert_q.csv",
        train_names,
        test_names,
    )
    sequential.add_source(Path("batch_generate_ik_experts.py"))
    sequential.add_source(DATA_ROOT / "cold_ik_test_timed")
    sequential.notes.append(
        "Stored under the expert label; cold_ik_test_timed is the timed test copy. "
        "Both are excluded from practical bootstrap selection."
    )
    candidates.append(sequential)

    diffusion_single = candidate_template("diffusion_v1_single_sample")
    load_known_path_tree(
        diffusion_single,
        DATA_ROOT / "diffusion_v1_samples",
        "test",
        "diffusion_pred_q.csv",
        train_names,
        test_names,
    )
    candidates.append(diffusion_single)

    ranked_root = DATA_ROOT / "diffusion_v1_ranked_samples"
    manifests = sorted(
        ranked_root.glob("k*/diffusion_v1_best_per_path.csv"),
        key=lambda path: (
            int(path.parent.name[1:])
            if path.parent.name[1:].isdigit()
            else sys.maxsize
        ),
    )
    manifest_candidates: List[Tuple[float, int, Candidate]] = []
    for manifest in manifests:
        k_text = manifest.parent.name
        name = f"diffusion_v1_best_of_k_{k_text}"
        item = candidate_template(name)
        try:
            load_diffusion_manifest(item, manifest, train_names, test_names)
        except Exception as exc:
            item.load_errors.append(str(exc))
        costs: List[float] = []
        try:
            with manifest.open("r", newline="", encoding="utf-8-sig") as handle:
                for row in csv.DictReader(handle):
                    costs.append(float(row["total_cost"]))
        except Exception as exc:
            item.notes.append(f"could not summarize manifest total_cost: {exc}")
        mean_cost = float(np.mean(costs)) if costs else float("inf")
        k_value = int(k_text[1:]) if k_text[1:].isdigit() else -1
        manifest_candidates.append((mean_cost, k_value, item))

    if manifest_candidates:
        selected_index = min(
            range(len(manifest_candidates)),
            key=lambda index: (
                manifest_candidates[index][0],
                -manifest_candidates[index][1],
            ),
        )
        for index, (_, _, item) in enumerate(manifest_candidates):
            if index == selected_index:
                item.name = "diffusion_v1_best_of_k"
                item.notes.append(
                    "Canonical best-of-K artifact chosen by the manifest's original "
                    "multi-factor total_cost mean, not Cartesian error alone."
                )
            candidates.append(item)
    else:
        missing = candidate_template("diffusion_v1_best_of_k")
        missing.unavailable_reason = (
            f"No k*/diffusion_v1_best_per_path.csv manifests under {ranked_root}"
        )
        candidates.append(missing)

    reranked_manifests = sorted(ranked_root.glob("k*/reranked_best_*.csv"))
    for manifest in reranked_manifests:
        k_text = manifest.parent.name
        objective = manifest.stem.removeprefix("reranked_best_")
        method_name = f"diffusion_v1_{k_text}_reranked_{objective}"
        reranked = candidate_template(method_name)
        try:
            load_diffusion_manifest(
                reranked,
                manifest,
                train_names,
                test_names,
                split_hint="test",
            )
        except Exception as exc:
            reranked.load_errors.append(str(exc))
            reranked.unavailable_reason = str(exc)
        reranked.notes.append(
            "Reranked from already generated diffusion-v1 samples using the documented "
            f"{objective} objective; no expert joint trajectory is used for ranking."
        )
        candidates.append(reranked)

    newer_manifest_specs = (
        (
            "diffusion_v2_ddpm_single_sample",
            DATA_ROOT / "diffusion_v2/samples/diffusion_v2_sample_summary.csv",
        ),
        (
            "diffusion_v2_ddim_single_sample",
            DATA_ROOT / "diffusion_v2/ddim_samples/diffusion_v2_ddim_sample_summary.csv",
        ),
        (
            "diffusion_v3_x0_single_sample",
            DATA_ROOT / "diffusion_v3_x0/samples/diffusion_v3_x0_sample_summary.csv",
        ),
    )
    for method_name, manifest in newer_manifest_specs:
        diffusion_candidate = candidate_template(method_name)
        try:
            load_diffusion_manifest(
                diffusion_candidate,
                manifest,
                train_names,
                test_names,
                split_hint="test",
            )
        except Exception as exc:
            diffusion_candidate.load_errors.append(str(exc))
            diffusion_candidate.unavailable_reason = str(exc)
        diffusion_candidate.notes.append(
            "The repository manifest is loaded once; an output_folder row and its "
            "path/<name>/best fallback are alternative references, not extra samples."
        )
        candidates.append(diffusion_candidate)

    v4_root = DATA_ROOT / "diffusion_v4_unet"
    v4_single = candidate_template("diffusion_v4_single_sample")
    load_indexed_test_tree(
        v4_single,
        v4_root / "samples_single",
        "diffusion_v4_pred_q.csv",
        train_names,
        test_names,
        test.names,
    )
    v4_single.notes.append(
        "samples_single/path_### encodes the diffusion-v2 test row index; it is mapped "
        "through test.path_names rather than interpreted as a literal path ID. The v4 "
        "best checkpoint was selected on this benchmark test split, so the stored output "
        "is diagnostic-only for an unbiased comparison."
    )
    candidates.append(v4_single)

    v4_debug_duplicate = Candidate(
        name="diffusion_v4_single_debugged_duplicate",
        method_class="duplicate_result_inventory",
        expert_dependence_status="evaluation_test_label_dependent",
        practical_at_inference="yes",
        generation_uses_expert_information=(
            "no_at_runtime_but_benchmark_test_labels_used_for_checkpoint_selection"
        ),
        loader_format="duplicate indexed test tree",
        unavailable_reason=(
            "samples_single_debugged duplicates samples_single and is inventoried without "
            "loading a second copy into the comparison."
        ),
    )
    v4_debug_duplicate.add_source(v4_root / "samples_single_debugged")
    candidates.append(v4_debug_duplicate)

    refinement_root = v4_root / "prior_refinement_outputs"
    mlp_refinement_root = refinement_root / "mlp_prior_v4_refine"
    for time_dir in sorted(
        (mlp_refinement_root / "prior_refined").glob("t_*"),
        key=lambda path: int(path.name.removeprefix("t_")),
    ):
        refined = candidate_template(
            f"diffusion_v4_mlp_prior_refined_{time_dir.name}"
        )
        load_known_path_tree(
            refined,
            time_dir,
            "test",
            "predicted_q.csv",
            train_names,
            test_names,
        )
        refined.notes.append(
            "Runtime-practical v4 denoising refinement initialized from the deployable "
            "MLP prior; benchmark-test checkpoint selection makes this stored result "
            "recommendation-ineligible."
        )
        candidates.append(refined)
    for time_dir in sorted(
        (mlp_refinement_root / "pure_gaussian").glob("t_*"),
        key=lambda path: int(path.name.removeprefix("t_")),
    ):
        gaussian = candidate_template(
            f"diffusion_v4_pure_gaussian_{time_dir.name}"
        )
        load_known_path_tree(
            gaussian,
            time_dir,
            "test",
            "predicted_q.csv",
            train_names,
            test_names,
        )
        gaussian.notes.append(
            "Canonical copy of the pure-Gaussian v4 result family. Its runtime input is "
            "expert-independent, but the v4 checkpoint was selected on benchmark test labels."
        )
        candidates.append(gaussian)

    mlp_prior_duplicate = Candidate(
        name="diffusion_v4_mlp_prior_only_duplicate",
        method_class="duplicate_result_inventory",
        expert_dependence_status="independent",
        practical_at_inference="yes",
        generation_uses_expert_information="no_at_generation",
        loader_format="duplicate path tree",
        unavailable_reason=(
            "prior_only duplicates the current MLP-v3 delta-export source and is not "
            "double-counted as a distinct method."
        ),
    )
    mlp_prior_duplicate.add_source(mlp_refinement_root / "prior_only")
    candidates.append(mlp_prior_duplicate)

    mlp_noised_expert = candidate_template("diffusion_v4_mlp_noised_expert_oracle")
    mlp_noised_expert.add_source(mlp_refinement_root / "noised_expert")
    mlp_noised_expert.unavailable_reason = (
        "noised_expert starts from the stored expert trajectory and is oracle-only; "
        "it is inventoried but excluded from practical bootstrap comparison."
    )
    candidates.append(mlp_noised_expert)

    v1_refinement_root = refinement_root / "v1_prior_v4_refine"
    v1_prior_only = candidate_template(
        "diffusion_v4_v1_prior_only_unknown_lineage"
    )
    load_known_path_tree(
        v1_prior_only,
        v1_refinement_root / "prior_only",
        "test",
        "predicted_q.csv",
        train_names,
        test_names,
    )
    v1_prior_only.notes.append(
        "The generator permits expert fallback and automatic expert-RMSE source selection; "
        "stored provenance cannot prove this prior is expert-independent."
    )
    candidates.append(v1_prior_only)
    for time_dir in sorted(
        (v1_refinement_root / "prior_refined").glob("t_*"),
        key=lambda path: int(path.name.removeprefix("t_")),
    ):
        v1_refined = candidate_template(
            f"diffusion_v4_v1_prior_refined_{time_dir.name}_unknown_lineage"
        )
        load_known_path_tree(
            v1_refined,
            time_dir,
            "test",
            "predicted_q.csv",
            train_names,
            test_names,
        )
        v1_refined.notes.append(
            "Evaluated for visibility but recommendation-ineligible: the generator permits "
            "expert fallback and automatic expert-RMSE prior selection."
        )
        candidates.append(v1_refined)

    v1_gaussian_duplicate = Candidate(
        name="diffusion_v4_v1_root_pure_gaussian_duplicate",
        method_class="duplicate_result_inventory",
        expert_dependence_status="evaluation_test_label_dependent",
        practical_at_inference="yes",
        generation_uses_expert_information=(
            "no_at_runtime_but_benchmark_test_labels_used_for_checkpoint_selection"
        ),
        loader_format="duplicate path tree family",
        unavailable_reason=(
            "This pure_gaussian family duplicates the canonical copy under "
            "mlp_prior_v4_refine and is not loaded twice."
        ),
    )
    v1_gaussian_duplicate.add_source(v1_refinement_root / "pure_gaussian")
    candidates.append(v1_gaussian_duplicate)

    v1_noised_expert = candidate_template("diffusion_v4_v1_noised_expert_oracle")
    v1_noised_expert.add_source(v1_refinement_root / "noised_expert")
    v1_noised_expert.unavailable_reason = (
        "noised_expert is expert-initialized oracle output and cannot be a deployable prior."
    )
    candidates.append(v1_noised_expert)

    v5c_root = (
        DATA_ROOT
        / "residual_window_predictor_v5c_fk_condition/full_trajectory_fk_eval"
    )
    for alpha in ("0.05", "0.1", "0.25"):
        v5c = candidate_template(f"residual_v5c_alpha_{alpha}")
        load_known_path_tree(
            v5c,
            v5c_root,
            "test",
            f"candidate_q_alpha_{alpha}.csv",
            train_names,
            test_names,
        )
        v5c.notes.append(
            f"Full-trajectory v5c FK-conditioned residual refinement, alpha={alpha}. "
            "The best checkpoint was selected with test-window expert RMSE, so this is "
            "evaluated diagnostically but cannot be recommended on the same benchmark."
        )
        candidates.append(v5c)
    v5c_prior_duplicate = Candidate(
        name="residual_v5c_prior_q_duplicate",
        method_class="duplicate_result_inventory",
        expert_dependence_status="independent",
        practical_at_inference="yes",
        generation_uses_expert_information="no_at_generation",
        loader_format="duplicate path tree",
        unavailable_reason=(
            "full_trajectory_fk_eval/<path>/prior_q.csv duplicates the current v5b "
            "source prior and is not treated as a new method."
        ),
    )
    v5c_prior_duplicate.add_source(v5c_root)
    candidates.append(v5c_prior_duplicate)

    documented_rollout_specs = (
        (
            "scaled_tapered_buffer_only",
            DATA_ROOT / "scaled_tapered_receding_horizon_rollout/trajectories",
            "buffer_only_q.csv",
        ),
        (
            "scaled_tapered_diffusion_lexicographic",
            DATA_ROOT / "scaled_tapered_receding_horizon_rollout/trajectories",
            "diffusion_lexicographic_q.csv",
        ),
        (
            "scaled_tapered_diffusion_discounted_hard_gate",
            DATA_ROOT / "scaled_tapered_receding_horizon_rollout/trajectories",
            "diffusion_discounted_hard_gate_q.csv",
        ),
        (
            "global_anchored_buffer_only",
            DATA_ROOT / "global_anchored_receding_horizon_rollout/trajectories",
            "buffer_only_q.csv",
        ),
        (
            "global_anchored_global_reference",
            DATA_ROOT / "global_anchored_receding_horizon_rollout/trajectories",
            "global_reference_q.csv",
        ),
        (
            "global_anchored_base_tail",
            DATA_ROOT / "global_anchored_receding_horizon_rollout/trajectories",
            "base_tail_q.csv",
        ),
        (
            "global_anchored_full_selected_tail",
            DATA_ROOT / "global_anchored_receding_horizon_rollout/trajectories",
            "full_selected_tail_q.csv",
        ),
        (
            "global_anchored_decayed_selected_tail",
            DATA_ROOT / "global_anchored_receding_horizon_rollout/trajectories",
            "decayed_selected_tail_q.csv",
        ),
        (
            "global_anchored_final_tail",
            DATA_ROOT / "global_anchored_receding_horizon_rollout/trajectories",
            "global_anchored_tail_q.csv",
        ),
    )
    for method_name, result_root, q_filename in documented_rollout_specs:
        rollout_candidate = candidate_template(method_name)
        load_known_path_tree(
            rollout_candidate,
            result_root,
            "test",
            q_filename,
            train_names,
            test_names,
        )
        rollout_candidate.notes.append(
            "Documented full-trajectory diagnostic/rollout result; evaluated as another "
            "available trajectory but not full train/test bootstrap coverage. The recorded "
            "v5b best checkpoint was selected on benchmark test-window target loss, so this "
            "stored result is recommendation-ineligible on the same benchmark."
        )
        candidates.append(rollout_candidate)

    incompatible = Candidate(
        name="warm_start_concatenated_diagnostic",
        method_class="documented_but_not_path_aligned",
        expert_dependence_status="independent",
        practical_at_inference="yes",
        generation_uses_expert_information="no_at_generation_or_ranking",
        loader_format="incompatible concatenated path-range folders",
        unavailable_reason=(
            "warm_start_action_buffer_diagnostic stores concatenated path-range folders "
            "rather than one [T,6] trajectory per diffusion-v2 path, so it cannot be "
            "compared path-for-path without fabricating an alignment."
        ),
    )
    incompatible.add_source(DATA_ROOT / "warm_start_action_buffer_diagnostic")
    candidates.append(incompatible)

    return candidates


def parse_candidate_assignment(text: str) -> Tuple[str, Path]:
    if "=" not in text:
        raise ValueError(f"--candidate_prior must be NAME=PATH, got {text!r}")
    name, raw_path = text.split("=", 1)
    name = name.strip()
    raw_path = raw_path.strip()
    if not name or not raw_path:
        raise ValueError(f"--candidate_prior must be NAME=PATH, got {text!r}")
    return name, Path(raw_path).expanduser()


def load_explicit_npz(
    candidate: Candidate,
    path: Path,
    train: SplitDataset,
    test: SplitDataset,
) -> None:
    data = load_npz_dict(path)
    candidate.add_source(path)
    train_names = set(train.names)
    test_names = set(test.names)
    if {"prior_q_window", "path_names", "window_start_indices"}.issubset(data):
        overlaps_train = set(decode_names(data["path_names"])) & train_names
        overlaps_test = set(decode_names(data["path_names"])) & test_names
        path_parts = {part.lower() for part in path.parts}
        train_hint = any("train" in part for part in path_parts)
        test_hint = any("test" in part for part in path_parts)
        if train_hint and not test_hint:
            split = train
        elif test_hint and not train_hint:
            split = test
        elif overlaps_train and not overlaps_test:
            split = train
        elif overlaps_test and not overlaps_train:
            split = test
        else:
            split = None
        if split is None:
            raise ValueError(
                f"{path}: could not disambiguate train/test window NPZ from path names; "
                "include train or test in its path"
            )
        reconstructed, reconstruction_audit, _ = reconstruct_current_prior(
            split=split,
            window_path=path,
            output_dir=Path("."),
            overlap_atol=OVERLAP_ATOL,
            overlap_rtol=OVERLAP_RTOL,
            save_reconstructions=False,
        )
        valid_names = {
            str(row["path_name"])
            for row in reconstruction_audit
            if bool(row.get("audit_pass"))
        }
        target = candidate.train if split.label == "train" else candidate.test
        target.update(
            {
                name: q
                for name, q in reconstructed.items()
                if name in valid_names and np.all(np.isfinite(q))
            }
        )
        failed_names = sorted(set(reconstructed) - valid_names)
        if failed_names:
            candidate.notes.append(
                "Explicit window reconstructions failed audit and were excluded: "
                + join_names(failed_names)
            )
        candidate.loader_format = "v5/v5b prior_q_window NPZ"
        return
    if {"expert_q", "path_names"}.issubset(data):
        q_all = np.asarray(data["expert_q"], dtype=np.float64)
        names = decode_names(data["path_names"])
        if q_all.ndim != 3 or q_all.shape[-1] != JOINT_DIM or len(names) != q_all.shape[0]:
            raise ValueError(f"{path}: expert_q/path_names shape mismatch")
        candidate.expert_dependence_status = "oracle_or_expert_dependent"
        candidate.practical_at_inference = "no"
        candidate.generation_uses_expert_information = "yes"
        candidate.loader_format = "diffusion dataset expert_q NPZ (oracle)"
        for index, name in enumerate(names):
            split_name = split_for_name(name, train_names, test_names, str(path))
            if split_name is None:
                candidate.notes.append(f"{path}: ignored unmatched NPZ path {name}")
                continue
            candidate.add_trajectory(split_name, name, q_all[index], path)
        return
    raise ValueError(
        f"{path}: unsupported NPZ structure. Supported repository keys are "
        "prior_q_window/path_names/window_start_indices or expert_q/path_names; "
        f"available={sorted(data)}"
    )


def csv_headers(path: Path) -> List[str]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.reader(handle)
        try:
            return [field.strip() for field in next(reader)]
        except StopIteration:
            return []


def infer_single_csv_name(path: Path, train: SplitDataset, test: SplitDataset) -> str:
    for part in reversed(path.parts[:-1]):
        if part in set(train.names) or part in set(test.names):
            return part
    return path.parent.name


def load_adaptive_summary_csv(
    candidate: Candidate,
    path: Path,
    train: SplitDataset,
    test: SplitDataset,
) -> None:
    roots = (
        path.parent / "experts/test",
        path.parent.parent / "experts/test",
        DATA_ROOT / "experts/test",
    )
    q_root = next((root for root in roots if root.exists()), roots[-1])
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            name = str(row.get("path_id") or row.get("path_name") or "")
            if not name:
                continue
            add_csv_to_candidate(
                candidate,
                q_root / safe_path_name(name) / "refined_mlp_ik_q.csv",
                name,
                set(train.names),
                set(test.names),
                split_hint="test",
            )
    candidate.loader_format = "adaptive MLP+IK summary plus refined_mlp_ik_q.csv tree"
    candidate.notes.append(
        "The summary does not contain trajectory paths; repository experts/test layout was used."
    )


def load_explicit_csv(
    candidate: Candidate,
    path: Path,
    train: SplitDataset,
    test: SplitDataset,
) -> None:
    headers = set(csv_headers(path))
    candidate.add_source(path)
    train_names = set(train.names)
    test_names = set(test.names)
    if set(JOINT_COLUMNS).issubset(headers):
        name = infer_single_csv_name(path, train, test)
        add_csv_to_candidate(candidate, path, name, train_names, test_names)
        candidate.loader_format = "single repository joint CSV"
        return
    if {"path_name", "output_folder"}.issubset(headers):
        load_diffusion_manifest(candidate, path, train_names, test_names)
        candidate.loader_format = "diffusion path manifest"
        return
    if "path_id" in headers and "adaptive_stage" in headers:
        load_adaptive_summary_csv(candidate, path, train, test)
        return
    if "path_id" in headers and "after_mean_error" in headers:
        raise ValueError(
            f"{path}: fixed MLP+IK summary has metrics only. Its shared "
            "refined_mlp_ik_q.csv files may have been overwritten by the adaptive run, "
            "so the fixed trajectories are unavailable."
        )
    raise ValueError(
        f"{path}: aggregate/result CSV has no loadable trajectories. "
        "Supported actual structures are q1..q6 rows, diffusion path manifests "
        "with path_name/output_folder, and adaptive MLP+IK path summaries."
    )


def preferred_filenames_for_candidate(name: str) -> Tuple[str, ...]:
    normalized = name.lower()
    if normalized == "mlp_only":
        return ("path_conditioned_pred_q.csv",)
    if "adaptive" in normalized and "ik" in normalized:
        return ("refined_mlp_ik_q.csv",)
    if normalized.startswith(("diffusion_v1", "diffusion_v2", "diffusion_v3")):
        return ("diffusion_pred_q.csv",)
    if normalized.startswith("diffusion_v4"):
        if "single" in normalized:
            return ("diffusion_v4_pred_q.csv",)
        return ("predicted_q.csv",)
    if normalized.startswith("residual_v5c"):
        for alpha in ("0.05", "0.1", "0.25"):
            if alpha in normalized:
                return (f"candidate_q_alpha_{alpha}.csv",)
        return (
            "candidate_q_alpha_0.05.csv",
            "candidate_q_alpha_0.1.csv",
            "candidate_q_alpha_0.25.csv",
        )
    if "sequential" in normalized or "expert" in normalized or "oracle" in normalized:
        return ("expert_q.csv",)
    if "v5b" in normalized or "delta_export" in normalized:
        return ("predicted_q.csv",)
    return KNOWN_Q_FILENAMES


def load_explicit_directory(
    candidate: Candidate,
    root: Path,
    train: SplitDataset,
    test: SplitDataset,
) -> None:
    candidate.add_source(root)
    preferred = preferred_filenames_for_candidate(candidate.name)
    train_names = set(train.names)
    test_names = set(test.names)

    if (
        candidate.name.lower().startswith("diffusion_v4")
        and "single" in candidate.name.lower()
    ):
        load_indexed_test_tree(
            candidate,
            root,
            "diffusion_v4_pred_q.csv",
            train_names,
            test_names,
            test.names,
        )
        if candidate.test or candidate.invalid_shapes:
            candidate.loader_format = "indexed diffusion-v4 test tree"
            return

    def load_split_root(split_root: Path, split_name: Optional[str]) -> int:
        count = 0
        for path_dir in sorted(path for path in split_root.iterdir() if path.is_dir()):
            matches = [path_dir / filename for filename in preferred if (path_dir / filename).exists()]
            if not matches:
                matches = [
                    path_dir / "best" / filename
                    for filename in preferred
                    if (path_dir / "best" / filename).exists()
                ]
            if len(matches) > 1:
                candidate.notes.append(
                    f"{path_dir}: multiple known q artifacts; ignored ambiguous directory"
                )
                continue
            if len(matches) == 1:
                add_csv_to_candidate(
                    candidate,
                    matches[0],
                    path_dir.name,
                    train_names,
                    test_names,
                    split_hint=split_name,
                )
                count += 1
        return count

    found = 0
    if (root / "train").is_dir():
        found += load_split_root(root / "train", "train")
    if (root / "test").is_dir():
        found += load_split_root(root / "test", "test")
    if found == 0:
        found += load_split_root(root, None)
    if found == 0:
        direct_matches = [root / filename for filename in preferred if (root / filename).exists()]
        if len(direct_matches) == 1:
            name = infer_single_csv_name(direct_matches[0], train, test)
            add_csv_to_candidate(
                candidate,
                direct_matches[0],
                name,
                train_names,
                test_names,
            )
            found += 1
    if found == 0:
        candidate.unavailable_reason = (
            f"No unambiguous repository-native joint trajectory files found under {root}"
        )
    candidate.loader_format = "repository path tree"


def load_explicit_candidate(
    name: str,
    path: Path,
    train: SplitDataset,
    test: SplitDataset,
) -> Candidate:
    candidate = candidate_template(name)
    if not path.exists():
        candidate.unavailable_reason = f"Explicit candidate path does not exist: {path}"
        candidate.add_source(path)
        return candidate
    try:
        if path.is_dir():
            load_explicit_directory(candidate, path, train, test)
        elif path.suffix.lower() == ".npz":
            load_explicit_npz(candidate, path, train, test)
        elif path.suffix.lower() == ".csv":
            load_explicit_csv(candidate, path, train, test)
        else:
            raise ValueError(
                f"{path}: candidate must be a repository NPZ, CSV, or path tree"
            )
    except Exception as exc:
        candidate.load_errors.append(str(exc))
        if not candidate.available:
            candidate.unavailable_reason = str(exc)
    return candidate


def apply_explicit_candidates(
    candidates: List[Candidate],
    assignments: Sequence[str],
    train: SplitDataset,
    test: SplitDataset,
) -> List[Candidate]:
    if not assignments:
        return candidates
    parsed: List[Tuple[str, Path]] = []
    names_seen: set[str] = set()
    for assignment in assignments:
        name, path = parse_candidate_assignment(assignment)
        if name == "expert_ceiling":
            raise ValueError(
                "expert_ceiling is a reserved reference method and cannot be replaced"
            )
        if name in names_seen:
            raise ValueError(f"Repeated explicit candidate name {name!r}")
        names_seen.add(name)
        parsed.append((name, path))
    by_name = {candidate.name: candidate for candidate in candidates}
    order = [candidate.name for candidate in candidates]
    for name, path in parsed:
        explicit = load_explicit_candidate(name, path, train, test)
        if name not in by_name:
            order.append(name)
        else:
            explicit.notes.append("Explicit --candidate_prior replaced auto-discovery.")
        by_name[name] = explicit
    return [by_name[name] for name in order]


def make_expert_ceiling(train: SplitDataset, test: SplitDataset) -> Candidate:
    candidate = candidate_template("expert_ceiling")
    candidate.train.update({name: np.array(q, copy=True) for name, q in train.expert_q.items()})
    candidate.test.update({name: np.array(q, copy=True) for name, q in test.expert_q.items()})
    candidate.add_source(train.source)
    candidate.add_source(test.source)
    candidate.notes.append("Reference ceiling only; never eligible as a practical prior.")
    return candidate


def trajectory_fingerprint(q: np.ndarray) -> str:
    array = np.asarray(q, dtype=np.float64)
    rounded = np.round(array, decimals=8)
    payload = str(tuple(int(value) for value in rounded.shape)).encode("utf-8")
    payload += rounded.tobytes(order="C")
    return hashlib.sha256(payload).hexdigest()


def cross_split_identical_trajectories(
    train_trajectories: Mapping[str, np.ndarray],
    test_trajectories: Mapping[str, np.ndarray],
) -> List[str]:
    train_by_fingerprint: Dict[str, List[str]] = defaultdict(list)
    for name, q in train_trajectories.items():
        train_by_fingerprint[trajectory_fingerprint(q)].append(name)
    pairs: List[str] = []
    for test_name, q in test_trajectories.items():
        fingerprint = trajectory_fingerprint(q)
        for train_name in train_by_fingerprint.get(fingerprint, []):
            pairs.append(f"train:{train_name}=test:{test_name}")
    return sorted(pairs)


def candidate_availability_row(
    candidate: Candidate,
    train: SplitDataset,
    test: SplitDataset,
) -> Dict[str, Any]:
    train_expected = set(train.names)
    test_expected = set(test.names)
    train_loaded = set(candidate.train)
    test_loaded = set(candidate.test)
    train_available = train_loaded & train_expected
    test_available = test_loaded & test_expected
    train_missing = sorted(train_expected - train_available)
    test_missing = sorted(test_expected - test_available)
    train_extra = sorted(train_loaded - train_expected)
    test_extra = sorted(test_loaded - test_expected)

    train_shape_errors: List[str] = []
    test_shape_errors: List[str] = []
    for split_name, trajectories, dataset in (
        ("train", candidate.train, train),
        ("test", candidate.test, test),
    ):
        expected_shape = (dataset.trajectory_length, JOINT_DIM)
        for name, q in trajectories.items():
            if q.shape != expected_shape:
                error = f"{split_name}:{name}:{tuple(int(value) for value in q.shape)}"
                if split_name == "train":
                    train_shape_errors.append(error)
                else:
                    test_shape_errors.append(error)
    shape_errors = list(candidate.invalid_shapes) + train_shape_errors + test_shape_errors
    train_overlap_test = sorted(train_loaded & test_loaded)
    identical_trajectory_pairs = cross_split_identical_trajectories(
        candidate.train, candidate.test
    )
    identical_desired_path_pairs = cross_split_identical_trajectories(
        train.desired, test.desired
    )
    full_train = (
        len(train_missing) == 0
        and not train_shape_errors
        and not candidate.duplicate_train_names
        and not candidate.invalid_shapes
    )
    full_test = (
        len(test_missing) == 0
        and not test_shape_errors
        and not candidate.duplicate_test_names
        and not candidate.invalid_shapes
    )
    available = candidate.available
    unavailable_reason = candidate.unavailable_reason
    if not available and not unavailable_reason:
        unavailable_reason = "; ".join(candidate.load_errors) or "No trajectory artifacts found"
    return {
        "method": candidate.name,
        "method_class": candidate.method_class,
        "available": available,
        "loader_format": candidate.loader_format,
        "source_result_paths": join_names(candidate.source_paths),
        "source_checkpoint": candidate.source_checkpoint,
        "generation_uses_expert_information": candidate.generation_uses_expert_information,
        "expert_dependence_status": candidate.expert_dependence_status,
        "practical_at_inference": candidate.practical_at_inference,
        "expected_training_paths": len(train_expected),
        "expected_test_paths": len(test_expected),
        "loaded_training_trajectories": len(train_loaded),
        "loaded_test_trajectories": len(test_loaded),
        "training_paths_available": len(train_available),
        "test_paths_available": len(test_available),
        "training_coverage_fraction": (
            len(train_available) / len(train_expected) if train_expected else float("nan")
        ),
        "test_coverage_fraction": (
            len(test_available) / len(test_expected) if test_expected else float("nan")
        ),
        "full_training_coverage": full_train,
        "full_test_coverage": full_test,
        "full_train_and_test_coverage": full_train and full_test,
        "missing_training_path_names": join_names(train_missing),
        "missing_test_path_names": join_names(test_missing),
        "unexpected_training_path_names": join_names(train_extra),
        "unexpected_test_path_names": join_names(test_extra),
        "duplicated_training_path_names": join_names(
            sorted(set(candidate.duplicate_train_names))
        ),
        "duplicated_test_path_names": join_names(
            sorted(set(candidate.duplicate_test_names))
        ),
        "all_trajectories_shape_T_6": available and not shape_errors,
        "invalid_trajectory_shapes": join_names(shape_errors),
        "train_test_path_name_collision_count": len(train_overlap_test),
        "train_test_path_name_collisions": join_names(train_overlap_test),
        "train_test_identical_trajectory_count": len(identical_trajectory_pairs),
        "train_test_identical_trajectories": join_names(identical_trajectory_pairs),
        "evaluation_dataset_train_test_identical_desired_path_count": len(
            identical_desired_path_pairs
        ),
        "evaluation_dataset_train_test_identical_desired_paths": join_names(
            identical_desired_path_pairs
        ),
        "unavailable_reason": unavailable_reason,
        "load_errors": join_names(candidate.load_errors),
        "notes": join_names(candidate.notes),
    }


class FKComputer:
    def __init__(self, robot: Any, joint_names: Sequence[str], ee_link: str) -> None:
        if len(joint_names) != JOINT_DIM or len(set(joint_names)) != JOINT_DIM:
            raise ValueError(
                "xMateCR7 audit requires exactly six unique active joints, "
                f"got {joint_names}"
            )
        self.robot = robot
        self.joint_names = tuple(joint_names)
        self.ee_link = str(ee_link)
        self.cache: Dict[str, np.ndarray] = {}

    def positions(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=np.float64)
        if q.ndim != 2 or q.shape[1] != JOINT_DIM:
            raise ValueError(f"FK q must have [T,6], got {q.shape}")
        digest = hashlib.sha256(q.tobytes(order="C")).hexdigest()
        cached = self.cache.get(digest)
        if cached is not None:
            return np.array(cached, copy=True)
        positions = np.empty((q.shape[0], 3), dtype=np.float64)
        for index, row in enumerate(q):
            cfg = {
                joint_name: float(value)
                for joint_name, value in zip(self.joint_names, row)
            }
            self.robot.update_cfg(cfg)
            transform = self.robot.get_transform(frame_to=self.ee_link)
            matrix = np.asarray(transform, dtype=np.float64)
            if matrix.shape != (4, 4):
                raise ValueError(
                    f"get_transform(frame_to={self.ee_link!r}) returned {matrix.shape}, "
                    "expected a 4x4 matrix"
                )
            positions[index] = matrix[:3, 3]
        self.cache[digest] = np.array(positions, copy=True)
        return positions


def resolve_project_path(path: Path) -> Path:
    if path.exists():
        return path
    script_dir = Path(__file__).resolve().parent
    for candidate in (script_dir / path, script_dir.parent / path):
        if candidate.exists():
            return candidate
    return path


def load_fk_context(args: argparse.Namespace) -> Tuple[FKComputer, np.ndarray, np.ndarray, Dict[str, str]]:
    try:
        from generate_ik_seed_path import (
            DEFAULT_EE_LINK,
            DEFAULT_JOINT_NAMES,
            DEFAULT_URDF_PATH,
            get_joint_bounds,
            load_robot,
        )
    except Exception as exc:
        raise ImportError(
            "Could not import the repository xMateCR7 FK helpers from "
            "generate_ik_seed_path.py"
        ) from exc

    urdf_path = resolve_project_path(
        Path(DEFAULT_URDF_PATH) if args.urdf_path is None else args.urdf_path
    )
    ee_link = DEFAULT_EE_LINK if args.ee_link is None else args.ee_link
    if args.joint_names is None:
        joint_names = tuple(DEFAULT_JOINT_NAMES)
    else:
        joint_names = tuple(
            name.strip() for name in args.joint_names.split(",") if name.strip()
        )
    if len(joint_names) != JOINT_DIM or len(set(joint_names)) != JOINT_DIM:
        raise ValueError(
            "--joint_names must contain exactly six unique active joints, "
            f"got {joint_names}"
        )
    robot = load_robot(urdf_path)
    bounds = get_joint_bounds(
        robot,
        joint_names,
        args.fallback_joint_min,
        args.fallback_joint_max,
    )
    lower = np.asarray([bound[0] for bound in bounds], dtype=np.float64)
    upper = np.asarray([bound[1] for bound in bounds], dtype=np.float64)
    device_note = (
        "yourdfpy FK is CPU-only; CUDA request was not used"
        if args.device == "cuda"
        else "yourdfpy FK uses CPU"
    )
    return (
        FKComputer(robot, joint_names, str(ee_link)),
        lower,
        upper,
        {
            "requested_device": args.device,
            "fk_device": "cpu",
            "device_note": device_note,
            "urdf_path": str(urdf_path),
            "ee_link": str(ee_link),
            "joint_names": ",".join(joint_names),
        },
    )


def safe_ratio(numerator: float, denominator: float) -> float:
    if denominator > RANGE_EPS:
        return float(numerator / denominator)
    if numerator <= RANGE_EPS:
        return 1.0
    return float("nan")


def arc_length(points: np.ndarray) -> float:
    if points.shape[0] < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1)))


def derivative_cost(q: np.ndarray, order: int) -> float:
    if q.shape[0] <= order:
        return 0.0
    derivative = np.diff(q, n=order, axis=0)
    return float(np.mean(np.sum(np.square(derivative), axis=1)))


def trajectory_metrics(
    q: np.ndarray,
    ee: np.ndarray,
    desired: np.ndarray,
    expert_q: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> Dict[str, Any]:
    error = np.linalg.norm(ee - desired, axis=1)
    desired_range = np.ptp(desired, axis=0)
    actual_range = np.ptp(ee, axis=0)
    steps = np.linalg.norm(np.diff(q, axis=0), axis=1)
    below = np.maximum(lower.reshape(1, JOINT_DIM) - q, 0.0)
    above = np.maximum(q - upper.reshape(1, JOINT_DIM), 0.0)
    violation = below + above
    return {
        "mean_cartesian_error": float(np.mean(error)),
        "rms_cartesian_error": float(np.sqrt(np.mean(np.square(error)))),
        "maximum_cartesian_error": float(np.max(error)),
        "median_cartesian_error": float(np.median(error)),
        "p95_cartesian_error": float(np.percentile(error, 95.0)),
        "starting_point_error": float(error[0]),
        "ending_point_error": float(error[-1]),
        "x_range_ratio": safe_ratio(float(actual_range[0]), float(desired_range[0])),
        "y_range_ratio": safe_ratio(float(actual_range[1]), float(desired_range[1])),
        "z_range_ratio": safe_ratio(float(actual_range[2]), float(desired_range[2])),
        "cartesian_arc_length_ratio": safe_ratio(
            arc_length(ee), arc_length(desired)
        ),
        "mean_joint_step": float(np.mean(steps)) if steps.size else 0.0,
        "maximum_joint_step": float(np.max(steps)) if steps.size else 0.0,
        "velocity_cost": derivative_cost(q, 1),
        "acceleration_cost": derivative_cost(q, 2),
        "jerk_cost": derivative_cost(q, 3),
        "joint_limit_violation_count": int(np.count_nonzero(violation > 0.0)),
        "joint_limit_violation_magnitude": float(np.sum(violation)),
        "joint_rmse_vs_expert": float(
            np.sqrt(np.mean(np.square(q - expert_q)))
        ),
    }


def evaluate_candidates(
    candidates: Sequence[Candidate],
    test: SplitDataset,
    selected_names: Sequence[str],
    fk: FKComputer,
    lower: np.ndarray,
    upper: np.ndarray,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    expected_shape = (test.trajectory_length, JOINT_DIM)
    for candidate in candidates:
        for name in selected_names:
            base: Dict[str, Any] = {
                "method": candidate.name,
                "method_class": candidate.method_class,
                "path_name": name,
                "trajectory_available": False,
                "evaluation_status": "",
                "source_result_paths": join_names(candidate.source_paths),
                "expert_dependence_status": candidate.expert_dependence_status,
                "practical_at_inference": candidate.practical_at_inference,
            }
            q = candidate.test.get(name)
            if q is None:
                base["evaluation_status"] = (
                    candidate.unavailable_reason or "missing_test_trajectory"
                )
                rows.append(base)
                continue
            if q.shape != expected_shape:
                base["evaluation_status"] = (
                    f"invalid_shape:{tuple(int(value) for value in q.shape)}"
                )
                rows.append(base)
                continue
            if not np.all(np.isfinite(q)):
                base["evaluation_status"] = "non_finite_trajectory"
                rows.append(base)
                continue
            try:
                ee = fk.positions(q)
                metrics = trajectory_metrics(
                    q,
                    ee,
                    test.desired[name],
                    test.expert_q[name],
                    lower,
                    upper,
                )
            except Exception as exc:
                base["evaluation_status"] = f"evaluation_error:{exc}"
                rows.append(base)
                continue
            base["trajectory_available"] = True
            base["evaluation_status"] = "evaluated"
            base.update(metrics)
            rows.append(base)
    return rows


def finite_numeric(values: Iterable[Any]) -> np.ndarray:
    output: List[float] = []
    for value in values:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(numeric):
            output.append(numeric)
    return np.asarray(output, dtype=np.float64)


def aggregate_per_method(
    candidates: Sequence[Candidate],
    per_path_rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in per_path_rows:
        if row.get("evaluation_status") == "evaluated":
            grouped[str(row["method"])].append(row)
    aggregate_rows: List[Dict[str, Any]] = []
    for candidate in candidates:
        rows = grouped.get(candidate.name, [])
        output: Dict[str, Any] = {
            "method": candidate.name,
            "method_class": candidate.method_class,
            "evaluated_path_count": len(rows),
        }
        for metric in METRIC_FIELDS:
            values = finite_numeric(row.get(metric) for row in rows)
            if values.size == 0:
                for statistic in AGGREGATE_STATS:
                    output[f"{metric}_{statistic}"] = ""
                continue
            output[f"{metric}_mean"] = float(np.mean(values))
            output[f"{metric}_median"] = float(np.median(values))
            output[f"{metric}_std"] = float(np.std(values))
            output[f"{metric}_minimum"] = float(np.min(values))
            output[f"{metric}_maximum"] = float(np.max(values))
        aggregate_rows.append(output)
    return aggregate_rows


def lower_is_better_scale(values: Mapping[str, float]) -> Dict[str, float]:
    finite = {key: value for key, value in values.items() if np.isfinite(value)}
    if not finite:
        return {key: 1.0 for key in values}
    low = min(finite.values())
    high = max(finite.values())
    if high - low <= 1.0e-15:
        return {key: 0.0 if key in finite else 1.0 for key in values}
    return {
        key: (value - low) / (high - low) if key in finite else 1.0
        for key, value in values.items()
    }


def candidate_metric_rows(
    per_path_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, List[Mapping[str, Any]]]:
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for row in per_path_rows:
        if row.get("evaluation_status") == "evaluated":
            grouped[str(row["method"])].append(row)
    return grouped


def mean_or_inf(values: Iterable[Any]) -> float:
    array = finite_numeric(values)
    return float(np.mean(array)) if array.size else float("inf")


def all_between(rows: Sequence[Mapping[str, Any]], field: str, low: float, high: float) -> bool:
    values = finite_numeric(row.get(field) for row in rows)
    return bool(values.size == len(rows) and values.size > 0 and np.all((values >= low) & (values <= high)))


def build_recommendations(
    candidates: Sequence[Candidate],
    availability_rows: Sequence[Mapping[str, Any]],
    per_path_rows: Sequence[Mapping[str, Any]],
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    availability = {str(row["method"]): row for row in availability_rows}
    grouped = candidate_metric_rows(per_path_rows)
    practical = [
        candidate
        for candidate in candidates
        if candidate.practical_at_inference == "yes"
        and candidate.expert_dependence_status == "independent"
        and grouped.get(candidate.name)
    ]

    cartesian_raw: Dict[str, float] = {}
    shape_raw: Dict[str, float] = {}
    arc_raw: Dict[str, float] = {}
    smoothness_raw: Dict[str, float] = {}
    safety_raw: Dict[str, float] = {}
    coverage_raw: Dict[str, float] = {}
    for candidate in practical:
        rows = grouped[candidate.name]
        cartesian_raw[candidate.name] = mean_or_inf(
            row.get("mean_cartesian_error") for row in rows
        )
        range_deviations: List[float] = []
        for row in rows:
            for field in ("x_range_ratio", "y_range_ratio", "z_range_ratio"):
                try:
                    ratio = float(row[field])
                except (KeyError, TypeError, ValueError):
                    continue
                if np.isfinite(ratio) and ratio > 0.0:
                    range_deviations.append(abs(math.log(ratio)))
        shape_raw[candidate.name] = (
            float(np.mean(range_deviations)) if range_deviations else float("inf")
        )
        arc_raw[candidate.name] = mean_or_inf(
            abs(float(row["cartesian_arc_length_ratio"]) - 1.0)
            for row in rows
            if row.get("cartesian_arc_length_ratio") not in ("", None)
        )
        smoothness_raw[candidate.name] = mean_or_inf(
            float(row["velocity_cost"])
            + float(row["acceleration_cost"])
            + float(row["jerk_cost"])
            for row in rows
        )
        safety_raw[candidate.name] = mean_or_inf(
            float(row["joint_limit_violation_count"])
            + float(row["joint_limit_violation_magnitude"])
            for row in rows
        )
        avail = availability[candidate.name]
        train_coverage = float(avail["training_coverage_fraction"])
        test_coverage = float(avail["test_coverage_fraction"])
        coverage_raw[candidate.name] = 1.0 - 0.5 * (train_coverage + test_coverage)

    scaled = {
        "cartesian": lower_is_better_scale(cartesian_raw),
        "shape": lower_is_better_scale(shape_raw),
        "arc": lower_is_better_scale(arc_raw),
        "smoothness": lower_is_better_scale(smoothness_raw),
        "safety": lower_is_better_scale(safety_raw),
        "coverage": lower_is_better_scale(coverage_raw),
    }
    weights = {
        "cartesian": 0.30,
        "shape": 0.15,
        "arc": 0.10,
        "smoothness": 0.15,
        "safety": 0.10,
        "coverage": 0.20,
    }

    recommendation_rows: List[Dict[str, Any]] = []
    for candidate in candidates:
        rows = grouped.get(candidate.name, [])
        avail = availability[candidate.name]
        is_practical = (
            candidate.practical_at_inference == "yes"
            and candidate.expert_dependence_status == "independent"
            and bool(rows)
        )
        score = (
            sum(
                weights[factor] * scaled[factor][candidate.name]
                for factor in weights
            )
            if is_practical
            else float("inf")
        )
        mean_error = mean_or_inf(row.get("mean_cartesian_error") for row in rows)
        median_error = (
            float(np.median(finite_numeric(row.get("median_cartesian_error") for row in rows)))
            if rows
            and finite_numeric(row.get("median_cartesian_error") for row in rows).size
            else float("inf")
        )
        no_limits = bool(
            rows
            and all(int(row["joint_limit_violation_count"]) == 0 for row in rows)
        )
        arc_ok = all_between(rows, "cartesian_arc_length_ratio", 0.8, 1.2)
        x_ok = all_between(rows, "x_range_ratio", 0.8, 1.2)
        y_ok = all_between(rows, "y_range_ratio", 0.8, 1.2)
        full_coverage = bool(avail["full_train_and_test_coverage"])
        no_expert_leakage = (
            candidate.expert_dependence_status == "independent"
            and candidate.generation_uses_expert_information
            in {
                "no_at_generation",
                "no_at_generation_or_ranking",
            }
        )
        no_split_trajectory_leakage = (
            int(avail["train_test_identical_trajectory_count"]) == 0
        )
        selection_eligible = bool(
            is_practical
            and full_coverage
            and no_expert_leakage
            and no_split_trajectory_leakage
            and bool(avail["all_trajectories_shape_T_6"])
        )
        checks = {
            "check_mean_cartesian_error_le_0_02_m": mean_error <= 0.02,
            "check_median_cartesian_error_le_0_02_m": median_error <= 0.02,
            "check_no_joint_limit_violations": no_limits,
            "check_arc_length_ratio_0_8_to_1_2": arc_ok,
            "check_x_range_ratio_0_8_to_1_2": x_ok,
            "check_y_range_ratio_0_8_to_1_2": y_ok,
            "check_full_train_and_test_coverage": full_coverage,
            "check_no_expert_leakage": no_expert_leakage,
            "check_no_train_test_trajectory_leakage": no_split_trajectory_leakage,
        }
        recommendation_rows.append(
            {
                "method": candidate.name,
                "is_available_practical_candidate": is_practical,
                "selection_eligible": selection_eligible,
                "composite_score_lower_is_better": (
                    score if np.isfinite(score) else ""
                ),
                "cartesian_accuracy_factor": (
                    scaled["cartesian"].get(candidate.name, "") if is_practical else ""
                ),
                "shape_coverage_factor": (
                    scaled["shape"].get(candidate.name, "") if is_practical else ""
                ),
                "arc_length_factor": (
                    scaled["arc"].get(candidate.name, "") if is_practical else ""
                ),
                "joint_smoothness_factor": (
                    scaled["smoothness"].get(candidate.name, "") if is_practical else ""
                ),
                "joint_limit_safety_factor": (
                    scaled["safety"].get(candidate.name, "") if is_practical else ""
                ),
                "train_test_coverage_factor": (
                    scaled["coverage"].get(candidate.name, "") if is_practical else ""
                ),
                "mean_cartesian_error_m": (
                    mean_error if np.isfinite(mean_error) else ""
                ),
                "median_cartesian_error_m": (
                    median_error if np.isfinite(median_error) else ""
                ),
                "training_coverage_fraction": avail["training_coverage_fraction"],
                "test_coverage_fraction": avail["test_coverage_fraction"],
                "expert_dependence_status": candidate.expert_dependence_status,
                "practical_at_inference": candidate.practical_at_inference,
                **checks,
                "all_acceptance_checks_pass": all(checks.values()),
                "recommended": False,
                "practical_rank": "",
            }
        )

    practical_rows = [
        row
        for row in recommendation_rows
        if row["is_available_practical_candidate"]
    ]
    practical_rows.sort(
        key=lambda row: (
            float(row["composite_score_lower_is_better"]),
            str(row["method"]) != "current_v5b_prior",
            str(row["method"]),
        )
    )
    for rank, row in enumerate(practical_rows, start=1):
        row["practical_rank"] = rank
    eligible = [row for row in practical_rows if row["selection_eligible"]]
    recommended = str(eligible[0]["method"]) if eligible else None
    if recommended is not None:
        for row in recommendation_rows:
            row["recommended"] = row["method"] == recommended
    recommendation_rows.sort(
        key=lambda row: (
            not bool(row["is_available_practical_candidate"]),
            int(row["practical_rank"]) if row["practical_rank"] != "" else sys.maxsize,
            str(row["method"]),
        )
    )
    return recommendation_rows, recommended


def ordered_fields(rows: Sequence[Mapping[str, Any]], preferred: Sequence[str]) -> List[str]:
    fields = list(preferred)
    seen = set(fields)
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key)
                seen.add(key)
    return fields


def write_dict_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    preferred_fields: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ordered_fields(rows, preferred_fields)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def repo_location(path: str, start: int, end: Optional[int] = None) -> str:
    suffix = f"{start}" if end is None else f"{start}-{end}"
    return f"{path}:{suffix}"


def write_lineage_report(
    path: Path,
    candidates: Sequence[Candidate],
    availability_rows: Sequence[Mapping[str, Any]],
    train_meta: Mapping[str, Any],
    test_meta: Mapping[str, Any],
    fk_meta: Mapping[str, str],
    recommended: Optional[str],
) -> None:
    availability = {str(row["method"]): row for row in availability_rows}
    lines = [
        "# Bootstrap prior lineage and comparison audit",
        "",
        "## Current v5b prior: conclusion",
        "",
        "The current v5b prior_q_window artifact is an **MLP-only trajectory export**. "
        "It is not Adaptive MLP+IK, sequential IK, or diffusion v1. The v5b builder "
        "reads mlp_v3_{train,test}_predictions/<path>/predicted_q.csv and slices that "
        "full trajectory into raw-radian prior_q_window entries.",
        "",
        f"- Builder script: {repo_location('build_diffusion_v5b_residual_window_dataset_fk_condition.py', 27, 32)} "
        "defines the diffusion-v2 inputs, MLP prior directories, and v5b output directory.",
        f"- Upstream CSV read and window construction: {repo_location('build_diffusion_v5b_residual_window_dataset_fk_condition.py', 344, 390)}.",
        f"- NPZ key and normalization metadata writes: {repo_location('build_diffusion_v5b_residual_window_dataset_fk_condition.py', 490, 533)}.",
        f"- Train/test exporter intent and transformation: {repo_location('generate_mlp_v3_train_test_predictions.py', 2, 14)}, "
        f"{repo_location('generate_mlp_v3_train_test_predictions.py', 203, 231)}, and "
        f"{repo_location('generate_mlp_v3_train_test_predictions.py', 260, 292)}.",
        f"- Default MLP checkpoint: {repo_location('generate_mlp_v3_test_predictions.py', 26, 30)} "
        f"identifies {DATA_ROOT / 'path_conditioned_mlp_v3.pt'}.",
        f"- The MLP trainer uses actions directly as the six-dimensional target at "
        f"{repo_location('train_path_conditioned_mlp.py', 87, 101)} and standardizes that "
        f"target at {repo_location('train_path_conditioned_mlp.py', 145, 149)}.",
        f"- The canonical full-q MLP predictor denormalizes model output with checkpoint "
        f"y_mean/y_std at {repo_location('predict_path_conditioned_mlp.py', 92, 105)}; "
        "this is evaluated separately as mlp_only.",
        "",
        "### Transformations and normalization",
        "",
        "1. The mlp_v3 train/test exporter treats raw model output as predicted_delta_q_norm.",
        "2. It denormalizes with delta-q mean/std recomputed from diffusion_train_v2.npz.",
        "3. It adds q_start to obtain predicted_q.csv.",
        "4. The v5b builder performs FK on predicted_q and slices it at each stride-aware start.",
        "5. The builder normalizes the condition and expert-minus-prior residual; "
        "prior_q_window itself remains an unnormalized joint trajectory in radians.",
        "6. No IK refinement occurs in this chain.",
        "",
        "### Source and confidence",
        "",
        f"- Upstream train result tree: {DATA_ROOT / 'mlp_v3_train_predictions'}",
        f"- Upstream test result tree: {DATA_ROOT / 'mlp_v3_test_predictions'}",
        "- Source model class: path-conditioned MLP.",
        f"- Source checkpoint identifiable from defaults: {DATA_ROOT / 'path_conditioned_mlp_v3.pt'}",
        "- Artifact-chain confidence: high, because builder and exporter defaults and keys agree.",
        "- Semantic confidence for the export interpretation: medium. The checkpoint's canonical "
        "predictor is a full-q/y-normalized model, while the mlp_v3 exporter reinterprets raw "
        "output with diffusion delta-q statistics. The NPZ contains no embedded invocation or "
        "checkpoint provenance that can prove whether command-line overrides were used.",
        "",
        "## Stride-aware reconstruction audit",
        "",
        f"- Train: H={train_meta['horizon']}, stride={train_meta['stride']}, "
        f"expected starts={train_meta['expected_starts']}; source={train_meta['stride_source']}; "
        f"audit_pass={train_meta['audit_pass']}.",
        f"- Test: H={test_meta['horizon']}, stride={test_meta['stride']}, "
        f"expected starts={test_meta['expected_starts']}; source={test_meta['stride_source']}; "
        f"audit_pass={test_meta['audit_pass']}.",
        "- Expected starts use range(0, trajectory_length - horizon + 1, stride). "
        "A missing start is therefore stride-relative, not a requirement for a window at every timestep.",
        "",
        "## Candidate lineage and availability",
        "",
        "| Method | Class | Train | Test | Expert dependence | Practical | Status | Source/checkpoint |",
        "|---|---|---:|---:|---|---|---|---|",
    ]
    for candidate in candidates:
        row = availability[candidate.name]
        source = join_names(candidate.source_paths)
        if candidate.source_checkpoint:
            source = f"{source}; checkpoint={candidate.source_checkpoint}" if source else candidate.source_checkpoint
        lines.append(
            f"| {candidate.name} | {candidate.method_class} | "
            f"{row['training_paths_available']}/{row['expected_training_paths']} | "
            f"{row['test_paths_available']}/{row['expected_test_paths']} | "
            f"{candidate.expert_dependence_status} | {candidate.practical_at_inference} | "
            f"{'available' if row['available'] else row['unavailable_reason']} | "
            f"{source or 'none'} |"
        )
    lines.extend(
        [
            "",
            "### Candidate-specific evidence",
            "",
            f"- Adaptive MLP+IK reads path_conditioned_pred_q.csv and writes "
            f"refined_mlp_ik_q.csv: {repo_location('refine_mlp_predictions_with_ik.py', 1, 35)} "
            f"and {repo_location('refine_mlp_predictions_with_ik.py', 124, 230)}. "
            "The adaptive wrapper can overwrite the same filename, so last-writer provenance is unresolved.",
            f"- Sequential IK uses the previous solution as the next initialization: "
            f"{repo_location('generate_ik_seed_path.py', 282, 379)}. Stored expert_q artifacts "
            "are treated as oracle/expert-dependent for selection.",
            f"- Diffusion-v1 best-of-K scoring uses desired Cartesian error and robot-aware "
            f"smoothness, and selects by total_cost: "
            f"{repo_location('sample_ranked_diffusion_candidates_v1.py', 272, 327)} and "
            f"{repo_location('sample_ranked_diffusion_candidates_v1.py', 425, 480)}. "
            "expert_q is exported for evaluation but is not used in best-of-K ranking. "
            f"However, the v1 checkpoint was selected on the benchmark test split at "
            f"{repo_location('train_conditional_diffusion_trajectory.py', 294, 306)} and "
            f"{repo_location('train_conditional_diffusion_trajectory.py', 335, 360)}, so "
            "the stored v1 candidates are not unbiased benchmark contenders.",
            "- Every available k*/reranked_best_*.csv manifest is loaded as a distinct "
            "diffusion-v1 result because each row points to a genuine diffusion_pred_q.csv.",
            f"- Diffusion-v2 DDPM/DDIM and diffusion-v3 x0 result manifests are loaded "
            "directly. Diffusion-v4 samples_single is mapped by test-row index, while its "
            "debugged tree and duplicate refinement families are inventory-only.",
            f"- Diffusion-v2 and v3 likewise use the benchmark test NPZ for validation and "
            f"best-checkpoint selection: {repo_location('train_conditional_diffusion_trajectory_v2.py', 214, 248)} "
            f"and {repo_location('train_conditional_diffusion_trajectory_v3_x0.py', 220, 276)}. "
            "Their sampling is runtime-practical, but their stored benchmark results are "
            "evaluation-test-label-dependent and recommendation-ineligible.",
            f"- Diffusion-v4 maps the configured test dataset to validation at "
            f"{repo_location('train_conditional_diffusion_trajectory_v4_unet.py', 124, 132)} "
            f"and {repo_location('train_conditional_diffusion_trajectory_v4_unet.py', 226, 230)}, "
            f"then selects best_model.pt on validation target loss at "
            f"{repo_location('train_conditional_diffusion_trajectory_v4_unet.py', 291, 296)}. "
            "All v4-model outputs are therefore evaluation-test-label-dependent and "
            "recommendation-ineligible on this benchmark, even when runtime generation "
            "does not consume expert trajectories.",
            f"- Residual v5c loads the benchmark test windows at "
            f"{repo_location('train_residual_window_predictor_v5c.py', 448, 463)} and "
            f"{repo_location('train_residual_window_predictor_v5c.py', 490, 497)}, then "
            f"selects its best checkpoint using candidate RMSE to test expert q at "
            f"{repo_location('train_residual_window_predictor_v5c.py', 551, 579)}. Its "
            "alpha outputs remain evaluable diagnostics but are not unbiased candidates.",
            f"- Full-q scaled-tapered and global-anchored rollout trees are inventoried and "
            f"evaluated with their observed test coverage. Their recorded v5b best checkpoint "
            f"loads benchmark test windows and is selected on test target loss at "
            f"{repo_location('train_conditional_diffusion_trajectory_v5_residual_unet.py', 668, 693)} "
            f"and {repo_location('train_conditional_diffusion_trajectory_v5_residual_unet.py', 762, 815)}; "
            "the stored rollout results are therefore evaluation-test-label-dependent and "
            "recommendation-ineligible. Warm-start concatenated range folders are explicitly "
            "unavailable because they are not one trajectory per evaluation path.",
            "- Unified result CSVs are aggregate metrics only and are not trajectory loaders.",
            "",
            "## FK convention",
            "",
            f"- URDF: {fk_meta['urdf_path']}",
            f"- End-effector link: {fk_meta['ee_link']}",
            f"- Active joints: {fk_meta['joint_names']}",
            "- Call convention: robot.update_cfg(cfg), then robot.get_transform(frame_to=ee_link).",
            f"- Device: {fk_meta['device_note']}.",
            "",
            "## Recommendation",
            "",
            (
                f"Recommended deployable practical prior: **{recommended}**."
                if recommended is not None
                else "No candidate met deployable eligibility (independent inference plus full train/test coverage)."
            ),
            "The ranking is multi-factor: Cartesian accuracy 30%, range retention 15%, "
            "arc-length retention 10%, smoothness 15%, joint-limit safety 10%, and "
            "train/test coverage 20%. Expert-dependent, evaluation-test-label-dependent, "
            "and unknown-lineage candidates cannot be selected.",
            "Acceptance checks use the project diagnostics exactly: mean and median error "
            "at most 0.02 m, no limit violations, arc ratio in [0.8, 1.2], x/y range "
            "ratios in [0.8, 1.2] on every evaluated path, full train/test coverage, and "
            "no expert or identical-trajectory leakage.",
            "",
            "## Unresolved ambiguities",
            "",
            "- Window NPZ files do not embed the exact builder command, git revision, prior directory, "
            "or checkpoint hash. Defaults establish lineage but cannot rule out CLI overrides.",
            "- The mlp_v3 exporter applies delta-q normalization to a checkpoint whose canonical "
            "predictor applies checkpoint y-normalization to full q. The audit compares both artifacts "
            "rather than silently treating them as equivalent.",
            "- Adaptive and fixed MLP+IK runs share refined_mlp_ik_q.csv, so trajectory last-writer "
            "identity is not encoded in the CSV.",
            "- A candidate supplied under an unknown --candidate_prior name is evaluated but is "
            "not recommendation-eligible until its inference/leakage lineage is known.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def aggregate_value(
    aggregate_by_method: Mapping[str, Mapping[str, Any]],
    method: str,
    field: str,
) -> float:
    try:
        value = float(aggregate_by_method[method][field])
    except (KeyError, TypeError, ValueError):
        return float("nan")
    return value


def make_plots(
    output_dir: Path,
    candidates: Sequence[Candidate],
    aggregate_rows: Sequence[Mapping[str, Any]],
    per_path_rows: Sequence[Mapping[str, Any]],
    availability_rows: Sequence[Mapping[str, Any]],
) -> List[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    aggregate = {str(row["method"]): row for row in aggregate_rows}
    availability = {str(row["method"]): row for row in availability_rows}
    evaluated_methods = [
        candidate.name
        for candidate in candidates
        if int(aggregate[candidate.name]["evaluated_path_count"]) > 0
    ]
    colors = plt.cm.tab20(np.linspace(0.0, 1.0, max(len(evaluated_methods), 1)))
    outputs: List[Path] = []

    def save_bar(metric: str, ylabel: str, filename: str) -> None:
        values = [aggregate_value(aggregate, method, f"{metric}_mean") for method in evaluated_methods]
        errors = [aggregate_value(aggregate, method, f"{metric}_std") for method in evaluated_methods]
        fig, axis = plt.subplots(figsize=(max(8.0, len(evaluated_methods) * 0.8), 5.5))
        x = np.arange(len(evaluated_methods))
        axis.bar(x, values, yerr=errors, color=colors[: len(evaluated_methods)], capsize=3)
        axis.set_xticks(x, evaluated_methods, rotation=35, ha="right")
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        path = plots_dir / filename
        fig.savefig(path, dpi=180)
        plt.close(fig)
        outputs.append(path)

    save_bar(
        "mean_cartesian_error",
        "Mean Cartesian error (m)",
        "mean_cartesian_error_by_prior.png",
    )
    save_bar(
        "rms_cartesian_error",
        "RMS Cartesian error (m)",
        "rms_cartesian_error_by_prior.png",
    )

    fig, axis = plt.subplots(figsize=(7.5, 5.5))
    for color, method in zip(colors, evaluated_methods):
        x_value = aggregate_value(aggregate, method, "mean_cartesian_error_mean")
        y_value = sum(
            aggregate_value(aggregate, method, f"{metric}_mean")
            for metric in ("velocity_cost", "acceleration_cost", "jerk_cost")
        )
        axis.scatter(x_value, y_value, color=color, s=55, label=method)
    axis.set_xlabel("Mean Cartesian error (m)")
    axis.set_ylabel("Velocity + acceleration + jerk cost")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8, loc="best")
    fig.tight_layout()
    path = plots_dir / "drawing_smoothness_cost_vs_cartesian_error.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    outputs.append(path)

    fig, axis = plt.subplots(figsize=(max(8.0, len(evaluated_methods) * 0.9), 5.5))
    x = np.arange(len(evaluated_methods))
    width = 0.24
    for offset, dimension in zip((-width, 0.0, width), XYZ_COLUMNS):
        values = [
            aggregate_value(aggregate, method, f"{dimension}_range_ratio_mean")
            for method in evaluated_methods
        ]
        axis.bar(x + offset, values, width=width, label=dimension)
    axis.axhspan(0.8, 1.2, color="green", alpha=0.08)
    axis.axhline(1.0, color="black", linewidth=1)
    axis.set_xticks(x, evaluated_methods, rotation=35, ha="right")
    axis.set_ylabel("Executed / desired Cartesian range")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = plots_dir / "path_shape_range_retention_by_prior.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    outputs.append(path)

    save_bar(
        "cartesian_arc_length_ratio",
        "Executed / desired Cartesian arc length",
        "arc_length_ratio_by_prior.png",
    )

    valid_rows = candidate_metric_rows(per_path_rows)
    current = {
        str(row["path_name"]): float(row["mean_cartesian_error"])
        for row in valid_rows.get("current_v5b_prior", [])
    }
    fig, axis = plt.subplots(figsize=(6.5, 6.0))
    all_values: List[float] = []
    for color, method in zip(colors, evaluated_methods):
        if method == "current_v5b_prior":
            continue
        paired_x: List[float] = []
        paired_y: List[float] = []
        for row in valid_rows.get(method, []):
            name = str(row["path_name"])
            if name in current:
                paired_x.append(current[name])
                paired_y.append(float(row["mean_cartesian_error"]))
        if paired_x:
            axis.scatter(paired_x, paired_y, alpha=0.7, s=24, color=color, label=method)
            all_values.extend(paired_x)
            all_values.extend(paired_y)
    if all_values:
        low, high = min(all_values), max(all_values)
        axis.plot([low, high], [low, high], "k--", linewidth=1)
    axis.set_xlabel("Current v5b mean Cartesian error (m)")
    axis.set_ylabel("Candidate mean Cartesian error (m)")
    axis.grid(alpha=0.25)
    if len(evaluated_methods) > 1:
        axis.legend(fontsize=7)
    fig.tight_layout()
    path = plots_dir / "paired_scatter_vs_current_v5b_prior.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    outputs.append(path)

    methods = [candidate.name for candidate in candidates]
    train_coverage = [
        float(availability[method]["training_coverage_fraction"]) for method in methods
    ]
    test_coverage = [
        float(availability[method]["test_coverage_fraction"]) for method in methods
    ]
    fig, axis = plt.subplots(figsize=(max(9.0, len(methods) * 0.8), 5.5))
    x = np.arange(len(methods))
    width = 0.38
    axis.bar(x - width / 2.0, train_coverage, width, label="train")
    axis.bar(x + width / 2.0, test_coverage, width, label="test")
    axis.set_ylim(0.0, 1.05)
    axis.set_xticks(x, methods, rotation=35, ha="right")
    axis.set_ylabel("Path coverage fraction")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    path = plots_dir / "train_test_coverage.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    outputs.append(path)
    return outputs


def print_console_summary(
    candidates: Sequence[Candidate],
    availability_rows: Sequence[Mapping[str, Any]],
    aggregate_rows: Sequence[Mapping[str, Any]],
    recommended: Optional[str],
    output_paths: Sequence[Path],
    train_meta: Mapping[str, Any],
    test_meta: Mapping[str, Any],
    fk_meta: Mapping[str, str],
) -> None:
    availability = {str(row["method"]): row for row in availability_rows}
    aggregates = {str(row["method"]): row for row in aggregate_rows}
    print("\nCurrent v5b prior lineage")
    print("  builder: build_diffusion_v5b_residual_window_dataset_fk_condition.py")
    print("  source: mlp_v3_{train,test}_predictions/<path>/predicted_q.csv")
    print(f"  checkpoint default: {DATA_ROOT / 'path_conditioned_mlp_v3.pt'}")
    print("  method: MLP-only delta-export artifact; no IK and no diffusion sampling")
    print(
        f"  train reconstruction: H={train_meta['horizon']}, stride={train_meta['stride']}, "
        f"pass={train_meta['audit_pass']}"
    )
    print(
        f"  test reconstruction: H={test_meta['horizon']}, stride={test_meta['stride']}, "
        f"pass={test_meta['audit_pass']}"
    )
    print(f"  FK: {fk_meta['device_note']}; ee_link={fk_meta['ee_link']}")

    print("\nDiscovered candidate priors")
    for candidate in candidates:
        available = availability[candidate.name]
        aggregate = aggregates[candidate.name]
        mean_error = aggregate.get("mean_cartesian_error_mean", "")
        error_text = (
            f"{float(mean_error):.6g} m" if mean_error not in ("", None) else "not evaluated"
        )
        print(
            f"  {candidate.name}: train={available['training_paths_available']}/"
            f"{available['expected_training_paths']}, test={available['test_paths_available']}/"
            f"{available['expected_test_paths']}, expert={candidate.expert_dependence_status}, "
            f"practical={candidate.practical_at_inference}, mean_cartesian={error_text}"
        )
    unavailable = [
        candidate.name for candidate in candidates if not candidate.available
    ]
    print("\nUnavailable candidates")
    if unavailable:
        for name in unavailable:
            print(f"  {name}: {availability[name]['unavailable_reason']}")
    else:
        print("  none")

    print("\nRecommended practical prior")
    print(f"  {recommended or 'none (no full-coverage independent candidate)'}")
    print("\nOutput files")
    for path in output_paths:
        print(f"  {path}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train = load_split_dataset(args.train_npz, "train")
    test = load_split_dataset(args.test_npz, "test")
    selected_test_names = list(test.names)
    if args.max_paths > 0:
        selected_test_names = selected_test_names[: args.max_paths]

    train_reconstruction, train_audit_rows, train_meta = reconstruct_current_prior(
        split=train,
        window_path=args.window_train_npz,
        output_dir=args.output_dir,
        overlap_atol=args.overlap_atol,
        overlap_rtol=args.overlap_rtol,
    )
    test_reconstruction, test_audit_rows, test_meta = reconstruct_current_prior(
        split=test,
        window_path=args.window_test_npz,
        output_dir=args.output_dir,
        overlap_atol=args.overlap_atol,
        overlap_rtol=args.overlap_rtol,
    )
    reconstruction_rows = train_audit_rows + test_audit_rows

    current = make_current_candidate(
        train_reconstruction,
        test_reconstruction,
        train_audit_rows,
        test_audit_rows,
        args.window_train_npz,
        args.window_test_npz,
    )
    candidates = auto_discover_candidates(train, test, current)
    candidates = apply_explicit_candidates(
        candidates, args.candidate_prior, train, test
    )
    candidates.append(make_expert_ceiling(train, test))

    availability_rows = [
        candidate_availability_row(candidate, train, test)
        for candidate in candidates
    ]
    fk, lower, upper, fk_meta = load_fk_context(args)
    per_path_rows = evaluate_candidates(
        candidates,
        test,
        selected_test_names,
        fk,
        lower,
        upper,
    )
    aggregate_rows = aggregate_per_method(candidates, per_path_rows)
    recommendation_rows, recommended = build_recommendations(
        candidates, availability_rows, per_path_rows
    )

    per_path_path = args.output_dir / "prior_comparison_per_path.csv"
    aggregate_path = args.output_dir / "prior_comparison_aggregate.csv"
    availability_path = args.output_dir / "prior_availability.csv"
    lineage_path = args.output_dir / "prior_lineage_report.md"
    reconstruction_path = args.output_dir / "current_prior_reconstruction_audit.csv"
    recommendation_path = args.output_dir / "recommended_prior.csv"

    write_dict_csv(
        per_path_path,
        per_path_rows,
        (
            "method",
            "method_class",
            "path_name",
            "trajectory_available",
            "evaluation_status",
        )
        + METRIC_FIELDS,
    )
    write_dict_csv(
        aggregate_path,
        aggregate_rows,
        ("method", "method_class", "evaluated_path_count"),
    )
    write_dict_csv(
        availability_path,
        availability_rows,
        (
            "method",
            "method_class",
            "available",
            "training_paths_available",
            "test_paths_available",
            "full_train_and_test_coverage",
            "expert_dependence_status",
            "practical_at_inference",
        ),
    )
    write_dict_csv(
        reconstruction_path,
        reconstruction_rows,
        (
            "split",
            "path_name",
            "trajectory_length",
            "horizon",
            "stride",
            "expected_window_count",
            "observed_window_count",
            "audit_pass",
        ),
    )
    write_dict_csv(
        recommendation_path,
        recommendation_rows,
        (
            "practical_rank",
            "method",
            "recommended",
            "selection_eligible",
            "composite_score_lower_is_better",
            "all_acceptance_checks_pass",
        ),
    )
    write_lineage_report(
        lineage_path,
        candidates,
        availability_rows,
        train_meta,
        test_meta,
        fk_meta,
        recommended,
    )
    plot_paths = make_plots(
        args.output_dir,
        candidates,
        aggregate_rows,
        per_path_rows,
        availability_rows,
    )
    output_paths = [
        per_path_path,
        aggregate_path,
        availability_path,
        lineage_path,
        reconstruction_path,
        recommendation_path,
        *plot_paths,
    ]
    print_console_summary(
        candidates,
        availability_rows,
        aggregate_rows,
        recommended,
        output_paths,
        train_meta,
        test_meta,
        fk_meta,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
