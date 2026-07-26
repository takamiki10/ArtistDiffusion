#!/usr/bin/env python3
"""Frozen v8.1 path-disjoint confirmation on adaptive test_prior trajectories."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import multiprocessing
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import matplotlib
import numpy as np

matplotlib.use("Agg")

import evaluate_diffusion_v7_teacher_forced_validation as v7_evaluator
import evaluate_diffusion_v8_1_anchored_recursive_jerk_guard as v81
import evaluate_diffusion_v8_anchored_recursive_rollout as v8
from generate_ik_seed_path import DEFAULT_URDF_PATH


EXPECTED_TRAJECTORY_LENGTH = 100
EXPECTED_MANIFEST_PATHS = 30
SOURCE_SPLIT = "test"
SOURCE_ARCHIVE_DEFAULT = Path(
    "data/cartesian_expert_dataset_v3/adaptive_mlp_ik_bootstrap_prior/test_prior.npz"
)
MANIFEST_DEFAULT = Path(
    "results/diffusion_v8_1_path_disjoint_confirmation_test30_seeds53_57/"
    "metadata/path_disjoint_test_paths_30.csv"
)


@dataclass(frozen=True)
class TestPriorRecord:
    evaluator_path_id: str
    source_path_name: str
    source_method: str
    source_checkpoint: str
    manifest_selection_rank: int
    source_sorted_success_index: int
    physical_path: v8.PhysicalPathRecord


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--training_dataset_dir",
        type=Path,
        default=Path(
            "data/cartesian_expert_dataset_v3/"
            "diffusion_v8_multitarget_scaled_training_dataset_100paths"
        ),
    )
    parser.add_argument(
        "--model_dir",
        type=Path,
        default=Path(
            "models/"
            "diffusion_v8_multitarget_scaled_residual_unet_100paths_"
            "epsilon_only_seed42"
        ),
    )
    parser.add_argument("--test_prior_npz", type=Path, default=SOURCE_ARCHIVE_DEFAULT)
    parser.add_argument("--path_manifest", type=Path, default=MANIFEST_DEFAULT)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/diffusion_v8_1_path_disjoint_confirmation"),
    )
    parser.add_argument("--checkpoint_state", default="raw_last_epoch187")
    parser.add_argument("--target_scale", type=float, default=1.0)
    parser.add_argument("--output_alpha", type=float, default=0.125)
    parser.add_argument("--k_values", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--sampling_seed", type=int, default=53)
    parser.add_argument("--ddim_steps", type=int, default=50)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--execution_horizon", type=int, default=8)
    parser.add_argument("--anchoring_horizon", type=int, default=8)
    parser.add_argument("--num_cpu_workers", type=int, default=8)
    parser.add_argument("--gpu_batch_size", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--source_path_names", nargs="*", default=None)
    parser.add_argument("--max_paths", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def as_text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def json_safe_strict(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): json_safe_strict(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe_strict(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe_strict(value.tolist())
    if isinstance(value, np.generic):
        return json_safe_strict(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def read_manifest(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "split",
        "path_name",
        "successful_prior",
        "selection_rank",
        "source_sorted_success_index",
    }
    missing = sorted(required - set(rows[0] if rows else ()))
    if missing:
        raise ValueError(f"Manifest is missing required columns: {missing}")
    return rows


def manifest_success(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def validate_manifest(rows: Sequence[Mapping[str, str]]) -> None:
    if len(rows) != EXPECTED_MANIFEST_PATHS:
        raise ValueError(f"Manifest has {len(rows)} rows; expected 30")
    path_names = [str(row["path_name"]) for row in rows]
    ranks = [int(row["selection_rank"]) for row in rows]
    source_indices = [int(row["source_sorted_success_index"]) for row in rows]
    if len(set(path_names)) != EXPECTED_MANIFEST_PATHS:
        raise ValueError("Manifest path_name values are not unique")
    if ranks != list(range(EXPECTED_MANIFEST_PATHS)):
        raise ValueError(
            "Manifest selection_rank values must appear in ascending order 0..29"
        )
    if len(set(source_indices)) != EXPECTED_MANIFEST_PATHS:
        raise ValueError("Manifest source_sorted_success_index values are not unique")
    bad_split = [row for row in rows if str(row["split"]) != SOURCE_SPLIT]
    if bad_split:
        raise ValueError("Manifest split must be exactly test for every row")
    bad_success = [row for row in rows if not manifest_success(row["successful_prior"])]
    if bad_success:
        raise ValueError("Manifest contains unsuccessful_prior rows")


def selected_manifest_rows(
    rows: Sequence[Mapping[str, str]],
    source_path_names: Optional[Sequence[str]],
    max_paths: Optional[int],
) -> Tuple[List[Mapping[str, str]], bool]:
    selected = list(rows)
    diagnostic = False
    if source_path_names:
        requested = list(dict.fromkeys(source_path_names))
        if len(requested) != len(source_path_names):
            raise ValueError("--source_path_names contains duplicates")
        requested_set = set(requested)
        available = {str(row["path_name"]) for row in rows}
        missing = sorted(requested_set - available)
        if missing:
            raise ValueError(f"Requested source paths are not in manifest: {missing}")
        selected = [row for row in rows if str(row["path_name"]) in requested_set]
        diagnostic = True
    if max_paths is not None:
        if max_paths < 1:
            raise ValueError("--max_paths must be positive")
        selected = selected[:max_paths]
        diagnostic = True
    return selected, diagnostic


def archive_index(npz: Mapping[str, Any]) -> Dict[str, int]:
    names = [as_text(name) for name in npz["path_names"]]
    if len(names) != len(set(names)):
        raise ValueError("test_prior.npz path_names are not unique")
    return {name: index for index, name in enumerate(names)}


def sorted_success_index_by_name(npz: Mapping[str, Any]) -> Dict[str, int]:
    names = [as_text(name) for name in npz["path_names"]]
    success = np.asarray(npz["generation_success"], dtype=bool)
    if success.shape != (len(names),):
        raise ValueError("generation_success shape does not match path_names")
    successful_names = sorted(name for name, ok in zip(names, success) if bool(ok))
    return {name: index for index, name in enumerate(successful_names)}


def validate_status(generation_success: Any, generation_status: Any, path_name: str) -> None:
    if not bool(generation_success):
        raise ValueError(f"{path_name}: generation_success is false")
    status = as_text(generation_status).lower()
    if "success" not in status:
        raise ValueError(f"{path_name}: generation_status does not indicate success: {status}")


def validate_shapes_and_numbers(
    source_path_name: str,
    desired_path: np.ndarray,
    prior_q: np.ndarray,
    prior_ee: np.ndarray,
) -> None:
    expected = {
        "desired_path": (EXPECTED_TRAJECTORY_LENGTH, 3),
        "prior_q": (EXPECTED_TRAJECTORY_LENGTH, 6),
        "prior_ee": (EXPECTED_TRAJECTORY_LENGTH, 3),
    }
    arrays = {
        "desired_path": desired_path,
        "prior_q": prior_q,
        "prior_ee": prior_ee,
    }
    for label, array in arrays.items():
        if array.shape != expected[label]:
            raise ValueError(f"{source_path_name}: {label} has shape {array.shape}")
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{source_path_name}: {label} contains nonfinite values")


def validate_fk_and_fallback(
    record: v8.PhysicalPathRecord,
    stored_prior_ee: np.ndarray,
    robot: Any,
    args: argparse.Namespace,
) -> None:
    recomputed = v8.compute_fk_positions(robot, record.strong_prior_q)
    if not np.allclose(recomputed, stored_prior_ee, rtol=1.0e-5, atol=2.0e-5):
        raise ValueError(f"{record.path_id}: stored prior_ee does not match FK")
    start = 0
    while start < len(record.strong_prior_q) - 1:
        current_q = record.strong_prior_q[start]
        previous_q = record.strong_prior_q[start - 1] if start > 0 else None
        execution_count = min(args.execution_horizon, len(record.strong_prior_q) - 1 - start)
        anchored = v8.build_anchored_prior_window(
            record.strong_prior_q,
            start,
            current_q,
            args.horizon,
            args.anchoring_horizon,
        )
        desired = v8.padded_window(record.desired_path, start, args.horizon)
        context = v8.make_action_context(
            record,
            start,
            current_q,
            previous_q,
            anchored,
            desired,
            robot,
            execution_count,
        )
        metrics = v7_evaluator.evaluate_metrics(
            robot, context, context.prior_q, execution_count
        )
        reasons = v8.recursive_executed_prefix_hard_safety_reasons(metrics)
        if reasons:
            raise ValueError(
                f"{record.path_id}@{start}: prior fallback hard-safety failed: {reasons}"
            )
        start += execution_count


def load_test_prior_records(
    args: argparse.Namespace,
    robot: Any,
) -> Tuple[List[TestPriorRecord], bool]:
    manifest_rows = read_manifest(args.path_manifest)
    validate_manifest(manifest_rows)
    selected_rows, diagnostic_subset = selected_manifest_rows(
        manifest_rows, args.source_path_names, args.max_paths
    )
    selected_names = [str(row["path_name"]) for row in selected_rows]
    with np.load(args.test_prior_npz) as data:
        required = {
            "path_names",
            "desired_paths",
            "prior_q",
            "prior_ee",
            "generation_success",
            "generation_status",
            "source_method",
            "source_checkpoint",
        }
        missing = sorted(required - set(data.files))
        if missing:
            raise ValueError(f"test_prior.npz is missing keys: {missing}")
        by_name = archive_index(data)
        sorted_success_indices = sorted_success_index_by_name(data)
        manifest_names = [str(row["path_name"]) for row in manifest_rows]
        missing_manifest_names = sorted(set(manifest_names) - set(by_name))
        if missing_manifest_names:
            raise ValueError(
                f"Manifest paths missing from test_prior.npz: {missing_manifest_names}"
            )
        for manifest_row in manifest_rows:
            source_name = str(manifest_row["path_name"])
            index = by_name[source_name]
            archive_success = bool(data["generation_success"][index])
            if manifest_success(manifest_row["successful_prior"]) != archive_success:
                raise ValueError(
                    f"{source_name}: manifest successful_prior disagrees with archive"
                )
            actual_success_index = sorted_success_indices.get(source_name)
            manifest_success_index = int(manifest_row["source_sorted_success_index"])
            if actual_success_index != manifest_success_index:
                raise ValueError(
                    f"{source_name}: manifest source_sorted_success_index="
                    f"{manifest_success_index}, archive-derived value="
                    f"{actual_success_index}"
                )
        missing_names = sorted(set(selected_names) - set(by_name))
        if missing_names:
            raise ValueError(f"Manifest paths missing from test_prior.npz: {missing_names}")
        if len(selected_names) != len(set(selected_names)):
            raise ValueError("Selected manifest paths are not unique")
        records: List[TestPriorRecord] = []
        for manifest_row in selected_rows:
            source_name = str(manifest_row["path_name"])
            index = by_name[source_name]
            archive_success = bool(data["generation_success"][index])
            validate_status(
                archive_success,
                data["generation_status"][index],
                source_name,
            )
            desired_path = np.asarray(data["desired_paths"][index], dtype=np.float64)
            prior_q = np.asarray(data["prior_q"][index], dtype=np.float64)
            prior_ee = np.asarray(data["prior_ee"][index], dtype=np.float64)
            validate_shapes_and_numbers(source_name, desired_path, prior_q, prior_ee)
            evaluator_path_id = f"test__{source_name}"
            physical = v8.PhysicalPathRecord(
                path_id=evaluator_path_id,
                path_index=int(index),
                population="ordinary",
                desired_path=desired_path,
                strong_prior_q=prior_q,
                prior_ee=prior_ee,
            )
            validate_fk_and_fallback(physical, prior_ee, robot, args)
            records.append(
                TestPriorRecord(
                    evaluator_path_id=evaluator_path_id,
                    source_path_name=source_name,
                    source_method=as_text(data["source_method"][index]),
                    source_checkpoint=as_text(data["source_checkpoint"][index]),
                    manifest_selection_rank=int(manifest_row["selection_rank"]),
                    source_sorted_success_index=int(
                        manifest_row["source_sorted_success_index"]
                    ),
                    physical_path=physical,
                )
            )
    if not diagnostic_subset and len(records) != EXPECTED_MANIFEST_PATHS:
        raise ValueError("Full run must contain exactly 30 records")
    if not diagnostic_subset and set(selected_names) != {record.source_path_name for record in records}:
        raise ValueError("Manifest path set does not match selected archive records")
    return records, diagnostic_subset


def metadata(record: TestPriorRecord, args: argparse.Namespace, diagnostic: bool) -> Dict[str, Any]:
    return {
        "sampling_seed": int(args.sampling_seed),
        "source_split": SOURCE_SPLIT,
        "source_path_name": record.source_path_name,
        "source_archive": str(args.test_prior_npz),
        "source_method": record.source_method,
        "source_checkpoint": record.source_checkpoint,
        "manifest_selection_rank": record.manifest_selection_rank,
        "source_sorted_success_index": record.source_sorted_success_index,
        "evaluator_path_id": record.evaluator_path_id,
        "path_disjoint_confirmation_population": int(not diagnostic),
        "diagnostic_subset_run": int(diagnostic),
    }


def annotate_rows(
    rows: Sequence[MutableMapping[str, Any]],
    record_by_evaluator_id: Mapping[str, TestPriorRecord],
    args: argparse.Namespace,
    diagnostic: bool,
) -> None:
    for row in rows:
        path_id = str(row.get("path_id", ""))
        record = record_by_evaluator_id[path_id]
        row.update(metadata(record, args, diagnostic))


def save_trajectory_npz(
    output_dir: Path,
    result: v8.RolloutResult,
    record: TestPriorRecord,
    args: argparse.Namespace,
) -> None:
    path_dir = output_dir / "trajectories" / result.path.path_id
    path_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path_dir / f"anchored_rollout_k{result.k}.npz",
        strong_prior_q=result.path.strong_prior_q,
        rollout_q=result.rollout_q,
        desired_path=result.path.desired_path,
        prior_ee=result.path.prior_ee,
        rollout_ee=result.rollout_ee,
        executed_source=result.executed_source,
        accepted_step_mask=result.accepted_step_mask,
        fallback_step_mask=result.fallback_step_mask,
        selected_candidate_indices=result.selected_candidate_indices,
        applied_correction_norms=result.applied_correction_norms,
        window_start_indices=result.window_start_indices,
        executed_indices=result.executed_indices,
        sampling_seed=int(args.sampling_seed),
        source_split=SOURCE_SPLIT,
        source_path_name=record.source_path_name,
        source_archive=str(args.test_prior_npz),
        source_method=record.source_method,
        source_checkpoint=record.source_checkpoint,
        manifest_selection_rank=record.manifest_selection_rank,
        evaluator_path_id=record.evaluator_path_id,
    )


def validate_args(args: argparse.Namespace) -> None:
    args.dataset_dir = args.training_dataset_dir
    args.include_difficult_paths = False
    args.smoke_test = False
    args.disable_history_aware_jerk_guard = False
    v8.validate_args(args)
    if (
        args.test_prior_npz.resolve()
        == args.training_dataset_dir.resolve()
        or args.training_dataset_dir.name == args.test_prior_npz.name
    ):
        raise ValueError("test_prior.npz must not be used as the training dataset")
    if tuple(args.k_values) != v8.FROZEN_K_VALUES:
        raise ValueError("K values must be 1 4 8 so K=8 uses eight nested candidates")
    if (
        v81.run_rollout_v8_1.__module__
        != "evaluate_diffusion_v8_1_anchored_recursive_jerk_guard"
    ):
        raise AssertionError("run_rollout_v8_1 is not imported from frozen v8.1 module")
    if args.disable_history_aware_jerk_guard:
        raise AssertionError("Path-disjoint confirmation requires the jerk guard enabled")


def main() -> int:
    args = parse_args()
    validate_args(args)
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.overwrite:
        raise FileExistsError(
            f"Output directory is nonempty: {args.output_dir}; pass --overwrite"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    robot = v7_evaluator.make_robot_context(Path(DEFAULT_URDF_PATH))
    records, diagnostic_subset = load_test_prior_records(args, robot)
    inference = v81.load_validated_inference_bundle(
        args.training_dataset_dir,
        args.model_dir,
        args.checkpoint_state,
        args.device,
        args.ddim_steps,
    )
    if v81.HISTORY_AWARE_JERK_TOLERANCE != 1.0e-12:
        raise AssertionError("Frozen v8.1 jerk tolerance changed")
    if list(v8.FROZEN_K_VALUES) != [1, 4, 8]:
        raise AssertionError("Frozen K values changed")
    executor: Optional[concurrent.futures.ProcessPoolExecutor] = None
    if args.num_cpu_workers > 1:
        executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=args.num_cpu_workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=v7_evaluator.initialize_candidate_worker,
            initargs=(str(Path(DEFAULT_URDF_PATH)),),
        )
    path_rows: List[Dict[str, Any]] = []
    decision_rows: List[Dict[str, Any]] = []
    candidate_rows: List[Dict[str, Any]] = []
    record_by_id = {record.evaluator_path_id: record for record in records}
    try:
        for record_index, record in enumerate(records, start=1):
            sample_cache: Dict[Tuple[bytes, Tuple[int, ...]], np.ndarray] = {}
            for k in v8.FROZEN_K_VALUES:
                result = v81.run_rollout_v8_1(
                    record.physical_path,
                    k,
                    inference,
                    robot,
                    executor,
                    args,
                    sample_cache,
                )
                save_trajectory_npz(args.output_dir, result, record, args)
                v8.save_path_plots(args.output_dir, result)
                v8.save_manipulability_plot(args.output_dir, result, robot)
                result.metrics.update(metadata(record, args, diagnostic_subset))
                path_rows.append(result.metrics)
                annotate_rows(result.decision_rows, record_by_id, args, diagnostic_subset)
                annotate_rows(result.candidate_rows, record_by_id, args, diagnostic_subset)
                decision_rows.extend(result.decision_rows)
                candidate_rows.extend(result.candidate_rows)
            print(
                f"Completed {record_index}/{len(records)} source paths: "
                f"{record.source_path_name} as {record.evaluator_path_id}"
            )
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=False)

    ordinary = v81.aggregate_rows_v8_1(path_rows, "ordinary")
    combined = v81.aggregate_rows_v8_1(path_rows, "combined_diagnostic")
    v8.write_csv(args.output_dir / "anchored_rollout_decisions.csv", decision_rows)
    v8.write_csv(args.output_dir / "anchored_candidate_results.csv", candidate_rows)
    v8.write_csv(args.output_dir / "anchored_full_path_metrics.csv", path_rows)
    v8.write_csv(args.output_dir / "anchored_ordinary_aggregate.csv", ordinary)
    v8.write_csv(
        args.output_dir / "anchored_combined_diagnostic_aggregate.csv", combined
    )
    v8.save_aggregate_plots(args.output_dir, path_rows)
    summary = {
        "status": "complete",
        "experiment": "v8.1_path_disjoint_confirmation",
        "sampling_seed": args.sampling_seed,
        "checkpoint_state": inference.checkpoint_state,
        "checkpoint_epoch": inference.checkpoint_epoch,
        "checkpoint_state_hash": inference.checkpoint_state_hash,
        "training_dataset_dir": str(args.training_dataset_dir),
        "model_dir": str(args.model_dir),
        "test_prior_npz": str(args.test_prior_npz),
        "path_manifest": str(args.path_manifest),
        "manifest_row_count": EXPECTED_MANIFEST_PATHS,
        "manifest_source_path_names": [record.source_path_name for record in records],
        "manifest_selection_ranks": [
            record.manifest_selection_rank for record in records
        ],
        "history_aware_jerk_tolerance": v81.HISTORY_AWARE_JERK_TOLERANCE,
        "frozen_k_values": list(v8.FROZEN_K_VALUES),
        "diagnostic_subset_run": int(diagnostic_subset),
        "path_disjoint_confirmation_population": int(not diagnostic_subset),
        "source_path_names": [record.source_path_name for record in records],
        "evaluator_path_ids": [record.evaluator_path_id for record in records],
        "ordinary": ordinary,
        "combined_diagnostic": combined,
        "notes": [
            "Path-disjoint evaluation population loaded from test_prior.npz.",
            "The physical paths are distinct from the train-side v8 validation population.",
            "Model weights and normalization remain frozen from the original v8 training artifacts.",
            "Collision-safe evaluator path IDs are used for candidate seed identity.",
        ],
    }
    (args.output_dir / "anchored_rollout_summary.json").write_text(
        json.dumps(
            json_safe_strict(summary),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    report = [
        "Diffusion v8.1 path-disjoint confirmation on adaptive test prior",
        "",
        "Path-disjoint evaluation population loaded from test_prior.npz.",
        "The physical paths are distinct from the train-side v8 validation population.",
        "Model weights and normalization remain frozen from the original v8 training artifacts.",
        f"sampling_seed: {args.sampling_seed}",
        f"source paths: {len(records)}",
        f"diagnostic subset: {diagnostic_subset}",
        f"training dataset dir: {args.training_dataset_dir}",
        f"model dir: {args.model_dir}",
        f"checkpoint: {inference.checkpoint_state}",
    ]
    (args.output_dir / "anchored_rollout_report.txt").write_text(
        "\n".join(report) + "\n"
    )
    print("classification: V8_1_PATH_DISJOINT_EVALUATION_COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
