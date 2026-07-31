"""Tests for the saved-artifact multi-path smoothness benchmark."""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import importlib
import io
import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

import numpy as np

benchmark = importlib.import_module("benchmark_ik_mlp_pipeline_smoothness")
comparison = importlib.import_module("compare_ik_mlp_pipeline_jerk_over_time")

SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parent


def digest(path: Path) -> str:
    """Return the SHA-256 digest of one test input."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_joint_csv(path: Path, timestamps: np.ndarray, q: np.ndarray) -> None:
    """Write one repository-style six-joint CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("t", "q1", "q2", "q3", "q4", "q5", "q6"))
        for index, timestamp in enumerate(timestamps):
            writer.writerow((float(timestamp), *q[index].tolist()))


def write_target_csv(path: Path, target: np.ndarray) -> None:
    """Write one repository-style desired Cartesian path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("x", "y", "z"))
        writer.writerows(target.tolist())


def polynomial_q(
    timestamps: np.ndarray,
    coefficients: tuple[float, float, float, float],
) -> np.ndarray:
    """Create six scaled copies of a cubic polynomial."""
    a0, a1, a2, a3 = coefficients
    base = a0 + a1 * timestamps + a2 * timestamps**2 + a3 * timestamps**3
    return np.column_stack([(index + 1.0) * base for index in range(6)])


def metric_bundle(
    q: np.ndarray,
    timestamps: np.ndarray,
    *,
    endpoint_exclusion: int = 3,
    execution_horizon: int = 8,
    boundary_radius: int = 2,
    spectral_signal: str = "acceleration",
    high_frequency_fraction: float = 0.25,
) -> dict[str, Any]:
    """Return only the metric dictionary from the benchmark calculation."""
    return benchmark.smoothness_metrics(
        q,
        timestamps,
        endpoint_exclusion=endpoint_exclusion,
        execution_horizon=execution_horizon,
        boundary_radius=boundary_radius,
        spectral_signal=spectral_signal,
        high_frequency_fraction=high_frequency_fraction,
    )[0]


def fake_fk(_context: Any, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Treat the first three synthetic joints as Cartesian position."""
    rotations = np.repeat(np.eye(3)[None, :, :], len(q), axis=0)
    return np.asarray(q[:, :3], dtype=np.float64), rotations


class SyntheticRepository:
    """Small immutable saved-artifact fixture used by end-to-end tests."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.ik = root / "ik"
        self.mlp = root / "mlp"
        self.pipeline = root / "pipeline"
        self.output = root / "output"
        self.timestamps = np.linspace(0.0, 2.0, 25)
        self.files: list[Path] = []

    def add_path(
        self,
        path_number: int,
        *,
        seeds: tuple[tuple[int, bool, float], ...] = (
            (11, True, 0.05),
            (23, True, 0.08),
        ),
        mlp_tracking_offset: float = 0.0,
    ) -> None:
        """Add deterministic artifacts and selected pipeline rollouts."""
        path_id = f"path_{path_number:04d}"
        progress = np.linspace(0.0, 1.0, len(self.timestamps))
        target = np.column_stack(
            (
                0.25 + 0.02 * progress,
                -0.10 + 0.01 * progress,
                0.35 + 0.015 * progress,
            )
        )
        base = np.column_stack(
            (
                target,
                0.1 * progress,
                -0.05 * progress**2,
                0.02 * progress**3,
            )
        )
        ik_q = np.array(base, copy=True)
        mlp_q = np.array(base, copy=True)
        mlp_q[:, :3] += mlp_tracking_offset
        mlp_q[:, 3:] += 0.015 * np.sin(2.0 * np.pi * progress)[:, None]
        ik_file = self.ik / path_id / "expert_q.csv"
        mlp_file = self.mlp / path_id / "predicted_q.csv"
        write_joint_csv(ik_file, self.timestamps, ik_q)
        write_joint_csv(mlp_file, self.timestamps, mlp_q)
        write_target_csv(self.ik / path_id / "desired_path.csv", target)
        write_target_csv(self.mlp / path_id / "desired_path.csv", target)
        self.files.extend(
            (
                ik_file,
                mlp_file,
                self.ik / path_id / "desired_path.csv",
                self.mlp / path_id / "desired_path.csv",
            )
        )
        for seed, accepted, amplitude in seeds:
            seed_dir = self.pipeline / f"seed_{seed}"
            trajectory_dir = seed_dir / "trajectories" / f"test__{path_id}"
            trajectory_dir.mkdir(parents=True, exist_ok=True)
            q = np.array(base, copy=True)
            q[:, 3:] += amplitude * np.sin(4.0 * np.pi * progress)[:, None]
            artifact = trajectory_dir / "anchored_rollout_k8.npz"
            np.savez(
                artifact,
                rollout_q=q,
                timestamps=self.timestamps,
                desired_path=target,
            )
            self.files.append(artifact)
            metrics = seed_dir / "anchored_full_path_metrics.csv"
            existing: list[dict[str, str]] = []
            if metrics.is_file():
                with metrics.open("r", encoding="utf-8", newline="") as handle:
                    existing = list(csv.DictReader(handle))
            existing.append(
                {
                    "path_id": path_id,
                    "k": "8",
                    "full_path_safety_pass": "1" if accepted else "0",
                    "failed_hard_safety_gate": "" if accepted else "rejected",
                    "runtime_s": str(seed / 100.0),
                }
            )
            with metrics.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(existing[0]))
                writer.writeheader()
                writer.writerows(existing)
            if metrics not in self.files:
                self.files.append(metrics)

    def args(
        self,
        *,
        path_ids: tuple[str, ...] = ("path_0001",),
        strict: bool = False,
        minimum_pipeline_seeds: int = 1,
    ) -> argparse.Namespace:
        """Create validated command-line arguments for this fixture."""
        argv = [
            "--ik_root",
            str(self.ik),
            "--mlp_root",
            str(self.mlp),
            "--pipeline_root",
            str(self.pipeline),
            "--output_dir",
            str(self.output),
            "--path_ids",
            *path_ids,
            "--common_duration_s",
            "4",
            "--common_samples",
            "32",
            "--minimum_pipeline_seeds",
            str(minimum_pipeline_seeds),
        ]
        if strict:
            argv.append("--strict")
        return benchmark.parse_args(argv)


class CommandAndDiscoveryTests(unittest.TestCase):
    """Cover CLI defaults, path discovery, and seed discovery."""

    def test_01_cli_defaults(self) -> None:
        args = benchmark.parse_args(
            [
                "--ik_root",
                "ik",
                "--mlp_root",
                "mlp",
                "--pipeline_root",
                "pipeline",
                "--output_dir",
                "out",
            ]
        )
        self.assertEqual(args.timing_policy, "common_duration")
        self.assertEqual((args.common_duration_s, args.common_samples), (10.0, 100))
        self.assertEqual((args.execution_horizon, args.boundary_radius), (8, 2))
        self.assertEqual(args.boundary_sensitivity_radii, [0, 1, 2])
        self.assertTrue(args.require_accepted_pipeline)

    def test_02_multiple_matched_paths_discovered(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SyntheticRepository(Path(temporary))
            for path_id in ("path_0001", "path_0002", "path_0003"):
                (fixture.ik / path_id).mkdir(parents=True)
            for path_id in ("path_0001", "path_0002", "path_0004"):
                (fixture.mlp / path_id).mkdir(parents=True)
            args = fixture.args(path_ids=())
            self.assertEqual(
                benchmark.discover_requested_paths(args),
                ["path_0001", "path_0002"],
            )

    def test_03_path_list_file_supports_csv(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            listing = root / "paths.csv"
            listing.write_text("id\npath_9\npath_0012\n", encoding="utf-8")
            self.assertEqual(
                benchmark.read_path_list(listing), ["path_0009", "path_0012"]
            )

    def test_04_multiple_pipeline_seeds_discovered(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SyntheticRepository(Path(temporary))
            fixture.add_path(1)
            found = benchmark.discover_pipeline_artifacts(
                fixture.pipeline, "*seed*", None
            )
            self.assertEqual([item.seed for item in found["path_0001"]], [11, 23])

    def test_05_pipeline_seed_filter(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SyntheticRepository(Path(temporary))
            fixture.add_path(1)
            found = benchmark.discover_pipeline_artifacts(
                fixture.pipeline, "*seed*", [23]
            )
            self.assertEqual([item.seed for item in found["path_0001"]], [23])

    def test_06_rejected_seed_verdict_is_retained(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SyntheticRepository(Path(temporary))
            fixture.add_path(1, seeds=((11, True, 0.1), (23, False, 0.2)))
            found = benchmark.discover_pipeline_artifacts(
                fixture.pipeline, "*seed*", None
            )
            self.assertEqual([item.accepted for item in found["path_0001"]], [True, False])


class MetricTests(unittest.TestCase):
    """Exercise analytic derivatives, integration, continuity, and boundaries."""

    def setUp(self) -> None:
        self.timestamps = np.linspace(0.0, 2.0, 201)

    def test_07_quadratic_rms_acceleration(self) -> None:
        q = polynomial_q(self.timestamps, (0.0, 0.0, 0.5, 0.0))
        metrics = metric_bundle(q, self.timestamps)
        expected = math.sqrt(np.mean(np.square(np.arange(1.0, 7.0))))
        self.assertAlmostEqual(metrics["rms_acceleration_rad_s2"], expected, places=7)

    def test_08_cubic_rms_jerk(self) -> None:
        q = polynomial_q(self.timestamps, (0.0, 0.0, 0.0, 1.0 / 6.0))
        metrics = metric_bundle(q, self.timestamps)
        _, _, jerk = comparison.compute_derivatives(q, self.timestamps)
        expected = math.sqrt(float(np.mean(np.square(jerk))))
        analytic = math.sqrt(np.mean(np.square(np.arange(1.0, 7.0))))
        self.assertAlmostEqual(metrics["rms_jerk_rad_s3"], expected, places=12)
        self.assertAlmostEqual(
            metrics["interior_rms_jerk_rad_s3"], analytic, places=6
        )

    def test_09_integrated_squared_jerk_uses_timestamps(self) -> None:
        q = polynomial_q(self.timestamps, (0.0, 0.0, 0.0, 1.0 / 6.0))
        metrics = metric_bundle(q, self.timestamps)
        _, _, jerk = comparison.compute_derivatives(q, self.timestamps)
        expected = sum(
            comparison.integrate(np.square(jerk[:, joint]), self.timestamps)
            for joint in range(6)
        )
        self.assertAlmostEqual(
            metrics["integrated_squared_jerk_rad2_s5"], expected, places=10
        )

    def test_10_endpoint_exclusion_only_changes_interior_metrics(self) -> None:
        q = polynomial_q(self.timestamps, (0.0, 0.0, 0.0, 0.1))
        q[0] += 1.0
        full_a = metric_bundle(q, self.timestamps, endpoint_exclusion=1)
        full_b = metric_bundle(q, self.timestamps, endpoint_exclusion=5)
        self.assertEqual(full_a["rms_jerk_rad_s3"], full_b["rms_jerk_rad_s3"])
        self.assertNotEqual(
            full_a["interior_rms_jerk_rad_s3"],
            full_b["interior_rms_jerk_rad_s3"],
        )

    def test_11_endpoint_jerk_energy_fraction(self) -> None:
        q = np.zeros((40, 6))
        q[0, 0] = 1.0
        timestamps = np.linspace(0.0, 4.0, 40)
        metrics = metric_bundle(q, timestamps, endpoint_exclusion=4)
        self.assertGreater(metrics["endpoint_jerk_energy_fraction"], 0.9)

    def test_12_rollout_boundary_indices(self) -> None:
        np.testing.assert_array_equal(
            benchmark.boundary_indices(34, 8), np.asarray([8, 16, 24, 32])
        )

    def test_13_boundary_radius_mask(self) -> None:
        mask = benchmark.boundary_mask(20, np.asarray([8, 16]), 2)
        self.assertEqual(np.flatnonzero(mask).tolist(), list(range(6, 11)) + list(range(14, 19)))

    def test_14_boundary_energy_fraction(self) -> None:
        timestamps = np.linspace(0.0, 4.0, 41)
        q = np.zeros((41, 6))
        q[8, 0] = 1.0
        metrics = metric_bundle(q, timestamps, execution_horizon=8, boundary_radius=2)
        self.assertGreater(metrics["boundary_jerk_energy_fraction"], 0.5)

    def test_15_control_methods_use_same_boundary_indices(self) -> None:
        q = np.zeros((40, 6))
        first = metric_bundle(q, np.linspace(0.0, 4.0, 40))
        second = metric_bundle(q, np.linspace(0.0, 4.0, 40))
        self.assertEqual(first["rollout_boundary_indices"], second["rollout_boundary_indices"])

    def test_16_maximum_joint_step(self) -> None:
        q = np.zeros((20, 6))
        q[10:, :2] = (3.0, 4.0)
        metrics = metric_bundle(q, np.linspace(0.0, 2.0, 20))
        self.assertAlmostEqual(metrics["max_joint_step_l2_rad"], 5.0)
        self.assertEqual(json.loads(metrics["max_joint_step_per_joint_rad"])[:2], [3.0, 4.0])

    def test_17_acceleration_total_variation(self) -> None:
        timestamps = np.linspace(0.0, 1.0, 31)
        q = polynomial_q(timestamps, (0.0, 0.0, 0.0, 1.0 / 6.0))
        _, acceleration, _ = comparison.compute_derivatives(q, timestamps)
        expected = float(np.sum(np.abs(np.diff(acceleration, axis=0))))
        self.assertAlmostEqual(
            metric_bundle(q, timestamps)["acceleration_total_variation_rad_s2"],
            expected,
        )

    def test_18_per_joint_metric_families_are_present(self) -> None:
        metrics = metric_bundle(
            polynomial_q(self.timestamps, (0.0, 0.0, 0.1, 0.01)),
            self.timestamps,
        )
        for key in (
            "velocity_integrated_squared_per_joint_rad2_s",
            "acceleration_mean_abs_per_joint_rad_s2",
            "acceleration_integrated_squared_per_joint_rad2_s3",
            "jerk_median_abs_per_joint_rad_s3",
            "jerk_integrated_squared_per_joint_rad2_s5",
        ):
            self.assertEqual(len(json.loads(metrics[key])), 6)


class BoundaryNormalizationTests(unittest.TestCase):
    """Verify coverage-normalized boundary energy independently of derivatives."""

    def setUp(self) -> None:
        self.timestamps = np.linspace(0.0, 10.0, 100)
        self.weights = benchmark.integration_weights(self.timestamps)
        self.boundaries = benchmark.boundary_indices(100, 8)

    def test_18a_sample_fraction_for_default_grid(self) -> None:
        mask = benchmark.boundary_mask(100, self.boundaries, 2)
        metrics = benchmark.boundary_energy_metrics(
            self.weights, self.weights, mask
        )
        self.assertEqual(int(np.sum(mask)), 60)
        self.assertAlmostEqual(metrics["boundary_sample_fraction"], 0.6)

    def test_18b_time_fraction_uses_trapezoidal_weights(self) -> None:
        weights = np.asarray([0.5, 1.0, 1.5, 2.0])
        mask = np.asarray([True, True, False, False])
        metrics = benchmark.boundary_energy_metrics(weights, weights, mask)
        self.assertAlmostEqual(metrics["boundary_sample_fraction"], 0.5)
        self.assertAlmostEqual(metrics["boundary_time_fraction"], 0.3)

    def test_18c_uniform_energy_density_has_unit_enrichment(self) -> None:
        mask = benchmark.boundary_mask(100, self.boundaries, 2)
        metrics = benchmark.boundary_energy_metrics(
            3.0 * self.weights, self.weights, mask
        )
        self.assertAlmostEqual(
            metrics["boundary_jerk_energy_enrichment"], 1.0, places=12
        )
        self.assertAlmostEqual(
            metrics["boundary_to_nonboundary_energy_density_ratio"],
            1.0,
            places=12,
        )

    def test_18d_boundary_only_energy_is_enriched(self) -> None:
        mask = benchmark.boundary_mask(100, self.boundaries, 2)
        energy = np.zeros(100)
        energy[mask] = self.weights[mask]
        metrics = benchmark.boundary_energy_metrics(energy, self.weights, mask)
        self.assertGreater(metrics["boundary_jerk_energy_enrichment"], 1.0)

    def test_18e_nonboundary_only_energy_is_depleted(self) -> None:
        mask = benchmark.boundary_mask(100, self.boundaries, 2)
        energy = np.zeros(100)
        energy[~mask] = self.weights[~mask]
        metrics = benchmark.boundary_energy_metrics(energy, self.weights, mask)
        self.assertLess(metrics["boundary_jerk_energy_enrichment"], 1.0)

    def test_18f_energy_densities_and_ratio(self) -> None:
        metrics = benchmark.boundary_energy_metrics(
            np.asarray([2.0, 2.0, 1.0, 1.0]),
            np.ones(4),
            np.asarray([True, True, False, False]),
        )
        self.assertEqual(
            metrics["boundary_jerk_energy_density_rad2_s6"], 2.0
        )
        self.assertEqual(
            metrics["nonboundary_jerk_energy_density_rad2_s6"], 1.0
        )
        self.assertEqual(
            metrics["boundary_to_nonboundary_energy_density_ratio"], 2.0
        )

    def test_18g_zero_boundary_energy_has_zero_enrichment(self) -> None:
        metrics = benchmark.boundary_energy_metrics(
            np.asarray([0.0, 0.0, 1.0, 1.0]),
            np.ones(4),
            np.asarray([True, True, False, False]),
        )
        self.assertEqual(metrics["boundary_jerk_energy_enrichment"], 0.0)

    def test_18h_zero_nonboundary_energy_has_nan_density_ratio(self) -> None:
        metrics = benchmark.boundary_energy_metrics(
            np.asarray([1.0, 1.0, 0.0, 0.0]),
            np.ones(4),
            np.asarray([True, True, False, False]),
        )
        ratio = metrics["boundary_to_nonboundary_energy_density_ratio"]
        self.assertTrue(math.isnan(ratio))
        self.assertFalse(math.isinf(ratio))

    def test_18i_radius_zero_contains_exact_boundaries_only(self) -> None:
        mask = benchmark.boundary_mask(25, np.asarray([8, 16, 24]), 0)
        self.assertEqual(np.flatnonzero(mask).tolist(), [8, 16, 24])

    def test_18j_radius_one_and_two_masks(self) -> None:
        radius_one = benchmark.boundary_mask(20, np.asarray([8]), 1)
        radius_two = benchmark.boundary_mask(20, np.asarray([8]), 2)
        self.assertEqual(np.flatnonzero(radius_one).tolist(), [7, 8, 9])
        self.assertEqual(np.flatnonzero(radius_two).tolist(), [6, 7, 8, 9, 10])

    def test_18k_sensitivity_metrics_for_all_default_radii(self) -> None:
        output = benchmark.boundary_sensitivity_metrics(
            self.weights,
            self.weights,
            100,
            self.boundaries,
            [0, 1, 2],
        )
        for radius in (0, 1, 2):
            for suffix in (
                "sample_fraction",
                "time_fraction",
                "jerk_energy_fraction",
                "jerk_energy_enrichment",
                "energy_density_ratio",
            ):
                self.assertIn(f"boundary_r{radius}_{suffix}", output)

    def test_18l_configured_radius_added_and_duplicates_removed(self) -> None:
        args = benchmark.parse_args(
            [
                "--ik_root",
                "ik",
                "--mlp_root",
                "mlp",
                "--pipeline_root",
                "pipeline",
                "--output_dir",
                "out",
                "--boundary_radius",
                "3",
                "--boundary_sensitivity_radii",
                "2",
                "1",
                "2",
            ]
        )
        benchmark.validate_args(args)
        self.assertEqual(args.boundary_sensitivity_radii, [1, 2, 3])

    def test_18m_negative_sensitivity_radius_rejected(self) -> None:
        args = benchmark.parse_args(
            [
                "--ik_root",
                "ik",
                "--mlp_root",
                "mlp",
                "--pipeline_root",
                "pipeline",
                "--output_dir",
                "out",
                "--boundary_sensitivity_radii",
                "0",
                "-1",
            ]
        )
        with self.assertRaisesRegex(benchmark.BenchmarkError, "nonnegative"):
            benchmark.validate_args(args)


class SpectralTests(unittest.TestCase):
    """Verify low/high/zero FFT behavior."""

    def test_19_low_frequency_signal_has_negligible_high_frequency_energy(self) -> None:
        timestamps = np.linspace(0.0, 10.0, 256, endpoint=False)
        signal = np.tile(np.sin(2.0 * np.pi * 0.2 * timestamps)[:, None], (1, 6))
        metrics = benchmark.spectral_metrics(signal, timestamps, 0.25)
        self.assertLess(metrics["mean_high_frequency_energy_ratio"], 1.0e-20)

    def test_20_high_frequency_signal_has_high_ratio(self) -> None:
        timestamps = np.linspace(0.0, 10.0, 256, endpoint=False)
        sampling_frequency = 1.0 / np.median(np.diff(timestamps))
        frequency = 0.42 * sampling_frequency
        signal = np.tile(np.sin(2.0 * np.pi * frequency * timestamps)[:, None], (1, 6))
        metrics = benchmark.spectral_metrics(signal, timestamps, 0.25)
        self.assertGreater(metrics["mean_high_frequency_energy_ratio"], 0.9)

    def test_21_zero_energy_returns_nan(self) -> None:
        metrics = benchmark.spectral_metrics(
            np.zeros((64, 6)), np.linspace(0.0, 1.0, 64), 0.25
        )
        self.assertTrue(math.isnan(metrics["mean_high_frequency_energy_ratio"]))
        self.assertFalse(math.isinf(metrics["mean_high_frequency_energy_ratio"]))

    def test_22_frequency_audit_fields_are_recorded(self) -> None:
        metrics = benchmark.spectral_metrics(
            np.random.default_rng(1).normal(size=(64, 6)),
            np.linspace(0.0, 2.0, 64),
            0.25,
        )
        self.assertGreater(metrics["spectral_sampling_frequency_hz"], 0.0)
        self.assertEqual(metrics["spectral_window"], "none")


class TrackingAndAggregationTests(unittest.TestCase):
    """Exercise eligibility, seed aggregation, and path-level statistics."""

    def test_23_cartesian_fk_error(self) -> None:
        q = np.zeros((10, 6))
        target = np.zeros((10, 3))
        target[:, 0] = 0.003
        rotations = np.repeat(np.eye(3)[None, :, :], len(q), axis=0)
        metrics, eligible, _ = benchmark.tracking_metrics(
            q,
            target,
            object(),
            mean_cartesian_threshold_m=0.01,
            max_cartesian_threshold_m=0.03,
            orientation_threshold_rad=None,
            fk_override=(q[:, :3], rotations),
        )
        self.assertAlmostEqual(metrics["mean_cartesian_error_m"], 0.003)
        self.assertTrue(eligible)

    def test_24_tracking_ineligible_is_reported(self) -> None:
        q = np.zeros((10, 6))
        target = np.full((10, 3), 0.02)
        rotations = np.repeat(np.eye(3)[None, :, :], len(q), axis=0)
        metrics, eligible, reason = benchmark.tracking_metrics(
            q,
            target,
            object(),
            mean_cartesian_threshold_m=0.01,
            max_cartesian_threshold_m=0.03,
            orientation_threshold_rad=None,
            fk_override=(q[:, :3], rotations),
        )
        self.assertFalse(eligible)
        self.assertGreater(metrics["max_cartesian_error_m"], 0.01)
        self.assertIn("cartesian", reason)

    def test_24a_separate_mean_and_maximum_thresholds(self) -> None:
        q = np.zeros((10, 6))
        rotations = np.repeat(np.eye(3)[None, :, :], len(q), axis=0)

        mean_target = np.zeros((10, 3))
        mean_target[:, 0] = 0.02
        _, eligible, reason = benchmark.tracking_metrics(
            q,
            mean_target,
            object(),
            mean_cartesian_threshold_m=0.01,
            max_cartesian_threshold_m=0.03,
            orientation_threshold_rad=None,
            fk_override=(q[:, :3], rotations),
        )
        self.assertFalse(eligible)
        self.assertIn("mean_cartesian_error_threshold", reason)
        self.assertNotIn("max_cartesian_error_threshold", reason)

        max_target = np.zeros((10, 3))
        max_target[0, 0] = 0.02
        _, eligible, reason = benchmark.tracking_metrics(
            q,
            max_target,
            object(),
            mean_cartesian_threshold_m=0.01,
            max_cartesian_threshold_m=0.015,
            orientation_threshold_rad=None,
            fk_override=(q[:, :3], rotations),
        )
        self.assertFalse(eligible)
        self.assertNotIn("mean_cartesian_error_threshold", reason)
        self.assertIn("max_cartesian_error_threshold", reason)

    def test_24aa_deprecated_threshold_alias_sets_both(self) -> None:
        args = benchmark.parse_args(
            [
                "--ik_root",
                "ik",
                "--mlp_root",
                "mlp",
                "--pipeline_root",
                "pipeline",
                "--output_dir",
                "out",
                "--cartesian_error_threshold_m",
                "0.02",
            ]
        )
        benchmark.validate_args(args)
        mean_source, max_source = benchmark.resolve_cartesian_thresholds(args)
        self.assertEqual(args.mean_cartesian_error_threshold_m, 0.02)
        self.assertEqual(args.max_cartesian_error_threshold_m, 0.02)
        self.assertIn("deprecated_alias", mean_source)
        self.assertIn("deprecated_alias", max_source)

    def test_24ab_deprecated_alias_conflicts_with_new_thresholds(self) -> None:
        args = benchmark.parse_args(
            [
                "--ik_root",
                "ik",
                "--mlp_root",
                "mlp",
                "--pipeline_root",
                "pipeline",
                "--output_dir",
                "out",
                "--cartesian_error_threshold_m",
                "0.02",
                "--mean_cartesian_error_threshold_m",
                "0.01",
            ]
        )
        with self.assertRaisesRegex(benchmark.BenchmarkError, "cannot be combined"):
            benchmark.validate_args(args)

    def test_24ac_authoritative_maximum_threshold(self) -> None:
        value, source = benchmark.authoritative_max_cartesian_threshold()
        self.assertEqual(value, 0.03)
        self.assertEqual(
            source, "rerank_diffusion_candidates.ACCEPTANCE_MAX_ERROR"
        )

    def test_24ad_missing_authoritative_maximum_threshold_fails(self) -> None:
        with mock.patch.object(
            benchmark.importlib, "import_module", side_effect=ImportError("missing")
        ):
            with self.assertRaisesRegex(
                benchmark.BenchmarkError, "maximum Cartesian-error gate"
            ):
                benchmark.authoritative_max_cartesian_threshold()

    def test_24b_repository_fk_known_trajectory(self) -> None:
        path = (
            SCRIPT_DIR
            / "data"
            / "cartesian_expert_dataset_v3"
            / "cold_ik_test_timed"
            / "test"
            / "path_0001"
        )
        if not path.is_dir():
            self.skipTest("known repository IK artifact is unavailable")
        trajectory = comparison.load_trajectory(path, "ik")
        target = benchmark.target_from_trajectory(trajectory)
        if len(target) != len(trajectory.q):
            target = benchmark.resample_target(target, len(trajectory.q))
        context = benchmark.make_robot_context()
        positions, rotations = benchmark.authoritative_fk(context, trajectory.q)
        metrics, _, _ = benchmark.tracking_metrics(
            trajectory.q,
            target,
            context,
            mean_cartesian_threshold_m=1.0,
            max_cartesian_threshold_m=1.0,
            orientation_threshold_rad=None,
            fk_override=(positions, rotations),
        )
        expected = np.linalg.norm(positions - target, axis=1)
        self.assertAlmostEqual(
            metrics["mean_cartesian_error_m"], float(np.mean(expected)), places=12
        )
        self.assertAlmostEqual(
            metrics["max_cartesian_error_m"], float(np.max(expected)), places=12
        )

    def test_25_rejected_seed_excluded_from_aggregation(self) -> None:
        rows = [
            {
                "seed": 11,
                "accepted": True,
                "smoothness_claim_eligible": True,
                "rms_jerk_rad_s3": 2.0,
            },
            {
                "seed": 23,
                "accepted": False,
                "smoothness_claim_eligible": False,
                "rms_jerk_rad_s3": 0.01,
            },
        ]
        aggregate, _, _ = benchmark.aggregate_pipeline_path("path_0001", rows)
        assert aggregate is not None
        self.assertEqual(aggregate["rms_jerk_rad_s3"], 2.0)

    def test_26_pipeline_median_uses_only_eligible_seeds(self) -> None:
        rows = [
            {"seed": 1, "accepted": True, "smoothness_claim_eligible": True, "rms_jerk_rad_s3": 1.0},
            {"seed": 2, "accepted": True, "smoothness_claim_eligible": True, "rms_jerk_rad_s3": 5.0},
            {"seed": 3, "accepted": True, "smoothness_claim_eligible": False, "rms_jerk_rad_s3": 100.0},
        ]
        aggregate, _, _ = benchmark.aggregate_pipeline_path("path_0001", rows)
        assert aggregate is not None
        self.assertEqual(aggregate["rms_jerk_rad_s3"], 3.0)

    def test_27_best_seed_is_secondary(self) -> None:
        rows = [
            {"seed": 11, "accepted": True, "smoothness_claim_eligible": True, "rms_jerk_rad_s3": 3.0},
            {"seed": 23, "accepted": True, "smoothness_claim_eligible": True, "rms_jerk_rad_s3": 1.0},
        ]
        aggregate, selection, best = benchmark.aggregate_pipeline_path("path_0001", rows)
        assert aggregate is not None and best is not None
        self.assertEqual(aggregate["aggregation"], "median_across_eligible_accepted_seeds")
        self.assertEqual(best["seed"], 23)
        self.assertEqual(best["aggregation"], "secondary_best_seed_by_rms_jerk")
        self.assertEqual(selection[0]["best_seed"], 23)

    def test_28_win_rates_use_matched_paths(self) -> None:
        rows = []
        for path_id, ik, pipeline in (("path_0001", 2.0, 1.0), ("path_0002", 1.0, 2.0)):
            for method, value in (("ik", ik), ("pipeline", pipeline), ("mlp", ik)):
                row = {"path_id": path_id, "method": method, "smoothness_claim_eligible": True}
                row.update({metric: value for metric in benchmark.PRIMARY_METRICS})
                rows.append(row)
        match = next(
            row
            for row in benchmark.win_rates(rows)
            if row["reference_method"] == "ik"
            and row["comparison_method"] == "pipeline"
            and row["metric"] == "rms_jerk_rad_s3"
        )
        self.assertEqual(match["matched_path_count"], 2)
        self.assertEqual(match["comparison_win_rate"], 0.5)

    def test_29_matched_values_have_one_value_per_path(self) -> None:
        rows = [
            {"path_id": "path_0001", "method": method, "smoothness_claim_eligible": True, "x": value}
            for method, value in (("ik", 1.0), ("mlp", 2.0), ("pipeline", 3.0))
        ]
        paths, arrays = benchmark.matched_values(rows, benchmark.METHODS, "x")
        self.assertEqual(paths, ["path_0001"])
        self.assertTrue(all(array.shape == (1,) for array in arrays))

    def test_30_bootstrap_is_deterministic(self) -> None:
        left = np.asarray([1.0, 2.0, 3.0, 4.0])
        right = np.asarray([0.5, 1.5, 2.5, 3.5])
        self.assertEqual(
            benchmark.paired_bootstrap(left, right, seed=42, iterations=200),
            benchmark.paired_bootstrap(left, right, seed=42, iterations=200),
        )

    def test_31_holm_bonferroni(self) -> None:
        adjusted = benchmark.holm_bonferroni([0.01, 0.04, 0.03])
        np.testing.assert_allclose(adjusted, [0.03, 0.06, 0.06])

    def test_32_statistics_report_insufficient_samples(self) -> None:
        rows = [
            {
                "path_id": "path_0001",
                "method": method,
                "smoothness_claim_eligible": True,
                **{metric: float(index + 1) for metric in benchmark.PRIMARY_METRICS},
            }
            for index, method in enumerate(benchmark.METHODS)
        ]
        results = benchmark.statistical_tests(rows)
        self.assertTrue(any("insufficient" in row["notes"] for row in results))
        self.assertTrue(all(row["sample_count"] <= 1 for row in results))

    def test_32a_method_summary_includes_normalized_boundary_metrics(self) -> None:
        rows = [
            {
                "path_id": "path_0001",
                "method": method,
                "smoothness_claim_eligible": True,
                **{metric: 1.0 for metric in benchmark.SUMMARY_METRICS},
                "boundary_r0_jerk_energy_enrichment": 1.0,
            }
            for method in benchmark.METHODS
        ]
        summaries = benchmark.method_summaries(rows)
        pipeline_metrics = {
            row["metric"] for row in summaries if row["method"] == "pipeline"
        }
        self.assertIn("boundary_time_fraction", pipeline_metrics)
        self.assertIn("boundary_jerk_energy_enrichment", pipeline_metrics)
        self.assertIn("boundary_r0_jerk_energy_enrichment", pipeline_metrics)

    def test_32b_thesis_summary_includes_boundary_coverage(self) -> None:
        required = {
            "rms_acceleration_rad_s2": 1.0,
            "rms_jerk_rad_s3": 1.0,
            "integrated_squared_jerk_rad2_s5": 1.0,
            "max_abs_jerk_rad_s3": 1.0,
            "interior_rms_jerk_rad_s3": 1.0,
            "boundary_time_fraction": 0.6,
            "boundary_jerk_energy_fraction": 0.6,
            "boundary_jerk_energy_enrichment": 1.0,
            "boundary_to_nonboundary_energy_density_ratio": 1.0,
            "mean_high_frequency_energy_ratio": 0.1,
            "rms_cartesian_error_m": 0.001,
        }
        rows = [
            {
                "path_id": "path_0001",
                "method": method,
                "smoothness_claim_eligible": True,
                **required,
            }
            for method in benchmark.METHODS
        ]
        summary = benchmark.thesis_summary(rows, [])
        self.assertTrue(
            all(
                "median_boundary_jerk_energy_enrichment" in row
                and "median_boundary_time_fraction" in row
                and "median_boundary_to_nonboundary_energy_density_ratio" in row
                for row in summary
            )
        )

    def test_32c_enrichment_uses_one_value_per_path_in_stats(self) -> None:
        rows = []
        for path_index in range(3):
            for method_index, method in enumerate(benchmark.METHODS):
                row = {
                    "path_id": f"path_{path_index:04d}",
                    "method": method,
                    "smoothness_claim_eligible": True,
                }
                row.update(
                    {
                        metric: float(path_index + method_index + 1)
                        for metric in benchmark.PRIMARY_METRICS
                    }
                )
                rows.append(row)
        paths, arrays = benchmark.matched_values(
            rows, benchmark.METHODS, "boundary_jerk_energy_enrichment"
        )
        self.assertEqual(len(paths), 3)
        self.assertTrue(all(len(array) == 3 for array in arrays))
        statistics = benchmark.statistical_tests(rows)
        enrichment_rows = [
            row
            for row in statistics
            if row["metric"] == "boundary_jerk_energy_enrichment"
        ]
        self.assertTrue(enrichment_rows)
        self.assertTrue(all(row["sample_count"] == 3 for row in enrichment_rows))


class SafetyAndEndToEndTests(unittest.TestCase):
    """Cover strictness, output safety, provenance, and the full synthetic run."""

    def test_33_output_overwrite_protection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "output"
            output.mkdir()
            (output / "existing.txt").write_text("preserve", encoding="utf-8")
            with self.assertRaises(benchmark.BenchmarkError):
                benchmark.prepare_output(output, False)
            self.assertEqual((output / "existing.txt").read_text(encoding="utf-8"), "preserve")

    def test_34_empty_dataset_fails_clearly(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SyntheticRepository(Path(temporary))
            args = fixture.args(path_ids=())
            with self.assertRaisesRegex(benchmark.BenchmarkError, "No requested"):
                benchmark.run_benchmark(args, robot_context=object())

    def test_35_strict_missing_path_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SyntheticRepository(Path(temporary))
            fixture.add_path(1)
            args = fixture.args(path_ids=("path_0001", "path_9999"), strict=True)
            with mock.patch.object(benchmark, "authoritative_fk", side_effect=fake_fk):
                with self.assertRaisesRegex(benchmark.BenchmarkError, "path_9999"):
                    benchmark.run_benchmark(args, robot_context=object())

    def test_36_insufficient_pipeline_seeds_fail(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SyntheticRepository(Path(temporary))
            fixture.add_path(1, seeds=((11, True, 0.05),))
            args = fixture.args(minimum_pipeline_seeds=2)
            with mock.patch.object(benchmark, "authoritative_fk", side_effect=fake_fk):
                with self.assertRaisesRegex(benchmark.BenchmarkError, "No paths"):
                    benchmark.run_benchmark(args, robot_context=object())

    def test_37_full_synthetic_benchmark_outputs_and_audit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = SyntheticRepository(Path(temporary))
            fixture.add_path(1)
            fixture.add_path(
                2,
                seeds=((11, True, 0.04), (23, False, 0.001)),
                mlp_tracking_offset=0.02,
            )
            before = {path: digest(path) for path in fixture.files}
            args = fixture.args(
                path_ids=("path_0001", "path_0002", "path_9999")
            )
            with mock.patch.object(benchmark, "authoritative_fk", side_effect=fake_fk):
                result = benchmark.run_benchmark(args, robot_context=object())
            after = {path: digest(path) for path in fixture.files}
            self.assertEqual(before, after)
            self.assertEqual(result["metadata"]["paths_evaluated"], ["path_0001", "path_0002"])
            self.assertTrue(
                any(row["path_id"] == "path_9999" for row in result["excluded_paths"])
            )
            mlp_row = next(
                row
                for row in result["trajectory_rows"]
                if row["path_id"] == "path_0002" and row["method"] == "mlp"
            )
            self.assertFalse(mlp_row["tracking_eligible"])
            self.assertNotIn(
                ("path_0002", "mlp"),
                {
                    (row["path_id"], row["method"])
                    for row in result["path_rows"]
                    if row["smoothness_claim_eligible"]
                },
            )
            required_csv = {
                "per_trajectory_smoothness.csv",
                "per_path_method_smoothness.csv",
                "pipeline_seed_selection.csv",
                "pipeline_best_seed_smoothness.csv",
                "method_smoothness_summary.csv",
                "method_win_rates.csv",
                "path_level_statistical_tests.csv",
                "thesis_smoothness_summary.csv",
                "excluded_trajectories.csv",
                "excluded_paths.csv",
            }
            self.assertTrue(all((fixture.output / name).is_file() for name in required_csv))
            self.assertTrue((fixture.output / "benchmark_metadata.json").is_file())
            for stem in benchmark.REQUIRED_FIGURE_STEMS:
                self.assertTrue((fixture.output / f"{stem}.png").is_file())
                self.assertTrue((fixture.output / f"{stem}.pdf").is_file())
            metadata = json.loads(
                (fixture.output / "benchmark_metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["pipeline_primary_aggregation"], "median across eligible accepted diffusion seeds per complete path")
            self.assertTrue(metadata["exclusion_reasons"]["paths"])
            self.assertEqual(metadata["boundary_sensitivity_radii"], [0, 1, 2])
            self.assertEqual(metadata["boundary_primary_radius"], 2)
            self.assertEqual(metadata["mean_cartesian_error_threshold_m"], 0.01)
            self.assertEqual(metadata["max_cartesian_error_threshold_m"], 0.03)
            self.assertIn(
                "MAXIMUM_ALLOWED_MEAN_ERROR_GATE_M",
                metadata["mean_cartesian_threshold_source"],
            )
            self.assertIn(
                "ACCEPTANCE_MAX_ERROR",
                metadata["max_cartesian_threshold_source"],
            )
            module_file = benchmark.__file__
            assert module_file is not None
            source = Path(module_file).read_text(encoding="utf-8")
            for forbidden in ("generate_trajectory(", "run_inference(", "sample_diffusion("):
                self.assertNotIn(forbidden, source)

    def test_38_complete_trajectories_share_standardized_duration(self) -> None:
        timestamps_short = np.linspace(0.0, 1.0, 11)
        timestamps_long = np.linspace(2.0, 7.0, 21)
        trajectories = {}
        for method, timestamps in zip(
            benchmark.METHODS,
            (timestamps_short, timestamps_long, np.linspace(0.0, 3.0, 15)),
        ):
            q = polynomial_q(np.linspace(0.0, 1.0, len(timestamps)), (0.0, 0.1, 0.0, 0.0))
            trajectories[method] = comparison.Trajectory(
                method=method,
                provided_path=Path(f"/tmp/{method}/path_0001"),
                selected_file=Path(f"/tmp/{method}/path_0001/q.csv"),
                selected_array_key=None,
                original_shape=q.shape,
                q=q,
                timestamps=timestamps,
                timestamp_source="test",
                identities={"path_0001"},
            )
        aligned = comparison.align_trajectories(
            trajectories,
            timing_policy="common_duration",
            common_duration_s=10.0,
            common_samples=100,
        )
        self.assertTrue(aligned.complete_trajectory_used)
        self.assertEqual(aligned.timestamps[-1] - aligned.timestamps[0], 10.0)
        self.assertEqual({len(value) for value in aligned.q.values()}, {100})

    def test_39_script_imports_from_repository_root_and_script_directory(self) -> None:
        script = SCRIPT_DIR / "benchmark_ik_mlp_pipeline_smoothness.py"
        for cwd, command in (
            (SCRIPT_DIR, [sys.executable, script.name, "--help"]),
            (REPOSITORY_ROOT, [sys.executable, str(script.relative_to(REPOSITORY_ROOT)), "--help"]),
        ):
            with self.subTest(cwd=cwd):
                result = subprocess.run(
                    command,
                    cwd=cwd,
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=30,
                )
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_40_console_completion_sentinel(self) -> None:
        result = {
            "metadata": {
                "path_ids_requested": [],
                "paths_discovered": [],
                "paths_evaluated": [],
                "common_duration_s": 10.0,
                "common_samples": 100,
                "mean_cartesian_error_threshold_m": 0.01,
                "max_cartesian_error_threshold_m": 0.03,
            },
            "trajectory_rows": [],
            "path_rows": [
                {
                    "method": "pipeline",
                    "boundary_time_fraction": 0.08,
                    "boundary_jerk_energy_fraction": 0.1,
                    "boundary_jerk_energy_enrichment": 1.25,
                    "boundary_to_nonboundary_energy_density_ratio": 1.3,
                    "mean_high_frequency_energy_ratio": 0.2,
                }
            ],
            "thesis": [
                {
                    "method": method,
                    "median_rms_jerk": 1.0,
                    "median_integrated_squared_jerk": 2.0,
                }
                for method in benchmark.METHODS
            ],
            "wins": [],
            "excluded_paths": [],
            "output_dir": Path("/tmp/output"),
        }
        stream = io.StringIO()
        with contextlib.redirect_stdout(stream):
            benchmark.print_summary(result)
        output = stream.getvalue()
        self.assertIn("Mean Cartesian eligibility threshold: 0.01 m", output)
        self.assertIn("Maximum Cartesian eligibility threshold: 0.03 m", output)
        self.assertIn("Pipeline median boundary time fraction: 0.08", output)
        self.assertIn("Pipeline median boundary jerk-energy fraction: 0.1", output)
        self.assertIn("Pipeline median boundary jerk-energy enrichment: 1.25", output)
        self.assertIn("Enrichment near 1 indicates no concentration", output)
        self.assertTrue(
            output.rstrip().endswith(
                "IK_MLP_PIPELINE_SMOOTHNESS_BENCHMARK_COMPLETE"
            )
        )


if __name__ == "__main__":
    unittest.main()
