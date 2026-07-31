"""Focused tests for the saved-trajectory jerk comparison."""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import importlib
import io
import json
import math
import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np

comparison = importlib.import_module("compare_ik_mlp_pipeline_jerk_over_time")


def file_digest(path: Path) -> str:
    """Return a test input's content digest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_joint_csv(path: Path, timestamps: np.ndarray, q: np.ndarray) -> None:
    """Write a repository-style saved joint trajectory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("t", "q1", "q2", "q3", "q4", "q5", "q6"))
        for sample_index, time_s in enumerate(timestamps):
            writer.writerow((float(time_s), *q[sample_index].tolist()))


def polynomial_trajectory(
    timestamps: np.ndarray, coefficients: tuple[float, float, float, float]
) -> np.ndarray:
    """Create six scaled copies of one cubic polynomial."""
    a0, a1, a2, a3 = coefficients
    base = a0 + a1 * timestamps + a2 * timestamps**2 + a3 * timestamps**3
    return np.column_stack(
        [(joint_index + 1.0) * base for joint_index in range(6)]
    )


def make_trajectories(
    timestamps_by_method: dict[str, np.ndarray],
    q_by_method: dict[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Create matched in-memory trajectories for timing-policy tests."""
    trajectories: dict[str, Any] = {}
    for method in comparison.METHODS:
        timestamps = np.asarray(timestamps_by_method[method], dtype=np.float64)
        q = (
            np.asarray(q_by_method[method], dtype=np.float64)
            if q_by_method is not None
            else polynomial_trajectory(
                np.linspace(0.0, 1.0, len(timestamps)),
                (0.0, 0.2, 0.03, 0.004),
            )
        )
        trajectories[method] = comparison.Trajectory(
            method=method,
            provided_path=Path(f"/tmp/{method}/path_0001"),
            selected_file=Path(f"/tmp/{method}/path_0001/q.csv"),
            selected_array_key=None,
            original_shape=q.shape,
            q=q,
            timestamps=timestamps,
            timestamp_source=f"test:{method}",
            identities={"path_0001"},
        )
    return trajectories


class DerivativeTests(unittest.TestCase):
    """Verify the repository derivative convention on analytic trajectories."""

    def setUp(self) -> None:
        self.timestamps = np.linspace(0.0, 2.0, 201, dtype=np.float64)

    def test_constant_position_has_zero_derivatives(self) -> None:
        q = polynomial_trajectory(self.timestamps, (2.0, 0.0, 0.0, 0.0))
        velocity, acceleration, jerk = comparison.compute_derivatives(
            q, self.timestamps
        )
        self.assertTrue(np.allclose(velocity, 0.0, atol=1.0e-10))
        self.assertTrue(np.allclose(acceleration, 0.0, atol=1.0e-8))
        self.assertTrue(np.allclose(jerk, 0.0, atol=1.0e-6))

    def test_linear_position_has_constant_velocity(self) -> None:
        q = polynomial_trajectory(self.timestamps, (0.3, 1.5, 0.0, 0.0))
        velocity, acceleration, jerk = comparison.compute_derivatives(
            q, self.timestamps
        )
        expected = np.arange(1.0, 7.0) * 1.5
        self.assertTrue(np.allclose(velocity, expected, atol=1.0e-10))
        self.assertTrue(np.allclose(acceleration, 0.0, atol=1.0e-8))
        self.assertTrue(np.allclose(jerk, 0.0, atol=1.0e-6))

    def test_quadratic_position_has_constant_acceleration(self) -> None:
        q = polynomial_trajectory(self.timestamps, (0.1, -0.4, 0.75, 0.0))
        _, acceleration, jerk = comparison.compute_derivatives(q, self.timestamps)
        expected = np.arange(1.0, 7.0) * 1.5
        self.assertTrue(np.allclose(acceleration, expected, atol=1.0e-8))
        self.assertTrue(np.allclose(jerk, 0.0, atol=1.0e-6))

    def test_cubic_position_has_constant_jerk(self) -> None:
        q = polynomial_trajectory(self.timestamps, (0.0, 0.0, 0.0, 0.5))
        _, _, jerk = comparison.compute_derivatives(q, self.timestamps)
        expected = np.arange(1.0, 7.0) * 3.0
        self.assertTrue(np.allclose(jerk[3:-3], expected, atol=2.0e-8))


class LoadingAndValidationTests(unittest.TestCase):
    """Exercise shape, finite-value, identity, and input-safety gates."""

    def test_both_supported_joint_orientations(self) -> None:
        q = np.arange(120, dtype=np.float64).reshape(20, 6)
        normalized, original = comparison.normalize_joint_array(q, "rows")
        transposed, transposed_original = comparison.normalize_joint_array(
            q.T, "columns"
        )
        self.assertEqual(original, (20, 6))
        self.assertEqual(transposed_original, (6, 20))
        np.testing.assert_array_equal(normalized, q)
        np.testing.assert_array_equal(transposed, q)

    def test_invalid_joint_dimensions_are_rejected(self) -> None:
        for invalid in (
            np.zeros((20, 5)),
            np.zeros((20, 7)),
            np.zeros((6, 6)),
            np.zeros((20,)),
        ):
            with self.subTest(shape=invalid.shape):
                with self.assertRaises(comparison.ComparisonError):
                    comparison.normalize_joint_array(invalid, "invalid")

    def test_nan_and_infinity_are_rejected(self) -> None:
        for nonfinite in (math.nan, math.inf, -math.inf):
            q = np.zeros((20, 6), dtype=np.float64)
            q[4, 2] = nonfinite
            with self.subTest(nonfinite=nonfinite):
                with self.assertRaises(comparison.ComparisonError):
                    comparison.normalize_joint_array(q, "nonfinite")

    def test_mismatched_path_metadata_is_rejected(self) -> None:
        timestamps = np.linspace(0.0, 1.0, 20)
        q = np.zeros((20, 6))

        def trajectory(method: str, identity: str) -> Any:
            return comparison.Trajectory(
                method=method,
                provided_path=Path(f"/tmp/{identity}"),
                selected_file=Path(f"/tmp/{identity}/q.csv"),
                selected_array_key=None,
                original_shape=q.shape,
                q=q,
                timestamps=timestamps,
                timestamp_source="test",
                identities={identity},
            )

        trajectories = {
            "ik": trajectory("ik", "path_0001"),
            "mlp": trajectory("mlp", "path_0002"),
            "pipeline": trajectory("pipeline", "path_0001"),
        }
        with self.assertRaisesRegex(
            comparison.ComparisonError, "path metadata disagree"
        ):
            comparison.validate_path_compatibility(trajectories)

    def test_loading_does_not_modify_input(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "path_0007" / "ik_seed_q.csv"
            timestamps = np.linspace(0.0, 1.0, 30)
            write_joint_csv(source, timestamps, np.zeros((30, 6)))
            before = file_digest(source)
            loaded = comparison.load_trajectory(source, "ik")
            after = file_digest(source)
            self.assertEqual(before, after)
            self.assertEqual(loaded.q.shape, (30, 6))


class SummaryTests(unittest.TestCase):
    """Verify lower-is-better comparison arithmetic."""

    def test_percentage_improvement_signs(self) -> None:
        self.assertEqual(comparison.percent_improvement(10.0, 8.0), 20.0)
        self.assertEqual(comparison.percent_improvement(10.0, 12.0), -20.0)

    def test_zero_reference_returns_nan_not_infinity(self) -> None:
        value = comparison.percent_improvement(0.0, 1.0)
        self.assertTrue(math.isnan(value))
        self.assertFalse(math.isinf(value))


class TimingPolicyTests(unittest.TestCase):
    """Verify strict, standardized-duration, and diagnostic alignment behavior."""

    def equal_trajectories(self, count: int = 21) -> dict[str, Any]:
        timestamps = np.linspace(0.0, 1.0, count)
        return make_trajectories(
            {method: timestamps for method in comparison.METHODS}
        )

    def test_command_line_default_policy_is_require_equal(self) -> None:
        args = comparison.parse_args(
            [
                "--ik_path",
                "ik.csv",
                "--mlp_path",
                "mlp.csv",
                "--pipeline_path",
                "pipeline.csv",
                "--output_dir",
                "output",
            ]
        )
        self.assertEqual(args.timing_policy, "require_equal")

    def test_require_equal_accepts_equal_relative_timestamp_grids(self) -> None:
        aligned = comparison.align_trajectories(
            self.equal_trajectories(), "require_equal"
        )
        self.assertTrue(aligned.claim_eligible)
        self.assertFalse(aligned.interpolation_used)
        self.assertTrue(aligned.complete_trajectory_used)

    def test_require_equal_accepts_different_absolute_starts(self) -> None:
        base = np.linspace(0.0, 1.0, 21)
        trajectories = make_trajectories(
            {"ik": base, "mlp": base + 5.0, "pipeline": base + 100.0}
        )
        aligned = comparison.align_trajectories(
            trajectories, "require_equal"
        )
        np.testing.assert_allclose(aligned.timestamps, base)

    def test_require_equal_rejects_unequal_durations(self) -> None:
        trajectories = make_trajectories(
            {
                "ik": np.linspace(0.0, 1.0, 21),
                "mlp": np.linspace(0.0, 1.0, 21),
                "pipeline": np.linspace(0.0, 10.0, 21),
            }
        )
        with self.assertRaisesRegex(comparison.ComparisonError, "duration"):
            comparison.align_trajectories(trajectories, "require_equal")

    def test_require_equal_rejects_unequal_sample_counts(self) -> None:
        trajectories = make_trajectories(
            {
                "ik": np.linspace(0.0, 1.0, 20),
                "mlp": np.linspace(0.0, 1.0, 21),
                "pipeline": np.linspace(0.0, 1.0, 22),
            }
        )
        with self.assertRaisesRegex(comparison.ComparisonError, "sample_count"):
            comparison.align_trajectories(trajectories, "require_equal")

    def test_require_equal_rejects_unequal_relative_grids(self) -> None:
        base = np.linspace(0.0, 1.0, 21)
        nonuniform = base**2
        trajectories = make_trajectories(
            {"ik": base, "mlp": base, "pipeline": nonuniform}
        )
        with self.assertRaisesRegex(comparison.ComparisonError, "relative"):
            comparison.align_trajectories(trajectories, "require_equal")

    def test_common_duration_uses_complete_endpoints_and_never_crops(self) -> None:
        trajectories = make_trajectories(
            {
                "ik": np.linspace(0.0, 1.0, 21),
                "mlp": np.linspace(2.0, 4.0, 21),
                "pipeline": np.linspace(0.0, 10.0, 21),
            }
        )
        aligned = comparison.align_trajectories(
            trajectories, "common_duration", 8.0
        )
        self.assertTrue(aligned.complete_trajectory_used)
        self.assertFalse(aligned.shared_interval_crop_used)
        for method in comparison.METHODS:
            np.testing.assert_array_equal(
                aligned.q[method][0], trajectories[method].q[0]
            )
            np.testing.assert_array_equal(
                aligned.q[method][-1], trajectories[method].q[-1]
            )

    def test_common_duration_equal_lengths_retain_sample_count(self) -> None:
        aligned = comparison.align_trajectories(
            self.equal_trajectories(27), "common_duration", 4.0
        )
        self.assertEqual(aligned.common_sample_count, 27)
        self.assertEqual(len(aligned.timestamps), 27)

    def test_common_duration_unequal_lengths_require_common_samples(self) -> None:
        trajectories = make_trajectories(
            {
                "ik": np.linspace(0.0, 1.0, 20),
                "mlp": np.linspace(0.0, 1.0, 21),
                "pipeline": np.linspace(0.0, 1.0, 22),
            }
        )
        with self.assertRaisesRegex(comparison.ComparisonError, "common_samples"):
            comparison.align_trajectories(
                trajectories, "common_duration", 5.0
            )

    def test_explicit_common_samples_controls_output_length_and_duration(self) -> None:
        trajectories = make_trajectories(
            {
                "ik": np.linspace(0.0, 1.0, 20),
                "mlp": np.linspace(0.0, 2.0, 21),
                "pipeline": np.linspace(0.0, 10.0, 22),
            }
        )
        aligned = comparison.align_trajectories(
            trajectories, "common_duration", 7.5, 35
        )
        self.assertEqual(len(aligned.timestamps), 35)
        self.assertEqual(aligned.common_sample_count, 35)
        self.assertEqual(aligned.timestamps[0], 0.0)
        self.assertEqual(aligned.timestamps[-1], 7.5)
        self.assertTrue(
            all(
                row["interpolation_sample_count"] == 35
                for row in aligned.method_alignment.values()
            )
        )

    def test_standardization_makes_same_geometry_derivatives_identical(self) -> None:
        count = 41
        progress = np.linspace(0.0, 1.0, count)
        q = polynomial_trajectory(progress, (0.0, 0.1, 0.2, 0.3))
        trajectories = make_trajectories(
            {
                "ik": progress,
                "mlp": progress * 2.0,
                "pipeline": progress * 10.0,
            },
            {method: q for method in comparison.METHODS},
        )
        aligned = comparison.align_trajectories(
            trajectories, "common_duration", 6.0
        )
        derivatives = {
            method: comparison.compute_derivatives(
                aligned.q[method], aligned.timestamps
            )
            for method in comparison.METHODS
        }
        for derivative_index in range(3):
            np.testing.assert_allclose(
                derivatives["ik"][derivative_index],
                derivatives["mlp"][derivative_index],
                atol=1.0e-10,
            )
            np.testing.assert_allclose(
                derivatives["ik"][derivative_index],
                derivatives["pipeline"][derivative_index],
                atol=1.0e-10,
            )

    def test_doubling_common_duration_reduces_jerk_by_eight(self) -> None:
        trajectories = self.equal_trajectories(101)
        one_second = comparison.align_trajectories(
            trajectories, "common_duration", 1.0
        )
        two_seconds = comparison.align_trajectories(
            trajectories, "common_duration", 2.0
        )
        jerk_one = comparison.compute_derivatives(
            one_second.q["ik"], one_second.timestamps
        )[2]
        jerk_two = comparison.compute_derivatives(
            two_seconds.q["ik"], two_seconds.timestamps
        )[2]
        ratio = float(
            np.sqrt(np.mean(np.square(jerk_one)))
            / np.sqrt(np.mean(np.square(jerk_two)))
        )
        self.assertAlmostEqual(ratio, 8.0, places=7)

    def test_equal_dt_with_unequal_counts_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "path_0001"
            trajectories: dict[str, Any] = {}
            for method, count in zip(comparison.METHODS, (20, 21, 22)):
                source = root / f"{method}.csv"
                saved_times = np.linspace(0.0, 1.0, count)
                write_joint_csv(
                    source,
                    saved_times,
                    polynomial_trajectory(
                        np.linspace(0.0, 1.0, count),
                        (0.0, 0.2, 0.0, 0.0),
                    ),
                )
                trajectories[method] = comparison.load_trajectory(
                    source, method, dt_override=0.05
                )
                self.assertEqual(
                    trajectories[method].timestamp_source,
                    "command_line:--dt",
                )
            with self.assertRaises(comparison.ComparisonError):
                comparison.align_trajectories(
                    trajectories, "require_equal"
                )

    def test_shared_interval_diagnostic_retains_legacy_crop(self) -> None:
        trajectories = make_trajectories(
            {
                "ik": np.linspace(0.0, 1.0, 21),
                "mlp": np.linspace(0.0, 1.0, 21),
                "pipeline": np.linspace(0.0, 10.0, 101),
            }
        )
        aligned = comparison.align_trajectories(
            trajectories, "shared_interval_diagnostic"
        )
        self.assertEqual(aligned.timestamps[0], 0.0)
        self.assertAlmostEqual(aligned.timestamps[-1], 1.0)
        self.assertTrue(aligned.shared_interval_crop_used)
        self.assertFalse(aligned.complete_trajectory_used)
        self.assertFalse(aligned.claim_eligible)

    def test_claim_eligibility_by_policy(self) -> None:
        trajectories = self.equal_trajectories()
        strict = comparison.align_trajectories(
            trajectories, "require_equal"
        )
        standardized = comparison.align_trajectories(
            trajectories, "common_duration", 3.0
        )
        diagnostic = comparison.align_trajectories(
            trajectories, "shared_interval_diagnostic"
        )
        self.assertTrue(strict.claim_eligible)
        self.assertTrue(standardized.claim_eligible)
        self.assertFalse(diagnostic.claim_eligible)


class OutputTests(unittest.TestCase):
    """Run a complete lightweight analysis and verify all output families."""

    def test_csv_and_figure_outputs_are_created(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            timestamps = np.linspace(0.0, 1.0, 31)
            sources: dict[str, Path] = {}
            for method_index, method in enumerate(comparison.METHODS):
                source = root / method / "path_0042" / f"{method}.csv"
                q = polynomial_trajectory(
                    timestamps, (0.1 * method_index, 0.2, 0.03, 0.001)
                )
                write_joint_csv(source, timestamps, q)
                sources[method] = source

            output = root / "output"
            args = argparse.Namespace(
                ik_path=sources["ik"],
                mlp_path=sources["mlp"],
                pipeline_path=sources["pipeline"],
                output_dir=output,
                dt=None,
                path_id="path_0042",
                stroke_id=None,
                title="Synthetic comparison",
                show=False,
                overwrite=False,
                smoothing="none",
            )
            result = comparison.run_comparison(args)
            self.assertEqual(result["output_dir"], output.resolve())
            required_csv_json = (
                "aligned_joint_positions.csv",
                "joint_velocity_over_time.csv",
                "joint_acceleration_over_time.csv",
                "joint_jerk_over_time.csv",
                "joint_jerk_summary.csv",
                "method_jerk_summary.csv",
                "jerk_pairwise_comparison.csv",
                "comparison_metadata.json",
            )
            for filename in required_csv_json:
                with self.subTest(filename=filename):
                    self.assertTrue((output / filename).is_file())
            for filename in (
                "aligned_joint_positions.csv",
                "joint_velocity_over_time.csv",
                "joint_acceleration_over_time.csv",
                "joint_jerk_over_time.csv",
            ):
                with (output / filename).open(
                    "r", encoding="utf-8", newline=""
                ) as handle:
                    header = next(csv.reader(handle))
                self.assertEqual(header[:2], ["time_s", "progress_0_1"])
            metadata = json.loads(
                (output / "comparison_metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["timing_policy"], "require_equal")
            self.assertTrue(metadata["claim_eligible"])
            self.assertTrue(metadata["complete_trajectory_used"])
            self.assertFalse(metadata["shared_interval_crop_used"])
            figure_stems = (
                "joint_jerk_over_time",
                "joint_absolute_jerk_over_time",
                "aggregate_jerk_over_time",
                "per_joint_rms_jerk",
                "joint_positions_over_time",
                "joint_acceleration_over_time",
            )
            for stem in figure_stems:
                for extension in ("png", "pdf"):
                    with self.subTest(stem=stem, extension=extension):
                        self.assertTrue((output / f"{stem}.{extension}").is_file())

    def test_end_to_end_analysis_preserves_all_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            timestamps = np.linspace(0.0, 1.0, 21)
            sources: dict[str, Path] = {}
            for method in comparison.METHODS:
                source = root / method / "path_0011" / f"{method}.csv"
                write_joint_csv(
                    source,
                    timestamps,
                    polynomial_trajectory(timestamps, (0.0, 0.1, 0.0, 0.0)),
                )
                sources[method] = source
            before = {method: file_digest(path) for method, path in sources.items()}
            args = argparse.Namespace(
                ik_path=sources["ik"],
                mlp_path=sources["mlp"],
                pipeline_path=sources["pipeline"],
                output_dir=root / "analysis",
                dt=None,
                path_id="path_0011",
                stroke_id=None,
                title=None,
                show=False,
                overwrite=False,
                smoothing="none",
            )
            comparison.run_comparison(args)
            after = {method: file_digest(path) for method, path in sources.items()}
            self.assertEqual(before, after)

    def test_common_duration_metadata_records_alignment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "source.csv"
            source.write_text("saved trajectory placeholder\n", encoding="utf-8")
            trajectories = make_trajectories(
                {
                    "ik": np.linspace(0.0, 1.0, 21),
                    "mlp": np.linspace(0.0, 2.0, 21),
                    "pipeline": np.linspace(0.0, 10.0, 21),
                }
            )
            for item in trajectories.values():
                item.selected_file = source
            aligned = comparison.align_trajectories(
                trajectories, "common_duration", 10.0
            )
            args = argparse.Namespace(dt=None)
            metadata = comparison.create_metadata(
                trajectories, aligned, {}, args, aligned.warnings
            )
            self.assertEqual(metadata["timing_policy"], "common_duration")
            self.assertTrue(metadata["claim_eligible"])
            self.assertTrue(metadata["complete_trajectory_used"])
            self.assertTrue(metadata["duration_standardized"])
            self.assertTrue(metadata["progress_interpolation_used"])
            self.assertFalse(metadata["shared_interval_crop_used"])
            self.assertEqual(metadata["common_duration_s"], 10.0)
            self.assertEqual(metadata["common_sample_count"], 21)
            for method in comparison.METHODS:
                alignment = metadata["methods"][method]["alignment"]
                self.assertEqual(alignment["original_progress_range"], [0.0, 1.0])
                self.assertEqual(
                    alignment[
                        "first_joint_position_difference_after_interpolation"
                    ],
                    [0.0] * 6,
                )
                self.assertEqual(
                    alignment[
                        "last_joint_position_difference_after_interpolation"
                    ],
                    [0.0] * 6,
                )

    def test_inspect_only_creates_no_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            timestamps = np.linspace(0.0, 1.0, 21)
            sources: dict[str, Path] = {}
            for method in comparison.METHODS:
                source = root / method / "path_0033" / f"{method}.csv"
                write_joint_csv(
                    source,
                    timestamps,
                    polynomial_trajectory(
                        timestamps, (0.0, 0.1, 0.0, 0.0)
                    ),
                )
                sources[method] = source
            output = root / "must_not_exist"
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                return_code = comparison.main(
                    [
                        "--ik_path",
                        str(sources["ik"]),
                        "--mlp_path",
                        str(sources["mlp"]),
                        "--pipeline_path",
                        str(sources["pipeline"]),
                        "--output_dir",
                        str(output),
                        "--path_id",
                        "path_0033",
                        "--inspect_only",
                    ]
                )
            self.assertEqual(return_code, 0)
            self.assertIn(
                "IK_MLP_PIPELINE_TIMING_INSPECTION_COMPLETE",
                stdout.getvalue(),
            )
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
