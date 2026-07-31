#!/usr/bin/env python3
"""CPU-only tests for independent legacy SmartJoint CSV validation."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from build_multistroke_full_pose_execution import (
    LocalSegment,
    append_segments,
    export_legacy_smartjoint_csv,
)
from orientation_aware_adaptive_ik import quaternion_from_rotation_matrix
from validate_multistroke_full_pose_execution import (
    LEGACY_CSV_COLUMNS,
    LEGACY_CSV_NAME,
    ManifestRow,
    ValidationError,
    validate_legacy_smartjoint_csv,
)


def local_segment(
    segment_type: str,
    stroke_index: int,
    values: np.ndarray,
) -> LocalSegment:
    values = np.asarray(values, dtype=np.float64)
    q = np.repeat(values[:, np.newaxis], 6, axis=1)
    desired = np.column_stack(
        (values, np.zeros(len(values)), np.zeros(len(values)))
    )
    times = np.arange(len(values), dtype=np.float64) * 0.05
    return LocalSegment(
        segment_type=segment_type,
        stroke_index=stroke_index,
        local_times=times,
        q=q,
        desired_position=desired,
        planned_duration_seconds=float(times[-1]),
    )


def manifest_rows(rows: Sequence[Dict[str, Any]]) -> List[ManifestRow]:
    return [
        ManifestRow(
            segment_index=int(row["segment_index"]),
            segment_type=str(row["segment_type"]),
            stroke_index=int(row["stroke_index"]),
            start_sample=int(row["start_sample"]),
            end_sample=int(row["end_sample"]),
            start_time=float(row["start_time_seconds"]),
            end_time=float(row["end_time_seconds"]),
            sample_count=int(row["sample_count"]),
            duration=float(row["duration_seconds"]),
            planned_duration=float(row["planned_duration_seconds"]),
            boundary_removed=int(row["duplicated_first_boundary_removed"]),
        )
        for row in rows
    ]


def read_legacy(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def write_legacy(
    path: Path,
    columns: Sequence[str],
    rows: Sequence[Dict[str, Any]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        writer.writerows(rows)


class LegacyValidationFixture:
    def __init__(self, root: Path, three_strokes: bool = False) -> None:
        if three_strokes:
            segments = [
                local_segment("initial_hover", 0, np.asarray([0.0])),
                local_segment("initial_descent", 0, np.arange(0.0, 31.0)),
                local_segment("drawing_stroke", 0, np.arange(30.0, 130.0)),
                local_segment("lift", 0, np.arange(129.0, 150.0)),
                local_segment("hover_travel", 1, np.arange(149.0, 190.0)),
                local_segment("descent", 1, np.arange(189.0, 210.0)),
                local_segment("drawing_stroke", 1, np.arange(209.0, 309.0)),
                local_segment("lift", 1, np.arange(308.0, 329.0)),
                local_segment("hover_travel", 2, np.arange(328.0, 369.0)),
                local_segment("descent", 2, np.arange(368.0, 389.0)),
                local_segment("drawing_stroke", 2, np.arange(388.0, 488.0)),
                local_segment("final_lift", 2, np.arange(487.0, 508.0)),
            ]
            self.stroke_count = 3
        else:
            segments = [
                local_segment("initial_hover", 0, np.asarray([0.0])),
                local_segment("initial_descent", 0, np.arange(0.0, 3.0)),
                local_segment("drawing_stroke", 0, np.arange(2.0, 102.0)),
                local_segment("final_lift", 0, np.arange(101.0, 104.0)),
            ]
            self.stroke_count = 1
        (
            self.timestamps,
            self.q,
            _,
            _,
            self.segment_types,
            self.stroke_indices,
            raw_manifest,
        ) = append_segments(segments)
        self.manifest = manifest_rows(raw_manifest)
        trajectory_rows: List[Dict[str, Any]] = []
        for row_index in range(len(self.q)):
            trajectory_rows.append(
                {
                    "time_seconds": float(self.timestamps[row_index]),
                    **{
                        f"q{joint + 1}": float(self.q[row_index, joint])
                        for joint in range(6)
                    },
                    "segment_type": str(self.segment_types[row_index]),
                    "stroke_index": int(self.stroke_indices[row_index]),
                }
            )
        self.root = root
        self.legacy_path = root / LEGACY_CSV_NAME
        export_legacy_smartjoint_csv(trajectory_rows, self.legacy_path)
        self.metrics: Dict[str, Any] = {
            "legacy_smartjoint_csv": str(self.legacy_path.resolve()),
            "legacy_smartjoint_row_count": len(self.timestamps),
            "legacy_drawing_stroke_ids": list(
                range(1, self.stroke_count + 1)
            ),
        }
        (root / "multistroke_execution_report.txt").write_text(
            "\n".join(
                (
                    f"legacy_smartjoint_csv: "
                    f"{self.metrics['legacy_smartjoint_csv']}",
                    f"legacy_smartjoint_rows: "
                    f"{self.metrics['legacy_smartjoint_row_count']}",
                    f"legacy_drawing_stroke_ids: "
                    f"{self.metrics['legacy_drawing_stroke_ids']}",
                )
            )
            + "\n",
            encoding="utf-8",
        )

    def validate(
        self,
        *,
        accepted: bool = True,
        metrics: Dict[str, Any] | None = None,
    ) -> None:
        validate_legacy_smartjoint_csv(
            self.root,
            accepted,
            self.timestamps,
            self.q,
            self.segment_types,
            self.stroke_indices,
            self.manifest,
            self.metrics if metrics is None else metrics,
            self.stroke_count,
        )


class ValidateLegacySmartJointCsvTest(unittest.TestCase):
    def test_valid_accepted_legacy_csv(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            fixture = LegacyValidationFixture(Path(temporary_directory))
            fixture.validate()

    def test_missing_accepted_legacy_csv(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            fixture = LegacyValidationFixture(Path(temporary_directory))
            fixture.legacy_path.unlink()
            with self.assertRaisesRegex(
                ValidationError,
                "Required file is missing",
            ):
                fixture.validate()

    def test_legacy_csv_present_for_rejected_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            fixture = LegacyValidationFixture(Path(temporary_directory))
            rejected_metrics = {
                "legacy_smartjoint_csv": None,
                "legacy_smartjoint_row_count": 0,
                "legacy_drawing_stroke_ids": [],
            }
            with self.assertRaisesRegex(
                ValidationError,
                "Rejected execution must not contain",
            ):
                fixture.validate(accepted=False, metrics=rejected_metrics)

    def test_extra_or_reordered_columns(self) -> None:
        for mode in ("extra", "reordered"):
            with self.subTest(mode=mode):
                with tempfile.TemporaryDirectory() as temporary_directory:
                    fixture = LegacyValidationFixture(Path(temporary_directory))
                    _, rows = read_legacy(fixture.legacy_path)
                    if mode == "extra":
                        columns = [*LEGACY_CSV_COLUMNS, "Extra"]
                        for row in rows:
                            row["Extra"] = "unexpected"
                    else:
                        columns = list(reversed(LEGACY_CSV_COLUMNS))
                    write_legacy(fixture.legacy_path, columns, rows)
                    with self.assertRaisesRegex(
                        ValidationError,
                        "columns must be exactly",
                    ):
                        fixture.validate()

    def test_modified_joint_is_rejected_exactly(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            fixture = LegacyValidationFixture(Path(temporary_directory))
            columns, rows = read_legacy(fixture.legacy_path)
            rows[4]["joint3"] = str(float(rows[4]["joint3"]) + 1.0e-12)
            write_legacy(fixture.legacy_path, columns, rows)
            with self.assertRaisesRegex(
                ValidationError,
                "legacy SmartJoint q is not an exact match",
            ):
                fixture.validate()

    def test_incorrect_drawing_start_ownership(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            fixture = LegacyValidationFixture(Path(temporary_directory))
            columns, rows = read_legacy(fixture.legacy_path)
            drawing = next(
                row
                for row in fixture.manifest
                if row.segment_type == "drawing_stroke"
            )
            rows[drawing.start_sample]["TouchType"] = "Air"
            rows[drawing.start_sample]["OriginalStatus"] = "MOVING_FAST"
            write_legacy(fixture.legacy_path, columns, rows)
            with self.assertRaisesRegex(
                ValidationError,
                "TouchType mismatch",
            ):
                fixture.validate()

    def test_incorrect_drawing_count(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            fixture = LegacyValidationFixture(Path(temporary_directory))
            columns, rows = read_legacy(fixture.legacy_path)
            drawing = next(
                row
                for row in fixture.manifest
                if row.segment_type == "drawing_stroke"
            )
            changed = drawing.start_sample + 1
            rows[changed]["TouchType"] = "Air"
            rows[changed]["OriginalStatus"] = "MOVING_FAST"
            write_legacy(fixture.legacy_path, columns, rows)
            with self.assertRaisesRegex(
                ValidationError,
                "TouchType mismatch",
            ):
                fixture.validate()

    def test_incorrect_stroke_numbering(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            fixture = LegacyValidationFixture(Path(temporary_directory))
            columns, rows = read_legacy(fixture.legacy_path)
            drawing = next(
                row
                for row in fixture.manifest
                if row.segment_type == "drawing_stroke"
            )
            rows[drawing.start_sample]["OriginalStatus"] = "DRAWING_STROKE_0"
            write_legacy(fixture.legacy_path, columns, rows)
            with self.assertRaisesRegex(
                ValidationError,
                "OriginalStatus mismatch",
            ):
                fixture.validate()

    def test_incorrect_metrics(self) -> None:
        changes = (
            ("row_count", {"legacy_smartjoint_row_count": -1}),
            ("stroke_ids", {"legacy_drawing_stroke_ids": [2]}),
            ("path", {"legacy_smartjoint_csv": "/wrong/path.csv"}),
        )
        for label, change in changes:
            with self.subTest(label=label):
                with tempfile.TemporaryDirectory() as temporary_directory:
                    fixture = LegacyValidationFixture(Path(temporary_directory))
                    metrics = dict(fixture.metrics)
                    metrics.update(change)
                    with self.assertRaises(ValidationError):
                        fixture.validate(metrics=metrics)

    def test_current_synthetic_508_row_structure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            fixture = LegacyValidationFixture(
                Path(temporary_directory),
                three_strokes=True,
            )
            fixture.validate()
            self.assertEqual(len(fixture.timestamps), 508)
            self.assertTrue(np.all(np.diff(fixture.timestamps) > 0.0))
            drawing = [
                row
                for row in fixture.manifest
                if row.segment_type == "drawing_stroke"
            ]
            self.assertEqual(
                [(row.start_sample, row.end_sample) for row in drawing],
                [(30, 129), (209, 308), (388, 487)],
            )
            _, rows = read_legacy(fixture.legacy_path)
            for stroke_number in (1, 2, 3):
                self.assertEqual(
                    sum(
                        row["TouchType"] == "Pen"
                        and row["OriginalStatus"]
                        == f"DRAWING_STROKE_{stroke_number}"
                        for row in rows
                    ),
                    100,
                )

    def test_read_only_rotation_matrix_is_supported(self) -> None:
        rotation = np.eye(3, dtype=np.float64)
        rotation.setflags(write=False)
        quaternion = quaternion_from_rotation_matrix(rotation)
        self.assertEqual(quaternion.shape, (4,))
        self.assertTrue(np.all(np.isfinite(quaternion)))


if __name__ == "__main__":
    unittest.main()
