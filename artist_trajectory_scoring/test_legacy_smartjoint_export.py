#!/usr/bin/env python3
"""Lightweight tests for the legacy SmartJoint trajectory exporter."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from build_multistroke_full_pose_execution import (
    LEGACY_SMARTJOINT_COLUMNS,
    LocalSegment,
    append_segments,
    export_legacy_smartjoint_csv,
    find_exact_subsequence,
)


def trajectory_row(
    row_index: int,
    segment_type: str,
    stroke_index: Any,
) -> Dict[str, Any]:
    return {
        "time_seconds": row_index * 0.1,
        "q1": row_index + 0.01,
        "q2": row_index + 0.02,
        "q3": row_index + 0.03,
        "q4": row_index + 0.04,
        "q5": row_index + 0.05,
        "q6": row_index + 0.06,
        "segment_type": segment_type,
        "stroke_index": stroke_index,
    }


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


class LegacySmartJointExportTest(unittest.TestCase):
    def source_rows(self) -> List[Dict[str, Any]]:
        return [
            trajectory_row(0, "initial_hover", 0),
            trajectory_row(1, "initial_descent", 0),
            trajectory_row(2, "drawing_stroke", 0),
            trajectory_row(3, "lift", 0),
            trajectory_row(4, "hover_travel", 1),
            trajectory_row(5, "descent", 1),
            trajectory_row(6, "drawing_stroke", 1),
            trajectory_row(7, "final_lift", 1),
        ]

    def test_exact_mapping_headers_statuses_and_rows(self) -> None:
        source = self.source_rows()
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "SmartJoint_Data_diffusion.csv"
            row_count, drawing_ids = export_legacy_smartjoint_csv(
                source,
                output_path,
            )
            with output_path.open(
                "r",
                encoding="utf-8",
                newline="",
            ) as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(
                    reader.fieldnames,
                    list(LEGACY_SMARTJOINT_COLUMNS),
                )
                output = list(reader)

        self.assertEqual(row_count, len(source))
        self.assertEqual(len(output), len(source))
        self.assertEqual(drawing_ids, [1, 2])
        self.assertEqual(
            [row["OriginalStatus"] for row in output],
            [
                "RECORDING_START",
                "MOVING_FAST",
                "DRAWING_STROKE_1",
                "END_STROKE_1",
                "MOVING_FAST",
                "MOVING_FAST",
                "DRAWING_STROKE_2",
                "END_STROKE_2",
            ],
        )
        self.assertEqual(
            [row["TouchType"] for row in output],
            ["Air", "Air", "Pen", "Air", "Air", "Air", "Pen", "Air"],
        )
        for source_row, output_row in zip(source, output):
            self.assertEqual(
                float(output_row["Timestamp"]),
                source_row["time_seconds"],
            )
            for joint in range(1, 7):
                self.assertEqual(
                    float(output_row[f"joint{joint}"]),
                    source_row[f"q{joint}"],
                )

    def test_missing_required_column_is_rejected(self) -> None:
        source = self.source_rows()
        del source[2]["q4"]
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "legacy.csv"
            with self.assertRaisesRegex(
                ValueError,
                "missing required columns",
            ):
                export_legacy_smartjoint_csv(source, output_path)

    def test_invalid_required_stroke_index_is_rejected(self) -> None:
        for segment_type, invalid_value in (
            ("drawing_stroke", None),
            ("lift", float("nan")),
        ):
            with self.subTest(segment_type=segment_type):
                source = self.source_rows()
                source[2] = trajectory_row(2, segment_type, invalid_value)
                with tempfile.TemporaryDirectory() as temporary_directory:
                    output_path = Path(temporary_directory) / "legacy.csv"
                    with self.assertRaisesRegex(
                        ValueError,
                        "stroke_index",
                    ):
                        export_legacy_smartjoint_csv(source, output_path)

    def test_drawing_segment_owns_shared_boundary(self) -> None:
        segments = [
            local_segment("initial_descent", 0, np.arange(0.0, 3.0)),
            local_segment("drawing_stroke", 0, np.arange(2.0, 6.0)),
            local_segment("lift", 0, np.arange(5.0, 8.0)),
        ]
        (
            timestamps,
            q,
            _,
            _,
            segment_types,
            stroke_indices,
            manifest,
        ) = append_segments(segments)

        expected_drawing = segments[1].q
        drawing_row = manifest[1]
        drawing_start = int(drawing_row["start_sample"])
        drawing_end = int(drawing_row["end_sample"])
        self.assertEqual(len(q), sum(len(item.q) for item in segments) - 2)
        self.assertTrue(np.all(np.diff(timestamps) > 0.0))
        self.assertEqual(
            find_exact_subsequence(q, segments[0].q[-1:]),
            [drawing_start],
        )
        self.assertEqual(manifest[0]["end_sample"], drawing_start - 1)
        self.assertEqual(manifest[0]["sample_count"], 2)
        self.assertEqual(drawing_row["start_sample"], 2)
        self.assertEqual(drawing_row["end_sample"], 5)
        self.assertEqual(drawing_row["sample_count"], 4)
        self.assertEqual(
            drawing_row["duplicated_first_boundary_removed"],
            0,
        )
        self.assertTrue(
            np.array_equal(q[drawing_start : drawing_end + 1], expected_drawing)
        )
        self.assertTrue(
            np.all(
                segment_types[drawing_start : drawing_end + 1]
                == "drawing_stroke"
            )
        )
        self.assertTrue(
            np.all(stroke_indices[drawing_start : drawing_end + 1] == 0)
        )
        self.assertEqual(manifest[2]["start_sample"], drawing_end + 1)
        for previous, current in zip(manifest, manifest[1:]):
            self.assertEqual(
                int(current["start_sample"]),
                int(previous["end_sample"]) + 1,
            )

    def test_three_100_sample_drawings_export_as_pen(self) -> None:
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
        (
            timestamps,
            q,
            _,
            _,
            segment_types,
            stroke_indices,
            manifest,
        ) = append_segments(segments)
        self.assertEqual(len(q), 508)
        self.assertTrue(np.all(np.diff(timestamps) > 0.0))
        drawing_rows = [
            row for row in manifest if row["segment_type"] == "drawing_stroke"
        ]
        self.assertEqual(
            [
                (row["start_sample"], row["end_sample"], row["sample_count"])
                for row in drawing_rows
            ],
            [(30, 129, 100), (209, 308, 100), (388, 487, 100)],
        )

        source_rows: List[Dict[str, Any]] = []
        for row_index in range(len(q)):
            source_rows.append(
                {
                    "time_seconds": float(timestamps[row_index]),
                    **{
                        f"q{joint + 1}": float(q[row_index, joint])
                        for joint in range(6)
                    },
                    "segment_type": str(segment_types[row_index]),
                    "stroke_index": int(stroke_indices[row_index]),
                }
            )
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_path = Path(temporary_directory) / "SmartJoint_Data_diffusion.csv"
            row_count, drawing_ids = export_legacy_smartjoint_csv(
                source_rows,
                output_path,
            )
            with output_path.open(
                "r",
                encoding="utf-8",
                newline="",
            ) as handle:
                output_rows = list(csv.DictReader(handle))

        self.assertEqual(row_count, 508)
        self.assertEqual(drawing_ids, [1, 2, 3])
        for stroke_number, drawing_row in enumerate(drawing_rows, start=1):
            start = int(drawing_row["start_sample"])
            end = int(drawing_row["end_sample"])
            self.assertEqual(
                sum(
                    row["TouchType"] == "Pen"
                    and row["OriginalStatus"]
                    == f"DRAWING_STROKE_{stroke_number}"
                    for row in output_rows
                ),
                100,
            )
            self.assertEqual(output_rows[start]["TouchType"], "Pen")
            self.assertEqual(
                output_rows[start]["OriginalStatus"],
                f"DRAWING_STROKE_{stroke_number}",
            )
            self.assertTrue(
                np.array_equal(
                    q[start : end + 1],
                    segments[2 + (stroke_number - 1) * 4].q,
                )
            )


if __name__ == "__main__":
    unittest.main()
