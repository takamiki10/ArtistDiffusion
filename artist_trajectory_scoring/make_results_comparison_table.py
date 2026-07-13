#!/usr/bin/env python3
"""
Create a clean results comparison table for the v3 artist trajectory pipeline.

The script reads the evaluation/refinement CSVs already produced by the project
and writes both CSV and Markdown summaries.

Default inputs match the paths recorded in handoff.md.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import pandas as pd


def parse_time_to_seconds(text: str | None) -> float | None:
    if text is None:
        return None
    s = str(text).strip().lower()
    if not s:
        return None

    # Accept raw seconds.
    try:
        return float(s)
    except ValueError:
        pass

    # Accept forms like 91m10.374s, 31.7min, 00:31:42.
    m = re.fullmatch(r"(?:(\d+(?:\.\d+)?)h)?(?:(\d+(?:\.\d+)?)m)?(?:(\d+(?:\.\d+)?)s)?", s)
    if m and any(m.groups()):
        hours = float(m.group(1) or 0)
        minutes = float(m.group(2) or 0)
        seconds = float(m.group(3) or 0)
        return hours * 3600 + minutes * 60 + seconds

    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*(?:min|mins|minute|minutes)", s)
    if m:
        return float(m.group(1)) * 60

    parts = s.split(":")
    if len(parts) == 3:
        h, m_, sec = parts
        return float(h) * 3600 + float(m_) * 60 + float(sec)
    if len(parts) == 2:
        m_, sec = parts
        return float(m_) * 60 + float(sec)

    raise ValueError(f"Could not parse time string: {text!r}")


def fmt_time(seconds: float | None) -> str:
    if seconds is None or (isinstance(seconds, float) and math.isnan(seconds)):
        return ""
    if seconds < 60:
        return f"{seconds:.1f} s"
    return f"{seconds / 60:.1f} min"


def fmt_markdown_cell(value: object, floatfmt: str = ".6g") -> str:
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
        [fmt_markdown_cell(value, floatfmt=floatfmt) for value in row]
        for row in df.itertuples(index=False, name=None)
    ]

    widths = [
        max(len(str(column)), *(len(row[i]) for row in rows)) if rows else len(str(column))
        for i, column in enumerate(columns)
    ]

    def render_row(values: list[str]) -> str:
        return "| " + " | ".join(value.ljust(widths[i]) for i, value in enumerate(values)) + " |"

    header = render_row([str(column) for column in columns])
    separator = "| " + " | ".join("-" * width for width in widths) + " |"
    body = [render_row(row) for row in rows]
    return "\n".join([header, separator, *body])


def mean_col(df: pd.DataFrame, *names: str) -> float | None:
    for name in names:
        if name in df.columns:
            return float(pd.to_numeric(df[name], errors="coerce").mean())
    return None


def max_col(df: pd.DataFrame, *names: str) -> float | None:
    for name in names:
        if name in df.columns:
            return float(pd.to_numeric(df[name], errors="coerce").max())
    return None


def count_accepted(df: pd.DataFrame) -> int:
    if "accepted" in df.columns:
        return int(pd.to_numeric(df["accepted"], errors="coerce").fillna(0).astype(bool).sum())
    return len(df)


def add_row(
    rows: list[dict],
    *,
    method: str,
    evaluated: int | None,
    accepted: int | None,
    mean_path_error: float | None,
    mean_cartesian_error: float | None,
    mean_max_error: float | None,
    worst_max_error: float | None,
    mean_solve_time_sec: float | None,
    total_time_sec: float | None,
    notes: str,
) -> None:
    rows.append(
        {
            "method": method,
            "evaluated": evaluated,
            "accepted": accepted,
            "mean_path_error": mean_path_error,
            "mean_cartesian_error_m": mean_cartesian_error,
            "mean_max_error_m": mean_max_error,
            "worst_max_error_m": worst_max_error,
            "mean_solve_time_sec": mean_solve_time_sec,
            "total_time_sec": total_time_sec,
            "total_time_readable": fmt_time(total_time_sec),
            "notes": notes,
        }
    )


def read_optional(path: Path) -> pd.DataFrame | None:
    if path.exists():
        return pd.read_csv(path)
    print(f"Warning: missing input, skipped: {path}")
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build v3 result comparison table.")
    parser.add_argument("--mlp_eval_csv", type=Path, default=Path("data/cartesian_expert_dataset_v3/test_eval.csv"))
    parser.add_argument("--cold_ik_csv", type=Path, default=Path("data/cartesian_expert_dataset_v3/cold_ik_test_timed/ik_generation_summary.csv"))
    parser.add_argument("--refine_001_csv", type=Path, default=Path("data/cartesian_expert_dataset_v3/mlp_ik_refine_test_summary.csv"))
    parser.add_argument("--refine_0001_csv", type=Path, default=Path("data/cartesian_expert_dataset_v3/mlp_ik_refine_test_summary_smooth001.csv"))
    parser.add_argument("--adaptive_csv", type=Path, default=None, help="Optional adaptive refinement summary CSV.")
    parser.add_argument("--cold_total_time", default="91m10.374s", help="Cold IK total runtime, e.g. 91m10.374s or seconds.")
    parser.add_argument("--output_csv", type=Path, default=Path("data/cartesian_expert_dataset_v3/results_comparison_table.csv"))
    parser.add_argument("--output_md", type=Path, default=Path("data/cartesian_expert_dataset_v3/results_comparison_table.md"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[dict] = []

    mlp = read_optional(args.mlp_eval_csv)
    if mlp is not None:
        add_row(
            rows,
            method="MLP only",
            evaluated=len(mlp),
            accepted=len(mlp),
            mean_path_error=mean_col(mlp, "path_error", "before_path_error"),
            mean_cartesian_error=mean_col(mlp, "mean_error", "cartesian_error", "before_mean_error"),
            mean_max_error=mean_col(mlp, "max_error", "before_max_error"),
            worst_max_error=max_col(mlp, "max_error", "before_max_error"),
            mean_solve_time_sec=mean_col(mlp, "solve_time_sec"),
            total_time_sec=None,
            notes="Standalone prediction; fast but not reliable on held-out paths.",
        )

    cold = read_optional(args.cold_ik_csv)
    if cold is not None:
        accepted_df = cold[cold["accepted"].astype(bool)] if "accepted" in cold.columns else cold
        cold_total_sec = parse_time_to_seconds(args.cold_total_time)
        add_row(
            rows,
            method="Cold IK",
            evaluated=len(cold),
            accepted=count_accepted(cold),
            mean_path_error=mean_col(accepted_df, "path_error"),
            mean_cartesian_error=mean_col(accepted_df, "mean_error"),
            mean_max_error=mean_col(accepted_df, "max_error"),
            worst_max_error=max_col(accepted_df, "max_error"),
            mean_solve_time_sec=(cold_total_sec / max(count_accepted(cold), 1)) if cold_total_sec else None,
            total_time_sec=cold_total_sec,
            notes="Reliable teacher baseline; timed over full raw test attempts.",
        )

    for label, csv_path, note in [
        (
            "MLP + IK, smooth=0.01",
            args.refine_001_csv,
            "Fast first-stage refinement; good mean error but local spike cases remain.",
        ),
        (
            "MLP + IK, smooth=0.001",
            args.refine_0001_csv,
            "Best current full-test accuracy; slower but suppresses local spikes.",
        ),
    ]:
        df = read_optional(csv_path)
        if df is None:
            continue
        mean_solve = mean_col(df, "solve_time_sec")
        total = mean_solve * len(df) if mean_solve is not None else None
        add_row(
            rows,
            method=label,
            evaluated=len(df),
            accepted=count_accepted(df),
            mean_path_error=mean_col(df, "after_path_error"),
            mean_cartesian_error=mean_col(df, "after_mean_error"),
            mean_max_error=mean_col(df, "after_max_error"),
            worst_max_error=max_col(df, "after_max_error"),
            mean_solve_time_sec=mean_solve,
            total_time_sec=total,
            notes=note,
        )

    if args.adaptive_csv is not None:
        adaptive = read_optional(args.adaptive_csv)
        if adaptive is not None:
            mean_solve = mean_col(adaptive, "solve_time_sec")
            total = mean_solve * len(adaptive) if mean_solve is not None else None
            add_row(
                rows,
                method="Adaptive MLP + IK",
                evaluated=len(adaptive),
                accepted=count_accepted(adaptive),
                mean_path_error=mean_col(adaptive, "after_path_error"),
                mean_cartesian_error=mean_col(adaptive, "after_mean_error"),
                mean_max_error=mean_col(adaptive, "after_max_error"),
                worst_max_error=max_col(adaptive, "after_max_error"),
                mean_solve_time_sec=mean_solve,
                total_time_sec=total,
                notes="Two-stage method: fast refinement first, low-smoothness rerun only for spike cases.",
            )

    table = pd.DataFrame(rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_csv, index=False)

    display_cols = [
        "method",
        "accepted",
        "evaluated",
        "mean_path_error",
        "mean_cartesian_error_m",
        "mean_max_error_m",
        "worst_max_error_m",
        "mean_solve_time_sec",
        "total_time_readable",
        "notes",
    ]
    md = dataframe_to_markdown(table[display_cols], floatfmt=".6g")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(md + "\n", encoding="utf-8")

    print(md)
    print(f"\nSaved CSV: {args.output_csv}")
    print(f"Saved Markdown: {args.output_md}")


if __name__ == "__main__":
    main()
