#!/usr/bin/env python3
"""Evaluate a path-conditioned MLP over Cartesian path folders."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch

from predict_path_conditioned_mlp import load_model, predict_q, read_desired_path, save_q_csv


def parse_score_output(stdout: str) -> Dict[str, Optional[float]]:
    results: Dict[str, Optional[float]] = {
        "total_score": None,
        "path_error": None,
        "smoothness_cost": None,
    }
    patterns = {
        "total_score": r"total[_ ]score\s*[:=]\s*([-+0-9.eE]+)",
        "path_error": r"path[_ ]error\s*[:=]\s*([-+0-9.eE]+)",
        "smoothness_cost": r"smoothness[_ ]cost\s*[:=]\s*([-+0-9.eE]+)",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, stdout, flags=re.IGNORECASE)
        if match:
            results[key] = float(match.group(1))
    return results


def score_prediction(
    score_script: Path,
    candidate_csv: Path,
    desired_path_csv: Path,
    ee_csv: Path,
    python_executable: str,
) -> Dict[str, Any]:
    cmd = [
        python_executable,
        str(score_script),
        "--q_csv",
        str(candidate_csv),
        "--path_csv",
        str(desired_path_csv),
        "--save_ee_csv",
        str(ee_csv),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            "score_trajectory.py failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"STDOUT:\n{proc.stdout}\n"
            f"STDERR:\n{proc.stderr}"
        )
    return {
        **parse_score_output(proc.stdout),
        "score_stdout": proc.stdout.strip(),
    }


def path_dirs_with_desired_path(dataset_dir: Path) -> List[Path]:
    return sorted(
        p for p in dataset_dir.iterdir()
        if p.is_dir() and (p / "desired_path.csv").exists()
    )


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate path-conditioned MLP with FK scoring.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset_dir", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path, required=True)
    parser.add_argument("--score_script", type=Path, default=Path("score_trajectory.py"))
    parser.add_argument("--num_paths", type=int, default=None)
    parser.add_argument("--pred_name", type=str, default="path_conditioned_pred_q.csv")
    parser.add_argument("--ee_name", type=str, default="path_conditioned_pred_ee.csv")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model, ckpt = load_model(args.model, device)

    path_dirs = path_dirs_with_desired_path(args.dataset_dir)
    if args.num_paths is not None:
        path_dirs = path_dirs[: args.num_paths]
    if not path_dirs:
        raise FileNotFoundError(f"No path folders with desired_path.csv found in {args.dataset_dir}")

    rows: List[Dict[str, Any]] = []
    for path_dir in path_dirs:
        desired_csv = path_dir / "desired_path.csv"
        pred_q_csv = path_dir / args.pred_name
        pred_ee_csv = path_dir / args.ee_name

        print(f"Evaluating {path_dir.name}")
        times, desired_path = read_desired_path(desired_csv)
        q = predict_q(model, ckpt, times, desired_path, device)
        save_q_csv(pred_q_csv, times, q)

        score = score_prediction(
            score_script=args.score_script,
            candidate_csv=pred_q_csv,
            desired_path_csv=desired_csv,
            ee_csv=pred_ee_csv,
            python_executable=sys.executable,
        )
        rows.append(
            {
                "path_id": path_dir.name,
                "pred_q_csv": str(pred_q_csv),
                "pred_ee_csv": str(pred_ee_csv),
                **score,
            }
        )

    df = pd.DataFrame(rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)

    print(f"\nSaved evaluation: {args.output_csv}")
    if "path_error" in df.columns and df["path_error"].notna().any():
        mean_path_error = float(df["path_error"].mean())
        rms_cartesian_error = float(np.sqrt(mean_path_error))
        print(f"mean path_error: {mean_path_error:.8e}")
        print(f"RMS Cartesian error: {rms_cartesian_error:.6f} m")
    else:
        print("WARNING: Could not parse path_error from score_trajectory.py output.")


if __name__ == "__main__":
    main()
