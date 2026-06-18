#!/usr/bin/env python3
"""
Evaluate the trained time-conditioned MLP on drawing-like Cartesian test paths.

This script:
    1. Finds folders under data/cartesian_test_paths
    2. Reads desired_path.csv
    3. Runs predict_time_conditioned_mlp.py
    4. Runs score_trajectory.py
    5. Saves a summary CSV

Example:
    python evaluate_cartesian_test_paths.py \
      --model data/synthetic_paths_train_2000/time_conditioned_mlp.pt \
      --dataset_dir data/cartesian_test_paths \
      --predict_script predict_time_conditioned_mlp.py \
      --score_script score_trajectory.py \
      --output_csv data/cartesian_test_paths/time_conditioned_eval.csv
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def parse_score_output(stdout: str) -> Dict[str, Optional[float]]:
    results: Dict[str, Optional[float]] = {
        "total_score": None,
        "path_error": None,
        "smoothness_cost": None,
        "mean_error": None,
        "max_error": None,
    }

    patterns = {
        "total_score": r"total[_ ]score\s*[:=]\s*([-+0-9.eE]+)",
        "path_error": r"path[_ ]error\s*[:=]\s*([-+0-9.eE]+)",
        "smoothness_cost": r"smoothness[_ ]cost\s*[:=]\s*([-+0-9.eE]+)",
        "mean_error": r"mean[_ ]error\s*[:=]\s*([-+0-9.eE]+)",
        "max_error": r"max[_ ]error\s*[:=]\s*([-+0-9.eE]+)",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, stdout, flags=re.IGNORECASE)
        if match:
            results[key] = float(match.group(1))

    return results


def run_predict(
    python_executable: str,
    predict_script: Path,
    model: Path,
    desired_path: Path,
    output_q: Path,
    device: str,
) -> None:
    cmd = [
        python_executable,
        str(predict_script),
        "--model",
        str(model),
        "--desired_path",
        str(desired_path),
        "--output_csv",
        str(output_q),
        "--device",
        device,
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True)

    if proc.returncode != 0:
        raise RuntimeError(
            "Prediction failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"STDOUT:\n{proc.stdout}\n"
            f"STDERR:\n{proc.stderr}"
        )


def run_score(
    python_executable: str,
    score_script: Path,
    q_csv: Path,
    path_csv: Path,
    ee_csv: Path,
) -> Dict[str, Any]:
    cmd = [
        python_executable,
        str(score_script),
        "--q_csv",
        str(q_csv),
        "--path_csv",
        str(path_csv),
        "--save_ee_csv",
        str(ee_csv),
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True)

    if proc.returncode != 0:
        raise RuntimeError(
            "Scoring failed.\n"
            f"Command: {' '.join(cmd)}\n"
            f"STDOUT:\n{proc.stdout}\n"
            f"STDERR:\n{proc.stderr}"
        )

    scores = parse_score_output(proc.stdout)
    return {**scores, "score_stdout": proc.stdout.strip()}


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset_dir", type=Path, default=Path("data/cartesian_test_paths"))
    parser.add_argument("--predict_script", type=Path, default=Path("predict_time_conditioned_mlp.py"))
    parser.add_argument("--score_script", type=Path, default=Path("score_trajectory.py"))
    parser.add_argument("--output_csv", type=Path, default=Path("data/cartesian_test_paths/time_conditioned_eval.csv"))
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])

    parser.add_argument("--pred_name", type=str, default="time_conditioned_pred_q.csv")
    parser.add_argument("--ee_name", type=str, default="time_conditioned_pred_ee.csv")

    args = parser.parse_args()

    path_dirs = sorted([p for p in args.dataset_dir.iterdir() if p.is_dir()])
    if not path_dirs:
        raise FileNotFoundError(f"No subfolders found in {args.dataset_dir}")

    rows: List[Dict[str, Any]] = []

    for path_dir in path_dirs:
        desired_csv = path_dir / "desired_path.csv"
        if not desired_csv.exists():
            print(f"Skipping {path_dir.name}: missing desired_path.csv")
            continue

        pred_q_csv = path_dir / args.pred_name
        pred_ee_csv = path_dir / args.ee_name

        print(f"Evaluating {path_dir.name}")

        run_predict(
            python_executable=sys.executable,
            predict_script=args.predict_script,
            model=args.model,
            desired_path=desired_csv,
            output_q=pred_q_csv,
            device=args.device,
        )

        score = run_score(
            python_executable=sys.executable,
            score_script=args.score_script,
            q_csv=pred_q_csv,
            path_csv=desired_csv,
            ee_csv=pred_ee_csv,
        )

        rows.append({
            "path_id": path_dir.name,
            "desired_path_csv": str(desired_csv),
            "pred_q_csv": str(pred_q_csv),
            "pred_ee_csv": str(pred_ee_csv),
            **score,
        })

    df = pd.DataFrame(rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)

    print(f"\nSaved evaluation: {args.output_csv}")

    if "path_error" in df.columns and df["path_error"].notna().any():
        mean_path_error = float(df["path_error"].mean())
        rms_error = float(np.sqrt(mean_path_error))

        print(f"mean path_error: {mean_path_error:.8e}")
        print(f"RMS Cartesian error: {rms_error:.6f} m")

        best = df.sort_values("path_error").iloc[0]
        worst = df.sort_values("path_error").iloc[-1]

        print("\nBest path:")
        print(f"  {best['path_id']}: path_error={float(best['path_error']):.8e}")

        print("Worst path:")
        print(f"  {worst['path_id']}: path_error={float(worst['path_error']):.8e}")
    else:
        print("\nWARNING: Could not parse path_error. Check score_stdout in the CSV.")


if __name__ == "__main__":
    main()
