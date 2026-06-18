#!/usr/bin/env python3
"""
Evaluate a trained GRU sequence baseline over path folders.

For each path_XXX folder:
    desired_path.csv
        ↓
    predict sequence_gru_pred_q.csv
        ↓
    call score_trajectory.py
        ↓
    save FK path and collect scores

Example:
    python evaluate_sequence_gru.py \
      --model data/synthetic_paths_train_2000/sequence_gru.pt \
      --dataset_dir data/synthetic_paths_test \
      --score_script score_trajectory.py \
      --output_csv data/synthetic_paths_test/sequence_gru_eval_train2000.csv
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


class SequenceGRU(nn.Module):
    def __init__(
        self,
        input_dim: int = 4,
        output_dim: int = 6,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.0,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()

        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")

        gru_dropout = dropout if num_layers > 1 else 0.0

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=gru_dropout,
            bidirectional=bidirectional,
        )

        gru_out_dim = hidden_dim * (2 if bidirectional else 1)

        self.head = nn.Sequential(
            nn.Linear(gru_out_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.gru(x)
        y = self.head(h)
        return y


def load_model(model_path: Path, device: torch.device) -> Tuple[SequenceGRU, Dict[str, Any]]:
    checkpoint = torch.load(model_path, map_location=device)

    model = SequenceGRU(
        input_dim=int(checkpoint.get("input_dim", 4)),
        output_dim=int(checkpoint.get("output_dim", 6)),
        hidden_dim=int(checkpoint["hidden_dim"]),
        num_layers=int(checkpoint["num_layers"]),
        dropout=float(checkpoint.get("dropout", 0.0)),
        bidirectional=bool(checkpoint.get("bidirectional", True)),
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def predict_one_path(
    model: SequenceGRU,
    checkpoint: Dict[str, Any],
    desired_path_csv: Path,
    output_q_csv: Path,
    device: torch.device,
) -> None:
    df = pd.read_csv(desired_path_csv)
    required = ["t", "x", "y", "z"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"{desired_path_csv} is missing columns {missing}. Found: {list(df.columns)}")

    time = df[["t"]].to_numpy(dtype=np.float32)          # (T,1)
    xyz = df[["x", "y", "z"]].to_numpy(dtype=np.float32) # (T,3)

    x_raw = np.concatenate([xyz, time], axis=1).astype(np.float32)  # (T,4)

    x_mean = checkpoint["x_mean"].astype(np.float32)  # (1,1,4)
    x_std = checkpoint["x_std"].astype(np.float32)
    y_mean = checkpoint["y_mean"].astype(np.float32)  # (1,1,6)
    y_std = checkpoint["y_std"].astype(np.float32)

    x = ((x_raw[None, :, :] - x_mean) / x_std).astype(np.float32)

    with torch.no_grad():
        pred_norm = model(torch.from_numpy(x).to(device)).cpu().numpy()

    q = pred_norm * y_std + y_mean
    q = q[0].astype(np.float32)

    out = pd.DataFrame(
        np.concatenate([time, q], axis=1),
        columns=["t", "q1", "q2", "q3", "q4", "q5", "q6"],
    )

    output_q_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_q_csv, index=False)


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


def score_prediction(
    score_script: Path,
    candidate_csv: Path,
    desired_path_csv: Path,
    ee_csv: Path,
    python_executable: str,
) -> Dict[str, Any]:
    # This matches your local score_trajectory.py argument names.
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

    parsed_scores = parse_score_output(proc.stdout)

    parsed: Dict[str, Any] = {
        **parsed_scores,
        "score_stdout": proc.stdout.strip(),
    }

    return parsed


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset_dir", type=Path, required=True)
    parser.add_argument("--score_script", type=Path, default=Path("score_trajectory.py"))
    parser.add_argument("--output_csv", type=Path, required=True)
    parser.add_argument("--num_paths", type=int, default=None)

    parser.add_argument("--baseline_path_error", type=float, default=5.16256127e-05)
    parser.add_argument("--pred_name", type=str, default="sequence_gru_pred_q.csv")
    parser.add_argument("--ee_name", type=str, default="sequence_gru_pred_ee.csv")

    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])

    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model, checkpoint = load_model(args.model, device)

    path_dirs = sorted([p for p in args.dataset_dir.glob("path_*") if p.is_dir()])
    if args.num_paths is not None:
        path_dirs = path_dirs[: args.num_paths]

    if not path_dirs:
        raise FileNotFoundError(f"No path_* folders found in {args.dataset_dir}")

    rows: List[Dict[str, Any]] = []

    for path_dir in path_dirs:
        desired_csv = path_dir / "desired_path.csv"
        if not desired_csv.exists():
            print(f"Skipping {path_dir.name}: missing desired_path.csv")
            continue

        pred_q_csv = path_dir / args.pred_name
        pred_ee_csv = path_dir / args.ee_name

        print(f"Evaluating {path_dir.name}")

        predict_one_path(
            model=model,
            checkpoint=checkpoint,
            desired_path_csv=desired_csv,
            output_q_csv=pred_q_csv,
            device=device,
        )

        score = score_prediction(
            score_script=args.score_script,
            candidate_csv=pred_q_csv,
            desired_path_csv=desired_csv,
            ee_csv=pred_ee_csv,
            python_executable=sys.executable,
        )

        row: Dict[str, Any] = {
            "path_id": path_dir.name,
            "pred_q_csv": str(pred_q_csv),
            "pred_ee_csv": str(pred_ee_csv),
            **score,
        }

        rows.append(row)

    df = pd.DataFrame(rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)

    print(f"\nSaved evaluation: {args.output_csv}")

    if "path_error" in df.columns and df["path_error"].notna().any():
        mean_path_error = float(df["path_error"].mean())
        rms_cartesian_error = float(np.sqrt(mean_path_error))
        print(f"mean path_error: {mean_path_error:.8e}")
        print(f"RMS Cartesian error: {rms_cartesian_error:.6f} m")
        print(f"time-conditioned MLP baseline path_error: {args.baseline_path_error:.8e}")

        if mean_path_error < args.baseline_path_error:
            print("Result: improved over time-conditioned MLP baseline.")
        else:
            print("Result: did not improve over time-conditioned MLP baseline.")
            print("Interpretation: GRU sequence context did not improve this dataset/config.")
    else:
        print(
            "\nWARNING: Could not parse path_error from score_trajectory.py output. "
            "Check the score_stdout column in the output CSV."
        )


if __name__ == "__main__":
    main()
