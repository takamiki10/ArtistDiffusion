#!/usr/bin/env python3
"""
Predict one joint trajectory using a trained GRU sequence baseline.

Input:
    desired_path.csv with columns:
        t,x,y,z

Output:
    predicted joint trajectory CSV with columns:
        t,q1,q2,q3,q4,q5,q6

Example:
    python predict_sequence_gru.py \
      --model data/synthetic_paths_train_2000/sequence_gru.pt \
      --desired_path data/synthetic_paths_test/path_001/desired_path.csv \
      --output_csv data/synthetic_paths_test/path_001/sequence_gru_pred_q.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Tuple

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


def load_desired_path(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    required = ["t", "x", "y", "z"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"{path} is missing columns {missing}. Found: {list(df.columns)}")

    time = df[["t"]].to_numpy(dtype=np.float32)          # (T,1)
    xyz = df[["x", "y", "z"]].to_numpy(dtype=np.float32) # (T,3)
    return time, xyz


def predict_one(
    model: SequenceGRU,
    checkpoint: Dict[str, Any],
    time: np.ndarray,
    xyz: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """
    Returns:
        q trajectory, shape (T, 6)
    """
    x_raw = np.concatenate([xyz, time], axis=1).astype(np.float32)  # (T,4)

    x_mean = checkpoint["x_mean"].astype(np.float32)  # (1,1,4)
    x_std = checkpoint["x_std"].astype(np.float32)
    y_mean = checkpoint["y_mean"].astype(np.float32)  # (1,1,6)
    y_std = checkpoint["y_std"].astype(np.float32)

    expected_input_dim = int(checkpoint.get("input_dim", 4))
    if x_raw.shape[1] != expected_input_dim:
        raise ValueError(f"Input dim mismatch: got {x_raw.shape[1]}, expected {expected_input_dim}")

    x = ((x_raw[None, :, :] - x_mean) / x_std).astype(np.float32)  # (1,T,4)

    with torch.no_grad():
        pred_norm = model(torch.from_numpy(x).to(device)).cpu().numpy()  # (1,T,6)

    q = pred_norm * y_std + y_mean
    return q[0].astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--desired_path", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path, required=True)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model, checkpoint = load_model(args.model, device)
    time, xyz = load_desired_path(args.desired_path)

    q = predict_one(
        model=model,
        checkpoint=checkpoint,
        time=time,
        xyz=xyz,
        device=device,
    )

    out = pd.DataFrame(
        np.concatenate([time, q], axis=1),
        columns=["t", "q1", "q2", "q3", "q4", "q5", "q6"],
    )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)

    print(f"Saved predicted trajectory: {args.output_csv}")


if __name__ == "__main__":
    main()
