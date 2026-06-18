#!/usr/bin/env python3
"""
Predict a joint trajectory for one desired Cartesian path using a trained
time-conditioned MLP.

Input desired path CSV:
    t,x,y,z

Output predicted trajectory CSV:
    t,q1,q2,q3,q4,q5,q6

Example:
python predict_time_conditioned_mlp.py \
  --model data/synthetic_paths_train_2000/time_conditioned_mlp.pt \
  --desired_path data/synthetic_paths_test/path_001/desired_path.csv \
  --output_csv data/synthetic_paths_test/path_001/time_conditioned_pred_q.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


class TimeConditionedMLP(nn.Module):
    def __init__(
        self,
        input_dim: int = 4,
        output_dim: int = 6,
        hidden_dim: int = 256,
        num_layers: int = 4,
    ) -> None:
        super().__init__()

        if num_layers < 2:
            raise ValueError("--num_layers must be at least 2")

        layers = []
        in_dim = input_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def load_model(model_path: Path, device: torch.device):
    checkpoint = torch.load(model_path, map_location=device)

    model = TimeConditionedMLP(
        input_dim=int(checkpoint.get("input_dim", 4)),
        output_dim=int(checkpoint.get("output_dim", 6)),
        hidden_dim=int(checkpoint["hidden_dim"]),
        num_layers=int(checkpoint["num_layers"]),
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def read_desired_path(path: Path) -> np.ndarray:
    df = pd.read_csv(path)
    required = ["t", "x", "y", "z"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"{path} is missing columns {missing}. Found: {list(df.columns)}")
    return df[required].to_numpy(dtype=np.float32)


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

    model, ckpt = load_model(args.model, device)

    desired = read_desired_path(args.desired_path)  # columns: t,x,y,z
    t = desired[:, 0:1]
    xyz = desired[:, 1:4]
    x_raw = np.concatenate([xyz, t], axis=1).astype(np.float32)  # [x,y,z,t]

    x_mean = ckpt["x_mean"].astype(np.float32)
    x_std = ckpt["x_std"].astype(np.float32)
    y_mean = ckpt["y_mean"].astype(np.float32)
    y_std = ckpt["y_std"].astype(np.float32)

    x = (x_raw - x_mean) / x_std

    with torch.no_grad():
        pred_norm = model(torch.from_numpy(x).to(device)).cpu().numpy()

    q = pred_norm * y_std + y_mean

    out = pd.DataFrame(
        np.concatenate([t, q], axis=1),
        columns=["t", "q1", "q2", "q3", "q4", "q5", "q6"],
    )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)
    print(f"Saved predicted trajectory: {args.output_csv}")


if __name__ == "__main__":
    main()
