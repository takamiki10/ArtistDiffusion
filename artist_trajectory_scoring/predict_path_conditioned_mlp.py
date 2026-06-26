#!/usr/bin/env python3
"""Predict q trajectory with a trained path-conditioned MLP."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


class PathConditionedMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 6,
        hidden_dim: int = 512,
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


def load_model(model_path: Path, device: torch.device) -> Tuple[PathConditionedMLP, dict]:
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model = PathConditionedMLP(
        input_dim=int(checkpoint["input_dim"]),
        output_dim=int(checkpoint.get("output_dim", 6)),
        hidden_dim=int(checkpoint["hidden_dim"]),
        num_layers=int(checkpoint["num_layers"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint


def read_desired_path(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    required = ["t", "x", "y", "z"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns {missing}. Found: {list(df.columns)}")
    return (
        df["t"].to_numpy(dtype=np.float32),
        df[["x", "y", "z"]].to_numpy(dtype=np.float32),
    )


def desired_from_npz(npz_path: Path, index: int) -> Tuple[np.ndarray, np.ndarray, str]:
    data = np.load(npz_path, allow_pickle=True)
    desired_paths = data["desired_paths"].astype(np.float32)
    times = data["times"].astype(np.float32)
    path_ids = data["path_ids"] if "path_ids" in data.files else np.arange(len(desired_paths))
    if index < 0 or index >= len(desired_paths):
        raise IndexError(f"--index must be in [0, {len(desired_paths) - 1}]")
    return times[index], desired_paths[index], str(path_ids[index])


def make_features(desired_path: np.ndarray, times: np.ndarray, ckpt: dict) -> np.ndarray:
    num_steps = int(ckpt["num_steps"])
    if desired_path.shape != (num_steps, 3):
        raise ValueError(
            f"Desired path shape {desired_path.shape} does not match model T={num_steps}. "
            "Regenerate or retrain with matching timesteps."
        )

    path_flat = desired_path.reshape(1, -1)
    path_context = np.repeat(path_flat, num_steps, axis=0)
    features = [path_context, times.reshape(-1, 1).astype(np.float32)]
    if bool(ckpt.get("include_current_point", True)):
        features.append(desired_path.astype(np.float32))
    return np.concatenate(features, axis=1).astype(np.float32)


def predict_q(
    model: PathConditionedMLP,
    ckpt: dict,
    times: np.ndarray,
    desired_path: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    x_raw = make_features(desired_path, times, ckpt)
    x = (x_raw - ckpt["x_mean"].astype(np.float32)) / ckpt["x_std"].astype(np.float32)

    with torch.no_grad():
        pred_norm = model(torch.from_numpy(x).to(device)).cpu().numpy()

    return pred_norm * ckpt["y_std"].astype(np.float32) + ckpt["y_mean"].astype(np.float32)


def save_q_csv(path: Path, times: np.ndarray, q: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        np.concatenate([times.reshape(-1, 1), q], axis=1),
        columns=["t", "q1", "q2", "q3", "q4", "q5", "q6"],
    )
    df.to_csv(path, index=False)


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict q CSV with path-conditioned MLP.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output_csv", type=Path, required=True)
    parser.add_argument("--desired_path", type=Path, default=None)
    parser.add_argument("--npz", type=Path, default=None)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    if (args.desired_path is None) == (args.npz is None):
        raise ValueError("Pass exactly one of --desired_path or --npz")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model, ckpt = load_model(args.model, device)

    path_id: Optional[str] = None
    if args.desired_path is not None:
        times, desired_path = read_desired_path(args.desired_path)
        path_id = args.desired_path.parent.name
    else:
        times, desired_path, path_id = desired_from_npz(args.npz, args.index)

    q = predict_q(model, ckpt, times, desired_path, device)
    save_q_csv(args.output_csv, times, q)

    print(f"Path ID: {path_id}")
    print(f"Saved predicted trajectory to: {args.output_csv}")
    print(f"Predicted q shape: {q.shape}")


if __name__ == "__main__":
    main()
