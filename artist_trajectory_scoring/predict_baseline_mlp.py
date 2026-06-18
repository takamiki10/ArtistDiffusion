import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn


class BaselineMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 256):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def main():
    parser = argparse.ArgumentParser(
        description="Run baseline MLP inference and save predicted joint trajectory CSV."
    )

    parser.add_argument(
        "--npz",
        required=True,
        help="Path to episodes.npz.",
    )

    parser.add_argument(
        "--model",
        required=True,
        help="Path to trained baseline_mlp.pt.",
    )

    parser.add_argument(
        "--output_csv",
        required=True,
        help="Output predicted joint trajectory CSV.",
    )

    args = parser.parse_args()

    npz_path = Path(args.npz)
    model_path = Path(args.model)
    output_csv = Path(args.output_csv)

    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ file does not exist: {npz_path}")

    if not model_path.exists():
        raise FileNotFoundError(f"Model file does not exist: {model_path}")

    data = np.load(npz_path, allow_pickle=True)

    desired_path = data["desired_path"].astype(np.float32)  # (T, 3)
    time = data["time"].astype(np.float32)                  # (T,)

    checkpoint = torch.load(model_path, map_location="cpu")

    input_dim = checkpoint["input_dim"]
    output_dim = checkpoint["output_dim"]
    hidden_dim = checkpoint.get("hidden_dim", 256)

    model = BaselineMLP(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=hidden_dim,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    condition = torch.from_numpy(desired_path.reshape(1, -1))

    with torch.no_grad():
        pred_flat = model(condition).cpu().numpy()[0]

    num_steps = desired_path.shape[0]
    num_joints = output_dim // num_steps

    pred_q = pred_flat.reshape(num_steps, num_joints)

    if num_joints != 6:
        raise ValueError(f"Expected 6 joints, got {num_joints}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)

    pred_df = pd.DataFrame({
        "t": time,
        "q1": pred_q[:, 0],
        "q2": pred_q[:, 1],
        "q3": pred_q[:, 2],
        "q4": pred_q[:, 3],
        "q5": pred_q[:, 4],
        "q6": pred_q[:, 5],
    })

    pred_df.to_csv(output_csv, index=False)

    print(f"Saved predicted trajectory to: {output_csv}")
    print(f"Predicted trajectory shape: {pred_q.shape}")


if __name__ == "__main__":
    main()