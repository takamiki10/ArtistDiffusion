import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn


class BaselineMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 512):
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
        description="Predict one trajectory from a trained multi-path baseline MLP."
    )

    parser.add_argument("--npz", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--output_csv", required=True)

    args = parser.parse_args()

    data = np.load(args.npz, allow_pickle=True)

    desired_paths = data["desired_paths"].astype(np.float32)
    times = data["times"].astype(np.float32)

    if "path_ids" in data.files:
        path_ids = data["path_ids"]
    else:
        path_ids = np.array([f"path_{i:03d}" for i in range(len(desired_paths))])

    if args.index < 0 or args.index >= len(desired_paths):
        raise IndexError(f"--index must be between 0 and {len(desired_paths) - 1}")

    checkpoint = torch.load(args.model, map_location="cpu")

    model = BaselineMLP(
        input_dim=checkpoint["input_dim"],
        output_dim=checkpoint["output_dim"],
        hidden_dim=checkpoint.get("hidden_dim", 512),
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    desired_path = desired_paths[args.index]
    time = times[args.index]

    condition = torch.from_numpy(desired_path.reshape(1, -1))

    with torch.no_grad():
        pred_flat = model(condition).cpu().numpy()[0]

    num_steps = checkpoint["num_steps"]
    num_joints = checkpoint["num_joints"]

    pred_q = pred_flat.reshape(num_steps, num_joints)

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame({
        "t": time,
        "q1": pred_q[:, 0],
        "q2": pred_q[:, 1],
        "q3": pred_q[:, 2],
        "q4": pred_q[:, 3],
        "q5": pred_q[:, 4],
        "q6": pred_q[:, 5],
    })

    df.to_csv(output_csv, index=False)

    print(f"Predicted path index: {args.index}")
    print(f"Path ID: {path_ids[args.index]}")
    print(f"Saved predicted trajectory to: {output_csv}")
    print(f"Predicted q shape: {pred_q.shape}")


if __name__ == "__main__":
    main()