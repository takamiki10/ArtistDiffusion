import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


class MultiPathTrajectoryDataset(Dataset):
    def __init__(self, npz_path: str):
        data = np.load(npz_path, allow_pickle=True)

        self.desired_paths = data["desired_paths"].astype(np.float32)  # (N, T, 3)
        self.actions = data["actions"].astype(np.float32)              # (N, T, 6)

        if self.desired_paths.ndim != 3:
            raise ValueError(f"desired_paths must have shape (N, T, 3), got {self.desired_paths.shape}")

        if self.actions.ndim != 3:
            raise ValueError(f"actions must have shape (N, T, 6), got {self.actions.shape}")

        if self.desired_paths.shape[0] != self.actions.shape[0]:
            raise ValueError(
                f"Sample count mismatch: desired_paths={self.desired_paths.shape}, "
                f"actions={self.actions.shape}"
            )

        if self.desired_paths.shape[1] != self.actions.shape[1]:
            raise ValueError(
                f"Timestep mismatch: desired_paths={self.desired_paths.shape}, "
                f"actions={self.actions.shape}"
            )

        if "path_ids" in data.files:
            self.path_ids = data["path_ids"]
        else:
            self.path_ids = np.array([f"path_{i:03d}" for i in range(len(self.actions))])

    def __len__(self):
        return self.actions.shape[0]

    def __getitem__(self, idx):
        condition = self.desired_paths[idx].reshape(-1)  # (T*3,)
        action = self.actions[idx].reshape(-1)           # (T*6,)

        return {
            "condition": torch.from_numpy(condition),
            "action": torch.from_numpy(action),
            "path_id": self.path_ids[idx],
        }


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
        description="Train baseline MLP on multi-path artist trajectory dataset."
    )

    parser.add_argument("--npz", required=True, help="Path to multipath_episodes.npz.")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--output_model", required=True)

    args = parser.parse_args()

    dataset = MultiPathTrajectoryDataset(args.npz)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    input_dim = dataset.desired_paths.shape[1] * dataset.desired_paths.shape[2]
    output_dim = dataset.actions.shape[1] * dataset.actions.shape[2]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BaselineMLP(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    print(f"Using device: {device}")
    print(f"Dataset samples: {len(dataset)}")
    print(f"Input dim: {input_dim}")
    print(f"Output dim: {output_dim}")
    print(f"Hidden dim: {args.hidden_dim}")
    print()

    for epoch in range(1, args.epochs + 1):
        losses = []

        for batch in loader:
            condition = batch["condition"].to(device)
            action = batch["action"].to(device)

            pred = model(condition)
            loss = loss_fn(pred, action)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

        if epoch == 1 or epoch % 100 == 0:
            print(f"epoch {epoch:04d} | loss {float(np.mean(losses)):.8f}")

    output_model = Path(args.output_model)
    output_model.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": input_dim,
            "output_dim": output_dim,
            "hidden_dim": args.hidden_dim,
            "npz": args.npz,
            "num_steps": dataset.desired_paths.shape[1],
            "num_joints": dataset.actions.shape[2],
        },
        output_model,
    )

    print()
    print(f"Saved model to: {output_model}")


if __name__ == "__main__":
    main()