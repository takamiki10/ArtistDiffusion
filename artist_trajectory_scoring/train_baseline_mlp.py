import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


class ArtistTrajectoryTorchDataset(Dataset):
    def __init__(self, npz_path: str):
        data = np.load(npz_path, allow_pickle=True)

        self.desired_path = data["desired_path"].astype(np.float32)  # (T, 3)
        self.actions = data["actions"].astype(np.float32)            # (N, T, 6)

        self.num_samples = self.actions.shape[0]
        self.T = self.actions.shape[1]

        if self.desired_path.shape[0] != self.T:
            raise ValueError(
                f"Timestep mismatch: desired_path={self.desired_path.shape}, "
                f"actions={self.actions.shape}"
            )

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        condition = self.desired_path.reshape(-1)      # (T*3,)
        action = self.actions[idx].reshape(-1)         # (T*6,)

        return {
            "condition": torch.from_numpy(condition),
            "action": torch.from_numpy(action),
        }


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
        description="Train a tiny baseline MLP from desired path to joint trajectory."
    )

    parser.add_argument(
        "--npz",
        required=True,
        help="Path to episodes.npz.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=1000,
        help="Number of training epochs.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size.",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate.",
    )

    parser.add_argument(
        "--output_model",
        default="baseline_mlp.pt",
        help="Path to save trained model checkpoint.",
    )

    args = parser.parse_args()

    dataset = ArtistTrajectoryTorchDataset(args.npz)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    input_dim = dataset.desired_path.size
    output_dim = dataset.actions.shape[1] * dataset.actions.shape[2]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = BaselineMLP(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_dim=256,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    print(f"Using device: {device}")
    print(f"Dataset samples: {len(dataset)}")
    print(f"Input dim: {input_dim}")
    print(f"Output dim: {output_dim}")
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
            mean_loss = float(np.mean(losses))
            print(f"epoch {epoch:04d} | loss {mean_loss:.8f}")

    output_model = Path(args.output_model)
    output_model.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": input_dim,
            "output_dim": output_dim,
            "hidden_dim": 256,
            "npz": args.npz,
        },
        output_model,
    )

    print()
    print(f"Saved model to: {output_model}")


if __name__ == "__main__":
    main()