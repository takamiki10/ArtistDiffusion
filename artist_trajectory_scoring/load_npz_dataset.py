import argparse
from pathlib import Path

import numpy as np


class ArtistTrajectoryDataset:
    """
    Simple loader for artist robot trajectory datasets.

    Current dataset format:
        desired_path: (T, 3)
        actions:      (N, T, 6)
        time:         (T,)
    """

    def __init__(self, npz_path: str):
        self.npz_path = Path(npz_path)

        if not self.npz_path.exists():
            raise FileNotFoundError(f"NPZ file does not exist: {self.npz_path}")

        data = np.load(self.npz_path, allow_pickle=True)

        self.desired_path = data["desired_path"].astype(np.float32)
        self.actions = data["actions"].astype(np.float32)
        self.time = data["time"].astype(np.float32)

        if "trajectory_files" in data.files:
            self.trajectory_files = data["trajectory_files"]
        else:
            self.trajectory_files = np.array([f"trajectory_{i}" for i in range(len(self.actions))])

        if self.actions.ndim != 3:
            raise ValueError(f"actions must have shape (N, T, J), got {self.actions.shape}")

        if self.desired_path.ndim != 2:
            raise ValueError(f"desired_path must have shape (T, 3), got {self.desired_path.shape}")

        if self.desired_path.shape[0] != self.actions.shape[1]:
            raise ValueError(
                f"Timestep mismatch: desired_path has {self.desired_path.shape[0]}, "
                f"actions have {self.actions.shape[1]}"
            )

    def __len__(self):
        return self.actions.shape[0]

    def __getitem__(self, idx: int):
        """
        Returns one training sample.

        condition:
            desired_path, shape (T, 3)

        action:
            joint trajectory, shape (T, 6)
        """
        return {
            "condition": self.desired_path,
            "action": self.actions[idx],
            "time": self.time,
            "trajectory_file": self.trajectory_files[idx],
        }


def main():
    parser = argparse.ArgumentParser(
        description="Load and inspect an artist trajectory NPZ dataset."
    )

    parser.add_argument(
        "--npz",
        required=True,
        help="Path to episodes.npz.",
    )

    args = parser.parse_args()

    dataset = ArtistTrajectoryDataset(args.npz)

    print(f"Loaded dataset: {args.npz}")
    print(f"Number of expert trajectories: {len(dataset)}")
    print()

    sample = dataset[0]

    print("Sample 0:")
    print(f"  condition shape: {sample['condition'].shape}")
    print(f"  action shape:    {sample['action'].shape}")
    print(f"  time shape:      {sample['time'].shape}")
    print(f"  file:            {sample['trajectory_file']}")
    print()

    print("Dataset-level shapes:")
    print(f"  desired_path: {dataset.desired_path.shape}")
    print(f"  actions:      {dataset.actions.shape}")
    print(f"  time:         {dataset.time.shape}")


if __name__ == "__main__":
    main()