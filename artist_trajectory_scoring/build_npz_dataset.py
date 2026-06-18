import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REQUIRED_Q_COLS = ["q1", "q2", "q3", "q4", "q5", "q6"]
REQUIRED_PATH_COLS = ["x", "y", "z"]


def main():
    parser = argparse.ArgumentParser(
        description="Pack expert trajectory CSV dataset into a single NPZ file."
    )

    parser.add_argument(
        "--dataset_dir",
        required=True,
        help="Expert dataset folder created by keep_top_n.py.",
    )

    parser.add_argument(
        "--output_npz",
        default=None,
        help="Output NPZ path. Default: dataset_dir/episodes.npz",
    )

    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")

    desired_path_file = dataset_dir / "desired_path.csv"
    selected_ranking_file = dataset_dir / "selected_ranking.csv"

    if not desired_path_file.exists():
        raise FileNotFoundError(f"Missing desired path file: {desired_path_file}")

    if not selected_ranking_file.exists():
        raise FileNotFoundError(f"Missing selected ranking file: {selected_ranking_file}")

    output_npz = Path(args.output_npz) if args.output_npz else dataset_dir / "episodes.npz"

    desired_df = pd.read_csv(desired_path_file)
    selected_df = pd.read_csv(selected_ranking_file)

    for col in REQUIRED_PATH_COLS:
        if col not in desired_df.columns:
            raise ValueError(f"{desired_path_file} is missing required column: {col}")

    if "t" not in desired_df.columns:
        raise ValueError(f"{desired_path_file} is missing required column: t")

    if "copied_file" not in selected_df.columns:
        raise ValueError(f"{selected_ranking_file} is missing required column: copied_file")

    time = desired_df["t"].to_numpy(dtype=np.float32)
    desired_path = desired_df[REQUIRED_PATH_COLS].to_numpy(dtype=np.float32)

    actions_list = []
    trajectory_files = []

    for row in selected_df.itertuples(index=False):
        copied_file = getattr(row, "copied_file")
        traj_file = dataset_dir / copied_file

        if not traj_file.exists():
            raise FileNotFoundError(f"Missing trajectory file: {traj_file}")

        traj_df = pd.read_csv(traj_file)

        for col in REQUIRED_Q_COLS:
            if col not in traj_df.columns:
                raise ValueError(f"{traj_file} is missing required column: {col}")

        if len(traj_df) != len(desired_df):
            raise ValueError(
                f"Timestep mismatch for {traj_file.name}: "
                f"trajectory={len(traj_df)}, desired_path={len(desired_df)}"
            )

        q = traj_df[REQUIRED_Q_COLS].to_numpy(dtype=np.float32)

        actions_list.append(q)
        trajectory_files.append(copied_file)

    actions = np.stack(actions_list, axis=0)
    trajectory_files = np.asarray(trajectory_files)

    np.savez_compressed(
        output_npz,
        desired_path=desired_path,
        actions=actions,
        time=time,
        trajectory_files=trajectory_files,
    )

    print(f"Saved NPZ dataset to: {output_npz}")
    print()
    print("Dataset arrays:")
    print(f"  desired_path:     {desired_path.shape}")
    print(f"  actions:          {actions.shape}")
    print(f"  time:             {time.shape}")
    print(f"  trajectory_files: {trajectory_files.shape}")


if __name__ == "__main__":
    main()