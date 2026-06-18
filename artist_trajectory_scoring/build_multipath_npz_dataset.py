import argparse
from pathlib import Path

import numpy as np
import pandas as pd


Q_COLS = ["q1", "q2", "q3", "q4", "q5", "q6"]
PATH_COLS = ["x", "y", "z"]


def main():
    parser = argparse.ArgumentParser(
        description="Pack synthetic multi-path CSV dataset into one NPZ file."
    )

    parser.add_argument(
        "--dataset_dir",
        required=True,
        help="Directory containing path_XXX folders and summary.csv.",
    )

    parser.add_argument(
        "--output_npz",
        default=None,
        help="Output NPZ path. Default: dataset_dir/multipath_episodes.npz",
    )

    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    summary_csv = dataset_dir / "summary.csv"

    if not summary_csv.exists():
        raise FileNotFoundError(f"Missing summary file: {summary_csv}")

    output_npz = Path(args.output_npz) if args.output_npz else dataset_dir / "multipath_episodes.npz"

    summary_df = pd.read_csv(summary_csv)

    required_summary_cols = ["path_id", "desired_path_csv", "expert_q_csv"]
    for col in required_summary_cols:
        if col not in summary_df.columns:
            raise ValueError(f"summary.csv missing required column: {col}")

    desired_paths = []
    actions = []
    times = []
    path_ids = []

    for row in summary_df.itertuples(index=False):
        path_id = row.path_id
        desired_path_file = Path(row.desired_path_csv)
        expert_q_file = Path(row.expert_q_csv)

        if not desired_path_file.exists():
            raise FileNotFoundError(f"Missing desired path: {desired_path_file}")

        if not expert_q_file.exists():
            raise FileNotFoundError(f"Missing expert q: {expert_q_file}")

        desired_df = pd.read_csv(desired_path_file)
        q_df = pd.read_csv(expert_q_file)

        for col in ["t"] + PATH_COLS:
            if col not in desired_df.columns:
                raise ValueError(f"{desired_path_file} missing column: {col}")

        for col in ["t"] + Q_COLS:
            if col not in q_df.columns:
                raise ValueError(f"{expert_q_file} missing column: {col}")

        if len(desired_df) != len(q_df):
            raise ValueError(
                f"Timestep mismatch in {path_id}: "
                f"desired={len(desired_df)}, q={len(q_df)}"
            )

        desired_paths.append(desired_df[PATH_COLS].to_numpy(dtype=np.float32))
        actions.append(q_df[Q_COLS].to_numpy(dtype=np.float32))
        times.append(desired_df["t"].to_numpy(dtype=np.float32))
        path_ids.append(path_id)

    desired_paths = np.stack(desired_paths, axis=0)  # (N, T, 3)
    actions = np.stack(actions, axis=0)              # (N, T, 6)
    times = np.stack(times, axis=0)                  # (N, T)
    path_ids = np.asarray(path_ids)

    np.savez_compressed(
        output_npz,
        desired_paths=desired_paths,
        actions=actions,
        times=times,
        path_ids=path_ids,
    )

    print(f"Saved multi-path NPZ dataset to: {output_npz}")
    print()
    print("Dataset arrays:")
    print(f"  desired_paths: {desired_paths.shape}")
    print(f"  actions:       {actions.shape}")
    print(f"  times:         {times.shape}")
    print(f"  path_ids:      {path_ids.shape}")


if __name__ == "__main__":
    main()