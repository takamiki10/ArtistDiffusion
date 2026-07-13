import argparse
from pathlib import Path

import pandas as pd


REQUIRED_Q_COLS = ["t", "q1", "q2", "q3", "q4", "q5", "q6"]
REQUIRED_PATH_COLS = ["t", "x", "y", "z"]


def check_columns(df: pd.DataFrame, required_cols, file_path: Path):
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"{file_path} is missing columns: {missing}")


def validate_episode(path_dir: Path) -> bool:
    desired_path_file = path_dir / "desired_path.csv"
    expert_q_file = path_dir / "expert_q.csv"

    if not desired_path_file.exists():
        print(f"[FAIL] Missing desired_path.csv: {path_dir}")
        return False

    if not expert_q_file.exists():
        print(f"[FAIL] Missing expert_q.csv: {path_dir}")
        return False

    desired_df = pd.read_csv(desired_path_file)
    expert_q_df = pd.read_csv(expert_q_file)

    check_columns(desired_df, REQUIRED_PATH_COLS, desired_path_file)
    check_columns(expert_q_df, REQUIRED_Q_COLS, expert_q_file)

    if len(desired_df) != len(expert_q_df):
        print(
            f"[FAIL] {path_dir.name}: timestep mismatch. "
            f"desired_path={len(desired_df)}, expert_q={len(expert_q_df)}"
        )
        return False

    print(f"[OK] {path_dir.name}: {len(desired_df)} timesteps")
    return True


def validate_episode_dataset(dataset_dir: Path) -> None:
    episode_dirs = sorted(
        path
        for path in dataset_dir.iterdir()
        if path.is_dir()
        and ((path / "desired_path.csv").exists() or (path / "expert_q.csv").exists())
    )

    if not episode_dirs:
        raise FileNotFoundError(
            f"No expert episode folders found in {dataset_dir}. "
            "Expected subfolders containing desired_path.csv and expert_q.csv."
        )

    print(f"Dataset directory: {dataset_dir}")
    print(f"Episode folders: {len(episode_dirs)}")
    print()

    all_ok = True
    expected_len = None

    for episode_dir in episode_dirs:
        try:
            ok = validate_episode(episode_dir)
        except ValueError as e:
            print(f"[FAIL] {e}")
            ok = False

        if ok:
            desired_len = len(pd.read_csv(episode_dir / "desired_path.csv"))
            if expected_len is None:
                expected_len = desired_len
            elif desired_len != expected_len:
                print(
                    f"[FAIL] {episode_dir.name}: timestep mismatch across episodes. "
                    f"episode={desired_len}, expected={expected_len}"
                )
                ok = False

        all_ok = all_ok and ok

    print()

    if all_ok:
        print("Dataset validation passed.")
    else:
        raise RuntimeError("Dataset validation failed. See messages above.")


def validate_flat_dataset(dataset_dir: Path) -> None:
    desired_path_file = dataset_dir / "desired_path.csv"
    selected_ranking_file = dataset_dir / "selected_ranking.csv"

    if not desired_path_file.exists():
        raise FileNotFoundError(f"Missing required file: {desired_path_file}")

    if not selected_ranking_file.exists():
        raise FileNotFoundError(f"Missing required file: {selected_ranking_file}")

    desired_df = pd.read_csv(desired_path_file)
    selected_df = pd.read_csv(selected_ranking_file)

    check_columns(desired_df, REQUIRED_PATH_COLS, desired_path_file)

    if "copied_file" not in selected_df.columns:
        raise ValueError(f"{selected_ranking_file} is missing column: copied_file")

    desired_len = len(desired_df)

    print(f"Dataset directory: {dataset_dir}")
    print(f"Desired path: {desired_path_file}")
    print(f"Desired path timesteps: {desired_len}")
    print()

    all_ok = True

    for row in selected_df.itertuples(index=False):
        copied_file = getattr(row, "copied_file")
        traj_file = dataset_dir / copied_file

        if not traj_file.exists():
            print(f"[FAIL] Missing trajectory file: {traj_file}")
            all_ok = False
            continue

        traj_df = pd.read_csv(traj_file)

        try:
            check_columns(traj_df, REQUIRED_Q_COLS, traj_file)
        except ValueError as e:
            print(f"[FAIL] {e}")
            all_ok = False
            continue

        traj_len = len(traj_df)

        if traj_len != desired_len:
            print(
                f"[FAIL] {traj_file.name}: timestep mismatch. "
                f"trajectory={traj_len}, desired_path={desired_len}"
            )
            all_ok = False
            continue

        print(f"[OK] {traj_file.name}: {traj_len} timesteps")

    print()

    if all_ok:
        print("Dataset validation passed.")
    else:
        raise RuntimeError("Dataset validation failed. See messages above.")


def main():
    parser = argparse.ArgumentParser(
        description="Validate an expert trajectory dataset folder."
    )

    parser.add_argument(
        "--dataset_dir",
        required=True,
        help="Expert dataset folder created by keep_top_n.py.",
    )

    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")

    if (dataset_dir / "selected_ranking.csv").exists():
        validate_flat_dataset(dataset_dir)
    else:
        validate_episode_dataset(dataset_dir)


if __name__ == "__main__":
    main()
