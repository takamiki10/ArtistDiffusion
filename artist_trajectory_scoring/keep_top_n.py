import argparse
import shutil
from pathlib import Path

import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Copy the top-N ranked candidate trajectories into an expert dataset folder."
    )

    parser.add_argument(
        "--ranking_csv",
        required=True,
        help="Ranking CSV produced by rank_trajectories.py.",
    )

    parser.add_argument(
        "--candidate_dir",
        required=True,
        help="Directory containing the original candidate trajectory CSV files.",
    )

    parser.add_argument(
        "--output_dir",
        default="expert_dataset",
        help="Directory where selected expert trajectories will be copied.",
    )

    parser.add_argument(
        "--top_n",
        type=int,
        default=10,
        help="Number of best trajectories to keep.",
    )

    parser.add_argument(
        "--max_score",
        type=float,
        default=None,
        help="Optional maximum total_score threshold. Candidates above this are discarded.",
    )

    parser.add_argument(
        "--path_csv",
        required=True,
        help="Desired Cartesian path CSV used to score/rank the candidates.",
    )

    args = parser.parse_args()

    ranking_csv = Path(args.ranking_csv)
    candidate_dir = Path(args.candidate_dir)
    output_dir = Path(args.output_dir)
    path_csv = Path(args.path_csv)

    if not ranking_csv.exists():
        raise FileNotFoundError(f"Ranking CSV does not exist: {ranking_csv}")

    if not candidate_dir.exists():
        raise FileNotFoundError(f"Candidate directory does not exist: {candidate_dir}")
    
    if not path_csv.exists():
        raise FileNotFoundError(f"Desired path CSV does not exist: {path_csv}")

    output_dir.mkdir(parents=True, exist_ok=True)
    desired_path_dst = output_dir / "desired_path.csv"
    shutil.copy2(path_csv, desired_path_dst)
    print(f"Copied desired path -> {desired_path_dst}")

    ranking_df = pd.read_csv(ranking_csv)

    required_cols = ["candidate_file", "total_score", "path_error", "smoothness_cost"]
    for col in required_cols:
        if col not in ranking_df.columns:
            raise ValueError(f"Ranking CSV is missing required column: {col}")

    ranking_df = ranking_df.sort_values("total_score", ascending=True)

    if args.max_score is not None:
        ranking_df = ranking_df[ranking_df["total_score"] <= args.max_score]

    selected_df = ranking_df.head(args.top_n).copy()

    if len(selected_df) == 0:
        raise RuntimeError("No candidates selected. Try increasing --top_n or --max_score.")

    copied_files = []

    for rank_idx, row in enumerate(selected_df.itertuples(index=False), start=1):
        candidate_file = row.candidate_file
        src_path = candidate_dir / candidate_file

        if not src_path.exists():
            print(f"Skipping missing file: {src_path}")
            continue

        dst_name = f"rank_{rank_idx:03d}_{candidate_file}"
        dst_path = output_dir / dst_name

        shutil.copy2(src_path, dst_path)
        copied_files.append(dst_name)

        print(f"Copied rank {rank_idx}: {candidate_file} -> {dst_path}")

    selected_df.insert(0, "rank", range(1, len(selected_df) + 1))
    selected_df["copied_file"] = copied_files[:len(selected_df)]

    selected_ranking_path = output_dir / "selected_ranking.csv"
    selected_df.to_csv(selected_ranking_path, index=False)

    print()
    print(f"Saved selected ranking to: {selected_ranking_path}")
    print(f"Copied {len(copied_files)} trajectory files into: {output_dir}")


if __name__ == "__main__":
    main()