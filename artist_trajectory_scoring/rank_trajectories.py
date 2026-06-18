import argparse
from pathlib import Path

import pandas as pd

from score_trajectory import (
    rokae_forward_kinematics,
    compute_path_error,
    compute_smoothness_cost,
)


def score_one_candidate(q_csv_path: Path, desired_path: pd.DataFrame, w_path: float, w_smooth: float):
    q_df = pd.read_csv(q_csv_path)

    required_q_cols = ["q1", "q2", "q3", "q4", "q5", "q6"]
    required_path_cols = ["x", "y", "z"]

    for col in required_q_cols:
        if col not in q_df.columns:
            raise ValueError(f"{q_csv_path} is missing required column: {col}")

    for col in required_path_cols:
        if col not in desired_path.columns:
            raise ValueError(f"desired path is missing required column: {col}")

    q = q_df[required_q_cols].to_numpy()
    p_des = desired_path[required_path_cols].to_numpy()

    if len(q) != len(p_des):
        raise ValueError(
            f"Timestep mismatch for {q_csv_path}: "
            f"candidate has {len(q)} rows, desired path has {len(p_des)} rows."
        )

    p_ee = rokae_forward_kinematics(q)

    path_error = compute_path_error(p_ee, p_des)
    smoothness_cost = compute_smoothness_cost(q)
    total_score = w_path * path_error + w_smooth * smoothness_cost

    return {
        "candidate_file": q_csv_path.name,
        "total_score": total_score,
        "path_error": path_error,
        "smoothness_cost": smoothness_cost,
        "num_steps": q.shape[0],
        "num_joints": q.shape[1],
    }


def main():
    parser = argparse.ArgumentParser(
        description="Rank candidate joint trajectories using FK-based trajectory scoring."
    )

    parser.add_argument(
        "--candidate_dir",
        required=True,
        help="Directory containing candidate joint trajectory CSV files.",
    )

    parser.add_argument(
        "--path_csv",
        required=True,
        help="Desired Cartesian path CSV with columns t,x,y,z.",
    )

    parser.add_argument(
        "--output_csv",
        default="ranking.csv",
        help="Output ranking CSV path.",
    )

    parser.add_argument(
        "--w_path",
        type=float,
        default=1.0,
        help="Weight for Cartesian path tracking error.",
    )

    parser.add_argument(
        "--w_smooth",
        type=float,
        default=0.01,
        help="Weight for joint acceleration smoothness cost.",
    )

    args = parser.parse_args()

    candidate_dir = Path(args.candidate_dir)
    if not candidate_dir.exists():
        raise FileNotFoundError(f"Candidate directory does not exist: {candidate_dir}")

    desired_path = pd.read_csv(args.path_csv)

    candidate_files = sorted(candidate_dir.glob("*.csv"))
    if len(candidate_files) == 0:
        raise FileNotFoundError(f"No CSV files found in candidate directory: {candidate_dir}")

    results = []

    for q_csv_path in candidate_files:
        print(f"Scoring: {q_csv_path}")

        try:
            result = score_one_candidate(
                q_csv_path=q_csv_path,
                desired_path=desired_path,
                w_path=args.w_path,
                w_smooth=args.w_smooth,
            )
            results.append(result)

        except Exception as e:
            print(f"  Skipped due to error: {e}")
            results.append({
                "candidate_file": q_csv_path.name,
                "total_score": float("inf"),
                "path_error": float("inf"),
                "smoothness_cost": float("inf"),
                "num_steps": None,
                "num_joints": None,
                "error": str(e),
            })

    ranking_df = pd.DataFrame(results)
    ranking_df = ranking_df.sort_values("total_score", ascending=True)

    ranking_df.to_csv(args.output_csv, index=False)

    print()
    print(f"Saved ranking to: {args.output_csv}")
    print()
    print("Top candidates:")
    print(ranking_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()