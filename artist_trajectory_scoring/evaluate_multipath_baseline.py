import argparse
import subprocess
from pathlib import Path

import pandas as pd


def run_command(cmd):
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )
    return result.stdout


def parse_score_output(output: str):
    values = {}

    for line in output.splitlines():
        if ":" not in line:
            continue

        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()

        if key in ["total_score", "path_error", "smoothness_cost"]:
            values[key] = float(value)

    return values


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate multi-path baseline MLP over many synthetic paths."
    )

    parser.add_argument("--npz", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--num_paths", type=int, default=20)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--w_path", type=float, default=1.0)
    parser.add_argument("--w_smooth", type=float, default=0.01)

    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    results = []

    for idx in range(args.num_paths):
        path_id = f"path_{idx + 1:03d}"
        path_dir = dataset_dir / path_id

        predicted_q = path_dir / "predicted_mlp_q.csv"
        predicted_fk = path_dir / "predicted_mlp_fk.csv"
        desired_path = path_dir / "desired_path.csv"

        print(f"Evaluating {path_id}")

        predict_cmd = [
            "python",
            "predict_multipath_baseline_mlp.py",
            "--npz",
            args.npz,
            "--model",
            args.model,
            "--index",
            str(idx),
            "--output_csv",
            str(predicted_q),
        ]

        run_command(predict_cmd)

        score_cmd = [
            "python",
            "score_trajectory.py",
            "--q_csv",
            str(predicted_q),
            "--path_csv",
            str(desired_path),
            "--w_path",
            str(args.w_path),
            "--w_smooth",
            str(args.w_smooth),
            "--save_ee_csv",
            str(predicted_fk),
        ]

        score_output = run_command(score_cmd)
        scores = parse_score_output(score_output)

        results.append({
            "path_id": path_id,
            "index": idx,
            "predicted_q": str(predicted_q),
            "predicted_fk": str(predicted_fk),
            "total_score": scores["total_score"],
            "path_error": scores["path_error"],
            "smoothness_cost": scores["smoothness_cost"],
        })

    results_df = pd.DataFrame(results)
    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_csv, index=False)

    print()
    print(f"Saved evaluation results to: {output_csv}")
    print()
    print(results_df[["path_id", "total_score", "path_error", "smoothness_cost"]].to_string(index=False))
    print()
    print("Summary:")
    print(results_df[["total_score", "path_error", "smoothness_cost"]].describe().to_string())


if __name__ == "__main__":
    main()