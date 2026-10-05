import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(
        description="Plot Cartesian tracking error over time."
    )

    parser.add_argument(
        "--desired_csv",
        required=True,
        help="Desired Cartesian path CSV with columns t,x,y,z.",
    )

    parser.add_argument(
        "--predicted_fk_csv",
        required=True,
        help="Predicted FK Cartesian path CSV with columns t,x,y,z.",
    )

    parser.add_argument(
        "--output_png",
        required=True,
        help="Output PNG path.",
    )

    args = parser.parse_args()

    desired = pd.read_csv(args.desired_csv)
    predicted = pd.read_csv(args.predicted_fk_csv)

    required_cols = ["t", "x", "y", "z"]

    for col in required_cols:
        if col not in desired.columns:
            raise ValueError(f"Desired CSV missing column: {col}")
        if col not in predicted.columns:
            raise ValueError(f"Predicted FK CSV missing column: {col}")

    if len(desired) != len(predicted):
        raise ValueError(
            f"Timestep mismatch: desired={len(desired)}, predicted={len(predicted)}"
        )

    t = desired["t"].to_numpy()

    p_des = desired[["x", "y", "z"]].to_numpy()
    p_pred = predicted[["x", "y", "z"]].to_numpy()

    error = np.linalg.norm(p_pred - p_des, axis=1)

    mean_error = float(np.mean(error))
    max_error = float(np.max(error))

    plt.figure()
    plt.plot(t, error)
    plt.xlabel("time [s]")
    plt.ylabel("Cartesian error [m]")
    plt.title(
        f"Tracking Error Over Time\n"
        f"mean={mean_error:.4f} m, max={max_error:.4f} m"
    )
    plt.grid(True)

    output_png = Path(args.output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, dpi=200, bbox_inches="tight")

    print(f"Saved tracking error plot to: {output_png}")
    print(f"mean error: {mean_error:.6f} m")
    print(f"max error:  {max_error:.6f} m")


if __name__ == "__main__":
    main()