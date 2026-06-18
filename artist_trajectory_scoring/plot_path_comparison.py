import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(
        description="Plot desired Cartesian path and predicted FK path."
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
        default="path_comparison.png",
        help="Output plot image path.",
    )

    args = parser.parse_args()

    desired_csv = Path(args.desired_csv)
    predicted_fk_csv = Path(args.predicted_fk_csv)
    output_png = Path(args.output_png)

    desired = pd.read_csv(desired_csv)
    predicted = pd.read_csv(predicted_fk_csv)

    required_cols = ["x", "y", "z"]

    for col in required_cols:
        if col not in desired.columns:
            raise ValueError(f"{desired_csv} is missing column: {col}")

        if col not in predicted.columns:
            raise ValueError(f"{predicted_fk_csv} is missing column: {col}")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.plot(
        desired["x"],
        desired["y"],
        desired["z"],
        label="desired path",
    )

    ax.plot(
        predicted["x"],
        predicted["y"],
        predicted["z"],
        label="predicted FK path",
    )

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title("Desired Path vs Predicted FK Path")
    ax.legend()

    output_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_png, dpi=200, bbox_inches="tight")

    print(f"Saved plot to: {output_png}")


if __name__ == "__main__":
    main()