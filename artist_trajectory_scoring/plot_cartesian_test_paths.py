#!/usr/bin/env python3
"""
Create plots for all evaluated Cartesian test paths.

This wraps plot_diffusion_diagnostic.py over each test path folder.

Example:
    python plot_cartesian_test_paths.py \
      --dataset_dir data/cartesian_test_paths \
      --plot_script plot_diffusion_diagnostic.py
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=Path, default=Path("data/cartesian_test_paths"))
    parser.add_argument("--plot_script", type=Path, default=Path("plot_diffusion_diagnostic.py"))
    parser.add_argument("--q_name", type=str, default="time_conditioned_pred_q.csv")
    parser.add_argument("--ee_name", type=str, default="time_conditioned_pred_ee.csv")
    parser.add_argument("--plot_name", type=str, default="time_conditioned_plot.png")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()

    path_dirs = sorted([p for p in args.dataset_dir.iterdir() if p.is_dir()])
    if not path_dirs:
        raise FileNotFoundError(f"No subfolders found in {args.dataset_dir}")

    for path_dir in path_dirs:
        desired_csv = path_dir / "desired_path.csv"
        q_csv = path_dir / args.q_name
        ee_csv = path_dir / args.ee_name
        output_png = path_dir / args.plot_name

        if not desired_csv.exists():
            print(f"Skipping {path_dir.name}: missing desired_path.csv")
            continue
        if not q_csv.exists():
            print(f"Skipping {path_dir.name}: missing {args.q_name}")
            continue
        if not ee_csv.exists():
            print(f"Skipping {path_dir.name}: missing {args.ee_name}")
            continue

        print(f"Plotting {path_dir.name}")

        cmd = [
            sys.executable,
            str(args.plot_script),
            "--desired_path",
            str(desired_csv),
            "--ee_csv",
            str(ee_csv),
            "--q_csv",
            str(q_csv),
            "--output_png",
            str(output_png),
        ]

        if args.show:
            cmd.append("--show")

        proc = subprocess.run(cmd, capture_output=True, text=True)

        if proc.returncode != 0:
            raise RuntimeError(
                "Plotting failed.\n"
                f"Command: {' '.join(cmd)}\n"
                f"STDOUT:\n{proc.stdout}\n"
                f"STDERR:\n{proc.stderr}"
            )

    print("\nDone. Plots saved in each Cartesian test path folder.")


if __name__ == "__main__":
    main()
