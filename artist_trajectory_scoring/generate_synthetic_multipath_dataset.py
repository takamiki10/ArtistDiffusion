import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from score_trajectory import rokae_forward_kinematics


def save_joint_csv(path: Path, t: np.ndarray, q: np.ndarray):
    df = pd.DataFrame({
        "t": t,
        "q1": q[:, 0],
        "q2": q[:, 1],
        "q3": q[:, 2],
        "q4": q[:, 3],
        "q5": q[:, 4],
        "q6": q[:, 5],
    })
    df.to_csv(path, index=False)


def save_path_csv(path: Path, t: np.ndarray, p: np.ndarray):
    df = pd.DataFrame({
        "t": t,
        "x": p[:, 0],
        "y": p[:, 1],
        "z": p[:, 2],
    })
    df.to_csv(path, index=False)


def make_smooth_joint_trajectory(
    t: np.ndarray,
    duration: float,
    rng: np.random.Generator,
) -> np.ndarray:
    num_steps = len(t)
    q = np.zeros((num_steps, 6), dtype=np.float64)

    # Keep amplitudes moderate so trajectories are smooth and plausible.
    amplitudes = rng.uniform(0.03, 0.25, size=6)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=6)
    offsets = rng.uniform(-0.15, 0.15, size=6)

    # Mostly one-cycle sinusoidal motions, with small second harmonic variation.
    for j in range(6):
        base = amplitudes[j] * np.sin(2.0 * np.pi * t / duration + phases[j])
        harmonic = 0.25 * amplitudes[j] * np.sin(
            4.0 * np.pi * t / duration + 0.5 * phases[j]
        )
        q[:, j] = offsets[j] + base + harmonic

    return q


def main():
    parser = argparse.ArgumentParser(
        description="Generate a synthetic multi-path artist trajectory dataset."
    )

    parser.add_argument(
        "--output_dir",
        default="data/synthetic_paths",
        help="Output directory for synthetic path dataset.",
    )

    parser.add_argument(
        "--num_paths",
        type=int,
        default=20,
        help="Number of different path examples to generate.",
    )

    parser.add_argument(
        "--num_steps",
        type=int,
        default=100,
        help="Number of timesteps per trajectory.",
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=1.0,
        help="Trajectory duration in seconds.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed.",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    t = np.linspace(0.0, args.duration, args.num_steps)

    summary_rows = []

    for i in range(args.num_paths):
        path_dir = output_dir / f"path_{i + 1:03d}"
        path_dir.mkdir(parents=True, exist_ok=True)

        q = make_smooth_joint_trajectory(
            t=t,
            duration=args.duration,
            rng=rng,
        )

        p_ee = rokae_forward_kinematics(q)

        expert_q_file = path_dir / "expert_q.csv"
        desired_path_file = path_dir / "desired_path.csv"

        save_joint_csv(expert_q_file, t, q)
        save_path_csv(desired_path_file, t, p_ee)

        summary_rows.append({
            "path_id": f"path_{i + 1:03d}",
            "desired_path_csv": str(desired_path_file),
            "expert_q_csv": str(expert_q_file),
            "num_steps": args.num_steps,
        })

        print(f"[OK] Generated {path_dir}")

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = output_dir / "summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    print()
    print(f"Saved synthetic dataset summary to: {summary_csv}")
    print(f"Generated {args.num_paths} path examples.")


if __name__ == "__main__":
    main()