import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def save_candidate(path: Path, t: np.ndarray, q: np.ndarray):
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


def main():
    parser = argparse.ArgumentParser(
        description="Generate controlled test candidate joint trajectories."
    )

    parser.add_argument(
        "--output_dir",
        default="test_candidates",
        help="Directory where candidate CSV files will be saved.",
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
        "--noise_std",
        type=float,
        default=0.03,
        help="Noise standard deviation in radians for noisy trajectory.",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    t = np.linspace(0.0, args.duration, args.num_steps)

    # Smooth baseline trajectory.
    # Small sinusoidal joint motion in radians.
    q_smooth = np.zeros((args.num_steps, 6))
    q_smooth[:, 0] = 0.20 * np.sin(2.0 * np.pi * t / args.duration)
    q_smooth[:, 1] = 0.15 * np.sin(2.0 * np.pi * t / args.duration + 0.5)
    q_smooth[:, 2] = 0.10 * np.sin(2.0 * np.pi * t / args.duration + 1.0)
    q_smooth[:, 3] = 0.08 * np.sin(2.0 * np.pi * t / args.duration + 1.5)
    q_smooth[:, 4] = 0.05 * np.sin(2.0 * np.pi * t / args.duration + 2.0)
    q_smooth[:, 5] = 0.04 * np.sin(2.0 * np.pi * t / args.duration + 2.5)

    # Noisy version: should have similar path but worse smoothness.
    rng = np.random.default_rng(seed=0)
    q_noisy = q_smooth + rng.normal(
        loc=0.0,
        scale=args.noise_std,
        size=q_smooth.shape,
    )

    # Offset version: should have worse path tracking.
    q_offset = q_smooth.copy()
    q_offset[:, 0] += 0.25
    q_offset[:, 1] -= 0.15

    # Acceleration spike version: mostly same path, but one sharp joint jump.
    q_accel_spike = q_smooth.copy()
    spike_idx = args.num_steps // 2
    q_accel_spike[spike_idx:, 2] += 0.20

    save_candidate(output_dir / "candidate_smooth.csv", t, q_smooth)
    save_candidate(output_dir / "candidate_noisy.csv", t, q_noisy)
    save_candidate(output_dir / "candidate_offset.csv", t, q_offset)
    save_candidate(output_dir / "candidate_accel_spike.csv", t, q_accel_spike)

    print(f"Saved test candidates to: {output_dir}")
    print("Generated:")
    print("  candidate_smooth.csv")
    print("  candidate_noisy.csv")
    print("  candidate_offset.csv")
    print("  candidate_accel_spike.csv")


if __name__ == "__main__":
    main()