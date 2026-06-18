#!/usr/bin/env python3
"""
Generate drawing-like Cartesian test paths.

Creates folders containing desired_path.csv with columns:
    t,x,y,z

Example:
    python generate_cartesian_test_paths.py \
      --output_dir data/cartesian_test_paths \
      --num_steps 100 \
      --duration 1.0
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable, Dict

import numpy as np
import pandas as pd


def make_time(num_steps: int, duration: float) -> np.ndarray:
    if num_steps < 2:
        raise ValueError("--num_steps must be at least 2")
    return np.linspace(0.0, duration, num_steps, dtype=np.float32)


def save_path(output_dir: Path, name: str, t: np.ndarray, xyz: np.ndarray) -> None:
    path_dir = output_dir / name
    path_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame({
        "t": t,
        "x": xyz[:, 0],
        "y": xyz[:, 1],
        "z": xyz[:, 2],
    })

    out_csv = path_dir / "desired_path.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved {out_csv}")


def line_path(t: np.ndarray) -> np.ndarray:
    s = np.linspace(0.0, 1.0, len(t), dtype=np.float32)
    x = -0.25 + 0.50 * s
    y = -0.12 + 0.24 * s
    z = np.full_like(x, 1.18)
    return np.stack([x, y, z], axis=1).astype(np.float32)


def vertical_line_path(t: np.ndarray) -> np.ndarray:
    s = np.linspace(0.0, 1.0, len(t), dtype=np.float32)
    x = np.full_like(s, 0.0)
    y = -0.22 + 0.44 * s
    z = np.full_like(s, 1.18)
    return np.stack([x, y, z], axis=1).astype(np.float32)


def arc_path(t: np.ndarray) -> np.ndarray:
    s = np.linspace(0.0, 1.0, len(t), dtype=np.float32)
    theta = np.pi * (0.15 + 0.70 * s)
    r = 0.28
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = np.full_like(x, 1.18)
    return np.stack([x, y, z], axis=1).astype(np.float32)


def circle_path(t: np.ndarray) -> np.ndarray:
    s = np.linspace(0.0, 1.0, len(t), endpoint=False, dtype=np.float32)
    theta = 2.0 * np.pi * s
    r = 0.20
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = np.full_like(x, 1.18)
    return np.stack([x, y, z], axis=1).astype(np.float32)


def sine_path(t: np.ndarray) -> np.ndarray:
    s = np.linspace(0.0, 1.0, len(t), dtype=np.float32)
    x = -0.30 + 0.60 * s
    y = 0.08 * np.sin(2.0 * np.pi * 2.0 * s)
    z = np.full_like(x, 1.18)
    return np.stack([x, y, z], axis=1).astype(np.float32)


def zigzag_path(t: np.ndarray) -> np.ndarray:
    s = np.linspace(0.0, 1.0, len(t), dtype=np.float32)
    x = -0.30 + 0.60 * s
    phase = (s * 4.0) % 1.0
    tri = 2.0 * np.abs(2.0 * phase - 1.0) - 1.0
    y = 0.10 * tri
    z = np.full_like(x, 1.18)
    return np.stack([x, y, z], axis=1).astype(np.float32)


def spiral_path(t: np.ndarray) -> np.ndarray:
    s = np.linspace(0.0, 1.0, len(t), dtype=np.float32)
    theta = 2.0 * np.pi * 2.0 * s
    r = 0.03 + 0.20 * s
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = np.full_like(x, 1.18)
    return np.stack([x, y, z], axis=1).astype(np.float32)


def square_path(t: np.ndarray) -> np.ndarray:
    s = np.linspace(0.0, 1.0, len(t), endpoint=False, dtype=np.float32)
    half = 0.16
    x = np.empty_like(s)
    y = np.empty_like(s)

    for i, si in enumerate(s):
        u = si * 4.0
        edge = int(np.floor(u))
        local = u - edge

        if edge == 0:
            x[i] = -half + 2.0 * half * local
            y[i] = -half
        elif edge == 1:
            x[i] = half
            y[i] = -half + 2.0 * half * local
        elif edge == 2:
            x[i] = half - 2.0 * half * local
            y[i] = half
        else:
            x[i] = -half
            y[i] = half - 2.0 * half * local

    z = np.full_like(x, 1.18)
    return np.stack([x, y, z], axis=1).astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, default=Path("data/cartesian_test_paths"))
    parser.add_argument("--num_steps", type=int, default=100)
    parser.add_argument("--duration", type=float, default=1.0)
    args = parser.parse_args()

    t = make_time(args.num_steps, args.duration)

    path_generators: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
        "line_001": line_path,
        "vertical_line_001": vertical_line_path,
        "arc_001": arc_path,
        "circle_001": circle_path,
        "sine_001": sine_path,
        "zigzag_001": zigzag_path,
        "spiral_001": spiral_path,
        "square_001": square_path,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for name, fn in path_generators.items():
        save_path(args.output_dir, name, t, fn(t))

    print(f"\nDone. Generated {len(path_generators)} Cartesian test paths in {args.output_dir}")


if __name__ == "__main__":
    main()
