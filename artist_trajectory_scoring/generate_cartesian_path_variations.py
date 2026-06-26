#!/usr/bin/env python3
"""
Generate randomized Cartesian drawing path variations for supervised IK experts.

Output layout:
    data/cartesian_expert_dataset_v2/raw_paths/
      train/path_0001/desired_path.csv
      train/path_0001/path_meta.json
      test/path_0001/desired_path.csv
      test/path_0001/path_meta.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Callable, Dict, Tuple

import numpy as np
import pandas as pd


WORKSPACE_X = (-0.35, 0.35)
WORKSPACE_Y = (-0.25, 0.35)
Z_BASE = 1.18


def make_time(timesteps: int) -> np.ndarray:
    if timesteps < 2:
        raise ValueError("--timesteps must be at least 2")
    return np.linspace(0.0, 1.0, timesteps, dtype=np.float32)


def rotation_matrix(theta: float) -> np.ndarray:
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    return np.array([[c, -s], [s, c]], dtype=np.float32)


def place_xy_in_workspace(
    rng: np.random.Generator,
    xy: np.ndarray,
    rotation: float,
) -> Tuple[np.ndarray, Tuple[float, float]]:
    rotated = xy @ rotation_matrix(rotation).T

    min_xy = rotated.min(axis=0)
    max_xy = rotated.max(axis=0)

    cx_low = WORKSPACE_X[0] - float(min_xy[0])
    cx_high = WORKSPACE_X[1] - float(max_xy[0])
    cy_low = WORKSPACE_Y[0] - float(min_xy[1])
    cy_high = WORKSPACE_Y[1] - float(max_xy[1])

    if cx_low > cx_high or cy_low > cy_high:
        # Very rare with current scales, but keep generation robust.
        scale = min(
            (WORKSPACE_X[1] - WORKSPACE_X[0]) / max(float(max_xy[0] - min_xy[0]), 1e-6),
            (WORKSPACE_Y[1] - WORKSPACE_Y[0]) / max(float(max_xy[1] - min_xy[1]), 1e-6),
            1.0,
        ) * 0.90
        rotated *= scale
        min_xy = rotated.min(axis=0)
        max_xy = rotated.max(axis=0)
        cx_low = WORKSPACE_X[0] - float(min_xy[0])
        cx_high = WORKSPACE_X[1] - float(max_xy[0])
        cy_low = WORKSPACE_Y[0] - float(min_xy[1])
        cy_high = WORKSPACE_Y[1] - float(max_xy[1])

    cx = float(rng.uniform(cx_low, cx_high))
    cy = float(rng.uniform(cy_low, cy_high))
    return rotated + np.array([cx, cy], dtype=np.float32), (cx, cy)


def finish_xyz(
    rng: np.random.Generator,
    xy: np.ndarray,
    rotation: float,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    xy_world, center = place_xy_in_workspace(rng, xy, rotation)
    z = np.full((len(xy_world), 1), Z_BASE + rng.uniform(-0.006, 0.006), dtype=np.float32)
    xyz = np.concatenate([xy_world.astype(np.float32), z], axis=1)
    return xyz, {"center": [center[0], center[1]], "rotation": float(rotation), "z": float(z[0, 0])}


def line_path(rng: np.random.Generator, t: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    length = float(rng.uniform(0.18, 0.55))
    angle = float(rng.uniform(0.0, 2.0 * np.pi))
    s0 = float(rng.uniform(-0.15, 0.15))
    s1 = s0 + length
    s = np.linspace(s0, s1, len(t), dtype=np.float32)
    xy = np.stack([s - 0.5 * (s0 + s1), np.zeros_like(s)], axis=1)
    xyz, meta = finish_xyz(rng, xy, angle)
    meta.update({"length": length, "local_start": s0, "local_end": s1})
    return xyz, meta


def vertical_line_path(rng: np.random.Generator, t: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    length = float(rng.uniform(0.20, 0.50))
    angle = float((np.pi / 2.0) + rng.uniform(-0.20, 0.20))
    s = np.linspace(-0.5 * length, 0.5 * length, len(t), dtype=np.float32)
    xy = np.stack([s, np.zeros_like(s)], axis=1)
    xyz, meta = finish_xyz(rng, xy, angle)
    meta.update({"length": length})
    return xyz, meta


def arc_path(rng: np.random.Generator, t: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    radius = float(rng.uniform(0.12, 0.30))
    sweep = float(rng.uniform(0.45 * np.pi, 1.45 * np.pi))
    theta0 = float(rng.uniform(-np.pi, np.pi))
    theta = np.linspace(theta0, theta0 + sweep, len(t), dtype=np.float32)
    xy = np.stack([radius * np.cos(theta), radius * np.sin(theta)], axis=1)
    xy -= xy.mean(axis=0, keepdims=True)
    rotation = float(rng.uniform(0.0, 2.0 * np.pi))
    xyz, meta = finish_xyz(rng, xy, rotation)
    meta.update({"radius": radius, "sweep": sweep, "theta0": theta0})
    return xyz, meta


def circle_path(rng: np.random.Generator, t: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    radius = float(rng.uniform(0.08, 0.23))
    phase = float(rng.uniform(0.0, 2.0 * np.pi))
    direction = int(rng.choice([-1, 1]))
    theta = phase + direction * np.linspace(0.0, 2.0 * np.pi, len(t), endpoint=False, dtype=np.float32)
    xy = np.stack([radius * np.cos(theta), radius * np.sin(theta)], axis=1)
    rotation = float(rng.uniform(0.0, 2.0 * np.pi))
    xyz, meta = finish_xyz(rng, xy, rotation)
    meta.update({"radius": radius, "phase": phase, "direction": direction})
    return xyz, meta


def sine_path(rng: np.random.Generator, t: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    length = float(rng.uniform(0.30, 0.62))
    amplitude = float(rng.uniform(0.035, 0.12))
    frequency = float(rng.uniform(1.0, 3.5))
    phase = float(rng.uniform(0.0, 2.0 * np.pi))
    s = np.linspace(-0.5, 0.5, len(t), dtype=np.float32)
    x = length * s
    y = amplitude * np.sin(2.0 * np.pi * frequency * s + phase)
    xy = np.stack([x, y], axis=1)
    rotation = float(rng.uniform(0.0, 2.0 * np.pi))
    xyz, meta = finish_xyz(rng, xy, rotation)
    meta.update({"length": length, "amplitude": amplitude, "frequency": frequency, "phase": phase})
    return xyz, meta


def zigzag_path(rng: np.random.Generator, t: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    length = float(rng.uniform(0.32, 0.62))
    amplitude = float(rng.uniform(0.045, 0.14))
    count = int(rng.integers(3, 8))
    s = np.linspace(0.0, 1.0, len(t), dtype=np.float32)
    phase = (s * count) % 1.0
    tri = 2.0 * np.abs(2.0 * phase - 1.0) - 1.0
    x = length * (s - 0.5)
    y = amplitude * tri
    xy = np.stack([x, y], axis=1)
    rotation = float(rng.uniform(0.0, 2.0 * np.pi))
    xyz, meta = finish_xyz(rng, xy, rotation)
    meta.update({"length": length, "amplitude": amplitude, "zigzag_count": count})
    return xyz, meta


def spiral_path(rng: np.random.Generator, t: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    turns = float(rng.uniform(1.2, 2.8))
    r0 = float(rng.uniform(0.01, 0.05))
    r1 = float(rng.uniform(0.12, 0.24))
    phase = float(rng.uniform(0.0, 2.0 * np.pi))
    direction = int(rng.choice([-1, 1]))
    s = np.linspace(0.0, 1.0, len(t), dtype=np.float32)
    theta = phase + direction * 2.0 * np.pi * turns * s
    r = r0 + (r1 - r0) * s
    xy = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)
    xy -= xy.mean(axis=0, keepdims=True)
    rotation = float(rng.uniform(0.0, 2.0 * np.pi))
    xyz, meta = finish_xyz(rng, xy, rotation)
    meta.update({"turns": turns, "r0": r0, "r1": r1, "phase": phase, "direction": direction})
    return xyz, meta


def square_path(rng: np.random.Generator, t: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    size = float(rng.uniform(0.16, 0.34))
    half = 0.5 * size
    s = np.linspace(0.0, 1.0, len(t), endpoint=False, dtype=np.float32)
    xy = np.empty((len(t), 2), dtype=np.float32)
    start_edge_offset = float(rng.uniform(0.0, 4.0))
    direction = int(rng.choice([-1, 1]))

    for i, si in enumerate(s):
        u = (start_edge_offset + direction * si * 4.0) % 4.0
        edge = int(np.floor(u))
        local = float(u - edge)
        if edge == 0:
            xy[i] = [-half + size * local, -half]
        elif edge == 1:
            xy[i] = [half, -half + size * local]
        elif edge == 2:
            xy[i] = [half - size * local, half]
        else:
            xy[i] = [-half, half - size * local]

    xy -= xy.mean(axis=0, keepdims=True)
    rotation = float(rng.uniform(0.0, 2.0 * np.pi))
    xyz, meta = finish_xyz(rng, xy, rotation)
    meta.update({"size": size, "start_edge_offset": start_edge_offset, "direction": direction})
    return xyz, meta


PATH_GENERATORS: Dict[str, Callable[[np.random.Generator, np.ndarray], Tuple[np.ndarray, Dict[str, Any]]]] = {
    "line": line_path,
    "vertical_line": vertical_line_path,
    "arc": arc_path,
    "circle": circle_path,
    "sine": sine_path,
    "zigzag": zigzag_path,
    "spiral": spiral_path,
    "square": square_path,
}


PATH_TYPE_WEIGHTS = {
    "line": 1.0,
    "vertical_line": 1.0,
    "arc": 1.0,
    "circle": 1.0,
    "sine": 1.0,
    "spiral": 1.0,
    "square": 2.5,
    "zigzag": 2.5,
}


def choose_path_type(rng: np.random.Generator) -> str:
    names = np.array(list(PATH_TYPE_WEIGHTS.keys()))
    weights = np.array([PATH_TYPE_WEIGHTS[name] for name in names], dtype=np.float64)
    weights /= weights.sum()
    return str(rng.choice(names, p=weights))


def save_path(path_dir: Path, t: np.ndarray, xyz: np.ndarray, meta: Dict[str, Any]) -> None:
    path_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"t": t, "x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2]}).to_csv(
        path_dir / "desired_path.csv",
        index=False,
    )
    with open(path_dir / "path_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)


def generate_split(
    rng: np.random.Generator,
    output_dir: Path,
    split: str,
    count: int,
    timesteps: int,
) -> None:
    t = make_time(timesteps)
    for idx in range(1, count + 1):
        path_id = f"path_{idx:04d}"
        path_type = choose_path_type(rng)
        xyz, params = PATH_GENERATORS[path_type](rng, t)
        meta = {
            "path_id": path_id,
            "split": split,
            "path_type": path_type,
            "parameters": params,
            "timesteps": timesteps,
        }
        save_path(output_dir / split / path_id, t, xyz, meta)
        print(f"Saved {split}/{path_id}: {path_type}")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate randomized Cartesian path variations.")
    parser.add_argument("--output_dir", type=Path, default=Path("data/cartesian_expert_dataset_v2/raw_paths"))
    parser.add_argument("--num_train", type=int, default=80)
    parser.add_argument("--num_test", type=int, default=20)
    parser.add_argument("--timesteps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    if args.num_train < 0 or args.num_test < 0:
        raise ValueError("--num_train and --num_test must be non-negative")

    rng = np.random.default_rng(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    generate_split(rng, args.output_dir, "train", args.num_train, args.timesteps)
    generate_split(rng, args.output_dir, "test", args.num_test, args.timesteps)

    print()
    print(f"Done. Wrote raw path variations to: {args.output_dir}")
    print(f"  train: {args.num_train}")
    print(f"  test:  {args.num_test}")
    print(f"  T:     {args.timesteps}")


if __name__ == "__main__":
    main()
