#!/usr/bin/env python3
"""Tabular Q-learning environment for debugging Cartesian path costs.

This module intentionally has no robot, URDF, forward-kinematics, or Gym
dependency.  A point moves on an 8-connected x-y grid built around a desired
path loaded from a ``t,x,y,z`` CSV file.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd


ACTION_DELTAS = np.asarray(
    [
        [0, 1],   # up
        [0, -1],  # down
        [-1, 0],  # left
        [1, 0],   # right
        [-1, 1],  # up-left
        [1, 1],   # up-right
        [-1, -1], # down-left
        [1, -1],  # down-right
    ],
    dtype=np.int32,
)
ACTION_NAMES = (
    "up", "down", "left", "right",
    "up_left", "up_right", "down_left", "down_right",
)
NO_PREVIOUS_ACTION = len(ACTION_DELTAS)


@dataclass(frozen=True)
class RewardWeights:
    """Non-negative weights; costs are subtracted from reward."""

    point: float = 2.0
    segment: float = 5.0
    alignment: float = 0.25
    backward: float = 1.0
    cusp: float = 1.0
    step: float = 0.01
    finish: float = 25.0


def load_desired_path(path_csv: str | Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load and validate time and xyz arrays, preserving z for outputs."""
    path_csv = Path(path_csv)
    if not path_csv.exists():
        raise FileNotFoundError(
            f"Desired path not found: {path_csv}\n"
            "Generate the Cartesian test paths first or pass --path_csv."
        )
    frame = pd.read_csv(path_csv)
    required = {"t", "x", "y", "z"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path_csv} is missing columns: {sorted(missing)}")
    if len(frame) < 2:
        raise ValueError("desired_path.csv must contain at least two rows")

    t = frame["t"].to_numpy(dtype=np.float64)
    xyz = frame[["x", "y", "z"]].to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(xyz)):
        raise ValueError("desired_path.csv contains NaN or infinite values")
    if np.any(np.diff(t) < 0.0):
        raise ValueError("The t column must be non-decreasing")
    return t, xyz


class CartesianPathDebugEnv:
    """Small deterministic MDP for exposing path-cost failure modes.

    State is ``(grid_x, grid_y, closest_path_index, previous_action)``.
    The previous-action sentinel is ``num_actions`` immediately after reset.
    """

    def __init__(
        self,
        path_csv: str | Path,
        grid_spacing: float = 0.02,
        margin: float = 0.10,
        max_steps: int = 250,
        finish_tolerance: float | None = None,
        weights: RewardWeights | None = None,
    ) -> None:
        if grid_spacing <= 0.0:
            raise ValueError("grid_spacing must be positive")
        if margin < 0.0:
            raise ValueError("margin must be non-negative")
        if max_steps < 1:
            raise ValueError("max_steps must be positive")

        self.path_csv = Path(path_csv)
        self.path_t, self.path_xyz = load_desired_path(path_csv)
        self.path_xy = self.path_xyz[:, :2]
        self.grid_spacing = float(grid_spacing)
        self.margin = float(margin)
        self.max_steps = int(max_steps)
        self.finish_tolerance = float(
            1.5 * grid_spacing if finish_tolerance is None else finish_tolerance
        )
        self.weights = weights or RewardWeights()
        if any(value < 0.0 for value in vars(self.weights).values()):
            raise ValueError("Reward weights must be non-negative")

        low = np.floor((self.path_xy.min(axis=0) - margin) / grid_spacing) * grid_spacing
        high = np.ceil((self.path_xy.max(axis=0) + margin) / grid_spacing) * grid_spacing
        self.x_values = np.arange(low[0], high[0] + 0.5 * grid_spacing, grid_spacing)
        self.y_values = np.arange(low[1], high[1] + 0.5 * grid_spacing, grid_spacing)

        self._segment_start = self.path_xy[:-1]
        self._segment_vector = np.diff(self.path_xy, axis=0)
        self._segment_length_sq = np.sum(self._segment_vector ** 2, axis=1)
        tangent = np.empty_like(self.path_xy)
        tangent[0] = self.path_xy[1] - self.path_xy[0]
        tangent[-1] = self.path_xy[-1] - self.path_xy[-2]
        if len(self.path_xy) > 2:
            tangent[1:-1] = self.path_xy[2:] - self.path_xy[:-2]
        norms = np.linalg.norm(tangent, axis=1)
        tangent[norms > 0.0] /= norms[norms > 0.0, None]
        tangent[norms == 0.0] = 0.0
        self._tangent = tangent

        self.num_actions = len(ACTION_DELTAS)
        self.state_shape = (
            len(self.x_values),
            len(self.y_values),
            len(self.path_xy),
            self.num_actions + 1,
        )
        self.grid_index = np.zeros(2, dtype=np.int32)
        self.closest_index = 0
        self.previous_action = NO_PREVIOUS_ACTION
        self.steps = 0

    @property
    def position(self) -> np.ndarray:
        return np.asarray(
            [self.x_values[self.grid_index[0]], self.y_values[self.grid_index[1]]],
            dtype=np.float64,
        )

    def _nearest_grid_index(self, point: np.ndarray) -> np.ndarray:
        return np.asarray(
            [
                int(np.argmin(np.abs(self.x_values - point[0]))),
                int(np.argmin(np.abs(self.y_values - point[1]))),
            ],
            dtype=np.int32,
        )

    def _closest_point(self, position: np.ndarray) -> Tuple[int, float]:
        distances = np.linalg.norm(self.path_xy - position, axis=1)
        index = int(np.argmin(distances))
        return index, float(distances[index])

    def _closest_segment_distance(self, position: np.ndarray) -> Tuple[int, float]:
        relative = position - self._segment_start
        projection = np.divide(
            np.sum(relative * self._segment_vector, axis=1),
            self._segment_length_sq,
            out=np.zeros_like(self._segment_length_sq),
            where=self._segment_length_sq > 0.0,
        )
        projection = np.clip(projection, 0.0, 1.0)
        closest = self._segment_start + projection[:, None] * self._segment_vector
        distances = np.linalg.norm(closest - position, axis=1)
        index = int(np.argmin(distances))
        return index, float(distances[index])

    def _state(self) -> Tuple[int, int, int, int]:
        return (
            int(self.grid_index[0]),
            int(self.grid_index[1]),
            int(self.closest_index),
            int(self.previous_action),
        )

    def reset(self) -> Tuple[Tuple[int, int, int, int], Dict[str, float | int]]:
        self.grid_index = self._nearest_grid_index(self.path_xy[0])
        self.closest_index, point_distance = self._closest_point(self.position)
        self.previous_action = NO_PREVIOUS_ACTION
        self.steps = 0
        _, segment_distance = self._closest_segment_distance(self.position)
        return self._state(), {
            "closest_index": self.closest_index,
            "point_distance": point_distance,
            "segment_distance": segment_distance,
        }

    def step(
        self, action: int
    ) -> Tuple[Tuple[int, int, int, int], float, bool, bool, Dict[str, float | int | bool]]:
        if not 0 <= action < self.num_actions:
            raise ValueError(f"action must be in [0, {self.num_actions - 1}]")

        old_index = self.closest_index
        target_index = min(old_index + 1, len(self.path_xy) - 1)
        old_previous_action = self.previous_action

        proposed = self.grid_index + ACTION_DELTAS[action]
        clipped = np.clip(proposed, [0, 0], np.asarray(self.state_shape[:2]) - 1)
        hit_boundary = bool(np.any(clipped != proposed))
        self.grid_index = clipped.astype(np.int32)
        self.steps += 1

        position = self.position
        self.closest_index, nearest_point_distance = self._closest_point(position)
        segment_index, segment_distance = self._closest_segment_distance(position)
        target_point_distance = float(np.linalg.norm(position - self.path_xy[target_index]))

        motion = ACTION_DELTAS[action].astype(np.float64)
        motion /= np.linalg.norm(motion)
        tangent_dot = float(np.clip(np.dot(motion, self._tangent[old_index]), -1.0, 1.0))
        alignment_cost = 1.0 - tangent_dot
        backward_steps = max(0, old_index - self.closest_index)

        cusp_cost = 0.0
        action_dot = 1.0
        if old_previous_action != NO_PREVIOUS_ACTION:
            previous_motion = ACTION_DELTAS[old_previous_action].astype(np.float64)
            previous_motion /= np.linalg.norm(previous_motion)
            action_dot = float(np.clip(np.dot(motion, previous_motion), -1.0, 1.0))
            cusp_cost = max(0.0, -action_dot)

        terms = {
            "point": -self.weights.point * target_point_distance,
            "segment": -self.weights.segment * segment_distance,
            "alignment": -self.weights.alignment * alignment_cost,
            "backward": -self.weights.backward * backward_steps,
            "cusp": -self.weights.cusp * cusp_cost,
            "step": -self.weights.step,
            "finish": 0.0,
        }

        endpoint_distance = float(np.linalg.norm(position - self.path_xy[-1]))
        terminated = bool(
            self.closest_index >= len(self.path_xy) - 2
            and endpoint_distance <= self.finish_tolerance
        )
        if terminated:
            terms["finish"] = self.weights.finish
        truncated = bool(self.steps >= self.max_steps and not terminated)
        reward = float(sum(terms.values()))
        self.previous_action = int(action)

        info: Dict[str, float | int | bool] = {
            "x": float(position[0]),
            "y": float(position[1]),
            "z": float(self.path_xyz[self.closest_index, 2]),
            "closest_index": self.closest_index,
            "nearest_point_distance": nearest_point_distance,
            "target_point_index": target_index,
            "target_point_distance": target_point_distance,
            "nearest_segment_index": segment_index,
            "segment_distance": segment_distance,
            "tangent_dot": tangent_dot,
            "action_dot": action_dot,
            "backward_steps": backward_steps,
            "cusp_cost": cusp_cost,
            "endpoint_distance": endpoint_distance,
            "hit_boundary": hit_boundary,
        }
        info.update({f"reward_{name}": value for name, value in terms.items()})
        return self._state(), reward, terminated, truncated, info

