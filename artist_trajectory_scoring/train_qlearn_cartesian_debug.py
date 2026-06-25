#!/usr/bin/env python3
"""Train a tabular Q-learner against Cartesian path-following costs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from plot_qlearn_cartesian_debug import create_debug_plot
from qlearn_cartesian_debug_env import CartesianPathDebugEnv, RewardWeights


DEFAULT_PATH = Path("data/cartesian_test_paths/arc_001/desired_path.csv")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Debug Cartesian path reward terms with tabular Q-learning."
    )
    parser.add_argument("--path_csv", type=Path, default=DEFAULT_PATH)
    parser.add_argument("--output_dir", type=Path, default=Path("qlearn_cartesian_debug_output"))
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--max_steps", type=int, default=250)
    parser.add_argument("--grid_spacing", type=float, default=0.02)
    parser.add_argument("--margin", type=float, default=0.10)
    parser.add_argument("--finish_tolerance", type=float, default=None)
    parser.add_argument("--learning_rate", type=float, default=0.15)
    parser.add_argument("--discount", type=float, default=0.98)
    parser.add_argument("--epsilon_start", type=float, default=1.0)
    parser.add_argument("--epsilon_end", type=float, default=0.03)
    parser.add_argument("--epsilon_decay", type=float, default=0.997)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--log_every", type=int, default=100)

    rewards = parser.add_argument_group("reward weights")
    rewards.add_argument("--w_point", type=float, default=2.0)
    rewards.add_argument("--w_segment", type=float, default=5.0)
    rewards.add_argument("--w_alignment", type=float, default=0.25)
    rewards.add_argument("--w_backward", type=float, default=1.0)
    rewards.add_argument("--w_cusp", type=float, default=1.0)
    rewards.add_argument("--w_step", type=float, default=0.01)
    rewards.add_argument("--finish_bonus", type=float, default=25.0)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.episodes < 1:
        raise ValueError("--episodes must be positive")
    if not 0.0 < args.learning_rate <= 1.0:
        raise ValueError("--learning_rate must be in (0, 1]")
    if not 0.0 <= args.discount <= 1.0:
        raise ValueError("--discount must be in [0, 1]")
    if not 0.0 <= args.epsilon_end <= args.epsilon_start <= 1.0:
        raise ValueError("Require 0 <= epsilon_end <= epsilon_start <= 1")
    if not 0.0 < args.epsilon_decay <= 1.0:
        raise ValueError("--epsilon_decay must be in (0, 1]")


def make_env(args: argparse.Namespace) -> CartesianPathDebugEnv:
    weights = RewardWeights(
        point=args.w_point,
        segment=args.w_segment,
        alignment=args.w_alignment,
        backward=args.w_backward,
        cusp=args.w_cusp,
        step=args.w_step,
        finish=args.finish_bonus,
    )
    return CartesianPathDebugEnv(
        path_csv=args.path_csv,
        grid_spacing=args.grid_spacing,
        margin=args.margin,
        max_steps=args.max_steps,
        finish_tolerance=args.finish_tolerance,
        weights=weights,
    )


def choose_action(
    q_table: np.ndarray,
    state: Tuple[int, int, int, int],
    epsilon: float,
    rng: np.random.Generator,
) -> int:
    if rng.random() < epsilon:
        return int(rng.integers(q_table.shape[-1]))
    values = q_table[state]
    best = np.flatnonzero(values == values.max())
    return int(rng.choice(best))


def greedy_rollout(
    env: CartesianPathDebugEnv,
    q_table: np.ndarray,
) -> Tuple[pd.DataFrame, Dict[str, float | int | bool]]:
    state, reset_info = env.reset()
    rows: List[Dict[str, float | int | bool]] = [
        {
            "step": 0,
            "t": float(env.path_t[0]),
            "x": float(env.position[0]),
            "y": float(env.position[1]),
            "z": float(env.path_xyz[env.closest_index, 2]),
            "action": -1,
            "action_name": "start",
            "closest_path_index": env.closest_index,
            "tracking_error": float(reset_info["segment_distance"]),
            "target_point_error": float(reset_info["point_distance"]),
            "reward": 0.0,
            "backward_steps": 0,
            "cusp_cost": 0.0,
            "tangent_dot": 0.0,
            "reward_point": 0.0,
            "reward_segment": 0.0,
            "reward_alignment": 0.0,
            "reward_backward": 0.0,
            "reward_cusp": 0.0,
            "reward_step": 0.0,
            "reward_finish": 0.0,
        }
    ]
    total_reward = 0.0
    reward_totals = {name: 0.0 for name in (
        "point", "segment", "alignment", "backward", "cusp", "step", "finish"
    )}
    terminated = truncated = False

    while not (terminated or truncated):
        action = int(np.argmax(q_table[state]))
        state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        for name in reward_totals:
            reward_totals[name] += float(info[f"reward_{name}"])
        closest = int(info["closest_index"])
        if len(env.path_t) > 1:
            t_value = float(env.path_t[closest])
        else:  # guarded by environment validation, kept for output robustness
            t_value = float(env.steps)
        rows.append(
            {
                "step": env.steps,
                "t": t_value,
                "x": float(info["x"]),
                "y": float(info["y"]),
                "z": float(info["z"]),
                "action": action,
                "action_name": env_action_name(action),
                "closest_path_index": closest,
                "tracking_error": float(info["segment_distance"]),
                "target_point_error": float(info["target_point_distance"]),
                "reward": reward,
                "backward_steps": int(info["backward_steps"]),
                "cusp_cost": float(info["cusp_cost"]),
                "tangent_dot": float(info["tangent_dot"]),
                **{
                    f"reward_{name}": float(info[f"reward_{name}"])
                    for name in reward_totals
                },
            }
        )

    path_frame = pd.DataFrame(rows)
    closest_indices = path_frame["closest_path_index"].to_numpy(dtype=np.int64)
    metrics: Dict[str, float | int | bool] = {
        "rollout_reward": total_reward,
        "rollout_steps": int(env.steps),
        "finished": bool(terminated),
        "mean_tracking_error": float(path_frame["tracking_error"].mean()),
        "max_tracking_error": float(path_frame["tracking_error"].max()),
        "final_path_index": int(closest_indices[-1]),
        "backward_transitions": int(np.sum(np.diff(closest_indices) < 0)),
        "backward_index_total": int(np.maximum(-np.diff(closest_indices), 0).sum()),
        "cusp_count": int((path_frame.get("cusp_cost", pd.Series(dtype=float)) > 0.0).sum()),
    }
    metrics.update({f"rollout_reward_{key}": value for key, value in reward_totals.items()})
    return path_frame, metrics


def env_action_name(action: int) -> str:
    from qlearn_cartesian_debug_env import ACTION_NAMES

    return ACTION_NAMES[action]


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    env = make_env(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    q_shape = env.state_shape + (env.num_actions,)
    q_table = np.zeros(q_shape, dtype=np.float32)
    size_mb = q_table.nbytes / (1024.0 ** 2)
    print(f"Q-table shape: {q_shape} ({size_mb:.1f} MiB)")

    epsilon = args.epsilon_start
    logs: List[Dict[str, float | int | bool]] = []
    for episode in range(1, args.episodes + 1):
        state, _ = env.reset()
        total_reward = 0.0
        terminated = truncated = False
        backward_total = 0
        cusp_count = 0
        error_total = 0.0

        while not (terminated or truncated):
            action = choose_action(q_table, state, epsilon, rng)
            next_state, reward, terminated, truncated, info = env.step(action)
            bootstrap = 0.0 if (terminated or truncated) else float(np.max(q_table[next_state]))
            td_target = reward + args.discount * bootstrap
            q_table[state + (action,)] += args.learning_rate * (
                td_target - q_table[state + (action,)]
            )
            state = next_state
            total_reward += reward
            backward_total += int(info["backward_steps"])
            cusp_count += int(float(info["cusp_cost"]) > 0.0)
            error_total += float(info["segment_distance"])

        logs.append(
            {
                "episode": episode,
                "reward": total_reward,
                "steps": env.steps,
                "finished": terminated,
                "epsilon": epsilon,
                "final_path_index": env.closest_index,
                "mean_tracking_error": error_total / env.steps,
                "backward_index_total": backward_total,
                "cusp_count": cusp_count,
            }
        )
        epsilon = max(args.epsilon_end, epsilon * args.epsilon_decay)
        if episode == 1 or episode % args.log_every == 0 or episode == args.episodes:
            recent = logs[-min(args.log_every, len(logs)):]
            mean_reward = float(np.mean([row["reward"] for row in recent]))
            success_rate = float(np.mean([row["finished"] for row in recent]))
            print(
                f"episode {episode:5d} | reward {mean_reward:9.3f} | "
                f"success {success_rate:6.1%} | epsilon {epsilon:.3f}"
            )

    training_log = pd.DataFrame(logs)
    learned_path, rollout_metrics = greedy_rollout(env, q_table)
    q_path = args.output_dir / "q_table.npy"
    learned_path_file = args.output_dir / "learned_path.csv"
    log_path = args.output_dir / "training_log.csv"
    metrics_path = args.output_dir / "metrics.json"
    plot_path = args.output_dir / "qlearn_debug_plot.png"
    np.save(q_path, q_table)
    learned_path.to_csv(learned_path_file, index=False)
    training_log.to_csv(log_path, index=False)

    metrics: Dict[str, object] = {
        **rollout_metrics,
        "path_csv": str(args.path_csv),
        "q_table_shape": list(q_shape),
        "q_table_size_mb": size_mb,
        "episodes": args.episodes,
        "training_success_rate_last_100": float(training_log["finished"].tail(100).mean()),
        "training_mean_reward_last_100": float(training_log["reward"].tail(100).mean()),
        "grid": {
            "spacing": args.grid_spacing,
            "margin": args.margin,
            "x_cells": len(env.x_values),
            "y_cells": len(env.y_values),
        },
        "reward_weights": vars(env.weights),
        "hyperparameters": {
            "learning_rate": args.learning_rate,
            "discount": args.discount,
            "epsilon_start": args.epsilon_start,
            "epsilon_end": args.epsilon_end,
            "epsilon_decay": args.epsilon_decay,
            "max_steps": args.max_steps,
            "finish_tolerance": env.finish_tolerance,
            "seed": args.seed,
        },
    }
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    create_debug_plot(args.path_csv, learned_path_file, log_path, plot_path)
    print(f"Saved debug artifacts to {args.output_dir}")


if __name__ == "__main__":
    main()
