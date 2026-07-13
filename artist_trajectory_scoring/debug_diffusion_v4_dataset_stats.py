"""Print dataset and normalization statistics for diffusion v4 debugging."""

import argparse
from pathlib import Path
from typing import Iterable, Optional

import numpy as np


CONDITION_KEYS = (
    "condition_features_norm",
    "condition_norm",
    "conditions_norm",
    "cond_norm",
    "normalized_condition",
    "normalized_conditions",
    "condition_features",
    "condition",
    "conditions",
    "cond",
    "X",
    "x",
)
TARGET_KEYS = (
    "target_norm",
    "targets_norm",
    "delta_q_norm",
    "normalized_target",
    "normalized_targets",
    "target",
    "targets",
    "delta_q",
    "y",
)
Q_START_KEYS = ("q_start", "q_starts", "start_q", "q0")
EXPERT_Q_KEYS = ("expert_q", "q", "joint_trajectory", "joint_trajectories")
DELTA_Q_KEYS = ("delta_q", "target", "targets", "y")


def pick_key(keys: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    key_set = set(keys)
    for key in candidates:
        if key in key_set:
            return key
    return None


def describe_array(name: str, arr: np.ndarray) -> None:
    arr = np.asarray(arr)
    if not np.issubdtype(arr.dtype, np.number):
        print(f"{name}: shape={arr.shape} dtype={arr.dtype}")
        return

    numeric = arr.astype(np.float64)
    print(
        f"{name}: shape={arr.shape} dtype={arr.dtype} "
        f"min={float(np.min(numeric)):.6f} max={float(np.max(numeric)):.6f} "
        f"mean={float(np.mean(numeric)):.6f} std={float(np.std(numeric)):.6f}"
    )
    if arr.ndim >= 2:
        flat = numeric.reshape(-1, numeric.shape[-1])
        mean = np.mean(flat, axis=0)
        std = np.std(flat, axis=0)
        mins = np.min(flat, axis=0)
        maxs = np.max(flat, axis=0)
        for idx, (vmin, vmax, vmean, vstd) in enumerate(zip(mins, maxs, mean, std), start=1):
            print(
                f"  dim{idx}: min={float(vmin): .6f} max={float(vmax): .6f} "
                f"mean={float(vmean): .6f} std={float(vstd): .6f}"
            )


def looks_normalized(arr: np.ndarray) -> str:
    numeric = np.asarray(arr, dtype=np.float64)
    mean = float(np.mean(numeric))
    std = float(np.std(numeric))
    max_abs_mean = abs(mean)
    if max_abs_mean < 0.1 and 0.5 <= std <= 1.5:
        return "yes-ish: global mean near 0 and std near 1"
    return "no-ish: global mean/std are not close to standard normal"


def inspect_npz(path: Path) -> None:
    print(f"\n{path}")
    with np.load(path, allow_pickle=True) as data:
        keys = sorted(data.keys())
        print("keys:", keys)

        condition_key = pick_key(keys, CONDITION_KEYS)
        target_key = pick_key(keys, TARGET_KEYS)
        q_start_key = pick_key(keys, Q_START_KEYS)
        expert_q_key = pick_key(keys, EXPERT_Q_KEYS)
        delta_q_key = pick_key(keys, DELTA_Q_KEYS)

        if condition_key is not None:
            condition = np.asarray(data[condition_key])
            describe_array(f"condition[{condition_key}]", condition)
            print("condition appears normalized:", looks_normalized(condition))
        else:
            print("condition: missing")

        if target_key is not None:
            target = np.asarray(data[target_key])
            describe_array(f"target[{target_key}]", target)
            print("target appears normalized:", looks_normalized(target))
        else:
            print("target: missing")

        if q_start_key is not None:
            describe_array(f"q_start source[{q_start_key}]", np.asarray(data[q_start_key]))
        elif expert_q_key is not None:
            expert_q = np.asarray(data[expert_q_key])
            describe_array(f"q_start source[{expert_q_key}[:, 0, :]]", expert_q[:, 0, :])
        elif delta_q_key is not None:
            delta_q = np.asarray(data[delta_q_key])
            zeros = np.zeros((delta_q.shape[0], delta_q.shape[-1]), dtype=np.float32)
            describe_array("q_start source[implicit zeros from delta_q]", zeros)
        else:
            print("q_start source: missing")

        if expert_q_key is not None:
            describe_array(f"expert_q[{expert_q_key}]", np.asarray(data[expert_q_key]))
        else:
            print("expert_q: missing")

        if delta_q_key is not None:
            describe_array(f"delta_q[{delta_q_key}]", np.asarray(data[delta_q_key]))
        else:
            print("delta_q: missing")

        for mean_key, std_key in (
            ("condition_mean", "condition_std"),
            ("cond_mean", "cond_std"),
            ("target_mean", "target_std"),
            ("delta_q_mean", "delta_q_std"),
            ("y_mean", "y_std"),
        ):
            if mean_key in data or std_key in data:
                if mean_key in data:
                    describe_array(mean_key, np.asarray(data[mean_key]))
                if std_key in data:
                    describe_array(std_key, np.asarray(data[std_key]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", default="data/cartesian_expert_dataset_v3/diffusion_v2")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    npz_files = sorted(dataset_dir.glob("*.npz"))
    print(f"dataset_dir: {dataset_dir}")
    print("available .npz files:")
    for path in npz_files:
        print(f"  {path.name}")

    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in {dataset_dir}")

    for path in npz_files:
        inspect_npz(path)


if __name__ == "__main__":
    main()
