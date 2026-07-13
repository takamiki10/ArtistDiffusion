"""Sample and evaluate v4 conditional U-Net diffusion trajectories."""

import argparse
import csv
import importlib
import json
import random
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

from conditional_unet1d_artist import ConditionalUnet1DArtist

try:
    import trajectory_costs as tc
except Exception:
    tc = None


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
RAW_CONDITION_KEYS = ("condition_features", "condition", "conditions", "cond", "X", "x")
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
RAW_TARGET_KEYS = ("target", "targets", "delta_q", "y")
Q_START_KEYS = ("q_start", "q_starts", "start_q", "q0")
DESIRED_PATH_KEYS = ("desired_paths", "desired_path", "desired_xyz", "xyz", "cartesian_path", "path_xyz")
TIME_KEYS = ("t", "time", "times")
EXPERT_Q_KEYS = ("expert_q", "q", "joint_trajectory", "joint_trajectories")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_key(keys: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    key_set = set(keys)
    for key in candidates:
        if key in key_set:
            return key
    return None


def find_test_file(dataset_dir: Path) -> Path:
    npz_files = sorted(dataset_dir.glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in {dataset_dir}")
    for needle in ("test", "val", "valid"):
        match = next((p for p in npz_files if needle in p.stem.lower()), None)
        if match is not None:
            return match
    return npz_files[0]


def to_array(data: np.lib.npyio.NpzFile, key: Optional[str]) -> Optional[np.ndarray]:
    if key is None:
        return None
    return np.asarray(data[key], dtype=np.float32)


def summarize_array(name: str, arr: np.ndarray) -> None:
    arr = np.asarray(arr, dtype=np.float32)
    if arr.ndim == 1:
        print(
            f"{name}: shape={arr.shape} min={float(np.min(arr)):.6f} "
            f"max={float(np.max(arr)):.6f} mean={float(np.mean(arr)):.6f} std={float(np.std(arr)):.6f}"
        )
        return

    flat = arr.reshape(-1, arr.shape[-1])
    mins = np.min(flat, axis=0)
    maxs = np.max(flat, axis=0)
    means = np.mean(flat, axis=0)
    stds = np.std(flat, axis=0)
    print(f"{name}: shape={arr.shape}")
    for idx, (vmin, vmax, mean, std) in enumerate(zip(mins, maxs, means, stds), start=1):
        print(
            f"  dim{idx}: min={float(vmin): .6f} max={float(vmax): .6f} "
            f"mean={float(mean): .6f} std={float(std): .6f}"
        )


def summarize_tensor(name: str, tensor: torch.Tensor) -> None:
    summarize_array(name, tensor.detach().cpu().numpy())


class DiffusionV2TestSet:
    def __init__(self, dataset_dir: Path, checkpoint_norm: Dict[str, object]) -> None:
        self.path = find_test_file(dataset_dir)
        with np.load(self.path, allow_pickle=True) as data:
            print(f"Loading {self.path}")
            print("Available npz keys:", sorted(data.keys()))
            keys = list(data.keys())
            checkpoint_cond_key = checkpoint_norm.get("condition_key")
            checkpoint_target_key = checkpoint_norm.get("target_key")
            cond_key = checkpoint_cond_key if isinstance(checkpoint_cond_key, str) and checkpoint_cond_key in keys else None
            if cond_key is None:
                cond_key = pick_key(keys, CONDITION_KEYS)
            if cond_key is None:
                raise KeyError(f"Could not find condition key. Available keys: {keys}")
            target_key = checkpoint_target_key if isinstance(checkpoint_target_key, str) and checkpoint_target_key in keys else None
            if target_key is None:
                target_key = pick_key(keys, TARGET_KEYS)
            raw_cond_key = pick_key(keys, RAW_CONDITION_KEYS)
            raw_target_key = pick_key(keys, RAW_TARGET_KEYS)

            self.cond = np.asarray(data[cond_key], dtype=np.float32)
            self.raw_cond = to_array(data, raw_cond_key)
            self.raw_target = to_array(data, raw_target_key)
            self.target_norm = to_array(data, target_key)
            q_start_key = pick_key(keys, Q_START_KEYS)
            self.q_start = to_array(data, q_start_key)
            self.q_start_source = q_start_key if q_start_key is not None else ""
            self.desired_path = to_array(data, pick_key(keys, DESIRED_PATH_KEYS))
            self.times = to_array(data, pick_key(keys, TIME_KEYS))
            self.expert_q = to_array(data, pick_key(keys, EXPERT_Q_KEYS))

        if self.q_start is None:
            if self.expert_q is not None:
                self.q_start = self.expert_q[:, 0, :]
                self.q_start_source = "expert_q[:, 0, :]"
            elif self.raw_target is not None:
                self.q_start = np.zeros((self.raw_target.shape[0], self.raw_target.shape[-1]), dtype=np.float32)
                self.q_start_source = "implicit zeros from raw delta_q"
            else:
                raise ValueError(
                    "Could not recover unnormalized q_start. Expected q_start, expert_q, or raw delta_q."
                )

        self.target_mean = stat_array(checkpoint_norm, "target_mean", default=0.0)
        self.target_std = stat_array(checkpoint_norm, "target_std", default=1.0)
        self.target_norm_source = "checkpoint normalization target_mean/target_std"

    def __len__(self) -> int:
        return int(self.cond.shape[0])


def stat_array(norm: Dict[str, object], key: str, default: float) -> np.ndarray:
    value = norm.get(key)
    if value is None:
        return np.asarray(default, dtype=np.float32)
    arr = np.asarray(value, dtype=np.float32)
    while arr.ndim > 1 and arr.shape[0] == 1:
        arr = np.squeeze(arr, axis=0)
    return arr


def make_beta_schedule(num_steps: int, device: torch.device) -> Dict[str, torch.Tensor]:
    betas = torch.linspace(1e-4, 0.02, num_steps, device=device)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return {
        "betas": betas,
        "alphas": alphas,
        "alpha_bars": alpha_bars,
        "sqrt_recip_alphas": torch.sqrt(1.0 / alphas),
        "sqrt_one_minus_alpha_bars": torch.sqrt(1.0 - alpha_bars),
    }


@torch.no_grad()
def sample_ddpm(
    model: ConditionalUnet1DArtist,
    cond: torch.Tensor,
    schedule: Dict[str, torch.Tensor],
    action_dim: int,
    debug: bool = False,
) -> torch.Tensor:
    batch_size, horizon, _ = cond.shape
    x = torch.randn(batch_size, horizon, action_dim, device=cond.device)
    if debug:
        summarize_tensor("initial_gaussian_noise", x)
    num_steps = int(schedule["betas"].shape[0])
    for step in reversed(range(num_steps)):
        t = torch.full((batch_size,), step, device=cond.device, dtype=torch.long)
        pred_noise = model(x, cond, t)
        beta_t = schedule["betas"][step]
        alpha_bar_t = schedule["alpha_bars"][step]
        x = schedule["sqrt_recip_alphas"][step] * (
            x - beta_t / schedule["sqrt_one_minus_alpha_bars"][step] * pred_noise
        )
        if step > 0:
            alpha_bar_prev = schedule["alpha_bars"][step - 1]
            posterior_var = beta_t * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)
            x = x + torch.sqrt(torch.clamp(posterior_var, min=1e-20)) * torch.randn_like(x)
    if debug:
        summarize_tensor("final_normalized_sample", x)
    return x


def unnormalize_delta(delta_norm: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return delta_norm * std.reshape((1,) * (delta_norm.ndim - std.ndim) + std.shape) + mean.reshape(
        (1,) * (delta_norm.ndim - mean.ndim) + mean.shape
    )


def assert_checkpoint_matches_dataset(checkpoint_norm: Dict[str, object], dataset: DiffusionV2TestSet) -> None:
    condition_key = checkpoint_norm.get("condition_key")
    target_key = checkpoint_norm.get("target_key")
    if condition_key is not None:
        print(f"Checkpoint condition_key: {condition_key}")
    if target_key is not None:
        print(f"Checkpoint target_key: {target_key}")
    print("Sampling condition source: dataset.cond loaded with normalized-key priority")
    summarize_array("sample_time_condition_features", dataset.cond)
    if dataset.target_norm is not None:
        summarize_array("expert_delta_q_norm", dataset.target_norm)
    if dataset.raw_target is not None:
        summarize_array("expert_delta_q_unnormalized", dataset.raw_target)
    if dataset.expert_q is not None:
        summarize_array("expert_q_unnormalized", dataset.expert_q)
    summarize_array("q_start_unnormalized", dataset.q_start)
    print(f"q_start source: {dataset.q_start_source}")
    print(f"target normalization source: {dataset.target_norm_source}")
    summarize_array("target_mean_used_for_delta_q", dataset.target_mean)
    summarize_array("target_std_used_for_delta_q", dataset.target_std)


def write_q_csv(path: Path, q: np.ndarray, times: Optional[np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    q = np.asarray(q, dtype=np.float32)
    if q.ndim != 2 or q.shape[1] != 6:
        raise ValueError(f"Expected q trajectory with shape (T, 6), got {q.shape}")
    if times is None:
        times = np.arange(q.shape[0], dtype=np.float32)
    times = np.asarray(times).reshape(-1)
    if times.shape[0] != q.shape[0]:
        raise ValueError(f"Expected {q.shape[0]} time values, got {times.shape[0]}")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["t", "q1", "q2", "q3", "q4", "q5", "q6"])
        for i in range(q.shape[0]):
            writer.writerow([float(times[i])] + [float(v) for v in q[i]])


def joint_velocity_cost(q: np.ndarray) -> float:
    if q.shape[0] < 2:
        return 0.0
    return float(np.mean(np.diff(q, axis=0) ** 2))


def joint_acceleration_cost(q: np.ndarray) -> float:
    if q.shape[0] < 3:
        return 0.0
    return float(np.mean(np.diff(q, n=2, axis=0) ** 2))


def joint_jerk_cost(q: np.ndarray) -> float:
    if q.shape[0] < 4:
        return 0.0
    return float(np.mean(np.diff(q, n=3, axis=0) ** 2))


def call_cost_function(names: Iterable[str], *args) -> Optional[float]:
    if tc is None:
        return None
    for name in names:
        fn = getattr(tc, name, None)
        if callable(fn):
            try:
                value = fn(*args)
                return float(value)
            except Exception:
                continue
    return None


def try_existing_fk(q: np.ndarray) -> Optional[np.ndarray]:
    """Use an existing project FK helper if one is importable.

    This deliberately does not implement or replace the xMateCR7 FK system here.
    Existing helpers should internally use robot.update_cfg(cfg) and
    robot.get_transform(frame_to=ee_link), not robot.link_fk(...).
    """

    try:
        import score_trajectory

        fk = getattr(score_trajectory, "rokae_forward_kinematics", None)
        if callable(fk):
            xyz = np.asarray(fk(q), dtype=np.float32)
            if xyz.ndim == 2 and xyz.shape[0] == q.shape[0] and xyz.shape[1] >= 3:
                return xyz[:, -3:]
    except Exception:
        pass

    module_names = (
        "fk_utils",
        "xmate_fk_utils",
        "trajectory_fk",
        "evaluate_diffusion_trajectory_v2",
        "evaluate_diffusion_trajectory_v3",
        "adaptive_refine_mlp_predictions_with_ik",
    )
    function_names = (
        "compute_fk_path",
        "forward_kinematics_path",
        "joint_trajectory_to_cartesian",
        "q_to_xyz_path",
        "evaluate_fk_path",
        "compute_cartesian_path",
    )
    for module_name in module_names:
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        for function_name in function_names:
            fn = getattr(module, function_name, None)
            if not callable(fn):
                continue
            try:
                xyz = np.asarray(fn(q), dtype=np.float32)
            except Exception:
                continue
            if xyz.ndim == 2 and xyz.shape[0] == q.shape[0] and xyz.shape[1] >= 3:
                return xyz[:, -3:]
    return None


def evaluate_basic(
    q: np.ndarray,
    desired_path: Optional[np.ndarray],
    pred_xyz: Optional[np.ndarray],
    accept_mean_error: float,
    accept_max_error: float,
) -> Dict[str, float]:
    metrics: Dict[str, float] = {
        "joint_velocity_cost": joint_velocity_cost(q),
        "joint_acceleration_cost": joint_acceleration_cost(q),
        "joint_jerk_cost": joint_jerk_cost(q),
    }

    if desired_path is not None and pred_xyz is not None:
        desired_xyz = desired_path[..., -3:] if desired_path.shape[-1] >= 3 else desired_path
        err = np.linalg.norm(pred_xyz - desired_xyz, axis=-1)
        weighted = call_cost_function(
            ("weighted_xyz_loss", "weighted_cartesian_loss", "cartesian_weighted_loss"),
            pred_xyz,
            desired_xyz,
        )
        path_error = call_cost_function(
            ("cartesian_path_error", "path_error", "compute_cartesian_path_error"),
            pred_xyz,
            desired_xyz,
        )
        metrics.update(
            {
                "weighted_cartesian_loss": float(weighted) if weighted is not None else float(np.mean(err**2)),
                "path_error": float(path_error) if path_error is not None else float(np.mean(err**2)),
                "mean_cartesian_error": float(np.mean(err)),
                "max_cartesian_error": float(np.max(err)),
            }
        )
    else:
        metrics.update(
            {
                "weighted_cartesian_loss": float("nan"),
                "path_error": float("nan"),
                "mean_cartesian_error": float("nan"),
                "max_cartesian_error": float("nan"),
            }
        )

    metrics["accepted"] = bool(
        np.isfinite(metrics["mean_cartesian_error"])
        and metrics["mean_cartesian_error"] <= accept_mean_error
        and metrics["max_cartesian_error"] <= accept_max_error
    )
    return metrics


def maybe_plot(path: Path, q: np.ndarray, desired_path: Optional[np.ndarray]) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(q)
    ax.set_xlabel("timestep")
    ax.set_ylabel("joint angle")
    ax.set_title("Diffusion v4 predicted q")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="data/cartesian_expert_dataset_v3/diffusion_v4_unet/best_model.pt")
    parser.add_argument("--dataset_dir", default="data/cartesian_expert_dataset_v3/diffusion_v2")
    parser.add_argument("--max_paths", type=int, default=83)
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--output_dir", default="data/cartesian_expert_dataset_v3/diffusion_v4_unet/samples_single")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--accept_mean_error", type=float, default=0.01)
    parser.add_argument("--accept_max_error", type=float, default=0.03)
    parser.add_argument("--print_diagnostics", action="store_true", help="Print dataset-wide normalization diagnostics.")
    parser.add_argument(
        "--no_range_diagnostics",
        action="store_true",
        help="Disable per-path generated/expert q and delta_q range diagnostics.",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    print(f"checkpoint path: {args.checkpoint}")
    print(f"checkpoint epoch: {checkpoint.get('epoch')}")
    print(f"checkpoint best_validation_loss: {checkpoint.get('best_validation_loss')}")
    model_config = checkpoint.get("model_config", {"action_dim": 6, "cond_dim": 13, "hidden_dim": 256})
    diffusion_config = checkpoint.get("diffusion_config", {"num_diffusion_steps": 100})
    model = ConditionalUnet1DArtist(**model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dataset = DiffusionV2TestSet(Path(args.dataset_dir), checkpoint.get("normalization", {}))
    print(f"target normalization source: {dataset.target_norm_source}")
    summarize_array("target_mean", dataset.target_mean)
    summarize_array("target_std", dataset.target_std)
    summarize_array("condition_range_used_for_sampling", dataset.cond)
    print(f"q_start source: {dataset.q_start_source}")
    if args.print_diagnostics:
        assert_checkpoint_matches_dataset(checkpoint.get("normalization", {}), dataset)
    schedule = make_beta_schedule(int(diffusion_config.get("num_diffusion_steps", 100)), device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_metrics: List[Dict[str, float]] = []
    generation_times: List[float] = []
    count = min(args.max_paths, len(dataset))

    for path_idx in range(count):
        cond_np = dataset.cond[path_idx : path_idx + 1]
        cond = torch.from_numpy(cond_np).to(device=device, dtype=torch.float32)
        path_dir = output_dir / f"path_{path_idx:03d}"
        path_dir.mkdir(parents=True, exist_ok=True)

        best_metrics: Optional[Dict[str, float]] = None
        best_q: Optional[np.ndarray] = None
        for sample_idx in range(args.num_samples):
            start = time.perf_counter()
            delta_norm = sample_ddpm(
                model,
                cond,
                schedule,
                int(model_config.get("action_dim", 6)),
                debug=not args.no_range_diagnostics and path_idx == 0 and sample_idx == 0,
            )
            generation_times.append(time.perf_counter() - start)

            delta_np = delta_norm.squeeze(0).detach().cpu().numpy()
            delta_q = unnormalize_delta(delta_np, dataset.target_mean, dataset.target_std)
            q_start = np.asarray(dataset.q_start[path_idx], dtype=np.float32).reshape(1, -1)
            q = q_start + delta_q
            if not args.no_range_diagnostics and sample_idx == 0:
                summarize_array(f"path_{path_idx:03d}_final_normalized_sample", delta_np)
                summarize_array(f"path_{path_idx:03d}_generated_delta_q", delta_q)
                summarize_array(f"path_{path_idx:03d}_generated_q", q)
                if dataset.raw_target is not None:
                    summarize_array(f"path_{path_idx:03d}_expert_delta_q", dataset.raw_target[path_idx])
                if dataset.expert_q is not None:
                    summarize_array(f"path_{path_idx:03d}_expert_q", dataset.expert_q[path_idx])

            desired = None if dataset.desired_path is None else dataset.desired_path[path_idx]
            pred_xyz = try_existing_fk(q)
            metrics = evaluate_basic(q, desired, pred_xyz, args.accept_mean_error, args.accept_max_error)
            metrics["path_index"] = path_idx
            metrics["sample_index"] = sample_idx
            metrics["generation_time_sec"] = generation_times[-1]

            if best_metrics is None or metrics["joint_velocity_cost"] < best_metrics["joint_velocity_cost"]:
                best_metrics = metrics
                best_q = q

        assert best_metrics is not None and best_q is not None
        times = None if dataset.times is None else dataset.times[path_idx]
        write_q_csv(path_dir / "diffusion_v4_pred_q.csv", best_q, times)
        maybe_plot(path_dir / "plot.png", best_q, None)
        (path_dir / "metrics.json").write_text(json.dumps(best_metrics, indent=2), encoding="utf-8")
        all_metrics.append(best_metrics)
        print(
            f"path {path_idx:03d}: accepted={best_metrics['accepted']} "
            f"mean_cartesian_error={best_metrics['mean_cartesian_error']} "
            f"max_cartesian_error={best_metrics['max_cartesian_error']}"
        )

    summary = summarize(all_metrics, generation_times)
    (output_dir / "summary_metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Summary")
    for key, value in summary.items():
        print(f"{key}: {value}")


def finite_mean(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(arr.mean())


def finite_max(values: Iterable[float]) -> float:
    arr = np.asarray(list(values), dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(arr.max())


def summarize(metrics: List[Dict[str, float]], generation_times: List[float]) -> Dict[str, float]:
    return {
        "evaluated_count": len(metrics),
        "accepted_count": int(sum(bool(m.get("accepted", False)) for m in metrics)),
        "mean_path_error": finite_mean(m["path_error"] for m in metrics),
        "mean_cartesian_error": finite_mean(m["mean_cartesian_error"] for m in metrics),
        "mean_max_cartesian_error": finite_mean(m["max_cartesian_error"] for m in metrics),
        "worst_max_cartesian_error": finite_max(m["max_cartesian_error"] for m in metrics),
        "mean_joint_velocity_cost": finite_mean(m["joint_velocity_cost"] for m in metrics),
        "mean_joint_acceleration_cost": finite_mean(m["joint_acceleration_cost"] for m in metrics),
        "mean_generation_time_sec": finite_mean(generation_times),
    }


if __name__ == "__main__":
    main()
