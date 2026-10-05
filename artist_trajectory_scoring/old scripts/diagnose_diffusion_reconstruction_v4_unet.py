#!/usr/bin/env python3
"""Diagnose v4 diffusion reconstruction without conflating math and model error.

Modes:
  identity: predicted x0 is exactly dataset delta_q_norm.
  oracle:   DDPM forward process is inverted with the exact sampled epsilon.
  model:    U-Net predicts epsilon from x_t, timestep, and condition_features_norm.

The target is always delta_q_norm. Raw joint trajectories are reconstructed as:
  q = q_start[:, None, :] + unnormalize(delta_q_norm)
"""

from __future__ import annotations

import argparse
import importlib
import inspect
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch


DEFAULT_DATASET_DIR = Path("data/cartesian_expert_dataset_v3/diffusion_v2")
STAT_NAME_PAIRS = (
    ("delta_q_mean", "delta_q_std"),
    ("target_mean", "target_std"),
    ("action_mean", "action_std"),
    ("y_mean", "y_std"),
)
MODEL_MODULES = (
    "sample_conditional_diffusion_trajectory_v4_unet",
    "train_conditional_diffusion_trajectory_v4_unet",
)
SUMMARY_THRESHOLD = 1e-5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Diagnose v4 diffusion reconstruction in identity/oracle/model modes."
    )
    parser.add_argument("--checkpoint", type=Path, default=None, help="v4 U-Net checkpoint for --mode model")
    parser.add_argument(
        "--dataset_dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help=f"Directory containing diffusion_train_v2.npz and diffusion_test_v2.npz. Default: {DEFAULT_DATASET_DIR}",
    )
    parser.add_argument("--split", choices=("test", "train"), default="test")
    parser.add_argument("--timesteps", default="0,10,25,50,75,99")
    parser.add_argument("--max_paths", type=int, default=None)
    parser.add_argument("--mode", choices=("identity", "oracle", "model"), default="model")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--num_diffusion_steps", type=int, default=100)
    return parser.parse_args()


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Missing dataset file: {path}")
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def split_path(dataset_dir: Path, split: str) -> Path:
    return dataset_dir / f"diffusion_{split}_v2.npz"


def print_keys_and_shapes(label: str, data: Dict[str, np.ndarray]) -> None:
    print(f"\n[{label}] keys and shapes")
    for key in sorted(data.keys()):
        value = data[key]
        print(f"  {key}: shape={value.shape}, dtype={value.dtype}")


def subset_data(data: Dict[str, np.ndarray], max_paths: Optional[int]) -> Dict[str, np.ndarray]:
    if max_paths is None:
        return data
    if max_paths <= 0:
        raise ValueError("--max_paths must be positive")
    out: Dict[str, np.ndarray] = {}
    for key, value in data.items():
        if value.ndim > 0 and value.shape[0] >= max_paths:
            out[key] = value[:max_paths]
        else:
            out[key] = value
    return out


def require_keys(data: Dict[str, np.ndarray], keys: Sequence[str], context: str) -> None:
    missing = [key for key in keys if key not in data]
    if missing:
        raise KeyError(f"{context} missing required key(s): {', '.join(missing)}")


def as_float64(data: Dict[str, np.ndarray], key: str) -> np.ndarray:
    return np.asarray(data[key], dtype=np.float64)


def path_names(data: Dict[str, np.ndarray], count: int) -> List[str]:
    if "path_names" not in data:
        return [f"path_{idx}" for idx in range(count)]
    raw = np.asarray(data["path_names"])
    names: List[str] = []
    for idx in range(count):
        item = raw[idx]
        names.append(item.decode("utf-8", errors="replace") if isinstance(item, bytes) else str(item))
    return names


def parse_timesteps(raw: str, num_steps: int) -> List[int]:
    values: List[int] = []
    for piece in raw.split(","):
        piece = piece.strip()
        if not piece:
            continue
        timestep = int(piece)
        if timestep < 0 or timestep >= num_steps:
            raise ValueError(f"timestep {timestep} is outside [0, {num_steps - 1}]")
        values.append(timestep)
    if not values:
        raise ValueError("--timesteps must include at least one timestep")
    return values


def rmse(error: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(error))))


def max_abs(error: np.ndarray) -> float:
    return float(np.max(np.abs(error)))


def worst_path_name(error: np.ndarray, names: Sequence[str]) -> str:
    per_path = np.sqrt(np.mean(np.square(error), axis=tuple(range(1, error.ndim))))
    return names[int(np.argmax(per_path))]


def print_q_metrics(label: str, q_recon: np.ndarray, expert_q: np.ndarray, names: Sequence[str]) -> Tuple[float, float, bool]:
    assert q_recon.shape == expert_q.shape, f"{label}: q_recon {q_recon.shape} != expert_q {expert_q.shape}"
    error = q_recon - expert_q
    q_rmse = rmse(error)
    q_max = max_abs(error)
    passed = q_max <= SUMMARY_THRESHOLD
    status = "PASS" if passed else "FAIL"
    print(
        f"[{label}] {status}: q RMSE={q_rmse:.12e}, max q error={q_max:.12e}, "
        f"worst path={worst_path_name(error, names)}"
    )
    return q_rmse, q_max, passed


def q_error_metrics(q_recon: np.ndarray, expert_q: np.ndarray) -> Tuple[float, float]:
    assert q_recon.shape == expert_q.shape, f"q_recon {q_recon.shape} != expert_q {expert_q.shape}"
    error = q_recon - expert_q
    return rmse(error), max_abs(error)


def find_stats(data: Dict[str, np.ndarray]) -> Optional[Tuple[np.ndarray, np.ndarray, str]]:
    for mean_key, std_key in STAT_NAME_PAIRS:
        if mean_key in data and std_key in data:
            return (
                np.asarray(data[mean_key], dtype=np.float64),
                np.asarray(data[std_key], dtype=np.float64),
                f"{mean_key}/{std_key}",
            )
    return None


def get_delta_stats(selected: Dict[str, np.ndarray], train: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray, str]:
    stats = find_stats(selected)
    if stats is not None:
        return stats
    stats = find_stats(train)
    if stats is not None:
        return stats
    require_keys(train, ("delta_q",), "train split for recomputed normalization stats")
    delta_q = as_float64(train, "delta_q")
    return delta_q.mean(axis=(0, 1)), delta_q.std(axis=(0, 1)), "recomputed from train delta_q"


def stat_for_broadcast(stat: np.ndarray, target_ndim: int) -> np.ndarray:
    value = np.squeeze(np.asarray(stat, dtype=np.float64))
    while value.ndim < target_ndim:
        value = value[None, ...]
    return value


def unnormalize_delta_q(delta_q_norm: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    mean_b = stat_for_broadcast(mean, delta_q_norm.ndim)
    std_b = stat_for_broadcast(std, delta_q_norm.ndim)
    safe_std = np.where(np.abs(std_b) < 1e-12, 1.0, std_b)
    out = delta_q_norm * safe_std + mean_b
    assert out.shape == delta_q_norm.shape, f"unnormalized shape {out.shape} != normalized shape {delta_q_norm.shape}"
    return out


def reconstruct_q(q_start: np.ndarray, pred_x0: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    assert q_start.ndim == 2, f"q_start must be (B, C), got {q_start.shape}"
    assert pred_x0.ndim == 3, f"pred_x0 must be (B, T, C), got {pred_x0.shape}"
    assert q_start.shape[0] == pred_x0.shape[0], f"B mismatch: q_start {q_start.shape}, pred_x0 {pred_x0.shape}"
    assert q_start.shape[1] == pred_x0.shape[2], f"C mismatch: q_start {q_start.shape}, pred_x0 {pred_x0.shape}"
    delta_q = unnormalize_delta_q(pred_x0, mean, std)
    q_recon = q_start[:, None, :] + delta_q
    assert q_recon.shape == pred_x0.shape, f"q_recon shape {q_recon.shape} != pred_x0 {pred_x0.shape}"
    return q_recon


def print_joint_std(delta_q: np.ndarray, std: np.ndarray) -> None:
    selected_std = delta_q.std(axis=(0, 1))
    stored_std = np.squeeze(std)
    print("\n[normalization] per-joint std")
    for idx, value in enumerate(selected_std):
        stored = stored_std[idx] if idx < stored_std.shape[0] else float("nan")
        marker = "  <-- q6" if idx == 5 else ""
        print(f"  q{idx + 1}: stats std={stored:.12e}, selected raw std={value:.12e}{marker}")


def beta_schedule(num_steps: int, device: torch.device) -> torch.Tensor:
    for module_name in MODEL_MODULES:
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        for fn_name in ("get_beta_schedule", "make_beta_schedule", "linear_beta_schedule", "create_beta_schedule"):
            fn = getattr(module, fn_name, None)
            if not callable(fn):
                continue
            for kwargs in ({"num_steps": num_steps}, {"timesteps": num_steps}, {}):
                try:
                    betas = fn(**kwargs) if kwargs else fn(num_steps)
                except Exception:
                    continue
                betas = torch.as_tensor(betas, dtype=torch.float32, device=device).reshape(-1)
                if betas.numel() == num_steps:
                    print(f"[schedule] source: {module_name}.{fn_name}")
                    return betas
    print("[schedule] source: mirrored linear beta schedule 1e-4..2e-2")
    return torch.linspace(1e-4, 2e-2, num_steps, dtype=torch.float32, device=device)


def alpha_bars(num_steps: int, device: torch.device) -> torch.Tensor:
    betas = beta_schedule(num_steps, device)
    assert betas.shape == (num_steps,), f"betas must be ({num_steps},), got {betas.shape}"
    alphas = 1.0 - betas
    return torch.cumprod(alphas, dim=0)


def import_model_modules() -> List[Any]:
    modules = []
    for module_name in MODEL_MODULES:
        try:
            modules.append(importlib.import_module(module_name))
        except Exception as exc:
            print(f"[model] could not import {module_name}: {exc}")
    return modules


def try_sample_loader(
    modules: Sequence[Any],
    checkpoint: Path,
    device: torch.device,
    condition_dim: int,
    trajectory_dim: int,
    horizon: int,
    num_steps: int,
) -> Optional[torch.nn.Module]:
    loader_names = (
        "load_model",
        "load_checkpoint",
        "load_diffusion_model",
        "load_trained_model",
        "build_model_from_checkpoint",
    )
    for module in modules:
        for name in loader_names:
            fn = getattr(module, name, None)
            if not callable(fn):
                continue
            attempts = (
                (checkpoint, device),
                (str(checkpoint), device),
                (checkpoint, device, condition_dim, trajectory_dim, horizon),
                (str(checkpoint), device, condition_dim, trajectory_dim, horizon),
                (checkpoint, device, condition_dim, trajectory_dim, horizon, num_steps),
                (str(checkpoint), device, condition_dim, trajectory_dim, horizon, num_steps),
            )
            for args in attempts:
                try:
                    loaded = fn(*args)
                except Exception:
                    continue
                model = loaded[0] if isinstance(loaded, tuple) else loaded
                if isinstance(model, torch.nn.Module):
                    print(f"[model] loaded with {module.__name__}.{name}")
                    return model.to(device).eval()
    return None


def likely_model_classes(modules: Sequence[Any]) -> Iterable[type]:
    preferred_names = (
        "ConditionalUNet1D",
        "ConditionalUnet1D",
        "ConditionalTrajectoryUNet",
        "TrajectoryUNet",
        "UNet1D",
        "ConditionalUNet",
    )
    yielded = set()
    for module in modules:
        for name in preferred_names:
            cls = getattr(module, name, None)
            if inspect.isclass(cls) and issubclass(cls, torch.nn.Module):
                yielded.add(cls)
                yield cls
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if cls in yielded:
                continue
            if issubclass(cls, torch.nn.Module) and ("UNet" in cls.__name__ or "Unet" in cls.__name__):
                yielded.add(cls)
                yield cls


def instantiate_model_class(
    cls: type,
    condition_dim: int,
    trajectory_dim: int,
    horizon: int,
    num_steps: int,
) -> Optional[torch.nn.Module]:
    attempts = (
        {"condition_dim": condition_dim, "trajectory_dim": trajectory_dim, "horizon": horizon},
        {"cond_dim": condition_dim, "input_dim": trajectory_dim, "horizon": horizon},
        {"condition_dim": condition_dim, "out_dim": trajectory_dim, "horizon": horizon},
        {"condition_dim": condition_dim, "trajectory_dim": trajectory_dim, "num_timesteps": num_steps},
        {"condition_dim": condition_dim, "input_channels": trajectory_dim},
        {"cond_dim": condition_dim, "channels": trajectory_dim},
        {"in_channels": trajectory_dim, "cond_dim": condition_dim},
        {"input_dim": trajectory_dim, "condition_dim": condition_dim},
        {},
    )
    for kwargs in attempts:
        try:
            return cls(**kwargs)
        except Exception:
            continue
    return None


def checkpoint_state_dict(raw: Any) -> Dict[str, torch.Tensor]:
    if isinstance(raw, dict):
        for key in ("model_state_dict", "state_dict", "model", "ema_model_state_dict"):
            value = raw.get(key)
            if isinstance(value, dict):
                return value
        if raw and all(torch.is_tensor(value) for value in raw.values()):
            return raw
    raise ValueError("Could not find model state_dict in checkpoint")


def strip_state_dict_prefix(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    prefixes = ("module.", "model.", "net.")
    out: Dict[str, torch.Tensor] = {}
    for key, value in state.items():
        new_key = key
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
                    changed = True
        out[new_key] = value
    return out


def load_model(
    checkpoint: Optional[Path],
    device: torch.device,
    condition_dim: int,
    trajectory_dim: int,
    horizon: int,
    num_steps: int,
) -> torch.nn.Module:
    if checkpoint is None:
        raise ValueError("--checkpoint is required for --mode model")
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    modules = import_model_modules()
    model = try_sample_loader(modules, checkpoint, device, condition_dim, trajectory_dim, horizon, num_steps)
    if model is not None:
        return model

    raw = torch.load(checkpoint, map_location=device)
    state = strip_state_dict_prefix(checkpoint_state_dict(raw))
    last_error: Optional[Exception] = None
    for cls in likely_model_classes(modules):
        candidate = instantiate_model_class(cls, condition_dim, trajectory_dim, horizon, num_steps)
        if candidate is None:
            continue
        try:
            candidate.load_state_dict(state, strict=True)
            print(f"[model] loaded {cls.__module__}.{cls.__name__} with strict=True")
            return candidate.to(device).eval()
        except Exception as exc:
            last_error = exc
        try:
            missing, unexpected = candidate.load_state_dict(state, strict=False)
            print(
                f"[model] loaded {cls.__module__}.{cls.__name__} with strict=False; "
                f"missing={len(missing)}, unexpected={len(unexpected)}"
            )
            return candidate.to(device).eval()
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Could not load v4 U-Net checkpoint with discovered classes. Last error: {last_error}")


def call_model(
    model: torch.nn.Module,
    x_model: torch.Tensor,
    t: torch.Tensor,
    cond: torch.Tensor,
    print_shapes: bool = True,
) -> torch.Tensor:
    if print_shapes:
        print(f"  model input x_t shape: {tuple(x_model.shape)}")
        print(f"  model condition shape: {tuple(cond.shape)}")
    attempts = (
        (x_model, t, cond),
        (x_model, cond, t),
        (x_model, t, cond, None),
    )
    last_error: Optional[Exception] = None
    for args in attempts:
        try:
            out = model(*args)
            if isinstance(out, tuple):
                out = out[0]
            if print_shapes:
                print(f"  model output shape: {tuple(out.shape)}")
            return out
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"Model forward failed for known signatures. Last error: {last_error}")


def condition_variants(cond_btd: torch.Tensor, prefer_bdt: bool) -> List[Tuple[torch.Tensor, str]]:
    if cond_btd.ndim == 2:
        return [(cond_btd, "(B, D)")]
    assert cond_btd.ndim == 3, f"condition must be (B, D) or (B, T, D), got {cond_btd.shape}"
    btd = (cond_btd, "(B, T, D)")
    bdt = (cond_btd.transpose(1, 2).contiguous(), "(B, D, T)")
    return [bdt, btd] if prefer_bdt else [btd, bdt]


def infer_model_layout(
    model: torch.nn.Module,
    x_t_btc: torch.Tensor,
    t: torch.Tensor,
    cond_btd: torch.Tensor,
) -> Tuple[bool, bool]:
    """Infer trajectory and condition layouts without accepting broadcasted output."""

    candidates: List[Tuple[bool, torch.Tensor, str, bool, torch.Tensor, str]] = []
    x_bct = x_t_btc.transpose(1, 2).contiguous()
    for cond_tensor, cond_label in condition_variants(cond_btd, prefer_bdt=True):
        candidates.append((True, x_bct, "(B, C, T)", cond_label == "(B, D, T)", cond_tensor, cond_label))
    for cond_tensor, cond_label in condition_variants(cond_btd, prefer_bdt=False):
        candidates.append((False, x_t_btc, "(B, T, C)", cond_label == "(B, D, T)", cond_tensor, cond_label))

    with torch.no_grad():
        for x_is_bct, x_model, x_label, cond_is_bdt, cond_model, cond_label in candidates:
            try:
                out = call_model(model, x_model, t, cond_model, print_shapes=False)
            except Exception:
                continue
            if tuple(out.shape) == tuple(x_model.shape):
                print(f"[model] inferred U-Net trajectory layout: {x_label}")
                print(f"[model] inferred condition layout: {cond_label}")
                return x_is_bct, cond_is_bdt

    raise RuntimeError(
        "Could not infer model input layout. Tried x as (B,C,T)/(B,T,C) and condition as "
        "(B,D,T)/(B,T,D) where applicable, but no call returned the same shape as x."
    )


def predict_epsilon(
    model: torch.nn.Module,
    x_t_btc: torch.Tensor,
    timestep: int,
    cond: torch.Tensor,
    layout: Optional[Tuple[bool, bool]],
) -> Tuple[torch.Tensor, Tuple[bool, bool]]:
    assert x_t_btc.ndim == 3, f"x_t must be (B, T, C), got {x_t_btc.shape}"
    assert cond.ndim in (2, 3), f"condition_features_norm must be (B, D) or (B, T, D), got {cond.shape}"
    assert x_t_btc.shape[0] == cond.shape[0], f"B mismatch: x_t {x_t_btc.shape}, cond {cond.shape}"
    if cond.ndim == 3:
        assert x_t_btc.shape[1] == cond.shape[1], f"T mismatch: x_t {x_t_btc.shape}, cond {cond.shape}"
    t = torch.full((x_t_btc.shape[0],), timestep, dtype=torch.long, device=x_t_btc.device)

    if layout is None:
        layout = infer_model_layout(model, x_t_btc, t, cond)

    expects_bct, condition_bdt = layout
    if expects_bct:
        x_model = x_t_btc.transpose(1, 2).contiguous()
        cond_model = cond.transpose(1, 2).contiguous() if cond.ndim == 3 and condition_bdt else cond
        out = call_model(model, x_model, t, cond_model)
        assert out.shape == x_model.shape, f"model output {out.shape} != model input {x_model.shape}"
        pred_epsilon = out.transpose(1, 2).contiguous()
    else:
        x_model = x_t_btc
        cond_model = cond.transpose(1, 2).contiguous() if cond.ndim == 3 and condition_bdt else cond
        out = call_model(model, x_model, t, cond_model)
        assert out.shape == x_t_btc.shape, f"model output {out.shape} != model input {x_t_btc.shape}"
        pred_epsilon = out

    assert pred_epsilon.shape == x_t_btc.shape, f"pred_epsilon {pred_epsilon.shape} != x_t {x_t_btc.shape}"
    return pred_epsilon, layout


def import_fk_callable() -> Tuple[Optional[Any], Optional[str]]:
    for module_name in ("score_trajectory", "validate_expert_dataset"):
        try:
            module = importlib.import_module(module_name)
        except Exception:
            continue
        for name in ("compute_fk_positions", "compute_fk_path", "forward_kinematics", "get_fk_positions", "fk_positions"):
            fn = getattr(module, name, None)
            if callable(fn):
                return fn, f"{module_name}.{name}"
    return None, None


def fk_positions_for(q: np.ndarray, data: Dict[str, np.ndarray]) -> Tuple[Optional[np.ndarray], Optional[str]]:
    fn, source = import_fk_callable()
    if fn is None:
        return None, None
    for args in ((q,), (q, data), (q, data.get("q_start"))):
        try:
            result = fn(*args)
        except Exception:
            continue
        arr = np.asarray(result, dtype=np.float64)
        if arr.ndim >= 2 and arr.shape[-1] >= 3:
            return arr[..., :3], source
    return None, source


def print_fk_metrics(label: str, data: Dict[str, np.ndarray], q_recon: np.ndarray) -> None:
    if "desired_paths" not in data:
        print(f"[{label}] FK: SKIP, desired_paths unavailable")
        return
    fk_pos, source = fk_positions_for(q_recon, data)
    if fk_pos is None:
        print(f"[{label}] FK: SKIP, no callable FK helper found" if source is None else f"[{label}] FK: SKIP, {source} could not run")
        return
    desired = np.asarray(data["desired_paths"], dtype=np.float64)[..., :3]
    if fk_pos.shape != desired.shape:
        print(f"[{label}] FK: SKIP, FK shape {fk_pos.shape} != desired_paths shape {desired.shape}")
        return
    cart_error = np.linalg.norm(fk_pos - desired, axis=-1)
    print(
        f"[{label}] FK: source={source}, mean Cartesian error={float(np.mean(cart_error)):.12e}, "
        f"max Cartesian error={float(np.max(cart_error)):.12e}"
    )


def maybe_print_accepted_count(label: str, q_recon: np.ndarray, expert_q: np.ndarray) -> None:
    error = np.max(np.abs(q_recon - expert_q), axis=(1, 2))
    accepted = int(np.sum(error <= SUMMARY_THRESHOLD))
    print(f"[{label}] accepted count @ max_q_error<={SUMMARY_THRESHOLD:g}: {accepted}/{error.shape[0]}")


def tensor_stats(value: torch.Tensor) -> Tuple[float, float]:
    return float(value.mean().detach().cpu()), float(value.std().detach().cpu())


def epsilon_metrics(eps_pred: torch.Tensor, eps_true: torch.Tensor) -> Dict[str, Any]:
    assert eps_pred.shape == eps_true.shape, f"eps_pred {eps_pred.shape} != eps_true {eps_true.shape}"
    diff = eps_pred - eps_true
    mse = float(torch.mean(diff * diff).detach().cpu())
    rmse_value = float(torch.sqrt(torch.mean(diff * diff)).detach().cpu())
    mae = float(torch.mean(torch.abs(diff)).detach().cpu())
    per_joint = torch.sqrt(torch.mean(diff * diff, dim=(0, 1))).detach().cpu().numpy()
    true_mean, true_std = tensor_stats(eps_true)
    pred_mean, pred_std = tensor_stats(eps_pred)
    return {
        "mse": mse,
        "rmse": rmse_value,
        "mae": mae,
        "per_joint_rmse": per_joint,
        "true_mean": true_mean,
        "true_std": true_std,
        "pred_mean": pred_mean,
        "pred_std": pred_std,
    }


def reconstruct_q_from_epsilon_candidate(
    x_t: torch.Tensor,
    eps_candidate: torch.Tensor,
    sqrt_ab: torch.Tensor,
    sqrt_omab: torch.Tensor,
    q_start: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    assert x_t.shape == eps_candidate.shape, f"x_t {x_t.shape} != eps_candidate {eps_candidate.shape}"
    pred_x0 = (x_t - sqrt_omab * eps_candidate) / sqrt_ab
    return reconstruct_q(q_start, pred_x0.detach().cpu().numpy().astype(np.float64), mean, std)


def print_epsilon_detail(
    timestep: int,
    model_metrics: Dict[str, Any],
    baseline_metrics: Dict[str, Dict[str, Any]],
) -> None:
    print(
        f"  epsilon stats: true mean={model_metrics['true_mean']:.6e}, "
        f"true std={model_metrics['true_std']:.6e}, pred mean={model_metrics['pred_mean']:.6e}, "
        f"pred std={model_metrics['pred_std']:.6e}"
    )
    print(
        f"  model epsilon errors: mse={model_metrics['mse']:.6e}, "
        f"rmse={model_metrics['rmse']:.6e}, mae={model_metrics['mae']:.6e}"
    )
    print(f"  per-joint epsilon RMSE at t={timestep}")
    values = model_metrics["per_joint_rmse"]
    print("    " + "  ".join(f"q{idx + 1}={value:.6e}" for idx, value in enumerate(values)))
    print("  baseline epsilon RMSE")
    print(
        "    "
        + "  ".join(
            f"{name}={metrics['eps_rmse']:.6e}" for name, metrics in baseline_metrics.items()
        )
    )


def print_model_comparison_table(rows: Sequence[Dict[str, Any]]) -> None:
    if not rows:
        return
    print("\nModel epsilon baseline comparison")
    print(
        "t | model_eps_rmse | zero_eps_rmse | random_eps_rmse | "
        "model_q_rmse | zero_q_rmse | random_q_rmse | model_beats_zero"
    )
    for row in rows:
        print(
            f"{row['t']:>3d} | "
            f"{row['model_eps_rmse']:.6e} | "
            f"{row['zero_eps_rmse']:.6e} | "
            f"{row['random_eps_rmse']:.6e} | "
            f"{row['model_q_rmse']:.6e} | "
            f"{row['zero_q_rmse']:.6e} | "
            f"{row['random_q_rmse']:.6e} | "
            f"{'yes' if row['model_beats_zero'] else 'no'}"
        )


def run_identity(
    data: Dict[str, np.ndarray],
    mean: np.ndarray,
    std: np.ndarray,
) -> Dict[str, bool]:
    require_keys(data, ("delta_q_norm", "q_start", "expert_q"), "identity mode")
    x0 = as_float64(data, "delta_q_norm")
    q_start = as_float64(data, "q_start")
    expert_q = as_float64(data, "expert_q")
    names = path_names(data, expert_q.shape[0])

    print("\n[identity] predicted x0 is exactly ground truth delta_q_norm")
    q_recon = reconstruct_q(q_start, x0, mean, std)
    _, _, passed = print_q_metrics("identity", q_recon, expert_q, names)
    print_fk_metrics("identity", data, q_recon)
    maybe_print_accepted_count("identity", q_recon, expert_q)
    if not passed:
        print("[identity] likely broke: diagnostic script normalization or q_start + delta logic")
    return {"identity": passed}


def run_oracle(
    data: Dict[str, np.ndarray],
    mean: np.ndarray,
    std: np.ndarray,
    timesteps: Sequence[int],
    num_steps: int,
    device: torch.device,
) -> Dict[str, bool]:
    require_keys(data, ("delta_q_norm", "q_start", "expert_q"), "oracle mode")
    x0_np = as_float64(data, "delta_q_norm")
    q_start = as_float64(data, "q_start")
    expert_q = as_float64(data, "expert_q")
    names = path_names(data, expert_q.shape[0])

    x0 = torch.as_tensor(x0_np, dtype=torch.float32, device=device)
    bars = alpha_bars(num_steps, device)
    epsilon = torch.randn_like(x0)
    results: Dict[str, bool] = {}

    print("\n[oracle] exact epsilon is used to invert DDPM forward process")
    with torch.no_grad():
        for timestep in timesteps:
            alpha_bar_t = bars[timestep]
            sqrt_ab = torch.sqrt(alpha_bar_t)
            sqrt_omab = torch.sqrt(1.0 - alpha_bar_t)
            x_t = sqrt_ab * x0 + sqrt_omab * epsilon
            oracle_x0 = (x_t - sqrt_omab * epsilon) / sqrt_ab
            q_recon = reconstruct_q(q_start, oracle_x0.cpu().numpy().astype(np.float64), mean, std)
            _, _, passed = print_q_metrics(f"oracle t={timestep}", q_recon, expert_q, names)
            print_fk_metrics(f"oracle t={timestep}", data, q_recon)
            maybe_print_accepted_count(f"oracle t={timestep}", q_recon, expert_q)
            if not passed:
                print(f"[oracle t={timestep}] likely broke: DDPM schedule or reconstruction formula")
            results[f"oracle_t_{timestep}"] = passed
    return results


def run_model(
    data: Dict[str, np.ndarray],
    mean: np.ndarray,
    std: np.ndarray,
    timesteps: Sequence[int],
    num_steps: int,
    device: torch.device,
    checkpoint: Optional[Path],
) -> Dict[str, bool]:
    require_keys(data, ("delta_q_norm", "condition_features_norm", "q_start", "expert_q"), "model mode")
    x0_np = as_float64(data, "delta_q_norm")
    cond_np = as_float64(data, "condition_features_norm")
    q_start = as_float64(data, "q_start")
    expert_q = as_float64(data, "expert_q")
    names = path_names(data, expert_q.shape[0])

    assert x0_np.ndim == 3, f"delta_q_norm must be (B, T, C), got {x0_np.shape}"
    assert cond_np.ndim in (2, 3), f"condition_features_norm must be (B, D) or (B, T, D), got {cond_np.shape}"
    assert x0_np.shape[0] == cond_np.shape[0], f"B mismatch: delta_q_norm {x0_np.shape}, condition {cond_np.shape}"
    if cond_np.ndim == 3:
        assert x0_np.shape[1] == cond_np.shape[1], f"T mismatch: delta_q_norm {x0_np.shape}, condition {cond_np.shape}"
    assert x0_np.shape[1:] == (100, 6), f"delta_q_norm must be (B, 100, 6), got {x0_np.shape}"
    assert cond_np.shape == (x0_np.shape[0], 100, 13), (
        f"condition_features_norm must be (B, 100, 13), got {cond_np.shape}"
    )

    model = load_model(
        checkpoint,
        device,
        condition_dim=cond_np.shape[-1],
        trajectory_dim=x0_np.shape[2],
        horizon=x0_np.shape[1],
        num_steps=num_steps,
    )

    x0 = torch.as_tensor(x0_np, dtype=torch.float32, device=device)
    cond = torch.as_tensor(cond_np, dtype=torch.float32, device=device)
    bars = alpha_bars(num_steps, device)
    epsilon = torch.randn_like(x0)
    results: Dict[str, bool] = {}
    layout: Optional[Tuple[bool, bool]] = None
    comparison_rows: List[Dict[str, Any]] = []

    print("\n[model] U-Net predicts epsilon; pred_x0 is reconstructed from predicted epsilon")
    with torch.no_grad():
        for timestep in timesteps:
            alpha_bar_t = bars[timestep]
            sqrt_ab = torch.sqrt(alpha_bar_t)
            sqrt_omab = torch.sqrt(1.0 - alpha_bar_t)
            x_t = sqrt_ab * x0 + sqrt_omab * epsilon
            print(f"\n[model t={timestep}]")
            print(f"  x_t dataset layout before model handling: {tuple(x_t.shape)}  # (B, T, C)")
            pred_epsilon, layout = predict_epsilon(model, x_t, timestep, cond, layout)
            assert tuple(x_t.shape) == (x0_np.shape[0], 100, 6), f"x_t must be (B, 100, 6), got {tuple(x_t.shape)}"
            assert tuple(cond.shape) == (x0_np.shape[0], 100, 13), (
                f"condition_features_norm must be (B, 100, 13), got {tuple(cond.shape)}"
            )
            assert tuple(pred_epsilon.shape) == (x0_np.shape[0], 100, 6), (
                f"eps_pred must be (B, 100, 6), got {tuple(pred_epsilon.shape)}"
            )

            eps_zero = torch.zeros_like(epsilon)
            eps_random = torch.randn_like(epsilon)
            eps_mean = epsilon.mean(dim=(0, 1), keepdim=True).expand_as(epsilon)

            model_eps_metrics = epsilon_metrics(pred_epsilon, epsilon)
            baseline_q_metrics: Dict[str, Dict[str, Any]] = {}
            for baseline_name, eps_candidate in (
                ("zero", eps_zero),
                ("random", eps_random),
                ("mean", eps_mean),
            ):
                candidate_eps_metrics = epsilon_metrics(eps_candidate, epsilon)
                candidate_q = reconstruct_q_from_epsilon_candidate(
                    x_t,
                    eps_candidate,
                    sqrt_ab,
                    sqrt_omab,
                    q_start,
                    mean,
                    std,
                )
                candidate_q_rmse, candidate_q_max = q_error_metrics(candidate_q, expert_q)
                baseline_q_metrics[baseline_name] = {
                    "eps_rmse": candidate_eps_metrics["rmse"],
                    "q_rmse": candidate_q_rmse,
                    "q_max": candidate_q_max,
                }

            pred_x0 = (x_t - sqrt_omab * pred_epsilon) / sqrt_ab
            assert pred_x0.shape == x0.shape, f"pred_x0 {pred_x0.shape} != x0 {x0.shape}"

            q_recon = reconstruct_q(q_start, pred_x0.cpu().numpy().astype(np.float64), mean, std)
            model_q_rmse, model_q_max, passed = print_q_metrics(f"model t={timestep}", q_recon, expert_q, names)
            print_epsilon_detail(timestep, model_eps_metrics, baseline_q_metrics)
            print(
                "  baseline q errors: "
                + "  ".join(
                    f"{name}_q_rmse={metrics['q_rmse']:.6e}, {name}_q_max={metrics['q_max']:.6e}"
                    for name, metrics in baseline_q_metrics.items()
                )
            )
            print_fk_metrics(f"model t={timestep}", data, q_recon)
            maybe_print_accepted_count(f"model t={timestep}", q_recon, expert_q)
            comparison_rows.append(
                {
                    "t": timestep,
                    "model_eps_rmse": model_eps_metrics["rmse"],
                    "zero_eps_rmse": baseline_q_metrics["zero"]["eps_rmse"],
                    "random_eps_rmse": baseline_q_metrics["random"]["eps_rmse"],
                    "model_q_rmse": model_q_rmse,
                    "zero_q_rmse": baseline_q_metrics["zero"]["q_rmse"],
                    "random_q_rmse": baseline_q_metrics["random"]["q_rmse"],
                    "model_beats_zero": model_eps_metrics["rmse"] < baseline_q_metrics["zero"]["eps_rmse"],
                    "model_q_max": model_q_max,
                }
            )
            results[f"model_t_{timestep}"] = passed
    print_model_comparison_table(comparison_rows)
    return results


def print_summary(mode: str, results: Dict[str, bool]) -> None:
    print("\nSummary")
    print(f"  mode: {mode}")
    for key, passed in results.items():
        print(f"  {key}: {'PASS' if passed else 'FAIL'}")
    if mode == "identity" and not all(results.values()):
        print("  identity failure means this diagnostic script is wrong.")
    if mode == "oracle" and not all(results.values()):
        print("  oracle failure means this diagnostic has wrong DDPM math or schedule.")
    if mode == "model":
        print("  model failures isolate learned epsilon prediction quality, assuming identity/oracle pass.")


def main() -> int:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[device] CUDA requested but unavailable; using CPU")
        args.device = "cpu"
    device = torch.device(args.device)

    train_data = load_npz(split_path(args.dataset_dir, "train"))
    test_data = load_npz(split_path(args.dataset_dir, "test"))
    print(f"Dataset directory: {args.dataset_dir}")
    print_keys_and_shapes("train", train_data)
    print_keys_and_shapes("test", test_data)

    selected = train_data if args.split == "train" else test_data
    selected = subset_data(selected, args.max_paths)
    print(f"\nUsing split: {args.split}")
    if args.max_paths is not None:
        print(f"Using first {args.max_paths} paths")

    require_keys(selected, ("delta_q_norm", "delta_q", "q_start", "expert_q"), "selected split")
    mean, std, stats_source = get_delta_stats(selected, train_data)
    print(f"[normalization] stats source: {stats_source}")
    print_joint_std(as_float64(selected, "delta_q"), std)

    timesteps = parse_timesteps(args.timesteps, args.num_diffusion_steps)
    if args.mode == "identity":
        results = run_identity(selected, mean, std)
    elif args.mode == "oracle":
        results = run_oracle(selected, mean, std, timesteps, args.num_diffusion_steps, device)
    else:
        results = run_model(selected, mean, std, timesteps, args.num_diffusion_steps, device, args.checkpoint)

    print_summary(args.mode, results)
    return 0 if all(results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
