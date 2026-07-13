#!/usr/bin/env python3
"""Generate MLP-only v3 test predictions for diffusion prior refinement.

Outputs one CSV per path:
  data/cartesian_expert_dataset_v3/mlp_v3_test_predictions/<path_name>/predicted_q.csv

and a summary CSV:
  data/cartesian_expert_dataset_v3/mlp_v3_test_predictions_summary.csv

This script does not modify training, diffusion, or evaluation files.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import inspect
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch


DEFAULT_CHECKPOINT = Path("data/cartesian_expert_dataset_v3/path_conditioned_mlp_v3.pt")
DEFAULT_TEST_NPZ = Path("data/cartesian_expert_dataset_v3/diffusion_v2/diffusion_test_v2.npz")
DEFAULT_TRAIN_NPZ = Path("data/cartesian_expert_dataset_v3/diffusion_v2/diffusion_train_v2.npz")
DEFAULT_OUTPUT_DIR = Path("data/cartesian_expert_dataset_v3/mlp_v3_test_predictions")
DEFAULT_SUMMARY_CSV = Path("data/cartesian_expert_dataset_v3/mlp_v3_test_predictions_summary.csv")
MLP_MODULES = (
    "predict_path_conditioned_mlp",
    "evaluate_path_conditioned_mlp",
    "train_path_conditioned_mlp",
)


class StateDictMLP(torch.nn.Module):
    """Fallback MLP reconstructed from Linear layer weights in a checkpoint."""

    def __init__(self, state_dict: Dict[str, torch.Tensor]) -> None:
        super().__init__()
        self.linear_items: List[Tuple[str, torch.Tensor]] = [
            (key, value)
            for key, value in state_dict.items()
            if key.endswith(".weight") and value.ndim == 2
        ]
        if not self.linear_items:
            raise ValueError("Could not find Linear layer weights in checkpoint state_dict")

        layers: List[torch.nn.Module] = []
        self.linear_layers = torch.nn.ModuleList()
        for idx, (_, weight) in enumerate(self.linear_items):
            out_features, in_features = int(weight.shape[0]), int(weight.shape[1])
            layer = torch.nn.Linear(in_features, out_features)
            self.linear_layers.append(layer)
            layers.append(layer)
            if idx < len(self.linear_items) - 1:
                layers.append(torch.nn.ReLU())
        self.net = torch.nn.Sequential(*layers)

    def load_checkpoint_weights(self, state_dict: Dict[str, torch.Tensor]) -> None:
        for layer, (weight_key, weight) in zip(self.linear_layers, self.linear_items):
            bias_key = weight_key[:-len("weight")] + "bias"
            layer.weight.data.copy_(weight)
            if bias_key in state_dict:
                layer.bias.data.copy_(state_dict[bias_key])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate path-conditioned MLP v3 predictions on diffusion test paths.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--test_npz", "--dataset_npz", dest="test_npz", type=Path, default=DEFAULT_TEST_NPZ)
    parser.add_argument("--train_npz", type=Path, default=DEFAULT_TRAIN_NPZ)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--summary_csv", type=Path, default=DEFAULT_SUMMARY_CSV)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--batch_size", type=int, default=256)
    return parser.parse_args()


def load_npz(path: Path) -> Dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Missing test dataset: {path}")
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in data.files}


def require_keys(data: Dict[str, np.ndarray], keys: Sequence[str]) -> None:
    missing = [key for key in keys if key not in data]
    if missing:
        raise KeyError(f"Missing required dataset key(s): {', '.join(missing)}")


def path_names(data: Dict[str, np.ndarray]) -> List[str]:
    raw = np.asarray(data["path_names"])
    names: List[str] = []
    for item in raw:
        names.append(item.decode("utf-8", errors="replace") if isinstance(item, bytes) else str(item))
    return names


def safe_path_name(name: str) -> str:
    return Path(str(name)).name.replace("/", "_").replace("\\", "_")


def load_checkpoint(path: Path, device: torch.device) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {path}")
    raw = torch.load(path, map_location=device)
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, torch.nn.Module):
        return {"model": raw}
    raise TypeError(f"Unsupported checkpoint type: {type(raw)!r}")


def checkpoint_state_dict(checkpoint: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    for key in ("model_state_dict", "state_dict", "model"):
        value = checkpoint.get(key)
        if isinstance(value, dict) and value and all(torch.is_tensor(v) for v in value.values()):
            return value
    if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
        return checkpoint  # raw state_dict loaded as dict
    raise ValueError("Could not find raw/model_state_dict/state_dict in checkpoint")


def strip_prefixes(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
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


def import_mlp_modules() -> List[Any]:
    modules = []
    for module_name in MLP_MODULES:
        try:
            modules.append(importlib.import_module(module_name))
        except Exception as exc:
            print(f"[mlp] could not import {module_name}: {exc}")
    return modules


def find_model_classes(modules: Sequence[Any]) -> Iterable[type]:
    preferred = (
        "PathConditionedMLP",
        "PathConditionedMLPV3",
        "TrajectoryMLP",
        "MLP",
    )
    yielded = set()
    for module in modules:
        for name in preferred:
            cls = getattr(module, name, None)
            if inspect.isclass(cls) and issubclass(cls, torch.nn.Module):
                yielded.add(cls)
                yield cls
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if cls in yielded:
                continue
            if issubclass(cls, torch.nn.Module) and ("MLP" in cls.__name__ or "Path" in cls.__name__):
                yielded.add(cls)
                yield cls


def first_linear_dims(state: Dict[str, torch.Tensor]) -> Tuple[int, int]:
    linear_weights = [value for key, value in state.items() if key.endswith(".weight") and value.ndim == 2]
    if not linear_weights:
        raise ValueError("No 2D Linear weights found in state_dict")
    return int(linear_weights[0].shape[1]), int(linear_weights[-1].shape[0])


def checkpoint_config(checkpoint: Dict[str, Any]) -> Dict[str, Any]:
    for key in ("config", "model_config", "args", "model_args", "hparams", "hyperparameters"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return dict(value)
        if hasattr(value, "__dict__"):
            return vars(value)
    return {}


def instantiate_existing_model(
    checkpoint: Dict[str, Any],
    state: Dict[str, torch.Tensor],
    modules: Sequence[Any],
) -> Optional[torch.nn.Module]:
    if isinstance(checkpoint.get("model"), torch.nn.Module):
        return checkpoint["model"]

    input_dim, output_dim = first_linear_dims(state)
    config = checkpoint_config(checkpoint)
    attempts = [
        dict(config),
        {"input_dim": input_dim, "output_dim": output_dim, **config},
        {"in_dim": input_dim, "out_dim": output_dim, **config},
        {"path_dim": input_dim, "output_dim": output_dim, **config},
        {"condition_dim": input_dim, "trajectory_dim": output_dim, **config},
        {"input_size": input_dim, "output_size": output_dim, **config},
        {},
    ]
    for cls in find_model_classes(modules):
        for kwargs in attempts:
            try:
                model = cls(**kwargs)
            except Exception:
                continue
            try:
                model.load_state_dict(state, strict=True)
                print(f"[mlp] loaded {cls.__module__}.{cls.__name__} with strict=True")
                return model
            except Exception:
                continue
    return None


def fallback_model_from_state(state: Dict[str, torch.Tensor]) -> torch.nn.Module:
    stripped = strip_prefixes(state)
    model = StateDictMLP(stripped)
    model.load_checkpoint_weights(stripped)
    print("[mlp] loaded fallback StateDictMLP reconstructed from checkpoint Linear layers")
    return model


def load_mlp_model(checkpoint_path: Path, device: torch.device) -> Tuple[torch.nn.Module, Dict[str, Any], Dict[str, torch.Tensor]]:
    try:
        from predict_path_conditioned_mlp import load_model as existing_load_model

        model, checkpoint = existing_load_model(checkpoint_path, device)
        state = checkpoint_state_dict(checkpoint)
        print("[mlp] loaded with predict_path_conditioned_mlp.load_model")
        return model.to(device).eval(), checkpoint, state
    except Exception as exc:
        print(f"[mlp] existing predict_path_conditioned_mlp.load_model unavailable: {exc}")

    checkpoint = load_checkpoint(checkpoint_path, device)
    state = checkpoint_state_dict(checkpoint)
    modules = import_mlp_modules()
    model = instantiate_existing_model(checkpoint, state, modules)
    if model is None:
        model = fallback_model_from_state(state)
    return model.to(device).eval(), checkpoint, strip_prefixes(state)


def infer_times(data: Dict[str, np.ndarray], checkpoint: Dict[str, Any]) -> np.ndarray:
    desired = np.asarray(data["desired_paths"], dtype=np.float32)
    bsz, steps, _ = desired.shape
    if "times" in data:
        times = np.asarray(data["times"], dtype=np.float32)
        if times.shape == (bsz, steps):
            return times
        if times.shape == (steps,):
            return np.repeat(times.reshape(1, steps), bsz, axis=0).astype(np.float32)
        raise ValueError(f"times key has unsupported shape {times.shape}; expected {(bsz, steps)} or {(steps,)}")

    num_steps = int(checkpoint.get("num_steps", steps))
    if num_steps != steps:
        raise ValueError(f"checkpoint num_steps={num_steps} but desired_paths has T={steps}")

    print("[mlp] dataset has no times key; using np.linspace(0, 1, T) to match path CSV convention")
    times_1d = np.linspace(0.0, 1.0, steps, dtype=np.float32)
    return np.repeat(times_1d.reshape(1, steps), bsz, axis=0)


def make_timestep_features_from_diffusion_npz(
    data: Dict[str, np.ndarray],
    checkpoint: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    desired = np.asarray(data["desired_paths"], dtype=np.float32)
    if desired.ndim != 3 or desired.shape[-1] != 3:
        raise ValueError(f"desired_paths must be (B,T,3), got {desired.shape}")
    bsz, steps, _ = desired.shape
    times = infer_times(data, checkpoint)
    include_current_point = bool(checkpoint.get("include_current_point", True))

    path_flat = desired.reshape(bsz, 1, steps * 3)
    path_context = np.repeat(path_flat, steps, axis=1)
    features = [path_context, times[..., None].astype(np.float32)]
    if include_current_point:
        features.append(desired.astype(np.float32))
    x_raw = np.concatenate(features, axis=-1).astype(np.float32)
    return x_raw.reshape(bsz * steps, -1), times


def standardize_inputs(x_raw: np.ndarray, checkpoint: Dict[str, Any]) -> np.ndarray:
    if "x_mean" not in checkpoint or "x_std" not in checkpoint:
        raise KeyError("Checkpoint is missing x_mean/x_std; cannot match MLP training preprocessing")
    x_mean = np.asarray(checkpoint["x_mean"], dtype=np.float32)
    x_std = np.asarray(checkpoint["x_std"], dtype=np.float32)
    if x_mean.shape[-1] != x_raw.shape[1] or x_std.shape[-1] != x_raw.shape[1]:
        raise ValueError(f"x_mean/x_std shape {x_mean.shape}/{x_std.shape} does not match input {x_raw.shape}")
    return ((x_raw - x_mean.reshape(1, -1)) / x_std.reshape(1, -1)).astype(np.float32)


def build_input_candidates(data: Dict[str, np.ndarray], checkpoint: Optional[Dict[str, Any]] = None) -> Dict[str, np.ndarray]:
    condition = np.asarray(data["condition_features"], dtype=np.float32)
    condition_norm = np.asarray(data["condition_features_norm"], dtype=np.float32)
    desired = np.asarray(data["desired_paths"], dtype=np.float32)
    q_start = np.asarray(data["q_start"], dtype=np.float32)
    bsz = condition.shape[0]
    candidates: Dict[str, np.ndarray] = {
        "condition_features_norm_flat": condition_norm.reshape(bsz, -1),
        "condition_features_flat": condition.reshape(bsz, -1),
        "desired_paths_flat_plus_q_start": np.concatenate([desired.reshape(bsz, -1), q_start], axis=1),
        "desired_paths_flat": desired.reshape(bsz, -1),
        "condition_features_norm_first_step": condition_norm[:, 0, :],
        "condition_features_first_step": condition[:, 0, :],
    }
    if checkpoint is not None:
        timestep_features, _ = make_timestep_features_from_diffusion_npz(data, checkpoint)
        candidates["path_conditioned_mlp_timestep_features"] = standardize_inputs(timestep_features, checkpoint)
    return candidates


def infer_input_array(data: Dict[str, np.ndarray], state: Dict[str, torch.Tensor], checkpoint: Dict[str, Any]) -> Tuple[str, np.ndarray]:
    input_dim, _ = first_linear_dims(state)
    config = checkpoint_config(checkpoint)
    preferred_key = (
        config.get("input_key")
        or config.get("feature_key")
        or config.get("input_format")
        or config.get("features")
    )
    candidates = build_input_candidates(data, checkpoint)
    if "path_conditioned_mlp_timestep_features" in candidates:
        value = candidates["path_conditioned_mlp_timestep_features"]
        if value.shape[1] == input_dim:
            return "path_conditioned_mlp_timestep_features", value
    if isinstance(preferred_key, str) and preferred_key in candidates and candidates[preferred_key].shape[1] == input_dim:
        return preferred_key, candidates[preferred_key]

    matches = [(name, value) for name, value in candidates.items() if value.shape[1] == input_dim]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        priority = (
            "condition_features_norm_flat",
            "condition_features_flat",
            "desired_paths_flat_plus_q_start",
            "desired_paths_flat",
            "condition_features_norm_first_step",
            "condition_features_first_step",
        )
        for name in priority:
            for match_name, value in matches:
                if match_name == name:
                    print(f"[mlp] multiple input candidates matched {input_dim}; using {name}")
                    return match_name, value

    raise RuntimeError(
        f"Could not infer MLP input convention from checkpoint input_dim={input_dim}. "
        f"Candidate dimensions: {', '.join(f'{k}={v.shape[1]}' for k, v in candidates.items())}. "
        "Please expose the original preprocessing helper in train/predict/evaluate MLP scripts or add config input_key."
    )


def forward_batches(
    model: torch.nn.Module,
    inputs: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    outputs: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, inputs.shape[0], batch_size):
            batch = torch.as_tensor(inputs[start:start + batch_size], dtype=torch.float32, device=device)
            pred = model(batch)
            if isinstance(pred, tuple):
                pred = pred[0]
            outputs.append(pred.detach().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def output_to_q(pred: np.ndarray, data: Dict[str, np.ndarray], checkpoint: Dict[str, Any]) -> np.ndarray:
    num_paths = np.asarray(data["desired_paths"]).shape[0]
    num_steps = np.asarray(data["desired_paths"]).shape[1]
    config = checkpoint_config(checkpoint)
    output_type = str(config.get("output_type") or config.get("target") or config.get("target_key") or "").lower()

    if pred.ndim == 2 and pred.shape == (num_paths * num_steps, 6):
        q_like = pred.reshape(num_paths, num_steps, 6).astype(np.float64)
    elif pred.ndim == 3 and pred.shape[1:] == (100, 6):
        q_like = pred.astype(np.float64)
    elif pred.ndim == 2 and pred.shape[1] == num_steps * 6:
        q_like = pred.reshape(num_paths, num_steps, 6).astype(np.float64)
    else:
        raise RuntimeError(
            f"Unsupported MLP output shape {pred.shape}; expected "
            f"({num_paths * num_steps},6), (B,{num_steps * 6}), or (B,100,6)"
        )

    if "y_mean" in checkpoint and "y_std" in checkpoint:
        y_mean = np.asarray(checkpoint["y_mean"], dtype=np.float64).reshape(1, 1, -1)
        y_std = np.asarray(checkpoint["y_std"], dtype=np.float64).reshape(1, 1, -1)
        q_like = q_like * y_std + y_mean

    q_start = np.asarray(data["q_start"], dtype=np.float64)
    if output_type in {"delta_q", "delta", "dq"}:
        return q_start[:, None, :] + q_like
    if output_type in {"expert_q", "q", "joint", "joints", "trajectory", "q_traj"}:
        return q_like

    expert_q = np.asarray(data.get("expert_q"), dtype=np.float64) if "expert_q" in data else None
    if expert_q is not None:
        direct_rmse = rmse(q_like - expert_q)
        delta_rmse = rmse((q_start[:, None, :] + q_like) - expert_q)
        if delta_rmse + 1e-12 < direct_rmse:
            print("[mlp] output convention inferred as delta_q from lower RMSE against expert_q")
            return q_start[:, None, :] + q_like
        print("[mlp] output convention inferred as expert_q from lower RMSE against expert_q")
    else:
        print("[mlp] output convention unspecified; treating output as expert_q")
    return q_like


def rmse(error: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(error))))


def reshape_model_output(pred: np.ndarray, data: Dict[str, np.ndarray]) -> np.ndarray:
    num_paths = np.asarray(data["desired_paths"]).shape[0]
    num_steps = np.asarray(data["desired_paths"]).shape[1]
    if pred.ndim == 2 and pred.shape == (num_paths * num_steps, 6):
        return pred.reshape(num_paths, num_steps, 6).astype(np.float64)
    if pred.ndim == 3 and pred.shape == (num_paths, num_steps, 6):
        return pred.astype(np.float64)
    if pred.ndim == 2 and pred.shape == (num_paths, num_steps * 6):
        return pred.reshape(num_paths, num_steps, 6).astype(np.float64)
    raise RuntimeError(
        f"Unsupported MLP output shape {pred.shape}; expected "
        f"({num_paths * num_steps},6), ({num_paths},{num_steps * 6}), or ({num_paths},{num_steps},6)"
    )


def load_train_stats(train_npz: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_data = load_npz(train_npz)
    require_keys(train_data, ("delta_q", "expert_q"))
    delta_q = np.asarray(train_data["delta_q"], dtype=np.float64)
    expert_q = np.asarray(train_data["expert_q"], dtype=np.float64)
    delta_mean = delta_q.mean(axis=(0, 1))
    delta_std = delta_q.std(axis=(0, 1))
    q_mean = expert_q.mean(axis=(0, 1))
    q_std = expert_q.std(axis=(0, 1))
    delta_std = np.where(delta_std < 1e-12, 1e-12, delta_std)
    q_std = np.where(q_std < 1e-12, 1e-12, q_std)
    print("[normalization] train delta_q std per joint")
    for idx, value in enumerate(delta_std):
        marker = "  <-- q6" if idx == 5 else ""
        print(f"  q{idx + 1}: mean={delta_mean[idx]:.12e}, std={value:.12e}{marker}")
    return (
        delta_mean.reshape(1, 1, 6),
        delta_std.reshape(1, 1, 6),
        q_mean.reshape(1, 1, 6),
        q_std.reshape(1, 1, 6),
    )


def write_q_csv(path: Path, values: np.ndarray, times: Optional[np.ndarray] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["t", "q1", "q2", "q3", "q4", "q5", "q6"])
        if times is None:
            times = np.arange(values.shape[0], dtype=np.float64)
        for t_value, row in zip(times, values):
            writer.writerow([f"{float(t_value):.10f}"] + [f"{float(value):.10f}" for value in row])


def write_raw_output_csv(path: Path, values: np.ndarray, times: Optional[np.ndarray] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["t", "y1", "y2", "y3", "y4", "y5", "y6"])
        if times is None:
            times = np.arange(values.shape[0], dtype=np.float64)
        for t_value, row in zip(times, values):
            writer.writerow([f"{float(t_value):.10f}"] + [f"{float(value):.10f}" for value in row])


def old_interpretation_metrics(
    raw_model_output: np.ndarray,
    data: Dict[str, np.ndarray],
    train_delta_mean: np.ndarray,
    train_delta_std: np.ndarray,
    train_q_mean: np.ndarray,
    train_q_std: np.ndarray,
) -> None:
    expert_q = np.asarray(data["expert_q"], dtype=np.float64)
    q_start = np.asarray(data["q_start"], dtype=np.float64)
    candidates = {
        "raw_as_full_q": raw_model_output,
        "raw_as_delta_q": q_start[:, None, :] + raw_model_output,
        "raw_as_normalized_delta_q": q_start[:, None, :] + raw_model_output * train_delta_std + train_delta_mean,
        "raw_as_normalized_full_q": raw_model_output * train_q_std + train_q_mean,
    }
    print("\n[debug] raw_model_output interpretation metrics")
    print("interpretation | q_rmse | max_q_error | q1_rmse | q2_rmse | q3_rmse | q4_rmse | q5_rmse | q6_rmse")
    for name, candidate in sorted(candidates.items(), key=lambda item: rmse(item[1] - expert_q)):
        error = candidate - expert_q
        per_joint = np.sqrt(np.mean(np.square(error), axis=(0, 1)))
        print(
            f"{name} | {rmse(error):.12e} | {float(np.max(np.abs(error))):.12e} | "
            + " | ".join(f"{value:.12e}" for value in per_joint)
        )


def try_fk_metrics(q_pred: np.ndarray, data: Dict[str, np.ndarray]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    try:
        for module_name in ("evaluate_path_conditioned_mlp", "score_trajectory", "validate_expert_dataset"):
            try:
                module = importlib.import_module(module_name)
            except Exception:
                continue
            for name in ("compute_fk_positions", "compute_fk_path", "forward_kinematics", "get_fk_positions", "fk_positions"):
                fn = getattr(module, name, None)
                if not callable(fn):
                    continue
                for args in ((q_pred,), (q_pred, data), (q_pred, data.get("q_start"))):
                    try:
                        fk = np.asarray(fn(*args), dtype=np.float64)[..., :3]
                    except Exception:
                        continue
                    desired = np.asarray(data["desired_paths"], dtype=np.float64)
                    if fk.shape == desired.shape:
                        errors = np.linalg.norm(fk - desired, axis=-1)
                        return np.mean(errors, axis=1), np.max(errors, axis=1)
    except Exception:
        pass
    return None, None


def write_summary_csv(path: Path, rows: Sequence[Dict[str, Any]], include_fk: bool) -> None:
    fields = [
        "path_name",
        "q_rmse_vs_expert",
        "max_q_error_vs_expert",
        "q1_rmse_vs_expert",
        "q2_rmse_vs_expert",
        "q3_rmse_vs_expert",
        "q4_rmse_vs_expert",
        "q5_rmse_vs_expert",
        "q6_rmse_vs_expert",
    ]
    if include_fk:
        fields += ["mean_cartesian_error", "max_cartesian_error"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def main() -> int:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[device] CUDA requested but unavailable; using CPU")
        args.device = "cpu"
    device = torch.device(args.device)

    data = load_npz(args.test_npz)
    require_keys(
        data,
        ("condition_features", "condition_features_norm", "desired_paths", "q_start", "path_names", "expert_q"),
    )
    names = path_names(data)
    print(f"Loaded test dataset: {args.test_npz}")
    for key in sorted(data.keys()):
        print(f"  {key}: shape={data[key].shape}, dtype={data[key].dtype}")

    model, checkpoint, state = load_mlp_model(args.checkpoint, device)
    input_name, inputs = infer_input_array(data, state, checkpoint)
    print(f"[mlp] input convention: {input_name}, shape={inputs.shape}")
    times = infer_times(data, checkpoint)
    train_delta_mean, train_delta_std, train_q_mean, train_q_std = load_train_stats(args.train_npz)

    raw_pred = forward_batches(model, inputs, args.batch_size, device)
    raw_model_output = reshape_model_output(raw_pred, data)
    predicted_delta_q_norm = raw_model_output
    predicted_delta_q = predicted_delta_q_norm * train_delta_std + train_delta_mean
    q_start = np.asarray(data["q_start"], dtype=np.float64)
    q_pred = q_start[:, None, :] + predicted_delta_q

    if q_pred.shape != (len(names), 100, 6):
        raise RuntimeError(f"Predicted q must have shape ({len(names)}, 100, 6), got {q_pred.shape}")
    if raw_model_output.shape != q_pred.shape:
        raise RuntimeError(f"Raw model output must have shape {q_pred.shape}, got {raw_model_output.shape}")

    old_interpretation_metrics(
        raw_model_output,
        data,
        train_delta_mean,
        train_delta_std,
        train_q_mean,
        train_q_std,
    )

    expert_q = np.asarray(data["expert_q"], dtype=np.float64) if "expert_q" in data else None
    fk_mean, fk_max = try_fk_metrics(q_pred, data) if "desired_paths" in data else (None, None)
    include_fk = fk_mean is not None and fk_max is not None

    summary_rows: List[Dict[str, Any]] = []
    for idx, name in enumerate(names):
        path_dir = args.output_dir / safe_path_name(name)
        write_raw_output_csv(path_dir / "raw_model_output.csv", raw_model_output[idx], times[idx])
        write_q_csv(path_dir / "predicted_delta_q_norm.csv", predicted_delta_q_norm[idx], times[idx])
        write_q_csv(path_dir / "predicted_delta_q.csv", predicted_delta_q[idx], times[idx])
        write_q_csv(path_dir / "predicted_q.csv", q_pred[idx], times[idx])

        row: Dict[str, Any] = {"path_name": name}
        if expert_q is not None:
            error = q_pred[idx] - expert_q[idx]
            per_joint = np.sqrt(np.mean(np.square(error), axis=0))
            row["q_rmse_vs_expert"] = f"{rmse(error):.12e}"
            row["max_q_error_vs_expert"] = f"{float(np.max(np.abs(error))):.12e}"
            for joint_idx, value in enumerate(per_joint):
                row[f"q{joint_idx + 1}_rmse_vs_expert"] = f"{float(value):.12e}"
        else:
            row["q_rmse_vs_expert"] = ""
            row["max_q_error_vs_expert"] = ""
            for joint_idx in range(6):
                row[f"q{joint_idx + 1}_rmse_vs_expert"] = ""
        if include_fk:
            row["mean_cartesian_error"] = f"{float(fk_mean[idx]):.12e}"
            row["max_cartesian_error"] = f"{float(fk_max[idx]):.12e}"
        summary_rows.append(row)

    write_summary_csv(args.summary_csv, summary_rows, include_fk)
    print(f"Saved 4 prediction/debug CSVs for each of {len(names)} paths under: {args.output_dir}")
    print(f"Saved summary CSV: {args.summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
