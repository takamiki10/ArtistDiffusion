"""Train the v4 conditional U-Net DDPM trajectory model."""

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset, random_split

from conditional_unet1d_artist import ConditionalUnet1DArtist


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


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_key(npz: np.lib.npyio.NpzFile, candidates: Iterable[str], kind: str) -> str:
    keys = list(npz.keys())
    for key in candidates:
        if key in npz:
            return key
    raise KeyError(f"Could not find {kind} key. Available keys: {keys}")


def maybe_stat(npz: np.lib.npyio.NpzFile, candidates: Iterable[str]) -> Optional[np.ndarray]:
    for key in candidates:
        if key in npz:
            return np.asarray(npz[key], dtype=np.float32)
    return None


def stats_from_array(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    mean = arr.mean(axis=(0, 1), keepdims=True).astype(np.float32)
    std = arr.std(axis=(0, 1), keepdims=True).astype(np.float32)
    std = np.maximum(std, 1e-6)
    return mean, std


class DiffusionV2Dataset(Dataset):
    """Loads one v2 diffusion npz split and exposes normalized cond/target."""

    def __init__(self, npz_path: Path) -> None:
        self.npz_path = Path(npz_path)
        with np.load(self.npz_path, allow_pickle=True) as data:
            print(f"Loading {self.npz_path}")
            print("Available npz keys:", sorted(data.keys()))
            cond_key = pick_key(data, CONDITION_KEYS, "condition")
            target_key = pick_key(data, TARGET_KEYS, "target")

            self.cond = np.asarray(data[cond_key], dtype=np.float32)
            self.target = np.asarray(data[target_key], dtype=np.float32)
            if self.cond.ndim != 3 or self.target.ndim != 3:
                raise ValueError(
                    f"Expected 3D arrays, got cond {self.cond.shape} and target {self.target.shape}"
                )
            if self.cond.shape[:2] != self.target.shape[:2]:
                raise ValueError(
                    f"Condition and target must share N,T, got {self.cond.shape} and {self.target.shape}"
                )

            self.metadata: Dict[str, np.ndarray] = {}
            for key in data.keys():
                if key not in {cond_key, target_key}:
                    value = data[key]
                    if isinstance(value, np.ndarray) and value.dtype != object:
                        self.metadata[key] = np.asarray(value)

            self.normalization = {
                "condition_key": cond_key,
                "target_key": target_key,
                "condition_mean": maybe_stat(data, ("condition_mean", "cond_mean", "X_mean")),
                "condition_std": maybe_stat(data, ("condition_std", "cond_std", "X_std")),
                "target_mean": maybe_stat(data, ("target_mean", "delta_q_mean", "y_mean")),
                "target_std": maybe_stat(data, ("target_std", "delta_q_std", "y_std")),
            }

        if self.normalization["condition_mean"] is None:
            self.normalization["condition_mean"], self.normalization["condition_std"] = stats_from_array(self.cond)
        if self.normalization["target_mean"] is None:
            self.normalization["target_mean"], self.normalization["target_std"] = stats_from_array(self.target)

    def __len__(self) -> int:
        return int(self.target.shape[0])

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.from_numpy(self.cond[idx]), torch.from_numpy(self.target[idx])


def find_split_files(dataset_dir: Path) -> Tuple[Optional[Path], Optional[Path], Path]:
    npz_files = sorted(dataset_dir.glob("*.npz"))
    if not npz_files:
        raise FileNotFoundError(f"No .npz files found in {dataset_dir}")

    train = next((p for p in npz_files if "train" in p.stem.lower()), None)
    val = next((p for p in npz_files if any(s in p.stem.lower() for s in ("val", "valid", "test"))), None)
    combined = next((p for p in npz_files if p not in {train, val}), npz_files[0])
    return train, val, combined


def make_beta_schedule(num_steps: int, device: torch.device) -> Dict[str, torch.Tensor]:
    betas = torch.linspace(1e-4, 0.02, num_steps, device=device)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return {
        "betas": betas,
        "alphas": alphas,
        "alpha_bars": alpha_bars,
        "sqrt_alpha_bars": torch.sqrt(alpha_bars),
        "sqrt_one_minus_alpha_bars": torch.sqrt(1.0 - alpha_bars),
    }


def smoothness_loss(pred_noise: torch.Tensor) -> torch.Tensor:
    if pred_noise.shape[1] < 2:
        return pred_noise.new_tensor(0.0)
    return (pred_noise[:, 1:] - pred_noise[:, :-1]).pow(2).mean()


def run_epoch(
    model: ConditionalUnet1DArtist,
    loader: DataLoader,
    schedule: Dict[str, torch.Tensor],
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    smoothness_weight: float,
) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_count = 0

    for cond, x0 in loader:
        cond = cond.to(device=device, dtype=torch.float32)
        x0 = x0.to(device=device, dtype=torch.float32)
        batch_size = x0.shape[0]
        t = torch.randint(0, schedule["betas"].shape[0], (batch_size,), device=device)
        noise = torch.randn_like(x0)
        x_t = (
            schedule["sqrt_alpha_bars"][t].view(batch_size, 1, 1) * x0
            + schedule["sqrt_one_minus_alpha_bars"][t].view(batch_size, 1, 1) * noise
        )

        with torch.set_grad_enabled(training):
            pred_noise = model(x_t, cond, t)
            loss = F.mse_loss(pred_noise, noise)
            if smoothness_weight > 0.0:
                loss = loss + smoothness_weight * smoothness_loss(pred_noise)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

        total_loss += float(loss.detach().cpu()) * batch_size
        total_count += batch_size

    return total_loss / max(total_count, 1)


def np_stats_to_json(stats: Dict[str, object]) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for key, value in stats.items():
        if isinstance(value, np.ndarray):
            out[key] = value.tolist()
        else:
            out[key] = value
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", default="data/cartesian_expert_dataset_v3/diffusion_v2")
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_diffusion_steps", type=int, default=100)
    parser.add_argument("--output_dir", default="data/cartesian_expert_dataset_v3/diffusion_v4_unet")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--smoothness_loss_weight", type=float, default=0.0)
    args = parser.parse_args()

    set_seed(args.seed)
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_file, val_file, combined_file = find_split_files(dataset_dir)
    if train_file is not None and val_file is not None:
        train_dataset = DiffusionV2Dataset(train_file)
        val_dataset = DiffusionV2Dataset(val_file)
        norm_stats = train_dataset.normalization
    else:
        full_dataset = DiffusionV2Dataset(combined_file)
        val_size = max(1, int(round(len(full_dataset) * args.val_fraction)))
        train_size = max(1, len(full_dataset) - val_size)
        if train_size + val_size > len(full_dataset):
            val_size = len(full_dataset) - train_size
        generator = torch.Generator().manual_seed(args.seed)
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)
        norm_stats = full_dataset.normalization

    sample_cond, sample_target = train_dataset[0]
    cond_dim = int(sample_cond.shape[-1])
    action_dim = int(sample_target.shape[-1])

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    model_config = {
        "action_dim": action_dim,
        "cond_dim": cond_dim,
        "hidden_dim": args.hidden_dim,
        "dim_mults": [1, 2, 4],
    }
    diffusion_config = {
        "num_diffusion_steps": args.num_diffusion_steps,
        "beta_start": 1e-4,
        "beta_end": 0.02,
        "prediction_type": "epsilon",
    }
    model = ConditionalUnet1DArtist(**model_config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    schedule = make_beta_schedule(args.num_diffusion_steps, device)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    config = {
        "args": vars(args),
        "model_config": model_config,
        "diffusion_config": diffusion_config,
        "normalization": np_stats_to_json(norm_stats),
    }
    (output_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    log_path = output_dir / "train_log.csv"
    best_val = float("inf")
    with log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "best_val_loss"])
        writer.writeheader()

        for epoch in range(1, args.epochs + 1):
            train_loss = run_epoch(model, train_loader, schedule, optimizer, device, args.smoothness_loss_weight)
            val_loss = run_epoch(model, val_loader, schedule, None, device, args.smoothness_loss_weight)
            if val_loss < best_val:
                best_val = val_loss
                save_checkpoint(output_dir / "best_model.pt", model, optimizer, epoch, best_val, norm_stats, model_config, diffusion_config)

            save_checkpoint(output_dir / "last_model.pt", model, optimizer, epoch, best_val, norm_stats, model_config, diffusion_config)
            writer.writerow(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "best_val_loss": best_val,
                }
            )
            f.flush()
            print(
                f"epoch {epoch:04d} train_loss={train_loss:.6f} "
                f"val_loss={val_loss:.6f} best_val={best_val:.6f}"
            )


def save_checkpoint(
    path: Path,
    model: ConditionalUnet1DArtist,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_val: float,
    norm_stats: Dict[str, object],
    model_config: Dict[str, object],
    diffusion_config: Dict[str, object],
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_validation_loss": best_val,
            "normalization": np_stats_to_json(norm_stats),
            "model_config": model_config,
            "diffusion_config": diffusion_config,
        },
        path,
    )


if __name__ == "__main__":
    main()
