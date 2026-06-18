#!/usr/bin/env python3
"""
Sample a joint trajectory from the diagnostic conditional diffusion model.

Goal:
    desired_path.csv
        ↓
    trained diagnostic diffusion model
        ↓
    predicted q trajectory CSV

This pairs with:

    train_diffusion_diagnostic.py

Input desired path CSV format:
    t,x,y,z

Output predicted joint trajectory CSV format:
    t,q1,q2,q3,q4,q5,q6

Example:
    cd /workspace/artist_trajectory_scoring

    source /opt/conda/etc/profile.d/conda.sh
    conda activate robodiff

    python sample_diffusion_diagnostic.py \
      --model data/synthetic_paths_train_2000/diffusion_diagnostic.pt \
      --desired_path data/synthetic_paths_test/path_001/desired_path.csv \
      --output_csv data/synthetic_paths_test/path_001/diffusion_diagnostic_pred_q.csv \
      --num_samples 1 \
      --device auto
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn


# ------------------------------------------------------------
# Diffusion schedule
# ------------------------------------------------------------

class DiffusionSchedule:
    """
    Basic DDPM-style linear beta schedule.

    During sampling, we start from random Gaussian noise and repeatedly denoise:

        x_T ~ N(0, I)
        x_t -> x_{t-1}

    The model predicts the noise component at each diffusion step.
    """

    def __init__(
        self,
        num_steps: int,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        device: torch.device | str = "cpu",
    ) -> None:
        self.num_steps = int(num_steps)
        self.device = torch.device(device)

        betas = torch.linspace(
            beta_start,
            beta_end,
            self.num_steps,
            dtype=torch.float32,
            device=self.device,
        )

        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)

        self.betas = betas
        self.alphas = alphas
        self.alpha_bars = alpha_bars

        self.sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - alpha_bars)

    def to(self, device: torch.device | str):
        device = torch.device(device)
        self.device = device

        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alpha_bars = self.alpha_bars.to(device)
        self.sqrt_recip_alphas = self.sqrt_recip_alphas.to(device)
        self.sqrt_one_minus_alpha_bars = self.sqrt_one_minus_alpha_bars.to(device)

        return self


# ------------------------------------------------------------
# Model
# ------------------------------------------------------------

class SinusoidalTimestepEmbedding(nn.Module):
    """
    Standard sinusoidal embedding for diffusion timestep.
    Must match train_diffusion_diagnostic.py.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()

        if dim % 2 != 0:
            raise ValueError("Timestep embedding dimension must be even.")

        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half_dim = self.dim // 2

        freqs = torch.exp(
            -np.log(10000.0)
            * torch.arange(half_dim, device=device, dtype=torch.float32)
            / max(half_dim - 1, 1)
        )

        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)

        return torch.cat([torch.sin(args), torch.cos(args)], dim=1)


class ConditionalNoisePredictor(nn.Module):
    """
    Same architecture as train_diffusion_diagnostic.py.

    Inputs:
        noisy_action:
            noisy q trajectory, shape (B, T*6)

        condition:
            desired path, shape (B, T*3)

        diffusion timestep:
            integer timestep, shape (B,)

    Output:
        predicted noise, shape (B, T*6)
    """

    def __init__(
        self,
        action_dim: int,
        cond_dim: int,
        hidden_dim: int = 1024,
        timestep_embed_dim: int = 128,
        num_layers: int = 4,
    ) -> None:
        super().__init__()

        if num_layers < 2:
            raise ValueError("num_layers must be at least 2")

        self.action_dim = action_dim
        self.cond_dim = cond_dim
        self.hidden_dim = hidden_dim
        self.timestep_embed_dim = timestep_embed_dim
        self.num_layers = num_layers

        self.t_embed = SinusoidalTimestepEmbedding(timestep_embed_dim)

        input_dim = action_dim + cond_dim + timestep_embed_dim

        layers = []
        in_dim = input_dim

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, action_dim))

        self.net = nn.Sequential(*layers)

    def forward(
        self,
        noisy_action: torch.Tensor,
        condition: torch.Tensor,
        diffusion_timestep: torch.Tensor,
    ) -> torch.Tensor:
        t_emb = self.t_embed(diffusion_timestep)

        x = torch.cat(
            [
                noisy_action,
                condition,
                t_emb,
            ],
            dim=1,
        )

        return self.net(x)


# ------------------------------------------------------------
# Loading and preprocessing
# ------------------------------------------------------------

def load_checkpoint(model_path: Path, device: torch.device):
    checkpoint = torch.load(model_path, map_location=device)

    required = [
        "model_state_dict",
        "cond_dim",
        "action_dim",
        "hidden_dim",
        "timestep_embed_dim",
        "num_layers",
        "num_diffusion_steps",
        "beta_start",
        "beta_end",
        "cond_mean",
        "cond_std",
        "action_mean",
        "action_std",
    ]

    missing = [key for key in required if key not in checkpoint]
    if missing:
        raise KeyError(
            f"Checkpoint is missing keys: {missing}. "
            "Make sure it was created by train_diffusion_diagnostic.py."
        )

    model = ConditionalNoisePredictor(
        action_dim=int(checkpoint["action_dim"]),
        cond_dim=int(checkpoint["cond_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        timestep_embed_dim=int(checkpoint["timestep_embed_dim"]),
        num_layers=int(checkpoint["num_layers"]),
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    schedule = DiffusionSchedule(
        num_steps=int(checkpoint["num_diffusion_steps"]),
        beta_start=float(checkpoint["beta_start"]),
        beta_end=float(checkpoint["beta_end"]),
        device=device,
    )

    return model, schedule, checkpoint


def load_desired_path_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        time:
            shape (T, 1)

        desired_xyz:
            shape (T, 3)
    """
    df = pd.read_csv(path)

    required = ["t", "x", "y", "z"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise KeyError(f"{path} is missing columns {missing}. Found: {list(df.columns)}")

    time = df[["t"]].to_numpy(dtype=np.float32)
    desired_xyz = df[["x", "y", "z"]].to_numpy(dtype=np.float32)

    return time, desired_xyz


def prepare_condition(
    desired_xyz: np.ndarray,
    checkpoint: dict,
) -> torch.Tensor:
    """
    Convert desired path from (T, 3) to standardized flattened condition (1, T*3).
    """
    cond_raw = desired_xyz.reshape(1, -1).astype(np.float32)

    cond_dim = int(checkpoint["cond_dim"])
    if cond_raw.shape[1] != cond_dim:
        raise ValueError(
            f"Desired path has flattened dim {cond_raw.shape[1]}, "
            f"but model expects cond_dim {cond_dim}. "
            "This usually means the number of timesteps T differs between "
            "training and this desired_path.csv."
        )

    cond_mean = checkpoint["cond_mean"].astype(np.float32)
    cond_std = checkpoint["cond_std"].astype(np.float32)

    cond = (cond_raw - cond_mean) / cond_std
    return torch.from_numpy(cond.astype(np.float32))


# ------------------------------------------------------------
# Sampling
# ------------------------------------------------------------

@torch.no_grad()
def sample_ddpm(
    model: ConditionalNoisePredictor,
    schedule: DiffusionSchedule,
    condition: torch.Tensor,
    action_dim: int,
    device: torch.device,
    num_samples: int = 1,
) -> torch.Tensor:
    """
    Generate normalized action trajectory samples.

    Returns:
        x:
            normalized generated actions, shape (num_samples, action_dim)
    """
    model.eval()

    condition = condition.to(device)

    if condition.shape[0] == 1 and num_samples > 1:
        condition = condition.repeat(num_samples, 1)

    if condition.shape[0] != num_samples:
        raise ValueError(
            f"condition batch size {condition.shape[0]} does not match "
            f"num_samples {num_samples}"
        )

    x = torch.randn(num_samples, action_dim, device=device)

    for step in reversed(range(schedule.num_steps)):
        t = torch.full(
            size=(num_samples,),
            fill_value=step,
            device=device,
            dtype=torch.long,
        )

        pred_noise = model(
            noisy_action=x,
            condition=condition,
            diffusion_timestep=t,
        )

        beta_t = schedule.betas[step]
        sqrt_recip_alpha_t = schedule.sqrt_recip_alphas[step]
        sqrt_one_minus_alpha_bar_t = schedule.sqrt_one_minus_alpha_bars[step]

        # DDPM reverse mean:
        # x_{t-1} = 1/sqrt(alpha_t) *
        #           (x_t - beta_t / sqrt(1 - alpha_bar_t) * eps_theta)
        model_mean = sqrt_recip_alpha_t * (
            x - (beta_t / sqrt_one_minus_alpha_bar_t) * pred_noise
        )

        if step > 0:
            noise = torch.randn_like(x)
            sigma_t = torch.sqrt(beta_t)
            x = model_mean + sigma_t * noise
        else:
            x = model_mean

    return x


def save_sample_to_csv(
    normalized_action: np.ndarray,
    checkpoint: dict,
    time: np.ndarray,
    output_csv: Path,
) -> None:
    """
    Convert normalized flattened action back to q trajectory and save CSV.
    """
    action_mean = checkpoint["action_mean"].astype(np.float32)
    action_std = checkpoint["action_std"].astype(np.float32)

    action_raw = normalized_action * action_std + action_mean

    action_dim = int(checkpoint["action_dim"])
    if action_raw.shape[1] != action_dim:
        raise ValueError(
            f"Generated action dim {action_raw.shape[1]} does not match "
            f"checkpoint action_dim {action_dim}"
        )

    if action_dim % 6 != 0:
        raise ValueError(
            f"action_dim {action_dim} is not divisible by 6. "
            "Expected flattened shape T*6."
        )

    t_steps = action_dim // 6

    if time.shape[0] != t_steps:
        raise ValueError(
            f"Time has {time.shape[0]} steps, but generated action implies {t_steps} steps."
        )

    q = action_raw.reshape(t_steps, 6)

    out = pd.DataFrame(
        np.concatenate([time, q], axis=1),
        columns=["t", "q1", "q2", "q3", "q4", "q5", "q6"],
    )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=Path,
        required=True,
        help="Path to diffusion_diagnostic.pt",
    )

    parser.add_argument(
        "--desired_path",
        type=Path,
        required=True,
        help="Path to desired_path.csv with columns t,x,y,z",
    )

    parser.add_argument(
        "--output_csv",
        type=Path,
        required=True,
        help="Output predicted joint trajectory CSV",
    )

    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help=(
            "Number of samples to generate. If >1, files are saved with suffix "
            "_sample001, _sample002, etc."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for sampling",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.num_samples < 1:
        raise ValueError("--num_samples must be at least 1")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Device: {device}")

    model, schedule, checkpoint = load_checkpoint(args.model, device)

    time, desired_xyz = load_desired_path_csv(args.desired_path)

    condition = prepare_condition(
        desired_xyz=desired_xyz,
        checkpoint=checkpoint,
    )

    action_dim = int(checkpoint["action_dim"])

    samples = sample_ddpm(
        model=model,
        schedule=schedule,
        condition=condition,
        action_dim=action_dim,
        device=device,
        num_samples=args.num_samples,
    )

    samples_np = samples.cpu().numpy().astype(np.float32)

    if args.num_samples == 1:
        save_sample_to_csv(
            normalized_action=samples_np,
            checkpoint=checkpoint,
            time=time,
            output_csv=args.output_csv,
        )
        print(f"Saved sample: {args.output_csv}")
    else:
        stem = args.output_csv.stem
        suffix = args.output_csv.suffix
        parent = args.output_csv.parent

        for i in range(args.num_samples):
            out_path = parent / f"{stem}_sample{i + 1:03d}{suffix}"
            save_sample_to_csv(
                normalized_action=samples_np[i : i + 1],
                checkpoint=checkpoint,
                time=time,
                output_csv=out_path,
            )
            print(f"Saved sample {i + 1}: {out_path}")

    print("Done.")


if __name__ == "__main__":
    main()