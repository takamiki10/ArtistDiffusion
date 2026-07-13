"""Conditional 1D U-Net for ArtistDiffusion trajectory denoising.

The model follows the broad shape of Diffusion Policy's conditional 1D U-Net:
it denoises an action sequence while conditioning each residual block with a
global FiLM vector built from the diffusion timestep and the desired path
condition. Local per-timestep condition features are also injected at the input.
"""

import math
from typing import Iterable, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPosEmb(nn.Module):
    """Standard sinusoidal embedding for integer diffusion timesteps."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("embedding dim must be positive")
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 1:
            raise ValueError(f"timesteps must have shape (B,), got {tuple(x.shape)}")

        device = x.device
        half_dim = self.dim // 2
        if half_dim == 0:
            return x.float().unsqueeze(-1)

        scale = math.log(10000.0) / max(half_dim - 1, 1)
        freqs = torch.exp(torch.arange(half_dim, device=device, dtype=torch.float32) * -scale)
        emb = x.float().unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class Downsample1d(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample1d(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class FiLMResidualBlock1d(nn.Module):
    """Conv1d residual block with FiLM modulation from a global condition."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        kernel_size: int = 5,
        groups: int = 8,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd to preserve temporal length")

        groups = min(groups, out_channels)
        while out_channels % groups != 0:
            groups -= 1

        padding = kernel_size // 2
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding),
            nn.GroupNorm(groups, out_channels),
            nn.Mish(),
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding),
            nn.GroupNorm(groups, out_channels),
            nn.Mish(),
        )
        self.film = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, out_channels * 2),
        )
        self.residual = (
            nn.Conv1d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        y = self.block1(x)
        scale, bias = self.film(cond).chunk(2, dim=-1)
        y = y * (1.0 + scale.unsqueeze(-1)) + bias.unsqueeze(-1)
        y = self.block2(y)
        return y + self.residual(x)


class ConditionalUnet1DArtist(nn.Module):
    """Conditional 1D U-Net for normalized delta-q trajectory diffusion.

    Args:
        action_dim: Noisy trajectory channel count. For xMateCR7 active joints,
            this is 6.
        cond_dim: Per-timestep condition feature count. The v2 dataset uses 13.
        hidden_dim: Base U-Net width.
        dim_mults: Width multipliers for each downsampling level.
        timestep_embed_dim: Dimension of the sinusoidal timestep MLP output.
        local_cond_dim: Projection width for local condition injection. Defaults
            to ``hidden_dim``.
    """

    def __init__(
        self,
        action_dim: int = 6,
        cond_dim: int = 13,
        hidden_dim: int = 256,
        dim_mults: Sequence[int] = (1, 2, 4),
        timestep_embed_dim: Optional[int] = None,
        local_cond_dim: Optional[int] = None,
        kernel_size: int = 5,
    ) -> None:
        super().__init__()
        if action_dim <= 0 or cond_dim <= 0 or hidden_dim <= 0:
            raise ValueError("action_dim, cond_dim, and hidden_dim must be positive")
        if not dim_mults:
            raise ValueError("dim_mults must contain at least one level")

        self.action_dim = action_dim
        self.cond_dim = cond_dim
        self.hidden_dim = hidden_dim
        self.dim_mults = tuple(int(m) for m in dim_mults)

        timestep_embed_dim = timestep_embed_dim or hidden_dim
        local_cond_dim = local_cond_dim or hidden_dim
        global_cond_dim = timestep_embed_dim + hidden_dim

        self.local_cond_encoder = nn.Sequential(
            nn.Conv1d(cond_dim, local_cond_dim, kernel_size=1),
            nn.Mish(),
            nn.Conv1d(local_cond_dim, local_cond_dim, kernel_size=1),
        )
        self.global_cond_encoder = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(timestep_embed_dim),
            nn.Linear(timestep_embed_dim, timestep_embed_dim * 4),
            nn.Mish(),
            nn.Linear(timestep_embed_dim * 4, timestep_embed_dim),
        )

        self.input_proj = nn.Conv1d(action_dim + local_cond_dim, hidden_dim, kernel_size=1)

        dims = [hidden_dim] + [hidden_dim * m for m in self.dim_mults]
        self.down_modules = nn.ModuleList()
        for level, (dim_in, dim_out) in enumerate(zip(dims[:-1], dims[1:])):
            is_last = level == len(dims) - 2
            self.down_modules.append(
                nn.ModuleDict(
                    {
                        "res1": FiLMResidualBlock1d(dim_in, dim_out, global_cond_dim, kernel_size),
                        "res2": FiLMResidualBlock1d(dim_out, dim_out, global_cond_dim, kernel_size),
                        "down": nn.Identity() if is_last else Downsample1d(dim_out),
                    }
                )
            )

        mid_dim = dims[-1]
        self.mid_modules = nn.ModuleList(
            [
                FiLMResidualBlock1d(mid_dim, mid_dim, global_cond_dim, kernel_size),
                FiLMResidualBlock1d(mid_dim, mid_dim, global_cond_dim, kernel_size),
            ]
        )

        self.up_modules = nn.ModuleList()
        rev_dims = list(reversed(dims[1:]))
        for level, dim_in in enumerate(rev_dims):
            is_last = level == len(rev_dims) - 1
            skip_dim = dim_in
            dim_out = hidden_dim if is_last else rev_dims[level + 1]
            self.up_modules.append(
                nn.ModuleDict(
                    {
                        "res1": FiLMResidualBlock1d(dim_in + skip_dim, dim_out, global_cond_dim, kernel_size),
                        "res2": FiLMResidualBlock1d(dim_out, dim_out, global_cond_dim, kernel_size),
                        "up": nn.Identity() if is_last else Upsample1d(dim_out),
                    }
                )
            )

        self.final_conv = nn.Sequential(
            FiLMResidualBlock1d(hidden_dim, hidden_dim, global_cond_dim, kernel_size),
            nn.Conv1d(hidden_dim, action_dim, kernel_size=1),
        )

    @staticmethod
    def _match_length(x: torch.Tensor, length: int) -> torch.Tensor:
        """Crop or pad the temporal dimension after strided upsampling."""

        current = x.shape[-1]
        if current == length:
            return x
        if current > length:
            return x[..., :length]
        return F.pad(x, (0, length - current))

    def forward(self, x: torch.Tensor, cond: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"x must have shape (B, T, action_dim), got {tuple(x.shape)}")
        if cond.ndim != 3:
            raise ValueError(f"cond must have shape (B, T, cond_dim), got {tuple(cond.shape)}")
        if x.shape[:2] != cond.shape[:2]:
            raise ValueError(f"x and cond must share B,T dimensions, got {tuple(x.shape)} and {tuple(cond.shape)}")
        if x.shape[-1] != self.action_dim:
            raise ValueError(f"expected x action_dim={self.action_dim}, got {x.shape[-1]}")
        if cond.shape[-1] != self.cond_dim:
            raise ValueError(f"expected cond_dim={self.cond_dim}, got {cond.shape[-1]}")
        if timesteps.shape != (x.shape[0],):
            raise ValueError(f"timesteps must have shape ({x.shape[0]},), got {tuple(timesteps.shape)}")

        x_ch = x.transpose(1, 2)
        cond_ch = cond.transpose(1, 2)

        local_cond = self.local_cond_encoder(cond_ch)
        global_cond = self.global_cond_encoder(cond.mean(dim=1))
        time_cond = self.time_mlp(timesteps)
        film_cond = torch.cat([time_cond, global_cond], dim=-1)

        h = self.input_proj(torch.cat([x_ch, local_cond], dim=1))
        skips: List[torch.Tensor] = []

        for module in self.down_modules:
            h = module["res1"](h, film_cond)
            h = module["res2"](h, film_cond)
            skips.append(h)
            h = module["down"](h)

        for module in self.mid_modules:
            h = module(h, film_cond)

        for module in self.up_modules:
            skip = skips.pop()
            h = self._match_length(h, skip.shape[-1])
            h = torch.cat([h, skip], dim=1)
            h = module["res1"](h, film_cond)
            h = module["res2"](h, film_cond)
            h = module["up"](h)

        h = self._match_length(h, x_ch.shape[-1])
        out = self.final_conv[0](h, film_cond)
        out = self.final_conv[1](out)
        return out.transpose(1, 2)


if __name__ == "__main__":
    batch_size, horizon, action_dim, cond_dim = 4, 100, 6, 13
    model = ConditionalUnet1DArtist(action_dim=action_dim, cond_dim=cond_dim, hidden_dim=64)
    noisy = torch.randn(batch_size, horizon, action_dim)
    condition = torch.randn(batch_size, horizon, cond_dim)
    t = torch.randint(0, 100, (batch_size,))
    pred = model(noisy, condition, t)
    assert pred.shape == noisy.shape, f"expected {tuple(noisy.shape)}, got {tuple(pred.shape)}"
    print("ConditionalUnet1DArtist smoke test passed:", tuple(pred.shape))
