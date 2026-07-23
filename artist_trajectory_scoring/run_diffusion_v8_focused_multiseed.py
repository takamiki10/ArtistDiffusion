#!/usr/bin/env python3
"""Run the focused v8 teacher-forced evaluation across sampling seeds."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List, Sequence

import numpy as np


EXPECTED_K_VALUES = (1, 4, 8)
CHECKPOINT_FILES = {
    "--best_raw_total_checkpoint": "best_raw_total_loss_checkpoint.pt",
    "--best_ema_total_checkpoint": "best_ema_total_loss_checkpoint.pt",
    "--best_raw_epsilon_checkpoint": "best_raw_epsilon_loss_checkpoint.pt",
    "--best_ema_epsilon_checkpoint": "best_ema_epsilon_loss_checkpoint.pt",
    "--last_checkpoint": "last_checkpoint.pt",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run one focused v8 teacher-forced evaluation per diffusion "
            "sampling seed and aggregate the completed results."
        )
    )
    parser.add_argument("--dataset_dir", type=Path, required=True)
    parser.add_argument("--target_generation_dir", type=Path, required=True)
    parser.add_argument("--model_dir", type=Path, required=True)
    parser.add_argument("--results_root", type=Path, required=True)
    parser.add_argument(
        "--checkpoint_state", type=str, default="raw_last_epoch187"
    )
    parser.add_argument("--target_scale", type=float, default=1.0)
    parser.add_argument("--output_alpha", type=float, default=0.125)
    parser.add_argument(
        "--k_values", type=int, nargs="+", default=list(EXPECTED_K_VALUES)
    )
    parser.add_argument(
        "--sampling_seeds", type=int, nargs="+", default=[43, 44, 45, 46, 47]
    )
    parser.add_argument("--num_workers", type=int, default=10)
    parser.add_argument(
        "--device", choices=("auto", "cpu", "cuda"), default="cuda"
    )
    parser.add_argument("--historical_v7_rate", type=float, default=0.361)
    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.checkpoint_state.strip():
        raise ValueError("--checkpoint_state cannot be empty")
    for name in ("target_scale", "output_alpha"):
        value = float(getattr(args, name))
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"--{name} must be positive and finite")
    k_values = tuple(sorted(int(value) for value in args.k_values))
    if k_values != EXPECTED_K_VALUES or len(set(args.k_values)) != len(args.k_values):
        raise ValueError("--k_values must be exactly 1 4 8")
    if not args.sampling_seeds:
        raise ValueError("--sampling_seeds cannot be empty")
    if len(set(args.sampling_seeds)) != len(args.sampling_seeds):
        raise ValueError("--sampling_seeds cannot contain duplicates")
    if args.num_workers < 1:
        raise ValueError("--num_workers must be at least 1")
    historical = float(args.historical_v7_rate)
    if not np.isfinite(historical) or not 0.0 <= historical <= 1.0:
        raise ValueError("--historical_v7_rate must be finite and in [0, 1]")


def require_file(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def require_directory(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_dir():
        raise NotADirectoryError(resolved)
    return resolved


def command_text(command: Sequence[str]) -> str:
    return shlex.join(list(command))


def evaluator_command(
    args: argparse.Namespace,
    script_dir: Path,
    sampling_seed: int,
    output_dir: Path,
    checkpoint_paths: dict[str, Path],
) -> List[str]:
    command = [
        sys.executable,
        str(script_dir / "evaluate_diffusion_v8_teacher_forced_all_windows.py"),
        "--dataset_dir",
        str(args.dataset_dir),
        "--target_generation_dir",
        str(args.target_generation_dir),
    ]
    for argument, _filename in CHECKPOINT_FILES.items():
        command.extend((argument, str(checkpoint_paths[argument])))
    command.extend(
        (
            "--checkpoint_states",
            args.checkpoint_state,
            "--output_dir",
            str(output_dir),
            "--target_scales",
            format(float(args.target_scale), ".12g"),
            "--output_alphas",
            format(float(args.output_alpha), ".12g"),
            "--k_values",
            *(str(value) for value in sorted(args.k_values)),
            "--ddim_steps",
            "50",
            "--eta",
            "0.0",
            "--device",
            args.device,
            "--num_cpu_workers",
            str(args.num_workers),
            "--gpu_batch_size",
            str(max(EXPECTED_K_VALUES)),
            "--seed",
            "42",
            "--sampling_seed",
            str(sampling_seed),
        )
    )
    if args.smoke_test:
        command.extend(("--max_primary_windows", "5", "--no-include-difficult-paths"))
    else:
        command.append("--include_difficult_paths")
    if args.overwrite:
        command.append("--overwrite")
    return command


def summarizer_command(
    args: argparse.Namespace,
    script_dir: Path,
) -> List[str]:
    return [
        sys.executable,
        str(script_dir / "summarize_diffusion_v8_focused_multiseed.py"),
        "--results_root",
        str(args.results_root),
        "--checkpoint_state",
        args.checkpoint_state,
        "--target_scale",
        format(float(args.target_scale), ".12g"),
        "--output_alpha",
        format(float(args.output_alpha), ".12g"),
        "--k_values",
        *(str(value) for value in sorted(args.k_values)),
        "--sampling_seeds",
        *(str(value) for value in args.sampling_seeds),
        "--historical_v7_rate",
        format(float(args.historical_v7_rate), ".12g"),
    ]


def main() -> int:
    args = parse_args()
    validate_args(args)
    script_dir = Path(__file__).resolve().parent
    require_file(script_dir / "evaluate_diffusion_v8_teacher_forced_all_windows.py")
    require_file(script_dir / "summarize_diffusion_v8_focused_multiseed.py")
    args.dataset_dir = require_directory(args.dataset_dir)
    args.target_generation_dir = require_directory(args.target_generation_dir)
    args.model_dir = require_directory(args.model_dir)
    args.results_root = args.results_root.expanduser().resolve()
    checkpoint_paths = {
        argument: require_file(args.model_dir / filename)
        for argument, filename in CHECKPOINT_FILES.items()
    }

    seed_directories = {
        int(seed): args.results_root / f"seed_{int(seed)}"
        for seed in args.sampling_seeds
    }
    blocked = [
        directory
        for directory in seed_directories.values()
        if directory.is_dir() and any(directory.iterdir())
    ]
    if blocked and not args.overwrite:
        raise FileExistsError(
            "Focused seed directories are nonempty; pass --overwrite to reuse "
            f"them: {[str(path) for path in blocked]}"
        )
    args.results_root.mkdir(parents=True, exist_ok=True)

    for sampling_seed in args.sampling_seeds:
        output_dir = seed_directories[int(sampling_seed)]
        command = evaluator_command(
            args,
            script_dir,
            int(sampling_seed),
            output_dir,
            checkpoint_paths,
        )
        print(f"sampling seed {sampling_seed} evaluator command:")
        print(command_text(command), flush=True)
        subprocess.run(command, cwd=script_dir, check=True)

    command = summarizer_command(args, script_dir)
    print("focused multi-seed summarizer command:")
    print(command_text(command), flush=True)
    subprocess.run(command, cwd=script_dir, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
