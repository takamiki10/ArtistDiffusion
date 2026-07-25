#!/usr/bin/env python3
"""Run the frozen v8 anchored rollout for sampling seeds 43 through 47."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


FROZEN_SEEDS = (43, 44, 45, 46, 47)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset_dir",
        type=Path,
        default=Path(
            "data/cartesian_expert_dataset_v3/"
            "diffusion_v8_multitarget_scaled_training_dataset_100paths"
        ),
    )
    parser.add_argument(
        "--target_generation_dir",
        type=Path,
        default=Path(
            "data/cartesian_expert_dataset_v3/"
            "diffusion_v8_multitarget_scaled_residual_targets_100paths"
        ),
    )
    parser.add_argument(
        "--model_dir",
        type=Path,
        default=Path(
            "models/"
            "diffusion_v8_multitarget_scaled_residual_unet_100paths_"
            "epsilon_only_seed42"
        ),
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path("results/diffusion_v8_anchored_recursive_multiseed"),
    )
    parser.add_argument("--checkpoint_state", default="raw_last_epoch187")
    parser.add_argument("--target_scale", type=float, default=1.0)
    parser.add_argument("--output_alpha", type=float, default=0.125)
    parser.add_argument("--k_values", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--ddim_steps", type=int, default=50)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--execution_horizon", type=int, default=8)
    parser.add_argument("--anchoring_horizon", type=int, default=8)
    parser.add_argument("--num_cpu_workers", type=int, default=8)
    parser.add_argument("--gpu_batch_size", type=int, default=8)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--include_difficult_paths",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--continue_on_error", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--evaluator",
        type=Path,
        default=Path(__file__).with_name(
            "evaluate_diffusion_v8_anchored_recursive_rollout.py"
        ),
    )
    parser.add_argument(
        "--summarizer",
        type=Path,
        default=Path(__file__).with_name(
            "summarize_diffusion_v8_anchored_multiseed.py"
        ),
    )
    return parser.parse_args()


def evaluator_command(
    args: argparse.Namespace,
    seed: int,
    output_dir: Path,
) -> List[str]:
    command = [
        args.python,
        str(args.evaluator),
        "--dataset_dir",
        str(args.dataset_dir),
        "--target_generation_dir",
        str(args.target_generation_dir),
        "--model_dir",
        str(args.model_dir),
        "--output_dir",
        str(output_dir),
        "--checkpoint_state",
        args.checkpoint_state,
        "--target_scale",
        str(args.target_scale),
        "--output_alpha",
        str(args.output_alpha),
        "--k_values",
        *(str(value) for value in args.k_values),
        "--sampling_seed",
        str(seed),
        "--ddim_steps",
        str(args.ddim_steps),
        "--eta",
        str(args.eta),
        "--horizon",
        str(args.horizon),
        "--execution_horizon",
        str(args.execution_horizon),
        "--anchoring_horizon",
        str(args.anchoring_horizon),
        "--num_cpu_workers",
        str(args.num_cpu_workers),
        "--gpu_batch_size",
        str(args.gpu_batch_size),
        "--device",
        args.device,
        (
            "--include_difficult_paths"
            if args.include_difficult_paths
            else "--no-include_difficult_paths"
        ),
    ]
    if args.overwrite:
        command.append("--overwrite")
    return command


def main() -> int:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    statuses: List[Dict[str, Any]] = []
    for run_index, seed in enumerate(FROZEN_SEEDS, start=1):
        output_dir = args.output_root / f"seed_{seed}"
        if output_dir.exists() and any(output_dir.iterdir()) and not args.overwrite:
            raise FileExistsError(
                f"Seed output is nonempty: {output_dir}; pass --overwrite"
            )
        command = evaluator_command(args, seed, output_dir)
        print(f"[{run_index}/{len(FROZEN_SEEDS)}] {shlex.join(command)}")
        log_path = args.output_root / f"seed_{seed}.log"
        with log_path.open("w") as log:
            completed = subprocess.run(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        statuses.append(
            {
                "sampling_seed": seed,
                "return_code": completed.returncode,
                "success": completed.returncode == 0,
                "output_dir": str(output_dir),
                "log_path": str(log_path),
            }
        )
        (args.output_root / "anchored_multiseed_status.json").write_text(
            json.dumps({"runs": statuses}, indent=2) + "\n"
        )
        if completed.returncode != 0 and not args.continue_on_error:
            return completed.returncode

    successful_count = sum(bool(row["success"]) for row in statuses)
    if successful_count == 0:
        return 1
    summary_command = [
        args.python,
        str(args.summarizer),
        "--input_root",
        str(args.output_root),
    ]
    if successful_count != len(FROZEN_SEEDS):
        summary_command.append("--allow_incomplete")
    print(shlex.join(summary_command))
    completed_summary = subprocess.run(summary_command, check=False)
    return completed_summary.returncode


if __name__ == "__main__":
    raise SystemExit(main())
