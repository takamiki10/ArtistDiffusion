#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 4 ]; then
    echo "Usage: $0 SHARD_ID CPU_CORE PATH_FILE OUTPUT_DIR"
    exit 1
fi

SHARD_ID="$1"
CPU_CORE="$2"
PATH_FILE="$3"
OUTPUT_DIR="$4"

PROJECT="/mnt/ssd/artistDiffusion/ArtistDiffusion/artist_trajectory_scoring"
VENV="/mnt/ssd/artistDiffusion/ArtistDiffusion/.venv/bin/activate"

cd "$PROJECT"
source "$VENV"

PRIOR="data/cartesian_expert_dataset_v3/adaptive_mlp_ik_bootstrap_prior"
V6="data/cartesian_expert_dataset_v3/diffusion_v6_strong_prior_residual_windows"

mapfile -t PATH_NAMES < "$PATH_FILE"

if [ "${#PATH_NAMES[@]}" -eq 0 ]; then
    echo "Shard $SHARD_ID has no paths."
    exit 0
fi

mkdir -p "$OUTPUT_DIR"

echo "Starting shard $SHARD_ID"
echo "CPU core: $CPU_CORE"
echo "Paths: ${#PATH_NAMES[@]}"
echo "Output: $OUTPUT_DIR"

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

taskset -c "$CPU_CORE" \
python generate_diffusion_v7_cost_improving_residual_targets.py \
  --train_prior "$PRIOR/train_prior.npz" \
  --train_windows "$V6/train_windows.npz" \
  --split_manifest "$V6/split_manifest.csv" \
  --output_dir "$OUTPUT_DIR" \
  --path_names "${PATH_NAMES[@]}" \
  --horizon 32 \
  --execution_horizon 8 \
  --targets_per_window 8 \
  --candidate_methods \
    jacobian_dls \
    sequential_ik \
  --min_cartesian_improvement_m 1e-5 \
  --min_cartesian_improvement_fraction 0.005 \
  --smoothness_relative_tolerance 0.10 \
  --boundary_absolute_tolerance 0.01 \
  --max_joint_step_gate 0.20 \
  --minimum_residual_distance 0.005 \
  --seed 42 \
  --device cpu \
  --resume \
  2>&1 | tee -a "$OUTPUT_DIR/generation.log"
