# Cartesian Expert Dataset v2

This workflow creates a larger supervised Cartesian expert dataset for the
path-conditioned MLP. It does not add diffusion. The goal is to generate many
deliberate Cartesian drawing paths, solve sequential numerical IK for each, and
train on the accepted expert demonstrations.

The current 8-path expert set proved the path-conditioned MLP can overfit known
paths, but square and zigzag paths still show rounded/deformed sharp corners.
This v2 dataset intentionally includes extra square and zigzag variations.

## 1. Generate randomized raw Cartesian paths

```bash
python generate_cartesian_path_variations.py \
  --output_dir data/cartesian_expert_dataset_v2/raw_paths \
  --num_train 80 \
  --num_test 20 \
  --timesteps 100 \
  --seed 0
```

This creates:

```text
data/cartesian_expert_dataset_v2/raw_paths/
  train/path_0001/desired_path.csv
  train/path_0001/path_meta.json
  test/path_0001/desired_path.csv
  test/path_0001/path_meta.json
```

Each `desired_path.csv` has columns:

```text
t,x,y,z
```

Each `path_meta.json` records the path type and randomized parameters.

## 2. Generate IK experts

```bash
python batch_generate_ik_experts.py \
  --raw_dir data/cartesian_expert_dataset_v2/raw_paths \
  --output_dir data/cartesian_expert_dataset_v2/experts \
  --split all \
  --smooth_weight 0.01 \
  --num_restarts 50 \
  --retry_error_threshold 0.02 \
  --max_mean_error 0.010 \
  --max_max_error 0.030 \
  --overwrite
```

Accepted experts are written to:

```text
data/cartesian_expert_dataset_v2/experts/train/path_0001/
  desired_path.csv
  expert_q.csv
  expert_ee.csv
  metrics.json
  plot.png
  path_meta.json
```

Rejected paths are kept under:

```text
data/cartesian_expert_dataset_v2/rejected/
```

The batch script also writes:

```text
data/cartesian_expert_dataset_v2/experts/ik_generation_summary.csv
```

Use this file to inspect accepted/rejected paths and identify hard path types.

## 3. Build train/test NPZ files

```bash
python build_cartesian_expert_npz.py \
  --dataset_dir data/cartesian_expert_dataset_v2/experts/train \
  --output_npz data/cartesian_expert_dataset_v2/train_episodes.npz
```

```bash
python build_cartesian_expert_npz.py \
  --dataset_dir data/cartesian_expert_dataset_v2/experts/test \
  --output_npz data/cartesian_expert_dataset_v2/test_episodes.npz
```

The NPZ files contain:

```text
desired_paths: (N,T,3)
actions:       (N,T,6)
times:         (N,T)
path_ids:      (N,)
```

## 4. Train the path-conditioned MLP

```bash
python train_path_conditioned_mlp.py \
  --npz data/cartesian_expert_dataset_v2/train_episodes.npz \
  --epochs 3000 \
  --batch_size 512 \
  --hidden_dim 256 \
  --num_layers 4 \
  --output_model data/cartesian_expert_dataset_v2/path_conditioned_mlp_v2.pt \
  --device cpu
```

The path-conditioned model receives the flattened full desired path plus the
current timestep and current desired point, giving it global path context.

## 5. Evaluate on held-out test paths

```bash
python evaluate_path_conditioned_mlp.py \
  --model data/cartesian_expert_dataset_v2/path_conditioned_mlp_v2.pt \
  --dataset_dir data/cartesian_expert_dataset_v2/experts/test \
  --output_csv data/cartesian_expert_dataset_v2/test_eval.csv \
  --device cpu
```

The evaluator writes predicted q CSVs, computes FK through `score_trajectory.py`,
and reports:

```text
mean path_error
RMS Cartesian error
```

## Notes

- Do not implement diffusion yet.
- If too many paths are rejected, inspect `ik_generation_summary.csv` and the
  rejected plots before loosening thresholds.
- Square and zigzag paths may remain harder because their sharp corners require
  more abrupt joint-space changes.
