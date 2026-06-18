# Time-Conditioned MLP Baseline

These scripts implement the next step from `handoff.md`.

## 1. Copy scripts into the project folder

```bash
cd /workspace/artist_trajectory_scoring
```

Put these files there:

```text
train_time_conditioned_mlp.py
predict_time_conditioned_mlp.py
evaluate_time_conditioned_mlp.py
```

## 2. Activate environment

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate robodiff
```

## 3. Train on the 2000-path dataset

```bash
python train_time_conditioned_mlp.py \
  --npz data/synthetic_paths_train_2000/multipath_episodes.npz \
  --epochs 2000 \
  --batch_size 4096 \
  --hidden_dim 256 \
  --num_layers 4 \
  --output_model data/synthetic_paths_train_2000/time_conditioned_mlp.pt
```

## 4. Evaluate on unseen test set

```bash
python evaluate_time_conditioned_mlp.py \
  --model data/synthetic_paths_train_2000/time_conditioned_mlp.pt \
  --dataset_dir data/synthetic_paths_test \
  --score_script score_trajectory.py \
  --output_csv data/synthetic_paths_test/time_conditioned_eval_train2000.csv
```

## 5. Compare result

Current flattened MLP baseline from handoff:

```text
mean path_error ≈ 0.000502
RMS Cartesian error ≈ 2.2 cm
```

Decision:

```text
If time-conditioned MLP mean path_error < 0.000502:
    preserving timestep structure helped.

If not:
    per-timestep model probably lacks global path context.
    Next model should be GRU/Transformer/diffusion-style sequence model.
```
