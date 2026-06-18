# Sequence GRU Baseline

These scripts implement the next baseline after the time-conditioned MLP.

## Files

```text
train_sequence_gru.py
predict_sequence_gru.py
evaluate_sequence_gru.py
```

## 1. Copy scripts into project folder

```bash
cd /workspace/artist_trajectory_scoring
```

Place the three Python scripts there.

## 2. Activate environment

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate robodiff
```

## 3. Train bidirectional GRU

```bash
python train_sequence_gru.py \
  --npz data/synthetic_paths_train_2000/multipath_episodes.npz \
  --epochs 2000 \
  --batch_size 64 \
  --hidden_dim 256 \
  --num_layers 2 \
  --bidirectional \
  --output_model data/synthetic_paths_train_2000/sequence_gru.pt
```

For a faster first test, use:

```bash
python train_sequence_gru.py \
  --npz data/synthetic_paths_train_2000/multipath_episodes.npz \
  --epochs 200 \
  --batch_size 64 \
  --hidden_dim 256 \
  --num_layers 2 \
  --bidirectional \
  --output_model data/synthetic_paths_train_2000/sequence_gru_test.pt
```

## 4. Evaluate on unseen test paths

```bash
python evaluate_sequence_gru.py \
  --model data/synthetic_paths_train_2000/sequence_gru.pt \
  --dataset_dir data/synthetic_paths_test \
  --score_script score_trajectory.py \
  --output_csv data/synthetic_paths_test/sequence_gru_eval_train2000.csv
```

## 5. Compare to current baseline

Current best baseline:

```text
time-conditioned MLP mean path_error ≈ 5.16256127e-05
RMS error ≈ 7.2 mm
```

Decision:

```text
If GRU mean path_error < 5.16256127e-05:
    sequence context helped.

If GRU mean path_error >= 5.16256127e-05:
    the synthetic dataset may be simple enough that local [x,y,z,t] already works well,
    or the GRU needs tuning.
```

## 6. Visualize worst case

After evaluation, find the worst path in the CSV, then run:

```bash
python plot_diffusion_diagnostic.py \
  --desired_path data/synthetic_paths_test/path_019/desired_path.csv \
  --ee_csv data/synthetic_paths_test/path_019/sequence_gru_pred_ee.csv \
  --q_csv data/synthetic_paths_test/path_019/sequence_gru_pred_q.csv \
  --output_png data/synthetic_paths_test/path_019/sequence_gru_plot.png
```
