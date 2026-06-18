# Cartesian Test Path Scripts

These scripts test whether the current best model, the time-conditioned MLP, can follow drawing-like paths that were not generated from random joint trajectories.

## Files

```text
generate_cartesian_test_paths.py
evaluate_cartesian_test_paths.py
plot_cartesian_test_paths.py
```

## 1. Generate drawing-like paths

```bash
cd /workspace/artist_trajectory_scoring

python generate_cartesian_test_paths.py \
  --output_dir data/cartesian_test_paths \
  --num_steps 100 \
  --duration 1.0
```

## 2. Evaluate the time-conditioned MLP

```bash
source /opt/conda/etc/profile.d/conda.sh
conda activate robodiff

python evaluate_cartesian_test_paths.py \
  --model data/synthetic_paths_train_2000/time_conditioned_mlp.pt \
  --dataset_dir data/cartesian_test_paths \
  --predict_script predict_time_conditioned_mlp.py \
  --score_script score_trajectory.py \
  --output_csv data/cartesian_test_paths/time_conditioned_eval.csv
```

## 3. Plot all results

```bash
python plot_cartesian_test_paths.py \
  --dataset_dir data/cartesian_test_paths \
  --plot_script plot_diffusion_diagnostic.py
```

Each folder gets:

```text
time_conditioned_plot.png
```

## 4. What to check

Good:
- FK path overlaps desired path
- error remains low
- joint trajectories are smooth

Bad:
- FK path leaves the drawing area
- endpoint error spikes
- joints oscillate heavily
