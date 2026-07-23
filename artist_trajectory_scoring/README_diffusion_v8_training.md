# Diffusion v8 Training

V8 trains the existing conditional residual 1D U-Net on multiple independently
validated targets per window. It preserves the v7 epsilon-prediction model,
linear DDPM schedule, tensor ordering, AdamW optimizer, and EMA implementation.
The controlled changes are the 39-dimensional scale-conditioned dataset,
balanced row sampling, reconstructed-x0 auxiliary losses, and checkpoint
diagnostics.

Training loss is not evidence of better robot behavior. Periodic checkpoints
must later be compared with all-window, path-disjoint teacher-forced FK and
robot-aware evaluation. No FK or robot model is initialized by the trainer.

## Dataset

Pass the dataset directory with `--dataset_dir`. The trainer requires:

- `train_windows.npz`
- `validation_windows.npz`
- `normalization_stats.npz`
- `dataset_metadata.json`
- `train_rows.csv`
- `validation_rows.csv`

The NPZ inputs provide `condition_norm` with shape `(N,32,39)` and
`residual_q_norm` with shape `(N,32,6)`. Physical `residual_q` is used only to
verify normalization and derive auxiliary-loss scales. The trainer consumes
the stored normalized arrays directly and does not normalize them again.

The final condition channel is `target_scale`. It remains an interpretable raw
value in both `condition` and `condition_norm`; its recorded normalization is
mean zero and standard deviation one. All other condition and residual
statistics come from training rows only.

The trainer verifies path and physical-window disjointness, metadata counts,
feature ordering, finite values, normalization reconstruction, and training-
only statistics before creating the model. `path_0306` and `path_0370` have no
retained supervised targets and are intentionally not given artificial zero
targets. They remain difficult all-window evaluation paths.

## Balanced Sampling

Balanced sampling is the default. For training row `i`, define:

```text
window_factor_i = count(path_name_i, window_start_i)^(-window_balance_power)
scale_factor_i  = count(target_scale_i)^(-scale_balance_power)
quality_factor_i = quality_weight_i^(quality_weight_power)

raw_weight_i = window_factor_i * scale_factor_i * quality_factor_i
```

The raw weights are scaled and clipped so their final mean is exactly one and
they remain within `sampler_weight_clip_min` and
`sampler_weight_clip_max`. A seeded `WeightedRandomSampler` draws exactly one
training-row count per epoch with replacement. Sampling weights are not
multiplied into the loss, so quality is not applied twice. `--sampling_mode
uniform` is available only as an ablation; validation is always sequential,
deterministic, unweighted, and unbalanced.

`sampling_diagnostics.csv` contains per-scale and per-window-target-count
summaries. `training_metadata.json` also records original and expected sampled
scale distributions, sampler-weight statistics, effective sample size, and
expected draws for every physical window.

## Diffusion And Losses

The model predicts epsilon for normalized residual trajectories:

```text
x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * epsilon

x0_hat = (x_t - sqrt(1 - alpha_bar_t) * epsilon_hat)
         / sqrt(alpha_bar_t)
```

`x0_hat` remains attached to the computation graph. It is denormalized once
using the training residual mean and standard deviation before auxiliary
losses are calculated.

Let `S0`, `S1`, `S2`, and `S3` be robust per-joint scales for residual
position and first, second, and third temporal differences. Each scale uses a
finite nonzero standard deviation when available, then `1.4826 * MAD`, then a
documented epsilon floor. They are derived only from physical training targets
and saved in `auxiliary_loss_normalization.npz`.

The losses are:

```text
L_epsilon = mean((epsilon_hat - epsilon)^2)

L_x0 = weighted_mean(((x0_hat_physical - x0_physical) / S0)^2)

L_velocity = weighted_mean((diff1(error_physical) / S1)^2)
L_acceleration = weighted_mean((diff2(error_physical) / S2)^2)
L_jerk = weighted_mean((diff3(error_physical) / S3)^2)

L_boundary = supervised normalized error at:
             first residual point, last residual point,
             first velocity, and last velocity

L_total = lambda_epsilon * L_epsilon
        + lambda_x0 * L_x0
        + lambda_velocity * L_velocity
        + lambda_acceleration * L_acceleration
        + lambda_jerk * L_jerk
        + lambda_boundary * L_boundary
```

Boundary loss matches the retained target boundaries; it does not force them
to zero. Position and derivative losses weight quantities associated with the
first `execution_horizon` steps by `execution_prefix_weight`. Difference
orders map the prefix to `E-1`, `E-2`, and `E-3` entries. Every temporal weight
vector is normalized to mean one, so changing prefix emphasis does not simply
rescale the loss.

## Validation And EMA

Fixed CPU-generated timestep and noise banks make epoch-to-epoch validation
directly comparable. Raw and EMA states are evaluated separately over every
validation row. Reports include all loss components, physical full-window and
execution-prefix residual RMSE, per-target-scale metrics, and early/middle/
late diffusion-timestep bins.

The raw and EMA states are also evaluated on the same deterministic,
unweighted training pass for directly comparable training diagnostics.
Balanced optimization-pass metrics are recorded separately with an
`optimization_raw_` prefix.

## Data Loading And Seeds

Training and validation DataLoaders are created once. With the defaults they
use six workers, persistent workers, prefetching, CUDA-pinned memory, and
non-blocking transfers. Each worker receives a deterministic PyTorch-derived
seed, sets PyTorch to one CPU thread, and requests one OpenMP/MKL/OpenBLAS/
NumExpr thread to avoid oversubscription. GPU training stays in the main
process; no `ProcessPoolExecutor` is used.

Python, NumPy, PyTorch CPU, PyTorch CUDA, sampler, DataLoader workers, and fixed
validation banks are seeded. Strict deterministic algorithms are optional via
`--deterministic_algorithms`; without them, sampling remains seeded but exact
bitwise CUDA reproducibility is not guaranteed.

CUDA AMP uses dynamic loss scaling with a conservative initial scale of 128,
configurable through `--amp_initial_scale`. A detected scaled-gradient
overflow skips that optimizer step, lowers the scale, and does not update EMA;
overflow counts and the resulting loss scale are recorded in training history.

## Checkpoints

Training writes:

- `best_raw_total_loss_checkpoint.pt`
- `best_ema_total_loss_checkpoint.pt`
- `best_raw_epsilon_loss_checkpoint.pt`
- `best_ema_epsilon_loss_checkpoint.pt`
- `last_checkpoint.pt`
- `checkpoints/epoch_0025.pt`, `epoch_0050.pt`, and so on

Every checkpoint contains raw and EMA model states, optimizer state, AMP
scaler state, the explicit null scheduler state, epoch and early-stopping
state, model configuration, diffusion schedule, feature ordering,
normalization data and hashes, auxiliary scales, sampler configuration,
dataset metadata, random-generator states, history, and command arguments.

EMA total validation loss controls early stopping by default. The four “best”
files describe conventional supervised metrics only. Periodic and last
checkpoints are retained because v7 showed that the best downstream generative
checkpoint need not minimize conventional validation loss.

Resume with `--resume_checkpoint`. Resume restores raw/EMA states, optimizer,
AMP scaler, epoch, best metrics, early-stopping counter, sampler/DataLoader
generator states, and global random states. It rejects differences in model
configuration, condition dimension, target shape, feature order,
normalization, dataset metadata, auxiliary scales, loss weights, sampler
configuration, optimizer, EMA decay, AMP mode, execution-prefix settings, or
diffusion schedule.

`--init_checkpoint` initializes only model weights and requires an exactly
compatible 39-D model, architecture, feature order, target shape, and
normalization. A v7 38-D checkpoint is rejected.

## Smoke Test

```bash
python train_conditional_diffusion_trajectory_v8.py \
  --dataset_dir data/cartesian_expert_dataset_v3/diffusion_v8_multitarget_scaled_training_dataset_100paths \
  --output_dir models/diffusion_v8_multitarget_smoke_seed42 \
  --epochs 2 \
  --max_train_batches 4 \
  --max_validation_batches 2 \
  --device cuda \
  --seed 42 \
  --overwrite
```

A successful limited run prints `V8_TRAINING_SMOKE_TEST_COMPLETE`, saves all
non-periodic checkpoints and diagnostics, and verifies that every enabled loss
component has a finite nonzero model gradient.

## Full Training

```bash
python train_conditional_diffusion_trajectory_v8.py \
  --dataset_dir data/cartesian_expert_dataset_v3/diffusion_v8_multitarget_scaled_training_dataset_100paths \
  --output_dir models/diffusion_v8_multitarget_seed42 \
  --device cuda \
  --seed 42 \
  --epochs 500 \
  --batch_size 256 \
  --learning_rate 1e-4 \
  --weight_decay 1e-6 \
  --num_data_workers 6 \
  --prefetch_factor 2 \
  --sampling_mode balanced \
  --checkpoint_interval 25 \
  --early_stopping_patience 80 \
  --overwrite
```

## Outputs

Besides checkpoints, the output directory contains:

- `training_history.csv` and `.json`: epoch-level raw/EMA train and validation
  components, physical RMSEs, gradient norm, timing, early stopping, and
  sampled-scale counts.
- `validation_loss_by_scale.csv`: raw and EMA validation metrics by target
  scale and epoch.
- `validation_loss_by_timestep_bin.csv`: raw and EMA validation metrics by
  diffusion-noise bin and epoch.
- `sampling_diagnostics.csv`: scale and window-multiplicity sampler summaries.
- `training_and_validation_loss.png` and `loss_component_history.png`.
- `training_metadata.json`, `dataset_integrity_report.json`, and
  `auxiliary_loss_normalization.npz`.

After training, run the dedicated all-window teacher-forced FK evaluation over
periodic, last, and metric-selected checkpoints. Loss curves alone must not be
used to claim drawing improvement or deployment safety.
