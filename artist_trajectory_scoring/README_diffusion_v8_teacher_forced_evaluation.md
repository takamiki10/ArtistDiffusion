# Diffusion v8 All-Window Teacher-Forced Evaluation

`evaluate_diffusion_v8_teacher_forced_all_windows.py` evaluates whether the
trained v8 scale-conditioned residual diffusion model improves the frozen
strong prior under the same FK, safety, compatibility gates, and robot-aware
score used by v7 target generation and evaluation.

This is a teacher-forced window diagnostic. It does not perform recursive or
anchored rollout, train a model, or deploy to custom paths.

## Why All Windows

The v8 validation NPZ contains retained supervised target rows. It contains
270 unique windows from the 20 validation paths, not all 360 physical windows.
Windows for which target generation retained nothing are absent. Evaluating
only those rows would condition the scientific result on prior target-generator
success and overstate recovery.

The evaluator instead reconstructs every original 32-step window from the
authoritative v6 strong-prior source used by v7/v8 target generation. The
primary headline denominator is therefore:

```text
accepted unique validation windows / all unique validation windows
```

The expected primary population is 20 path-disjoint validation paths with 18
windows each, or 360 unique `(path_name, window_start)` pairs.

## Primary And Difficult Populations

The primary result uses the 20 paths marked `validation` in the v8
`path_split.csv`. Their target-covered and zero-target windows are also
reported separately.

`path_0306` and `path_0370` had no retained v8 supervised targets and therefore
do not appear in the 98-path train/validation split. By default they are
evaluated as a separate 36-window difficult stress test. Their results are not
included in the primary headline percentage. A combined row is diagnostic
only.

The evaluator fails if a validation path appears in training, a physical
window is duplicated or missing, or a path lacks starts `0,4,...,68`.

## No Target Leakage

Sampling and selection use only:

- the reconstructed condition;
- the frozen strong-prior joint window;
- the generated residual;
- FK and Cartesian tracking metrics;
- hard safety and v7 compatibility gates;
- the v7 robot-aware delta score.

Retained target identities label post-hoc target-covered subsets and audit the
condition reconstruction. Target residuals, target joint trajectories, target
improvement, target score, nearest-target distance, and oracle identity never
enter candidate generation or selection.

## Scale Conditioning And Output Alpha

The raw v7 condition has 38 features. V8 appends `target_scale` as feature 39
before normalization:

```text
[exact v7 condition (38), target_scale (1)]
```

The evaluator checks the feature order against `dataset_metadata.json`,
`normalization_stats.npz`, and every checkpoint. `target_scale` must be index
38, with recorded mean zero and standard deviation one. Defaults are:

```text
--target_scales 0.125 0.25 0.50 0.75 1.00
```

`target_scale` asks the model for a scale-conditioned residual distribution.
`output_alpha` is a separate post-generation calibration diagnostic:

```text
candidate_q = prior_q + output_alpha * generated_residual
```

The native result uses `--output_alphas 1.0`, because scale is already an
explicit model condition. Positive non-unit alphas are calibration diagnostics.
Base Gaussian DDIM samples are reused across output alphas so an alpha
diagnostic changes only residual application, not noise. Every CSV row records
both fields.

Reporting keeps the two scopes separate:

- `best_native_configuration`: the best primary configuration among rows with
  `output_alpha == 1.0`;
- `best_diagnostic_configuration`: the best primary configuration among every
  evaluated positive output alpha.

Diagnostic-only runs are valid. In that case the native optimum and native
headline are `null`, while diagnostic reporting, plots, and comparisons remain
available. `evaluation_summary.json`, `checkpoint_summary.csv`,
`scale_summary.csv`, and `v7_v8_comparison_summary.csv` identify the optimum
scope explicitly.

## Checkpoint State Deduplication

The evaluator accepts the best raw/EMA total-loss checkpoints, best raw/EMA
epsilon-loss checkpoints, and last checkpoint. Both raw and EMA states are
inspected inside every file. Variants receive explicit names such as
`raw_total_epoch487`, `ema_total_epoch500`, `raw_last_epoch500`, and
`ema_last_epoch500`.

Each state is hashed with SHA-256 over sorted parameter names, tensor dtype,
shape, and bytes. Exact duplicates are compared tensor-by-tensor and evaluated
only once. `checkpoint_state_manifest.csv` records the source path, state type,
epoch, hash, representative in `duplicate_of`, and whether the row was
evaluated. `--checkpoint_states` can restrict evaluation to exact deduplicated
variant labels such as `raw_last_epoch187`; it does not alter checkpoint
contents or state hashing.

## Sampling And Nested K

Sampling is true reverse DDIM from independent Gaussian noise. Defaults are:

```text
--ddim_steps 50
--eta 0.0
--k_values 1 4 8
--sampling_seed 42
```

For each unique checkpoint state, target scale, and physical window, the
evaluator generates `max(K)` candidates once. K=1, K=4, and K=8 use the nested
first-1, first-4, and first-8 samples. Stable SHA-256-derived seeds and IDs make
the subsets independent of worker completion order. No target or oracle
trajectory initializes denoising.

`--sampling_seed` seeds Python, NumPy, PyTorch CPU, PyTorch CUDA, and the stable
per-sample DDIM seed derivation. It changes stochastic diffusion inference only.
It does not change the path split, validation ordering, reconstructed windows,
strong priors, target coverage, checkpoint state, worker scoring, safety gates,
selection, or fallback. Omitting it uses legacy `--seed`, preserving historical
commands. The resolved sampling seed is recorded in configuration, window,
path, sample, bootstrap, JSON, and metadata outputs.

## FK, Safety, Cost, And Selection

The implementation imports the v7 evaluator's scientific path directly. It
uses:

- ROKAE xMateCR7;
- active joints `joint1` through `joint6`;
- end-effector `xMateCR7_link6`;
- `robot.update_cfg(cfg)` and `robot.get_transform(frame_to=...)`;
- horizon 32 and execution horizon 8;
- authoritative hard limits and tolerance;
- maximum joint step 0.20 rad;
- prefix Cartesian mean, RMS, p95, and maximum error;
- velocity, acceleration, jerk, entry/exit boundaries, manipulability, and
  singularity penalty;
- the complete v7 robot-aware delta score and compatibility gates.

A generated candidate is selectable only if it is hard-safe, lowers execution-
prefix Cartesian mean error, has strictly negative delta score, and passes all
v7 compatibility gates. Among selectable candidates in a nested K subset, the
minimum delta score wins. If none is selectable, the final output is asserted
equal to the unchanged strong prior. Any unsafe final selected/fallback output
is a fatal integrity error.

## Raw And Gated Classifications

Raw generation and final system behavior are classified separately.

- `RAW_GENERATOR_UNSAFE`: no generated sample is hard-safe.
- `RAW_GENERATOR_SAFE`: all generated samples are hard-safe and at least one is
  selectable.
- `RAW_GENERATOR_PARTIALLY_SAFE`: every other mixture of hard-safe and
  selectable samples.

The raw classification does not determine final deployed safety because unsafe
or nonimproving samples are rejected.

- `GATED_SYSTEM_UNSAFE`: at least one final selected/fallback output is unsafe.
- `GATED_SYSTEM_NO_GAIN`: final outputs are safe but there is no positive gated
  gain.
- `GATED_SYSTEM_MEANINGFUL_GAIN`: the primary accepted-window rate is above the
  historical v7 rate and its 95% path-bootstrap lower bound is also above v7.
- `GATED_SYSTEM_SMALL_GAIN`: positive safe gain that does not meet that
  uncertainty-aware threshold.

Fallback alone never makes the gated system unsafe.

## Path-Level Bootstrap

The evaluator performs deterministic path-level resampling with replacement.
All windows belonging to a sampled validation path remain together, preserving
within-path correlation. It does not bootstrap individual windows.

With `--bootstrap_samples 2000 --bootstrap_seed 42`, it reports 95% percentile
intervals for every primary configuration's:

- accepted-window rate;
- mean Cartesian improvement over all windows;
- mean robot-aware improvement (`-delta_score`) over accepted windows.

The v7 comparison uses 36.1% as an externally supplied historical accepted-
window reference. The evaluator does not claim significant improvement unless
the path-level interval supports it.

## Multiprocessing Architecture

CUDA model loading, schedule construction, seeded DDIM, denormalization, and
candidate joint construction remain in the main process. `--gpu_batch_size`
limits simultaneously generated CUDA samples.

With `--num_cpu_workers` greater than one, one persistent
`ProcessPoolExecutor` uses the `spawn` context. Every worker sets PyTorch and
common numerical libraries to one CPU thread, creates one reusable xMateCR7
robot in its initializer, and receives only CPU NumPy arrays plus serializable
metadata. The serial and multiprocessing paths call the same v7 candidate
evaluation function. Results are indexed by stable candidate ID and restored
to canonical order before nested-K selection or CSV writing.

## Pilot

This five-window pilot still performs real checkpoint loading, Gaussian DDIM,
FK, safety, robot-aware scoring, fallback, and output writing:

```bash
python evaluate_diffusion_v8_teacher_forced_all_windows.py \
  --dataset_dir data/cartesian_expert_dataset_v3/diffusion_v8_multitarget_scaled_training_dataset_100paths \
  --target_generation_dir data/cartesian_expert_dataset_v3/diffusion_v8_multitarget_scaled_residual_targets_100paths \
  --output_dir results/diffusion_v8_teacher_forced_pilot_seed42 \
  --max_primary_windows 5 \
  --no-include-difficult-paths \
  --target_scales 0.125 0.25 0.50 0.75 1.00 \
  --output_alphas 1.0 \
  --k_values 1 4 8 \
  --ddim_steps 50 \
  --num_cpu_workers 8 \
  --gpu_batch_size 8 \
  --device cuda \
  --sampling_seed 42 \
  --overwrite
```

## Full Evaluation

```bash
python evaluate_diffusion_v8_teacher_forced_all_windows.py \
  --dataset_dir data/cartesian_expert_dataset_v3/diffusion_v8_multitarget_scaled_training_dataset_100paths \
  --target_generation_dir data/cartesian_expert_dataset_v3/diffusion_v8_multitarget_scaled_residual_targets_100paths \
  --source_windows_npz data/cartesian_expert_dataset_v3/diffusion_v6_strong_prior_residual_windows/train_windows.npz \
  --best_raw_total_checkpoint models/diffusion_v8_multitarget_scaled_residual_unet_100paths_seed42/best_raw_total_loss_checkpoint.pt \
  --best_ema_total_checkpoint models/diffusion_v8_multitarget_scaled_residual_unet_100paths_seed42/best_ema_total_loss_checkpoint.pt \
  --best_raw_epsilon_checkpoint models/diffusion_v8_multitarget_scaled_residual_unet_100paths_seed42/best_raw_epsilon_loss_checkpoint.pt \
  --best_ema_epsilon_checkpoint models/diffusion_v8_multitarget_scaled_residual_unet_100paths_seed42/best_ema_epsilon_loss_checkpoint.pt \
  --last_checkpoint models/diffusion_v8_multitarget_scaled_residual_unet_100paths_seed42/last_checkpoint.pt \
  --output_dir results/diffusion_v8_teacher_forced_all_windows_seed42 \
  --target_scales 0.125 0.25 0.50 0.75 1.00 \
  --output_alphas 1.0 \
  --k_values 1 4 8 \
  --ddim_steps 50 \
  --eta 0.0 \
  --bootstrap_samples 2000 \
  --bootstrap_seed 42 \
  --num_cpu_workers 8 \
  --gpu_batch_size 8 \
  --include_difficult_paths \
  --save_per_sample_results \
  --device cuda \
  --sampling_seed 42 \
  --overwrite
```

## Focused Multi-Seed Confirmation

The focused runner evaluates the current epsilon-only configuration with one
independent output directory per diffusion sampling seed:

```bash
python run_diffusion_v8_focused_multiseed.py \
  --dataset_dir data/cartesian_expert_dataset_v3/diffusion_v8_multitarget_scaled_training_dataset_100paths \
  --target_generation_dir data/cartesian_expert_dataset_v3/diffusion_v8_multitarget_scaled_residual_targets_100paths \
  --model_dir models/diffusion_v8_multitarget_scaled_residual_unet_100paths_epsilon_only_seed42 \
  --results_root results/diffusion_v8_focused_multiseed_raw_last_epoch187 \
  --checkpoint_state raw_last_epoch187 \
  --target_scale 1.0 \
  --output_alpha 0.125 \
  --k_values 1 4 8 \
  --sampling_seeds 43 44 45 46 47 \
  --num_workers 10 \
  --device cuda \
  --historical_v7_rate 0.361
```

Each evaluation is written to `<results_root>/seed_<seed>`. The runner refuses
to reuse a nonempty seed directory unless `--overwrite` is supplied, prints
the exact list-form evaluator command, and calls the standalone summarizer only
after every requested seed succeeds. `--smoke_test` limits the primary
population to five windows and omits the difficult stress-test paths while
still exercising checkpoint loading, nested K, native/diagnostic reporting,
result writing, and multi-seed aggregation.

The standalone aggregation command has the same focused configuration:

```bash
python summarize_diffusion_v8_focused_multiseed.py \
  --results_root results/diffusion_v8_focused_multiseed_raw_last_epoch187 \
  --checkpoint_state raw_last_epoch187 \
  --target_scale 1.0 \
  --output_alpha 0.125 \
  --k_values 1 4 8 \
  --sampling_seeds 43 44 45 46 47 \
  --historical_v7_rate 0.361
```

It creates:

- `focused_multiseed_per_seed.csv`;
- `focused_multiseed_aggregate.csv`;
- `focused_multiseed_aggregate.json`;
- `focused_multiseed_per_path.csv`;
- `focused_multiseed_report.txt`.

The per-path file contains seed-level path results plus mean, minimum, maximum,
and seed-coverage stability fields. The pooled accepted rate is descriptive
only: repeated seeds evaluate the same physical windows, so windows across
seeds are not treated as statistically independent.

The provisional engineering decision rule is:

```text
advance_to_anchored_rollout =
    K8 mean accepted rate >= 0.40
    AND at least 4 of 5 seeds exceed 0.361
    AND every seed has final_safe_window_rate == 1.0
    AND mean fraction_paths_with_at_least_one_accepted_window >= 0.75
```

This is not a formal statistical significance test. The 36 windows from
`path_0306` and `path_0370` remain a separate difficult-path stress test and
never enter the 360-window primary accepted rate.

## Outputs

- `evaluation_summary.json`: native and diagnostic optima, diagnostic headline
  numerator and denominator, subset results, classifications, sampling seed,
  interval, and v7 comparison.
- `per_sample_results.csv`: raw sample gates, score, metrics, seed, and nested-K
  selections; header-only unless `--save_per_sample_results` is enabled.
- `per_window_results.csv`: selected/fallback output and prior/selected metrics
  for every state, scale, alpha, K, and physical window.
- `configuration_summary.csv`: required sample- and window-level metrics for
  every configuration and every available evaluation subset.
- `per_path_summary.csv`: accepted counts, fallback, safety, and improvement by
  path and configuration.
- `scale_summary.csv`: native and diagnostic best primary checkpoint rows for
  each scale, alpha, and K.
- `checkpoint_summary.csv`: native and diagnostic best primary result for each
  unique state.
- `checkpoint_state_manifest.csv`: source/state/hash deduplication audit.
- `target_coverage_subset_summary.csv`: target-covered and zero-target primary
  configuration summaries.
- `difficult_path_summary.csv`: separate no-target-path stress-test summaries.
- `bootstrap_confidence_intervals.csv`: deterministic path-level intervals.
- `v7_v8_comparison_summary.csv`: separate native and diagnostic historical-v7
  comparisons and uncertainty status.
- `timing_summary.json`: generated/scored counts, GPU time, worker CPU time,
  scoring wall time, and total wall time.
- `evaluation_metadata.json`: feature, population, checkpoint, robot, safety,
  selection, leakage, multiprocessing, and classification contracts.

The six PNG files summarize acceptance, hard safety, improvement, per-path
coverage, fallback, and representative selected Cartesian trajectories. Plots
use Matplotlib only.

## Advancement Criteria

Before starting anchored rollout, require all of the following provisional
evidence:

- final safe output rate is 100%;
- accepted-window rate is clearly above v7's 36.1%;
- the preferred target scale is stable across validation paths;
- improvement is not dominated by a few paths;
- zero-target primary windows show nonzero recovery;
- the path-level bootstrap interval supports a meaningful improvement;
- K=1 improves relative to v7 even if K=8 remains best.

The evaluator reports this evidence but never launches anchored rollout.
