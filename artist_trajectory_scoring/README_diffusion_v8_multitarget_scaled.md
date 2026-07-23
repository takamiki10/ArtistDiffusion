# ArtistDiffusion v8: Multitarget Scaled Residuals

## Purpose

V8 is a target-generation and dataset-building revision. It does not introduce
or train a new diffusion model.

The full v7 teacher-forced evaluation showed that useful corrections existed
for only about 36.1% of validation windows in the best configuration. K=1 was
ineffective, best-of-K sampling was necessary, and residual scales around 0.25
or 0.50 were often more useful than 1.0. Many samples improved Cartesian
tracking while degrading acceleration, jerk, boundary continuity, or the total
robot-aware score. V7 also retained too few targets per condition to represent
a useful conditional distribution.

V8 addresses the data side of those findings by:

1. generating deterministic restarts of the established v7 optimization-based
   candidate methods;
2. retaining multiple robot-aware, residual-space-diverse base targets;
3. independently evaluating scaled versions of every retained base target;
4. appending target scale to the conditioning tensor; and
5. creating a deterministic path-disjoint train/validation split.

Do not train a v8 model until the pilot summaries described below have been
inspected.

## Scientific Authority

`generate_diffusion_v8_multitarget_scaled_residual_targets.py` imports and
reuses `generate_diffusion_v7_cost_improving_residual_targets.py` for:

- candidate generation;
- xMateCR7 FK;
- Cartesian tracking metrics;
- hard joint-limit checks and maximum-joint-step checks;
- velocity, acceleration, and jerk costs;
- entry and exit boundary metrics;
- manipulability and singularity penalties;
- v7 acceptance reasons;
- Pareto-front calculation; and
- the complete robot-aware `delta_score`.

The active joints remain `joint1` through `joint6`, in that order. FK remains
`robot.update_cfg(cfg)` followed by
`robot.get_transform(frame_to="xMateCR7_link6")`.

## Valid Targets

A base or scaled candidate is valid only when all of the following hold:

1. all explicit v7 acceptance conditions pass;
2. the v7 `hard_safe` check passes;
3. execution-prefix Cartesian mean error improves strictly; and
4. robot-aware `delta_score` is strictly negative.

Cartesian improvement alone is insufficient. V8 does not weaken a v7 safety
gate or promote target count at the expense of robot-aware score.

## Base-Target Diversity

Valid base candidates are Pareto-filtered using the v7 implementation and then
ranked by robot-aware `delta_score`, best first. V8 greedily retains a candidate
only when its residual differs sufficiently from every retained base target.

Distances are RMS values after dividing each joint by a robust joint-wise
residual scale. That scale is `1.4826 * MAD` across available valid base
candidates, with conventional standard deviation and then `1.0` as fallbacks.
V8 records:

- normalized execution-prefix RMS distance over the first 8 steps; and
- normalized full-window RMS distance over all 32 steps.

The defaults are `0.10` prefix RMS, `0.05` full RMS, and at most four base
targets per window. Removed valid candidates are explicitly labeled
`duplicate_or_near_duplicate`, `base_target_limit_reached`, or
`dominated_by_better_candidate`.

## Scale Conditioning

Each retained base residual is multiplied by the default scales:

```text
0.125 0.25 0.50 0.75 1.00
```

Every scaled trajectory is reevaluated from scratch. Scaling can alter FK
error, boundary behavior, derivative costs, maximum steps, and robot-aware
score, so validity is never inherited from the base target.

The scaled residual itself is the training target. `target_scale` is retained
as metadata and appended as the final condition feature. Selection greedily
prioritizes previously unrepresented scales and base-target IDs, then uses
`delta_score`, Cartesian improvement, and stable target ID as deterministic
tie-breakers. The default final limit is eight targets per window.

## Multiprocessing

Target generation supports `--num_workers 8` using
`ProcessPoolExecutor` with the `spawn` multiprocessing context.

Each worker:

- constructs one xMateCR7 robot in its initializer;
- reuses that model for all assigned windows;
- receives only serializable CPU NumPy data and metadata; and
- sets Torch and common BLAS/OpenMP thread counts to one.

Serial and multiprocessing execution call the same generation/evaluation
functions. Work items have stable IDs. Results are collected by ID and restored
to canonical `(path_name, window_start)` order before diversity filtering,
scale selection, and writing. Worker completion order therefore cannot change
targets or output ordering.

## Target Generator CLI

```text
--train_prior PATH
--train_windows PATH
--split_manifest PATH
--output_dir PATH
--robot_urdf PATH
--horizon INT                         default 32
--execution_horizon INT               default 8
--restarts_per_method INT             default 4
--candidate_seed INT                  default 42
--max_base_targets_per_window INT     default 4
--min_prefix_diversity_rms FLOAT      default 0.10
--min_full_diversity_rms FLOAT        default 0.05
--scales FLOAT [FLOAT ...]            default 0.125 0.25 0.50 0.75 1.00
--max_targets_per_window INT          default 8
--num_workers INT                     default 1
--max_paths INT
--max_windows INT
--path_names NAME [NAME ...]
--overwrite
```

The defaults use the same authoritative v7 training inputs:

- `adaptive_mlp_ik_bootstrap_prior/train_prior.npz`;
- `diffusion_v6_strong_prior_residual_windows/train_windows.npz`; and
- `diffusion_v6_strong_prior_residual_windows/split_manifest.csv`.

All paths are under `data/cartesian_expert_dataset_v3/` unless overridden.

### Pilot Procedure

Start with a deterministic path-limited pilot:

```bash
python generate_diffusion_v8_multitarget_scaled_residual_targets.py \
  --output_dir data/cartesian_expert_dataset_v3/diffusion_v8_multitarget_scaled_targets_pilot \
  --max_paths 5 \
  --restarts_per_method 4 \
  --num_workers 8
```

Before full generation, inspect at minimum:

- `per_window_summary.csv` for zero-target windows and target multiplicity;
- `scale_summary.csv` for scale validity and window coverage;
- `candidate_method_summary.csv` for method yield;
- `rejection_reason_summary.csv` for dominant failures; and
- `diversity_summary.csv` for near-duplicate rejection behavior.

Investigate unexpected safety failures, one-scale collapse, low path/window
coverage, or large numbers of nearly identical retained targets before
continuing.

### Full Generation

After the pilot is accepted:

```bash
python generate_diffusion_v8_multitarget_scaled_residual_targets.py \
  --output_dir data/cartesian_expert_dataset_v3/diffusion_v8_multitarget_scaled_residual_targets \
  --restarts_per_method 4 \
  --num_workers 8
```

No arbitrary Gaussian residual is introduced. Restarts remain inside the v7
candidate-generation and optimization conventions.

## Target-Generation Outputs

The generator writes:

- `selected_targets.npz`: retained scaled residuals, conditions, prior and
  target trajectories, IDs, scales, metrics, costs, safety, and diversity;
- `candidate_results.csv`: one row per base candidate and scaled evaluation;
- `selected_target_summary.csv`: one row per retained scaled target;
- `per_window_summary.csv`: candidate funnels, coverage, best improvements,
  scores, and rejection totals;
- `per_path_summary.csv`: path-level yield and coverage;
- `scale_summary.csv`: evaluated/valid/retained counts and scale statistics;
- `candidate_method_summary.csv`: method-level generation and retention yield;
- `rejection_reason_summary.csv`: explicit rejection counts;
- `diversity_summary.csv`: base-candidate diversity decisions; and
- `target_generation_summary.json`: arguments, schemas, conventions, timing,
  seed policy, and distributions.

In diversity fields, `-1.0` means there was no previously retained target from
which to calculate a nearest-neighbor distance.

## Training-Dataset Builder CLI

```text
--targets_npz PATH                   required
--output_dir PATH                    required
--validation_path_count INT          default 20
--split_seed INT                     default 42
--overwrite
```

Build the dataset only after target-generation diagnostics are accepted:

```bash
python build_diffusion_v8_multitarget_scaled_training_dataset.py \
  --targets_npz data/cartesian_expert_dataset_v3/diffusion_v8_multitarget_scaled_residual_targets/selected_targets.npz \
  --output_dir data/cartesian_expert_dataset_v3/diffusion_v8_multitarget_scaled_training_dataset \
  --validation_path_count 20 \
  --split_seed 42
```

## Dataset Schema

The source v7 condition tensor is preserved exactly. `target_scale` is repeated
over the 32-step horizon and appended as the final feature. If the v7 condition
dimension is 38, the v8 condition dimension is 39. The builder reads the v7
feature names from `selected_targets.npz` rather than hardcoding 38.

`target_scale` remains raw in both `condition` and `condition_norm`: its saved
normalization convention is mean `0`, standard deviation `1`. All other
condition features and the `(32, 6)` residual target use statistics calculated
from training rows only. Validation never contributes to normalization.

The builder writes:

- `train_windows.npz` and `validation_windows.npz`;
- `normalization_stats.npz`;
- `dataset_metadata.json`;
- `train_rows.csv` and `validation_rows.csv`;
- `path_split.csv`;
- `targets_per_window_summary.csv`;
- `scale_distribution_by_split.csv`; and
- `dataset_build_summary.json`.

Compatibility keys such as `condition_norm`, `residual_q_norm`, `path_names`,
`window_start_indices`, and `sample_weight` are retained for a future v8
training script. `train_conditional_diffusion_trajectory_v7.py` is not modified.

## Sample Weights

Weights never discard targets.

- `quality_weight` combines normalized Cartesian improvement and normalized
  negative `delta_score`, clips the result to `[0.25, 4.0]`, and normalizes its
  mean to one within each split.
- `window_balance_weight` is `1 / targets_in_window`.
- `combined_sample_weight` is the mean-normalized product of quality and window
  balance weights.

## Integrity Checks

Target generation fails on unsafe, non-improving, non-negative-score,
nonfinite, dimensionally invalid, duplicated, scale-inconsistent, or
reconstruction-inconsistent retained targets. A zero-target window is reported
but does not terminate generation.

Dataset building fails on path leakage, window leakage, inconsistent shapes,
validation-derived normalization, missing scale, nonfinite values, target
shape mismatch, or count mismatch.

## Reading the Diagnostics

`targets_per_window_summary.csv` should show whether v8 actually increases
target multiplicity without concentrating all targets in a few windows. Review
both the mean and the zero-target count by split.

`scale_summary.csv` and `scale_distribution_by_split.csv` should show whether
smaller useful scales survive independent robot-aware validation and whether
train/validation distributions are reasonably aligned. High Cartesian
improvement with poor negative-score yield indicates that smoothness,
continuity, singularity, or another robot-aware term is offsetting the drawing
gain.

No v8 diffusion model should be trained until these pilot diagnostics have
been reviewed and accepted.
# Adaptive Multitarget Generation

The v8 generator supports two policies through `--generation_policy`:

- `exhaustive` is the default and preserves the original behavior. Every
  configured v7 method runs for every canonical window using
  `--restarts_per_method`.
- `adaptive` uses staged candidate generation while retaining the same v7 FK,
  hard-safety, acceptance, score, diversity, and independent scale-validation
  functions.

Adaptive generation runs in this order:

1. `primary_jacobian`: the first `--primary_methods` entry (normally
   `jacobian_dls`) runs for every window with
   `--primary_restarts_per_method`.
2. `primary_sequential_ik`: remaining primary methods (normally
   `sequential_ik`) run only for windows that did not meet both early-stop
   thresholds. With `--no-enable_early_stop`, this stage runs for every
   window.
3. `fallback`: `--fallback_methods` (normally `smooth_perturbation` and
   `spline_cem`) run only where the retained final target count after Stage B
   is strictly below `--fallback_trigger_final_target_count`.

After every stage, new candidates are merged with all candidates already
available for that window. The generator reruns the original deterministic
ranking, Pareto/diversity filtering, evaluation at every requested scale, and
final target retention. Targets are never appended past a cap without this
full reranking.

The default early-stop condition requires both four retained diverse bases
and eight retained final scaled targets. These thresholds are controlled by
`--minimum_base_targets_before_stop` and
`--minimum_final_targets_before_stop`; disabling early stopping only changes
Stage B eligibility and does not weaken any scientific gate.

## Determinism And Workers

One spawn-based `ProcessPoolExecutor` is created for the complete run and is
reused across all stages. Each worker constructs one xMateCR7 model and reuses
it. Adaptive random streams are derived from the global candidate seed, path,
window start, method, and restart, so skipping work for one window cannot
change another window's candidates. Worker results are restored to canonical
window and stable candidate-ID order before ranking. Selected target IDs and
NPZ ordering are therefore independent of worker completion order.

Exhaustive mode retains its original combined-method restart stream for
reproducibility. Existing candidate and target ID formulas are preserved.

## Resume

Adaptive mode atomically checkpoints after each completed stage. The public
`adaptive_generation_state.json` points to a stage-versioned pickle containing
the canonical intermediate candidates and selections. The JSON file is
replaced only after the intermediate file is durable, so an interrupted write
leaves the previous completed stage resumable.

Resume with the same scientific arguments and add `--resume`. Completed
stages are not regenerated. `--resume` is adaptive-only and cannot be combined
with `--overwrite`.

## Candidate Storage

`--candidate_results_mode` controls only `candidate_results.csv`; it never
changes `selected_targets.npz`, selected target IDs, or aggregate summaries.

- `all` preserves all detailed candidate rows.
- `retained_and_summary` stores retained bases, retained scaled targets, valid
  bases rejected by diversity/retention, and failed candidates from windows
  that ended with no target. All candidates still contribute to aggregate
  summaries.
- `none` omits `candidate_results.csv` while retaining selected outputs and
  all aggregate diagnostics.

For large adaptive runs, `retained_and_summary` is recommended.

## Diagnostics

`scale_summary.csv` uses every independently evaluated scaled row as its
denominator. For each scale it reports evaluated, hard-safe,
Cartesian-improving, negative-score, independently valid, and retained counts,
plus rates divided by `evaluated_count`. Runtime integrity checks require the
per-scale evaluated, independently valid, and retained totals to match the
global totals.

`candidate_method_window_contribution.csv` reports method yield, attempted and
covered windows, unique window contributions, and cumulative/per-candidate
generation and FK-scoring time. `candidate_method_summary.csv` also includes
the timing fields.

`adaptive_stage_summary.csv` records stage entry, attempts, candidates,
valid/retained counts, completion and zero-target counts, and cumulative wall
time. The same records are embedded in `target_generation_summary.json`.

## Recommended Comparison Pilot

Run matched 10-path jobs in separate output directories:

```bash
python generate_diffusion_v8_multitarget_scaled_residual_targets.py \
  --generation_policy exhaustive \
  --output_dir data/cartesian_expert_dataset_v3/diffusion_v8_exhaustive_10paths \
  --max_paths 10 \
  --num_workers 8 \
  --restarts_per_method 4 \
  --max_base_targets_per_window 4 \
  --max_targets_per_window 8 \
  --scales 0.125 0.25 0.50 0.75 1.00 \
  --candidate_seed 42 \
  --overwrite

python generate_diffusion_v8_multitarget_scaled_residual_targets.py \
  --generation_policy adaptive \
  --output_dir data/cartesian_expert_dataset_v3/diffusion_v8_adaptive_10paths \
  --max_paths 10 \
  --num_workers 8 \
  --primary_restarts_per_method 2 \
  --fallback_restarts_per_method 4 \
  --minimum_base_targets_before_stop 4 \
  --minimum_final_targets_before_stop 8 \
  --fallback_trigger_final_target_count 4 \
  --max_base_targets_per_window 4 \
  --max_targets_per_window 8 \
  --scales 0.125 0.25 0.50 0.75 1.00 \
  --candidate_results_mode retained_and_summary \
  --candidate_seed 42 \
  --overwrite
```

Compare coverage, retained-target counts, scale validity, diversity, method
contributions, and wall time. Adaptive generation is a scheduling policy, so
scientific acceptance criteria must match the exhaustive run.

## Recommended Full Run

```bash
python generate_diffusion_v8_multitarget_scaled_residual_targets.py \
  --generation_policy adaptive \
  --output_dir data/cartesian_expert_dataset_v3/diffusion_v8_multitarget_scaled_residual_targets \
  --num_workers 8 \
  --primary_methods jacobian_dls sequential_ik \
  --fallback_methods smooth_perturbation spline_cem \
  --primary_restarts_per_method 2 \
  --fallback_restarts_per_method 4 \
  --minimum_base_targets_before_stop 4 \
  --minimum_final_targets_before_stop 8 \
  --fallback_trigger_final_target_count 4 \
  --scales 0.125 0.25 0.50 0.75 1.00 \
  --max_base_targets_per_window 4 \
  --max_targets_per_window 8 \
  --candidate_results_mode retained_and_summary \
  --candidate_seed 42 \
  --overwrite
```

To resume that run, repeat the same command, remove `--overwrite`, and add
`--resume`.
