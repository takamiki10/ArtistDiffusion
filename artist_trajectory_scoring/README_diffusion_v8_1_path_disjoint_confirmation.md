# Diffusion v8.1 Path-Disjoint Confirmation

This is the frozen v8.1 confirmation evaluation on complete trajectories from:

```text
data/cartesian_expert_dataset_v3/adaptive_mlp_ik_bootstrap_prior/test_prior.npz
```

The frozen 30-path manifest is:

```text
results/diffusion_v8_1_path_disjoint_confirmation_test30_seeds53_57/metadata/path_disjoint_test_paths_30.csv
```

The path-disjoint evaluator and summarizer are separate from the frozen v8 and
v8.1 development evaluators.

## Why This Is Path-Disjoint

The physical rollout population is loaded from the adaptive MLP+IK test-prior
archive, not from the train-side v8 validation population used during v8/v8.1
development. The archive supplies only physical evaluation records:

- `desired_paths`
- `prior_q`
- `prior_ee`
- archive/source metadata

The trained model, checkpoint, normalization, schedule, and model metadata still
come from the original v8 training artifacts:

```text
data/cartesian_expert_dataset_v3/diffusion_v8_multitarget_scaled_training_dataset_100paths
models/diffusion_v8_multitarget_scaled_residual_unet_100paths_epsilon_only_seed42
raw_last_epoch187
```

`test_prior.npz` is never used as a training, normalization, or model metadata
artifact.

## Archive To PhysicalPathRecord Mapping

Each selected archive row is converted to the frozen `v8.PhysicalPathRecord`:

```text
desired_path   = desired_paths[index]
strong_prior_q = prior_q[index]
prior_ee       = prior_ee[index]
population     = ordinary
```

The archive `path_name` remains source metadata. The evaluator identity is
collision-safe:

```text
source_path_name = path_0001
evaluator_path_id = test__path_0001
```

The collision-safe evaluator ID is used when calling frozen
`run_rollout_v8_1(...)`, so candidate seed identities cannot collide with
train-side records that use plain `path_XXXX` names. Seeds `53-57` therefore
define a new stochastic confirmation population and are not expected to match
earlier candidate seeds.

Every metrics, decision, candidate, and trajectory NPZ output preserves:

- `sampling_seed`
- `source_split`
- `source_path_name`
- `source_archive`
- `source_method`
- `source_checkpoint`
- `manifest_selection_rank`
- `evaluator_path_id`

## Manifest Integrity

The full formal run requires exactly 30 manifest rows with:

- `split = test`
- 30 unique `path_name` values
- unique `selection_rank`
- every path found exactly once in `test_prior.npz`
- every path has `generation_success=True`
- `generation_status` indicates success

For every selected path the evaluator validates:

- `desired_path.shape == (100, 3)`
- `strong_prior_q.shape == (100, 6)`
- `prior_ee.shape == (100, 3)`
- all arrays are finite
- stored `prior_ee` matches authoritative FK with `rtol=1e-5`, `atol=2e-5`
- frozen fallback hard-safety requirements pass before diffusion sampling

The evaluator also configures Matplotlib with the headless `Agg` backend before
importing project plotting modules. This avoids Tkinter/main-thread crashes
during repeated multiprocessing evaluations without modifying frozen plotting
functions.

No path may be removed, replaced, or reordered after results are inspected.

## Frozen v8.1 Behavior Reused

The evaluator imports and calls:

```python
evaluate_diffusion_v8_1_anchored_recursive_jerk_guard.run_rollout_v8_1(...)
```

It does not copy or reimplement:

- candidate generation
- DDIM sampling
- condition construction
- anchored-prior construction
- history-aware jerk guard
- hard-safety gates
- candidate ordering
- fallback
- full-path metric calculation

The jerk guard remains enabled. `K=8` uses the same eight nested candidates.

## Diagnostic Subsets

`--source_path_names` and `--max_paths` are smoke-test subset filters. They
preserve manifest order, but any subset run is marked:

```text
diagnostic_subset_run = 1
path_disjoint_confirmation_population = 0
```

Diagnostic subsets are not formal path-disjoint confirmation runs and must not
receive a formal PASS classification.

## Formal Confirmation Summary

The summarizer defaults to:

```text
input_root = results/diffusion_v8_1_path_disjoint_confirmation_test30_seeds53_57
seeds = 53 54 55 56 57
decision K = 8
expected paths per seed = 30
```

The formal key is:

```text
(sampling_seed, source_split, source_path_name, k)
```

Before classification, the summarizer requires:

- five seeds exactly `53,54,55,56,57`
- 30 unique source test paths per seed
- 150 unique K=8 path-seed keys
- identical 30-path `source_path_name` set across all seeds
- no duplicate keys
- all records have `source_split=test`
- all records have `path_disjoint_confirmation_population=1`
- all records have `diagnostic_subset_run=0`
- every CSV row stores `sampling_seed`
- each row's stored `sampling_seed` matches its `seed_XX` directory
- each seed's `anchored_rollout_summary.json` stores the same seed as its directory
- all five seed summaries use identical checkpoint state, checkpoint hash,
  training dataset, model directory, test-prior archive, manifest, source path
  set, and evaluator path ID set

If any key is missing, duplicated, or unexpected, the summarizer raises a
descriptive error and does not issue an engineering classification.

## Confirmation Criteria

The summarizer classifies:

```text
V8_1_PATH_DISJOINT_CONFIRMATION_PASS
```

only when all conditions pass:

- complete five-seed, 30-path population
- all final trajectories safe
- maximum actual internal joint step `<= 0.20` rad
- mean internal full-path robot-aware delta score `< 0`
- mean Cartesian mean-error delta `<= 0`
- 95th percentile internal full-path score `<= 0`
- worst path-seed internal full-path score `<= 0`

Otherwise it classifies:

```text
V8_1_PATH_DISJOINT_CONFIRMATION_HOLD
```

Correction-growth slope is reported but is not a pass/fail rule.

All pass/fail metrics must be finite. If any gated metric is NaN or infinite,
the summarizer raises an error instead of issuing PASS or HOLD.

NaN jerk-rejection rates are handled only for reported, non-gated aggregates:
`mean_jerk_rejection_rate_among_v8_selectable_candidates` excludes nonfinite
per-path rates from its denominator. It reports null/NaN only when no finite
values exist.

Reported but not gated:

- fraction of path-seed runs with negative internal score
- mean internal jerk contribution
- mean accepted rollout-step rate
- mean fallback rate
- mean jerk rejection rate among v8-selectable candidates
- per-seed metrics
- per-physical-path metrics across seeds

## Outputs

The evaluator preserves the core output filenames:

- `anchored_rollout_decisions.csv`
- `anchored_candidate_results.csv`
- `anchored_full_path_metrics.csv`
- `anchored_ordinary_aggregate.csv`
- `anchored_combined_diagnostic_aggregate.csv`
- `anchored_rollout_summary.json`
- `anchored_rollout_report.txt`

The summarizer writes:

- `path_disjoint_per_seed.csv`
- `path_disjoint_per_path.csv`
- `path_disjoint_aggregate.csv`
- `path_disjoint_confirmation_summary.json`
- `path_disjoint_confirmation_report.txt`

and plots for ordinary `K=8`:

- per-seed mean internal score
- per-seed mean Cartesian delta
- per-seed acceptance and fallback rates
- per-path mean and maximum internal score
- per-path negative-score seed rate
- internal-score distribution
- Cartesian-delta distribution
- jerk-contribution distribution
