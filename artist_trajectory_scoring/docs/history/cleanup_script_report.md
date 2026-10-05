# ArtistDiffusion Script Cleanup Report

Date: 2026-07-31

## Active Entry Points Retained

- v8.1 anchored recursive rollout: `evaluate_diffusion_v8_1_anchored_recursive_jerk_guard.py`
- v8.1 multiseed summary: `summarize_diffusion_v8_1_anchored_multiseed.py`
- v8.1 path-disjoint confirmation: `evaluate_diffusion_v8_1_path_disjoint_test_prior.py`, `summarize_diffusion_v8_1_path_disjoint_confirmation.py`
- v8.1 deployment: `generate_joint_trajectory_diffusion_v8_1.py`, `validate_diffusion_v8_1_deployment_output.py`
- Frozen v8 baseline and helpers: `evaluate_diffusion_v8_anchored_recursive_rollout.py`, `summarize_diffusion_v8_anchored_multiseed.py`, `evaluate_diffusion_v8_teacher_forced_all_windows.py`
- v8 data/model reproduction: `generate_diffusion_v8_multitarget_scaled_residual_targets.py`, `build_diffusion_v8_multitarget_scaled_training_dataset.py`, `train_conditional_diffusion_trajectory_v8.py`
- Shared scientific helpers: `generate_diffusion_v7_cost_improving_residual_targets.py`, `evaluate_diffusion_v7_teacher_forced_validation.py`, `evaluate_diffusion_v6_teacher_forced_validation.py`, `train_conditional_diffusion_trajectory_v6_strong_prior_residual_unet.py`, `generate_ik_seed_path.py`

## Baseline Scripts Retained

- IK baseline and expert generation: `batch_generate_ik_experts.py`, `generate_ik_seed_path.py`, `orientation_aware_adaptive_ik.py`
- Adaptive MLP+IK baseline: `generate_adaptive_mlp_ik_bootstrap_prior.py`, `refine_mlp_predictions_with_ik.py`, `adaptive_refine_mlp_predictions_with_ik.py`
- MLP baseline: `train_path_conditioned_mlp.py`, `predict_path_conditioned_mlp.py`, `generate_mlp_v3_train_test_predictions.py`, `evaluate_path_conditioned_mlp.py`
- v4/v5/v6 comparison scripts were retained where they support benchmark lineage, comparison artifacts, or imported diagnostics.
- Presentation/benchmark scripts retained: `benchmark_ik_mlp_pipeline_smoothness.py`, `compare_ik_mlp_pipeline_jerk_over_time.py`, `make_final_result_plots.py`, `make_results_comparison_table.py`, `plot_cartesian_tracking_error_over_time.py`

## Scripts Deleted

1. `build_diffusion_trajectory_dataset.py`
2. `build_diffusion_trajectory_dataset_v2.py`
3. `train_conditional_diffusion_trajectory.py`
4. `train_conditional_diffusion_trajectory_v2.py`
5. `train_conditional_diffusion_trajectory_v3_x0.py`
6. `sample_conditional_diffusion_trajectory.py`
7. `sample_conditional_diffusion_trajectory_v2.py`
8. `sample_conditional_diffusion_trajectory_v2_ddim.py`
9. `sample_conditional_diffusion_trajectory_v3_x0.py`
10. `sample_ranked_diffusion_candidates_v1.py`
11. `summarize_ranked_diffusion_candidates.py`
12. `rerank_diffusion_candidates.py`
13. `diagnose_diffusion_reconstruction_v2.py`
14. `debug_diffusion_v4_dataset_stats.py`
15. `sanity_check_diffusion_v4_reconstruction_math.py`
16. `monitor_v7_w9.py`
17. `run_v7_fast_shard.sh`
18. `run_v7_fast_9shard.sh`

Approximate source reduction from tracked diff: 18 deleted files, 5,437 deleted lines, 21 inserted lines in retained source/tests.

## Other Source Changes

- `audit_and_compare_bootstrap_priors.py`: replaced historical line references to deleted v1/v2/v3 scripts with Git-history provenance text.
- `benchmark_ik_mlp_pipeline_smoothness.py`: removed the dynamic dependency on deleted `rerank_diffusion_candidates.py`; retained its historical max Cartesian-error gate as `LEGACY_RERANKED_DIFFUSION_ACCEPTANCE_MAX_ERROR_M = 0.030`.
- `test_benchmark_ik_mlp_pipeline_smoothness.py`: updated expected threshold-source text and invalid-threshold test.
- `cleanup_script_audit.md`: added required pre-deletion audit and classification.

## Validation Commands Executed

```bash
python -m compileall -q .
```

Result: passed.

```bash
python - <<'PY'
import importlib
mods = [
    'evaluate_diffusion_v8_1_anchored_recursive_jerk_guard',
    'summarize_diffusion_v8_1_anchored_multiseed',
    'evaluate_diffusion_v8_1_path_disjoint_test_prior',
    'summarize_diffusion_v8_1_path_disjoint_confirmation',
    'generate_joint_trajectory_diffusion_v8_1',
    'validate_diffusion_v8_1_deployment_output',
    'train_conditional_diffusion_trajectory_v8',
    'generate_diffusion_v8_multitarget_scaled_residual_targets',
    'build_diffusion_v8_multitarget_scaled_training_dataset',
    'benchmark_ik_mlp_pipeline_smoothness',
]
for m in mods:
    importlib.import_module(m)
    print('OK', m)
PY
```

Result: passed for all listed active modules.

```bash
python evaluate_diffusion_v8_1_anchored_recursive_jerk_guard.py --help
python generate_joint_trajectory_diffusion_v8_1.py --help
python validate_diffusion_v8_1_deployment_output.py --help
python train_conditional_diffusion_trajectory_v8.py --help
python generate_diffusion_v8_multitarget_scaled_residual_targets.py --help
python build_diffusion_v8_multitarget_scaled_training_dataset.py --help
```

Result: passed.

```bash
python -m unittest test_benchmark_ik_mlp_pipeline_smoothness.py
```

Result: passed, 62 tests.

```bash
for f in build_diffusion_trajectory_dataset.py build_diffusion_trajectory_dataset_v2.py train_conditional_diffusion_trajectory.py train_conditional_diffusion_trajectory_v2.py train_conditional_diffusion_trajectory_v3_x0.py sample_conditional_diffusion_trajectory.py sample_conditional_diffusion_trajectory_v2.py sample_conditional_diffusion_trajectory_v2_ddim.py sample_conditional_diffusion_trajectory_v3_x0.py sample_ranked_diffusion_candidates_v1.py summarize_ranked_diffusion_candidates.py rerank_diffusion_candidates.py diagnose_diffusion_reconstruction_v2.py debug_diffusion_v4_dataset_stats.py sanity_check_diffusion_v4_reconstruction_math.py monitor_v7_w9.py run_v7_fast_shard.sh run_v7_fast_9shard.sh; do
  rg -n --glob '!cleanup_script_audit.md' --glob '!data/**' --glob '!results/**' --glob '!models/**' "$f" . || true
done
```

Result: no dangling references outside `cleanup_script_audit.md`.

## Checks Not Run

- No training, full evaluation, smoke rollout, deployment generation, CUDA benchmarking, ROS, or physical-robot execution was launched.
- Existing broader unit-test suites were not all run; only the touched smoothness benchmark unit file was run.

## Unresolved Files Requiring Manual Review

None for this pass. A second pass could review retained v4/v5/v6 diagnostic scripts more aggressively after confirming which historical comparison figures must remain reproducible.

## Remaining References To Old Versions

- `cleanup_script_audit.md` intentionally lists deleted filenames as the cleanup record.
- Data/result directories may still contain historical metadata references to old scripts; those artifacts were not modified because the cleanup was limited to source scripts and clearly obsolete launchers.

## Exact `git diff --stat`

```text
 .../audit_and_compare_bootstrap_priors.py          |  26 +-
 .../benchmark_ik_mlp_pipeline_smoothness.py        |  14 +-
 .../build_diffusion_trajectory_dataset.py          | 267 ----------
 .../build_diffusion_trajectory_dataset_v2.py       | 192 -------
 .../debug_diffusion_v4_dataset_stats.py            | 165 ------
 .../diagnose_diffusion_reconstruction_v2.py        | 441 ----------------
 artist_trajectory_scoring/monitor_v7_w9.py         |  46 --
 .../rerank_diffusion_candidates.py                 | 255 ---------
 artist_trajectory_scoring/run_v7_fast_9shard.sh    |  64 ---
 artist_trajectory_scoring/run_v7_fast_shard.sh     |  64 ---
 .../sample_conditional_diffusion_trajectory.py     | 377 -------------
 .../sample_conditional_diffusion_trajectory_v2.py  | 314 -----------
 ...ple_conditional_diffusion_trajectory_v2_ddim.py | 557 --------------------
 ...ample_conditional_diffusion_trajectory_v3_x0.py | 521 ------------------
 .../sample_ranked_diffusion_candidates_v1.py       | 529 -------------------
 ...anity_check_diffusion_v4_reconstruction_math.py | 585 ---------------------
 .../summarize_ranked_diffusion_candidates.py       | 107 ----
 .../test_benchmark_ik_mlp_pipeline_smoothness.py   |   9 +-
 .../train_conditional_diffusion_trajectory.py      | 375 -------------
 .../train_conditional_diffusion_trajectory_v2.py   | 264 ----------
 ...train_conditional_diffusion_trajectory_v3_x0.py | 286 ----------
 21 files changed, 21 insertions(+), 5437 deletions(-)
```

Note: untracked report files (`cleanup_script_audit.md`, `cleanup_script_report.md`) are not included in `git diff --stat` until added to Git.

## Second-Pass Recommendation

For a second cleanup pass, inspect retained v4/v5/v6 diagnostics against the exact final presentation figure/table dependencies. Several are likely removable, but this pass kept them because they support comparison lineage and are entangled through imports.
