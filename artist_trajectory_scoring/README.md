# ArtistDiffusion: start here

This directory contains the **v8 training → v8.1 trajectory generation** pipeline,
its shared implementation, and the current paper/physical-validation tools.
Run commands from this directory: many defaults use relative `data/`, `models/`,
`results/`, and `robot_model/` paths.

## Where things live

| Location | Contents |
| --- | --- |
| Root Python scripts | Current pipeline, required helpers, baseline reproduction, benchmarks, and regression tests |
| [old scripts](<old scripts/README.md>) | Superseded experiments, diagnostics, historical plots, and a backup; preserved source, not current entry points |
| [docs/history](docs/history/README.md) | Earlier cleanup reports and the historical v4 guide |
| `data/` | Datasets, normalization metadata, prior archives, and some historical checkpoints |
| `models/` | Trained diffusion models |
| `results/`, `logs/` | Saved outputs and experiment logs; result-specific plotting scripts stay with their results |
| `robot_model/` | Robot description and assets |
| `k_candidate_ablation/` (local supplement) | Candidate-count ablation and reproduction instructions |
| `physical_validation/` (local supplement) | Physical-validation protocols, preparation scripts, and handoff artifacts |
| `deployment_bridge/` | Dataset-specific deployment conversion |

## Main workflow

| Stage | Entry points | Guide |
| --- | --- | --- |
| Prepare expert/MLP data | `generate_cartesian_path_variations.py`, `batch_generate_ik_experts.py`, `build_cartesian_expert_npz.py`, `validate_expert_dataset.py` | [Expert dataset workflow](README_cartesian_expert_dataset_v2.md); the guide's v2 paths are historical examples |
| Train/export the MLP baseline | `train_path_conditioned_mlp.py`, `predict_path_conditioned_mlp.py`, `generate_mlp_v3_train_test_predictions.py`, `evaluate_path_conditioned_mlp.py` | Inspect each script's `--help` for dataset paths |
| Build the strong prior and training windows | `generate_adaptive_mlp_ik_bootstrap_prior.py`, `build_diffusion_v6_strong_prior_residual_window_dataset.py` | [Target generation](README_diffusion_v8_multitarget_scaled.md) |
| Generate v8 targets and dataset | `generate_diffusion_v8_multitarget_scaled_residual_targets.py`, `build_diffusion_v8_multitarget_scaled_training_dataset.py` | [Target generation](README_diffusion_v8_multitarget_scaled.md) |
| Train v8 | `train_conditional_diffusion_trajectory_v8.py` | [Training guide](README_diffusion_v8_training.md) |
| Evaluate v8.1 | `evaluate_diffusion_v8_1_anchored_recursive_jerk_guard.py`, `evaluate_diffusion_v8_1_path_disjoint_test_prior.py` | [Jerk guard](README_diffusion_v8_1_anchored_recursive_jerk_guard.md), [path-disjoint confirmation](README_diffusion_v8_1_path_disjoint_confirmation.md) |
| Summarize v8.1 evaluation | `summarize_diffusion_v8_1_anchored_multiseed.py`, `summarize_diffusion_v8_1_path_disjoint_confirmation.py` | Same evaluation guides |
| Convert Cartesian input | `generate_deployment_input_from_cartesian_csv.py`, `generate_branch_continuous_deployment_input_from_cartesian_csv.py` | Run `--help` for pose constraints and input options |
| Generate and validate a trajectory | `generate_joint_trajectory_diffusion_v8_1.py`, `validate_diffusion_v8_1_deployment_output.py` | [Deployment guide](README_diffusion_v8_1_deployment_generator.md) |
| Assemble and validate multiple strokes | `build_multistroke_full_pose_execution.py`, `validate_multistroke_full_pose_execution.py` | Inspect each script's `--help`; additional protocols are in the local physical-validation supplement |

Deployment uses the frozen `raw_last_epoch187` checkpoint in
`models/diffusion_v8_multitarget_scaled_residual_unet_100paths_epsilon_only_seed42/`
and metadata in
`data/cartesian_expert_dataset_v3/diffusion_v8_multitarget_scaled_training_dataset_100paths/`.
Keep the checkpoint and its matching normalization/metadata together. Training
a new model is a separate workflow from reproducing this frozen deployment.

For an already prepared input NPZ, the basic commands are:

```bash
python generate_joint_trajectory_diffusion_v8_1.py \
  --input_npz data/my_deployment_input.npz \
  --output_dir results/my_deployment_trajectory \
  --sampling_seed 53 --device cuda

python validate_diffusion_v8_1_deployment_output.py \
  --output_dir results/my_deployment_trajectory --require_accepted
```

The input above is a placeholder; prepare it with a Cartesian-input converter.
The deployment guide specifies the required arrays and acceptance checks.

## Current analysis and reproduction

| Purpose | Scripts |
| --- | --- |
| Smoothness/runtime and jerk comparisons | `benchmark_ik_mlp_pipeline_smoothness.py`, `compare_ik_mlp_pipeline_jerk_over_time.py` |
| Cartesian tracking | `plot_cartesian_tracking_error_over_time.py` |
| Prior contribution and lineage | `analyze_prior_vs_diffusion_contribution_v8_1.py`, `audit_and_compare_bootstrap_priors.py` |
| Frozen v8 comparison | `evaluate_diffusion_v8_teacher_forced_all_windows.py`, `evaluate_diffusion_v8_anchored_recursive_rollout.py`, `run_diffusion_v8_anchored_multiseed.py`, `summarize_diffusion_v8_anchored_multiseed.py` |
| Focused v8 evaluation | `run_diffusion_v8_focused_multiseed.py`, `summarize_diffusion_v8_focused_multiseed.py`; see the [teacher-forced guide](README_diffusion_v8_teacher_forced_evaluation.md) |

## Additional local research work

The working directory also contains research additions that are **not included
in this cleanup commit**. They remain available locally, but a fresh GitHub
clone will not include them:

- `audit_reference_qp_current_paper.py`, `evaluate_reference_qp_current_paper_audit.py`
- `make_figure4_qp_vs_proposed.py`
- `host_generate_physical_a_10.py`, `host_parallel_physical_a_10.py`
- `k_candidate_ablation/` and `physical_validation/`
- Untracked experiment outputs under `results/`

Those supplements need a separate source/artifact handoff if required. The
core training, generation, validation, and benchmark entry points listed above
are included in Git.

## Why older filenames remain in the root

Version numbers do not determine whether a script is obsolete. These files are
still required by the current pipeline or its reproduction workflow:

| Files | Reason retained |
| --- | --- |
| `train_conditional_diffusion_trajectory_v5_residual_unet.py`, `conditional_unet1d_artist.py` | Model implementation; v6/v8 dynamically locate the v5 `LocalResidualConditionalUNet1D` class |
| `train_conditional_diffusion_trajectory_v6_strong_prior_residual_unet.py` | Shared training and sampling implementation used by v8 |
| `evaluate_diffusion_v6_teacher_forced_validation.py`, `evaluate_diffusion_v7_teacher_forced_validation.py` | Shared inference, FK, and evaluation helpers |
| `generate_diffusion_v7_cost_improving_residual_targets.py`, `build_diffusion_v7_cost_improving_training_dataset.py` | Reused target-generation/scoring and dataset helpers |
| `build_diffusion_v5b_residual_window_dataset_fk_condition.py` | Imported by the v6 evaluator |
| `generate_ik_seed_path.py`, `orientation_aware_adaptive_ik.py` | Authoritative robot FK/IK and pose handling |
| `refine_mlp_predictions_with_ik.py`, `adaptive_refine_mlp_predictions_with_ik.py` | Prior/baseline refinement and provenance |
| `generate_mlp_v3_test_predictions.py` | Imported by the train/test MLP exporter |
| `score_trajectory.py` | Called by MLP evaluation |
| `test_*.py` | Regression checks for current comparisons and deployment export |

Keep these filenames and locations stable unless their callers, dynamic imports,
checkpoint compatibility, and source-hash checks are also reviewed.

## Environment and checks

The existing dependency list is [../requirements.txt](../requirements.txt).
The scripts also use SciPy; use an environment with NumPy, pandas, matplotlib,
PyTorch, SciPy, and yourdfpy available. GPU runs require a compatible PyTorch/CUDA
installation. Use `python SCRIPT.py --help` to inspect an entry point.

Run the current root regression suites with:

```bash
python -m unittest \
  test_benchmark_ik_mlp_pipeline_smoothness \
  test_compare_ik_mlp_pipeline_jerk_over_time \
  test_legacy_smartjoint_export \
  test_validate_multistroke_legacy_csv
```

## Organization performed on 2026-10-05

37 source/backup files were moved into `old scripts/`, with original bytes
recorded in its [manifest](<old scripts/manifest.json>). Three historical
documents moved to `docs/history/`. Classification follows the documented
v8/v8.1 workflow, imports (including dynamic model loading), script calls, and
recent paper/physical-validation work. The archive is reversible if an older
experiment is needed again.

Datasets, checkpoints, results, logs, and nested experiment folders were left
in place. The 18 pending deletions from the earlier July cleanup were already
present before this organization and were not restored or newly deleted here.
Earlier edits were preserved; the lineage audit's references to moved source
files now include `old scripts/`.

Validation after organization: all 108 tests in the four root regression suites
passed; current training/deployment imports and `--help` commands passed; the
dynamic model loader still selects the same v5 model class. Syntax, archive
checksums, active import dependencies, nested workflow filename references,
and the new documentation links were checked. No full training, trajectory
generation, or hardware execution was performed for this file organization.
