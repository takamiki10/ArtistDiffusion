# ArtistDiffusion Script Cleanup Audit

Date: 2026-07-31

Scope: `/mnt/ssd/artistDiffusion/ArtistDiffusion/artist_trajectory_scoring`

## Current Pipeline Dependency Map

- Dataset preparation: `generate_cartesian_path_variations.py`, `batch_generate_ik_experts.py`, `build_cartesian_expert_npz.py`, `generate_adaptive_mlp_ik_bootstrap_prior.py`, `build_diffusion_v6_strong_prior_residual_window_dataset.py`, `generate_diffusion_v8_multitarget_scaled_residual_targets.py`, `build_diffusion_v8_multitarget_scaled_training_dataset.py`
- Residual-target generation: `generate_diffusion_v8_multitarget_scaled_residual_targets.py`, importing `generate_diffusion_v7_cost_improving_residual_targets.py`
- Model training/reproduction: `train_conditional_diffusion_trajectory_v8.py`, importing `train_conditional_diffusion_trajectory_v6_strong_prior_residual_unet.py`
- Teacher-forced evaluation: `evaluate_diffusion_v8_teacher_forced_all_windows.py`, importing v7/v6 validation and v8/v7 training/generation helpers
- Anchored recursive generation/evaluation: `evaluate_diffusion_v8_1_anchored_recursive_jerk_guard.py`, importing frozen v8 rollout, v8 teacher-forced helpers, v7 evaluator, and v7 target generator
- Path-disjoint confirmation: `evaluate_diffusion_v8_1_path_disjoint_test_prior.py`, `summarize_diffusion_v8_1_path_disjoint_confirmation.py`
- Deployment trajectory generation: `generate_joint_trajectory_diffusion_v8_1.py`, `validate_diffusion_v8_1_deployment_output.py`
- FK and Cartesian-error evaluation: `generate_ik_seed_path.py`, `score_trajectory.py`, `evaluate_prior_refinement_fk_robot_costs.py`, v7/v8 evaluator helpers
- IK baseline generation: `generate_ik_seed_path.py`, `batch_generate_ik_experts.py`, `orientation_aware_adaptive_ik.py`, `generate_adaptive_mlp_ik_bootstrap_prior.py`, `refine_mlp_predictions_with_ik.py`, `adaptive_refine_mlp_predictions_with_ik.py`
- MLP baseline generation: `train_path_conditioned_mlp.py`, `predict_path_conditioned_mlp.py`, `generate_mlp_v3_train_test_predictions.py`, `evaluate_path_conditioned_mlp.py`
- Smoothness and jerk comparison: `benchmark_ik_mlp_pipeline_smoothness.py`, `compare_ik_mlp_pipeline_jerk_over_time.py`, tests with matching names
- Runtime benchmarking: `benchmark_ik_mlp_pipeline_smoothness.py`, `evaluate_diffusion_v8_teacher_forced_all_windows.py`, v8/v8.1 rollout timing outputs
- Physical-robot CSV/export: `generate_deployment_input_from_cartesian_csv.py`, `generate_branch_continuous_deployment_input_from_cartesian_csv.py`, `build_multistroke_full_pose_execution.py`, `validate_multistroke_full_pose_execution.py`, `deployment_bridge/touch_20260122/merge_force_altitude_v2.py`
- Presentation figures/tables: `make_final_result_plots.py`, `make_results_comparison_table.py`, `make_unified_artistdiffusion_results_table.py`, `plot_cartesian_tracking_error_over_time.py`, `plot_tracking_error.py`, `plot_path_comparison.py`

## Import And Reference Findings

- Current v8.1 deployment imports `evaluate_diffusion_v8_1_anchored_recursive_jerk_guard.py`.
- v8.1 rollout imports frozen v8 rollout and v8 teacher-forced helpers.
- Frozen v8 teacher-forced helpers import v7 and v6 evaluator/training utilities.
- v8 target generation imports v7 target generation.
- v7/v8/v8.1 active code imports `generate_ik_seed_path.py`.
- The legacy v1/v2/v3 diffusion scripts are not imported by active v8/v8.1/deployment code. Their historical line-reference strings in `audit_and_compare_bootstrap_priors.py` were replaced with Git-history provenance text before deletion.

## Proposed Deletions

| File | Category | Reason | Replacement | Imports? | Documentation/current command references |
|---|---|---|---|---|---|
| `build_diffusion_trajectory_dataset.py` | DELETE_SUPERSEDED | v1 full-trajectory dataset builder, superseded by v6/v8 strong-prior residual and multitarget builders | `build_diffusion_v8_multitarget_scaled_training_dataset.py` | no active importers | no current docs/shell refs found |
| `build_diffusion_trajectory_dataset_v2.py` | DELETE_SUPERSEDED | v2 full-trajectory dataset builder, superseded by v6/v8 builders | `build_diffusion_v8_multitarget_scaled_training_dataset.py` | no active importers | no current docs/shell refs found |
| `train_conditional_diffusion_trajectory.py` | DELETE_SUPERSEDED | v1 diffusion trainer, superseded by v8 trainer; historical audit text updated | `train_conditional_diffusion_trajectory_v8.py` | only legacy v1 sample/import chain | historical string refs removed from active audit |
| `train_conditional_diffusion_trajectory_v2.py` | DELETE_SUPERSEDED | v2 diffusion trainer, superseded by v8 trainer; historical audit text updated | `train_conditional_diffusion_trajectory_v8.py` | only legacy v2 sample/debug chain | historical string refs removed from active audit |
| `train_conditional_diffusion_trajectory_v3_x0.py` | DELETE_SUPERSEDED | v3 x0 trainer, superseded by v8 trainer; historical audit text updated | `train_conditional_diffusion_trajectory_v8.py` | only legacy v3 sampler | historical string refs removed from active audit |
| `sample_conditional_diffusion_trajectory.py` | DELETE_SUPERSEDED | v1 sampler, superseded by v8.1 anchored recursive generation/deployment | `evaluate_diffusion_v8_1_anchored_recursive_jerk_guard.py`, `generate_joint_trajectory_diffusion_v8_1.py` | no active importers | no current docs/shell refs found |
| `sample_conditional_diffusion_trajectory_v2.py` | DELETE_SUPERSEDED | v2 sampler, superseded by v8.1 anchored recursive generation/deployment | same as above | no active importers | no current docs/shell refs found |
| `sample_conditional_diffusion_trajectory_v2_ddim.py` | DELETE_SUPERSEDED | v2 DDIM sampler, superseded by v8.1 DDIM through validated v8 helper | `evaluate_diffusion_v8_1_anchored_recursive_jerk_guard.py` | no active importers | no current docs/shell refs found |
| `sample_conditional_diffusion_trajectory_v3_x0.py` | DELETE_SUPERSEDED | v3 sampler, superseded by v8.1 anchored recursive generation/deployment | same as above | no active importers | no current docs/shell refs found |
| `sample_ranked_diffusion_candidates_v1.py` | DELETE_SUPERSEDED | v1 ranked candidate sampler, superseded by v8/v8.1 nested-K candidate selection; historical audit text updated | v8/v8.1 evaluators | no active importers | historical string refs removed from active audit |
| `summarize_ranked_diffusion_candidates.py` | DELETE_SUPERSEDED | companion v1 ranked-candidate summarizer, superseded by v8/v8.1 summarizers | `summarize_diffusion_v8_1_anchored_multiseed.py`, path-disjoint summarizer | no active importers | no current docs/shell refs found |
| `rerank_diffusion_candidates.py` | DELETE_SUPERSEDED | legacy reranking helper for old candidate trees, superseded by v8/v8.1 validated candidate selection | v8/v8.1 evaluators | no active importers | no current docs/shell refs found |
| `diagnose_diffusion_reconstruction_v2.py` | DELETE_EXPERIMENTAL | v2 reconstruction diagnostic, not needed for current v8.1 accepted pipeline | v8 teacher-forced/all-window diagnostics | no active importers | no current docs/shell refs found |
| `debug_diffusion_v4_dataset_stats.py` | DELETE_EXPERIMENTAL | one-off v4 dataset debug; v4 training/sample scripts retained for baseline reproducibility | v8 dataset integrity checks and v4 README/trainer | no active importers | no current docs/shell refs found |
| `sanity_check_diffusion_v4_reconstruction_math.py` | DELETE_EXPERIMENTAL | one-off v4 reconstruction sanity check; v4 trainer/sample retained | v4 trainer/sample and v8 evaluation stack | no active importers | no current docs/shell refs found |
| `monitor_v7_w9.py` | DELETE_EXPERIMENTAL | monitor for obsolete nine-shard v7 target-generation run; v8 generator has multiprocessing and summaries | `generate_diffusion_v8_multitarget_scaled_residual_targets.py` | no active importers | no current docs/shell refs found |
| `run_v7_fast_shard.sh` | DELETE_EXPERIMENTAL | obsolete v7 shard launcher; v8 target generator supersedes it and v7 generator remains import utility | `generate_diffusion_v8_multitarget_scaled_residual_targets.py` | shell launcher only | not a current command |
| `run_v7_fast_9shard.sh` | DELETE_EXPERIMENTAL | obsolete v7 nine-shard launcher; v8 target generator supersedes it | same as above | shell launcher only | not a current command |

## Full Script Classification

| File | Classification | Notes |
|---|---|---|
| `adaptive_refine_mlp_predictions_with_ik.py` | KEEP_BASELINE | MLP+IK baseline variant |
| `analyze_prior_vs_diffusion_contribution_v8_1.py` | KEEP_ACTIVE | v8.1 contribution analysis |
| `audit_adaptive_prior_residual_branches.py` | KEEP_UTILITY | bootstrap-prior audit |
| `audit_and_compare_bootstrap_priors.py` | KEEP_BASELINE | current IK/MLP/diffusion comparison and lineage audit |
| `batch_generate_ik_experts.py` | KEEP_BASELINE | IK expert/baseline generation |
| `benchmark_ik_mlp_pipeline_smoothness.py` | KEEP_ACTIVE | current smoothness/runtime benchmark |
| `build_cartesian_expert_npz.py` | KEEP_BASELINE | IK/MLP dataset preparation |
| `build_diffusion_trajectory_dataset.py` | DELETE_SUPERSEDED | v1 dataset builder |
| `build_diffusion_trajectory_dataset_v2.py` | DELETE_SUPERSEDED | v2 dataset builder |
| `build_diffusion_v5_residual_window_dataset.py` | KEEP_BASELINE | v5 comparison lineage |
| `build_diffusion_v5b_residual_window_dataset_fk_condition.py` | KEEP_BASELINE | v5b comparison lineage and v6/v7 provenance |
| `build_diffusion_v6_strong_prior_residual_window_dataset.py` | KEEP_ACTIVE | strong-prior windows used upstream of v7/v8 |
| `build_diffusion_v7_cost_improving_training_dataset.py` | KEEP_ACTIVE | imported by v8 evaluator/generator |
| `build_diffusion_v8_multitarget_scaled_training_dataset.py` | KEEP_ACTIVE | current v8 dataset builder |
| `build_multistroke_full_pose_execution.py` | KEEP_ACTIVE | physical-robot multistroke export |
| `compare_ik_mlp_pipeline_jerk_over_time.py` | KEEP_ACTIVE | current comparison experiment |
| `conditional_unet1d_artist.py` | KEEP_UTILITY | v4/v5 model utility retained for baselines |
| `debug_diffusion_v4_dataset_stats.py` | DELETE_EXPERIMENTAL | one-off v4 debug |
| `debug_mlp_v3_prediction_format.py` | KEEP_BASELINE | MLP baseline format audit |
| `debug_prior_prediction_format.py` | KEEP_BASELINE | prior format audit |
| `diagnose_action_buffer_candidate_selection.py` | KEEP_BASELINE | v5/v6 rollout comparison diagnostic |
| `diagnose_diffusion_reconstruction_v2.py` | DELETE_EXPERIMENTAL | v2 diagnostic |
| `diagnose_diffusion_reconstruction_v4_unet.py` | KEEP_BASELINE | v4 baseline diagnostic imported by v4 diagnostics |
| `diagnose_diffusion_v4_prior_initialized_refinement.py` | KEEP_BASELINE | v4 baseline diagnostic |
| `diagnose_diffusion_v4_pure_sampling_vs_noised_expert.py` | KEEP_BASELINE | v4 baseline diagnostic |
| `diagnose_diffusion_v4_reverse_rollout.py` | KEEP_BASELINE | v4 baseline diagnostic |
| `diagnose_diffusion_v5_sampling_modes.py` | KEEP_BASELINE | v5 comparison diagnostic |
| `diagnose_residual_predictor_v5c_alpha_sweep.py` | KEEP_BASELINE | v5c comparison diagnostic |
| `diagnose_scaled_tapered_action_buffer_candidates.py` | KEEP_BASELINE | v5/v6 rollout diagnostic |
| `diagnose_v5c_scaled_residual_diffusion_refinement.py` | KEEP_BASELINE | v5c comparison diagnostic |
| `diagnose_warm_start_action_buffer_rollout.py` | KEEP_BASELINE | warm-start baseline diagnostic |
| `evaluate_base_tail_final_benchmark.py` | KEEP_BASELINE | v5/v6 benchmark comparison |
| `evaluate_diffusion_v6_teacher_forced_validation.py` | KEEP_ACTIVE | imported by v7/v8 evaluators |
| `evaluate_diffusion_v7_teacher_forced_validation.py` | KEEP_ACTIVE | imported by v8/v8.1/deployment |
| `evaluate_diffusion_v8_1_anchored_recursive_jerk_guard.py` | KEEP_ACTIVE | accepted v8.1 development rollout |
| `evaluate_diffusion_v8_1_path_disjoint_test_prior.py` | KEEP_ACTIVE | accepted path-disjoint confirmation |
| `evaluate_diffusion_v8_anchored_recursive_rollout.py` | KEEP_ACTIVE | frozen v8 baseline imported by v8.1/deployment |
| `evaluate_diffusion_v8_teacher_forced_all_windows.py` | KEEP_ACTIVE | v8 teacher-forced evaluation and helper module |
| `evaluate_global_anchored_receding_horizon_rollout.py` | KEEP_BASELINE | v5/v6 rollout comparison |
| `evaluate_path_conditioned_mlp.py` | KEEP_BASELINE | MLP baseline evaluation |
| `evaluate_prior_refinement_fk_robot_costs.py` | KEEP_BASELINE | FK robot-cost comparison |
| `evaluate_scaled_tapered_receding_horizon_rollout.py` | KEEP_BASELINE | scaled/tapered baseline |
| `evaluate_v5c_scaled_residual_full_trajectory_fk.py` | KEEP_BASELINE | v5c comparison |
| `generate_adaptive_mlp_ik_bootstrap_prior.py` | KEEP_ACTIVE | strong prior and IK/MLP baseline source |
| `generate_branch_continuous_deployment_input_from_cartesian_csv.py` | KEEP_ACTIVE | deployment input preparation |
| `generate_cartesian_path_variations.py` | KEEP_BASELINE | dataset/path generation |
| `generate_deployment_input_from_cartesian_csv.py` | KEEP_ACTIVE | deployment input preparation |
| `generate_diffusion_v7_cost_improving_residual_targets.py` | KEEP_ACTIVE | imported by v8 target/evaluation stack |
| `generate_diffusion_v7_cost_improving_residual_targets_gpu.py` | KEEP_BASELINE | retained v7 GPU acceleration experiment |
| `generate_diffusion_v8_multitarget_scaled_residual_targets.py` | KEEP_ACTIVE | current v8 target generator |
| `generate_ik_seed_path.py` | KEEP_ACTIVE | authoritative robot FK/IK/joint-limit utility |
| `generate_joint_trajectory_diffusion_v8_1.py` | KEEP_ACTIVE | deployment trajectory generator |
| `generate_mlp_v3_test_predictions.py` | KEEP_BASELINE | MLP baseline prediction helper |
| `generate_mlp_v3_train_test_predictions.py` | KEEP_BASELINE | MLP baseline/prior export |
| `make_final_result_plots.py` | KEEP_ACTIVE | presentation figure generation |
| `make_results_comparison_table.py` | KEEP_ACTIVE | presentation table generation |
| `make_unified_artistdiffusion_results_table.py` | KEEP_ACTIVE | presentation table generation |
| `make_v4_diffusion_refinement_results_table.py` | KEEP_BASELINE | v4 comparison table |
| `monitor_v7_w9.py` | DELETE_EXPERIMENTAL | obsolete v7 shard monitor |
| `orientation_aware_adaptive_ik.py` | KEEP_ACTIVE | deployment/IK utility |
| `plot_cartesian_tracking_error_over_time.py` | KEEP_ACTIVE | presentation plot |
| `plot_path_comparison.py` | KEEP_UTILITY | plotting utility |
| `plot_prior_refinement_fk_paths.py` | KEEP_BASELINE | prior/refinement comparison plot |
| `plot_tracking_error.py` | KEEP_UTILITY | plotting utility |
| `plot_unified_artistdiffusion_results.py` | KEEP_ACTIVE | presentation plot |
| `predict_path_conditioned_mlp.py` | KEEP_BASELINE | MLP model inference utility |
| `refine_mlp_predictions_with_ik.py` | KEEP_BASELINE | MLP+IK baseline |
| `rerank_diffusion_candidates.py` | DELETE_SUPERSEDED | legacy v1 candidate reranker |
| `run_diffusion_v8_anchored_multiseed.py` | KEEP_ACTIVE | frozen v8 runner retained for reproducibility |
| `run_diffusion_v8_focused_multiseed.py` | KEEP_BASELINE | v8 focused diagnostic runner |
| `sample_conditional_diffusion_trajectory.py` | DELETE_SUPERSEDED | v1 sampler |
| `sample_conditional_diffusion_trajectory_v2.py` | DELETE_SUPERSEDED | v2 sampler |
| `sample_conditional_diffusion_trajectory_v2_ddim.py` | DELETE_SUPERSEDED | v2 DDIM sampler |
| `sample_conditional_diffusion_trajectory_v3_x0.py` | DELETE_SUPERSEDED | v3 x0 sampler |
| `sample_conditional_diffusion_trajectory_v4_unet.py` | KEEP_BASELINE | v4 baseline sampler |
| `sample_conditional_diffusion_trajectory_v5_residual_unet.py` | KEEP_BASELINE | v5 baseline sampler |
| `sample_ranked_diffusion_candidates_v1.py` | DELETE_SUPERSEDED | v1 ranked candidate sampler |
| `sanity_check_diffusion_v4_reconstruction_math.py` | DELETE_EXPERIMENTAL | one-off v4 sanity diagnostic |
| `score_trajectory.py` | KEEP_UTILITY | shared FK/scoring utility and MLP baseline support |
| `summarize_diffusion_v8_1_anchored_multiseed.py` | KEEP_ACTIVE | v8.1 development summarizer |
| `summarize_diffusion_v8_1_path_disjoint_confirmation.py` | KEEP_ACTIVE | accepted path-disjoint summarizer |
| `summarize_diffusion_v8_anchored_multiseed.py` | KEEP_ACTIVE | frozen v8 baseline summarizer |
| `summarize_diffusion_v8_focused_multiseed.py` | KEEP_BASELINE | v8 focused diagnostic summarizer |
| `summarize_ranked_diffusion_candidates.py` | DELETE_SUPERSEDED | v1 ranked candidate summarizer |
| `test_benchmark_ik_mlp_pipeline_smoothness.py` | KEEP_UTILITY | current benchmark test |
| `test_compare_ik_mlp_pipeline_jerk_over_time.py` | KEEP_UTILITY | current comparison test |
| `test_legacy_smartjoint_export.py` | KEEP_UTILITY | physical export regression test |
| `test_load_urdf.py` | KEEP_UTILITY | robot model sanity test |
| `test_validate_multistroke_legacy_csv.py` | KEEP_UTILITY | physical export validation test |
| `train_conditional_diffusion_trajectory.py` | DELETE_SUPERSEDED | v1 trainer |
| `train_conditional_diffusion_trajectory_v2.py` | DELETE_SUPERSEDED | v2 trainer |
| `train_conditional_diffusion_trajectory_v3_x0.py` | DELETE_SUPERSEDED | v3 trainer |
| `train_conditional_diffusion_trajectory_v4_unet.py` | KEEP_BASELINE | v4 baseline trainer |
| `train_conditional_diffusion_trajectory_v5_residual_unet.py` | KEEP_BASELINE | v5 baseline and v5b compatibility |
| `train_conditional_diffusion_trajectory_v6_strong_prior_residual_unet.py` | KEEP_ACTIVE | imported by v7/v8 training/evaluation |
| `train_conditional_diffusion_trajectory_v7.py` | KEEP_ACTIVE | v7/v8 lineage and imported by v8 trainer/evaluator |
| `train_conditional_diffusion_trajectory_v8.py` | KEEP_ACTIVE | current model trainer |
| `train_path_conditioned_mlp.py` | KEEP_BASELINE | MLP baseline trainer |
| `train_residual_window_predictor_v5c.py` | KEEP_BASELINE | v5c comparison |
| `trajectory_costs.py` | KEEP_UTILITY | shared legacy cost utility |
| `validate_diffusion_v8_1_deployment_output.py` | KEEP_ACTIVE | deployment validator |
| `validate_expert_dataset.py` | KEEP_UTILITY | dataset validation |
| `validate_multistroke_full_pose_execution.py` | KEEP_ACTIVE | physical export validator |
| `deployment_bridge/touch_20260122/merge_force_altitude_v2.py` | KEEP_ACTIVE | physical robot deployment bridge |
| `run_v7_fast_shard.sh` | DELETE_EXPERIMENTAL | obsolete v7 shard launcher |
| `run_v7_fast_9shard.sh` | DELETE_EXPERIMENTAL | obsolete v7 shard launcher |

## Manual Review Bucket

No Python scripts are left in `REVIEW_MANUALLY` for this pass. Several retained baseline/diagnostic scripts are intentionally conservative because they support IK, MLP, v4/v5/v6 comparison artifacts, runtime benchmarks, or presentation tables.
