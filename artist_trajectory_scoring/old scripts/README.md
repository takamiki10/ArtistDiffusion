# Old scripts

These files were moved out of the main pipeline directory on 2026-10-05.
They are retained for historical reproduction and reference. Start with the
[current workflow](../README.md) for training, evaluation, and deployment.

All 37 moved files retain their original bytes. [manifest.json](manifest.json)
records each original path, archive path, and SHA-256 checksum. No model,
dataset, or saved result was moved. Older-named files still imported by the
current pipeline remain in the parent directory.

## Using archived code

This is a source archive, not a separately installed package. Historical
commands assumed the scripts lived in the parent directory. Some defaults
and subprocess paths are derived from `__file__`, so moving a script changes
those paths. Review and supply explicit input/output paths before reproducing
an old experiment; the archive has not been validated end to end.

For simple archived commands, run from the parent directory and expose its
shared helpers on `PYTHONPATH`, for example:

```bash
PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" python "old scripts/plot_path_comparison.py" --help
```

For exact historical relative-path behavior, work in a separate checkout or
copy of the project and restore the needed files to the original paths listed
in the manifest. Check for existing files before restoring. Historical guides
and the earlier cleanup record are in [docs/history](../docs/history/README.md).
The 18 files deleted in that earlier cleanup were already absent and are not
part of this archive.

## Inventory

| File | Why archived |
| --- | --- |
| [audit_adaptive_prior_residual_branches.py](audit_adaptive_prior_residual_branches.py) | Historical format, prior, or sampling diagnostic outside the v8.1 dependency chain. |
| [build_diffusion_v5_residual_window_dataset.py](build_diffusion_v5_residual_window_dataset.py) | Historical v5 dataset builder; current strong-prior/v8 builders remain in the root. |
| [build_multistroke_full_pose_execution.py.backup](build_multistroke_full_pose_execution.py.backup) | Saved backup of the active multistroke builder. |
| [debug_mlp_v3_prediction_format.py](debug_mlp_v3_prediction_format.py) | Historical format, prior, or sampling diagnostic outside the v8.1 dependency chain. |
| [debug_prior_prediction_format.py](debug_prior_prediction_format.py) | Historical format, prior, or sampling diagnostic outside the v8.1 dependency chain. |
| [diagnose_action_buffer_candidate_selection.py](diagnose_action_buffer_candidate_selection.py) | Historical format, prior, or sampling diagnostic outside the v8.1 dependency chain. |
| [diagnose_diffusion_reconstruction_v4_unet.py](diagnose_diffusion_reconstruction_v4_unet.py) | Historical format, prior, or sampling diagnostic outside the v8.1 dependency chain. |
| [diagnose_diffusion_v4_prior_initialized_refinement.py](diagnose_diffusion_v4_prior_initialized_refinement.py) | Historical format, prior, or sampling diagnostic outside the v8.1 dependency chain. |
| [diagnose_diffusion_v4_pure_sampling_vs_noised_expert.py](diagnose_diffusion_v4_pure_sampling_vs_noised_expert.py) | Historical format, prior, or sampling diagnostic outside the v8.1 dependency chain. |
| [diagnose_diffusion_v4_reverse_rollout.py](diagnose_diffusion_v4_reverse_rollout.py) | Historical format, prior, or sampling diagnostic outside the v8.1 dependency chain. |
| [diagnose_diffusion_v5_sampling_modes.py](diagnose_diffusion_v5_sampling_modes.py) | Historical format, prior, or sampling diagnostic outside the v8.1 dependency chain. |
| [diagnose_residual_predictor_v5c_alpha_sweep.py](diagnose_residual_predictor_v5c_alpha_sweep.py) | Historical format, prior, or sampling diagnostic outside the v8.1 dependency chain. |
| [diagnose_scaled_tapered_action_buffer_candidates.py](diagnose_scaled_tapered_action_buffer_candidates.py) | Historical format, prior, or sampling diagnostic outside the v8.1 dependency chain. |
| [diagnose_v5c_scaled_residual_diffusion_refinement.py](diagnose_v5c_scaled_residual_diffusion_refinement.py) | Historical format, prior, or sampling diagnostic outside the v8.1 dependency chain. |
| [diagnose_warm_start_action_buffer_rollout.py](diagnose_warm_start_action_buffer_rollout.py) | Historical format, prior, or sampling diagnostic outside the v8.1 dependency chain. |
| [evaluate_base_tail_final_benchmark.py](evaluate_base_tail_final_benchmark.py) | Historical residual/buffer rollout comparison, superseded by the documented v8/v8.1 evaluation. |
| [evaluate_global_anchored_receding_horizon_rollout.py](evaluate_global_anchored_receding_horizon_rollout.py) | Historical residual/buffer rollout comparison, superseded by the documented v8/v8.1 evaluation. |
| [evaluate_prior_refinement_fk_robot_costs.py](evaluate_prior_refinement_fk_robot_costs.py) | Historical residual/buffer rollout comparison, superseded by the documented v8/v8.1 evaluation. |
| [evaluate_scaled_tapered_receding_horizon_rollout.py](evaluate_scaled_tapered_receding_horizon_rollout.py) | Historical residual/buffer rollout comparison, superseded by the documented v8/v8.1 evaluation. |
| [evaluate_v5c_scaled_residual_full_trajectory_fk.py](evaluate_v5c_scaled_residual_full_trajectory_fk.py) | Historical residual/buffer rollout comparison, superseded by the documented v8/v8.1 evaluation. |
| [fit_touch_mlp_visualization_only.py](fit_touch_mlp_visualization_only.py) | One-trajectory visualization fit; not an independent baseline or deployment step. |
| [generate_diffusion_v7_cost_improving_residual_targets_gpu.py](generate_diffusion_v7_cost_improving_residual_targets_gpu.py) | Experimental v7 GPU target generator, superseded by the v8 target-generation entry point. |
| [make_final_result_plots.py](make_final_result_plots.py) | Historical v3/v4/v5 comparison table or plotting utility; current paper plots remain in the root. |
| [make_results_comparison_table.py](make_results_comparison_table.py) | Historical v3/v4/v5 comparison table or plotting utility; current paper plots remain in the root. |
| [make_unified_artistdiffusion_results_table.py](make_unified_artistdiffusion_results_table.py) | Historical v3/v4/v5 comparison table or plotting utility; current paper plots remain in the root. |
| [make_v4_diffusion_refinement_results_table.py](make_v4_diffusion_refinement_results_table.py) | Historical v3/v4/v5 comparison table or plotting utility; current paper plots remain in the root. |
| [plot_path_comparison.py](plot_path_comparison.py) | Historical v3/v4/v5 comparison table or plotting utility; current paper plots remain in the root. |
| [plot_prior_refinement_fk_paths.py](plot_prior_refinement_fk_paths.py) | Historical v3/v4/v5 comparison table or plotting utility; current paper plots remain in the root. |
| [plot_tracking_error.py](plot_tracking_error.py) | Historical v3/v4/v5 comparison table or plotting utility; current paper plots remain in the root. |
| [plot_unified_artistdiffusion_results.py](plot_unified_artistdiffusion_results.py) | Historical v3/v4/v5 comparison table or plotting utility; current paper plots remain in the root. |
| [sample_conditional_diffusion_trajectory_v4_unet.py](sample_conditional_diffusion_trajectory_v4_unet.py) | Historical standalone sampler, superseded by v8.1 generation. |
| [sample_conditional_diffusion_trajectory_v5_residual_unet.py](sample_conditional_diffusion_trajectory_v5_residual_unet.py) | Historical standalone sampler, superseded by v8.1 generation. |
| [test_load_urdf.py](test_load_urdf.py) | Standalone early URDF smoke script; current export/FK regression tests remain in the root. |
| [train_conditional_diffusion_trajectory_v4_unet.py](train_conditional_diffusion_trajectory_v4_unet.py) | Superseded training entry point, not imported by current training/deployment. |
| [train_conditional_diffusion_trajectory_v7.py](train_conditional_diffusion_trajectory_v7.py) | Superseded training entry point, not imported by current training/deployment. |
| [train_residual_window_predictor_v5c.py](train_residual_window_predictor_v5c.py) | Superseded training entry point, not imported by current training/deployment. |
| [trajectory_costs.py](trajectory_costs.py) | Legacy scoring implementation used by the archived v4 sampler. |
