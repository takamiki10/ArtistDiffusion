# Dependencies

Shared model implementations, robot FK/IK, MLP and expert-data preparation, and earlier-version helpers still required by v8/v8.1. Older version numbers here do not mean the file is obsolete. Run helper commands from the pipeline root using `python dependencies/SCRIPT.py --help`.

[Back to the pipeline overview](../README.md)

## Scripts

- [adaptive_refine_mlp_predictions_with_ik.py](adaptive_refine_mlp_predictions_with_ik.py)
- [batch_generate_ik_experts.py](batch_generate_ik_experts.py)
- [build_cartesian_expert_npz.py](build_cartesian_expert_npz.py)
- [build_diffusion_v5b_residual_window_dataset_fk_condition.py](build_diffusion_v5b_residual_window_dataset_fk_condition.py)
- [build_diffusion_v6_strong_prior_residual_window_dataset.py](build_diffusion_v6_strong_prior_residual_window_dataset.py)
- [build_diffusion_v7_cost_improving_training_dataset.py](build_diffusion_v7_cost_improving_training_dataset.py)
- [conditional_unet1d_artist.py](conditional_unet1d_artist.py)
- [generate_cartesian_path_variations.py](generate_cartesian_path_variations.py)
- [generate_diffusion_v7_cost_improving_residual_targets.py](generate_diffusion_v7_cost_improving_residual_targets.py)
- [generate_ik_seed_path.py](generate_ik_seed_path.py)
- [generate_mlp_v3_test_predictions.py](generate_mlp_v3_test_predictions.py)
- [generate_mlp_v3_train_test_predictions.py](generate_mlp_v3_train_test_predictions.py)
- [orientation_aware_adaptive_ik.py](orientation_aware_adaptive_ik.py)
- [predict_path_conditioned_mlp.py](predict_path_conditioned_mlp.py)
- [refine_mlp_predictions_with_ik.py](refine_mlp_predictions_with_ik.py)
- [score_trajectory.py](score_trajectory.py)
- [train_conditional_diffusion_trajectory_v5_residual_unet.py](train_conditional_diffusion_trajectory_v5_residual_unet.py)
- [train_conditional_diffusion_trajectory_v6_strong_prior_residual_unet.py](train_conditional_diffusion_trajectory_v6_strong_prior_residual_unet.py)
- [train_path_conditioned_mlp.py](train_path_conditioned_mlp.py)

`__init__.py` defines `PROJECT_ROOT`, initializes source search paths, and provides
`script_path(filename)` for resolving current script locations. The existing bare
module names are retained for checkpoint metadata and multiprocessing; current
scripts import this initializer before importing shared helpers.
