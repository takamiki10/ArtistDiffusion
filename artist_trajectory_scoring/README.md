# ArtistDiffusion: start here

The top level contains the eight entry points for the **v8 training → v8.1
trajectory generation** pipeline. Supporting scripts are grouped by purpose.
Run commands from this directory (`artist_trajectory_scoring/`) so relative
input and output paths resolve consistently.

```text
artist_trajectory_scoring/
├── generate_adaptive_mlp_ik_bootstrap_prior.py
├── generate_diffusion_v8_multitarget_scaled_residual_targets.py
├── build_diffusion_v8_multitarget_scaled_training_dataset.py
├── train_conditional_diffusion_trajectory_v8.py
├── generate_deployment_input_from_cartesian_csv.py
├── generate_branch_continuous_deployment_input_from_cartesian_csv.py
├── generate_joint_trajectory_diffusion_v8_1.py
├── build_multistroke_full_pose_execution.py
├── evaluation/       # Evaluation, validation, multiseed runners and summaries
├── dependencies/     # Shared model, robot, IK/MLP and dataset helpers
├── benchmark/        # Comparisons, audits and benchmark figures
├── tests/            # Automated regression tests
├── docs/             # Detailed workflow guides and historical documentation
└── old scripts/      # Archived experiments and backups
```

Existing `data/`, `models/`, `robot_model/`, `results/`, `logs/`, and
`deployment_bridge/` directories keep their locations. They hold inputs,
artifacts, and deployment assets rather than top-level pipeline entry points.

## Main workflow

| Step | Entry point | Guide |
| --- | --- | --- |
| Build the strong IK/MLP prior | `generate_adaptive_mlp_ik_bootstrap_prior.py` | Inspect `--help` for input datasets and checkpoint options |
| Generate residual targets | `generate_diffusion_v8_multitarget_scaled_residual_targets.py` | [Target-generation guide](docs/guides/README_diffusion_v8_multitarget_scaled.md) |
| Assemble the training dataset | `build_diffusion_v8_multitarget_scaled_training_dataset.py` | Same target-generation guide |
| Train v8 | `train_conditional_diffusion_trajectory_v8.py` | [Training guide](docs/guides/README_diffusion_v8_training.md) |
| Prepare Cartesian deployment input | `generate_deployment_input_from_cartesian_csv.py` or `generate_branch_continuous_deployment_input_from_cartesian_csv.py` | Inspect `--help` for pose constraints and input options |
| Generate a joint trajectory | `generate_joint_trajectory_diffusion_v8_1.py` | [Deployment guide](docs/guides/README_diffusion_v8_1_deployment_generator.md) |
| Assemble multiple strokes | `build_multistroke_full_pose_execution.py` | Inspect `--help`; validate with `evaluation/validate_multistroke_full_pose_execution.py` |

For an already prepared input NPZ:

```bash
python generate_joint_trajectory_diffusion_v8_1.py \
  --input_npz data/my_deployment_input.npz \
  --output_dir results/my_deployment_trajectory \
  --sampling_seed 53 --device cuda

python evaluation/validate_diffusion_v8_1_deployment_output.py \
  --output_dir results/my_deployment_trajectory --require_accepted
```

The example input path is a placeholder. The deployment guide specifies its
required arrays. Frozen deployment uses `raw_last_epoch187` in
`models/diffusion_v8_multitarget_scaled_residual_unet_100paths_epsilon_only_seed42/`
and matching metadata in
`data/cartesian_expert_dataset_v3/diffusion_v8_multitarget_scaled_training_dataset_100paths/`.
Keep the checkpoint and normalization metadata together.

## Supporting workflows

- [Evaluation](evaluation/README.md): v8/v8.1 evaluation, path-disjoint confirmation, summaries, and deployment validators.
- [Dependencies](dependencies/README.md): model implementations, FK/IK, MLP training/inference, expert-data preparation, and shared dataset builders.
- [Benchmark](benchmark/README.md): smoothness/runtime comparisons, jerk and Cartesian-error plots, and prior analysis.
- [Tests](tests/README.md): regression checks and commands.
- [Guides](docs/README.md): detailed instructions for each stage.
- [Old scripts](<old scripts/README.md>): 37 historical scripts/backups with original checksums; not current entry points.

## Environment and tests

Use [../requirements.txt](../requirements.txt) as the existing dependency list;
SciPy is also required by the robot/trajectory helpers. GPU runs need a
compatible PyTorch/CUDA installation.

```bash
python -m unittest discover -s tests -v
```

Scripts initialize the shared module search paths themselves; no custom
`PYTHONPATH` or installation step is needed for current command-line entry
points. In an interactive session, run `import dependencies` first before
importing helpers by their established module names. These names are preserved
for saved checkpoint metadata and worker-process compatibility.

## Existing artifacts and local supplements

The reorganization changes source locations and source-file hashes, not model
weights or saved results. Validators still enforce recorded source hashes.
To reproduce or validate a historical artifact against its exact source,
use the Git commit recorded in that artifact in a separate checkout; do not
replace its recorded hashes with new ones. Archived scripts retain their
original bytes and may require their historical layout.

The following research supplements remain local and are not included in this
layout commit: `physical_validation/`, `k_candidate_ablation/`, new result
folders, and these five scripts now located under `benchmark/`:
`audit_reference_qp_current_paper.py`, `evaluate_reference_qp_current_paper_audit.py`,
`make_figure4_qp_vs_proposed.py`, `host_generate_physical_a_10.py`, and
`host_parallel_physical_a_10.py`. They need a separate handoff when required.
