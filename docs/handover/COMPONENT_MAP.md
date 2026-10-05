# System component map

Review date: 5 October 2026.
Current Jetson baseline: `f3e217c`, fetched 5 October 2026, with the current workflow README and historical source organization. Initial reviewed baseline: `028dfe8bfaf42786f3d1eec01453cd53a365705e`.

## Component responsibilities

| Component | Location | Observed responsibility | Status |
| --- | --- | --- | --- |
| Jetson trajectory stack | `linux/artist_trajectory_scoring/` | Data generation, MLP/IK reference construction, residual diffusion, scoring, validation, and multistroke exports | Source inspected; not executed here |
| Windows ROKAE component | `windows/RokaeSDK/` | Source and documentation from user-confirmed `C:/RokaeSDK` | 66 byte-verified files copied; large artifacts and dependencies pending |
| Drawing end-effector | Not yet inventoried | Inclination and contact-force control described in engineering report | Implementation and ownership to locate |
| Experiment archive | Separate storage, indexed from documentation | Model weights, normalization, datasets, trajectories, and experiment evidence | Collection pending |

## Jetson-to-Windows boundary

`build_multistroke_full_pose_execution.py` includes a legacy ROKAE export named `SmartJoint_Data_diffusion.csv`. It exports `Timestamp`, `TouchType`, `joint1` through `joint6`, and `OriginalStatus`. The exporter retains the source `time_seconds` and joint values. Stroke/transition markers are generated in the exporter.

The Windows importer must be checked against this exact format. Do not assume matching angle units, controller timing, pen-up semantics, or playback speed from the filename alone. The newer simulation export uses `time_seconds,q1,q2,q3,q4,q5,q6` and is a different interface.

## Reproduction files missing from the clone

The repository's ignore rules exclude `data/`, checkpoints (`*.pt`, `*.pth`, `*.ckpt`), NPZ/NPY, CSV, images, and archives. Some model metadata and result JSON/logs are present, but the final model directory inspected contains only metadata/history/integrity JSON, not weights.

Prioritize collection of:

1. Final model bundle: `models/diffusion_v8_multitarget_scaled_residual_unet_100paths_epsilon_only_seed42`, including the checkpoint state identified as `raw_last_epoch187` by the deployment README.
2. Dataset bundle: `data/cartesian_expert_dataset_v3/diffusion_v8_multitarget_scaled_training_dataset_100paths`, including loader-required normalization, schedule, metadata, and split files.
3. MLP checkpoint and metadata used to generate deployment priors.
4. Test prior: `data/cartesian_expert_dataset_v3/adaptive_mlp_ik_bootstrap_prior/test_prior.npz` and held-out path manifest under `results/diffusion_v8_1_path_disjoint_confirmation_test30_seeds53_57/metadata/`.
5. Retained result trajectories, seed-level metrics, final figures, configurations, and a small reproducible example.
6. The authoritative URDF and its identity. The deployment generator and validator record/check its path and SHA-256.

Confirm exact dependencies through the loaders before declaring the archive complete. This list is a collection priority, not an exhaustive dependency closure.

## Change history anchors

- `dee6e2b`, 26 June 2026: Updated IK and MLP training.
- `5b9adaa`, 14 July 2026: Added warm start generation.
- `e5780a8`, 23 July 2026: Confirm v8 epsilon-only teacher-forced performance.
- `2109ac5`, 25 July 2026: v8.1 savepoint.
- `082d46b`, 26 July 2026: Add v8.1 path-disjoint confirmation pipeline.
- `f05da70`, 26 July 2026: Add validated v8.1 deployment trajectory generator.
- `028dfe8`, 31 July 2026: Final Version.

These are existing commit titles, not yet a detailed explanation of each diff. For the final handover, pair each meaningful change with before/after behavior, reason, files, result evidence, and validation status.

## Validation boundary

The deployment README distinguishes internally validated simulation output from physical execution. The engineering report states that final diffusion trajectories had numerical evaluation and physical execution remained future work. Local SDK notes describe later experimental preparation, so physical status must be reconciled with retained logs before finalizing the PDF.

## Next collection step

The user confirmed `C:/RokaeSDK` as the Windows component. Its source and documentation have been copied locally, with a full file inventory under `windows/provenance/`. Collect essential physical sessions, scans, prepared inputs, SDK/runtime dependencies, and referenced `C:/laptop_handoff/laptop_handoff` files next. The updated Jetson README also identifies local physical-validation, candidate-count ablation, reference-QP analysis, and host-generation supplements missing from a fresh GitHub clone. Preserve these separately where needed.

The latest layout groups supporting code under dependencies/, evaluation/, benchmark/, tests/, and docs/. Data archives will be stored on Box; see BOX_ARCHIVE_PLAN.md.
