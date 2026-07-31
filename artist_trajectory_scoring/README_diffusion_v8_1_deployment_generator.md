# Diffusion v8.1 Deployment Trajectory Generator

`generate_joint_trajectory_diffusion_v8_1.py` is a deployment-oriented
inference interface for the frozen diffusion v8.1 method. It takes one desired
Cartesian path and one strong 6-DoF joint-space prior, then produces one final
100-step joint trajectory plus FK, dynamics, safety, selection, fallback, and
provenance outputs.

It does not train, fine-tune, or modify any model checkpoint.

## Frozen Method

The generator imports and reuses the frozen rollout implementation:

```python
evaluate_diffusion_v8_1_anchored_recursive_jerk_guard.run_rollout_v8_1
evaluate_diffusion_v8_1_anchored_recursive_jerk_guard.load_validated_inference_bundle
```

It also reuses frozen v8/v7 helpers for FK, candidate scoring, hard safety,
fallback, full-path metrics, and worker initialization. It does not copy or
reimplement DDIM sampling, condition construction, anchored-prior construction,
the history-aware jerk guard, candidate ordering, or robot-aware metrics.

Frozen settings:

```text
training_dataset_dir = data/cartesian_expert_dataset_v3/diffusion_v8_multitarget_scaled_training_dataset_100paths
model_dir = models/diffusion_v8_multitarget_scaled_residual_unet_100paths_epsilon_only_seed42
checkpoint_state = raw_last_epoch187
target_scale = 1.0
output_alpha = 0.125
k = 8
ddim_steps = 50
eta = 0.0
horizon = 32
execution_horizon = 8
anchoring_horizon = 8
history-aware jerk tolerance = 1e-12
```

The script fails clearly if a frozen parameter is changed. `sampling_seed` may
vary and is recorded in every deployment artifact.

The checkpoint state is also frozen. The deployment generator only accepts
`checkpoint_state = raw_last_epoch187`; changing the checkpoint state is a
configuration error, not an experimental option.

## Input NPZ

Required arrays:

```text
desired_path    shape=(100,3)
strong_prior_q  shape=(100,6)
target_rpy      shape=(3,)
target_quaternion shape=(4,), XYZW order
target_rotation_matrix shape=(3,3)
strong_prior_quaternion shape=(100,4), XYZW order
strong_prior_orientation_error_rad shape=(100,)
```

Required orientation metadata:

```text
mean_orientation_error_rad
maximum_orientation_error_rad
maximum_orientation_error_gate_rad
orientation_constraint_enforced = True
orientation_fk_frame = xMateCR7_link6
target_z
maximum_z_error_gate_m
strong_prior_z_error_m shape=(100,)
mean_strong_prior_z_error_m
maximum_strong_prior_z_error_m
z_constraint_enforced = True
z_fk_frame = xMateCR7_link6
```

The orientation gate must satisfy `0 < gate <= 0.05 rad`; larger values are
rejected because they weaken the deployment orientation constraint. The fixed-Z
gate must satisfy `0 < gate <= 0.001 m`; stricter values are allowed, but larger
values are rejected.

The target quaternion uses the same fixed-axis roll-pitch-yaw convention as
`quaternion_from_euler(roll, pitch, yaw)`. The generator reconstructs the
target rotation independently from both RPY and quaternion and requires them
to agree.

Additional arrays and scalar/string metadata:

```text
strong_prior_ee shape=(100,3)
timestamps      shape=(100,)
path_name
source_method
source_checkpoint
source_checkpoint_sha256
source_description
```

If `strong_prior_ee` is absent, it is computed with authoritative FK. If
`timestamps` is absent, uniform timestamps are generated over
`--trajectory_duration_seconds` with both endpoints included.

The strong prior is the safe fallback. Before diffusion sampling, the generator
validates the prior with the same segmented receding-horizon fallback hard
safety rules used by the frozen evaluation stack. It also recomputes the
`xMateCR7_link6` full transform for all 100 prior states, verifies the stored
quaternions and per-sample orientation errors, and rejects the input when its
maximum orientation error exceeds the recorded positive gate.
The prior Z coordinate is independently recomputed from authoritative
`xMateCR7_link6` FK and must remain within the recorded non-weakening Z gate.

## Path Identity

The generator creates a collision-safe deployment identity:

```text
deployment__<sanitized path name>__<8-char hash>
```

The hash is derived from `desired_path`, `strong_prior_q`, `target_rpy`,
`target_z`, the orientation gate, and the Z gate. This deployment ID is used as
`PhysicalPathRecord.path_id`, which controls candidate-seed identity. Different
pose constraints therefore cannot share a deployment identity.
Original input metadata is stored separately as:

```text
input_path_name
deployment_path_id
input_file
input_sha256
```

## Example Command

```bash
python generate_joint_trajectory_diffusion_v8_1.py \
  --input_npz data/my_deployment_input.npz \
  --output_dir results/my_deployment_trajectory \
  --sampling_seed 53 \
  --device cuda \
  --overwrite
```

## Outputs

The output directory contains:

```text
deployment_input_copy.npz
deployment_trajectory_full.npz
deployment_trajectory.csv
deployment_joint_positions.csv
deployment_joint_dynamics.csv
deployment_cartesian_tracking.csv
deployment_segment_decisions.csv
deployment_candidate_results.csv
deployment_metrics.json
deployment_report.txt
plots/
```

Only accepted trajectories also produce:

```text
approved_simulation_trajectory.csv
approved_simulation_trajectory.npz
```

The approved CSV is simulation-friendly:

```text
time_seconds,q1,q2,q3,q4,q5,q6
```

The full NPZ and metrics JSON store the exact URDF path and SHA-256 used to
construct the robot model. The input copy and approved simulation NPZ also
store the same URDF identity.

The full deployment NPZ and metrics JSON also store:

```text
target_rpy
target_quaternion
target_rotation_matrix
strong_prior_quaternion
final_quaternion
strong_prior_orientation_error_rad
final_orientation_error_rad
mean_prior_orientation_error_rad
maximum_prior_orientation_error_rad
mean_final_orientation_error_rad
maximum_final_orientation_error_rad
maximum_orientation_error_gate_rad
orientation_constraint_enforced
orientation_fk_frame
target_z
strong_prior_z_error_m
final_z_error_m
mean_prior_z_error_m
maximum_prior_z_error_m
mean_final_z_error_m
maximum_final_z_error_m
maximum_z_error_gate_m
z_constraint_enforced
z_fk_frame
```

The main process resolves the URDF path once, records that resolved path, and
passes the same resolved URDF to every CPU worker process used for candidate FK
and robot-aware scoring.

## Safety Verdict

The generator prints and records one of:

```text
V8_1_DEPLOYMENT_TRAJECTORY_ACCEPTED
V8_1_DEPLOYMENT_TRAJECTORY_REJECTED
```

Acceptance requires:

- frozen full-path safety pass;
- maximum actual internal joint step `<= 0.20` rad;
- finite final joint positions;
- finite FK positions;
- finite full-transform rotations, quaternions, and orientation errors;
- joint-limit checks pass;
- timestamp checks pass;
- strong-prior maximum orientation error is within the recorded gate;
- final maximum orientation error is within the recorded gate;
- strong-prior and final maximum Z errors are within the recorded fixed-Z gate;
- no frozen evaluator integrity assertion failed.

Robot-aware delta score, Cartesian error delta, accepted-step rate, and fallback
rate are reported but do not independently reject an otherwise safe trajectory.
A trajectory equal to the complete safe prior may therefore be accepted.

If rejected, the generator still writes diagnostic JSON/report/CSV/NPZ outputs,
but it does not create approved simulation exports and returns a nonzero exit
code.

### Frozen sampler orientation boundary

The frozen v8.1 sampler and candidate selector are not claimed to optimize
orientation or Z independently. Both constraints are protected after rollout:
the deployment generator recomputes full-transform FK and rejects the complete
trajectory if either gate is violated. An orientation- or Z-rejected trajectory
never receives approved simulation exports.

Adding candidate-level orientation scoring or an orientation-aware fallback
inside frozen selection would be a separate future method change; it is not
silently introduced by this deployment wrapper.

## Prior Fallback Versus Diffusion Modification

v8.1 may select diffusion corrections for some receding-horizon segments or
fall back to the anchored strong prior. Fallback is not failure: if the prior is
safe and the diffusion candidates do not pass the frozen gates plus
history-aware jerk guard, the safe prior segment is executed.

Segment-level provenance is written to:

```text
deployment_segment_decisions.csv
deployment_candidate_results.csv
```

Each row includes:

```text
sampling_seed
deployment_path_id
input_path_name
input_sha256
verdict
```

## Provenance

`deployment_metrics.json` records:

- generator script SHA-256;
- orientation-aware IK/FK helper SHA-256;
- frozen v8.1 script SHA-256;
- frozen v8 script SHA-256;
- training dataset path;
- model-directory file hashes;
- checkpoint state hash;
- source path-conditioned MLP checkpoint SHA-256;
- input NPZ SHA-256;
- URDF path and SHA-256;
- Python, NumPy, PyTorch, CUDA information;
- Git commit when available.

Git metadata failure does not block inference; it is recorded as null plus an
error reason.

## Validator

Validate an output directory with:

```bash
python validate_diffusion_v8_1_deployment_output.py \
  --output_dir results/my_deployment_trajectory \
  --require_accepted
```

The validator checks required files, strict JSON, trajectory shapes,
timestamps, finite values, CSV/NPZ agreement, FK recomputation, internal joint
step recomputation, frozen config, provenance fields, and verdict consistency.

Validation uses the exact recorded URDF path and SHA-256 from the deployment
artifact. It does not fall back to a default robot model. The validator
independently:

- recomputes prior FK and final FK;
- reconstructs the target rotation independently from recorded RPY and
  quaternion and requires both representations to agree;
- recomputes prior and final `xMateCR7_link6` full-transform FK, including
  XYZW quaternions;
- recomputes all 100 prior/final rotation-geodesic errors and compares the
  stored quaternion arrays, error arrays, means, maxima, and gate;
- independently recomputes all 100 prior/final Z errors from full-transform FK
  and compares target Z, arrays, means, maxima, gate, enforcement boolean, and
  FK frame across the full artifact, input copy, metrics JSON, and approved NPZ;
- calls the frozen full-path metric helper and compares safety-critical metrics;
- repeats every executed-prefix receding-horizon hard-safety check;
- validates all full NPZ arrays and segment arrays;
- checks accepted/fallback segment masks before dtype conversion, verifies that
  they are complements, and cross-validates selected candidate indices;
- compares segment counts, selected-diffusion counts, fallback counts,
  accepted-step rate, and fallback rate against `deployment_metrics.json`;
- recomputes joint velocity, acceleration, and jerk from physical timestamps
  with `np.gradient(..., edge_order=2)`;
- checks every sample-level CSV against the full NPZ, including `sample_index`
  and `time_seconds` in all four sample-level CSV files;
- checks segment-decision and candidate provenance by `rollout_step`, not row
  order, and verifies selected-candidate consistency;
- derives the executed sample ranges from decision rows and verifies that
  samples 1..99 are contiguous, non-overlapping, and sourced from either
  `accepted_diffusion_candidate` or `anchored_prior_fallback` as recorded;
- compares all scalar metadata in `deployment_trajectory_full.npz` with
  `deployment_metrics.json`, including checkpoint state, URDF identity, model
  identity, input identity, frozen settings, and verdict;
- recomputes independent safety flags instead of trusting stored flags,
  including timestamp, finite joint/FK, joint-limit count and magnitude, and
  executed-prefix failure reasons;
- includes finite-orientation, prior-orientation, and final-orientation passes
  in its independently derived accepted condition;
- includes finite-Z, prior-Z, and final-Z passes in its independently derived
  accepted condition;
- checks approved simulation CSV/NPZ contents when the verdict is accepted;
- verifies generator, frozen v8.1, frozen v8, URDF, and model-file hashes where
  local files are present.

It prints:

```text
V8_1_DEPLOYMENT_OUTPUT_VALIDATION_PASSED
```

on success.

A passed validator means the artifact is internally consistent and
simulation-ready. It does not authorize unreviewed full-speed hardware
execution.

## Deployment Progression

Accepted output is simulation-ready, but it is not automatically approved for
full-speed hardware execution. Recommended progression:

1. offline validation;
2. simulation;
3. reduced-speed robot test;
4. gradual speed increase after review.
