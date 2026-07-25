# Diffusion v8 Anchored Recursive Rollout

## Evaluation Modes

**Teacher-forced evaluation** scores independent physical windows while every
window starts from the frozen strong-prior state.

**Anchored recursive evaluation** executes selected prefixes recursively. The
current state therefore comes from earlier accepted or fallback actions, but
every prediction window is deterministically guided back to the original
strong prior.

**Fully closed-loop evaluation** would additionally update observations and
the task state from a live or simulated environment. This implementation is
not a fully closed-loop robot evaluation.

## Frozen Configuration

- Model:
  `models/diffusion_v8_multitarget_scaled_residual_unet_100paths_epsilon_only_seed42`
- Checkpoint: raw state from `last_checkpoint.pt`, epoch 187
- Checkpoint label: `raw_last_epoch187`
- Target scale: `1.0`
- Output alpha: `0.125`
- Nested K: `1`, `4`, `8`
- Primary K: `8`
- DDIM steps: `50`
- Eta: `0.0`
- Horizon: `32`
- Execution horizon: `8`
- Anchoring horizon: `8`
- Sampling seeds: `43`, `44`, `45`, `46`, `47`

## Validated Backend

`evaluate_diffusion_v8_teacher_forced_all_windows.py` exposes concrete
adapters used by the anchored evaluator:

- `load_authoritative_physical_path_population()`
- `load_validated_inference_bundle()`
- `build_recursive_condition_norm()`
- `sample_ddim_candidates()`
- `evaluate_candidate_with_validated_semantics()`
- `compute_fk_positions()`
- `compute_full_trajectory_metrics()`

These adapters reuse the existing v6/v7/v8 checkpoint validation, v6 model
construction and linear diffusion schedule, v6 DDIM `sample_batch()`, v7
xMateCR7 FK, hard-safety gates, compatibility gates, singularity and
manipulability metrics, and v7 robot-aware `delta_score`.

The anchored implementation contains no alternate cosine schedule, model
loader, FK implementation, limit checker, compatibility formula, or custom
robot-aware score.

## Authoritative Population

The loader reads `path_split.csv` and verifies that training and validation
paths do not overlap. The ordinary population is the exact deterministic set
of 20 validation paths.

Complete 100-sample trajectories are reconstructed from the strong-prior
window source recorded in the v8 dataset metadata. Retained target rows in
`validation_windows.npz` do not define the rollout population, so physical
zero-target windows are not omitted.

`path_0306` and `path_0370` form a separate difficult diagnostic population.
They are additional to the 20 ordinary paths and never affect the advancement
decision.

## Anchoring Formula

For original strong prior `p`, recursively executed current state `q`, window
offset `i`, and anchoring horizon `A=8`:

```text
d = q - p[start]
u = clip(i / A, 0, 1)
smoothstep(u) = u^2 (3 - 2u)
anchored[i] = p[start + i] + (1 - smoothstep(u)) d
```

The implementation sets:

```text
anchored[0] = q exactly
anchored[A:] = original strong prior exactly
```

The boundary state is stored once. Each rollout step executes window samples
`1:execution_count+1`; it never re-executes sample zero. Explicit assertions
verify that all `T=100` physical indices are written exactly once with no
duplicates or gaps.

If no candidate is selectable, fallback executes the anchored prior prefix,
not the unanchored original prior.

## Recursive Condition

The condition builder reconstructs the exact 38 v7 features from the anchored
prior, desired path, current executed state, authoritative path start, and
validated xMateCR7 FK. Raw `target_scale` is appended at feature index 38
before applying the saved 39-dimensional normalization.

Feature ordering and normalization round-trip checks are mandatory.

## Nested K

Each recursive state generates exactly eight deterministic candidates once:

- K=1 evaluates candidate 0.
- K=4 evaluates candidates 0 through 3.
- K=8 evaluates candidates 0 through 7.

K trajectories are separate recursive rollouts and may diverge. Generated
candidates are cached across K only when the complete normalized condition
and deterministic seed tuple are byte-identical. Diverged recursive states
cannot share cached samples.

Selection requires all of:

```text
validated hard safety
execution-prefix Cartesian mean error improvement over anchored prior
validated robot-aware delta_score < 0
all v7 compatibility gates
```

The selectable candidate with minimum `delta_score` is executed.

Local boundary scoring is unchanged and remains physically meaningful. Every
local action window has a real entry from the previously executed state and a
real exit toward its anchored continuation, so the established boundary-step
and boundary-acceleration terms remain part of local compatibility and
selection.

## Single Seed

```bash
python evaluate_diffusion_v8_anchored_recursive_rollout.py \
  --dataset_dir data/cartesian_expert_dataset_v3/diffusion_v8_multitarget_scaled_training_dataset_100paths \
  --target_generation_dir data/cartesian_expert_dataset_v3/diffusion_v8_multitarget_scaled_residual_targets_100paths \
  --model_dir models/diffusion_v8_multitarget_scaled_residual_unet_100paths_epsilon_only_seed42 \
  --output_dir results/diffusion_v8_anchored_recursive_seed43 \
  --checkpoint_state raw_last_epoch187 \
  --target_scale 1.0 \
  --output_alpha 0.125 \
  --k_values 1 4 8 \
  --sampling_seed 43 \
  --ddim_steps 50 \
  --eta 0.0 \
  --horizon 32 \
  --execution_horizon 8 \
  --anchoring_horizon 8 \
  --num_cpu_workers 8 \
  --gpu_batch_size 8 \
  --device cuda \
  --include_difficult_paths \
  --overwrite
```

`--path_ids` preserves explicit user ordering after membership validation.
`--smoke_test` selects two deterministic ordinary paths. `--max_paths` and
smoke mode are diagnostics and must not be used for the final decision.

## Five Seeds

```bash
python run_diffusion_v8_anchored_multiseed.py \
  --device cuda \
  --num_cpu_workers 8 \
  --gpu_batch_size 8 \
  --overwrite
```

The runner prints each complete argument-safe command, evaluates seeds 43–47,
and invokes the summarizer. It stops on failure unless
`--continue_on_error` is supplied.

## Outputs

Each seed writes:

- `anchored_rollout_decisions.csv`
- `anchored_candidate_results.csv`
- `anchored_full_path_metrics.csv`
- `anchored_ordinary_aggregate.csv`
- `anchored_difficult_aggregate.csv`
- `anchored_combined_diagnostic_aggregate.csv`
- `anchored_rollout_summary.json`
- `anchored_rollout_report.txt`
- `trajectories/<path>/anchored_rollout_k{1,4,8}.npz`
- plots under `plots/`

Every trajectory NPZ contains:

- `strong_prior_q`
- `rollout_q`
- `desired_path`
- `prior_ee`
- `rollout_ee`
- `executed_source`
- `accepted_step_mask`
- `fallback_step_mask`
- `selected_candidate_indices`
- `applied_correction_norms`
- `window_start_indices`
- `executed_indices`

Plots cover Cartesian error over depth, cumulative cost components, correction
norm, accepted/fallback timeline, all six joints, joint deviation,
manipulability, per-path robot-aware score delta, per-path Cartesian delta,
K comparison, the exact score contributions, and cumulative selected-local
score versus the recomputed full-path score.

## Score Decomposition

The established v7 `delta_score` is decomposed into its exact weighted,
normalized terms:

- Cartesian mean error
- Cartesian p95 error
- Cartesian maximum error
- acceleration
- jerk
- boundary maximum step
- boundary acceleration discontinuity
- singularity penalty

The decomposition uses the existing `ScoreWeights`, `MetricFloors`, and
`relative_delta()` implementation. Negative contributions improve the score;
positive contributions worsen it. Velocity and joint-limit metrics remain
reported physical/safety quantities, but they are not terms in the established
v7 score.

Every full-path row records the individual contributions, their sum, and the
decomposition residual. The contribution sum is asserted equal to the
unchanged v7 `delta_score` using `rtol=1e-7` and `atol=1e-9`.

Full-path reporting has two explicit modes:

**Legacy full-path score:** preserves the earlier diagnostic calculation,
including a transition from the final rollout posture back to the final
strong-prior posture. `total_robot_aware_delta_score` remains an alias of this
legacy value for file compatibility. The older
`full_path_recomputed_delta_score` and `local_vs_full_delta_score_gap` fields
also remain legacy aliases; explicit `legacy_` and `internal_` fields should
be used in new analysis.

**Internal full-path score:** evaluates the physically executed complete
trajectory without inventing a sample after its terminal point. It retains
Cartesian, derivative, singularity, joint-limit, and actual internal
continuity effects, while neutralizing only the nonexistent post-terminal
boundary terms.

Both modes use the same established weights, floors, normalization, and sign
conventions. Their decompositions are reported with `legacy_` and `internal_`
prefixes and independently checked against their corresponding score.

Each accepted rollout decision also records its local `delta_score`; fallback
records exactly zero. The path summary compares:

```text
sum_selected_local_delta_score
full_path_recomputed_delta_score
local_vs_full_delta_score_gap
```

This distinction matters because locally improving windows can interact after
stitching and produce a worse complete-trajectory score.

Terminal joint and Cartesian deviation from the strong prior are reported
separately. They are posture diagnostics, not executed discontinuities and
not independent advancement failures. Actual receding-horizon joins are
measured from consecutive executed samples using joint-step and acceleration
diagnostics; no join metric compares rollout samples to same-index prior
samples.

The multiseed summarizer writes:

- `anchored_multiseed_per_seed.csv`
- `anchored_multiseed_per_path.csv`
- `anchored_multiseed_aggregate.csv`
- `anchored_multiseed_summary.json`
- `anchored_multiseed_report.txt`

## Provisional Engineering Rule

The decision uses only ordinary paths, K=8, and all five seeds:

```text
advance_to_closed_loop =
    every seed has 100% finally safe ordinary paths
    AND mean fraction of ordinary paths with
        internal_full_path_robot_aware_delta_score < 0
        >= 12/20
    AND mean internal_full_path_robot_aware_delta_score < 0
    AND mean Cartesian mean-error delta <= 0
    AND mean segment-max correction growth slope
        <= 1e-5 rad per rollout step
    AND mean Cartesian-error-delta growth slope <= 0
    AND no maximum actual internal joint step exceeds 0.20 rad
```

This is a provisional engineering rule, not a formal statistical test. The
five seeds are repeated stochastic evaluations of the same 20 physical paths,
not 100 independent path samples.

The `1e-5 rad/rollout-step` segment-max tolerance is fixed before the full
20-path, five-seed evaluation. The older sample-level
`correction_growth_slope` remains available for backward-compatible
diagnostics but is not used for advancement: it begins with a forced zero
correction and can indicate positive growth even when actual recursively
executed segments remain bounded. The decision instead fits slopes over real
rollout steps only, without adding a synthetic initial point.

## Smoke-Test Interpretation

A two-path smoke test verifies execution and can reveal genuine behavior on
those two physical trajectories, including real safety outcomes, score
contributions, and local-versus-full disagreement. It cannot estimate
population-level acceptance, safety, or improvement rates for the fixed
20-path validation population. Only the complete five-seed ordinary K=8
summary applies the advancement rule.

The internal-versus-legacy score correction was made after that two-path
software smoke test and before the complete 20-path, five-seed scientific
evaluation. It changes no model configuration, checkpoint, sampling seed,
target scale, output alpha, candidate, selected index, fallback decision,
rollout trajectory, Cartesian metric, or hard-safety result.

## Limitations

- Teacher-forced compatibility does not prove closed-loop robustness.
- The global strong prior remains available throughout rollout.
- Tail padding is deterministic and constant after the final physical sample.
- The difficult paths are stress tests, not part of the advancement rule.
- Multi-seed variation measures sampling stability on a fixed path set.
