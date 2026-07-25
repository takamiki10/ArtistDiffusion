# Diffusion v8.1 Anchored Recursive Rollout With History-Aware Jerk Guard

This is a separate development experiment from the frozen v8 anchored
recursive rollout. It must not overwrite or retroactively reclassify the v8
five-seed result.

## Baseline Preserved

v8.1 reuses the validated v8 machinery for:

- trained diffusion model and checkpoint;
- target scale `1.0`;
- output alpha `0.125`;
- DDIM sampling with 50 steps and `eta=0.0`;
- candidate seeds and nested `K=1,4,8`;
- condition construction;
- anchored prior construction;
- hard-safety gates;
- Cartesian improvement requirement;
- v7 compatibility gates;
- safe anchored-prior fallback;
- full-path internal score definition.

For a given rollout state, v8.1 generates the same eight diffusion candidates
as v8. The only added selection rule is the history-aware executed-prefix jerk
guard. The original v8/v7 compatibility gates remain separate from this guard
and are reported separately.

## History-Aware Jerk Guard

Before candidate selection, v8.1 constructs:

```text
last up to three already executed joint states
+
candidate execution prefix
```

It constructs the equivalent sequence for the anchored-prior fallback. Jerk is
computed with the same discrete convention used by v7 metrics:

```python
np.diff(sequence, n=3, axis=0)
```

Only jerk stencils whose newest sample belongs to the proposed execution
prefix are counted.

The reported fields are:

- `history_aware_candidate_incremental_jerk_cost`
- `history_aware_fallback_incremental_jerk_cost`
- `history_aware_incremental_jerk_delta`
- `history_aware_jerk_guard_pass`
- `rejected_only_by_history_aware_jerk_guard`

where:

```text
history_aware_incremental_jerk_delta
= candidate incremental jerk cost - fallback incremental jerk cost
```

Negative values mean the candidate has lower realized incremental jerk than
fallback.

## Candidate Eligibility

A candidate is eligible only if:

```text
existing v8 sample_is_selectable(candidate)
AND
history_aware_incremental_jerk_delta <= 1e-12
```

The tolerance `1e-12` is numerical only. If no candidate passes, v8.1 executes
the unchanged anchored-prior fallback.

Among eligible candidates, v8.1 preserves the v8 primary ordering by exact
minimum validated v7 `delta_score`. A numerical tie is defined with:

```python
np.isclose(score, minimum_score, rtol=0.0, atol=1e-12)
```

The jerk delta is only a deterministic tie-breaker among those tied candidates.
The candidate index is the final deterministic tie-breaker.

Candidate CSV field semantics:

- `compatibility_gates_pass`: original v8/v7 compatibility result.
- `v8_selectable_before_history_aware_jerk_guard`: original v8 selectable state.
- `v8_1_selectable_after_history_aware_jerk_guard`: final v8.1 eligibility.
- `selectable`: compatibility alias for final v8.1 eligibility.
- `v8_1_rejection_reasons`: v8.1-only rejection reasons, including
  `history_aware_jerk_worsening`.

The new jerk rejection reason is not appended to the established acceptance
criteria field.

Use:

```bash
--disable_history_aware_jerk_guard
```

as an explicit diagnostic mode. In that mode, v8.1 should reproduce baseline
v8 candidate seeds, candidate trajectories, selected indices, rollout
trajectories, local scores, and full-path metrics. This mode uses the frozen
v8 anchoring call signature exactly.

## Outputs

The evaluator writes the same core filenames as v8:

- `anchored_rollout_decisions.csv`
- `anchored_candidate_results.csv`
- `anchored_full_path_metrics.csv`
- `anchored_ordinary_aggregate.csv`
- `anchored_difficult_aggregate.csv`
- `anchored_combined_diagnostic_aggregate.csv`
- trajectory NPZ files and plots

Additional v8.1 path-level fields include:

- `evaluated_candidate_count_total`
- `v8_selectable_candidate_count_before_jerk_guard`
- `history_aware_jerk_rejection_count`
- `history_aware_jerk_rejection_rate`
- `history_aware_jerk_rejection_rate_all_evaluated`
- `history_aware_jerk_rejection_rate_among_v8_selectable`
- `selected_history_aware_jerk_delta_sum`
- `selected_history_aware_jerk_delta_max_including_fallback`
- `selected_history_aware_jerk_delta_max_accepted_only`
- `selected_history_aware_jerk_delta_mean_accepted_only`
- `internal_robot_score_contribution_jerk`
- `internal_full_path_robot_aware_delta_score`

The rejection-rate denominators are:

- `history_aware_jerk_rejection_rate_all_evaluated` =
  jerk rejections divided by all evaluated candidates.
- `history_aware_jerk_rejection_rate_among_v8_selectable` =
  jerk rejections divided by candidates that were selectable under original v8.
- `history_aware_jerk_rejection_rate` is a compatibility alias for the rate
  among originally v8-selectable candidates.

If there are zero originally v8-selectable candidates, the selectable-denominator
rate is reported as NaN.

Fallback contributes `0.0` to `selected_history_aware_jerk_delta_sum`. Accepted
only max/mean fields use `0.0` when no candidate is accepted.

Correction-growth slopes remain reported, but they are not used as v8.1
pass/fail criteria because the completed v8 experiment showed they did not
track internal trajectory quality.

## Paired Development Summary

The summarizer compares v8.1 against:

```text
results/diffusion_v8_anchored_recursive_multiseed
```

using the same physical path, `K`, and sampling seed. For ordinary `K=8`, it
reports paired differences in:

- internal full-path robot-aware score;
- internal jerk contribution;
- Cartesian mean-error delta;
- accepted rollout-step rate;
- fallback rate;
- maximum actual internal joint step.

Seeds `43-47` are repeated stochastic evaluations of fixed physical paths.
They are not independent physical path samples.

The paired summarizer constructs explicit keys:

```text
(sampling_seed, population, k, path_id)
```

For the decision population it asserts:

- `population = ordinary`;
- `K = 8`;
- seeds are exactly `43,44,45,46,47`;
- each seed has 20 unique physical paths;
- v8.1 has 100 unique decision keys;
- baseline v8 has 100 unique decision keys;
- the two decision key sets are exactly equal;
- 100 paired decision rows exist;
- no duplicate keys exist.

If any key is missing, duplicated, or unexpected, the summarizer raises an
error and does not issue an engineering classification.

## Development Criteria

For paired seeds `43-47`, v8.1 is classified as a successful engineering
improvement only when:

- all final trajectories remain safe;
- maximum actual internal joint step is `<= 0.20` rad;
- mean paired internal-score change is `< 0`;
- mean paired jerk-contribution change is `< 0`;
- 95th percentile internal score is lower than baseline v8;
- mean Cartesian mean-error delta remains `<= 0`.

This is a development comparison, not final confirmation. Seeds `43-47` became
development data after inspecting the v8 results. A frozen v8.1 method should
later be confirmed using fresh stochastic seeds such as `48-52`, and preferably
additional path-disjoint trajectories.
