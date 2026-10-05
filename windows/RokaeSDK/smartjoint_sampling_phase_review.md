# SmartJoint single-owner observation review

Historical active-sampling review. The subsequent pre-START epoch fix and current Trial 6 readiness are documented in [SmartJoint pre-START sampling epochs](smartjoint_prestart_epochs_review.md).

Implemented and tested offline on 2026-09-14. No vendor robot object was constructed, no robot connection or motion was made, and no robot queue or trial reservation was reset. All 79 preserved SmartJoint session files, including the current Trial 1 reservation, retain their original bytes.

The recorded self-induced status-query gap is resolved in software. **152 offline tests passed**: 143 main tests in 57.925 seconds and 9 NRT tests in 0.089 seconds, with no failures/errors. Hardware validation of the new scheduling remains outstanding. The local Trial 1 reservation still points to failed session `trial_01_20260914T131354Z_82930f8f`; it must be archived before that trial number can be invoked again. No automatic retry is implemented.

## Recorded failure and exact attribution

`q_m` samples 6769 and 6770 were decoded at host monotonic timestamps **166208509089900** and **166208561429700 ns**. Their difference was **52,339,800 ns = 52.3398 ms**, exceeding the unchanged 50 ms limit.

| Between those samples | Duration |
|---|---:|
| `operateMode` | 46.6232 ms |
| `powerState` | 2.3895 ms |
| `operationState` | 2.5446 ms |
| Combined status calls | 51.5573 ms |
| Remaining host work and next sample | 0.7825 ms |
| Total | **52.3398 ms** |

The former shared `pump()` read a sample, drained callbacks, and periodically called `robot_state()` on the same thread. That method serially called all three status APIs. It normally ran every approximately 0.25 seconds, not literally every sampling cycle. The next sample therefore included the synchronous query time in its host observation gap and failed the gate. The logging records establish the call occupancy; they do not identify why `operateMode` itself took 46.6232 ms (network/controller/host scheduling attribution is unavailable).

The failure was 16.0042305 seconds after START returned. First measured movement above the existing threshold occurred at +0.1516366 seconds. There were 53 callbacks: 51 ordinary callbacks, from index 0 through 52, and the two accepted diagnostics at indices 1 and 37. The last callback was waypoint 52, `DRAWING_STROKE_1`. The last measured joints changed and the final status query returned `moving`. No recorded SDK error or hard event fault caused this failure; the 600-second watchdog was not reached.

The [replay fixture](tests/fixtures/smartjoint_sampling_gap/recorded_gap.json) retains the two original q records, exact query records and hashes of their source logs. The replay reproduces the old serial pump's **exact 52.3398 ms failure**. Under the new pump, those status calls are not scheduled during sampling; the same fake SDK latencies occur only after the captured final stationary window and do not create an active observation gap. The replay uses one fake robot and one calling thread, with no simulated concurrency.

## Query classification and local evidence

Classes: **A** before START; **B** repeatedly during sampling; **C** after captured stationary final-target evidence; **D** diagnostic-only. Calls with different uses receive multiple classifications.

| SDK operation | Former active use | New classification and reason |
|---|---|---|
| `operateMode` | START-boundary review, periodic `robot_state`, completion checks | **A + C.** Before START and at final confirmation, require a known mode equal to the initially checked mode. The SDK describes a current-mode query; no inspected local documentation mandates continuous polling. This was a validation check, not merely a diagnostic. |
| `powerState` | Same three locations | **A + C.** Require existing power ON before START and at confirmation. The SDK returns on/off/emergency-stop/safety-door/unknown categories. It does not document a continuous-polling requirement. |
| `operationState` | Same locations, plus idle completion checks | **A + C.** Require stationary/idle pre-start and controller idle for completion. Sampling alone detects a candidate, not controller completion. |
| `updateRobotState` | Every state-read attempt, including draining/freshness reads | **A + B.** Required to receive each state frame before decoding q_m; return validity and existing missing-state checks remain active. No separate final synchronous status query substitutes for these samples. |
| `getStateData('q_m', ..., 6)` | After each received state frame | **A + B.** Required for pre-start stationarity, motion evidence, active 50 ms gap accounting, and capture of the final stationary window. It decodes the subscribed frame. |
| `robotInfo` | None during active motion | **A.** Type/joint-count identity check at connection remains unchanged. |
| `moveStart` | One control call at the boundary, not a status query | Still exactly once after all six appends and existing authorization/integrity gates. Existing separately reviewed START-boundary accounting remains. |
| Callback snapshot/logging | Motion notifications, without SDK queries | Retained during active sampling and confirmation. Ordinary events supply progress evidence; exact approved `adjacent points` is **D**, supplies no progress or completion, and stays printed/logged. Error/schema gates always apply. |
| Cleanup unsubscribe/watcher removal/disconnect | Could run while motion was ongoing on any exit | Lifecycle operations, not diagnostic queries. Their existing order and physical consequences are unchanged. |

No additional active synchronous query was found. Calls to controller logs, jointPos, extra robotInfo, or a second robot connection are not introduced.

Local primary sources inspected:

- [`__init__.pyi`](nrt_v050/sdk/xCoreSDK_python/__init__.pyi): `operateMode` line 589, `operationState` 599, `powerState` 634, `getStateData` overloads 490–503, `updateRobotState` 1047, `setEventWatcher` 811, `disconnectFromRobot` 341.
- [`MoveExecution.pyi`](nrt_v050/sdk/xCoreSDK_python/EventInfoKey/MoveExecution.pyi): errors before/during command execution, waypoint-associated information, and close-point remarks.
- [`Safety.pyi`](nrt_v050/sdk/xCoreSDK_python/EventInfoKey/Safety.pyi): the documented safety-event key is `collided`; it does not establish comprehensive power/mode-change notification.
- [`read_robot_state_example.py`](evidence/v050/xcoresdk_python-v0.5.0__example__read_robot_state_example.py): update-before-decode and non-overwriting receive queue. It demonstrates reader/motion threads but does **not** state a general same-object thread-safety guarantee, so this patch does not adopt that architecture.
- [`move_example.py`](evidence/v050/xcoresdk_python-v0.5.0__example__move_example.py): demonstrates polling `operationState` to wait for completion, without a 50 ms observation contract. An example is not a mandate to poll power/mode continuously.

## Implemented phases and unchanged completion criteria

**Pre-start:** all identity, source hash, command/batch integrity, mode, power and q0/stationarity checks remain. The latest full `robot_state(True)` occurs in `pre_control_check` before `moveStart`. No motion command values or control order changed.

**START_BOUNDARY:** the existing check of the last pre-start and first post-start samples, SDK result, timing and event faults remains. Its mode/power query is deferred only for SmartJoint; the inherited neutral/path behavior remains unchanged. The deferral is logged as `START_STATUS_POLL_DEFERRED`.

**ACTIVE_SAMPLING:** the single owner drains events, reads q_m, and drains events. It does not poll mode, power or operation state. A runtime guard rejects accidental use of these synchronous status queries during this phase before entering the SDK. No second SDK client, concurrent SDK call, sleep, retry, speed change, or host trajectory scheduling was added.

**COMPLETION_CONFIRMATION:** first capture the existing `stationary_evidence` result and log `FINAL_STATIONARY_WINDOW_CAPTURED`. This requires observed physical movement, **at least 100 valid samples spanning at least 1 second**, every adjacent gap **<=50 ms**, per-joint stationary span **<=1e-5 rad**, and every sample's joint distance from the final command **<=1e-3 rad**. The helper and numerical criteria are unchanged.

Only after that evidence is captured does the owner call `operateMode`, `powerState`, and `operationState`. Acceptance requires unchanged known mode, power ON, controller **idle**, clean SDK results and clean event validation. Events are drained before/after confirmation and again during existing cleanup. The terminal callback remains optional; warnings never supply completion or progress. If the controller is still moving or has another unexpected state, confirmation fails closed; the runner does not resume sampling across an unobserved interval or retry confirmation automatically.

There is no post-confirmation q read, so the intentional status-query interval is not silently included in, or excused from, a purported active stationary window. The stored evidence is explicitly the window captured **before** confirmation. The former second status-query/extra-read sequence is replaced by this phase boundary, as requested; the stated measured-window-plus-idle rule is retained.

The total watchdog remains fixed at **600.0 seconds**, recorded in metadata. Confirmation also checks the deadline after status calls and cannot accept a late result. As with the prior serial design, an SDK call cannot be preempted by this Python deadline; this is not a guarantee of process exit at exactly 600 seconds.

## Power/mode visibility and safety limitations

No numerical completion or trajectory criterion was relaxed. **Continuous software visibility of power/mode was reduced**: the old periodic and post-START status checks are now pre-start/final-confirmation checks. This is an intentional observation-policy change, recorded in session metadata, and must not be described as identical continuous fault coverage.

The inspected docs do not guarantee that every power/mode transition generates a `moveExecution` error or that a transient mode change necessarily stops motion. Existing callbacks and SDK read errors remain fatal when received, but they cannot be claimed to cover all such changes. A persistent abnormal state at a qualifying endpoint is rejected by final status confirmation. If motion stops away from the endpoint, completion evidence never qualifies and the run fails at the fixed watchdog (or earlier on an actual read/event failure); this does not identify the underlying power/mode cause. A transient change that returns to normal before final confirmation can be unobserved.

No continuous-polling safety requirement was found in the inspected local sources. This implementation follows the user's requested single-owner architecture; it does not claim software polling replaces the robot's hardware safety system. Windows scheduling, SDK receive latency, callback/console work, and disk-logger pressure can still cause a real host observation gap, which remains fatal. Removing synchronous status polling resolves the demonstrated cause, not every possible timing failure.

## Gap accounting

During `ACTIVE_SAMPLING`, every new valid sample must be no more than **50,000,000 ns** after the previous decoded sample; equal to 50 ms passes, greater fails. The gate still measures host observation spacing, not a controller timestamp or guaranteed packet age. The one existing START-boundary review remains separate. Pre-start stationarity/freshness checks retain their previous behavior.

Every SmartJoint q_m row now includes `sample_timestamp_ns`, `previous_sample_timestamp_ns`, `gap_ns`, `phase`, `sampling_critical`, `sdk_call_since_previous_sample`, and `sdk_calls_since_previous_sample`. Measured request/return/duration records include intervening checked SDK calls, timeout reads, and the sample's `updateRobotState` and `getStateData(q_m)` calls. The latter are necessarily SDK calls even when there are no status queries. These fields are written before the gap can raise, so the offending sample and attribution are retained. Existing raw q values, source labels, monotonic timestamps and call logs remain.

## Disconnect and every relevant failure path

The local SDK states that `disconnectFromRobot` stops robot motion before disconnecting. Cleanup remains **stopReceiveRobotState -> setNoneEventWatcher -> disconnectFromRobot**, attempting later cleanup operations even if an earlier one fails. No explicit stop command was added or removed. A successful disconnect can therefore physically stop ongoing motion, as the latest trial observed; this is not a passive cleanup promise.

The common `finally` cleanup is reached after any caught exception once connection was attempted. After START this includes:

- a `moveStart` SDK error/exception, including uncertainty about whether motion started;
- rejected START-boundary data, displacement or event evidence;
- invalid q data/return values/timestamps, state-read exceptions, missing-state timeout, or >50 ms active observation gap;
- callback failure/overflow, SDK execution errors, malformed ID/index/customInfo/payload, disallowed remarks, duplicate events or completion regression;
- no observed motion, no qualifying endpoint/stationarity before 600 seconds, or late completion confirmation;
- unexpected mode/power/controller state or SDK error during final confirmation;
- logging/filesystem failures, other Python exceptions, or a caught KeyboardInterrupt.

Pre-start failures and successful completion also enter this same cleanup if connection was attempted. No new stop-on-fault branch was added: this documents the existing disconnect side effect. Cleanup itself can fail, and a terminated/killed process is not guaranteed to finish cleanup or retain logs. The final queued-event drain can invalidate a provisional success. All paths still preserve no automatic reset/resume/retry.

Misleading exception text was corrected in SmartJoint, neutral, and path_0003 runners; SmartJoint's preconnection banner also explicitly describes stop-on-disconnect. Historical logs were not edited. The new wording distinguishes no explicit stop command from the documented physical consequence of cleanup disconnect.

## Code, tests, and invariant evidence

- `smartjoint_csv_nrt_runner.py`: observation policy at line 44; q_m attribution at 201; status guard at 250; SmartJoint START-status deferral at 266; active pump at 272; captured-window confirmation at 280; phase transitions at 412; metadata in session/final status; truthful banner/exception text.
- `neutral_v050_nrt_runner.py`: default-preserving START-status hook at 575, sample SDK timings at 648, truthful exception text at 959. Default neutral/path polling remains unchanged.
- `path_0003_nrt_runner.py`: exception wording only at 179–181.
- `tests/test_smartjoint_csv_runner.py`: allow a supplied fake robot class and retain lifecycle records for the new tests.
- `tests/test_smartjoint_sampling_phase.py`: **13 new tests**, covering exact old failure, new active scheduling, pre-start checks, post-window blocking queries, true gaps, sample-count/duration/validity/span/proximity gates, query guard, active hard events, abnormal confirmation states, confirmation deadline, detailed attribution, cleanup consequences/text and single ownership.

The existing warning, event-order, immutable-target/policy and watchdog regression suites also pass, including exact/nonduplicate/duplicate adjacent warnings, invalid/error-bearing diagnostics, terminal-evidence separation, completion regression, all 508 targets and six batch sizes.

**Complete result: 143 + 9 = 152 passing tests, zero failures/errors.** Five changed Python files compile successfully. The independent offline CSV audit passes 508 rows and **3,048/3,048 binary64 values** unchanged. Source SHA256 remains `3a592a1de99a63abe97182fa6dc7260dc1d3402f42685b6f522cbd574303ba9f`. Controller policy bytes, command builder/validator, source parser/pin, warning/completion validator, stationary helper, shared settings, batching and 600-second constant were also compared against the pre-change baseline.

All targets remain in their original order, with batches **100+100+100+100+100+8**, **jointSpeed=0.05**, **speed=50**, **507 interior zones of 1**, and **final zone=0**. No CSV timestamps or motion timing inputs changed.

Complete artifacts: [main test log](nrt_v050/evidence/smartjoint_sampling_phase/main_tests.txt), [NRT test log](nrt_v050/evidence/smartjoint_sampling_phase/nrt_tests.txt), [CSV audit](nrt_v050/evidence/smartjoint_sampling_phase/csv_validation.json), [integrity verification](nrt_v050/evidence/smartjoint_sampling_phase/integrity_verification.json), [SmartJoint diff](nrt_v050/evidence/smartjoint_sampling_phase/smartjoint_csv_nrt_runner.py.diff), [shared-runner diff](nrt_v050/evidence/smartjoint_sampling_phase/neutral_v050_nrt_runner.py.diff), [path wording diff](nrt_v050/evidence/smartjoint_sampling_phase/path_0003_nrt_runner.py.diff).

The implementation has no known failing offline check. **Another Trial 1 invocation is still blocked by the preserved local reservation.** This investigation did not clear it or authorize/start another attempt. A START command is not supplied as ready-to-run while that reservation remains.
