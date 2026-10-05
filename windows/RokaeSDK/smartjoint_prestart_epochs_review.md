# SmartJoint pre-START sampling epochs

This change is confined to SmartJoint. The shared neutral/path runners are byte-for-byte unchanged. No vendor robot object was constructed, no robot connection was made, and no controller queue or motion operation was executed during this work. Tests use recorded frames, temporary reservations and serial fake SDK/clock implementations.

## Exact failure path

The recorded attempt is `trial_06_20260915T030645Z_7c4ae2d9`. Samples 1750 and 1751 were decoded at host timestamps 172133155906100 and 172133223086800 ns: a **67.1807 ms** interval. The next `updateRobotState` call lasted only **0.0404 ms**. No synchronous status query occurred between those samples. There were zero APPEND calls and zero START calls; the recorded power/state were ON/idle. The log does not isolate which host operation caused the interval.

Before this patch, the exact path was:

1. `CsvSession.execute()` called inherited `PreparedSession.premotion_call('moveReset')`.
2. That method set `strict_window=True`, changed the phase to `POST_CALL_STATIONARITY`, and called `wait_settled()`. The flag remained true after it returned.
3. SmartJoint then constructed/validated 508 commands, hashed and wrote metadata, and entered `authorize('APPEND')`.
4. Inherited `prompt()` started its input thread and called `self.pump()` while awaiting the answer.
5. Pre-START `CsvSession.pump()` delegated to `PreparedSession.pump()`, which called `CsvSession.read_sample()` and then shared `PreparedSession.read_sample()`.
6. The shared reader computed `gap_ns = decoded - self.last_good`. Its `if (self.strict_window or self.start_attempted) and not exempt` branch raised `sampling gap >50 ms within observation/motion; validation failed`. Here `strict_window` was true, `start_attempted` false, and there was no pending exempt control boundary.

The fatal assertion is in `neutral_v050_nrt_runner.py` at lines 705–707. `settled()` only rejects candidate evidence by returning false, and `path_0003_nrt_runner.stationary_evidence()` returns None for a bad final candidate. Neither helper raised this failure. `fresh_barrier()` drains/records buffered frames but did not create a new preparation epoch. Motion-gap counters remained zero because START had not been attempted. The phase label alone did not control the hard assertion.

## New epoch model

`PRESTART_SOFTWARE` explicitly suspends candidate evidence. It clears only the in-memory stationarity candidate, not raw q logs or controller data. Hashing, command construction/validation, metadata writes, prompts, control-call intervals and append-acknowledgement metadata are recorded as suspension reasons. Available q samples still undergo validity, event and q0-drift checks; arbitrary preparation intervals do not acquire a continuous 50 ms contract.

`begin_prestart_stationarity_epoch(reference_q, reason)` starts a new fixed **10-second** acquisition deadline using the existing `settle_timeout_s`. It resets the in-memory gap baseline, verifies mode/power/idle, performs the existing fresh-state barrier, drops drained candidate frames, and seeds the candidate with the newly received sample. All raw frames remain logged. As before, this establishes host receipt freshness, not an undocumented controller timestamp or guaranteed network packet age.

Inside `PRESTART_STATIONARITY`, a valid q sample more than **50 ms** after the previous sample restarts the candidate at that new sample. Earlier candidate samples are discarded from the in-memory window. The epoch ID and deadline do not restart. A later clean window can pass; repeated gaps expire at the fixed outer deadline. Invalid q, SDK/state/event faults, backlog/freshness failures, missing state for the existing timeout, and q0/reference drift remain fatal. No controller command is retried.

Acceptance uses unchanged `n.settled()` criteria: **>=100 valid samples**, **>=1 second**, positive adjacent spacing **<=50 ms**, per-joint span **<=1e-5 rad**, and every sample within the existing **1e-5 rad/joint** pre-START q0 reference tolerance. Status is checked again and a fresh barrier applied before final acceptance; if those operations cause a >50 ms candidate gap, the same acquisition loop restarts its candidate. An initial q0 acquisition has no prior reference; subsequent epochs use that measured q0.

`START_BOUNDARY` and `ACTIVE_SAMPLING` retain the previously reviewed behavior. The reader's `start_attempted` branch still makes every >50 ms post-boundary observation gap fatal. A runtime guard prevents the pre-START suspension/helper from being used after START. The shared START-boundary review and motion baselines were not changed.

Final completion is unchanged: physical motion must have been observed; a final-target stationary window must meet >=100 samples, >=1 second, <=50 ms gaps, <=1e-5 joint span and <=1e-3 rad/joint final-target residual. Only then can blocking mode/power/controller-idle confirmation occur outside the active sampling window. The terminal callback remains optional. Warning/error/progress policy and the fixed **600-second** completion watchdog remain unchanged.

## Exactly when new evidence is required

- Initial q0 acquisition begins its own fresh epoch after state subscription.
- SETUP, APPEND and START prompts suspend the prior candidate before waiting for input. Neither typing time nor samples obtained before/during the prompt can serve as the later control window.
- Each `checked_call` for `setMotionControlMode`, `moveReset`, `moveAppend`, or `moveStart` validates its arguments, completes log flushing, then acquires a new q0 epoch immediately before entering the shared control-call gate. The existing last-sample freshness check at the actual control boundary remains in force.
- Each pre-motion call additionally reacquires a fresh q0 epoch after its return, before setup/append acknowledgement is marked complete.
- Thus the first append has fresh evidence after command construction, hashing, metadata and APPEND authorization. Every later append has fresh evidence after the preceding batch and its host work. START has fresh evidence after the START prompt and final source/command validation.

This adds pre-START observation time without altering controller motion timing, batch order or targets. Acquisition retries concern only candidate evidence; append/start/reset attempt limits remain unchanged. True drift or other hard faults still stop preparation rather than being hidden as window restarts.

## Code and logging

Only production file `smartjoint_csv_nrt_runner.py` changed:

| Resulting location | Change |
|---|---|
| observation policy near line 45 | Record opt-in pre-START epoch semantics and policy revision. |
| lines 246–248 | Epoch state. |
| line 250 `suspend_prestart_sampling` | Clear candidate and log non-monitored software interval/reason. |
| line 260 `begin_prestart_stationarity_epoch` | Fresh barrier, bounded acquisition, unchanged stationarity criteria, detailed accepted evidence. |
| line 308 `premotion_call` | SmartJoint-local pre/post-control epoch flow and unchanged completion flags. |
| line 323 `checked_call` | Fresh epoch immediately before each control API; existing identity/order/attempt guards retained. |
| line 380 `read_sample` | Local pre-START candidate-gap semantics; delegate post-START hard checks unchanged. |
| lines 398/407 | Suspend prompt and command-validation/hash work. |
| lines 467–493 | Initial q0 epoch and explicit construction, metadata and append-acknowledgement boundaries. |

Lifecycle records include `PRESTART_SAMPLING_SUSPENDED`, `PRESTART_STATIONARITY_EPOCH_BEGIN`, `PRESTART_STATIONARITY_WINDOW_RESTART`, and `PRESTART_STATIONARITY_EPOCH_ACCEPTED`. They retain epoch IDs/reasons; first/last sequences/timestamps; actual restart gap; sample count/duration/maximum gap; six joint spans; reference vector, residual and worst reference residual. Existing q_m timestamps, raw SDK-call attribution, phases and full raw event logging remain intact. `sampling_critical` continues to identify the active-motion phase; candidate phases/restarts are identified explicitly by the new lifecycle records.

The shared `neutral_v050_nrt_runner.py` and `path_0003_nrt_runner.py` were not edited. Existing neutral/path behavior remains unchanged. Disconnect behavior and truthful stop-on-disconnect text are unchanged.

## Regression coverage and verification

`tests/test_smartjoint_prestart_epochs.py` adds 11 tests:

1. Exact archived 67.1807 ms data reproduces the old shared-reader pre-START failure.
2. The same gap outside a new epoch is nonfatal.
3. The same gap inside an epoch removes the old candidate and logs a restart.
4. A restarted candidate later obtains a clean >=1-second / >=100-sample window without repeating control commands.
5. Repeated gaps expire the fixed acquisition timeout.
6. Five-second operator prompts cannot supply old samples to SETUP/APPEND/START.
7. Slow construction (67.1807 ms), hashing (80 ms), and metadata writes (100 ms) are outside the subsequent control window.
8. Every control call has fresh preceding evidence; each >50 ms fake append has a new post-call window.
9. Invalid q, q0 drift and power faults remain fatal with existing disconnect cleanup.
10. The identical 67.1807 ms gap after START still fails, and post-START suspension is prohibited.
11. Final completion independently rejects insufficient duration/count, gaps, excessive span and endpoint error.

The test helper also permits delayed prompts and retains authorization logs. The recorded fixture preserves source-log hashes. Existing full suites cover warning/event policies, controller faults, Trial 6 reservation gates, all targets/batches, control parameters, watchdog and cleanup behavior.

**All 172 offline tests passed, zero failures/errors:** 159 main tests in 111.556 seconds, 9 NRT tests in 0.200 seconds, and 4 standalone reservation tests in 0.174 seconds. The three changed Python files passed syntax compilation. Artifacts: [main tests](nrt_v050/evidence/smartjoint_prestart_epochs/main_tests.txt), [NRT tests](nrt_v050/evidence/smartjoint_prestart_epochs/nrt_tests.txt), [reservation tests](nrt_v050/evidence/smartjoint_prestart_epochs/reservation_tests.txt), [CSV audit](nrt_v050/evidence/smartjoint_prestart_epochs/csv_validation.json), [pre-archival integrity](nrt_v050/evidence/smartjoint_prestart_epochs/integrity_before_archival.json), [exact production diff](nrt_v050/evidence/smartjoint_prestart_epochs/smartjoint_csv_nrt_runner.py.diff).

Source SHA256 remains `3a592a1de99a63abe97182fa6dc7260dc1d3402f42685b6f522cbd574303ba9f`. All **508 targets / 3,048 binary64 joint values** match the independent CSV parse and command-data tests. Batches remain **100+100+100+100+100+8**, jointSpeed **0.05**, speed **50**, interior zones **1**, final zone **0**. The controller-policy file is byte-for-byte unchanged. No timestamp-driven sleeps or motion retries were introduced.

## Trial 6 readiness and authorized archival

After all tests passed and the software fix was reported ready, the local marker was rechecked. It pointed to the finished `FAILED_EXECUTION` session `trial_06_20260915T030645Z_7c4ae2d9`, with no START attempt and successful cleanup. No local Python or SmartJoint runner process was active. The marker has no stored PID and the runner has no automatic stale detection/reset mechanism; finished-session evidence plus the process check established staleness.

Only `C:\RokaeSDK\nrt_v050\smartjoint_sessions\trial_06.json` was moved, using native PowerShell `Move-Item -LiteralPath`, to `trial_06_prestart_gap_7c4ae2d9_reservation.json` in the same directory. The marker's bytes are preserved. All **345 original session files** were verified byte-for-byte, with the sole inventory change being that rename. Accepted Trial 1–5 markers and logs remain unchanged. See [archival verification](nrt_v050/evidence/smartjoint_prestart_epochs/reservation_archival.json).

No known software blocker remains for a new supervised Trial 6. Hardware completion is not guaranteed by offline testing. The operator gates, current-pose clearance review and required power/enabling state still apply. The following command is provided only and was **not executed**:

```powershell
Set-Location C:\RokaeSDK
py -3.12 .\smartjoint_csv_nrt_runner.py --run-trial --trial 6 --csv .\SmartJoint_Data_diffusion.csv --confirm-native-radians-seconds
```

Use the newly displayed confirmation phrases/hashes exactly. Code and observation-policy metadata changed, so earlier confirmation hashes are not reusable.
