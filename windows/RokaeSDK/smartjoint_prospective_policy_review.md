# SmartJoint prospective warning and watchdog policy

Historical warning/watchdog review. For the subsequent observation-phase change and current readiness, see [SmartJoint single-owner observation review](smartjoint_sampling_phase_review.md). The prior ready-to-retry statement below predates the later sampling-gap failure.

Both requested software changes are implemented and offline validation passed. No known software blocker remains for a supervised Trial 1. No vendor robot object was constructed, no robot connection was made, and no hardware motion or queue operation was performed. Lifecycle tests use fake SDK objects and a simulated clock. Hardware completion within 600 seconds remains unverified.

This report supersedes the prospective recommendations in the earlier duplicate/watchdog review. Historical failed-attempt logs and historical audit results remain unchanged; no failed attempt has been reclassified as successful.

## Exact production changes

Line numbers refer to the resulting files. Full before/after diffs are retained beside the test logs.

| File | Resulting lines | Change |
|---|---:|---|
| `smartjoint_csv_nrt_runner.py` | 35–44 | Fixed `COMPLETION_TIMEOUT_S = 600.0` and explicit warning-policy metadata. |
| `smartjoint_csv_nrt_runner.py` | 142–143 | Offline report records the SmartJoint timeout and policy. |
| `smartjoint_csv_nrt_runner.py` | 155–156, 164–178 | Keep prior faults fatal, preserve original payload while mapping batch indices, and classify only the exact allowed remark after inherited validation. |
| `smartjoint_csv_nrt_runner.py` | 267–270 | Record 600 seconds in session settings and `completion_watchdog_s`, alongside warning policy. |
| `smartjoint_csv_nrt_runner.py` | 290 | Operator banner reports the configured 600-second bound. |
| `smartjoint_csv_nrt_runner.py` | 347 | Total completion deadline uses the fixed 600-second constant. |
| `smartjoint_csv_nrt_runner.py` | 396–397 | Final status retains diagnostics, configured watchdog, and warning policy. |
| `neutral_v050_nrt_runner.py` | 364, 387–391, 402–407 | Add diagnostic storage and an experiment-specific remark hook after existing validation. Default neutral/path policy remains disqualifying. |
| `neutral_v050_nrt_runner.py` | 712–717 | Log and print each accepted SmartJoint diagnostic. Raw motion-event logging still precedes validation. |

Exact diffs: [SmartJoint](nrt_v050/evidence/smartjoint_prospective_policy/smartjoint_csv_nrt_runner.py.diff), [shared runner](nrt_v050/evidence/smartjoint_prospective_policy/neutral_v050_nrt_runner.py.diff).

## Warning behavior

Only the exact, case-sensitive string `adjacent points` is nonfatal, after the existing payload, command ID, batch-local index, source customInfo, callback-time, and SDK-success checks pass. The existing success schema requires integer `ec=0` and message `success`; malformed or contradictory error data remains fatal. A previously recorded hard execution fault remains fatal, and controller-state/error checks still apply independently.

There is no source-duplicate or numerical-proximity requirement. Source indices 0 and 1 are not bit-identical; their maximum joint delta is exactly `2.7093647105014274e-06` rad. The recorded index-1 payload now passes as a diagnostic and the subsequent ordinary index-0 completion is valid.

Each allowed diagnostic prints `WARNING: 'adjacent points' at source waypoint ... - diagnostic only; not execution progress.` It retains the full original payload in `motion_events.jsonl`, records command ID, batch-local index, source index and raw payload in an `ADJACENT_POINTS_DIAGNOSTIC` lifecycle record, and appears in final-status `diagnostic_warnings`.

The diagnostic leaves `last_index`, `seen`, and `terminal_ns` unchanged. It does not invalidate the trial, establish terminal callback evidence, or supply execution progress. Ordinary callbacks continue to pass the original monotonicity and duplicate checks. Every other nonempty remark, including case/whitespace variants, raises a fatal validation error. Invalid IDs, indices, customInfo, payloads, and SDK errors remain fatal.

The existing independent measured-motion/final-stationarity completion protocol is unchanged. A diagnostic supplies no completion evidence to that protocol; terminal callback tracking also remains separate.

## Fixed watchdog and preserved motion

The SmartJoint total completion watchdog is **600.0 seconds**, chosen before execution. The loop uses this constant; session metadata and final status record the same value. The independent neutral/path experiments retain their previous shared 120-second setting. No adaptive behavior or new sleeps were introduced. The deadline remains in the same position in the lifecycle, after start returns and the initial sample is read.

All 508 source vectors and all 3,048 parsed binary64 joint values match an independent CSV parse and fake-SDK command construction exactly, in original order. CSV SHA256 remains:

`3a592a1de99a63abe97182fa6dc7260dc1d3402f42685b6f522cbd574303ba9f`

The controller-policy file and `path_0003_nrt_runner.py` are byte-for-byte unchanged. Batches remain **100 + 100 + 100 + 100 + 100 + 8**; every command keeps `jointSpeed=0.05`, `speed=50`; zones remain 507 interior values of 1 and final value 0. CSV timestamps remain metadata with their original 40.5-second span, not a motion schedule or expected duration. Source checks, identity gates, stationarity checks, q_m logging, final measured stationary verification, controller commands and automatic-retry prohibition remain unchanged. Both production diffs were reviewed for this scope.

All **40 preserved session files** have the same names and hashes as the pre-change baseline. `trial_01.json` is absent; no trial reservation was created or cleared during this work.

## Offline regression results

**139 tests passed, zero failures or errors:** 130 main tests in 72.695 seconds and 9 NRT tests in 0.080 seconds. All six changed Python files also passed syntax compilation. The standalone offline CSV audit returned `OFFLINE_NUMERIC_PASS`, 508 rows and 3,048/3,048 bitwise-equal joint values.

The new `tests/test_smartjoint_prospective_policy.py` contains 16 passing tests:

1. Exact archived index-1 warning payload is diagnostic only; later archived completion 0 remains valid.
2. Nonduplicate near-point source pair is diagnostic only.
3. All six exact-duplicate pairs accept the diagnostic and subsequent ordinary progress.
4. Nonzero or malformed SDK error remains fatal.
5. Every different nonempty remark remains fatal.
6. Invalid local indices and index types remain fatal.
7. Invalid or mismatched command/customInfo remains fatal.
8. Missing fields and malformed payload values remain fatal.
9. Diagnostic at final waypoint cannot establish terminal completion.
10. Later ordinary valid completion remains accepted.
11. Completion 5 followed by completion 4 still fails, even with an intervening allowed diagnostic.
12. An allowed diagnostic cannot erase a prior hard error.
13. SmartJoint watchdog and report metadata equal 600; shared setting remains 120.
14. Independent source parse, all command targets, batch sizes, speeds and zones remain exact.
15. Fake execution extending beyond 120 seconds completes under the 600-second bound, records matching session/final metadata, and starts once.
16. Allowed diagnostic followed by controller-state fault still fails.

Existing event-order, duplicate/watchdog, and lifecycle tests were updated to assert the newly authorized policy. FIFO replay tests also verify unchanged raw-payload logging and visible warning output. The duplicate/watchdog tests continue to inspect the frozen historical audit rather than rewriting historical evidence using current code.

Complete outputs: [130 main tests](nrt_v050/evidence/smartjoint_prospective_policy/main_tests.txt), [9 NRT tests](nrt_v050/evidence/smartjoint_prospective_policy/nrt_tests.txt), [CSV validation](nrt_v050/evidence/smartjoint_prospective_policy/csv_validation.json), [integrity verification](nrt_v050/evidence/smartjoint_prospective_policy/integrity_verification.json).

## Trial 1 command — provided only, not executed

Both requested prospective software blockers are resolved. The existing operator gates remain in place. Run manually from PowerShell when ready for the supervised trial:

```powershell
Set-Location C:\RokaeSDK
py -3.12 .\smartjoint_csv_nrt_runner.py --run-trial --trial 1 --csv .\SmartJoint_Data_diffusion.csv --confirm-native-radians-seconds
```
