# SmartJoint duplicate warnings and completion watchdog

Both parts were investigated offline. No SDK import, robot connection, queue reset, motion, source CSV edit, reservation reset, or production-runner change occurred. The event-order correction from the preceding investigation remains in place.

## Part A: exact duplicate proof

All indices in this report are zero-based source indices. A CSV physical line is source index + 2. Comparison uses `struct.pack("<d", value)` for each independently parsed binary64 value, not a tolerance and not just a numerical-zero delta. Original tokens and float hexadecimal representations are retained in the machine-readable report.

Source SHA256: `3a592a1de99a63abe97182fa6dc7260dc1d3402f42685b6f522cbd574303ba9f`. There are exactly 508 rows.

| Preceding index | Current index | Preceding TouchType | Current TouchType | Binary64 equality J1-J6 | Maximum absolute joint delta |
|---:|---:|---|---|---|---:|
| 187 | 188 | Air | Air | true, true, true, true, true, true | 0.0 rad |
| 188 | 189 | Air | Air | true, true, true, true, true, true | 0.0 rad |
| 217 | 218 | Pen | Pen | true, true, true, true, true, true | 0.0 rad |
| 328 | 329 | Air | Air | true, true, true, true, true, true | 0.0 rad |
| 367 | 368 | Air | Air | true, true, true, true, true, true | 0.0 rad |
| 388 | 389 | Pen | Pen | true, true, true, true, true, true | 0.0 rad |

Exact joint vectors for both rows of every pair:

**Indices 187 -> 188 (Air / Air)**

```text
preceding_q_rad = [-2.2657988335583505, -0.2305280838018476, 2.1990331221635158, -3.1415904159640684, -0.7120355286222821, 4.0173816806359826]
current_q_rad   = [-2.2657988335583505, -0.2305280838018476, 2.1990331221635158, -3.1415904159640684, -0.7120355286222821, 4.0173816806359826]
binary64_equal_per_joint = [True, True, True, True, True, True]
maximum_abs_joint_delta_rad = 0.0
```

**Indices 188 -> 189 (Air / Air)**

```text
preceding_q_rad = [-2.2657988335583505, -0.2305280838018476, 2.1990331221635158, -3.1415904159640684, -0.7120355286222821, 4.0173816806359826]
current_q_rad   = [-2.2657988335583505, -0.2305280838018476, 2.1990331221635158, -3.1415904159640684, -0.7120355286222821, 4.0173816806359826]
binary64_equal_per_joint = [True, True, True, True, True, True]
maximum_abs_joint_delta_rad = 0.0
```

**Indices 217 -> 218 (Pen / Pen)**

```text
preceding_q_rad = [-2.26528945529952, -0.33523048184733706, 2.263406353865516, -3.141574964659362, -0.5429553761035188, 4.017877059221594]
current_q_rad   = [-2.26528945529952, -0.33523048184733706, 2.263406353865516, -3.141574964659362, -0.5429553761035188, 4.017877059221594]
binary64_equal_per_joint = [True, True, True, True, True, True]
maximum_abs_joint_delta_rad = 0.0
```

**Indices 328 -> 329 (Air / Air)**

```text
preceding_q_rad = [-2.3832365409195644, -0.19117408016286924, 2.2557552759446358, -3.1416, -0.6946631903263245, 3.8999554220587127]
current_q_rad   = [-2.3832365409195644, -0.19117408016286924, 2.2557552759446358, -3.1416, -0.6946631903263245, 3.8999554220587127]
binary64_equal_per_joint = [True, True, True, True, True, True]
maximum_abs_joint_delta_rad = 0.0
```

**Indices 367 -> 368 (Air / Air)**

```text
preceding_q_rad = [-2.3333425304040913, -0.24093295653680014, 2.1836544029181146, -3.141586348879235, -0.7170032957521287, 3.9498362745205395]
current_q_rad   = [-2.3333425304040913, -0.24093295653680014, 2.1836544029181146, -3.141586348879235, -0.7170032957521287, 3.9498362745205395]
binary64_equal_per_joint = [True, True, True, True, True, True]
maximum_abs_joint_delta_rad = 0.0
```

**Indices 388 -> 389 (Pen / Pen)**

```text
preceding_q_rad = [-2.333356809011266, -0.34382979731810753, 2.2472648162902287, -3.14158616555929, -0.5505017714257039, 3.9498199681775374]
current_q_rad   = [-2.333356809011266, -0.34382979731810753, 2.2472648162902287, -3.14158616555929, -0.5505017714257039, 3.9498199681775374]
binary64_equal_per_joint = [True, True, True, True, True, True]
maximum_abs_joint_delta_rad = 0.0
```

**Index 1 is not one of these pairs.** All six joint values differ from index 0:

```text
index_0_q_rad = [-2.227512509127758, -0.18441560441027577, 2.2651860822247576, -3.1415984105959325, -0.6919914095179537, 4.055677012796968]
index_1_q_rad = [-2.2275126200399753, -0.18441831377498627, 2.2651884412021164, -3.1415959965677875, -0.6919913089954207, 4.055677126474754]
binary64_equal_per_joint = [False, False, False, False, False, False]
signed_delta_rad = [-1.1091221718828592e-07, -2.7093647105014274e-06, 2.358977358785097e-06, 2.414028144936964e-06, 1.0052253296422009e-07, 1.1367778629534087e-07]
maximum_abs_joint_delta_rad = 2.7093647105014274e-06
```

Both rows are Air; index 0 is RECORDING_START at Timestamp 0.0 and index 1 is MOVING_FAST at Timestamp 0.05. The recorded warning was `absj#0`, local index 1, customInfo suffix k001, so it maps to source index 1 rather than to a later duplicate.

### Warning-policy decision

The observed warning does not satisfy the proposed exact-duplicate predicate. Close points are not necessarily identical points. The six exact duplicates elsewhere in the file do not explain or justify excusing this particular event.

The instruction makes implementation conditional on confirming that relationship. That condition is false for the observed event. **No production exception was implemented.** Enabling a general duplicate-only exception would also not solve index 1; that event would still fail its exact-equality condition. This report does not broaden the exception to a small numerical tolerance.

The production behavior remains: validate payload, ID, index, customInfo and SDK error; log and surface the diagnostic; do not advance execution progress or terminal evidence; keep the trial disqualified. Hard SDK/controller errors and malformed events remain fatal. Here, fail-closed means the run cannot be accepted and later trial progression is blocked; the existing passive-observation behavior for zero-error remarks has not been converted into a new immediate-stop policy.

### Implementation and tests

Production changes: **none**. New offline-only files are `nrt_v050/audit_smartjoint_duplicates_watchdog.py` and `tests/test_smartjoint_duplicates_watchdog.py`. The audit emits `audit.json` and `duplicate_pairs.csv` under `nrt_v050/evidence/smartjoint_duplicates_watchdog/`. All three execution scripts, the CSV, frozen policy, and 40 preserved session/reservation files pass unchanged-hash checks (45 protected files total).

Twelve new tests verify the six exact pairs and labels, index 1 differing on every joint, signed-zero bitwise distinction, absence of an enabled duplicate exception, continued rejection of nonduplicate/different remarks, fatal nonzero errors and invalid ID/index/customInfo, inability of a diagnostic to complete a run, acceptance of clean completion evidence, retained rejection after a warning, and the watchdog findings/settings.

The requested positive scenarios "duplicate warning -> diagnostic only without rejection" and "later completion after that warning -> accepted" belong to the conditional exception. Since its activation premise was not met and the exception was not enabled, those outcomes are not claimed or implemented. Tests explicitly protect the current disqualification instead. A later ordinary completion is valid progress evidence, but does not erase a latched policy failure.

Full offline result: **114 main-suite tests + 9 NRT offline tests = 123 passed, zero failures**. The new evidence suite has 12 tests. Test logs: `nrt_v050/evidence/smartjoint_duplicates_watchdog/main_tests.txt` and `nrt_v050/evidence/smartjoint_duplicates_watchdog/nrt_tests.txt`.

## Part B: preserved-session timing evidence

All timestamps below are local monotonic host timestamps. No controller sample/event timestamps are available. Source CSV Timestamp is neither of those clocks. The original 40.5-second span is metadata only, not an NRT duration request. Both sessions submitted all 508 commands in six batches of 100+100+100+100+100+8.

| Session suffix | START return, host ns | First observed movement, host ns | Movement onset after START return | q_m coverage after START return | q_m coverage after movement detection | Failure |
|---|---:|---:|---:|---:|---:|---|
| trial_01_20260914T114926Z_ec2b87c3 | 161154892703400 | 161155064081200 | 0.1713778 s | 120.0133021 s | 119.8419243 s | MEASURED_COMPLETION_NOT_CONFIRMED |
| trial_01_20260914T115514Z_1c2de807 | 161457025338400 | 161457205178900 | 0.1798405 s | 17.8681477 s | 17.6883072 s | event index regressed |

### Attempt ec2b87c3: the 120-second timeout

The last q_m sample was sequence 19357 at host ns 161274906005500, 120.0133021 seconds after START returned. The final operationState poll reported moving at host ns 161274768063300, 119.8753599 seconds after START returned and 0.1379422 seconds before the final q_m sample. Samples continued changing through the end of observation. These records support ongoing approach motion at timeout; they do not establish the robot state after disconnect.

The largest initial current-pose-to-first-target difference was J4: 3.6617049031619775 rad. At timeout J4 had moved 3.1774861262348297 rad toward it, approximately 86.776% of that initial displacement. The maximum remaining joint error to the first target was 0.4842187769271478 rad. This was a long transfer from the starting pose to the first stored CSV pose, not 120 seconds spent drawing the three strokes.

| Seconds after START return | Maximum measured joint error to first CSV target, rad |
|---:|---:|
| 0.0003318 | 3.66170490316198 |
| 30.0039261 | 2.84201052461488 |
| 59.9965494 | 2.05281911435039 |
| 89.9970668 | 1.30263923236681 |
| 118.9972924 | 0.512314354110767 |
| 119.9975205 | 0.48466674725411 |
| 120.0133021 | 0.484218776927148 |

The average observed J4 approach rate was 0.026476116152418 rad/s; the last-ten-second rate was 0.027651477840591 rad/s. No adjacent signed J4 sample step moved backward by more than 1e-5 rad; tiny quantization-scale reverse steps were retained, not smoothed. This is evidence of measured progress toward target 0, not proof of completed intermediate drawing commands.

Only one execution callback was saved: the nonduplicate index-1 adjacent-points warning. There was **no non-diagnostic completion/progress event** in this attempt. The old q_m metadata field reporting index 1 came from that diagnostic and must not be treated as executed progress.

No nonzero SDK-call error or hard event fault was recorded, all saved q_m records were valid, and cleanup reported success. However, the adjacent-points policy failure was already latched. **Timeout was the immediate reason for FAILED_EXECUTION, but it was not the sole obstacle to acceptance.** A longer watchdog alone would not have made this trial acceptable.

### Attempt 1c2de807: separate event-order failure

This later attempt began closer to the first CSV target: initial maximum joint difference 0.47300777421268414 rad. It ran for 17.8681477 seconds after START returned; its observed J4 average approach rate was 0.026490592261468 rad/s. The maximum residual to target 0 fell to 0.0006618045474624523 rad.

The last ordinary completion report was source index 0. The preceding warning reported index 1. That sequence caused the already-investigated runner false positive, not another 120-second timeout. The final state poll was moving; the recorded hard_event_fault was the runner ordering failure. The warning policy failure also remained. The two interrupted attempts must not be added together and described as one completed physical trajectory.

### Fixed watchdog recommendation: 600 seconds, not applied

Recommend **600 seconds** as a prospective fixed completion watchdog for a separately reviewed future run once Part A is resolved. It is an observation/failure-detection bound, not a command-duration setting. The existing code still uses **120 seconds**. No shared or per-run timeout was changed.

The evidence supports replacing a 120-second bound that expired during continuing approach motion. The following calculation makes the margin explicit but is only an empirical workload estimate, not an SDK planner model:

- Observed dominant-joint average rates in the two approach recordings were approximately 0.02648 and 0.02649 rad/s.
- Use a slightly lower illustrative rate of 0.025 rad/s for the calculation.
- Original starting-pose approach workload: max-joint displacement 3.6617049031619775 rad / 0.025 = 146.468 seconds.
- Complete unchanged CSV workload proxy: sum of all 507 adjacent maximum-joint deltas = 1.7226220819768343 rad; / 0.025 = 68.905 seconds. This includes all Air and Pen rows and retains zero-delta pairs.
- Add the existing one-second settling-window minimum: illustrative total 216.373 seconds.
- A 600-second bound leaves 383.627 seconds above that illustration, approximately 2.773 times the illustrated total.

The large margin accounts provisionally for short-segment planning, acceleration/deceleration, blend behavior, and settling that these approach-only recordings do not characterize. It does **not** guarantee completion in 600 seconds. Per-joint rates, actual controller overrides, pauses, many short segments, duplicate handling, and controller planning could make the drawing portion slower than this proxy. No complete 508-row duration is present in the evidence, so a validated worst-case duration cannot be supplied.

The frozen jointSpeed=0.05, speed=50, interior zone=1, final zone=0 policy is unchanged. The recommendation is not obtained by scaling the original 40.5 seconds or inserting host sleeps. It must be selected before the next run and remain fixed during that run; adoption is a separate prospective decision. All shorter freshness, state, SDK-error, and stationarity gates remain necessary and unchanged.

## Readiness and offline reproduction

**Another Trial 1 is not ready under the requested conditions.** The observed warning is nonduplicate and therefore remains disqualifying; the proposed exact-duplicate justification does not resolve it. The recommended 600-second watchdog has been reported but not adopted. No START command is supplied or executed.

Offline commands:

```powershell
Set-Location C:\RokaeSDK
py -3.12 nrt_v050/audit_smartjoint_duplicates_watchdog.py
py -3.12 -m unittest discover -s tests -p 'test_smartjoint_duplicates_watchdog.py' -v
```
