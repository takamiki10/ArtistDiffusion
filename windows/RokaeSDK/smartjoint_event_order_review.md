# SmartJoint trial 1: diagnostic callback order investigation

**Finding B: the runner made an unsupported event-order assumption.** A
remark-bearing callback advanced the same progress cursor used for ordinary
execution notifications. The later index-0 completion was then rejected solely
because the earlier warning mentioned index 1. The available documentation and
records do not establish backward robot execution or a controller ordering-contract
violation. The correction separates diagnostics from progress without discarding
warnings, accepting warning-bearing trials, or removing strict progress checks.

Everything in this investigation was offline: local file reads, Python-validator
replay, fake-SDK tests, syntax compilation, and report generation. No vendor SDK
was imported for this investigation. No robot was constructed/connected, no queues
were reset, no trial reservation was changed, and no motion was sent.

## Exact session and chronology

The most recent trial-1 folder with the recorded failure is:

```text
C:\RokaeSDK\nrt_v050\smartjoint_sessions\trial_01_20260914T115514Z_1c2de807
```

`final_status.json` records `FAILED_EXECUTION`, `event index regressed`, and
`hard_event_fault: true`. It records six appends, one START, 508 appended rows,
and successful cleanup calls. It does not record measured completion.

The watcher was registered for **Event.moveExecution**. The `setEventWatcher`
request/return were logged at host ns `161418171297500` / `161418171344100`;
the return had `ec=0`, `message="success"`. This event type is known from the
registered watcher, not a separate field embedded in each callback payload.

`moveStart` request/return: host ns **161456981226800 / 161457025338400**.
There are exactly **two** rows in `motion_events.jsonl`:

| Log line | Local callback timestamp, monotonic ns | Seconds after START request | Controller/event timestamp | Event type | Severity/classification | Command and batch | Index | Raw payload |
|---|---:|---:|---|---|---|---|---:|---|
| 1 | 161457025423000 | 0.0441962 | Not supplied | Event.moveExecution | Warning/diagnostic, inferred from nonempty remark | `absj#0`, batch 1, source range [0,100) | Local 1, source 1 | `{"customInfo":"sj-70b9606a6b074491b9dc25570bda56a4:k001","wayPointIndex":1,"cmdID":"absj#0","remark":"adjacent points","error":{"ec":0,"message":"success"},"reachTarget":true}` |
| 2 | 161474893439000 | 17.9122122 | Not supplied | Event.moveExecution | Informational completion report, inferred from empty remark, zero error, reachTarget=true | `absj#0`, batch 1, source range [0,100) | Local 0, source 0 | `{"customInfo":"sj-70b9606a6b074491b9dc25570bda56a4:k000","wayPointIndex":0,"cmdID":"absj#0","remark":"","error":{"ec":0,"message":"success"},"reachTarget":true}` |

The callback gap is **17.868016 seconds**. There is no explicit severity enum,
controller timestamp, or separate warning/completion event-type discriminator
in either payload. In particular, **the warning also has reachTarget=true**;
that boolean alone cannot distinguish these two observed kinds of report.
No exact callback wall-clock UTC timestamp is supplied or invented here.

These are the two records that caused the exception. The original validator
set `last_index=1` and inserted `(1, True)` into its duplicate/progress set on
line 1. On line 2, `0 >= 1` failed. The exception was logged at host ns
`161474903190400`, before cleanup. This is entirely within **one batch**; it is
not a batch-local/global-index mapping error.

`append_batches.json` maps `absj#0` through `absj#5` to [0,100), [100,200),
[200,300), [300,400), [400,500), [500,508). The callback customInfo suffixes
`k001` and `k000` agree with source indices 1 and 0. The complete source metadata
shows row/index 1 is Air / MOVING_FAST / Timestamp 0.05, and index 0 is
Air / RECORDING_START / Timestamp 0.0. CSV Timestamp is original trajectory
metadata, not a controller event timestamp.

## Measured-state and callback handling evidence

All **7,089** saved q_m records are valid. Their controller timestamp fields are
null. Host-timed samples surrounding the warning still show the initial pose;
the maximum joint residual to its indexed target is about **0.4730056 rad**.
Near the later index-0 report, the maximum residual to target 0 is about
**0.0006620 rad**. This is consistent with a diagnostic reported while the arm
was still approaching the first target, followed by a completion notification.
Host receipt timing is not a controller sampling clock; these samples are
supporting evidence, not a proof of unobserved physical behavior.

The nearest samples are preserved in `reconstruction.json`:

| Callback | q_m sample immediately before | q_m sample immediately after |
|---|---|---|
| Warning index 1 | sequence 4849, host ns 161456980997800 | sequence 4850, host ns 161457025612400 |
| Completion index 0 | sequence 7087, host ns 161474885403000 | sequence 7088, host ns 161474893486100 |

`EventInbox.callback()` timestamps and snapshots each payload into a FIFO queue.
It does not sort by waypoint. `PreparedSession.drain_events()` logs the raw event
before validating it. The main thread drains this queue; no secondary thread
performs robot calls. Nothing in these paths explains the sequence as host-side
sorting or an index conversion. The old lifecycle log surfaced the first warning
as `POLICY_VALIDATION_FAILURE`; the second event then entered the hard-failure path.

## What v0.5.0 actually documents

Primary local source:
`nrt_v050/sdk/xCoreSDK_python/EventInfoKey/MoveExecution.pyi`, lines 2–9.
Its SHA256 matches the exact Windows v0.5.0 package manifest:
`bec29c3037587ea38410f675d34aa624df271b4e47f31ffdaee71ace40965b0a`.

The documented meanings, translated from the supplied Chinese:

- `ID`: path ID corresponding to moveAppend's second parameter.
- `WaypointIndex`: the currently executing trajectory target-point index,
  starting at zero. It is an index associated with the identified command path,
  not a global callback sequence number or a timestamp.
- `ReachTarget`: whether the trajectory has reached its target point.
- `Error`: errors before or during motion-command execution.
- `Remark`: additional execution information, currently including warnings
  about target points being too close.
- `CustomInfo`: the corresponding command's user-defined information.

`nrt_v050/sdk/xCoreSDK_python/__init__.pyi`, lines 811–822, documents that
moveExecution callbacks run on the same thread and should avoid lengthy work.
**Same-thread callbacks do not imply monotonic waypoint-index delivery.** Neither
this declaration nor the inspected key definitions promise ordering across
diagnostics and completion reports. No completion-only monotonic-delivery guarantee
was found either. That is a limit of the available evidence, not a claim that
every possible callback sequence is legal.

The archived v0.5.0 example `print_move_info()` prints ID, reachTarget, index,
error and remark independently, without an ordering assertion. Its hash and
the main stub's hash also match `evidence/v050/package_manifest.json`. The
additional local C++ key/watcher declarations agree but originate from the
separate v0.5.1.rt_0 archive; they are corroboration, not substituted v0.5.0 proof.
No SDK example was executed, and no network documentation was needed.

The precise controller-internal reason for emitting a warning with reachTarget=true
ahead of index 0 is not specified by these materials. It would be unsupported to
call that warning an authoritative record of physical advancement to waypoint 1.
Consequently the subsequent index-0 report is **not evidence of regression from
waypoint 1**, and the runner's hard-fault inference was a false positive.

## Exact correction and retained gates

Only `Completion._observe()` in `neutral_v050_nrt_runner.py` changes the
validation flow. After the existing payload/schema, time boundary, ID, index,
customInfo, reachTarget-type and SDK-error checks:

1. A nonempty `remark` is appended to `policy_failures` and returned from the
   validator's diagnostic branch. The raw callback has already been logged.
   It does **not** modify `last_index`, `seen`, or `terminal_ns`.
2. Empty-remark execution reports continue through the monotonic-index,
   duplicate, reachTarget-regression and terminal-evidence checks.

The diagnostic branch does not run before error validation. A warning with a
nonzero SDK/controller error, an invalid index, an unknown command ID, malformed
fields, or mismatched customInfo is still fatal. Warning indices neither advance
nor regress authoritative progress. A final-index warning cannot supply terminal
completion evidence. Repeated diagnostics remain logged and recorded as policy
failures; they do not collide with completion duplicate tracking.

`BatchCompletion` still validates each raw cmdID and batch-local index before
normalization. Its mapping is unchanged. The q_m metadata description now says
"latest non-diagnostic reported waypoint" because its existing progress cursor
no longer follows warnings. All original source labels and raw events remain
available.

The monotonic guard has **not** been deleted. Completion 5 followed by completion
4 still raises a hard validation fault. Because delivery ordering is undocumented,
the new message describes the evidence precisely:

```text
non-diagnostic execution-progress index regressed; execution order unconfirmed
```

This remains a conservative fail-closed consistency gate, not an assertion that
callback order alone proves the mechanism moved backward. Ordinary reachTarget=false
progress reports retain the same conservative checks too. Establishing physical
backward execution would require stronger controller-side execution evidence.

The exact recorded two events replay against the saved original validator as
`last_index: -1 -> 1 -> exception`. Against the corrected validator they replay
as `last_index: -1 -> -1 -> 0`, one retained warning, no terminal evidence, and
no hard event fault. The warning still prevents `COMPLETED_ACCEPTED`.

## Preserved trajectory, policy, and lifecycle

The source SHA256 remains:

```text
3a592a1de99a63abe97182fa6dc7260dc1d3402f42685b6f522cbd574303ba9f
```

All 508 archived source rows and all 508 logged command targets still match
the current CSV at binary64 precision. The six batch spans remain unchanged;
jointSpeed=0.05, speed=50, interior zone=1, and final zone=0 are unchanged.
No timestamp-driven sleeps were added. The 120-second timeout and all other
lifecycle settings are unchanged.

AST comparisons show the entire `CsvSession` and `PreparedSession` classes,
joint serialization, command builder/validator, and policy reader are unchanged.
The intended behavioral difference is confined to event handling: this diagnostic
sequence no longer triggers premature hard-fault cleanup; passive observation can
continue under the existing warning-rejected status. No new motion API calls,
command targets, or recovery actions were introduced.

SHA checks verified **all 40 preserved session/reservation files unchanged**.
Trial 1's reservation remains absent, as it was at the start of this investigation.
The earlier failed physical attempt remains failed in its original logs; it has
not been rewritten or retroactively accepted. Code snapshots and diffs are in
`build_offline/smartjoint_event_order_baseline/`.

## Tests and reproducibility

Ten new tests in `tests/test_smartjoint_event_order.py` cover:

1. The exact archived warning(1, reachTarget=true) then completion(0) payloads.
2. Normal monotonic completions.
3. Completion 5 then completion 4 remains fatal.
4. Invalid negative/out-of-range/noninteger indices on warning/error events.
5. Nonzero controller errors remain fatal even with warning text.
6. Invalid IDs, tags and malformed diagnostic payloads remain fatal.
7. Warnings before/after progress neither advance nor regress the cursor or
   alter completion duplicate tracking; repeated warnings remain retained.
8. Final-index warnings cannot complete or make a rejected run accepted.
9. Nonterminal progress contradictions remain fatal.
10. The recorded sequence passes through the existing FIFO drain, keeps both
    raw payloads, and still prints/logs POLICY_VALIDATION_FAILURE, with no robot.

The older terminal-warning test was corrected to require an independent
non-diagnostic terminal report; a warning alone no longer counts as terminal
evidence. No SDK error/order failure test was removed.

Full offline result: **102 tests in `tests/` passed; 9 NRT offline tests passed;
111 total, no failed tests.** Syntax compilation passed. Test transcripts are
`nrt_v050/evidence/smartjoint_event_order_tests.txt` and
`nrt_v050/evidence/smartjoint_event_order_nrt_tests.txt`. PowerShell may wrap the
unittest stderr output as NativeCommandError text; both process exit codes were
0 and both unittest summaries report OK.

Offline reproduction commands only:

```powershell
Set-Location C:\RokaeSDK
py -3.12 -m unittest discover -s tests -v
py -3.12 -m unittest discover -s nrt_v050 -p 'test_nrt_offline.py' -v
py -3.12 nrt_v050/audit_smartjoint_event_order.py
```

Machine-readable reconstruction, exact raw data, old/new replay and surrounding
q_m samples are in `nrt_v050/evidence/smartjoint_event_order/reconstruction.json`.
The chronological table is also in `chronology.csv`. Exact callback fixtures
with provenance/hashes are in `tests/fixtures/smartjoint_event_order/`.

## Readiness for another trial

**This event-order false-positive defect is resolved and tested offline.** The
software can now observe this callback sequence without aborting falsely.
It is **not yet a basis for an accepted Trial 1**: "adjacent points" still rejects
the trial under the unchanged policy, and the earlier 120-second timeout remains
an independent issue. No completed 508-row physical run is established by these
logs. Further physical execution needs its own operator review; this report does
not supply or execute a robot START command.
