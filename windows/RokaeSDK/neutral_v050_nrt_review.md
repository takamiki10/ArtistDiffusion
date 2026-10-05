# Corrected neutral NRT repeat — operator review

**READY FOR ONE MORE NEUTRAL HARDWARE TEST, under operator supervision and the
runtime checks below.** The runner is enabled through `--run-neutral` and
independent PRECONNECTION, SETUP, APPEND and START authorizations. During this
coding task only syntax and fake-SDK tests were run: no vendor robot object,
connection, setup probe or physical motion was executed.

**The NRT mode/queue-reset prerequisite is resolved. Do not require another
standalone setup probe.** The unconditional mode-evidence blocker is removed.
Each neutral session performs the validated setup on its own single robot
object and single connection, after measured q0 acquisition and explicit
SETUP authorization. This does not invent a readable NRT mode state or assume
that settings survive disconnect/reconnection.

## Established live evidence — 2026-09-13

The operator's live no-motion setup report establishes:

| Item | Observed result |
|---|---|
| robotInfo.type / joint_num | `XMC7-R850-W7G3B4C-S5` / 6 |
| Previously reported controller version | `2.3.1.1.C89.20250306` |
| powerState / operateMode / operationState | on / manual / idle |
| `setMotionControlMode(MotionControlMode.NrtCommandMode, ec)` | SUCCESS |
| `moveReset(ec)` | SUCCESS |
| Successful error dictionary | `{"ec": 0, "message": "success"}` |
| Reset blocking duration | Approximately 64 ms |
| First valid measured q after reset | Effectively unchanged from q0, within normal quantization/noise |

The standalone probe's FAILED status came from applying its strict 50 ms
host-sampling-gap rule across the synchronous reset call, not a failed setter
or reset. The runner now distinguishes that known pre-motion blind interval
from inadequate logging during possible motion. No missing samples are
fabricated, original probe records are unchanged, and the coding agent did
not repeat the hardware test. The earlier import, data round-trip, passive
connection, 1 Hz smoke and 8 ms/3,783-record characterization remain valid
background evidence. No physical R/D result or accepted neutral trajectory is claimed.

## First live neutral result: FAILED / NOT ACCEPTED

The operator reports that the first neutral run successfully appended all 100
commands in one moveAppend call, returned **cmdID `absj#0`**, and successfully
called moveStart. moveExecution callbacks worked. Events at waypoint indices
**1 and 2** contained `remark="adjacent points"`,
`error={"ec": 0, "message": "success"}`, `reachTarget=true`, and correctly
matching customInfo. These observations confirm the event/error schema and
customInfo for those events, not terminal index-99 semantics or full coverage.

The former any-remark exception path aborted observation and disconnected.
Only one RUNNING q_m row was captured and no terminal index 99 event was
observed. **This neutral validation failed and must not be called accepted.**
No R/D motion is authorized. Original live records are preserved; the coding
agent did not replay the test.

The following changes are prospective, before the next neutral test: replace
only the neutral profile with the asymmetric triangle below, and retain passive observation after
valid zero-error warning events while permanently rejecting that session.
The endpoint flattening of sin^4 produced effectively coincident targets and
triggered the observed warnings. This run does not establish that zone=1 itself
is invalid. The frozen R/D trajectories, manifest and controller policy remain
unchanged; **"adjacent points" is not whitelisted**.

## Second live neutral result and prospective START-boundary revision

Session `nrt_v050/neutral_sessions/20260913T130713Z_ff139818` on 2026-09-13
recorded successful mode setup, reset, append and START. moveStart took
56.0026 ms (56,002,600 ns); the cross-call q gap was 69.2786 ms (69,278,600 ns).
Pre/post J6 were exactly equal at 3.8982042293955788 rad. Other differences were
approximately one q_m quantization increment (~2.4e-7 rad); the post-start
sample was still within 1e-5 rad of reviewed q0. The success ec was exactly
`{"ec": 0, "message": "success"}`. Policy rejection and hard event flags were false.
The old rule rejected this first post-start sample before normal event draining.
The empty motion_events.jsonl is NOT evidence that the controller sent no events.
These historical records and their FAILED_EXECUTION result remain unchanged.

The physical metric is geometric, not host-time trajectory dynamics. Conditional
stationary START-boundary handling below is adopted prospectively before one
final supervised neutral hardware validation. No hardware was run during this
coding revision; this does not retroactively accept the previous session.

## Exact environment and local APIs

Use **Python 3.12.10 x64** and only `C:\RokaeSDK\nrt_v050\sdk`. The explicit
local PYD loader checks interpreter version/architecture, rejects an already
loaded SDK, and checks these SHA-256 values. It retains the DLL search handle
for the session; no v0.5.1 or newer SDK is used.

- `xCoreSDK.dll`: `7b72eaf137121036b6fcc7bf585d71331452262cea6f6d4082b754326d794c02`
- `xCoreSDK_python.cp312-win_amd64.pyd`: `0bef1547fc74617c8835b0bf7f1dd13bc175612fa7a5ad7827c51371f7426455`

Require the exact model code and six joints. Save and hash actual robot ID,
type and version. The previously reported version is provenance, not a version
pin. Power must already be ON; operating mode must be known manual/automatic
and remain unchanged; operation must be idle before motion and at acceptance.
No power setter or setOperateMode call exists.

The local v0.5.0 Windows stubs and matching examples establish the following
signatures. Runtime class docstrings were inspected offline in the earlier
revision; see `nrt_v050/evidence/neutral_runtime_api_inspection.json`. Line
references below are to `nrt_v050/sdk/xCoreSDK_python/__init__.pyi`.

| API | Exact relevant signature / source |
|---|---|
| Construction | `xMateRobot()` or `(remoteIP: str, localIP: str='')`; line 4897 |
| Connect | `connectToRobot(remoteIP: str, localIP: str='') -> None`; alternate `connectToRobot(ec: dict)`; lines 4105–4125 |
| Disconnect | `disconnectFromRobot(ec: dict) -> None`; documents stopping motion before disconnect; line 341 |
| Identity/states | `robotInfo(ec) -> Info`, `powerState(ec) -> PowerState`, `operateMode(ec) -> OperateMode`, `operationState(ec) -> OperationState`; lines 710, 634, 589, 599 |
| NRT setup | `setMotionControlMode(mode: MotionControlMode, ec: dict) -> None`; corresponding mode required before motion; lines 835–845; NrtCommandMode enum at 2380–2424 |
| Reset | `moveReset(ec: dict) -> None`; clears submitted commands/execution info; initialization documents an implicit reset; lines 572–580 |
| Subscribe | `startReceiveRobotState(interval: timedelta, fields: list[str]) -> None` (stub args arg0/arg1); 8 ms supported, first-frame timeout 3 s, fields <=1024 bytes; 4166–4176 |
| Update/q | `updateRobotState(timeout: timedelta) -> int`, then `getStateData('q_m', PyTypeVectorDouble, 6) -> int`; `.content() -> list[float]`; lines 1047, 499, 3900 onward |
| Unsubscribe | `stopReceiveRobotState() -> None`; line 1013 |
| Joint data | `JointPosition(joints: list[float])`, radians; .joints and .external; lines 1819–1868 |
| Command | `MoveAbsJCommand(target: JointPosition, speed: int=-1, zone: int=-1)` or list target; nonnegative .jointSpeed overrides speed-derived joint speed; lines 2425–2463 |
| Common fields | .speed, .zone, .customInfo; NrtCommand lines 2768–2809 |
| Append | `moveAppend(cmds: list[MoveAbsJCommand], cmdID: PyString, ec: dict) -> None`; homogeneous list, line 534 and matching move example 104–116; ID via cmdID.content() |
| Start | `moveStart(ec: dict) -> None`, starts OR resumes; line 582; one attempt only |
| Events | `setEventWatcher(Event.moveExecution, callback: Callable[[dict], None], ec: dict)` and `setNoneEventWatcher(Event.moveExecution, ec)`; lines 811/846; move example 37–38, 55–57 |
| Stop/pause | `stop(ec)` is planned Stop2 without power-off; `pause(ec)` equivalent; lines 1003/609. Neither is called. |

`RtSupportedFields.pyi` identifies measured arm q as q_m in radians, with no
controller sample timestamp or NRT-mode getter. The matching state example
lines 35–37 states queued frames do not overwrite older frames: the runner
drains/logs backlog before requiring a new receipt. The local homogeneous
append overload does not prove numeric capacity 100. The separate planning
cache setter's documented 1–300 range is not that proof; cache size is unchanged.

## Launch and authorization stages

Run deliberately in PowerShell:

```powershell
& "C:\Users\tgm10\AppData\Local\Programs\Python\Python312\python.exe" -I "C:\RokaeSDK\neutral_v050_nrt_runner.py" --run-neutral
```

Import, no flag, and help cannot create a robot or connect. With the flag,
pinned SDK import precedes the connection gate; the robot itself is constructed
only after the preconnection phrase. There is no automatic start on launch.

1. **PRECONNECTION:** stationary arm, power already ON, other motion owners
   stopped, RCI OFF, no pending independent movement; initialization reset and
   disconnect stop reviewed. Confirm exactly:
   `CONNECT FOR SUPERVISED NEUTRAL VALIDATION`.
2. **CONNECTING / ACQUIRING_Q0:** one default xMateRobot, one connection to
   `192.168.2.160`, local `192.168.2.100`. Verify robot/state, register watcher,
   subscribe q_m at 8 ms. Establish >=1 second / >=100 stationary samples.
   q0 is the last raw measured vector, not an average or prerecorded pose.
   Display radians/degrees and retain the source row.
3. **SETUP:** display both setup operations and require
   `SETUP NEUTRAL <setup-SHA256>`. This hash binds q0 bits, identity,
   source/SDK hashes, settings, session ID and the two setup actions.
4. Call setMotionControlMode(NrtCommandMode) once; require exact success and a
   NEW fresh stationary window. Then call moveReset once; require success and
   another NEW fresh stationary window. Failure prevents generating/appending
   the motion batch. No retries or separate standalone probe.
5. **GENERATE / VALIDATE:** only after post-reset observation, ask for joint
   **1–6**, direction **+ or -**, and degrees **>0 and <=0.25**; no defaults.
   Print physical-clearance warning. Generate/validate 100 commands, save
   artifacts, and print q0, selection, amplitude, count, policy and session hash.
6. **APPEND:** require `APPEND NEUTRAL <session-SHA256>`. Revalidate unchanged
   batch/policy and q/state/logger health plus completed setup. Call moveAppend
   once with all 100 commands. No split/truncate/retry/cache change. Retain ec,
   ID, call duration and pre/post q; a NEW post-append stationary window must pass.
7. **READY:** shown only after successful append, valid acknowledged cmdID and
   post-append observation. No moveStart yet. The operator may cancel here;
   cancellation does not clear the queue or authorize a later resume/start.
8. **START:** require `START NEUTRAL <same-session-SHA256>`. APPEND cannot
   authorize START. Repeat batch/state/logger checks. Initialize completion and
   mark start attempted immediately before the one moveStart call: motion may
   begin before its return. No repeat or recovery call is allowed.
9. **RUNNING / COMPLETION:** preserve the START sampling interval; keep q and
   event logging through terminal evidence and fresh post-motion settling.
   Invalid state/event/error/gap fails validation without policy adjustment.
10. Independently attempt unsubscribe, watcher cancellation and disconnect.
    Persist final result/errors; physically observe the arm after disconnect.

Phrases are case/whitespace exact. SETUP has a pre-generation hash; APPEND and
START share the later full session/batch hash. Prompts time out after 300 s;
EOF, cancellation and exceptions fail closed. Logging continues during SETUP,
joint selection, APPEND and START prompts. All authorizations are timestamped.

## Pre-motion gaps versus START/motion gaps

One main thread serializes SDK calls and disk writes, following the validated
passive capture design: updateRobotState then getStateData; retain raw q,
byte count, return code, validity, sequence and perf_counter_ns read-start,
read-end and decode-end. No controller timestamp is invented (null field).
A console-input daemon never calls the SDK. Logs are flushed and fsync barriers
protect artifacts/authorizations. No second passive client is introduced.

**Only setMotionControlMode, moveReset and moveAppend get the pre-motion
cross-call exemption**, and only before any start attempt. For each, preserve
last q before call, request/return times and duration, first valid q after call,
cross-call host gap and the exemption flag. Write an initial boundary even if
no later sample is available (post data stays null); append its completed
counterpart on receipt. Missing time is never represented as sampled.

After each successful call, log/drain buffered SDK frames, obtain a fresh q,
and seed a NEW in-memory observation window. No raw rows are deleted. Require
>=1.0 second, >=100 valid samples, internal adjacent gaps <=50 ms, per-joint
span <=1e-5 rad, every observed pre-start q within 1e-5 rad of reviewed q0,
power still ON, operating mode unchanged and operation idle. This includes
checking the first post-call sample for displacement. A >50 ms gap INSIDE the
new window fails immediately rather than aging out. Thus a 64 ms reset call
does not by itself invalidate the subsequent stationary observation.

**moveStart has one conditional boundary rule, not a global gap exemption.**
Retain the last valid pre-start q, request_ns, return_ns, duration_ns, first
post-start q, cross_call_gap_ns, every joint's pre/post delta from reviewed q0,
and pre-to-post delta. Persist accepted/rejected and its exact reason in a
`start_boundary_review` lifecycle record and final_status.json. Missing post
samples remain explicitly null; invalid samples are preserved and rejected.

A boundary <=50 ms uses the ordinary gap rule. A boundary >50 ms may be
accepted for GEOMETRIC validation only when START returned exact success,
the first post-start sample is valid, every pre AND post joint is within
1e-5 rad of q0, no hard event fault has been observed, power remains ON and
operate mode remains unchanged. Queued events are drained and validated before
acceptance; robot state is queried serially on the existing SDK thread.
Any post-start joint displaced >1e-5 rad rejects with
`START_BOUNDARY_MOTION_UNOBSERVED`. A pre-start violation, invalid sample,
START error, hard event or changed state also rejects with the recorded reason.

An accepted long boundary is tagged
`START_BOUNDARY_STATIONARY_UNOBSERVED_INTERVAL`, with
`valid_for_geometric_analysis=true` and `valid_for_timing_analysis=false`.
These boundary flags do not assert acceptance of the whole trajectory.
The FIRST post-start sample seeds the observable motion window; all later
adjacent gaps must be <=50 ms, through terminal evidence and settling. No
further START exemption exists. Raw samples are never removed from the log.
The maximum raw start/motion gap and maximum post-boundary gap are separate.

A subsequently observed selected-joint signed displacement >=1e-4 rad from
q0 is mandatory. The boundary sample itself cannot satisfy it. Persist the
first qualifying sequence, timestamp and signed displacement, and the maximum
observed signed excursion with its sequence and timestamp. This is an
observability threshold, not a tracking target. No qualifying observation
fails with `NO_OBSERVED_NEUTRAL_MOTION`, even if terminal/settling evidence exists.
At 0.1 degrees it is much smaller than the ~0.00174533 rad commanded excursion.

COMPLETED_ACCEPTED additionally requires all existing event, terminal, idle
and final q0 settling conditions, with no disallowed remark/error. A zero-error
nonempty remark still latches rejection while passive observation continues;
a later hard fault still yields FAILED_EXECUTION. Final evidence separately
reports geometric observability and start timing observability. Accepted long
boundaries explicitly fail start timing observability. No run here certifies
execution timing, velocity, acceleration, jerk or Section-V 10-second timing-law
reproduction; `valid_for_timing_analysis` remains false. Host sample timestamps
are observation diagnostics, not verified controller execution timestamps.

Invalid q fails after preservation. Timeouts are logged; >1 second without
valid data fails, and a subsequent valid row exposes any >50 ms window/motion
gap. Disk/API/event faults never cause hidden retries. No application can
guarantee final persistence on a failed disk or abrupt process termination.

## Exact neutral targets, policy and hashes

For i=0..99, sign s and selected one-based joint j:

```text
A = radians(excursion_degrees)
d[i] = A*i/49          for i=0..49
d[i] = A*(99-i)/50     for i=50..99
d[49] = A             explicitly, avoiding binary64 multiply/divide round trip
Q[i,j-1] = q0[j-1] + s*d[i]
Q[0] = Q[99] = bit-for-bit copies of q0
Q[i,k] = q0[k]         for every unselected joint
```

The smooth symmetric sin^4 profile caused controller "adjacent points" warnings
near the endpoints. The subsequent symmetric sin profile improved endpoint
separation but, because N=100 is even, produced an exact duplicate central pair
at indices 49-50. Neither symmetric profile is used by the current generator.
The asymmetric triangular neutral profile provides a deterministic 100-command
out-and-back test with a unique peak at index 49 and positive adjacent-command
spacing floor. It is intentionally a NEUTRAL validation probe, not an R/D
scientific trajectory or a controller time law. Its peak is not smooth.

All 600 target values are binary64. No clipping, wrapping, internal rounding,
IK, R/D input or post-generation smoothing is applied. The displacement profile
has exact binary64 A at index 49. Adding it to a nonzero measured q0 can round
the resulting target: exact mathematical subtraction need not equal A. The
existing exact rational bound check remains enforced; a target rounded beyond
A fails closed before APPEND rather than being clipped or granted a tolerance.
Strict increase through 49 and strict decrease through 99 are also enforced
on actual targets, rejecting collapsed adjacent pairs or an unrepresentably
small excursion. The operator-selected excursion cap remains <=0.25 degrees.

The runner prints and saves `neutral_spacing.json` for actual q0, selected
joint/sign and excursion, with all 99 adjacent differences, minimum/maximum,
peak and zero-gap pairs in radians/degrees. Session hashes include diagnostics;
command snapshots preserve hex-float targets. No diagnostic modifies targets.

At **A=0.1 degrees**, q0=0, positive joint 1, actual binary64 results are:

| Quantity | Radians | Degrees |
|---|---|---|
| Minimum adjacent spacing | 3.490658503988646e-05 | 0.0019999999999999922 |
| Maximum adjacent spacing | 3.561896432641501e-05 | 0.0020408163265306194 |
| Unique peak, index 49 | 0.0017453292519943296 | 0.1 |

Index 50 is approximately 0.098 degrees. Outward spacing is approximately
0.1/49 degrees; peak-to-50 and return spacing are approximately 0.002 degrees.
There are **zero consecutive duplicate vectors and zero zero-spacing pairs**.
`nrt_v050/evidence/neutral_triangle_validation.json` preserves all 100 vectors
and all 99 spacings, with exact binary64 hex encodings. Historical sine audit
JSON remains unchanged; the offline audit helper now labels the current neutral
comparison `asymmetric_triangle`. Actual-q0 session diagnostics remain authoritative.
Positive spacing does not establish controller acceptance; nonempty remarks
still latch rejection while passive observation continues.

Exactly 100 distinct homogeneous MoveAbsJCommand objects use jointSpeed=0.05,
speed=50, zone=1 at indices 0–98 and zone=0 at 99. Stable customInfo is
`neutral-<UUID>:k000` through `:k099`. Check JointPosition and command target
bitwise round trips, six finite joints, no external axes, type/count,
policy fields and tags; revalidate after APPEND/START authorization.

The frozen policy is unchanged and pinned to SHA-256
`0d465d14bbd750befbae60902bcb8d9bdf4b8a2d2b3b04f418e7a336b831ffbb`.
Historical policy flags are copied unchanged, not used as authorization.
q0/waypoints are hashed as little-endian binary64 bytes; commands as canonical
JSON including hex-float targets and policy/tags. Session hashing includes
these digests, setup hash/evidence, robot identity, source/SDK hashes,
addresses, selected joint/sign/amplitude and all acceptance settings.

## Error/event interpretation and completion

Every ec starts as a new dict. Preserve standard ec snapshot plus full repr,
return snapshot/repr, request/return times, duration and exceptions before
interpretation. Accept only the exact observed success dictionary
`{"ec": 0, "message": "success"}`, with integer ec, string message and no
extra keys. Void returns must be None. Other representations/codes fail;
there is no truthiness fallback or silent normalization/relaxation.

The watcher callback timestamps and snapshots the payload into a bounded
4,096-event in-process queue, without disk I/O or robot calls. Main-thread
processing preserves then validates events. Overflow/copy failure is a fault.
Primitive unknown fields are retained; opaque types are marked. The observed
event error dictionary is now confirmed for indices 1 and 2, while terminal
semantics/full coverage remain validation targets. PyErrorCode.value()
and .message() may be snapshotted; .get() is unused (earlier offline inspection
found its std::error_code return conversion unsupported). A wrapper/value-shaped
error is not silently converted to the observed ec schema; contradictory
event assumptions fail the run, with evidence retained.

The local EventInfoKey/MoveExecution stub documents string cmdID, zero-based
wayPointIndex, boolean reachTarget, error, remark and customInfo. Every event
is saved before validation. cmdID, index type/range, customInfo, error schema,
ordering and reachTarget remain checked. Malformed, foreign/old-ID, duplicate
index/reach, regressing or nonzero-error events latch a HARD event fault and
use the existing failure/cleanup path. A false-to-true pair is allowed.

For a fully valid event with zero error but ANY nonempty string remark, latch
**POLICY_VALIDATION_FAILURE**. Save the exact remark, host timestamp, cmdID and
affected index in lifecycle.jsonl and final policy_failures. Do not reinterpret
the warning as success, issue new motion/control calls, change targets/policy,
or immediately disconnect merely because of the remark. Continue only passive
q_m/event observation and read-only state checks while the controller remains
normal (same operating mode, power ON, operation idle/moving), logger/events
remain healthy and timeout/gap bounds hold. The already-started batch may reach
terminal/idle naturally. A later hard fault or timeout still aborts immediately.
Only the usual cleanup occurs at completion or a hard failure; there is no
stop/pause/reset/recovery response to the warning.

After the terminal event require >=1 second / >=100 fresh valid stationary
samples, adjacent gaps <=50 ms, span <=1e-5 rad, all q within 1e-5 rad of q0,
then idle state and fresh q/event checks. Start returning alone is not
completion. Completion timeout is 120 s; fresh-observation timeout 10 s,
neither a prescribed trajectory duration. Completion requires successful
append/start, matching acknowledged cmdID, terminal index 99/reachTarget=True,
matching customInfo, then the settling/idle checks above. Results are distinct:

- **COMPLETED_ACCEPTED:** terminal and settling observed, no disallowed remarks,
  no errors, and successful cleanup. CLI exit code 0.
- **COMPLETED_REJECTED:** terminal and settling observed but at least one
  non-error remark occurred. CLI exit code 1; policy failure is latched forever.
- **FAILED_EXECUTION:** SDK/nonzero execution error, malformed event, logger or
  state failure, timeout, unacceptable gap, cancellation or cleanup failure.
  CLI exit code 1. Any earlier policy-rejection evidence is retained as well.

Late queued remarks are also considered before final status is persisted.
**COMPLETED_REJECTED never authorizes R/D motion.** All final records explicitly
set rd_motion_authorized=false; even an accepted neutral result is not an
automatic launcher/authorization for an R/D experiment.

**VALIDATION TARGETS, not prerequisites requiring another experiment:**
repeat one-call 100-command acceptance (first append succeeded);
callback cadence/interior coverage, terminal semantics and terminal customInfo;
zone-1/too-close-target behavior; physical realization of the small excursion;
moveStart blocking duration; measured-state logging quality during motion.
An event-plus-settling pass does not prove every waypoint was physically
visited. Inspect raw q excursion and coverage before claiming realization or
considering R/D work. No outcome-driven policy tuning is automatic.

## Artifacts, cleanup and physical checks

Unique `nrt_v050/neutral_sessions/` folders contain session.json, setup.json,
robot_identity.json, commands.json, waypoints.json, exact waypoints.f64le,
neutral_spacing.json,
byte-for-byte controller_policy.json, q_m.jsonl, motion_events.jsonl,
calls.jsonl, lifecycle.jsonl, authorizations.jsonl, exceptions.jsonl and
final_status.json. Early failures retain only data actually acquired. Final
records include attempt counts, post-setup/post-append verification, maximum
motion gap, terminal evidence, q count, exact policy remarks, policy/hard-event
latches, R/D authorization=false and cleanup errors.

Cleanup independently attempts state unsubscribe, watcher cancellation and
disconnect. No stop/pause/reset, power/operating-mode/safety/collision/tool/
limit/cache change or recovery action is added to cleanup. There is no
automatic fault-stop handler. SDK disconnect itself documents a stop and
initialization itself a reset; these acknowledged library effects are not
described as a validated application safety system. No queue-retention
guarantee is inferred for an appended-but-cancelled session.

**Before START:** visually verify clearance for the selected joint and sign
at the current pose, including arm, tool, payload, workpiece and equipment.
Confirm the arm stayed stationary through SETUP and APPEND, check the printed
joint/sign/excursion/hash, and ensure no other motion owner is active.
The <=0.25 degree bound is **not a safety guarantee**. Physical E-stop and the
normal supervised safety procedure must remain immediately available. Use
that procedure for unexpected movement; do not rely on Python cleanup as a
safety controller. Watch the arm through and after disconnect.

There is **no known unresolved NRT setup prerequisite** and no additional
standalone probe is required. Unmet physical supervision/clearance conditions,
runtime hard faults or contradictory controller/event structure must abort the
attempt; non-error remarks reject acceptance but allow bounded passive
observation as specified above. Neither authorizes retries or policy tuning.

## Frozen path_0003 R/D adjacent-spacing audit (offline only)

`nrt_v050/adjacent_spacing_audit.py` verifies prepared batch hashes against the
frozen experiment manifest, reads the exact 100×6 targets, and reports all 99
six-dimensional Euclidean adjacent distances. It does not load a vendor SDK,
write a trajectory, or use R/D hardware outcomes (none exist). R is frozen
path_0003_A, D is frozen path_0003_B; their five planned trial numbers are
recorded in the evidence. No threshold/policy is selected from this diagnostic.

Results: [`adjacent_spacing_audit.json`](nrt_v050/evidence/adjacent_spacing_audit.json).

| Frozen batch | Minimum L2 (rad) | Median L2 (rad) | Maximum L2 (rad) | Exact consecutive duplicates |
|---|---:|---:|---:|---|
| R | 0.008231748964741313 | 0.0555055439259477 | 0.13544148516032825 | None |
| D | 0.008191697866519312 | 0.055504751356752195 | 0.13525692847574916 | None |

The smallest five pairs, in ascending distance order, are identical for R/D:
**[54,55], [82,83], [66,67], [10,11], [67,68]**, zero-based. Individual distances
and all 99 entries are in the JSON. These submitted joint-space distances do
not prove physical clearance, blending behavior or controller acceptance.
Neutral old/new diagnostics (including maxima and peaks) are also in that file.

Reproduce offline: `python -m nrt_v050.adjacent_spacing_audit` from C:\RokaeSDK.

## Offline verification of this revision

**45 offline tests pass**: 27 data/validation/audit tests plus 18 fake lifecycle/boundary
tests. Coverage includes exact setup/reset/append/start order and one-call
counts, 100 commands/frozen policy, independent authorizations, post-setup
failure inhibiting append, recorded 64 ms pre-motion intervals, >50 ms fresh
window failure, conditional stationary START boundaries, strictly enforced later motion gaps, errors without retries and no
forbidden setters or post-start setup/reset. New checks cover the asymmetric triangular formula,
exact endpoints, strict monotonicity, unique exact profile peak, no center or
other consecutive duplicates, the 0.002-degree spacing floor, and conservative
rejection of binary64 bound violations,
frozen audit fixtures, zero-error remark latching with continued >=1-second
post-terminal logging, COMPLETED_REJECTED versus clean COMPLETED_ACCEPTED,
and later nonzero error/state fault becoming FAILED_EXECUTION. New START tests cover
geometry-only acceptance, displaced/invalid/pre-q0 boundary rejection, power/mode
changes, failed START and hard events, mandatory signed motion, no-motion failure,
normal short boundaries, and mandatory terminal/settling evidence. No vendor SDK import, physical
robot construction or connection occurs. Syntax compilation also passes.

```powershell
python -m py_compile neutral_v050_nrt_runner.py tests/test_neutral_v050_nrt.py tests/test_neutral_v050_lifecycle.py
python -m unittest discover -s tests -p "test_neutral_v050*.py" -v
```

Earlier evidence JSON files identify older source revisions and remain
historical records; they are not rewritten to claim this revision has passed
a hardware trial.


## Frozen neutral-derived physical completion amendment, before R/D

The current protocol for the actual geometric experiment is implemented in
`C:\RokaeSDK\path_0003_nrt_runner.py`; no additional neutral motion is needed.
The previous neutral executable and historical session dispositions remain
unchanged. The physical completion amendment is frozen now, before any R/D
hardware result was inspected. It must not be tuned using R/D outcomes.

Read-only inspection of neutral session `20260913T131737Z_4f690650` passes the
amended geometric rule. The final 127-sample window spans 1.007836 s, with
maximum per-joint span 9.58737992440284e-7 rad, maximum internal gap 8.8656 ms,
and maximum final residual 0.0006890929320575445 rad. The entire post-boundary
record has maximum gap 15.632 ms, actual measured motion was observed, controller
idle was repeatedly recorded, and no motion SDK error or disallowed event was
found. The conditional stationary START boundary was accepted for geometry.
Raw residuals are preserved in `nrt_v050/evidence/neutral_measured_completion_report.json`.

**Correction to the operator's account:** the saved file has 100 valid events,
including waypoint 99 reachTarget=true, and terminal_event_ns=92579046191300.
Thus terminal evidence itself WAS satisfied in this session. The old combined
terminal-plus-strict-q0-settling criterion was not satisfied. We cannot describe
the saved failure as proof of a missing terminal callback. Removing mandatory
terminal callbacks is a prospective protocol decision, tested with absent
callbacks offline. Original logs and FAILED_EXECUTION remain unchanged.

The R/D runner requires successful APPEND and START, the existing conditional
START boundary, actual post-boundary movement of >=1e-4 rad on at least one
joint, and strictly <=50 ms gaps thereafter. Completion requires idle plus
>=100 valid samples spanning >=1 second, per-joint span <=1e-5 rad, adjacent
gaps <=50 ms, and every sample in that final window within 1e-3 rad per joint
of the final frozen command. The last measured residual and maximum residual
throughout the stationary window are saved without correction. No interpolation
is performed. Terminal callback absence alone does not fail. Received callbacks
remain fully validated: malformed/foreign/nonzero error events fail; zero-error
nonempty remarks latch rejection while passive logging continues. Final cleanup
events can still reject the trial. queryEventInfo is not required or introduced.

Accepted trials are valid_for_geometric_analysis=true and
valid_for_timing_analysis=false. No velocity, acceleration, jerk or Section-V
timing-law result is calculated. The later primary analysis is raw measured
q_m -> authoritative URDF FK -> xMateCR7_link6 realized curve -> symmetric
geometric target/realized curve distance. The runner does not compare R versus D.

Frozen order is R,D,D,R,D,R,R,D,R,D (five pairs). The manifest and both original
prepared path_0003 knot files are SHA-256 pinned. All 100 six-joint vectors are
submitted unchanged and persisted as JSON, little-endian binary64, command
hex snapshots and hashes. No IK, diffusion, smoothing, clipping, wrapping,
rounding, resampling or trajectory generator is used by the R/D runner.
Policy remains jointSpeed=0.05, speed=50, zone=1 for 0..98 and zone=0 at 99.
One owning SDK thread, one robot object, one connection and one trial per
invocation are retained; no automatic positioning, retries or recovery motion.

Trial 1 launch:

```powershell
& "C:\Users\tgm10\AppData\Local\Programs\Python\Python312\python.exe" -I "C:\RokaeSDK\path_0003_nrt_runner.py" --run-trial --trial 1
```

Inputs are exactly `CONNECT PATH0003 TRIAL 1 R`, then the displayed
`SETUP PATH0003 TRIAL 1 R <hash>`, `APPEND PATH0003 TRIAL 1 R <hash>`, and
`START PATH0003 TRIAL 1 R <hash>`. SETUP has its own hash; APPEND/START share the
final batch hash. Review the full path, initial target transition, clearance,
existing power/mode and physical safety controls before authorizing. There is
no joint/direction/excursion input for frozen R/D. Later launches change only
--trial to 2..10. An exclusive trial reservation prevents accidental duplicate
runs; preceding trials require a persisted final disposition. A canceled or
failed trial is retained, not silently repeated or retuned.

Each unique trial folder under nrt_v050/path_0003_sessions saves exact commands,
source bytes/hashes, condition/pair/trial, raw q_m, events, operation-state calls,
START boundary, final stationary window summary/residual, logging gaps and
accepted/rejected geometric validity. Protocol evidence is also saved per trial.
