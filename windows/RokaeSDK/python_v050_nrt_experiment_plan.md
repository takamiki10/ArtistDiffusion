# Python v0.5.0 NRT hardware-transfer experiment plan

**The local v0.5.0 stubs declare all essential NRT APIs. A dedicated Python 3.12 x64 environment is prepared, but Windows Code Integrity blocks the SDK DLL import. Stationary capture and real SDK command-object round-trip verification are therefore blocked.** No robot object was constructed, no SDK connection or robot motion occurred, and no controller setting or original R/D knot changed.

This is a prospective protocol change before physical R/D outcomes. The primary experiment is now **hardware transfer of exact waypoint sequences under one fixed controller interpolation policy**, not replay of the Section V 10 s law or the prior 30 s C2 law. Python xCoreSDK **v0.5.0 only** is selected. The old drawing socket pipeline and v0.5.1 are excluded.

## 1. Exact local NRT capability audit

Evidence comes only from the intact local `xcoresdk_python-v0.5.0.zip`: archive SHA-256 `0d0d1db19fd3340712606531063f3ea0fc65c01a3226d0fb2878c4452ffb1832`. [Archive/member manifest](evidence/v050/package_manifest.json) records its exact source path and every member hash. Windows stubs, matching examples and PE metadata were inspected; no newer SDK document supplies a missing method.

| Capability | Exact local evidence and limits |
|---|---|
| connectToRobot / disconnectFromRobot | Declared on robot base classes; disconnect documents stopping motion before disconnect |
| setMotionControlMode / NrtCommandMode | Setter declared; `MotionControlMode.NrtCommandMode` explicitly present |
| JointPosition | Constructor from list of joint radians; readable/writable `joints` and `external` properties |
| MoveAbsJCommand | Constructor `(target: JointPosition or list[float], speed: int=-1, zone: int=-1)` |
| moveAppend | Overloads include `list[MoveAbsJCommand]`, returned command ID through PyString and error dict; example says multiple commands must have the same type |
| moveStart | Explicit start or resume; loading/constructing a list is not this call |
| stop / pause | Both declared; pause is same as stop. Stop is a planned Stop2 pause without power-off, not a hardware emergency stop or queue reset |
| moveReset | Clears submitted commands/execution information. **Its doc states Robot initialization calls motion reset once**, and switching RL/SDK command control requires reset |
| State subscription | startReceiveRobotState, stopReceiveRobotState, updateRobotState and getStateData declared; 1/2/4/8 ms or 1 s supported, max 1024 bytes |
| Measured joints | RtSupportedFields.jointPos_m = `q_m`, radians; `getStateData(field, PyTypeVectorDouble, 6)` overload |
| Measured velocity | No subscribed arm `dq_m` declared. Separate `jointVel(ec)` query exists, rad/s; external-axis `ex_dq_m` is not arm velocity |
| Pose | `tcpPose_m=pos_m`, row-major base-relative 4x4; `tcpPoseAbc_m` also available. Quantitative use needs frame/link6 registration |
| Motion events | `setEventWatcher(Event.moveExecution, callback, ec)` and `setNoneEventWatcher` declared; keys cmdID, wayPointIndex, reachTarget, error, remark, customInfo |
| Identity/state | robotInfo returns Info.id/type/version/joint_num; operationState, powerState and operateMode are declared |
| Controller sample timestamp | Not declared in subscription fields; LogInfo.timestamp is a log timestamp and must not be substituted |

No essential NRT method is missing from the declarations. **Runtime exposure and successful binding calls remain unverified because import is blocked**, so “declared” is not reported as “executed successfully.” The stubs document implicit initialization reset; a first stationary connection cannot be described as having no controller-side lifecycle effects merely because the application omits explicit setters.

### Exactly 100 commands in one append

All six prepared batches contain exactly **100 objects' worth of six-joint data**, and the declared overload accepts a homogeneous list. The local example demonstrates a list and allows multiple same-type commands. However, the inspected Python stub and example do **not** state the numeric 1–100 bound. The user-referenced public bound is not upgraded into verified local binding/controller acceptance.

Plan a single `moveAppend(commands, cmdID, ec)` for the complete 100-element homogeneous batch after validation. Do not split, stream, truncate or append after execution begins. Confirm actual 100-command acceptance with the separately reviewed 100-waypoint neutral batch; capture the result ID/error and complete index sequence. This cannot be proven by constructing a Python list or by calling an unbound method without a robot. No append call was made offline.

## 2. Isolated Python 3.12 environment and import result

Prepared directory: `nrt_v050/runtime`, with exact local SDK files in `nrt_v050/sdk`. The runtime is **CPython 3.12.10, AMD64, 64 bit, MSC v.1943**, using the official Windows embeddable distribution. It is isolated with an explicit `_pth`, environment-path isolation and site import disabled; this is a dedicated runtime, not a pip-managed venv. Existing Python 3.14 and global PATH/registry/package configuration are unchanged.

The runtime came from the [official Python 3.12.10 release](https://www.python.org/downloads/release/python-31210/). Its listed archive checksum was verified, and python.exe's Authenticode signature is valid from the Python Software Foundation. The [runtime source record](nrt_v050/downloads/runtime_source.json) includes URL and SHA-256. Python 3.12.10 was the last full maintenance release with this official Windows package; it is not represented as the newest security release.

The exact SDK PYD targets `cp312-win_amd64`, imports python312.dll and its matching xCoreSDK.dll, and requires compatible MSVCP140/VCRUNTIME140/UCRT. PE machine is x64 and observed SDK linker version 14.16. [Environment manifest](nrt_v050/evidence/environment_manifest.json) records Python version/architecture, SDK archive/DLL/PYD hashes, dependencies and package/runtime file hashes.

**Offline import failed** inside and outside the sandbox. Windows Code Integrity event IDs **3077 and 3033** identify `nrt_v050/sdk/xCoreSDK.dll` as not meeting Enterprise signing requirements/policy `{0283ac0f-fff1-49ae-ada1-8a933130cad6}`. Both supplied SDK DLL/PYD are unsigned. See [import result](nrt_v050/evidence/import_only.json) and [Code Integrity events](nrt_v050/evidence/code_integrity_events.json). This is an operating-system application-control blocker, not evidence of a missing Python NRT method or a robot communication failure.

No security policy was weakened, no DLL renamed or placed elsewhere to evade it, and no protected setting changed. A legitimately approved execution environment or an appropriately signed/authorized vendor build of the **same permitted version** is required. After that, rerun the import-only audit before constructing a robot object. A newer SDK is not a substitute.

## 3. Exact waypoints and offline construction audit

[prepare_nrt.py](nrt_v050/prepare_nrt.py) verifies all 56 package hashes and the original 100-row/radian input schema before preparing data. The original timestamps remain provenance only; no NRT arrival time is inferred from them. All six command batches are retained under [prepared](nrt_v050/prepared), preserving all 3600 joint values through JSON binary64 round trips. There is no IK, diffusion regeneration, smoothing, clipping, wrapping, decimal rounding or remapping.

The prepared `construct_commands` builds JointPosition, then MoveAbsJCommand, assigns the fixed fields and reads `position.joints` and `command.target.joints` back for exact binary64 comparison. It rejects length, value, speed/zone and customInfo mismatches before any append/start. [binding_roundtrip_audit.py](nrt_v050/binding_roundtrip_audit.py) is ready to perform this using the **real SDK**, for all six sequences and all three policy candidates, without any robot object.

**Real SDK construction/round-trip audit has NOT passed or run**, because its prerequisite import failed. Nine offline tests passed in Python 3.12 using synthetic command classes and geometric/FK fixtures; these verify the audit logic, not the blocked vendor binding. Current `binding_round_trip_verified=false` and `motion_authorized=false` remain explicit in the policy manifest. Controller transport/internal precision and physical visitation of waypoints remain later measurements even after object precision passes.

## 4. Fixed controller policy and blending candidates

The [proposed policy](nrt_v050/prepared/controller_policy.json) is selected before physical outcomes for low joint speed and the smallest positive integer blending zone. It is not selected from R/D performance, and is not copied from the drawing project's v500/z50.

| Field | Proposed fixed value | Local semantics / qualification |
|---|---|---|
| target | Each original JointPosition, in original order | All 100 knots, unchanged |
| jointSpeed | **0.05** | Explicit fraction in [0,1]; nonnegative overrides joint-speed calculation from speed |
| speed | **50** | Explicit mm/s field; avoids inherited/default -1. It is not a promise of actual TCP speed for MoveAbsJ with jointSpeed override |
| zone, indices 0–98 | **1** | Smallest positive integer radius field, documented in mm; actual joint blending needs neutral validation |
| zone, index 99 | **0** | Fine endpoint for completion, applied identically to every R/D trial |
| customInfo | Stable trial/command tag and original index 000–099 | Metadata returned by event feedback; no waypoint numeric encoding |

At nominal 5% of the supplied configured maximum joint speeds, reference joint-speed ceilings would be **[9,9,11.7,12,12,12] deg/s**, before other planner, override, mode and dynamics constraints. This percentage calculation is an engineering selection, not verified actual motion. Do not alter live acceleration/jerk, smoothing, dynamics or safety settings.

Local NrtCommand properties describe speed in mm/s and zone radius in mm. The local default-setting method documentation describes speed bands (<100 → 10%, etc.) and zone bands (<1 → fine; 1–20 → 10%; 20–60 → 30%; >60 → 100%). These describe conversion semantics, not a guarantee that per-command MoveAbsJ interpolation maintains a Cartesian tube of the entered radius. The neutral test must verify the per-command behavior. Explicit jointSpeed avoids relying on the speed-band mapping for joint velocity.

| Candidate | Zone policy | Intended interpretation |
|---|---|---|
| A: fine | 0 for every waypoint | Documented zero/fine category; conservatively treat as stop-and-go at each waypoint, not a proven continuous-path policy |
| B: selected small blend | 1 at 0–98, final 0 | Proposed continuous interior path with fine finish; smallest positive integer zone. Must validate accepted semantics on neutral motion |
| C: comparison only | 5 at 0–98, final 0 | Another small value inside documented range; may share the same joint percentage category as 1 and need not be geometrically distinct |

No claim that 1 mm blending preserves every physical waypoint, preserves the R/D difference or bounds Cartesian error by 1 mm is made. Exact **submitted** knots and a common policy define the hardware-transfer ablation. Blending can attenuate small differences; that is a transfer outcome to report, not a reason to tune the policy using R/D runs. Freeze B before R/D; neutral testing may validate or reject its semantics. If it is unsupported/unsafe, stop and amend prospectively before any R/D data, rather than automatically choosing whichever candidate favors D.

## 5. One object, separate readiness and start

One experimental process owns **one robot object/connection** for motion and logging. Artist_movement_finish and independent RL programs remain stopped; no second passive client runs. Use one ordered state reader and one disk writer in-process. SDK thread safety for state updates, motion events and command operations must be validated on the neutral test; Python threads alone do not establish it.

Future motion sequence: CONNECT → verify identity/state → subscribe → LOAD → VALIDATE all 100 Q → construct and round-trip audit → prepare and acknowledge complete NRT batch → READY → explicit operator authorization bound to command/policy/settings hashes → moveStart → log through confirmed completion → unsubscribe → disconnect. NrtCommandMode and any required queue reset belong to the later explicitly authorized setup, never the first stationary test. Power enabling remains operator-controlled and separately reviewed; no automatic power setter is proposed.

Loading, parsing, constructing or acknowledging a batch must not start it. Any malformed input, append error, missing command ID, logging failure or partial acknowledgement enters FAULT and prohibits moveStart. Do not resume after a fault. Reset clears queue/history and is not a stop; after a motion fault use the separately validated stop policy and only reset once stopped. `stop`/`pause` is Stop2, not emergency protection. Record and account for initialization's documented implicit motion reset before permitting even a stationary connection.

## 6. Measured-state logging and completion

Prefer **8 ms / 125 Hz q_m** only after stationary testing shows reliable acquisition. A low-rate 1 s smoke precedes it. Request optional pos_m only with declared frame provenance. Subscribed arm dq_m is absent; a separately timed jointVel query could be added later, but do not silently label finite differences or external-axis velocity as native measured dq.

Retain unsmoothed raw q doubles, host monotonic update-start/end and getStateData-end, logger sequence, update byte count, per-field return codes/validity, timeouts, drops/backlog diagnostics and errors. No controller timestamp/source counter is invented. Preserve all raw rows; invalidity flags and any exclusion are separate. Event callbacks copy payload plus host receipt time into a bounded queue and return quickly; never block callback threads with CSV/disk writes. Record cmdID, wayPointIndex, reachTarget, error, remark and customInfo in the event file.

Record start-request and start-return times separately from the first observed moving state/event. `moveStart` returning success is not controller application time. Completion requires the acknowledged cmdID, final original waypoint index 99 (zero-based), reachTarget=true, no error, and confirmed idle/stationary state under a predefined settling criterion. Missing/mismatched/duplicate events or a timeout are not success. Idle alone can mean paused/faulted; never infer completion solely from it. Preserve event anomalies. A final fine waypoint is intended to make completion interpretable; verify actual event semantics on the neutral batch.

Use the existing decoded-raw/logger design with an NRT event stream and session manifest. The real SDK-to-persistent-logger integration and completion logic remain to be exercised after import resolution; no live logger result is claimed here.

## 7. Primary geometric analysis

Authoritative analysis model is the immutable package `robot_model/xMateCR7.urdf`, SHA-256 `921e6ae2402ccbf8112ece1b88776c22c841d6e6a04519c3e6b903d2c1bd427a`, joints joint1–joint6, endpoint **xMateCR7_link6**. This is the authoritative benchmark geometry, not independently established physical metrology or a certified dynamics model. The file warns its inertia is incorrect; do not use those inertias to clear motion.

Measured q → verified joint conventions → URDF FK → **“link6 trajectory reconstructed from measured joint states.”** Never label it externally measured Cartesian position. Verify zero/turn/order and base-frame registration prospectively; apply no error-minimizing rigid fit, Procrustes/ICP, per-trial shift, scale, or flexible alignment.

Use the prescribed `target_paths/path_0003.csv` Cartesian polyline in its recorded frame. Use the reconstructed measured polyline between acquired states. Define directional arc-length-weighted mean-square nearest-**segment** distances:

`E(T→R)^2 = (1 / length(T)) ∫_T distance(x, realized_polyline)^2 ds`

and reverse `E(R→T)` analogously. Primary metric: **symmetric RMS = sqrt((E(T→R)^2 + E(R→T)^2)/2)**. Report both directional RMS values, symmetric arc-length 95th percentile and sampled maximum with discretization bounds as secondary metrics. Target→realized detects missing coverage; realized→target detects excursions. Arc-length weighting avoids privileging slow/dwell-heavy controller sections. Nearest vertices and time-aligned numerical sample errors are not the primary metric.

[geometric_metrics.py](nrt_v050/geometric_metrics.py) implements a reference nearest-segment metric and hashed-URDF FK. It uses midpoint arc-length quadrature at 0.1 mm spacing; derived analysis only, without resampling or changing raw measurements. Before reporting, refine to 0.05 mm and require RMS/p95 change below 0.001 mm; if needed continue deterministic refinement without reference to R/D benefit. Report sampled max and its spacing/2 Lipschitz bound for the recorded polylines; this does not bound unobserved motion between state samples. Missing-data gaps invalidate curve coverage rather than being bridged silently.

Geometric distance alone cannot detect reversed traversal, repeated loops or all ordering errors. Also retain ordered command events, start/end checks, target coverage/progress and backtracking diagnostics; no ordered-execution failure is rescued by a small symmetric distance. Retain joint-space following analyses: waypoint order, nearest six-dimensional commanded-sequence distance, per-waypoint closest approach and final residual. Blended interior knots need not be reached exactly; report that behavior rather than applying a fabricated time target.

The NRT primary metric is new and not numerically interchangeable with the old time-aligned benchmark RMS. Physical timing/duration remains a measured secondary outcome. The 30 s C2 calculation is a numerical feasibility comparison only, not the realized interpolation or a bound on NRT demands. No Section V physical-jerk comparison is made. Any later measured-jerk metric requires reliable timing, explicit differentiation/noise treatment and a separately labeled physical analysis.

Prospective paper wording: “The physical hardware-transfer ablation will submit the original reference and diffusion-refined joint waypoint sequences using an identical, fixed ROKAE non-real-time interpolation policy. Evaluation will use geometric deviation of the link6 trajectory reconstructed from measured joint states, rather than replay of the standardized numerical timing law.” Use past tense only after those experiments actually occur.

## 8. Prospective path_0003, ten-trial structure

The first physical ablation uses **path_0003 only: five R and five D executions**. Paths 0001/0006 remain in the project and their exact batches are retained. This scope reduction is prospective and time-constrained, not a selection based on physical outcomes.

Selection rationale is recorded as supplied: path_0003 was already deterministically selected, has **three committed diffusion segments**, and has the largest planned separation among the three selected paths. Existing metadata reports 38 changed samples, max joint difference 0.0018384712029086425 rad and a planned time-aligned RMS gain 0.058532385529350694 mm. These justify the pre-data prioritization, not a predicted gain in the new geometric metric or evidence of physical success.

The [experiment manifest](nrt_v050/prepared/experiment_manifest.json) freezes five adjacent R/D pairs using seed 20260913 to shuffle three RD and two DR orders. Each pair uses identical operator-reviewed approach/start configuration, policy, load, controller settings, logging and rest conditions. Approach/return/settling are separate staging operations and excluded from the geometric trial window by explicit events, not outcome-based cropping. They are not automatically generated or executed here.

| Trial | Pair | Condition | Source |
|---:|---:|---|---|
| 1 | 1 | R | trajectories/path_0003/A.csv |
| 2 | 1 | D | trajectories/path_0003/B.csv |
| 3 | 2 | D | trajectories/path_0003/B.csv |
| 4 | 2 | R | trajectories/path_0003/A.csv |
| 5 | 3 | D | trajectories/path_0003/B.csv |
| 6 | 3 | R | trajectories/path_0003/A.csv |
| 7 | 4 | R | trajectories/path_0003/A.csv |
| 8 | 4 | D | trajectories/path_0003/B.csv |
| 9 | 5 | R | trajectories/path_0003/A.csv |
| 10 | 5 | D | trajectories/path_0003/B.csv |

Primary paired effect is `R symmetric RMS − D symmetric RMS`, positive for lower D error. Report all ten planned attempts, each condition's values and paired differences, variability and technical failures. Ten attempts are not ten independent targets. Do not discard failed/higher-error D or R runs or silently add replacements. Technical failures stop the protocol and remain recorded; any changed attempt budget requires an explicit amendment. With only five pairs, two-sided exact sign/randomization inference has minimum attainable p=2/32=0.0625; emphasize effect estimates and uncertainty, not a guaranteed 5% significance claim. Do not tune zone, trial alignment, metrics or exclusions from observed R/D outcomes.

## 9. First hardware step: stationary only

**Do not execute now.** After legitimate resolution of the Windows import blocker, pass the import-only and real command-object audits without a robot first. A subsequently authorized stationary smoke contains no NRT motion, no queued R/D commands and no call to setMotionControlMode, moveAppend, moveStart, explicit moveReset, power, alarm or safety setters. Account for the SDK's implicit initialization reset and stop-on-disconnect with all RL/motion tasks stopped; do not describe it as harmless to an independent task.

Operator checks: Artist_movement_finish and every RL motion program stopped; arm stationary; no pending independent movement/queue; RCI OFF; recorded power/mode state. Connect once, compare robotInfo type/version/joint count with live evidence, record actual UID, and read power/operate/operation state before and after. Subscribe q_m at 1 s for approximately 10 s, check finite six-joint returns and plausible timing, unsubscribe, record final state and disconnect. No automatic change if data require a different mode/power state; fail closed. The operator observes after disconnect as the SDK cannot query once disconnected. No unexpected motion, power or mode changes are acceptable.

If accepted, separately authorize an 8 ms/approximately 30 s stationary characterization to assess cadence, gaps, backlog, unique digital values and variability. Record actual rates rather than treating a requested period as measured performance. No R/D batch is loaded in either initial stationary test.

## 10. Separate neutral NRT validation

Prepare a candidate from a newly recorded stationary q0 after smoke, with an **operator-selected joint** and an excursion **no greater than 0.25 degrees** in a reviewed safe direction/region. This is a small-excursion proposal, not a safety guarantee. Current q, clearance, effective limits, load and braking margin must be reviewed before assigning absolute commands; none were invented from R/D data.

To validate one-call capacity, the proposed neutral batch contains 100 same-type waypoints sampled from a smooth small out-and-back cycle around q0, exact q0 at both ends, with the selected policy and tagged indices. It is not an R/D path and is not derived from diffusion differences. If spacing/zone interaction generates too-close-point warnings or unresolvable skips, record and reject the policy validation rather than changing R/D knots. No automatic neutral generator or runner is enabled here.

Under separate motion authorization, verify NrtCommandMode, queue reset/empty state, acknowledgement of all 100 commands in one append, READY before explicit start, q logging at the approved rate, speed/zone behavior, event order/final completion, Stop2 pause and safe reset semantics. A stop/reset validation must be its own reviewed part of the neutral test, not an improvised interruption of R/D. Keep actual power operation under operator control. Only after neutral acceptance and policy freeze may the ten-trial R/D plan be considered for authorization.

## 11. Verification and next concrete action

Completed offline: isolated Python runtime/signature verification; local NRT stub/example audit; six exact waypoint batches and fixed schedule/policy; prepared real-SDK data-object audit; reference FK/geometric implementation; **nine passing synthetic/offline tests**, including detecting artificial command rounding and reproducing the archived URDF FK reference RMS. No test invoked a vendor robot object or motion API. All 56 package hashes verified unchanged.

Blocked: SDK import by Windows application control, hence actual binary NRT exposure, JointPosition/MoveAbsJ precision audit, runtime list handling, stationary capture and neutral validation. The next concrete action is legitimate resolution of the recorded Windows signing-policy block for this exact SDK environment, followed by the prepared offline audits. Do not disable protection or substitute v0.5.1 to bypass it. There is no current authorization or readiness for hardware capture.

OTHER CONCRETE BLOCKER
