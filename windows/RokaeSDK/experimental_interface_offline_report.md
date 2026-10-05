# Experimental interface offline report

Prepared 2026-09-13. **The offline components are implemented and tested; passive connection and physical execution remain blocked.** No robot connection, motion, RobotAssist launch, SDK robot construction or stationary-capture invocation occurred. The existing drawing stack and immutable handoff package remain untouched.

This separate project is `C:\Users\画家ロボット\Desktop\system\rokae_experiment_offline`. The scope is offline preparation, not a deployable motion or acquisition backend. Supporting procedures and artifacts are linked below.

## 1. Recommended SDK/control architecture

Use the locally bundled C++ **xCoreSDK-v0.5.1.rt_0**, subject to exact firmware compatibility confirmation. Its CHANGELOG dates this version 2025-07-31. Inspected archive SHA-256: `9b1f02410899d143340b53a745f62964550cbbe406d90d623402de361de0bb42`. Headers examined include `robot.h`, `motion_control_rt.h`, `data_types.h`, `utility.h`, with local examples, README/CHANGELOG and C++ manual V3.1_A. [Source manifest](evidence/sdk_source_manifest.json) records individual hashes. No newer SDK was downloaded or assumed. The available Python extension targets CPython 3.12, unlike the installed 3.14 interpreter.

The local compatibility statements conflict: README/USB notes say xCore ≥3.0.1; CHANGELOG v0.5.1 says ≥3.0.2. RobotAssist's cached firmware is `3.0.1.1.C89.20250306`. Current firmware and vendor confirmation are needed; this is not resolved by selecting the less restrictive statement. The current executable links only the C++ standard library/thread facilities, not the SDK.

## 2. One process owns motion and logging

**Recommend one future process/robot object for both**, with one state reader and a separate in-process writer. Local APIs/examples support RT control and state subscription through one object. No demonstrably isolated multi-client passive-logger contract was found. RobotAssist controls can affect shared state, and the manual recommends a single control source.

`disconnectFromRobot` stops before disconnect; the Cobot destructor stops a moving robot; RT network disconnect also stops motion. Effects on independently running RL tasks are not explicitly isolated. Subscription examples do not prove harmless coexistence with `Artist_movement_finish`. The [architecture](experiment_interface_architecture.md) documents these findings and the future LOAD → VALIDATE → READY → explicit authorization → EXECUTE design. In the offline implementation, EXECUTE always throws.

## 3. Exact proposed interpolation

PL interpolation preserves knots but introduces velocity discontinuities and acceleration impulses. The local SDK manual §5.3.1 requires smooth joint-space references with continuously differentiable velocity and recommends zero endpoint velocity/acceleration. PL is retained for audit only.

The prospective host reference is **C2 piecewise quintic Hermite**, selected for regularity and stationary endpoints, identically for R and D. With interval h_i and slope s_i, interior v_i is the adjacent-interval-weighted mean slope and a_i=2(s_i−s_(i−1))/(h_i+h_(i−1)); both endpoint derivatives are zero. Each quintic matches its two knots and these first/second derivatives. [execution_timing_design.md](execution_timing_design.md) gives every coefficient and was written before implementation.

At every original stored timestamp, evaluation returns the original binary64 knot exactly. No clipping, wrapping, knot smoothing or retiming occurs. This is a mathematically suitable host-reference candidate for documented JointPosition commands, not a built-in ROKAE interpolation API or a certified executable trajectory. Between-knot overshoot, jerk and undocumented filtering remain material blockers. The unchanged Jetson analysis needs a separately reviewed prospective extension to match this convention before collection.

## 4. Proposed control/update rate

Propose **1 ms / 1000 Hz**, because that is the local `setControlLoop` cycle. State subscriptions separately support 1/2/4/8 ms or 1 second, with maximum documented payload 1024 bytes. A 1 ms command stream has 10001 reference values for k=0…10000 over 10 seconds, including both endpoints.

The 10/99-second knots are not on the millisecond grid except at endpoints; nearest-grid displacement reaches 0.494949… ms. Preserve their continuous definition rather than snap them. Exact host knot interpolation does not imply exact physical passage at those times. Future t=0 means application of the first experimental q0 after separately authorized positioning; host callback time is only a proxy. The t=10 q99 command has zero mathematical endpoint velocity/acceleration. Final-command/`setFinished()` ordering, controller application time and filter effects remain unverified.

The SDK documents filtering of callback commands. Neither SDK nor controller filter bypass/default transfer functions were established. Do not assume 0 or 1000 Hz disables filters. Windows 1 kHz performance is untested. Host deadlines, read brackets, command index and timing uncertainty must be retained separately; no silent catch-up or time stretching is acceptable.

## 5. Representation and strict loading

All six trajectories pass the strict loader: exactly 100 rows and seven fields, six finite radians, strictly increasing times, exact endpoints 0 and 10, grid/spacing tolerance **1e-12 seconds**. Bad numbers, missing/extra rows, bad headers, invalid spacing and untrusted/mismatched manifests fail closed. Invalid values are never replaced with zero. The CLI checks every manifest file before accepting trajectories and rechecks each loaded trajectory hash.

The package manifest is pinned to `dbaba32746457240385aaf123a91f84164cba3a4986337addbf7a0c3d2a6506f`; all **56 files** verified. Native binary64 and 17-significant-digit diagnostic text both round-trip with **zero bit mismatches and zero radian serialization error** on all 3600 joint values. Exact knot checks likewise have zero bit mismatches. This concerns host representation; SDK transport/controller internal precision remains to be verified.

| Path | R/D joint entries that differ | Erased by old .4f | Erased by new representation | Old maximum serialization error (rad) | New maximum error |
|---|---:|---:|---:|---:|---:|
| 0001 | 180 | 94 | 0 | 4.9998431411291122e-5 | 0 |
| 0003 | 228 | 106 | 0 | 4.9804902076733271e-5 | 0 |
| 0006 | 246 | 112 | 0 | 4.9740982055457295e-5 | 0 |

The old formatting erases **312 of 654 changed entries**. The new representation erases none. See [precision audit](artifacts/run_20260913T073830/audit/representation_audit.csv).

## 6. Offline velocity, acceleration and step audit

The [36-row per-joint table](artifacts/run_20260913T073830/audit/demand_audit_summary.md) and [full-precision CSV](artifacts/run_20260913T073830/audit/command_audit.csv) contain all six commands: PL maximum interval speed, interior finite-difference acceleration, C2 maximum speed/acceleration/jerk, 1 ms maximum requested step, continuous position envelope and PL endpoint slopes. [R/D differences](artifacts/run_20260913T073830/audit/rd_demand_differences.csv) explicitly report D−R for every joint and demand metric.

| Command | Largest C2 speed across joints (rad/s) | Largest C2 acceleration (rad/s²) | Largest C2 jerk (rad/s³) | Largest 1 ms step (rad) |
|---|---:|---:|---:|---:|
| 0001 R | 1.21840192 | 8.54467665 | 436.589370 | 0.00121839344 |
| 0001 D | 1.21845094 | 8.54571301 | 436.589370 | 0.00121844226 |
| 0003 R | 1.34121306 | 13.6237783 | 1201.58813 | 0.00134121121 |
| 0003 D | 1.33925219 | 13.6758886 | 1206.35009 | 0.00133924796 |
| 0006 R | 0.414951723 | 7.53833327 | 694.767012 | 0.000414948016 |
| 0006 D | 0.414951723 | 7.51369129 | 692.247419 | 0.000414948016 |

Cached motion settings give joint speed maxima [180,180,234,240,240,240] deg/s and acceleration maxima [1500]×6 deg/s² (26.1799388 rad/s²). All candidate speed/acceleration maxima are below these cached numbers. That comparison is not a dynamic feasibility result or a current RT-limit check.

Cached jerk is [5000]×6 deg/s³ (87.2664626 rad/s³), while `JERK_LIMIT_JOINT=0`. Every candidate exceeds that numeric value on joints 1–5. If this is an active hard RT constraint, all six candidates as defined fail it. Enforcement, flag interpretation and RT applicability require verification. Units follow local xCore V3.0_B §15.4.16.22–23 (PDF p.311, printed p.297). SDK ER3/ER7 tables are not CR7 limits.

Extrema use numerical derivative-root searches, not certified interval bounds. Finite differences do not establish torque, load, collision, tracking reserve, stability, filtering fidelity or mode-specific feasibility.

## 7. Logger fields and timing provenance

The implemented logger preserves required measured q and optional measured dq, 4×4 end pose, target q and separately timed controller state, plus logger sequence, host read-start/end, bytes returned, reported/validated masks and overall/per-field SDK status. [logger_format.md](logger_format.md) defines the 445-byte framed decoded binary record, incremental parser, JSONL events, diagnostic CSV and provenance limits. [Session metadata template](session_metadata.template.json) supplies unresolved hardware gates as null/false, not invented values.

No controller sample timestamp or source counter is assumed. CSV time is host read-end relative to a declared origin. A future `useStateDataInLoop=true` callback cannot bracket the SDK's preceding internal state read; use truthful callback provenance or a separately verified explicit-reader design. Optional state queried by other calls gets its own bracket. Dropped controller frames cannot be counted from logger sequence alone.

Timeouts generate events without stale samples. Malformed/unavailable required fields, inconsistent validity, duplicate times, source-counter gaps where actually exposed, queue overflow and writer failure are tested. Invalid acquired records remain raw; invalid CSV fields remain empty. The mutex/deque implementation is an offline prototype, not a qualified 1 kHz queue. General hardware metadata validation, SDK acquisition and a hardware-file conversion CLI remain future work.

## 8. Stationary-capture readiness

**Not ready.** The named mode is present as a fail-closed blocked entry point and was not executed. It has no backend and cannot subscribe or connect. It issues no motion, power, mode, alarm, tool or safety commands. Enabling it requires the lifecycle/compatibility evidence below plus a reviewed acquisition adapter and complete metadata handling. Synthetic success does not authorize that step.

## 9. Current-controller information needed

Collect the exact evidence in [current_limit_verification.md](current_limit_verification.md): physical model/serial, current full firmware, live identity/project/task/control state, normal/reduced joint-position and soft limits with all enable flags and units, active safety mode/configuration/checksum, dynamic speed/acceleration/jerk limits and RT applicability, selected TCP/workobject and load mass/CoM/inertia, calibration/zero/turn conventions and SDK/RCI availability. Preserve screenshots or direction-verified read-only exports with timestamps and hashes. No setting changes or program starts are needed for this task.

## 10. Is a path already impossible from offline evidence alone?

No unconditional physical impossibility of the stored knot paths is established. The old drawing protocol cannot provide the required timestamp/precision/measurement semantics, and PL is unsuitable under the inspected smoothness requirements. Those are protocol conclusions, not proof that every physical realization is impossible.

The chosen C2 candidate for **path_0006** reaches joint 1 = **3.1425140906387883 rad** for both conditions, exceeding the Jetson 3.1416 bound by about 0.000914091 rad between knots. Its largest stored knot is 3.1415998935699463 rad. The candidate therefore fails that modeled continuous bound if it is retained as a hard constraint. Cached normal/reduced soft/safety bounds are ±360° and contain the candidate, but are not verified current limits. Resolve model/current-limit consistency and operational margin; do not clip or declare current hardware eligibility from either alone. The cached jerk comparison is another conditional rejection described above. No trajectory or target selection was changed.

## 11. Before the first passive SDK connection

Require exact firmware/SDK pairing confirmation; documented construction/subscription/destruction/disconnect behavior, including errors and unrelated RL tasks; a verified single-owner idle state; confirmation that passive subscriptions need no prohibited state change; approved connection/exit/fault procedure; known read timing/field validity and pose semantics; and a reviewed acquisition implementation that cannot call setters or motion. Confirm endpoint/host network configuration without probing the robot in this task. The local cache shows RCI disabled, which is evidence to verify, not permission to enable it.

Review offline fault-injection results, metadata completion, disk capacity/error handling and the operator's stationary observation/stop arrangements. Only after these gates and explicit authorization should the first passive capture occur. The [characterization plan](measurability_characterization_plan.md) defines Stage A and its acceptance-rule freeze.

## 12. Before the first trajectory motion

In addition to passive gates and successful Stage A: verify current robot/mapping/limits/tool/load and safe workspace; resolve candidate overshoot/jerk and full dynamic/clearance checks; resolve SDK/controller filter transfer, q_c semantics, start/final-command application ordering, timing uncertainty and supported deployment environment; qualify the queue/backend under load; implement separate authorization tied to command/settings hashes and a reviewed fault/abort path. No load, read error or partial input may initiate motion.

Review and authorize a **neutral Stage B command separately**, including positioning and stopping. It must not be any of the six R/D commands or be chosen by diffusion performance. Stage B characterizes repeatability; freeze the measurement metric, alignment, uncertainty floor and threshold before any R/D data. A later six-path experiment additionally requires its own reviewed commands, matched analysis convention, eligibility and explicit operator authorization. No automatic motion backend was built here.

## Verification and retained limitations

Final successful run: [artifacts/run_20260913T073830](artifacts/run_20260913T073830). g++ 15.2.0 under WSL compiled the offline C++17 components with warnings, AddressSanitizer and UndefinedBehaviorSanitizer; the final build emitted no warnings. **2,897 assertions passed**, covering parsing, checksums/pinning/tampering, interpolation, raw framing, logger failures and blocked execution transitions. The independent standard-library Python verifier passed **360,336 numeric checks**, with maximum independent reference discrepancy **4.440892098500626e-16 rad** against its 5e-14 tolerance and zero raw-to-CSV bit mismatches. All 56 package hashes reverified. This does not include a native Windows build, linked SDK test, formal extrema proof, full robot dynamics or physical measurement.

[Protected-file verification](evidence/protected_files_verification.json) confirms **24 controller/configuration files** retain their prior precheck hashes. The production sender was only read and its current hash recorded; no unsupported baseline-hash claim is made for that file. WSL compiler/CMake installation was approved; it did not install or run the robot SDK. All task outputs are in this separate project.

The blockers are concrete: incompatible/ambiguous local version requirements, undocumented passive ownership/lifecycle isolation, no enabled acquisition backend, unresolved filtering/timing fidelity and conditional continuous-position/jerk failures. The offline preparation is complete within the authorized boundary; hardware readiness is not.

**OFFLINE SAFETY/PROTOCOL BLOCKER**
