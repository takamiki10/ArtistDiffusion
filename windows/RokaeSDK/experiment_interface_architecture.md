> CURRENT PROTOCOL: [python_v050_nrt_experiment_plan.md](python_v050_nrt_experiment_plan.md) supersedes prior RT/30 s physical-execution assumptions. The experiment now uses exact waypoint sequences under a fixed Python v0.5.0 NRT policy, geometric metrics, and initially path_0003 with 5 R + 5 D trials. Earlier 10/30 s analyses remain numerical/historical references, not actual NRT timing. Windows SDK import is currently blocked; no hardware is authorized.

> CURRENT AMENDMENT: one-owner architecture remains; SDK v0.5.1 is excluded. Exact v0.5.0 package/API status is in [sdk_v050_capability_audit.md](sdk_v050_capability_audit.md). Physical duration is 30 s. Older version-specific findings below are historical evidence only.

# Experimental interface architecture

**Recommendation A: one future C++ experimental process owns both execution and measured-state subscription.** The current artifact is offline only: it has no robot SDK link, networking code, IP option, or executable motion transition. The production drawing system remains untouched.

## Local documentation findings

The inspected archive is `C:\Users\画家ロボット\Desktop\ROKAE_USBdata_20260107\sdk\2026_01_07_Readme\xCoreSDK-v0.5.1.rt_0.zip`. Its CHANGELOG labels v0.5.1.rt_0, 2025-07-31. Read `include/rokae/robot.h`, `motion_control_rt.h`, `data_types.h`, `utility.h`, examples, README and CHANGELOG, alongside local `sdk\xCore_SDK_C++_UserManual_V3.1_A.pdf`. Exact hashes are recorded in the generated source manifest. No newer SDK is downloaded.

| Question | Evidence and conclusion |
|---|---|
| Same client can execute and read state? | Yes: `Robot_T::startReceiveRobotState`, `getStateData` and `getRtMotionController`; `setControlLoop(...,useStateDataInLoop=true)` updates state before each callback and requires a matching 1 ms subscription. Examples use one robot object for both. |
| State subscription outside RT? | The local read-state example subscribes under `NrtCommand`. It also starts a motion thread and is unsafe to run for discovery. This demonstrates an API pattern, not harmless coexistence with an independently running RL project. |
| RobotAssist coexistence? | Manual §5.1 lists shared controls: operating/power state displays, speed override, pause and selected tool/workobject can interact. Some settings/project pointers do not synchronize. The manual recommends a single control source. Do not assume a second GUI or SDK client is isolated. |
| Multiple SDK clients on one controller? | No affirmative safe multi-owner contract found. Changelog support for controlling multiple robots is not permission for multiple clients controlling one robot. |
| Disconnect and destruction? | `BaseRobot::disconnectFromRobot` stops motion before disconnect. `Cobot::~Cobot` stops a moving robot. RT `disconnectNetwork` closes RT ports and immediately stops motion if moving. Scope for an unrelated RL task is not explicitly isolated; therefore no safe separate passive-client assumption. |
| RT/RL coexistence? | `RtCommand` and `NrtRLTask` are different motion-control modes. The SDK's old RCI-client compatibility rules do not promise simultaneous control. Do not switch modes while the drawing project owns the arm. |

There is a **version discrepancy**: archive README and the USB Readme state xCore≥3.0.1, but CHANGELOG's v0.5.1 entry states xCore≥3.0.2. RobotAssist cached firmware is 3.0.1.1.C89.20250306. Vendor confirmation of this exact pairing or a verified compatible *local* SDK is required before the first connection; do not silently use the least restrictive statement.

## Future ownership and states

One robot object, one subscription owner, one ordered state reader; a separate in-process disk writer consumes owned record copies. With `useStateDataInLoop=true`, do not call `updateRobotState` again inside the callback. The callback cannot bracket the SDK's pre-callback state read with host read-start/end times; that provenance must be recorded accurately. If using an explicit reader instead, bracket its `updateRobotState` calls and record that states are asynchronous with the command callback. Choice/thread-safety must be verified with the vendor; do not pretend both designs have identical timing.

Conceptual future states: LOAD → VALIDATE → READY → explicit operator authorization tied to command hash/settings → EXECUTE → STOP/COMPLETE, with errors entering FAULT. Loading a file only produces validated immutable data. No malformed input, upload completion, read timeout, disconnect, or partial buffer can initiate execution. The current executable never advances to EXECUTE; READY means offline data readiness only and never hardware readiness.

For future stationary capture, use the same ownership policy, with no execution state and no mode, power, alarm, tool or safety setter. Require an externally verified safe lifecycle and a controller with no other motion owner. The present `--stationary-capture` mode is an explicit blocked entry point; it cannot connect. Backend substitution is not a runtime switch.

## Implementation boundary

Offline C++ components provide strict checksum-verified CSV loading, continuous reference evaluation, demand audits, full-precision representation, raw decoded logger, bounded queue, synthetic source, event logs, binary parsing and CSV conversion. The SDK is used only as documentation; no robot object is constructed. Python standard-library helpers generate reports and corroborate checksums, not a replacement physical backend.

The decoded record source contract represents `q_m`, optional `dq_m`, `pos_m`, valid RT `q_c`, and separately timed status. It cannot claim native packet bytes or undocumented controller timestamps. A future backend must fill validity and error fields from actual API results, and preserve every successful frame. Acquisition faults invalidate a session. A logging fault must signal the future motion owner; only its separately verified abort policy may stop motion.
