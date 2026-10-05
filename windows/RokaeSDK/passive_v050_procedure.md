# Future v0.5.0 stationary smoke and characterization

**Prepared, not executed and not ready to connect.** Use only the exact Python package in the capability audit if its passive fallback is later enabled. Do not load v0.5.1. The supplied source has no SDK import, hardware loader or enabled hardware CLI; tests inject a synthetic SDK. An external reviewed runner and persistent sinks are still required.

## Operator prerequisites

RobotAssist/controller on; arm stationary; no trajectory running; Artist_movement_finish and every independent RL motion task stopped; no other motion owner. RCI stays OFF. Preserve the existing power and control/operating modes. Prefer servo/motor power OFF **only if subscription in that existing state is supported**; support is currently unresolved. Do not power on, change mode, enable RCI/RT motion, reset alarms or call tool/safety/limit setters to obtain a sample. A failure must stop the test, not trigger a setter or automatic retry.

Before enabling a runner, verify the exact CPython 3.12 x64 runtime, matching archive/member hashes, DLL dependencies, existing-state subscription support and teardown behavior. Version minima match the selected firmware prospectively; this is not a test of the binary ABI/runtime or servo-off support. The prerequisites in code default to false. Operator stationary/idle observations do not themselves establish an undocumented API requirement.

## Profiles from v0.5.0 stubs

| Profile | Requested period | Nominal acquisition window | Expected order of record count |
|---|---:|---:|---:|
| Smoke | **1 s / 1 Hz** | **10 s** | About 10 |
| Characterization, after accepted smoke | **8 ms / 125 Hz** | **30 s** | About 3750 |

Both periods are explicitly permitted by v0.5.0 startReceiveRobotState. The initial connection/first-frame wait is outside the acquisition window; first-frame wait can take up to 3 s. One final blocking update may extend the window by about one requested period. Counts are not guaranteed by the request; record actual host intervals and duration. The prepared timeout policy fails on the first timeout, keeps its diagnostic and never emits cached values as a new sample.

Only q_m is required/requested. Do not request an invented dq_m or source timestamp. Optional pose can be added after frame provenance is known; separate jointVel calls are not part of this minimal lifecycle. Do not run vendor read-state examples: they change mode and command motion.

## Exact prepared sequence

Create default xMateRobot without IP, connectToRobot, startReceiveRobotState(period,['q_m']), then updateRobotState(period) and getStateData('q_m',vector,6), stopReceiveRobotState, disconnectFromRobot. No RT controller object is created. Host read-start/end bracket updateRobotState; a separate decoded-end timestamp follows getStateData. Preserve native six doubles, logger sequence, byte count, return code, validity, origin and events. Controller timestamp remains absent. The queue in the SDK does not overwrite old records automatically: preserve received order and inspect backlog/age; do not silently flush initial samples or equate receipt time with controller acquisition time.

The prepared function uses injected record/event sinks. A real runner must open/write metadata and raw/event files before connecting, preserve binary64 and rejected records, bound queues, flush/verify files and mark writer faults invalid using the existing logger design. The code is not yet wired to those persistent sinks. No favorable timing or freshness claim follows from successful synthetic tests.

On malformed q, timeout, read or sink error, stop acquisition and attempt unsubscribe/disconnect; no retries or setters. A partially started subscription also gets cleanup. The disconnect error-dict shape must be checked with this runtime: unknown schema fails closed rather than assuming success. Destruction and connect-failure cleanup semantics remain review items. Disconnect documents stopping motion, hence the strict absence of independent motion owners matters.

After a future successful smoke, separately authorize characterization with the same preconditions. Characterize actual interval/jitter, gaps, digital q variation and stationary variability. Do not count host gaps as exact lost controller frames without a source counter. These short sessions supersede the earlier 1 ms stationary plan for this workaround; they do not validate future RT execution timing or establish physical tracking accuracy.
