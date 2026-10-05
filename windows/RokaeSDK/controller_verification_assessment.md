> CURRENT PROTOCOL: [python_v050_nrt_experiment_plan.md](python_v050_nrt_experiment_plan.md) supersedes prior RT/30 s physical-execution assumptions. The experiment now uses exact waypoint sequences under a fixed Python v0.5.0 NRT policy, geometric metrics, and initially path_0003 with 5 R + 5 D trials. Earlier 10/30 s analyses remain numerical/historical references, not actual NRT timing. Windows SDK import is currently blocked; no hardware is authorized.

> PROSPECTIVE UPDATE: [time_constrained_workaround_report.md](time_constrained_workaround_report.md) supersedes the SDK v0.5.1 selection and 10 s physical-demand conclusions below. Live settings remain valid evidence. The selected physical duration is now 30 s, with positive configured jerk margins; v0.5.0 package/API availability and passive runtime/lifecycle remain concrete gates.

# Controller verification assessment - live-evidence update

**Vendor confirmation of SDK compatibility and passive connection lifecycle is now the principal blocker.** The supplied live observations resolve model identifier, firmware, displayed position safety bounds, reported motion parameters/enable states, SDK authorization and RCI status. First passive SDK capture is not yet authorized.

Assessment date: 2026-09-13. Evidence is accepted as **LIVE VERIFIED, operator-reported from connected RobotAssist**, as instructed. This is not an independent SDK read or screenshot review. Exact observation timestamp and physical robot/controller serial were not supplied and are not fabricated.

## Evidence and precedence

The [supplied live evidence](evidence/controller_live_operator_evidence.txt) is preserved verbatim with [provenance](evidence/controller_live_verification.json). It supersedes corresponding July cached observations. The [cached snapshot](evidence/controller_assessment_cached_snapshot.json), [prior offline report](experimental_interface_offline_report.md) and [SDK source manifest](evidence/sdk_source_manifest.json) remain historical evidence. Their statements that no live evidence existed are superseded here.

Live collision protection **ON** supersedes the old cached collision-enable=false observation; exact subsystem/threshold equivalence is not established. Unsupplied fields remain cached-only or unknown. The user's RobotAssist connection produced these observations; this assessment did not connect to the SDK, run examples, launch RobotAssist, enable RCI, change settings, invoke capture or move the robot.

## 1. Identity and authorization

| Field | Live verified value |
|---|---|
| RobotAssist/HMI | 5.0.11.0313 |
| Firmware | **3.0.1.1.C89.20250306** |
| Robot model identifier | **XMC7-R850-W7G3B4C-S5** |
| Cabinet/model | XBC_XMATE |
| Safety module | ROKAE_RSC |
| Controller interface | 192.168.2.160 |
| Observation context | Connected physical controller; super-administrator access |
| Authorized features | SDK and Japanese language |
| RCI | **DISABLED**; IP 127.0.0.1; port 1337; packet-loss threshold 0 |

Local model metadata maps the now-live-confirmed identifier to xMate_CR7-C. That friendly-name mapping remains local-file evidence. M82522101354 is still a workspace identifier, not a verified physical serial. Serial/session time should be added to acquisition provenance later; repeat evidence for the supplied model/firmware is unnecessary.

SDK licensing is resolved. Exact binary compatibility and current RT-motion readiness are not. The [README](evidence/README.md.txt) says SDK use does not require enabling related RobotAssist functions and separately requires SDK authorization. No evidence establishes that xCoreSDK requires RCI. **Do not enable RCI** or interpret its disabled flag as absence of an SDK license.

## 2. Position safety bounds and enable states

The live joint-position limiting function is **ENABLED**. Normal and collaborative position bounds are **-360 to +360 degrees on all six joints**, equivalent to +/-2*pi rad. These are no longer merely cached values.

Preserve the supplied label **collaborative mode**. Its mapping to historical reduced_* fields and the currently selected mode/input are not independently established. Both displayed position columns are identical. Separate soft-limit settings were not supplied: cached soft_limit.enable=true and +/-360 remain cached-only, not a newly verified complete effective-bound intersection.

| Safety function | Live state | Live displayed values |
|---|---|---|
| Joint position | ENABLED | Normal/collaborative +/-360 degrees, J1-J6 |
| Joint speed | DISABLED | 0 deg/s; **not an active zero-speed bound** |
| Joint torque | DISABLED | [182,182,182,182,54,54] Nm |
| Robot TCP linear/angular speed limits | DISABLED | Normal 0 mm/s, 100 deg/s; collaborative 0 mm/s, 50 deg/s |
| Robot elbow linear/angular speed limits | DISABLED | Normal 1000 mm/s, 100 deg/s; collaborative 500 mm/s, 50 deg/s |
| Collaborative speed restriction | DISABLED | J1-J6 [45,37.5,45,45,56.25,56.25] deg/s; TCP 500 mm/s |
| Overall-power limitation | DISABLED | Numeric live value not supplied |
| Momentum limitation | DISABLED | Numeric live value not supplied |
| Drag limitation | DISABLED | - |
| Collision protection | ON | Thresholds/response not supplied |
| Gravity compensation | ON | - |

Disabled displayed values are not normal planner limits or proof that no other guards apply. Per-joint power limits, full safety checksum, collision thresholds, virtual walls and monitor settings were not supplied live. No complete active-safety verification is claimed.

## 3. Live motion and dynamics settings

| Parameter | Live configured value |
|---|---|
| Maximum joint velocity | [180,180,234,240,240,240] deg/s |
| Maximum joint acceleration | [1500]x6 deg/s^2 = [26.179938780]x6 rad/s^2 |
| Maximum joint jerk | **[5000]x6 deg/s^3 = [87.266462600]x6 rad/s^3** |
| Joint stiffness gain | [1]x6 |
| Acceleration ratio / speed smoothing coefficient | 1 / 1 |
| Acceleration rise / jog acceleration rise time | 0.4 s / 0.4 s |
| Target-position-arrival acceleration rise time | 0.01 s |
| Look-ahead parameter count | 1000 |

Live dynamics: feedforward **all axes ON**; dynamics constraints **ENABLED**; source **factory-estimated dynamics parameters**; vibration suppression **DISABLED**; friction parameters populated, without individual numbers supplied. These establish configuration, not dynamic feasibility.

The existence and current value of **5000 deg/s^3 are resolved**. Its applicability to SDK RT JointPosition remains unknown. Cached JERK_LIMIT_JOINT=0 is not live-confirmed and does not establish a bypass. Gain/smoothing coefficient 1 and rise-time values do not establish an RT filter transfer function. RT applicability of speed, acceleration and enabled dynamics constraints also requires documentation.

## 4. Continuous C2 comparison of all six commands

The unchanged [timing definition](execution_timing_design.md) and [full-precision audit](artifacts/run_20260913T073830/audit/command_audit.csv) provide continuous polynomial position/derivative extrema, not merely 100-knot or 1 ms sample checks. All 56 immutable package hashes were reverified and every audit trajectory hash matched. Existing numerical extrema are reused because inputs/convention are unchanged; this is not a new execution or formal interval certificate.

| Command | Max speed rad/s | Max acceleration rad/s^2 | Max jerk rad/s^3 | Live-parameter comparison |
|---|---:|---:|---:|---|
| path_0001 R | 1.218401918 | 8.544676653 | 436.5893698 | Position/speed/acceleration fit; jerk exceeds J1-J5 |
| path_0001 D | 1.218450938 | 8.545713007 | 436.5893698 | Position/speed/acceleration fit; jerk exceeds J1-J5 |
| path_0003 R | 1.341213056 | 13.62377829 | 1201.588129 | Position/speed/acceleration fit; jerk exceeds J1-J5 |
| path_0003 D | 1.339252187 | 13.67588864 | 1206.350087 | Position/speed/acceleration fit; jerk exceeds J1-J5 |
| path_0006 R | 0.4149517227 | 7.538333269 | 694.7670116 | Position/speed/acceleration fit; jerk exceeds J1-J5 |
| path_0006 D | 0.4149517227 | 7.513691293 | 692.2474193 | Position/speed/acceleration fit; jerk exceeds J1-J5 |

Every joint fits the **live displayed +/-360-degree position safety bounds**. Speeds and accelerations are below corresponding **live configured motion parameters**. Every command exceeds the **live configured jerk value on joints 1-5**. These are verified numerical comparisons to settings, not verified RT acceptance/rejection results.

If 5000 deg/s^3 is a hard bound on the unfiltered host C2 reference, all six fail that particular constraint as defined. If filtering enforces the bound, the resulting path may differ from the experimental reference. Neither interpretation is assumed. A configured planner value alone is insufficient to classify the commands as failing verified RT constraints.

## 5. path_0006 position conclusion

Both C2 conditions reach J1 = **3.1425140906387883 rad = 180.052794454 degrees**. Against live +360 degrees, upper margin is **3.140671216540798 rad = 179.947205546 degrees**. Therefore **path_0006 does not violate the currently displayed joint-position safety bound**, using the stated offline/controller coordinate convention.

The maximum stored knot is 3.1415998935699463 rad; checking only knots misses the larger continuous maximum. C2 exceeds the Jetson 3.1416-rad model bound by **0.000914090638788 rad**, and exact pi by **0.000921437048995 rad**. Keep this model/controller discrepancy explicit in prospective modeling/eligibility. No clipping, wrapping, retiming or trajectory edits occurred.

Full physical eligibility remains unresolved: separate soft/mechanical bounds and reserve, joint-zero/turn mapping, dynamics/load, collision, RT guards/filters and timing are independent requirements.

## 6. Free-space endpoint, load and frames

The first experiment is **free-space joint motion**, evaluated at **xMateCR7_link6 / robot joint motion**. Paper-plane, pen-tip, ink, drawing-force and contact calibration are **not prerequisites**. The drawing tool is not assumed to be the experimental endpoint.

Actual attached mass/CoM/inertia and hardware clearance remain relevant to later dynamics/collision checks. Do not infer zero physical load from cached tool0. Current TCP/workobject selections, calibration/zero conventions and installed load were not supplied live; saved identity tool0/wobj0 and zero load remain historical.

Joint-only stationary variability does not require pen/TCP calibration. If controller Cartesian pose is used quantitatively, first identify its frame and link6 offset; otherwise keep that channel optional/unverified. SDK q_m order/radian semantics and later Jetson zero/turn/frame registration are separate from drawing calibration.

## 7. Gate A: passive stationary acquisition

**First connection is not ready. Vendor compatibility and lifecycle confirmation are the principal external blockers.** No repeat screenshots of supplied values are needed.

| Passive question | Evidence and remaining requirement |
|---|---|
| Exact SDK/firmware pairing | README says xCore >=3.0.1; CHANGELOG v0.5.1 says >=3.0.2; rt_0 does not reconcile them. Live firmware is 3.0.1.1.C89.20250306. Obtain explicit confirmation; select neither statement by assumption. |
| Connect and teardown | Robot header documents stop-before-disconnect and stopping a moving robot on Cobot destruction. Connection, subscription start/stop, exception unwinding and process/network failure effects on independent RL tasks remain unresolved. |
| Loaded versus running project | Loaded project is not proof of motion or exclusive ownership. Current task states, power state and control/operating modes were not supplied. Verify supported coexistence/idle prerequisites; a stationary arm alone is insufficient. |
| Subscription without power/mode/setters | startReceiveRobotState documents 1/2/4/8 ms or 1 s, max 1024-byte fields, up to 3 s first-frame wait, but not every operating-state prerequisite. The read-state example explicitly sets NrtCommand and later powers/commands motion. It does not prove a setter-free passive lifecycle and must not be run. |
| Without RCI or RT-motion enablement | README says related RobotAssist function enablement is unnecessary. Subscription and RT motion-controller creation are distinct APIs; the latter explicitly requires RT mode. This supports separate treatment, but exact passive prerequisites in the existing mode still need confirmation. Do not enable either to investigate. |
| Acquisition implementation | Stationary entry point remains blocked without an SDK backend. A reviewed adapter must implement the supported passive lifecycle, timing/errors and metadata. Subscription start/stop are intended acquisition operations; no unrelated setter is permitted. |

Specific vendor questions, without probing or sending messages in this task:

1. Is xCoreSDK-v0.5.1.rt_0.zip, SHA-256 **9b1f02410899d143340b53a745f62964550cbbe406d90d623402de361de0bb42**, supported for measured-state subscription on this XMC7-R850-W7G3B4C-S5 / XBC_XMATE / ROKAE_RSC with **3.0.1.1.C89.20250306**? Which compatibility statement governs?
2. What construct/connect/subscribe/read/unsubscribe/disconnect/destruct sequence works with RCI disabled, without power, mode, RT-motion enable or controller setters? In which existing states and at which subscription rates?
3. Can that lifecycle, timeout, exception, process exit or network failure affect a loaded/independently running Artist_movement_finish task or RobotAssist ownership? What idle/single-owner and teardown conditions are guaranteed?

One future owner remains preferred. A compatibility answer alone does not authorize connection: lifecycle confirmation, supported then-current state, reviewed passive backend and explicit authorization remain necessary. Trajectory jerk/overshoot, moving-load validation and Stage B results are later-motion gates, not requirements to establish whether stationary joints can be read.

Evidence: [README](evidence/README.md.txt), [CHANGELOG](evidence/CHANGELOG.md.txt), [robot header](evidence/include_rokae_robot.h.txt), [RT header](evidence/include_rokae_motion_control_rt.h.txt), [read-state example](evidence/example_read_robot_state.cpp.txt).

## 8. Gate B: later RT JointPosition motion

Additional motion requirements remain separate from Gate A:

- **RT enforcement:** identify controlling jerk parameter, constrained signal (host, filtered reference, other trajectory), and exact response: reject/filter/stop/fault/other. Confirm speed/acceleration/dynamics-constraint applicability. The 5000 value itself no longer needs verification.
- **Filtering:** establish SDK/controller transfer functions, clipping and supported identity behavior. Coefficient 1 or a high cutoff does not prove bypass.
- **Command timing/application:** qualify 1 ms deployment, deadlines/queue age, application origin, q_c semantics, final q99/setFinished ordering and host/controller uncertainty. Passive reads also need truthful timing; command-specific questions arise with motion.
- **Physical feasibility:** effective mechanical/soft/safety bounds/reserve, actual load/CoM/inertia, guard/collision behavior, clearance, dynamics/torque/power evaluation and joint-zero/turn/link6 registration. Disabled safety pages do not remove actuator/dynamics constraints.
- **Characterization:** stationary Stage A, then separately reviewed/authorized neutral Stage B; freeze measurability metrics/thresholds and matching C2 analysis before R/D collection. No six-command motion or target reselection is authorized.

All six commands remain conditionally assessed, not fully physically eligible.

## 9. Additional RobotAssist evidence

**No repeat screenshots are necessary for supplied live values.** Another jerk screenshot cannot establish RT enforcement; another license screenshot cannot resolve binary compatibility. Vendor compatibility/lifecycle answers should come first.

Before a subsequently authorized passive session, record then-current serial/session time, task running/stopped status, active control/operating mode, power state and other control owners to check vendor prerequisites. These are values to observe, not settings to change. Text/read-only export is sufficient; screenshots are optional. No screenshot request is made now.

Before later motion, obtain missing effective soft/mechanical bounds/reserve, actual load/CoM/inertia, calibration/zeros and relevant RT/collision configuration. Cartesian frame identity is necessary only if that optional channel is used quantitatively. No paper-plane or pen-contact evidence is required.

## 10. Decision and retained per-joint evidence

Resolved live: model identifier/cabinet/safety module and firmware; enabled normal/collaborative +/-360 position safety bounds; reported safety enable states; configured joint speed/acceleration/jerk and supplied motion/dynamics settings; SDK license; RCI disabled; gravity compensation and collision protection ON. Path_0006 fits the displayed live J1 safety bound.

Passive unknowns: exact SDK pairing, setter-free subscription prerequisites, interference/teardown lifecycle, then-current idle/ownership state and reviewed acquisition backend. Additional motion unknowns: RT enforcement/filter/application behavior, dynamics/clearance/effective limits, registration, Stage B and measurability. **Vendor confirmation is now the principal external blocker.** No verified RT-failure or passive-readiness finding is asserted.

Per-joint continuous extrema retained for traceability (numerical root searches, not formal interval certification):

| Command | Joint | C2 minimum (rad) | C2 maximum (rad) | Max speed (rad/s) | Max acceleration (rad/s?) | Max jerk (rad/s?) |
|---|---:|---:|---:|---:|---:|---:|
| path_0001 R | 1 | -1.9740766971 | -0.988967359066 | 1.21840191775 | 8.54467665261 | 410.736731544 |
| path_0001 R | 2 | -0.0372841172279 | 0.40629866368 | 0.211824145466 | 4.82230897423 | 436.589369824 |
| path_0001 R | 3 | 0.440827068945 | 0.816007974042 | 0.245672712427 | 2.95731449068 | 265.906158619 |
| path_0001 R | 4 | -0.909639713898 | 0.0143844923005 | 0.729236135509 | 4.53280828226 | 221.046615704 |
| path_0001 R | 5 | 0.0961176306009 | 0.498256236624 | 0.321463294663 | 2.31083805543 | 112.434876952 |
| path_0001 R | 6 | -2.00789165497 | -2.00789165497 | 0 | 0 | 0 |
| path_0001 D | 1 | -1.9740766971 | -0.988967359066 | 1.21845093796 | 8.54571300737 | 410.789051013 |
| path_0001 D | 2 | -0.0373321956208 | 0.40630358414 | 0.211838025893 | 4.82230897423 | 436.589369824 |
| path_0001 D | 3 | 0.440654359355 | 0.816085856035 | 0.245752058344 | 2.95731449068 | 265.906158619 |
| path_0001 D | 4 | -0.909639713898 | 0.0143844923005 | 0.729179852867 | 4.53280828226 | 221.046615704 |
| path_0001 D | 5 | 0.0961176306009 | 0.498256236624 | 0.321658063427 | 2.31083805543 | 112.434876952 |
| path_0001 D | 6 | -2.00790063143 | -2.00788738734 | 0.000129296891488 | 0.0029946993271 | 0.194992279495 |
| path_0003 R | 1 | -0.404623149298 | 0.346318137215 | 0.82314898837 | 13.6237782926 | 1201.58812882 |
| path_0003 R | 2 | -0.524024669493 | 0.0930408760905 | 0.25461480739 | 3.97783955548 | 361.396127122 |
| path_0003 R | 3 | -0.344022962779 | 0.0209529232234 | 0.150864002608 | 2.24121717482 | 196.415611918 |
| path_0003 R | 4 | -0.142080643668 | 1.38000323875 | 1.34121305616 | 6.24527737149 | 576.91962241 |
| path_0003 R | 5 | -1.18960127134 | -0.797692060471 | 0.16860872764 | 2.96275146619 | 263.117271692 |
| path_0003 R | 6 | 1.68705998868 | 1.68706036598 | 1.62273645401e-06 | 2.95576890443e-05 | 0.00208203577995 |
| path_0003 D | 1 | -0.404950520375 | 0.346318137215 | 0.82314898837 | 13.6758886366 | 1206.35008698 |
| path_0003 D | 2 | -0.524024669493 | 0.0930408760905 | 0.254607535116 | 3.99270835809 | 362.820807345 |
| path_0003 D | 3 | -0.344022962779 | 0.0209529232234 | 0.150864002608 | 2.30814359493 | 202.862428383 |
| path_0003 D | 4 | -0.142084188241 | 1.38000323875 | 1.33925218729 | 6.24527737149 | 576.91962241 |
| path_0003 D | 5 | -1.18960127134 | -0.797692060471 | 0.16860872764 | 3.04604382274 | 271.347983066 |
| path_0003 D | 6 | 1.68700547883 | 1.68706950387 | 0.000727256606164 | 0.0142201020774 | 0.958571073075 |
| path_0006 R | 1 | 2.61535442113 | 3.14251409064 | 0.2842002502 | 7.53833326886 | 694.767011627 |
| path_0006 R | 2 | -0.409633937146 | 0.00742216325007 | 0.163016601266 | 3.21704313482 | 317.176597736 |
| path_0006 R | 3 | -0.667667143775 | -0.414971417976 | 0.157035718953 | 2.12895750817 | 160.908833981 |
| path_0006 R | 4 | 0.828074336052 | 1.6374881807 | 0.414951722698 | 5.7583011427 | 471.885081383 |
| path_0006 R | 5 | -0.516606157398 | 0.140552043915 | 0.367071722836 | 5.12989592204 | 419.272752924 |
| path_0006 R | 6 | -0.896199946554 | -0.896199698298 | 8.11368227005e-07 | 1.47788445221e-05 | 0.00104101788998 |
| path_0006 D | 1 | 2.61535442113 | 3.14251409064 | 0.283419981467 | 7.51369129288 | 692.247419267 |
| path_0006 D | 2 | -0.409789780192 | 0.00731617154846 | 0.161445203072 | 3.25402311736 | 320.733757547 |
| path_0006 D | 3 | -0.667667143775 | -0.41508840443 | 0.155519554035 | 2.17139585165 | 165.907811756 |
| path_0006 D | 4 | 0.828074336052 | 1.6376748352 | 0.414951722698 | 5.7583011427 | 471.885081383 |
| path_0006 D | 5 | -0.516606157398 | 0.140552043915 | 0.367071722836 | 5.12989592204 | 419.272752924 |
| path_0006 D | 6 | -0.896215633189 | -0.896185794121 | 0.000201889556416 | 0.0041933838778 | 0.396331880831 |

NEEDS VENDOR COMPATIBILITY CONFIRMATION
