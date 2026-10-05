> CURRENT PROTOCOL: [python_v050_nrt_experiment_plan.md](python_v050_nrt_experiment_plan.md) supersedes prior RT/30 s physical-execution assumptions. The experiment now uses exact waypoint sequences under a fixed Python v0.5.0 NRT policy, geometric metrics, and initially path_0003 with 5 R + 5 D trials. Earlier 10/30 s analyses remain numerical/historical references, not actual NRT timing. Windows SDK import is currently blocked; no hardware is authorized.

# Time-constrained workaround report

The prospective **30.0-second physical timing amendment is implemented and numerically verified**. It preserves every R/D joint knot and path selection. All six references now have positive margins to the live configured speed, acceleration and jerk values. **No robot SDK was loaded, no connection or motion occurred, and no controller setting or original trajectory changed.**

The SDK version decision is accepted: **C++ v0.5.0 is the prospective choice; v0.5.1.rt_0 is excluded.** The current local materials do not yet provide a usable complete v0.5.0 RT implementation. The first fallback was inspected rather than silently substituting a newer version.

## SDK/package result

[sdk_v050_capability_audit.md](sdk_v050_capability_audit.md) records version-specific APIs, compiler/runtime evidence and exact archive/member hashes.

- C++ v0.5.0: only two identical incomplete `.crdownload` files found; 75,497,472 bytes, SHA-256 `07bfef90cf5605384f37224126c3c64bd0b2573ec344dc329c6c71feff74916c`, invalid ZIP. No complete Windows x64 headers/import-library package found in the searched laptop/USB-copy locations.
- Python v0.5.0: intact archive, SHA-256 `0d0d1db19fd3340712606531063f3ea0fc65c01a3226d0fb2878c4452ffb1832`, CRC check passed. Exact paths are in the capability audit and its evidence manifests.
- Python stubs provide connect/disconnect, subscribe/unsubscribe, update/get measured q, and optional pose. They explicitly support 1 s and 8 ms subscriptions. Arm dq is available as a separate query, not a declared subscription field; no sample timestamp is declared.
- JointPosition and an untyped getRtMotionController are declared, but no usable version-specific RT loop/send interface is established. Normal MoveAbsJ is not a substitute. This is an unavailable documented API path, not proof that an uninspected binary could never expose additional behavior.
- Windows fallback binary requires CPython 3.12 x64, compatible VC++ 14.x/UCRT and its matching xCoreSDK.dll. Only the existing Python 3.14 installation was identified. No 3.12 runtime or SDK was installed/loaded because the requested complete RT fallback capability was not established. This remains a concrete runtime preparation item if proceeding with passive-only Python acquisition separately.

The C++ v0.5.0 minimum >=3.0.1 matches the live firmware range prospectively; the Python release has its own older minima, also below the live firmware. The old v0.5.1 minimum-version conflict is no longer the selected-backend blocker. Missing package/API, runtime and existing-state lifecycle evidence take its place. No latest online API was substituted and no newer SDK was downloaded.

## Physical reference and verification

[physical_timing_amendment_30s.md](physical_timing_amendment_30s.md) is the current physical convention and analysis amendment. The numerical benchmark stays 100 samples/10 s. Physical execution uses the exact same 100 Q knots and T_i=3*t_i, C2 quintic interpolation, and paths 0001/0003/0006 unchanged for both R and D.

[physical_reference_30s.py](physical_reference_30s.py) reconstructs the scaled polynomials and finds continuous extrema. It checks the old C++ extrema, analytic 1/3, 1/9 and 1/27 derivative scaling, coefficient invariance and bit-exact knots. Position envelopes are unchanged mathematically, with floating-point discrepancies at most 2.220446049250313e-16 rad. Phase identity error is at most 1.9984014443252818e-15 rad. All 3600 Q values pass bit-exact checks, and all 56 package hashes verify.

Worst jerk is **44.67963285109179 rad/s^3** at **path_0003 D, J1**, compared with the live configured **87.26646259971648 rad/s^3**. Minimum margin is **42.58682974862469 rad/s^3**. Degree values and every joint's maximum/minimum margin are below. These are numerical continuous-extrema results, not formal interval certification or dynamic feasibility.

| Command | Joint | Max jerk rad/s^3 | Max jerk deg/s^3 | Minimum margin rad/s^3 | Minimum margin deg/s^3 |
|---|---:|---:|---:|---:|---:|
| path_0001 R | 1 | 15.2124715386629 | 871.610415128269 | 72.0539910610536 | 4128.38958487173 |
| path_0001 R | 2 | 16.1699766601328 | 926.471417450658 | 71.0964859395836 | 4073.52858254934 |
| path_0001 R | 3 | 9.84837624514103 | 564.270393903478 | 77.4180863545755 | 4435.72960609652 |
| path_0001 R | 4 | 8.18691169273871 | 469.075487240233 | 79.0795509069778 | 4530.92451275977 |
| path_0001 R | 5 | 4.16425470194228 | 238.594219238801 | 83.1022078977742 | 4761.4057807612 |
| path_0001 R | 6 | 0 | 0 | 87.2664625997165 | 5000 |
| path_0001 D | 1 | 15.2144092967646 | 871.721440489215 | 72.0520533029519 | 4128.27855951079 |
| path_0001 D | 2 | 16.1699766601328 | 926.471417450658 | 71.0964859395836 | 4073.52858254934 |
| path_0001 D | 3 | 9.84837624514103 | 564.270393903478 | 77.4180863545755 | 4435.72960609652 |
| path_0001 D | 4 | 8.18691169273871 | 469.075487240233 | 79.0795509069778 | 4530.92451275977 |
| path_0001 D | 5 | 4.16425470194228 | 238.594219238801 | 83.1022078977742 | 4761.4057807612 |
| path_0001 D | 6 | 0.00722193627760784 | 0.413786468619349 | 87.2592406634389 | 4999.58621353138 |
| path_0003 R | 1 | 44.5032640304672 | 2549.84920350214 | 42.7631985692492 | 2450.15079649786 |
| path_0003 R | 2 | 13.3850417452455 | 766.906400608986 | 73.881420854471 | 4233.09359939101 |
| path_0003 R | 3 | 7.27465229326487 | 416.806873829242 | 79.9918103064516 | 4583.19312617076 |
| path_0003 R | 4 | 21.3673934226041 | 1224.26146231081 | 65.8990691771124 | 3775.73853768919 |
| path_0003 R | 5 | 9.74508413672447 | 558.352192034201 | 77.521378462992 | 4441.6478079658 |
| path_0003 R | 6 | 7.71124362945588e-05 | 0.00441821714764965 | 87.2663854872802 | 4999.99558178285 |
| path_0003 D | 1 | 44.6796328510918 | 2559.95439256162 | 42.5868297486247 | 2440.04560743838 |
| path_0003 D | 2 | 13.4378076794416 | 769.929665940491 | 73.8286549202749 | 4230.07033405951 |
| path_0003 D | 3 | 7.51342327344725 | 430.487443263895 | 79.7530393262692 | 4569.5125567361 |
| path_0003 D | 4 | 21.3673934226041 | 1224.26146231081 | 65.8990691771124 | 3775.73853768919 |
| path_0003 D | 5 | 10.049925298751 | 575.818304040185 | 77.2165373009655 | 4424.18169595982 |
| path_0003 D | 6 | 0.0355026323361267 | 2.03415099446474 | 87.2309599673803 | 4997.96584900554 |
| path_0006 R | 1 | 25.7321115417486 | 1474.34138930207 | 61.5343510579678 | 3525.65861069793 |
| path_0006 R | 2 | 11.7472813976193 | 673.06964483613 | 75.5191812020972 | 4326.93035516387 |
| path_0006 R | 3 | 5.95958644375207 | 341.459150870373 | 81.3068761559644 | 4658.54084912963 |
| path_0006 R | 4 | 17.4772252364159 | 1001.37124364616 | 69.7892373633006 | 3998.62875635384 |
| path_0006 R | 5 | 15.5286204786524 | 889.724415087204 | 71.7378421210641 | 4110.2755849128 |
| path_0006 R | 6 | 3.85562181472785e-05 | 0.00220910857382477 | 87.2664240434983 | 4999.99779089143 |
| path_0006 D | 1 | 25.6387933061885 | 1468.99464825287 | 61.627669293528 | 3531.00535174713 |
| path_0006 D | 2 | 11.8790280572982 | 680.618172400676 | 75.3874345424183 | 4319.38182759932 |
| path_0006 D | 3 | 6.14473376875673 | 352.067311181277 | 81.1217288309598 | 4647.93268881872 |
| path_0006 D | 4 | 17.4772252364159 | 1001.37124364616 | 69.7892373633006 | 3998.62875635384 |
| path_0006 D | 5 | 15.5286204786524 | 889.724415087204 | 71.7378421210641 | 4110.2755849128 |
| path_0006 D | 6 | 0.0146789585492934 | 0.841042372521989 | 87.2517836411672 | 4999.15895762748 |

Complete position, speed, acceleration, jerk and margins: [continuous_demands_30s.csv](artifacts/physical_30s_v050/continuous_demands_30s.csv). Numerical checks: [verification.json](artifacts/physical_30s_v050/verification.json). Six derived physical-knot CSVs reside beside those files; the original package was not edited.

path_0006's maximum remains 3.1425140906387883 rad: it fits the live displayed +/-360-degree safety bound but exceeds the Jetson 3.1416 model bound. This discrepancy is retained, not fixed by clipping or target replacement.

## Gate A: passive measured-state smoke using v0.5.0

**Not ready to execute yet.** The intact Python package documents the required passive API/rates, and [prepared lifecycle source](passive_v050_prepared.py) uses only connect, subscribe, update/read q, unsubscribe and disconnect. It has no SDK import or hardware runner. [Operator procedure](passive_v050_procedure.md) defines smoke at 1 s for approximately 10 s, then separately authorized characterization at 8 ms for approximately 30 s.

The operator prerequisites are controller/RobotAssist ON, stationary arm, Artist_movement_finish and all independent RL motion tasks stopped, no other motion owner, RCI OFF, and no power/mode/RT/setter actions. Servo OFF is preferred only if supported in that existing state. The v0.5.0 example changes mode and moves, so it does not establish that support. If subscriptions require a forbidden action, fail closed; no automatic workaround is implemented.

Remaining concrete passive items: dedicated verified CPython 3.12 x64/binary dependencies; checked binding return/error semantics; safe existing-power/mode subscription and teardown/partial-failure handling; reviewed persistent logger/metadata wiring and a separately enabled, explicitly authorized runner. The new no-RL-owner procedure reduces disconnect interference risk but does not prove undocumented lifecycle behavior. Six synthetic tests verify prepared call order, no setters, precondition blocking, timeout/malformed-data cleanup, writer failure and scaled-reference behavior. No hardware smoke or characterization ran.

## Gate B: later 30 s RT motion

**Not ready.** First obtain a complete exact C++ v0.5.0 Windows x64 package with its version-specific RT API, or an established equivalent RT interface in the permitted fallback. Do not borrow v0.5.1 headers or execute blended NRT moves instead.

The 30 s host reference no longer deliberately exceeds the configured jerk number, so testing an over-limit host path is not part of the plan. RT parameter semantics, command application, safe completion/abort, actual timing, dynamic/load/collision feasibility and effective bounds remain relevant. Filtering and physical tracking are to be measured for the frozen reference, not optimized by altering the trajectory. A usable interface, acquisition evidence and separately reviewed neutral Stage B precede any physical R/D run.

Physical comparisons must use q_30(T); Section V's standardized 10 s jerk remains a different metric. The amendment supplies prospective paper wording, time/derivative transforms and a separate physical evaluator; the immutable benchmark analysis is not patched. No drawing-contact calibration is introduced for the free-space link6 experiment.

## Deliverables and limits

Added the version-specific capability audit, prepared passive lifecycle/procedure, 30 s evaluator, continuous demand artifacts, analysis/timing amendment and tests. Existing design documents now point to this prospective amendment; their 10 s/v0.5.1 material remains explicitly historical. The original offline C++ executable still performs a labeled 10 s benchmark audit and has no physical execution backend. Its generated 10 s audit files must not be used as current physical commands.

The completed work removes the selected v0.5.1/firmware mismatch and the deliberate configured-jerk exceedance. It does not manufacture a missing v0.5.0 RT API or authorize an unverified passive runtime. The final classification concerns the unavailable complete version-specific package/API path, despite the intact passive-capable Python archive.

V0.5.0 PACKAGE/API UNAVAILABLE
