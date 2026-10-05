> CURRENT PROTOCOL: [python_v050_nrt_experiment_plan.md](python_v050_nrt_experiment_plan.md) supersedes prior RT/30 s physical-execution assumptions. The experiment now uses exact waypoint sequences under a fixed Python v0.5.0 NRT policy, geometric metrics, and initially path_0003 with 5 R + 5 D trials. Earlier 10/30 s analyses remain numerical/historical references, not actual NRT timing. Windows SDK import is currently blocked; no hardware is authorized.

# Version-specific SDK v0.5.0 capability audit

## Decision and local search

**C++ v0.5.0 is selected prospectively; v0.5.1.rt_0 is excluded from the experiment.** Its release-note minimum xCore >=3.0.1 is consistent with the live 3.0.1.1.C89.20250306 firmware. This version choice removes the known v0.5.1 >=3.0.2 minimum conflict; it does not establish an untested connection lifecycle.

Search covered user-profile filenames, Desktop USB materials, the Downloads USB copy, Documents and C:/Rokae. Only C: was mounted. No complete C++ v0.5.0 Windows x64 development package was found. Both copies of `xCoreSDK_cpp-v0.5.0.zip.c96465c9-15cf-4f70-a881-2f76ee680e6e.crdownload` are 75,497,472 bytes and fail ZIP validation; they are not installable packages. No archive repair, resumed download, SDK substitution or mixed-version headers were used. [Search/runtime evidence](evidence/v050/search_and_runtime.json) records exact Unicode paths, sizes and hashes.

Primary incomplete C++ path:

`C:\Users\画家ロボット\Desktop\ROKAE_USBdata_20260107\sdk\2026_01_07_Readme\xCoreSDK_cpp-v0.5.0.zip.c96465c9-15cf-4f70-a881-2f76ee680e6e.crdownload`

SHA-256: `07bfef90cf5605384f37224126c3c64bd0b2573ec344dc329c6c71feff74916c`.

No complete exact C++ v0.5.0 headers/import library were available to establish its compiler/runtime requirements or enumerate its APIs. The MSVC requirements from v0.5.1 are not silently relabeled v0.5.0. The Python package's `xCoreSDK_python.lib` is not assumed to be a replacement C++ robot SDK development library.

## First fallback: intact Python v0.5.0

Exact path:

`C:\Users\画家ロボット\Desktop\ROKAE_USBdata_20260107\sdk\2026_01_07_xcoresdk_python-v0.6.0\2026_01_07_xcoresdk_python-v0.5.0\xcoresdk_python-v0.5.0.zip`

Archive SHA-256: `0d0d1db19fd3340712606531063f3ea0fc65c01a3226d0fb2878c4452ffb1832`. ZIP CRC validation passed. The Downloads copy has the same hash. [Package manifest](evidence/v050/package_manifest.json) records every member hash, including Windows DLL/PYD, stubs and examples. Parent folder labels do not change the actual selected package version.

This is the **Python v0.5.0 release dated 2025-01-08**, not proof of an identical C++ release/API. Its own README says xCore >=2.2.0; its version-specific CHANGELOG says >=2.3. The live 3.0.1.1 clears both stated minima. Thus this fallback does not have the previous above-firmware minimum conflict. That compatibility-range observation is not a hardware test or proof of every optional feature.

The archive contains `Release/windows/xCoreSDK_python.cp312-win_amd64.pyd` and `xCoreSDK.dll`. Static PE inspection confirms machine 0x8664 (x64), linker 14.16, and imports of **python312.dll**, **MSVCP140.dll**, **VCRUNTIME140.dll**, Windows/UCRT libraries and xCoreSDK.dll. This exact extension needs **64-bit CPython 3.12 and compatible VC++ 14.x/UCRT runtime**. The README's generic Python >=3.8 text does not make this cp312 binary loadable in Python 3.14. Linker version is observed binary metadata, not a declared supported C++ build matrix.

Only Python314 was found in the normal local Python installation directory; no Python312 runtime was identified by the profile search. No 3.12 environment was installed, SDK binary imported, DLL loaded, or example executed. The user made environment preparation conditional on required measured-state and RT APIs being available; a usable RT command interface is not established below. A dedicated 3.12 environment remains a concrete passive-runtime prerequisite, not an already completed setup.

## Version-specific capability matrix

All Python entries below come from this archive's **Windows** stubs/examples, not v0.5.1 or an online latest API. The extracted files are under `evidence/v050`.

| Required capability | v0.5.0 evidence / conclusion |
|---|---|
| Default robot construction | `xMateRobot()` declared, with separate IP constructor; includes six-axis CR7 family |
| connectToRobot | `connectToRobot(remoteIP: str, localIP: str='')` and error-dict overload declared |
| disconnectFromRobot | `disconnectFromRobot(ec: dict)` declared; doc explicitly says motion is stopped before disconnect |
| startReceiveRobotState | Declared on Robot_T_Collaborative_6; `timedelta`, list of field strings; **1/2/4/8 ms or 1 s**, max 1024 bytes; first-frame wait up to 3 s |
| stopReceiveRobotState | Declared on BaseRobot; stops receive and controller sending |
| updateRobotState | `updateRobotState(timeout: timedelta) -> int`; returned byte count, zero on timeout; parse/network exception documented |
| getStateData | Vector overload `getStateData(fieldName, PyTypeVectorDouble, size) -> int`; example uses size 6 for joints, 16 for matrix |
| Measured joint position | `RtSupportedFields.jointPos_m = 'q_m'`, six-axis array, radians |
| Measured joint velocity | No arm `jointVel_m`/`dq_m` subscribed field declared. Separate `jointVel(ec)` query exists, rad/s; it would be separately timed and is outside the minimal lifecycle. `exJointVel_m` is external-axis velocity, not arm velocity |
| End Cartesian pose | `tcpPose_m='pos_m'`: base-relative row-major 4x4 matrix; `tcpPoseAbc_m='pos_abc_m'` also listed. Exact configured end/TCP identity still requires provenance |
| Controller/sample timestamp | No sample timestamp in declared subscription fields. LogInfo.timestamp is a log date/time, not a state sample clock. NTP configuration/synchronization methods are not timestamp fields and will not be called |
| JointPosition | Data type declared, but existence alone is not a usable RT execution interface |
| getRtMotionController | Declared with unresolved return annotation `...`; says RT mode required. No typed usable controller class/method contract supplied |
| RT control-loop/send interface | No `setControlLoop`, RT `startMove`, RtControllerMode or documented per-cycle joint-position send method in these stubs/examples. A complete RT joint-position execution path **cannot be established** from this package |

Do not claim that static stubs prove a binary can never expose anything else. The concrete result is that no usable version-specific documented RT loop/send interface is available for implementation here. No C++ v0.5.1 signatures may fill that gap. Python's normal MoveAbsJ command API is not a substitute for the prescribed timed reference.

## Passive lifecycle and readiness

The v0.5.0 example sets a motion-control mode, powers on and starts a motion thread. It is not a passive smoke test and must not be run. The user-selected stationary/no-RL-owner procedure reduces the known disconnect-stop concern, but existing-power/mode support and hidden lifecycle effects are not established just by the example.

[Prepared passive source](passive_v050_prepared.py) uses only default construction, connect, subscribe, update/get q, unsubscribe and disconnect, with no SDK import or enabled hardware runner. [Prepared procedure](passive_v050_procedure.md) defines 1 s / ~10 s smoke and 8 ms / ~30 s characterization. Six synthetic tests passed, including no-setter call inspection and failures. This is prepared code, not an installed, runtime-verified capture tool.

Passive gate: not ready because the exact 3.12 runtime/binding and safe existing-state subscription/teardown are not verified. Later motion gate: exact C++ package missing; Python fallback lacks an established RT command interface. The version-minimum workaround is accepted, but these concrete package/API/lifecycle gaps remain.
