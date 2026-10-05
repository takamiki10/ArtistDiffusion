# SmartJoint CSV preparation — offline only, 2026-09-14

Event-validation update: [SmartJoint event-order investigation](smartjoint_event_order_review.md)
separates warning diagnostics from reported execution progress. The original
preparation findings below remain historical; see that report for the latest
event-validation change and offline tests.

The software is prepared and numerically validated. No robot object was constructed,
no controller connection was opened, and no motion/controller-setting call was made.
Only data objects from the pinned local xCoreSDK-Python v0.5.0 were constructed in
the optional binding audit. Physical 508-row execution has not been tested here.

## 1. Existing runner and compatibility decision

The appropriate execution basis is `C:\RokaeSDK\path_0003_nrt_runner.py`, which
inherits `PreparedSession` from `neutral_v050_nrt_runner.py`. Its current code is
more relevant than older README statements about blocked SDK imports or 30-second
host references. Existing saved path_0003 runs include failures to confirm measured
completion; this report does not describe those as successful experiments.

**C: the current fixed experiment runner needs a CSV execution wrapper.** Its
underlying NRT lifecycle is reusable. A file-format conversion alone is insufficient:

- `load_trial()` selects SHA-pinned A/B JSON files, not a supplied CSV filename.
- Trial selection is a frozen 10-trial R/D schedule, not five repeats of this drawing.
- Joint serialization and command validation require exactly 100 rows.
- Only one `moveAppend` attempt and one `cmdID` are tracked.
- Event indices are limited to 0–99, and command 99 receives the final zone.

The new wrapper reuses the command builder and the serial SDK lifecycle, q_m
acquisition, stationarity checks, error/event validation, START-boundary review,
any-joint motion observation, and measured endpoint criterion. It does not use
the neutral trajectory generator. The existing path_0003 script is unchanged.
The shared neutral module has optional count arguments (default still 100) and
two override hooks for append-attempt limits and event-completion construction.
Its original 100-waypoint behavior passes regression tests.

Other inspected loaders: `nrt_v050/prepare_nrt.py` and
`physical_reference_30s.py` expect pinned 100-row `time_seconds,q1,...,q6` CSVs.
The latter creates historical interpolated references and is not used by this
adapter. No NPZ trajectory loader was found in the current Python code.

## 2. Exact source and CSV findings

Expected source location on this laptop and a copied setup:
`C:\RokaeSDK\SmartJoint_Data_diffusion.csv`.
The user moved it from Downloads during inspection. Its bytes were not edited.

SHA256:

```text
3a592a1de99a63abe97182fa6dc7260dc1d3402f42685b6f522cbd574303ba9f
```

Columns, in order:

```text
Timestamp,TouchType,joint1,joint2,joint3,joint4,joint5,joint6,OriginalStatus
```

There are exactly **508 complete data rows**, with **300 Pen + 208 Air**.
`joint1` through `joint6` are the six candidate commanded values, in that order.
All 3,048 values are finite and parse as binary64 without modification.

The CSV header does not declare units. Values are consistent with native radians
(J4 near -pi; J6 around 4 rad), and timestamps are consistent with seconds.
This is an interpretation, not proof from the header or an inspected original
producer. The run command therefore requires an explicit
`--confirm-native-radians-seconds` assertion by the operator. Do not use that
assertion unless native J1–J6 radians and seconds are confirmed. No conversion,
wrapping, joint-zero offset, or joint remapping is performed. In particular,
J6 is not wrapped into [-pi, pi].

Timestamps are strictly increasing, from 0.0 to 40.5. Air intervals are approximately
0.05 and Pen intervals approximately 0.10101010101; the full sequence is not a
uniform 508-point time grid. Small floating-point timestamp differences are retained.

Inclusive **one-based data-row** segmentation (CSV line number is data row + 1):

| Rows | Count | TouchType | Timestamp range |
|---|---:|---|---|
| 1–30 | 30 | Air | 0.0–1.45 |
| 31–130 | 100 | Pen, stroke 1 | 1.5–11.5 |
| 131–209 | 79 | Air | 11.55–15.45 |
| 210–309 | 100 | Pen, stroke 2 | 15.5–25.5 |
| 310–388 | 79 | Air | 25.55–29.45 |
| 389–488 | 100 | Pen, stroke 3 | 29.5–39.5 |
| 489–508 | 20 | Air | 39.55–40.5 |

`OriginalStatus` contains RECORDING_START (1), MOVING_FAST (147),
DRAWING_STROKE_1/2/3 (100 each), and END_STROKE_1/2/3 (20 each).
All labels and their original strings are preserved.

First joint vector, row 1 / source index 0:

```text
[-2.227512509127758, -0.18441560441027577, 2.2651860822247576,
 -3.1415984105959325, -0.6919914095179537, 4.055677012796968]
```

Final joint vector, row 508 / source index 507:

```text
[-2.385008290567847, -0.2107033771154584, 2.2279644800565017,
 -3.1415970725388194, -0.7029249867113442, 3.8981800656445262]
```

The first row is Air / RECORDING_START. The first 30 rows contain changing joint
targets and appear to be the original approach portion. The ending also has
changing Air targets. This supports preserving the complete sequence, but joint
values and labels alone do not establish pen clearance, contact, or collision safety.
The CSV begins at a particular recorded pose; it does not contain a transition
from every possible current robot pose to that pose. The wrapper prints measured
q0, the first vector, and their difference before SETUP. MoveAbsJ will plan the
current-pose-to-first-target move, followed by every CSV row in sequence.

## 3. NRT commands, batching, and timing

Every target is constructed as `JointPosition(list(original_row))`, then
`MoveAbsJCommand(position, 50, zone)` with `cmd.jointSpeed = 0.05`.
Every target and policy property is read back and checked before append/start.
The preserved policy is:

| Parameter | Value |
|---|---:|
| jointSpeed | 0.05 |
| speed | 50 |
| interior zone | 1 on source indices 0–506 |
| final zone | 0 only on source index 507 |

The local v0.5.0 stub documents joint positions in radians, speed in mm/s, and
zone radius in mm. Explicit jointSpeed retains the established joint-speed
fraction instead of relying on speed-band conversion. Zone does not constitute
a verified Cartesian path-tolerance guarantee for this drawing.

Six contiguous batches have sizes **100,100,100,100,100,8**. All six are appended
serially, with post-call stationarity checks, before one `moveStart`. Chunking
does not remove samples, change their order, create artificial final zones, or
follow the Pen/Air segmentation. Five batch boundaries cut the stored sequence
only for SDK transport. No append occurs after START, and no append is retried.

Evidence scope matters: the exact v0.5.0 Python stub declares homogeneous-list
overloads but no numeric append bound. The local C++ header documents 1–100
commands per call; `evidence/sdk_source_manifest.json` identifies that header as
v0.5.1.rt_0, not exact Python v0.5.0 evidence. The established runner has prior
100-command append experience. Using at most 100 per call is a conservative
transport choice; it does not prove six-batch acceptance or continuous blending
on this controller. The exact v0.5.0 `setMaxCacheSize` documentation describes
a separate planning cache (1–300, initially 30), not the append-list size. The
adapter never calls that setter. An append error or malformed/repeated cmdID
prevents START; partial acknowledgements are retained in the logs.

Each returned cmdID maps to its batch's original index range. Raw SDK event
payloads are retained, with additional source-row metadata. Batch-local event
indices are mapped to global source indices before the inherited checks for
ordering, duplicates, errors, remarks, and customInfo are applied.

**CSV timestamps are provenance only for this NRT policy.** There are no sleeps
based on CSV timestamps, no Python interpolation, and no requested 40.5-second
execution. Repeated targets remain repeated commands; their timestamp spacing
does not become a dwell. Controller planning/blending determines actual timing
and intermediate motion. Exact command values do not imply an identical
time-parameterized physical trajectory to the original experiment.

The existing 120-second completion timeout is unchanged. Whether 508 commands
complete within it is unknown. The path_0003 measured completion test is retained:
observed joint excursion at least 1e-4 rad, idle state, and at least 100 valid
samples over at least 1 second near the final target (each joint within 1e-3 rad,
span at most 1e-5 rad, gaps at most 50 ms). A terminal callback is not required by
that existing protocol. Acceptance is geometric endpoint/observation evidence,
not proof that every intermediate point was attained or that contact was correct.

## 4. Offline verification and limitations

`nrt_v050/evidence/smartjoint_csv_validation.json` contains the full numerical
audit, including all 507 signed adjacent joint deltas. Checks passed for strict
parsing/schema/count, finite six-joint rows, source hash, timestamp monotonicity,
first/final rows, all-row independent parsing, JSON round-trip, and pinned SDK
data-object round-trip. **3048/3048 binary64 values match bit for bit; maximum
absolute parsing difference is 0.0.** No robot object is needed for this check.

Position checks use the same saved normal/collaborative bounds in
`evidence/controller_live_verification.json`: **-360 to +360 degrees on J1–J6**.
All rows pass. That evidence file is hash-pinned and unchanged. These are the
project's recorded displayed bounds, not a fresh controller query or a complete
intersection of mechanical/soft/safety limits. The older Jetson model's 3.1416
bound is a distinct model convention; it is not used to wrap or clip this CSV.

| Joint | Maximum absolute adjacent delta, rad |
|---|---:|
| J1 | 0.005585163852861452 |
| J2 | 0.010484145799953437 |
| J3 | 0.007545770379125738 |
| J4 | 0.0000509808065971562 |
| J5 | 0.016708238656928054 |
| J6 | 0.005597057699174179 |

Six identical adjacent pairs are retained. In zero-based source indices:
`[187,188], [188,189], [217,218], [328,329], [367,368], [388,389]`.
Minimum nonzero adjacent maximum-joint difference is
`2.245096338615582e-06` rad. These may interact with controller close-point/zone
handling. No points were deleted or adjusted. Nonempty controller remarks retain
the existing policy-rejection behavior; hard errors fail execution.

Validation performed: syntax compilation; 12 new CSV/fake-lifecycle tests;
45 existing neutral tests; 7 existing path_0003 tests. **64 tests passed.**
Fake lifecycle tests cover six appends before one START, exact targets/zones,
event mapping through index 507, partial append failure, duplicate IDs, operator
denials, state/sampling faults, warnings, and measured completion. They do not
simulate real controller capacity or dynamics.

## 5. Files and exact commands

New files:

- `C:\RokaeSDK\smartjoint_csv_nrt_runner.py`
- `C:\RokaeSDK\tests\test_smartjoint_csv_runner.py`
- This report and `nrt_v050\evidence\smartjoint_csv_validation.json`

Modified: `C:\RokaeSDK\neutral_v050_nrt_runner.py`, with backward-compatible
count arguments and override hooks. Its prior bytes, baseline hashes, and diff
are in `build_offline\smartjoint_baseline\`. The CSV, path_0003 runner, controller
policy, and limit evidence hashes are unchanged.

Use the installed **Python 3.12.10 x64** via `py -3.12`. It was checked and used
successfully here. The copied embedded runtime's `_pth` still references old
machine paths, so these commands deliberately do not use it. On a new laptop,
copy the project including the pinned SDK DLL/PYD, prepared policy, both shared
runner scripts, and limit evidence; install the same interpreter version.

PowerShell, offline validation including SDK **data objects only**:

```powershell
Set-Location C:\RokaeSDK
py -3.12 .\smartjoint_csv_nrt_runner.py --offline --csv .\SmartJoint_Data_diffusion.csv --check-sdk-data
py -3.12 -m unittest discover -s tests -p 'test_smartjoint_csv_runner.py' -v
```

Omit `--check-sdk-data` for a pure standard-library audit with no vendor import.
Inspect `OFFLINE_NUMERIC_PASS`, the exact source SHA256, row/label counts,
`binary64_bitwise_equal`, `sdk_data_objects_bitwise_equal`, first/final vectors,
segments, adjacent deltas/duplicates, and `position_violations: []`. A numeric
PASS is not permission to bypass the physical approach and controller review.

**Prepared robot execution command — not run during this task:**

```powershell
py -3.12 .\smartjoint_csv_nrt_runner.py --run-trial --trial 1 --csv .\SmartJoint_Data_diffusion.csv --confirm-native-radians-seconds
```

The units/convention flag is an operator assertion, not an automatic conversion.
All interactive phrases must be typed exactly as printed:

1. `CONNECT SMARTJOINT TRIAL 1` before the single robot object/connection. The
   banner requires existing power ON, stationary arm, stopped other motion
   owners, RCI OFF, and physical review of the transition to the first target.
   SDK initialization may reset its queue; disconnection has documented motion
   effects. Neither connection nor cleanup is claimed to be operationally inert.
2. Identity must match `XMC7-R850-W7G3B4C-S5` with six joints. Power must already
   be ON; operate mode must be known and remain unchanged; operation must be idle.
   The runner subscribes q_m at 8 ms and establishes a stationary q0 window.
3. `SETUP SMARTJOINT TRIAL 1 <printed-hash>` after reviewing measured q0,
   first target, and current-to-first delta. This explicitly authorizes
   `setMotionControlMode(NrtCommandMode)` and `moveReset`; it does not set power,
   manual/automatic operate mode, safety settings, or RCI. Each call is followed
   by fresh stationarity observation. These are the existing gated setup actions,
   not automatic mode changes on launch.
4. `APPEND SMARTJOINT TRIAL 1 <printed-hash>` authorizes the six exact ordered
   batches. The runner rechecks source bytes, policy, command bytes, q0 stability,
   state, and logger/event health. Each append must succeed with a distinct cmdID
   and pass its post-call stationarity window. No partial batch sequence can start.
5. `START SMARTJOINT TRIAL 1 <printed-hash>` after READY and all six appends.
   The same pre-control checks run again, followed by one `moveStart` attempt.

Hashes are computed during the session and cannot be supplied ahead of time.
The three connected prompts have the inherited 300-second timeout, with q_m
logging continuing while input is pending. Initial settling has a 10-second
timeout. Premotion q0 drift must stay within 1e-5 rad; sampled stationary span
must be at most 1e-5 rad. The existing 50 ms freshness/gap rules and conditional
stationary START-boundary exception remain. RCI OFF is an operator prerequisite;
this wrapper does not query or enable RCI. There are no automatic retries,
resumes, power changes, alarm resets, or cache/safety adjustments.

After a start fault/timeout, cleanup unsubscribes, removes the watcher, and
disconnects as in the existing runner. It does not establish that the robot is
stationary. The operator's reviewed physical procedure remains necessary.

## 6. q_m logs and five identical repetitions

Each attempted run reserves a unique folder:

```text
C:\RokaeSDK\nrt_v050\smartjoint_sessions\trial_01_<UTC>_<unique-id>\
```

`q_m.jsonl` contains native measured joint vectors and host receipt/decode timing.
`source_rows.jsonl` contains all 508 source indices, original joint tokens,
timestamps, TouchType, and OriginalStatus. `commands.json`, `append_batches.json`,
and `motion_events.jsonl` associate commands/events with exact source rows.
The q_m log also records the latest validated event's source index and TouchType;
this is explicitly historical waypoint metadata, not a measured contact state
or a timestamp-based alignment of q_m to the source. With no events that field
remains null, while all original labels remain available in source_rows.jsonl.

`source.csv` is a byte-identical archived copy. `waypoints.f64le` and
`waypoints.json` retain loaded targets. `session.json`, `offline_validation.json`,
`controller_policy.json`, `authorizations.jsonl`, `calls.jsonl`,
`lifecycle.jsonl`, `exceptions.jsonl`, `final_stationary.json`, and
`final_status.json` retain provenance and the outcome. No physical logs were
created by this task; test logs used temporary fake sessions.

Run the command above once for trial 1. After reviewing its accepted outcome and
the next current-pose-to-first-target transition, invoke separate fresh processes
with `--trial 2`, then `3`, `4`, and `5`, using the same filename and flag. Each
repetition has all four interactive phrases with its own number/hash. Every
process enforces the same built-in source SHA256. No generation/conversion step
is needed between trials, and the wrapper does not automatically return the robot
to the first pose between them.

Reservations prevent accidental reuse of a trial number. A later trial requires
earlier trials to have `COMPLETED_ACCEPTED` and the same source hash. A failed,
aborted, or rejected trial stops progression for review; the software does not
delete reservations or silently retry. Do not run the five commands concurrently.
