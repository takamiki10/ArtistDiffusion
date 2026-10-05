"""Operator-supervised neutral NRT validation, pinned SDK v0.5.0.

Import is inert. Hardware entry requires --run-neutral plus operator gates.
SETUP, APPEND and START are independent authorizations. No automatic recovery.
"""
from __future__ import annotations

from collections import deque
from datetime import datetime, timedelta, timezone
from fractions import Fraction
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import queue
import struct
import sys
import threading
import time
import traceback
import uuid

ROOT = Path(__file__).resolve().parent
SDK_DIR = ROOT / "nrt_v050" / "sdk"
POLICY_PATH = ROOT / "nrt_v050" / "prepared" / "controller_policy.json"
POLICY_SHA256 = "0d465d14bbd750befbae60902bcb8d9bdf4b8a2d2b3b04f418e7a336b831ffbb"
SDK_HASHES = {
    "xCoreSDK.dll": "7b72eaf137121036b6fcc7bf585d71331452262cea6f6d4082b754326d794c02",
    "xCoreSDK_python.cp312-win_amd64.pyd":
        "0bef1547fc74617c8835b0bf7f1dd13bc175612fa7a5ad7827c51371f7426455",
}
REMOTE_IP = "192.168.2.160"
LOCAL_IP = "192.168.2.100"
SETTINGS = {
    "period_s": 0.008, "read_timeout_s": 0.250,
    "no_valid_state_timeout_s": 1.0,
    "settle_window_s": 1.0, "settle_min_samples": 100,
    "settle_max_gap_s": 0.050, "settle_span_rad": 1e-5,
    "q0_drift_rad": 1e-5, "final_residual_rad": 1e-5,
    "fresh_host_receipt_s": 0.050, "state_query_period_s": 0.250,
    "settle_timeout_s": 10.0, "completion_timeout_s": 120.0,
    "prompt_timeout_s": 300.0, "event_queue_capacity": 4096,
    "fresh_drain_limit": 4096,
}
EXPECTED_ROBOT_TYPE = "XMC7-R850-W7G3B4C-S5"
PRECONNECTION_PHRASE = "CONNECT FOR SUPERVISED NEUTRAL VALIDATION"
PREMOTION_APIS = frozenset({"setMotionControlMode", "moveReset", "moveAppend"})
CONTROL_APIS = PREMOTION_APIS | {"moveStart"}
CHECKED_APIS = CONTROL_APIS | {"robotInfo", "operateMode", "powerState", "operationState",
                               "setEventWatcher"}
SETUP_EVIDENCE = {
    "date": "2026-09-13", "source": "operator-reported live no-motion setup result",
    "robot_type": EXPECTED_ROBOT_TYPE, "joint_num": 6,
    "controller_version": "2.3.1.1.C89.20250306",
    "NrtCommandMode": "SUCCESS", "moveReset": "SUCCESS",
    "ec": {"ec": 0, "message": "success"}, "moveReset_duration_ms_approx": 64,
    "measured_q": "effectively unchanged within normal quantization/noise",
    "prior_probe_failure": "50 ms host gap rule applied across synchronous reset",
    "mode_getter_available": False,
}


class ValidationError(RuntimeError):
    """An invariant failed; append/start must not proceed."""


def require(condition, message):
    if not condition:
        raise ValidationError(message)


def stamp():
    return {"host_ns": time.perf_counter_ns(),
            "utc": datetime.now(timezone.utc).isoformat()}


def digest_bytes(data):
    return hashlib.sha256(data).hexdigest()


def digest_json(value):
    return digest_bytes(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                  ensure_ascii=True, allow_nan=False).encode("ascii"))


def joint_bytes(values):
    values = list(values)
    require(len(values) == 6, "six joints required")
    require(all(type(v) is float and math.isfinite(v) for v in values),
            "six finite Python binary64 floats required")
    return struct.pack("<6d", *values)


def matrix_bytes(rows, count=100):
    require(type(count) is int and count > 0 and len(rows) == count,
            f"exactly {count} waypoints required")
    return b"".join(joint_bytes(row) for row in rows)


def generate_neutral(q0, joint, direction, excursion_deg):
    """Asymmetric 100-sample triangular NEUTRAL validation probe.

    d_i=A*i/49 through index 49; d_i=A*(99-i)/50 thereafter.
    Assign the peak directly to A to avoid a multiply/divide round trip.
    End rows are copied bit for bit. Reject binary64 bound violations and
    collapsed adjacent targets, never clip. No R/D trajectory is changed.
    joint is one-based; direction must be the explicit literal '+' or '-'.
    """
    joint_bytes(q0)
    require(type(joint) is int and 1 <= joint <= 6, "select joint 1 through 6")
    require(direction in ("+", "-"), "explicit + or - direction required")
    require(type(excursion_deg) is float and math.isfinite(excursion_deg)
            and 0 < excursion_deg <= 0.25, "excursion must be >0 and <=0.25 degrees")
    amplitude = math.radians(excursion_deg)
    rows = []
    for i in range(100):
        row = list(q0)
        if i not in (0, 99):
            displacement = (amplitude if i == 49 else
                            amplitude * i / 49 if i < 49 else
                            amplitude * (99 - i) / 50)
            row[joint - 1] = q0[joint - 1] + (displacement if direction == "+" else -displacement)
        rows.append(row)
    validate_neutral(rows, q0, joint, direction, excursion_deg)
    return rows


def neutral_spacing_diagnostics(rows, q0, joint):
    """Offline arithmetic on generated binary64 targets, no target modification.

    Pairs are zero-based [from, to]. Report zero gaps explicitly rather than
    silently hiding them in the requested minimum-NONZERO statistic.
    """
    matrix_bytes(rows)
    joint_bytes(q0)
    require(type(joint) is int and 1 <= joint <= 6, "invalid diagnostic joint")
    values = [row[joint - 1] for row in rows]
    distances = [abs(b - a) for a, b in zip(values, values[1:])]
    nonzero = [v for v in distances if v > 0.0]
    minimum_nonzero = min(nonzero) if nonzero else None

    def measure(value):
        return {"rad": value, "deg": None if value is None else math.degrees(value)}

    return {"joint": joint, "count": len(rows), "pair_indexing": "zero-based [from, to]",
            "minimum_nonzero_adjacent": measure(minimum_nonzero),
            "minimum_adjacent_including_zero": measure(min(distances)),
            "maximum_adjacent": measure(max(distances)),
            "peak_excursion": measure(max(abs(v - q0[joint - 1]) for v in values)),
            "first_interior_displacement": measure(abs(values[1] - q0[joint - 1])),
            "minimum_adjacent_pairs": [[i, i + 1] for i, d in enumerate(distances) if d == min(distances)],
            "minimum_nonzero_adjacent_pairs": [[i, i + 1] for i, d in enumerate(distances) if d == minimum_nonzero],
            "zero_adjacent_pairs": [[i, i + 1] for i, d in enumerate(distances) if d == 0.0],
            "adjacent_selected_joint_rad": distances}


def validate_neutral(rows, q0, joint, direction, excursion_deg):
    matrix_bytes(rows)
    require(joint_bytes(rows[0]) == joint_bytes(q0) == joint_bytes(rows[99]),
            "endpoints must preserve q0 bits")
    bound = Fraction.from_float(math.radians(excursion_deg))
    changed = False
    for row in rows:
        for j, (q, base) in enumerate(zip(row, q0)):
            if j != joint - 1:
                require(struct.pack("<d", q) == struct.pack("<d", base),
                        "unselected joint changed")
            else:
                delta = Fraction.from_float(q) - Fraction.from_float(base)
                require(abs(delta) <= bound, "generated binary64 displacement exceeds excursion")
                require(delta >= 0 if direction == "+" else delta <= 0, "wrong direction")
                changed |= delta != 0
    require(changed, "excursion is not representable at measured q0")
    distances = [abs(row[joint - 1] - q0[joint - 1]) for row in rows]
    require(all(a < b for a, b in zip(distances[:49], distances[1:50])),
            "neutral outward targets must strictly increase")
    require(all(a > b for a, b in zip(distances[49:99], distances[50:])),
            "neutral return targets must strictly decrease")
    require(all(a != b for a, b in zip(rows, rows[1:])),
            "consecutive neutral targets must differ")


def read_policy():
    raw = POLICY_PATH.read_bytes()
    require(digest_bytes(raw) == POLICY_SHA256, "frozen policy bytes changed")
    policy = json.loads(raw)
    require((policy["jointSpeed"], policy["speed"], policy["interior_zone"],
             policy["final_zone"]) == (0.05, 50, 1, 0), "unexpected frozen policy")
    return raw, policy


def construct_commands(sdk, rows, tag, count=100):
    """Data objects only. Does not construct a robot or communicate with it."""
    matrix_bytes(rows, count)
    require(type(tag) is str and tag.isascii() and 0 < len(tag) <= 64, "invalid batch tag")
    commands = []
    for i, row in enumerate(rows):
        position = sdk.JointPosition(list(row))
        require(joint_bytes(position.joints) == joint_bytes(row), "JointPosition changed target")
        cmd = sdk.MoveAbsJCommand(position, 50, 0 if i == count - 1 else 1)
        cmd.jointSpeed = 0.05
        cmd.customInfo = f"{tag}:k{i:03d}"
        commands.append(cmd)
    validate_commands(sdk, commands, rows, tag, count)
    return commands


def validate_commands(sdk, commands, rows, tag, count=100):
    matrix_bytes(rows, count)
    require(len(commands) == count, f"exactly {count} commands required")
    require(len({id(cmd) for cmd in commands}) == count, "aliased command objects")
    snapshots = []
    for i, (cmd, row) in enumerate(zip(commands, rows)):
        require(type(cmd) is sdk.MoveAbsJCommand, "batch must be homogeneous MoveAbsJCommand")
        target = list(cmd.target.joints)
        require(joint_bytes(target) == joint_bytes(row), f"target round-trip mismatch at {i}")
        require(list(cmd.target.external) == [], "unexpected external-axis target")
        require(type(cmd.jointSpeed) is float and cmd.jointSpeed == 0.05,
                f"jointSpeed mismatch at {i}")
        require(type(cmd.speed) is int and cmd.speed == 50, f"speed mismatch at {i}")
        require(type(cmd.zone) is int and cmd.zone == (0 if i == count - 1 else 1), f"zone mismatch at {i}")
        require(cmd.customInfo == f"{tag}:k{i:03d}", f"customInfo mismatch at {i}")
        snapshots.append({"index": i, "target_hex": [q.hex() for q in target],
                          "target_rad": target, "jointSpeed": cmd.jointSpeed,
                          "speed": cmd.speed, "zone": cmd.zone, "customInfo": cmd.customInfo})
    return snapshots


def load_exact_sdk_data_only():
    """Offline import of pinned local data binding; returns DLL handle to keep alive."""
    require(sys.version_info[:3] == (3, 12, 10) and struct.calcsize("P") == 8,
            "Python 3.12.10 x64 required")
    for name, expected in SDK_HASHES.items():
        require(digest_bytes((SDK_DIR / name).read_bytes()) == expected, f"SDK hash mismatch: {name}")
    module_name = "xCoreSDK_python"
    path = SDK_DIR / "xCoreSDK_python.cp312-win_amd64.pyd"
    require(module_name not in sys.modules, "SDK already loaded; use a fresh process")
    handle = os.add_dll_directory(str(SDK_DIR))
    spec = importlib.util.spec_from_file_location(module_name, path)
    require(spec is not None and spec.loader is not None, "missing pinned SDK loader")
    sdk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sdk)
    require(Path(sdk.__file__).resolve() == path.resolve(), "wrong SDK imported")
    return sdk, handle


def snapshot(value, sdk, depth=0):
    """Bounded payload copy, no file I/O or robot calls. Unknown types fail later."""
    if depth > 8:
        return {"unsupported_type": "nested_payload_limit"}
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        return value if math.isfinite(value) else {"nonfinite": repr(value)}
    if isinstance(value, sdk.PyErrorCode):
        # .get() is NOT used: it raises for unregistered std::error_code locally.
        return {"sdk_type": "PyErrorCode", "value": value.value(), "message": value.message()}
    if type(value) is dict and len(value) <= 128 and all(type(k) is str for k in value):
        return {k: snapshot(v, sdk, depth + 1) for k, v in value.items()}
    if type(value) in (list, tuple) and len(value) <= 128:
        return [snapshot(v, sdk, depth + 1) for v in value]
    return {"unsupported_type": type(value).__module__ + "." + type(value).__qualname__}


def success_error(ec):
    """Exact observed controller success schema; preserve other payloads then fail."""
    require(type(ec) is dict and set(ec) == {"ec", "message"}
            and type(ec["ec"]) is int and ec["ec"] == 0
            and type(ec["message"]) is str and ec["message"] == "success",
            f"unexpected SDK error structure/code: {ec!r}")


class SessionStore:
    """One synchronous disk writer (SDK owner/main thread), like passive capture.

    Line flushes preserve raw rows; fsync barriers precede control calls. Slow
    storage may cause gaps and will reject freshness/settling, never hide gaps.
    """
    STREAMS = ("q_m", "motion_events", "authorizations", "calls", "lifecycle", "exceptions")

    def __init__(self, folder):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=False)
        self.files = {}
        try:
            for name in self.STREAMS:
                self.files[name] = (self.folder / f"{name}.jsonl").open("x", encoding="utf-8")
        except BaseException:
            self.close()
            raise

    def write(self, stream, value):
        f = self.files[stream]
        f.write(json.dumps(value, ensure_ascii=True, allow_nan=False) + "\n")
        f.flush()

    def json(self, name, value):
        temp = self.folder / (name + ".tmp")
        with temp.open("w", encoding="utf-8") as f:
            json.dump(value, f, indent=2, ensure_ascii=True, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, self.folder / name)

    def barrier(self):
        for f in self.files.values():
            f.flush()
            os.fsync(f.fileno())

    def bytes(self, name, value):
        with (self.folder / name).open("xb") as f:
            f.write(value)
            f.flush()
            os.fsync(f.fileno())

    def close(self):
        failures = []
        for f in self.files.values():
            try:
                f.close()
            except Exception as exc:
                failures.append(repr(exc))
        if failures:
            raise OSError("closing session logs: " + "; ".join(failures))


class EventInbox:
    def __init__(self, sdk):
        self.sdk = sdk
        self.items = queue.Queue(maxsize=SETTINGS["event_queue_capacity"])
        self.failed = threading.Event()
        self.failure = None

    def callback(self, payload):
        host_ns = time.perf_counter_ns()
        try:
            self.items.put_nowait({"host_ns": host_ns, "payload": snapshot(payload, self.sdk)})
        except BaseException as exc:
            self.failure = type(exc).__name__  # no printing/I/O from SDK callback
            self.failed.set()


class Completion:
    """Separate diagnostics from non-diagnostic reported execution progress.

    The v0.5.0 MoveExecution key docs identify remark as additional information
    including close-point warnings. The watcher docs promise one callback thread,
    not waypoint-ordered diagnostic delivery. A remark-bearing event therefore
    cannot establish progress, even with reachTarget=True (observed on hardware).
    Progress contradictions remain fail-closed; they are not proof of physical
    backward execution because completion-delivery ordering is undocumented too.
    """
    def __init__(self, cmd_id, tag, start_ns, count=100):
        self.cmd_id, self.tag, self.start_ns = cmd_id, tag, start_ns
        self.count = count
        self.last_index = -1
        self.seen = set()
        self.terminal_ns = None
        self.policy_failures = []
        self.diagnostic_warnings = []
        self.hard_execution_fault = False

    def observe(self, event):
        try:
            self._observe(event)
        except Exception:
            self.hard_execution_fault = True
            raise

    def _observe(self, event):
        p = event["payload"]
        require(type(p) is dict, "event payload is not a dictionary")
        for key in ("cmdID", "wayPointIndex", "reachTarget", "error", "remark", "customInfo"):
            require(key in p, f"event missing {key}")
        require(event["host_ns"] >= self.start_ns, "event predates start attempt")
        require(type(p["cmdID"]) is str and p["cmdID"] == self.cmd_id, "event cmdID mismatch")
        i = p["wayPointIndex"]
        require(type(i) is int and 0 <= i < self.count, "invalid event waypoint index")
        require(type(p["reachTarget"]) is bool, "reachTarget must be bool")
        require(type(p["remark"]) is str, "remark must be a string")
        require(p["customInfo"] == f"{self.tag}:k{i:03d}", "event customInfo/index mismatch")
        success_error(p["error"])
        if p["remark"]:
            # All identity/schema/error gates precede the experiment-specific
            # diagnostic policy. No remark can establish progress or completion.
            self.handle_remark(event)
            return
        require(i >= self.last_index,
                "non-diagnostic execution-progress index regressed; execution order unconfirmed")
        key = (i, p["reachTarget"])
        require(key not in self.seen, "duplicate execution event")
        require((i, True) not in self.seen, "event regressed after reachTarget")
        self.seen.add(key)
        self.last_index = i
        if i == self.count - 1 and p["reachTarget"] is True:
            self.terminal_ns = event["host_ns"]

    def handle_remark(self, event):
        """Default neutral/path_0003 policy stays disqualifying; explicit override only."""
        p = event["payload"]
        self.policy_failures.append({"host_ns": event["host_ns"], "cmdID": p["cmdID"],
                                     "wayPointIndex": p["wayPointIndex"], "remark": p["remark"],
                                     "classification": "POLICY_VALIDATION_FAILURE"})

    def completed_status(self):
        """Call only after independent terminal+q/state settling validation."""
        require(self.terminal_ns is not None and not self.hard_execution_fault,
                "completion requires terminal evidence without a hard fault")
        return "COMPLETED_REJECTED" if self.policy_failures else "COMPLETED_ACCEPTED"


def settled(records, after_ns=0, reference=None):
    if len(records) < SETTINGS["settle_min_samples"]:
        return False
    records = [r for r in records if r["decoded_end_ns"] >= after_ns]
    if len(records) < SETTINGS["settle_min_samples"]:
        return False
    times = [r["decoded_end_ns"] for r in records]
    if times[-1] - times[0] < int(SETTINGS["settle_window_s"] * 1e9):
        return False
    if any(b - a > int(SETTINGS["settle_max_gap_s"] * 1e9) for a, b in zip(times, times[1:])):
        return False
    for j in range(6):
        values = [r["q_native_rad"][j] for r in records]
        if max(values) - min(values) > SETTINGS["settle_span_rad"]:
            return False
        if reference is not None and any(abs(q - reference[j]) > SETTINGS["final_residual_rad"]
                                         for q in values):
            return False
    return True


class PreparedSession:
    """All robot calls serialized on one owning thread; operator-gated motion."""
    def __init__(self, sdk, store):
        self.sdk, self.store = sdk, store
        self.robot = None
        self.connect_attempted = False
        self.inbox = EventInbox(sdk)
        self.records = deque()
        self.sequence = 0
        self.origin = time.perf_counter_ns()
        self.last_good = self.origin
        self.last_state_query = 0
        self.initial_mode = None
        self.completion = None
        self.phase = "PREPARED"
        self.start_attempted = False
        self.cmd_id = None
        self.q0 = None
        self.attempts = {name: 0 for name in CONTROL_APIS}
        self.authorized = set()
        self.setup_complete = False
        self.append_complete = False
        self.mode_succeeded = False
        self.strict_window = False
        self.last_sample = None
        self.pending_boundary = None
        self.max_motion_gap_ns = 0
        self.max_post_boundary_gap_ns = 0
        self.start_succeeded = False
        self.start_boundary = None
        self.motion_joint = None
        self.motion_sign = None
        self.motion_observation = {"threshold_rad": 1e-4, "first_threshold_sample": None,
                                   "maximum_signed_excursion_rad": None, "maximum_sequence": None}
        self.failure_reason = None

    def control_attempt_limit(self, name):
        return 1

    def make_completion(self, start_ns):
        return Completion(self.cmd_id, self.tag, start_ns)

    def state(self, phase, **extra):
        self.phase = phase
        self.store.write("lifecycle", {**stamp(), "phase": phase, **extra})
        print(phase, flush=True)

    def checked_call(self, name, *args, void=True):
        require(name in CHECKED_APIS, "API outside neutral-runner allowlist")
        if name in CONTROL_APIS:
            require(self.attempts[name] < self.control_attempt_limit(name), "control call retry rejected")
            if name in ("setMotionControlMode", "moveReset"):
                require(not self.start_attempted and "SETUP" in self.authorized, "SETUP not authorized")
                require(name != "moveReset" or self.mode_succeeded, "mode setup must succeed before reset")
            elif name == "moveAppend":
                require(not self.start_attempted and self.setup_complete and "APPEND" in self.authorized,
                        "APPEND requires successful setup observation and authorization")
            else:
                require(self.append_complete and "START" in self.authorized, "START requires verified append and authorization")
        ec = {}
        self.store.write("calls", {**stamp(), "api": name, "phase": "request"})
        if name in CONTROL_APIS:
            require(not self.inbox.failed.is_set(), "event watcher unhealthy at control boundary")
            require(time.perf_counter_ns() - self.last_good <= int(SETTINGS["fresh_host_receipt_s"] * 1e9),
                    "state became stale during control-boundary logging")
            require(self.last_sample is not None, "control call requires a measured q sample")
            self.attempts[name] += 1
            if name == "moveStart":
                self.start_attempted = True  # May begin motion before returning, including on exception.
                self.completion = self.make_completion(time.perf_counter_ns())
        request_ns = time.perf_counter_ns()
        result = None
        exception = None
        try:
            result = getattr(self.robot, name)(*args, ec)
        except BaseException:
            exception = traceback.format_exc()
        return_ns = time.perf_counter_ns()
        if name in CONTROL_APIS:
            self.pending_boundary = {"api": name, "last_q_before": self.last_sample,
                                     "request_ns": request_ns, "return_ns": return_ns,
                                     "duration_ns": return_ns - request_ns,
                                     "first_q_after": None, "cross_call_gap_ns": None,
                                     "premotion_gap_exemption": name in PREMOTION_APIS}
            self.store.write("lifecycle", {"code": "call_sampling_boundary", **self.pending_boundary})
        self.store.write("calls", {"host_ns": return_ns, "api": name,
                                  "phase": "exception" if exception else "return",
                                  "request_ns": request_ns, "return_ns": return_ns,
                                  "duration_ns": return_ns - request_ns, "traceback": exception,
                                  "return_value": snapshot(result, self.sdk),
                                  "return_repr": repr(result), "error": snapshot(ec, self.sdk),
                                  "ec_repr": repr(ec)})
        require(exception is None, f"{name} exception; see calls.jsonl: {exception}")
        success_error(snapshot(ec, self.sdk))
        if void:
            require(result is None, f"unexpected {name} return")
        if name == "setMotionControlMode":
            self.mode_succeeded = True
        elif name == "moveStart":
            self.start_succeeded = True
        return result

    def review_start_boundary(self, boundary, row):
        """Review one preserved START boundary, on the owning SDK thread."""
        self.start_boundary = boundary
        boundary.update(accepted=False, reason="START_BOUNDARY_REVIEW_INCOMPLETE",
                        valid_for_geometric_analysis=False, valid_for_timing_analysis=False,
                        moveStart_success=self.start_succeeded,
                        pre_delta_q0_rad=[a-b for a, b in zip(boundary["last_q_before"]["q_native_rad"], self.q0)],
                        post_delta_q0_rad=None, pre_post_delta_rad=None)
        try:
            require(row["valid_q"], "START_BOUNDARY_INVALID_POST_Q")
            q = row["q_native_rad"]
            boundary["post_delta_q0_rad"] = [a-b for a, b in zip(q, self.q0)]
            boundary["pre_post_delta_rad"] = [a-b for a, b in zip(q, boundary["last_q_before"]["q_native_rad"])]
            require(self.start_succeeded, "START_BOUNDARY_START_NOT_SUCCESSFUL")
            large = boundary["cross_call_gap_ns"] > int(SETTINGS["settle_max_gap_s"] * 1e9)
            if large:
                require(all(abs(d) <= SETTINGS["q0_drift_rad"] for d in boundary["post_delta_q0_rad"]),
                        "START_BOUNDARY_MOTION_UNOBSERVED")
                require(all(abs(d) <= SETTINGS["q0_drift_rad"] for d in boundary["pre_delta_q0_rad"]),
                        "START_BOUNDARY_PRE_Q_NOT_AT_Q0")
            self.drain_events()  # Preserve/validate queued events before accepting the boundary.
            require(not self.completion.hard_execution_fault, "START_BOUNDARY_HARD_EVENT_ERROR")
            self.review_start_status()
            boundary.update(accepted=True, reason=("START_BOUNDARY_STATIONARY_UNOBSERVED_INTERVAL"
                                                   if large else "START_BOUNDARY_WITHIN_50_MS"),
                            valid_for_geometric_analysis=True,
                            valid_for_timing_analysis=False,
                            start_timing_observability_passed=not large,
                            observable_window_first_sequence=row["sequence"],
                            observable_window_start_ns=row["decoded_end_ns"])
        except BaseException as exc:
            boundary["reason"] = str(exc)
            raise
        finally:
            self.store.write("lifecycle", {"code": "start_boundary_review", **boundary})

    def review_start_status(self):
        self.robot_state(before_start=False)  # Default neutral/path behavior.

    def observe_neutral_motion(self, row):
        if not self.start_boundary or not self.start_boundary["accepted"]:
            return
        if row["sequence"] <= self.start_boundary["observable_window_first_sequence"]:
            return
        displacement = self.motion_sign * (row["q_native_rad"][self.motion_joint - 1] - self.q0[self.motion_joint - 1])
        evidence = self.motion_observation
        if evidence["maximum_signed_excursion_rad"] is None or displacement > evidence["maximum_signed_excursion_rad"]:
            evidence.update(maximum_signed_excursion_rad=displacement, maximum_sequence=row["sequence"],
                            maximum_timestamp_ns=row["decoded_end_ns"])
        if displacement >= evidence["threshold_rad"] and evidence["first_threshold_sample"] is None:
            evidence["first_threshold_sample"] = {"sequence": row["sequence"], "timestamp_ns": row["decoded_end_ns"],
                                                  "signed_displacement_rad": displacement}
            self.store.write("lifecycle", {"code": "NEUTRAL_MOTION_OBSERVED", **evidence["first_threshold_sample"]})

    def premotion_call(self, name, *args):
        """Three explicitly allowed blind intervals, followed by fresh observation.

        This resets only the in-memory stationarity window, never the raw log.
        moveStart cannot enter this path, even after an error.
        """
        require(name in PREMOTION_APIS and not self.start_attempted, "not a pre-motion call")
        self.robot_state(before_start=True)
        self.fresh_barrier()
        require(settled(list(self.records), reference=self.q0), "pre-call q0 not stationary")
        self.store.barrier()
        self.checked_call(name, *args)
        self.fresh_barrier()  # Records first post-call q; drains/logs buffered frames.
        seed = self.last_sample
        self.records.clear()
        self.records.append(seed)
        self.strict_window = True
        self.state("POST_CALL_STATIONARITY", api=name)
        self.wait_settled(reference=self.q0, after_ns=seed["decoded_end_ns"])
        self.store.write("lifecycle", {**stamp(), "code": "post_call_stationarity_passed", "api": name,
                                      "first_fresh_sequence": seed["sequence"],
                                      "last_sequence": self.last_sample["sequence"]})
        if name == "moveReset":
            self.setup_complete = True
        elif name == "moveAppend":
            self.append_complete = True

    def robot_state(self, before_start):
        mode = self.checked_call("operateMode", void=False)
        power = self.checked_call("powerState", void=False)
        operation = self.checked_call("operationState", void=False)
        require(type(mode) is self.sdk.OperateMode and type(power) is self.sdk.PowerState
                and type(operation) is self.sdk.OperationState, "unexpected robot-state return type")
        require(mode in (self.sdk.OperateMode.manual, self.sdk.OperateMode.automatic), "unknown operate mode")
        if self.initial_mode is None:
            self.initial_mode = mode
        require(mode == self.initial_mode, "operating mode changed")
        require(power == self.sdk.PowerState.on, "existing power state is not on")
        allowed = (self.sdk.OperationState.idle,) if before_start else (
            self.sdk.OperationState.idle, self.sdk.OperationState.moving)
        require(operation in allowed, "inconsistent operation state")
        self.last_state_query = time.perf_counter_ns()
        return operation

    def read_sample(self, timeout_s=None):
        start = time.perf_counter_ns()
        count = self.robot.updateRobotState(timedelta(seconds=(SETTINGS["read_timeout_s"]
                                                   if timeout_s is None else timeout_s)))
        end = time.perf_counter_ns()
        require(type(count) is int and count >= 0, "unexpected updateRobotState return")
        if count == 0:
            self.store.write("lifecycle", {"code": "timeout_no_sample", "read_start_ns": start,
                                          "read_end_ns": end, "gap_since_last_good_ns": end - self.last_good})
            require(end - self.last_good <= int(SETTINGS["no_valid_state_timeout_s"] * 1e9),
                    "no valid robot state for >1 second")
            return False
        values = self.sdk.PyTypeVectorDouble()
        decode_start = time.perf_counter_ns()
        rc = self.robot.getStateData("q_m", values, 6)
        decoded = time.perf_counter_ns()
        q = list(values.content())
        valid = type(rc) is int and rc == 0 and len(q) == 6 and all(
            type(v) is float and math.isfinite(v) for v in q)
        row = {"sequence": self.sequence, "read_start_ns": start, "read_end_ns": end,
               "decoded_end_ns": decoded, "origin_ns": self.origin, "update_bytes": count,
               "q_return_code": rc, "valid_q": valid, "q_native_rad": q,
               "controller_timestamp": None, "phase": self.phase,
               "sdk_sample_calls": [
                   {"api": "updateRobotState", "request_ns": start, "return_ns": end,
                    "duration_ns": end-start},
                   {"api": "getStateData(q_m)", "request_ns": decode_start, "return_ns": decoded,
                    "duration_ns": decoded-decode_start}]}
        # Preserve invalid non-finite values explicitly, without invalid JSON tokens.
        if not valid:
            row["q_native_repr"] = repr(q)
            row["q_native_rad"] = snapshot(q, self.sdk)
        self.store.write("q_m", row)
        self.sequence += 1
        if not valid and self.pending_boundary is not None and self.pending_boundary["api"] == "moveStart":
            self.pending_boundary.update(first_q_after=row, cross_call_gap_ns=decoded-self.pending_boundary["last_q_before"]["decoded_end_ns"])
            self.review_start_boundary(self.pending_boundary, row)
        require(valid, "invalid measured q retained; aborting")
        require(start <= end <= decoded and decoded > self.last_good, "invalid host timestamps")
        gap_ns = decoded - self.last_good
        if self.start_attempted:
            self.max_motion_gap_ns = max(self.max_motion_gap_ns, gap_ns)
        exempt = False
        start_boundary_sample = False
        if self.pending_boundary is not None:
            boundary = self.pending_boundary
            boundary.update(first_q_after=row, cross_call_gap_ns=decoded - boundary["last_q_before"]["decoded_end_ns"])
            self.store.write("lifecycle", {"code": "call_sampling_boundary_completed", **boundary})
            exempt = boundary["premotion_gap_exemption"] and not self.start_attempted
            self.pending_boundary = None
            if boundary["api"] == "moveStart":
                self.review_start_boundary(boundary, row)
                start_boundary_sample = True
                exempt = True  # This single reviewed boundary only; all later gaps remain strict.
                self.records.clear()
        self.last_good = decoded
        self.last_sample = row
        self.records.append(row)
        cutoff = decoded - int(SETTINGS["settle_window_s"] * 1e9)
        # Retain one point at/before the one-second boundary.
        while len(self.records) > 1 and self.records[1]["decoded_end_ns"] <= cutoff:
            self.records.popleft()
        if self.q0 is not None and not self.start_attempted:
            require(all(abs(a - b) <= SETTINGS["q0_drift_rad"] for a, b in zip(q, self.q0)),
                    "measured q left reviewed q0 tolerance")
        if self.start_attempted:
            if not start_boundary_sample:
                self.max_post_boundary_gap_ns = max(self.max_post_boundary_gap_ns, gap_ns)
        if (self.strict_window or self.start_attempted) and not exempt:
            require(gap_ns <= int(SETTINGS["settle_max_gap_s"] * 1e9),
                    "sampling gap >50 ms within observation/motion; validation failed")
        self.observe_neutral_motion(row)
        return True

    def drain_events(self):
        require(not self.inbox.failed.is_set(), f"event callback failed: {self.inbox.failure}")
        while True:
            try:
                event = self.inbox.items.get_nowait()
            except queue.Empty:
                break
            self.store.write("motion_events", event)
            require(self.completion is not None, "unexpected execution event before start")
            prior_remarks = len(self.completion.policy_failures)
            prior_diagnostics = len(self.completion.diagnostic_warnings)
            self.completion.observe(event)
            for diagnostic in self.completion.diagnostic_warnings[prior_diagnostics:]:
                self.store.write("lifecycle", {"code": "ADJACENT_POINTS_DIAGNOSTIC", **diagnostic})
                print("WARNING:", ascii(diagnostic["remark"]), "at source waypoint",
                      diagnostic["source_index"], "- diagnostic only; not execution progress.", flush=True)
            for rejection in self.completion.policy_failures[prior_remarks:]:
                self.store.write("lifecycle", {"code": "POLICY_VALIDATION_FAILURE", **rejection})
                print("POLICY_VALIDATION_FAILURE:", ascii(rejection["remark"]), "at waypoint",
                      rejection["wayPointIndex"], "- continuing passive observation; run cannot be accepted.", flush=True)

    def pump(self):
        self.drain_events()
        self.read_sample()
        self.drain_events()
        if time.perf_counter_ns() - self.last_state_query > int(SETTINGS["state_query_period_s"] * 1e9):
            self.robot_state(before_start=not self.start_attempted)
        if self.q0 is not None and not self.start_attempted and self.records:
            require(all(abs(a - b) <= SETTINGS["q0_drift_rad"]
                        for a, b in zip(self.records[-1]["q_native_rad"], self.q0)),
                    "measured q drifted from reviewed q0; regenerate in a new session")

    def fresh_barrier(self):
        """Log/drain old SDK frames to empty, then read one newly received frame.

        Example explicitly says SDK queue does not overwrite old frames. Host
        freshness is not a controller sample timestamp or a network-age proof.
        """
        for _ in range(SETTINGS["fresh_drain_limit"]):
            if not self.read_sample(0.0):
                break
        else:
            raise ValidationError("SDK state backlog cannot be drained")
        require(self.read_sample(), "no fresh state after draining queue")
        self.drain_events()
        require(time.perf_counter_ns() - self.last_good <= int(SETTINGS["fresh_host_receipt_s"] * 1e9),
                "stale host state receipt")

    def wait_settled(self, reference=None, after_ns=0):
        deadline = time.perf_counter() + SETTINGS["settle_timeout_s"]
        while time.perf_counter() < deadline:
            self.pump()
            if settled(list(self.records), after_ns, reference):
                if self.robot_state(before_start=not self.start_attempted) == self.sdk.OperationState.idle:
                    self.fresh_barrier()
                    if settled(list(self.records), after_ns, reference):
                        return
        raise ValidationError("stationary measured-q criterion not met")

    def prompt(self, text):
        # Only console input runs in this daemon; it never receives an SDK object.
        answers = queue.Queue(maxsize=1)

        def reader():
            try:
                answers.put((time.perf_counter_ns(), input(text), None))
            except BaseException as exc:
                answers.put((time.perf_counter_ns(), None, repr(exc)))

        threading.Thread(target=reader, name="operator-input", daemon=True).start()
        deadline = time.perf_counter() + SETTINGS["prompt_timeout_s"]
        while time.perf_counter() < deadline:
            self.pump()  # logging and state/event health continue throughout READY
            try:
                received, answer, error = answers.get_nowait()
            except queue.Empty:
                continue
            self.store.write("authorizations", {**stamp(), "input_received_ns": received,
                                                "prompt": text, "answer": answer, "error": error})
            require(error is None, "operator input failed")
            return answer
        raise ValidationError("operator input timed out; no automatic continuation")

    def authorize(self, action, session_hash):
        require(action in ("SETUP", "APPEND", "START") and action not in self.authorized,
                "invalid/repeated authorization stage")
        phrase = f"{action} NEUTRAL {session_hash}"
        require(self.prompt(f"Type exactly {phrase}: ") == phrase, f"{action} not authorized")
        self.store.write("authorizations", {**stamp(), "action": action, "accepted": True,
                                            "session_sha256": session_hash})
        self.store.barrier()
        self.authorized.add(action)

    def execute(self):
        return self._reviewed_sequence()

    def _reviewed_sequence(self):
        """A single connected session, with setup before generating any commands."""
        final_status = "FAILED_EXECUTION"
        cleanup_errors = []
        try:
            self.store.json("session.json", {**stamp(), "status": "PRECONNECTION",
                                             "settings": SETTINGS, "setup_evidence": SETUP_EVIDENCE})
            print("Arm stationary; all other motion owners stopped; RCI OFF; no pending movement.\n"
                  "Review initialization reset and disconnect stop. Power must already be ON.\n"
                  "Physical E-stop and normal safety procedure must remain immediately available.", flush=True)
            preflight = PRECONNECTION_PHRASE
            answer = input("Operator preflight (separate from APPEND/START):\n" + preflight + "\n> ")
            self.store.write("authorizations", {**stamp(), "action": "PRECONNECTION",
                                                "required_phrase": preflight, "answer": answer,
                                                "accepted": answer == preflight})
            require(answer == preflight, "preconnection review not authorized")
            self.store.barrier()
            self.state("CONNECTING")
            self.robot = self.sdk.xMateRobot()  # exactly one robot object
            self.connect_attempted = True
            require(self.robot.connectToRobot(REMOTE_IP, LOCAL_IP) is None, "unexpected connect return")
            info = self.checked_call("robotInfo", void=False)
            require(info.joint_num == 6, "not a six-joint robot")
            identity = {k: getattr(info, k) for k in ("id", "type", "version", "joint_num")}
            self.store.json("robot_identity.json", identity)
            require(info.type == EXPECTED_ROBOT_TYPE and type(info.joint_num) is int,
                    "unexpected controller model")
            self.robot_state(before_start=True)
            self.checked_call("setEventWatcher", self.sdk.Event.moveExecution, self.inbox.callback)
            require(self.robot.startReceiveRobotState(timedelta(seconds=SETTINGS["period_s"]), ["q_m"]) is None,
                    "unexpected subscription return")
            self.state("ACQUIRING_Q0")
            self.fresh_barrier()
            self.records.clear()
            self.wait_settled()
            self.q0 = list(self.records[-1]["q_native_rad"])
            q0_record = self.records[-1]
            print("q0 radians:", self.q0, "\nq0 degrees:", [math.degrees(q) for q in self.q0], flush=True)
            setup_config = {"session_id": self.store.folder.name, "q0_hex": [q.hex() for q in self.q0],
                            "identity": identity, "settings": SETTINGS, "sdk_sha256": SDK_HASHES,
                            "source_sha256": digest_bytes(Path(__file__).read_bytes()),
                            "actions": ["setMotionControlMode(NrtCommandMode)", "moveReset"]}
            setup_hash = digest_json(setup_config)
            self.store.json("setup.json", {**setup_config, "setup_sha256": setup_hash, "evidence": SETUP_EVIDENCE})
            print("SETUP authorizes NrtCommandMode then one queue reset. No waypoint motion at this stage.", flush=True)
            self.authorize("SETUP", setup_hash)
            self.premotion_call("setMotionControlMode", self.sdk.MotionControlMode.NrtCommandMode)
            self.premotion_call("moveReset")
            require(self.setup_complete, "post-setup stationary verification required")
            print("Visually verify selected joint/direction clearance at this pose.\n"
                  "An excursion <=0.25 degrees is NOT a safety guarantee. Keep E-stop immediately available.", flush=True)
            joint_text = self.prompt("Joint number (1-6; no default): ")
            require(joint_text in ("1", "2", "3", "4", "5", "6"), "invalid joint choice")
            joint = int(joint_text)
            direction = self.prompt("Direction (+ or -; no default): ")
            self.motion_joint, self.motion_sign = joint, (1 if direction == "+" else -1)
            excursion = float(self.prompt("Excursion degrees (>0, <=0.25; no default): "))
            rows = generate_neutral(self.q0, joint, direction, excursion)
            spacing = neutral_spacing_diagnostics(rows, self.q0, joint)
            self.store.json("neutral_spacing.json", spacing)
            print("Neutral spacing diagnostics:", json.dumps({k: v for k, v in spacing.items()
                                                              if k != "adjacent_selected_joint_rad"}, indent=2), flush=True)
            raw_policy, policy = read_policy()
            tag = "neutral-" + uuid.uuid4().hex
            self.tag = tag
            commands = construct_commands(self.sdk, rows, tag)
            snapshots = validate_commands(self.sdk, commands, rows, tag)
            config = {"sdk": "xCoreSDK-Python v0.5.0", "sdk_hashes": SDK_HASHES,
                      "runner_sha256": digest_bytes(Path(__file__).read_bytes()),
                      "remote_ip": REMOTE_IP, "local_ip": LOCAL_IP, "tag": tag,
                      "robot_identity": identity,
                      "setup_sha256": setup_hash, "setup_evidence": SETUP_EVIDENCE,
                      "settings": SETTINGS, "joint": joint, "direction": direction,
                      "excursion_deg": excursion, "excursion_rad": math.radians(excursion),
                      "q0_sha256": digest_bytes(joint_bytes(self.q0)),
                      "waypoints_sha256": digest_bytes(matrix_bytes(rows)),
                      "policy_sha256": digest_bytes(raw_policy), "commands_sha256": digest_json(snapshots),
                      "existing_operate_mode": self.initial_mode.name,
                      "formula": "d=A*i/49 for i<=49; d=A*(99-i)/50 for i>=50; peak d[49]=A; q0+sign*d on selected joint; exact endpoints",
                      "neutral_spacing_sha256": digest_json(spacing)}
            session_hash = digest_json(config)
            self.store.json("session.json", {**config, "session_sha256": session_hash,
                                             "q0_rad": self.q0, "q0_source_record": q0_record,
                                             "status": "VALIDATED_NOT_APPENDED"})
            self.store.json("waypoints.json", {"q_rad": rows, "count": 100, "q0_rad": self.q0})
            self.store.bytes("waypoints.f64le", matrix_bytes(rows))
            self.store.bytes("controller_policy.json", raw_policy)
            self.store.json("commands.json", snapshots)
            self.state("VALIDATED")
            summary = {"q0_rad": self.q0, "q0_deg": [math.degrees(q) for q in self.q0],
                       "joint": joint, "direction": direction, "excursion_deg": excursion,
                       "excursion_rad": math.radians(excursion), "commands": 100,
                       "jointSpeed": policy["jointSpeed"], "speed": policy["speed"],
                       "zone_0_to_98": 1, "zone_99": 0, "session_sha256": session_hash}
            print(json.dumps(summary, indent=2), flush=True)
            self.authorize("APPEND", session_hash)
            self.pre_control_check(commands, rows, tag, config)
            cmd_id = self.sdk.PyString()
            try:
                self.premotion_call("moveAppend", commands, cmd_id)
            finally:
                self.store.write("calls", {**stamp(), "api": "moveAppend", "phase": "command_id",
                                          "cmdID": snapshot(cmd_id.content(), self.sdk)})
            self.cmd_id = cmd_id.content()
            require(type(self.cmd_id) is str and bool(self.cmd_id.strip()) and "\x00" not in self.cmd_id,
                    "missing/invalid append command ID")
            self.state("READY")
            self.store.barrier()
            print("Batch appended. Motion has NOT been started. Cancel here by entering anything else.\n"
                  "Cancellation does not clear the queue; do not later start/resume it.\n"
                  "Before START: verify the chosen joint/direction has physical clearance, arm still stationary,\n"
                  "no other motion owner, and physical E-stop/normal safety procedure immediately available.", flush=True)
            self.authorize("START", session_hash)
            self.pre_control_check(commands, rows, tag, config)
            self.state("START_AUTHORIZED")
            self.store.barrier()
            self.checked_call("moveStart")  # exactly one attempt, never resume
            self.state("RUNNING")
            # Preserve and conditionally review the first START boundary, never hide it.
            self.read_sample()  # Persist START's first post-call sample/gap before event validation.
            deadline = time.perf_counter() + SETTINGS["completion_timeout_s"]
            while time.perf_counter() < deadline:
                self.pump()
                terminal = self.completion.terminal_ns
                if terminal is not None and settled(list(self.records), terminal, self.q0):
                    if self.robot_state(before_start=False) == self.sdk.OperationState.idle:
                        self.fresh_barrier()
                        if settled(list(self.records), terminal, self.q0):
                            self.drain_events()
                            require(self.start_succeeded and self.start_boundary and self.start_boundary["accepted"],
                                    "START_BOUNDARY_NOT_ACCEPTED")
                            require(self.motion_observation["first_threshold_sample"] is not None,
                                    "NO_OBSERVED_NEUTRAL_MOTION")
                            self.state("COMPLETION_CONFIRMED")
                            final_status = self.completion.completed_status()
                            break
            require(self.motion_observation["first_threshold_sample"] is not None, "NO_OBSERVED_NEUTRAL_MOTION")
            require(final_status in ("COMPLETED_ACCEPTED", "COMPLETED_REJECTED"),
                    "terminal event and subsequent settling not confirmed")
        except BaseException as exc:
            final_status = "FAILED_EXECUTION"
            self.failure_reason = str(exc)
            failed_phase = self.phase
            error_traceback = traceback.format_exc()
            self.phase = "FAULT"
            try:
                self.store.write("exceptions", {**stamp(), "phase": failed_phase,
                                                 "traceback": error_traceback})
            except BaseException as log_error:
                print("Exception log failure:", repr(log_error), error_traceback, file=sys.stderr)
            if self.start_attempted:
                print("MOTION STATUS UNCERTAIN. Operator must use the reviewed physical safety procedure.\n"
                      "No explicit stop command is sent; cleanup disconnect stops ongoing motion per SDK documentation. "
                      "No automatic reset or recovery.", flush=True)
            raise
        finally:
            # Attempt all cleanup even if a prior attempt/log write failed.
            if self.connect_attempted:
                for name in ("stopReceiveRobotState", "setNoneEventWatcher", "disconnectFromRobot"):
                    ec = {}
                    try:
                        if name == "stopReceiveRobotState":
                            result = self.robot.stopReceiveRobotState()
                        elif name == "setNoneEventWatcher":
                            result = self.robot.setNoneEventWatcher(self.sdk.Event.moveExecution, ec)
                        else:
                            result = self.robot.disconnectFromRobot(ec)
                        # Perform cleanup BEFORE logging so failed disks cannot
                        # prevent unsubscribe, watcher cancellation or disconnect.
                        self.store.write("calls", {**stamp(), "api": name, "phase": "cleanup",
                                                  "return_repr": repr(result), "error": snapshot(ec, self.sdk)})
                        require(result is None, "unexpected cleanup return")
                        if name != "stopReceiveRobotState":
                            success_error(snapshot(ec, self.sdk))
                    except BaseException as exc:
                        cleanup_errors.append({"api": name, "error": repr(exc)})
                try:
                    self.drain_events()
                except BaseException as exc:
                    cleanup_errors.append({"api": "final_event_drain", "error": repr(exc)})
            if cleanup_errors:
                final_status = "FAILED_EXECUTION"
            elif final_status in ("COMPLETED_ACCEPTED", "COMPLETED_REJECTED"):
                # Late queued remarks during cleanup still cannot yield acceptance.
                final_status = self.completion.completed_status()
            if self.start_attempted and self.start_boundary is None:
                self.start_boundary = dict(self.pending_boundary or {})
                self.start_boundary.update(accepted=False, reason=self.failure_reason or "NO_VALID_POST_START_SAMPLE",
                                           valid_for_geometric_analysis=False, valid_for_timing_analysis=False,
                                           post_delta_q0_rad=None, pre_post_delta_rad=None,
                                           pre_delta_q0_rad=None if self.q0 is None or self.last_sample is None else
                                           [a-b for a,b in zip(self.last_sample["q_native_rad"], self.q0)])
                self.store.write("lifecycle", {"code": "start_boundary_review", **self.start_boundary})
            self.store.json("final_status.json", {**stamp(), "status": final_status,
                                                  "failure_reason": self.failure_reason,
                                                  "start_boundary": self.start_boundary,
                                                  "motion_observation": self.motion_observation,
                                                  "valid_for_geometric_analysis": final_status == "COMPLETED_ACCEPTED",
                                                  "valid_for_timing_analysis": False,
                                                  "geometric_observability_passed": final_status in ("COMPLETED_ACCEPTED", "COMPLETED_REJECTED"),
                                                  "start_timing_observability_passed": bool(self.start_boundary and self.start_boundary.get("start_timing_observability_passed", False)),
                                                  "max_post_boundary_gap_ns": self.max_post_boundary_gap_ns,
                                                  "policy_validation_failed": bool(self.completion and self.completion.policy_failures),
                                                  "policy_failures": [] if self.completion is None else self.completion.policy_failures,
                                                  "hard_event_fault": bool(self.completion and self.completion.hard_execution_fault),
                                                  "rd_motion_authorized": False,
                                                  "start_attempted": self.start_attempted, "cmdID": self.cmd_id,
                                                  "attempts": self.attempts, "setup_complete": self.setup_complete,
                                                  "append_post_observation_passed": self.append_complete,
                                                  "max_motion_gap_ns": self.max_motion_gap_ns,
                                                  "terminal_event_ns": None if self.completion is None else self.completion.terminal_ns,
                                                  "observed_index_reach_pairs": [] if self.completion is None else sorted(self.completion.seen),
                                                  "cleanup_errors": cleanup_errors, "q_records": self.sequence})
            self.store.barrier()
        return final_status

    def pre_control_check(self, commands, rows, tag, config):
        require(self.setup_complete and not self.start_attempted, "setup observation required before control")
        require(read_policy()[0] == (self.store.folder / "controller_policy.json").read_bytes(), "policy changed")
        require(digest_json(validate_commands(self.sdk, commands, rows, tag)) == config["commands_sha256"],
                "command batch changed after authorization")
        require(digest_bytes(matrix_bytes(rows)) == config["waypoints_sha256"], "waypoints changed")
        self.robot_state(before_start=True)
        self.fresh_barrier()
        require(settled(list(self.records), reference=self.q0), "q0 no longer stationary")
        self.drain_events()
        self.store.barrier()


def main(argv=None):
    """Deliberate hardware CLI; import/no flag/--help never load the vendor SDK."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-neutral", action="store_true", help="enter supervised neutral validation with operator gates")
    args = parser.parse_args(argv)
    if not args.run_neutral:
        parser.print_help()
        return 2
    folder = ROOT / "nrt_v050" / "neutral_sessions" / (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8])
    store = SessionStore(folder)
    handle = None
    try:
        print("SUPERVISED NEUTRAL VALIDATION\nSession:", folder, flush=True)
        sdk, handle = load_exact_sdk_data_only()
        status = PreparedSession(sdk, store).execute()
        print("Final status:", status, "\nObserve robot remains stationary after disconnect.", flush=True)
        return 0 if status == "COMPLETED_ACCEPTED" else 1
    except BaseException:
        error = traceback.format_exc()
        print(error, file=sys.stderr)
        if not (folder / "final_status.json").exists():
            store.json("final_status.json", {**stamp(), "status": "FAILED_EXECUTION", "exception": error,
                                             "rd_motion_authorized": False})
        return 1
    finally:
        try:
            store.close()
        finally:
            if handle is not None:
                handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
