"""Operator-gated NO-MOTION NRT setup probe, pinned xCoreSDK-Python v0.5.0.

Only --run-setup-probe enters hardware code. Import and --help are offline.
No trajectory/neutral runner is imported. All SDK operations are serialized.
"""
from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timedelta, timezone
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
SESSION_ROOT = ROOT / "nrt_v050" / "setup_probe_sessions"
REMOTE_IP, LOCAL_IP = "192.168.2.160", "192.168.2.100"
EXPECTED_ROBOT_TYPE = "XMC7-R850-W7G3B4C-S5"
# Operator-reported first connection evidence, not a required version match.
OBSERVED_CONTROLLER_VERSION = "2.3.1.1.C89.20250306"
SDK_HASHES = {
    "xCoreSDK.dll": "7b72eaf137121036b6fcc7bf585d71331452262cea6f6d4082b754326d794c02",
    "xCoreSDK_python.cp312-win_amd64.pyd":
        "0bef1547fc74617c8835b0bf7f1dd13bc175612fa7a5ad7827c51371f7426455",
}
CONNECT_PHRASE = "CONNECT STATIONARY CR7 FOR NO-MOTION SETUP"
SETUP_PHRASE = "AUTHORIZE NRT MODE AND ONE QUEUE RESET ONLY"
SETTINGS = {
    "period_s": 0.008, "read_timeout_s": 0.250, "no_sample_timeout_s": 1.0,
    "settle_window_s": 1.0, "settle_min_samples": 100,
    "max_gap_s": 0.050, "stationary_span_rad": 1e-5,
    "max_displacement_rad": 1e-5, "freshness_s": 0.050,
    "settle_timeout_s": 10.0, "prompt_timeout_s": 300.0,
    "query_period_s": 0.250, "drain_limit": 4096,
}
# A closed dispatch table prevents arbitrary method names from reaching the SDK.
EC_APIS = frozenset({"robotInfo", "powerState", "operateMode", "operationState",
                     "setMotionControlMode", "moveReset", "disconnectFromRobot"})


class ProbeFailure(RuntimeError):
    pass


def require(condition, message):
    if not condition:
        raise ProbeFailure(message)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def raw_evidence(value):
    """Keep a JSON-compatible object unmodified, PLUS its repr/type.

    No renaming/filtering/truncation of ec fields and no default=str conversion.
    For non-JSON objects retain every mapping entry in a typed tree and fail
    closed on ec interpretation; opaque objects cannot be losslessly serialized.
    """
    def native(v):
        if v is None or type(v) in (str, bool, int):
            return True
        if type(v) is float:
            return math.isfinite(v)
        if type(v) is list:
            return all(native(x) for x in v)
        if type(v) is dict:
            return all(type(k) is str and native(x) for k, x in v.items())
        return False

    def tree(v):
        typ = type(v).__module__ + "." + type(v).__qualname__
        if type(v) is dict:
            return {"type": typ, "entries": [[tree(k), tree(x)] for k, x in v.items()]}
        if type(v) in (list, tuple):
            return {"type": typ, "items": [tree(x) for x in v]}
        return {"type": typ, "repr": repr(v)}

    compatible = native(value)
    return {"raw": value if compatible else None,
            "repr": repr(value), "type": type(value).__module__ + "." + type(value).__qualname__,
            "json_exact": compatible, "typed_fallback": None if compatible else tree(value)}


def interpret_ec(ec):
    """Interpret separately from raw evidence; unknown is never success.

    Discover recognized numeric code fields without assuming one exact schema.
    Do not infer success from {}, message text, absence of exceptions or truthiness.
    Extra fields remain in the raw record. Alternative/opaque schemas need review.
    """
    if type(ec) is not dict or not raw_evidence(ec)["json_exact"]:
        return {"status": "unknown", "reason": "nonstandard/non-JSON ec structure"}
    candidates = {k: ec[k] for k in ("ec", "value", "code", "error_code") if k in ec}
    if any(type(v) is int and v != 0 for v in candidates.values()):
        return {"status": "error", "reason": "nonzero numeric code", "code_fields": candidates}
    if ec.get("success") is False or ec.get("error") is True:
        return {"status": "error", "reason": "explicit failure flag", "code_fields": candidates}
    if "error" in ec and ec["error"] not in (None, False, 0, ""):
        return {"status": "unknown", "reason": "additional error field needs review"}
    if not candidates or any(type(v) is not int for v in candidates.values()):
        return {"status": "unknown", "reason": "no unambiguous integer code", "code_fields": candidates}
    return {"status": "success", "reason": "all exposed recognized integer codes are zero",
            "code_fields": candidates}


class Store:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=False)
        self.files = {}
        try:
            for name in ("q_m", "calls", "lifecycle", "authorizations"):
                self.files[name] = (self.folder / (name + ".jsonl")).open("x", encoding="utf-8")
        except BaseException:
            self.close()
            raise

    def write(self, name, record):
        f = self.files[name]
        f.write(json.dumps(record, allow_nan=False, ensure_ascii=True) + "\n")
        f.flush()

    def json(self, name, record):
        temp = self.folder / (name + ".tmp")
        with temp.open("w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, allow_nan=False, ensure_ascii=True)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, self.folder / name)

    def sync(self):
        for f in self.files.values():
            f.flush()
            os.fsync(f.fileno())

    def close(self):
        errors = []
        for f in self.files.values():
            try:
                f.close()
            except Exception as exc:
                errors.append(repr(exc))
        if errors:
            raise OSError("Log close failed: " + repr(errors))


def load_pinned_sdk():
    require(sys.version_info[:3] == (3, 12, 10) and struct.calcsize("P") == 8,
            "Python 3.12.10 x64 is required")
    for name, expected in SDK_HASHES.items():
        require(sha256((SDK_DIR / name).read_bytes()) == expected, "SDK hash mismatch: " + name)
    require("xCoreSDK_python" not in sys.modules, "SDK already imported; use a fresh process")
    handle = os.add_dll_directory(str(SDK_DIR))
    try:
        path = SDK_DIR / "xCoreSDK_python.cp312-win_amd64.pyd"
        spec = importlib.util.spec_from_file_location("xCoreSDK_python", path)
        require(spec is not None and spec.loader is not None, "SDK loader unavailable")
        sdk = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(sdk)
        require(Path(sdk.__file__).resolve() == path.resolve(), "Wrong SDK module loaded")
        return sdk, handle
    except BaseException:
        handle.close()
        raise


class Probe:
    """Accept one already-created robot. Offline tests supply only Python fakes."""
    def __init__(self, robot, sdk, store, *, clock=time.perf_counter_ns, ask=None):
        self.robot, self.sdk, self.store, self.clock = robot, sdk, store, clock
        self.ask = ask  # Optional fake console for deterministic offline tests.
        self.origin = clock()
        self.last_good = self.origin
        self.last_query = self.origin
        self.records = deque()
        self.sequence = 0
        self.q0 = None
        self.initial = None
        self.phase = "PREPARED"
        self.connected_attempt = False
        self.setup_authorized = False
        self.mode_attempts = self.reset_attempts = 0
        self.mode_succeeded = self.reset_succeeded = False
        self.maximum_displacement = [0.0] * 6
        self.max_observed_gap_ns = 0

    def stamp(self):
        return {"host_ns": self.clock(), "utc": datetime.now(timezone.utc).isoformat()}

    def state(self, phase, **extra):
        self.phase = phase
        self.store.write("lifecycle", {**self.stamp(), "phase": phase, **extra})
        print(phase, flush=True)

    def call_ec(self, api, *, cleanup=False):
        require(api in EC_APIS, "API is outside the setup probe allowlist")
        require(not cleanup or api == "disconnectFromRobot", "only disconnect is an ec cleanup call")
        if api in ("setMotionControlMode", "moveReset"):
            require(self.setup_authorized, "setup is not authorized")
            require(self.mode_attempts == 0 if api == "setMotionControlMode" else
                    self.mode_succeeded and self.reset_attempts == 0, "setup order/retry rejected")
        ec = {}  # Always fresh; never reuse a prior call's success code.
        request = self.stamp()
        # Cleanup calls must still be attempted if disk logging has failed.
        if not cleanup:
            self.store.write("calls", {"api": api, "phase": "request", **request})
            if api in ("setMotionControlMode", "moveReset"):
                self.store.sync()
                require(self.clock() - self.last_good <= int(SETTINGS["freshness_s"] * 1e9),
                        "measured q became stale before setup call")
        result, exception = None, None
        try:
            if api == "robotInfo":
                result = self.robot.robotInfo(ec)
            elif api == "powerState":
                result = self.robot.powerState(ec)
            elif api == "operateMode":
                result = self.robot.operateMode(ec)
            elif api == "operationState":
                result = self.robot.operationState(ec)
            elif api == "setMotionControlMode":
                self.mode_attempts += 1
                result = self.robot.setMotionControlMode(self.sdk.MotionControlMode.NrtCommandMode, ec)
            elif api == "moveReset":
                self.reset_attempts += 1
                result = self.robot.moveReset(ec)
            elif api == "disconnectFromRobot":
                result = self.robot.disconnectFromRobot(ec)
        except BaseException:
            exception = traceback.format_exc()
        returned = self.stamp()  # Before serialization/interpretation work.
        # Persist raw evidence first, in its own record, before interpreting it.
        self.store.write("calls", {"api": api, "phase": "return" if exception is None else "exception",
                                   "request": request, "returned": returned,
                                   "return_value": raw_evidence(result), "ec": raw_evidence(ec),
                                   "exception": exception})
        parsed = interpret_ec(ec)
        self.store.write("calls", {"api": api, "phase": "interpretation", **self.stamp(),
                                   "interpretation": parsed})
        require(exception is None, api + " raised; inspect calls.jsonl")
        require(parsed["status"] == "success", api + ": " + parsed["status"] + "; inspect raw ec")
        if api in ("setMotionControlMode", "moveReset", "disconnectFromRobot"):
            require(result is None, api + " returned an unexpected value")
        if api == "setMotionControlMode":
            self.mode_succeeded = True
        elif api == "moveReset":
            self.reset_succeeded = True
        return result

    def read_states(self):
        # Capture all four reads before validating identity/power/mode/state.
        info = self.call_ec("robotInfo")
        power = self.call_ec("powerState")
        mode = self.call_ec("operateMode")
        operation = self.call_ec("operationState")
        identity = {k: getattr(info, k) for k in ("id", "type", "version", "joint_num")}
        record = {"identity": identity, "power": {"name": power.name, "value": power.value},
                  "operate_mode": {"name": mode.name, "value": mode.value},
                  "operation": {"name": operation.name, "value": operation.value}}
        self.store.write("lifecycle", {**self.stamp(), "code": "robot_snapshot", **record})
        if self.initial is None:
            session_path = self.store.folder / "session.json"
            metadata = json.loads(session_path.read_text(encoding="utf-8")) if session_path.exists() else {}
            metadata.update(expected_robot_type=EXPECTED_ROBOT_TYPE,
                            previously_observed_controller_version=OBSERVED_CONTROLLER_VERSION,
                            observed_controller_identity=identity)
            self.store.json("session.json", metadata)
        require(type(info.joint_num) is int and info.joint_num == 6, "robot is not six-axis")
        require(all(type(identity[k]) is str and identity[k].strip() for k in ("id", "type", "version")),
                "incomplete robot identity")
        require(info.type == EXPECTED_ROBOT_TYPE,
                "unexpected robot model; inspect raw robotInfo, do not bypass identity check")
        require(type(power) is self.sdk.PowerState and power == self.sdk.PowerState.on, "power is not already ON")
        require(type(mode) is self.sdk.OperateMode and mode in (
            self.sdk.OperateMode.manual, self.sdk.OperateMode.automatic), "unknown existing operateMode")
        require(type(operation) is self.sdk.OperationState and operation == self.sdk.OperationState.idle,
                "operationState is not idle")
        if self.initial is not None:
            require(record == self.initial, "robot identity/power/operateMode/operationState changed")
        self.last_query = self.clock()
        return record

    def read_sample(self, timeout_s=None):
        start = self.clock()
        count = self.robot.updateRobotState(timedelta(seconds=SETTINGS["read_timeout_s"]
                                                     if timeout_s is None else timeout_s))
        end = self.clock()
        require(type(count) is int and count >= 0, "unexpected updateRobotState return")
        if count == 0:
            self.store.write("lifecycle", {"code": "timeout_no_sample", "read_start_ns": start,
                                          "read_end_ns": end, "gap_since_last_good_ns": end - self.last_good})
            require(end - self.last_good <= int(SETTINGS["no_sample_timeout_s"] * 1e9),
                    "no valid measured state for >1 second")
            return False
        values = self.sdk.PyTypeVectorDouble()
        rc = self.robot.getStateData("q_m", values, 6)
        decoded = self.clock()
        q = list(values.content())
        valid = type(rc) is int and rc == 0 and len(q) == 6 and all(
            type(v) is float and math.isfinite(v) for v in q)
        row = {"sequence": self.sequence, "phase": self.phase, "read_start_ns": start,
               "read_end_ns": end, "decoded_end_ns": decoded, "origin_ns": self.origin,
               "update_bytes": count, "q_return_code": raw_evidence(rc)["raw"],
               "valid_q": valid, "q_native_rad": q if valid else None, "controller_timestamp": None}
        if not valid:
            row["invalid_q_evidence"] = raw_evidence(q)
        self.store.write("q_m", row)
        self.sequence += 1
        require(valid, "invalid measured q recorded")
        require(start <= end <= decoded and decoded > self.last_good, "invalid host timestamps")
        gap = decoded - self.last_good
        self.last_good = decoded
        self.records.append(row)
        cutoff = decoded - int(SETTINGS["settle_window_s"] * 1e9)
        while len(self.records) > 1 and self.records[1]["decoded_end_ns"] <= cutoff:
            self.records.popleft()
        if self.q0 is not None:
            self.max_observed_gap_ns = max(self.max_observed_gap_ns, gap)
            for j in range(6):
                self.maximum_displacement[j] = max(self.maximum_displacement[j], abs(q[j] - self.q0[j]))
            require(max(self.maximum_displacement) <= SETTINGS["max_displacement_rad"],
                    "measured displacement exceeds 1e-5 rad; no-motion validation failed")
            require(gap <= int(SETTINGS["max_gap_s"] * 1e9),
                    "sampling gap >50 ms; cannot validate stationary behavior through setup")
        return True

    def fresh(self):
        # The local SDK example states its queue does not overwrite old frames.
        for _ in range(SETTINGS["drain_limit"]):
            if not self.read_sample(0.0):
                break
        else:
            raise ProbeFailure("state backlog did not drain")
        require(self.read_sample(), "no new q frame after draining")
        require(self.clock() - self.last_good <= int(SETTINGS["freshness_s"] * 1e9), "stale q receipt")

    def pump(self):
        self.read_sample()
        if self.clock() - self.last_query >= int(SETTINGS["query_period_s"] * 1e9):
            self.read_states()

    def stationary(self, after_ns=0):
        rows = [r for r in self.records if r["decoded_end_ns"] >= after_ns]
        if len(rows) < SETTINGS["settle_min_samples"]:
            return False
        times = [r["decoded_end_ns"] for r in rows]
        if times[-1] - times[0] < int(SETTINGS["settle_window_s"] * 1e9):
            return False
        if any(b - a > int(SETTINGS["max_gap_s"] * 1e9) for a, b in zip(times, times[1:])):
            return False
        return all(max(r["q_native_rad"][j] for r in rows) - min(r["q_native_rad"][j] for r in rows)
                   <= SETTINGS["stationary_span_rad"] for j in range(6))

    def wait_stationary(self, after_ns=0):
        deadline = self.clock() + int(SETTINGS["settle_timeout_s"] * 1e9)
        while self.clock() < deadline:
            self.pump()
            if self.stationary(after_ns):
                self.read_states()
                self.fresh()
                if self.stationary(after_ns):
                    return
        raise ProbeFailure("stationary criterion not established within 10 seconds")

    def authorize_setup(self):
        print("Current q radians:", self.q0, flush=True)
        print("Current q degrees:", [math.degrees(q) for q in self.q0], flush=True)
        print("Exactly once: setMotionControlMode(NrtCommandMode), then moveReset on success.\n"
              "No waypoint append. No moveStart. No intended robot movement.\n"
              "Confirm the arm/tool/workpiece are visibly stationary, power is already ON,\n"
              "all other robot programs/clients are stopped, and the area is clear.\n"
              "Do not authorize if anything moved during connection or baseline acquisition.", flush=True)
        prompt = "Type exactly " + SETUP_PHRASE + ": "
        if self.ask is not None:
            answer = self.ask(prompt)  # Offline fake only; production uses nonblocking input below.
        else:
            responses = queue.Queue(maxsize=1)

            def reader():
                try:
                    responses.put((input(prompt), None))
                except BaseException as exc:
                    responses.put((None, repr(exc)))

            threading.Thread(target=reader, name="setup-operator-input", daemon=True).start()
            deadline = self.clock() + int(SETTINGS["prompt_timeout_s"] * 1e9)
            while True:
                require(self.clock() < deadline, "operator authorization timed out")
                self.pump()  # Keep logging while the operator considers the two calls.
                try:
                    answer, error = responses.get_nowait()
                except queue.Empty:
                    continue
                require(error is None, "operator input failed: " + str(error))
                break
        self.store.write("authorizations", {**self.stamp(), "action": "NRT_MODE_AND_ONE_RESET",
                                            "expected_phrase": SETUP_PHRASE, "answer": answer,
                                            "accepted": answer == SETUP_PHRASE})
        require(answer == SETUP_PHRASE, "two setup calls were not authorized")
        self.store.sync()
        self.read_states()
        self.fresh()
        require(self.stationary(), "q not stationary at authorization boundary")
        self.setup_authorized = True

    def run(self):
        failure, cleanup_errors = None, []
        try:
            self.state("CONNECTING")
            self.connected_attempt = True
            require(self.robot.connectToRobot(REMOTE_IP, LOCAL_IP) is None, "unexpected connect return")
            self.initial = self.read_states()
            self.state("SUBSCRIBING", requested_period_s=SETTINGS["period_s"], fields=["q_m"])
            require(self.robot.startReceiveRobotState(timedelta(seconds=SETTINGS["period_s"]), ["q_m"]) is None,
                    "unexpected subscription return")
            self.last_good = self.clock()
            self.fresh()
            self.records.clear()
            self.state("BASELINE")
            self.wait_stationary()
            self.q0 = list(self.records[-1]["q_native_rad"])
            self.state("AWAITING_SETUP_AUTHORIZATION", q0_rad=self.q0,
                       q0_source_sequence=self.records[-1]["sequence"])
            self.authorize_setup()
            self.state("SETTING_NRT_MODE")
            self.call_ec("setMotionControlMode")
            # Recheck measured stationarity and existing robot states before reset.
            self.fresh()
            self.read_states()
            self.fresh()
            self.state("RESETTING_QUEUE")
            self.call_ec("moveReset")
            after_setup = self.clock()
            self.state("POST_SETUP_OBSERVATION")
            self.fresh()
            self.wait_stationary(after_ns=after_setup)
            self.state("MEASURED_NO_MOTION_CONFIRMED",
                       maximum_displacement_rad=self.maximum_displacement,
                       max_observed_gap_ns=self.max_observed_gap_ns)
        except BaseException:
            failure = traceback.format_exc()
            try:
                self.state("FAILED", exception=failure)
            except BaseException:
                print(failure, file=sys.stderr)
        finally:
            if self.connected_attempt:
                try:
                    returned = self.robot.stopReceiveRobotState()
                    self.store.write("calls", {**self.stamp(), "api": "stopReceiveRobotState",
                                               "phase": "cleanup", "return_value": raw_evidence(returned)})
                    require(returned is None, "unexpected unsubscribe return")
                except BaseException:
                    cleanup_errors.append({"api": "stopReceiveRobotState", "exception": traceback.format_exc()})
                try:
                    self.call_ec("disconnectFromRobot", cleanup=True)
                except BaseException:
                    cleanup_errors.append({"api": "disconnectFromRobot", "exception": traceback.format_exc()})
            status = "PASS" if failure is None and not cleanup_errors else "FAILED"
            final = {**self.stamp(), "status": status, "exception": failure, "cleanup_errors": cleanup_errors,
                     "mode_attempts": self.mode_attempts, "reset_attempts": self.reset_attempts,
                     "mode_setter_succeeded": self.mode_succeeded, "reset_succeeded": self.reset_succeeded,
                     "nrt_mode_getter_used": False, "q_records": self.sequence,
                     "q0_rad": self.q0, "initial_robot_snapshot": self.initial,
                     "maximum_displacement_rad": self.maximum_displacement,
                     "max_observed_gap_ns": self.max_observed_gap_ns,
                     "evidence_limit": "Host-timed q samples and setter results; no readable NRT mode, "
                                       "queue-depth getter, or proof of unobserved/inter-sample motion."}
            self.store.json("final_status.json", final)
            self.store.sync()
        return final


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-setup-probe", action="store_true", help="enter the operator-gated hardware setup probe")
    args = parser.parse_args(argv)
    if not args.run_setup_probe:
        parser.print_help()
        return 2  # No directory, SDK import, constructor or connection.
    folder = SESSION_ROOT / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex)
    store = Store(folder)
    handle = None
    try:
        store.json("session.json", {"created_utc": datetime.now(timezone.utc).isoformat(),
                                    "probe": "NO-MOTION NRT setup", "python": sys.version,
                                    "executable": sys.executable, "sdk": "xCoreSDK-Python v0.5.0",
                                    "sdk_dir": str(SDK_DIR), "sdk_sha256": SDK_HASHES,
                                    "source_sha256": sha256(Path(__file__).read_bytes()),
                                    "remote_ip": REMOTE_IP, "local_ip": LOCAL_IP, "settings": SETTINGS,
                                    "expected_robot_type": EXPECTED_ROBOT_TYPE,
                                    "previously_observed_controller_version": OBSERVED_CONTROLLER_VERSION,
                                    "neutral_motion_enabled": False})
        print("NO-MOTION SETUP PROBE. Output:", folder, flush=True)
        print("Before connection: arm stationary; power already ON; all other motion owners stopped; RCI OFF.\n"
              "SDK initialization documents an implicit reset. Disconnect documents a stop.\n"
              "Watch the physical robot through connection, setup and disconnect.\n"
              "The two explicit setup calls require a SECOND phrase after measured baseline.", flush=True)
        answer = input("Type exactly " + CONNECT_PHRASE + ": ")
        store.write("authorizations", {"host_ns": time.perf_counter_ns(), "action": "CONNECT",
                                       "expected_phrase": CONNECT_PHRASE, "answer": answer,
                                       "accepted": answer == CONNECT_PHRASE})
        require(answer == CONNECT_PHRASE, "connection not authorized")
        store.sync()
        sdk, handle = load_pinned_sdk()
        robot = sdk.xMateRobot()  # Only constructor, reachable only after CLI + connection phrase.
        final = Probe(robot, sdk, store).run()
        print("Setup probe:", final["status"], "\nSession:", folder, flush=True)
        print("Observe that the robot also remains stationary after disconnect.\n"
              "Do not run the neutral experiment; this probe does not enable it.", flush=True)
        return 0 if final["status"] == "PASS" else 1
    except BaseException:
        failure = traceback.format_exc()
        print(failure, file=sys.stderr)
        store.json("final_status.json", {"status": "FAILED", "exception": failure,
                                         "host_ns": time.perf_counter_ns()})
        return 1
    finally:
        try:
            store.close()
        finally:
            if handle is not None:
                handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
