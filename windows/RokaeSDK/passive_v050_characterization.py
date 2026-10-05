import json
import sys
import time
from pathlib import Path
from datetime import datetime

import xCoreSDK_python as sdk
from passive_v050_prepared import capture

REMOTE_IP = "192.168.2.160"
LOCAL_IP = "192.168.2.100"

outdir = Path("nrt_v050") / "live_characterization" / datetime.now().strftime("%Y%m%d_%H%M%S")
outdir.mkdir(parents=True, exist_ok=False)

records_path = outdir / "q_records.jsonl"
events_path = outdir / "events.jsonl"
meta_path = outdir / "metadata.json"

metadata = {
    "remote_ip": REMOTE_IP,
    "local_ip": LOCAL_IP,
    "profile": "characterization",
    "requested_period_s": 0.008,
    "requested_duration_s": 30.0,
    "sdk": "xCoreSDK-Python v0.5.0",
    "motion_commands_permitted": False,
    "rci_expected_off": True,
}
meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

rf = records_path.open("a", encoding="utf-8", buffering=1)
ef = events_path.open("a", encoding="utf-8", buffering=1)

def emit_record(obj):
    rf.write(json.dumps(obj, separators=(",", ":")) + "\n")

def emit_event(obj):
    ef.write(json.dumps(obj, separators=(",", ":")) + "\n")
    ef.flush()
    print("EVENT:", obj)

prerequisites = {
    "operator_authorized": True,
    "runtime_and_hashes_verified": True,
    "robot_stationary": True,
    "all_rl_tasks_stopped": True,
    "no_other_motion_owner": True,
    "rci_off": True,
    "existing_power_mode_supported_without_setters": True,
    "disconnect_lifecycle_reviewed": True,
}

print("PASSIVE STATIONARY CHARACTERIZATION")
print("Remote:", REMOTE_IP)
print("Local :", LOCAL_IP)
print("Output:", outdir)
print()
print("NO motion, mode, power, reset, moveAppend, or moveStart commands are used.")
print()
answer = input("Type READONLY to continue: ")

if answer != "READONLY":
    print("Cancelled.")
    rf.close()
    ef.close()
    sys.exit(1)

try:
    n = capture(
        sdk=sdk,
        remote_ip=REMOTE_IP,
        local_ip=LOCAL_IP,
        profile="characterization",
        prerequisites=prerequisites,
        emit_record=emit_record,
        emit_event=emit_event,
        clock=time.perf_counter_ns,
    )
    print(f"PASS: captured {n} measured-q records")

except Exception as exc:
    print("FAILED:", repr(exc))
    raise

finally:
    rf.close()
    ef.close()
