"""Prepared lifecycle only. No SDK import, hardware loader, or enabled capture CLI.

The sdk argument is dependency-injected. Only synthetic doubles are used in tests.
Real binding and existing-state support must be reviewed before enabling a runner.
"""
from datetime import timedelta
import math
import time

PROFILES = {'smoke': (1.0, 10.0), 'characterization': (0.008, 30.0)}
REQUIRED = ('operator_authorized', 'runtime_and_hashes_verified',
            'robot_stationary', 'all_rl_tasks_stopped', 'no_other_motion_owner',
            'rci_off', 'existing_power_mode_supported_without_setters',
            'disconnect_lifecycle_reviewed')

def capture(sdk, remote_ip, local_ip, profile, prerequisites, emit_record,
            emit_event, clock=time.monotonic_ns):
    """Prepared future v0.5.0 call sequence; callers supply persistent sinks.

    emit_record must durably enqueue/preserve native values and times or raise.
    No automatic mode/power/recovery fallback exists. Prerequisites default false.
    """
    if profile not in PROFILES:
        raise ValueError('unsupported profile')
    if any(prerequisites.get(k) is not True for k in REQUIRED):
        raise RuntimeError('BLOCKED: passive preconditions not verified')
    period, duration = PROFILES[profile]
    robot = sdk.xMateRobot()  # Documented default construction; no IP here.
    attempted = False
    subscribed = False
    sequence = 0
    last_end = None
    error = None
    try:
        attempted = True
        robot.connectToRobot(remote_ip, local_ip)
        # No control-mode change, RT controller creation or power/setter calls.
        robot.startReceiveRobotState(timedelta(seconds=period), ['q_m'])
        subscribed = True
        origin = clock()
        deadline = origin + int(duration*1e9)
        last_good_end = origin
        emit_event({'code':'acquisition_start','host_ns':origin,'period_s':period,
                    'duration_s':duration,'controller_timestamp':None})
        while clock() < deadline:
            start = clock()
            read_timeout = 0.250 if profile == "characterization" else period
            count = robot.updateRobotState(timedelta(seconds=read_timeout))
            end = clock()
            if count == 0:
                gap_ns = end - last_good_end

                emit_event({
                    'code': 'timeout_no_sample',
                    'read_start_ns': start,
                    'read_end_ns': end,
                    'gap_since_last_good_ns': gap_ns,
                })

                # Keep the acquisition alive through an isolated host/network stall.
                # Give up only if no valid state has been received for >1 second.
                if gap_ns > 1_000_000_000:
                    raise RuntimeError('no valid robot state received for >1 second')

                continue
            if count < 0:
                raise RuntimeError('invalid SDK update length')
            values = sdk.PyTypeVectorDouble()
            rc = robot.getStateData('q_m', values, 6)
            decoded_end = clock()
            q = list(values.content())
            valid = rc == 0 and len(q) == 6 and all(math.isfinite(v) for v in q)
            emit_record({'sequence':sequence,'read_start_ns':start,'read_end_ns':end,
                         'decoded_end_ns':decoded_end,'origin_ns':origin,
                         'update_bytes':count,'q_return_code':rc,'valid_q':valid,
                         'q_native_rad':q,'controller_timestamp':None})
            sequence += 1
            last_good_end = end
            if not valid:
                raise RuntimeError('invalid measured q retained; stopping')
            if end < start or (last_end is not None and end <= last_end):
                raise RuntimeError('invalid/duplicate host time')
            last_end = end
    except Exception as exc:
        error = exc
    finally:
        # Even if subscription partly starts and then raises, request unsubscribe.
        if attempted:
            try:
                robot.stopReceiveRobotState()
            except Exception as exc:
                error = error or exc
            try:
                ec = {}
                robot.disconnectFromRobot(ec)
                # Exact successful error-dict schema requires runtime validation.
                code = ec.get('ec', ec.get('value', None))
                if code != 0:
                    raise RuntimeError('disconnect error: ' + repr(ec))
            except Exception as exc:
                error = error or exc
        emit_event({'code':'capture_end','records':sequence,'subscription_started':subscribed,
                    'valid':error is None,'error':None if error is None else str(error)})
    if error is not None:
        raise error
    return sequence

if __name__ == '__main__':
    raise SystemExit('BLOCKED: prepared lifecycle only; no SDK loader or hardware runner')

