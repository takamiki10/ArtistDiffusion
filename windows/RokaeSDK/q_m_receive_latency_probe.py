"""Operator-invoked stationary q_m latency measurement. Importing does not load SDK.

No motion/control setters. One SDK owner; no background reader. Measurements are
buffered in memory and saved after cleanup, so per-sample disk logging is excluded.
Connection/SDK initialization and disconnect have their documented side effects.
"""
from __future__ import annotations
import argparse
from collections import deque
from datetime import datetime, timedelta, timezone
import gc
import json
import math
from pathlib import Path
import subprocess
import time
import uuid

from q_m_latency_analysis import distribution

ROOT = Path(__file__).resolve().parent
PERIOD_S = .008
READ_TIMEOUT_S = .250
IDENTITY = {'id': '23e6a44f-2f77-4c51-a3f4-b32e7f64e5dc',
            'type': 'XMC7-R850-W7G3B4C-S5', 'joint_num': 6}
PHRASE = 'CONNECT Q_M LATENCY PROBE'


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read_one(robot, sdk, rows, phase, previous_ns=None, timeout_s=READ_TIMEOUT_S):
    """Retain each call including timeout/error. No 50 ms rejection here."""
    rec = {'sequence': len(rows), 'phase': phase, 'timeout_s': timeout_s,
           'valid_q': False, 'decoded_end_ns': None, 'gap_ms': None}
    timeout = timedelta(seconds=timeout_s)
    process_start = time.process_time_ns()
    thread_start = time.thread_time_ns()
    start = time.perf_counter_ns()
    rec['read_start_ns'] = start
    try:
        count = robot.updateRobotState(timeout)
    except BaseException as exc:
        rec['error'] = repr(exc)
        raise
    else:
        rec['update_bytes'] = count
    finally:
        end = time.perf_counter_ns()
        thread_end = time.thread_time_ns()
        process_end = time.process_time_ns()
        rec.update(read_end_ns=end, read_duration_ms=(end-start)/1e6,
                   update_thread_cpu_ms=(thread_end-thread_start)/1e6,
                   update_process_cpu_ms=(process_end-process_start)/1e6,
                   host_before_read_ms=None if previous_ns is None else (start-previous_ns)/1e6)
        rows.append(rec)
    require(type(count) is int and count >= 0, 'Invalid updateRobotState return')
    if count == 0:
        rec['timeout_no_sample'] = True
        return rec
    values = sdk.PyTypeVectorDouble()
    decode_start = time.perf_counter_ns()
    try:
        rc = robot.getStateData('q_m', values, 6)
        decoded = time.perf_counter_ns()
        q = list(values.content())
    except BaseException as exc:
        rec['decode_error'] = repr(exc)
        raise
    valid = type(rc) is int and rc == 0 and len(q) == 6 and all(
        type(v) is float and math.isfinite(v) for v in q)
    rec.update(q_return_code=rc, decoded_end_ns=decoded,
               getStateData_start_ns=decode_start, getStateData_end_ns=decoded,
               getStateData_duration_ms=(decoded-decode_start)/1e6,
               read_end_to_decoded_ms=(decoded-end)/1e6,
               gap_ms=None if previous_ns is None else (decoded-previous_ns)/1e6,
               valid_q=valid, q_native_rad=q if valid else None)
    if not valid:
        rec['q_native_repr'] = repr(q)
    require(valid, 'Invalid q_m payload retained; measurement aborted')
    return rec


def qualify_stationarity(robot, sdk, rows):
    # Drain old FIFO data using the documented zero-timeout example pattern.
    for _ in range(4096):
        rec = read_one(robot, sdk, rows, 'DRAIN', timeout_s=0)
        if rec.get('timeout_no_sample'):
            break
    else:
        raise RuntimeError('State queue did not drain within 4096 reads')
    window = deque()
    deadline = time.perf_counter_ns() + 10_000_000_000
    previous = None
    while time.perf_counter_ns() < deadline:
        rec = read_one(robot, sdk, rows, 'STATIONARITY', previous)
        if not rec['valid_q']:
            continue
        previous = rec['decoded_end_ns']
        if rec['gap_ms'] is not None and rec['gap_ms'] > 50:
            window.clear()
        window.append(rec)
        while len(window) > 1 and window[1]['decoded_end_ns'] <= previous-1_000_000_000:
            window.popleft()
        if len(window) >= 100 and previous-window[0]['decoded_end_ns'] >= 1_000_000_000:
            span = [max(r['q_native_rad'][j] for r in window)-min(r['q_native_rad'][j] for r in window)
                    for j in range(6)]
            if max(span) <= 1e-5:
                return {'first_sequence': window[0]['sequence'], 'last_sequence': rec['sequence'],
                        'span_rad': span, 'q0_rad': rec['q_native_rad']}
    raise RuntimeError('Fresh stationary evidence not established in 10 seconds')


def measure(robot, sdk, duration, rows):
    start = time.perf_counter_ns()
    previous = None
    while time.perf_counter_ns()-start < duration*1e9:
        require(len(rows) < 500_000, 'Measurement memory record bound reached')
        rec = read_one(robot, sdk, rows, 'MEASUREMENT', previous)
        if rec['valid_q']:
            previous = rec['decoded_end_ns']
    return {'measurement_start_ns': start, 'measurement_end_ns': time.perf_counter_ns()}


def summarize(rows, gc_events):
    measured = [r for r in rows if r['phase'] == 'MEASUREMENT']
    valid = [r for r in measured if r['valid_q']]
    gc_intervals = []
    active = {}
    for event in gc_events:
        generation = event['generation']
        if event['phase'] == 'start':
            active[generation] = event['host_ns']
        elif generation in active:
            gc_intervals.append((active.pop(generation), event['host_ns']))
    tails = []
    for r in measured:
        if r['read_duration_ms'] > 20 or (r.get('gap_ms') or 0) > 50:
            since = r['read_start_ns'] if r.get('gap_ms') is None else r['decoded_end_ns']-round(r['gap_ms']*1e6)
            end = r['decoded_end_ns'] or r['read_end_ns']
            tails.append({**r, 'gc_intervals_overlapping_gap': [list(x) for x in gc_intervals if x[0] < end and x[1] > since]})
    return {'read_ms': distribution([r['read_duration_ms'] for r in measured]),
            'gap_ms': distribution([r['gap_ms'] for r in valid if r['gap_ms'] is not None]),
            'update_thread_cpu_ms': distribution([r['update_thread_cpu_ms'] for r in measured]),
            'update_process_cpu_ms': distribution([r['update_process_cpu_ms'] for r in measured]),
            'valid_samples': len(valid), 'timeout_reads': sum(r.get('timeout_no_sample', False) for r in measured),
            'q_span_rad': [max(r['q_native_rad'][j] for r in valid)-min(r['q_native_rad'][j] for r in valid)
                           for j in range(6)] if valid else None,
            'long_intervals': tails, 'gc_intervals_ns': gc_intervals,
            'limitations': 'Host receipt timestamps, not controller timestamps. CPU wall-time difference does not isolate network, SDK wait, or descheduling. Disk writes excluded during measurement. No motion trial acceptance.'}


def nic_snapshot():
    """Passive OS counters only; invoked outside sampling, only with CLI opt-in."""
    try:
        result = subprocess.run(['powershell', '-NoProfile', '-NonInteractive', '-Command',
            'Get-NetAdapterStatistics | Select-Object Name,ReceivedBytes,SentBytes,ReceivedPacketErrors,OutboundPacketErrors,ReceivedDiscardedPackets,OutboundDiscardedPackets | ConvertTo-Json -Compress'],
            capture_output=True, text=True, timeout=10)
        return {'host_ns': time.perf_counter_ns(), 'returncode': result.returncode,
                'stdout': result.stdout, 'stderr': result.stderr}
    except Exception as exc:
        return {'error': repr(exc)}


def run_probe(args):
    # This function is reachable from main only after explicit flag and phrase.
    import neutral_v050_nrt_runner as n
    folder = ROOT/'nrt_v050/latency_probes'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
    folder.mkdir(parents=True, exist_ok=False)
    rows, gc_events, calls = [], [], []
    meta = {'duration_s': args.duration, 'period_s': PERIOD_S, 'read_timeout_s': READ_TIMEOUT_S,
            'remote_ip': n.REMOTE_IP, 'local_ip': n.LOCAL_IP, 'sdk_hashes': n.SDK_HASHES,
            'expected_identity': IDENTITY, 'disk_writes_during_measurement': False,
            'utc': datetime.now(timezone.utc).isoformat(), 'status': 'NOT_STARTED'}
    robot = handle = None
    connect_attempted = receive_attempted = False
    def checked(name):
        ec = {}; start = time.perf_counter_ns()
        value = getattr(robot, name)(ec)
        calls.append({'api': name, 'request_ns': start, 'return_ns': time.perf_counter_ns(),
                      'return_repr': repr(value), 'error': ec.copy()})
        n.success_error(ec)
        return value
    def gc_callback(phase, info):
        gc_events.append({'host_ns': time.perf_counter_ns(), 'phase': phase, 'generation': info['generation']})
    try:
        if args.nic_counters:
            meta['nic_before'] = nic_snapshot()
        sdk, handle = n.load_exact_sdk_data_only()
        robot = sdk.xMateRobot()
        connect_attempted = True
        require(robot.connectToRobot(n.REMOTE_IP, n.LOCAL_IP) is None, 'Unexpected connection return')
        info = checked('robotInfo')
        meta['identity'] = {k: getattr(info, k) for k in ('id', 'type', 'version', 'joint_num')}
        require(all(meta['identity'][k] == v for k, v in IDENTITY.items()), 'Robot identity mismatch')
        mode, power, operation = checked('operateMode'), checked('powerState'), checked('operationState')
        require(type(mode) is sdk.OperateMode and mode in (sdk.OperateMode.manual, sdk.OperateMode.automatic), 'Invalid mode')
        require(type(power) is sdk.PowerState and power == sdk.PowerState.on, 'Power must already be ON')
        require(type(operation) is sdk.OperationState and operation == sdk.OperationState.idle, 'Robot must be idle')
        receive_attempted = True
        require(robot.startReceiveRobotState(timedelta(seconds=PERIOD_S), ['q_m']) is None, 'Unexpected subscription return')
        meta['stationarity'] = qualify_stationarity(robot, sdk, rows)
        print(f'Stationarity verified. Measuring for {args.duration:g} s; gaps over 50 ms will be retained.', flush=True)
        gc.callbacks.append(gc_callback)
        try:
            meta.update(measure(robot, sdk, args.duration, rows))
        finally:
            gc.callbacks.remove(gc_callback)
        meta['status'] = 'MEASUREMENT_FINISHED'
    except BaseException as exc:
        meta.update(status='FAILED', error=repr(exc))
    finally:
        cleanup_errors = []
        if receive_attempted:
            try:
                require(robot.stopReceiveRobotState() is None, 'Unexpected unsubscribe return')
            except BaseException as exc:
                cleanup_errors.append({'api': 'stopReceiveRobotState', 'error': repr(exc)})
        if connect_attempted:
            try:
                require(robot.disconnectFromRobot() is None, 'Unexpected disconnect return')
            except BaseException as exc:
                cleanup_errors.append({'api': 'disconnectFromRobot', 'error': repr(exc)})
        meta['cleanup_errors'] = cleanup_errors
        if cleanup_errors:
            meta['status'] = 'FAILED'
        if args.nic_counters:
            meta['nic_after'] = nic_snapshot()
        # Keep DLL directory handle alive through robot cleanup and destruction.
        robot = None
        if handle is not None:
            handle.close()
        for name, records in [('reads', rows), ('gc', gc_events), ('checks', calls)]:
            with (folder/(name+'.jsonl')).open('x', encoding='utf-8') as stream:
                for rec in records:
                    stream.write(json.dumps(rec, allow_nan=False)+'\n')
        (folder/'session.json').write_text(json.dumps(meta, indent=2, allow_nan=False), encoding='utf-8')
        (folder/'summary.json').write_text(json.dumps(summarize(rows, gc_events), indent=2, allow_nan=False), encoding='utf-8')
    print(f"{meta['status']}: {folder}")
    return 0 if meta['status'] == 'MEASUREMENT_FINISHED' else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-probe', action='store_true', help='Enable explicit operator connection prompt')
    parser.add_argument('--duration', type=float, default=60)
    parser.add_argument('--nic-counters', action='store_true', help='Read passive Windows NIC counters before/after only')
    args = parser.parse_args(argv)
    if not math.isfinite(args.duration) or not 1 <= args.duration <= 3600:
        parser.error('duration must be finite and between 1 and 3600 seconds')
    if not args.run_probe:
        parser.print_help()
        return 0
    print('NO MOTION COMMANDS. One state-receive client, no setters or queue reset.\n'
          'Arm stationary, power already ON, all other motion owners stopped, RCI OFF.\n'
          'SDK initialization/connection effects apply. Cleanup disconnect stops ongoing motion per SDK documentation.\n'
          'Measurement buffers are saved after cleanup; forced process termination can lose probe data.', flush=True)
    if input(f'Type exactly {PHRASE}: ') != PHRASE:
        print('Cancelled before SDK loading or connection.')
        return 1
    return run_probe(args)


if __name__ == '__main__':
    raise SystemExit(main())
