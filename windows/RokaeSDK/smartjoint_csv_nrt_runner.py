"""Exact SmartJoint CSV adapter for the reviewed v0.5.0 serial NRT lifecycle.

Inert import. --offline never constructs a robot. --run-trial requires four
operator gates. Timestamps/labels are provenance, never a host motion schedule.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import io
import json
import math
from pathlib import Path
import sys
import time
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parent))
import neutral_v050_nrt_runner as n
from path_0003_nrt_runner import TrialSession, stationary_evidence, PROTOCOL

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'SmartJoint_Data_diffusion.csv'
SOURCE_SHA256 = '3a592a1de99a63abe97182fa6dc7260dc1d3402f42685b6f522cbd574303ba9f'
COUNT = 508
MAX_TRIALS = 6
HEADER = ('Timestamp', 'TouchType', 'joint1', 'joint2', 'joint3', 'joint4',
          'joint5', 'joint6', 'OriginalStatus')
LIMITS_PATH = ROOT / 'evidence/controller_live_verification.json'
LIMITS_SHA256 = '0a56e540cdb9c966f05e3f82fcd847b5821e6fdd358c1f12b88cdf20f94504c5'
BATCHES = tuple((i, min(i + 100, COUNT)) for i in range(0, COUNT, 100))
SESSIONS = ROOT / 'nrt_v050/smartjoint_sessions'
COMPLETION_TIMEOUT_S = 600.0  # Fixed total observation bound; never a motion-timing input.
WARNING_POLICY = {
    'revision': 'smartjoint-exact-adjacent-points-diagnostic-20260914',
    'allowed_remark': 'adjacent points', 'sdk_error_code': 0,
    'identity_schema_and_source_tag_checks_required': True,
    'source_duplicate_required': False, 'advances_progress': False,
    'establishes_terminal_completion': False, 'invalidates_trial': False,
    'other_nonempty_remarks': 'fatal',
}
OBSERVATION_POLICY = {
    'revision': 'single-owner-prestart-epochs-20260915',
    'prestart_gap': 'restart candidate window within fixed settle timeout; no control retry',
    'prestart_software_work': 'outside accepted stationarity windows',
    'active_status_polling': False, 'active_gap_limit_s': 0.05,
    'completion': 'captured stationary final-target window, then unchanged mode, power on and controller idle',
    'mode_power_detection': 'pre-start and completion only; transient changes may be unobserved',
    'terminal_callback_required': False,
    'cleanup_disconnect_stops_motion': True,
}
STATUS_QUERIES = ('operateMode', 'powerState', 'operationState')


@dataclass(frozen=True)
class Trajectory:
    raw: bytes
    cells: tuple
    q: tuple
    timestamps: tuple

    def metadata(self, index):
        return {'source_index': index, 'csv_line': index + 2,
                'Timestamp': self.cells[index][0],
                'TouchType': self.cells[index][1],
                'OriginalStatus': self.cells[index][8]}


def parse_csv(raw):
    """Parse each joint token once to Python binary64. No numeric transformation."""
    records = list(csv.reader(io.StringIO(raw.decode('utf-8-sig'), newline=''), strict=True))
    n.require(records and tuple(records[0]) == HEADER, 'CSV header mismatch')
    cells = tuple(tuple(row) for row in records[1:])
    n.require(len(cells) == COUNT, 'exactly 508 complete CSV rows required')
    rows, timestamps = [], []
    for i, row in enumerate(cells):
        n.require(len(row) == len(HEADER) and all(v and v == v.strip() for v in row),
                  f'CSV field count/empty/whitespace at source index {i}')
        n.require(row[1] in ('Pen', 'Air'), f'unknown TouchType at {i}')
        q = tuple(float(v) for v in row[2:8])
        n.joint_bytes(q)
        t = float(row[0])
        n.require(math.isfinite(t), f'nonfinite timestamp at {i}')
        n.require(not timestamps or t > timestamps[-1], f'timestamps not strictly increasing at {i}')
        rows.append(q)
        timestamps.append(t)
    n.matrix_bytes(rows, COUNT)
    return Trajectory(raw, cells, tuple(rows), tuple(timestamps))


def load_csv(path=SOURCE):
    raw = Path(path).read_bytes()
    n.require(n.digest_bytes(raw) == SOURCE_SHA256, 'source CSV SHA256 mismatch; do not regenerate or edit')
    return parse_csv(raw)


def audit(data):
    """Independent DictReader parse and bit comparison, plus frozen position bounds."""
    raw_limits = LIMITS_PATH.read_bytes()
    n.require(n.digest_bytes(raw_limits) == LIMITS_SHA256, 'frozen position evidence changed')
    bounds_deg = json.loads(raw_limits)['position_bounds_deg']
    n.require(bounds_deg == [[-360, 360]] * 6, 'unexpected project position bounds')
    bounds = [[math.radians(v) for v in pair] for pair in bounds_deg]
    independent = tuple(tuple(float(row[f'joint{j}']) for j in range(1, 7))
                        for row in csv.DictReader(io.StringIO(data.raw.decode('utf-8-sig'), newline='')))
    loaded_bytes = n.matrix_bytes(data.q, COUNT)
    n.require(loaded_bytes == n.matrix_bytes(independent, COUNT), 'independent CSV parse differs')
    back = json.loads(json.dumps(data.q, allow_nan=False))
    n.require(n.matrix_bytes(back, COUNT) == loaded_bytes, 'JSON round trip differs')
    violations = [{'source_index': i, 'joint': j + 1, 'value_rad': q}
                  for i, row in enumerate(data.q) for j, q in enumerate(row)
                  if not bounds[j][0] <= q <= bounds[j][1]]
    n.require(not violations, f'frozen position bounds failed: {violations}')
    segments = []
    for i in range(COUNT):
        state = data.cells[i][1]
        if not segments or segments[-1]['TouchType'] != state:
            segments.append({'TouchType': state, 'first_index': i, 'last_index': i, 'count': 1})
        else:
            segments[-1]['last_index'] = i
            segments[-1]['count'] += 1
    deltas = [[b[j] - a[j] for j in range(6)] for a, b in zip(data.q, data.q[1:])]
    gaps = [max(map(abs, d)) for d in deltas]
    dt = [b-a for a, b in zip(data.timestamps, data.timestamps[1:])]
    return {'status': 'OFFLINE_NUMERIC_PASS', 'source_sha256': n.digest_bytes(data.raw),
            'columns': HEADER, 'row_count': COUNT, 'joint_columns': HEADER[2:8],
            'joint_units': 'radians required; CSV header does not declare units',
            'timestamp_units': 'seconds interpretation; CSV header does not declare units',
            'robot_constructed': False, 'robot_connected': False,
            'loaded_joint_array_equals_independent_csv_array': True,
            'binary64_bitwise_equal': True, 'compared_joint_values': COUNT * 6,
            'maximum_abs_parse_difference': 0.0, 'json_roundtrip_bitwise_equal': True,
            'joint_array_f64le_sha256': n.digest_bytes(loaded_bytes),
            'touch_counts': dict(Counter(row[1] for row in data.cells)), 'segments': segments,
            'original_status_counts': dict(Counter(row[8] for row in data.cells)),
            'first_row': {**data.metadata(0), 'q_rad': data.q[0]},
            'final_row': {**data.metadata(COUNT-1), 'q_rad': data.q[-1]},
            'timestamp_range': [data.timestamps[0], data.timestamps[-1]],
            'timestamp_strictly_increasing': True, 'timestamp_step_range': [min(dt), max(dt)],
            'timing_policy': 'controller planned; CSV timestamps are metadata; no timestamp sleeps',
            'position_bounds_deg': bounds_deg, 'position_evidence_sha256': LIMITS_SHA256,
            'position_violations': violations,
            'position_scope': 'frozen displayed bounds only; not all effective limits or collision safety',
            'per_joint_min_rad': [min(row[j] for row in data.q) for j in range(6)],
            'per_joint_max_rad': [max(row[j] for row in data.q) for j in range(6)],
            'maximum_adjacent_abs_delta_rad_per_joint': [max(abs(d[j]) for d in deltas) for j in range(6)],
            'adjacent_signed_deltas_rad': deltas,
            'minimum_nonzero_adjacent_linf_rad': min(g for g in gaps if g),
            'identical_adjacent_pairs': [[i, i+1] for i, g in enumerate(gaps) if g == 0.0],
            'batches': [{'start_index': a, 'end_index_exclusive': b, 'count': b-a} for a,b in BATCHES],
            'controller_policy': n.read_policy()[1], 'completion_timeout_s': COMPLETION_TIMEOUT_S,
            'warning_policy': WARNING_POLICY,
            'observation_policy': OBSERVATION_POLICY,
            'hardware_batch_acceptance_verified': False}


class BatchCompletion(n.Completion):
    """Map SDK batch-local indices to unchanged source indices; retain strict gates."""
    def __init__(self, batches, tag, start_ns):
        super().__init__('whole-csv', tag, start_ns, COUNT)
        n.require(len(batches) == len(BATCHES), 'not all append acknowledgements present')
        self.batches = dict(batches)

    def _observe(self, event):
        n.require(not self.hard_execution_fault and not self.policy_failures,
                  'prior execution fault remains fatal')
        p = event['payload']
        n.require(type(p) is dict, 'event payload is not a dictionary')
        cmd_id = p.get('cmdID')
        n.require(type(cmd_id) is str and cmd_id in self.batches, 'event cmdID mismatch')
        a, b = self.batches[cmd_id]
        i = p.get('wayPointIndex')
        n.require(type(i) is int and 0 <= i < b-a, 'invalid batch-local event waypoint index')
        super()._observe({**event, 'source_event': event,
                          'payload': {**p, 'cmdID': self.cmd_id, 'wayPointIndex': a+i}})

    def handle_remark(self, event):
        """Prospectively approved exact diagnostic, after all inherited hard gates."""
        p = event['payload']
        n.require(p['remark'] == 'adjacent points', 'unapproved SDK remark: '+repr(p['remark']))
        raw = event['source_event']
        self.diagnostic_warnings.append({
            'host_ns': event['host_ns'], 'classification': 'NONFATAL_ADJACENT_POINTS',
            'remark': p['remark'], 'cmdID': raw['payload']['cmdID'],
            'batch_local_index': raw['payload']['wayPointIndex'], 'source_index': p['wayPointIndex'],
            'raw_payload': raw['payload'],
        })


class CsvStore(n.SessionStore):
    STREAMS = n.SessionStore.STREAMS + ('source_rows',)

    def __init__(self, folder, data):
        self.data = data
        self.batches = {}
        self.last_reported_index = None
        self.previous_sample_ns = None
        self.inter_sample_calls = []
        super().__init__(folder)

    def write(self, stream, value):
        if stream == 'calls' and value.get('phase') in ('return', 'exception'):
            self.inter_sample_calls.append({k: value[k] for k in
                                           ('api', 'request_ns', 'return_ns', 'duration_ns')})
        if stream == 'lifecycle' and value.get('code') == 'timeout_no_sample':
            self.inter_sample_calls.append({'api': 'updateRobotState',
                'request_ns': value['read_start_ns'], 'return_ns': value['read_end_ns'],
                'duration_ns': value['read_end_ns']-value['read_start_ns']})
        if stream == 'q_m':
            index = self.last_reported_index
            calls = self.inter_sample_calls + value.get('sdk_sample_calls', [])
            value = {**value, 'source_csv_sha256': SOURCE_SHA256,
                     'previous_sample_timestamp_ns': self.previous_sample_ns,
                     'sample_timestamp_ns': value['decoded_end_ns'],
                     'gap_ns': None if self.previous_sample_ns is None else value['decoded_end_ns']-self.previous_sample_ns,
                     'sampling_critical': value.get('phase') == 'ACTIVE_SAMPLING',
                     'sdk_calls_since_previous_sample': calls,
                     'sdk_call_since_previous_sample': bool(calls),
                     'last_validated_event_source_index': index,
                     'last_validated_event_TouchType': None if index is None else self.data.cells[index][1],
                     'label_semantics': 'latest non-diagnostic reported waypoint metadata; not measured pen contact or time alignment'}
            self.previous_sample_ns = value['decoded_end_ns']
            self.inter_sample_calls = []
        elif stream == 'motion_events':
            payload = value.get('payload', {})
            if type(payload) is dict and type(payload.get('cmdID')) is str:
                span = self.batches.get(payload['cmdID'])
                index = payload.get('wayPointIndex')
                if span and type(index) is int and 0 <= index < span[1]-span[0]:
                    value = {**value, 'source_metadata': self.data.metadata(span[0]+index)}
        super().write(stream, value)


class CsvSession(TrialSession):
    """Reuse q_m logger, stationarity, SDK errors, START review and endpoint test."""
    def __init__(self, sdk, store, data, source, number):
        n.PreparedSession.__init__(self, sdk, store)
        self.data, self.rows, self.source = data, data.q, Path(source)
        self.number = number
        self.batches = {}
        self.final_stationary = None
        self.inflight_span = None
        self.prestart_epoch = 0
        self.prestart_epoch_active = False
        self.prestart_evidence = None

    def suspend_prestart_sampling(self, reason):
        n.require(not self.start_attempted, 'cannot suspend post-START sampling')
        self.prestart_epoch_active = False
        self.prestart_evidence = None
        self.strict_window = False  # SmartJoint opts into candidate-window checks below.
        self.records.clear()
        self.phase = 'PRESTART_SOFTWARE'
        self.store.write('lifecycle', {**n.stamp(), 'code': 'PRESTART_SAMPLING_SUSPENDED',
                                     'reason': reason, 'epoch': self.prestart_epoch})

    def begin_prestart_stationarity_epoch(self, reference_q, reason):
        """Acquire new evidence after host work; never reset the outer deadline on gaps."""
        n.require(not self.start_attempted, 'pre-start epoch forbidden after START')
        self.suspend_prestart_sampling(reason)
        deadline = time.perf_counter() + n.SETTINGS['settle_timeout_s']
        self.prestart_epoch += 1
        self.last_good = time.perf_counter_ns()  # No old software interval in this epoch.
        self.robot_state(before_start=True)
        # Flush/log buffered frames before accepting the first fresh seed.
        self.fresh_barrier()
        seed = self.last_sample
        self.records.clear()
        self.records.append(seed)
        self.prestart_epoch_active = True
        self.state('PRESTART_STATIONARITY', epoch=self.prestart_epoch)
        self.store.write('lifecycle', {**n.stamp(), 'code': 'PRESTART_STATIONARITY_EPOCH_BEGIN',
            'epoch': self.prestart_epoch, 'reason': reason, 'first_sequence': seed['sequence'],
            'first_timestamp_ns': seed['decoded_end_ns'], 'reference_q_rad': reference_q})
        while time.perf_counter() < deadline:
            self.pump()
            if not n.settled(list(self.records), reference=reference_q):
                continue
            # Status checks are still required pre-control. If they cause a gap,
            # read_sample restarts the candidate and this same deadline continues.
            self.robot_state(before_start=True)
            self.fresh_barrier()
            rows = list(self.records)
            if not n.settled(rows, reference=reference_q):
                continue
            n.require(time.perf_counter() < deadline, 'pre-start stationarity timeout')
            evidence = {
                'first_sequence': rows[0]['sequence'], 'last_sequence': rows[-1]['sequence'],
                'first_timestamp_ns': rows[0]['decoded_end_ns'],
                'last_timestamp_ns': rows[-1]['decoded_end_ns'], 'sample_count': len(rows),
                'duration_ns': rows[-1]['decoded_end_ns']-rows[0]['decoded_end_ns'],
                'maximum_gap_ns': max(b['decoded_end_ns']-a['decoded_end_ns'] for a,b in zip(rows,rows[1:])),
                'per_joint_span_rad': [max(r['q_native_rad'][j] for r in rows)-min(r['q_native_rad'][j] for r in rows) for j in range(6)],
                'reference_q_rad': reference_q,
                'reference_residual_rad': None if reference_q is None else [a-b for a,b in zip(rows[-1]['q_native_rad'],reference_q)],
                'window_max_abs_reference_residual_rad': None if reference_q is None else [max(abs(r['q_native_rad'][j]-reference_q[j]) for r in rows) for j in range(6)],
            }
            self.prestart_evidence = evidence
            self.prestart_epoch_active = False
            self.store.write('lifecycle', {**n.stamp(), 'code': 'PRESTART_STATIONARITY_EPOCH_ACCEPTED',
                                         'epoch': self.prestart_epoch, 'reason': reason, **evidence})
            return evidence
        raise n.ValidationError('pre-start stationarity timeout; no clean candidate window')

    def premotion_call(self, name, *args):
        n.require(name in n.PREMOTION_APIS and not self.start_attempted, 'not a pre-motion call')
        self.checked_call(name, *args)  # Fresh epoch immediately before the call.
        self.begin_prestart_stationarity_epoch(self.q0, 'after_'+name)
        if name == 'moveReset':
            self.setup_complete = True
        elif name == 'moveAppend':
            self.append_complete = True

    def control_attempt_limit(self, name):
        return len(BATCHES) if name == 'moveAppend' else 1

    def make_completion(self, start_ns):
        return BatchCompletion(self.batches, self.tag, start_ns)

    def checked_call(self, name, *args, void=True):
        n.require(not (self.start_attempted and self.phase != 'COMPLETION_CONFIRMATION' and name in STATUS_QUERIES),
                  'synchronous status query forbidden during sampling-critical execution')
        if name == 'moveAppend':
            index = self.attempts[name]
            n.require(index == len(self.batches) and index < len(BATCHES), 'append retry or missing acknowledgement')
            n.require(self.inflight_span == BATCHES[index], 'out-of-order append')
            a, b = self.inflight_span
            n.require(len(args[0]) == b-a, 'wrong batch length')
            for cmd, row in zip(args[0], self.rows[a:b]):
                n.require(n.joint_bytes(cmd.target.joints) == n.joint_bytes(row), 'batch target changed')
        if name == 'moveStart':
            n.require(len(self.batches) == len(BATCHES) and self.attempts['moveAppend'] == len(BATCHES),
                      'START requires all six successful appends')
        if name in n.CONTROL_APIS and not self.start_attempted:
            n.require(self.q0 is not None, 'control requires reviewed q0')
            self.suspend_prestart_sampling('control_validation_and_log_flush:'+name)
            self.store.barrier()
            self.begin_prestart_stationarity_epoch(self.q0, 'before_'+name)
            self.suspend_prestart_sampling(name)
        return super().checked_call(name, *args, void=void)

    def review_start_status(self):
        # pre_control_check just validated mode/power/idle before moveStart.
        # Never place blocking status reads after the first post-START q sample.
        self.store.write('lifecycle', {**n.stamp(), 'code': 'START_STATUS_POLL_DEFERRED',
            'reason': 'pre-start status verified; mode/power rechecked at stationary completion'})

    def pump(self):
        if not self.start_attempted:
            return super().pump()
        n.require(self.phase == 'ACTIVE_SAMPLING', 'sampling outside active phase')
        self.drain_events()
        self.read_sample()
        self.drain_events()

    def confirm_completion(self, deadline):
        motion = self.motion_observation['first_threshold_sample']
        if motion is None:
            return False
        evidence = stationary_evidence(list(self.records), self.rows[-1], motion['timestamp_ns'])
        if evidence is None:
            return False
        # Freeze the already captured, gap-checked window before any blocking call.
        self.store.write('lifecycle', {**n.stamp(), 'code': 'FINAL_STATIONARY_WINDOW_CAPTURED',
                                     'evidence': evidence})
        self.state('COMPLETION_CONFIRMATION')
        self.drain_events()
        operation = self.robot_state(False)  # Unchanged mode, power ON, allowed state.
        self.drain_events()
        n.require(operation == self.sdk.OperationState.idle, 'controller not idle after final stationary window')
        n.require(time.perf_counter() < deadline, 'completion confirmation exceeded total watchdog')
        n.require(not self.completion.hard_execution_fault and not self.completion.policy_failures,
                  'completion has execution fault')
        self.final_stationary = evidence
        return True

    def read_sample(self, timeout_s=None):
        if self.completion and self.completion.last_index >= 0:
            self.store.last_reported_index = self.completion.last_index
        previous_ns = self.last_good
        if not self.start_attempted:
            self.strict_window = False
        result = super().read_sample(timeout_s)
        if result and not self.start_attempted and self.prestart_epoch_active:
            row = self.last_sample
            gap = row['decoded_end_ns']-previous_ns
            if gap > int(n.SETTINGS['settle_max_gap_s']*1e9):
                self.records.clear()
                self.records.append(row)
                self.store.write('lifecycle', {**n.stamp(), 'code': 'PRESTART_STATIONARITY_WINDOW_RESTART',
                    'epoch': self.prestart_epoch, 'gap_ns': gap, 'previous_sample_timestamp_ns': previous_ns,
                    'first_sequence': row['sequence'], 'first_timestamp_ns': row['decoded_end_ns']})
        return result

    def authorize(self, action, session_hash):
        n.require(action in ('SETUP', 'APPEND', 'START') and action not in self.authorized, 'invalid authorization')
        self.suspend_prestart_sampling('operator_authorization:'+action)
        phrase = f'{action} SMARTJOINT TRIAL {self.number} {session_hash}'
        n.require(self.prompt('Type exactly '+phrase+': ') == phrase, action+' not authorized')
        self.store.write('authorizations', {**n.stamp(), 'action': action, 'accepted': True, 'phrase': phrase})
        self.store.barrier()
        self.authorized.add(action)

    def pre_control_check(self, commands, rows, tag, config):
        n.require(self.setup_complete and not self.start_attempted, 'setup observation required before control')
        self.suspend_prestart_sampling('command_validation_and_hashing')
        n.require(n.read_policy()[0] == (self.store.folder/'controller_policy.json').read_bytes(), 'policy changed')
        n.require(self.source.read_bytes() == self.data.raw, 'source CSV changed after loading')
        n.require(n.digest_json(n.validate_commands(self.sdk, commands, rows, tag, COUNT)) == config['commands_sha256'],
                  'command batch changed after authorization')
        n.require(n.digest_bytes(n.matrix_bytes(rows, COUNT)) == config['waypoints_sha256'], 'waypoints changed')
        # checked_call reacquires a fresh q0 window after all validation/prompt work.
        self.drain_events()

    def execute(self):
        status, errors = 'FAILED_EXECUTION', []
        self.tag = 'sj-'+uuid.uuid4().hex
        config = {'trial': self.number, 'source_csv_sha256': SOURCE_SHA256, 'source_path': str(self.source.resolve()),
                  'source_units_confirmed_by_cli': 'native J1-J6 radians; Timestamp seconds',
                  'row_count': COUNT, 'protocol': PROTOCOL, 'batch_spans': BATCHES,
                  'settings': {**n.SETTINGS, 'completion_timeout_s': COMPLETION_TIMEOUT_S},
                  'completion_watchdog_s': COMPLETION_TIMEOUT_S, 'warning_policy': WARNING_POLICY,
                  'observation_policy': OBSERVATION_POLICY,
                  'sdk_hashes': n.SDK_HASHES,
                  'runner_sha256': n.digest_bytes(Path(__file__).read_bytes()),
                  'shared_runner_sha256': n.digest_bytes(Path(n.__file__).read_bytes()),
                  'path_runner_sha256': n.digest_bytes((ROOT/'path_0003_nrt_runner.py').read_bytes()),
                  'waypoints_sha256': n.digest_bytes(n.matrix_bytes(self.rows, COUNT))}
        try:
            self.store.json('offline_validation.json', audit(self.data))
            self.store.json('session.json', config)
            self.store.bytes('source.csv', self.data.raw)
            self.store.bytes('waypoints.f64le', n.matrix_bytes(self.rows, COUNT))
            self.store.json('waypoints.json', {'q_rad': self.rows, 'count': COUNT})
            self.store.bytes('controller_policy.json', n.read_policy()[0])
            for i, row in enumerate(self.rows):
                self.store.write('source_rows', {**self.data.metadata(i), 'q_rad': row,
                                               'joint_tokens': self.data.cells[i][2:8]})
            phrase = f'CONNECT SMARTJOINT TRIAL {self.number}'
            print('508 unchanged rows; six sequential appends; one START. RCI OFF.\n'
                  'Power already ON; arm stationary; other motion owners stopped.\n'
                  'Review physical clearance from the CURRENT pose to row 0 and the complete path.\n'
                  'Air labels do not prove clearance. SETUP explicitly selects NRT mode and resets the queue.\n'
                  'No power/operate-mode/safety/cache setters. SDK initialization/reset and disconnect effects apply.\n'
                  'No explicit stop command is sent; cleanup disconnect stops ongoing motion per SDK documentation.\n'
                  f'508-row completion and physical duration are unverified; fixed watchdog is {COMPLETION_TIMEOUT_S:g} seconds.', flush=True)
            answer = input('Type exactly '+phrase+': ')
            self.store.write('authorizations', {**n.stamp(), 'action': 'PRECONNECTION', 'phrase': phrase,
                                               'answer': answer, 'accepted': answer == phrase})
            n.require(answer == phrase, 'connection not authorized')
            self.store.barrier()
            self.robot = self.sdk.xMateRobot()
            self.connect_attempted = True
            n.require(self.robot.connectToRobot(n.REMOTE_IP, n.LOCAL_IP) is None, 'unexpected connect return')
            info = self.checked_call('robotInfo', void=False)
            identity = {k: getattr(info, k) for k in ('id', 'type', 'version', 'joint_num')}
            self.store.json('robot_identity.json', identity)
            n.require(info.type == n.EXPECTED_ROBOT_TYPE and type(info.joint_num) is int and info.joint_num == 6,
                      'robot identity mismatch')
            self.robot_state(True)
            self.checked_call('setEventWatcher', self.sdk.Event.moveExecution, self.inbox.callback)
            n.require(self.robot.startReceiveRobotState(n.timedelta(seconds=.008), ['q_m']) is None, 'subscription return')
            self.state('ACQUIRING_Q0')
            self.begin_prestart_stationarity_epoch(None, 'initial_q0')
            self.q0 = list(self.last_sample['q_native_rad'])
            self.suspend_prestart_sampling('q0_review')
            print('Measured q0:', self.q0, '\nFirst CSV target (Air):', self.rows[0],
                  '\nCurrent-to-first delta:', [a-b for a,b in zip(self.rows[0], self.q0)], flush=True)
            config.update(q0_rad=self.q0, robot_identity=identity)
            self.authorize('SETUP', n.digest_json(config))
            self.premotion_call('setMotionControlMode', self.sdk.MotionControlMode.NrtCommandMode)
            self.premotion_call('moveReset')
            self.suspend_prestart_sampling('command_construction')
            commands = n.construct_commands(self.sdk, self.rows, self.tag, COUNT)
            snapshots = n.validate_commands(self.sdk, commands, self.rows, self.tag, COUNT)
            config.update(commands_sha256=n.digest_json(snapshots), tag=self.tag)
            session_hash = n.digest_json(config)
            self.suspend_prestart_sampling('metadata_write')
            self.store.json('session.json', {**config, 'session_sha256': session_hash})
            self.store.json('commands.json', [{**cmd, **self.data.metadata(i)} for i,cmd in enumerate(snapshots)])
            self.authorize('APPEND', session_hash)
            for a, b in BATCHES:
                self.pre_control_check(commands, self.rows, self.tag, config)
                self.inflight_span = (a, b)
                cmd_id = self.sdk.PyString()
                try:
                    self.premotion_call('moveAppend', commands[a:b], cmd_id)
                finally:
                    self.suspend_prestart_sampling('append_acknowledgement_metadata')
                    self.store.write('calls', {**n.stamp(), 'api': 'moveAppend', 'phase': 'command_id',
                                              'source_span': [a, b], 'cmdID': n.snapshot(cmd_id.content(), self.sdk)})
                ident = cmd_id.content()
                n.require(type(ident) is str and bool(ident.strip()) and '\x00' not in ident and ident not in self.batches,
                          'invalid or duplicate append command ID')
                self.batches[ident] = (a, b)
                self.store.batches[ident] = (a, b)
                self.store.json('append_batches.json', self.batches)
            self.cmd_id = ident
            self.state('READY')
            self.authorize('START', session_hash)
            self.pre_control_check(commands, self.rows, self.tag, config)
            self.checked_call('moveStart')
            self.state('START_BOUNDARY')
            self.read_sample()
            self.state('ACTIVE_SAMPLING')
            deadline = time.perf_counter() + COMPLETION_TIMEOUT_S
            while time.perf_counter() < deadline:
                self.pump()
                if self.confirm_completion(deadline):
                    status = 'COMPLETED_ACCEPTED'
                    break
            n.require(status != 'FAILED_EXECUTION', 'NO_OBSERVED_PHYSICAL_MOTION' if
                      self.motion_observation['first_threshold_sample'] is None else 'MEASURED_COMPLETION_NOT_CONFIRMED')
        except BaseException as exc:
            self.failure_reason = str(exc)
            self.store.write('exceptions', {**n.stamp(), 'error': repr(exc)})
            if self.start_attempted:
                print('Motion status requires operator observation and reviewed physical safety procedure. '
                      'No explicit stop command is sent; cleanup disconnect stops ongoing motion per SDK documentation. '
                      'No automatic reset/resume/recovery.', flush=True)
        finally:
            if self.connect_attempted:
                for name in ('stopReceiveRobotState', 'setNoneEventWatcher', 'disconnectFromRobot'):
                    ec = {}
                    try:
                        if name == 'stopReceiveRobotState': result = self.robot.stopReceiveRobotState()
                        elif name == 'setNoneEventWatcher': result = self.robot.setNoneEventWatcher(self.sdk.Event.moveExecution, ec)
                        else: result = self.robot.disconnectFromRobot(ec)
                        self.store.write('calls', {**n.stamp(), 'api': name, 'phase': 'cleanup',
                                                  'return_repr': repr(result), 'error': n.snapshot(ec, self.sdk)})
                        n.require(result is None, 'unexpected cleanup return')
                        if name != 'stopReceiveRobotState': n.success_error(n.snapshot(ec, self.sdk))
                    except BaseException as exc: errors.append({'api': name, 'error': repr(exc)})
                try: self.drain_events()
                except BaseException as exc: errors.append({'api': 'final_event_drain', 'error': repr(exc)})
            if errors or (self.completion and self.completion.hard_execution_fault): status = 'FAILED_EXECUTION'
            elif status == 'COMPLETED_ACCEPTED' and self.completion.policy_failures: status = 'COMPLETED_REJECTED'
            self.store.json('final_stationary.json', self.final_stationary)
            self.store.json('final_status.json', {**n.stamp(), 'status': status, 'trial': self.number,
                'source_csv_sha256': SOURCE_SHA256, 'failure_reason': self.failure_reason,
                'valid_for_geometric_analysis': status == 'COMPLETED_ACCEPTED', 'valid_for_timing_analysis': False,
                'start_boundary': self.start_boundary or self.pending_boundary, 'moveStart_success': self.start_succeeded,
                'appended_rows': sum(b-a for a,b in self.batches.values()), 'attempts': self.attempts,
                'motion_observation': self.motion_observation, 'final_stationary': self.final_stationary,
                'max_motion_gap_ns': self.max_motion_gap_ns, 'max_post_boundary_gap_ns': self.max_post_boundary_gap_ns,
                'policy_failures': [] if self.completion is None else self.completion.policy_failures,
                'diagnostic_warnings': [] if self.completion is None else self.completion.diagnostic_warnings,
                'completion_watchdog_s': COMPLETION_TIMEOUT_S, 'warning_policy': WARNING_POLICY,
                'observation_policy': OBSERVATION_POLICY,
                'hard_event_fault': bool(self.completion and self.completion.hard_execution_fault), 'cleanup_errors': errors})
            self.store.barrier()
        return status


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--offline', action='store_true')
    mode.add_argument('--run-trial', action='store_true')
    parser.add_argument('--csv', type=Path, default=SOURCE)
    parser.add_argument('--report', type=Path, default=ROOT/'nrt_v050/evidence/smartjoint_csv_validation.json')
    parser.add_argument('--check-sdk-data', action='store_true', help='offline data-object round trip; no robot construction')
    parser.add_argument('--confirm-native-radians-seconds', action='store_true')
    parser.add_argument('--trial', type=int, choices=range(1, MAX_TRIALS + 1))
    args = parser.parse_args(argv)
    data = load_csv(args.csv)
    report = audit(data)
    if args.offline:
        if args.check_sdk_data:
            sdk, handle = n.load_exact_sdk_data_only()
            try:
                commands = n.construct_commands(sdk, data.q, 'offline-smartjoint', COUNT)
                report['sdk_data_objects_bitwise_equal'] = (n.matrix_bytes([list(c.target.joints) for c in commands], COUNT)
                                                          == n.matrix_bytes(data.q, COUNT))
                report['sdk_command_count'] = len(commands)
                report['sdk_hashes'] = n.SDK_HASHES
            finally:
                handle.close()
        args.report.parent.mkdir(parents=True, exist_ok=True)
        n.require(args.report.resolve() != args.csv.resolve(), 'report cannot overwrite source CSV')
        args.report.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
        print('OFFLINE_NUMERIC_PASS: 508 rows; Pen=300 Air=208; 3048/3048 joint values bitwise equal.\n'
              f'Source SHA256: {SOURCE_SHA256}\nReport: {args.report}\nNo robot constructed or connected.')
        return 0
    n.require(args.trial is not None and args.confirm_native_radians_seconds,
              f'--trial 1..{MAX_TRIALS} and --confirm-native-radians-seconds required before SDK import')
    n.require(not args.check_sdk_data, '--check-sdk-data is offline only')
    SESSIONS.mkdir(parents=True, exist_ok=True)
    # One reservation per repetition. No automatic retries or deletion of failed trials.
    for prior in range(1, args.trial):
        marker = SESSIONS/f'trial_{prior:02d}.json'
        n.require(marker.exists(), 'previous trial has not been attempted')
        folder = Path(json.loads(marker.read_text(encoding='utf-8'))['folder'])
        final = json.loads((folder/'final_status.json').read_text(encoding='utf-8'))
        n.require(final['status'] == 'COMPLETED_ACCEPTED', 'previous trial not accepted; stop and review')
        n.require(final['source_csv_sha256'] == SOURCE_SHA256, 'previous source differs')
    folder = SESSIONS/(f'trial_{args.trial:02d}_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
    with (SESSIONS/f'trial_{args.trial:02d}.json').open('x', encoding='utf-8') as f:
        json.dump({'trial': args.trial, 'folder': str(folder), 'source_csv_sha256': SOURCE_SHA256}, f)
        f.flush()
        n.os.fsync(f.fileno())
    store = CsvStore(folder, data)
    handle = None
    try:
        sdk, handle = n.load_exact_sdk_data_only()
        status = CsvSession(sdk, store, data, args.csv, args.trial).execute()
        print('Trial status:', status, '\nSession:', folder, flush=True)
        return 0 if status == 'COMPLETED_ACCEPTED' else 1
    finally:
        store.close()
        if handle is not None: handle.close()


if __name__ == '__main__':
    raise SystemExit(main())
