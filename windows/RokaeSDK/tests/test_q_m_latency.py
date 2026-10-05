"""Offline only: synthetic clocks/state receivers; never load vendor SDK."""
import ast
from enum import Enum
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import q_m_latency_analysis as a
import q_m_receive_latency_probe as p


class Clock:
    def __init__(self): self.ns = 0
    def now(self): return self.ns


class Vector:
    def content(self): return self.q


class Receiver:
    def __init__(self, clock, durations=(8_000_000,), count=76, q=None, rc=0):
        self.clock, self.durations, self.count, self.rc = clock, iter(durations), count, rc
        self.last_duration = 8_000_000
        self.q = [0.0]*6 if q is None else q
        self.calls = []
    def updateRobotState(self, timeout):
        self.calls.append(('updateRobotState', timeout.total_seconds()))
        self.last_duration = next(self.durations, self.last_duration)
        self.clock.ns += self.last_duration
        return self.count
    def getStateData(self, field, values, size):
        self.calls.append(('getStateData', field, size))
        values.q = self.q
        return self.rc


class AnalysisTests(unittest.TestCase):
    def test_percentiles_and_strict_thresholds(self):
        d = a.distribution([0, 20, 30, 40, 50, 60])
        self.assertEqual(d['count'], 6)
        self.assertEqual(d['minimum'], 0)
        self.assertEqual(d['median'], 35)
        self.assertEqual(d['p90'], 55)
        self.assertEqual(d['p95'], 57.5)
        self.assertEqual(d['p99'], 59.5)
        self.assertEqual([d[f'greater_than_{t}_ms'] for t in (20,30,40,50)], [4,3,2,1])
    def test_empty_and_singleton(self):
        self.assertIsNone(a.distribution([])['median'])
        self.assertEqual(a.distribution([69.4783])['p99'], 69.4783)
    def test_boundary_is_separate(self):
        b = {'first_q_after': {'sequence': 10}}
        self.assertEqual([a.scope_for({'sequence': i}, b) for i in (9,10,11)],
                         ['prestart', 'start_boundary', 'post_start'])
    def test_analyzer_preserves_fixture_and_extracts_timeout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); session = root/'sessions'/'trial'; session.mkdir(parents=True)
            q = [{'sequence': i, 'read_start_ns': i*80_000_000, 'read_end_ns': i*80_000_000+69_478_300,
                  'decoded_end_ns': i*80_000_000+70_000_000, 'valid_q': True,
                  'q_native_rad': [0.0]*6, 'phase': 'ACTIVE_SAMPLING'} for i in range(2)]
            (session/'q_m.jsonl').write_text('\n'.join(map(json.dumps, q)))
            (session/'lifecycle.jsonl').write_text(json.dumps({'code': 'timeout_no_sample',
                'read_start_ns': 151_000_000, 'read_end_ns': 152_000_000}))
            (session/'final_status.json').write_text(json.dumps({'start_boundary': {'first_q_after': {'sequence': 0}}}))
            before = {x.name:x.read_bytes() for x in session.iterdir()}
            r = a.analyze_sessions(root/'sessions', root/'output')
            self.assertEqual(r['sessions'][0]['timeout_reads'], 1)
            self.assertEqual(r['groups']['post_start']['all_update_calls_ms']['count'], 2)
            self.assertEqual(r['groups']['post_start']['gap_ms']['maximum'], 80)
            self.assertEqual(before, {x.name:x.read_bytes() for x in session.iterdir()})
            with self.assertRaises(ValueError): a.analyze_sessions(root/'sessions', session/'output')


class ProbeTests(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.sdk = types.SimpleNamespace(PyTypeVectorDouble=Vector)
        self.timer = patch.object(p.time, 'perf_counter_ns', self.clock.now)
        self.timer.start(); self.addCleanup(self.timer.stop)
    def test_long_reads_and_gaps_continue(self):
        robot = Receiver(self.clock, [8_000_000, 69_478_300, 8_000_000])
        rows = []
        p.measure(robot, self.sdk, .080, rows)
        self.assertEqual(len(rows), 3)
        summary = p.summarize(rows, [])
        self.assertEqual(summary['read_ms']['greater_than_50_ms'], 1)
        self.assertEqual(summary['gap_ms']['greater_than_50_ms'], 1)
        self.assertTrue(rows[-1]['valid_q'])
        self.assertEqual(set(c[0] for c in robot.calls), {'updateRobotState', 'getStateData'})
    def test_timeout_counted_and_next_read_continues(self):
        robot = Receiver(self.clock, count=0); rows = []
        p.read_one(robot, self.sdk, rows, 'MEASUREMENT')
        self.assertEqual(p.summarize(rows, [])['timeout_reads'], 1)
        self.assertEqual(len(robot.calls), 1)
        robot.count = 76
        self.assertTrue(p.read_one(robot, self.sdk, rows, 'MEASUREMENT')['valid_q'])
    def test_invalid_q_fails_with_record(self):
        for q in ([float('nan')]*6, [0.0]*5):
            rows = []
            with self.assertRaisesRegex(RuntimeError, 'Invalid q_m'):
                p.read_one(Receiver(self.clock, q=q), self.sdk, rows, 'MEASUREMENT')
            self.assertIn('q_native_repr', rows[0])
            json.dumps(rows, allow_nan=False)
    def test_sdk_error_is_retained_and_fatal(self):
        robot = Receiver(self.clock); rows = []
        with patch.object(robot, 'updateRobotState', side_effect=RuntimeError('SDK error')):
            with self.assertRaisesRegex(RuntimeError, 'SDK error'):
                p.read_one(robot, self.sdk, rows, 'MEASUREMENT')
        self.assertIn('SDK error', rows[0]['error'])
    def test_bad_decode_return_fails(self):
        with self.assertRaisesRegex(RuntimeError, 'Invalid q_m'):
            p.read_one(Receiver(self.clock, rc=-1), self.sdk, [], 'MEASUREMENT')
    def test_gc_overlap_is_reported(self):
        rows = []; p.read_one(Receiver(self.clock, [70_000_000]), self.sdk, rows, 'MEASUREMENT')
        gc_events = [{'host_ns': 10, 'phase': 'start', 'generation': 0},
                     {'host_ns': 20, 'phase': 'stop', 'generation': 0}]
        self.assertEqual(p.summarize(rows, gc_events)['long_intervals'][0]['gc_intervals_overlapping_gap'], [[10,20]])
    def test_stationarity_requires_fresh_window(self):
        robot = Receiver(self.clock); rows = []
        original = robot.updateRobotState
        def update(timeout):
            return 0 if timeout.total_seconds() == 0 else original(timeout)
        with patch.object(robot, 'updateRobotState', side_effect=update):
            result = p.qualify_stationarity(robot, self.sdk, rows)
        self.assertGreaterEqual(result['last_sequence']-result['first_sequence']+1, 100)
        self.assertEqual(result['span_rad'], [0.0]*6)
    def test_no_flag_and_wrong_phrase_cannot_run(self):
        with patch.object(p, 'run_probe', side_effect=AssertionError('must not run')), redirect_stdout(io.StringIO()):
            self.assertEqual(p.main([]), 0)
            with patch('builtins.input', return_value='wrong'):
                self.assertEqual(p.main(['--run-probe']), 1)
    def test_explicit_gate_supports_both_durations(self):
        with patch.object(p, 'run_probe', return_value=0) as run, patch('builtins.input', return_value=p.PHRASE), redirect_stdout(io.StringIO()):
            for duration in (60,300):
                self.assertEqual(p.main(['--run-probe','--duration',str(duration)]), 0)
                self.assertEqual(run.call_args.args[0].duration, duration)
    def test_invalid_duration_cannot_run(self):
        with patch.object(p, 'run_probe', side_effect=AssertionError('must not run')), patch('sys.stderr', io.StringIO()):
            for value in ('nan','inf','0','3601'):
                with self.assertRaises(SystemExit): p.main(['--run-probe','--duration',value])
    def test_sdk_call_allowlist_and_no_threads(self):
        source = Path(p.__file__).read_text(); tree = ast.parse(source)
        direct = {n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call)
                  and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name)
                  and n.func.value.id == 'robot'}
        self.assertEqual(direct, {'updateRobotState','getStateData','connectToRobot',
            'startReceiveRobotState','stopReceiveRobotState','disconnectFromRobot'})
        checked = {n.args[0].value for n in ast.walk(tree) if isinstance(n, ast.Call)
                   and isinstance(n.func, ast.Name) and n.func.id == 'checked'}
        self.assertEqual(checked, {'robotInfo','operateMode','powerState','operationState'})
        attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        self.assertFalse(attrs & {'Thread','moveStart','moveReset','moveAppend','MoveAbsJCommand',
            'setMotionControlMode','setPowerState','setOperateMode','setEventWatcher'})
        self.assertEqual(p.PERIOD_S, .008); self.assertEqual(p.READ_TIMEOUT_S, .250)
    def lifecycle(self, bad_identity=False, not_idle=False):
        import neutral_v050_nrt_runner as n
        Mode = Enum('Mode', 'manual automatic')
        Power = Enum('Power', 'on off')
        Operation = Enum('Operation', 'idle moving')
        robot = Receiver(self.clock, [70_000_000, 8_000_000])
        controls = []
        robot.connectToRobot = lambda *ips: controls.append(('connect', ips))
        robot.startReceiveRobotState = lambda *args: controls.append(('subscribe', args))
        robot.stopReceiveRobotState = lambda: controls.append(('unsubscribe',))
        robot.disconnectFromRobot = lambda: controls.append(('disconnect',))
        def query(name, value):
            def call(ec):
                controls.append((name,))
                ec.update(ec=0, message='success')
                return value
            return call
        identity = {**p.IDENTITY, 'version': 'fake'}
        if bad_identity: identity['id'] = 'wrong'
        robot.robotInfo = query('robotInfo', types.SimpleNamespace(**identity))
        robot.operateMode = query('operateMode', Mode.manual)
        robot.powerState = query('powerState', Power.on)
        robot.operationState = query('operationState', Operation.moving if not_idle else Operation.idle)
        sdk = types.SimpleNamespace(PyTypeVectorDouble=Vector, OperateMode=Mode, PowerState=Power,
                                    OperationState=Operation, xMateRobot=lambda: robot)
        args = types.SimpleNamespace(duration=.080, nic_counters=False)
        with tempfile.TemporaryDirectory() as tmp, patch.object(p, 'ROOT', Path(tmp)), \
             patch.object(n, 'load_exact_sdk_data_only', return_value=(sdk, None)) as loader, \
             patch.object(p, 'qualify_stationarity', return_value={'q0_rad': [0.0]*6}) as stationary, \
             redirect_stdout(io.StringIO()):
            result = p.run_probe(args)
            self.assertEqual(loader.call_count, 1)
            summaries = list(Path(tmp).rglob('summary.json'))
            summary = json.loads(summaries[0].read_text())
            self.assertEqual(controls[-1], ('disconnect',))
            if bad_identity or not_idle:
                self.assertEqual(result, 1); stationary.assert_not_called()
                self.assertNotIn('subscribe', [x[0] for x in controls])
            else:
                self.assertEqual(result, 0)
                self.assertEqual(summary['read_ms']['greater_than_50_ms'], 1)
                self.assertEqual([x[0] for x in controls], ['connect', 'robotInfo', 'operateMode',
                    'powerState', 'operationState', 'subscribe', 'unsubscribe', 'disconnect'])
    def test_fake_lifecycle_only_queries_before_measurement(self):
        self.lifecycle()
    def test_wrong_identity_prevents_subscription(self):
        self.lifecycle(bad_identity=True)
    def test_moving_state_prevents_subscription(self):
        self.lifecycle(not_idle=True)
    def test_production_trajectory_policy_and_reservation_unchanged(self):
        root = Path(p.__file__).parent
        baseline = json.loads((root/'nrt_v050/evidence/q_m_receive_latency/baseline.json').read_text())
        paths = dict(baseline['files'])
        marker = 'nrt_v050\\smartjoint_sessions\\trial_06.json'
        paths[marker] = baseline['sessions'][marker]
        for name, digest in paths.items():
            self.assertEqual(hashlib.sha256((root/name).read_bytes()).hexdigest(), digest, name)


if __name__ == '__main__': unittest.main()
