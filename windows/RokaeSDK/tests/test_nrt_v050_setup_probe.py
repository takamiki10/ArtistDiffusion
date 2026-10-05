"""Offline setup-probe tests: Python fakes only; never import the vendor SDK."""
import ast
from contextlib import redirect_stdout
from datetime import timedelta
from enum import Enum
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import nrt_v050_setup_probe as p


class Power(Enum):
    on = 0
    off = 1


class Mode(Enum):
    manual = 0
    automatic = 1
    unknown = -1


class Operation(Enum):
    idle = 0
    moving = 9


class Clock:
    def __init__(self):
        self.now = 1_000_000_000

    def __call__(self):
        return self.now

    def advance(self, ns):
        self.now += ns


class Vector:
    def content(self):
        return self.values


SDK = types.SimpleNamespace(PowerState=Power, OperateMode=Mode, OperationState=Operation,
                            MotionControlMode=types.SimpleNamespace(NrtCommandMode=1),
                            PyTypeVectorDouble=Vector)


class FakeRobot:
    """Does not inherit any vendor type; missing forbidden APIs is intentional."""
    def __init__(self, clock, fault=None):
        self.clock, self.fault = clock, fault
        self.calls = []
        self.reset_count = 0
        self.mode_count = 0
        self.info = types.SimpleNamespace(id='test-cr7', type='XMC7-R850-W7G3B4C-S5',
                                         version='2.3.1.1.C89.20250306', joint_num=6)
        self.power, self.mode, self.operation = Power.on, Mode.manual, Operation.idle

    def connectToRobot(self, remote, local):
        self.calls.append('connect')
        assert (remote, local) == (p.REMOTE_IP, p.LOCAL_IP)
        if self.fault == 'connect':
            raise RuntimeError('fake connect failure')

    def robotInfo(self, ec):
        ec.update(ec=0, message='ok', extra={'preserve': [1, 2]})
        return self.info

    def powerState(self, ec):
        ec.update(value=0)
        return self.power

    def operateMode(self, ec):
        ec.update(code=0)
        return self.mode

    def operationState(self, ec):
        ec.update(error_code=0)
        return self.operation

    def startReceiveRobotState(self, interval, fields):
        self.calls.append('subscribe')
        assert interval == timedelta(milliseconds=8) and fields == ['q_m']
        if self.fault == 'subscribe':
            raise RuntimeError('fake subscribe failure')

    def updateRobotState(self, timeout):
        if timeout == timedelta(0):
            return 0
        self.clock.advance(8_000_000)
        if self.fault == 'timeout':
            return 0
        return 48

    def getStateData(self, field, values, size):
        assert field == 'q_m' and size == 6
        values.values = [-0.0, 0.1, -0.2, 0.3, -0.4, 0.5]
        if self.fault == 'nan':
            values.values[1] = float('nan')
        if self.mode_count and self.fault == 'displacement':
            values.values[1] += .001
        if self.reset_count and self.fault == 'post_displacement':
            values.values[1] += .001
        return 0

    def setMotionControlMode(self, mode, ec):
        self.calls.append('setMotionControlMode')
        self.mode_count += 1
        assert mode == 1
        ec.update(value=0, message='ok', arbitrary={'nested': [None, True, 1.5]})
        if self.fault == 'mode_throw':
            ec.update(value=5, detail='before exception')
            raise RuntimeError('fake mode failure')
        if self.fault == 'mode_error':
            ec['value'] = 3
        if self.fault == 'mode_unknown':
            ec.clear()
            ec['vendor_unknown'] = 'record without assuming success'
        if self.fault == 'mode_return':
            return 0
        if self.fault == 'power_change':
            self.power = Power.off
        if self.fault == 'operate_change':
            self.mode = Mode.automatic
        if self.fault == 'operation_change':
            self.operation = Operation.moving
        if self.fault == 'gap':
            self.clock.advance(60_000_000)

    def moveReset(self, ec):
        self.calls.append('moveReset')
        self.reset_count += 1
        ec.update(ec=0, message='ok', reset_extra=['unmodified'])
        if self.fault == 'reset_error':
            ec['ec'] = 9
        if self.fault == 'reset_throw':
            raise RuntimeError('fake reset failure')

    def stopReceiveRobotState(self):
        self.calls.append('unsubscribe')
        if self.fault == 'unsubscribe':
            raise RuntimeError('fake unsubscribe failure')

    def disconnectFromRobot(self, ec):
        self.calls.append('disconnect')
        ec.update(code=0, message='disconnected', original_extra={'a': [3, 4]})
        if self.fault == 'disconnect':
            ec['code'] = 7


class ProbeTests(unittest.TestCase):
    def exercise(self, fault=None, answer=p.SETUP_PHRASE, mutate=None):
        with tempfile.TemporaryDirectory() as temp:
            store = p.Store(Path(temp) / 'session')
            clock = Clock()
            robot = FakeRobot(clock, fault)
            if mutate:
                mutate(robot)
            probe = p.Probe(robot, SDK, store, clock=clock, ask=lambda _: answer)
            with redirect_stdout(io.StringIO()):
                final = probe.run()
            store.close()
            logs = {name: [json.loads(line) for line in (store.folder / (name + '.jsonl')).read_text().splitlines()]
                    for name in ('calls', 'q_m', 'lifecycle', 'authorizations')}
            self.assertEqual(final, json.loads((store.folder / 'final_status.json').read_text()))
            return robot, probe, final, logs

    def test_success_exact_once_and_observation_windows(self):
        robot, probe, final, logs = self.exercise()
        self.assertEqual(final['status'], 'PASS')
        self.assertEqual(robot.calls, ['connect', 'subscribe', 'setMotionControlMode', 'moveReset', 'unsubscribe', 'disconnect'])
        self.assertEqual((final['mode_attempts'], final['reset_attempts']), (1, 1))
        self.assertFalse(final['nrt_mode_getter_used'])
        for phase in ('BASELINE', 'POST_SETUP_OBSERVATION'):
            rows = [r for r in logs['q_m'] if r['phase'] == phase]
            self.assertGreaterEqual(len(rows), 100)
            self.assertGreaterEqual(rows[-1]['decoded_end_ns'] - rows[0]['decoded_end_ns'], 1_000_000_000)
        for row in logs['q_m']:
            self.assertIsNone(row['controller_timestamp'])

    def test_raw_ec_and_return_preserved_before_interpretation(self):
        _, _, _, logs = self.exercise()
        calls = logs['calls']
        for api, expected in (
            ('setMotionControlMode', {'value': 0, 'message': 'ok', 'arbitrary': {'nested': [None, True, 1.5]}}),
            ('moveReset', {'ec': 0, 'message': 'ok', 'reset_extra': ['unmodified']}),
            ('disconnectFromRobot', {'code': 0, 'message': 'disconnected', 'original_extra': {'a': [3, 4]}})):
            result_index = next(i for i, row in enumerate(calls) if row['api'] == api and row['phase'] == 'return')
            raw = calls[result_index]
            self.assertEqual(raw['ec']['raw'], expected)
            self.assertTrue(raw['ec']['json_exact'])
            self.assertIsNone(raw['return_value']['raw'])
            self.assertLessEqual(raw['request']['host_ns'], raw['returned']['host_ns'])
            self.assertEqual(calls[result_index + 1]['phase'], 'interpretation')

    def test_denied_phrase_means_no_setup(self):
        for answer in ('', p.CONNECT_PHRASE, p.SETUP_PHRASE.lower(), p.SETUP_PHRASE + ' '):
            robot, _, final, _ = self.exercise(answer=answer)
            self.assertEqual(final['status'], 'FAILED')
            self.assertEqual(robot.mode_count, 0)
            self.assertEqual(robot.reset_count, 0)
            self.assertEqual(robot.calls[-2:], ['unsubscribe', 'disconnect'])

    def test_bad_preconditions_inhibit_setup(self):
        for mutate in (lambda r: setattr(r.info, 'joint_num', 7),
                       lambda r: setattr(r.info, 'type', 'xMate CR5'),
                       lambda r: setattr(r.info, 'id', ''),
                       lambda r: setattr(r, 'power', Power.off),
                       lambda r: setattr(r, 'mode', Mode.unknown),
                       lambda r: setattr(r, 'operation', Operation.moving)):
            robot, _, final, _ = self.exercise(mutate=mutate)
            self.assertEqual(final['status'], 'FAILED')
            self.assertEqual(robot.mode_count, 0)
            self.assertEqual(robot.reset_count, 0)

    def test_exact_observed_model_with_six_joints_accepted(self):
        _, _, final, _ = self.exercise()
        self.assertEqual(p.EXPECTED_ROBOT_TYPE, 'XMC7-R850-W7G3B4C-S5')
        self.assertEqual(final['status'], 'PASS')
        self.assertEqual(final['initial_robot_snapshot']['identity']['type'], p.EXPECTED_ROBOT_TYPE)
        self.assertEqual(final['initial_robot_snapshot']['identity']['joint_num'], 6)

    def test_other_models_and_former_aliases_rejected(self):
        for model in ('XMC7-R850-W7G3B4C-S6', 'XMC7-R900-W7G3B4C-S5',
                      'xmatecr7', 'ROKAE xMate CR7', 'CR7',
                      'xmc7-r850-w7g3b4c-s5', 'XMC7-R850-W7G3B4C-S5 '):
            with self.subTest(model=model):
                robot, _, final, _ = self.exercise(mutate=lambda r: setattr(r.info, 'type', model))
                self.assertEqual(final['status'], 'FAILED')
                self.assertEqual((robot.mode_count, robot.reset_count), (0, 0))

    def test_exact_model_wrong_joint_count_rejected(self):
        robot, _, final, _ = self.exercise(mutate=lambda r: setattr(r.info, 'joint_num', 7))
        self.assertEqual(final['status'], 'FAILED')
        self.assertIn('not six-axis', final['exception'])
        self.assertEqual((robot.mode_count, robot.reset_count), (0, 0))

    def test_live_power_off_state_rejected_before_setup(self):
        robot, _, final, _ = self.exercise(mutate=lambda r: setattr(r, 'power', Power.off))
        self.assertEqual(final['status'], 'FAILED')
        self.assertIn('power is not already ON', final['exception'])
        self.assertEqual((robot.mode_count, robot.reset_count), (0, 0))
        self.assertNotIn('subscribe', robot.calls)

    def test_raw_version_metadata_preserved_without_version_pin(self):
        for version in ('2.3.1.1.C89.20250306', 'future-controller-version'):
            with tempfile.TemporaryDirectory() as temp:
                store = p.Store(Path(temp) / 'session')
                store.json('session.json', {'existing_metadata': 'retained'})
                clock = Clock()
                robot = FakeRobot(clock)
                robot.info.version = version
                probe = p.Probe(robot, SDK, store, clock=clock)
                record = probe.read_states()
                store.close()
                metadata = json.loads((store.folder / 'session.json').read_text())
                self.assertEqual(record['identity']['version'], version)
                self.assertEqual(metadata['observed_controller_identity']['version'], version)
                self.assertEqual(metadata['previously_observed_controller_version'], '2.3.1.1.C89.20250306')
                self.assertEqual(metadata['existing_metadata'], 'retained')

    def test_observed_success_ec_preserved(self):
        ec = {'ec': 0, 'message': 'success'}
        self.assertEqual(p.raw_evidence(ec)['raw'], {'ec': 0, 'message': 'success'})
        self.assertEqual(p.interpret_ec(ec)['status'], 'success')
        self.assertEqual(ec, {'ec': 0, 'message': 'success'})

    def test_mode_failures_never_reset(self):
        for fault in ('mode_error', 'mode_unknown', 'mode_throw', 'mode_return'):
            with self.subTest(fault=fault):
                robot, _, final, logs = self.exercise(fault)
                self.assertEqual(final['status'], 'FAILED')
                self.assertEqual((robot.mode_count, robot.reset_count), (1, 0))
                self.assertEqual(robot.calls[-2:], ['unsubscribe', 'disconnect'])
                self.assertTrue(any(row['api'] == 'setMotionControlMode' and 'ec' in row for row in logs['calls']))

    def test_stationarity_or_state_change_after_mode_inhibits_reset(self):
        for fault in ('displacement', 'gap', 'power_change', 'operate_change', 'operation_change'):
            robot, _, final, _ = self.exercise(fault)
            self.assertEqual(final['status'], 'FAILED')
            self.assertEqual((robot.mode_count, robot.reset_count), (1, 0))

    def test_reset_failure_or_post_motion_has_no_retry(self):
        for fault in ('reset_error', 'reset_throw', 'post_displacement'):
            robot, _, final, _ = self.exercise(fault)
            self.assertEqual(final['status'], 'FAILED')
            self.assertEqual((robot.mode_count, robot.reset_count), (1, 1))
            self.assertEqual(robot.calls[-2:], ['unsubscribe', 'disconnect'])

    def test_acquisition_connection_failures_cleanup(self):
        for fault in ('connect', 'subscribe', 'nan', 'timeout'):
            robot, _, final, _ = self.exercise(fault)
            self.assertEqual(final['status'], 'FAILED')
            self.assertEqual(robot.mode_count, 0)
            self.assertEqual(robot.calls[-2:], ['unsubscribe', 'disconnect'])

    def test_cleanup_errors_prevent_pass_and_still_disconnect(self):
        for fault in ('unsubscribe', 'disconnect'):
            robot, _, final, _ = self.exercise(fault)
            self.assertEqual(final['status'], 'FAILED')
            self.assertEqual(robot.calls[-1], 'disconnect')
            self.assertEqual(robot.reset_count, 1)

    def test_error_schema_is_observed_not_assumed(self):
        for ec in ({'ec': 0}, {'code': 0, 'extra': [1]}, {'value': 0, 'message': 'localized text'},
                   {'error_code': 0, 'value': 0}):
            self.assertEqual(p.interpret_ec(ec)['status'], 'success')
        for ec in ({}, {'message': 'success'}, {'ec': False}, {'ec': '0'}, {'vendor': 0}):
            self.assertEqual(p.interpret_ec(ec)['status'], 'unknown')
        for ec in ({'ec': 0, 'value': 7}, {'code': 2}, {'ec': 0, 'success': False}):
            self.assertEqual(p.interpret_ec(ec)['status'], 'error')

    def test_opaque_ec_is_preserved_and_fails_closed(self):
        ec = {('a', 1): [float('nan')], 'opaque': object()}
        raw = p.raw_evidence(ec)
        self.assertFalse(raw['json_exact'])
        self.assertEqual(len(raw['typed_fallback']['entries']), 2)
        self.assertEqual(p.interpret_ec(ec)['status'], 'unknown')

    def test_no_cli_flag_cannot_import_sdk(self):
        with patch.object(p, 'load_pinned_sdk', side_effect=AssertionError('must not load')), redirect_stdout(io.StringIO()):
            self.assertEqual(p.main([]), 2)
            with self.assertRaises(SystemExit) as exited:
                p.main(['--help'])
            self.assertEqual(exited.exception.code, 0)

    def test_dispatch_authorization_and_retry_guards(self):
        robot, probe, _, _ = self.exercise()
        for api in ('moveAppend', 'moveStart', 'stop', 'pause', 'setPowerState', 'setOperateMode'):
            with self.assertRaises(p.ProbeFailure):
                probe.call_ec(api)
        # Successful prior calls cannot be repeated (checked before any log I/O).
        for api in ('setMotionControlMode', 'moveReset'):
            with self.assertRaises(p.ProbeFailure):
                probe.call_ec(api)
        self.assertEqual(robot.reset_count, 1)

    def test_unauthorized_calls_rejected_before_io(self):
        clock = Clock()
        robot = FakeRobot(clock)
        probe = p.Probe(robot, SDK, None, clock=clock)
        for api in ('setMotionControlMode', 'moveReset'):
            with self.assertRaises(p.ProbeFailure):
                probe.call_ec(api)
        self.assertEqual((robot.mode_count, robot.reset_count), (0, 0))

    def test_log_failure_prevents_setup_or_followup_and_cleanup_still_runs(self):
        for phase in ('request', 'return'):
            with tempfile.TemporaryDirectory() as temp:
                store = p.Store(Path(temp) / 'session')
                original_write = store.write

                def faulty_write(name, record):
                    if name == 'calls' and record.get('api') == 'setMotionControlMode' and record['phase'] == phase:
                        raise OSError('synthetic disk failure')
                    original_write(name, record)

                store.write = faulty_write
                clock = Clock()
                robot = FakeRobot(clock)
                probe = p.Probe(robot, SDK, store, clock=clock, ask=lambda _: p.SETUP_PHRASE)
                with redirect_stdout(io.StringIO()):
                    final = probe.run()
                store.close()
                self.assertEqual(final['status'], 'FAILED')
                self.assertEqual(robot.mode_count, 0 if phase == 'request' else 1)
                self.assertEqual(robot.reset_count, 0)
                self.assertEqual(robot.calls[-2:], ['unsubscribe', 'disconnect'])

    def test_ast_no_forbidden_paths_or_neutral_import(self):
        tree = ast.parse(Path(p.__file__).read_text())
        robot_calls = {node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call)
                       and isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Attribute)
                       and node.func.value.attr == 'robot'}
        expected = p.EC_APIS | {'connectToRobot', 'startReceiveRobotState', 'stopReceiveRobotState',
                               'updateRobotState', 'getStateData'}
        self.assertEqual(robot_calls, expected)
        self.assertFalse(any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                             and node.func.id in ('getattr', 'eval', 'exec')
                             and (node.func.id != 'getattr' or ast.unparse(node.args[0]) == 'self.robot')
                             for node in ast.walk(tree)))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                self.assertNotIn('neutral', node.module or '')
            if isinstance(node, ast.Import):
                self.assertTrue(all('neutral' not in alias.name and 'xCoreSDK' not in alias.name for alias in node.names))


if __name__ == '__main__':
    unittest.main()
