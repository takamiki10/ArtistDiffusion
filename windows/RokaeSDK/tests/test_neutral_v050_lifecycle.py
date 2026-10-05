"""Full neutral lifecycle with fake SDK and clock only. No vendor imports."""
from contextlib import redirect_stdout
from enum import Enum
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import neutral_v050_nrt_runner as n
from test_neutral_v050_nrt import JointPosition, MoveAbsJCommand, PyErrorCode


class Power(Enum):
    on = 0
    off = 1


class Mode(Enum):
    manual = 0
    automatic = 1


class Operation(Enum):
    idle = 0
    moving = 9
    unknown = -1


class Clock:
    def __init__(self):
        self.ns = 1_000_000_000

    def counter_ns(self):
        return self.ns

    def counter(self):
        return self.ns / 1e9


class Vector:
    q = [0.0] * 6
    def content(self):
        return self.q.copy()


class String:
    value = ''

    def content(self):
        return self.value


class FakeRobot:
    def __init__(self, clock, fault):
        self.clock, self.fault = clock, fault
        self.calls = []
        self.started = False
        self.event_sent = False
        self.remark_sent = False
        self.last_control = None
        self.samples_since_control = 0
        self.commands = []

    def record(self, name, ec=None):
        self.calls.append(name)
        if ec is not None:
            ec.update(ec=0, message='success')

    def connectToRobot(self, *args):
        self.record('connect')

    def robotInfo(self, ec):
        ec.update(ec=0, message='success')
        return types.SimpleNamespace(id='fake', type=n.EXPECTED_ROBOT_TYPE, joint_num=6, version='test-version')

    def operateMode(self, ec):
        ec.update(ec=0, message='success')
        return Mode.automatic if self.started and self.fault == 'start_mode_change' else Mode.manual

    def powerState(self, ec):
        ec.update(ec=0, message='success')
        return Power.off if self.started and self.fault == 'start_power_off' else Power.on

    def operationState(self, ec):
        ec.update(ec=0, message='success')
        if self.fault == 'remark_then_state_fault' and self.remark_sent:
            return Operation.unknown
        return Operation.idle

    def setEventWatcher(self, kind, callback, ec):
        self.callback = callback
        self.record('watch', ec)

    def startReceiveRobotState(self, interval, fields):
        self.record('subscribe')

    def updateRobotState(self, timeout):
        if timeout.total_seconds() == 0:
            return 0
        self.samples_since_control += 1
        self.clock.ns += 8_000_000
        if self.fault == 'post_reset_gap' and self.last_control == 'reset' and self.samples_since_control == 12:
            self.clock.ns += 60_000_000
        if self.fault in ('motion_gap', 'start_gap_then_gap') and self.started and self.samples_since_control == 15:
            self.clock.ns += 60_000_000
        if self.started and not self.event_sent and self.fault != 'no_terminal':
            if self.fault in ('remark_then_terminal', 'remark_then_state_fault', 'remark_then_error') and not self.remark_sent:
                for index in (1, 2):
                    self.callback({'cmdID': 'ack', 'wayPointIndex': index, 'reachTarget': True,
                                   'error': {'ec': 0, 'message': 'success'}, 'remark': 'adjacent points',
                                   'customInfo': self.commands[index].customInfo})
                self.remark_sent = True
                return 48
            self.callback({'cmdID': 'ack', 'wayPointIndex': 99, 'reachTarget': True,
                           'error': {'ec': 5, 'message': 'failure'} if self.fault in ('remark_then_error', 'start_hard_event') else {'ec': 0, 'message': 'success'}, 'remark': '',
                           'customInfo': self.commands[-1].customInfo})
            self.event_sent = True
        return 48

    def getStateData(self, field, values, size):
        values.q = [0.0] * 6
        if self.started:
            if self.samples_since_control == 1 and self.fault == 'start_displaced':
                values.q[5] = 1.00001e-5  # Any joint, not only the selected joint.
            if self.samples_since_control == 1 and self.fault == 'start_invalid':
                return 1
            if 2 <= self.samples_since_control <= 5 and self.fault not in ('no_motion', 'start_gap_no_motion', 'wrong_direction'):
                values.q[0] = 1e-4
            if self.fault == 'wrong_direction' and 2 <= self.samples_since_control <= 5:
                values.q[0] = -1e-3
            if self.fault == 'no_settle' and self.samples_since_control >= 2:
                values.q[0] = 1e-3
        return 0

    def control(self, name, ec):
        self.record(name, ec)
        self.last_control = name
        self.samples_since_control = 0
        if name != 'start' or self.fault in ('start_gap', 'start_displaced', 'start_invalid', 'start_gap_no_motion',
                                            'start_gap_then_gap', 'start_power_off', 'start_mode_change', 'start_hard_event'):
            self.clock.ns += 64_000_000
        if self.fault == name + '_error':
            ec['ec'] = 9

    def setMotionControlMode(self, mode, ec):
        self.control('setup', ec)

    def moveReset(self, ec):
        self.control('reset', ec)

    def moveAppend(self, commands, cmd_id, ec):
        self.commands = commands
        cmd_id.value = 'ack'
        self.control('append', ec)

    def moveStart(self, ec):
        self.started = True
        self.control('start', ec)

    def stopReceiveRobotState(self):
        self.record('unsubscribe')

    def setNoneEventWatcher(self, kind, ec):
        self.record('unwatch', ec)

    def disconnectFromRobot(self, ec):
        self.record('disconnect', ec)


class LifecycleTests(unittest.TestCase):
    def exercise(self, fault=None, denied=None):
        clock = Clock()
        robot = FakeRobot(clock, fault)
        constructed = []

        def factory():
            constructed.append(True)
            return robot

        sdk = types.SimpleNamespace(xMateRobot=factory, JointPosition=JointPosition,
                                    MoveAbsJCommand=MoveAbsJCommand, PyString=String,
                                    PyTypeVectorDouble=Vector, PyErrorCode=PyErrorCode,
                                    PowerState=Power, OperateMode=Mode, OperationState=Operation,
                                    Event=types.SimpleNamespace(moveExecution=0),
                                    MotionControlMode=types.SimpleNamespace(NrtCommandMode=1))
        with tempfile.TemporaryDirectory() as temp, \
                patch.object(n.time, 'perf_counter_ns', clock.counter_ns), \
                patch.object(n.time, 'perf_counter', clock.counter), \
                patch('builtins.input', return_value=n.PRECONNECTION_PHRASE), redirect_stdout(io.StringIO()):
            store = n.SessionStore(Path(temp) / 'session')
            session = n.PreparedSession(sdk, store)

            def prompt(text):
                session.pump()
                if text.startswith('Type exactly '):
                    phrase = text[len('Type exactly '):-2]
                    return 'CANCEL' if phrase.startswith(str(denied) + ' ') else phrase
                return {'Joint': '1', 'Direction': '+', 'Excursion': '0.25'}[text.split()[0]]

            session.prompt = prompt
            error = None
            try:
                session.execute()
            except Exception as exc:
                error = str(exc)
            store.close()
            logs = {name: [json.loads(line) for line in (store.folder / (name + '.jsonl')).read_text().splitlines()]
                    for name in ('q_m', 'lifecycle', 'calls', 'authorizations', 'motion_events')}
            final = json.loads((store.folder / 'final_status.json').read_text())
        self.assertEqual(len(constructed), 1)
        self.assertEqual(robot.calls.count('connect'), 1)
        return robot, final, logs, error

    def test_complete_order_with_64ms_premotion_calls(self):
        robot, final, logs, error = self.exercise()
        self.assertIsNone(error)
        self.assertEqual(final['status'], 'COMPLETED_ACCEPTED')
        self.assertFalse(final['policy_validation_failed'])
        self.assertFalse(final['rd_motion_authorized'])
        control_calls = [x for x in robot.calls if x in ('setup', 'reset', 'append', 'start')]
        self.assertEqual(control_calls, ['setup', 'reset', 'append', 'start'])
        self.assertEqual(final['attempts'], {k: 1 for k in n.CONTROL_APIS})
        self.assertEqual(len(robot.commands), 100)
        self.assertEqual([c.zone for c in robot.commands], [1] * 99 + [0])
        self.assertTrue(all(c.jointSpeed == .05 and c.speed == 50 for c in robot.commands))
        passes = [r for r in logs['lifecycle'] if r.get('code') == 'post_call_stationarity_passed']
        self.assertEqual([r['api'] for r in passes], ['setMotionControlMode', 'moveReset', 'moveAppend'])
        requests = {r['api']: r['request_ns'] for r in logs['calls'] if r['phase'] == 'return' and r['api'] in n.CONTROL_APIS}
        self.assertLess(passes[1]['host_ns'], requests['moveAppend'])
        self.assertLess(passes[2]['host_ns'], requests['moveStart'])
        for row in passes:
            self.assertGreaterEqual(row['last_sequence'] - row['first_fresh_sequence'], 125)

    def test_premotion_gaps_preserved_with_samples(self):
        _, _, logs, _ = self.exercise()
        boundaries = [r for r in logs['lifecycle'] if r.get('code') == 'call_sampling_boundary_completed']
        self.assertEqual(len(boundaries), 4)
        for r in boundaries[:3]:
            self.assertEqual(r['duration_ns'], 64_000_000)
            self.assertGreater(r['cross_call_gap_ns'], 50_000_000)
            self.assertTrue(r['premotion_gap_exemption'])
            self.assertEqual(r['first_q_after']['q_native_rad'], r['last_q_before']['q_native_rad'])
        self.assertFalse(boundaries[-1]['premotion_gap_exemption'])

    def test_post_reset_window_gap_fails_before_append(self):
        robot, final, _, error = self.exercise('post_reset_gap')
        self.assertIn('gap >50 ms', error)
        self.assertEqual(final['status'], 'FAILED_EXECUTION')
        self.assertFalse(final['setup_complete'])
        self.assertNotIn('append', robot.calls)
        self.assertNotIn('start', robot.calls)

    def test_stationary_start_gap_accepted_only_for_geometry(self):
        robot, final, logs, error = self.exercise('start_gap')
        self.assertIsNone(error)
        self.assertEqual(final['status'], 'COMPLETED_ACCEPTED')
        self.assertTrue(final['valid_for_geometric_analysis'])
        self.assertFalse(final['valid_for_timing_analysis'])
        self.assertFalse(final['start_timing_observability_passed'])
        self.assertEqual(final['start_boundary']['reason'], 'START_BOUNDARY_STATIONARY_UNOBSERVED_INTERVAL')
        self.assertEqual(final['start_boundary']['pre_post_delta_rad'], [0.0]*6)
        self.assertEqual(final['motion_observation']['first_threshold_sample']['signed_displacement_rad'], 1e-4)
        self.assertGreater(final['max_motion_gap_ns'], 50_000_000)
        gap = next(r for r in logs['lifecycle'] if r.get('code') == 'call_sampling_boundary_completed' and r['api'] == 'moveStart')
        self.assertFalse(gap['premotion_gap_exemption'])
        self.assertGreater(gap['cross_call_gap_ns'], 50_000_000)
        self.assertEqual(robot.calls[robot.calls.index('start') + 1:], ['unsubscribe', 'unwatch', 'disconnect'])

    def test_gap_during_motion_fails_even_after_terminal_event(self):
        _, final, _, error = self.exercise('motion_gap')
        self.assertIn('gap >50 ms', error)
        self.assertEqual(final['status'], 'FAILED_EXECUTION')

    def test_each_stage_denied_prevents_its_control_call(self):
        for stage, prohibited in [('SETUP', 'setup'), ('APPEND', 'append'), ('START', 'start')]:
            with self.subTest(stage=stage):
                robot, final, _, error = self.exercise(denied=stage)
                self.assertIn('not authorized', error)
                self.assertNotIn(prohibited, robot.calls)
                self.assertEqual(final['status'], 'FAILED_EXECUTION')

    def test_control_error_never_retries_or_proceeds(self):
        for fault, prohibited in [('setup_error', 'reset'), ('reset_error', 'append'), ('append_error', 'start')]:
            robot, final, _, error = self.exercise(fault)
            self.assertIn('error structure/code', error)
            self.assertNotIn(prohibited, robot.calls)
            self.assertEqual(robot.calls.count(fault.split('_')[0]), 1)
            self.assertEqual(final['status'], 'FAILED_EXECUTION')

    def test_no_reset_setup_after_motion_even_with_authorization(self):
        session = n.PreparedSession(types.SimpleNamespace(PyErrorCode=PyErrorCode), None)
        session.start_attempted = True
        session.authorized.add('SETUP')
        session.mode_succeeded = True
        for api in ('setMotionControlMode', 'moveReset'):
            with self.assertRaises(n.ValidationError):
                session.checked_call(api)
            with self.assertRaises(n.ValidationError):
                session.premotion_call(api)
        with self.assertRaises(n.ValidationError):
            session.premotion_call('moveStart')

    def test_remark_keeps_logging_until_terminal_settling_but_rejects(self):
        robot, final, logs, error = self.exercise('remark_then_terminal')
        self.assertIsNone(error)
        self.assertEqual(final['status'], 'COMPLETED_REJECTED')
        self.assertTrue(final['policy_validation_failed'])
        self.assertFalse(final['hard_event_fault'])
        self.assertFalse(final['rd_motion_authorized'])
        self.assertEqual([r['wayPointIndex'] for r in final['policy_failures']], [1, 2])
        self.assertTrue(all(r['remark'] == 'adjacent points' for r in final['policy_failures']))
        self.assertEqual([r['payload']['wayPointIndex'] for r in logs['motion_events']], [1, 2, 99])
        samples_after_warning = [r for r in logs['q_m'] if r['decoded_end_ns'] > final['policy_failures'][0]['host_ns']]
        self.assertGreaterEqual(len(samples_after_warning), 100)
        self.assertGreaterEqual(samples_after_warning[-1]['decoded_end_ns'] - final['terminal_event_ns'], 1_000_000_000)
        self.assertEqual(robot.calls[robot.calls.index('start') + 1:], ['unsubscribe', 'unwatch', 'disconnect'])

    def test_start_displaced_or_invalid_or_state_changed_rejected(self):
        for fault, reason in [('start_displaced', 'START_BOUNDARY_MOTION_UNOBSERVED'),
                              ('start_invalid', 'START_BOUNDARY_INVALID_POST_Q'),
                              ('start_power_off', 'existing power state is not on'),
                              ('start_mode_change', 'operating mode changed')]:
            with self.subTest(fault=fault):
                _, final, _, error = self.exercise(fault)
                self.assertIn(reason, error)
                self.assertEqual(final['status'], 'FAILED_EXECUTION')
                self.assertFalse(final['start_boundary']['accepted'])
                self.assertEqual(final['start_boundary']['reason'], reason)

    def test_observed_signed_motion_is_mandatory(self):
        for fault in ('no_motion', 'start_gap_no_motion', 'wrong_direction'):
            _, final, _, error = self.exercise(fault)
            self.assertEqual(error, 'NO_OBSERVED_NEUTRAL_MOTION')
            self.assertEqual(final['status'], 'FAILED_EXECUTION')
            self.assertTrue(final['start_boundary']['accepted'])

    def test_accepted_start_boundary_does_not_exempt_later_gap(self):
        _, final, _, error = self.exercise('start_gap_then_gap')
        self.assertIn('gap >50 ms', error)
        self.assertTrue(final['start_boundary']['accepted'])
        self.assertEqual(final['status'], 'FAILED_EXECUTION')
        self.assertGreater(final['max_post_boundary_gap_ns'], 50_000_000)

    def test_terminal_and_settling_still_mandatory(self):
        for fault in ('no_terminal', 'no_settle'):
            _, final, _, error = self.exercise(fault)
            self.assertIn('terminal event and subsequent settling not confirmed', error)
            self.assertEqual(final['status'], 'FAILED_EXECUTION')

    def test_normal_start_boundary(self):
        _, final, _, error = self.exercise()
        self.assertIsNone(error)
        self.assertEqual(final['start_boundary']['reason'], 'START_BOUNDARY_WITHIN_50_MS')
        self.assertTrue(final['start_timing_observability_passed'])

    def test_start_error_and_boundary_hard_event_cannot_be_accepted(self):
        for fault in ('start_error', 'start_hard_event'):
            _, final, logs, error = self.exercise(fault)
            self.assertIsNotNone(error)
            self.assertEqual(final['status'], 'FAILED_EXECUTION')
            self.assertFalse(final['start_boundary']['accepted'])
            if fault == 'start_hard_event':
                self.assertTrue(final['hard_event_fault'])
                self.assertTrue(logs['motion_events'])

    def test_boundary_pre_q0_tolerance_is_not_relaxed(self):
        clock = Clock()
        robot = FakeRobot(clock, None)
        sdk = types.SimpleNamespace(PyErrorCode=PyErrorCode)
        with tempfile.TemporaryDirectory() as temp:
            store = n.SessionStore(Path(temp) / 'session')
            session = n.PreparedSession(sdk, store)
            session.q0 = [0.0]*6
            session.start_succeeded = True
            pre = {'q_native_rad': [0.0]*5+[1.00001e-5]}
            row = {'valid_q': True, 'q_native_rad': [0.0]*6}
            boundary = {'last_q_before': pre, 'cross_call_gap_ns': 60_000_000}
            with self.assertRaisesRegex(n.ValidationError, 'START_BOUNDARY_PRE_Q_NOT_AT_Q0'):
                session.review_start_boundary(boundary, row)
            self.assertFalse(boundary['accepted'])
            store.close()

    def test_hard_event_after_warning_overrides_policy_completion(self):
        _, final, logs, error = self.exercise('remark_then_error')
        self.assertIsNotNone(error)
        self.assertEqual(final['status'], 'FAILED_EXECUTION')
        self.assertTrue(final['policy_validation_failed'])
        self.assertTrue(final['hard_event_fault'])
        self.assertFalse(final['rd_motion_authorized'])
        self.assertEqual(len(logs['motion_events']), 3)

    def test_state_fault_after_remark_stops_passive_observation(self):
        robot, final, _, error = self.exercise('remark_then_state_fault')
        self.assertIn('inconsistent operation state', error)
        self.assertEqual(final['status'], 'FAILED_EXECUTION')
        self.assertTrue(final['policy_validation_failed'])
        self.assertFalse(final['rd_motion_authorized'])
        self.assertEqual(robot.calls[robot.calls.index('start') + 1:], ['unsubscribe', 'unwatch', 'disconnect'])


if __name__ == '__main__':
    unittest.main()
