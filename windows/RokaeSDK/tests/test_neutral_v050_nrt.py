"""Offline tests. Never construct a vendor robot or invoke a control API."""
import ast
import copy
import json
import math
from pathlib import Path
import struct
import tempfile
import types
import unittest
from unittest.mock import patch

import neutral_v050_nrt_runner as n
from nrt_v050.adjacent_spacing_audit import legacy_neutral, joint_space_spacing, build_audit


class JointPosition:
    def __init__(self, joints):
        self.joints = list(joints)
        self.external = []


class MoveAbsJCommand:
    def __init__(self, target, speed, zone):
        self.target = target
        self.speed = speed
        self.zone = zone


class PyErrorCode:
    def value(self):
        return 0

    def message(self):
        return "success"


def forbidden_robot():
    raise AssertionError("Tests must never construct a physical robot")


SDK = types.SimpleNamespace(JointPosition=JointPosition, MoveAbsJCommand=MoveAbsJCommand,
                            PyErrorCode=PyErrorCode, xMateRobot=forbidden_robot)


def q_records(span_ns=1_008_000_000, gap=8_000_000):
    return [{"decoded_end_ns": t, "q_native_rad": [0.0] * 6}
            for t in range(1, span_ns + 2, gap)]


def event(i=99, reach=True, **changes):
    p = {"cmdID": "ack", "wayPointIndex": i, "reachTarget": reach,
         "error": {"ec": 0, "message": "success"}, "remark": "",
         "customInfo": f"test:k{i:03d}"}
    p.update(changes)
    return {"host_ns": 20, "payload": p}


class NeutralTests(unittest.TestCase):
    def setUp(self):
        self.q0 = [-0.0, 1.2, -2.0, 0.03, -0.9, 0.2]
        self.rows = n.generate_neutral(self.q0, 2, "+", 0.25)

    def test_all_joints_both_directions(self):
        for joint in range(1, 7):
            for direction in ("+", "-"):
                with self.subTest(joint=joint, direction=direction):
                    q0 = self.q0.copy()
                    q0[joint - 1] = -0.0
                    rows = n.generate_neutral(q0, joint, direction, 0.25)
                    self.assertEqual(len(rows), 100)
                    self.assertEqual(n.joint_bytes(rows[0]), n.joint_bytes(q0))
                    self.assertEqual(n.joint_bytes(rows[-1]), n.joint_bytes(q0))
                    self.assertEqual(len(n.matrix_bytes(rows)), 4800)
                    displacement = [abs(r[joint - 1] - q0[joint - 1]) for r in rows]
                    self.assertLessEqual(max(displacement), math.radians(0.25))
                    self.assertEqual(displacement[49], math.radians(.25))
                    self.assertTrue(all(a < b for a, b in zip(displacement[:49], displacement[1:50])))
                    self.assertTrue(all(a > b for a, b in zip(displacement[49:99], displacement[50:])))
                    self.assertTrue(all(a != b for a, b in zip(rows, rows[1:])))
                    for row in rows:
                        for j in range(6):
                            self.assertIs(type(row[j]), float)
                            if j != joint - 1:
                                self.assertEqual(struct.pack('<d', row[j]), struct.pack('<d', self.q0[j]))

    def test_triangle_unique_peak_and_nonzero_spacing(self):
        q0 = [0.0] * 6
        rows = n.generate_neutral(q0, 1, '+', .1)
        old = legacy_neutral(q0, 1, '+', .1)
        new_d = n.neutral_spacing_diagnostics(rows, q0, 1)
        old_d = n.neutral_spacing_diagnostics(old, q0, 1)
        self.assertEqual([i for i, row in enumerate(rows) if row[0] == max(r[0] for r in rows)], [49])
        self.assertEqual(rows[49][0], math.radians(.1))
        self.assertNotEqual(rows[49], rows[50])
        self.assertTrue(all(a != b for a, b in zip(rows, rows[1:])))
        self.assertTrue(all(d > 0 for d in new_d['adjacent_selected_joint_rad']))
        self.assertEqual(n.joint_bytes(rows[0]), n.joint_bytes(q0))
        self.assertEqual(n.joint_bytes(rows[99]), n.joint_bytes(q0))
        for i, row in enumerate(rows[1:-1], 1):
            a = math.radians(.1)
            self.assertEqual(row[0], a if i == 49 else a*i/49 if i < 49 else a*(99-i)/50)
        self.assertAlmostEqual(math.degrees(rows[50][0]), .098, places=15)
        self.assertAlmostEqual(new_d['first_interior_displacement']['deg'], .1/49, places=15)
        self.assertGreater(new_d['first_interior_displacement']['rad'], 20000 * old_d['first_interior_displacement']['rad'])
        self.assertGreater(new_d['minimum_nonzero_adjacent']['rad'], 900 * old_d['minimum_nonzero_adjacent']['rad'])
        self.assertEqual(new_d['zero_adjacent_pairs'], [])
        self.assertAlmostEqual(new_d['minimum_adjacent_including_zero']['deg'], .002, places=15)
        self.assertAlmostEqual(new_d['maximum_adjacent']['deg'], .1/49, places=15)

    def test_triangle_rejects_collapsed_pair_and_binary64_bound_violation(self):
        rows = copy.deepcopy(self.rows)
        rows[50] = rows[49].copy()
        with self.assertRaises(n.ValidationError):
            n.validate_neutral(rows, self.q0, 2, '+', .25)
        with self.assertRaisesRegex(n.ValidationError, 'exceeds excursion'):
            n.generate_neutral([.2]*6, 1, '+', .25)

    def test_spacing_audit_known_fixture_and_frozen_batches(self):
        rows = [[float(i)] + [0.0] * 5 for i in range(100)]
        rows[1] = rows[0].copy()
        diagnostic = joint_space_spacing(rows)
        self.assertEqual(diagnostic['minimum_l2_rad'], 0.0)
        self.assertEqual(diagnostic['median_l2_rad'], 1.0)
        self.assertEqual(diagnostic['maximum_l2_rad'], 2.0)
        self.assertEqual(diagnostic['duplicate_pairs'], [[0, 1]])
        audit = build_audit()  # Reads only pinned prepared files; writes nothing.
        self.assertTrue(audit['frozen_files_unchanged'])
        self.assertFalse(audit['physical_rd_outcomes_used'])
        for batch in audit['path_0003'].values():
            self.assertEqual(batch['waypoint_count'], 100)
            self.assertEqual(len(batch['smallest_five']), 5)
            self.assertFalse(batch['exact_duplicate_consecutive_vectors'])

    def test_invalid_choices(self):
        for joint, direction, amount in [(0, "+", .1), (7, "+", .1), (True, "+", .1),
                                         (1, "", .1), (1, "positive", .1), (1, "+", .250000001),
                                         (1, "+", 0.0), (1, "+", -.1), (1, "+", float('nan')),
                                         (1, "+", float('inf')), (1, "+", 1)]:
            with self.subTest(joint=joint, direction=direction, amount=amount):
                with self.assertRaises(n.ValidationError):
                    n.generate_neutral(self.q0, joint, direction, amount)

    def test_bad_q_and_unrepresentable_excursion(self):
        for q in ([0.0] * 5, [0] * 6, [float('nan')] * 6, [float('inf')] * 6):
            with self.assertRaises(n.ValidationError):
                n.generate_neutral(q, 1, "+", .1)
        with self.assertRaises(n.ValidationError):
            n.generate_neutral([1e100] * 6, 1, "+", .25)

    def test_exact_json_binary_hash(self):
        encoded = json.dumps(self.rows, allow_nan=False)
        self.assertEqual(n.matrix_bytes(self.rows), n.matrix_bytes(json.loads(encoded)))
        self.assertEqual(n.digest_json({'b': 1, 'a': 2}), n.digest_json({'a': 2, 'b': 1}))
        altered = copy.deepcopy(self.rows)
        altered[8][1] = math.nextafter(altered[8][1], math.inf)
        self.assertNotEqual(n.digest_bytes(n.matrix_bytes(self.rows)), n.digest_bytes(n.matrix_bytes(altered)))

    def test_commands_valid(self):
        c = n.construct_commands(SDK, self.rows, "test")
        snapshots = n.validate_commands(SDK, c, self.rows, "test")
        self.assertEqual([s['zone'] for s in snapshots], [1] * 99 + [0])
        self.assertEqual(snapshots[99]['customInfo'], 'test:k099')

    def test_command_corruption(self):
        mutations = [lambda c: c.pop(), lambda c: c.append(c[0]),
                     lambda c: setattr(c[20], 'speed', 49),
                     lambda c: setattr(c[20], 'speed', True),
                     lambda c: setattr(c[20], 'jointSpeed', .1),
                     lambda c: setattr(c[99], 'zone', 1),
                     lambda c: setattr(c[98], 'zone', 0),
                     lambda c: setattr(c[3], 'customInfo', 'test:k004'),
                     lambda c: c[4].target.joints.__setitem__(2, float('nan')),
                     lambda c: c[4].target.joints.__setitem__(2, math.nextafter(-2.0, math.inf)),
                     lambda c: c[4].target.external.append(0.0),
                     lambda c: c.__setitem__(0, object()),
                     lambda c: c.__setitem__(1, c[0])]
        for mutate in mutations:
            c = n.construct_commands(SDK, self.rows, "test")
            mutate(c)
            with self.assertRaises(n.ValidationError):
                n.validate_commands(SDK, c, self.rows, "test")

    def test_rounding_binding_rejected(self):
        class RoundedJoint(JointPosition):
            def __init__(self, joints):
                super().__init__([round(q, 3) for q in joints])
        sdk = types.SimpleNamespace(JointPosition=RoundedJoint, MoveAbsJCommand=MoveAbsJCommand)
        with self.assertRaises(n.ValidationError):
            n.construct_commands(sdk, self.rows, "test")

    def test_signed_zero_roundtrip_rejected(self):
        c = n.construct_commands(SDK, self.rows, "test")
        c[0].target.joints[0] = 0.0
        with self.assertRaises(n.ValidationError):
            n.validate_commands(SDK, c, self.rows, "test")

    def test_error_adapter_fail_closed(self):
        for error in ({}, None, 0, False, {'message': 'ok'}, {'ec': False, 'message': 'ok'},
                      {'ec': 0, 'value': 0, 'message': 'ok'}, {'value': 1, 'message': 'bad'},
                      {'value': '0', 'message': 'ok'}):
            with self.assertRaises(n.ValidationError):
                n.success_error(error)
        n.success_error({'ec': 0, 'message': 'success'})
        for unexpected in ({'value': 0, 'message': 'success'}, {'ec': 0, 'message': 'ok'},
                           n.snapshot(PyErrorCode(), SDK)):
            with self.assertRaises(n.ValidationError):
                n.success_error(unexpected)

    def test_completion_requires_terminal_true(self):
        c = n.Completion('ack', 'test', 10)
        c.observe(event(0, False))
        c.observe(event(99, False))
        self.assertIsNone(c.terminal_ns)
        c.observe(event())
        self.assertEqual(c.terminal_ns, 20)
        self.assertEqual(c.completed_status(), 'COMPLETED_ACCEPTED')

    def test_zero_error_remark_latches_rejection_without_exception(self):
        c = n.Completion('ack', 'test', 10)
        c.observe(event(1, True, remark='adjacent points'))
        self.assertFalse(c.hard_execution_fault)
        self.assertEqual(c.policy_failures[0]['remark'], 'adjacent points')
        self.assertEqual(c.policy_failures[0]['wayPointIndex'], 1)
        self.assertIsNone(c.terminal_ns)
        c.observe(event())
        self.assertEqual(c.completed_status(), 'COMPLETED_REJECTED')

    def test_nonzero_error_remains_hard_even_with_remark(self):
        c = n.Completion('ack', 'test', 10)
        with self.assertRaises(n.ValidationError):
            c.observe(event(1, True, remark='adjacent points', error={'ec': 7, 'message': 'failure'}))
        self.assertTrue(c.hard_execution_fault)
        with self.assertRaises(n.ValidationError):
            c.completed_status()

    def test_bad_events_fail(self):
        cases = [event(cmdID='other'), event(reachTarget=1), event(100), event(-1),
                 event(wayPointIndex=True), event(customInfo='other'), event(remark=17),
                 event(error={'value': 2, 'message': 'fault'}), event(error=None),
                 {'host_ns': 0, 'payload': event()['payload']},
                 {'host_ns': 20, 'payload': {}}, {'host_ns': 20, 'payload': None}]
        for bad in cases:
            with self.subTest(event=bad):
                with self.assertRaises(n.ValidationError):
                    n.Completion('ack', 'test', 10).observe(bad)

    def test_event_duplicates_regression_and_late_error(self):
        for later in (event(), event(98), event(error={'value': 1, 'message': 'fault'})):
            c = n.Completion('ack', 'test', 10)
            c.observe(event())
            with self.assertRaises(n.ValidationError):
                c.observe(later)

    def test_callback_copy_and_overflow(self):
        inbox = n.EventInbox(SDK)
        payload = event()['payload']
        inbox.callback(payload)
        payload['error']['ec'] = 7
        self.assertEqual(inbox.items.get_nowait()['payload']['error']['ec'], 0)
        for _ in range(n.SETTINGS['event_queue_capacity'] + 1):
            inbox.callback({})
        self.assertTrue(inbox.failed.is_set())

    def test_settle_span_gap_duration_and_reference(self):
        records = q_records()
        self.assertTrue(n.settled(records))
        self.assertFalse(n.settled(records[:50]))
        self.assertFalse(n.settled(records, after_ns=records[-50]['decoded_end_ns']))
        self.assertFalse(n.settled(records, reference=[.001] * 6))
        records[40]['q_native_rad'][1] = 1e-3
        self.assertFalse(n.settled(records))
        records = q_records()
        del records[50:60]
        self.assertFalse(n.settled(records))

    def test_policy_pin(self):
        raw, policy = n.read_policy()
        self.assertEqual(n.digest_bytes(raw), n.POLICY_SHA256)
        self.assertEqual(policy['jointSpeed'], .05)

    def test_cli_requires_deliberate_flag_without_sdk_import(self):
        with patch.object(n, 'load_exact_sdk_data_only', side_effect=AssertionError('no import')), patch('builtins.print'):
            self.assertEqual(n.main([]), 2)

    def test_store_preserves_raw_values(self):
        with tempfile.TemporaryDirectory() as temp:
            store = n.SessionStore(Path(temp) / 'session')
            store.write('q_m', {'q_native_rad': self.q0, 'decoded_end_ns': 123456789012345})
            store.json('final_status.json', {'status': 'OFFLINE_TEST'})
            store.barrier()
            store.close()
            record = json.loads((store.folder / 'q_m.jsonl').read_text())
            self.assertEqual(n.joint_bytes(record['q_native_rad']), n.joint_bytes(self.q0))
            self.assertEqual(record['decoded_end_ns'], 123456789012345)

    def test_separate_authorization_phrases(self):
        with tempfile.TemporaryDirectory() as temp:
            store = n.SessionStore(Path(temp) / 'session')
            session = n.PreparedSession(SDK, store)
            session.prompt = lambda _: 'APPEND NEUTRAL abc'
            session.authorize('APPEND', 'abc')
            with self.assertRaises(n.ValidationError):
                session.authorize('START', 'abc')
            session.prompt = lambda _: 'START NEUTRAL changed-hash'
            with self.assertRaises(n.ValidationError):
                session.authorize('START', 'abc')
            store.close()

    def test_state_logger_invalid_q_preserved(self):
        class Vector:
            def content(self):
                return [float('nan')] + [0.0] * 5
        robot = types.SimpleNamespace(updateRobotState=lambda _: 48,
                                      getStateData=lambda *args: 0)
        sdk = types.SimpleNamespace(PyTypeVectorDouble=Vector, PyErrorCode=PyErrorCode)
        with tempfile.TemporaryDirectory() as temp:
            store = n.SessionStore(Path(temp) / 'session')
            session = n.PreparedSession(sdk, store)
            session.robot = robot  # synthetic namespace, no vendor object
            with self.assertRaises(n.ValidationError):
                session.read_sample()
            store.close()
            row = json.loads((store.folder / 'q_m.jsonl').read_text())
            self.assertFalse(row['valid_q'])
            self.assertEqual(row['q_native_rad'][0], {'nonfinite': 'nan'})

    def test_disk_failure_prevents_control_call(self):
        class BrokenStore:
            def write(self, *args):
                raise OSError('synthetic full disk')
        called = []
        session = n.PreparedSession(SDK, BrokenStore())
        session.setup_complete = True
        session.authorized.add('APPEND')
        session.robot = types.SimpleNamespace(moveAppend=lambda *args: called.append('append'))
        with self.assertRaises(OSError):
            session.checked_call('moveAppend', [], None)
        self.assertEqual(called, [])

    def test_live_sequence_failure_cleanup_without_physical_sdk(self):
        """Force a pre-start subscription fault on a strictly synthetic SDK.

        The supplied robot is a Python fake, never a vendor xMateRobot instance.
        """
        called = []

        class FakeRobot:
            def connectToRobot(self, *args):
                called.append('connect')

            def robotInfo(self, ec):
                ec.update(ec=0, message='success')
                return types.SimpleNamespace(id='fake', type=n.EXPECTED_ROBOT_TYPE, version='fake', joint_num=6)

            def setEventWatcher(self, event_type, callback, ec):
                ec.update(ec=0, message='success')
                called.append('watch')

            def startReceiveRobotState(self, *args):
                raise RuntimeError('synthetic subscription failure')

            def stopReceiveRobotState(self):
                called.append('unsubscribe')
                raise RuntimeError('synthetic unsubscribe failure')

            def setNoneEventWatcher(self, event_type, ec):
                called.append('unwatch')
                ec.update(ec=0, message='success')

            def disconnectFromRobot(self, ec):
                called.append('disconnect')
                ec.update(ec=0, message='success')

        sdk = types.SimpleNamespace(xMateRobot=FakeRobot, PyErrorCode=PyErrorCode,
                                    Event=types.SimpleNamespace(moveExecution=0))
        with tempfile.TemporaryDirectory() as temp:
            store = n.SessionStore(Path(temp) / 'session')
            session = n.PreparedSession(sdk, store)
            session.robot_state = lambda **kwargs: None
            with patch('builtins.input', lambda prompt: prompt.split('\n')[1]), patch('builtins.print'):
                with self.assertRaisesRegex(RuntimeError, 'synthetic subscription failure'):
                    session.execute()
            store.close()
            self.assertEqual(called, ['connect', 'watch', 'unsubscribe', 'unwatch', 'disconnect'])
            final = json.loads((store.folder / 'final_status.json').read_text())
            self.assertFalse(final['start_attempted'])
            self.assertEqual(final['status'], 'FAILED_EXECUTION')

    def test_no_unreviewed_control_calls_or_top_level_sdk_import(self):
        tree = ast.parse(Path(n.__file__).read_text(encoding='utf-8'))
        forbidden = {'stop', 'pause', 'setOperateMode',
                     'setPowerState', 'setMaxCacheSize', 'setDefaultSpeed', 'setDefaultZone'}
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                self.assertNotIn(node.func.attr, forbidden)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'checked_call':
                if isinstance(node.args[0], ast.Constant):
                    self.assertNotIn(node.args[0].value, forbidden)
        self.assertFalse(n.CHECKED_APIS & forbidden)
        self.assertNotIn('import xCoreSDK_python', Path(n.__file__).read_text(encoding='utf-8'))


if __name__ == '__main__':
    unittest.main()
