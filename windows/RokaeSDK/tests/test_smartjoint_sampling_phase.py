"""Single-owner observation tests: fake SDK/clock, no vendor robot or network."""
import ast
import copy
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import neutral_v050_nrt_runner as n
import smartjoint_csv_nrt_runner as s
import test_smartjoint_csv_runner as csv_tests

FIXTURE = json.loads((Path(__file__).parent/'fixtures/smartjoint_sampling_gap/recorded_gap.json').read_text())
LATENCIES = {c['api']: c['duration_ns'] for c in FIXTURE['calls']}


class StatusLatencyRobot(csv_tests.CsvRobot):
    """All calls run serially on the calling thread, including simulated delays."""
    def __init__(self, *args):
        super().__init__(*args)
        self.after_status_triplet = False

    def delay(self, name):
        if self.started:
            self.clock.ns += LATENCIES[name]
            if self.fault == 'long_confirmation' and name == 'operateMode':
                self.clock.ns += 200_000_000
            if self.fault == 'confirmation_watchdog' and name == 'operateMode':
                self.clock.ns += 601_000_000_000

    def operateMode(self, ec):
        self.delay('operateMode')
        return super().operateMode(ec)

    def powerState(self, ec):
        self.delay('powerState')
        return super().powerState(ec)

    def operationState(self, ec):
        self.delay('operationState')
        if self.started:
            self.after_status_triplet = True
        value = super().operationState(ec)
        return csv_tests.Operation.moving if self.started and self.fault == 'completion_moving' else value

    def updateRobotState(self, timeout):
        replay = self.after_status_triplet and timeout.total_seconds() != 0
        self.after_status_triplet = False
        result = super().updateRobotState(timeout)
        if replay:
            # Recorded non-query overhead + next sample: 0.7825 ms.
            self.clock.ns += FIXTURE['gap_ns']-sum(LATENCIES.values())-8_000_000
        return result


class SamplingPhaseTests(unittest.TestCase):
    def run_fake(self, fault=None):
        return csv_tests.LifecycleTests().run_fake(fault, robot_type=StatusLatencyRobot)

    def test_recorded_old_serial_polling_reproduces_exact_gap_failure(self):
        self.assertEqual(FIXTURE['gap_ns'],52_339_800)
        self.assertEqual([r['sequence'] for r in FIXTURE['q_samples']],[6769,6770])
        self.assertEqual(LATENCIES,{'operateMode':46_623_200,'powerState':2_389_500,'operationState':2_544_600})
        # The shared legacy pump still implements the former SmartJoint polling.
        # Remove only the new scheduling guard to replay that previous architecture.
        with patch.object(s.CsvSession,'pump',n.PreparedSession.pump), \
             patch.object(s.CsvSession,'checked_call',n.PreparedSession.checked_call):
            status,final,_,logs=self.run_fake()
        self.assertEqual(status,'FAILED_EXECUTION')
        self.assertIn('sampling gap >50 ms',final['failure_reason'])
        self.assertEqual(final['max_post_boundary_gap_ns'],FIXTURE['gap_ns'])
        sample=logs['q_m'][-1]
        self.assertEqual(sample['gap_ns'],FIXTURE['gap_ns'])
        queries=[c for c in sample['sdk_calls_since_previous_sample'] if c['api'] in s.STATUS_QUERIES]
        self.assertEqual({c['api']:c['duration_ns'] for c in queries},LATENCIES)

    def test_new_owner_does_not_schedule_status_during_active_sampling(self):
        status,final,_,logs=self.run_fake()
        self.assertEqual(status,'COMPLETED_ACCEPTED')
        active=next(r['host_ns'] for r in logs['lifecycle'] if r.get('phase')=='ACTIVE_SAMPLING')
        confirm=next(r['host_ns'] for r in logs['lifecycle'] if r.get('phase')=='COMPLETION_CONFIRMATION')
        queries=[c for c in logs['calls'] if c.get('phase')=='return' and c['api'] in s.STATUS_QUERIES]
        self.assertFalse(any(active <= c['request_ns'] < confirm for c in queries))
        self.assertLessEqual(final['max_post_boundary_gap_ns'],50_000_000)
        self.assertTrue(all(r['sampling_critical'] for r in logs['q_m'] if r['phase']=='ACTIVE_SAMPLING'))

    def test_pre_start_still_checks_all_three_statuses(self):
        _,final,_,logs=self.run_fake()
        start=final['start_boundary']['request_ns']
        before=[c['api'] for c in logs['calls'] if c.get('phase')=='return' and c['host_ns']<start]
        for name in s.STATUS_QUERIES:self.assertIn(name,before)

    def test_captured_window_precedes_blocking_completion_and_no_later_q_gap(self):
        status,final,_,logs=self.run_fake('long_confirmation')
        self.assertEqual(status,'COMPLETED_ACCEPTED')
        captured=next(r for r in logs['lifecycle'] if r.get('code')=='FINAL_STATIONARY_WINDOW_CAPTURED')
        calls=[c for c in logs['calls'] if c.get('phase')=='return' and c['api'] in s.STATUS_QUERIES
               and c['request_ns']>=captured['host_ns']]
        self.assertEqual([c['api'] for c in calls],list(s.STATUS_QUERIES))
        self.assertGreater(calls[0]['duration_ns'],200_000_000)
        self.assertGreaterEqual(captured['evidence']['duration_ns'],1_000_000_000)
        self.assertGreaterEqual(captured['evidence']['sample_count'],100)
        self.assertEqual(captured['evidence'],final['final_stationary'])
        self.assertLessEqual(logs['q_m'][-1]['decoded_end_ns'],calls[0]['request_ns'])
        self.assertLessEqual(final['max_post_boundary_gap_ns'],50_000_000)

    def test_real_active_gap_still_fails_and_cleanup_disconnects(self):
        status,final,robot,logs=self.run_fake('motion_gap')
        self.assertEqual(status,'FAILED_EXECUTION')
        self.assertIn('sampling gap >50 ms',final['failure_reason'])
        self.assertGreater(logs['q_m'][-1]['gap_ns'],50_000_000)
        self.assertTrue(logs['q_m'][-1]['sampling_critical'])
        self.assertEqual(robot.calls[-3:],['unsubscribe','unwatch','disconnect'])
        self.assertNotIn('stop',robot.calls)

    def candidate(self):
        # Evidence-only harness: no SDK or robot constructed.
        session=object.__new__(s.CsvSession)
        session.motion_observation={'first_threshold_sample':{'timestamp_ns':1}}
        session.rows=[[0.0]*6 for _ in range(508)]
        session.records=[{'sequence':i,'decoded_end_ns':1+i*8_000_000,'valid_q':True,
                          'q_native_rad':[0.0]*6} for i in range(126)]
        session.robot_state=lambda *a: self.fail('query must not run without qualifying evidence')
        return session

    def test_stationary_window_requires_one_second_and_100_valid_samples(self):
        session=self.candidate()
        session.records=session.records[:-1]
        self.assertFalse(session.confirm_completion(float('inf')))
        session=self.candidate();session.records[10]['valid_q']=False
        self.assertFalse(session.confirm_completion(float('inf')))
        session=self.candidate();session.records=session.records[::2]
        self.assertFalse(session.confirm_completion(float('inf')))

    def test_final_proximity_and_span_required_before_any_status_query(self):
        session=self.candidate()
        for row in session.records:row['q_native_rad'][0]=0.00101
        self.assertFalse(session.confirm_completion(float('inf')))
        session=self.candidate();session.records[5]['q_native_rad'][0]=1.00001e-5
        self.assertFalse(session.confirm_completion(float('inf')))

    def test_active_status_call_guard_fails_before_sdk_access(self):
        session=object.__new__(s.CsvSession)
        session.start_attempted=True;session.phase='ACTIVE_SAMPLING'
        for name in s.STATUS_QUERIES:
            with self.assertRaisesRegex(n.ValidationError,'forbidden'):
                session.checked_call(name,void=False)

    def test_hard_event_still_fatal_during_active_sampling(self):
        status,final,robot,_=self.run_fake('remark_then_error')
        self.assertEqual(status,'FAILED_EXECUTION')
        self.assertTrue(final['hard_event_fault'])
        self.assertEqual(robot.calls[-1],'disconnect')

    def test_persistent_mode_power_and_controller_changes_fail_confirmation(self):
        for fault in ('start_mode_change','start_power_off','remark_then_state_fault','completion_moving'):
            with self.subTest(fault=fault):
                status,final,_,logs=self.run_fake(fault)
                self.assertEqual(status,'FAILED_EXECUTION')
                self.assertIsNone(final['final_stationary'])
                self.assertTrue(any(r.get('code')=='FINAL_STATIONARY_WINDOW_CAPTURED' for r in logs['lifecycle']))

    def test_blocking_confirmation_cannot_accept_after_600_second_deadline(self):
        status,final,_,_=self.run_fake('confirmation_watchdog')
        self.assertEqual(status,'FAILED_EXECUTION')
        self.assertIn('total watchdog',final['failure_reason'])
        self.assertEqual(final['completion_watchdog_s'],600.0)

    def test_gap_accounting_has_previous_time_calls_and_durations(self):
        _,_,_,logs=self.run_fake()
        samples=logs['q_m']
        for previous,current in zip(samples,samples[1:]):
            self.assertEqual(current['previous_sample_timestamp_ns'],previous['decoded_end_ns'])
            self.assertEqual(current['gap_ns'],current['decoded_end_ns']-previous['decoded_end_ns'])
            self.assertTrue(current['sdk_call_since_previous_sample'])
            for call in current['sdk_calls_since_previous_sample']:
                self.assertEqual(call['duration_ns'],call['return_ns']-call['request_ns'])

    def test_cleanup_text_and_single_owner_are_explicit(self):
        text=Path(s.__file__).read_text()
        self.assertNotIn('No automatic stop',text)
        self.assertIn('cleanup disconnect stops ongoing motion',text)
        tree=ast.parse(text)
        constructors=[x for x in ast.walk(tree) if isinstance(x,ast.Call) and isinstance(x.func,ast.Attribute)
                      and x.func.attr=='xMateRobot']
        self.assertEqual(len(constructors),1)
        self.assertNotIn('Thread(',text)
        self.assertEqual(s.OBSERVATION_POLICY['active_gap_limit_s'],n.SETTINGS['settle_max_gap_s'])


if __name__=='__main__':unittest.main()
