"""Pre-start epochs with recorded data and serial fake SDK/clock only."""
import json
from pathlib import Path
import types
import unittest
from unittest.mock import patch

import neutral_v050_nrt_runner as n
import smartjoint_csv_nrt_runner as s
from path_0003_nrt_runner import stationary_evidence
import test_smartjoint_csv_runner as csv_tests

FIXTURE=json.loads((Path(__file__).parent/'fixtures/smartjoint_prestart_gap/recorded_gap.json').read_text())


class EpochRobot(csv_tests.CsvRobot):
    latest=None

    def __init__(self,*args):
        super().__init__(*args)
        EpochRobot.latest=self

    def updateRobotState(self,timeout):
        result=super().updateRobotState(timeout)
        if not self.started and self.last_control=='reset' and timeout.total_seconds()!=0:
            if (self.fault=='pre_gap_once' and self.samples_since_control==12) or (
                    self.fault=='pre_gaps_forever' and self.samples_since_control%10==2):
                self.clock.ns += FIXTURE['gap_ns']-8_000_000
        return result

    def getStateData(self,field,values,size):
        result=super().getStateData(field,values,size)
        if not self.started and self.last_control=='reset' and self.samples_since_control==12:
            if self.fault=='pre_invalid_q':return 1
            if self.fault=='pre_q0_drift':values.q[0]+=1.1e-5
        return result

    def powerState(self,ec):
        result=super().powerState(ec)
        if self.fault=='pre_power_off' and self.last_control=='reset':return csv_tests.Power.off
        return result


class PrestartEpochTests(unittest.TestCase):
    def run_fake(self,fault=None,prompt_delay_s=0):
        return csv_tests.LifecycleTests().run_fake(fault,robot_type=EpochRobot,prompt_delay_s=prompt_delay_s)

    def replay_session(self):
        previous,current=FIXTURE['q_samples']
        clock=csv_tests.Clock();clock.ns=current['read_start_ns']
        output=[]
        store=types.SimpleNamespace(write=lambda stream,row:output.append((stream,row)))
        sdk=types.SimpleNamespace(PyTypeVectorDouble=csv_tests.Vector)
        session=s.CsvSession(sdk,store,s.load_csv(),s.SOURCE,6)
        # State-source stub only; no robot constructor, connection or control API.
        def update(timeout):clock.ns=current['read_end_ns'];return 76
        def decode(field,values,size):
            values.q=current['q_native_rad'];clock.ns=current['decoded_end_ns'];return 0
        session.robot=types.SimpleNamespace(updateRobotState=update,getStateData=decode)
        session.phase='POST_CALL_STATIONARITY';session.strict_window=True
        session.sequence=current['sequence']
        session.last_good=previous['decoded_end_ns'];session.last_sample=previous
        session.q0=previous['q_native_rad'];session.records.append(previous)
        return session,clock,output

    def test_exact_recorded_old_shared_read_path_fails_prestart_gap(self):
        self.assertEqual(FIXTURE['gap_ns'],67_180_700)
        session,clock,_=self.replay_session()
        with patch.object(n.time,'perf_counter_ns',clock.counter_ns):
            with self.assertRaisesRegex(n.ValidationError,'sampling gap >50 ms'):
                n.PreparedSession.read_sample(session)

    def test_exact_recorded_gap_outside_prestart_epoch_is_not_fatal(self):
        session,clock,_=self.replay_session()
        with patch.object(n.time,'perf_counter_ns',clock.counter_ns):self.assertTrue(session.read_sample())
        self.assertFalse(session.start_attempted)
        self.assertEqual(session.max_motion_gap_ns,0)

    def test_exact_recorded_gap_inside_epoch_discards_old_candidate(self):
        session,clock,output=self.replay_session();session.prestart_epoch_active=True
        with patch.object(n.time,'perf_counter_ns',clock.counter_ns):session.read_sample()
        self.assertEqual([r['sequence'] for r in session.records],[1751])
        restart=next(r for stream,r in output if r.get('code')=='PRESTART_STATIONARITY_WINDOW_RESTART')
        self.assertEqual(restart['gap_ns'],67_180_700)

    def test_candidate_restarts_then_accepts_clean_second_without_retrying_control(self):
        status,final,robot,logs=self.run_fake('pre_gap_once')
        self.assertEqual(status,'COMPLETED_ACCEPTED')
        restart=next(r for r in logs['lifecycle'] if r.get('code')=='PRESTART_STATIONARITY_WINDOW_RESTART')
        accepted=next(r for r in logs['lifecycle'] if r.get('code')=='PRESTART_STATIONARITY_EPOCH_ACCEPTED'
                      and r['epoch']==restart['epoch'])
        self.assertGreaterEqual(accepted['first_sequence'],restart['first_sequence'])
        self.assertGreaterEqual(accepted['sample_count'],100)
        self.assertGreaterEqual(accepted['duration_ns'],1_000_000_000)
        self.assertLessEqual(accepted['maximum_gap_ns'],50_000_000)
        self.assertEqual(robot.calls.count('reset'),1)
        self.assertEqual(robot.calls.count('append'),6)
        self.assertEqual(robot.calls.count('start'),1)

    def test_repeated_bad_candidates_expire_fixed_outer_timeout(self):
        status,final,robot,logs=self.run_fake('pre_gaps_forever')
        self.assertEqual(status,'FAILED_EXECUTION')
        self.assertIn('stationarity timeout',final['failure_reason'])
        self.assertNotIn('start',robot.calls)
        restarts=[r for r in logs['lifecycle'] if r.get('code')=='PRESTART_STATIONARITY_WINDOW_RESTART']
        self.assertGreater(len(restarts),1)
        self.assertLess(restarts[-1]['host_ns']-restarts[0]['host_ns'],11_000_000_000)

    def test_long_prompts_cannot_supply_old_stationarity_to_append_or_start(self):
        status,_,_,logs=self.run_fake(prompt_delay_s=5)
        self.assertEqual(status,'COMPLETED_ACCEPTED')
        for action,api in [('SETUP','setMotionControlMode'),('APPEND','moveAppend'),('START','moveStart')]:
            authorization=next(r for r in logs['authorizations'] if r.get('action')==action)
            accepted=next(r for r in logs['lifecycle'] if r.get('code')=='PRESTART_STATIONARITY_EPOCH_ACCEPTED'
                          and r['reason']=='before_'+api)
            self.assertGreater(accepted['first_timestamp_ns'],authorization['host_ns'])
            self.assertGreaterEqual(accepted['duration_ns'],1_000_000_000)

    def test_slow_construction_hashing_and_metadata_are_outside_new_window(self):
        construct=n.construct_commands;digest=n.digest_json;write=s.CsvStore.json
        def slow_construct(*args,**kwargs):
            EpochRobot.latest.clock.ns+=FIXTURE['gap_ns'];return construct(*args,**kwargs)
        def slow_digest(*args,**kwargs):
            EpochRobot.latest.clock.ns+=80_000_000;return digest(*args,**kwargs)
        def slow_write(store,*args,**kwargs):
            EpochRobot.latest.clock.ns+=100_000_000;return write(store,*args,**kwargs)
        with patch.object(n,'construct_commands',slow_construct),patch.object(n,'digest_json',slow_digest),patch.object(s.CsvStore,'json',slow_write):
            status,_,_,logs=self.run_fake()
        self.assertEqual(status,'COMPLETED_ACCEPTED')
        reasons=[r['reason'] for r in logs['lifecycle'] if r.get('code')=='PRESTART_SAMPLING_SUSPENDED']
        for reason in ('command_construction','command_validation_and_hashing','metadata_write'):
            self.assertIn(reason,reasons)

    def test_every_control_has_fresh_window_and_every_append_has_post_window(self):
        status,_,_,logs=self.run_fake()
        self.assertEqual(status,'COMPLETED_ACCEPTED')
        for api in ('setMotionControlMode','moveReset','moveAppend','moveStart'):
            calls=[r for r in logs['calls'] if r.get('phase')=='return' and r['api']==api]
            accepted=[r for r in logs['lifecycle'] if r.get('code')=='PRESTART_STATIONARITY_EPOCH_ACCEPTED'
                      and r['reason']=='before_'+api]
            self.assertEqual(len(calls),len(accepted))
            for call,evidence in zip(calls,accepted):
                self.assertLessEqual(evidence['last_timestamp_ns'],call['request_ns'])
                self.assertLessEqual(call['request_ns']-evidence['last_timestamp_ns'],50_000_000)
        calls=[r for r in logs['calls'] if r.get('phase')=='return' and r['api']=='moveAppend']
        after=[r for r in logs['lifecycle'] if r.get('code')=='PRESTART_STATIONARITY_EPOCH_ACCEPTED' and r['reason']=='after_moveAppend']
        for call,evidence in zip(calls,after):
            self.assertGreater(call['duration_ns'],50_000_000)
            self.assertGreater(evidence['first_timestamp_ns'],call['return_ns'])
            self.assertLessEqual(max(evidence['window_max_abs_reference_residual_rad']),1e-5)

    def test_invalid_q_reference_drift_and_power_fault_remain_fatal(self):
        for fault in ('pre_invalid_q','pre_q0_drift','pre_power_off'):
            with self.subTest(fault=fault):
                status,_,robot,_=self.run_fake(fault)
                self.assertEqual(status,'FAILED_EXECUTION')
                self.assertNotIn('start',robot.calls)
                self.assertEqual(robot.calls[-1],'disconnect')

    def test_same_gap_after_start_remains_fatal(self):
        session,clock,_=self.replay_session()
        session.start_attempted=True;session.phase='ACTIVE_SAMPLING';session.q0=None
        with patch.object(n.time,'perf_counter_ns',clock.counter_ns):
            with self.assertRaisesRegex(n.ValidationError,'sampling gap >50 ms'):session.read_sample()
        self.assertEqual(session.max_post_boundary_gap_ns,67_180_700)
        with self.assertRaises(n.ValidationError):session.suspend_prestart_sampling('forbidden')

    def test_final_stationary_gate_still_rejects_each_invalid_criterion(self):
        def rows():return [{'sequence':i,'decoded_end_ns':1+i*8_000_000,'valid_q':True,'q_native_rad':[0.]*6} for i in range(126)]
        self.assertIsNotNone(stationary_evidence(rows(),[0.]*6,1))
        short=rows()[:-1];self.assertIsNone(stationary_evidence(short,[0.]*6,1))
        few=rows()[::2];self.assertIsNone(stationary_evidence(few,[0.]*6,1))
        gap=rows()
        for r in gap[60:]:r['decoded_end_ns']+=67_000_000
        self.assertIsNone(stationary_evidence(gap,[0.]*6,1))
        span=rows();span[10]['q_native_rad'][0]=1.00001e-5
        self.assertIsNone(stationary_evidence(span,[0.]*6,1))
        far=rows()
        for r in far:r['q_native_rad'][0]=.00101
        self.assertIsNone(stationary_evidence(far,[0.]*6,1))


if __name__=='__main__':unittest.main()
