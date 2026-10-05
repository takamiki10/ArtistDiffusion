"""Prospective SmartJoint warning/watchdog policy. Fake SDK/data objects only."""
import csv
import hashlib
import io
import json
from pathlib import Path
import struct
import unittest

import neutral_v050_nrt_runner as n
import smartjoint_csv_nrt_runner as s
from test_neutral_v050_nrt import SDK
import test_smartjoint_csv_runner as csv_tests


class ProspectivePolicyTests(unittest.TestCase):
    def completion(self):
        return s.BatchCompletion({f'absj#{i}':span for i,span in enumerate(s.BATCHES)},'test',0)

    def event(self,index,remark='',error=None):
        batch=index//100
        return {'host_ns':1,'payload':{'cmdID':f'absj#{batch}','wayPointIndex':index%100,
                'customInfo':f'test:k{index:03d}','remark':remark,'reachTarget':True,
                'error':{'ec':0,'message':'success'} if error is None else error}}

    def test_exact_observed_index_one_payload_is_diagnostic_only(self):
        fixture=Path(__file__).parent/'fixtures/smartjoint_event_order'
        events=[json.loads(line) for line in (fixture/'motion_events.jsonl').read_text().splitlines()]
        meta=json.loads((fixture/'provenance.json').read_text())
        batches=json.loads((fixture/'append_batches.json').read_text())
        c=s.BatchCompletion(batches,meta['tag'],meta['start_ns'])
        c.observe(events[0])
        self.assertEqual(c.policy_failures,[])
        self.assertFalse(c.hard_execution_fault)
        self.assertEqual(c.diagnostic_warnings[0]['raw_payload'],events[0]['payload'])
        self.assertEqual(c.diagnostic_warnings[0]['source_index'],1)
        self.assertEqual(c.diagnostic_warnings[0]['cmdID'],'absj#0')
        self.assertEqual(c.last_index,-1)
        self.assertIsNone(c.terminal_ns)
        c.observe(events[1])
        self.assertEqual(c.last_index,0)

    def test_near_nonduplicate_source_pair_needs_no_equality_exception(self):
        data=s.load_csv()
        self.assertNotEqual(n.joint_bytes(data.q[0]),n.joint_bytes(data.q[1]))
        self.assertEqual(max(abs(a-b) for a,b in zip(data.q[0],data.q[1])),2.7093647105014274e-6)
        c=self.completion();c.observe(self.event(1,'adjacent points'))
        self.assertEqual(c.policy_failures,[])
        self.assertEqual(len(c.diagnostic_warnings),1)

    def test_all_exact_duplicates_are_diagnostic_only(self):
        data=s.load_csv()
        for i in (188,189,218,329,368,389):
            self.assertEqual(n.joint_bytes(data.q[i-1]),n.joint_bytes(data.q[i]))
            c=self.completion();c.observe(self.event(i-1));c.observe(self.event(i,'adjacent points'))
            self.assertEqual(c.last_index,i-1)
            self.assertEqual(c.policy_failures,[])
            c.observe(self.event(i));c.observe(self.event(507))
            self.assertEqual(c.completed_status(),'COMPLETED_ACCEPTED')

    def test_nonzero_or_malformed_sdk_error_remains_fatal(self):
        for error in ({'ec':7,'message':'failure'},{'ec':False,'message':'success'},
                      {'ec':0.0,'message':'success'},{'ec':0,'message':'other'},
                      {'ec':0,'message':'success','other_error':7}):
            c=self.completion()
            with self.assertRaises(n.ValidationError):c.observe(self.event(1,'adjacent points',error))
            self.assertTrue(c.hard_execution_fault)
            self.assertEqual(c.diagnostic_warnings,[])

    def test_any_other_nonempty_remark_is_fatal(self):
        for remark in ('other warning','Adjacent points','adjacent points ','adjacent points\n',' adjacent points'):
            c=self.completion()
            with self.assertRaises(n.ValidationError):c.observe(self.event(1,remark))
            self.assertTrue(c.hard_execution_fault)
            self.assertEqual(c.diagnostic_warnings,[])

    def test_invalid_local_index_is_fatal(self):
        for index in (-1,100,True,'1',1.0):
            c=self.completion();e=self.event(1,'adjacent points');e['payload']['wayPointIndex']=index
            with self.assertRaises(n.ValidationError):c.observe(e)
            self.assertTrue(c.hard_execution_fault)

    def test_invalid_command_or_custom_info_is_fatal(self):
        for key,value in (('cmdID','unknown'),('cmdID','absj#1'),('customInfo','test:k000'),('customInfo',None)):
            c=self.completion();e=self.event(1,'adjacent points');e['payload'][key]=value
            with self.assertRaises(n.ValidationError):c.observe(e)
            self.assertTrue(c.hard_execution_fault)

    def test_malformed_payload_is_fatal(self):
        for key in ('cmdID','wayPointIndex','customInfo','remark','reachTarget','error'):
            c=self.completion();e=self.event(1,'adjacent points');del e['payload'][key]
            with self.assertRaises((n.ValidationError,KeyError)):c.observe(e)
            self.assertTrue(c.hard_execution_fault)
        c=self.completion();e=self.event(1,'adjacent points');e['payload']['reachTarget']=1
        with self.assertRaises(n.ValidationError):c.observe(e)

    def test_warning_cannot_establish_terminal_completion(self):
        c=self.completion();c.observe(self.event(507,'adjacent points'))
        self.assertIsNone(c.terminal_ns)
        self.assertEqual(c.seen,set())
        with self.assertRaises(n.ValidationError):c.completed_status()

    def test_later_valid_completion_can_be_accepted(self):
        c=self.completion();c.observe(self.event(1,'adjacent points'));c.observe(self.event(0))
        self.assertEqual(c.last_index,0)
        c.observe(self.event(507))
        self.assertEqual(c.completed_status(),'COMPLETED_ACCEPTED')

    def test_completion_regression_remains_fatal(self):
        c=self.completion();c.observe(self.event(5));c.observe(self.event(1,'adjacent points'))
        with self.assertRaises(n.ValidationError):c.observe(self.event(4))
        self.assertTrue(c.hard_execution_fault)

    def test_prior_hard_error_cannot_be_erased_by_diagnostic(self):
        c=self.completion()
        with self.assertRaises(n.ValidationError):c.observe(self.event(0,error={'ec':7,'message':'failure'}))
        with self.assertRaises(n.ValidationError):c.observe(self.event(1,'adjacent points'))
        self.assertTrue(c.hard_execution_fault)
        self.assertEqual(c.diagnostic_warnings,[])

    def test_fixed_watchdog_and_unchanged_shared_settings(self):
        self.assertEqual(s.COMPLETION_TIMEOUT_S,600.0)
        # Do not silently change the independent neutral/path_0003 experiments.
        self.assertEqual(n.SETTINGS['completion_timeout_s'],120.0)
        report=s.audit(s.load_csv())
        self.assertEqual(report['completion_timeout_s'],600.0)
        self.assertEqual(report['warning_policy']['allowed_remark'],'adjacent points')

    def test_all_508_targets_batches_and_motion_parameters_are_unchanged(self):
        raw=s.SOURCE.read_bytes();data=s.load_csv()
        self.assertEqual(hashlib.sha256(raw).hexdigest(),s.SOURCE_SHA256)
        independent=[[float(r[f'joint{j}']) for j in range(1,7)]
                     for r in csv.DictReader(io.StringIO(raw.decode('utf-8-sig')))]
        commands=n.construct_commands(SDK,data.q,'test',508)
        self.assertEqual(n.matrix_bytes(independent,508),n.matrix_bytes(data.q,508))
        self.assertEqual(n.matrix_bytes([list(c.target.joints) for c in commands],508),n.matrix_bytes(independent,508))
        self.assertEqual([b-a for a,b in s.BATCHES],[100,100,100,100,100,8])
        self.assertEqual([c.zone for c in commands],[1]*507+[0])
        self.assertTrue(all(c.jointSpeed==.05 and c.speed==50 for c in commands))

    def test_simulated_execution_past_120_seconds_completes_with_600_metadata(self):
        status,final,robot,logs=csv_tests.LifecycleTests().run_fake('slow_completion')
        self.assertEqual(status,'COMPLETED_ACCEPTED')
        self.assertEqual(final['completion_watchdog_s'],600.0)
        self.assertEqual(logs['session_metadata']['settings']['completion_timeout_s'],600.0)
        self.assertEqual(logs['session_metadata']['completion_watchdog_s'],600.0)
        elapsed=(final['final_stationary']['last_sequence']-final['motion_observation']['first_threshold_sample']['sequence'])*.008
        self.assertGreater(elapsed,120)
        self.assertLess(elapsed,600)
        self.assertEqual(robot.calls.count('start'),1)

    def test_allowed_warning_with_later_controller_state_fault_still_fails(self):
        status,final,_,_=csv_tests.LifecycleTests().run_fake('remark_then_state_fault')
        self.assertEqual(status,'FAILED_EXECUTION')
        self.assertIn('operation state',final['failure_reason'])


if __name__=='__main__':unittest.main()
