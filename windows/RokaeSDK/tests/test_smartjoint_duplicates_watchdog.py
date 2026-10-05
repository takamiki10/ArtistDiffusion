"""Frozen historical source/watchdog evidence and the prospectively approved policy."""
import copy
import json
from pathlib import Path
import struct
import unittest

import neutral_v050_nrt_runner as n
import smartjoint_csv_nrt_runner as s
from nrt_v050.audit_smartjoint_duplicates_watchdog import pair_evidence


class DuplicateWatchdogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = s.load_csv()
        # Preserve the previous investigation as historical evidence. It predates
        # the user's explicit adoption of the new warning policy/watchdog.
        cls.report = json.loads((s.ROOT/'nrt_v050/evidence/smartjoint_duplicates_watchdog/audit.json').read_text(encoding='utf-8'))

    def completion(self):
        return s.BatchCompletion({f'absj#{i}':span for i,span in enumerate(s.BATCHES)},'test',0)

    def event(self, source_index, remark='', ec=0):
        batch = next(i for i,(a,b) in enumerate(s.BATCHES) if a <= source_index < b)
        a,b = s.BATCHES[batch]
        return {'host_ns':1,'payload':{'cmdID':f'absj#{batch}', 'wayPointIndex':source_index-a,
                'customInfo':f'test:k{source_index:03d}', 'remark':remark,
                'reachTarget':True,'error':{'ec':ec,'message':'success' if ec == 0 else 'failure'}}}

    def test_six_exact_duplicate_pairs_and_states(self):
        pairs = self.report['duplicate_pairs']
        self.assertEqual([(p['preceding_index'],p['current_index']) for p in pairs],
                         [(187,188),(188,189),(217,218),(328,329),(367,368),(388,389)])
        self.assertEqual([p['current_TouchType'] for p in pairs],['Air','Air','Pen','Air','Air','Pen'])
        for p in pairs:
            self.assertEqual(p['binary64_equal_per_joint'],[True]*6)
            self.assertEqual(p['maximum_abs_joint_delta_rad'],0.0)
            self.assertEqual(p['preceding_TouchType'],p['current_TouchType'])
            self.assertEqual(p['preceding_binary64_hex'],p['current_binary64_hex'])

    def test_index_one_is_not_duplicate_in_any_joint(self):
        p = pair_evidence(self.data,1)
        self.assertEqual(p['binary64_equal_per_joint'],[False]*6)
        self.assertFalse(p['all_six_bitwise_identical'])
        self.assertEqual(p['maximum_abs_joint_delta_rad'],2.7093647105014274e-6)
        self.assertFalse(self.report['observed_warning_exception_justified'])

    def test_binary64_proof_distinguishes_signed_zero(self):
        data = copy.copy(self.data)
        rows = list(data.q)
        rows[1] = rows[0] = (0.0,)*6
        rows[1] = (-0.0,)+(0.0,)*5
        object.__setattr__(data,'q',tuple(rows))
        p = pair_evidence(data,1)
        self.assertEqual(p['maximum_abs_joint_delta_rad'],0.0)
        self.assertEqual(p['binary64_equal_per_joint'],[False,True,True,True,True,True])
        self.assertFalse(p['all_six_bitwise_identical'])

    def test_duplicate_warning_is_now_diagnostic_only(self):
        for index in (188,189,218,329,368,389):
            c = self.completion();c.observe(self.event(index,'adjacent points'))
            self.assertEqual(c.policy_failures,[])
            self.assertEqual(len(c.diagnostic_warnings),1)
            self.assertEqual(c.last_index,-1)
            self.assertIsNone(c.terminal_ns)

    def test_nonduplicate_adjacent_points_is_now_diagnostic_only(self):
        c = self.completion();c.observe(self.event(1,'adjacent points'))
        c.observe(self.event(507))
        self.assertEqual(c.completed_status(),'COMPLETED_ACCEPTED')

    def test_other_remarks_on_duplicate_are_fatal(self):
        for text in ('other warning','Adjacent points','adjacent points ','adjacent points\n'):
            c=self.completion()
            with self.assertRaises(n.ValidationError):c.observe(self.event(188,text))
            self.assertTrue(c.hard_execution_fault)

    def test_nonzero_error_on_duplicate_remains_fatal(self):
        c=self.completion()
        with self.assertRaises(n.ValidationError):c.observe(self.event(188,'adjacent points',7))
        self.assertTrue(c.hard_execution_fault)

    def test_invalid_id_index_custom_info_remain_fatal(self):
        for key,value in (('cmdID','absj#99'),('wayPointIndex',100),('wayPointIndex',-1),
                          ('customInfo','test:k187')):
            c=self.completion();e=self.event(188,'adjacent points');e['payload'][key]=value
            with self.assertRaises(n.ValidationError):c.observe(e)
            self.assertTrue(c.hard_execution_fault)

    def test_diagnostic_never_satisfies_terminal(self):
        c=self.completion();c.observe(self.event(507,'adjacent points'))
        self.assertIsNone(c.terminal_ns)
        self.assertEqual(c.last_index,-1)
        with self.assertRaises(n.ValidationError):c.completed_status()

    def test_clean_and_approved_warning_then_completion_are_accepted(self):
        clean=self.completion();clean.observe(self.event(0));clean.observe(self.event(507))
        self.assertEqual(clean.completed_status(),'COMPLETED_ACCEPTED')
        warned=self.completion();warned.observe(self.event(188,'adjacent points'));warned.observe(self.event(507))
        self.assertEqual(warned.completed_status(),'COMPLETED_ACCEPTED')

    def test_timeout_is_approach_progress_with_independent_warning_rejection(self):
        r=next(x for x in self.report['sessions'] if x['failure_reason']=='MEASURED_COMPLETION_NOT_CONFIRMED')
        self.assertAlmostEqual(r['q_observation_after_start_return_s'],120.0133021)
        self.assertAlmostEqual(r['q_motion_duration_after_first_observed_movement_s'],119.8419243)
        self.assertTrue(r['last_state_was_moving'])
        self.assertIsNone(r['last_non_diagnostic_source_index'])
        self.assertEqual(r['last_callback_index'],1)
        self.assertEqual(r['sdk_call_errors'],[])
        self.assertFalse(r['hard_event_fault'])
        self.assertFalse(r['sole_obstacle_to_acceptance_was_timeout'])
        self.assertEqual(r['dominant_backward_steps_over_1e_5_rad'],0)
        errors=[p['max_abs_residual_to_first_target_rad'] for p in r['approach_snapshots']]
        self.assertTrue(all(a>b for a,b in zip(errors,errors[1:])))

    def test_historical_watchdog_report_preserved_and_new_adoption_is_explicit(self):
        self.assertEqual(n.SETTINGS['completion_timeout_s'],120.0)
        self.assertEqual(s.COMPLETION_TIMEOUT_S,600.0)
        self.assertFalse(self.report['watchdog']['applied'])
        self.assertEqual(self.report['watchdog']['recommended_prospective_s'],600)
        self.assertFalse(self.report['production_exception_implemented'])
        self.assertEqual([b-a for a,b in s.BATCHES],[100,100,100,100,100,8])
        self.assertFalse(self.report['ready_for_start'])


if __name__=='__main__':unittest.main()
