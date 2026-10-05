"""Offline fake SDK only; no vendor SDK construction/import."""
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch
import path_0003_nrt_runner as r
import neutral_v050_nrt_runner as n
from test_neutral_v050_lifecycle import FakeRobot, Clock, Vector, String, Power, Mode, Operation
from test_neutral_v050_nrt import JointPosition, MoveAbsJCommand, PyErrorCode


class TrialTests(unittest.TestCase):
    def run_fake(self, fault=None, denied=None):
        clock=Clock(); robot=FakeRobot(clock,fault); creations=[]
        original=robot.getStateData
        def get(field,values,size):
            rc=original(field,values,size)
            if robot.started and robot.samples_since_control>=2 and fault!='no_motion':
                values.q=robot.commands[-1].target.joints.copy()
                if fault=='bad_endpoint': values.q[0]+=.00101
                elif fault=='offset': values.q[0]+=.000689
            return rc
        robot.getStateData=get
        def factory(): creations.append(1); return robot
        sdk=types.SimpleNamespace(xMateRobot=factory,JointPosition=JointPosition,MoveAbsJCommand=MoveAbsJCommand,
            PyString=String,PyTypeVectorDouble=Vector,PyErrorCode=PyErrorCode,PowerState=Power,OperateMode=Mode,
            OperationState=Operation,Event=types.SimpleNamespace(moveExecution=0),MotionControlMode=types.SimpleNamespace(NrtCommandMode=1))
        with tempfile.TemporaryDirectory() as temp, patch.object(n.time,'perf_counter_ns',clock.counter_ns), patch.object(n.time,'perf_counter',clock.counter), patch('builtins.input',return_value='CONNECT PATH0003 TRIAL 1 R'), redirect_stdout(io.StringIO()):
            store=n.SessionStore(Path(temp)/'session'); session=r.TrialSession(sdk,store,1)
            def prompt(text):
                session.pump()
                return 'CANCEL' if denied and text.startswith('Type exactly '+denied) else text[len('Type exactly '):-2]
            session.prompt=prompt
            status=session.execute(); store.close()
            final=json.loads((store.folder/'final_status.json').read_text())
            submitted=(store.folder/'waypoints.f64le').read_bytes()
        self.assertEqual(len(creations),1); self.assertEqual(robot.calls.count('connect'),1)
        self.assertEqual(submitted,n.matrix_bytes(r.load_trial(1)[1]))
        return status,final,robot

    def test_absent_terminal_is_accepted(self):
        status,final,_=self.run_fake('no_terminal')
        self.assertEqual(status,'COMPLETED_ACCEPTED')
        self.assertTrue(final['valid_for_geometric_analysis'])
        self.assertFalse(final['valid_for_timing_analysis'])
        self.assertGreaterEqual(final['final_stationary']['duration_ns'],1_000_000_000)

    def test_endpoint_offset_preserved_and_bound_enforced(self):
        status,final,_=self.run_fake('offset')
        self.assertEqual(status,'COMPLETED_ACCEPTED')
        self.assertAlmostEqual(final['final_stationary']['endpoint_residual_rad'][0],.000689)
        self.assertEqual(self.run_fake('bad_endpoint')[0],'FAILED_EXECUTION')

    def test_events_and_later_gaps_remain_strict(self):
        for fault in ('motion_gap','start_gap_then_gap','remark_then_error','start_displaced'):
            self.assertEqual(self.run_fake(fault)[0],'FAILED_EXECUTION',fault)
        self.assertEqual(self.run_fake('remark_then_terminal')[0],'COMPLETED_REJECTED')

    def test_stationary_start_boundary(self):
        status,final,_=self.run_fake('start_gap')
        self.assertEqual(status,'COMPLETED_ACCEPTED')
        self.assertEqual(final['start_boundary']['reason'],'START_BOUNDARY_STATIONARY_UNOBSERVED_INTERVAL')

    def test_gates_and_control_policy(self):
        for action,api in [('SETUP','setup'),('APPEND','append'),('START','start')]:
            status,_,robot=self.run_fake(denied=action)
            self.assertEqual(status,'FAILED_EXECUTION'); self.assertNotIn(api,robot.calls)
        _,final,robot=self.run_fake()
        self.assertEqual([x for x in robot.calls if x in ('setup','reset','append','start')],['setup','reset','append','start'])
        self.assertTrue(all(c.jointSpeed==.05 and c.speed==50 for c in robot.commands))
        self.assertEqual([c.zone for c in robot.commands],[1]*99+[0])

    def test_all_ten_frozen_trials(self):
        for i,condition in enumerate(r.ORDER,1):
            trial,rows,blob=r.load_trial(i)
            self.assertEqual(trial['condition'],condition)
            self.assertEqual(trial['pair'],(i+1)//2)
            self.assertEqual(n.matrix_bytes(rows),n.matrix_bytes(json.loads(blob)['q_rad']))

    def test_no_motion_cannot_complete(self):
        status,final,_=self.run_fake('no_motion')
        self.assertEqual(status,'FAILED_EXECUTION')
        self.assertEqual(final['failure_reason'],'NO_OBSERVED_PHYSICAL_MOTION')
        rows=[{'sequence':i,'decoded_end_ns':i*8_000_000,'q_native_rad':[0.0]*6,'valid_q':True} for i in range(127)]
        self.assertIsNotNone(r.stationary_evidence(rows,[0.0]*6,0))
        self.assertIsNone(r.stationary_evidence(rows[:99],[0.0]*6,0))
        self.assertIsNone(r.stationary_evidence(rows,[.00101]*6,0))
        rows[-1]['decoded_end_ns']+=60_000_000
        self.assertIsNone(r.stationary_evidence(rows,[0.0]*6,0))


if __name__=='__main__': unittest.main()
