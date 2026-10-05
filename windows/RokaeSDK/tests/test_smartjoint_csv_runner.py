"""Offline CSV and fake SDK lifecycle checks. No vendor robot is constructed."""
from contextlib import redirect_stdout
import csv
import io
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import smartjoint_csv_nrt_runner as s
import neutral_v050_nrt_runner as n
from test_neutral_v050_lifecycle import FakeRobot, Clock, Vector, String, Power, Mode, Operation
from test_neutral_v050_nrt import JointPosition, MoveAbsJCommand, PyErrorCode


class CsvTests(unittest.TestCase):
    def setUp(self):
        self.data = s.load_csv()

    def altered(self, change):
        rows = list(csv.reader(io.StringIO(self.data.raw.decode('utf-8-sig'))))
        change(rows)
        out = io.StringIO(newline='')
        csv.writer(out).writerows(rows)
        return out.getvalue().encode()

    def test_independent_parse_and_complete_structure(self):
        report = s.audit(self.data)
        self.assertEqual(report['row_count'], 508)
        self.assertEqual(report['touch_counts'], {'Pen': 300, 'Air': 208})
        self.assertTrue(report['binary64_bitwise_equal'])
        self.assertEqual(report['compared_joint_values'], 3048)
        self.assertEqual(report['position_violations'], [])
        self.assertEqual([x['count'] for x in report['segments']], [30,100,79,100,79,100,20])
        self.assertEqual(self.data.q[0], (-2.227512509127758,-0.18441560441027577,2.2651860822247576,
                                       -3.1415984105959325,-0.6919914095179537,4.055677012796968))
        self.assertEqual(self.data.q[-1], (-2.385008290567847,-0.2107033771154584,2.2279644800565017,
                                        -3.1415970725388194,-0.7029249867113442,3.8981800656445262))

    def test_malformed_and_nonfinite_rejected(self):
        changes = [lambda r: r.pop(), lambda r: r[1].pop(), lambda r: r[0].__setitem__(2,'q1'),
                   lambda r: r[1].__setitem__(1,'Unknown'), lambda r: r[2].__setitem__(0,r[1][0]),
                   lambda r: r[2].__setitem__(0,'-1'), lambda r: r[1].__setitem__(0,'nan')]
        changes += [lambda r, value=v: r[1].__setitem__(2,value) for v in ('nan','inf','-inf','')]
        for change in changes:
            with self.assertRaises((n.ValidationError, ValueError)):
                s.parse_csv(self.altered(change))

    def test_position_violation_is_rejected_without_clipping(self):
        data = s.parse_csv(self.altered(lambda r:r[1].__setitem__(2,'7.0')))
        with self.assertRaisesRegex(n.ValidationError,'position bounds'):
            s.audit(data)
        self.assertEqual(data.q[0][0],7.0)

    def test_source_pin_catches_any_edit(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'changed.csv'
            path.write_bytes(self.data.raw+b'\n')
            with self.assertRaisesRegex(n.ValidationError,'SHA256'):
                s.load_csv(path)

    def test_commands_and_batch_boundaries(self):
        sdk=types.SimpleNamespace(JointPosition=JointPosition,MoveAbsJCommand=MoveAbsJCommand)
        commands=n.construct_commands(sdk,self.data.q,'test',508)
        self.assertEqual(n.matrix_bytes([list(c.target.joints) for c in commands],508),n.matrix_bytes(self.data.q,508))
        self.assertEqual([b-a for a,b in s.BATCHES],[100,100,100,100,100,8])
        self.assertEqual([c.zone for c in commands],[1]*507+[0])
        commands[99].zone=0
        with self.assertRaises(n.ValidationError):
            n.validate_commands(sdk,commands,self.data.q,'test',508)
        # Existing callers remain frozen at 100, including their rejection behavior.
        with self.assertRaises(n.ValidationError):
            n.matrix_bytes(self.data.q)

    def test_offline_cli_never_imports_sdk_by_default(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(n,'load_exact_sdk_data_only',side_effect=AssertionError('SDK forbidden')), redirect_stdout(io.StringIO()):
            self.assertEqual(s.main(['--offline','--report',str(Path(temp)/'report.json')]),0)
            with self.assertRaises(n.ValidationError):
                s.main(['--run-trial','--trial','1'])


class EventTests(unittest.TestCase):
    def completion(self):
        return s.BatchCompletion({f'ack{i}':span for i,span in enumerate(s.BATCHES)},'test',0)

    def event(self, batch, local, **changes):
        p={'cmdID':f'ack{batch}','wayPointIndex':local,'reachTarget':True,
           'error':{'ec':0,'message':'success'},'remark':'','customInfo':f'test:k{batch*100+local:03d}'}
        p.update(changes)
        return {'host_ns':1000,'payload':p}

    def test_all_508_indices_map_and_only_final_is_terminal(self):
        c=self.completion()
        for batch,(a,b) in enumerate(s.BATCHES):
            for local in range(b-a):
                c.observe(self.event(batch,local))
                self.assertEqual(c.last_index,a+local)
                if a+local < 507: self.assertIsNone(c.terminal_ns)
        self.assertEqual(len(c.seen),508)
        self.assertEqual(c.terminal_ns,1000)

    def test_event_rejections_and_remarks(self):
        for ev in (self.event(5,8),self.event(0,100),self.event(0,0,cmdID='unknown'),
                   self.event(1,0,customInfo='test:k000'),self.event(1,0,error={'ec':1,'message':'fault'})):
            c=self.completion()
            with self.assertRaises(n.ValidationError): c.observe(ev)
            self.assertTrue(c.hard_execution_fault)
        c=self.completion();c.observe(self.event(1,0))
        with self.assertRaises(n.ValidationError):c.observe(self.event(0,99))
        c=self.completion();c.observe(self.event(1,0))
        with self.assertRaises(n.ValidationError):c.observe(self.event(1,0))
        c=self.completion();c.observe(self.event(5,7,remark='adjacent points'))
        self.assertIsNone(c.terminal_ns)  # A warning cannot supply terminal evidence.
        c.observe(self.event(5,7))
        self.assertEqual(c.completed_status(),'COMPLETED_ACCEPTED')


class CsvRobot(FakeRobot):
    def __init__(self,clock,fault,data):
        super().__init__(clock,fault)
        self.data=data
        self.append_sizes=[]
        self.started_at_ns=None

    def moveStart(self,ec):
        super().moveStart(ec)
        self.started_at_ns=self.clock.ns

    def moveAppend(self,commands,cmd_id,ec):
        self.commands.extend(commands)
        self.append_sizes.append(len(commands))
        cmd_id.value='ack'+str(len(self.append_sizes)-1)
        if self.fault=='duplicate_id':cmd_id.value='ack0'
        self.control('append',ec)
        if self.fault=='append3_error' and len(self.append_sizes)==3:ec['ec']=9

    def setEventWatcher(self,kind,callback,ec):
        def mapped(payload):
            if payload['wayPointIndex']==99:
                payload={**payload,'cmdID':'ack5','wayPointIndex':7}
            else:
                payload={**payload,'cmdID':'ack0'}
            callback(payload)
        super().setEventWatcher(kind,mapped,ec)

    def getStateData(self,field,values,size):
        rc=super().getStateData(field,values,size)
        values.q=[a+b for a,b in zip(self.data.q[0],values.q)]
        if self.started and self.samples_since_control>=2 and self.fault not in ('no_motion',):
            values.q=list(self.data.q[-1])
            if self.fault=='bad_endpoint':values.q[0]+=.00101
            if self.fault=='slow_completion' and self.clock.ns-self.started_at_ns < 130_000_000_000:
                values.q=[(a+b)/2 for a,b in zip(self.data.q[0],self.data.q[-1])]
        return rc


class LifecycleTests(unittest.TestCase):
    def run_fake(self,fault=None,denied=None,robot_type=CsvRobot,prompt_delay_s=0):
        data=s.load_csv();clock=Clock();robot=robot_type(clock,fault,data);creations=[]
        def factory():creations.append(1);return robot
        sdk=types.SimpleNamespace(xMateRobot=factory,JointPosition=JointPosition,MoveAbsJCommand=MoveAbsJCommand,
            PyString=String,PyTypeVectorDouble=Vector,PyErrorCode=PyErrorCode,PowerState=Power,OperateMode=Mode,
            OperationState=Operation,Event=types.SimpleNamespace(moveExecution=0),MotionControlMode=types.SimpleNamespace(NrtCommandMode=1))
        with tempfile.TemporaryDirectory() as temp, patch.object(n.time,'perf_counter_ns',clock.counter_ns), patch.object(n.time,'perf_counter',clock.counter), patch('builtins.input',return_value='CANCEL' if denied=='CONNECT' else 'CONNECT SMARTJOINT TRIAL 1'), redirect_stdout(io.StringIO()):
            store=s.CsvStore(Path(temp)/'session',data)
            session=s.CsvSession(sdk,store,data,s.SOURCE,1)
            def prompt(text):
                clock.ns += int(prompt_delay_s*1e9)
                session.pump()
                return 'CANCEL' if denied and text.startswith('Type exactly '+denied) else text[len('Type exactly '):-2]
            session.prompt=prompt
            status=session.execute();store.close()
            final=json.loads((store.folder/'final_status.json').read_text())
            logs={name:[json.loads(line) for line in (store.folder/(name+'.jsonl')).read_text().splitlines()]
                  for name in ('q_m','motion_events','source_rows','calls','lifecycle','authorizations')}
            logs['session_metadata']=json.loads((store.folder/'session.json').read_text())
            self.assertEqual((store.folder/'source.csv').read_bytes(),data.raw)
            self.assertEqual((store.folder/'waypoints.f64le').read_bytes(),n.matrix_bytes(data.q,508))
        self.assertEqual(len(creations),0 if denied=='CONNECT' else 1)
        return status,final,robot,logs

    def test_full_lifecycle_and_logs(self):
        status,final,robot,logs=self.run_fake()
        self.assertEqual(status,'COMPLETED_ACCEPTED')
        self.assertEqual(robot.append_sizes,[100,100,100,100,100,8])
        self.assertEqual(robot.calls.count('start'),1)
        self.assertLess(max(i for i,x in enumerate(robot.calls) if x=='append'),robot.calls.index('start'))
        self.assertEqual([c.zone for c in robot.commands],[1]*507+[0])
        self.assertTrue(all(c.jointSpeed==.05 and c.speed==50 for c in robot.commands))
        self.assertEqual(len(logs['source_rows']),508)
        self.assertEqual([r['TouchType'] for r in logs['source_rows']],[r[1] for r in s.load_csv().cells])
        self.assertEqual(logs['motion_events'][-1]['source_metadata']['source_index'],507)
        self.assertTrue(any(r['last_validated_event_TouchType']=='Air' for r in logs['q_m']))
        self.assertEqual(final['appended_rows'],508)
        self.assertFalse(final['valid_for_timing_analysis'])

    def test_each_interactive_gate(self):
        for denied,forbidden in [('CONNECT','connect'),('SETUP','setup'),('APPEND','append'),('START','start')]:
            status,_,robot,_=self.run_fake(denied=denied)
            self.assertEqual(status,'FAILED_EXECUTION')
            self.assertNotIn(forbidden,robot.calls)

    def test_partial_append_or_duplicate_id_never_starts(self):
        for fault,appends in [('append3_error',3),('duplicate_id',2)]:
            status,final,robot,_=self.run_fake(fault)
            self.assertEqual(status,'FAILED_EXECUTION')
            self.assertEqual(len(robot.append_sizes),appends)
            self.assertNotIn('start',robot.calls)
            self.assertLess(final['appended_rows'],508)

    def test_inherited_fault_and_completion_gates(self):
        for fault in ('motion_gap','start_gap_then_gap','remark_then_error','start_displaced',
                      'start_power_off','start_mode_change','bad_endpoint','no_motion'):
            self.assertEqual(self.run_fake(fault)[0],'FAILED_EXECUTION',fault)
        self.assertEqual(self.run_fake('no_terminal')[0],'COMPLETED_ACCEPTED')
        self.assertEqual(self.run_fake('remark_then_terminal')[0],'COMPLETED_ACCEPTED')
        self.assertEqual(self.run_fake('start_gap')[0],'COMPLETED_ACCEPTED')


if __name__=='__main__':unittest.main()
