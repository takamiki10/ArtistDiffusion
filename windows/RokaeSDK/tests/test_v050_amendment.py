"""Synthetic only: no vendor SDK import or hardware mode invocation."""
import ast
import math
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import passive_v050_prepared as passive
from physical_reference_30s import Reference, derivative, evaluate, bits

class Clock:
    def __init__(self): self.n=0
    def __call__(self): self.n+=1_000_000_000; return self.n

class Vector:
    def content(self): return [0.1,-0.0,0.2,0.3,-0.4,0.5]

class SDK:
    PyTypeVectorDouble=Vector
    def __init__(self,fail=None): self.calls=[]; self.fail=fail
    def xMateRobot(self): self.calls.append('construct'); return self
    def connectToRobot(self,*args): self.calls.append('connect')
    def startReceiveRobotState(self,*args):
        self.calls.append('subscribe')
        if self.fail=='subscribe': raise RuntimeError('synthetic start error')
    def updateRobotState(self,*args):
        self.calls.append('update')
        return 0 if self.fail=='timeout' else 48
    def getStateData(self,*args): self.calls.append('read'); return -1 if self.fail=='field' else 0
    def stopReceiveRobotState(self): self.calls.append('unsubscribe')
    def disconnectFromRobot(self,ec): self.calls.append('disconnect'); ec['value']=0

class AmendmentTests(unittest.TestCase):
    def run_capture(self,sdk,records,events):
        return passive.capture(sdk,'SYNTHETIC','SYNTHETIC','smoke',
            dict.fromkeys(passive.REQUIRED,True),records.append,events.append,Clock())
    def test_preconditions_block_before_construct(self):
        sdk=SDK()
        with self.assertRaises(RuntimeError):
            passive.capture(sdk,'','','smoke',{},lambda r:None,lambda e:None)
        self.assertEqual(sdk.calls,[])
    def test_lifecycle_and_native_q(self):
        sdk=SDK(); records=[];events=[]
        self.assertGreater(self.run_capture(sdk,records,events),0)
        self.assertEqual(sdk.calls[:3],['construct','connect','subscribe'])
        self.assertEqual(sdk.calls[-2:],['unsubscribe','disconnect'])
        self.assertEqual(bits(records[0]['q_native_rad'][1]),bits(-0.0))
        self.assertIsNone(records[0]['controller_timestamp'])
    def test_faults_stop_and_disconnect(self):
        for fault in ['subscribe','timeout','field']:
            with self.subTest(fault=fault):
                sdk=SDK(fault);records=[];events=[]
                with self.assertRaises(RuntimeError): self.run_capture(sdk,records,events)
                self.assertEqual(sdk.calls[-2:],['unsubscribe','disconnect'])
                self.assertFalse(events[-1]['valid'])
                if fault=='timeout': self.assertEqual(records,[])
    def test_sink_failure_stops(self):
        sdk=SDK();events=[]
        def fail(r): raise OSError('synthetic disk failure')
        with self.assertRaises(OSError):
            passive.capture(sdk,'','','smoke',dict.fromkeys(passive.REQUIRED,True),fail,events.append,Clock())
        self.assertEqual(sdk.calls[-2:],['unsubscribe','disconnect'])
    def test_only_permitted_robot_methods(self):
        tree=ast.parse(Path(passive.__file__).read_text())
        methods={n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and isinstance(n.func.value,ast.Name) and n.func.value.id=='robot'}
        self.assertEqual(methods,{'connectToRobot','startReceiveRobotState','updateRobotState','getStateData','stopReceiveRobotState','disconnectFromRobot'})
    def test_scaled_knots_derivatives_and_holds(self):
        data=[[i*10/99,math.sin(i/15),-0.0,0,0,0,0] for i in range(100)]
        a,b=Reference(data,1),Reference(data,3)
        for i,t in enumerate(b.times):
            self.assertEqual(t,3*data[i][0])
            for x,y in zip(b.at(t),data[i][1:]): self.assertEqual(bits(x),bits(y))
        self.assertEqual(b.at(31),data[-1][1:])
        for j in range(6):
            for order in (1,2):
                for index,u in ((0,0),(98,1)):
                    p=b.c[j][index]
                    for _ in range(order): p=derivative(p)
                    self.assertAlmostEqual(evaluate(p,u)/b.h[index]**order,0,places=10)
        for t in [0.7,5.0,15.0,29.5]:
            for x,y in zip(b.at(t),a.at(t/3)):self.assertAlmostEqual(x,y,places=12)

if __name__=='__main__': unittest.main()
