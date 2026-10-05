import copy
import math
from pathlib import Path
import unittest
from geometric_metrics import compare,point_segment_distance,Link6FK
from prepare_nrt import construct_commands,validate_waypoints,same_bits

class Position:
    def __init__(self,q):self.joints=list(q)
class Command:
    def __init__(self,target,speed,zone):self.target=copy.deepcopy(target);self.speed=speed;self.zone=zone
class FakeSDK:
    JointPosition=Position
    MoveAbsJCommand=Command
class RoundedPosition:
    def __init__(self,q):self.joints=[round(v,4) for v in q]

class Tests(unittest.TestCase):
    def test_100_commands_exact_all_policies(self):
        q=[[0.1234567890123456+i*1e-8,-0.0,0.2,0.3,0.4,0.5] for i in range(100)]
        for zone in (0,1,5):
            commands=construct_commands(FakeSDK,q,'SYNTHETIC',zone)
            self.assertEqual(len(commands),100)
            self.assertEqual(commands[-1].zone,0)
            for row,cmd in zip(q,commands):
                self.assertTrue(all(same_bits(x,y) for x,y in zip(row,cmd.target.joints)))
    def test_reject_partial_and_bad_values(self):
        for q in ([[0.0]*6]*99,[[0.0]*7]*100,[[math.nan]*6]*100):
            with self.assertRaises(ValueError):validate_waypoints(q)
    def test_detect_sdk_rounding(self):
        class Bad(FakeSDK):JointPosition=RoundedPosition
        with self.assertRaises(ValueError):construct_commands(Bad,[[0.123456789]*6]*100,'test')
    def test_distance_to_segment_interior(self):
        self.assertEqual(point_segment_distance([0.5,0,0],[0,0,0],[1,0,0]),0)
    def test_parallel_line_metric(self):
        r=compare([[0,0,0],[1,0,0]],[[0,0.01,0],[1,0.01,0]],0.01)
        self.assertAlmostEqual(r['symmetric_rms_m'],0.01,places=12)
        self.assertAlmostEqual(r['symmetric_arclength_p95_m'],0.01,places=12)
        self.assertFalse(r['rigid_alignment_applied'])
    def test_dwell_duplicate_invariance(self):
        target=[[0,0,0],[1,0,0]];curve=[[0,0.01,0],[1,0.01,0]]
        a=compare(target,curve,0.01);b=compare(target,[curve[0]]*100+curve,0.01)
        self.assertAlmostEqual(a['symmetric_rms_m'],b['symmetric_rms_m'],places=12)
    def test_missing_path_coverage(self):
        r=compare([[0,0,0],[1,0,0]],[[0,0,0],[0.5,0,0]],0.001)
        self.assertGreater(r['target_to_realized_rms_m'],0.2)
        self.assertLess(r['realized_to_target_rms_m'],1e-14)
    def test_zero_length_rejected(self):
        with self.assertRaises(ValueError):compare([[0,0,0]]*2,[[0,0,0],[1,0,0]])
    def test_authoritative_fk_matches_archived_reference_check(self):
        import csv,json
        package=Path(__file__).resolve().parents[2]/'laptop_handoff/laptop_handoff'
        fk=Link6FK(package/'robot_model/xMateCR7.urdf')
        with (package/'trajectories/path_0003/A.csv').open() as f:q=[list(map(float,r))[1:] for r in list(csv.reader(f))[1:]]
        with (package/'target_paths/path_0003.csv').open() as f:target=[list(map(float,r))[1:] for r in list(csv.reader(f))[1:]]
        rms=1000*math.sqrt(sum(math.dist(fk.position(a),b)**2 for a,b in zip(q,target))/100)
        planned=json.loads((package/'metadata/path_0003/planned_metrics.json').read_text())['R']['cartesian_rms_error_mm']
        self.assertAlmostEqual(rms,planned,places=7)

if __name__=='__main__':unittest.main()
