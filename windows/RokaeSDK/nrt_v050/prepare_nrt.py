"""Offline waypoint/policy preparation. No robot object or connection code."""
from pathlib import Path
import hashlib
import json
import math
import random
import struct
from physical_reference_30s import load_all

HERE=Path(__file__).resolve().parent
POLICY={'name':'NRT_5percent_zone1_final_fine_v1','jointSpeed':0.05,'speed':50,
        'interior_zone':1,'final_zone':0,'units':{'jointSpeed':'fraction','speed':'mm/s','zone':'mm'},
        'candidate_zones':{'A_fine':0,'B_selected_small':1,'C_comparison_only':5},
        'motion_authorized':False,'controller_semantics_validated':False,
        'binding_round_trip_verified':False,'physical_timing':'controller planned; no prescribed duration'}

def same_bits(a,b):
    return struct.pack('<d',a)==struct.pack('<d',b)

def validate_waypoints(q):
    if len(q)!=100 or any(len(row)!=6 for row in q):
        raise ValueError('exactly 100 six-joint waypoints required')
    if not all(type(v) in (float,int) and math.isfinite(v) for row in q for v in row):
        raise ValueError('finite joint values required')

def construct_commands(sdk,q,tag,zone=1):
    """Only data objects. Calling this never appends or starts robot motion."""
    validate_waypoints(q)
    if zone not in (0,1,5) or not tag.isascii():
        raise ValueError('unregistered policy/tag')
    commands=[]
    for i,row in enumerate(q):
        position=sdk.JointPosition(list(row))
        if len(position.joints)!=6 or any(not same_bits(x,y) for x,y in zip(row,position.joints)):
            raise ValueError('JointPosition changed a waypoint')
        cmd=sdk.MoveAbsJCommand(position,50,0 if i==99 else zone)
        cmd.jointSpeed=0.05
        cmd.customInfo=f'{tag}:k{i:03d}'
        if len(cmd.target.joints)!=6 or any(not same_bits(x,y) for x,y in zip(row,cmd.target.joints)):
            raise ValueError('MoveAbsJCommand changed a waypoint')
        if cmd.jointSpeed!=0.05 or cmd.speed!=50 or cmd.zone!=(0 if i==99 else zone):
            raise ValueError('SDK policy property did not round trip')
        if cmd.customInfo!=f'{tag}:k{i:03d}':
            raise ValueError('customInfo did not round trip')
        commands.append(cmd)
    return commands

def prepare():
    loaded=load_all()
    out=HERE/'prepared';out.mkdir(exist_ok=False)
    batches={}
    for name,digest,data in loaded:
        label=name.replace('trajectories/','').replace('/','_').removesuffix('.csv')
        q=[row[1:] for row in data];validate_waypoints(q)
        file=out/(label+'_waypoints.json')
        obj={'source':name,'source_sha256':digest,'input_times_for_provenance_only':[r[0] for r in data],
             'q_rad':q,'count':100,'timing':'NRT controller planned','original_knots_unchanged':True}
        file.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n',encoding='utf-8')
        back=json.loads(file.read_text())['q_rad']
        assert all(same_bits(x,y) for a,b in zip(q,back) for x,y in zip(a,b))
        batches[label]={'source_sha256':digest,'prepared_sha256':hashlib.sha256(file.read_bytes()).hexdigest()}
    (out/'controller_policy.json').write_text(json.dumps(POLICY,indent=2)+'\n',encoding='utf-8')
    orders=['RD']*3+['DR']*2;random.Random(20260913).shuffle(orders)
    schedule=[]
    for pair,order in enumerate(orders,1):
        for condition in order:
            schedule.append({'trial':len(schedule)+1,'pair':pair,'condition':condition,'path':'path_0003',
                             'source':f'trajectories/path_0003/{"A" if condition=="R" else "B"}.csv',
                             'status':'PLANNED_NOT_EXECUTED'})
    plan={'protocol':'prospective NRT hardware-transfer ablation','no_physical_outcomes_observed':True,
          'primary_path':'path_0003','other_paths_retained':['path_0001','path_0006'],
          'trial_count':10,'repetitions_each':5,'seed':20260913,'trials':schedule,
          'policy':POLICY,'batches':batches,'sdk_constructed':False,'source_import_blocked':True}
    (out/'experiment_manifest.json').write_text(json.dumps(plan,indent=2)+'\n',encoding='utf-8')
    print('Prepared six unchanged 100-waypoint batches; path_0003 schedule is 5 R + 5 D; no SDK import.')

if __name__=='__main__':prepare()
