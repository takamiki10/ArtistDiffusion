"""Prepared offline SDK data-object audit. Never constructs a robot object.

Blocked by current Windows Code Integrity policy; run only after legitimate
runtime resolution. No motion methods are called by this script.
"""
from pathlib import Path
import json
import os
import struct
import sys
from physical_reference_30s import load_all
from prepare_nrt import construct_commands

HERE=Path(__file__).resolve().parent
assert sys.version_info[:2]==(3,12) and struct.calcsize('P')==8
handle=os.add_dll_directory(str(HERE/'sdk'))
import xCoreSDK_python as sdk

count=0
for name,digest,data in load_all():
    for zone in (0,1,5):
        commands=construct_commands(sdk,[row[1:] for row in data],name.replace('/','_'),zone)
        assert len(commands)==100 and all(isinstance(c,sdk.MoveAbsJCommand) for c in commands)
        count+=100*6
out={'status':'PASS','joint_value_checks':count,'policies':[0,1,5],'robot_objects_constructed':0,
     'moveAppend_called':False,'moveStart_called':False,
     'note':'Data-object roundtrip only, not serialization/controller acceptance or list capacity proof.'}
(HERE/'evidence/binding_roundtrip.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
print(json.dumps(out))
