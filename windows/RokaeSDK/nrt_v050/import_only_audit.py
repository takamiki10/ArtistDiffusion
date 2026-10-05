"""Import and class inspection only. Never construct any robot/controller object."""
from pathlib import Path
import ctypes
import hashlib
import json
import os
import platform
import struct
import sys

HERE=Path(__file__).resolve().parent
assert sys.version_info[:2]==(3,12) and struct.calcsize('P')==8
handle=os.add_dll_directory(str(HERE/'sdk'))
try:
    import xCoreSDK_python as sdk
except ImportError as exc:
    out=HERE/'evidence';out.mkdir(exist_ok=True)
    (out/'import_only.json').write_text(json.dumps({
        'scope':'IMPORT_ONLY_NO_ROBOT_OBJECT_NO_CONNECTION','status':'BLOCKED',
        'python':sys.version,'executable':sys.executable,'architecture_bits':64,
        'isolated':bool(sys.flags.isolated),'site_imported':'site' in sys.modules,
        'error':str(exc),'error_type':type(exc).__name__,
        'robot_objects_constructed':0},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    raise SystemExit('BLOCKED: SDK import failed; see evidence/import_only.json') from exc

required=['connectToRobot','disconnectFromRobot','setMotionControlMode',
          'moveAppend','moveStart','stop','pause','moveReset','startReceiveRobotState',
          'stopReceiveRobotState','updateRobotState','getStateData','robotInfo',
          'operationState','powerState','operateMode','setEventWatcher']
missing=[name for name in required if not hasattr(sdk.xMateRobot,name)]
assert not missing,missing
assert hasattr(sdk.MotionControlMode,'NrtCommandMode')
assert hasattr(sdk,'JointPosition') and hasattr(sdk,'MoveAbsJCommand')
assert sdk.RtSupportedFields.jointPos_m=='q_m'
assert Path(sdk.__file__).resolve().is_relative_to((HERE/'sdk').resolve())
kernel=ctypes.WinDLL('kernel32',use_last_error=True)
kernel.GetModuleHandleW.argtypes=[ctypes.c_wchar_p]
kernel.GetModuleHandleW.restype=ctypes.c_void_p
kernel.GetModuleFileNameW.argtypes=[ctypes.c_void_p,ctypes.c_wchar_p,ctypes.c_uint]
loaded={}
for name in ['xCoreSDK.dll','python312.dll','MSVCP140.dll','VCRUNTIME140.dll']:
    module=kernel.GetModuleHandleW(name)
    buf=ctypes.create_unicode_buffer(32768)
    assert module and kernel.GetModuleFileNameW(module,buf,len(buf))
    path=Path(buf.value)
    loaded[name]={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
assert Path(loaded['xCoreSDK.dll']['path']).resolve()==(HERE/'sdk/xCoreSDK.dll').resolve()
result={'scope':'IMPORT_ONLY_NO_ROBOT_OBJECT_NO_CONNECTION','python':sys.version,
        'executable':sys.executable,'architecture_bits':64,'platform':platform.platform(),
        'isolated':bool(sys.flags.isolated),'site_imported':'site' in sys.modules,
        'sys_path':sys.path,'sdk_module':sdk.__file__,'loaded_libraries':loaded,
        'required_methods_present':required,'NrtCommandMode':str(sdk.MotionControlMode.NrtCommandMode),
        'measured_joint_field':sdk.RtSupportedFields.jointPos_m,
        'binary_docstrings':{name:getattr(sdk.xMateRobot,name).__doc__ for name in required},
        'command_docstrings':{name:getattr(sdk,name).__doc__ for name in ['NrtCommand','JointPosition','MoveAbsJCommand']},
        'robot_objects_constructed':0}
out=HERE/'evidence';out.mkdir(exist_ok=True)
(out/'import_only.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('PASS: Python 3.12 x64 imported exact local SDK; all required NRT methods present; zero robot objects.')
