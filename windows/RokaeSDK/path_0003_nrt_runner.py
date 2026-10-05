"""Frozen path_0003 geometric experiment. Inert import; one operator-gated trial per CLI invocation."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
import uuid

# Share the reviewed serial SDK lifecycle, raw logger, error/event validation,
# conditional START boundary and immutable controller command construction.
sys.path.insert(0, str(Path(__file__).resolve().parent))  # Explicit local import under Python -I.
import neutral_v050_nrt_runner as n

ROOT = Path(__file__).resolve().parent
PREPARED = ROOT / 'nrt_v050' / 'prepared'
SESSIONS = ROOT / 'nrt_v050' / 'path_0003_sessions'
ORDER = ('R', 'D', 'D', 'R', 'D', 'R', 'R', 'D', 'R', 'D')
PINS = {
    'experiment_manifest.json': '3c17a003b808bf234d6585c7f8c49f04a44ff0d21bcac2b66a6ffd01e5518218',
    'path_0003_A_waypoints.json': '8bf57a6e7d69fe2d089f6436c8c031910e9a5f3c23a5f70d18cff113be644780',
    'path_0003_B_waypoints.json': '556b9c3c3c38b482ad58bac0aa1135bb1c2e1e9c077f53f4bbf65312588801ba',
}
PROTOCOL = {'revision': 'neutral-derived-geometric-completion-20260913',
            'neutral_evidence': '20260913T131737Z_4f690650',
            'endpoint_tolerance_rad_per_joint': 1e-3, 'stationary_span_rad': 1e-5,
            'window_s': 1.0, 'minimum_samples': 100, 'maximum_gap_ns': 50_000_000,
            'observed_motion_rad': 1e-4, 'terminal_callback_required': False,
            'valid_for_timing_analysis': False, 'rd_outcomes_used_for_amendment': False}


def load_trial(number):
    n.require(type(number) is int and 1 <= number <= 10, 'trial must be 1..10')
    blobs = {name: (PREPARED/name).read_bytes() for name in PINS}
    for name, blob in blobs.items():
        n.require(n.digest_bytes(blob) == PINS[name], 'frozen file hash changed: '+name)
    manifest = json.loads(blobs['experiment_manifest.json'])
    n.require(tuple(t['condition'] for t in manifest['trials']) == ORDER, 'frozen order mismatch')
    trial = manifest['trials'][number-1]
    n.require(trial['trial'] == number and trial['pair'] == (number+1)//2 and trial['path'] == 'path_0003', 'trial mapping mismatch')
    filename = 'path_0003_'+('A' if trial['condition']=='R' else 'B')+'_waypoints.json'
    batch = json.loads(blobs[filename])
    n.require(batch['source'] == trial['source'], 'source mismatch')
    rows = batch['q_rad']
    n.matrix_bytes(rows)
    n.read_policy()
    return trial, rows, blobs[filename]


def stationary_evidence(records, target, after_ns):
    """Measured completion only: no interpolation, residual correction or event dependency."""
    rows = [r for r in records if r['decoded_end_ns'] >= after_ns]
    if len(rows) < PROTOCOL['minimum_samples'] or any(not r.get('valid_q', False) for r in rows):
        return None
    duration = rows[-1]['decoded_end_ns']-rows[0]['decoded_end_ns']
    if duration < 1_000_000_000:
        return None
    gaps = [b['decoded_end_ns']-a['decoded_end_ns'] for a,b in zip(rows, rows[1:])]
    spans = [max(r['q_native_rad'][j] for r in rows)-min(r['q_native_rad'][j] for r in rows) for j in range(6)]
    residual = [a-b for a,b in zip(rows[-1]['q_native_rad'],target)]
    worst = [max(abs(r['q_native_rad'][j]-target[j]) for r in rows) for j in range(6)]
    if min(gaps)<=0 or max(gaps)>50_000_000 or max(spans)>1e-5 or max(worst)>1e-3:
        return None
    return {'first_sequence':rows[0]['sequence'], 'last_sequence':rows[-1]['sequence'],
            'sample_count':len(rows), 'duration_ns':duration, 'per_joint_span_rad':spans,
            'maximum_gap_ns':max(gaps), 'final_measured_q_rad':rows[-1]['q_native_rad'],
            'final_commanded_q_rad':target, 'endpoint_residual_rad':residual,
            'window_max_abs_endpoint_residual_rad':worst}


class TrialSession(n.PreparedSession):
    def __init__(self, sdk, store, number):
        super().__init__(sdk, store)
        self.trial, self.rows, self.batch_bytes = load_trial(number)
        self.final_stationary = None

    def authorize(self, action, session_hash):
        n.require(action in ('SETUP','APPEND','START') and action not in self.authorized, 'invalid authorization')
        phrase = f"{action} PATH0003 TRIAL {self.trial['trial']} {self.trial['condition']} {session_hash}"
        n.require(self.prompt('Type exactly '+phrase+': ') == phrase, action+' not authorized')
        self.store.write('authorizations', {**n.stamp(), 'action':action,'accepted':True,'phrase':phrase})
        self.store.barrier()
        self.authorized.add(action)

    def observe_neutral_motion(self, row):
        # Physical movement may involve any of the six joints in frozen R/D.
        if not self.start_boundary or not self.start_boundary['accepted'] or row['sequence']<=self.start_boundary['observable_window_first_sequence']:
            return
        delta = [a-b for a,b in zip(row['q_native_rad'],self.q0)]
        excursion = max(abs(d) for d in delta)
        e = self.motion_observation
        if e.get('maximum_observed_excursion_rad') is None or excursion>e['maximum_observed_excursion_rad']:
            e.update(maximum_observed_excursion_rad=excursion, maximum_sequence=row['sequence'], maximum_timestamp_ns=row['decoded_end_ns'])
        if excursion>=1e-4 and e['first_threshold_sample'] is None:
            e['first_threshold_sample']={'sequence':row['sequence'],'timestamp_ns':row['decoded_end_ns'],'per_joint_delta_rad':delta}
            self.store.write('lifecycle',{'code':'PHYSICAL_MOTION_OBSERVED',**e['first_threshold_sample']})

    def execute(self):
        status, errors = 'FAILED_EXECUTION', []
        self.tag = 'p3-'+uuid.uuid4().hex
        config = {'trial':self.trial, 'protocol':PROTOCOL, 'pins':PINS,
                  'runner_sha256':n.digest_bytes(Path(__file__).read_bytes()),
                  'shared_runner_sha256':n.digest_bytes(Path(n.__file__).read_bytes()),
                  'waypoints_sha256':n.digest_bytes(n.matrix_bytes(self.rows))}
        try:
            self.store.json('session.json',config)
            self.store.bytes('frozen_source_waypoints.json',self.batch_bytes)
            self.store.json('waypoints.json',{'q_rad':self.rows,'count':100})
            self.store.bytes('waypoints.f64le',n.matrix_bytes(self.rows))
            self.store.bytes('controller_policy.json',n.read_policy()[0])
            phrase=f"CONNECT PATH0003 TRIAL {self.trial['trial']} {self.trial['condition']}"
            print('Frozen R/D trial:', self.trial, '\nPower already ON; stationary arm; other motion owners stopped; RCI OFF.\nReview full path and first-target transition clearance. Physical safety controls available.\nInitialization/reset and disconnect behavior remain as reviewed.',flush=True)
            answer=input('Type exactly '+phrase+': ')
            self.store.write('authorizations',{**n.stamp(),'action':'PRECONNECTION','phrase':phrase,'answer':answer,'accepted':answer==phrase})
            n.require(answer==phrase,'connection not authorized')
            self.store.barrier()
            self.robot=self.sdk.xMateRobot()
            self.connect_attempted=True
            n.require(self.robot.connectToRobot(n.REMOTE_IP,n.LOCAL_IP) is None,'unexpected connect return')
            info=self.checked_call('robotInfo',void=False)
            identity={k:getattr(info,k) for k in ('id','type','version','joint_num')}
            self.store.json('robot_identity.json',identity)
            n.require(info.type==n.EXPECTED_ROBOT_TYPE and type(info.joint_num) is int and info.joint_num==6,'robot identity mismatch')
            self.robot_state(True)
            self.checked_call('setEventWatcher',self.sdk.Event.moveExecution,self.inbox.callback)
            n.require(self.robot.startReceiveRobotState(n.timedelta(seconds=.008),['q_m']) is None,'subscription return')
            self.state('ACQUIRING_Q0')
            self.fresh_barrier()
            self.records.clear()
            self.wait_settled()
            self.q0=list(self.last_sample['q_native_rad'])
            print('Measured q0:',self.q0,'\nFirst frozen target:',self.rows[0], '\nFirst target delta:',[a-b for a,b in zip(self.rows[0],self.q0)],flush=True)
            config.update(q0_rad=self.q0,robot_identity=identity)
            self.authorize('SETUP',n.digest_json(config))
            self.premotion_call('setMotionControlMode',self.sdk.MotionControlMode.NrtCommandMode)
            self.premotion_call('moveReset')
            commands=n.construct_commands(self.sdk,self.rows,self.tag)
            snapshots=n.validate_commands(self.sdk,commands,self.rows,self.tag)
            config.update(commands_sha256=n.digest_json(snapshots),tag=self.tag)
            session_hash=n.digest_json(config)
            self.store.json('session.json',{**config,'session_sha256':session_hash})
            self.store.json('commands.json',snapshots)
            self.authorize('APPEND',session_hash)
            self.pre_control_check(commands,self.rows,self.tag,config)
            cmd_id=self.sdk.PyString()
            try:
                self.premotion_call('moveAppend',commands,cmd_id)
            finally:
                self.store.write('calls',{**n.stamp(),'api':'moveAppend','phase':'command_id','cmdID':n.snapshot(cmd_id.content(),self.sdk)})
            self.cmd_id=cmd_id.content()
            n.require(type(self.cmd_id) is str and bool(self.cmd_id.strip()) and '\x00' not in self.cmd_id,'invalid append command ID')
            self.state('READY')
            self.authorize('START',session_hash)
            self.pre_control_check(commands,self.rows,self.tag,config)
            self.checked_call('moveStart')
            self.state('RUNNING')
            self.read_sample()
            deadline=time.perf_counter()+n.SETTINGS['completion_timeout_s']
            while time.perf_counter()<deadline:
                self.pump()
                motion=self.motion_observation['first_threshold_sample']
                if motion is None:
                    continue
                evidence=stationary_evidence(list(self.records),self.rows[-1],motion['timestamp_ns'])
                if evidence and self.robot_state(False)==self.sdk.OperationState.idle:
                    self.fresh_barrier()
                    evidence=stationary_evidence(list(self.records),self.rows[-1],motion['timestamp_ns'])
                    if evidence and self.robot_state(False)==self.sdk.OperationState.idle:
                        self.drain_events()
                        self.final_stationary=evidence
                        status='COMPLETED_REJECTED' if self.completion.policy_failures else 'COMPLETED_ACCEPTED'
                        break
            n.require(status!='FAILED_EXECUTION','NO_OBSERVED_PHYSICAL_MOTION' if self.motion_observation['first_threshold_sample'] is None else 'MEASURED_COMPLETION_NOT_CONFIRMED')
        except BaseException as exc:
            self.failure_reason=str(exc)
            self.store.write('exceptions',{**n.stamp(),'error':repr(exc)})
            if self.start_attempted:
                print('Motion status requires operator observation; use reviewed physical safety procedure. '
                      'No explicit stop command is sent; cleanup disconnect stops ongoing motion per SDK documentation. '
                      'No automatic reset or recovery.',flush=True)
        finally:
            if self.connect_attempted:
                for name in ('stopReceiveRobotState','setNoneEventWatcher','disconnectFromRobot'):
                    ec={}
                    try:
                        if name=='stopReceiveRobotState': result=self.robot.stopReceiveRobotState()
                        elif name=='setNoneEventWatcher': result=self.robot.setNoneEventWatcher(self.sdk.Event.moveExecution,ec)
                        else: result=self.robot.disconnectFromRobot(ec)
                        self.store.write('calls',{**n.stamp(),'api':name,'phase':'cleanup','return_repr':repr(result),'error':n.snapshot(ec,self.sdk),'ec_repr':repr(ec)})
                        n.require(result is None,'unexpected cleanup return')
                        if name!='stopReceiveRobotState': n.success_error(n.snapshot(ec,self.sdk))
                    except BaseException as exc: errors.append({'api':name,'error':repr(exc)})
                try: self.drain_events()
                except BaseException as exc: errors.append({'api':'final_event_drain','error':repr(exc)})
            if errors or (self.completion and self.completion.hard_execution_fault): status='FAILED_EXECUTION'
            elif status=='COMPLETED_ACCEPTED' and self.completion.policy_failures: status='COMPLETED_REJECTED'
            self.store.json('final_stationary.json',self.final_stationary)
            self.store.json('final_status.json', {**n.stamp(),'status':status,'trial':self.trial,'failure_reason':self.failure_reason,
                'valid_for_geometric_analysis':status=='COMPLETED_ACCEPTED','valid_for_timing_analysis':False,
                'start_boundary':self.start_boundary or self.pending_boundary,'moveStart_success':self.start_succeeded,
                'append_success':self.append_complete,'motion_observation':self.motion_observation,
                'final_stationary':self.final_stationary,'max_motion_gap_ns':self.max_motion_gap_ns,
                'max_post_boundary_gap_ns':self.max_post_boundary_gap_ns,'attempts':self.attempts,
                'policy_failures':[] if self.completion is None else self.completion.policy_failures,
                'hard_event_fault':bool(self.completion and self.completion.hard_execution_fault),'cleanup_errors':errors})
            self.store.barrier()
        return status


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-trial',action='store_true')
    parser.add_argument('--trial',type=int,choices=range(1,11),required=True)
    args=parser.parse_args(argv)
    if not args.run_trial:
        parser.print_help(); return 2
    load_trial(args.trial)  # Offline pinned-file checks before SDK import.
    SESSIONS.mkdir(parents=True,exist_ok=True)
    for prior in range(1,args.trial):
        marker=SESSIONS/f'trial_{prior:02d}.json'
        n.require(marker.exists(),'previous trial has not been attempted')
        folder=Path(json.loads(marker.read_text())['folder'])
        n.require((folder/'final_status.json').exists(),'previous trial has no final disposition')
    folder=SESSIONS/(f'trial_{args.trial:02d}_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:8])
    # Exclusive durable reservation prevents accidental duplicate trial execution.
    with (SESSIONS/f'trial_{args.trial:02d}.json').open('x',encoding='utf-8') as f:
        json.dump({'trial':args.trial,'folder':str(folder)},f)
    store=n.SessionStore(folder)
    handle=None
    try:
        sdk,handle=n.load_exact_sdk_data_only()
        status=TrialSession(sdk,store,args.trial).execute()
        print('Trial status:',status,'\nSession:',folder,flush=True)
        return 0 if status=='COMPLETED_ACCEPTED' else 1
    finally:
        store.close()
        if handle is not None: handle.close()


if __name__=='__main__':
    raise SystemExit(main())
