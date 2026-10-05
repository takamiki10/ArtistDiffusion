"""OFFLINE evidence audit only. Does not modify the runner or propose motion calls."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import struct
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import smartjoint_csv_nrt_runner as s

OUT = ROOT/'nrt_v050/evidence/smartjoint_duplicates_watchdog'


def json_file(path):
    return json.loads(path.read_text(encoding='utf-8'))


def json_lines(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def pair_evidence(data, index):
    if type(index) is not int or not 1 <= index < len(data.q):
        raise ValueError('an exact pair requires a valid current index with a predecessor')
    a, b = data.q[index-1], data.q[index]
    equal = [struct.pack('<d', x) == struct.pack('<d', y) for x,y in zip(a,b)]
    return {'preceding_index': index-1, 'current_index': index,
            'preceding_csv_line': index+1, 'current_csv_line': index+2,
            'preceding_TouchType': data.cells[index-1][1], 'current_TouchType': data.cells[index][1],
            'preceding_joint_tokens': data.cells[index-1][2:8], 'current_joint_tokens': data.cells[index][2:8],
            'preceding_q_rad': a, 'current_q_rad': b,
            'preceding_binary64_hex': [x.hex() for x in a], 'current_binary64_hex': [x.hex() for x in b],
            'binary64_equal_per_joint': equal, 'all_six_bitwise_identical': all(equal),
            'signed_delta_rad': [y-x for x,y in zip(a,b)],
            'maximum_abs_joint_delta_rad': max(abs(y-x) for x,y in zip(a,b))}


def session_evidence(folder, data):
    session = json_file(folder/'session.json')
    final = json_file(folder/'final_status.json')
    samples = json_lines(folder/'q_m.jsonl')
    calls = json_lines(folder/'calls.jsonl')
    events = json_lines(folder/'motion_events.jsonl')
    batches = json_file(folder/'append_batches.json')
    start = next(r for r in calls if r['api'] == 'moveStart' and r['phase'] == 'return')
    t0 = start['return_ns']
    running = [r for r in samples if r['decoded_end_ns'] >= t0]
    last = running[-1]
    first_movement = final['motion_observation']['first_threshold_sample']
    required_delta = [target-q0 for target,q0 in zip(data.q[0],session['q0_rad'])]
    dominant_joint = max(range(6), key=lambda j:abs(required_delta[j]))
    sign = 1 if required_delta[dominant_joint] > 0 else -1
    duration = (last['decoded_end_ns']-t0)/1e9
    displacement = sign*(last['q_native_rad'][dominant_joint]-session['q0_rad'][dominant_joint])
    steps = [sign*(b['q_native_rad'][dominant_joint]-a['q_native_rad'][dominant_joint])
             for a,b in zip(running,running[1:])]
    tail = [r for r in running if r['decoded_end_ns'] >= last['decoded_end_ns']-10_000_000_000]
    tail_duration = (tail[-1]['decoded_end_ns']-tail[0]['decoded_end_ns'])/1e9
    tail_rate = sign*(tail[-1]['q_native_rad'][dominant_joint]-tail[0]['q_native_rad'][dominant_joint])/tail_duration
    state = next(r for r in reversed(calls) if r['api'] == 'operationState' and r['phase'] == 'return')
    normal_progress = [r for r in events if not r['payload']['remark'] and r['payload']['error'] == {'ec':0,'message':'success'}]
    source_progress = [batches[r['payload']['cmdID']][0]+r['payload']['wayPointIndex'] for r in normal_progress]
    snapshots = []
    for seconds in (0,30,60,90,119,120):
        if seconds > duration:
            continue
        row = min(running, key=lambda r:abs(r['decoded_end_ns']-(t0+seconds*1e9)))
        snapshots.append({'seconds_after_start_return': (row['decoded_end_ns']-t0)/1e9,
                          'sequence': row['sequence'], 'q_rad': row['q_native_rad'],
                          'max_abs_residual_to_first_target_rad': max(abs(a-b) for a,b in zip(row['q_native_rad'],data.q[0]))})
    if not snapshots or snapshots[-1]['sequence'] != last['sequence']:
        snapshots.append({'seconds_after_start_return': duration, 'sequence': last['sequence'],
                          'q_rad': last['q_native_rad'],
                          'max_abs_residual_to_first_target_rad': max(abs(a-b) for a,b in zip(last['q_native_rad'],data.q[0]))})
    sdk_call_errors = [r for r in calls if r.get('phase') in ('return','exception','cleanup')
                       and (r.get('phase') == 'exception' or
                            (r.get('error') not in (None, {}, {'ec':0,'message':'success'})))]
    return {'session': folder.name, 'final_utc': final['utc'], 'status': final['status'],
            'failure_reason': final['failure_reason'], 'start_return_host_ns': t0,
            'first_observed_movement': first_movement,
            'movement_onset_after_start_return_s': None if first_movement is None else (first_movement['timestamp_ns']-t0)/1e9,
            'last_q': last, 'q_observation_after_start_return_s': duration,
            'q_motion_duration_after_first_observed_movement_s': None if first_movement is None else
                (last['decoded_end_ns']-first_movement['timestamp_ns'])/1e9,
            'q_record_count': len(samples), 'invalid_q_count': sum(not r['valid_q'] for r in samples),
            'last_operation_state': state, 'last_state_after_start_return_s': (state['host_ns']-t0)/1e9,
            'last_state_age_at_last_q_s': (last['decoded_end_ns']-state['host_ns'])/1e9,
            'last_state_was_moving': state['return_repr'] == '<OperationState.moving: 9>',
            'raw_events': events, 'last_callback_index': None if not events else events[-1]['payload']['wayPointIndex'],
            'last_non_diagnostic_source_index': None if not source_progress else source_progress[-1],
            'non_diagnostic_source_indices': source_progress,
            'policy_failures': final['policy_failures'], 'hard_event_fault': final['hard_event_fault'],
            'sdk_call_errors': sdk_call_errors, 'cleanup_errors': final['cleanup_errors'],
            'sole_obstacle_to_acceptance_was_timeout': final['failure_reason'] == 'MEASURED_COMPLETION_NOT_CONFIRMED'
                and not final['policy_failures'] and not final['hard_event_fault'] and not sdk_call_errors,
            'submitted_waypoints': final['appended_rows'], 'batches': batches,
            'observed_progress_scope': 'measured approach toward first target; not completed drawing progress',
            'initial_q0': session['q0_rad'], 'first_target': data.q[0], 'initial_target_delta_rad': required_delta,
            'dominant_approach_joint': dominant_joint+1, 'dominant_displacement_rad': displacement,
            'dominant_fraction_of_initial_displacement': displacement/abs(required_delta[dominant_joint]),
            'dominant_mean_rate_rad_s': displacement/duration,
            'last_10s_dominant_rate_rad_s': tail_rate,
            'dominant_backward_steps_over_1e_5_rad': sum(d < -1e-5 for d in steps),
            'minimum_signed_dominant_step_rad': min(steps), 'approach_snapshots': snapshots,
            'controller_sample_timestamp_available': any(r.get('controller_timestamp') is not None for r in samples)}


def build_report():
    data = s.load_csv()
    pairs = [pair_evidence(data,i) for i in range(1,len(data.q))]
    duplicates = [p for p in pairs if p['all_six_bitwise_identical']]
    sessions = [session_evidence(f,data) for f in sorted((ROOT/'nrt_v050/smartjoint_sessions').iterdir())
                if f.is_dir() and (f/'final_status.json').exists()]
    first_timeout = next(r for r in sessions if r['failure_reason'] == 'MEASURED_COMPLETION_NOT_CONFIRMED')
    path_budget = sum(max(abs(a-b) for a,b in zip(x,y)) for x,y in zip(data.q,data.q[1:]))
    approach_budget = max(map(abs,first_timeout['initial_target_delta_rad']))
    # Explicit offline planning heuristic, NOT a controller speed model or upper bound.
    illustrative_rate = 0.025
    illustrative_duration = (approach_budget+path_budget)/illustrative_rate+1.0
    protected = json_file(ROOT/'build_offline/smartjoint_duplicates_watchdog_baseline/sha256.json')
    for name,digest in protected.items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != digest:
            raise ValueError('protected file changed: '+name)
    return {'offline_only': True, 'source_sha256': hashlib.sha256(data.raw).hexdigest(),
            'row_count': len(data.q), 'timestamp_span_metadata_only': data.timestamps[-1]-data.timestamps[0],
            'duplicate_pairs': duplicates, 'index_one_pair': pairs[0],
            'index_one_is_exact_duplicate': pairs[0]['all_six_bitwise_identical'],
            'observed_warning_exception_justified': False,
            'production_exception_implemented': False,
            'exception_decision': 'The stated relationship to an exact duplicate is false for observed index 1; leave production warning policy unchanged.',
            'sessions': sessions, 'protected_files_unchanged': len(protected),
            'watchdog': {'current_s': s.n.SETTINGS['completion_timeout_s'], 'recommended_prospective_s': 600,
                         'applied': False, 'recommendation_scope': 'review-only fixed diagnostic bound, not an execution-duration promise',
                         'observed_mean_approach_rates_rad_s': [r['dominant_mean_rate_rad_s'] for r in sessions],
                         'illustrative_rate_rad_s': illustrative_rate,
                         'initial_approach_max_joint_displacement_rad': approach_budget,
                         'source_sum_adjacent_max_joint_deltas_rad': path_budget,
                         'illustrative_approach_s': approach_budget/illustrative_rate,
                         'illustrative_source_motion_s': path_budget/illustrative_rate,
                         'settling_allowance_in_illustration_s': 1,
                         'illustrative_total_s': illustrative_duration,
                         'margin_over_illustrative_total_s': 600-illustrative_duration,
                         'ratio_to_illustrative_total': 600/illustrative_duration,
                         'limitations': 'Only approach motion was observed. The 508-point short-segment motion, planning delays and settling time have not been measured. No guaranteed upper bound for completion can be derived.'},
            'ready_for_start': False, 'unresolved': ['nonduplicate adjacent-points warning remains disqualifying',
                                                   'watchdog recommendation not yet prospectively adopted']}


def main():
    report = build_report()
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'audit.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    with (OUT/'duplicate_pairs.csv').open('w',encoding='utf-8',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=list(report['duplicate_pairs'][0]))
        writer.writeheader()
        writer.writerows(report['duplicate_pairs'])
    print(json.dumps({'duplicates':[[p['preceding_index'],p['current_index']] for p in report['duplicate_pairs']],
                      'index_one_exact_duplicate':report['index_one_is_exact_duplicate'],
                      'watchdog_recommendation_s':600,'watchdog_applied':False,'audit':str(OUT/'audit.json')},indent=2))


if __name__ == '__main__':main()
