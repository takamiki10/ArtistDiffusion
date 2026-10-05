"""Reconstruct preserved SmartJoint event-order failure using local files only."""
from __future__ import annotations

import ast
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import neutral_v050_nrt_runner as n
import smartjoint_csv_nrt_runner as s


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_lines(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def definition(path, name):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    return ast.dump(next(node for node in tree.body if getattr(node, 'name', None) == name), include_attributes=False)


def main():
    sessions = ROOT/'nrt_v050/smartjoint_sessions'
    candidates = []
    for folder in sessions.iterdir():
        final_path = folder/'final_status.json'
        if folder.is_dir() and final_path.exists():
            final = json.loads(final_path.read_text())
            if final.get('trial') == 1 and 'event index regressed' in final.get('failure_reason', ''):
                candidates.append((final['utc'], folder))
    _, folder = max(candidates)
    events = read_lines(folder/'motion_events.jsonl')
    q = read_lines(folder/'q_m.jsonl')
    source_rows = read_lines(folder/'source_rows.jsonl')
    calls = read_lines(folder/'calls.jsonl')
    session = json.loads((folder/'session.json').read_text())
    final = json.loads((folder/'final_status.json').read_text())
    batches = json.loads((folder/'append_batches.json').read_text())
    start_ns = final['start_boundary']['request_ns']
    baseline = ROOT/'build_offline/smartjoint_event_order_baseline'
    preserved = json.loads((baseline/'sha256.json').read_text())
    for name, digest in preserved['sessions'].items():
        n.require(sha(ROOT/name) == digest, 'preserved session changed: '+name)
    for name in ('SmartJoint_Data_diffusion.csv', 'nrt_v050/prepared/controller_policy.json'):
        n.require(sha(ROOT/name) == preserved['files'][name], 'protected input changed: '+name)
    n.require(definition(ROOT/'smartjoint_csv_nrt_runner.py', 'CsvSession') ==
              definition(baseline/'smartjoint_csv_nrt_runner.py', 'CsvSession'), 'motion lifecycle changed')
    for name in ('PreparedSession', 'construct_commands', 'validate_commands', 'matrix_bytes', 'read_policy'):
        n.require(definition(ROOT/'neutral_v050_nrt_runner.py', name) ==
                  definition(baseline/'neutral_v050_nrt_runner.py', name), 'shared motion definition changed: '+name)
    data = s.load_csv()
    n.require(n.matrix_bytes([r['q_rad'] for r in source_rows], 508) == n.matrix_bytes(data.q, 508), 'source rows differ')
    commands = json.loads((folder/'commands.json').read_text())
    n.require(n.matrix_bytes([r['target_rad'] for r in commands], 508) == n.matrix_bytes(data.q, 508), 'commands differ')
    n.require([r['zone'] for r in commands] == [1]*507+[0], 'zones differ')
    n.require(all(r['jointSpeed'] == .05 and r['speed'] == 50 for r in commands), 'speed policy differs')
    n.require(list(batches.values()) == [list(x) for x in s.BATCHES], 'batch spans differ')
    n.require(n.SETTINGS == session['settings'], 'lifecycle settings changed')
    chronology = []
    evidence_q = []
    for line_number, event in enumerate(events, 1):
        p = event['payload']
        cmd_id = p['cmdID']
        span = batches[cmd_id]
        source_index = span[0] + p['wayPointIndex']
        severity = 'warning (derived from remark)' if p['remark'] else 'informational (derived)'
        row = {'event_log_line': line_number, 'callback_host_ns': event['host_ns'],
               'seconds_after_start_request': (event['host_ns']-start_ns)/1e9,
               'controller_event_timestamp': None, 'event_type': 'Event.moveExecution',
               'event_type_source': 'registered watcher; not an explicit payload field',
               'severity': severity, 'cmdID': cmd_id, 'batch_number_1_based': list(batches).index(cmd_id)+1,
               'batch_source_span': span, 'batch_local_index': p['wayPointIndex'],
               'source_index': source_index, 'reachTarget': p['reachTarget'], 'error': p['error'],
               'remark': p['remark'], 'classification': 'diagnostic, not progress evidence' if p['remark'] else 'completion report',
               'raw_event_json': json.dumps(event, ensure_ascii=False, separators=(',', ':'))}
        chronology.append(row)
        before = max((r for r in q if r['decoded_end_ns'] <= event['host_ns']), key=lambda r:r['decoded_end_ns'])
        after = min((r for r in q if r['decoded_end_ns'] >= event['host_ns']), key=lambda r:r['decoded_end_ns'], default=None)
        evidence_q.append({'event_log_line': line_number, 'before': before, 'after': after,
                           'target_rad': source_rows[source_index]['q_rad'],
                           'before_max_abs_target_residual_rad': max(abs(a-b) for a,b in zip(before['q_native_rad'],source_rows[source_index]['q_rad']))})
    # Import only the saved inert Python validator; no SDK import or robot.
    spec = importlib.util.spec_from_file_location('smartjoint_pre_fix_validator', baseline/'neutral_v050_nrt_runner.py')
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    legacy = old.Completion(events[0]['payload']['cmdID'], session['tag'], start_ns, 508)
    replay_before = []
    for event in events:
        try:
            legacy.observe(event)
            replay_before.append({'last_index': legacy.last_index, 'error': None})
        except old.ValidationError as exc:
            replay_before.append({'last_index': legacy.last_index, 'error': str(exc)})
            break
    fixed = s.BatchCompletion(batches, session['tag'], start_ns)
    replay_after = []
    for event in events:
        fixed.observe(event)
        replay_after.append({'last_index': fixed.last_index, 'seen': sorted(fixed.seen),
                             'warning_count': len(fixed.policy_failures), 'terminal_ns': fixed.terminal_ns,
                             'hard_execution_fault': fixed.hard_execution_fault})
    documentation = [
        'nrt_v050/sdk/xCoreSDK_python/EventInfoKey/MoveExecution.pyi',
        'nrt_v050/sdk/xCoreSDK_python/__init__.pyi',
        'evidence/v050/xcoresdk_python-v0.5.0__example__move_example.py',
        'evidence/v050/package_manifest.json']
    report = {'offline_only': True, 'robot_constructed': False, 'sdk_imported': False,
              'source_session': str(folder), 'final_status': final,
              'chronology': chronology, 'q_near_callbacks': evidence_q,
              'q_sample_count': len(q), 'invalid_q_samples': sum(not r['valid_q'] for r in q),
              'controller_timestamps_present': any(r['controller_timestamp'] is not None for r in q),
              'callback_gap_seconds': (events[1]['host_ns']-events[0]['host_ns'])/1e9,
              'event_watcher_calls': [r for r in calls if r['api'] == 'setEventWatcher'],
              'batches': batches, 'source_sha256': sha(s.SOURCE),
              'protected_session_files_unchanged': len(preserved['sessions']),
              'trajectory_commands_batching_policy_and_lifecycle_unchanged': True,
              'documentation_sha256': {p:sha(ROOT/p) for p in documentation},
              'replay_original_validator': replay_before, 'replay_fixed_validator': replay_after,
              'finding': 'B: diagnostic index incorrectly advanced the shared progress cursor; no proof of backward execution',
              'ordering_guarantee': 'No total or completion-only callback waypoint-order guarantee found in the inspected local v0.5.0 docs.',
              'remaining_progress_gate': 'Non-diagnostic regressions fail closed as contradictory progress evidence, not proven physical reversal.',
              'physical_readiness': 'False-regression defect resolved offline; adjacent-points policy rejection and 120-second timeout remain.'}
    out = ROOT/'nrt_v050/evidence/smartjoint_event_order'
    out.mkdir(parents=True, exist_ok=True)
    (out/'reconstruction.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    with (out/'chronology.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(chronology[0]))
        writer.writeheader()
        writer.writerows(chronology)
    print(json.dumps({'source_session':str(folder),'original_replay':replay_before,'corrected_replay':replay_after,
                      'preserved_session_files':len(preserved['sessions']),'report':str(out/'reconstruction.json')},indent=2))


if __name__ == '__main__':
    main()
