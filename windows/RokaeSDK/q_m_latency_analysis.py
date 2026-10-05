"""Offline latency analysis of preserved SmartJoint logs. No SDK import."""
from __future__ import annotations
import argparse
from bisect import bisect_left, bisect_right
from collections import defaultdict
import gzip
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def distribution(values):
    """Finite numeric observations; linear-interpolated percentiles, strict thresholds."""
    xs=sorted(values)
    def percentile(p):
        if not xs:return None
        at=(len(xs)-1)*p;lo=int(at);hi=min(lo+1,len(xs)-1)
        return xs[lo]+(xs[hi]-xs[lo])*(at-lo)
    return {'count':len(xs),'minimum':xs[0] if xs else None,'median':percentile(.5),
            'p90':percentile(.90),'p95':percentile(.95),'p99':percentile(.99),
            'maximum':xs[-1] if xs else None,
            **{f'greater_than_{t}_ms':sum(v>t for v in xs) for t in (20,30,40,50)}}


def read_jsonl(path):
    if path.exists():
        with path.open(encoding='utf-8-sig') as f:
            for line in f:
                if line.strip():yield json.loads(line)


def load_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig')) if path.exists() else {}


def scope_for(row,boundary):
    first=boundary.get('observable_window_first_sequence')
    if first is None:first=(boundary.get('first_q_after') or {}).get('sequence')
    if first is not None:
        if row['sequence']==first:return 'start_boundary'
        if row['sequence']>first:return 'post_start'
    return 'prestart'


def summarize_groups(groups):
    return {scope:{metric:distribution(values) for metric,values in metrics.items()}
            for scope,metrics in groups.items()}


def analyze_sessions(sessions,output):
    sessions=Path(sessions).resolve();output=Path(output).resolve()
    if output==sessions or sessions in output.parents:
        raise ValueError('Analysis output must be outside preserved sessions')
    output.mkdir(parents=True,exist_ok=True)
    all_groups=defaultdict(lambda:defaultdict(list));summary=[];stalls=[]
    with gzip.open(output/'all_reads.jsonl.gz','wt',encoding='utf-8') as exported:
        for folder in sorted(p for p in sessions.iterdir() if p.is_dir()):
            final=load_json(folder/'final_status.json');meta=load_json(folder/'session.json')
            boundary=final.get('start_boundary') or {}
            q=list(read_jsonl(folder/'q_m.jsonl'));events=list(read_jsonl(folder/'motion_events.jsonl'))
            events.sort(key=lambda e:e['host_ns']);event_times=[e['host_ns'] for e in events]
            calls=[c for c in read_jsonl(folder/'calls.jsonl') if c.get('phase') in ('return','exception')]
            states=[c for c in calls if c['api']=='operationState'];state_times=[c['return_ns'] for c in states]
            qtimes=[r['decoded_end_ns'] for r in q]
            groups=defaultdict(lambda:defaultdict(list));phases=defaultdict(lambda:defaultdict(list))
            long_read_times=[];previous=None
            def add(scope,phase,metric,value):
                if value is not None:
                    groups[scope][metric].append(value);phases[phase][metric].append(value)
                    all_groups[scope][metric].append(value)
            for row in q:
                start,end,decoded=row['read_start_ns'],row['read_end_ns'],row['decoded_end_ns']
                duration=(end-start)/1e6
                gap=None if previous is None else (decoded-previous['decoded_end_ns'])/1e6
                scope=scope_for(row,boundary);phase=row.get('phase','unknown')
                state_i=bisect_right(state_times,decoded)-1
                latest_state=states[state_i] if state_i>=0 else None
                since=previous['decoded_end_ns'] if previous else start
                ei,ej=bisect_right(event_times,since),bisect_right(event_times,decoded)
                event_count=ej-ei
                nearest=min((abs(t-decoded)/1e6 for t in event_times[max(0,ej-1):ej+1]),default=None)
                delta=None
                if previous and row.get('valid_q') and previous.get('valid_q'):
                    delta=max(abs(a-b) for a,b in zip(row['q_native_rad'],previous['q_native_rad']))
                get_call=next((c for c in row.get('sdk_sample_calls',[]) if c['api']=='getStateData(q_m)'),None)
                rec={'session':folder.name,'trial':final.get('trial',meta.get('trial')),
                    'sequence':row['sequence'],'read_start_ns':start,'read_end_ns':end,
                    'read_duration_ms':duration,'decoded_end_ns':decoded,'gap_ms':gap,
                    'scope':scope,'phase':phase,'valid_q':row.get('valid_q'),
                    'q_native_rad':row.get('q_native_rad'), 'q_change_linf_rad':delta,
                    'last_reported_operation_state':None if latest_state is None else latest_state.get('return_repr'),
                    'operation_state_age_ms':None if latest_state is None else (decoded-latest_state['return_ns'])/1e6,
                    'latest_validated_source_index':row.get('last_validated_event_source_index'),
                    'controller_timestamp':row.get('controller_timestamp'),
                    'host_before_read_ms':None if previous is None else (start-previous['decoded_end_ns'])/1e6,
                    'read_end_to_decoded_ms':(decoded-end)/1e6,
                    'getStateData_duration_ms':None if get_call is None else get_call['duration_ns']/1e6,
                    'events_in_gap':event_count,'nearest_event_to_decode_ms':nearest,
                    'timeout_no_sample':False}
                exported.write(json.dumps(rec,allow_nan=False)+'\n')
                for metric,value in [('sample_read_ms',duration),('all_update_calls_ms',duration),('gap_ms',gap)]:add(scope,phase,metric,value)
                if duration>20 or (gap is not None and gap>50):
                    status_calls=[{'api':c['api'],'duration_ms':c['duration_ns']/1e6}
                                  for c in calls if c['request_ns']<decoded and c['return_ns']>since]
                    stalls.append({**rec,'gap_minus_read_ms':None if gap is None else gap-duration,
                                   'overlapping_checked_calls':status_calls})
                if duration>20:long_read_times.append(decoded)
                previous=row
            timeout_count=0
            for r in read_jsonl(folder/'lifecycle.jsonl'):
                if r.get('code')!='timeout_no_sample':continue
                timeout_count+=1;start,end=r['read_start_ns'],r['read_end_ns']
                i=bisect_right(qtimes,start)-1;prev=q[i] if i>=0 else None
                scope=scope_for(prev,boundary) if prev else 'prestart'
                phase=prev.get('phase','unknown') if prev else 'unknown'
                duration=(end-start)/1e6;add(scope,phase,'all_update_calls_ms',duration)
                exported.write(json.dumps({'session':folder.name,'read_start_ns':start,'read_end_ns':end,
                    'read_duration_ms':duration,'decoded_end_ns':None,'gap_ms':None,'scope':scope,
                    'phase':phase,'timeout_no_sample':True})+'\n')
            spans=[(b-a)/1e9 for a,b in zip(long_read_times,long_read_times[1:])]
            summary.append({'session':folder.name,'trial':final.get('trial',meta.get('trial')),
                'status':final.get('status','NO_FINAL_STATUS'),'failure_reason':final.get('failure_reason'),
                'q_records':len(q),'timeout_reads':timeout_count,'by_scope':summarize_groups(groups),
                'by_phase':summarize_groups(phases),'read_gt20_intervals_s':spans,
                'post_start_elapsed_s':None if not q or not boundary.get('return_ns') else (q[-1]['decoded_end_ns']-boundary['return_ns'])/1e9,
                'metadata':{k:meta.get(k) for k in ('settings','sdk_hashes','source_csv_sha256','runner_sha256','shared_runner_sha256','path_runner_sha256','observation_policy')},
                'event_count':len(events),'q_schema_keys':sorted(q[0]) if q else []})
    result={'percentile_method':'linear interpolation at (n-1)*p; units ms; thresholds strictly greater',
            'scopes':'post_start excludes the separately reviewed first START-boundary sample; prestart includes intentional preparation gaps',
            'coverage':'all logged sample-producing reads plus timeout_no_sample records; SDK exceptions without timing records cannot be reconstructed',
            'groups':summarize_groups(all_groups),'sessions':summary,'long_intervals':stalls}
    (output/'analysis.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sessions',type=Path,default=ROOT/'nrt_v050/smartjoint_sessions')
    p.add_argument('--output',type=Path,default=ROOT/'nrt_v050/evidence/q_m_receive_latency')
    args=p.parse_args(argv);r=analyze_sessions(args.sessions,args.output)
    print(f"Analyzed {len(r['sessions'])} sessions; {sum(s['q_records'] for s in r['sessions'])} q records.")
    for s in r['sessions']:
        g=s['by_scope'].get('post_start',{})
        print(s['session'],s['status'],'post-start reads:',g.get('sample_read_ms',{}).get('count',0),
              'max read/gap ms:',g.get('sample_read_ms',{}).get('maximum'),g.get('gap_ms',{}).get('maximum'))


if __name__=='__main__':main()
