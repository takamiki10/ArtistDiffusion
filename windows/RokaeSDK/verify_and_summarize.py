"""Offline corroboration only. Reads immutable inputs, writes inside a new audit run."""
import csv
import hashlib
import json
import math
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent / 'laptop_handoff/laptop_handoff'
PIN = 'dbaba32746457240385aaf123a91f84164cba3a4986337addbf7a0c3d2a6506f'
run = Path(sys.argv[1]).resolve()
assert run.is_relative_to(HERE / 'artifacts')
manifest_bytes = (PACKAGE / 'SHA256SUMS.json').read_bytes()
assert hashlib.sha256(manifest_bytes).hexdigest() == PIN
manifest = json.loads(manifest_bytes)
for name, digest in manifest.items():
    assert hashlib.sha256((PACKAGE / name).read_bytes()).hexdigest() == digest, name

def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

audit = rows(run / 'audit/command_audit.csv')
checks = 0
max_reference_error = 0.0
for path in ('path_0001', 'path_0003', 'path_0006'):
    for condition in ('A', 'B'):
        name = f'trajectories/{path}/{condition}.csv'
        data = rows(PACKAGE / name)
        ts = [float(r['time_seconds']) for r in data]
        q = [[float(r[f'q{j}']) for j in range(1, 7)] for r in data]
        grid = rows(run / f'audit/{path}_{condition}_reference_1ms.csv')
        assert len(grid) == 10001
        h = [b-a for a, b in zip(ts, ts[1:])]
        coefficients = []
        for j in range(6):
            s = [(q[i+1][j]-q[i][j])/h[i] for i in range(99)]
            v = [0.0]+[(h[i]*s[i-1]+h[i-1]*s[i])/(h[i-1]+h[i]) for i in range(1, 99)]+[0.0]
            acc = [0.0]+[2*(s[i]-s[i-1])/(h[i-1]+h[i]) for i in range(1, 99)]+[0.0]
            cs = []
            for i in range(99):
                c0, c1, c2 = q[i][j], h[i]*v[i], h[i]**2*acc[i]/2
                a, b, c = q[i+1][j]-c0-c1-c2, h[i]*v[i+1]-c1-2*c2, h[i]**2*acc[i+1]-2*c2
                cs.append((c0,c1,c2,10*a-4*b+c/2,-15*a+7*b-c,6*a-3*b+c/2))
            coefficients.append(cs)
            rec = next(r for r in audit if r['trajectory'] == name and int(r['joint']) == j+1)
            assert math.isclose(float(rec['pl_max_speed_rad_s']), max(map(abs,s)), abs_tol=1e-13)
            assert math.isclose(float(rec['interior_fd_acc_rad_s2']), max(map(abs,acc)), abs_tol=1e-12)
            step = max(abs(float(grid[i][f'q{j+1}'])-float(grid[i-1][f'q{j+1}'])) for i in range(1,10001))
            assert step == float(rec['rt_max_step_rad'])
            checks += 3
        interval = 0
        for k, row in enumerate(grid):
            t = k/1000
            assert float(row['nominal_time_seconds']) == t
            while interval < 98 and t >= ts[interval+1]:
                interval += 1
            u = (t-ts[interval])/h[interval]
            for j in range(6):
                expected = sum(c*u**power for power,c in enumerate(coefficients[j][interval]))
                if k in (0,10000):
                    expected = q[0 if k == 0 else 99][j]
                error = abs(expected-float(row[f'q{j+1}']))
                max_reference_error = max(error,max_reference_error)
                assert error <= 5e-14
                checks += 1

raw = (run / 'synthetic/SYNTHETIC_ONLY.sdk_decoded_raw.bin').read_bytes()
converted = rows(run / 'synthetic/SYNTHETIC_ONLY_measured_joints.csv')
assert len(raw) == 32*445 and len(converted) == 32
for i in range(32):
    frame = raw[i*445:(i+1)*445]
    assert frame[:8] == b'XCRAW001'
    assert struct.unpack_from('<I',frame,8)[0] == 369
    payload = frame[12:381]
    assert hashlib.sha256(payload).hexdigest().encode() == frame[381:]
    assert struct.unpack_from('<Q',payload)[0] == i
    native = struct.unpack_from('<6d',payload,76)
    for j,v in enumerate(native,1):
        assert struct.pack('<d',v) == struct.pack('<d',float(converted[i][f'q{j}_measured']))
        checks += 1
    assert converted[i]['controller_timestamp'] == ''

fields = ['pl_max_speed_rad_s','interior_fd_acc_rad_s2','c2_max_speed_rad_s','c2_max_acc_rad_s2','c2_max_jerk_rad_s3','rt_max_step_rad']
with (run/'audit/rd_demand_differences.csv').open('w',newline='',encoding='utf-8') as f:
    w = csv.writer(f)
    w.writerow(['path','joint']+['D_minus_R_'+x for x in fields])
    for path in ('path_0001','path_0003','path_0006'):
        for j in range(1,7):
            a,b = [next(r for r in audit if r['trajectory']==f'trajectories/{path}/{c}.csv' and int(r['joint'])==j) for c in ('A','B')]
            w.writerow([path,j]+[float(b[x])-float(a[x]) for x in fields])

lines = ['# Offline demand audit','', 'A = R; B = D. Values below are prospective host references, not measured motion. Full precision is retained in command_audit.csv and rd_demand_differences.csv beside this file.', '',
         '| Path/condition | Joint | PL max v (rad/s) | Interior FD a (rad/s²) | C2 max v (rad/s) | C2 max a (rad/s²) | C2 max jerk (rad/s³) | 1 ms max step (rad) |',
         '|---|---:|---:|---:|---:|---:|---:|---:|']
for r in audit:
    label = r['trajectory'].replace('trajectories/','').replace('.csv','').replace('/A','/R').replace('/B','/D')
    lines.append('| '+label+' | '+r['joint']+' | '+' | '.join(f'{float(r[f]):.9g}' for f in fields)+' |')
lines += ['', 'Static comparison: cached JOINT_MAX_SPEED = [180,180,234,240,240,240] deg/s; JOINT_MAX_ACC = 1500 deg/s² on all joints (26.1799388 rad/s²). All candidate speeds and accelerations are below these cached values. They are not verified current RT limits.', '',
          'Cached JOINT_MAX_JERK = 5000 deg/s³ (87.2664626 rad/s³), with JERK_LIMIT_JOINT=0. Every candidate exceeds that numeric jerk value on joints 1–5. Do not infer enforcement or RT applicability from this flag; current settings/vendor interpretation are required. If 5000 deg/s³ is an active hard RT constraint, all six candidates as defined fail it. Units are documented in xCore V3.0_B §15.4.16.22–23, PDF p.311 / printed p.297. ER3/ER7 limits in the SDK manual are not CR7 limits.', '',
          'C2 path_0006 joint 1 reaches 3.1425140906387883 rad for both R and D: about 0.000914091 rad above the Jetson bound 3.1416, and 0.000921437 rad above π. This is between-knot overshoot; no input knot changed. The cached ±360° bounds contain it, but current limits and the prospective modeling convention remain unresolved. Do not clip it or declare hardware ineligibility from the Jetson bound alone.', '',
          'PL finite differences describe neighboring slopes; true PL acceleration has impulses at slope changes. C2 extrema use numerical derivative-root searches, not formal interval certification. No torque, payload, collision, tracking, filter or current safety feasibility is established.']
(run/'audit/demand_audit_summary.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
result = {'package_files_reverified':len(manifest),'independent_numeric_checks':checks,'maximum_independent_reference_error_rad':max_reference_error,'reference_tolerance_rad':5e-14,'raw_binary64_csv_mismatches':0,'scope':'OFFLINE_ONLY'}
(run/'independent_verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result))
