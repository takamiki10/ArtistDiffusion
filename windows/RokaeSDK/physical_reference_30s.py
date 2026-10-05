"""Offline-only 30 s reference. Never imports or connects to a robot SDK."""
import bisect
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import struct

PACKAGE = Path(__file__).resolve().parent.parent / 'laptop_handoff/laptop_handoff'
PIN = 'dbaba32746457240385aaf123a91f84164cba3a4986337addbf7a0c3d2a6506f'
SCALE = 3.0

def bits(v):
    return struct.pack('<d', v)

def load_all():
    manifest_bytes = (PACKAGE/'SHA256SUMS.json').read_bytes()
    if hashlib.sha256(manifest_bytes).hexdigest() != PIN:
        raise ValueError('untrusted manifest')
    manifest = json.loads(manifest_bytes)
    for name, digest in manifest.items():
        path = (PACKAGE/name).resolve()
        if not path.is_relative_to(PACKAGE.resolve()):
            raise ValueError('manifest traversal')
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError('changed input: '+name)
    result = []
    for path in ('path_0001','path_0003','path_0006'):
        for condition in ('A','B'):
            name = f'trajectories/{path}/{condition}.csv'
            raw = (PACKAGE/name).read_bytes()
            if hashlib.sha256(raw).hexdigest() != manifest[name]:
                raise ValueError('input changed after verification')
            rows = list(csv.reader(io.StringIO(raw.decode('utf-8-sig'))))
            if rows[0] != ['time_seconds','q1','q2','q3','q4','q5','q6'] or len(rows)!=101:
                raise ValueError('header/row count')
            data=[]
            for i,row in enumerate(rows[1:]):
                if len(row)!=7 or any(x!=x.strip() or not x for x in row):
                    raise ValueError('field count/format')
                values=list(map(float,row))
                if not all(map(math.isfinite,values)):
                    raise ValueError('nonfinite value')
                t=values[0]
                if abs(t-i*10/99)>1e-12 or (i and (t<=data[-1][0] or abs(t-data[-1][0]-10/99)>1e-12)):
                    raise ValueError('timestamp grid')
                data.append(values)
            if data[0][0]!=0 or data[-1][0]!=10:
                raise ValueError('endpoints')
            result.append((name,manifest[name],data))
    return result

def derivative(p):
    return [i*x for i,x in enumerate(p)][1:] or [0.0]

def evaluate(p,u):
    value=0.0
    for c in reversed(p):
        value=value*u+c
    return value

def roots(p):
    p=list(p)
    while len(p)>1 and p[-1]==0:
        p.pop()
    if len(p)<2:
        return []
    if len(p)==2:
        r=-p[0]/p[1]
        return [r] if 0<=r<=1 else []
    critical=sorted([0.0,1.0]+roots(derivative(p)))
    tolerance=1e-12*(1+sum(map(abs,p)))
    answer=[r for r in critical if abs(evaluate(p,r))<=tolerance]
    for lo,hi in zip(critical,critical[1:]):
        fl,fh=evaluate(p,lo),evaluate(p,hi)
        if fl*fh<0:
            for _ in range(64):
                mid=(lo+hi)/2
                fm=evaluate(p,mid)
                if fl*fm>0:
                    lo,fl=mid,fm
                else:
                    hi=mid
            answer.append((lo+hi)/2)
    return sorted(set(answer))

class Reference:
    def __init__(self,data,scale=3.0):
        if not math.isfinite(scale) or scale<=0:
            raise ValueError('invalid scale')
        self.times=[row[0]*scale for row in data]
        self.q=[row[1:] for row in data]
        self.h=[b-a for a,b in zip(self.times,self.times[1:])]
        self.c=[]
        for j in range(6):
            slope=[(self.q[i+1][j]-self.q[i][j])/self.h[i] for i in range(99)]
            v=[0.0]+[(self.h[i]*slope[i-1]+self.h[i-1]*slope[i])/(self.h[i]+self.h[i-1]) for i in range(1,99)]+[0.0]
            a=[0.0]+[2*(slope[i]-slope[i-1])/(self.h[i]+self.h[i-1]) for i in range(1,99)]+[0.0]
            cs=[]
            for i,h in enumerate(self.h):
                c0,c1,c2=self.q[i][j],h*v[i],h*h*a[i]/2
                A=self.q[i+1][j]-c0-c1-c2
                B=h*v[i+1]-c1-2*c2
                C=h*h*a[i+1]-2*c2
                cs.append([c0,c1,c2,10*A-4*B+C/2,-15*A+7*B-C,6*A-3*B+C/2])
            self.c.append(cs)

    def at(self,t):
        if not math.isfinite(t):
            raise ValueError('nonfinite time')
        if t<=self.times[0]:
            return list(self.q[0])
        if t>=self.times[-1]:
            return list(self.q[-1])
        index=bisect.bisect_left(self.times,t)
        if self.times[index]==t:
            return list(self.q[index])
        i=index-1
        return [evaluate(self.c[j][i],(t-self.times[i])/self.h[i]) for j in range(6)]

    def extrema(self,j):
        low,high=math.inf,-math.inf
        maximum=[0.0]*3
        for i,h in enumerate(self.h):
            p=self.c[j][i]
            values=[evaluate(p,u) for u in [0,1]+roots(derivative(p))]
            low,high=min(low,*values),max(high,*values)
            for order in range(1,4):
                p=derivative(p)
                maximum[order-1]=max(maximum[order-1],*(abs(evaluate(p,u))/h**order for u in [0,1]+roots(derivative(p))))
        return [low,high]+maximum

def audit(out):
    data=load_all()
    out=Path(out)
    out.mkdir(exist_ok=False)
    rows=[]
    discrepancy=[0.0]*5
    coefficient_error=0.0
    knot_checks=0
    phase_error=0.0
    oldfile=Path(__file__).resolve().parent/'artifacts/run_20260913T073830/audit/command_audit.csv'
    with oldfile.open() as f:
        old={(r['trajectory'],int(r['joint'])):r for r in csv.DictReader(f)}
    oldkeys=['c2_min_rad','c2_max_rad','c2_max_speed_rad_s','c2_max_acc_rad_s2','c2_max_jerk_rad_s3']
    for name,digest,knots in data:
        base,physical=Reference(knots,1),Reference(knots,3)
        for t,q in zip(physical.times,physical.q):
            for x,y in zip(physical.at(t),q):
                assert bits(x)==bits(y)
                knot_checks+=1
        for k in range(3001):
            phase_error=max(phase_error,*(abs(x-y) for x,y in zip(physical.at(k/100),base.at(k/300))))
        for j in range(6):
            expected=base.extrema(j)
            actual=physical.extrema(j)
            for i in range(99):
                coefficient_error=max(coefficient_error,*(abs(x-y) for x,y in zip(base.c[j][i],physical.c[j][i])))
            for n,factor in enumerate([1,1,3,9,27]):
                target=expected[n]/factor
                discrepancy[n]=max(discrepancy[n],abs(actual[n]-target))
                assert math.isclose(actual[n],target,rel_tol=1e-10,abs_tol=1e-11)
                assert math.isclose(expected[n],float(old[(name,j+1)][oldkeys[n]]),rel_tol=1e-9,abs_tol=1e-9)
            jerk_limit=math.radians(5000)
            speed_limit=math.radians([180,180,234,240,240,240][j])
            acc_limit=math.radians(1500)
            assert actual[0]>=-2*math.pi and actual[1]<=2*math.pi
            assert actual[2]<speed_limit and actual[3]<acc_limit and actual[4]<jerk_limit
            rows.append([name,digest,j+1,*actual,math.degrees(actual[4]),jerk_limit-actual[4],5000-math.degrees(actual[4]),speed_limit-actual[2],acc_limit-actual[3]])
        label=name.replace('trajectories/','').replace('/','_')
        with (out/(label.removesuffix('.csv')+'_physical_knots_30s.csv')).open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['time_seconds','q1','q2','q3','q4','q5','q6'])
            w.writerows([[t,*q] for t,q in zip(physical.times,physical.q)])
    assert phase_error<1e-12 and knot_checks==3600
    with (out/'continuous_demands_30s.csv').open('w',newline='') as f:
        w=csv.writer(f);w.writerow(['trajectory','sha256','joint','minimum_rad','maximum_rad','max_speed_rad_s','max_acc_rad_s2','max_jerk_rad_s3','max_jerk_deg_s3','min_jerk_margin_rad_s3','min_jerk_margin_deg_s3','speed_margin_rad_s','acc_margin_rad_s2']);w.writerows(rows)
    result={'duration_seconds':30.0,'scale':3.0,'package_manifest_sha256':PIN,'files_verified':56,'bit_exact_q_checks':knot_checks,'max_absolute_extrema_scaling_errors':discrepancy,'max_normalized_coefficient_error':coefficient_error,'max_phase_identity_error_rad':phase_error,'worst_jerk_rad_s3':max(r[7] for r in rows),'minimum_jerk_margin_rad_s3':min(r[9] for r in rows),'extrema_method':'numerical derivative roots; analytic uniform-scaling proof in timing amendment; not interval certified','sdk_imported':False}
    (out/'verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result))

if __name__=='__main__':
    import sys
    if len(sys.argv)!=2:
        raise SystemExit('Usage: physical_reference_30s.py NEW_OFFLINE_OUTPUT_DIRECTORY')
    audit(sys.argv[1])
