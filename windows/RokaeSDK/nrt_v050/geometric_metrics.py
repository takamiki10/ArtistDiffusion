"""Reference geometric analysis; no SDK, alignment, raw-data rewriting or smoothing."""
import math
from pathlib import Path
import hashlib
import xml.etree.ElementTree as ET

URDF_SHA='921e6ae2402ccbf8112ece1b88776c22c841d6e6a04519c3e6b903d2c1bd427a'

def identity(): return [[float(i==j) for j in range(4)] for i in range(4)]
def multiply(a,b): return [[sum(a[i][k]*b[k][j] for k in range(4)) for j in range(4)] for i in range(4)]
def rotation(axis,angle):
    norm=math.sqrt(sum(v*v for v in axis))
    if not norm: raise ValueError('zero axis')
    x,y,z=[v/norm for v in axis];c=math.cos(angle);s=math.sin(angle);v=1-c
    return [[c+x*x*v,x*y*v-z*s,x*z*v+y*s,0],
            [y*x*v+z*s,c+y*y*v,y*z*v-x*s,0],
            [z*x*v-y*s,z*y*v+x*s,c+z*z*v,0],[0,0,0,1]]

class Link6FK:
    def __init__(self,path):
        path=Path(path)
        if hashlib.sha256(path.read_bytes()).hexdigest()!=URDF_SHA:
            raise ValueError('authoritative package URDF hash mismatch')
        root=ET.parse(path).getroot()
        by_child={j.find('child').get('link'):j for j in root.findall('joint')}
        chain=[];node='xMateCR7_link6'
        while node!='xMateCR7_base':
            j=by_child[node];chain.append(j);node=j.find('parent').get('link')
        self.chain=list(reversed(chain))
        if [j.get('name') for j in self.chain]!=[f'joint{i}' for i in range(1,7)]:
            raise ValueError('unexpected kinematic chain')
    def position(self,q):
        if len(q)!=6 or not all(map(math.isfinite,q)): raise ValueError('six finite joint radians required')
        transform=identity()
        for joint,value in zip(self.chain,q):
            origin=joint.find('origin')
            xyz=list(map(float,origin.get('xyz','0 0 0').split()))
            r,p,y=map(float,origin.get('rpy','0 0 0').split())
            fixed=multiply(multiply(rotation([0,0,1],y),rotation([0,1,0],p)),rotation([1,0,0],r))
            for i in range(3):fixed[i][3]=xyz[i]
            axis=list(map(float,joint.find('axis').get('xyz').split()))
            transform=multiply(transform,multiply(fixed,rotation(axis,value)))
        return tuple(transform[i][3] for i in range(3))

def validate(curve):
    if len(curve)<2 or any(len(p)!=3 or not all(map(math.isfinite,p)) for p in curve):
        raise ValueError('finite 3D polyline required')

def point_segment_distance(p,a,b):
    v=[y-x for x,y in zip(a,b)];den=sum(x*x for x in v)
    t=0 if den==0 else max(0.0,min(1.0,sum((x-y)*z for x,y,z in zip(p,a,v))/den))
    return math.sqrt(sum((x-(y+t*z))**2 for x,y,z in zip(p,a,v)))

def nearest(p,curve):
    return min(point_segment_distance(p,a,b) for a,b in zip(curve,curve[1:]))

def directional(source,target,spacing_m):
    validate(source);validate(target)
    if not math.isfinite(spacing_m) or spacing_m<=0:raise ValueError('positive spacing required')
    weighted=[];length=0.0
    maximum=max(nearest(p,target) for p in source)
    for a,b in zip(source,source[1:]):
        segment=math.dist(a,b)
        if not segment:continue
        n=max(1,math.ceil(segment/spacing_m));weight=segment/n;length+=segment
        for i in range(n):
            p=[x+(y-x)*(i+0.5)/n for x,y in zip(a,b)]
            distance=nearest(p,target);weighted.append((distance,weight));maximum=max(maximum,distance)
    if length==0:raise ValueError('zero-length source cannot define arc-length metric')
    mean_square=sum(d*d*w for d,w in weighted)/length
    return {'rms_m':math.sqrt(mean_square),'weighted_distances':weighted,'length_m':length,
            'sampled_max_m':maximum,'polyline_max_upper_bound_m':maximum+spacing_m/2}

def compare(target,realized,spacing_m=0.0001):
    """Arc-length weighting eliminates dwell/speed weighting, not acquisition gaps."""
    tr=directional(target,realized,spacing_m);rt=directional(realized,target,spacing_m)
    mix=sorted([(d,0.5*w/x['length_m']) for x in (tr,rt) for d,w in x['weighted_distances']])
    accum=0.0;p95=0.0
    for d,w in mix:
        accum+=w;p95=d
        if accum>=0.95:break
    return {'target_to_realized_rms_m':tr['rms_m'],'realized_to_target_rms_m':rt['rms_m'],
            'symmetric_rms_m':math.sqrt((tr['rms_m']**2+rt['rms_m']**2)/2),
            'symmetric_arclength_p95_m':p95,'sampled_symmetric_max_m':max(tr['sampled_max_m'],rt['sampled_max_m']),
            'symmetric_polyline_max_upper_bound_m':max(tr['polyline_max_upper_bound_m'],rt['polyline_max_upper_bound_m']),
            'quadrature_spacing_m':spacing_m,'rigid_alignment_applied':False,
            'label':'link6 trajectory reconstructed from measured joint states'}
