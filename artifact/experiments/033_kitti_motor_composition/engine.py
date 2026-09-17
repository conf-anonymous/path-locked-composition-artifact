"""Recorded KITTI windows and original PGA models; no generated observations."""
import importlib.util
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

import acquire as a
import full_frontend as frontend

spec=importlib.util.spec_from_file_location('kitti_original_models',a.HERE.parent/'029_eth3d_mergeable_motor_gate/run.py')
original=importlib.util.module_from_spec(spec)
sys.modules[spec.name]=original
spec.loader.exec_module(original)
base=original.v.base
LeafCalibrator=original.v.v1.study.LeafCalibrator
Head=original.study.Head
MergeableGate=original.MergeableGate
loss=original.v.v2.robust_motor_loss
quaternion_from_matrix=original.v.v1.quaternion_from_matrix

LENGTHS=(32,64,128,256)
AUDIT_LENGTHS=LENGTHS+(512,)
HEADS=('constant','contextual','feature_direct','feature_residual','mergeable')
MODULES=('calibrated',)+HEADS
ARMS=('identity','raw')+MODULES
SEEDS=tuple(range(5))
RUNS=a.RAW/'learned_composition_v1'


def configure():
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


def encode(matrices):
    q=quaternion_from_matrix(matrices)
    # Canonicalize recorded leaves/targets once, never an intermediate product.
    for row in q:
        first=np.flatnonzero(row)
        if row[first[0]]<0:
            row*=-1
    return base.make_motor(torch.from_numpy(q),torch.from_numpy(matrices[:,:3,3].copy()))


def matrices(motors):
    q,t=base.decode_motor(motors)
    w,x,y,z=q.unbind(-1)
    rotation=torch.stack((1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),
        2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),
        2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)),-1).reshape(q.shape[:-1]+(3,3))
    out=torch.eye(4,dtype=motors.dtype).expand(q.shape[:-1]+(4,4)).clone()
    out[...,:3,:3]=rotation;out[...,:3,3]=t
    return out


def checked(motor):
    if not bool(torch.isfinite(motor).all()):
        raise FloatingPointError('Nonfinite motor; no repair or exclusion.')
    minimum=float(torch.linalg.vector_norm(motor[...,:4],dim=-1).detach().min())
    if minimum<=1e-6:
        raise FloatingPointError(f'Projection domain failure: real norm {minimum}')
    return minimum


def product(leaves,path='balanced',rng=None):
    length=leaves.shape[1]
    if not length:
        raise ValueError('Empty motion window')
    if path=='balanced' and length & (length-1)==0:
        # Same midpoint tree as the original reducer for power-of-two lengths;
        # vectorize its independent nodes, without changing algebra or grouping.
        value=leaves
        while value.shape[1]>1:
            value=base.motor_product(value[:,::2],value[:,1::2])
        return value[:,0]
    return base.reduce_states(list(leaves.unbind(1)),base.motor_product,path,rng)


def calibrated_leaves(calibrator,leaves):
    before=leaves+calibrator.network(leaves)
    minimum=checked(before)
    return base.normalize_motor(before),minimum


def make_module(arm,seed,dtype=torch.float32):
    if arm not in MODULES or seed not in SEEDS:
        raise ValueError((arm,seed))
    torch.manual_seed(seed if arm=='calibrated' else 50000+seed)
    model=LeafCalibrator() if arm=='calibrated' else MergeableGate() if arm=='mergeable' else Head(arm)
    return model.to(dtype=dtype)


def forward(arm,leaves,calibrator=None,head=None,cached=None):
    minimum=checked(leaves)
    raw=product(leaves) if cached is None else cached[0]
    minimum=min(minimum,checked(raw))
    if arm=='identity':
        result=torch.zeros_like(raw);result[:,0]=1
        return result,minimum
    if arm=='raw':
        return raw,minimum
    if cached is None:
        cal_leaves,norm=calibrated_leaves(calibrator,leaves)
        cal=product(cal_leaves);minimum=min(minimum,norm)
    else:
        cal=cached[1]
    minimum=min(minimum,checked(cal))
    if arm=='calibrated':
        return cal,minimum
    before=head.before_projection(leaves,raw,cal)
    minimum=min(minimum,checked(before))
    return base.normalize_motor(before),minimum


class Sequence:
    def __init__(self,sequence):
        # The authorization check MUST precede paths, metadata or numeric reads.
        import workflow
        workflow.require_access(sequence)
        folder=frontend.OUTPUT if sequence in frontend.TRAIN else RUNS/'frontend'
        self.receipt=json.loads((folder/f'{sequence}.json').read_text())
        if sequence not in frontend.TRAIN:
            workflow.validate_hashes(self.receipt['dependencies'])
            if self.receipt['implementation_seal_sha256']!=a.sha(RUNS/'implementation_seal.json'):
                raise RuntimeError('Frontend belongs to another implementation')
        frame_path=folder/f'{sequence}.frames.jsonl'
        if a.sha(frame_path)!=self.receipt['frame_records_sha256']:
            raise RuntimeError('Frontend frame checksum changed')
        rows=[json.loads(line) for line in frame_path.read_text().splitlines()]
        _,self.component,_=frontend.predicted_components(rows)
        self.name=sequence;self.n=len(rows)
        self.truth=np.tile(np.eye(4),(self.n,1,1))
        self.truth[:,:3]=np.loadtxt(a.RAW/'dataset/poses'/f'{sequence}.txt').reshape(-1,3,4)
        self.times=np.loadtxt(a.RAW/'dataset/sequences'/sequence/'times.txt')
        if len(self.times)!=self.n or not np.isfinite(self.truth).all() or not np.all(np.diff(self.times)>0):
            raise ValueError('Invalid recording dimensions/values')
        valid=[i for i,r in enumerate(rows[1:]) if r['success']]
        transforms=np.tile(np.eye(4),(len(valid),1,1))
        transforms[:,:3]=np.array([rows[i+1]['previous_to_current'] for i in valid]).reshape(-1,3,4)
        transforms=np.linalg.inv(transforms)
        # Missing edges remain NaN sentinels, never identity or synthetic motion.
        self.leaves=torch.full((self.n-1,8),float('nan'),dtype=torch.float64)
        self.leaves[valid]=encode(transforms)
        self.starts={length:np.flatnonzero(self.component[:-length]==self.component[length:])
                     if length<self.n else np.array([],dtype=int) for length in AUDIT_LENGTHS}
        self.targets={}
        if sequence in frontend.TRAIN:
            for length in LENGTHS:
                # Reference-only label cache; never exposed to model features.
                self.targets[length]=encode(np.linalg.solve(self.truth[:-length],self.truth[length:])).float()

    def window(self,start,length,dtype=torch.float32):
        end=start+length
        if not (0<=start<end<self.n) or self.component[start]!=self.component[end]:
            raise ValueError('Out-of-recording or gap-crossing window')
        target=self.targets[length][start] if length in self.targets and dtype==torch.float32 else encode(np.linalg.solve(self.truth[start:start+1],self.truth[end:end+1]))[0]
        return self.leaves[start:end].to(dtype),target.to(dtype)

    def evaluation_windows(self):
        for segment in self.receipt['validation']['segments']:
            yield dict(segment,kind='distance',group=str(segment['length_m']))
        for length in AUDIT_LENGTHS:
            # "Every 10th valid start": subsample the eligible-start inventory,
            # not an independently reset grid in each connected component.
            for start in self.starts[length][::10]:
                yield dict(first=int(start),last=int(start+length),length_m=None,
                           valid=True,kind='fixed',group=str(length))


class TrainingData:
    def __init__(self):
        self.sequences={name:Sequence(name) for name in frontend.TRAIN}
        self.inventory={length:[name for name,s in self.sequences.items() if len(s.starts[length])]
                        for length in LENGTHS}
        displacement=[]
        for s in self.sequences.values():
            starts=s.starts[128]
            displacement.extend(np.linalg.norm(s.truth[starts+128,:3,3]-s.truth[starts,:3,3],axis=1).tolist())
        self.scale=max(.001,float(np.median(displacement)))

    def receipt(self):
        return dict(translation_scale_m=self.scale,scale_windows=sum(len(s.starts[128]) for s in self.sequences.values()),
            inventory={name:{str(length):s.starts[length].tolist() for length in LENGTHS}
                       for name,s in self.sequences.items()},
            encoded_leaves_sha256={name:__import__('hashlib').sha256(s.leaves.numpy().tobytes()).hexdigest()
                                   for name,s in self.sequences.items()},source='Recorded training frontend and reference only')

    def sample(self,generator,batch_size=64):
        length=LENGTHS[int(torch.randint(len(LENGTHS),(1,),generator=generator))]
        names=self.inventory[length];leaves=[];targets=[];indices=[]
        for _ in range(batch_size):
            name=names[int(torch.randint(len(names),(1,),generator=generator))]
            seq=self.sequences[name]
            start=int(seq.starts[length][int(torch.randint(len(seq.starts[length]),(1,),generator=generator))])
            x,y=seq.window(start,length)
            leaves.append(x);targets.append(y);indices.append((name,start,length))
        return torch.stack(leaves),torch.stack(targets),indices


def errors(prediction,target_matrices):
    predicted=matrices(prediction.double()).detach().numpy()
    relative=np.linalg.solve(predicted,target_matrices)
    translation=np.linalg.norm(relative[:,:3,3],axis=-1)
    angle=np.arccos(np.clip((np.trace(relative[:,:3,:3],axis1=1,axis2=2)-1)/2,-1,1))
    return translation,np.degrees(angle)


def discrepancy(a_motor,b_motor):
    aq,at=base.decode_motor(a_motor);bq,bt=base.decode_motor(b_motor)
    relative=base.qmul(aq,base.qconj(bq))
    angle=2*torch.atan2(torch.linalg.vector_norm(relative[...,1:],dim=-1),relative[...,0].abs())
    return float(torch.linalg.vector_norm(at-bt,dim=-1).max()),float(angle.max())


@torch.no_grad()
def audit_window(leaves,calibrator,head):
    leaves=leaves.double();calibrator=calibrator.double();head=head.double()
    cal,min_norm=calibrated_leaves(calibrator,leaves)
    def read(raw,corrected):
        nonlocal min_norm
        before=head.before_projection(leaves,raw,corrected)
        min_norm=min(min_norm,checked(raw),checked(corrected),checked(before))
        return (raw,corrected,base.normalize_motor(before))
    baseline=read(product(leaves),product(cal))
    max_t=max_r=max_coeff=0.
    schedules=[('left',None),('right',None),('balanced',None)]+[('random',330911+k) for k in range(16)]+[('chunks',None)]
    for path,seed in schedules:
        def reduce(x):
            if path=='chunks':
                length=x.shape[1];cuts=(0,length//3,2*length//3,length)
                parts=[product(x[:,cuts[i]:cuts[i+1]]) for i in range(3)]
                return base.motor_product(base.motor_product(parts[0],parts[1]),parts[2])
            return product(x,path,random.Random(seed) if seed is not None else None)
        actual=read(reduce(leaves),reduce(cal))
        for expected,observed in zip(baseline,actual):
            t,r=discrepancy(expected,observed)
            max_t=max(max_t,t);max_r=max(max_r,r)
            max_coeff=max(max_coeff,float((expected-observed).abs().max()))
    # Independent coordinate realization of EXACTLY the same calibrated leaves.
    transforms=matrices(cal).numpy()
    matrix_product=np.tile(np.eye(4),(leaves.shape[0],1,1))
    for i in range(leaves.shape[1]):
        matrix_product=matrix_product@transforms[:,i]
    coordinate_error=float(np.max(abs(matrix_product-matrices(baseline[1]).numpy())))
    return dict(max_translation_m=max_t,max_rotation_rad=max_r,max_state_coefficient=max_coeff,
        min_preprojection_real_norm=min_norm,max_matrix_coordinate_error=coordinate_error,
        passed=max_t<1e-8 and max_r<1e-9 and coordinate_error<1e-8 and min_norm>1e-6,
        schedules=len(schedules),arms=['raw','calibrated','mergeable'])
