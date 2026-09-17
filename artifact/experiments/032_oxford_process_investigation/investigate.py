"""Read-only source diagnosis and explicitly post-outcome training sensitivity.

Only recorded train/dev observations. No access to confirmation, interpolation,
learned frame alignment, or mutation of original experiments.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent/'031_oxford_mergeable_confirmation'))
import oxford_workflow as wf
ox = wf.original()
base = ox.base

def save(path, obj):
    with path.open('x') as f:
        json.dump(obj, f, indent=2, allow_nan=False)
        f.write('\n')

def matrices(values):
    values = np.asarray(values, dtype=np.float64)
    r,p,y = values[:,3:].T
    cr,sr,cp,sp,cy,sy = np.cos(r),np.sin(r),np.cos(p),np.sin(p),np.cos(y),np.sin(y)
    m = np.tile(np.eye(4), (len(values),1,1))
    m[:,0,:3] = np.stack([cy*cp,cy*sp*sr-sy*cr,cy*sp*cr+sy*sr],axis=1)
    m[:,1,:3] = np.stack([sy*cp,sy*sp*sr+cy*cr,sy*sp*cr-cy*sr],axis=1)
    m[:,2,:3] = np.stack([-sp,cp*sr,cp*cr],axis=1)
    m[:,:3,3] = values[:,:3]
    return m

def to_motor(m):
    # Stable matrix-to-quaternion via symmetric eigenproblem, wxyz output.
    # Only selected recorded anchors are encoded, not interpolated poses.
    R=m[:,:3,:3]; k=np.empty((len(R),4,4))
    k[:,0,0]=R[:,0,0]-R[:,1,1]-R[:,2,2]
    k[:,1,1]=R[:,1,1]-R[:,0,0]-R[:,2,2]
    k[:,2,2]=R[:,2,2]-R[:,0,0]-R[:,1,1]
    k[:,3,3]=R[:,0,0]+R[:,1,1]+R[:,2,2]
    k[:,0,1]=k[:,1,0]=R[:,0,1]+R[:,1,0]
    k[:,0,2]=k[:,2,0]=R[:,0,2]+R[:,2,0]
    k[:,1,2]=k[:,2,1]=R[:,1,2]+R[:,2,1]
    k[:,0,3]=k[:,3,0]=R[:,2,1]-R[:,1,2]
    k[:,1,3]=k[:,3,1]=R[:,0,2]-R[:,2,0]
    k[:,2,3]=k[:,3,2]=R[:,1,0]-R[:,0,1]
    q=np.linalg.eigh(k)[1][:,:,-1][:,[3,0,1,2]]
    q*=np.where(q[:,0:1]<0,-1,1)
    return base.make_motor(torch.from_numpy(q),torch.from_numpy(m[:,:3,3].copy()))

def stats(a):
    a=np.asarray(a)
    return {'n':int(a.size),'mean':float(a.mean()),'min':float(a.min()),
            'p05':float(np.quantile(a,.05)),'median':float(np.median(a)),
            'p95':float(np.quantile(a,.95)),'max':float(a.max())} if a.size else {'n':0}

def wrap(x): return (x+np.pi)%(2*np.pi)-np.pi

def selected_records():
    m=wf.read(wf.MANIFEST)
    records=[r for r in m['records'] if r['split'] in ('train','dev')]
    assert len(records)==48
    return records

def load_record(record):
    assert record['split'] in ('train','dev')
    paths=[wf.DATA/record[k]['path'] for k in ('vo','rtk')]
    for k,p in zip(('vo','rtk'),paths): assert wf.sha(p)==record[k]['sha256']
    v=np.loadtxt(paths[0],delimiter=',',skiprows=1)
    r=np.loadtxt(paths[1],delimiter=',',skiprows=1,usecols=tuple(range(7))+tuple(range(8,14)))
    vt=v[:,0].astype(np.int64); rt=r[:,0].astype(np.int64)
    assert np.all(np.diff(vt)>0) and np.all(np.diff(rt)>0)
    assert np.isfinite(v).all() and np.isfinite(r).all()
    assert np.all(v[:,0]>v[:,1])
    gap=np.r_[False,v[1:,1]!=v[:-1,0]]
    components=np.cumsum(gap)
    transforms=matrices(v[:,2:8]); va=np.empty_like(transforms); current=np.eye(4)
    for i,step in enumerate(transforms):
        current=current@step
        va[i]=current
    values=np.c_[r[:,4:7],r[:,10:13]]
    # Center UTM coordinates before relative arithmetic to avoid subtracting
    # million-meter coordinates in motor products; this is a rigid world gauge.
    values[:,:3]-=values[0,:3]
    ra=matrices(values)
    extr=matrices(np.loadtxt(wf.DATA/'extrinsics/ins.txt')[None,:])[0]
    quarter=matrices(np.array([[0,0,0,0,0,-np.pi/2]]))[0]
    targets={'original':ra@np.linalg.inv(extr),
             'body_quarter':ra@quarter@np.linalg.inv(extr)}
    values[:,5]-=np.pi/2
    targets['yaw_subtract']=matrices(values)@np.linalg.inv(extr)
    grid=np.arange(max(vt[0],rt[0]),min(vt[-1],rt[-1])+1,500000,dtype=np.int64)
    ri=np.searchsorted(rt,grid)
    vi=np.searchsorted(vt,rt[ri]); vi_safe=np.minimum(vi,len(vt)-1)
    valid=(vi<len(vt)) & (vt[vi_safe]-rt[ri]<=100000)
    nearest=np.where(np.abs(vt[np.maximum(vi_safe-1,0)]-rt[ri])<=np.abs(vt[vi_safe]-rt[ri]),np.maximum(vi_safe-1,0),vi_safe)
    spans=[]; start=None
    for i,ok in enumerate(np.r_[valid,False]):
        if ok and start is None: start=i
        if not ok and start is not None:
            if i-start>32: spans.append((start,i))
            start=None
    velocity=r[:,7:10]; moving=np.linalg.norm(velocity[:,:2],axis=1)>5
    angle=np.degrees(wrap(r[:,12]-np.arctan2(velocity[:,1],velocity[:,0])))
    dt=np.diff(rt)/1e6
    # Centered position differences are observed displacement/time, not new labels.
    posvel=np.diff(r[:,4:7],axis=0)/dt[:,None]
    velocity_mid=(velocity[:-1]+velocity[1:])/2
    speed_mask=(np.linalg.norm(velocity_mid[:,:2],axis=1)>5)&(dt<.2)
    convergence=np.degrees(wrap(np.arctan2(posvel[:,1],posvel[:,0])-np.arctan2(velocity_mid[:,1],velocity_mid[:,0])))
    report={'traversal':record['traversal'],'split':record['split'],
       'sha256':{k:record[k]['sha256'] for k in ('vo','rtk')},
       'vo_rows':len(v),'rtk_rows':len(r),'vo_disconnected_edges':int(gap.sum()),
       'vo_missing_elapsed_s':float((v[1:,1]-v[:-1,0]).sum()/1e6),
       'vo_max_gap_s':float((v[1:,1]-v[:-1,0]).max()/1e6),
       'yaw_minus_velocity_heading_deg':stats(angle[moving]),
       'position_minus_velocity_heading_deg':stats(convergence[speed_mask]),
       'position_velocity_speed_ratio':stats(np.linalg.norm(posvel[speed_mask,:2],axis=1)/np.linalg.norm(velocity_mid[speed_mask,:2],axis=1)),
       'ceil_vo_rtk_offset_s':stats((vt[vi_safe[valid]]-rt[ri[valid]])/1e6),
       'nearest_vo_rtk_offset_s':stats((vt[nearest[valid]]-rt[ri[valid]])/1e6)}
    return v,va,targets,ri,vi_safe,nearest,spans,components,report

class DiagnosticData:
    allowed_splits=('train','dev')
    batch=ox.OxfordData.batch
    def __init__(self):
        self.examples={}; self.tensors={}; self.connected={}

def prepare():
    wf.verify_implementation()
    data=DiagnosticData(); pools={}; reports=[]
    saved=wf.read(wf.ORIGINAL/'results_development.json')
    for record in selected_records():
        v,va,targets,ri,vi,near,spans,components,report=load_record(record)
        split=record['split']; name=record['traversal']; lengths={}
        # Preserve the original continuous quaternion lift, not merely the
        # same SE(3) pose. A leafwise neural map is sensitive to sign encoding.
        original=ox.OxfordTraversal(name, wf.DATA/record['vo']['path'],
            wf.DATA/record['rtk']['path'], ox.load_extrinsic(wf.DATA/'extrinsics/ins.txt'), ox.Config())
        assert len(original.runs)==len(spans)
        original_leaves={span:ox.relative_from_absolute(vm[:-1],vm[1:])
                         for span,(vm,_) in zip(spans,original.runs)}
        for length in ox.EVAL_LENGTHS:
            metrics={k:[] for k in ('original','body_quarter','yaw_subtract','nearest_quarter','distance_only','connected_quarter')}
            connection=[]; count=0
            pool=pools.setdefault((split,length),[[],[],[],[],[]])
            for a,b in spans:
                first=np.arange(a,b-length); last=first+length
                p=np.linalg.solve(va[vi[first]],va[vi[last]])
                tt={k:np.linalg.solve(t[ri[first]],t[ri[last]]) for k,t in targets.items()}
                clean=components[vi[first]]==components[vi[last]]
                connection.extend(clean)
                for k in ('original','body_quarter','yaw_subtract'):
                    metrics[k].extend(np.linalg.norm(p[:,:3,3]-tt[k][:,:3,3],axis=1))
                npred=np.linalg.solve(va[near[first]],va[near[last]])
                metrics['nearest_quarter'].extend(np.linalg.norm(npred[:,:3,3]-tt['body_quarter'][:,:3,3],axis=1))
                metrics['connected_quarter'].extend(np.linalg.norm(p[clean,:3,3]-tt['body_quarter'][clean,:3,3],axis=1))
                metrics['distance_only'].extend(np.abs(np.linalg.norm(p[:,:3,3],axis=1)-np.linalg.norm(tt['original'][:,:3,3],axis=1)))
                leaves=original_leaves[(a,b)]
                target=to_motor(tt['body_quarter'])
                # Independent product-to-matrix check on real recorded windows.
                product=base.reduce_states(list(leaves.unfold(0,length,1).permute(0,2,1).unbind(1)),base.motor_product,'left')
                qp,tp=base.decode_motor(product)
                assert np.max(np.linalg.norm(tp.numpy()-p[:,:3,3],axis=1))<1e-7
                pool[0].append(leaves.unfold(0,length,1).permute(0,2,1).contiguous())
                pool[1].append(target)
                pool[2].extend([name]*len(first)); pool[3].append(torch.from_numpy(clean))
                pool[4].append(torch.from_numpy(np.linalg.norm(tt['body_quarter'][:,:3,3],axis=1)>=5))
                count+=len(first)
            lengths[str(length)]={'n':count,'gap_crossing_windows':int(count-np.sum(connection)),
                'raw_translation_m':{k:float(np.mean(vals)) if vals else None for k,vals in metrics.items()}}
            if split=='dev':
                ref=saved['baselines'][str(length)]['raw_motor']['clusters'][name]
                assert count==ref['n']
                delta=abs(lengths[str(length)]['raw_translation_m']['original']-ref['translation_mean_m'])
                assert delta<1e-6,(name,length,delta)
                lengths[str(length)]['original_reproduction_delta_m']=delta
        report['lengths']=lengths; reports.append(report)
        print(name,split,'gaps',report['vo_disconnected_edges'],'L32',lengths['32'],flush=True)
    for key,(leaves,targets,names,clean,moving) in pools.items():
        data.examples[key]=names
        data.tensors[key]=(torch.cat(leaves),torch.cat(targets),torch.cat(moving))
        data.connected[key]=torch.cat(clean)
    data.train_examples=data.examples[('train',8)]
    _,t=base.decode_motor(data.tensors[('train',8)][1])
    data.translation_scale=max(float(t.norm(dim=-1).median()),1e-3)
    wf.verify_implementation()
    assert not wf.STARTED.exists() and not wf.RELEASE.exists()
    seal={'source_sha256':wf.sha(Path(__file__)),'plan_sha256':wf.sha(HERE/'PLAN.md'),
          'original_fingerprint':wf.fingerprint(),'confirmation_accessed':False}
    save(HERE/'source_diagnosis.json',{'provenance':seal,'records':reports})
    path=HERE/'diagnostic_data.pt'
    with path.open('xb') as f: torch.save({'examples':data.examples,'tensors':data.tensors,'connected':data.connected,'scale':data.translation_scale,'provenance':seal},f)
    save(HERE/'cache_integrity.json',{'sha256':wf.sha(path),**seal})

if __name__=='__main__':
    torch.set_num_threads(1)
    prepare()
