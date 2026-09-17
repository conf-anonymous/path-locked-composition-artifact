"""Original INS vs RTK export diagnostic: recorded development rows only."""
from __future__ import annotations
import collections
import csv
import io
import json
import shutil
import sys
import tarfile
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import investigate as a

HERE=Path(__file__).resolve().parent
NAME='2014-11-21-16-07-03'
EXPECTED_MD5='d7455bfe51b17a8b42b0e8b0f40ae228'
RAW=a.wf.DATA/'ins_crosscheck'/NAME

def nearest(times, queries):
    right=np.searchsorted(times,queries).clip(0,len(times)-1)
    left=np.maximum(right-1,0)
    return np.where(abs(times[left]-queries)<=abs(times[right]-queries),left,right)

def rotation(rpy):return a.matrices(np.c_[np.zeros((len(rpy),3)),rpy])[:,:3,:3]

def angle(matrix):
    # atan2 gives stable small-angle differences; no learned or fitted alignment.
    v=np.stack((matrix[:,2,1]-matrix[:,1,2],matrix[:,0,2]-matrix[:,2,0],matrix[:,1,0]-matrix[:,0,1]),axis=1)
    return np.degrees(np.arctan2(np.linalg.norm(v,axis=1),np.trace(matrix,axis1=1,axis2=2)-1))

def summarize(ins,rtk,indices,mask):
    ii=ins[indices[mask]];rr=rtk[mask]
    ip=ii[:,7:10];rp=rr[:,7:10]
    out={'n':len(ii),'angle_correlations':np.corrcoef(np.c_[ip[:,:2],rp[:,:2]].T).tolist(),
         'angle_correlation_order':['ins_roll','ins_pitch','rtk_roll','rtk_pitch'],
         'roll_sum_deg':a.stats(np.degrees(ip[:,0]+rp[:,0])),
         'pitch_difference_deg':a.stats(np.degrees(rp[:,1]-ip[:,1])),
         'yaw_difference_deg':a.stats(np.degrees(a.wrap(rp[:,2]-ip[:,2]))),
         'position_difference_m':a.stats(np.linalg.norm(rr[:,1:4]-ii[:,1:4],axis=1))}
    for name,data in [('ins',ii),('rtk',rr)]:
        moving=np.linalg.norm(data[:,4:6],axis=1)>5
        h=np.arctan2(data[:,5],data[:,4])
        out[name+'_yaw_minus_velocity_heading_deg']=a.stats(np.degrees(a.wrap(data[moving,9]-h[moving])))
    reference=rotation(ip);q=rotation(np.array([[0,0,-np.pi/2]]))[0]
    modes={'sdk_unchanged':rotation(rp),'body_quarter_turn':rotation(rp)@q}
    yaw=rp.copy();yaw[:,2]-=np.pi/2
    modes['yaw_subtract']=rotation(yaw)
    flip=yaw.copy();flip[:,0]*=-1
    modes['yaw_subtract_roll_negate']=rotation(flip)
    out['diagnostic_attitude_difference_to_ins_deg']={k:a.stats(angle(reference.transpose(0,2,1)@v)) for k,v in modes.items()}
    return out

def main():
    torch.set_num_threads(1);a.wf.verify_implementation()
    assert not a.wf.STARTED.exists() and not a.wf.RELEASE.exists()
    record=next(r for r in a.selected_records() if r['traversal']==NAME)
    assert record['split']=='dev'
    source=Path(sys.argv[1]) if len(sys.argv)>1 else RAW/(NAME+'_gps.tar')
    md5=a.hashlib.md5(source.read_bytes()).hexdigest();assert md5==EXPECTED_MD5
    RAW.mkdir(parents=True,exist_ok=True)
    archive=RAW/(NAME+'_gps.tar')
    if not archive.exists():shutil.copyfile(source,archive)
    assert a.wf.sha(archive)==a.wf.sha(source)
    with tarfile.open(archive) as tar:
        allowed={f'{NAME}/gps/ins.csv',f'{NAME}/gps/gps.csv'}
        assert {m.name for m in tar.getmembers() if m.isfile()}==allowed
        for name in ('ins.csv','gps.csv'):
            member=tar.getmember(f'{NAME}/gps/{name}');assert member.isfile()
            blob=tar.extractfile(member).read();path=RAW/name
            if not path.exists():
                with path.open('xb') as f:f.write(blob)
            assert path.read_bytes()==blob
    with (RAW/'ins.csv').open(newline='') as f:
        reader=csv.DictReader(f);rows=list(reader);header=reader.fieldnames
    required=['timestamp','northing','easting','down','velocity_north','velocity_east','velocity_down','roll','pitch','yaw']
    it=np.array([int(r['timestamp']) for r in rows],dtype=np.int64)
    ins=np.array([[float(r[k]) for k in required] for r in rows])
    states=np.array([r['ins_status'] for r in rows])
    rpath=a.wf.DATA/record['rtk']['path'];assert a.wf.sha(rpath)==record['rtk']['sha256']
    with rpath.open(newline='') as f:rtkrows=list(csv.DictReader(f))
    rt=np.array([int(r['timestamp']) for r in rtkrows],dtype=np.int64)
    rtk=np.array([[float(r[k]) for k in required] for r in rtkrows])
    all_ins_rows=len(ins);all_status_counts=dict(collections.Counter(states.tolist()))
    disorder=np.flatnonzero(np.diff(it)<=0)
    disorder_report=[{'first_csv_line':int(j+2),'first_timestamp':int(it[j]),
                     'second_timestamp':int(it[j+1])} for j in disorder]
    # Full INS archive has out-of-order timestamps outside RTK coverage. Keep
    # those observations unchanged, report them, and search only the common
    # time interval. Never silently sort, de-duplicate, or interpolate.
    overlap=(it>=rt[0]-20000)&(it<=rt[-1]+20000)
    assert all(not overlap[j] and not overlap[j+1] for j in disorder)
    ins=ins[overlap];states=states[overlap];it=it[overlap]
    assert np.all(np.diff(it)>0) and np.all(np.diff(rt)>0)
    assert np.isfinite(ins).all() and np.isfinite(rtk).all()
    index=nearest(it,rt);mask=abs(it[index]-rt)<=20000
    good=mask&(states[index]=='INS_SOLUTION_GOOD')
    midpoint=(int(rt[mask][0])+int(rt[mask][-1]))//2
    groups={'all_paired':mask,'ins_solution_good':good,
            'good_first_time_half':good&(rt<=midpoint),'good_second_time_half':good&(rt>midpoint)}
    report={'traversal':NAME,'split':'dev','diagnostic_only':True,'training_performed':False,
      'confirmation_accessed':False,'shares_physical_sensor':True,'no_offset_fitting':True,
      'archive_md5':md5,'archive_sha256':a.wf.sha(archive),
      'csv_sha256':{n:a.wf.sha(RAW/n) for n in ('ins.csv','gps.csv')},
      'rtk_sha256':record['rtk']['sha256'],'ins_header':header,'ins_rows':all_ins_rows,'rtk_rows':len(rtk),
      'ins_rows_in_rtk_interval':len(ins),'ins_status_counts':all_status_counts,
      'out_of_order_ins_edges_outside_rtk_interval':disorder_report,
      'paired_status_counts':dict(collections.Counter(states[index[mask]].tolist())),
      'ins_minus_rtk_timestamp_ms':a.stats((it[index[mask]]-rt[mask])/1000),
      'subsets':{k:summarize(ins,rtk,index,m) for k,m in groups.items()},
      'original_fingerprint':a.wf.fingerprint(),'source_sha256':a.wf.sha(Path(__file__)),
      'plan_sha256':a.wf.sha(HERE/'PLAN.md')}
    assert report['subsets']['all_paired']['angle_correlations'][0][2]<-.99
    assert report['subsets']['all_paired']['angle_correlations'][1][3]>.99
    a.wf.verify_implementation();assert not a.wf.STARTED.exists() and not a.wf.RELEASE.exists()
    a.save(HERE/'results.json',report)
    for k,v in report['subsets'].items():
        print(k,'n',v['n'],'roll sum',v['roll_sum_deg']['median'],'pitch diff',v['pitch_difference_deg']['median'],'yaw diff',v['yaw_difference_deg']['median'])

if __name__=='__main__':main()
