"""Guarded extraction of evaluation recordings; never official KITTI test data."""
import argparse
import hashlib
import io
import json
import os
import struct
import subprocess
import time
import zipfile

import numpy as np
from PIL import Image

import acquire as a
import engine as e
import full_frontend as full
import workflow as w
from build_frontend import FRONT
from frontend_smoke import save


def validate_recording(sequence,rows):
    """Same native formula and fixed engineering thresholds as full training."""
    w.require_access(sequence)
    predicted,component,max_ortho=full.predicted_components(rows)
    truth=np.loadtxt(a.RAW/'dataset/poses'/f'{sequence}.txt').reshape(-1,3,4)
    times=np.loadtxt(a.RAW/'dataset/sequences'/sequence/'times.txt')
    if len(truth)!=len(rows) or len(times)!=len(rows) or not np.all(np.diff(times)>0) or not np.isfinite(truth).all():
        raise ValueError('Invalid recording counts/times/poses')
    rotation=truth[:,:3,:3]
    if np.max(abs(rotation.transpose(0,2,1)@rotation-np.eye(3)))>=1e-4 or np.max(abs(np.linalg.det(rotation)-1))>=1e-4:
        raise ValueError('Reference is not a rounded rigid transform')
    lines=[str(len(rows))]+[str(component[i])+' '+' '.join(format(float(v),'.17g')
        for v in np.r_[truth[i].ravel(),predicted[i,:3].ravel()]) for i in range(len(rows))]
    binary=full.OUTPUT/'continuous_metrics'
    if a.sha(binary)!=w.json_read(full.OUTPUT/'metric_build.json')['binary_sha256']:
        raise RuntimeError('Native metric binary changed')
    output=subprocess.run([str(binary)],input='\n'.join(lines)+'\n',text=True,capture_output=True,check=True)
    segments=[]
    for line in output.stdout.splitlines():
        fields=line.split();first,last,length=map(int,fields[1:4])
        if fields[0] not in ('S','F'):
            raise ValueError('Unknown native metric status')
        item=dict(first=first,last=last,length_m=length,valid=fields[0]=='S')
        if item['valid']!=bool(component[first]==component[last]):
            raise RuntimeError('Native continuity disagreement')
        if item['valid']:
            t,r,ti,ri=map(float,fields[4:])
            if not np.isfinite([t,r,ti,ri]).all():
                raise FloatingPointError('Nonfinite native metric')
            item.update(raw_translation_percent=100*t,raw_rotation_deg_per_m=float(np.degrees(r)),
                        identity_translation_percent=100*ti,identity_rotation_deg_per_m=float(np.degrees(ri)))
        segments.append(item)
    by_length={}
    for length in range(100,801,100):
        candidates=[s for s in segments if s['length_m']==length];valid=[s for s in candidates if s['valid']]
        summary=dict(potential=len(candidates),valid=len(valid),coverage=len(valid)/len(candidates) if candidates else None)
        for key in ('raw_translation_percent','raw_rotation_deg_per_m','identity_translation_percent','identity_rotation_deg_per_m'):
            summary[key]=float(np.mean([s[key] for s in valid])) if valid else None
        by_length[str(length)]=summary
    p=by_length['100'];valid_edges=sum(r['success'] for r in rows[1:])
    gates=dict(valid_motion_fraction=valid_edges/(len(rows)-1)>=.95,
        primary_coverage=p['coverage'] is not None and p['coverage']>=.8,primary_exists=p['valid']>0,
        raw_translation=p['raw_translation_percent'] is not None and p['raw_translation_percent']<10,
        raw_rotation=p['raw_rotation_deg_per_m'] is not None and p['raw_rotation_deg_per_m']<.2,
        raw_beats_identity=p['valid']>0 and p['raw_translation_percent']<p['identity_translation_percent'])
    return dict(by_length=by_length,segments=segments,component_count=int(component[-1]+1),
        max_rotation_orthogonality_error=max_ortho,valid_edges=valid_edges,total_edges=len(rows)-1,
        valid_edge_fraction=valid_edges/(len(rows)-1),failed_edges=[r['frame'] for r in rows[1:] if not r['success']],
        fixed_length_training_window_counts={str(length):int(sum(component[i]==component[i+length]
            for i in range(len(rows)-length))) for length in e.AUDIT_LENGTHS},gates=gates,admitted=all(gates.values()),
        metric_scope='Native KITTI formula on continuous valid windows; full sequence start grid, no gap bridging')


def extract(sequence):
    w.require_access(sequence)
    if sequence in full.TRAIN:
        raise ValueError('Training extraction already immutable; use its existing receipts')
    folder=e.RUNS/'frontend';folder.mkdir(parents=True,exist_ok=True)
    final=folder/f'{sequence}.json';stream=folder/f'{sequence}.frames.jsonl'
    if final.exists():
        r=w.json_read(final)
        w.validate_hashes(r['dependencies'])
        if a.sha(stream)!=r['frame_records_sha256']:
            raise RuntimeError('Evaluation frontend changed')
        return
    if stream.exists():
        raise RuntimeError('Partial extraction retained; no automatic overwrite')
    acquisition=w.json_read(a.RAW/'acquisition.json')
    archives={r['file']:r for r in acquisition['archives']}
    for r in archives.values():
        path=a.Path(r['source_path'])
        if (path.stat().st_size,path.stat().st_mtime_ns)!=(r['bytes'],r['mtime_ns']):
            raise RuntimeError('Original archive changed')
    staged={}
    for filename,member in (('data_odometry_calib.zip',f'dataset/sequences/{sequence}/calib.txt'),
                            ('data_odometry_calib.zip',f'dataset/sequences/{sequence}/times.txt'),
                            ('data_odometry_poses.zip',f'dataset/poses/{sequence}.txt')):
        with zipfile.ZipFile(archives[filename]['source_path']) as archive:
            staged[str((a.RAW/member).relative_to(a.ROOT))]=a.preserve(a.RAW/member,archive.read(member))
    calibration={}
    for line in (a.RAW/f'dataset/sequences/{sequence}/calib.txt').read_text().splitlines():
        key,value=line.split(':',1);calibration[key]=np.fromstring(value,sep=' ').reshape(3,4)
    p0,p1=calibration['P0'],calibration['P1']
    np.testing.assert_array_equal(p0[:,:3],p1[:,:3])
    baseline=p0[0,3]/p0[0,0]-p1[0,3]/p1[0,0]
    if p0[0,0]!=p0[1,1] or p0[0,1]!=0 or p1[1,3]!=0 or baseline<=0:
        raise ValueError('Unsupported stereo calibration')
    parameters=[float(p0[0,0]),float(p0[0,2]),float(p0[1,2]),float(baseline)]
    if not np.isfinite(parameters).all() or parameters[0]<=0:
        raise ValueError('Nonfinite or invalid camera parameters')
    command=['/usr/bin/arch','-x86_64',str(FRONT/'stereo_stream_x86_64')]+[repr(v) for v in parameters]
    image_archive=archives['data_odometry_gray.zip'];counts=image_archive['images'][sequence]
    n=counts['0']['count']
    if n!=counts['1']['count']:
        raise ValueError('Unpaired stereo inventory')
    rows=[];started=time.perf_counter()
    process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    try:
        with zipfile.ZipFile(image_archive['source_path']) as archive,stream.open('x') as output:
            for frame in range(n):
                arrays=[];hashes=[]
                for camera in (0,1):
                    member=f'dataset/sequences/{sequence}/image_{camera}/{frame:06}.png';blob=archive.read(member)
                    with Image.open(io.BytesIO(blob)) as image:
                        if image.mode!='L' or image.format!='PNG':
                            raise ValueError('Not original grayscale PNG')
                        pixels=np.array(image)
                    arrays.append(pixels)
                    hashes.append(dict(member=member,png_sha256=hashlib.sha256(blob).hexdigest(),
                                       decoded_sha256=hashlib.sha256(pixels.tobytes()).hexdigest()))
                if arrays[0].shape!=arrays[1].shape or arrays[0].dtype!=np.uint8:
                    raise ValueError('Stereo pixel mismatch')
                height,width=arrays[0].shape;process.stdin.write(struct.pack('<II',width,height))
                for pixels in arrays:process.stdin.write(pixels.tobytes())
                process.stdin.flush();fields=process.stdout.readline().decode().split()
                if not fields or int(fields[0])!=frame or fields[1] not in ('0','1'):
                    raise RuntimeError('Frontend response failed')
                success=fields[1]=='1'
                if len(fields)!=(17 if success else 5):
                    raise ValueError('Malformed frontend response')
                row=dict(frame=frame,width=width,height=height,success=success,matches=int(fields[2]),
                    inliers=int(fields[3]),process_seconds=float(fields[4]),image_hashes=hashes,
                    previous_to_current=list(map(float,fields[5:])) if success else None)
                rows.append(row);output.write(json.dumps(row,allow_nan=False)+'\n')
                if frame%250==0:output.flush();print(sequence,frame,'/',n-1,flush=True)
            output.flush();os.fsync(output.fileno())
        process.stdin.close()
        if process.wait(timeout=30)!=0:
            raise RuntimeError(process.stderr.read().decode())
    finally:
        if process.poll() is None:
            process.terminate()
            try:process.wait(timeout=5)
            except subprocess.TimeoutExpired:process.kill();process.wait()
    validation=validate_recording(sequence,rows)
    w.require_access(sequence)
    save(final,dict(sequence=sequence,role=a.ROLES[sequence],frames=n,calibration=parameters,
        validation=validation,frame_records_sha256=a.sha(stream),dependencies=staged,
        implementation_seal_sha256=a.sha(e.RUNS/'implementation_seal.json'),
        image_archive_sha256=image_archive['sha256'],command=command,seconds=time.perf_counter()-started))
    print('Frontend complete',sequence,'admitted=',validation['admitted'],flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('split',choices=['dev','study_holdout'])
    args=parser.parse_args()
    for sequence,role in a.ROLES.items():
        if role==args.split:extract(sequence)
