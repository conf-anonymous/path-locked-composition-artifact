"""Fixed sequential TRAIN-only frontend extraction; no dev/holdout mode exists."""
import hashlib
import io
import json
import os
from pathlib import Path
import resource
import struct
import subprocess
import time
import zipfile
import numpy as np
from PIL import Image, __version__ as PIL_VERSION
import acquire as a
import frontend_smoke as smoke
from build_frontend import FRONT

OUTPUT=FRONT/'full_train'
TRAIN=tuple(f'{i:02}' for i in range(7))
LOCAL_SOURCES=('EXPERIMENT_PROTOCOL.md','PREPARATION_PROTOCOL.md','FRONTEND_SMOKE_PROTOCOL.md',
    'acquire.py','build_frontend.py','stereo_stream.cpp','frontend_smoke.py',
    'full_frontend.py','continuous_metrics.cpp')
MODEL_DEPENDENCIES=('017_tum_motor_composition/run.py','020_oxford_robotcar_vo/run.py',
    '021_eth3d_rgbd_motor/run.py','022_eth3d_adaptive_motor/run.py',
    '023_eth3d_family_heldout/run.py','028_eth3d_attribution_controls/run.py',
    '029_eth3d_mergeable_motor_gate/run.py')


def require_train(sequence):
    if sequence not in TRAIN or a.ROLES[sequence]!='train':
        raise ValueError('Only fixed training sequences 00–06 are admitted.')


def fingerprint():
    sources={name:a.sha(a.HERE/name) for name in LOCAL_SOURCES}
    dependencies={name:a.sha(a.HERE.parent/name) for name in MODEL_DEPENDENCIES}
    return dict(local_sources=sources,model_dependencies=dependencies,
        acquisition_sha256=a.sha(a.RAW/'acquisition.json'),
        frontend_build_sha256=a.sha(FRONT/'build.json'),
        frontend_binary_sha256=a.sha(FRONT/'stereo_stream_x86_64'),
        roles=a.ROLES,training_implementation_complete=False,
        study_holdout_release_implemented=False)


def verify_seal():
    seal=json.loads((OUTPUT/'specification_seal.json').read_text())
    if seal['fingerprint']!=fingerprint():
        raise RuntimeError('Source/specification changed after full-training seal.')
    smoke.check_inputs()
    return seal


def build_metrics():
    binary=OUTPUT/'continuous_metrics'
    source=a.RAW/'devkit/cpp'
    args=['clang++','-O2','-std=c++11','-I',str(source),
          str(a.HERE/'continuous_metrics.cpp'),str(source/'matrix.cpp'),'-o',str(binary)]
    compiled=subprocess.run(args,capture_output=True,text=True,check=True)
    smoke.save(OUTPUT/'metric_build.json',dict(command=args,compiler_output=compiled.stdout+compiled.stderr,
        binary_sha256=a.sha(binary),source_sha256=a.sha(a.HERE/'continuous_metrics.cpp'),
        original_sources_sha256={name:a.sha(source/name) for name in ('evaluate_odometry.cpp','matrix.cpp','matrix.h')}))


def calibration_for(sequence):
    require_train(sequence)
    calibration={}
    for line in (a.RAW/'dataset/sequences'/sequence/'calib.txt').read_text().splitlines():
        key,value=line.split(':',1)
        calibration[key]=np.fromstring(value,sep=' ').reshape(3,4)
    p0,p1=calibration['P0'],calibration['P1']
    np.testing.assert_array_equal(p0[:,:3],p1[:,:3])
    assert p0[0,0]==p0[1,1] and p0[0,1]==0 and p1[1,3]==0
    baseline=p0[0,3]/p0[0,0]-p1[0,3]/p1[0,0]
    assert baseline>0
    return [float(p0[0,0]),float(p0[0,2]),float(p0[1,2]),float(baseline)]


def pixels_for(z,sequence,frame):
    require_train(sequence)
    arrays=[]; hashes=[]
    for camera in (0,1):
        member=f'dataset/sequences/{sequence}/image_{camera}/{frame:06}.png'
        blob=z.read(member)
        with Image.open(io.BytesIO(blob)) as image:
            assert image.mode=='L' and image.format=='PNG'
            pixels=np.array(image)
        assert pixels.ndim==2 and pixels.dtype==np.uint8
        arrays.append(pixels)
        hashes.append(dict(member=member,png_sha256=hashlib.sha256(blob).hexdigest(),
            decoded_sha256=hashlib.sha256(pixels.tobytes()).hexdigest()))
    assert arrays[0].shape==arrays[1].shape
    return arrays,hashes


def predicted_components(rows):
    n=len(rows); predicted=np.tile(np.eye(4),(n,1,1))
    component=np.zeros(n,dtype=np.int32)
    max_orthogonality=0.;current=0
    assert not rows[0]['success'] and rows[0]['previous_to_current'] is None
    for i,row in enumerate(rows[1:],1):
        if not row['success']:
            assert row['previous_to_current'] is None
            current+=1
            # Identity here defines a NEW component's coordinate origin; no
            # identity motion is inserted and crossing windows are forbidden.
        else:
            motion=np.eye(4);motion[:3]=np.array(row['previous_to_current']).reshape(3,4)
            assert np.isfinite(motion).all()
            rotation=motion[:3,:3]
            error=float(np.max(abs(rotation.T@rotation-np.eye(3))))
            max_orthogonality=max(max_orthogonality,error)
            assert error<1e-10 and abs(np.linalg.det(rotation)-1)<1e-10
            predicted[i]=predicted[i-1]@np.linalg.inv(motion)
        component[i]=current
    return predicted,component,max_orthogonality


def metrics_for(sequence,rows):
    require_train(sequence)
    predicted,component,max_ortho=predicted_components(rows)
    values=np.loadtxt(a.RAW/'dataset/poses'/f'{sequence}.txt').reshape(-1,3,4)
    assert len(values)==len(rows)
    lines=[str(len(rows))]
    for i in range(len(rows)):
        lines.append(str(component[i])+' '+' '.join(format(float(v),'.17g')
            for v in np.r_[values[i].ravel(),predicted[i,:3].ravel()]))
    binary=OUTPUT/'continuous_metrics'
    build=json.loads((OUTPUT/'metric_build.json').read_text())
    assert a.sha(binary)==build['binary_sha256']
    run=subprocess.run([str(binary)],input='\n'.join(lines)+'\n',capture_output=True,text=True,check=True)
    segments=[]
    for line in run.stdout.splitlines():
        fields=line.split();assert fields[0] in ('S','F')
        first,last,length=map(int,fields[1:4])
        assert 0<=first<last<len(rows)
        item=dict(first=first,last=last,length_m=length,valid=fields[0]=='S')
        assert item['valid']==bool(component[first]==component[last])
        if item['valid']:
            t,r,ti,ri=map(float,fields[4:]);assert np.isfinite([t,r,ti,ri]).all()
            item.update(raw_translation_percent=100*t,raw_rotation_deg_per_m=float(np.degrees(r)),
                        identity_translation_percent=100*ti,identity_rotation_deg_per_m=float(np.degrees(ri)))
        segments.append(item)
    by_length={}
    for length in range(100,801,100):
        all_rows=[s for s in segments if s['length_m']==length]
        valid=[s for s in all_rows if s['valid']]
        summary=dict(potential=len(all_rows),valid=len(valid),
                     coverage=len(valid)/len(all_rows) if all_rows else None)
        for key in ('raw_translation_percent','raw_rotation_deg_per_m',
                    'identity_translation_percent','identity_rotation_deg_per_m'):
            summary[key]=float(np.mean([s[key] for s in valid])) if valid else None
        by_length[str(length)]=summary
    primary=by_length['100'];valid_edges=sum(r['success'] for r in rows[1:])
    gates=dict(valid_motion_fraction=valid_edges/(len(rows)-1)>=.95,
        primary_coverage=primary['coverage'] is not None and primary['coverage']>=.8,
        primary_exists=primary['valid']>0,
        raw_translation=primary['raw_translation_percent'] is not None and primary['raw_translation_percent']<10,
        raw_rotation=primary['raw_rotation_deg_per_m'] is not None and primary['raw_rotation_deg_per_m']<.2,
        raw_beats_identity=primary['valid']>0 and primary['raw_translation_percent']<primary['identity_translation_percent'])
    fixed_counts={str(length):int(sum(component[i]==component[i+length]
        for i in range(len(rows)-length))) for length in (32,64,128,256,512)}
    return dict(by_length=by_length,segments=segments,component_count=int(component[-1]+1),
        max_rotation_orthogonality_error=max_ortho,valid_edges=valid_edges,
        total_edges=len(rows)-1,valid_edge_fraction=valid_edges/(len(rows)-1),
        failed_edges=[r['frame'] for r in rows[1:] if not r['success']],
        fixed_length_training_window_counts=fixed_counts,gates=gates,admitted=all(gates.values()),
        metric_scope='Native KITTI formula on continuous valid windows; full sequence start grid, no gap bridging')


def extract_one(sequence,path,archive):
    require_train(sequence);verify_seal()
    final=OUTPUT/f'{sequence}.json';stream=OUTPUT/f'{sequence}.frames.jsonl'
    if final.exists():
        result=json.loads(final.read_text())
        assert result['frame_records_sha256']==a.sha(stream)
        assert result['seal_sha256']==a.sha(OUTPUT/'specification_seal.json')
        print(sequence,'already complete; immutable record verified',flush=True)
        return result
    assert not stream.exists(), 'Partial attempt retained; do not overwrite it.'
    images=archive['images'][sequence]
    n=images['0']['count'];assert n==images['1']['count']
    calibration=calibration_for(sequence)
    command=['/usr/bin/arch','-x86_64',str(FRONT/'stereo_stream_x86_64')]+[repr(x) for x in calibration]
    before=resource.getrusage(resource.RUSAGE_CHILDREN);started=time.monotonic()
    process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    rows=[]
    try:
        with zipfile.ZipFile(path) as z,stream.open('x') as f:
            for frame in range(n):
                arrays,hashes=pixels_for(z,sequence,frame)
                height,width=arrays[0].shape
                process.stdin.write(struct.pack('<II',width,height))
                for pixels in arrays:process.stdin.write(pixels.tobytes())
                process.stdin.flush()
                line=process.stdout.readline().decode()
                if not line:raise RuntimeError('Frontend stopped: '+process.stderr.read().decode())
                fields=line.split();assert int(fields[0])==frame
                success=bool(int(fields[1]));assert len(fields)==(17 if success else 5)
                row=dict(frame=frame,width=width,height=height,success=success,
                    matches=int(fields[2]),inliers=int(fields[3]),process_seconds=float(fields[4]),
                    image_hashes=hashes,previous_to_current=list(map(float,fields[5:])) if success else None)
                f.write(json.dumps(row,allow_nan=False)+'\n')
                rows.append(row)
                if success:assert 0<=row['inliers']<=row['matches']
                if frame%250==0:
                    f.flush()
                    print(f'{sequence}: {frame}/{n-1}, failures={sum(not r["success"] for r in rows[1:])}',flush=True)
            f.flush();os.fsync(f.fileno())
        process.stdin.close();code=process.wait(timeout=30)
        stderr=process.stderr.read().decode();assert code==0,stderr
    finally:
        if process.poll() is None:
            process.terminate()
            try:process.wait(timeout=5)
            except subprocess.TimeoutExpired:process.kill();process.wait()
    after=resource.getrusage(resource.RUSAGE_CHILDREN)
    frontend_seconds=time.monotonic()-started
    validation=metrics_for(sequence,rows)
    result=dict(sequence=sequence,role='train',frames=n,calibration=calibration,
        frontend_wall_seconds=frontend_seconds,
        child_cpu_seconds=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,
        child_high_water_rss_bytes=int(after.ru_maxrss),
        memory_scope='macOS RUSAGE_CHILDREN high-water across completed children, including metric compilation; not isolated per-sequence peak',
        validation=validation,frame_records_sha256=a.sha(stream),
        seal_sha256=a.sha(OUTPUT/'specification_seal.json'),image_archive_sha256=archive['sha256'],
        command=command,pillow_version=PIL_VERSION,numpy_version=np.__version__,
        development_accessed=False,heldout_accessed=False,ga_training_performed=False)
    verify_seal();smoke.save(final,result)
    print(sequence,'COMPLETE',json.dumps(validation['by_length']['100']),
          'admitted=',validation['admitted'],flush=True)
    return result


def main():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    seal=OUTPUT/'specification_seal.json'
    if not seal.exists():
        smoke.save(seal,dict(created_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
            fingerprint=fingerprint()))
    verify_seal()
    if not (OUTPUT/'metric_build.json').exists():build_metrics()
    path,archive,_=smoke.check_inputs()
    results=[extract_one(sequence,path,archive) for sequence in TRAIN]
    destination=OUTPUT/'completed.json'
    if not destination.exists():
        smoke.save(destination,dict(training_sequences=list(TRAIN),
            records_sha256={f'{r["sequence"]}.json':a.sha(OUTPUT/f'{r["sequence"]}.json') for r in results},
            specification_seal_sha256=a.sha(seal),
            all_training_frontend_gates_pass=all(r['validation']['admitted'] for r in results),
            ga_training_performed=False,development_accessed=False,heldout_accessed=False,
            training_runner_implemented=False,holdout_release_implemented=False))
    print('All training extraction complete; frontend admission:',
          all(r['validation']['admitted'] for r in results),flush=True)


if __name__=='__main__':
    main()
