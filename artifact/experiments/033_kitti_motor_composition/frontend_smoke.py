"""Fixed 401-pair KITTI training smoke, original frontend, no learned models."""
import hashlib
import io
import json
from pathlib import Path
import resource
import struct
import subprocess
import time
import zipfile
import numpy as np
from PIL import Image, __version__ as PIL_VERSION
import acquire as a
from build_frontend import FRONT

SEQUENCE='00'
FRAMES=401


def save(path,report):
    with path.open('x') as f:
        json.dump(report,f,indent=2,allow_nan=False);f.write('\n')


def check_inputs():
    build=json.loads((FRONT/'build.json').read_text())
    assert build['returncode']==0
    assert build['binary_sha256']==a.sha(FRONT/'stereo_stream_x86_64')
    assert build['wrapper_sha256']==a.sha(a.HERE/'stereo_stream.cpp')
    assert build['protocol_sha256']==a.sha(a.HERE/'FRONTEND_SMOKE_PROTOCOL.md')
    for name,digest in build['upstream_sha256'].items():
        assert a.sha(FRONT/'upstream'/name)==digest
    acquisition=json.loads((a.RAW/'acquisition.json').read_text())
    assert a.ROLES[SEQUENCE]=='train'
    for name,digest in acquisition['staged_sha256'].items():
        assert a.sha(a.RAW/name)==digest
    archive=next(r for r in acquisition['archives'] if r['file']=='data_odometry_gray.zip')
    path=Path(archive['source_path'])
    assert path.stat().st_size==archive['bytes'] and path.stat().st_mtime_ns==archive['mtime_ns']
    calibration={}
    for line in (a.RAW/'dataset/sequences'/SEQUENCE/'calib.txt').read_text().splitlines():
        key,value=line.split(':',1)
        calibration[key]=np.fromstring(value,sep=' ').reshape(3,4)
    p0,p1=calibration['P0'],calibration['P1']
    np.testing.assert_array_equal(p0[:,:3],p1[:,:3])
    assert p0[0,0]==p0[1,1] and p0[0,1]==0 and p1[1,3]==0
    baseline=p0[0,3]/p0[0,0]-p1[0,3]/p1[0,0]
    assert baseline>0
    return path,archive,[float(p0[0,0]),float(p0[0,2]),float(p0[1,2]),float(baseline)]


def decoded_pair(z,frame):
    arrays=[]; hashes=[]
    for camera in (0,1):
        member=f'dataset/sequences/{SEQUENCE}/image_{camera}/{frame:06}.png'
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


def rotation_angle(r):
    return float(np.arccos(np.clip((np.trace(r)-1)/2,-1,1)))


def diagnostics(rows):
    values=np.loadtxt(a.RAW/'dataset/poses'/f'{SEQUENCE}.txt').reshape(-1,3,4)[:FRAMES]
    gt=np.tile(np.eye(4),(FRAMES,1,1));gt[:,:3]=values
    estimated=np.tile(np.eye(4),(FRAMES,1,1))
    component=np.zeros(FRAMES,dtype=int)
    origin=0; final=[]; per_frame=[]
    for i,row in enumerate(rows):
        if i==0:
            assert not row['success'], 'first frame must only prime the tracker'
            continue
        if not row['success']:
            if i-1>origin:
                final.append((origin,i-1))
            origin=i
            component[i:]+=1
            continue
        motion=np.eye(4);motion[:3]=np.array(row['previous_to_current']).reshape(3,4)
        assert np.isfinite(motion).all()
        rotation=motion[:3,:3]
        np.testing.assert_allclose(rotation.T@rotation,np.eye(3),atol=1e-10,rtol=0)
        assert abs(np.linalg.det(rotation)-1)<1e-10
        np.testing.assert_allclose(motion@np.linalg.inv(motion),np.eye(4),atol=1e-10,rtol=0)
        estimated[i]=estimated[i-1]@np.linalg.inv(motion)
        reference=np.linalg.solve(gt[origin],gt[i])
        error=np.linalg.solve(estimated[i],reference)
        per_frame.append(dict(frame=i,component=int(component[i]),
            relative_translation_error_m=float(np.linalg.norm(error[:3,3])),
            relative_rotation_error_deg=float(np.degrees(rotation_angle(error[:3,:3])))))
    if FRAMES-1>origin:
        final.append((origin,FRAMES-1))
    components=[]
    for start,end in final:
        reference=np.linalg.solve(gt[start],gt[end])
        error=np.linalg.solve(estimated[end],reference)
        components.append(dict(first_frame=start,last_frame=end,
            reference_path_length_m=float(np.linalg.norm(np.diff(gt[start:end+1,:3,3],axis=0),axis=1).sum()),
            reference_displacement_m=float(np.linalg.norm(reference[:3,3])),
            estimated_displacement_m=float(np.linalg.norm(estimated[end,:3,3])),
            endpoint_translation_error_m=float(np.linalg.norm(error[:3,3])),
            endpoint_rotation_error_deg=float(np.degrees(rotation_angle(error[:3,:3])))))
    # Same distance selection and error convention as the native-validated
    # reference. Float64 error computation; results are smoke-only, not a server
    # submission. Never bridge failed edges.
    delta=np.diff(gt[:,:3,3],axis=0).astype(np.float32)
    steps=np.sqrt((delta[:,0]*delta[:,0]+delta[:,1]*delta[:,1])+delta[:,2]*delta[:,2])
    distance=np.r_[np.float32(0),np.cumsum(steps,dtype=np.float32)]
    segments=[]; excluded=0
    for first in range(0,FRAMES,10):
        for length in range(100,801,100):
            last=int(np.searchsorted(distance,np.float32(distance[first]+np.float32(length)),side='right'))
            if last>=FRAMES:continue
            if component[first]!=component[last]:
                excluded+=1;continue
            true_delta=np.linalg.solve(gt[first],gt[last])
            pred_delta=np.linalg.solve(estimated[first],estimated[last])
            error=np.linalg.solve(pred_delta,true_delta)
            segments.append(dict(first=first,last=last,length_m=length,
                translation_percent=float(100*np.linalg.norm(error[:3,3])/length),
                rotation_deg_per_m=float(np.degrees(rotation_angle(error[:3,:3]))/length)))
    return dict(components=components,per_frame=per_frame,segments=segments,
        segments_excluded_for_failed_edges=excluded,
        segment_translation_percent_mean=float(np.mean([s['translation_percent'] for s in segments])) if segments else None,
        segment_rotation_deg_per_m_mean=float(np.mean([s['rotation_deg_per_m'] for s in segments])) if segments else None)


def run_one(number,path,archive,calibration):
    output=FRONT/f'smoke_run{number}.json'
    assert not output.exists()
    command=['/usr/bin/arch','-x86_64',str(FRONT/'stereo_stream_x86_64')]+[repr(x) for x in calibration]
    before=resource.getrusage(resource.RUSAGE_CHILDREN)
    started=time.monotonic()
    process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    rows=[]
    try:
        with zipfile.ZipFile(path) as z:
            for frame in range(FRAMES):
                arrays,hashes=decoded_pair(z,frame)
                height,width=arrays[0].shape
                process.stdin.write(struct.pack('<II',width,height))
                for pixels in arrays:
                    process.stdin.write(pixels.tobytes())
                process.stdin.flush()
                line=process.stdout.readline().decode()
                if not line:
                    raise RuntimeError('Frontend stopped: '+process.stderr.read().decode())
                fields=line.split()
                assert int(fields[0])==frame
                success=bool(int(fields[1]))
                assert len(fields)==(17 if success else 5)
                row=dict(frame=frame,width=width,height=height,success=success,
                    matches=int(fields[2]),inliers=int(fields[3]),
                    process_seconds=float(fields[4]),image_hashes=hashes,
                    previous_to_current=list(map(float,fields[5:])) if success else None)
                if success:
                    assert 0<=row['inliers']<=row['matches']
                rows.append(row)
                if frame%100==0:
                    print(f'run {number}: {frame}/{FRAMES-1}, success={success}, matches={row["matches"]}',flush=True)
        process.stdin.close()
        code=process.wait(timeout=30)
        stderr=process.stderr.read().decode()
        assert code==0,stderr
    finally:
        if process.poll() is None:
            process.terminate()
            try:process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill();process.wait()
    after=resource.getrusage(resource.RUSAGE_CHILDREN)
    report=dict(sequence=SEQUENCE,role='train',frames=FRAMES,rows=rows,
        frontend_calibration=calibration,command=command,wall_seconds=time.monotonic()-started,
        child_cpu_seconds=(after.ru_utime+after.ru_stime)-(before.ru_utime+before.ru_stime),
        child_high_water_rss_bytes=int(after.ru_maxrss),
        memory_note='macOS RUSAGE_CHILDREN maximum RSS, high-water across completed children, not summed memory or native ARM speed',
        first_frame_is_initialization=True,failed_edges=[r['frame'] for r in rows[1:] if not r['success']],
        diagnostics=diagnostics(rows),source_sha256=a.sha(Path(__file__)),
        build_sha256=a.sha(FRONT/'build.json'),protocol_sha256=a.sha(a.HERE/'FRONTEND_SMOKE_PROTOCOL.md'),
        image_archive_sha256=archive['sha256'],pillow_version=PIL_VERSION,numpy_version=np.__version__,
        learned_model_training=False,holdout_accessed=False,submission_evidence=False,stderr=stderr)
    save(output,report)
    return report


def main():
    path,archive,calibration=check_inputs()
    first=run_one(1,path,archive,calibration)
    second=run_one(2,path,archive,calibration)
    semantic=lambda r:{k:v for k,v in r.items() if k!='process_seconds'}
    identical=all(semantic(x)==semantic(y) for x,y in zip(first['rows'],second['rows']))
    summary=dict(source_sha256=a.sha(Path(__file__)),
        run_sha256={f'smoke_run{i}.json':a.sha(FRONT/f'smoke_run{i}.json') for i in (1,2)},
        semantic_repeatability_exact=identical,ga_training_performed=False,
        heldout_accessed=False,submission_evidence=False)
    save(FRONT/'smoke_completed.json',summary)
    print('Semantic replay exact:',identical,flush=True)
    print('Tracking failures:',first['failed_edges'],flush=True)
    print('Connected component diagnostics:',first['diagnostics']['components'],flush=True)
    print('Segment translation mean (%):',first['diagnostics']['segment_translation_percent_mean'],flush=True)
    print('Wall seconds / child peak MiB:',first['wall_seconds'],first['child_high_water_rss_bytes']/2**20,flush=True)


if __name__=='__main__':
    main()
