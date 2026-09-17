"""Bounded-memory archive checks and training-only KITTI recording validation."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import time
import zipfile
import zlib
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RAW = ROOT / 'data/raw/kitti_odometry'
ARCHIVES = ('devkit_odometry.zip', 'data_odometry_calib.zip',
            'data_odometry_poses.zip', 'data_odometry_gray.zip')
ROLES = {f'{i:02}':('train' if i <= 6 else 'dev' if i <= 8 else
                   'study_holdout' if i <= 10 else 'official_test') for i in range(22)}
EXPECTED_TRAINING_COUNTS = (4541,1101,4661,801,271,2761,1101)
BUFFER = 4 * 1024 * 1024


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(BUFFER), b''):
            h.update(block)
    return h.hexdigest()


def safe_member(info):
    path = PurePosixPath(info.filename)
    assert not path.is_absolute() and '..' not in path.parts
    assert '\\' not in info.filename and not info.flag_bits & 1
    assert not stat.S_ISLNK(info.external_attr >> 16)


def verify_archive(path):
    started = time.monotonic()
    before = path.stat()
    print(f'{path.name}: SHA-256 ({before.st_size/1e9:.2f} GB)', flush=True)
    digest = sha(path)
    sequences = {}
    with zipfile.ZipFile(path) as z:
        members = z.infolist()
        assert len({m.filename for m in members}) == len(members)
        total = 0
        for j,m in enumerate(members):
            safe_member(m)
            if m.is_dir():
                continue
            crc = 0; count = 0
            with z.open(m) as f:
                for block in iter(lambda:f.read(BUFFER), b''):
                    count += len(block); crc = zlib.crc32(block,crc)
            assert count == m.file_size and crc == m.CRC
            total += count
            match = re.fullmatch(r'dataset/sequences/(\d{2})/image_([01])/(\d{6})\.png',m.filename)
            if match:
                seq,cam,frame = match.groups()
                assert seq in ROLES
                bucket = sequences.setdefault(seq,{}).setdefault(cam,[])
                bucket.append(int(frame))
            if j and j % 10000 == 0:
                print(f'{path.name}: CRC checked {j}/{len(members)} members',flush=True)
        for seq,cameras in sequences.items():
            assert set(cameras) == {'0','1'}
            assert sorted(cameras['0']) == sorted(cameras['1'])
            for cam,indices in cameras.items():
                assert sorted(indices) == list(range(len(indices)))
                cameras[cam] = {'count':len(indices),'first':min(indices),'last':max(indices)}
    after = path.stat()
    assert (before.st_size,before.st_mtime_ns)==(after.st_size,after.st_mtime_ns)
    return dict(file=path.name,source_path=str(path.resolve()),bytes=before.st_size,
        mtime_ns=before.st_mtime_ns,sha256=digest,zip_crc_verified=True,
        publisher_digest_verified=False,member_count=len(members),
        uncompressed_bytes=total,images=sequences,seconds=time.monotonic()-started)


def preserve(path,blob):
    path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists():
        with path.open('xb') as f:
            f.write(blob)
    assert path.read_bytes()==blob
    return hashlib.sha256(blob).hexdigest()


def stage_metadata(downloads):
    staged = {}
    with zipfile.ZipFile(downloads/'devkit_odometry.zip') as z:
        for m in z.infolist():
            safe_member(m)
            if not m.is_dir():
                p = RAW/m.filename
                staged[str(p.relative_to(RAW))]=preserve(p,z.read(m))
    with zipfile.ZipFile(downloads/'data_odometry_calib.zip') as cz, \
         zipfile.ZipFile(downloads/'data_odometry_poses.zip') as pz:
        for seq,role in ROLES.items():
            if role != 'train':
                continue
            for z,member in ((cz,f'dataset/sequences/{seq}/calib.txt'),
                             (cz,f'dataset/sequences/{seq}/times.txt'),
                             (pz,f'dataset/poses/{seq}.txt')):
                p=RAW/member
                staged[str(p.relative_to(RAW))]=preserve(p,z.read(member))
    return staged


def summaries(x):
    return dict(min=float(np.min(x)),median=float(np.median(x)),
                p95=float(np.quantile(x,.95)),max=float(np.max(x)))


def inspect_training(seq,images):
    assert ROLES[seq]=='train', 'Non-training numeric inspection is forbidden.'
    folder=RAW/'dataset/sequences'/seq
    times=np.loadtxt(folder/'times.txt')
    poses=np.loadtxt(RAW/'dataset/poses'/f'{seq}.txt').reshape(-1,3,4)
    assert len(times)==len(poses)==images[seq]['0']['count']==EXPECTED_TRAINING_COUNTS[int(seq)]
    assert np.isfinite(times).all() and np.isfinite(poses).all()
    dt=np.diff(times); assert np.all(dt>0)
    rotations=poses[:,:,:3]
    ortho=np.max(abs(rotations.transpose(0,2,1)@rotations-np.eye(3)),axis=(1,2))
    det=np.linalg.det(rotations)
    assert max(ortho)<1e-4 and np.max(abs(det-1))<1e-4
    assert np.max(abs(poses[0]-np.eye(4)[:3]))<1e-4
    calibration={}
    for line in (folder/'calib.txt').read_text().splitlines():
        key,value=line.split(':',1)
        calibration[key]=np.fromstring(value,sep=' ').reshape(3,4)
    assert all(np.isfinite(v).all() for v in calibration.values())
    p0,p1=calibration['P0'],calibration['P1']
    assert p0[0,0]>0 and p1[0,0]>0
    baseline=p0[0,3]/p0[0,0]-p1[0,3]/p1[0,0]
    assert baseline>0
    displacement=np.linalg.norm(np.diff(poses[:,:,3],axis=0),axis=1)
    return dict(sequence=seq,role='train',frames=len(times),
        duration_s=float(times[-1]-times[0]),dt_s=summaries(dt),
        stereo_baseline_m=float(baseline),focal_x=float(p0[0,0]),
        rotation_orthogonality_max=float(ortho.max()),
        rotation_determinant=summaries(det),
        observed_step_m=summaries(displacement),
        implied_speed_m_per_s=summaries(displacement/dt),
        reference_path_length_m=float(displacement.sum()),
        physical_accuracy_certified=False)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--downloads',required=True,type=Path)
    args=parser.parse_args()
    result=RAW/'acquisition.json'
    assert not result.exists(), 'Acquisition record already exists; do not overwrite.'
    RAW.mkdir(parents=True,exist_ok=True)
    protocol_sha=sha(HERE/'PREPARATION_PROTOCOL.md')
    with (RAW/'preparation_seal.json').open('x') as f:
        json.dump(dict(protocol_sha256=protocol_sha,source_sha256=sha(Path(__file__)),
                       roles=ROLES,created_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())),f,indent=2)
    archives=[verify_archive(args.downloads/name) for name in ARCHIVES]
    images=next(r for r in archives if r['file']=='data_odometry_gray.zip')['images']
    assert set(images)==set(ROLES)
    staged=stage_metadata(args.downloads)
    training=[inspect_training(s,images) for s,r in ROLES.items() if r=='train']
    report=dict(created_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        archives=archives,staged_sha256=staged,roles=ROLES,training_checks=training,
        numeric_sequences_inspected=[r['sequence'] for r in training],
        held_out_content_integrity_read=True,held_out_scientific_inspection=False,
        images_decoded=False,images_extracted=False,training_performed=False,
        source_sha256=sha(Path(__file__)),protocol_sha256=protocol_sha,
        image_archive_copied=False,reference_source='Author downloads from official KITTI site; not independently authenticated by publisher digest.')
    with result.open('x') as f:
        json.dump(report,f,indent=2,allow_nan=False); f.write('\n')
    print('Acquisition complete:',result,flush=True)
    for t in training:
        print(t['sequence'],t['frames'],'frames; baseline',t['stereo_baseline_m'],
              'm; max step speed',t['implied_speed_m_per_s']['max'],'m/s',flush=True)


if __name__=='__main__':
    main()
