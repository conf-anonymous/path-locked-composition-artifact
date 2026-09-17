"""Post-outcome reference audit; recorded development data only, no fitting.

SDK matrix conversion/accumulation independently checks the motor pipeline.
The fixed quarter-turn is a diagnostic of the previously reported RTK frame
issue, NOT an authorized corrected extrinsic or a new scientific result.
No SDK interpolation is invoked. All output observations are recorded rows.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE / 'reference'))
from transform import build_se3_transform
sys.path.insert(0, str(ROOT / 'experiments/031_oxford_mergeable_confirmation'))
import oxford_workflow as wf
ox = wf.original()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path):
    return json.loads(path.read_text())


def csv_numeric(path, columns):
    with path.open(newline='') as stream:
        reader = csv.reader(stream)
        header = next(reader)
        rows = list(reader)
    return (np.asarray([int(r[0]) for r in rows], dtype=np.int64),
            np.asarray([[float(r[i]) for i in columns] for r in rows]), header)


def matrix_rows(xyzrpy):
    return np.asarray([np.asarray(build_se3_transform(row)) for row in xyzrpy])


def relative(a, b):
    return np.linalg.solve(a, b)


def motor_matrices(motors):
    q, t = ox.base.decode_motor(motors)
    # Independent closed-form quaternion-to-matrix map; not a product helper.
    w, x, y, z = q.numpy().T
    out = np.tile(np.eye(4), (len(q), 1, 1))
    out[:, 0, :3] = np.stack((1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)), -1)
    out[:, 1, :3] = np.stack((2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)), -1)
    out[:, 2, :3] = np.stack((2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)), -1)
    out[:, :3, 3] = t.numpy()
    return out


def describe(values):
    a = np.asarray(values)
    return {'n': int(a.size), 'min': float(a.min()), 'median': float(np.median(a)),
            'p95': float(np.quantile(a, .95)), 'max': float(a.max())}


def audit_record(record, extrinsic, original):
    assert record['split'] == 'dev'  # No option for confirmation or split substitution.
    name = record['traversal']
    vp, rp = (wf.DATA / record[k]['path'] for k in ('vo', 'rtk'))
    assert sha(vp) == record['vo']['sha256'] and sha(rp) == record['rtk']['sha256']
    vt, v, _ = csv_numeric(vp, (2,3,4,5,6,7))
    rt, r, _ = csv_numeric(rp, (4,5,6,11,12,13))
    assert np.all(np.diff(vt) > 0) and np.all(np.diff(rt) > 0)
    transforms = matrix_rows(v)
    current = np.eye(4)
    absolute = np.empty_like(transforms)
    for i, step in enumerate(transforms):
        current = current @ step  # Official SDK row order and column-zero timestamp.
        absolute[i] = current
    grid = np.arange(max(vt[0], rt[0]), min(vt[-1], rt[-1])+1, 500000, dtype=np.int64)
    ri = np.searchsorted(rt, grid, side='left')
    vi = np.searchsorted(vt, rt[ri], side='left')
    valid = vi < len(vt)
    safe = np.minimum(vi, len(vt)-1)
    valid &= (vt[safe]-rt[ri] >= 0) & (vt[safe]-rt[ri] <= 100000)
    rr = matrix_rows(r[ri])
    target_abs = rr @ np.linalg.inv(extrinsic)
    # Reported 90-degree body-axis issue; no optimization or angle sweep.
    quarter = np.asarray(build_se3_transform([0,0,0,0,0,-np.pi/2]))
    diagnostic_abs = rr @ quarter @ np.linalg.inv(extrinsic)
    spans, start = [], None
    for i, ok in enumerate(np.r_[valid, False]):
        if ok and start is None:
            start = i
        if not ok and start is not None:
            if i-start > 32:
                spans.append((start, i))
            start = None
    # Exercise actual frozen loader separately, without touching confirmations.
    pipeline = ox.OxfordTraversal(name, vp, rp, ox.load_extrinsic(wf.DATA/'extrinsics/ins.txt'), ox.Config())
    assert len(pipeline.runs) == len(spans)
    max_vo, max_rtk = 0., 0.
    for (a,b), (pv, pr) in zip(spans, pipeline.runs):
        for matrices, motors, kind in ((absolute[vi[a:b]], pv, 'vo'), (target_abs[a:b], pr, 'rtk')):
            reference = relative(matrices[:-1], matrices[1:])
            observed = motor_matrices(ox.relative_from_absolute(motors[:-1], motors[1:]))
            discrepancy = float(np.max(np.abs(reference-observed)))
            if kind == 'vo': max_vo = max(max_vo, discrepancy)
            else: max_rtk = max(max_rtk, discrepancy)
    lengths = {}
    for length in (4,8,16,32):
        raw, identity, quarter_error, durations, displacements, direction, moving_raw = [],[],[],[],[],[],[]
        for a,b in spans:
            first = np.arange(a, b-length)
            last = first+length
            predicted = relative(absolute[vi[first]], absolute[vi[last]])
            target = relative(target_abs[first], target_abs[last])
            diagnostic = relative(diagnostic_abs[first], diagnostic_abs[last])
            pt, tt = predicted[:, :3, 3], target[:, :3, 3]
            err = np.linalg.norm(pt-tt, axis=1)
            disp = np.linalg.norm(tt, axis=1)
            raw.extend(err); identity.extend(disp)
            quarter_error.extend(np.linalg.norm(pt-diagnostic[:, :3, 3], axis=1))
            durations.extend((rt[ri[last]]-rt[ri[first]])/1e6)
            displacements.extend(disp)
            moving_raw.extend(err[disp >= 5])
            moving = (disp >= 5) & (np.linalg.norm(pt, axis=1) >= 5)
            cos = (pt[moving]*tt[moving]).sum(1)/(np.linalg.norm(pt[moving], axis=1)*disp[moving])
            direction.extend(np.degrees(np.arccos(np.clip(cos, -1, 1))))
        saved = original['baselines'][str(length)]['raw_motor']['clusters'][name]
        assert len(raw) == saved['n']
        delta = abs(float(np.mean(raw))-saved['translation_mean_m'])
        assert delta < 1e-5, (name, length, delta)
        lengths[str(length)] = {'n':len(raw), 'raw_m':float(np.mean(raw)),
            'identity_m':float(np.mean(identity)), 'fixed_quarter_turn_diagnostic_m':float(np.mean(quarter_error)),
            'raw_moving_m':float(np.mean(moving_raw)), 'saved_raw_mean_discrepancy_m':delta,
            'actual_rtk_duration_s':describe(durations), 'rtk_displacement_m':describe(displacements),
            'raw_vs_target_direction_deg_moving':describe(direction)}
    out = {'traversal':name, 'split':'dev', 'n_runs':len(spans),
        'source_sha256':{'vo':sha(vp),'rtk':sha(rp)},
        'matrix_motor_relative_max_abs':{'vo':max_vo,'rtk':max_rtk},
        'rtk_grid_lag_s':describe((rt[ri[valid]]-grid[valid])/1e6),
        'vo_rtk_offset_s':describe((vt[vi[valid]]-rt[ri[valid]])/1e6),
        'duplicate_selected_rtk_rows':int(np.sum(np.diff(ri[valid])==0)), 'lengths':lengths}
    assert max(max_vo,max_rtk) < 1e-5
    print(name, 'L32', {k:lengths['32'][k] for k in ('raw_m','fixed_quarter_turn_diagnostic_m')}, flush=True)
    return out


def main():
    torch.set_num_threads(1)
    wf.verify_implementation()
    manifest = read_json(wf.MANIFEST)
    original = read_json(wf.ORIGINAL/'results_development.json')
    selected = [r for r in manifest['records'] if r['split']=='dev']
    assert len(selected)==14
    extrinsic = np.asarray(build_se3_transform(np.loadtxt(wf.DATA/'extrinsics/ins.txt')))
    records = [audit_record(r,extrinsic,original) for r in selected]
    metrics = ('raw_m','identity_m','fixed_quarter_turn_diagnostic_m','raw_moving_m')
    aggregate = {str(l):{k:float(np.mean([r['lengths'][str(l)][k] for r in records])) for k in metrics} for l in (4,8,16,32)}
    report = {'audit_date':'2026-09-10', 'post_outcome':True, 'training_performed':False,
        'confirmation_accessed':False, 'sdk_commit':'16ce3329223ca418fe5106277b91aea8d9b672b2',
        'sdk_source_sha256':{p.name:sha(p) for p in sorted((HERE/'reference').glob('*.py'))},
        'audit_source_sha256':sha(Path(__file__)), 'fingerprint':wf.fingerprint(),
        'source_results_sha256':sha(wf.ORIGINAL/'results_development.json'),
        'status':'frame_convention_confounded_not_validated_odometry_gains',
        'diagnostic_not_corrected_protocol':True,
        'diagnostic_definition':'T_world_rtk @ Rz(-pi/2) @ inverse(T_vehicle_ins); fixed angle, no fitting',
        'independent_reference':'official SDK SE3 matrices, chronological products, exact recorded rows; no interpolation',
        'aggregate_traversal_equal':aggregate,'records':records}
    wf.verify_implementation()
    assert not wf.STARTED.exists() and not wf.RELEASE.exists()
    with (HERE/'audit_results.json').open('x') as stream:
        json.dump(report,stream,indent=2,allow_nan=False); stream.write('\n')
    print(json.dumps(aggregate,indent=2))


if __name__=='__main__':
    main()
