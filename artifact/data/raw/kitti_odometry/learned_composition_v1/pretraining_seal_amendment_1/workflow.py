"""Fail-closed provenance, immutable checkpoints, and staged KITTI access."""
import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

import acquire as a
import engine as e
import full_frontend as frontend
from frontend_smoke import save


def json_read(path):
    return json.loads(path.read_text())


def hash_map(paths):
    return {str(p.relative_to(a.ROOT)):a.sha(p) for p in paths}


def validate_hashes(mapping):
    for name,digest in mapping.items():
        if a.sha(a.ROOT/name)!=digest:
            raise RuntimeError(f'Immutable dependency changed: {name}')


def fingerprint():
    dependencies=set()
    for module in list(sys.modules.values()):
        name=getattr(module,'__file__',None)
        if name:
            path=Path(name).resolve()
            # Torch exposes virtual modules such as "_classes.py" with no
            # on-disk source. Pin actual local files, not invented cwd paths.
            if path.is_file() and path.is_relative_to(a.ROOT) and path.parent!=a.HERE and path.suffix=='.py':
                dependencies.add(path)
    sources=set(a.HERE.glob('*.py'))|set(a.HERE.glob('*.cpp'))|dependencies
    sources|={a.HERE/name for name in ('EXPERIMENT_PROTOCOL.md','IMPLEMENTATION_NOTES.md','PREPARATION_PROTOCOL.md','FRONTEND_SMOKE_PROTOCOL.md')}
    inputs={frontend.OUTPUT/'completed.json',frontend.OUTPUT/'specification_seal.json',e.RUNS/'training_data.json',e.RUNS/'preflight.json'}
    inputs|={frontend.OUTPUT/f'{seq}{suffix}' for seq in frontend.TRAIN for suffix in ('.json','.frames.jsonl')}
    inputs|={a.RAW/name for name in json_read(a.RAW/'acquisition.json')['staged_sha256']}
    return dict(sources=hash_map(sorted(sources)),inputs=hash_map(sorted(inputs)),
        python=sys.version,torch=torch.__version__,numpy=np.__version__,platform=platform.platform(),
        roles=a.ROLES,arms=list(e.ARMS),seeds=list(e.SEEDS),steps=5000,batch_size=64,
        torch_threads=1,deterministic_algorithms=True)


def verify_seal():
    e.configure()
    path=e.RUNS/'implementation_seal.json'
    if not path.exists():
        raise PermissionError('Implementation seal does not exist; nontraining access/fits prohibited.')
    seal=json_read(path)
    if seal['fingerprint']!=fingerprint():
        raise RuntimeError('Implementation/environment/data changed after seal')
    frontend.verify_seal()
    return seal


def completed_fits():
    paths=[]
    for seed in e.SEEDS:
        for arm in e.MODULES:
            folder=e.RUNS/'fits'/f'seed{seed}'/arm
            if (folder/'failure.json').exists():
                raise PermissionError(f'Recorded failure blocks campaign: {folder}')
            report=folder/'completed.json'
            if not report.exists():
                raise PermissionError(f'Incomplete prespecified fit: seed {seed}, {arm}')
            r=json_read(report)
            if r['steps']!=5000 or r['seed']!=seed or r['arm']!=arm:
                raise RuntimeError('Checkpoint identity/budget mismatch')
            if r['implementation_seal_sha256']!=a.sha(e.RUNS/'implementation_seal.json'):
                raise RuntimeError('Checkpoint belongs to another implementation')
            if a.sha(folder/'step5000.pt')!=r['checkpoint_sha256']:
                raise RuntimeError('Final checkpoint changed')
            paths.extend((report,folder/'step5000.pt'))
    return paths


def require_access(sequence):
    role=a.ROLES.get(sequence)
    if role=='train':
        return
    if role not in ('dev','study_holdout'):
        raise PermissionError('Official KITTI test and unknown sequences are outside scope')
    verify_seal()
    completed_fits()  # Deliberately keep even development pixels closed until final fits.
    if role=='study_holdout':
        path=e.RUNS/'holdout_release.json'
        if not path.exists():
            raise PermissionError('Study holdout release is absent')
        r=json_read(path)
        if r['implementation_seal_sha256']!=a.sha(e.RUNS/'implementation_seal.json'):
            raise RuntimeError('Stale release')
        validate_hashes(r['dependencies'])


def create_seal():
    e.configure();frontend.verify_seal()
    if not json_read(frontend.OUTPUT/'completed.json')['all_training_frontend_gates_pass']:
        raise PermissionError('Training frontend not admitted')
    e.RUNS.mkdir(parents=True,exist_ok=True)
    if (e.RUNS/'implementation_seal.json').exists():
        verify_seal();print('Existing implementation seal verified');return
    if (e.RUNS/'fits').exists():
        raise RuntimeError('Cannot create a prospective seal after fits exist')
    data=e.TrainingData().receipt()
    data_path=e.RUNS/'training_data.json'
    if data_path.exists():
        if json_read(data_path)!=data:
            raise RuntimeError('Training data receipt changed')
    else:
        save(data_path,data)
    before=fingerprint()
    tests=subprocess.run([sys.executable,'-m','unittest','discover','-s',str(a.HERE),'-p','test_*.py','-v'],
                         text=True,capture_output=True)
    if tests.returncode:
        print(tests.stdout+tests.stderr);raise RuntimeError('Preseal test suite failed')
    if before!=fingerprint():
        raise RuntimeError('Inputs changed while testing')
    save(e.RUNS/'implementation_seal.json',dict(created_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
         fingerprint=before,tests_stdout=tests.stdout,tests_stderr=tests.stderr,tests_returncode=tests.returncode,
         training_implementation_complete=True,holdout_release_implemented=True,
         model_updates_before_seal=0,development_accessed=False,heldout_accessed=False))
    print('Implementation sealed; training scale (m):',data['translation_scale_m'],flush=True)


def load_model(seed,arm,dtype=torch.float32):
    folder=e.RUNS/'fits'/f'seed{seed}'/arm
    report=json_read(folder/'completed.json')
    if report['checkpoint_sha256']!=a.sha(folder/'step5000.pt'):
        raise RuntimeError('Checkpoint changed')
    if report['implementation_seal_sha256']!=a.sha(e.RUNS/'implementation_seal.json'):
        raise RuntimeError('Stale checkpoint')
    model=e.make_module(arm,seed,dtype)
    model.load_state_dict(torch.load(folder/'step5000.pt',weights_only=True)['model'])
    return model.eval()


def fit_one(data,seed,arm):
    verify_seal()
    folder=e.RUNS/'fits'/f'seed{seed}'/arm
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/'failure.json').exists():
        raise RuntimeError('Failed fit retained; no automatic restart')
    if (folder/'completed.json').exists():
        load_model(seed,arm);print('Verified completed',seed,arm,flush=True);return
    model=e.make_module(arm,seed)
    calibrator=model if arm=='calibrated' else load_model(seed,'calibrated')
    if arm!='calibrated':
        calibrator.requires_grad_(False)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.002 if arm=='calibrated' else .001,
                               weight_decay=1e-4,betas=(.9,.999),eps=1e-8)
    generator=torch.Generator().manual_seed(30000+seed)
    step=0;elapsed=0.;trace=[];min_norm=float('inf')
    checkpoints=sorted(folder.glob('step*.pt'),key=lambda p:int(p.stem[4:]))
    if checkpoints:
        latest=checkpoints[-1];receipt=json_read(latest.with_suffix('.json'))
        if a.sha(latest)!=receipt['sha256']:
            raise RuntimeError('Resume checkpoint changed')
        state=torch.load(latest,weights_only=True)
        if state['seal']!=a.sha(e.RUNS/'implementation_seal.json') or state['seed']!=seed or state['arm']!=arm:
            raise RuntimeError('Invalid resume provenance')
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer'])
        generator.set_state(state['sampler_rng']);torch.set_rng_state(state['torch_rng'])
        step=state['step'];elapsed=state['elapsed'];trace=state['trace'];min_norm=state['minimum_norm']
    elif not (folder/'started.json').exists():
        save(folder/'started.json',dict(seed=seed,arm=arm,steps=5000,seal=a.sha(e.RUNS/'implementation_seal.json')))
    else:
        raise RuntimeError('Interrupted before first checkpoint; preserve attempt and request explicit recovery')
    started=time.perf_counter();last_save=started
    try:
        while step<5000:
            leaves,target,indices=data.sample(generator)
            optimizer.zero_grad(set_to_none=True)
            if arm=='calibrated':
                prediction,norm=e.forward(arm,leaves,calibrator)
            else:
                with torch.no_grad():
                    raw=e.product(leaves);cal,norm=e.calibrated_leaves(calibrator,leaves)
                    cached=(raw,e.product(cal))
                prediction,head_norm=e.forward(arm,leaves,calibrator,model,cached)
                norm=min(norm,head_norm)
            value=e.loss(prediction,target,data.scale)
            if not bool(torch.isfinite(value)):
                raise FloatingPointError('Nonfinite loss')
            value.backward()
            grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step();step+=1;min_norm=min(min_norm,norm)
            if not all(bool(torch.isfinite(p).all()) for p in model.parameters()):
                raise FloatingPointError('Nonfinite updated parameters')
            if step==1 or step%100==0:
                trace.append(dict(step=step,loss=float(value.detach()),gradient_norm=float(grad),
                    length=leaves.shape[1],sample_sha256=hashlib.sha256(json.dumps(indices).encode()).hexdigest()))
                print(f'seed={seed} arm={arm} step={step}/5000 loss={float(value.detach()):.6g}',flush=True)
            if step%500==0 or step==5000:
                now=time.perf_counter();elapsed+=now-last_save;last_save=now
                checkpoint=folder/f'step{step}.pt'
                state=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),sampler_rng=generator.get_state(),
                    torch_rng=torch.get_rng_state(),step=step,seed=seed,arm=arm,elapsed=elapsed,trace=trace,
                    minimum_norm=min_norm,seal=a.sha(e.RUNS/'implementation_seal.json'))
                # Exclusive creation; partial files are never silently overwritten.
                with checkpoint.open('xb') as stream:
                    torch.save(state,stream);stream.flush();os.fsync(stream.fileno())
                save(checkpoint.with_suffix('.json'),dict(sha256=a.sha(checkpoint),step=step))
        verify_seal()
        save(folder/'completed.json',dict(seed=seed,arm=arm,steps=step,seconds=elapsed,
            parameters=sum(p.numel() for p in model.parameters()),minimum_preprojection_norm=min_norm,
            checkpoint_sha256=a.sha(folder/'step5000.pt'),implementation_seal_sha256=a.sha(e.RUNS/'implementation_seal.json'),
            trace=trace,development_accessed=False,heldout_accessed=False))
    except Exception:
        save(folder/'failure.json',dict(seed=seed,arm=arm,step=step,error=traceback.format_exc()))
        raise


def train():
    verify_seal();data=e.TrainingData()
    if data.receipt()!=json_read(e.RUNS/'training_data.json'):
        raise RuntimeError('Derived training data changed')
    # OS advisory lock is released even after a crash; only this campaign uses it.
    import fcntl
    with (e.RUNS/'training.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for seed in e.SEEDS:
            for arm in e.MODULES:
                fit_one(data,seed,arm)
    completed_fits();print('All 30 fixed fits complete.',flush=True)


def release_holdout():
    verify_seal();dependencies=completed_fits()
    for split,sequences in (('train',frontend.TRAIN),('dev',('07','08'))):
        for seed in e.SEEDS:
            for sequence in sequences:
                folder=e.RUNS/'evaluation'/split/f'seed{seed}'/sequence
                report=folder/'completed.json';r=json_read(report)
                if not r['technical_pass'] or set(r['arms'])!=set(e.ARMS) or r['seed']!=seed or r['sequence']!=sequence:
                    raise PermissionError('Incomplete/failed eight-arm report or invariant audit')
                primary=r['summary']['distance_100']
                if primary['evaluated']<=0 or set(primary['arms'])!=set(e.ARMS):
                    raise PermissionError('Missing primary predictions')
                if not all(np.isfinite(list(metrics.values())).all() for metrics in primary['arms'].values()):
                    raise PermissionError('Nonfinite primary report')
                if not r['frontend_admitted'] or r['minimum_preprojection_real_norm']<=1e-6:
                    raise PermissionError('Invalid frontend/projection domain')
                if set(r['audits'])!=set(map(str,e.AUDIT_LENGTHS)) or not all(x['passed'] for x in r['audits'].values()):
                    raise PermissionError('Missing or failed geometric audit')
                if r['seal']!=a.sha(e.RUNS/'implementation_seal.json'):
                    raise RuntimeError('Stale evaluation report')
                validate_hashes(r['dependencies']);dependencies.append(report)
                dependencies.extend(a.ROOT/name for name in r['dependencies'])
    # No accuracy comparison or positive-effect condition appears here.
    save(e.RUNS/'holdout_release.json',dict(implementation_seal_sha256=a.sha(e.RUNS/'implementation_seal.json'),
        dependencies=hash_map(sorted(set(dependencies))),arms=list(e.ARMS),seeds=list(e.SEEDS),
        allowed_sequences=['09','10'],positive_development_effect_required=False))


if __name__=='__main__':
    # Keep imports canonical when this file is also imported by engine.Sequence.
    sys.modules['workflow']=sys.modules[__name__]
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=['seal','verify','train','release-holdout'])
    args=parser.parse_args()
    {'seal':create_seal,'verify':verify_seal,'train':train,'release-holdout':release_holdout}[args.action]()
