"""All original arms/seeds, quarter-turn-only POST-OUTCOME sensitivity.

This cannot authorize or access confirmation. Results are not physically
validated benchmark evidence. No tuning; all fits precede evaluation.
"""
import copy
import json
import random
from dataclasses import asdict
import numpy as np
import torch
import investigate as a

HERE=a.HERE
ox=a.ox
base=a.base

def load_data():
    integrity=a.wf.read(HERE/'cache_integrity.json')
    assert a.wf.sha(HERE/'diagnostic_data.pt')==integrity['sha256']
    assert a.wf.sha(HERE/'investigate.py')==integrity['source_sha256']
    assert a.wf.sha(HERE/'PLAN.md')==integrity['plan_sha256']
    stored=torch.load(HERE/'diagnostic_data.pt',weights_only=False)
    d=a.DiagnosticData()
    for k in ('examples','tensors','connected'): setattr(d,k,stored[k])
    d.translation_scale=stored['scale']; d.train_examples=d.examples[('train',8)]
    assert {k[0] for k in d.tensors}=={'train','dev'}
    return d,integrity

def clustered(t,r,names,mask):
    names=np.asarray(names); t=np.asarray(t); r=np.asarray(r); mask=np.asarray(mask)
    clusters={}
    for name in sorted(set(names[mask])):
        select=(names==name)&mask
        clusters[name]={'n':int(select.sum()),'translation_mean_m':float(t[select].mean()),
                        'rotation_mean_deg':float(r[select].mean())}
    return {'clusters':clusters,'n':int(mask.sum()),
        'traversal_equal_translation_m':float(np.mean([x['translation_mean_m'] for x in clusters.values()])),
        'traversal_equal_rotation_deg':float(np.mean([x['rotation_mean_deg'] for x in clusters.values()]))}

@torch.no_grad()
def evaluate(d,model,length,seed):
    leaves,targets,moving=d.tensors[('dev',length)]; names=d.examples[('dev',length)]
    paths={'left':[], 'right':[], 'balanced':[], **{f'random_{i}':[] for i in range(16)}}
    model=model.cpu().float().eval()
    for offset in range(0,len(leaves),1024):
        batch=leaves[offset:offset+1024].float()
        for path in paths:
            mode='random' if path.startswith('random_') else path
            tree_seed=seed*100+int(path.split('_')[1]) if mode=='random' else seed
            rng=random.Random(tree_seed*1_000_003+length*1009+offset)
            p=model(batch,mode,rng if mode=='random' else None).double()
            assert p.isfinite().all()
            paths[path].append(p)
    out={}
    for path,values in paths.items():
        p=torch.cat(values); t,r=base.errors(p,targets)
        out[path]={subset:clustered(t,r,names,mask.numpy()) for subset,mask in
            [('all',torch.ones(len(leaves),dtype=torch.bool)),('connected',d.connected[('dev',length)]),('moving',moving)]}
    return out

def main():
    torch.set_num_threads(1); a.wf.verify_implementation()
    assert not a.wf.STARTED.exists() and not a.wf.RELEASE.exists()
    d,integrity=load_data(); cfg=ox.Config(); runs=HERE/'runs'; runs.mkdir(exist_ok=True)
    manifest={'diagnostic_only':True,'physical_frame_validated':False,'confirmation_accessed':False,
      'source_sha256':a.wf.sha(HERE/'retry.py'),'data_integrity':integrity,'config':asdict(cfg)}
    provenance=HERE/'retry_provenance.json'
    if provenance.exists(): assert a.wf.read(provenance)==manifest
    else: a.save(provenance,manifest)
    for seed in ox.SEEDS:
        for arm in ox.ARMS:
            path=runs/f'{arm}_seed{seed}.pt'
            if path.exists():
                assert a.wf.sha(path)==a.wf.read(path.with_suffix('.integrity.json'))['sha256']
                continue
            trained=ox.train_one(d,arm,seed,cfg,torch.device('cpu'))
            model=trained.pop('model')
            assert all(v.isfinite().all() for v in model.state_dict().values())
            with path.open('xb') as f:
                torch.save({'state_dict':model.cpu().state_dict(),'training':trained,'provenance':manifest,'seed':seed,'arm':arm},f)
            a.save(path.with_suffix('.integrity.json'),{'sha256':a.wf.sha(path)})
            print('FROZEN',arm,seed,trained,flush=True)
    for seed in ox.SEEDS:
        for arm in ox.ARMS:
            path=runs/f'{arm}_seed{seed}.pt'; out=runs/f'{arm}_seed{seed}_evaluation.json'
            if out.exists(): continue
            stored=torch.load(path,weights_only=False)
            assert stored['provenance']==manifest
            model=ox.OxfordComposer(arm);model.load_state_dict(stored['state_dict'])
            lengths={str(l):evaluate(d,model,l,seed) for l in ox.EVAL_LENGTHS}
            a.save(out,{'checkpoint_sha256':a.wf.sha(path),'lengths':lengths})
            print('EVALUATED',arm,seed,{p:lengths['32'][p]['all']['traversal_equal_translation_m'] for p in ('left','right','balanced')},flush=True)
    a.wf.verify_implementation()
    assert not a.wf.STARTED.exists() and not a.wf.RELEASE.exists()
    a.save(HERE/'retry_completed.json',{'provenance':manifest,'checkpoints':25,'evaluations':25})

if __name__=='__main__': main()
