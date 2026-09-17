"""Documented post-outcome quality-screen sensitivity; no confirmation access."""
import json
import zipfile
from dataclasses import asdict
import numpy as np
import torch
import investigate as a
import retry

HERE=a.HERE
OUT=HERE/'quality_screened'

def masks_and_audit(d):
    pools={key:[] for key in d.examples}; reports=[]
    archive=a.wf.DATA/'official_archives/rtk.zip'
    with zipfile.ZipFile(archive) as z:
        members=z.namelist()
        for rec in a.selected_records():
            name=rec['traversal'];split=rec['split']
            matches=[s for s in members if s.endswith('/'+name+'/rtk.csv') or s==name+'/rtk.csv']
            assert len(matches)==1,(name,matches)
            # Only selected train/dev members are opened, never confirmation.
            content=z.read(matches[0]); assert a.hashlib.sha256(content).hexdigest()==rec['rtk']['sha256']
            _,va,targets,ri,vi,near,spans,components,report=a.load_record(rec)
            r=np.loadtxt(a.wf.DATA/rec['rtk']['path'],delimiter=',',skiprows=1,usecols=(0,4,5,6,8,9,10))
            dt=np.diff(r[:,0])/1e6
            speed=np.linalg.norm(np.diff(r[:,1:4],axis=0),axis=1)/dt
            reported=np.linalg.norm(r[:,4:7],axis=1)
            bad=(speed>50)|(reported[:-1]>50)|(reported[1:]>50)
            prefix=np.r_[0,np.cumsum(bad)]
            worst=np.argsort(speed)[-5:][::-1]
            item={'traversal':name,'split':split,'archive_member':matches[0],
                'archive_matches_extracted_csv':True,'bad_rtk_edges':int(bad.sum()),
                'max_implied_speed_mps':float(speed.max()),'max_reported_speed_mps':float(reported.max()),
                'worst_edges':[{'first_csv_line':int(i+2),'first_timestamp':int(r[i,0]),
                    'second_timestamp':int(r[i+1,0]),'dt_s':float(dt[i]),
                    'displacement_m':float(np.linalg.norm(r[i+1,1:4]-r[i,1:4])),
                    'implied_speed_mps':float(speed[i])} for i in worst], 'lengths':{}}
            for length in a.ox.EVAL_LENGTHS:
                clean_parts=[];lag=[];duplicate=[];duration=[]
                for start,end in spans:
                    first=np.arange(start,end-length);last=first+length
                    clean=(components[vi[first]]==components[vi[last]]) & (prefix[ri[first]]==prefix[ri[last]])
                    clean_parts.append(torch.from_numpy(clean))
                    duplicate.extend(ri[first]==ri[last]);duration.extend((r[ri[last],0]-r[ri[first],0])/1e6)
                mask=torch.cat(clean_parts)
                pools[(split,length)].append(mask)
                item['lengths'][str(length)]={'total':len(mask),'retained':int(mask.sum()),
                    'duplicate_endpoint_windows':int(np.sum(duplicate)), 'duration_s':a.stats(duration)}
            reports.append(item)
    masks={k:torch.cat(v) for k,v in pools.items()}
    for key in masks: assert len(masks[key])==len(d.examples[key])
    return masks,reports

def main():
    torch.set_num_threads(1); a.wf.verify_implementation()
    assert not a.wf.STARTED.exists() and not a.wf.RELEASE.exists()
    OUT.mkdir(exist_ok=True); d,integrity=retry.load_data()
    if (OUT/'masks.pt').exists():
        saved=a.wf.read(OUT/'mask_integrity.json');assert saved['sha256']==a.wf.sha(OUT/'masks.pt')
        assert saved['source_sha256']==a.wf.sha(HERE/'quality_retry.py')
        masks=torch.load(OUT/'masks.pt',weights_only=False)
    else:
        masks,reports=masks_and_audit(d)
        a.save(OUT/'source_quality.json',{'rtk_archive_sha256':a.wf.sha(a.wf.DATA/'official_archives/rtk.zip'),
            'confirmation_accessed':False,'records':reports})
        with (OUT/'masks.pt').open('xb') as f:torch.save(masks,f)
        a.save(OUT/'mask_integrity.json',{'sha256':a.wf.sha(OUT/'masks.pt'),'source_sha256':a.wf.sha(HERE/'quality_retry.py')})
    # A diagnostic subclass exposes the quality mask as the connected subset
    # to the unchanged evaluator. Output renames it explicitly below.
    d.connected=masks
    for key,mask in masks.items():
        if key[0]!='train':continue
        d.tensors[key]=tuple(x[mask] for x in d.tensors[key])
        d.examples[key]=[n for n,keep in zip(d.examples[key],mask.tolist()) if keep]
    d.train_examples=d.examples[('train',8)]
    _,t=a.base.decode_motor(d.tensors[('train',8)][1]);d.translation_scale=max(float(t.norm(dim=-1).median()),1e-3)
    cfg=a.ox.Config();runs=OUT/'runs';runs.mkdir(exist_ok=True)
    provenance={'diagnostic_only':True,'physical_frame_validated':False,'confirmation_accessed':False,
        'source_sha256':a.wf.sha(HERE/'quality_retry.py'),'addendum_sha256':a.wf.sha(HERE/'QUALITY_ADDENDUM.md'),
        'retry_helper_sha256':a.wf.sha(HERE/'retry.py'),'data_integrity':integrity,'config':asdict(cfg),
        'training_windows':len(d.train_examples),'translation_scale':d.translation_scale,
        'quality_mask_sha256':a.wf.sha(OUT/'masks.pt')}
    if (OUT/'provenance.json').exists():assert a.wf.read(OUT/'provenance.json')==provenance
    else:a.save(OUT/'provenance.json',provenance)
    for seed in a.ox.SEEDS:
        for arm in a.ox.ARMS:
            path=runs/f'{arm}_seed{seed}.pt'
            if path.exists():
                assert a.wf.sha(path)==a.wf.read(path.with_suffix('.integrity.json'))['sha256'];continue
            fit=a.ox.train_one(d,arm,seed,cfg,torch.device('cpu'));model=fit.pop('model')
            assert all(v.isfinite().all() for v in model.state_dict().values())
            with path.open('xb') as f:torch.save({'state_dict':model.cpu().state_dict(),'training':fit,'provenance':provenance},f)
            a.save(path.with_suffix('.integrity.json'),{'sha256':a.wf.sha(path)})
            print('QUALITY FROZEN',arm,seed,fit,flush=True)
    for seed in a.ox.SEEDS:
        for arm in a.ox.ARMS:
            path=runs/f'{arm}_seed{seed}.pt';out=runs/f'{arm}_seed{seed}_evaluation.json'
            if out.exists():continue
            checkpoint=torch.load(path,weights_only=False);assert checkpoint['provenance']==provenance
            model=a.ox.OxfordComposer(arm);model.load_state_dict(checkpoint['state_dict'])
            lengths={str(l):retry.evaluate(d,model,l,seed) for l in a.ox.EVAL_LENGTHS}
            for record in lengths.values():
                for paths in record.values():paths['quality_screened']=paths.pop('connected')
            a.save(out,{'checkpoint_sha256':a.wf.sha(path),'lengths':lengths})
            print('QUALITY EVALUATED',arm,seed,{p:lengths['32'][p]['quality_screened']['traversal_equal_translation_m'] for p in ('left','right','balanced')},flush=True)
    a.wf.verify_implementation();assert not a.wf.STARTED.exists() and not a.wf.RELEASE.exists()
    a.save(OUT/'completed.json',{'provenance':provenance,'checkpoints':25,'evaluations':25})

if __name__=='__main__':main()
