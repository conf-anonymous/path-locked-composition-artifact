"""Report both prespecified diagnostic retries; no best seed or subset choice."""
import json
import numpy as np
import torch
import investigate as a
import retry

HERE=a.HERE

def ci(values):
    values=np.asarray(values,dtype=float)
    rng=np.random.default_rng(320910)
    samples=values[rng.integers(len(values),size=(10000,len(values)))].mean(1)
    return {'mean':float(values.mean()),'ci95':np.quantile(samples,[.025,.975]).tolist(),
            'n_traversals':len(values)}

def campaign(folder,d,masks):
    records={}
    for arm in a.ox.ARMS:
        records[arm]=[a.wf.read(folder/'runs'/f'{arm}_seed{s}_evaluation.json') for s in a.ox.SEEDS]
        for s,r in enumerate(records[arm]):
            assert r['checkpoint_sha256']==a.wf.sha(folder/'runs'/f'{arm}_seed{s}.pt')
    summary={}
    for length in a.ox.EVAL_LENGTHS:
        leaves,target,moving=d.tensors[('dev',length)]; names=d.examples[('dev',length)]
        raw=a.base.reduce_states(list(leaves.unbind(1)),a.base.motor_product,'left')
        rt,rr=a.base.errors(raw,target)
        identity=raw.new_zeros(raw.shape);identity[:,0]=1
        it,ir=a.base.errors(identity,target)
        subsets={'all':torch.ones(len(leaves),dtype=torch.bool),'moving':moving,**{k:v[('dev',length)] for k,v in masks.items()}}
        result={}
        for subset,mask in subsets.items():
            raw_result=retry.clustered(rt,rr,names,mask.numpy())
            identity_result=retry.clustered(it,ir,names,mask.numpy())
            keys=sorted(raw_result['clusters'])
            raw_values=np.array([raw_result['clusters'][n]['translation_mean_m'] for n in keys])
            arms={}
            for arm in a.ox.ARMS:
                per_path={}
                for path in ('left','right','balanced'):
                    values=np.array([[r['lengths'][str(length)][path][subset]['clusters'][n]['translation_mean_m'] for n in keys] for r in records[arm]])
                    rotations=np.array([[r['lengths'][str(length)][path][subset]['clusters'][n]['rotation_mean_deg'] for n in keys] for r in records[arm]])
                    per_path[path]={'seed_translation_means_m':values.mean(1).tolist(),
                        'translation_mean_m':float(values.mean()),'rotation_mean_deg':float(rotations.mean()),
                        'minus_raw':ci(values.mean(0)-raw_values)}
                left=np.array([[r['lengths'][str(length)]['left'][subset]['clusters'][n]['translation_mean_m'] for n in keys] for r in records[arm]])
                right=np.array([[r['lengths'][str(length)]['right'][subset]['clusters'][n]['translation_mean_m'] for n in keys] for r in records[arm]])
                per_path['right_minus_left']=ci((right-left).mean(0))
                per_path['random16_translation_m']=float(np.mean([r['lengths'][str(length)][f'random_{j}'][subset]['traversal_equal_translation_m'] for r in records[arm] for j in range(16)]))
                arms[arm]=per_path
            result[subset]={'raw':raw_result,'identity':identity_result,'arms':arms}
        summary[str(length)]=result
    return summary

def main():
    torch.set_num_threads(1);d,integrity=retry.load_data()
    for completed in (HERE/'retry_completed.json',HERE/'quality_screened/completed.json'):assert completed.exists()
    masks=torch.load(HERE/'quality_screened/masks.pt',weights_only=False)
    out={'diagnostic_only':True,'physical_frame_validated':False,'confirmation_accessed':False,
         'uncertainty':'95% percentile bootstrap of whole development traversals, 10000 replicates, seed 320910; seed means within traversal first; exploratory, not confirmatory p-values',
         'frame_only':campaign(HERE,d,{'connected':d.connected}),
         'quality_screened_training':campaign(HERE/'quality_screened',d,{'quality_screened':masks}),
         'data_integrity':integrity,'source_sha256':a.wf.sha(HERE/'analyze.py')}
    a.wf.verify_implementation();assert not a.wf.STARTED.exists() and not a.wf.RELEASE.exists()
    a.save(HERE/'analysis.json',out)
    for campaign_name,key in [('frame_only','all'),('quality_screened_training','all'),('quality_screened_training','quality_screened')]:
        r=out[campaign_name]['32'][key]
        print(campaign_name,key,'raw',r['raw']['traversal_equal_translation_m'],'identity',r['identity']['traversal_equal_translation_m'])
        for arm,v in r['arms'].items():print(arm, 'left',v['left']['translation_mean_m'],'right',v['right']['translation_mean_m'],'minus_raw',v['left']['minus_raw'])

if __name__=='__main__':main()
