"""Reproduce all 25 original checkpoints on original targets, without writes there."""
import numpy as np
import torch
import investigate as a
import retry

@torch.no_grad()
def main():
    torch.set_num_threads(1);d,integrity=retry.load_data()
    saved=a.wf.read(a.wf.ORIGINAL/'results_development.json')
    e=a.ox.load_extrinsic(a.wf.DATA/'extrinsics/ins.txt')
    quarter=a.ox.motors_from_xyzrpy(np.zeros((1,3)),np.array([[0,0,-np.pi/2]]))[0]
    change=a.base.motor_product(a.base.motor_product(e,quarter),a.ox.inverse_motor(e))
    original_targets={}
    for length in a.ox.EVAL_LENGTHS:
        corrected=d.tensors[('dev',length)][1]
        original_targets[length]=a.base.motor_product(a.base.motor_product(change,corrected),a.ox.inverse_motor(change))
    records=[]
    for arm in a.ox.ARMS:
        for seed in a.ox.SEEDS:
            path=a.wf.ORIGINAL/'runs'/f'{arm}_seed{seed}.pt'
            checkpoint=torch.load(path,weights_only=False)
            model=a.ox.OxfordComposer(arm);model.load_state_dict(checkpoint['state_dict']);model.eval()
            errors={}
            for length in a.ox.EVAL_LENGTHS:
                leaves=d.tensors[('dev',length)][0];names=d.examples[('dev',length)]
                predictions=torch.cat([model(leaves[i:i+1024].float(),'left').double() for i in range(0,len(leaves),1024)])
                t,r=a.base.errors(predictions,original_targets[length])
                result=retry.clustered(t,r,names,np.ones(len(names),dtype=bool))
                expected=saved['arms'][arm][str(seed)]['lengths'][str(length)]['paths']['left']
                discrepancy=max(abs(v['translation_mean_m']-expected['clusters'][n]['translation_mean_m']) for n,v in result['clusters'].items())
                assert discrepancy<1e-6,(arm,seed,length,discrepancy)
                assert all(v['n']==expected['clusters'][n]['n'] for n,v in result['clusters'].items())
                errors[str(length)]=discrepancy
            records.append({'arm':arm,'seed':seed,'checkpoint_sha256':a.wf.sha(path),'max_cluster_translation_discrepancy_m':errors})
            print('REPLAY PASS',arm,seed,max(errors.values()),flush=True)
    a.wf.verify_implementation();assert not a.wf.STARTED.exists() and not a.wf.RELEASE.exists()
    a.save(a.HERE/'checkpoint_replay.json',{'confirmation_accessed':False,'records':records,'data_integrity':integrity,'source_sha256':a.wf.sha(a.HERE/'checkpoint_replay.py')})

if __name__=='__main__':main()
