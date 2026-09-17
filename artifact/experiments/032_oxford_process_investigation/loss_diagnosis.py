"""Quantify recorded-target outliers and their squared-loss concentration."""
import numpy as np
import torch
import investigate as a
import retry

def main():
    torch.set_num_threads(1);d,integrity=retry.load_data();records={}
    for split in ('train','dev'):
        for length in (8,32):
            leaves,target,_=d.tensors[(split,length)]
            pred=a.base.reduce_states(list(leaves.unbind(1)),a.base.motor_product,'left')
            _,t=a.base.decode_motor(target);_,pt=a.base.decode_motor(pred)
            displacement=t.norm(dim=-1);error=(t-pt).norm(dim=-1)
            squared=error.square(); k=max(1,len(error)//100)
            worst=displacement.argsort(descending=True)[:5].tolist()
            records[f'{split}_L{length}']={'n':len(error),
                'target_displacement_quantiles_m':np.quantile(displacement.numpy(),[0,.5,.9,.99,.999,1]).tolist(),
                'raw_translation_error_quantiles_m':np.quantile(error.numpy(),[0,.5,.9,.99,.999,1]).tolist(),
                'quantile_levels':[0,.5,.9,.99,.999,1],
                'top_one_percent_count':k,
                'top_one_percent_squared_translation_loss_fraction':float(squared.topk(k).values.sum()/squared.sum()),
                'largest_targets':[{'window_index':i,'traversal':d.examples[(split,length)][i],
                    'displacement_m':float(displacement[i])} for i in worst]}
    a.save(a.HERE/'loss_diagnosis.json',{'data_integrity':integrity,'records':records,
        'diagnostic_frame':True,'confirmation_accessed':False,'source_sha256':a.wf.sha(a.HERE/'loss_diagnosis.py')})

if __name__=='__main__':main()
