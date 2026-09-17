"""Preseal timing/backward checks on real train windows; zero optimizer updates."""
import time
import torch
import engine as e
from frontend_smoke import save


def main():
    e.configure();e.RUNS.mkdir(parents=True,exist_ok=True)
    if (e.RUNS/'implementation_seal.json').exists():
        raise RuntimeError('Preflight is a before-seal action')
    data=e.TrainingData();sequence=data.sequences['00'];results=[]
    for arm in e.MODULES:
        for length in e.LENGTHS:
            leaves=torch.stack([sequence.window(start,length)[0] for start in range(64)])
            targets=torch.stack([sequence.window(start,length)[1] for start in range(64)])
            model=e.make_module(arm,0);cal=model if arm=='calibrated' else e.make_module('calibrated',0)
            if arm!='calibrated':cal.requires_grad_(False)
            times=[]
            for _ in range(3):
                model.zero_grad(set_to_none=True);started=time.perf_counter()
                pred,norm=e.forward(arm,leaves,cal,model)
                value=e.loss(pred,targets,data.scale);value.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                times.append(time.perf_counter()-started)
            row=dict(arm=arm,length=length,median_forward_backward_seconds=sorted(times)[1],
                     observed_times_seconds=times,finite_loss=bool(torch.isfinite(value)),minimum_norm=norm)
            results.append(row);print(row,flush=True)
    estimate=sum(r['median_forward_backward_seconds'] for r in results)/len(e.LENGTHS)*5000*len(e.SEEDS)
    save(e.RUNS/'preflight.json',dict(results=results,estimated_30_fit_compute_hours=estimate/3600,
         estimate_scope='Initial recorded sequence-00 batches; excludes sampling/checkpoint/evaluation and is not a completion promise',
         optimizer_updates=0,train_sequence='00',development_accessed=False,heldout_accessed=False))
    print('Estimated fitting compute hours:',estimate/3600,flush=True)


if __name__=='__main__':main()
