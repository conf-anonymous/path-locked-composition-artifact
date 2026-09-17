"""Complete eight-arm segment-relative evaluation and geometric invariant audits."""
import argparse
from collections import defaultdict
import json
import os
import time
import traceback

import numpy as np
import torch

import acquire as a
import engine as e
import full_frontend as full
import workflow as w
from frontend_smoke import save


def summarize(records):
    output={}
    for kind,groups in (('distance',[str(n) for n in range(100,801,100)]),('fixed',list(map(str,e.AUDIT_LENGTHS)))):
        for group in groups:
            windows=[r for r in records if r['kind']==kind and r['group']==group]
            valid=[r for r in windows if r['valid']]
            item=dict(potential=len(windows),evaluated=len(valid),coverage=len(valid)/len(windows) if windows else None,arms={})
            if valid:
                item['edge_count_range']=[min(r['edges'] for r in valid),max(r['edges'] for r in valid)]
                item['beyond_training_length_count']=sum(r['edges']>256 for r in valid)
                item['duration_seconds_range']=[min(r['duration_seconds'] for r in valid),max(r['duration_seconds'] for r in valid)]
                for arm in e.ARMS:
                    item['arms'][arm]={key:float(np.mean([r['arms'][arm][key] for r in valid]))
                        for key in ('translation_m','rotation_deg')+ (('translation_percent','rotation_deg_per_m') if kind=='distance' else ())}
            output[f'{kind}_{group}']=item
    return output


@torch.no_grad()
def evaluate_one(seed,sequence):
    w.require_access(sequence);w.verify_seal();w.completed_fits()
    split=a.ROLES[sequence]
    folder=e.RUNS/'evaluation'/split/f'seed{seed}'/sequence
    folder.mkdir(parents=True,exist_ok=True)
    destination=folder/'completed.json';stream=folder/'windows.jsonl'
    if destination.exists():
        report=w.json_read(destination);w.validate_hashes(report['dependencies'])
        if report['seal']!=a.sha(e.RUNS/'implementation_seal.json'):
            raise RuntimeError('Stale evaluation')
        return report
    if stream.exists() or (folder/'failure.json').exists():
        raise RuntimeError('Partial/failed evaluation retained; no automatic repeat')
    started=time.perf_counter();data=e.Sequence(sequence)
    cal=w.load_model(seed,'calibrated');heads={arm:w.load_model(seed,arm) for arm in e.HEADS}
    windows=list(data.evaluation_windows());buckets=defaultdict(list)
    for index,window in enumerate(windows):
        if window['valid']:
            buckets[window['last']-window['first']].append(index)
    minimum=float('inf');timings={arm:0. for arm in e.ARMS}
    # Exclusive claim before predictions; failures never silently replace output.
    try:
        with stream.open('x') as out:
            for length,indices in sorted(buckets.items()):
                for offset in range(0,len(indices),64):
                    batch_indices=indices[offset:offset+64]
                    leaves=torch.stack([data.window(windows[i]['first'],length)[0] for i in batch_indices])
                    truth=np.stack([np.linalg.solve(data.truth[windows[i]['first']],data.truth[windows[i]['last']]) for i in batch_indices])
                    before=time.perf_counter();raw=e.product(leaves);raw_seconds=time.perf_counter()-before
                    before=time.perf_counter();cal_leaves,norm=e.calibrated_leaves(cal,leaves);corrected=e.product(cal_leaves)
                    calibration_seconds=time.perf_counter()-before;minimum=min(minimum,norm)
                    for arm in e.ARMS:
                        before=time.perf_counter()
                        pred,norm=e.forward(arm,leaves,cal,heads.get(arm),cached=(raw,corrected))
                        timings[arm]+=time.perf_counter()-before
                        if arm!='identity':timings[arm]+=raw_seconds
                        if arm not in ('identity','raw'):timings[arm]+=calibration_seconds
                        minimum=min(minimum,norm)
                        translation,rotation=e.errors(pred,truth)
                        for j,index in enumerate(batch_indices):
                            window=windows[index];first,last=window['first'],window['last']
                            window.update(edges=length,duration_seconds=float(data.times[last]-data.times[first]),
                                reference_path_length_m=float(np.linalg.norm(np.diff(data.truth[first:last+1,:3,3],axis=0),axis=1).sum()),
                                beyond_training_length=length>256)
                            value=dict(translation_m=float(translation[j]),rotation_deg=float(rotation[j]),motor=pred[j].tolist())
                            if window['kind']=='distance':
                                value.update(translation_percent=100*float(translation[j])/window['length_m'],
                                             rotation_deg_per_m=float(rotation[j])/window['length_m'])
                            window.setdefault('arms',{})[arm]=value
                if length%50==0:
                    print('evaluated',seed,sequence,'edge length',length,'windows',len(indices),flush=True)
            for window in windows:
                out.write(json.dumps(window,allow_nan=False)+'\n')
            out.flush();os.fsync(out.fileno())
        audits={};cal64=w.load_model(seed,'calibrated',torch.float64);gate64=w.load_model(seed,'mergeable',torch.float64)
        for length in e.AUDIT_LENGTHS:
            starts=data.starts[length][:20]
            if not len(starts):
                audits[str(length)]=dict(eligible=0,evaluated=0,passed=True,status='no eligible window')
                continue
            leaves=torch.stack([data.window(int(start),length,torch.float64)[0] for start in starts])
            audits[str(length)]=dict(e.audit_window(leaves,cal64,gate64),eligible=len(data.starts[length]),
                                    evaluated=len(starts),starts=starts.tolist())
        dependencies=[stream,e.RUNS/'implementation_seal.json']
        for arm in e.MODULES:
            dependencies.extend((e.RUNS/'fits'/f'seed{seed}'/arm/name) for name in ('completed.json','step5000.pt'))
        source_folder=full.OUTPUT if split=='train' else e.RUNS/'frontend'
        dependencies.extend((source_folder/f'{sequence}.json',source_folder/f'{sequence}.frames.jsonl',
            a.RAW/'dataset/poses'/f'{sequence}.txt',a.RAW/'dataset/sequences'/sequence/'times.txt',
            a.RAW/'dataset/sequences'/sequence/'calib.txt'))
        w.verify_seal()
        report=dict(seed=seed,sequence=sequence,split=split,arms=list(e.ARMS),summary=summarize(windows),audits=audits,
            minimum_preprojection_real_norm=minimum,frontend_admitted=data.receipt['validation']['admitted'],
            technical_pass=data.receipt['validation']['admitted'] and all(r['passed'] for r in audits.values()) and minimum>1e-6,
            fixed_window_coverage={str(length):dict(potential_starts=max(0,data.n-length),
                valid_starts=len(data.starts[length]),evaluated=len(data.starts[length][::10])) for length in e.AUDIT_LENGTHS},
            seconds=time.perf_counter()-started,model_seconds=timings,
            timing_scope='CPU serial inference diagnostics including shared product costs; excludes frontend/metric/audit and is not a speedup benchmark',
            seal=a.sha(e.RUNS/'implementation_seal.json'),dependencies=w.hash_map(dependencies))
        save(destination,report);print('Evaluation complete',seed,sequence,'technical_pass=',report['technical_pass'],flush=True)
        return report
    except Exception:
        save(folder/'failure.json',dict(seed=seed,sequence=sequence,error=traceback.format_exc()))
        raise


def aggregate(split):
    sequences=[name for name,role in a.ROLES.items() if role==split]
    reports={(seed,seq):w.json_read(e.RUNS/'evaluation'/split/f'seed{seed}'/seq/'completed.json')
             for seed in e.SEEDS for seq in sequences}
    for r in reports.values():w.validate_hashes(r['dependencies'])
    output=dict(split=split,sequences=sequences,seeds=list(e.SEEDS),primary={},contrasts={},
                technical_pass=all(r['technical_pass'] for r in reports.values()),
                scope='Descriptive equal-sequence/equal-seed results; no population significance inference')
    for arm in e.ARMS:
        values={seq:[reports[seed,seq]['summary']['distance_100']['arms'][arm]['translation_percent'] for seed in e.SEEDS]
                for seq in sequences}
        output['primary'][arm]=dict(per_sequence_seed=values,mean=float(np.mean(list(values.values()))))
    comparisons=('raw','identity','feature_direct','feature_residual')
    for arm in comparisons:
        ga=np.array(list(output['primary']['mergeable']['per_sequence_seed'].values()))
        other=np.array(list(output['primary'][arm]['per_sequence_seed'].values()))
        output['contrasts'][arm]=dict(mean_difference_percentage_points=float((ga-other).mean()),
            relative_improvement=float((other.mean()-ga.mean())/other.mean()) if other.mean()>0 else None,
            each_sequence_improves=bool(np.all(ga.mean(1)<other.mean(1))),seeds_improving=int(np.sum(ga.mean(0)<other.mean(0))))
    rotation={arm:float(np.mean([r['summary']['distance_100']['arms'][arm]['rotation_deg_per_m'] for r in reports.values()]))
              for arm in ('raw','mergeable')}
    output['rotation_guard']=dict(means=rotation,passed=rotation['mergeable']<=1.10*rotation['raw'])
    output['descriptive_support_criterion_met']=output['technical_pass'] and output['rotation_guard']['passed'] and all(
        r['relative_improvement'] is not None and r['relative_improvement']>=.05 and r['each_sequence_improves']
        and r['seeds_improving']>=4 for r in output['contrasts'].values())
    save(e.RUNS/'evaluation'/split/'aggregate.json',output)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('split',choices=['train','dev','study_holdout']);args=parser.parse_args()
    e.configure()
    for seed in e.SEEDS:
        for sequence,role in a.ROLES.items():
            if role==args.split:evaluate_one(seed,sequence)
    if not (e.RUNS/'evaluation'/args.split/'aggregate.json').exists():aggregate(args.split)
