"""Verify completion, unchanged originals, and GA path invariance on recorded data."""
import torch
import investigate as a
import retry

def main():
    torch.set_num_threads(1);d,integrity=retry.load_data();report={}
    for label,folder,marker in [('frame_only',a.HERE,'retry_completed.json'),
                               ('quality_screened',a.HERE/'quality_screened','completed.json')]:
        assert (folder/marker).exists()
        for arm in a.ox.ARMS:
            for seed in a.ox.SEEDS:
                path=folder/'runs'/f'{arm}_seed{seed}.pt'
                assert a.wf.sha(path)==a.wf.read(path.with_suffix('.integrity.json'))['sha256']
                result=a.wf.read(folder/'runs'/f'{arm}_seed{seed}_evaluation.json')
                assert result['checkpoint_sha256']==a.wf.sha(path)
                assert set(result['lengths'])=={'4','8','16','32'}
                for entry in result['lengths'].values():
                    assert set(entry)=={'left','right','balanced',*[f'random_{i}' for i in range(16)]}
        exact={}
        for seed in a.ox.SEEDS:
            path=folder/'runs'/f'ga_calibrated_seed{seed}.pt'
            model=a.ox.OxfordComposer('ga_calibrated')
            model.load_state_dict(torch.load(path,weights_only=False)['state_dict'])
            exact[str(seed)]={str(l):a.ox.exact_path_audit(d,'dev',l,a.ox.Config(),model) for l in a.ox.EVAL_LENGTHS}
            assert all(v['passes'] for v in exact[str(seed)].values())
        report[label]={'all_25_checkpoints_and_evaluations_verified':True,'ga_float64_exactness':exact}
    a.wf.verify_implementation();assert not a.wf.STARTED.exists() and not a.wf.RELEASE.exists()
    a.save(a.HERE/'verification.json',{'confirmation_accessed':False,'original_sources_unchanged':True,
        'data_integrity':integrity,'campaigns':report,'source_sha256':a.wf.sha(a.HERE/'verify_completed.py')})
    print('PASS: 50 checkpoints, 50 complete evaluation records, all GA exactness checks; originals unchanged; confirmation closed')

if __name__=='__main__':main()
