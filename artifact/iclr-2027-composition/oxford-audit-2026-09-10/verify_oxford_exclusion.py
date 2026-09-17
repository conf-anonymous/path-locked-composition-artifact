"""Read-only verification of excluded Oxford records, not outcome admission.

The anonymous export omits one non-scientific local acquisition helper whose
default path identifies its author. Its historical hash is retained explicitly;
this verifies all available sealed files, NOT the bytes of that omitted file.
Nothing in this module opens confirmation or authorizes a training run.
"""
import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

OMITTED = 'experiments/020_oxford_robotcar_vo/stage_downloads.py'
OMITTED_SHA = 'c879c88f7834bd371b03eb3ec06fea4286cbf1c4522646bdc2db49eba56a81fa'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_sources(root):
    folder = root/'experiments/031_oxford_mergeable_confirmation'
    sealed = read(folder/'implementation_seal.json')
    omitted = []
    export = root/'ANONYMITY_OMISSIONS.json'
    for name, expected in sealed['sha256'].items():
        path = root/name
        if not path.exists() and name == OMITTED and export.exists():
            item = read(export)[name]
            assert expected == OMITTED_SHA == item['original_sha256']
            omitted.append(name)
        else:
            assert sha(path) == expected, name
    if export.exists():
        assert set(read(export)) == set(omitted) == {OMITTED}
    return {'verified_available_files':len(sealed['sha256'])-len(omitted),
            'unverifiable_omitted_acquisition_helpers':omitted}


def verify(root):
    sources = verify_sources(root)
    folder = root/'experiments/031_oxford_mergeable_confirmation'
    sys.path.insert(0,str(folder))
    spec = importlib.util.spec_from_file_location('oxford_joint_readonly',folder/'joint.py')
    joint = importlib.util.module_from_spec(spec); spec.loader.exec_module(joint)
    original = read(root/'experiments/020_oxford_robotcar_vo/results_development.json')
    extension = read(folder/'results_development.json')
    gate = joint.validate_original(original)
    joint.validate_extension_artifacts(extension)
    assert gate == read(folder/'release_decision.json')['original_gate']
    assert not gate['passes'] and not read(folder/'release_decision.json')['release_allowed']
    assert joint.analysis.analyze(extension) == read(folder/'analysis_development.json')
    assert len(list((folder/'runs').glob('*.pt'))) == 25
    assert len(list((root/'experiments/020_oxford_robotcar_vo/runs').glob('*.pt'))) == 25
    for name in ('confirmation_release.json','confirmation_started.json',
                 'confirmation_completed.json','results_confirmation.json'):
        assert not (folder/name).exists(), name
    assert not (root/'experiments/020_oxford_robotcar_vo/results_confirmation.json').exists()
    audit_folder = root/'iclr-2027-composition/oxford-audit-2026-09-10'
    audit = read(audit_folder/'audit_results.json')
    assert sha(audit_folder/'audit.py') == audit['audit_source_sha256']
    assert sha(root/'experiments/020_oxford_robotcar_vo/results_development.json') == audit['source_results_sha256']
    assert audit['fingerprint'] == joint.wf.fingerprint()
    assert audit['status']=='frame_convention_confounded_not_validated_odometry_gains'
    assert not audit['confirmation_accessed'] and not audit['training_performed']
    assert audit['post_outcome'] and audit['diagnostic_not_corrected_protocol']
    for name, expected in audit['sdk_source_sha256'].items():
        assert sha(audit_folder/'reference'/name)==expected
    names = joint.expected_names('dev')
    assert {r['traversal'] for r in audit['records']} == names
    assert len(names)==14
    for record in audit['records']:
        assert record['split']=='dev'
        assert record['vo_rtk_offset_s']['max']<=.1
        assert record['duplicate_selected_rtk_rows']==0
        assert max(record['matrix_motor_relative_max_abs'].values())<1e-5
        for length, metric in record['lengths'].items():
            saved=original['baselines'][length]['raw_motor']['clusters'][record['traversal']]
            assert saved['n']==metric['n'] and abs(saved['translation_mean_m']-metric['raw_m'])<1e-5
    for length, aggregate in audit['aggregate_traversal_equal'].items():
        for key, value in aggregate.items():
            assert abs(sum(r['lengths'][length][key] for r in audit['records'])/14-value)<1e-10
    print(json.dumps({'status':'PASS: excluded records consistent; not validated accuracy evidence',
                      'original_gate_passes':False,'confirmation_unopened':True,**sources},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
    args=parser.parse_args()
    verify(args.root.resolve())
