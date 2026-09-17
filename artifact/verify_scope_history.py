"""Replay unchanged ETH3D scope audit against its actual historical dependencies.

Experiment 030's broad source inventory included then-unused Oxford utilities.
Later Oxford schema/runner work changed three such files. Validate the retained
exact historical snapshots for those entries, never replace their sealed hashes.
"""
import importlib.util
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parent
FOLDER=ROOT/'experiments/030_eth3d_frozen_scope'
sys.path.insert(0,str(FOLDER))
spec=importlib.util.spec_from_file_location('scope_historical_audit',FOLDER/'audit_results.py')
audit=importlib.util.module_from_spec(spec); spec.loader.exec_module(audit)
original_sha=audit.study.study.sha
snapshots={ROOT/f'experiments/020_oxford_robotcar_vo/{name}':
           ROOT/f'experiments/020_oxford_robotcar_vo/pre_amendment_2026-09-09/{name}'
           for name in ('prepare_data.py','run.py','verify_protocol_seal.py')}
seal=json.loads((FOLDER/'seal.json').read_text())
for current,historical in snapshots.items():
    assert original_sha(historical)==seal[str(current.relative_to(ROOT))]


def historical_sha(path):
    return original_sha(snapshots.get(Path(path),Path(path)))


if __name__=='__main__':
    audit.study.study.sha=historical_sha
    audit.main()
    print('Oxford-only metadata dependencies verified against three exact historical snapshots; ETH3D code/results unchanged.')
