"""Scoped read-only replay with an explicit, verified anonymity omission."""
import argparse
import importlib.util
import json
import shutil
import tempfile
from pathlib import Path

import verify_oxford_exclusion as verifier

ROOT=Path(__file__).resolve().parent
FOLDER=ROOT/'iclr-2027-composition/oxford-audit-2026-09-10'


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-root',type=Path,help='Existing official Oxford CSV root; reads development files only')
    args=parser.parse_args()
    verifier.verify(ROOT)
    spec=importlib.util.spec_from_file_location('oxford_matrix_replay',FOLDER/'audit.py')
    audit=importlib.util.module_from_spec(spec); spec.loader.exec_module(audit)
    if args.data_root is not None:
        audit.wf.DATA=args.data_root.resolve()
    # Only the audit's two source checks change; no outcome function is called.
    audit.wf.verify_implementation=lambda:verifier.verify_sources(ROOT)
    with tempfile.TemporaryDirectory(prefix='oxford-recorded-reference-') as name:
        scratch=Path(name)
        shutil.copytree(FOLDER/'reference',scratch/'reference')
        audit.HERE=scratch
        audit.main()
        observed=json.loads((scratch/'audit_results.json').read_text())
        expected=json.loads((FOLDER/'audit_results.json').read_text())
        assert observed==expected, 'Reference audit changed; preserve diagnostic output for investigation.'
    print('PASS: full recorded-development matrix replay matches archived audit; confirmation untouched.')


if __name__=='__main__':main()
