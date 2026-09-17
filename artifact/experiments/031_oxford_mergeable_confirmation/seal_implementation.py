"""Seal implementation only after recorded-data engineering tests, before Oxford outcomes."""
import argparse
import subprocess
import sys
from datetime import datetime, timezone

import oxford_workflow as wf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seal", action="store_true")
    args = parser.parse_args()
    if args.seal:
        if wf.SEAL.exists():
            raise RuntimeError("refusing to replace implementation seal")
        for folder in (wf.ORIGINAL, wf.HERE):
            if list(folder.glob("results_*.json")) or list((folder / "runs").glob("*.pt")):
                raise RuntimeError("cannot claim pre-outcome implementation seal after Oxford execution")
        wf.verify_history()
        subprocess.run([sys.executable, str(wf.HERE / "test_recorded_workflow.py")], check=True)
        subprocess.run([sys.executable, str(wf.ORIGINAL / "test_schema_and_guard.py")], check=True)
        evidence = wf.read(wf.HERE / "engineering_tests_final.json")
        if not evidence["passed"] or evidence["oxford_outcomes_inspected"]:
            raise RuntimeError("invalid engineering evidence")
        wf.save(wf.SEAL, {"created_utc": datetime.now(timezone.utc).isoformat(),
                          "authority": "local post-acquisition, pre-outcome integrity seal; not external preregistration",
                          "eth3d_engineering_tests_passed": True,
                          "sha256": {str(p.relative_to(wf.ROOT)): wf.sha(p) for p in wf.source_paths()}})
    wf.verify_implementation()
    print("Original seals, schema amendment, and active implementation verified. "
          "Development enabled; confirmation requires the joint release workflow.")


if __name__ == "__main__":
    main()
