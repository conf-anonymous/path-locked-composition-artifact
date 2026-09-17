#!/usr/bin/env python3
"""Check retained results, or create a separate copy for experimental reruns."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from verify import verify

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    commands.add_parser("audit", help="Verify all retained outcomes without acquiring data or fitting models")
    workspace = commands.add_parser("workspace", help="Copy the artifact to a new directory without overwriting anything")
    workspace.add_argument("--destination", required=True, type=Path)
    workspace.add_argument("--sources-only", action="store_true",
                           help="Copy code/protocol documentation without historical data, fits, seals or results")
    args = parser.parse_args()
    checked = verify(ROOT)
    print("PASS: exact repository inventory", flush=True)
    if args.action == "workspace":
        destination = args.destination.expanduser().resolve()
        if destination.exists() or destination.is_relative_to(ROOT):
            raise SystemExit("Destination must be absent and outside this repository.")
        if args.sources_only:
            destination.mkdir(parents=True)
            for p in (ROOT / "artifact").rglob("*"):
                if p.is_file() and p.suffix in {".py", ".md", ".toml", ".cpp", ".h", ".c"}:
                    relative = p.relative_to(ROOT / "artifact")
                    if "__pycache__" not in relative.parts and relative.parts[0] != "manuscript":
                        target = destination / relative
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(p, target)
        else:
            shutil.copytree(ROOT / "artifact", destination,
                            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".DS_Store"))
        shutil.copyfile(ROOT / "docs/REPRODUCING.md", destination / "REPRODUCTION_WORKSPACE.md")
        print(json.dumps({"workspace": str(destination), "source_manifest_sha256": checked["manifest_sha256"],
                          "sources_only": args.sources_only,
                          "scope": "Copy only. No datasets downloaded, outputs removed or experiments run."}, indent=2))
        return
    output = ROOT / "verification-output"
    output.mkdir(exist_ok=True)
    receipt = output / f"audit-{time.time_ns()}.json"
    logfile = receipt.with_suffix(".log")
    started = time.monotonic()
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="1",
               MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    command = [sys.executable, "verify_claims.py"]
    with logfile.open("x") as log:
        completed = subprocess.run(command, cwd=ROOT / "artifact", env=env,
                                   stdout=log, stderr=subprocess.STDOUT)
        if completed.returncode == 0:
            completed = subprocess.run(
                [sys.executable, "experiments/040_eth3d_readout_scope_audit/verify.py", "--bootstrap"],
                cwd=ROOT / "artifact", env=env, stdout=log, stderr=subprocess.STDOUT)
    # A successful checker must not silently mutate the reference evidence.
    after = verify(ROOT)
    record = {"status": "PASS" if completed.returncode == 0 else "FAIL",
              "returncode": completed.returncode, "seconds": time.monotonic()-started,
              "package_manifest_sha256": after["manifest_sha256"],
              "reference_files_unchanged": True, "log": logfile.name,
              "scope": "Retained-evidence audits and arithmetic; no fresh training or sensor extraction."}
    with receipt.open("x") as f:
        json.dump(record, f, indent=2); f.write("\n")
    print(logfile.read_text())
    print(json.dumps(record, indent=2))
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
