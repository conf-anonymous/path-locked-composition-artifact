"""Verify that the pre-acquisition Oxford protocol and implementation are unchanged."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEAL = HERE / "PROTOCOL_SEAL.json"
AMENDMENT = HERE / "AMENDMENT_SEAL_2026-09-09.json"
SNAPSHOTS = HERE / "pre_amendment_2026-09-09"
ROOT = HERE.parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify() -> dict:
    seal = json.loads(SEAL.read_text())
    changed = {"run.py", "prepare_data.py"}
    observed = {name: sha256((SNAPSHOTS if name in changed else HERE) / name)
                for name in seal["sha256"]}
    mismatches = {
        name: {"expected": seal["sha256"][name], "observed": digest}
        for name, digest in observed.items()
        if digest != seal["sha256"][name]
    }
    result = {
        "passes": not mismatches,
        "sealed_date": seal["sealed_date"],
        "data_status_at_seal": seal["data_status_at_seal"],
        "mismatches": mismatches,
    }
    if mismatches:
        raise SystemExit("OXFORD PROTOCOL SEAL BROKEN")
    amendment = json.loads(AMENDMENT.read_text())
    if amendment["original_seal_sha256"] != sha256(SEAL):
        raise SystemExit("ORIGINAL OXFORD SEAL WAS MODIFIED")
    extension_seal = HERE.parent / "031_oxford_mergeable_confirmation/protocol_seal.json"
    if amendment["extension_original_seal_sha256"] != sha256(extension_seal):
        raise SystemExit("ORIGINAL OXFORD EXTENSION SEAL WAS MODIFIED")
    amended_mismatches = {
        name: {"expected": expected, "observed": sha256(ROOT / name)}
        for name, expected in amendment["sha256"].items()
        if sha256(ROOT / name) != expected
    }
    if amended_mismatches:
        raise SystemExit(f"OXFORD AMENDMENT SEAL BROKEN: {amended_mismatches}")
    result["original_sources_verified_via_snapshots"] = sorted(changed)
    result["amendment"] = amendment["scope"]
    result["amendment_passes"] = True
    return result


def main() -> None:
    print(json.dumps(verify(), indent=2))


if __name__ == "__main__":
    main()
