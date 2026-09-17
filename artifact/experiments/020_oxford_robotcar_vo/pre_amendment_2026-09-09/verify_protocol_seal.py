"""Verify that the pre-acquisition Oxford protocol and implementation are unchanged."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SEAL = HERE / "PROTOCOL_SEAL.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    seal = json.loads(SEAL.read_text())
    observed = {name: sha256(HERE / name) for name in seal["sha256"]}
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
    print(json.dumps(result, indent=2))
    if mismatches:
        raise SystemExit("OXFORD PROTOCOL SEAL BROKEN")


if __name__ == "__main__":
    main()
