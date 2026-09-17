"""Seal a prospective protocol without claiming a completed Oxford runner."""
import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seal", action="store_true")
    args = parser.parse_args()
    original = HERE.parent / "020_oxford_robotcar_vo"
    subprocess.run([sys.executable, str(original / "verify_protocol_seal.py")], check=True)
    paths = [HERE / "PROTOCOL.md", Path(__file__), original / "PROTOCOL_SEAL.json",
             original / "PROTOCOL.md", original / "run.py",
             HERE.parent / "022_eth3d_adaptive_motor/run.py",
             HERE.parent / "028_eth3d_attribution_controls/run.py",
             HERE.parent / "029_eth3d_mergeable_motor_gate/run.py"]
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    destination = HERE / "protocol_seal.json"
    if args.seal:
        data = ROOT / "data/raw/oxford_robotcar"
        if (data / "manifest.json").exists() or list(data.rglob("*.csv")):
            raise RuntimeError("Oxford data are already present; pre-acquisition seal prohibited")
        record = {"created_utc": datetime.now(timezone.utc).isoformat(),
                  "status": "protocol and existing architectures sealed; Oxford-specific runner pending",
                  "authority": "local integrity record, not external preregistration",
                  "sha256": hashes}
        with destination.open("x") as stream:
            json.dump(record, stream, indent=2)
            stream.write("\n")
    sealed = json.loads(destination.read_text())
    # Preserve the actual pre-acquisition chain; the schema/access amendment is
    # separately verified above, including this current verifier's own hash.
    snapshots = original / "pre_amendment_2026-09-09"
    originals = dict(hashes)
    originals[str((original / "run.py").relative_to(ROOT))] = sha(snapshots / "run.py")
    originals[str(Path(__file__).relative_to(ROOT))] = sha(snapshots / "extension_seal_protocol.py")
    assert sealed["sha256"] == originals, "prospective dependencies changed"
    print(json.dumps({"passes": True, "status": sealed["status"],
                      "confirmation_ready": False}, indent=2))


if __name__ == "__main__":
    main()
