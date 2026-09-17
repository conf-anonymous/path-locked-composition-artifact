"""Download only official TUM RGB-D recorded ground-truth trajectories."""

from __future__ import annotations

import hashlib
import json
import urllib.request
from pathlib import Path

from data_manifest import SEQUENCES, source_url

ROOT = Path(__file__).resolve().parents[2]
DEST = ROOT / "data" / "raw" / "tum_rgbd_groundtruth"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    records = []
    for index, name in enumerate(SEQUENCES, 1):
        target = DEST / f"{name}.txt"
        url = source_url(name)
        if not target.exists():
            print(f"[{index:02d}/{len(SEQUENCES)}] {name}", flush=True)
            request = urllib.request.Request(url, headers={"User-Agent": "path-locking-research/1.0"})
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = response.read()
            if b"# ground truth trajectory" not in payload[:512].lower():
                raise RuntimeError(f"unexpected response for {name}")
            target.write_bytes(payload)
        records.append({
            "sequence": name,
            "source": url,
            "bytes": target.stat().st_size,
            "sha256": sha256(target),
        })
    manifest = DEST / "checksums.json"
    manifest.write_text(json.dumps(records, indent=2) + "\n")
    print(f"Verified {len(records)} official trajectories; manifest: {manifest}")


if __name__ == "__main__":
    main()
