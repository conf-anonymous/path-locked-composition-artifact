"""Download only the 56 official real ETH3D RGB-D motion-capture sequences."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import urllib.request
import zipfile
from pathlib import Path

from data_manifest import MODALITIES, SEQUENCES, source_url

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "raw" / "eth3d_rgbd"
ARCHIVES = DATA / "archives"
SEQUENCE_ROOT = DATA / "sequences"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    with zipfile.ZipFile(archive) as source:
        for member in source.infolist():
            target = (destination / member.filename).resolve()
            if root not in target.parents and target != root:
                raise RuntimeError(f"unsafe archive member: {member.filename}")
        source.extractall(destination)


def download(url: str, destination: Path) -> None:
    partial = destination.with_suffix(destination.suffix + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "path-locking-research/1.0"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=120) as response:
        append = offset > 0 and getattr(response, "status", None) == 206
        stream = partial.open("ab" if append else "wb")
        with stream:
            shutil.copyfileobj(response, stream, length=1 << 20)
    partial.replace(destination)


def locate_sequence_root(root: Path, name: str) -> Path:
    candidates = [path.parent for path in root.rglob("groundtruth.txt")]
    candidates = [path for path in candidates if (path / "rgb").is_dir() and (path / "depth").is_dir()]
    if len(candidates) != 1:
        raise RuntimeError(f"expected one extracted RGB-D root for {name}, found {candidates}")
    return candidates[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--list-only", action="store_true",
        help="print the frozen official URLs without downloading",
    )
    args = parser.parse_args()
    if args.list_only:
        for name in SEQUENCES:
            for modality in MODALITIES:
                print(source_url(name, modality))
        return
    ARCHIVES.mkdir(parents=True, exist_ok=True)
    SEQUENCE_ROOT.mkdir(parents=True, exist_ok=True)
    records = []
    for index, name in enumerate(SEQUENCES, 1):
        print(f"[{index:02d}/{len(SEQUENCES)}] {name}", flush=True)
        destination = SEQUENCE_ROOT / name
        archive_records = {}
        for modality in MODALITIES:
            archive = ARCHIVES / f"{name}_{modality}.zip"
            if not archive.exists():
                download(source_url(name, modality), archive)
            if not zipfile.is_zipfile(archive):
                raise RuntimeError(f"invalid ZIP response: {archive}")
            with zipfile.ZipFile(archive) as source:
                broken_member = source.testzip()
            if broken_member is not None:
                raise RuntimeError(f"ZIP CRC failure in {archive}: {broken_member}")
            archive_digest = sha256(archive)
            extraction_marker = destination / f".{modality}_extraction_complete"
            if not extraction_marker.exists():
                safe_extract(archive, destination)
                extraction_marker.write_text(archive_digest + "\n")
            elif extraction_marker.read_text().strip() != archive_digest:
                raise RuntimeError(f"archive changed after extraction: {archive}")
            archive_records[modality] = {
                "source": source_url(name, modality),
                "path": str(archive.relative_to(DATA)),
                "bytes": archive.stat().st_size,
                "sha256": archive_digest,
            }
        sequence_root = locate_sequence_root(destination, name)
        records.append({
            "sequence": name,
            "archives": archive_records,
            "root": str(sequence_root.relative_to(DATA)),
        })
    manifest = DATA / "downloads.json"
    manifest.write_text(json.dumps(records, indent=2) + "\n")
    print(
        f"Verified and extracted {len(records) * len(MODALITIES)} official "
        f"archives for {len(records)} sequences; {manifest}"
    )


if __name__ == "__main__":
    main()
