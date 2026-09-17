"""Inventory official Oxford RobotCar VO and RTK files before training.

This script does not download, interpolate, or transform observations. It
validates the locally extracted official files and freezes their checksums,
eligibility, exclusions, and whole-traversal split in ``manifest.json``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import zipfile
from collections import Counter
from pathlib import Path

from data_manifest import split_for, traversal_id
from pose_csv import recorded_rows, VO_HEADER, RTK_HEADER
from alignment_metadata import inspect_alignment

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "raw" / "oxford_robotcar"
MANIFEST = DATA / "manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def candidate_csvs(root: Path, kind: str) -> dict[str, Path]:
    candidates: dict[str, list[Path]] = {}
    for path in sorted(root.rglob("*.csv")):
        name = traversal_id(path)
        if name is None:
            continue
        lower = path.name.lower()
        if kind == "vo" and lower != "vo.csv" and not lower.endswith("_vo.csv"):
            continue
        # The RTK archive has changed directory/filename wrappers across
        # releases. Every CSV below the dedicated rtk root is eligible for
        # schema validation; traversal IDs still must be unambiguous.
        candidates.setdefault(name, []).append(path)
    duplicates = {name: paths for name, paths in candidates.items() if len(paths) != 1}
    if duplicates:
        rendered = {name: [str(path) for path in paths] for name, paths in duplicates.items()}
        raise RuntimeError(f"ambiguous {kind} files: {rendered}")
    return {name: paths[0] for name, paths in candidates.items()}


def inspect_csv(path: Path, kind: str) -> dict:
    rows = 0
    first_timestamp = None
    zones = set()
    for values in recorded_rows(path, kind):
        if first_timestamp is None:
            first_timestamp = values[0]
        if kind == "rtk":
            zones.add(values[7])
        rows += 1
    if rows < 2 or first_timestamp is None:
        raise RuntimeError(f"insufficient {kind} rows: {path}")
    result = {"header": VO_HEADER if kind == "vo" else RTK_HEADER,
              "rows": rows, "first_timestamp": first_timestamp}
    if kind == "rtk":
        result["utm_zones"] = sorted(zones)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DATA)
    args = parser.parse_args()
    root = args.data_root.resolve()
    vo = candidate_csvs(root / "vo", "vo")
    rtk = candidate_csvs(root / "rtk", "rtk")
    extrinsic = root / "extrinsics" / "ins.txt"
    if not extrinsic.exists():
        raise FileNotFoundError(
            f"missing official SDK extrinsic {extrinsic}; copy extrinsics/ins.txt unchanged"
        )
    eligible = sorted(set(vo) & set(rtk))
    if not eligible:
        raise FileNotFoundError(
            f"no matched official VO/RTK traversal files under {root}; see README.md"
        )
    exclusions = [
        {"source_path": str(path.relative_to(root)), "reason": "malformed_traversal_id"}
        for path in sorted((root / "rtk").rglob("*.csv"))
        if traversal_id(path) is None
    ]
    for name in sorted(set(vo) ^ set(rtk)):
        exclusions.append({
            "traversal": name,
            "reason": "missing_rtk" if name in vo else "missing_vo",
        })
    records = []
    acquisition_path = root / "acquisition-verified-2026-09-09.json"
    acquisition = json.loads(acquisition_path.read_text())
    rtk_archive = root / "official_archives/rtk.zip"
    if sha256(rtk_archive) != acquisition["rtk_archive_sha256"]:
        raise RuntimeError("RTK archive changed after acquisition verification")
    with zipfile.ZipFile(rtk_archive) as archive:
        rtk_hashes = {name: hashlib.sha256(archive.read(name)).hexdigest()
                      for name in archive.namelist() if name.endswith("/rtk.csv")}
    archive_records = {item["file"].removesuffix("_vo.tar"): item
                       for item in acquisition["archives"]}
    for name in eligible:
        provenance = archive_records[name]
        if sha256(rtk[name]) != rtk_hashes[f"{name}/rtk.csv"]:
            raise RuntimeError(f"extracted RTK bytes differ from original ZIP: {name}")
        archive = root / "official_archives" / provenance["file"]
        if sha256(archive) != provenance["sha256"] or sha256(vo[name]) != provenance["csv_sha256"]:
            raise RuntimeError(f"verified VO acquisition bytes changed: {name}")
        digest = hashlib.md5(archive.read_bytes()).hexdigest()
        if digest != provenance["official_md5"]:
            raise RuntimeError(f"official VO archive MD5 mismatch: {name}")
        try:
            vo_info = inspect_csv(vo[name], "vo")
            rtk_info = inspect_csv(rtk[name], "rtk")
        except RuntimeError as error:
            exclusions.append({"traversal": name, "reason": str(error)})
            continue
        alignment = inspect_alignment(vo[name], rtk[name])
        if not alignment["supports_l32"]:
            exclusions.append({"traversal": name, "reason": "no_valid_contiguous_anchor_run",
                               "alignment_metadata": alignment})
            continue
        records.append({
            "traversal": name,
            "split": split_for(name),
            "alignment_metadata": alignment,
            "vo": {
                "path": str(vo[name].relative_to(root)),
                "bytes": vo[name].stat().st_size,
                "sha256": sha256(vo[name]),
                **vo_info,
            },
            "rtk": {
                "path": str(rtk[name].relative_to(root)),
                "bytes": rtk[name].stat().st_size,
                "sha256": sha256(rtk[name]),
                **rtk_info,
            },
        })
    if not records:
        raise RuntimeError("no matched VO/RTK traversal passed the frozen schema rules")
    counts = Counter(record["split"] for record in records)
    manifest = {
        "protocol": "Oxford RobotCar VO-to-RTK v1, frozen 2026-09-03",
        "source": "https://robotcar-dataset.robots.ox.ac.uk/",
        "schema_amendment": "AMENDMENT_2026-09-09.md",
        "alignment_amendment": "031_oxford_mergeable_confirmation/ALIGNMENT_AMENDMENT_2026-09-09.md",
        "acquisition_report_sha256": sha256(acquisition_path),
        "eligible_count": len(records),
        "split_counts": {key: counts[key] for key in ("train", "dev", "confirm")},
        "exclusions": exclusions,
        "ins_extrinsic": {
            "path": str(extrinsic.relative_to(root)),
            "bytes": extrinsic.stat().st_size,
            "sha256": sha256(extrinsic),
        },
        "records": records,
    }
    destination = root / "manifest.json"
    rendered = json.dumps(manifest, indent=2) + "\n"
    if destination.exists():
        if destination.read_text() != rendered:
            raise RuntimeError("refusing to replace nonidentical scientific manifest")
    else:
        with destination.open("x") as stream:
            stream.write(rendered)
    print(json.dumps({
        "manifest": str(destination),
        "eligible_count": len(records),
        "split_counts": manifest["split_counts"],
        "exclusions": exclusions,
    }, indent=2))


if __name__ == "__main__":
    main()
