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
from collections import Counter
from pathlib import Path

from data_manifest import split_for, traversal_id

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
    with path.open(newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader, None)
        if header is None:
            raise RuntimeError(f"empty {kind} file: {path}")
        expected = 8 if kind == "vo" else 14
        rows = 0
        first_timestamp = None
        previous_timestamp = None
        for line_number, row in enumerate(reader, start=2):
            if not row:
                continue
            if len(row) < expected:
                raise RuntimeError(
                    f"{kind} schema has {len(row)} columns, expected at least "
                    f"{expected}: {path}:{line_number}"
                )
            try:
                timestamp = int(row[0])
                for value in row[1:expected]:
                    float(value)
            except ValueError as error:
                raise RuntimeError(
                    f"non-numeric {kind} row: {path}:{line_number}"
                ) from error
            if timestamp <= 0 or (
                previous_timestamp is not None and timestamp <= previous_timestamp
            ):
                raise RuntimeError(
                    f"non-increasing {kind} timestamp: {path}:{line_number}"
                )
            if first_timestamp is None:
                first_timestamp = timestamp
            previous_timestamp = timestamp
            rows += 1
    if rows < 2 or first_timestamp is None:
        raise RuntimeError(f"insufficient {kind} rows: {path}")
    return {"header": header, "rows": rows, "first_timestamp": first_timestamp}


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
    exclusions = []
    for name in sorted(set(vo) ^ set(rtk)):
        exclusions.append({
            "traversal": name,
            "reason": "missing_rtk" if name in vo else "missing_vo",
        })
    records = []
    for name in eligible:
        try:
            vo_info = inspect_csv(vo[name], "vo")
            rtk_info = inspect_csv(rtk[name], "rtk")
        except RuntimeError as error:
            exclusions.append({"traversal": name, "reason": str(error)})
            continue
        records.append({
            "traversal": name,
            "split": split_for(name),
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
    destination.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({
        "manifest": str(destination),
        "eligible_count": len(records),
        "split_counts": manifest["split_counts"],
        "exclusions": exclusions,
    }, indent=2))


if __name__ == "__main__":
    main()
