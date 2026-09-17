#!/usr/bin/env python3
"""Verify the exact repository inventory without importing research code."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat

MANIFEST = "PACKAGE_MANIFEST.json"
CHUNK = 1 << 20


class VerificationError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise VerificationError(message)


def canonical(name):
    require(isinstance(name, str) and name and "\\" not in name and "\0" not in name,
            "invalid relative path")
    p = PurePosixPath(name)
    require(not p.is_absolute() and ".." not in p.parts and p.as_posix() == name
            and ":" not in p.parts[0], "noncanonical path: " + name)
    return p


def safe_path(root, name):
    p = root
    for part in canonical(name).parts:
        p = p / part
        require(not p.is_symlink(), "symlink is forbidden: " + name)
    require(p.resolve().is_relative_to(root.resolve()), "path escapes repository")
    return p


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def ignored(parts):
    return (parts[0] in {".git", ".venv", "verification-output"}
            or "__pycache__" in parts or parts[-1] == ".DS_Store"
            or parts[-1].endswith((".pyc", ".pyo")))


def members(root):
    found = set()
    for directory, dirs, files in os.walk(root, followlinks=False):
        for name in list(dirs):
            p = Path(directory) / name
            rel = p.relative_to(root)
            require(not p.is_symlink(), "symlink directory: " + rel.as_posix())
            if ignored(rel.parts):
                dirs.remove(name)
        for name in files:
            p = Path(directory) / name
            rel = p.relative_to(root)
            require(not p.is_symlink(), "symlink file: " + rel.as_posix())
            if ignored(rel.parts):
                continue
            require(stat.S_ISREG(p.stat().st_mode), "nonregular file: " + rel.as_posix())
            canonical(rel.as_posix())
            found.add(rel.as_posix())
    return found


def unique_pairs(pairs):
    value = {}
    for key, item in pairs:
        require(key not in value, "duplicate JSON key")
        value[key] = item
    return value


def verify(root):
    root = Path(root).resolve()
    path = safe_path(root, MANIFEST)
    require(path.is_file(), "package manifest is missing")
    document = json.loads(path.read_text(), object_pairs_hook=unique_pairs)
    require(document.get("schema") == "anonymous-research-repository-v1", "wrong manifest schema")
    files = document.get("files")
    require(isinstance(files, dict) and files, "empty file inventory")
    for name, row in files.items():
        canonical(name)
        require(name != MANIFEST and not ignored(PurePosixPath(name).parts), "forbidden inventory member")
        require(set(row) == {"bytes", "sha256", "mode"}, "invalid file metadata")
        require(type(row["bytes"]) is int and row["bytes"] >= 0, "invalid byte size")
        require(row["mode"] in (0o644, 0o755), "invalid file mode")
        require(isinstance(row["sha256"], str) and re.fullmatch(r"[0-9a-f]{64}", row["sha256"]),
                "invalid digest")
    actual = members(root) - {MANIFEST}
    require(actual == set(files), f"inventory differs: missing={sorted(set(files)-actual)}, unexpected={sorted(actual-set(files))}")
    total = 0
    for name, row in files.items():
        p = safe_path(root, name)
        require(p.stat().st_size == row["bytes"], "size differs: " + name)
        require(sha(p) == row["sha256"], "hash differs: " + name)
        # Git preserves executable permission, not every Unix permission bit.
        require(bool(p.stat().st_mode & 0o111) == (row["mode"] == 0o755), "executable mode differs: " + name)
        total += row["bytes"]
    return {"status": "PASS", "files": len(files), "bytes": total,
            "manifest_sha256": sha(path),
            "scope": "File membership, bytes and executable modes; not fresh scientific execution."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    try:
        print(json.dumps(verify(args.root), indent=2))
    except (VerificationError, OSError, ValueError, TypeError, KeyError) as error:
        raise SystemExit("FAIL: " + str(error)) from error


if __name__ == "__main__":
    main()
