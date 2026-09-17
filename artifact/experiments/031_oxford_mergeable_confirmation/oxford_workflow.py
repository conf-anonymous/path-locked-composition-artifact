"""Immutable artifacts, provenance chain, and shared Oxford admission checks."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ORIGINAL = HERE.parent / "020_oxford_robotcar_vo"
DATA = ROOT / "data/raw/oxford_robotcar"
MANIFEST = DATA / "manifest.json"
SEAL = HERE / "implementation_seal.json"
RELEASE = HERE / "confirmation_release.json"
STARTED = HERE / "confirmation_started.json"
COMPLETED = HERE / "confirmation_completed.json"
_active_release = None


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, record):
    with Path(path).open("x") as stream:
        json.dump(record, stream, indent=2, allow_nan=False)
        stream.write("\n")


def load_module(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def original():
    sys.path.insert(0, str(ORIGINAL))
    return load_module("oxford_original_campaign", ORIGINAL / "run.py")


def verify_history():
    old = ORIGINAL / "pre_amendment_2026-09-09"
    for name, expected in read(ORIGINAL / "PROTOCOL_SEAL.json")["sha256"].items():
        path = (old if name in ("run.py", "prepare_data.py") else ORIGINAL) / name
        if sha(path) != expected:
            raise RuntimeError(f"original Oxford seal mismatch: {name}")
    for name, expected in read(HERE / "protocol_seal.json")["sha256"].items():
        path = ROOT / name
        if path == ORIGINAL / "run.py":
            path = old / "run.py"
        elif path == HERE / "seal_protocol.py":
            path = old / "extension_seal_protocol.py"
        if sha(path) != expected:
            raise RuntimeError(f"original extension seal mismatch: {name}")
    amendment = read(ORIGINAL / "AMENDMENT_SEAL_2026-09-09.json")
    if sha(ORIGINAL / "PROTOCOL_SEAL.json") != amendment["original_seal_sha256"]:
        raise RuntimeError("original seal JSON changed")
    if sha(HERE / "protocol_seal.json") != amendment["extension_original_seal_sha256"]:
        raise RuntimeError("original extension seal JSON changed")
    replacements = {"run.py": "original_run.py", "execution_guard.py": "execution_guard.py",
                    "verify_protocol_seal.py": "verify_protocol_seal.py",
                    "test_schema_and_guard.py": "test_schema_and_guard.py"}
    for name, expected in amendment["sha256"].items():
        path = ROOT / name
        if path.parent == ORIGINAL and path.name in replacements:
            path = HERE / "pre_runner_2026-09-09" / replacements[path.name]
        elif path == ORIGINAL / "prepare_data.py":
            path = HERE / "pre_alignment_2026-09-09/prepare_data.py"
        elif path == MANIFEST:
            path = HERE / "pre_alignment_2026-09-09/manifest.json"
        if sha(path) != expected:
            raise RuntimeError(f"schema amendment history mismatch: {name}")
    prior = HERE / "pre_alignment_2026-09-09"
    for name, expected in read(prior / "implementation_seal.json")["sha256"].items():
        path = ROOT / name
        if path in (ORIGINAL / "prepare_data.py", ORIGINAL / "test_schema_and_guard.py",
                    HERE / "oxford_workflow.py", MANIFEST):
            path = prior / path.name
        if sha(path) != expected:
            raise RuntimeError(f"pre-alignment implementation history mismatch: {name}")


def source_paths():
    paths = [MANIFEST, ORIGINAL / "PROTOCOL_SEAL.json",
             ORIGINAL / "AMENDMENT_SEAL_2026-09-09.json", HERE / "protocol_seal.json",
             HERE / "engineering_tests_final.json"]
    paths.extend([HERE / "pre_alignment_2026-09-09/manifest.json",
                  HERE / "pre_alignment_2026-09-09/implementation_seal.json"])
    for folder in (ORIGINAL, HERE):
        paths.extend(folder.glob("*.py"))
        paths.extend(folder.glob("*.md"))
        paths.extend(folder.glob("pre_*/*.py"))
    # Include the complete experiment-module import chain and unchanged models.
    for folder in ("017_tum_motor_composition", "018_tum_residual_motor",
                   "019_tum_long_horizon_confirmation", "021_eth3d_rgbd_motor",
                   "022_eth3d_adaptive_motor", "023_eth3d_family_heldout",
                   "028_eth3d_attribution_controls", "029_eth3d_mergeable_motor_gate"):
        paths.extend((HERE.parent / folder).glob("*.py"))
    return sorted(set(paths))


def verify_implementation():
    if not SEAL.exists():
        raise RuntimeError("Oxford outcome inspection is not enabled: implementation seal missing")
    verify_history()
    sealed = read(SEAL)
    if not sealed.get("eth3d_engineering_tests_passed"):
        raise RuntimeError("public-recorded-data engineering tests not attested")
    for name, expected in sealed["sha256"].items():
        if sha(ROOT / name) != expected:
            raise RuntimeError(f"active Oxford implementation mismatch: {name}")
    if set(sealed["sha256"]) != {str(p.relative_to(ROOT)) for p in source_paths()}:
        raise RuntimeError("active source inventory changed")
    return sealed


def fingerprint():
    return {"implementation_sha256": sha(SEAL), "manifest_sha256": sha(MANIFEST)}


def verify_inputs():
    manifest = read(MANIFEST)
    for record in manifest["records"]:
        for kind in ("vo", "rtk"):
            if sha(DATA / record[kind]["path"]) != record[kind]["sha256"]:
                raise RuntimeError(f"input changed: {record['traversal']} {kind}")
    item = manifest["ins_extrinsic"]
    if sha(DATA / item["path"]) != item["sha256"]:
        raise RuntimeError("extrinsic changed")


def require_development_ready():
    verify_implementation()
    if RELEASE.exists() or STARTED.exists():
        raise RuntimeError("development is frozen after joint release")


def require_confirmation_release():
    # Denial occurs before any data read when no release or start marker exists.
    if not RELEASE.exists() or not STARTED.exists():
        raise RuntimeError("Oxford confirmation remains sealed: no admitted joint invocation")
    if COMPLETED.exists() or _active_release is None:
        raise RuntimeError("Oxford confirmation remains sealed: no active one-shot invocation")
    verify_implementation()
    release, start = read(RELEASE), read(STARTED)
    if (start["pid"] != os.getpid() or start["release_sha256"] != sha(RELEASE)
            or _active_release != sha(RELEASE)):
        raise RuntimeError("Oxford confirmation remains sealed: use the single joint invocation")
    if release["fingerprint"] != fingerprint():
        raise RuntimeError("release provenance changed")
    for name, expected in release["sha256"].items():
        if sha(ROOT / name) != expected:
            raise RuntimeError(f"frozen development artifact changed: {name}")


def finite_tree(value):
    if isinstance(value, float) and not math.isfinite(value):
        raise RuntimeError("nonfinite recorded result")
    if isinstance(value, dict):
        for child in value.values():
            finite_tree(child)
    elif isinstance(value, list):
        for child in value:
            finite_tree(child)


def checked_checkpoint(path, expected_meta, device="cpu"):
    import torch
    sidecar = path.with_suffix(".integrity.json")
    record = read(sidecar)
    if record["sha256"] != sha(path) or record["meta"] != expected_meta:
        raise RuntimeError(f"checkpoint integrity mismatch: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint["meta"] != expected_meta:
        raise RuntimeError(f"checkpoint metadata mismatch: {path}")
    for group in (checkpoint.get("state_dict", {}), checkpoint.get("head", {})):
        if any(not bool(t.isfinite().all()) for t in group.values()):
            raise RuntimeError(f"nonfinite checkpoint: {path}")
    return checkpoint


def save_checkpoint(path, checkpoint):
    import torch
    with path.open("xb") as stream:
        torch.save(checkpoint, stream)
    save(path.with_suffix(".integrity.json"), {"sha256": sha(path), "meta": checkpoint["meta"]})
