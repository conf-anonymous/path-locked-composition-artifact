"""Joint development/release/one-shot confirmation entry point for Oxford 020+031."""
from __future__ import annotations

import argparse
import os
from dataclasses import asdict

import torch

import oxford_workflow as wf
import analysis
import run as extension


def expected_names(split):
    return {r["traversal"] for r in wf.read(wf.MANIFEST)["records"] if r["split"] == split}


def validate_original(result):
    wf.finite_tree(result)
    ox = wf.original()
    if result["split"] != "dev" or result["config"] != asdict(ox.Config()):
        raise RuntimeError("original configuration/split mismatch")
    if result["fingerprint"] != wf.fingerprint() or set(result["arms"]) != set(ox.ARMS):
        raise RuntimeError("original provenance/arm mismatch")
    if set(result["baselines"]) != set(analysis.LENGTHS):
        raise RuntimeError("original baseline horizon missing")
    names = expected_names("dev")
    for length in analysis.LENGTHS:
        baseline = result["baselines"][length]
        analysis.validate_metric(baseline["raw_motor"], names)
        analysis.validate_metric(baseline["raw_motor_moving"], allow_empty=True)
        if not baseline["exact_audit"]["passes"]:
            raise RuntimeError("raw numerical audit failed")
    for arm, seeds in result["arms"].items():
        if set(seeds) != set(map(str, range(5))):
            raise RuntimeError("original seed coverage missing")
        for seed, record in seeds.items():
            if set(record["lengths"]) != set(analysis.LENGTHS) or record["fingerprint"] != wf.fingerprint():
                raise RuntimeError("original horizon/provenance mismatch")
            checkpoint = ox.RUNS / f"{arm}_seed{seed}.pt"
            meta = {"arm": arm, "seed": int(seed), "config": asdict(ox.Config()), **wf.fingerprint()}
            wf.checked_checkpoint(checkpoint, meta)
            if record["checkpoint_sha256"] != wf.sha(checkpoint):
                raise RuntimeError("original checkpoint/result mismatch")
            for length, row in record["lengths"].items():
                for group in ("paths", "moving_paths"):
                    if set(row[group]) != {"left", "right", "balanced", "random"}:
                        raise RuntimeError("original evaluation paths missing")
                    for metric in row[group].values():
                        analysis.validate_metric(metric, names if group == "paths" else None,
                                                 allow_empty=(group == "moving_paths"))
    audit = wf.load_module("oxford_original_gate", wf.ORIGINAL / "audit_results.py")
    return audit.audit(result)


def validate_extension_artifacts(result):
    analysis.validate_extension(result, expected_names("dev"))
    if result["split"] != "dev" or result["fingerprint"] != wf.fingerprint() or result["config"] != extension.HEAD_CONFIG:
        raise RuntimeError("extension provenance/configuration mismatch")
    for seed in range(5):
        _, calibrator_sha = extension.calibrator_for(seed, extension.ox.Config())
        record = result["seeds"][str(seed)]
        if record["calibrator_sha256"] != calibrator_sha or record["fingerprint"] != wf.fingerprint():
            raise RuntimeError("extension calibrator/result mismatch")
        for arm in extension.HEADS:
            path = extension.RUNS / f"{arm}_seed{seed}.pt"
            saved = wf.checked_checkpoint(path, extension.head_meta(arm, seed, calibrator_sha))
            if saved["training"]["steps"] != 5000 or record["head_sha256"][arm] != wf.sha(path):
                raise RuntimeError("extension fit/result mismatch")


def development(device):
    wf.require_development_ready()
    original_path = wf.ORIGINAL / "results_development.json"
    if not original_path.exists():
        result = extension.ox.run("dev", extension.ox.Config(), torch.device(device))
        wf.save(original_path, result)
    else:
        validate_original(wf.read(original_path))
    extension_path = wf.HERE / "results_development.json"
    if not extension_path.exists():
        extension.run("dev", device)
    else:
        validate_extension_artifacts(wf.read(extension_path))
    analysis_path = wf.HERE / "analysis_development.json"
    if not analysis_path.exists():
        wf.save(analysis_path, analysis.analyze(wf.read(extension_path)))
    # Attempted release records a gate failure but never opens confirmation.
    return release()


def release():
    wf.require_development_ready()
    wf.verify_inputs()
    original_result = wf.read(wf.ORIGINAL / "results_development.json")
    extension_result = wf.read(wf.HERE / "results_development.json")
    gate = validate_original(original_result)
    validate_extension_artifacts(extension_result)
    decision = {"fingerprint": wf.fingerprint(), "original_gate": gate,
                "extension_technical_completeness": True, "release_allowed": gate["passes"]}
    decision_path = wf.HERE / "release_decision.json"
    if not decision_path.exists():
        wf.save(decision_path, decision)
    elif wf.read(decision_path) != decision:
        raise RuntimeError("release decision changed")
    if not gate["passes"]:
        print("Original Oxford development gate failed. Confirmation remains unopened.", flush=True)
        return decision
    artifacts = [wf.MANIFEST, wf.SEAL, decision_path,
                 wf.ORIGINAL / "results_development.json", wf.HERE / "results_development.json"]
    for folder in (extension.ox.RUNS, extension.RUNS):
        artifacts.extend(folder.glob("*.pt"))
        artifacts.extend(folder.glob("*.integrity.json"))
        artifacts.extend(folder.glob("evaluation_dev*.json"))
    if sum(p.suffix == ".pt" for p in artifacts) != 50:
        raise RuntimeError("expected exactly 50 frozen original/extension checkpoints")
    wf.save(wf.RELEASE, {"fingerprint": wf.fingerprint(), "original_gate_passes": True,
                        "sha256": {str(p.relative_to(wf.ROOT)): wf.sha(p) for p in sorted(artifacts)}})
    print("Joint release manifest created. Confirmation has NOT been run.", flush=True)
    return decision


def confirm(device):
    wf.verify_implementation()
    if not wf.RELEASE.exists():
        raise RuntimeError("Oxford confirmation remains sealed: release manifest missing")
    record = wf.read(wf.RELEASE)
    if record["fingerprint"] != wf.fingerprint() or not record["original_gate_passes"]:
        raise RuntimeError("invalid joint release")
    for name, checksum in record["sha256"].items():
        if wf.sha(wf.ROOT / name) != checksum:
            raise RuntimeError(f"release-bound artifact changed: {name}")
    # Recompute admission rather than trusting a user-editable boolean alone.
    if not validate_original(wf.read(wf.ORIGINAL / "results_development.json"))["passes"]:
        raise RuntimeError("original gate no longer passes")
    validate_extension_artifacts(wf.read(wf.HERE / "results_development.json"))
    wf.verify_inputs()
    wf.save(wf.STARTED, {"pid": os.getpid(), "release_sha256": wf.sha(wf.RELEASE)})
    wf._active_release = wf.sha(wf.RELEASE)
    wf.require_confirmation_release()
    result = extension.ox.run("confirm", extension.ox.Config(), torch.device(device))
    wf.save(wf.ORIGINAL / "results_confirmation.json", result)
    ext = extension.run("confirm", device)
    wf.save(wf.HERE / "analysis_confirmation.json", analysis.analyze(ext))
    wf.save(wf.HERE / "confirmation_completed.json", {
        "release_sha256": wf.sha(wf.RELEASE),
        "original_result_sha256": wf.sha(wf.ORIGINAL / "results_confirmation.json"),
        "extension_result_sha256": wf.sha(wf.HERE / "results_confirmation.json")})
    wf._active_release = None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phase", choices=("development", "release", "confirm"))
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(1)
    {"development": development, "release": lambda device: release(), "confirm": confirm}[args.phase](args.device)
