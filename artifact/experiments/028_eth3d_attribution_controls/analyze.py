"""Auditable aggregates and conditional uncertainty; no outcome-based selection."""
from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import torch

import run as study

HERE, ROOT, v = study.HERE, study.ROOT, study.v
FAMILIES = sorted(v.FAMILY_SEQUENCES)


def family_values(clusters, metric):
    return {f: float(np.mean([r[metric] for r in clusters.values() if r["family"] == f]))
            for f in sorted({r["family"] for r in clusters.values()})}


def contrast(a, b):
    delta = a - b  # seeds x families, shared ordering
    values = delta.mean(0)
    rng = np.random.default_rng(280906)
    draws = values[rng.integers(0, len(FAMILIES), (10000, len(FAMILIES)))].mean(1)
    fold_sums = [sum(values[FAMILIES.index(f)] for f in v.FOLD_FAMILIES[k]) for k in range(5)]
    distribution = [np.dot(signs, fold_sums)/16 for signs in itertools.product((-1, 1), repeat=5)]
    return {"mean_delta": float(values.mean()), "conditional_family_bootstrap_ci95": np.quantile(draws, [.025, .975]).tolist(),
            "family_wins": int(np.sum(values < 0)), "seed_deltas": delta.mean(1).tolist(),
            "fold_mean_deltas": [float(fold_sums[k]/len(v.FOLD_FAMILIES[k])) for k in range(5)],
            "fold_block_sign_flip_sensitivity_p": float(np.mean(np.abs(distribution) >= abs(values.mean())-1e-12*max(1., abs(values.mean())))),
            "per_family_deltas": dict(zip(FAMILIES, values.tolist()))}


def duration_summary():
    manifest = json.loads(v.MANIFEST.read_text())
    durations = []
    for record in manifest["records"]:
        arrays = np.load(v.DATA / record["derived_path"], allow_pickle=False)
        valid = arrays["pair_valid"].astype(bool)
        times = arrays["timestamps"]
        start = None
        for index, keep in enumerate(np.append(valid, False)):
            if keep and start is None:
                start = index
            elif not keep and start is not None:
                if index - start >= 32:
                    durations.extend(times[start+32:index+1]-times[start:index-31])
                start = None
    return {"n": len(durations), "min_seconds": float(np.min(durations)),
            "median_seconds": float(np.median(durations)), "max_seconds": float(np.max(durations))}


@torch.inference_mode()
def main():
    torch.set_num_threads(1)
    data = v.ETH3DAll()
    operands, target = data.tensors[32]
    # Constant identity is a prediction on recorded targets, not generated data.
    identity = torch.zeros_like(target)
    identity[:, 0] = 1.
    identity_record = study.metrics(identity, target, data.examples[32])
    original = json.loads((v.HERE / "results.json").read_text())
    direct = json.loads((HERE.parent / "024_eth3d_calibrated_direct_control/results.json").read_text())
    arrays = {metric: {arm: np.empty((5, 16)) for arm in
              ("identity", "original_contextual", "original_direct", "original_calibrated", "raw")}
              for metric in ("translation_mean_m", "rotation_mean_deg")}
    for metric, methods in arrays.items():
        identity_family = family_values(identity_record["clusters"], metric)
        methods["identity"][:] = [identity_family[f] for f in FAMILIES]
        for fold, r in original["folds"].items():
            for seed, record in r["seeds"].items():
                for arm, source in (("original_contextual", record["lengths"]["32"]["contextual_ga"]),
                                    ("original_calibrated", record["lengths"]["32"]["ga_calibrated"]),
                                    ("original_direct", direct["folds"][fold]["seeds"][seed]["lengths"]["32"]),
                                    ("raw", r["baselines"]["32"]["raw_motor"])):
                    for f, value in family_values(source["clusters"], metric).items():
                        methods[arm][int(seed), FAMILIES.index(f)] = value
    result = {"statistical_scope": "Conditional on fitted models, named families and fixed folds; overlapping training sets are dependent. Fold sign flips are sensitivity, not unconditional exact inference.",
              "n_windows": len(target), "identity_sequence_metrics": identity_record,
              "durations": duration_summary(), "metrics": {}}
    new_path = HERE / "results.json"
    if new_path.exists():
        new = json.loads(new_path.read_text())
        assert len(new["records"]) == 25
        for metric, methods in arrays.items():
            for arm in study.ARMS:
                methods[arm] = np.full((5, 16), np.nan)
            for r in new["records"]:
                for arm in study.ARMS:
                    for f, value in family_values(r["arms"][arm]["clusters"], metric).items():
                        methods[arm][r["seed"], FAMILIES.index(f)] = value
            assert all(np.isfinite(a).all() for a in methods.values())
        result["calibrator_reproduction_max"] = max(r["calibrator_reproduction_max"] for r in new["records"])
        result["minimum_real_norm"] = {arm: min(r["arms"][arm]["minimum_real_norm_before_projection"] for r in new["records"]) for arm in study.ARMS}
        result["training_seconds_total"] = sum(r["calibrator_training"]["seconds"] + sum(a["training_seconds"] for a in r["arms"].values()) for r in new["records"])
        result["parameter_counts"] = {arm: new["records"][0]["arms"][arm]["parameters"] for arm in study.ARMS}
        result["max_path_discrepancy"] = {unit: max(a[unit] for r in new["records"] for a in r["arms"]["contextual"]["numerical_audit"]["path_discrepancy"].values()) for unit in ("translation_m", "rotation_rad")}
        result["max_cached_update_coordinate_delta"] = max(r["arms"]["contextual"]["numerical_audit"]["cached_recorded_calibration_update_max_coordinate_delta"] for r in new["records"])
    for metric, methods in arrays.items():
        comparisons = [("original_contextual", "identity"), ("original_contextual", "original_direct")]
        if new_path.exists():
            comparisons += [("contextual", arm) for arm in ("identity", "constant", "feature_direct", "feature_residual", "original_contextual")]
        result["metrics"][metric] = {
            "methods": {arm: {"mean": float(a.mean()), "per_seed": a.mean(1).tolist(),
                              "per_family": dict(zip(FAMILIES, a.mean(0).tolist()))}
                        for arm, a in methods.items()},
            "contrasts": {f"{a}_minus_{b}": contrast(methods[a], methods[b]) for a, b in comparisons}}
    destination = HERE / ("analysis.json" if new_path.exists() else "identity_analysis.json")
    study.save(destination, result)
    print(json.dumps({"output": str(destination), "translation": result["metrics"]["translation_mean_m"],
                      "rotation_means": {a: r["mean"] for a, r in result["metrics"]["rotation_mean_deg"]["methods"].items()},
                      "durations": result["durations"]}, indent=2))


if __name__ == "__main__":
    main()
