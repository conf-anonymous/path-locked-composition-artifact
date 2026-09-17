"""Prespecified traversal-clustered Oxford extension analysis and completeness."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import oxford_workflow as wf

HEADS = ("constant", "contextual", "feature_direct", "feature_residual", "mergeable")
ARMS = ("identity", "raw", "ga_calibrated", *HEADS)
LENGTHS = ("4", "8", "16", "32")
PRIMARY = ("raw", "identity", "feature_direct", "feature_residual")


def validate_metric(metric, expected_names=None, allow_empty=False):
    clusters = metric["clusters"]
    if metric["n_trajectories"] != len(clusters):
        raise RuntimeError("trajectory count mismatch")
    if metric["n_windows"] != sum(v["n"] for v in clusters.values()):
        raise RuntimeError("window count mismatch")
    if not clusters and not allow_empty:
        raise RuntimeError("missing all-window results")
    if expected_names is not None and set(clusters) != set(expected_names):
        raise RuntimeError("traversal coverage mismatch")
    for values in clusters.values():
        if values["n"] < 1 or values["translation_mean_m"] < 0 or values["rotation_mean_deg"] < 0:
            raise RuntimeError("invalid traversal metric")
    wf.finite_tree(metric)


def validate_extension(result, expected_names=None):
    wf.finite_tree(result)
    if result["split"] not in ("dev", "confirm") or set(result["seeds"]) != set(map(str, range(5))):
        raise RuntimeError("extension seed/split coverage incomplete")
    for seed in result["seeds"].values():
        if set(seed["lengths"]) != set(LENGTHS) or set(seed["head_sha256"]) != set(HEADS):
            raise RuntimeError("extension horizon/head coverage incomplete")
        for record in seed["lengths"].values():
            if set(record["arms"]) != set(ARMS):
                raise RuntimeError("extension arm coverage incomplete")
            reference = record["arms"]["raw"]
            for metrics in record["arms"].values():
                validate_metric(metrics["all"], expected_names)
                validate_metric(metrics["moving"], reference["moving"]["clusters"], allow_empty=True)
                if metrics["all"]["clusters"].keys() != reference["all"]["clusters"].keys():
                    raise RuntimeError("paired traversal coverage changed")
                for subset in ("all", "moving"):
                    if any(v["n"] != reference[subset]["clusters"][n]["n"]
                           for n, v in metrics[subset]["clusters"].items()):
                        raise RuntimeError("paired window coverage changed")
            numeric = record["numerical_audit"]
            if not numeric["finite"] or numeric["minimum_unclamped_real_norm"] <= 0:
                raise RuntimeError("invalid paired-state numeric audit")
            if set(numeric["maxima"]) != {"right", "balanced", "random"}:
                raise RuntimeError("incomplete paired-state tree audit")
            observed_pass = all(v["translation_m"] < 1e-9 and v["rotation_rad"] < 1e-9
                                for v in numeric["maxima"].values())
            if numeric["passes"] != observed_pass:
                raise RuntimeError("numeric audit verdict does not match measured maxima")
    return True


def comparison(result, length, subset, comparator, metric):
    differences = []
    names = None
    for seed in range(5):
        arms = result["seeds"][str(seed)]["lengths"][length]["arms"]
        left = arms[comparator][subset]["clusters"]
        right = arms["mergeable"][subset]["clusters"]
        current = sorted(left)
        if set(left) != set(right) or (names is not None and names != current):
            raise RuntimeError("seed/pair traversal mismatch")
        names = current
        differences.append([right[n][metric] - left[n][metric] for n in names])
    if not names:
        return {"status": "no moving windows", "n_traversals": 0}
    array = np.asarray(differences)
    by_traversal = array.mean(0)  # Average seeds first, not windows or seeds as IID units.
    rng = np.random.default_rng(310906)
    draws = rng.integers(0, len(names), (10000, len(names)))
    distribution = by_traversal[draws].mean(1)
    return {"n_traversals": len(names), "mean_delta": float(by_traversal.mean()),
            "ci95": np.quantile(distribution, (.025, .975)).tolist(),
            "ci98_75": np.quantile(distribution, (.00625, .99375)).tolist(),
            "per_seed_mean_delta": array.mean(1).tolist(),
            "seeds_improved": int(np.sum(array.mean(1) < 0)),
            "per_traversal_delta": dict(zip(names, by_traversal.tolist()))}


def analyze(result):
    validate_extension(result)
    primary = {arm: comparison(result, "32", "all", arm, "translation_mean_m") for arm in PRIMARY}
    numeric = all(s["lengths"][length]["numerical_audit"]["passes"]
                  for s in result["seeds"].values() for length in LENGTHS)
    utility = all(v["ci98_75"][1] < 0 and v["seeds_improved"] >= 4 for v in primary.values())
    comparisons = {length: {subset: {metric: {
        arm: comparison(result, length, subset, arm, metric)
        for arm in ARMS if arm != "mergeable"}
        for metric in ("translation_mean_m", "rotation_mean_deg")}
        for subset in ("all", "moving")} for length in LENGTHS}
    return {"split": result["split"], "primary_l32_translation": primary,
            "primary_utility_passes": utility, "paired_state_exactness_passes": numeric,
            "external_architectural_confirmation": result["split"] == "confirm" and utility and numeric,
            "other_comparisons": comparisons,
            "uncertainty": "10000 paired whole-traversal draws after averaging seeds; not new-city generalization",
            "secondary_intervals": "ordinary 95% exploratory; adjusted intervals support only four primary comparisons"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirm", action="store_true")
    args = parser.parse_args()
    name = "confirmation" if args.confirm else "development"
    wf.save(wf.HERE / f"analysis_{name}.json", analyze(wf.read(wf.HERE / f"results_{name}.json")))
