"""Fail closed against the prospectively frozen Experiment 017 gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def mean_seed_metric(results: dict, arm: str, length: int, path: str, metric: str) -> float:
    return float(np.mean([
        seed["lengths"][str(length)]["paths"][path][metric]
        for seed in results["arms"][arm].values()
    ]))


def mean_seed_delta(results: dict, arm: str, length: int, path: str) -> float:
    return float(np.mean([
        seed["lengths"][str(length)]["translation_deltas"][path]["trajectory_equal_weight_mean_m"]
        for seed in results["arms"][arm].values()
    ]))


def development_audit(results: dict) -> dict:
    baseline = results["baselines"]["8"]["decoupled"]["translation_median_m"]
    competence = {
        arm: mean_seed_metric(results, arm, 8, "left", "translation_median_m")
        for arm in ("bilinear", "mlp")
    }
    interval_pass = {}
    for path in ("right", "balanced"):
        intervals = [
            seed["lengths"]["8"]["translation_deltas"][path]["ci95_m"]
            for seed in results["arms"]["bilinear"].values()
        ]
        interval_pass[path] = all(interval[0] > 0 for interval in intervals)

    bilinear_gap = {path: mean_seed_delta(results, "bilinear", 8, path)
                    for path in ("right", "balanced")}
    mixed_gap = {path: mean_seed_delta(results, "mixed", 8, path)
                 for path in ("right", "balanced")}
    bilinear_left = mean_seed_metric(results, "bilinear", 8, "left", "translation_median_m")
    mixed_left = mean_seed_metric(results, "mixed", 8, "left", "translation_median_m")
    mixed_pass = (
        all(bilinear_gap[path] > 0 and abs(mixed_gap[path]) <= 0.25 * bilinear_gap[path]
            for path in bilinear_gap)
        and mixed_left <= 1.25 * bilinear_left
    )

    grows = {}
    for arm in ("bilinear", "mlp"):
        gap8 = max(mean_seed_delta(results, arm, 8, path) for path in ("right", "balanced"))
        gap32 = max(mean_seed_delta(results, arm, 32, path) for path in ("right", "balanced"))
        grows[arm] = {"length8_m": gap8, "length32_m": gap32, "passes": gap32 > gap8}

    checks = {
        "competence": all(value < baseline and value < 0.10 for value in competence.values()),
        "bilinear_intervals": all(interval_pass.values()),
        "motor_exactness": all(item["passes"] for item in results["exact_audit"].values()),
        "mixed_intervention": mixed_pass,
        "depth_growth": any(item["passes"] for item in grows.values()),
    }
    return {
        "split": "dev",
        "passes": all(checks.values()),
        "checks": checks,
        "decoupled_median_translation_m": baseline,
        "learned_left_median_translation_m": competence,
        "bilinear_interval_checks": interval_pass,
        "bilinear_gap_m": bilinear_gap,
        "mixed_gap_m": mixed_gap,
        "mixed_left_median_translation_m": mixed_left,
        "depth_growth": grows,
    }


def confirmation_audit(results: dict) -> dict:
    interval_pass = {}
    for path in ("right", "balanced"):
        intervals = [
            seed["lengths"]["8"]["translation_deltas"][path]["ci95_m"]
            for seed in results["arms"]["bilinear"].values()
        ]
        interval_pass[path] = all(interval[0] > 0 for interval in intervals)
    checks = {
        "bilinear_intervals": all(interval_pass.values()),
        "motor_exactness": all(item["passes"] for item in results["exact_audit"].values()),
    }
    return {"split": "confirm", "passes": all(checks.values()), "checks": checks,
            "bilinear_interval_checks": interval_pass}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirm", action="store_true")
    args = parser.parse_args()
    source = HERE / ("results_confirmation.json" if args.confirm else "results_development.json")
    results = json.loads(source.read_text())
    audit = confirmation_audit(results) if args.confirm else development_audit(results)
    print(json.dumps(audit, indent=2))
    if not audit["passes"]:
        raise SystemExit("FROZEN GATE FAILED")


if __name__ == "__main__":
    main()
