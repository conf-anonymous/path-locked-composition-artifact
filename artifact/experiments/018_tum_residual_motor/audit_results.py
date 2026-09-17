"""Fail closed against the frozen Experiment 018 gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def seed_metric(results, arm, seed, length, path, metric):
    return results["arms"][arm][str(seed)]["lengths"][str(length)]["paths"][path][metric]


def seed_delta(results, arm, seed, length, path):
    return results["arms"][arm][str(seed)]["lengths"][str(length)]["translation_deltas"][path]


def audit(results: dict) -> dict:
    baseline = results["baselines"]["8"]["decoupled"]["translation_median_m"]
    competence = {
        arm: [seed_metric(results, arm, seed, 8, "left", "translation_median_m")
              for seed in range(5)]
        for arm in ("bilinear", "mlp")
    }
    competence_pass = (
        all(value < baseline for value in competence["bilinear"])
        and sum(value < baseline for value in competence["mlp"]) >= 4
    )
    interval_pass = {
        path: all(seed_delta(results, "bilinear", seed, 8, path)["ci95_m"][0] > 0
                  for seed in range(5))
        for path in ("right", "balanced")
    }
    gaps = {
        arm: {
            path: float(np.mean([
                seed_delta(results, arm, seed, 8, path)["trajectory_equal_weight_mean_m"]
                for seed in range(5)
            ]))
            for path in ("right", "balanced")
        }
        for arm in ("bilinear", "mixed")
    }
    mixed_left = float(np.mean([
        seed_metric(results, "mixed", seed, 8, "left", "translation_median_m")
        for seed in range(5)
    ]))
    mixed_pass = (
        all(gaps["bilinear"][path] > 0
            and abs(gaps["mixed"][path]) <= 0.25 * gaps["bilinear"][path]
            for path in ("right", "balanced"))
        and mixed_left < baseline
    )
    depth = {}
    for arm in ("bilinear", "mlp", "mixed", "penalty"):
        values = {}
        for length in (8, 32):
            values[length] = max(float(np.mean([
                seed_delta(results, arm, seed, length, path)["trajectory_equal_weight_mean_m"]
                for seed in range(5)
            ])) for path in ("right", "balanced"))
        depth[arm] = {"length8_m": values[8], "length32_m": values[32],
                      "passes": values[32] > values[8]}
    checks = {
        "competence": competence_pass,
        "bilinear_intervals": all(interval_pass.values()),
        "motor_exactness": all(item["passes"] for item in results["exact_audit"].values()),
        "mixed_intervention": mixed_pass,
        "depth_growth": any(item["passes"] for item in depth.values()),
    }
    # Confirmation requires conditions 1--4; depth was selected on development.
    required = list(checks.values()) if results["split"] == "dev" else [
        checks[key] for key in ("competence", "bilinear_intervals", "motor_exactness", "mixed_intervention")
    ]
    return {
        "split": results["split"], "passes": all(required), "checks": checks,
        "decoupled_median_translation_m": baseline,
        "left_median_translation_m_by_seed": competence,
        "bilinear_interval_checks": interval_pass,
        "mean_path_gaps_m": gaps,
        "mixed_left_median_translation_m": mixed_left,
        "depth_growth": depth,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirm", action="store_true")
    args = parser.parse_args()
    source = HERE / ("results_confirmation.json" if args.confirm else "results_development.json")
    result = audit(json.loads(source.read_text()))
    print(json.dumps(result, indent=2))
    if not result["passes"]:
        raise SystemExit("FROZEN GATE FAILED")


if __name__ == "__main__":
    main()
