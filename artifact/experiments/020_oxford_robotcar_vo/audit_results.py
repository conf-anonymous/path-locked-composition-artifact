"""Fail closed against the prospectively frozen Oxford development/confirmation gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PRIMARY_LENGTH = "32"


def path(results, arm, seed, name):
    return results["arms"][arm][str(seed)]["lengths"][PRIMARY_LENGTH]["paths"][name]


def cluster_mean(record):
    return float(np.mean([
        values["translation_mean_m"] for values in record["clusters"].values()
    ]))


def delta(results, arm, seed, name):
    return results["arms"][arm][str(seed)]["lengths"][PRIMARY_LENGTH][
        "translation_deltas"
    ][name]


def audit(results: dict) -> dict:
    raw_record = results["baselines"][PRIMARY_LENGTH]["raw_motor"]
    raw = cluster_mean(raw_record)
    ga = [cluster_mean(path(results, "ga_calibrated", seed, "left")) for seed in range(5)]
    ga_delta = [value - raw for value in ga]
    ga_intervals = [
        results["arms"]["ga_calibrated"][str(seed)]["lengths"][PRIMARY_LENGTH][
            "left_vs_raw"
        ]
        for seed in range(5)
    ]
    ga_improvement = (
        all(value < 0 for value in ga_delta)
        and all(item["ci95_m"][1] < 0 for item in ga_intervals)
        and float(np.mean(ga)) <= 0.95 * raw
    )

    exactness = all(
        results["arms"]["ga_calibrated"][str(seed)]["lengths"][str(length)][
            "exact_audit"
        ]["passes"]
        for seed in range(5)
        for length in (4, 8, 16, 32)
    )

    learned = [cluster_mean(path(results, "learned", seed, "left")) for seed in range(5)]
    learned_competence = all(value <= 1.10 * ga[seed] for seed, value in enumerate(learned))
    learned_intervals = [delta(results, "learned", seed, "right") for seed in range(5)]
    learned_gap = float(np.mean([
        item["trajectory_equal_weight_mean_m"] for item in learned_intervals
    ]))
    path_locking = (
        sum(item["ci95_m"][0] > 0 for item in learned_intervals) >= 4
        and learned_gap >= max(0.5, 0.20 * float(np.mean(learned)))
    )

    mixed = [cluster_mean(path(results, "mixed", seed, "left")) for seed in range(5)]
    mixed_gap = float(np.mean([
        delta(results, "mixed", seed, "right")["trajectory_equal_weight_mean_m"]
        for seed in range(5)
    ]))
    mixed_intervention = (
        learned_gap > 0
        and abs(mixed_gap) <= 0.25 * learned_gap
        and float(np.mean(mixed)) <= 1.10 * float(np.mean(ga))
    )

    complete = True
    for length in (4, 8, 16, 32):
        baseline = results.get("baselines", {}).get(str(length), {})
        complete &= all(key in baseline for key in (
            "raw_motor", "raw_motor_moving", "exact_audit"
        ))
    for arm in ("ga_calibrated", "learned", "mlp", "mixed", "penalty"):
        seeds = results.get("arms", {}).get(arm, {})
        complete &= len(seeds) == 5
        for seed in range(5):
            for length in (4, 8, 16, 32):
                record = seeds.get(str(seed), {}).get("lengths", {}).get(str(length), {})
                complete &= all(key in record for key in (
                    "paths", "translation_deltas", "left_vs_raw",
                    "moving_paths", "moving_translation_deltas", "moving_left_vs_raw",
                ))
                complete &= all(
                    path_name in record.get(path_group, {})
                    for path_group in ("paths", "moving_paths")
                    for path_name in ("left", "right", "balanced", "random")
                )

    checks = {
        "ga_useful_accuracy": ga_improvement,
        "ga_exactness": exactness,
        "learned_competence": learned_competence,
        "learned_path_locking": path_locking,
        "mixed_intervention": mixed_intervention,
        "complete_reporting": complete,
    }
    return {
        "split": results["split"],
        "passes": all(checks.values()),
        "checks": checks,
        "raw_trajectory_equal_mean_m": raw,
        "ga_left_m_by_seed": ga,
        "ga_mean_relative_change": float(np.mean(ga) / raw - 1.0),
        "ga_interval_upper_m_by_seed": [item["ci95_m"][1] for item in ga_intervals],
        "learned_left_m_by_seed": learned,
        "learned_right_gap_m": learned_gap,
        "learned_right_interval_lower_m_by_seed": [
            item["ci95_m"][0] for item in learned_intervals
        ],
        "mixed_left_m_by_seed": mixed,
        "mixed_right_gap_m": mixed_gap,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirm", action="store_true")
    args = parser.parse_args()
    source = HERE / (
        "results_confirmation.json" if args.confirm else "results_development.json"
    )
    result = audit(json.loads(source.read_text()))
    print(json.dumps(result, indent=2))
    if not result["passes"]:
        raise SystemExit("FROZEN OXFORD GATE FAILED")


if __name__ == "__main__":
    main()
