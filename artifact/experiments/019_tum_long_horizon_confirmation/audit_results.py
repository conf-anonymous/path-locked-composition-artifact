"""Fail closed against the Experiment 019 confirmation gate."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "018_tum_residual_motor" / "results_development.json"


def audit(results: dict) -> dict:
    length = "32"
    baseline = results["baselines"][length]["decoupled"]["translation_median_m"]

    def median(arm, seed, path="left"):
        return results["arms"][arm][str(seed)]["lengths"][length]["paths"][path]["translation_median_m"]

    def delta(arm, seed):
        return results["arms"][arm][str(seed)]["lengths"][length]["translation_deltas"]["right"]

    competence = {arm: [median(arm, seed) for seed in range(5)]
                  for arm in ("bilinear", "mixed", "penalty")}
    interval_lower = {arm: [delta(arm, seed)["ci95_m"][0] for seed in range(5)]
                      for arm in ("bilinear", "penalty")}
    gaps = {arm: float(np.mean([delta(arm, seed)["trajectory_equal_weight_mean_m"]
                                for seed in range(5)]))
            for arm in ("bilinear", "mixed", "penalty")}
    checks = {
        "bilinear_competence": all(value < baseline for value in competence["bilinear"]),
        "bilinear_right_intervals": all(value > 0 for value in interval_lower["bilinear"]),
        "motor_exactness": all(item["passes"] for item in results["exact_audit"].values()),
        "mixed_intervention": (
            all(value < baseline for value in competence["mixed"])
            and abs(gaps["mixed"]) <= 0.25 * gaps["bilinear"]
        ),
        "penalty_noncertificate": (
            all(value < baseline for value in competence["penalty"])
            and all(value > 0 for value in interval_lower["penalty"])
        ),
    }
    return {
        "split": results["split"], "passes": all(checks.values()), "checks": checks,
        "decoupled_median_translation_m": baseline,
        "left_median_translation_m_by_seed": competence,
        "right_interval_lower_m_by_seed": interval_lower,
        "right_gap_m": gaps,
    }


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--development-check", action="store_true")
    args = parser.parse_args()
    source = SOURCE if args.development_check else HERE / "results_confirmation.json"
    result = audit(json.loads(source.read_text()))
    print(json.dumps(result, indent=2))
    if not result["passes"]:
        raise SystemExit("FROZEN GATE FAILED")


if __name__ == "__main__":
    main()
