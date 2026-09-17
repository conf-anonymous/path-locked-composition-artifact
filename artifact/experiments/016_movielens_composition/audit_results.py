"""Fail loudly if recorded Experiment 016 results do not satisfy the frozen gate."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXPECTED_ARMS = {"exact", "bilinear", "mlp", "mixed", "penalty"}
EXPECTED_SEEDS = {0, 1, 2, 3, 4}


def runs(payload, arm):
    selected = [row for row in payload["runs"] if row["arm"] == arm]
    assert {row["seed"] for row in selected} == EXPECTED_SEEDS
    return selected


def mean_accuracy(payload, arm, length, path="left"):
    return float(np.mean([
        row["lengths"][str(length)]["metrics"][path]["accuracy"]
        for row in runs(payload, arm)
    ]))


def mean_delta(payload, arm, length, path):
    return float(np.mean([
        row["lengths"][str(length)]["path_comparisons"][path]["paired_accuracy_delta"]
        for row in runs(payload, arm)
    ]))


def audit(payload, split):
    assert payload["split"] == split
    assert set(payload["arms"]) == EXPECTED_ARMS
    assert set(payload["seeds"]) == EXPECTED_SEEDS
    baseline = payload["baselines"]["8"]["movie_mode"]["accuracy"]
    assert mean_accuracy(payload, "bilinear", 8) > baseline + 0.02
    assert mean_delta(payload, "bilinear", 8, "right") < -0.05
    assert mean_delta(payload, "bilinear", 8, "balanced") < -0.04
    assert abs(mean_delta(payload, "exact", 8, "right")) == 0.0
    assert abs(mean_delta(payload, "exact", 8, "balanced")) == 0.0
    assert abs(mean_delta(payload, "mixed", 8, "right")) < 0.01
    assert abs(mean_delta(payload, "mixed", 8, "balanced")) < 0.01
    for row in runs(payload, "bilinear"):
        for path in ("right", "balanced"):
            upper = row["lengths"]["8"]["path_comparisons"][path]["user_clustered_95ci"][1]
            assert upper < 0.0, (split, row["seed"], path, upper)
    exact_max = max(
        value
        for row in runs(payload, "exact")
        for length in row["float64_invariance_audit"].values()
        for value in length.values()
    )
    assert exact_max <= 1e-5
    return {
        "split": split,
        "baseline": baseline,
        "bilinear_left": mean_accuracy(payload, "bilinear", 8),
        "bilinear_right_delta": mean_delta(payload, "bilinear", 8, "right"),
        "bilinear_balanced_delta": mean_delta(payload, "bilinear", 8, "balanced"),
        "mixed_right_delta": mean_delta(payload, "mixed", 8, "right"),
        "exact_float64_max": exact_max,
    }


def main():
    summaries = []
    for filename, split in (
        ("results_development.json", "dev"),
        ("results_confirmation.json", "test"),
    ):
        payload = json.loads((HERE / filename).read_text())
        summaries.append(audit(payload, split))
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
