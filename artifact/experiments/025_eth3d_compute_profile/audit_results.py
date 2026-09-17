"""Audit completeness and frozen-model reproduction for Experiment 025."""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
results = json.loads((HERE / "results.json").read_text())

checks = {
    "recorded_public_data_only": results["data_boundary"]
    == "recorded checksum-verified ETH3D operands only",
    "post_freeze_diagnostic": results["selection_status"]
    == "post-freeze descriptive diagnostic",
    "contextual_reproduced": results["reproduction_max"]["contextual_ga"] <= 1e-6,
    "direct_reproduced": results["reproduction_max"]["calibrated_direct_gru"] <= 1e-6,
    "parameter_match": abs(
        results["parameter_counts"]["contextual_ga_primary"]
        / results["parameter_counts"]["calibrated_direct_gru_primary"]
        - 1.0
    ) <= 0.05,
    "complete_batches": set(results["inference"]) == {"1", "32", "256"},
    "complete_operators": all(
        set(record)
        == {
            "actual_batch_size",
            "contextual_ga",
            "calibrated_direct_gru",
            "exact_motor_left",
            "exact_motor_balanced",
        }
        for record in results["inference"].values()
    ),
}
report = {"passes": all(checks.values()), "checks": checks}
print(json.dumps(report, indent=2))
if not report["passes"]:
    raise SystemExit("ETH3D COMPUTE PROFILE AUDIT FAILED")
