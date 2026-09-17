"""Audit the immutable-source boundary and reported heterogeneity facts."""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
result = json.loads((HERE / "results.json").read_text())

checks = {
    "post_hoc_label": result["status"] == "post hoc; no model or endpoint selection",
    "source_023_frozen": result["source_sha256"]["experiment_023"]
    == "f25987c0687078d4fb42021a25ee11a031c4b29d741ada91ca017cf3442458e1",
    "source_024_frozen": result["source_sha256"]["experiment_024"]
    == "cebd8ab9363fb481b85f46aede19377c3dd4d63cec104528b72851f350f24d5c",
    "all_families": result["n_families"] == 16 and len(result["families"]) == 16,
    "leave_one_out_robust": result["leave_one_family_out"]["all_means_below_zero"],
    "exact_sign_enumeration": result["exact_sign_flip"]["enumerated_assignments"]
    == 2**16,
}
report = {"passes": all(checks.values()), "checks": checks}
print(json.dumps(report, indent=2))
if not report["passes"]:
    raise SystemExit("ETH3D HETEROGENEITY AUDIT FAILED")
