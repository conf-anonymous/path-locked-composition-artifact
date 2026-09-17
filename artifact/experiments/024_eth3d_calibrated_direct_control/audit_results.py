"""Adjudicate the frozen two-stage-matched direct-control comparison."""

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
CONTROL = json.loads((HERE / "results.json").read_text())
SOURCE = json.loads((HERE.parent / "023_eth3d_family_heldout" / "results.json").read_text())
FAMILIES = sorted(SOURCE["family_sequences"])


def family_mean(record, family, metric):
    return float(np.mean([
        value[metric] for value in record["clusters"].values()
        if value["family"] == family
    ]))


contextual_t = {seed: {} for seed in range(5)}
contextual_r = {seed: {} for seed in range(5)}
direct_t = {seed: {} for seed in range(5)}
direct_r = {seed: {} for seed in range(5)}
reproduction = 0.0
parameter_match = True
complete = set(CONTROL["folds"]) == set(map(str, range(5)))
for fold in range(5):
    families = SOURCE["folds"][str(fold)]["families"]
    complete &= set(CONTROL["folds"][str(fold)]["seeds"]) == set(map(str, range(5)))
    for seed in range(5):
        old = SOURCE["folds"][str(fold)]["seeds"][str(seed)]
        new = CONTROL["folds"][str(fold)]["seeds"][str(seed)]
        reproduction = max(reproduction, new["calibrator_reproduction_max"])
        parameter_match &= (
            abs(new["parameter_counts"]["primary_trainable"]
                - old["parameter_counts"]["contextual_ga"])
            / old["parameter_counts"]["contextual_ga"] < 0.05
        )
        complete &= set(new["lengths"]) == {"4", "8", "16", "32"}
        contextual = old["lengths"]["32"]["contextual_ga"]
        direct = new["lengths"]["32"]
        for family in families:
            contextual_t[seed][family] = family_mean(
                contextual, family, "translation_mean_m"
            )
            contextual_r[seed][family] = family_mean(
                contextual, family, "rotation_mean_deg"
            )
            direct_t[seed][family] = family_mean(direct, family, "translation_mean_m")
            direct_r[seed][family] = family_mean(direct, family, "rotation_mean_deg")

ct = np.asarray([[contextual_t[s][f] for f in FAMILIES] for s in range(5)])
cr = np.asarray([[contextual_r[s][f] for f in FAMILIES] for s in range(5)])
dt = np.asarray([[direct_t[s][f] for f in FAMILIES] for s in range(5)])
dr = np.asarray([[direct_r[s][f] for f in FAMILIES] for s in range(5)])
delta = ct - dt
rng = np.random.default_rng(240905)
draws = rng.integers(0, len(FAMILIES), size=(10_000, len(FAMILIES)))
distribution = delta[:, draws].mean(axis=(0, 2))
interval = np.quantile(distribution, (0.025, 0.975))
outcome = (
    "contextual_ga_strict_advantage" if interval[1] < 0 else
    "calibrated_direct_strict_advantage" if interval[0] > 0 else
    "statistically_unresolved"
)
report = {
    "outcome": outcome,
    "contextual_ga_family_equal_translation_m": float(ct.mean()),
    "calibrated_direct_gru_family_equal_translation_m": float(dt.mean()),
    "contextual_minus_direct_mean_m": float(delta.mean()),
    "contextual_minus_direct_ci95_m": interval.tolist(),
    "contextual_ga_family_equal_rotation_deg": float(cr.mean()),
    "calibrated_direct_gru_family_equal_rotation_deg": float(dr.mean()),
    "contextual_better_families": int((delta.mean(0) < 0).sum()),
    "calibrator_reproduction_max": reproduction,
    "parameter_match": bool(parameter_match),
    "complete_reporting": bool(complete),
    "per_family": {
        family: {
            "contextual_ga_m": float(ct[:, index].mean()),
            "calibrated_direct_gru_m": float(dt[:, index].mean()),
            "contextual_minus_direct_m": float(delta[:, index].mean()),
        }
        for index, family in enumerate(FAMILIES)
    },
}
print(json.dumps(report, indent=2))
(HERE / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
if not parameter_match or not complete or reproduction > 1e-6:
    raise SystemExit("CONTROL AUDIT INCOMPLETE")
