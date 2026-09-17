"""Audit Experiment 022's logged five-seed model-selection record."""

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
results = json.loads(
    (HERE / "results_pilot_gated_0-1-2-3-4_5000.json").read_text()
)
if results["model_kind"] != "gated" or results["seeds"] != list(range(5)):
    raise SystemExit("not the designated five-seed gated selection record")

raw = results["lengths"]["32"]["raw"]
names = sorted(raw["clusters"])
seed_deltas = []
for seed in range(5):
    adaptive = results["models"][str(seed)]["lengths"]["32"]["adaptive"]
    seed_deltas.append(np.asarray([
        adaptive["clusters"][name]["translation_mean_m"]
        - raw["clusters"][name]["translation_mean_m"] for name in names
    ]))
deltas = np.stack(seed_deltas)
rng = np.random.default_rng(220905)
draws = rng.integers(0, len(names), size=(10_000, len(names)))
distribution = deltas[:, draws].mean(axis=(0, 2))
raw_mean = np.mean([raw["clusters"][name]["translation_mean_m"] for name in names])
adaptive_by_seed = raw_mean + deltas.mean(1)
checks = {
    "five_seeds_improve": bool(np.all(adaptive_by_seed < raw_mean)),
    "mean_gain_at_least_10pct": bool(adaptive_by_seed.mean() <= 0.90 * raw_mean),
    "pooled_sequence_bootstrap_below_zero": bool(np.quantile(distribution, .975) < 0),
    "exact_all_horizons": all(
        results["models"][str(seed)]["lengths"][str(length)]["exact_audit"]["passes"]
        for seed in range(5) for length in (4, 8, 16, 32)
    ),
}
report = {
    "passes": all(checks.values()), "checks": checks,
    "selection_status": "development-selected exploratory record",
    "raw_sequence_equal_mean_m": float(raw_mean),
    "adaptive_sequence_equal_mean_m_by_seed": adaptive_by_seed.tolist(),
    "relative_change": float(adaptive_by_seed.mean() / raw_mean - 1),
    "pooled_ci95_delta_m": np.quantile(distribution, (.025, .975)).tolist(),
}
print(json.dumps(report, indent=2))
if not report["passes"]:
    raise SystemExit("ADAPTIVE MOTOR DEVELOPMENT CRITERION FAILED")
