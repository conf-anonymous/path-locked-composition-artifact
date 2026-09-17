"""Fail-closed audit of the frozen Experiment 022 confirmation result."""

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
results = json.loads((HERE / "results_confirmation.json").read_text())
raw = results["lengths"]["32"]["raw"]
names = sorted(raw["clusters"])
raw_mean = np.mean([raw["clusters"][name]["translation_mean_m"] for name in names])
deltas = []
for seed in range(5):
    adaptive = results["models"][str(seed)]["lengths"]["32"]["adaptive"]
    deltas.append([
        adaptive["clusters"][name]["translation_mean_m"]
        - raw["clusters"][name]["translation_mean_m"] for name in names
    ])
deltas = np.asarray(deltas)
adaptive_by_seed = raw_mean + deltas.mean(1)
rng = np.random.default_rng(220906)
draws = rng.integers(0, len(names), size=(10_000, len(names)))
distribution = deltas[:, draws].mean(axis=(0, 2))
interval = np.quantile(distribution, (0.025, 0.975))
checks = {
    "every_seed_improves": bool(np.all(adaptive_by_seed < raw_mean)),
    "mean_reduction_at_least_10pct": bool(adaptive_by_seed.mean() <= 0.90 * raw_mean),
    "pooled_sequence_interval_below_zero": bool(interval[1] < 0),
    "exact_all_seeds_and_horizons": all(
        results["models"][str(seed)]["lengths"][str(length)]["exact_audit"]["passes"]
        for seed in range(5) for length in (4, 8, 16, 32)
    ),
}
report = {
    "passes": all(checks.values()), "checks": checks,
    "n_confirmation_sequences": len(names),
    "raw_sequence_equal_mean_m": float(raw_mean),
    "adaptive_sequence_equal_mean_m_by_seed": adaptive_by_seed.tolist(),
    "mean_relative_change": float(adaptive_by_seed.mean() / raw_mean - 1),
    "pooled_ci95_delta_m": interval.tolist(),
    "worst_exact_si": max(
        results["models"][str(seed)]["lengths"][str(length)]["exact_audit"]["worst_si"]
        for seed in range(5) for length in (4, 8, 16, 32)
    ),
}
print(json.dumps(report, indent=2))
if not report["passes"]:
    raise SystemExit("FROZEN ADAPTIVE MOTOR CONFIRMATION FAILED")
