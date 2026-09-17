"""Fail-closed audit of the frozen ETH3D family-held-out study."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LENGTH = "32"


def family_mean(record: dict, family: str, metric: str) -> float:
    values = [
        sequence[metric] for sequence in record["clusters"].values()
        if sequence["family"] == family
    ]
    if not values:
        raise RuntimeError(f"no evaluable sequence for family {family}")
    return float(np.mean(values))


def main() -> None:
    results = json.loads((HERE / "results.json").read_text())
    if not results.get("complete"):
        raise SystemExit("result is not the complete frozen run")
    families = sorted(results["family_sequences"])
    raw_t, raw_r = {}, {}
    contextual_t = {seed: {} for seed in range(5)}
    contextual_r = {seed: {} for seed in range(5)}
    direct_t = {seed: {} for seed in range(5)}
    complete = set(results["folds"]) == set(map(str, range(5)))
    exact = True
    parameter_match = True
    for fold in range(5):
        fold_record = results["folds"][str(fold)]
        fold_families = fold_record["families"]
        raw = fold_record["baselines"][LENGTH]["raw_motor"]
        exact &= all(
            fold_record["baselines"][str(length)]["exact_audit"]["passes"]
            for length in (4, 8, 16, 32)
        )
        for family in fold_families:
            raw_t[family] = family_mean(raw, family, "translation_mean_m")
            raw_r[family] = family_mean(raw, family, "rotation_mean_deg")
        complete &= set(fold_record["seeds"]) == set(map(str, range(5)))
        for seed in range(5):
            seed_record = fold_record["seeds"][str(seed)]
            counts = seed_record["parameter_counts"]
            parameter_match &= (
                abs(counts["contextual_ga"] - counts["direct_gru"])
                / counts["contextual_ga"] < 0.05
            )
            for length in (4, 8, 16, 32):
                item = seed_record["lengths"][str(length)]
                complete &= all(name in item for name in (
                    "ga_calibrated", "contextual_ga", "direct_gru",
                    "ga_calibrated_exact_audit", "contextual_ga_exact_audit",
                ))
                exact &= item["ga_calibrated_exact_audit"]["passes"]
                exact &= item["contextual_ga_exact_audit"]["passes"]
            primary = seed_record["lengths"][LENGTH]
            for family in fold_families:
                contextual_t[seed][family] = family_mean(
                    primary["contextual_ga"], family, "translation_mean_m"
                )
                contextual_r[seed][family] = family_mean(
                    primary["contextual_ga"], family, "rotation_mean_deg"
                )
                direct_t[seed][family] = family_mean(
                    primary["direct_gru"], family, "translation_mean_m"
                )

    complete &= set(raw_t) == set(families)
    raw_t_array = np.asarray([raw_t[family] for family in families])
    raw_r_array = np.asarray([raw_r[family] for family in families])
    contextual_t_array = np.asarray([
        [contextual_t[seed][family] for family in families] for seed in range(5)
    ])
    contextual_r_array = np.asarray([
        [contextual_r[seed][family] for family in families] for seed in range(5)
    ])
    direct_t_array = np.asarray([
        [direct_t[seed][family] for family in families] for seed in range(5)
    ])
    raw_delta = contextual_t_array - raw_t_array[None, :]
    direct_delta = contextual_t_array - direct_t_array
    rng = np.random.default_rng(230905)
    draws = rng.integers(0, len(families), size=(10_000, len(families)))
    raw_distribution = raw_delta[:, draws].mean(axis=(0, 2))
    direct_distribution = direct_delta[:, draws].mean(axis=(0, 2))
    raw_interval = np.quantile(raw_distribution, (0.025, 0.975))
    direct_interval = np.quantile(direct_distribution, (0.025, 0.975))
    contextual_by_seed = contextual_t_array.mean(1)
    raw_mean = raw_t_array.mean()
    family_mean_delta = raw_delta.mean(0)
    leave_one_out = [np.delete(raw_delta, index, axis=1).mean()
                     for index in range(len(families))]
    checks = {
        "every_seed_improves": bool(np.all(contextual_by_seed < raw_mean)),
        "mean_reduction_at_least_10pct": bool(
            contextual_by_seed.mean() <= 0.90 * raw_mean
        ),
        "family_interval_below_zero": bool(raw_interval[1] < 0),
        "at_least_10_of_16_families_improve": bool((family_mean_delta < 0).sum() >= 10),
        "median_family_delta_below_zero": bool(np.median(family_mean_delta) < 0),
        "leave_any_one_family_out_improves": bool(max(leave_one_out) < 0),
        "beats_parameter_matched_direct_gru": bool(direct_interval[1] < 0),
        "rotation_noninferiority": bool(
            contextual_r_array.mean() <= 1.10 * raw_r_array.mean()
        ),
        "exact_all_product_paths": bool(exact),
        "parameter_match": bool(parameter_match),
        "complete_reporting": bool(complete),
    }
    report = {
        "passes": all(checks.values()), "checks": checks,
        "n_families": len(families),
        "raw_family_equal_translation_m": float(raw_mean),
        "contextual_family_equal_translation_m_by_seed": contextual_by_seed.tolist(),
        "contextual_mean_relative_change": float(
            contextual_by_seed.mean() / raw_mean - 1
        ),
        "contextual_vs_raw_ci95_m": raw_interval.tolist(),
        "direct_gru_family_equal_translation_m": float(direct_t_array.mean()),
        "contextual_vs_direct_ci95_m": direct_interval.tolist(),
        "improved_families": int((family_mean_delta < 0).sum()),
        "median_family_delta_m": float(np.median(family_mean_delta)),
        "worst_leave_one_family_out_delta_m": float(max(leave_one_out)),
        "raw_family_equal_rotation_deg": float(raw_r_array.mean()),
        "contextual_family_equal_rotation_deg": float(contextual_r_array.mean()),
        "per_family": {
            family: {
                "raw_translation_m": float(raw_t_array[index]),
                "contextual_translation_m": float(contextual_t_array[:, index].mean()),
                "direct_gru_translation_m": float(direct_t_array[:, index].mean()),
                "contextual_minus_raw_m": float(family_mean_delta[index]),
            }
            for index, family in enumerate(families)
        },
    }
    print(json.dumps(report, indent=2))
    (HERE / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    if not report["passes"]:
        raise SystemExit("FROZEN FAMILY-HELD-OUT GATE FAILED")


if __name__ == "__main__":
    main()
