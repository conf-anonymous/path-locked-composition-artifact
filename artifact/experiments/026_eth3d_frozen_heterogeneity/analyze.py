"""Post-hoc heterogeneity diagnostics over immutable ETH3D family records."""

from __future__ import annotations

import itertools
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
P3_PATH = HERE.parent / "023_eth3d_family_heldout/results.json"
P4_PATH = HERE.parent / "024_eth3d_calibrated_direct_control/results.json"
P3 = json.loads(P3_PATH.read_text())
P4 = json.loads(P4_PATH.read_text())


def family_mean(clusters: dict, family: str, metric: str) -> float:
    values = [
        record[metric] for record in clusters.values() if record["family"] == family
    ]
    if not values:
        raise RuntimeError(f"missing family {family}")
    return float(np.mean(values))


def ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    output = np.empty(len(values), dtype=float)
    output[order] = np.arange(len(values), dtype=float)
    return output


families = sorted(family for fold in P3["folds"].values() for family in fold["families"])
records = {}
for family in families:
    fold_key = next(key for key, fold in P3["folds"].items() if family in fold["families"])
    fold3, fold4 = P3["folds"][fold_key], P4["folds"][fold_key]
    raw_t = family_mean(
        fold3["baselines"]["32"]["raw_motor"]["clusters"],
        family,
        "translation_mean_m",
    )
    contextual_t, contextual_r, direct_t, direct_r = [], [], [], []
    for seed in map(str, range(5)):
        contextual = fold3["seeds"][seed]["lengths"]["32"]["contextual_ga"]["clusters"]
        direct = fold4["seeds"][seed]["lengths"]["32"]["clusters"]
        contextual_t.append(family_mean(contextual, family, "translation_mean_m"))
        contextual_r.append(family_mean(contextual, family, "rotation_mean_deg"))
        direct_t.append(family_mean(direct, family, "translation_mean_m"))
        direct_r.append(family_mean(direct, family, "rotation_mean_deg"))
    context_t, context_r = float(np.mean(contextual_t)), float(np.mean(contextual_r))
    matched_t, matched_r = float(np.mean(direct_t)), float(np.mean(direct_r))
    records[family] = {
        "raw_translation_m": raw_t,
        "contextual_translation_m": context_t,
        "calibrated_direct_translation_m": matched_t,
        "contextual_minus_raw_m": context_t - raw_t,
        "contextual_minus_direct_m": context_t - matched_t,
        "contextual_rotation_deg": context_r,
        "calibrated_direct_rotation_deg": matched_r,
        "contextual_minus_direct_rotation_deg": context_r - matched_r,
    }

raw = np.asarray([records[name]["raw_translation_m"] for name in families])
delta = np.asarray([records[name]["contextual_minus_direct_m"] for name in families])
raw_delta = np.asarray([records[name]["contextual_minus_raw_m"] for name in families])
rotation_delta = np.asarray(
    [records[name]["contextual_minus_direct_rotation_deg"] for name in families]
)

sign_means = np.asarray([
    np.mean(delta * np.asarray(signs))
    for signs in itertools.product((-1.0, 1.0), repeat=len(delta))
])
observed = float(np.mean(delta))
leave_one = {
    family: float(np.delete(delta, index).mean())
    for index, family in enumerate(families)
}
ranked_raw = [families[index] for index in np.argsort(raw)[::-1]]
remove_high_raw = {
    str(count): {
        "removed": ranked_raw[:count],
        "contextual_minus_direct_mean_m": float(
            np.delete(delta, [families.index(name) for name in ranked_raw[:count]]).mean()
        ),
        "contextual_minus_raw_mean_m": float(
            np.delete(raw_delta, [families.index(name) for name in ranked_raw[:count]]).mean()
        ),
    }
    for count in (1, 2, 3)
}

result = {
    "study": "ETH3D immutable-result heterogeneity v1",
    "status": "post hoc; no model or endpoint selection",
    "source_sha256": {
        "experiment_023": hashlib.sha256(P3_PATH.read_bytes()).hexdigest(),
        "experiment_024": hashlib.sha256(P4_PATH.read_bytes()).hexdigest(),
    },
    "n_families": len(families),
    "contextual_minus_direct_mean_m": observed,
    "contextual_minus_direct_median_m": float(np.median(delta)),
    "contextual_translation_wins": int(np.sum(delta < 0)),
    "contextual_rotation_wins": int(np.sum(rotation_delta < 0)),
    "contextual_minus_direct_rotation_mean_deg": float(np.mean(rotation_delta)),
    "spearman_raw_error_vs_contextual_advantage_over_direct": float(
        np.corrcoef(ranks(raw), ranks(-delta))[0, 1]
    ),
    "spearman_raw_error_vs_contextual_advantage_over_raw": float(
        np.corrcoef(ranks(raw), ranks(-raw_delta))[0, 1]
    ),
    "exact_sign_flip": {
        "alternative": "contextual minus direct mean < 0",
        "one_sided_p": float(np.mean(sign_means <= observed + 1e-15)),
        "two_sided_p": float(np.mean(np.abs(sign_means) >= abs(observed) - 1e-15)),
        "enumerated_assignments": len(sign_means),
    },
    "leave_one_family_out": {
        "all_means_below_zero": all(value < 0 for value in leave_one.values()),
        "least_favorable_mean_m": max(leave_one.values()),
        "least_favorable_removed_family": max(leave_one, key=leave_one.get),
        "means_m": leave_one,
    },
    "remove_highest_raw_error": remove_high_raw,
    "families": records,
}
(HERE / "results.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
