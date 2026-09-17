"""Independently check retained scope/readout records without model execution."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
LENGTHS = (4, 8, 16, 32, 64, 128)
COHORTS = ("full", "common_sequences", "matched_starts")
METRICS = ("translation_mean_m", "rotation_mean_deg")
ORIGINAL = ("identity", "raw", "calibrated", "constant", "contextual",
            "feature_direct", "feature_residual", "mergeable")
ARMS = (*ORIGINAL, "contextual_aligned", "mergeable_aligned")


def sha(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def same(a, b):
    assert math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12), (a, b)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-data", action="store_true")
    parser.add_argument("--bootstrap", action="store_true",
                        help="Also recompute all intervals; requires NumPy")
    args = parser.parse_args()
    summary = json.loads((HERE / "analysis.json").read_text())
    seal = json.loads((HERE / "seal.json").read_text())
    assert sha(HERE / "seal.json") == summary["seal_sha256"]
    absent = []
    for name, digest in seal.items():
        assert not Path(name).is_absolute() and ".." not in Path(name).parts
        p = ROOT / name
        if p.is_file():
            assert sha(p) == digest, name
        else:
            assert not args.require_data
            assert name.startswith("data/raw/eth3d_rgbd/derived_odometry/") and name.endswith(".npz"), name
            absent.append(name)
    expected = {f"runs/fold{f}_seed{s}_L{l}.json" for f in range(5)
                for s in range(5) for l in LENGTHS}
    assert set(summary["record_sha256"]) == expected
    assert {str(p.relative_to(HERE)) for p in (HERE / "runs").glob("*.json")} == expected
    records = []
    ids = defaultdict(set)
    reproduction = 0.0
    for name in sorted(expected):
        assert sha(HERE / name) == summary["record_sha256"][name], name
        r = json.loads((HERE / name).read_text())
        assert r["seal_sha256"] == summary["seal_sha256"]
        assert name == f"runs/fold{r['fold']}_seed{r['seed']}_L{r['length']}.json"
        ids[(r["fold"], r["seed"])].add(r["matched_start_id_sha256"])
        old = json.loads((ROOT / "experiments/030_eth3d_frozen_scope" / name).read_text())
        for arm in ORIGINAL:
            actual = r["cohorts"]["full"][arm]
            prior = old["arms"][arm]
            assert actual["n_windows"] == prior["n_windows"]
            assert set(actual["clusters"]) == set(prior["clusters"])
            for sequence, row in actual["clusters"].items():
                assert row["n"] == prior["clusters"][sequence]["n"]
                for metric in METRICS:
                    difference = abs(row[metric] - prior["clusters"][sequence][metric])
                    reproduction = max(reproduction, difference)
                    same(row[metric], prior["clusters"][sequence][metric])
        assert r["maximum_original_reproduction_difference"] == 0
        assert r["numerical_audit"]["sign_branch_changes"] == 0
        assert r["readout_diagnostics"]["minimum_alignment_bound_slack"] >= -2e-6
        assert not any(r["readout_diagnostics"]["below_arithmetic_clamp_cases"].values())
        records.append(r)
    assert len(records) == summary["n_records"] == 150
    assert all(len(values) == 1 for values in ids.values())
    assert reproduction == summary["maximum_original_reproduction_difference"] == 0
    for cohort in COHORTS:
        for length in LENGTHS:
            selected = [r for r in records if r["length"] == length and r["cohorts"][cohort]]
            reported = summary["cohorts"][cohort][str(length)]
            families = reported["families"]
            sequences = {n for r in selected for n in r["cohorts"][cohort]["mergeable"]["clusters"]}
            assert sorted(sequences) == reported["sequences"]
            assert sum(r["cohorts"][cohort]["mergeable"]["n_windows"] for r in selected if r["seed"] == 0) == reported["n_windows"]
            if cohort != "full":
                assert len(sequences) == 23 and len(families) == 13
            if cohort == "matched_starts":
                assert reported["n_windows"] == 3435
            for metric in METRICS:
                calculated = {}
                for arm in ARMS:
                    cells = defaultdict(list)
                    for r in selected:
                        for row in r["cohorts"][cohort][arm]["clusters"].values():
                            cells[(r["seed"], row["family"])].append(row[metric])
                    assert set(cells) == {(s, f) for s in range(5) for f in families}
                    values = {key: statistics.mean(rows) for key, rows in cells.items()}
                    actual = reported["metrics"][metric]["methods"][arm]
                    same(statistics.mean(values.values()), actual["mean"])
                    calculated[arm] = values
                    for s in range(5):
                        same(statistics.mean(values[(s, f)] for f in families), actual["per_seed"][s])
                    for f in families:
                        same(statistics.mean(values[(s, f)] for s in range(5)), actual["per_family"][f])
                for name, contrast in reported["metrics"][metric]["contrasts"].items():
                    lhs, rhs = name.split("_minus_")
                    delta = [statistics.mean(calculated[lhs][(s, f)] - calculated[rhs][(s, f)]
                                             for s in range(5)) for f in families]
                    same(statistics.mean(delta), contrast["mean_delta"])
                    assert sum(d < 0 for d in delta) == contrast["family_wins"]
                    if args.bootstrap:
                        import numpy as np
                        rng = np.random.default_rng(160926)
                        draws = np.asarray(delta)[rng.integers(0, len(families), (10000, len(families)))].mean(1)
                        for a, b in zip(np.quantile(draws, [.025, .975]), contrast["conditional_ci95"]):
                            same(a, b)
    print(json.dumps({"status": "PASS", "records": len(records),
                      "original_metrics_reproduced_exactly": True,
                      "matched_start_hashes_identical_across_horizons": True,
                      "aggregates_recomputed_without_importing_experiment_implementation": True,
                      "bootstrap_recomputed": args.bootstrap,
                      "external_recording_files_absent": len(absent)}, indent=2))


if __name__ == "__main__":
    main()
