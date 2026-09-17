"""Recompute all scope aggregates; do not gate on favorable results."""
import argparse
import json

import numpy as np

import run as study


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-data", action="store_true")
    args = parser.parse_args()
    here = study.HERE
    summary = json.loads((here / "analysis.json").read_text())
    sealed = json.loads((here / "seal.json").read_text())
    missing_data = []
    for relative, expected in sealed.items():
        path = study.ROOT / relative
        if not path.exists() and relative.startswith("data/"):
            missing_data.append(relative)
            continue
        assert path.exists() and study.study.sha(path) == expected, relative
    if args.require_data:
        assert not missing_data, "raw recordings required for source-backed audit"
    assert study.study.sha(here / "seal.json") == summary["seal_sha256"]
    records = []
    for relative, expected in summary["run_record_sha256"].items():
        assert study.study.sha(here / relative) == expected
        records.append(json.loads((here / relative).read_text()))
    assert {(r["fold"], r["seed"], r["length"]) for r in records} == {
        (f, s, length) for f in range(5) for s in range(5) for length in study.LENGTHS}
    reproduction_max = 0.
    for r in records:
        assert set(r["arms"]) == set(study.ARMS)
        clusters = r["arms"]["mergeable"]["clusters"]
        for arm in study.ARMS:
            a = r["arms"][arm]
            assert a["clusters"].keys() == clusters.keys()
            assert sum(c["n"] for c in a["clusters"].values()) == a["n_windows"]
            for n, c in a["clusters"].items():
                assert c["n"] == clusters[n]["n"]
                assert c["family"] in study.v.FOLD_FAMILIES[r["fold"]]
                assert all(np.isfinite(c[m]) and c[m] >= 0 for m in study.METRICS)
            if r["length"] == 32 and arm in (*study.study.ARMS, "mergeable"):
                root = study.experiment.HERE if arm == "mergeable" else study.study.HERE
                source = json.loads((root / "runs" / f"fold{r['fold']}_seed{r['seed']}.json").read_text())
                expected = source["evaluation"] if arm == "mergeable" else source["arms"][arm]
                delta = max(abs(c[m]-expected["clusters"][n][m]) for n,c in a["clusters"].items() for m in study.METRICS)
                reproduction_max = max(reproduction_max, delta)
    assert reproduction_max == summary["max_L32_reproduction_absolute_difference"]
    assert reproduction_max < 1e-5
    # The anonymous artifact does not redistribute recordings. Recompute ALL
    # metric means/contrasts from run records there, but explicitly retain the
    # stored duration metadata. Source-backed duration checks require data.
    duration_function = study.durations
    if missing_data:
        study.durations = lambda length: summary["lengths"][str(length)]["duration"]
    recalculated = study.summarize(records)
    study.durations = duration_function
    assert recalculated["lengths"] == summary["lengths"]
    print(f"PASS: {len(records)} fold/seed/horizon records; eight arms; frozen hashes; all aggregates and L32 reproduction")
    print("Duration/source status: " + ("stored metadata only; recordings not redistributed" if missing_data else "verified against all recorded source files"))


if __name__ == "__main__":
    main()
