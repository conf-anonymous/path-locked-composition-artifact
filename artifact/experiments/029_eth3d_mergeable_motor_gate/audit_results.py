"""Verify every mergeable head and associative-summary result without win gates."""
import json

import numpy as np

import run as experiment


def main():
    here, study = experiment.HERE, experiment.study
    result = json.loads((here / "results.json").read_text())
    summary = json.loads((here / "analysis.json").read_text())
    assert {(r["fold"], r["seed"]) for r in result["records"]} == {(f,s) for f in range(5) for s in range(5)}
    for relative, expected in result["seal"].items():
        assert study.sha(study.ROOT / relative) == expected
    values = []
    for r in result["records"]:
        stem = f"fold{r['fold']}_seed{r['seed']}"
        assert study.sha(here / "runs" / f"{stem}.pt") == r["checkpoint_sha256"]
        assert study.sha(study.HERE / "runs" / f"{stem}.pt") == r["source_checkpoint_sha256"]
        assert r["parameters"] == 577
        assert r["evaluation"]["minimum_real_norm_before_projection"] > 1e-8
        clusters = r["evaluation"]["clusters"]
        for f in study.v.FOLD_FAMILIES[r["fold"]]:
            values.append(np.mean([c["translation_mean_m"] for c in clusters.values() if c["family"] == f]))
    assert len(values) == 80
    assert abs(np.mean(values)-summary["metrics"]["translation_mean_m"]["mean"]) < 1e-12
    assert summary["max_path_discrepancy"]["translation_m"] < 1e-8
    assert summary["max_path_discrepancy"]["rotation_rad"] < 1e-8
    profile = json.loads((here / "profile.json").read_text())
    for model in profile["models"].values():
        assert model["cached_online_max_difference"] <= 1e-6
        for batch, values in model["batches"].items():
            assert abs(values["windows_per_second"]-int(batch)*1000/values["median_ms"]) < 1e-8
    print("PASS: 25 GRU-free mergeable motor gates, 577 head parameters, fixed folds/calibrators, separate numerical tolerances")


if __name__ == "__main__":
    main()
