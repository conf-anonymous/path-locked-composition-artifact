"""No-data audit of complete controls and reported aggregates, without win gates."""
import json
from pathlib import Path

import numpy as np

import analyze
import run as study

HERE = Path(__file__).resolve().parent


def main():
    result = json.loads((HERE / "results.json").read_text())
    summary = json.loads((HERE / "analysis.json").read_text())
    for relative, expected in result["seal"].items():
        assert study.sha(study.ROOT / relative) == expected, relative
    assert {(r["fold"], r["seed"]) for r in result["records"]} == {(f,s) for f in range(5) for s in range(5)}
    arrays = {metric: {arm: np.full((5,16), np.nan) for arm in study.ARMS}
              for metric in ("translation_mean_m", "rotation_mean_deg")}
    for r in result["records"]:
        checkpoint = HERE / "runs" / f"fold{r['fold']}_seed{r['seed']}.pt"
        assert study.sha(checkpoint) == r["checkpoint_sha256"]
        assert r["calibrator_reproduction_max"] < 1e-6
        assert set(r["arms"]) == set(study.ARMS)
        for arm, values in r["arms"].items():
            assert values["minimum_real_norm_before_projection"] > 1e-8
            assert sum(c["n"] for c in values["clusters"].values()) == values["n_windows"]
            for metric in arrays:
                family = analyze.family_values(values["clusters"], metric)
                assert set(family) == set(study.v.FOLD_FAMILIES[r["fold"]])
                for f, x in family.items():
                    arrays[metric][arm][r["seed"], analyze.FAMILIES.index(f)] = x
    for metric, methods in arrays.items():
        for arm, array in methods.items():
            assert np.isfinite(array).all()
            assert abs(array.mean()-summary["metrics"][metric]["methods"][arm]["mean"]) < 1e-12
    assert summary["n_windows"] == 11219
    for metric in summary["metrics"].values():
        for comparison in metric["contrasts"].values():
            # Exhaustive two-sided five-block enumeration includes the observed
            # assignment and its negative; numerical comparisons must retain both.
            assert comparison["fold_block_sign_flip_sensitivity_p"] >= 2/32
    identity = summary["metrics"]["translation_mean_m"]["methods"]["identity"]["mean"]
    assert abs(identity - .9150570433593836) < 1e-12
    assert summary["max_path_discrepancy"]["translation_m"] < 1e-8
    assert summary["max_path_discrepancy"]["rotation_rad"] < 1e-8
    check = json.loads((HERE / "cached_inference_check.json").read_text())
    assert check["maximum_coordinate_difference"] <= 1e-6
    cache = json.loads((HERE / "cache_deployment.json").read_text())
    assert cache["cached_product_calls_all_updates"] == 160
    assert cache["full_recompute_product_calls_all_updates"] == 992
    assert cache["max_translation_difference_m"] < 1e-10
    print("PASS: all four arms, 25 calibrator reproductions, 100 trained heads, identity control, cache/path audits, and aggregate arithmetic")
    print("Intervals are conditional; no requirement that a GA arm win was used as an integrity gate.")


if __name__ == "__main__":
    main()
