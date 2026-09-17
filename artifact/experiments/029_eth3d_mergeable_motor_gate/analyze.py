"""Aggregate mergeable-summary ablation; preserve shared-helper metadata history."""
import importlib.util
import json
import sys

import numpy as np

import run as experiment

HERE, study = experiment.HERE, experiment.study
# The sibling analyzer imports `run`; bind that name only while loading it.
original_run = sys.modules["run"]
sys.modules["run"] = study
spec = importlib.util.spec_from_file_location("comparison029", study.HERE / "analyze.py")
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)
sys.modules["run"] = original_run


def main():
    result = json.loads((HERE / "results.json").read_text())
    original = json.loads((study.HERE / "analysis.json").read_text())
    controls = json.loads((study.HERE / "results.json").read_text())
    assert len(result["records"]) == 25
    summary = {"scope": "associative 16-coordinate paired-product state plus fixed-size readout; no GRU",
               "metadata_notice": "The generic numerical_audit.scope field in raw run records was inherited from Experiment 028's GRU helper. It is not applicable to this GRU-free class; numerical operations and metrics are unchanged.",
               "parameters": 577, "total_parameters_including_calibrator": 857,
               "training_seconds_total": sum(r["training_seconds"] for r in result["records"]), "metrics": {}}
    for metric in ("translation_mean_m", "rotation_mean_deg"):
        values = np.full((5,16), np.nan)
        comparators = {name: np.full((5,16), np.nan) for name in study.ARMS}
        for r in result["records"]:
            for f, value in analysis.family_values(r["evaluation"]["clusters"], metric).items():
                values[r["seed"], analysis.FAMILIES.index(f)] = value
        for r in controls["records"]:
            for arm in study.ARMS:
                for f, value in analysis.family_values(r["arms"][arm]["clusters"], metric).items():
                    comparators[arm][r["seed"], analysis.FAMILIES.index(f)] = value
        identity = original["metrics"][metric]["methods"]["identity"]["per_family"]
        comparators["identity"] = np.tile([identity[f] for f in analysis.FAMILIES], (5,1))
        assert np.isfinite(values).all()
        summary["metrics"][metric] = {"mean": float(values.mean()), "per_seed": values.mean(1).tolist(),
            "per_family": dict(zip(analysis.FAMILIES, values.mean(0).tolist())),
            "contrasts": {name: analysis.contrast(values, array) for name, array in comparators.items()}}
    summary["minimum_real_norm_before_projection"] = min(r["evaluation"]["minimum_real_norm_before_projection"] for r in result["records"])
    summary["max_path_discrepancy"] = {unit: max(a[unit] for r in result["records"]
        for a in r["numerical_audit"]["path_discrepancy"].values()) for unit in ("translation_m", "rotation_rad")}
    study.save(HERE / "analysis.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
