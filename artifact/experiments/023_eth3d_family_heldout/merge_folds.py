"""Merge independently scheduled frozen outer folds without altering outcomes."""

import json
from pathlib import Path

from run import Config, FOLDS, SEEDS

HERE = Path(__file__).resolve().parent
records = []
for fold in FOLDS:
    path = HERE / f"results_smoke_f{fold}_s0-1-2-3-4_5000.json"
    record = json.loads(path.read_text())
    if record["folds_requested"] != [fold] or record["seeds_requested"] != list(SEEDS):
        raise RuntimeError(f"unexpected partial-run scope: {path}")
    if record["config"] != {
        "calibrator_steps": Config.calibrator_steps,
        "contextual_steps": Config.contextual_steps,
        "batch_size": Config.batch_size,
        "calibrator_learning_rate": Config.calibrator_learning_rate,
        "contextual_learning_rate": Config.contextual_learning_rate,
        "weight_decay": Config.weight_decay,
        "hidden_size": Config.hidden_size,
        "evaluation_batch_size": Config.evaluation_batch_size,
        "bootstrap_resamples": Config.bootstrap_resamples,
        "moving_threshold_m": Config.moving_threshold_m,
    }:
        raise RuntimeError(f"unexpected partial-run configuration: {path}")
    if set(record["folds"]) != {str(fold)}:
        raise RuntimeError(f"unexpected partial-run content: {path}")
    records.append(record)

first = records[0]
for record in records[1:]:
    for key in ("study", "config", "family_sequences", "fold_families", "front_end"):
        if record[key] != first[key]:
            raise RuntimeError(f"partial runs disagree on {key}")

merged = {
    "study": first["study"], "complete": True, "config": first["config"],
    "folds_requested": list(FOLDS), "seeds_requested": list(SEEDS),
    "family_sequences": first["family_sequences"],
    "fold_families": first["fold_families"], "front_end": first["front_end"],
    "execution": "independent fold scheduling; frozen protocol unchanged",
    "folds": {
        str(fold): records[fold]["folds"][str(fold)] for fold in FOLDS
    },
}
destination = HERE / "results.json"
if destination.exists():
    raise RuntimeError("complete result already exists; refusing overwrite")
destination.write_text(json.dumps(merged, indent=2) + "\n")
print(f"wrote {destination}")
