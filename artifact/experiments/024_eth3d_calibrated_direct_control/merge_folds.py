"""Merge independently scheduled complete outer folds."""

import json
from pathlib import Path

from run import v3

HERE = Path(__file__).resolve().parent
records = []
for fold in v3.FOLDS:
    path = HERE / f"results_fold{fold}_seed0-1-2-3-4.json"
    record = json.loads(path.read_text())
    if record["folds_requested"] != [fold] or record["seeds_requested"] != list(v3.SEEDS):
        raise RuntimeError(f"unexpected scope: {path}")
    if set(record["folds"]) != {str(fold)}:
        raise RuntimeError(f"unexpected fold content: {path}")
    records.append(record)
first = records[0]
for record in records[1:]:
    for key in ("study", "source_results_sha256", "config"):
        if record[key] != first[key]:
            raise RuntimeError(f"fold records disagree on {key}")
merged = {
    "study": first["study"],
    "source_results_sha256": first["source_results_sha256"],
    "config": first["config"], "folds_requested": list(v3.FOLDS),
    "seeds_requested": list(v3.SEEDS),
    "execution": "independent fold scheduling; frozen protocol unchanged",
    "folds": {str(fold): records[fold]["folds"][str(fold)] for fold in v3.FOLDS},
}
destination = HERE / "results.json"
if destination.exists():
    raise RuntimeError(f"refusing to overwrite {destination}")
destination.write_text(json.dumps(merged, indent=2) + "\n")
print(f"wrote {destination}")
