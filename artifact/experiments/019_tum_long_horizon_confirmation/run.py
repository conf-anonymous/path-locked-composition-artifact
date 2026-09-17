"""Evaluate frozen Experiment 018 checkpoints once on TUM confirmation."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent / "018_tum_residual_motor"
sys.path.insert(0, str(SOURCE))
spec = importlib.util.spec_from_file_location("tum_residual_frozen", SOURCE / "run.py")
experiment = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = experiment
spec.loader.exec_module(experiment)


def main() -> None:
    result = experiment.run("confirm", experiment.base.Config(), torch.device("cpu"))
    result["confirmation_protocol"] = "Experiment 019, frozen before confirmation"
    result["checkpoint_source"] = "Experiment 018 frozen development checkpoints"
    destination = HERE / "results_confirmation.json"
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
