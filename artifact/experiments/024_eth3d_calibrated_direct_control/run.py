"""Fully two-stage-matched direct GRU control for Experiment 023."""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
BASE_DIR = HERE.parent / "023_eth3d_family_heldout"
SOURCE_RESULTS = BASE_DIR / "results.json"
sys.path.insert(0, str(BASE_DIR))
spec = importlib.util.spec_from_file_location("family_v3", BASE_DIR / "run.py")
v3 = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = v3
spec.loader.exec_module(v3)


class CalibratedDirectGRU(nn.Module):
    def __init__(self, calibrator: nn.Module, hidden_size: int):
        super().__init__()
        self.calibrator = copy.deepcopy(calibrator)
        for parameter in self.calibrator.parameters():
            parameter.requires_grad_(False)
        self.encoder = nn.GRU(8, hidden_size, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(
            nn.Linear(2 * hidden_size, 38), nn.GELU(), nn.Linear(38, 8)
        )

    def forward(self, operands: torch.Tensor, path: str = "balanced") -> torch.Tensor:
        del path
        calibrated = self.calibrator(operands)
        encoded, _ = self.encoder(calibrated)
        return v3.base.normalize_motor(self.head(encoded.mean(1)))


def maximum_reproduction_difference(reference: dict, reproduced: dict) -> float:
    maximum = 0.0
    for name in reference["clusters"]:
        for metric in ("translation_mean_m", "rotation_mean_deg"):
            maximum = max(maximum, abs(
                reference["clusters"][name][metric]
                - reproduced["clusters"][name][metric]
            ))
    return maximum


def run_fold(all_data, source: dict, fold: int, seeds: tuple[int, ...],
             cfg, device: torch.device) -> dict:
    data = v3.FoldData(all_data, fold)
    result = {"families": list(v3.FOLD_FAMILIES[fold]), "seeds": {}}
    for seed in seeds:
        calibrator, calibration_training = v3.train_calibrator(
            data, cfg, seed, device
        )
        reproduction = v3.evaluate(data, 32, cfg, calibrator)
        expected = source["folds"][str(fold)]["seeds"][str(seed)][
            "lengths"
        ]["32"]["ga_calibrated"]
        difference = maximum_reproduction_difference(expected, reproduction)
        if difference > 1e-6:
            raise RuntimeError(
                f"fold={fold} seed={seed} calibrator did not reproduce: {difference}"
            )
        model, training = v3.train_primary(
            data, cfg, seed, device,
            CalibratedDirectGRU(calibrator.calibrator, cfg.hidden_size),
        )
        record = {
            "calibrator_training": calibration_training,
            "primary_training": training,
            "calibrator_reproduction_max": difference,
            "parameter_counts": {
                "leaf_calibrator": v3.trainable_count(calibrator),
                "primary_trainable": v3.trainable_count(model),
            },
            "lengths": {},
        }
        for length in v3.LENGTHS:
            record["lengths"][str(length)] = v3.evaluate(
                data, length, cfg, model
            )
        result["seeds"][str(seed)] = record
        print(f"completed calibrated-direct fold={fold} seed={seed}", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--folds", nargs="+", type=int, default=list(v3.FOLDS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(v3.SEEDS))
    args = parser.parse_args()
    folds, seeds = tuple(args.folds), tuple(args.seeds)
    if any(x not in v3.FOLDS for x in folds) or any(x not in v3.SEEDS for x in seeds):
        raise SystemExit("folds and seeds must be drawn from 0..4")
    source = json.loads(SOURCE_RESULTS.read_text())
    cfg = v3.Config(**source["config"])
    all_data = v3.ETH3DAll()
    result = {
        "study": "ETH3D calibrated direct control v1, frozen 2026-09-05",
        "source_results_sha256": v3.sha256(SOURCE_RESULTS),
        "config": asdict(cfg), "folds_requested": list(folds),
        "seeds_requested": list(seeds), "folds": {},
    }
    for fold in folds:
        result["folds"][str(fold)] = run_fold(
            all_data, source, fold, seeds, cfg, torch.device(args.device)
        )
    complete = folds == v3.FOLDS and seeds == v3.SEEDS
    destination = HERE / (
        "results.json" if complete else
        f"results_fold{'-'.join(map(str, folds))}_seed{'-'.join(map(str, seeds))}.json"
    )
    if destination.exists():
        raise RuntimeError(f"refusing to overwrite {destination}")
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
