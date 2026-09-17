"""Profile frozen ETH3D architectures on recorded public-data operands only."""

from __future__ import annotations

import importlib.util
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
V3_DIR = HERE.parent / "023_eth3d_family_heldout"
V4_DIR = HERE.parent / "024_eth3d_calibrated_direct_control"


def load_module(name: str, path: Path):
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


v3 = load_module("compute_family_v3", V3_DIR / "run.py")
v4 = load_module("compute_control_v4", V4_DIR / "run.py")


def metric_difference(expected: dict, observed: dict) -> float:
    maximum = 0.0
    for family, values in expected["clusters"].items():
        for metric in ("translation_mean_m", "rotation_mean_deg"):
            maximum = max(
                maximum,
                abs(values[metric] - observed["clusters"][family][metric]),
            )
    return maximum


def latency(model, operands: torch.Tensor, path: str | None = None) -> dict:
    def invoke():
        output = model(operands) if path is None else model(operands, path)
        # Materialize one recorded-input result so lazy execution cannot escape timing.
        return float(output[0, 0])

    with torch.inference_mode():
        for _ in range(20):
            invoke()
        timings = []
        for _ in range(100):
            started = time.perf_counter_ns()
            invoke()
            timings.append((time.perf_counter_ns() - started) / 1e6)
    q1, q3 = np.quantile(timings, (0.25, 0.75))
    median = statistics.median(timings)
    return {
        "median_ms": median,
        "q1_ms": float(q1),
        "q3_ms": float(q3),
        "examples_per_second": float(len(operands) * 1000.0 / median),
        "warmup_calls": 20,
        "timed_calls": 100,
    }


class ExactReduction(torch.nn.Module):
    def __init__(self, base):
        super().__init__()
        self.base = base

    def forward(self, operands: torch.Tensor, path: str):
        return self.base.reduce_states(
            list(operands.unbind(1)), self.base.motor_product, path
        )


def training_times(source: dict, contextual_key: str) -> list[float]:
    return [
        float(seed["training"][contextual_key]["seconds"])
        for fold in source["folds"].values()
        for seed in fold["seeds"].values()
    ]


def direct_training_times(source: dict) -> list[float]:
    return [
        float(seed["primary_training"]["seconds"])
        for fold in source["folds"].values()
        for seed in fold["seeds"].values()
    ]


def describe(values: list[float]) -> dict:
    q1, q3 = np.quantile(values, (0.25, 0.75))
    return {
        "n": len(values),
        "median_seconds": statistics.median(values),
        "q1_seconds": float(q1),
        "q3_seconds": float(q3),
    }


def main() -> None:
    torch.set_num_threads(1)
    device = torch.device("cpu")
    source3 = json.loads((V3_DIR / "results.json").read_text())
    source4 = json.loads((V4_DIR / "results.json").read_text())
    cfg = v3.Config(**source3["config"])
    all_data = v3.ETH3DAll()
    data = v3.FoldData(all_data, 0)

    calibrator_ga, _ = v3.train_calibrator(data, cfg, 0, device)
    contextual, _ = v3.train_primary(
        data,
        cfg,
        0,
        device,
        v3.ContextualGA(calibrator_ga.calibrator, cfg.hidden_size),
    )
    observed_ga = v3.evaluate(data, 32, cfg, contextual)
    expected_ga = source3["folds"]["0"]["seeds"]["0"]["lengths"]["32"][
        "contextual_ga"
    ]
    ga_difference = metric_difference(expected_ga, observed_ga)

    calibrator_direct, _ = v3.train_calibrator(data, cfg, 0, device)
    direct, _ = v3.train_primary(
        data,
        cfg,
        0,
        device,
        v4.CalibratedDirectGRU(calibrator_direct.calibrator, cfg.hidden_size),
    )
    observed_direct = v3.evaluate(data, 32, cfg, direct)
    expected_direct = source4["folds"]["0"]["seeds"]["0"]["lengths"]["32"]
    direct_difference = metric_difference(expected_direct, observed_direct)
    if max(ga_difference, direct_difference) > 1e-6:
        raise RuntimeError(
            "refusing cost measurement: retrained model did not reproduce frozen metrics"
        )

    contextual.eval()
    direct.eval()
    exact = ExactReduction(v3.base).eval()
    operands_all = data.tensors[("test", 32)][0].to(dtype=torch.float32)
    inference = {}
    for requested in (1, 32, 256):
        count = min(requested, len(operands_all))
        operands = operands_all[:count]
        inference[str(requested)] = {
            "actual_batch_size": count,
            "contextual_ga": latency(contextual, operands),
            "calibrated_direct_gru": latency(direct, operands),
            "exact_motor_left": latency(exact, operands, "left"),
            "exact_motor_balanced": latency(exact, operands, "balanced"),
        }

    result = {
        "study": "ETH3D frozen architecture compute profile v1",
        "data_boundary": "recorded checksum-verified ETH3D operands only",
        "selection_status": "post-freeze descriptive diagnostic",
        "fold": 0,
        "seed": 0,
        "length": 32,
        "reproduction_max": {
            "contextual_ga": ga_difference,
            "calibrated_direct_gru": direct_difference,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "torch_threads": torch.get_num_threads(),
            "device": str(device),
        },
        "parameter_counts": {
            "shared_leaf_calibrator": v3.trainable_count(calibrator_ga),
            "contextual_ga_primary": v3.trainable_count(contextual),
            "calibrated_direct_gru_primary": v3.trainable_count(direct),
        },
        "recorded_training_time": {
            "contextual_ga_primary": describe(training_times(source3, "contextual_ga")),
            "calibrated_direct_gru_primary": describe(direct_training_times(source4)),
            "shared_leaf_calibrator": describe(training_times(source3, "ga_calibrated")),
        },
        "inference": inference,
    }
    destination = HERE / "results.json"
    if destination.exists():
        raise RuntimeError(f"refusing to overwrite {destination}")
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
