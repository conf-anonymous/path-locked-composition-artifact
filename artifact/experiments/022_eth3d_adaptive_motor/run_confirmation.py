"""Single frozen confirmation evaluation for the ETH3D contextual motor gate."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from dataclasses import asdict
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("adaptive_dev", HERE / "run.py")
dev = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = dev
spec.loader.exec_module(dev)

EXPECTED_RESULT = "165ed43af1aa531f5f8c551e92751c7f127b49fbd418088e3221562be6897b44"
EXPECTED_CHECKPOINTS = {
    0: "0d0fa073cdabf8bc9b5dd49821516989ef309f550b72c456fa257aeb2c586986",
    1: "39c689e0fe5b6a79b06bce99d864616c0af678b0d66a6b792f262bf0cb84ae9c",
    2: "5e983f29e40eeecd6e6ce2cd68c2eca0c1e8154a1c7f5c7b90fa4d75acfd5941",
    3: "6e836068b0720be2e5d72b032b033a2c964719709fa414c19e23355ec32b7a55",
    4: "2c645dcc8f603606dbf9e099b1d12ca84bfe334dd908b82be8b6171a3ba59f53",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


@torch.no_grad()
def exact_prediction_audit(data, length: int, cfg, model) -> dict:
    exact_model = copy.deepcopy(model).cpu().double()
    maxima = {"right": 0.0, "balanced": 0.0}
    examples = data.examples[("confirm", length)]
    for offset in range(0, len(examples), cfg.evaluation_batch_size):
        indices = range(offset, min(offset + cfg.evaluation_batch_size, len(examples)))
        operands, _, _, _ = data.batch("confirm", length, indices)
        predictions = {
            path: exact_model(operands, path) for path in ("left", "right", "balanced")
        }
        q0, t0 = dev.base.decode_motor(predictions["left"])
        for path in ("right", "balanced"):
            q1, t1 = dev.base.decode_motor(predictions[path])
            translation = torch.linalg.vector_norm(t0 - t1, dim=-1).max().item()
            relative = dev.base.qmul(dev.base.qconj(q0), q1)
            rotation = (2 * torch.atan2(
                torch.linalg.vector_norm(relative[..., 1:], dim=-1),
                relative[..., :1].abs().squeeze(-1),
            )).max().item()
            maxima[path] = max(maxima[path], translation, rotation)
    worst = max(maxima.values())
    return {"maxima_si": maxima, "worst_si": worst, "tolerance": 1e-9,
            "passes": worst < 1e-9}


def main() -> None:
    development_path = HERE / "results_pilot_gated_0-1-2-3-4_5000.json"
    if sha256(development_path) != EXPECTED_RESULT:
        raise RuntimeError("frozen development record hash mismatch")
    development = json.loads(development_path.read_text())
    cfg = dev.Config(**development["config"])
    legacy_cfg = dev.v1.study.Config(
        steps=1, sample_period_seconds=0.1, maximum_alignment_seconds=0.02,
        moving_threshold_m=cfg.moving_threshold_m,
        evaluation_batch_size=cfg.evaluation_batch_size,
        bootstrap_resamples=cfg.bootstrap_resamples,
    )
    data = dev.v1.ETH3DData(legacy_cfg, "confirm")
    if set(key[0] for key in data.tensors) != {"train", "confirm"}:
        raise RuntimeError("confirmation process loaded an unauthorized split")
    output = {
        "study": "ETH3D contextual motor gate frozen confirmation",
        "protocol_frozen": "2026-09-05",
        "config": asdict(cfg),
        "development_sha256": EXPECTED_RESULT,
        "lengths": {}, "models": {},
    }
    for length in dev.LENGTHS:
        output["lengths"][str(length)] = {
            "raw": dev.evaluate(data, "confirm", length, cfg, None)
        }
    for seed, expected in EXPECTED_CHECKPOINTS.items():
        checkpoint = HERE / "runs" / f"gated_seed{seed}_steps5000.pt"
        if sha256(checkpoint) != expected:
            raise RuntimeError(f"frozen checkpoint hash mismatch: seed {seed}")
        saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if saved["seed"] != seed or saved["config"] != asdict(cfg):
            raise RuntimeError(f"frozen checkpoint metadata mismatch: seed {seed}")
        model = dev.GatedCalibrator(cfg.hidden_size, seed)
        model.load_state_dict(saved["state_dict"])
        record = {"checkpoint_sha256": expected, "lengths": {}}
        for length in dev.LENGTHS:
            adaptive = dev.evaluate(data, "confirm", length, cfg, model)
            raw = output["lengths"][str(length)]["raw"]
            record["lengths"][str(length)] = {
                "adaptive": adaptive,
                "paired": dev.paired_bootstrap(
                    raw, adaptive, seed * 100 + length, cfg.bootstrap_resamples
                ),
                "exact_audit": exact_prediction_audit(data, length, cfg, model),
            }
        output["models"][str(seed)] = record
        print(f"evaluated frozen confirmation seed={seed}", flush=True)
    destination = HERE / "results_confirmation.json"
    if destination.exists():
        raise RuntimeError("confirmation result already exists; refusing a second attempt")
    destination.write_text(json.dumps(output, indent=2) + "\n")
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
