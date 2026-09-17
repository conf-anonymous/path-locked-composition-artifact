"""Residual learned composition over an invariant public-motion baseline."""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
BASE_DIR = HERE.parent / "017_tum_motor_composition"
sys.path.insert(0, str(BASE_DIR))
spec = importlib.util.spec_from_file_location("tum_motor_base", BASE_DIR / "run.py")
base = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = base
spec.loader.exec_module(base)

RUNS = HERE / "runs"
ARMS = ("bilinear", "mlp", "mixed", "penalty")


def decoupled_pair(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    qa, ta = base.decode_motor(a)
    qb, tb = base.decode_motor(b)
    return base.make_motor(base.qmul(qa, qb), ta + tb)


class ResidualComposer(nn.Module):
    def __init__(self, arm: str):
        super().__init__()
        self.arm = arm
        if arm == "mlp":
            self.network = nn.Sequential(nn.Linear(16, 20), nn.GELU(), nn.Linear(20, 8))
            nn.init.zeros_(self.network[-1].weight)
            nn.init.zeros_(self.network[-1].bias)
        else:
            self.table = nn.Parameter(torch.zeros(8, 8, 8))

    def op(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        foundation = decoupled_pair(a, b)
        if self.arm == "mlp":
            residual = self.network(torch.cat((a, b), dim=-1))
        else:
            residual = torch.einsum("bi,ijk,bj->bk", a, self.table, b)
        return base.normalize_motor(foundation + residual)

    def forward(self, operands: torch.Tensor, path: str, rng: random.Random | None = None) -> torch.Tensor:
        return base.reduce_states(list(operands.unbind(1)), self.op, path, rng)


def train_one(data, arm: str, seed: int, cfg, device: torch.device) -> dict:
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed + 10_000)
    rng = random.Random(seed + 20_000)
    model = ResidualComposer(arm).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    losses = []
    started = time.perf_counter()
    model.train()
    for step in range(1, cfg.steps + 1):
        indices = torch.randint(len(data.train_examples), (cfg.batch_size,), generator=generator)
        operands, target, _ = data.batch("train", base.TRAIN_LENGTH, indices)
        operands = operands.to(device=device, dtype=torch.float32)
        target = target.to(device=device, dtype=torch.float32)
        path = "left"
        if arm == "mixed":
            path = ("left", "right", "balanced", "random")[rng.randrange(4)]
        prediction = model(operands, path, rng if path == "random" else None)
        loss = base.motor_loss(prediction, target, data.translation_scale)
        if arm == "penalty":
            start = rng.randrange(base.TRAIN_LENGTH - 2)
            a, b, c = (operands[:, start + offset] for offset in range(3))
            associator = model.op(model.op(a, b), c) - model.op(a, model.op(b, c))
            loss = loss + cfg.penalty_weight * associator.square().mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
        if step % 1000 == 0:
            print(f"{arm} seed={seed} step={step} loss={np.mean(losses[-100:]):.6f}", flush=True)
    return {"model": model, "training_seconds": time.perf_counter() - started,
            "final_loss": float(np.mean(losses[-100:]))}


def run(split: str, cfg, device: torch.device) -> dict:
    data = base.TUMData(cfg)
    RUNS.mkdir(parents=True, exist_ok=True)
    output = {
        "protocol_frozen": "2026-09-03",
        "predecessor": "Experiment 017 development gate failed; confirmation unopened",
        "split": split,
        "config": asdict(cfg),
        "sequence_counts": {key: sum(base.split_for(name) == key for name in base.SEQUENCES)
                            for key in ("train", "dev", "confirm")},
        "translation_scale_m": data.translation_scale,
        "baselines": {}, "exact_audit": {}, "arms": {},
    }
    for length in base.EVAL_LENGTHS:
        output["baselines"][str(length)] = {
            "motor": base.evaluate(data, split, length, "motor", cfg),
            "decoupled": base.evaluate(data, split, length, "decoupled", cfg),
        }
        output["exact_audit"][str(length)] = base.exact_path_audit(data, split, length, cfg)

    for arm in ARMS:
        output["arms"][arm] = {}
        for seed in base.SEEDS:
            checkpoint = RUNS / f"{arm}_seed{seed}.pt"
            if split == "dev":
                trained = train_one(data, arm, seed, cfg, device)
                model = trained.pop("model")
                torch.save({"state_dict": model.state_dict(), "arm": arm, "seed": seed,
                            "config": asdict(cfg)}, checkpoint)
            else:
                if not checkpoint.exists():
                    raise FileNotFoundError(f"frozen checkpoint missing: {checkpoint}")
                saved = torch.load(checkpoint, map_location=device, weights_only=True)
                model = ResidualComposer(arm).to(device)
                model.load_state_dict(saved["state_dict"])
                trained = {"training_seconds": None, "final_loss": None}
            record = {"training": trained, "lengths": {}}
            for length in base.EVAL_LENGTHS:
                paths = {path: base.evaluate(data, split, length, path, cfg, model, seed)
                         for path in base.PATHS}
                paths["random"] = base.random_tree_evaluation(data, split, length, cfg, model, seed)
                intervals = {
                    path: base.bootstrap_delta(paths["left"], paths[path],
                                               seed * 10_000 + length * 10 + index,
                                               cfg.bootstrap_resamples)
                    for index, path in enumerate(("right", "balanced", "random"))
                }
                record["lengths"][str(length)] = {
                    "paths": paths, "translation_deltas": intervals,
                    "left_vs_decoupled": base.bootstrap_delta(
                        output["baselines"][str(length)]["decoupled"], paths["left"],
                        900_000 + seed * 100 + length, cfg.bootstrap_resamples),
                }
            output["arms"][arm][str(seed)] = record
            print(f"evaluated {split}: {arm} seed={seed}", flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    split = "confirm" if args.confirm else "dev"
    result = run(split, base.Config(), torch.device(args.device))
    destination = HERE / ("results_confirmation.json" if args.confirm else "results_development.json")
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
