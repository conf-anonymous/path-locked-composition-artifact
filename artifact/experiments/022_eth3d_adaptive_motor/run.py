"""Development-only adaptive PGA-motor correction on public ETH3D data."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
BASE_DIR = HERE.parent / "021_eth3d_rgbd_motor"
RUNS = HERE / "runs"

sys.path.insert(0, str(BASE_DIR))
spec = importlib.util.spec_from_file_location("eth3d_v1", BASE_DIR / "run.py")
v1 = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = v1
spec.loader.exec_module(v1)

base = v1.study.base
LENGTHS = (4, 8, 16, 32)


@dataclass(frozen=True)
class Config:
    steps: int = 5000
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    hidden_size: int = 32
    evaluation_batch_size: int = 1024
    bootstrap_resamples: int = 10_000
    moving_threshold_m: float = 0.15


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def correction_motor(twist: torch.Tensor) -> torch.Tensor:
    """Map an endpoint-frame rotation/translation twist to a unit motor."""
    rotation, translation = twist[..., :3], twist[..., 3:]
    angle = torch.linalg.vector_norm(rotation, dim=-1, keepdim=True)
    half = 0.5 * angle
    # sin(angle / 2) / angle = sinc(angle / (2*pi)) / 2, including angle=0.
    scale = 0.5 * torch.sinc(half / math.pi)
    quaternion = torch.cat((torch.cos(half), rotation * scale), dim=-1)
    return base.make_motor(quaternion, translation)


class AdaptiveMotor(nn.Module):
    def __init__(self, hidden_size: int, translation_scale: float):
        super().__init__()
        self.translation_scale = translation_scale
        self.encoder = nn.GRU(
            input_size=8, hidden_size=hidden_size, batch_first=True,
            bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.Linear(2 * hidden_size + 8, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 6),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, operands: torch.Tensor, path: str = "balanced") -> torch.Tensor:
        raw = base.reduce_states(
            list(operands.unbind(1)), base.motor_product, path
        )
        encoded, _ = self.encoder(operands)
        context = torch.cat((encoded.mean(1), raw), dim=-1)
        twist = self.head(context)
        # Keep initial and learned correction scales physically interpretable.
        bounded = torch.cat((
            math.pi * torch.tanh(twist[..., :3]),
            (4.0 * self.translation_scale) * torch.tanh(twist[..., 3:]),
        ), dim=-1)
        return base.motor_product(raw, correction_motor(bounded))


class GatedCalibrator(nn.Module):
    """Contextually choose between identity and a frozen GA leaf calibration."""
    def __init__(self, hidden_size: int, seed: int):
        super().__init__()
        saved = torch.load(
            BASE_DIR / "runs" / f"ga_calibrated_seed{seed}.pt",
            map_location="cpu", weights_only=True,
        )
        legacy = v1.study.OxfordComposer("ga_calibrated")
        legacy.load_state_dict(saved["state_dict"])
        self.calibrator = legacy.calibrator
        for parameter in self.calibrator.parameters():
            parameter.requires_grad_(False)
        self.encoder = nn.GRU(
            input_size=8, hidden_size=hidden_size, batch_first=True,
            bidirectional=True,
        )
        self.gate = nn.Sequential(
            nn.Linear(2 * hidden_size + 16, hidden_size), nn.GELU(),
            nn.Linear(hidden_size, 1),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, -2.0)

    def forward(self, operands: torch.Tensor, path: str = "balanced") -> torch.Tensor:
        raw = base.reduce_states(list(operands.unbind(1)), base.motor_product, path)
        calibrated_leaves = self.calibrator(operands)
        calibrated = base.reduce_states(
            list(calibrated_leaves.unbind(1)), base.motor_product, path
        )
        encoded, _ = self.encoder(operands)
        context = torch.cat((encoded.mean(1), raw, calibrated), dim=-1)
        weight = torch.sigmoid(self.gate(context))
        return base.normalize_motor(raw + weight * (calibrated - raw))


def robust_motor_loss(prediction: torch.Tensor, target: torch.Tensor,
                      translation_scale: float) -> torch.Tensor:
    pred_q, pred_t = base.decode_motor(prediction)
    target_q, target_t = base.decode_motor(target)
    translation = F.smooth_l1_loss(
        pred_t / translation_scale, target_t / translation_scale,
        reduction="none", beta=1.0,
    ).sum(-1)
    rotation = 1.0 - (pred_q * target_q).sum(-1).square().clamp(max=1.0)
    return (translation + 2.0 * rotation).mean()


class BalancedSampler:
    def __init__(self, data: v1.ETH3DData):
        self.indices = {}
        for length in LENGTHS:
            by_name = {}
            for index, name in enumerate(data.examples[("train", length)]):
                by_name.setdefault(name, []).append(index)
            self.indices[length] = {
                name: torch.tensor(values, dtype=torch.long)
                for name, values in by_name.items()
            }

    def sample(self, length: int, batch_size: int,
               generator: torch.Generator) -> torch.Tensor:
        groups = self.indices[length]
        names = sorted(groups)
        chosen_groups = torch.randint(
            len(names), (batch_size,), generator=generator
        )
        result = []
        for group_index in chosen_groups.tolist():
            values = groups[names[group_index]]
            position = int(torch.randint(len(values), (1,), generator=generator))
            result.append(int(values[position]))
        return torch.tensor(result, dtype=torch.long)


def train_one(data: v1.ETH3DData, cfg: Config, seed: int,
              device: torch.device, model_kind: str) -> tuple[nn.Module, dict]:
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed + 10_000)
    sampler = BalancedSampler(data)
    model = (
        GatedCalibrator(cfg.hidden_size, seed)
        if model_kind == "gated" else AdaptiveMotor(cfg.hidden_size, data.translation_scale)
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    losses = []
    started = time.perf_counter()
    model.train()
    for step in range(1, cfg.steps + 1):
        length = (32 if model_kind == "gated" else
                  LENGTHS[int(torch.randint(len(LENGTHS), (1,), generator=generator))])
        indices = sampler.sample(length, cfg.batch_size, generator)
        operands, target, _, _ = data.batch("train", length, indices)
        operands = operands.to(device=device, dtype=torch.float32)
        target = target.to(device=device, dtype=torch.float32)
        loss = robust_motor_loss(model(operands), target, data.translation_scale)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
        if step % 1000 == 0:
            print(f"seed={seed} step={step} loss={np.mean(losses[-100:]):.6f}", flush=True)
    return model, {
        "training_seconds": time.perf_counter() - started,
        "final_loss": float(np.mean(losses[-100:])),
    }


@torch.no_grad()
def evaluate(data: v1.ETH3DData, split: str, length: int, cfg: Config,
             model: nn.Module | None) -> dict:
    names_all = data.examples[(split, length)]
    clusters: dict[str, list[tuple[float, float]]] = {}
    trans_all, rot_all = [], []
    for offset in range(0, len(names_all), cfg.evaluation_batch_size):
        indices = range(offset, min(offset + cfg.evaluation_batch_size, len(names_all)))
        operands, target, _, names = data.batch(split, length, indices)
        if model is None:
            prediction = base.reduce_states(
                list(operands.unbind(1)), base.motor_product, "balanced"
            )
        else:
            model.eval()
            prediction = model(
                operands.to(next(model.parameters()).device, dtype=torch.float32)
            ).cpu().double()
        trans, rot = base.errors(prediction, target)
        trans_all.append(np.asarray(trans))
        rot_all.append(np.asarray(rot))
        for name, t_error, r_error in zip(names, trans, rot):
            clusters.setdefault(name, []).append((float(t_error), float(r_error)))
    translation = np.concatenate(trans_all)
    rotation = np.concatenate(rot_all)
    return {
        "n_windows": int(len(translation)),
        "n_sequences": len(clusters),
        "translation_mean_m": float(translation.mean()),
        "translation_median_m": float(np.median(translation)),
        "rotation_mean_deg": float(rotation.mean()),
        "rotation_median_deg": float(np.median(rotation)),
        "clusters": {
            name: {
                "translation_mean_m": float(np.mean([item[0] for item in values])),
                "rotation_mean_deg": float(np.mean([item[1] for item in values])),
                "n": len(values),
            }
            for name, values in clusters.items()
        },
    }


def paired_bootstrap(raw: dict, adaptive: dict, seed: int,
                     resamples: int) -> dict:
    names = sorted(raw["clusters"])
    differences = np.asarray([
        adaptive["clusters"][name]["translation_mean_m"]
        - raw["clusters"][name]["translation_mean_m"] for name in names
    ])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(names), size=(resamples, len(names)))
    distribution = differences[draws].mean(1)
    return {
        "sequence_equal_delta_m": float(differences.mean()),
        "ci95_m": [float(x) for x in np.quantile(distribution, (0.025, 0.975))],
        "improved_sequences": int((differences < 0).sum()),
        "total_sequences": len(names),
    }


def exact_audit(data: v1.ETH3DData, length: int, cfg: Config,
                model: nn.Module) -> dict:
    maximum = 0.0
    exact_model = copy.deepcopy(model).cpu().double()
    for offset in range(0, len(data.examples[("dev", length)]), cfg.evaluation_batch_size):
        indices = range(offset, min(
            offset + cfg.evaluation_batch_size, len(data.examples[("dev", length)])
        ))
        operands, _, _, _ = data.batch("dev", length, indices)
        predictions = [exact_model(operands, path) for path in ("left", "right", "balanced")]
        q0, t0 = base.decode_motor(predictions[0])
        for other in predictions[1:]:
            q1, t1 = base.decode_motor(other)
            maximum = max(maximum, float(torch.linalg.vector_norm(t0 - t1, dim=-1).max()))
            relative = base.qmul(base.qconj(q0), q1)
            angle = 2 * torch.atan2(
                torch.linalg.vector_norm(relative[..., 1:], dim=-1),
                relative[..., :1].abs().squeeze(-1),
            )
            maximum = max(maximum, float(angle.max()))
    return {"worst_si": maximum, "tolerance": 1e-9, "passes": maximum < 1e-9}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=Config.steps)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(5)))
    parser.add_argument("--model", choices=("adaptive", "gated"), default="adaptive")
    args = parser.parse_args()
    if any(seed not in range(5) for seed in args.seeds):
        raise SystemExit("seeds must be drawn from 0..4")
    cfg = Config(steps=args.steps)
    legacy_cfg = v1.study.Config(
        steps=1, sample_period_seconds=0.1, maximum_alignment_seconds=0.02,
        moving_threshold_m=cfg.moving_threshold_m,
        evaluation_batch_size=cfg.evaluation_batch_size,
        bootstrap_resamples=cfg.bootstrap_resamples,
    )
    data = v1.ETH3DData(legacy_cfg, "dev")
    if set(key[0] for key in data.tensors) != {"train", "dev"}:
        raise RuntimeError("development process loaded a non-development split")
    RUNS.mkdir(parents=True, exist_ok=True)
    output = {
        "study": f"ETH3D adaptive motor development v1/{args.model}",
        "model_kind": args.model,
        "exploratory": (args.steps != Config.steps or args.seeds != list(range(5))
                        or args.model != "adaptive"),
        "confirmation_accessible": False,
        "config": asdict(cfg),
        "seeds": args.seeds,
        "parameter_count": None,
        "translation_scale_m": data.translation_scale,
        "lengths": {},
        "models": {},
    }
    for length in LENGTHS:
        output["lengths"][str(length)] = {"raw": evaluate(data, "dev", length, cfg, None)}
    for seed in args.seeds:
        model, training = train_one(
            data, cfg, seed, torch.device(args.device), args.model
        )
        output["parameter_count"] = sum(p.numel() for p in model.parameters())
        checkpoint = RUNS / f"{args.model}_seed{seed}_steps{args.steps}.pt"
        torch.save({
            "state_dict": model.state_dict(), "seed": seed,
            "config": asdict(cfg), "study": output["study"],
        }, checkpoint)
        record = {"training": training, "checkpoint_sha256": sha256(checkpoint), "lengths": {}}
        for length in LENGTHS:
            adaptive = evaluate(data, "dev", length, cfg, model)
            raw = output["lengths"][str(length)]["raw"]
            record["lengths"][str(length)] = {
                "adaptive": adaptive,
                "paired": paired_bootstrap(raw, adaptive, seed * 100 + length,
                                            cfg.bootstrap_resamples),
                "exact_audit": exact_audit(data, length, cfg, model),
            }
        output["models"][str(seed)] = record
        print(f"evaluated development seed={seed}", flush=True)
    destination = HERE / (
        "results_development.json" if not output["exploratory"]
        else f"results_pilot_{args.model}_{'-'.join(map(str, args.seeds))}_{args.steps}.json"
    )
    destination.write_text(json.dumps(output, indent=2) + "\n")
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
