"""Path-locking on public recorded TUM RGB-D motion-capture trajectories.

Development is the default. ``--confirm`` evaluates frozen checkpoints on the
held-out confirmation trajectories. No image, pose, motion, label, negative,
or perturbation is generated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from data_manifest import SEQUENCES

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DATA = ROOT / "data" / "raw" / "tum_rgbd_groundtruth"
RUNS = HERE / "runs"
WIDTH = 8
TRAIN_LENGTH = 8
EVAL_LENGTHS = (4, 8, 16, 32)
PATHS = ("left", "right", "balanced")
RANDOM_TREES = 16
SEEDS = tuple(range(5))
ARMS = ("bilinear", "mlp", "mixed", "penalty")


@dataclass(frozen=True)
class Config:
    steps: int = 5000
    batch_size: int = 256
    learning_rate: float = 2e-3
    weight_decay: float = 1e-5
    penalty_weight: float = 0.1
    evaluation_batch_size: int = 1024
    sample_period_seconds: float = 0.1
    bootstrap_resamples: int = 10_000


def split_for(name: str) -> str:
    bucket = int(hashlib.sha256(f"tum-rgbd-v1:{name}".encode()).hexdigest()[:8], 16) % 10
    return "train" if bucket < 5 else ("dev" if bucket < 7 else "confirm")


def qmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack((
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ), dim=-1)


def qconj(q: torch.Tensor) -> torch.Tensor:
    return torch.cat((q[..., :1], -q[..., 1:]), dim=-1)


def qrotate(q: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    pure = torch.cat((torch.zeros_like(vector[..., :1]), vector), dim=-1)
    return qmul(qmul(q, pure), qconj(q))[..., 1:]


def make_motor(q: torch.Tensor, translation: torch.Tensor) -> torch.Tensor:
    pure = torch.cat((torch.zeros_like(translation[..., :1]), translation), dim=-1)
    dual = 0.5 * qmul(pure, q)
    return torch.cat((q, dual), dim=-1)


def motor_product(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    ar, ad = a[..., :4], a[..., 4:]
    br, bd = b[..., :4], b[..., 4:]
    return torch.cat((qmul(ar, br), qmul(ar, bd) + qmul(ad, br)), dim=-1)


def normalize_motor(motor: torch.Tensor) -> torch.Tensor:
    real, dual = motor[..., :4], motor[..., 4:]
    norm = torch.linalg.vector_norm(real, dim=-1, keepdim=True).clamp_min(1e-8)
    real = real / norm
    dual = dual / norm
    dual = dual - real * (real * dual).sum(-1, keepdim=True)
    return torch.cat((real, dual), dim=-1)


def decode_motor(motor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    motor = normalize_motor(motor)
    real, dual = motor[..., :4], motor[..., 4:]
    translation = 2.0 * qmul(dual, qconj(real))[..., 1:]
    return real, translation


def relative_motor(q0: torch.Tensor, t0: torch.Tensor, q1: torch.Tensor, t1: torch.Tensor) -> torch.Tensor:
    relative_q = qmul(qconj(q0), q1)
    relative_t = qrotate(qconj(q0), t1 - t0)
    return make_motor(relative_q, relative_t)


def reduce_states(states: list[torch.Tensor], op, path: str, rng: random.Random | None = None) -> torch.Tensor:
    if len(states) == 1:
        return states[0]
    if path == "left":
        out = states[0]
        for state in states[1:]:
            out = op(out, state)
        return out
    if path == "right":
        out = states[-1]
        for state in reversed(states[:-1]):
            out = op(state, out)
        return out
    if path == "balanced":
        split = len(states) // 2
    elif path == "random":
        if rng is None:
            raise ValueError("random path requires an RNG")
        split = rng.randrange(1, len(states))
    else:
        raise ValueError(path)
    return op(
        reduce_states(states[:split], op, path, rng),
        reduce_states(states[split:], op, path, rng),
    )


class Trajectory:
    def __init__(self, name: str, path: Path, period: float):
        rows = []
        for line in path.read_text().splitlines():
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) != 8:
                raise RuntimeError(f"malformed row in {path}: {line[:80]}")
            rows.append(tuple(float(field) for field in fields))
        values = np.asarray(rows, dtype=np.float64)
        if len(values) < 2 or not np.all(np.diff(values[:, 0]) >= 0):
            raise RuntimeError(f"invalid timestamps in {path}")

        first, last = values[0, 0], values[-1, 0]
        requested = first + np.arange(math.floor((last - first) / period) + 1) * period
        indices = np.searchsorted(values[:, 0], requested, side="left")
        indices = np.unique(indices[indices < len(values)])
        selected = values[indices]
        translation = selected[:, 1:4]
        quaternion = selected[:, [7, 4, 5, 6]]  # source xyzw -> internal wxyz
        quaternion /= np.linalg.norm(quaternion, axis=1, keepdims=True)
        for index in range(1, len(quaternion)):
            if np.dot(quaternion[index - 1], quaternion[index]) < 0:
                quaternion[index] *= -1

        self.name = name
        self.timestamps = torch.from_numpy(selected[:, 0].copy())
        self.translation = torch.from_numpy(translation.copy())
        self.quaternion = torch.from_numpy(quaternion.copy())
        self.increments = relative_motor(
            self.quaternion[:-1], self.translation[:-1],
            self.quaternion[1:], self.translation[1:],
        )

    def target(self, start: torch.Tensor, length: int) -> torch.Tensor:
        return relative_motor(
            self.quaternion[start], self.translation[start],
            self.quaternion[start + length], self.translation[start + length],
        )


class TUMData:
    def __init__(self, cfg: Config):
        missing = [name for name in SEQUENCES if not (DATA / f"{name}.txt").exists()]
        if missing:
            raise FileNotFoundError(
                f"missing {len(missing)} official trajectory files under {DATA}; "
                f"run {HERE / 'download_data.py'}"
            )
        self.trajectories = {
            name: Trajectory(name, DATA / f"{name}.txt", cfg.sample_period_seconds)
            for name in SEQUENCES
        }
        too_short = [name for name, traj in self.trajectories.items() if len(traj.increments) < max(EVAL_LENGTHS)]
        if too_short:
            raise RuntimeError(f"trajectories too short after recorded-row selection: {too_short}")
        self.examples: dict[tuple[str, int], list[str]] = {}
        self.tensors: dict[tuple[str, int], tuple[torch.Tensor, torch.Tensor]] = {}
        for split in ("train", "dev", "confirm"):
            for length in EVAL_LENGTHS:
                operands, targets, names = [], [], []
                for name, trajectory in self.trajectories.items():
                    if split_for(name) != split:
                        continue
                    count = len(trajectory.increments) - length + 1
                    # Tensor.unfold appends the window dimension after the
                    # coordinate dimension: [N, 8, L] -> [N, L, 8].
                    windows = trajectory.increments.unfold(0, length, 1).permute(0, 2, 1)
                    target = relative_motor(
                        trajectory.quaternion[:count], trajectory.translation[:count],
                        trajectory.quaternion[length:length + count],
                        trajectory.translation[length:length + count],
                    )
                    operands.append(windows.contiguous())
                    targets.append(target)
                    names.extend([name] * count)
                self.examples[(split, length)] = names
                self.tensors[(split, length)] = (torch.cat(operands), torch.cat(targets))
        self.train_examples = self.examples[("train", TRAIN_LENGTH)]
        _, train_targets = self.tensors[("train", TRAIN_LENGTH)]
        _, train_translation = decode_motor(train_targets)
        lengths = torch.linalg.vector_norm(train_translation, dim=-1).numpy()
        self.translation_scale = max(float(np.median(lengths)), 1e-3)

    def batch(self, split: str, length: int, indices) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
        index = torch.as_tensor(list(indices) if isinstance(indices, range) else indices, dtype=torch.long)
        operands, targets = self.tensors[(split, length)]
        names = [self.examples[(split, length)][int(item)] for item in index]
        return operands[index], targets[index], names


class LearnedComposer(nn.Module):
    def __init__(self, arm: str):
        super().__init__()
        self.arm = arm
        if arm == "mlp":
            self.network = nn.Sequential(nn.Linear(16, 20), nn.GELU(), nn.Linear(20, 8))
        else:
            self.table = nn.Parameter(torch.empty(8, 8, 8))
            nn.init.normal_(self.table, std=0.08)

    def op(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        if self.arm == "mlp":
            raw = self.network(torch.cat((a, b), dim=-1))
        else:
            raw = torch.einsum("bi,ijk,bj->bk", a, self.table, b)
        return normalize_motor(raw)

    def forward(self, operands: torch.Tensor, path: str, rng: random.Random | None = None) -> torch.Tensor:
        return reduce_states(list(operands.unbind(1)), self.op, path, rng)


def motor_loss(prediction: torch.Tensor, target: torch.Tensor, scale: float) -> torch.Tensor:
    pred_q, pred_t = decode_motor(prediction)
    target_q, target_t = decode_motor(target)
    translation = ((pred_t - target_t) / scale).square().sum(-1)
    rotation = 1.0 - (pred_q * target_q).sum(-1).square().clamp(max=1.0)
    return (translation + 2.0 * rotation).mean()


def train_one(data: TUMData, arm: str, seed: int, cfg: Config, device: torch.device) -> dict:
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed + 10_000)
    rng = random.Random(seed + 20_000)
    model = LearnedComposer(arm).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    started = time.perf_counter()
    losses = []
    model.train()
    for step in range(1, cfg.steps + 1):
        indices = torch.randint(len(data.train_examples), (cfg.batch_size,), generator=generator)
        operands, target, _ = data.batch("train", TRAIN_LENGTH, indices)
        operands = operands.to(device=device, dtype=torch.float32)
        target = target.to(device=device, dtype=torch.float32)
        path = "left"
        if arm == "mixed":
            path = ("left", "right", "balanced", "random")[rng.randrange(4)]
        prediction = model(operands, path, rng if path == "random" else None)
        loss = motor_loss(prediction, target, data.translation_scale)
        if arm == "penalty":
            start = rng.randrange(TRAIN_LENGTH - 2)
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
    return {
        "model": model,
        "training_seconds": time.perf_counter() - started,
        "final_loss": float(np.mean(losses[-100:])),
    }


def errors(prediction: torch.Tensor, target: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    pred_q, pred_t = decode_motor(prediction)
    target_q, target_t = decode_motor(target)
    translation = torch.linalg.vector_norm(pred_t - target_t, dim=-1)
    relative = qmul(qconj(target_q), pred_q)
    rotation = 2.0 * torch.atan2(
        torch.linalg.vector_norm(relative[..., 1:], dim=-1), relative[..., :1].abs().squeeze(-1)
    ) * (180.0 / math.pi)
    return translation.cpu().numpy(), rotation.cpu().numpy()


def decoupled_reduce(operands: torch.Tensor) -> torch.Tensor:
    rotations, translations = decode_motor(operands)
    rotation = rotations[:, 0]
    for index in range(1, rotations.shape[1]):
        rotation = qmul(rotation, rotations[:, index])
    return make_motor(rotation, translations.sum(1))


@torch.no_grad()
def evaluate(
    data: TUMData,
    split: str,
    length: int,
    path: str,
    cfg: Config,
    model: LearnedComposer | None = None,
    seed: int = 0,
) -> dict:
    examples = data.examples[(split, length)]
    translation_parts, rotation_parts, dispersion_parts = [], [], []
    cluster_values: dict[str, list[tuple[float, float]]] = {}
    for offset in range(0, len(examples), cfg.evaluation_batch_size):
        indices = range(offset, min(offset + cfg.evaluation_batch_size, len(examples)))
        operands, target, names = data.batch(split, length, indices)
        if model is None and path == "motor":
            prediction = reduce_states(list(operands.unbind(1)), motor_product, "left")
            reference = prediction
        elif model is None and path == "decoupled":
            prediction = decoupled_reduce(operands)
            reference = prediction
        else:
            assert model is not None
            model.eval()
            operands32 = operands.to(next(model.parameters()).device, dtype=torch.float32)
            tree_rng = random.Random(seed * 1_000_003 + length * 1009 + offset)
            prediction = model(operands32, path, tree_rng if path == "random" else None).cpu().double()
            left = model(operands32, "left").cpu().double()
            numerator = torch.linalg.vector_norm(prediction - left, dim=-1)
            denominator = torch.linalg.vector_norm(left, dim=-1).clamp_min(1e-12)
            dispersion_parts.append((numerator / denominator).numpy())
            reference = left
        trans, rot = errors(prediction, target)
        translation_parts.append(trans)
        rotation_parts.append(rot)
        for name, t_error, r_error in zip(names, trans, rot):
            cluster_values.setdefault(name, []).append((float(t_error), float(r_error)))
    translation = np.concatenate(translation_parts)
    rotation = np.concatenate(rotation_parts)
    clusters = {
        name: {
            "translation_mean_m": float(np.mean([value[0] for value in values])),
            "rotation_mean_deg": float(np.mean([value[1] for value in values])),
            "n": len(values),
        }
        for name, values in cluster_values.items()
    }
    result = {
        "n_windows": len(examples),
        "n_trajectories": len(clusters),
        "translation_mean_m": float(translation.mean()),
        "translation_median_m": float(np.median(translation)),
        "rotation_mean_deg": float(rotation.mean()),
        "rotation_median_deg": float(np.median(rotation)),
        "clusters": clusters,
    }
    if dispersion_parts:
        dispersion = np.concatenate(dispersion_parts)
        result["dispersion_mean"] = float(dispersion.mean())
        result["dispersion_median"] = float(np.median(dispersion))
    else:
        result["dispersion_mean"] = 0.0
        result["dispersion_median"] = 0.0
    return result


def random_tree_evaluation(data, split, length, cfg, model, seed) -> dict:
    runs = [evaluate(data, split, length, "random", cfg, model, seed * 100 + index)
            for index in range(RANDOM_TREES)]
    keys = (
        "translation_mean_m", "translation_median_m", "rotation_mean_deg",
        "rotation_median_deg", "dispersion_mean", "dispersion_median",
    )
    result = {key: float(np.mean([run[key] for run in runs])) for key in keys}
    result["n_windows"] = runs[0]["n_windows"]
    result["n_trajectories"] = runs[0]["n_trajectories"]
    result["clusters"] = {
        name: {
            "translation_mean_m": float(np.mean([run["clusters"][name]["translation_mean_m"] for run in runs])),
            "rotation_mean_deg": float(np.mean([run["clusters"][name]["rotation_mean_deg"] for run in runs])),
            "n": runs[0]["clusters"][name]["n"],
        }
        for name in runs[0]["clusters"]
    }
    return result


def bootstrap_delta(left: dict, other: dict, seed: int, resamples: int) -> dict:
    names = sorted(left["clusters"])
    differences = np.asarray([
        other["clusters"][name]["translation_mean_m"]
        - left["clusters"][name]["translation_mean_m"]
        for name in names
    ])
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(names), size=(resamples, len(names)))
    distribution = differences[draws].mean(1)
    return {
        "trajectory_equal_weight_mean_m": float(differences.mean()),
        "ci95_m": [float(value) for value in np.quantile(distribution, (0.025, 0.975))],
    }


def exact_path_audit(data: TUMData, split: str, length: int, cfg: Config) -> dict:
    examples = data.examples[(split, length)]
    maxima = {"right": 0.0, "balanced": 0.0, "random": 0.0}
    for offset in range(0, len(examples), cfg.evaluation_batch_size):
        indices = range(offset, min(offset + cfg.evaluation_batch_size, len(examples)))
        operands, _, _ = data.batch(split, length, indices)
        left = reduce_states(list(operands.unbind(1)), motor_product, "left")
        for path in maxima:
            repetitions = RANDOM_TREES if path == "random" else 1
            for repeat in range(repetitions):
                rng = random.Random(length * 100_003 + offset * 101 + repeat)
                other = reduce_states(
                    list(operands.unbind(1)), motor_product, path,
                    rng if path == "random" else None,
                )
                left_q, left_t = decode_motor(left)
                other_q, other_t = decode_motor(other)
                trans = torch.linalg.vector_norm(left_t - other_t, dim=-1).max().item()
                relative = qmul(qconj(left_q), other_q)
                rot = (2.0 * torch.atan2(
                    torch.linalg.vector_norm(relative[..., 1:], dim=-1),
                    relative[..., :1].abs().squeeze(-1),
                )).max().item()
                maxima[path] = max(maxima[path], trans, rot)
    return {"max_translation_or_rotation_si": maxima, "tolerance": 1e-9,
            "passes": max(maxima.values()) < 1e-9}


def run(split: str, cfg: Config, device: torch.device) -> dict:
    data = TUMData(cfg)
    RUNS.mkdir(parents=True, exist_ok=True)
    output = {
        "protocol_frozen": "2026-09-03",
        "split": split,
        "config": asdict(cfg),
        "sequence_counts": {
            key: sum(split_for(name) == key for name in SEQUENCES)
            for key in ("train", "dev", "confirm")
        },
        "translation_scale_m": data.translation_scale,
        "baselines": {},
        "exact_audit": {},
        "arms": {},
    }
    for length in EVAL_LENGTHS:
        output["baselines"][str(length)] = {
            "motor": evaluate(data, split, length, "motor", cfg),
            "decoupled": evaluate(data, split, length, "decoupled", cfg),
        }
        output["exact_audit"][str(length)] = exact_path_audit(data, split, length, cfg)

    for arm in ARMS:
        output["arms"][arm] = {}
        for seed in SEEDS:
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
                model = LearnedComposer(arm).to(device)
                model.load_state_dict(saved["state_dict"])
                trained = {"training_seconds": None, "final_loss": None}
            record = {"training": trained, "lengths": {}}
            for length in EVAL_LENGTHS:
                paths = {
                    path: evaluate(data, split, length, path, cfg, model, seed)
                    for path in PATHS
                }
                paths["random"] = random_tree_evaluation(data, split, length, cfg, model, seed)
                left = paths["left"]
                intervals = {
                    path: bootstrap_delta(left, paths[path], seed * 10_000 + length * 10 + index,
                                          cfg.bootstrap_resamples)
                    for index, path in enumerate(("right", "balanced", "random"))
                }
                record["lengths"][str(length)] = {"paths": paths, "translation_deltas": intervals}
            output["arms"][arm][str(seed)] = record
            print(f"evaluated {split}: {arm} seed={seed}", flush=True)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirm", action="store_true", help="open the frozen confirmation split")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=Config.steps)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config(steps=args.steps)
    split = "confirm" if args.confirm else "dev"
    result = run(split, cfg, torch.device(args.device))
    destination = HERE / ("results_confirmation.json" if args.confirm else "results_development.json")
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
