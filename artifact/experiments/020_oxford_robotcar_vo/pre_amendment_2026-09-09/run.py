"""Prospective Oxford RobotCar VO-to-RTK motor-composition experiment.

Development is the default. ``--confirm`` only evaluates checkpoints already
frozen by the development run. Inputs are official Oxford stereo-VO estimates;
targets are independent official RTK poses. No observation is generated.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import importlib.util
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

import data_manifest as oxford_manifest

split_for = oxford_manifest.split_for

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DATA = ROOT / "data" / "raw" / "oxford_robotcar"
MANIFEST = DATA / "manifest.json"
RUNS = HERE / "runs"

BASE_DIR = HERE.parent / "017_tum_motor_composition"
sys.path.insert(0, str(BASE_DIR))
saved_manifest_module = sys.modules.pop("data_manifest", None)
spec = importlib.util.spec_from_file_location("motor_base", BASE_DIR / "run.py")
base = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = base
spec.loader.exec_module(base)
if saved_manifest_module is not None:
    sys.modules["data_manifest"] = saved_manifest_module

TRAIN_LENGTH = 8
EVAL_LENGTHS = (4, 8, 16, 32)
PATHS = ("left", "right", "balanced")
RANDOM_TREES = 16
SEEDS = tuple(range(5))
ARMS = ("ga_calibrated", "learned", "mlp", "mixed", "penalty")


@dataclass(frozen=True)
class Config:
    steps: int = 5000
    batch_size: int = 256
    learning_rate: float = 2e-3
    weight_decay: float = 1e-4
    penalty_weight: float = 0.1
    evaluation_batch_size: int = 1024
    sample_period_seconds: float = 0.5
    maximum_alignment_seconds: float = 0.1
    moving_threshold_m: float = 5.0
    bootstrap_resamples: int = 10_000


def quaternion_from_rpy(rpy: np.ndarray) -> np.ndarray:
    """Oxford SDK convention Rz(yaw) Ry(pitch) Rx(roll), returned wxyz."""
    roll, pitch, yaw = np.moveaxis(rpy, -1, 0)
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    return np.stack((
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ), axis=-1)


def inverse_motor(motor: torch.Tensor) -> torch.Tensor:
    q, t = base.decode_motor(motor)
    qi = base.qconj(q)
    return base.make_motor(qi, -base.qrotate(qi, t))


def relative_from_absolute(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return base.motor_product(inverse_motor(a), b)


def motors_from_xyzrpy(xyz: np.ndarray, rpy: np.ndarray) -> torch.Tensor:
    q = torch.from_numpy(quaternion_from_rpy(rpy).astype(np.float64, copy=False))
    t = torch.from_numpy(xyz.astype(np.float64, copy=False))
    return base.make_motor(q, t)


def read_numeric_csv(path: Path) -> tuple[list[str], np.ndarray]:
    with path.open(newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        rows = [[float(value) for value in row] for row in reader if row]
    values = np.asarray(rows, dtype=np.float64)
    if values.ndim != 2 or len(values) < 2:
        raise RuntimeError(f"insufficient numeric rows: {path}")
    if np.any(np.diff(values[:, 0]) <= 0):
        raise RuntimeError(f"timestamps must be strictly increasing: {path}")
    return header, values


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_extrinsic(path: Path) -> torch.Tensor:
    fields = [float(value) for value in path.read_text().split()]
    if len(fields) != 6:
        raise RuntimeError(f"expected six official INS extrinsic values: {path}")
    return motors_from_xyzrpy(
        np.asarray(fields[:3], dtype=np.float64)[None, :],
        np.asarray(fields[3:], dtype=np.float64)[None, :],
    )[0]


class OxfordTraversal:
    def __init__(self, name: str, vo_path: Path, rtk_path: Path,
                 vehicle_from_ins: torch.Tensor, cfg: Config):
        _, vo = read_numeric_csv(vo_path)
        _, rtk = read_numeric_csv(rtk_path)
        if vo.shape[1] < 8 or rtk.shape[1] < 14:
            raise RuntimeError(f"unexpected official schema for {name}")

        # Match the official SDK: accumulate row[2:8] and attach the result to
        # the timestamp in column 0. Column 1 remains provenance, not an anchor.
        vo_relative = motors_from_xyzrpy(vo[:, 2:5], vo[:, 5:8])
        identity = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                dtype=torch.float64)
        current = identity
        accumulated = []
        for relative in vo_relative:
            current = base.motor_product(current, relative)
            accumulated.append(current)
        vo_absolute = torch.stack(accumulated)
        vo_timestamps = vo[:, 0].astype(np.int64)

        # Official SDK RTK schema: UTM xyz columns 4:7, RPY columns 11:14.
        # Its projection code states that VO and vehicle frames coincide; the
        # fixed ins.txt transform therefore puts RTK targets in that VO frame.
        rtk_ins = motors_from_xyzrpy(rtk[:, 4:7], rtk[:, 11:14])
        ins_from_vehicle = inverse_motor(vehicle_from_ins)
        rtk_vehicle = base.motor_product(
            rtk_ins, ins_from_vehicle.expand_as(rtk_ins)
        )
        rtk_timestamps = rtk[:, 0].astype(np.int64)

        period_us = int(round(cfg.sample_period_seconds * 1_000_000))
        tolerance_us = int(round(cfg.maximum_alignment_seconds * 1_000_000))
        first = max(int(vo_timestamps[0]), int(rtk_timestamps[0]))
        last = min(int(vo_timestamps[-1]), int(rtk_timestamps[-1]))
        grid = np.arange(first, last + 1, period_us, dtype=np.int64)
        rtk_indices = np.searchsorted(rtk_timestamps, grid, side="left")

        anchors: list[tuple[int, torch.Tensor, torch.Tensor] | None] = []
        for grid_index, rtk_index in enumerate(rtk_indices):
            if rtk_index >= len(rtk_timestamps):
                anchors.append(None)
                continue
            vo_index = int(np.searchsorted(
                vo_timestamps, rtk_timestamps[rtk_index], side="left"
            ))
            if vo_index >= len(vo_timestamps):
                anchors.append(None)
                continue
            separation = int(vo_timestamps[vo_index] - rtk_timestamps[rtk_index])
            if separation < 0 or separation > tolerance_us:
                anchors.append(None)
                continue
            anchors.append((grid_index, vo_absolute[vo_index], rtk_vehicle[rtk_index]))

        self.name = name
        self.runs: list[tuple[torch.Tensor, torch.Tensor]] = []
        active: list[tuple[int, torch.Tensor, torch.Tensor]] = []
        for anchor in anchors + [None]:
            if anchor is not None and (not active or anchor[0] == active[-1][0] + 1):
                active.append(anchor)
                continue
            if len(active) > max(EVAL_LENGTHS):
                self.runs.append((
                    torch.stack([item[1] for item in active]),
                    torch.stack([item[2] for item in active]),
                ))
            active = [] if anchor is None else [anchor]
        if not self.runs:
            raise RuntimeError(f"no valid contiguous anchor run for {name}")


class OxfordData:
    def __init__(self, cfg: Config):
        if not MANIFEST.exists():
            raise FileNotFoundError(
                f"missing {MANIFEST}; acquire official files and run prepare_data.py"
            )
        manifest = json.loads(MANIFEST.read_text())
        if manifest.get("protocol") != "Oxford RobotCar VO-to-RTK v1, frozen 2026-09-03":
            raise RuntimeError("manifest protocol identifier does not match frozen experiment")
        extrinsic = load_extrinsic(DATA / manifest["ins_extrinsic"]["path"])
        extrinsic_path = DATA / manifest["ins_extrinsic"]["path"]
        if sha256(extrinsic_path) != manifest["ins_extrinsic"]["sha256"]:
            raise RuntimeError(f"file changed after manifest freeze: {extrinsic_path}")
        self.trajectories = {}
        exclusions = []
        for record in manifest["records"]:
            name = record["traversal"]
            if record["split"] != split_for(name):
                raise RuntimeError(f"manifest split mismatch: {name}")
            vo_path = DATA / record["vo"]["path"]
            rtk_path = DATA / record["rtk"]["path"]
            for path, expected in (
                (vo_path, record["vo"]["sha256"]),
                (rtk_path, record["rtk"]["sha256"]),
            ):
                if sha256(path) != expected:
                    raise RuntimeError(f"file changed after manifest freeze: {path}")
            try:
                self.trajectories[name] = OxfordTraversal(
                    name, vo_path, rtk_path, extrinsic, cfg,
                )
            except RuntimeError as error:
                exclusions.append({"traversal": name, "reason": str(error)})
        if exclusions:
            raise RuntimeError(
                "schema-valid manifest contained an unusable traversal; record a pre-training "
                f"protocol amendment before continuing: {exclusions}"
            )

        self.examples: dict[tuple[str, int], list[str]] = {}
        self.tensors: dict[tuple[str, int], tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
        for split in ("train", "dev", "confirm"):
            for length in EVAL_LENGTHS:
                operands, targets, moving, names = [], [], [], []
                for name, trajectory in self.trajectories.items():
                    if split_for(name) != split:
                        continue
                    for vo_absolute, rtk_absolute in trajectory.runs:
                        vo_leaves = relative_from_absolute(vo_absolute[:-1], vo_absolute[1:])
                        count = len(vo_leaves) - length + 1
                        if count <= 0:
                            continue
                        windows = vo_leaves.unfold(0, length, 1).permute(0, 2, 1).contiguous()
                        target = relative_from_absolute(
                            rtk_absolute[:count], rtk_absolute[length:length + count]
                        )
                        _, target_t = base.decode_motor(target)
                        operands.append(windows)
                        targets.append(target)
                        moving.append(torch.linalg.vector_norm(target_t, dim=-1) >= cfg.moving_threshold_m)
                        names.extend([name] * count)
                if not operands:
                    raise RuntimeError(f"no eligible windows for {split=} {length=}")
                self.examples[(split, length)] = names
                self.tensors[(split, length)] = (
                    torch.cat(operands), torch.cat(targets), torch.cat(moving)
                )
        self.train_examples = self.examples[("train", TRAIN_LENGTH)]
        _, train_targets, _ = self.tensors[("train", TRAIN_LENGTH)]
        _, train_translation = base.decode_motor(train_targets)
        self.translation_scale = max(
            float(torch.linalg.vector_norm(train_translation, dim=-1).median()), 1e-3
        )

    def batch(self, split: str, length: int, indices):
        index = torch.as_tensor(
            list(indices) if isinstance(indices, range) else indices, dtype=torch.long
        )
        operands, targets, moving = self.tensors[(split, length)]
        names = [self.examples[(split, length)][int(item)] for item in index]
        return operands[index], targets[index], moving[index], names


class LeafCalibrator(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(8, 16), nn.GELU(), nn.Linear(16, 8))
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, motor: torch.Tensor) -> torch.Tensor:
        return base.normalize_motor(motor + self.network(motor))


class OxfordComposer(nn.Module):
    def __init__(self, arm: str):
        super().__init__()
        self.arm = arm
        self.calibrator = LeafCalibrator()
        if arm == "mlp":
            self.composition = nn.Sequential(
                nn.Linear(16, 20), nn.GELU(), nn.Linear(20, 8)
            )
            nn.init.zeros_(self.composition[-1].weight)
            nn.init.zeros_(self.composition[-1].bias)
        elif arm != "ga_calibrated":
            self.table = nn.Parameter(torch.zeros(8, 8, 8))

    def op(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        foundation = base.motor_product(a, b)
        if self.arm == "ga_calibrated":
            return foundation
        if self.arm == "mlp":
            residual = self.composition(torch.cat((a, b), dim=-1))
        else:
            residual = torch.einsum("bi,ijk,bj->bk", a, self.table, b)
        return base.normalize_motor(foundation + residual)

    def forward(self, operands: torch.Tensor, path: str,
                rng: random.Random | None = None) -> torch.Tensor:
        leaves = self.calibrator(operands)
        return base.reduce_states(list(leaves.unbind(1)), self.op, path, rng)


def train_one(data: OxfordData, arm: str, seed: int, cfg: Config,
              device: torch.device) -> dict:
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed + 10_000)
    rng = random.Random(seed + 20_000)
    model = OxfordComposer(arm).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )
    losses = []
    started = time.perf_counter()
    model.train()
    for step in range(1, cfg.steps + 1):
        indices = torch.randint(len(data.train_examples), (cfg.batch_size,), generator=generator)
        operands, target, _, _ = data.batch("train", TRAIN_LENGTH, indices)
        operands = operands.to(device=device, dtype=torch.float32)
        target = target.to(device=device, dtype=torch.float32)
        path = "left"
        if arm == "mixed":
            path = ("left", "right", "balanced", "random")[rng.randrange(4)]
        prediction = model(operands, path, rng if path == "random" else None)
        loss = base.motor_loss(prediction, target, data.translation_scale)
        if arm == "penalty":
            leaves = model.calibrator(operands)
            start = rng.randrange(TRAIN_LENGTH - 2)
            a, b, c = (leaves[:, start + offset] for offset in range(3))
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


@torch.no_grad()
def evaluate(data: OxfordData, split: str, length: int, path: str, cfg: Config,
             model: OxfordComposer | None = None, seed: int = 0,
             moving_only: bool = False) -> dict:
    examples = data.examples[(split, length)]
    translation_parts, rotation_parts, dispersion_parts = [], [], []
    clusters: dict[str, list[tuple[float, float]]] = {}
    for offset in range(0, len(examples), cfg.evaluation_batch_size):
        indices = range(offset, min(offset + cfg.evaluation_batch_size, len(examples)))
        operands, target, moving, names = data.batch(split, length, indices)
        if model is None:
            prediction = base.reduce_states(list(operands.unbind(1)), base.motor_product, "left")
            left = prediction
        else:
            model.eval()
            operands32 = operands.to(next(model.parameters()).device, dtype=torch.float32)
            tree_rng = random.Random(seed * 1_000_003 + length * 1009 + offset)
            prediction = model(
                operands32, path, tree_rng if path == "random" else None
            ).cpu().double()
            left = model(operands32, "left").cpu().double()
        mask = moving if moving_only else torch.ones_like(moving, dtype=torch.bool)
        if not bool(mask.any()):
            continue
        prediction, target, left = prediction[mask], target[mask], left[mask]
        selected_names = [name for name, keep in zip(names, mask.tolist()) if keep]
        trans, rot = base.errors(prediction, target)
        translation_parts.append(trans)
        rotation_parts.append(rot)
        numerator = torch.linalg.vector_norm(prediction - left, dim=-1)
        denominator = torch.linalg.vector_norm(left, dim=-1).clamp_min(1e-12)
        dispersion_parts.append((numerator / denominator).numpy())
        for name, t_error, r_error in zip(selected_names, trans, rot):
            clusters.setdefault(name, []).append((float(t_error), float(r_error)))
    if not translation_parts:
        return {"n_windows": 0, "n_trajectories": 0, "clusters": {}}
    translation = np.concatenate(translation_parts)
    rotation = np.concatenate(rotation_parts)
    dispersion = np.concatenate(dispersion_parts)
    cluster_summary = {
        name: {
            "translation_mean_m": float(np.mean([value[0] for value in values])),
            "rotation_mean_deg": float(np.mean([value[1] for value in values])),
            "n": len(values),
        }
        for name, values in clusters.items()
    }
    return {
        "n_windows": int(len(translation)),
        "n_trajectories": len(cluster_summary),
        "translation_mean_m": float(translation.mean()),
        "translation_median_m": float(np.median(translation)),
        "rotation_mean_deg": float(rotation.mean()),
        "rotation_median_deg": float(np.median(rotation)),
        "dispersion_mean": float(dispersion.mean()),
        "dispersion_median": float(np.median(dispersion)),
        "clusters": cluster_summary,
    }


def random_tree_evaluation(data, split, length, cfg, model, seed, moving_only=False):
    runs = [
        evaluate(data, split, length, "random", cfg, model, seed * 100 + index, moving_only)
        for index in range(RANDOM_TREES)
    ]
    keys = (
        "translation_mean_m", "translation_median_m", "rotation_mean_deg",
        "rotation_median_deg", "dispersion_mean", "dispersion_median",
    )
    result = {key: float(np.mean([run[key] for run in runs])) for key in keys}
    result["n_windows"] = runs[0]["n_windows"]
    result["n_trajectories"] = runs[0]["n_trajectories"]
    result["clusters"] = {
        name: {
            "translation_mean_m": float(np.mean([
                run["clusters"][name]["translation_mean_m"] for run in runs
            ])),
            "rotation_mean_deg": float(np.mean([
                run["clusters"][name]["rotation_mean_deg"] for run in runs
            ])),
            "n": runs[0]["clusters"][name]["n"],
        }
        for name in runs[0]["clusters"]
    }
    return result


def bootstrap_delta(left: dict, other: dict, seed: int, resamples: int) -> dict:
    names = sorted(set(left["clusters"]) & set(other["clusters"]))
    if not names:
        raise RuntimeError("no complete traversal clusters for bootstrap")
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


@torch.no_grad()
def exact_path_audit(data: OxfordData, split: str, length: int, cfg: Config,
                     model: OxfordComposer | None = None) -> dict:
    examples = data.examples[(split, length)]
    maxima = {"right": {"translation_m": 0.0, "rotation_rad": 0.0},
              "balanced": {"translation_m": 0.0, "rotation_rad": 0.0},
              "random": {"translation_m": 0.0, "rotation_rad": 0.0}}
    exact_model = copy.deepcopy(model).cpu().double() if model is not None else None
    for offset in range(0, len(examples), cfg.evaluation_batch_size):
        indices = range(offset, min(offset + cfg.evaluation_batch_size, len(examples)))
        operands, _, _, _ = data.batch(split, length, indices)
        if exact_model is None:
            leaves = operands
        else:
            leaves = exact_model.calibrator(operands)
        left = base.reduce_states(list(leaves.unbind(1)), base.motor_product, "left")
        for path in maxima:
            repetitions = RANDOM_TREES if path == "random" else 1
            for repeat in range(repetitions):
                rng = random.Random(length * 100_003 + offset * 101 + repeat)
                other = base.reduce_states(
                    list(leaves.unbind(1)), base.motor_product, path,
                    rng if path == "random" else None,
                )
                left_q, left_t = base.decode_motor(left)
                other_q, other_t = base.decode_motor(other)
                trans = torch.linalg.vector_norm(left_t - other_t, dim=-1).max().item()
                relative = base.qmul(base.qconj(left_q), other_q)
                rot = (2.0 * torch.atan2(
                    torch.linalg.vector_norm(relative[..., 1:], dim=-1),
                    relative[..., :1].abs().squeeze(-1),
                )).max().item()
                maxima[path]["translation_m"] = max(maxima[path]["translation_m"], trans)
                maxima[path]["rotation_rad"] = max(maxima[path]["rotation_rad"], rot)
    worst = max(value for path in maxima.values() for value in path.values())
    return {"maxima": maxima, "tolerance": 1e-9, "worst_si": worst, "passes": worst < 1e-9}


def run(split: str, cfg: Config, device: torch.device) -> dict:
    data = OxfordData(cfg)
    RUNS.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST.read_text())
    output = {
        "protocol_frozen": "2026-09-03",
        "split": split,
        "config": asdict(cfg),
        "eligible_count": manifest["eligible_count"],
        "split_counts": manifest["split_counts"],
        "translation_scale_m": data.translation_scale,
        "baselines": {},
        "arms": {},
    }
    for length in EVAL_LENGTHS:
        raw = evaluate(data, split, length, "left", cfg)
        moving = evaluate(data, split, length, "left", cfg, moving_only=True)
        output["baselines"][str(length)] = {
            "raw_motor": raw,
            "raw_motor_moving": moving,
            "exact_audit": exact_path_audit(data, split, length, cfg),
        }

    for arm in ARMS:
        output["arms"][arm] = {}
        for seed in SEEDS:
            checkpoint = RUNS / f"{arm}_seed{seed}.pt"
            if split == "dev":
                trained = train_one(data, arm, seed, cfg, device)
                model = trained.pop("model")
                torch.save({
                    "state_dict": model.state_dict(), "arm": arm, "seed": seed,
                    "config": asdict(cfg),
                }, checkpoint)
            else:
                if not checkpoint.exists():
                    raise FileNotFoundError(f"frozen checkpoint missing: {checkpoint}")
                saved = torch.load(checkpoint, map_location=device, weights_only=True)
                if saved["arm"] != arm or saved["seed"] != seed or saved["config"] != asdict(cfg):
                    raise RuntimeError(f"checkpoint metadata mismatch: {checkpoint}")
                model = OxfordComposer(arm).to(device)
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
                raw = output["baselines"][str(length)]["raw_motor"]
                deltas = {
                    path: bootstrap_delta(
                        left, paths[path], seed * 10_000 + length * 10 + index,
                        cfg.bootstrap_resamples,
                    )
                    for index, path in enumerate(("right", "balanced", "random"))
                }
                moving_paths = {
                    path: evaluate(data, split, length, path, cfg, model, seed, True)
                    for path in PATHS
                }
                moving_paths["random"] = random_tree_evaluation(
                    data, split, length, cfg, model, seed, True
                )
                moving_left = moving_paths["left"]
                record["lengths"][str(length)] = {
                    "paths": paths,
                    "translation_deltas": deltas,
                    "left_vs_raw": bootstrap_delta(
                        raw, left, 900_000 + seed * 100 + length, cfg.bootstrap_resamples
                    ),
                    "moving_paths": moving_paths,
                    "moving_translation_deltas": {
                        path: bootstrap_delta(
                            moving_left, moving_paths[path],
                            2_000_000 + seed * 10_000 + length * 10 + index,
                            cfg.bootstrap_resamples,
                        )
                        for index, path in enumerate(("right", "balanced", "random"))
                    },
                    "moving_left_vs_raw": bootstrap_delta(
                        output["baselines"][str(length)]["raw_motor_moving"],
                        moving_left, 3_000_000 + seed * 100 + length,
                        cfg.bootstrap_resamples,
                    ),
                }
                if arm == "ga_calibrated":
                    record["lengths"][str(length)]["exact_audit"] = exact_path_audit(
                        data, split, length, cfg, model
                    )
            output["arms"][arm][str(seed)] = record
            print(f"evaluated {split}: {arm} seed={seed}", flush=True)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config()
    if args.confirm:
        development_path = HERE / "results_development.json"
        if not development_path.exists():
            raise RuntimeError(
                "confirmation is sealed until results_development.json exists and passes"
            )
        from audit_results import audit

        development_audit = audit(json.loads(development_path.read_text()))
        if development_audit["split"] != "dev" or not development_audit["passes"]:
            raise RuntimeError("confirmation is sealed: frozen development gate did not pass")
    split = "confirm" if args.confirm else "dev"
    result = run(split, cfg, torch.device(args.device))
    destination = HERE / (
        "results_confirmation.json" if args.confirm else "results_development.json"
    )
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
