"""Frozen family-held-out evaluation on public real ETH3D recordings."""

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

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BASE_DIR = HERE.parent / "022_eth3d_adaptive_motor"
DATA = ROOT / "data" / "raw" / "eth3d_rgbd"
MANIFEST = DATA / "manifest.json"
RUNS = HERE / "runs"

sys.path.insert(0, str(BASE_DIR))
spec = importlib.util.spec_from_file_location("adaptive_v2", BASE_DIR / "run.py")
v2 = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = v2
spec.loader.exec_module(v2)
v1 = v2.v1
base = v2.base

LENGTHS = (4, 8, 16, 32)
SEEDS = tuple(range(5))
FOLDS = tuple(range(5))

FAMILY_SEQUENCES = {
    "cables": ("cables_1", "cables_2", "cables_3"),
    "camera_shake": ("camera_shake_1", "camera_shake_2", "camera_shake_3"),
    "ceiling": ("ceiling_1", "ceiling_2"),
    "desk": ("desk_3", "desk_changing_1"),
    "einstein": (
        "einstein_1", "einstein_2", "einstein_dark", "einstein_flashlight",
        "einstein_global_light_changes_1", "einstein_global_light_changes_2",
        "einstein_global_light_changes_3",
    ),
    "kidnap": ("kidnap_1", "kidnap_dark"),
    "large_loop": ("large_loop_1",),
    "mannequin": (
        "mannequin_1", "mannequin_3", "mannequin_4", "mannequin_5",
        "mannequin_7", "mannequin_face_1", "mannequin_face_2",
        "mannequin_face_3", "mannequin_head",
    ),
    "motion": ("motion_1",),
    "planar": ("planar_2", "planar_3"),
    "plant": (
        "plant_1", "plant_2", "plant_3", "plant_4", "plant_5", "plant_dark",
        "plant_scene_1", "plant_scene_2", "plant_scene_3",
    ),
    "reflective": ("reflective_1",),
    "repetitive": ("repetitive",),
    "sofa": (
        "sofa_1", "sofa_2", "sofa_3", "sofa_4", "sofa_dark_1",
        "sofa_dark_2", "sofa_dark_3", "sofa_shake",
    ),
    "table": ("table_3", "table_4", "table_7"),
    "vicon_light": ("vicon_light_1", "vicon_light_2"),
}

FOLD_FAMILIES = {
    0: ("mannequin", "kidnap", "repetitive"),
    1: ("plant", "planar"),
    2: ("sofa", "desk", "motion"),
    3: ("einstein", "ceiling", "vicon_light"),
    4: ("cables", "camera_shake", "table", "large_loop", "reflective"),
}
FAMILY_FOR = {
    sequence: family for family, sequences in FAMILY_SEQUENCES.items()
    for sequence in sequences
}


@dataclass(frozen=True)
class Config:
    calibrator_steps: int = 5000
    contextual_steps: int = 5000
    batch_size: int = 256
    calibrator_learning_rate: float = 2e-3
    contextual_learning_rate: float = 1e-3
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


class ETH3DAll:
    def __init__(self):
        manifest = json.loads(MANIFEST.read_text())
        if manifest.get("protocol") != "ETH3D RGB-D motor v1, frozen 2026-09-04":
            raise RuntimeError("unexpected ETH3D manifest protocol")
        if manifest.get("open3d_version") != "0.19.0":
            raise RuntimeError("unexpected Open3D version")
        if set(FAMILY_FOR) != set(v1.SEQUENCES):
            raise RuntimeError("family map is not an exact partition of eligible sequences")
        expected_folds = {
            0: ("mannequin", "kidnap", "repetitive"),
            1: ("plant", "planar"),
            2: ("sofa", "desk", "motion"),
            3: ("einstein", "ceiling", "vicon_light"),
            4: ("cables", "camera_shake", "table", "large_loop", "reflective"),
        }
        if FOLD_FAMILIES != expected_folds:
            raise RuntimeError("fold map changed after protocol freeze")
        self.front_end = manifest["front_end"]
        self.runs = {}
        for record in manifest["records"]:
            name = record["sequence"]
            path = DATA / record["derived_path"]
            if sha256(path) != record["derived_sha256"]:
                raise RuntimeError(f"derived odometry hash mismatch: {name}")
            arrays = np.load(path, allow_pickle=False)
            valid = arrays["pair_valid"].astype(bool)
            leaves = v1.motors_from_matrices(arrays["pair_motions"])
            reference = v1.motors_from_matrices(arrays["reference_poses"])
            runs, start = [], None
            for index, keep in enumerate(np.append(valid, False)):
                if keep and start is None:
                    start = index
                elif not keep and start is not None:
                    if index - start >= max(LENGTHS):
                        runs.append((leaves[start:index], reference[start:index + 1]))
                    start = None
            self.runs[name] = runs

        self.examples = {}
        self.tensors = {}
        for length in LENGTHS:
            operands, targets, names = [], [], []
            for name in v1.SEQUENCES:
                for leaves, reference in self.runs[name]:
                    count = len(leaves) - length + 1
                    if count <= 0:
                        continue
                    operands.append(
                        leaves.unfold(0, length, 1).permute(0, 2, 1).contiguous()
                    )
                    targets.append(v1.study.relative_from_absolute(
                        reference[:count], reference[length:length + count]
                    ))
                    names.extend([name] * count)
            self.examples[length] = names
            self.tensors[length] = (torch.cat(operands), torch.cat(targets))


class FoldData:
    def __init__(self, all_data: ETH3DAll, fold: int):
        test_families = set(FOLD_FAMILIES[fold])
        self.examples = {}
        self.tensors = {}
        for split in ("train", "test"):
            for length in LENGTHS:
                names = all_data.examples[length]
                selected = [
                    index for index, name in enumerate(names)
                    if ((FAMILY_FOR[name] in test_families) == (split == "test"))
                ]
                index = torch.tensor(selected, dtype=torch.long)
                operands, targets = all_data.tensors[length]
                self.examples[(split, length)] = [names[i] for i in selected]
                self.tensors[(split, length)] = (
                    operands[index], targets[index],
                    torch.ones(len(index), dtype=torch.bool),
                )
        self.train_examples = self.examples[("train", 8)]
        _, targets, _ = self.tensors[("train", 8)]
        _, translation = base.decode_motor(targets)
        self.translation_scale = max(
            float(torch.linalg.vector_norm(translation, dim=-1).median()), 1e-3
        )
        represented = {FAMILY_FOR[name] for name in self.examples[("test", 32)]}
        if represented != test_families:
            raise RuntimeError(
                f"fold {fold} has family without a valid L32 window: "
                f"{sorted(test_families - represented)}"
            )

    def batch(self, split: str, length: int, indices):
        index = torch.as_tensor(
            list(indices) if isinstance(indices, range) else indices,
            dtype=torch.long,
        )
        operands, targets, moving = self.tensors[(split, length)]
        names = [self.examples[(split, length)][int(item)] for item in index]
        return operands[index], targets[index], moving[index], names


class CalibratedMotor(nn.Module):
    def __init__(self):
        super().__init__()
        self.calibrator = v1.study.LeafCalibrator()

    def forward(self, operands: torch.Tensor, path: str = "balanced") -> torch.Tensor:
        leaves = self.calibrator(operands)
        return base.reduce_states(list(leaves.unbind(1)), base.motor_product, path)


class ContextualGA(nn.Module):
    def __init__(self, calibrator: nn.Module, hidden_size: int):
        super().__init__()
        self.calibrator = copy.deepcopy(calibrator)
        for parameter in self.calibrator.parameters():
            parameter.requires_grad_(False)
        self.encoder = nn.GRU(8, hidden_size, batch_first=True, bidirectional=True)
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


class DirectGRU(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.encoder = nn.GRU(8, hidden_size, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(
            nn.Linear(2 * hidden_size, 38), nn.GELU(), nn.Linear(38, 8)
        )

    def forward(self, operands: torch.Tensor, path: str = "balanced") -> torch.Tensor:
        del path
        encoded, _ = self.encoder(operands)
        return base.normalize_motor(self.head(encoded.mean(1)))


def train_calibrator(data: FoldData, cfg: Config, seed: int,
                     device: torch.device) -> tuple[CalibratedMotor, dict]:
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed + 10_000)
    model = CalibratedMotor().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.calibrator_learning_rate,
        weight_decay=cfg.weight_decay,
    )
    losses, started = [], time.perf_counter()
    model.train()
    for step in range(1, cfg.calibrator_steps + 1):
        indices = torch.randint(
            len(data.train_examples), (cfg.batch_size,), generator=generator
        )
        operands, target, _, _ = data.batch("train", 8, indices)
        prediction = model(operands.to(device=device, dtype=torch.float32), "left")
        loss = base.motor_loss(
            prediction, target.to(device=device, dtype=torch.float32),
            data.translation_scale,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    return model, {
        "seconds": time.perf_counter() - started,
        "final_loss": float(np.mean(losses[-100:])),
    }


def train_primary(data: FoldData, cfg: Config, seed: int, device: torch.device,
                  model: nn.Module) -> tuple[nn.Module, dict]:
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed + 30_000)
    sampler = v2.BalancedSampler(data)
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=cfg.contextual_learning_rate, weight_decay=cfg.weight_decay,
    )
    losses, started = [], time.perf_counter()
    model.train()
    for step in range(1, cfg.contextual_steps + 1):
        indices = sampler.sample(32, cfg.batch_size, generator)
        operands, target, _, _ = data.batch("train", 32, indices)
        operands = operands.to(device=device, dtype=torch.float32)
        target = target.to(device=device, dtype=torch.float32)
        loss = v2.robust_motor_loss(model(operands), target, data.translation_scale)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    return model, {
        "seconds": time.perf_counter() - started,
        "final_loss": float(np.mean(losses[-100:])),
    }


@torch.no_grad()
def evaluate(data: FoldData, length: int, cfg: Config,
             model: nn.Module | None) -> dict:
    examples = data.examples[("test", length)]
    clusters, all_t, all_r = {}, [], []
    for offset in range(0, len(examples), cfg.evaluation_batch_size):
        indices = range(offset, min(offset + cfg.evaluation_batch_size, len(examples)))
        operands, target, _, names = data.batch("test", length, indices)
        if model is None:
            prediction = base.reduce_states(
                list(operands.unbind(1)), base.motor_product, "balanced"
            )
        else:
            model.eval()
            prediction = model(
                operands.to(next(model.parameters()).device, dtype=torch.float32)
            ).cpu().double()
        translation, rotation = base.errors(prediction, target)
        all_t.append(np.asarray(translation))
        all_r.append(np.asarray(rotation))
        for name, t_error, r_error in zip(names, translation, rotation):
            clusters.setdefault(name, []).append((float(t_error), float(r_error)))
    translation, rotation = np.concatenate(all_t), np.concatenate(all_r)
    return {
        "n_windows": int(len(translation)),
        "n_sequences": len(clusters),
        "translation_mean_m": float(translation.mean()),
        "translation_median_m": float(np.median(translation)),
        "rotation_mean_deg": float(rotation.mean()),
        "rotation_median_deg": float(np.median(rotation)),
        "clusters": {
            name: {
                "family": FAMILY_FOR[name],
                "translation_mean_m": float(np.mean([x[0] for x in values])),
                "rotation_mean_deg": float(np.mean([x[1] for x in values])),
                "n": len(values),
            }
            for name, values in clusters.items()
        },
    }


@torch.no_grad()
def exact_audit(data: FoldData, length: int, cfg: Config,
                model: nn.Module | None) -> dict:
    exact_model = copy.deepcopy(model).cpu().double() if model is not None else None
    maxima = {"right": 0.0, "balanced": 0.0}
    examples = data.examples[("test", length)]
    for offset in range(0, len(examples), cfg.evaluation_batch_size):
        indices = range(offset, min(offset + cfg.evaluation_batch_size, len(examples)))
        operands, _, _, _ = data.batch("test", length, indices)
        def predict(path):
            if exact_model is not None:
                return exact_model(operands, path)
            return base.reduce_states(list(operands.unbind(1)), base.motor_product, path)
        left = predict("left")
        q0, t0 = base.decode_motor(left)
        for path in maxima:
            q1, t1 = base.decode_motor(predict(path))
            translation = torch.linalg.vector_norm(t0 - t1, dim=-1).max().item()
            relative = base.qmul(base.qconj(q0), q1)
            rotation = (2 * torch.atan2(
                torch.linalg.vector_norm(relative[..., 1:], dim=-1),
                relative[..., :1].abs().squeeze(-1),
            )).max().item()
            maxima[path] = max(maxima[path], translation, rotation)
    worst = max(maxima.values())
    return {"maxima_si": maxima, "worst_si": worst, "passes": worst < 1e-9}


def trainable_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters()
               if parameter.requires_grad)


def run_fold(all_data: ETH3DAll, fold: int, seeds: tuple[int, ...], cfg: Config,
             device: torch.device) -> dict:
    data = FoldData(all_data, fold)
    result = {
        "families": list(FOLD_FAMILIES[fold]),
        "translation_scale_m": data.translation_scale,
        "counts": {}, "baselines": {}, "seeds": {},
    }
    for length in LENGTHS:
        raw = evaluate(data, length, cfg, None)
        result["counts"][str(length)] = {
            "train_windows": len(data.examples[("train", length)]),
            "test_windows": len(data.examples[("test", length)]),
            "test_sequences": raw["n_sequences"],
        }
        result["baselines"][str(length)] = {
            "raw_motor": raw,
            "exact_audit": exact_audit(data, length, cfg, None),
        }
    for seed in seeds:
        calibrator, cal_training = train_calibrator(data, cfg, seed, device)
        contextual, contextual_training = train_primary(
            data, cfg, seed, device,
            ContextualGA(calibrator.calibrator, cfg.hidden_size),
        )
        direct, direct_training = train_primary(
            data, cfg, seed, device, DirectGRU(cfg.hidden_size),
        )
        models = {
            "ga_calibrated": calibrator,
            "contextual_ga": contextual,
            "direct_gru": direct,
        }
        record = {
            "training": {
                "ga_calibrated": cal_training,
                "contextual_ga": contextual_training,
                "direct_gru": direct_training,
            },
            "parameter_counts": {
                "leaf_calibrator": trainable_count(calibrator),
                "contextual_ga": trainable_count(contextual),
                "direct_gru": trainable_count(direct),
            },
            "lengths": {},
        }
        for length in LENGTHS:
            record["lengths"][str(length)] = {}
            for name, model in models.items():
                record["lengths"][str(length)][name] = evaluate(
                    data, length, cfg, model
                )
                if name != "direct_gru":
                    record["lengths"][str(length)][name + "_exact_audit"] = exact_audit(
                        data, length, cfg, model
                    )
        result["seeds"][str(seed)] = record
        print(f"completed fold={fold} seed={seed}", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--folds", nargs="+", type=int, default=list(FOLDS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--steps", type=int)
    args = parser.parse_args()
    folds, seeds = tuple(args.folds), tuple(args.seeds)
    if any(fold not in FOLDS for fold in folds) or any(seed not in SEEDS for seed in seeds):
        raise SystemExit("folds and seeds must be drawn from 0..4")
    cfg = Config(
        calibrator_steps=args.steps or Config.calibrator_steps,
        contextual_steps=args.steps or Config.contextual_steps,
    )
    complete = folds == FOLDS and seeds == SEEDS and args.steps is None
    all_data = ETH3DAll()
    result = {
        "study": "ETH3D family-held-out motor v1, frozen 2026-09-05",
        "complete": complete, "config": asdict(cfg),
        "folds_requested": list(folds), "seeds_requested": list(seeds),
        "family_sequences": {k: list(v) for k, v in FAMILY_SEQUENCES.items()},
        "fold_families": {str(k): list(v) for k, v in FOLD_FAMILIES.items()},
        "front_end": all_data.front_end, "folds": {},
    }
    for fold in folds:
        result["folds"][str(fold)] = run_fold(
            all_data, fold, seeds, cfg, torch.device(args.device)
        )
    destination = HERE / (
        "results.json" if complete else
        f"results_smoke_f{'-'.join(map(str, folds))}_s{'-'.join(map(str, seeds))}_"
        f"{cfg.contextual_steps}.json"
    )
    if complete and destination.exists():
        raise RuntimeError("complete result already exists; refusing overwrite")
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
