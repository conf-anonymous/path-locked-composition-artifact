"""Prospectively frozen ETH3D RGB-D odometry motor-composition study."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

import data_manifest as eth3d_manifest

split_for = eth3d_manifest.split_for
SEQUENCES = eth3d_manifest.SEQUENCES

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DATA = ROOT / "data" / "raw" / "eth3d_rgbd"
MANIFEST = DATA / "manifest.json"
RUNS = HERE / "runs"

# Reuse the already-audited motor algebra, matched models, training loop, tree
# schedules, clustered bootstrap, and exactness audit from Experiment 020.
BASE_DIR = HERE.parent / "020_oxford_robotcar_vo"
sys.path.insert(0, str(BASE_DIR))
saved_manifest_module = sys.modules.pop("data_manifest", None)
spec = importlib.util.spec_from_file_location("eth3d_study_base", BASE_DIR / "run.py")
study = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = study
spec.loader.exec_module(study)
if saved_manifest_module is not None:
    sys.modules["data_manifest"] = saved_manifest_module

TRAIN_LENGTH = study.TRAIN_LENGTH
EVAL_LENGTHS = study.EVAL_LENGTHS
PATHS = study.PATHS
SEEDS = study.SEEDS
ARMS = study.ARMS


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def quaternion_from_matrix(matrix: np.ndarray) -> np.ndarray:
    """Stable rotation-matrix to wxyz quaternion conversion."""
    result = np.empty((len(matrix), 4), dtype=np.float64)
    for index, rotation in enumerate(matrix[:, :3, :3]):
        trace = np.trace(rotation)
        if trace > 0:
            scale = math.sqrt(trace + 1.0) * 2
            q = (0.25 * scale,
                 (rotation[2, 1] - rotation[1, 2]) / scale,
                 (rotation[0, 2] - rotation[2, 0]) / scale,
                 (rotation[1, 0] - rotation[0, 1]) / scale)
        else:
            diagonal = np.diag(rotation)
            axis = int(np.argmax(diagonal))
            if axis == 0:
                scale = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2
                q = ((rotation[2, 1] - rotation[1, 2]) / scale, 0.25 * scale,
                     (rotation[0, 1] + rotation[1, 0]) / scale,
                     (rotation[0, 2] + rotation[2, 0]) / scale)
            elif axis == 1:
                scale = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2
                q = ((rotation[0, 2] - rotation[2, 0]) / scale,
                     (rotation[0, 1] + rotation[1, 0]) / scale, 0.25 * scale,
                     (rotation[1, 2] + rotation[2, 1]) / scale)
            else:
                scale = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2
                q = ((rotation[1, 0] - rotation[0, 1]) / scale,
                     (rotation[0, 2] + rotation[2, 0]) / scale,
                     (rotation[1, 2] + rotation[2, 1]) / scale, 0.25 * scale)
        result[index] = q
    result /= np.linalg.norm(result, axis=1, keepdims=True)
    return result


def motors_from_matrices(matrices: np.ndarray) -> torch.Tensor:
    quaternion = torch.from_numpy(quaternion_from_matrix(matrices))
    translation = torch.from_numpy(matrices[:, :3, 3].copy())
    return study.base.make_motor(quaternion, translation)


class ETH3DData:
    def __init__(self, cfg, evaluation_split: str):
        if evaluation_split not in ("dev", "confirm"):
            raise ValueError(evaluation_split)
        if not MANIFEST.exists():
            raise FileNotFoundError(
                f"missing {MANIFEST}; run download_data.py and extract_odometry.py"
            )
        manifest = json.loads(MANIFEST.read_text())
        if manifest.get("protocol") != "ETH3D RGB-D motor v1, frozen 2026-09-04":
            raise RuntimeError("manifest protocol does not match frozen experiment")
        if manifest.get("open3d_version") != "0.19.0":
            raise RuntimeError("derived odometry was not produced by frozen Open3D 0.19.0")
        if [record["sequence"] for record in manifest["records"]] != list(SEQUENCES):
            raise RuntimeError("manifest sequence order differs from frozen eligibility list")

        required_splits = {"train", evaluation_split}
        self.runs = {}
        for record in manifest["records"]:
            name = record["sequence"]
            if record["split"] != split_for(name):
                raise RuntimeError(f"split mismatch: {name}")
            if record["split"] not in required_splits:
                continue
            path = DATA / record["derived_path"]
            if sha256(path) != record["derived_sha256"]:
                raise RuntimeError(f"derived odometry changed after freeze: {path}")
            arrays = np.load(path, allow_pickle=False)
            pair_valid = arrays["pair_valid"].astype(bool)
            pair_motors = motors_from_matrices(arrays["pair_motions"])
            reference_motors = motors_from_matrices(arrays["reference_poses"])
            runs = []
            start = None
            for index, valid in enumerate(np.append(pair_valid, False)):
                if valid and start is None:
                    start = index
                elif not valid and start is not None:
                    if index - start >= max(EVAL_LENGTHS):
                        runs.append((pair_motors[start:index], reference_motors[start:index + 1]))
                    start = None
            self.runs[name] = runs

        self.examples = {}
        self.tensors = {}
        for split in ("train", evaluation_split):
            for length in EVAL_LENGTHS:
                operands, targets, moving, names = [], [], [], []
                for name in SEQUENCES:
                    if split_for(name) != split:
                        continue
                    for leaves, reference in self.runs[name]:
                        count = len(leaves) - length + 1
                        if count <= 0:
                            continue
                        windows = leaves.unfold(0, length, 1).permute(0, 2, 1).contiguous()
                        target = study.relative_from_absolute(
                            reference[:count], reference[length:length + count]
                        )
                        _, target_translation = study.base.decode_motor(target)
                        operands.append(windows)
                        targets.append(target)
                        moving.append(
                            torch.linalg.vector_norm(target_translation, dim=-1)
                            >= cfg.moving_threshold_m
                        )
                        names.extend([name] * count)
                if not operands:
                    raise RuntimeError(f"no valid windows for {split=} {length=}")
                combined_moving = torch.cat(moving)
                if not bool(combined_moving.any()):
                    raise RuntimeError(f"no moving-subset windows for {split=} {length=}")
                self.examples[(split, length)] = names
                self.tensors[(split, length)] = (
                    torch.cat(operands), torch.cat(targets), combined_moving
                )

        self.train_examples = self.examples[("train", TRAIN_LENGTH)]
        _, train_targets, _ = self.tensors[("train", TRAIN_LENGTH)]
        _, train_translation = study.base.decode_motor(train_targets)
        self.translation_scale = max(
            float(torch.linalg.vector_norm(train_translation, dim=-1).median()), 1e-3
        )
        self.front_end_summary = {
            "valid_pairs": sum(record["valid_pairs"] for record in manifest["records"]),
            "front_end_failures": sum(
                record["front_end_failures"] for record in manifest["records"]
            ),
            "oversized_frame_gaps": sum(
                record["oversized_frame_gaps"] for record in manifest["records"]
            ),
            "front_end_seconds": sum(
                record["front_end_seconds"] for record in manifest["records"]
            ),
        }

    def batch(self, split: str, length: int, indices):
        index = torch.as_tensor(
            list(indices) if isinstance(indices, range) else indices, dtype=torch.long
        )
        operands, targets, moving = self.tensors[(split, length)]
        names = [self.examples[(split, length)][int(item)] for item in index]
        return operands[index], targets[index], moving[index], names


def run(split: str, cfg, device: torch.device) -> dict:
    data = ETH3DData(cfg, split)
    RUNS.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST.read_text())
    output = {
        "protocol_frozen": "2026-09-04",
        "split": split,
        "config": asdict(cfg),
        "eligible_count": manifest["eligible_count"],
        "split_counts": manifest["split_counts"],
        "front_end": manifest["front_end"],
        "front_end_summary": data.front_end_summary,
        "parameter_counts": {
            arm: sum(parameter.numel() for parameter in study.OxfordComposer(arm).parameters())
            for arm in ARMS
        },
        "translation_scale_m": data.translation_scale,
        "baselines": {},
        "arms": {},
    }
    for length in EVAL_LENGTHS:
        raw = study.evaluate(data, split, length, "left", cfg)
        moving = study.evaluate(data, split, length, "left", cfg, moving_only=True)
        output["baselines"][str(length)] = {
            "raw_motor": raw,
            "raw_motor_moving": moving,
            "exact_audit": study.exact_path_audit(data, split, length, cfg),
        }

    for arm in ARMS:
        output["arms"][arm] = {}
        for seed in SEEDS:
            checkpoint = RUNS / f"{arm}_seed{seed}.pt"
            if split == "dev":
                trained = study.train_one(data, arm, seed, cfg, device)
                model = trained.pop("model")
                torch.save({
                    "state_dict": model.state_dict(), "arm": arm, "seed": seed,
                    "config": asdict(cfg), "protocol": output["protocol_frozen"],
                }, checkpoint)
            else:
                if not checkpoint.exists():
                    raise FileNotFoundError(f"frozen checkpoint missing: {checkpoint}")
                saved = torch.load(checkpoint, map_location=device, weights_only=True)
                if (
                    saved["arm"] != arm or saved["seed"] != seed
                    or saved["config"] != asdict(cfg)
                    or saved.get("protocol") != output["protocol_frozen"]
                ):
                    raise RuntimeError(f"checkpoint metadata mismatch: {checkpoint}")
                model = study.OxfordComposer(arm).to(device)
                model.load_state_dict(saved["state_dict"])
                trained = {"training_seconds": None, "final_loss": None}
            record = {"training": trained, "lengths": {}}
            for length in EVAL_LENGTHS:
                paths = {
                    path: study.evaluate(data, split, length, path, cfg, model, seed)
                    for path in PATHS
                }
                paths["random"] = study.random_tree_evaluation(
                    data, split, length, cfg, model, seed
                )
                left = paths["left"]
                raw = output["baselines"][str(length)]["raw_motor"]
                deltas = {
                    path: study.bootstrap_delta(
                        left, paths[path], seed * 10_000 + length * 10 + index,
                        cfg.bootstrap_resamples,
                    )
                    for index, path in enumerate(("right", "balanced", "random"))
                }
                moving_paths = {
                    path: study.evaluate(data, split, length, path, cfg, model, seed, True)
                    for path in PATHS
                }
                moving_paths["random"] = study.random_tree_evaluation(
                    data, split, length, cfg, model, seed, True
                )
                moving_left = moving_paths["left"]
                record["lengths"][str(length)] = {
                    "paths": paths,
                    "translation_deltas": deltas,
                    "left_vs_raw": study.bootstrap_delta(
                        raw, left, 900_000 + seed * 100 + length,
                        cfg.bootstrap_resamples,
                    ),
                    "moving_paths": moving_paths,
                    "moving_translation_deltas": {
                        path: study.bootstrap_delta(
                            moving_left, moving_paths[path],
                            2_000_000 + seed * 10_000 + length * 10 + index,
                            cfg.bootstrap_resamples,
                        )
                        for index, path in enumerate(("right", "balanced", "random"))
                    },
                    "moving_left_vs_raw": study.bootstrap_delta(
                        output["baselines"][str(length)]["raw_motor_moving"],
                        moving_left, 3_000_000 + seed * 100 + length,
                        cfg.bootstrap_resamples,
                    ),
                }
                if arm == "ga_calibrated":
                    record["lengths"][str(length)]["exact_audit"] = study.exact_path_audit(
                        data, split, length, cfg, model
                    )
            output["arms"][arm][str(seed)] = record
            print(f"evaluated {split}: {arm} seed={seed}", flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=study.Config.steps)
    args = parser.parse_args()
    cfg = study.Config(
        steps=args.steps,
        sample_period_seconds=0.1,
        maximum_alignment_seconds=0.02,
        moving_threshold_m=0.15,
    )
    split = "confirm" if args.confirm else "dev"
    result = run(split, cfg, torch.device(args.device))
    destination = HERE / (
        "results_confirmation.json" if args.confirm else "results_development.json"
    )
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
