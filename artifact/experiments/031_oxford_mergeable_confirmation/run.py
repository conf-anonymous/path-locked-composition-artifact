"""Oxford-trained architectural replication; no tuning and no standalone release."""
from __future__ import annotations

import argparse
import copy
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
sys.path.insert(0, str(HERE))
import oxford_workflow as wf

ox = wf.original()
saved_manifest_module = sys.modules.pop("data_manifest", None)
try:
    merge = wf.load_module("oxford_mergeable_architecture", HERE.parent / "029_eth3d_mergeable_motor_gate/run.py")
finally:
    if saved_manifest_module is not None:
        sys.modules["data_manifest"] = saved_manifest_module
study = merge.study
base = ox.base
HEADS = ("constant", "contextual", "feature_direct", "feature_residual", "mergeable")
ALL_ARMS = ("identity", "raw", "ga_calibrated", *HEADS)
RUNS = HERE / "runs"
HEAD_CONFIG = {"steps": 5000, "batch_size": 256, "learning_rate": .001,
               "weight_decay": .0001, "gradient_clip": 1., "train_length": 8}


def new_head(arm, seed):
    torch.manual_seed(50000 + seed)
    return merge.MergeableGate() if arm == "mergeable" else study.Head(arm)


@torch.no_grad()
def cache(leaves, targets, calibrator, batch_size=1024):
    raw, calibrated = [], []
    calibrator = copy.deepcopy(calibrator).cpu().float().eval()
    leaves = leaves.float()
    for offset in range(0, len(leaves), batch_size):
        batch = leaves[offset:offset + batch_size]
        raw.append(base.reduce_states(list(batch.unbind(1)), base.motor_product, "balanced"))
        corrected = calibrator(batch)
        calibrated.append(base.reduce_states(list(corrected.unbind(1)), base.motor_product, "balanced"))
    return leaves, torch.cat(raw), torch.cat(calibrated), targets.float()


def groups_for(names):
    groups = {}
    for index, name in enumerate(names):
        groups.setdefault(name, []).append(index)
    return [torch.tensor(groups[name], dtype=torch.long) for name in sorted(groups)]


def sample_indices(groups, batch_size, generator):
    chosen = torch.randint(len(groups), (batch_size,), generator=generator)
    return torch.tensor([int(groups[g][int(torch.randint(len(groups[g]), (1,), generator=generator))])
                         for g in chosen.tolist()], dtype=torch.long)


def fit_head(head, cached, names, scale, seed, device, *, steps=5000):
    """Short steps are used only by recorded-ETH3D engineering tests, never CLI runs."""
    head = head.to(device).train()
    optimizer = torch.optim.AdamW(head.parameters(), lr=.001, weight_decay=.0001)
    groups = groups_for(names)
    generator = torch.Generator().manual_seed(30000 + seed)
    start = time.perf_counter()
    loss_value = None
    for step in range(steps):
        index = sample_indices(groups, 256, generator)
        leaves, raw, calibrated, targets = (t[index].to(device) for t in cached)
        loss = study.v.v2.robust_motor_loss(head(leaves, raw, calibrated), targets, scale)
        if not bool(loss.isfinite()):
            raise RuntimeError("nonfinite extension training loss; retain failure, do not retune")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(head.parameters(), 1.)
        optimizer.step()
        loss_value = float(loss.detach())
        if (step + 1) % 1000 == 0:
            print(f"extension seed={seed} step={step+1} loss={loss_value:.6f}", flush=True)
    return {"seconds": time.perf_counter() - start, "final_loss": loss_value, "steps": steps}


def calibrator_for(seed, cfg, device="cpu"):
    path = ox.RUNS / f"ga_calibrated_seed{seed}.pt"
    meta = {"arm": "ga_calibrated", "seed": seed, "config": asdict(cfg), **wf.fingerprint()}
    checkpoint = wf.checked_checkpoint(path, meta, device)
    model = ox.OxfordComposer("ga_calibrated").to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.calibrator.requires_grad_(False).eval()
    return model.calibrator, wf.sha(path)


def head_meta(arm, seed, calibrator_sha):
    return {"arm": arm, "seed": seed, "config": HEAD_CONFIG,
            "calibrator_sha256": calibrator_sha, **wf.fingerprint()}


def fit_all(data, cfg, device):
    RUNS.mkdir(exist_ok=True)
    leaves, targets, _ = data.tensors[("train", 8)]
    names = data.examples[("train", 8)]
    # Complete every head fit before the caller evaluates development utility.
    for seed in ox.SEEDS:
        calibrator, calibrator_sha = calibrator_for(seed, cfg)
        cached = cache(leaves, targets, calibrator)
        for arm in HEADS:
            path = RUNS / f"{arm}_seed{seed}.pt"
            meta = head_meta(arm, seed, calibrator_sha)
            if path.exists():
                wf.checked_checkpoint(path, meta)
                continue
            head = new_head(arm, seed)
            training = fit_head(head, cached, names, data.translation_scale, seed, device)
            wf.save_checkpoint(path, {"meta": meta, "head": head.cpu().state_dict(),
                                      "training": training})
            print(f"frozen extension {arm} seed={seed}", flush=True)


def metrics(prediction, targets, names, mask):
    if not bool(mask.any()):
        return {"n_windows": 0, "n_trajectories": 0, "clusters": {}}
    translation, rotation = base.errors(prediction[mask].double(), targets[mask].double())
    translation, rotation = np.asarray(translation), np.asarray(rotation)
    if not (np.isfinite(translation).all() and np.isfinite(rotation).all()):
        raise RuntimeError("nonfinite extension evaluation")
    selected = np.asarray(names)[mask.numpy()]
    clusters = {name: {"n": int(np.sum(selected == name)),
                       "translation_mean_m": float(translation[selected == name].mean()),
                       "rotation_mean_deg": float(rotation[selected == name].mean())}
                for name in sorted(set(selected))}
    return {"n_windows": int(mask.sum()), "n_trajectories": len(clusters), "clusters": clusters}


@torch.no_grad()
def predict(head, cached, device):
    leaves, raw, calibrated, _ = cached
    predictions = []
    minimum = float("inf")
    head.eval().to(device)
    for offset in range(0, len(leaves), 1024):
        batch = [t[offset:offset+1024].to(device) for t in (leaves, raw, calibrated)]
        before = head.before_projection(*batch)
        minimum = min(minimum, float(before[:, :4].norm(dim=-1).min()))
        result = base.normalize_motor(before)
        if not bool(result.isfinite().all()):
            raise RuntimeError("nonfinite head prediction")
        predictions.append(result.cpu())
    return torch.cat(predictions), minimum


@torch.no_grad()
def paired_state_audit(head, calibrator, operands, length):
    head = copy.deepcopy(head).cpu().double().eval()
    calibrator = copy.deepcopy(calibrator).cpu().double().eval()
    maxima = {p: {"state_absolute": 0., "translation_m": 0., "rotation_rad": 0.}
              for p in ("right", "balanced", "random")}
    minimum = float("inf")
    def product(a, b):
        return torch.cat((base.motor_product(a[:, :8], b[:, :8]),
                          base.motor_product(a[:, 8:], b[:, 8:])), -1)
    for offset in range(0, len(operands), 1024):
        leaves = operands[offset:offset+1024].double()
        pair_leaves = torch.cat((leaves, calibrator(leaves)), -1)
        left = base.reduce_states(list(pair_leaves.unbind(1)), product, "left")
        before = head.before_projection(leaves, left[:, :8], left[:, 8:])
        minimum = min(minimum, float(before[:, :4].norm(dim=-1).min()))
        q0, t0 = base.decode_motor(base.normalize_motor(before))
        for path in maxima:
            for repeat in range(16 if path == "random" else 1):
                rng = random.Random(length * 100003 + offset * 101 + repeat)
                other = base.reduce_states(list(pair_leaves.unbind(1)), product, path,
                                           rng if path == "random" else None)
                before_other = head.before_projection(leaves, other[:, :8], other[:, 8:])
                minimum = min(minimum, float(before_other[:, :4].norm(dim=-1).min()))
                q1, t1 = base.decode_motor(base.normalize_motor(before_other))
                relative = base.qmul(base.qconj(q0), q1)
                angle = 2 * torch.atan2(relative[:, 1:].norm(dim=-1), relative[:, 0].abs())
                values = (float((other-left).abs().max()), float((t1-t0).norm(dim=-1).max()),
                          float(angle.max()))
                if not all(np.isfinite(values)) or not bool(other.isfinite().all()):
                    raise RuntimeError("nonfinite complete paired-summary audit")
                for key, value in zip(maxima[path], values):
                    maxima[path][key] = max(maxima[path][key], value)
    return {"maxima": maxima, "minimum_unclamped_real_norm": minimum,
            "passes": all(v["translation_m"] < 1e-9 and v["rotation_rad"] < 1e-9
                          for v in maxima.values()), "finite": True}


def evaluate_all(data, cfg, split, device):
    result = {"split": split, "config": HEAD_CONFIG, "fingerprint": wf.fingerprint(),
              "calibrator_config": asdict(cfg), "seeds": {}}
    # Verify every fit before any evaluation, including on confirmation.
    for seed in ox.SEEDS:
        _, checksum = calibrator_for(seed, cfg)
        for arm in HEADS:
            wf.checked_checkpoint(RUNS / f"{arm}_seed{seed}.pt", head_meta(arm, seed, checksum))
    for seed in ox.SEEDS:
        record_path = RUNS / f"evaluation_{split}_seed{seed}.json"
        if record_path.exists():
            record = wf.read(record_path)
            if record["fingerprint"] != wf.fingerprint():
                raise RuntimeError("evaluation provenance changed")
            result["seeds"][str(seed)] = record
            continue
        calibrator, checksum = calibrator_for(seed, cfg)
        heads = {}
        hashes = {}
        for arm in HEADS:
            path = RUNS / f"{arm}_seed{seed}.pt"
            saved = wf.checked_checkpoint(path, head_meta(arm, seed, checksum))
            heads[arm] = new_head(arm, seed)
            heads[arm].load_state_dict(saved["head"])
            hashes[arm] = wf.sha(path)
        record = {"fingerprint": wf.fingerprint(), "calibrator_sha256": checksum,
                  "head_sha256": hashes, "lengths": {}}
        for length in ox.EVAL_LENGTHS:
            leaves, targets, moving = data.tensors[(split, length)]
            names = data.examples[(split, length)]
            cached = cache(leaves, targets, calibrator)
            identity = cached[1].new_zeros(cached[1].shape)
            identity[:, 0] = 1.
            predictions = {"identity": (identity, None), "raw": (cached[1], None),
                           "ga_calibrated": (cached[2], None)}
            predictions.update({arm: predict(head, cached, device) for arm, head in heads.items()})
            record["lengths"][str(length)] = {
                "arms": {arm: {"all": metrics(pred, targets, names, torch.ones_like(moving)),
                               "moving": metrics(pred, targets, names, moving),
                               "minimum_unclamped_real_norm": minimum}
                         for arm, (pred, minimum) in predictions.items()},
                "numerical_audit": paired_state_audit(heads["mergeable"], calibrator, leaves, length),
            }
        wf.finite_tree(record)
        wf.save(record_path, record)
        result["seeds"][str(seed)] = record
        print(f"extension evaluated {split} seed={seed}", flush=True)
    return result


def run(split="dev", device="cpu"):
    if split == "confirm":
        wf.require_confirmation_release()
    elif split == "dev":
        wf.require_development_ready()
    else:
        raise ValueError(split)
    wf.verify_inputs()
    path = HERE / ("results_development.json" if split == "dev" else "results_confirmation.json")
    if path.exists():
        raise RuntimeError(f"refusing to rerun completed campaign: {path}")
    cfg = ox.Config()
    data = ox.OxfordData(cfg, include_confirmation=(split == "confirm"))
    if split == "dev":
        fit_all(data, cfg, torch.device(device))
    result = evaluate_all(data, cfg, split, torch.device(device))
    wf.save(path, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--confirm", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    run("confirm" if args.confirm else "dev", args.device)
