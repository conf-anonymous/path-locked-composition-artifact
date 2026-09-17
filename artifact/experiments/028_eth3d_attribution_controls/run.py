"""Feature-matched GA attribution on unmodified public ETH3D recordings."""
from __future__ import annotations

import concurrent.futures
import copy
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
spec = importlib.util.spec_from_file_location("eth028", HERE.parent / "023_eth3d_family_heldout/run.py")
v = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = v
spec.loader.exec_module(v)
ARMS = ("constant", "contextual", "feature_direct", "feature_residual")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


class Head(nn.Module):
    def __init__(self, arm):
        super().__init__()
        self.arm = arm
        if arm == "constant":
            self.alpha = nn.Parameter(torch.tensor(-2.))
        else:
            self.encoder = nn.GRU(8, 32, batch_first=True, bidirectional=True)
            self.head = nn.Sequential(nn.Linear(80, 32), nn.GELU(),
                                      nn.Linear(32, 1 if arm == "contextual" else 8))
            nn.init.zeros_(self.head[-1].weight)
            nn.init.zeros_(self.head[-1].bias)
            if arm == "contextual":
                nn.init.constant_(self.head[-1].bias, -2.)
            elif arm == "feature_direct":
                with torch.no_grad():
                    self.head[-1].bias[0] = 1.

    def before_projection(self, leaves, raw, calibrated):
        if self.arm == "constant":
            weight = self.alpha.sigmoid()
            return raw + weight * (calibrated - raw)
        encoded, _ = self.encoder(leaves)
        features = torch.cat((encoded.mean(1), raw, calibrated), -1)
        output = self.head(features)
        if self.arm == "feature_direct":
            return output
        weight = output.sigmoid() if self.arm == "contextual" else torch.sigmoid(raw.new_tensor(-2.))
        blend = raw + weight * (calibrated - raw)
        return blend + output if self.arm == "feature_residual" else blend

    def forward(self, leaves, raw, calibrated):
        return v.base.normalize_motor(self.before_projection(leaves, raw, calibrated))


@torch.inference_mode()
def cache(data, calibrator, split):
    leaves, target, _ = data.tensors[(split, 32)]
    leaves = leaves.float()
    raw, cal = [], []
    for start in range(0, len(leaves), 1024):
        batch = leaves[start:start+1024]
        raw.append(v.base.reduce_states(list(batch.unbind(1)), v.base.motor_product, "balanced"))
        cal.append(calibrator(batch))
    # Clone outside inference_mode before training uses these fixed tensors.
    return leaves, torch.cat(raw), torch.cat(cal), target.float()


def metrics(prediction, targets, names):
    trans, rot = v.base.errors(prediction.double(), targets.double())
    trans, rot, names = np.asarray(trans), np.asarray(rot), np.asarray(names)
    return {"n_windows": len(names), "clusters": {
        name: {"family": v.FAMILY_FOR[name], "n": int(np.sum(names == name)),
               "translation_mean_m": float(trans[names == name].mean()),
               "rotation_mean_deg": float(rot[names == name].mean())}
        for name in sorted(set(names))}}


@torch.inference_mode()
def evaluate(head, cached, data):
    leaves, raw, cal, targets = cached
    predictions, min_norm = [], float("inf")
    for start in range(0, len(leaves), 1024):
        batch = (leaves[start:start+1024], raw[start:start+1024], cal[start:start+1024])
        before = head.before_projection(*batch)
        min_norm = min(min_norm, float(before[:, :4].norm(dim=-1).min()))
        predictions.append(v.base.normalize_motor(before))
    result = metrics(torch.cat(predictions), targets, data.examples[("test", 32)])
    result["minimum_real_norm_before_projection"] = min_norm
    return result


@torch.inference_mode()
def numeric_audit(head, calibrator, data):
    head, calibrator = copy.deepcopy(head).double(), copy.deepcopy(calibrator).double()
    leaves, _, _ = data.tensors[("test", 32)]
    maxima = {p: {"translation_m": 0., "rotation_rad": 0.} for p in ("right", "balanced", "chunks_5_11_16")}
    min_norm = float("inf")
    cache_delta = 0.
    for start in range(0, len(leaves), 512):
        batch = leaves[start:start+512]
        cal_leaves = calibrator.calibrator(batch)
        def product(x, path):
            states = list(x.unbind(1))
            if path == "chunks_5_11_16":
                states = [v.base.reduce_states(states[a:b], v.base.motor_product, "left")
                          for a, b in ((0, 5), (5, 16), (16, 32))]
                path = "balanced"
            return v.base.reduce_states(states, v.base.motor_product, path)
        def predict(path):
            return head.before_projection(batch, product(batch, path), product(cal_leaves, path))
        before = predict("left")
        min_norm = min(min_norm, float(before[:, :4].norm(dim=-1).min()))
        q0, t0 = v.base.decode_motor(before)
        for path in maxima:
            q1, t1 = v.base.decode_motor(predict(path))
            relative = v.base.qmul(v.base.qconj(q0), q1)
            angle = 2 * torch.atan2(relative[:, 1:].norm(dim=-1), relative[:, 0].abs())
            maxima[path]["translation_m"] = max(maxima[path]["translation_m"], float((t0-t1).norm(dim=-1).max()))
            maxima[path]["rotation_rad"] = max(maxima[path]["rotation_rad"], float(angle.max()))
        # Replace one cache leaf with its *recorded calibrated* value, not a perturbation.
        # Reuse left/right cached blocks and compare to a complete corrected reduction.
        changed = batch.clone()
        changed[:, 15] = cal_leaves[:, 15]
        cached = v.base.motor_product(v.base.motor_product(
            v.base.reduce_states(list(batch[:, :15].unbind(1)), v.base.motor_product, "balanced"),
            cal_leaves[:, 15]),
            v.base.reduce_states(list(batch[:, 16:].unbind(1)), v.base.motor_product, "balanced"))
        full = product(changed, "left")
        cache_delta = max(cache_delta, float((cached-full).abs().max()))
    return {"path_discrepancy": maxima, "minimum_float64_real_norm": min_norm,
            "cached_recorded_calibration_update_max_coordinate_delta": cache_delta,
            "scope": "products regrouped; identical full-sequence GRU retained"}


def run_job(job):
    fold, seed = job
    torch.set_num_threads(1)
    destination = HERE / "runs" / f"fold{fold}_seed{seed}.json"
    if destination.exists():
        return json.loads(destination.read_text())
    source = json.loads((v.HERE / "results.json").read_text())
    cfg = v.Config(**source["config"])
    data = v.FoldData(v.ETH3DAll(), fold)
    calibrator, training = v.train_calibrator(data, cfg, seed, torch.device("cpu"))
    expected = source["folds"][str(fold)]["seeds"][str(seed)]["lengths"]["32"]["ga_calibrated"]["clusters"]
    observed = v.evaluate(data, 32, cfg, calibrator)["clusters"]
    difference = max(abs(expected[n][k] - observed[n][k]) for n in expected
                     for k in ("translation_mean_m", "rotation_mean_deg"))
    if difference > 1e-6:
        raise RuntimeError(f"calibrator failed reproduction {fold}/{seed}: {difference}")
    train_cache = tuple(t.clone() for t in cache(data, calibrator, "train"))
    test_cache = tuple(t.clone() for t in cache(data, calibrator, "test"))
    sampler = v.v2.BalancedSampler(data)
    result = {"fold": fold, "seed": seed, "calibrator_reproduction_max": difference,
              "calibrator_training": training, "arms": {}}
    checkpoints = {"calibrator": calibrator.state_dict()}
    for arm in ARMS:
        torch.manual_seed(50000 + seed)
        head = Head(arm)
        gen = torch.Generator().manual_seed(seed + 30000)
        optimizer = torch.optim.AdamW(head.parameters(), lr=cfg.contextual_learning_rate,
                                     weight_decay=cfg.weight_decay)
        started = time.perf_counter()
        for step in range(cfg.contextual_steps):
            idx = sampler.sample(32, cfg.batch_size, gen)
            leaves, raw, cal, target = (t[idx] for t in train_cache)
            loss = v.v2.robust_motor_loss(head(leaves, raw, cal), target, data.translation_scale)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), 1.)
            optimizer.step()
        seconds = time.perf_counter() - started
        head.eval()
        result["arms"][arm] = evaluate(head, test_cache, data)
        result["arms"][arm].update(training_seconds=seconds, parameters=sum(p.numel() for p in head.parameters()))
        if arm == "contextual":
            result["arms"][arm]["numerical_audit"] = numeric_audit(head, calibrator, data)
        checkpoints[arm] = head.state_dict()
        print(f"finished fold={fold} seed={seed} {arm} {seconds:.1f}s", flush=True)
    checkpoint = destination.with_suffix(".pt")
    torch.save(checkpoints, checkpoint)
    result["checkpoint_sha256"] = sha(checkpoint)
    save(destination, result)
    return result


def main():
    (HERE / "runs").mkdir(exist_ok=True)
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in (Path(__file__), HERE / "PROTOCOL.md", Path(v.__file__))}
    if not (HERE / "seal.json").exists():
        save(HERE / "seal.json", hashes)
    if json.loads((HERE / "seal.json").read_text()) != hashes:
        raise RuntimeError("source changed after seal; do not silently resume")
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:
        records = list(pool.map(run_job, [(f, s) for f in range(5) for s in range(5)]))
    save(HERE / "results.json", {"status": "post-hoc grouped attribution, no held-out tuning",
                               "seal": hashes, "records": records})


if __name__ == "__main__":
    main()
