"""Fixed-size mergeable PGA product-pair state with learned endpoint gate."""
from __future__ import annotations

import concurrent.futures
import importlib.util
import json
import sys
import time
from pathlib import Path

import torch
from torch import nn

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("attribution029", HERE.parent / "028_eth3d_attribution_controls/run.py")
study = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = study
spec.loader.exec_module(study)
v = study.v


class MergeableGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = nn.Sequential(nn.Linear(16, 32), nn.GELU(), nn.Linear(32, 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.constant_(self.head[-1].bias, -2.)

    def before_projection(self, leaves, raw, calibrated):
        del leaves
        alpha = self.head(torch.cat((raw, calibrated), -1)).sigmoid()
        return raw + alpha * (calibrated-raw)

    def forward(self, leaves, raw, calibrated):
        return v.base.normalize_motor(self.before_projection(leaves, raw, calibrated))


def run_job(job):
    fold, seed = job
    torch.set_num_threads(1)
    path = HERE / "runs" / f"fold{fold}_seed{seed}.json"
    if path.exists():
        return json.loads(path.read_text())
    source_path = study.HERE / "runs" / f"fold{fold}_seed{seed}.pt"
    if not source_path.exists():
        return None  # Parent only schedules completed, checksum-recorded dependencies.
    source_record = json.loads(source_path.with_suffix(".json").read_text())
    assert study.sha(source_path) == source_record["checkpoint_sha256"]
    cfg = v.Config(**json.loads((v.HERE / "results.json").read_text())["config"])
    data = v.FoldData(v.ETH3DAll(), fold)
    cal = v.CalibratedMotor()
    cal.load_state_dict(torch.load(source_path, weights_only=True)["calibrator"])
    train = tuple(t.clone() for t in study.cache(data, cal, "train"))
    test = tuple(t.clone() for t in study.cache(data, cal, "test"))
    torch.manual_seed(seed + 50000)
    head = MergeableGate()
    gen = torch.Generator().manual_seed(seed + 30000)
    sampler = v.v2.BalancedSampler(data)
    optimizer = torch.optim.AdamW(head.parameters(), lr=cfg.contextual_learning_rate, weight_decay=cfg.weight_decay)
    started = time.perf_counter()
    for _ in range(cfg.contextual_steps):
        idx = sampler.sample(32, cfg.batch_size, gen)
        leaves, raw, calibrated, targets = (t[idx] for t in train)
        loss = v.v2.robust_motor_loss(head(leaves, raw, calibrated), targets, data.translation_scale)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(head.parameters(), 1.)
        optimizer.step()
    seconds = time.perf_counter() - started
    head.eval()
    result = {"fold": fold, "seed": seed, "source_checkpoint_sha256": study.sha(source_path),
              "training_seconds": seconds, "parameters": sum(p.numel() for p in head.parameters()),
              "evaluation": study.evaluate(head, test, data),
              "numerical_audit": study.numeric_audit(head, cal, data)}
    torch.save(head.state_dict(), path.with_suffix(".pt"))
    result["checkpoint_sha256"] = study.sha(path.with_suffix(".pt"))
    study.save(path, result)
    print(f"mergeable fold={fold} seed={seed} {seconds:.1f}s", flush=True)
    return result


def main():
    (HERE / "runs").mkdir(exist_ok=True)
    hashes = {str(p.relative_to(study.ROOT)): study.sha(p) for p in (Path(__file__), HERE / "PROTOCOL.md", Path(study.__file__))}
    seal = HERE / "seal.json"
    if not seal.exists():
        study.save(seal, hashes)
    assert json.loads(seal.read_text()) == hashes
    ready = [(f,s) for f in range(5) for s in range(5)
             if (study.HERE / "runs" / f"fold{f}_seed{s}.json").exists()]
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:
        records = list(pool.map(run_job, ready))
    assert all(r is not None for r in records)
    if len(records) == 25:
        study.save(HERE / "results.json", {"status": "post-hoc mergeable-state ablation, no held-out tuning", "seal": hashes, "records": records})
    print(f"Completed {len(records)}/25 available dependencies. Rerun after Experiment 028 completes if needed.")


if __name__ == "__main__":
    main()
