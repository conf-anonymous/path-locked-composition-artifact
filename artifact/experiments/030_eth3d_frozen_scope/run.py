"""No-fit, public-recording scope audit of every frozen ETH3D head."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
spec = importlib.util.spec_from_file_location("mergeable030", HERE.parent / "029_eth3d_mergeable_motor_gate/run.py")
experiment = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = experiment
spec.loader.exec_module(experiment)
study, v = experiment.study, experiment.v
LENGTHS = (4, 8, 16, 32, 64, 128)
ARMS = ("identity", "raw", "calibrated", *study.ARMS, "mergeable")
METRICS = ("translation_mean_m", "rotation_mean_deg")


def dependencies():
    paths = [HERE / "PROTOCOL.md", Path(__file__), v.MANIFEST]
    for dirname in ("017_tum_motor_composition", "019_tum_long_horizon_confirmation",
                    "020_oxford_robotcar_vo", "021_eth3d_rgbd_motor", "022_eth3d_adaptive_motor",
                    "023_eth3d_family_heldout", "028_eth3d_attribution_controls",
                    "029_eth3d_mergeable_motor_gate"):
        paths.extend((HERE.parent / dirname).glob("*.py"))
    manifest = json.loads(v.MANIFEST.read_text())
    paths.extend(v.DATA / r["derived_path"] for r in manifest["records"])
    for directory in (study.HERE, experiment.HERE):
        paths.extend((directory / "runs").glob("*.pt"))
        paths.extend((directory / "runs").glob("*.json"))
        paths.append(directory / "results.json")
    return {str(p.relative_to(ROOT)): study.sha(p) for p in sorted(set(paths))}


def load_recorded_windows(all_data, length):
    if length in all_data.tensors:
        leaves, targets = all_data.tensors[length]
        return leaves, targets, all_data.examples[length]
    operands, targets, names = [], [], []
    for name in v.v1.SEQUENCES:
        for leaves, reference in all_data.runs[name]:
            count = len(leaves) - length + 1
            if count <= 0:
                continue
            operands.append(leaves.unfold(0, length, 1).permute(0, 2, 1).contiguous())
            targets.append(v.v1.study.relative_from_absolute(reference[:count], reference[length:length+count]))
            names.extend([name] * count)
    if not operands:
        return None
    return torch.cat(operands), torch.cat(targets), names


def durations(length):
    values = []
    for record in json.loads(v.MANIFEST.read_text())["records"]:
        arrays = np.load(v.DATA / record["derived_path"], allow_pickle=False)
        start = None
        for index, keep in enumerate(np.append(arrays["pair_valid"].astype(bool), False)):
            if keep and start is None:
                start = index
            elif not keep and start is not None:
                if index-start >= max(32, length):
                    times = arrays["timestamps"]
                    values.extend(times[start+length:index+1] - times[start:index-length+1])
                start = None
    return {"n": len(values), "min_seconds": float(np.min(values)),
            "median_seconds": float(np.median(values)), "max_seconds": float(np.max(values))} if values else {"n": 0}


@torch.inference_mode()
def evaluate(leaves, targets, names, cal, heads):
    predictions = {a: [] for a in ARMS}
    minima = {a: float("inf") for a in heads}
    for start in range(0, len(leaves), 512):
        batch = leaves[start:start+512].float()
        raw = v.base.reduce_states(list(batch.unbind(1)), v.base.motor_product, "balanced")
        corrected = cal(batch)
        identity = torch.zeros_like(raw)
        identity[:, 0] = 1.
        for name, p in (("identity", identity), ("raw", raw), ("calibrated", corrected)):
            predictions[name].append(p)
        for name, head in heads.items():
            before = head.before_projection(batch, raw, corrected)
            minima[name] = min(minima[name], float(before[:, :4].norm(dim=-1).min()))
            predictions[name].append(v.base.normalize_motor(before))
    result = {}
    for name, chunks in predictions.items():
        prediction = torch.cat(chunks)
        assert torch.isfinite(prediction).all(), name
        result[name] = study.metrics(prediction, targets.float(), names)
        if name in minima:
            result[name]["minimum_real_norm_before_projection"] = minima[name]
    return result


def summarize(records):
    summary = {"scope": "Post-hoc frozen-checkpoint robustness, not untouched confirmation; exploratory conditional intervals, no multiplicity adjustment.", "lengths": {}}
    for length in LENGTHS:
        selected = [r for r in records if r["length"] == length]
        if not selected:
            summary["lengths"][str(length)] = {"status": "no eligible windows"}
            continue
        families = sorted({c["family"] for r in selected for c in r["arms"]["mergeable"]["clusters"].values()})
        arrays = {m: {a: np.full((5, len(families)), np.nan) for a in ARMS} for m in METRICS}
        for r in selected:
            for arm in ARMS:
                clusters = r["arms"][arm]["clusters"]
                for family in {c["family"] for c in clusters.values()}:
                    for metric in METRICS:
                        arrays[metric][arm][r["seed"], families.index(family)] = np.mean([c[metric] for c in clusters.values() if c["family"] == family])
        assert all(np.isfinite(a).all() for methods in arrays.values() for a in methods.values())
        item = {"families": families, "missing_families": sorted(set(v.FAMILY_SEQUENCES)-set(families)), "duration": durations(length), "metrics": {}}
        count = sum(r["arms"]["mergeable"]["n_windows"] for r in selected if r["seed"] == 0)
        assert item["duration"]["n"] == count
        item["n_windows"] = count
        item["n_sequences"] = len({n for r in selected for n in r["arms"]["mergeable"]["clusters"]})
        for metric, methods in arrays.items():
            means = {a: {"mean": float(x.mean()), "per_seed": x.mean(1).tolist(), "per_family": dict(zip(families, x.mean(0).tolist()))} for a, x in methods.items()}
            contrasts = {}
            for arm in ARMS:
                if arm == "mergeable":
                    continue
                delta = methods["mergeable"] - methods[arm]
                family_delta = delta.mean(0)
                rng = np.random.default_rng(300906)
                draws = family_delta[rng.integers(0, len(families), (10000, len(families)))].mean(1)
                contrasts[arm] = {"mean_delta": float(delta.mean()), "conditional_ci95": np.quantile(draws, [.025, .975]).tolist(), "family_wins": int(np.sum(family_delta < 0))}
            item["metrics"][metric] = {"methods": means, "mergeable_minus": contrasts}
        points = {a: np.array([arrays[m][a].mean() for m in METRICS]) for a in ARMS}
        item["descriptive_pareto_frontier"] = [a for a, p in points.items() if not any(np.all(q <= p) and np.any(q < p) for b, q in points.items() if b != a)]
        item["minimum_real_norm"] = {a: min(r["arms"][a]["minimum_real_norm_before_projection"] for r in selected) for a in (*study.ARMS, "mergeable")}
        summary["lengths"][str(length)] = item
    return summary


def main():
    torch.set_num_threads(1)
    (HERE / "runs").mkdir(exist_ok=True)
    observed = dependencies()
    seal = HERE / "seal.json"
    if not seal.exists():
        study.save(seal, observed)
    assert json.loads(seal.read_text()) == observed, "dependency changed after seal"
    all_data = v.ETH3DAll()
    windows = {length: load_recorded_windows(all_data, length) for length in LENGTHS}
    records = []
    max_reproduction = 0.
    for fold in range(5):
        for seed in range(5):
            stem = f"fold{fold}_seed{seed}"
            source = study.HERE / "runs" / f"{stem}.pt"
            merge_path = experiment.HERE / "runs" / f"{stem}.pt"
            original = json.loads(source.with_suffix(".json").read_text())
            merge_record = json.loads(merge_path.with_suffix(".json").read_text())
            assert study.sha(source) == original["checkpoint_sha256"] == merge_record["source_checkpoint_sha256"]
            assert study.sha(merge_path) == merge_record["checkpoint_sha256"]
            saved = torch.load(source, weights_only=True)
            cal = v.CalibratedMotor().eval()
            cal.load_state_dict(saved["calibrator"])
            heads = {a: study.Head(a).eval() for a in study.ARMS}
            for a, head in heads.items():
                head.load_state_dict(saved[a])
            heads["mergeable"] = experiment.MergeableGate().eval()
            heads["mergeable"].load_state_dict(torch.load(merge_path, weights_only=True))
            # Reproduction runs first, before any additional horizon is evaluated.
            for length in (32, 4, 8, 16, 64, 128):
                if windows[length] is None:
                    continue
                leaves, targets, names = windows[length]
                indices = [i for i, n in enumerate(names) if v.FAMILY_FOR[n] in v.FOLD_FAMILIES[fold]]
                if not indices:
                    continue
                destination = HERE / "runs" / f"{stem}_L{length}.json"
                if destination.exists():
                    record = json.loads(destination.read_text())
                    assert record["checkpoint_sha256"] == study.sha(source)
                    assert record["mergeable_checkpoint_sha256"] == study.sha(merge_path)
                else:
                    result = evaluate(leaves[indices], targets[indices], [names[i] for i in indices], cal, heads)
                    record = {"fold": fold, "seed": seed, "length": length, "checkpoint_sha256": study.sha(source), "mergeable_checkpoint_sha256": study.sha(merge_path), "arms": result}
                    study.save(destination, record)
                if length == 32:
                    for arm in heads:
                        expected = merge_record["evaluation"] if arm == "mergeable" else original["arms"][arm]
                        actual = record["arms"][arm]
                        assert expected["n_windows"] == actual["n_windows"]
                        assert expected["clusters"].keys() == actual["clusters"].keys()
                        delta = max(abs(expected["clusters"][n][m] - actual["clusters"][n][m]) for n in actual["clusters"] for m in METRICS)
                        max_reproduction = max(max_reproduction, delta)
                        assert delta < 1e-5, (fold, seed, arm, delta)
                records.append(record)
            print(f"completed fold={fold} seed={seed}", flush=True)
    summary = summarize(records)
    summary["max_L32_reproduction_absolute_difference"] = max_reproduction
    summary["seal_sha256"] = study.sha(seal)
    summary["run_record_sha256"] = {str(p.relative_to(HERE)): study.sha(p) for p in sorted((HERE / "runs").glob("*.json"))}
    study.save(HERE / "analysis.json", summary)
    for length, item in summary["lengths"].items():
        print(length, {m: {a: round(r["mean"], 6) for a, r in x["methods"].items()} for m, x in item.get("metrics", {}).items()}, flush=True)


if __name__ == "__main__":
    main()
