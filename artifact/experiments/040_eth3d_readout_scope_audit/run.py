"""Frozen readout and matched-start sensitivity on real ETH3D recordings."""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import platform
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
spec = importlib.util.spec_from_file_location(
    "scope040", HERE.parent / "030_eth3d_frozen_scope/run.py")
scope = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = scope
spec.loader.exec_module(scope)
experiment, study, v = scope.experiment, scope.study, scope.v
LENGTHS = scope.LENGTHS
ORIGINAL = scope.ARMS
ALIGNED = ("contextual_aligned", "mergeable_aligned")
ARMS = (*ORIGINAL, *ALIGNED)
COHORTS = ("full", "common_sequences", "matched_starts")
METRICS = scope.METRICS


def sha(path):
    return v.sha256(Path(path))


def save(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def dependencies():
    paths = {Path(__file__), HERE / "PROTOCOL.md", v.MANIFEST}
    for module in tuple(sys.modules.values()):
        filename = getattr(module, "__file__", None)
        if filename:
            path = Path(filename).resolve()
            if path.is_relative_to(ROOT / "experiments") and path.suffix == ".py":
                paths.add(path)
    manifest = json.loads(v.MANIFEST.read_text())
    paths.update(v.DATA / r["derived_path"] for r in manifest["records"])
    for directory in (study.HERE, experiment.HERE, scope.HERE):
        paths.update((directory / "runs").glob("*.json"))
        paths.update((directory / "runs").glob("*.pt"))
        paths.add(directory / "analysis.json")
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(paths)}


def windows(data, length):
    operands, targets, names, ids, matched = [], [], [], [], []
    for name in v.v1.SEQUENCES:
        for run_id, (leaves, reference) in enumerate(data.runs[name]):
            count = len(leaves) - length + 1
            if count <= 0:
                continue
            operands.append(leaves.unfold(0, length, 1).permute(0, 2, 1).contiguous())
            targets.append(v.v1.study.relative_from_absolute(
                reference[:count], reference[length:length + count]))
            names.extend([name] * count)
            ids.extend((name, run_id, start) for start in range(count))
            matched.extend(start < len(leaves) - 128 + 1 for start in range(count))
    result = (torch.cat(operands), torch.cat(targets), names, ids,
              np.asarray(matched, dtype=bool))
    old_leaves, old_targets, old_names = scope.load_recorded_windows(data, length)
    assert names == old_names
    assert torch.equal(result[0], old_leaves) and torch.equal(result[1], old_targets)
    return result


def weight(head, leaves, raw, calibrated, kind):
    if kind == "mergeable":
        features = torch.cat((raw, calibrated), -1)
    else:
        encoded, _ = head.encoder(leaves)
        features = torch.cat((encoded.mean(1), raw, calibrated), -1)
    return head.head(features).sigmoid()


def aligned_blend(raw, calibrated, alpha):
    dot = (raw[:, :4] * calibrated[:, :4]).sum(-1, keepdim=True)
    sign = torch.where(dot < 0, -torch.ones_like(dot), torch.ones_like(dot))
    return raw + alpha * (sign * calibrated - raw)


def check_against_record(result, expected):
    assert result["n_windows"] == expected["n_windows"]
    assert result["clusters"].keys() == expected["clusters"].keys()
    maximum = 0.0
    for name, actual in result["clusters"].items():
        assert actual["n"] == expected["clusters"][name]["n"]
        for metric in METRICS:
            maximum = max(maximum, abs(actual[metric] - expected["clusters"][name][metric]))
    assert maximum < 1e-5, maximum
    return maximum


@torch.inference_mode()
def evaluate(leaves, targets, names, masks, cal, heads):
    predictions = {arm: [] for arm in ARMS}
    norm = {arm: [] for arm in heads}
    norm.update({arm: [] for arm in ALIGNED})
    dots, normalized_dots, input_norms, bound_slacks = [], [], [], []
    for start in range(0, len(leaves), 512):
        batch = leaves[start:start + 512].float()
        raw = v.base.reduce_states(list(batch.unbind(1)), v.base.motor_product, "balanced")
        calibrated = cal(batch)
        identity = torch.zeros_like(raw)
        identity[:, 0] = 1
        for arm, prediction in (("identity", identity), ("raw", raw), ("calibrated", calibrated)):
            predictions[arm].append(prediction)
        raw_norm, cal_norm = raw[:, :4].norm(dim=-1), calibrated[:, :4].norm(dim=-1)
        assert torch.all(raw_norm > 0) and torch.all(cal_norm > 0)
        dot = (raw[:, :4] * calibrated[:, :4]).sum(-1)
        dots.append(dot)
        normalized_dots.append(dot / (raw_norm * cal_norm))
        input_norms.append(torch.stack((raw_norm, cal_norm), -1))
        lower_bound = torch.minimum(raw_norm, cal_norm) / np.sqrt(2)
        for arm, head in heads.items():
            before = head.before_projection(batch, raw, calibrated)
            norm[arm].append(before[:, :4].norm(dim=-1))
            predictions[arm].append(v.base.normalize_motor(before))
            if arm in ("contextual", "mergeable"):
                alpha = weight(head, batch, raw, calibrated, arm)
                assert torch.equal(before, raw + alpha * (calibrated - raw))
                aligned = aligned_blend(raw, calibrated, alpha)
                actual_norm = aligned[:, :4].norm(dim=-1)
                slack = actual_norm - lower_bound
                assert float(slack.min()) >= -2e-6
                bound_slacks.append(slack)
                key = arm + "_aligned"
                norm[key].append(actual_norm)
                predictions[key].append(v.base.normalize_motor(aligned))
    predictions = {a: torch.cat(x) for a, x in predictions.items()}
    norms = {a: torch.cat(x) for a, x in norm.items()}
    for prediction in predictions.values():
        assert torch.isfinite(prediction).all()
    cohorts = {}
    for cohort, mask in masks.items():
        indices = np.flatnonzero(mask)
        selected_names = [names[i] for i in indices]
        cohorts[cohort] = {a: study.metrics(p[indices], targets[indices].float(), selected_names)
                           for a, p in predictions.items()} if len(indices) else {}
    dots, normalized_dots, input_norms = torch.cat(dots), torch.cat(normalized_dots), torch.cat(input_norms)
    diagnostics = {
        "n_window_seed_cases": len(names),
        "negative_dot_cases": int((dots < 0).sum()),
        "exact_zero_dot_cases": int((dots == 0).sum()),
        "minimum_dot": float(dots.min()),
        "minimum_absolute_dot": float(dots.abs().min()),
        "minimum_normalized_dot": float(normalized_dots.min()),
        "minimum_input_real_norm": float(input_norms.min()),
        "maximum_input_real_norm_deviation_from_one": float((input_norms - 1).abs().max()),
        "minimum_alignment_bound_slack": float(torch.cat(bound_slacks).min()),
        "minimum_real_norm": {a: float(x.min()) for a, x in norms.items()},
        "below_arithmetic_clamp_cases": {a: int((x < 1e-8).sum()) for a, x in norms.items()},
    }
    return cohorts, diagnostics


@torch.inference_mode()
def numeric_audit(leaves, names, cal, heads):
    indices, seen = [], {}
    for i, name in enumerate(names):
        seen[name] = seen.get(name, 0) + 1
        if seen[name] <= 20:
            indices.append(i)
    cal = copy.deepcopy(cal).double()
    selected_heads = {a: copy.deepcopy(heads[a]).double() for a in ("contextual", "mergeable")}
    results = {a: {"translation_m": 0.0, "rotation_rad": 0.0} for a in ALIGNED}
    sign_changes = 0
    minimum_abs_dot = float("inf")
    length = leaves.shape[1]

    def product(x, path):
        states = list(x.unbind(1))
        if path == "chunks":
            cuts = (0, length // 3, 2 * length // 3, length)
            states = [v.base.reduce_states(states[a:b], v.base.motor_product, "left")
                      for a, b in zip(cuts[:-1], cuts[1:])]
            path = "balanced"
        return v.base.reduce_states(states, v.base.motor_product, path)

    for offset in range(0, len(indices), 256):
        batch = leaves[indices[offset:offset + 256]].double()
        calibrated_leaves = cal.calibrator(batch)
        original = {}
        reference_sign = None
        for path in ("left", "right", "balanced", "chunks"):
            raw, calibrated = product(batch, path), product(calibrated_leaves, path)
            dot = (raw[:, :4] * calibrated[:, :4]).sum(-1)
            minimum_abs_dot = min(minimum_abs_dot, float(dot.abs().min()))
            sign = dot < 0
            if reference_sign is None:
                reference_sign = sign
            else:
                sign_changes += int((sign != reference_sign).sum())
            for arm, head in selected_heads.items():
                before = aligned_blend(raw, calibrated, weight(head, batch, raw, calibrated, arm))
                q, t = v.base.decode_motor(before)
                if path == "left":
                    original[arm] = (q, t)
                    continue
                q0, t0 = original[arm]
                relative = v.base.qmul(v.base.qconj(q0), q)
                angle = 2 * torch.atan2(relative[:, 1:].norm(dim=-1), relative[:, 0].abs())
                result = results[arm + "_aligned"]
                result["translation_m"] = max(result["translation_m"], float((t - t0).norm(dim=-1).max()))
                result["rotation_rad"] = max(result["rotation_rad"], float(angle.max()))
    assert sign_changes == 0
    assert all(r["translation_m"] < 1e-8 and r["rotation_rad"] < 1e-9 for r in results.values())
    return {"n_window_seed_cases": len(indices), "paths": ["left", "right", "balanced", "chunks"],
            "minimum_absolute_dot": minimum_abs_dot, "sign_branch_changes": sign_changes,
            "max_path_discrepancy": results}


def aggregate(records):
    result = {"scope": "Post-hoc frozen-checkpoint sensitivity; conditional uncertainty, no model selection",
              "cohorts": {}, "readout_diagnostics": {}}
    for cohort in COHORTS:
        result["cohorts"][cohort] = {}
        for length in LENGTHS:
            selected = [r for r in records if r["length"] == length and r["cohorts"][cohort]]
            families = sorted({c["family"] for r in selected
                               for c in r["cohorts"][cohort]["mergeable"]["clusters"].values()})
            sequences = sorted({n for r in selected for n in r["cohorts"][cohort]["mergeable"]["clusters"]})
            item = {"families": families, "sequences": sequences,
                    "n_windows": sum(r["cohorts"][cohort]["mergeable"]["n_windows"]
                                     for r in selected if r["seed"] == 0), "metrics": {}}
            for metric in METRICS:
                arrays = {a: np.full((5, len(families)), np.nan) for a in ARMS}
                for r in selected:
                    for a in ARMS:
                        clusters = r["cohorts"][cohort][a]["clusters"]
                        for family in {c["family"] for c in clusters.values()}:
                            arrays[a][r["seed"], families.index(family)] = np.mean(
                                [c[metric] for c in clusters.values() if c["family"] == family])
                assert all(np.isfinite(a).all() for a in arrays.values())
                methods = {a: {"mean": float(x.mean()), "per_seed": x.mean(1).tolist(),
                               "per_family": dict(zip(families, x.mean(0).tolist()))}
                           for a, x in arrays.items()}
                contrasts = {}
                for lhs, rhs in (("mergeable", "identity"), ("mergeable", "feature_direct"),
                                 ("mergeable_aligned", "mergeable"), ("contextual_aligned", "contextual")):
                    delta = arrays[lhs] - arrays[rhs]
                    family_delta = delta.mean(0)
                    rng = np.random.default_rng(160926)
                    draws = family_delta[rng.integers(0, len(families), (10000, len(families)))].mean(1)
                    contrasts[lhs + "_minus_" + rhs] = {
                        "mean_delta": float(delta.mean()), "conditional_ci95": np.quantile(draws, [.025, .975]).tolist(),
                        "family_wins": int((family_delta < 0).sum())}
                item["metrics"][metric] = {"methods": methods, "contrasts": contrasts}
            result["cohorts"][cohort][str(length)] = item
    for length in LENGTHS:
        selected = [r for r in records if r["length"] == length]
        diagnostics = [r["readout_diagnostics"] for r in selected]
        numerical = [r["numerical_audit"] for r in selected]
        item = {key: sum(d[key] for d in diagnostics) for key in
                ("n_window_seed_cases", "negative_dot_cases", "exact_zero_dot_cases")}
        for key in ("minimum_dot", "minimum_absolute_dot", "minimum_normalized_dot",
                    "minimum_input_real_norm", "minimum_alignment_bound_slack"):
            item[key] = min(d[key] for d in diagnostics)
        item["maximum_input_real_norm_deviation_from_one"] = max(d["maximum_input_real_norm_deviation_from_one"] for d in diagnostics)
        item["minimum_real_norm"] = {a: min(d["minimum_real_norm"][a] for d in diagnostics)
                                     for a in diagnostics[0]["minimum_real_norm"]}
        item["below_arithmetic_clamp_cases"] = {a: sum(d["below_arithmetic_clamp_cases"][a] for d in diagnostics)
                                               for a in diagnostics[0]["minimum_real_norm"]}
        item["numerical_audit"] = {
            "n_window_seed_cases": sum(n["n_window_seed_cases"] for n in numerical),
            "sign_branch_changes": sum(n["sign_branch_changes"] for n in numerical),
            "minimum_absolute_dot": min(n["minimum_absolute_dot"] for n in numerical),
            "max_path_discrepancy": {a: {m: max(n["max_path_discrepancy"][a][m] for n in numerical)
                                         for m in ("translation_m", "rotation_rad")} for a in ALIGNED}}
        result["readout_diagnostics"][str(length)] = item
    result["maximum_original_reproduction_difference"] = max(r["maximum_original_reproduction_difference"] for r in records)
    return result


def main():
    torch.set_num_threads(1)
    (HERE / "runs").mkdir(exist_ok=True)
    initial = dependencies()
    seal = HERE / "seal.json"
    if not seal.exists():
        save(seal, initial)
    assert json.loads(seal.read_text()) == initial, "sealed input changed"
    data = v.ETH3DAll()
    common_sequences = {name for name, runs in data.runs.items() if any(len(x) >= 128 for x, _ in runs)}
    loaded = {length: windows(data, length) for length in LENGTHS}
    common_ids = {tuple(i) for i in loaded[128][3]}
    assert len(common_sequences) == 23 and len(common_ids) == 3435
    for length, (_, _, names, ids, mask) in loaded.items():
        assert {i for i, keep in zip(ids, mask) if keep} == common_ids
        assert {n for n in names if n in common_sequences} == common_sequences
    records = []
    for fold in range(5):
        for seed in range(5):
            stem = f"fold{fold}_seed{seed}"
            source, merge_path = study.HERE / "runs" / (stem + ".pt"), experiment.HERE / "runs" / (stem + ".pt")
            source_record = json.loads(source.with_suffix(".json").read_text())
            merge_record = json.loads(merge_path.with_suffix(".json").read_text())
            assert sha(source) == source_record["checkpoint_sha256"] == merge_record["source_checkpoint_sha256"]
            assert sha(merge_path) == merge_record["checkpoint_sha256"]
            saved = torch.load(source, weights_only=True)
            cal = v.CalibratedMotor().eval()
            cal.load_state_dict(saved["calibrator"])
            heads = {a: study.Head(a).eval() for a in study.ARMS}
            for a, head in heads.items():
                head.load_state_dict(saved[a])
            heads["mergeable"] = experiment.MergeableGate().eval()
            heads["mergeable"].load_state_dict(torch.load(merge_path, weights_only=True))
            for length in (32, 4, 8, 16, 64, 128):
                destination = HERE / "runs" / f"{stem}_L{length}.json"
                if destination.exists():
                    record = json.loads(destination.read_text())
                    assert record["seal_sha256"] == sha(seal)
                else:
                    leaves, targets, names, ids, matched = loaded[length]
                    indices = [i for i, n in enumerate(names) if v.FAMILY_FOR[n] in v.FOLD_FAMILIES[fold]]
                    selected_names = [names[i] for i in indices]
                    masks = {"full": np.ones(len(indices), dtype=bool),
                             "common_sequences": np.asarray([n in common_sequences for n in selected_names]),
                             "matched_starts": matched[indices]}
                    cohorts, diagnostics = evaluate(leaves[indices], targets[indices], selected_names, masks, cal, heads)
                    old = json.loads((scope.HERE / "runs" / f"{stem}_L{length}.json").read_text())
                    reproduction = max(check_against_record(cohorts["full"][a], old["arms"][a]) for a in ORIGINAL)
                    numeric = numeric_audit(leaves[indices], selected_names, cal, heads)
                    record = {"fold": fold, "seed": seed, "length": length, "seal_sha256": sha(seal),
                              "window_id_sha256": hashlib.sha256(json.dumps([ids[i] for i in indices]).encode()).hexdigest(),
                              "matched_start_id_sha256": hashlib.sha256(json.dumps([ids[i] for i in indices if matched[i]]).encode()).hexdigest(),
                              "maximum_original_reproduction_difference": reproduction, "cohorts": cohorts,
                              "readout_diagnostics": diagnostics, "numerical_audit": numeric}
                    save(destination, record)
                records.append(record)
            print(f"completed fold={fold} seed={seed}", flush=True)
    assert dependencies() == initial, "input changed during evaluation"
    summary = aggregate(records)
    summary.update(seal_sha256=sha(seal), common_sequences=sorted(common_sequences),
                   matched_start_count=len(common_ids), n_records=len(records),
                   environment={"python": platform.python_version(), "torch": torch.__version__,
                                "numpy": np.__version__, "platform": platform.platform(), "threads": 1},
                   record_sha256={str(p.relative_to(HERE)): sha(p) for p in sorted((HERE / "runs").glob("*.json"))})
    destination = HERE / "analysis.json"
    if destination.exists():
        assert json.loads(destination.read_text()) == summary
    else:
        save(destination, summary)
    print("PASS: original arms reproduced; all records retained; inputs unchanged", flush=True)


if __name__ == "__main__":
    main()
