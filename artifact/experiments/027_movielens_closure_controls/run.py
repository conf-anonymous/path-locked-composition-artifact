"""Post-hoc controls on unchanged public recorded MovieLens histories."""
from __future__ import annotations

import concurrent.futures
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
spec = importlib.util.spec_from_file_location("ml027", HERE.parent / "016_movielens_composition/run.py")
m = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = m
spec.loader.exec_module(m)
KINDS = ("primitive_mse", "intermediate_mse", "intermediate_cosine")
WEIGHTS = (.1, 1., 10.)
PATHS = ("left", "right", "balanced", "history_right", "history_balanced")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2)
        stream.write("\n")


def regularizer(model, states, kind):
    levels = []
    while len(states) >= 3:
        # Concatenate triples along batch, so every consecutive triple contributes.
        a = torch.cat(states[:-2])
        b = torch.cat(states[1:-1])
        c = torch.cat(states[2:])
        lhs, rhs = model.op(model.op(a, b), c), model.op(a, model.op(b, c))
        value = ((1 - nn.functional.cosine_similarity(lhs, rhs)) ** 2).mean() \
            if kind.endswith("cosine") else (lhs - rhs).square().mean()
        levels.append(value)
        if kind.startswith("primitive"):
            break
        states = [model.op(states[i], states[i + 1]) for i in range(0, len(states)-1, 2)] \
            + ([states[-1]] if len(states) % 2 else [])
    return torch.stack(levels).mean()


@torch.inference_mode()
def evaluate(model, data, split, length):
    examples = data.examples[(split, length)]
    hits = {p: [] for p in PATHS}
    users, assoc = [], []
    for start in range(0, len(examples), 1024):
        movies, ratings, labels, user = data.batch(examples, length,
            range(start, min(start + 1024, len(examples))))
        states = model.encode(movies, ratings)
        for path in PATHS:
            rep = model.op(model.reduce(states[:-1], path[8:]), states[-1]) \
                if path.startswith("history_") else model.reduce(states, path)
            hit = model.head(model.final_norm(rep)).argmax(-1) == labels
            hits[path].append(hit.numpy().astype(float))
        # MSE diagnostics measure primitive support separately from all multilevel support.
        assoc.append((len(labels), float(regularizer(model, states, "primitive_mse")),
                      float(regularizer(model, states, "intermediate_mse"))))
        users.append(user.numpy())
    hits = {p: np.concatenate(v) for p, v in hits.items()}
    users = np.concatenate(users)
    values = np.asarray(assoc)
    result = {"n": len(examples), "users": len(set(users)),
              "accuracy": {p: float(v.mean()) for p, v in hits.items()},
              "primitive_mse": float(np.average(values[:, 1], weights=values[:, 0])),
              "multilevel_mse": float(np.average(values[:, 2], weights=values[:, 0]))}
    if split == "test":
        result["delta_ci95_conditional_user_bootstrap"] = {
            p: m.user_clustered_delta_ci(hits["left"], hits[p], users, 270906)
            for p in PATHS[1:]}
    return result


def fit_job(job):
    kind, weight, seed = job
    torch.set_num_threads(1)
    data, cfg = m.MovieLens(), m.Config()
    stem = f"{kind}_w{weight:g}_seed{seed}"
    record = HERE / "runs" / f"{stem}.json"
    checkpoint = record.with_suffix(".pt")
    if record.exists():
        return json.loads(record.read_text())
    torch.manual_seed(seed)
    model = m.Composer("bilinear", data.n_movies)
    generator = torch.Generator().manual_seed(seed + 10000)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    started = time.perf_counter()
    for step in range(cfg.steps):
        idx = torch.randint(len(data.train_examples), (cfg.batch_size,), generator=generator)
        movies, ratings, labels, _ = data.batch(data.train_examples, 8, idx)
        states = model.encode(movies, ratings)
        logits = model.head(model.final_norm(model.reduce(states, "left")))
        loss = nn.functional.cross_entropy(logits, labels) + weight * regularizer(model, states, kind)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()
    seconds = time.perf_counter() - started
    model.eval()
    result = {"kind": kind, "weight": weight, "seed": seed, "training_seconds": seconds,
              "dev": evaluate(model, data, "dev", 8)}
    torch.save(model.state_dict(), checkpoint)
    result["checkpoint_sha256"] = sha(checkpoint)
    save(record, result)
    print(f"finished {stem}, dev={result['dev']['accuracy']}, {seconds:.1f}s", flush=True)
    return result


def history_baselines(data):
    counts = np.ones((data.n_movies, 5), dtype=float)
    for movies, ratings in data.sequences["train"].values():
        np.add.at(counts, (movies, ratings - 1), 1)
    probs = counts / counts.sum(1, keepdims=True)
    def evaluate_mix(split, weights):
        hits = {str(w): [] for w in weights}
        history_hits = []
        for user, endpoint in data.examples[(split, 8)]:
            movies, ratings = data.sequences[split][user]
            hist = np.bincount(ratings[endpoint-8:endpoint]-1, minlength=5).astype(float)
            truth = ratings[endpoint] - 1
            history_hits.append(int(hist.argmax() == truth))
            hist /= hist.sum()
            for w in weights:
                hits[str(w)].append(int(((1-w)*probs[movies[endpoint]] + w*hist).argmax() == truth))
        return {k: float(np.mean(v)) for k, v in hits.items()}, float(np.mean(history_hits))
    dev, _ = evaluate_mix("dev", (0., .25, .5, .75, 1.))
    selected = min(dev, key=lambda w: (-dev[w], float(w)))
    # Selection recorded before the test labels are used.
    save(HERE / "baseline_selection.json", {"dev_accuracy": dev, "weight": float(selected)})
    test, history = evaluate_mix("test", (float(selected),))
    return {"selected_history_weight": float(selected), "dev_accuracy": dev,
            "test_mixture_accuracy": test[selected], "test_history_mode_accuracy": history}


def main():
    torch.set_num_threads(1)
    (HERE / "runs").mkdir(exist_ok=True)
    hashes = {str(p.relative_to(ROOT)): sha(p) for p in (Path(__file__), HERE / "PROTOCOL.md", Path(m.__file__))}
    seal = HERE / "seal.json"
    if not seal.exists():
        save(seal, hashes)
    if json.loads(seal.read_text()) != hashes:
        raise RuntimeError("source changed after seal; do not silently resume")
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:
        candidates = list(pool.map(fit_job, [(k, w, 0) for k in KINDS for w in WEIGHTS]))
        selected = {}
        for kind in KINDS:
            choices = [r for r in candidates if r["kind"] == kind]
            best = min(choices, key=lambda r: (-np.mean(list(r["dev"]["accuracy"].values())), r["weight"]))
            selected[kind] = best["weight"]
        if not (HERE / "selection.json").exists():
            save(HERE / "selection.json", {"selected": selected, "candidates": candidates})
        records = list(pool.map(fit_job, [(k, w, s) for k, w in selected.items() for s in range(5)]))
    data = m.MovieLens()
    for r in records:
        stem = f"{r['kind']}_w{r['weight']:g}_seed{r['seed']}"
        model = m.Composer("bilinear", data.n_movies)
        checkpoint = HERE / "runs" / f"{stem}.pt"
        assert sha(checkpoint) == r["checkpoint_sha256"]
        model.load_state_dict(torch.load(checkpoint, weights_only=True))
        model.eval()
        r["test"] = {str(length): evaluate(model, data, "test", length) for length in (8, 16)}
        print(f"evaluated {stem}", flush=True)
    save(HERE / "results.json", {"status": "post-hoc; existing test outcomes known", "seal": hashes,
         "selection": selected, "records": records, "history_baselines": history_baselines(data)})


if __name__ == "__main__":
    main()
