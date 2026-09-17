"""Development-only boundary expansion; original result stays immutable."""
import concurrent.futures
import json
from pathlib import Path

import numpy as np
import torch

import run as study


def expand(kind):
    candidates = [json.loads(p.read_text()) for p in (study.HERE / "runs").glob(f"{kind}_w*_seed0.json")]
    candidates = [r for r in candidates if r["weight"] <= 10]
    score = lambda r: np.mean(list(r["dev"]["accuracy"].values()))
    best = min(candidates, key=lambda r: (-score(r), r["weight"]))
    for weight in (100., 1000., 10000.):
        candidate = study.fit_job((kind, weight, 0))
        candidates.append(candidate)
        if score(candidate) <= score(best):
            break
        best = candidate
    return {"kind": kind, "selected": best["weight"], "candidates": candidates}


def main():
    torch.set_num_threads(1)
    study.save(study.HERE / "boundary_seal.json", {
        p.name: study.sha(p) for p in (Path(__file__), study.HERE / "BOUNDARY_CHECK.md")})
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:
        selections = list(pool.map(expand, study.KINDS))
        study.save(study.HERE / "boundary_selection.json", selections)
        jobs = [(s["kind"], s["selected"], seed) for s in selections for seed in range(5)]
        records = list(pool.map(study.fit_job, jobs))
    old = json.loads((study.HERE / "results.json").read_text())
    data = study.m.MovieLens()
    for record in records:
        same = next((r for r in old["records"] if all(r[k] == record[k] for k in ("kind", "weight", "seed"))), None)
        if same is not None:
            record["test"] = same["test"]
        else:
            model = study.m.Composer("bilinear", data.n_movies)
            path = study.HERE / "runs" / f"{record['kind']}_w{record['weight']:g}_seed{record['seed']}.pt"
            assert study.sha(path) == record["checkpoint_sha256"]
            model.load_state_dict(torch.load(path, weights_only=True))
            model.eval()
            record["test"] = {str(length): study.evaluate(model, data, "test", length) for length in (8, 16)}
    study.save(study.HERE / "boundary_results.json", {"selection": selections, "records": records,
        "original_result_sha256": study.sha(study.HERE / "results.json"),
        "status": "development-boundary post-hoc extension; original outcomes known"})
    print("Boundary selections:", {r["kind"]: r["selected"] for r in selections})


if __name__ == "__main__":
    main()
