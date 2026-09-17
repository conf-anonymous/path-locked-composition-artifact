"""Final common grid with three development seeds, preserving prior searches."""
import concurrent.futures
import json
from pathlib import Path

import numpy as np
import torch

import run as study

WEIGHTS = (.1, 1., 10., 100., 1000., 10000.)


def main():
    torch.set_num_threads(1)
    study.save(study.HERE / "robust_seal.json", {
        p.name: study.sha(p) for p in (Path(__file__), study.HERE / "ROBUST_SELECTION.md")})
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:
        candidates = list(pool.map(study.fit_job,
            [(k, w, s) for k in study.KINDS for w in WEIGHTS for s in range(3)]))
        selected, scores = {}, {}
        for kind in study.KINDS:
            scores[kind] = {str(w): float(np.mean([list(r["dev"]["accuracy"].values())
                for r in candidates if r["kind"] == kind and r["weight"] == w])) for w in WEIGHTS}
            selected[kind] = float(min(scores[kind], key=lambda w: (-scores[kind][w], float(w))))
        study.save(study.HERE / "robust_selection.json", {"selected": selected, "scores": scores,
                                                        "candidates": candidates})
        records = list(pool.map(study.fit_job,
            [(kind, weight, seed) for kind, weight in selected.items() for seed in range(5)]))
    previous = json.loads((study.HERE / "results.json").read_text())["records"] \
        + json.loads((study.HERE / "boundary_results.json").read_text())["records"]
    data = study.m.MovieLens()
    for record in records:
        same = next((r for r in previous if all(r[k] == record[k] for k in ("kind", "weight", "seed"))), None)
        if same is not None:
            record["test"] = same["test"]
        else:
            model = study.m.Composer("bilinear", data.n_movies)
            path = study.HERE / "runs" / f"{record['kind']}_w{record['weight']:g}_seed{record['seed']}.pt"
            assert study.sha(path) == record["checkpoint_sha256"]
            model.load_state_dict(torch.load(path, weights_only=True))
            model.eval()
            record["test"] = {str(length): study.evaluate(model, data, "test", length) for length in (8, 16)}
    run_records = [json.loads(p.read_text()) for p in (study.HERE / "runs").glob("*.json")]
    study.save(study.HERE / "robust_results.json", {"selection": selected, "records": records,
        "n_distinct_fits": len(run_records), "training_seconds_total": sum(r["training_seconds"] for r in run_records),
        "status": "final fixed grid, three dev selection seeds; already observed public test set"})
    print("Final development selections:", selected, flush=True)


if __name__ == "__main__":
    main()
