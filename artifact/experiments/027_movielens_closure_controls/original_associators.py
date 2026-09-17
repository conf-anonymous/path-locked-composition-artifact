"""Support diagnostics on immutable original checkpoints and recorded inputs."""
import json

import numpy as np
import torch

import run as study


@torch.inference_mode()
def main():
    torch.set_num_threads(1)
    data = study.m.MovieLens()
    records = []
    for arm in ("bilinear", "penalty", "mixed", "exact"):
        for seed in range(5):
            path = study.m.HERE / "runs" / f"{arm}_seed{seed}.pt"
            model = study.m.Composer(arm, data.n_movies)
            model.load_state_dict(torch.load(path, weights_only=True))
            model.eval()
            record = {"arm": arm, "seed": seed, "checkpoint_sha256": study.sha(path), "lengths": {}}
            for length in (8, 16):
                examples = data.examples[("test", length)]
                values = []
                for start in range(0, len(examples), 1024):
                    movies, ratings, _, _ = data.batch(examples, length, range(start, min(start+1024, len(examples))))
                    states = model.encode(movies, ratings)
                    values.append((len(movies), float(study.regularizer(model, states, "primitive_mse")),
                                   float(study.regularizer(model, states, "intermediate_mse"))))
                values = np.asarray(values)
                primitive, multilevel = np.average(values[:, 1:], weights=values[:, 0], axis=0)
                nlevels = 3 if length == 8 else 4
                record["lengths"][str(length)] = {"n": len(examples), "primitive_mse": float(primitive),
                    "multilevel_mse": float(multilevel),
                    "internal_only_mse": float((nlevels*multilevel-primitive)/(nlevels-1))}
            records.append(record)
    study.save(study.HERE / "original_associators.json", {"status": "post-hoc frozen-checkpoint support diagnostic", "records": records})
    for arm in ("bilinear", "penalty", "mixed", "exact"):
        rows = [r["lengths"]["8"] for r in records if r["arm"] == arm]
        print(arm, {k: float(np.mean([r[k] for r in rows])) for k in ("primitive_mse", "internal_only_mse")})


if __name__ == "__main__":
    main()
