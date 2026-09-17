"""Read-only, post-hoc adversarial diagnostics; emits JSON to stdout.

Uses existing public recordings and frozen checkpoints. No training, synthetic
observations, model selection, or modification of original evidence. Run from
the repository root with .venv/bin/python iclr-2027-composition/review_diagnostics.py.
These diagnostics do not amend or replace the frozen experimental protocols.
"""

import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def movielens():
    m = load('review_ml', 'experiments/016_movielens_composition/run.py')
    data = m.MovieLens()
    examples = data.examples[('test', 8)]
    output = {'n_endpoints': len(examples), 'n_users': len(data.sequences['test']),
              'history_length': 8, 'arms': {}, 'checkpoint_sha256': {}}
    paths = ('left', 'right', 'history_right_query_last', 'history_balanced_query_last')
    with torch.inference_mode():
        for arm in m.ARMS:
            seeds = []
            for seed in range(5):
                checkpoint = m.HERE / 'runs' / f'{arm}_seed{seed}.pt'
                output['checkpoint_sha256'][checkpoint.name] = digest(checkpoint)
                model = m.Composer(arm, data.n_movies)
                model.load_state_dict(torch.load(checkpoint, weights_only=True, map_location='cpu'))
                model.eval()
                correct = {path: [] for path in paths}
                user_chunks = []
                for start in range(0, len(examples), 1024):
                    movies, ratings, y, users = data.batch(
                        examples, 8, range(start, min(start + 1024, len(examples))))
                    states = model.encode(movies, ratings)
                    user_chunks.append(users.numpy())
                    for path in paths:
                        if path.startswith('history_'):
                            hp = 'right' if path == 'history_right_query_last' else 'balanced'
                            rep = model.op(model.reduce(states[:-1], hp), states[-1])
                        else:
                            rep = model.reduce(states, path)
                        prediction = model.head(model.final_norm(rep)).argmax(-1)
                        correct[path].append((prediction == y).numpy().astype(float))
                correct = {k: np.concatenate(v) for k, v in correct.items()}
                users = np.concatenate(user_chunks)
                seeds.append({
                    'seed': seed,
                    'accuracy': {k: float(v.mean()) for k, v in correct.items()},
                    'history_only_delta_ci95': {
                        path: m.user_clustered_delta_ci(
                            correct['left'], correct[path], users, seed=960906 + seed)
                        for path in paths[2:]
                    },
                })
            output['arms'][arm] = {
                'mean_accuracy': {k: float(np.mean([s['accuracy'][k] for s in seeds])) for k in paths},
                'seeds': seeds,
            }
    return output


def eth3d():
    m = load('review_eth3d', 'experiments/023_eth3d_family_heldout/run.py')
    data = m.ETH3DAll()  # Verifies hashes of recorded derived odometry.
    _, translation = m.base.decode_motor(data.tensors[32][1])
    distance = translation.norm(dim=-1).numpy()
    names = np.asarray(data.examples[32])
    sequence_means = {n: float(distance[names == n].mean()) for n in sorted(set(names))}
    family_means = {
        f: float(np.mean([v for n, v in sequence_means.items() if m.FAMILY_FOR[n] == f]))
        for f in m.FAMILY_SEQUENCES
    }
    source = json.loads((m.HERE / 'results.json').read_text())
    control = json.loads((ROOT / 'experiments/024_eth3d_calibrated_direct_control/results.json').read_text())
    sums, folds = [], {}
    for fk, fold in source['folds'].items():
        values = []
        for family in fold['families']:
            deltas = []
            for sk, seed in fold['seeds'].items():
                contextual = seed['lengths']['32']['contextual_ga']['clusters']
                direct = control['folds'][fk]['seeds'][sk]['lengths']['32']['clusters']
                def mean(clusters):
                    return np.mean([v['translation_mean_m'] for v in clusters.values()
                                    if v['family'] == family])
                deltas.append(mean(contextual) - mean(direct))
            values.append(float(np.mean(deltas)))
        sums.append(sum(values))
        folds[fk] = {'families': fold['families'], 'mean_delta_m': float(np.mean(values))}
    sums = np.asarray(sums)
    distribution = np.asarray([np.dot(sums, signs) / 16
                               for signs in itertools.product((-1, 1), repeat=5)])
    return {
        'n_windows': len(distance),
        'identity_prediction_family_equal_translation_m': float(np.mean(list(family_means.values()))),
        'identity_prediction_per_family_translation_m': family_means,
        'fraction_windows_displacement_at_most_0_15m': float(np.mean(distance <= .15)),
        'folds': folds,
        'fold_block_sign_flip_two_sided_p': float(np.mean(np.abs(distribution) >= abs(sums.sum() / 16) - 1e-15)),
        'fold_block_assignments': len(distribution),
        'statistical_caution': 'Sensitivity only: five folds are few and their training sets overlap. '
                               'Neither this nor family-wise enumeration provides unconditional exact inference.',
    }


if __name__ == '__main__':
    torch.set_num_threads(1)
    sources = (
        'iclr-2027-composition/paper/main.tex',
        'iclr-2027-composition/paper/main.pdf',
        'experiments/016_movielens_composition/run.py',
        'experiments/023_eth3d_family_heldout/run.py',
        'experiments/023_eth3d_family_heldout/results.json',
        'experiments/024_eth3d_calibrated_direct_control/results.json',
    )
    result = {'status': 'post-hoc adversarial audit; not confirmatory evidence',
              'torch_version': torch.__version__, 'numpy_version': np.__version__,
              'source_sha256': {p: digest(ROOT / p) for p in sources},
              'movielens': movielens(), 'eth3d': eth3d()}
    print(json.dumps(result, indent=2))
