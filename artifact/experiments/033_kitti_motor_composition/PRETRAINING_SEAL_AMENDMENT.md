# Pretraining implementation-seal correction

2026-09-11. Before any optimizer update, development access or holdout access,
the first implementation seal was found to enumerate already-imported Python
package files inside the repository's `.venv`. That list can grow when PyTorch
lazily imports code during optimization, incorrectly treating runtime
initialization as a research-source change.

Preserved unchanged under
`data/raw/kitti_odometry/learned_composition_v1/pretraining_seal_amendment_1/`:
the original `implementation_seal.json` and its `workflow.py`. The first seal
recorded 30 passing tests. It is superseded, not destroyed or used for any fit.

Correction: enumerate actual imported research-source dependencies under
`experiments/`, plus all experiment-033 Python/C++ sources and specified
instructions; pin Python, PyTorch, NumPy and platform versions separately.
Retain the complete raw/derived training input hashes and existing frontend
and scientific seals. Test fingerprint stability across a real-data model
forward/backward initialization before replacing the implementation seal.

No scientific threshold, arm, seed, data split, loss, training budget, checkpoint
selection or holdout-release accuracy criterion changes. No data outcome
motivates this correction. All preserved frontend results remain unchanged.
