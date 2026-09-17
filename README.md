# Path-Locked Neural Composition

Anonymous research artifact for **Path-Locked Neural Composition: Exact
Evaluation-Path Invariance with Geometric Algebra**.

The artifact contains the geometric-algebra motor implementations, learned
composition models, protocols, frozen checkpoints, per-run results and checks
supporting the manuscript. It includes the paired-motor mergeable state and
ETH3D controls, the MovieLens/TUM path-locking experiments, all KITTI comparison
arms and horizons, the frozen ETH3D readout/matched-start sensitivity audit,
and the excluded Oxford diagnostic records. No original
dataset archives or recordings are redistributed.

## Quickstart

The integrity checker uses Python 3.11+ and only the standard library:

```sh
python3 verify.py
python3 -m unittest discover -s tests -v
```

To check the retained experimental evidence, create an environment with the
recorded dependencies. The recorded neural environment was Python 3.14.7 on
macOS ARM64; the requirements pin its NumPy and PyTorch versions.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-audit.txt
.venv/bin/python reproduce.py audit
```

The final command checks the complete file inventory, runs all retained-evidence
audits, recomputes KITTI aggregates from all 125,595 per-window/seed records,
and verifies that the reference files have not changed. It keeps its log and
receipt in `verification-output/`. No dataset download, model fitting, external
service call or confirmation-set access is part of this command.
The audit also independently recomputes Experiment 040 aggregates and
conditional bootstrap intervals from all 150 retained records.

Passing this audit means the retained evidence and its arithmetic are internally
consistent. It does **not** mean every experimental success criterion passes:
the original ETH3D development failure, KITTI primary translation failure and
Oxford exclusion must remain intact. It is not a fresh training or sensor replay.

## Reproducing the computations

[REPRODUCING.md](docs/REPRODUCING.md) separates three levels:

1. Integrity and stored-result verification, requiring no original dataset.
2. Evaluation from frozen checkpoints, after separately obtaining public data.
3. Fresh extraction and training, in a new working directory with new receipts.

[DATA_AND_ENVIRONMENTS.md](docs/DATA_AND_ENVIRONMENTS.md) gives dataset sources,
dependency groups, platform constraints and external-input boundaries.
[CLAIMS.md](docs/CLAIMS.md) maps the manuscript's principal results to their
implementations and retained evidence.
[READOUT_SCOPE_AUDIT.md](docs/READOUT_SCOPE_AUDIT.md) documents the later
frozen-checkpoint sensitivity analyses, their replay boundary and all outcomes.

The manuscript is submitted separately. This repository does not include the
paper PDF, LaTeX sources, bibliography, figure assets or publication templates.

## Contents

- `artifact/experiments/016_...` through `033_...`: scientific code, protocols,
  retained checkpoints, outcomes and experiment-specific audits.
- `artifact/experiments/040_eth3d_readout_scope_audit/`: a separate post-hoc
  evaluation using the unchanged models and real recordings, plus independent
  arithmetic verification. No checkpoint or earlier result was replaced.
- `artifact/data/raw/`: provenance metadata and derived evaluation records,
  **not** the original public datasets. The name reflects the historical source
  tree needed by the unmodified scientific scripts.
- `artifact/iclr-2027-composition/`: the retained diagnostic helpers and Oxford
  matrix-audit reference files used by the existing relative import paths.
- `verify.py`, `reproduce.py`, `tests/`: packaging checks and safe entry points.
- `PACKAGE_MANIFEST.json`: exact delivered file names, sizes, executable modes
  and SHA-256 hashes. It excludes itself and documented local execution caches.

No other paper, parent Git history, account configuration, private planning
record or credential is included. All historical scientific sources, checkpoints and
experimental records retain their original archived bytes. New documentation
supersedes historical progress statements, but does not rewrite their evidence.

## Scope

The associativity claim concerns ordered composition and its specified readout,
not permutation invariance or bitwise equality in floating point. ETH3D benefits
are dataset- and comparator-specific. KITTI is not a positive primary translation
result or an official leaderboard submission. Oxford is diagnostic only; its
original confirmation remains closed. Experiments 034-039 and the separate
recall, world-model and qualification papers are not part of this artifact.

See [NOTICE.md](NOTICE.md) for third-party and dataset notices and
[ANONYMITY.md](docs/ANONYMITY.md) for the export boundary. The repository
has no telemetry or visitor-tracking code. Any hosting service has its own
policies; the integrity checker makes no network requests.
