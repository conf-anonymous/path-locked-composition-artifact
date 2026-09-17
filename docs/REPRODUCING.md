# Reproduction guide

Keep the delivered repository unchanged. Its checksums identify retained
reference evidence; they are not expected to identify newly generated runs.
All examples below use only original public observations and labels.

## 1. Verify the retained evidence

From the repository root:

```sh
python3 verify.py
python3 -m unittest discover -s tests -v
.venv/bin/python reproduce.py audit
```

Install `requirements-audit.txt` first for the third command. The first two use
only the standard library. The audit runs 15 inherited positive consistency
checks, verifies the expected original ETH3D gate failure, recomputes all KITTI
stored-window aggregates and checks the additional Oxford diagnostic records.
It requires no original dataset and performs no fresh training or sensor
extraction. Logs and a new run receipt go to `verification-output/`.

The audit distinguishes absence of external data from verified included bytes.
In particular, KITTI's 62 external inputs remain explicitly unverified in this
no-data mode. A missing or corrupted included source, checkpoint or result is
an error, not an allowed omission.

## 2. Create a separate working directory

To copy the frozen artifact for checkpoint evaluation:

```sh
python3 reproduce.py workspace --destination /path/to/new-checkpoint-replay
```

To start fresh extraction and fitting without historical outputs or seals:

```sh
python3 reproduce.py workspace --sources-only --destination /path/to/new-training-run
```

The destination must not exist and must be outside the repository. The command
only copies files. It does not download data, change a scientific source, run
an experiment or remove an old output. `--sources-only` copies Python/C/C++
code, protocol documentation and project metadata, not JSON outcomes, data,
historical seals or checkpoints. Acquisition scripts create fresh input
manifests; training/evaluation scripts create fresh outputs in their documented
order. A full-data replay is a new run, not new untouched confirmation.

Set up the appropriate environments using the repository's pinned requirement
files, then run the commands below from the new workspace. When using a full
checkpoint copy, some producers skip existing results and others refuse to
overwrite them. Preserve and move the **target experiment's generated result
JSON** aside in that copy before requesting a new evaluation. Keep its frozen
checkpoints and input dependencies. A skipped completed record is not a fresh
model evaluation. Never modify the delivered reference repository to make room.

## 3. MovieLens and TUM

Acquire MovieLens-1M as described in `DATA_AND_ENVIRONMENTS.md`. From a new
workspace, Experiment 016 develops the five-seed models, then evaluates the
fixed models on confirmation users:

```sh
python experiments/016_movielens_composition/run.py --seeds 0 1 2 3 4 --steps 3000 --batch-size 256
python experiments/016_movielens_composition/run.py --seeds 0 1 2 3 4 --steps 3000 --batch-size 256 --confirm
```

The presence of checkpoints suppresses retraining. For the post-hoc controls,
use Experiment 027 in the retained order: `run.py`, `boundary_check.py`, then
`robust_selection.py`. The final development selection uses three seeds; the
earlier search records are not interchangeable confirmation results. Use
`original_associators.py` for the preserved original-model diagnostic and the
query-last helpers documented in `README_PRE_CONSOLIDATION.md`.

For TUM:

```sh
python experiments/017_tum_motor_composition/download_data.py
python experiments/017_tum_motor_composition/run.py
python experiments/018_tum_residual_motor/run.py
python experiments/019_tum_long_horizon_confirmation/run.py
```

Experiments 017 and 018 retain failed earlier development gates. Experiment 019
is the explicitly frozen long-horizon confirmation protocol using Experiment
018 checkpoints. Do not replace it with a retrospective relaxation of an older
`--confirm` gate. Read all three `PROTOCOL.md` files before a fresh campaign.

## 4. ETH3D

Use the separate Python 3.12/Open3D environment for acquisition and extraction:

```sh
python experiments/021_eth3d_rgbd_motor/download_data.py
python experiments/021_eth3d_rgbd_motor/extract_odometry.py
```

Then select the neural environment for the research scripts. The main sequence
is Experiment 021 development, 022 contextual development, the separate 022
`run_confirmation.py`, 023 grouped family evaluation, 024 calibrated direct
control, 025 profile and 026 heterogeneity analysis. Their local protocols and
READMEs fix configurations and gates. Preserve the original 021 failed outcome;
do not open its failed-gate confirmation route merely to complete a command list.

The later information-matched and fully mergeable studies use:

```sh
python experiments/028_eth3d_attribution_controls/run.py
python experiments/028_eth3d_attribution_controls/check_cached_inference.py
python experiments/028_eth3d_attribution_controls/cache_deployment.py
python experiments/028_eth3d_attribution_controls/analyze.py
python experiments/029_eth3d_mergeable_motor_gate/run.py
python experiments/029_eth3d_mergeable_motor_gate/analyze.py
python experiments/029_eth3d_mergeable_motor_gate/cpu_profile.py
python experiments/030_eth3d_frozen_scope/run.py
```

Experiment 028 requires the completed 023 models and records. Experiment 029
requires 028; Experiment 030 reads the fixed earlier checkpoints. Remove no
upstream dependency in a checkpoint replay. Newly trained dependencies require
new downstream seals in the fresh workspace, not altered historical seals.
Run profiling without concurrent training. Expected wall time and numerical
agreement depend on the machine; neither bitwise equality nor the published
speed ratio is promised on another platform.

## 5. KITTI

The complete source pipeline is retained. Use a **sources-only** workspace,
the original four KITTI archives, `libviso2.zip`, the KITTI Python requirements,
and the documented macOS/Rosetta/compiler environment. The image archive is
about 23.17 GB; acquisition scans it without bulk image extraction.

```sh
python experiments/033_kitti_motor_composition/acquire.py --downloads /path/to/official-downloads
python experiments/033_kitti_motor_composition/check_native_reference.py
python experiments/033_kitti_motor_composition/build_frontend.py /path/to/official-downloads/libviso2.zip
python experiments/033_kitti_motor_composition/frontend_smoke.py
python experiments/033_kitti_motor_composition/check_smoke_metrics.py
python experiments/033_kitti_motor_composition/full_frontend.py
python experiments/033_kitti_motor_composition/preflight.py
python experiments/033_kitti_motor_composition/workflow.py seal
python experiments/033_kitti_motor_composition/workflow.py train
python experiments/033_kitti_motor_composition/evaluate.py train
python experiments/033_kitti_motor_composition/stage_evaluation.py dev
python experiments/033_kitti_motor_composition/evaluate.py dev
python experiments/033_kitti_motor_composition/workflow.py release-holdout
python experiments/033_kitti_motor_composition/stage_evaluation.py study_holdout
python experiments/033_kitti_motor_composition/evaluate.py study_holdout
```

Run one command at a time and stop at any error or failed gate. Read the
preparation, frontend, experiment protocols and `IMPLEMENTATION_NOTES.md` first.
The technical-only holdout gate does not require a favorable development gain.
The old `test_*.py` suite is stage-specific: some tests require that development
does not yet exist and are not valid all-stage assertions after its release.

Original build/input hashes will differ for a new machine or acquisition path.
The newly generated seals bind that new run. Compare numerical outcomes and
retain its execution records; do not relabel it as the historical campaign.
The published primary KITTI translation result remains unfavorable, regardless
of any new reproduction's outcome. `KITTI_REPRODUCTION.md` documents the original
external-input boundary in detail.

## 6. Oxford diagnostics

Oxford is retained to explain exclusion, not to establish a positive result.
Use the no-data audit or the development-only matrix replay in `REPLAY.md`.
Do not run its old training/release/confirmation workflow from this export.
Its original admission checker intentionally fails closed for a disclosed,
omitted identifying acquisition helper. No retrospective replacement is provided.

## Validation limits

The distribution supports the retained evidence audit and provides code and
protocols for fresh computation. An artifact packaging check is not a claim
that every acquisition, extraction, training or profiling campaign was rerun
on a second machine. Public data and platform-specific toolchains remain
external prerequisites. All original outcomes, experimental choices and
limitations remain visible in the unchanged research records.
