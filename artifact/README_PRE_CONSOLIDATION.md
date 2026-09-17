# Anonymous reproducibility artifact

This artifact supports *Path-Locked Neural Composition: Exact Evaluation-Path
Invariance with Geometric Algebra*. It contains the complete MovieLens-1M, TUM RGB-D, and
ETH3D development and evaluation trail reported in the paper, together with
the excluded Oxford RobotCar development attempt and its post-outcome frame
audit. Oxford is not supporting accuracy evidence. It contains no generated
dataset, private institutional data, repository history, author information,
or credentials.

## Fast numerical verification (no dataset required)

```bash
python verify_claims.py
```

The verifier reads immutable JSON records and fails if any reported competence,
path-degradation, intervention, family-held-out, matched-control, seed-coverage,
or exactness result is inconsistent. It also verifies that Experiment 021's
negative development outcome remains negative, and separately checks Oxford's
failed gate, all 50 checkpoints, available sealed sources, and reference audit.
Oxford's original seals and documented pre-training amendments are preserved;
they are not retroactively relabeled as an unchanged pre-acquisition implementation.
Follow-up audits additionally verify complete control coverage, development-only
selection and arithmetic, without requiring a favored model to win. These checks
establish numerical consistency, not novelty, causal adequacy, or acceptance odds.

The unchanged Experiment 030 scope audit recorded a broad source inventory that
also included then-unused Oxford utilities. `verify_scope_history.py` resolves
three later-amended Oxford files to their exact retained pre-amendment snapshots
and checks the original hashes; it does not change any ETH3D model, result,
threshold or statistical calculation. Use this wrapper instead of invoking
Experiment 030's old source-inventory check directly in the combined snapshot.

## Environment

- Python 3.11 or newer
- PyTorch 2.6 or newer
- NumPy 1.26 or newer
- CPU execution is sufficient for audits

The new training runs used Python 3.14.7, PyTorch 2.12.0, NumPy 2.4.6,
macOS ARM64, one Torch thread per worker (two concurrent workers per campaign).
Use these versions for close numerical retraining; dependency lower bounds are
installation bounds, not a promise of bitwise results across versions/hardware.

Install from the artifact root:

```bash
python -m pip install -e .
```

Open3D is required only to re-extract ETH3D RGB-D odometry from the official
archives. Open3D 0.19 supports Python 3.11--3.12; a separate Python 3.12
environment is recommended for that step.

## Public datasets

### MovieLens-1M

Download the stable MovieLens-1M archive from the official GroupLens page and
place it at `data/raw/ml-1m.zip`. The experiment refuses to run unless its MD5
is `c4d9eecfca2ab87c1945afe126590906`. The archive is not redistributed because
its research-use terms require users to obtain it from GroupLens.

The experiment reads every row of `ratings.dat`, does not use demographic
attributes, and assigns users to disjoint splits through a deterministic
SHA-256 function of user ID.

### TUM RGB-D ground-truth trajectories

```bash
python experiments/017_tum_motor_composition/download_data.py
```

The downloader obtains only the 47 official ground-truth text files, records
SHA-256 checksums, and does not redistribute source trajectories. The study
selects recorded rows without interpolation and splits complete trajectories
25/9/13.

### ETH3D RGB-D SLAM benchmark

Using a Python 3.12 environment with Open3D 0.19:

```bash
python experiments/021_eth3d_rgbd_motor/download_data.py
python experiments/021_eth3d_rgbd_motor/extract_odometry.py
```

The downloader admits exactly the 56 named real motion-capture training
sequences and refuses ETH3D synthetic, test, calibration, SfM, and bundled TUM
archives. It records official archive hashes, and the extractor records hashes
of the fixed Open3D odometry outputs. Experiments 021--024 preserve the negative
first protocol, development-selected contextual correction, grouped
family-held-out evaluation, and a two-stage budget-matched direct control. The
original control did not match information interfaces; Experiment 028 addresses
that limitation without modifying the original result.

### Oxford RobotCar

Experiments 020 and 031 completed all 50 prescribed development fits. The
original conjunctive gate failed and shared confirmation remained closed.
The subsequent independent homogeneous-matrix audit reproduced the pipeline's
arithmetic but exposed a systematic approximately 90-degree VO/RTK frame
discrepancy. A fixed quarter-turn diagnostic sharply reduced raw error without
learning. Therefore the apparent learned accuracy gains are confounded and
excluded from the paper's supporting results; Appendix J discloses the attempt.

The full immutable development records/checkpoints and their integrity metadata
are included for transparency, not as a validated benchmark. No raw VO/RTK CSV
or downloaded dataset archive is redistributed. The manifest contains hashes
and timing metadata, not pose observations. All nine confirmation traversals
remain unopened. Do not run training, confirmation, or a retrospective gate
relaxation from this historical export.

One unused local acquisition helper, `stage_downloads.py`, is omitted because
its default Downloads path identifies an author. `ANONYMITY_OMISSIONS.json`
records its original hash and reason. The read-only verifier checks every other
file in the active seal and explicitly reports the unverifiable omission. The
original full-source admission verifier intentionally fails closed in this
export; it is not silently rewritten or presented as a complete original seal
verification. Scientific models, results and protocols are unchanged.

`iclr-2027-composition/oxford-audit-2026-09-10/` contains the audit code, all
14-traversal/four-horizon diagnostic records, reference SDK functions with
original licensing notices, and source links. See `REPLAY.md` for a read-only
matrix replay after obtaining official development CSVs. The diagnostic frame
turn is not an authoritative corrected calibration or a new model ranking.

## Layout

- `experiments/016_movielens_composition/`: MovieLens code, frozen protocol,
  checkpoints, development result, sealed confirmation, and audit.
- `experiments/017_tum_motor_composition/`: original frozen TUM protocol,
  official-data downloader, exact GA motor implementation, failed pilot, and
  audit.
- `experiments/018_tum_residual_motor/`: residual-over-invariant-control
  parameterization, frozen checkpoints, development record, and failed gate.
- `experiments/019_tum_long_horizon_confirmation/`: confirmation-frozen TUM
  protocol, untouched confirmation record, and conjunctive audit.
- `experiments/020_oxford_robotcar_vo/`: prospectively sealed Oxford protocol,
  loaders, runner, failed gate and excluded development outcomes; original
  sources and pre-training amendments remain traceable.
- `experiments/021_eth3d_rgbd_motor/`: official ETH3D acquisition, fixed Open3D
  extraction, negative development result, and frozen checkpoints.
- `experiments/022_eth3d_adaptive_motor/`: contextual exact-motor development,
  sequence-held-out confirmation, complete pilot trail, and checkpoints.
- `experiments/023_eth3d_family_heldout/`: five-fold, five-seed evaluation over
  16 indivisible capture families and its clustered audit.
- `experiments/024_eth3d_calibrated_direct_control/`: two-stage budget-matched
  direct-GRU comparison and its clustered audit.
- `experiments/025_eth3d_compute_profile/`: post-freeze public-input latency,
  throughput, parameter, and recorded training-time profile.
- `experiments/026_eth3d_frozen_heterogeneity/`: post-hoc, frozen-output family
  robustness and failure analysis.
- `experiments/027_movielens_closure_controls/`: stronger primitive/intermediate
  regularizers, complete development search history, five-seed selected models,
  history-aware baselines, and numerical audits. Final selection uses three dev
  seeds; earlier seed-zero searches remain explicitly historical records.
- `experiments/028_eth3d_attribution_controls/`: four fixed heads with matched
  information where applicable, all folds/seeds/checkpoints, identity-motion
  evaluation, conditional uncertainty, and product-cache/numerical audits.
- `experiments/029_eth3d_mergeable_motor_gate/`: a GRU-free 577-parameter gate
  reading the associative 16-coordinate pair of motor products; all 25
  fold/seed fits, numerical audits, and paired comparisons.
- `experiments/030_eth3d_frozen_scope/`: all eight arms at six horizons from
  every frozen fold/seed checkpoint, including all unfavorable outcomes.
  Fast auditing recomputes metrics from records; with public data present,
  `audit_results.py --require-data` also checks source hashes and durations.
- `experiments/031_oxford_mergeable_confirmation/`: implemented joint workflow,
  excluded development records/checkpoints, failed release decision and preserved
  protocol/source history. No confirmation outcome exists.
- `iclr-2027-composition/review_diagnostics.py`: original frozen-checkpoint
  query-last diagnostic; the adjacent JSON preserves its pre-revision output.

## Reproduce evaluation from frozen checkpoints

After placing the official MovieLens archive:

```bash
python experiments/016_movielens_composition/run.py \
  --seeds 0 1 2 3 4 --steps 3000 --batch-size 256
python experiments/016_movielens_composition/run.py \
  --seeds 0 1 2 3 4 --steps 3000 --batch-size 256 --confirm
```

Because checkpoints are present, these commands reproduce evaluation without
retraining. Moving `runs/` aside first reruns the development workflow;
`--confirm` refuses to train a missing checkpoint.

After downloading TUM:

```bash
python experiments/019_tum_long_horizon_confirmation/run.py
python experiments/019_tum_long_horizon_confirmation/audit_results.py
```

After downloading and extracting ETH3D, follow the commands in Experiments
021--024. The fast verifier needs no raw dataset because it audits immutable
reported result records directly.

## Post-hoc follow-ups (not new untouched confirmation)

After making public data available, the follow-up scripts train/evaluate only
recorded inputs. Existing output JSON is never overwritten. To retrain from
scratch, work in a separate copy and move the follow-up `runs/` and generated
JSON aside there, retaining the original evidence package. Run Experiment 027
in the recorded order: `run.py`, `boundary_check.py`, `robust_selection.py`.
The original searches are preserved to disclose selection history, not offered
as interchangeable confirmatory analyses. `robust_results.json` is the final
three-development-seed selection. Experiment 028 uses `run.py`, then
`check_cached_inference.py`, `cache_deployment.py`, and `analyze.py`.
After Experiment 028, run Experiment 029's `run.py` then `analyze.py` and
`cpu_profile.py` (after other training completes).
Its summary explicitly corrects the generic numerical helper's inherited
GRU-specific scope label: Experiment 029 has no GRU, as its source and
checkpoints show. The numerical checks are unchanged; raw records are preserved.

Experiment 030 performs no fitting. For a fresh inference reproduction, work
in a separate copy, retain its sealed protocol/dependencies, and move its
generated `runs/` and `analysis.json` aside before `python run.py`. The original
L32 results must reproduce before new horizons are interpreted. Frozen L32-
trained heads are not retrained at shorter or longer horizons. Long horizons
lose eligible sequences/families; all comparisons share windows within each
horizon, but comparisons across horizons do not hold recording coverage fixed.

For query-last inference without accessing obsolete manuscript hashes:

```bash
python -c 'import runpy, json; m = runpy.run_path("iclr-2027-composition/review_diagnostics.py"); m["torch"].set_num_threads(1); print(json.dumps(m["movielens"](), indent=2))'
```

None of the follow-ups restores untouched test data. ETH3D intervals condition
on the fitted models and named families; overlapping fold training sets remain
dependent. Neither family nor fold sign-flip enumeration alone proves valid
unconditional statistical inference. Oxford confirmation remains sealed;
its completed development attempt is excluded for the reasons above.

## Data boundary

All model inputs and labels are recorded MovieLens ratings, recorded TUM poses,
or recorded ETH3D RGB-D frames and motion-capture poses. The excluded Oxford
attempt used official recorded VO and RTK poses. The frame audit changes only
coordinate interpretation for diagnosis; it does not generate observations or
replace training targets. Random initialization
and minibatch selection sample model state or existing training endpoints.
Random-tree schedules reparenthesize the same ordered observations. Clustered
bootstraps resample whole users, trajectories, or capture families. None of
these procedures creates or modifies an observation or label.
