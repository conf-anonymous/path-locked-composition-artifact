# KITTI learned-composition implementation and run status

2026-09-11. **Implementation complete and sealed; prescribed fitting started.**
This milestone is not completion of the 30-fit campaign or new GA accuracy
evidence. Read the immutable per-fit receipts for the live completion count.

## Implemented and tested

- Recorded training-window loading, canonical leaf motor encoding, reference-only
  labels, fixed sampler, and the training-only translation scale: **108.0190199435 m**.
- Original 280-parameter leaf calibrator and five original heads. Head parameters:
  constant 1; contextual 10,689; feature-direct 10,920; feature-residual 10,920;
  mergeable 577. Include the frozen 280-parameter calibrator in full model totals.
- Serial five-seed, six-module-per-seed fitting: 5,000 AdamW updates per module,
  fixed learning rates/loss/batch size, CPU float32, one torch thread.
- Immutable 500-step checkpoints retaining model, optimizer, sampler RNG and
  torch RNG; failure receipts and no automatic restart of failed fits.
- Eight-arm segment-relative evaluation, complete per-window predictions,
  primary/secondary summaries, scope/coverage and extrapolation reporting.
- Float64 ordered-tree, chunk-merge and equivalent-matrix audits on identical
  recorded/calibrated leaves; no claim that associativity is unique to GA.
- Separate guarded evaluation-frontend staging, technical-only holdout release,
  and rejection of official KITTI test access. Development is conservatively
  delayed until all 30 final fits exist; successful holdout release requires all
  prescribed train/development reports and audits, never a positive GA effect.

**31 real-training-data/preseal tests pass.** Tests cover leaf direction,
coordinate round-trip, sampler replay, original model equivalence and gradients,
balanced-reducer equivalence, native metric precision, recorded tree/coordinate
audits, eight-arm aggregation, denied access, and stable provenance across lazy
model/optimizer initialization. Preflight and preseal tests made zero optimizer
updates. Successful post-fit release is not claimed exercised before its required
real reports exist. Historical frontend/preparation tests are also included.

## Seal and preserved correction

Active seal:
`data/raw/kitti_odometry/learned_composition_v1/implementation_seal.json`

SHA-256:
`3ee4326a9cffbcf4f914d5e7ddd7c894739a2dfdc60256bbbb52c09a50bd640d`

The seal pins 34 research/instruction source files, source/derived training
input identities, runtime/package versions, protocol, exact arms/budgets and
captured passing test output. Fresh-process verification passed before fitting.

The first, unused implementation seal and its workflow source are preserved
under `pretraining_seal_amendment_1/`. Its dynamic `.venv` import inventory was
corrected before any fit; [PRETRAINING_SEAL_AMENDMENT.md](PRETRAINING_SEAL_AMENDMENT.md)
documents the correction. No scientific protocol or dataset split changed.

## Live campaign

Started with:

```sh
.venv/bin/python experiments/033_kitti_motor_composition/workflow.py train
```

At this milestone, seed-0 calibrator and constant-gate fits had completed
5,000/5,000 updates; contextual fitting was underway. No failures had been
recorded. The completed calibrator checkpoint was reloaded with its optimizer:
all AdamW state counters equal 5,000, with 280 parameters and minimum observed
preprojection real norm 0.7464153. Calibrator/constant recorded sampling hashes
match at every logged step. These are execution checks, not scientific gains.

Last handoff check: **3/30 fits complete**, with seed-0 feature-direct fitting
active (step-4000 checkpoint observed), zero failure receipts. The current
command runs fitting only; later evaluation stages have not been queued.

All checkpoints, traces and completion/failure receipts live under
`data/raw/kitti_odometry/learned_composition_v1/fits/seed*/<arm>/`.
An OS advisory lock prevents concurrent fitting drivers for this campaign.
The command verifies completed fits on resumption; it does not overwrite them.

Preflight measured approximately **0.374 hours (22.45 minutes)** of fitting
compute for the complete prescribed budget, excluding data sampling, optimizer
step/checkpoint overhead, evaluation and later audits. This is an estimate, not
a completion time promise. The fit process is single-threaded; original images
are not decoded again for model training.

## Remaining work

Finish all 30 fits, evaluate all training arms/audits, stage and evaluate
development, then release the two study holdouts only if all technical checks
pass. Run every prespecified arm/seed once on those holdouts regardless of the
development effect's direction. Commands are in
[IMPLEMENTATION_NOTES.md](IMPLEMENTATION_NOTES.md).

At this milestone, 07–10 remain scientifically uninspected and 11–21 remain
outside scope. No manuscript/PDF/artifact change or new GA gain claim has been
made. Oxford remains excluded pending authoritative clarification.
