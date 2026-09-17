# KITTI implementation v1

2026-09-11. Implements the previously sealed scientific specification; does not
edit the frontend/scientific seal or change budgets, seeds, arms or endpoints.
The author's subsequent "Proceed" authorizes this implementation and fitting
stage; Section 9 of the scientific specification records the earlier stage.

- Original classes/functions are imported through experiment 029's existing
  import chain. No historical experiment data loader or training routine runs.
  The implementation fingerprint includes the loaded local source dependencies.
- Balanced reduction vectorizes independent nodes for power-of-two lengths.
  It retains the original midpoint tree and is tested for exact float32
  agreement against the original reducer on real training motion. Other lengths
  use the original reducer. No intermediate sign canonicalization occurs.
- Sampling uses a CPU torch Generator, draws length once per batch, then
  sequence/start independently for each of 64 examples. All heads restart that
  sampler at 30000+seed. Training labels may be cached; none enter the features.
- Fixed-length secondary evaluation takes every tenth entry of the eligible
  start inventory. Its report includes total/valid starts and sampled count;
  do not call that subsampled count all-window coverage. Distance evaluation
  retains every official-grid candidate and its valid/failed status.
- Models fit serially with one CPU torch thread. Every 500 updates yields an
  immutable checkpoint with optimizer, model, sampler and torch RNG state.
  Resumption uses that exact state, not a new seed/restart. Numerical failures
  create a failure receipt and block the campaign; incomplete files are not
  overwritten. Preflight/tests perform backward checks, no optimizer updates.
- Even development extraction is conservatively delayed until all 30 final
  checkpoints exist. Nontraining metadata/images are read only behind access
  checks. The official KITTI test is never allowed. A holdout release requires
  complete train/dev reports, all eight arms/seeds, frontend checks and all
  geometric audits, but never a favorable development accuracy comparison.
- Evaluation predictions are float32; error calculation and geometric audits
  are float64. Geometric audits also bound equivalent-matrix coefficient error
  below 1e-8, a numerical consistency tolerance rather than a superiority claim.
  The original devkit casts translation/rotation error coefficients to float32;
  cross-precision metric tests retain the earlier smoke's 1e-5 tolerance, not
  bitwise equality or seven-decimal identity. A preseal overly strict equality
  test exposed a 1.20e-7 percentage-point rounding difference; the implementation
  formula and frozen scientific thresholds were not changed in response.
- A partial evaluation remains an explicit partial attempt, not an automatic
  repeat of the holdout. Tests before sealing cover denied access and real train
  replay; successful release cannot be empirically exercised before the required
  real completed reports exist. This is a research-workflow guard, not a claim
  of adversarial operating-system isolation against a user editing files.
- Timing is diagnostic, not a published speedup benchmark. Report model costs
  including shared raw/calibrated products; exclude frontend only explicitly.

Commands (stage boundaries are deliberate):

```sh
.venv/bin/python experiments/033_kitti_motor_composition/preflight.py
.venv/bin/python experiments/033_kitti_motor_composition/workflow.py seal
.venv/bin/python experiments/033_kitti_motor_composition/workflow.py train
.venv/bin/python experiments/033_kitti_motor_composition/evaluate.py train
.venv/bin/python experiments/033_kitti_motor_composition/stage_evaluation.py dev
.venv/bin/python experiments/033_kitti_motor_composition/evaluate.py dev
.venv/bin/python experiments/033_kitti_motor_composition/workflow.py release-holdout
.venv/bin/python experiments/033_kitti_motor_composition/stage_evaluation.py study_holdout
.venv/bin/python experiments/033_kitti_motor_composition/evaluate.py study_holdout
```

The entire `test_*.py` suite is a TRAIN-only preseal suite, including historical
tests that assert no nontraining extraction exists. After legitimate development
access those historical assertions no longer describe the stage; use the seal,
stage receipts and current-stage audits, not a misleading all-stage pass claim.
