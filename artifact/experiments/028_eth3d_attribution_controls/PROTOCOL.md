# Recorded ETH3D attribution follow-up (2026-09-06)

The author explicitly authorized this campaign after the original ETH3D freeze
and adversarial review. It does not modify or supersede Experiments 021--026.
All old outcomes were known; these are post-hoc grouped robustness controls,
not untouched confirmation. Oxford remains out of scope pending access.

Use exactly the recorded, checksum-verified ETH3D inputs, reference endpoints,
16-family/five-fold partition, seeds 0--4, L8 calibration and L32 primary task
from Experiment 023. No synthetic data, artificial perturbations or augmentation.
No tuning/early stopping using held-out families; use fixed original budgets.

First reproduce the frozen leaf calibrator in every fold/seed. Cache its leaves
and exact raw/calibrated products outside autograd (calibrator is frozen in all
heads), verifying cached versus online inference on actual recorded inputs.
Train four heads with 5,000 AdamW updates and exactly the same sequence-balanced
training batches, learning rate, loss and clipping:

1. Constant gate: one learned scalar between raw/calibrated exact endpoints.
2. Contextual gate: original raw-leaf bidirectional GRU + BOTH exact endpoints;
   one sigmoid coefficient, initial bias -2, zero last-layer weights.
3. Feature-matched direct head: identical raw-leaf GRU + BOTH exact endpoints,
   80->32->8 head, unit-motor projection. Initialize last layer to identity
   prediction (zero weights, bias real quaternion identity), avoiding arbitrary
   near-zero initial quaternion normalization. It is not denied exact features.
4. Feature-matched residual head: same 80->32->8 head and raw-leaf GRU, zero
   initial residual added to the same initial raw/calibrated blend as gate (2),
   followed by unit-motor projection.

Common GRU/first head layer initialization per seed. Heads differ in output
constraint, not available inputs or update budget. Constant has no GRU.
New seed stream (50000+seed) distinguishes these from immutable earlier runs.
Train every listed arm; report all, irrespective of ranking. Save checkpoints.
Exact products are fixed algebraic features; this is an attribution of the
calibration/gating prior, not a test of GA superiority over equivalent matrices.

Report paired sequence-then-family-equal translation AND rotation, all folds
and seeds, conditional family bootstrap intervals and fold-block sensitivity.
Formally evaluate a parameter-free identity-motion predictor on all 11,219
held-out endpoints. Report per-family failures, not only the aggregate gain.
Numerical audit: minimum un-clamped real-quaternion norm of predictions; separate
meters/radians tree discrepancy; recorded chunk/cache regrouping of product
branches. The complete bidirectional context is not a mergeable summary.
No universal denominator guarantee or end-to-end parallel speed claim follows.

Source/protocol hashes are frozen locally before fitting; they document file
integrity, not independently timestamped preregistration.
