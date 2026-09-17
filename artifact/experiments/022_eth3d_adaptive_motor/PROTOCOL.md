# Development protocol: adaptive motor correction on ETH3D

Created 2026-09-05 after Experiment 021 failed its frozen development gate and
before training the model specified here. This is a new hypothesis, not an
amendment or rescue of Experiment 021. Experiment 021 and its negative decision
remain immutable.

## Observation motivating the hypothesis

On the Experiment 021 development split, a shared leafwise calibrator improved
the sequence-equal mean at 3.2 seconds but helped high-drift sequences while
harming several already accurate sequences. Its training distribution sampled
overlapping windows uniformly and its correction was repeatedly applied at
every leaf.

## Hypothesis

RGB-D odometry error is regime dependent. A path-invariant model should infer a
correction from the ordered, observed motor increments, but the correction
should be applied once to the exact aggregate motor rather than indiscriminately
to every leaf. Equal-probability sequence sampling and a robust loss should stop
long or catastrophically drifting recordings from dominating optimization.

For ordered input motors M_1,...,M_L, first compute

    M_raw = M_1 M_2 ... M_L

using only the exact PGA motor product. A bidirectional recurrent encoder reads
the recorded leaf motors and emits a six-dimensional corrective twist xi. Map
xi to a unit corrective motor C(xi), then predict

    M_pred = M_raw C(xi).

The algebraic product remains the sole aggregation law. The learned component
estimates one context-dependent motor in the endpoint frame; it does not learn
or replace the binary composition operator. Since M_raw is associative and the
correction is applied after reduction, predictions are path invariant by
construction.

## Data boundary

Use exactly the public real ETH3D recordings, deterministic Open3D odometry,
immutable sequence split, and derived-file hashes in Experiment 021. Load only
training and development records. This exploratory program contains no code
path that can load or evaluate confirmation records. No synthetic observation,
pose, motion, perturbation, augmentation, or noise is permitted.

## Initial development configuration

- input representation: recorded eight-coordinate unit motors;
- exact raw motor plus a bidirectional GRU with 32 hidden units per direction;
- zero-initialized six-coordinate correction head;
- train jointly at 4, 8, 16, and 32 leaves;
- sample a horizon uniformly, then a training sequence uniformly, then a valid
  window uniformly within that sequence;
- five seeds, batch size 256, AdamW, learning rate 1e-3, weight decay 1e-4,
  gradient clipping at one, and 5,000 updates;
- robust translation loss (componentwise smooth L1 after normalization by the
  frozen training translation scale) plus quaternion geodesic surrogate;
- report every window and sequence-equal uncertainty at every horizon.

Architecture and optimization work on development is exploratory and must be
logged rather than retroactively described as prospective. After a satisfactory
development result, freeze a new confirmation gate and checkpoint hashes before
adding any confirmation-capable code or reading any confirmation outcome.

## Criterion for a satisfactory development result

At the primary 32-leaf horizon, all five seeds must improve the sequence-equal
mean translation error over raw composition, their across-sequence mean gain
must be at least 10%, and the pooled five-seed sequence-level paired bootstrap
95% interval must lie below zero. Exact path invariance must hold below 1e-9 in
float64 at every horizon. Also report per-sequence outcomes, shorter horizons,
rotation errors, parameter count, and runtime. This is a development selection
criterion, not yet a confirmation claim.

## Development log

### Iteration A — direct corrective motor

The initial seed-0, 5,000-update pilot improved the sequence-equal translation
mean by 5.6% at 8 leaves but worsened it by 3.8% at the primary 32 leaves. It
therefore was not expanded to five seeds.

### Iteration B — contextual identity/calibration gate

Specified after observing Iteration A and before its own training. Experiment
021 already contains a path-invariant candidate correction: its GA-calibrated
branch. The new model freezes that training-only calibrator and computes both
the identity branch (raw exact product) and calibrated branch (calibrated leaves,
then exact product). A recurrent gate sees only recorded input motors and mixes
the two motors with a scalar in [0,1], followed by unit-motor normalization.
Thus it can retain identity on clean windows and invoke calibration in regimes
where the input motion indicates drift. Both alternatives use the PGA product
exclusively, and the gate is independent of reduction path. Train the gate at
the primary 32-leaf horizon with sequence-balanced sampling and the same robust
loss. This is a model-selection iteration on development, not confirmatory
evidence.
