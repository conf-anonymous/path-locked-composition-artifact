# Frozen protocol: fully two-stage-matched direct control

Frozen 2026-09-05 after Experiment 023 passed and before this control was
trained or evaluated. This control addresses the disclosed optimization-budget
difference; it cannot alter Experiment 023's recorded protocol or outcomes.

Use the identical public real ETH3D data, 16-family map, five outer folds,
five seeds, targets, horizons, and metrics from Experiment 023. No synthetic
data, generated observation, perturbation, augmentation, or imputation is
permitted.

For every fold and seed, reproduce Experiment 023's first stage from scratch:
train the same 280-parameter leaf calibrator at eight leaves for 5,000 updates
using the same initialization, sampling, optimizer, loss, and training families.
Verify its held-out predictions reproduce the recorded Experiment 023 values to
numerical tolerance.

Freeze that calibrator and pass its calibrated leaf sequence to the same direct
bidirectional GRU control: 32 hidden units per direction and a 38-unit GELU
head predicting the endpoint motor without any exact reduction. Train for 5,000
updates at 32 leaves with the same sequence-balanced batches, seeds, robust
loss, optimizer, and gradient clipping used for contextual GA. The resulting
pipeline has the same two training stages as contextual GA; primary-stage
trainable parameter counts must differ by less than 5%.

Length 32 is primary. Aggregate each sequence within each wholly held-out
family, then weight the 16 families equally. Use 10,000 matched family bootstrap
resamples with RNG seed 240905.

Interpretation is frozen:

- contextual GA has a strict accuracy advantage if the 95% interval for
  contextual GA minus calibrated direct GRU lies below zero;
- the methods are statistically unresolved if the interval includes zero;
- the calibrated direct GRU has a strict advantage if the interval lies above
  zero.

Report the interval and mean regardless of direction. Also report rotation,
all horizons, parameter counts, training time, per-family values, and the exact
path audit for contextual GA from Experiment 023. No subsequent ETH3D model
development is authorized by this control.
