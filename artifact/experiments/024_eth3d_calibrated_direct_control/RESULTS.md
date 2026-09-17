# Fully matched direct-control result

The frozen control supports a strict contextual-GA accuracy advantage at the
primary 32-leaf horizon.

| Method | Family-equal translation | Family-equal rotation |
|---|---:|---:|
| Contextual GA | **0.440 m** | 28.09 deg |
| Calibrated direct GRU | 0.503 m | **26.53 deg** |

The contextual-minus-direct family-equal translation difference is -0.063 m,
with a 10,000-resample matched-family 95% interval of
[-0.116, -0.013] m. Contextual GA is better on 11 of 16 families. The direct
control has slightly lower rotation error, so the advantage is specifically in
the frozen primary translation endpoint and must be described that way.

Both pipelines use the identical fold-specific 280-parameter leaf calibrator,
which was retrained from scratch for every fold and seed and frozen before the
primary model. Reproduced calibrator metrics match Experiment 023 exactly. Both
primary models use the same calibrated leaves, grouped folds, sequence-balanced
batches, five seeds, 5,000 updates, loss, optimizer, and gradient clipping.
Their trainable counts are 10,689 and 10,846 (1.47% apart). The direct model
predicts an endpoint motor without an exact product; contextual GA gates between
raw and calibrated motors, each reduced by the exact PGA product.

Only public real ETH3D recordings and derived Open3D odometry were used. No
synthetic data, generated observation, perturbation, augmentation, or imputation
was used.
