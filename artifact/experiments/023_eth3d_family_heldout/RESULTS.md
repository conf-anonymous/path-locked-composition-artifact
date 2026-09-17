# Family-held-out ETH3D results

## Decision

The frozen family-held-out gate passed every condition. This study uses only
public real ETH3D RGB-D recordings, deterministic Open3D odometry derived from
those recordings, and motion-capture targets. No synthetic data, generated
observation, perturbation, augmentation, or imputation was used.

All 16 capture families were each evaluated exactly once in a five-fold grouped
design. Both the leaf calibrator and contextual gate were retrained from scratch
for every fold and seed; no sequence from an evaluated family was available to
either training stage.

## Primary result at 32 leaves

| Method | Family-equal translation | Family-equal rotation |
|---|---:|---:|
| Raw exact motor product | 4.156 m | 36.03 deg |
| GA leaf calibration | 0.530 m | 41.12 deg |
| Contextual GA gate | **0.440 m** | **28.09 deg** |
| Parameter-matched direct GRU | 0.562 m | 28.01 deg |

The contextual GA seed means were 0.431, 0.442, 0.461, 0.444, and 0.422 m.
Every seed improved raw composition. The across-seed mean improvement was
89.4%, and the 10,000-resample family-clustered 95% interval for contextual GA
minus raw was [-7.200, -0.980] m.

Contextual GA improved 12 of 16 held-out families. The median family delta was
-0.774 m. Removing any single family left a beneficial mean; the least favorable
leave-one-family-out delta was -2.429 m. Thus neither `camera_shake` nor any
other single high-error family determines the conclusion.

The trainable contextual gate has 10,689 parameters and the direct GRU 10,846,
a 1.47% difference. Contextual GA beat the direct GRU with a family-bootstrap
95% interval of [-0.170, -0.076] m. Its family-equal rotation error also improved
raw by 22.0%. The worst exact-product prediction discrepancy over every fold,
seed, horizon, and evaluated product arm was `3.43e-12`, below the frozen `1e-9`
tolerance.

## Horizon results

| Leaves | Raw | GA calibrated | Contextual GA | Direct GRU |
|---:|---:|---:|---:|---:|
| 4 | 0.804 m | 0.096 m | 0.170 m | 0.759 m |
| 8 | 1.409 m | 0.170 m | 0.203 m | 0.722 m |
| 16 | 2.323 m | 0.301 m | 0.276 m | 0.616 m |
| 32 | 4.156 m | 0.530 m | **0.440 m** | 0.562 m |

Length 32 was the frozen primary endpoint. Shorter horizons are complete
descriptive results; they were not used to rescue the decision.

## Boundary of the claim

This removes the named-family leakage concern in Experiment 022: evaluated
families are wholly absent from training. Because ETH3D had already been
inspected before this protocol was frozen, it remains a prospectively specified
grouped robustness study rather than a newly untouched dataset confirmation.
The justified claim is unseen-ETH3D-capture-family generalization. Cross-dataset
generalization remains for Oxford or another independently untouched public
real-sensor dataset.

Experiment 024 closes the original control's preprocessing and two-stage-compute
difference. Its calibrated direct GRU receives the identical retrained and
frozen leaf calibration before matched primary-stage training. Contextual GA
remains better at 0.440 versus 0.503 m, with a matched-family 95% interval of
[-0.116, -0.013] m. The direct control has slightly lower rotation error
(26.53 versus 28.09 degrees), so the supported comparative advantage is the
prespecified translation endpoint rather than every metric.

## Records

- `PROTOCOL.md`: protocol frozen before grouped-fold training
- `results.json`: complete five-fold, five-seed record, SHA-256
  `f25987c0687078d4fb42021a25ee11a031c4b29d741ada91ca017cf3442458e1`
- `audit.json`: machine-readable frozen decision
