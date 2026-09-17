# ETH3D adaptive motor results

## Status

Experiment 022 is a development-selected, confirmation-frozen follow-up to the
negative Experiment 021 gate. It uses only public real ETH3D RGB-D recordings,
Open3D odometry derived from those recordings, and motion-capture targets. No
synthetic data, perturbation, augmentation, or generated observation is used.

The initially specified direct corrective-motor model failed its seed-0 pilot
at the primary 32-leaf horizon and was stopped. The subsequently logged
contextual identity/calibration gate passed its five-seed development criterion.
Its architecture, checkpoints, hashes, confirmation analysis, and thresholds
were then frozen in `CONFIRMATION_PROTOCOL.md`. The confirmation evaluation was
run once and passed every frozen condition.

## Primary sequence-equal translation result

| Split | Raw at L32 | Contextual gate at L32 | Relative change | Pooled sequence 95% CI for delta |
|---|---:|---:|---:|---:|
| Development (11 evaluable sequences) | 0.966 m | 0.420 m | -56.5% | [-1.211, -0.024] m |
| Confirmation (8 sequences) | 4.948 m | 0.284 m | -94.3% | [-10.552, -0.118] m |

Development seed means were 0.428, 0.477, 0.390, 0.415, and 0.390 m.
Confirmation seed means were 0.237, 0.315, 0.287, 0.299, and 0.280 m. Every
seed improved over raw on both splits.

## Confirmation by horizon

| Leaves | Raw translation | Contextual gate | Relative change | Raw rotation | Gate rotation |
|---:|---:|---:|---:|---:|---:|
| 4 | 1.547 m | 0.143 m | -90.8% | 23.52 deg | 11.28 deg |
| 8 | 2.591 m | 0.129 m | -95.0% | 29.48 deg | 11.67 deg |
| 16 | 3.640 m | 0.176 m | -95.2% | 32.13 deg | 16.82 deg |
| 32 | 4.948 m | 0.284 m | -94.3% | 35.23 deg | 24.82 deg |

The worst float64 prediction discrepancy across left, right, and balanced
parenthesizations was `1.46e-14`, versus the frozen `1e-9` tolerance.

## Interpretation and limitation

The gate behaves as intended. In a post-confirmation diagnostic, its mean value
was approximately one on both high-drift `camera_shake` sequences and generally
0.03--0.27 on already accurate confirmation sequences. The ungated frozen GA
calibrator obtained 0.465--0.641 m across seeds at L32; contextual gating reduced
that to 0.237--0.315 m. This diagnostic was not part of the frozen confirmation
gate and must be labeled as such if reported.

The unusually large confirmation effect is driven mainly by `camera_shake_1`
and `camera_shake_3`, with a further large gain on `plant_scene_2`. Related
named sequences occur in training (`camera_shake_2`) and development
(`plant_scene_3`). Thus the immutable hash split establishes held-out-sequence
generalization, not held-out-capture-family generalization. The mean translation
gain remains negative after removing any single confirmation sequence, but the
paper must not characterize this experiment as evidence for unseen-family or
unseen-domain transfer. A group-held-out study would be required for that
stronger claim.

## Immutable records

- development/model-selection record:
  `results_pilot_gated_0-1-2-3-4_5000.json`
- confirmation record: `results_confirmation.json`, SHA-256
  `6b8833a9ee8a636357d8110c57fa5cf79f911c53793c526a0baa118e3ddf2abd`
- original negative experiment: `../021_eth3d_rgbd_motor/`
