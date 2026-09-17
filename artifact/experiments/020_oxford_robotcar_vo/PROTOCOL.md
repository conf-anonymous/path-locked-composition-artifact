# Prospectively frozen protocol: Oxford RobotCar VO-to-RTK motor composition

Frozen on 2026-09-03 before any Oxford RobotCar file was downloaded locally,
parsed, split-counted, trained on, or evaluated. This experiment may be amended
only for source-schema corrections discovered during loader validation and
before any model training or outcome inspection. Every amendment must be
recorded here.

Amendment 1, still before local acquisition, training, or outcome inspection:
Oxford's official SDK accumulates each VO relative transform at the timestamp
in column 0 of `vo.csv`. The alignment rule below therefore uses that published
SDK convention rather than calling column 1 the destination timestamp. This is
a source-schema clarification and changes no observation.

Amendment 2, still before local acquisition, training, or outcome inspection:
the implementation was hardened to remove the command-line training-step
override and to refuse confirmation evaluation unless the complete frozen
development audit exists and passes. This changes no scientific choice; it
makes the 5,000-update budget and confirmation gate fail closed in code.

## Scientific question

Can a geometric-algebra motor pipeline improve useful endpoint prediction from
real, imperfect visual-odometry increments while remaining exactly invariant to
evaluation path? If an unconstrained learned binary correction is competent on
the same task, does changing only its parenthesization expose path-locking?

This study addresses the limitation of Experiment 019: its operands were
derived from motion-capture poses, making the exact motor arm an algebraic
verification. Here the operands are Oxford's released stereo visual-odometry
estimates. Oxford explicitly describes them as a smooth local pose source that
drifts and is not ground truth. Targets come independently from the released
RTK reference poses.

## Eligible public recorded data

Eligible traversals are exactly the intersection of:

1. traversal IDs contained in Oxford's official RTK ground-truth archive;
2. traversal pages exposing an official `<traversal>_vo.tar`; and
3. files passing the official archive MD5 when one is published.

No image, pose, motion, perturbation, negative example, augmentation, or noise
may be generated. No visual-odometry system may be rerun or substituted. Only
the official `vo.csv`, official RTK CSV, and fixed official sensor extrinsics
are eligible. Missing or malformed traversals are excluded by a deterministic
schema rule and listed before training; they may not be replaced.

## Immutable split

The independent unit is a complete traversal. Assignment is

`int(SHA256("oxford-robotcar-vo-rtk-v1:" + traversal)[:8], 16) mod 10`:

- buckets 0--5: training;
- buckets 6--7: development;
- buckets 8--9: untouched confirmation.

The eligible list, checksums, exact split counts, and exclusion reasons are
written to a manifest before training. No window crosses a traversal or split.
Confirmation traversal targets must not be evaluated without `--confirm`.

## Recorded-row alignment and window construction

The official VO file gives source/destination camera timestamps and a relative
SE(3) estimate. These rows are accumulated in immutable file order to obtain a
VO pose at every recorded SDK pose timestamp (column 0, per Amendment 1). The
official RTK file supplies recorded reference poses at its own timestamps.

Starting at the first common time, define a 0.5-second grid. At each grid point,
select the first recorded RTK row at or after the grid time and the first
recorded VO pose timestamp, following the official SDK's column-0 convention,
at or after that RTK timestamp. Reject an anchor if
the VO/RTK timestamp separation exceeds 0.1 seconds. Never interpolate, average,
smooth, or alter a recorded row.

For two consecutive valid anchors, the input leaf is the relative motion between
their accumulated official VO poses. A window target is the relative motion
between its two recorded RTK endpoint poses, expressed in the vehicle frame
using Oxford's fixed official INS-to-vehicle extrinsic. Oxford's SDK explicitly
treats the VO and vehicle frames as identical. Discontinuous anchor runs are
never bridged. Sliding windows select existing leaves and targets; they do not
create observations.

Train at 8 leaves (4 seconds). Evaluate lengths 4, 8, 16, and 32 (2, 4, 8, and
16 seconds). The primary endpoint is length 32. Report all eligible windows and
also a prespecified moving subset whose RTK endpoint displacement is at least 5
meters; the all-window analysis is primary.

## Geometric representation

Every relative SE(3) motion is encoded as a unit dual quaternion, equivalently
a motor in the even subalgebra of 3D projective geometric algebra. Motor
composition is the dual-quaternion/geometric product

`(q1 + eps d1)(q2 + eps d2) = q1 q2 + eps(q1 d2 + d1 q2)`.

No normalization is applied inside an exact reduction. Normalization is allowed
only after a learned map and for decoding. Operand order is never changed.

## Frozen arms

1. `raw_motor`: official VO leaves composed by the fixed motor product; no
   trainable parameter.
2. `ga_calibrated`: a shared zero-initialized leafwise residual map corrects
   each official VO motor independently, after which the fixed motor product
   composes the corrected leaves. The composer remains exactly associative.
3. `learned`: the same leaf calibrator plus a zero-initialized learned bilinear
   residual around the motor product at every binary composition.
4. `mlp`: the same leaf calibrator plus a parameter-matched nonlinear residual
   around the motor product.
5. `mixed`: the learned bilinear arm trained uniformly on left, right, balanced,
   and sampled order-preserving trees.
6. `penalty`: the left-trained bilinear arm plus a local associator penalty on
   a consecutive triple from the same recorded traversal.

All learned arms use the same motor endpoint loss, optimizer, task batches,
training length, and update budget where their paths permit. The bilinear and
MLP composition residuals must be within 5% in trainable parameter count. Use
five seeds, 5,000 AdamW updates, batch size 256, learning rate 2e-3, weight decay
1e-4, and gradient clipping at one. Development may authorize one documented
learning-rate reduction for all learned arms together, but no arm-specific
search.

## Evaluation and uncertainty

For the identical checkpoint and identical ordered VO leaves, evaluate left,
right, balanced, and 16 independently seeded order-preserving random trees.
Report median and mean translation error in meters, rotation geodesic error in
degrees, path-minus-left paired changes, representation dispersion, and
per-traversal metrics at every length.

Confidence intervals use 10,000 resamples of complete traversals with equal
traversal weight. Overlapping windows are never treated as independent
replicates.

## Development gate

Confirmation may be opened only if all conditions hold on development at the
primary 16-second horizon:

1. every `ga_calibrated` seed reduces left-fold trajectory-equal mean
   translation error relative to `raw_motor`, every paired traversal-clustered
   interval lies below zero, and the across-seed mean reduction is at least 5%;
2. `ga_calibrated` changes by less than 1e-9 meters and 1e-9 radians across every
   tested tree and length in float64;
3. every `learned` bilinear seed has left-fold mean translation error no more
   than 10% above its matched `ga_calibrated` seed, so the composer is competent;
4. at least four of five `learned` seeds have a trajectory-clustered 95% interval
   strictly above zero for right-minus-left mean translation error, and their
   across-seed mean gap is at least max(0.5 m, 20% of learned left error);
5. mixed-path training reduces the absolute mean right-fold gap by at least 75%
   while its left error remains no more than 10% above `ga_calibrated`; and
6. the complete result, including MLP, penalty, moving-subset, and shorter-horizon
   outcomes, is written before the confirmation decision.

If this conjunction fails, confirmation remains unopened. The result is retained
as a development outcome and does not enter the ICLR manuscript.

## Confirmation claim

The Oxford result enters the paper only if conditions 1--5 repeat without any
threshold, seed, arm, metric, split, or checkpoint change on the untouched
confirmation traversals. Failure is reported internally and cannot be rescued
by selecting weather, route segments, moving windows, horizons, or individual
seeds. The local-penalty outcome is diagnostic and is never a condition for
opening or declaring confirmation.
