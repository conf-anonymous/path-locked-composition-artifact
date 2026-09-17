# Prospectively frozen protocol: ETH3D RGB-D odometry motor composition

Frozen on 2026-09-04 before any ETH3D SLAM archive was downloaded locally,
any RGB-D odometry was computed, any eligible sequence was parsed or
split-counted from local files, or any model was trained or evaluated.
Source-schema corrections discovered during loader validation may be recorded
as dated amendments here, but only before model training or outcome inspection.

Amendment 1, 2026-09-04, after downloading and listing the first official
`cables_1_rgbd.zip` but before odometry, training, or outcome inspection:
ETH3D's modality archives are additive. The `*_rgbd.zip` archive contains depth
and association files, while the matching `*_mono.zip` supplies the unchanged
RGB frames, intrinsics, and motion-capture trajectory required by the documented
RGB-D format. Eligibility therefore requires both official archives for each
sequence. This is a source-layout correction; it changes no sequence, split,
front-end setting, observation, target, arm, metric, or gate.

## Scientific question

Can a geometric-algebra motor pipeline improve endpoint prediction from real,
imperfect RGB-D odometry increments while remaining invariant to evaluation
path? On the identical recorded motions, does a competent learned binary
composer become locked to its training parenthesization, and does mixed-path
training remove that dependence?

ETH3D complements TUM's algebraic verification and Oxford's prospective outdoor
study. Inputs are estimated exclusively from recorded RGB and active-stereo
depth. Targets are independently measured by the motion-capture system. The
geometric premise remains primary: rigid camera motions are PGA motors, and the
fixed compositional law is their geometric product.

## Eligible public recorded data

The eligible population is the 56 real ETH3D SLAM training sequences that the
official benchmark overview identifies as motion-capture recordings. The five
`sfm_*` training sequences, all test sequences with hidden ground truth, all
calibration sequences, the separately listed synthetic datasets, and bundled
TUM sequences are ineligible. Only each eligible sequence's official
`<sequence>_mono.zip` and `<sequence>_rgbd.zip` are acquired. The two archives
are the minimal official pair needed for RGB, depth, intrinsics, and reference
poses under ETH3D's additive modality layout (Amendment 1).

No input image, input depth map, reference pose, perturbation, augmentation,
failure replacement, or noise may be generated. RGB and depth frames remain
unchanged. The only derived observations are the deterministic odometry
estimates described below: measurements computed from public real sensor
recordings that are not labels and never access ground truth values.

## Immutable split

For sequence name `s`, compute

`int(SHA256("eth3d-rgbd-v1:" + s)[:8], 16) mod 10`.

- buckets 0--5: training;
- buckets 6--7: development;
- buckets 8--9: untouched confirmation.

Over the official 56-name eligible list this yields 35/13/8 complete sequences.
No frame, odometry pair, or window crosses a sequence or split. Confirmation
targets may not be evaluated unless `--confirm` is supplied after the complete
development gate passes.

## Frozen RGB-D odometry front end

Use Open3D 0.19.0 `compute_rgbd_odometry` with
`RGBDOdometryJacobianFromHybridTerm`, identity initialization, and its published
defaults: pyramid iterations `[20, 10, 5]`, maximum depth difference 0.03 m,
minimum depth 0 m, and maximum depth 4 m. Construct RGB-D frames with ETH3D's
documented depth scale 5000, depth truncation 4 m, and intensity conversion.
Use the sequence's official `calibration.txt` intrinsics.

Starting at the first recorded RGB-D time, define a 0.1-second grid and select
the first recorded synchronized RGB-D frame at or after each grid time. Do not
interpolate or reuse a frame. Estimate motion only between consecutive selected
frames. Open3D returns the rigid coordinate transform from source-frame points
to target-frame points; invert it to obtain the source-to-target camera motion.
If Open3D reports failure, produces a non-finite matrix, or the selected-frame
gap exceeds 0.2 seconds, reject that pair and break the contiguous run. Also
reject a returned matrix whose final homogeneous row differs from `[0,0,0,1]`,
whose rotation orthogonality error exceeds 1e-5, or whose rotation determinant
differs from one by more than 1e-5. There is no projection-to-rigid, retry,
alternate initializer, tuned parameter, or ground-truth-based filter.

For every selected frame, choose the nearest recorded motion-capture row; reject
the anchor if its absolute time separation exceeds 0.02 seconds. Never
interpolate, average, smooth, or otherwise alter a reference pose. A valid leaf
requires both valid endpoint anchors and a successful odometry estimate.

Train at 8 leaves (0.8 seconds). Evaluate lengths 4, 8, 16, and 32 (0.4, 0.8,
1.6, and 3.2 seconds). Length 32 is primary. Report all valid windows and a
prespecified moving subset whose motion-capture endpoint displacement is at
least 0.15 m. All-window results are primary.

## Geometric representation

Encode every relative SE(3) camera motion as a unit dual quaternion, equivalently
a motor in the even subalgebra of 3D projective geometric algebra. The fixed law
is the motor geometric product

`(q1 + eps d1)(q2 + eps d2) = q1 q2 + eps(q1 d2 + d1 q2)`.

No normalization occurs inside an exact reduction. Normalization is permitted
only after a learned map and during decoding. Operand order never changes.

## Frozen arms and optimization

1. `raw_motor`: fixed geometric product of the Open3D odometry leaves.
2. `ga_calibrated`: a shared zero-initialized leafwise residual correction,
   followed exclusively by the exact motor product.
3. `learned`: the same leaf calibrator plus a zero-initialized learned bilinear
   residual around the motor product at every binary composition.
4. `mlp`: the same calibrator plus a parameter-matched nonlinear residual around
   the motor product.
5. `mixed`: the learned bilinear arm trained uniformly over left, right,
   balanced, and sampled order-preserving trees.
6. `penalty`: the left-trained bilinear arm with a local associator penalty on a
   consecutive triple from the same recorded sequence.

Use five seeds, 5,000 AdamW updates, batch size 256, learning rate 2e-3, weight
decay 1e-4, penalty weight 0.1, gradient clipping at one, and the same endpoint
motor loss and task batches for all compatible arms. Bilinear and MLP composer
residuals must differ by less than 5% in trainable parameter count. Development
may authorize one documented global learning-rate reduction applying to every
learned arm; arm-specific searches are prohibited.

## Evaluation and uncertainty

For every identical checkpoint and ordered leaf sequence, evaluate left, right,
balanced, and 16 independently seeded order-preserving random trees. Report mean
and median translation error in meters, rotation geodesic error in degrees,
path-minus-left changes, representation dispersion, per-sequence metrics, and
the same fields on the moving subset at every horizon.

Confidence intervals use 10,000 equal-weight resamples of complete sequences.
Overlapping windows are never treated as independent replicates.

## Development gate

Confirmation may be opened only if every condition holds on development at the
primary 3.2-second horizon:

1. every `ga_calibrated` seed improves trajectory-equal mean translation error
   over `raw_motor`, every paired sequence-clustered 95% interval lies below
   zero, and the across-seed mean improvement is at least 5%;
2. `ga_calibrated` changes by less than 1e-9 m and 1e-9 rad across every tested
   path, random tree, horizon, and seed in float64;
3. every `learned` seed's left-fold error is at most 10% above its matched
   `ga_calibrated` seed;
4. at least four of five `learned` right-minus-left clustered intervals are
   strictly above zero, and the across-seed gap is at least
   `max(0.05 m, 20% of learned left error)`;
5. mixed-path training removes at least 75% of the absolute learned right-fold
   gap while its left error remains at most 10% above `ga_calibrated`; and
6. complete MLP, penalty, moving-subset, shorter-horizon, failure, provenance,
   and runtime records exist before the confirmation decision.

If any condition fails, confirmation remains unopened and ETH3D does not enter
the ICLR manuscript.

## Confirmation claim

ETH3D enters the paper only if conditions 1--5 repeat on the untouched eight
confirmation sequences without changing any threshold, seed, arm, front-end
parameter, horizon, metric, split, or checkpoint. A failure cannot be rescued
by selecting particular scenes, conditions, horizons, moving windows, or seeds.
