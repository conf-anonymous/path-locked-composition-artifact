# Frozen protocol: TUM RGB-D motor composition

Frozen on 2026-09-03 before any model was trained or any outcome was inspected.

Data-loader amendment, also before training or outcome inspection: the official
`freiburg2_xyz` file contains eight duplicated timestamp strings. The loader
retains the first row in immutable source-file order when first-at-or-after
selection lands on such a timestamp. No pose is averaged, interpolated, or
otherwise changed.

## Scientific question

When a learned binary composer is competent on a serial reduction of recorded
rigid motions, does it preserve the result under algebraically equivalent
parenthesizations? Does an associative geometric-algebra substrate eliminate
that dependence by construction?

## Eligible data

Only the public TUM RGB-D ground-truth trajectory files listed in
`data_manifest.py` are eligible. TUM obtained these trajectories using an
external high-accuracy motion-capture system. Images, depth maps, generated
motions, interpolated poses, perturbations, augmentation, and simulated data
are not used.

Each source row is `(timestamp, tx, ty, tz, qx, qy, qz, qw)`. We require
nondecreasing source timestamps and retain immutable source-file order for the
rare duplicate timestamp; first-at-or-after selection therefore chooses the
first source row. We enforce the representation-only quaternion sign convention
`dot(q[t-1], q[t]) >= 0`, and select the first recorded row at or after each
0.1-second grid point. We never interpolate. Consecutive selected poses yield
recorded relative motions. Sliding windows are index selections from those
motions, not generated observations.

## Geometric representation

A rigid motion is encoded as a unit dual quaternion, equivalently a motor in
the even subalgebra of three-dimensional projective geometric algebra. The
fixed motor arm composes states with the dual-quaternion/geometric product.
This arm has no trainable composition parameters. Its product is associative;
normalization is applied only for decoding and never inside the reduction.

The target for a window beginning at recorded pose `T_t` and ending at
`T_{t+L}` is the relative motion `T_t^{-1} T_{t+L}`, computed directly from the
two recorded endpoint poses. The input operands are the `L` consecutive
recorded relative motions between them.

## Frozen split

The independent unit is a complete named TUM trajectory. Assignment is
`int(SHA256("tum-rgbd-v1:" + sequence)[:8], 16) mod 10`: buckets 0--4 train,
5--6 development, and 7--9 confirmation. This yields 25/9/13 complete
trajectories. No window crosses a trajectory or split.
Confirmation trajectories are not evaluated without `--confirm`.

## Models and interventions

- `motor`: fixed associative dual-quaternion product.
- `bilinear`: unrestricted learned bilinear law on the eight motor coordinates.
- `mlp`: parameter-matched nonlinear binary composer on the same coordinates.
- `mixed`: the bilinear model trained uniformly across left, right, balanced,
  and sampled order-preserving trees.
- `penalty`: the left-trained bilinear model plus a local associator penalty on
  one consecutive recorded triple per batch.
- `decoupled`: invariant non-learned baseline that multiplies rotations but
  adds translations without the rotational action.

Learned intermediate states are projected to the unit-motor manifold after
each application so invalid scale cannot manufacture a failure. This projection
does not impose the geometric product or associativity. The bilinear class still
contains the exact motor law, and the nonlinear class can approximate it.

All learned arms train for 5,000 AdamW updates, batch size 256, at history
length 8. Seeds are 0--4. No confirmation choice may change the arms, seeds,
optimization, paths, lengths, metrics, or thresholds.

## Evaluation

Evaluate lengths 4, 8, 16, and 32 under left, right, balanced, and 16 sampled
order-preserving binary trees. Report:

- median and mean translation error in centimeters;
- median and mean rotation geodesic error in degrees;
- normalized eight-coordinate representation dispersion;
- paired path-minus-left changes;
- 95% trajectory-clustered bootstrap intervals (10,000 resamples).

The primary learned arm is `bilinear`; the primary alternate paths are right
and balanced at length 8. MLP is a nonlinear replication. Length 32 is the
primary extrapolation diagnostic.

## Development gate

Development passes only if all conditions hold:

1. bilinear and MLP left-fold median translation error are each below the
   decoupled baseline and below 10 cm at length 8;
2. bilinear right-minus-left and balanced-minus-left translation-error
   intervals both exclude zero in the harmful direction;
3. fixed motor composition changes by less than 1e-9 meters and 1e-9 radians
   across every evaluated tree in float64;
4. mixed-path training reduces both primary bilinear path gaps by at least 75%
   without increasing left-fold median translation error by more than 25%;
5. at least one unconstrained learned composer has a larger path gap at length
   32 than at length 8.

If development fails, confirmation remains unopened and the study is reported
as a negative result internally. No threshold may be relaxed. If development
passes, frozen checkpoints are evaluated once on confirmation. A confirmatory
claim requires both bilinear trajectory-clustered intervals to exclude zero in
the harmful direction and the motor tolerance to pass.
