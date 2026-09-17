# Frozen protocol: residual learning over an invariant motion baseline

Frozen on 2026-09-03 after Experiment 017 development failed its competence and
mixed-path gates, and before any TUM confirmation trajectory was evaluated.
Experiment 017 remains an immutable failed pilot. Its development result showed
that a free bilinear law could fit the length-eight left fold but was worse than
the strong decoupled baseline, while alternative trees failed by meters.

Experiment 018 changes the learned parameterization, not the data, split,
targets, metrics, confirmation set, or standards. Every learned law begins with
the invariant decoupled operation

`(q1,t1) o (q2,t2) = (q1 q2, t1 + t2)`

and learns an eight-coordinate residual before projection to a unit motor. The
base is associative but physically incomplete because it omits rotation of the
second translation by the first motion. A zero-initialized residual reproduces
the baseline exactly. A learned residual can improve endpoint reconstruction,
but is not constrained to remain associative. This isolates whether a useful
learned correction introduces path-locking.

The exact arm remains the dual-quaternion motor/geometric product, equivalent
to composition in the even subalgebra of 3D projective geometric algebra. It
contains the required semidirect coupling and is associative by construction.

## Data and split

All provenance, duplicate-timestamp handling, recorded-row selection, 0.1 s
sampling, window construction, and the deterministic 25/9/13 whole-trajectory
train/development/confirmation split are identical to Experiment 017. No
synthetic or private data are used. Confirmation remains unopened.

## Arms and optimization

- `motor`: fixed GA motor product;
- `decoupled`: fixed associative but physically incomplete baseline;
- `bilinear`: decoupled base plus a learned bilinear residual;
- `mlp`: decoupled base plus a parameter-matched nonlinear residual;
- `mixed`: bilinear residual trained uniformly on left, right, balanced, and
  sampled order-preserving trees;
- `penalty`: left-trained bilinear residual plus the same local associator loss.

Residual output layers are initialized to zero. All other optimization,
five-seed, path, length, bootstrap, and metric settings are unchanged from
Experiment 017.

## Development gate

Development passes only if:

1. every bilinear seed and at least four of five MLP seeds improve the fixed
   decoupled baseline's median translation error at length 8 on the left fold;
2. every bilinear seed's trajectory-clustered 95% interval excludes zero in the
   harmful direction for right-minus-left and balanced-minus-left translation
   error at length 8;
3. the fixed motor stays below 1e-9 meters/radians across all tested trees;
4. mixed training reduces both mean primary bilinear path gaps by at least 75%,
   while its mean left-fold median error remains below the decoupled baseline;
5. at least one learned arm has a larger primary path gap at length 32 than at
   length 8.

If this gate fails, confirmation remains unopened. If it passes, confirmation
is evaluated once from the frozen checkpoints. Confirmatory claims require
conditions 1--4 to replicate on confirmation, with no threshold changes.

