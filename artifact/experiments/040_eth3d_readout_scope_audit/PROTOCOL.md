# Frozen ETH3D readout and matched-horizon audit

Specified 2026-09-16 after inspection of the original results. This is a post-hoc
sensitivity analysis, not new confirmation, model selection, or retraining.

## Fixed inputs

Use only the unchanged public real ETH3D recordings and derived Open3D motions
admitted by Experiment 023. Preserve its folds, all five seeds, the Experiment
028 calibrators and heads, and the Experiment 029 mergeable heads. Check
checkpoint hashes and reproduce all eight original arms against Experiment 030
at lengths 4, 8, 16, 32, 64 and 128. No generated observations, perturbations,
augmentations, new labels, parameter fitting, or held-out tuning are permitted.

## Readout sensitivity

Retain the original predictor. Separately evaluate a sign-aligned readout for
the contextual and mergeable gates. Compute alpha from the original input
coordinates and fixed weights. Only in the final interpolation, replace the
calibrated endpoint by its negative when the raw/calibrated real-quaternion dot
product is negative; keep the sign positive at a zero dot product. This negates
all eight motor coordinates. Do not canonicalize intermediate products.

This isolates output sign alignment without claiming that the gate is invariant
to arbitrary changes of input representatives. It is not a newly trained model.
Report both metrics and all outcomes at every horizon, sign-disagreement counts,
projection norms, and the finite-data lower-bound check. For unit real parts and
alpha in [0,1], alignment implies a real-blend norm of at least 1/sqrt(2).
For finite-precision, possibly non-unit inputs, check the corresponding lower
bound min(norm(q_raw), norm(q_cal))/sqrt(2). This bound concerns the final blend,
not the leaf calibrator, global accuracy, or full gauge invariance.

Audit float64 left, right, balanced and three-chunk predictions for both aligned
readouts on the first 20 eligible starts per sequence/fold/seed/horizon (all if
fewer). Retain maximum translation/rotation discrepancies and sign-branch
changes; do not describe the finite audit as exhaustive enumeration.

## Matched horizon populations

Report three populations, without choosing between them by performance:

1. Full coverage: exactly the original Experiment 030 windows at each horizon.
2. Common sequences: only sequences supporting length 128; retain all originally
   eligible windows at each shorter horizon. This fixes sequence/family membership
   but not the distribution of within-sequence starting positions.
3. Matched starts: use exactly the (sequence, admitted contiguous run, start)
   indices supporting length 128 for every horizon. Shorter windows are prefixes
   of these same recorded runs. Reference endpoints change with horizon in the
   usual way, but no reference or input is generated. This is a robustness check
   conditional on surviving 128-step runs, not a population-wide length claim.

Aggregate windows within sequence, sequences within family, then average families
and five seeds equally. Retain all original eight arms plus both readout
sensitivities. Report conditional family-bootstrap intervals with 10,000 draws
and seed 160926. These condition on frozen models, have no multiplicity correction,
and do not remove training-fold dependence or create independent confirmation.

## Integrity and decision boundary

Seal this protocol, analysis source, imported local implementation files,
input manifest/derived recordings, checkpoint files and original record files
before evaluation. Verify these inputs again at completion. Write new records
only to this experiment. Refuse to overwrite original studies or an existing
incompatible run. Numerical assertions test reproduction and implementation,
not whether a variant improves accuracy. Retain unfavorable results. Any later
decision to train or adopt another architecture requires a separately specified
experiment and cannot inherit the original predictor's results.
