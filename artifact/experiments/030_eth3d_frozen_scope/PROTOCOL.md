# Frozen-checkpoint scope audit (2026-09-06)

Post-hoc robustness, NOT independent confirmation. Experiment 028/029 outcomes
at length 32 are known. Freeze this protocol, implementation and all source
checkpoints before inspecting this audit's additional outcomes.

Use only the original checksum-verified ETH3D odometry and recorded references.
Reuse Experiment 023's exact 16-family/five-fold partition and all five seeds.
Use all valid windows at lengths 4, 8, 16, 32, 64, 128, selected from the same
contiguous runs of at least 32 valid increments admitted by Experiment 023.
Never bridge failed odometry pairs. Targets remain the relative pose between
recorded endpoints. Longer windows are additional slices of recordings, not
generated observations. Preserve original eligibility at shorter horizons.

Frozen arms: identity, raw motor, calibrated motor, Experiment 028 constant,
contextual, feature_direct and feature_residual, and Experiment 029 mergeable.
No training, checkpoint choice, loss reweighting or family-dependent routing.
Use float32 inference as in the original utility evaluation; errors are decoded
in float64. Report minimum pre-projection real norm for every learned head.

Report all per-sequence metrics and family-equal translation/rotation means,
averaged over all five seeds; all methods share exactly the same windows at a
given horizon. Report sequence/family coverage and actual timestamp durations.
Missing long-horizon families must be disclosed rather than imputed. Across-
horizon means may have differing coverage and are not paired length effects.

Compute paired mergeable-minus-comparator contrasts using 10,000 bootstrap
resamples of the available families (seed 300906), conditional on fixed models,
families and overlapping folds. These exploratory intervals are not corrected
for the multiple arms/horizons, nor are they independent-population tests.
Report the mean translation/rotation Pareto frontier at each horizon. This is
a descriptive tradeoff, not a combined metric or a significance test.

Require reproduction of existing L32 sequence means to within 1e-5 in meters
and degrees for all learned heads and all 25 checkpoints. Checkpoint hashes
must match original immutable run records. Fail on nonfinite output, not on
unfavorable utility. Write every horizon and arm before scientific conclusions.

Seal is local integrity evidence, not independent preregistration. Preserve
Experiments 020 and 023--029 without edits. Do not modify the model after seeing
these results; any future model would be a separately disclosed development.
