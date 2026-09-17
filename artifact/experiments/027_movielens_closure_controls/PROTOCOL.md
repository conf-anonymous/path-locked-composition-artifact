# Recorded MovieLens regularization follow-up (2026-09-06)

Authorized after adversarial review; the original test outcomes are already
known. This is a post-hoc benchmark extension, not a new untouched confirmation.
Original Experiment 016 files and checkpoints are immutable.

Only the checksum-verified MovieLens-1M archive is used, with its existing user
partitions, chronology, labels and training budget. No generated observations,
perturbations, augmented ratings or simulated data are permitted.

Compare three regularizers on the same bilinear-residual composer: primitive
coordinate MSE, multilevel intermediate coordinate MSE, and multilevel squared
cosine distance (inspired by Pert et al., Appendix G; not an exact reproduction
of their architecture). At each multilevel reduction stage, include every
consecutive triple and then reduce adjacent pairs, retaining an odd last state.
Average triples within levels, then levels. Primitive uses every consecutive
primitive triple, improving coverage over the original one-triple control.

For each regularizer sweep weights 0.1, 1, 10 using development seed 0 only.
Select the largest mean development accuracy across left, right, balanced,
history-right/query-last and history-balanced/query-last; ties prefer the
smaller weight. No test outcome enters selection. Train the selected weight
at seeds 1--4; report all five seeds at lengths 8 and 16. This is a small
development search, not an exhaustive regularization optimum. Report accuracy,
primitive/internal associators and measured training compute. Retain all
candidate development outcomes. Fixed 3,000 updates, batch 256, original AdamW.

Add recorded-history empirical baselines: preceding-history rating mode and
a mixture of train-only movie rating frequencies with the observed history's
rating frequencies. Select mixture weight from {0,.25,.5,.75,1} on dev only.
These simple controls assess history utility, not recommender state of the art.

Freeze source/protocol hashes before the first fit; save selected settings
before test evaluation. Hashes are local integrity records, not external
preregistration. Preserve unfavorable as well as favorable outcomes.
