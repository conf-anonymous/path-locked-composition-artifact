# Frozen public real-data protocol

Recorded: 2026-09-03

## Dataset and task

Use only the official public MovieLens-1M `ratings.dat` observations. Predict
the next recorded 1--5 star rating from the preceding recorded movie/rating
events and the queried movie. Do not generate observations, labels, negative
examples, transformations, or augmentations.

Verify the archive MD5 as `c4d9eecfca2ab87c1945afe126590906`. Sort events
within user by timestamp and use source-file order as the deterministic tie
break. Assign users, not endpoints, by SHA-256: buckets 0--7 train, 8
development, and 9 untouched confirmation. Train at history length 8 and
evaluate lengths 4, 8, and 16.

## Frozen arms

1. Exact 3x3 matrix multiplication.
2. Free bilinear composition trained through the left fold.
3. Parameter-matched nonlinear MLP trained through the left fold.
4. Bilinear composition trained across left, right, balanced, and sampled
   order-preserving trees.
5. Left-trained bilinear composition with a local squared associator penalty on
   consecutive triples from recorded histories.

All arms share width, embeddings, rating head, optimizer, task batches, and
step budget where their types permit. The exact rule has no learned composition
parameters. The bilinear and MLP rules must be within 5% in composition
parameter count.

## Frozen evaluation

Evaluate each checkpoint on the identical recorded history under left fold,
right fold, balanced tree, and 16 independently seeded order-preserving random
tree schedules. Report accuracy, macro-F1, mean absolute rating error, negative
log likelihood, paired accuracy changes, representation dispersion, and
runtime at each history length. Use five independent training seeds.

Confidence intervals resample whole users with all their endpoints. For the
exact arm, report maximum absolute logit differences in float32 and float64;
the predeclared float64 tolerance is `1e-5`.

Baselines use training observations only: global rating mode, per-movie rating
mode with global fallback, and the per-movie empirical rating distribution.

## Development gate and confirmation decision

Confirmation is permitted only if a learned arm beats the strongest train-only
development baseline, degrades materially on at least two alternative tree
families with user-clustered intervals excluding zero, and the exact arm passes
its invariance tolerance.

All five development seeds passed. At history length 8, bilinear left-fold
accuracy was 0.4400 versus a 0.4117 per-movie baseline; right and balanced
changes were -0.0872 and -0.0639. Every seed's interval excluded zero. The
exact arm changed no prediction and its worst float64 discrepancy was
`5.78e-15`. This decision was recorded before confirmation labels were opened,
without changing arms, seeds, lengths, paths, metrics, or optimization.

Confirmation then replicated the result: bilinear left-fold accuracy 0.4319
versus baseline 0.3992, with right/balanced changes -0.0837/-0.0608; exact
float64 discrepancy at most `6.22e-15`; and mixed-path changes
-0.0021/approximately zero.

## TUM breadth study and confirmation

The complete, immutable protocol history is preserved in Experiments 017--019.
Only the 47 official TUM RGB-D motion-capture ground-truth trajectories are
eligible. Complete trajectories are hash-split 25/9/13; the first recorded pose
at or after each 0.1-second grid point is selected without interpolation.
Relative rigid motions are unit dual quaternions, equivalently motors in the
even subalgebra of 3D projective geometric algebra. The exact arm is the motor
geometric product.

The unrestricted pilot (017) and the original residual gate (018) failed on
development, and confirmation remained unopened. A 3.2-second hypothesis was
then selected from development, frozen in Experiment 019, and evaluated once
using unchanged Experiment 018 checkpoints. This is confirmation-frozen, not
claimed as preregistered before development.

All five conditions passed on 13 untouched trajectories and 6,102 length-32
windows. The decoupled baseline's median translation error was 14.28 cm. The
bilinear residual achieved 6.83 cm on its trained left fold and 36.53 cm under
the right fold; all five trajectory-clustered intervals excluded zero. Mixed
paths achieved 5.22/4.58 cm, while the local penalty achieved 6.86/33.83 cm and
all five intervals remained harmful. The exact GA motor's worst decoded path
discrepancy across every tested length and tree was `2.28e-15` m.
