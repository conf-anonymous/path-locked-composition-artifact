# Confirmation-frozen protocol: long-horizon residual motor composition

Frozen on 2026-09-03 after Experiments 017 and 018 development, and before any
TUM confirmation trajectory was evaluated. This is not claimed as
development-preregistered. It is a development-selected hypothesis tested once
on the untouched 13-trajectory confirmation split.

## Development selection

Experiment 018 introduced learned corrections over a strong invariant
decoupled motion baseline. Its original gate failed because it required every
seed to beat the baseline already at the trained 0.8-second horizon and imposed
a balanced-tree effect there. The registered long-horizon observation is more
scientifically relevant to deployment: at length 32 (3.2 seconds), every one of
the five residual-bilinear left folds beat the decoupled baseline, while every
right-fold path gap had a trajectory-clustered interval excluding zero. Mixed
paths retained the accuracy gain and removed the gap; the local associator
penalty did not.

## Immutable inputs

Experiment 019 reuses the exact frozen Experiment 018 checkpoints. It performs
no training, model selection, threshold tuning, or data transformation. Data,
25/9/13 whole-trajectory split, source checksums, 0.1-second recorded-row
selection, arms, seeds, and metrics are unchanged. Only the untouched
confirmation split is evaluated.

Primary endpoint: length 32. Primary learned arm: residual bilinear. Primary
counterfactual path: right fold versus left fold on the identical 32 recorded
relative motions. The balanced and random trees, MLP, and lengths 4/8/16 are
secondary descriptive results.

## Confirmation gate

The hypothesis confirms only if all conditions hold on the 13 complete
confirmation trajectories:

1. every residual-bilinear seed has lower left-fold median translation error
   than the fixed decoupled baseline at length 32;
2. every residual-bilinear seed's trajectory-clustered 95% interval for
   right-minus-left mean translation error lies strictly above zero;
3. the fixed GA motor product stays below 1e-9 meters/radians across every tree
   and length;
4. every mixed-path seed has lower left-fold median error than the decoupled
   baseline at length 32, and the absolute mean mixed right-fold gap is at most
   25% of the residual-bilinear gap;
5. every locally penalized seed beats the decoupled baseline on its left fold
   yet has a right-minus-left interval strictly above zero.

No partial confirmatory claim is permitted if this conjunction fails. The
complete results remain reported internally regardless of outcome.

