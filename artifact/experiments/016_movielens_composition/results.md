# Results: MovieLens-1M evaluation-path composition

Status: five-seed development and sealed confirmation complete, 2026-09-03.

## Result in one sentence

On public, recorded MovieLens-1M histories, unconstrained composers learned
useful left-fold predictors but lost 6--15 accuracy points when the identical
events were reparenthesized; exact matrix composition changed no prediction,
and mixed-path training nearly eliminated the task-level gap.

## Data and protocol integrity

- Source: official MovieLens-1M archive, 1,000,209 ratings, 6,040 users, 3,706
  observed movies.
- Local archive MD5: `c4d9eecfca2ab87c1945afe126590906` (matches GroupLens).
- Local archive SHA-256:
  `a6898adb50b9ca05aa231689da44c217cb524e7ebd39d264c56e2832f2c54e20`.
- User-disjoint split fixed by SHA-256: 4,857 train / 596 development / 587
  confirmation users.
- Confirmation contained 93,992 observed rating endpoints at history length 8.
- Five training seeds; 3,000 optimizer steps per arm; width 9; history length 8.
- Bilinear and MLP operators have 747 and 727 composition parameters (2.7%
  difference). The exact arm has no learned composition parameters.
- Every tree preserves event order. The 16 random trees change only binary
  evaluation scheduling. No observation, label, negative example,
  transformation, or augmentation is generated.

The protocol and development decision were recorded in
`../../PROTOCOL.md` before confirmation was opened.

## Confirmation results

Accuracy is the mean across five independently trained seeds. Parenthesized
values are changes from each arm's own left fold. `Random` averages 16 sampled,
order-preserving binary trees.

| History | Arm | Left | Right | Balanced | Random |
|---:|---|---:|---:|---:|---:|
| 4 | Exact matrix | 0.4037 | 0.4037 (+0.0000) | 0.4037 (+0.0000) | 0.4037 (+0.0000) |
| 4 | Bilinear | 0.4180 | 0.3736 (-0.0444) | 0.3801 (-0.0379) | 0.3946 (-0.0234) |
| 4 | MLP | 0.4184 | 0.2942 (-0.1242) | 0.3423 (-0.0761) | 0.3390 (-0.0794) |
| 4 | Mixed-path | 0.4174 | 0.4168 (-0.0006) | 0.4174 (-0.0000) | 0.4172 (-0.0002) |
| 4 | Local penalty | 0.4187 | 0.3950 (-0.0237) | 0.3972 (-0.0215) | 0.4053 (-0.0134) |
| 8 | Exact matrix | 0.4175 | 0.4175 (+0.0000) | 0.4175 (+0.0000) | 0.4175 (+0.0000) |
| 8 | Bilinear | **0.4319** | 0.3482 (-0.0837) | 0.3711 (-0.0608) | 0.3881 (-0.0438) |
| 8 | MLP | **0.4325** | 0.2855 (-0.1470) | 0.3359 (-0.0966) | 0.3336 (-0.0989) |
| 8 | Mixed-path | 0.4272 | 0.4251 (-0.0021) | 0.4272 (-0.0000) | 0.4267 (-0.0005) |
| 8 | Local penalty | **0.4322** | 0.3722 (-0.0600) | 0.3904 (-0.0418) | 0.4026 (-0.0296) |
| 16 | Exact matrix | 0.4152 | 0.4152 (+0.0000) | 0.4152 (+0.0000) | 0.4152 (+0.0000) |
| 16 | Bilinear | **0.4332** | 0.3256 (-0.1076) | 0.3538 (-0.0794) | 0.3725 (-0.0607) |
| 16 | MLP | **0.4323** | 0.2701 (-0.1622) | 0.3138 (-0.1185) | 0.3172 (-0.1151) |
| 16 | Mixed-path | 0.4295 | 0.3949 (-0.0346) | 0.4220 (-0.0075) | 0.4231 (-0.0064) |
| 16 | Local penalty | **0.4344** | 0.3418 (-0.0926) | 0.3692 (-0.0652) | 0.3868 (-0.0476) |

The train-only per-movie mode baselines are 0.3995, 0.3992, and 0.3980 for
lengths 4, 8, and 16. The corresponding empirical-distribution NLLs are 1.3311,
1.3319, and 1.3346. Full accuracy, macro-F1, MAE, NLL, dispersion, runtime, and
per-seed values are in `results_confirmation.json`.

At length 8, the bilinear right-fold degradation ranges from 5.27 to 10.32
points across seeds; the balanced degradation ranges from 3.28 to 8.41 points.
Every seed's user-clustered 95% interval excludes zero for both comparisons.
The replication is therefore across independent optimization runs, not just
across correlated ratings.

## Exactness audit

The exact arm changes no predicted label under any tested tree in float32. Its
largest float32 absolute logit discrepancy is `3.10e-6` on development and
`2.86e-6` on confirmation. On confirmation, its largest float64 discrepancy across all examples, history
lengths, seeds, and tested trees is `6.22e-15`, far below the registered `1e-5`
tolerance.

## Interpretation

1. **Task competence does not imply path freedom.** The left-trained bilinear
   and MLP arms beat the strongest train-only accuracy baseline, including at
   twice the training history, but fail under alternative parenthesizations of
   the same histories.
2. **The failure grows with composition depth.** Bilinear right-fold loss grows
   from 4.44 points at length 4 to 10.76 points at length 16; the MLP loss grows
   from 12.42 to 16.22 points.
3. **Structural and training remedies are distinct.** Matrix multiplication is
   invariant to numerical tolerance without learning the law. Mixed-path
   exposure makes task predictions nearly invariant at the trained length but
   leaves a 3.46-point right-fold gap at twice the training history.
4. **A local penalty is insufficient here.** It reduces but does not remove the
   gap, and is consistently weaker than mixed-path exposure.
5. **Invariance and accuracy must remain separate claims.** The exact arm is
   invariant and beats the per-movie baseline at lengths 8 and 16, but its
   predictive accuracy remains below the learned fold-only arms.

## Development result

Development showed the same ordering before confirmation. At history length 8,
mean left accuracies were 0.4225 (exact), 0.4400 (bilinear), 0.4419 (MLP),
0.4361 (mixed-path), and 0.4403 (local penalty), against a 0.4117 per-movie
baseline. Bilinear right/balanced changes were -0.0872/-0.0639; MLP
-0.1494/-0.0999; mixed-path -0.0025/+0.0000; and local penalty
-0.0640/-0.0450. Full records are in `results_development.json`.

## Reproduction

```bash
.venv/bin/python experiments/016_movielens_composition/run.py
.venv/bin/python experiments/016_movielens_composition/run.py --confirm
```

The second command requires the frozen checkpoints and should not be rerun for
model or protocol selection.
