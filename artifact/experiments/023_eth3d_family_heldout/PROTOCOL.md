# Frozen protocol: family-held-out ETH3D motor correction

Frozen 2026-09-05 before any model was trained or evaluated under the grouped
folds below. ETH3D outcomes from Experiments 021 and 022 were already known;
this is therefore a prospectively specified robustness study, not a recovery of
an untouched ETH3D test set. No architecture or threshold may be selected from
the outer-fold outcomes.

## Question

Does contextual gating of exact PGA motor calibration generalize to entirely
unseen ETH3D capture families, rather than merely to new sequences from capture
families represented during training?

## Public real data

Use exactly the 56 eligible public real ETH3D motion-capture recordings and the
hash-verified Open3D 0.19 RGB-D odometry outputs from Experiment 021. No
synthetic observation, motion, pose, perturbation, noise, augmentation, or
failure replacement is permitted. Ground truth is used only for training
targets inside the training folds and evaluation inside the held-out fold.

## Immutable family map

Family membership follows the semantic capture prefix in the official sequence
names, with condition suffixes retained in the same family:

- `cables`: `cables_*`
- `camera_shake`: `camera_shake_*`
- `ceiling`: `ceiling_*`
- `desk`: `desk_3`, `desk_changing_1`
- `einstein`: every `einstein*` sequence
- `kidnap`: `kidnap_1`, `kidnap_dark`
- `large_loop`: `large_loop_1`
- `mannequin`: every `mannequin*` sequence
- `motion`: `motion_1`
- `planar`: `planar_*`
- `plant`: every `plant*` sequence
- `reflective`: `reflective_1`
- `repetitive`: `repetitive`
- `sofa`: every `sofa*` sequence
- `table`: `table_*`
- `vicon_light`: `vicon_light_*`

Sort families by decreasing number of named sequences, breaking ties
lexicographically, and greedily assign each to the fold with the fewest assigned
sequences, breaking fold ties by lower fold number. This produces:

- fold 0: `mannequin`, `kidnap`, `repetitive` (12 sequences);
- fold 1: `plant`, `planar` (11 sequences);
- fold 2: `sofa`, `desk`, `motion` (11 sequences);
- fold 3: `einstein`, `ceiling`, `vicon_light` (11 sequences);
- fold 4: `cables`, `camera_shake`, `table`, `large_loop`, `reflective`
  (11 sequences).

For a fold, all sequences in its families are evaluation-only and all remaining
families are training-only. A sequence with no contiguous window at a requested
horizon contributes no fabricated or imputed window; its absence and the count
of evaluable sequences are reported. Every family must have at least one
evaluable 32-leaf sequence.

## Frozen methods

`raw_motor` reduces recorded odometry leaves exclusively with the exact PGA
motor product.

`ga_calibrated` trains a shared zero-initialized 8-16-8 leaf residual at eight
leaves for 5,000 AdamW updates, using the Experiment 021 endpoint motor loss,
uniform training-window sampling, batch size 256, learning rate 2e-3, weight
decay 1e-4, and gradient clipping at one. Calibrated leaves are reduced only by
the exact motor product.

`contextual_ga` freezes that fold-and-seed calibrator. A bidirectional GRU with
32 hidden units per direction and a 32-unit GELU head observes only the ordered
input motors and emits a scalar gate between raw and calibrated exact-product
motors. Train at 32 leaves for 5,000 updates with sequence-balanced sampling,
batch size 256, learning rate 1e-3, weight decay 1e-4, the Experiment 022 robust
motor loss, and gradient clipping at one.

`direct_gru` is a parameter-matched non-compositional control: the same
bidirectional GRU directly predicts an endpoint unit motor through a 38-unit
GELU head. It has no exact motor-product backbone. Train it with the identical
32-leaf batches, robust loss, optimizer, updates, and seeds as `contextual_ga`.
The trainable parameter counts of `contextual_ga` and `direct_gru` must differ
by less than 5%; the separately trained 280-parameter leaf calibrator is also
reported.

Use seeds 0--4. Evaluate all methods at 4, 8, 16, and 32 leaves. Length 32 is
primary. Report window metrics, sequence means, family-equal means, per-family
results, rotation errors, runtime, front-end failures, and float64 exactness.

## Frozen family-level analysis

Each of the 16 families appears in exactly one outer fold. Within a family,
average windows within sequence and then sequences within family. Primary
uncertainty uses 10,000 bootstrap resamples of the 16 complete families, with
all five matched seeds retained inside every resample; RNG seed 230905.

The structured method earns a definitive positive ETH3D family-generalization
result only if every condition holds at 32 leaves:

1. every contextual-GA seed improves the family-equal translation mean over
   raw, and the across-seed reduction is at least 10%;
2. the pooled family-bootstrap 95% interval for contextual-GA minus raw lies
   strictly below zero;
3. at least 10 of 16 families improve after averaging seeds, the median family
   delta is below zero, and the mean delta remains below zero after removing
   any single family;
4. the pooled family-bootstrap 95% interval for contextual-GA minus the
   parameter-matched direct GRU lies strictly below zero;
5. family-equal rotation error does not exceed raw by more than 10%;
6. every exact-product arm has worst left/right/balanced prediction discrepancy
   below 1e-9 in float64 at every horizon and seed; and
7. reporting is complete, every family is represented at length 32, and the
   parameter-count match is valid.

Shorter horizons, window-weighted metrics, individual folds, or a subset of
families cannot rescue a failed primary condition. Engineering smoke tests may
use fewer steps and seeds only to correct code defects; their outcomes cannot
change this protocol.
