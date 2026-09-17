# Oxford architectural confirmation extension

Specified 2026-09-06, before Oxford acquisition or outcome inspection. This is
a SEPARATE prospective study of the architectures introduced in Experiments
028--029. It does not amend, supersede or bypass Experiment 020's seal/gate.
The ETH3D scope results of Experiment 030 are already known when specifying
this extension; they are development context, not external confirmation.

## Data and boundary

Use exactly Experiment 020's official VO/RTK/extrinsics, deterministic eligible
traversals, recorded-row alignment, whole-traversal train/dev/confirmation split,
and lengths 4, 8, 16, 32. No alternate front end, interpolation, augmentation,
generated observations or selection of weather/routes. Primary horizon remains
L32 (nominal 16 seconds); all-window analysis is primary, displacement >=5 m
is a fixed secondary subset. This is Oxford-trained replication, not zero-shot
transfer of ETH3D weights. Repeated traversals do not establish new-city transfer.

No Oxford observations are locally available at specification. A local seal
records integrity only, not an externally timestamped preregistration.

## Fixed architecture and training

Arms: identity motion, raw exact motor product, the original Experiment 020
calibrated motor product, constant gate, contextual gate, feature-matched direct
head, feature-matched residual head, and GRU-free mergeable gate.

Reuse each of Experiment 020's five final ga_calibrated leaf calibrators,
matching its checkpoint hash and original configuration (including any uniformly
authorized development LR reduction); freeze it for ALL heads of that seed.
Train all five heads from scratch on Oxford training traversals at L8, not
on L32 endpoints. This is a length-extrapolation test and differs from the
ETH3D heads' L32 training. All heads use 5,000 AdamW updates, batch size 256,
LR .001, weight decay .0001, gradient clipping 1, and the existing robust
motor loss in Experiment 022 with translation scale equal to the median
training-L8 endpoint displacement (floor .001). No loss reweighting or tuning.

Architecture/initialization are exactly Experiment 028 Head and Experiment 029
MergeableGate: 32-wide bidirectional GRU where applicable, same endpoint feature
pair, same initialization and output-prior differences. Head RNG seed 50000+s;
training sampler RNG seed 30000+s; s=0..4. Uniformly sample traversal then a
recorded training window, using the same batches for every head of a seed.
All head fits are complete before evaluating development utility. No head or
seed may be dropped. No development-selected ensemble, horizon or checkpoint.

## Execution and shared confirmation discipline

The protocol and exact existing model dependencies are locally sealed now.
The Oxford-specific runner remains to be implemented and must be tested on
public recorded ETH3D inputs and sealed BEFORE Oxford outcome inspection.
No synthetic mock data may be used. A schema-only correction must be logged
before outcomes; a scientific change requires an explicitly new protocol.

Train/freeze ALL original and extension checkpoints, configuration, code and
input hashes; store ALL extension development outcomes before any confirmation
call. A shared confirmation release manifest must identify both campaigns.
Original Experiment 020's complete development gate must pass; otherwise the
shared confirmation traversals remain unopened for both studies. Extension
performance does not decide which arms are confirmed: all are evaluated once
if original gate and technical completeness allow release. Technical completeness
means all five seeds, every arm/horizon, finite outputs, provenance verified,
and checkpoints immutable. Failure of any technical check blocks release.

The existing Experiment 020 command alone does not yet enforce this extension's
release manifest. Until the joint release wrapper has been implemented and
verified, DO NOT run either confirmation command. This is a protocol-ready
extension, not a claim of an implemented or executed confirmation system.

## Analysis, success conditions and transparent reporting

Report per-traversal means for translation (meters) and rotation (degrees),
average seeds within traversal, then give equal traversal weight. Use 10,000
whole-traversal paired bootstrap draws, RNG seed 310906; retain per-seed and
per-traversal results, moving subset and all horizons. Overlapping windows are
not independent samples; uncertainty concerns traversals under this route and
acquisition setup, not independent geographies.

Primary utility comparisons: mergeable minus (raw, identity, feature_direct,
feature_residual) in L32 mean translation. Report ordinary 95% intervals plus
98.75% intervals for the four prespecified comparisons (Bonferroni familywise
95% convention; bootstrap coverage is approximate). Strong architectural
confirmation requires ALL four adjusted interval upper bounds <0 and
translation improvement in at least four of five seeds for each comparator.
No arbitrary rescaling or joint translation/rotation score is introduced.

Rotation, other horizons and subsets are reported regardless of direction,
with exploratory 95% intervals clearly distinguished from primary comparisons.
No non-inferiority or equivalence claim against the contextual gate is planned.

Separately audit float64 complete paired-summary regrouping over left/right/
balanced/16 seeded ordered random trees at every length: maximum decoded
translation and rotation discrepancy <1e-9 m and <1e-9 rad, respectively.
Audit full-state coordinates, finiteness and minimum un-clamped real norm;
do not present observed projection safety as a global theorem.

Archive and disclose ALL development and confirmation outcomes, including
negative ones. Failure is not grounds to suppress the study or choose another
horizon. A primary claim of external architectural confirmation requires the
prespecified conditions on untouched confirmation, not merely development.
If original gate blocks confirmation, report the extension as development-only
and explicitly state that no external confirmation claim was established.
