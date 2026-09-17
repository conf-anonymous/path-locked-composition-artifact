# KITTI GA composition experiment — scientific specification v1

Specified 2026-09-11, after the disclosed 401-frame sequence-00 frontend smoke
and training-reference structural checks, BEFORE full frontend extraction, new
GA fits, or development/holdout scientific access. This is a local prospective
specification; hashes prove subsequent file integrity, not independent public
preregistration. The software implementation of training and holdout release
must be completed, tested and separately sealed before training begins.

## 1. Question and evidentiary boundary

Can the existing learned PGA motor calibration and paired-product readout
improve relative outdoor motion estimates over a competent fixed stereo frontend
while retaining exact, order-preserving summary mergeability? This is
KITTI-trained architectural replication, not zero-shot transfer from ETH3D.
It does not by itself replicate the learned-binary-composer path-locking test.

The primary neural task is **segment-relative prediction**, not globally
consistent full-trajectory SLAM. Each head is evaluated on the same segment's
ordered recorded frontend increments, with a fresh recurrent context if any.
Distance boundaries follow KITTI conventions, but a head applied independently
to each segment does not necessarily yield the relative poses of one globally
coherent trajectory. Call neural results "KITTI-distance segment-relative
evaluation on a split of the public training set", not official KITTI benchmark
scores. Separately report native KITTI metrics for the raw frontend trajectory.
Do not hide this distinction by labeling all neural output official odometry.

No result is presumed positive. Existing Oxford failures and post-outcome
diagnostics stay recorded and excluded from positive evidence. The current
MovieLens/TUM/ETH3D manuscript evidence remains intact.

## 2. Public data and access stages

- Train 00–06; development 07–08; study holdout 09–10.
- KITTI official test 11–21 is outside scope. No inference or server submission.
- Fixed split and exact archive identities: `PREPARATION_PROTOCOL.md` and
  `data/raw/kitti_odometry/acquisition.json`.
- Public archive bytes were CRC-checked; only train poses/calibration/timestamps
  and sequence-00 smoke images have been scientifically inspected at specification.
- Train extraction/validation runs now. Development is opened only after the
  training implementation seal; full development predictions only after all
  final checkpoints are saved. Study holdout stays closed until Section 8.
- Distinct named raw drives do not prove geographically disjoint routes. The
  two study-held-out drives do not support broad population significance.

Only real recorded stereo images and supplied reference poses. No synthetic
observations, added noise, augmentation, interpolation, time warping, reference-
derived input motion or fabricated tracking repairs. Pixel decoding, estimated
motion, motor encoding and derived metrics are computations on recorded data.

## 3. Fixed frontend and failure semantics

LIBVISO2 archive SHA-256
`cbefe217840f136a12718b59bb57d43a3173694d5e5a5ed7704e2c1be2ac293d`.
Use the exact existing `stereo_stream_x86_64` binary and unmodified upstream
sources in `libviso2/build.json`. Rosetta, default parameters including 200
RANSAC iterations, upstream srand(0), and `replace=false` throughout. Only
documented per-sequence P0/P1 calibration varies; no tuning or alternate frontend.
Each sequence starts a fresh process. Use all consecutive recorded image pairs.

GetMotion maps previous-camera coordinates to current; invert it once to form
the forward-in-time composable motion. Pixel count, SHA, status, original motion,
match/inlier count and frame index are retained. First frame initializes, not
a failed edge. Failed edges have no motion; never reuse the stale getMotion,
insert identity, extrapolate, or bridge a gap. Every failure starts a new
connected component. All arms share exactly the same admissible windows.

Primary evaluation is conditional on valid continuous frontend input. Report
all potential window counts, admissible counts, failure locations and coverage;
do not call conditional error all-window end-to-end accuracy. No reference-error
screen based on model outcomes. Nonfinite/malformed poses or matrices block the
run, not silent exclusion. Existing numerical tolerances remain as in preparation.

## 4. Fixed data representations and windows

Use all contiguous valid training windows of 32,64,128,256 frontend edges,
starting at every recorded frame that fits inside a connected component. No
downsampling or resampling of observations. Sampling: choose length uniformly,
then uniformly among train sequences having at least one window at that length,
then uniformly among eligible starts for that sequence and length. A batch has
one sampled length and 64 windows. Record all eligibility counts, including zero.

Encode each inverse frontend transform as a unit PGA motor using Hamilton
quaternion order wxyz and dual part 0.5*(0,t)*q. Use experiment 021's documented
matrix-to-quaternion branch formula and normalize the quaternion. For each
individual leaf, fix the sign so its first nonzero quaternion coefficient is
positive (w first); fix it ONCE before any learned map or tree evaluation.
Never independently recanonicalize products at intermediate tree nodes. Reference
targets are inverse(T_0_start) @ T_0_end, encoded likewise; reference sign is
irrelevant to the sign-invariant loss. Motor algebra is experiment 017's product
and normalization, not a renamed unstructured binary network.

Training translation scale: median recorded endpoint displacement over all
eligible L128 training windows, floor 0.001 m. Freeze its value in a data receipt.
No scale fitted from development or held-out targets.

Primary evaluation windows: start every 10 frames; choose the first endpoint
whose supplied reference path distance exceeds start distance +100 m, matching
KITTI's float32 cumulative-distance convention. Reject only windows crossing a
failed frontend edge. All eligible 100 m windows are evaluated, with variable
recorded edge counts and no truncation. The head is given its ordered increments,
not target length, reference poses, reference speed or future reference features.

Secondary distance windows use 200,300,400,500,600,700,800 m by the same rule.
Also evaluate every 10th valid start at fixed edge lengths 32,64,128,256,512 for
scope and tree diagnostics. Report coverage, durations, edge counts and physical
lengths. Lengths beyond training support are explicitly extrapolation. Very short
reference windows are diagnostics, not the main utility endpoint.

## 5. Eight arms, six trained modules per seed

1. Identity motion.
2. Raw exact PGA product of frontend leaves.
3. Calibrated exact PGA product: experiment 020 `LeafCalibrator`, residual
   8→16→8 GELU MLP, last layer zero initialized, projected per leaf (280 parameters).
4. Constant sigmoid gate initialized at logit -2 between raw/calibrated products.
5. Contextual GA gate: experiment 028 `Head('contextual')`, bidirectional
   GRU(8,32), mean pooling, concatenated raw/calibrated endpoint products,
   80→32→1 GELU head; final layer zero weights, bias -2.
6. Feature-matched direct head: experiment 028 `Head('feature_direct')`, same
   GRU/features, 80→32→8 head, zero final weights and identity-motor bias.
7. Feature-matched residual head: experiment 028 `Head('feature_residual')`, same
   GRU/features, 80→32→8 zero-initialized output added to the fixed sigmoid(-2)
   blend before projection. Preserve this prior, do not describe it as identical
   initialization to direct regression.
8. Fully mergeable paired-motor gate: experiment 029 `MergeableGate`, 16→32→1
   GELU head, zero final weights and -2 bias (577 head parameters; 857 including
   leaf calibrator). Summary is raw/calibrated product pair; componentwise motor
   multiplication merges summaries. No GRU or hidden history in this arm.

The leaf calibrator is fit once per seed and frozen for all five heads of that
seed. Heads receive the same frozen products and batches where interfaces match.
Do not compare parameter totals without including the frozen leaf calibrator.
Report each arm's original input and recurrent/sequential access explicitly.

No standalone learned binary composer is added to this new KITTI experiment;
the existing MovieLens/TUM work supports that question. Regrouping fixed motor
products while keeping a GRU context unchanged is not full GRU mergeability.

## 6. Training, checkpoints and implementation freeze

Five seeds s=0..4. CPU float32, one torch thread, one fit at a time. Set torch
deterministic algorithms true. Calibrator initialization torch seed s; head
initialization 50000+s separately for every arm. Sampling generator 30000+s
reset at the start of each module's fit, ensuring identical task batches across
heads and corresponding training windows across modules. Data sampling uses
ordered length/sequence/start inventories; its implementation must be sealed.

Each of six modules: exactly 5,000 AdamW updates, batch 64, weight decay 1e-4,
betas (0.9,0.999), eps 1e-8, clip gradient norm 1. Calibrator LR .002; head LR
.001. No scheduler, dropout, early stopping, validation-selected checkpoint,
additional restarts, or hyperparameter sweep. Balanced product reduction is used
in training; every learned leaf remains the same under later regrouping.

Loss is experiment 022's robust_motor_loss: coordinatewise SmoothL1(beta=1) of
decoded translation divided by the fixed training scale, sum xyz, plus
2*(1-clamp(dot(q_pred,q_target)^2,max=1)); batch mean. Reuse its exact sign-
invariant formula. Heads use frozen calibrated products; only their own trainable
parameters update. Final iteration checkpoint is the sole evaluation checkpoint.
Nonfinite loss/gradient/output is a recorded failure, not grounds to rerun until
positive. Store all seeds and failed runs. No single best-seed reporting.

Before any fits: implement/test the dataset and sampler, heads, metrics, failure
coverage, model audits and fail-closed holdout wrapper on train data; pin all
source dependencies, environment/configuration, raw and derived hashes. The
present scientific seal does not falsely claim those runners already exist.

## 7. Metrics, uncertainty and success claims

For a segment target G and predicted motor decoded as P, use E=inverse(P)*G.
Translation is 100*norm(E.translation)/nominal_segment_length (percentage points
for paired differences); rotation is angle(E.rotation) in degrees divided by
nominal length. Report unnormalized meters/degrees alongside them. Raw trajectory
metrics additionally use the original native evaluator on connected components,
preserving global segment starts and reporting gaps; never insert missing poses.

Primary: 100 m segment-relative translation error. Within each sequence average
eligible windows; average five seeds within sequence; give sequences equal
weight. Also retain all per-seed/per-sequence values and pooled-window means
as secondary descriptions. Primary contrasts: mergeable minus raw, identity,
feature_direct, feature_residual. Constant/contextual/calibrated are reported
comparators, not hidden if they win. All secondary distances, rotation, runtime,
projection norms, and failure coverage are reported regardless of direction.

With two development and two study-held-out sequences, do NOT claim a
well-powered population test, bootstrap overlapping windows as independent, or
manufacture significant p-values. Report exact paired changes for both sequences,
five-seed mean/range, and the aggregate. No formal population CI is primary.

The prespecified descriptive support criterion on the study holdout is:
mergeable improves mean primary error by >=5% relative to each of the four
primary comparators; mean-seed change has the improving direction on both
holdout sequences; and at least 4/5 seeds improve the equal-sequence average
against each comparator. This is a descriptive acceptance criterion, not proof
of statistical significance or universal superiority. Report any failure to
meet it. For an overall positive utility interpretation, also require mergeable
mean 100 m rotation error <=1.10 times raw; otherwise disclose the tradeoff and
do not headline an unqualified improvement. No adaptive change of primary length.

Numerical invariant audit in float64: left/right/balanced and 16 ordered random
trees (Python Random seeded 330911+k, k=0..15), plus three contiguous chunks
with cuts floor(L/3), floor(2L/3), every fixed diagnostic length on the first
20 eligible starts per sequence. Same leaves/checkpoint/readout throughout.
For raw, calibrated and fully mergeable arms require maximum decoded path
discrepancy <1e-8 m and <1e-9 rad; report coordinate discrepancies and minimum
real norm before projection. Values <=1e-6 real norm are a projection-domain
failure, not made safe by the implementation's normalization clamp. This is
tested precision, not bitwise exactness or a new associativity theorem.

Audit the same fixed calibrated leaves through homogeneous SE(3) composition
and PGA composition on those windows. Agreement is a coordinate consistency
check, not GA superiority over an equivalent representation. No separate fitted
matrix model is smuggled in as an "identical" control.

## 8. Frontend validation and holdout release

Full training frontend admission, fixed before full outputs: every training
sequence has finite valid motions and verified frame/calibration identities,
>=95% valid noninitial edges, >=80% coverage of all potential 100 m windows,
and at least one valid 100 m window. Raw mean 100 m translation <10% and rotation
<0.2 deg/m on each sequence, and raw primary error beats identity on each.
These generous engineering checks reject a grossly failing frontend; they do
not prove state of the art. Failing a sequence blocks the campaign; do not drop
it, tune the frontend on its outcomes, or silently alter the criteria.

Apply the same frontend checks to both development sequences once authorized by
the implementation seal. Before holdout release require these checks, all 30
final fits/checkpoints, complete eight-arm development reports, finite outputs,
valid projection domains, tested mergeability/coordinate audits, immutable
source/data provenance, and an implemented/tested release manifest naming every
arm and seed. A runtime/resource limitation requires an explicit amendment
before development outcomes, not an unrecorded smaller experiment.

**Do not gate holdout access on a positive GA development effect.** If technical
and frontend checks pass, evaluate all prespecified models on 09–10 once, even
if GA development utility is negative. No checkpoint selection, seed selection
or post-holdout tuning. A failed holdout tracking/data check is reported; no
substitute sequence. Any new hypothesis after outcomes is a new development
study, not a repaired original success. KITTI official test stays excluded.

## 9. Deferred work and handoff

Current authorization is protocol specification and full TRAIN frontend
validation before model fitting. Scientific protocol, source implementation
seal, frontend admission, complete fits, and holdout release are distinct states.
Do not label any later stage complete from the existence of this document.
Manuscript/artifact consolidation and final adversarial audit remain authorized
separate work; KITTI gains cannot enter the manuscript until actually supported.
