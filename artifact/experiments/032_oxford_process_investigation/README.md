# Oxford process investigation — 10 September 2026

**Latest: multi-date INS check completed.** All three additional GPS archives
match official MD5s. Across four dates, 90,959 recorded pairs (85,088 good-status)
replicate roll sign reversal, pitch agreement, and approximately 90-degree yaw
offset. Median roll sums span only 1.839509–1.841635 degrees; the fixed sign/yaw
probe still leaves 1.868–1.884 degrees of attitude disagreement. The relationship
is systematic, but a full physical frame conversion remains unvalidated.
See [`ins_multidate_check/README.md`](ins_multidate_check/README.md). No model or
confirmation evaluation was performed; no further GPS downloads are needed for
this cross-date check. Earlier download requests below are historical.

**Later INS cross-check completed:** the requested original GPS archive confirms
the yaw discrepancy and reveals near-perfect RTK/INS roll anticorrelation.
The previous quarter-turn is not a validated full attitude conversion. See
[`ins_frame_check/README.md`](ins_frame_check/README.md) for the 11,930-pair audit,
good-status and chronological-half checks, residual offsets, and the next three
small GPS archives requested for cross-traversal validation. No models were
refitted or original results changed during this cross-check.

**Status: diagnostic work, not submission evidence.** Requested after the
original Oxford development run and first reference audit. Experiments 020/031
remain unchanged and excluded; the nine confirmation traversals are unopened.

## Findings established independently of new model performance

### 1. Correct archive identities; no local extraction corruption found

All 48 training/development VO archives match the official MD5 values recorded
at acquisition. Every extracted VO CSV matches its archive member byte-for-byte.
The 48 RTK CSVs also match their original ZIP members exactly. The two RTK ZIPs
in Downloads have identical SHA256, matching the staged archive:
`a6bb10488c8e5c692f0dbbefc3f79c93ab4afa731b7a2c72e33b4e987e5e0ff8`.

The files are the intended public stereo-VO inputs and independent RTK targets,
not the 2019 radar benchmark or poses generated from RTK. Image archives are not
needed for this released-VO composition experiment.

Evidence: `archive_recheck.json`, `quality_screened/source_quality.json`, original
acquisition receipt and official traversal pages linked in the former report.

### 2. Frame contract is not physically validated

Across the 14 development traversals, median RTK yaw minus the heading of its
own recorded horizontal velocity is 88.683–90.395 degrees (speed >5 m/s).
This does not use VO predictions or a trained model. Position-derived heading
also differs from velocity heading by approximately -1.30 to -1.38 degrees;
true-north versus projected-grid orientation needs explicit treatment.

The already disclosed fixed body-quarter-turn diagnostic changes raw L32 error
from 102.311022 m to 5.170367 m. Subtracting pi/2 from yaw **before** building the
orientation instead gives 4.485777 m. These are different transformations when
roll/pitch are nonzero; the better number cannot choose the physical convention.
Neither conversion has been established as Oxford's authoritative frame fix.

Oxford's SDK follows Rz(yaw) Ry(pitch) Rx(roll), uses RTK north/east/down and
roll/pitch/yaw columns, and applies the INS extrinsic. Reproducing that SDK does
not validate an RTK export that may use a different attitude convention.
NovAtel's nominal Inertial Explorer body frame differs from Oxford's vehicle
frame, but its documentation alone does not establish Oxford's export settings.

Original GPS/INS for development traversal `2014-11-21-16-07-03` has been requested
as an independent cross-check, not as a replacement target chosen for accuracy.
Its official GPS archive is 30.12 MB with MD5
`D7455BFE51B17A8B42B0E8B0F40AE228`.

### 3. Serious RTK target failures were admitted to training

In training traversal `2015-02-06-13-57-16`, consecutive recorded RTK rows imply
about 1,647 m displacement in 0.101599 s: 16,212.59 m/s. The same source also
reports vehicle speeds up to 177.05 m/s. A four-second training window has
7,739.62 m of target displacement. These values occur in the ZIP itself, not
only in a transformed tensor.

Nine training and five development traversals contain edges flagged by the
generous >50 m/s reference sanity rule (position-derived or reported speed).
There are 1,262 flagged training RTK edges and 18 development edges. This is a
diagnostic threshold, **not a certificate that remaining reference poses are
accurate**. No filtering uses any model's prediction error.

On the frame-diagnostic training set, the largest 1% of raw L8 squared translation
errors account for approximately 99.95% of that loss component. This explains
why a mean-squared-loss training process requires target quality checks even
when the underlying public dataset is reputable. It does not by itself establish
which architecture should win after correcting the experiment.

### 4. VO edge continuity was not checked

The original loader verified sampled-anchor continuity but did not require each
VO row's destination timestamp to equal the preceding row's source timestamp.
There are 1,389 disconnected edges in training and 1,024 in development; the
largest development gap is 96.945617 seconds. Globally accumulating across such
gaps silently omits unknown motion.

At L32, 652/36,120 original development windows cross a disconnected VO edge.
Removing those windows, without changing the frame diagnostic, changes the
traversal-equal raw error from 5.170367 m to 4.884429 m. Windows never crossing a
gap are not contaminated merely because a gap occurred earlier: the preceding
global gauge cancels in their relative transform.

### 5. Timestamp and arithmetic checks do not explain the main discrepancy

Original raw metrics reproduce with independent homogeneous matrices at all
four horizons and all 14 development traversals, with per-traversal differences
below 1e-6 m (actual errors much smaller; see JSON). All 25 saved original
checkpoints reproduce all four left-path metrics to below 1e-6 m per traversal.
There is no evidence of a stale checkpoint or a result-reporting mix-up.

Changing only ceil-VO timestamp selection to nearest recorded VO selection
changes the frame-diagnostic L32 raw mean from 5.170367 m to 5.165241 m. It is
not the explanation for the original 102 m result. There are no duplicate RTK
endpoint windows. The longest actual L8 target duration is 4.106728 s in training,
so the kilometre-scale targets are not caused by treating a long time gap as four
seconds. No interpolation or invented observations were used.

## Controlled retries

`PLAN.md` specifies the frame-only sensitivity. `QUALITY_ADDENDUM.md` records the
later discovery of reference failures and specifies the additional quality-screen
sensitivity before its model fitting. Both keep the original five arms, five
seeds, 5,000 updates, learning rate, optimizer, architecture and 4-second training
length. The quality-screened study keeps 93,846/95,152 training windows (98.63%).
All development windows are still evaluated, alongside the explicitly screened
subset (35,395/36,120 windows at L32).

Results belong to `runs/` and `quality_screened/runs/`, respectively. No best seed,
checkpoint, transformation or subset is selected for the manuscript. The
independent matrix loader initially introduced quaternion-sign changes into
input coefficients; those prototype fits were stopped and preserved under
`pre_input_lift_check/INVALID_PROTOTYPE.md`. Current retries take the **exact
original input coefficients** from the unchanged original loader. Regression
tests verify coefficient identity, the intended target-frame conjugation and
independent SDK matrix agreement using recorded observations only.

Both retries completed: 50 fits and 50 complete development evaluation records,
covering all four horizons, left/right/balanced and 16 random trees, all-window,
moving and explicitly screened subsets. The numerical comparison is in
`analysis.json`; checkpoint completeness and float64 GA path audits are checked
in `verification.json`. No result resolves the physical RTK attitude convention
or authorizes original confirmation.

Verification completed: all 50 checkpoint hashes and evaluation records pass;
five recorded-data regression tests pass. Across both retries, five seeds and
all four horizons, GA's maximum float64 path discrepancy is 3.9932e-13 m and
3.1101e-15 rad. The original manuscript TeX and PDF hashes are unchanged.

### Completed results: conditional on the diagnostic frame

Five-seed means, equal weight per development traversal, L32 (~16 seconds):

| Training / evaluation | Raw VO motor | GA-calibrated motor | GA minus raw, exploratory 95% traversal-bootstrap CI |
| --- | ---: | ---: | ---: |
| Frame-only / ALL windows | 5.170367 m | 5.687240 m | +0.516874 m [0.242703, 0.777741] |
| Quality-screened training / ALL windows | 5.170367 m | 3.456373 m | -1.713994 m [-1.981070, -1.447841] |
| Quality-screened training / screened windows | 4.822307 m | 3.248300 m | -1.574007 m [-1.801781, -1.325223] |

The important diagnostic is the second row: a 33.15% reduction without removing
any development evaluation windows. All five seeds improve on raw (3.213008,
3.366050, 3.360241, 3.778104, 3.564461 m), and seed-averaged error improves in all
14 development traversals. Mean rotation error also declines, from 4.004995 to
2.922573 degrees. The first row is retained: changing the frame alone did NOT
produce improved GA calibration.

At L8 (~4 seconds), the training horizon, quality-screened training evaluated on
ALL development windows gives:

| Composer | Left reduction | Right reduction |
| --- | ---: | ---: |
| Raw motor | 1.401170 m | Same mathematical composition |
| GA calibrated | 0.860172 m | 0.860172 m |
| Learned bilinear | 1.072228 m | 6.283145 m |
| MLP residual | 0.853522 m | 3.613283 m |
| Mixed-path bilinear | 1.200492 m | 1.306091 m |
| Associator-penalized bilinear | 0.955844 m | 6.045718 m |

The MLP is a matched-accuracy comparator here. Its right-minus-left change is
2.759761 m, with exploratory whole-traversal CI [2.452112, 3.085411] m. This is
distinct from the fourfold length-extrapolation failure at L32, where its left
error is 37.624152 m and the bilinear arm's is 61.158989 m. The quality repairs
do NOT make the original L32 comparator-competence gate pass. L8 must not be
retroactively substituted as the original primary endpoint.

Interpretation: the original Oxford result was confounded by real process/data
problems. The new diagnostics show a viable source of useful calibration and a
matched-accuracy path-sensitivity comparison worth independently validating.
They are NOT a repaired prospective result, a new confirmation pass, or proof
that GA beats equivalent SE(3)/dual-quaternion coordinate implementations.

## What is required before an Oxford scientific rerun

1. Independently identify the RTK attitude export frame, roll/pitch conventions,
   heading reference (true/grid north), and the appropriate lever-arm transform.
   A body quarter-turn and a yaw offset are not interchangeable. The requested
   original GPS/INS file can provide an additional check; if still ambiguous,
   obtain clarification from the dataset maintainers before selecting a fix.
2. Specify a reference-quality policy and disconnected-edge treatment before
   running new validation. The present 50 m/s screen flags blatant failures,
   not every inaccurate RTK pose. Do not select exclusions on model errors.
3. Separate same-horizon, matched-accuracy path comparisons from long-horizon
   generalization. Retain both, with the original failed endpoint disclosed.
4. Only then define an explicitly new corrected protocol and confirmation
   decision. Do not reuse the failed original release gate as if it had passed.

## Sources

- [Oxford data format and frames](https://robotcar-dataset.robots.ox.ac.uk/documentation/)
- [Oxford RTK release](https://robotcar-dataset.robots.ox.ac.uk/ground_truth/)
- [Pinned Oxford SDK](https://github.com/ori-mrg/robotcar-dataset-sdk/tree/16ce3329223ca418fe5106277b91aea8d9b672b2)
- [Earlier RTK axis report, not a maintainer-certified fix](https://github.com/ori-mrg/robotcar-dataset-sdk/issues/40)
- [Earlier projected-frame orientation inquiry](https://github.com/ori-mrg/robotcar-dataset-sdk/issues/55)
- [NovAtel SPAN/Inertial Explorer frame definitions](https://docs.novatel.com/Waypoint/Content/AppNotes/DeterminingRotationSPAN_IE.htm)
- [Original GPS/INS download page](https://robotcar-dataset.robots.ox.ac.uk/datasets/2014-11-21-16-07-03/)

## Reproduction and preservation

Run `test_recorded_pipeline.py` for non-mutating regression tests. JSON outputs
and checkpoints are exclusive-write, with provenance hashes. For a fresh replay,
copy the current Python and Markdown sources into a new sibling experiment
directory before running `investigate.py`, `retry.py`, `quality_retry.py`,
`checkpoint_replay.py`, `archive_recheck.py`, `loss_diagnosis.py`, `analyze.py`, and
`verify_completed.py` in that order. Keep the same original data and source
history. Do not overwrite archived outputs or modify 020/031 to bypass a seal.
The large derived tensor caches are locally git-ignored; recorded source files
remain unchanged in the raw-data store.
