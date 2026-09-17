# Multi-date INS / RTK export validation

2026-09-10, after the single-date cross-check and before inspecting the three
new archives' observations. All three downloaded archive MD5s match the official
values recorded in the preceding report. This is a source-only diagnostic,
not model evaluation or a confirmation experiment.

Use exactly these existing split assignments:

- 2014-11-14-16-34-33: train, MD5 e09887f22a26dbed63b1af994162834f.
- 2015-08-13-16-02-58: train, MD5 1ee294931920c557bea17614280081b2.
- 2015-08-14-14-54-57: dev, MD5 fd18b53ea3275405fbc51de3eb232e90.
- 2014-11-21-16-07-03: existing dev cross-check, reproduce as a control.

Stage exact archives and exact ins.csv/gps.csv members outside the old manifest.
Verify RTK bytes against that manifest. Read no VO observations or confirmation
observations. Check original implementation seals before and after execution.

Reuse the existing nearest-recorded-timestamp rule, <=20 ms tolerance, without
interpolation, sorting, deduplication, augmentation, or fabricated measurements.
Report every timestamp reversal. Restrict INS pairing to RTK coverage plus the
tolerance only if every reversed edge is outside that interval; otherwise mark
that traversal blocked and report the problem, without silently filtering it.
Report coverage and statuses, all matched pairs, INS_SOLUTION_GOOD pairs, and
both chronological halves of good-status pairs. Small/empty groups must be
reported as insufficient, not hidden or treated as a success.

Reuse exactly the four previous fixed attitude interpretations and angle
relationships; do not fit offsets or choose a convention using VO/model scores.
Compare the sign correlations, per-traversal median roll sum, pitch difference,
wrapped yaw difference, their within-traversal spread, and attitude disagreement
for the fixed yaw-subtraction/roll-negation probe. Report the ranges of these
medians separately for train and dev. These descriptive checks have no
prevalidated physical acceptance threshold; agreement alone cannot certify the
target frame. INS and RTK share physical measurements. Neither export is assumed
to be error-free. No development-driven calibration is authorized by this plan.

Loader correction during first execution: the repeated original control has
nonfinite INS values outside the pairing interval; the original single-date
loader tested finiteness after restricting to that interval. The new loader
initially tested the whole archive and aborted before writing any results.
Report full-archive nonfinite counts and interval counts; block if any lie in
the pairing interval, and otherwise use the unchanged original pairing rule.
Do not discard nonfinite observations inside the interval or change tolerances.

Preserve all original reports, fits, sources, manuscript, and gate. Keep Oxford
accuracy gains diagnostic. Any future empirical calibration requires a separate
training-only fit protocol, validation criteria, and treatment of true/grid
heading, and may still require Oxford's export/body-rotation settings.
