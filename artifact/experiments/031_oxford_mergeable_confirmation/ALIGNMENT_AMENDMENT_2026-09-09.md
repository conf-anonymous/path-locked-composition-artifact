# Pre-training recorded-timestamp eligibility correction

The first development launch stopped inside `OxfordData` before training,
baseline evaluation, any metric calculation, or creation of an Oxford checkpoint.
The sole unusable source was training traversal `2015-08-20-12-00-47`.
Its official VO timestamps span [1440068447820035, 1440069934016490], while
its RTK timestamps span [1440067718193439, 1440068334098336]. There is no overlap.
Its maximum contiguous valid-anchor count under the existing rule is zero.

The same timestamp-only check was applied to all 58 source pairs, including
confirmation metadata but never confirmation poses, targets, or errors. The
other 57 have at least 33 consecutive admissible anchors, supporting L32.
Preparation now applies this existing run-usability condition before writing
the manifest, rather than deferring rejection to data construction.

This corrects technical eligibility; it does not modify the 0.5-second grid,
0.1-second maximum offset, recorded-row selection, minimum run length, input
order, or outcome gate. No interpolation, altered timestamps, nearby traversal,
or artificial observations are substituted. No source alias is repaired.
The unused source remains in the raw archive and exclusions are explicit.

The new counts are 34 training / 14 development / 9 confirmation (57 total).
The previous manifest and initial implementation seal are preserved in
`pre_alignment_2026-09-09/`, together with changed source snapshots. Their
hashes remain verified as historical links. Active implementation is resealed
after repeated recorded-data tests, still before any Oxford outcome inspection.
The original scientific protocol, initial seals, schema amendment, all data
bytes, seed assignments, architectures, and fitting/analysis choices remain.
