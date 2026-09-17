# Acquisition-time schema correction and confirmation-access hardening

Recorded 2026-09-09, after acquisition and schema-only inspection, before any
Oxford model training, endpoint-error calculation, or outcome inspection.
This is not a new pre-acquisition registration. Original protocol texts and
original seal JSON files are retained unchanged.

## Source-schema correction

All 58 matched RTK files were rejected by the original loader because it
attempted to parse `utm_zone` as a float. The source uses the text label `30U`.
The correction recognizes the official RTK header, validates zone labels as
metadata, and preserves their text separately from numerical pose calculations.
All numeric pose fields and recorded-row order remain unchanged. Raw files
are never rewritten. There is no interpolation, augmentation, generated data,
changed frame convention, target redefinition, or split reassignment.

Data preparation must also explicitly list malformed source directory names
(rather than silently ignoring them), preserve the original exclusions, and
refuse to overwrite a nonidentical scientific manifest. Official VO archive
MD5 verification and extracted-byte SHA-256 matching precede eligibility.
Unresolved traversal aliases remain excluded pending source clarification;
no nearby traversal is substituted and no malformed date is silently repaired.

## Implementation hardening (no change to scientific release conditions)

Inspection found `OxfordData` eagerly materialized train, development, and
confirmation tensors even for a development-only run. The corrected loader
defaults to train/development only and checks the shared release guard BEFORE
loading any confirmation traversal. Original command-line and programmatic
confirmation entry points must use that same guard. Confirmation remains
unconditionally disabled while the Experiment 031 runner/joint release
implementation is incomplete, as already required by its sealed protocol.

The Experiment 031 protocol also requires its runner to be implemented,
tested on public recorded ETH3D inputs, and sealed before Oxford outcome
inspection. Therefore this implementation phase permits schema validation
and manifest preparation, but not an Oxford experiment run yet.

## Integrity and testing

Copies of affected pre-amendment sources are retained in
`pre_amendment_2026-09-09/` and checked against their original seals. A separate
amendment seal binds corrected sources and tests without changing the old seals
or representing the correction as pre-acquisition work. Protocol and model
architecture hashes outside this documented correction must remain unchanged.

Tests use the actual public recorded source files. Negative tests exercise
metadata/header validation and access-control calls, not synthetic observations.
Checks include unchanged numerical fields and source hashes, intact original
snapshots, manifest consistency, and denial before confirmation file access.
