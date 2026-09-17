# KITTI training frontend validation

2026-09-11. **All seven training sequences pass the prospectively specified
frontend admission checks.** This establishes a usable fixed baseline for the
planned learned GA comparison, not a GA improvement or a held-out result.

## Observed results

The unchanged LIBVISO2 stereo implementation processed 15,237 recorded stereo
frames from sequences 00–06, estimating all 15,230 noninitial motion edges.
There were zero tracking failures. All 1,458 primary 100 m windows and all
10,015 eligible 100–800 m windows are continuous and admissible. No images,
motion observations or tracking repairs were synthesized.

Metrics below use the original KITTI devkit functions on the raw frontend
trajectory, with the original 10-frame start stride and first strictly
distance-exceeding endpoint. These are training-split diagnostics, not official
test/leaderboard results. Windows overlap and are not independent samples.

| Training sequence | Frames | 100 m windows | Raw translation (%) | Raw rotation (deg/m) | Identity translation (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 00 | 4,541 | 445 | 1.329244 | 0.01842281 | 90.119102 |
| 01 | 1,101 | 104 | 4.707098 | 0.01101309 | 99.052718 |
| 02 | 4,661 | 459 | 1.208519 | 0.01196456 | 93.776063 |
| 03 | 801 | 62 | 1.447823 | 0.01076435 | 97.506312 |
| 04 | 271 | 21 | 1.075290 | 0.00820666 | 100.671815 |
| 05 | 2,761 | 267 | 1.209950 | 0.01755547 | 91.837676 |
| 06 | 1,101 | 100 | 1.038112 | 0.01359983 | 90.499517 |

Equal-sequence mean raw 100 m translation error: **1.716576%**. Each sequence
beats identity and satisfies the frozen <10% translation and <0.2 deg/m
rotation checks, >=95% valid-edge and >=80% primary-window coverage checks,
and existence of a primary window. These are engineering admission thresholds,
not evidence of state-of-the-art frontend quality. Sequence 01 remains in the
study with its larger error; no sequence was dropped or tuned after results.

Identity error need not equal exactly 100%: it uses endpoint displacement,
which differs from traveled path length, divided by nominal segment length;
the first endpoint beyond the threshold can overshoot it.

## What was fixed before these outcomes

[EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md) specifies all eight arms,
five seeds, training windows, losses and budgets, matching of input information,
primary/secondary metrics, negative-result reporting, geometric-product/tree
audits, and development/holdout access rules. It was written after the disclosed
401-frame sequence-00 smoke test, but before this full extraction, GA fits or
development/holdout inspection. This is a local prospective specification, not
independently timestamped public preregistration.

The learned task is segment-relative prediction using the existing PGA leaf
calibrator and paired-product gate, with matched recurrent direct/residual
controls. It does not produce an assumed globally coherent corrected trajectory.
Its future metrics must be labeled KITTI-distance segment-relative evaluation,
not official full-trajectory odometry scores. Matrix-versus-motor consistency
will test the geometry, not assert GA-unique associativity.

Holdout release is conditional on technical validity, frontend checks, complete
prespecified fits/reports and implemented access safeguards, **not on a positive
GA development effect**. No training software completion is implied by having
specified the scientific protocol.

## Integrity and verification

Raw receipts are under `data/raw/kitti_odometry/libviso2/full_train/`:

- `specification_seal.json`: pins the scientific specification, extraction
  sources, existing model-source dependencies, acquisition receipt and binary.
- `00.frames.jsonl` through `06.frames.jsonl`: each frame's source/decoded pixel
  hashes, calibration-linked run identity, tracking status, matches/inliers,
  processing time, and original previous-to-current transform.
- `00.json` through `06.json`: per-sequence provenance, resource scope,
  window inventory, native metrics, continuity and admission checks.
- `metric_build.json`: original devkit source identities and helper build.
- `completed.json`: hashes of all seven completed sequence receipts and scope.

SHA-256 identities:

```text
EXPERIMENT_PROTOCOL.md
815e6d2c32beeb0a2f2c442094a3ca9e6810f0287a12647948466ab80886b09f
specification_seal.json
984b0cacadb8c7bfd158b98460fa40d7e1ee7f2f493d93f3cf91405ab75279b8
completed.json
3d240feabb5e0f97f988f87e14ec680ed37a23bcbb71abfc53d8533b22b23d13
```

All **19 read-only tests pass**. The suite includes full native metric replay on all seven sequences,
independent NumPy endpoint selection for all 10,015 segments, independent
coverage/mean/admission reconstruction, exact semantic replay of the original
401-frame smoke prefix, native helper agreement with the original evaluator,
source/data hashes, and guards rejecting development/holdout before reads.
No real tracking gap occurred, so this run cannot empirically validate recovery
after a gap; the code excludes cross-component windows, but observed zero
failures must not be sold as a failure-recovery stress test.

## Resources and current boundary

Sequential frontend extraction totaled 485.42 seconds (8.09 minutes), excluding
native metric compilation and later audit tests. The recorded completed-child
high-water RSS was 347.38 MiB; this includes the compiler and is not an isolated
per-sequence or whole-pipeline memory measurement. Original images were streamed
from Downloads without copying or bulk-extracting the 23.17 GB image archive.
All extraction workers completed.

Development 07–08, study holdout 09–10, and official test 11–21 remain
scientifically uninspected. Their archive bytes previously received integrity
checks only. No GA fitting, model selection, new accuracy-gain claim, manuscript
edit or artifact rebuild occurred at this stage. Oxford remains excluded pending
an authoritative frame/export clarification.

Next: implement and test the frozen training/evaluation runner and fail-closed
holdout release, freeze the training-only data scale and implementation receipt,
then execute the 30 prespecified fits. Retain negative outcomes and all seeds.
Only verified later evidence can support manuscript changes.

## Recheck

```sh
.venv/bin/python -m unittest discover -s experiments/033_kitti_motor_composition -p 'test_*.py' -v
```

`full_frontend.py` resumes completed receipts by checking hashes; it refuses to
overwrite a partial extraction. Do not edit sealed sources or replace an attempt
without preserving its provenance and documenting an explicit amendment.
