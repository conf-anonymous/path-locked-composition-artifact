# LIBVISO2 smoke test: works on this Mac, without an algorithm port

Completed 2026-09-11. The downloaded `libviso2.zip` passes ZIP CRC verification;
its local SHA-256 is
`cbefe217840f136a12718b59bb57d43a3173694d5e5a5ed7704e2c1be2ac293d`.
No independent publisher hash was verified. Original archive, source files and
license notices are retained under `data/raw/kitti_odometry/libviso2/`.

## Implementation and scope

The original SSE-dependent C++ sources compile unchanged as x86_64 with Apple
clang, `-msse3 -std=c++11 -O2 -ffp-contract=off`, and execute through Rosetta.
This is **not native ARM execution**. No SSE-to-NEON port, algorithm rewrite,
source patch, custom replacement frontend, or numerical approximation was used.
`build.json` records every source hash, command, compiler version and warning.

A local adapter streams unmodified decoded grayscale pixels to LIBVISO2 and
returns its status, counts and motion matrix. It receives calibration but never
reference poses. All matcher/RANSAC defaults and the upstream `srand(0)` are
unchanged. Images come only from real KITTI sequence 00, frames 000000–000400.
They were decoded directly from the existing ZIP, not bulk-extracted, resized,
augmented, corrupted, or synthesized. Bundled LIBVISO2 demo images were not used.

The motion convention is explicit in upstream `readme.txt`, `viso.h` and
`demo.cpp`: `getMotion()` takes previous-camera points to current-camera points.
KITTI T_0_i accumulation therefore uses **inverse(getMotion())**, exactly as in
the upstream demo. This direction was fixed before results, not selected by
whichever scored better. No fitted frame alignment or scale was applied.

## Recorded observations — smoke-only, not paper results

Two fresh processes used the same 401 stereo pairs and upstream settings.
Every frame's success, feature/inlier counts, image hashes and motion coefficients
match exactly between runs. Only runtime fields differ.

| Check | Result |
| --- | ---: |
| Actual stereo pairs | 401 |
| Valid motion edges | 400 / 400 |
| Initialization frames | 1, with no motion output |
| Failed motion edges | 0 |
| Reference path length | 292.244148 m |
| Reference endpoint displacement | 243.488020 m |
| Estimated endpoint displacement | 242.195330 m |
| Endpoint translation error, no fitted alignment | 8.206540 m |
| Endpoint rotation error | 5.886464 degrees |
| Eligible official-style segments | 39 (100 m and 200 m only) |
| Native KITTI mean segment translation error | 1.412111% |
| Native KITTI mean segment rotation error | 0.02144483 degrees/m |

The nonzero accumulated rotation/position errors are retained: tracking success
does not imply negligible drift. The 39 overlapping segments from one short
training clip are not independent trials. This is not KITTI leaderboard
performance, general frontend competence, a complete sequence evaluation, or
evidence of a GA improvement. No GA model was trained.

## Metric and recording verification

The smoke's actual frontend estimates were passed to the original KITTI C++
`calcSequenceErrors`, not just checked by comparing reference poses to themselves.
All 39 segment start/length selections agree with the Python implementation.
Maximum numerical differences are 2.461e-7 percentage points for translation
and 3.209e-6 degrees/m for rotation, consistent with the devkit's float32 metric
rounding. Native means above are from `native_smoke_metrics.json`.

The local native-metric adapter provides the same no-I/O Mail compatibility
class as the earlier reference probe, because the shipped server wrapper refers
to an absent method. No original metric source was edited; server main, email
and plotting routines are never called. Actual observed frontend estimates and
recorded reference rows are the only poses used in the nonzero metric test.

Six frontend tests pass: upstream bytes/build provenance, exact semantic replay,
source-PNG decoding hashes, initialization/failure and scope boundaries, geometry
replay, and native metric agreement. The five acquisition/reference tests still
pass (11 total). There were no failed interior edges in this clip, so its success
does not empirically exercise mid-trajectory recovery. The adapter emits no
motion on failure and the analysis splits components instead of inventing poses;
real full-sequence failures must still be audited before any general evaluation.

## Resource use

| Run | Wall time including image streaming | Child CPU time | Recorded child peak RSS |
| --- | ---: | ---: | ---: |
| 1 | 12.039 s | 9.331 s | 212.05 MiB |
| 2 | 11.498 s | 9.474 s | 212.05 MiB |

Memory is macOS RUSAGE_CHILDREN's high-water RSS across completed child
processes, not aggregate system pressure or a separately sampled whole-pipeline
peak. Runtime is this bounded Rosetta smoke, not a native ARM benchmark or a
guarantee of full-dataset runtime. No job from this smoke remains running.

## Next work

Frontend feasibility is established. No additional source download is currently
needed. Preserve 07–08 as uninspected development and 09–10 as study holdout;
11–21 remain outside the study. Preparation and smoke receipts record the earlier
integrity-only reading of all archive bytes separately from scientific access.

Next, specify a complete prospective experiment before fitting any GA model:
full training/development frontend extraction with unchanged settings; recorded
failure/continuity handling; exact windows and distance metrics; model and
information-matched baseline definitions; training budgets/seeds; selection and
reporting rules; competence checks; and the holdout release decision. Full
training-sequence tracking/metric checks must precede claiming that this frontend
is a competent general baseline. The preparation/smoke protocols alone are not
a complete frozen training study. No confirmation is authorized yet.

Manuscript/artifact consolidation and the final adversarial audit remain pending
approved work. Existing source/PDF and published-evidence boundaries are
unchanged. This smoke must not be promoted to a scientific result in the paper.

## Files and rerunning

- `build_frontend.py`, `stereo_stream.cpp`: source verification and original-code build.
- `frontend_smoke.py`: two fresh deterministic fixed-clip runs and diagnostics.
- `check_smoke_metrics.py`, `metric_stream.cpp`: native evaluator on actual estimates.
- `test_frontend.py`: repeatable read-only checks.
- Raw receipts and per-frame records: `data/raw/kitti_odometry/libviso2/`.

Build/run scripts refuse to overwrite completed receipts. Do not delete prior
results to rerun; use a documented new output directory for additional work.
Repeatable verification:

```sh
.venv/bin/python experiments/033_kitti_motor_composition/test_frontend.py -v
.venv/bin/python experiments/033_kitti_motor_composition/test_preparation.py -v
```
