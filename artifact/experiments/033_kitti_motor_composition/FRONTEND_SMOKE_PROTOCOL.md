# LIBVISO2 feasibility smoke test

2026-09-11, before KITTI image decoding or frontend execution. The official
LIBVISO2 source archive was supplied by the author. Readme/source inspection
established SSE2/SSE3 use and previous-to-current point-transform semantics.
Rosetta execution is available (`arch -x86_64 /usr/bin/true` succeeded).

## Fixed scope

- Only KITTI training sequence 00, recorded frames 000000–000400 inclusive.
- Two identical runs in separate processes, upstream defaults and upstream
  constructor's srand(0). This is a repeatability check, not seed selection.
- KITTI P0 focal/principal point and the P0/P1 derived positive stereo baseline.
  Verify shared stereo intrinsics before invoking the frontend. No fitted
  calibration, rescaling, resizing, intensity changes, augmentation, artificial
  noise, or synthetic measurements. Decode unchanged grayscale PNGs only.
- No development, study-holdout or official-test image/pose inspection. No GA
  training, tuned frontend variants, or replacement of old experimental results.

## Build and recording rules

Verify ZIP CRCs and SHA-256; stage original source/readme/CMake members exactly.
Do not use bundled demo images as research data. Compile required upstream
sources unchanged as x86_64 SSE3 with clang++ C++11, -O2, and -ffp-contract=off.
Run through Rosetta. This is not a native ARM throughput measurement. Preserve
license notices and build/source fingerprints. No software is distributed here.

A small wrapper accepts raw decoded image bytes on stdin, calls the established
frontend and returns success, match/inlier counts, elapsed time, and original
3x4 getMotion() output. It never receives ground truth. The first frame primes
the frontend and is not a failed motion edge. `replace=false` for every frame.
Do not reuse getMotion() after failure: upstream explicitly returns stale motion.
Report every failed edge, represent its motion as absent, and split connected
trajectory components at failures; never fill with identity, extrapolation, or
fabricated increments. Preserve all frame records.

## Checks and interpretation

Accumulate inverse(getMotion()) as specified by upstream demo and KITTI T_0_i
conventions. Check numeric finiteness, SO(3) and inverse closure on every valid
edge. Test byte decoding against original PNGs and semantic repeatability of
all success/match/inlier/motion outputs; timing is not expected to be identical.

Compute translation and rotation disagreement against recorded training poses
within connected components, without fitted alignment or scale. The shared
initial frame is the coordinate origin, not a learned alignment. Report final
component-relative position error and reference displacement, path length, and
eligible official 100–800 m segment metrics, using the original native evaluator
functions where practical. Short smoke-test diagnostics are not submission
results and are not a substitute for full-sequence frontend competence checks.

Record wall/CPU time and peak memory where available. Fail cleanly on structural,
decode or execution problems; preserve results if tracking/accuracy is poor.
Do not claim a high-accuracy or universally competent frontend from one short
training clip. A successful smoke test permits preparing a complete experimental
protocol; it does not authorize scientific holdout access or establish GA gains.
