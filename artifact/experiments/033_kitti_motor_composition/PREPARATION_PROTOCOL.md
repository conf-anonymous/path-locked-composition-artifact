# KITTI composition study: acquisition and validation boundary

2026-09-11. Written after reading the official devkit readme and ZIP directory
metadata, before parsing any KITTI pose values, decoding images, or fitting a
model. This is a prospective local preparation record, not externally timestamped
preregistration and not a complete or sealed training/evaluation protocol.

## Scope and fixed sequence roles

Only the real KITTI odometry dataset is allowed; not Virtual KITTI or generated
measurements. No artificial noise, synthetic trajectories, augmentation, or
interpolation. Motion estimates computed from recorded images are derived
estimates, not new recorded ground truth.

- Training: 00, 01, 02, 03, 04, 05, 06.
- Development: 07, 08.
- Our study's holdout: 09, 10, kept closed for scientific inspection until a
  separately specified training protocol, competent-baseline checks and release
  conditions have been fixed and development evaluated.
- KITTI's official test sequences: 11–21, outside this study. No leaderboard
  submission, model inference, parameter selection or image inspection on them.

This is a custom split of KITTI's public training set, NOT KITTI's official
test split. Two study-held-out sequences cannot support strong population-level
inference by treating overlapping windows as independent. The devkit maps these
11 sequences to distinct named raw drives; this does not establish geographical
route disjointness. Training and holdout can share collection dates and roads.
No spatial/general-purpose distribution-shift claim follows from the split.

## Acquisition

Read the author's four original Downloads ZIPs; retain them without modifying,
renaming, moving, or duplicating the large image archive. Compute streamed
SHA-256 fingerprints and independently stream/decompress every member to check
ZIP CRCs, sizes, duplicate names, encryption and unsafe paths. This includes
byte-level integrity reads of held-out/test archive members, but no scientific
inspection of their content. Preserve that distinction in reporting.

The official website and author report establish the acquisition route. Local
SHA-256 and embedded ZIP CRC are integrity checks, NOT independent publisher
authentication: no published reference SHA-256 has been verified.

Record sequence file counts and metadata inventories. Stage the exact devkit
files and only training-sequence calibration, timestamps and reference poses.
Do not extract images in bulk. Do not parse development/holdout poses in this
phase. Use a single CPU process and bounded buffers to limit memory use.

## Frame and recording validation

The supplied devkit specifies row-major 3x4 poses T_0_i for the rectified left
camera, with z forward. Each pose maps camera-i coordinates into camera-0.
Relative motion is inverse(T_0_i) @ T_0_j. P0/P1 are the rectified left/right
projection matrices. No Oxford extrinsic, yaw offset, roll sign, world-frame
alignment, or fitted correction is imported.

On training sequences only, check finite pose/timestamp/calibration values,
strictly increasing timestamps, matching image/pose/time counts, near-unit
rotation determinants, orthogonality (1e-4 tolerance for rounded text), initial
identity (1e-4), and positive stereo baseline and focal length. Report all
outcomes. Abort for structural violations rather than silently correcting them.
Report recorded inter-frame displacement and implied speed for diagnosis; do
not filter recordings by model performance or invent thresholds after outcomes.

These checks validate format and numerical geometry, not independent physical
accuracy. A later fixed frontend smoke test must verify transform direction,
units, tracking continuity and calibration using training images, before any
GA training. Frame corrections must derive from the documented conventions.

## Intended scientific design — not runnable training authorization

Inputs must come from a fixed established stereo frontend applied to actual
images, not from differencing ground truth. LIBVISO2 is the first feasibility
candidate, subject to source/build/replay validation on this machine; it is not
yet a verified working dependency. Do not substitute a weak custom frontend
merely to obtain a favorable correction result.

The study concerns learned GA motor calibration and the paired 16-coordinate
mergeable state. Preserve the original geometric-product semantics. Include raw
frontend, identity, constant/feature-matched controls, contextual and mergeable
GA models, and an equivalent-coordinate composition audit. Associativity is
not unique to GA; distinguish its geometric construction from coordinate labels.
Evaluate learned-composer path effects only where trained-path competence is
demonstrated. Do not mistake length-OOD collapse for a clean path-locking result.

Use official 100–800 m segment metrics where eligible, plus separately labeled
fixed-input/fixed-length tree audits. KITTI explicitly warns about reference
error bias on very short segments. Freeze exact windows, targets, pooling,
training lengths, losses, seeds, budgets, baselines, inference and rejection
handling in a later complete protocol BEFORE new fits. Report eligibility by
sequence and distance; never claim every sequence supports all 800 m segments.
Do not silently replace the primary metric with whichever horizon looks best.

All outcomes, including failure, must be retained. No future result is a reason
to hide the Oxford failures. Existing manuscript claims and prior frozen
experiments remain unchanged. Manuscript/artifact consolidation and final
adversarial audit are approved work, still pending separately from preparation.

## Sources

- https://www.cvlibs.net/datasets/kitti/eval_odometry.php
- Original `devkit_odometry.zip`, `devkit/readme.txt` and C++ evaluation source.
- https://www.cvlibs.net/software/libviso/
- User-supplied KITTI registration and submission policy, 2026-09-11.
