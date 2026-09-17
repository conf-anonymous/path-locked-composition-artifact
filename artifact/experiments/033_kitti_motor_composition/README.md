# KITTI GA motor composition — evaluation complete

**Current result, 2026-09-11:** All 30 fits and all 55 sequence/seed reports
completed; technical checks pass. Frozen study holdout 09–10 has primary
100 m translation error **1.061867 m raw versus 1.168216 m mergeable**:
the learned correction is **10.02% worse**, and the prespecified scientific
success criterion fails. Rotation improves 7.37%. This is a valid negative
accuracy result, not a technical failure. Official test 11–21 remains excluded.
No workers remain queued by this completed campaign. Original source/checkpoints
and manuscript are unchanged. See [EVALUATION_STATUS.md](EVALUATION_STATUS.md).
The separately specified post-hoc selector analysis is complete in
[experiment034](../034_kitti_selective_correction/RESULTS.md); it does not
replace this frozen result or establish benefits-only selective correction.

**Historical execution milestone, 2026-09-11:** All 30 fits completed their fixed
5,000 updates, without recorded failures; final checkpoint identities verify.
Eight-arm training evaluation is running (7/35 sequence/seed reports complete
at the first check, all technical audits passing). Development and conditional
holdout stages are now queued behind the prescribed checks. No accuracy-gain
claim or manuscript change yet. See [EVALUATION_STATUS.md](EVALUATION_STATUS.md)
and the operational supervisor status for current progress.

**Earlier implementation milestone, 2026-09-11:** The training/evaluation implementation is
sealed, 31 preseal tests pass, and the prescribed 30 serial fits have started.
The seed-0 calibrator and constant gate completed; contextual fitting was active
at this milestone. Development/holdout remain unopened. This is execution
progress, not a GA improvement result. See
[IMPLEMENTATION_STATUS.md](IMPLEMENTATION_STATUS.md) and the live per-fit
receipts. The unused first implementation seal was preserved and corrected
before fitting; no scientific protocol changed.

**Earlier frontend milestone, 2026-09-11:** The scientific specification is sealed, and full
LIBVISO2 extraction on training sequences 00–06 passes every frozen frontend
check: 15,237 stereo frames, 15,230 valid motion edges, zero failures, 1,458
primary 100 m windows, 100% coverage. Equal-sequence mean raw 100 m translation
error is 1.716576%. This is a competent-baseline diagnostic, NOT a learned GA
gain. No GA fitting, development/holdout scientific access, manuscript edit or
artifact rebuild occurred. See [FULL_TRAINING_RESULTS.md](FULL_TRAINING_RESULTS.md)
and [EXPERIMENT_PROTOCOL.md](EXPERIMENT_PROTOCOL.md). Next: implement, test and
seal the training/evaluation runner before the prescribed fits. The earlier
smoke milestone is preserved in [FRONTEND_RESULTS.md](FRONTEND_RESULTS.md).

## Historical preparation milestone (before frontend execution)

The remaining preparation narrative records the earlier state, not pending
download requests or the current scope of training-image access.

2026-09-11. The author downloaded the four official KITTI odometry archives.
Their original files remain unchanged in Downloads; the 23.17 GB decimal
(21.58 GiB) image ZIP is neither copied nor bulk-extracted. Acquisition checks
use one process and 4 MiB streaming buffers. No frontend or neural model has run.

## State at preparation completion

- All four complete ZIPs pass streamed member CRC/size checks, unique-member
  and safe-path checks. SHA-256 receipts are in
  `data/raw/kitti_odometry/acquisition.json`.
- Fingerprints preserve local byte integrity; they are **not** verification
  against a separately published authoritative digest. Download origin is based
  on the author's official-site acquisition. No publisher SHA-256 was verified.
- Only the original devkit and training calibration/timestamps/poses are staged
  under `data/raw/kitti_odometry/`. No image was decoded or extracted.
- Training checks pass for 15,237 frames: counts, timestamps, finite values,
  initial reference pose, SO(3) rounding tolerance and stereo calibration. The
  rectified stereo baseline is approximately 0.53715–0.53717 m. The maximum
  training rotation orthogonality deviation is 2.802e-7.
- The official C++ reference and NumPy select exactly the same 10,015 eligible
  training segments across 100–800 m; maximum relative-matrix coefficient
  discrepancy is 9.095e-13. Their cumulative-distance implementations differ
  by at most 0.0001221 m from floating-point rounding, without changing segment
  endpoints. Reference-versus-itself checks pass. These are geometry/metric
  sanity checks, **not odometry estimates or scientific model results**.
- Five preparation tests pass. No KITTI accuracy gain, frontend competence,
  end-to-end frame correctness, or physical reference accuracy is claimed.

## Fixed preparation split (inspection status at that milestone)

| Role | Sequences | Numeric content inspected? |
| --- | --- | --- |
| Training | 00–06 | Reference poses/calibration/timestamps only |
| Development | 07–08 | No |
| Our study's holdout | 09–10 | No |
| KITTI official test, outside this study | 11–21 | No |

All ZIP members, including held-out/test members, were read only for byte-level
integrity checks; directory metadata were counted. This is not a claim that
their bytes were never read. Their images were not decoded, their pose values
were not parsed, and no model or outcome evaluation accessed them.

The split is a custom split of public KITTI training sequences, not the official
test set. Only two study holdouts are available. Distinct raw-drive IDs do not
prove geographical route separation. No confidence calculation may treat all
overlapping windows as independent sequences.

## Frame contract

The original devkit describes each pose as a row-major 3x4 transform taking
points from the rectified left camera at frame i to the first camera frame:
`T_0_i`. Thus relative motion is `inverse(T_0_i) @ T_0_j`. The left camera has
z forward; P0/P1 are the rectified stereo projection matrices. Nothing from
Oxford's disputed yaw/roll/extrinsic mapping is imported.

The native reference probe includes the untouched original evaluator and matrix
source. The distributed evaluator's unused server main calls `Mail::finalize`,
which is absent from its distributed mail header. A local no-I/O Mail adapter
in `native_probe.cpp` resolves compilation without changing any metric code or
source archive, and without sending mail. Original main/eval/plot functions are
never invoked. Compiler warnings/output are retained in the reference receipt.

## Then-planned next step (now completed through full training frontend checks)

LIBVISO2 remains the first frontend feasibility candidate, not an installed or
validated dependency. Its official download requires an emailed link:
https://www.cvlibs.net/download.php?file=libviso2.zip

The author has been asked to save the source archive as `libviso2.zip` in
Downloads. No email address or form was submitted on the author's behalf.
Native Apple Silicon compatibility must be checked; the upstream download
advertises Linux/Windows, so a successful Mac build is not assumed.

After source verification: build and run a bounded **training-only** smoke test,
audit pose direction, metric scale, stereo calibration and tracking continuity,
and record resource usage. Do not invent motion for dropped frames. If a port
is needed, preserve upstream bytes and document/test the compatibility changes.
Do not quietly replace it with a deliberately weak frontend.

Then fix the complete training/evaluation protocol: exact model arms, seeds,
losses, budgets, selection rules, primary metrics, pooling, failure handling,
competence requirements and conditions for opening our study holdout. The
preparation protocol's design outline is NOT a complete frozen experiment.
No training is authorized by a claim that preparation alone is complete.

The GA focus remains motor-valued calibration and paired geometric-product
summaries. This study must distinguish task accuracy from exact associativity
and include competent information-matched controls. The official warning about
very short reference segments requires distance-based metrics in addition to
separately labeled fixed-input tree audits. Training sequences 03 and 04 do not
support every 100–800 m length: preserve the recorded eligibility counts.

Oxford remains excluded pending frame clarification, not overwritten by KITTI.
The author reported sending the maintainer questions on 2026-09-10. Manuscript
consolidation, stale-record reconciliation and final claim audit are authorized
but have not been completed as part of this acquisition step. Existing TeX/PDF
and the audited submission artifact are unchanged.

## Commands

Completed once (will refuse to overwrite acquisition receipts):

```sh
.venv/bin/python experiments/033_kitti_motor_composition/acquire.py --downloads /path/to/downloads
.venv/bin/python experiments/033_kitti_motor_composition/check_native_reference.py
```

Repeatable, read-only checks:

```sh
.venv/bin/python -m unittest discover -s experiments/033_kitti_motor_composition -p 'test_*.py' -v
```

The tests recheck staged hashes, original archive stat/inventory, split guards,
training geometry and native-reference provenance. They do not rehash the full
image archive on each invocation. The initial full streamed check is recorded.
