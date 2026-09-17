# KITTI reproduction scope

## Without original data

```sh
python verify_kitti_records.py --root . --artifact
```

This uses only the Python standard library and retained records. It verifies
every final checkpoint and included dependency, checks all 55 report hashes,
recomputes every available arm/group mean from the 125,595 window/seed rows,
checks per-seed minibatch-schedule matching, and verifies the recorded numerical
thresholds. It does not independently rerun the geometric operations or models.
The full-data local version was run without `--artifact`, verifying 214 distinct
files and zero omitted inputs; its receipt is `kitti_audit.json`.

`KITTI_EXTERNAL_INPUTS.json` is the exact hash inventory of inputs omitted from
this distribution: KITTI poses/calibration/timestamps, native devkit sources,
and source-derived frontend records/receipts. Some frontend receipts include
machine-specific execution paths and are not redistributed verbatim. This is
an explicit no-data audit boundary, not a claim to authenticate missing bytes.
Original images and archives must also be obtained separately for extraction.

## With separately obtained original data

Register at https://www.cvlibs.net/datasets/kitti/ and obtain the official
odometry grayscale stereo images, calibration/timestamps, training poses and
devkit under its terms. Obtain LIBVISO2 from
https://www.cvlibs.net/software/libviso/. No account or credential is included.

Use a **fresh working copy**, leaving the delivered results untouched. The
unchanged experiment 033 `README.md`, `PREPARATION_PROTOCOL.md`,
`EXPERIMENT_PROTOCOL.md` and `IMPLEMENTATION_NOTES.md` document acquisition,
unchanged upstream compilation, frame inversion, frontend checks, training-only
scaling, all 30 fits and the technical-only holdout gate. The scripts write
exclusive completion receipts and refuse to overwrite old attempts. Do not run
training against this delivered evidence directory or relax a gate to obtain
a preferred outcome. New environments/binaries generate new identities; preserve
their receipts as replications rather than replacing historical hashes.

For a KITTI-only replay, if `KITTI_SOURCE_SNAPSHOTS.json` names newer imported
sources, copy those exact snapshot files to their listed relative locations
in the fresh working copy. Do not overwrite the multi-study evidence snapshot
or claim its older source seals stayed unchanged.

The recorded implementation was Python 3.14.7 / PyTorch 2.12.0 / NumPy 2.4.6,
single Torch thread, macOS ARM64. LIBVISO2 was unchanged x86_64 SSE C++ under
Rosetta, compiled with `-msse3 -std=c++11 -O2 -ffp-contract=off`. Numerical
agreement on another platform is to be measured, not assumed. The earlier
preseal `test_*.py` suite includes stage assertions that later development
outputs do not exist; it is not an all-stage test suite for a completed run.

Study split: train 00–06, development 07–08, holdout 09–10. Official test 11–21
is never allowed for training/tuning and this work makes no server submission.
Predictions are independently read out for each segment, not a globally coherent
corrected trajectory. Primary 100 m and secondary 200–800 m metrics must not
be labeled official KITTI leaderboard scores. Preserve the primary negative
translation outcome and all seeds when comparing a new reproduction.
