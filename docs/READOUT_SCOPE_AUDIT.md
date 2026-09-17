# Frozen ETH3D readout and matched-start sensitivity

Experiment 040 evaluates all 25 original Experiment 028/029 checkpoints at
lengths 4, 8, 16, 32, 64 and 128. It was specified after inspecting the earlier
results. It is post-hoc sensitivity analysis, not independent confirmation,
retraining or model selection. Inputs are unchanged public ETH3D recordings;
no observations, targets, augmentations or perturbations are generated.

## Questions and retained outcomes

1. Does antipodal sign alignment change the primary result? Keeping the gate
   coefficient and all parameters fixed, align the calibrated endpoint's real
   quaternion to the raw endpoint only in the final blend. All eight motor
   coordinates receive the same sign. The mergeable model's primary translation
   error changes from 0.439929 to 0.439645 m; rotation changes from 28.6001 to
   28.6061 degrees. The original model is retained. The aligned final blend has
   real norm at least 1/sqrt(2) for unit endpoint real parts, but this is not an
   accuracy guarantee or invariance of the learned gate to input sign changes.
2. Does the horizon pattern depend on changing recording coverage? In addition
   to all original windows, evaluate the 23 sequences/13 families supporting
   length 128, then exactly the same 3,435 recorded starting points at every
   horizon. On matched starts, mergeable/identity translation errors are
   0.358/0.819 m at length 32 and 0.666/1.146 m at length 64. The length-128
   translation uncertainty still includes zero and rotation remains worse than
   identity. This cohort conditions on run availability, not a new population.

All eight original arms and both aligned variants are present in all three
cohorts at all six lengths. The 150 records exactly reproduce the original
sequence-level metrics. Family-bootstrap intervals are conditional on the
fixed models, without multiplicity adjustment or removal of fold dependence.
The original intervals retain their original random draws.

## Verification without sensor data

From the repository root:

```sh
python3 artifact/experiments/040_eth3d_readout_scope_audit/verify.py
```

The standard-library checker verifies checkpoint/source/record hashes,
recomputes sequence/family/seed aggregates, checks identical matched-start
hashes across horizons, and independently compares every original arm with
Experiment 030. It reports the 56 external derived-recording files as absent,
not verified. Add `--bootstrap` in the recorded NumPy environment to recompute
all conditional intervals. The repository's `reproduce.py audit` includes this
check with bootstrap recomputation.

## Frozen-model replay

Use a separate full reproduction workspace, as described in REPRODUCING.md.
Obtain the source recordings from the official ETH3D provider and reproduce the
recorded Open3D extraction; original files and derived pose arrays are not
redistributed. The included `data/raw/eth3d_rgbd/manifest.json` is the original
admission metadata and records the expected hashes of all 56 derived NPZ files.
Exact replay requires those bytes, including their recorded extraction metadata.
Any environment-dependent extraction mismatch must remain visible, not be
resolved by changing the admission hashes.

After independently reproducing the input files, verify them with:

```sh
python experiments/040_eth3d_readout_scope_audit/verify.py --require-data --bootstrap
```

`run.py` validates its entire input seal but resumes existing Experiment 040
records. Running it in the delivered tree is therefore not a fresh replay.
For a fresh replay, retain an unchanged reference copy and create a separate
full workspace. Move only that workspace's Experiment 040 `runs/` directory
and `analysis.json` to a backup location, retain its `seal.json`, source and
protocol, and execute:

```sh
PYTHONDONTWRITEBYTECODE=1 python experiments/040_eth3d_readout_scope_audit/run.py
python experiments/040_eth3d_readout_scope_audit/verify.py --require-data --bootstrap
```

This evaluates existing models; it does not invoke model training or access
Oxford confirmation. Never overwrite historical inputs or original studies.
Environment differences may change floating-point outcomes; retain new records
separately and compare them with the supplied reference rather than replacing it.

The float64 path audit samples the first 20 starts per sequence/seed/horizon
(all starts when fewer), totaling 27,215 cases across four ordered reductions.
It reports no sign-branch changes, with maximum aligned prediction differences
of 9.65e-14 m and 2.91e-14 rad. It is not exhaustive tree enumeration.
