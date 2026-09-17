# Oxford execution implementation — 2026-09-09

Implementation of the unchanged Experiment 031 protocol after acquisition and
schema validation, but before Oxford model training or outcome inspection.
The original 020/031 protocols and their original seals, and the acquisition-time
schema amendment seal, are retained. Changed preparation-stage code is archived
under `pre_runner_2026-09-09/`. A new implementation seal records the active code,
all model dependencies, input manifest, and public ETH3D test evidence.

## Execution discipline

- Development uses the existing 35/14 traversal assignment; nine confirmation
  traversals are not loaded until joint release is explicitly admitted.
- The original five arms/five seeds keep their fixed 5,000-step configuration.
  Completed checkpoint/seed records are saved with input and code hashes, so a
  restart reuses a completed fit instead of retraining or overwriting it.
- Extension fits all five heads for each of five calibrator seeds before any
  extension development evaluation. All heads for a seed receive the same
  balanced traversal/window sampling stream and use L8, 5,000 steps, and the
  fixed robust loss. No parameter, model, seed, horizon, or checkpoint selection.
- Result and checkpoint files are exclusive-create and hash-checked. A partial
  checkpoint or evaluation record is not silently overwritten or admitted.
- Both studies' full development artifacts and all 50 checkpoints are frozen
  in a shared release manifest only after original gate success and technical
  completeness. Extension utility is never a release-selection gate.
- Confirmation is an explicit joint command with one exclusive start marker.
  The marker binds the release hash and invoking process. Standalone commands
  cannot open confirmation data. A crash after release start does not silently
  rerun confirmation: it requires a documented recovery procedure.
- Scientific/accuracy failure blocks release; failures and partial artifacts
  are retained. This is experiment orchestration/integrity, not an adversarial
  security boundary against someone editing local code or files.

## Measurements and limits

Extension results cover all eight arms, all five seeds, four horizons, and both
all-window and displacement>=5m subsets. Metrics first aggregate within traversal
and seed; uncertainty resamples whole traversals after seed averaging.
Primary mergeable comparisons use 10,000 bootstrap draws and both ordinary 95%
and Bonferroni-adjusted 98.75% intervals. Other comparisons are exploratory.
The complete 16-coordinate raw/calibrated motor-pair summary is audited in
float64 across ordered trees before readout; decoded predictions are not
advertised as mergeable state. No equivalence claim against contextual gating.

Engineering tests use existing recorded ETH3D VO/reference windows and retained
real ETH3D checkpoints. Short optimization tests are execution checks only, not
new research results; they do not inspect Oxford outcomes. Negative control-flow
tests mutate metadata/manifests, never generate or perturb observations.
