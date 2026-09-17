# KITTI evaluation campaign

## Completed result — supersedes execution snapshots below

2026-09-11: 30/30 fits and **55/55 sequence/seed evaluations complete**
(35 training, 10 development, 10 study holdout). All technical reports pass.
Supervisor records completion at 16:40:51 UTC; no queued scientific work remains
in this campaign. Development 07–08 and holdout 09–10 have now been inspected;
the original official test 11–21 was not scientifically accessed. Holdout
release followed the fixed technical-only gates, not a positive-result gate.

Primary 100 m holdout translation error, equal recording/seed weighting:

| Arm | Error (m) |
| --- | ---: |
| Identity | 95.439705 |
| Raw | 1.061867 |
| Calibrated | 3.103417 |
| Constant | 1.129674 |
| Contextual | 1.176280 |
| Feature-direct | 7.285866 |
| Feature-residual | 1.332537 |
| Mergeable | 1.168216 |

Mergeable translation is 10.02% worse than raw; only 2/5 seed-aggregated
comparisons improve. Mean rotation is 0.01364878 versus raw 0.01473394 deg/m
(7.37% better). Prespecified scientific success criterion: **false**.
Development translation is 1.552750 mergeable versus 1.537872 raw; training
1.644715 versus 1.716576. No seed/checkpoint selection or retraining.

On 2026-09-11 a separate bounded post-hoc analysis of final-readout selection
was completed over 07–10, with whole-recording selector cross-fitting and frozen
base models. Its primary mean gain is only 0.24%; one drive worsens 7.17%.
See [experiment034 results](../034_kitti_selective_correction/RESULTS.md).
It neither changes the frozen 09–10 outcome nor establishes a no-harm policy.

Original implementation/input seal verifies. Manuscript/PDF/artifact remain
unchanged; scientific consolidation and final claim audit are still pending.

## Historical launch snapshot

2026-09-11. All 30 prescribed fits completed 5,000 updates without a recorded
failure. Final checkpoint identities and the implementation seal were verified.
Across all five seeds, the six module training traces use identical logged
batch schedules within each seed. Total recorded fitting time: 1,415.97 seconds
(23.60 minutes). No best-seed or checkpoint selection occurred.

### Execution milestone at launch

Training evaluation started with the unchanged sealed `evaluate.py train`.
At the first handoff checkpoint, **7/35 sequence/seed reports were complete**:
all seven training sequences for seed 0, all with `technical_pass: true`.
No evaluation failure receipt was present. Remaining seeds were running.

Across those first seven reports, float64 geometric audits observed:

- Maximum decoded tree-path translation discrepancy: 5.4051e-12 m.
- Maximum decoded tree-path rotation discrepancy: 1.1781e-14 rad.
- Maximum equivalent-matrix coefficient discrepancy: 2.0464e-12.

These are partial technical consistency results, not held-out utility or a GA
accuracy-gain claim. The complete fixed protocol and numerical bounds remain
unchanged. All eight arms and all prespecified distance/fixed-length evaluations
are retained regardless of direction.

### Stages queued at that historical milestone (now complete)

An operational supervisor waits for the full training aggregate, checks its
technical status, and invokes these unchanged sealed commands serially:

1. `stage_evaluation.py dev`
2. `evaluate.py dev`
3. `workflow.py release-holdout`
4. `stage_evaluation.py study_holdout`
5. `evaluate.py study_holdout`

It stops on a failed command or failed train/development technical checks and
does not retry or change experiments. It never inspects development gains to
decide whether to release holdout. The existing release wrapper verifies the
complete checkpoint/report/audit provenance before allowing 09–10 access.
Official KITTI test sequences 11–21 remain excluded.

The supervisor is outside the sealed research-source directory because it only
orchestrates already-authorized commands; it changes no model, data processing,
evaluation formula, checkpoint, seed, budget, or release predicate. Its original
source hash is logged in an append-only execution history.

Operational files, all under
`data/raw/kitti_odometry/learned_composition_v1/`:

- `evaluation_supervisor.py`: command orchestration.
- `evaluation_supervisor_status.json`: latest execution status (mutable view).
- `evaluation_supervisor_history.jsonl`: append-only stage history.
- `supervisor_*.log`: output from each separately executed later stage.
- `evaluation/<split>/seed<seed>/<sequence>/completed.json`: immutable reports.
- `evaluation/<split>/aggregate.json`: complete split-level results.

Supervisor source SHA-256 at launch:
`a5a495ed26751b3c79dd54198d52685f2fa8e880812e7f1adb1a7dae8a0fa64e`.

Session-local execution handles at launch: training evaluator **42778**;
supervisor **69156**. Verify current process/status before assuming either is
still active. A completed supervisor still requires scientific interpretation
of the retained results; it does not rewrite the manuscript or declare success.

At this milestone development and holdout were still scientifically unopened,
no new GA utility claim was made, and the manuscript/PDF/artifact remained
unchanged. Later stage access/status must be read from the execution history,
not inferred from this historical snapshot.
