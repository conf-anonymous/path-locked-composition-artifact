# Oxford joint architectural confirmation

Active implementation: `IMPLEMENTATION_2026-09-09.md`. The original scientific
protocol and historical seals are unchanged. Historical status strings in older
seals describe those milestones, not the active implementation state.

```sh
.venv/bin/python experiments/031_oxford_mergeable_confirmation/seal_implementation.py
.venv/bin/python experiments/031_oxford_mergeable_confirmation/joint.py development --device cpu
```

The development command runs/resumes all original fits, then all extension fits
and evaluations. A valid release manifest is produced only if the original gate
passes and all original/extension outputs and checkpoints are complete and
immutable. A failed scientific gate is recorded and leaves confirmation closed.
The command NEVER automatically runs confirmation.

Only after a successful joint release:

```sh
.venv/bin/python experiments/031_oxford_mergeable_confirmation/joint.py confirm --device cpu
```

Confirmation is one-shot. A start marker records the release and process;
standalone `--confirm` calls and a second joint invocation fail closed. A crash
after the start marker requires an explicit documented recovery procedure, not
deleting the marker and rerunning. The mechanism protects research workflow
integrity, not against a malicious user editing their own local code.

Data are official recorded VO/RTK trajectories; no synthetic observations,
interpolation, new odometry front end, or augmentation. Engineering tests use
retained real ETH3D windows and checkpoints and are not Oxford scientific results.
No head, seed, horizon, or checkpoint is selected based on development utility.
