# Oxford RobotCar VO-to-RTK composition

## Current status (2026-09-09)

58 checksum-verified VO/RTK pairs pass corrected schema validation, assigned
35 train / 14 development / 9 confirmation by the original immutable rule.
Raw observations are unchanged. See `AMENDMENT_2026-09-09.md` for the textual
UTM-zone correction, original-source snapshots, and separate amendment seal.
The original seal JSON and both original scientific protocol texts are retained.

Confirmation files are no longer eagerly materialized during development.
The shared execution guard currently denies BOTH Oxford experiment execution
and confirmation access: Experiment 031's runner must be implemented, tested
on recorded ETH3D inputs, and sealed before Oxford outcome inspection.
This is a fail-closed preparation stage, NOT a completed joint release workflow.

This prospective public-data experiment uses only Oxford's released stereo
visual-odometry relative poses and independent RTK reference trajectories. Read
`PROTOCOL.md` before acquiring or processing data.

Verify the pre-acquisition protocol and implementation seal at any time:

```bash
.venv/bin/python experiments/020_oxford_robotcar_vo/verify_protocol_seal.py
```

Expected local layout:

```text
data/raw/oxford_robotcar/
  rtk/
    <traversal>/rtk.csv
  vo/
    <traversal>/vo.csv
  extrinsics/
    ins.txt
  official_archives/
    <traversal>_vo.tar
```

The RTK archive is approximately 91 MB. Individual VO archives are generally a
few megabytes; raw imagery is neither required nor eligible for this protocol.
The repository already contains the unchanged official SDK `extrinsics/ins.txt`
value, retrieved from Oxford's public SDK.
The Oxford website requires registration for downloads. Do not add credentials,
cookies, or private download URLs to the repository.

During acquisition, verify every downloaded archive against the MD5 printed on
its official Oxford download page. The preparation script then validates the
extracted schemas, enumerates the exact VO/RTK intersection, applies the
immutable traversal split, and freezes SHA-256 fingerprints. The runner refuses
files that change after that manifest is written.

```bash
.venv/bin/python experiments/020_oxford_robotcar_vo/prepare_data.py
.venv/bin/python experiments/020_oxford_robotcar_vo/test_schema_and_guard.py
.venv/bin/python experiments/031_oxford_mergeable_confirmation/seal_protocol.py
```

Do not run development or `--confirm` yet. A passing original development audit
alone is not sufficient: the unchanged Experiment 031 protocol requires all
original and extension checkpoints, development results, provenance, and the
verified joint release manifest before either study's confirmation evaluation.
