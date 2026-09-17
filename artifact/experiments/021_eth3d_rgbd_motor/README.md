# ETH3D RGB-D motor composition

Prospectively frozen public real-data experiment using conventional RGB-D
odometry increments and independent ETH3D motion-capture targets. Read
`PROTOCOL.md` before running anything.

Only the 56 official motion-capture training sequences named in
`data_manifest.py` are eligible. ETH3D's archives are additive, so the
downloader obtains the minimal `*_mono.zip` + `*_rgbd.zip` pair for each. It
cannot request `sfm_*`, test, calibration, bundled TUM, or synthetic data.

The RGB-D front end requires Open3D 0.19.0. The repository's primary Python 3.14
environment cannot use that wheel, so use the dedicated Python 3.12 environment:

```bash
uv venv --python 3.12 .venv-eth3d
uv pip install --python .venv-eth3d/bin/python open3d==0.19.0 numpy==2.2.6
```

Then acquire and prepare the official public archives:

```bash
.venv-eth3d/bin/python experiments/021_eth3d_rgbd_motor/download_data.py
.venv-eth3d/bin/python experiments/021_eth3d_rgbd_motor/extract_odometry.py
```

Model training uses the main PyTorch environment and the frozen derived
odometry manifest:

```bash
.venv/bin/python experiments/021_eth3d_rgbd_motor/run.py --device mps
.venv/bin/python experiments/021_eth3d_rgbd_motor/audit_results.py
```

Do not run confirmation unless the development audit succeeds. If it does:

```bash
.venv/bin/python experiments/021_eth3d_rgbd_motor/run.py --device mps --confirm
.venv/bin/python experiments/021_eth3d_rgbd_motor/audit_results.py --confirm
```

Raw and derived data live under `data/raw/eth3d_rgbd/`. No outcome from that
directory is eligible until the manifest and odometry extraction complete.
