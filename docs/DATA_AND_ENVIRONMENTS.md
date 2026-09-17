# Data and execution environments

## Dependency groups

| Task | Environment | Additional requirements |
| --- | --- | --- |
| Inventory and packaging tests | Python 3.11+, standard library | None |
| Retained-evidence audits | Python; `requirements-audit.txt` | No original dataset |
| Neural training/evaluation | Recorded: Python 3.14.7, NumPy 2.4.6, PyTorch 2.12.0, macOS ARM64 | Separately acquired recorded inputs |
| ETH3D RGB-D extraction | Python 3.12; `requirements-eth3d.txt` | Official 56 real training sequences |
| KITTI stereo extraction | `requirements-kitti.txt`; clang++ | macOS ARM64 with Rosetta; original x86_64 SSE LIBVISO2 and native devkit |

Version pins describe the recorded environments, not a cross-platform bitwise
reproducibility guarantee. The KITTI builder explicitly uses macOS `arch` and
`clang++ -arch x86_64`; it is not a tested Linux or native Windows builder.
Other platforms require a separately documented port and new build identities.

Run audits and model jobs sequentially with one Torch/BLAS thread. Do not run
training concurrently with latency measurements. The reported throughput is a
one-machine, one-thread inference measurement excluding odometry extraction.

## Public inputs

| Dataset | Official source | Role and acquisition |
| --- | --- | --- |
| MovieLens-1M | https://grouplens.org/datasets/movielens/1m/ | Place `ml-1m.zip` at the reproduction workspace's `data/raw/ml-1m.zip`; required MD5 `c4d9eecfca2ab87c1945afe126590906` |
| TUM RGB-D | https://cvg.cit.tum.de/data/datasets/rgbd-dataset | Experiment 017 downloads 47 official ground-truth trajectories |
| ETH3D SLAM | https://www.eth3d.net/slam_datasets | Experiment 021 downloads the admitted 56 real training sequences and motion-capture labels, then extracts Open3D odometry |
| KITTI odometry | https://www.cvlibs.net/datasets/kitti/eval_odometry.php | Obtain `data_odometry_gray.zip`, `data_odometry_calib.zip`, `data_odometry_poses.zip`, `devkit_odometry.zip` through the official registration process |
| LIBVISO2 | https://www.cvlibs.net/software/libviso/ | Obtain original `libviso2.zip` through its provider |
| Oxford RobotCar | https://robotcar-dataset.robots.ox.ac.uk/ | Excluded diagnostics only; follow the restricted development replay in `artifact/REPLAY.md` |

Download original data into a reproduction workspace, not the retained-evidence
repository. Dataset ZIPs, RGB-D images, pose files and derived frontend caches
are not part of the published inventory. Their omission is deliberate and
does not mean a dataset-free audit has rerun those computations.

ETH3D acquisition rejects synthetic, test, calibration, SfM and bundled TUM
archives. KITTI uses public labeled sequences 00-06 for training, 07-08 for
development and 09-10 for study holdout. Official test 11-21 is excluded from
scientific use. No KITTI evaluation-server submission is required.

The no-data KITTI audit reports 62 explicitly omitted external inputs. These
include original metadata/native sources and machine-specific frontend receipts.
New extraction produces its own receipt identities; do not substitute new hashes
into historical seals. The one omitted Oxford acquisition helper and its
original hash remain disclosed in `artifact/ANONYMITY_OMISSIONS.json`. Its absence
does not affect supporting models or the permitted read-only matrix replay.

Random model initialization, minibatch sampling, reduction-tree schedules and
clustered resampling do not generate new empirical observations or labels.
There is no observation synthesis, augmentation or gap filling in the supplied
research workflows.
