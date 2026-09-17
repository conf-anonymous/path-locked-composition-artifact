# ETH3D frozen-model compute profile

This post-freeze diagnostic retrains only the already-fixed fold-0/seed-0 models,
verifies their stored public-data metrics, and benchmarks them on recorded
ETH3D operands.

```bash
.venv/bin/python experiments/025_eth3d_compute_profile/run.py
```
