# ETH3D adaptive motor correction

Development-only follow-up to Experiment 021. Read `PROTOCOL.md` first.

The experiment uses only the already verified public ETH3D recordings and
derived Open3D odometry. It deliberately has no confirmation option.

```bash
.venv/bin/python experiments/022_eth3d_adaptive_motor/run.py --device cpu
.venv/bin/python experiments/022_eth3d_adaptive_motor/audit_results.py
```

Use `--seeds 0 --steps 2000` for a clearly labeled engineering pilot. Pilot
results cannot satisfy the five-seed development criterion.
