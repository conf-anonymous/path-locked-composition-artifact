# ETH3D family-held-out evaluation

Five grouped folds over 16 capture families using only public real ETH3D data.
Read `PROTOCOL.md` before execution.

```bash
.venv/bin/python experiments/023_eth3d_family_heldout/run.py --device cpu
.venv/bin/python experiments/023_eth3d_family_heldout/audit_results.py
```

`--folds 0 --seeds 0 --steps 100` is permitted only as an implementation smoke
test and is saved separately. It cannot satisfy or modify the frozen gate.
