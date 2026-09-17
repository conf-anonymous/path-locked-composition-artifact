# Experiment 018: residual motor composition

This is the prospectively frozen successor to failed development pilot 017.
It uses the same official TUM files and keeps confirmation unopened until its
own development gate passes.

```bash
.venv/bin/python experiments/018_tum_residual_motor/run.py
.venv/bin/python experiments/018_tum_residual_motor/audit_results.py
```

Only after a passing audit:

```bash
.venv/bin/python experiments/018_tum_residual_motor/run.py --confirm
.venv/bin/python experiments/018_tum_residual_motor/audit_results.py --confirm
```
