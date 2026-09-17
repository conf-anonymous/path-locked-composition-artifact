# Experiment 017: TUM RGB-D motor composition

This experiment uses only the official public recorded ground-truth trajectories
from the TUM RGB-D benchmark. Read `PROTOCOL.md` before running it.

```bash
python experiments/017_tum_motor_composition/download_data.py
python experiments/017_tum_motor_composition/run.py
python experiments/017_tum_motor_composition/audit_results.py
```

Only if the development audit passes without any protocol change:

```bash
python experiments/017_tum_motor_composition/run.py --confirm
python experiments/017_tum_motor_composition/audit_results.py --confirm
```

The fixed `motor` arm uses unit dual quaternions, the motor representation in
the even subalgebra of 3D projective geometric algebra. Its product is the
geometric product specialized to rigid motions. The experiment therefore tests
the paper's original GA intuition directly while separating a structural
guarantee from empirical task accuracy.
