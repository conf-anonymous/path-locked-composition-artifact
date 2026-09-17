# Frozen compute profile result

The fold-0/seed-0 contextual GA and calibrated direct GRU reproduced their
stored family metrics with maximum differences of `0` and `4.28e-9`,
respectively, before timing. Every timed input was a checksum-verified recorded
ETH3D length-32 window.

On macOS ARM, PyTorch 2.12, CPU, and one PyTorch thread:

| Batch | Contextual GA | Calibrated direct GRU | Exact motor, left | Exact motor, balanced |
|---:|---:|---:|---:|---:|
| 1 | 2.456 ms | 0.241 ms | 1.127 ms | 1.127 ms |
| 32 | 3.069 ms | 0.729 ms | 1.173 ms | 1.173 ms |
| 256 | 6.719 ms | 3.734 ms | 1.488 ms | 1.454 ms |

At batch 256 this corresponds to approximately 38.1k contextual-GA windows/s,
68.6k direct-GRU windows/s, and 176.1k balanced exact-motor reductions/s.
The contextual model is therefore about 1.80 times slower than the direct model
in this implementation. This is a descriptive cost, not a favorable selection
criterion.

Across the 25 already-recorded family-fold/seed runs, median primary training
time was 99.05 seconds for contextual GA and 79.32 seconds for the calibrated
direct GRU. The shared calibrator median was 18.14 seconds. These timings came
from the original experiment records and were not rerun or selected here.

The near-identical left and balanced exact-reduction times show that this eager
Python/PyTorch implementation does not realize parallel tree depth. The paper
may state the algebraic scheduling opportunity, but must not claim a measured
parallel speedup.
