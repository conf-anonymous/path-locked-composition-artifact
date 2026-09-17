# Frozen systems diagnostic: ETH3D composition cost

Frozen on 2026-09-05 after Experiments 023--024 and before running this
diagnostic. This study may characterize cost but cannot select a model, alter an
endpoint, or change any scientific result.

## Inputs and models

- Use only the checksum-verified, recorded ETH3D operands already admitted by
  Experiment 023. No generated, perturbed, augmented, or synthetic input is
  permitted.
- Reproduce fold 0, seed 0 from the exact frozen training procedures of
  Experiments 023 and 024. Refuse benchmarking unless both models reproduce
  their stored family-level translation and rotation metrics within `1e-6`.
- Benchmark the first recorded held-out fold-0 windows at length 32. Fixed batch
  sizes are 1, 32, and 256 (or all available windows if fewer exist).

## Measurement

- Train on CPU and benchmark on CPU with one PyTorch thread.
- Use evaluation mode and inference mode, 20 warm-up calls and 100 timed calls.
- Report median and interquartile latency, throughput, parameter counts, and
  the already-recorded 25-run training-time distributions.
- Separately time left-fold and balanced-tree exact motor reduction on the same
  operands. This is an implementation measurement, not an asymptotic claim.
- Record Python, platform, processor, PyTorch, and NumPy versions.

All results are descriptive systems measurements. They do not reopen the
ETH3D freeze and are not conditions for manuscript inclusion.
