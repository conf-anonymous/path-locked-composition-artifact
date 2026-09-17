# Frozen confirmation protocol: ETH3D contextual motor gate

Frozen 2026-09-05 after the complete development/model-selection evaluation and
before any Experiment 022 confirmation target was loaded or evaluated. The
negative Experiment 021 study remains a separate, unchanged result.

## Selected model

Select Iteration B exactly as documented in `PROTOCOL.md`: a scalar contextual
gate between (1) raw leaves reduced by the exact PGA motor product and (2) the
frozen Experiment 021 GA leaf calibrator followed by the same exact product.
The gate is a bidirectional GRU with 32 hidden units per direction and a
32-unit GELU head. It is trained only at 32 leaves with sequence-balanced
sampling, smooth-L1 endpoint translation loss plus the quaternion surrogate,
5,000 AdamW updates, batch size 256, learning rate 1e-3, weight decay 1e-4,
and gradient clipping at one. All other settings are those in the recorded
development result.

The immutable development record is
`results_pilot_gated_0-1-2-3-4_5000.json`, SHA-256
`165ed43af1aa531f5f8c551e92751c7f127b49fbd418088e3221562be6897b44`.

Frozen checkpoint SHA-256 values are:

- seed 0: `0d0fa073cdabf8bc9b5dd49821516989ef309f550b72c456fa257aeb2c586986`
- seed 1: `39c689e0fe5b6a79b06bce99d864616c0af678b0d66a6b792f262bf0cb84ae9c`
- seed 2: `5e983f29e40eeecd6e6ce2cd68c2eca0c1e8154a1c7f5c7b90fa4d75acfd5941`
- seed 3: `6e836068b0720be2e5d72b032b033a2c964719709fa414c19e23355ec32b7a55`
- seed 4: `2c645dcc8f603606dbf9e099b1d12ca84bfe334dd908b82be8b6171a3ba59f53`

## Single confirmation evaluation

Load the eight sequences assigned to confirmation by the immutable Experiment
021 hash split. Evaluate the five checkpoints once at 4, 8, 16, and 32 leaves.
Do not train, select, early-stop, calibrate, exclude a usable sequence, alter a
front-end result, or change a checkpoint. Report all windows, all sequences,
translation and rotation metrics, per-sequence outcomes, and exactness.

At 32 leaves, confirmation succeeds only if:

1. every seed's sequence-equal mean translation error is lower than raw exact
   composition;
2. the across-seed sequence-equal mean reduction is at least 10%;
3. a prespecified pooled bootstrap that resamples the eight complete sequences
   with replacement and averages over all five seeds has a 95% delta interval
   strictly below zero; and
4. the worst float64 prediction discrepancy across left, right, and balanced
   exact-product reductions is below 1e-9 at every horizon and seed.

Use 10,000 bootstrap resamples and RNG seed 220906. Shorter horizons and
individual sequences are descriptive and cannot rescue or invalidate the
primary gate. A failure ends this ETH3D claim; there is no second confirmation
attempt or post-confirmation model revision.
