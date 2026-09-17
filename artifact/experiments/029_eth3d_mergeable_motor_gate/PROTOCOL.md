# Mergeable GA summary ablation (2026-09-06)

Motivation: adversarial finding 1 distinguishes associative motor products from
the full bidirectional-GRU context model. Test a fully mergeable alternative,
not merely a wording correction. This is a post-hoc architectural ablation on
already inspected public ETH3D data, not new untouched confirmation.

For each unchanged Experiment 028 fold/seed, reuse its exactly reproduced
Experiment 023 leaf calibrator. Represent a recorded sequence by the pair of
exact products (raw motor, calibrated motor). Merge pairs componentwise with
the GA motor product. This fixed 16-coordinate state is associative; a 16->32->1
GELU/sigmoid gate reads only that state and blends the same two endpoint motors.
No GRU, sequence identifier or reference endpoint enters the head. Its last
layer starts at zero, bias -2. Use the same 5,000 updates, sequence-balanced
batches, seeds 50000+seed, loss, optimizer, and clipping as Experiment 028.
Train all 25 fold/seed heads without tuning or selection on held-out families.

Report translation and rotation, paired family/seed/fold summaries against
identity, constant gate, and full contextual gate; include all unfavorable
outcomes. If utility declines, this is a measured mergeability/accuracy tradeoff,
not a reason to change the protocol. Report head parameter count and separate
meters/radians discrepancies across left/right/balanced/chunked products.
The final decoded prediction is a readout, not itself the composable summary.
Complexity applies to the paired product state plus its fixed-size readout;
recalibrating all leaves after a parameter change is not a local O(log L) edit.

Only unchanged public recorded ETH3D inputs and labels are admitted. No
generated observations, simulations, perturbations, augmentation or new data.
No changes to frozen Experiments 021--028 or the Oxford protocol.
