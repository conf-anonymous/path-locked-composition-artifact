# Oxford process investigation — 2026-09-10

Post-outcome diagnosis requested by the author. This is NOT an amendment to
020/031, not a confirmation run, and not yet a corrected scientific benchmark.
All original sources, checkpoints, results and gates remain unchanged.

## Questions and fixed diagnostic sequence

1. Verify exact public VO/RTK file hashes for the existing 34 training and 14
   development traversals; never read confirmation observations.
2. Check RTK yaw against its own recorded horizontal velocity (speed >5 m/s),
   position finite differences against recorded velocity, timestamp ordering,
   VO edge source/destination continuity, and original window gap incidence.
3. Reconstruct SE(3) with independently implemented homogeneous matrices and
   reproduce the original raw development means. Report all 14 traversals and
   all lengths 4/8/16/32; no selection by outcome.
4. Separately quantify fixed-quarter-turn, nearest recorded VO timestamps,
   and exclusion of windows crossing disconnected VO edges. No interpolation,
   fabricated motion, fitted alignment, angle search, augmentation or new data.
   The quarter-turn is the already inspected diagnostic Rz(-pi/2), NOT a
   manufacturer-certified extrinsic. Compare a world-yaw subtraction as a
   sensitivity, not a second candidate selected on predictive accuracy.
5. Refit all five original arms and seeds on the quarter-turn-only target
   interpretation, holding architecture, 5,000 updates, loss, sampling, seeds,
   windows and timestamps fixed. This isolates whether the known frame problem
   explains the earlier failure of comparator competence. It does not establish
   the physical validity of the quarter-turn convention. Also report strict
   connected-edge subsets without selecting checkpoints on them.
6. Do not train more variants or change the optimizer to chase a positive
   result. Preserve every fit, including failures. Do not open confirmation,
   issue a release, modify the submission PDF, or promote exploratory results.

The current SDK references reproduce arithmetic, but that is not independent
validation of the physical frame. Original GPS/INS for a previously inspected
development traversal has been requested to cross-check the RTK export.

Source provenance: Oxford official documentation and SDK commit
16ce3329223ca418fe5106277b91aea8d9b672b2; SDK issues 40 and 55 contain reports,
not maintainer-certified fixes. NovAtel documents Inertial Explorer's nominal
X-right/Y-forward/Z-up frame; this alone does not identify Oxford's export.

Original confirmation remains unavailable to this investigation regardless of
the diagnostic results. A future corrected study needs an explicit frame
contract and a separately documented decision about its held-out evaluation.
