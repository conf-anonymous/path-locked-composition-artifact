# Additional discovery and diagnostic — 2026-09-10

After PLAN.md was written and the quarter-turn-only refits started, source
inspection found impossible reference dynamics, including a 1,647 m RTK step
over 0.101599 s in training traversal 2015-02-06-13-57-16. The original archive
and parsed CSV must be compared byte-for-byte before attributing this to the
released data. The frozen frame-only sensitivity continues unchanged.

This additional exploratory diagnostic is specified before its model fitting:

- Reject a training window if it spans a disconnected official VO edge.
- Reject a window if any enclosed consecutive RTK pair has position-derived
  speed >50 m/s, or either endpoint's reported 3D velocity norm >50 m/s.
  This generous physical sanity bound is not a certificate of RTK accuracy.
- Apply the same rule to report a separate development quality-screened subset;
  also evaluate ALL original development windows to expose subset dependence.
- Keep the existing body-quarter-turn diagnostic convention, timestamps,
  architecture, five seeds, 5,000 updates and optimizer. Compute the loss scale
  from retained training observations only. No checkpoint selection or tuning.
- Train all five original arms, not only the GA arm, and report all outcomes.
- No fabricated/filled observations and no source-file edits. Excluded windows
  remain in the cache, audit and unscreened evaluation.

The purpose is causal diagnosis of an invalid training process, not to discard
hard examples for a paper. Neither this screen nor the quarter-turn establishes
a validated reference frame. Confirmation stays closed. These new results
cannot retrospectively turn 020/031 into a successful prospective experiment.
