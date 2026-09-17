# Oxford audit and manuscript/artifact integration

## Outcome

The independent source-backed audit found a material validity problem, not
confirmation of the apparent large Oxford accuracy gains. The frozen pipeline
has a systematic VO/RTK displacement-frame discrepancy of approximately 90
degrees. The official SDK's homogeneous-matrix reference reproduces its raw
means, but a fixed quarter-turn diagnostic reduces raw L32 error from 102.311
to 5.170 m without learning. No authoritative corrected calibration was found.
The existing training gains therefore cannot be attributed to odometry correction.

This qualifies and supersedes the earlier conversational assessment of positive
Oxford development evidence. The former numbers remain in immutable records;
they are not erased, repaired, or repurposed as supporting accuracy evidence.
The original gate failed as recorded, and confirmation remains unopened.

## Completed work

1. Independent reference audit on all 14 development traversals, all four
   horizons and exactly matching recorded windows. It checks source hashes,
   timestamp monotonicity, actual endpoint durations, duplicates, offsets,
   matrix/motor relative transforms and saved raw means. No confirmation data,
   synthetic observations, interpolation, fitting or frame-parameter search.
2. Reviewed official SDK source at commit
   `16ce3329223ca418fe5106277b91aea8d9b672b2` and its issue 40. The latter is
   explicitly a user report, not an official calibration correction.
3. Main-text breadth limitation updated and Appendix J added, disclosing both
   the failed gate and subsequent frame-validity finding. Oxford performance
   figures are not added to positive results or the abstract. The GA motivation,
   exact paired-motor architecture and MovieLens/TUM/ETH3D evidence are preserved.
4. New anonymous artifact snapshot retains all 50 Oxford fits and outcomes,
   protocol amendments and original snapshots, gate decisions and matrix audit.
   Existing delivered ZIPs and previous manuscript PDF/LaTeX are preserved.
5. One author-identifying local acquisition helper is omitted explicitly from
   the anonymous export. Its historical hash and omission reason are supplied;
   no false claim is made to verify omitted bytes. Original source admission
   remains fail-closed in the export; a separate read-only verifier checks all
   available scientific files and records. Three old Oxford metadata hashes in
   the unchanged ETH3D scope audit resolve to exact retained historical sources.

## Verification

- Fast anonymous artifact verification: all supporting evidence checks pass;
  the original negative ETH3D result and excluded Oxford result remain negative.
- All 50 Oxford checkpoint hashes/metadata, development coverage, adjusted
  interval arithmetic, failed release decision and audit-source hashes verify.
- Full anonymous-export matrix replay on official development CSVs equals the
  archived audit report exactly. It never accesses confirmation observations.
- PDF rebuilt: 21 total pages, complete main text/conclusion on page 9; no
  overfull boxes or unresolved citations. All 21 rendered pages inspected. The
  new Oxford appendix occupies its own page; anonymous PDF author metadata empty.
- Claims/evidence matrix and local-only progress records updated to prevent
  stale Oxford-positive claims from reappearing.

## Remaining boundary

This completes the requested audit and integration, not the separate final
adversarial review. No additional model training or confirmation is authorized
by this change. Oxford would need independently justified coordinate semantics
and an explicit post-outcome corrected-study protocol before reuse as accuracy
evidence. The original protocol cannot be relabeled as passing or silently fixed.
