# Development-grid boundary check, 2026-09-06

The initial sweep selected its upper boundary (10) for all three penalties.
Before fitting further candidates, fix this extension: examine weights 100,
1000, 10000 in that order per penalty; stop that penalty's expansion at the
first non-improvement in the SAME seed-zero development score. Retain all
visited candidates. If selection changes, run the new selected weight on all
five seeds and both test lengths; otherwise reuse the existing selected record.
This is a transparent post-hoc search extension, not a prospectively registered
confirmation. Only development scores determine expansion/selection, not test
accuracy. No new or altered data, no changes to the old protocol/results.

This checks an obvious search-boundary concern, not global optimality or
optimization across all regularization formulations. Save selection before
evaluating a newly selected candidate on the already-observed test split.
