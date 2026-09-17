# Oxford INS / RTK: discrepancy replicated across four dates

Completed 2026-09-10. All three newly supplied GPS archives match the official
MD5s recorded before this check. Exact archives and both CSV members are staged
under `data/raw/oxford_robotcar/ins_crosscheck/<traversal>/`, outside the sealed
manifest. The original single-date result reproduces exactly.

**Outcome:** the export relationship is systematic across the inspected dates,
not peculiar to the first traversal. This strengthens the diagnosis of a
frame/convention problem. It does **not** validate the previous quarter-turn
repair or turn its 33.15% diagnostic GA gain into paper evidence.

## Fixed-protocol source observations

90,959 recorded timestamp pairs across four traversals, including 85,088 pairs
marked INS_SOLUTION_GOOD. Every paired row is an actual recording; no generated,
interpolated, augmented, or synthetic measurements were used. The same <=20 ms
nearest-timestamp rule and four fixed attitude interpretations were applied to
every traversal. No angular offsets were fitted, no model was run, and no VO or
confirmation observations were read.

The following are medians over good-status pairs; all angles are degrees.

| Traversal | Existing split | Good pairs | RTK roll + INS roll | RTK pitch - INS pitch | Wrapped RTK yaw - INS yaw | Attitude disagreement after fixed yaw/roll-sign probe |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 2014-11-14-16-34-33 | train | 24,187 | 1.841005 | -0.263107 | 90.185279 | 1.872386 |
| 2015-08-13-16-02-58 | train | 28,041 | 1.841207 | -0.259351 | 90.254038 | 1.884194 |
| 2015-08-14-14-54-57 | dev | 23,012 | 1.841635 | -0.261779 | 90.192046 | 1.877434 |
| 2014-11-21-16-07-03 | dev, previous control | 9,848 | 1.839509 | -0.273975 | 90.046046 | 1.868265 |

The fixed probe subtracts pi/2 from exported RTK yaw and negates exported RTK
roll before the SDK-style Rz Ry Rx construction. It is not an empirically fitted
calibration or an authoritative conversion specification.

On all four good-status subsets, roll correlations range from -0.999855 to
-0.999708, and pitch correlations from +0.999526 to +0.999754. Both chronological
halves reproduce the roll sign reversal and pitch sign agreement on every date.
The whole-traversal median roll sum spans just 0.002126 degrees. Pitch-difference
medians span 0.014624 degrees. These stable relationships are consistent with a
shared export convention and residual calibration difference, but do not prove
which frame each export represents.

Yaw needs more care: good-status traversal medians span 90.046046–90.254038
degrees, and the two time halves of the 2015 training traversal differ by
approximately 0.497404 degrees. Per-row yaw spread is larger still. Do not
interpret a stable overall median as a fully constant three-axis transform.
The known true/grid heading issue is not resolved by this check.

## Coverage and source anomalies

| Traversal | Paired / RTK rows | Maximum absolute admitted offset |
| --- | ---: | ---: |
| 2014-11-14-16-34-33 | 25,041 / 25,041 | 8.862 ms |
| 2015-08-13-16-02-58 | 29,670 / 29,671 | 8.895 ms |
| 2015-08-14-14-54-57 | 24,318 / 24,321 | 20.000 ms |
| 2014-11-21-16-07-03 | 11,930 / 11,930 | 8.760 ms |

The four unpaired RTK rows fail the unchanged 20 ms tolerance; none was
interpolated. Every admitted pair uses a distinct INS observation. Full status
counts and all-pair results, not just good-status subsets, are in `results.json`.
The three new archives have no nonincreasing INS timestamp edges or nonfinite
INS observations in the fields inspected. The original control has five
timestamp reversals and 3,391 nonfinite INS rows, all outside the RTK pairing
interval. No interval rows were sorted, deduplicated, or removed for those issues.

The first invocation selected an environment without torch and did not read
data. A subsequent invocation initially checked the entire control archive for
finiteness and aborted before writing results. The documented loader correction
matches the old check's interval-scoped finiteness rule while explicitly reporting
the full-archive anomalies. The corrected control reproduces every prior subset
and timing statistic exactly.

## Verification and limits

Five recorded-data tests pass: source/plan/archive fingerprints and scope guards;
exact original-control reproduction; timestamp/status accounting; nearest search
against exhaustive recorded-timestamp search; and matrix construction against
independent elemental rotation products on recorded angles. A fresh full replay
also reproduces `results.json` byte-for-byte. Original 020/031 implementation
seals, manuscript TeX/PDF hashes, and the unopened confirmation gate are unchanged.

These are many time-correlated rows from only four traversals, not 90,959
independent experiments. INS and RTK are processing products of the same physical
GNSS/IMU observations. INS_SOLUTION_GOOD does not make INS independent or perfect
ground truth. No physical acceptance threshold was prevalidated for this check.
Cross-export agreement cannot itself establish correct VO-to-target geometry.

## Next decision

No further GPS downloads are needed to answer the present cross-date question:
the relationship repeats. The next task is full-frame validation, before another
neural training campaign. Two complementary paths are available:

1. Obtain the actual RTK export convention, INS/body rotation settings, and
   true-versus-grid heading definition from the dataset maintainers. This is the
   strongest way to resolve the physical contract; see `maintainer_questions.md`.
2. If investigating an empirical conversion, first write a separate protocol
   for a physically constrained rotation mapping fitted only on the two training
   exports, with fixed validation criteria and explicit true/grid treatment.
   Development data here are already inspected: later validation would be
   post-selection development evidence, not untouched confirmation. Do not fit
   independent per-traversal Euler offsets, pick the best VO score, or reopen the
   original confirmation gate to bypass the failed original experiment.

Even successful frame validation would not erase the RTK quality problems,
missing VO continuity checks, or L32 learned-comparator competence failure.
A corrected experiment must address and disclose those separately. The GA
composition mechanism and existing non-Oxford evidence remain unchanged.

## Reproduce

From the repository root, using the environment that contains torch:

```sh
.venv/bin/python experiments/032_oxford_process_investigation/ins_multidate_check/check.py --downloads /path/to/downloads --output /private/tmp/oxford-ins-multidate-new-replay.json
.venv/bin/python experiments/032_oxford_process_investigation/ins_multidate_check/test_recorded.py -v
```

Choose a nonexistent output path: completed results are never overwritten.
The download directory must contain the three new archives and original control.
Archive identities and source/plan hashes are embedded in `results.json`.

Official archive sources are linked in the
[preceding cross-check](../ins_frame_check/README.md#next-bounded-check).
