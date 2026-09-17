# Original INS / RTK export cross-check

The author supplied the requested GPS archive after experiment 032 completed.
Only development traversal 2014-11-21-16-07-03 is in scope. No fitting, model
evaluation, confirmation access, interpolation, frame-angle optimization or
replacement of existing results is authorized by this diagnostic script.

Verify official MD5 D7455BFE51B17A8B42B0E8B0F40AE228. Preserve unchanged archive
and its two CSV members under raw-data/ins_crosscheck, outside the old manifest.
Pair each recorded RTK row with the nearest recorded original INS timestamp;
require separation <=20 ms. Report all paired rows and separately rows with
INS_SOLUTION_GOOD status. Heading versus velocity uses recorded speed >5 m/s.

Initial read-only inspection found approximately 90 degrees between yaw exports,
near-perfect anticorrelation between roll exports, and positive correlation
between pitch exports. Quantify those relationships and each chronological half
without optimizing any offset. Compare these fixed diagnostic interpretations:
unchanged SDK, the previous right-multiplied body quarter-turn, yaw subtraction,
and yaw subtraction with roll negation. The last is an export-consistency probe,
not a new calibrated pose target. Do not select among them using VO/model error.

Original INS and RTK are two processing products of the SAME GNSS/IMU sensor;
this is independent of the learned model, not independent physical ground truth.
INS itself is not uniformly accurate. Manufacturer output conventions and
status descriptions aid interpretation but do not specify Oxford's exact
post-processing/export or body-to-sensor rotation settings.

Loader validation found five out-of-order INS timestamp edges, all outside the
RTK interval. Report all five, limit pairing to the RTK-covered INS interval
(plus the fixed 20 ms tolerance), and require that interval to be strictly
monotonic. Do not silently reorder the entire archive or discard a conflicting
record inside the pairing interval. This does not change any paired observation.

Archive exact hashes and all results. Keep experiments 020/031, existing 032
results, the manuscript, and the original confirmation gate untouched.
