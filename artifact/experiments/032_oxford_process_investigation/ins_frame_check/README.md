# INS cross-check: the quarter-turn is not a complete frame repair

Completed 2026-09-10. Source archive
`2014-11-21-16-07-03_gps.tar` matches official MD5
`d7455bfe51b17a8b42b0e8b0f40ae228`. Archive and both CSV members are preserved
unchanged under `data/raw/oxford_robotcar/ins_crosscheck/2014-11-21-16-07-03/`.
This is an existing **development** traversal. No confirmation, model fitting,
prediction evaluation, fitted angular offset or interpolation was used.

## Observations

11,930 recorded RTK rows pair with original INS within 8.76 ms maximum absolute
offset (fixed admission tolerance 20 ms). Of those, 9,848 INS rows are marked
`INS_SOLUTION_GOOD`; 492 are `INS_ALIGNMENT_COMPLETE` and 1,590 are
`INS_BAD_GPS_AGREEMENT`. Results are reported for all pairs, good-status pairs,
and the first/second chronological halves of the good-status pairs.

On good-status pairs:

| Relationship | Observation |
| --- | ---: |
| Median RTK yaw minus INS yaw (wrapped) | +90.046046 degrees |
| Correlation of RTK roll with INS roll | -0.999725 |
| Correlation of RTK pitch with INS pitch | +0.999526 |
| Median RTK roll PLUS INS roll | +1.839509 degrees |
| Median RTK pitch minus INS pitch | -0.273975 degrees |

Thus the roll signal is sign-reversed relative to the original INS export,
with a remaining offset. Pitch tracks in the same direction. These are observed
cross-export relationships, not fitted calibration parameters we applied.
Both chronological halves reproduce the relationships; full statistics are in
`results.json`.

Four fixed interpretations, compared with original INS attitude on good pairs:

| Interpretation of RTK RPY | Median / 95th percentile attitude difference |
| --- | ---: |
| Unchanged SDK Rz(yaw) Ry(pitch) Rx(roll) | 90.088 / 90.420 degrees |
| Previous body-quarter-turn diagnostic | 2.519 / 4.535 degrees |
| Subtract pi/2 from yaw only | 3.312 / 5.244 degrees |
| Subtract pi/2 from yaw and negate roll | 1.868 / 1.965 degrees |

The fourth row is a sign-convention diagnostic derived from cross-export
behavior. It still leaves a material attitude difference. **It is not a newly
validated calibration or a reason to pick a better-performing target frame.**
The original INS and RTK share GNSS/IMU hardware and observations: agreement is
an independent check of our processing, not independent physical ground truth.
Neither an INS status flag nor a high correlation certifies the full target pose.

The original full INS file also contains five timestamp reversals, all outside
RTK coverage. The loader initially rejected these. The diagnostic now reports
them and restricts pairing to the RTK-covered interval, which is strictly
monotonic. No rows were reordered, altered or interpolated. This did not change
any previously inspected paired observation.

## Consequence for the paper

The previous 33.15% GA improvement remains exactly what it was: a result under
an explicitly unvalidated quarter-turn interpretation. This new evidence rules
out treating that interpretation as a complete physical repair. It does not
invalidate the GA product or the existing MovieLens/TUM/ETH3D evidence.
No Oxford claim was promoted, and no manuscript, original result, or gate was
changed. Learning should not be credited with recovering an undocumented
coordinate/calibration conversion.

## Next bounded check

Use two original training traversals spanning 2014 and 2015 to determine whether
the sign/offset relationship is stable; reserve another development INS export
to check it without using VO/model accuracy to choose the mapping. This is a
proposal for further validation, not an assertion that three files will resolve
all settings or authorize fitting a calibration to development targets.

| Role | GPS archive | Published size | Official MD5 |
| --- | --- | ---: | --- |
| Training, early | 2014-11-14-16-34-33_gps.tar | 21.43 MB | E09887F22A26DBED63B1AF994162834F |
| Training, late | 2015-08-13-16-02-58_gps.tar | 25.50 MB | 1EE294931920C557BEA17614280081B2 |
| Independent development export check | 2015-08-14-14-54-57_gps.tar | 20.92 MB | FD18B53EA3275405FBC51DE3EB232E90 |

Before any empirical calibration fit, specify its form, training-only source
selection, held-out validation criteria, and treatment of true/grid heading.
If these exports do not support a consistent mapping, obtain the original RTK
export and body-rotation settings from the maintainers. In either case the
original confirmation set remains closed until a separately documented corrected
protocol exists. Do not substitute original INS for RTK targets on the basis of
a better score.

## Sources and scope of authority

- [Oxford frames, INS units and VO documentation](https://robotcar-dataset.robots.ox.ac.uk/documentation/)
- [NovAtel SPAN attitude and configurable output frame](https://docs.novatel.com/OEM7/Content/SPAN_Logs/INSATT.htm)
- [NovAtel SPAN / Inertial Explorer rotation conventions](https://docs.novatel.com/Waypoint/Content/AppNotes/DeterminingRotationSPAN_IE.htm)
- [2014 training archive](https://robotcar-dataset.robots.ox.ac.uk/datasets/2014-11-14-16-34-33/)
- [2015 training archive](https://robotcar-dataset.robots.ox.ac.uk/datasets/2015-08-13-16-02-58/)
- [2015 development archive](https://robotcar-dataset.robots.ox.ac.uk/datasets/2015-08-14-14-54-57/)

Manufacturer documentation warns that named attitude angles depend on the output
frame and provides configurable sensor/body rotations. Current OEM7 documentation
explains the convention issue; it is not evidence of the exact settings used by
Oxford's older SPAN-CPT or the 2020 RTK exporter. That distinction remains open.

## Reproduction

`check.py` uses the staged archive by default, or an explicit archive path as its
single argument. It writes its JSON exclusively. For a fresh replay, copy this
subdirectory's current Python/Markdown source files into a new sibling directory
under experiment 032, then run its `check.py`; the unchanged raw inputs are reused.
`test_check.py` performs read-only verification on the saved result and recorded
timestamps. Existing 020/031 seals are checked before and after computation.
