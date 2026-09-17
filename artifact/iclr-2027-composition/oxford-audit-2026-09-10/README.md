# Oxford reference audit: validity failure, not positive benchmark evidence

Date: 2026-09-10. Post-outcome, read-only scientific diagnosis on the 14 existing
development traversals. No fitting, new data, interpolation, angle optimization,
confirmation access, or modification of frozen sources/results was performed.

## Findings

1. All four horizons reproduce the saved original raw-error means and window
   counts with independent homogeneous matrices. Maximum per-traversal mean
   discrepancy is 2.0842e-10 m. Maximum relative-transform matrix disagreement
   with the motor implementation is 2.1925e-12 for VO and 9.2041e-9 for RTK;
   the latter uses large UTM coordinates. This validates arithmetic against the
   SDK, not the physical correctness of the source-to-target frame contract.
2. The frozen alignment selected no duplicate RTK rows. Maximum RTK-to-grid lag
   was .138473 s, and L32 RTK endpoint durations ranged from 15.872374 to
   16.128387 s. The manifest's VO-to-RTK offset rule is checked independently.
3. Median moving-window VO/target displacement angles range from 89.9098 to
   91.3075 degrees across all 14 traversals. This is a systematic frame concern,
   not ordinary accumulated VO drift or a GA multiplication discrepancy.
4. Without learning, the fixed quarter-turn diagnostic reduces the
   traversal-equal L32 raw mean from 102.311022 to 5.170367 m. At L4/L8/L16 its
   errors are .740814/1.401170/2.672389 m. These are **not corrected-protocol
   results**: the new frame convention lacks authoritative calibration support.
5. Consequently, neither the original nor extension learned gains may be
   interpreted as validated odometry improvement. The frozen original gate
   also failed independently of this audit. No confirmation is authorized.

## Reference and interpretation

Official SDK commit: `16ce3329223ca418fe5106277b91aea8d9b672b2`.
Unmodified reference source hashes are in `audit_results.json`.

- [SDK transform conversion](https://github.com/ori-mrg/robotcar-dataset-sdk/blob/16ce3329223ca418fe5106277b91aea8d9b672b2/python/transform.py)
- [SDK accumulation and RTK columns](https://github.com/ori-mrg/robotcar-dataset-sdk/blob/16ce3329223ca418fe5106277b91aea8d9b672b2/python/interpolate_poses.py)
- [SDK frame usage](https://github.com/ori-mrg/robotcar-dataset-sdk/blob/16ce3329223ca418fe5106277b91aea8d9b672b2/python/project_laser_into_camera.py)
- [Earlier report of 90-degree RTK axis discrepancy](https://github.com/ori-mrg/robotcar-dataset-sdk/issues/40)

The issue and its two comments contain no maintainer-endorsed calibration fix.
It corroborates the observed problem; it does not justify silently changing the
source-to-target contract. SDK source is licensed CC BY-NC-SA 4.0; original
copyright and license notices remain intact in the reference files.

The diagnostic uses `T_world_rtk @ Rz(-pi/2) @ inverse(T_vehicle_ins)` before
forming relative targets. It was motivated by the observed discrepancy and
prior issue report, **after** outcome inspection, without fitting any angle.
The original target construction, checkpoints and gate remain unchanged.
Its error is not directly comparable as a model ranking against predictions
trained on the old target convention.

## Reproduction and disposition

`audit.py` checks the original local seal before and after reading development
records and writes `audit_results.json` exclusively. Preserve that report
before replaying in a separate copy. It imports the frozen loader solely to
compare transforms; its matrix accumulation and metric computation are separate.
The reference interpolator is retained for source inspection, never invoked.

Do not rerun or repair Oxford training under the existing protocol. A future
study needs independent frame justification, explicit post-outcome correction
history and a new development protocol before any decision about confirmation.
The current ICLR evidence remains MovieLens, TUM and ETH3D; Oxford is disclosed
only as an excluded attempt. No claims about GA's coordinate superiority follow.
