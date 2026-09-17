# Claim-to-evidence map

Paths below are relative to `artifact/`. The separately submitted manuscript
is the controlling description of scope and metrics; it is not bundled here.
All seeds, arms and applicable horizons are
retained, including unfavorable results.

| Manuscript evidence | Implementation and records | Audit |
| --- | --- | --- |
| MovieLens path sensitivity and exact-composition comparison | `experiments/016_movielens_composition/` | `audit_results.py` in that directory |
| GA motor construction and TUM long-horizon path sensitivity | `experiments/017_tum_motor_composition/`, `018_tum_residual_motor/`, `019_tum_long_horizon_confirmation/` | Experiment 019 `audit_results.py` |
| ETH3D original negative development protocol | `experiments/021_eth3d_rgbd_motor/` | Its gate is expected to fail; the aggregate checker verifies that failure |
| Development-selected contextual correction | `experiments/022_eth3d_adaptive_motor/` | `audit_results.py`, `audit_confirmation.py` |
| Grouped family evaluation and stage-matched direct control | `experiments/023_eth3d_family_heldout/`, `024_eth3d_calibrated_direct_control/` | Per-experiment `audit_results.py` |
| Compute and heterogeneity controls | `experiments/025_eth3d_compute_profile/`, `026_eth3d_frozen_heterogeneity/` | Per-experiment `audit_results.py` |
| MovieLens query-last and intermediate-state penalties | `experiments/027_movielens_closure_controls/` | `audit_results.py`, `audit_robust.py`, `audit_query_last.py` |
| Information-matched heads, identity baseline, product-cache checks | `experiments/028_eth3d_attribution_controls/` | `audit_results.py`; cached-inference/deployment records |
| Fully mergeable paired-motor state, 577-parameter head, CPU profile | `experiments/029_eth3d_mergeable_motor_gate/` | `audit_results.py`; profile and analysis records |
| All eight arms at six frozen horizons | `experiments/030_eth3d_frozen_scope/` | `verify_scope_history.py`, preserving exact historical dependency hashes |
| KITTI outdoor replication, 30 fits, 55 reports, 125,595 rows | `experiments/033_kitti_motor_composition/`, `data/raw/kitti_odometry/learned_composition_v1/` | `verify_kitti_records.py --root . --artifact` |
| Excluded Oxford development, frame and continuity investigation | Experiments 020, 031, 032 and `iclr-2027-composition/oxford-audit-2026-09-10/` | `verify_oxford_exclusion.py`; aggregate checker; separate matrix replay |

The principal positive empirical anchors include contextual ETH3D translation
error of approximately 0.440 m against the 0.915 m identity-motion baseline,
and the compact mergeable variant's approximately 0.440 m with measured 2.17x
batch-256 throughput relative to the contextual model on the recorded CPU setup.
The gate has 577 parameters; its shared calibrator brings the total to 857.
These are not claims of global superiority or guaranteed non-harm.

KITTI's primary means are 1.0618665954166722 m raw and 1.1682159773485046 m
mergeable; the prespecified descriptive utility criterion fails. Its favorable
rotation or secondary-distance results do not replace that endpoint. Oxford
does not supply validated positive accuracy evidence, and its original nine
confirmation traversals remain uninspected.

Mathematical statements are in the manuscript and its proof appendix. Finite
numerical tree checks are distinct from a formal proof over every floating-point
input. This artifact does not claim that the GA coordinate representation
outperforms mathematically equivalent matrix or dual-quaternion implementations.
