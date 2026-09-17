"""Fast verification of every immutable result record used in the paper."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

PASSING_AUDITS = (
    ("MovieLens sealed confirmation", "experiments/016_movielens_composition/audit_results.py"),
    ("TUM long-horizon confirmation", "experiments/019_tum_long_horizon_confirmation/audit_results.py"),
    ("ETH3D adaptive development", "experiments/022_eth3d_adaptive_motor/audit_results.py"),
    ("ETH3D adaptive confirmation", "experiments/022_eth3d_adaptive_motor/audit_confirmation.py"),
    ("ETH3D family-held-out evaluation", "experiments/023_eth3d_family_heldout/audit_results.py"),
    ("ETH3D calibrated direct control", "experiments/024_eth3d_calibrated_direct_control/audit_results.py"),
    ("ETH3D frozen compute profile", "experiments/025_eth3d_compute_profile/audit_results.py"),
    ("ETH3D frozen heterogeneity", "experiments/026_eth3d_frozen_heterogeneity/audit_results.py"),
    ("MovieLens initial regularization follow-up", "experiments/027_movielens_closure_controls/audit_results.py"),
    ("MovieLens final development-selected regularizers", "experiments/027_movielens_closure_controls/audit_robust.py"),
    ("MovieLens frozen query-last sensitivity", "experiments/027_movielens_closure_controls/audit_query_last.py"),
    ("ETH3D feature-matched controls and identity", "experiments/028_eth3d_attribution_controls/audit_results.py"),
    ("ETH3D fully mergeable motor-state gate", "experiments/029_eth3d_mergeable_motor_gate/audit_results.py"),
    ("ETH3D frozen six-horizon scope audit with historical Oxford metadata", "verify_scope_history.py"),
    ("Oxford excluded development records, available sealed sources and frame audit", "verify_oxford_exclusion.py"),
)


def main() -> None:
    failures: list[str] = []
    for label, relative in PASSING_AUDITS:
        completed = subprocess.run(
            [sys.executable, str(ROOT / relative)],
            cwd=ROOT,
            text=True,
            capture_output=True,
        )
        status = "PASS" if completed.returncode == 0 else "FAIL"
        print(f"{status}: {label}")
        if completed.returncode != 0:
            failures.append(label)
            print(completed.stdout)
            print(completed.stderr, file=sys.stderr)

    negative = subprocess.run(
        [sys.executable, str(ROOT / "experiments/021_eth3d_rgbd_motor/audit_results.py")],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    negative_preserved = (
        negative.returncode != 0
        and "FROZEN ETH3D GATE FAILED" in negative.stderr
        and '"passes": false' in negative.stdout
    )
    print(f"{'PASS' if negative_preserved else 'FAIL'}: ETH3D negative development result preserved")
    if not negative_preserved:
        failures.append("ETH3D negative development result")

    if failures:
        raise SystemExit("artifact verification failed: " + ", ".join(failures))
    print("PASS: supporting results and excluded Oxford records are internally consistent; Oxford is not validated accuracy evidence")


if __name__ == "__main__":
    main()
