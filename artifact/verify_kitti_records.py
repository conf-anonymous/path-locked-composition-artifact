"""Read-only KITTI record audit; recomputes all arms/groups, not selected wins.

This verifies stored evidence and its hashes, not a fresh sensor/model replay.
In a distribution without original datasets, --artifact permits only the exact
missing inputs named and hashed in KITTI_EXTERNAL_INPUTS.json. No source,
checkpoint, outcome, or numerical-test record may silently be skipped.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

ARMS = ("identity", "raw", "calibrated", "constant", "contextual",
        "feature_direct", "feature_residual", "mergeable")
SPLITS = {"train": tuple(f"{s:02d}" for s in range(7)),
          "dev": ("07", "08"), "study_holdout": ("09", "10")}
PREFIX = "data/raw/kitti_odometry/learned_composition_v1/"
SEAL = "3ee4326a9cffbcf4f914d5e7ddd7c894739a2dfdc60256bbbb52c09a50bd640d"


def read(path):
    return json.loads(path.read_text())


def close(a, b):
    assert math.isfinite(a) and math.isfinite(b)
    assert math.isclose(a, b, rel_tol=1e-11, abs_tol=1e-11), (a, b)


def audit(root, artifact=False):
    checked, external = {}, {}
    permitted = read(root / "KITTI_EXTERNAL_INPUTS.json") if artifact else {}

    def verify(relative, expected):
        assert not Path(relative).is_absolute() and ".." not in Path(relative).parts
        path = root / relative
        # Historical dependency snapshots are explicit, never hash replacements.
        snapshot = root / "kitti-source-snapshot" / relative
        if snapshot.exists():
            path = snapshot
        if not path.exists():
            assert permitted.get(relative) == expected, ("missing input", relative)
            external[relative] = expected
            return
        if relative not in checked:
            with path.open("rb") as stream:
                checked[relative] = hashlib.file_digest(stream, "sha256").hexdigest()
        assert checked[relative] == expected, ("hash mismatch", relative)

    verify(PREFIX + "implementation_seal.json", SEAL)
    seal = read(root / PREFIX / "implementation_seal.json")
    for inventory in ("sources", "inputs"):
        for relative, digest in seal["fingerprint"][inventory].items():
            verify(relative, digest)
    assert seal["model_updates_before_seal"] == 0
    assert not seal["development_accessed"] and not seal["heldout_accessed"]
    fits = []
    for seed in range(5):
        schedules = []
        for arm in ARMS[2:]:
            folder = root / PREFIX / "fits" / f"seed{seed}" / arm
            result = read(folder / "completed.json")
            assert result["seed"] == seed and result["arm"] == arm
            assert result["steps"] == 5000
            assert result["implementation_seal_sha256"] == SEAL
            verify(str((folder / "step5000.pt").relative_to(root)), result["checkpoint_sha256"])
            schedules.append([(r["step"], r["length"], r["sample_sha256"])
                              for r in result["trace"]])
            fits.append(result)
        assert all(schedule == schedules[0] for schedule in schedules)

    reports, aggregate = {}, {}
    max_keys = ("max_translation_m", "max_rotation_rad", "max_state_coefficient",
                "max_matrix_coordinate_error")
    maxima = dict.fromkeys(max_keys, 0.)
    minimum_norm = float("inf")
    row_count, audited_windows = 0, 0
    for split, sequences in SPLITS.items():
        per_split = []
        for seed in range(5):
            for sequence in sequences:
                folder = root / PREFIX / "evaluation" / split / f"seed{seed}" / sequence
                report = read(folder / "completed.json")
                assert report["seed"] == seed and report["sequence"] == sequence
                assert report["split"] == split and report["arms"] == list(ARMS)
                assert report["technical_pass"] and report["frontend_admitted"]
                assert report["seal"] == SEAL
                for relative, digest in report["dependencies"].items():
                    verify(relative, digest)
                groups = defaultdict(list)
                with (folder / "windows.jsonl").open() as stream:
                    for line in stream:
                        row = json.loads(line)
                        assert row["valid"] and set(row["arms"]) == set(ARMS)
                        assert row["last"] > row["first"]
                        groups[f'{row["kind"]}_{row["group"]}'].append(row)
                        row_count += 1
                assert set(groups) <= set(report["summary"])
                for group in set(report["summary"]) - set(groups):
                    assert report["summary"][group]["evaluated"] == 0
                    assert report["summary"][group]["potential"] == 0
                for group, rows in groups.items():
                    summary = report["summary"][group]
                    assert len(rows) == summary["evaluated"] == summary["potential"]
                    close(summary["coverage"], 1.)
                    for arm in ARMS:
                        for metric, expected in summary["arms"][arm].items():
                            close(mean(row["arms"][arm][metric] for row in rows), expected)
                    for row in rows:
                        if row["kind"] == "distance":
                            length = int(row["group"])
                            for values in row["arms"].values():
                                close(values["translation_percent"], 100 * values["translation_m"] / length)
                                close(values["rotation_deg_per_m"], values["rotation_deg"] / length)
                assert set(report["audits"]) == {"32", "64", "128", "256", "512"}
                for check in report["audits"].values():
                    assert check["passed"]
                    if not check["evaluated"]:
                        assert not check["eligible"]
                        continue
                    assert check["schedules"] == 20
                    assert check["arms"] == ["raw", "calibrated", "mergeable"]
                    assert check["max_translation_m"] < 1e-8
                    assert check["max_rotation_rad"] < 1e-9
                    assert check["max_matrix_coordinate_error"] < 1e-8
                    assert check["min_preprojection_real_norm"] > 1e-6
                    audited_windows += check["evaluated"]
                    for key in max_keys:
                        maxima[key] = max(maxima[key], check[key])
                minimum_norm = min(minimum_norm, report["minimum_preprojection_real_norm"])
                per_split.append(report)
        saved = read(root / PREFIX / "evaluation" / split / "aggregate.json")
        for arm in ARMS:
            close(saved["primary"][arm]["mean"],
                  mean(r["summary"]["distance_100"]["arms"][arm]["translation_m"] for r in per_split))
        reports[split] = per_split
        aggregate[split] = saved

    holdout = reports["study_holdout"]
    table = {}
    for group in holdout[0]["summary"]:
        table[group] = {
            arm: {metric: mean(r["summary"][group]["arms"][arm][metric] for r in holdout)
                  for metric in holdout[0]["summary"][group]["arms"][arm]}
            for arm in ARMS}
    per_sequence = {
        seq: {arm: mean(r["summary"]["distance_100"]["arms"][arm]["translation_m"]
                       for r in holdout if r["sequence"] == seq) for arm in ARMS}
        for seq in SPLITS["study_holdout"]}
    per_seed = {
        seed: {arm: mean(r["summary"]["distance_100"]["arms"][arm]["translation_m"]
                        for r in holdout if r["seed"] == seed) for arm in ARMS}
        for seed in range(5)}
    primary = table["distance_100"]
    raw, merge = primary["raw"], primary["mergeable"]
    close(raw["translation_m"], 1.0618665954166722)
    close(merge["translation_m"], 1.1682159773485046)
    # Winning controls does not satisfy the prespecified conjunctive criterion.
    assert aggregate["study_holdout"]["descriptive_support_criterion_met"] is False
    assert minimum_norm > 1e-6
    return dict(
        scope="stored outputs, source/checkpoint hashes and arithmetic; not a fresh model/frontend replay",
        fits=len(fits), reports=sum(map(len, reports.values())), window_seed_records=row_count,
        checked_files=len(checked), external_input_files_not_replayed=len(external),
        external_inputs=external, numerical_maxima=maxima,
        audited_window_seed_cases=audited_windows, minimum_preprojection_real_norm=minimum_norm,
        study_holdout_unique_primary_segments=sum(r["summary"]["distance_100"]["evaluated"]
                                                 for r in holdout if r["seed"] == 0),
        primary_translation_change_percent=100*(merge["translation_m"]/raw["translation_m"]-1),
        primary_rotation_reduction_percent=100*(1-merge["rotation_deg_per_m"]/raw["rotation_deg_per_m"]),
        per_sequence_primary_translation_m=per_sequence, per_seed_primary_translation_m=per_seed,
        holdout_all_groups=table, scientific_success=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--artifact", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.root.resolve(), args.artifact)
    if args.output:
        with args.output.open("x") as stream:
            json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ("external_inputs", "holdout_all_groups")}, indent=2))


if __name__ == "__main__":
    main()
