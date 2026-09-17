"""No-data verification of the frozen query-last checkpoint diagnostic."""
import json

import numpy as np

import run as study


def main():
    result = json.loads((study.ROOT / "iclr-2027-composition/ADVERSARIAL_DIAGNOSTICS_2026-09-06.json").read_text())["movielens"]
    assert result["n_endpoints"] == 93992 and result["n_users"] == 587
    assert len(result["checkpoint_sha256"]) == 25
    for name, digest in result["checkpoint_sha256"].items():
        assert study.sha(study.m.HERE / "runs" / name) == digest
    for arm, records in result["arms"].items():
        assert {s["seed"] for s in records["seeds"]} == set(range(5))
        for path, mean in records["mean_accuracy"].items():
            assert abs(np.mean([s["accuracy"][path] for s in records["seeds"]])-mean) < 1e-12
        for seed in records["seeds"]:
            if arm in ("bilinear", "mlp", "penalty"):
                assert all(ci[1] < 0 for ci in seed["history_only_delta_ci95"].values())
            if arm == "exact":
                assert max(seed["accuracy"].values()) == min(seed["accuracy"].values())
    print("PASS: 25 immutable query-last checkpoints, all users/endpoints, paired gaps and intervals")


if __name__ == "__main__":
    main()
