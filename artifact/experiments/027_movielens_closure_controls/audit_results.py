"""Fast numerical integrity audit; passing does not mean a favorable result."""
import json
from pathlib import Path

import numpy as np

import run as study

HERE = Path(__file__).resolve().parent


def main():
    result = json.loads((HERE / "results.json").read_text())
    selection = json.loads((HERE / "selection.json").read_text())
    for relative, expected in result["seal"].items():
        assert study.sha(study.ROOT / relative) == expected, relative
    assert result["selection"] == selection["selected"]
    for kind in study.KINDS:
        candidates = [r for r in selection["candidates"] if r["kind"] == kind]
        assert {r["weight"] for r in candidates} == set(study.WEIGHTS)
        assert all(r["seed"] == 0 and "test" not in r for r in candidates)
        best = min(candidates, key=lambda r: (-np.mean(list(r["dev"]["accuracy"].values())), r["weight"]))
        assert result["selection"][kind] == best["weight"]
        records = [r for r in result["records"] if r["kind"] == kind]
        assert {r["seed"] for r in records} == set(range(5))
        for r in records:
            path = HERE / "runs" / f"{kind}_w{r['weight']:g}_seed{r['seed']}.pt"
            assert study.sha(path) == r["checkpoint_sha256"]
            for length, expected_n in (("8", 93992), ("16", 89296)):
                record = r["test"][length]
                assert record["n"] == expected_n and record["users"] == 587
                assert set(record["accuracy"]) == set(study.PATHS)
                assert all(0 <= a <= 1 for a in record["accuracy"].values())
                assert record["primitive_mse"] >= 0 and record["multilevel_mse"] >= 0
    print("PASS: all 21 recorded-data fits, dev-only weight selection, 15 selected checkpoints, and five-seed results")


if __name__ == "__main__":
    main()
