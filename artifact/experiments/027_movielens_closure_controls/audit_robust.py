"""Check final three-seed selection and every selected checkpoint."""
import json

import numpy as np

import robust_selection as search
import run as study


def main():
    result = json.loads((study.HERE / "robust_results.json").read_text())
    selection = json.loads((study.HERE / "robust_selection.json").read_text())
    for name, digest in json.loads((study.HERE / "robust_seal.json").read_text()).items():
        assert study.sha(study.HERE / name) == digest
    assert len(selection["candidates"]) == 54
    for kind in study.KINDS:
        scores = {}
        for weight in search.WEIGHTS:
            candidates = [r for r in selection["candidates"] if r["kind"] == kind and r["weight"] == weight]
            assert {r["seed"] for r in candidates} == {0, 1, 2}
            assert all("test" not in r for r in candidates)
            scores[weight] = np.mean([list(r["dev"]["accuracy"].values()) for r in candidates])
        best = min(scores, key=lambda w: (-scores[w], w))
        assert result["selection"][kind] == best == selection["selected"][kind]
        records = [r for r in result["records"] if r["kind"] == kind]
        assert {r["seed"] for r in records} == set(range(5))
        for r in records:
            assert r["weight"] == best
            assert study.sha(study.HERE / "runs" / f"{kind}_w{best:g}_seed{r['seed']}.pt") == r["checkpoint_sha256"]
            for length, n in (("8", 93992), ("16", 89296)):
                assert r["test"][length]["n"] == n
                assert all(np.isfinite(a) and 0<=a<=1 for a in r["test"][length]["accuracy"].values())
    print("PASS: final 54-candidate development grid, three-seed selection, five-seed evaluation; earlier searches retained")


if __name__ == "__main__":
    main()
