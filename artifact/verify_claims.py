"""One-command verification of the consolidated stored-evidence artifact."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    inventory = ROOT / "CONTENTS_SHA256.json"
    if inventory.exists():
        for relative, expected in json.loads(inventory.read_text()).items():
            with (ROOT / relative).open("rb") as stream:
                assert hashlib.file_digest(stream, "sha256").hexdigest() == expected, relative
        print("PASS: complete artifact byte inventory", flush=True)
    for command in ((sys.executable, "verify_prior_claims.py"),
                    (sys.executable, "verify_kitti_records.py", "--root", ".", "--artifact")):
        subprocess.run(command, cwd=ROOT, check=True)
    oxford = ROOT / "experiments/032_oxford_process_investigation"
    count = 0
    for campaign in (oxford / "runs", oxford / "quality_screened/runs"):
        records = sorted(campaign.glob("*_evaluation.json"))
        assert len(records) == 25
        for path in records:
            record = json.loads(path.read_text())
            stem = path.name.removesuffix("_evaluation.json")
            with (campaign / f"{stem}.pt").open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            integrity = json.loads((campaign / f"{stem}.integrity.json").read_text())
            assert digest == integrity["sha256"] == record["checkpoint_sha256"]
            assert set(record["lengths"]) == {"4", "8", "16", "32"}
            for paths in record["lengths"].values():
                assert set(paths) == {"left", "right", "balanced", *[f"random_{i}" for i in range(16)]}
            count += 1
    investigation = json.loads((oxford / "verification.json").read_text())
    assert not investigation["confirmation_accessed"] and investigation["original_sources_unchanged"]
    ins = json.loads((oxford / "ins_multidate_check/results.json").read_text())
    assert not ins["training_performed"] and not ins["physical_frame_validated"]
    assert sum(t["paired_rows"] for t in ins["traversals"]) == 90959
    assert sum(t["subsets"]["ins_solution_good"]["n"] for t in ins["traversals"]) == 85088
    print(f"PASS: {count} additional Oxford diagnostic checkpoints/reports and INS exclusion flags; no physical-frame validation")
    print("PASS: consolidated stored-evidence audit; KITTI utility failure and Oxford exclusion preserved")


if __name__ == "__main__":
    main()
