"""Engineering validation on real retained ETH3D inputs/checkpoints, never Oxford outcomes."""
from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

import oxford_workflow as wf
import analysis
import run as extension


class RecordedWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.source = wf.ROOT / "data/raw/eth3d_rgbd/derived_odometry/cables_1.npz"
        manifest = wf.read(wf.ROOT / "data/raw/eth3d_rgbd/manifest.json")
        source_record = next(r for r in manifest["records"] if r["sequence"] == "cables_1")
        assert wf.sha(cls.source) == source_record["derived_sha256"]
        cls.evidence = {str(cls.source.relative_to(wf.ROOT)): wf.sha(cls.source)}
        with np.load(cls.source, allow_pickle=False) as arrays:
            valid = arrays["pair_valid"].astype(bool)
            leaves = extension.study.v.v1.motors_from_matrices(arrays["pair_motions"])
            reference = extension.study.v.v1.motors_from_matrices(arrays["reference_poses"])
        cls.windows = {}
        for length in (4, 8, 16, 32):
            starts = [i for i in range(len(valid)-length+1) if valid[i:i+length].all()][:16]
            operands = torch.stack([leaves[i:i+length] for i in starts])
            targets = extension.ox.relative_from_absolute(reference[starts], reference[[i+length for i in starts]])
            cls.windows[length] = (operands, targets)
        cls.saved = {}
        for seed in range(5):
            parent = extension.study.HERE / "runs" / f"fold0_seed{seed}.pt"
            record = wf.read(parent.with_suffix(".json"))
            assert wf.sha(parent) == record["checkpoint_sha256"]
            merged = extension.merge.HERE / "runs" / f"fold0_seed{seed}.pt"
            merged_record = wf.read(merged.with_suffix(".json"))
            assert wf.sha(merged) == merged_record["checkpoint_sha256"]
            cls.evidence[str(parent.relative_to(wf.ROOT))] = wf.sha(parent)
            cls.evidence[str(merged.relative_to(wf.ROOT))] = wf.sha(merged)
            saved = torch.load(parent, weights_only=True)
            cal = extension.ox.LeafCalibrator()
            cal.load_state_dict({k.removeprefix("calibrator."): v for k, v in saved["calibrator"].items()})
            heads = {}
            for arm in extension.HEADS:
                heads[arm] = extension.new_head(arm, seed)
                heads[arm].load_state_dict(torch.load(merged, weights_only=True) if arm == "mergeable" else saved[arm])
            cls.saved[seed] = (cal.eval(), heads, wf.sha(parent),
                               {arm: wf.sha(merged if arm == "mergeable" else parent) for arm in extension.HEADS})
        # Real re-evaluation fixture, explicitly NOT an Oxford result or release artifact.
        cls.result = {"split": "dev", "dataset": "ETH3D engineering replay only", "seeds": {}}
        for seed, (cal, heads, cal_sha, head_sha) in cls.saved.items():
            record = {"head_sha256": head_sha, "calibrator_sha256": cal_sha, "lengths": {}}
            for length, (leaves, targets) in cls.windows.items():
                cached = extension.cache(leaves, targets, cal)
                identity = cached[1].new_zeros(cached[1].shape); identity[:, 0] = 1
                predictions = {"raw": (cached[1], None), "identity": (identity, None),
                               "ga_calibrated": (cached[2], None)}
                predictions.update({arm: extension.predict(head, cached, "cpu") for arm, head in heads.items()})
                _, translation = extension.base.decode_motor(targets)
                moving = translation.norm(dim=-1) >= 5.
                names = ["cables_1"] * len(leaves)
                record["lengths"][str(length)] = {
                    "arms": {arm: {"all": extension.metrics(pred, targets, names, torch.ones_like(moving)),
                                   "moving": extension.metrics(pred, targets, names, moving),
                                   "minimum_unclamped_real_norm": minimum}
                             for arm, (pred, minimum) in predictions.items()},
                    "numerical_audit": extension.paired_state_audit(heads["mergeable"], cal, leaves, length)}
            cls.result["seeds"][str(seed)] = record

    def test_unchanged_history(self):
        wf.verify_history()

    def test_cache_matches_independent_existing_forward(self):
        leaves, targets = self.windows[8]
        cal = self.saved[0][0]
        cached = extension.cache(leaves, targets, cal)
        with torch.no_grad():
            expected = extension.base.reduce_states(list(cal(leaves.float()).unbind(1)), extension.base.motor_product, "balanced")
        torch.testing.assert_close(cached[2], expected, rtol=0, atol=0)
        self.assertEqual(sum(p.numel() for p in extension.new_head("mergeable", 0).parameters()), 577)

    def test_all_heads_train_reproducibly_on_recorded_windows(self):
        leaves, targets = self.windows[8]
        cal = self.saved[0][0]
        cached = extension.cache(leaves, targets, cal)
        _, t = extension.base.decode_motor(targets)
        scale = max(float(t.norm(dim=-1).median()), .001)
        before = copy.deepcopy(cal.state_dict())
        for arm in extension.HEADS:
            first, second = extension.new_head(arm, 0), extension.new_head(arm, 0)
            for head in (first, second):
                record = extension.fit_head(head, cached, ["cables_1"]*len(leaves), scale, 0, "cpu", steps=2)
                self.assertTrue(np.isfinite(record["final_loss"]))
            for a, b in zip(first.parameters(), second.parameters()):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
        for name, value in cal.state_dict().items():
            torch.testing.assert_close(before[name], value, rtol=0, atol=0)

    def test_complete_real_replay_and_analysis(self):
        self.assertTrue(analysis.validate_extension(self.result, {"cables_1"}))
        result = analysis.analyze(self.result)
        self.assertFalse(result["external_architectural_confirmation"])
        self.assertTrue(result["paired_state_exactness_passes"])
        for row in result["primary_l32_translation"].values():
            self.assertEqual(len(row["per_seed_mean_delta"]), 5)
            self.assertLessEqual(row["ci98_75"][0], row["ci95"][0])
            self.assertGreaterEqual(row["ci98_75"][1], row["ci95"][1])

    def test_missing_seed_arm_horizon_and_nonfinite_fail(self):
        for mutate in (
            lambda r: r["seeds"].pop("4"),
            lambda r: r["seeds"]["0"]["lengths"].pop("32"),
            lambda r: r["seeds"]["0"]["lengths"]["8"]["arms"].pop("mergeable"),
            lambda r: r["seeds"]["0"]["lengths"]["8"]["arms"]["raw"]["all"]["clusters"]["cables_1"].update(translation_mean_m=float("nan")),
        ):
            invalid = copy.deepcopy(self.result); mutate(invalid)
            with self.assertRaises((RuntimeError, KeyError)):
                analysis.validate_extension(invalid)

    def test_checkpoint_roundtrip_tamper_and_no_overwrite(self):
        with tempfile.TemporaryDirectory(prefix="oxford-checkpoint-test-") as directory:
            path = Path(directory)/"recorded_head.pt"
            meta = {"engineering_replay_source": self.evidence}
            wf.save_checkpoint(path, {"meta": meta, "head": self.saved[0][1]["mergeable"].state_dict()})
            wf.checked_checkpoint(path, meta)
            with self.assertRaises(FileExistsError):
                wf.save_checkpoint(path, {"meta": meta})
            with self.assertRaises(RuntimeError):
                wf.checked_checkpoint(path, {"wrong_provenance": True})
            # Mutate checkpoint bytes, not observations.
            with path.open("ab") as stream: stream.write(b"integrity-test")
            with self.assertRaises(RuntimeError): wf.checked_checkpoint(path, meta)

    def test_release_denied_before_confirmation_access(self):
        with tempfile.TemporaryDirectory(prefix="oxford-release-test-") as directory:
            missing = Path(directory)/"missing.json"
            with patch.object(wf, "RELEASE", missing), patch.object(wf, "STARTED", missing), \
                 patch.object(Path, "open", side_effect=AssertionError("confirmation file access")):
                with self.assertRaisesRegex(RuntimeError, "confirmation remains sealed"):
                    wf.require_confirmation_release()
                with self.assertRaisesRegex(RuntimeError, "confirmation remains sealed"):
                    extension.run("confirm")

    def test_presence_of_release_files_does_not_authorize_standalone(self):
        with tempfile.TemporaryDirectory(prefix="oxford-standalone-test-") as directory:
            release, started = Path(directory)/"release.json", Path(directory)/"started.json"
            # Control metadata only; no benchmark or scientific data is fabricated.
            wf.save(release, {"engineering_control": "presence alone is not authority"})
            wf.save(started, {"engineering_control": "stale marker"})
            with patch.object(wf, "RELEASE", release), patch.object(wf, "STARTED", started), \
                 patch.object(wf, "COMPLETED", Path(directory)/"completed.json"), \
                 patch.object(wf, "_active_release", None), \
                 patch.object(Path, "open", side_effect=AssertionError("unexpected file access")):
                with self.assertRaisesRegex(RuntimeError, "no active one-shot"):
                    wf.require_confirmation_release()

    def test_traversal_average_is_seed_first(self):
        observed = analysis.comparison(self.result, "8", "all", "raw", "translation_mean_m")
        values = []
        for seed in self.result["seeds"].values():
            arms = seed["lengths"]["8"]["arms"]
            values.append(arms["mergeable"]["all"]["clusters"]["cables_1"]["translation_mean_m"]
                          - arms["raw"]["all"]["clusters"]["cables_1"]["translation_mean_m"])
        self.assertAlmostEqual(observed["mean_delta"], float(np.mean(values)))


if __name__ == "__main__":
    import sys
    report_path = wf.HERE / "engineering_tests_final.json"
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(RecordedWorkflowTests)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    if result.wasSuccessful():
        report = {"passed": True, "tests_run": result.testsRun,
                  "data": "unaltered recorded ETH3D cables_1 windows, lengths 4/8/16/32",
                  "source_sha256": RecordedWorkflowTests.evidence,
                  "test_source_sha256": wf.sha(Path(__file__)),
                  "training_test": "two updates per head, repeated determinism check; not Oxford training",
                  "oxford_outcomes_inspected": False}
        if not report_path.exists(): wf.save(report_path, report)
        elif wf.read(report_path) != report: raise RuntimeError("engineering test evidence changed")
    sys.exit(0 if result.wasSuccessful() else 1)
