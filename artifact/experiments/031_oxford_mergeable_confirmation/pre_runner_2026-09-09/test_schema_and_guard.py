"""Actual-source schema regression and control-only denial tests; no generated poses."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from pose_csv import RTK_HEADER, VO_HEADER, read_pose_csv, recorded_rows, validate_header
from prepare_data import inspect_csv

spec = importlib.util.spec_from_file_location("oxford_schema_test_runner", HERE / "run.py")
study = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = study
spec.loader.exec_module(study)
DATA = study.DATA


class RecordedSchemaTests(unittest.TestCase):
    def test_numeric_fields_and_zone_preserved(self):
        manifest = json.loads((DATA / "manifest.json").read_text())
        self.assertEqual(len(manifest["records"]), 58)
        for record in manifest["records"]:
            for kind in ("vo", "rtk"):
                path = DATA / record[kind]["path"]
                before = hashlib.sha256(path.read_bytes()).hexdigest()
                rows = recorded_rows(path, kind)
                with path.open(newline="") as stream:
                    source = csv.reader(stream)
                    header = next(source)
                    self.assertEqual(header, RTK_HEADER if kind == "rtk" else VO_HEADER)
                    # Same existing rows, same fields; no target construction.
                    for _, original, parsed in zip(range(128), source, rows):
                        self.assertEqual(parsed[0], int(original[0]))
                        for index in range(1, len(original)):
                            if kind == "rtk" and index == 7:
                                self.assertEqual(parsed[index], original[index])
                            else:
                                self.assertEqual(parsed[index], float(original[index]))
                rows.close()
                self.assertEqual(before, record[kind]["sha256"])
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before)

    def test_numpy_pose_slices_are_numeric_and_unshifted(self):
        path = DATA / "rtk/2014-12-12-10-45-15/rtk.csv"
        header, values = read_pose_csv(path)
        with path.open(newline="") as stream:
            source = list(csv.reader(stream))[1:]
        self.assertEqual(header[7], "utm_zone")
        self.assertEqual(values[0, 7], "30U")
        for start, end in ((4, 7), (11, 14)):
            expected = np.asarray([[float(v) for v in row[start:end]] for row in source])
            np.testing.assert_array_equal(values[:, start:end].astype(np.float64), expected)

    def test_header_only_source_is_rejected(self):
        path = DATA / "rtk/2015-11-02-15-17-19/rtk.csv"
        with self.assertRaisesRegex(RuntimeError, "insufficient rtk rows"):
            inspect_csv(path, "rtk")

    def test_wrong_schema_is_rejected(self):
        # Existing VO header is not an RTK header: no fabricated observations.
        with self.assertRaisesRegex(RuntimeError, "unexpected official rtk header"):
            validate_header(VO_HEADER, "rtk", "header-validation-only")

    def test_development_selection_excludes_confirmation(self):
        manifest = json.loads((DATA / "manifest.json").read_text())
        selected = study.admitted_records(manifest)
        self.assertEqual(len(selected), 49)
        self.assertEqual({r["split"] for r in selected}, {"train", "dev"})
        self.assertFalse({r["traversal"] for r in selected} &
                         {r["traversal"] for r in manifest["records"] if r["split"] == "confirm"})

    def test_confirmation_denied_before_file_access(self):
        with patch.object(Path, "open", side_effect=AssertionError("unexpected file access")):
            with self.assertRaisesRegex(RuntimeError, "confirmation remains sealed"):
                study.OxfordData(study.Config(), include_confirmation=True)
            with self.assertRaisesRegex(RuntimeError, "confirmation remains sealed"):
                study.run("confirm", study.Config(), study.torch.device("cpu"))
            with self.assertRaisesRegex(RuntimeError, "confirmation remains sealed"):
                study.admitted_records({}, include_confirmation=True)

    def test_development_outcomes_wait_for_extension_seal(self):
        with patch.object(Path, "open", side_effect=AssertionError("unexpected file access")):
            with self.assertRaisesRegex(RuntimeError, "outcome inspection is not enabled"):
                study.run("dev", study.Config(), study.torch.device("cpu"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
