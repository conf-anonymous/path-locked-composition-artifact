"""Packaging rejection checks on a small copy of real delivered source files."""
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from verify import VerificationError, canonical, verify


class RepositoryIntegrity(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="repository-check-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        files = {}
        for name in ("verify.py", "reproduce.py"):
            target = self.root / name
            shutil.copyfile(ROOT / name, target)
            target.chmod(0o644)
            files[name] = {"bytes": target.stat().st_size,
                           "sha256": hashlib.sha256(target.read_bytes()).hexdigest(), "mode": 0o644}
        self.manifest = {"schema": "anonymous-research-repository-v1", "files": files}
        self.save()

    def save(self):
        (self.root / "PACKAGE_MANIFEST.json").write_text(json.dumps(self.manifest))

    def test_pristine_and_permitted_cache(self):
        self.assertEqual(verify(self.root)["status"], "PASS")
        (self.root / "__pycache__").mkdir()
        (self.root / "__pycache__/test.pyc").write_bytes(b"cache")
        self.assertEqual(verify(self.root)["status"], "PASS")

    def test_modified_source_rejected(self):
        with (self.root / "verify.py").open("ab") as f:
            f.write(b"\n")
        with self.assertRaises(VerificationError): verify(self.root)

    def test_missing_source_rejected(self):
        (self.root / "verify.py").unlink()
        with self.assertRaises(VerificationError): verify(self.root)

    def test_unexpected_private_file_rejected(self):
        (self.root / ".env").write_text("ANONYMOUS_ONLY=true\n")
        with self.assertRaises(VerificationError): verify(self.root)

    def test_symlink_rejected(self):
        p = self.root / "verify.py"
        p.unlink(); p.symlink_to(ROOT / "verify.py")
        with self.assertRaises(VerificationError): verify(self.root)

    def test_traversal_rejected(self):
        for value in ("../escape", "/absolute", "a/../b", "a\\b", "a//b"):
            with self.subTest(value=value), self.assertRaises(VerificationError): canonical(value)

    def test_duplicate_inventory_key_rejected(self):
        p = self.root / "PACKAGE_MANIFEST.json"
        p.write_text('{"schema":"anonymous-research-repository-v1","files":{},"files":{}}')
        with self.assertRaises(VerificationError): verify(self.root)

    def test_executable_bit_rejected(self):
        (self.root / "verify.py").chmod(0o755)
        with self.assertRaises(VerificationError): verify(self.root)


if __name__ == "__main__":
    unittest.main()
