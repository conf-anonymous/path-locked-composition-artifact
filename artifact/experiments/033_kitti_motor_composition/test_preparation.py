"""Integrity, split and geometry checks using recorded KITTI training data."""
import json
import unittest
import zipfile
import acquire as a


class PreparationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r=json.loads((a.RAW/'acquisition.json').read_text())

    def test_protocol_source_and_staged_hashes(self):
        self.assertEqual(self.r['source_sha256'],a.sha(a.HERE/'acquire.py'))
        self.assertEqual(self.r['protocol_sha256'],a.sha(a.HERE/'PREPARATION_PROTOCOL.md'))
        seal=json.loads((a.RAW/'preparation_seal.json').read_text())
        self.assertEqual(seal['roles'],a.ROLES)
        self.assertEqual(seal['source_sha256'],self.r['source_sha256'])
        for name,digest in self.r['staged_sha256'].items():
            self.assertEqual(digest,a.sha(a.RAW/name))

    def test_archive_identity_and_metadata(self):
        # Full CRC and SHA were already streamed by acquire.py. Check stat and
        # ZIP inventory here without repeatedly decompressing the 22 GiB archive.
        self.assertEqual(len(self.r['archives']),4)
        for r in self.r['archives']:
            path=a.Path(r['source_path'])
            self.assertEqual(path.stat().st_size,r['bytes'])
            self.assertEqual(path.stat().st_mtime_ns,r['mtime_ns'])
            self.assertTrue(r['zip_crc_verified'])
            self.assertFalse(r['publisher_digest_verified'])
            with zipfile.ZipFile(path) as z:
                self.assertEqual(len(z.infolist()),r['member_count'])

    def test_split_and_no_heldout_extraction(self):
        self.assertEqual(self.r['numeric_sequences_inspected'],[f'{i:02}' for i in range(7)])
        self.assertFalse(self.r['held_out_scientific_inspection'])
        self.assertFalse(self.r['training_performed'])
        self.assertFalse(self.r['images_decoded'])
        for seq,role in a.ROLES.items():
            if role!='train':
                self.assertFalse((a.RAW/'dataset/poses'/f'{seq}.txt').exists())
                self.assertFalse((a.RAW/'dataset/sequences'/seq).exists())
                with self.assertRaises(AssertionError):
                    a.inspect_training(seq,{})

    def test_training_checks_replay(self):
        images=next(r['images'] for r in self.r['archives'] if r['file']=='data_odometry_gray.zip')
        for row in self.r['training_checks']:
            self.assertEqual(row,a.inspect_training(row['sequence'],images))

    def test_native_reference_provenance(self):
        r=json.loads((a.RAW/'native_reference_checks.json').read_text())
        self.assertEqual(r['acquisition_sha256'],a.sha(a.RAW/'acquisition.json'))
        self.assertEqual(r['source_sha256'],a.sha(a.HERE/'check_native_reference.py'))
        self.assertEqual(r['probe_sha256'],a.sha(a.HERE/'native_probe.cpp'))
        self.assertEqual(r['binary_sha256'],a.sha(a.RAW/'native_probe'))
        self.assertEqual(len(r['sequences']),7)
        self.assertTrue(r['training_only'])
        self.assertFalse(r['model_evaluation'])


if __name__=='__main__':
    unittest.main()
