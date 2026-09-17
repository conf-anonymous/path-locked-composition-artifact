"""Read-only checks on original sources and real-image smoke outputs."""
import json
import unittest
import zipfile
import acquire as a
from build_frontend import FRONT
import frontend_smoke as smoke


class FrontendTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.first=json.loads((FRONT/'smoke_run1.json').read_text())
        cls.second=json.loads((FRONT/'smoke_run2.json').read_text())
        cls.build=json.loads((FRONT/'build.json').read_text())

    def test_original_source_and_build_provenance(self):
        smoke.check_inputs()
        self.assertFalse(self.build['upstream_sources_modified'])
        self.assertEqual(self.build['builder_sha256'],a.sha(a.HERE/'build_frontend.py'))
        self.assertEqual(self.build['archive_sha256'],a.sha(FRONT/'libviso2.zip'))
        with zipfile.ZipFile(FRONT/'libviso2.zip') as z:
            for name in self.build['upstream_sha256']:
                self.assertEqual(z.read(name),(FRONT/'upstream'/name).read_bytes())

    def test_saved_runs_and_exact_semantic_replay(self):
        summary=json.loads((FRONT/'smoke_completed.json').read_text())
        self.assertTrue(summary['semantic_repeatability_exact'])
        for name,digest in summary['run_sha256'].items():
            self.assertEqual(digest,a.sha(FRONT/name))
        self.assertEqual(len(self.first['rows']),401)
        for r in (self.first,self.second):
            self.assertEqual(r['source_sha256'],a.sha(a.HERE/'frontend_smoke.py'))
        strip=lambda row:{k:v for k,v in row.items() if k!='process_seconds'}
        self.assertEqual(list(map(strip,self.first['rows'])),list(map(strip,self.second['rows'])))

    def test_real_image_decoding_and_hashes(self):
        path,_,_=smoke.check_inputs()
        with zipfile.ZipFile(path) as z:
            for frame in (0,100,200,300,400):
                arrays,hashes=smoke.decoded_pair(z,frame)
                self.assertEqual(hashes,self.first['rows'][frame]['image_hashes'])
                self.assertEqual(list(arrays[0].shape),[self.first['rows'][frame]['height'],self.first['rows'][frame]['width']])

    def test_failure_and_scope_boundaries(self):
        for r in (self.first,self.second):
            self.assertEqual(r['sequence'],'00')
            self.assertEqual(r['role'],'train')
            self.assertFalse(r['holdout_accessed'])
            self.assertFalse(r['learned_model_training'])
            self.assertFalse(r['submission_evidence'])
            self.assertFalse(r['rows'][0]['success'])
            self.assertIsNone(r['rows'][0]['previous_to_current'])
            self.assertEqual(r['failed_edges'],[])
            self.assertTrue(all(row['success'] for row in r['rows'][1:]))
        for seq,role in a.ROLES.items():
            if role!='train':
                self.assertFalse((a.RAW/'dataset/poses'/f'{seq}.txt').exists())
                self.assertFalse((a.RAW/'dataset/sequences'/seq).exists())

    def test_geometry_and_diagnostics_replay(self):
        self.assertEqual(smoke.diagnostics(self.first['rows']),self.first['diagnostics'])
        self.assertEqual(self.first['diagnostics'],self.second['diagnostics'])

    def test_native_metric_agreement(self):
        r=json.loads((FRONT/'native_smoke_metrics.json').read_text())
        self.assertEqual(r['source_sha256'],a.sha(a.HERE/'check_smoke_metrics.py'))
        self.assertEqual(r['wrapper_sha256'],a.sha(a.HERE/'metric_stream.cpp'))
        self.assertEqual(r['smoke_run_sha256'],a.sha(FRONT/'smoke_run1.json'))
        self.assertEqual(r['binary_sha256'],a.sha(FRONT/'metric_stream'))
        self.assertLess(r['max_translation_difference_percentage_points'],1e-5)
        self.assertLess(r['max_rotation_difference_deg_per_m'],1e-5)


if __name__=='__main__':
    unittest.main()
