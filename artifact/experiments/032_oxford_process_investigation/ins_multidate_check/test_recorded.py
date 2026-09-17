"""Read-only tests using public recorded observations, never generated data."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('multi_check', HERE/'check.py')
c = importlib.util.module_from_spec(spec)
spec.loader.exec_module(c)


class RecordedMultiDateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = c.a.wf.read(HERE/'results.json')

    def test_provenance_and_scope(self):
        r = self.result
        for key,path in [('source_sha256',HERE/'check.py'),
                         ('plan_sha256',HERE/'PLAN.md'),
                         ('previous_source_sha256',Path(c.previous.__file__)),
                         ('investigate_source_sha256',Path(c.a.__file__))]:
            self.assertEqual(r[key],c.a.wf.sha(path))
        self.assertFalse(r['training_performed'])
        self.assertFalse(r['physical_frame_validated'])
        c.a.wf.verify_implementation()
        self.assertFalse(c.a.wf.STARTED.exists())
        self.assertFalse(c.a.wf.RELEASE.exists())
        self.assertEqual(r['original_fingerprint'],c.a.wf.fingerprint())
        for t in r['traversals']:
            c.record_for(t['traversal'],t['split'])
            raw=c.a.wf.DATA/'ins_crosscheck'/t['traversal']
            self.assertEqual(t['archive_sha256'],c.a.wf.sha(raw/(t['traversal']+'_gps.tar')))
            for name,sha in t['csv_sha256'].items():
                self.assertEqual(sha,c.a.wf.sha(raw/name))
        with self.assertRaises(AssertionError):
            c.record_for(c.SOURCES[0][0],'confirm')

    def test_original_control_exactly_reproduces(self):
        old=c.a.wf.read(HERE.parent/'ins_frame_check/results.json')
        new=next(t for t in self.result['traversals'] if t['traversal']==old['traversal'])
        for key in ('subsets','paired_status_counts','ins_minus_rtk_timestamp_ms',
                    'ins_rows','rtk_rows','ins_rows_in_rtk_interval','csv_sha256'):
            self.assertEqual(old[key],new[key])

    def test_pairing_and_status_accounting(self):
        for t in self.result['traversals']:
            self.assertNotIn('blocked',t)
            self.assertEqual(t['paired_rows']+t['unpaired_rtk_rows'],t['rtk_rows'])
            self.assertEqual(sum(t['paired_status_counts'].values()),t['paired_rows'])
            self.assertEqual(t['unique_paired_ins_rows'],t['paired_rows'])
            self.assertEqual(t['nonfinite_ins_rows_in_pairing_interval'],0)
            self.assertFalse(any(e['inside_pairing_interval'] for e in t['nonincreasing_ins_edges']))
            s=t['subsets']
            self.assertEqual(s['ins_solution_good']['n'],
                s['good_first_time_half']['n']+s['good_second_time_half']['n'])
            dt=t['ins_minus_rtk_timestamp_ms']
            self.assertLessEqual(max(abs(dt['min']),abs(dt['max'])),20)

    def test_nearest_against_exhaustive_recorded_search(self):
        for t in self.result['traversals']:
            name=t['traversal']; rec=c.record_for(name,t['split'])
            times,_,_,_=c.read_csv(c.a.wf.DATA/'ins_crosscheck'/name/'ins.csv')
            queries,_,_,_=c.read_csv(c.a.wf.DATA/rec['rtk']['path'])
            times=times[(times>=queries[0]-20000)&(times<=queries[-1]+20000)]
            queries=queries[::119]
            expected=np.array([np.argmin(abs(times-q)) for q in queries])
            np.testing.assert_array_equal(c.previous.nearest(times,queries),expected)

    def test_rotations_against_elementary_products_on_recorded_rows(self):
        for t in self.result['traversals']:
            rec=c.record_for(t['traversal'],t['split'])
            _,values,_,_=c.read_csv(c.a.wf.DATA/rec['rtk']['path'])
            rpy=values[::119,7:10]
            reference=[]
            for r,p,y in rpy:
                cr,sr,cp,sp,cy,sy=np.cos(r),np.sin(r),np.cos(p),np.sin(p),np.cos(y),np.sin(y)
                rx=np.array([[1,0,0],[0,cr,-sr],[0,sr,cr]])
                ry=np.array([[cp,0,sp],[0,1,0],[-sp,0,cp]])
                rz=np.array([[cy,-sy,0],[sy,cy,0],[0,0,1]])
                reference.append(rz@ry@rx)
            np.testing.assert_allclose(c.previous.rotation(rpy),np.array(reference),atol=1e-15,rtol=0)


if __name__=='__main__':
    unittest.main()
