import csv
import json
import unittest
import numpy as np
import check as c

class RecordedCrosscheckTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r=c.a.wf.read(c.HERE/'results.json')

    def test_provenance_and_original_guard(self):
        self.assertEqual(self.r['source_sha256'],c.a.wf.sha(c.HERE/'check.py'))
        self.assertEqual(self.r['plan_sha256'],c.a.wf.sha(c.HERE/'PLAN.md'))
        self.assertEqual(self.r['archive_sha256'],c.a.wf.sha(c.RAW/(c.NAME+'_gps.tar')))
        c.a.wf.verify_implementation()
        self.assertFalse(c.a.wf.STARTED.exists());self.assertFalse(c.a.wf.RELEASE.exists())

    def test_status_and_time_accounting(self):
        self.assertEqual(sum(self.r['paired_status_counts'].values()),11930)
        self.assertEqual(self.r['subsets']['ins_solution_good']['n'],9848)
        self.assertLess(max(abs(self.r['ins_minus_rtk_timestamp_ms']['min']),abs(self.r['ins_minus_rtk_timestamp_ms']['max'])),20)
        self.assertEqual(len(self.r['out_of_order_ins_edges_outside_rtk_interval']),5)

    def test_relationship_repeats_in_both_halves(self):
        for key in ('good_first_time_half','good_second_time_half'):
            s=self.r['subsets'][key]
            self.assertLess(s['angle_correlations'][0][2],-.99)
            self.assertGreater(s['angle_correlations'][1][3],.99)
            self.assertLess(abs(s['yaw_difference_deg']['median']-90),1)

    def test_nearest_matches_exhaustive_recorded_timestamp_search(self):
        with (c.RAW/'ins.csv').open() as f:times=np.array([int(r['timestamp']) for r in csv.DictReader(f)],dtype=np.int64)
        rec=next(r for r in c.a.selected_records() if r['traversal']==c.NAME)
        with (c.a.wf.DATA/rec['rtk']['path']).open() as f:queries=np.array([int(r['timestamp']) for r in csv.DictReader(f)],dtype=np.int64)
        times=times[(times>=queries[0]-20000)&(times<=queries[-1]+20000)]
        # Existing rows at fixed indices, not artificial timestamp fixtures.
        queries=queries[::119]
        expected=np.array([np.argmin(abs(times-t)) for t in queries])
        self.assertTrue(np.array_equal(c.nearest(times,queries),expected))

if __name__=='__main__':unittest.main()
