"""Read-only full-training audits; every numerical input is recorded KITTI data."""
import json
import subprocess
import unittest

import numpy as np

import acquire as a
from build_frontend import FRONT
import full_frontend as full


def read_rows(sequence):
    return [json.loads(line) for line in
            (full.OUTPUT/f'{sequence}.frames.jsonl').read_text().splitlines()]


class FullFrontendTests(unittest.TestCase):
    def test_specification_and_native_provenance(self):
        full.verify_seal()
        build=json.loads((full.OUTPUT/'metric_build.json').read_text())
        self.assertEqual(build['binary_sha256'],a.sha(full.OUTPUT/'continuous_metrics'))
        self.assertEqual(build['source_sha256'],a.sha(a.HERE/'continuous_metrics.cpp'))
        for name,digest in build['original_sources_sha256'].items():
            self.assertEqual(digest,a.sha(a.RAW/'devkit/cpp'/name))

    def test_access_guard_runs_before_any_read(self):
        for sequence,role in a.ROLES.items():
            if role=='train':
                full.require_train(sequence)
                continue
            with self.assertRaises(ValueError):
                full.calibration_for(sequence)
            with self.assertRaises(ValueError):
                full.pixels_for(None,sequence,0)
            with self.assertRaises(ValueError):
                full.metrics_for(sequence,[])
            with self.assertRaises(ValueError):
                full.extract_one(sequence,None,None)
            self.assertFalse((a.RAW/'dataset/sequences'/sequence).exists())
            self.assertFalse((a.RAW/'dataset/poses'/f'{sequence}.txt').exists())

    def test_complete_sequence_prefix_exactly_repeats_smoke(self):
        smoke=json.loads((FRONT/'smoke_run1.json').read_text())['rows']
        complete=read_rows('00')[:len(smoke)]
        strip=lambda row:{k:v for k,v in row.items() if k!='process_seconds'}
        self.assertEqual(list(map(strip,complete)),list(map(strip,smoke)))

    def test_continuous_helper_matches_original_metric_on_real_smoke(self):
        smoke=json.loads((FRONT/'smoke_run1.json').read_text())
        rows=smoke['rows']
        predicted,component,_=full.predicted_components(rows)
        self.assertTrue(np.all(component==0))
        truth=np.loadtxt(a.RAW/'dataset/poses/00.txt').reshape(-1,3,4)[:len(rows)]
        common=[' '.join(format(float(v),'.17g') for v in
                         np.r_[truth[i].ravel(),predicted[i,:3].ravel()])
                for i in range(len(rows))]
        inputs=[str(len(rows))]+common
        original=subprocess.run([str(FRONT/'metric_stream')],
            input='\n'.join(inputs)+'\n',text=True,capture_output=True,check=True)
        inputs=[str(len(rows))]+['0 '+line for line in common]
        continuous=subprocess.run([str(full.OUTPUT/'continuous_metrics')],
            input='\n'.join(inputs)+'\n',text=True,capture_output=True,check=True)
        original_rows=[list(map(float,line.split())) for line in original.stdout.splitlines()]
        continuous_rows=[line.split() for line in continuous.stdout.splitlines()]
        self.assertEqual(len(original_rows),39)
        self.assertEqual(len(original_rows),len(continuous_rows))
        for old,new in zip(original_rows,continuous_rows):
            self.assertEqual(new[0],'S')
            self.assertEqual(old[:2],[float(new[1]),float(new[3])])
            # The original error record stores float32; the helper prints the
            # native function result without that extra storage rounding.
            np.testing.assert_allclose(old[2:],list(map(float,new[4:6])),rtol=2e-7,atol=1e-10)

    def test_full_records_and_native_replay(self):
        for sequence in full.TRAIN:
            with self.subTest(sequence=sequence):
                receipt=json.loads((full.OUTPUT/f'{sequence}.json').read_text())
                rows=read_rows(sequence)
                self.assertEqual(receipt['frames'],len(rows))
                self.assertEqual(receipt['frame_records_sha256'],a.sha(full.OUTPUT/f'{sequence}.frames.jsonl'))
                self.assertEqual(receipt['seal_sha256'],a.sha(full.OUTPUT/'specification_seal.json'))
                self.assertEqual(receipt['validation'],full.metrics_for(sequence,rows))
                self.assertEqual([row['frame'] for row in rows],list(range(len(rows))))
                for row in rows:
                    self.assertEqual(row['previous_to_current'] is not None,row['success'])
                    self.assertEqual(len(row['image_hashes']),2)
                for flag in ('development_accessed','heldout_accessed','ga_training_performed'):
                    self.assertFalse(receipt[flag])

    def test_window_coverage_and_means_independently(self):
        for sequence in full.TRAIN:
            with self.subTest(sequence=sequence):
                receipt=json.loads((full.OUTPUT/f'{sequence}.json').read_text())
                rows=read_rows(sequence);v=receipt['validation']
                failed=np.array([not row['success'] for row in rows])
                failed[0]=False
                failures=np.cumsum(failed)
                self.assertEqual(v['component_count'],int(failures[-1])+1)
                self.assertEqual(v['failed_edges'],np.flatnonzero(failed).tolist())
                for segment in v['segments']:
                    self.assertEqual(segment['valid'],
                        bool(failures[segment['last']]==failures[segment['first']]))
                for length,summary in v['by_length'].items():
                    candidates=[s for s in v['segments'] if s['length_m']==int(length)]
                    valid=[s for s in candidates if s['valid']]
                    self.assertEqual(summary['potential'],len(candidates))
                    self.assertEqual(summary['valid'],len(valid))
                    self.assertEqual(summary['coverage'],len(valid)/len(candidates) if candidates else None)
                    for key in ('raw_translation_percent','raw_rotation_deg_per_m',
                                'identity_translation_percent','identity_rotation_deg_per_m'):
                        self.assertEqual(summary[key],float(np.mean([s[key] for s in valid])) if valid else None)
                for length,count in v['fixed_length_training_window_counts'].items():
                    length=int(length)
                    self.assertEqual(count,int(np.sum(failures[length:]==failures[:-length])))
                p=v['by_length']['100']
                independently_admitted=(np.mean(~failed[1:])>=.95 and p['valid']>0
                    and p['coverage']>=.8 and p['raw_translation_percent']<10
                    and p['raw_rotation_deg_per_m']<.2
                    and p['raw_translation_percent']<p['identity_translation_percent'])
                self.assertEqual(v['admitted'],independently_admitted)

    def test_completion_receipt_and_scope(self):
        receipt=json.loads((full.OUTPUT/'completed.json').read_text())
        self.assertEqual(receipt['training_sequences'],list(full.TRAIN))
        self.assertEqual(receipt['specification_seal_sha256'],a.sha(full.OUTPUT/'specification_seal.json'))
        admitted=[]
        for name,digest in receipt['records_sha256'].items():
            self.assertEqual(digest,a.sha(full.OUTPUT/name))
            admitted.append(json.loads((full.OUTPUT/name).read_text())['validation']['admitted'])
        self.assertEqual(len(admitted),7)
        self.assertEqual(receipt['all_training_frontend_gates_pass'],all(admitted))
        for flag in ('ga_training_performed','development_accessed','heldout_accessed',
                     'training_runner_implemented','holdout_release_implemented'):
            self.assertFalse(receipt[flag])

    def test_segment_inventory_against_independent_numpy_selector(self):
        total=0
        for sequence in full.TRAIN:
            poses=np.loadtxt(a.RAW/'dataset/poses'/f'{sequence}.txt').reshape(-1,3,4)
            delta=np.diff(poses[:,:,3],axis=0).astype(np.float32)
            steps=np.sqrt((delta[:,0]**2+delta[:,1]**2)+delta[:,2]**2)
            distance=np.r_[np.float32(0),np.cumsum(steps,dtype=np.float32)]
            expected=[]
            for start in range(0,len(poses),10):
                for length in range(100,801,100):
                    threshold=np.float32(distance[start]+np.float32(length))
                    end=int(np.searchsorted(distance,threshold,side='right'))
                    if end<len(poses):
                        expected.append((start,end,length))
            receipt=json.loads((full.OUTPUT/f'{sequence}.json').read_text())
            observed=[(s['first'],s['last'],s['length_m']) for s in receipt['validation']['segments']]
            self.assertEqual(observed,expected)
            total+=len(observed)
        self.assertEqual(total,10015)


if __name__=='__main__':
    unittest.main()
