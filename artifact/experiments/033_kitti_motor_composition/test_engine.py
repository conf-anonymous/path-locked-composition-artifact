"""Preseal tests using only the recorded, admitted KITTI training sequences."""
import json
import unittest
import torch
import numpy as np

import acquire as a
import engine as e
import full_frontend as full
import stage_evaluation as stage
import workflow as w


class EngineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        e.configure();cls.data=e.TrainingData();cls.seq=cls.data.sequences['00']

    def test_recorded_leaf_direction_and_matrix_roundtrip(self):
        rows=[json.loads(line) for line in (full.OUTPUT/'00.frames.jsonl').read_text().splitlines()]
        for index in (0,100,200,1000,3000):
            m=np.eye(4);m[:3]=np.array(rows[index+1]['previous_to_current']).reshape(3,4)
            np.testing.assert_allclose(e.matrices(self.seq.leaves[index]).numpy(),np.linalg.inv(m),atol=1e-12,rtol=0)
            self.assertGreater(float(self.seq.leaves[index,0]),0)

    def test_window_labels_scale_and_inventory(self):
        for seq in self.data.sequences.values():
            for length in e.LENGTHS:
                self.assertEqual(len(seq.starts[length]),seq.receipt['validation']['fixed_length_training_window_counts'][str(length)])
                for start in (int(seq.starts[length][0]),int(seq.starts[length][-1])):
                    leaves,target=seq.window(start,length,torch.float64)
                    self.assertTrue(torch.equal(leaves,seq.leaves[start:start+length]))
                    expected=np.linalg.solve(seq.truth[start],seq.truth[start+length])
                    np.testing.assert_allclose(e.matrices(target).numpy(),expected,atol=4e-7,rtol=0)
                    with self.assertRaises(ValueError):seq.window(seq.n-length,length)
        all_lengths=np.concatenate([np.linalg.norm(s.truth[s.starts[128]+128,:3,3]-s.truth[s.starts[128],:3,3],axis=-1)
                                    for s in self.data.sequences.values()])
        self.assertEqual(self.data.scale,max(.001,float(np.median(all_lengths))))

    def test_sampler_reproducibility_and_recorded_sources(self):
        for seed in e.SEEDS:
            left=torch.Generator().manual_seed(30000+seed);right=torch.Generator().manual_seed(30000+seed)
            x,y,indices=self.data.sample(left);xx,yy,other=self.data.sample(right)
            self.assertEqual(indices,other);self.assertTrue(torch.equal(x,xx));self.assertTrue(torch.equal(y,yy))
            self.assertEqual(x.shape,(64,indices[0][2],8))
            self.assertEqual(len(set(length for _,_,length in indices)),1)
            for i,(name,start,length) in enumerate(indices):
                actual,target=self.data.sequences[name].window(start,length)
                self.assertTrue(torch.equal(x[i],actual));self.assertTrue(torch.equal(y[i],target))

    def test_batched_balanced_reducer_matches_original(self):
        for length in e.AUDIT_LENGTHS+(33,101):
            leaves=self.seq.leaves[:length].unsqueeze(0).float()
            expected=e.base.reduce_states(list(leaves.unbind(1)),e.base.motor_product,'balanced')
            self.assertTrue(torch.equal(e.product(leaves),expected))

    def test_original_models_parameters_and_backward(self):
        x,y,_=self.data.sample(torch.Generator().manual_seed(30000))
        counts={'calibrated':280,'constant':1,'contextual':10689,'feature_direct':10920,'feature_residual':10920,'mergeable':577}
        cal=e.make_module('calibrated',0)
        cal_leaves,_=e.calibrated_leaves(cal,x)
        self.assertTrue(torch.equal(cal_leaves,cal(x)))
        raw=e.product(x);corrected=e.product(cal_leaves).detach()
        for arm in e.MODULES:
            model=e.make_module(arm,0)
            self.assertEqual(sum(p.numel() for p in model.parameters()),counts[arm])
            result,norm=e.forward(arm,x,model if arm=='calibrated' else cal,model,
                                   None if arm=='calibrated' else (raw,corrected))
            if arm!='calibrated':
                self.assertTrue(torch.equal(result,model(x,raw,corrected)))
            value=e.loss(result,y,self.data.scale)
            self.assertTrue(torch.isfinite(value));self.assertGreater(norm,1e-6)
            self.assertTrue(torch.equal(value,e.loss(result,-y,self.data.scale)))
            value.backward()
            self.assertTrue(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in model.parameters()))
            self.assertTrue(any(bool(p.grad.abs().max()>0) for p in model.parameters()))
            # Deliberately NO optimizer.step: preseal tests do not fit a model.

    def test_native_raw_metric_agreement(self):
        primary=[s for s in self.seq.receipt['validation']['segments'] if s['length_m']==100]
        for segment in (primary[0],primary[100],primary[-1]):
            first,last=segment['first'],segment['last']
            x,_=self.seq.window(first,last-first,torch.float64)
            prediction=e.product(x[None])
            target=np.linalg.solve(self.seq.truth[first:first+1],self.seq.truth[last:last+1])
            t,r=e.errors(prediction,target)
            # Native translationError and rotationError explicitly cast matrix
            # coefficients to float32; the neural evaluator remains float64.
            # Same cross-precision tolerance as the earlier native smoke audit.
            self.assertLess(abs(float(t[0])-segment['raw_translation_percent']),1e-5)
            self.assertLess(abs(float(r[0]/100)-segment['raw_rotation_deg_per_m']),1e-5)

    def test_geometric_tree_and_matrix_audits_on_recorded_motion(self):
        cal=e.make_module('calibrated',0,torch.float64);gate=e.make_module('mergeable',0,torch.float64)
        for length in (32,512):
            leaves=torch.stack([self.seq.leaves[start:start+length] for start in (0,100)])
            report=e.audit_window(leaves,cal,gate)
            self.assertTrue(report['passed'],report)
            self.assertEqual(report['schedules'],20)

    def test_guarded_frontend_validator_exact_replay(self):
        rows=[json.loads(line) for line in (full.OUTPUT/'00.frames.jsonl').read_text().splitlines()]
        self.assertEqual(stage.validate_recording('00',rows),self.seq.receipt['validation'])

    def test_evaluation_access_denied_before_seal_or_release(self):
        for sequence in ('11','12','21','invalid'):
            with self.assertRaises(PermissionError):e.Sequence(sequence)
            with self.assertRaises(PermissionError):stage.extract(sequence)
        if not (e.RUNS/'holdout_release.json').exists():
            for sequence in ('09','10'):
                with self.assertRaises(PermissionError):e.Sequence(sequence)
        if not (e.RUNS/'implementation_seal.json').exists():
            for sequence in ('07','08'):
                with self.assertRaises(PermissionError):stage.extract(sequence)
            with self.assertRaises(PermissionError):w.train()
            with self.assertRaises(PermissionError):w.release_holdout()

    def test_sampler_state_resume_without_fitting(self):
        generator=torch.Generator().manual_seed(30002)
        self.data.sample(generator);saved=generator.get_state()
        expected=self.data.sample(generator)
        restored=torch.Generator();restored.set_state(saved)
        actual=self.data.sample(restored)
        self.assertEqual(expected[2],actual[2])
        self.assertTrue(torch.equal(expected[0],actual[0]))

    def test_eight_arm_summary_of_real_recorded_windows(self):
        import evaluate
        cal=e.make_module('calibrated',0);heads={arm:e.make_module(arm,0) for arm in e.HEADS}
        segments=[s for s in self.seq.receipt['validation']['segments'] if s['length_m']==100][:3]
        records=[]
        with torch.no_grad():
            for segment in segments:
                first,last=segment['first'],segment['last'];x,_=self.seq.window(first,last-first)
                truth=np.linalg.solve(self.seq.truth[first:first+1],self.seq.truth[last:last+1])
                row=dict(kind='distance',group='100',valid=True,edges=last-first,
                         duration_seconds=float(self.seq.times[last]-self.seq.times[first]),arms={})
                for arm in e.ARMS:
                    pred,_=e.forward(arm,x[None],cal,heads.get(arm));t,r=e.errors(pred,truth)
                    row['arms'][arm]=dict(translation_m=float(t[0]),rotation_deg=float(r[0]),
                                         translation_percent=float(t[0]),rotation_deg_per_m=float(r[0]/100))
                records.append(row)
        result=evaluate.summarize(records)['distance_100']
        self.assertEqual(result['potential'],3);self.assertEqual(result['evaluated'],3)
        self.assertEqual(set(result['arms']),set(e.ARMS))
        for arm in e.ARMS:
            for metric in result['arms'][arm]:
                self.assertEqual(result['arms'][arm][metric],float(np.mean([r['arms'][arm][metric] for r in records])))

    def test_fingerprint_stability_across_model_initialization(self):
        if not (e.RUNS/'training_data.json').exists():
            self.skipTest('Training data receipt is created before seal tests')
        before=w.fingerprint()
        x,y,_=self.data.sample(torch.Generator().manual_seed(30000))
        cal=e.make_module('calibrated',0);head=e.make_module('contextual',0)
        result,_=e.forward('contextual',x,cal,head);e.loss(result,y,self.data.scale).backward()
        # Initialize AdamW's lazy dependencies without performing an update.
        torch.optim.AdamW(head.parameters(),lr=.001)
        self.assertEqual(before,w.fingerprint())
        self.assertFalse(any(name.startswith('.venv/') for name in before['sources']))


if __name__=='__main__':
    unittest.main()
