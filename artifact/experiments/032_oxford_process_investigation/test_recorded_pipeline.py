"""Regression checks on public recorded rows only; no synthetic fixtures."""
import unittest
import sys
import numpy as np
import torch
import investigate as a
import retry

class RecordedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.d,cls.integrity=retry.load_data()

    def test_splits_and_original_integrity(self):
        a.wf.verify_implementation()
        self.assertEqual({s for s,_ in self.d.tensors},{'train','dev'})
        self.assertFalse(a.wf.STARTED.exists())
        self.assertFalse(a.wf.RELEASE.exists())

    def test_real_motor_matrices_match_sdk(self):
        sys.path.insert(0,str(a.ROOT/'iclr-2027-composition/oxford-audit-2026-09-10/reference'))
        from transform import build_se3_transform
        rec=a.selected_records()[0]
        rows=np.loadtxt(a.wf.DATA/rec['vo']['path'],delimiter=',',skiprows=1)[:256,2:8]
        actual=a.matrices(rows)
        reference=np.asarray([np.asarray(build_se3_transform(row)) for row in rows])
        self.assertLess(float(np.max(abs(actual-reference))),1e-12)
        motor=a.to_motor(actual)
        direct=a.ox.motors_from_xyzrpy(rows[:,:3],rows[:,3:])
        q,t=a.base.decode_motor(motor);qd,td=a.base.decode_motor(direct)
        self.assertLess(float((t-td).abs().max()),1e-12)
        self.assertLess(float(torch.minimum((q-qd).norm(dim=-1),(q+qd).norm(dim=-1)).max()),1e-12)

    def test_original_leaf_coefficients_and_fixed_conjugation(self):
        rec=next(r for r in a.selected_records() if r['split']=='dev')
        original=a.ox.OxfordTraversal(rec['traversal'],a.wf.DATA/rec['vo']['path'],
            a.wf.DATA/rec['rtk']['path'],a.ox.load_extrinsic(a.wf.DATA/'extrinsics/ins.txt'),a.ox.Config())
        length=8;windows=[];targets=[]
        for v,r in original.runs:
            leaf=a.ox.relative_from_absolute(v[:-1],v[1:])
            count=len(leaf)-length+1
            windows.append(leaf.unfold(0,length,1).permute(0,2,1).contiguous())
            targets.append(a.ox.relative_from_absolute(r[:count],r[length:length+count]))
        ix=torch.tensor([n==rec['traversal'] for n in self.d.examples[('dev',length)]])
        self.assertTrue(torch.equal(torch.cat(windows),self.d.tensors[('dev',length)][0][ix]))
        e=a.ox.load_extrinsic(a.wf.DATA/'extrinsics/ins.txt')
        quarter=a.ox.motors_from_xyzrpy(np.zeros((1,3)),np.array([[0,0,-np.pi/2]]))[0]
        change=a.base.motor_product(a.base.motor_product(e,quarter),a.ox.inverse_motor(e))
        expected=a.base.motor_product(a.base.motor_product(a.ox.inverse_motor(change),torch.cat(targets)),change)
        actual=self.d.tensors[('dev',length)][1][ix]
        q,t=a.base.decode_motor(actual);qe,te=a.base.decode_motor(expected)
        self.assertLess(float((t-te).norm(dim=-1).max()),1e-7)
        self.assertLess(float(torch.minimum((q-qe).norm(dim=-1),(q+qe).norm(dim=-1)).max()),1e-7)

    def test_confirmation_record_rejected_before_read(self):
        record=next(r for r in a.wf.read(a.wf.MANIFEST)['records'] if r['split']=='confirm')
        with self.assertRaises(AssertionError):a.load_record(record)

    def test_quality_screen_flags_recorded_extreme_target(self):
        path=a.HERE/'quality_screened/masks.pt'
        masks=torch.load(path,weights_only=False)
        for key,mask in masks.items():
            self.assertEqual(len(mask),len(self.d.examples[key]))
            self.assertTrue(bool(mask.any()))
            self.assertTrue(bool((~mask | self.d.connected[key]).all()))
        _,t=a.base.decode_motor(self.d.tensors[('train',8)][1])
        index=int(t.norm(dim=-1).argmax())
        self.assertEqual(self.d.examples[('train',8)][index],'2015-02-06-13-57-16')
        self.assertGreater(float(t[index].norm()),7000)
        self.assertFalse(bool(masks[('train',8)][index]))

if __name__=='__main__':unittest.main()
