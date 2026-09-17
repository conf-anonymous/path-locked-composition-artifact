"""Native devkit versus NumPy on recorded training poses; no model evaluation."""
import json
import subprocess
from pathlib import Path
import numpy as np
import acquire as a


def main():
    output=a.RAW/'native_reference_checks.json'
    assert not output.exists()
    acquisition=json.loads((a.RAW/'acquisition.json').read_text())
    for name,digest in acquisition['staged_sha256'].items():
        assert a.sha(a.RAW/name)==digest
    source=a.RAW/'devkit/cpp'
    binary=a.RAW/'native_probe'
    command=['clang++','-O2','-std=c++11','-I',str(source),
             str(a.HERE/'native_probe.cpp'),str(source/'matrix.cpp'),'-o',str(binary)]
    build=subprocess.run(command,capture_output=True,text=True,check=True)
    reports=[]
    for seq,role in a.ROLES.items():
        if role!='train':
            continue
        pose_path=a.RAW/'dataset/poses'/f'{seq}.txt'
        values=np.loadtxt(pose_path).reshape(-1,3,4)
        poses=np.tile(np.eye(4),(len(values),1,1)); poses[:,:3]=values
        run=subprocess.run([str(binary),str(pose_path)],capture_output=True,text=True,check=True)
        distances=[]; segments=[]; metrics=[]
        for line in run.stdout.splitlines():
            parts=line.split()
            if parts[0]=='D':
                assert int(parts[1])==len(distances)
                distances.append(float(parts[2]))
            elif parts[0]=='S':
                segments.append((int(parts[1]),int(parts[2]),float(parts[3]),
                                 np.array(list(map(float,parts[4:]))).reshape(3,4)))
            elif parts[0]=='E':
                metrics.append(list(map(float,parts[1:])))
            else:
                raise ValueError(line)
        distances=np.array(distances,dtype=np.float32)
        delta=np.diff(poses[:,:3,3],axis=0).astype(np.float32)
        # Match the devkit's left-to-right float expression and cumulative sum.
        steps=np.sqrt((delta[:,0]*delta[:,0]+delta[:,1]*delta[:,1])+delta[:,2]*delta[:,2])
        own_dist=np.r_[np.float32(0),np.cumsum(steps,dtype=np.float32)]
        assert len(distances)==len(poses)
        max_distance_diff=float(np.max(abs(distances-own_dist)))
        assert max_distance_diff<.01
        expected=[]
        for first in range(0,len(poses),10):
            for length in range(100,801,100):
                threshold=np.float32(own_dist[first]+np.float32(length))
                last=int(np.searchsorted(own_dist,threshold,side='right'))
                if last<len(poses):
                    expected.append((first,last,float(length)))
        assert expected==[(f,l,d) for f,l,d,_ in segments]
        assert len(segments)==len(metrics)
        max_relative_diff=0.
        for (first,last,length,matrix),metric in zip(segments,metrics):
            expected_matrix=np.linalg.solve(poses[first],poses[last])[:3]
            diff=float(np.max(abs(matrix-expected_matrix)))
            max_relative_diff=max(max_relative_diff,diff)
            assert metric[:2]==[first,length]
        assert max_relative_diff<1e-7
        metrics=np.array(metrics)
        assert len(metrics)>0 and np.isfinite(metrics).all()
        max_t=float(np.max(metrics[:,2])); max_r=float(np.max(metrics[:,3]))
        assert max_t<1e-9 and max_r<1e-6
        reports.append(dict(sequence=seq,role=role,frames=len(poses),
            eligible_segments=len(segments),segments_by_length={str(n):sum(d==n for _,_,d,_ in segments) for n in range(100,801,100)},
            max_distance_difference_m=max_distance_diff,
            max_relative_matrix_coefficient_difference=max_relative_diff,
            self_comparison_max_translation_fraction=max_t,
            self_comparison_max_rotation_rad_per_m=max_r))
        print(seq,len(segments),'segments; matrix difference',max_relative_diff,flush=True)
    report=dict(training_only=True,model_evaluation=False,physical_accuracy_certified=False,
        acquisition_sha256=a.sha(a.RAW/'acquisition.json'),
        source_sha256=a.sha(Path(__file__)),probe_sha256=a.sha(a.HERE/'native_probe.cpp'),
        binary_sha256=a.sha(binary),build_command=command,
        compiler_version=subprocess.check_output(['clang++','--version'],text=True),
        compiler_output=build.stdout+build.stderr,sequences=reports)
    with output.open('x') as f:
        json.dump(report,f,indent=2,allow_nan=False); f.write('\n')


if __name__=='__main__':
    main()
