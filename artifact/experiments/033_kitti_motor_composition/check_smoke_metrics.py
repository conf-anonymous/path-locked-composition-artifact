"""Actual frontend estimates versus original devkit, smoke-only training data."""
import json
import subprocess
from pathlib import Path
import numpy as np
import acquire as a
from build_frontend import FRONT
import frontend_smoke as smoke


def main():
    destination=FRONT/'native_smoke_metrics.json'
    assert not destination.exists()
    smoke.check_inputs()
    record=json.loads((FRONT/'smoke_run1.json').read_text())
    # This bounded native check is defined for the observed continuous smoke.
    # Refuse to insert poses if later recordings have missing estimates.
    assert not record['failed_edges'] and record['frames']==401
    source=a.RAW/'devkit/cpp'
    binary=FRONT/'metric_stream'
    command=['clang++','-O2','-std=c++11','-I',str(source),
             str(a.HERE/'metric_stream.cpp'),str(source/'matrix.cpp'),'-o',str(binary)]
    build=subprocess.run(command,capture_output=True,text=True,check=True)
    values=np.loadtxt(a.RAW/'dataset/poses/00.txt').reshape(-1,3,4)[:401]
    predicted=np.tile(np.eye(4),(401,1,1))
    lines=['401']
    for i,row in enumerate(record['rows']):
        if i:
            motion=np.eye(4);motion[:3]=np.array(row['previous_to_current']).reshape(3,4)
            predicted[i]=predicted[i-1]@np.linalg.inv(motion)
        lines.append(' '.join(format(float(v),'.17g') for v in np.r_[values[i].ravel(),predicted[i,:3].ravel()]))
    run=subprocess.run([str(binary)],input='\n'.join(lines)+'\n',capture_output=True,text=True,check=True)
    native=[list(map(float,line.split())) for line in run.stdout.splitlines()]
    python=record['diagnostics']['segments']
    assert len(native)==len(python)>0
    errors=[]
    for cpp,py in zip(native,python):
        first,length,t,r=cpp
        assert first==py['first'] and length==py['length_m']
        errors.append([abs(t*100-py['translation_percent']),
                       abs(np.degrees(r)-py['rotation_deg_per_m'])])
    errors=np.array(errors)
    assert errors[:,0].max()<1e-5 and errors[:,1].max()<1e-5
    report=dict(sequence='00',training_smoke_only=True,segments=len(native),
        max_translation_difference_percentage_points=float(errors[:,0].max()),
        max_rotation_difference_deg_per_m=float(errors[:,1].max()),
        native_translation_percent_mean=float(np.mean([r[2]*100 for r in native])),
        native_rotation_deg_per_m_mean=float(np.mean([np.degrees(r[3]) for r in native])),
        smoke_run_sha256=a.sha(FRONT/'smoke_run1.json'),
        source_sha256=a.sha(Path(__file__)),wrapper_sha256=a.sha(a.HERE/'metric_stream.cpp'),
        binary_sha256=a.sha(binary),build_command=command,compiler_output=build.stdout+build.stderr,
        devkit_sources_sha256={p:a.sha(source/p) for p in ('evaluate_odometry.cpp','matrix.cpp','matrix.h')})
    smoke.save(destination,report)
    print(json.dumps({k:v for k,v in report.items() if k in ('segments',
        'max_translation_difference_percentage_points','max_rotation_difference_deg_per_m',
        'native_translation_percent_mean','native_rotation_deg_per_m_mean')},indent=2))


if __name__=='__main__':
    main()
