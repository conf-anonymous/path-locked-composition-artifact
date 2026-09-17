"""Retain upstream bytes and build x86_64 LIBVISO2 for Rosetta without a port."""
import argparse
import json
import platform
import subprocess
import time
import zipfile
from pathlib import Path
import acquire as a

FRONT=a.RAW/'libviso2'
SOURCES=('filter.cpp','matcher.cpp','matrix.cpp','triangle.cpp','viso.cpp','viso_stereo.cpp')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('archive',type=Path)
    args=parser.parse_args()
    FRONT.mkdir(parents=True,exist_ok=True)
    receipt=FRONT/'build.json'
    assert not receipt.exists()
    seal=dict(protocol_sha256=a.sha(a.HERE/'FRONTEND_SMOKE_PROTOCOL.md'),
              created_at_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    with (FRONT/'smoke_protocol_seal.json').open('x') as f:
        json.dump(seal,f,indent=2)
    digest=a.sha(args.archive)
    original=FRONT/'libviso2.zip'
    a.preserve(original,args.archive.read_bytes())
    staged={}
    with zipfile.ZipFile(original) as z:
        assert len(set(z.namelist()))==len(z.namelist())
        assert z.testzip() is None
        for member in z.infolist():
            a.safe_member(member)
            if member.is_dir():
                continue
            if member.filename.startswith('src/') or member.filename in ('readme.txt','CMakeLists.txt'):
                path=FRONT/'upstream'/member.filename
                staged[member.filename]=a.preserve(path,z.read(member))
    subprocess.run(['/usr/bin/arch','-x86_64','/usr/bin/true'],check=True)
    upstream=FRONT/'upstream/src'
    binary=FRONT/'stereo_stream_x86_64'
    command=['clang++','-arch','x86_64','-msse3','-std=c++11','-O2',
             '-ffp-contract=off','-I',str(upstream),str(a.HERE/'stereo_stream.cpp')]
    command += [str(upstream/name) for name in SOURCES]+['-o',str(binary)]
    build=subprocess.run(command,capture_output=True,text=True)
    report=dict(archive_sha256=digest,source_path=str(args.archive.resolve()),
        publisher_digest_verified=False,zip_crc_verified=True,upstream_sha256=staged,
        upstream_sources_modified=False,build_command=command,returncode=build.returncode,
        compiler_output=build.stdout+build.stderr,host_architecture=platform.machine(),
        execution_architecture='x86_64 through Rosetta',
        compiler_version=subprocess.check_output(['clang++','--version'],text=True),
        builder_sha256=a.sha(Path(__file__)),wrapper_sha256=a.sha(a.HERE/'stereo_stream.cpp'),
        **seal)
    if build.returncode==0:
        report['binary_sha256']=a.sha(binary)
        report['binary_file_description']=subprocess.check_output(['file',str(binary)],text=True)
    with receipt.open('x') as f:
        json.dump(report,f,indent=2);f.write('\n')
    if build.returncode:
        print(build.stderr)
        raise SystemExit(build.returncode)
    print('Built original LIBVISO2 sources without modification:',binary)


if __name__=='__main__':
    main()
