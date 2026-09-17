"""Verify archive/extraction identity for train/dev only."""
import hashlib
import tarfile
import investigate as a

def main():
    receipt=a.wf.read(a.wf.DATA/'acquisition-verified-2026-09-09.json')
    expected={a.Path(r['file']).name:r for r in receipt['archives']}
    records=[]
    for r in a.selected_records():
        name=r['traversal'];p=a.wf.DATA/'official_archives'/f'{name}_vo.tar'
        entry=expected[p.name];md5=hashlib.md5(p.read_bytes()).hexdigest()
        assert md5.lower()==entry['official_md5'].lower()
        assert a.wf.sha(p)==entry['sha256']
        with tarfile.open(p) as archive:
            matches=[m for m in archive.getmembers() if m.isfile() and m.name.endswith('/vo.csv')]
            assert len(matches)==1
            digest=hashlib.sha256(archive.extractfile(matches[0]).read()).hexdigest()
            assert digest==r['vo']['sha256']==a.wf.sha(a.wf.DATA/r['vo']['path'])
        records.append({'traversal':name,'split':r['split'],'archive_md5':md5,
            'official_page':entry['official_page'],'csv_sha256':digest})
    a.wf.verify_implementation()
    a.save(a.HERE/'archive_recheck.json',{'records':records,'confirmation_observations_read':False,
        'rtk_archive_sha256':a.wf.sha(a.wf.DATA/'official_archives/rtk.zip'),
        'receipt_sha256':a.wf.sha(a.wf.DATA/'acquisition-verified-2026-09-09.json'),
        'source_sha256':a.wf.sha(a.HERE/'archive_recheck.py')})
    print('PASS: all 48 selected VO archives match recorded official MD5 and extracted CSV bytes')

if __name__=='__main__':main()
