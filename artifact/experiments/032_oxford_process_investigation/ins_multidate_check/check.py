"""Fixed source-only cross-export checks; never reads VO or confirmation rows."""
from __future__ import annotations
import argparse
import collections
import csv
import hashlib
import shutil
import sys
import tarfile
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'ins_frame_check'))
import check as previous
a = previous.a
SOURCES = (
    ('2014-11-14-16-34-33', 'train', 'e09887f22a26dbed63b1af994162834f'),
    ('2015-08-13-16-02-58', 'train', '1ee294931920c557bea17614280081b2'),
    ('2015-08-14-14-54-57', 'dev', 'fd18b53ea3275405fbc51de3eb232e90'),
    ('2014-11-21-16-07-03', 'dev', 'd7455bfe51b17a8b42b0e8b0f40ae228'),
)
FIELDS = ('timestamp', 'northing', 'easting', 'down', 'velocity_north',
          'velocity_east', 'velocity_down', 'roll', 'pitch', 'yaw')
TOLERANCE_US = 20000


def record_for(name, split):
    assert split in ('train', 'dev'), 'confirmation is forbidden'
    record = next(r for r in a.selected_records() if r['traversal'] == name)
    assert record['split'] == split
    return record


def stage(downloads, name, expected_md5):
    raw = a.wf.DATA / 'ins_crosscheck' / name
    source = downloads / (name + '_gps.tar')
    assert hashlib.md5(source.read_bytes()).hexdigest() == expected_md5
    raw.mkdir(parents=True, exist_ok=True)
    archive = raw / source.name
    if not archive.exists():
        shutil.copyfile(source, archive)
    assert a.wf.sha(archive) == a.wf.sha(source)
    with tarfile.open(archive) as tar:
        expected = {f'{name}/gps/{n}' for n in ('ins.csv', 'gps.csv')}
        assert {m.name for m in tar.getmembers() if m.isfile()} == expected
        for member_name in sorted(expected):
            member = tar.getmember(member_name)
            assert member.isfile()
            blob = tar.extractfile(member).read()
            target = raw / Path(member_name).name
            if not target.exists():
                with target.open('xb') as f:
                    f.write(blob)
            assert target.read_bytes() == blob
    return raw, archive


def read_csv(path):
    with path.open(newline='') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        header = reader.fieldnames
    times = np.array([int(r['timestamp']) for r in rows], dtype=np.int64)
    values = np.array([[float(r[k]) for k in FIELDS] for r in rows])
    return times, values, rows, header


def inspect(downloads, source):
    name, split, md5 = source
    record = record_for(name, split)  # guard before observations are opened
    raw, archive = stage(downloads, name, md5)
    rpath = a.wf.DATA / record['rtk']['path']
    assert a.wf.sha(rpath) == record['rtk']['sha256']
    it, ins, rows, header = read_csv(raw / 'ins.csv')
    rt, rtk, _, _ = read_csv(rpath)
    assert np.all(np.diff(rt) > 0)
    assert np.isfinite(rtk).all()
    states = np.array([r['ins_status'] for r in rows])
    overlap = (it >= rt[0] - TOLERANCE_US) & (it <= rt[-1] + TOLERANCE_US)
    disorder = np.flatnonzero(np.diff(it) <= 0)
    nonfinite = ~np.isfinite(ins).all(axis=1)
    report = dict(traversal=name, split=split, archive_md5=md5,
        archive_sha256=a.wf.sha(archive),
        csv_sha256={n:a.wf.sha(raw / n) for n in ('ins.csv', 'gps.csv')},
        rtk_sha256=record['rtk']['sha256'], ins_header=header,
        ins_rows=len(ins), rtk_rows=len(rtk), ins_rows_in_rtk_interval=int(overlap.sum()),
        ins_status_counts=dict(collections.Counter(states.tolist())),
        nonfinite_ins_rows=int(nonfinite.sum()),
        nonfinite_ins_rows_in_pairing_interval=int((nonfinite & overlap).sum()),
        nonincreasing_ins_edges=[dict(first_csv_line=int(j+2),
            first_timestamp=int(it[j]), second_timestamp=int(it[j+1]),
            inside_pairing_interval=bool(overlap[j] or overlap[j+1])) for j in disorder])
    if any(overlap[j] or overlap[j+1] for j in disorder):
        report['blocked'] = 'Nonincreasing INS edge in RTK pairing interval; no sorting/filtering performed.'
        return report
    if (nonfinite & overlap).any():
        report['blocked'] = 'Nonfinite INS values in RTK pairing interval; no filtering performed.'
        return report
    it, ins, states = it[overlap], ins[overlap], states[overlap]
    if len(it) == 0:
        report['blocked'] = 'No recorded INS rows in RTK interval.'
        return report
    assert np.all(np.diff(it) > 0)
    index = previous.nearest(it, rt)
    mask = abs(it[index] - rt) <= TOLERANCE_US
    if not mask.any():
        report['blocked'] = 'No pairs within 20 ms.'
        return report
    good = mask & (states[index] == 'INS_SOLUTION_GOOD')
    midpoint = (int(rt[mask][0]) + int(rt[mask][-1])) // 2
    groups = dict(all_paired=mask, ins_solution_good=good,
        good_first_time_half=good & (rt <= midpoint),
        good_second_time_half=good & (rt > midpoint))
    report.update(paired_rows=int(mask.sum()), unpaired_rtk_rows=int((~mask).sum()),
        unique_paired_ins_rows=int(len(np.unique(index[mask]))),
        paired_status_counts=dict(collections.Counter(states[index[mask]].tolist())),
        ins_minus_rtk_timestamp_ms=a.stats((it[index[mask]] - rt[mask])/1000),
        subsets={k:previous.summarize(ins, rtk, index, m) if m.sum() >= 3
                 else {'n':int(m.sum()), 'insufficient':True} for k,m in groups.items()})
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--downloads', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=HERE/'results.json')
    args = parser.parse_args()
    assert not args.output.exists(), 'Do not overwrite a completed diagnostic.'
    a.torch.set_num_threads(1)
    a.wf.verify_implementation()
    assert not a.wf.STARTED.exists() and not a.wf.RELEASE.exists()
    reports = []
    for source in SOURCES:
        report = inspect(args.downloads, source)
        reports.append(report)
        print(source[0], report.get('blocked', 'checked'), flush=True)
    ranges = {}
    for split in ('train', 'dev'):
        selected = [r['subsets']['ins_solution_good'] for r in reports
                    if r['split'] == split and 'blocked' not in r
                    and not r['subsets']['ins_solution_good'].get('insufficient')]
        ranges[split] = {k:a.stats([s[k]['median'] for s in selected]) for k in
            ('roll_sum_deg', 'pitch_difference_deg', 'yaw_difference_deg')}
    report = dict(diagnostic_only=True, training_performed=False,
        confirmation_accessed=False, vo_observations_accessed=False,
        physical_frame_validated=False, no_offset_fitting=True,
        shares_physical_sensor=True, tolerance_us=TOLERANCE_US,
        traversals=reports, ranges_of_good_status_traversal_medians=ranges,
        original_fingerprint=a.wf.fingerprint(),
        source_sha256=a.wf.sha(Path(__file__)),
        previous_source_sha256=a.wf.sha(Path(previous.__file__)),
        investigate_source_sha256=a.wf.sha(Path(a.__file__)),
        plan_sha256=a.wf.sha(HERE/'PLAN.md'))
    a.wf.verify_implementation()
    assert not a.wf.STARTED.exists() and not a.wf.RELEASE.exists()
    a.save(args.output, report)


if __name__ == '__main__':
    main()
