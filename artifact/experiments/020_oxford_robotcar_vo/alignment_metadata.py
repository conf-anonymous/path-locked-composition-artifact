"""Timestamp-only usability check implementing the unchanged Oxford anchor rule."""
import csv

import numpy as np


def recorded_timestamps(path):
    with path.open(newline="") as stream:
        reader = csv.reader(stream)
        next(reader)
        return np.asarray([int(row[0]) for row in reader if row], dtype=np.int64)


def inspect_alignment(vo_path, rtk_path):
    vo, rtk = recorded_timestamps(vo_path), recorded_timestamps(rtk_path)
    first, last = max(vo[0], rtk[0]), min(vo[-1], rtk[-1])
    grid = np.arange(first, last + 1, 500000, dtype=np.int64)
    rtk_indices = np.searchsorted(rtk, grid, side="left")
    best = current = 0
    for index in rtk_indices:
        valid = False
        if index < len(rtk):
            vi = int(np.searchsorted(vo, rtk[index], side="left"))
            valid = vi < len(vo) and 0 <= int(vo[vi] - rtk[index]) <= 100000
        current = current + 1 if valid else 0
        best = max(best, current)
    return {"max_contiguous_anchors": best, "supports_l32": best > 32,
            "vo_timestamp_range": [int(vo[0]), int(vo[-1])],
            "rtk_timestamp_range": [int(rtk[0]), int(rtk[-1])]}
