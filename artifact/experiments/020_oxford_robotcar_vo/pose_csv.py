"""Read official recorded VO/RTK rows; UTM zones are text, not observations."""
from __future__ import annotations

import csv
import math
import re
from pathlib import Path

VO_HEADER = "source_timestamp,destination_timestamp,x,y,z,roll,pitch,yaw".split(",")
RTK_HEADER = ("timestamp,latitude,longitude,altitude,northing,easting,down,"
              "utm_zone,velocity_north,velocity_east,velocity_down,roll,pitch,yaw").split(",")


def validate_header(header, kind, path):
    expected = VO_HEADER if kind == "vo" else RTK_HEADER if kind == "rtk" else None
    if expected is None or header != expected:
        raise RuntimeError(f"unexpected official {kind} header: {path}")


def parse_row(row, kind, path, line_number):
    expected = 8 if kind == "vo" else 14
    if len(row) != expected:
        raise RuntimeError(f"unexpected {kind} column count: {path}:{line_number}")
    result = []
    try:
        for index, value in enumerate(row):
            if kind == "rtk" and index == 7:
                if re.fullmatch(r"(?:[1-9]|[1-5][0-9]|60)[C-HJ-NP-X]", value) is None:
                    raise ValueError("invalid UTM zone metadata")
                result.append(value)
            elif index == 0:
                timestamp = int(value)
                if timestamp <= 0:
                    raise ValueError("nonpositive timestamp")
                result.append(timestamp)
            else:
                number = float(value)
                if not math.isfinite(number):
                    raise ValueError("nonfinite numeric field")
                result.append(number)
    except ValueError as error:
        raise RuntimeError(f"invalid {kind} field: {path}:{line_number}: {error}") from error
    return result


def recorded_rows(path: Path, kind: str):
    with path.open(newline="") as stream:
        reader = csv.reader(stream)
        validate_header(next(reader, None), kind, path)
        previous = None
        for line_number, row in enumerate(reader, 2):
            if not row:
                continue
            values = parse_row(row, kind, path, line_number)
            if previous is not None and values[0] <= previous:
                raise RuntimeError(f"non-increasing {kind} timestamp: {path}:{line_number}")
            previous = values[0]
            yield values


def read_pose_csv(path: Path):
    import numpy as np

    with path.open(newline="") as stream:
        header = next(csv.reader(stream), None)
    kind = "rtk" if header == RTK_HEADER else "vo"
    validate_header(header, kind, path)
    rows = list(recorded_rows(path, kind))
    if len(rows) < 2:
        raise RuntimeError(f"insufficient {kind} rows: {path}")
    # Object dtype retains textual UTM metadata and exact integer timestamps.
    # Numerical pose slices are converted explicitly by the consuming code.
    return header, np.asarray(rows, dtype=object if kind == "rtk" else np.float64)
