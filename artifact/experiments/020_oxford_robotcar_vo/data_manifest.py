"""Immutable Oxford RobotCar eligibility and split rules."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

TRAVERSAL_RE = re.compile(r"20\d{2}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}")


def traversal_id(path: Path) -> str | None:
    """Return the first official RobotCar traversal timestamp in a path."""
    match = TRAVERSAL_RE.search(str(path))
    return match.group(0) if match else None


def split_for(name: str) -> str:
    digest = hashlib.sha256(f"oxford-robotcar-vo-rtk-v1:{name}".encode()).hexdigest()
    bucket = int(digest[:8], 16) % 10
    return "train" if bucket < 6 else ("dev" if bucket < 8 else "confirm")
