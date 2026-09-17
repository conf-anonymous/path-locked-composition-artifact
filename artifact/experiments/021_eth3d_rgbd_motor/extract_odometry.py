"""Freeze Open3D RGB-D odometry leaves from official real ETH3D recordings."""

from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import open3d as o3d

from data_manifest import SEQUENCES, split_for

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "raw" / "eth3d_rgbd"
DOWNLOADS = DATA / "downloads.json"
DERIVED = DATA / "derived_odometry"
MANIFEST = DATA / "manifest.json"

SAMPLE_PERIOD_SECONDS = 0.1
MAXIMUM_FRAME_GAP_SECONDS = 0.2
MAXIMUM_REFERENCE_SEPARATION_SECONDS = 0.02


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_index(path: Path) -> list[tuple[float, Path]]:
    rows = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 2:
            raise RuntimeError(f"malformed index row: {path}:{line_number}")
        rows.append((float(fields[0]), path.parent / fields[1]))
    if len(rows) < 2 or any(rows[i][0] >= rows[i + 1][0] for i in range(len(rows) - 1)):
        raise RuntimeError(f"invalid timestamp order: {path}")
    return rows


def synchronized_frames(root: Path) -> list[tuple[float, Path, Path]]:
    rgb = parse_index(root / "rgb.txt")
    depth = parse_index(root / "depth.txt")
    depth_by_name = {path.name: (timestamp, path) for timestamp, path in depth}
    frames = []
    for timestamp, rgb_path in rgb:
        match = depth_by_name.get(rgb_path.name)
        if match is None or abs(match[0] - timestamp) > 1e-9:
            continue
        if not rgb_path.is_file() or not match[1].is_file():
            raise FileNotFoundError(f"missing recorded RGB-D frame: {rgb_path} / {match[1]}")
        frames.append((timestamp, rgb_path, match[1]))
    if len(frames) < 2:
        raise RuntimeError(f"no synchronized RGB-D frame set under {root}")
    return frames


def parse_groundtruth(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 8:
            raise RuntimeError(f"malformed ground truth: {path}:{line_number}")
        rows.append([float(value) for value in fields])
    values = np.asarray(rows, dtype=np.float64)
    if len(values) < 2 or np.any(np.diff(values[:, 0]) <= 0):
        raise RuntimeError(f"invalid ground-truth timestamps: {path}")
    return values[:, 0], values[:, 1:]


def select_grid(frames: list[tuple[float, Path, Path]]) -> list[tuple[float, Path, Path]]:
    timestamps = np.asarray([frame[0] for frame in frames])
    count = math.floor((timestamps[-1] - timestamps[0]) / SAMPLE_PERIOD_SECONDS) + 1
    grid = timestamps[0] + np.arange(count) * SAMPLE_PERIOD_SECONDS
    indices = np.searchsorted(timestamps, grid, side="left")
    indices = np.unique(indices[indices < len(frames)])
    return [frames[int(index)] for index in indices]


def nearest_reference_indices(frame_times: np.ndarray, reference_times: np.ndarray):
    upper = np.searchsorted(reference_times, frame_times, side="left")
    upper = np.clip(upper, 0, len(reference_times) - 1)
    lower = np.clip(upper - 1, 0, len(reference_times) - 1)
    choose_lower = np.abs(reference_times[lower] - frame_times) <= np.abs(
        reference_times[upper] - frame_times
    )
    indices = np.where(choose_lower, lower, upper)
    separation = np.abs(reference_times[indices] - frame_times)
    return indices, separation <= MAXIMUM_REFERENCE_SEPARATION_SECONDS


def quaternion_matrix_pose(values: np.ndarray) -> np.ndarray:
    translation = values[:3]
    x, y, z, w = values[3:]
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = np.asarray([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
    matrix[:3, 3] = translation
    return matrix


def load_intrinsic(path: Path, width: int, height: int):
    fields = [float(value) for value in path.read_text().split()]
    if len(fields) != 4:
        raise RuntimeError(f"expected fx fy cx cy: {path}")
    return o3d.camera.PinholeCameraIntrinsic(width, height, *fields)


def make_rgbd(color_path: Path, depth_path: Path):
    color = o3d.io.read_image(str(color_path))
    depth = o3d.io.read_image(str(depth_path))
    return o3d.geometry.RGBDImage.create_from_color_and_depth(
        color, depth, depth_scale=5000.0, depth_trunc=4.0,
        convert_rgb_to_intensity=True,
    )


def is_rigid(matrix: np.ndarray) -> bool:
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        return False
    rotation = matrix[:3, :3]
    return bool(
        np.max(np.abs(matrix[3] - np.asarray([0.0, 0.0, 0.0, 1.0]))) <= 1e-5
        and np.max(np.abs(rotation.T @ rotation - np.eye(3))) <= 1e-5
        and abs(np.linalg.det(rotation) - 1.0) <= 1e-5
    )


def process_sequence(name: str, root: Path, destination: Path) -> dict:
    started = time.perf_counter()
    selected = select_grid(synchronized_frames(root))
    frame_times = np.asarray([frame[0] for frame in selected], dtype=np.float64)
    reference_times, reference_values = parse_groundtruth(root / "groundtruth.txt")
    reference_indices, anchors_valid = nearest_reference_indices(frame_times, reference_times)
    reference_poses = np.stack([
        quaternion_matrix_pose(reference_values[index]) for index in reference_indices
    ])

    first_color = np.asarray(o3d.io.read_image(str(selected[0][1])))
    height, width = first_color.shape[:2]
    intrinsic = load_intrinsic(root / "calibration.txt", width, height)
    option = o3d.pipelines.odometry.OdometryOption()
    option.iteration_number_per_pyramid_level = o3d.utility.IntVector([20, 10, 5])
    option.depth_diff_max = 0.03
    option.depth_min = 0.0
    option.depth_max = 4.0
    jacobian = o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm()

    pair_motions = np.repeat(np.eye(4, dtype=np.float64)[None, :, :], len(selected) - 1, axis=0)
    pair_valid = np.zeros(len(selected) - 1, dtype=bool)
    previous = make_rgbd(selected[0][1], selected[0][2])
    front_end_failures = 0
    oversized_frame_gaps = 0
    for index in range(len(selected) - 1):
        current = make_rgbd(selected[index + 1][1], selected[index + 1][2])
        gap = frame_times[index + 1] - frame_times[index]
        if gap <= MAXIMUM_FRAME_GAP_SECONDS:
            success, target_from_source, _ = o3d.pipelines.odometry.compute_rgbd_odometry(
                previous, current, intrinsic, np.eye(4), jacobian, option
            )
            if success and is_rigid(target_from_source):
                # Open3D aligns source points into target coordinates. Camera
                # motion is the inverse: source-camera from target-camera.
                pair_motions[index] = np.linalg.inv(target_from_source)
                pair_valid[index] = True
            else:
                front_end_failures += 1
        else:
            oversized_frame_gaps += 1
        previous = current
        if (index + 1) % 100 == 0:
            print(f"  {name}: {index + 1}/{len(selected) - 1} pairs", flush=True)

    pair_valid &= anchors_valid[:-1] & anchors_valid[1:]
    np.savez_compressed(
        destination,
        timestamps=frame_times,
        reference_poses=reference_poses,
        anchors_valid=anchors_valid,
        pair_motions=pair_motions,
        pair_valid=pair_valid,
    )
    return {
        "sequence": name,
        "split": split_for(name),
        "selected_frames": len(selected),
        "valid_reference_anchors": int(anchors_valid.sum()),
        "valid_pairs": int(pair_valid.sum()),
        "front_end_failures": front_end_failures,
        "oversized_frame_gaps": oversized_frame_gaps,
        "front_end_seconds": time.perf_counter() - started,
        "derived_path": str(destination.relative_to(DATA)),
        "derived_bytes": destination.stat().st_size,
        "derived_sha256": sha256(destination),
    }


def main() -> None:
    if o3d.__version__ != "0.19.0":
        raise RuntimeError(f"frozen front end requires Open3D 0.19.0, got {o3d.__version__}")
    downloads = json.loads(DOWNLOADS.read_text())
    if [record["sequence"] for record in downloads] != list(SEQUENCES):
        raise RuntimeError("download manifest does not match frozen eligible sequence list")
    DERIVED.mkdir(parents=True, exist_ok=True)
    records = []
    for index, record in enumerate(downloads, start=1):
        for archive_record in record["archives"].values():
            archive = DATA / archive_record["path"]
            if sha256(archive) != archive_record["sha256"]:
                raise RuntimeError(f"archive changed after acquisition: {archive}")
        name = record["sequence"]
        print(f"[{index:02d}/{len(SEQUENCES)}] {name}", flush=True)
        destination = DERIVED / f"{name}.npz"
        records.append(process_sequence(name, DATA / record["root"], destination))

    split_counts = {
        split: sum(record["split"] == split for record in records)
        for split in ("train", "dev", "confirm")
    }
    manifest = {
        "protocol": "ETH3D RGB-D motor v1, frozen 2026-09-04",
        "source": "https://www.eth3d.net/slam_datasets",
        "eligible_count": len(records),
        "split_counts": split_counts,
        "open3d_version": o3d.__version__,
        "front_end": {
            "method": "RGBDOdometryJacobianFromHybridTerm",
            "sample_period_seconds": SAMPLE_PERIOD_SECONDS,
            "maximum_frame_gap_seconds": MAXIMUM_FRAME_GAP_SECONDS,
            "maximum_reference_separation_seconds": MAXIMUM_REFERENCE_SEPARATION_SECONDS,
            "depth_scale": 5000.0,
            "depth_truncation_m": 4.0,
            "iterations": [20, 10, 5],
            "depth_difference_max_m": 0.03,
            "initialization": "identity",
        },
        "downloads_sha256": sha256(DOWNLOADS),
        "records": records,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({
        "manifest": str(MANIFEST),
        "eligible_count": len(records),
        "split_counts": split_counts,
        "valid_pairs": sum(record["valid_pairs"] for record in records),
        "front_end_failures": sum(record["front_end_failures"] for record in records),
        "oversized_frame_gaps": sum(record["oversized_frame_gaps"] for record in records),
    }, indent=2))


if __name__ == "__main__":
    main()
