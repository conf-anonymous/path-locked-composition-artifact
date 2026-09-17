"""Immutable eligibility, download, and split rules for ETH3D Experiment 021."""

from __future__ import annotations

import hashlib

SEQUENCES = (
    "cables_1", "cables_2", "cables_3",
    "camera_shake_1", "camera_shake_2", "camera_shake_3",
    "ceiling_1", "ceiling_2", "desk_3", "desk_changing_1",
    "einstein_1", "einstein_2", "einstein_dark", "einstein_flashlight",
    "einstein_global_light_changes_1", "einstein_global_light_changes_2",
    "einstein_global_light_changes_3", "kidnap_1", "kidnap_dark",
    "large_loop_1", "mannequin_1", "mannequin_3", "mannequin_4",
    "mannequin_5", "mannequin_7", "mannequin_face_1", "mannequin_face_2",
    "mannequin_face_3", "mannequin_head", "motion_1", "planar_2", "planar_3",
    "plant_1", "plant_2", "plant_3", "plant_4", "plant_5", "plant_dark",
    "plant_scene_1", "plant_scene_2", "plant_scene_3", "reflective_1",
    "repetitive", "sofa_1", "sofa_2", "sofa_3", "sofa_4", "sofa_dark_1",
    "sofa_dark_2", "sofa_dark_3", "sofa_shake", "table_3", "table_4",
    "table_7", "vicon_light_1", "vicon_light_2",
)

INELIGIBLE_SFM = (
    "sfm_bench", "sfm_garden", "sfm_house_loop", "sfm_lab_room_1",
    "sfm_lab_room_2",
)


MODALITIES = ("mono", "rgbd")


def source_url(name: str, modality: str) -> str:
    if name not in SEQUENCES:
        raise ValueError(f"ineligible ETH3D sequence: {name}")
    if modality not in MODALITIES:
        raise ValueError(f"ineligible ETH3D modality: {modality}")
    return f"https://www.eth3d.net/data/slam/datasets/{name}_{modality}.zip"


def split_for(name: str) -> str:
    if name not in SEQUENCES:
        raise ValueError(f"ineligible ETH3D sequence: {name}")
    digest = hashlib.sha256(f"eth3d-rgbd-v1:{name}".encode()).hexdigest()
    bucket = int(digest[:8], 16) % 10
    return "train" if bucket < 6 else ("dev" if bucket < 8 else "confirm")
