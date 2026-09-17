"""Official public TUM RGB-D ground-truth trajectory manifest."""

from __future__ import annotations

BASE = "https://cvg.cit.tum.de/rgbd/dataset/{group}/rgbd_dataset_{name}-groundtruth.txt"

# Public ground-truth sequences in the official download catalog. Validation
# sequences without public ground truth are deliberately absent.
SEQUENCES = (
    "freiburg1_xyz", "freiburg1_rpy", "freiburg1_360",
    "freiburg1_floor", "freiburg1_desk", "freiburg1_desk2",
    "freiburg1_room", "freiburg1_plant", "freiburg1_teddy",
    "freiburg2_xyz", "freiburg2_rpy", "freiburg2_360_hemisphere",
    "freiburg2_360_kidnap", "freiburg2_desk", "freiburg2_large_no_loop",
    "freiburg2_large_with_loop", "freiburg2_pioneer_360",
    "freiburg2_pioneer_slam", "freiburg2_pioneer_slam2",
    "freiburg2_pioneer_slam3", "freiburg2_desk_with_person",
    "freiburg2_coke", "freiburg2_dishes", "freiburg2_flowerbouquet",
    "freiburg2_flowerbouquet_brownbackground", "freiburg2_metallic_sphere",
    "freiburg2_metallic_sphere2", "freiburg3_long_office_household",
    "freiburg3_nostructure_notexture_far",
    "freiburg3_nostructure_notexture_near_withloop",
    "freiburg3_nostructure_texture_far",
    "freiburg3_nostructure_texture_near_withloop",
    "freiburg3_structure_notexture_far", "freiburg3_structure_notexture_near",
    "freiburg3_structure_texture_far", "freiburg3_structure_texture_near",
    "freiburg3_sitting_static", "freiburg3_sitting_xyz",
    "freiburg3_sitting_halfsphere", "freiburg3_sitting_rpy",
    "freiburg3_walking_static", "freiburg3_walking_xyz",
    "freiburg3_walking_halfsphere", "freiburg3_walking_rpy",
    "freiburg3_cabinet", "freiburg3_large_cabinet", "freiburg3_teddy",
)


def source_url(name: str) -> str:
    group = name.split("_", 1)[0]
    return BASE.format(group=group, name=name)

