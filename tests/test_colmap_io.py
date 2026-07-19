from pathlib import Path

import numpy as np

from src.common.colmap_io import compute_scene_bbox, load_sparse_scene

CHAIR_SPARSE = Path("VAI_NVS_DATA_ROUND2/chair/train/sparse/0")


def test_load_sparse_scene_reads_chair_scene():
    scene = load_sparse_scene(CHAIR_SPARSE)
    # NOT 205 (the file count in train/images/): COLMAP's images.bin
    # registers more cameras than are distributed as files — the dataset
    # runs SfM over train+test (and, for the HCM scenes, extra calibration
    # frames) combined for pose accuracy, then withholds some images'
    # pixels. Verified directly against the real chair data: 263
    # registered, 205 with files on disk, and the 58 without a file are
    # exactly the 58 names in test/test_poses.csv (see
    # test_load_sparse_scene_registers_more_images_than_are_distributed_as_files).
    # Do not "fix" this number down to 205 — that would be re-introducing
    # the bug this test exists to catch.
    assert len(scene.images) == 263
    assert len(scene.cameras) >= 1
    assert len(scene.points3d) > 0


def test_load_sparse_scene_registers_more_images_than_are_distributed_as_files():
    import csv
    import os

    scene = load_sparse_scene(CHAIR_SPARSE)
    registered_names = {img.name for img in scene.images.values()}
    folder_names = set(os.listdir("VAI_NVS_DATA_ROUND2/chair/train/images"))
    registered_without_file = registered_names - folder_names
    assert len(registered_without_file) == 58

    # Don't just trust the magic number 58 — cross-check it's actually the
    # scene's real test_poses.csv image names, not a coincidence.
    with open("VAI_NVS_DATA_ROUND2/chair/test/test_poses.csv", newline="") as f:
        test_pose_names = {row["image_name"] for row in csv.DictReader(f)}
    assert registered_without_file == test_pose_names


def test_load_sparse_scene_preserves_real_point3d_ids_not_file_order_index():
    # Regression test: COLMAP point3D_ids are NOT contiguous (point
    # culling/merging during reconstruction leaves gaps), so re-keying
    # points3d by file-storage order (0..N-1) instead of the real id
    # silently returns the WRONG point for any id that coincidentally
    # collides with a valid 0..N-1 index, corrupting anything that looks
    # points up via images[...].point3D_ids (e.g. depth-target lookups).
    scene = load_sparse_scene(CHAIR_SPARSE)

    referenced_ids = {
        int(pid)
        for img in scene.images.values()
        for pid in img.point3D_ids
        if pid != -1
    }
    assert referenced_ids, "fixture sanity: chair images must reference some points"
    # every id an image references must resolve to a real point
    assert referenced_ids <= set(scene.points3d.keys())
    # ids are not just 0..N-1 (proves this isn't accidentally still using
    # file-order indexing) — real COLMAP ids have gaps and go well beyond
    # the point count.
    assert max(referenced_ids) >= len(scene.points3d)
    # each point's own .id field must match the dict key it's stored under
    for point_id, point in scene.points3d.items():
        assert point.id == point_id


def test_compute_scene_bbox_returns_min_less_than_max_with_margin():
    scene = load_sparse_scene(CHAIR_SPARSE)
    min_xyz, max_xyz = compute_scene_bbox(scene.points3d, margin_ratio=0.1)
    assert min_xyz.shape == (3,)
    assert max_xyz.shape == (3,)
    assert np.all(min_xyz < max_xyz)
    assert np.all(np.isfinite(min_xyz))
    assert np.all(np.isfinite(max_xyz))


def test_compute_scene_bbox_margin_expands_tight_bbox():
    raw_points = {
        i: type("P", (), {"xyz": np.array([float(i), 0.0, 0.0])})()
        for i in range(3)
    }  # points at x=0,1,2
    tight_min, tight_max = compute_scene_bbox(raw_points, margin_ratio=0.0)
    wide_min, wide_max = compute_scene_bbox(raw_points, margin_ratio=0.5)
    assert wide_min[0] < tight_min[0]
    assert wide_max[0] > tight_max[0]
