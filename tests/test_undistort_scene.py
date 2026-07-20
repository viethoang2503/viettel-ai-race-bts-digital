import struct
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.common.colmap_io import read_intrinsics_binary
from src.common.config import SceneConfig
from src.training.colmap_writer import write_images_binary
from src.training.undistort_scene import undistort_scene


class _FakeImage:
    def __init__(self, name, camera_id):
        self.qvec = np.array([1.0, 0.0, 0.0, 0.0])
        self.tvec = np.array([0.0, 0.0, 0.0])
        self.camera_id = camera_id
        self.name = name


def _write_camera(path, camera_id, model_id, width, height, params):
    with open(path, "wb") as fid:
        fid.write(struct.pack("<Q", 1))
        fid.write(struct.pack("<iiQQ", camera_id, model_id, width, height))
        fid.write(struct.pack("<" + "d" * len(params), *params))


def _write_empty_points3d(path):
    with open(path, "wb") as fid:
        fid.write(struct.pack("<Q", 0))


def _make_scene(tmp_path, model_id, params, width=64, height=48):
    root = tmp_path / "scene"
    images_dir = root / "train" / "images"
    sparse_dir = root / "train" / "sparse" / "0"
    images_dir.mkdir(parents=True)
    sparse_dir.mkdir(parents=True)

    _write_camera(sparse_dir / "cameras.bin", 1, model_id, width, height, params)
    write_images_binary({1: _FakeImage("0001.jpg", 1)}, sparse_dir / "images.bin")
    _write_empty_points3d(sparse_dir / "points3D.bin")

    # A synthetic image with a visible pattern so undistortion has
    # something to actually warp (a flat color would look the same either way).
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:, width // 2:] = 255
    cv2.imwrite(str(images_dir / "0001.jpg"), img)

    return SceneConfig(
        name="scene", root=root, train_images_dir=images_dir,
        sparse_dir=sparse_dir, test_poses_csv=root / "test" / "test_poses.csv",
    )


def test_undistort_scene_converts_simple_radial_to_pinhole(tmp_path):
    # model_id 2 = SIMPLE_RADIAL, params [f, cx, cy, k1]
    scene = _make_scene(tmp_path, model_id=2, params=[80.0, 32.0, 24.0, 0.05])
    output_dir = tmp_path / "undistorted"

    result = undistort_scene(scene, output_dir)

    assert result.sparse_dir == output_dir / "sparse" / "0"
    assert result.train_images_dir == output_dir / "images"
    cameras = read_intrinsics_binary(str(result.sparse_dir / "cameras.bin"))
    camera = cameras[1]
    assert camera.model == "PINHOLE"
    fx, fy, cx, cy = camera.params
    assert fx == pytest.approx(80.0)
    assert fy == pytest.approx(80.0)
    assert cx == pytest.approx(32.0)
    assert cy == pytest.approx(24.0)
    assert (result.train_images_dir / "0001.jpg").exists()


def test_undistort_scene_is_noop_for_already_supported_model(tmp_path):
    # model_id 0 = SIMPLE_PINHOLE, params [f, cx, cy] — chair/bonsai's real model.
    scene = _make_scene(tmp_path, model_id=0, params=[80.0, 32.0, 24.0])
    output_dir = tmp_path / "unused_output"

    result = undistort_scene(scene, output_dir)

    assert result is scene
    assert not output_dir.exists()


def test_undistort_scene_rejects_unsupported_model(tmp_path):
    # model_id 3 = RADIAL (2 distortion coeffs), not handled.
    scene = _make_scene(tmp_path, model_id=3, params=[80.0, 32.0, 24.0, 0.01, 0.0])

    with pytest.raises(ValueError, match="RADIAL"):
        undistort_scene(scene, tmp_path / "unused_output")
