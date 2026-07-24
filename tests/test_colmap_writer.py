from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from src.common.colmap_io import read_extrinsics_binary, read_intrinsics_binary
from src.training.colmap_writer import write_cameras_binary, write_images_binary


@dataclass
class _Camera:
    model: str
    width: int
    height: int
    params: np.ndarray


@dataclass
class _Image:
    id: int
    qvec: np.ndarray
    tvec: np.ndarray
    camera_id: int
    name: str
    xys: np.ndarray
    point3D_ids: np.ndarray


def test_write_images_binary_preserves_point2d_tracks(tmp_path):
    image = _Image(
        id=7,
        qvec=np.array([1.0, 0.0, 0.0, 0.0]),
        tvec=np.array([1.0, 2.0, 3.0]),
        camera_id=2,
        name="frame.jpg",
        xys=np.array([[10.5, 20.25], [30.0, 40.0]]),
        point3D_ids=np.array([11, -1]),
    )
    out_path = tmp_path / "images.bin"

    write_images_binary({7: image}, out_path)

    loaded = read_extrinsics_binary(str(out_path))[7]
    np.testing.assert_allclose(loaded.xys, image.xys)
    np.testing.assert_array_equal(loaded.point3D_ids, image.point3D_ids)


def test_write_images_binary_rejects_mismatched_track_lengths(tmp_path):
    image = _Image(
        id=7,
        qvec=np.array([1.0, 0.0, 0.0, 0.0]),
        tvec=np.zeros(3),
        camera_id=1,
        name="frame.jpg",
        xys=np.array([[10.5, 20.25]]),
        point3D_ids=np.array([11, 12]),
    )

    with pytest.raises(ValueError, match="track length mismatch"):
        write_images_binary({7: image}, tmp_path / "images.bin")


def test_write_cameras_binary_round_trips_with_the_vendored_reader(tmp_path):
    cameras = {
        1: _Camera(model="PINHOLE", width=1320, height=989, params=np.array([926.4, 926.4, 660.0, 494.5])),
        2: _Camera(model="PINHOLE", width=720, height=1280, params=np.array([1113.99, 1113.99, 360.0, 640.0])),
    }
    out_path = tmp_path / "cameras.bin"

    write_cameras_binary(cameras, out_path)

    reloaded = read_intrinsics_binary(str(out_path))
    assert set(reloaded.keys()) == {1, 2}
    for camera_id, original in cameras.items():
        round_tripped = reloaded[camera_id]
        assert round_tripped.model == "PINHOLE"
        assert round_tripped.width == original.width
        assert round_tripped.height == original.height
        np.testing.assert_allclose(round_tripped.params, original.params, atol=1e-9)


def test_write_cameras_binary_rejects_non_pinhole_model(tmp_path):
    cameras = {1: _Camera(model="SIMPLE_RADIAL", width=64, height=48, params=np.array([80.0, 32.0, 24.0, 0.01]))}
    with pytest.raises(ValueError, match="PINHOLE"):
        write_cameras_binary(cameras, tmp_path / "cameras.bin")
