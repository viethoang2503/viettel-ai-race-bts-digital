from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from src.common.colmap_io import read_intrinsics_binary
from src.training.colmap_writer import write_cameras_binary


@dataclass
class _Camera:
    model: str
    width: int
    height: int
    params: np.ndarray


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
