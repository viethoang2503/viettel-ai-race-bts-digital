import math

import numpy as np
import pytest

from src.common.pose_utils import (
    camera_extrinsics_from_colmap,
    camera_focal_lengths,
    camera_params_from_csv_row,
    focal2fov,
    qvec2rotmat,
)


def test_identity_quaternion_gives_identity_rotation():
    r = qvec2rotmat(np.array([1.0, 0.0, 0.0, 0.0]))
    np.testing.assert_allclose(r, np.eye(3), atol=1e-10)


def test_90_degree_z_rotation_quaternion():
    # 90 deg about Z: qw=cos(45deg), qz=sin(45deg)
    half = math.pi / 4
    qvec = np.array([math.cos(half), 0.0, 0.0, math.sin(half)])
    r = qvec2rotmat(qvec)
    expected = np.array([
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    np.testing.assert_allclose(r, expected, atol=1e-10)


def test_focal2fov_matches_known_value():
    # focal=1000, pixels=2000 -> 2*atan(1) = pi/2
    fov = focal2fov(1000.0, 2000)
    assert fov == pytest.approx(math.pi / 2, abs=1e-10)


def test_camera_extrinsics_from_colmap_is_R_transpose_T_unchanged():
    qw, qx, qy, qz = 1.0, 0.0, 0.0, 0.0
    tx, ty, tz = 1.0, 2.0, 3.0
    R, T = camera_extrinsics_from_colmap(qw, qx, qy, qz, tx, ty, tz)
    np.testing.assert_allclose(R, np.eye(3), atol=1e-10)
    np.testing.assert_allclose(T, np.array([1.0, 2.0, 3.0]), atol=1e-10)


def test_camera_extrinsics_from_colmap_transposes_nontrivial_rotation():
    # 90 deg about Z: qw=cos(45deg), qz=sin(45deg). Non-identity so that
    # transpose(raw) != raw, which lets this test actually distinguish a
    # correctly-transposed R from a silently-dropped .transpose() call
    # (unlike the identity-quaternion test above, where raw == raw.T).
    half = math.pi / 4
    qw, qx, qy, qz = math.cos(half), 0.0, 0.0, math.sin(half)
    tx, ty, tz = 1.0, 2.0, 3.0

    raw = qvec2rotmat(np.array([qw, qx, qy, qz]))
    R, T = camera_extrinsics_from_colmap(qw, qx, qy, qz, tx, ty, tz)

    np.testing.assert_allclose(R, raw.T, atol=1e-10)
    assert not np.allclose(R, raw, atol=1e-10)
    np.testing.assert_allclose(T, np.array([1.0, 2.0, 3.0]), atol=1e-10)


def test_camera_focal_lengths_simple_pinhole_shares_one_focal_length():
    fx, fy = camera_focal_lengths("SIMPLE_PINHOLE", [1113.99, 360.0, 640.0])
    assert fx == pytest.approx(1113.99)
    assert fy == pytest.approx(1113.99)


def test_camera_focal_lengths_pinhole_has_independent_fx_fy():
    fx, fy = camera_focal_lengths("PINHOLE", [800.0, 850.0, 320.0, 240.0])
    assert fx == pytest.approx(800.0)
    assert fy == pytest.approx(850.0)


def test_camera_focal_lengths_rejects_unsupported_model():
    with pytest.raises(ValueError, match="RADIAL"):
        camera_focal_lengths("SIMPLE_RADIAL", [800.0, 320.0, 240.0, 0.01])


def test_camera_params_from_csv_row_computes_fov_and_keeps_metadata():
    row = {
        "image_name": "frame_000025.jpg",
        "qw": 1.0, "qx": 0.0, "qy": 0.0, "qz": 0.0,
        "tx": 0.0, "ty": 0.0, "tz": 0.0,
        "fx": 1000.0, "fy": 1000.0, "cx": 500.0, "cy": 500.0,
        "width": 1000, "height": 1000,
    }
    params = camera_params_from_csv_row(row)
    assert params.image_name == "frame_000025.jpg"
    assert params.width == 1000
    assert params.height == 1000
    assert params.fov_x == pytest.approx(2 * math.atan(0.5), abs=1e-10)
    assert params.fov_y == pytest.approx(2 * math.atan(0.5), abs=1e-10)
