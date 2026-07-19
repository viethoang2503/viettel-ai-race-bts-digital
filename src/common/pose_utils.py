from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def qvec2rotmat(qvec: np.ndarray) -> np.ndarray:
    """COLMAP quaternion (qw, qx, qy, qz) -> 3x3 rotation matrix.

    Same convention as scene/colmap_loader.py in the vendored baseline repo.
    """
    qw, qx, qy, qz = qvec
    return np.array([
        [1 - 2 * qy**2 - 2 * qz**2, 2 * qx * qy - 2 * qw * qz, 2 * qz * qx + 2 * qw * qy],
        [2 * qx * qy + 2 * qw * qz, 1 - 2 * qx**2 - 2 * qz**2, 2 * qy * qz - 2 * qw * qx],
        [2 * qz * qx - 2 * qw * qy, 2 * qy * qz + 2 * qw * qx, 1 - 2 * qx**2 - 2 * qy**2],
    ])


def focal2fov(focal: float, pixels: int) -> float:
    """Pinhole focal length (pixels) -> field of view (radians)."""
    return 2 * math.atan(pixels / (2 * focal))


def camera_extrinsics_from_colmap(
    qw: float, qx: float, qy: float, qz: float, tx: float, ty: float, tz: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert COLMAP world-to-camera (qvec, tvec) into the (R, T) convention
    expected by the vendored gaussian-splatting `scene.cameras.Camera` class:
    R is the transpose of the COLMAP world-to-camera rotation, T is the
    COLMAP world-to-camera translation unchanged.
    """
    qvec = np.array([qw, qx, qy, qz], dtype=np.float64)
    r_world_to_cam = qvec2rotmat(qvec)
    r = np.transpose(r_world_to_cam)
    t = np.array([tx, ty, tz], dtype=np.float64)
    return r, t


@dataclass(frozen=True)
class CameraParams:
    image_name: str
    R: np.ndarray
    T: np.ndarray
    fov_x: float
    fov_y: float
    width: int
    height: int


def camera_params_from_csv_row(row: dict) -> CameraParams:
    """Build CameraParams from one parsed row of test_poses.csv.

    `row` values may be strings (if read via csv.DictReader) or already
    numeric (if read via pandas) — this function coerces explicitly.
    """
    r, t = camera_extrinsics_from_colmap(
        float(row["qw"]), float(row["qx"]), float(row["qy"]), float(row["qz"]),
        float(row["tx"]), float(row["ty"]), float(row["tz"]),
    )
    width = int(row["width"])
    height = int(row["height"])
    fov_x = focal2fov(float(row["fx"]), width)
    fov_y = focal2fov(float(row["fy"]), height)
    return CameraParams(
        image_name=str(row["image_name"]),
        R=r, T=t, fov_x=fov_x, fov_y=fov_y, width=width, height=height,
    )
