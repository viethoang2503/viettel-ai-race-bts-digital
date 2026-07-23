from __future__ import annotations

import numpy as np

from src.common.pose_utils import qvec2rotmat


def compute_sparse_depth_targets(
    qvec: np.ndarray,
    tvec: np.ndarray,
    xys: np.ndarray,
    point3d_ids: np.ndarray,
    points3d: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Depth-supervision targets for one training image, from its COLMAP
    2D-3D track associations. Uses the RAW COLMAP world-to-camera (R, t) --
    NOT the transposed convention used for the vendored Camera class --
    because depth is a camera-space quantity computed directly from this
    rotation, not passed through the renderer's own view-matrix transpose.
    """
    r_world_to_cam = qvec2rotmat(np.asarray(qvec, dtype=np.float64))
    t_world_to_cam = np.asarray(tvec, dtype=np.float64)

    valid = np.array([
        pid != -1 and pid in points3d for pid in point3d_ids
    ])

    pixel_xy = np.asarray(xys)[valid]
    valid_ids = np.asarray(point3d_ids)[valid]

    depths = []
    for pid in valid_ids:
        world_xyz = points3d[pid].xyz
        cam_xyz = r_world_to_cam @ world_xyz + t_world_to_cam
        depths.append(cam_xyz[2])

    return pixel_xy.reshape(-1, 2), np.array(depths, dtype=np.float64)
