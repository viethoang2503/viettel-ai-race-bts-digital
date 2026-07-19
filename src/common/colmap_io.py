from __future__ import annotations

import importlib.util
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_VENDORED_REPO = Path(__file__).resolve().parents[2] / "third_party" / "gaussian-splatting"
if str(_VENDORED_REPO) not in sys.path:
    sys.path.insert(0, str(_VENDORED_REPO))

# Import scene/colmap_loader.py directly by file path rather than via
# `from scene.colmap_loader import ...`. The latter executes
# third_party/gaussian-splatting/scene/__init__.py, which pulls in
# gaussian_model.py -> simple_knn._C, a compiled CUDA extension that is not
# built in this environment. colmap_loader.py itself only depends on numpy/
# collections/struct, so loading it standalone avoids that CUDA dependency
# entirely while still using the vendored implementation.
_COLMAP_LOADER_PATH = _VENDORED_REPO / "scene" / "colmap_loader.py"
_spec = importlib.util.spec_from_file_location("_gs_colmap_loader", _COLMAP_LOADER_PATH)
_colmap_loader = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_colmap_loader)

read_extrinsics_binary = _colmap_loader.read_extrinsics_binary
read_intrinsics_binary = _colmap_loader.read_intrinsics_binary
_Point3D = _colmap_loader.Point3D


@dataclass(frozen=True)
class SparseScene:
    cameras: dict
    images: dict
    points3d: dict


def _read_points3d_binary_preserving_ids(path: Path) -> dict:
    """Parse points3D.bin directly, keyed by the REAL COLMAP point3D_id.

    The vendored `read_points3D_binary` (scene/colmap_loader.py) reads each
    point's id off the wire and then discards it, returning flat
    `(xyzs, rgbs, errors)` arrays indexed 0..N-1 in file-storage order
    instead. COLMAP point3D_ids are NOT contiguous (point culling/merging
    during reconstruction leaves gaps) — verified against the real chair
    scene: images.bin's point3D_ids reference ids up to 105456 across
    80491 points, while file-order indexing only produces keys 0..80490.
    Re-keying by file order doesn't just drop out-of-range lookups, it
    silently returns the WRONG point for any id that happens to collide
    with a valid 0..80490 index (~77% of real ids in the chair scene) —
    this matters because Plan 2's depth regularization looks points up by
    exactly this id (`images[...].point3D_ids`), so a wrong or dropped
    point corrupts that loss silently, no crash, no visible symptom in
    training loss. Binary layout verified against the same file:
    - uint64 num_points
    - per point: struct "<QdddBBBd" (id, x,y,z, r,g,b, error) = 43 bytes,
      then uint64 track_length, then track_length * "ii" (image_id,
      point2D_idx) pairs, 8 bytes each.
    """
    points3d: dict[int, _Point3D] = {}
    with open(path, "rb") as fid:
        num_points = struct.unpack("<Q", fid.read(8))[0]
        for _ in range(num_points):
            point_id, x, y, z, r, g, b, error = struct.unpack("<QdddBBBd", fid.read(43))
            track_length = struct.unpack("<Q", fid.read(8))[0]
            track_elems = struct.unpack("<" + "ii" * track_length, fid.read(8 * track_length))
            image_ids = np.array(track_elems[0::2], dtype=int)
            point2D_idxs = np.array(track_elems[1::2], dtype=int)
            points3d[point_id] = _Point3D(
                id=point_id,
                xyz=np.array([x, y, z]),
                rgb=np.array([r, g, b]),
                error=error,
                image_ids=image_ids,
                point2D_idxs=point2D_idxs,
            )
    return points3d


def load_sparse_scene(sparse_dir: Path) -> SparseScene:
    sparse_dir = Path(sparse_dir)
    cameras = read_intrinsics_binary(str(sparse_dir / "cameras.bin"))
    images = read_extrinsics_binary(str(sparse_dir / "images.bin"))
    points3d = _read_points3d_binary_preserving_ids(sparse_dir / "points3D.bin")
    return SparseScene(cameras=cameras, images=images, points3d=points3d)


def compute_scene_bbox(
    points3d: dict, margin_ratio: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    xyz = np.stack([p.xyz for p in points3d.values()], axis=0)
    min_xyz = xyz.min(axis=0)
    max_xyz = xyz.max(axis=0)
    extent = max_xyz - min_xyz
    margin = extent * margin_ratio
    return min_xyz - margin, max_xyz + margin
