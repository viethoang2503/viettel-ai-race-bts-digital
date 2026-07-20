from __future__ import annotations

import shutil
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np

from src.common.colmap_io import load_sparse_scene
from src.common.config import SceneConfig
from src.training.colmap_writer import write_cameras_binary

ALREADY_SUPPORTED_MODELS = {"PINHOLE", "SIMPLE_PINHOLE"}
UNDISTORTABLE_MODELS = {"SIMPLE_RADIAL"}


@dataclass(frozen=True)
class _PinholeCamera:
    model: str
    width: int
    height: int
    params: np.ndarray


def undistort_scene(scene: SceneConfig, output_dir: Path) -> SceneConfig:
    """Undistort a SIMPLE_RADIAL scene into a PINHOLE copy the vendored 3DGS
    can train on directly; a no-op passthrough (no copy) for scenes already
    using a supported model (chair, bonsai).

    Real BTS scenes are registered by COLMAP as SIMPLE_RADIAL
    (params = [f, cx, cy, k1]) — third_party/gaussian-splatting/scene/
    dataset_readers.py only handles PINHOLE/SIMPLE_PINHOLE and asserts on
    anything else. This keeps the same focal length and principal point
    (newCameraMatrix=k_matrix, no re-centering/cropping) and only removes
    the k1 radial term from the pixels via cv2.undistort — poses in
    images.bin and points in points3D.bin are unaffected by this and are
    copied through unchanged; only cameras.bin (model -> PINHOLE) and the
    image pixels change.
    """
    sparse = load_sparse_scene(scene.sparse_dir)
    camera_models = {cam.model for cam in sparse.cameras.values()}

    if camera_models <= ALREADY_SUPPORTED_MODELS:
        return scene

    unsupported = camera_models - ALREADY_SUPPORTED_MODELS - UNDISTORTABLE_MODELS
    if unsupported:
        raise ValueError(f"cannot undistort camera model(s) {sorted(unsupported)}")

    output_dir = Path(output_dir)
    images_out = output_dir / "images"
    sparse_out = output_dir / "sparse" / "0"
    images_out.mkdir(parents=True, exist_ok=True)
    sparse_out.mkdir(parents=True, exist_ok=True)

    new_cameras: dict[int, _PinholeCamera] = {}
    undistort_maps: dict[int, tuple] = {}
    for camera_id, cam in sparse.cameras.items():
        if cam.model == "SIMPLE_RADIAL":
            f, cx, cy, k1 = cam.params
            k_matrix = np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]])
            dist_coeffs = np.array([k1, 0.0, 0.0, 0.0])
            undistort_maps[camera_id] = (k_matrix, dist_coeffs)
            new_cameras[camera_id] = _PinholeCamera(
                model="PINHOLE", width=cam.width, height=cam.height,
                params=np.array([f, f, cx, cy]),
            )
        else:
            undistort_maps[camera_id] = None
            new_cameras[camera_id] = _PinholeCamera(
                model=cam.model, width=cam.width, height=cam.height, params=cam.params,
            )

    for img in sparse.images.values():
        src_path = scene.train_images_dir / img.name
        if not src_path.is_file():
            continue  # registered-without-file images have no pixels to undistort
        dst_path = images_out / img.name
        mapping = undistort_maps[img.camera_id]
        if mapping is None:
            shutil.copy2(src_path, dst_path)
            continue
        k_matrix, dist_coeffs = mapping
        pixels = cv2.imread(str(src_path))
        if pixels is None:
            raise ValueError(f"cv2 could not read image: {src_path}")
        undistorted = cv2.undistort(pixels, k_matrix, dist_coeffs, newCameraMatrix=k_matrix)
        cv2.imwrite(str(dst_path), undistorted)

    write_cameras_binary(new_cameras, sparse_out / "cameras.bin")
    shutil.copy2(scene.sparse_dir / "images.bin", sparse_out / "images.bin")
    shutil.copy2(scene.sparse_dir / "points3D.bin", sparse_out / "points3D.bin")

    return replace(scene, root=output_dir, train_images_dir=images_out, sparse_dir=sparse_out)
