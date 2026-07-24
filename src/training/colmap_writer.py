from __future__ import annotations

import struct
from pathlib import Path


def write_images_binary(images: dict, path: Path) -> None:
    """Write a COLMAP images.bin containing exactly the given images.

    `images` maps image_id -> an object with `.qvec` (length-4 array,
    [qw,qx,qy,qz]), `.tvec` (length-3 array), `.camera_id` (int), `.name`
    (str) — i.e. the same shape as entries returned by
    scene/colmap_loader.py::read_extrinsics_binary in the vendored repo.

    Per-image point2D tracks are preserved. The variant training path uses
    these observations for sparse-depth supervision, so dropping them while
    building a holdout-filtered scene would silently disable depth loss.
    """
    path = Path(path)
    with open(path, "wb") as fid:
        fid.write(struct.pack("<Q", len(images)))
        for image_id, img in images.items():
            fid.write(struct.pack(
                "<idddddddi",
                int(image_id),
                float(img.qvec[0]), float(img.qvec[1]), float(img.qvec[2]), float(img.qvec[3]),
                float(img.tvec[0]), float(img.tvec[1]), float(img.tvec[2]),
                int(img.camera_id),
            ))
            fid.write(img.name.encode("utf-8") + b"\x00")
            xys = getattr(img, "xys", ())
            point3d_ids = getattr(img, "point3D_ids", ())
            if len(xys) != len(point3d_ids):
                raise ValueError(
                    f"{img.name}: track length mismatch: "
                    f"{len(xys)} xys vs {len(point3d_ids)} point3D_ids"
                )
            fid.write(struct.pack("<Q", len(xys)))
            for xy, point3d_id in zip(xys, point3d_ids):
                fid.write(struct.pack(
                    "<ddq",
                    float(xy[0]),
                    float(xy[1]),
                    int(point3d_id),
                ))


def write_cameras_binary(cameras: dict, path: Path) -> None:
    """Write a COLMAP cameras.bin containing exactly the given cameras.

    Only PINHOLE is supported (model_id=1, num_params=4, params
    [fx, fy, cx, cy]) — the only model this pipeline ever writes, produced
    by undistort_scene.py. Binary layout verified against
    third_party/gaussian-splatting/scene/colmap_loader.py's
    read_intrinsics_binary: uint64 num_cameras, then per camera
    int32 camera_id, int32 model_id, uint64 width, uint64 height, then
    num_params * float64 params.
    """
    path = Path(path)
    with open(path, "wb") as fid:
        fid.write(struct.pack("<Q", len(cameras)))
        for camera_id, cam in cameras.items():
            if cam.model != "PINHOLE":
                raise ValueError(
                    f"write_cameras_binary only supports PINHOLE, got {cam.model}"
                )
            fid.write(struct.pack("<iiQQ", int(camera_id), 1, int(cam.width), int(cam.height)))
            fx, fy, cx, cy = cam.params
            fid.write(struct.pack("<dddd", float(fx), float(fy), float(cx), float(cy)))
