from __future__ import annotations

import struct
from pathlib import Path


def write_images_binary(images: dict, path: Path) -> None:
    """Write a COLMAP images.bin containing exactly the given images.

    `images` maps image_id -> an object with `.qvec` (length-4 array,
    [qw,qx,qy,qz]), `.tvec` (length-3 array), `.camera_id` (int), `.name`
    (str) — i.e. the same shape as entries returned by
    scene/colmap_loader.py::read_extrinsics_binary in the vendored repo.

    Per-image point2D track data is not supported by this writer:
    num_points2D is always written as 0. This is safe for this pipeline
    because the baseline train.py never reads point2D tracks from
    images.bin, only pose/camera_id/name.
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
            fid.write(struct.pack("<Q", 0))  # num_points2D
