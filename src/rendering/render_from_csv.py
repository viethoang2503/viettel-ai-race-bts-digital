from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image

from src.common.pose_utils import CameraParams, camera_params_from_csv_row


def load_test_poses_csv(csv_path: Path) -> list[CameraParams]:
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        return [camera_params_from_csv_row(row) for row in reader]


def _pil_save_kwargs(out_path: Path) -> dict:
    """Extra kwargs for Image.save() to avoid unnecessary lossy artifacts.

    Only JPEG needs this: at PIL's default quality=75, re-compressing an
    already-rendered image throws away detail that directly lowers
    PSNR/SSIM/LPIPS for no reason. quality=100 + subsampling=0 (4:4:4, no
    chroma subsampling) keeps JPEG output as close to lossless as the
    format allows. PNG is lossless by default and needs no extra kwargs.
    """
    if out_path.suffix.lower() in (".jpg", ".jpeg"):
        return {"quality": 100, "subsampling": 0}
    return {}


def render_all(
    checkpoint_ply,
    csv_path,
    output_dir: Path,
    render_fn,
    params_list: list[CameraParams] | None = None,
    gaussians=None,
) -> list[Path]:
    """Render every camera in params_list (or loaded from csv_path if
    params_list is None) and write one image per row into output_dir, named
    with the EXACT `image_name` string from test_poses.csv (original
    extension preserved, e.g. `.JPG`/`.jpg` — never rewritten to `.png`).
    PIL infers the output format from the filename extension. For
    JPEG-extension outputs, quality/subsampling are maximized (see
    `_pil_save_kwargs`) since our rendered pixels are already the best the
    model can produce — any avoidable lossy re-compression only throws away
    PSNR/SSIM/LPIPS score for no benefit.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if params_list is None:
        params_list = load_test_poses_csv(csv_path)

    written = []
    for params in params_list:
        img_array = render_fn(params, gaussians)
        assert img_array.shape == (params.height, params.width, 3), (
            f"{params.image_name}: expected {(params.height, params.width, 3)}, "
            f"got {img_array.shape}"
        )
        out_path = output_dir / params.image_name
        Image.fromarray(img_array.astype(np.uint8)).save(
            out_path, **_pil_save_kwargs(out_path),
        )
        written.append(out_path)
    return written
