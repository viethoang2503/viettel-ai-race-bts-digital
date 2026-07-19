from __future__ import annotations

import os
import shutil
from dataclasses import replace
from pathlib import Path

from src.common.colmap_io import load_sparse_scene
from src.common.config import SceneConfig
from src.training.colmap_writer import write_images_binary


def build_filtered_scene(
    scene: SceneConfig, holdout_names: set[str], output_dir: Path,
) -> SceneConfig:
    """Create a copy of `scene` with every image in `holdout_names` EXCLUDED
    from both the images folder and sparse/0/images.bin, so training on the
    returned SceneConfig cannot see the holdout images at all.

    ALSO always excludes any image registered in images.bin that has no
    corresponding file in scene.train_images_dir, regardless of
    holdout_names — this is not optional (see Interfaces note above): the
    real dataset always registers more cameras than it distributes files
    for, and the vendored loader crashes on `Image.open()` for any of
    them. Calling this with `holdout_names=set()` is the correct way to
    get a "full data" scene that is still safe to train on.
    """
    output_dir = Path(output_dir)
    images_out = output_dir / "images"
    sparse_out = output_dir / "sparse" / "0"
    images_out.mkdir(parents=True, exist_ok=True)
    sparse_out.mkdir(parents=True, exist_ok=True)

    sparse = load_sparse_scene(scene.sparse_dir)
    file_backed_names = {p.name for p in scene.train_images_dir.iterdir() if p.is_file()}
    exclude = set(holdout_names) | (
        {img.name for img in sparse.images.values()} - file_backed_names
    )
    kept_images = {
        img_id: img for img_id, img in sparse.images.items()
        if img.name not in exclude
    }

    for img in kept_images.values():
        src = (scene.train_images_dir / img.name).resolve()
        dst = images_out / img.name
        if not dst.exists():
            os.symlink(src, dst)

    write_images_binary(kept_images, sparse_out / "images.bin")
    shutil.copy2(scene.sparse_dir / "cameras.bin", sparse_out / "cameras.bin")
    shutil.copy2(scene.sparse_dir / "points3D.bin", sparse_out / "points3D.bin")

    return replace(
        scene,
        root=output_dir,
        train_images_dir=images_out,
        sparse_dir=sparse_out,
    )
