from pathlib import Path

import numpy as np

from src.common.colmap_io import load_sparse_scene
from src.common.config import load_scenes
from src.training.colmap_writer import write_images_binary
from src.training.holdout_scene import build_filtered_scene


def _get_scene(name):
    return next(s for s in load_scenes("configs/scenes.yaml") if s.name == name)


def test_write_images_binary_round_trips_with_the_vendored_reader(tmp_path):
    scene = _get_scene("chair")
    sparse = load_sparse_scene(scene.sparse_dir)
    # keep just the first 5 images so the test is fast
    subset = dict(list(sparse.images.items())[:5])

    out_path = tmp_path / "images.bin"
    write_images_binary(subset, out_path)

    reloaded = load_sparse_scene.__globals__["read_extrinsics_binary"](str(out_path))
    assert set(reloaded.keys()) == set(subset.keys())
    for image_id, original in subset.items():
        round_tripped = reloaded[image_id]
        assert round_tripped.name == original.name
        assert round_tripped.camera_id == original.camera_id
        np.testing.assert_allclose(round_tripped.qvec, original.qvec, atol=1e-9)
        np.testing.assert_allclose(round_tripped.tvec, original.tvec, atol=1e-9)


def _file_backed_names(scene) -> set[str]:
    import os
    return set(os.listdir(scene.train_images_dir))


def test_build_filtered_scene_excludes_holdout_images_from_bin_and_folder(tmp_path):
    scene = _get_scene("chair")
    sparse = load_sparse_scene(scene.sparse_dir)
    file_backed = sorted(_file_backed_names(scene))  # only names with a real file
    holdout = set(file_backed[:5])

    filtered = build_filtered_scene(scene, holdout, tmp_path / "filtered_chair")

    filtered_sparse = load_sparse_scene(filtered.sparse_dir)
    filtered_names = {img.name for img in filtered_sparse.images.values()}

    # Expected kept set: file-backed names minus the chosen holdout.
    # Registered-without-file names (e.g. test_poses.csv images) must be
    # gone too even though they were never in `holdout`.
    assert filtered_names == set(file_backed) - holdout
    registered_without_file = {img.name for img in sparse.images.values()} - set(file_backed)
    assert registered_without_file, "test fixture assumption broken: chair should have some"
    assert filtered_names.isdisjoint(registered_without_file)

    for name in filtered_names:
        assert (filtered.train_images_dir / name).exists()
    for name in holdout:
        assert not (filtered.train_images_dir / name).exists()
    for name in registered_without_file:
        assert not (filtered.train_images_dir / name).exists()

    # cameras.bin and points3D.bin are carried over unchanged
    assert (filtered.sparse_dir / "cameras.bin").read_bytes() == \
        (scene.sparse_dir / "cameras.bin").read_bytes()
    assert (filtered.sparse_dir / "points3D.bin").read_bytes() == \
        (scene.sparse_dir / "points3D.bin").read_bytes()

    # retained images keep identical pose data (no silent corruption)
    orig_by_name = {img.name: img for img in sparse.images.values()}
    filt_by_name = {img.name: img for img in filtered_sparse.images.values()}
    for name in filtered_names:
        np.testing.assert_allclose(orig_by_name[name].qvec, filt_by_name[name].qvec, atol=1e-9)
        np.testing.assert_allclose(orig_by_name[name].tvec, filt_by_name[name].tvec, atol=1e-9)


def test_build_filtered_scene_excludes_registered_without_file_even_with_empty_holdout(tmp_path):
    # This is the exact case Task 12's "final full training" phase relies
    # on: build_filtered_scene(scene, set(), ...) must still be safe to
    # feed into the real Scene()/train.py loader, i.e. it must never leave
    # a registered-without-file image (like a test_poses.csv image) in the
    # output, even though holdout_names is empty.
    scene = _get_scene("chair")
    sparse = load_sparse_scene(scene.sparse_dir)
    file_backed = _file_backed_names(scene)
    registered_without_file = {img.name for img in sparse.images.values()} - file_backed
    assert registered_without_file  # fixture sanity: chair has 58 of these

    filtered = build_filtered_scene(scene, set(), tmp_path / "filtered_chair_full")

    filtered_sparse = load_sparse_scene(filtered.sparse_dir)
    filtered_names = {img.name for img in filtered_sparse.images.values()}
    assert filtered_names == file_backed
    assert filtered_names.isdisjoint(registered_without_file)
    for name in filtered_names:
        assert (filtered.train_images_dir / name).exists()
