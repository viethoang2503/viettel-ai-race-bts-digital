from src.common.config import load_scenes
from src.data_validation.validate_scene import validate_scene


def _get_scene(name):
    return next(s for s in load_scenes("configs/scenes.yaml") if s.name == name)


def test_validate_scene_chair_has_no_problems():
    report = validate_scene(_get_scene("chair"))
    assert report.scene_name == "chair"
    # 263 registered in images.bin, 205 with files on disk — the other 58
    # are exactly test_poses.csv's image names (verified against the real
    # data). This is expected dataset structure, not a problem: see the
    # Interfaces note above.
    assert report.registered_image_count == 263
    assert report.folder_image_count == 205
    assert len(report.registered_without_file) == 58
    assert report.extra_images == []
    assert report.test_pose_row_count == 58
    assert report.camera_model in {"PINHOLE", "SIMPLE_PINHOLE"}
    assert report.problems == []


def test_validate_scene_flags_extra_image_not_registered(tmp_path):
    # The OTHER direction (a file with no COLMAP registration) IS a real
    # anomaly and must still be flagged, unlike registered_without_file.
    # Symlink (not copy) the real files to keep this test fast and cheap.
    import os
    from src.common.config import SceneConfig

    real_scene = _get_scene("chair")
    fake_images_dir = tmp_path / "images"
    fake_images_dir.mkdir()
    for p in real_scene.train_images_dir.iterdir():
        os.symlink(p.resolve(), fake_images_dir / p.name)
    (fake_images_dir / "not_registered_anywhere.jpg").write_bytes(b"fake")

    scene = SceneConfig(
        name="fake", root=tmp_path, train_images_dir=fake_images_dir,
        sparse_dir=real_scene.sparse_dir, test_poses_csv=real_scene.test_poses_csv,
    )
    report = validate_scene(scene)
    assert "not_registered_anywhere.jpg" in report.extra_images
    assert any("not registered" in p.lower() for p in report.problems)


def test_validate_scene_detects_missing_folder_gracefully(tmp_path):
    from src.common.config import SceneConfig

    fake_scene = SceneConfig(
        name="fake",
        root=tmp_path,
        train_images_dir=tmp_path / "does_not_exist",
        sparse_dir=tmp_path / "also_missing",
        test_poses_csv=tmp_path / "test_poses.csv",
    )
    report = validate_scene(fake_scene)
    assert report.problems != []
    assert any("sparse" in p.lower() or "not found" in p.lower() for p in report.problems)


def test_validate_scene_reports_problem_for_corrupt_sparse_binary(tmp_path):
    from src.common.config import SceneConfig

    root = tmp_path / "broken_scene"
    images_dir = root / "train" / "images"
    sparse_dir = root / "train" / "sparse" / "0"
    images_dir.mkdir(parents=True)
    sparse_dir.mkdir(parents=True)

    # cameras.bin exists but is truncated garbage — not a valid COLMAP file.
    (sparse_dir / "cameras.bin").write_bytes(b"\x01\x02\x03")
    (sparse_dir / "images.bin").write_bytes(b"")
    (sparse_dir / "points3D.bin").write_bytes(b"")

    scene = SceneConfig(
        name="broken_scene", root=root, train_images_dir=images_dir,
        sparse_dir=sparse_dir, test_poses_csv=root / "test" / "test_poses.csv",
    )

    report = validate_scene(scene)

    assert report.problems != []
    assert any("sparse" in p.lower() for p in report.problems)


def test_validate_scene_reports_problem_for_missing_sparse_file(tmp_path):
    from src.common.config import SceneConfig

    root = tmp_path / "missing_camera_file"
    images_dir = root / "train" / "images"
    sparse_dir = root / "train" / "sparse" / "0"
    images_dir.mkdir(parents=True)
    sparse_dir.mkdir(parents=True)
    # images.bin and points3D.bin present, cameras.bin missing entirely.
    (sparse_dir / "images.bin").write_bytes(b"")
    (sparse_dir / "points3D.bin").write_bytes(b"")

    scene = SceneConfig(
        name="missing_camera_file", root=root, train_images_dir=images_dir,
        sparse_dir=sparse_dir, test_poses_csv=root / "test" / "test_poses.csv",
    )

    report = validate_scene(scene)

    assert report.problems != []
    assert any("cameras.bin" in p for p in report.problems)


def _write_test_csv(path, rows):
    import csv as csv_module
    with open(path, "w", newline="") as f:
        writer = csv_module.DictWriter(f, fieldnames=[
            "image_name", "qw", "qx", "qy", "qz", "tx", "ty", "tz",
            "fx", "fy", "cx", "cy", "width", "height",
        ])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _valid_row(name="a.jpg", width=64, height=32):
    return {
        "image_name": name, "qw": 1, "qx": 0, "qy": 0, "qz": 0,
        "tx": 0, "ty": 0, "tz": 0, "fx": 100, "fy": 100, "cx": 32, "cy": 16,
        "width": width, "height": height,
    }


def test_validate_scene_flags_duplicate_image_name(tmp_path):
    from src.common.config import SceneConfig

    csv_path = tmp_path / "test_poses.csv"
    _write_test_csv(csv_path, [_valid_row("dup.jpg"), _valid_row("dup.jpg")])
    scene = SceneConfig(
        name="fake", root=tmp_path, train_images_dir=tmp_path,
        sparse_dir=tmp_path, test_poses_csv=csv_path,
    )
    # bypass sparse-dir checks by pointing at a scene with valid sparse data
    real_sparse = _get_scene("chair").sparse_dir
    real_images = _get_scene("chair").train_images_dir
    scene = SceneConfig(
        name="fake", root=tmp_path, train_images_dir=real_images,
        sparse_dir=real_sparse, test_poses_csv=csv_path,
    )
    report = validate_scene(scene)
    assert any("duplicate" in p.lower() for p in report.problems)


def test_validate_scene_flags_non_numeric_and_non_positive_dims(tmp_path):
    from src.common.config import SceneConfig

    csv_path = tmp_path / "test_poses.csv"
    bad_row = _valid_row("bad.jpg")
    bad_row["width"] = "not_a_number"
    zero_row = _valid_row("zero.jpg", width=0, height=10)
    _write_test_csv(csv_path, [bad_row, zero_row])

    real_sparse = _get_scene("chair").sparse_dir
    real_images = _get_scene("chair").train_images_dir
    scene = SceneConfig(
        name="fake", root=tmp_path, train_images_dir=real_images,
        sparse_dir=real_sparse, test_poses_csv=csv_path,
    )
    report = validate_scene(scene)
    assert any("bad.jpg" in p and "numeric" in p.lower() for p in report.problems)
    assert any("zero.jpg" in p and ("width" in p.lower() or "height" in p.lower()) for p in report.problems)


def test_validate_scene_flags_test_pose_with_no_colmap_registration(tmp_path):
    # This IS the one case where a name outside train/images/ is a real
    # problem: a test pose whose image_name was never registered in
    # images.bin at all has no corresponding COLMAP camera anywhere.
    from src.common.config import SceneConfig

    csv_path = tmp_path / "test_poses.csv"
    _write_test_csv(csv_path, [_valid_row("totally_unknown_image.jpg")])

    real_scene = _get_scene("chair")
    scene = SceneConfig(
        name="fake", root=tmp_path, train_images_dir=real_scene.train_images_dir,
        sparse_dir=real_scene.sparse_dir, test_poses_csv=csv_path,
    )
    report = validate_scene(scene)
    assert any(
        "totally_unknown_image.jpg" in p and "not registered" in p.lower()
        for p in report.problems
    )
