# tests/test_validate_submission.py
import zipfile

from src.common.config import SceneConfig
from src.submission.validate_submission import validate_submission


def _write_csv(path, rows):
    import csv
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "image_name", "qw", "qx", "qy", "qz", "tx", "ty", "tz",
            "fx", "fy", "cx", "cy", "width", "height",
        ])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _make_row(name, width=64, height=32):
    return {
        "image_name": name, "qw": 1, "qx": 0, "qy": 0, "qz": 0,
        "tx": 0, "ty": 0, "tz": 0, "fx": 100, "fy": 100, "cx": 32, "cy": 16,
        "width": width, "height": height,
    }


def _make_png_bytes(width, height):
    from io import BytesIO
    from PIL import Image
    buf = BytesIO()
    Image.new("RGB", (width, height)).save(buf, format="PNG")
    return buf.getvalue()


def test_validate_submission_passes_for_correct_zip(tmp_path):
    csv_path = tmp_path / "scene_a_test_poses.csv"
    _write_csv(csv_path, [_make_row("0001.png"), _make_row("0002.png")])
    scene = SceneConfig(
        name="scene_a", root=tmp_path, train_images_dir=tmp_path,
        sparse_dir=tmp_path, test_poses_csv=csv_path,
    )

    zip_path = tmp_path / "submission.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("scene_a/0001.png", _make_png_bytes(64, 32))
        zf.writestr("scene_a/0002.png", _make_png_bytes(64, 32))

    problems = validate_submission(zip_path, [scene])
    assert problems == []


def test_validate_submission_preserves_original_extension_not_forced_to_png(tmp_path):
    # image_name in the CSV keeps its real extension (.JPG here, matching
    # the real HCM drone scenes) — validator must look for that exact name,
    # not silently expect a renamed .png.
    csv_path = tmp_path / "scene_a_test_poses.csv"
    _write_csv(csv_path, [_make_row("DJI_20241230093428_0050_V.JPG", width=64, height=32)])
    scene = SceneConfig(
        name="scene_a", root=tmp_path, train_images_dir=tmp_path,
        sparse_dir=tmp_path, test_poses_csv=csv_path,
    )

    zip_path = tmp_path / "submission.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("scene_a/DJI_20241230093428_0050_V.JPG", _make_png_bytes(64, 32))

    problems = validate_submission(zip_path, [scene])
    assert problems == []


def test_validate_submission_flags_missing_image(tmp_path):
    csv_path = tmp_path / "scene_a_test_poses.csv"
    _write_csv(csv_path, [_make_row("0001.png"), _make_row("0002.png")])
    scene = SceneConfig(
        name="scene_a", root=tmp_path, train_images_dir=tmp_path,
        sparse_dir=tmp_path, test_poses_csv=csv_path,
    )

    zip_path = tmp_path / "submission.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("scene_a/0001.png", _make_png_bytes(64, 32))
        # 0002.png intentionally missing

    problems = validate_submission(zip_path, [scene])
    assert any("0002.png" in p for p in problems)


def test_validate_submission_flags_wrong_size(tmp_path):
    csv_path = tmp_path / "scene_a_test_poses.csv"
    _write_csv(csv_path, [_make_row("0001.png", width=64, height=32)])
    scene = SceneConfig(
        name="scene_a", root=tmp_path, train_images_dir=tmp_path,
        sparse_dir=tmp_path, test_poses_csv=csv_path,
    )

    zip_path = tmp_path / "submission.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("scene_a/0001.png", _make_png_bytes(32, 32))  # wrong width

    problems = validate_submission(zip_path, [scene])
    assert any("size" in p.lower() for p in problems)


def test_validate_submission_flags_missing_scene_entirely(tmp_path):
    csv_path = tmp_path / "scene_a_test_poses.csv"
    _write_csv(csv_path, [_make_row("0001.png")])
    scene = SceneConfig(
        name="scene_a", root=tmp_path, train_images_dir=tmp_path,
        sparse_dir=tmp_path, test_poses_csv=csv_path,
    )

    zip_path = tmp_path / "submission.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("placeholder.txt", b"empty submission")

    problems = validate_submission(zip_path, [scene])
    assert any("scene_a" in p for p in problems)


def test_validate_submission_uses_effective_submission_dir_not_scene_name(tmp_path):
    # scene.name (internal dataset id) can differ from the folder name
    # required inside submission.zip — validator must key off
    # effective_submission_dir, not name.
    csv_path = tmp_path / "test_poses.csv"
    _write_csv(csv_path, [_make_row("0001.png")])
    scene = SceneConfig(
        name="HCM0421", root=tmp_path, train_images_dir=tmp_path,
        sparse_dir=tmp_path, test_poses_csv=csv_path, submission_dir="scene_001",
    )

    zip_path = tmp_path / "submission.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("scene_001/0001.png", _make_png_bytes(64, 32))

    problems = validate_submission(zip_path, [scene])
    assert problems == []


def test_validate_submission_flags_extra_image_within_a_scene(tmp_path):
    csv_path = tmp_path / "scene_a_test_poses.csv"
    _write_csv(csv_path, [_make_row("0001.png")])
    scene = SceneConfig(
        name="scene_a", root=tmp_path, train_images_dir=tmp_path,
        sparse_dir=tmp_path, test_poses_csv=csv_path,
    )

    zip_path = tmp_path / "submission.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("scene_a/0001.png", _make_png_bytes(64, 32))
        zf.writestr("scene_a/9999.png", _make_png_bytes(64, 32))  # not in test_poses.csv

    problems = validate_submission(zip_path, [scene])
    assert any("9999.png" in p and "unexpected" in p.lower() for p in problems)


def test_validate_submission_flags_extra_top_level_scene_not_in_expected_list(tmp_path):
    csv_path = tmp_path / "scene_a_test_poses.csv"
    _write_csv(csv_path, [_make_row("0001.png")])
    scene = SceneConfig(
        name="scene_a", root=tmp_path, train_images_dir=tmp_path,
        sparse_dir=tmp_path, test_poses_csv=csv_path,
    )

    zip_path = tmp_path / "submission.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("scene_a/0001.png", _make_png_bytes(64, 32))
        zf.writestr("scene_zzz_not_expected/0001.png", _make_png_bytes(64, 32))

    problems = validate_submission(zip_path, [scene])
    assert any("scene_zzz_not_expected" in p and "unexpected" in p.lower() for p in problems)


def test_validate_submission_flags_junk_files_like_macosx(tmp_path):
    csv_path = tmp_path / "scene_a_test_poses.csv"
    _write_csv(csv_path, [_make_row("0001.png")])
    scene = SceneConfig(
        name="scene_a", root=tmp_path, train_images_dir=tmp_path,
        sparse_dir=tmp_path, test_poses_csv=csv_path,
    )

    zip_path = tmp_path / "submission.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("scene_a/0001.png", _make_png_bytes(64, 32))
        zf.writestr("__MACOSX/scene_a/._0001.png", b"junk")

    problems = validate_submission(zip_path, [scene])
    assert any("__MACOSX" in p and "unexpected" in p.lower() for p in problems)


def test_validate_submission_flags_unreadable_image_instead_of_crashing(tmp_path):
    csv_path = tmp_path / "scene_a_test_poses.csv"
    _write_csv(csv_path, [_make_row("0001.png")])
    scene = SceneConfig(
        name="scene_a", root=tmp_path, train_images_dir=tmp_path,
        sparse_dir=tmp_path, test_poses_csv=csv_path,
    )

    zip_path = tmp_path / "submission.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        # Right name, right path, but not decodable image data.
        zf.writestr("scene_a/0001.png", b"not a real image, just garbage bytes")

    problems = validate_submission(zip_path, [scene])
    assert any("0001.png" in p and "not a valid image" in p.lower() for p in problems)


def test_validate_submission_flags_truncated_image_data(tmp_path):
    csv_path = tmp_path / "scene_a_test_poses.csv"
    _write_csv(csv_path, [_make_row("0001.png", width=64, height=32)])
    scene = SceneConfig(
        name="scene_a", root=tmp_path, train_images_dir=tmp_path,
        sparse_dir=tmp_path, test_poses_csv=csv_path,
    )

    full_png = _make_png_bytes(64, 32)
    truncated = full_png[: len(full_png) // 2]  # header intact, pixel data cut

    zip_path = tmp_path / "submission.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("scene_a/0001.png", truncated)

    problems = validate_submission(zip_path, [scene])
    assert any("0001.png" in p and "not a valid image" in p.lower() for p in problems)


def test_validate_submission_flags_non_rgb_image(tmp_path):
    from io import BytesIO
    from PIL import Image as PILImage

    csv_path = tmp_path / "scene_a_test_poses.csv"
    _write_csv(csv_path, [_make_row("0001.png", width=64, height=32)])
    scene = SceneConfig(
        name="scene_a", root=tmp_path, train_images_dir=tmp_path,
        sparse_dir=tmp_path, test_poses_csv=csv_path,
    )

    buf = BytesIO()
    PILImage.new("L", (64, 32)).save(buf, format="PNG")  # grayscale, correct size

    zip_path = tmp_path / "submission.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("scene_a/0001.png", buf.getvalue())

    problems = validate_submission(zip_path, [scene])
    assert any("0001.png" in p and "mode" in p.lower() for p in problems)


def test_validate_submission_flags_duplicate_zip_entries(tmp_path):
    csv_path = tmp_path / "scene_a_test_poses.csv"
    _write_csv(csv_path, [_make_row("0001.png")])
    scene = SceneConfig(
        name="scene_a", root=tmp_path, train_images_dir=tmp_path,
        sparse_dir=tmp_path, test_poses_csv=csv_path,
    )

    zip_path = tmp_path / "submission.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("scene_a/0001.png", _make_png_bytes(64, 32))
        zf.writestr("scene_a/0001.png", _make_png_bytes(64, 32))  # duplicate arcname

    problems = validate_submission(zip_path, [scene])
    assert any("duplicate" in p.lower() and "0001.png" in p for p in problems)


def test_validate_submission_ignores_pure_directory_entries(tmp_path):
    # zip directory marker entries (ending in "/") are not real files and
    # must not be flagged as unexpected — only actual file entries count.
    csv_path = tmp_path / "scene_a_test_poses.csv"
    _write_csv(csv_path, [_make_row("0001.png")])
    scene = SceneConfig(
        name="scene_a", root=tmp_path, train_images_dir=tmp_path,
        sparse_dir=tmp_path, test_poses_csv=csv_path,
    )

    zip_path = tmp_path / "submission.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("scene_a/", b"")  # directory marker
        zf.writestr("scene_a/0001.png", _make_png_bytes(64, 32))

    problems = validate_submission(zip_path, [scene])
    assert problems == []
