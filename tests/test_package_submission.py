import zipfile

from src.submission.package_submission import package_submission


def test_package_submission_writes_expected_zip_structure(tmp_path):
    scene_a_dir = tmp_path / "rendered_a"
    scene_a_dir.mkdir()
    (scene_a_dir / "0001.png").write_bytes(b"fake-png-a1")
    (scene_a_dir / "0002.png").write_bytes(b"fake-png-a2")

    scene_b_dir = tmp_path / "rendered_b"
    scene_b_dir.mkdir()
    (scene_b_dir / "0001.png").write_bytes(b"fake-png-b1")

    output_zip = tmp_path / "submission.zip"
    result = package_submission(
        scene_render_dirs={"scene_001": scene_a_dir, "scene_002": scene_b_dir},
        output_zip=output_zip,
    )

    assert result == output_zip
    with zipfile.ZipFile(output_zip) as zf:
        names = set(zf.namelist())
        assert names == {
            "scene_001/0001.png", "scene_001/0002.png", "scene_002/0001.png",
        }
        assert zf.read("scene_001/0001.png") == b"fake-png-a1"
