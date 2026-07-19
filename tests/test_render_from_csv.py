# tests/test_render_from_csv.py
from pathlib import Path

import numpy as np

from src.rendering.render_from_csv import load_test_poses_csv, render_all

CHAIR_TEST_CSV = Path("VAI_NVS_DATA_ROUND2/chair/test/test_poses.csv")


def test_load_test_poses_csv_reads_all_rows_with_correct_size():
    params_list = load_test_poses_csv(CHAIR_TEST_CSV)
    assert len(params_list) == 58
    first = params_list[0]
    assert first.image_name.endswith(".jpg")
    assert first.width == 720
    assert first.height == 1280


def test_render_all_writes_one_png_per_row_with_correct_name_and_size(tmp_path):
    params_list = load_test_poses_csv(CHAIR_TEST_CSV)[:3]  # keep the test fast

    def fake_render_fn(camera_params, gaussians):
        return np.zeros((camera_params.height, camera_params.width, 3), dtype=np.uint8)

    written = render_all(
        checkpoint_ply=None,  # unused by fake_render_fn
        csv_path=None,
        output_dir=tmp_path,
        render_fn=fake_render_fn,
        params_list=params_list,
    )

    assert len(written) == 3
    for path, params in zip(written, params_list):
        # Filename must match test_poses.csv image_name EXACTLY, including
        # its original extension — the exam spec (debai.md section 1.4)
        # says image_name IS the required output filename; nothing in the
        # spec asks for extension normalization to .png, and the real CSVs
        # use .jpg/.JPG.
        assert path.name == params.image_name
        assert path.exists()
        from PIL import Image
        img = Image.open(path)
        assert img.size == (params.width, params.height)


def test_render_all_preserves_uppercase_jpg_extension_from_real_drone_naming():
    # Real HCM scene CSVs use names like DJI_20241230093428_0050_V.JPG —
    # this must round-trip exactly, not get rewritten to .png.
    from src.common.pose_utils import CameraParams
    import numpy as np

    params = CameraParams(
        image_name="DJI_20241230093428_0050_V.JPG",
        R=np.eye(3), T=np.zeros(3), fov_x=1.0, fov_y=1.0, width=8, height=4,
    )

    def fake_render_fn(camera_params, gaussians):
        return np.zeros((camera_params.height, camera_params.width, 3), dtype=np.uint8)

    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        written = render_all(
            checkpoint_ply=None, csv_path=None, output_dir=Path(tmp),
            render_fn=fake_render_fn, params_list=[params],
        )
        assert written[0].name == "DJI_20241230093428_0050_V.JPG"


def test_pil_save_kwargs_maximizes_jpeg_quality_but_leaves_png_alone():
    from src.rendering.render_from_csv import _pil_save_kwargs

    assert _pil_save_kwargs(Path("a.JPG")) == {"quality": 100, "subsampling": 0}
    assert _pil_save_kwargs(Path("a.jpg")) == {"quality": 100, "subsampling": 0}
    assert _pil_save_kwargs(Path("a.jpeg")) == {"quality": 100, "subsampling": 0}
    assert _pil_save_kwargs(Path("a.png")) == {}
