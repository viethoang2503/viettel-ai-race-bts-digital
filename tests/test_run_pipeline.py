from pathlib import Path

import numpy as np
import pytest
import torch

from src.common.config import SceneConfig
from src.orchestrator.run_pipeline import run_baseline_pipeline


def _chair_scene():
    return SceneConfig(
        name="chair",
        root=Path("VAI_NVS_DATA_ROUND2/chair"),
        train_images_dir=Path("VAI_NVS_DATA_ROUND2/chair/train/images"),
        sparse_dir=Path("VAI_NVS_DATA_ROUND2/chair/train/sparse/0"),
        test_poses_csv=Path("VAI_NVS_DATA_ROUND2/chair/test/test_poses.csv"),
        submission_dir="chair",
    )


class _StubLpipsModel:
    """Same stub as Task 6 — avoids downloading real AlexNet weights just
    to test orchestration wiring, which needs no network access."""

    def __call__(self, pred_tensor, gt_tensor):
        identical = torch.allclose(pred_tensor, gt_tensor)
        return torch.tensor(0.0 if identical else 1.0)


def _fake_render_fn(checkpoint, params_list, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for params in params_list:
        from PIL import Image
        path = output_dir / params.image_name
        Image.fromarray(
            np.zeros((params.height, params.width, 3), dtype=np.uint8)
        ).save(path)
        written.append(path)
    return written


def _wrong_size_render_fn(checkpoint, params_list, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for params in params_list:
        from PIL import Image
        path = output_dir / params.image_name
        # Deliberately the wrong size so validate_submission flags it.
        Image.fromarray(np.zeros((10, 10, 3), dtype=np.uint8)).save(path)
        written.append(path)
    return written


def test_run_baseline_pipeline_produces_scores_and_valid_zip(tmp_path):
    scene = _chair_scene()
    train_calls = []

    def fake_train_fn(scene_arg, output_dir):
        train_calls.append(Path(scene_arg.root).resolve())
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        ckpt = output_dir / "fake_checkpoint.pth"
        ckpt.touch()
        return ckpt

    result = run_baseline_pipeline(
        scenes=[scene],
        train_fn=fake_train_fn,
        render_fn=_fake_render_fn,
        lpips_model=_StubLpipsModel(),
        psnr_max=30.0,
        output_root=tmp_path,
    )

    # train_fn must be called exactly twice, and NEITHER call may use the
    # original scene.root directly: images.bin registers more cameras than
    # have files on disk for this dataset (test_poses.csv images among
    # them), and the real loader crashes on any of them — so Phase A (
    # holdout-excluded) and Phase B (empty-holdout "full data") must BOTH
    # go through build_filtered_scene into a distinct scratch directory,
    # proving the leak-free AND crash-free wiring from Task 8b is actually
    # used for both phases, not just defined and ignored for one of them.
    assert len(train_calls) == 2
    original_root = Path(scene.root).resolve()
    assert original_root not in train_calls, (
        "both phases must pass a build_filtered_scene copy, never the raw "
        "scene.root, since images.bin registers images with no file on disk"
    )
    assert train_calls[0] != train_calls[1], (
        "Phase A (holdout-excluded) and Phase B (full data) must use distinct scene copies"
    )

    assert result.skipped_scenes == {}
    assert "chair" in result.per_scene_scores
    assert 0.0 <= result.per_scene_scores["chair"] <= 1.0
    assert result.submission_zip is not None

    # combine_score blends lpips/ssim/psnr into one number — the raw
    # per-metric averages must also be kept, so a low score can be
    # diagnosed instead of only ever seeing the single blended value.
    assert "chair" in result.per_scene_metrics
    scene_metrics = result.per_scene_metrics["chair"]
    assert set(scene_metrics.keys()) == {"lpips", "ssim", "psnr"}
    for value in scene_metrics.values():
        assert isinstance(value, float)
    assert result.submission_zip.exists()
    # black-image render vs real holdout images should not be a perfect score
    assert result.per_scene_scores["chair"] < 0.9


def test_run_baseline_pipeline_skips_invalid_scene_without_calling_train_fn(tmp_path):
    broken_scene = SceneConfig(
        name="broken",
        root=tmp_path / "broken",
        train_images_dir=tmp_path / "broken" / "does_not_exist",
        sparse_dir=tmp_path / "broken" / "also_missing",
        test_poses_csv=tmp_path / "broken" / "test_poses.csv",
        submission_dir="broken",
    )
    train_calls = []

    def fake_train_fn(scene_arg, output_dir):
        train_calls.append(scene_arg)
        raise AssertionError("train_fn must not be called for an invalid scene")

    result = run_baseline_pipeline(
        scenes=[broken_scene],
        train_fn=fake_train_fn,
        render_fn=_fake_render_fn,
        lpips_model=_StubLpipsModel(),
        psnr_max=30.0,
        output_root=tmp_path,
    )

    assert train_calls == []
    assert "broken" in result.skipped_scenes
    assert result.skipped_scenes["broken"] != []
    assert "broken" not in result.per_scene_scores
    # Fail-closed: a skipped scene must withhold the whole submission, not
    # just omit that scene from an otherwise-produced zip. The exam voids
    # the ENTIRE score for a missing scene (debai.md section 1.6/8.4), so
    # packaging a zip that's already known to be incomplete would be worse
    # than not packaging one at all.
    assert result.submission_zip is None
    assert any("broken" in p and "skipped" in p.lower() for p in result.validation_problems)


def test_run_baseline_pipeline_withholds_submission_even_if_other_scenes_succeed(tmp_path):
    good_scene = _chair_scene()
    broken_scene = SceneConfig(
        name="broken",
        root=tmp_path / "broken",
        train_images_dir=tmp_path / "broken" / "does_not_exist",
        sparse_dir=tmp_path / "broken" / "also_missing",
        test_poses_csv=tmp_path / "broken" / "test_poses.csv",
        submission_dir="broken",
    )

    def fake_train_fn(scene_arg, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        ckpt = output_dir / "fake_checkpoint.pth"
        ckpt.touch()
        return ckpt

    result = run_baseline_pipeline(
        scenes=[good_scene, broken_scene],
        train_fn=fake_train_fn,
        render_fn=_fake_render_fn,
        lpips_model=_StubLpipsModel(),
        psnr_max=30.0,
        output_root=tmp_path,
    )

    # "chair" succeeded and has a score, but the overall submission must
    # still be withheld because "broken" was skipped — one good scene does
    # not entitle the pipeline to ship a partial zip.
    assert "chair" in result.per_scene_scores
    assert "broken" in result.skipped_scenes
    assert result.submission_zip is None


def test_run_baseline_pipeline_withholds_zip_when_validation_finds_problems(tmp_path):
    scene = _chair_scene()

    def fake_train_fn(scene_arg, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        ckpt = output_dir / "fake_checkpoint.pth"
        ckpt.touch()
        return ckpt

    result = run_baseline_pipeline(
        scenes=[scene],
        train_fn=fake_train_fn,
        render_fn=_wrong_size_render_fn,
        lpips_model=_StubLpipsModel(),
        psnr_max=30.0,
        output_root=tmp_path,
    )

    # validate_submission must have found the wrong-size images and the
    # pipeline must withhold the zip as a valid submission...
    assert result.validation_problems != []
    assert result.submission_zip is None
    # ...but the zip file itself must still exist on disk for debugging —
    # fail-closed means "don't expose it as valid", not "delete evidence".
    assert (tmp_path / "submission.zip").exists()


def test_run_baseline_pipeline_computes_matching_fov_for_simple_pinhole_scene(tmp_path):
    scene = _chair_scene()
    captured_holdout_params = []

    def fake_train_fn(scene_arg, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        ckpt = output_dir / "fake_checkpoint.pth"
        ckpt.touch()
        return ckpt

    def capturing_render_fn(checkpoint, params_list, output_dir):
        if "holdout_render" in str(output_dir):
            captured_holdout_params.extend(params_list)
        return _fake_render_fn(checkpoint, params_list, output_dir)

    run_baseline_pipeline(
        scenes=[scene],
        train_fn=fake_train_fn,
        render_fn=capturing_render_fn,
        lpips_model=_StubLpipsModel(),
        psnr_max=30.0,
        output_root=tmp_path,
    )

    assert captured_holdout_params, "expected at least one holdout camera"
    # chair's real COLMAP camera is SIMPLE_PINHOLE with params
    # [f=1113.98975937, cx=360.0, cy=640.0] and image size 720x1280 (verified
    # directly against VAI_NVS_DATA_ROUND2/chair). fov_x and fov_y need not
    # be equal (the image isn't square), but both must derive from the same
    # shared focal length f — before the fix, fov_y was derived from cx
    # (360.0) instead, giving a very different, wrong value.
    import math
    expected_fov_x = 2 * math.atan(720 / (2 * 1113.98975937))
    expected_fov_y = 2 * math.atan(1280 / (2 * 1113.98975937))
    for params in captured_holdout_params:
        assert params.fov_x == pytest.approx(expected_fov_x, rel=1e-6)
        assert params.fov_y == pytest.approx(expected_fov_y, rel=1e-6)


class _FakeImageForRadial:
    def __init__(self, name, camera_id, qvec, tvec):
        self.qvec = qvec
        self.tvec = tvec
        self.camera_id = camera_id
        self.name = name


def _make_simple_radial_scene(tmp_path):
    import struct

    from src.training.colmap_writer import write_images_binary

    root = tmp_path / "hcm_like_scene"
    images_dir = root / "train" / "images"
    sparse_dir = root / "train" / "sparse" / "0"
    images_dir.mkdir(parents=True)
    sparse_dir.mkdir(parents=True)

    with open(sparse_dir / "cameras.bin", "wb") as fid:
        fid.write(struct.pack("<Q", 1))
        fid.write(struct.pack("<iiQQ", 1, 2, 64, 48))  # model_id 2 = SIMPLE_RADIAL
        fid.write(struct.pack("<dddd", 80.0, 32.0, 24.0, 0.02))

    images = {
        i: _FakeImageForRadial(
            f"{i:04d}.jpg", 1,
            qvec=np.array([1.0, 0.0, 0.0, 0.0]),
            tvec=np.array([float(i) * 0.1, 0.0, 0.0]),
        )
        for i in range(1, 9)  # 8 images so holdout_ratio=0.125 selects exactly 1
    }

    from PIL import Image as PILImage
    for name in images.values():
        PILImage.new("RGB", (64, 48), color=(10, 20, 30)).save(images_dir / name.name)

    # Real scenes register test_poses.csv image names in images.bin with no
    # corresponding file (verified against VAI_NVS_DATA_ROUND2 — see Task 4/5
    # of the core plan) — match that here, AFTER writing the 8 real training
    # image files above, so validate_scene doesn't flag the test pose as
    # unregistered (an unrelated fixture gap) while also not accidentally
    # writing a file for it (which would defeat the "registered without
    # file" case this is meant to reproduce).
    images[9] = _FakeImageForRadial(
        "test_0001.jpg", 1,
        qvec=np.array([1.0, 0.0, 0.0, 0.0]), tvec=np.array([0.0, 0.0, 0.0]),
    )
    write_images_binary(images, sparse_dir / "images.bin")
    with open(sparse_dir / "points3D.bin", "wb") as fid:
        fid.write(struct.pack("<Q", 0))

    test_poses_dir = root / "test"
    test_poses_dir.mkdir()
    csv_path = test_poses_dir / "test_poses.csv"
    with open(csv_path, "w", newline="") as f:
        import csv
        writer = csv.DictWriter(f, fieldnames=[
            "image_name", "qw", "qx", "qy", "qz", "tx", "ty", "tz",
            "fx", "fy", "cx", "cy", "width", "height",
        ])
        writer.writeheader()
        writer.writerow({
            "image_name": "test_0001.jpg", "qw": 1, "qx": 0, "qy": 0, "qz": 0,
            "tx": 0, "ty": 0, "tz": 0, "fx": 80, "fy": 80, "cx": 32, "cy": 24,
            "width": 64, "height": 48,
        })

    return SceneConfig(
        name="hcm_like_scene", root=root, train_images_dir=images_dir,
        sparse_dir=sparse_dir, test_poses_csv=csv_path, submission_dir="hcm_like_scene",
    )


def test_run_baseline_pipeline_trains_simple_radial_scene_via_undistortion(tmp_path):
    scene = _make_simple_radial_scene(tmp_path)

    def fake_train_fn(scene_arg, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        ckpt = output_dir / "fake_checkpoint.pth"
        ckpt.touch()
        return ckpt

    result = run_baseline_pipeline(
        scenes=[scene],
        train_fn=fake_train_fn,
        render_fn=_fake_render_fn,
        lpips_model=_StubLpipsModel(),
        psnr_max=30.0,
        output_root=tmp_path,
    )

    # Before this fix, validate_scene would flag SIMPLE_RADIAL as
    # unsupported and this scene would land in skipped_scenes, withholding
    # the whole submission.
    assert result.skipped_scenes == {}
    assert "hcm_like_scene" in result.per_scene_scores
    assert result.submission_zip is not None
