from pathlib import Path

from src.common.config import SceneConfig
from src.training.train_wrapper import build_train_argv, checkpoint_iteration, find_latest_checkpoint


def test_find_latest_checkpoint_returns_none_when_empty(tmp_path):
    assert find_latest_checkpoint(tmp_path) is None


def test_find_latest_checkpoint_picks_highest_iteration(tmp_path):
    (tmp_path / "chkpnt7000.pth").touch()
    (tmp_path / "chkpnt30000.pth").touch()
    (tmp_path / "chkpnt15000.pth").touch()
    result = find_latest_checkpoint(tmp_path)
    assert result == tmp_path / "chkpnt30000.pth"


def test_checkpoint_iteration_parses_number():
    assert checkpoint_iteration(Path("outputs/chair/chkpnt15000.pth")) == 15000


def test_checkpoint_iteration_returns_none_for_non_matching_name():
    assert checkpoint_iteration(Path("outputs/chair/model_final.pth")) is None


def test_build_train_argv_without_resume():
    scene = SceneConfig(
        name="chair",
        root=Path("VAI_NVS_DATA_ROUND2/chair"),
        train_images_dir=Path("VAI_NVS_DATA_ROUND2/chair/train/images"),
        sparse_dir=Path("VAI_NVS_DATA_ROUND2/chair/train/sparse/0"),
        test_poses_csv=Path("VAI_NVS_DATA_ROUND2/chair/test/test_poses.csv"),
    )
    argv = build_train_argv(
        scene, output_dir=Path("outputs/chair/baseline"), iterations=30000,
        resume_checkpoint=None,
    )
    assert "--source_path" in argv
    source_path_arg = argv[argv.index("--source_path") + 1]
    # Must be absolute: the manual Colab step (Task 8 Step 6) runs the
    # subprocess with cwd="third_party/gaussian-splatting", so a relative
    # path would resolve against the wrong directory and silently fail to
    # find the dataset. Must be scene.gs_source_dir, NOT scene.root: the
    # real dataset nests images/ and sparse/0/ one level deeper
    # (<scene>/train/...) than the baseline's expected --source_path
    # layout (<source_path>/images/, <source_path>/sparse/0/ as direct
    # children) — passing scene.root here would point train.py at a
    # directory with no images/ subfolder at all.
    assert Path(source_path_arg).is_absolute()
    assert Path(source_path_arg) == scene.gs_source_dir.resolve()
    assert Path(source_path_arg) != scene.root.resolve()
    assert "--model_path" in argv
    model_path_arg = argv[argv.index("--model_path") + 1]
    assert Path(model_path_arg).is_absolute()
    assert "--iterations" in argv
    assert "30000" in argv
    assert "--start_checkpoint" not in argv


def test_build_train_argv_with_resume_checkpoint():
    scene = SceneConfig(
        name="chair",
        root=Path("VAI_NVS_DATA_ROUND2/chair"),
        train_images_dir=Path("VAI_NVS_DATA_ROUND2/chair/train/images"),
        sparse_dir=Path("VAI_NVS_DATA_ROUND2/chair/train/sparse/0"),
        test_poses_csv=Path("VAI_NVS_DATA_ROUND2/chair/test/test_poses.csv"),
    )
    ckpt = Path("outputs/chair/baseline/chkpnt15000.pth")
    argv = build_train_argv(
        scene, output_dir=Path("outputs/chair/baseline"), iterations=30000,
        resume_checkpoint=ckpt,
    )
    assert "--start_checkpoint" in argv
    ckpt_arg = argv[argv.index("--start_checkpoint") + 1]
    assert Path(ckpt_arg).is_absolute()
    assert Path(ckpt_arg) == ckpt.resolve()
