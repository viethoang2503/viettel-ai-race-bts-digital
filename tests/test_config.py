# tests/test_config.py
from pathlib import Path
from src.common.config import load_scenes


def test_load_scenes_returns_seven_scenes_with_correct_paths():
    scenes = load_scenes("configs/scenes.yaml")
    assert len(scenes) == 7
    names = {s.name for s in scenes}
    assert names == {
        "HCM0421", "HCM0539", "HCM0540", "HCM0644", "HCM0674", "chair", "bonsai",
    }
    chair = next(s for s in scenes if s.name == "chair")
    assert chair.root == Path("VAI_NVS_DATA_ROUND2/chair")
    assert chair.train_images_dir == Path("VAI_NVS_DATA_ROUND2/chair/train/images")
    assert chair.sparse_dir == Path("VAI_NVS_DATA_ROUND2/chair/train/sparse/0")
    assert chair.test_poses_csv == Path("VAI_NVS_DATA_ROUND2/chair/test/test_poses.csv")
    assert chair.test_poses_csv.exists()
    assert chair.submission_dir == "chair"
    assert chair.effective_submission_dir == "chair"
    # gs_source_dir must be the directory that directly contains images/
    # and sparse/0/ as siblings, per the baseline's expected layout —
    # NOT chair.root, which is one level too shallow for this dataset.
    assert chair.gs_source_dir == Path("VAI_NVS_DATA_ROUND2/chair/train")
    assert (chair.gs_source_dir / "images").exists()
    assert (chair.gs_source_dir / "sparse" / "0").exists()
