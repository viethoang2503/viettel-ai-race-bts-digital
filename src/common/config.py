from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class SceneConfig:
    name: str
    root: Path
    train_images_dir: Path
    sparse_dir: Path
    test_poses_csv: Path
    submission_dir: str = ""

    @property
    def effective_submission_dir(self) -> str:
        """Folder name to use inside submission.zip for this scene.

        Falls back to `name` when `submission_dir` is not set, so existing
        code/tests that construct SceneConfig without the new field keep
        working unchanged.
        """
        return self.submission_dir or self.name

    @property
    def gs_source_dir(self) -> Path:
        """Directory to pass as gaussian-splatting's --source_path.

        The vendored baseline expects `<source_path>/images/` and
        `<source_path>/sparse/0/` as DIRECT children. The real dataset
        nests these one level deeper (`<scene>/train/images/`,
        `<scene>/train/sparse/0/`), so `scene.root` itself is NOT a valid
        --source_path — passing it silently points train.py at a directory
        with no `images/`, which fails immediately on Colab.

        Always derived from `train_images_dir` (never a separately-set
        field) so it cannot drift out of sync: `train_images_dir.parent`
        is `<root>/train` for the raw dataset layout, and is `<root>`
        itself for a Task 8b filtered scene copy (which is already built
        flat, with `images/` and `sparse/0/` directly under its root) —
        both cases resolve correctly without any special-casing.
        """
        return self.train_images_dir.parent


def load_scenes(config_path: str = "configs/scenes.yaml") -> list[SceneConfig]:
    with open(config_path, "r") as f:
        data = yaml.safe_load(f)

    dataset_root = Path(data["dataset_root"])
    scenes = []
    for entry in data["scenes"]:
        name = entry["name"]
        root = dataset_root / name
        scenes.append(
            SceneConfig(
                name=name,
                root=root,
                train_images_dir=root / "train" / "images",
                sparse_dir=root / "train" / "sparse" / "0",
                test_poses_csv=root / "test" / "test_poses.csv",
                submission_dir=entry.get("submission_dir", "") or name,
            )
        )
    return scenes
