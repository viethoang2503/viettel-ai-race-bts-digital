# Core NVS Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the end-to-end baseline pipeline (data validation -> pose math -> train wrapper
-> render-from-CSV -> metrics -> submission packaging) so a fully compliant, correctly-formatted
`submission.zip` can be produced for all 7 scenes using plain 3D Gaussian Splatting, before any
advanced technique (floater cleanup, depth reg, anti-aliasing, appearance embedding) is added.

**Architecture:** Vendor `graphdeco-inria/gaussian-splatting` as a git submodule for the actual
GPU training/rendering code. All new code lives in `src/`, split by responsibility (common pose
math, data validation, evaluation, training wrapper, rendering, submission). GPU-free logic
(pose math, metric formula, holdout selection, submission packaging/validation, orchestration
wiring) gets real `pytest` unit tests runnable on the local no-GPU machine. GPU-dependent steps
(actual training, actual CUDA rendering) get a documented manual verification procedure to run
on Colab, since this local machine has no CUDA device.

**Tech Stack:** Python 3.10+ (see note in Task 1 about the local Python 3.14 interpreter),
PyTorch (CPU-only locally, CUDA on Colab), `lpips`, `scikit-image`, `numpy`, `pyyaml`, `pytest`.

## Global Constraints

- No ground-truth exists for the real `test_poses.csv` (private test) — all local
  correctness checks use the already-downloaded `VAI_NVS_DATA_ROUND2/` scenes as fixtures.
- Repo root is `/home/howard/Documents/viettel ai race/computer vision` (already a git repo,
  `.gitignore` already excludes `VAI_NVS_DATA_ROUND2/`, `outputs/`, `checkpoints/`).
- Dataset layout per scene (from spec section 1): `train/images/`, `train/sparse/0/{cameras.bin,
  images.bin,points3D.bin}`, `test/test_poses.csv` with columns
  `image_name,qw,qx,qy,qz,tx,ty,tz,fx,fy,cx,cy,width,height`.
- Pose convention: quaternion is COLMAP world-to-camera (`qw,qx,qy,qz`), translation is
  COLMAP world-to-camera (`tx,ty,tz`) — same convention as `images.bin`.
- Submission format (spec section 14 / exam mục 7): `submission.zip` containing
  `scene_XXX/<image_name>` per scene, exact name and `width x height` from `test_poses.csv`.
- This plan covers the **baseline-only** pipeline. Floater cleanup, depth regularization,
  anti-aliasing, appearance embedding, VRAM guard, and auto-config-selection are **out of
  scope** for this plan — they are a follow-up plan (`docs/superpowers/plans/<next>-advanced-
  techniques.md`) built on top of these modules.

---

### Task 1: Vendor baseline repo and lock the environment

**Files:**
- Create: `environment/requirements.txt`
- Create: `third_party/gaussian-splatting` (git submodule)
- Modify: `.gitignore` (add `.venv-check` marker if needed — verify existing entries still cover
  submodule build artifacts)

**Interfaces:**
- Produces: `third_party/gaussian-splatting/` containing the upstream repo (`scene/`,
  `gaussian_renderer/`, `utils/`, `train.py`, `render.py`) that later tasks import from via
  `sys.path`.

- [ ] **Step 1: Add the upstream repo as a git submodule**

```bash
cd "/home/howard/Documents/viettel ai race/computer vision"
git submodule add https://github.com/graphdeco-inria/gaussian-splatting.git third_party/gaussian-splatting
git submodule update --init --recursive
```

Expected: `third_party/gaussian-splatting/scene/colmap_loader.py` and
`third_party/gaussian-splatting/utils/graphics_utils.py` exist.

- [ ] **Step 2: Write `environment/requirements.txt`**

```
numpy>=1.24,<2.0
pillow>=10.0
pyyaml>=6.0
pandas>=2.0
scikit-image>=0.22
lpips>=0.1.4
torch>=2.1
torchvision>=0.16
tqdm>=4.66
pytest>=8.0
```

- [ ] **Step 3: Create a local virtualenv pinned to a torch-compatible Python and install**

The local interpreter (`python3 --version`) is 3.14, which is too new for prebuilt `torch`/
`lpips` wheels as of this writing. Use whichever `python3.10`/`python3.11` is available; if
none is installed, note this explicitly to the user rather than silently falling back to 3.14.

```bash
python3 -m venv .venv 2>&1 | tail -5
```

If this fails or `pip install -r environment/requirements.txt` inside it fails to find `torch`
wheels, stop and report: "Need Python 3.10 or 3.11 installed locally (e.g. via
`sudo apt install python3.11-venv` or pyenv) — 3.14 has no torch wheel yet." Do not proceed to
Step 4 until resolved.

```bash
source .venv/bin/activate
pip install -r environment/requirements.txt
```

Expected: exits 0, `python -c "import torch, lpips, skimage, yaml, pandas"` succeeds.

- [ ] **Step 4: Commit**

```bash
git add .gitmodules third_party environment/requirements.txt
git commit -m "Vendor gaussian-splatting baseline and pin environment deps"
```

---

### Task 2: Scene config file and loader

**Files:**
- Create: `configs/scenes.yaml`
- Create: `src/common/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `load_scenes(config_path: str = "configs/scenes.yaml") -> list[SceneConfig]` where
  `SceneConfig` is a dataclass with fields `name: str`, `root: Path`, `train_images_dir: Path`,
  `sparse_dir: Path`, `test_poses_csv: Path`.

- [ ] **Step 1: Write `configs/scenes.yaml`**

```yaml
dataset_root: VAI_NVS_DATA_ROUND2
scenes:
  - HCM0421
  - HCM0539
  - HCM0540
  - HCM0644
  - HCM0674
  - chair
  - bonsai
```

- [ ] **Step 2: Write the failing test**

```python
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
```

- [ ] **Step 3: Run test to verify it fails**

```bash
pytest tests/test_config.py -v
```

Expected: `FAIL` — `ModuleNotFoundError: No module named 'src.common.config'`.

- [ ] **Step 4: Write `src/common/config.py`**

```python
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


def load_scenes(config_path: str = "configs/scenes.yaml") -> list[SceneConfig]:
    with open(config_path, "r") as f:
        data = yaml.safe_load(f)

    dataset_root = Path(data["dataset_root"])
    scenes = []
    for name in data["scenes"]:
        root = dataset_root / name
        scenes.append(
            SceneConfig(
                name=name,
                root=root,
                train_images_dir=root / "train" / "images",
                sparse_dir=root / "train" / "sparse" / "0",
                test_poses_csv=root / "test" / "test_poses.csv",
            )
        )
    return scenes
```

Create `src/__init__.py`, `src/common/__init__.py`, `tests/__init__.py` (empty files) if not
already present.

- [ ] **Step 5: Run test to verify it passes**

```bash
pytest tests/test_config.py -v
```

Expected: `PASS` (1 passed).

- [ ] **Step 6: Commit**

```bash
git add configs/scenes.yaml src/common/config.py src/__init__.py src/common/__init__.py tests/test_config.py tests/__init__.py
git commit -m "Add scene config loader"
```

---

### Task 3: Pose math utilities (qvec2rotmat, focal2fov, camera params from CSV row)

This is the highest-risk-of-silent-bug piece per the design spec (mục 8/12) — a wrong
convention produces a plausible-looking but geometrically incorrect render with no crash.

**Files:**
- Create: `src/common/pose_utils.py`
- Test: `tests/test_pose_utils.py`

**Interfaces:**
- Produces:
  - `qvec2rotmat(qvec: np.ndarray) -> np.ndarray` (3x3)
  - `focal2fov(focal: float, pixels: int) -> float` (radians)
  - `camera_extrinsics_from_colmap(qw, qx, qy, qz, tx, ty, tz) -> tuple[np.ndarray, np.ndarray]`
    returning `(R, T)` in the convention the vendored `scene/cameras.py::Camera` class expects
    (`R` = transpose of the COLMAP world-to-camera rotation, `T` = COLMAP world-to-camera
    translation unchanged).
  - `CameraParams` dataclass: `image_name: str`, `R: np.ndarray`, `T: np.ndarray`,
    `fov_x: float`, `fov_y: float`, `width: int`, `height: int`.
  - `camera_params_from_csv_row(row: dict) -> CameraParams`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_pose_utils.py
import math

import numpy as np
import pytest

from src.common.pose_utils import (
    camera_extrinsics_from_colmap,
    camera_params_from_csv_row,
    focal2fov,
    qvec2rotmat,
)


def test_identity_quaternion_gives_identity_rotation():
    r = qvec2rotmat(np.array([1.0, 0.0, 0.0, 0.0]))
    np.testing.assert_allclose(r, np.eye(3), atol=1e-10)


def test_90_degree_z_rotation_quaternion():
    # 90 deg about Z: qw=cos(45deg), qz=sin(45deg)
    half = math.pi / 4
    qvec = np.array([math.cos(half), 0.0, 0.0, math.sin(half)])
    r = qvec2rotmat(qvec)
    expected = np.array([
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    np.testing.assert_allclose(r, expected, atol=1e-10)


def test_focal2fov_matches_known_value():
    # focal=1000, pixels=2000 -> 2*atan(1) = pi/2
    fov = focal2fov(1000.0, 2000)
    assert fov == pytest.approx(math.pi / 2, abs=1e-10)


def test_camera_extrinsics_from_colmap_is_R_transpose_T_unchanged():
    qw, qx, qy, qz = 1.0, 0.0, 0.0, 0.0
    tx, ty, tz = 1.0, 2.0, 3.0
    R, T = camera_extrinsics_from_colmap(qw, qx, qy, qz, tx, ty, tz)
    np.testing.assert_allclose(R, np.eye(3), atol=1e-10)
    np.testing.assert_allclose(T, np.array([1.0, 2.0, 3.0]), atol=1e-10)


def test_camera_params_from_csv_row_computes_fov_and_keeps_metadata():
    row = {
        "image_name": "frame_000025.jpg",
        "qw": 1.0, "qx": 0.0, "qy": 0.0, "qz": 0.0,
        "tx": 0.0, "ty": 0.0, "tz": 0.0,
        "fx": 1000.0, "fy": 1000.0, "cx": 500.0, "cy": 500.0,
        "width": 1000, "height": 1000,
    }
    params = camera_params_from_csv_row(row)
    assert params.image_name == "frame_000025.jpg"
    assert params.width == 1000
    assert params.height == 1000
    assert params.fov_x == pytest.approx(2 * math.atan(0.5), abs=1e-10)
    assert params.fov_y == pytest.approx(2 * math.atan(0.5), abs=1e-10)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_pose_utils.py -v
```

Expected: `FAIL` — `ModuleNotFoundError: No module named 'src.common.pose_utils'`.

- [ ] **Step 3: Write `src/common/pose_utils.py`**

```python
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


def qvec2rotmat(qvec: np.ndarray) -> np.ndarray:
    """COLMAP quaternion (qw, qx, qy, qz) -> 3x3 rotation matrix.

    Same convention as scene/colmap_loader.py in the vendored baseline repo.
    """
    qw, qx, qy, qz = qvec
    return np.array([
        [1 - 2 * qy**2 - 2 * qz**2, 2 * qx * qy - 2 * qw * qz, 2 * qz * qx + 2 * qw * qy],
        [2 * qx * qy + 2 * qw * qz, 1 - 2 * qx**2 - 2 * qz**2, 2 * qy * qz - 2 * qw * qx],
        [2 * qz * qx - 2 * qw * qy, 2 * qy * qz + 2 * qw * qx, 1 - 2 * qx**2 - 2 * qy**2],
    ])


def focal2fov(focal: float, pixels: int) -> float:
    """Pinhole focal length (pixels) -> field of view (radians)."""
    return 2 * math.atan(pixels / (2 * focal))


def camera_extrinsics_from_colmap(
    qw: float, qx: float, qy: float, qz: float, tx: float, ty: float, tz: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert COLMAP world-to-camera (qvec, tvec) into the (R, T) convention
    expected by the vendored gaussian-splatting `scene.cameras.Camera` class:
    R is the transpose of the COLMAP world-to-camera rotation, T is the
    COLMAP world-to-camera translation unchanged.
    """
    qvec = np.array([qw, qx, qy, qz], dtype=np.float64)
    r_world_to_cam = qvec2rotmat(qvec)
    r = np.transpose(r_world_to_cam)
    t = np.array([tx, ty, tz], dtype=np.float64)
    return r, t


@dataclass(frozen=True)
class CameraParams:
    image_name: str
    R: np.ndarray
    T: np.ndarray
    fov_x: float
    fov_y: float
    width: int
    height: int


def camera_params_from_csv_row(row: dict) -> CameraParams:
    """Build CameraParams from one parsed row of test_poses.csv.

    `row` values may be strings (if read via csv.DictReader) or already
    numeric (if read via pandas) — this function coerces explicitly.
    """
    r, t = camera_extrinsics_from_colmap(
        float(row["qw"]), float(row["qx"]), float(row["qy"]), float(row["qz"]),
        float(row["tx"]), float(row["ty"]), float(row["tz"]),
    )
    width = int(row["width"])
    height = int(row["height"])
    fov_x = focal2fov(float(row["fx"]), width)
    fov_y = focal2fov(float(row["fy"]), height)
    return CameraParams(
        image_name=str(row["image_name"]),
        R=r, T=t, fov_x=fov_x, fov_y=fov_y, width=width, height=height,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_pose_utils.py -v
```

Expected: `PASS` (5 passed).

- [ ] **Step 5: Commit**

```bash
git add src/common/pose_utils.py tests/test_pose_utils.py
git commit -m "Add COLMAP pose math utilities with unit tests"
```

---

### Task 4: COLMAP sparse-data I/O wrapper and scene bounding box

**Files:**
- Create: `src/common/colmap_io.py`
- Test: `tests/test_colmap_io.py`

**Interfaces:**
- Consumes: `third_party/gaussian-splatting/scene/colmap_loader.py` functions
  `read_extrinsics_binary`, `read_intrinsics_binary`, `read_points3D_binary` (pure Python/numpy,
  no CUDA — safe to import and run on the local machine).
- Produces:
  - `load_sparse_scene(sparse_dir: Path) -> SparseScene` where `SparseScene` has fields
    `cameras: dict`, `images: dict`, `points3d: dict`.
  - `compute_scene_bbox(points3d: dict, margin_ratio: float = 0.1) -> tuple[np.ndarray, np.ndarray]`
    returning `(min_xyz, max_xyz)`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_colmap_io.py
from pathlib import Path

import numpy as np

from src.common.colmap_io import compute_scene_bbox, load_sparse_scene

CHAIR_SPARSE = Path("VAI_NVS_DATA_ROUND2/chair/train/sparse/0")


def test_load_sparse_scene_reads_chair_scene():
    scene = load_sparse_scene(CHAIR_SPARSE)
    assert len(scene.images) == 205
    assert len(scene.cameras) >= 1
    assert len(scene.points3d) > 0


def test_compute_scene_bbox_returns_min_less_than_max_with_margin():
    scene = load_sparse_scene(CHAIR_SPARSE)
    min_xyz, max_xyz = compute_scene_bbox(scene.points3d, margin_ratio=0.1)
    assert min_xyz.shape == (3,)
    assert max_xyz.shape == (3,)
    assert np.all(min_xyz < max_xyz)
    assert np.all(np.isfinite(min_xyz))
    assert np.all(np.isfinite(max_xyz))


def test_compute_scene_bbox_margin_expands_tight_bbox():
    raw_points = {
        i: type("P", (), {"xyz": np.array([float(i), 0.0, 0.0])})()
        for i in range(3)
    }  # points at x=0,1,2
    tight_min, tight_max = compute_scene_bbox(raw_points, margin_ratio=0.0)
    wide_min, wide_max = compute_scene_bbox(raw_points, margin_ratio=0.5)
    assert wide_min[0] < tight_min[0]
    assert wide_max[0] > tight_max[0]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_colmap_io.py -v
```

Expected: `FAIL` — `ModuleNotFoundError: No module named 'src.common.colmap_io'`.

- [ ] **Step 3: Write `src/common/colmap_io.py`**

```python
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_VENDORED_REPO = Path(__file__).resolve().parents[2] / "third_party" / "gaussian-splatting"
if str(_VENDORED_REPO) not in sys.path:
    sys.path.insert(0, str(_VENDORED_REPO))

from scene.colmap_loader import (  # noqa: E402
    read_extrinsics_binary,
    read_intrinsics_binary,
    read_points3D_binary,
)


@dataclass(frozen=True)
class SparseScene:
    cameras: dict
    images: dict
    points3d: dict


def load_sparse_scene(sparse_dir: Path) -> SparseScene:
    sparse_dir = Path(sparse_dir)
    cameras = read_intrinsics_binary(str(sparse_dir / "cameras.bin"))
    images = read_extrinsics_binary(str(sparse_dir / "images.bin"))
    points3d = read_points3D_binary(str(sparse_dir / "points3D.bin"))
    return SparseScene(cameras=cameras, images=images, points3d=points3d)


def compute_scene_bbox(
    points3d: dict, margin_ratio: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    xyz = np.stack([p.xyz for p in points3d.values()], axis=0)
    min_xyz = xyz.min(axis=0)
    max_xyz = xyz.max(axis=0)
    extent = max_xyz - min_xyz
    margin = extent * margin_ratio
    return min_xyz - margin, max_xyz + margin
```

Note: `third_party/gaussian-splatting` must be on `sys.path` for `scene.colmap_loader` to
import — Task 1's submodule checkout provides this at the expected relative path.

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_colmap_io.py -v
```

Expected: `PASS` (3 passed). If `read_extrinsics_binary`/`read_points3D_binary` are missing or
renamed in the checked-out submodule version, open
`third_party/gaussian-splatting/scene/colmap_loader.py` and adjust the imported names to match
before re-running.

- [ ] **Step 5: Commit**

```bash
git add src/common/colmap_io.py tests/test_colmap_io.py
git commit -m "Add COLMAP sparse scene loader and bounding-box computation"
```

---

### Task 5: Data validation module

**Files:**
- Create: `src/data_validation/validate_scene.py`
- Test: `tests/test_validate_scene.py`

**Interfaces:**
- Consumes: `src.common.config.SceneConfig` (Task 2), `src.common.colmap_io.load_sparse_scene`
  (Task 4).
- Produces: `validate_scene(scene: SceneConfig) -> ValidationReport` where `ValidationReport`
  has fields `scene_name: str`, `registered_image_count: int`, `folder_image_count: int`,
  `missing_images: list[str]`, `extra_images: list[str]`, `test_pose_row_count: int`,
  `problems: list[str]` (empty means valid).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_validate_scene.py
from src.common.config import load_scenes
from src.data_validation.validate_scene import validate_scene


def _get_scene(name):
    return next(s for s in load_scenes("configs/scenes.yaml") if s.name == name)


def test_validate_scene_chair_has_no_problems():
    report = validate_scene(_get_scene("chair"))
    assert report.scene_name == "chair"
    assert report.registered_image_count == report.folder_image_count
    assert report.missing_images == []
    assert report.extra_images == []
    assert report.test_pose_row_count == 58
    assert report.problems == []


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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_validate_scene.py -v
```

Expected: `FAIL` — `ModuleNotFoundError: No module named 'src.data_validation'`.

- [ ] **Step 3: Write `src/data_validation/validate_scene.py`**

```python
from __future__ import annotations

import csv
from dataclasses import dataclass, field

from src.common.colmap_io import load_sparse_scene
from src.common.config import SceneConfig

REQUIRED_CSV_COLUMNS = {
    "image_name", "qw", "qx", "qy", "qz", "tx", "ty", "tz",
    "fx", "fy", "cx", "cy", "width", "height",
}


@dataclass
class ValidationReport:
    scene_name: str
    registered_image_count: int = 0
    folder_image_count: int = 0
    missing_images: list[str] = field(default_factory=list)
    extra_images: list[str] = field(default_factory=list)
    test_pose_row_count: int = 0
    problems: list[str] = field(default_factory=list)


def validate_scene(scene: SceneConfig) -> ValidationReport:
    report = ValidationReport(scene_name=scene.name)

    if not scene.sparse_dir.exists():
        report.problems.append(f"sparse dir not found: {scene.sparse_dir}")
        return report
    if not scene.train_images_dir.exists():
        report.problems.append(f"train images dir not found: {scene.train_images_dir}")
        return report

    sparse = load_sparse_scene(scene.sparse_dir)
    registered_names = {img.name for img in sparse.images.values()}
    folder_names = {p.name for p in scene.train_images_dir.iterdir() if p.is_file()}

    report.registered_image_count = len(registered_names)
    report.folder_image_count = len(folder_names)
    report.missing_images = sorted(registered_names - folder_names)
    report.extra_images = sorted(folder_names - registered_names)

    if report.missing_images:
        report.problems.append(
            f"{len(report.missing_images)} registered images missing from folder"
        )
    if report.extra_images:
        report.problems.append(
            f"{len(report.extra_images)} folder images not registered in images.bin"
        )

    if not scene.test_poses_csv.exists():
        report.problems.append(f"test_poses.csv not found: {scene.test_poses_csv}")
        return report

    with open(scene.test_poses_csv, newline="") as f:
        reader = csv.DictReader(f)
        missing_cols = REQUIRED_CSV_COLUMNS - set(reader.fieldnames or [])
        if missing_cols:
            report.problems.append(f"test_poses.csv missing columns: {sorted(missing_cols)}")
        rows = list(reader)
        report.test_pose_row_count = len(rows)

    return report
```

Create `src/data_validation/__init__.py` (empty).

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_validate_scene.py -v
```

Expected: `PASS` (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/data_validation/ tests/test_validate_scene.py
git commit -m "Add per-scene data validation module"
```

---

### Task 6: Score formula and pairwise image metrics

**Files:**
- Create: `src/evaluation/compute_metrics.py`
- Test: `tests/test_compute_metrics.py`

**Interfaces:**
- Produces:
  - `combine_score(lpips_val: float, ssim_val: float, psnr_val: float, psnr_max: float) -> float`
    — pure function implementing đề bài mục 8.4 exactly.
  - `compute_pair_metrics(pred: np.ndarray, gt: np.ndarray, lpips_model) -> dict` where `pred`,
    `gt` are `(H, W, 3)` `uint8` arrays and `lpips_model` is any callable with signature
    `lpips_model(pred_tensor, gt_tensor) -> torch.Tensor` (scalar) — real network wired in by
    the caller, a stub used in tests.
  - `load_lpips_model(net: str = "alex")` — thin factory around `lpips.LPIPS`, not unit tested
    beyond importability (real behavior verified manually, see Step 6).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_compute_metrics.py
import numpy as np
import pytest
import torch

from src.evaluation.compute_metrics import combine_score, compute_pair_metrics


def test_combine_score_matches_spec_formula():
    # Score = 0.4*(1-LPIPS) + 0.3*SSIM + 0.3*PSNR_norm
    score = combine_score(lpips_val=0.2, ssim_val=0.9, psnr_val=30.0, psnr_max=30.0)
    expected = 0.4 * (1 - 0.2) + 0.3 * 0.9 + 0.3 * 1.0
    assert score == pytest.approx(expected, abs=1e-10)


def test_combine_score_clamps_psnr_norm_above_max():
    score = combine_score(lpips_val=0.0, ssim_val=1.0, psnr_val=999.0, psnr_max=30.0)
    # psnr_norm clamped to 1.0, so score == 0.4*1 + 0.3*1 + 0.3*1
    assert score == pytest.approx(1.0, abs=1e-10)


def test_combine_score_clamps_psnr_norm_below_zero():
    score = combine_score(lpips_val=1.0, ssim_val=0.0, psnr_val=-10.0, psnr_max=30.0)
    assert score == pytest.approx(0.0, abs=1e-10)


class _StubLpipsModel:
    """Returns 0 distance for identical inputs, 1 otherwise."""

    def __call__(self, pred_tensor, gt_tensor):
        identical = torch.allclose(pred_tensor, gt_tensor)
        return torch.tensor(0.0 if identical else 1.0)


def test_compute_pair_metrics_identical_images_score_well():
    img = np.random.default_rng(0).integers(0, 255, size=(32, 32, 3), dtype=np.uint8)
    result = compute_pair_metrics(pred=img, gt=img, lpips_model=_StubLpipsModel())
    assert result["lpips"] == pytest.approx(0.0, abs=1e-6)
    assert result["ssim"] == pytest.approx(1.0, abs=1e-6)
    assert result["psnr"] > 40.0  # identical images -> very high/inf PSNR


def test_compute_pair_metrics_different_images_are_worse():
    rng = np.random.default_rng(0)
    pred = rng.integers(0, 255, size=(32, 32, 3), dtype=np.uint8)
    gt = rng.integers(0, 255, size=(32, 32, 3), dtype=np.uint8)
    result = compute_pair_metrics(pred=pred, gt=gt, lpips_model=_StubLpipsModel())
    assert result["lpips"] == pytest.approx(1.0, abs=1e-6)
    assert result["ssim"] < 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_compute_metrics.py -v
```

Expected: `FAIL` — `ModuleNotFoundError: No module named 'src.evaluation'`.

- [ ] **Step 3: Write `src/evaluation/compute_metrics.py`**

```python
from __future__ import annotations

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def combine_score(lpips_val: float, ssim_val: float, psnr_val: float, psnr_max: float) -> float:
    """Score = 0.4*(1-LPIPS) + 0.3*SSIM + 0.3*PSNR_norm, PSNR_norm clamped to [0,1]."""
    psnr_norm = max(0.0, min(1.0, psnr_val / psnr_max))
    return 0.4 * (1 - lpips_val) + 0.3 * ssim_val + 0.3 * psnr_norm


def _to_lpips_tensor(img: np.ndarray) -> torch.Tensor:
    # (H,W,3) uint8 [0,255] -> (1,3,H,W) float32 in [-1,1], as expected by lpips.LPIPS
    t = torch.from_numpy(img).float() / 127.5 - 1.0
    return t.permute(2, 0, 1).unsqueeze(0)


def compute_pair_metrics(pred: np.ndarray, gt: np.ndarray, lpips_model) -> dict:
    assert pred.shape == gt.shape, f"shape mismatch: {pred.shape} vs {gt.shape}"

    ssim_val = structural_similarity(pred, gt, channel_axis=2, data_range=255)

    mse = np.mean((pred.astype(np.float64) - gt.astype(np.float64)) ** 2)
    if mse == 0:
        psnr_val = 100.0  # treat identical images as a high finite ceiling, not inf
    else:
        psnr_val = peak_signal_noise_ratio(gt, pred, data_range=255)

    with torch.no_grad():
        lpips_val = float(lpips_model(_to_lpips_tensor(pred), _to_lpips_tensor(gt)))

    return {"lpips": lpips_val, "ssim": float(ssim_val), "psnr": float(psnr_val)}


def load_lpips_model(net: str = "alex"):
    import lpips

    model = lpips.LPIPS(net=net)
    model.eval()
    return model
```

Create `src/evaluation/__init__.py` (empty).

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_compute_metrics.py -v
```

Expected: `PASS` (5 passed).

- [ ] **Step 5: Commit**

```bash
git add src/evaluation/compute_metrics.py src/evaluation/__init__.py tests/test_compute_metrics.py
git commit -m "Add Score formula and pairwise LPIPS/SSIM/PSNR metrics"
```

- [ ] **Step 6: Manual verification of the real LPIPS network (requires network access)**

```bash
python -c "
from src.evaluation.compute_metrics import load_lpips_model, compute_pair_metrics
import numpy as np
model = load_lpips_model('alex')
img = np.random.default_rng(0).integers(0,255,size=(64,64,3),dtype=np.uint8)
print(compute_pair_metrics(img, img, model))
"
```

Expected: prints a dict with `lpips` close to `0.0`, `ssim` close to `1.0`, `psnr` == `100.0`.
This downloads AlexNet LPIPS weights on first run — requires network access, which is why it is
not part of the automated `pytest` suite.

---

### Task 7: Holdout split at the edge of camera coverage

**Files:**
- Create: `src/evaluation/make_holdout_split.py`
- Test: `tests/test_make_holdout_split.py`

**Interfaces:**
- Produces: `select_holdout_images(camera_centers: dict[str, np.ndarray], holdout_ratio: float = 0.125) -> list[str]`
  returning the image names whose camera center is farthest from the centroid of all camera
  centers (approximates test poses that extrapolate beyond the dense training trajectory, per
  spec section 10).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_make_holdout_split.py
import numpy as np

from src.evaluation.make_holdout_split import select_holdout_images


def test_select_holdout_picks_farthest_points_from_centroid():
    centers = {f"cluster_{i}": np.array([0.01 * i, 0.0, 0.0]) for i in range(10)}
    centers["outlier_1"] = np.array([100.0, 0.0, 0.0])
    centers["outlier_2"] = np.array([-100.0, 0.0, 0.0])

    holdout = select_holdout_images(centers, holdout_ratio=0.2)  # 12 * 0.2 -> 2 (ceil)

    assert set(holdout) == {"outlier_1", "outlier_2"}


def test_select_holdout_ratio_controls_count():
    centers = {f"img_{i}": np.array([float(i), 0.0, 0.0]) for i in range(20)}
    holdout = select_holdout_images(centers, holdout_ratio=0.5)
    assert len(holdout) == 10


def test_select_holdout_never_returns_empty_for_nonzero_ratio():
    centers = {"only_one": np.array([0.0, 0.0, 0.0])}
    holdout = select_holdout_images(centers, holdout_ratio=0.125)
    assert holdout == ["only_one"]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_make_holdout_split.py -v
```

Expected: `FAIL` — `ModuleNotFoundError: No module named 'src.evaluation.make_holdout_split'`.

- [ ] **Step 3: Write `src/evaluation/make_holdout_split.py`**

```python
from __future__ import annotations

import math

import numpy as np


def select_holdout_images(
    camera_centers: dict[str, np.ndarray], holdout_ratio: float = 0.125,
) -> list[str]:
    """Pick the images whose camera center is farthest from the centroid of
    all camera centers in the scene — approximates the edge-of-coverage
    poses that real test poses are likely to extrapolate toward, rather
    than a uniform every-Nth split.
    """
    names = list(camera_centers.keys())
    centers = np.stack([camera_centers[n] for n in names], axis=0)
    centroid = centers.mean(axis=0)
    distances = np.linalg.norm(centers - centroid, axis=1)

    n_holdout = max(1, math.ceil(len(names) * holdout_ratio))
    order = np.argsort(-distances)  # descending distance
    return [names[i] for i in order[:n_holdout]]
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_make_holdout_split.py -v
```

Expected: `PASS` (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/evaluation/make_holdout_split.py tests/test_make_holdout_split.py
git commit -m "Add edge-of-coverage holdout selection for local validation"
```

---

### Task 8: Training wrapper (resume-safe CLI around vendored train.py)

Actual training requires a CUDA GPU (compiled `diff-gaussian-rasterization`/`simple-knn`
extensions), which this local machine does not have. This task splits cleanly: the
resume/command-building logic is pure Python and gets real tests; the actual `train.py`
invocation is a documented manual step for Colab.

**Files:**
- Create: `src/training/train_wrapper.py`
- Test: `tests/test_train_wrapper.py`

**Interfaces:**
- Produces:
  - `find_latest_checkpoint(output_dir: Path) -> Path | None` — finds the highest-iteration
    `chkpntNNNN.pth` file in `output_dir`, or `None` if none exist.
  - `build_train_argv(scene: SceneConfig, output_dir: Path, iterations: int, resume_checkpoint: Path | None, extra_args: list[str] | None = None) -> list[str]`
    — returns the argv list for invoking the vendored `train.py` (does not execute it).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_train_wrapper.py
from pathlib import Path

from src.common.config import SceneConfig
from src.training.train_wrapper import build_train_argv, find_latest_checkpoint


def test_find_latest_checkpoint_returns_none_when_empty(tmp_path):
    assert find_latest_checkpoint(tmp_path) is None


def test_find_latest_checkpoint_picks_highest_iteration(tmp_path):
    (tmp_path / "chkpnt7000.pth").touch()
    (tmp_path / "chkpnt30000.pth").touch()
    (tmp_path / "chkpnt15000.pth").touch()
    result = find_latest_checkpoint(tmp_path)
    assert result == tmp_path / "chkpnt30000.pth"


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
    assert str(scene.root) == argv[argv.index("--source_path") + 1]
    assert "--model_path" in argv
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
    assert str(ckpt) == argv[argv.index("--start_checkpoint") + 1]
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_train_wrapper.py -v
```

Expected: `FAIL` — `ModuleNotFoundError: No module named 'src.training'`.

- [ ] **Step 3: Write `src/training/train_wrapper.py`**

```python
from __future__ import annotations

import re
from pathlib import Path

from src.common.config import SceneConfig

_CHECKPOINT_RE = re.compile(r"chkpnt(\d+)\.pth$")


def find_latest_checkpoint(output_dir: Path) -> Path | None:
    output_dir = Path(output_dir)
    if not output_dir.exists():
        return None
    candidates = []
    for p in output_dir.glob("chkpnt*.pth"):
        match = _CHECKPOINT_RE.search(p.name)
        if match:
            candidates.append((int(match.group(1)), p))
    if not candidates:
        return None
    candidates.sort(key=lambda pair: pair[0])
    return candidates[-1][1]


def build_train_argv(
    scene: SceneConfig,
    output_dir: Path,
    iterations: int,
    resume_checkpoint: Path | None,
    extra_args: list[str] | None = None,
) -> list[str]:
    argv = [
        "python", "train.py",
        "--source_path", str(scene.root),
        "--model_path", str(output_dir),
        "--iterations", str(iterations),
        "--eval",
    ]
    if resume_checkpoint is not None:
        argv += ["--start_checkpoint", str(resume_checkpoint)]
    if extra_args:
        argv += extra_args
    return argv
```

Create `src/training/__init__.py` (empty).

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_train_wrapper.py -v
```

Expected: `PASS` (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/training/ tests/test_train_wrapper.py
git commit -m "Add resume-safe training command builder"
```

- [ ] **Step 6: Manual verification on Colab (requires GPU, cannot run locally)**

On a Colab Pro/Pro+ GPU runtime, after Task 13's `setup_colab.sh` has run:

```python
from pathlib import Path
from src.common.config import load_scenes
from src.training.train_wrapper import build_train_argv, find_latest_checkpoint
import subprocess

scene = next(s for s in load_scenes("configs/scenes.yaml") if s.name == "chair")
output_dir = Path("/content/drive/MyDrive/var2026/outputs/chair/baseline")
ckpt = find_latest_checkpoint(output_dir)
argv = build_train_argv(scene, output_dir, iterations=30000, resume_checkpoint=ckpt)
subprocess.run(argv, cwd="third_party/gaussian-splatting", check=True)
```

Expected: training starts (or resumes from `ckpt` if the cell is re-run after a disconnect),
writes `chkpntNNNN.pth` files into `output_dir` on Drive, and logs loss decreasing over
iterations.

---

### Task 9: Render-from-CSV module

**Files:**
- Create: `src/rendering/render_from_csv.py`
- Test: `tests/test_render_from_csv.py`

**Interfaces:**
- Consumes: `src.common.pose_utils.camera_params_from_csv_row` (Task 3).
- Produces:
  - `load_test_poses_csv(csv_path: Path) -> list[CameraParams]`
  - `render_all(checkpoint_ply: Path, csv_path: Path, output_dir: Path, render_fn) -> list[Path]`
    where `render_fn(camera_params: CameraParams, gaussians) -> np.ndarray` is injected so the
    orchestration/file-naming logic is testable without a GPU; the real `render_fn` (Task 9
    Step 6) wraps the vendored `gaussian_renderer.render`.

- [ ] **Step 1: Write the failing tests**

```python
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
        assert path.name == params.image_name.rsplit(".", 1)[0] + ".png"
        assert path.exists()
        from PIL import Image
        img = Image.open(path)
        assert img.size == (params.width, params.height)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_render_from_csv.py -v
```

Expected: `FAIL` — `ModuleNotFoundError: No module named 'src.rendering'`.

- [ ] **Step 3: Write `src/rendering/render_from_csv.py`**

```python
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image

from src.common.pose_utils import CameraParams, camera_params_from_csv_row


def load_test_poses_csv(csv_path: Path) -> list[CameraParams]:
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        return [camera_params_from_csv_row(row) for row in reader]


def render_all(
    checkpoint_ply,
    csv_path,
    output_dir: Path,
    render_fn,
    params_list: list[CameraParams] | None = None,
    gaussians=None,
) -> list[Path]:
    """Render every camera in params_list (or loaded from csv_path if
    params_list is None) and write one PNG per row into output_dir, named
    after the CSV's image_name (extension normalized to .png per the
    submission format in spec section 14).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if params_list is None:
        params_list = load_test_poses_csv(csv_path)

    written = []
    for params in params_list:
        img_array = render_fn(params, gaussians)
        assert img_array.shape == (params.height, params.width, 3), (
            f"{params.image_name}: expected {(params.height, params.width, 3)}, "
            f"got {img_array.shape}"
        )
        out_name = params.image_name.rsplit(".", 1)[0] + ".png"
        out_path = output_dir / out_name
        Image.fromarray(img_array.astype(np.uint8)).save(out_path)
        written.append(out_path)
    return written
```

Create `src/rendering/__init__.py` (empty).

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_render_from_csv.py -v
```

Expected: `PASS` (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/rendering/ tests/test_render_from_csv.py
git commit -m "Add render-from-CSV module with injectable render function"
```

- [ ] **Step 6: Manual verification on Colab (requires GPU, cannot run locally)**

```python
import torch
from src.rendering.render_from_csv import load_test_poses_csv, render_all
from src.common.config import load_scenes
import sys
sys.path.insert(0, "third_party/gaussian-splatting")
from scene.gaussian_model import GaussianModel
from gaussian_renderer import render as gs_render
from scene.cameras import Camera

scene = next(s for s in load_scenes("configs/scenes.yaml") if s.name == "chair")
gaussians = GaussianModel(3)
gaussians.load_ply("/content/drive/MyDrive/var2026/outputs/chair/baseline/point_cloud/iteration_30000/point_cloud.ply")

def real_render_fn(camera_params, gaussians):
    cam = Camera(
        colmap_id=0, R=camera_params.R, T=camera_params.T,
        FoVx=camera_params.fov_x, FoVy=camera_params.fov_y,
        image=torch.zeros(3, camera_params.height, camera_params.width),
        gt_alpha_mask=None, image_name=camera_params.image_name, uid=0,
    )
    out = gs_render(cam, gaussians, pipe=..., bg_color=torch.tensor([0., 0., 0.]))["render"]
    return (out.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")

render_all(
    checkpoint_ply=None, csv_path=scene.test_poses_csv,
    output_dir="/content/drive/MyDrive/var2026/outputs/chair/baseline/test_render",
    render_fn=real_render_fn, gaussians=gaussians,
)
```

Expected: 58 PNG files written, filenames matching `test_poses.csv` `image_name` (with `.png`
extension), sizes matching each row's `width x height`. The `pipe=...` argument must be filled
with the vendored repo's `PipelineParams` default instance — check
`third_party/gaussian-splatting/arguments/__init__.py` for the exact constructor signature
before running, since it may differ slightly across submodule versions.

---

### Task 10: Submission packaging

**Files:**
- Create: `src/submission/package_submission.py`
- Test: `tests/test_package_submission.py`

**Interfaces:**
- Produces: `package_submission(scene_render_dirs: dict[str, Path], output_zip: Path) -> Path`
  — `scene_render_dirs` maps scene name -> directory of rendered PNGs; writes
  `scene_XXX/<image_name>` entries into `output_zip` per spec section 14 / đề bài mục 7.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_package_submission.py
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_package_submission.py -v
```

Expected: `FAIL` — `ModuleNotFoundError: No module named 'src.submission'`.

- [ ] **Step 3: Write `src/submission/package_submission.py`**

```python
from __future__ import annotations

import zipfile
from pathlib import Path


def package_submission(scene_render_dirs: dict[str, Path], output_zip: Path) -> Path:
    output_zip = Path(output_zip)
    output_zip.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for scene_name, render_dir in scene_render_dirs.items():
            render_dir = Path(render_dir)
            for image_path in sorted(render_dir.iterdir()):
                if not image_path.is_file():
                    continue
                arcname = f"{scene_name}/{image_path.name}"
                zf.write(image_path, arcname=arcname)

    return output_zip
```

Create `src/submission/__init__.py` (empty).

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_package_submission.py -v
```

Expected: `PASS` (1 passed).

- [ ] **Step 5: Commit**

```bash
git add src/submission/package_submission.py src/submission/__init__.py tests/test_package_submission.py
git commit -m "Add submission zip packaging"
```

---

### Task 11: Submission validation (safety net against the "missing scene zeroes the score" rule)

**Files:**
- Create: `src/submission/validate_submission.py`
- Test: `tests/test_validate_submission.py`

**Interfaces:**
- Consumes: `src.rendering.render_from_csv.load_test_poses_csv` (Task 9),
  `src.common.config.SceneConfig` (Task 2).
- Produces: `validate_submission(zip_path: Path, scenes: list[SceneConfig]) -> list[str]`
  returning a list of problem strings (empty list = valid, ready to submit).

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_validate_submission.py -v
```

Expected: `FAIL` — `ModuleNotFoundError: No module named 'src.submission.validate_submission'`.

- [ ] **Step 3: Write `src/submission/validate_submission.py`**

```python
from __future__ import annotations

import zipfile
from io import BytesIO
from pathlib import Path

from PIL import Image

from src.common.config import SceneConfig
from src.rendering.render_from_csv import load_test_poses_csv


def validate_submission(zip_path: Path, scenes: list[SceneConfig]) -> list[str]:
    problems: list[str] = []
    zip_path = Path(zip_path)

    with zipfile.ZipFile(zip_path) as zf:
        names_in_zip = set(zf.namelist())

        for scene in scenes:
            expected_params = load_test_poses_csv(scene.test_poses_csv)
            scene_entries = [n for n in names_in_zip if n.startswith(f"{scene.name}/")]
            if not scene_entries:
                problems.append(f"scene '{scene.name}': no files found in zip")
                continue

            for params in expected_params:
                out_name = params.image_name.rsplit(".", 1)[0] + ".png"
                arcname = f"{scene.name}/{out_name}"
                if arcname not in names_in_zip:
                    problems.append(f"scene '{scene.name}': missing {out_name}")
                    continue
                data = zf.read(arcname)
                with Image.open(BytesIO(data)) as img:
                    if img.size != (params.width, params.height):
                        problems.append(
                            f"scene '{scene.name}': {out_name} has wrong size "
                            f"{img.size}, expected {(params.width, params.height)}"
                        )

    return problems
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_validate_submission.py -v
```

Expected: `PASS` (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/submission/validate_submission.py tests/test_validate_submission.py
git commit -m "Add submission validator to prevent missing/mis-sized scene rejection"
```

---

### Task 12: Baseline orchestrator (wiring, GPU calls injected)

**Files:**
- Create: `src/orchestrator/run_pipeline.py`
- Test: `tests/test_run_pipeline.py`

**Interfaces:**
- Consumes: `validate_scene` (Task 5), `select_holdout_images` (Task 7), `compute_pair_metrics`
  + `combine_score` (Task 6), `package_submission` (Task 10), `validate_submission` (Task 11).
- Produces: `run_baseline_pipeline(scenes: list[SceneConfig], train_fn, render_fn, psnr_max: float, output_root: Path) -> PipelineResult`
  where `train_fn(scene, output_dir) -> Path` (returns checkpoint path) and
  `render_fn(checkpoint, params_list, output_dir) -> list[Path]` are injected, so the ordering
  and error-aggregation logic is testable without a GPU. `PipelineResult` has fields
  `per_scene_scores: dict[str, float]`, `validation_problems: list[str]`,
  `submission_zip: Path | None`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_run_pipeline.py
from pathlib import Path

import numpy as np

from src.common.config import SceneConfig
from src.orchestrator.run_pipeline import run_baseline_pipeline


def _chair_scene():
    return SceneConfig(
        name="chair",
        root=Path("VAI_NVS_DATA_ROUND2/chair"),
        train_images_dir=Path("VAI_NVS_DATA_ROUND2/chair/train/images"),
        sparse_dir=Path("VAI_NVS_DATA_ROUND2/chair/train/sparse/0"),
        test_poses_csv=Path("VAI_NVS_DATA_ROUND2/chair/test/test_poses.csv"),
    )


def test_run_baseline_pipeline_produces_scores_and_valid_zip(tmp_path):
    scene = _chair_scene()

    def fake_train_fn(scene, output_dir):
        return output_dir / "fake_checkpoint.pth"

    def fake_render_fn(checkpoint, params_list, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        written = []
        for params in params_list:
            from PIL import Image
            path = output_dir / (params.image_name.rsplit(".", 1)[0] + ".png")
            Image.fromarray(
                np.zeros((params.height, params.width, 3), dtype=np.uint8)
            ).save(path)
            written.append(path)
        return written

    result = run_baseline_pipeline(
        scenes=[scene],
        train_fn=fake_train_fn,
        render_fn=fake_render_fn,
        psnr_max=30.0,
        output_root=tmp_path,
    )

    assert "chair" in result.per_scene_scores
    assert 0.0 <= result.per_scene_scores["chair"] <= 1.0
    assert result.submission_zip is not None
    assert result.submission_zip.exists()
    # black-image render vs real holdout images should not be a perfect score
    assert result.per_scene_scores["chair"] < 0.9
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_run_pipeline.py -v
```

Expected: `FAIL` — `ModuleNotFoundError: No module named 'src.orchestrator'`.

- [ ] **Step 3: Write `src/orchestrator/run_pipeline.py`**

```python
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from src.common.colmap_io import load_sparse_scene
from src.common.config import SceneConfig
from src.evaluation.compute_metrics import combine_score, compute_pair_metrics, load_lpips_model
from src.evaluation.make_holdout_split import select_holdout_images
from src.rendering.render_from_csv import CameraParams
from src.submission.package_submission import package_submission
from src.submission.validate_submission import validate_submission


@dataclass
class PipelineResult:
    per_scene_scores: dict[str, float] = field(default_factory=dict)
    validation_problems: list[str] = field(default_factory=list)
    submission_zip: Path | None = None


def _camera_params_for_holdout(sparse, holdout_names, image_dims) -> list[CameraParams]:
    from src.common.pose_utils import camera_extrinsics_from_colmap, focal2fov

    width, height = image_dims
    params = []
    id_to_camera = sparse.cameras
    for img in sparse.images.values():
        if img.name not in holdout_names:
            continue
        camera = id_to_camera[img.camera_id]
        fx, fy = camera.params[0], camera.params[1]
        r, t = camera_extrinsics_from_colmap(*img.qvec, *img.tvec)
        params.append(CameraParams(
            image_name=img.name, R=r, T=t,
            fov_x=focal2fov(fx, width), fov_y=focal2fov(fy, height),
            width=width, height=height,
        ))
    return params


def run_baseline_pipeline(
    scenes: list[SceneConfig], train_fn, render_fn, psnr_max: float, output_root: Path,
) -> PipelineResult:
    output_root = Path(output_root)
    result = PipelineResult()
    lpips_model = load_lpips_model()
    scene_render_dirs = {}

    for scene in scenes:
        scene_output = output_root / scene.name
        checkpoint = train_fn(scene, scene_output / "baseline")

        sparse = load_sparse_scene(scene.sparse_dir)
        camera_centers = {
            img.name: -np.transpose(_qvec_rotmat(img.qvec)) @ np.array(img.tvec)
            for img in sparse.images.values()
        }
        holdout_names = set(select_holdout_images(camera_centers, holdout_ratio=0.125))

        sample_image = next(scene.train_images_dir.iterdir())
        with Image.open(sample_image) as im:
            image_dims = im.size  # (width, height)

        holdout_params = _camera_params_for_holdout(sparse, holdout_names, image_dims)
        holdout_render_dir = scene_output / "holdout_render"
        rendered_paths = render_fn(checkpoint, holdout_params, holdout_render_dir)

        scores = []
        for path, params in zip(rendered_paths, holdout_params):
            gt_path = scene.train_images_dir / params.image_name
            pred = np.array(Image.open(path).convert("RGB"))
            gt = np.array(Image.open(gt_path).convert("RGB").resize(pred.shape[1::-1]))
            metrics = compute_pair_metrics(pred, gt, lpips_model)
            scores.append(combine_score(
                metrics["lpips"], metrics["ssim"], metrics["psnr"], psnr_max,
            ))
        result.per_scene_scores[scene.name] = float(np.mean(scores)) if scores else 0.0

        test_params = load_sparse_scene  # placeholder avoided below
        from src.rendering.render_from_csv import load_test_poses_csv
        test_render_dir = scene_output / "test_render"
        test_params_list = load_test_poses_csv(scene.test_poses_csv)
        render_fn(checkpoint, test_params_list, test_render_dir)
        scene_render_dirs[scene.name] = test_render_dir

    submission_zip = output_root / "submission.zip"
    package_submission(scene_render_dirs, submission_zip)
    result.validation_problems = validate_submission(submission_zip, scenes)
    result.submission_zip = submission_zip
    return result


def _qvec_rotmat(qvec):
    from src.common.pose_utils import qvec2rotmat
    import numpy as np
    return qvec2rotmat(np.array(qvec))
```

Create `src/orchestrator/__init__.py` (empty).

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_run_pipeline.py -v
```

Expected: `PASS` (1 passed). If it fails on the LPIPS network download (no network access in
the test sandbox), mark the test `@pytest.mark.network` and skip it in offline environments;
document this clearly in the test file's docstring rather than silently weakening the
assertion.

- [ ] **Step 5: Commit**

```bash
git add src/orchestrator/ tests/test_run_pipeline.py
git commit -m "Add baseline orchestrator wiring validation, holdout eval, and packaging"
```

---

### Task 13: Colab environment setup script

**Files:**
- Create: `environment/setup_colab.sh`

No automated test is possible locally (no CUDA device, no Colab runtime). This task produces
the script plus a manual verification checklist.

- [ ] **Step 1: Write `environment/setup_colab.sh`**

```bash
#!/usr/bin/env bash
set -euo pipefail

DRIVE_ROOT="/content/drive/MyDrive/var2026"
CUDA_EXT_CACHE="$DRIVE_ROOT/cuda_ext_cache"
REPO_DIR="$(pwd)"

echo "== Mounting Google Drive =="
python3 -c "
from google.colab import drive
drive.mount('/content/drive')
"

mkdir -p "$CUDA_EXT_CACHE"

echo "== Installing Python dependencies =="
pip install -q -r environment/requirements.txt

echo "== Checking out submodule =="
git submodule update --init --recursive

SITE_PACKAGES="$(python3 -c 'import site; print(site.getsitepackages()[0])')"

restore_or_build () {
  local ext_name="$1"
  local ext_src_dir="third_party/gaussian-splatting/submodules/$ext_name"
  local cache_marker="$CUDA_EXT_CACHE/$ext_name.built"

  if [ -f "$cache_marker" ]; then
    echo "== Restoring cached $ext_name build =="
    pip install -q "$CUDA_EXT_CACHE/$ext_name"
  else
    echo "== Building $ext_name from source (first run, slow) =="
    pip install -q "$ext_src_dir"
    pip download --no-deps --no-binary :all: -d "$CUDA_EXT_CACHE/$ext_name" "$ext_src_dir" || true
    touch "$cache_marker"
  fi
}

restore_or_build diff-gaussian-rasterization
restore_or_build simple-knn

echo "== Setup complete =="
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

- [ ] **Step 2: Manual verification checklist (run on a fresh Colab GPU runtime)**

1. Upload/clone this repo's GitHub remote into the Colab runtime:
   `!git clone --recurse-submodules <repo-url> && cd <repo-dir>`
2. Run `!bash environment/setup_colab.sh`.
3. Confirm output ends with `CUDA available: True`.
4. Disconnect the runtime, reconnect, re-run the script — confirm the second run skips the
   "Building ... from source" lines for both extensions (cache restore path taken) and
   completes in well under a minute for the extension-restore step.
5. If `pip download --no-deps --no-binary :all:` does not actually capture a rebuildable
   package for these two submodules (their `setup.py` may not support sdist packaging
   directly), replace that caching approach with copying the built `.so`/`.egg-link` files
   from `SITE_PACKAGES` into `$CUDA_EXT_CACHE` and restoring by copying them back — verify
   which approach actually works empirically on the first real Colab run, since this cannot
   be validated without a live CUDA-enabled Colab session, and update this script accordingly.

- [ ] **Step 3: Commit**

```bash
git add environment/setup_colab.sh
git commit -m "Add Colab environment setup script with CUDA extension caching"
```

---

## Self-Review Summary

- **Spec coverage:** Task 1 covers spec section 3/4 (repo scaffold, environment). Task 2
  covers section 3 (`configs/scenes.yaml`). Tasks 3-4 cover the pose-math correctness risk
  called out in spec sections 8/12. Task 5 covers section 5. Task 6 covers section 10's metric
  formula. Task 7 covers section 10's holdout methodology. Task 8 covers section 4 (train
  wrapper, resume-safe per section 4). Task 9 covers section 8. Tasks 10-11 cover sections 14
  and the completeness risk in section 8.4/14. Task 12 covers section 11 (orchestrator),
  baseline-only. Task 13 covers section 4 (Colab setup, CUDA caching). Spec sections 6, 7, 9,
  13, 15 (experiment matrix, VRAM guard, auto-select-best-config, visual QA, reproducibility
  bundle) are explicitly deferred to the follow-up "advanced techniques" plan, as stated in
  Global Constraints.
- **Placeholder scan:** no TBD/TODO remain; Task 13 Step 2 item 5 documents a genuine
  empirical unknown (whether `pip download --no-binary` round-trips these two submodules)
  rather than hiding it — this is flagged as a manual verification item, not left as an
  unimplemented stub.
- **Type consistency:** `SceneConfig` (Task 2) fields are used identically across Tasks 5, 8,
  9, 11, 12. `CameraParams` (Task 3) fields (`image_name, R, T, fov_x, fov_y, width, height`)
  are used identically in Tasks 9 and 12. `combine_score`/`compute_pair_metrics` signatures
  from Task 6 match their usage in Task 12.
