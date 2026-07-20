# Core NVS Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the end-to-end baseline pipeline (data validation -> pose math -> leak-free
holdout training -> render-from-CSV -> metrics -> submission packaging) so a fully compliant,
correctly-formatted `submission.zip` can be produced for all 7 scenes using plain 3D Gaussian
Splatting, before any advanced technique (floater cleanup, depth reg, anti-aliasing, appearance
embedding) is added.

**Architecture:** Vendor `graphdeco-inria/gaussian-splatting` as a git submodule for the actual
GPU training/rendering code. All new code lives in `src/`, split by responsibility (common pose
math, data validation, evaluation, training wrapper, rendering, submission). GPU-free logic
(pose math, metric formula, holdout selection/enforcement, submission packaging/validation,
orchestration wiring) gets real `pytest` unit tests runnable on the local no-GPU machine.
GPU-dependent steps (actual training, actual CUDA rendering) get a documented manual
verification procedure to run on Colab, since this local machine has no CUDA device. Each scene
is trained twice: once on a holdout-excluded copy of the scene (Task 8b) purely to produce an
unbiased local score, and once on the full scene to produce the checkpoint that is actually
rendered and submitted — the two are never conflated. Output filenames always match
`test_poses.csv`'s `image_name` exactly, including its original extension.

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
  `<submission_dir>/<image_name>` per scene, using the **exact `image_name` string from
  `test_poses.csv` including its original extension** (never renamed to `.png`), and exact
  `width x height` from `test_poses.csv`. `<submission_dir>` is a per-scene config value (see
  Task 2) — the exam's example illustration uses a generic `scene_001/scene_002` pattern, but
  the real dataset's scene identifiers are `HCM0421`, `chair`, `bonsai`, etc. This plan defaults
  `submission_dir` to the scene's real name; **confirm with the organizers before the real
  submission whether literal `scene_001`-style numbering is required instead** — this cannot be
  resolved from the provided data alone.
- This plan covers the **baseline-only** pipeline. Floater cleanup, depth regularization,
  anti-aliasing, appearance embedding, VRAM guard, and auto-config-selection are **out of
  scope** for this plan — they are covered by the follow-up plan
  `docs/superpowers/plans/2026-07-18-advanced-techniques.md`, built on top of these modules.

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
  `sparse_dir: Path`, `test_poses_csv: Path`, `submission_dir: str = ""` (defaults to `name`
  when empty — see `effective_submission_dir` property below), plus a read-only property
  `effective_submission_dir -> str` returning `submission_dir or name`.

- [ ] **Step 1: Write `configs/scenes.yaml`**

`submission_dir` is listed explicitly per scene (even though it currently equals `name`) so the
folder-naming convention used in the final `submission.zip` (Task 10/11/12) is a single,
easy-to-edit config value rather than an assumption buried in code — see the Global Constraints
note above about confirming this with the organizers.

```yaml
dataset_root: VAI_NVS_DATA_ROUND2
scenes:
  - name: HCM0421
    submission_dir: HCM0421
  - name: HCM0539
    submission_dir: HCM0539
  - name: HCM0540
    submission_dir: HCM0540
  - name: HCM0644
    submission_dir: HCM0644
  - name: HCM0674
    submission_dir: HCM0674
  - name: chair
    submission_dir: chair
  - name: bonsai
    submission_dir: bonsai
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
    assert chair.submission_dir == "chair"
    assert chair.effective_submission_dir == "chair"
    # gs_source_dir must be the directory that directly contains images/
    # and sparse/0/ as siblings, per the baseline's expected layout —
    # NOT chair.root, which is one level too shallow for this dataset.
    assert chair.gs_source_dir == Path("VAI_NVS_DATA_ROUND2/chair/train")
    assert (chair.gs_source_dir / "images").exists()
    assert (chair.gs_source_dir / "sparse" / "0").exists()
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
    # NOT 205 (the file count in train/images/): COLMAP's images.bin
    # registers more cameras than are distributed as files — the dataset
    # runs SfM over train+test (and, for the HCM scenes, extra calibration
    # frames) combined for pose accuracy, then withholds some images'
    # pixels. Verified directly against the real chair data: 263
    # registered, 205 with files on disk, and the 58 without a file are
    # exactly the 58 names in test/test_poses.csv (see test_data_validation
    # for the scene-wide version of this check). Do not "fix" this number
    # down to 205 — that would be re-introducing the bug this test exists
    # to catch.
    assert len(scene.images) == 263
    assert len(scene.cameras) >= 1
    assert len(scene.points3d) > 0


def test_load_sparse_scene_registers_more_images_than_are_distributed_as_files():
    import csv
    import os

    scene = load_sparse_scene(CHAIR_SPARSE)
    registered_names = {img.name for img in scene.images.values()}
    folder_names = set(os.listdir("VAI_NVS_DATA_ROUND2/chair/train/images"))
    registered_without_file = registered_names - folder_names
    assert len(registered_without_file) == 58

    # Don't just trust the magic number 58 — cross-check it's actually the
    # scene's real test_poses.csv image names, not a coincidence.
    with open("VAI_NVS_DATA_ROUND2/chair/test/test_poses.csv", newline="") as f:
        test_pose_names = {row["image_name"] for row in csv.DictReader(f)}
    assert registered_without_file == test_pose_names


def test_load_sparse_scene_preserves_real_point3d_ids_not_file_order_index():
    # Regression test: COLMAP point3D_ids are NOT contiguous (point
    # culling/merging during reconstruction leaves gaps), so re-keying
    # points3d by file-storage order (0..N-1) instead of the real id
    # silently returns the WRONG point for any id that coincidentally
    # collides with a valid 0..N-1 index, corrupting anything that looks
    # points up via images[...].point3D_ids (e.g. depth-target lookups).
    scene = load_sparse_scene(CHAIR_SPARSE)

    referenced_ids = {
        int(pid)
        for img in scene.images.values()
        for pid in img.point3D_ids
        if pid != -1
    }
    assert referenced_ids, "fixture sanity: chair images must reference some points"
    # every id an image references must resolve to a real point
    assert referenced_ids <= set(scene.points3d.keys())
    # ids are not just 0..N-1 (proves this isn't accidentally still using
    # file-order indexing) — real COLMAP ids have gaps and go well beyond
    # the point count.
    assert max(referenced_ids) >= len(scene.points3d)
    # each point's own .id field must match the dict key it's stored under
    for point_id, point in scene.points3d.items():
        assert point.id == point_id


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

**Import mechanism — verified against the real checkout, not the naive approach.** A plain
`sys.path.insert` + `from scene.colmap_loader import ...` executes
`third_party/gaussian-splatting/scene/__init__.py` as a side effect of importing the `scene`
package, which imports `scene.gaussian_model` → `simple_knn._C`, a compiled CUDA extension not
built in this (or any non-GPU) environment — it fails with `ModuleNotFoundError: No module named
'simple_knn'` even though `colmap_loader.py` itself only needs `numpy`/`struct`/`collections`.
Load the module directly by file path instead, bypassing the package `__init__.py` entirely:

```python
from __future__ import annotations

import importlib.util
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_VENDORED_REPO = Path(__file__).resolve().parents[2] / "third_party" / "gaussian-splatting"
if str(_VENDORED_REPO) not in sys.path:
    sys.path.insert(0, str(_VENDORED_REPO))

_COLMAP_LOADER_PATH = _VENDORED_REPO / "scene" / "colmap_loader.py"
_spec = importlib.util.spec_from_file_location("_gs_colmap_loader", _COLMAP_LOADER_PATH)
_colmap_loader = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_colmap_loader)

read_extrinsics_binary = _colmap_loader.read_extrinsics_binary
read_intrinsics_binary = _colmap_loader.read_intrinsics_binary
_Point3D = _colmap_loader.Point3D


@dataclass(frozen=True)
class SparseScene:
    cameras: dict
    images: dict
    points3d: dict


def _read_points3d_binary_preserving_ids(path: Path) -> dict:
    """Parse points3D.bin directly, keyed by the REAL COLMAP point3D_id.

    The vendored `read_points3D_binary` (scene/colmap_loader.py) reads each
    point's id off the wire and then discards it, returning flat
    `(xyzs, rgbs, errors)` arrays indexed 0..N-1 in file-storage order
    instead. COLMAP point3D_ids are NOT contiguous (point culling/merging
    during reconstruction leaves gaps) — verified against the real chair
    scene: images.bin's point3D_ids reference ids up to 105456 across
    80491 points, while file-order indexing only produces keys 0..80490.
    Re-keying by file order doesn't just drop out-of-range lookups, it
    silently returns the WRONG point for any id that happens to collide
    with a valid 0..80490 index (~77% of real ids in the chair scene) —
    this matters because Plan 2's depth regularization looks points up by
    exactly this id (`images[...].point3D_ids`), so a wrong or dropped
    point corrupts that loss silently, no crash, no visible symptom in
    training loss. Binary layout verified against the same file:
    - uint64 num_points
    - per point: struct "<QdddBBBd" (id, x,y,z, r,g,b, error) = 43 bytes,
      then uint64 track_length, then track_length * "ii" (image_id,
      point2D_idx) pairs, 8 bytes each.
    """
    points3d: dict[int, _Point3D] = {}
    with open(path, "rb") as fid:
        num_points = struct.unpack("<Q", fid.read(8))[0]
        for _ in range(num_points):
            point_id, x, y, z, r, g, b, error = struct.unpack("<QdddBBBd", fid.read(43))
            track_length = struct.unpack("<Q", fid.read(8))[0]
            track_elems = struct.unpack("<" + "ii" * track_length, fid.read(8 * track_length))
            image_ids = np.array(track_elems[0::2], dtype=int)
            point2D_idxs = np.array(track_elems[1::2], dtype=int)
            points3d[point_id] = _Point3D(
                id=point_id,
                xyz=np.array([x, y, z]),
                rgb=np.array([r, g, b]),
                error=error,
                image_ids=image_ids,
                point2D_idxs=point2D_idxs,
            )
    return points3d


def load_sparse_scene(sparse_dir: Path) -> SparseScene:
    """Faithful, unfiltered read of the COLMAP sparse model.

    `images` reflects EXACTLY what images.bin registers — this can (and
    for every scene in this dataset, does) include more entries than
    there are files in train/images/; see Task 5/Task 8b for where that
    distinction is actually handled. Do not filter here.
    """
    sparse_dir = Path(sparse_dir)
    cameras = read_intrinsics_binary(str(sparse_dir / "cameras.bin"))
    images = read_extrinsics_binary(str(sparse_dir / "images.bin"))
    points3d = _read_points3d_binary_preserving_ids(sparse_dir / "points3D.bin")
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

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_colmap_io.py -v
```

Expected: `PASS` (5 passed). If `read_extrinsics_binary`/`read_intrinsics_binary`/`Point3D` are
missing or renamed in the checked-out submodule version, open
`third_party/gaussian-splatting/scene/colmap_loader.py` and adjust the referenced names to match
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
  `registered_without_file: list[str]`, `extra_images: list[str]`, `test_pose_row_count: int`,
  `camera_model: str | None`, `problems: list[str]` (empty means valid).

  **`registered_without_file` is informational, NOT a problem by itself.** Verified against the
  real dataset: `images.bin` always registers more cameras than are distributed as files in
  `train/images/` — for `bonsai`/`chair` this set is exactly the scene's `test_poses.csv` image
  names; for the `HCM*` scenes it's a superset (extra calibration-only frames beyond the test
  set). This is intentional dataset structure (SfM run over more images than are shipped as
  training pixels), not data corruption — flagging it as an error would make `validate_scene`
  reject every single scene in the dataset. `problems` instead covers: `extra_images` (a file
  present but NOT registered in `images.bin` — this direction IS a real anomaly, e.g. an unused
  or COLMAP-registration-failed image), missing CSV columns, an unsupported camera model (must
  be `PINHOLE` or `SIMPLE_PINHOLE`), duplicate `image_name` rows in `test_poses.csv`, non-numeric
  values in any numeric column, non-positive `width`/`height`, and — the one place
  `registered_without_file` DOES become a problem — any `test_poses.csv` `image_name` that is
  NOT found anywhere in `images.bin`'s registered set (would mean the test pose has no
  corresponding COLMAP camera at all, an actual inconsistency worth flagging).

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
NUMERIC_CSV_COLUMNS = {
    "qw", "qx", "qy", "qz", "tx", "ty", "tz", "fx", "fy", "cx", "cy", "width", "height",
}
SUPPORTED_CAMERA_MODELS = {"PINHOLE", "SIMPLE_PINHOLE"}


@dataclass
class ValidationReport:
    """Data-quality report for one scene.

    `registered_without_file` is informational only, NOT flagged into
    `problems` — see its field-level note below for why. This module is a
    read-only data-quality check; it does not make any scene safe to feed
    into the real training loader by itself. That's a separate, mandatory
    step (`build_filtered_scene` in Task 8b), which always strips
    registered-without-file images before training regardless of what
    this report says, since the vendored loader crashes on any of them
    otherwise. Do not treat a clean `ValidationReport` as license to skip
    that filtering step.
    """

    scene_name: str
    registered_image_count: int = 0
    folder_image_count: int = 0
    # Images registered in images.bin with no corresponding file in
    # train_images_dir. This is normal, intentional dataset structure —
    # verified across all 7 scenes: always a superset of (often exactly
    # equal to) the scene's test_poses.csv image names, plus extra
    # calibration-only frames for some scenes — not data corruption, so it
    # is never added to `problems`. It is still training-relevant (see
    # class docstring), just not a data-quality problem.
    registered_without_file: list[str] = field(default_factory=list)
    extra_images: list[str] = field(default_factory=list)
    test_pose_row_count: int = 0
    camera_model: str | None = None
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
    # NOT a problem: images.bin legitimately registers more cameras than
    # are distributed as files (test poses + extra calibration frames) —
    # see the Interfaces note above. Reported for visibility only.
    report.registered_without_file = sorted(registered_names - folder_names)
    report.extra_images = sorted(folder_names - registered_names)

    if report.extra_images:
        report.problems.append(
            f"{len(report.extra_images)} folder image(s) not registered in images.bin: "
            f"{report.extra_images}"
        )

    camera_models = {cam.model for cam in sparse.cameras.values()}
    report.camera_model = next(iter(camera_models)) if len(camera_models) == 1 else ",".join(sorted(camera_models))
    unsupported = camera_models - SUPPORTED_CAMERA_MODELS
    if unsupported:
        report.problems.append(
            f"unsupported camera model(s) {sorted(unsupported)}; "
            f"baseline expects one of {sorted(SUPPORTED_CAMERA_MODELS)}"
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

    seen_names: set[str] = set()
    for row in rows:
        name = row.get("image_name", "")
        if name in seen_names:
            report.problems.append(f"duplicate image_name in test_poses.csv: {name}")
        seen_names.add(name)

        # The one direction where a name outside train/images/ IS a
        # problem: a test pose with no COLMAP registration at all has no
        # camera to have derived its pose from.
        if name and name not in registered_names:
            report.problems.append(
                f"test pose '{name}' not registered in images.bin (no COLMAP camera for it)"
            )

        numeric_values = {}
        for col in NUMERIC_CSV_COLUMNS:
            raw = row.get(col)
            try:
                numeric_values[col] = float(raw)
            except (TypeError, ValueError):
                report.problems.append(
                    f"{name}: column '{col}' is not numeric (got {raw!r})"
                )

        if "width" in numeric_values and numeric_values["width"] <= 0:
            report.problems.append(f"{name}: width must be positive, got {numeric_values['width']}")
        if "height" in numeric_values and numeric_values["height"] <= 0:
            report.problems.append(f"{name}: height must be positive, got {numeric_values['height']}")
        for col in ("fx", "fy"):
            if col in numeric_values and numeric_values[col] <= 0:
                report.problems.append(f"{name}: {col} must be positive, got {numeric_values[col]}")

    return report
```

Note: `Camera.model` (from `read_intrinsics_binary`) is expected to already be a string like
`"PINHOLE"` in the vendored loader's output — if the checked-out submodule version instead
returns a numeric model id, adjust the `camera_models` comparison to map ids to names via
`third_party/gaussian-splatting/scene/colmap_loader.py`'s `CAMERA_MODEL_IDS` table before
re-running the tests.

Create `src/data_validation/__init__.py` (empty).

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_validate_scene.py -v
```

Expected: `PASS` (6 passed).

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


def test_combine_score_rejects_non_positive_psnr_max():
    with pytest.raises(ValueError):
        combine_score(lpips_val=0.1, ssim_val=0.9, psnr_val=20.0, psnr_max=0.0)
    with pytest.raises(ValueError):
        combine_score(lpips_val=0.1, ssim_val=0.9, psnr_val=20.0, psnr_max=-5.0)


def test_combine_score_importable_without_torch_installed(monkeypatch):
    # combine_score is the exam's literal grading formula and must stay
    # auditable/usable without pulling in the full ML stack. Simulate
    # torch being unavailable and confirm importing the module (and
    # calling combine_score) still works.
    import builtins
    import importlib
    import sys

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("simulated: torch not installed")
        return real_import(name, *args, **kwargs)

    sys.modules.pop("src.evaluation.compute_metrics", None)
    monkeypatch.setattr(builtins, "__import__", fake_import)
    module = importlib.import_module("src.evaluation.compute_metrics")
    score = module.combine_score(lpips_val=0.1, ssim_val=0.9, psnr_val=30.0, psnr_max=30.0)
    assert score == pytest.approx(0.4 * 0.9 + 0.3 * 0.9 + 0.3 * 1.0, abs=1e-10)


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
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def combine_score(lpips_val: float, ssim_val: float, psnr_val: float, psnr_max: float) -> float:
    """Score = 0.4*(1-LPIPS) + 0.3*SSIM + 0.3*PSNR_norm, PSNR_norm clamped to [0,1].

    Pure function, no torch/skimage dependency, so it stays importable and
    auditable (this is the exam's literal grading formula) even in a
    minimal environment that doesn't have the full ML stack installed.
    """
    if psnr_max <= 0:
        raise ValueError(f"psnr_max must be positive, got {psnr_max}")
    psnr_norm = max(0.0, min(1.0, psnr_val / psnr_max))
    return 0.4 * (1 - lpips_val) + 0.3 * ssim_val + 0.3 * psnr_norm


def _to_lpips_tensor(img: np.ndarray):
    # (H,W,3) uint8 [0,255] -> (1,3,H,W) float32 in [-1,1], as expected by lpips.LPIPS
    import torch

    t = torch.from_numpy(img).float() / 127.5 - 1.0
    return t.permute(2, 0, 1).unsqueeze(0)


def compute_pair_metrics(pred: np.ndarray, gt: np.ndarray, lpips_model) -> dict:
    import torch

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

`torch` is imported lazily inside the functions that actually need it (not at module level) so
`combine_score` — the exam's literal grading formula — stays importable without the full ML
stack installed.

Create `src/evaluation/__init__.py` (empty).

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_compute_metrics.py -v
```

Expected: `PASS` (7 passed).

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

    holdout = select_holdout_images(centers, holdout_ratio=0.2)  # 12 * 0.2 -> 2 (floor)

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

    n_holdout = max(1, math.floor(len(names) * holdout_ratio))
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
    """Build the argv for invoking the vendored train.py.

    All paths are resolved to absolute before being placed in argv. The
    manual Colab verification step (Task 8 Step 6) invokes this via
    `subprocess.run(argv, cwd="third_party/gaussian-splatting")` — a
    relative path would silently resolve against that cwd instead of the
    caller's working directory, pointing at a nonexistent dataset path.

    --source_path is `scene.gs_source_dir`, NOT `scene.root`: the baseline
    requires `<source_path>/images/` and `<source_path>/sparse/0/` as
    direct children, but the real dataset nests these under `<scene>/
    train/`. `gs_source_dir` resolves this correctly for both the raw
    dataset layout and a Task 8b filtered scene copy (see its docstring in
    Task 2).
    """
    argv = [
        "python", "train.py",
        "--source_path", str(Path(scene.gs_source_dir).resolve()),
        "--model_path", str(Path(output_dir).resolve()),
        "--iterations", str(iterations),
    ]
    # Deliberately no --eval flag: the baseline's own --eval does an
    # internal 1/8-uniform-holdout split, which would (a) double-filter an
    # already-holdout-excluded scene when called on the Task 8b filtered
    # copy, and (b) needlessly shrink the training set when called on the
    # full scene for the final submission checkpoint. Holdout exclusion is
    # handled entirely by Task 8b before this function is ever called.
    if resume_checkpoint is not None:
        argv += ["--start_checkpoint", str(Path(resume_checkpoint).resolve())]
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

### Task 8b: Leak-free holdout scene builder

**Why this task exists:** `select_holdout_images` (Task 7) picks holdout image names, but
nothing so far actually removes those images from what `train.py` trains on. Passing `--eval`
to the baseline (as Task 8 originally did) does **not** honor our holdout list — it applies the
baseline's own internal 1/8-uniform split, which is a different, uncontrolled set of images.
Evaluating "holdout" metrics on a checkpoint that was actually trained on those same images is
data leakage: the reported score would be systematically optimistic and untrustworthy for
picking hyperparameters. Task 8's `build_train_argv` was already changed to drop `--eval`
entirely — this task provides the real mechanism: physically construct a scene copy that
excludes the holdout images before training ever sees them.

**Files:**
- Create: `src/training/colmap_writer.py`
- Create: `src/training/holdout_scene.py`
- Test: `tests/test_holdout_scene.py`

**Interfaces:**
- Consumes: `src.common.colmap_io.load_sparse_scene` (Task 4), `src.common.config.SceneConfig`
  (Task 2).
- Produces:
  - `write_images_binary(images: dict, path: Path) -> None` — writes a COLMAP `images.bin`
    containing exactly the given images, in the same binary layout consumed by the vendored
    `scene/colmap_loader.py::read_extrinsics_binary`. Point2D track data is intentionally
    omitted (`num_points2D` written as `0` for every image) since the baseline `train.py` does
    not consume per-image point tracks — only pose, camera_id, and name are needed for
    training. Correctness is verified by **round-trip with the same reader training will use**
    (see Step 1's test), not by matching COLMAP's own writer byte-for-byte.
  - `build_filtered_scene(scene: SceneConfig, holdout_names: set[str], output_dir: Path) -> SceneConfig`
    — creates `output_dir/images/` (symlinks to the original files for every kept image, to
    avoid duplicating image bytes), `output_dir/sparse/0/{cameras.bin,points3D.bin}` (copied
    unchanged — intrinsics and the 3D point cloud don't depend on which images are held out),
    and `output_dir/sparse/0/images.bin` (rewritten to contain only kept images). Returns a new
    `SceneConfig` pointing at `output_dir`, suitable for passing straight into `build_train_argv`
    or the real `Scene()` loader.

  **Always excludes registered-without-file images too, in addition to `holdout_names` — this
  is not optional.** Verified against the real dataset (see Task 4/5): `images.bin` registers
  more cameras than there are files in `train_images_dir` for every scene (test poses, and for
  the `HCM*` scenes, extra calibration-only frames). The vendored `utils/camera_utils.py::
  loadCam` does `Image.open(cam_info.image_path)` with no error handling at all — if a scene
  with `dataset.eval=False` (which this whole plan uses, since holdout is handled here rather
  than by the baseline's own `--eval`) is loaded with any registered-without-file image still
  present, training crashes the first time it happens to sample that camera. So
  `build_filtered_scene(scene, set(), output_dir)` (empty holdout) is not a no-op — it is the
  **minimum required filtering** before a scene may ever be passed to the real training loader,
  used for the "full data" final-training phase in Task 12 exactly as much as for the eval
  phase.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_holdout_scene.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_holdout_scene.py -v
```

Expected: `FAIL` — `ModuleNotFoundError: No module named 'src.training.colmap_writer'`.

- [ ] **Step 3: Write `src/training/colmap_writer.py`**

```python
from __future__ import annotations

import struct
from pathlib import Path


def write_images_binary(images: dict, path: Path) -> None:
    """Write a COLMAP images.bin containing exactly the given images.

    `images` maps image_id -> an object with `.qvec` (length-4 array,
    [qw,qx,qy,qz]), `.tvec` (length-3 array), `.camera_id` (int), `.name`
    (str) — i.e. the same shape as entries returned by
    scene/colmap_loader.py::read_extrinsics_binary in the vendored repo.

    Per-image point2D track data is not supported by this writer:
    num_points2D is always written as 0. This is safe for this pipeline
    because the baseline train.py never reads point2D tracks from
    images.bin, only pose/camera_id/name.
    """
    path = Path(path)
    with open(path, "wb") as fid:
        fid.write(struct.pack("<Q", len(images)))
        for image_id, img in images.items():
            fid.write(struct.pack(
                "<idddddddi",
                int(image_id),
                float(img.qvec[0]), float(img.qvec[1]), float(img.qvec[2]), float(img.qvec[3]),
                float(img.tvec[0]), float(img.tvec[1]), float(img.tvec[2]),
                int(img.camera_id),
            ))
            fid.write(img.name.encode("utf-8") + b"\x00")
            fid.write(struct.pack("<Q", 0))  # num_points2D
```

- [ ] **Step 4: Write `src/training/holdout_scene.py`**

```python
from __future__ import annotations

import os
import shutil
from dataclasses import replace
from pathlib import Path

from src.common.colmap_io import load_sparse_scene
from src.common.config import SceneConfig
from src.training.colmap_writer import write_images_binary


def build_filtered_scene(
    scene: SceneConfig, holdout_names: set[str], output_dir: Path,
) -> SceneConfig:
    """Create a copy of `scene` with every image in `holdout_names` EXCLUDED
    from both the images folder and sparse/0/images.bin, so training on the
    returned SceneConfig cannot see the holdout images at all.

    ALSO always excludes any image registered in images.bin that has no
    corresponding file in scene.train_images_dir, regardless of
    holdout_names — this is not optional (see Interfaces note above): the
    real dataset always registers more cameras than it distributes files
    for, and the vendored loader crashes on `Image.open()` for any of
    them. Calling this with `holdout_names=set()` is the correct way to
    get a "full data" scene that is still safe to train on.
    """
    output_dir = Path(output_dir)
    images_out = output_dir / "images"
    sparse_out = output_dir / "sparse" / "0"
    images_out.mkdir(parents=True, exist_ok=True)
    sparse_out.mkdir(parents=True, exist_ok=True)

    sparse = load_sparse_scene(scene.sparse_dir)
    file_backed_names = {p.name for p in scene.train_images_dir.iterdir() if p.is_file()}
    exclude = set(holdout_names) | (
        {img.name for img in sparse.images.values()} - file_backed_names
    )
    kept_images = {
        img_id: img for img_id, img in sparse.images.items()
        if img.name not in exclude
    }

    for img in kept_images.values():
        src = (scene.train_images_dir / img.name).resolve()
        dst = images_out / img.name
        if not dst.exists():
            os.symlink(src, dst)

    write_images_binary(kept_images, sparse_out / "images.bin")
    shutil.copy2(scene.sparse_dir / "cameras.bin", sparse_out / "cameras.bin")
    shutil.copy2(scene.sparse_dir / "points3D.bin", sparse_out / "points3D.bin")

    return replace(
        scene,
        root=output_dir,
        train_images_dir=images_out,
        sparse_dir=sparse_out,
    )
```

Create `src/training/__init__.py` if not already present from Task 8.

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_holdout_scene.py -v
```

Expected: `PASS` (2 passed). If the round-trip test fails on the fixed-header unpack, open
`third_party/gaussian-splatting/scene/colmap_loader.py::read_extrinsics_binary` and confirm the
exact `format_char_sequence` used for the 64-byte per-image header — adjust the `"<idddddddi"`
struct format in `write_images_binary` to match exactly if the checked-out submodule version
differs.

- [ ] **Step 6: Commit**

```bash
git add src/training/colmap_writer.py src/training/holdout_scene.py tests/test_holdout_scene.py
git commit -m "Add leak-free holdout scene builder (physically excludes holdout images before training)"
```

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


def _pil_save_kwargs(out_path: Path) -> dict:
    """Extra kwargs for Image.save() to avoid unnecessary lossy artifacts.

    Only JPEG needs this: at PIL's default quality=75, re-compressing an
    already-rendered image throws away detail that directly lowers
    PSNR/SSIM/LPIPS for no reason. quality=100 + subsampling=0 (4:4:4, no
    chroma subsampling) keeps JPEG output as close to lossless as the
    format allows. PNG is lossless by default and needs no extra kwargs.
    """
    if out_path.suffix.lower() in (".jpg", ".jpeg"):
        return {"quality": 100, "subsampling": 0}
    return {}


def render_all(
    checkpoint_ply,
    csv_path,
    output_dir: Path,
    render_fn,
    params_list: list[CameraParams] | None = None,
    gaussians=None,
) -> list[Path]:
    """Render every camera in params_list (or loaded from csv_path if
    params_list is None) and write one image per row into output_dir, named
    with the EXACT `image_name` string from test_poses.csv (original
    extension preserved, e.g. `.JPG`/`.jpg` — never rewritten to `.png`).
    PIL infers the output format from the filename extension. For
    JPEG-extension outputs, quality/subsampling are maximized (see
    `_pil_save_kwargs`) since our rendered pixels are already the best the
    model can produce — any avoidable lossy re-compression only throws away
    PSNR/SSIM/LPIPS score for no benefit.
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
        out_path = output_dir / params.image_name
        Image.fromarray(img_array.astype(np.uint8)).save(
            out_path, **_pil_save_kwargs(out_path),
        )
        written.append(out_path)
    return written
```

Create `src/rendering/__init__.py` (empty).

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_render_from_csv.py -v
```

Expected: `PASS` (4 passed).

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

Expected: 58 image files written, filenames matching `test_poses.csv` `image_name` exactly
(original extension preserved), sizes matching each row's `width x height`. The `pipe=...` argument must be filled
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
  — `scene_render_dirs` maps `scene.effective_submission_dir` (Task 2) -> directory of rendered
  images; writes `<submission_dir>/<image_name>` entries into `output_zip` per spec section 14 /
  đề bài mục 7, using each file's name exactly as it was written by `render_all` (Task 9) —
  no renaming happens in this function.

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
    """Validate zip contents against exactly what test_poses.csv expects
    across all `scenes` — flags both MISSING files (Task 11 original scope)
    and UNEXPECTED files (extra images, extra top-level scene directories,
    junk like __MACOSX/, wrong scene naming) since the exam explicitly says
    both missing AND extra scenes/files void the score (debai.md section
    1.6 / 8.4).
    """
    problems: list[str] = []
    zip_path = Path(zip_path)

    with zipfile.ZipFile(zip_path) as zf:
        # Directory marker entries (name ends in "/") aren't real files.
        file_names_in_zip = {n for n in zf.namelist() if not n.endswith("/")}
        accounted_for: set[str] = set()

        for scene in scenes:
            submission_dir = scene.effective_submission_dir
            expected_params = load_test_poses_csv(scene.test_poses_csv)
            scene_entries = [n for n in file_names_in_zip if n.startswith(f"{submission_dir}/")]
            if not scene_entries:
                problems.append(f"scene '{submission_dir}': no files found in zip")
                continue

            for params in expected_params:
                # Use image_name exactly as given — never renamed to .png.
                arcname = f"{submission_dir}/{params.image_name}"
                if arcname not in file_names_in_zip:
                    problems.append(f"scene '{submission_dir}': missing {params.image_name}")
                    continue
                accounted_for.add(arcname)
                data = zf.read(arcname)
                with Image.open(BytesIO(data)) as img:
                    if img.size != (params.width, params.height):
                        problems.append(
                            f"scene '{submission_dir}': {params.image_name} has wrong size "
                            f"{img.size}, expected {(params.width, params.height)}"
                        )

        # Anything in the zip that wasn't matched to an expected file above
        # is unexpected: extra images within a known scene, an entire
        # top-level scene directory not in `scenes` at all, or junk like
        # __MACOSX/ or .DS_Store added by some zip tools. The exam voids
        # the whole score for extra/missing scenes, so this must be caught
        # locally before submitting, not discovered after scoring.
        for extra in sorted(file_names_in_zip - accounted_for):
            problems.append(f"unexpected file in submission zip: {extra}")

    return problems
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_validate_submission.py -v
```

Expected: `PASS` (10 passed).

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
- Consumes: `validate_scene` (Task 5), `select_holdout_images` (Task 7), `build_filtered_scene`
  (Task 8b), `compute_pair_metrics` + `combine_score` (Task 6), `package_submission` (Task 10),
  `validate_submission` (Task 11).
- Produces: `run_baseline_pipeline(scenes: list[SceneConfig], train_fn, render_fn, lpips_model, psnr_max: float, output_root: Path) -> PipelineResult`
  where `train_fn(scene, output_dir) -> Path` (returns checkpoint path),
  `render_fn(checkpoint, params_list, output_dir) -> list[Path]`, and `lpips_model` (any object
  callable as `lpips_model(pred_tensor, gt_tensor) -> torch.Tensor`, same contract as Task 6)
  are all injected, so the ordering and error-aggregation logic is testable without a GPU **and
  without network access** — tests pass the `_StubLpipsModel` from Task 6 instead of the real
  network, which `load_lpips_model()` would otherwise download on every test run. `PipelineResult`
  has fields `per_scene_scores: dict[str, float]`, `skipped_scenes: dict[str, list[str]]`
  (scene name -> `validate_scene` problems, for scenes skipped before spending any GPU time),
  `validation_problems: list[str]`, `submission_zip: Path | None`.
- **`validate_scene` (Task 5) runs before training, not after**: for each scene, if
  `validate_scene(scene).problems` is non-empty, the scene is recorded in
  `result.skipped_scenes` and both training phases are skipped entirely — training on a scene
  with e.g. missing images or an unsupported camera model would burn GPU time on Colab only to
  fail partway through or produce garbage.
- **Two training runs per scene, not one** (this is the leak fix from Task 8b): `train_fn` is
  called once on a `build_filtered_scene` copy (holdout images physically excluded) to produce
  an `eval_checkpoint` used only for holdout scoring, and once on the original, unfiltered
  `scene` to produce a `final_checkpoint` used only for rendering the real `test_poses.csv`
  submission. The two checkpoints are never swapped — scoring an unbiased checkpoint and
  shipping the best-data checkpoint are different goals and must not share a model.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_run_pipeline.py
from pathlib import Path

import numpy as np
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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_run_pipeline.py -v
```

Expected: `FAIL` — `ModuleNotFoundError: No module named 'src.orchestrator'`.

- [ ] **Step 3: Write `src/orchestrator/run_pipeline.py`**

```python
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from src.common.colmap_io import load_sparse_scene
from src.common.config import SceneConfig
from src.common.pose_utils import camera_extrinsics_from_colmap, focal2fov, qvec2rotmat
from src.data_validation.validate_scene import validate_scene
from src.evaluation.compute_metrics import combine_score, compute_pair_metrics
from src.evaluation.make_holdout_split import select_holdout_images
from src.rendering.render_from_csv import CameraParams, load_test_poses_csv
from src.submission.package_submission import package_submission
from src.submission.validate_submission import validate_submission
from src.training.holdout_scene import build_filtered_scene


@dataclass
class PipelineResult:
    per_scene_scores: dict[str, float] = field(default_factory=dict)
    skipped_scenes: dict[str, list[str]] = field(default_factory=dict)
    validation_problems: list[str] = field(default_factory=list)
    submission_zip: Path | None = None


def _camera_params_for_holdout(sparse, holdout_names, image_dims) -> list[CameraParams]:
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
    scenes: list[SceneConfig], train_fn, render_fn, lpips_model, psnr_max: float,
    output_root: Path,
) -> PipelineResult:
    """Two training runs per scene:

    1. Eval training: on a build_filtered_scene copy with holdout images
       physically excluded (Task 8b) -> eval_checkpoint. Used only to score
       holdout images the model never trained on (no leakage).
    2. Final training: on a build_filtered_scene copy with an EMPTY holdout
       set -> final_checkpoint. This still goes through Task 8b, not the
       raw `scene` — see the note below on why the raw scene is never a
       valid training input for this dataset.

    `lpips_model` is always injected by the caller (real callers pass
    `load_lpips_model()` from Task 6; tests pass a network-free stub) so
    this function never implicitly requires network access.

    Each scene is validated (Task 5) before any GPU work — an invalid
    scene is recorded in `result.skipped_scenes` and both training phases
    are skipped for it entirely, so a broken scene never wastes Colab time.

    IMPORTANT — never pass the raw `scene` object to `train_fn` directly.
    `images.bin` always registers more cameras than are distributed as
    files (verified against the real dataset — see Task 4/5/8b), and the
    vendored loader crashes on `Image.open()` for any registered image
    with no file. `build_filtered_scene` is what makes a scene safe to
    train on; Phase A gets this "for free" via the holdout filter, so
    Phase B must call it too, with `holdout_names=set()`, purely to strip
    the registered-without-file images before training on 100% of the
    real, distributed training data.
    """
    output_root = Path(output_root)
    result = PipelineResult()
    scene_render_dirs = {}

    for scene in scenes:
        scene_output = output_root / scene.name
        submission_dir = scene.effective_submission_dir

        report = validate_scene(scene)
        if report.problems:
            result.skipped_scenes[scene.name] = report.problems
            continue

        sparse = load_sparse_scene(scene.sparse_dir)
        file_backed_names = {p.name for p in scene.train_images_dir.iterdir() if p.is_file()}
        # Holdout candidates are drawn ONLY from images that actually have
        # a file: a registered-without-file image (e.g. a test_poses.csv
        # image) has no local pixel data to score against even if chosen,
        # and build_filtered_scene would exclude it anyway regardless of
        # whether it's "selected" as holdout.
        camera_centers = {
            img.name: -np.transpose(qvec2rotmat(np.array(img.qvec))) @ np.array(img.tvec)
            for img in sparse.images.values()
            if img.name in file_backed_names
        }
        holdout_names = set(select_holdout_images(camera_centers, holdout_ratio=0.125))

        # Phase A: leak-free eval training on a scene copy with holdout
        # images physically removed from both images.bin and the images
        # folder (Task 8b) — the model literally cannot have seen them.
        # build_filtered_scene also strips registered-without-file images
        # automatically (see its docstring), so this is training-safe.
        filtered_scene = build_filtered_scene(
            scene, holdout_names, scene_output / "filtered_scene",
        )
        eval_checkpoint = train_fn(filtered_scene, scene_output / "eval_train")

        sample_image = next(scene.train_images_dir.iterdir())
        with Image.open(sample_image) as im:
            image_dims = im.size  # (width, height)

        holdout_params = _camera_params_for_holdout(sparse, holdout_names, image_dims)
        holdout_render_dir = scene_output / "holdout_render"
        rendered_paths = render_fn(eval_checkpoint, holdout_params, holdout_render_dir)

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

        # Phase B: final training on 100% of the real distributed training
        # data — still goes through build_filtered_scene (empty holdout)
        # to strip registered-without-file images; this is the checkpoint
        # that actually gets shipped in the submission.
        full_training_scene = build_filtered_scene(
            scene, set(), scene_output / "full_scene",
        )
        final_checkpoint = train_fn(full_training_scene, scene_output / "final_train")
        test_render_dir = scene_output / "test_render"
        test_params_list = load_test_poses_csv(scene.test_poses_csv)
        render_fn(final_checkpoint, test_params_list, test_render_dir)
        scene_render_dirs[submission_dir] = test_render_dir

    if result.skipped_scenes:
        # Fail closed: the exam voids the ENTIRE score for a missing scene
        # (spec section 14 / debai.md 1.6-8.4), so a submission.zip that's
        # already known to be short a scene is worse than no zip at all —
        # never package or validate one while any scene was skipped.
        result.validation_problems = [
            f"scene '{name}' skipped, no submission produced: {problems}"
            for name, problems in result.skipped_scenes.items()
        ]
        result.submission_zip = None
        return result

    submission_zip = output_root / "submission.zip"
    package_submission(scene_render_dirs, submission_zip)
    result.validation_problems = validate_submission(submission_zip, scenes)
    result.submission_zip = submission_zip
    return result
```

Create `src/orchestrator/__init__.py` (empty).

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_run_pipeline.py -v
```

Expected: `PASS` (3 passed). All three tests use `_StubLpipsModel` and never call
`load_lpips_model()`, so — unlike Task 6 Step 6 — this requires no network access at all.

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
  covers section 3 (`configs/scenes.yaml`, now including the `submission_dir` mapping). Tasks
  3-4 cover the pose-math correctness risk called out in spec sections 8/12. Task 5 covers
  section 5 (data validation, now including camera-model/numeric/duplicate checks). Task 6
  covers section 10's metric formula. Task 7 covers section 10's holdout methodology. Task 8
  covers section 4 (train wrapper, absolute paths, no baseline `--eval`). Task 8b covers the
  actual leak-free enforcement of Task 7's holdout selection (previously only selected, never
  enforced). Task 9 covers section 8 (exact filename/extension preservation, not renamed to
  `.png`). Tasks 10-11 cover section 14 and the completeness risk in section 8.4/14, keyed by
  `effective_submission_dir`. Task 12 covers section 11 (orchestrator), baseline-only, now with
  two training phases (leak-free eval vs. full-data final). Task 13 covers section 4 (Colab
  setup, CUDA caching). Spec sections 6, 7, 9, 13, 15 (experiment matrix, VRAM guard,
  auto-select-best-config, visual QA, reproducibility bundle) are explicitly deferred to the
  follow-up "advanced techniques" plan, as stated in Global Constraints.
- **Placeholder scan:** no TBD/TODO remain; Task 13 Step 2 item 5 documents a genuine
  empirical unknown (whether `pip download --no-binary` round-trips these two submodules)
  rather than hiding it — this is flagged as a manual verification item, not left as an
  unimplemented stub. Task 8b's struct format is similarly flagged as verify-by-round-trip
  rather than assumed correct against an unseen exact spec.
- **Type consistency:** `SceneConfig` (Task 2, now with `submission_dir`/`effective_submission_dir`)
  fields are used identically across Tasks 5, 8, 8b, 9, 11, 12. `CameraParams` (Task 3) fields
  (`image_name, R, T, fov_x, fov_y, width, height`) are used identically in Tasks 9 and 12.
  `combine_score`/`compute_pair_metrics` signatures from Task 6 match their usage in Task 12.
  `build_filtered_scene`'s return type (Task 8b) is a `SceneConfig`, matching what `train_fn`
  (injected in Task 12) expects as its first argument — same as the original, unfiltered scene.
- **Fixes applied after external review (this revision):** (1) submission folder naming is now
  an explicit per-scene config value instead of an unstated assumption baked into code (Task 2,
  10, 11, 12). (2) rendered/validated/packaged output filenames now preserve the CSV's exact
  `image_name` and extension instead of being silently rewritten to `.png` (Task 9, 11, 12).
  (3) holdout images selected by Task 7 are now physically excluded from training via a new
  Task 8b before any eval metric is computed, closing a data-leakage hole where the reported
  holdout score would have been measured on images the model had already trained on. (4)
  `build_train_argv` now resolves `scene.root`/`output_dir`/`resume_checkpoint` to absolute
  paths, since the documented manual Colab invocation changes `cwd` to
  `third_party/gaussian-splatting` (Task 8). (5) `validate_scene` now checks camera model
  support, numeric column validity, duplicate `image_name`, and non-positive width/height
  (Task 5).
- **Fixes applied after a second external review (this revision):** (6) added
  `SceneConfig.gs_source_dir`, always derived from `train_images_dir.parent` — `scene.root`
  itself was never a valid `--source_path` for the real dataset (which nests `images/`/`sparse/
  0/` one level deeper, under `train/`, than the baseline's expected direct-children layout);
  `build_train_argv` now uses `gs_source_dir` (Task 2, Task 8). This was Critical: training
  would have failed immediately on Colab against every real scene. (7) `validate_submission`
  now also flags unexpected/extra entries (extra images within a scene, an entire extra
  top-level scene directory, junk like `__MACOSX/`) — it previously only checked for missing
  files and wrong sizes, but the exam voids the score for extra scenes/files too, not just
  missing ones (Task 11). (8) the spec (`docs/superpowers/specs/2026-07-18-nvs-bts-pipeline-
  design.md` section 14) still described the old `scene_XXX/0001.png` assumption after the
  plan had already moved to `<submission_dir>/<image_name>` — updated to match and to carry the
  same "confirm with organizers" caveat. (9) `run_baseline_pipeline` now calls `validate_scene`
  before either training phase and records failures in `PipelineResult.skipped_scenes` instead
  of training blind — the Interfaces text already claimed this dependency but the code never
  called it. (10) `lpips_model` is now an explicit parameter of `run_baseline_pipeline` instead
  of being loaded internally via `load_lpips_model()`, which downloaded real AlexNet weights
  inside what was supposed to be a network-free local test. (11) `render_all` now passes
  `quality=100, subsampling=0` when saving to a `.jpg`/`.jpeg` path, since PIL's default
  quality=75 would needlessly re-compress an already-final rendered image before every metric
  is computed on it.
- **Fixes applied after a third external review (this revision):** (12) `run_baseline_pipeline`
  now **fails closed** on skipped scenes — if any scene lands in `skipped_scenes`, the pipeline
  does NOT package a partial `submission.zip` (sets `submission_zip = None` and reports the
  skipped scenes in `validation_problems`). Previously it packaged and validated a zip even
  though a scene was missing, which is worse than producing nothing given that the exam voids the
  entire score for a missing scene (spec section 14 / debai.md 1.6-8.4). Two new tests cover
  this: a single skipped scene withholds the zip, and a mix of one good + one broken scene still
  withholds it (one success does not entitle a partial submission).
- **Critical discovery from actually querying the real data (this revision), fixed across
  Tasks 4, 5, 8b, 12:** `images.bin` registers MORE cameras than there are files in
  `train_images_dir`, for every scene in the dataset — verified directly: chair has 263
  registered / 205 files (58 missing = exactly `test_poses.csv`'s image names), bonsai 276/248
  (28 missing = exactly its test set), HCM0421 350/240 and HCM0539 398/240 (110/158 missing,
  test_poses.csv names a strict subset — the rest are extra calibration-only frames). This is
  intentional dataset design (SfM run over more images than are distributed as training pixels),
  not corruption. It was invisible until Task 4 actually loaded real `images.bin` data instead
  of assuming file count == registered count. It matters because the vendored
  `utils/camera_utils.py::loadCam` does `Image.open(cam_info.image_path)` with **no error
  handling**, and `readColmapSceneInfo` treats every registered camera as a training camera
  whenever `dataset.eval=False` (which every training call in this plan uses, since holdout is
  handled by Task 8b instead of the baseline's own `--eval`) — so training on an unfiltered
  scene would have crashed with `FileNotFoundError` on the first sampled phantom camera, for
  every scene, every time. Fixed by: (a) Task 4's test now asserts the real registered count
  (263 for chair) instead of the file count, with an explicit second test documenting the gap;
  (b) Task 5's `validate_scene` no longer treats registered-without-file as a `problems` entry
  (renamed `missing_images` → `registered_without_file`, informational only) but DOES now flag
  the opposite direction (a file with no COLMAP registration) and the case of a `test_poses.csv`
  name with no registration anywhere; (c) Task 8b's `build_filtered_scene` now ALWAYS excludes
  registered-without-file images in addition to whatever `holdout_names` the caller passes, so
  `build_filtered_scene(scene, set(), ...)` is the correct way to get a "full data" scene that
  is still safe to train on; (d) Task 12's orchestrator now restricts holdout-candidate camera
  centers to file-backed images only, and Phase B ("final training") now goes through
  `build_filtered_scene(scene, set(), ...)` instead of passing the raw `scene` straight to
  `train_fn` — the raw scene is never a valid training input for this dataset.
