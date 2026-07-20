# Fix Pipeline Review Findings Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix 8 verified review findings blocking the baseline pipeline (Plan 1) from producing a
valid submission on the real dataset — most critically, all 5 real BTS scenes use a COLMAP camera
model the vendored 3DGS can't train on at all, and the 2 scenes that do train have a corrupted
eval score.

**Architecture:** The Critical finding (unsupported SIMPLE_RADIAL camera model) is fixed by adding
an undistortion preprocessing step (`src/training/undistort_scene.py`) that runs once per scene,
right after `validate_scene` passes and before any GPU work, producing a PINHOLE copy the vendored
loader can train on — a no-op passthrough for scenes already using a supported model (chair,
bonsai). This composes with the existing `build_filtered_scene` (Task 8b of the core plan)
exactly like `scene` did before: `undistort_scene` output feeds into `build_filtered_scene`
unchanged. The remaining 7 findings are independent, smaller fixes to
`src/orchestrator/run_pipeline.py`, `src/data_validation/validate_scene.py`,
`src/submission/validate_submission.py`, `src/training/gs_train_fn.py`, and
`environment/requirements.txt`.

**Tech Stack:** Same as the core plan, plus `opencv-python-headless` (for `cv2.undistort`,
already an unconditional import in the vendored code but never declared) and `plyfile`.

## Global Constraints

- Real dataset camera models (verified by running `load_sparse_scene` against
  `VAI_NVS_DATA_ROUND2/`): `HCM0421/0539/0540/0644/0674` are `SIMPLE_RADIAL` (params
  `[f, cx, cy, k1]`); `chair`/`bonsai` are `SIMPLE_PINHOLE` (params `[f, cx, cy]`). All 5 real
  scenes have exactly 1 shared camera.
- Vendored `third_party/gaussian-splatting/scene/dataset_readers.py:98` hard-asserts on any model
  other than `PINHOLE`/`SIMPLE_PINHOLE` — relaxing `validate_scene`'s allowlist alone is not
  sufficient, training would still crash. Undistortion to PINHOLE is required.
- COLMAP `cameras.bin` binary layout (verified against
  `third_party/gaussian-splatting/scene/colmap_loader.py:215-239`): `uint64 num_cameras`, then per
  camera `int32 camera_id, int32 model_id, uint64 width, uint64 height`, then
  `num_params * float64 params`. `PINHOLE` is `model_id=1`, `num_params=4`, params
  `[fx, fy, cx, cy]`. `SIMPLE_RADIAL` is `model_id=2`, `num_params=4`, params `[f, cx, cy, k1]`.
- Do not break existing tests: `tests/test_run_pipeline.py`, `tests/test_validate_scene.py`,
  `tests/test_gs_train_fn.py`, `tests/test_validate_submission.py`, `tests/test_holdout_scene.py`.
- Test runner is `.venv/bin/python -m pytest` (system `python3` lacks pytest).

---

### Task 1: Add missing vendored runtime dependencies

**Files:**
- Modify: `environment/requirements.txt`

**Interfaces:**
- Produces: `opencv-python-headless`, `plyfile`, `joblib` installed in `.venv`, available for
  every later task in this plan (Task 4 imports `cv2`) and required for `real_train_fn`/
  `real_render_fn` to even import the vendored code on Colab.

- [ ] **Step 1: Add the three packages to `environment/requirements.txt`**

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
opencv-python-headless>=4.8
plyfile>=1.0
joblib>=1.3
```

(Only the three new lines are added at the end; every existing line is unchanged.)

- [ ] **Step 2: Install into the local `.venv` so later tasks can test against real `cv2`**

```bash
.venv/bin/pip install -q opencv-python-headless plyfile joblib
```

Expected: installs cleanly (all three are pure-CPU packages, no CUDA needed).

- [ ] **Step 3: Verify**

```bash
.venv/bin/python -c "import cv2, plyfile, joblib; print('ok')"
```

Expected: `ok`.

- [ ] **Step 4: Commit**

```bash
git add environment/requirements.txt
git commit -m "Add missing opencv-python-headless/plyfile/joblib deps for vendored 3DGS"
```

---

### Task 2: Fix FoV calculation for SIMPLE_PINHOLE cameras (High finding)

**Files:**
- Modify: `src/common/pose_utils.py`
- Modify: `src/orchestrator/run_pipeline.py:36-37`
- Test: `tests/test_pose_utils.py`
- Test: `tests/test_run_pipeline.py`

**Interfaces:**
- Produces: `camera_focal_lengths(model: str, params) -> tuple[float, float]` — correctly
  interprets COLMAP camera params depending on model: `SIMPLE_PINHOLE` (`params=[f, cx, cy]`) has
  one shared focal length; `PINHOLE` (`params=[fx, fy, cx, cy]`) has independent fx/fy. This
  replaces `_camera_params_for_holdout`'s current `camera.params[0], camera.params[1]`, which is
  correct for PINHOLE but silently wrong for SIMPLE_PINHOLE (`params[1]` is `cx`, not a second
  focal length) — verified against real `chair` data: `params = [1113.99, 360.0, 640.0]`, so the
  current code computes `fov_y` from `360.0` (a principal-point coordinate) instead of `1113.99`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_pose_utils.py`:

```python
from src.common.pose_utils import camera_focal_lengths


def test_camera_focal_lengths_simple_pinhole_shares_one_focal_length():
    fx, fy = camera_focal_lengths("SIMPLE_PINHOLE", [1113.99, 360.0, 640.0])
    assert fx == pytest.approx(1113.99)
    assert fy == pytest.approx(1113.99)


def test_camera_focal_lengths_pinhole_has_independent_fx_fy():
    fx, fy = camera_focal_lengths("PINHOLE", [800.0, 850.0, 320.0, 240.0])
    assert fx == pytest.approx(800.0)
    assert fy == pytest.approx(850.0)


def test_camera_focal_lengths_rejects_unsupported_model():
    with pytest.raises(ValueError, match="RADIAL"):
        camera_focal_lengths("SIMPLE_RADIAL", [800.0, 320.0, 240.0, 0.01])
```

Add `import pytest` at the top of `tests/test_pose_utils.py` if not already present (check first —
it likely already imports `numpy` only).

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/python -m pytest tests/test_pose_utils.py::test_camera_focal_lengths_simple_pinhole_shares_one_focal_length -v
```

Expected: FAIL — `ImportError: cannot import name 'camera_focal_lengths'`.

- [ ] **Step 3: Add `camera_focal_lengths` to `src/common/pose_utils.py`**

Add right after `focal2fov`:

```python
def camera_focal_lengths(model: str, params) -> tuple[float, float]:
    """COLMAP camera params -> (fx, fy), for the two models this pipeline
    ever holds a camera in past undistortion (see undistort_scene.py):
    SIMPLE_PINHOLE has one shared focal length (params = [f, cx, cy]);
    PINHOLE has independent fx/fy (params = [fx, fy, cx, cy]). Verified
    against real chair scene data: SIMPLE_PINHOLE params[1] is cx, NOT a
    second focal length — using it as fy silently corrupts fov_y.
    """
    if model == "SIMPLE_PINHOLE":
        return float(params[0]), float(params[0])
    if model == "PINHOLE":
        return float(params[0]), float(params[1])
    raise ValueError(f"unsupported camera model for focal length extraction: {model}")
```

- [ ] **Step 4: Run pose_utils tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/test_pose_utils.py -v
```

Expected: PASS (all tests, including the 3 new ones).

- [ ] **Step 5: Write the failing orchestrator test**

Add to `tests/test_run_pipeline.py` (needs a scene whose holdout eval would visibly differ under
the old vs. fixed FoV — since `chair` is SIMPLE_PINHOLE, its holdout render call already exercises
this path; add an assertion to the existing
`test_run_baseline_pipeline_produces_scores_and_valid_zip` test by capturing the `holdout_params`
passed to `render_fn` and checking `fov_x == fov_y`, which is only true for SIMPLE_PINHOLE — before
the fix, `fov_y` was derived from `cx`, giving a different (wrong) value than `fov_x`):

```python
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
    for params in captured_holdout_params:
        # chair is SIMPLE_PINHOLE (one shared focal length) — fov_x and
        # fov_y must both derive from that same focal length. Before the
        # fix, fov_y was derived from the principal point's x-coordinate
        # instead, which would not match fov_x here.
        assert params.fov_x == pytest.approx(params.fov_y, rel=1e-6)
```

Add `import pytest` to `tests/test_run_pipeline.py` if not already present.

- [ ] **Step 6: Run test to verify it fails**

```bash
.venv/bin/python -m pytest tests/test_run_pipeline.py::test_run_baseline_pipeline_computes_matching_fov_for_simple_pinhole_scene -v
```

Expected: FAIL — `fov_x` and `fov_y` differ (current bug).

- [ ] **Step 7: Fix `src/orchestrator/run_pipeline.py`**

Replace the import line:

```python
from src.common.pose_utils import camera_extrinsics_from_colmap, focal2fov, qvec2rotmat
```

with:

```python
from src.common.pose_utils import camera_extrinsics_from_colmap, camera_focal_lengths, focal2fov, qvec2rotmat
```

Replace `_camera_params_for_holdout`'s body:

```python
        camera = id_to_camera[img.camera_id]
        fx, fy = camera.params[0], camera.params[1]
```

with:

```python
        camera = id_to_camera[img.camera_id]
        fx, fy = camera_focal_lengths(camera.model, camera.params)
```

- [ ] **Step 8: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/test_run_pipeline.py -v
```

Expected: PASS (all tests, including the new one).

- [ ] **Step 9: Commit**

```bash
git add src/common/pose_utils.py src/orchestrator/run_pipeline.py tests/test_pose_utils.py tests/test_run_pipeline.py
git commit -m "Fix FoV calculation for SIMPLE_PINHOLE holdout cameras"
```

---

### Task 3: Add `write_cameras_binary` (needed by undistortion)

**Files:**
- Modify: `src/training/colmap_writer.py`
- Test: `tests/test_colmap_writer.py`

**Interfaces:**
- Produces: `write_cameras_binary(cameras: dict, path: Path) -> None` — writes a COLMAP
  `cameras.bin` containing exactly the given cameras. `cameras` maps `camera_id -> object` with
  `.model` (must be `"PINHOLE"` — the only model this pipeline ever needs to write), `.width`,
  `.height` (int), `.params` (length-4 iterable `[fx, fy, cx, cy]`) — same duck-typed shape as
  `write_images_binary` already uses for its `images` argument. Task 4's `undistort_scene` is the
  consumer.

- [ ] **Step 1: Write the failing test**

Create `tests/test_colmap_writer.py`:

```python
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from src.common.colmap_io import read_intrinsics_binary
from src.training.colmap_writer import write_cameras_binary


@dataclass
class _Camera:
    model: str
    width: int
    height: int
    params: np.ndarray


def test_write_cameras_binary_round_trips_with_the_vendored_reader(tmp_path):
    cameras = {
        1: _Camera(model="PINHOLE", width=1320, height=989, params=np.array([926.4, 926.4, 660.0, 494.5])),
        2: _Camera(model="PINHOLE", width=720, height=1280, params=np.array([1113.99, 1113.99, 360.0, 640.0])),
    }
    out_path = tmp_path / "cameras.bin"

    write_cameras_binary(cameras, out_path)

    reloaded = read_intrinsics_binary(str(out_path))
    assert set(reloaded.keys()) == {1, 2}
    for camera_id, original in cameras.items():
        round_tripped = reloaded[camera_id]
        assert round_tripped.model == "PINHOLE"
        assert round_tripped.width == original.width
        assert round_tripped.height == original.height
        np.testing.assert_allclose(round_tripped.params, original.params, atol=1e-9)


def test_write_cameras_binary_rejects_non_pinhole_model(tmp_path):
    cameras = {1: _Camera(model="SIMPLE_RADIAL", width=64, height=48, params=np.array([80.0, 32.0, 24.0, 0.01]))}
    with pytest.raises(ValueError, match="PINHOLE"):
        write_cameras_binary(cameras, tmp_path / "cameras.bin")
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/python -m pytest tests/test_colmap_writer.py -v
```

Expected: FAIL — `ImportError: cannot import name 'write_cameras_binary'`.

- [ ] **Step 3: Add `write_cameras_binary` to `src/training/colmap_writer.py`**

Append to the end of the file:

```python
def write_cameras_binary(cameras: dict, path: Path) -> None:
    """Write a COLMAP cameras.bin containing exactly the given cameras.

    Only PINHOLE is supported (model_id=1, num_params=4, params
    [fx, fy, cx, cy]) — the only model this pipeline ever writes, produced
    by undistort_scene.py. Binary layout verified against
    third_party/gaussian-splatting/scene/colmap_loader.py's
    read_intrinsics_binary: uint64 num_cameras, then per camera
    int32 camera_id, int32 model_id, uint64 width, uint64 height, then
    num_params * float64 params.
    """
    path = Path(path)
    with open(path, "wb") as fid:
        fid.write(struct.pack("<Q", len(cameras)))
        for camera_id, cam in cameras.items():
            if cam.model != "PINHOLE":
                raise ValueError(
                    f"write_cameras_binary only supports PINHOLE, got {cam.model}"
                )
            fid.write(struct.pack("<iiQQ", int(camera_id), 1, int(cam.width), int(cam.height)))
            fx, fy, cx, cy = cam.params
            fid.write(struct.pack("<dddd", float(fx), float(fy), float(cx), float(cy)))
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/test_colmap_writer.py -v
```

Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/training/colmap_writer.py tests/test_colmap_writer.py
git commit -m "Add write_cameras_binary for the undistortion output"
```

---

### Task 4: Undistort SIMPLE_RADIAL scenes to PINHOLE (Critical finding)

**Files:**
- Create: `src/training/undistort_scene.py`
- Test: `tests/test_undistort_scene.py`

**Interfaces:**
- Consumes: `load_sparse_scene` (existing), `write_cameras_binary` (Task 3).
- Produces: `undistort_scene(scene: SceneConfig, output_dir: Path) -> SceneConfig` — a no-op
  passthrough (returns `scene` unchanged, no copy) when every camera in the scene is already
  `PINHOLE`/`SIMPLE_PINHOLE`; otherwise undistorts every `SIMPLE_RADIAL` image via `cv2.undistort`
  into `output_dir` and returns a new `SceneConfig` pointing at it. `images.bin`/`points3D.bin` are
  copied through unchanged (poses and 3D points are unaffected by undistorting pixels). Task 5's
  orchestrator wiring is the consumer — its output feeds directly into `build_filtered_scene`
  exactly like the raw `scene` did before.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_undistort_scene.py`:

```python
import struct
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.common.colmap_io import read_intrinsics_binary
from src.common.config import SceneConfig
from src.training.colmap_writer import write_images_binary
from src.training.undistort_scene import undistort_scene


class _FakeImage:
    def __init__(self, name, camera_id):
        self.qvec = np.array([1.0, 0.0, 0.0, 0.0])
        self.tvec = np.array([0.0, 0.0, 0.0])
        self.camera_id = camera_id
        self.name = name


def _write_camera(path, camera_id, model_id, width, height, params):
    with open(path, "wb") as fid:
        fid.write(struct.pack("<Q", 1))
        fid.write(struct.pack("<iiQQ", camera_id, model_id, width, height))
        fid.write(struct.pack("<" + "d" * len(params), *params))


def _write_empty_points3d(path):
    with open(path, "wb") as fid:
        fid.write(struct.pack("<Q", 0))


def _make_scene(tmp_path, model_id, params, width=64, height=48):
    root = tmp_path / "scene"
    images_dir = root / "train" / "images"
    sparse_dir = root / "train" / "sparse" / "0"
    images_dir.mkdir(parents=True)
    sparse_dir.mkdir(parents=True)

    _write_camera(sparse_dir / "cameras.bin", 1, model_id, width, height, params)
    write_images_binary({1: _FakeImage("0001.jpg", 1)}, sparse_dir / "images.bin")
    _write_empty_points3d(sparse_dir / "points3D.bin")

    # A synthetic image with a visible pattern so undistortion has
    # something to actually warp (a flat color would look the same either way).
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:, width // 2:] = 255
    cv2.imwrite(str(images_dir / "0001.jpg"), img)

    return SceneConfig(
        name="scene", root=root, train_images_dir=images_dir,
        sparse_dir=sparse_dir, test_poses_csv=root / "test" / "test_poses.csv",
    )


def test_undistort_scene_converts_simple_radial_to_pinhole(tmp_path):
    # model_id 2 = SIMPLE_RADIAL, params [f, cx, cy, k1]
    scene = _make_scene(tmp_path, model_id=2, params=[80.0, 32.0, 24.0, 0.05])
    output_dir = tmp_path / "undistorted"

    result = undistort_scene(scene, output_dir)

    assert result.sparse_dir == output_dir / "sparse" / "0"
    assert result.train_images_dir == output_dir / "images"
    cameras = read_intrinsics_binary(str(result.sparse_dir / "cameras.bin"))
    camera = cameras[1]
    assert camera.model == "PINHOLE"
    fx, fy, cx, cy = camera.params
    assert fx == pytest.approx(80.0)
    assert fy == pytest.approx(80.0)
    assert cx == pytest.approx(32.0)
    assert cy == pytest.approx(24.0)
    assert (result.train_images_dir / "0001.jpg").exists()


def test_undistort_scene_is_noop_for_already_supported_model(tmp_path):
    # model_id 0 = SIMPLE_PINHOLE, params [f, cx, cy] — chair/bonsai's real model.
    scene = _make_scene(tmp_path, model_id=0, params=[80.0, 32.0, 24.0])
    output_dir = tmp_path / "unused_output"

    result = undistort_scene(scene, output_dir)

    assert result is scene
    assert not output_dir.exists()


def test_undistort_scene_rejects_unsupported_model(tmp_path):
    # model_id 3 = RADIAL (2 distortion coeffs), not handled.
    scene = _make_scene(tmp_path, model_id=3, params=[80.0, 32.0, 24.0, 0.01, 0.0])

    with pytest.raises(ValueError, match="RADIAL"):
        undistort_scene(scene, tmp_path / "unused_output")
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/test_undistort_scene.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'src.training.undistort_scene'`.

- [ ] **Step 3: Write `src/training/undistort_scene.py`**

```python
from __future__ import annotations

import shutil
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np

from src.common.colmap_io import load_sparse_scene
from src.common.config import SceneConfig
from src.training.colmap_writer import write_cameras_binary

ALREADY_SUPPORTED_MODELS = {"PINHOLE", "SIMPLE_PINHOLE"}
UNDISTORTABLE_MODELS = {"SIMPLE_RADIAL"}


@dataclass(frozen=True)
class _PinholeCamera:
    model: str
    width: int
    height: int
    params: np.ndarray


def undistort_scene(scene: SceneConfig, output_dir: Path) -> SceneConfig:
    """Undistort a SIMPLE_RADIAL scene into a PINHOLE copy the vendored 3DGS
    can train on directly; a no-op passthrough (no copy) for scenes already
    using a supported model (chair, bonsai).

    Real BTS scenes are registered by COLMAP as SIMPLE_RADIAL
    (params = [f, cx, cy, k1]) — third_party/gaussian-splatting/scene/
    dataset_readers.py only handles PINHOLE/SIMPLE_PINHOLE and asserts on
    anything else. This keeps the same focal length and principal point
    (newCameraMatrix=k_matrix, no re-centering/cropping) and only removes
    the k1 radial term from the pixels via cv2.undistort — poses in
    images.bin and points in points3D.bin are unaffected by this and are
    copied through unchanged; only cameras.bin (model -> PINHOLE) and the
    image pixels change.
    """
    sparse = load_sparse_scene(scene.sparse_dir)
    camera_models = {cam.model for cam in sparse.cameras.values()}

    if camera_models <= ALREADY_SUPPORTED_MODELS:
        return scene

    unsupported = camera_models - ALREADY_SUPPORTED_MODELS - UNDISTORTABLE_MODELS
    if unsupported:
        raise ValueError(f"cannot undistort camera model(s) {sorted(unsupported)}")

    output_dir = Path(output_dir)
    images_out = output_dir / "images"
    sparse_out = output_dir / "sparse" / "0"
    images_out.mkdir(parents=True, exist_ok=True)
    sparse_out.mkdir(parents=True, exist_ok=True)

    new_cameras: dict[int, _PinholeCamera] = {}
    undistort_maps: dict[int, tuple] = {}
    for camera_id, cam in sparse.cameras.items():
        if cam.model == "SIMPLE_RADIAL":
            f, cx, cy, k1 = cam.params
            k_matrix = np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]])
            dist_coeffs = np.array([k1, 0.0, 0.0, 0.0])
            undistort_maps[camera_id] = (k_matrix, dist_coeffs)
            new_cameras[camera_id] = _PinholeCamera(
                model="PINHOLE", width=cam.width, height=cam.height,
                params=np.array([f, f, cx, cy]),
            )
        else:
            undistort_maps[camera_id] = None
            new_cameras[camera_id] = _PinholeCamera(
                model=cam.model, width=cam.width, height=cam.height, params=cam.params,
            )

    for img in sparse.images.values():
        src_path = scene.train_images_dir / img.name
        if not src_path.is_file():
            continue  # registered-without-file images have no pixels to undistort
        dst_path = images_out / img.name
        mapping = undistort_maps[img.camera_id]
        if mapping is None:
            shutil.copy2(src_path, dst_path)
            continue
        k_matrix, dist_coeffs = mapping
        pixels = cv2.imread(str(src_path))
        if pixels is None:
            raise ValueError(f"cv2 could not read image: {src_path}")
        undistorted = cv2.undistort(pixels, k_matrix, dist_coeffs, newCameraMatrix=k_matrix)
        cv2.imwrite(str(dst_path), undistorted)

    write_cameras_binary(new_cameras, sparse_out / "cameras.bin")
    shutil.copy2(scene.sparse_dir / "images.bin", sparse_out / "images.bin")
    shutil.copy2(scene.sparse_dir / "points3D.bin", sparse_out / "points3D.bin")

    return replace(scene, root=output_dir, train_images_dir=images_out, sparse_dir=sparse_out)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/test_undistort_scene.py -v
```

Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/training/undistort_scene.py tests/test_undistort_scene.py
git commit -m "Add undistort_scene to convert SIMPLE_RADIAL scenes to PINHOLE"
```

---

### Task 5: Wire undistortion into the orchestrator (Critical finding, completes the fix)

**Files:**
- Modify: `src/orchestrator/run_pipeline.py`
- Modify: `src/data_validation/validate_scene.py:16`
- Test: `tests/test_run_pipeline.py`
- Test: `tests/test_validate_scene.py`

**Interfaces:**
- Consumes: `undistort_scene` (Task 4).
- Produces: `run_baseline_pipeline` now undistorts every scene right after `validate_scene` passes,
  before any GPU work, and uses the undistorted `working_scene` for every downstream step except
  `test_poses_csv`/`effective_submission_dir` (untouched by undistortion — test camera params are
  given directly, no distortion field in `test_poses.csv`'s schema).
  `validate_scene`'s `SUPPORTED_CAMERA_MODELS` now includes `SIMPLE_RADIAL` since it is now a
  handled, not-a-problem case.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_run_pipeline.py` (needs its own synthetic SIMPLE_RADIAL scene fixture, same
approach as `tests/test_undistort_scene.py`):

```python
import struct


class _FakeImageForRadial:
    def __init__(self, name, camera_id, qvec, tvec):
        self.qvec = qvec
        self.tvec = tvec
        self.camera_id = camera_id
        self.name = name


def _make_simple_radial_scene(tmp_path):
    import numpy as np
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
    write_images_binary(images, sparse_dir / "images.bin")
    with open(sparse_dir / "points3D.bin", "wb") as fid:
        fid.write(struct.pack("<Q", 0))

    from PIL import Image as PILImage
    for name in images.values():
        PILImage.new("RGB", (64, 48), color=(10, 20, 30)).save(images_dir / name.name)

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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/python -m pytest tests/test_run_pipeline.py::test_run_baseline_pipeline_trains_simple_radial_scene_via_undistortion -v
```

Expected: FAIL — `hcm_like_scene` lands in `skipped_scenes` (unsupported camera model).

- [ ] **Step 3: Relax `validate_scene`'s allowlist**

In `src/data_validation/validate_scene.py`, change:

```python
SUPPORTED_CAMERA_MODELS = {"PINHOLE", "SIMPLE_PINHOLE"}
```

to:

```python
# SIMPLE_RADIAL is "supported" because run_baseline_pipeline always routes
# scenes through undistort_scene before training, converting SIMPLE_RADIAL
# to PINHOLE — see src/training/undistort_scene.py. This module never
# undistorts anything itself; it only decides whether the model is one
# undistort_scene knows how to handle.
SUPPORTED_CAMERA_MODELS = {"PINHOLE", "SIMPLE_PINHOLE", "SIMPLE_RADIAL"}
```

- [ ] **Step 4: Wire `undistort_scene` into `run_baseline_pipeline`**

In `src/orchestrator/run_pipeline.py`, add the import:

```python
from src.training.holdout_scene import build_filtered_scene
from src.training.undistort_scene import undistort_scene
```

Replace the loop body from `report = validate_scene(scene)` through the end of Phase B (i.e. lines
87-146 of the current file) with:

```python
        report = validate_scene(scene)
        if report.problems:
            result.skipped_scenes[scene.name] = report.problems
            continue

        # Real BTS scenes are SIMPLE_RADIAL (radially distorted); the
        # vendored 3DGS only handles PINHOLE/SIMPLE_PINHOLE.
        # undistort_scene is a no-op passthrough for scenes already using a
        # supported model (chair, bonsai) and produces a PINHOLE copy
        # otherwise. Every downstream step in this loop operates on
        # working_scene, never the raw scene, except test_poses_csv /
        # effective_submission_dir — undistortion never touches test poses.
        working_scene = undistort_scene(scene, scene_output / "undistorted")

        sparse = load_sparse_scene(working_scene.sparse_dir)
        file_backed_names = {p.name for p in working_scene.train_images_dir.iterdir() if p.is_file()}
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
            working_scene, holdout_names, scene_output / "filtered_scene",
        )
        eval_checkpoint = train_fn(filtered_scene, scene_output / "eval_train")

        sample_image = next(working_scene.train_images_dir.iterdir())
        with Image.open(sample_image) as im:
            image_dims = im.size  # (width, height)

        holdout_params = _camera_params_for_holdout(sparse, holdout_names, image_dims)
        holdout_render_dir = scene_output / "holdout_render"
        rendered_paths = render_fn(eval_checkpoint, holdout_params, holdout_render_dir)

        scores = []
        for path, params in zip(rendered_paths, holdout_params):
            gt_path = working_scene.train_images_dir / params.image_name
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
            working_scene, set(), scene_output / "full_scene",
        )
        final_checkpoint = train_fn(full_training_scene, scene_output / "final_train")
        test_render_dir = scene_output / "test_render"
        test_params_list = load_test_poses_csv(scene.test_poses_csv)
        render_fn(final_checkpoint, test_params_list, test_render_dir)
        scene_render_dirs[submission_dir] = test_render_dir
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/test_run_pipeline.py tests/test_validate_scene.py -v
```

Expected: PASS (all tests, including the new SIMPLE_RADIAL one). The existing chair/bonsai-based
tests must still pass unchanged — `undistort_scene` is a no-op passthrough for them.

- [ ] **Step 6: Run the full suite to check for regressions**

```bash
.venv/bin/python -m pytest -q
```

Expected: all pass.

- [ ] **Step 7: Commit**

```bash
git add src/orchestrator/run_pipeline.py src/data_validation/validate_scene.py tests/test_run_pipeline.py tests/test_validate_scene.py
git commit -m "Wire undistortion into the orchestrator so SIMPLE_RADIAL scenes train"
```

---

### Task 6: Fix subprocess portability (Medium finding)

**Files:**
- Modify: `src/training/train_wrapper.py`
- Modify: `src/training/gs_train_fn.py`
- Test: `tests/test_train_wrapper.py`
- Test: `tests/test_gs_train_fn.py`

**Interfaces:**
- Produces: `build_train_argv` now uses `sys.executable` instead of the hardcoded string
  `"python"` (guarantees the same interpreter the calling process is running under, not whatever
  `python` happens to resolve to on `PATH`). `gs_train_fn.GS_ROOT` is a new module-level constant
  (`Path(__file__).resolve().parents[2] / "third_party" / "gaussian-splatting"`, same pattern
  already used as `_GS_ROOT` in `gs_render_fn.py`) and `real_train_fn` passes
  `cwd=str(GS_ROOT)` instead of the relative string `"third_party/gaussian-splatting"`, so it no
  longer depends on the calling process's current working directory.

- [ ] **Step 1: Update `tests/test_train_wrapper.py`**

Add an assertion to `test_build_train_argv_without_resume` (append at the end of the function):

```python
    import sys
    assert argv[0] == sys.executable
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/python -m pytest tests/test_train_wrapper.py::test_build_train_argv_without_resume -v
```

Expected: FAIL — `argv[0] == "python"`, not `sys.executable`.

- [ ] **Step 3: Fix `build_train_argv` in `src/training/train_wrapper.py`**

Add `import sys` at the top of the file (alongside the existing `import re`). Change:

```python
    argv = [
        "python", "train.py",
```

to:

```python
    argv = [
        sys.executable, "train.py",
```

- [ ] **Step 4: Run train_wrapper tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/test_train_wrapper.py -v
```

Expected: PASS (all tests).

- [ ] **Step 5: Update `tests/test_gs_train_fn.py`'s cwd assertions**

Replace the import line:

```python
from src.training import gs_train_fn
```

stays the same (no change needed to the import itself), but every test that currently asserts
`cwd == "third_party/gaussian-splatting"` needs updating. Replace:

```python
    argv, cwd, check = calls[0]
    assert cwd == "third_party/gaussian-splatting"
```

with:

```python
    argv, cwd, check = calls[0]
    assert cwd == str(gs_train_fn.GS_ROOT)
```

(in `test_real_train_fn_runs_subprocess_from_scratch_when_no_checkpoint`).

- [ ] **Step 6: Run test to verify it fails**

```bash
.venv/bin/python -m pytest tests/test_gs_train_fn.py::test_real_train_fn_runs_subprocess_from_scratch_when_no_checkpoint -v
```

Expected: FAIL — `ImportError`/`AttributeError`, `gs_train_fn.GS_ROOT` does not exist yet, or the
assertion fails against the old relative string.

- [ ] **Step 7: Fix `src/training/gs_train_fn.py`**

Add the constant near the top of the file (after the imports):

```python
GS_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "gaussian-splatting"
```

Change:

```python
    subprocess.run(argv, cwd="third_party/gaussian-splatting", check=True)
```

to:

```python
    subprocess.run(argv, cwd=str(GS_ROOT), check=True)
```

- [ ] **Step 8: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/test_gs_train_fn.py -v
```

Expected: PASS (all tests).

- [ ] **Step 9: Commit**

```bash
git add src/training/train_wrapper.py src/training/gs_train_fn.py tests/test_train_wrapper.py tests/test_gs_train_fn.py
git commit -m "Fix train subprocess to use sys.executable and an absolute cwd"
```

---

### Task 7: Guard against stale checkpoint reuse (Medium finding)

**Files:**
- Modify: `src/training/gs_train_fn.py`
- Test: `tests/test_gs_train_fn.py`

**Interfaces:**
- Produces: `real_train_fn` now writes a fingerprint sidecar file
  (`<output_dir>/.gs_train_fn_fingerprint`) alongside the checkpoint, derived from
  `scene.train_images_dir`'s resolved path, its sorted file list, and `iterations`. On the next
  call, if a checkpoint exists but the fingerprint file is missing or doesn't match, the entire
  `output_dir` is wiped and training starts clean instead of silently reusing a checkpoint trained
  on different data.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_gs_train_fn.py`:

```python
def test_real_train_fn_retrains_when_scene_contents_changed(tmp_path, monkeypatch):
    scene_dir = tmp_path / "scene_images"
    scene_dir.mkdir()
    (scene_dir / "0001.jpg").touch()
    scene = SceneConfig(
        name="chair", root=Path("VAI_NVS_DATA_ROUND2/chair"),
        train_images_dir=scene_dir,
        sparse_dir=Path("VAI_NVS_DATA_ROUND2/chair/train/sparse/0"),
        test_poses_csv=Path("VAI_NVS_DATA_ROUND2/chair/test/test_poses.csv"),
        submission_dir="chair",
    )

    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "chkpnt30000.pth").touch()
    (output_dir / "stale_marker.txt").touch()  # simulates leftover state from the old run

    calls = []

    def fake_run(argv, cwd, check):
        calls.append(argv)
        (output_dir / "chkpnt30000.pth").touch()

    monkeypatch.setattr(gs_train_fn.subprocess, "run", fake_run)

    result = gs_train_fn.real_train_fn(scene, output_dir, iterations=30000)

    # No fingerprint file existed yet, so the old checkpoint must be
    # treated as stale/untrusted and training must actually run, not skip.
    assert len(calls) == 1
    assert not (output_dir / "stale_marker.txt").exists(), (
        "output_dir must be wiped before retraining on a fingerprint mismatch"
    )
    assert result == output_dir / "chkpnt30000.pth"


def test_real_train_fn_reuses_checkpoint_when_fingerprint_matches(tmp_path, monkeypatch):
    scene_dir = tmp_path / "scene_images"
    scene_dir.mkdir()
    (scene_dir / "0001.jpg").touch()
    scene = SceneConfig(
        name="chair", root=Path("VAI_NVS_DATA_ROUND2/chair"),
        train_images_dir=scene_dir,
        sparse_dir=Path("VAI_NVS_DATA_ROUND2/chair/train/sparse/0"),
        test_poses_csv=Path("VAI_NVS_DATA_ROUND2/chair/test/test_poses.csv"),
        submission_dir="chair",
    )
    output_dir = tmp_path / "output"

    calls = []

    def fake_run(argv, cwd, check):
        calls.append(argv)
        (output_dir / "chkpnt30000.pth").touch()

    monkeypatch.setattr(gs_train_fn.subprocess, "run", fake_run)

    # First call: trains from scratch and records a fingerprint.
    gs_train_fn.real_train_fn(scene, output_dir, iterations=30000)
    assert len(calls) == 1

    # Second call, same scene contents: must skip, not retrain.
    result = gs_train_fn.real_train_fn(scene, output_dir, iterations=30000)
    assert len(calls) == 1  # unchanged — no new subprocess call
    assert result == output_dir / "chkpnt30000.pth"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/test_gs_train_fn.py::test_real_train_fn_retrains_when_scene_contents_changed tests/test_gs_train_fn.py::test_real_train_fn_reuses_checkpoint_when_fingerprint_matches -v
```

Expected: FAIL — the first test currently skips training (finds the pre-existing checkpoint,
never wipes `stale_marker.txt`); no fingerprinting exists yet.

- [ ] **Step 3: Fix `src/training/gs_train_fn.py`**

Add `import hashlib` and `import shutil` to the top imports. Add this helper right after the
`GS_ROOT` constant (from Task 6):

```python
_FINGERPRINT_FILENAME = ".gs_train_fn_fingerprint"


def _scene_fingerprint(scene: SceneConfig, iterations: int) -> str:
    names = sorted(p.name for p in Path(scene.train_images_dir).iterdir() if p.is_file())
    payload = f"{Path(scene.train_images_dir).resolve()}|{iterations}|{','.join(names)}"
    return hashlib.sha256(payload.encode()).hexdigest()
```

Replace the body of `real_train_fn`:

```python
    output_dir = Path(output_dir)
    existing = find_latest_checkpoint(output_dir)
    if existing is not None and (checkpoint_iteration(existing) or 0) >= iterations:
        return existing

    argv = build_train_argv(
        scene, output_dir, iterations,
        resume_checkpoint=existing,
        extra_args=["--checkpoint_iterations", str(iterations)],
    )
    subprocess.run(argv, cwd=str(GS_ROOT), check=True)

    checkpoint = find_latest_checkpoint(output_dir)
    if checkpoint is None or (checkpoint_iteration(checkpoint) or 0) < iterations:
        raise RuntimeError(
            f"train.py finished but no checkpoint at target iteration {iterations} "
            f"found in {output_dir}"
        )
    return checkpoint
```

with:

```python
    output_dir = Path(output_dir)
    fingerprint = _scene_fingerprint(scene, iterations)
    fingerprint_path = output_dir / _FINGERPRINT_FILENAME

    existing = find_latest_checkpoint(output_dir)
    if existing is not None and (
        not fingerprint_path.exists() or fingerprint_path.read_text() != fingerprint
    ):
        # Scene contents changed since this checkpoint was produced (e.g.
        # dataset re-uploaded, holdout selection changed) — it is not safe
        # to resume from or silently reuse. Wipe and start clean rather
        # than risk shipping a stale model.
        shutil.rmtree(output_dir)
        existing = None

    if existing is not None and (checkpoint_iteration(existing) or 0) >= iterations:
        return existing

    output_dir.mkdir(parents=True, exist_ok=True)
    fingerprint_path.write_text(fingerprint)

    argv = build_train_argv(
        scene, output_dir, iterations,
        resume_checkpoint=existing,
        extra_args=["--checkpoint_iterations", str(iterations)],
    )
    subprocess.run(argv, cwd=str(GS_ROOT), check=True)

    checkpoint = find_latest_checkpoint(output_dir)
    if checkpoint is None or (checkpoint_iteration(checkpoint) or 0) < iterations:
        raise RuntimeError(
            f"train.py finished but no checkpoint at target iteration {iterations} "
            f"found in {output_dir}"
        )
    return checkpoint
```

- [ ] **Step 4: Run all gs_train_fn tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/test_gs_train_fn.py -v
```

Expected: PASS (all tests — including the 4 pre-existing ones from Task 4 of the colab-runner
plan, since a freshly created `output_dir` with no prior checkpoint has no fingerprint mismatch to
trigger, and the resume-from-partial-checkpoint test's `output_dir` also starts with no
fingerprint file, so its first-ever call behaves the same as before).

- [ ] **Step 5: Commit**

```bash
git add src/training/gs_train_fn.py tests/test_gs_train_fn.py
git commit -m "Guard against reusing a checkpoint trained on stale scene contents"
```

---

### Task 8: Harden `validate_submission` image decoding (Medium finding)

**Files:**
- Modify: `src/submission/validate_submission.py`
- Test: `tests/test_validate_submission.py`

**Interfaces:**
- Produces: `validate_submission` now forces a full image decode (`img.load()`, not just the
  header `Image.open()` parses) and flags non-`RGB` images, catching both a truncated-but-valid-
  header PNG and a correctly-sized-but-grayscale PNG — both currently pass silently, verified by
  reproducing each locally (`Image.open()` succeeds and reports the right `.size` on truncated
  data; `.load()` is what actually raises).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_validate_submission.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/test_validate_submission.py::test_validate_submission_flags_truncated_image_data tests/test_validate_submission.py::test_validate_submission_flags_non_rgb_image -v
```

Expected: FAIL — both currently pass with no problems reported.

- [ ] **Step 3: Fix `src/submission/validate_submission.py`**

Replace:

```python
                accounted_for.add(arcname)
                data = zf.read(arcname)
                try:
                    with Image.open(BytesIO(data)) as img:
                        if img.size != (params.width, params.height):
                            problems.append(
                                f"scene '{submission_dir}': {params.image_name} has wrong size "
                                f"{img.size}, expected {(params.width, params.height)}"
                            )
                except (UnidentifiedImageError, OSError) as e:
                    problems.append(
                        f"scene '{submission_dir}': {params.image_name} is not a valid image ({e})"
                    )
```

with:

```python
                accounted_for.add(arcname)
                data = zf.read(arcname)
                try:
                    with Image.open(BytesIO(data)) as img:
                        img.load()  # force full decode, not just the header
                        if img.size != (params.width, params.height):
                            problems.append(
                                f"scene '{submission_dir}': {params.image_name} has wrong size "
                                f"{img.size}, expected {(params.width, params.height)}"
                            )
                        if img.mode != "RGB":
                            problems.append(
                                f"scene '{submission_dir}': {params.image_name} has mode "
                                f"'{img.mode}', expected 'RGB'"
                            )
                except (UnidentifiedImageError, OSError) as e:
                    problems.append(
                        f"scene '{submission_dir}': {params.image_name} is not a valid image ({e})"
                    )
```

- [ ] **Step 4: Run all validate_submission tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/test_validate_submission.py -v
```

Expected: PASS (all tests, including the existing `test_validate_submission_passes_for_correct_zip`
— its fixture images are `Image.new("RGB", ...)`, already RGB and fully decodable).

- [ ] **Step 5: Commit**

```bash
git add src/submission/validate_submission.py tests/test_validate_submission.py
git commit -m "Force full image decode and enforce RGB mode in validate_submission"
```

---

### Task 9: Make `validate_scene` exception-safe against corrupt sparse files (Medium finding)

**Files:**
- Modify: `src/data_validation/validate_scene.py`
- Test: `tests/test_validate_scene.py`

**Interfaces:**
- Produces: `validate_scene` now checks that `cameras.bin`, `images.bin`, `points3D.bin` each
  exist individually (not just that the parent directory exists) and catches any exception
  `load_sparse_scene` raises (missing file, truncated/corrupt binary), turning it into a
  `problems` entry instead of letting it propagate and crash the whole pipeline run.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_validate_scene.py` (check the existing file first for its scene-fixture
helper — reuse whatever pattern it already uses to build a `SceneConfig` pointing at a `tmp_path`
scene; the tests below assume a `_write_minimal_valid_scene(tmp_path)`-style helper already exists
per the existing test file's conventions — if it doesn't, build the sparse dir directly):

```python
def test_validate_scene_reports_problem_for_corrupt_sparse_binary(tmp_path):
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
```

Check the top of `tests/test_validate_scene.py` for its existing imports (`SceneConfig`,
`validate_scene`) — both tests above only need those two, already imported by the existing file.

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/test_validate_scene.py::test_validate_scene_reports_problem_for_corrupt_sparse_binary tests/test_validate_scene.py::test_validate_scene_reports_problem_for_missing_sparse_file -v
```

Expected: FAIL — both currently raise an unhandled exception (`struct.error` or similar) instead
of returning a report.

- [ ] **Step 3: Fix `src/data_validation/validate_scene.py`**

Add `import struct` to the top imports. Replace:

```python
    if not scene.sparse_dir.exists():
        report.problems.append(f"sparse dir not found: {scene.sparse_dir}")
        return report
    if not scene.train_images_dir.exists():
        report.problems.append(f"train images dir not found: {scene.train_images_dir}")
        return report

    sparse = load_sparse_scene(scene.sparse_dir)
```

with:

```python
    if not scene.sparse_dir.exists():
        report.problems.append(f"sparse dir not found: {scene.sparse_dir}")
        return report
    if not scene.train_images_dir.exists():
        report.problems.append(f"train images dir not found: {scene.train_images_dir}")
        return report

    for required_file in ("cameras.bin", "images.bin", "points3D.bin"):
        if not (scene.sparse_dir / required_file).exists():
            report.problems.append(f"missing required sparse file: {required_file}")
            return report

    try:
        sparse = load_sparse_scene(scene.sparse_dir)
    except (OSError, struct.error, ValueError) as e:
        report.problems.append(f"failed to parse sparse reconstruction: {e}")
        return report
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/test_validate_scene.py -v
```

Expected: PASS (all tests, including the pre-existing ones — real scenes have all 3 files intact
and parse cleanly, so the new checks are no-ops for them).

- [ ] **Step 5: Commit**

```bash
git add src/data_validation/validate_scene.py tests/test_validate_scene.py
git commit -m "Make validate_scene exception-safe against corrupt or missing sparse files"
```

---

### Task 10: Close CSV validation gaps — NaN/Inf, fractional dimensions, empty names (Low/Medium finding)

**Files:**
- Modify: `src/data_validation/validate_scene.py`
- Test: `tests/test_validate_scene.py`

**Interfaces:**
- Produces: `validate_scene`'s CSV row validation now rejects NaN/Infinity in any numeric column
  (`float("nan") <= 0` and `float("inf") <= 0` are both `False` in Python, so the existing
  positivity checks silently let them through — verified locally), rejects fractional
  `width`/`height` (verified: `int("640.5")` raises downstream in
  `camera_params_from_csv_row`, a mismatch with `validate_scene`'s own `float()`-based check that
  currently treats `"640.5"` as valid), and flags an empty `image_name` explicitly (verified:
  `Path(some_dir) / ""` returns `some_dir` unchanged, so an empty name would make rendering write
  into the scene's output directory path itself instead of a distinct file).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_validate_scene.py` (reusing whatever CSV-row-building helper the existing file
already has — if none exists, write the CSV directly as shown):

```python
def test_validate_scene_flags_nan_and_inf_numeric_values(tmp_path):
    root = tmp_path / "scene_with_nan"
    images_dir = root / "train" / "images"
    sparse_dir = root / "train" / "sparse" / "0"
    images_dir.mkdir(parents=True)
    sparse_dir.mkdir(parents=True)
    # Minimal valid-enough sparse files aren't needed for this test since
    # the CSV loop runs after sparse parsing succeeds — use a truly empty
    # but well-formed sparse scene (0 cameras/images/points) so
    # load_sparse_scene doesn't itself fail first.
    import struct
    with open(sparse_dir / "cameras.bin", "wb") as fid:
        fid.write(struct.pack("<Q", 0))
    with open(sparse_dir / "images.bin", "wb") as fid:
        fid.write(struct.pack("<Q", 0))
    with open(sparse_dir / "points3D.bin", "wb") as fid:
        fid.write(struct.pack("<Q", 0))

    test_dir = root / "test"
    test_dir.mkdir()
    csv_path = test_dir / "test_poses.csv"
    with open(csv_path, "w", newline="") as f:
        import csv
        writer = csv.DictWriter(f, fieldnames=[
            "image_name", "qw", "qx", "qy", "qz", "tx", "ty", "tz",
            "fx", "fy", "cx", "cy", "width", "height",
        ])
        writer.writeheader()
        writer.writerow({
            "image_name": "0001.png", "qw": 1, "qx": 0, "qy": 0, "qz": 0,
            "tx": 0, "ty": 0, "tz": 0, "fx": "nan", "fy": "inf", "cx": 0, "cy": 0,
            "width": 64, "height": 32,
        })

    scene = SceneConfig(
        name="scene_with_nan", root=root, train_images_dir=images_dir,
        sparse_dir=sparse_dir, test_poses_csv=csv_path,
    )

    report = validate_scene(scene)

    assert any("fx" in p and "finite" in p.lower() for p in report.problems)
    assert any("fy" in p and "finite" in p.lower() for p in report.problems)


def test_validate_scene_flags_fractional_width_height(tmp_path):
    root = tmp_path / "scene_with_fractional_dims"
    images_dir = root / "train" / "images"
    sparse_dir = root / "train" / "sparse" / "0"
    images_dir.mkdir(parents=True)
    sparse_dir.mkdir(parents=True)
    import struct
    with open(sparse_dir / "cameras.bin", "wb") as fid:
        fid.write(struct.pack("<Q", 0))
    with open(sparse_dir / "images.bin", "wb") as fid:
        fid.write(struct.pack("<Q", 0))
    with open(sparse_dir / "points3D.bin", "wb") as fid:
        fid.write(struct.pack("<Q", 0))

    test_dir = root / "test"
    test_dir.mkdir()
    csv_path = test_dir / "test_poses.csv"
    with open(csv_path, "w", newline="") as f:
        import csv
        writer = csv.DictWriter(f, fieldnames=[
            "image_name", "qw", "qx", "qy", "qz", "tx", "ty", "tz",
            "fx", "fy", "cx", "cy", "width", "height",
        ])
        writer.writeheader()
        writer.writerow({
            "image_name": "0001.png", "qw": 1, "qx": 0, "qy": 0, "qz": 0,
            "tx": 0, "ty": 0, "tz": 0, "fx": 80, "fy": 80, "cx": 32, "cy": 16,
            "width": "640.5", "height": 32,
        })

    scene = SceneConfig(
        name="scene_with_fractional_dims", root=root, train_images_dir=images_dir,
        sparse_dir=sparse_dir, test_poses_csv=csv_path,
    )

    report = validate_scene(scene)

    assert any("width" in p and "whole number" in p.lower() for p in report.problems)


def test_validate_scene_flags_empty_image_name(tmp_path):
    root = tmp_path / "scene_with_empty_name"
    images_dir = root / "train" / "images"
    sparse_dir = root / "train" / "sparse" / "0"
    images_dir.mkdir(parents=True)
    sparse_dir.mkdir(parents=True)
    import struct
    with open(sparse_dir / "cameras.bin", "wb") as fid:
        fid.write(struct.pack("<Q", 0))
    with open(sparse_dir / "images.bin", "wb") as fid:
        fid.write(struct.pack("<Q", 0))
    with open(sparse_dir / "points3D.bin", "wb") as fid:
        fid.write(struct.pack("<Q", 0))

    test_dir = root / "test"
    test_dir.mkdir()
    csv_path = test_dir / "test_poses.csv"
    with open(csv_path, "w", newline="") as f:
        import csv
        writer = csv.DictWriter(f, fieldnames=[
            "image_name", "qw", "qx", "qy", "qz", "tx", "ty", "tz",
            "fx", "fy", "cx", "cy", "width", "height",
        ])
        writer.writeheader()
        writer.writerow({
            "image_name": "", "qw": 1, "qx": 0, "qy": 0, "qz": 0,
            "tx": 0, "ty": 0, "tz": 0, "fx": 80, "fy": 80, "cx": 32, "cy": 16,
            "width": 64, "height": 32,
        })

    scene = SceneConfig(
        name="scene_with_empty_name", root=root, train_images_dir=images_dir,
        sparse_dir=sparse_dir, test_poses_csv=csv_path,
    )

    report = validate_scene(scene)

    assert any("empty image_name" in p.lower() for p in report.problems)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/python -m pytest tests/test_validate_scene.py::test_validate_scene_flags_nan_and_inf_numeric_values tests/test_validate_scene.py::test_validate_scene_flags_fractional_width_height tests/test_validate_scene.py::test_validate_scene_flags_empty_image_name -v
```

Expected: FAIL — none of these are currently flagged as problems.

- [ ] **Step 3: Fix `src/data_validation/validate_scene.py`**

Add `import math` to the top imports. Replace the CSV row loop body:

```python
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
```

with:

```python
        if not name:
            report.problems.append("test_poses.csv row has empty image_name")

        numeric_values = {}
        for col in NUMERIC_CSV_COLUMNS:
            raw = row.get(col)
            try:
                value = float(raw)
            except (TypeError, ValueError):
                report.problems.append(
                    f"{name}: column '{col}' is not numeric (got {raw!r})"
                )
                continue
            if not math.isfinite(value):
                report.problems.append(
                    f"{name}: column '{col}' is not finite (got {raw!r})"
                )
                continue
            numeric_values[col] = value

        if "width" in numeric_values and numeric_values["width"] <= 0:
            report.problems.append(f"{name}: width must be positive, got {numeric_values['width']}")
        if "height" in numeric_values and numeric_values["height"] <= 0:
            report.problems.append(f"{name}: height must be positive, got {numeric_values['height']}")
        for col in ("fx", "fy"):
            if col in numeric_values and numeric_values[col] <= 0:
                report.problems.append(f"{name}: {col} must be positive, got {numeric_values[col]}")
        for col in ("width", "height"):
            if col in numeric_values and not numeric_values[col].is_integer():
                report.problems.append(
                    f"{name}: {col} must be a whole number, got {numeric_values[col]}"
                )
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/bin/python -m pytest tests/test_validate_scene.py -v
```

Expected: PASS (all tests).

- [ ] **Step 5: Run the full suite to check for regressions**

```bash
.venv/bin/python -m pytest -q
```

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/data_validation/validate_scene.py tests/test_validate_scene.py
git commit -m "Reject NaN/Inf, fractional width/height, and empty image_name in test_poses.csv"
```

---

## Self-Review Summary

- **Spec coverage:** All 8 verified review findings map to a task: Critical (SIMPLE_RADIAL) →
  Tasks 3-5. High (FoV) → Task 2. High (missing deps) → Task 1. Medium (cwd/interpreter) → Task 6.
  Medium (checkpoint staleness) → Task 7. Medium (validate_submission decode) → Task 8. Medium
  (validate_scene exception-safety) → Task 9. Low/Medium (CSV gaps) → Task 10.
- **Placeholder scan:** no TBD/TODO. Task 4's exact `cv2.undistort` behavior on the real dataset
  (vs. the synthetic test fixture) is a genuine unknown that can only be confirmed on a real Colab
  run against real HCM imagery — flagged in Task 4's docstring, not hidden, same pattern as the
  colab-runner plan's GPU-only items.
- **Type consistency:** `undistort_scene(scene, output_dir) -> SceneConfig` (Task 4) matches how
  Task 5 calls it and feeds its output into `build_filtered_scene(working_scene, ...)`, which
  already expects a `SceneConfig` (verified against its existing signature in
  `src/training/holdout_scene.py`). `write_cameras_binary(cameras, path)` (Task 3) is called by
  Task 4 with `_PinholeCamera` objects exposing exactly the `.model`/`.width`/`.height`/`.params`
  attributes the writer expects. `camera_focal_lengths(model, params)` (Task 2) is called by
  `_camera_params_for_holdout` with `camera.model`/`camera.params` from the same `Camera`
  namedtuple already used everywhere else in this codebase (`scene/colmap_loader.py`).
