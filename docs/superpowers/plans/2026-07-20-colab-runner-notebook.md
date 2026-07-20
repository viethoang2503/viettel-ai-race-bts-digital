# Colab Runner Notebook + Review Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix 3 review findings (fail-closed submission zip, empty-holdout guard, broken CUDA
extension cache) and build the real GPU glue + `notebooks/colab_runner.ipynb` so a user can open
one notebook on Colab and go from "repo + Drive dataset" to `submission.zip`.

**Architecture:** `src/orchestrator/run_pipeline.py::run_baseline_pipeline` already exists and
takes `train_fn`/`render_fn` as injected callables (tested with fakes, no GPU needed for that
test). This plan adds the real implementations — `src/training/gs_train_fn.py` (subprocess
wrapper around vendored `train.py`, resume-safe) and `src/rendering/gs_render_fn.py` (in-process
call into vendored `gaussian_renderer.render()`, reusing the existing `render_from_csv.render_all`
writer) — plus fixes the two logic bugs and the CUDA-cache bug those real implementations depend
on, then wires everything into one notebook.

**Tech Stack:** Python 3.10+, PyTorch (CPU locally, CUDA on Colab), pytest, vendored
`third_party/gaussian-splatting` (git submodule).

## Global Constraints

- Local machine has no GPU/CUDA — any code that imports `third_party/gaussian-splatting`'s CUDA
  extensions or calls `gaussian_renderer.render()` cannot be unit-tested locally. Such code must
  still be written in full (no placeholders), with a manual Colab verification checklist, exactly
  like the existing `environment/setup_colab.sh` (Task 13 of
  `docs/superpowers/plans/2026-07-18-core-nvs-pipeline.md`) already does.
- Repo root: `/home/howard/Documents/viettel ai race/computer vision`.
- `output_root` for any real pipeline run must live on Google Drive (`/content/drive/MyDrive/...`),
  never `/content` — Colab sessions can disconnect at any time.
- Do not break existing tests: `tests/test_run_pipeline.py`, `tests/test_make_holdout_split.py`,
  `tests/test_train_wrapper.py`.
- `SceneConfig`, `CameraParams`, `PipelineResult` field names/types are fixed by existing code —
  reuse them as-is, do not rename.

---

### Task 1: Fix fail-closed bug in `run_baseline_pipeline` (Medium finding)

**Files:**
- Modify: `src/orchestrator/run_pipeline.py:160-164`
- Test: `tests/test_run_pipeline.py`

**Interfaces:**
- Consumes: existing `run_baseline_pipeline` signature, unchanged.
- Produces: `result.submission_zip` is `None` whenever `result.validation_problems` is non-empty,
  even though `package_submission` already wrote the zip to disk (kept for debugging, just not
  exposed as a valid submission).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_run_pipeline.py` (after the existing `_fake_render_fn` and before the first
test function):

```python
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
```

Add this test function at the end of the file:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_run_pipeline.py::test_run_baseline_pipeline_withholds_zip_when_validation_finds_problems -v`
Expected: FAIL — `assert result.submission_zip is None` fails because current code always sets it.

- [ ] **Step 3: Fix `src/orchestrator/run_pipeline.py`**

Replace lines 160-164 (the current tail of `run_baseline_pipeline`):

```python
    submission_zip = output_root / "submission.zip"
    package_submission(scene_render_dirs, submission_zip)
    result.validation_problems = validate_submission(submission_zip, scenes)
    result.submission_zip = submission_zip
    return result
```

with:

```python
    submission_zip = output_root / "submission.zip"
    package_submission(scene_render_dirs, submission_zip)
    result.validation_problems = validate_submission(submission_zip, scenes)
    # Fail closed: an artifact validate_submission has already flagged as
    # wrong (bad size, missing/extra file) must never be exposed as a
    # valid submission, even though the zip stays on disk for debugging —
    # same philosophy already applied above for skipped_scenes.
    result.submission_zip = submission_zip if not result.validation_problems else None
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_run_pipeline.py -v`
Expected: PASS (5 passed) — the new test passes, and the existing
`test_run_baseline_pipeline_produces_scores_and_valid_zip` still passes because its fake render
output has no validation problems.

- [ ] **Step 5: Commit**

```bash
git add src/orchestrator/run_pipeline.py tests/test_run_pipeline.py
git commit -m "Fix orchestrator to fail closed when submission validation finds problems"
```

---

### Task 2: Guard empty input in `select_holdout_images` (Low finding)

**Files:**
- Modify: `src/evaluation/make_holdout_split.py:16`
- Test: `tests/test_make_holdout_split.py`

**Interfaces:**
- Produces: `select_holdout_images({}, holdout_ratio=...)` raises `ValueError` with a clear
  message instead of crashing inside `np.stack`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_make_holdout_split.py`:

```python
import pytest


def test_select_holdout_raises_on_empty_input():
    with pytest.raises(ValueError, match="empty"):
        select_holdout_images({}, holdout_ratio=0.125)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_make_holdout_split.py::test_select_holdout_raises_on_empty_input -v`
Expected: FAIL — currently raises a numpy error from `np.stack`, not `ValueError` with "empty" in
the message (or the match fails / wrong exception type).

- [ ] **Step 3: Fix `src/evaluation/make_holdout_split.py`**

Add a guard as the first line of the function body (right after the docstring, before
`names = list(...)`):

```python
def select_holdout_images(
    camera_centers: dict[str, np.ndarray], holdout_ratio: float = 0.125,
) -> list[str]:
    """Pick the images whose camera center is farthest from the centroid of
    all camera centers in the scene — approximates the edge-of-coverage
    poses that real test poses are likely to extrapolate toward, rather
    than a uniform every-Nth split.
    """
    if not camera_centers:
        raise ValueError("camera_centers is empty, cannot select holdout images")

    names = list(camera_centers.keys())
    centers = np.stack([camera_centers[n] for n in names], axis=0)
    centroid = centers.mean(axis=0)
    distances = np.linalg.norm(centers - centroid, axis=1)

    n_holdout = max(1, math.floor(len(names) * holdout_ratio))
    order = np.argsort(-distances)  # descending distance
    return [names[i] for i in order[:n_holdout]]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_make_holdout_split.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/evaluation/make_holdout_split.py tests/test_make_holdout_split.py
git commit -m "Raise clear error instead of crashing on empty holdout input"
```

---

### Task 3: Add `checkpoint_iteration()` helper to `train_wrapper.py`

**Files:**
- Modify: `src/training/train_wrapper.py`
- Test: `tests/test_train_wrapper.py`

**Interfaces:**
- Produces: `checkpoint_iteration(path: Path) -> int | None` — parses the iteration number out of
  a `chkpntNNNN.pth` filename, `None` if it doesn't match. Task 4's `gs_train_fn.py` depends on
  this to decide whether an existing checkpoint already meets the target iteration count.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_train_wrapper.py`:

```python
from src.training.train_wrapper import build_train_argv, checkpoint_iteration, find_latest_checkpoint


def test_checkpoint_iteration_parses_number():
    assert checkpoint_iteration(Path("outputs/chair/chkpnt15000.pth")) == 15000


def test_checkpoint_iteration_returns_none_for_non_matching_name():
    assert checkpoint_iteration(Path("outputs/chair/model_final.pth")) is None
```

(This replaces the existing `from src.training.train_wrapper import build_train_argv,
find_latest_checkpoint` import line at the top of the file — just add `checkpoint_iteration` to
that same import.)

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_train_wrapper.py::test_checkpoint_iteration_parses_number -v`
Expected: FAIL — `ImportError: cannot import name 'checkpoint_iteration'`.

- [ ] **Step 3: Add the function to `src/training/train_wrapper.py`**

Add this function right after `find_latest_checkpoint` (which already uses `_CHECKPOINT_RE`):

```python
def checkpoint_iteration(path: Path) -> int | None:
    match = _CHECKPOINT_RE.search(Path(path).name)
    return int(match.group(1)) if match else None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_train_wrapper.py -v`
Expected: PASS (6 passed).

- [ ] **Step 5: Commit**

```bash
git add src/training/train_wrapper.py tests/test_train_wrapper.py
git commit -m "Add checkpoint_iteration helper for resume-safe training"
```

---

### Task 4: Real resume-safe `train_fn` (`src/training/gs_train_fn.py`)

**Files:**
- Create: `src/training/gs_train_fn.py`
- Test: `tests/test_gs_train_fn.py`

**Interfaces:**
- Consumes: `build_train_argv`, `find_latest_checkpoint`, `checkpoint_iteration` (Task 3),
  `SceneConfig` (existing).
- Produces: `real_train_fn(scene: SceneConfig, output_dir: Path, iterations: int = 30000) -> Path`
  — matches the `train_fn(scene, output_dir) -> Path` contract `run_baseline_pipeline` already
  calls (the notebook binds `iterations` via `functools.partial` if a non-default value is
  needed). GPU-free logic (resume/skip decision, argv construction) is fully unit-tested here by
  mocking `subprocess.run`; the actual `train.py` execution is GPU-only and only verified manually
  on Colab (Task 7's checklist).

- [ ] **Step 1: Write the failing tests**

Create `tests/test_gs_train_fn.py`:

```python
from pathlib import Path

from src.common.config import SceneConfig
from src.training import gs_train_fn


def _chair_scene():
    return SceneConfig(
        name="chair",
        root=Path("VAI_NVS_DATA_ROUND2/chair"),
        train_images_dir=Path("VAI_NVS_DATA_ROUND2/chair/train/images"),
        sparse_dir=Path("VAI_NVS_DATA_ROUND2/chair/train/sparse/0"),
        test_poses_csv=Path("VAI_NVS_DATA_ROUND2/chair/test/test_poses.csv"),
        submission_dir="chair",
    )


def test_real_train_fn_skips_subprocess_when_checkpoint_already_at_target(tmp_path, monkeypatch):
    (tmp_path / "chkpnt30000.pth").touch()
    calls = []
    monkeypatch.setattr(gs_train_fn.subprocess, "run", lambda *a, **k: calls.append((a, k)))

    result = gs_train_fn.real_train_fn(_chair_scene(), tmp_path, iterations=30000)

    assert calls == []
    assert result == tmp_path / "chkpnt30000.pth"


def test_real_train_fn_runs_subprocess_from_scratch_when_no_checkpoint(tmp_path, monkeypatch):
    calls = []

    def fake_run(argv, cwd, check):
        calls.append((argv, cwd, check))
        (tmp_path / "chkpnt30000.pth").touch()

    monkeypatch.setattr(gs_train_fn.subprocess, "run", fake_run)

    result = gs_train_fn.real_train_fn(_chair_scene(), tmp_path, iterations=30000)

    assert len(calls) == 1
    argv, cwd, check = calls[0]
    assert cwd == "third_party/gaussian-splatting"
    assert check is True
    assert "--start_checkpoint" not in argv
    assert "--checkpoint_iterations" in argv
    assert argv[argv.index("--checkpoint_iterations") + 1] == "30000"
    assert result == tmp_path / "chkpnt30000.pth"


def test_real_train_fn_resumes_from_partial_checkpoint(tmp_path, monkeypatch):
    partial = tmp_path / "chkpnt15000.pth"
    partial.touch()
    calls = []

    def fake_run(argv, cwd, check):
        calls.append(argv)
        (tmp_path / "chkpnt30000.pth").touch()

    monkeypatch.setattr(gs_train_fn.subprocess, "run", fake_run)

    result = gs_train_fn.real_train_fn(_chair_scene(), tmp_path, iterations=30000)

    argv = calls[0]
    assert "--start_checkpoint" in argv
    assert str(partial.resolve()) == argv[argv.index("--start_checkpoint") + 1]
    assert result == tmp_path / "chkpnt30000.pth"


def test_real_train_fn_raises_if_subprocess_produces_no_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(gs_train_fn.subprocess, "run", lambda *a, **k: None)

    try:
        gs_train_fn.real_train_fn(_chair_scene(), tmp_path, iterations=30000)
        assert False, "expected RuntimeError"
    except RuntimeError as e:
        assert "30000" in str(e)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_gs_train_fn.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.training.gs_train_fn'`.

- [ ] **Step 3: Write `src/training/gs_train_fn.py`**

```python
from __future__ import annotations

import subprocess
from pathlib import Path

from src.common.config import SceneConfig
from src.training.train_wrapper import build_train_argv, checkpoint_iteration, find_latest_checkpoint


def real_train_fn(scene: SceneConfig, output_dir: Path, iterations: int = 30000) -> Path:
    """Resume-safe wrapper around the vendored train.py, invoked via
    subprocess so a Colab GPU is required only for the actual training —
    the resume/skip decision and argv construction are pure Python, tested
    without a GPU by mocking subprocess.run (see tests/test_gs_train_fn.py).

    If output_dir already has a checkpoint at `iterations`, training is
    skipped entirely — a Colab session that disconnects and gets re-run
    from the top of the notebook must not burn GPU time re-training a
    scene that already finished.
    """
    output_dir = Path(output_dir)
    existing = find_latest_checkpoint(output_dir)
    if existing is not None and (checkpoint_iteration(existing) or 0) >= iterations:
        return existing

    argv = build_train_argv(
        scene, output_dir, iterations,
        resume_checkpoint=existing,
        extra_args=["--checkpoint_iterations", str(iterations)],
    )
    subprocess.run(argv, cwd="third_party/gaussian-splatting", check=True)

    checkpoint = find_latest_checkpoint(output_dir)
    if checkpoint is None or (checkpoint_iteration(checkpoint) or 0) < iterations:
        raise RuntimeError(
            f"train.py finished but no checkpoint at target iteration {iterations} "
            f"found in {output_dir}"
        )
    return checkpoint
```

Create `src/training/__init__.py` if it does not already exist (it already does — no action
needed, this file already exists per the current repo state).

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_gs_train_fn.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/training/gs_train_fn.py tests/test_gs_train_fn.py
git commit -m "Add resume-safe real train_fn wrapping vendored train.py"
```

---

### Task 5: Real `render_fn` (`src/rendering/gs_render_fn.py`)

**Files:**
- Create: `src/rendering/gs_render_fn.py`
- Test: `tests/test_gs_render_fn.py`

**Interfaces:**
- Consumes: `render_from_csv.render_all` (existing, Task 9 of the core plan), `CameraParams`
  (existing).
- Produces: `real_render_fn(checkpoint: Path, params_list: list[CameraParams], output_dir: Path)
  -> list[Path]` — matches the `render_fn(checkpoint, params_list, output_dir)` contract
  `run_baseline_pipeline` already calls. Two pure helpers, `_placeholder_image(width, height)` and
  `_tensor_to_uint8_image(tensor)`, are split out and unit-tested locally with CPU tensors (no
  CUDA needed for these two). The rest of `real_render_fn` — loading the checkpoint into a real
  `GaussianModel` and calling the vendored `gaussian_renderer.render()` — imports CUDA-only
  vendored modules and can only be verified manually on Colab (Task 7's checklist); this is the
  same category of risk `environment/setup_colab.sh` already carries in this codebase, not new.

**Verified from source (not guessed):** `third_party/gaussian-splatting/scene/gaussian_model.py`
`GaussianModel.restore(self, model_args, training_args)` calls
`self.training_setup(training_args)`, so `training_args` must be a real `OptimizationParams`-like
object, not `None`. `third_party/gaussian-splatting/arguments/__init__.py` shows the correct way
to build one: `OptimizationParams(parser)`/`PipelineParams(parser)` register argparse defaults on
a fresh `ArgumentParser`, then `.extract(parser.parse_args([]))` produces the actual params object
with default values. `scene/cameras.py`'s `Camera.__init__` takes `(resolution, colmap_id, R, T,
FoVx, FoVy, depth_params, image, invdepthmap, image_name, uid, ...)` — `image` is only used to
determine pixel dimensions via `PILtoTorch`, so a blank placeholder image of the target size is
correct for rendering a novel test pose with no ground truth.

- [ ] **Step 1: Write the failing tests for the pure helpers**

Create `tests/test_gs_render_fn.py`:

```python
import numpy as np
import torch

from src.rendering.gs_render_fn import _placeholder_image, _tensor_to_uint8_image


def test_placeholder_image_has_requested_size():
    img = _placeholder_image(width=64, height=48)
    assert img.size == (64, 48)
    assert img.mode == "RGB"


def test_tensor_to_uint8_image_converts_chw_float_to_hwc_uint8():
    # 3x2x2 CHW tensor: red channel at max, others at 0.
    tensor = torch.zeros(3, 2, 2)
    tensor[0, :, :] = 1.0

    array = _tensor_to_uint8_image(tensor)

    assert array.shape == (2, 2, 3)
    assert array.dtype == np.uint8
    assert (array[:, :, 0] == 255).all()
    assert (array[:, :, 1] == 0).all()
    assert (array[:, :, 2] == 0).all()


def test_tensor_to_uint8_image_clamps_out_of_range_values():
    tensor = torch.full((3, 1, 1), 2.0)  # out of [0, 1] range
    array = _tensor_to_uint8_image(tensor)
    assert (array == 255).all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_gs_render_fn.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.rendering.gs_render_fn'`.

- [ ] **Step 3: Write `src/rendering/gs_render_fn.py`**

```python
from __future__ import annotations

import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.common.pose_utils import CameraParams
from src.rendering.render_from_csv import render_all

_GS_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "gaussian-splatting"
if str(_GS_ROOT) not in sys.path:
    sys.path.insert(0, str(_GS_ROOT))


def _placeholder_image(width: int, height: int) -> Image.Image:
    """Blank image only used to give Camera.__init__ a pixel size to
    resize to (via PILtoTorch) — novel test poses have no ground-truth
    image, so its content is irrelevant, only its (width, height)."""
    return Image.new("RGB", (width, height), color=(0, 0, 0))


def _tensor_to_uint8_image(tensor: torch.Tensor) -> np.ndarray:
    """Convert gaussian_renderer.render()'s CHW float tensor in [0, 1]
    (values can exceed this range slightly, hence the clamp) to the HWC
    uint8 array render_from_csv.render_all expects."""
    array = tensor.detach().clamp(0.0, 1.0).cpu().numpy()
    array = np.transpose(array, (1, 2, 0))
    return (array * 255.0).round().astype(np.uint8)


def _default_opt_and_pipe():
    from arguments import OptimizationParams, PipelineParams

    parser = ArgumentParser()
    opt_group = OptimizationParams(parser)
    pipe_group = PipelineParams(parser)
    args = parser.parse_args([])
    return opt_group.extract(args), pipe_group.extract(args)


def _load_gaussians(checkpoint_path: Path):
    from scene.gaussian_model import GaussianModel

    model_args, _first_iter = torch.load(checkpoint_path)
    opt, _pipe = _default_opt_and_pipe()
    gaussians = GaussianModel(sh_degree=3)  # matches ModelParams default
    gaussians.restore(model_args, opt)
    return gaussians


def real_render_fn(
    checkpoint: Path, params_list: list[CameraParams], output_dir: Path,
) -> list[Path]:
    """GPU-only: loads the trained GaussianModel and renders every camera
    in params_list via the vendored gaussian_renderer.render(). Manual
    Colab verification only (no CUDA locally) — see
    docs/superpowers/plans/2026-07-20-colab-runner-notebook.md Task 7.
    """
    from gaussian_renderer import render as gs_render
    from scene.cameras import Camera

    gaussians = _load_gaussians(checkpoint)
    _opt, pipe = _default_opt_and_pipe()
    background = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda")

    def _render_one(params: CameraParams, gaussians) -> np.ndarray:
        camera = Camera(
            resolution=(params.width, params.height),
            colmap_id=0,
            R=params.R,
            T=params.T,
            FoVx=params.fov_x,
            FoVy=params.fov_y,
            depth_params=None,
            image=_placeholder_image(params.width, params.height),
            invdepthmap=None,
            image_name=params.image_name,
            uid=0,
        )
        rendered = gs_render(camera, gaussians, pipe, background)["render"]
        return _tensor_to_uint8_image(rendered)

    return render_all(
        checkpoint, None, output_dir, _render_one,
        params_list=params_list, gaussians=gaussians,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_gs_render_fn.py -v`
Expected: PASS (3 passed) — only the two pure helpers are exercised; `real_render_fn` itself is
not called by any local test since it requires CUDA.

- [ ] **Step 5: Commit**

```bash
git add src/rendering/gs_render_fn.py tests/test_gs_render_fn.py
git commit -m "Add real render_fn wrapping vendored gaussian_renderer"
```

---

### Task 6: Fix CUDA extension cache in `environment/setup_colab.sh` (High finding)

**Files:**
- Modify: `environment/setup_colab.sh`

No automated test is possible locally (no CUDA device). This task produces the fixed script plus
an updated manual verification checklist.

- [ ] **Step 1: Replace `environment/setup_colab.sh` in full**

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
  local py_name="${ext_name//-/_}"
  local cache_dir="$CUDA_EXT_CACHE/$ext_name"
  local cache_marker="$cache_dir.built"

  if [ -f "$cache_marker" ]; then
    echo "== Restoring cached $ext_name build =="
    cp -r "$cache_dir"/. "$SITE_PACKAGES/"
  else
    echo "== Building $ext_name from source (first run, slow) =="
    pip install -q "$ext_src_dir"
    # Copy the actual compiled artifacts (the .so plus install metadata)
    # out of site-packages into the Drive cache, instead of the old
    # `pip download --no-binary :all:` approach — these two submodules
    # don't package a round-trippable sdist, so that download silently
    # produced nothing to restore from on the next session.
    rm -rf "$cache_dir"
    mkdir -p "$cache_dir"
    cp -r "$SITE_PACKAGES/${py_name}"* "$cache_dir/"
    # Only mark the cache built after a successful copy — no more
    # `|| true` masking a failed/empty cache write.
    touch "$cache_marker"
  fi
}

restore_or_build diff-gaussian-rasterization
restore_or_build simple-knn

echo "== Setup complete =="
python3 -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

- [ ] **Step 2: Manual verification checklist (run on a fresh Colab GPU runtime)**

1. `!git clone --recurse-submodules <repo-url> && cd <repo-dir>`.
2. Run `!bash environment/setup_colab.sh`.
3. Confirm output ends with `CUDA available: True`, and confirm both `restore_or_build` calls took
   the "Building ... from source" branch (first run, no cache yet).
4. Inspect `$CUDA_EXT_CACHE` on Drive — confirm `diff-gaussian-rasterization/` and `simple-knn/`
   subdirectories exist and are non-empty, and both `.built` marker files exist.
5. Disconnect the runtime, reconnect, re-run `!bash environment/setup_colab.sh` — confirm the
   second run takes the "Restoring cached" branch for both extensions, completes in well under a
   minute for that step, and `import diff_gaussian_rasterization` / `import simple_knn` both
   succeed in a fresh Python cell afterward.
6. If step 5 fails (e.g. the glob in `restore_or_build` doesn't capture every file pip actually
   installed for one of the two packages — different Colab base images can differ here), inspect
   `$SITE_PACKAGES` for the exact set of `${py_name}*` entries pip created and adjust the glob
   accordingly — this is the one genuinely unverifiable-without-CUDA detail flagged in the design
   spec section 4.

- [ ] **Step 3: Commit**

```bash
git add environment/setup_colab.sh
git commit -m "Fix CUDA extension cache to copy built artifacts instead of broken sdist download"
```

---

### Task 7: `notebooks/colab_runner.ipynb`

**Files:**
- Create: `notebooks/colab_runner.ipynb`

No automated test is possible locally (requires Colab + CUDA + mounted Drive dataset). This task
produces the notebook plus a manual verification checklist.

- [ ] **Step 1: Create `notebooks/colab_runner.ipynb`**

```json
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# VAR 2026 Colab runner\n",
    "Run every cell top to bottom. Safe to re-run after a disconnect: cell 1 skips re-cloning if the repo is already present, cell 3's `real_train_fn` skips scenes that already finished training, and all output lives on Drive."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import os\n",
    "\n",
    "REPO_URL = \"https://github.com/viethoang2503/viettel-ai-race-bts-digital.git\"\n",
    "REPO_DIR = \"/content/viettel-ai-race-bts-digital\"\n",
    "\n",
    "if not os.path.isdir(REPO_DIR):\n",
    "    !git clone --recurse-submodules {REPO_URL} {REPO_DIR}\n",
    "%cd {REPO_DIR}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "!bash environment/setup_colab.sh"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import functools\n",
    "from pathlib import Path\n",
    "\n",
    "from src.common.config import load_scenes\n",
    "from src.evaluation.compute_metrics import load_lpips_model\n",
    "from src.orchestrator.run_pipeline import run_baseline_pipeline\n",
    "from src.rendering.gs_render_fn import real_render_fn\n",
    "from src.training.gs_train_fn import real_train_fn\n",
    "\n",
    "ITERATIONS = 30000\n",
    "OUTPUT_ROOT = Path(\"/content/drive/MyDrive/var2026/outputs\")\n",
    "\n",
    "scenes = load_scenes()\n",
    "train_fn = functools.partial(real_train_fn, iterations=ITERATIONS)\n",
    "\n",
    "result = run_baseline_pipeline(\n",
    "    scenes=scenes,\n",
    "    train_fn=train_fn,\n",
    "    render_fn=real_render_fn,\n",
    "    lpips_model=load_lpips_model(),\n",
    "    psnr_max=30.0,\n",
    "    output_root=OUTPUT_ROOT,\n",
    ")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(\"Per-scene scores:\")\n",
    "for name, score in result.per_scene_scores.items():\n",
    "    print(f\"  {name}: {score:.4f}\")\n",
    "\n",
    "if result.skipped_scenes:\n",
    "    print(\"\\nSkipped scenes:\")\n",
    "    for name, problems in result.skipped_scenes.items():\n",
    "        print(f\"  {name}: {problems}\")\n",
    "\n",
    "if result.validation_problems:\n",
    "    print(\"\\nValidation problems (submission withheld):\")\n",
    "    for problem in result.validation_problems:\n",
    "        print(f\"  {problem}\")\n",
    "\n",
    "print(f\"\\nsubmission_zip: {result.submission_zip}\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "name": "python3"
  },
  "language_info": {
   "name": "python"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
```

Note: `load_lpips_model` is confirmed to exist at `src/evaluation/compute_metrics.py:47` — this is
the same loader `run_baseline_pipeline`'s docstring already refers to as the real caller's LPIPS
source.

- [ ] **Step 2: Manual verification checklist (run on a fresh Colab GPU runtime, after Task 6)**

1. Upload nothing manually — open this notebook directly in Colab (from GitHub or a copy placed
   on Drive) and run all cells top to bottom.
2. Confirm cell 1 clones the repo and `%cd`s into it without error.
3. Confirm cell 2 ends with `CUDA available: True`.
4. Before running cell 3 against all 7 scenes, temporarily edit `scenes = load_scenes()` to
   `scenes = load_scenes()[-1:]` (just `bonsai`, the public Mip-NeRF360 reference scene) for the
   first end-to-end run, since it has a published benchmark to sanity-check against (spec section
   10). Confirm a `submission.zip` is produced and `result.per_scene_scores["bonsai"]` is in a
   plausible range before running the full 7-scene set.
5. Disconnect and reconnect mid-run once on a real scene; confirm re-running cell 3 skips
   already-finished training phases (via `real_train_fn`'s resume logic, Task 4) instead of
   restarting from scratch.

- [ ] **Step 3: Commit**

```bash
git add notebooks/colab_runner.ipynb
git commit -m "Add Colab runner notebook wiring real train/render into the baseline pipeline"
```

---

## Self-Review Summary

- **Spec coverage:** Design spec `docs/superpowers/specs/2026-07-20-colab-runner-notebook-design.md`
  section 3.1 (fail-closed submission) → Task 1. Section 3.2 (empty holdout guard) → Task 2.
  Section 4 (CUDA cache fix) → Task 6. Section 5.1 (`real_train_fn`) → Tasks 3-4. Section 5.2
  (`real_render_fn`) → Task 5. Section 6 (notebook) → Task 7. Section 7 (testing split between
  local-testable and manual-Colab-verify) is reflected in every task's Interfaces block.
- **Placeholder scan:** no TBD/TODO. The two genuinely-unverifiable-without-CUDA items (exact
  `site-packages` glob in Task 6, exact `GaussianModel.restore`/`Camera` wiring in Task 5) are
  written as concrete code plus an explicit manual-verification checklist step, not left vague —
  same pattern the existing codebase already uses in Task 13 of the core plan.
- **Type consistency:** `real_train_fn(scene, output_dir) -> Path` and `real_render_fn(checkpoint,
  params_list, output_dir) -> list[Path]` match the `train_fn`/`render_fn` contracts
  `run_baseline_pipeline` already defines (verified against `src/orchestrator/run_pipeline.py`'s
  actual calls: `train_fn(filtered_scene, scene_output / "eval_train")` and
  `render_fn(eval_checkpoint, holdout_params, holdout_render_dir)`). `checkpoint_iteration` (Task
  3) and `find_latest_checkpoint`/`build_train_argv` (existing) are used with matching signatures
  in Task 4. `CameraParams` fields (`image_name, R, T, fov_x, fov_y, width, height`) used in Task
  5 match their definition already used identically in `run_pipeline.py`'s
  `_camera_params_for_holdout`.
