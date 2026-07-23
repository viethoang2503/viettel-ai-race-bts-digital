# Score Push (Plan 2 Execution) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement everything needed to push the 7-scene baseline (Score ≈0.622) toward ≈0.80
within the remaining 7-day window: the advanced-techniques building blocks (VRAM guard, floater
pruning, depth regularization, appearance embedding, auto-select, reproducibility bundle) plus
the diagnosis step, screening/final-iteration split, and bounded per-scene hyperparameter search
defined in `docs/superpowers/specs/2026-07-23-score-push-design.md`.

**Architecture:** Pure-Python/pure-math pieces (VRAM estimation, floater mask computation, sparse
depth target computation, appearance-embedding tensor math, depth-loss tensor math, candidate
selection, diagnosis ranking, tie-break decision, hyperparameter candidate construction,
reproducibility bundle) get real `pytest` unit tests on the local no-GPU machine. The pieces that
require the actual differentiable CUDA rasterizer (the training loop itself, the checkpoint prune
wrapper, the experiment-matrix orchestrator's GPU-touching calls) are written completely against
the vendored API as documented in the upstream repo, but flagged for manual verification on
Colab, since this local machine cannot execute CUDA code to verify them directly.

**Tech Stack:** Python 3.10+, PyTorch, `lpips`, `scikit-image`, `numpy`, `pyyaml`, `pytest`, the
vendored `graphdeco-inria/gaussian-splatting` submodule at `third_party/gaussian-splatting`.

**Note on provenance:** this plan supersedes and fully absorbs
`docs/superpowers/plans/2026-07-18-advanced-techniques.md` — Tasks 1-8 and 10 below carry that
document's already-reviewed design over verbatim (or with a small, explicitly-noted correction).
Do not execute `2026-07-18-advanced-techniques.md` directly; execute this document instead. Its
own Task 11 (visual QA) is the one piece already implemented — `src/submission/visual_qa.py`, in
use since Plan 1 — and is not repeated here.

## Global Constraints

- Deadline: 30/07/2026. GPU: Google Colab Pro, L4, one session at a time (no parallel-session
  dependency). GPU cost is not a constraint; calendar time is.
- Tests: run with `.venv/bin/python -m pytest -q` (system `python3` lacks `pytest`).
- Submission format constraint (exam spec 1.5): `submission.zip` must contain **only** per-scene
  rendered images — reproducibility artifacts (config YAML, score CSVs) must be written to a
  **separate** zip/folder, never merged into `submission.zip`.
- Real BTS scenes are `SIMPLE_RADIAL`; the vendored 3DGS only handles `PINHOLE`/`SIMPLE_PINHOLE`
  — every task that trains or renders must go through Plan 1's `undistort_scene` first, never the
  raw `SceneConfig`.

---

### Task 1: VRAM budget guard

**Files:**
- Create: `src/postprocess/vram_guard.py`
- Test: `tests/test_vram_guard.py`

**Interfaces:**
- Produces:
  - `count_gaussians_in_ply(ply_path: Path) -> int` — parses the ASCII PLY header (`element
    vertex N`) without loading the (potentially large) binary body.
  - `estimate_vram_bytes(num_gaussians: int, sh_degree: int = 3, dtype_bytes: int = 4) -> int`
    — a **conservative heuristic**, not a guaranteed prediction; the authoritative check is
    always `torch.cuda.max_memory_allocated()` measured for real on Colab (Step 6).
  - `fits_within_vram_budget(ply_path: Path, budget_bytes: int) -> bool`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_vram_guard.py
from pathlib import Path

import pytest

from src.postprocess.vram_guard import (
    count_gaussians_in_ply,
    estimate_vram_bytes,
    fits_within_vram_budget,
)


def _write_fake_ply(path: Path, num_vertices: int) -> None:
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {num_vertices}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "end_header\n"
    )
    path.write_bytes(header.encode("ascii") + b"\x00" * (num_vertices * 12))


def test_count_gaussians_in_ply_reads_header_only(tmp_path):
    ply_path = tmp_path / "cloud.ply"
    _write_fake_ply(ply_path, 12345)
    assert count_gaussians_in_ply(ply_path) == 12345


def test_estimate_vram_bytes_scales_linearly_with_count():
    small = estimate_vram_bytes(1000)
    large = estimate_vram_bytes(2000)
    assert large == pytest.approx(2 * small, rel=1e-6)
    assert small > 0


def test_estimate_vram_bytes_increases_with_sh_degree():
    low_degree = estimate_vram_bytes(1000, sh_degree=0)
    high_degree = estimate_vram_bytes(1000, sh_degree=3)
    assert high_degree > low_degree


def test_fits_within_vram_budget_true_for_small_cloud(tmp_path):
    ply_path = tmp_path / "small.ply"
    _write_fake_ply(ply_path, 1000)
    assert fits_within_vram_budget(ply_path, budget_bytes=16 * 1024**3) is True


def test_fits_within_vram_budget_false_for_absurdly_large_cloud(tmp_path):
    ply_path = tmp_path / "huge.ply"
    _write_fake_ply(ply_path, 1)  # header claims 1, we lie about the count instead:
    ply_path.write_bytes(
        ply_path.read_bytes().replace(b"element vertex 1\n", b"element vertex 500000000\n")
    )
    assert fits_within_vram_budget(ply_path, budget_bytes=16 * 1024**3) is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_vram_guard.py -v`
Expected: `FAIL` — `ModuleNotFoundError: No module named 'src.postprocess'`.

- [ ] **Step 3: Write `src/postprocess/vram_guard.py`**

```python
from __future__ import annotations

from pathlib import Path


def count_gaussians_in_ply(ply_path: Path) -> int:
    with open(ply_path, "rb") as f:
        for raw_line in f:
            line = raw_line.decode("ascii", errors="ignore").strip()
            if line.startswith("element vertex"):
                return int(line.split()[-1])
            if line == "end_header":
                break
    raise ValueError(f"no 'element vertex' line found in PLY header: {ply_path}")


def estimate_vram_bytes(num_gaussians: int, sh_degree: int = 3, dtype_bytes: int = 4) -> int:
    """Conservative heuristic estimate of VRAM needed to RENDER (not train)
    a checkpoint with this many Gaussians at the given SH degree.
    """
    floats_per_gaussian = 3 + 4 + 3 + 1 + (sh_degree + 1) ** 2 * 3
    rendering_overhead_multiplier = 2.0
    return int(num_gaussians * floats_per_gaussian * dtype_bytes * rendering_overhead_multiplier)


def fits_within_vram_budget(ply_path: Path, budget_bytes: int, sh_degree: int = 3) -> bool:
    num_gaussians = count_gaussians_in_ply(ply_path)
    return estimate_vram_bytes(num_gaussians, sh_degree=sh_degree) <= budget_bytes
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_vram_guard.py -v`
Expected: `PASS` (5 passed).

- [ ] **Step 5: Commit**

```bash
git add src/postprocess/vram_guard.py src/postprocess/__init__.py tests/test_vram_guard.py
git commit -m "Add VRAM budget estimation guard for A4000 inference target"
```

- [ ] **Step 6: Manual verification on Colab (requires GPU)**

After rendering a real checkpoint (any Plan 1 scene), compare the heuristic against reality:

```python
import torch
from src.postprocess.vram_guard import count_gaussians_in_ply, estimate_vram_bytes

torch.cuda.reset_peak_memory_stats()
# ... run the real render for one scene ...
actual_bytes = torch.cuda.max_memory_allocated()
n = count_gaussians_in_ply("path/to/point_cloud.ply")
print("estimated:", estimate_vram_bytes(n), "actual:", actual_bytes)
```

If the heuristic underestimates actual usage by more than ~30%, tighten the
`rendering_overhead_multiplier` before relying on it in Task 6's auto-selection.

---

### Task 2: Floater/background prune mask

**Files:**
- Create: `src/postprocess/prune_floaters.py`
- Test: `tests/test_prune_floaters.py`

**Interfaces:**
- Consumes: `compute_scene_bbox` (Plan 1, `src/common/colmap_io.py`).
- Produces: `compute_prune_mask(xyz: np.ndarray, opacity: np.ndarray, scales: np.ndarray, bbox_min: np.ndarray, bbox_max: np.ndarray, opacity_threshold: float = 0.05, max_scale_percentile: float = 99.5) -> np.ndarray`
  returning a boolean `(N,)` keep-mask (`True` = keep). A Gaussian is pruned if its center is
  outside `[bbox_min, bbox_max]`, its opacity is below `opacity_threshold`, or its largest scale
  axis is above the `max_scale_percentile`-th percentile of all Gaussians' largest scale axis
  (percentile-based, not an absolute cutoff, so it doesn't need per-scene retuning).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_prune_floaters.py
import numpy as np

from src.postprocess.prune_floaters import compute_prune_mask


def test_keeps_normal_gaussians_inside_bbox():
    xyz = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
    opacity = np.array([0.5, 0.8])
    scales = np.array([[0.1, 0.1, 0.1], [0.1, 0.1, 0.1]])
    mask = compute_prune_mask(
        xyz, opacity, scales, bbox_min=np.array([-1, -1, -1]), bbox_max=np.array([1, 1, 1]),
    )
    assert mask.tolist() == [True, True]


def test_prunes_gaussian_outside_bbox():
    xyz = np.array([[0.0, 0.0, 0.0], [100.0, 100.0, 100.0]])
    opacity = np.array([0.5, 0.5])
    scales = np.array([[0.1, 0.1, 0.1], [0.1, 0.1, 0.1]])
    mask = compute_prune_mask(
        xyz, opacity, scales, bbox_min=np.array([-1, -1, -1]), bbox_max=np.array([1, 1, 1]),
    )
    assert mask.tolist() == [True, False]


def test_prunes_low_opacity_gaussian():
    xyz = np.array([[0.0, 0.0, 0.0], [0.1, 0.1, 0.1]])
    opacity = np.array([0.5, 0.01])
    scales = np.array([[0.1, 0.1, 0.1], [0.1, 0.1, 0.1]])
    mask = compute_prune_mask(
        xyz, opacity, scales, bbox_min=np.array([-1, -1, -1]), bbox_max=np.array([1, 1, 1]),
        opacity_threshold=0.05,
    )
    assert mask.tolist() == [True, False]


def test_prunes_outlier_scale_gaussian():
    xyz = np.tile(np.array([0.0, 0.0, 0.0]), (100, 1))
    opacity = np.full(100, 0.5)
    scales = np.full((100, 3), 0.1)
    scales[0] = [50.0, 50.0, 50.0]  # one giant outlier
    mask = compute_prune_mask(
        xyz, opacity, scales, bbox_min=np.array([-1, -1, -1]), bbox_max=np.array([1, 1, 1]),
        max_scale_percentile=99.5,
    )
    assert mask[0] == False
    assert mask[1:].all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_prune_floaters.py -v`
Expected: `FAIL` — `ModuleNotFoundError: No module named 'src.postprocess.prune_floaters'`.

- [ ] **Step 3: Write `src/postprocess/prune_floaters.py`**

```python
from __future__ import annotations

import numpy as np


def compute_prune_mask(
    xyz: np.ndarray, opacity: np.ndarray, scales: np.ndarray,
    bbox_min: np.ndarray, bbox_max: np.ndarray,
    opacity_threshold: float = 0.05, max_scale_percentile: float = 99.5,
) -> np.ndarray:
    inside_bbox = np.all((xyz >= bbox_min) & (xyz <= bbox_max), axis=1)
    opaque_enough = opacity >= opacity_threshold

    max_scale_per_gaussian = scales.max(axis=1)
    scale_cutoff = np.percentile(max_scale_per_gaussian, max_scale_percentile)
    not_outlier_scale = max_scale_per_gaussian <= scale_cutoff

    return inside_bbox & opaque_enough & not_outlier_scale
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_prune_floaters.py -v`
Expected: `PASS` (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/postprocess/prune_floaters.py tests/test_prune_floaters.py
git commit -m "Add floater/background prune mask computation"
```

---

### Task 3: Sparse depth targets from COLMAP tracks

**Files:**
- Create: `src/training/sparse_depth.py`
- Test: `tests/test_sparse_depth.py`

**Interfaces:**
- Produces: `compute_sparse_depth_targets(qvec: np.ndarray, tvec: np.ndarray, xys: np.ndarray, point3d_ids: np.ndarray, points3d: dict) -> tuple[np.ndarray, np.ndarray]`
  returning `(pixel_xy, depth)` arrays for every 2D keypoint in this training image that has a
  valid associated 3D point (`point3d_ids[i] != -1`). `pixel_xy` comes directly from the observed
  COLMAP keypoint location, not a re-projection. `depth` is the camera-space Z of that 3D point,
  computed using the **raw COLMAP world-to-camera** `(R, t)` — deliberately NOT the transposed
  convention used for the vendored `Camera` class, since depth is a camera-space quantity.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_sparse_depth.py
import numpy as np

from src.training.sparse_depth import compute_sparse_depth_targets


class _FakePoint3D:
    def __init__(self, xyz):
        self.xyz = np.array(xyz)


def test_identity_rotation_depth_equals_camera_z_translation():
    qvec = np.array([1.0, 0.0, 0.0, 0.0])  # identity rotation
    tvec = np.array([0.0, 0.0, 5.0])
    xys = np.array([[10.0, 20.0]])
    point3d_ids = np.array([0])
    points3d = {0: _FakePoint3D([0.0, 0.0, 0.0])}  # world origin

    pixel_xy, depth = compute_sparse_depth_targets(qvec, tvec, xys, point3d_ids, points3d)

    assert pixel_xy.shape == (1, 2)
    np.testing.assert_allclose(pixel_xy[0], [10.0, 20.0])
    np.testing.assert_allclose(depth, [5.0], atol=1e-10)


def test_filters_out_unassociated_keypoints():
    qvec = np.array([1.0, 0.0, 0.0, 0.0])
    tvec = np.array([0.0, 0.0, 5.0])
    xys = np.array([[10.0, 20.0], [30.0, 40.0]])
    point3d_ids = np.array([0, -1])  # second keypoint has no 3D point
    points3d = {0: _FakePoint3D([0.0, 0.0, 0.0])}

    pixel_xy, depth = compute_sparse_depth_targets(qvec, tvec, xys, point3d_ids, points3d)

    assert pixel_xy.shape == (1, 2)
    assert depth.shape == (1,)


def test_filters_out_point_ids_not_present_in_points3d_dict():
    qvec = np.array([1.0, 0.0, 0.0, 0.0])
    tvec = np.array([0.0, 0.0, 5.0])
    xys = np.array([[10.0, 20.0]])
    point3d_ids = np.array([999])  # not in points3d
    points3d = {0: _FakePoint3D([0.0, 0.0, 0.0])}

    pixel_xy, depth = compute_sparse_depth_targets(qvec, tvec, xys, point3d_ids, points3d)

    assert pixel_xy.shape == (0, 2)
    assert depth.shape == (0,)


def test_nonzero_translation_and_offset_point():
    qvec = np.array([1.0, 0.0, 0.0, 0.0])
    tvec = np.array([1.0, 2.0, 3.0])
    xys = np.array([[0.0, 0.0]])
    point3d_ids = np.array([0])
    points3d = {0: _FakePoint3D([0.0, 0.0, 2.0])}  # world point at z=2

    pixel_xy, depth = compute_sparse_depth_targets(qvec, tvec, xys, point3d_ids, points3d)

    np.testing.assert_allclose(depth, [5.0], atol=1e-10)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_sparse_depth.py -v`
Expected: `FAIL` — `ModuleNotFoundError: No module named 'src.training.sparse_depth'`.

- [ ] **Step 3: Write `src/training/sparse_depth.py`**

```python
from __future__ import annotations

import numpy as np

from src.common.pose_utils import qvec2rotmat


def compute_sparse_depth_targets(
    qvec: np.ndarray,
    tvec: np.ndarray,
    xys: np.ndarray,
    point3d_ids: np.ndarray,
    points3d: dict,
) -> tuple[np.ndarray, np.ndarray]:
    r_world_to_cam = qvec2rotmat(np.asarray(qvec, dtype=np.float64))
    t_world_to_cam = np.asarray(tvec, dtype=np.float64)

    valid = np.array([
        pid != -1 and pid in points3d for pid in point3d_ids
    ])

    pixel_xy = np.asarray(xys)[valid]
    valid_ids = np.asarray(point3d_ids)[valid]

    depths = []
    for pid in valid_ids:
        world_xyz = points3d[pid].xyz
        cam_xyz = r_world_to_cam @ world_xyz + t_world_to_cam
        depths.append(cam_xyz[2])

    return pixel_xy.reshape(-1, 2), np.array(depths, dtype=np.float64)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_sparse_depth.py -v`
Expected: `PASS` (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/training/sparse_depth.py tests/test_sparse_depth.py
git commit -m "Add sparse depth target computation from COLMAP 2D-3D tracks"
```

---

### Task 4: Appearance embedding (pure tensor math)

**Files:**
- Create: `src/training/appearance_embedding.py`
- Test: `tests/test_appearance_embedding.py`

**Interfaces:**
- Produces:
  - `AppearanceEmbedding(nn.Module)` — one learnable `(3,3)` affine + `(3,)` bias per training
    image, initialized to identity/zero so training starts as a no-op.
  - `apply_appearance(rgb: torch.Tensor, affine: torch.Tensor, bias: torch.Tensor) -> torch.Tensor`
    — `rgb` is `(3, H, W)` in `[0,1]`; applies `affine @ rgb_pixel + bias` per pixel, clamped
    back to `[0,1]`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_appearance_embedding.py
import torch

from src.training.appearance_embedding import AppearanceEmbedding, apply_appearance


def test_identity_affine_and_zero_bias_is_a_no_op():
    rgb = torch.rand(3, 4, 4)
    affine = torch.eye(3)
    bias = torch.zeros(3)
    out = apply_appearance(rgb, affine, bias)
    torch.testing.assert_close(out, rgb)


def test_bias_shifts_every_pixel():
    rgb = torch.zeros(3, 2, 2)
    affine = torch.eye(3)
    bias = torch.tensor([0.1, 0.0, 0.0])
    out = apply_appearance(rgb, affine, bias)
    assert torch.allclose(out[0], torch.full((2, 2), 0.1))
    assert torch.allclose(out[1], torch.zeros(2, 2))


def test_output_is_clamped_to_valid_range():
    rgb = torch.ones(3, 2, 2)
    affine = torch.eye(3) * 2.0
    bias = torch.zeros(3)
    out = apply_appearance(rgb, affine, bias)
    assert out.max() <= 1.0
    assert out.min() >= 0.0


def test_appearance_embedding_module_initializes_to_identity_no_op():
    module = AppearanceEmbedding(num_images=3)
    rgb = torch.rand(3, 4, 4)
    for image_idx in range(3):
        affine, bias = module(image_idx)
        out = apply_appearance(rgb, affine, bias)
        torch.testing.assert_close(out, rgb, atol=1e-6, rtol=1e-6)


def test_appearance_embedding_has_learnable_parameters_per_image():
    module = AppearanceEmbedding(num_images=5)
    params = list(module.parameters())
    assert len(params) > 0
    assert all(p.requires_grad for p in params)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_appearance_embedding.py -v`
Expected: `FAIL` — `ModuleNotFoundError: No module named 'src.training.appearance_embedding'`.

- [ ] **Step 3: Write `src/training/appearance_embedding.py`**

```python
from __future__ import annotations

import torch
import torch.nn as nn


def apply_appearance(rgb: torch.Tensor, affine: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    c, h, w = rgb.shape
    flat = rgb.reshape(c, h * w)
    transformed = affine @ flat + bias.unsqueeze(1)
    return transformed.reshape(c, h, w).clamp(0.0, 1.0)


class AppearanceEmbedding(nn.Module):
    """One learnable (3,3) affine + (3,) bias per training image. Must
    never be applied when rendering novel test poses -- there is no "true"
    appearance code for an unseen view; use mean_affine_bias() instead.
    """

    def __init__(self, num_images: int):
        super().__init__()
        self.affine = nn.Parameter(torch.eye(3).unsqueeze(0).repeat(num_images, 1, 1))
        self.bias = nn.Parameter(torch.zeros(num_images, 3))

    def forward(self, image_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.affine[image_idx], self.bias[image_idx]

    def mean_affine_bias(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.affine.mean(dim=0).detach(), self.bias.mean(dim=0).detach()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_appearance_embedding.py -v`
Expected: `PASS` (5 passed).

- [ ] **Step 5: Commit**

```bash
git add src/training/appearance_embedding.py tests/test_appearance_embedding.py
git commit -m "Add per-image appearance embedding module with mean-fallback for novel views"
```

---

### Task 5: Depth regularization loss (pure tensor math)

**Files:**
- Create: `src/training/depth_loss.py`
- Test: `tests/test_depth_loss.py`

**Interfaces:**
- Consumes: `compute_sparse_depth_targets` (Task 3).
- Produces: `depth_regularization_loss(rendered_depth: torch.Tensor, pixel_xy: np.ndarray, sparse_depths: np.ndarray) -> torch.Tensor`
  — `rendered_depth` is `(H, W)`; samples it at the given integer-rounded pixel coordinates and
  returns the mean L1 distance to `sparse_depths`. Returns `torch.tensor(0.0)` when there are
  zero sparse points, so the training loop can always add this term unconditionally.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_depth_loss.py
import numpy as np
import pytest
import torch

from src.training.depth_loss import depth_regularization_loss


def test_zero_loss_when_rendered_depth_matches_targets_exactly():
    rendered_depth = torch.full((10, 10), 5.0)
    pixel_xy = np.array([[3.0, 4.0], [7.0, 2.0]])
    sparse_depths = np.array([5.0, 5.0])
    loss = depth_regularization_loss(rendered_depth, pixel_xy, sparse_depths)
    assert loss.item() == 0.0


def test_positive_loss_when_rendered_depth_differs():
    rendered_depth = torch.full((10, 10), 5.0)
    pixel_xy = np.array([[3.0, 4.0]])
    sparse_depths = np.array([8.0])
    loss = depth_regularization_loss(rendered_depth, pixel_xy, sparse_depths)
    assert loss.item() == pytest.approx(3.0, abs=1e-5)


def test_zero_loss_and_no_crash_with_no_sparse_points():
    rendered_depth = torch.full((10, 10), 5.0)
    pixel_xy = np.zeros((0, 2))
    sparse_depths = np.zeros((0,))
    loss = depth_regularization_loss(rendered_depth, pixel_xy, sparse_depths)
    assert loss.item() == 0.0
    assert not torch.isnan(loss)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_depth_loss.py -v`
Expected: `FAIL` — `ModuleNotFoundError: No module named 'src.training.depth_loss'`.

- [ ] **Step 3: Write `src/training/depth_loss.py`**

```python
from __future__ import annotations

import numpy as np
import torch


def depth_regularization_loss(
    rendered_depth: torch.Tensor, pixel_xy: np.ndarray, sparse_depths: np.ndarray,
) -> torch.Tensor:
    if len(sparse_depths) == 0:
        return torch.tensor(0.0, device=rendered_depth.device)

    height, width = rendered_depth.shape
    cols = np.clip(np.round(pixel_xy[:, 0]).astype(int), 0, width - 1)
    rows = np.clip(np.round(pixel_xy[:, 1]).astype(int), 0, height - 1)

    sampled = rendered_depth[rows, cols]
    targets = torch.as_tensor(sparse_depths, dtype=sampled.dtype, device=sampled.device)
    return torch.abs(sampled - targets).mean()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_depth_loss.py -v`
Expected: `PASS` (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/training/depth_loss.py tests/test_depth_loss.py
git commit -m "Add sparse-depth L1 regularization loss"
```

---

### Task 6: Auto-select best config per scene

**Files:**
- Create: `src/evaluation/select_best_config.py`
- Test: `tests/test_select_best_config.py`

**Interfaces:**
- Produces: `select_best_candidate(candidates: list[dict], vram_budget_bytes: int) -> dict`
  where each candidate dict has keys `variant: str`, `floater_cleanup: bool`, `score: float`,
  `estimated_vram_bytes: int`, `checkpoint_path: str` (and, from Task 11, optionally
  `candidate_name`/`hyperparam_overrides`). Filters to candidates with
  `estimated_vram_bytes <= vram_budget_bytes`; among those, returns the one with the highest
  `score`. If none fit the budget, returns the smallest-`estimated_vram_bytes` candidate instead
  and sets `fallback_reason: "no candidate fit the VRAM budget"`, rather than raising — a scene
  must always ship *something*.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_select_best_config.py
from src.evaluation.select_best_config import select_best_candidate


def _candidate(variant, floater_cleanup, score, vram_bytes):
    return {
        "variant": variant, "floater_cleanup": floater_cleanup, "score": score,
        "estimated_vram_bytes": vram_bytes, "checkpoint_path": f"{variant}_{floater_cleanup}.ply",
    }


def test_picks_highest_score_among_candidates_that_fit_budget():
    candidates = [
        _candidate("baseline", False, score=0.70, vram_bytes=1_000),
        _candidate("full_stack", True, score=0.85, vram_bytes=1_500),
        _candidate("depth_reg", False, score=0.78, vram_bytes=1_200),
    ]
    best = select_best_candidate(candidates, vram_budget_bytes=2_000)
    assert best["variant"] == "full_stack"
    assert "fallback_reason" not in best


def test_excludes_candidates_over_vram_budget_even_if_higher_score():
    candidates = [
        _candidate("full_stack", True, score=0.95, vram_bytes=999_999),  # too big
        _candidate("baseline", False, score=0.70, vram_bytes=1_000),
    ]
    best = select_best_candidate(candidates, vram_budget_bytes=2_000)
    assert best["variant"] == "baseline"


def test_falls_back_to_smallest_when_nothing_fits_budget():
    candidates = [
        _candidate("full_stack", True, score=0.95, vram_bytes=999_999),
        _candidate("depth_reg", False, score=0.80, vram_bytes=500_000),
    ]
    best = select_best_candidate(candidates, vram_budget_bytes=2_000)
    assert best["variant"] == "depth_reg"
    assert best["fallback_reason"] == "no candidate fit the VRAM budget"


def test_raises_on_empty_candidate_list():
    import pytest
    with pytest.raises(ValueError):
        select_best_candidate([], vram_budget_bytes=2_000)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_select_best_config.py -v`
Expected: `FAIL` — `ModuleNotFoundError: No module named 'src.evaluation.select_best_config'`.

- [ ] **Step 3: Write `src/evaluation/select_best_config.py`**

```python
from __future__ import annotations


def select_best_candidate(candidates: list[dict], vram_budget_bytes: int) -> dict:
    if not candidates:
        raise ValueError("select_best_candidate requires at least one candidate")

    fitting = [c for c in candidates if c["estimated_vram_bytes"] <= vram_budget_bytes]
    if fitting:
        return max(fitting, key=lambda c: c["score"])

    smallest = min(candidates, key=lambda c: c["estimated_vram_bytes"])
    return {**smallest, "fallback_reason": "no candidate fit the VRAM budget"}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_select_best_config.py -v`
Expected: `PASS` (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/evaluation/select_best_config.py tests/test_select_best_config.py
git commit -m "Add VRAM-aware best-candidate selection per scene"
```

---

### Task 7: Anti-aliasing variant — confirm the baseline's native `antialiasing` flag

The vendored `graphdeco-inria/gaussian-splatting` already ships native anti-aliasing — this is
not a different rasterizer, just a boolean the render loop already reads. No new files, no
submodule to vendor, no CUDA to rebuild per variant.

**Files:** none — this task only records a verified finding and defines a contract Task 13
implements.

- [ ] **Step 1: Confirm the flag exists in the checkout (no code change)**

```bash
cd "/home/howard/Documents/viettel ai race/computer vision"
grep -n "antialiasing" third_party/gaussian-splatting/arguments/__init__.py
grep -n "antialiasing" third_party/gaussian-splatting/gaussian_renderer/__init__.py
```

Expected: `PipelineParams` defines `self.antialiasing = False`, and `render()` reads
`antialiasing=pipe.antialiasing`. If either grep comes back empty (an older submodule pin), STOP
and escalate — only then would vendoring Mip-Splatting become necessary.

- [ ] **Step 2: Record the mapping (no code, just the contract Task 13 implements)**

The `anti_alias` and `full_stack` training variants set `pipe.antialiasing = True` for both the
training render loop AND every later render of that checkpoint (holdout eval and the final
`test_poses.csv` render). A checkpoint trained with anti-aliasing on must be rendered with it on.

---

### Task 8: Reproducibility bundle

**Files:**
- Create: `src/submission/reproducibility_bundle.py`
- Test: `tests/test_reproducibility_bundle.py`

**Interfaces:**
- Produces: `write_reproducibility_bundle(scene_name: str, chosen_config: dict, all_candidates: list[dict], output_dir: Path) -> Path`
  — writes `output_dir/<scene_name>/chosen_config.yaml`,
  `output_dir/<scene_name>/all_candidates_scores.csv`, returns `output_dir/<scene_name>`. Per
  exam spec 1.7, this is what gets handed to the organizers if requested — config + full score
  comparison table, so the choice of variant per scene is traceable and justified. The CSV
  includes `candidate_name`, `hyperparam_overrides` (JSON-encoded), and `fallback_reason` columns
  — Giai đoạn 2's bounded-search candidates (Task 11's `build_hyperparam_candidates`) and
  `select_best_candidate`'s VRAM-fallback path (Task 6) both set these keys, and a bundle that
  silently dropped them would defeat its own purpose: nobody could tell WHICH hyperparameter
  override actually won for `bonsai`/`chair`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_reproducibility_bundle.py
import csv

import yaml

from src.submission.reproducibility_bundle import write_reproducibility_bundle


def test_write_reproducibility_bundle_creates_config_and_scores_csv(tmp_path):
    chosen_config = {
        "variant": "full_stack", "floater_cleanup": True, "score": 0.87,
        "estimated_vram_bytes": 12_000_000_000, "checkpoint_path": "final.ply",
    }
    all_candidates = [
        chosen_config,
        {"variant": "baseline", "floater_cleanup": False, "score": 0.70,
         "estimated_vram_bytes": 8_000_000_000, "checkpoint_path": "b.ply"},
    ]

    bundle_dir = write_reproducibility_bundle(
        "chair", chosen_config, all_candidates, tmp_path,
    )

    assert bundle_dir == tmp_path / "chair"
    config_path = bundle_dir / "chosen_config.yaml"
    assert config_path.exists()
    loaded = yaml.safe_load(config_path.read_text())
    assert loaded["variant"] == "full_stack"

    csv_path = bundle_dir / "all_candidates_scores.csv"
    assert csv_path.exists()
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert {row["variant"] for row in rows} == {"full_stack", "baseline"}


def test_write_reproducibility_bundle_preserves_giai_doan_2_and_fallback_fields(tmp_path):
    chosen_config = {
        "variant": "baseline", "floater_cleanup": False, "score": 0.60,
        "estimated_vram_bytes": 999_999_999_999, "checkpoint_path": "b.ply",
        "fallback_reason": "no candidate fit the VRAM budget",
    }
    all_candidates = [
        chosen_config,
        {"variant": "baseline", "floater_cleanup": False, "candidate_name": "bonsai_0",
         "score": 0.75, "estimated_vram_bytes": 999_999_999_999, "checkpoint_path": "e0.ply",
         "hyperparam_overrides": {"densify_grad_threshold": 0.0005}},
    ]

    bundle_dir = write_reproducibility_bundle("bonsai", chosen_config, all_candidates, tmp_path)

    with open(bundle_dir / "all_candidates_scores.csv", newline="") as f:
        rows = {row["candidate_name"] or row["checkpoint_path"]: row for row in csv.DictReader(f)}

    assert rows["b.ply"]["fallback_reason"] == "no candidate fit the VRAM budget"
    assert rows["bonsai_0"]["candidate_name"] == "bonsai_0"
    assert "0.0005" in rows["bonsai_0"]["hyperparam_overrides"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_reproducibility_bundle.py -v`
Expected: `FAIL` — `ModuleNotFoundError: No module named 'src.submission.reproducibility_bundle'`.

- [ ] **Step 3: Write `src/submission/reproducibility_bundle.py`**

```python
from __future__ import annotations

import csv
import json
from pathlib import Path

import yaml


def write_reproducibility_bundle(
    scene_name: str, chosen_config: dict, all_candidates: list[dict], output_dir: Path,
) -> Path:
    bundle_dir = Path(output_dir) / scene_name
    bundle_dir.mkdir(parents=True, exist_ok=True)

    with open(bundle_dir / "chosen_config.yaml", "w") as f:
        yaml.safe_dump(chosen_config, f)

    fieldnames = [
        "variant", "floater_cleanup", "candidate_name", "score", "estimated_vram_bytes",
        "checkpoint_path", "hyperparam_overrides", "fallback_reason",
    ]
    with open(bundle_dir / "all_candidates_scores.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for candidate in sorted(all_candidates, key=lambda c: -c["score"]):
            row = {k: candidate.get(k) for k in fieldnames}
            if row["hyperparam_overrides"] is not None:
                row["hyperparam_overrides"] = json.dumps(row["hyperparam_overrides"])
            writer.writerow(row)

    return bundle_dir
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_reproducibility_bundle.py -v`
Expected: `PASS` (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/submission/reproducibility_bundle.py tests/test_reproducibility_bundle.py
git commit -m "Add reproducibility bundle writer (config + full candidate score table per scene)"
```

---

### Task 9: Per-image holdout metrics for diagnosis

**Files:**
- Create: `src/diagnostics/__init__.py` (empty)
- Create: `src/diagnostics/scene_diagnosis.py`
- Test: `tests/test_scene_diagnosis.py`

**Interfaces:**
- Consumes: `compute_pair_metrics(pred, gt, lpips_model) -> dict` and
  `combine_score(lpips_val, ssim_val, psnr_val, psnr_max) -> float` from
  `src/evaluation/compute_metrics.py` (Plan 1).
- Produces: `compute_per_image_metrics(pred_dir: Path, gt_dir: Path, lpips_model, psnr_max: float) -> dict[str, dict[str, float]]`
  — maps `image_name -> {"lpips": ..., "ssim": ..., "psnr": ..., "score": ...}` for every file in
  `pred_dir` that has a same-named file in `gt_dir`; files with no match are skipped, not errored.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_scene_diagnosis.py
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.diagnostics.scene_diagnosis import compute_per_image_metrics


class _StubLpipsModel:
    def __call__(self, pred_tensor, gt_tensor):
        identical = torch.allclose(pred_tensor, gt_tensor)
        return torch.tensor(0.0 if identical else 1.0)


def _write_image(path: Path, fill: int) -> None:
    Image.fromarray(np.full((8, 8, 3), fill, dtype=np.uint8)).save(path)


def test_compute_per_image_metrics_matches_pred_and_gt_by_filename(tmp_path):
    pred_dir = tmp_path / "pred"
    gt_dir = tmp_path / "gt"
    pred_dir.mkdir()
    gt_dir.mkdir()

    _write_image(pred_dir / "frame_0001.jpg", fill=100)
    _write_image(gt_dir / "frame_0001.jpg", fill=100)  # identical -> best score
    _write_image(pred_dir / "frame_0002.jpg", fill=50)
    _write_image(gt_dir / "frame_0002.jpg", fill=200)  # very different -> worst score

    result = compute_per_image_metrics(pred_dir, gt_dir, _StubLpipsModel(), psnr_max=30.0)

    assert set(result.keys()) == {"frame_0001.jpg", "frame_0002.jpg"}
    for metrics in result.values():
        assert set(metrics.keys()) == {"lpips", "ssim", "psnr", "score"}
    assert result["frame_0001.jpg"]["score"] > result["frame_0002.jpg"]["score"]


def test_compute_per_image_metrics_skips_predictions_with_no_matching_ground_truth(tmp_path):
    pred_dir = tmp_path / "pred"
    gt_dir = tmp_path / "gt"
    pred_dir.mkdir()
    gt_dir.mkdir()

    _write_image(pred_dir / "frame_0001.jpg", fill=100)
    _write_image(gt_dir / "frame_0001.jpg", fill=100)
    _write_image(pred_dir / "frame_orphan.jpg", fill=10)  # no matching GT file

    result = compute_per_image_metrics(pred_dir, gt_dir, _StubLpipsModel(), psnr_max=30.0)

    assert set(result.keys()) == {"frame_0001.jpg"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_scene_diagnosis.py -v`
Expected: `FAIL` — `ModuleNotFoundError: No module named 'src.diagnostics'`.

- [ ] **Step 3: Write `src/diagnostics/scene_diagnosis.py`**

```python
from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from src.evaluation.compute_metrics import combine_score, compute_pair_metrics


def compute_per_image_metrics(
    pred_dir: Path, gt_dir: Path, lpips_model, psnr_max: float,
) -> dict[str, dict[str, float]]:
    pred_dir = Path(pred_dir)
    gt_dir = Path(gt_dir)
    result: dict[str, dict[str, float]] = {}

    for pred_path in sorted(pred_dir.iterdir()):
        if not pred_path.is_file():
            continue
        gt_path = gt_dir / pred_path.name
        if not gt_path.is_file():
            continue  # no ground truth to compare against -- skip, don't guess

        pred = np.array(Image.open(pred_path).convert("RGB"))
        gt = np.array(Image.open(gt_path).convert("RGB").resize(pred.shape[1::-1]))
        metrics = compute_pair_metrics(pred, gt, lpips_model)
        score = combine_score(metrics["lpips"], metrics["ssim"], metrics["psnr"], psnr_max)
        result[pred_path.name] = {**metrics, "score": score}

    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_scene_diagnosis.py -v`
Expected: `PASS` (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/diagnostics/__init__.py src/diagnostics/scene_diagnosis.py tests/test_scene_diagnosis.py
git commit -m "Add per-image holdout metrics for scene diagnosis"
```

---

### Task 10: Rank holdout images worst-first

**Files:**
- Modify: `src/diagnostics/scene_diagnosis.py`
- Test: `tests/test_scene_diagnosis.py`

**Interfaces:**
- Consumes: the `dict[str, dict[str, float]]` shape produced by Task 9's
  `compute_per_image_metrics` (each value has a `"score"` key).
- Produces: `rank_holdout_by_score(per_image_metrics: dict[str, dict[str, float]]) -> list[tuple[str, float]]`
  — `(image_name, score)` pairs sorted ascending by score (worst first).

- [ ] **Step 1: Write the failing tests (append to `tests/test_scene_diagnosis.py`)**

```python
from src.diagnostics.scene_diagnosis import rank_holdout_by_score


def test_rank_holdout_by_score_sorts_worst_first():
    per_image = {
        "good.jpg": {"lpips": 0.1, "ssim": 0.9, "psnr": 28.0, "score": 0.85},
        "bad.jpg": {"lpips": 0.6, "ssim": 0.4, "psnr": 12.0, "score": 0.30},
        "medium.jpg": {"lpips": 0.3, "ssim": 0.6, "psnr": 20.0, "score": 0.55},
    }
    ranked = rank_holdout_by_score(per_image)
    assert ranked == [
        ("bad.jpg", 0.30),
        ("medium.jpg", 0.55),
        ("good.jpg", 0.85),
    ]


def test_rank_holdout_by_score_handles_empty_input():
    assert rank_holdout_by_score({}) == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_scene_diagnosis.py -v -k rank_holdout`
Expected: `FAIL` — `ImportError: cannot import name 'rank_holdout_by_score'`.

- [ ] **Step 3: Add to `src/diagnostics/scene_diagnosis.py`**

```python
def rank_holdout_by_score(per_image_metrics: dict[str, dict[str, float]]) -> list[tuple[str, float]]:
    return sorted(
        ((name, metrics["score"]) for name, metrics in per_image_metrics.items()),
        key=lambda pair: pair[1],
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_scene_diagnosis.py -v`
Expected: `PASS` (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/diagnostics/scene_diagnosis.py tests/test_scene_diagnosis.py
git commit -m "Add worst-first holdout ranking for scene diagnosis"
```

---

### Task 11: Screening tie-break and bounded hyperparameter candidates

**Files:**
- Create: `src/evaluation/screening.py`
- Test: `tests/test_screening.py`

**Interfaces:**
- Produces:
  - `needs_tiebreak_rerun(candidates: list[dict], threshold: float = 0.01) -> list[str]` — given
    dicts with `"variant"` and `"score"` keys, returns the `variant` names (excluding the single
    highest scorer) whose score is within `threshold` of the leader.
  - `variants_needing_full_iteration_verification(candidates: list[dict], threshold: float = 0.01) -> list[str]`
    — wraps `needs_tiebreak_rerun`, but when it returns any runner-up, the LEADER is included too
    (leader first, then runner-ups, no duplicates). A reduced-iteration screening leader is not a
    trustworthy baseline to compare a full-iteration runner-up against — re-verifying only the
    runner-up would unfairly let it win purely from getting more training, not from actually
    being better. Returns `[]` when there is no close call (nothing needs re-verification,
    including the leader).
  - `build_hyperparam_candidates(base_overrides: dict[str, object], extra_overrides: list[dict[str, object]], label_prefix: str) -> list[dict[str, object]]`
    — for each entry in `extra_overrides`, merges it on top of `base_overrides` and tags the
    result with a unique `candidate_name` (`f"{label_prefix}_{i}"`). Deliberately NOT a
    combinatorial grid — a hand-picked, bounded list (spec: max 4 per scene).

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_screening.py
from src.evaluation.screening import (
    build_hyperparam_candidates,
    needs_tiebreak_rerun,
    variants_needing_full_iteration_verification,
)


def test_needs_tiebreak_rerun_flags_close_runner_up():
    candidates = [
        {"variant": "full_stack", "score": 0.700},
        {"variant": "depth_reg", "score": 0.695},
        {"variant": "baseline", "score": 0.500},
    ]
    assert needs_tiebreak_rerun(candidates, threshold=0.01) == ["depth_reg"]


def test_needs_tiebreak_rerun_returns_empty_when_leader_is_clear():
    candidates = [
        {"variant": "full_stack", "score": 0.700},
        {"variant": "baseline", "score": 0.500},
    ]
    assert needs_tiebreak_rerun(candidates, threshold=0.01) == []


def test_needs_tiebreak_rerun_handles_empty_list():
    assert needs_tiebreak_rerun([], threshold=0.01) == []


def test_variants_needing_full_iteration_verification_includes_leader_with_close_runner_up():
    candidates = [
        {"variant": "full_stack", "score": 0.700},
        {"variant": "depth_reg", "score": 0.695},
        {"variant": "baseline", "score": 0.500},
    ]
    result = variants_needing_full_iteration_verification(candidates, threshold=0.01)
    # Leader must be re-verified too -- comparing a full-iteration
    # runner-up against the leader's un-verified screening score would be
    # an unfair fight.
    assert result == ["full_stack", "depth_reg"]


def test_variants_needing_full_iteration_verification_empty_when_leader_is_clear():
    candidates = [
        {"variant": "full_stack", "score": 0.700},
        {"variant": "baseline", "score": 0.500},
    ]
    assert variants_needing_full_iteration_verification(candidates, threshold=0.01) == []


def test_variants_needing_full_iteration_verification_handles_empty_list():
    assert variants_needing_full_iteration_verification([], threshold=0.01) == []


def test_build_hyperparam_candidates_merges_and_labels():
    base = {"variant": "depth_reg", "densify_grad_threshold": 0.001}
    extra = [
        {"densify_grad_threshold": 0.0015},
        {"iterations": 45000},
    ]
    candidates = build_hyperparam_candidates(base, extra, label_prefix="bonsai")

    assert candidates == [
        {"variant": "depth_reg", "densify_grad_threshold": 0.0015, "candidate_name": "bonsai_0"},
        {"variant": "depth_reg", "densify_grad_threshold": 0.001, "iterations": 45000,
         "candidate_name": "bonsai_1"},
    ]


def test_build_hyperparam_candidates_handles_empty_overrides():
    assert build_hyperparam_candidates({"variant": "baseline"}, [], "chair") == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_screening.py -v`
Expected: `FAIL` — `ModuleNotFoundError: No module named 'src.evaluation.screening'`.

- [ ] **Step 3: Write `src/evaluation/screening.py`**

```python
from __future__ import annotations


def needs_tiebreak_rerun(candidates: list[dict], threshold: float = 0.01) -> list[str]:
    if not candidates:
        return []
    ranked = sorted(candidates, key=lambda c: c["score"], reverse=True)
    leader_score = ranked[0]["score"]
    return [c["variant"] for c in ranked[1:] if leader_score - c["score"] <= threshold]


def variants_needing_full_iteration_verification(candidates: list[dict], threshold: float = 0.01) -> list[str]:
    runner_ups = needs_tiebreak_rerun(candidates, threshold=threshold)
    if not runner_ups:
        return []
    leader = max(candidates, key=lambda c: c["score"])["variant"]
    return list(dict.fromkeys([leader, *runner_ups]))


def build_hyperparam_candidates(
    base_overrides: dict[str, object], extra_overrides: list[dict[str, object]], label_prefix: str,
) -> list[dict[str, object]]:
    candidates = []
    for i, override in enumerate(extra_overrides):
        merged = {**base_overrides, **override}
        merged["candidate_name"] = f"{label_prefix}_{i}"
        candidates.append(merged)
    return candidates
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_screening.py -v`
Expected: `PASS` (8 passed).

- [ ] **Step 5: Commit**

```bash
git add src/evaluation/screening.py tests/test_screening.py
git commit -m "Add screening tie-break decision, leader re-verification, and hyperparameter candidate builder"
```

---

### Task 12: Diagnosis notebook cell (Giai đoạn 0)

**Files:**
- Modify: `notebooks/colab_runner_hcm.ipynb`

**Interfaces:**
- Consumes: `compute_per_image_metrics`, `rank_holdout_by_score` (Task 9/10); the already-existing
  `OUTPUT_ROOT/<scene>/holdout_render/` directories and `undistort_scene` (Plan 1) — both already
  on Drive from the finished Plan 1 baseline run, so this costs no GPU time, only CPU + human
  eyeballing.

No GPU-dependent training code in this task, but it is Colab-only (needs the already-mounted
Drive + `load_lpips_model()`) so there is no local `pytest` step — verify manually on Colab per
Step 2.

- [ ] **Step 1: Insert a new markdown + code cell after the existing Bước 7 cell**

Markdown cell:

```markdown
## Bước 8 — Chẩn đoán holdout trước khi đầu tư thêm GPU (Giai đoạn 0)

Không tốn GPU — chỉ đọc lại ảnh holdout đã render sẵn từ Bước 5/6 và so trực tiếp với ảnh gốc
(ground-truth) tương ứng, khác với Bước 7 (không có ground-truth cho `test_poses.csv` thật). Với
mỗi scene, hiển thị 5 ảnh holdout tệ nhất (predicted cạnh ground-truth) kèm LPIPS/SSIM/PSNR từng
ảnh — dùng để quyết định `bonsai`/`chair` cần thử hyperparameter nào ở giai đoạn sau.
```

Code cell:

```python
from PIL import Image
import matplotlib.pyplot as plt

from src.common.config import load_scenes
from src.diagnostics.scene_diagnosis import compute_per_image_metrics, rank_holdout_by_score
from src.evaluation.compute_metrics import load_lpips_model
from src.training.undistort_scene import undistort_scene

lpips_model = load_lpips_model()
all_7_scenes = load_scenes()

for scene in all_7_scenes:
    scene_output = OUTPUT_ROOT / scene.name
    pred_dir = scene_output / "holdout_render"
    if not pred_dir.exists():
        print(f"== {scene.name}: chưa có holdout_render, bỏ qua ==")
        continue

    working_scene = undistort_scene(scene, scene_output / "undistorted")
    per_image = compute_per_image_metrics(pred_dir, working_scene.train_images_dir, lpips_model, psnr_max=30.0)
    worst = rank_holdout_by_score(per_image)[:5]

    print(f"== {scene.name}: {len(per_image)} ảnh holdout, 5 ảnh tệ nhất ==")
    for name, score in worst:
        m = per_image[name]
        print(f"  {name}: score={score:.4f}  lpips={m['lpips']:.4f} ssim={m['ssim']:.4f} psnr={m['psnr']:.2f}")

    fig, axes = plt.subplots(2, len(worst), figsize=(5 * len(worst), 10))
    for i, (name, score) in enumerate(worst):
        axes[0, i].imshow(Image.open(pred_dir / name).convert("RGB"))
        axes[0, i].set_title(f"predicted: {name}")
        axes[0, i].axis("off")
        axes[1, i].imshow(Image.open(working_scene.train_images_dir / name).convert("RGB"))
        axes[1, i].set_title(f"ground-truth: {name}")
        axes[1, i].axis("off")
    plt.suptitle(scene.name)
    plt.show()
```

- [ ] **Step 2: Manual verification on Colab (required)**

Run this cell after Bước 7 with Drive mounted (Bước 2) and the environment set up (Bước 3). For
each of the 7 scenes, confirm: predicted/ground-truth pairs display side by side, LPIPS/SSIM/PSNR
values are printed, and no exception is raised for scenes whose `holdout_render/` already exists
from the finished Plan 1 run. Note down, per scene (especially `bonsai`/`chair`), what the worst
frames have in common — this human judgment feeds Task 17's `extra_overrides` choice for Giai
đoạn 2.

- [ ] **Step 3: Commit**

```bash
git add notebooks/colab_runner_hcm.ipynb
git commit -m "Add holdout diagnosis cell (Buoc 8) for pre-training-investment triage"
```

---

### Task 13: Variant-aware training loop with hyperparameter overrides

This is the one piece of this plan that cannot be given a from-scratch, independently-verified
implementation without the actual checked-out vendored source in front of the implementer — it
directly drives the CUDA-differentiable renderer. It is written completely, reusing the
pure/tested pieces from Tasks 3-5, against the well-established vendored API (`GaussianModel`,
`render()`, `l1_loss`/`ssim` from `utils.loss_utils`) — but must be verified against the actual
checked-out `third_party/gaussian-splatting/train.py` before trusting it.

**Files:**
- Create: `src/training/train_variant.py`
- Test: `tests/test_train_variant.py` (variant config logic only — no GPU)

**Interfaces:**
- Produces:
  - `TrainingVariant` frozen dataclass: `name: str`, `use_depth_reg: bool`, `use_anti_alias: bool`,
    `use_appearance_embed: bool`.
  - `ALL_TRAINING_VARIANTS: list[TrainingVariant]` — `baseline`, `depth_reg`, `anti_alias`,
    `appearance_embed`, `full_stack`.
  - `run_training_variant(scene: SceneConfig, variant: TrainingVariant, output_dir: Path, iterations: int, hyperparam_overrides: dict[str, object] | None = None) -> Path`
    — GPU-dependent; runs the actual training loop and returns the final checkpoint `.ply` path.
    Unknown override keys raise `ValueError` immediately rather than silently doing nothing — a
    typo would otherwise waste a full Colab run before anyone notices the override never applied.
    `hyperparam_overrides["iterations"]`, if present, actually controls the loop length, the save
    call, and the returned checkpoint path (via `effective_iterations`) — it is NOT just an
    `opt.iterations = ...` `setattr` like every other override key, since the training loop's
    `range()` bound is the function parameter `iterations`, not `opt.iterations`.

- [ ] **Step 1: Write the failing test (variant config only, no GPU)**

```python
# tests/test_train_variant.py
import inspect

from src.training.train_variant import ALL_TRAINING_VARIANTS, TrainingVariant, run_training_variant


def test_all_training_variants_have_unique_names():
    names = [v.name for v in ALL_TRAINING_VARIANTS]
    assert len(names) == len(set(names))
    assert set(names) == {"baseline", "depth_reg", "anti_alias", "appearance_embed", "full_stack"}


def test_baseline_variant_has_no_techniques_enabled():
    baseline = next(v for v in ALL_TRAINING_VARIANTS if v.name == "baseline")
    assert baseline.use_depth_reg is False
    assert baseline.use_anti_alias is False
    assert baseline.use_appearance_embed is False


def test_full_stack_variant_has_all_techniques_enabled():
    full_stack = next(v for v in ALL_TRAINING_VARIANTS if v.name == "full_stack")
    assert full_stack.use_depth_reg is True
    assert full_stack.use_anti_alias is True
    assert full_stack.use_appearance_embed is True


def test_run_training_variant_signature_accepts_hyperparam_overrides():
    params = inspect.signature(run_training_variant).parameters
    assert "hyperparam_overrides" in params
    assert params["hyperparam_overrides"].default is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_train_variant.py -v`
Expected: `FAIL` — `ModuleNotFoundError: No module named 'src.training.train_variant'`.

- [ ] **Step 3: Write `src/training/train_variant.py`**

```python
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from src.common.config import SceneConfig

_VENDORED_REPO = Path(__file__).resolve().parents[2] / "third_party" / "gaussian-splatting"
if str(_VENDORED_REPO) not in sys.path:
    sys.path.insert(0, str(_VENDORED_REPO))


@dataclass(frozen=True)
class TrainingVariant:
    name: str
    use_depth_reg: bool
    use_anti_alias: bool
    use_appearance_embed: bool


ALL_TRAINING_VARIANTS: list[TrainingVariant] = [
    TrainingVariant("baseline", False, False, False),
    TrainingVariant("depth_reg", True, False, False),
    TrainingVariant("anti_alias", False, True, False),
    TrainingVariant("appearance_embed", False, False, True),
    TrainingVariant("full_stack", True, True, True),
]


def _build_dataset_args(gs_source_dir: Path, model_path: Path, use_anti_alias: bool):
    """Construct the args object the vendored Scene/render expect. Rather
    than hand-roll a Namespace and risk missing a field, build the real
    ModelParams/PipelineParams/OptimizationParams via an ArgumentParser
    (their ParamGroup base fills every default), then override the few
    paths we control. Returns (dataset, pipe, opt).
    """
    from argparse import ArgumentParser
    from arguments import ModelParams, OptimizationParams, PipelineParams

    parser = ArgumentParser()
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    args = parser.parse_args([])  # all defaults

    dataset = lp.extract(args)
    dataset.source_path = str(Path(gs_source_dir).resolve())
    dataset.model_path = str(Path(model_path).resolve())
    dataset.eval = False  # holdout handling is done upstream by the orchestrator

    pipe = pp.extract(args)
    pipe.antialiasing = use_anti_alias  # Task 7: native flag, no separate rasterizer

    opt = op.extract(args)
    return dataset, pipe, opt


def run_training_variant(
    scene: SceneConfig, variant: TrainingVariant, output_dir: Path, iterations: int,
    hyperparam_overrides: dict[str, object] | None = None,
) -> Path:
    """GPU-dependent training loop for one variant. Mirrors the vendored
    train.py::training() structure, verified against the checked-out
    third_party/gaussian-splatting (commit 54c035f).
    """
    import torch
    from random import randint

    from gaussian_renderer import render as gs_render
    from scene import GaussianModel, Scene
    from utils.loss_utils import l1_loss, ssim

    from src.common.colmap_io import load_sparse_scene
    from src.training.appearance_embedding import AppearanceEmbedding, apply_appearance
    from src.training.depth_loss import depth_regularization_loss
    from src.training.sparse_depth import compute_sparse_depth_targets

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset, pipe, opt = _build_dataset_args(
        scene.gs_source_dir, output_dir, variant.use_anti_alias,
    )

    # "iterations" is handled separately from the rest of hyperparam_overrides:
    # everything else is a plain opt.<key> = value, but the actual loop bound
    # below is driven by the *function parameter* iterations, not opt.iterations
    # -- setattr(opt, "iterations", ...) alone would silently leave the loop,
    # the final gs_scene.save() call, and the returned checkpoint path all
    # still using the un-overridden iteration count.
    overrides = dict(hyperparam_overrides or {})
    effective_iterations = overrides.pop("iterations", iterations)
    opt.iterations = effective_iterations

    for key, value in overrides.items():
        if not hasattr(opt, key):
            raise ValueError(f"unknown training hyperparameter override: {key!r}")
        setattr(opt, key, value)

    gaussians = GaussianModel(dataset.sh_degree)
    gs_scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    sparse = load_sparse_scene(scene.sparse_dir) if variant.use_depth_reg else None

    train_cameras = gs_scene.getTrainCameras()
    appearance = (
        AppearanceEmbedding(num_images=len(train_cameras)).cuda()
        if variant.use_appearance_embed else None
    )
    if appearance is not None:
        appearance_optimizer = torch.optim.Adam(appearance.parameters(), lr=1e-3)

    bg_color = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda")

    for iteration in range(1, effective_iterations + 1):
        gaussians.update_learning_rate(iteration)
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        viewpoint_cam = train_cameras[randint(0, len(train_cameras) - 1)]
        render_pkg = gs_render(viewpoint_cam, gaussians, pipe, bg_color)
        image = render_pkg["render"]
        viewspace_points = render_pkg["viewspace_points"]
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]

        if appearance is not None:
            affine, bias = appearance(viewpoint_cam.uid)
            image = apply_appearance(image, affine, bias)

        gt_image = viewpoint_cam.original_image.cuda()
        loss = (1.0 - opt.lambda_dssim) * l1_loss(image, gt_image) \
            + opt.lambda_dssim * (1.0 - ssim(image, gt_image))

        if variant.use_depth_reg:
            colmap_image = sparse.images[viewpoint_cam.colmap_id]
            pixel_xy, sparse_depths = compute_sparse_depth_targets(
                colmap_image.qvec, colmap_image.tvec,
                colmap_image.xys, colmap_image.point3D_ids, sparse.points3d,
            )
            loss = loss + 0.1 * depth_regularization_loss(
                render_pkg["depth"], pixel_xy, sparse_depths,
            )

        loss.backward()

        with torch.no_grad():
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter], radii[visibility_filter],
                )
                gaussians.add_densification_stats(viewspace_points, visibility_filter)
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold, 0.005,
                        gs_scene.cameras_extent, size_threshold, radii,
                    )
                if iteration % opt.opacity_reset_interval == 0:
                    gaussians.reset_opacity()

            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)
            if appearance is not None:
                appearance_optimizer.step()
                appearance_optimizer.zero_grad(set_to_none=True)

    if appearance is not None:
        _save_mean_appearance(appearance, output_dir)

    gs_scene.save(effective_iterations)
    final_ply = output_dir / "point_cloud" / f"iteration_{effective_iterations}" / "point_cloud.ply"
    return final_ply


def _save_mean_appearance(appearance, output_dir: Path) -> None:
    import torch

    affine, bias = appearance.mean_affine_bias()
    torch.save(
        {"affine": affine.cpu(), "bias": bias.cpu()},
        Path(output_dir) / "mean_appearance.pt",
    )
```

Note on anti-aliasing (Task 7): the `anti_alias`/`full_stack` variants set
`pipe.antialiasing = True` via `_build_dataset_args`. The render step used later for holdout
eval and the final `test_poses.csv` render MUST set the same flag for that checkpoint.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_train_variant.py -v`
Expected: `PASS` (4 passed) — these tests only exercise `ALL_TRAINING_VARIANTS` and the
`run_training_variant` signature, not its body (GPU-dependent, verified on Colab in Step 6).

- [ ] **Step 5: Commit**

```bash
git add src/training/train_variant.py tests/test_train_variant.py
git commit -m "Add variant-aware training loop with hyperparameter override support"
```

- [ ] **Step 6: Manual verification and correction on Colab (required, not optional)**

Run in this order and fix any remaining mismatch before trusting it:
1. `"baseline"` variant on the `chair` scene (cheapest) — confirm it produces
   `point_cloud/iteration_<N>/point_cloud.ply` and that Plan 1's `render_all` can load and render
   it without error.
2. Then `depth_reg`, `anti_alias`, `appearance_embed` each individually.
3. Only then `full_stack`.
4. `hyperparam_overrides={"densify_grad_threshold": 0.0005}` for a short `iterations=200` smoke
   run on `baseline`/`chair` — print `opt.densify_grad_threshold` right after the override loop
   to confirm it actually changed from the vendored default (`0.0002`). Confirm
   `hyperparam_overrides={"not_a_real_field": 1}` raises `ValueError` immediately.
5. `run_training_variant(chair_scene, baseline_variant, some_dir, iterations=200, hyperparam_overrides={"iterations": 50})`
   — confirm training actually stops at iteration 50 (not 200), and the returned path contains
   `iteration_50`, not `iteration_200`. This is the exact bug Task 16's Giai đoạn 2
   `{"iterations": 45000}` candidates depend on being fixed.

Known things most likely to still need adjustment on real hardware: whether `Scene.save` writes
exactly to `point_cloud/iteration_<N>/point_cloud.ply` for this submodule pin, and whether
`viewpoint_cam.uid` is the right per-image index for the appearance embedding (must be a stable
0..N-1 index over `getTrainCameras()`; if not, enumerate the camera list instead).

---

### Task 14: Checkpoint-level floater prune wrapper

**Files:**
- Modify: `src/postprocess/prune_floaters.py` (Task 2 only produces `compute_prune_mask`, a
  boolean-array function; this task wraps the actual file I/O)

**Interfaces:**
- Consumes: `compute_prune_mask` (Task 2).
- Produces: `prune_checkpoint(checkpoint_path: Path, bbox_min: np.ndarray, bbox_max: np.ndarray, sh_degree: int = 3) -> Path`
  — loads the `.ply` at `checkpoint_path` via the vendored `GaussianModel.load_ply`, prunes
  floaters via `compute_prune_mask`, saves the result as `<stem>_pruned.ply` next to the
  original, and returns that path. This is the exact `prune_fn` callable Task 16's
  `run_experiment_matrix_pipeline` (and Task 17's notebook cells) inject.

This wraps the vendored `GaussianModel.load_ply`/`.save_ply` (verified present in the checked-out
`third_party/gaussian-splatting/scene/gaussian_model.py:239,263`), which hardcode `device="cuda"`
internally — same GPU-dependent category as Task 13, so there is no local `pytest` step;
`compute_prune_mask` itself (Task 2) already has its own local tests.

**Does NOT use `GaussianModel.prune_points()`.** That method (verified at
`third_party/gaussian-splatting/scene/gaussian_model.py:331,349`) goes through
`_prune_optimizer`, which dereferences `self.optimizer.param_groups` — but `self.optimizer` is
only ever created by `training_setup()`, which a render/prune-only `load_ply()` never calls, so
`self.optimizer` is `None` and this would crash with `AttributeError: 'NoneType' object has no
attribute 'param_groups'`. Instead, `prune_checkpoint` filters the six raw parameter tensors
`save_ply` reads (`_xyz`, `_features_dc`, `_features_rest`, `_opacity`, `_scaling`, `_rotation`)
directly, with no optimizer involved at all.

- [ ] **Step 1: Append to `src/postprocess/prune_floaters.py`**

```python
def prune_checkpoint(checkpoint_path, bbox_min, bbox_max, sh_degree: int = 3):
    from pathlib import Path

    import torch
    import torch.nn as nn

    from scene import GaussianModel

    checkpoint_path = Path(checkpoint_path)
    gaussians = GaussianModel(sh_degree)
    gaussians.load_ply(str(checkpoint_path))

    xyz = gaussians._xyz.detach().cpu().numpy()
    opacity = gaussians.get_opacity.detach().cpu().numpy().squeeze(-1)
    scales = gaussians.get_scaling.detach().cpu().numpy()

    keep_mask = compute_prune_mask(xyz, opacity, scales, bbox_min, bbox_max)
    keep_mask_t = torch.from_numpy(keep_mask).to(gaussians._xyz.device)

    # Deliberately NOT GaussianModel.prune_points(): that method routes
    # through _prune_optimizer, which needs self.optimizer -- only set by
    # training_setup(), which a load-prune-save-only path never calls.
    # Filter the six tensors save_ply() actually reads directly instead.
    gaussians._xyz = nn.Parameter(gaussians._xyz[keep_mask_t].detach())
    gaussians._features_dc = nn.Parameter(gaussians._features_dc[keep_mask_t].detach())
    gaussians._features_rest = nn.Parameter(gaussians._features_rest[keep_mask_t].detach())
    gaussians._opacity = nn.Parameter(gaussians._opacity[keep_mask_t].detach())
    gaussians._scaling = nn.Parameter(gaussians._scaling[keep_mask_t].detach())
    gaussians._rotation = nn.Parameter(gaussians._rotation[keep_mask_t].detach())

    output_path = checkpoint_path.with_name(f"{checkpoint_path.stem}_pruned.ply")
    gaussians.save_ply(str(output_path))
    return output_path
```

- [ ] **Step 2: Manual verification on Colab (required, not optional)**

Run `prune_checkpoint` on a real `final_train` checkpoint from Task 13 Step 6's verification run.
Confirm it does NOT raise (in particular, does not hit the `self.optimizer is None` crash
`prune_points()` would have caused),
`count_gaussians_in_ply(output_path) < count_gaussians_in_ply(checkpoint_path)` (the real
`chair`/HCM scenes have sky/background floaters per the diagnosis in Task 12, so some pruning is
expected), and that `real_render_fn` loads and renders the pruned `.ply` without error.

- [ ] **Step 3: Commit**

```bash
git add src/postprocess/prune_floaters.py
git commit -m "Add checkpoint-level floater prune wrapper (load/prune/save .ply)"
```

---

### Task 15: Variant-aware rendering (render_fn config)

**Problem this closes:** `real_render_fn` (`src/rendering/gs_render_fn.py`, already implemented
and in use since Plan 1) builds its `pipe`/appearance state from all-defaults
(`_default_opt_and_pipe()`), with no way to turn `pipe.antialiasing` on or apply a saved
appearance correction. Task 13's `anti_alias`/`full_stack` variants train with
`pipe.antialiasing=True`, and `appearance_embed`/`full_stack` train a per-image color
correction whose *mean* (`mean_appearance.pt`) is meant to stand in for novel views — but nothing
in this plan actually applied either at render/evaluation time before this task. Every holdout
score for those variants would silently be scored against a checkpoint rendered in a DIFFERENT
configuration than it was trained in, defeating the entire point of comparing variants.

**Files:**
- Modify: `src/rendering/gs_render_fn.py`

**Interfaces:**
- Modifies: `real_render_fn(checkpoint, params_list, output_dir) -> list[Path]` gains a 4th
  parameter, `render_config: dict | None = None`, with optional keys `"antialiasing": bool` and
  `"appearance_path": Path | None`. Default `None` behaves exactly as before (no behavior change
  for Plan 1's existing calls, which never pass this argument) — `pipe.antialiasing` stays
  `False` and no appearance correction is applied.

- [ ] **Step 1: Modify `src/rendering/gs_render_fn.py`**

Change the `real_render_fn` signature from:

```python
def real_render_fn(
    checkpoint: Path, params_list: list[CameraParams], output_dir: Path,
) -> list[Path]:
```

to:

```python
def real_render_fn(
    checkpoint: Path, params_list: list[CameraParams], output_dir: Path,
    render_config: dict | None = None,
) -> list[Path]:
```

Replace the body from `gaussians = _load_gaussians(checkpoint)` through the `return render_all(...)`
call with:

```python
    from src.training.appearance_embedding import apply_appearance

    gaussians = _load_gaussians(checkpoint)
    _opt, pipe = _default_opt_and_pipe()
    render_config = render_config or {}
    pipe.antialiasing = render_config.get("antialiasing", False)

    appearance_affine_bias = None
    appearance_path = render_config.get("appearance_path")
    if appearance_path is not None and Path(appearance_path).exists():
        # weights_only=False: our own checkpoint (a plain dict of two
        # tensors), produced by Task 13's _save_mean_appearance -- trusted,
        # same reasoning as _load_gaussians' torch.load above.
        saved = torch.load(appearance_path, weights_only=False)
        appearance_affine_bias = (saved["affine"].cuda(), saved["bias"].cuda())

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
        if appearance_affine_bias is not None:
            affine, bias = appearance_affine_bias
            rendered = apply_appearance(rendered, affine, bias)
        return _tensor_to_uint8_image(rendered)

    return render_all(
        checkpoint, None, output_dir, _render_one,
        params_list=params_list, gaussians=gaussians,
    )
```

(`Path` and `torch` are already imported at module level; `gs_render`/`Camera` are already
imported inside the function, unchanged.)

- [ ] **Step 2: Manual verification on Colab (required, not optional)**

Using a `chair` checkpoint trained with the `anti_alias` variant (Task 13 Step 6): render once
with `render_config=None` and once with `render_config={"antialiasing": True}`, confirm the two
outputs differ (antialiasing is actually toggling something). Using an `appearance_embed`
checkpoint: confirm `render_config={"appearance_path": <output_dir>/"mean_appearance.pt"}`
produces a visibly different (color-corrected) image than `render_config=None`, and that a
`render_config` with a non-existent `appearance_path` behaves identically to `render_config=None`
(no crash, silently skipped) — Task 16's screening loop passes an `appearance_path` unconditionally
whenever `variant.use_appearance_embed` is true, even before the file may have been written by a
still-running training call in a different code path, and this must not crash.

- [ ] **Step 3: Commit**

```bash
git add src/rendering/gs_render_fn.py
git commit -m "Add render_config (antialiasing, appearance) to real_render_fn"
```

---

### Task 16: Experiment-matrix orchestrator

**Files:**
- Create: `src/orchestrator/run_experiment_matrix.py`
- Test: `tests/test_run_experiment_matrix.py`

**Interfaces:**
- Consumes: `ALL_TRAINING_VARIANTS`/`run_training_variant` (Task 13), `select_best_candidate`
  (Task 6), `estimate_vram_bytes`/`count_gaussians_in_ply` (Task 1),
  `variants_needing_full_iteration_verification` (Task 11), `camera_focal_lengths`
  (`src/common/pose_utils.py`, Plan 1), everything Plan 1's `run_baseline_pipeline` already wires
  (holdout split, `undistort_scene`, `build_filtered_scene`, `compute_pair_metrics`,
  `package_submission`, `validate_submission`), Task 15's `render_fn(checkpoint, params, output_dir, render_config=None)`.
- Produces: `run_experiment_matrix_pipeline(scenes, screening_train_fn, final_train_fn, render_fn, prune_fn, lpips_model, psnr_max, vram_budget_bytes, output_root, extra_candidates_by_scene=None, tiebreak_threshold=0.01) -> ExperimentPipelineResult`
  — `screening_train_fn(scene, variant, output_dir) -> Path` and
  `final_train_fn(scene, variant, output_dir, hyperparam_overrides=None) -> Path` are injected
  exactly like Plan 1's `train_fn`/`render_fn`, so this stays GPU-free and network-free to test.
  `ExperimentPipelineResult` carries `per_scene_scores`, `skipped_scenes`,
  `chosen_config: dict[str, dict]`, `all_candidates: dict[str, list[dict]]`,
  `validation_problems`, `submission_zip`.

Design points worth calling out (spec sections 3-6):
1. `screening_train_fn`/`final_train_fn` split — reduced-iteration screening vs. full-iteration
   final retrain.
2. Scenes are undistorted (`undistort_scene`) before any training — real BTS scenes are
   `SIMPLE_RADIAL` and would crash the vendored loader otherwise.
3. `extra_candidates_by_scene` and `tiebreak_threshold`/`variants_needing_full_iteration_verification`
   wiring for Giai đoạn 2's bounded hyperparameter search and the screening tie-break rule — the
   current screening LEADER is re-verified at full iterations too whenever any runner-up is close
   enough to challenge it, not just the runner-up, so the eventual comparison is always
   full-iterations-vs-full-iterations, never full-vs-screening.
4. Every render/score call passes a `render_config` built from the variant actually used
   (`{"antialiasing": variant.use_anti_alias, "appearance_path": ...}`, via `_render_config_for`)
   — a checkpoint trained with anti-aliasing or appearance embedding on must be rendered and
   scored with the same setting on, both during screening/tie-break/extra-candidate scoring and
   for the final `test_poses.csv` render, or the holdout score and the shipped renders would
   reflect a different model than what was actually trained.
5. Holdout camera focal lengths use `camera_focal_lengths(camera.model, camera.params)`, not
   `camera.params[0]`/`params[1]` directly — for `SIMPLE_PINHOLE` (`chair`, `bonsai`),
   `params[1]` is `cx`, not a second focal length; see `src/common/pose_utils.py:27`'s own
   docstring for why this is a documented trap, not a hypothetical one.
6. Fails closed exactly like Plan 1's `run_baseline_pipeline`
   (`src/orchestrator/run_pipeline.py:169-188`): any skipped scene, or any `validate_submission`
   problem, forces `submission_zip = None` — the zip stays on disk for debugging but is never
   reported as valid, since the exam voids the entire score for a missing/malformed scene.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_run_experiment_matrix.py
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.common.config import SceneConfig
from src.orchestrator.run_experiment_matrix import run_experiment_matrix_pipeline


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
    def __call__(self, pred_tensor, gt_tensor):
        diff = torch.mean(torch.abs(pred_tensor - gt_tensor)).item()
        return torch.tensor(diff)


def _write_fake_ply(path: Path) -> None:
    path.write_bytes(
        b"ply\nformat binary_little_endian 1.0\nelement vertex 10\n"
        b"property float x\nproperty float y\nproperty float z\nend_header\n" + b"\x00" * 120
    )


def test_run_experiment_matrix_screens_all_variants_and_uses_full_iterations_for_winner(tmp_path):
    scene = _chair_scene()
    screening_calls = []
    final_calls = []
    render_calls = []

    def fake_screening_train_fn(scene_arg, variant, output_dir):
        screening_calls.append(variant.name)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        ply = output_dir / "point_cloud.ply"
        _write_fake_ply(ply)
        return ply

    def fake_final_train_fn(scene_arg, variant, output_dir, hyperparam_overrides=None):
        final_calls.append((variant.name, str(output_dir), hyperparam_overrides))
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        ply = output_dir / "point_cloud.ply"
        _write_fake_ply(ply)
        return ply

    def fake_render_fn(checkpoint, params_list, output_dir, render_config=None):
        render_calls.append((str(checkpoint), render_config))
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        fill = hash(str(checkpoint)) % 200
        written = []
        for params in params_list:
            path = output_dir / params.image_name
            Image.fromarray(
                np.full((params.height, params.width, 3), fill, dtype=np.uint8)
            ).save(path)
            written.append(path)
        return written

    def fake_prune_fn(checkpoint_path, bbox_min, bbox_max):
        pruned = Path(checkpoint_path).with_name("point_cloud_pruned.ply")
        pruned.write_bytes(Path(checkpoint_path).read_bytes())
        return pruned

    extra_candidates = {
        "chair": [
            {"variant": "baseline", "candidate_name": "chair_extra_0", "densify_grad_threshold": 0.002},
            {"variant": "baseline", "candidate_name": "chair_extra_1", "iterations": 45000},
        ],
    }

    result = run_experiment_matrix_pipeline(
        scenes=[scene],
        screening_train_fn=fake_screening_train_fn,
        final_train_fn=fake_final_train_fn,
        render_fn=fake_render_fn,
        prune_fn=fake_prune_fn,
        lpips_model=_StubLpipsModel(),
        psnr_max=30.0,
        vram_budget_bytes=16 * 1024**3,
        output_root=tmp_path,
        extra_candidates_by_scene=extra_candidates,
    )

    assert set(screening_calls) == {
        "baseline", "depth_reg", "anti_alias", "appearance_embed", "full_stack",
    }
    assert len(result.all_candidates["chair"]) == 12
    assert any(c.get("candidate_name") == "chair_extra_0" for c in result.all_candidates["chair"])
    assert any("final_train" in call[1] for call in final_calls)

    # anti_alias/full_stack screening renders must have been asked to
    # render WITH antialiasing on -- a checkpoint trained with it on must
    # never be scored/shipped as if it were off.
    anti_alias_render_configs = [
        cfg for ckpt, cfg in render_calls if "eval_anti_alias" in ckpt
    ]
    assert anti_alias_render_configs
    assert all(cfg is not None and cfg["antialiasing"] is True for cfg in anti_alias_render_configs)
    baseline_render_configs = [
        cfg for ckpt, cfg in render_calls if "eval_baseline" in ckpt
    ]
    assert all(cfg is not None and cfg["antialiasing"] is False for cfg in baseline_render_configs)

    assert "chair" in result.chosen_config
    assert result.submission_zip is not None
    assert result.submission_zip.exists()


def test_run_experiment_matrix_fails_closed_when_a_scene_is_skipped(tmp_path):
    from src.common.config import SceneConfig

    good_scene = _chair_scene()
    bad_scene = SceneConfig(
        name="does_not_exist",
        root=Path("VAI_NVS_DATA_ROUND2/does_not_exist"),
        train_images_dir=Path("VAI_NVS_DATA_ROUND2/does_not_exist/train/images"),
        sparse_dir=Path("VAI_NVS_DATA_ROUND2/does_not_exist/train/sparse/0"),
        test_poses_csv=Path("VAI_NVS_DATA_ROUND2/does_not_exist/test/test_poses.csv"),
        submission_dir="does_not_exist",
    )

    def fake_train_fn(scene_arg, variant, output_dir, hyperparam_overrides=None):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        ply = output_dir / "point_cloud.ply"
        _write_fake_ply(ply)
        return ply

    def fake_render_fn(checkpoint, params_list, output_dir, render_config=None):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        written = []
        for params in params_list:
            path = output_dir / params.image_name
            Image.fromarray(
                np.zeros((params.height, params.width, 3), dtype=np.uint8)
            ).save(path)
            written.append(path)
        return written

    def fake_prune_fn(checkpoint_path, bbox_min, bbox_max):
        pruned = Path(checkpoint_path).with_name("point_cloud_pruned.ply")
        pruned.write_bytes(Path(checkpoint_path).read_bytes())
        return pruned

    # bad_scene is skipped by validate_scene (no data at that path), but
    # good_scene (chair, real local data) is still fully trained -- a
    # skipped scene must not silently stop OTHER scenes' progress, only
    # the final submission_zip.
    result = run_experiment_matrix_pipeline(
        scenes=[good_scene, bad_scene],
        screening_train_fn=fake_train_fn,
        final_train_fn=fake_train_fn,
        render_fn=fake_render_fn,
        prune_fn=fake_prune_fn,
        lpips_model=_StubLpipsModel(),
        psnr_max=30.0,
        vram_budget_bytes=16 * 1024**3,
        output_root=tmp_path,
    )

    assert "does_not_exist" in result.skipped_scenes
    assert "chair" in result.chosen_config  # chair's own work was not wasted
    assert result.submission_zip is None  # but the exam voids a missing-scene submission
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_run_experiment_matrix.py -v`
Expected: `FAIL` — `ModuleNotFoundError: No module named 'src.orchestrator.run_experiment_matrix'`.

- [ ] **Step 3: Write `src/orchestrator/run_experiment_matrix.py`**

```python
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from src.common.colmap_io import compute_scene_bbox, load_sparse_scene
from src.common.config import SceneConfig
from src.common.pose_utils import camera_extrinsics_from_colmap, camera_focal_lengths, focal2fov, qvec2rotmat
from src.data_validation.validate_scene import validate_scene
from src.evaluation.compute_metrics import combine_score, compute_pair_metrics
from src.evaluation.make_holdout_split import select_holdout_images
from src.evaluation.screening import variants_needing_full_iteration_verification
from src.evaluation.select_best_config import select_best_candidate
from src.postprocess.vram_guard import count_gaussians_in_ply, estimate_vram_bytes
from src.rendering.render_from_csv import CameraParams, load_test_poses_csv
from src.submission.package_submission import package_submission
from src.submission.validate_submission import validate_submission
from src.training.holdout_scene import build_filtered_scene
from src.training.train_variant import ALL_TRAINING_VARIANTS
from src.training.undistort_scene import undistort_scene


@dataclass
class ExperimentPipelineResult:
    per_scene_scores: dict[str, float] = field(default_factory=dict)
    skipped_scenes: dict[str, list[str]] = field(default_factory=dict)
    chosen_config: dict[str, dict] = field(default_factory=dict)
    all_candidates: dict[str, list[dict]] = field(default_factory=dict)
    validation_problems: list[str] = field(default_factory=list)
    submission_zip: Path | None = None


def _camera_params_for_holdout(sparse, holdout_names, image_dims) -> list[CameraParams]:
    width, height = image_dims
    params = []
    for img in sparse.images.values():
        if img.name not in holdout_names:
            continue
        camera = sparse.cameras[img.camera_id]
        # NOT camera.params[0], camera.params[1] directly -- for
        # SIMPLE_PINHOLE (chair, bonsai) params[1] is cx, not a second
        # focal length. camera_focal_lengths() handles both camera models
        # this pipeline ever holds (see its docstring).
        fx, fy = camera_focal_lengths(camera.model, camera.params)
        r, t = camera_extrinsics_from_colmap(*img.qvec, *img.tvec)
        params.append(CameraParams(
            image_name=img.name, R=r, T=t,
            fov_x=focal2fov(fx, width), fov_y=focal2fov(fy, height),
            width=width, height=height,
        ))
    return params


def _render_config_for(variant, train_output_dir: Path) -> dict:
    """A checkpoint trained with anti-aliasing/appearance-embedding on must
    be rendered (holdout scoring AND the final test_poses.csv render) with
    the same setting on -- see Task 15's real_render_fn(render_config=...).
    train_output_dir is the SAME directory passed as output_dir to the
    train_fn that produced this variant's checkpoint (not derived from the
    checkpoint path itself, which after Task 14's floater pruning is a
    sibling file, not train_output_dir -- but Task 13's
    _save_mean_appearance always writes directly under train_output_dir
    regardless of pruning, so this stays correct for pruned checkpoints
    too).
    """
    appearance_path = Path(train_output_dir) / "mean_appearance.pt"
    return {
        "antialiasing": variant.use_anti_alias,
        "appearance_path": appearance_path if variant.use_appearance_embed else None,
    }


def _score_checkpoint(checkpoint, holdout_params, render_fn, render_dir, gt_dir, lpips_model, psnr_max, render_config):
    rendered_paths = render_fn(checkpoint, holdout_params, render_dir, render_config=render_config)
    scores = []
    for path, params in zip(rendered_paths, holdout_params):
        gt_path = gt_dir / params.image_name
        pred = np.array(Image.open(path).convert("RGB"))
        gt = np.array(Image.open(gt_path).convert("RGB").resize(pred.shape[1::-1]))
        metrics = compute_pair_metrics(pred, gt, lpips_model)
        scores.append(combine_score(metrics["lpips"], metrics["ssim"], metrics["psnr"], psnr_max))
    return float(np.mean(scores)) if scores else 0.0


def run_experiment_matrix_pipeline(
    scenes: list[SceneConfig],
    screening_train_fn, final_train_fn, render_fn, prune_fn, lpips_model,
    psnr_max: float, vram_budget_bytes: int, output_root: Path,
    extra_candidates_by_scene: dict[str, list[dict]] | None = None,
    tiebreak_threshold: float = 0.01,
) -> ExperimentPipelineResult:
    output_root = Path(output_root)
    result = ExperimentPipelineResult()
    scene_render_dirs = {}
    extra_candidates_by_scene = extra_candidates_by_scene or {}

    for scene in scenes:
        report = validate_scene(scene)
        if report.problems:
            result.skipped_scenes[scene.name] = report.problems
            continue

        scene_output = output_root / scene.name
        submission_dir = scene.effective_submission_dir

        working_scene = undistort_scene(scene, scene_output / "undistorted")

        sparse = load_sparse_scene(working_scene.sparse_dir)
        bbox_min, bbox_max = compute_scene_bbox(sparse.points3d, margin_ratio=0.1)
        file_backed_names = {p.name for p in working_scene.train_images_dir.iterdir() if p.is_file()}
        camera_centers = {
            img.name: -np.transpose(qvec2rotmat(np.array(img.qvec))) @ np.array(img.tvec)
            for img in sparse.images.values()
            if img.name in file_backed_names
        }
        holdout_names = set(select_holdout_images(camera_centers, holdout_ratio=0.125))
        filtered_scene = build_filtered_scene(
            working_scene, holdout_names, scene_output / "filtered_scene",
        )

        sample_image = next(working_scene.train_images_dir.iterdir())
        with Image.open(sample_image) as im:
            image_dims = im.size
        holdout_params = _camera_params_for_holdout(sparse, holdout_names, image_dims)

        candidates = []
        output_dir_by_variant: dict[str, Path] = {}
        for variant in ALL_TRAINING_VARIANTS:
            train_output_dir = scene_output / f"eval_{variant.name}"
            output_dir_by_variant[variant.name] = train_output_dir
            eval_checkpoint = screening_train_fn(filtered_scene, variant, train_output_dir)
            render_config = _render_config_for(variant, train_output_dir)
            for use_floater_cleanup in (False, True):
                checkpoint = (
                    prune_fn(eval_checkpoint, bbox_min, bbox_max)
                    if use_floater_cleanup else eval_checkpoint
                )
                score = _score_checkpoint(
                    checkpoint, holdout_params, render_fn,
                    scene_output / f"holdout_{variant.name}_{use_floater_cleanup}",
                    working_scene.train_images_dir, lpips_model, psnr_max, render_config,
                )
                candidates.append({
                    "variant": variant.name, "floater_cleanup": use_floater_cleanup,
                    "score": score,
                    "estimated_vram_bytes": estimate_vram_bytes(count_gaussians_in_ply(checkpoint)),
                    "checkpoint_path": str(checkpoint),
                })

        # Tie-break: whenever any variant's BEST (floater on or off)
        # screening score is within tiebreak_threshold of the leader, the
        # LEADER is re-run at full iterations too (not just the
        # runner-up) -- comparing a full-iteration runner-up against the
        # leader's un-verified screening score would unfairly let it win
        # purely from getting more training. See Task 11's
        # variants_needing_full_iteration_verification.
        best_per_variant = {}
        for c in candidates:
            if c["variant"] not in best_per_variant or c["score"] > best_per_variant[c["variant"]]:
                best_per_variant[c["variant"]] = c["score"]
        variants_to_rerun = variants_needing_full_iteration_verification(
            [{"variant": v, "score": s} for v, s in best_per_variant.items()],
            threshold=tiebreak_threshold,
        )
        for variant_name in variants_to_rerun:
            variant = next(v for v in ALL_TRAINING_VARIANTS if v.name == variant_name)
            tiebreak_output_dir = scene_output / f"tiebreak_{variant.name}"
            full_checkpoint = final_train_fn(filtered_scene, variant, tiebreak_output_dir)
            render_config = _render_config_for(variant, tiebreak_output_dir)
            for candidate in candidates:
                if candidate["variant"] != variant_name:
                    continue
                checkpoint = (
                    prune_fn(full_checkpoint, bbox_min, bbox_max)
                    if candidate["floater_cleanup"] else full_checkpoint
                )
                candidate["score"] = _score_checkpoint(
                    checkpoint, holdout_params, render_fn,
                    scene_output / f"tiebreak_holdout_{variant_name}_{candidate['floater_cleanup']}",
                    working_scene.train_images_dir, lpips_model, psnr_max, render_config,
                )
                candidate["estimated_vram_bytes"] = estimate_vram_bytes(count_gaussians_in_ply(checkpoint))
                candidate["checkpoint_path"] = str(checkpoint)

        # Giai doan 2: bounded extra hyperparameter candidates (bonsai/chair only).
        for extra in extra_candidates_by_scene.get(scene.name, []):
            variant = next(v for v in ALL_TRAINING_VARIANTS if v.name == extra["variant"])
            overrides = {
                k: v for k, v in extra.items()
                if k not in ("variant", "floater_cleanup", "candidate_name")
            }
            extra_output_dir = scene_output / f"extra_{extra['candidate_name']}"
            checkpoint = final_train_fn(
                filtered_scene, variant, extra_output_dir, hyperparam_overrides=overrides,
            )
            render_config = _render_config_for(variant, extra_output_dir)
            use_floater_cleanup = bool(extra.get("floater_cleanup", False))
            if use_floater_cleanup:
                checkpoint = prune_fn(checkpoint, bbox_min, bbox_max)
            score = _score_checkpoint(
                checkpoint, holdout_params, render_fn,
                scene_output / f"extra_holdout_{extra['candidate_name']}",
                working_scene.train_images_dir, lpips_model, psnr_max, render_config,
            )
            candidates.append({
                "variant": extra["variant"], "floater_cleanup": use_floater_cleanup,
                "candidate_name": extra["candidate_name"], "score": score,
                "estimated_vram_bytes": estimate_vram_bytes(count_gaussians_in_ply(checkpoint)),
                "checkpoint_path": str(checkpoint),
                "hyperparam_overrides": overrides,
            })

        result.all_candidates[scene.name] = candidates
        winner = select_best_candidate(candidates, vram_budget_bytes)
        result.chosen_config[scene.name] = winner
        result.per_scene_scores[scene.name] = winner["score"]

        winning_variant = next(v for v in ALL_TRAINING_VARIANTS if v.name == winner["variant"])
        full_training_scene = build_filtered_scene(
            working_scene, set(), scene_output / "full_scene",
        )
        final_output_dir = scene_output / "final_train"
        final_checkpoint = final_train_fn(
            full_training_scene, winning_variant, final_output_dir,
            hyperparam_overrides=winner.get("hyperparam_overrides"),
        )
        final_render_config = _render_config_for(winning_variant, final_output_dir)
        if winner["floater_cleanup"]:
            final_checkpoint = prune_fn(final_checkpoint, bbox_min, bbox_max)

        test_render_dir = scene_output / "test_render"
        test_params_list = load_test_poses_csv(scene.test_poses_csv)
        render_fn(final_checkpoint, test_params_list, test_render_dir, render_config=final_render_config)
        scene_render_dirs[submission_dir] = test_render_dir

    if result.skipped_scenes:
        # Fail closed: the exam voids the ENTIRE score for a missing scene,
        # so a submission.zip already known to be short a scene is worse
        # than no zip at all -- same reasoning as Plan 1's
        # run_baseline_pipeline (src/orchestrator/run_pipeline.py:169-179).
        result.validation_problems = [
            f"scene '{name}' skipped, no submission produced: {problems}"
            for name, problems in result.skipped_scenes.items()
        ]
        result.submission_zip = None
        return result

    submission_zip = output_root / "submission.zip"
    package_submission(scene_render_dirs, submission_zip)
    result.validation_problems = validate_submission(submission_zip, scenes)
    # Fail closed: an artifact validate_submission already flagged as wrong
    # must never be exposed as valid, even though the zip stays on disk for
    # debugging -- same philosophy as Plan 1's run_pipeline.py:180-188.
    result.submission_zip = submission_zip if not result.validation_problems else None
    return result
```

Create `src/orchestrator/__init__.py` if not already present from Plan 1.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_run_experiment_matrix.py -v`
Expected: `PASS` (2 passed).

- [ ] **Step 5: Commit**

```bash
git add src/orchestrator/run_experiment_matrix.py tests/test_run_experiment_matrix.py
git commit -m "Add experiment-matrix orchestrator (screening/final split, undistort fix, fair tie-break, render config, fail-closed)"
```

- [ ] **Step 6: Manual verification on Colab (required, not optional)**

Run `run_experiment_matrix_pipeline` for `chair` alone first (cheapest scene) with
`screening_train_fn = functools.partial(run_training_variant, iterations=15000)` and
`final_train_fn = functools.partial(run_training_variant, iterations=30000)`, no extra
candidates. Confirm: all 5 variants train, `chosen_config["chair"]` is populated, `test_render/`
has real images, `submission_zip` validates clean. Then specifically confirm the `anti_alias` and
`appearance_embed` candidates' holdout renders visibly reflect their trained behavior (compare
against the `baseline` candidate's holdout renders for the same scene) — this is what Task 15
and this task's `_render_config_for` wiring exist to guarantee; do not trust it from code review
alone. Only then move to Task 17's full notebook wiring for all 7 scenes.

---

### Task 17: Wire Giai đoạn 1+2 into the Colab notebooks

**Files:**
- Modify: `notebooks/colab_runner_hcm.ipynb` (5 HCM scenes — Giai đoạn 1 only, no extras)
- Modify: `notebooks/colab_runner_bonsai.ipynb` (`bonsai` + `chair` — Giai đoạn 1 + Giai đoạn 2)

**Interfaces:**
- Consumes: `run_experiment_matrix_pipeline` (Task 16), `build_hyperparam_candidates` (Task 11),
  `write_reproducibility_bundle` (Task 8), `prune_checkpoint` (Task 14).

Each scene is run through its own call to `run_experiment_matrix_pipeline` (`scenes=[scene]`),
not one call for all scenes at once — a Colab disconnect partway through then loses at most one
scene's progress, not all of them.

- [ ] **Step 1: Add a new Bước 9 cell to `notebooks/colab_runner_hcm.ipynb` (after the Bước 8
  diagnosis cell from Task 12)**

Markdown cell:

```markdown
## Bước 9 — Giai đoạn 1: ma trận 5 biến thể cho từng scene HCM

Mỗi scene chạy riêng 1 lệnh (không gộp cả 5 scene vào 1 lệnh) — nếu Colab bị ngắt giữa chừng chỉ
mất tiến độ của đúng 1 scene, không mất cả 5. Screening ở 15000 iteration; biến thể thắng (hoặc
sít sao, tự động re-run) được train lại ở 30000 iteration đầy đủ trước khi ship.
```

Code cell:

```python
import functools

from src.orchestrator.run_experiment_matrix import run_experiment_matrix_pipeline
from src.postprocess.prune_floaters import prune_checkpoint
from src.submission.reproducibility_bundle import write_reproducibility_bundle
from src.training.train_variant import run_training_variant

screening_train_fn = functools.partial(run_training_variant, iterations=15000)
final_train_fn = functools.partial(run_training_variant, iterations=30000)

hcm_scenes = [s for s in load_scenes() if s.name.startswith("HCM")]
matrix_results = {}
for scene in hcm_scenes:
    result = run_experiment_matrix_pipeline(
        scenes=[scene],
        screening_train_fn=screening_train_fn,
        final_train_fn=final_train_fn,
        render_fn=real_render_fn,
        prune_fn=prune_checkpoint,
        lpips_model=load_lpips_model(),
        psnr_max=30.0,
        vram_budget_bytes=20 * 1024**3,
        output_root=OUTPUT_ROOT,
    )
    matrix_results[scene.name] = result
    write_reproducibility_bundle(
        scene.name, result.chosen_config[scene.name], result.all_candidates[scene.name],
        OUTPUT_ROOT / "reproducibility",
    )
    print(f"{scene.name}: winner={result.chosen_config[scene.name]['variant']} "
          f"floater_cleanup={result.chosen_config[scene.name]['floater_cleanup']} "
          f"score={result.per_scene_scores[scene.name]:.4f}")
```

- [ ] **Step 2: Add a new Bước 9 cell to `notebooks/colab_runner_bonsai.ipynb` (after the
  existing Bước 7 cell)**

Markdown cell:

```markdown
## Bước 9 — Giai đoạn 1+2: ma trận 5 biến thể + bounded search cho bonsai/chair

`bonsai` và `chair` là 2 scene thấp điểm nhất — ngoài ma trận 5 biến thể chuẩn, thêm tối đa 4
candidate hyperparameter mỗi scene (`EXTRA_OVERRIDES_BY_SCENE` bên dưới), chọn dựa trên Bước 8
(chẩn đoán) của `colab_runner_hcm.ipynb`. Sửa `EXTRA_OVERRIDES_BY_SCENE` trước khi chạy nếu chẩn
đoán chỉ ra hướng khác.
```

Code cell:

```python
import functools

from src.evaluation.screening import build_hyperparam_candidates
from src.orchestrator.run_experiment_matrix import run_experiment_matrix_pipeline
from src.postprocess.prune_floaters import prune_checkpoint
from src.submission.reproducibility_bundle import write_reproducibility_bundle
from src.training.train_variant import run_training_variant

screening_train_fn = functools.partial(run_training_variant, iterations=15000)
final_train_fn = functools.partial(run_training_variant, iterations=30000)

# Sửa theo phát hiện của Bước 8 (chẩn đoán) trước khi chạy. Mặc định dưới
# đây thử: threshold thấp hơn (giữ Gaussian nhỏ lâu hơn, giảm floater),
# threshold cao hơn (ngược lại, kiểm tra không lệch hướng), và 45000
# iteration (kiểm tra điểm còn tăng hay đã bão hoà ở 30000).
EXTRA_OVERRIDES_BY_SCENE = {
    "bonsai": [
        {"densify_grad_threshold": 0.0005},
        {"densify_grad_threshold": 0.0015},
        {"iterations": 45000},
    ],
    "chair": [
        {"densify_grad_threshold": 0.0005},
        {"densify_grad_threshold": 0.0015},
        {"iterations": 45000},
    ],
}

extra_candidates_by_scene = {
    scene_name: build_hyperparam_candidates(
        {"variant": "baseline"}, overrides, label_prefix=scene_name,
    )
    for scene_name, overrides in EXTRA_OVERRIDES_BY_SCENE.items()
}

bonsai_chair_scenes = [s for s in load_scenes() if s.name in ("bonsai", "chair")]
for scene in bonsai_chair_scenes:
    result = run_experiment_matrix_pipeline(
        scenes=[scene],
        screening_train_fn=screening_train_fn,
        final_train_fn=final_train_fn,
        render_fn=real_render_fn,
        prune_fn=prune_checkpoint,
        lpips_model=load_lpips_model(),
        psnr_max=30.0,
        vram_budget_bytes=20 * 1024**3,
        output_root=OUTPUT_ROOT,
        extra_candidates_by_scene=extra_candidates_by_scene,
    )
    write_reproducibility_bundle(
        scene.name, result.chosen_config[scene.name], result.all_candidates[scene.name],
        OUTPUT_ROOT / "reproducibility",
    )
    print(f"{scene.name}: winner={result.chosen_config[scene.name]} "
          f"score={result.per_scene_scores[scene.name]:.4f}")
```

- [ ] **Step 3: Manual verification on Colab (required)**

Run Bước 9 in `colab_runner_hcm.ipynb` for at least one HCM scene, and Bước 9 in
`colab_runner_bonsai.ipynb` for `chair`, confirming: `reproducibility/<scene>/chosen_config.yaml`
and `all_candidates_scores.csv` are written to Drive, and `all_candidates_scores.csv` for
`chair`/`bonsai` has `10 + len(EXTRA_OVERRIDES_BY_SCENE[scene])` rows.

- [ ] **Step 4: Commit**

```bash
git add notebooks/colab_runner_hcm.ipynb notebooks/colab_runner_bonsai.ipynb
git commit -m "Wire Giai doan 1+2 experiment matrix into both Colab notebooks"
```

---

### Task 18: Final merge and reproducibility bundle close-out (Giai đoạn 3)

**Files:**
- Modify: `notebooks/colab_runner_hcm.ipynb`

**Interfaces:**
- Consumes: the existing Bước 8 (7-scene) merge cell — already present, already scans
  `OUTPUT_ROOT/<scene>/test_render/` for all 7 scenes, which is exactly where Task 16's
  `run_experiment_matrix_pipeline` writes its final render. No change needed to that cell logic.
  This task only adds packaging the reproducibility bundle **separately** from
  `submission.zip` — the exam's submission format (spec 1.5) allows only rendered images inside
  it.

- [ ] **Step 1: Add a new cell after the existing 7-scene merge cell**

Markdown cell:

```markdown
## Bước 10 — Đóng gói reproducibility bundle (riêng, KHÔNG nằm trong submission.zip)

`submission.zip` chỉ được chứa ảnh render theo đúng định dạng đề bài — mọi config/log/score
table nộp kèm (nếu BTC yêu cầu) phải nằm trong 1 file zip riêng.
```

Code cell:

```python
import shutil

reproducibility_dir = OUTPUT_ROOT / "reproducibility"
reproducibility_zip = OUTPUT_ROOT / "reproducibility_bundle"
if reproducibility_dir.exists():
    shutil.make_archive(str(reproducibility_zip), "zip", str(reproducibility_dir))
    print(f"reproducibility_bundle.zip: {reproducibility_zip}.zip")
else:
    print("WARNING: reproducibility/ chưa tồn tại -- chạy Bước 9 ở cả 2 notebook trước.")
```

- [ ] **Step 2: Manual verification on Colab (required)**

After all 7 scenes have gone through Task 17's Bước 9 in their respective notebooks, run the
existing Bước 8 merge cell, then this new Bước 10 cell. Confirm `submission.zip` validates clean
(existing `validate_submission` check in Bước 8) and `reproducibility_bundle.zip` contains a
subfolder per scene with `chosen_config.yaml` and `all_candidates_scores.csv`.

- [ ] **Step 3: Commit**

```bash
git add notebooks/colab_runner_hcm.ipynb
git commit -m "Add reproducibility bundle packaging, kept separate from submission.zip"
```

---

## Self-Review Summary

- **Spec coverage:** advanced-techniques building blocks → Task 1-8 (carried over from
  `docs/superpowers/plans/2026-07-18-advanced-techniques.md`, its own Task 11 already done and
  not repeated). Giai đoạn 0 (diagnosis) → Task 9/10/12. Giai đoạn 1 (5-variant matrix,
  reduced-iteration screening, tie-break) → Task 11/13/15/16/17. Giai đoạn 2 (bounded
  hyperparameter search for bonsai/chair) → Task 11/16/17. Giai đoạn 3 (final retrain, blind
  render, reproducibility bundle, merge) → Task 16 (retrain+render+package are inside the
  orchestrator itself) + Task 18 (reproducibility close-out). Out-of-scope items from the spec
  (full grid across all 7 scenes, A100, parallel sessions) are not implemented anywhere in this
  plan.
- **Placeholder scan:** no TBD/TODO; every code step has complete, runnable code; no "similar to
  Task N" shortcuts. Task 14 closes a real gap discovered while first drafting this plan: the
  original advanced-techniques document's Task 2 only produces `compute_prune_mask` (a
  boolean-array function over already-loaded Gaussian attributes, not file I/O) — nothing in that
  document ever wraps it into a `checkpoint_path -> checkpoint_path` callable, which Task 16's
  `prune_fn` parameter requires. Task 14 closes that gap.
- **Type consistency:** `run_training_variant`'s `hyperparam_overrides` parameter (Task 13,
  included from the first draft rather than bolted on later) is used consistently with the same
  name/shape (`dict[str, object] | None`) in Task 16's `final_train_fn` calls and Task 17's
  notebook cells. `ExperimentPipelineResult.chosen_config` entries carry an optional
  `hyperparam_overrides` key (only present for Giai đoạn 2 candidates) — Task 16's final-retrain
  call uses `.get("hyperparam_overrides")`, `None` for the 10 base variant/floater candidates,
  the dict for extras. `prune_fn` is consistently `prune_checkpoint` (Task 14) everywhere it's
  referenced: Task 16's test/interface, and both notebook cells in Task 17.
- **External review pass (2026-07-23):** a second reviewer read the first draft of this plan
  against the actual codebase and found 7 real defects, all confirmed by independently checking
  the referenced files (not taken on faith — see the `superpowers:receiving-code-review` process
  for how each was verified) and fixed in this revision:
  1. *Critical* — `real_render_fn` rendered every checkpoint with all-default `pipe`/appearance
     state regardless of which variant trained it, so `anti_alias`/`full_stack`/
     `appearance_embed` checkpoints would have been scored and shipped as if those techniques
     were off. Fixed by Task 15 (new: `render_config` parameter on `real_render_fn`) and Task 16
     (`_render_config_for`, threaded through every render/score call site).
  2. *High* — the tie-break re-run only re-verified the runner-up at full iterations, comparing
     it unfairly against the leader's un-verified screening-iteration score. Fixed by Task 11's
     `variants_needing_full_iteration_verification` (re-runs the leader too whenever any
     runner-up is close) and Task 16 using it.
  3. *High* — holdout camera focal length used `camera.params[0]`/`params[1]` directly, which is
     wrong for `SIMPLE_PINHOLE` (`params[1]` is `cx`, not `fy`) — exactly the trap
     `src/common/pose_utils.py`'s `camera_focal_lengths()` docstring already documents. Fixed in
     Task 16 by using that existing helper instead of raw indexing.
  4. *High* — a `{"iterations": N}` entry in `hyperparam_overrides` set `opt.iterations` but the
     training loop's actual bound was the separate function parameter `iterations`, so the
     override never changed how long training actually ran. Fixed in Task 13 via
     `effective_iterations`, popped out of the overrides dict and used everywhere the loop bound,
     save call, and checkpoint path are derived.
  5. *High* — `prune_checkpoint` called `GaussianModel.prune_points()`, which requires
     `self.optimizer` (only set by `training_setup()`, never called on a load-only path) and
     would crash with `AttributeError: 'NoneType' object has no attribute 'param_groups'`. Fixed
     in Task 14 by filtering the six tensors `save_ply` reads directly, with no optimizer
     involved.
  6. *Medium* — the orchestrator always produced a `submission_zip` regardless of skipped scenes
     or validation problems, unlike Plan 1's deliberately fail-closed `run_baseline_pipeline`.
     Fixed in Task 16 by replicating that exact fail-closed logic.
  7. *Medium* — the reproducibility CSV's fixed `fieldnames` silently dropped
     `candidate_name`/`hyperparam_overrides`/`fallback_reason`, losing exactly the information
     needed to trace which Giai đoạn 2 override actually won for `bonsai`/`chair`. Fixed in
     Task 8 by adding those columns (JSON-encoding `hyperparam_overrides`).
