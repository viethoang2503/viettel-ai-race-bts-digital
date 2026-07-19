# Advanced Techniques & Auto-Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Turn the working baseline pipeline (`docs/superpowers/plans/2026-07-18-core-nvs-
pipeline.md`) into a score-maximizing pipeline by adding the techniques deferred there: floater
cleanup, depth regularization, anti-aliasing, appearance embedding, a VRAM guard for the A4000
20GB inference target, auto-selection of the best-scoring config per scene, a real
reproducibility bundle, and a visual QA step — covering spec sections 6, 7, 9, 13, 15.

**Architecture:** Every technique is trained/evaluated as one of several *candidates* per scene
(baseline / +depth_reg / +anti_alias / +appearance_embed / full_stack, each optionally with
floater cleanup applied as a post-process), scored on the same leak-free holdout mechanism from
Plan 1's Task 8b, filtered by the VRAM guard, and the highest-scoring candidate that fits the
budget is auto-selected per scene — never hand-picked or assumed in advance. Pure-Python/pure-
math pieces (VRAM estimation, floater mask computation, sparse depth target computation,
appearance-embedding tensor math, depth-loss tensor math, candidate selection, reproducibility
bundle, visual QA sampling) get real `pytest` unit tests on the local no-GPU machine, exactly
like Plan 1. The pieces that require the actual differentiable CUDA rasterizer (the training
loop wiring itself) are written with complete code against the vendored API as documented in
the upstream repo, but — like Plan 1's GPU-dependent tasks — are flagged for manual
verification/adjustment on Colab, since this local machine cannot execute CUDA code to verify
them directly.

**Tech Stack:** Same as Plan 1 (Python 3.10+, PyTorch, `lpips`, `scikit-image`, `numpy`,
`pyyaml`, `pytest`), plus the vendored `graphdeco-inria/gaussian-splatting` submodule from
Plan 1 Task 1, plus a second vendored submodule for anti-aliasing (Task 4).

## Global Constraints

- **Prerequisite:** Plan 1 (`docs/superpowers/plans/2026-07-18-core-nvs-pipeline.md`) must be
  implemented first — this plan imports `SceneConfig`, `load_sparse_scene`, `combine_score`,
  `compute_pair_metrics`, `load_lpips_model`, `select_holdout_images`, `build_filtered_scene`,
  `build_train_argv`, `validate_scene`, `render_all`, `load_test_poses_csv`,
  `package_submission`, `validate_submission` from Plan 1's modules without redefining them.
- Floater cleanup is a **post-processing step**, not a training-time variant — it can be applied
  (or not) to the output of any trained checkpoint. The training-time variants are: `baseline`,
  `depth_reg`, `anti_alias`, `appearance_embed`, `full_stack` (all three combined). Combined
  with the floater-cleanup on/off choice, this yields up to 10 scored candidates per scene
  (5 trained checkpoints x {raw, floater-cleaned}); `select_best_candidate` (Task 6) picks the
  winner per scene from whichever candidates were actually trained and fit the VRAM budget.
- VRAM budget: target 20GB (RTX A4000, per exam round-1 infra note), with a conservative safety
  margin — this plan uses 16GB as the pass/fail threshold in `fits_within_vram_budget`.
- Where this plan documents an assumption about the exact vendored API (e.g. `GaussianModel`
  constructor args, `render()` return dict keys, `prune_points()` signature) that cannot be
  verified without the actual checked-out `third_party/gaussian-splatting` submodule, the task
  says so explicitly and gives the adjustment procedure — the same pattern Plan 1 used for
  `scene/colmap_loader.py` imports.

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
    — a **conservative heuristic**, not a guaranteed prediction (documented as such); the
    authoritative check is always `torch.cuda.max_memory_allocated()` measured for real on
    Colab (Task 1 Step 6).
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

```bash
pytest tests/test_vram_guard.py -v
```

Expected: `FAIL` — `ModuleNotFoundError: No module named 'src.postprocess'`.

- [ ] **Step 3: Write `src/postprocess/vram_guard.py`**

```python
from __future__ import annotations

from pathlib import Path


def count_gaussians_in_ply(ply_path: Path) -> int:
    """Read only the ASCII PLY header to get the vertex count, without
    loading the (potentially multi-GB) binary body.
    """
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

    Per-Gaussian float count: position(3) + rotation quaternion(4) +
    scale(3) + opacity(1) + spherical harmonics coefficients
    ((sh_degree+1)^2 * 3 channels, including the DC term). A 2x multiplier
    accounts for the CUDA rasterizer's tile-based intermediate buffers,
    which is a rough approximation, not a guarantee — always confirm with
    a real `torch.cuda.max_memory_allocated()` measurement on Colab before
    trusting this near the A4000's 20GB ceiling (see Task 1 Step 6).
    """
    floats_per_gaussian = 3 + 4 + 3 + 1 + (sh_degree + 1) ** 2 * 3
    rendering_overhead_multiplier = 2.0
    return int(num_gaussians * floats_per_gaussian * dtype_bytes * rendering_overhead_multiplier)


def fits_within_vram_budget(ply_path: Path, budget_bytes: int, sh_degree: int = 3) -> bool:
    num_gaussians = count_gaussians_in_ply(ply_path)
    return estimate_vram_bytes(num_gaussians, sh_degree=sh_degree) <= budget_bytes
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_vram_guard.py -v
```

Expected: `PASS` (5 passed).

- [ ] **Step 5: Commit**

```bash
git add src/postprocess/vram_guard.py src/postprocess/__init__.py tests/test_vram_guard.py
git commit -m "Add VRAM budget estimation guard for A4000 inference target"
```

- [ ] **Step 6: Manual verification on Colab (requires GPU)**

After rendering a real checkpoint (Plan 1 Task 9 Step 6), compare the heuristic against reality:

```python
import torch
from src.postprocess.vram_guard import count_gaussians_in_ply, estimate_vram_bytes

torch.cuda.reset_peak_memory_stats()
# ... run the real render_all() from Plan 1 on this checkpoint ...
actual_bytes = torch.cuda.max_memory_allocated()
n = count_gaussians_in_ply("path/to/point_cloud.ply")
print("estimated:", estimate_vram_bytes(n), "actual:", actual_bytes)
```

If the heuristic underestimates actual usage by more than ~30%, tighten the
`rendering_overhead_multiplier` in `estimate_vram_bytes` accordingly before relying on it in
Task 6's auto-selection.

---

### Task 2: Floater/background prune mask

**Files:**
- Create: `src/postprocess/prune_floaters.py`
- Test: `tests/test_prune_floaters.py`

**Interfaces:**
- Consumes: `compute_scene_bbox` (Plan 1 Task 4).
- Produces: `compute_prune_mask(xyz: np.ndarray, opacity: np.ndarray, scales: np.ndarray, bbox_min: np.ndarray, bbox_max: np.ndarray, opacity_threshold: float = 0.05, max_scale_percentile: float = 99.5) -> np.ndarray`
  returning a boolean `(N,)` keep-mask (`True` = keep). A Gaussian is pruned if its center is
  outside `[bbox_min, bbox_max]`, its opacity is below `opacity_threshold`, or its largest scale
  axis is above the `max_scale_percentile`-th percentile of all Gaussians' largest scale axis
  (percentile-based, not an absolute cutoff, so it doesn't need per-scene retuning — an
  oversized floater blob is, by construction, an outlier relative to the rest of the scene's
  Gaussians).

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
    xyz = np.zeros((2, 3))
    opacity = np.array([0.5, 0.01])
    scales = np.full((2, 3), 0.1)
    mask = compute_prune_mask(
        xyz, opacity, scales, bbox_min=np.array([-1, -1, -1]), bbox_max=np.array([1, 1, 1]),
        opacity_threshold=0.05,
    )
    assert mask.tolist() == [True, False]


def test_prunes_outlier_scale_gaussian():
    rng = np.random.default_rng(0)
    n_normal = 200
    xyz = rng.uniform(-0.5, 0.5, size=(n_normal, 3))
    xyz = np.vstack([xyz, [[0.0, 0.0, 0.0]]])  # one more, will get huge scale
    opacity = np.full(n_normal + 1, 0.5)
    scales = np.full((n_normal + 1, 3), 0.05)
    scales[-1] = [50.0, 50.0, 50.0]  # floater: absurdly large

    mask = compute_prune_mask(
        xyz, opacity, scales, bbox_min=np.array([-1, -1, -1]), bbox_max=np.array([1, 1, 1]),
        max_scale_percentile=99.5,
    )
    assert mask[-1] == False
    assert mask[:n_normal].all()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_prune_floaters.py -v
```

Expected: `FAIL` — `ModuleNotFoundError: No module named 'src.postprocess.prune_floaters'`.

- [ ] **Step 3: Write `src/postprocess/prune_floaters.py`**

```python
from __future__ import annotations

import numpy as np


def compute_prune_mask(
    xyz: np.ndarray,
    opacity: np.ndarray,
    scales: np.ndarray,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    opacity_threshold: float = 0.05,
    max_scale_percentile: float = 99.5,
) -> np.ndarray:
    inside_bbox = np.all((xyz >= bbox_min) & (xyz <= bbox_max), axis=1)
    opacity_ok = opacity.reshape(-1) >= opacity_threshold

    max_scale_per_gaussian = scales.max(axis=1)
    scale_cutoff = np.percentile(max_scale_per_gaussian, max_scale_percentile)
    scale_ok = max_scale_per_gaussian <= scale_cutoff

    return inside_bbox & opacity_ok & scale_ok
```

Create `src/postprocess/__init__.py` (empty, if not already present from Task 1).

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_prune_floaters.py -v
```

Expected: `PASS` (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/postprocess/prune_floaters.py tests/test_prune_floaters.py
git commit -m "Add percentile-based floater/background prune mask computation"
```

- [ ] **Step 6: Manual verification on Colab — apply the mask to a real checkpoint**

The vendored `GaussianModel` (in `third_party/gaussian-splatting/scene/gaussian_model.py`)
already implements a `prune_points(mask)` method used internally during adaptive density
control — reuse it rather than re-implementing Gaussian-tensor pruning:

```python
import sys
sys.path.insert(0, "third_party/gaussian-splatting")
import torch
import numpy as np
from scene.gaussian_model import GaussianModel
from src.common.colmap_io import load_sparse_scene, compute_scene_bbox
from src.postprocess.prune_floaters import compute_prune_mask

gaussians = GaussianModel(3)
gaussians.load_ply("path/to/point_cloud.ply")

sparse = load_sparse_scene("VAI_NVS_DATA_ROUND2/chair/train/sparse/0")
bbox_min, bbox_max = compute_scene_bbox(sparse.points3d, margin_ratio=0.1)

xyz = gaussians.get_xyz.detach().cpu().numpy()
opacity = gaussians.get_opacity.detach().cpu().numpy()
scales = gaussians.get_scaling.detach().cpu().numpy()

keep_mask = compute_prune_mask(xyz, opacity, scales, bbox_min, bbox_max)
prune_mask = torch.from_numpy(~keep_mask).to(gaussians.get_xyz.device)
gaussians.prune_points(prune_mask)
gaussians.save_ply("path/to/point_cloud_pruned.ply")
print("kept", keep_mask.sum(), "of", len(keep_mask))
```

Expected: a smaller `.ply` file; visually, floating blobs in the sky/background should be gone
when the pruned checkpoint is re-rendered (Plan 1 Task 9). If `prune_points` doesn't exist or
has a different signature in the checked-out submodule version, open `scene/gaussian_model.py`
and adjust the call accordingly — its internal densification logic always needs some form of
"drop these indices" method, so an equivalent exists even if named differently.

---

### Task 3: Sparse depth targets from COLMAP tracks

**Files:**
- Create: `src/training/sparse_depth.py`
- Test: `tests/test_sparse_depth.py`

**Interfaces:**
- Produces: `compute_sparse_depth_targets(qvec: np.ndarray, tvec: np.ndarray, xys: np.ndarray, point3d_ids: np.ndarray, points3d: dict) -> tuple[np.ndarray, np.ndarray]`
  returning `(pixel_xy, depth)` arrays for every 2D keypoint in this training image that has a
  valid associated 3D point (`point3d_ids[i] != -1`). `pixel_xy` comes directly from the
  observed COLMAP keypoint location (`xys[i]`), not a re-projection — COLMAP already recorded
  where in the image this 3D point was observed. `depth` is the camera-space Z of that 3D point,
  computed using the **raw COLMAP world-to-camera** `(R, t)` from `qvec2rotmat(qvec)` and `tvec`
  directly — this is deliberately NOT the transposed `(R, T)` convention used for the vendored
  `Camera` class (Plan 1 Task 3); depth is a camera-space quantity, so using the wrong rotation
  convention here would silently produce wrong depth values without any error.

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
    # can happen if points3D.bin and images.bin are slightly out of sync
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

    # camera-space z = (I @ [0,0,2]) + [1,2,3] -> z component = 2 + 3 = 5
    np.testing.assert_allclose(depth, [5.0], atol=1e-10)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_sparse_depth.py -v
```

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
    """Depth-supervision targets for one training image, from its COLMAP
    2D-3D track associations. Uses the RAW COLMAP world-to-camera (R, t) —
    NOT the transposed convention used for the vendored Camera class —
    because depth is a camera-space quantity computed directly from this
    rotation, not passed through the renderer's own view-matrix transpose.
    """
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

```bash
pytest tests/test_sparse_depth.py -v
```

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
  - `AppearanceEmbedding(nn.Module)` — holds one learnable `(3,3)` affine matrix and `(3,)` bias
    per training image (`num_images`), initialized to identity/zero so training starts as a
    no-op.
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
    affine = torch.eye(3) * 2.0  # would push values to 2.0
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

```bash
pytest tests/test_appearance_embedding.py -v
```

Expected: `FAIL` — `ModuleNotFoundError: No module named 'src.training.appearance_embedding'`.

- [ ] **Step 3: Write `src/training/appearance_embedding.py`**

```python
from __future__ import annotations

import torch
import torch.nn as nn


def apply_appearance(rgb: torch.Tensor, affine: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """rgb: (3,H,W) in [0,1]. affine: (3,3). bias: (3,). Returns (3,H,W)."""
    c, h, w = rgb.shape
    flat = rgb.reshape(c, h * w)
    transformed = affine @ flat + bias.unsqueeze(1)
    return transformed.reshape(c, h, w).clamp(0.0, 1.0)


class AppearanceEmbedding(nn.Module):
    """One learnable (3,3) affine + (3,) bias per training image, indexed
    by image position in the training set. Initialized to identity/zero so
    training starts as a photometric no-op and only diverges from it where
    the loss actually benefits from explaining away per-image exposure
    variation — it must never be applied when rendering novel test poses,
    since there is no "true" appearance code for an unseen view (Plan
    section 6 in the design spec: use the mean of all training embeddings
    instead, computed by the caller, not by this module).
    """

    def __init__(self, num_images: int):
        super().__init__()
        self.affine = nn.Parameter(torch.eye(3).unsqueeze(0).repeat(num_images, 1, 1))
        self.bias = nn.Parameter(torch.zeros(num_images, 3))

    def forward(self, image_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.affine[image_idx], self.bias[image_idx]

    def mean_affine_bias(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Canonical appearance for novel-view rendering: the average
        learned correction across all training images, since a test pose
        has no ground-truth appearance to match.
        """
        return self.affine.mean(dim=0).detach(), self.bias.mean(dim=0).detach()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_appearance_embedding.py -v
```

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
  returns the mean L1 distance to `sparse_depths`. Returns `torch.tensor(0.0)` (not NaN, not an
  error) when there are zero sparse points for an image, so the training loop can always add
  this term unconditionally.

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

```bash
pytest tests/test_depth_loss.py -v
```

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

```bash
pytest tests/test_depth_loss.py -v
```

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
  `estimated_vram_bytes: int`, `checkpoint_path: str`. Filters to candidates with
  `estimated_vram_bytes <= vram_budget_bytes`; among those, returns the one with the highest
  `score`. If none fit the budget, returns the smallest-`estimated_vram_bytes` candidate instead
  and sets `fallback_reason: "no candidate fit the VRAM budget"` on the returned dict, rather
  than raising — a scene must always ship *something*.

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
    assert best["variant"] == "depth_reg"  # smaller of the two, even though not under budget
    assert best["fallback_reason"] == "no candidate fit the VRAM budget"


def test_raises_on_empty_candidate_list():
    import pytest
    with pytest.raises(ValueError):
        select_best_candidate([], vram_budget_bytes=2_000)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_select_best_config.py -v
```

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

```bash
pytest tests/test_select_best_config.py -v
```

Expected: `PASS` (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/evaluation/select_best_config.py tests/test_select_best_config.py
git commit -m "Add VRAM-aware best-candidate selection per scene"
```

---

### Task 7: Anti-aliasing variant — use the baseline's native `antialiasing` flag

**IMPORTANT — this task was rewritten after inspecting the actual checked-out submodule.** The
originally-planned approach (vendoring the Mip-Splatting fork as a second rasterizer submodule)
is unnecessary and must NOT be done: the vendored `graphdeco-inria/gaussian-splatting` already
ships native anti-aliasing. Verified in the checkout:
- `third_party/gaussian-splatting/arguments/__init__.py` → `PipelineParams` has
  `self.antialiasing = False`.
- `third_party/gaussian-splatting/gaussian_renderer/__init__.py` → `render()` passes
  `antialiasing=pipe.antialiasing` into `GaussianRasterizationSettings`.
- `third_party/gaussian-splatting/submodules/diff-gaussian-rasterization/` → the CUDA kernels
  (`forward.cu`, `backward.cu`, `rasterize_points.cu`) all branch on an `antialiasing` flag.

So "anti-aliasing" is not a different rasterizer — it is a single boolean on the pipeline params
the render loop already reads. This eliminates the second submodule, the `setup_colab.sh`
`--variant` rebuild dance, the `--force-reinstall` swap, and the import-ordering hazard entirely.
There is nothing to vendor and no CUDA to rebuild per variant.

**Files:**
- None. This task adds no new files and no submodule. It exists only to record the finding above
  and to define how the `anti_alias` variant maps onto the existing flag — the actual wiring
  lives in Task 8's `run_training_variant` (`pipe.antialiasing = variant.use_anti_alias`) and in
  the render call used for evaluation/submission (`pipe.antialiasing` must match the value the
  chosen checkpoint was trained with).

- [ ] **Step 1: Confirm the flag exists in the checkout (no code change)**

```bash
cd "/home/howard/Documents/viettel ai race/computer vision"
grep -n "antialiasing" third_party/gaussian-splatting/arguments/__init__.py
grep -n "antialiasing" third_party/gaussian-splatting/gaussian_renderer/__init__.py
```

Expected: `PipelineParams` defines `self.antialiasing = False`, and `render()` reads
`antialiasing=pipe.antialiasing`. If either grep comes back empty (an older submodule pin than
the one checked out here, which predates native anti-aliasing), STOP and revisit — only then
would vendoring Mip-Splatting become necessary, and that decision should be escalated rather
than assumed.

- [ ] **Step 2: Record the mapping (no code, just the contract Task 8 implements)**

The `anti_alias` and `full_stack` training variants set `pipe.antialiasing = True` for both the
training render loop AND every later render of that checkpoint (holdout eval in Task 9 and the
final `test_poses.csv` render). A checkpoint trained with anti-aliasing on must be rendered with
it on — mismatching the flag between train and inference produces subtly wrong images. Task 8's
`run_training_variant` and Task 9's render calls are where this is enforced; there is no separate
artifact for this task.

---

### Task 8: Variant-aware training loop

This is the one piece of this plan that cannot be given a from-scratch, independently-verified
implementation without the actual checked-out vendored source in front of the implementer — it
directly drives the CUDA-differentiable renderer. It is written completely, reusing the
pure/tested pieces from Tasks 3-5, against the well-established vendored API (`GaussianModel`,
`render()`, `l1_loss`/`ssim` from `utils.loss_utils`) — but must be verified against the actual
checked-out `third_party/gaussian-splatting/train.py` before trusting it, exactly like Plan 1's
GPU-dependent tasks.

**Files:**
- Create: `src/training/train_variant.py`
- Test: `tests/test_train_variant.py` (variant config logic only — no GPU)

**Interfaces:**
- Produces:
  - `TrainingVariant` frozen dataclass: `name: str`, `use_depth_reg: bool`,
    `use_anti_alias: bool`, `use_appearance_embed: bool`.
  - `ALL_TRAINING_VARIANTS: list[TrainingVariant]` — the 5 named in Global Constraints.
  - `run_training_variant(scene: SceneConfig, variant: TrainingVariant, output_dir: Path, iterations: int) -> Path`
    — GPU-dependent; runs the actual training loop and returns the final checkpoint `.ply` path.

- [ ] **Step 1: Write the failing test (variant config only, no GPU)**

```python
# tests/test_train_variant.py
from src.training.train_variant import ALL_TRAINING_VARIANTS, TrainingVariant


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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_train_variant.py -v
```

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
    """Construct the args object the vendored Scene/render expect.

    Verified against the checkout: `scene.Scene.__init__(self, args, gaussians, ...)`
    reads `args.model_path`, `args.source_path`, `args.images`, `args.depths`,
    `args.eval`, `args.train_test_exp`, `args.white_background`, `args.data_device`,
    `args.sh_degree`; `gaussian_renderer.render` reads `pipe.antialiasing`,
    `pipe.debug`, `pipe.convert_SHs_python`, `pipe.compute_cov3D_python`. Rather
    than hand-roll a Namespace and risk missing a field, build the real
    ModelParams/PipelineParams/OptimizationParams via an ArgumentParser (their
    ParamGroup base fills every default), then override the few paths we control.
    Returns (dataset, pipe, opt) — the three extracted GroupParams objects the
    vendored `training()` uses.
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
    dataset.eval = False  # holdout handling is done upstream by Plan 1 Task 8b

    pipe = pp.extract(args)
    pipe.antialiasing = use_anti_alias  # Task 7: native flag, no separate rasterizer

    opt = op.extract(args)
    return dataset, pipe, opt


def run_training_variant(
    scene: SceneConfig, variant: TrainingVariant, output_dir: Path, iterations: int,
) -> Path:
    """GPU-dependent training loop for one variant. Mirrors the vendored
    train.py::training() structure, verified against the checked-out
    third_party/gaussian-splatting (commit 54c035f):
      - Scene(dataset_args, gaussians) — takes a ModelParams-like args
        object, NOT a path (see _build_dataset_args).
      - GaussianModel(sh_degree, optimizer_type="default")
      - render(cam, gaussians, pipe, bg) -> dict with keys "render",
        "viewspace_points", "visibility_filter", "radii", "depth".
      - anti-aliasing is pipe.antialiasing (Task 7), not a separate module.

    Still GPU-dependent and must be run on Colab; the pieces that could be
    verified against the real source without a GPU (constructor shapes,
    param fields, render dict keys, flag names) have been.
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
    opt.iterations = iterations

    gaussians = GaussianModel(dataset.sh_degree)
    gs_scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)

    # Depth regularization reads COLMAP 2D-3D tracks; only load sparse when needed.
    sparse = load_sparse_scene(scene.sparse_dir) if variant.use_depth_reg else None

    train_cameras = gs_scene.getTrainCameras()
    appearance = (
        AppearanceEmbedding(num_images=len(train_cameras)).cuda()
        if variant.use_appearance_embed else None
    )
    if appearance is not None:
        appearance_optimizer = torch.optim.Adam(appearance.parameters(), lr=1e-3)

    bg_color = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda")

    for iteration in range(1, iterations + 1):
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

    gs_scene.save(iterations)  # writes point_cloud/iteration_<N>/point_cloud.ply
    final_ply = output_dir / "point_cloud" / f"iteration_{iterations}" / "point_cloud.ply"
    return final_ply


def _save_mean_appearance(appearance, output_dir: Path) -> None:
    """Persist the mean appearance embedding for novel-view rendering (there
    is no per-image code for an unseen test pose — see AppearanceEmbedding
    docstring). Saved next to the checkpoint so the render step can apply it.
    """
    import torch

    affine, bias = appearance.mean_affine_bias()
    torch.save(
        {"affine": affine.cpu(), "bias": bias.cpu()},
        Path(output_dir) / "mean_appearance.pt",
    )
```

Note on anti-aliasing (Task 7): the `anti_alias`/`full_stack` variants set
`pipe.antialiasing = True` via `_build_dataset_args`. The render step used later for holdout
eval and the final `test_poses.csv` render MUST set the same flag for that checkpoint — Task 9
carries `variant.use_anti_alias` through so its render calls match. There is no second rasterizer
submodule.

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_train_variant.py -v
```

Expected: `PASS` (3 passed) — these three tests only exercise `ALL_TRAINING_VARIANTS`, not
`run_training_variant`/`_build_dataset_args` (both GPU-dependent). The variant-config dataclass
logic is what's unit-tested here; the training loop is verified on Colab in Step 6.

- [ ] **Step 5: Commit**

```bash
git add src/training/train_variant.py tests/test_train_variant.py
git commit -m "Add variant-aware training loop (baseline/depth_reg/anti_alias/appearance_embed/full_stack)"
```

- [ ] **Step 6: Manual verification and correction on Colab (required, not optional)**

Even though the vendored API was checked (Scene args, GaussianModel ctor, render dict keys,
antialiasing flag, save path), the full loop still cannot be executed without a CUDA GPU, so run
it on Colab in this order and fix any remaining mismatch before trusting it:
1. `"baseline"` variant on the `chair` scene (cheapest) — confirm it produces
   `point_cloud/iteration_<N>/point_cloud.ply` and that Plan 1's `render_all` can load and render
   it without error.
2. Then `depth_reg`, `anti_alias`, `appearance_embed` each individually.
3. Only then `full_stack`.
Known things most likely to still need adjustment on real hardware: whether `Scene.save` writes
exactly to `point_cloud/iteration_<N>/point_cloud.ply` for this submodule pin (adjust `final_ply`
if it differs), and whether `viewpoint_cam.uid` is the right per-image index to key the
appearance embedding on (it must be a stable 0..N-1 index over `getTrainCameras()`; if `uid`
isn't that, enumerate the camera list and pass the enumeration index instead).

---

### Task 9: Extend orchestrator to run the full experiment matrix

**Files:**
- Create: `src/orchestrator/run_experiment_matrix.py`
- Test: `tests/test_run_experiment_matrix.py`

**Interfaces:**
- Consumes: `ALL_TRAINING_VARIANTS` (Task 8), `compute_prune_mask` (Task 2),
  `fits_within_vram_budget`/`estimate_vram_bytes`/`count_gaussians_in_ply` (Task 1),
  `select_best_candidate` (Task 6), everything Plan 1's `run_baseline_pipeline` already wires
  (holdout split, leak-free filtered training, metrics, packaging, validation).
- Produces: `run_experiment_matrix_pipeline(scenes: list[SceneConfig], train_variant_fn, render_fn, prune_fn, lpips_model, psnr_max: float, vram_budget_bytes: int, output_root: Path) -> ExperimentPipelineResult`
  — `train_variant_fn(scene, variant, output_dir) -> Path`, `prune_fn(checkpoint_path, bbox_min, bbox_max) -> Path`
  (returns a new pruned `.ply` path) are injected exactly like Plan 1's `train_fn`/`render_fn`,
  so this stays GPU-free and network-free to test. `ExperimentPipelineResult` extends Plan 1's
  `PipelineResult` with `chosen_config: dict[str, dict]` (scene name -> the winning candidate
  dict from `select_best_candidate`) and `all_candidates: dict[str, list[dict]]` (full score
  table per scene, for the reproducibility bundle in Task 10).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_run_experiment_matrix.py
from pathlib import Path

import numpy as np
import torch

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
        identical = torch.allclose(pred_tensor, gt_tensor)
        return torch.tensor(0.0 if identical else 1.0)


def test_run_experiment_matrix_trains_all_variants_and_picks_one_winner(tmp_path):
    scene = _chair_scene()
    variant_train_calls = []

    def fake_train_variant_fn(scene_arg, variant, output_dir):
        variant_train_calls.append(variant.name)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        ply = output_dir / "point_cloud.ply"
        ply.write_bytes(
            b"ply\nformat binary_little_endian 1.0\nelement vertex 10\n"
            b"property float x\nproperty float y\nproperty float z\nend_header\n" + b"\x00" * 120
        )
        return ply

    def fake_render_fn(checkpoint, params_list, output_dir):
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

    def fake_prune_fn(checkpoint_path, bbox_min, bbox_max):
        pruned_path = Path(checkpoint_path).with_name("point_cloud_pruned.ply")
        pruned_path.write_bytes(Path(checkpoint_path).read_bytes())
        return pruned_path

    result = run_experiment_matrix_pipeline(
        scenes=[scene],
        train_variant_fn=fake_train_variant_fn,
        render_fn=fake_render_fn,
        prune_fn=fake_prune_fn,
        lpips_model=_StubLpipsModel(),
        psnr_max=30.0,
        vram_budget_bytes=16 * 1024**3,
        output_root=tmp_path,
    )

    # all 5 named variants must actually have been trained
    assert set(variant_train_calls) == {
        "baseline", "depth_reg", "anti_alias", "appearance_embed", "full_stack",
    }

    assert "chair" in result.chosen_config
    assert "chair" in result.all_candidates
    # up to 5 variants x {raw, floater-cleaned} = 10 candidates scored
    assert len(result.all_candidates["chair"]) == 10
    assert result.submission_zip is not None
    assert result.submission_zip.exists()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_run_experiment_matrix.py -v
```

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
from src.common.pose_utils import camera_extrinsics_from_colmap, focal2fov, qvec2rotmat
from src.data_validation.validate_scene import validate_scene
from src.evaluation.compute_metrics import combine_score, compute_pair_metrics
from src.evaluation.make_holdout_split import select_holdout_images
from src.evaluation.select_best_config import select_best_candidate
from src.postprocess.vram_guard import estimate_vram_bytes, count_gaussians_in_ply
from src.rendering.render_from_csv import CameraParams, load_test_poses_csv
from src.submission.package_submission import package_submission
from src.submission.validate_submission import validate_submission
from src.training.holdout_scene import build_filtered_scene
from src.training.train_variant import ALL_TRAINING_VARIANTS


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
        fx, fy = camera.params[0], camera.params[1]
        r, t = camera_extrinsics_from_colmap(*img.qvec, *img.tvec)
        params.append(CameraParams(
            image_name=img.name, R=r, T=t,
            fov_x=focal2fov(fx, width), fov_y=focal2fov(fy, height),
            width=width, height=height,
        ))
    return params


def run_experiment_matrix_pipeline(
    scenes: list[SceneConfig], train_variant_fn, render_fn, prune_fn, lpips_model,
    psnr_max: float, vram_budget_bytes: int, output_root: Path,
) -> ExperimentPipelineResult:
    output_root = Path(output_root)
    result = ExperimentPipelineResult()
    scene_render_dirs = {}

    for scene in scenes:
        report = validate_scene(scene)
        if report.problems:
            result.skipped_scenes[scene.name] = report.problems
            continue

        scene_output = output_root / scene.name
        submission_dir = scene.effective_submission_dir

        sparse = load_sparse_scene(scene.sparse_dir)
        bbox_min, bbox_max = compute_scene_bbox(sparse.points3d, margin_ratio=0.1)
        file_backed_names = {p.name for p in scene.train_images_dir.iterdir() if p.is_file()}
        # Holdout candidates only from images with a real file — same
        # reasoning as Plan 1 Task 12: images.bin registers more cameras
        # than are distributed as files for every scene in this dataset,
        # and those extras have no local pixel data to score against.
        camera_centers = {
            img.name: -np.transpose(qvec2rotmat(np.array(img.qvec))) @ np.array(img.tvec)
            for img in sparse.images.values()
            if img.name in file_backed_names
        }
        holdout_names = set(select_holdout_images(camera_centers, holdout_ratio=0.125))
        filtered_scene = build_filtered_scene(
            scene, holdout_names, scene_output / "filtered_scene",
        )

        sample_image = next(scene.train_images_dir.iterdir())
        with Image.open(sample_image) as im:
            image_dims = im.size
        holdout_params = _camera_params_for_holdout(sparse, holdout_names, image_dims)

        candidates = []
        for variant in ALL_TRAINING_VARIANTS:
            eval_checkpoint = train_variant_fn(
                filtered_scene, variant, scene_output / f"eval_{variant.name}",
            )
            for use_floater_cleanup in (False, True):
                checkpoint = (
                    prune_fn(eval_checkpoint, bbox_min, bbox_max)
                    if use_floater_cleanup else eval_checkpoint
                )
                holdout_render_dir = (
                    scene_output / f"holdout_{variant.name}_{use_floater_cleanup}"
                )
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

                candidates.append({
                    "variant": variant.name,
                    "floater_cleanup": use_floater_cleanup,
                    "score": float(np.mean(scores)) if scores else 0.0,
                    "estimated_vram_bytes": estimate_vram_bytes(count_gaussians_in_ply(checkpoint)),
                    "checkpoint_path": str(checkpoint),
                })

        result.all_candidates[scene.name] = candidates
        winner = select_best_candidate(candidates, vram_budget_bytes)
        result.chosen_config[scene.name] = winner
        result.per_scene_scores[scene.name] = winner["score"]

        winning_variant = next(v for v in ALL_TRAINING_VARIANTS if v.name == winner["variant"])
        # Never pass the raw `scene` to train_variant_fn — same reasoning
        # as Plan 1 Task 12's Phase B: build_filtered_scene(scene, set(), ...)
        # strips registered-without-file images (which crash the real
        # loader's Image.open()) while keeping 100% of the real distributed
        # training data.
        full_training_scene = build_filtered_scene(
            scene, set(), scene_output / "full_scene",
        )
        final_checkpoint = train_variant_fn(
            full_training_scene, winning_variant, scene_output / "final_train",
        )
        if winner["floater_cleanup"]:
            final_checkpoint = prune_fn(final_checkpoint, bbox_min, bbox_max)

        test_render_dir = scene_output / "test_render"
        test_params_list = load_test_poses_csv(scene.test_poses_csv)
        render_fn(final_checkpoint, test_params_list, test_render_dir)
        scene_render_dirs[submission_dir] = test_render_dir

    submission_zip = output_root / "submission.zip"
    package_submission(scene_render_dirs, submission_zip)
    result.validation_problems = validate_submission(submission_zip, scenes)
    result.submission_zip = submission_zip
    return result
```

Create `src/orchestrator/__init__.py` if not already present from Plan 1 Task 12.

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_run_experiment_matrix.py -v
```

Expected: `PASS` (1 passed).

- [ ] **Step 5: Commit**

```bash
git add src/orchestrator/run_experiment_matrix.py tests/test_run_experiment_matrix.py
git commit -m "Add full experiment-matrix orchestrator with VRAM-aware auto-selection"
```

---

### Task 10: Reproducibility bundle

**Files:**
- Create: `src/submission/reproducibility_bundle.py`
- Test: `tests/test_reproducibility_bundle.py`

**Interfaces:**
- Produces: `write_reproducibility_bundle(scene_name: str, chosen_config: dict, all_candidates: list[dict], output_dir: Path) -> Path`
  — writes `output_dir/<scene_name>/chosen_config.yaml`, `output_dir/<scene_name>/
  all_candidates_scores.csv`, returns `output_dir/<scene_name>`. Per spec section 15 / exam
  mục 10.3, this is what gets handed to the organizers if requested — config + full score
  comparison table, so the choice of variant per scene is traceable and justified, not just
  asserted.

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
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_reproducibility_bundle.py -v
```

Expected: `FAIL` — `ModuleNotFoundError: No module named 'src.submission.reproducibility_bundle'`.

- [ ] **Step 3: Write `src/submission/reproducibility_bundle.py`**

```python
from __future__ import annotations

import csv
from pathlib import Path

import yaml


def write_reproducibility_bundle(
    scene_name: str, chosen_config: dict, all_candidates: list[dict], output_dir: Path,
) -> Path:
    bundle_dir = Path(output_dir) / scene_name
    bundle_dir.mkdir(parents=True, exist_ok=True)

    with open(bundle_dir / "chosen_config.yaml", "w") as f:
        yaml.safe_dump(chosen_config, f)

    fieldnames = ["variant", "floater_cleanup", "score", "estimated_vram_bytes", "checkpoint_path"]
    with open(bundle_dir / "all_candidates_scores.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for candidate in sorted(all_candidates, key=lambda c: -c["score"]):
            writer.writerow({k: candidate.get(k) for k in fieldnames})

    return bundle_dir
```

- [ ] **Step 4: Run test to verify it passes**

```bash
pytest tests/test_reproducibility_bundle.py -v
```

Expected: `PASS` (1 passed).

- [ ] **Step 5: Commit**

```bash
git add src/submission/reproducibility_bundle.py tests/test_reproducibility_bundle.py
git commit -m "Add reproducibility bundle writer (config + full candidate score table per scene)"
```

---

### Task 11: Visual QA sampler

**Files:**
- Create: `src/submission/visual_qa.py`
- Test: `tests/test_visual_qa.py`

**Interfaces:**
- Produces:
  - `sample_images_for_review(render_dir: Path, n: int = 5, seed: int = 0) -> list[Path]` —
    deterministic random sample of `n` rendered images from `render_dir` (or all of them if
    fewer than `n` exist), for a human to actually look at before packaging — per spec section
    13, automated holdout metrics don't guarantee the real extrapolated test poses look right.
  - `build_contact_sheet(image_paths: list[Path], output_html: Path) -> Path` — writes a
    minimal, dependency-free HTML page with each image's filename and a thumbnail `<img>` tag,
    so the sample can be opened in a browser and scanned quickly instead of opening files one
    by one.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_visual_qa.py
from pathlib import Path

from PIL import Image

from src.submission.visual_qa import build_contact_sheet, sample_images_for_review


def _make_images(dir_path: Path, n: int) -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        Image.new("RGB", (4, 4)).save(dir_path / f"{i:04d}.png")


def test_sample_images_for_review_returns_requested_count(tmp_path):
    _make_images(tmp_path, 20)
    sample = sample_images_for_review(tmp_path, n=5, seed=0)
    assert len(sample) == 5
    assert all(p.exists() for p in sample)


def test_sample_images_for_review_is_deterministic_for_same_seed(tmp_path):
    _make_images(tmp_path, 20)
    sample_a = sample_images_for_review(tmp_path, n=5, seed=42)
    sample_b = sample_images_for_review(tmp_path, n=5, seed=42)
    assert sample_a == sample_b


def test_sample_images_for_review_returns_all_when_fewer_than_n(tmp_path):
    _make_images(tmp_path, 3)
    sample = sample_images_for_review(tmp_path, n=5, seed=0)
    assert len(sample) == 3


def test_build_contact_sheet_writes_html_referencing_each_image(tmp_path):
    _make_images(tmp_path, 3)
    image_paths = sorted(tmp_path.glob("*.png"))
    output_html = tmp_path / "contact_sheet.html"

    result = build_contact_sheet(image_paths, output_html)

    assert result == output_html
    content = output_html.read_text()
    for path in image_paths:
        assert path.name in content
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_visual_qa.py -v
```

Expected: `FAIL` — `ModuleNotFoundError: No module named 'src.submission.visual_qa'`.

- [ ] **Step 3: Write `src/submission/visual_qa.py`**

```python
from __future__ import annotations

import random
from pathlib import Path


def sample_images_for_review(render_dir: Path, n: int = 5, seed: int = 0) -> list[Path]:
    render_dir = Path(render_dir)
    all_images = sorted(p for p in render_dir.iterdir() if p.is_file())
    rng = random.Random(seed)
    if len(all_images) <= n:
        return all_images
    return sorted(rng.sample(all_images, n))


def build_contact_sheet(image_paths: list[Path], output_html: Path) -> Path:
    output_html = Path(output_html)
    rows = "\n".join(
        f'<div><p>{p.name}</p><img src="{p.resolve()}" style="max-width:300px"></div>'
        for p in image_paths
    )
    output_html.write_text(
        f"<html><body><h1>Visual QA sample ({len(image_paths)} images)</h1>{rows}</body></html>"
    )
    return output_html
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_visual_qa.py -v
```

Expected: `PASS` (4 passed).

- [ ] **Step 5: Commit**

```bash
git add src/submission/visual_qa.py tests/test_visual_qa.py
git commit -m "Add visual QA sampler and contact sheet generator"
```

- [ ] **Step 6: Manual step (not automatable) — actually look at the sample**

After Task 9's orchestrator produces `test_render/` for each scene, run:

```python
from src.submission.visual_qa import build_contact_sheet, sample_images_for_review

for scene_name in ["HCM0421", "HCM0539", "HCM0540", "HCM0644", "HCM0674", "chair", "bonsai"]:
    sample = sample_images_for_review(f"outputs/{scene_name}/test_render", n=8)
    build_contact_sheet(sample, f"outputs/{scene_name}/qa_contact_sheet.html")
```

Open each `qa_contact_sheet.html` in a browser and look for black frames, obvious floaters, or
completely wrong geometry before running Plan 1's `package_submission`/`validate_submission`.
This step is a **read-only inspection** to decide whether to retrain/reconfigure — it must never
involve editing the images themselves (exam mục 10.4 forbids manual output editing).

---

## Self-Review Summary

- **Spec coverage:** Task 1 covers section 7 (VRAM guard). Task 2 covers the floater-cleanup
  half of section 6. Tasks 3-5 cover the depth-reg and appearance-embed halves of section 6's
  technique list (pure-math pieces). Task 7 covers the anti-alias half of section 6 (via the
  baseline's native `pipe.antialiasing` flag — see the revision note below). Task 8 wires all
  three training-time techniques into an actual loop. Task 6 covers section 9 (auto-select).
  Task 9 covers section 11 (orchestrator) extended to the full matrix. Task 10 covers section 15
  (reproducibility bundle). Task 11 covers section 13 (visual QA).
- **Revisions after inspecting the actual checked-out submodule (this revision):**
  (1) **Task 7 no longer vendors Mip-Splatting.** The checked-out
  `graphdeco-inria/gaussian-splatting` (commit 54c035f) already has native anti-aliasing:
  `PipelineParams.antialiasing`, `render()` reading `antialiasing=pipe.antialiasing`, and CUDA
  kernels branching on it. Anti-aliasing is now a single boolean, not a second rasterizer
  submodule — removing the extra submodule, the `setup_colab.sh --variant` rebuild dance, the
  `--force-reinstall` swap, and the import-ordering hazard the original Task 7/8 had.
  (2) **Task 8's `run_training_variant` was rewritten against the real API.** Verified in the
  checkout: `Scene(args, gaussians)` takes a ModelParams-like args object (not a `Path`) and
  reads `args.model_path/source_path/images/eval/...`; `_build_dataset_args` now constructs the
  real `ModelParams`/`PipelineParams`/`OptimizationParams` via `ArgumentParser` and overrides
  only the paths and the `antialiasing` flag, instead of hand-rolling a `Namespace` that would
  have been rejected. `GaussianModel(sh_degree)`, the `render()` return-dict keys (`render`,
  `viewspace_points`, `visibility_filter`, `radii`, `depth`), the densification calls, and
  `Scene.save` were all matched to the real source. The earlier import-ordering bug (importing
  `gaussian_renderer` before mutating `sys.path` for the anti-alias fork) is gone with the fork.
  (3) **Task 9's orchestrator no longer passes the raw `scene` to `train_variant_fn`.** Same
  root cause as Plan 1 Task 12: `images.bin` registers more cameras than are distributed as
  files for every scene, and the vendored loader crashes on `Image.open()` for any of them
  whenever `dataset.eval=False` (which this plan always uses). The "final training" call now
  goes through `build_filtered_scene(scene, set(), ...)` first, and holdout-candidate camera
  centers are restricted to file-backed images only — both fixes mirror Plan 1 Task 12 exactly.
- **Placeholder scan:** no TBD/TODO remain. Task 8's loop is still **GPU-dependent** and cannot
  be executed on this no-CUDA machine, so Step 6 keeps a required Colab verification pass — but
  the parts verifiable against the real source without a GPU (constructor signatures, param
  fields, render-dict keys, flag names, save path) now have been, so it is far more than a
  hopeful scaffold. Two residual real-hardware unknowns are named explicitly in Step 6 (exact
  `Scene.save` output path for this pin; whether `viewpoint_cam.uid` is the right 0..N-1
  appearance index) rather than hidden.
- **Type consistency:** `TrainingVariant` (Task 8) fields match its usage in Task 9. Candidate
  dicts produced in Task 9's `candidates.append(...)` use exactly the keys `select_best_candidate`
  (Task 6) and `write_reproducibility_bundle` (Task 10) expect (`variant`, `floater_cleanup`,
  `score`, `estimated_vram_bytes`, `checkpoint_path`). `ExperimentPipelineResult` reuses the same
  field names as Plan 1's `PipelineResult` (`per_scene_scores`, `skipped_scenes`,
  `validation_problems`, `submission_zip`) plus the two new ones, so callers familiar with Plan
  1's orchestrator aren't surprised.
