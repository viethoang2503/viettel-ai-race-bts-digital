# Training Pipeline Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Task 1–18 experiment pipeline geometrically correct, deterministic, resume-safe, VRAM fail-closed, and reproducible before the real seven-scene GPU run.

**Architecture:** Fix geometry at the COLMAP scene-copy boundary so every downstream depth consumer receives valid tracks. Add small, CPU-testable state/metadata helpers around the CUDA training loop, then enforce final-artifact safety in the orchestrator and renderer. Keep notebook cells thin and defensive.

**Tech Stack:** Python 3.14, PyTorch, NumPy, OpenCV, COLMAP binary format, pytest, Jupyter notebooks.

## Global Constraints

- Work directly on `master`; do not create a branch or worktree.
- Preserve unrelated edits in `.gitignore` and the 2026-07-23 implementation plan.
- Every production behavior change starts with a failing regression test.
- Do not run real training or require CUDA locally.
- `submission.zip` contains rendered images only; reproducibility artifacts remain separate.
- All seven real scenes must remain supported.

---

### Task 1: Preserve and undistort COLMAP point tracks

**Files:**
- Modify: `src/training/colmap_writer.py`
- Modify: `src/training/undistort_scene.py`
- Test: `tests/test_colmap_writer.py`
- Test: `tests/test_undistort_scene.py`
- Test: `tests/test_sparse_depth.py`

**Interfaces:**
- `write_images_binary(images: dict, path: Path)` writes each image's `xys` and
  `point3D_ids`, not zero tracks.
- `_undistort_observations(xys, camera_matrix, dist_coeffs) -> np.ndarray`
  uses `cv2.undistortPoints(..., P=camera_matrix)`.
- `undistort_scene` writes a transformed `images.bin`.

- [ ] **Step 1: Add failing writer round-trip test**

```python
def test_write_images_binary_preserves_point2d_tracks(tmp_path):
    image = _Image(
        id=7, qvec=np.array([1, 0, 0, 0]), tvec=np.zeros(3),
        camera_id=1, name="frame.jpg",
        xys=np.array([[10.5, 20.25], [30.0, 40.0]]),
        point3D_ids=np.array([11, -1]),
    )
    write_images_binary({7: image}, tmp_path / "images.bin")
    loaded = read_extrinsics_binary(str(tmp_path / "images.bin"))[7]
    np.testing.assert_allclose(loaded.xys, image.xys)
    np.testing.assert_array_equal(loaded.point3D_ids, image.point3D_ids)
```

- [ ] **Step 2: Run RED**

Run: `.venv/bin/python -m pytest tests/test_colmap_writer.py::test_write_images_binary_preserves_point2d_tracks -v`

Expected: track arrays are empty.

- [ ] **Step 3: Write track data in COLMAP binary layout**

After the image name, write `<Q` track count and one `<ddq` record per
`(xy, point3D_id)`. Validate equal lengths.

- [ ] **Step 4: Add failing undistortion tests**

Test `_undistort_observations` against a direct OpenCV calculation and assert
the scene output's reloaded `images.bin` contains transformed `xys` with
unchanged `point3D_ids`.

- [ ] **Step 5: Run RED**

Run: `.venv/bin/python -m pytest tests/test_undistort_scene.py -v`

Expected: output observations still equal distorted input observations.

- [ ] **Step 6: Implement observation transformation**

Build transformed image records with `image._replace(xys=...)`, reject
non-finite results, and call `write_images_binary` instead of copying the raw
`images.bin`. Supported PINHOLE passthrough remains unchanged.

- [ ] **Step 7: Run GREEN and real-data geometry check**

Run:

```bash
.venv/bin/python -m pytest tests/test_colmap_writer.py tests/test_undistort_scene.py tests/test_sparse_depth.py -v
```

Then use the real HCM sparse data to assert transformed observations equal
OpenCV output and differ from the raw distorted coordinates.

- [ ] **Step 8: Commit**

```bash
git add src/training/colmap_writer.py src/training/undistort_scene.py tests/test_colmap_writer.py tests/test_undistort_scene.py tests/test_sparse_depth.py
git commit -m "Preserve and undistort COLMAP tracks for depth supervision"
```

---

### Task 2: Bound search and make training runs deterministic

**Files:**
- Modify: `src/evaluation/screening.py`
- Modify: `src/training/train_variant.py`
- Test: `tests/test_screening.py`
- Test: `tests/test_train_variant.py`

**Interfaces:**
- `build_hyperparam_candidates` raises when `len(extra_overrides) > 4`.
- `_seed_everything(seed: int)` seeds Python, NumPy, Torch, and CUDA.
- `run_training_variant(..., seed: int = 0, checkpoint_interval: int = 5000)`.

- [ ] **Step 1: Add failing maximum-candidate test**

Assert five overrides raise `ValueError("at most 4")`.

- [ ] **Step 2: Add failing deterministic-seed tests**

Patch Torch CUDA availability/state calls, invoke `_seed_everything(123)`
twice, and assert Python/NumPy/Torch random samples repeat exactly. Inspect
`run_training_variant` defaults for `seed=0` and `checkpoint_interval=5000`.

- [ ] **Step 3: Run RED**

Run:

```bash
.venv/bin/python -m pytest tests/test_screening.py tests/test_train_variant.py -v
```

- [ ] **Step 4: Implement bounds and seed setup**

Validate candidate count before building candidates. Call `_seed_everything`
before constructing `GaussianModel` or `Scene`. Reject negative seed,
non-positive iterations, and non-positive checkpoint interval before GPU work.

- [ ] **Step 5: Run GREEN**

Run:

```bash
.venv/bin/python -m pytest tests/test_screening.py tests/test_train_variant.py -v
```

- [ ] **Step 6: Commit**

```bash
git add src/evaluation/screening.py src/training/train_variant.py tests/test_screening.py tests/test_train_variant.py
git commit -m "Bound candidate search and seed variant training"
```

---

### Task 3: Add atomic resume-safe variant checkpoints

**Files:**
- Modify: `src/training/train_variant.py`
- Test: `tests/test_train_variant.py`

**Interfaces:**
- `_checkpoint_schedule(iterations, interval) -> list[int]`.
- `_variant_run_fingerprint(scene, variant, iterations, overrides, seed) -> str`.
- `_atomic_torch_save(payload, path)` writes a sibling temporary file and
  `Path.replace()`s it.
- Variant checkpoint payload stores Gaussian capture, appearance states, RNG
  states, iteration, and fingerprint.

- [ ] **Step 1: Add failing helper tests**

Test schedules for exact/non-exact intervals, fingerprint changes for seed,
variant and overrides, and atomic save leaves only the final path.

- [ ] **Step 2: Run RED**

Run: `.venv/bin/python -m pytest tests/test_train_variant.py -v`

- [ ] **Step 3: Implement pure checkpoint helpers**

Reuse the existing content-aware `_scene_fingerprint` and hash a canonical
JSON payload containing variant flags, effective overrides, iterations, and
seed. Add RNG capture/restore helpers for Python, NumPy, Torch CPU, and CUDA.

- [ ] **Step 4: Add failing completed-run and resume-state seam tests**

Test pure helpers that validate a checkpoint fingerprint, locate the latest
matching `variant_chkpnt<N>.pth`, and recognize a completed PLY only when the
fingerprint manifest and required appearance artifact match.

- [ ] **Step 5: Implement loop integration**

Before fresh initialization, validate reusable final artifacts. Otherwise
initialize the scene, restore the latest matching checkpoint, restore
appearance/optimizer/RNG state, and continue at `iteration + 1`. Save atomic
state at every scheduled iteration. Write a completion manifest only after
mean appearance and final PLY both exist.

- [ ] **Step 6: Run GREEN**

Run:

```bash
.venv/bin/python -m pytest tests/test_train_variant.py tests/test_gs_train_fn.py -v
```

- [ ] **Step 7: Commit**

```bash
git add src/training/train_variant.py tests/test_train_variant.py
git commit -m "Add atomic resume checkpoints to variant training"
```

---

### Task 4: Enforce final-model VRAM and safe inference

**Files:**
- Modify: `src/rendering/gs_render_fn.py`
- Modify: `src/orchestrator/run_experiment_matrix.py`
- Test: `tests/test_gs_render_fn.py`
- Test: `tests/test_run_experiment_matrix.py`

**Interfaces:**
- Render config accepts `vram_budget_bytes`.
- Appearance path is required when supplied.
- `real_render_fn` renders under `torch.inference_mode()` and raises if peak
  CUDA allocation exceeds the supplied budget.
- Orchestrator checks `final_checkpoint` estimate after optional pruning and
  before render.

- [ ] **Step 1: Add failing renderer tests**

Test that a missing requested appearance file raises `FileNotFoundError`;
test the render closure executes with grad disabled through an injected fake
renderer; test VRAM budget parsing and overflow helper behavior.

- [ ] **Step 2: Add failing final-VRAM orchestration tests**

Make screening PLYs tiny and the fake final PLY claim an over-budget vertex
count. Assert the pipeline returns no submission and records a final VRAM
validation problem without calling final render.

- [ ] **Step 3: Run RED**

Run:

```bash
.venv/bin/python -m pytest tests/test_gs_render_fn.py tests/test_run_experiment_matrix.py -v
```

- [ ] **Step 4: Implement safe render path**

Strictly load requested appearance state, wrap the render body in
`torch.inference_mode()`, reset/read CUDA peak stats when available, and
raise a descriptive VRAM error after rendering. Keep default config backward
compatible.

- [ ] **Step 5: Implement final PLY preflight**

Estimate the actual final checkpoint after optional pruning. Store it in
chosen metadata. If it exceeds the budget, record a problem and skip test
render/package. Thread the budget into final render config and convert measured
overflow to fail-closed validation output.

- [ ] **Step 6: Run GREEN**

Run:

```bash
.venv/bin/python -m pytest tests/test_gs_render_fn.py tests/test_run_experiment_matrix.py tests/test_run_pipeline.py -v
```

- [ ] **Step 7: Commit**

```bash
git add src/rendering/gs_render_fn.py src/orchestrator/run_experiment_matrix.py tests/test_gs_render_fn.py tests/test_run_experiment_matrix.py
git commit -m "Enforce final checkpoint VRAM during inference"
```

---

### Task 5: Make reproducibility metadata truthful and complete

**Files:**
- Modify: `src/orchestrator/run_experiment_matrix.py`
- Modify: `src/submission/reproducibility_bundle.py`
- Test: `tests/test_run_experiment_matrix.py`
- Test: `tests/test_reproducibility_bundle.py`

**Interfaces:**
- Winner candidate itself receives `fallback_reason`.
- Chosen config distinguishes `selection_checkpoint_path` and
  `final_checkpoint_path`.
- `validate_reproducibility_bundle(root, expected_scene_names) -> list[str]`.
- `package_reproducibility_bundle(root, output_zip, expected_scene_names)`.

- [ ] **Step 1: Add failing fallback and final-metadata tests**

Assert actual pipeline output preserves fallback reason in
`all_candidates`, and chosen config includes seed, selection checkpoint,
final checkpoint, render config, and final estimated VRAM.

- [ ] **Step 2: Add failing bundle completeness tests**

Assert missing scene directory, missing YAML, and missing CSV are reported;
assert packaging refuses incomplete input and creates a ZIP with all expected
scene paths for complete input.

- [ ] **Step 3: Run RED**

Run:

```bash
.venv/bin/python -m pytest tests/test_reproducibility_bundle.py tests/test_run_experiment_matrix.py -v
```

- [ ] **Step 4: Implement metadata propagation**

Annotate the selected object in `candidates`; copy it into chosen config only
after final preflight metadata is known. Record `seed` on every candidate.

- [ ] **Step 5: Implement validated bundle packaging**

Add the validation and packaging functions to the existing submission module,
using `zipfile` and refusing any incomplete expected scene set.

- [ ] **Step 6: Run GREEN**

Run:

```bash
.venv/bin/python -m pytest tests/test_reproducibility_bundle.py tests/test_run_experiment_matrix.py -v
```

- [ ] **Step 7: Commit**

```bash
git add src/orchestrator/run_experiment_matrix.py src/submission/reproducibility_bundle.py tests/test_run_experiment_matrix.py tests/test_reproducibility_bundle.py
git commit -m "Record and validate final reproducibility metadata"
```

---

### Task 6: Harden both Colab notebooks

**Files:**
- Modify: `notebooks/colab_runner_hcm.ipynb`
- Modify: `notebooks/colab_runner_bonsai.ipynb`
- Create: `tests/test_notebook_hardening.py`

**Interfaces:**
- Bước 9 creates one LPIPS model outside scene loops.
- Result access is guarded when a scene is skipped or final validation fails.
- Diagnosis skips plotting when no matched holdout frames exist.
- Bước 11 calls `package_reproducibility_bundle` for all seven names.

- [ ] **Step 1: Add failing structural notebook tests**

Load notebook JSON and assert required calls/guards occur in the intended
cells, every non-magic code cell parses with `ast.parse`, and direct
`shutil.make_archive` is absent.

- [ ] **Step 2: Run RED**

Run: `.venv/bin/python -m pytest tests/test_notebook_hardening.py -v`

- [ ] **Step 3: Patch notebook cells**

Reuse `matrix_lpips_model`; add `if scene.name not in result.chosen_config`;
skip empty `worst`; use validated bundle packaging with all scene names; add
resume-safe explanatory text.

- [ ] **Step 4: Run GREEN**

Run:

```bash
.venv/bin/python -m pytest tests/test_notebook_hardening.py -v
```

- [ ] **Step 5: Run complete verification**

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m compileall -q src tests
git diff --check
```

- [ ] **Step 6: Commit**

```bash
git add notebooks/colab_runner_hcm.ipynb notebooks/colab_runner_bonsai.ipynb tests/test_notebook_hardening.py
git commit -m "Harden Colab experiment and bundle workflow"
```

---

## Manual Colab Gate Before Full Training

Run these in order after pushing:

1. `chair`, baseline, 200 iterations; interrupt after a checkpoint and verify
   resume continues instead of restarting.
2. `chair`, depth and appearance variants, 200 iterations.
3. One HCM depth variant, 200 iterations; log non-empty sparse target count.
4. Prune and render the short PLY with antialiasing/appearance enabled.
5. Confirm final estimated and measured VRAM metadata are present.
6. Only after all five checks pass, run the full seven-scene matrix.
