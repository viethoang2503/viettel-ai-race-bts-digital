# Score Push (Plan 2 Execution) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the diagnosis step, screening/final-iteration split, bounded per-scene
hyperparameter search, and final-assembly wiring defined in
`docs/superpowers/specs/2026-07-23-score-push-design.md`, so the 7-scene baseline
(Score ≈0.622) can be pushed toward ≈0.80 within the remaining 7-day window.

**Architecture:** Pure-Python/pure-math pieces (diagnosis ranking, tie-break decision,
hyperparameter candidate list construction) get real `pytest` unit tests on the local no-GPU
machine, exactly like Plan 1 and the advanced-techniques plan. The orchestrator
(`run_experiment_matrix_pipeline`) and the training-loop modification are GPU-dependent — written
completely against the vendored API, but flagged for manual verification/adjustment on Colab,
like every GPU-touching task in this repo so far.

**Tech Stack:** Same as Plan 1/2 (Python 3.10+, PyTorch, `lpips`, `scikit-image`, `numpy`,
`pyyaml`, `pytest`), the vendored `graphdeco-inria/gaussian-splatting` submodule.

## Global Constraints

- **Prerequisite — must be done first, unmodified:** Tasks 1-7 and Task 10 of
  `docs/superpowers/plans/2026-07-18-advanced-techniques.md` (VRAM guard, floater/background
  prune mask, sparse depth targets, appearance embedding, depth regularization loss, auto-select
  best config, anti-aliasing flag confirmation) are complete, self-contained, and correct as
  written there — implement them exactly as specified (use `subagent-driven-development` or
  `executing-plans` directly on that document for these tasks) before starting Task 1 below. Task
  11 of that document (visual QA) is **already implemented** at `src/submission/visual_qa.py` —
  skip it.
- **Deliberately NOT implemented from that document: Task 8 as originally written, and Task 9.**
  This plan's Task 5 implements Task 8's `src/training/train_variant.py` with one added
  parameter (`hyperparam_overrides`) baked in from the start — do not implement Task 8 verbatim
  first and then modify it. This plan's Task 6 fully replaces Task 9's
  `run_experiment_matrix_pipeline` — do not implement that document's Task 9 at all.
- Deadline: 30/07/2026. GPU: Google Colab Pro, L4, one session at a time (no parallel-session
  dependency). GPU cost is not a constraint; calendar time is.
- Tests: run with `.venv/bin/python -m pytest -q` (system `python3` lacks `pytest`).
- Submission format constraint (exam spec 1.5): `submission.zip` must contain **only** per-scene
  rendered images — reproducibility artifacts (config YAML, score CSVs) must be written to a
  **separate** zip/folder, never merged into `submission.zip`.

---

### Task 1: Per-image holdout metrics for diagnosis

**Files:**
- Create: `src/diagnostics/__init__.py` (empty)
- Create: `src/diagnostics/scene_diagnosis.py`
- Test: `tests/test_scene_diagnosis.py`

**Interfaces:**
- Consumes: `compute_pair_metrics(pred, gt, lpips_model) -> dict` and
  `combine_score(lpips_val, ssim_val, psnr_val, psnr_max) -> float` from
  `src/evaluation/compute_metrics.py` (already implemented, Plan 1).
- Produces: `compute_per_image_metrics(pred_dir: Path, gt_dir: Path, lpips_model, psnr_max: float) -> dict[str, dict[str, float]]`
  — maps `image_name -> {"lpips": ..., "ssim": ..., "psnr": ..., "score": ...}` for every file in
  `pred_dir` that has a same-named file in `gt_dir`; files with no match are skipped, not errored
  (a scene's `holdout_render/` may legitimately not cover every ground-truth file).

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

### Task 2: Rank holdout images worst-first

**Files:**
- Modify: `src/diagnostics/scene_diagnosis.py`
- Test: `tests/test_scene_diagnosis.py`

**Interfaces:**
- Consumes: the `dict[str, dict[str, float]]` shape produced by Task 1's
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

### Task 3: Screening tie-break and bounded hyperparameter candidates

**Files:**
- Create: `src/evaluation/screening.py`
- Test: `tests/test_screening.py`

**Interfaces:**
- Produces:
  - `needs_tiebreak_rerun(candidates: list[dict], threshold: float = 0.01) -> list[str]` — given
    dicts with `"variant"` and `"score"` keys, returns the `variant` names (excluding the single
    highest scorer) whose score is within `threshold` of the leader — these must be re-verified
    at full iterations before a winner is trusted (spec section 5).
  - `build_hyperparam_candidates(base_overrides: dict[str, object], extra_overrides: list[dict[str, object]], label_prefix: str) -> list[dict[str, object]]`
    — for each entry in `extra_overrides`, merges it on top of `base_overrides` and tags the
    result with a unique `candidate_name` (`f"{label_prefix}_{i}"`). Deliberately NOT a
    combinatorial grid — `extra_overrides` is a hand-picked, bounded list (spec: max 4 per
    scene), not every parameter crossed with every other.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_screening.py
from src.evaluation.screening import build_hyperparam_candidates, needs_tiebreak_rerun


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
Expected: `PASS` (5 passed).

- [ ] **Step 5: Commit**

```bash
git add src/evaluation/screening.py tests/test_screening.py
git commit -m "Add screening tie-break decision and bounded hyperparameter candidate builder"
```

---

### Task 4: Diagnosis notebook cell (Giai đoạn 0)

**Files:**
- Modify: `notebooks/colab_runner_hcm.ipynb`

**Interfaces:**
- Consumes: `compute_per_image_metrics`, `rank_holdout_by_score` (Task 1/2); the already-existing
  `OUTPUT_ROOT/<scene>/holdout_render/` directories and `undistort_scene` (Plan 1) — both already
  on Drive from the finished Plan 1 baseline run, so this costs no GPU time, only CPU + human
  eyeballing.

No GPU-dependent training code in this task, but it is Colab-only (needs the already-mounted
Drive + `load_lpips_model()` from the finished Plan 1 run) so there is no local `pytest` step —
verify manually on Colab per Step 2.

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
        Image.open(pred_dir / name).convert("RGB")
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
frames have in common — this human judgment feeds Task 7's `extra_overrides` choice for Giai
đoạn 2.

- [ ] **Step 3: Commit**

```bash
git add notebooks/colab_runner_hcm.ipynb
git commit -m "Add holdout diagnosis cell (Buoc 8) for pre-training-investment triage"
```

---

### Task 5: Add hyperparameter override support to the variant training loop

**Files:**
- Modify: `src/training/train_variant.py` (created by the prerequisite Task 8 of
  `docs/superpowers/plans/2026-07-18-advanced-techniques.md` — see Global Constraints)
- Modify: `tests/test_train_variant.py`

**Interfaces:**
- Modifies: `run_training_variant(scene, variant, output_dir, iterations) -> Path` gains a 5th
  parameter, `hyperparam_overrides: dict[str, object] | None = None`. Unknown override keys
  raise `ValueError` immediately rather than silently doing nothing — a typo here would otherwise
  waste a full Colab training run before anyone notices the override never applied.

- [ ] **Step 1: Write the failing test (append to `tests/test_train_variant.py`)**

```python
def test_run_training_variant_signature_accepts_hyperparam_overrides():
    import inspect

    from src.training.train_variant import run_training_variant

    params = inspect.signature(run_training_variant).parameters
    assert "hyperparam_overrides" in params
    assert params["hyperparam_overrides"].default is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_train_variant.py -v -k hyperparam_overrides`
Expected: `FAIL` — `AssertionError` (parameter not present yet).

- [ ] **Step 3: Modify `run_training_variant` in `src/training/train_variant.py`**

Change the function signature from:

```python
def run_training_variant(
    scene: SceneConfig, variant: TrainingVariant, output_dir: Path, iterations: int,
) -> Path:
```

to:

```python
def run_training_variant(
    scene: SceneConfig, variant: TrainingVariant, output_dir: Path, iterations: int,
    hyperparam_overrides: dict[str, object] | None = None,
) -> Path:
```

And immediately after the existing line `opt.iterations = iterations` (inside the function body,
before `gaussians = GaussianModel(dataset.sh_degree)`), add:

```python
    for key, value in (hyperparam_overrides or {}).items():
        if not hasattr(opt, key):
            raise ValueError(f"unknown training hyperparameter override: {key!r}")
        setattr(opt, key, value)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_train_variant.py -v`
Expected: `PASS` (4 passed — the 3 original `ALL_TRAINING_VARIANTS` tests plus the new signature
test).

- [ ] **Step 5: Commit**

```bash
git add src/training/train_variant.py tests/test_train_variant.py
git commit -m "Add hyperparameter override support to the variant training loop"
```

- [ ] **Step 6: Manual verification on Colab (required, not optional)**

Before trusting this on a real Giai đoạn 2 run: call `run_training_variant` for the `baseline`
variant on `chair` with `hyperparam_overrides={"densify_grad_threshold": 0.0005}` for a short
`iterations=200` smoke run, and print `opt.densify_grad_threshold` right after the override loop
to confirm it actually changed from the vendored default (`0.0002`). Also confirm
`hyperparam_overrides={"not_a_real_field": 1}` raises `ValueError` immediately rather than
starting training.

---

### Task 6: Corrected experiment-matrix orchestrator

**Files:**
- Create: `src/orchestrator/run_experiment_matrix.py`
- Test: `tests/test_run_experiment_matrix.py`

**Interfaces:**
- Consumes: `ALL_TRAINING_VARIANTS`, `run_training_variant` (Task 5's corrected version),
  `select_best_candidate` (prerequisite Task 6), `estimate_vram_bytes`/`count_gaussians_in_ply`
  (prerequisite Task 1), `needs_tiebreak_rerun` (Task 3), everything Plan 1's
  `run_baseline_pipeline` already wires (holdout split, `undistort_scene`,
  `build_filtered_scene`, `compute_pair_metrics`, `package_submission`, `validate_submission`).
- Produces: `run_experiment_matrix_pipeline(scenes, screening_train_fn, final_train_fn, render_fn, prune_fn, lpips_model, psnr_max, vram_budget_bytes, output_root, extra_candidates_by_scene=None, tiebreak_threshold=0.01) -> ExperimentPipelineResult`
  — `screening_train_fn(scene, variant, output_dir) -> Path` and
  `final_train_fn(scene, variant, output_dir, hyperparam_overrides=None) -> Path` are injected
  exactly like Plan 1's `train_fn`/`render_fn`, so this stays GPU-free and network-free to test.
  `ExperimentPipelineResult` extends Plan 1's `PipelineResult` shape with
  `chosen_config: dict[str, dict]` and `all_candidates: dict[str, list[dict]]`.

**This supersedes Task 9 of `docs/superpowers/plans/2026-07-18-advanced-techniques.md` — do not
implement that version.** Three differences, all needed for the 7-day plan (spec section 3-6):
1. `screening_train_fn`/`final_train_fn` split (reduced-iteration screening vs. full-iteration
   final retrain — spec section 5).
2. Scenes are undistorted (`undistort_scene`) before any training — the original Task 9 omitted
   this and would crash on the 5 real BTS scenes (`SIMPLE_RADIAL` cameras raise inside the
   vendored loader).
3. `extra_candidates_by_scene` and `tiebreak_threshold`/`needs_tiebreak_rerun` wiring for Giai
   đoạn 2's bounded hyperparameter search (spec section 6) and the screening tie-break rule
   (spec section 5).

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

    def fake_render_fn(checkpoint, params_list, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        # Deterministic per-checkpoint pixel fill so different checkpoints
        # score differently against the real chair GT images (avoids every
        # candidate tying exactly, which would make the tie-break path
        # untestable).
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

    # All 5 named variants were screened via the reduced-iteration path.
    assert set(screening_calls) == {
        "baseline", "depth_reg", "anti_alias", "appearance_embed", "full_stack",
    }
    # 10 variant x floater candidates + 2 extra chair candidates.
    assert len(result.all_candidates["chair"]) == 12
    assert any(c.get("candidate_name") == "chair_extra_0" for c in result.all_candidates["chair"])

    # The winner was retrained at full iterations via final_train_fn, into a
    # "final_train" output dir -- never shipped straight from screening.
    assert any("final_train" in call[1] for call in final_calls)

    assert "chair" in result.chosen_config
    assert result.submission_zip is not None
    assert result.submission_zip.exists()
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
from src.common.pose_utils import camera_extrinsics_from_colmap, focal2fov, qvec2rotmat
from src.data_validation.validate_scene import validate_scene
from src.evaluation.compute_metrics import combine_score, compute_pair_metrics
from src.evaluation.make_holdout_split import select_holdout_images
from src.evaluation.screening import needs_tiebreak_rerun
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
        fx, fy = camera.params[0], camera.params[1]
        r, t = camera_extrinsics_from_colmap(*img.qvec, *img.tvec)
        params.append(CameraParams(
            image_name=img.name, R=r, T=t,
            fov_x=focal2fov(fx, width), fov_y=focal2fov(fy, height),
            width=width, height=height,
        ))
    return params


def _score_checkpoint(checkpoint, holdout_params, render_fn, render_dir, gt_dir, lpips_model, psnr_max):
    rendered_paths = render_fn(checkpoint, holdout_params, render_dir)
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
        for variant in ALL_TRAINING_VARIANTS:
            eval_checkpoint = screening_train_fn(
                filtered_scene, variant, scene_output / f"eval_{variant.name}",
            )
            for use_floater_cleanup in (False, True):
                checkpoint = (
                    prune_fn(eval_checkpoint, bbox_min, bbox_max)
                    if use_floater_cleanup else eval_checkpoint
                )
                score = _score_checkpoint(
                    checkpoint, holdout_params, render_fn,
                    scene_output / f"holdout_{variant.name}_{use_floater_cleanup}",
                    working_scene.train_images_dir, lpips_model, psnr_max,
                )
                candidates.append({
                    "variant": variant.name, "floater_cleanup": use_floater_cleanup,
                    "score": score,
                    "estimated_vram_bytes": estimate_vram_bytes(count_gaussians_in_ply(checkpoint)),
                    "checkpoint_path": str(checkpoint),
                })

        # Tie-break: re-run any variant whose BEST (floater on or off)
        # screening score is within tiebreak_threshold of the leader, at
        # full iterations, before trusting the reduced-iteration ranking.
        best_per_variant = {}
        for c in candidates:
            if c["variant"] not in best_per_variant or c["score"] > best_per_variant[c["variant"]]:
                best_per_variant[c["variant"]] = c["score"]
        variants_to_rerun = needs_tiebreak_rerun(
            [{"variant": v, "score": s} for v, s in best_per_variant.items()],
            threshold=tiebreak_threshold,
        )
        for variant_name in variants_to_rerun:
            variant = next(v for v in ALL_TRAINING_VARIANTS if v.name == variant_name)
            full_checkpoint = final_train_fn(
                filtered_scene, variant, scene_output / f"tiebreak_{variant.name}",
            )
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
                    working_scene.train_images_dir, lpips_model, psnr_max,
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
            checkpoint = final_train_fn(
                filtered_scene, variant, scene_output / f"extra_{extra['candidate_name']}",
                hyperparam_overrides=overrides,
            )
            use_floater_cleanup = bool(extra.get("floater_cleanup", False))
            if use_floater_cleanup:
                checkpoint = prune_fn(checkpoint, bbox_min, bbox_max)
            score = _score_checkpoint(
                checkpoint, holdout_params, render_fn,
                scene_output / f"extra_holdout_{extra['candidate_name']}",
                working_scene.train_images_dir, lpips_model, psnr_max,
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
        final_checkpoint = final_train_fn(
            full_training_scene, winning_variant, scene_output / "final_train",
            hyperparam_overrides=winner.get("hyperparam_overrides"),
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

Create `src/orchestrator/__init__.py` if not already present from Plan 1.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_run_experiment_matrix.py -v`
Expected: `PASS` (1 passed).

- [ ] **Step 5: Commit**

```bash
git add src/orchestrator/run_experiment_matrix.py tests/test_run_experiment_matrix.py
git commit -m "Add corrected experiment-matrix orchestrator (screening/final split, undistort fix, bounded search)"
```

- [ ] **Step 6: Manual verification on Colab (required, not optional)**

Run `run_experiment_matrix_pipeline` for `chair` alone first (cheapest scene) with
`screening_train_fn = functools.partial(run_training_variant, iterations=15000)` and
`final_train_fn = functools.partial(run_training_variant, iterations=30000)`, no extra
candidates. Confirm: all 5 variants train, `chosen_config["chair"]` is populated, `test_render/`
has real images, `submission_zip` validates clean. Only then move to Task 7's full notebook
wiring for all 7 scenes.

---

### Task 7: Checkpoint-level floater prune wrapper

**Files:**
- Modify: `src/postprocess/prune_floaters.py` (created by prerequisite Task 2 of
  `docs/superpowers/plans/2026-07-18-advanced-techniques.md` — that task only produces
  `compute_prune_mask`, a boolean array function; it does not wrap file I/O)

**Interfaces:**
- Consumes: `compute_prune_mask(xyz, opacity, scales, bbox_min, bbox_max) -> np.ndarray`
  (prerequisite Task 2).
- Produces: `prune_checkpoint(checkpoint_path: Path, bbox_min: np.ndarray, bbox_max: np.ndarray, sh_degree: int = 3) -> Path`
  — loads the `.ply` at `checkpoint_path` via the vendored `GaussianModel.load_ply`, prunes
  floaters via `compute_prune_mask`, saves the result as `<stem>_pruned.ply` next to the
  original, and returns that path. This is the exact `prune_fn` callable Task 6's
  `run_experiment_matrix_pipeline` (and Task 8's notebook cells) inject.

This wraps the vendored `GaussianModel.load_ply`/`.prune_points`/`.save_ply` (verified present in
the checked-out `third_party/gaussian-splatting/scene/gaussian_model.py:239,263,349`), which
hardcode `device="cuda"` internally — same GPU-dependent category as `run_training_variant`, so
there is no local `pytest` step; `compute_prune_mask` itself (prerequisite Task 2) already has
its own local tests.

- [ ] **Step 1: Append to `src/postprocess/prune_floaters.py`**

```python
def prune_checkpoint(checkpoint_path, bbox_min, bbox_max, sh_degree: int = 3):
    from pathlib import Path

    import torch

    from scene import GaussianModel

    checkpoint_path = Path(checkpoint_path)
    gaussians = GaussianModel(sh_degree)
    gaussians.load_ply(str(checkpoint_path))

    xyz = gaussians._xyz.detach().cpu().numpy()
    opacity = gaussians.get_opacity.detach().cpu().numpy().squeeze(-1)
    scales = gaussians.get_scaling.detach().cpu().numpy()

    # compute_prune_mask returns True=keep; GaussianModel.prune_points
    # expects True=prune (it internally does valid_points_mask = ~mask) --
    # invert here, once, so callers of prune_checkpoint never need to know
    # about this polarity difference.
    keep_mask = compute_prune_mask(xyz, opacity, scales, bbox_min, bbox_max)
    gaussians.prune_points(torch.from_numpy(~keep_mask).to(gaussians._xyz.device))

    output_path = checkpoint_path.with_name(f"{checkpoint_path.stem}_pruned.ply")
    gaussians.save_ply(str(output_path))
    return output_path
```

- [ ] **Step 2: Manual verification on Colab (required, not optional)**

Run `prune_checkpoint` on a real `final_train` checkpoint from Task 6 Step 6's verification run.
Confirm `count_gaussians_in_ply(output_path) < count_gaussians_in_ply(checkpoint_path)` (the real
`chair`/HCM scenes have sky/background floaters per the diagnosis in Task 4, so some pruning is
expected), and that `real_render_fn` loads and renders the pruned `.ply` without error, same as
any other checkpoint.

- [ ] **Step 3: Commit**

```bash
git add src/postprocess/prune_floaters.py
git commit -m "Add checkpoint-level floater prune wrapper (load/prune/save .ply)"
```

---

### Task 8: Wire Giai đoạn 1+2 into the Colab notebooks

**Files:**
- Modify: `notebooks/colab_runner_hcm.ipynb` (5 HCM scenes — Giai đoạn 1 only, no extras)
- Modify: `notebooks/colab_runner_bonsai.ipynb` (`bonsai` + `chair` — Giai đoạn 1 + Giai đoạn 2)

**Interfaces:**
- Consumes: `run_experiment_matrix_pipeline` (Task 6), `build_hyperparam_candidates` (Task 3),
  `write_reproducibility_bundle` (prerequisite Task 10).

Each scene is run through its own call to `run_experiment_matrix_pipeline` (`scenes=[scene]`),
not one call for all scenes at once — a Colab disconnect partway through then loses at most one
scene's progress, not all of them, matching the resumability property Plan 1's notebooks already
rely on.

- [ ] **Step 1: Add a new Bước 9 cell to `notebooks/colab_runner_hcm.ipynb` (after the new Bước 8
  diagnosis cell from Task 4)**

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

from src.evaluation.screening import needs_tiebreak_rerun
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

### Task 9: Final merge and reproducibility bundle close-out (Giai đoạn 3)

**Files:**
- Modify: `notebooks/colab_runner_hcm.ipynb`

**Interfaces:**
- Consumes: the existing Bước 8 merge cell (already present, see
  `docs/superpowers/specs/2026-07-23-score-push-design.md` section 3 — it already scans
  `OUTPUT_ROOT/<scene>/test_render/` for all 7 scenes, which is exactly where Task 6's
  `run_experiment_matrix_pipeline` writes its final render). No change needed to that cell logic.
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

After all 7 scenes have gone through Task 8's Bước 9 in their respective notebooks, run the
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

- **Spec coverage:** Giai đoạn 0 (diagnosis) → Task 1/2/4. Giai đoạn 1 (5-variant matrix,
  reduced-iteration screening, tie-break) → Task 3/5/6/7/8. Giai đoạn 2 (bounded hyperparameter
  search for bonsai/chair) → Task 3/6/7/8. Giai đoạn 3 (final retrain, blind render,
  reproducibility bundle, merge) → Task 6 (retrain+render+package are inside the orchestrator
  itself) + Task 9 (reproducibility close-out). Out-of-scope items from the spec (full grid
  across all 7 scenes, A100, parallel sessions) are not implemented anywhere in this plan,
  consistent with the spec.
- **Placeholder scan:** no TBD/TODO; every code step has complete, runnable code; no "similar to
  Task N" shortcuts. Task 7 was added during self-review — the original draft referenced a
  `prune_floaters`/`prune_fn` callable operating directly on checkpoint paths, but the
  prerequisite plan's Task 2 only produces `compute_prune_mask` (a boolean-array function over
  already-loaded Gaussian attributes, not file I/O) — Task 7 closes that gap with a real
  `prune_checkpoint(checkpoint_path, bbox_min, bbox_max) -> Path` wrapper, verified against the
  vendored `GaussianModel.load_ply`/`prune_points`/`save_ply` methods in the actual checked-out
  submodule (`third_party/gaussian-splatting/scene/gaussian_model.py:239,263,349`), including the
  keep/prune mask polarity inversion `prune_points` requires.
- **Type consistency:** `run_training_variant`'s new `hyperparam_overrides` parameter (Task 5) is
  used consistently with the same name/shape (`dict[str, object] | None`) in Task 6's
  `final_train_fn` calls and Task 8's notebook cells. `ExperimentPipelineResult.chosen_config`
  entries carry an optional `hyperparam_overrides` key (only present for Giai đoạn 2 candidates)
  — Task 6's final-retrain call uses `.get("hyperparam_overrides")`, which is `None` for the
  10 base variant/floater candidates and the dict for extras, matching Task 5's default.
  `prune_fn` is now consistently `prune_checkpoint` (Task 7) everywhere it's referenced: Task 6's
  test/interface, and both notebook cells in Task 8.
