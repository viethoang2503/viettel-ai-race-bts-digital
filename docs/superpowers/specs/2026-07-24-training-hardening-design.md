# Training Pipeline Hardening Design

## Goal

Make the Task 1–18 experiment pipeline safe to use for the real seven-scene
training run: geometrically correct after HCM undistortion, deterministic,
resume-safe, VRAM fail-closed, and fully reproducible.

## Scope

This hardening covers every finding from the 2026-07-24 review:

- undistorted sparse-depth observations;
- deterministic candidate comparisons;
- resumable variant training;
- final-checkpoint VRAM enforcement and measured render usage;
- truthful reproducibility metadata and complete bundle packaging;
- bounded extra search and defensive notebook behavior;
- inference-mode rendering and strict appearance artifact handling.

It does not run a real CUDA training job. GPU verification remains a separate
Colab gate after all CPU tests pass.

## Architecture

### Geometry

`undistort_scene` owns the coordinate-system conversion. For each
`SIMPLE_RADIAL` camera it undistorts both image pixels and every registered
COLMAP observation using the same original camera matrix and distortion
coefficients. It writes a new `images.bin` rather than copying the old one.
All downstream consumers, including sparse-depth supervision, therefore see
coordinates that match the undistorted images without special cases.

### Deterministic, resumable variant training

Every training invocation receives an explicit integer seed. The seed is
applied to Python, NumPy, Torch CPU, and all CUDA generators before scene/model
construction. Candidate metadata and reproducibility output record it.

Variant training writes atomic checkpoints at a configurable interval
(5,000 iterations by default). A checkpoint contains:

- Gaussian capture and current iteration;
- appearance module and optimizer state when enabled;
- Python, NumPy, Torch CPU, and CUDA RNG states;
- a fingerprint of scene content, variant flags, effective hyperparameters,
  target iterations, and seed.

Only a matching fingerprint may resume. A completed matching run returns its
existing PLY (and mean appearance artifact when required) without retraining.
Mismatched or incomplete artifacts are not silently reused.

### Candidate selection and VRAM safety

Screening still estimates candidate VRAM for ranking. After the winner is
trained on the full dataset, the pipeline recalculates its estimate on the
actual final PLY. A final checkpoint above the configured budget is rejected
before submission rendering.

Rendering runs under `torch.inference_mode()`. The final render configuration
includes the VRAM budget; the renderer measures CUDA peak allocation and
raises when the real render exceeds it. Missing appearance artifacts are an
error whenever an appearance path was requested.

The result records both the selection checkpoint and the actual final
checkpoint, along with estimated and measured final VRAM. Validation remains
fail-closed.

### Reproducibility and notebooks

The selected candidate in `all_candidates` receives any VRAM fallback reason;
`chosen_config.yaml` is enriched after final training instead of pointing only
at the filtered holdout checkpoint. Bundles include seed, render settings,
final checkpoint, and final VRAM fields.

The final bundle packager verifies that all expected seven scene directories
and their required files exist before creating the ZIP. Notebook cells reuse
one LPIPS model, guard skipped/failed scenes, handle empty diagnosis output,
and reject more than four extra candidates per scene.

## Error handling

- Invalid or non-finite undistorted observations raise a clear error.
- Unknown hyperparameters and invalid candidate counts fail before GPU work.
- Resume metadata mismatch prevents unsafe checkpoint reuse.
- Missing appearance artifacts fail rather than silently disabling a variant.
- Final estimated or measured VRAM overflow prevents a valid submission result.
- Missing scene reproducibility artifacts prevent bundle creation.

## Testing

CPU regression tests cover:

- `SIMPLE_RADIAL` observation transformation and rewritten `images.bin`;
- deterministic seed setup, checkpoint schedule/fingerprint, atomic save,
  completed-run reuse, and resume-state restoration through injectable helpers;
- final PLY VRAM recheck and fail-closed orchestration;
- inference-mode rendering and strict appearance loading without requiring
  CUDA through small seams/mocks;
- winner metadata propagation and fallback CSV preservation;
- complete seven-scene bundle validation;
- maximum four extra candidates;
- notebook JSON, Python syntax, and required guard statements.

The final verification command is:

```bash
.venv/bin/python -m pytest -q
```

Real Colab verification then runs one short `chair` baseline/depth/appearance
smoke test, one HCM depth smoke test, an interrupted/resumed run, floater
pruning, and a final render with measured VRAM before starting the full matrix.
