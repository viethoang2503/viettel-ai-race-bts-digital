from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from pathlib import Path

from src.common.config import SceneConfig
from src.training.train_wrapper import build_train_argv, checkpoint_iteration, find_latest_checkpoint

GS_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "gaussian-splatting"
_FINGERPRINT_FILENAME = ".gs_train_fn_fingerprint"

# Bump this whenever build_train_argv's fixed flags change in a way that
# would produce a materially different checkpoint from the same scene data
# (e.g. the --resolution 1 fix below) — the fingerprint otherwise only
# tracks scene DATA, so a config-only change would be invisible to it and
# a stale, wrongly-configured checkpoint would keep getting silently
# reused forever. Bumped once already: v2 forces native --resolution
# (v1 let the vendored loader auto-downscale anything wider than 1600px,
# producing checkpoints trained at a different resolution than what this
# pipeline renders/scores at — reproduced on a real Colab run as blurry
# renders and PSNR ~14.8 on bonsai).
_TRAIN_CONFIG_VERSION = "v2"


def _scene_fingerprint(scene: SceneConfig, iterations: int) -> str:
    """Hash of everything that determines what train.py would actually
    train on: image file CONTENT (not just names — a same-named re-upload
    with different pixels must not be mistaken for the same scene), the
    sparse reconstruction's content (cameras.bin/images.bin/points3D.bin —
    poses/intrinsics/points can change independently of the images), and
    _TRAIN_CONFIG_VERSION (so a fixed-flag change invalidates old
    checkpoints even when the scene data itself hasn't changed at all).
    """
    digest = hashlib.sha256()
    digest.update(f"{_TRAIN_CONFIG_VERSION}|{Path(scene.train_images_dir).resolve()}|{iterations}".encode())

    image_paths = sorted(
        (p for p in Path(scene.train_images_dir).iterdir() if p.is_file()),
        key=lambda p: p.name,
    )
    for path in image_paths:
        digest.update(path.name.encode())
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                digest.update(chunk)

    sparse_dir = Path(scene.sparse_dir)
    for sparse_filename in ("cameras.bin", "images.bin", "points3D.bin"):
        sparse_path = sparse_dir / sparse_filename
        digest.update(sparse_filename.encode())
        if sparse_path.is_file():
            with open(sparse_path, "rb") as f:
                digest.update(f.read())

    return digest.hexdigest()


def real_train_fn(scene: SceneConfig, output_dir: Path, iterations: int = 30000) -> Path:
    """Resume-safe wrapper around the vendored train.py, invoked via
    subprocess so a Colab GPU is required only for the actual training —
    the resume/skip decision and argv construction are pure Python, tested
    without a GPU by mocking subprocess.run (see tests/test_gs_train_fn.py).

    If output_dir already has a checkpoint at `iterations`, training is
    skipped entirely — a Colab session that disconnects and gets re-run
    from the top of the notebook must not burn GPU time re-training a
    scene that already finished. A fingerprint of the scene's image
    filenames is recorded alongside the checkpoint; if it doesn't match on
    a later call (dataset re-uploaded, holdout selection changed, etc.),
    the checkpoint is not safe to trust — output_dir is wiped and training
    starts clean rather than risk silently shipping a stale model.
    """
    output_dir = Path(output_dir)
    fingerprint = _scene_fingerprint(scene, iterations)
    fingerprint_path = output_dir / _FINGERPRINT_FILENAME

    existing = find_latest_checkpoint(output_dir)
    if existing is not None and fingerprint_path.exists() and fingerprint_path.read_text() != fingerprint:
        # A fingerprint was recorded by a previous real_train_fn call and it
        # no longer matches — the scene's contents changed since then (e.g.
        # dataset re-uploaded, holdout selection changed). Not safe to
        # resume from or reuse silently; wipe and start clean. A checkpoint
        # with NO fingerprint at all predates this check (or was placed
        # manually) — there's no evidence it's stale, so it's still trusted,
        # same as before this fix.
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
    # PYTHONUNBUFFERED forces train.py's own stdout (its tqdm progress bar
    # in particular) to flush immediately instead of block-buffering —
    # without it, Jupyter/Colab shows no output at all for long stretches
    # because the child inherits a piped (non-tty) stdout, which Python
    # buffers fully by default, making a genuinely-running 30k-iteration
    # training look stuck instead of showing live progress.
    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    subprocess.run(argv, cwd=str(GS_ROOT), check=True, env=env)

    checkpoint = find_latest_checkpoint(output_dir)
    if checkpoint is None or (checkpoint_iteration(checkpoint) or 0) < iterations:
        raise RuntimeError(
            f"train.py finished but no checkpoint at target iteration {iterations} "
            f"found in {output_dir}"
        )
    return checkpoint
