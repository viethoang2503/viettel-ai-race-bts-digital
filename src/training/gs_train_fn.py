from __future__ import annotations

import subprocess
from pathlib import Path

from src.common.config import SceneConfig
from src.training.train_wrapper import build_train_argv, checkpoint_iteration, find_latest_checkpoint

GS_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "gaussian-splatting"


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
    subprocess.run(argv, cwd=str(GS_ROOT), check=True)

    checkpoint = find_latest_checkpoint(output_dir)
    if checkpoint is None or (checkpoint_iteration(checkpoint) or 0) < iterations:
        raise RuntimeError(
            f"train.py finished but no checkpoint at target iteration {iterations} "
            f"found in {output_dir}"
        )
    return checkpoint
