from __future__ import annotations

import re
import sys
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


def checkpoint_iteration(path: Path) -> int | None:
    match = _CHECKPOINT_RE.search(Path(path).name)
    return int(match.group(1)) if match else None


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
        sys.executable, "train.py",
        "--source_path", str(Path(scene.gs_source_dir).resolve()),
        "--model_path", str(Path(output_dir).resolve()),
        "--iterations", str(iterations),
        # Without this, the vendored loader auto-downscales any image
        # wider than 1600px (its default --resolution=-1 behavior) — the
        # real BTS/bonsai/chair scenes exceed that, so training would
        # silently happen at a lower resolution than what we render at
        # (test_poses.csv's exact width/height) and score against (the
        # full-resolution holdout images). Reproduced on a real Colab run:
        # bonsai (1920x1080) trained without this flag scored PSNR ~14.8
        # with visibly blurry renders against a sharp ground truth —
        # exactly the symptom of Gaussians optimized for a coarser pixel
        # grid than they're displayed at. "1" forces native resolution.
        "--resolution", "1",
        # Real BTS scenes can OOM a 22GB GPU well before reaching
        # `iterations` — reproduced on a real Colab run (HCM0421, L4,
        # 240 images): CUDA out of memory at iteration ~5300 during
        # densify_and_prune, with the Gaussian count still growing (the
        # vendored default densifies every 100 iterations from 500 to
        # 15000, unbounded — no max-Gaussian-count flag exists in this
        # version). Both flags below cut total growth: a stricter split/
        # clone gradient threshold (5x the 0.0002 default) creates fewer
        # new Gaussians per event, and stopping densification 5000
        # iterations earlier caps how many events can happen at all. This
        # also helps satisfy the exam's real inference constraint (spec
        # section 7: BTC renders on a 20GB A4000, smaller than even this
        # 22GB training GPU) at some cost to fine detail versus an
        # unconstrained run — an unavoidable tradeoff given training
        # itself can't complete without it.
        "--densify_grad_threshold", "0.001",
        "--densify_until_iter", "10000",
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
