from __future__ import annotations

import re
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
        "python", "train.py",
        "--source_path", str(Path(scene.gs_source_dir).resolve()),
        "--model_path", str(Path(output_dir).resolve()),
        "--iterations", str(iterations),
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
