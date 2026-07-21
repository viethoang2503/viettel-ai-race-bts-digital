"""Auto-imported by Python at startup for any process whose PYTHONPATH
includes this directory (see the `site` module's sitecustomize hook).

Patches torch.load to default weights_only=False for the vendored
train.py subprocess, which we cannot edit directly: it's a git submodule
pinned to graphdeco-inria/gaussian-splatting, so any local edit here would
never reach a fresh `git clone --recurse-submodules` on Colab.

train.py's own --start_checkpoint resume path does
`(model_params, first_iter) = torch.load(checkpoint)` with no
weights_only argument — PyTorch 2.6+ changed torch.load's default from
weights_only=False to True, which rejects the plain Python/numpy types
mixed into our checkpoint's optimizer state (an actual Colab
CalledProcessError from train.py itself when resuming after a
disconnect, since a from-scratch run never hits --start_checkpoint at
all, only a resumed one). Trusted since these are checkpoints we
produced ourselves in this same pipeline, never external/untrusted
files — same reasoning already applied to our own torch.load calls in
src/evaluation/compute_metrics.py and src/rendering/gs_render_fn.py.
"""
import torch

_original_load = torch.load


def _load_trusting_source(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_load(*args, **kwargs)


torch.load = _load_trusting_source
