from __future__ import annotations

import sys
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.common.pose_utils import CameraParams
from src.rendering.render_from_csv import render_all

_GS_ROOT = Path(__file__).resolve().parents[2] / "third_party" / "gaussian-splatting"
if str(_GS_ROOT) not in sys.path:
    sys.path.insert(0, str(_GS_ROOT))


def _placeholder_image(width: int, height: int) -> Image.Image:
    """Blank image only used to give Camera.__init__ a pixel size to
    resize to (via PILtoTorch) — novel test poses have no ground-truth
    image, so its content is irrelevant, only its (width, height)."""
    return Image.new("RGB", (width, height), color=(0, 0, 0))


def _tensor_to_uint8_image(tensor: torch.Tensor) -> np.ndarray:
    """Convert gaussian_renderer.render()'s CHW float tensor in [0, 1]
    (values can exceed this range slightly, hence the clamp) to the HWC
    uint8 array render_from_csv.render_all expects."""
    array = tensor.detach().clamp(0.0, 1.0).cpu().numpy()
    array = np.transpose(array, (1, 2, 0))
    return (array * 255.0).round().astype(np.uint8)


def _default_opt_and_pipe():
    from arguments import OptimizationParams, PipelineParams

    parser = ArgumentParser()
    opt_group = OptimizationParams(parser)
    pipe_group = PipelineParams(parser)
    args = parser.parse_args([])
    return opt_group.extract(args), pipe_group.extract(args)


def _load_gaussians(checkpoint_path: Path):
    from scene.gaussian_model import GaussianModel

    # weights_only=False: this is our own checkpoint, produced by
    # gaussians.capture() (a tuple of tensors + optimizer state dict, not
    # just a plain state_dict) — PyTorch 2.6's default weights_only=True
    # rejects the plain Python types mixed into that tuple/optimizer
    # state. Trusted since we produced it ourselves in this same pipeline.
    model_args, _first_iter = torch.load(checkpoint_path, weights_only=False)
    opt, _pipe = _default_opt_and_pipe()
    gaussians = GaussianModel(sh_degree=3)  # matches ModelParams default
    # GaussianModel.restore() unconditionally calls training_setup(), which
    # builds an optimizer over self._exposure — but that attribute is only
    # ever created by create_from_pcd() (the fresh-training path in
    # train.py's Scene.__init__), never by restore() itself. A render-only
    # load skips create_from_pcd entirely (no point cloud, no camera list —
    # we're rendering novel test poses from an already-trained checkpoint),
    # so without this, training_setup() crashes with
    # AttributeError: 'GaussianModel' object has no attribute '_exposure'
    # (an actual Colab traceback). The placeholder's VALUE is irrelevant:
    # gaussian_renderer.render()'s use_trained_exp defaults to False and is
    # never overridden by real_render_fn below, so the exposure correction
    # is never applied during rendering — only _exposure's existence and
    # shape (matching create_from_pcd's [N, 3, 4] identity-transform
    # convention) matter, not its content.
    gaussians._exposure = torch.nn.Parameter(
        torch.eye(3, 4, device="cuda")[None].requires_grad_(True)
    )
    gaussians.restore(model_args, opt)
    return gaussians


def real_render_fn(
    checkpoint: Path, params_list: list[CameraParams], output_dir: Path,
) -> list[Path]:
    """GPU-only: loads the trained GaussianModel and renders every camera
    in params_list via the vendored gaussian_renderer.render(). Manual
    Colab verification only (no CUDA locally) — see
    docs/superpowers/plans/2026-07-20-colab-runner-notebook.md Task 7.
    """
    from gaussian_renderer import render as gs_render
    from scene.cameras import Camera

    gaussians = _load_gaussians(checkpoint)
    _opt, pipe = _default_opt_and_pipe()
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
        return _tensor_to_uint8_image(rendered)

    return render_all(
        checkpoint, None, output_dir, _render_one,
        params_list=params_list, gaussians=gaussians,
    )
