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


class VramBudgetExceededError(RuntimeError):
    """Raised when a checkpoint or render cannot fit the target GPU budget."""


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
    checkpoint_path = Path(checkpoint_path)
    suffix = checkpoint_path.suffix.lower()
    if suffix not in {".ply", ".pth"}:
        raise ValueError(
            f"unsupported gaussian checkpoint format: {checkpoint_path}"
        )

    from scene.gaussian_model import GaussianModel

    gaussians = GaussianModel(sh_degree=3)  # matches ModelParams default

    if suffix == ".ply":
        gaussians.load_ply(str(checkpoint_path))
        return gaussians

    # weights_only=False: this is our own checkpoint, produced by
    # gaussians.capture() (a tuple of tensors + optimizer state dict, not
    # just a plain state_dict) — PyTorch 2.6's default weights_only=True
    # rejects the plain Python types mixed into that tuple/optimizer
    # state. Trusted since we produced it ourselves in this same pipeline.
    model_args, _first_iter = torch.load(
        checkpoint_path,
        weights_only=False,
    )
    opt, _pipe = _default_opt_and_pipe()
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


def _parse_render_config(
    render_config: dict | None,
) -> tuple[bool, Path | None, int | None]:
    render_config = render_config or {}
    appearance_path = render_config.get("appearance_path")
    vram_budget = render_config.get("vram_budget_bytes")
    if vram_budget is not None:
        vram_budget = int(vram_budget)
        if vram_budget <= 0:
            raise ValueError("vram_budget_bytes must be positive")
    return (
        bool(render_config.get("antialiasing", False)),
        Path(appearance_path) if appearance_path is not None else None,
        vram_budget,
    )


def _load_appearance_affine_bias(
    appearance_path: Path | None,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    if appearance_path is None:
        return None
    appearance_path = Path(appearance_path)
    if not appearance_path.is_file():
        raise FileNotFoundError(
            f"requested appearance artifact does not exist: {appearance_path}"
        )
    saved = torch.load(appearance_path, weights_only=False)
    return saved["affine"].cuda(), saved["bias"].cuda()


def _run_inference(callable_, *args, **kwargs):
    with torch.inference_mode():
        return callable_(*args, **kwargs)


def _validate_peak_vram(
    peak_bytes: int,
    budget_bytes: int | None,
) -> None:
    if budget_bytes is not None and peak_bytes > budget_bytes:
        raise VramBudgetExceededError(
            f"render peak VRAM {peak_bytes} bytes exceeds budget "
            f"{budget_bytes} bytes"
        )


def real_render_fn(
    checkpoint: Path, params_list: list[CameraParams], output_dir: Path,
    render_config: dict | None = None,
) -> list[Path]:
    """GPU-only: loads the trained GaussianModel and renders every camera
    in params_list via the vendored gaussian_renderer.render(). Manual
    Colab verification only (no CUDA locally) — see
    docs/superpowers/plans/2026-07-20-colab-runner-notebook.md Task 7.
    """
    from gaussian_renderer import render as gs_render
    from scene.cameras import Camera
    from src.training.appearance_embedding import apply_appearance

    antialiasing, appearance_path, vram_budget = _parse_render_config(
        render_config
    )
    if vram_budget is not None and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    gaussians = _load_gaussians(checkpoint)
    _opt, pipe = _default_opt_and_pipe()
    pipe.antialiasing = antialiasing

    appearance_affine_bias = _load_appearance_affine_bias(appearance_path)

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
        if appearance_affine_bias is not None:
            affine, bias = appearance_affine_bias
            rendered = apply_appearance(rendered, affine, bias)
        return _tensor_to_uint8_image(rendered)

    written = _run_inference(
        render_all,
        checkpoint,
        None,
        output_dir,
        _render_one,
        params_list=params_list,
        gaussians=gaussians,
    )
    if vram_budget is not None:
        peak_bytes = (
            int(torch.cuda.max_memory_allocated())
            if torch.cuda.is_available()
            else 0
        )
        if render_config is not None:
            render_config["measured_peak_vram_bytes"] = peak_bytes
        _validate_peak_vram(peak_bytes, vram_budget)
    return written
