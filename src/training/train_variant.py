from __future__ import annotations

import sys
import hashlib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.common.config import SceneConfig
from src.training.gs_train_fn import _scene_fingerprint

_VENDORED_REPO = (
    Path(__file__).resolve().parents[2]
    / "third_party"
    / "gaussian-splatting"
)
_VARIANT_CHECKPOINT_RE = re.compile(r"variant_chkpnt(\d+)\.pth$")
_VARIANT_MANIFEST = "variant_run.json"
if str(_VENDORED_REPO) not in sys.path:
    sys.path.insert(0, str(_VENDORED_REPO))


@dataclass(frozen=True)
class TrainingVariant:
    name: str
    use_depth_reg: bool
    use_anti_alias: bool
    use_appearance_embed: bool


ALL_TRAINING_VARIANTS: list[TrainingVariant] = [
    TrainingVariant("baseline", False, False, False),
    TrainingVariant("depth_reg", True, False, False),
    TrainingVariant("anti_alias", False, True, False),
    TrainingVariant("appearance_embed", False, False, True),
    TrainingVariant("full_stack", True, True, True),
]


def _validate_training_request(
    iterations: int,
    seed: int,
    checkpoint_interval: int,
) -> None:
    if not isinstance(iterations, int) or iterations <= 0:
        raise ValueError("iterations must be a positive integer")
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if not isinstance(checkpoint_interval, int) or checkpoint_interval <= 0:
        raise ValueError("checkpoint_interval must be a positive integer")


def _seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _checkpoint_schedule(
    iterations: int,
    interval: int = 5000,
) -> list[int]:
    scheduled = list(range(interval, iterations, interval))
    scheduled.append(iterations)
    return scheduled


def _variant_run_fingerprint(
    scene: SceneConfig,
    variant: TrainingVariant,
    iterations: int,
    hyperparam_overrides: dict[str, object] | None,
    seed: int,
) -> str:
    payload = {
        "scene_fingerprint": _scene_fingerprint(scene, iterations),
        "variant": {
            "name": variant.name,
            "use_depth_reg": variant.use_depth_reg,
            "use_anti_alias": variant.use_anti_alias,
            "use_appearance_embed": variant.use_appearance_embed,
        },
        "iterations": iterations,
        "hyperparam_overrides": hyperparam_overrides or {},
        "seed": seed,
    }
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _atomic_torch_save(payload: dict, path: Path) -> None:
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _find_latest_matching_checkpoint(
    output_dir: Path,
    fingerprint: str,
) -> tuple[Path | None, dict | None]:
    import torch

    candidates = []
    for path in Path(output_dir).glob("variant_chkpnt*.pth"):
        match = _VARIANT_CHECKPOINT_RE.fullmatch(path.name)
        if match:
            candidates.append((int(match.group(1)), path))
    for _iteration, path in sorted(candidates, reverse=True):
        payload = torch.load(path, weights_only=False)
        if payload.get("fingerprint") == fingerprint:
            return path, payload
    return None, None


def _completed_variant_path(
    output_dir: Path,
    fingerprint: str,
    needs_appearance: bool,
) -> Path | None:
    output_dir = Path(output_dir)
    manifest_path = output_dir / _VARIANT_MANIFEST
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if manifest.get("fingerprint") != fingerprint:
        return None
    final_path = Path(manifest.get("final_checkpoint_path", ""))
    if not final_path.is_file():
        return None
    if needs_appearance and not (output_dir / "mean_appearance.pt").is_file():
        return None
    return final_path


def _capture_rng_state() -> dict:
    import torch

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": (
            torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None
        ),
    }


def _restore_rng_state(state: dict) -> None:
    import torch

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def _atomic_write_json(payload: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True))
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _build_variant_checkpoint_payload(
    gaussians,
    appearance,
    appearance_optimizer,
    iteration: int,
    fingerprint: str,
) -> dict:
    return {
        "iteration": iteration,
        "fingerprint": fingerprint,
        "gaussians": gaussians.capture(),
        "appearance": (
            appearance.state_dict()
            if appearance is not None
            else None
        ),
        "appearance_optimizer": (
            appearance_optimizer.state_dict()
            if appearance_optimizer is not None
            else None
        ),
        "rng": _capture_rng_state(),
    }


def _restore_variant_checkpoint(
    payload: dict,
    gaussians,
    opt,
    appearance,
    appearance_optimizer,
) -> None:
    gaussians.restore(payload["gaussians"], opt)
    if appearance is not None:
        if payload.get("appearance") is None:
            raise ValueError("resume checkpoint is missing appearance state")
        appearance.load_state_dict(payload["appearance"])
        if payload.get("appearance_optimizer") is None:
            raise ValueError(
                "resume checkpoint is missing appearance optimizer state"
            )
        appearance_optimizer.load_state_dict(
            payload["appearance_optimizer"]
        )
    _restore_rng_state(payload["rng"])


def _build_dataset_args(
    gs_source_dir: Path,
    model_path: Path,
    use_anti_alias: bool,
):
    """Build complete vendored argument objects with safe project defaults."""
    from argparse import ArgumentParser

    from arguments import ModelParams, OptimizationParams, PipelineParams

    parser = ArgumentParser()
    model_params = ModelParams(parser)
    optimization_params = OptimizationParams(parser)
    pipeline_params = PipelineParams(parser)
    args = parser.parse_args([])

    dataset = model_params.extract(args)
    dataset.source_path = str(Path(gs_source_dir).resolve())
    dataset.model_path = str(Path(model_path).resolve())
    dataset.eval = False
    # Match the proven Plan 1 baseline. The vendored -1 default silently
    # downsizes images wider than 1600 px and makes evaluation inconsistent.
    dataset.resolution = 1

    pipe = pipeline_params.extract(args)
    pipe.antialiasing = use_anti_alias

    opt = optimization_params.extract(args)
    # Match the OOM-safe defaults already validated by real_train_fn.
    # Per-candidate overrides may deliberately replace either value later.
    opt.densify_grad_threshold = 0.001
    opt.densify_until_iter = 10_000
    return dataset, pipe, opt


def _apply_hyperparam_overrides(
    opt,
    iterations: int,
    hyperparam_overrides: dict[str, object] | None,
) -> int:
    """Validate and apply overrides, returning the actual loop length."""
    overrides = dict(hyperparam_overrides or {})
    effective_iterations = overrides.pop("iterations", iterations)

    unknown_keys = [key for key in overrides if not hasattr(opt, key)]
    if unknown_keys:
        raise ValueError(
            "unknown training hyperparameter override: "
            f"{unknown_keys[0]!r}"
        )

    opt.iterations = effective_iterations
    for key, value in overrides.items():
        setattr(opt, key, value)
    return effective_iterations


def _prepare_depth_regularization_inputs(
    rendered_inverse_depth,
    pixel_xy: np.ndarray,
    sparse_depths: np.ndarray,
):
    """Align the vendored inverse-depth render with COLMAP Z targets.

    The CUDA rasterizer returns a singleton-channel ``(1, H, W)`` map of
    expected inverse depth. COLMAP tracks provide positive camera-space Z,
    so valid targets must be converted to ``1 / Z`` before comparison.
    """
    if rendered_inverse_depth.ndim == 3 and rendered_inverse_depth.shape[0] == 1:
        depth_map = rendered_inverse_depth[0]
    elif rendered_inverse_depth.ndim == 2:
        depth_map = rendered_inverse_depth
    else:
        raise ValueError(
            "rendered inverse depth must have shape (H, W) or (1, H, W), "
            f"got {tuple(rendered_inverse_depth.shape)}"
        )

    pixel_xy = np.asarray(pixel_xy)
    sparse_depths = np.asarray(sparse_depths, dtype=np.float64)
    valid = np.isfinite(sparse_depths) & (sparse_depths > 0)
    return depth_map, pixel_xy[valid], np.reciprocal(sparse_depths[valid])


def run_training_variant(
    scene: SceneConfig,
    variant: TrainingVariant,
    output_dir: Path,
    iterations: int,
    hyperparam_overrides: dict[str, object] | None = None,
    seed: int = 0,
    checkpoint_interval: int = 5000,
) -> Path:
    """Train one variant with the checked-out differentiable CUDA renderer."""
    _validate_training_request(iterations, seed, checkpoint_interval)
    from random import randint

    import torch
    from gaussian_renderer import render as gs_render
    from scene import GaussianModel, Scene
    from utils.loss_utils import l1_loss, ssim

    from src.common.colmap_io import load_sparse_scene
    from src.training.appearance_embedding import (
        AppearanceEmbedding,
        apply_appearance,
    )
    from src.training.depth_loss import depth_regularization_loss
    from src.training.sparse_depth import compute_sparse_depth_targets

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset, pipe, opt = _build_dataset_args(
        scene.gs_source_dir,
        output_dir,
        variant.use_anti_alias,
    )
    effective_iterations = _apply_hyperparam_overrides(
        opt,
        iterations,
        hyperparam_overrides,
    )
    _validate_training_request(
        effective_iterations,
        seed,
        checkpoint_interval,
    )
    fingerprint = _variant_run_fingerprint(
        scene,
        variant,
        effective_iterations,
        hyperparam_overrides,
        seed,
    )
    completed_path = _completed_variant_path(
        output_dir,
        fingerprint,
        needs_appearance=variant.use_appearance_embed,
    )
    if completed_path is not None:
        return completed_path

    _resume_path, resume_payload = _find_latest_matching_checkpoint(
        output_dir,
        fingerprint,
    )
    _seed_everything(seed)

    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    gs_scene = Scene(dataset, gaussians)

    sparse = (
        load_sparse_scene(scene.sparse_dir)
        if variant.use_depth_reg
        else None
    )
    sparse_images_by_name = (
        {image.name: image for image in sparse.images.values()}
        if sparse is not None
        else None
    )

    train_cameras = gs_scene.getTrainCameras()
    appearance = (
        AppearanceEmbedding(num_images=len(train_cameras)).cuda()
        if variant.use_appearance_embed
        else None
    )
    appearance_optimizer = None
    if appearance is not None:
        appearance_optimizer = torch.optim.Adam(
            appearance.parameters(),
            lr=1e-3,
        )

    if resume_payload is None:
        gaussians.training_setup(opt)
        first_iteration = 1
    else:
        _restore_variant_checkpoint(
            resume_payload,
            gaussians,
            opt,
            appearance,
            appearance_optimizer,
        )
        first_iteration = int(resume_payload["iteration"]) + 1

    background = torch.tensor(
        [0.0, 0.0, 0.0],
        dtype=torch.float32,
        device="cuda",
    )

    checkpoint_iterations = set(
        _checkpoint_schedule(
            effective_iterations,
            checkpoint_interval,
        )
    )
    for iteration in range(first_iteration, effective_iterations + 1):
        gaussians.update_learning_rate(iteration)
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        viewpoint_cam = train_cameras[randint(0, len(train_cameras) - 1)]
        render_pkg = gs_render(
            viewpoint_cam,
            gaussians,
            pipe,
            background,
        )
        image = render_pkg["render"]
        viewspace_points = render_pkg["viewspace_points"]
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]

        if appearance is not None:
            affine, bias = appearance(viewpoint_cam.uid)
            image = apply_appearance(image, affine, bias)

        gt_image = viewpoint_cam.original_image.cuda()
        loss = (
            (1.0 - opt.lambda_dssim) * l1_loss(image, gt_image)
            + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        )

        if variant.use_depth_reg:
            colmap_image = sparse_images_by_name[viewpoint_cam.image_name]
            pixel_xy, sparse_depths = compute_sparse_depth_targets(
                colmap_image.qvec,
                colmap_image.tvec,
                colmap_image.xys,
                colmap_image.point3D_ids,
                sparse.points3d,
            )
            depth_map, pixel_xy, sparse_inverse_depths = (
                _prepare_depth_regularization_inputs(
                    render_pkg["depth"],
                    pixel_xy,
                    sparse_depths,
                )
            )
            loss = loss + 0.1 * depth_regularization_loss(
                depth_map,
                pixel_xy,
                sparse_inverse_depths,
            )

        loss.backward()

        with torch.no_grad():
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter],
                    radii[visibility_filter],
                )
                gaussians.add_densification_stats(
                    viewspace_points,
                    visibility_filter,
                )
                if (
                    iteration > opt.densify_from_iter
                    and iteration % opt.densification_interval == 0
                ):
                    size_threshold = (
                        20
                        if iteration > opt.opacity_reset_interval
                        else None
                    )
                    gaussians.densify_and_prune(
                        opt.densify_grad_threshold,
                        0.005,
                        gs_scene.cameras_extent,
                        size_threshold,
                        radii,
                    )
                if iteration % opt.opacity_reset_interval == 0:
                    gaussians.reset_opacity()

            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none=True)
            if appearance is not None:
                appearance_optimizer.step()
                appearance_optimizer.zero_grad(set_to_none=True)

        if iteration in checkpoint_iterations:
            checkpoint_payload = _build_variant_checkpoint_payload(
                gaussians,
                appearance,
                appearance_optimizer,
                iteration,
                fingerprint,
            )
            _atomic_torch_save(
                checkpoint_payload,
                output_dir / f"variant_chkpnt{iteration}.pth",
            )

    if appearance is not None:
        _save_mean_appearance(appearance, output_dir)

    gs_scene.save(effective_iterations)
    final_path = (
        output_dir
        / "point_cloud"
        / f"iteration_{effective_iterations}"
        / "point_cloud.ply"
    )
    _atomic_write_json(
        {
            "fingerprint": fingerprint,
            "iteration": effective_iterations,
            "seed": seed,
            "final_checkpoint_path": str(final_path.resolve()),
        },
        output_dir / _VARIANT_MANIFEST,
    )
    return final_path


def _save_mean_appearance(appearance, output_dir: Path) -> None:
    import torch

    affine, bias = appearance.mean_affine_bias()
    torch.save(
        {"affine": affine.cpu(), "bias": bias.cpu()},
        Path(output_dir) / "mean_appearance.pt",
    )
