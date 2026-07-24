from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_GS_ROOT = (
    Path(__file__).resolve().parents[2]
    / "third_party"
    / "gaussian-splatting"
)
if str(_GS_ROOT) not in sys.path:
    sys.path.insert(0, str(_GS_ROOT))


def compute_prune_mask(
    xyz: np.ndarray, opacity: np.ndarray, scales: np.ndarray,
    bbox_min: np.ndarray, bbox_max: np.ndarray,
    opacity_threshold: float = 0.05, max_scale_percentile: float = 99.5,
) -> np.ndarray:
    inside_bbox = np.all((xyz >= bbox_min) & (xyz <= bbox_max), axis=1)
    opaque_enough = opacity >= opacity_threshold

    max_scale_per_gaussian = scales.max(axis=1)
    scale_cutoff = np.percentile(max_scale_per_gaussian, max_scale_percentile)
    not_outlier_scale = max_scale_per_gaussian <= scale_cutoff

    return inside_bbox & opaque_enough & not_outlier_scale


def _filter_gaussian_parameters(gaussians, keep_mask: np.ndarray) -> None:
    import torch
    import torch.nn as nn

    keep_mask_t = torch.from_numpy(
        np.asarray(keep_mask, dtype=bool),
    ).to(gaussians._xyz.device)

    for attribute in (
        "_xyz",
        "_features_dc",
        "_features_rest",
        "_opacity",
        "_scaling",
        "_rotation",
    ):
        tensor = getattr(gaussians, attribute)
        setattr(
            gaussians,
            attribute,
            nn.Parameter(tensor[keep_mask_t].detach()),
        )


def prune_checkpoint(
    checkpoint_path: Path,
    bbox_min: np.ndarray,
    bbox_max: np.ndarray,
    sh_degree: int = 3,
) -> Path:
    from scene import GaussianModel

    checkpoint_path = Path(checkpoint_path)
    gaussians = GaussianModel(sh_degree)
    gaussians.load_ply(str(checkpoint_path))

    xyz = gaussians._xyz.detach().cpu().numpy()
    opacity = gaussians.get_opacity.detach().cpu().numpy().squeeze(-1)
    scales = gaussians.get_scaling.detach().cpu().numpy()

    keep_mask = compute_prune_mask(
        xyz,
        opacity,
        scales,
        bbox_min,
        bbox_max,
    )

    # Do not call GaussianModel.prune_points(): it routes through
    # _prune_optimizer(), but load_ply() does not create an optimizer.
    _filter_gaussian_parameters(gaussians, keep_mask)

    output_path = checkpoint_path.with_name(
        f"{checkpoint_path.stem}_pruned.ply",
    )
    gaussians.save_ply(str(output_path))
    return output_path
