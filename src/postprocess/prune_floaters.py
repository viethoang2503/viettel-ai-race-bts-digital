from __future__ import annotations

import numpy as np


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
