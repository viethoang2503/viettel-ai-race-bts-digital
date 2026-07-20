from __future__ import annotations

import math

import numpy as np


def select_holdout_images(
    camera_centers: dict[str, np.ndarray], holdout_ratio: float = 0.125,
) -> list[str]:
    """Pick the images whose camera center is farthest from the centroid of
    all camera centers in the scene — approximates the edge-of-coverage
    poses that real test poses are likely to extrapolate toward, rather
    than a uniform every-Nth split.
    """
    if not camera_centers:
        raise ValueError("camera_centers is empty, cannot select holdout images")

    names = list(camera_centers.keys())
    centers = np.stack([camera_centers[n] for n in names], axis=0)
    centroid = centers.mean(axis=0)
    distances = np.linalg.norm(centers - centroid, axis=1)

    n_holdout = max(1, math.floor(len(names) * holdout_ratio))
    order = np.argsort(-distances)  # descending distance
    return [names[i] for i in order[:n_holdout]]
