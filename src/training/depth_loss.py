from __future__ import annotations

import numpy as np
import torch


def depth_regularization_loss(
    rendered_depth: torch.Tensor, pixel_xy: np.ndarray, sparse_depths: np.ndarray,
) -> torch.Tensor:
    if len(sparse_depths) == 0:
        return torch.tensor(0.0, device=rendered_depth.device)

    height, width = rendered_depth.shape
    cols = np.clip(np.round(pixel_xy[:, 0]).astype(int), 0, width - 1)
    rows = np.clip(np.round(pixel_xy[:, 1]).astype(int), 0, height - 1)

    sampled = rendered_depth[rows, cols]
    targets = torch.as_tensor(sparse_depths, dtype=sampled.dtype, device=sampled.device)
    return torch.abs(sampled - targets).mean()
