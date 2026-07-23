from __future__ import annotations

import torch
import torch.nn as nn


def apply_appearance(rgb: torch.Tensor, affine: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """rgb: (3,H,W) in [0,1]. affine: (3,3). bias: (3,). Returns (3,H,W)."""
    c, h, w = rgb.shape
    flat = rgb.reshape(c, h * w)
    transformed = affine @ flat + bias.unsqueeze(1)
    return transformed.reshape(c, h, w).clamp(0.0, 1.0)


class AppearanceEmbedding(nn.Module):
    """One learnable (3,3) affine + (3,) bias per training image, indexed
    by image position in the training set. Initialized to identity/zero so
    training starts as a photometric no-op and only diverges from it where
    the loss actually benefits from explaining away per-image exposure
    variation -- it must never be applied when rendering novel test poses,
    since there is no "true" appearance code for an unseen view: use the
    mean of all training embeddings instead (mean_affine_bias), computed by
    the caller, not by this module.
    """

    def __init__(self, num_images: int):
        super().__init__()
        self.affine = nn.Parameter(torch.eye(3).unsqueeze(0).repeat(num_images, 1, 1))
        self.bias = nn.Parameter(torch.zeros(num_images, 3))

    def forward(self, image_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.affine[image_idx], self.bias[image_idx]

    def mean_affine_bias(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Canonical appearance for novel-view rendering: the average
        learned correction across all training images, since a test pose
        has no ground-truth appearance to match.
        """
        return self.affine.mean(dim=0).detach(), self.bias.mean(dim=0).detach()
