import numpy as np
import pytest
import torch

from src.training.depth_loss import depth_regularization_loss


def test_zero_loss_when_rendered_depth_matches_targets_exactly():
    rendered_depth = torch.full((10, 10), 5.0)
    pixel_xy = np.array([[3.0, 4.0], [7.0, 2.0]])
    sparse_depths = np.array([5.0, 5.0])
    loss = depth_regularization_loss(rendered_depth, pixel_xy, sparse_depths)
    assert loss.item() == 0.0


def test_positive_loss_when_rendered_depth_differs():
    rendered_depth = torch.full((10, 10), 5.0)
    pixel_xy = np.array([[3.0, 4.0]])
    sparse_depths = np.array([8.0])
    loss = depth_regularization_loss(rendered_depth, pixel_xy, sparse_depths)
    assert loss.item() == pytest.approx(3.0, abs=1e-5)


def test_zero_loss_and_no_crash_with_no_sparse_points():
    rendered_depth = torch.full((10, 10), 5.0)
    pixel_xy = np.zeros((0, 2))
    sparse_depths = np.zeros((0,))
    loss = depth_regularization_loss(rendered_depth, pixel_xy, sparse_depths)
    assert loss.item() == 0.0
    assert not torch.isnan(loss)
