import numpy as np
import torch
import torch.nn as nn

from src.postprocess.prune_floaters import (
    _filter_gaussian_parameters,
    compute_prune_mask,
)


def test_keeps_normal_gaussians_inside_bbox():
    xyz = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
    opacity = np.array([0.5, 0.8])
    scales = np.array([[0.1, 0.1, 0.1], [0.1, 0.1, 0.1]])
    mask = compute_prune_mask(
        xyz, opacity, scales, bbox_min=np.array([-1, -1, -1]), bbox_max=np.array([1, 1, 1]),
    )
    assert mask.tolist() == [True, True]


def test_prunes_gaussian_outside_bbox():
    xyz = np.array([[0.0, 0.0, 0.0], [100.0, 100.0, 100.0]])
    opacity = np.array([0.5, 0.5])
    scales = np.array([[0.1, 0.1, 0.1], [0.1, 0.1, 0.1]])
    mask = compute_prune_mask(
        xyz, opacity, scales, bbox_min=np.array([-1, -1, -1]), bbox_max=np.array([1, 1, 1]),
    )
    assert mask.tolist() == [True, False]


def test_prunes_low_opacity_gaussian():
    xyz = np.array([[0.0, 0.0, 0.0], [0.1, 0.1, 0.1]])
    opacity = np.array([0.5, 0.01])
    scales = np.array([[0.1, 0.1, 0.1], [0.1, 0.1, 0.1]])
    mask = compute_prune_mask(
        xyz, opacity, scales, bbox_min=np.array([-1, -1, -1]), bbox_max=np.array([1, 1, 1]),
        opacity_threshold=0.05,
    )
    assert mask.tolist() == [True, False]


def test_prunes_outlier_scale_gaussian():
    xyz = np.tile(np.array([0.0, 0.0, 0.0]), (100, 1))
    opacity = np.full(100, 0.5)
    scales = np.full((100, 3), 0.1)
    scales[0] = [50.0, 50.0, 50.0]  # one giant outlier
    mask = compute_prune_mask(
        xyz, opacity, scales, bbox_min=np.array([-1, -1, -1]), bbox_max=np.array([1, 1, 1]),
        max_scale_percentile=99.5,
    )
    assert mask[0] == False
    assert mask[1:].all()


def test_filter_gaussian_parameters_filters_all_six_saved_tensors():
    class _FakeGaussians:
        pass

    gaussians = _FakeGaussians()
    gaussians._xyz = nn.Parameter(torch.arange(12).reshape(4, 3).float())
    gaussians._features_dc = nn.Parameter(torch.arange(12).reshape(4, 1, 3).float())
    gaussians._features_rest = nn.Parameter(torch.arange(24).reshape(4, 2, 3).float())
    gaussians._opacity = nn.Parameter(torch.arange(4).reshape(4, 1).float())
    gaussians._scaling = nn.Parameter(torch.arange(12).reshape(4, 3).float())
    gaussians._rotation = nn.Parameter(torch.arange(16).reshape(4, 4).float())

    _filter_gaussian_parameters(
        gaussians,
        np.array([True, False, True, False]),
    )

    for name in (
        "_xyz",
        "_features_dc",
        "_features_rest",
        "_opacity",
        "_scaling",
        "_rotation",
    ):
        tensor = getattr(gaussians, name)
        assert isinstance(tensor, nn.Parameter)
        assert tensor.shape[0] == 2
        assert tensor.requires_grad

    torch.testing.assert_close(
        gaussians._xyz,
        torch.tensor([[0.0, 1.0, 2.0], [6.0, 7.0, 8.0]]),
    )
