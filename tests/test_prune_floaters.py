import numpy as np

from src.postprocess.prune_floaters import compute_prune_mask


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
