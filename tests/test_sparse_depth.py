import numpy as np

from src.training.sparse_depth import compute_sparse_depth_targets


class _FakePoint3D:
    def __init__(self, xyz):
        self.xyz = np.array(xyz)


def test_identity_rotation_depth_equals_camera_z_translation():
    qvec = np.array([1.0, 0.0, 0.0, 0.0])  # identity rotation
    tvec = np.array([0.0, 0.0, 5.0])
    xys = np.array([[10.0, 20.0]])
    point3d_ids = np.array([0])
    points3d = {0: _FakePoint3D([0.0, 0.0, 0.0])}  # world origin

    pixel_xy, depth = compute_sparse_depth_targets(qvec, tvec, xys, point3d_ids, points3d)

    assert pixel_xy.shape == (1, 2)
    np.testing.assert_allclose(pixel_xy[0], [10.0, 20.0])
    np.testing.assert_allclose(depth, [5.0], atol=1e-10)


def test_filters_out_unassociated_keypoints():
    qvec = np.array([1.0, 0.0, 0.0, 0.0])
    tvec = np.array([0.0, 0.0, 5.0])
    xys = np.array([[10.0, 20.0], [30.0, 40.0]])
    point3d_ids = np.array([0, -1])  # second keypoint has no 3D point
    points3d = {0: _FakePoint3D([0.0, 0.0, 0.0])}

    pixel_xy, depth = compute_sparse_depth_targets(qvec, tvec, xys, point3d_ids, points3d)

    assert pixel_xy.shape == (1, 2)
    assert depth.shape == (1,)


def test_filters_out_point_ids_not_present_in_points3d_dict():
    # can happen if points3D.bin and images.bin are slightly out of sync
    qvec = np.array([1.0, 0.0, 0.0, 0.0])
    tvec = np.array([0.0, 0.0, 5.0])
    xys = np.array([[10.0, 20.0]])
    point3d_ids = np.array([999])  # not in points3d
    points3d = {0: _FakePoint3D([0.0, 0.0, 0.0])}

    pixel_xy, depth = compute_sparse_depth_targets(qvec, tvec, xys, point3d_ids, points3d)

    assert pixel_xy.shape == (0, 2)
    assert depth.shape == (0,)


def test_nonzero_translation_and_offset_point():
    qvec = np.array([1.0, 0.0, 0.0, 0.0])
    tvec = np.array([1.0, 2.0, 3.0])
    xys = np.array([[0.0, 0.0]])
    point3d_ids = np.array([0])
    points3d = {0: _FakePoint3D([0.0, 0.0, 2.0])}  # world point at z=2

    pixel_xy, depth = compute_sparse_depth_targets(qvec, tvec, xys, point3d_ids, points3d)

    # camera-space z = (I @ [0,0,2]) + [1,2,3] -> z component = 2 + 3 = 5
    np.testing.assert_allclose(depth, [5.0], atol=1e-10)
