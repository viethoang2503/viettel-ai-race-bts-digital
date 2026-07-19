import numpy as np

from src.evaluation.make_holdout_split import select_holdout_images


def test_select_holdout_picks_farthest_points_from_centroid():
    centers = {f"cluster_{i}": np.array([0.01 * i, 0.0, 0.0]) for i in range(10)}
    centers["outlier_1"] = np.array([100.0, 0.0, 0.0])
    centers["outlier_2"] = np.array([-100.0, 0.0, 0.0])

    holdout = select_holdout_images(centers, holdout_ratio=0.2)  # 12 * 0.2 -> 2 (floor)

    assert set(holdout) == {"outlier_1", "outlier_2"}


def test_select_holdout_ratio_controls_count():
    centers = {f"img_{i}": np.array([float(i), 0.0, 0.0]) for i in range(20)}
    holdout = select_holdout_images(centers, holdout_ratio=0.5)
    assert len(holdout) == 10


def test_select_holdout_never_returns_empty_for_nonzero_ratio():
    centers = {"only_one": np.array([0.0, 0.0, 0.0])}
    holdout = select_holdout_images(centers, holdout_ratio=0.125)
    assert holdout == ["only_one"]
