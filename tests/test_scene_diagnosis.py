from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.diagnostics.scene_diagnosis import (
    compute_per_image_metrics,
    rank_holdout_by_score,
)


class _StubLpipsModel:
    def __call__(self, pred_tensor, gt_tensor):
        identical = torch.allclose(pred_tensor, gt_tensor)
        return torch.tensor(0.0 if identical else 1.0)


def _write_image(path: Path, fill: int) -> None:
    Image.fromarray(np.full((8, 8, 3), fill, dtype=np.uint8)).save(path)


def test_compute_per_image_metrics_matches_pred_and_gt_by_filename(tmp_path):
    pred_dir = tmp_path / "pred"
    gt_dir = tmp_path / "gt"
    pred_dir.mkdir()
    gt_dir.mkdir()

    _write_image(pred_dir / "frame_0001.jpg", fill=100)
    _write_image(gt_dir / "frame_0001.jpg", fill=100)
    _write_image(pred_dir / "frame_0002.jpg", fill=50)
    _write_image(gt_dir / "frame_0002.jpg", fill=200)

    result = compute_per_image_metrics(
        pred_dir,
        gt_dir,
        _StubLpipsModel(),
        psnr_max=30.0,
    )

    assert set(result.keys()) == {"frame_0001.jpg", "frame_0002.jpg"}
    for metrics in result.values():
        assert set(metrics.keys()) == {"lpips", "ssim", "psnr", "score"}
    assert result["frame_0001.jpg"]["score"] > result["frame_0002.jpg"]["score"]


def test_compute_per_image_metrics_skips_predictions_with_no_matching_ground_truth(
    tmp_path,
):
    pred_dir = tmp_path / "pred"
    gt_dir = tmp_path / "gt"
    pred_dir.mkdir()
    gt_dir.mkdir()

    _write_image(pred_dir / "frame_0001.jpg", fill=100)
    _write_image(gt_dir / "frame_0001.jpg", fill=100)
    _write_image(pred_dir / "frame_orphan.jpg", fill=10)

    result = compute_per_image_metrics(
        pred_dir,
        gt_dir,
        _StubLpipsModel(),
        psnr_max=30.0,
    )

    assert set(result.keys()) == {"frame_0001.jpg"}


def test_rank_holdout_by_score_sorts_worst_first():
    per_image = {
        "good.jpg": {"lpips": 0.1, "ssim": 0.9, "psnr": 28.0, "score": 0.85},
        "bad.jpg": {"lpips": 0.6, "ssim": 0.4, "psnr": 12.0, "score": 0.30},
        "medium.jpg": {"lpips": 0.3, "ssim": 0.6, "psnr": 20.0, "score": 0.55},
    }
    ranked = rank_holdout_by_score(per_image)
    assert ranked == [
        ("bad.jpg", 0.30),
        ("medium.jpg", 0.55),
        ("good.jpg", 0.85),
    ]


def test_rank_holdout_by_score_handles_empty_input():
    assert rank_holdout_by_score({}) == []
