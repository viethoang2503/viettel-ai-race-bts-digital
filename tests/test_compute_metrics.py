import numpy as np
import pytest
import torch

from src.evaluation.compute_metrics import combine_score, compute_pair_metrics


def test_combine_score_matches_spec_formula():
    # Score = 0.4*(1-LPIPS) + 0.3*SSIM + 0.3*PSNR_norm
    score = combine_score(lpips_val=0.2, ssim_val=0.9, psnr_val=30.0, psnr_max=30.0)
    expected = 0.4 * (1 - 0.2) + 0.3 * 0.9 + 0.3 * 1.0
    assert score == pytest.approx(expected, abs=1e-10)


def test_combine_score_clamps_psnr_norm_above_max():
    score = combine_score(lpips_val=0.0, ssim_val=1.0, psnr_val=999.0, psnr_max=30.0)
    # psnr_norm clamped to 1.0, so score == 0.4*1 + 0.3*1 + 0.3*1
    assert score == pytest.approx(1.0, abs=1e-10)


def test_combine_score_clamps_psnr_norm_below_zero():
    score = combine_score(lpips_val=1.0, ssim_val=0.0, psnr_val=-10.0, psnr_max=30.0)
    assert score == pytest.approx(0.0, abs=1e-10)


def test_combine_score_rejects_non_positive_psnr_max():
    with pytest.raises(ValueError):
        combine_score(lpips_val=0.1, ssim_val=0.9, psnr_val=20.0, psnr_max=0.0)
    with pytest.raises(ValueError):
        combine_score(lpips_val=0.1, ssim_val=0.9, psnr_val=20.0, psnr_max=-5.0)


def test_combine_score_importable_without_torch_installed(monkeypatch):
    # combine_score is the exam's literal grading formula and must stay
    # auditable/usable without pulling in the full ML stack. Simulate
    # torch being unavailable and confirm importing the module (and
    # calling combine_score) still works.
    import builtins
    import importlib
    import sys

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "torch":
            raise ImportError("simulated: torch not installed")
        return real_import(name, *args, **kwargs)

    sys.modules.pop("src.evaluation.compute_metrics", None)
    monkeypatch.setattr(builtins, "__import__", fake_import)
    module = importlib.import_module("src.evaluation.compute_metrics")
    score = module.combine_score(lpips_val=0.1, ssim_val=0.9, psnr_val=30.0, psnr_max=30.0)
    assert score == pytest.approx(0.4 * 0.9 + 0.3 * 0.9 + 0.3 * 1.0, abs=1e-10)


class _StubLpipsModel:
    """Returns 0 distance for identical inputs, 1 otherwise."""

    def __call__(self, pred_tensor, gt_tensor):
        identical = torch.allclose(pred_tensor, gt_tensor)
        return torch.tensor(0.0 if identical else 1.0)


def test_compute_pair_metrics_identical_images_score_well():
    img = np.random.default_rng(0).integers(0, 255, size=(32, 32, 3), dtype=np.uint8)
    result = compute_pair_metrics(pred=img, gt=img, lpips_model=_StubLpipsModel())
    assert result["lpips"] == pytest.approx(0.0, abs=1e-6)
    assert result["ssim"] == pytest.approx(1.0, abs=1e-6)
    assert result["psnr"] > 40.0  # identical images -> very high/inf PSNR


def test_compute_pair_metrics_different_images_are_worse():
    rng = np.random.default_rng(0)
    pred = rng.integers(0, 255, size=(32, 32, 3), dtype=np.uint8)
    gt = rng.integers(0, 255, size=(32, 32, 3), dtype=np.uint8)
    result = compute_pair_metrics(pred=pred, gt=gt, lpips_model=_StubLpipsModel())
    assert result["lpips"] == pytest.approx(1.0, abs=1e-6)
    assert result["ssim"] < 1.0
