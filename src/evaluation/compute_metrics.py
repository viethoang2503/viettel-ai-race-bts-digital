from __future__ import annotations

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def combine_score(lpips_val: float, ssim_val: float, psnr_val: float, psnr_max: float) -> float:
    """Score = 0.4*(1-LPIPS) + 0.3*SSIM + 0.3*PSNR_norm, PSNR_norm clamped to [0,1].

    Pure function, no torch/skimage dependency, so it stays importable and
    auditable (this is the exam's literal grading formula) even in a
    minimal environment that doesn't have the full ML stack installed.
    """
    if psnr_max <= 0:
        raise ValueError(f"psnr_max must be positive, got {psnr_max}")
    psnr_norm = max(0.0, min(1.0, psnr_val / psnr_max))
    return 0.4 * (1 - lpips_val) + 0.3 * ssim_val + 0.3 * psnr_norm


def _to_lpips_tensor(img: np.ndarray):
    # (H,W,3) uint8 [0,255] -> (1,3,H,W) float32 in [-1,1], as expected by lpips.LPIPS
    import torch

    t = torch.from_numpy(img).float() / 127.5 - 1.0
    return t.permute(2, 0, 1).unsqueeze(0)


def compute_pair_metrics(pred: np.ndarray, gt: np.ndarray, lpips_model) -> dict:
    import torch

    assert pred.shape == gt.shape, f"shape mismatch: {pred.shape} vs {gt.shape}"

    ssim_val = structural_similarity(pred, gt, channel_axis=2, data_range=255)

    mse = np.mean((pred.astype(np.float64) - gt.astype(np.float64)) ** 2)
    if mse == 0:
        psnr_val = 100.0  # treat identical images as a high finite ceiling, not inf
    else:
        psnr_val = peak_signal_noise_ratio(gt, pred, data_range=255)

    with torch.no_grad():
        lpips_val = float(lpips_model(_to_lpips_tensor(pred), _to_lpips_tensor(gt)))

    return {"lpips": lpips_val, "ssim": float(ssim_val), "psnr": float(psnr_val)}


def load_lpips_model(net: str = "alex"):
    import torch
    import lpips

    # lpips's bundled weight file predates PyTorch 2.6's default
    # torch.load(weights_only=True) — loading it unpatched raises
    # UnpicklingError (numpy.core.multiarray.scalar not an allowed
    # global). We trust this file (it ships inside the lpips PyPI
    # package itself, not user-supplied), so force weights_only=False
    # just for this load, then restore torch.load immediately after.
    original_load = torch.load

    def _load_trusting_source(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_load(*args, **kwargs)

    torch.load = _load_trusting_source
    try:
        model = lpips.LPIPS(net=net)
    finally:
        torch.load = original_load
    model.eval()
    return model
