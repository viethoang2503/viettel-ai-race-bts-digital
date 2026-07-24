from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from src.evaluation.compute_metrics import combine_score, compute_pair_metrics


def compute_per_image_metrics(
    pred_dir: Path,
    gt_dir: Path,
    lpips_model,
    psnr_max: float,
) -> dict[str, dict[str, float]]:
    pred_dir = Path(pred_dir)
    gt_dir = Path(gt_dir)
    result: dict[str, dict[str, float]] = {}

    for pred_path in sorted(pred_dir.iterdir()):
        if not pred_path.is_file():
            continue

        gt_path = gt_dir / pred_path.name
        if not gt_path.is_file():
            continue

        pred = np.array(Image.open(pred_path).convert("RGB"))
        gt = np.array(Image.open(gt_path).convert("RGB").resize(pred.shape[1::-1]))
        metrics = compute_pair_metrics(pred, gt, lpips_model)
        score = combine_score(
            metrics["lpips"],
            metrics["ssim"],
            metrics["psnr"],
            psnr_max,
        )
        result[pred_path.name] = {**metrics, "score": score}

    return result
