from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def find_suspicious_renders(render_dir: Path, std_threshold: float = 15.0) -> list[tuple[Path, float]]:
    """Flag rendered images with suspiciously low pixel variance — a strong
    signal of a near-blank render (e.g. a test pose extrapolating beyond
    all training camera coverage, with no Gaussians visible from that
    viewpoint). A normal photo has pixel std well above this threshold;
    a blank/near-uniform image sits close to 0.

    This exists because validate_submission only checks size/format/
    decodability, never content plausibility — a blank image of the exact
    right dimensions passes it silently. This is a cheap heuristic scan to
    catch that before submitting, not a replacement for actually looking
    at the images (spec section 13's visual QA step).

    Returns (path, std) pairs for every flagged image, sorted by std
    ascending (most suspicious first).
    """
    render_dir = Path(render_dir)
    suspicious = []
    for path in sorted(render_dir.iterdir()):
        if not path.is_file():
            continue
        img = np.array(Image.open(path).convert("RGB"))
        std = float(img.std())
        if std < std_threshold:
            suspicious.append((path, std))
    suspicious.sort(key=lambda pair: pair[1])
    return suspicious
