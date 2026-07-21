import numpy as np
from PIL import Image

from src.submission.visual_qa import find_suspicious_renders


def test_find_suspicious_renders_flags_near_blank_image(tmp_path):
    blank = np.full((64, 64, 3), 230, dtype=np.uint8)  # near-uniform, like a blank extrapolated render
    Image.fromarray(blank).save(tmp_path / "blank.png")

    suspicious = find_suspicious_renders(tmp_path)

    assert len(suspicious) == 1
    path, std = suspicious[0]
    assert path.name == "blank.png"
    assert std < 15.0


def test_find_suspicious_renders_ignores_normal_photo(tmp_path):
    rng = np.random.default_rng(0)
    normal = rng.integers(0, 256, size=(64, 64, 3), dtype=np.uint8)  # high variance, like a real photo
    Image.fromarray(normal).save(tmp_path / "normal.png")

    suspicious = find_suspicious_renders(tmp_path)

    assert suspicious == []


def test_find_suspicious_renders_sorts_most_suspicious_first(tmp_path):
    # Perfectly uniform (std=0) — the most suspicious case.
    totally_blank = np.full((32, 32, 3), 128, dtype=np.uint8)
    Image.fromarray(totally_blank).save(tmp_path / "z_totally_blank.png")

    # A faint gradient — some variation, but still well under the
    # threshold, so it's flagged but less suspicious than a pure blank.
    slightly_off = np.tile(np.linspace(120, 136, 32, dtype=np.uint8), (32, 3, 1)).transpose(0, 2, 1)
    Image.fromarray(slightly_off).save(tmp_path / "a_slightly_off.png")

    suspicious = find_suspicious_renders(tmp_path)

    assert [p.name for p, _ in suspicious] == ["z_totally_blank.png", "a_slightly_off.png"]
    assert suspicious[0][1] < suspicious[1][1]


def test_find_suspicious_renders_handles_empty_directory(tmp_path):
    assert find_suspicious_renders(tmp_path) == []
