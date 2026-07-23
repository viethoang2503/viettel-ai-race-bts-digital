from pathlib import Path

import pytest

from src.postprocess.vram_guard import (
    count_gaussians_in_ply,
    estimate_vram_bytes,
    fits_within_vram_budget,
)


def _write_fake_ply(path: Path, num_vertices: int) -> None:
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {num_vertices}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "end_header\n"
    )
    path.write_bytes(header.encode("ascii") + b"\x00" * (num_vertices * 12))


def test_count_gaussians_in_ply_reads_header_only(tmp_path):
    ply_path = tmp_path / "cloud.ply"
    _write_fake_ply(ply_path, 12345)
    assert count_gaussians_in_ply(ply_path) == 12345


def test_estimate_vram_bytes_scales_linearly_with_count():
    small = estimate_vram_bytes(1000)
    large = estimate_vram_bytes(2000)
    assert large == pytest.approx(2 * small, rel=1e-6)
    assert small > 0


def test_estimate_vram_bytes_increases_with_sh_degree():
    low_degree = estimate_vram_bytes(1000, sh_degree=0)
    high_degree = estimate_vram_bytes(1000, sh_degree=3)
    assert high_degree > low_degree


def test_fits_within_vram_budget_true_for_small_cloud(tmp_path):
    ply_path = tmp_path / "small.ply"
    _write_fake_ply(ply_path, 1000)
    assert fits_within_vram_budget(ply_path, budget_bytes=16 * 1024**3) is True


def test_fits_within_vram_budget_false_for_absurdly_large_cloud(tmp_path):
    ply_path = tmp_path / "huge.ply"
    _write_fake_ply(ply_path, 1)  # header claims 1, we lie about the count instead:
    ply_path.write_bytes(
        ply_path.read_bytes().replace(b"element vertex 1\n", b"element vertex 500000000\n")
    )
    assert fits_within_vram_budget(ply_path, budget_bytes=16 * 1024**3) is False
