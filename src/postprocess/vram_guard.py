from __future__ import annotations

from pathlib import Path


def count_gaussians_in_ply(ply_path: Path) -> int:
    """Read only the ASCII PLY header to get the vertex count, without
    loading the (potentially multi-GB) binary body.
    """
    with open(ply_path, "rb") as f:
        for raw_line in f:
            line = raw_line.decode("ascii", errors="ignore").strip()
            if line.startswith("element vertex"):
                return int(line.split()[-1])
            if line == "end_header":
                break
    raise ValueError(f"no 'element vertex' line found in PLY header: {ply_path}")


def estimate_vram_bytes(num_gaussians: int, sh_degree: int = 3, dtype_bytes: int = 4) -> int:
    """Conservative heuristic estimate of VRAM needed to RENDER (not train)
    a checkpoint with this many Gaussians at the given SH degree.

    Per-Gaussian float count: position(3) + rotation quaternion(4) +
    scale(3) + opacity(1) + spherical harmonics coefficients
    ((sh_degree+1)^2 * 3 channels, including the DC term). A 2x multiplier
    accounts for the CUDA rasterizer's tile-based intermediate buffers,
    which is a rough approximation, not a guarantee -- always confirm with
    a real torch.cuda.max_memory_allocated() measurement on Colab before
    trusting this near the A4000's 20GB ceiling.
    """
    floats_per_gaussian = 3 + 4 + 3 + 1 + (sh_degree + 1) ** 2 * 3
    rendering_overhead_multiplier = 2.0
    return int(num_gaussians * floats_per_gaussian * dtype_bytes * rendering_overhead_multiplier)


def fits_within_vram_budget(ply_path: Path, budget_bytes: int, sh_degree: int = 3) -> bool:
    num_gaussians = count_gaussians_in_ply(ply_path)
    return estimate_vram_bytes(num_gaussians, sh_degree=sh_degree) <= budget_bytes
