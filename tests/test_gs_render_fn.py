import inspect

import numpy as np
import pytest
import torch

from src.rendering.gs_render_fn import (
    VramBudgetExceededError,
    _load_appearance_affine_bias,
    _load_gaussians,
    _parse_render_config,
    _placeholder_image,
    _run_inference,
    _tensor_to_uint8_image,
    _validate_peak_vram,
    real_render_fn,
)


def test_placeholder_image_has_requested_size():
    img = _placeholder_image(width=64, height=48)
    assert img.size == (64, 48)
    assert img.mode == "RGB"


def test_tensor_to_uint8_image_converts_chw_float_to_hwc_uint8():
    # 3x2x2 CHW tensor: red channel at max, others at 0.
    tensor = torch.zeros(3, 2, 2)
    tensor[0, :, :] = 1.0

    array = _tensor_to_uint8_image(tensor)

    assert array.shape == (2, 2, 3)
    assert array.dtype == np.uint8
    assert (array[:, :, 0] == 255).all()
    assert (array[:, :, 1] == 0).all()
    assert (array[:, :, 2] == 0).all()


def test_tensor_to_uint8_image_clamps_out_of_range_values():
    tensor = torch.full((3, 1, 1), 2.0)  # out of [0, 1] range
    array = _tensor_to_uint8_image(tensor)
    assert (array == 255).all()


def test_real_render_fn_accepts_optional_render_config():
    parameter = inspect.signature(real_render_fn).parameters["render_config"]
    assert parameter.default is None


def test_parse_render_config_preserves_backward_compatible_defaults():
    antialiasing, appearance_path, vram_budget = _parse_render_config(None)
    assert antialiasing is False
    assert appearance_path is None
    assert vram_budget is None


def test_parse_render_config_returns_variant_settings(tmp_path):
    appearance_path = tmp_path / "mean_appearance.pt"
    antialiasing, parsed_path, vram_budget = _parse_render_config(
        {
            "antialiasing": True,
            "appearance_path": appearance_path,
            "vram_budget_bytes": 1234,
        }
    )
    assert antialiasing is True
    assert parsed_path == appearance_path
    assert vram_budget == 1234


def test_missing_requested_appearance_artifact_fails_closed(tmp_path):
    with pytest.raises(FileNotFoundError, match="appearance artifact"):
        _load_appearance_affine_bias(tmp_path / "missing.pt")


def test_run_inference_disables_gradient_tracking():
    assert _run_inference(lambda: torch.is_grad_enabled()) is False


def test_validate_peak_vram_raises_descriptive_error():
    with pytest.raises(VramBudgetExceededError, match="1200.*1000"):
        _validate_peak_vram(peak_bytes=1200, budget_bytes=1000)

    _validate_peak_vram(peak_bytes=1000, budget_bytes=1000)


def test_load_gaussians_rejects_unsupported_checkpoint_before_cuda_work(tmp_path):
    checkpoint = tmp_path / "checkpoint.xyz"
    checkpoint.touch()

    with pytest.raises(ValueError, match="unsupported gaussian checkpoint format"):
        _load_gaussians(checkpoint)
