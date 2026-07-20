import numpy as np
import torch

from src.rendering.gs_render_fn import _placeholder_image, _tensor_to_uint8_image


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
