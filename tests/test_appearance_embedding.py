import torch

from src.training.appearance_embedding import AppearanceEmbedding, apply_appearance


def test_identity_affine_and_zero_bias_is_a_no_op():
    rgb = torch.rand(3, 4, 4)
    affine = torch.eye(3)
    bias = torch.zeros(3)
    out = apply_appearance(rgb, affine, bias)
    torch.testing.assert_close(out, rgb)


def test_bias_shifts_every_pixel():
    rgb = torch.zeros(3, 2, 2)
    affine = torch.eye(3)
    bias = torch.tensor([0.1, 0.0, 0.0])
    out = apply_appearance(rgb, affine, bias)
    assert torch.allclose(out[0], torch.full((2, 2), 0.1))
    assert torch.allclose(out[1], torch.zeros(2, 2))


def test_output_is_clamped_to_valid_range():
    rgb = torch.ones(3, 2, 2)
    affine = torch.eye(3) * 2.0
    bias = torch.zeros(3)
    out = apply_appearance(rgb, affine, bias)
    assert out.max() <= 1.0
    assert out.min() >= 0.0


def test_appearance_embedding_module_initializes_to_identity_no_op():
    module = AppearanceEmbedding(num_images=3)
    rgb = torch.rand(3, 4, 4)
    for image_idx in range(3):
        affine, bias = module(image_idx)
        out = apply_appearance(rgb, affine, bias)
        torch.testing.assert_close(out, rgb, atol=1e-6, rtol=1e-6)


def test_appearance_embedding_has_learnable_parameters_per_image():
    module = AppearanceEmbedding(num_images=5)
    params = list(module.parameters())
    assert len(params) > 0
    assert all(p.requires_grad for p in params)
