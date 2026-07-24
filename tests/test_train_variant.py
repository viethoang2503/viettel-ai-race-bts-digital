import inspect
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.training.train_variant import (
    ALL_TRAINING_VARIANTS,
    TrainingVariant,
    _apply_hyperparam_overrides,
    _build_dataset_args,
    _prepare_depth_regularization_inputs,
    _seed_everything,
    _validate_training_request,
    run_training_variant,
)


def test_all_training_variants_have_unique_names():
    names = [variant.name for variant in ALL_TRAINING_VARIANTS]
    assert len(names) == len(set(names))
    assert set(names) == {
        "baseline",
        "depth_reg",
        "anti_alias",
        "appearance_embed",
        "full_stack",
    }


def test_baseline_variant_has_no_techniques_enabled():
    baseline = next(
        variant for variant in ALL_TRAINING_VARIANTS
        if variant.name == "baseline"
    )
    assert baseline.use_depth_reg is False
    assert baseline.use_anti_alias is False
    assert baseline.use_appearance_embed is False


def test_full_stack_variant_has_all_techniques_enabled():
    full_stack = next(
        variant for variant in ALL_TRAINING_VARIANTS
        if variant.name == "full_stack"
    )
    assert full_stack.use_depth_reg is True
    assert full_stack.use_anti_alias is True
    assert full_stack.use_appearance_embed is True


def test_run_training_variant_signature_accepts_hyperparam_overrides():
    params = inspect.signature(run_training_variant).parameters
    assert "hyperparam_overrides" in params
    assert params["hyperparam_overrides"].default is None
    assert params["seed"].default == 0
    assert params["checkpoint_interval"].default == 5000


def test_seed_everything_repeats_python_numpy_and_torch_streams():
    _seed_everything(123)
    first = (
        random.random(),
        np.random.random(),
        torch.rand(3),
    )

    _seed_everything(123)
    second = (
        random.random(),
        np.random.random(),
        torch.rand(3),
    )

    assert first[0] == second[0]
    assert first[1] == second[1]
    torch.testing.assert_close(first[2], second[2])


@pytest.mark.parametrize(
    ("iterations", "seed", "checkpoint_interval", "message"),
    [
        (0, 0, 5000, "iterations"),
        (10, -1, 5000, "seed"),
        (10, 0, 0, "checkpoint_interval"),
    ],
)
def test_validate_training_request_rejects_invalid_values(
    iterations,
    seed,
    checkpoint_interval,
    message,
):
    with pytest.raises(ValueError, match=message):
        _validate_training_request(iterations, seed, checkpoint_interval)


def test_build_dataset_args_preserves_full_resolution_and_safe_densification(
    tmp_path,
):
    dataset, pipe, opt = _build_dataset_args(
        tmp_path / "source",
        tmp_path / "model",
        use_anti_alias=True,
    )

    assert dataset.resolution == 1
    assert pipe.antialiasing is True
    assert opt.densify_grad_threshold == pytest.approx(0.001)
    assert opt.densify_until_iter == 10_000


def test_apply_hyperparam_overrides_controls_iterations_and_known_fields():
    opt = SimpleNamespace(
        iterations=30_000,
        densify_grad_threshold=0.001,
    )

    effective_iterations = _apply_hyperparam_overrides(
        opt,
        iterations=200,
        hyperparam_overrides={
            "iterations": 50,
            "densify_grad_threshold": 0.0005,
        },
    )

    assert effective_iterations == 50
    assert opt.iterations == 50
    assert opt.densify_grad_threshold == pytest.approx(0.0005)


def test_apply_hyperparam_overrides_rejects_unknown_field():
    opt = SimpleNamespace(iterations=30_000)

    with pytest.raises(
        ValueError,
        match="unknown training hyperparameter override",
    ):
        _apply_hyperparam_overrides(
            opt,
            iterations=200,
            hyperparam_overrides={"not_a_real_field": 1},
        )


def test_prepare_depth_inputs_squeezes_channel_and_converts_z_to_inverse_depth():
    rendered_inverse_depth = torch.ones((1, 2, 3))
    pixel_xy = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 1.0]])
    sparse_depths = np.array([2.0, 4.0, -1.0])

    depth_map, valid_pixels, sparse_inverse_depths = (
        _prepare_depth_regularization_inputs(
            rendered_inverse_depth,
            pixel_xy,
            sparse_depths,
        )
    )

    assert depth_map.shape == (2, 3)
    np.testing.assert_allclose(valid_pixels, pixel_xy[:2])
    np.testing.assert_allclose(sparse_inverse_depths, [0.5, 0.25])
