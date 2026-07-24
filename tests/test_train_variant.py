import inspect
import json
import random
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.common.config import SceneConfig
from src.training.train_variant import (
    ALL_TRAINING_VARIANTS,
    TrainingVariant,
    _apply_hyperparam_overrides,
    _atomic_torch_save,
    _build_variant_checkpoint_payload,
    _build_dataset_args,
    _checkpoint_schedule,
    _completed_variant_path,
    _find_latest_matching_checkpoint,
    _prepare_depth_regularization_inputs,
    _restore_variant_checkpoint,
    _seed_everything,
    _variant_run_fingerprint,
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


def test_variant_checkpoint_schedule_includes_intervals_and_final():
    assert _checkpoint_schedule(12_000, 5_000) == [5_000, 10_000, 12_000]
    assert _checkpoint_schedule(10_000, 5_000) == [5_000, 10_000]
    assert _checkpoint_schedule(200, 5_000) == [200]


def _fingerprint_scene(tmp_path):
    images = tmp_path / "images"
    sparse = tmp_path / "sparse" / "0"
    images.mkdir(parents=True)
    sparse.mkdir(parents=True)
    (images / "frame.jpg").write_bytes(b"pixels")
    for name in ("cameras.bin", "images.bin", "points3D.bin"):
        (sparse / name).write_bytes(name.encode())
    return SceneConfig(
        name="chair",
        root=tmp_path,
        train_images_dir=images,
        sparse_dir=sparse,
        test_poses_csv=tmp_path / "test.csv",
    )


def test_variant_fingerprint_changes_with_seed_variant_and_overrides(tmp_path):
    scene = _fingerprint_scene(tmp_path)
    baseline = ALL_TRAINING_VARIANTS[0]
    depth = ALL_TRAINING_VARIANTS[1]
    original = _variant_run_fingerprint(
        scene, baseline, 100, {"densify_grad_threshold": 0.001}, 7,
    )

    assert original != _variant_run_fingerprint(
        scene, baseline, 100, {"densify_grad_threshold": 0.001}, 8,
    )
    assert original != _variant_run_fingerprint(
        scene, depth, 100, {"densify_grad_threshold": 0.001}, 7,
    )
    assert original != _variant_run_fingerprint(
        scene, baseline, 100, {"densify_grad_threshold": 0.002}, 7,
    )


def test_atomic_torch_save_replaces_target_without_leaving_temp(tmp_path):
    target = tmp_path / "variant_chkpnt5.pth"

    _atomic_torch_save({"iteration": 5}, target)

    assert torch.load(target, weights_only=False)["iteration"] == 5
    assert not list(tmp_path.glob("*.tmp"))


def test_find_latest_matching_checkpoint_ignores_stale_fingerprint(tmp_path):
    _atomic_torch_save(
        {"iteration": 5, "fingerprint": "matching"},
        tmp_path / "variant_chkpnt5.pth",
    )
    _atomic_torch_save(
        {"iteration": 10, "fingerprint": "stale"},
        tmp_path / "variant_chkpnt10.pth",
    )

    path, payload = _find_latest_matching_checkpoint(tmp_path, "matching")

    assert path.name == "variant_chkpnt5.pth"
    assert payload["iteration"] == 5


def test_completed_variant_requires_matching_manifest_and_appearance(tmp_path):
    final_ply = tmp_path / "point_cloud" / "iteration_10" / "point_cloud.ply"
    final_ply.parent.mkdir(parents=True)
    final_ply.write_bytes(b"ply")
    (tmp_path / "variant_run.json").write_text(json.dumps({
        "fingerprint": "expected",
        "iteration": 10,
        "final_checkpoint_path": str(final_ply),
    }))

    assert _completed_variant_path(
        tmp_path, "expected", needs_appearance=False,
    ) == final_ply
    assert _completed_variant_path(
        tmp_path, "other", needs_appearance=False,
    ) is None
    assert _completed_variant_path(
        tmp_path, "expected", needs_appearance=True,
    ) is None

    torch.save(
        {"affine": torch.eye(3), "bias": torch.zeros(3)},
        tmp_path / "mean_appearance.pt",
    )
    assert _completed_variant_path(
        tmp_path, "expected", needs_appearance=True,
    ) == final_ply


def test_variant_checkpoint_payload_contains_model_appearance_and_rng(monkeypatch):
    class _Stateful:
        def __init__(self, state):
            self.state = state

        def state_dict(self):
            return self.state

    gaussians = SimpleNamespace(capture=lambda: ("gaussian-state",))
    appearance = _Stateful({"affine": torch.tensor([1.0])})
    optimizer = _Stateful({"step": 7})
    monkeypatch.setattr(
        "src.training.train_variant._capture_rng_state",
        lambda: {"python": "rng"},
    )

    payload = _build_variant_checkpoint_payload(
        gaussians,
        appearance,
        optimizer,
        iteration=5000,
        fingerprint="abc",
    )

    assert payload["iteration"] == 5000
    assert payload["fingerprint"] == "abc"
    assert payload["gaussians"] == ("gaussian-state",)
    assert payload["appearance"] == {"affine": torch.tensor([1.0])}
    assert payload["appearance_optimizer"] == {"step": 7}
    assert payload["rng"] == {"python": "rng"}


def test_restore_variant_checkpoint_restores_all_available_state(monkeypatch):
    calls = []

    class _Restorable:
        def restore(self, state, opt):
            calls.append(("gaussians", state, opt))

        def load_state_dict(self, state):
            calls.append(("state", state))

    monkeypatch.setattr(
        "src.training.train_variant._restore_rng_state",
        lambda state: calls.append(("rng", state)),
    )
    opt = object()
    payload = {
        "gaussians": "gaussian-state",
        "appearance": {"affine": 1},
        "appearance_optimizer": {"step": 7},
        "rng": {"python": "rng"},
    }

    _restore_variant_checkpoint(
        payload,
        _Restorable(),
        opt,
        _Restorable(),
        _Restorable(),
    )

    assert ("gaussians", "gaussian-state", opt) in calls
    assert ("state", {"affine": 1}) in calls
    assert ("state", {"step": 7}) in calls
    assert ("rng", {"python": "rng"}) in calls


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
