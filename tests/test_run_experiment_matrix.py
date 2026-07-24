from pathlib import Path

import numpy as np
import torch
from PIL import Image

from src.common.config import SceneConfig
from src.orchestrator.run_experiment_matrix import (
    run_experiment_matrix_pipeline,
)


def _chair_scene():
    return SceneConfig(
        name="chair",
        root=Path("VAI_NVS_DATA_ROUND2/chair"),
        train_images_dir=Path("VAI_NVS_DATA_ROUND2/chair/train/images"),
        sparse_dir=Path("VAI_NVS_DATA_ROUND2/chair/train/sparse/0"),
        test_poses_csv=Path("VAI_NVS_DATA_ROUND2/chair/test/test_poses.csv"),
        submission_dir="chair",
    )


class _StubLpipsModel:
    def __call__(self, pred_tensor, gt_tensor):
        diff = torch.mean(torch.abs(pred_tensor - gt_tensor)).item()
        return torch.tensor(diff)


def _write_fake_ply(path: Path) -> None:
    path.write_bytes(
        b"ply\nformat binary_little_endian 1.0\nelement vertex 10\n"
        b"property float x\nproperty float y\nproperty float z\nend_header\n"
        + b"\x00" * 120
    )


def test_run_experiment_matrix_screens_all_variants_and_uses_full_iterations_for_winner(
    tmp_path,
):
    scene = _chair_scene()
    screening_calls = []
    final_calls = []
    render_calls = []

    def fake_screening_train_fn(scene_arg, variant, output_dir):
        screening_calls.append(variant.name)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        ply = output_dir / "point_cloud.ply"
        _write_fake_ply(ply)
        return ply

    def fake_final_train_fn(
        scene_arg,
        variant,
        output_dir,
        hyperparam_overrides=None,
    ):
        final_calls.append(
            (variant.name, str(output_dir), hyperparam_overrides)
        )
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        ply = output_dir / "point_cloud.ply"
        _write_fake_ply(ply)
        return ply

    def fake_render_fn(
        checkpoint,
        params_list,
        output_dir,
        render_config=None,
    ):
        render_calls.append((str(checkpoint), render_config))
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        fill = hash(str(checkpoint)) % 200
        written = []
        for params in params_list:
            path = output_dir / params.image_name
            Image.fromarray(
                np.full(
                    (params.height, params.width, 3),
                    fill,
                    dtype=np.uint8,
                )
            ).save(path)
            written.append(path)
        return written

    def fake_prune_fn(checkpoint_path, bbox_min, bbox_max):
        pruned = Path(checkpoint_path).with_name(
            "point_cloud_pruned.ply"
        )
        pruned.write_bytes(Path(checkpoint_path).read_bytes())
        return pruned

    extra_candidates = {
        "chair": [
            {
                "variant": "baseline",
                "candidate_name": "chair_extra_0",
                "densify_grad_threshold": 0.002,
            },
            {
                "variant": "baseline",
                "candidate_name": "chair_extra_1",
                "iterations": 45_000,
            },
        ],
    }

    result = run_experiment_matrix_pipeline(
        scenes=[scene],
        screening_train_fn=fake_screening_train_fn,
        final_train_fn=fake_final_train_fn,
        render_fn=fake_render_fn,
        prune_fn=fake_prune_fn,
        lpips_model=_StubLpipsModel(),
        psnr_max=30.0,
        vram_budget_bytes=16 * 1024**3,
        output_root=tmp_path,
        extra_candidates_by_scene=extra_candidates,
    )

    assert set(screening_calls) == {
        "baseline",
        "depth_reg",
        "anti_alias",
        "appearance_embed",
        "full_stack",
    }
    assert len(result.all_candidates["chair"]) == 12
    assert any(
        candidate.get("candidate_name") == "chair_extra_0"
        for candidate in result.all_candidates["chair"]
    )
    assert any("final_train" in call[1] for call in final_calls)

    anti_alias_render_configs = [
        config
        for checkpoint, config in render_calls
        if "eval_anti_alias" in checkpoint
    ]
    assert anti_alias_render_configs
    assert all(
        config is not None and config["antialiasing"] is True
        for config in anti_alias_render_configs
    )
    baseline_render_configs = [
        config
        for checkpoint, config in render_calls
        if "eval_baseline" in checkpoint
    ]
    assert all(
        config is not None and config["antialiasing"] is False
        for config in baseline_render_configs
    )

    assert "chair" in result.chosen_config
    assert result.submission_zip is not None
    assert result.submission_zip.exists()


def test_run_experiment_matrix_fails_closed_when_a_scene_is_skipped(
    tmp_path,
):
    good_scene = _chair_scene()
    bad_scene = SceneConfig(
        name="does_not_exist",
        root=Path("VAI_NVS_DATA_ROUND2/does_not_exist"),
        train_images_dir=Path(
            "VAI_NVS_DATA_ROUND2/does_not_exist/train/images"
        ),
        sparse_dir=Path(
            "VAI_NVS_DATA_ROUND2/does_not_exist/train/sparse/0"
        ),
        test_poses_csv=Path(
            "VAI_NVS_DATA_ROUND2/does_not_exist/test/test_poses.csv"
        ),
        submission_dir="does_not_exist",
    )

    def fake_train_fn(
        scene_arg,
        variant,
        output_dir,
        hyperparam_overrides=None,
    ):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        ply = output_dir / "point_cloud.ply"
        _write_fake_ply(ply)
        return ply

    def fake_render_fn(
        checkpoint,
        params_list,
        output_dir,
        render_config=None,
    ):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        written = []
        for params in params_list:
            path = output_dir / params.image_name
            Image.fromarray(
                np.zeros(
                    (params.height, params.width, 3),
                    dtype=np.uint8,
                )
            ).save(path)
            written.append(path)
        return written

    def fake_prune_fn(checkpoint_path, bbox_min, bbox_max):
        pruned = Path(checkpoint_path).with_name(
            "point_cloud_pruned.ply"
        )
        pruned.write_bytes(Path(checkpoint_path).read_bytes())
        return pruned

    result = run_experiment_matrix_pipeline(
        scenes=[good_scene, bad_scene],
        screening_train_fn=fake_train_fn,
        final_train_fn=fake_train_fn,
        render_fn=fake_render_fn,
        prune_fn=fake_prune_fn,
        lpips_model=_StubLpipsModel(),
        psnr_max=30.0,
        vram_budget_bytes=16 * 1024**3,
        output_root=tmp_path,
    )

    assert "does_not_exist" in result.skipped_scenes
    assert "chair" in result.chosen_config
    assert result.submission_zip is None
