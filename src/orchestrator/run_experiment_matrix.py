from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from src.common.colmap_io import compute_scene_bbox, load_sparse_scene
from src.common.config import SceneConfig
from src.common.pose_utils import (
    camera_extrinsics_from_colmap,
    camera_focal_lengths,
    focal2fov,
    qvec2rotmat,
)
from src.data_validation.validate_scene import validate_scene
from src.evaluation.compute_metrics import combine_score, compute_pair_metrics
from src.evaluation.make_holdout_split import select_holdout_images
from src.evaluation.screening import (
    variants_needing_full_iteration_verification,
)
from src.evaluation.select_best_config import select_best_candidate
from src.postprocess.vram_guard import (
    count_gaussians_in_ply,
    estimate_vram_bytes,
)
from src.rendering.gs_render_fn import VramBudgetExceededError
from src.rendering.render_from_csv import CameraParams, load_test_poses_csv
from src.submission.package_submission import package_submission
from src.submission.validate_submission import validate_submission
from src.training.holdout_scene import build_filtered_scene
from src.training.train_variant import ALL_TRAINING_VARIANTS
from src.training.undistort_scene import undistort_scene


@dataclass
class ExperimentPipelineResult:
    per_scene_scores: dict[str, float] = field(default_factory=dict)
    skipped_scenes: dict[str, list[str]] = field(default_factory=dict)
    chosen_config: dict[str, dict] = field(default_factory=dict)
    all_candidates: dict[str, list[dict]] = field(default_factory=dict)
    validation_problems: list[str] = field(default_factory=list)
    submission_zip: Path | None = None


def _camera_params_for_holdout(
    sparse,
    holdout_names,
    image_dims,
) -> list[CameraParams]:
    width, height = image_dims
    params = []
    for image in sparse.images.values():
        if image.name not in holdout_names:
            continue

        camera = sparse.cameras[image.camera_id]
        fx, fy = camera_focal_lengths(camera.model, camera.params)
        rotation, translation = camera_extrinsics_from_colmap(
            *image.qvec,
            *image.tvec,
        )
        params.append(
            CameraParams(
                image_name=image.name,
                R=rotation,
                T=translation,
                fov_x=focal2fov(fx, width),
                fov_y=focal2fov(fy, height),
                width=width,
                height=height,
            )
        )
    return params


def _render_config_for(variant, train_output_dir: Path) -> dict:
    appearance_path = Path(train_output_dir) / "mean_appearance.pt"
    return {
        "antialiasing": variant.use_anti_alias,
        "appearance_path": (
            appearance_path
            if variant.use_appearance_embed
            else None
        ),
    }


def _score_checkpoint(
    checkpoint,
    holdout_params,
    render_fn,
    render_dir,
    gt_dir,
    lpips_model,
    psnr_max,
    render_config,
):
    rendered_paths = render_fn(
        checkpoint,
        holdout_params,
        render_dir,
        render_config=render_config,
    )
    scores = []
    for path, params in zip(rendered_paths, holdout_params):
        gt_path = Path(gt_dir) / params.image_name
        with Image.open(path) as pred_image:
            pred = np.array(pred_image.convert("RGB"))
        with Image.open(gt_path) as gt_image:
            gt = np.array(
                gt_image.convert("RGB").resize(pred.shape[1::-1])
            )
        metrics = compute_pair_metrics(pred, gt, lpips_model)
        scores.append(
            combine_score(
                metrics["lpips"],
                metrics["ssim"],
                metrics["psnr"],
                psnr_max,
            )
        )
    return float(np.mean(scores)) if scores else 0.0


def _candidate_vram_bytes(checkpoint: Path) -> int:
    return estimate_vram_bytes(count_gaussians_in_ply(checkpoint))


def _ensure_checkpoint_fits_budget(
    checkpoint: Path,
    budget_bytes: int,
) -> int:
    estimated_bytes = _candidate_vram_bytes(checkpoint)
    if estimated_bytes > budget_bytes:
        raise VramBudgetExceededError(
            f"final checkpoint {checkpoint} requires an estimated "
            f"{estimated_bytes} bytes, exceeding budget {budget_bytes} bytes"
        )
    return estimated_bytes


def run_experiment_matrix_pipeline(
    scenes: list[SceneConfig],
    screening_train_fn,
    final_train_fn,
    render_fn,
    prune_fn,
    lpips_model,
    psnr_max: float,
    vram_budget_bytes: int,
    output_root: Path,
    extra_candidates_by_scene: dict[str, list[dict]] | None = None,
    tiebreak_threshold: float = 0.01,
    training_seed: int = 0,
    variants: list | None = None,
) -> ExperimentPipelineResult:
    output_root = Path(output_root)
    result = ExperimentPipelineResult()
    scene_render_dirs = {}
    extra_candidates_by_scene = extra_candidates_by_scene or {}
    # Screening always includes "baseline" -- it is the control every other
    # variant is compared against; dropping it would make the comparison
    # meaningless, not just cheaper.
    variants = list(variants) if variants is not None else ALL_TRAINING_VARIANTS
    if not any(variant.name == "baseline" for variant in variants):
        raise ValueError("variants must include the 'baseline' control variant")

    for scene in scenes:
        report = validate_scene(scene)
        if report.problems:
            result.skipped_scenes[scene.name] = report.problems
            continue

        scene_output = output_root / scene.name
        working_scene = undistort_scene(
            scene,
            scene_output / "undistorted",
        )

        sparse = load_sparse_scene(working_scene.sparse_dir)
        bbox_min, bbox_max = compute_scene_bbox(
            sparse.points3d,
            margin_ratio=0.1,
        )
        file_backed_names = {
            path.name
            for path in working_scene.train_images_dir.iterdir()
            if path.is_file()
        }
        camera_centers = {
            image.name: (
                -np.transpose(qvec2rotmat(np.array(image.qvec)))
                @ np.array(image.tvec)
            )
            for image in sparse.images.values()
            if image.name in file_backed_names
        }
        holdout_names = set(
            select_holdout_images(
                camera_centers,
                holdout_ratio=0.125,
            )
        )
        filtered_scene = build_filtered_scene(
            working_scene,
            holdout_names,
            scene_output / "filtered_scene",
        )

        sample_image = next(
            path
            for path in working_scene.train_images_dir.iterdir()
            if path.is_file()
        )
        with Image.open(sample_image) as image:
            image_dims = image.size
        holdout_params = _camera_params_for_holdout(
            sparse,
            holdout_names,
            image_dims,
        )

        candidates = []
        for variant in variants:
            train_output_dir = scene_output / f"eval_{variant.name}"
            eval_checkpoint = screening_train_fn(
                filtered_scene,
                variant,
                train_output_dir,
                seed=training_seed,
            )
            render_config = _render_config_for(
                variant,
                train_output_dir,
            )
            for use_floater_cleanup in (False, True):
                checkpoint = (
                    prune_fn(eval_checkpoint, bbox_min, bbox_max)
                    if use_floater_cleanup
                    else eval_checkpoint
                )
                score = _score_checkpoint(
                    checkpoint,
                    holdout_params,
                    render_fn,
                    scene_output
                    / f"holdout_{variant.name}_{use_floater_cleanup}",
                    working_scene.train_images_dir,
                    lpips_model,
                    psnr_max,
                    render_config,
                )
                candidates.append(
                    {
                        "variant": variant.name,
                        "floater_cleanup": use_floater_cleanup,
                        "score": score,
                        "estimated_vram_bytes": _candidate_vram_bytes(
                            checkpoint
                        ),
                        "checkpoint_path": str(checkpoint),
                        "seed": training_seed,
                    }
                )

        best_per_variant = {}
        for candidate in candidates:
            variant_name = candidate["variant"]
            if (
                variant_name not in best_per_variant
                or candidate["score"] > best_per_variant[variant_name]
            ):
                best_per_variant[variant_name] = candidate["score"]

        variants_to_rerun = (
            variants_needing_full_iteration_verification(
                [
                    {"variant": variant, "score": score}
                    for variant, score in best_per_variant.items()
                ],
                threshold=tiebreak_threshold,
            )
        )
        for variant_name in variants_to_rerun:
            variant = next(
                item
                for item in ALL_TRAINING_VARIANTS
                if item.name == variant_name
            )
            tiebreak_output_dir = (
                scene_output / f"tiebreak_{variant.name}"
            )
            full_checkpoint = final_train_fn(
                filtered_scene,
                variant,
                tiebreak_output_dir,
                seed=training_seed,
            )
            render_config = _render_config_for(
                variant,
                tiebreak_output_dir,
            )
            for candidate in candidates:
                if candidate["variant"] != variant_name:
                    continue

                checkpoint = (
                    prune_fn(full_checkpoint, bbox_min, bbox_max)
                    if candidate["floater_cleanup"]
                    else full_checkpoint
                )
                candidate["score"] = _score_checkpoint(
                    checkpoint,
                    holdout_params,
                    render_fn,
                    scene_output
                    / (
                        f"tiebreak_holdout_{variant_name}_"
                        f"{candidate['floater_cleanup']}"
                    ),
                    working_scene.train_images_dir,
                    lpips_model,
                    psnr_max,
                    render_config,
                )
                candidate["estimated_vram_bytes"] = (
                    _candidate_vram_bytes(checkpoint)
                )
                candidate["checkpoint_path"] = str(checkpoint)

        for extra in extra_candidates_by_scene.get(scene.name, []):
            variant = next(
                item
                for item in ALL_TRAINING_VARIANTS
                if item.name == extra["variant"]
            )
            overrides = {
                key: value
                for key, value in extra.items()
                if key
                not in (
                    "variant",
                    "floater_cleanup",
                    "candidate_name",
                )
            }
            extra_output_dir = (
                scene_output / f"extra_{extra['candidate_name']}"
            )
            checkpoint = final_train_fn(
                filtered_scene,
                variant,
                extra_output_dir,
                hyperparam_overrides=overrides,
                seed=training_seed,
            )
            render_config = _render_config_for(
                variant,
                extra_output_dir,
            )
            use_floater_cleanup = bool(
                extra.get("floater_cleanup", False)
            )
            if use_floater_cleanup:
                checkpoint = prune_fn(
                    checkpoint,
                    bbox_min,
                    bbox_max,
                )
            score = _score_checkpoint(
                checkpoint,
                holdout_params,
                render_fn,
                scene_output
                / f"extra_holdout_{extra['candidate_name']}",
                working_scene.train_images_dir,
                lpips_model,
                psnr_max,
                render_config,
            )
            candidates.append(
                {
                    "variant": extra["variant"],
                    "floater_cleanup": use_floater_cleanup,
                    "candidate_name": extra["candidate_name"],
                    "score": score,
                    "estimated_vram_bytes": _candidate_vram_bytes(
                        checkpoint
                    ),
                    "checkpoint_path": str(checkpoint),
                    "hyperparam_overrides": overrides,
                    "seed": training_seed,
                }
            )

        result.all_candidates[scene.name] = candidates
        winner = select_best_candidate(
            candidates,
            vram_budget_bytes,
        )
        chosen_config = {
            **winner,
            "selection_checkpoint_path": winner["checkpoint_path"],
            "seed": training_seed,
        }
        result.chosen_config[scene.name] = chosen_config
        result.per_scene_scores[scene.name] = winner["score"]

        winning_variant = next(
            variant
            for variant in ALL_TRAINING_VARIANTS
            if variant.name == winner["variant"]
        )
        full_training_scene = build_filtered_scene(
            working_scene,
            set(),
            scene_output / "full_scene",
        )
        final_output_dir = scene_output / "final_train"
        final_checkpoint = final_train_fn(
            full_training_scene,
            winning_variant,
            final_output_dir,
            hyperparam_overrides=winner.get(
                "hyperparam_overrides"
            ),
            seed=training_seed,
        )
        final_render_config = _render_config_for(
            winning_variant,
            final_output_dir,
        )
        if winner["floater_cleanup"]:
            final_checkpoint = prune_fn(
                final_checkpoint,
                bbox_min,
                bbox_max,
            )
        final_metadata = {
            "final_checkpoint_path": str(final_checkpoint),
        }
        winner.update(final_metadata)
        chosen_config.update(final_metadata)
        try:
            final_estimated_vram = _ensure_checkpoint_fits_budget(
                final_checkpoint,
                vram_budget_bytes,
            )
        except VramBudgetExceededError as error:
            estimated_bytes = _candidate_vram_bytes(final_checkpoint)
            winner["final_estimated_vram_bytes"] = estimated_bytes
            chosen_config["final_estimated_vram_bytes"] = estimated_bytes
            result.validation_problems.append(
                f"scene '{scene.name}': {error}"
            )
            continue
        winner["final_estimated_vram_bytes"] = final_estimated_vram
        chosen_config["final_estimated_vram_bytes"] = final_estimated_vram
        final_render_config["vram_budget_bytes"] = vram_budget_bytes
        winner["final_render_config"] = final_render_config
        chosen_config["final_render_config"] = final_render_config

        test_render_dir = scene_output / "test_render"
        test_params_list = load_test_poses_csv(
            scene.test_poses_csv
        )
        try:
            render_fn(
                final_checkpoint,
                test_params_list,
                test_render_dir,
                render_config=final_render_config,
            )
        except VramBudgetExceededError as error:
            result.validation_problems.append(
                f"scene '{scene.name}': {error}"
            )
            continue
        measured_peak = final_render_config.get(
            "measured_peak_vram_bytes"
        )
        winner["final_measured_peak_vram_bytes"] = measured_peak
        chosen_config["final_measured_peak_vram_bytes"] = measured_peak
        scene_render_dirs[
            scene.effective_submission_dir
        ] = test_render_dir

    if result.skipped_scenes:
        result.validation_problems.extend(
            f"scene '{name}' skipped, no submission produced: {problems}"
            for name, problems in result.skipped_scenes.items()
        )
    if result.skipped_scenes or result.validation_problems:
        result.submission_zip = None
        return result

    submission_zip = output_root / "submission.zip"
    package_submission(scene_render_dirs, submission_zip)
    result.validation_problems = validate_submission(
        submission_zip,
        scenes,
    )
    result.submission_zip = (
        submission_zip
        if not result.validation_problems
        else None
    )
    return result
