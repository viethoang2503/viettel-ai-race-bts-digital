from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from src.common.colmap_io import load_sparse_scene
from src.common.config import SceneConfig
from src.common.pose_utils import camera_extrinsics_from_colmap, camera_focal_lengths, focal2fov, qvec2rotmat
from src.data_validation.validate_scene import validate_scene
from src.evaluation.compute_metrics import combine_score, compute_pair_metrics
from src.evaluation.make_holdout_split import select_holdout_images
from src.rendering.render_from_csv import CameraParams, load_test_poses_csv
from src.submission.package_submission import package_submission
from src.submission.validate_submission import validate_submission
from src.training.holdout_scene import build_filtered_scene
from src.training.undistort_scene import undistort_scene


@dataclass
class PipelineResult:
    per_scene_scores: dict[str, float] = field(default_factory=dict)
    skipped_scenes: dict[str, list[str]] = field(default_factory=dict)
    validation_problems: list[str] = field(default_factory=list)
    submission_zip: Path | None = None


def _camera_params_for_holdout(sparse, holdout_names, image_dims) -> list[CameraParams]:
    width, height = image_dims
    params = []
    id_to_camera = sparse.cameras
    for img in sparse.images.values():
        if img.name not in holdout_names:
            continue
        camera = id_to_camera[img.camera_id]
        fx, fy = camera_focal_lengths(camera.model, camera.params)
        r, t = camera_extrinsics_from_colmap(*img.qvec, *img.tvec)
        params.append(CameraParams(
            image_name=img.name, R=r, T=t,
            fov_x=focal2fov(fx, width), fov_y=focal2fov(fy, height),
            width=width, height=height,
        ))
    return params


def run_baseline_pipeline(
    scenes: list[SceneConfig], train_fn, render_fn, lpips_model, psnr_max: float,
    output_root: Path,
) -> PipelineResult:
    """Two training runs per scene:

    1. Eval training: on a build_filtered_scene copy with holdout images
       physically excluded (Task 8b) -> eval_checkpoint. Used only to score
       holdout images the model never trained on (no leakage).
    2. Final training: on a build_filtered_scene copy with an EMPTY holdout
       set -> final_checkpoint. This still goes through Task 8b, not the
       raw `scene` — see the note below on why the raw scene is never a
       valid training input for this dataset.

    `lpips_model` is always injected by the caller (real callers pass
    `load_lpips_model()` from Task 6; tests pass a network-free stub) so
    this function never implicitly requires network access.

    Each scene is validated (Task 5) before any GPU work — an invalid
    scene is recorded in `result.skipped_scenes` and both training phases
    are skipped for it entirely, so a broken scene never wastes Colab time.

    IMPORTANT — never pass the raw `scene` object to `train_fn` directly.
    `images.bin` always registers more cameras than are distributed as
    files (verified against the real dataset — see Task 4/5/8b), and the
    vendored loader crashes on `Image.open()` for any registered image
    with no file. `build_filtered_scene` is what makes a scene safe to
    train on; Phase A gets this "for free" via the holdout filter, so
    Phase B must call it too, with `holdout_names=set()`, purely to strip
    the registered-without-file images before training on 100% of the
    real, distributed training data.
    """
    output_root = Path(output_root)
    result = PipelineResult()
    scene_render_dirs = {}

    for scene in scenes:
        scene_output = output_root / scene.name
        submission_dir = scene.effective_submission_dir

        report = validate_scene(scene)
        if report.problems:
            result.skipped_scenes[scene.name] = report.problems
            continue

        # Real BTS scenes are SIMPLE_RADIAL (radially distorted); the
        # vendored 3DGS only handles PINHOLE/SIMPLE_PINHOLE.
        # undistort_scene is a no-op passthrough for scenes already using a
        # supported model (chair, bonsai) and produces a PINHOLE copy
        # otherwise. Every downstream step in this loop operates on
        # working_scene, never the raw scene, except test_poses_csv /
        # effective_submission_dir — undistortion never touches test poses.
        working_scene = undistort_scene(scene, scene_output / "undistorted")

        sparse = load_sparse_scene(working_scene.sparse_dir)
        file_backed_names = {p.name for p in working_scene.train_images_dir.iterdir() if p.is_file()}
        # Holdout candidates are drawn ONLY from images that actually have
        # a file: a registered-without-file image (e.g. a test_poses.csv
        # image) has no local pixel data to score against even if chosen,
        # and build_filtered_scene would exclude it anyway regardless of
        # whether it's "selected" as holdout.
        camera_centers = {
            img.name: -np.transpose(qvec2rotmat(np.array(img.qvec))) @ np.array(img.tvec)
            for img in sparse.images.values()
            if img.name in file_backed_names
        }
        holdout_names = set(select_holdout_images(camera_centers, holdout_ratio=0.125))

        # Phase A: leak-free eval training on a scene copy with holdout
        # images physically removed from both images.bin and the images
        # folder (Task 8b) — the model literally cannot have seen them.
        # build_filtered_scene also strips registered-without-file images
        # automatically (see its docstring), so this is training-safe.
        filtered_scene = build_filtered_scene(
            working_scene, holdout_names, scene_output / "filtered_scene",
        )
        eval_checkpoint = train_fn(filtered_scene, scene_output / "eval_train")

        sample_image = next(working_scene.train_images_dir.iterdir())
        with Image.open(sample_image) as im:
            image_dims = im.size  # (width, height)

        holdout_params = _camera_params_for_holdout(sparse, holdout_names, image_dims)
        holdout_render_dir = scene_output / "holdout_render"
        rendered_paths = render_fn(eval_checkpoint, holdout_params, holdout_render_dir)

        scores = []
        for path, params in zip(rendered_paths, holdout_params):
            gt_path = working_scene.train_images_dir / params.image_name
            pred = np.array(Image.open(path).convert("RGB"))
            gt = np.array(Image.open(gt_path).convert("RGB").resize(pred.shape[1::-1]))
            metrics = compute_pair_metrics(pred, gt, lpips_model)
            scores.append(combine_score(
                metrics["lpips"], metrics["ssim"], metrics["psnr"], psnr_max,
            ))
        result.per_scene_scores[scene.name] = float(np.mean(scores)) if scores else 0.0

        # Phase B: final training on 100% of the real distributed training
        # data — still goes through build_filtered_scene (empty holdout)
        # to strip registered-without-file images; this is the checkpoint
        # that actually gets shipped in the submission.
        full_training_scene = build_filtered_scene(
            working_scene, set(), scene_output / "full_scene",
        )
        final_checkpoint = train_fn(full_training_scene, scene_output / "final_train")
        test_render_dir = scene_output / "test_render"
        test_params_list = load_test_poses_csv(scene.test_poses_csv)
        render_fn(final_checkpoint, test_params_list, test_render_dir)
        scene_render_dirs[submission_dir] = test_render_dir

    if result.skipped_scenes:
        # Fail closed: the exam voids the ENTIRE score for a missing scene
        # (spec section 14 / debai.md 1.6-8.4), so a submission.zip that's
        # already known to be short a scene is worse than no zip at all —
        # never package or validate one while any scene was skipped.
        result.validation_problems = [
            f"scene '{name}' skipped, no submission produced: {problems}"
            for name, problems in result.skipped_scenes.items()
        ]
        result.submission_zip = None
        return result

    submission_zip = output_root / "submission.zip"
    package_submission(scene_render_dirs, submission_zip)
    result.validation_problems = validate_submission(submission_zip, scenes)
    # Fail closed: an artifact validate_submission has already flagged as
    # wrong (bad size, missing/extra file) must never be exposed as a
    # valid submission, even though the zip stays on disk for debugging —
    # same philosophy already applied above for skipped_scenes.
    result.submission_zip = submission_zip if not result.validation_problems else None
    return result
