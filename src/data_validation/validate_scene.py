from __future__ import annotations

import csv
from dataclasses import dataclass, field

from src.common.colmap_io import load_sparse_scene
from src.common.config import SceneConfig

REQUIRED_CSV_COLUMNS = {
    "image_name", "qw", "qx", "qy", "qz", "tx", "ty", "tz",
    "fx", "fy", "cx", "cy", "width", "height",
}
NUMERIC_CSV_COLUMNS = {
    "qw", "qx", "qy", "qz", "tx", "ty", "tz", "fx", "fy", "cx", "cy", "width", "height",
}
# SIMPLE_RADIAL is "supported" because run_baseline_pipeline always routes
# scenes through undistort_scene before training, converting SIMPLE_RADIAL
# to PINHOLE — see src/training/undistort_scene.py. This module never
# undistorts anything itself; it only decides whether the model is one
# undistort_scene knows how to handle.
SUPPORTED_CAMERA_MODELS = {"PINHOLE", "SIMPLE_PINHOLE", "SIMPLE_RADIAL"}


@dataclass
class ValidationReport:
    """Data-quality report for one scene.

    `registered_without_file` is informational only, NOT flagged into
    `problems` — see its field-level note below for why. This module is a
    read-only data-quality check; it does not make any scene safe to feed
    into the real training loader by itself. That's a separate, mandatory
    step (`build_filtered_scene` in the training module), which always
    strips registered-without-file images before training regardless of
    what this report says, since the vendored loader crashes on any of
    them otherwise. Do not treat a clean `ValidationReport` as license to
    skip that filtering step.
    """

    scene_name: str
    registered_image_count: int = 0
    folder_image_count: int = 0
    # Images registered in images.bin with no corresponding file in
    # train_images_dir. This is normal, intentional dataset structure —
    # verified across all 7 scenes: always a superset of (often exactly
    # equal to) the scene's test_poses.csv image names, plus extra
    # calibration-only frames for some scenes — not data corruption, so it
    # is never added to `problems`. It is still training-relevant (see
    # class docstring), just not a data-quality problem.
    registered_without_file: list[str] = field(default_factory=list)
    extra_images: list[str] = field(default_factory=list)
    test_pose_row_count: int = 0
    camera_model: str | None = None
    problems: list[str] = field(default_factory=list)


def validate_scene(scene: SceneConfig) -> ValidationReport:
    report = ValidationReport(scene_name=scene.name)

    if not scene.sparse_dir.exists():
        report.problems.append(f"sparse dir not found: {scene.sparse_dir}")
        return report
    if not scene.train_images_dir.exists():
        report.problems.append(f"train images dir not found: {scene.train_images_dir}")
        return report

    sparse = load_sparse_scene(scene.sparse_dir)
    registered_names = {img.name for img in sparse.images.values()}
    folder_names = {p.name for p in scene.train_images_dir.iterdir() if p.is_file()}

    report.registered_image_count = len(registered_names)
    report.folder_image_count = len(folder_names)
    # NOT a problem: images.bin legitimately registers more cameras than
    # are distributed as files (test poses + extra calibration frames) —
    # see the Interfaces note above. Reported for visibility only.
    report.registered_without_file = sorted(registered_names - folder_names)
    report.extra_images = sorted(folder_names - registered_names)

    if report.extra_images:
        report.problems.append(
            f"{len(report.extra_images)} folder image(s) not registered in images.bin: "
            f"{report.extra_images}"
        )

    camera_models = {cam.model for cam in sparse.cameras.values()}
    report.camera_model = next(iter(camera_models)) if len(camera_models) == 1 else ",".join(sorted(camera_models))
    unsupported = camera_models - SUPPORTED_CAMERA_MODELS
    if unsupported:
        report.problems.append(
            f"unsupported camera model(s) {sorted(unsupported)}; "
            f"baseline expects one of {sorted(SUPPORTED_CAMERA_MODELS)}"
        )

    if not scene.test_poses_csv.exists():
        report.problems.append(f"test_poses.csv not found: {scene.test_poses_csv}")
        return report

    with open(scene.test_poses_csv, newline="") as f:
        reader = csv.DictReader(f)
        missing_cols = REQUIRED_CSV_COLUMNS - set(reader.fieldnames or [])
        if missing_cols:
            report.problems.append(f"test_poses.csv missing columns: {sorted(missing_cols)}")
        rows = list(reader)
        report.test_pose_row_count = len(rows)

    seen_names: set[str] = set()
    for row in rows:
        name = row.get("image_name", "")
        if name in seen_names:
            report.problems.append(f"duplicate image_name in test_poses.csv: {name}")
        seen_names.add(name)

        # The one direction where a name outside train/images/ IS a
        # problem: a test pose with no COLMAP registration at all has no
        # camera to have derived its pose from.
        if name and name not in registered_names:
            report.problems.append(
                f"test pose '{name}' not registered in images.bin (no COLMAP camera for it)"
            )

        numeric_values = {}
        for col in NUMERIC_CSV_COLUMNS:
            raw = row.get(col)
            try:
                numeric_values[col] = float(raw)
            except (TypeError, ValueError):
                report.problems.append(
                    f"{name}: column '{col}' is not numeric (got {raw!r})"
                )

        if "width" in numeric_values and numeric_values["width"] <= 0:
            report.problems.append(f"{name}: width must be positive, got {numeric_values['width']}")
        if "height" in numeric_values and numeric_values["height"] <= 0:
            report.problems.append(f"{name}: height must be positive, got {numeric_values['height']}")
        for col in ("fx", "fy"):
            if col in numeric_values and numeric_values[col] <= 0:
                report.problems.append(f"{name}: {col} must be positive, got {numeric_values[col]}")

    return report
