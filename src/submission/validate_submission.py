from __future__ import annotations

import zipfile
from collections import Counter
from io import BytesIO
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from src.common.config import SceneConfig
from src.rendering.render_from_csv import load_test_poses_csv


def validate_submission(zip_path: Path, scenes: list[SceneConfig]) -> list[str]:
    """Validate zip contents against exactly what test_poses.csv expects
    across all `scenes` — flags both MISSING files (Task 11 original scope)
    and UNEXPECTED files (extra images, extra top-level scene directories,
    junk like __MACOSX/, wrong scene naming) since the exam explicitly says
    both missing AND extra scenes/files void the score (debai.md section
    1.6 / 8.4).
    """
    problems: list[str] = []
    zip_path = Path(zip_path)

    with zipfile.ZipFile(zip_path) as zf:
        # Directory marker entries (name ends in "/") aren't real files.
        all_names = [n for n in zf.namelist() if not n.endswith("/")]
        name_counts = Counter(all_names)
        for name, count in sorted(name_counts.items()):
            if count > 1:
                problems.append(
                    f"duplicate entry in submission zip: {name} (appears {count} times)"
                )
        file_names_in_zip = set(name_counts)
        accounted_for: set[str] = set()

        for scene in scenes:
            submission_dir = scene.effective_submission_dir
            expected_params = load_test_poses_csv(scene.test_poses_csv)
            scene_entries = [n for n in file_names_in_zip if n.startswith(f"{submission_dir}/")]
            if not scene_entries:
                problems.append(f"scene '{submission_dir}': no files found in zip")
                continue

            for params in expected_params:
                # Use image_name exactly as given — never renamed to .png.
                arcname = f"{submission_dir}/{params.image_name}"
                if arcname not in file_names_in_zip:
                    problems.append(f"scene '{submission_dir}': missing {params.image_name}")
                    continue
                accounted_for.add(arcname)
                data = zf.read(arcname)
                try:
                    with Image.open(BytesIO(data)) as img:
                        img.load()  # force full decode, not just the header
                        if img.size != (params.width, params.height):
                            problems.append(
                                f"scene '{submission_dir}': {params.image_name} has wrong size "
                                f"{img.size}, expected {(params.width, params.height)}"
                            )
                        if img.mode != "RGB":
                            problems.append(
                                f"scene '{submission_dir}': {params.image_name} has mode "
                                f"'{img.mode}', expected 'RGB'"
                            )
                except (UnidentifiedImageError, OSError) as e:
                    problems.append(
                        f"scene '{submission_dir}': {params.image_name} is not a valid image ({e})"
                    )

        # Anything in the zip that wasn't matched to an expected file above
        # is unexpected: extra images within a known scene, an entire
        # top-level scene directory not in `scenes` at all, or junk like
        # __MACOSX/ or .DS_Store added by some zip tools. The exam voids
        # the whole score for extra/missing scenes, so this must be caught
        # locally before submitting, not discovered after scoring.
        for extra in sorted(file_names_in_zip - accounted_for):
            problems.append(f"unexpected file in submission zip: {extra}")

    return problems
