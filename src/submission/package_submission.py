from __future__ import annotations

import zipfile
from pathlib import Path


def package_submission(scene_render_dirs: dict[str, Path], output_zip: Path) -> Path:
    output_zip = Path(output_zip)
    output_zip.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for scene_name, render_dir in scene_render_dirs.items():
            render_dir = Path(render_dir)
            for image_path in sorted(render_dir.iterdir()):
                if not image_path.is_file():
                    continue
                arcname = f"{scene_name}/{image_path.name}"
                zf.write(image_path, arcname=arcname)

    return output_zip
