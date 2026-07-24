from __future__ import annotations

import csv
import json
import zipfile
from pathlib import Path

import yaml

_REQUIRED_BUNDLE_FILES = (
    "chosen_config.yaml",
    "all_candidates_scores.csv",
)


def _serializable(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {
            str(key): _serializable(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_serializable(item) for item in value]
    return value


def write_reproducibility_bundle(
    scene_name: str,
    chosen_config: dict,
    all_candidates: list[dict],
    output_dir: Path,
) -> Path:
    bundle_dir = Path(output_dir) / scene_name
    bundle_dir.mkdir(parents=True, exist_ok=True)

    with open(bundle_dir / "chosen_config.yaml", "w") as f:
        yaml.safe_dump(_serializable(chosen_config), f)

    fieldnames = [
        "variant",
        "floater_cleanup",
        "candidate_name",
        "score",
        "estimated_vram_bytes",
        "checkpoint_path",
        "hyperparam_overrides",
        "fallback_reason",
        "seed",
        "selection_checkpoint_path",
        "final_checkpoint_path",
        "final_estimated_vram_bytes",
        "final_measured_peak_vram_bytes",
        "final_render_config",
    ]
    with open(bundle_dir / "all_candidates_scores.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for candidate in sorted(all_candidates, key=lambda item: -item["score"]):
            row = {key: candidate.get(key) for key in fieldnames}
            if row["hyperparam_overrides"] is not None:
                row["hyperparam_overrides"] = json.dumps(
                    _serializable(row["hyperparam_overrides"]),
                    sort_keys=True,
                )
            if row["final_render_config"] is not None:
                row["final_render_config"] = json.dumps(
                    _serializable(row["final_render_config"]),
                    sort_keys=True,
                )
            writer.writerow(row)

    return bundle_dir


def validate_reproducibility_bundle(
    root_dir: Path,
    expected_scene_names: list[str],
) -> list[str]:
    root_dir = Path(root_dir)
    problems = []
    for scene_name in expected_scene_names:
        scene_dir = root_dir / scene_name
        if not scene_dir.is_dir():
            problems.append(
                f"scene '{scene_name}': reproducibility directory is missing"
            )
            continue
        for filename in _REQUIRED_BUNDLE_FILES:
            path = scene_dir / filename
            if not path.is_file():
                problems.append(f"missing {scene_name}/{filename}")
    return problems


def package_reproducibility_bundle(
    root_dir: Path,
    output_zip: Path,
    expected_scene_names: list[str],
) -> Path:
    root_dir = Path(root_dir)
    output_zip = Path(output_zip)
    problems = validate_reproducibility_bundle(
        root_dir,
        expected_scene_names,
    )
    if problems:
        raise ValueError(
            "incomplete reproducibility bundle: " + "; ".join(problems)
        )

    output_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as archive:
        for scene_name in expected_scene_names:
            scene_dir = root_dir / scene_name
            for path in sorted(scene_dir.rglob("*")):
                if path.is_file():
                    archive.write(
                        path,
                        arcname=str(path.relative_to(root_dir)),
                    )
    return output_zip
