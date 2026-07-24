from __future__ import annotations

import csv
import json
from pathlib import Path

import yaml


def write_reproducibility_bundle(
    scene_name: str,
    chosen_config: dict,
    all_candidates: list[dict],
    output_dir: Path,
) -> Path:
    bundle_dir = Path(output_dir) / scene_name
    bundle_dir.mkdir(parents=True, exist_ok=True)

    with open(bundle_dir / "chosen_config.yaml", "w") as f:
        yaml.safe_dump(chosen_config, f)

    fieldnames = [
        "variant",
        "floater_cleanup",
        "candidate_name",
        "score",
        "estimated_vram_bytes",
        "checkpoint_path",
        "hyperparam_overrides",
        "fallback_reason",
    ]
    with open(bundle_dir / "all_candidates_scores.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for candidate in sorted(all_candidates, key=lambda item: -item["score"]):
            row = {key: candidate.get(key) for key in fieldnames}
            if row["hyperparam_overrides"] is not None:
                row["hyperparam_overrides"] = json.dumps(row["hyperparam_overrides"])
            writer.writerow(row)

    return bundle_dir
