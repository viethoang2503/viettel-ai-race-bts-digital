import csv

import yaml

from src.submission.reproducibility_bundle import write_reproducibility_bundle


def test_write_reproducibility_bundle_creates_config_and_scores_csv(tmp_path):
    chosen_config = {
        "variant": "full_stack",
        "floater_cleanup": True,
        "score": 0.87,
        "estimated_vram_bytes": 12_000_000_000,
        "checkpoint_path": "final.ply",
    }
    all_candidates = [
        chosen_config,
        {
            "variant": "baseline",
            "floater_cleanup": False,
            "score": 0.70,
            "estimated_vram_bytes": 8_000_000_000,
            "checkpoint_path": "b.ply",
        },
    ]

    bundle_dir = write_reproducibility_bundle(
        "chair",
        chosen_config,
        all_candidates,
        tmp_path,
    )

    assert bundle_dir == tmp_path / "chair"
    config_path = bundle_dir / "chosen_config.yaml"
    assert config_path.exists()
    loaded = yaml.safe_load(config_path.read_text())
    assert loaded["variant"] == "full_stack"

    csv_path = bundle_dir / "all_candidates_scores.csv"
    assert csv_path.exists()
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert {row["variant"] for row in rows} == {"full_stack", "baseline"}


def test_write_reproducibility_bundle_preserves_giai_doan_2_and_fallback_fields(
    tmp_path,
):
    chosen_config = {
        "variant": "baseline",
        "floater_cleanup": False,
        "score": 0.60,
        "estimated_vram_bytes": 999_999_999_999,
        "checkpoint_path": "b.ply",
        "fallback_reason": "no candidate fit the VRAM budget",
    }
    all_candidates = [
        chosen_config,
        {
            "variant": "baseline",
            "floater_cleanup": False,
            "candidate_name": "bonsai_0",
            "score": 0.75,
            "estimated_vram_bytes": 999_999_999_999,
            "checkpoint_path": "e0.ply",
            "hyperparam_overrides": {"densify_grad_threshold": 0.0005},
        },
    ]

    bundle_dir = write_reproducibility_bundle(
        "bonsai",
        chosen_config,
        all_candidates,
        tmp_path,
    )

    with open(bundle_dir / "all_candidates_scores.csv", newline="") as f:
        rows = {
            row["candidate_name"] or row["checkpoint_path"]: row
            for row in csv.DictReader(f)
        }

    assert rows["b.ply"]["fallback_reason"] == "no candidate fit the VRAM budget"
    assert rows["bonsai_0"]["candidate_name"] == "bonsai_0"
    assert "0.0005" in rows["bonsai_0"]["hyperparam_overrides"]
