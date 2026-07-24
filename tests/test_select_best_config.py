from src.evaluation.select_best_config import select_best_candidate


def _candidate(variant, floater_cleanup, score, vram_bytes):
    return {
        "variant": variant,
        "floater_cleanup": floater_cleanup,
        "score": score,
        "estimated_vram_bytes": vram_bytes,
        "checkpoint_path": f"{variant}_{floater_cleanup}.ply",
    }


def test_picks_highest_score_among_candidates_that_fit_budget():
    candidates = [
        _candidate("baseline", False, score=0.70, vram_bytes=1_000),
        _candidate("full_stack", True, score=0.85, vram_bytes=1_500),
        _candidate("depth_reg", False, score=0.78, vram_bytes=1_200),
    ]
    best = select_best_candidate(candidates, vram_budget_bytes=2_000)
    assert best["variant"] == "full_stack"
    assert "fallback_reason" not in best


def test_excludes_candidates_over_vram_budget_even_if_higher_score():
    candidates = [
        _candidate("full_stack", True, score=0.95, vram_bytes=999_999),
        _candidate("baseline", False, score=0.70, vram_bytes=1_000),
    ]
    best = select_best_candidate(candidates, vram_budget_bytes=2_000)
    assert best["variant"] == "baseline"


def test_falls_back_to_smallest_when_nothing_fits_budget():
    candidates = [
        _candidate("full_stack", True, score=0.95, vram_bytes=999_999),
        _candidate("depth_reg", False, score=0.80, vram_bytes=500_000),
    ]
    best = select_best_candidate(candidates, vram_budget_bytes=2_000)
    assert best["variant"] == "depth_reg"
    assert best["fallback_reason"] == "no candidate fit the VRAM budget"
    assert candidates[1]["fallback_reason"] == (
        "no candidate fit the VRAM budget"
    )


def test_raises_on_empty_candidate_list():
    import pytest

    with pytest.raises(ValueError):
        select_best_candidate([], vram_budget_bytes=2_000)
