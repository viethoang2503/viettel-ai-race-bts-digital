import pytest

from src.evaluation.screening import (
    build_hyperparam_candidates,
    needs_tiebreak_rerun,
    variants_needing_full_iteration_verification,
)


def test_needs_tiebreak_rerun_flags_close_runner_up():
    candidates = [
        {"variant": "full_stack", "score": 0.700},
        {"variant": "depth_reg", "score": 0.695},
        {"variant": "baseline", "score": 0.500},
    ]
    assert needs_tiebreak_rerun(candidates, threshold=0.01) == ["depth_reg"]


def test_needs_tiebreak_rerun_returns_empty_when_leader_is_clear():
    candidates = [
        {"variant": "full_stack", "score": 0.700},
        {"variant": "baseline", "score": 0.500},
    ]
    assert needs_tiebreak_rerun(candidates, threshold=0.01) == []


def test_needs_tiebreak_rerun_handles_empty_list():
    assert needs_tiebreak_rerun([], threshold=0.01) == []


def test_variants_needing_full_iteration_verification_includes_leader_with_close_runner_up():
    candidates = [
        {"variant": "full_stack", "score": 0.700},
        {"variant": "depth_reg", "score": 0.695},
        {"variant": "baseline", "score": 0.500},
    ]
    result = variants_needing_full_iteration_verification(
        candidates,
        threshold=0.01,
    )
    assert result == ["full_stack", "depth_reg"]


def test_variants_needing_full_iteration_verification_empty_when_leader_is_clear():
    candidates = [
        {"variant": "full_stack", "score": 0.700},
        {"variant": "baseline", "score": 0.500},
    ]
    assert (
        variants_needing_full_iteration_verification(candidates, threshold=0.01)
        == []
    )


def test_variants_needing_full_iteration_verification_handles_empty_list():
    assert variants_needing_full_iteration_verification([], threshold=0.01) == []


def test_build_hyperparam_candidates_merges_and_labels():
    base = {"variant": "depth_reg", "densify_grad_threshold": 0.001}
    extra = [
        {"densify_grad_threshold": 0.0015},
        {"iterations": 45000},
    ]
    candidates = build_hyperparam_candidates(base, extra, label_prefix="bonsai")

    assert candidates == [
        {
            "variant": "depth_reg",
            "densify_grad_threshold": 0.0015,
            "candidate_name": "bonsai_0",
        },
        {
            "variant": "depth_reg",
            "densify_grad_threshold": 0.001,
            "iterations": 45000,
            "candidate_name": "bonsai_1",
        },
    ]


def test_build_hyperparam_candidates_handles_empty_overrides():
    assert build_hyperparam_candidates({"variant": "baseline"}, [], "chair") == []


def test_build_hyperparam_candidates_rejects_more_than_four_extras():
    with pytest.raises(ValueError, match="at most 4"):
        build_hyperparam_candidates(
            {"variant": "baseline"},
            [{"iterations": value} for value in range(5)],
            "chair",
        )
