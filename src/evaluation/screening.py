from __future__ import annotations


def needs_tiebreak_rerun(
    candidates: list[dict],
    threshold: float = 0.01,
) -> list[str]:
    if not candidates:
        return []

    ranked = sorted(candidates, key=lambda candidate: candidate["score"], reverse=True)
    leader_score = ranked[0]["score"]
    return [
        candidate["variant"]
        for candidate in ranked[1:]
        if leader_score - candidate["score"] <= threshold
    ]


def variants_needing_full_iteration_verification(
    candidates: list[dict],
    threshold: float = 0.01,
) -> list[str]:
    runner_ups = needs_tiebreak_rerun(candidates, threshold=threshold)
    if not runner_ups:
        return []

    leader = max(candidates, key=lambda candidate: candidate["score"])["variant"]
    return list(dict.fromkeys([leader, *runner_ups]))


def build_hyperparam_candidates(
    base_overrides: dict[str, object],
    extra_overrides: list[dict[str, object]],
    label_prefix: str,
) -> list[dict[str, object]]:
    candidates = []
    for index, override in enumerate(extra_overrides):
        merged = {**base_overrides, **override}
        merged["candidate_name"] = f"{label_prefix}_{index}"
        candidates.append(merged)
    return candidates
