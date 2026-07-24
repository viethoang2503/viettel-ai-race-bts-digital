from __future__ import annotations


def select_best_candidate(candidates: list[dict], vram_budget_bytes: int) -> dict:
    if not candidates:
        raise ValueError("select_best_candidate requires at least one candidate")

    fitting = [
        candidate
        for candidate in candidates
        if candidate["estimated_vram_bytes"] <= vram_budget_bytes
    ]
    if fitting:
        return max(fitting, key=lambda candidate: candidate["score"])

    smallest = min(candidates, key=lambda candidate: candidate["estimated_vram_bytes"])
    smallest["fallback_reason"] = "no candidate fit the VRAM budget"
    return smallest
