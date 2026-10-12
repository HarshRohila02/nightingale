"""Tests for D-13's rule (src/eval/safety_floor.py), on hand-built outcomes."""

from __future__ import annotations

import pytest

from src.eval.metrics import CaseOutcome
from src.eval.safety_floor import safety_floor

SERIOUS, BENIGN = "COND:serious", "COND:benign"
OTHERS = tuple(f"COND:other{i}" for i in range(6))


def outcome(i: int, truth: str, rank: int) -> CaseOutcome:
    """The true condition at ``rank`` (1-based) among filler conditions."""
    ranking = list(OTHERS)
    ranking.insert(rank - 1, truth)
    return CaseOutcome(case_id=f"c{i}", true_condition=truth, ranking=tuple(ranking))


def cohort(serious_ranks: list[int], benign_ranks: list[int]) -> list[CaseOutcome]:
    cases = [outcome(i, SERIOUS, r) for i, r in enumerate(serious_ranks)]
    start = len(cases)
    return cases + [outcome(start + i, BENIGN, r) for i, r in enumerate(benign_ranks)]


def verdict(candidate, comparator, golden=True):
    return safety_floor(
        {1.0: candidate, 0.25: candidate},
        {0.25: comparator},
        golden_cases_pass=golden,
        must_not_miss={SERIOUS},
        resamples=200,
    )


def test_a_candidate_as_good_as_the_comparator_passes():
    cases = cohort([1] * 100, [1] * 50 + [2] * 50)
    result = verdict(cases, cases)
    assert result["passed"] and len(result["checks"]) == 7
    assert all(c["passed"] for c in result["checks"])


def test_missing_the_recall_target_fails_even_against_an_equally_bad_comparator():
    cases = cohort([1] * 90 + [4] * 10, [1] * 100)  # recall@3 0.90 < 0.95
    result = verdict(cases, cases)
    assert not result["passed"]
    failed = [c["check"] for c in result["checks"] if not c["passed"]]
    assert failed == [
        "must-not-miss recall@3 >= 0.95 at 100%",
        "must-not-miss recall@3 >= 0.95 at 25%",
    ]


def test_too_many_dangerous_misses_fail():
    cases = cohort([1] * 96 + [6] * 4, [1] * 100)  # ranked 6th: outside the top 5, 4% > 2%
    failed = [c["check"] for c in verdict(cases, cases)["checks"] if not c["passed"]]
    assert "dangerous false-negative rate <= 0.02 at 25%" in failed


def test_a_clearly_lower_mrr_than_logistic_regression_fails():
    ours = cohort([1] * 100, [2] * 100)  # MRR on the benign half 0.5
    theirs = cohort([1] * 100, [1] * 100)
    checks = {c["check"].split(" not worse")[0]: c for c in verdict(ours, theirs)["checks"]}
    assert not checks["MRR"]["passed"] and checks["MRR"]["value"] == pytest.approx(-0.25)
    assert checks["must-not-miss recall@3"]["passed"]


def test_a_tiny_difference_inside_the_margin_passes():
    """Non-inferiority, not significance: one benign patient at rank 2 in 2,000 is −0.00025."""
    ours = cohort([1] * 1000, [2] + [1] * 999)
    theirs = cohort([1] * 1000, [1] * 1000)
    assert verdict(ours, theirs)["passed"]


def test_failing_golden_cases_fail_the_rule():
    cases = cohort([1] * 100, [1] * 100)
    result = verdict(cases, cases, golden=False)
    assert not result["passed"] and not result["checks"][-1]["passed"]


def test_different_patients_or_a_missing_level_are_refused():
    ours = cohort([1] * 10, [1] * 10)
    with pytest.raises(ValueError, match="different patients"):
        verdict(ours, ours[::-1])
    with pytest.raises(ValueError, match="needs the candidate"):
        safety_floor({1.0: ours}, {0.25: ours}, golden_cases_pass=True)
