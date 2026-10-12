"""D-13, safety first: may the deep ranker replace logistic regression? (EXP-021)

The owner chose one deep model, a small transformer (D-12), and decided on 2026-10-12 that it runs
the pipeline only if it is at least as safe as the logistic regression trained like it. This module
is that rule as code, written and tested **before any B1-DL model was trained**, so the decision
cannot be bent to the result. It is applied once, on validation, to the seed-42 models
(`docs/05` §6), and the report says for how many of the five seeds it would also hold.

The transformer passes when every check passes:

1. **The safety targets** of `docs/05` §3.2, at 100% and at 25% of the history: must-not-miss
   recall@3 ≥ 0.95 and the dangerous false-negative rate ≤ 0.02 (point estimates, as §7 compares
   targets).
2. **Not worse than logistic regression at 25%** (non-inferiority, margin 0.005): the paired
   bootstrap of the per-case difference, transformer − logistic regression (1,000 resamples, seed
   42), must have its 95% lower bound above −0.005, both for the must-not-miss top-3 hit (over the
   must-not-miss patients) and for the reciprocal rank (over everyone, MRR). A plain significance
   test would be the wrong question: at 34,000 patients it calls a 0.001 difference significant.
3. **The golden clinical cases** all pass with the transformer as the ranker, with red flags on and
   off. The caller runs them (``tests/test_ranker.py``'s checks) and passes the result in.

Otherwise logistic regression stays the ranker, and the transformer is reported as the comparison.
If it passes, the switch is recorded by a dated `docs/05` §9 entry, which §5 requires for a change
of A0's ML component before the test split opens.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from src.conditions import must_not_miss_ids
from src.eval.metrics import (
    DEFAULT_RESAMPLES,
    DEFAULT_SEED,
    CaseOutcome,
    dangerous_false_negative_rate,
    must_not_miss_recall_at_3,
    paired_bootstrap_difference,
    reciprocal_rank,
)

__all__ = ["MARGIN", "RECALL_TARGET", "DANGEROUS_TARGET", "safety_floor"]

RECALL_TARGET = 0.95
DANGEROUS_TARGET = 0.02
MARGIN = 0.005
TARGET_LEVELS = (1.0, 0.25)
COMPARISON_LEVEL = 0.25


def safety_floor(
    candidate: Mapping[float, Sequence[CaseOutcome]],
    comparator: Mapping[float, Sequence[CaseOutcome]],
    *,
    golden_cases_pass: bool,
    must_not_miss: Iterable[str] | None = None,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """D-13's verdict, with every check's figures.

    Args:
        candidate: {evidence level: the transformer's outcomes}, with 1.0 and 0.25.
        comparator: The logistic regression trained like it, at 0.25 at least, on the same
            patients in the same order.
        golden_cases_pass: Whether all four golden cases pass with the transformer as the ranker,
            red flags on and off.
        must_not_miss: The must-not-miss conditions; the registry's by default.

    Returns:
        ``passed``, and ``checks``: one entry per check, each with its ``value`` and ``passed``.

    Raises:
        ValueError: if a level is missing or the two were scored on different patients.
    """
    critical = frozenset(must_not_miss if must_not_miss is not None else must_not_miss_ids())
    missing = [level for level in TARGET_LEVELS if level not in candidate]
    if missing or COMPARISON_LEVEL not in comparator:
        raise ValueError(
            f"the rule needs the candidate at {TARGET_LEVELS} and the comparator at 0.25"
        )
    checks: list[dict[str, Any]] = []
    for level in TARGET_LEVELS:
        recall = must_not_miss_recall_at_3(candidate[level], critical).value
        dangerous = dangerous_false_negative_rate(candidate[level], critical).value
        checks.append(
            {
                "check": f"must-not-miss recall@3 >= {RECALL_TARGET} at {level:.0%}",
                "value": recall,
                "passed": recall >= RECALL_TARGET,
            }
        )
        checks.append(
            {
                "check": f"dangerous false-negative rate <= {DANGEROUS_TARGET} at {level:.0%}",
                "value": dangerous,
                "passed": dangerous <= DANGEROUS_TARGET,
            }
        )

    ours, theirs = candidate[COMPARISON_LEVEL], comparator[COMPARISON_LEVEL]
    if [o.case_id for o in ours] != [o.case_id for o in theirs]:
        raise ValueError("the candidate and the comparator were scored on different patients")
    serious = [i for i, o in enumerate(ours) if o.true_condition in critical]
    hit_ours = [float(ours[i].true_condition in ours[i].ranking[:3]) for i in serious]
    hit_theirs = [float(theirs[i].true_condition in theirs[i].ranking[:3]) for i in serious]
    for name, a, b in (
        ("must-not-miss recall@3", hit_ours, hit_theirs),
        ("MRR", reciprocal_rank(ours).numerator, reciprocal_rank(theirs).numerator),
    ):
        test = paired_bootstrap_difference(a, b, resamples=resamples, seed=seed)
        checks.append(
            {
                "check": f"{name} not worse than logistic regression at 25% (margin {MARGIN})",
                "value": test["difference"],
                "ci": [test["ci_low"], test["ci_high"]],
                "cases": test["cases"],
                "passed": test["ci_low"] > -MARGIN,
            }
        )
    checks.append(
        {
            "check": "all four golden cases pass, red flags on and off",
            "value": golden_cases_pass,
            "passed": bool(golden_cases_pass),
        }
    )
    return {
        "rule": "D-13 (2026-10-12)",
        "passed": all(c["passed"] for c in checks),
        "checks": checks,
    }
