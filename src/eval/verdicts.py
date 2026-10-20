"""The protocol's verdicts: H1-H6, the Minimum, Target and Stretch (docs/05 §1, §6, §7).

This turns the comparisons ``src.eval.ablation.compare`` makes into the verdicts the frozen
protocol defines. It adds no statistics: it applies the protocol's rules to numbers already
computed, so each rule can be read, tested and pre-registered (EXP-012) before the test split is
opened. Where the protocol leaves a choice open, the function says so in its docstring.

**The rules** (docs/05 §6, amendments 3 and 3a):

* A **hypothesis** is supported only when *every* one of its comparisons passes at α = 0.05: each
  comparison is significant **and** in the system's favour (McNemar's test on top-3 correctness,
  over all patients or over the must-not-miss patients alone; a paired bootstrap interval for MRR).
  One failed comparison defeats it (an intersection-union test, so no multiplicity correction).
* A **Target** compares point estimates; its tests are reported beside it.
* A **Stretch** needs every test to pass.
* **Primed and unprimed systems are judged separately** (§4): A0 uses B1-DL+aug and is compared
  with the B1 variants trained like it; A0′ uses B1-DL′+aug and the primed ones. Both verdicts are
  reported. The plain B1-XGB and B1-LR are comparators for both.
* A comparison against a comparator that is perfect, or that misses five cases or fewer, cannot be
  won (a point estimate cannot exceed 1, and McNemar cannot reach 0.05): the check fails and is
  marked ``ceiling`` so the report can say why (docs/05 §1).

**E-2.** H1-R, H2-R, H6 and the reduced-evidence Target and Stretch need ``reduced_levels``. As
approved (amendment 3a) that is both 50% and 25%; the team may judge them at 25% only (E-2 item 1
(b)). The runner computes the verdicts under each reading and records which one is in force.

**Not decided here, so pre-registered in EXP-012:** H4's test of "more than"
(``ablation.degradation_contrast``), and how H5's "ECE < 0.10" is read (the point estimate; the
interval is reported). H3 and the Target's unsupported-claim rate come from EXP-009/010, and the
Stretch's clinician-reviewed explainability from a clinically trained reviewer, if there is one.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

__all__ = [
    "ECE_TARGET",
    "FAMILIES",
    "MUST_NOT_MISS_TARGET",
    "POLICIES",
    "UNSUPPORTED_TARGET",
    "Family",
    "Table",
    "evaluate_claims",
]

ECE_TARGET = 0.10
MUST_NOT_MISS_TARGET = 0.95
UNSUPPORTED_TARGET = 0.20
UNREACHABLE_MISSES = 5
"""McNemar's test cannot reach α = 0.05 when the comparator misses this many cases or fewer."""

POLICIES: dict[str, tuple[float, ...]] = {
    "both": (0.5, 0.25),
    "25-only": (0.25,),
}
"""The levels the reduced-evidence claims need: as approved (amendment 3a), or E-2 item 1 (b)."""

FULL = 1.0
KG_ONLY = "B2"
FIRST_WRITTEN = "B1-XGB"
BESIDE = "B1-LR"
PLAIN = ("B1-XGB", "B1-LR")

TOP3, MNM, MRR = "top3", "must_not_miss_top3", "mrr"
LABELS = {TOP3: "top-3 accuracy", MNM: "must-not-miss recall@3", MRR: "MRR"}


@dataclass(frozen=True)
class Family:
    """A0 or A0′ and the comparators that go with it (docs/05 §4)."""

    system: str
    component: str
    like: tuple[str, ...]

    @property
    def ml_only(self) -> tuple[str, ...]:
        """Every ML-only system a reduced-evidence claim is judged against."""
        return (*PLAIN, *self.like, self.component)


FAMILIES: dict[str, Family] = {
    "unprimed": Family("A0", "B1-DL+aug", ("B1-XGB+aug", "B1-LR+aug")),
    "primed": Family("A0′", "B1-DL′+aug", ("B1-XGB′+aug", "B1-LR′+aug")),
}


@dataclass(frozen=True)
class Table:
    """What the verdicts read.

    Attributes:
        comparisons: ``{(a, b, level): ablation.compare(a, b)}`` for the pairs the claims use.
        points: ``{(system, level): ablation.point_metrics(system)}``.
        ece: ``{(system, level): {"value": float, "ci": [low, high]}}`` for top-label ECE.
        h4: ``{(full, ablated): ablation.degradation_contrast(...)}`` at 100%.
    """

    comparisons: Mapping[tuple[str, str, float], Mapping[str, Any]]
    points: Mapping[tuple[str, float], Mapping[str, float]]
    ece: Mapping[tuple[str, float], Mapping[str, Any]]
    h4: Mapping[tuple[str, str], Mapping[str, Any]]

    def comparison(self, a: str, b: str, level: float) -> Mapping[str, Any]:
        try:
            return self.comparisons[(a, b, level)]
        except KeyError:
            raise KeyError(f"the comparison {a} against {b} at {level} was not computed") from None


# --------------------------------------------------------------------------- #
# One check, one verdict
# --------------------------------------------------------------------------- #


def _ceiling(part: Mapping[str, Any], metric: str) -> str | None:
    """Why the comparator cannot be beaten, if it cannot (docs/05 §1)."""
    rate = float(part["b"])
    if rate >= 1.0:
        return "the comparator is perfect, so nothing can exceed it"
    if metric == MRR:
        return None
    misses = round((1.0 - rate) * int(part["cases"]))
    if misses <= UNREACHABLE_MISSES:
        return f"the comparator misses {misses} cases: McNemar cannot reach 0.05"
    return None


def _check(
    comparison: Mapping[str, Any], level: float, against: str, metric: str, *, significant: bool
) -> dict[str, Any]:
    part = comparison[metric]
    if significant:
        passed = bool(part["a_significantly_better"])
    else:
        passed = float(part["a"]) > float(part["b"])
    reason = None if passed else _ceiling(part, metric)
    out: dict[str, Any] = {
        "level": level,
        "against": against,
        "metric": LABELS[metric],
        "rule": "significant and better" if significant else "point estimate higher",
        "a": part["a"],
        "b": part["b"],
        "difference": part["difference"],
        "passed": passed,
    }
    if "p_value" in part:
        out["p_value"] = part["p_value"]
    if "ci_low" in part:
        out["ci"] = [part["ci_low"], part["ci_high"]]
    if reason:
        out["ceiling"] = reason
    return out


def _verdict(
    claim: str, checks: Sequence[dict[str, Any]], beside: Sequence[dict[str, Any]] = ()
) -> dict[str, Any]:
    failed = [c for c in checks if not c["passed"]]
    return {
        "claim": claim,
        "status": "supported" if checks and not failed else "not supported",
        "checks": list(checks),
        "failed": len(failed),
        "failed_at_a_ceiling": sum(1 for c in failed if "ceiling" in c),
        "reported_beside": list(beside),
    }


def _checks(
    table: Table,
    system: str,
    comparators: Sequence[str],
    levels: Sequence[float],
    metrics: Sequence[str],
    *,
    significant: bool,
) -> list[dict[str, Any]]:
    return [
        _check(
            table.comparison(system, other, level), level, other, metric, significant=significant
        )
        for level in levels
        for other in comparators
        for metric in metrics
    ]


# --------------------------------------------------------------------------- #
# The claims
# --------------------------------------------------------------------------- #


def evaluate_claims(
    table: Table,
    family: Family,
    reduced_levels: Sequence[float],
    *,
    unsupported_rate: float | None = None,
    clinician_review: bool | None = None,
) -> dict[str, Any]:
    """Every verdict for one family (A0 or A0′), judged at ``reduced_levels`` (E-2).

    Args:
        table: The comparisons, point metrics and ECE (:class:`Table`).
        family: :data:`FAMILIES` entry.
        reduced_levels: :data:`POLICIES`' levels for H1-R, H2-R, H6 and the reduced claims.
        unsupported_rate: A0's unsupported-claim rate from EXP-009/010, or None if not run.
        clinician_review: Whether a clinically trained reviewer judged the explanations, or None.

    Returns:
        ``{claim id: verdict}``, each with its checks, so a report can show every comparison,
        passed or not (docs/05 §8 rule 2).
    """
    a, kg = family.system, KG_ONLY
    levels = tuple(reduced_levels)
    first = (FIRST_WRITTEN, kg)

    h1 = _verdict(
        "H1",
        _checks(table, a, first, (FULL,), (TOP3, MRR), significant=True),
        _checks(table, a, (BESIDE, family.component), (FULL,), (TOP3, MRR), significant=True),
    )
    h2 = _verdict(
        "H2",
        _checks(table, a, (FIRST_WRITTEN,), (FULL,), (MNM,), significant=True),
        _checks(table, a, (BESIDE, family.component), (FULL,), (MNM,), significant=True),
    )
    h1r = _verdict(
        "H1-R",
        _checks(table, a, (*family.ml_only, kg), levels, (TOP3, MRR), significant=True),
    )
    h2r = _verdict("H2-R", _checks(table, a, family.ml_only, levels, (MNM,), significant=True))
    like = family.like
    h6 = _verdict(
        "H6",
        _checks(table, family.component, like, levels, (TOP3, MRR, MNM), significant=True),
        _checks(table, family.component, like, (FULL,), (TOP3, MRR, MNM), significant=True),
    )

    contrast = table.h4[(a, "A1")]
    beside_h4 = table.h4.get((a, "A1b"))
    h4 = {
        "claim": "H4",
        "status": "supported" if contrast["more_on_must_not_miss"] else "not supported",
        "rule": "the interval of (must-not-miss degradation − top-1 degradation) is above 0",
        "against": "A1",
        "figures": dict(contrast),
        "reported_beside": [{"against": "A1b", **dict(beside_h4)}] if beside_h4 else [],
    }

    ece = table.ece[(a, FULL)]
    h5 = {
        "claim": "H5",
        "status": "supported" if float(ece["value"]) < ECE_TARGET else "not supported",
        "rule": f"top-label ECE at 100% is below {ECE_TARGET} (the point estimate)",
        "ece": dict(ece),
    }

    return {
        "H1": h1,
        "H2": h2,
        "H1-R": h1r,
        "H2-R": h2r,
        "H4": h4,
        "H5": h5,
        "H6": h6,
        "Minimum": _minimum(table, a),
        "Target": _target(table, a, unsupported_rate),
        "Stretch": _stretch(table, a, clinician_review),
        "Target, reduced evidence": _reduced_target(table, family, levels),
        "Stretch, reduced evidence": _reduced_stretch(table, family, levels),
    }


def _status(checks: Sequence[Mapping[str, Any]], unassessed: Sequence[str]) -> str:
    if any(not c["passed"] for c in checks):
        return "not met"
    return "incomplete" if unassessed else "met"


def _minimum(table: Table, a: str) -> dict[str, Any]:
    """A0 beats B0 (top-3, a point estimate, the test beside it); the other two items (the
    pipeline runs end to end, every metric has an interval) are properties of the run."""
    checks = [_check(table.comparison(a, "B0", FULL), FULL, "B0", TOP3, significant=False)]
    return {
        "claim": "Minimum",
        "status": _status(checks, ()),
        "checks": checks,
        "reported_beside": [
            _check(table.comparison(a, "B0", FULL), FULL, "B0", TOP3, significant=True)
        ],
        "note": "also needs a complete run: every ablation scored, every metric with its interval",
    }


def _target(table: Table, a: str, unsupported_rate: float | None) -> dict[str, Any]:
    checks = _checks(table, a, (FIRST_WRITTEN,), (FULL,), (TOP3, MNM), significant=False)
    ece = table.ece[(a, FULL)]
    checks.append(
        {
            "level": FULL,
            "against": "—",
            "metric": "ECE",
            "rule": f"below {ECE_TARGET}",
            "a": ece["value"],
            "passed": float(ece["value"]) < ECE_TARGET,
        }
    )
    unassessed = []
    if unsupported_rate is None:
        unassessed.append("unsupported-claim rate (EXP-009/010 not run)")
    else:
        checks.append(
            {
                "level": FULL,
                "against": "—",
                "metric": "unsupported-claim rate",
                "rule": f"below {UNSUPPORTED_TARGET}",
                "a": unsupported_rate,
                "passed": unsupported_rate < UNSUPPORTED_TARGET,
            }
        )
    return {
        "claim": "Target",
        "status": _status(checks, unassessed),
        "checks": checks,
        "not_assessed": unassessed,
        "reported_beside": _checks(table, a, (BESIDE,), (FULL,), (TOP3, MNM), significant=False)
        + _checks(table, a, (FIRST_WRITTEN,), (FULL,), (TOP3, MNM), significant=True),
    }


def _stretch(table: Table, a: str, clinician_review: bool | None) -> dict[str, Any]:
    checks = _checks(table, a, (FIRST_WRITTEN,), (FULL,), (TOP3, MNM), significant=True)
    recall = float(table.points[(a, FULL)]["must-not-miss recall@3"])
    checks.append(
        {
            "level": FULL,
            "against": "—",
            "metric": "must-not-miss recall@3",
            "rule": f"at least {MUST_NOT_MISS_TARGET}",
            "a": recall,
            "passed": recall >= MUST_NOT_MISS_TARGET,
        }
    )
    unassessed = [] if clinician_review else ["clinician-reviewed explainability"]
    return {
        "claim": "Stretch",
        "status": _status(checks, unassessed),
        "checks": checks,
        "not_assessed": unassessed,
        "reported_beside": _checks(table, a, (BESIDE,), (FULL,), (TOP3, MNM), significant=True),
    }


def _reduced_target(table: Table, family: Family, levels: Sequence[float]) -> dict[str, Any]:
    """A0 against every ML-only B1 variant, plain and trained like it (amendment 3a), at each
    reduced level, on top-3 and must-not-miss recall@3 (point estimates, tests beside)."""
    comparators = (*PLAIN, *family.like)
    checks = _checks(table, family.system, comparators, levels, (TOP3, MNM), significant=False)
    return {
        "claim": "Target, reduced evidence",
        "status": _status(checks, ()),
        "checks": checks,
        "reported_beside": _checks(
            table, family.system, comparators, levels, (TOP3, MNM), significant=True
        ),
    }


def _reduced_stretch(table: Table, family: Family, levels: Sequence[float]) -> dict[str, Any]:
    comparators = (*PLAIN, *family.like)
    checks = _checks(table, family.system, comparators, levels, (TOP3, MNM), significant=True)
    for level in levels:
        recall = float(table.points[(family.system, level)]["must-not-miss recall@3"])
        checks.append(
            {
                "level": level,
                "against": "—",
                "metric": "must-not-miss recall@3",
                "rule": f"at least {MUST_NOT_MISS_TARGET}",
                "a": recall,
                "passed": recall >= MUST_NOT_MISS_TARGET,
            }
        )
    return {"claim": "Stretch, reduced evidence", "status": _status(checks, ()), "checks": checks}
