"""Per-condition error analysis (3e, EXP-023): where a ranking goes wrong, not only how often.

The headline metrics of docs/05 §3.1–§3.2 say how often a system is right. A report also has to
say *which* conditions it gets wrong, *what it mistakes them for*, and where a missed must-not-miss
condition goes (§8 rule 3: "aggregate numbers hide rare-condition failure"). Everything here reads
finished :class:`~src.eval.metrics.CaseOutcome` lists and returns aggregates: counts, shares and
their 95% bootstrap intervals (§6, 1,000 resamples, seed 42, through
:func:`~src.eval.metrics.bootstrap_ratio`). Nothing trains, tunes or chooses anything, and no
function returns a case id, so every output can be written to a results file as it is.

Two conventions hold throughout:

* **"Confused with" means the first-ranked condition** when it is not the true one: the top-1
  confusion. A case's true condition can be missed from the top 3 with several conditions above
  it; :func:`must_not_miss_misses` counts those too.
* **A share is of the true condition's own cases**, so a confusion of a common condition and of a
  rare one can be compared. Its interval resamples that condition's cases only, in their order.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

from src.conditions import CONDITIONS, must_not_miss_ids
from src.eval.metrics import (
    DEFAULT_RESAMPLES,
    DEFAULT_SEED,
    CaseOutcome,
    Ratio,
    bootstrap_ratio,
    calibration_error_by_condition,
    calibration_error_by_condition_interval,
    calibration_error_interval,
    dangerous_false_negative_rate,
    expected_calibration_error,
    must_not_miss_recall_at_3,
    one_vs_rest,
    reciprocal_rank,
    reliability_rows,
    reliability_table,
    top_k,
)

__all__ = [
    "calibration",
    "confusion_matrix",
    "error_overlap",
    "headline",
    "must_not_miss_misses",
    "recall_by_condition",
    "top_confusions",
]

_REGISTRY = tuple(c.id for c in CONDITIONS)


def _interval(ratio: Ratio, resamples: int, seed: int) -> list[float]:
    low, high = bootstrap_ratio(ratio, resamples=resamples, seed=seed)
    return [low, high]


def _share(hits: Iterable[bool], resamples: int, seed: int) -> dict[str, Any]:
    """A proportion over some cases, with its bootstrap interval."""
    numerator = np.fromiter((float(h) for h in hits), dtype=float)
    ratio = Ratio(numerator, np.ones_like(numerator))
    return {
        "cases": ratio.cases,
        "value": ratio.value,
        "ci": _interval(ratio, resamples, seed),
    }


def headline(
    outcomes: Sequence[CaseOutcome],
    *,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> dict[str, dict[str, Any]]:
    """The few overall figures an error analysis is read against, each with its interval: top-1
    and top-3 accuracy, MRR, must-not-miss recall@3 and the dangerous false-negative rate. They
    are the same functions :func:`~src.eval.metrics.evaluate` uses, so they reproduce its values."""
    ratios = {
        "top-1 accuracy": top_k(outcomes, 1),
        "top-3 accuracy": top_k(outcomes, 3),
        "MRR": reciprocal_rank(outcomes),
        "must-not-miss recall@3": must_not_miss_recall_at_3(outcomes),
        "dangerous false-negative rate": dangerous_false_negative_rate(outcomes),
    }
    return {
        name: {"value": r.value, "ci": _interval(r, resamples, seed), "cases": r.cases}
        for name, r in ratios.items()
    }


def recall_by_condition(
    outcomes: Sequence[CaseOutcome],
    k: int,
    conditions: Sequence[str] | None = None,
    *,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> dict[str, dict[str, Any]]:
    """Per true condition, the share of its cases ranked in the top ``k``, with its interval.

    This is ``reports.score``'s ``top3_by_condition`` for any ``k``: the cases of one condition,
    resampled among themselves. A condition with no case is left out (aortic dissection on
    DDXPlus): there is nothing to measure.
    """
    labels = list(conditions) if conditions is not None else list(_REGISTRY)
    out = {}
    for label in labels:
        subset = [o for o in outcomes if o.true_condition == label]
        if not subset:
            continue
        ratio = top_k(subset, k)
        out[label] = {
            "cases": len(subset),
            "value": ratio.value,
            "ci": _interval(ratio, resamples, seed),
        }
    return out


def confusion_matrix(
    outcomes: Sequence[CaseOutcome], rows: Sequence[str], columns: Sequence[str]
) -> np.ndarray:
    """Counts of true condition (row) against first-ranked condition (column).

    Raises:
        ValueError: for a true condition outside ``rows`` or a first-ranked one outside
            ``columns``, or a case with an empty ranking. Dropping such a case silently would
            make every row's share wrong.
    """
    row_index = {c: i for i, c in enumerate(rows)}
    column_index = {c: j for j, c in enumerate(columns)}
    matrix = np.zeros((len(rows), len(columns)), dtype=np.int64)
    for o in outcomes:
        if not o.ranking:
            raise ValueError(f"{o.case_id}: an empty ranking has no first-ranked condition")
        if o.true_condition not in row_index:
            raise ValueError(f"{o.case_id}: true condition {o.true_condition} is not a row")
        if o.ranking[0] not in column_index:
            raise ValueError(f"{o.case_id}: first-ranked {o.ranking[0]} is not a column")
        matrix[row_index[o.true_condition], column_index[o.ranking[0]]] += 1
    return matrix


def top_confusions(
    outcomes: Sequence[CaseOutcome],
    limit: int = 10,
    *,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> list[dict[str, Any]]:
    """The most frequent top-1 confusions, largest count first.

    Each entry gives the ``true`` and the ``predicted`` (first-ranked) condition, the ``count``,
    and the ``share`` of the true condition's cases, with its interval. Ties in the count go to
    the registry order of the true condition, then of the predicted one, so the list is fixed.
    """
    by_true: dict[str, list[CaseOutcome]] = {}
    for o in outcomes:
        by_true.setdefault(o.true_condition, []).append(o)
    pairs = Counter(
        (o.true_condition, o.ranking[0])
        for o in outcomes
        if o.ranking and o.ranking[0] != o.true_condition
    )
    order = {c: i for i, c in enumerate(_REGISTRY)}

    def key(item: tuple[tuple[str, str], int]) -> tuple:
        (truth, predicted), count = item
        return (-count, order.get(truth, len(order)), order.get(predicted, len(order)))

    out = []
    for (truth, predicted), count in sorted(pairs.items(), key=key)[:limit]:
        cases = by_true[truth]
        share = _share(
            (bool(o.ranking) and o.ranking[0] == predicted for o in cases), resamples, seed
        )
        out.append({"true": truth, "predicted": predicted, "count": count, "share": share})
    return out


def must_not_miss_misses(
    outcomes: Sequence[CaseOutcome],
    must_not_miss: Iterable[str] | None = None,
    *,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """Where a must-not-miss condition goes when it is ranked outside the top 3.

    Returns:
        ``by_condition``: for each must-not-miss condition with cases, its ``cases``, its
        ``misses`` and their share with an interval (one minus that condition's recall@3), the
        true condition's ``rank`` among the misses (``"4"``, ``"5"``, or ``">5"``, the dangerous
        false negatives), ``first``, how often each condition was ranked first in a miss, and
        ``above``, how often each condition was ranked above the true one. ``all``: the same
        counts over every must-not-miss case.
    """
    critical = frozenset(must_not_miss if must_not_miss is not None else must_not_miss_ids())
    order = {c: i for i, c in enumerate(_REGISTRY)}

    def ordered(counter: Counter) -> dict[str, int]:
        return dict(sorted(counter.items(), key=lambda kv: (-kv[1], order.get(kv[0], len(order)))))

    def summarise(cases: Sequence[CaseOutcome]) -> dict[str, Any]:
        missed = [o for o in cases if o.true_condition not in o.ranking[:3]]
        ranks: Counter = Counter()
        first: Counter = Counter()
        above: Counter = Counter()
        for o in missed:
            rank = o.rank()
            ranks["4" if rank == 4 else "5" if rank == 5 else ">5"] += 1
            first[o.ranking[0]] += 1
            stop = rank - 1 if rank is not None else len(o.ranking)
            above.update(o.ranking[:stop])
        return {
            "cases": len(cases),
            "misses": len(missed),
            "share_missed": _share(
                (o.true_condition not in o.ranking[:3] for o in cases), resamples, seed
            ),
            "rank": {r: ranks[r] for r in ("4", "5", ">5")},
            "first": ordered(first),
            "above": ordered(above),
        }

    serious = [o for o in outcomes if o.true_condition in critical]
    by_condition = {}
    for condition in sorted(critical, key=lambda c: order.get(c, len(order))):
        cases = [o for o in serious if o.true_condition == condition]
        if cases:
            by_condition[condition] = summarise(cases)
    return {"by_condition": by_condition, "all": summarise(serious)}


def error_overlap(
    fuller: Sequence[CaseOutcome], shorter: Sequence[CaseOutcome], k: int = 1
) -> dict[str, int]:
    """How the top-``k`` errors of two evidence levels of the same patients relate.

    docs/05 §3.7's masks are nested (a patient's 25% history is part of their 50% history), so
    an error at the shorter level either was already there (``persisting``), is ``new``, or an
    error of the fuller level went away (``resolved``: removing answers can remove a misleading
    one). The cases must be the same patients in the same order.

    Raises:
        ValueError: if the two lists are not the same cases in the same order.
    """
    if [o.case_id for o in fuller] != [o.case_id for o in shorter]:
        raise ValueError("the two levels must be the same cases in the same order")
    wrong_full = [o.true_condition not in o.ranking[:k] for o in fuller]
    wrong_short = [o.true_condition not in o.ranking[:k] for o in shorter]
    pairs = list(zip(wrong_full, wrong_short, strict=True))
    return {
        "errors_fuller": sum(wrong_full),
        "errors_shorter": sum(wrong_short),
        "persisting": sum(a and b for a, b in pairs),
        "new": sum(b and not a for a, b in pairs),
        "resolved": sum(a and not b for a, b in pairs),
    }


def calibration(
    outcomes: Sequence[CaseOutcome],
    conditions: Sequence[str] | None = None,
    *,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
) -> dict[str, Any]:
    """docs/05 §3.3's figures for one system at one level, per condition and overall: the
    top-label reliability table and ECE (10 bins) with its interval, and for every condition some
    case has, its one-vs-rest reliability table and ECE with its interval. These describe the
    scores as given; whether a calibrator should change them is EXP-007's question."""
    by_condition = calibration_error_by_condition(outcomes, conditions)
    intervals = calibration_error_by_condition_interval(
        outcomes, conditions, resamples=resamples, seed=seed
    )
    tables = {
        label: reliability_rows(given, positive)
        for label, (given, positive) in one_vs_rest(outcomes, conditions).items()
    }
    return {
        "top_label": {
            "ece": expected_calibration_error(outcomes),
            "ci": list(calibration_error_interval(outcomes, resamples=resamples, seed=seed)),
            "reliability": reliability_table(outcomes),
        },
        "one_vs_rest": {
            label: {
                "ece": by_condition[label],
                "ci": list(intervals[label]),
                "reliability": tables[label],
            }
            for label in by_condition
        },
    }
