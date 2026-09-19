"""Evaluation metrics, as docs/05-evaluation-protocol.md defines them (task 1e).

Everything here scores a finished ranking; nothing trains or tunes. The input is one
:class:`CaseOutcome` per case: the system's ranking, the ground truth, and optionally red flags
and probabilities. The functions follow the frozen protocol and its 2026-09-19 amendments:

* **§3.1 ranking:** top-1, top-3 (the primary ranking metric) and top-5 accuracy, MRR,
  Precision@3, Recall@5, and per-condition F1 (macro and micro). Precision@3 and Recall@5 use
  ``D_in``, the part of the ground-truth differential inside the 13 conditions (amendment 1,
  decision D-8). Recall@5 over the full ``D``, as first written, is reported next to it.
* **§3.2 safety:** must-not-miss recall@3, the dangerous false-negative rate, and red-flag
  sensitivity, measured against the true condition (amendment 2). Red-flag precision is
  reported but has no target.
* **§3.3 calibration:** expected calibration error over 10 bins, the Brier score, and the
  reliability table behind a reliability diagram.
* **§6 statistics:** a 95% bootstrap confidence interval for every metric (1,000 resamples of the
  cases, seed 42), and McNemar's test for paired top-3 correctness.

Most metrics are a ratio of per-case sums. Top-3 accuracy is the number of hits over the number
of cases; must-not-miss recall counts only the must-not-miss cases. :class:`Ratio` holds those
per-case numerators and denominators. That makes the bootstrap one routine for all of them: resample
the cases, then recompute the ratio.

Every result must be reported with its interval (docs/05 §6 and §8), with the closed-world
caveat (R-13), and, for KG results, with the circularity caveat (R-12).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.conditions import CONDITIONS, must_not_miss_ids, trainable_ids

__all__ = [
    "CaseOutcome",
    "MetricResult",
    "Ratio",
    "bootstrap_ratio",
    "brier_score",
    "evaluate",
    "expected_calibration_error",
    "f1_by_condition",
    "mcnemar",
    "outcome_from_record",
    "ranking_from_scores",
    "reliability_table",
]

DEFAULT_RESAMPLES = 1_000
DEFAULT_SEED = 42
DEFAULT_LEVEL = 0.95
CALIBRATION_BINS = 10
EXACT_MCNEMAR_BELOW = 25
"""Discordant pairs below this use the exact binomial test; above it, chi-square."""

_REGISTRY_ORDER = {c.id: i for i, c in enumerate(CONDITIONS)}


@dataclass(frozen=True)
class CaseOutcome:
    """One evaluated case.

    Attributes:
        case_id: The case.
        true_condition: ``y``, the true pathology as a ``COND:*`` id.
        ranking: The system's candidates, best first.
        differential: ``D_in``: the ground-truth differential's in-scope conditions.
        differential_size: ``|D|`` as DDXPlus lists it, out-of-scope entries included. It is
            only needed for Recall@5 over the full ``D``.
        flagged: The conditions a red flag fired for.
        probabilities: {condition: probability}, for calibration. Leave it out until the
            scores are meant to be probabilities (docs/05 §3.3).
    """

    case_id: str
    true_condition: str
    ranking: tuple[str, ...]
    differential: frozenset[str] = frozenset()
    differential_size: int = 0
    flagged: frozenset[str] = frozenset()
    probabilities: Mapping[str, float] | None = field(default=None, compare=False)

    def rank(self) -> int | None:
        """1-based rank of the true condition, or None if the system did not rank it."""
        try:
            return self.ranking.index(self.true_condition) + 1
        except ValueError:
            return None


def ranking_from_scores(scores: Mapping[str, float]) -> tuple[str, ...]:
    """Condition ids by descending score.

    Ties break by the registry order in src/conditions.py, which is fixed but clinically
    arbitrary. A scorer that ties often, such as the graph-only overlap score, should be
    reported with how often it ties.
    """
    return tuple(
        sorted(
            scores,
            key=lambda cid: (-scores[cid], _REGISTRY_ORDER.get(cid, len(_REGISTRY_ORDER)), cid),
        )
    )


def outcome_from_record(
    record: Mapping[str, Any],
    ranking: Sequence[str],
    *,
    flagged: Iterable[str] = (),
    probabilities: Mapping[str, float] | None = None,
) -> CaseOutcome:
    """A :class:`CaseOutcome` from a row of the chest-pain parquet (docs/03 §2.1).

    This reads the label columns, which is what evaluation is for. It is never used to build
    model inputs.
    """
    differential = list(record["label_differential"])
    return CaseOutcome(
        case_id=str(record["case_id"]),
        true_condition=str(record["label_condition_id"]),
        ranking=tuple(ranking),
        differential=frozenset(e["condition_id"] for e in differential if e["condition_id"]),
        differential_size=len(differential),
        flagged=frozenset(flagged),
        probabilities=probabilities,
    )


# --------------------------------------------------------------------------- #
# Ratios and the bootstrap
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Ratio:
    """A metric as per-case numerators over per-case denominators.

    The value is ``sum(numerator) / sum(denominator)``. A case with denominator 0 does not
    count: for must-not-miss recall, that is every case whose true condition is not
    must-not-miss.
    """

    numerator: np.ndarray
    denominator: np.ndarray

    @property
    def cases(self) -> int:
        """How many cases the metric is computed over."""
        return int(np.count_nonzero(self.denominator))

    @property
    def value(self) -> float:
        total = float(self.denominator.sum())
        return float(self.numerator.sum()) / total if total else math.nan


def bootstrap_ratio(
    ratio: Ratio,
    *,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    level: float = DEFAULT_LEVEL,
) -> tuple[float, float]:
    """Percentile bootstrap interval: resample the cases with replacement, recompute the ratio.

    With the same seed, every metric of an evaluation sees the same resamples.
    """
    n = len(ratio.numerator)
    if n == 0 or not ratio.cases:
        return (math.nan, math.nan)
    rng = np.random.default_rng(seed)
    values = np.empty(resamples)
    done = 0
    chunk = max(1, min(resamples, 20_000_000 // max(n, 1)))
    while done < resamples:
        size = min(chunk, resamples - done)
        index = rng.integers(0, n, size=(size, n), dtype=np.int32)
        numerators = ratio.numerator[index].sum(axis=1)
        denominators = ratio.denominator[index].sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            values[done : done + size] = numerators / denominators
        done += size
    tail = (1 - level) / 2
    low, high = np.nanquantile(values, [tail, 1 - tail])
    return (float(low), float(high))


@dataclass(frozen=True)
class MetricResult:
    """A metric with its bootstrap interval, ready to report (docs/05 §8)."""

    name: str
    value: float
    ci_low: float
    ci_high: float
    cases: int

    def __str__(self) -> str:
        return f"{self.name}: {self.value:.3f} [{self.ci_low:.3f}, {self.ci_high:.3f}] (n={self.cases})"


# --------------------------------------------------------------------------- #
# §3.1 ranking
# --------------------------------------------------------------------------- #


def _ratio(values: Iterable[float], counts: Iterable[float] | None = None) -> Ratio:
    """Per-case values over per-case counts. A case that does not count (count 0) contributes
    nothing to the numerator either: a GERD case's top-3 hit is not a must-not-miss hit."""
    numerator = np.fromiter(values, dtype=float)
    if counts is None:
        return Ratio(numerator, np.ones_like(numerator))
    denominator = np.fromiter(counts, dtype=float)
    return Ratio(np.where(denominator != 0, numerator, 0.0), denominator)


def top_k(outcomes: Sequence[CaseOutcome], k: int) -> Ratio:
    """Fraction of cases whose true condition is in the top ``k``."""
    return _ratio(float(o.true_condition in o.ranking[:k]) for o in outcomes)


def reciprocal_rank(outcomes: Sequence[CaseOutcome]) -> Ratio:
    """MRR: mean of 1 / rank of the true condition (0 if it is not ranked)."""
    return _ratio(1.0 / r if (r := o.rank()) else 0.0 for o in outcomes)


def precision_at_3(outcomes: Sequence[CaseOutcome]) -> Ratio:
    """|top-3 ∩ D_in| / 3. The same with the full D: every candidate is in scope."""
    return _ratio(len(set(o.ranking[:3]) & o.differential) / 3 for o in outcomes)


def recall_at_5(outcomes: Sequence[CaseOutcome], *, full_differential: bool = False) -> Ratio:
    """|top-5 ∩ D_in| / |D_in| (amendment 1), or / |D| with ``full_differential`` (as first
    written). A case with an empty differential does not count."""
    values, counts = [], []
    for o in outcomes:
        size = o.differential_size if full_differential else len(o.differential)
        hits = len(set(o.ranking[:5]) & o.differential)
        values.append(hits / size if size else 0.0)
        counts.append(1.0 if size else 0.0)
    return _ratio(values, counts)


def f1_by_condition(
    outcomes: Sequence[CaseOutcome], conditions: Sequence[str] | None = None
) -> dict[str, float]:
    """Per-condition F1 of the top-1 prediction, plus ``macro`` and ``micro``.

    With one true condition and one prediction per case, micro F1 equals top-1 accuracy.
    """
    labels = list(conditions) if conditions is not None else sorted(trainable_ids())
    truth = [o.true_condition for o in outcomes]
    predicted = [o.ranking[0] if o.ranking else "" for o in outcomes]
    scores: dict[str, float] = {}
    for label in labels:
        tp = sum(t == label and p == label for t, p in zip(truth, predicted, strict=True))
        fp = sum(t != label and p == label for t, p in zip(truth, predicted, strict=True))
        fn = sum(t == label and p != label for t, p in zip(truth, predicted, strict=True))
        scores[label] = 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else 0.0
    scores["macro"] = float(np.mean([scores[label] for label in labels])) if labels else math.nan
    correct = sum(t == p for t, p in zip(truth, predicted, strict=True))
    scores["micro"] = correct / len(outcomes) if outcomes else math.nan
    return scores


# --------------------------------------------------------------------------- #
# §3.2 safety
# --------------------------------------------------------------------------- #


def must_not_miss_recall_at_3(
    outcomes: Sequence[CaseOutcome], must_not_miss: Iterable[str] | None = None
) -> Ratio:
    """Of cases whose true condition is must-not-miss, the fraction with it in the top 3."""
    critical = frozenset(must_not_miss if must_not_miss is not None else must_not_miss_ids())
    return _ratio(
        (float(o.true_condition in o.ranking[:3]) for o in outcomes),
        (float(o.true_condition in critical) for o in outcomes),
    )


def dangerous_false_negative_rate(
    outcomes: Sequence[CaseOutcome], must_not_miss: Iterable[str] | None = None
) -> Ratio:
    """Of must-not-miss cases, the fraction whose true condition is ranked outside the top 5."""
    critical = frozenset(must_not_miss if must_not_miss is not None else must_not_miss_ids())
    return _ratio(
        (float(o.true_condition not in o.ranking[:5]) for o in outcomes),
        (float(o.true_condition in critical) for o in outcomes),
    )


def red_flag_sensitivity(
    outcomes: Sequence[CaseOutcome],
    must_not_miss: Iterable[str] | None = None,
    *,
    condition: str | None = None,
) -> Ratio:
    """Amendment 2: of cases whose true condition is must-not-miss, the fraction where a flag
    fired for that condition. A condition without a rule is never flagged, so it counts as
    missed. With ``condition``, only that condition's cases count."""
    critical = frozenset(must_not_miss if must_not_miss is not None else must_not_miss_ids())
    if condition is not None:
        critical &= {condition}
    return _ratio(
        (float(o.true_condition in o.flagged) for o in outcomes),
        (float(o.true_condition in critical) for o in outcomes),
    )


def red_flag_precision(
    outcomes: Sequence[CaseOutcome], appropriate: Mapping[str, Iterable[str]] | None = None
) -> Ratio:
    """Of fired flags, the fraction that were appropriate (reported, not targeted: docs/05 §3.2).

    By default a flag is appropriate only when it names the true condition. That is strict: a
    myocardial-infarction flag on an unstable-angina patient is clinically reasonable (EXP-014).
    ``appropriate`` maps a flagged condition to every true condition it is appropriate for.
    """
    allowed = {c: frozenset(v) for c, v in (appropriate or {}).items()}
    return _ratio(
        (float(sum(o.true_condition in allowed.get(f, {f}) for f in o.flagged)) for o in outcomes),
        (float(len(o.flagged)) for o in outcomes),
    )


# --------------------------------------------------------------------------- #
# §3.3 calibration
# --------------------------------------------------------------------------- #


def _top_label(outcomes: Sequence[CaseOutcome]) -> tuple[np.ndarray, np.ndarray]:
    """(confidence, correct) for each case: the probability of the top-ranked condition."""
    confidence, correct = [], []
    for o in outcomes:
        if o.probabilities is None or not o.ranking:
            raise ValueError(f"{o.case_id}: calibration needs probabilities and a ranking")
        top = o.ranking[0]
        confidence.append(float(o.probabilities.get(top, 0.0)))
        correct.append(float(top == o.true_condition))
    return np.array(confidence), np.array(correct)


def reliability_table(
    outcomes: Sequence[CaseOutcome], bins: int = CALIBRATION_BINS
) -> list[dict[str, float]]:
    """Equal-width confidence bins: cases, mean confidence and accuracy in each (the data behind
    a reliability diagram). Empty bins are left out."""
    confidence, correct = _top_label(outcomes)
    edges = np.minimum((confidence * bins).astype(int), bins - 1)
    rows = []
    for b in range(bins):
        inside = edges == b
        if inside.any():
            rows.append(
                {
                    "bin_low": b / bins,
                    "bin_high": (b + 1) / bins,
                    "cases": int(inside.sum()),
                    "confidence": float(confidence[inside].mean()),
                    "accuracy": float(correct[inside].mean()),
                }
            )
    return rows


def expected_calibration_error(
    outcomes: Sequence[CaseOutcome], bins: int = CALIBRATION_BINS
) -> float:
    """Top-label ECE: the gap between confidence and accuracy, averaged over bins by size."""
    table = reliability_table(outcomes, bins)
    total = sum(row["cases"] for row in table)
    return sum(row["cases"] / total * abs(row["confidence"] - row["accuracy"]) for row in table)


def brier_score(outcomes: Sequence[CaseOutcome], conditions: Sequence[str] | None = None) -> Ratio:
    """Multi-class Brier score: the squared error of the probability vector, per case."""
    labels = list(conditions) if conditions is not None else sorted(trainable_ids())
    values = []
    for o in outcomes:
        if o.probabilities is None:
            raise ValueError(f"{o.case_id}: the Brier score needs probabilities")
        values.append(
            sum(
                (o.probabilities.get(c, 0.0) - (1.0 if c == o.true_condition else 0.0)) ** 2
                for c in labels
            )
        )
    return _ratio(values)


# --------------------------------------------------------------------------- #
# §6 statistics
# --------------------------------------------------------------------------- #


def mcnemar(correct_a: Sequence[bool], correct_b: Sequence[bool]) -> dict[str, float | str]:
    """McNemar's test on paired correctness, e.g. top-3 hits of A0 and B1 (docs/05 §6).

    Returns:
        ``only_a`` and ``only_b`` (the discordant counts), the ``statistic``, the two-sided
        ``p_value``, and the ``method``: exact binomial below 25 discordant pairs, otherwise
        chi-square with continuity correction.
    """
    if len(correct_a) != len(correct_b):
        raise ValueError("McNemar's test needs the same cases for both systems")
    only_a = sum(bool(a) and not bool(b) for a, b in zip(correct_a, correct_b, strict=True))
    only_b = sum(bool(b) and not bool(a) for a, b in zip(correct_a, correct_b, strict=True))
    discordant = only_a + only_b
    if discordant == 0:
        return {"only_a": 0, "only_b": 0, "statistic": 0.0, "p_value": 1.0, "method": "none"}
    if discordant < EXACT_MCNEMAR_BELOW:
        tail = sum(math.comb(discordant, i) for i in range(min(only_a, only_b) + 1))
        p_value = min(1.0, 2 * tail / 2**discordant)
        return {
            "only_a": only_a,
            "only_b": only_b,
            "statistic": float(min(only_a, only_b)),
            "p_value": p_value,
            "method": "exact binomial",
        }
    statistic = (abs(only_a - only_b) - 1) ** 2 / discordant
    return {
        "only_a": only_a,
        "only_b": only_b,
        "statistic": statistic,
        "p_value": math.erfc(math.sqrt(statistic / 2)),
        "method": "chi-square, continuity-corrected",
    }


# --------------------------------------------------------------------------- #
# Everything at once
# --------------------------------------------------------------------------- #


def evaluate(
    outcomes: Sequence[CaseOutcome],
    *,
    must_not_miss: Iterable[str] | None = None,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    level: float = DEFAULT_LEVEL,
) -> list[MetricResult]:
    """Every §3.1 and §3.2 ratio metric with its bootstrap interval, in protocol order.

    Red-flag metrics appear when any case has flags; the Brier score when every case has
    probabilities. F1 and ECE are not ratios of per-case sums: use :func:`f1_by_condition` and
    :func:`expected_calibration_error`.
    """
    critical = list(must_not_miss if must_not_miss is not None else must_not_miss_ids())
    metrics: list[tuple[str, Callable[[], Ratio]]] = [
        ("top-1 accuracy", lambda: top_k(outcomes, 1)),
        ("top-3 accuracy", lambda: top_k(outcomes, 3)),
        ("top-5 accuracy", lambda: top_k(outcomes, 5)),
        ("MRR", lambda: reciprocal_rank(outcomes)),
        ("Precision@3", lambda: precision_at_3(outcomes)),
        ("Recall@5", lambda: recall_at_5(outcomes)),
        ("Recall@5, full D", lambda: recall_at_5(outcomes, full_differential=True)),
        ("must-not-miss recall@3", lambda: must_not_miss_recall_at_3(outcomes, critical)),
        (
            "dangerous false-negative rate",
            lambda: dangerous_false_negative_rate(outcomes, critical),
        ),
    ]
    if any(o.flagged for o in outcomes):
        metrics += [
            ("red-flag sensitivity", lambda: red_flag_sensitivity(outcomes, critical)),
            ("red-flag precision", lambda: red_flag_precision(outcomes)),
        ]
    if outcomes and all(o.probabilities is not None for o in outcomes):
        metrics.append(("Brier score", lambda: brier_score(outcomes)))
    results = []
    for name, build in metrics:
        ratio = build()
        low, high = bootstrap_ratio(ratio, resamples=resamples, seed=seed, level=level)
        results.append(MetricResult(name, ratio.value, low, high, ratio.cases))
    return results
