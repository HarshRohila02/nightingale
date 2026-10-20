"""Claim-level measures of an explanation, for H3 (docs/05 §3.4; EXP-009, EXP-010).

An explanation is judged by the claim check of guardrails v1 (``src/reasoning/guardrails.py``):
each of its sentences that says something must trace to one source the system was *given*. This
module counts the outcome per explanation (:func:`claim_stats`) and compares systems on the
pooled rate (:func:`pooled_rate`, :func:`paired_rate_difference`), so that every figure carries a
bootstrap interval over **cases**, never over sentences: the sentences of one explanation are not
independent.

**What a claim is.** The unit is guardrails v1's: a sentence with at least one word that is not a
stopword, a frame word, a number or part of the case id. A sentence with none (``"The patient has
a first-ranked condition."``) claims nothing and is not counted. An **unsupported** claim is a
sentence that traces to no single source (the check is lexical: it catches a word the system was
not given, not a wrong relation between words it was given; the module docstring of guardrails
says so).

**What a source is** depends on the system, because the question is "unsupported by what it was
given": the case's recorded findings and its age and sex, the candidates' labels, and, for a system
that has them, the graph's assessments, the red flags, the tests and the passages (for B3 and B4,
:func:`src.llm.baselines.given_result`; for the grounded explainer, the pipeline's own result).

Private helpers of guardrails (``_claim``, ``_words``, ``_FREE``) are imported rather than copied,
so that the count of claims can never drift from the check that judges them; a test pins them.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from src.contracts import DiagnosisResult, PatientCase
from src.eval.metrics import DEFAULT_LEVEL, DEFAULT_RESAMPLES, DEFAULT_SEED, Ratio
from src.reasoning.guardrails import (
    _FREE,
    _claim,
    _words,
    sentences,
    untraced_sentences,
)
from src.reasoning.safety import recommends_treatment

__all__ = [
    "ClaimStats",
    "citations_in",
    "claim_stats",
    "paired_rate_difference",
    "pooled_rate",
    "rate_interval",
]

_CITATION = re.compile(r"PMC\d+")


def citations_in(text: str) -> list[str]:
    """The PMC identifiers ``text`` cites, as ``PMCID:PMC…``, each once, in order. A bare
    ``PMC123`` counts as a citation too."""
    return list(dict.fromkeys(f"PMCID:{m}" for m in _CITATION.findall(text)))


@dataclass(frozen=True)
class ClaimStats:
    """The claim check's outcome for one explanation.

    Attributes:
        sentences: All sentences.
        claims: The sentences that say something (the denominator of every rate here).
        unsupported: How many of them trace to no source.
        untraced: Those sentences, as written.
        cited: The PMC identifiers the text cites.
        invalid_citations: Those among them that are not passages the system was shown.
        cited_sentences: Sentences that cite something.
        cited_unsupported: Of those, how many are unsupported.
        treatment_language: Whether the text has treatment language outside the known labels
            (the safety layer's lexicon, FR-6.5).
    """

    sentences: int
    claims: int
    unsupported: int
    untraced: tuple[str, ...]
    cited: tuple[str, ...]
    invalid_citations: tuple[str, ...]
    cited_sentences: int
    cited_unsupported: int
    treatment_language: bool

    @property
    def supported(self) -> int:
        return self.claims - self.unsupported

    def as_dict(self) -> dict[str, object]:
        return {
            "sentences": self.sentences,
            "claims": self.claims,
            "unsupported": self.unsupported,
            "untraced": list(self.untraced),
            "cited": list(self.cited),
            "invalid_citations": list(self.invalid_citations),
            "cited_sentences": self.cited_sentences,
            "cited_unsupported": self.cited_unsupported,
            "treatment_language": self.treatment_language,
        }


def claim_stats(text: str, result: DiagnosisResult, case: PatientCase | None = None) -> ClaimStats:
    """Check ``text`` against what a system was given.

    Args:
        text: The explanation, as the model wrote it (whitespace is the caller's to collapse).
        result: What the system was given: candidates (labels, assessments, paths, passages), red
            flags and tests. Its ``explanation`` is not read.
        case: The case, whose recorded findings a sentence may mention.
    """
    free = _FREE | _words(result.case_id)
    parts = sentences(text)
    untraced = set(untraced_sentences(text, result, case))
    claims = [s for s in parts if _claim(s, free)]
    bad = [s for s in claims if s in untraced]
    retrieved = {e.citation for c in result.candidates for e in c.evidence}
    cited = citations_in(text)
    cited_in = [s for s in claims if _CITATION.search(s)]
    labels = [c.label for c in result.candidates]
    labels += [a.label for c in result.candidates for a in c.assessments]
    if case is not None:
        labels += [f.label for f in [*case.findings, *case.risk_factors]]
    return ClaimStats(
        sentences=len(parts),
        claims=len(claims),
        unsupported=len(bad),
        untraced=tuple(bad),
        cited=tuple(cited),
        invalid_citations=tuple(c for c in cited if c not in retrieved),
        cited_sentences=len(cited_in),
        cited_unsupported=sum(s in untraced for s in cited_in),
        treatment_language=recommends_treatment(text, labels),
    )


def pooled_rate(unsupported: Sequence[int], claims: Sequence[int]) -> Ratio:
    """Per-case unsupported claims over per-case claims: the pooled rate, with the case as the
    unit a bootstrap resamples. A case with no claim contributes nothing to either side."""
    if len(unsupported) != len(claims):
        raise ValueError("one count of unsupported claims and one of claims per case")
    return Ratio(np.asarray(unsupported, dtype=float), np.asarray(claims, dtype=float))


def paired_rate_difference(
    a: Ratio,
    b: Ratio,
    *,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    level: float = DEFAULT_LEVEL,
) -> dict[str, float | bool | int | str | None]:
    """The pooled rate of ``a`` minus ``b``, and its percentile interval from resampling the
    cases, the same cases for both (docs/05 §6's paired bootstrap, here for a ratio of sums).

    Returns:
        ``difference`` (a − b), ``ci_low``, ``ci_high``, ``excludes_zero``, ``higher`` (``"a"``
        when the interval is above 0, ``"b"`` when below, else None: the system whose unsupported
        rate is the higher, that is the worse), and ``cases``.

    Raises:
        ValueError: for unequal or no cases, or a side with no claim at all.
    """
    if len(a.numerator) != len(b.numerator):
        raise ValueError("a paired test needs the same cases for both systems")
    n = len(a.numerator)
    if n == 0 or not a.denominator.sum() or not b.denominator.sum():
        raise ValueError("a paired rate difference needs claims on both sides")
    rng = np.random.default_rng(seed)
    values = np.empty(resamples)
    done = 0
    chunk = max(1, min(resamples, 20_000_000 // n))
    while done < resamples:
        size = min(chunk, resamples - done)
        index = rng.integers(0, n, size=(size, n), dtype=np.int32)
        with np.errstate(invalid="ignore", divide="ignore"):
            rate_a = a.numerator[index].sum(axis=1) / a.denominator[index].sum(axis=1)
            rate_b = b.numerator[index].sum(axis=1) / b.denominator[index].sum(axis=1)
        values[done : done + size] = rate_a - rate_b
        done += size
    tail = (1 - level) / 2
    low, high = np.nanquantile(values, [tail, 1 - tail])
    difference = a.value - b.value
    return {
        "difference": difference if math.isfinite(difference) else math.nan,
        "ci_low": float(low),
        "ci_high": float(high),
        "excludes_zero": bool(low > 0 or high < 0),
        "higher": "a" if low > 0 else "b" if high < 0 else None,
        "cases": n,
    }


def rate_interval(
    unsupported: Iterable[int],
    claims: Iterable[int],
    *,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = DEFAULT_SEED,
    level: float = DEFAULT_LEVEL,
) -> tuple[float, float, float]:
    """``(rate, low, high)``: a pooled rate with its percentile interval over cases."""
    from src.eval.metrics import bootstrap_ratio

    ratio = pooled_rate(list(unsupported), list(claims))
    low, high = bootstrap_ratio(ratio, resamples=resamples, seed=seed, level=level)
    return ratio.value, low, high
