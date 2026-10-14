"""Guardrails v1: every sentence of an explanation traces to the result, and citations link to PMC
only (task 3c; FR-8.2, FR-8.3, FR-7.3).

The pipeline runs :func:`check_explanation` after the explainer and before ``safety_check``
(src/pipeline.py), whatever wrote the explanation: the template explainer now, and later a grounded
LLM (3b), which falls back to the template whenever this check finds anything.

**Claim support.** The explanation is split into sentences, as the safety layer splits it, and
each must trace to one *source* the result holds:

* **a candidate**: its label, its graph assessments (supporting, against, missing) and its
  reasoning paths;
* **the red flags**: the flagged candidates, the lines of ``result.red_flags`` and the rules'
  reasons;
* **the model's reasons** for a candidate: its label and the case's own recorded findings, which
  are all a reason can name (src/ml/reasons.py; the result has no field for the reasons
  themselves);
* **a suggested test**: one line of ``suggested_next_tests``;
* **a cited passage**: its excerpt, title and citation, with the candidate it was retrieved for.

A sentence traces to a source when each of its words is a word of that source, a word of the
explanation's frame (:data:`FRAME`: the words that join facts, "supporting", "findings",
"against"...), a stopword, a number, or part of the case's id. A sentence that traces to no
source is listed in ``Explanation.unsupported_claims`` and ``grounded`` becomes False. Its text is
left in place: the claim is flagged, not suppressed (FR-8.3 allows either). The template
explainer builds its sentences from these sources, so they always trace, and a test holds it to
that.

**Citations.** :func:`citation_link` turns ``PMCID:PMC1234567`` into its page on PMC and refuses
anything else, so no link leads to another host. :func:`cited_sources` lists the result's cited
passages once each, in rank order. An explanation may cite only those: a citation the retriever
never returned is one the explainer made up, so it is removed and listed as unsupported. A sentence
that traces only to a passage adds that passage's citation.

**Limits.** v1 is lexical, like the safety layer v1. It catches a word the result does not hold: a
condition, a finding, a test, a drug, a unit. It does not catch a wrong relation between words it
does hold. "Supporting findings: chest pain" passes for any candidate, because chest pain is a
recorded finding and so in every candidate's model-reasons source. A negation ("no chest pain")
of a recorded finding passes as well. Words are matched after folding case and a plural "s", so
a paraphrase ("radiating" for "radiation") does not trace. An LLM's text then falls back to the
template, which is the safe direction.

Reference: docs/01-srs.md FR-7.3, FR-8.2, FR-8.3; docs/02-architecture.md §3; docs/17 T5.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass

from src.contracts import DiagnosisResult, Explanation, PatientCase
from src.rag.corpus import article_url, normalise_pmcid

logger = logging.getLogger(__name__)

__all__ = [
    "FRAME",
    "STOPWORDS",
    "UNCITED",
    "UNTRACED",
    "Citation",
    "check_explanation",
    "citation_link",
    "cited_sources",
    "sentences",
    "untraced_sentences",
]

UNTRACED = "not traced to the result (3c): "
"""The prefix of an unsupported sentence in ``unsupported_claims``, beside the safety layer's
"treatment advice removed (FR-6.5): "."""

UNCITED = "citation not among the retrieved passages (3c): "

_SENTENCE = re.compile(r"(?<=[.!])\s+")  # the safety layer's, so both see one sentence alike
_WORD = re.compile(r"[^\W_]+")


def _word_list(text: str) -> frozenset[str]:
    return frozenset(text.split())


STOPWORDS = _word_list(
    """a about above after all also an and any are as at be because been before being both but by
    can could did do does each either for from had has have having he her his how i if in into is
    it its itself may might more most much must neither nor of on once one or other our out over
    own same she should so some such than that the their them then there these they this those
    through to too under until up very was we were what when where whether which while who whom
    why will with within without would you your"""
)
"""Function words, which carry no claim of their own."""

FRAME = _word_list(
    """against alone candidate case clinician condition confirm consider decide decides decision
    denied differential direction evidence exclude finding first flag flagged generated graph help
    knowledge leading list listed model no not only order passage patient raised ranked reason
    recorded red removed retrieved second strongest support supported supporting supports third
    top"""
)
"""The words an explanation uses to join facts: the template explainer's, and a few an LLM needs
to say the same thing ("the patient", "the differential"). None of them names a condition, a
finding, a test or a treatment, so none can carry a medical claim on its own."""


def sentences(text: str) -> list[str]:
    """``text`` split into sentences, as the safety layer splits it (src/reasoning/safety.py)."""
    return [s.strip() for s in _SENTENCE.split(text) if s.strip()]


def _fold(word: str) -> str:
    word = word.lower()
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _words(*texts: str | None) -> frozenset[str]:
    return frozenset(_fold(w) for text in texts if text for w in _WORD.findall(text))


_FREE = frozenset(_fold(w) for w in STOPWORDS | FRAME)


@dataclass(frozen=True)
class _Source:
    kind: str
    words: frozenset[str]
    citation: str | None = None


def _sources(result: DiagnosisResult, case: PatientCase | None) -> list[_Source]:
    recorded = [f.label for f in [*case.findings, *case.risk_factors]] if case else []
    sources: list[_Source] = []
    for c in result.candidates:
        sources.append(
            _Source(
                "candidate",
                _words(
                    c.label,
                    *(a.label for a in c.assessments),
                    *(step for p in c.paths for step in p.path),
                ),
            )
        )
    flagged = [c for c in result.candidates if c.red_flag]
    sources.append(
        _Source(
            "red_flags",
            _words(
                *result.red_flags,
                *(c.label for c in flagged),
                *(c.red_flag_reason for c in flagged),
            ),
        )
    )
    sources += [_Source("model_reasons", _words(c.label, *recorded)) for c in result.candidates]
    sources += [_Source("test", _words(line)) for line in result.suggested_next_tests]
    sources += [
        _Source("passage", _words(c.label, e.passage, e.title, e.citation), e.citation)
        for c in result.candidates
        for e in c.evidence
    ]
    return sources


def _claim(sentence: str, free: frozenset[str]) -> frozenset[str]:
    """The words of ``sentence`` that say something: not free, not a number, not a single letter
    (the "s" of "the model's")."""
    return frozenset(
        w for w in _words(sentence) if w not in free and not w.isdigit() and len(w) > 1
    )


def _traced_by(claim: frozenset[str], sources: list[_Source]) -> list[_Source]:
    return [s for s in sources if claim <= s.words]


def untraced_sentences(
    text: str, result: DiagnosisResult, case: PatientCase | None = None
) -> list[str]:
    """The sentences of ``text`` that trace to no source of ``result`` (see the module docstring)."""
    sources = _sources(result, case)
    free = _FREE | _words(result.case_id)
    out = []
    for sentence in sentences(text):
        claim = _claim(sentence, free)
        if claim and not _traced_by(claim, sources):
            out.append(sentence)
    return out


def citation_link(citation: str) -> str | None:
    """The PMC page of ``PMCID:PMC1234567`` (or of a bare ``PMC1234567``), else None.

    Never a link to any other host: a citation that is not a PMC identifier gets no link.
    """
    pmcid = normalise_pmcid(citation.strip().removeprefix("PMCID:"))
    return article_url(pmcid) if pmcid else None


@dataclass(frozen=True)
class Citation:
    """One cited article, as it is shown: its identifier, title and PMC page."""

    citation: str
    title: str
    url: str | None


def cited_sources(candidates: Iterable) -> list[Citation]:
    """The passages the candidates cite, each article once, in rank order.

    Args:
        candidates: Anything with ``evidence`` (a list of :class:`~src.contracts.Evidence`), in the
            order they are shown.

    Returns:
        One :class:`Citation` per distinct citation, at its first appearance. Its ``url`` is
        :func:`citation_link`'s, never the evidence's own, so it can only lead to PMC.
    """
    seen: set[str] = set()
    out: list[Citation] = []
    for candidate in candidates:
        for evidence in candidate.evidence:
            if evidence.citation in seen:
                continue
            seen.add(evidence.citation)
            out.append(
                Citation(evidence.citation, evidence.title, citation_link(evidence.citation))
            )
    return out


def check_explanation(result: DiagnosisResult, case: PatientCase | None = None) -> DiagnosisResult:
    """``result`` with its explanation checked: claims traced, citations limited to the passages.

    Every sentence that traces to no source is added to ``unsupported_claims`` (prefixed
    :data:`UNTRACED`), and every citation that is not one of the result's passages is removed and
    added there too (prefixed :data:`UNCITED`); either makes ``grounded`` False. A sentence that
    traces only to cited passages adds their citations. Claims already listed, by the explainer or
    an earlier check, are kept. The text is not changed.

    Args:
        result: The pipeline's result, explanation included, before ``safety_check``.
        case: The case it answers. Its recorded findings are what the model's reasons can name.

    Returns:
        A copy of ``result`` with the checked explanation, or ``result`` itself when it has no
        explanation or nothing changed.
    """
    explanation = result.explanation
    if explanation is None:
        return result
    sources = _sources(result, case)
    free = _FREE | _words(result.case_id)
    untraced: list[str] = []
    cited: list[str] = []
    for sentence in sentences(explanation.text):
        claim = _claim(sentence, free)
        if not claim:
            continue  # it joins facts or addresses the clinician; it claims nothing
        traced = _traced_by(claim, sources)
        if not traced:
            untraced.append(sentence)
        elif all(s.kind == "passage" for s in traced):
            cited += [s.citation for s in traced if s.citation and s.citation not in cited]

    retrieved = {e.citation for c in result.candidates for e in c.evidence}
    kept = [c for c in dict.fromkeys(explanation.citations) if c in retrieved]
    invented = [c for c in explanation.citations if c not in retrieved]
    citations = kept + [c for c in cited if c not in kept]

    if not untraced and not invented and citations == explanation.citations:
        return result
    if untraced or invented:
        logger.warning(
            "case %s: %d sentence(s) traced to nothing and %d citation(s) not retrieved (3c)",
            result.case_id,
            len(untraced),
            len(invented),
        )
    checked: Explanation = explanation.model_copy(
        update={
            "grounded": explanation.grounded and not untraced and not invented,
            "unsupported_claims": [
                *explanation.unsupported_claims,
                *(UNTRACED + s for s in untraced),
                *(UNCITED + c for c in invented),
            ],
            "citations": citations,
        }
    )
    return result.model_copy(update={"explanation": checked})
