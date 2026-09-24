"""Scoring a case against the knowledge graph (task 2a, EXP-005).

The score is a **naive-Bayes log-likelihood**. For each condition ``c``::

    score(c) = sum over present findings k of  log P(k | c)
             + sum over denied findings k  of  log(1 - P(k | c))

with a uniform prior, so ``score`` ranks conditions exactly as their posterior does. It replaces
the weighted overlap ``(matched - 0.5 * denied) / total``, which had three defects (docs/02 §5.1,
EXP-014 to EXP-016):

1. **It never counted a finding against a condition that cannot explain it.** Stable angina's
   evidence set sits inside unstable angina's, so a stable picture *plus rest pain* scored stable
   1.00 and unstable 0.875, although rest pain defines unstable angina. Here rest pain is
   unexplained by stable angina and costs it ``log(LEAK)``.
2. **It divided by everything a condition might show,** so the four conditions BODHI-S enriches
   were punished for knowing more (EXP-016: MI's graph-only top-1 fell from 0.857 to 0.602). Here
   a finding the case does not mention contributes nothing, to any condition. That is the same
   rule EXP-017 forced on the ML model: not asked is not denied.
3. **It could not tell a denial from silence** except through a fixed half-penalty.

**The graph mixes two vocabularies, so it is closed under the crosswalk first** (docs/02 §5.2).
DDXPlus links its conditions to *questions* (``DDX:E_55``, "where is your pain?"); the
hand-authored aortic dissection and the BODHI-S facts link to *answers*
(``SYM:chest_pain``). A likelihood reads a missing edge as "this condition never shows this
finding", so without the closure every chest-pain patient's ``SYM:chest_pain`` would hand its
weight to aortic dissection, the one condition that names it. Two rules close the graph, both
from the table cases are already expanded with:

* **Upward** (:attr:`EdgeKind.IMPLIED`). Answers imply their question and its parents, as
  :func:`~src.medical_kg.crosswalk.expand_case` infers for a case. A condition with answer-level
  edges ``w1, w2, ...`` implying question ``q`` lists ``q`` with ``1 - prod(1 - w)``: the
  probability of giving at least one of those answers (noisy-OR). The maximum would be a lower
  bound, and it under-credited dissection on the generic questions every chest-pain patient
  answers, which let MI edge past it on GC-003.
* **Downward** (:attr:`EdgeKind.IMPUTED`). A condition that asks ``q`` but whose answer to it the
  graph does not state gets each answer-level concept on ``q`` at the **mean** of the conditions
  that do state it, capped by its own ``P(q | c)`` when the answer implies the question.
  Ignorance is "average", neither "impossible" nor "certain".

**The price of the downward rule is limitation 1 of the KG card, made explicit.** An answer that
only one condition states — tearing pain, stated only for aortic dissection — is imputed at that
condition's own value for every other condition that asks about pain character, so it cannot
separate them. The graph does not know how rarely pericarditis tears; the score must not invent
it. Separating them is the job of answer-level enrichment, the red-flag rules and the ML ranker.

**Constants, both fixed before any measurement** (EXP-005 reports a sensitivity check on the
hand-written cases, never on validate):

* :data:`CAP` = 0.9 — the middle of the top likelihood band. DDXPlus's weight 1.0 means "listed,
  frequency unknown", and ``log(1 - 1.0)`` would make one denial infinitely decisive.
* :data:`LEAK` = 0.01 — for a present finding the condition has no edge to at all. It must stay
  below the lowest band (rare, middle 0.03): an unassociated finding is less likely than a rare
  associated one. Every verdict in EXP-005 holds from 0.001 to 0.01; at 0.03 they start to move.

**A fact is counted once.** A case expanded through the crosswalk holds an answer together with
the questions it implies: *chest pain* arrives with "where is your pain?" (``DDX:E_55``) and
"pain anywhere?" (``DDX:E_53``). The answer entails them, so P(answer and its questions | c) is
P(answer | c), and the entailed questions are dropped whenever the answer is itself a usable
concept. A denial likewise entails the denial of a yes/no question no broader than it: denying
exertional pain denies ``DDX:E_218``. Counting both would score one finding up to three times: in
review, chest pain alone cost atrial fibrillation ``3 * log(LEAK)``. That inflated penalty became
the floor of the pipeline's min-max rescaling and compressed every other condition's graph score;
counting once moves GC-003's dissection from 4th to 3rd in the fused ranking with red flags off
(EXP-005). When the answer is *not* a usable concept (``SYM:diaphoresis`` lives on the
``DDX:E_50`` node), the question is its only carrier, and it stays.

A concept that no condition in the graph knows is skipped, since it cannot distinguish anything.
A concept asserted both present and denied is contradictory input, and neither side is used.

Reference: docs/02-architecture.md §5.1 (KG card), docs/08-experiment-log.md EXP-005.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import Enum

import numpy as np

from src.contracts import Assertion, PatientCase
from src.ddxplus import CONCEPT_PREFIX
from src.medical_kg.crosswalk import (
    BY_CONCEPT,
    CONCEPT_IMPLIES_ANSWER,
    DENIAL_CARRIES_OVER,
    PARENT_QUESTION,
    Match,
)

__all__ = ["CAP", "LEAK", "Contribution", "EdgeKind", "LikelihoodScorer"]

CAP = 0.9
"""Highest P(finding | condition) the score uses. See the module docstring."""

LEAK = 0.01
"""P(finding | condition) for a present finding the condition has no edge to. See the module
docstring; it must stay below the rare band's 0.03."""

ROUND = 12
"""Decimals kept, so conditions the graph cannot tell apart tie exactly instead of being ordered
by floating-point noise. The pipeline breaks exact ties by registry order."""

_NO_IMPUTATION = frozenset({Match.RELATED, Match.NONE})
"""Crosswalk matches too loose to say that a condition asking the question might give the answer."""


class EdgeKind(str, Enum):
    """Where the P(finding | condition) behind a contribution came from."""

    EXPLICIT = "explicit"
    """An edge the graph holds, from DDXPlus, BODHI-S or the hand-authored facts."""
    IMPLIED = "implied"
    """A question implied by answers the condition lists (upward, noisy-OR)."""
    IMPUTED = "imputed"
    """An answer to a question the condition asks, at the stating conditions' mean (downward)."""
    NONE = "none"
    """No edge: the condition does not explain the finding, which costs ``log(LEAK)``."""


@dataclass(frozen=True)
class Contribution:
    """One asserted finding's part in one condition's score. They sum to the score (which is
    rounded to ``ROUND`` decimals, so to within 1e-12)."""

    concept_id: str
    assertion: Assertion
    probability: float
    """The P(finding | condition) used, after the cap. LEAK for an unexplained present finding."""
    log_likelihood: float
    edge: EdgeKind


class LikelihoodScorer:
    """The naive-Bayes score over a crosswalk-closed graph. Immutable after construction.

    Args:
        expected: ``{condition id: {concept id: weight}}``, the explicit edges, with parallel
            edges from different sources already collapsed to the strongest.
        concepts: Every concept node in the graph, including any no condition links to.
    """

    def __init__(
        self, expected: Mapping[str, Mapping[str, float]], concepts: Iterable[str]
    ) -> None:
        self.conditions: tuple[str, ...] = tuple(expected)
        self.concepts: tuple[str, ...] = tuple(sorted(set(concepts)))
        self._column = {k: j for j, k in enumerate(self.concepts)}
        explicit = {c: {k: float(w) for k, w in expected[c].items()} for c in self.conditions}
        edges, kinds = self._close(explicit)

        weights = np.zeros((len(self.conditions), len(self.concepts)))
        for i, c in enumerate(self.conditions):
            for k, w in edges[c].items():
                weights[i, self._column[k]] = w
        self._kinds = kinds
        self._known = weights.sum(axis=0) > 0
        self._probability = np.where(weights > 0, np.minimum(weights, CAP), LEAK)
        self._present = np.log(self._probability)
        # A denied finding the condition has no edge to costs log(1 - 0) = 0: denying something a
        # condition never shows says nothing against it.
        self._denied = np.log1p(-np.minimum(weights, CAP))

    # -- scoring ------------------------------------------------------------ #

    def score(self, case: PatientCase) -> dict[str, float]:
        """{condition id: log-likelihood of the case's asserted findings}. Higher = more likely.

        Every condition gets a score. Findings that are unknown, unmentioned, contradictory or
        unknown to the whole graph contribute nothing.
        """
        present, denied = self._columns(case)
        values = self._present[:, present].sum(axis=1) + self._denied[:, denied].sum(axis=1)
        return {
            c: round(float(v), ROUND) + 0.0 for c, v in zip(self.conditions, values, strict=True)
        }

    def contributions(self, case: PatientCase, condition_id: str) -> list[Contribution]:
        """Each counted finding's signed part in ``condition_id``'s score, largest first.

        The parts sum to :meth:`score` for that condition, to within its rounding. A question
        entailed by an answer in the case is not listed: it is not counted. An empty list for an
        unknown condition.
        """
        if condition_id not in self.conditions:
            return []
        row = self.conditions.index(condition_id)
        present, denied = self._columns(case)
        parts = []
        for columns, assertion, table in (
            (present, Assertion.PRESENT, self._present),
            (denied, Assertion.ABSENT, self._denied),
        ):
            for j in columns:
                concept = self.concepts[j]
                kind = self._kinds.get((condition_id, concept), EdgeKind.NONE)
                probability = float(self._probability[row, j])
                if assertion is Assertion.ABSENT and kind is EdgeKind.NONE:
                    probability = 0.0
                parts.append(
                    Contribution(concept, assertion, probability, float(table[row, j]), kind)
                )
        return sorted(parts, key=lambda p: (-abs(p.log_likelihood), p.concept_id))

    def edge_kind(self, condition_id: str, concept_id: str) -> EdgeKind:
        """How the closed graph links a condition to a concept."""
        return self._kinds.get((condition_id, concept_id), EdgeKind.NONE)

    def probability(self, condition_id: str, concept_id: str) -> float:
        """The P(concept | condition) the score uses: capped, or LEAK when there is no edge."""
        return float(
            self._probability[self.conditions.index(condition_id), self._column[concept_id]]
        )

    # -- internals ---------------------------------------------------------- #

    def _columns(self, case: PatientCase) -> tuple[list[int], list[int]]:
        """The concepts that count, as columns: usable, not contradictory, not entailed."""
        present, absent = case.present_concept_ids(), case.absent_concept_ids()
        contradictory = present & absent

        def usable(ids: set[str]) -> set[str]:
            return {
                k for k in ids - contradictory if k in self._column and self._known[self._column[k]]
            }

        present, absent = usable(present), usable(absent)
        present -= {q for k in present for q in _entailed_by_presence(k)}
        absent -= {q for k in absent for q in _entailed_by_denial(k)}
        return sorted(self._column[k] for k in present), sorted(self._column[k] for k in absent)

    def _close(
        self, explicit: Mapping[str, Mapping[str, float]]
    ) -> tuple[dict[str, dict[str, float]], dict[tuple[str, str], EdgeKind]]:
        """The explicit edges plus the implied and imputed ones. See the module docstring."""
        edges = {c: dict(explicit[c]) for c in self.conditions}
        kinds = {(c, k): EdgeKind.EXPLICIT for c in self.conditions for k in explicit[c]}

        for c in self.conditions:  # upward: noisy-OR over the answers implying each question
            miss: dict[str, float] = {}
            for k, w in explicit[c].items():
                entry = BY_CONCEPT.get(k)
                if entry is None or entry.pattern is None:
                    continue
                if entry.match not in CONCEPT_IMPLIES_ANSWER:
                    continue
                for q in _question_chain(entry.pattern.code):
                    if q in self._column and q not in explicit[c]:
                        miss[q] = miss.get(q, 1.0) * (1.0 - w)
            for q, m in miss.items():
                edges[c][q] = 1.0 - m
                kinds[(c, q)] = EdgeKind.IMPLIED

        for k in self.concepts:  # downward: the stating conditions' mean, for the others asking
            entry = BY_CONCEPT.get(k)
            if entry is None or entry.pattern is None or entry.match in _NO_IMPUTATION:
                continue
            stated = [explicit[c][k] for c in self.conditions if k in explicit[c]]
            if not stated:
                continue
            mean = float(np.mean(stated))
            question = f"{CONCEPT_PREFIX}{entry.pattern.code}"
            capped = entry.match in CONCEPT_IMPLIES_ANSWER  # the answer implies the question
            for c in self.conditions:
                if k in explicit[c] or question not in edges[c]:
                    continue
                edges[c][k] = min(mean, edges[c][question]) if capped else mean
                kinds[(c, k)] = EdgeKind.IMPUTED
        return edges, kinds


def _entailed_by_presence(concept: str) -> list[str]:
    """The questions a present answer implies: its own and their parents, as ``expand_case`` adds."""
    entry = BY_CONCEPT.get(concept)
    if entry is None or entry.pattern is None or entry.match not in CONCEPT_IMPLIES_ANSWER:
        return []
    return _question_chain(entry.pattern.code)


def _entailed_by_denial(concept: str) -> list[str]:
    """The yes/no question a denial carries over to, as ``expand_case`` adds. Never its parents:
    denying radiation to the jaw does not deny pain."""
    entry = BY_CONCEPT.get(concept)
    if (
        entry is None
        or entry.pattern is None
        or entry.match not in DENIAL_CARRIES_OVER
        or not entry.pattern.whole_question
    ):
        return []
    return [f"{CONCEPT_PREFIX}{entry.pattern.code}"]


def _question_chain(code: str) -> list[str]:
    """``E_55`` -> ``["DDX:E_55", "DDX:E_53"]``: a question and the questions it follows up."""
    chain = [code]
    while chain[-1] in PARENT_QUESTION:
        chain.append(PARENT_QUESTION[chain[-1]])
    return [f"{CONCEPT_PREFIX}{c}" for c in chain]
