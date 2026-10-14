"""What to ask next: the findings whose answer would most change the graph's ranking (3d; FR-5.5).

The second half of "what to ask next". The first, :mod:`src.reasoning.investigations`, is a cited
table of the tests to consider for each condition. This half is computed from the knowledge graph:
of the findings the clinician can enter (the hand-authored ``SYM:*`` / ``RF:*`` concepts the form
offers, :data:`~src.medical_kg.crosswalk.BY_CONCEPT`) that the case has neither recorded nor
denied, which would most reduce the graph's uncertainty over the 14 conditions?

**The measure is the expected entropy reduction** (the expected information gain) of the graph's
posterior, under the same naive-Bayes likelihoods that rank the case
(:class:`~src.medical_kg.scoring.LikelihoodScorer`, docs/02 §5.1)::

    p(c)        the graph's posterior now: a softmax of its log-likelihoods (uniform prior),
                exactly the P_kg the fusion reads (src/fusion/pool.py)
    θ(c)        P(the finding is present | c), from the scorer (below)
    q           Σ_c p(c) θ(c), the predictive probability of "present"
    gain        H(p) − [q · H(p | present) + (1 − q) · H(p | denied)]      in bits

which equals the mutual information between the condition and the answer, so it is never negative,
and it is 0 exactly when every condition the posterior allows expects the same answer.

**θ comes from the scorer, through the crosswalk, exactly as the case would be scored.** For each
candidate the case is scored twice more, once with the finding recorded present and once denied,
after the crosswalk expansion the API and the dashboard apply (``expand_case``, mirrored on id sets
here and checked against it by a test), so the answer is judged by the arithmetic that would rank
it: the questions it implies, the entailed questions counted once, contradictions dropped. The two
scores' increments are the likelihoods of the two answers, L⁺(c) and L⁻(c), and
θ(c) = L⁺(c) / (L⁺(c) + L⁻(c)). **In the usual case, a finding that lands on one graph node the
condition has an edge to, L⁺ + L⁻ = 1, so θ is the scorer's own P(finding | c) and the posterior
after either answer is the one the graph would then show.** Three cases are not usual, and there
the model is the closest coherent reading of the scorer, not the scorer itself: θ keeps, for each
condition, the scorer's likelihood ratio between the two answers, but not its exact posterior
after them.

* A condition with no edge to the finding: the scorer charges LEAK for "present" and nothing for
  "denied", so θ = LEAK / (1 + LEAK) ≈ 0.0099 instead of LEAK.
* A finding the crosswalk carries one way only (``SYM:exertional`` without a node of its own:
  denying it denies DDXPlus's ``E_218``, asserting it implies nothing).
* An answer that replaces questions the case already counts: chest pain entails "pain anywhere?",
  which a back-pain answer without a node of its own had counted, so the increment is a ratio.

**What is never asked.** A concept the case records, present or denied; one the record already
settles through the crosswalk: unilateral leg swelling settles leg swelling, a denied leg swelling
settles both of its kinds, epigastric pain settles abdominal pain (:func:`settled`; only
inferences the crosswalk's match types allow are drawn, as in ``expand_case``); and one that
cannot apply to the patient's recorded sex (:data:`NEVER_ASKED_OF`: pregnancy, of a man).

**Fixed before any data was looked at; nothing is tuned.** The measure, the units (bits), the order
(gain to 10 significant digits, then concept id), and the display constants below. It reads no validation data, and no model: the ML ranker is not consulted, so a
question that would only separate what the model already separates scores as the graph sees it.

**Limits.** (1) The posterior is the graph's alone, a naive Bayes over question-level knowledge
(docs/02 §5.1), so it is often near-certain and its gains small; the order matters, not the
number, and the number is not a probability of anything. (2) Every condition counts the same: a
question that would rule out a must-not-miss condition is not preferred for that; the red-flag
rules and the tests table (every red-flagged candidate gets its tests) carry that weight. (3) One
question at a time (myopic): two findings that only matter together each look weak. (4) The
findings are answered as independent, given the condition (naive Bayes), so a finding correlated
with one already recorded can look more informative than it is. (5) The stub graph has no
likelihoods; its questions come from a naive-Bayes view of the stub map
(:attr:`src.stubs.InMemoryGraphStore.scorer`), a toy like the map.

**A degraded graph gives no questions** rather than a guess: a store without a likelihood view, or
one that fails, returns an empty :class:`QuestionRanking` whose ``entropy_bits`` is None.

The pipeline's result has no field for questions (``src/contracts.py`` needs all four team members
to change), so the API serves them at ``POST /next-questions`` and the dashboard calls
:func:`next_questions` directly.

Reference: docs/01-srs.md FR-5.5, docs/02-architecture.md §5.3.
"""

from __future__ import annotations

import logging
import math
import weakref
from collections.abc import Collection, Iterable
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field

from src.conditions import BY_ID
from src.contracts import Assertion, PatientCase, Sex
from src.ddxplus import CONCEPT_PREFIX
from src.medical_kg.crosswalk import (
    ANSWER_IMPLIES_CONCEPT,
    BY_CONCEPT,
    CONCEPT_IMPLIES_ANSWER,
    DENIAL_CARRIES_OVER,
    DERIVED_SOURCE,
    PARENT_QUESTION,
    CrosswalkEntry,
    Pattern,
)

logger = logging.getLogger(__name__)

__all__ = [
    "EXAMINATION",
    "NEVER_ASKED_OF",
    "QUESTIONS_SHOWN",
    "SEPARATES_SHARE",
    "SEPARATES_SHOWN",
    "NextQuestion",
    "QuestionRanker",
    "QuestionRanking",
    "Separated",
    "next_questions",
    "settled",
]

QUESTIONS_SHOWN = 5
"""How many questions are returned: a short list a clinician can take in at a glance."""

SEPARATES_SHOWN = 3
"""How many conditions a question names on each side ("a yes favours …", "a no favours …")."""

SEPARATES_SHARE = 0.01
"""A condition is named only if it carries at least this share of the question's gain, so a
condition the posterior has all but excluded is not named for a rounding error's worth of it."""

SIGNIFICANT = 10
"""Significant digits kept of a gain before ordering, so questions the graph cannot tell apart tie
exactly and fall to the concept id rather than to floating-point noise. Significant digits, not
decimals: on a near-certain posterior every gain is tiny, and their order is still the graph's."""

EXAMINATION = frozenset(
    {
        "SYM:interarm_bp_difference",
        "SYM:pulse_deficit",
        "SYM:focal_neuro_deficit",
        "SYM:aortic_regurgitation_murmur",
        "SYM:hypotension",
        "SYM:irregular_pulse",
    }
)
"""Concepts found by examining the patient rather than by asking, as the crosswalk's notes say. The
dashboard says "check" for these and "ask about" for the rest; nothing else differs."""

NEVER_ASKED_OF: dict[str, frozenset[Sex]] = {"RF:pregnancy": frozenset({Sex.MALE})}
"""Concepts that cannot apply to a patient of the recorded sex, so are never suggested for one.
The graph's likelihoods do not know the patient's sex, so without this a man could be asked
whether he is pregnant (found in review on GC-004 recorded as male, where it ranked fifth). A
patient recorded as other is still asked: the record cannot rule it out."""


class Separated(BaseModel):
    """A condition a question bears on."""

    condition_id: str
    label: str


class NextQuestion(BaseModel):
    """One finding worth asking about (or checking), and why.

    Not a contract type: ``DiagnosisResult`` has no field for questions (docs/02 §5.3).
    """

    concept_id: str
    label: str
    kind: Literal["history", "risk_factor", "examination"]
    gain_bits: float = Field(
        description="Expected reduction of the graph's entropy over the conditions, in bits. "
        "Orders the questions; it is not a probability and not calibrated."
    )
    if_present: list[Separated] = Field(
        default_factory=list, description="Conditions a 'present' answer would favour, most first"
    )
    if_denied: list[Separated] = Field(
        default_factory=list, description="Conditions a 'denied' answer would favour, most first"
    )


class QuestionRanking(BaseModel):
    """The questions for one case, best first."""

    questions: list[NextQuestion] = Field(default_factory=list)
    entropy_bits: float | None = Field(
        default=None,
        description="The graph's uncertainty over the conditions now, in bits; None when the "
        "graph could not answer, and then there are no questions.",
    )


# --------------------------------------------------------------------------- #
# The crosswalk, on id sets
# --------------------------------------------------------------------------- #


def _question_chain(code: str) -> frozenset[str]:
    chain = [code]
    while chain[-1] in PARENT_QUESTION:
        chain.append(PARENT_QUESTION[chain[-1]])
    return frozenset(f"{CONCEPT_PREFIX}{c}" for c in chain)


def _implied_by_presence(entry: CrosswalkEntry) -> frozenset[str]:
    """The questions ``expand_case`` adds for a present concept."""
    if entry.pattern is None or entry.match not in CONCEPT_IMPLIES_ANSWER:
        return frozenset()
    return _question_chain(entry.pattern.code)


def _implied_by_denial(entry: CrosswalkEntry) -> frozenset[str]:
    """The question ``expand_case`` denies for a denied concept: a yes/no one, no broader."""
    if (
        entry.pattern is None
        or entry.match not in DENIAL_CARRIES_OVER
        or not entry.pattern.whole_question
    ):
        return frozenset()
    return frozenset({f"{CONCEPT_PREFIX}{entry.pattern.code}"})


_UP = {k: _implied_by_presence(e) for k, e in BY_CONCEPT.items()}
_DOWN = {k: _implied_by_denial(e) for k, e in BY_CONCEPT.items()}


def _expand(
    present: Collection[str], denied: Collection[str], existing: Collection[str]
) -> tuple[frozenset[str], frozenset[str]]:
    """``expand_case`` on id sets: (present, denied) after the crosswalk adds its questions.

    A question implied both present and denied is left out, and a question the case already holds
    is not added again, as ``expand_case`` does. tests/test_next_steps.py checks the two agree.
    """
    up = set().union(*(_UP.get(k, ()) for k in present))
    down = set().union(*(_DOWN.get(k, ()) for k in denied))
    conflict = up & down
    blocked = conflict | set(existing)
    return frozenset(present) | (up - blocked), frozenset(denied) | (down - blocked)


def _contains(inner: Pattern, outer: Pattern) -> bool:
    """Whether every set of answers ``inner`` matches is matched by ``outer`` (same question)."""
    if inner.code != outer.code:
        return False
    if inner.min_ordinal is not None or outer.min_ordinal is not None:
        return (
            inner.min_ordinal is not None
            and outer.min_ordinal is not None
            and inner.min_ordinal >= outer.min_ordinal
        )
    if inner.whole_question or outer.whole_question:
        return inner.whole_question and outer.whole_question
    if outer.side is None:
        return inner.values <= outer.values
    return inner.side == outer.side and inner.values == outer.values


def _implies(k: str, j: str) -> bool:
    """Whether recording ``k`` present makes ``j`` present, by the crosswalk alone.

    ``k`` implies its answers (its match is exact, close or broader), those answers are among
    ``j``'s, and ``j``'s answers imply ``j`` (exact, close or narrower): the same sound inferences
    ``expand_case`` and ``concepts_from_evidences`` draw, chained.
    """
    a, b = BY_CONCEPT.get(k), BY_CONCEPT.get(j)
    if k == j or a is None or b is None or a.pattern is None or b.pattern is None:
        return False
    return (
        a.match in CONCEPT_IMPLIES_ANSWER
        and b.match in ANSWER_IMPLIES_CONCEPT
        and _contains(a.pattern, b.pattern)
    )


_SETTLED_BY_PRESENT = {j: frozenset(k for k in BY_CONCEPT if _implies(k, j)) for j in BY_CONCEPT}
_SETTLED_BY_DENIAL = {j: frozenset(k for k in BY_CONCEPT if _implies(j, k)) for j in BY_CONCEPT}


def settled(concept_id: str, present: Collection[str], denied: Collection[str]) -> bool:
    """Whether the record already answers ``concept_id`` through the crosswalk.

    A recorded present concept that implies it (unilateral leg swelling → leg swelling), or a
    recorded denial of a concept it implies (no leg swelling → no unilateral leg swelling).
    """
    return bool(
        _SETTLED_BY_PRESENT.get(concept_id, frozenset()) & set(present)
        or _SETTLED_BY_DENIAL.get(concept_id, frozenset()) & set(denied)
    )


# --------------------------------------------------------------------------- #
# The ranking
# --------------------------------------------------------------------------- #


def _entropy_bits(p: np.ndarray) -> np.ndarray:
    """Entropy along the last axis, in bits, with 0 · log 0 = 0."""
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(p > 0, -p * np.log2(p), 0.0)
    return terms.sum(axis=-1)


def _softmax(scores: np.ndarray) -> np.ndarray:
    shifted = np.exp(scores - scores.max(axis=-1, keepdims=True))
    return shifted / shifted.sum(axis=-1, keepdims=True)


def _recorded(case: PatientCase) -> tuple[set[str], set[str], set[str]]:
    """(present, denied, every id) of the case as entered: the questions ``expand_case`` added
    (source ``crosswalk`` with a ``crosswalk_from`` qualifier) are dropped and derived again."""
    present: set[str] = set()
    denied: set[str] = set()
    every: set[str] = set()
    for finding in [*case.findings, *case.risk_factors]:
        if finding.source == DERIVED_SOURCE and "crosswalk_from" in finding.qualifiers:
            continue
        every.add(finding.concept_id)
        if finding.assertion is Assertion.PRESENT:
            present.add(finding.concept_id)
        elif finding.assertion is Assertion.ABSENT:
            denied.add(finding.concept_id)
    return present, denied, every


class QuestionRanker:
    """Ranks the questions for cases on one graph. Built once per scorer; immutable.

    Args:
        scorer: The graph's :class:`~src.medical_kg.scoring.LikelihoodScorer` (a store's
            ``scorer``).
    """

    def __init__(self, scorer: Any) -> None:
        self.scorer = scorer
        self.conditions: tuple[str, ...] = tuple(scorer.conditions)
        self._labels = {c: BY_ID[c].label if c in BY_ID else c for c in self.conditions}
        nodes = set(scorer.concepts)
        # A concept can move the score only if it, or a question it implies or denies, is a node.
        self.candidates: tuple[str, ...] = tuple(
            k for k in sorted(BY_CONCEPT) if nodes & ({k} | _UP[k] | _DOWN[k])
        )

    def rank(self, case: PatientCase, limit: int = QUESTIONS_SHOWN) -> QuestionRanking:
        """The ``limit`` questions with the largest expected gain, best first.

        Only questions with a positive gain are returned, ordered by gain (to
        :data:`SIGNIFICANT` significant digits) and then by concept id.
        """
        present, denied, every = _recorded(case)
        candidates = [
            k
            for k in self.candidates
            if k not in present
            and k not in denied
            and not settled(k, present, denied)
            and case.sex not in NEVER_ASKED_OF.get(k, frozenset())
        ]
        base = _expand(present, denied, every)
        p = _softmax(self.scorer.score_many([base])[0])
        entropy = _significant(float(_entropy_bits(p)))
        if not candidates or len(self.conditions) < 2:
            return QuestionRanking(entropy_bits=entropy)

        hypotheses = []
        for k in candidates:
            hypotheses.append(_expand(present | {k}, denied, every | {k}))
            hypotheses.append(_expand(present, denied | {k}, every | {k}))
        # Each answer's increment over the case as it stands: log L⁺ and log L⁻, summed over the
        # columns the answer changes only, so a column every condition shares cancels exactly.
        increments = self.scorer.score_many(hypotheses, relative_to=base)
        # θ = L⁺ / (L⁺ + L⁻): a logistic of the difference between the two answers' increments.
        theta = 1.0 / (1.0 + np.exp(increments[1::2] - increments[0::2]))  # (questions, conditions)
        q = theta @ p  # the predictive probability of "present", per question
        contributions = p * _bernoulli_kl_bits(theta, q[:, None])
        # A question every condition answers alike splits nothing: its gain is 0, exactly.
        splits = theta.max(axis=1) > theta.min(axis=1)
        gains = [
            _significant(float(contributions[i].sum())) if splits[i] else 0.0
            for i in range(len(candidates))
        ]

        order = sorted(
            (i for i in range(len(candidates)) if gains[i] > 0),
            key=lambda i: (-gains[i], candidates[i]),
        )
        questions = [
            self._question(candidates[i], gains[i], theta[i], float(q[i]), contributions[i])
            for i in order[: max(limit, 0)]
        ]
        return QuestionRanking(questions=questions, entropy_bits=entropy)

    def _question(
        self,
        concept_id: str,
        gain: float,
        theta: np.ndarray,
        q: float,
        contributions: np.ndarray,
    ) -> NextQuestion:
        """The question, with the conditions that carry most of its gain, split by which answer
        favours them (θ above or below the predictive probability q)."""
        total = contributions.sum()
        named = sorted(
            (
                i
                for i in range(len(self.conditions))
                if total > 0 and contributions[i] >= SEPARATES_SHARE * total and theta[i] != q
            ),
            key=lambda i: (-contributions[i], i),
        )

        def side(favoured: Iterable[int]) -> list[Separated]:
            return [
                Separated(condition_id=self.conditions[i], label=self._labels[self.conditions[i]])
                for i in list(favoured)[:SEPARATES_SHOWN]
            ]

        entry = BY_CONCEPT[concept_id]
        kind = (
            "risk_factor"
            if concept_id.startswith("RF:")
            else "examination" if concept_id in EXAMINATION else "history"
        )
        return NextQuestion(
            concept_id=concept_id,
            label=entry.label,
            kind=kind,
            gain_bits=gain,
            if_present=side(i for i in named if theta[i] > q),
            if_denied=side(i for i in named if theta[i] < q),
        )


def _significant(value: float) -> float:
    """``value`` to :data:`SIGNIFICANT` significant digits (and -0.0 as 0.0)."""
    return float(f"{value:.{SIGNIFICANT}g}") + 0.0


def _bernoulli_kl_bits(theta: np.ndarray, q: np.ndarray) -> np.ndarray:
    """KL(Bernoulli(θ) ‖ Bernoulli(q)) in bits, elementwise, clipped at 0 against rounding.

    Each term is a divergence, so it is non-negative; their p-weighted sum is the mutual
    information, the gain. θ and q lie strictly inside (0, 1): the scorer's likelihoods are
    bounded away from 0 and 1 (LEAK, CAP).
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        kl = theta * np.log2(theta / q) + (1.0 - theta) * np.log2((1.0 - theta) / (1.0 - q))
    return np.clip(np.nan_to_num(kl, nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)


_RANKERS: weakref.WeakKeyDictionary[Any, QuestionRanker] = weakref.WeakKeyDictionary()


def _ranker_for(scorer: Any) -> QuestionRanker:
    """One :class:`QuestionRanker` per scorer, built on first use: the per-graph precomputation."""
    ranker = _RANKERS.get(scorer)
    if ranker is None:
        ranker = _RANKERS[scorer] = QuestionRanker(scorer)
    return ranker


def next_questions(case: PatientCase, graph: Any, limit: int = QUESTIONS_SHOWN) -> QuestionRanking:
    """The questions worth asking next for ``case`` on ``graph``, best first.

    Args:
        case: The case as entered, or as ``expand_case`` returned it (the derived questions are
            set aside and derived again for each hypothetical answer).
        graph: A graph store with a ``scorer`` (the NetworkX and Neo4j stores, and the stub).
        limit: How many questions at most.

    Returns:
        A :class:`QuestionRanking`. Empty, with ``entropy_bits`` None, when the graph has no
        likelihoods or fails: no guess is made for a degraded graph. Never raises.
    """
    scorer = getattr(graph, "scorer", None)
    if scorer is None or not hasattr(scorer, "score_many"):
        return QuestionRanking()
    try:
        ranking = _ranker_for(scorer).rank(case, limit)
    except Exception:  # noqa: BLE001 - a missing suggestion never fails the result
        logger.exception("case %s: the question ranking failed; no questions", case.case_id)
        return QuestionRanking()
    if ranking.entropy_bits is None or not math.isfinite(ranking.entropy_bits):
        return QuestionRanking()
    return ranking
