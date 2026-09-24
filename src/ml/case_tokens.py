"""Hand-authored cases as DDXPlus evidence tokens (task 1c, the ranker seam).

The pipeline hands :class:`~src.contracts.ConditionRanker` a case written in the hand-authored
vocabulary — ``SYM:chest_pain``, ``RF:diabetes`` — while :class:`~src.ml.features.EvidenceEncoder`
builds its row from raw DDXPlus tokens — ``E_55_@_V_101``. This module is the missing direction of
the crosswalk: **concept → the answer a patient with that concept would have given.**

:mod:`src.medical_kg.crosswalk` translates a concept into the *question* DDXPlus asks
(:func:`~src.medical_kg.crosswalk.expand_case`), because that is the level the knowledge graph
works at. The model works at the level of answers, so something has to choose one: a patient whose
pain radiates "to the jaw or arm" answered ``E_57`` with *some* location, and the encoder has a
column for each. Choosing which is a clinical judgement rather than a lookup, so the two tables
below are hand-curated:

* :data:`REPRESENTATIVE` — the answer that stands for a concept covering several. Taking the lowest
  value id would be wrong: it yields *avant-bras* (forearm) for ``SYM:radiation_jaw_arm``, where the
  discriminating answer is *mâchoire* (jaw), and *cheville* (ankle) for
  ``SYM:leg_swelling_unilateral``, where DDXPlus's own patients say *mollet* (calf).
* :data:`ORDINAL_REPRESENTATIVE` — a value comfortably inside the qualifying range, never the
  threshold itself, so that hand-written cases do not all land exactly on a decision boundary.

**This inversion has no ground truth** (risk R-17). No dataset of (hand-authored case → tokens)
pairs exists, and a wrong answer here looks exactly like a model error downstream. Three things
guard it: every token records the concept it came from, every finding that yields nothing is kept
in :attr:`CaseTokens.dropped` with a reason, and the tests round-trip each concept back through
:func:`~src.medical_kg.crosswalk.concepts_from_evidences`.

**Narrower matches** are admitted by default, as open decision **A-8**.
:func:`~src.medical_kg.crosswalk.expand_case` excludes them — a patient with the concept need not
give that answer — but a strict inversion silently drops ``SYM:sudden_onset``, ``SYM:exertional``
and ``SYM:relieved_by_rest``, the discriminators for pulmonary embolism, pneumothorax, dissection
and the anginas. Every token derived that way is listed in :attr:`CaseTokens.narrower`, so a result
can always say how much of its input was inferred rather than observed.

**Denials reach the model only through the encoder's "asked" channel** (R-18). Without it the
encoder gives a default answer no column, so ``0`` means "denied" and "never asked" alike — the
collapse docs/04-safety-ethics.md §3 forbids — and the B1 models, trained that way, see no denial
at all. :attr:`CaseTokens.asked` lists every question the case settles: those its tokens answer,
and those a denial answers "no". A denial settles a question under the same rule as
:func:`~src.medical_kg.crosswalk.expand_case`: only a yes/no question no broader than the concept
("no diaphoresis" settles "increased sweating?"; "no radiation to the back" does not settle "does
the pain radiate?", since it may radiate elsewhere). :attr:`CaseTokens.denials_asked` names the
denials that did, and every other denial is still in :attr:`CaseTokens.denied`, so a result can
say what the model was unable to see. An answered question is asked with all its answers: a case
recording radiation to the jaw tells the model the pain radiates nowhere else, the same
question-level reading a mask gives a DDXPlus patient (src/ml/evidence_masks.py).

Reference: docs/02-architecture.md §5.2 (the crosswalk card), docs/03-data-management.md §2.2 (the
feature card).
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np

from src.contracts import Assertion, Finding, PatientCase, Sex
from src.ddxplus import CONCEPT_PREFIX, TOKEN_SEP, code_sort_key
from src.medical_kg.crosswalk import (
    BY_CONCEPT,
    CONCEPT_IMPLIES_ANSWER,
    DENIAL_CARRIES_OVER,
    PARENT_QUESTION,
    CrosswalkEntry,
    Match,
)

logger = logging.getLogger(__name__)

__all__ = [
    "ORDINAL_REPRESENTATIVE",
    "REPRESENTATIVE",
    "CaseTokens",
    "DropReason",
    "Dropped",
    "UnrepresentableCase",
    "encode_case",
    "tokens_for_case",
]


class UnrepresentableCase(ValueError):
    """The case cannot become a model row at all.

    Raised only when the feature space has no room for the case, never when it merely loses
    findings on the way: losing findings is ordinary and is reported in
    :attr:`CaseTokens.dropped`. The pipeline catches this and records ``"ml"`` in
    ``degraded_components``.
    """


class DropReason(str, Enum):
    """Why a finding produced no token. Kept with the concept, as the audit trail for R-17."""

    NOT_IN_CROSSWALK = "not in the crosswalk"
    NO_DDXPLUS_EQUIVALENT = "DDXPlus asks nothing comparable"
    RELATED_ONLY = "related only, so no sound inference"
    NARROWER_EXCLUDED = "a narrower match, and include_narrower is False"
    NOT_ENCODED = "the encoder does not carry this evidence"
    NO_REPRESENTATIVE = "no representative answer is chosen for this concept"
    UNKNOWN_ASSERTION = "the finding is neither present nor denied"
    ALREADY_A_QUESTION = "a DDX: question id names no answer"


@dataclass(frozen=True)
class Dropped:
    """One finding that yielded nothing, and why."""

    concept_id: str
    reason: DropReason
    detail: str = ""


@dataclass(frozen=True)
class CaseTokens:
    """What a case became, and what it lost on the way.

    Attributes:
        tokens: DDXPlus EVIDENCES tokens in evidence-code order, ready for
            :meth:`~src.ml.features.EvidenceEncoder.encode`.
        by_concept: concept id → the tokens it produced, parent questions included.
        narrower: Those tokens derived through a NARROWER match (decision A-8); a subset of
            ``tokens``.
        denied: Concepts the case explicitly denies, all of them.
        dropped: Every finding that produced no token, with its reason.
        asked: The questions the case settles, in evidence-code order: those ``tokens`` answer
            and those ``denials_asked`` answer "no". The encoder's ``asked`` argument; only an
            encoder with the "asked" channel reads it.
        denials_asked: Denied concept → the yes/no question its denial answers "no". A denial
            not listed here reaches no model.
    """

    tokens: tuple[str, ...]
    by_concept: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    narrower: tuple[str, ...] = ()
    denied: tuple[str, ...] = ()
    dropped: tuple[Dropped, ...] = ()
    asked: tuple[str, ...] = ()
    denials_asked: Mapping[str, str] = field(default_factory=dict)

    def summary(self) -> str:
        """One line for a log or a report."""
        return (
            f"{len(self.tokens)} tokens from {len(self.by_concept)} concepts"
            f" ({len(self.narrower)} narrower), {len(self.denied)} denied"
            f" ({len(self.denials_asked)} as a question), {len(self.dropped)} dropped,"
            f" {len(self.asked)} questions asked"
        )


# --------------------------------------------------------------------------- #
# The two judgement tables
# --------------------------------------------------------------------------- #

# fmt: off
REPRESENTATIVE: Mapping[str, tuple[str, ...]] = {
    # --- Pain location, E_55 ------------------------------------------------- #
    "SYM:chest_pain": ("V_101",),                      # haut du thorax, upper chest
    "SYM:back_pain": ("V_39",),                        # colonne dorsale, thoracic spine
    "SYM:abdominal_pain": ("V_187",),                  # ventre, belly
    "SYM:epigastric_pain": ("V_197",),                 # épigastre
    "SYM:calf_pain": ("V_119",),                       # mollet (D), right calf
    # --- Pain character, E_54 ------------------------------------------------ #
    "SYM:pain_character_pressure": ("V_183",),         # une lourdeur, heaviness
    "SYM:pain_character_tearing": ("V_71",),           # déchirante ("heartbreaking" is wrong)
    "SYM:pain_character_burning": ("V_181",),          # une brûlure
    "SYM:pain_character_sharp": ("V_192",),            # vive, sharp
    # --- Radiation, E_57 ----------------------------------------------------- #
    "SYM:radiation_back": ("V_39",),                   # colonne dorsale, thoracic spine
    "SYM:radiation_jaw_arm": ("V_121",),               # mâchoire, jaw: the ischaemic pattern
    "SYM:radiation_neck": ("V_53",),                   # côté du cou (D), right side of the neck
    # --- Swelling, E_152 ----------------------------------------------------- #
    "SYM:leg_swelling": ("V_119",),                    # mollet (D)
    "SYM:leg_swelling_unilateral": ("V_119",),         # mollet (D): one side only
    "SYM:leg_swelling_bilateral": ("V_119", "V_120"),  # mollet (D) and (G): both calves
}
"""Concept → the DDXPlus answer, or answers, that stand for it.

One entry for every crosswalk concept whose pattern lists answers; a test fails if the two sets
differ. Each value is a clinical choice, and its comment gives the French, because the English
labels are machine-translated. ``SYM:leg_swelling_bilateral`` needs two answers because its pattern
asks for both sides (:class:`~src.medical_kg.crosswalk.Side`).
"""

ORDINAL_REPRESENTATIVE: Mapping[str, int] = {
    "SYM:severe_pain": 8,    # E_56 intensity: 7-10 is severe, and 8 sits inside the band
    "SYM:sudden_onset": 9,   # E_59 speed of onset: >= 8 is sudden (A-5), and 9 sits inside it
}
"""Concept → its answer on a 0-10 scale.

Inside the qualifying range rather than on the threshold, so that a hand-written case is not
decided by a cut-off the team may still move (A-5).
"""
# fmt: on


# --------------------------------------------------------------------------- #
# Concept -> tokens
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class _Request:
    """One concept's claim on one evidence question, before clashes are resolved."""

    concept_id: str
    code: str
    answers: tuple[str, ...]
    is_ordinal: bool
    narrower: bool


def _with_parents(code: str) -> list[str]:
    """``E_55`` → ``["E_55", "E_53"]``. Answering a follow-up implies a yes to its parent."""
    chain = [code]
    while chain[-1] in PARENT_QUESTION:
        chain.append(PARENT_QUESTION[chain[-1]])
    return chain


def _token_sort_key(token: str) -> tuple[int, int]:
    """Evidence order, then answer order: E_55_@_V_39 before E_55_@_V_101, as DDXPlus lists them."""
    code, _, value = token.partition(TOKEN_SEP)
    rank = -1 if not value else int(value.removeprefix("V_"))
    return code_sort_key(code), rank


def _answers_for(entry: CrosswalkEntry) -> tuple[str, ...] | None:
    """The answers expressing this concept, or None when no table entry chooses one."""
    pattern = entry.pattern
    if pattern is None:  # pragma: no cover - callers check
        return None
    if pattern.min_ordinal is not None:
        value = ORDINAL_REPRESENTATIVE.get(entry.concept_id)
        return None if value is None else (str(value),)
    if pattern.values:
        return REPRESENTATIVE.get(entry.concept_id)
    return ()  # a yes/no question: the code alone is the token


def _findings(case: PatientCase) -> Sequence[Finding]:
    return [*case.findings, *case.risk_factors]


def _denied_question(concept: str, allowed: set[str] | None) -> str | None:
    """The yes/no question a denial of ``concept`` answers "no", if it answers one."""
    entry = BY_CONCEPT.get(concept)
    if entry is None or entry.pattern is None or entry.match not in DENIAL_CARRIES_OVER:
        return None
    if not entry.pattern.whole_question:
        return None
    code = entry.pattern.code
    return code if allowed is None or code in allowed else None


def tokens_for_case(
    case: PatientCase, *, codes: Iterable[str] | None = None, include_narrower: bool = True
) -> CaseTokens:
    """The DDXPlus tokens a hand-authored case implies.

    Args:
        case: A case written with ``SYM:*`` / ``RF:*`` concepts, such as a golden clinical case.
        codes: The evidence codes the encoder carries (``EvidenceEncoder.codes``). A token for any
            other evidence is dropped rather than raised, because the encoder would otherwise
            refuse the whole row. None keeps every code.
        include_narrower: Admit concepts whose DDXPlus answer is *narrower* than the concept
            (decision A-8). See the module docstring.

    Returns:
        A :class:`CaseTokens` whose ``tokens`` are ready for the encoder, and whose other fields
        say what was inferred and what was lost.
    """
    allowed = None if codes is None else set(codes)
    requests: list[_Request] = []
    denied: list[str] = []
    dropped: list[Dropped] = []

    for finding in _findings(case):
        concept = finding.concept_id
        if finding.assertion is Assertion.ABSENT:
            denied.append(concept)
            continue
        if finding.assertion is not Assertion.PRESENT:
            dropped.append(Dropped(concept, DropReason.UNKNOWN_ASSERTION))
            continue
        if concept.startswith(CONCEPT_PREFIX):
            # expand_case() adds these. They name a question, never the answer given to it.
            dropped.append(Dropped(concept, DropReason.ALREADY_A_QUESTION))
            continue
        entry = BY_CONCEPT.get(concept)
        if entry is None:
            dropped.append(Dropped(concept, DropReason.NOT_IN_CROSSWALK))
            continue
        if entry.pattern is None:
            dropped.append(Dropped(concept, DropReason.NO_DDXPLUS_EQUIVALENT, entry.note))
            continue
        narrower = entry.match is Match.NARROWER
        if entry.match not in CONCEPT_IMPLIES_ANSWER and not narrower:
            dropped.append(Dropped(concept, DropReason.RELATED_ONLY, entry.note))
            continue
        if narrower and not include_narrower:
            dropped.append(Dropped(concept, DropReason.NARROWER_EXCLUDED, entry.note))
            continue
        code = entry.pattern.code
        if allowed is not None and code not in allowed:
            dropped.append(Dropped(concept, DropReason.NOT_ENCODED, code))
            continue
        answers = _answers_for(entry)
        if answers is None:
            logger.warning("no representative answer for %s; add one to case_tokens", concept)
            dropped.append(Dropped(concept, DropReason.NO_REPRESENTATIVE, code))
            continue
        requests.append(
            _Request(concept, code, answers, entry.pattern.min_ordinal is not None, narrower)
        )

    # A 0-10 scale takes exactly one answer, so two concepts claiming the same question agree on
    # the higher one: "severe" and "moderate" pain together is severe pain.
    strongest: dict[str, int] = {}
    for request in requests:
        if request.is_ordinal:
            value = int(request.answers[0])
            strongest[request.code] = max(strongest.get(request.code, value), value)

    chosen: dict[str, None] = {}
    by_concept: dict[str, list[str]] = {}
    from_narrower: set[str] = set()
    from_sound: set[str] = set()

    def keep(token: str, request: _Request) -> None:
        chosen.setdefault(token, None)
        produced = by_concept.setdefault(request.concept_id, [])
        if token not in produced:
            produced.append(token)
        (from_narrower if request.narrower else from_sound).add(token)

    for request in requests:
        if request.is_ordinal:
            keep(f"{request.code}{TOKEN_SEP}{strongest[request.code]}", request)
        elif request.answers:
            for answer in request.answers:
                keep(f"{request.code}{TOKEN_SEP}{answer}", request)
        else:
            keep(request.code, request)  # a yes/no question is its own token
        for parent in _with_parents(request.code)[1:]:
            if allowed is None or parent in allowed:
                keep(parent, request)

    tokens = tuple(sorted(chosen, key=_token_sort_key))
    answered = {token.partition(TOKEN_SEP)[0] for token in tokens}
    denials_asked: dict[str, str] = {}
    for concept in denied:
        code = _denied_question(concept, allowed)
        if code is None:
            continue
        if code in answered:
            logger.warning(
                "case %s: %s is denied, but %s is answered yes; the answer stands",
                case.case_id,
                concept,
                code,
            )
            continue
        denials_asked[concept] = code
    asked = tuple(sorted(answered | set(denials_asked.values()), key=code_sort_key))
    return CaseTokens(
        tokens=tokens,
        by_concept={c: tuple(t) for c, t in by_concept.items() if t},
        # Only tokens that *no* sound match also produced: a parent question reached both ways is
        # not an inference, and saying it was would overstate how much the row was guessed.
        narrower=tuple(sorted(from_narrower - from_sound, key=_token_sort_key)),
        denied=tuple(denied),
        dropped=tuple(dropped),
        asked=asked,
        denials_asked=denials_asked,
    )


def encode_case(
    case: PatientCase, encoder: Any, *, include_narrower: bool = True
) -> tuple[np.ndarray, CaseTokens]:
    """One case as a model row, with the audit trail that produced it.

    Args:
        case: A case written with ``SYM:*`` / ``RF:*`` concepts.
        encoder: An :class:`~src.ml.features.EvidenceEncoder`.
        include_narrower: As :func:`tokens_for_case`.

    Returns:
        ``(row, tokens)``: a 1-D float32 row in the encoder's column order, and the
        :class:`CaseTokens` it was built from. An encoder with the "asked" channel is told which
        questions the case settles; every other question is unasked.

    Raises:
        UnrepresentableCase: for a sex the feature space has no column for.
    """
    if case.sex is Sex.OTHER:
        raise UnrepresentableCase(
            f"case {case.case_id}: DDXPlus records sex as M or F only, so the model has no row "
            "for Sex.OTHER. The ranker degrades, and the ranking stays knowledge-graph only."
        )
    tokens = tokens_for_case(case, codes=encoder.codes, include_narrower=include_narrower)
    row = encoder.encode(case.age, case.sex.value, tokens.tokens, asked=tokens.asked)
    return row, tokens
