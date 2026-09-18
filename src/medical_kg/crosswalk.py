"""The crosswalk between hand-authored concepts and DDXPlus evidence (task 1b).

Nightingale speaks two vocabularies. The red-flag rules, the golden cases and the stub graph
use hand-authored concepts such as ``SYM:chest_pain`` and ``RF:diabetes``. The knowledge graph
and the DDXPlus patients use evidence *questions* and their *answers*: ``E_55``, "where is your
pain?", answered ``V_101``, "upper chest". :data:`CROSSWALK` is the table that links them. Each
entry names the DDXPlus answers that express one concept, and how well they match
(:class:`Match`).

Two translations use the table:

* :func:`expand_case`, concept → knowledge graph. It adds to a hand-authored case the
  ``DDX:E_nn`` findings that its concepts imply, so the DDXPlus-derived graph can score it. The
  original findings stay, so the red-flag rules still see them.
* :func:`concepts_from_evidences`, DDXPlus → concept. It lists the concepts a DDXPlus patient
  has, so the red-flag rules can run on DDXPlus rows. :func:`case_from_ddxplus` builds the
  whole case.

**Only sound inferences are drawn.** A concept implies a DDXPlus answer only when the answer is
at least as broad as the concept. An answer implies a concept only when it is at least as
narrow. A denial carries over only to a yes/no question: denying "radiation to the jaw or arm"
says nothing about radiation elsewhere, so it cannot deny ``E_57``, "does the pain radiate?".

**The graph knows questions, not answers** (docs/02 §5.1, limitation 1). Expanding
"pressure-type pain" therefore yields ``DDX:E_54``, "characterize your pain", which matches
every condition whose pain DDXPlus characterises, burning and tearing alike. Derived findings
are labelled with the question they stand for, never with the finding they came from, so a
reasoning path cannot claim that GERD presents with pressure-type pain.

Work from the French when checking an entry: DDXPlus English is machine-translated
(``déchirante``, *tearing*, became "heartbreaking").

Reference: docs/02-architecture.md §5.2, the crosswalk card.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from src.contracts import Assertion, Finding, PatientCase
from src.ddxplus import (
    CONCEPT_PREFIX,
    NA_VALUE,
    EvidenceSpec,
    TokenKind,
    code_sort_key,
    concept_id,
    parse_token,
)

logger = logging.getLogger(__name__)

__all__ = [
    "ARM",
    "BACK",
    "BY_CONCEPT",
    "CHEST",
    "CROSSWALK",
    "DDXPLUS_SOURCE",
    "DERIVED_SOURCE",
    "JAW",
    "LEFT_LEG",
    "PARENT_QUESTION",
    "RIGHT_LEG",
    "SIDE_OF",
    "SUDDEN_ONSET_MIN",
    "CrosswalkEntry",
    "Match",
    "Pattern",
    "Side",
    "case_from_ddxplus",
    "concepts_from_evidences",
    "expand_case",
    "validate_crosswalk",
]

DERIVED_SOURCE = "crosswalk"
"""``Finding.source`` of a finding this module derived rather than observed."""

DDXPLUS_SOURCE = "ddxplus"
"""``Finding.source`` of a finding read straight from a DDXPlus patient's EVIDENCES."""


class Match(str, Enum):
    """How the DDXPlus answers relate to the concept.

    These are the SKOS mapping relations, read as "the DDXPlus answer is ___ the concept".
    """

    EXACT = "exact"
    """The same meaning."""
    CLOSE = "close"
    """Near enough to use as exact. Reported as approximate on the crosswalk card."""
    BROADER = "broader"
    """Covers more: every patient with the concept gives this answer, but not the reverse."""
    NARROWER = "narrower"
    """Covers less: every patient giving this answer has the concept, but not the reverse."""
    RELATED = "related"
    """Associated only. No inference in either direction."""
    NONE = "none"
    """DDXPlus asks nothing comparable."""


CONCEPT_IMPLIES_ANSWER = frozenset({Match.EXACT, Match.CLOSE, Match.BROADER})
ANSWER_IMPLIES_CONCEPT = frozenset({Match.EXACT, Match.CLOSE, Match.NARROWER})
DENIAL_CARRIES_OVER = frozenset({Match.EXACT, Match.CLOSE, Match.NARROWER})
"""Denying the concept denies the answer, if the answer is no broader than the concept."""


class Side(str, Enum):
    """For paired body locations: which sides the chosen answers cover."""

    ONE = "one"
    BOTH = "both"


# Answers of DDXPlus's body-location list (E_55 pain, E_57 radiation, E_152 swelling). The
# French labels are given because the English ones are machine-translated.
CHEST = frozenset({"V_29", "V_101", "V_55", "V_56", "V_159", "V_160", "V_170", "V_171"})
"""bas / haut du thorax, côté du thorax (D/G), sein (D/G), thorax postérieur (D/G).
Not épigastre (V_197): epigastric pain is abdominal, however often it accompanies chest pain."""

BACK = frozenset({"V_39", "V_40", "V_127", "V_128", "V_170", "V_171"})
"""colonne dorsale, colonne lombaire, omoplate (D/G), thorax postérieur (D/G)."""

JAW = frozenset({"V_121", "V_163", "V_118"})
"""mâchoire, sous la mâchoire, menton."""

ARM = frozenset(
    {"V_194", "V_195", "V_30", "V_31", "V_177", "V_178", "V_45", "V_46", "V_27", "V_28"}
    | {"V_78", "V_79", "V_74", "V_75", "V_80", "V_81", "V_76", "V_77", "V_142", "V_143"}
    | {"V_63", "V_64", "V_65", "V_66", "V_67", "V_68", "V_69", "V_70", "V_151", "V_152"}
)
"""Shoulder to fingertips, both sides: épaule, biceps, triceps, coude, avant-bras (and its
palmar face), poignet, main, paume, doigts, pouce."""

RIGHT_LEG = frozenset(
    {"V_51", "V_92", "V_47", "V_105", "V_119", "V_172", "V_34", "V_23", "V_72", "V_43"}
    | {"V_164", "V_149"}
)
"""Right lower limb, thigh to sole, toes excluded: cuisse, genou, creux poplité, ischio,
mollet, tibia, cheville, arrière de la cheville, face dorsale du pied, côté latéral du pied,
talon, plante du pied — each (D)."""

LEFT_LEG = frozenset(
    {"V_52", "V_93", "V_48", "V_106", "V_120", "V_173", "V_35", "V_24", "V_73", "V_44"}
    | {"V_165", "V_150"}
)
"""The same locations, each (G)."""

SIDE_OF: Mapping[str, str] = {**{v: "R" for v in RIGHT_LEG}, **{v: "L" for v in LEFT_LEG}}

SUDDEN_ONSET_MIN = 8
"""Lowest answer to ``E_59``, "how fast did the pain appear?" (0-10), that counts as sudden.

A judgment call, recorded as open decision A-5 in docs/02 §9. DDXPlus draws this answer
uniformly within a range for each condition (5-10 for MI, 0-10 for PE), so it separates
conditions only by the ends of their ranges."""

PARENT_QUESTION: Mapping[str, str] = {
    "E_54": "E_53",
    "E_55": "E_53",
    "E_56": "E_53",
    "E_57": "E_53",
    "E_58": "E_53",
    "E_59": "E_53",
    "E_152": "E_151",
}
"""Follow-up question → the question it follows (``code_question`` in release_evidences.json).

An answer to a follow-up implies a yes to its parent: pain that radiates is pain. Only
parents that exist are listed. ``E_69`` (diabetes) names ``E_68`` as its parent, but DDXPlus
has no ``E_68``."""


@dataclass(frozen=True)
class Pattern:
    """The DDXPlus answers that express a concept.

    Attributes:
        code: The evidence question, e.g. ``E_55``.
        values: For categorical and multi-choice questions, the answers that count; any one is
            enough. Never the question's "no" default or ``V_11`` (NA).
        min_ordinal: For 0-10 scales, the lowest answer that counts.
        side: For paired body locations, whether the chosen answers must cover one side only or
            both. Every value then needs a side in :data:`SIDE_OF`.
    """

    code: str
    values: frozenset[str] = frozenset()
    min_ordinal: int | None = None
    side: Side | None = None

    @property
    def whole_question(self) -> bool:
        """True for a yes/no question, where the pattern is the question itself."""
        return not self.values and self.min_ordinal is None

    def matches(self, answers: _Answers) -> bool:
        if self.min_ordinal is not None:
            ordinal = answers.ordinals.get(self.code)
            return ordinal is not None and ordinal >= self.min_ordinal
        if not self.values:
            return self.code in answers.yes
        chosen = answers.values.get(self.code, frozenset()) & self.values
        if self.side is None:
            return bool(chosen)
        sides = {SIDE_OF[v] for v in chosen}
        return len(sides) == 2 if self.side is Side.BOTH else len(sides) == 1


@dataclass(frozen=True)
class CrosswalkEntry:
    """One hand-authored concept and the DDXPlus answers that express it.

    ``pattern`` is None exactly when ``match`` is NONE.
    """

    concept_id: str
    label: str
    match: Match
    pattern: Pattern | None = None
    note: str = field(default="", compare=False)


def _entry(
    concept: str, label: str, match: Match, code: str | None = None, note: str = "", **kw: Any
) -> CrosswalkEntry:
    pattern = Pattern(code, **kw) if code else None
    return CrosswalkEntry(concept, label, match, pattern, note)


# fmt: off
CROSSWALK: tuple[CrosswalkEntry, ...] = (
    # --- Chest pain and its character ------------------------------------------- #
    _entry("SYM:chest_pain", "Chest pain", Match.CLOSE, "E_55", values=CHEST,
           note="E_55 'Avez-vous de la douleur quelque part?' answered with a chest location"),
    _entry("SYM:pain_character_pressure", "Pressure-type pain", Match.CLOSE, "E_54",
           values=frozenset({"V_183"}),
           note="'une lourdeur' (heaviness). DDXPlus has no pressure or tightness answer; "
           "heaviness is the usual French word for ischaemic pain"),
    _entry("SYM:pain_character_tearing", "Tearing pain", Match.EXACT, "E_54",
           values=frozenset({"V_71"}),
           note="'déchirante'. Its English label, 'heartbreaking', is a mistranslation"),
    _entry("SYM:pain_character_burning", "Burning pain", Match.EXACT, "E_54",
           values=frozenset({"V_181"}), note="'une brûlure'"),
    _entry("SYM:pleuritic", "Pleuritic pain", Match.EXACT, "E_220",
           note="'douleur qui est pire à l'inspiration profonde'"),
    _entry("SYM:sudden_onset", "Sudden onset", Match.NARROWER, "E_59",
           min_ordinal=SUDDEN_ONSET_MIN,
           note="E_59 'À quelle vitesse la douleur est-elle apparue?' >= 8 of 10. Narrower: it "
           "covers pain only, not e.g. palpitations. The cut-off is open decision A-5"),
    _entry("SYM:exertional", "Exertional onset", Match.NARROWER, "E_218",
           note="E_218 also requires relief by rest"),
    _entry("SYM:relieved_by_rest", "Relieved by rest", Match.NARROWER, "E_218",
           note="E_218 also requires worsening with exertion"),
    _entry("SYM:post_prandial", "Post-prandial", Match.CLOSE, "E_215",
           note="'symptômes qui sont pires après les repas'"),
    _entry("SYM:worse_lying_flat", "Worse lying flat", Match.CLOSE, "E_217",
           note="E_217 'pire en position couchée et améliorés en position assise'"),
    _entry("SYM:positional_relief_sitting_forward", "Relieved by sitting forward",
           Match.BROADER, "E_217",
           note="E_217 covers any symptom better sitting up and worse lying down; DDXPlus "
           "uses it for acute pulmonary edema and GERD too, so it does not imply this concept"),
    # --- Radiation ---------------------------------------------------------------- #
    _entry("SYM:radiation_back", "Radiation to back", Match.CLOSE, "E_57", values=BACK,
           note="thoracic or lumbar spine, scapulae, posterior chest wall"),
    _entry("SYM:radiation_jaw_arm", "Radiation to jaw/arm", Match.CLOSE, "E_57",
           values=JAW | ARM,
           note="jaw, under the jaw, chin; shoulder to fingertips, either side. The neck and "
           "throat (V_33 thyroid cartilage, V_174 trachea) are a different concept"),
    # --- Breathing, heart, general -------------------------------------------------- #
    _entry("SYM:breathlessness", "Breathlessness", Match.CLOSE, "E_66",
           note="'essoufflé ou ... de la difficulté à respirer de façon importante'"),
    _entry("SYM:orthopnoea", "Orthopnoea", Match.BROADER, "E_217",
           note="E_217 covers any symptom worse lying down; orthopnoea is breathlessness"),
    _entry("SYM:frothy_sputum", "Frothy sputum", Match.NONE,
           note="DDXPlus asks about coloured or abundant sputum (E_77) and blood (E_45), not "
           "frothy sputum"),
    _entry("SYM:hyperventilation", "Hyperventilation", Match.NONE,
           note="No DDXPlus question. The nearest, E_75 (a feeling of choking or suffocating), "
           "is a different panic symptom"),
    _entry("SYM:diaphoresis", "Diaphoresis", Match.EXACT, "E_50",
           note="'des sueurs importantes'"),
    _entry("SYM:palpitations", "Palpitations", Match.CLOSE, "E_155",
           note="E_155 bundles a racing, irregular or pounding heartbeat"),
    _entry("SYM:irregular_pulse", "Irregular pulse", Match.RELATED, "E_164",
           note="E_164 is an irregular heartbeat the patient feels; an irregular pulse is found "
           "on examination"),
    _entry("SYM:lightheadedness", "Lightheadedness", Match.EXACT, "E_76",
           note="'légèrement étourdi ... déséquilibre léger'"),
    _entry("SYM:fatigue", "Fatigue", Match.BROADER, "E_175",
           note="E_175 is new fatigue, malaise or diffuse muscle aches"),
    _entry("SYM:anxiety", "Anxiety", Match.EXACT, "E_16",
           note="'une certaine anxiété, une fébrilité'"),
    _entry("SYM:paraesthesia", "Paraesthesia", Match.CLOSE, "E_177",
           note="numbness, loss of sensation or tingling anywhere, now or recently. The English "
           "'did you ever' is a mistranslation of 'récemment'"),
    _entry("SYM:interarm_bp_difference", "Inter-arm BP difference", Match.NONE,
           note="A blood-pressure measurement. DDXPlus records symptoms and history only, so "
           "the aortic-dissection rule can use tearing pain and back radiation, never this"),
    # --- Thrombosis, vomiting, infection -------------------------------------------- #
    _entry("SYM:leg_swelling_unilateral", "Unilateral leg swelling", Match.EXACT, "E_152",
           values=RIGHT_LEG | LEFT_LEG, side=Side.ONE,
           note="E_152 swelling locations on one leg only"),
    _entry("SYM:leg_swelling_bilateral", "Bilateral leg swelling", Match.EXACT, "E_152",
           values=RIGHT_LEG | LEFT_LEG, side=Side.BOTH,
           note="E_152 swelling locations on both legs"),
    _entry("SYM:recent_immobilisation", "Recent immobilisation", Match.CLOSE, "E_110",
           note="unable to get up for more than 3 consecutive days in the last 4 weeks: the "
           "Wells criterion"),
    _entry("SYM:recent_forceful_vomiting", "Recent forceful vomiting", Match.CLOSE, "E_211",
           note="'plusieurs vomissements ou ... plusieurs efforts pour vomir'"),
    _entry("SYM:recent_viral_illness", "Recent viral illness", Match.EXACT, "E_0",
           note="'infecté par un virus récemment'"),
    # --- Risk factors ---------------------------------------------------------------- #
    _entry("RF:diabetes", "Diabetes mellitus", Match.EXACT, "E_69"),
    _entry("RF:smoking", "Smoking", Match.CLOSE, "E_79",
           note="'fumez-vous la cigarette quotidiennement'"),
    _entry("RF:hypertension", "Hypertension", Match.EXACT, "E_104",
           note="high blood pressure, or medication for it"),
)
# fmt: on
"""Every hand-authored concept the codebase uses: the red-flag rules, the golden cases and
``STUB_SYMPTOM_MAP``. tests/test_crosswalk.py fails if one is missing."""

BY_CONCEPT: Mapping[str, CrosswalkEntry] = {e.concept_id: e for e in CROSSWALK}


# --------------------------------------------------------------------------- #
# DDXPlus answers
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class _Answers:
    """One patient's DDXPlus answers by question. NA answers are dropped."""

    yes: frozenset[str]
    values: Mapping[str, frozenset[str]]
    ordinals: Mapping[str, int]

    @classmethod
    def parse(cls, evidences: Iterable[str]) -> _Answers:
        yes: set[str] = set()
        values: dict[str, set[str]] = defaultdict(set)
        ordinals: dict[str, int] = {}
        for raw in evidences:
            token = parse_token(raw)
            if token.kind is TokenKind.BINARY:
                yes.add(token.code)
            elif token.kind is TokenKind.VALUE and token.value is not None:
                values[token.code].add(token.value)
            elif token.kind is TokenKind.ORDINAL and token.ordinal is not None:
                ordinals[token.code] = token.ordinal
        return cls(frozenset(yes), {k: frozenset(v) for k, v in values.items()}, ordinals)


def concepts_from_evidences(evidences: Iterable[str]) -> list[str]:
    """The hand-authored concepts a DDXPlus patient has, in :data:`CROSSWALK` order.

    Only entries whose answers imply the concept count: exact, close and narrower ones.

    Args:
        evidences: The patient's raw EVIDENCES tokens, e.g. ``["E_53", "E_55_@_V_101"]``.

    Raises:
        ValueError: on a malformed token.
    """
    answers = _Answers.parse(evidences)
    return [
        e.concept_id
        for e in CROSSWALK
        if e.pattern is not None
        and e.match in ANSWER_IMPLIES_CONCEPT
        and e.pattern.matches(answers)
    ]


def case_from_ddxplus(
    record: Mapping[str, Any],
    specs: Mapping[str, EvidenceSpec],
    labels: Mapping[str, str] | None = None,
) -> PatientCase:
    """Build a PatientCase from one row of the chest-pain parquet (docs/03 §2.1).

    Reads ``case_id`` and the input columns only, never a label column. The findings are one
    ``DDX:E_nn`` per positive code (source ``ddxplus``), then the concepts its answers imply
    (source ``crosswalk``). Antecedent questions and ``RF:*`` concepts go to ``risk_factors``.

    A question the patient's row does not list stays UNKNOWN rather than ABSENT: whether
    DDXPlus's silence means "no" is a modelling decision for 2a, not for the crosswalk.

    Args:
        record: A parquet row with ``case_id``, ``age``, ``sex``, ``evidences`` and
            ``positive_codes``.
        specs: From :func:`src.ddxplus.load_evidence_specs`, to tell antecedents apart.
        labels: ``DDX:E_nn`` → label, e.g. the knowledge graph's node labels.
    """
    labels = labels or {}
    findings: list[Finding] = []
    risk_factors: list[Finding] = []
    for code in record["positive_codes"]:
        cid = concept_id(code)
        finding = Finding(concept_id=cid, label=labels.get(cid, cid), source=DDXPLUS_SOURCE)
        spec = specs.get(code)
        (risk_factors if spec is not None and spec.is_antecedent else findings).append(finding)
    for concept in concepts_from_evidences(record["evidences"]):
        finding = Finding(
            concept_id=concept, label=BY_CONCEPT[concept].label, source=DERIVED_SOURCE
        )
        (risk_factors if concept.startswith("RF:") else findings).append(finding)
    return PatientCase(
        case_id=str(record["case_id"]),
        age=int(record["age"]),
        sex=str(record["sex"]),
        findings=findings,
        risk_factors=risk_factors,
    )


# --------------------------------------------------------------------------- #
# Hand-authored case → knowledge graph
# --------------------------------------------------------------------------- #


def expand_case(case: PatientCase, labels: Mapping[str, str] | None = None) -> PatientCase:
    """Add the ``DDX:E_nn`` findings that a hand-authored case implies, for the graph.

    Returns a new case holding the original findings unchanged, followed by the derived ones.
    Each derived finding has source ``crosswalk``, a qualifier ``crosswalk_from`` naming the
    concept it came from, and the label of the question it stands for.

    * A PRESENT concept implies its question, and the question that question follows up
      (:data:`PARENT_QUESTION`), when the answer is at least as broad as the concept.
    * An ABSENT concept denies its question only when that is a yes/no question no broader
      than the concept.
    * A question implied both present and absent is left out, and a warning is logged.
    * A question the case already holds is not added again, so expanding twice is harmless.

    Args:
        case: A case written with ``SYM:*`` / ``RF:*`` concepts.
        labels: ``DDX:E_nn`` → label, e.g. ``dict(store.graph.nodes(data="label"))``.
            A question without a label is labelled with its id.
    """
    labels = labels or {}
    existing = {f.concept_id for f in [*case.findings, *case.risk_factors]}
    present: dict[str, tuple[str, bool]] = {}  # DDX id -> (origin concept, is a risk factor)
    absent: dict[str, tuple[str, bool]] = {}
    for is_risk_factor, findings in ((False, case.findings), (True, case.risk_factors)):
        for finding in findings:
            entry = BY_CONCEPT.get(finding.concept_id)
            if entry is None or entry.pattern is None:
                continue
            origin = (finding.concept_id, is_risk_factor)
            if finding.assertion is Assertion.PRESENT and entry.match in CONCEPT_IMPLIES_ANSWER:
                for code in _with_parents(entry.pattern.code):
                    present.setdefault(concept_id(code), origin)
            elif (
                finding.assertion is Assertion.ABSENT
                and entry.match in DENIAL_CARRIES_OVER
                and entry.pattern.whole_question
            ):
                absent.setdefault(concept_id(entry.pattern.code), origin)

    conflicting = present.keys() & absent.keys()
    if conflicting:
        logger.warning(
            "case %s: %s implied both present and absent; left out",
            case.case_id,
            sorted(conflicting, key=_ddx_order),
        )
    added: dict[bool, list[Finding]] = {False: [], True: []}
    for assertion, derived in ((Assertion.PRESENT, present), (Assertion.ABSENT, absent)):
        for cid in sorted(derived, key=_ddx_order):
            if cid in conflicting or cid in existing:
                continue
            origin_concept, is_risk_factor = derived[cid]
            added[is_risk_factor].append(
                Finding(
                    concept_id=cid,
                    label=labels.get(cid, cid),
                    assertion=assertion,
                    qualifiers={"crosswalk_from": origin_concept},
                    source=DERIVED_SOURCE,
                )
            )
    return case.model_copy(
        update={
            "findings": [*case.findings, *added[False]],
            "risk_factors": [*case.risk_factors, *added[True]],
        }
    )


def _with_parents(code: str) -> list[str]:
    chain = [code]
    while chain[-1] in PARENT_QUESTION:
        chain.append(PARENT_QUESTION[chain[-1]])
    return chain


def _ddx_order(cid: str) -> int:
    return code_sort_key(cid[len(CONCEPT_PREFIX) :])


# --------------------------------------------------------------------------- #
# Validation against the DDXPlus release
# --------------------------------------------------------------------------- #


def validate_crosswalk(
    release_evidences: Mapping[str, Mapping[str, Any]],
    entries: Iterable[CrosswalkEntry] = CROSSWALK,
) -> list[str]:
    """Check the table against DDXPlus's ``release_evidences.json``.

    Every code must exist; answers must be possible for their question, and never its "no"
    default or NA; yes/no questions take no answers; 0-10 scales take ``min_ordinal``; paired
    locations must be labelled (D) or (G) as :data:`SIDE_OF` says; and :data:`PARENT_QUESTION`
    must agree with the release's ``code_question``.

    Returns:
        One message per problem. An empty list means the table is valid.
    """
    problems: list[str] = []
    seen: set[str] = set()
    for entry in entries:
        where = entry.concept_id
        if where in seen:
            problems.append(f"{where}: duplicate entry")
        seen.add(where)
        if not where.startswith(("SYM:", "RF:")):
            problems.append(f"{where}: concept ids are SYM:* or RF:*")
        if (entry.pattern is None) != (entry.match is Match.NONE):
            problems.append(f"{where}: a pattern is required unless the match is 'none'")
        if entry.pattern is not None:
            problems.extend(_check_pattern(where, entry.pattern, release_evidences))

    for child, parent in PARENT_QUESTION.items():
        spec = release_evidences.get(child)
        if spec is None or spec.get("code_question") != parent:
            problems.append(f"PARENT_QUESTION: {child} does not follow up {parent} in the release")
    return problems


def _check_pattern(where: str, p: Pattern, release: Mapping[str, Mapping[str, Any]]) -> list[str]:
    spec = release.get(p.code)
    if spec is None:
        return [f"{where}: {p.code} is not a DDXPlus question"]
    problems = []
    data_type = spec.get("data_type")
    possible = {str(v) for v in spec.get("possible-values") or []}
    default = str(spec.get("default_value"))
    is_scale = data_type == "C" and bool(possible) and all(v.isdigit() for v in possible)

    if data_type == "B":
        if not p.whole_question or p.side is not None:
            problems.append(f"{where}: {p.code} is a yes/no question; it takes no answers")
    elif is_scale:
        if p.min_ordinal is None or p.values:
            problems.append(f"{where}: {p.code} is a 0-10 scale; give min_ordinal, not values")
        elif str(p.min_ordinal) not in possible or str(p.min_ordinal) == default:
            problems.append(f"{where}: min_ordinal {p.min_ordinal} is not a positive answer")
    else:
        if not p.values or p.min_ordinal is not None:
            problems.append(f"{where}: {p.code} needs the answers that count")
        unknown = sorted(p.values - possible)
        if unknown:
            problems.append(f"{where}: {unknown} are not answers to {p.code}")
        if default in p.values or NA_VALUE in p.values:
            problems.append(f"{where}: a 'no' or NA answer cannot express a concept")

    if p.side is not None:
        meanings = spec.get("value_meaning") or {}
        for value in sorted(p.values):
            side = SIDE_OF.get(value)
            french = (meanings.get(value) or {}).get("fr", "")
            if side is None or not french.endswith("(D)" if side == "R" else "(G)"):
                problems.append(f"{where}: {value} ({french!r}) is not on side {side}")

    parent = spec.get("code_question")
    if parent and parent != p.code and parent in release and PARENT_QUESTION.get(p.code) != parent:
        problems.append(f"{where}: {p.code} follows up {parent}; add it to PARENT_QUESTION")
    return problems
