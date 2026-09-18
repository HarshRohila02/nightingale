"""Hand-authored knowledge: aortic dissection (task 1b).

DDXPlus has no aortic dissection, so neither the DDXPlus-derived graph nor the ML ranker can know
it (docs/10). These edges come from the clinical literature instead. Their ``source`` is
``hand_authored``, and they are the knowledge graph's first knowledge that does not come from
DDXPlus: the independent contribution that the R-12 disclosure counts.

**What goes in.** The markers of the ADD-RS, the aortic dissection detection risk score of the 2010
ACCF/AHA thoracic aortic disease guideline, which Rogers et al. (2011) validated on IRAD patients:

- high-risk conditions: connective tissue disease, a family history of aortic disease, known aortic
  valve disease, recent aortic manipulation, a known thoracic aortic aneurysm;
- high-risk pain features: pain that is abrupt in onset, severe in intensity, or ripping/tearing;
- high-risk examination features: a perfusion deficit (a pulse deficit, a blood-pressure difference
  between the arms, a focal neurological deficit with the pain), a new aortic regurgitation murmur,
  hypotension or shock.

Also in: where the pain is (chest, back, radiating to the back), sharp pain and syncope, because
IRAD reports them, and hypertension and cocaine use as predisposing factors.

**Weights.** Each likelihood is P(finding | aortic dissection), in the bands BODHI-S uses
(:data:`src.medical_kg.loader.LIKELIHOOD_WEIGHT`), taken from the IRAD registry wherever it reports
a frequency. A marker with no published frequency is ``low``, and its ``evidence`` says so.

The red-flag rule for aortic dissection (src/reasoning/red_flags.py) is separate and unchanged.
It is the safety net; these edges let the knowledge graph rank the condition too.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from src.medical_kg.loader import (
    LIKELIHOOD_WEIGHT,
    EdgeSource,
    KGEdge,
    KnowledgeGraph,
    Likelihood,
    concept_node,
    condition_nodes,
    relation_for,
)

__all__ = ["AORTIC_DISSECTION", "FACTS", "REFERENCES", "HandAuthoredFact", "load_hand_authored_kg"]

AORTIC_DISSECTION = "COND:aortic_dissection"

REFERENCES: Mapping[str, str] = {
    "IRAD-2000": "Hagan PG, Nienaber CA, Isselbacher EM, et al. The International Registry of "
    "Acute Aortic Dissection (IRAD): new insights into an old disease. JAMA. "
    "2000;283(7):897-903.",
    "IRAD-2016": "Evangelista A, Maldonado G, Gruosso D, et al. Insights from the International "
    "Registry of Acute Aortic Dissection. Glob Cardiol Sci Pract. 2016;2016(1):e201608. "
    "doi:10.21542/gcsp.2016.8",
    "ADD-RS": "Hiratzka LF, Bakris GL, Beckman JA, et al. 2010 ACCF/AHA/AATS/ACR/ASA/SCA/SCAI/"
    "SIR/STS/SVM guidelines for the diagnosis and management of patients with thoracic aortic "
    "disease. Circulation. 2010;121(13):e266-e369. Rogers AM, Hermann LK, Booher AM, et al. "
    "Sensitivity of the aortic dissection detection risk score. Circulation. "
    "2011;123(20):2213-2218.",
}


@dataclass(frozen=True)
class HandAuthoredFact:
    """One literature-backed edge: a condition, a concept, and how often they go together.

    Attributes:
        condition_id: A registry condition.
        concept: A crosswalk concept (``SYM:*`` / ``RF:*``).
        likelihood: P(concept | condition), banded.
        evidence: What the source says, in a few words, e.g. "abrupt onset in 85%".
        reference: A key into :data:`REFERENCES`.
    """

    condition_id: str
    concept: str
    likelihood: Likelihood
    evidence: str
    reference: str


def _ad(concept: str, likelihood: Likelihood, evidence: str, reference: str) -> HandAuthoredFact:
    return HandAuthoredFact(AORTIC_DISSECTION, concept, likelihood, evidence, reference)


FACTS: tuple[HandAuthoredFact, ...] = (
    # ADD-RS high-risk pain features
    _ad("SYM:sudden_onset", Likelihood.VERY_HIGH, "abrupt onset in 85%", "IRAD-2000"),
    _ad(
        "SYM:severe_pain",
        Likelihood.HIGH,
        "an ADD-RS pain feature; IRAD calls sudden, severe, sharp pain the commonest complaint. "
        "No frequency was transcribed, so it is banded with sharp pain",
        "ADD-RS",
    ),
    _ad(
        "SYM:pain_character_tearing",
        Likelihood.HIGH,
        "tearing or ripping pain in about 50%",
        "IRAD-2000",
    ),
    _ad("SYM:pain_character_sharp", Likelihood.HIGH, "sharp pain in about 60%", "IRAD-2000"),
    # Where the pain is
    _ad("SYM:chest_pain", Likelihood.HIGH, "chest pain in 73%", "IRAD-2000"),
    _ad("SYM:back_pain", Likelihood.HIGH, "back pain in 53%", "IRAD-2000"),
    _ad(
        "SYM:radiation_back",
        Likelihood.MEDIUM,
        "no separate figure. Pain felt in the back (53%) is often described as radiating there, "
        "so it is banded one step below back pain",
        "IRAD-2000",
    ),
    # ADD-RS high-risk examination features
    _ad("SYM:pulse_deficit", Likelihood.LOW, "pulse deficit in 15%", "IRAD-2000"),
    _ad(
        "SYM:interarm_bp_difference",
        Likelihood.LOW,
        "an ADD-RS perfusion deficit, like the pulse deficit; no separate IRAD figure",
        "ADD-RS",
    ),
    _ad(
        "SYM:focal_neuro_deficit",
        Likelihood.LOW,
        "an ADD-RS perfusion deficit; uncommon in IRAD, exact figure not transcribed",
        "ADD-RS",
    ),
    _ad(
        "SYM:aortic_regurgitation_murmur",
        Likelihood.MEDIUM,
        "aortic regurgitation murmur in 32%",
        "IRAD-2000",
    ),
    _ad(
        "SYM:hypotension",
        Likelihood.MEDIUM,
        "hypotension in over 25% of type A dissections",
        "IRAD-2016",
    ),
    _ad("SYM:syncope", Likelihood.LOW, "syncope in 13%", "IRAD-2016"),
    # ADD-RS high-risk conditions, and predisposing factors
    _ad("RF:hypertension", Likelihood.HIGH, "a history of hypertension in 72%", "IRAD-2000"),
    _ad(
        "RF:thoracic_aortic_aneurysm", Likelihood.LOW, "a known aortic aneurysm in 16%", "IRAD-2016"
    ),
    _ad("RF:connective_tissue_disease", Likelihood.LOW, "Marfan syndrome in 5%", "IRAD-2016"),
    _ad(
        "RF:aortic_valve_disease",
        Likelihood.LOW,
        "an ADD-RS high-risk condition; no overall IRAD figure",
        "ADD-RS",
    ),
    _ad(
        "RF:family_history_aortic_disease",
        Likelihood.LOW,
        "an ADD-RS high-risk condition; no IRAD figure",
        "ADD-RS",
    ),
    _ad("RF:aortic_manipulation", Likelihood.RARE, "iatrogenic dissection in 4%", "IRAD-2016"),
    _ad("RF:cocaine_use", Likelihood.RARE, "cocaine use in 1.8%", "IRAD-2016"),
)
"""Checked by tests/test_kg_sources.py: every concept is in the crosswalk, every reference
exists, and no condition-concept pair appears twice."""


def load_hand_authored_kg(labels: Mapping[str, str] | None = None) -> KnowledgeGraph:
    """The hand-authored facts as a :class:`KnowledgeGraph`, ready to merge.

    Args:
        labels: ``DDX:E_nn`` → label, for concepts that sit on a DDXPlus question's node
            (hypertension sits on ``DDX:E_104``). Without it, those nodes keep their crosswalk
            label.
    """
    labels = labels or {}
    nodes = condition_nodes()
    edges: list[KGEdge] = []
    for fact in FACTS:
        node = concept_node(fact.concept, labels, EdgeSource.HAND_AUTHORED)
        nodes.setdefault(node.id, node)
        edges.append(
            KGEdge(
                condition_id=fact.condition_id,
                concept_id=node.id,
                relation=relation_for(node.type),
                source=EdgeSource.HAND_AUTHORED,
                weight=LIKELIHOOD_WEIGHT[fact.likelihood],
                properties={
                    "concept": fact.concept,
                    "likelihood": fact.likelihood.value,
                    "evidence": fact.evidence,
                    "reference": fact.reference,
                },
            )
        )
    return KnowledgeGraph(nodes=nodes, edges=edges)
