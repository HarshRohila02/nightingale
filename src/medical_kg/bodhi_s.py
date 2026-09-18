"""BODHI-S enrichment of the cardiac knowledge graph (task 1b).

BODHI-S (Eka Care) is a symptom-condition knowledge base covering 555 conditions. Four of ours
match it exactly (docs/10): MI, pericarditis, PE and GERD. For them it adds what DDXPlus lacks:
**answer-level findings**, such as squeezing chest pain or pain radiating to the jaw, and **how
likely each finding is** in the condition. These edges carry ``source = bodhi_s``: knowledge that
does not come from DDXPlus (R-12).

Two kinds of BODHI-S fact are used (docs/03 §1.2):

* ``PRESENT_IN``: a symptom is present in a condition, with P(symptom | condition) in bands. The
  symptom is a BODHI-S id.
* ``IS_INFLUENCED_BY``: a condition is influenced by another condition, with a strength. A positive
  influence is read as a risk factor; the influencing condition is a SNOMED CT id.

**The mapping** (:data:`SYMPTOMS`, :data:`RISK_FACTORS`) sends each fact to the crosswalk concepts
it implies. Each concept must be at least as broad as the fact: "chest pain radiating to the jaw"
implies chest pain, and radiation to the jaw or arm. P(broader concept | condition) is then at
least P(fact | condition), so the fact's likelihood is a floor for the concept's; where several
facts land on one concept, the highest is kept. Qualifiers the vocabulary has no concept for
(duration, relief by antacids) are dropped, and each drop is noted. A fact with nothing to map to has
no concepts and a reason.

**Licence.** CC-BY-NC-4.0: non-commercial use only, with attribution to Eka Care (docs/03 §1.2).
This module contains no BODHI-S text: facts are keyed by BODHI-S's own ids, and the notes are
paraphrases.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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

__all__ = [
    "BODHI_CONDITIONS",
    "RISK_FACTORS",
    "SYMPTOMS",
    "FactMapping",
    "bodhi_coverage",
    "load_bodhi_kg",
]

TRIPLES_FILE = "triples.jsonl"

BODHI_CONDITIONS: Mapping[str, str] = {
    "57054005": "COND:nstemi_stemi",  # acute myocardial infarction
    "3238004": "COND:pericarditis",
    "59282003": "COND:pulmonary_embolism",
    "235595009": "COND:gerd",  # gastro-oesophageal reflux disease
}
"""SNOMED CT id → registry id, for the four exact matches (docs/10)."""


@dataclass(frozen=True)
class FactMapping:
    """Where one BODHI-S fact lands.

    Attributes:
        concepts: The crosswalk concepts the fact implies, each at least as broad as the fact.
            Empty when nothing fits.
        note: A paraphrase of the fact, and what was dropped or why it cannot be mapped.
    """

    concepts: tuple[str, ...]
    note: str


def _m(*concepts: str, note: str) -> FactMapping:
    return FactMapping(concepts, note)


SYMPTOMS: Mapping[str, FactMapping] = {
    # --- Chest pain and its qualifiers ----------------------------------------------- #
    "4722e7c8-a65c-11eb-8d02-1e003a340630": _m("SYM:chest_pain", note="chest pain"),
    "1fd2c106-af36-11eb-8fc3-1e003a340631": _m(
        "SYM:chest_pain", note="chest discomfort, taken as chest pain"
    ),
    "472402de-a65c-11eb-8d02-1e003a340630": _m("SYM:chest_pain", note="diffuse chest pain"),
    "f42bb86a-7bf6-46ef-a582-c47514cea501": _m("SYM:chest_pain", note="left-sided chest pain"),
    "4723da48-a65c-11eb-8d02-1e003a340630": _m("SYM:chest_pain", note="retrosternal chest pain"),
    "4723ee52-a65c-11eb-8d02-1e003a340630": _m("SYM:chest_pain", note="substernal chest pain"),
    "47230fd2-a65c-11eb-8d02-1e003a340630": _m(
        "SYM:chest_pain", note="chest pain lasting 1 to 8 hours; duration not modelled"
    ),
    "4722fc04-a65c-11eb-8d02-1e003a340630": _m(
        "SYM:chest_pain", note="chest pain lasting under an hour; duration not modelled"
    ),
    "47234c86-a65c-11eb-8d02-1e003a340630": _m(
        "SYM:chest_pain", note="chest pain for 1 to 4 weeks; duration not modelled"
    ),
    "47236248-a65c-11eb-8d02-1e003a340630": _m(
        "SYM:chest_pain", note="chest pain for over a month; duration not modelled"
    ),
    "4724a6ee-a65c-11eb-8d02-1e003a340630": _m(
        "SYM:chest_pain", note="chest pain worse with coughing or moving; not modelled"
    ),
    "47245608-a65c-11eb-8d02-1e003a340630": _m(
        "SYM:chest_pain", note="chest pain worse on bending forward; not modelled"
    ),
    "4724d06a-a65c-11eb-8d02-1e003a340630": _m(
        "SYM:chest_pain", note="chest pain eased by antacids; not modelled"
    ),
    "472492c6-a65c-11eb-8d02-1e003a340630": _m(
        "SYM:chest_pain", "SYM:pleuritic", note="chest pain worse on deep breathing"
    ),
    "47241774-a65c-11eb-8d02-1e003a340630": _m(
        "SYM:chest_pain", "SYM:exertional", note="chest pain brought on by exertion"
    ),
    "4725612e-a65c-11eb-8d02-1e003a340630": _m(
        "SYM:chest_pain", "SYM:sudden_onset", note="chest pain of sudden onset"
    ),
    "47254cfc-a65c-11eb-8d02-1e003a340630": _m(
        "SYM:chest_pain", "SYM:severe_pain", note="severe chest pain"
    ),
    "4723b27a-a65c-11eb-8d02-1e003a340630": _m(
        "SYM:chest_pain", "SYM:pain_character_pressure", note="a feeling of pressure in the chest"
    ),
    "47239e2a-a65c-11eb-8d02-1e003a340630": _m(
        "SYM:chest_pain", "SYM:pain_character_pressure", note="squeezing chest pain"
    ),
    "47237666-a65c-11eb-8d02-1e003a340630": _m(
        "SYM:chest_pain", "SYM:pain_character_burning", note="burning chest pain"
    ),
    "47238a5c-a65c-11eb-8d02-1e003a340630": _m(
        "SYM:chest_pain", "SYM:pain_character_sharp", note="sharp or stabbing chest pain"
    ),
    "1fd1d8f4-af36-11eb-8fc3-1e003a340631": _m(
        "SYM:chest_pain", "SYM:radiation_jaw_arm", note="chest pain spreading to the jaw"
    ),
    "1fd1f60e-af36-11eb-8fc3-1e003a340631": _m(
        "SYM:chest_pain", "SYM:radiation_jaw_arm", note="chest pain spreading to arm or shoulder"
    ),
    "1fd21378-af36-11eb-8fc3-1e003a340631": _m(
        "SYM:chest_pain", "SYM:radiation_neck", note="chest pain spreading to the neck"
    ),
    "1fd23074-af36-11eb-8fc3-1e003a340631": _m(
        "SYM:chest_pain", "SYM:radiation_back", note="chest pain spreading between the scapulae"
    ),
    "4724baee-a65c-11eb-8d02-1e003a340630": _m(
        "SYM:chest_pain",
        "SYM:positional_relief_sitting_forward",
        note="chest pain eased by leaning forward",
    ),
    # --- Abdominal pain -------------------------------------------------------------- #
    "47172c6c-a65c-11eb-8d02-1e003a340630": _m("SYM:abdominal_pain", note="abdominal pain"),
    "471a08ec-a65c-11eb-8d02-1e003a340630": _m("SYM:abdominal_pain", note="upper abdominal pain"),
    "471a33ee-a65c-11eb-8d02-1e003a340630": _m(
        "SYM:abdominal_pain", "SYM:epigastric_pain", note="epigastric pain"
    ),
    "471ce0bc-a65c-11eb-8d02-1e003a340630": _m(
        "SYM:abdominal_pain", "SYM:post_prandial", note="abdominal pain worse after food"
    ),
    "47193368-a65c-11eb-8d02-1e003a340630": _m(
        "SYM:abdominal_pain", "SYM:pain_character_burning", note="burning abdominal pain"
    ),
    "4718dbde-a65c-11eb-8d02-1e003a340630": _m(
        "SYM:abdominal_pain", "SYM:pain_character_sharp", note="sharp abdominal pain"
    ),
    "471d0132-a65c-11eb-8d02-1e003a340630": _m(
        "SYM:abdominal_pain", note="abdominal pain worse at night; timing not modelled"
    ),
    "471d8346-a65c-11eb-8d02-1e003a340630": _m(
        "SYM:abdominal_pain", note="abdominal pain worse on bending forward; not modelled"
    ),
    "471e5794-a65c-11eb-8d02-1e003a340630": _m(
        "SYM:abdominal_pain", note="abdominal pain of gradual onset; not modelled"
    ),
    "1fd100aa-af36-11eb-8fc3-1e003a340631": _m(
        "SYM:abdominal_pain", note="recurrent abdominal pain; pattern not modelled"
    ),
    "471c2db6-a65c-11eb-8d02-1e003a340630": _m(
        "SYM:abdominal_pain",
        note="abdominal pain spreading to the left shoulder or scapula: either one, so too vague "
        "for the arm or the back",
    ),
    "471c08a4-a65c-11eb-8d02-1e003a340630": _m(
        "SYM:abdominal_pain",
        note="abdominal pain spreading to the right shoulder or scapula: as above",
    ),
    # --- Breathing ------------------------------------------------------------------ #
    "473ca9f6-a65c-11eb-8d02-1e003a340630": _m("SYM:breathlessness", note="breathlessness"),
    "473d4a00-a65c-11eb-8d02-1e003a340630": _m(
        "SYM:breathlessness", note="moderate breathlessness"
    ),
    "473d5f72-a65c-11eb-8d02-1e003a340630": _m("SYM:breathlessness", note="severe breathlessness"),
    "473e0cce-a65c-11eb-8d02-1e003a340630": _m(
        "SYM:breathlessness", "SYM:exertional", note="breathlessness on exertion"
    ),
    "473d76ce-a65c-11eb-8d02-1e003a340630": _m(
        "SYM:breathlessness", "SYM:sudden_onset", note="breathlessness of sudden onset"
    ),
    # --- Gastro-oesophageal ----------------------------------------------------------- #
    "476efd70-a65c-11eb-8d02-1e003a340630": _m("SYM:heartburn", note="heartburn"),
    "476fae46-a65c-11eb-8d02-1e003a340630": _m(
        "SYM:heartburn", note="heartburn eased by antacids; not modelled"
    ),
    "47a7e612-a65c-11eb-8d02-1e003a340630": _m("SYM:regurgitation", note="regurgitation"),
    "47a80d7c-a65c-11eb-8d02-1e003a340630": _m("SYM:regurgitation", note="regurgitating food"),
    "47a85c1e-a65c-11eb-8d02-1e003a340630": _m(
        "SYM:regurgitation", note="regurgitation worse on bending forward; not modelled"
    ),
    "47a8488c-a65c-11eb-8d02-1e003a340630": _m(
        "SYM:regurgitation", "SYM:worse_lying_flat", note="regurgitation worse lying flat"
    ),
    "47a88392-a65c-11eb-8d02-1e003a340630": _m(
        "SYM:regurgitation", note="regurgitation eased by antacids; not modelled"
    ),
    "47a86fe2-a65c-11eb-8d02-1e003a340630": _m(
        "SYM:regurgitation", note="regurgitation eased by eating; not modelled"
    ),
    "47a8972e-a65c-11eb-8d02-1e003a340630": _m(
        "SYM:regurgitation", note="regurgitation with a positional relief; not modelled"
    ),
    "1fd4da4a-af36-11eb-8fc3-1e003a340631": _m("SYM:dysphagia", note="difficulty swallowing"),
    "4737ee0c-a65c-11eb-8d02-1e003a340630": _m("SYM:nausea_or_vomiting", note="vomiting"),
    "47483014-a65c-11eb-8d02-1e003a340630": _m("SYM:nausea_or_vomiting", note="nausea"),
    "4738b724-a65c-11eb-8d02-1e003a340630": _m(
        "SYM:nausea_or_vomiting", note="vomiting with nausea"
    ),
    "4738cbd8-a65c-11eb-8d02-1e003a340630": _m(
        "SYM:nausea_or_vomiting", "SYM:post_prandial", note="vomiting after food"
    ),
    "47387a3e-a65c-11eb-8d02-1e003a340630": _m(
        "SYM:nausea_or_vomiting", "SYM:haematemesis", note="vomiting blood"
    ),
    "47888560-a65c-11eb-8d02-1e003a340630": _m(note="belching: no DDXPlus question or concept"),
    "478972cc-a65c-11eb-8d02-1e003a340630": _m(note="hiccups: no DDXPlus question or concept"),
    "475d0764-a65c-11eb-8d02-1e003a340630": _m(note="indigestion: too vague to map"),
    # --- Circulation, general --------------------------------------------------------- #
    "cbade44f-adca-4d70-a02d-a607e57bddf7": _m("SYM:diaphoresis", note="bouts of sweating"),
    "47431318-a65c-11eb-8d02-1e003a340630": _m("SYM:palpitations", note="palpitations"),
    "1fd3a54e-af36-11eb-8fc3-1e003a340631": _m("SYM:syncope", note="fainting"),
    "3203bb98-28f0-4c5c-9c5b-adc989bf2f57": _m("SYM:hypotension", note="low blood pressure"),
    "477c3b2a-a65c-11eb-8d02-1e003a340630": _m("SYM:paraesthesia", note="numbness"),
    "47258974-a65c-11eb-8d02-1e003a340630": _m("SYM:fever", note="fever"),
    "2161cdd4-3d3c-45be-aecf-b23adb577b6a": _m(
        "RF:hypertension", note="high blood pressure, listed as a finding"
    ),
    "7b8abebf-8a32-4615-b865-f7c2e23b8e51": _m(
        "RF:hypercholesterolaemia", note="raised cholesterol, listed as a finding"
    ),
    # --- Legs -------------------------------------------------------------------------- #
    "47534f9e-a65c-11eb-8d02-1e003a340630": _m("SYM:calf_pain", note="calf pain"),
    "475414c4-a65c-11eb-8d02-1e003a340630": _m(
        "SYM:calf_pain", "SYM:leg_swelling", note="a painful, swollen calf"
    ),
    "47760d5e-a65c-11eb-8d02-1e003a340630": _m("SYM:leg_swelling", note="a swollen foot"),
    "4776981e-a65c-11eb-8d02-1e003a340630": _m(
        "SYM:leg_swelling", note="pitting oedema of the foot"
    ),
}
"""BODHI-S symptom id → the concepts its PRESENT_IN facts imply. Every symptom BODHI-S links to one
of our four conditions is here; tests/test_kg_sources.py checks that against the release."""

RISK_FACTORS: Mapping[str, FactMapping] = {
    "38341003": _m("RF:hypertension", note="hypertension"),
    "13644009": _m("RF:hypercholesterolaemia", note="high cholesterol"),
    "77176002": _m("RF:smoking", note="current smoker"),
    "44054006": _m("RF:diabetes", note="type 2 diabetes"),
    "414916001": _m("RF:obesity", note="obesity"),
    "77386006": _m("RF:pregnancy", note="pregnancy"),
    "161508001": _m("RF:previous_dvt", note="previous deep vein thrombosis"),
    "42343007": _m("RF:heart_failure", note="congestive heart failure"),
    "8510008": _m("SYM:recent_immobilisation", note="prolonged bed rest"),
    "3215": _m("RF:recent_surgery", note="major surgery, read as recent: the PE risk factor"),
    "40979000": _m(note="lack of exercise: DDXPlus only asks the opposite (E_143)"),
    "230094004": _m(note="spicy food: no DDXPlus question or concept"),
    "161414005": _m(note="past pulmonary tuberculosis: no DDXPlus question or concept"),
    "275905002": _m(note="a past 'myocardial problem': too vague to map"),
    "55464009": _m(note="lupus: BODHI-S gives it zero strength"),
    "69896004": _m(note="rheumatoid arthritis: BODHI-S gives it zero strength"),
}
"""SNOMED CT id of an influencing condition → the concepts it implies, for the IS_INFLUENCED_BY
facts about our four conditions."""


def _triples(bodhi_dir: Path) -> Iterator[dict[str, Any]]:
    path = Path(bodhi_dir) / TRIPLES_FILE
    with path.open(encoding="utf-8") as lines:
        for line in lines:
            if line.strip():
                yield json.loads(line)


def _band(value: Any) -> Likelihood:
    try:
        return Likelihood(str(value).strip().lower())
    except ValueError:
        raise ValueError(f"unknown BODHI-S likelihood {value!r}") from None


def _our_facts(bodhi_dir: Path) -> Iterator[tuple[str, str, FactMapping, Likelihood, dict]]:
    """(condition id, BODHI-S id, mapping, likelihood, properties) for each usable fact.

    Raises:
        ValueError: for a fact about one of our conditions that the mapping does not list. The
            BODHI-S release has changed; review the new fact and add it.
    """
    for triple in _triples(bodhi_dir):
        relation, properties = triple.get("relation"), triple.get("properties") or {}
        if relation == "PRESENT_IN" and triple.get("tail") in BODHI_CONDITIONS:
            condition, fact_id, table = BODHI_CONDITIONS[triple["tail"]], triple["head"], SYMPTOMS
            likelihood = _band(properties.get("likelihood_symptom_given_condition"))
        elif relation == "IS_INFLUENCED_BY" and triple.get("head") in BODHI_CONDITIONS:
            if properties.get("relation_polarity") != "positive":
                continue  # a protective influence is not a risk factor
            condition, fact_id, table = (
                BODHI_CONDITIONS[triple["head"]],
                triple["tail"],
                RISK_FACTORS,
            )
            likelihood = _band(properties.get("relation_strength"))
        else:
            continue
        mapping = table.get(fact_id)
        if mapping is None:
            raise ValueError(
                f"BODHI-S fact {fact_id} ({relation}, {condition}) is not in the mapping; "
                "review it and add it to src/medical_kg/bodhi_s.py"
            )
        yield condition, fact_id, mapping, likelihood, properties


def load_bodhi_kg(bodhi_dir: Path, labels: Mapping[str, str] | None = None) -> KnowledgeGraph:
    """The BODHI-S facts about our four conditions as a :class:`KnowledgeGraph`, ready to merge.

    One edge per condition and concept node, weighted by the strongest fact that maps there.
    Facts with zero likelihood, or nothing to map to, add no edge.

    Args:
        bodhi_dir: The folder holding ``triples.jsonl`` (``data/raw/bodhi_s``).
        labels: ``DDX:E_nn`` → label, for concepts that sit on a DDXPlus question's node.

    Raises:
        FileNotFoundError: if ``triples.jsonl`` is missing.
        ValueError: see :func:`_our_facts`, and for an unknown likelihood band.
    """
    labels = labels or {}
    nodes = condition_nodes()
    strongest: dict[tuple[str, str], Likelihood] = {}
    specificity: dict[tuple[str, str], str | None] = {}
    support: dict[tuple[str, str], dict[str, set[str]]] = defaultdict(
        lambda: {"bodhi_ids": set(), "concepts": set()}
    )
    for condition, fact_id, mapping, likelihood, properties in _our_facts(bodhi_dir):
        if likelihood is Likelihood.ZERO:
            continue
        for concept in mapping.concepts:
            node = concept_node(concept, labels, EdgeSource.BODHI_S)
            nodes.setdefault(node.id, node)
            key = (condition, node.id)
            current = strongest.get(key)
            if current is None or LIKELIHOOD_WEIGHT[likelihood] > LIKELIHOOD_WEIGHT[current]:
                strongest[key] = likelihood
                p_condition = properties.get("likelihood_condition_given_symptom")
                specificity[key] = str(p_condition).lower() if p_condition else None
            support[key]["bodhi_ids"].add(fact_id)
            support[key]["concepts"].add(concept)

    edges = [
        KGEdge(
            condition_id=condition,
            concept_id=node_id,
            relation=relation_for(nodes[node_id].type),
            source=EdgeSource.BODHI_S,
            weight=LIKELIHOOD_WEIGHT[likelihood],
            properties={
                "likelihood": likelihood.value,
                # P(condition | finding) of the strongest fact: how specific it is, for 2a
                "p_condition_given_finding": specificity[(condition, node_id)],
                "concepts": sorted(support[(condition, node_id)]["concepts"]),
                "bodhi_ids": sorted(support[(condition, node_id)]["bodhi_ids"]),
            },
        )
        for (condition, node_id), likelihood in sorted(strongest.items())
    ]
    return KnowledgeGraph(nodes=nodes, edges=edges)


def bodhi_coverage(bodhi_dir: Path) -> dict[str, dict[str, int]]:
    """Per condition: BODHI-S facts read, mapped, with nothing to map to, and zero-likelihood.

    For the KG card. It reports counts only, so nothing of BODHI-S's text is printed or stored.
    """
    coverage: dict[str, Counter] = defaultdict(Counter)
    for condition, _, mapping, likelihood, _ in _our_facts(bodhi_dir):
        row = coverage[condition]
        row["facts"] += 1
        if likelihood is Likelihood.ZERO:
            row["zero_likelihood"] += 1
        elif mapping.concepts:
            row["mapped"] += 1
        else:
            row["unmappable"] += 1
    return {condition: dict(row) for condition, row in sorted(coverage.items())}
