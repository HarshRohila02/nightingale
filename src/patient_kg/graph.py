"""The patient knowledge graph: one patient's history as a graph, and the findings it yields
(task 1d; README "The two knowledge graphs").

:func:`build_patient_graph` turns one Synthea patient (:mod:`src.patient_kg.synthea`) into a small
directed graph, as of a given day:

* one **patient** node: age on that day, sex, alive or not;
* one node per **condition**, **medication** and **procedure** in their history, each with its
  code, description and dates, and whether it is active on the day; and the **latest observation
  of each code** on or before the day (Synthea records vitals at every visit, so the history of
  each would swamp the graph);
* one node per ``RF:*`` **concept** the history supports, through the table of
  :mod:`src.patient_kg.mapping`, with an ``INDICATES`` edge from each entry that supports it.

:func:`patient_findings` reads the concepts back as :class:`~src.contracts.Finding` objects with
``source="synthea"``: what a hand-entered case's risk factors would say. A concept counts when a
condition mapped to it is active on the day, or, for smoking, when the latest smoking-status
observation says so. When the history says both (an old "smokes daily" still active, a later
"ex-smoker"), present wins: a risk factor is not discarded on the strength of a later "no".

**Not wired into the ranking.** Whether a patient's history feeds a case is a design decision for
the owner (docs/17 T7). This module stops at the findings; nothing in the pipeline calls it.

**Medications are history, never advice.** They are nodes with ``TOOK_MEDICATION`` edges, never
findings, and nothing here suggests one.

Reference: docs/16 §1, docs/17 T7; README "The two knowledge graphs".
"""

from __future__ import annotations

from datetime import date
from typing import Any

import networkx as nx

from src.contracts import Assertion, Finding, Sex
from src.medical_kg.crosswalk import BY_CONCEPT
from src.patient_kg.mapping import (
    CONDITION_CODES,
    NOT_MAPPED,
    SMOKING_STATUS,
    SMOKING_STATUS_LOINC,
    CodeMapping,
)
from src.patient_kg.synthea import Entry, PatientRecord

__all__ = [
    "PATIENT",
    "RELATIONS",
    "build_patient_graph",
    "patient_findings",
    "sex_of",
    "unmapped_codes",
]

PATIENT = "patient"
"""The patient node's id in every patient graph."""

RELATIONS = {
    "condition": "HAS_CONDITION",
    "observation": "HAS_OBSERVATION",
    "medication": "TOOK_MEDICATION",
    "procedure": "HAD_PROCEDURE",
}
"""The patient's edge to each kind of entry."""


def sex_of(record: PatientRecord) -> Sex | None:
    """Synthea's GENDER (M or F) as the contracts' ``Sex``, else None."""
    return {"M": Sex.MALE, "F": Sex.FEMALE}.get(record.gender)


def _latest_observations(entries: list[Entry], day: date) -> list[Entry]:
    latest: dict[str, Entry] = {}
    for entry in entries:
        if entry.kind != "observation" or entry.start is None or entry.start > day:
            continue
        held = latest.get(entry.code)
        if held is None or entry.start >= held.start:  # type: ignore[operator]
            latest[entry.code] = entry
    return list(latest.values())


def _mapping(entry: Entry, day: date) -> CodeMapping | None:
    """The concept ``entry`` supports on ``day``, if any."""
    if entry.kind == "condition" and entry.active_on(day):
        return CONDITION_CODES.get(entry.code)
    if entry.kind == "observation" and entry.code == SMOKING_STATUS_LOINC:
        return SMOKING_STATUS.get(entry.value or "")
    return None


def build_patient_graph(record: PatientRecord, as_of: date) -> nx.DiGraph:
    """The patient's graph as of ``as_of`` (see the module docstring)."""
    graph = nx.DiGraph(patient_id=record.patient_id, as_of=as_of.isoformat())
    sex = sex_of(record)
    graph.add_node(
        PATIENT,
        kind="patient",
        age=record.age_on(as_of),
        sex=sex.value if sex else None,
        alive=record.alive_on(as_of),
    )
    kept = [e for e in record.entries if e.kind != "observation" and (e.start or as_of) <= as_of]
    kept += _latest_observations(record.entries, as_of)
    for i, entry in enumerate(kept):
        node = f"{entry.kind}:{i}"
        graph.add_node(
            node,
            kind=entry.kind,
            code=entry.code,
            description=entry.description,
            start=entry.start.isoformat() if entry.start else None,
            stop=entry.stop.isoformat() if entry.stop else None,
            value=entry.value,
            units=entry.units,
            active=entry.active_on(as_of),
        )
        graph.add_edge(PATIENT, node, relation=RELATIONS[entry.kind])
        mapping = _mapping(entry, as_of)
        if mapping is None:
            continue
        concept = mapping.concept_id
        if concept not in graph:
            graph.add_node(concept, kind="concept", label=BY_CONCEPT[concept].label)
        graph.add_edge(
            node,
            concept,
            relation="INDICATES",
            assertion=mapping.assertion.value,
            snomed=mapping.code,
        )
    return graph


def patient_findings(graph: nx.DiGraph) -> list[Finding]:
    """The ``RF:*`` findings the patient's graph supports, in concept order.

    Each is present if any entry indicates it present, else absent; its qualifiers name the
    SNOMED codes behind it and, when known, the earliest date among them.
    """
    findings = []
    concepts = sorted(n for n, kind in graph.nodes(data="kind") if kind == "concept")
    for concept in concepts:
        sources = [(u, data) for u, _, data in graph.in_edges(concept, data=True)]
        present = any(data["assertion"] == Assertion.PRESENT.value for _, data in sources)
        dates = sorted(d for u, _ in sources if (d := graph.nodes[u].get("start")))
        qualifiers: dict[str, str] = {
            "snomed": ",".join(sorted({data["snomed"] for _, data in sources}))
        }
        if dates:
            qualifiers["since"] = dates[0]
        findings.append(
            Finding(
                concept_id=concept,
                label=graph.nodes[concept]["label"],
                assertion=Assertion.PRESENT if present else Assertion.ABSENT,
                qualifiers=qualifiers,
                source="synthea",
            )
        )
    return findings


def unmapped_codes(graph: nx.DiGraph) -> list[dict[str, Any]]:
    """Each active condition the table does not map, with why, for the builder's report."""
    out = []
    for _, data in graph.nodes(data=True):
        if data.get("kind") != "condition" or not data.get("active"):
            continue
        if data["code"] in CONDITION_CODES:
            continue
        out.append(
            {
                "code": data["code"],
                "description": data["description"],
                "reason": NOT_MAPPED.get(data["code"], "not in the SNOMED -> RF table"),
            }
        )
    return out
